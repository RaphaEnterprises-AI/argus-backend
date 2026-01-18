"""Real-time collaboration manager using Supabase Realtime.

Coordinates presence, cursors, CRDT edits, and comments across users.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from .crdt import CRDTOperation, TestSpecCRDT, VectorClock
from .cursors import CursorTracker
from .models import (
    BroadcastMessage,
    CollaborationEventType,
    CollaborativeComment,
    CursorPosition,
    PresenceStatus,
    SelectionRange,
    UserPresence,
)
from .presence import PresenceManager


@dataclass
class RealtimeConfig:
    """Configuration for real-time collaboration."""

    # Supabase configuration
    supabase_url: str = ""
    supabase_key: str = ""

    # Channel configuration
    workspace_channel_prefix: str = "workspace"
    test_channel_prefix: str = "test"

    # Timing
    presence_sync_interval_ms: int = 5000
    cursor_throttle_ms: int = 50
    operation_buffer_ms: int = 100

    # Limits
    max_concurrent_editors: int = 10
    max_pending_operations: int = 100


@dataclass
class RealtimeSession:
    """An active real-time collaboration session."""

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    workspace_id: str = ""
    test_id: str | None = None
    connected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_sync: datetime = field(default_factory=lambda: datetime.now(UTC))


class RealtimeManager:
    """Manages real-time collaboration for workspaces and tests.

    Integrates:
    - Presence tracking (who's online, their status)
    - Cursor positions (where users are editing)
    - CRDT operations (conflict-free edits)
    - Comments and @mentions
    """

    def __init__(self, config: RealtimeConfig | None = None):
        """Initialize real-time manager.

        Args:
            config: Optional configuration.
        """
        self.config = config or RealtimeConfig()
        self._presence = PresenceManager(broadcast_fn=self._broadcast)
        self._cursor_trackers: dict[str, CursorTracker] = {}  # test_id -> tracker
        self._crdt_docs: dict[str, TestSpecCRDT] = {}  # test_id -> CRDT
        self._sessions: dict[str, RealtimeSession] = {}  # session_id -> session
        self._comments: dict[str, list[CollaborativeComment]] = {}  # test_id -> comments
        self._broadcast_handlers: list[Callable[[BroadcastMessage], None]] = []
        self._operation_buffer: list[CRDTOperation] = []
        self._buffer_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the real-time manager."""
        await self._presence.start()
        self._buffer_task = asyncio.create_task(self._flush_operations_loop())

    async def stop(self) -> None:
        """Stop the real-time manager."""
        await self._presence.stop()
        if self._buffer_task:
            self._buffer_task.cancel()
            try:
                await self._buffer_task
            except asyncio.CancelledError:
                pass

    def add_broadcast_handler(
        self,
        handler: Callable[[BroadcastMessage], None],
    ) -> None:
        """Add a handler for broadcast messages.

        Args:
            handler: Function to call with broadcast messages.
        """
        self._broadcast_handlers.append(handler)

    def remove_broadcast_handler(
        self,
        handler: Callable[[BroadcastMessage], None],
    ) -> None:
        """Remove a broadcast handler.

        Args:
            handler: Handler to remove.
        """
        if handler in self._broadcast_handlers:
            self._broadcast_handlers.remove(handler)

    # =========================================================================
    # Session Management
    # =========================================================================

    async def connect(
        self,
        user_id: str,
        user_name: str,
        user_email: str,
        workspace_id: str,
        test_id: str | None = None,
        avatar_url: str | None = None,
    ) -> RealtimeSession:
        """Connect a user to real-time collaboration.

        Args:
            user_id: User identifier.
            user_name: Display name.
            user_email: User email.
            workspace_id: Workspace to join.
            test_id: Optional specific test to edit.
            avatar_url: Optional avatar URL.

        Returns:
            The created session.
        """
        async with self._lock:
            # Create session
            session = RealtimeSession(
                user_id=user_id,
                workspace_id=workspace_id,
                test_id=test_id,
            )
            self._sessions[session.id] = session

            # Join presence
            presence = await self._presence.user_join(
                user_id=user_id,
                user_name=user_name,
                user_email=user_email,
                workspace_id=workspace_id,
                test_id=test_id,
                avatar_url=avatar_url,
            )

            # Set up cursor tracker if editing a test
            if test_id:
                if test_id not in self._cursor_trackers:
                    self._cursor_trackers[test_id] = CursorTracker(
                        test_id=test_id,
                        broadcast_fn=self._broadcast,
                    )
                self._cursor_trackers[test_id].add_user(
                    user_id=user_id,
                    user_name=user_name,
                    color=presence.color,
                )

            return session

    async def disconnect(self, session_id: str) -> None:
        """Disconnect a user from real-time collaboration.

        Args:
            session_id: Session identifier.
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if not session:
                return

            # Leave presence
            await self._presence.user_leave(
                user_id=session.user_id,
                workspace_id=session.workspace_id,
                test_id=session.test_id,
            )

            # Remove from cursor tracker
            if session.test_id and session.test_id in self._cursor_trackers:
                self._cursor_trackers[session.test_id].remove_user(session.user_id)

    async def switch_test(
        self,
        session_id: str,
        new_test_id: str | None,
    ) -> None:
        """Switch which test a user is editing.

        Args:
            session_id: Session identifier.
            new_test_id: New test ID (None to leave test editing).
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return

            old_test_id = session.test_id

            # Leave old test cursor tracker
            if old_test_id and old_test_id in self._cursor_trackers:
                self._cursor_trackers[old_test_id].remove_user(session.user_id)

            # Update presence tracking for test change
            # Remove from old test's presence
            if old_test_id and old_test_id in self._presence._presence_by_test:
                self._presence._presence_by_test[old_test_id].pop(session.user_id, None)
                if not self._presence._presence_by_test[old_test_id]:
                    del self._presence._presence_by_test[old_test_id]

            # Get user presence
            presence = self._presence.get_user_presence(
                session.user_id,
                session.workspace_id,
            )

            # Join new test
            if new_test_id and presence:
                # Add to new test's presence
                if new_test_id not in self._presence._presence_by_test:
                    self._presence._presence_by_test[new_test_id] = {}
                self._presence._presence_by_test[new_test_id][session.user_id] = presence

                # Set up cursor tracker
                if new_test_id not in self._cursor_trackers:
                    self._cursor_trackers[new_test_id] = CursorTracker(
                        test_id=new_test_id,
                        broadcast_fn=self._broadcast,
                    )
                self._cursor_trackers[new_test_id].add_user(
                    user_id=session.user_id,
                    user_name=presence.user_name,
                    color=presence.color,
                )

            # Update presence test_id
            if presence:
                presence.test_id = new_test_id

            session.test_id = new_test_id

    # =========================================================================
    # Presence Operations
    # =========================================================================

    async def update_cursor(
        self,
        session_id: str,
        cursor: CursorPosition,
    ) -> None:
        """Update cursor position for a session.

        Args:
            session_id: Session identifier.
            cursor: New cursor position.
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        await self._presence.update_cursor(
            user_id=session.user_id,
            workspace_id=session.workspace_id,
            cursor=cursor,
            test_id=session.test_id,
        )

        if session.test_id and session.test_id in self._cursor_trackers:
            self._cursor_trackers[session.test_id].update_position(
                user_id=session.user_id,
                position=cursor,
            )

    async def update_selection(
        self,
        session_id: str,
        selection: SelectionRange,
    ) -> None:
        """Update text selection for a session.

        Args:
            session_id: Session identifier.
            selection: New selection range.
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        await self._presence.update_selection(
            user_id=session.user_id,
            workspace_id=session.workspace_id,
            selection=selection,
            test_id=session.test_id,
        )

        if session.test_id and session.test_id in self._cursor_trackers:
            self._cursor_trackers[session.test_id].update_selection(
                user_id=session.user_id,
                selection=selection,
            )

    async def set_status(
        self,
        session_id: str,
        status: PresenceStatus,
    ) -> None:
        """Set user status.

        Args:
            session_id: Session identifier.
            status: New status.
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        await self._presence.set_status(
            user_id=session.user_id,
            workspace_id=session.workspace_id,
            status=status,
        )

    def get_workspace_users(self, workspace_id: str) -> list[UserPresence]:
        """Get all users in a workspace.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            List of presence objects.
        """
        return self._presence.get_workspace_presence(workspace_id)

    def get_test_users(self, test_id: str) -> list[UserPresence]:
        """Get all users editing a test.

        Args:
            test_id: Test identifier.

        Returns:
            List of presence objects.
        """
        return self._presence.get_test_presence(test_id)

    # =========================================================================
    # CRDT Operations
    # =========================================================================

    async def load_test_spec(
        self,
        test_id: str,
        test_spec: dict,
        node_id: str | None = None,
    ) -> TestSpecCRDT:
        """Load a test specification for collaborative editing.

        Args:
            test_id: Test identifier.
            test_spec: Initial test specification.
            node_id: Optional node ID for CRDT (default: server).

        Returns:
            The CRDT document.
        """
        async with self._lock:
            if test_id not in self._crdt_docs:
                self._crdt_docs[test_id] = TestSpecCRDT(
                    node_id=node_id or "server",
                    test_spec=test_spec,
                )
            return self._crdt_docs[test_id]

    async def apply_edit(
        self,
        session_id: str,
        test_id: str,
        operation: CRDTOperation,
    ) -> bool:
        """Apply an edit operation from a user.

        Args:
            session_id: Session identifier.
            test_id: Test being edited.
            operation: The CRDT operation.

        Returns:
            True if operation was applied.
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        async with self._lock:
            crdt = self._crdt_docs.get(test_id)
            if not crdt:
                return False

            applied = crdt.apply(operation)
            if applied:
                # Buffer operation for broadcast
                self._operation_buffer.append(operation)

                # Broadcast edit event
                self._broadcast(BroadcastMessage(
                    channel=f"test:{test_id}",
                    event=CollaborationEventType.EDIT_CHANGE.value,
                    payload={
                        "operation": operation.to_dict(),
                        "user_id": session.user_id,
                    },
                ))

            return applied

    async def get_test_spec(self, test_id: str) -> dict | None:
        """Get current test specification.

        Args:
            test_id: Test identifier.

        Returns:
            Test spec dict or None.
        """
        crdt = self._crdt_docs.get(test_id)
        return crdt.test_spec if crdt else None

    async def get_operations_since(
        self,
        test_id: str,
        vector_clock: dict[str, int],
    ) -> list[CRDTOperation]:
        """Get operations since a vector clock (for sync).

        Args:
            test_id: Test identifier.
            vector_clock: Client's vector clock.

        Returns:
            List of operations the client hasn't seen.
        """
        crdt = self._crdt_docs.get(test_id)
        if not crdt:
            return []
        return crdt.get_operations_since(VectorClock.from_dict(vector_clock))

    # =========================================================================
    # Comments
    # =========================================================================

    async def add_comment(
        self,
        session_id: str,
        test_id: str,
        content: str,
        step_index: int | None = None,
        parent_id: str | None = None,
        mentions: list[str] | None = None,
    ) -> CollaborativeComment:
        """Add a comment to a test.

        Args:
            session_id: Session identifier.
            test_id: Test being commented on.
            content: Comment content.
            step_index: Optional step being commented on.
            parent_id: Optional parent comment ID (for replies).
            mentions: Optional list of mentioned user IDs.

        Returns:
            The created comment.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError("Invalid session")

        presence = self._presence.get_user_presence(
            session.user_id,
            session.workspace_id,
        )

        comment = CollaborativeComment(
            test_id=test_id,
            step_index=step_index,
            author_id=session.user_id,
            author_name=presence.user_name if presence else "Unknown",
            author_avatar=presence.avatar_url if presence else None,
            content=content,
            mentions=mentions or [],
            parent_id=parent_id,
        )

        async with self._lock:
            if test_id not in self._comments:
                self._comments[test_id] = []
            self._comments[test_id].append(comment)

        # Broadcast comment event
        self._broadcast(BroadcastMessage(
            channel=f"test:{test_id}",
            event=CollaborationEventType.COMMENT_ADDED.value,
            payload=comment.to_dict(),
        ))

        return comment

    async def resolve_comment(
        self,
        session_id: str,
        comment_id: str,
    ) -> bool:
        """Resolve a comment.

        Args:
            session_id: Session identifier.
            comment_id: Comment to resolve.

        Returns:
            True if comment was resolved.
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        async with self._lock:
            for test_id, comments in self._comments.items():
                for comment in comments:
                    if comment.id == comment_id:
                        comment.resolved = True
                        comment.resolved_by = session.user_id
                        comment.resolved_at = datetime.now(UTC)

                        # Broadcast resolve event
                        self._broadcast(BroadcastMessage(
                            channel=f"test:{test_id}",
                            event=CollaborationEventType.COMMENT_RESOLVED.value,
                            payload={
                                "comment_id": comment_id,
                                "resolved_by": session.user_id,
                            },
                        ))
                        return True
        return False

    def get_comments(
        self,
        test_id: str,
        step_index: int | None = None,
        include_resolved: bool = False,
    ) -> list[CollaborativeComment]:
        """Get comments for a test.

        Args:
            test_id: Test identifier.
            step_index: Optional step filter.
            include_resolved: Whether to include resolved comments.

        Returns:
            List of comments.
        """
        comments = self._comments.get(test_id, [])

        if step_index is not None:
            comments = [c for c in comments if c.step_index == step_index]

        if not include_resolved:
            comments = [c for c in comments if not c.resolved]

        return comments

    # =========================================================================
    # State Sync
    # =========================================================================

    async def get_full_state(
        self,
        session_id: str,
        test_id: str,
    ) -> dict:
        """Get full collaboration state for initial sync.

        Args:
            session_id: Session identifier.
            test_id: Test to get state for.

        Returns:
            Complete state dictionary.
        """
        session = self._sessions.get(session_id)
        if not session:
            return {}

        crdt = self._crdt_docs.get(test_id)
        tracker = self._cursor_trackers.get(test_id)

        return {
            "test_id": test_id,
            "test_spec": crdt.test_spec if crdt else None,
            "vector_clock": crdt._crdt.vector_clock.to_dict() if crdt else {},
            "presence": [p.to_dict() for p in self._presence.get_test_presence(test_id)],
            "cursors": tracker.get_full_state() if tracker else {"cursors": []},
            "comments": [c.to_dict() for c in self.get_comments(test_id)],
        }

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _broadcast(self, message: BroadcastMessage) -> None:
        """Send a broadcast message to all handlers.

        Args:
            message: Message to broadcast.
        """
        for handler in self._broadcast_handlers:
            try:
                handler(message)
            except Exception:
                pass

    async def _flush_operations_loop(self) -> None:
        """Background task to flush buffered operations."""
        while True:
            try:
                await asyncio.sleep(self.config.operation_buffer_ms / 1000)
                await self._flush_operations()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _flush_operations(self) -> None:
        """Flush buffered operations."""
        async with self._lock:
            if not self._operation_buffer:
                return

            operations = self._operation_buffer
            self._operation_buffer = []

            # Group operations by test_id
            by_test: dict[str, list[CRDTOperation]] = {}
            for op in operations:
                # Extract test_id from path if available
                for t_id in self._crdt_docs:
                    # Operations are tracked per test
                    if t_id not in by_test:
                        by_test[t_id] = []
                    by_test[t_id].append(op)
                    break


# Convenience function for creating manager
def create_realtime_manager(
    supabase_url: str = "",
    supabase_key: str = "",
) -> RealtimeManager:
    """Create a real-time manager with configuration.

    Args:
        supabase_url: Supabase project URL.
        supabase_key: Supabase anon key.

    Returns:
        Configured RealtimeManager.
    """
    config = RealtimeConfig(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
    )
    return RealtimeManager(config)
