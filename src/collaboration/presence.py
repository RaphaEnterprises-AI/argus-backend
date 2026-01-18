"""User presence tracking for real-time collaboration."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from .models import (
    BroadcastMessage,
    CollaborationEvent,
    CollaborationEventType,
    CursorPosition,
    PresenceStatus,
    SelectionRange,
    UserPresence,
)


class PresenceManager:
    """Manages user presence across workspaces and tests.

    Handles:
    - Tracking which users are online in a workspace
    - User cursor positions within test editors
    - Idle detection and automatic status updates
    - Presence broadcasting via callback
    """

    # Time after which a user is considered idle (no activity)
    IDLE_TIMEOUT_SECONDS = 120  # 2 minutes
    # Time after which an idle user is considered offline
    OFFLINE_TIMEOUT_SECONDS = 300  # 5 minutes
    # How often to check for idle/offline users
    CLEANUP_INTERVAL_SECONDS = 30

    def __init__(
        self,
        broadcast_fn: Callable[[BroadcastMessage], None] | None = None,
    ):
        """Initialize presence manager.

        Args:
            broadcast_fn: Optional callback for broadcasting presence updates.
        """
        # Map of workspace_id -> {user_id -> UserPresence}
        self._presence_by_workspace: dict[str, dict[str, UserPresence]] = {}
        # Map of test_id -> {user_id -> UserPresence} for fine-grained tracking
        self._presence_by_test: dict[str, dict[str, UserPresence]] = {}
        # Broadcast callback
        self._broadcast_fn = broadcast_fn
        # Cleanup task
        self._cleanup_task: asyncio.Task | None = None
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the presence manager background tasks."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the presence manager background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def user_join(
        self,
        user_id: str,
        user_name: str,
        user_email: str,
        workspace_id: str,
        test_id: str | None = None,
        avatar_url: str | None = None,
    ) -> UserPresence:
        """Record a user joining a workspace/test.

        Args:
            user_id: Unique user identifier.
            user_name: Display name.
            user_email: User email.
            workspace_id: Workspace being joined.
            test_id: Optional specific test being viewed.
            avatar_url: Optional avatar URL.

        Returns:
            The created UserPresence object.
        """
        async with self._lock:
            presence = UserPresence(
                id=str(uuid4()),
                user_id=user_id,
                user_name=user_name,
                user_email=user_email,
                avatar_url=avatar_url,
                workspace_id=workspace_id,
                test_id=test_id,
                status=PresenceStatus.ONLINE,
            )

            # Add to workspace presence
            if workspace_id not in self._presence_by_workspace:
                self._presence_by_workspace[workspace_id] = {}
            self._presence_by_workspace[workspace_id][user_id] = presence

            # Add to test presence if specified
            if test_id:
                if test_id not in self._presence_by_test:
                    self._presence_by_test[test_id] = {}
                self._presence_by_test[test_id][user_id] = presence

            # Broadcast join event
            await self._broadcast_event(
                CollaborationEvent(
                    type=CollaborationEventType.USER_JOINED,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    test_id=test_id,
                    payload=presence.to_dict(),
                )
            )

            return presence

    async def user_leave(
        self,
        user_id: str,
        workspace_id: str,
        test_id: str | None = None,
    ) -> None:
        """Record a user leaving a workspace/test.

        Args:
            user_id: User identifier.
            workspace_id: Workspace being left.
            test_id: Optional specific test being left.
        """
        async with self._lock:
            # Remove from workspace
            if workspace_id in self._presence_by_workspace:
                self._presence_by_workspace[workspace_id].pop(user_id, None)
                if not self._presence_by_workspace[workspace_id]:
                    del self._presence_by_workspace[workspace_id]

            # Remove from test
            if test_id and test_id in self._presence_by_test:
                self._presence_by_test[test_id].pop(user_id, None)
                if not self._presence_by_test[test_id]:
                    del self._presence_by_test[test_id]

            # Broadcast leave event
            await self._broadcast_event(
                CollaborationEvent(
                    type=CollaborationEventType.USER_LEFT,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    test_id=test_id,
                )
            )

    async def update_cursor(
        self,
        user_id: str,
        workspace_id: str,
        cursor: CursorPosition,
        test_id: str | None = None,
    ) -> None:
        """Update a user's cursor position.

        Args:
            user_id: User identifier.
            workspace_id: Workspace ID.
            cursor: New cursor position.
            test_id: Optional test ID.
        """
        async with self._lock:
            presence = self._get_presence(user_id, workspace_id)
            if presence:
                presence.cursor = cursor
                presence.last_active = datetime.now(UTC)
                presence.status = PresenceStatus.ONLINE

                # Update test presence if applicable
                if test_id and test_id in self._presence_by_test:
                    if user_id in self._presence_by_test[test_id]:
                        self._presence_by_test[test_id][user_id].cursor = cursor
                        self._presence_by_test[test_id][user_id].last_active = presence.last_active

                # Broadcast cursor move
                await self._broadcast_event(
                    CollaborationEvent(
                        type=CollaborationEventType.CURSOR_MOVE,
                        user_id=user_id,
                        workspace_id=workspace_id,
                        test_id=test_id,
                        payload={
                            "cursor": {
                                "x": cursor.x,
                                "y": cursor.y,
                                "element_id": cursor.element_id,
                                "step_index": cursor.step_index,
                                "field_name": cursor.field_name,
                            },
                            "color": presence.color,
                        },
                    )
                )

    async def update_selection(
        self,
        user_id: str,
        workspace_id: str,
        selection: SelectionRange,
        test_id: str | None = None,
    ) -> None:
        """Update a user's text selection.

        Args:
            user_id: User identifier.
            workspace_id: Workspace ID.
            selection: New selection range.
            test_id: Optional test ID.
        """
        async with self._lock:
            presence = self._get_presence(user_id, workspace_id)
            if presence:
                presence.selection = selection
                presence.last_active = datetime.now(UTC)
                presence.status = PresenceStatus.ONLINE

                # Broadcast selection change
                await self._broadcast_event(
                    CollaborationEvent(
                        type=CollaborationEventType.CURSOR_SELECT,
                        user_id=user_id,
                        workspace_id=workspace_id,
                        test_id=test_id,
                        payload={
                            "selection": {
                                "start": selection.start,
                                "end": selection.end,
                                "element_id": selection.element_id,
                            },
                            "color": presence.color,
                        },
                    )
                )

    async def update_activity(
        self,
        user_id: str,
        workspace_id: str,
    ) -> None:
        """Update a user's last activity timestamp.

        Call this on any user action to prevent idle status.

        Args:
            user_id: User identifier.
            workspace_id: Workspace ID.
        """
        async with self._lock:
            presence = self._get_presence(user_id, workspace_id)
            if presence:
                presence.last_active = datetime.now(UTC)
                if presence.status == PresenceStatus.IDLE:
                    presence.status = PresenceStatus.ONLINE

    async def set_status(
        self,
        user_id: str,
        workspace_id: str,
        status: PresenceStatus,
    ) -> None:
        """Manually set a user's status.

        Args:
            user_id: User identifier.
            workspace_id: Workspace ID.
            status: New status.
        """
        async with self._lock:
            presence = self._get_presence(user_id, workspace_id)
            if presence:
                old_status = presence.status
                presence.status = status
                presence.last_active = datetime.now(UTC)

                if old_status != status:
                    event_type = (
                        CollaborationEventType.USER_IDLE
                        if status == PresenceStatus.IDLE
                        else CollaborationEventType.USER_JOINED
                    )
                    await self._broadcast_event(
                        CollaborationEvent(
                            type=event_type,
                            user_id=user_id,
                            workspace_id=workspace_id,
                            payload={"status": status.value},
                        )
                    )

    def get_workspace_presence(self, workspace_id: str) -> list[UserPresence]:
        """Get all users present in a workspace.

        Args:
            workspace_id: Workspace ID.

        Returns:
            List of UserPresence objects.
        """
        if workspace_id not in self._presence_by_workspace:
            return []
        return list(self._presence_by_workspace[workspace_id].values())

    def get_test_presence(self, test_id: str) -> list[UserPresence]:
        """Get all users viewing a specific test.

        Args:
            test_id: Test ID.

        Returns:
            List of UserPresence objects.
        """
        if test_id not in self._presence_by_test:
            return []
        return list(self._presence_by_test[test_id].values())

    def get_user_presence(
        self,
        user_id: str,
        workspace_id: str,
    ) -> UserPresence | None:
        """Get a specific user's presence.

        Args:
            user_id: User identifier.
            workspace_id: Workspace ID.

        Returns:
            UserPresence if found, None otherwise.
        """
        return self._get_presence(user_id, workspace_id)

    def _get_presence(
        self,
        user_id: str,
        workspace_id: str,
    ) -> UserPresence | None:
        """Internal method to get presence without lock."""
        if workspace_id not in self._presence_by_workspace:
            return None
        return self._presence_by_workspace[workspace_id].get(user_id)

    async def _cleanup_loop(self) -> None:
        """Background task to clean up idle/offline users."""
        while True:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL_SECONDS)
                await self._check_idle_users()
            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue loop
                pass

    async def _check_idle_users(self) -> None:
        """Check for idle/offline users and update their status."""
        now = datetime.now(UTC)
        idle_threshold = now - timedelta(seconds=self.IDLE_TIMEOUT_SECONDS)
        offline_threshold = now - timedelta(seconds=self.OFFLINE_TIMEOUT_SECONDS)

        async with self._lock:
            users_to_remove: list[tuple[str, str]] = []

            for workspace_id, users in self._presence_by_workspace.items():
                for user_id, presence in users.items():
                    if presence.last_active < offline_threshold:
                        # User is offline - mark for removal
                        users_to_remove.append((user_id, workspace_id))
                    elif (
                        presence.last_active < idle_threshold
                        and presence.status == PresenceStatus.ONLINE
                    ):
                        # User is idle
                        presence.status = PresenceStatus.IDLE
                        await self._broadcast_event(
                            CollaborationEvent(
                                type=CollaborationEventType.USER_IDLE,
                                user_id=user_id,
                                workspace_id=workspace_id,
                                payload={"status": "idle"},
                            )
                        )

            # Remove offline users (outside the iteration)
            for user_id, workspace_id in users_to_remove:
                await self.user_leave(user_id, workspace_id)

    async def _broadcast_event(self, event: CollaborationEvent) -> None:
        """Broadcast an event via the configured callback."""
        if self._broadcast_fn:
            message = BroadcastMessage(
                channel=f"workspace:{event.workspace_id}",
                event=event.type.value,
                payload=event.to_dict(),
            )
            try:
                self._broadcast_fn(message)
            except Exception:
                # Log but don't fail
                pass
