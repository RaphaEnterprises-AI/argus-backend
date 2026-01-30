"""Sync manager for two-way IDE synchronization.

This module provides persistent synchronization state management for IDE integrations.
State is persisted to Supabase to survive server restarts while maintaining an
in-memory cache for performance.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog

from .change_detector import ChangeDetector, DiffResult
from .conflict_resolver import ConflictResolver
from .models import (
    ConflictResolutionStrategy,
    ProjectSyncState,
    SyncConflict,
    SyncEvent,
    SyncPullResult,
    SyncPushResult,
    SyncSource,
    SyncStatus,
    TestSyncState,
)

logger = structlog.get_logger(__name__)


@dataclass
class SyncConfig:
    """Configuration for sync manager."""

    # Auto-sync settings
    auto_sync_enabled: bool = True
    auto_sync_interval_ms: int = 5000  # 5 seconds
    auto_push_on_save: bool = True

    # Conflict handling
    default_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.MERGE

    # Limits
    max_events_per_push: int = 100
    max_retries: int = 3
    retry_delay_ms: int = 1000

    # Persistence settings
    persist_to_db: bool = True  # Enable database persistence
    db_sync_interval_ms: int = 2000  # Persist to DB every 2 seconds


class SyncPersistence:
    """Handles database persistence for sync state.

    Uses Supabase to persist:
    - Project sync state (versions, status, test states)
    - Local and base specs (for conflict detection)
    - Pending events (for recovery after restart)
    """

    def __init__(self):
        self._supabase = None
        self._initialized = False

    async def _get_supabase(self):
        """Lazily get the Supabase client."""
        if self._initialized:
            return self._supabase

        try:
            from src.integrations.supabase import get_supabase
            self._supabase = await get_supabase()
            self._initialized = True
        except Exception as e:
            logger.warning("Failed to initialize Supabase for sync persistence", error=str(e))
            self._initialized = True
            self._supabase = None

        return self._supabase

    async def load_project_state(
        self,
        project_id: str,
        user_id: str,
    ) -> tuple[ProjectSyncState | None, dict, dict, list[SyncEvent]]:
        """Load project state from database.

        Args:
            project_id: Project identifier.
            user_id: User identifier.

        Returns:
            Tuple of (ProjectSyncState, local_specs, base_specs, pending_events)
            Returns (None, {}, {}, []) if not found or error.
        """
        supabase = await self._get_supabase()
        if not supabase:
            return None, {}, {}, []

        try:
            # Load project state
            rows = await supabase.select(
                "sync_project_state",
                columns="*",
                filters={"project_id": f"eq.{project_id}"},
                limit=1,
            )

            project_state = None
            local_specs: dict = {}
            base_specs: dict = {}

            if rows:
                row = rows[0]
                state_data = row.get("state", {})
                local_specs = row.get("local_specs", {})
                base_specs = row.get("base_specs", {})

                # Reconstruct ProjectSyncState from JSON
                project_state = self._deserialize_project_state(project_id, state_data)
                logger.debug(
                    "Loaded project state from database",
                    project_id=project_id,
                    tests_count=len(project_state.tests) if project_state else 0,
                )

            # Load pending events
            event_rows = await supabase.select(
                "sync_pending_events",
                columns="*",
                filters={
                    "project_id": f"eq.{project_id}",
                    "synced": "eq.false",
                },
                order_by="local_version",
                ascending=True,
            )

            pending_events = [
                SyncEvent.from_dict(row["event_data"])
                for row in event_rows
            ]

            logger.debug(
                "Loaded pending events from database",
                project_id=project_id,
                events_count=len(pending_events),
            )

            return project_state, local_specs, base_specs, pending_events

        except Exception as e:
            logger.error("Failed to load project state from database", error=str(e))
            return None, {}, {}, []

    async def save_project_state(
        self,
        project_id: str,
        user_id: str,
        project_state: ProjectSyncState,
        local_specs: dict,
        base_specs: dict,
    ) -> bool:
        """Save project state to database.

        Args:
            project_id: Project identifier.
            user_id: User identifier.
            project_state: Project sync state.
            local_specs: Local test specifications.
            base_specs: Base test specifications.

        Returns:
            True if saved successfully.
        """
        supabase = await self._get_supabase()
        if not supabase:
            return False

        try:
            state_data = self._serialize_project_state(project_state)

            # Check if record exists
            existing = await supabase.select(
                "sync_project_state",
                columns="id",
                filters={"project_id": f"eq.{project_id}"},
                limit=1,
            )

            if existing:
                # Update existing record
                success = await supabase.update(
                    "sync_project_state",
                    values={
                        "state": state_data,
                        "local_specs": local_specs,
                        "base_specs": base_specs,
                        "last_sync": datetime.now(UTC).isoformat(),
                    },
                    filters={"project_id": project_id},
                )
            else:
                # Insert new record
                success = await supabase.insert(
                    "sync_project_state",
                    records=[{
                        "project_id": project_id,
                        "user_id": user_id,
                        "state": state_data,
                        "local_specs": local_specs,
                        "base_specs": base_specs,
                    }],
                )

            if success:
                logger.debug("Saved project state to database", project_id=project_id)
            return success

        except Exception as e:
            logger.error("Failed to save project state to database", error=str(e))
            return False

    async def save_pending_event(
        self,
        project_id: str,
        user_id: str,
        event: SyncEvent,
    ) -> bool:
        """Save a pending event to database.

        Args:
            project_id: Project identifier.
            user_id: User identifier.
            event: Sync event to save.

        Returns:
            True if saved successfully.
        """
        supabase = await self._get_supabase()
        if not supabase:
            return False

        try:
            success = await supabase.insert(
                "sync_pending_events",
                records=[{
                    "project_id": project_id,
                    "user_id": user_id,
                    "test_id": event.test_id,
                    "event_type": event.type.value,
                    "event_data": event.to_dict(),
                    "local_version": event.local_version,
                    "remote_version": event.remote_version,
                }],
            )
            return success
        except Exception as e:
            logger.error("Failed to save pending event", error=str(e))
            return False

    async def mark_events_synced(
        self,
        project_id: str,
        event_ids: list[str],
    ) -> bool:
        """Mark events as synced.

        Args:
            project_id: Project identifier.
            event_ids: List of event IDs to mark as synced.

        Returns:
            True if updated successfully.
        """
        supabase = await self._get_supabase()
        if not supabase:
            return False

        if not event_ids:
            return True

        try:
            # Update each event (Supabase doesn't support IN clause easily)
            for event_id in event_ids:
                # Find by event_data->>'id'
                rows = await supabase.select(
                    "sync_pending_events",
                    columns="id",
                    filters={
                        "project_id": f"eq.{project_id}",
                        "synced": "eq.false",
                    },
                )

                for row in rows:
                    # We need to check event_data.id matches
                    db_id = row.get("id")
                    if db_id:
                        await supabase.update(
                            "sync_pending_events",
                            values={
                                "synced": True,
                                "synced_at": datetime.now(UTC).isoformat(),
                            },
                            filters={"id": str(db_id)},
                        )

            return True
        except Exception as e:
            logger.error("Failed to mark events synced", error=str(e))
            return False

    async def delete_project_state(self, project_id: str) -> bool:
        """Delete project state from database.

        Args:
            project_id: Project identifier.

        Returns:
            True if deleted successfully.
        """
        supabase = await self._get_supabase()
        if not supabase:
            return False

        try:
            await supabase.delete(
                "sync_project_state",
                filters={"project_id": project_id},
            )
            await supabase.delete(
                "sync_pending_events",
                filters={"project_id": project_id},
            )
            return True
        except Exception as e:
            logger.error("Failed to delete project state", error=str(e))
            return False

    def _serialize_project_state(self, state: ProjectSyncState) -> dict:
        """Serialize ProjectSyncState to JSON-compatible dict."""
        return {
            "project_id": state.project_id,
            "last_full_sync": state.last_full_sync.isoformat() if state.last_full_sync else None,
            "sync_enabled": state.sync_enabled,
            "tests": {
                test_id: self._serialize_test_state(test_state)
                for test_id, test_state in state.tests.items()
            },
        }

    def _serialize_test_state(self, state: TestSyncState) -> dict:
        """Serialize TestSyncState to JSON-compatible dict."""
        return {
            "test_id": state.test_id,
            "local_version": state.local_version,
            "remote_version": state.remote_version,
            "status": state.status.value,
            "last_synced": state.last_synced.isoformat() if state.last_synced else None,
            "last_local_change": state.last_local_change.isoformat() if state.last_local_change else None,
            "last_remote_change": state.last_remote_change.isoformat() if state.last_remote_change else None,
            "checksum": state.checksum,
            "pending_changes": [e.to_dict() for e in state.pending_changes],
            "conflicts": [c.to_dict() for c in state.conflicts],
        }

    def _deserialize_project_state(self, project_id: str, data: dict) -> ProjectSyncState:
        """Deserialize ProjectSyncState from JSON dict."""
        state = ProjectSyncState(project_id=project_id)

        if data.get("last_full_sync"):
            state.last_full_sync = datetime.fromisoformat(data["last_full_sync"])

        state.sync_enabled = data.get("sync_enabled", True)

        tests_data = data.get("tests", {})
        for test_id, test_data in tests_data.items():
            state.tests[test_id] = self._deserialize_test_state(test_id, test_data)

        return state

    def _deserialize_test_state(self, test_id: str, data: dict) -> TestSyncState:
        """Deserialize TestSyncState from JSON dict."""
        state = TestSyncState(test_id=test_id)
        state.local_version = data.get("local_version", 0)
        state.remote_version = data.get("remote_version", 0)
        state.status = SyncStatus(data.get("status", "synced"))
        state.checksum = data.get("checksum")

        if data.get("last_synced"):
            state.last_synced = datetime.fromisoformat(data["last_synced"])
        if data.get("last_local_change"):
            state.last_local_change = datetime.fromisoformat(data["last_local_change"])
        if data.get("last_remote_change"):
            state.last_remote_change = datetime.fromisoformat(data["last_remote_change"])

        state.pending_changes = [
            SyncEvent.from_dict(e) for e in data.get("pending_changes", [])
        ]
        state.conflicts = [
            self._deserialize_conflict(c) for c in data.get("conflicts", [])
        ]

        return state

    def _deserialize_conflict(self, data: dict) -> SyncConflict:
        """Deserialize SyncConflict from JSON dict."""
        conflict = SyncConflict(
            id=data.get("id", ""),
            test_id=data.get("test_id", ""),
            path=data.get("path", []),
            local_value=data.get("local_value"),
            remote_value=data.get("remote_value"),
            local_version=data.get("local_version", 0),
            remote_version=data.get("remote_version", 0),
            resolved=data.get("resolved", False),
            resolved_value=data.get("resolved_value"),
        )

        if data.get("local_timestamp"):
            conflict.local_timestamp = datetime.fromisoformat(data["local_timestamp"])
        if data.get("remote_timestamp"):
            conflict.remote_timestamp = datetime.fromisoformat(data["remote_timestamp"])
        if data.get("resolution"):
            conflict.resolution = ConflictResolutionStrategy(data["resolution"])
        if data.get("created_at"):
            conflict.created_at = datetime.fromisoformat(data["created_at"])

        return conflict


class SyncManager:
    """Manages two-way synchronization between IDE and Argus cloud.

    Features:
    - Push local changes to cloud
    - Pull remote changes
    - Automatic conflict detection and resolution
    - Version tracking
    - Optimistic updates with rollback
    - **Persistent state storage** - survives server restarts
    """

    def __init__(
        self,
        config: SyncConfig | None = None,
        push_fn: Callable[[list[SyncEvent]], SyncPushResult] | None = None,
        pull_fn: Callable[[str, int], SyncPullResult] | None = None,
        user_id: str | None = None,
    ):
        """Initialize sync manager.

        Args:
            config: Sync configuration.
            push_fn: Function to push events to server.
            pull_fn: Function to pull events from server.
            user_id: User ID for persistence (required for multi-tenant).
        """
        self.config = config or SyncConfig()
        self._push_fn = push_fn
        self._pull_fn = pull_fn
        self._user_id = user_id or "anonymous"
        self._detector = ChangeDetector()
        self._resolver = ConflictResolver(
            default_strategy=self.config.default_resolution
        )

        # State tracking (in-memory cache)
        self._projects: dict[str, ProjectSyncState] = {}
        self._local_specs: dict[str, dict] = {}  # test_id -> spec
        self._base_specs: dict[str, dict] = {}  # test_id -> last synced spec
        self._pending_events: list[SyncEvent] = []

        # Persistence layer
        self._persistence = SyncPersistence() if self.config.persist_to_db else None
        self._dirty_projects: set[str] = set()  # Projects that need to be persisted
        self._loaded_projects: set[str] = set()  # Projects loaded from DB

        # Sync control
        self._sync_task: asyncio.Task | None = None
        self._persist_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the sync manager background tasks."""
        if self.config.auto_sync_enabled and self._sync_task is None:
            self._sync_task = asyncio.create_task(self._auto_sync_loop())

        # Start persistence background task
        if self._persistence and self._persist_task is None:
            self._persist_task = asyncio.create_task(self._persist_loop())

    async def stop(self) -> None:
        """Stop the sync manager and persist any pending state."""
        # Persist any dirty state before stopping
        if self._persistence and self._dirty_projects:
            await self._persist_dirty_projects()

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        if self._persist_task:
            self._persist_task.cancel()
            try:
                await self._persist_task
            except asyncio.CancelledError:
                pass
            self._persist_task = None

    def set_push_handler(
        self,
        handler: Callable[[list[SyncEvent]], SyncPushResult],
    ) -> None:
        """Set the push handler function.

        Args:
            handler: Function to handle push operations.
        """
        self._push_fn = handler

    def set_pull_handler(
        self,
        handler: Callable[[str, int], SyncPullResult],
    ) -> None:
        """Set the pull handler function.

        Args:
            handler: Function to handle pull operations.
        """
        self._pull_fn = handler

    # =========================================================================
    # Project Management
    # =========================================================================

    def get_project_state(self, project_id: str) -> ProjectSyncState:
        """Get or create project sync state (sync version).

        Note: Prefer using get_project_state_async for database-backed state.

        Args:
            project_id: Project identifier.

        Returns:
            Project sync state.
        """
        if project_id not in self._projects:
            self._projects[project_id] = ProjectSyncState(project_id=project_id)
        return self._projects[project_id]

    async def get_project_state_async(self, project_id: str) -> ProjectSyncState:
        """Get or create project sync state with database loading.

        This method will load state from the database if not already cached.

        Args:
            project_id: Project identifier.

        Returns:
            Project sync state.
        """
        # Return cached state if already loaded
        if project_id in self._projects:
            return self._projects[project_id]

        # Try to load from database
        if self._persistence and project_id not in self._loaded_projects:
            state, local_specs, base_specs, pending_events = await self._persistence.load_project_state(
                project_id,
                self._user_id,
            )

            if state:
                self._projects[project_id] = state
                self._local_specs.update(local_specs)
                self._base_specs.update(base_specs)
                self._pending_events.extend(pending_events)
                self._loaded_projects.add(project_id)

                logger.info(
                    "Loaded project state from database",
                    project_id=project_id,
                    tests_count=len(state.tests),
                    pending_events=len(pending_events),
                )
                return state

            self._loaded_projects.add(project_id)

        # Create new state if not found
        if project_id not in self._projects:
            self._projects[project_id] = ProjectSyncState(project_id=project_id)
            self._mark_dirty(project_id)

        return self._projects[project_id]

    def _mark_dirty(self, project_id: str) -> None:
        """Mark a project as needing persistence."""
        if self._persistence:
            self._dirty_projects.add(project_id)

    def get_sync_status(self, project_id: str) -> dict:
        """Get sync status for a project.

        Args:
            project_id: Project identifier.

        Returns:
            Status dictionary.
        """
        state = self.get_project_state(project_id)
        return state.to_dict()

    # =========================================================================
    # Local Changes
    # =========================================================================

    def track_test(
        self,
        project_id: str,
        test_id: str,
        spec: dict,
    ) -> None:
        """Start tracking a test for synchronization.

        Args:
            project_id: Project identifier.
            test_id: Test identifier.
            spec: Initial test specification.
        """
        self._local_specs[test_id] = spec.copy()
        self._base_specs[test_id] = spec.copy()

        project = self.get_project_state(project_id)
        test_state = project.get_test_state(test_id)
        test_state.checksum = self._detector.calculate_checksum(spec)

        # Mark project as needing persistence
        self._mark_dirty(project_id)

    async def track_test_async(
        self,
        project_id: str,
        test_id: str,
        spec: dict,
    ) -> None:
        """Start tracking a test for synchronization (async with DB persistence).

        Args:
            project_id: Project identifier.
            test_id: Test identifier.
            spec: Initial test specification.
        """
        self._local_specs[test_id] = spec.copy()
        self._base_specs[test_id] = spec.copy()

        project = await self.get_project_state_async(project_id)
        test_state = project.get_test_state(test_id)
        test_state.checksum = self._detector.calculate_checksum(spec)

        # Mark project as needing persistence
        self._mark_dirty(project_id)

    def update_local(
        self,
        project_id: str,
        test_id: str,
        new_spec: dict,
    ) -> DiffResult:
        """Record a local change to a test.

        Args:
            project_id: Project identifier.
            test_id: Test identifier.
            new_spec: New test specification.

        Returns:
            DiffResult showing what changed.
        """
        old_spec = self._local_specs.get(test_id)
        diff = self._detector.diff(old_spec, new_spec)

        if diff.has_changes:
            self._local_specs[test_id] = new_spec.copy()

            project = self.get_project_state(project_id)
            test_state = project.get_test_state(test_id)
            test_state.local_version += 1
            test_state.last_local_change = datetime.now(UTC)
            test_state.status = SyncStatus.PENDING
            test_state.checksum = diff.new_checksum

            # Generate and queue sync events
            events = self._detector.generate_events(
                diff,
                project_id,
                test_id,
                SyncSource.IDE,
                test_state.local_version,
            )
            test_state.pending_changes.extend(events)
            self._pending_events.extend(events)

            # Mark project as needing persistence
            self._mark_dirty(project_id)

        return diff

    async def update_local_async(
        self,
        project_id: str,
        test_id: str,
        new_spec: dict,
    ) -> DiffResult:
        """Record a local change to a test (async with DB persistence).

        Args:
            project_id: Project identifier.
            test_id: Test identifier.
            new_spec: New test specification.

        Returns:
            DiffResult showing what changed.
        """
        old_spec = self._local_specs.get(test_id)
        diff = self._detector.diff(old_spec, new_spec)

        if diff.has_changes:
            self._local_specs[test_id] = new_spec.copy()

            project = await self.get_project_state_async(project_id)
            test_state = project.get_test_state(test_id)
            test_state.local_version += 1
            test_state.last_local_change = datetime.now(UTC)
            test_state.status = SyncStatus.PENDING
            test_state.checksum = diff.new_checksum

            # Generate and queue sync events
            events = self._detector.generate_events(
                diff,
                project_id,
                test_id,
                SyncSource.IDE,
                test_state.local_version,
            )
            test_state.pending_changes.extend(events)
            self._pending_events.extend(events)

            # Save pending events to database immediately
            if self._persistence:
                for event in events:
                    await self._persistence.save_pending_event(
                        project_id,
                        self._user_id,
                        event,
                    )

            # Mark project as needing persistence
            self._mark_dirty(project_id)

        return diff

    def get_local_spec(self, test_id: str) -> dict | None:
        """Get current local spec for a test.

        Args:
            test_id: Test identifier.

        Returns:
            Local spec or None.
        """
        return self._local_specs.get(test_id)

    # =========================================================================
    # Push Operations
    # =========================================================================

    async def push(
        self,
        project_id: str,
        test_id: str | None = None,
    ) -> SyncPushResult:
        """Push local changes to the server.

        Args:
            project_id: Project identifier.
            test_id: Optional specific test to push.

        Returns:
            Push result.
        """
        if not self._push_fn:
            return SyncPushResult(
                success=False,
                error="No push handler configured",
            )

        async with self._lock:
            project = self.get_project_state(project_id)

            # Collect events to push
            if test_id:
                test_state = project.get_test_state(test_id)
                events = test_state.pending_changes[:self.config.max_events_per_push]
            else:
                events = []
                for test_state in project.tests.values():
                    events.extend(
                        test_state.pending_changes[:self.config.max_events_per_push]
                    )
                events = events[:self.config.max_events_per_push]

            if not events:
                return SyncPushResult(success=True, events_pushed=0)

            # Attempt push
            try:
                result = self._push_fn(events)

                if result.success:
                    # Collect pushed event IDs for database update
                    pushed_event_ids = []

                    # Update state on success
                    for event in events[:result.events_pushed]:
                        pushed_event_ids.append(event.id)
                        if event.test_id:
                            test_state = project.get_test_state(event.test_id)
                            if event in test_state.pending_changes:
                                test_state.pending_changes.remove(event)
                            test_state.remote_version = result.new_version

                            # Update base spec if all changes pushed
                            if not test_state.pending_changes:
                                test_state.status = SyncStatus.SYNCED
                                test_state.last_synced = datetime.now(UTC)
                                if event.test_id in self._local_specs:
                                    self._base_specs[event.test_id] = (
                                        self._local_specs[event.test_id].copy()
                                    )

                    # Remove pushed events from global queue
                    for event in events[:result.events_pushed]:
                        if event in self._pending_events:
                            self._pending_events.remove(event)

                    # Mark events as synced in database
                    if self._persistence and pushed_event_ids:
                        await self._persistence.mark_events_synced(project_id, pushed_event_ids)

                    # Mark project as dirty for persistence
                    self._mark_dirty(project_id)

                if result.conflicts:
                    # Handle conflicts
                    for conflict in result.conflicts:
                        if conflict.test_id:
                            test_state = project.get_test_state(conflict.test_id)
                            test_state.conflicts.append(conflict)
                            test_state.status = SyncStatus.CONFLICT

                return result

            except Exception as e:
                return SyncPushResult(
                    success=False,
                    error=str(e),
                )

    async def push_test(
        self,
        project_id: str,
        test_id: str,
        spec: dict,
    ) -> SyncPushResult:
        """Push a complete test spec to the server.

        Args:
            project_id: Project identifier.
            test_id: Test identifier.
            spec: Test specification.

        Returns:
            Push result.
        """
        # Record change
        self.update_local(project_id, test_id, spec)

        # Push
        return await self.push(project_id, test_id)

    # =========================================================================
    # Pull Operations
    # =========================================================================

    async def pull(
        self,
        project_id: str,
        since_version: int = 0,
    ) -> SyncPullResult:
        """Pull changes from the server.

        Args:
            project_id: Project identifier.
            since_version: Version to pull changes from.

        Returns:
            Pull result.
        """
        if not self._pull_fn:
            return SyncPullResult(
                success=False,
                error="No pull handler configured",
            )

        async with self._lock:
            try:
                result = self._pull_fn(project_id, since_version)

                if result.success:
                    # Apply remote events
                    for event in result.events:
                        await self._apply_remote_event(project_id, event)

                return result

            except Exception as e:
                return SyncPullResult(
                    success=False,
                    error=str(e),
                )

    async def _apply_remote_event(
        self,
        project_id: str,
        event: SyncEvent,
    ) -> None:
        """Apply a remote event locally.

        Args:
            project_id: Project identifier.
            event: Remote event.
        """
        if not event.test_id:
            return

        project = self.get_project_state(project_id)
        test_state = project.get_test_state(event.test_id)

        # Check for conflicts with pending local changes
        if test_state.pending_changes:
            # Detect conflict
            local_spec = self._local_specs.get(event.test_id)
            base_spec = self._base_specs.get(event.test_id)
            remote_spec = event.content.get("spec") if event.content else None

            if local_spec and remote_spec:
                conflicts = self._resolver.detect_conflicts(
                    base_spec,
                    local_spec,
                    remote_spec,
                    event.test_id,
                )

                if conflicts:
                    test_state.conflicts.extend(conflicts)
                    test_state.status = SyncStatus.CONFLICT
                    return

        # No conflicts - apply remote change
        if event.content and "spec" in event.content:
            self._local_specs[event.test_id] = event.content["spec"]
            self._base_specs[event.test_id] = event.content["spec"]

        test_state.remote_version = event.remote_version
        test_state.last_remote_change = event.timestamp
        if not test_state.pending_changes:
            test_state.status = SyncStatus.SYNCED
            test_state.last_synced = datetime.now(UTC)

        # Mark project as dirty for persistence
        self._mark_dirty(project_id)

    # =========================================================================
    # Conflict Resolution
    # =========================================================================

    async def resolve_conflict(
        self,
        project_id: str,
        conflict_id: str,
        strategy: ConflictResolutionStrategy,
        manual_value: Any = None,
    ) -> bool:
        """Resolve a specific conflict.

        Args:
            project_id: Project identifier.
            conflict_id: Conflict identifier.
            strategy: Resolution strategy.
            manual_value: Value for manual resolution.

        Returns:
            True if resolved.
        """
        async with self._lock:
            project = self.get_project_state(project_id)

            for test_state in project.tests.values():
                for conflict in test_state.conflicts:
                    if conflict.id == conflict_id:
                        resolved = self._resolver.resolve(
                            conflict,
                            strategy,
                            manual_value,
                        )

                        if resolved.resolved:
                            # Apply resolution
                            if resolved.resolved_value is not None:
                                local_spec = self._local_specs.get(test_state.test_id, {})
                                self._resolver._set_at_path(
                                    local_spec,
                                    resolved.path,
                                    resolved.resolved_value,
                                )
                                self._local_specs[test_state.test_id] = local_spec

                            # Update state
                            if all(c.resolved for c in test_state.conflicts):
                                test_state.status = SyncStatus.PENDING
                                # Queue a push for the resolved changes
                                self.update_local(
                                    project_id,
                                    test_state.test_id,
                                    self._local_specs.get(test_state.test_id, {}),
                                )

                            return True

            return False

    def get_conflicts(self, project_id: str) -> list[SyncConflict]:
        """Get all unresolved conflicts for a project.

        Args:
            project_id: Project identifier.

        Returns:
            List of unresolved conflicts.
        """
        project = self.get_project_state(project_id)
        conflicts: list[SyncConflict] = []

        for test_state in project.tests.values():
            conflicts.extend(c for c in test_state.conflicts if not c.resolved)

        return conflicts

    # =========================================================================
    # Auto-sync
    # =========================================================================

    async def _auto_sync_loop(self) -> None:
        """Background task for automatic synchronization."""
        while True:
            try:
                await asyncio.sleep(self.config.auto_sync_interval_ms / 1000)

                for project_id in self._projects:
                    project = self._projects[project_id]

                    # Skip if sync disabled
                    if not project.sync_enabled:
                        continue

                    # Push pending changes
                    if project.total_pending > 0:
                        await self.push(project_id)

            except asyncio.CancelledError:
                break
            except Exception:
                # Log but continue
                pass

    # =========================================================================
    # Full Sync
    # =========================================================================

    async def full_sync(self, project_id: str) -> dict:
        """Perform a full sync for a project.

        Args:
            project_id: Project identifier.

        Returns:
            Sync result summary.
        """
        async with self._lock:
            project = await self.get_project_state_async(project_id)

            # Pull all remote changes
            pull_result = await self.pull(project_id, 0)

            if not pull_result.success:
                return {
                    "success": False,
                    "error": pull_result.error,
                }

            # Push all local changes
            push_result = await self.push(project_id)

            project.last_full_sync = datetime.now(UTC)

            # Mark as dirty to persist the full sync timestamp
            self._mark_dirty(project_id)

            return {
                "success": push_result.success,
                "pulled_events": len(pull_result.events),
                "pushed_events": push_result.events_pushed,
                "conflicts": project.total_conflicts,
                "pending": project.total_pending,
            }

    # =========================================================================
    # Persistence
    # =========================================================================

    async def _persist_loop(self) -> None:
        """Background task for persisting state to database."""
        while True:
            try:
                await asyncio.sleep(self.config.db_sync_interval_ms / 1000)

                if self._dirty_projects:
                    await self._persist_dirty_projects()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in persistence loop", error=str(e))

    async def _persist_dirty_projects(self) -> None:
        """Persist all dirty projects to database."""
        if not self._persistence:
            return

        dirty_copy = self._dirty_projects.copy()
        self._dirty_projects.clear()

        for project_id in dirty_copy:
            if project_id in self._projects:
                project_state = self._projects[project_id]

                # Collect specs for this project's tests
                project_local_specs = {
                    test_id: self._local_specs.get(test_id, {})
                    for test_id in project_state.tests.keys()
                }
                project_base_specs = {
                    test_id: self._base_specs.get(test_id, {})
                    for test_id in project_state.tests.keys()
                }

                success = await self._persistence.save_project_state(
                    project_id,
                    self._user_id,
                    project_state,
                    project_local_specs,
                    project_base_specs,
                )

                if not success:
                    # Re-add to dirty set for retry
                    self._dirty_projects.add(project_id)
                    logger.warning("Failed to persist project state, will retry", project_id=project_id)

    async def load_all_projects(self, project_ids: list[str]) -> int:
        """Pre-load multiple projects from database.

        Args:
            project_ids: List of project IDs to load.

        Returns:
            Number of projects loaded.
        """
        loaded = 0
        for project_id in project_ids:
            state = await self.get_project_state_async(project_id)
            if state and state.tests:
                loaded += 1
        return loaded

    async def clear_project(self, project_id: str) -> bool:
        """Clear all state for a project from memory and database.

        Args:
            project_id: Project identifier.

        Returns:
            True if cleared successfully.
        """
        # Clear from memory
        if project_id in self._projects:
            project = self._projects[project_id]
            # Clear specs for this project's tests
            for test_id in project.tests.keys():
                self._local_specs.pop(test_id, None)
                self._base_specs.pop(test_id, None)
            del self._projects[project_id]

        self._loaded_projects.discard(project_id)
        self._dirty_projects.discard(project_id)

        # Remove pending events for this project
        self._pending_events = [
            e for e in self._pending_events
            if e.project_id != project_id
        ]

        # Clear from database
        if self._persistence:
            return await self._persistence.delete_project_state(project_id)

        return True

    async def force_persist(self, project_id: str | None = None) -> bool:
        """Force immediate persistence of state.

        Args:
            project_id: Optional specific project to persist.
                       If None, persists all dirty projects.

        Returns:
            True if persistence succeeded.
        """
        if not self._persistence:
            return True

        if project_id:
            self._dirty_projects.add(project_id)

        await self._persist_dirty_projects()
        return len(self._dirty_projects) == 0


def create_sync_manager(
    auto_sync: bool = True,
    resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MERGE,
    persist_to_db: bool = True,
    user_id: str | None = None,
) -> SyncManager:
    """Create a configured sync manager.

    Args:
        auto_sync: Enable automatic syncing.
        resolution_strategy: Default conflict resolution.
        persist_to_db: Enable database persistence (survives restarts).
        user_id: User ID for multi-tenant persistence.

    Returns:
        Configured SyncManager.
    """
    config = SyncConfig(
        auto_sync_enabled=auto_sync,
        default_resolution=resolution_strategy,
        persist_to_db=persist_to_db,
    )
    return SyncManager(config, user_id=user_id)
