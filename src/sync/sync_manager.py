"""Sync manager for two-way IDE synchronization."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

from .models import (
    SyncEvent,
    SyncEventType,
    SyncSource,
    SyncStatus,
    SyncConflict,
    ConflictResolutionStrategy,
    TestSyncState,
    ProjectSyncState,
    SyncPushResult,
    SyncPullResult,
    TestSpec,
)
from .change_detector import ChangeDetector, DiffResult
from .conflict_resolver import ConflictResolver, MergeResult


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


class SyncManager:
    """Manages two-way synchronization between IDE and Argus cloud.

    Features:
    - Push local changes to cloud
    - Pull remote changes
    - Automatic conflict detection and resolution
    - Version tracking
    - Optimistic updates with rollback
    """

    def __init__(
        self,
        config: Optional[SyncConfig] = None,
        push_fn: Optional[Callable[[list[SyncEvent]], SyncPushResult]] = None,
        pull_fn: Optional[Callable[[str, int], SyncPullResult]] = None,
    ):
        """Initialize sync manager.

        Args:
            config: Sync configuration.
            push_fn: Function to push events to server.
            pull_fn: Function to pull events from server.
        """
        self.config = config or SyncConfig()
        self._push_fn = push_fn
        self._pull_fn = pull_fn
        self._detector = ChangeDetector()
        self._resolver = ConflictResolver(
            default_strategy=self.config.default_resolution
        )

        # State tracking
        self._projects: dict[str, ProjectSyncState] = {}
        self._local_specs: dict[str, dict] = {}  # test_id -> spec
        self._base_specs: dict[str, dict] = {}  # test_id -> last synced spec
        self._pending_events: list[SyncEvent] = []

        # Sync control
        self._sync_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the sync manager background tasks."""
        if self.config.auto_sync_enabled and self._sync_task is None:
            self._sync_task = asyncio.create_task(self._auto_sync_loop())

    async def stop(self) -> None:
        """Stop the sync manager."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

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
        """Get or create project sync state.

        Args:
            project_id: Project identifier.

        Returns:
            Project sync state.
        """
        if project_id not in self._projects:
            self._projects[project_id] = ProjectSyncState(project_id=project_id)
        return self._projects[project_id]

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
            test_state.last_local_change = datetime.now(timezone.utc)
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

        return diff

    def get_local_spec(self, test_id: str) -> Optional[dict]:
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
        test_id: Optional[str] = None,
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
                    # Update state on success
                    for event in events[:result.events_pushed]:
                        if event.test_id:
                            test_state = project.get_test_state(event.test_id)
                            if event in test_state.pending_changes:
                                test_state.pending_changes.remove(event)
                            test_state.remote_version = result.new_version

                            # Update base spec if all changes pushed
                            if not test_state.pending_changes:
                                test_state.status = SyncStatus.SYNCED
                                test_state.last_synced = datetime.now(timezone.utc)
                                if event.test_id in self._local_specs:
                                    self._base_specs[event.test_id] = (
                                        self._local_specs[event.test_id].copy()
                                    )

                    # Remove pushed events from global queue
                    for event in events[:result.events_pushed]:
                        if event in self._pending_events:
                            self._pending_events.remove(event)

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
            test_state.last_synced = datetime.now(timezone.utc)

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
            project = self.get_project_state(project_id)

            # Pull all remote changes
            pull_result = await self.pull(project_id, 0)

            if not pull_result.success:
                return {
                    "success": False,
                    "error": pull_result.error,
                }

            # Push all local changes
            push_result = await self.push(project_id)

            project.last_full_sync = datetime.now(timezone.utc)

            return {
                "success": push_result.success,
                "pulled_events": len(pull_result.events),
                "pushed_events": push_result.events_pushed,
                "conflicts": project.total_conflicts,
                "pending": project.total_pending,
            }


def create_sync_manager(
    auto_sync: bool = True,
    resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MERGE,
) -> SyncManager:
    """Create a configured sync manager.

    Args:
        auto_sync: Enable automatic syncing.
        resolution_strategy: Default conflict resolution.

    Returns:
        Configured SyncManager.
    """
    config = SyncConfig(
        auto_sync_enabled=auto_sync,
        default_resolution=resolution_strategy,
    )
    return SyncManager(config)
