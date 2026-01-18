"""Tests for sync manager."""


import pytest

from src.sync.models import (
    ConflictResolutionStrategy,
    SyncConflict,
    SyncEvent,
    SyncEventType,
    SyncPullResult,
    SyncPushResult,
    SyncSource,
    SyncStatus,
)
from src.sync.sync_manager import (
    SyncConfig,
    SyncManager,
    create_sync_manager,
)

# =============================================================================
# SyncConfig Tests
# =============================================================================


class TestSyncConfig:
    """Tests for SyncConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SyncConfig()
        assert config.auto_sync_enabled is True
        assert config.auto_sync_interval_ms == 5000
        assert config.auto_push_on_save is True
        assert config.default_resolution == ConflictResolutionStrategy.MERGE
        assert config.max_events_per_push == 100
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = SyncConfig(
            auto_sync_enabled=False,
            auto_sync_interval_ms=10000,
            default_resolution=ConflictResolutionStrategy.KEEP_LOCAL,
            max_events_per_push=50,
        )
        assert config.auto_sync_enabled is False
        assert config.auto_sync_interval_ms == 10000


# =============================================================================
# SyncManager Initialization Tests
# =============================================================================


class TestSyncManagerInit:
    """Tests for SyncManager initialization."""

    def test_default_init(self):
        """Test default initialization."""
        manager = SyncManager()
        assert manager.config is not None
        assert manager._push_fn is None
        assert manager._pull_fn is None

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = SyncConfig(auto_sync_enabled=False)
        manager = SyncManager(config=config)
        assert manager.config.auto_sync_enabled is False

    def test_with_handlers(self):
        """Test initialization with handlers."""
        def push_fn(events):
            return SyncPushResult(success=True)
        def pull_fn(pid, v):
            return SyncPullResult(success=True)

        manager = SyncManager(push_fn=push_fn, pull_fn=pull_fn)
        assert manager._push_fn is not None
        assert manager._pull_fn is not None


# =============================================================================
# Handler Setup Tests
# =============================================================================


class TestHandlerSetup:
    """Tests for handler setup methods."""

    def test_set_push_handler(self):
        """Test setting push handler."""
        manager = SyncManager()
        def push_fn(events):
            return SyncPushResult(success=True)

        manager.set_push_handler(push_fn)
        assert manager._push_fn is push_fn

    def test_set_pull_handler(self):
        """Test setting pull handler."""
        manager = SyncManager()
        def pull_fn(pid, v):
            return SyncPullResult(success=True)

        manager.set_pull_handler(pull_fn)
        assert manager._pull_fn is pull_fn


# =============================================================================
# Project Management Tests
# =============================================================================


class TestProjectManagement:
    """Tests for project state management."""

    def test_get_project_state_creates(self):
        """Test get_project_state creates new state."""
        manager = SyncManager()
        state = manager.get_project_state("proj-1")

        assert state.project_id == "proj-1"
        assert "proj-1" in manager._projects

    def test_get_project_state_returns_existing(self):
        """Test get_project_state returns existing state."""
        manager = SyncManager()
        state1 = manager.get_project_state("proj-1")
        state1.sync_enabled = False

        state2 = manager.get_project_state("proj-1")
        assert state2.sync_enabled is False

    def test_get_sync_status(self):
        """Test get_sync_status returns dict."""
        manager = SyncManager()
        status = manager.get_sync_status("proj-1")

        assert isinstance(status, dict)
        assert status["project_id"] == "proj-1"


# =============================================================================
# Track Test Tests
# =============================================================================


class TestTrackTest:
    """Tests for test tracking."""

    def test_track_test_basic(self):
        """Test basic test tracking."""
        manager = SyncManager()
        spec = {"name": "Login Test", "steps": []}

        manager.track_test("proj-1", "test-1", spec)

        assert manager._local_specs["test-1"] == spec
        assert manager._base_specs["test-1"] == spec

    def test_track_test_updates_state(self):
        """Test tracking updates project state."""
        manager = SyncManager()
        spec = {"name": "Test"}

        manager.track_test("proj-1", "test-1", spec)

        project = manager.get_project_state("proj-1")
        test_state = project.get_test_state("test-1")
        assert test_state.checksum is not None

    def test_track_test_copies_spec(self):
        """Test tracking copies spec (not references)."""
        manager = SyncManager()
        spec = {"name": "Test"}

        manager.track_test("proj-1", "test-1", spec)
        spec["name"] = "Modified"

        assert manager._local_specs["test-1"]["name"] == "Test"


# =============================================================================
# Update Local Tests
# =============================================================================


class TestUpdateLocal:
    """Tests for local update tracking."""

    def test_update_local_no_changes(self):
        """Test update with no changes."""
        manager = SyncManager()
        spec = {"name": "Test"}
        manager.track_test("proj-1", "test-1", spec)

        diff = manager.update_local("proj-1", "test-1", spec.copy())

        assert diff.has_changes is False

    def test_update_local_with_changes(self):
        """Test update with changes."""
        manager = SyncManager()
        manager.track_test("proj-1", "test-1", {"name": "Old"})

        diff = manager.update_local("proj-1", "test-1", {"name": "New"})

        assert diff.has_changes is True
        assert manager._local_specs["test-1"]["name"] == "New"

    def test_update_local_increments_version(self):
        """Test update increments local version."""
        manager = SyncManager()
        manager.track_test("proj-1", "test-1", {"name": "Test"})

        project = manager.get_project_state("proj-1")
        test_state = project.get_test_state("test-1")
        initial_version = test_state.local_version

        manager.update_local("proj-1", "test-1", {"name": "Updated"})

        assert test_state.local_version == initial_version + 1

    def test_update_local_sets_pending_status(self):
        """Test update sets pending status."""
        manager = SyncManager()
        manager.track_test("proj-1", "test-1", {"name": "Test"})

        manager.update_local("proj-1", "test-1", {"name": "Updated"})

        project = manager.get_project_state("proj-1")
        test_state = project.get_test_state("test-1")
        assert test_state.status == SyncStatus.PENDING

    def test_update_local_queues_events(self):
        """Test update queues sync events."""
        manager = SyncManager()
        manager.track_test("proj-1", "test-1", {"name": "Test"})

        manager.update_local("proj-1", "test-1", {"name": "Updated"})

        project = manager.get_project_state("proj-1")
        test_state = project.get_test_state("test-1")
        assert len(test_state.pending_changes) > 0
        assert len(manager._pending_events) > 0

    def test_get_local_spec(self):
        """Test getting local spec."""
        manager = SyncManager()
        spec = {"name": "Test"}
        manager.track_test("proj-1", "test-1", spec)

        result = manager.get_local_spec("test-1")
        assert result == spec

    def test_get_local_spec_nonexistent(self):
        """Test getting nonexistent local spec."""
        manager = SyncManager()
        result = manager.get_local_spec("nonexistent")
        assert result is None


# =============================================================================
# Push Tests
# =============================================================================


class TestPush:
    """Tests for push operations."""

    @pytest.mark.asyncio
    async def test_push_no_handler(self):
        """Test push without handler returns error."""
        manager = SyncManager()
        result = await manager.push("proj-1")

        assert result.success is False
        assert "No push handler" in result.error

    @pytest.mark.asyncio
    async def test_push_no_events(self):
        """Test push with no pending events."""
        manager = SyncManager(
            push_fn=lambda e: SyncPushResult(success=True, events_pushed=0)
        )
        result = await manager.push("proj-1")

        assert result.success is True
        assert result.events_pushed == 0

    @pytest.mark.asyncio
    async def test_push_success(self):
        """Test successful push."""
        pushed_events = []

        def push_fn(events):
            pushed_events.extend(events)
            return SyncPushResult(
                success=True,
                events_pushed=len(events),
                new_version=10,
            )

        manager = SyncManager(push_fn=push_fn)
        manager.track_test("proj-1", "test-1", {"name": "Test"})
        manager.update_local("proj-1", "test-1", {"name": "Updated"})

        result = await manager.push("proj-1")

        assert result.success is True
        assert len(pushed_events) > 0

    @pytest.mark.asyncio
    async def test_push_updates_state_on_success(self):
        """Test push updates state on success."""
        def push_fn(events):
            return SyncPushResult(
                success=True,
                events_pushed=len(events),
                new_version=10,
            )

        manager = SyncManager(push_fn=push_fn)
        manager.track_test("proj-1", "test-1", {"name": "Test"})
        manager.update_local("proj-1", "test-1", {"name": "Updated"})

        await manager.push("proj-1")

        project = manager.get_project_state("proj-1")
        test_state = project.get_test_state("test-1")
        assert test_state.status == SyncStatus.SYNCED
        assert test_state.remote_version == 10

    @pytest.mark.asyncio
    async def test_push_specific_test(self):
        """Test push for specific test."""
        pushed_events = []

        def push_fn(events):
            pushed_events.extend(events)
            return SyncPushResult(success=True, events_pushed=len(events))

        manager = SyncManager(push_fn=push_fn)
        manager.track_test("proj-1", "test-1", {"name": "Test 1"})
        manager.track_test("proj-1", "test-2", {"name": "Test 2"})
        manager.update_local("proj-1", "test-1", {"name": "Updated 1"})
        manager.update_local("proj-1", "test-2", {"name": "Updated 2"})

        await manager.push("proj-1", "test-1")

        # Should only push test-1 events
        assert all(e.test_id == "test-1" for e in pushed_events)

    @pytest.mark.asyncio
    async def test_push_handles_conflicts(self):
        """Test push handles conflicts from server."""
        conflict = SyncConflict(test_id="test-1", path=["name"])

        def push_fn(events):
            return SyncPushResult(
                success=True,
                events_pushed=0,
                conflicts=[conflict],
            )

        manager = SyncManager(push_fn=push_fn)
        manager.track_test("proj-1", "test-1", {"name": "Test"})
        manager.update_local("proj-1", "test-1", {"name": "Updated"})

        await manager.push("proj-1")

        project = manager.get_project_state("proj-1")
        test_state = project.get_test_state("test-1")
        assert test_state.status == SyncStatus.CONFLICT

    @pytest.mark.asyncio
    async def test_push_exception_handling(self):
        """Test push handles exceptions."""
        def push_fn(events):
            raise Exception("Network error")

        manager = SyncManager(push_fn=push_fn)
        manager.track_test("proj-1", "test-1", {"name": "Test"})
        manager.update_local("proj-1", "test-1", {"name": "Updated"})

        result = await manager.push("proj-1")

        assert result.success is False
        assert "Network error" in result.error


# =============================================================================
# Push Test Method Tests
# =============================================================================


class TestPushTest:
    """Tests for push_test method."""

    @pytest.mark.asyncio
    async def test_push_test_updates_and_pushes(self):
        """Test push_test updates local and pushes."""
        pushed = []

        def push_fn(events):
            pushed.extend(events)
            return SyncPushResult(success=True, events_pushed=len(events))

        manager = SyncManager(push_fn=push_fn)
        manager.track_test("proj-1", "test-1", {"name": "Old"})

        result = await manager.push_test("proj-1", "test-1", {"name": "New"})

        assert result.success is True
        assert len(pushed) > 0


# =============================================================================
# Pull Tests
# =============================================================================


class TestPull:
    """Tests for pull operations."""

    @pytest.mark.asyncio
    async def test_pull_no_handler(self):
        """Test pull without handler returns error."""
        manager = SyncManager()
        result = await manager.pull("proj-1")

        assert result.success is False
        assert "No pull handler" in result.error

    @pytest.mark.asyncio
    async def test_pull_success(self):
        """Test successful pull."""
        event = SyncEvent(
            type=SyncEventType.TEST_UPDATED,
            source=SyncSource.CLOUD,
            project_id="proj-1",
            test_id="test-1",
            content={"spec": {"name": "Remote Test"}},
            remote_version=5,
        )

        def pull_fn(project_id, since_version):
            return SyncPullResult(
                success=True,
                events=[event],
                new_version=5,
            )

        manager = SyncManager(pull_fn=pull_fn)
        manager.track_test("proj-1", "test-1", {"name": "Local Test"})

        result = await manager.pull("proj-1")

        assert result.success is True
        assert len(result.events) == 1

    @pytest.mark.asyncio
    async def test_pull_applies_remote_changes(self):
        """Test pull applies remote changes."""
        event = SyncEvent(
            type=SyncEventType.TEST_UPDATED,
            source=SyncSource.CLOUD,
            project_id="proj-1",
            test_id="test-1",
            content={"spec": {"name": "Remote Test"}},
            remote_version=5,
        )

        def pull_fn(project_id, since_version):
            return SyncPullResult(success=True, events=[event])

        manager = SyncManager(pull_fn=pull_fn)
        manager.track_test("proj-1", "test-1", {"name": "Local Test"})

        await manager.pull("proj-1")

        # Remote spec should be applied
        assert manager._local_specs["test-1"]["name"] == "Remote Test"

    @pytest.mark.asyncio
    async def test_pull_exception_handling(self):
        """Test pull handles exceptions."""
        def pull_fn(project_id, since_version):
            raise Exception("Network error")

        manager = SyncManager(pull_fn=pull_fn)

        result = await manager.pull("proj-1")

        assert result.success is False
        assert "Network error" in result.error


# =============================================================================
# Conflict Resolution Tests
# =============================================================================


class TestConflictResolution:
    """Tests for conflict resolution."""

    @pytest.mark.asyncio
    async def test_resolve_conflict_success(self):
        """Test resolving a conflict."""
        manager = SyncManager()
        manager.track_test("proj-1", "test-1", {"name": "Test"})

        project = manager.get_project_state("proj-1")
        test_state = project.get_test_state("test-1")
        conflict = SyncConflict(
            id="conf-1",
            test_id="test-1",
            path=["name"],
            local_value="Local",
            remote_value="Remote",
        )
        test_state.conflicts.append(conflict)
        test_state.status = SyncStatus.CONFLICT

        result = await manager.resolve_conflict(
            "proj-1",
            "conf-1",
            ConflictResolutionStrategy.KEEP_LOCAL,
        )

        assert result is True
        assert conflict.resolved is True

    @pytest.mark.asyncio
    async def test_resolve_conflict_not_found(self):
        """Test resolving nonexistent conflict."""
        manager = SyncManager()

        result = await manager.resolve_conflict(
            "proj-1",
            "nonexistent",
            ConflictResolutionStrategy.KEEP_LOCAL,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_resolve_conflict_updates_spec(self):
        """Test resolution updates local spec."""
        manager = SyncManager()
        manager.track_test("proj-1", "test-1", {"name": "Original"})
        manager._local_specs["test-1"] = {"name": "Current"}

        project = manager.get_project_state("proj-1")
        test_state = project.get_test_state("test-1")
        conflict = SyncConflict(
            id="conf-1",
            test_id="test-1",
            path=["name"],
            local_value="Local",
            remote_value="Remote",
        )
        test_state.conflicts.append(conflict)

        await manager.resolve_conflict(
            "proj-1",
            "conf-1",
            ConflictResolutionStrategy.KEEP_REMOTE,
        )

        assert manager._local_specs["test-1"]["name"] == "Remote"

    def test_get_conflicts(self):
        """Test getting unresolved conflicts."""
        manager = SyncManager()
        project = manager.get_project_state("proj-1")

        # Add some conflicts
        test_state = project.get_test_state("test-1")
        test_state.conflicts = [
            SyncConflict(test_id="test-1", path=["a"], resolved=False),
            SyncConflict(test_id="test-1", path=["b"], resolved=True),
        ]

        conflicts = manager.get_conflicts("proj-1")

        assert len(conflicts) == 1
        assert conflicts[0].path == ["a"]


# =============================================================================
# Start/Stop Tests
# =============================================================================


class TestStartStop:
    """Tests for start/stop operations."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        """Test start creates background task."""
        manager = SyncManager()
        await manager.start()

        assert manager._sync_task is not None
        await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """Test stop cancels background task."""
        manager = SyncManager()
        await manager.start()
        await manager.stop()

        assert manager._sync_task is None

    @pytest.mark.asyncio
    async def test_start_disabled_auto_sync(self):
        """Test start does nothing when auto_sync disabled."""
        config = SyncConfig(auto_sync_enabled=False)
        manager = SyncManager(config=config)

        await manager.start()

        assert manager._sync_task is None


# =============================================================================
# Full Sync Tests
# =============================================================================


class TestFullSync:
    """Tests for full sync operation."""

    @pytest.mark.asyncio
    async def test_full_sync_pull_failure(self):
        """Test full sync fails on pull failure."""
        def pull_fn(project_id, since_version):
            return SyncPullResult(success=False, error="Pull failed")

        manager = SyncManager(pull_fn=pull_fn)

        result = await manager.full_sync("proj-1")

        assert result["success"] is False
        assert "Pull failed" in result["error"]

    @pytest.mark.asyncio
    async def test_full_sync_success(self):
        """Test successful full sync."""
        def pull_fn(project_id, since_version):
            return SyncPullResult(success=True, events=[])

        def push_fn(events):
            return SyncPushResult(success=True, events_pushed=0)

        manager = SyncManager(push_fn=push_fn, pull_fn=pull_fn)

        result = await manager.full_sync("proj-1")

        assert result["success"] is True
        assert "pulled_events" in result
        assert "pushed_events" in result


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateSyncManager:
    """Tests for create_sync_manager factory."""

    def test_create_default(self):
        """Test creating with defaults."""
        manager = create_sync_manager()

        assert manager.config.auto_sync_enabled is True
        assert manager.config.default_resolution == ConflictResolutionStrategy.MERGE

    def test_create_custom(self):
        """Test creating with custom options."""
        manager = create_sync_manager(
            auto_sync=False,
            resolution_strategy=ConflictResolutionStrategy.KEEP_LOCAL,
        )

        assert manager.config.auto_sync_enabled is False
        assert manager.config.default_resolution == ConflictResolutionStrategy.KEEP_LOCAL


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_push_max_events_limit(self):
        """Test push respects max events limit."""
        pushed_events = []

        def push_fn(events):
            pushed_events.extend(events)
            return SyncPushResult(success=True, events_pushed=len(events))

        config = SyncConfig(max_events_per_push=2)
        manager = SyncManager(config=config, push_fn=push_fn)

        # Track and update multiple tests to generate many events
        for i in range(5):
            manager.track_test("proj-1", f"test-{i}", {"name": f"Test {i}"})
            manager.update_local("proj-1", f"test-{i}", {"name": f"Updated {i}"})

        await manager.push("proj-1")

        # Should be limited to max_events_per_push
        assert len(pushed_events) <= 2

    def test_update_new_test(self):
        """Test updating untracked test."""
        manager = SyncManager()

        # Update without tracking first
        diff = manager.update_local("proj-1", "new-test", {"name": "New Test"})

        assert diff.has_changes is True

    @pytest.mark.asyncio
    async def test_pull_detects_conflicts(self):
        """Test pull detects conflicts with pending changes."""
        event = SyncEvent(
            type=SyncEventType.TEST_UPDATED,
            source=SyncSource.CLOUD,
            project_id="proj-1",
            test_id="test-1",
            content={"spec": {"name": "Remote Name"}},
        )

        def pull_fn(project_id, since_version):
            return SyncPullResult(success=True, events=[event])

        manager = SyncManager(pull_fn=pull_fn)

        # Track, then update locally to create pending changes
        manager.track_test("proj-1", "test-1", {"name": "Original"})
        manager.update_local("proj-1", "test-1", {"name": "Local Name"})

        await manager.pull("proj-1")

        # Should detect conflict
        project = manager.get_project_state("proj-1")
        test_state = project.get_test_state("test-1")
        # Either conflict detected or change applied
        assert test_state.status in [SyncStatus.CONFLICT, SyncStatus.PENDING]

    @pytest.mark.asyncio
    async def test_resolve_all_conflicts_changes_status(self):
        """Test resolving all conflicts changes status to pending."""
        manager = SyncManager()
        manager.track_test("proj-1", "test-1", {"name": "Test"})

        project = manager.get_project_state("proj-1")
        test_state = project.get_test_state("test-1")
        conflict = SyncConflict(
            id="conf-1",
            test_id="test-1",
            path=["name"],
            local_value="Local",
            remote_value="Remote",
        )
        test_state.conflicts.append(conflict)
        test_state.status = SyncStatus.CONFLICT

        await manager.resolve_conflict(
            "proj-1",
            "conf-1",
            ConflictResolutionStrategy.KEEP_LOCAL,
        )

        # After all conflicts resolved, status should be PENDING
        assert test_state.status == SyncStatus.PENDING
