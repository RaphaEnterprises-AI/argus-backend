"""Tests for sync models."""


from src.sync.models import (
    ConflictResolutionStrategy,
    ProjectSyncState,
    SyncConflict,
    SyncEvent,
    SyncEventType,
    SyncPullResult,
    SyncPushResult,
    SyncSource,
    SyncStatus,
    TestSpec,
    TestSyncState,
)

# =============================================================================
# Enum Tests
# =============================================================================


class TestSyncSource:
    """Tests for SyncSource enum."""

    def test_source_values(self):
        """Test source values."""
        assert SyncSource.IDE.value == "ide"
        assert SyncSource.CLOUD.value == "cloud"
        assert SyncSource.MCP.value == "mcp"


class TestSyncEventType:
    """Tests for SyncEventType enum."""

    def test_test_events(self):
        """Test test event types."""
        assert SyncEventType.TEST_CREATED.value == "test_created"
        assert SyncEventType.TEST_UPDATED.value == "test_updated"
        assert SyncEventType.TEST_DELETED.value == "test_deleted"

    def test_step_events(self):
        """Test step event types."""
        assert SyncEventType.STEP_ADDED.value == "step_added"
        assert SyncEventType.STEP_UPDATED.value == "step_updated"
        assert SyncEventType.STEP_REMOVED.value == "step_removed"

    def test_sync_events(self):
        """Test sync control events."""
        assert SyncEventType.FULL_SYNC.value == "full_sync"
        assert SyncEventType.CONFLICT_DETECTED.value == "conflict_detected"


class TestConflictResolutionStrategy:
    """Tests for ConflictResolutionStrategy enum."""

    def test_strategy_values(self):
        """Test strategy values."""
        assert ConflictResolutionStrategy.KEEP_LOCAL.value == "keep_local"
        assert ConflictResolutionStrategy.KEEP_REMOTE.value == "keep_remote"
        assert ConflictResolutionStrategy.MERGE.value == "merge"
        assert ConflictResolutionStrategy.LATEST_WINS.value == "latest_wins"


class TestSyncStatus:
    """Tests for SyncStatus enum."""

    def test_status_values(self):
        """Test status values."""
        assert SyncStatus.SYNCED.value == "synced"
        assert SyncStatus.PENDING.value == "pending"
        assert SyncStatus.CONFLICT.value == "conflict"
        assert SyncStatus.ERROR.value == "error"


# =============================================================================
# SyncEvent Tests
# =============================================================================


class TestSyncEvent:
    """Tests for SyncEvent dataclass."""

    def test_create_event(self):
        """Test creating an event."""
        event = SyncEvent(
            type=SyncEventType.TEST_UPDATED,
            source=SyncSource.IDE,
            project_id="proj-1",
            test_id="test-1"
        )
        assert event.type == SyncEventType.TEST_UPDATED
        assert event.source == SyncSource.IDE
        assert event.project_id == "proj-1"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = SyncEvent(
            id="evt-1",
            type=SyncEventType.STEP_ADDED,
            source=SyncSource.MCP,
            project_id="proj-1",
            test_id="test-1",
            path=["steps", "0"],
            content={"action": "click"},
            local_version=5
        )
        data = event.to_dict()

        assert data["id"] == "evt-1"
        assert data["type"] == "step_added"
        assert data["source"] == "mcp"
        assert data["path"] == ["steps", "0"]
        assert data["local_version"] == 5

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "evt-1",
            "type": "test_updated",
            "source": "cloud",
            "project_id": "proj-1",
            "test_id": "test-1",
            "local_version": 3,
            "timestamp": "2026-01-08T10:00:00+00:00"
        }
        event = SyncEvent.from_dict(data)

        assert event.id == "evt-1"
        assert event.type == SyncEventType.TEST_UPDATED
        assert event.source == SyncSource.CLOUD


# =============================================================================
# SyncConflict Tests
# =============================================================================


class TestSyncConflict:
    """Tests for SyncConflict dataclass."""

    def test_create_conflict(self):
        """Test creating a conflict."""
        conflict = SyncConflict(
            test_id="test-1",
            path=["steps", "0", "target"],
            local_value="#button",
            remote_value="#btn"
        )
        assert conflict.test_id == "test-1"
        assert conflict.local_value == "#button"
        assert conflict.resolved is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        conflict = SyncConflict(
            id="conf-1",
            test_id="test-1",
            path=["name"],
            local_value="Local Name",
            remote_value="Remote Name",
            resolved=True,
            resolution=ConflictResolutionStrategy.KEEP_LOCAL
        )
        data = conflict.to_dict()

        assert data["id"] == "conf-1"
        assert data["resolved"] is True
        assert data["resolution"] == "keep_local"


# =============================================================================
# TestSyncState Tests
# =============================================================================


class TestTestSyncState:
    """Tests for TestSyncState dataclass."""

    def test_create_state(self):
        """Test creating test state."""
        state = TestSyncState(test_id="test-1")
        assert state.test_id == "test-1"
        assert state.status == SyncStatus.SYNCED
        assert state.local_version == 0

    def test_has_conflicts(self):
        """Test has_conflicts property."""
        state = TestSyncState(test_id="test-1")
        assert state.has_conflicts is False

        state.conflicts.append(SyncConflict(test_id="test-1"))
        assert state.has_conflicts is True

    def test_has_conflicts_ignores_resolved(self):
        """Test resolved conflicts don't count."""
        state = TestSyncState(test_id="test-1")
        conflict = SyncConflict(test_id="test-1", resolved=True)
        state.conflicts.append(conflict)

        assert state.has_conflicts is False

    def test_has_pending_changes(self):
        """Test has_pending_changes property."""
        state = TestSyncState(test_id="test-1")
        assert state.has_pending_changes is False

        state.pending_changes.append(SyncEvent(project_id="p", test_id="test-1"))
        assert state.has_pending_changes is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = TestSyncState(
            test_id="test-1",
            local_version=5,
            remote_version=4,
            status=SyncStatus.PENDING
        )
        data = state.to_dict()

        assert data["test_id"] == "test-1"
        assert data["local_version"] == 5
        assert data["status"] == "pending"


# =============================================================================
# ProjectSyncState Tests
# =============================================================================


class TestProjectSyncState:
    """Tests for ProjectSyncState dataclass."""

    def test_create_project_state(self):
        """Test creating project state."""
        state = ProjectSyncState(project_id="proj-1")
        assert state.project_id == "proj-1"
        assert len(state.tests) == 0

    def test_get_test_state_creates(self):
        """Test get_test_state creates new state."""
        project = ProjectSyncState(project_id="proj-1")
        test_state = project.get_test_state("test-1")

        assert test_state.test_id == "test-1"
        assert "test-1" in project.tests

    def test_overall_status_synced(self):
        """Test overall status when all synced."""
        project = ProjectSyncState(project_id="proj-1")
        project.tests["t1"] = TestSyncState(test_id="t1", status=SyncStatus.SYNCED)
        project.tests["t2"] = TestSyncState(test_id="t2", status=SyncStatus.SYNCED)

        assert project.status == SyncStatus.SYNCED

    def test_overall_status_conflict(self):
        """Test overall status with conflict."""
        project = ProjectSyncState(project_id="proj-1")
        project.tests["t1"] = TestSyncState(test_id="t1", status=SyncStatus.SYNCED)
        project.tests["t2"] = TestSyncState(test_id="t2", status=SyncStatus.CONFLICT)

        assert project.status == SyncStatus.CONFLICT

    def test_overall_status_error_highest(self):
        """Test error status takes precedence."""
        project = ProjectSyncState(project_id="proj-1")
        project.tests["t1"] = TestSyncState(test_id="t1", status=SyncStatus.ERROR)
        project.tests["t2"] = TestSyncState(test_id="t2", status=SyncStatus.CONFLICT)

        assert project.status == SyncStatus.ERROR

    def test_total_pending(self):
        """Test total pending count."""
        project = ProjectSyncState(project_id="proj-1")
        t1 = project.get_test_state("t1")
        t1.pending_changes = [SyncEvent(project_id="p"), SyncEvent(project_id="p")]
        t2 = project.get_test_state("t2")
        t2.pending_changes = [SyncEvent(project_id="p")]

        assert project.total_pending == 3


# =============================================================================
# SyncPushResult Tests
# =============================================================================


class TestSyncPushResult:
    """Tests for SyncPushResult dataclass."""

    def test_success_result(self):
        """Test successful push result."""
        result = SyncPushResult(
            success=True,
            events_pushed=5,
            new_version=10
        )
        assert result.success is True
        assert result.events_pushed == 5

    def test_failure_result(self):
        """Test failed push result."""
        result = SyncPushResult(
            success=False,
            error="Network error"
        )
        assert result.success is False
        assert result.error == "Network error"


# =============================================================================
# SyncPullResult Tests
# =============================================================================


class TestSyncPullResult:
    """Tests for SyncPullResult dataclass."""

    def test_success_result(self):
        """Test successful pull result."""
        result = SyncPullResult(
            success=True,
            events=[SyncEvent(project_id="p")],
            new_version=15
        )
        assert result.success is True
        assert len(result.events) == 1


# =============================================================================
# TestSpec Tests
# =============================================================================


class TestTestSpec:
    """Tests for TestSpec dataclass."""

    def test_create_spec(self):
        """Test creating a spec."""
        spec = TestSpec(
            id="test-1",
            name="Login Test",
            steps=[{"action": "goto", "target": "/login"}]
        )
        assert spec.id == "test-1"
        assert spec.name == "Login Test"
        assert len(spec.steps) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        spec = TestSpec(
            id="test-1",
            name="Test",
            description="A test",
            steps=[{"action": "click"}],
            version=3
        )
        data = spec.to_dict()

        assert data["id"] == "test-1"
        assert data["version"] == 3

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "test-1",
            "name": "From Dict",
            "steps": [{"action": "fill"}],
            "metadata": {"tag": "smoke"}
        }
        spec = TestSpec.from_dict(data)

        assert spec.id == "test-1"
        assert spec.name == "From Dict"
        assert spec.metadata["tag"] == "smoke"
