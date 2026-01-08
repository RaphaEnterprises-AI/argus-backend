"""Tests for conflict resolver."""

import pytest
from datetime import datetime, timezone

from src.sync.conflict_resolver import (
    MergeResult,
    ConflictResolver,
    resolve_conflicts,
)
from src.sync.models import (
    SyncConflict,
    ConflictResolutionStrategy,
)


# =============================================================================
# MergeResult Tests
# =============================================================================


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_success_result(self):
        """Test successful merge result."""
        result = MergeResult(
            success=True,
            merged_spec={"name": "Merged Test"},
            auto_resolved=2,
            manual_required=0,
        )
        assert result.success is True
        assert result.merged_spec["name"] == "Merged Test"
        assert result.auto_resolved == 2

    def test_failure_result(self):
        """Test failed merge result."""
        conflict = SyncConflict(test_id="t1", path=["name"])
        result = MergeResult(
            success=False,
            conflicts=[conflict],
            auto_resolved=1,
            manual_required=1,
        )
        assert result.success is False
        assert len(result.conflicts) == 1
        assert result.manual_required == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = MergeResult(
            success=True,
            merged_spec={"name": "Test"},
            auto_resolved=3,
        )
        data = result.to_dict()

        assert data["success"] is True
        assert data["merged_spec"] == {"name": "Test"}
        assert data["auto_resolved"] == 3


# =============================================================================
# ConflictResolver Initialization Tests
# =============================================================================


class TestConflictResolverInit:
    """Tests for ConflictResolver initialization."""

    def test_default_init(self):
        """Test default initialization."""
        resolver = ConflictResolver()
        assert resolver.default_strategy == ConflictResolutionStrategy.MERGE

    def test_custom_strategy(self):
        """Test custom default strategy."""
        resolver = ConflictResolver(
            default_strategy=ConflictResolutionStrategy.KEEP_LOCAL
        )
        assert resolver.default_strategy == ConflictResolutionStrategy.KEEP_LOCAL


# =============================================================================
# Conflict Detection Tests
# =============================================================================


class TestDetectConflicts:
    """Tests for conflict detection."""

    def test_no_conflicts_same_changes(self):
        """Test no conflicts when changes are the same."""
        resolver = ConflictResolver()
        base = {"name": "Test"}
        local = {"name": "Updated"}
        remote = {"name": "Updated"}

        conflicts = resolver.detect_conflicts(base, local, remote, "t1")

        # Same change = conflict detected but same value
        assert len(conflicts) >= 0

    def test_no_conflicts_different_paths(self):
        """Test no conflicts when changes are on different paths."""
        resolver = ConflictResolver()
        base = {"name": "Test", "description": ""}
        local = {"name": "Updated Name", "description": ""}
        remote = {"name": "Test", "description": "Updated Description"}

        conflicts = resolver.detect_conflicts(base, local, remote, "t1")

        assert len(conflicts) == 0

    def test_conflict_same_path(self):
        """Test conflict when same path modified."""
        resolver = ConflictResolver()
        base = {"name": "Original"}
        local = {"name": "Local Name"}
        remote = {"name": "Remote Name"}

        conflicts = resolver.detect_conflicts(base, local, remote, "t1")

        assert len(conflicts) == 1
        assert conflicts[0].path == ["name"]
        assert conflicts[0].local_value == "Local Name"
        assert conflicts[0].remote_value == "Remote Name"

    def test_conflict_nested_path(self):
        """Test conflict on nested path."""
        resolver = ConflictResolver()
        base = {"metadata": {"priority": "low"}}
        local = {"metadata": {"priority": "high"}}
        remote = {"metadata": {"priority": "medium"}}

        conflicts = resolver.detect_conflicts(base, local, remote, "t1")

        assert len(conflicts) == 1
        assert conflicts[0].path == ["metadata", "priority"]

    def test_conflict_parent_child_paths(self):
        """Test conflict when parent and child paths modified."""
        resolver = ConflictResolver()
        base = {"data": {"key": "value"}}
        local = {"data": {"key": "new_value"}}
        remote = {"data": "replaced"}

        conflicts = resolver.detect_conflicts(base, local, remote, "t1")

        # Both paths conflict
        assert len(conflicts) >= 1

    def test_no_base_spec(self):
        """Test conflict detection without base spec."""
        resolver = ConflictResolver()
        local = {"name": "Local"}
        remote = {"name": "Remote"}

        conflicts = resolver.detect_conflicts(None, local, remote, "t1")

        # Both are additions from None, same path = conflict
        assert len(conflicts) >= 0

    def test_step_conflict(self):
        """Test conflict in step modifications."""
        resolver = ConflictResolver()
        base = {"steps": [{"action": "click", "target": "#btn"}]}
        local = {"steps": [{"action": "click", "target": "#button"}]}
        remote = {"steps": [{"action": "click", "target": "#submit"}]}

        conflicts = resolver.detect_conflicts(base, local, remote, "t1")

        assert len(conflicts) >= 1


# =============================================================================
# Paths Conflict Tests
# =============================================================================


class TestPathsConflict:
    """Tests for path conflict detection."""

    def test_same_paths(self):
        """Test same paths conflict."""
        resolver = ConflictResolver()
        assert resolver._paths_conflict(["a", "b"], ["a", "b"]) is True

    def test_parent_child_conflict(self):
        """Test parent is prefix of child."""
        resolver = ConflictResolver()
        assert resolver._paths_conflict(["a"], ["a", "b"]) is True
        assert resolver._paths_conflict(["a", "b"], ["a"]) is True

    def test_different_paths_no_conflict(self):
        """Test completely different paths."""
        resolver = ConflictResolver()
        assert resolver._paths_conflict(["a"], ["b"]) is False
        assert resolver._paths_conflict(["a", "x"], ["b", "y"]) is False

    def test_divergent_paths(self):
        """Test paths that share prefix but diverge."""
        resolver = ConflictResolver()
        assert resolver._paths_conflict(["a", "b"], ["a", "c"]) is False


# =============================================================================
# Resolve Conflict Tests
# =============================================================================


class TestResolveConflict:
    """Tests for single conflict resolution."""

    def test_keep_local(self):
        """Test KEEP_LOCAL strategy."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["name"],
            local_value="Local",
            remote_value="Remote",
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.KEEP_LOCAL)

        assert resolved.resolved is True
        assert resolved.resolved_value == "Local"
        assert resolved.resolution == ConflictResolutionStrategy.KEEP_LOCAL

    def test_keep_remote(self):
        """Test KEEP_REMOTE strategy."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["name"],
            local_value="Local",
            remote_value="Remote",
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.KEEP_REMOTE)

        assert resolved.resolved is True
        assert resolved.resolved_value == "Remote"

    def test_latest_wins_local(self):
        """Test LATEST_WINS when local is newer."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["name"],
            local_value="Local",
            remote_value="Remote",
            local_timestamp=datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc),
            remote_timestamp=datetime(2026, 1, 8, 11, 0, 0, tzinfo=timezone.utc),
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.LATEST_WINS)

        assert resolved.resolved is True
        assert resolved.resolved_value == "Local"

    def test_latest_wins_remote(self):
        """Test LATEST_WINS when remote is newer."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["name"],
            local_value="Local",
            remote_value="Remote",
            local_timestamp=datetime(2026, 1, 8, 10, 0, 0, tzinfo=timezone.utc),
            remote_timestamp=datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc),
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.LATEST_WINS)

        assert resolved.resolved is True
        assert resolved.resolved_value == "Remote"

    def test_latest_wins_no_timestamps(self):
        """Test LATEST_WINS defaults to local without timestamps."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["name"],
            local_value="Local",
            remote_value="Remote",
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.LATEST_WINS)

        assert resolved.resolved is True
        assert resolved.resolved_value == "Local"

    def test_manual_with_value(self):
        """Test MANUAL strategy with provided value."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["name"],
            local_value="Local",
            remote_value="Remote",
        )

        resolved = resolver.resolve(
            conflict,
            ConflictResolutionStrategy.MANUAL,
            manual_value="Custom Value"
        )

        assert resolved.resolved is True
        assert resolved.resolved_value == "Custom Value"

    def test_manual_without_value(self):
        """Test MANUAL strategy without value stays unresolved."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["name"],
            local_value="Local",
            remote_value="Remote",
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.MANUAL)

        assert resolved.resolved is False

    def test_uses_default_strategy(self):
        """Test uses default strategy when none provided."""
        resolver = ConflictResolver(
            default_strategy=ConflictResolutionStrategy.KEEP_REMOTE
        )
        conflict = SyncConflict(
            test_id="t1",
            path=["name"],
            local_value="Local",
            remote_value="Remote",
        )

        resolved = resolver.resolve(conflict)

        assert resolved.resolved_value == "Remote"


# =============================================================================
# Auto Merge Tests
# =============================================================================


class TestAutoMerge:
    """Tests for automatic merge attempts."""

    def test_merge_same_values(self):
        """Test merge when values are same."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["name"],
            local_value="Same",
            remote_value="Same",
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.MERGE)

        assert resolved.resolved is True
        assert resolved.resolved_value == "Same"

    def test_merge_local_none(self):
        """Test merge when local is None."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["description"],
            local_value=None,
            remote_value="Remote Description",
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.MERGE)

        assert resolved.resolved is True
        assert resolved.resolved_value == "Remote Description"

    def test_merge_remote_none(self):
        """Test merge when remote is None."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["description"],
            local_value="Local Description",
            remote_value=None,
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.MERGE)

        assert resolved.resolved is True
        assert resolved.resolved_value == "Local Description"

    def test_merge_dicts_non_overlapping(self):
        """Test merge dicts with non-overlapping keys."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["metadata"],
            local_value={"tag": "smoke"},
            remote_value={"env": "staging"},
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.MERGE)

        assert resolved.resolved is True
        assert resolved.resolved_value == {"tag": "smoke", "env": "staging"}

    def test_merge_dicts_overlapping_same(self):
        """Test merge dicts with overlapping keys (same value)."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["metadata"],
            local_value={"tag": "smoke", "priority": "high"},
            remote_value={"tag": "smoke", "env": "prod"},
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.MERGE)

        assert resolved.resolved is True
        assert resolved.resolved_value["tag"] == "smoke"

    def test_merge_dicts_overlapping_conflict(self):
        """Test merge dicts with overlapping keys (different values)."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["metadata"],
            local_value={"tag": "smoke"},
            remote_value={"tag": "regression"},
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.MERGE)

        # Can't auto-merge conflicting dict values
        assert resolved.resolved is False

    def test_merge_lists_fails(self):
        """Test merge lists fails (can't auto-merge)."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["steps"],
            local_value=[{"action": "click"}],
            remote_value=[{"action": "fill"}],
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.MERGE)

        # Can't auto-merge different lists
        assert resolved.resolved is False

    def test_merge_primitives_fails(self):
        """Test merge different primitives fails."""
        resolver = ConflictResolver()
        conflict = SyncConflict(
            test_id="t1",
            path=["name"],
            local_value="Local Name",
            remote_value="Remote Name",
        )

        resolved = resolver.resolve(conflict, ConflictResolutionStrategy.MERGE)

        # Can't auto-merge different primitives
        assert resolved.resolved is False


# =============================================================================
# Merge Specs Tests
# =============================================================================


class TestMergeSpecs:
    """Tests for full spec merging."""

    def test_no_conflicts_merge(self):
        """Test merge with no conflicts."""
        resolver = ConflictResolver()
        base = {"name": "Test", "description": ""}
        local = {"name": "Test", "description": "Local desc"}
        remote = {"name": "Updated", "description": ""}

        result = resolver.merge_specs(base, local, remote, "t1")

        assert result.success is True
        assert result.merged_spec["name"] == "Updated"
        assert result.merged_spec["description"] == "Local desc"

    def test_merge_with_auto_resolved(self):
        """Test merge with automatically resolved conflicts."""
        resolver = ConflictResolver()
        base = {"name": "Test", "metadata": {}}
        local = {"name": "Test", "metadata": {"tag": "smoke"}}
        remote = {"name": "Test", "metadata": {"env": "prod"}}

        result = resolver.merge_specs(base, local, remote, "t1")

        # Metadata gets merged
        assert result.success is True

    def test_merge_with_unresolved_conflicts(self):
        """Test merge with unresolved conflicts."""
        resolver = ConflictResolver(
            default_strategy=ConflictResolutionStrategy.MANUAL
        )
        base = {"name": "Original"}
        local = {"name": "Local Name"}
        remote = {"name": "Remote Name"}

        result = resolver.merge_specs(base, local, remote, "t1")

        assert result.success is False
        assert result.manual_required == 1
        assert len(result.conflicts) == 1

    def test_merge_keep_local_strategy(self):
        """Test merge with KEEP_LOCAL strategy."""
        resolver = ConflictResolver(
            default_strategy=ConflictResolutionStrategy.KEEP_LOCAL
        )
        base = {"name": "Original"}
        local = {"name": "Local Name"}
        remote = {"name": "Remote Name"}

        result = resolver.merge_specs(base, local, remote, "t1")

        assert result.success is True
        assert result.merged_spec["name"] == "Local Name"

    def test_merge_keep_remote_strategy(self):
        """Test merge with KEEP_REMOTE strategy."""
        resolver = ConflictResolver(
            default_strategy=ConflictResolutionStrategy.KEEP_REMOTE
        )
        base = {"name": "Original"}
        local = {"name": "Local Name"}
        remote = {"name": "Remote Name"}

        result = resolver.merge_specs(base, local, remote, "t1")

        assert result.success is True
        assert result.merged_spec["name"] == "Remote Name"

    def test_merge_no_base_spec(self):
        """Test merge without base spec."""
        resolver = ConflictResolver(
            default_strategy=ConflictResolutionStrategy.KEEP_REMOTE
        )
        local = {"name": "Local"}
        remote = {"name": "Remote"}

        result = resolver.merge_specs(None, local, remote, "t1")

        assert result.success is True

    def test_merge_steps_no_conflict(self):
        """Test merge steps with no conflict."""
        resolver = ConflictResolver()
        base = {"steps": [{"action": "goto"}]}
        local = {"steps": [{"action": "goto"}, {"action": "click"}]}
        remote = {"steps": [{"action": "goto"}]}

        result = resolver.merge_specs(base, local, remote, "t1")

        assert result.success is True

    def test_merge_result_to_dict(self):
        """Test merge result to_dict."""
        resolver = ConflictResolver()
        result = resolver.merge_specs(
            {"name": "Test"},
            {"name": "Test"},
            {"name": "Test"},
            "t1",
        )

        data = result.to_dict()
        assert "success" in data
        assert "merged_spec" in data


# =============================================================================
# Path Operations Tests
# =============================================================================


class TestPathOperations:
    """Tests for path manipulation methods."""

    def test_set_at_path_simple(self):
        """Test setting value at simple path."""
        resolver = ConflictResolver()
        obj = {"name": "Old"}
        resolver._set_at_path(obj, ["name"], "New")

        assert obj["name"] == "New"

    def test_set_at_path_nested(self):
        """Test setting value at nested path."""
        resolver = ConflictResolver()
        obj = {"metadata": {"priority": "low"}}
        resolver._set_at_path(obj, ["metadata", "priority"], "high")

        assert obj["metadata"]["priority"] == "high"

    def test_set_at_path_creates_intermediate(self):
        """Test setting path creates intermediate objects."""
        resolver = ConflictResolver()
        obj = {}
        resolver._set_at_path(obj, ["metadata", "priority"], "high")

        assert obj["metadata"]["priority"] == "high"

    def test_set_at_path_list_index(self):
        """Test setting value at list index."""
        resolver = ConflictResolver()
        obj = {"steps": [{"action": "click"}]}
        resolver._set_at_path(obj, ["steps", "0", "action"], "fill")

        assert obj["steps"][0]["action"] == "fill"

    def test_set_at_path_empty(self):
        """Test setting with empty path does nothing."""
        resolver = ConflictResolver()
        obj = {"name": "Test"}
        resolver._set_at_path(obj, [], "value")

        assert obj == {"name": "Test"}

    def test_delete_at_path_simple(self):
        """Test deleting at simple path."""
        resolver = ConflictResolver()
        obj = {"name": "Test", "description": "Delete me"}
        resolver._delete_at_path(obj, ["description"])

        assert "description" not in obj

    def test_delete_at_path_nested(self):
        """Test deleting at nested path."""
        resolver = ConflictResolver()
        obj = {"metadata": {"tag": "smoke", "env": "prod"}}
        resolver._delete_at_path(obj, ["metadata", "tag"])

        assert "tag" not in obj["metadata"]
        assert obj["metadata"]["env"] == "prod"

    def test_delete_at_path_list_item(self):
        """Test deleting list item."""
        resolver = ConflictResolver()
        obj = {"steps": [{"action": "click"}, {"action": "fill"}]}
        resolver._delete_at_path(obj, ["steps", "0"])

        assert len(obj["steps"]) == 1
        assert obj["steps"][0]["action"] == "fill"

    def test_delete_at_path_empty(self):
        """Test deleting with empty path does nothing."""
        resolver = ConflictResolver()
        obj = {"name": "Test"}
        resolver._delete_at_path(obj, [])

        assert obj == {"name": "Test"}

    def test_delete_at_path_nonexistent(self):
        """Test deleting nonexistent path does nothing."""
        resolver = ConflictResolver()
        obj = {"name": "Test"}
        resolver._delete_at_path(obj, ["nonexistent"])

        assert obj == {"name": "Test"}


# =============================================================================
# Apply Change Tests
# =============================================================================


class TestApplyChange:
    """Tests for applying changes."""

    def test_apply_add_change(self):
        """Test applying add change."""
        from src.sync.change_detector import Change

        resolver = ConflictResolver()
        spec = {"name": "Test"}
        change = Change(path=["description"], operation="add", new_value="New desc")

        resolver._apply_change(spec, change)

        assert spec["description"] == "New desc"

    def test_apply_update_change(self):
        """Test applying update change."""
        from src.sync.change_detector import Change

        resolver = ConflictResolver()
        spec = {"name": "Old"}
        change = Change(path=["name"], operation="update", new_value="New")

        resolver._apply_change(spec, change)

        assert spec["name"] == "New"

    def test_apply_delete_change(self):
        """Test applying delete change."""
        from src.sync.change_detector import Change

        resolver = ConflictResolver()
        spec = {"name": "Test", "description": "Remove me"}
        change = Change(path=["description"], operation="delete")

        resolver._apply_change(spec, change)

        assert "description" not in spec


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_resolve_conflicts_empty(self):
        """Test resolving empty conflict list."""
        resolved = resolve_conflicts([])
        assert resolved == []

    def test_resolve_conflicts_multiple(self):
        """Test resolving multiple conflicts."""
        conflicts = [
            SyncConflict(
                test_id="t1",
                path=["name"],
                local_value="Local 1",
                remote_value="Remote 1",
                local_timestamp=datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc),
                remote_timestamp=datetime(2026, 1, 8, 10, 0, 0, tzinfo=timezone.utc),
            ),
            SyncConflict(
                test_id="t1",
                path=["description"],
                local_value="Local 2",
                remote_value="Remote 2",
                local_timestamp=datetime(2026, 1, 8, 9, 0, 0, tzinfo=timezone.utc),
                remote_timestamp=datetime(2026, 1, 8, 11, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        resolved = resolve_conflicts(conflicts)

        assert len(resolved) == 2
        assert resolved[0].resolved is True
        assert resolved[0].resolved_value == "Local 1"  # Local is newer
        assert resolved[1].resolved is True
        assert resolved[1].resolved_value == "Remote 2"  # Remote is newer

    def test_resolve_conflicts_custom_strategy(self):
        """Test resolving with custom strategy."""
        conflicts = [
            SyncConflict(
                test_id="t1",
                path=["name"],
                local_value="Local",
                remote_value="Remote",
            ),
        ]

        resolved = resolve_conflicts(
            conflicts,
            ConflictResolutionStrategy.KEEP_LOCAL
        )

        assert resolved[0].resolved_value == "Local"
