"""Tests for change detector."""

import pytest

from src.sync.change_detector import (
    Change,
    DiffResult,
    ChangeDetector,
    diff_specs,
    calculate_checksum,
)
from src.sync.models import SyncEventType, SyncSource


# =============================================================================
# Change Tests
# =============================================================================


class TestChange:
    """Tests for Change dataclass."""

    def test_create_change(self):
        """Test creating a change."""
        change = Change(
            path=["steps", "0"],
            operation="add",
            new_value={"action": "click"}
        )
        assert change.path == ["steps", "0"]
        assert change.operation == "add"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        change = Change(
            path=["name"],
            operation="update",
            old_value="Old",
            new_value="New"
        )
        data = change.to_dict()

        assert data["path"] == ["name"]
        assert data["old_value"] == "Old"


# =============================================================================
# DiffResult Tests
# =============================================================================


class TestDiffResult:
    """Tests for DiffResult dataclass."""

    def test_no_changes(self):
        """Test empty diff result."""
        result = DiffResult()
        assert result.has_changes is False
        assert result.change_count == 0

    def test_with_changes(self):
        """Test diff result with changes."""
        result = DiffResult(
            has_changes=True,
            changes=[
                Change(path=["name"], operation="update"),
                Change(path=["steps", "0"], operation="add"),
            ]
        )
        assert result.has_changes is True
        assert result.change_count == 2

    def test_get_changes_by_path(self):
        """Test filtering changes by path."""
        result = DiffResult(
            changes=[
                Change(path=["steps", "0", "target"], operation="update"),
                Change(path=["steps", "1", "target"], operation="update"),
                Change(path=["name"], operation="update"),
            ]
        )

        step_changes = result.get_changes_by_path(["steps"])
        assert len(step_changes) == 2


# =============================================================================
# ChangeDetector Tests
# =============================================================================


class TestChangeDetector:
    """Tests for ChangeDetector class."""

    def test_create_detector(self):
        """Test creating detector."""
        detector = ChangeDetector()
        assert detector is not None

    def test_calculate_checksum(self):
        """Test checksum calculation."""
        spec = {"name": "Test", "steps": []}
        checksum = ChangeDetector.calculate_checksum(spec)

        assert isinstance(checksum, str)
        assert len(checksum) == 16

    def test_checksum_deterministic(self):
        """Test checksum is deterministic."""
        spec = {"name": "Test", "steps": [{"action": "click"}]}

        checksum1 = ChangeDetector.calculate_checksum(spec)
        checksum2 = ChangeDetector.calculate_checksum(spec)

        assert checksum1 == checksum2

    def test_checksum_different_for_different_specs(self):
        """Test different specs have different checksums."""
        spec1 = {"name": "Test 1"}
        spec2 = {"name": "Test 2"}

        assert ChangeDetector.calculate_checksum(spec1) != ChangeDetector.calculate_checksum(spec2)


class TestDiffOperations:
    """Tests for diff operations."""

    def test_diff_no_changes(self):
        """Test diff with no changes."""
        detector = ChangeDetector()
        spec = {"name": "Test", "steps": []}

        result = detector.diff(spec, spec.copy())

        assert result.has_changes is False

    def test_diff_creation(self):
        """Test diff for new spec (creation)."""
        detector = ChangeDetector()
        new_spec = {"name": "New Test", "steps": []}

        result = detector.diff(None, new_spec)

        assert result.has_changes is True
        assert result.changes[0].operation == "add"
        assert result.changes[0].path == []

    def test_diff_deletion(self):
        """Test diff for deleted spec."""
        detector = ChangeDetector()
        old_spec = {"name": "Old Test"}

        result = detector.diff(old_spec, None)

        assert result.has_changes is True
        assert result.changes[0].operation == "delete"

    def test_diff_primitive_change(self):
        """Test diff for primitive value change."""
        detector = ChangeDetector()
        old_spec = {"name": "Old Name"}
        new_spec = {"name": "New Name"}

        result = detector.diff(old_spec, new_spec)

        assert result.has_changes is True
        assert len(result.changes) == 1
        assert result.changes[0].path == ["name"]
        assert result.changes[0].operation == "update"

    def test_diff_nested_change(self):
        """Test diff for nested value change."""
        detector = ChangeDetector()
        old_spec = {"metadata": {"priority": "low"}}
        new_spec = {"metadata": {"priority": "high"}}

        result = detector.diff(old_spec, new_spec)

        assert result.has_changes is True
        assert result.changes[0].path == ["metadata", "priority"]

    def test_diff_key_added(self):
        """Test diff when key is added."""
        detector = ChangeDetector()
        old_spec = {"name": "Test"}
        new_spec = {"name": "Test", "description": "A test"}

        result = detector.diff(old_spec, new_spec)

        assert result.has_changes is True
        assert any(c.operation == "add" and c.path == ["description"] for c in result.changes)

    def test_diff_key_removed(self):
        """Test diff when key is removed."""
        detector = ChangeDetector()
        old_spec = {"name": "Test", "description": "Remove me"}
        new_spec = {"name": "Test"}

        result = detector.diff(old_spec, new_spec)

        assert result.has_changes is True
        assert any(c.operation == "delete" and c.path == ["description"] for c in result.changes)

    def test_diff_list_item_added(self):
        """Test diff when list item is added."""
        detector = ChangeDetector()
        old_spec = {"steps": [{"action": "click"}]}
        new_spec = {"steps": [{"action": "click"}, {"action": "fill"}]}

        result = detector.diff(old_spec, new_spec)

        assert result.has_changes is True
        assert any(c.operation == "add" and c.path == ["steps", "1"] for c in result.changes)

    def test_diff_list_item_removed(self):
        """Test diff when list item is removed."""
        detector = ChangeDetector()
        old_spec = {"steps": [{"action": "click"}, {"action": "fill"}]}
        new_spec = {"steps": [{"action": "click"}]}

        result = detector.diff(old_spec, new_spec)

        assert result.has_changes is True
        assert any(c.operation == "delete" for c in result.changes)

    def test_diff_list_item_changed(self):
        """Test diff when list item is changed."""
        detector = ChangeDetector()
        old_spec = {"steps": [{"action": "click", "target": "#old"}]}
        new_spec = {"steps": [{"action": "click", "target": "#new"}]}

        result = detector.diff(old_spec, new_spec)

        assert result.has_changes is True
        assert any("target" in c.path for c in result.changes)

    def test_diff_skips_updated_at(self):
        """Test diff skips updated_at field."""
        detector = ChangeDetector()
        old_spec = {"name": "Test", "updated_at": "2026-01-01"}
        new_spec = {"name": "Test", "updated_at": "2026-01-02"}

        result = detector.diff(old_spec, new_spec)

        assert result.has_changes is False

    def test_diff_skips_version(self):
        """Test diff skips version field."""
        detector = ChangeDetector()
        old_spec = {"name": "Test", "version": 1}
        new_spec = {"name": "Test", "version": 2}

        result = detector.diff(old_spec, new_spec)

        assert result.has_changes is False


class TestEventGeneration:
    """Tests for event generation from diffs."""

    def test_generate_events_no_changes(self):
        """Test no events for no changes."""
        detector = ChangeDetector()
        result = DiffResult(has_changes=False)

        events = detector.generate_events(result, "proj-1", "test-1")

        assert len(events) == 0

    def test_generate_test_created_event(self):
        """Test event for test creation."""
        detector = ChangeDetector()
        result = DiffResult(
            has_changes=True,
            changes=[Change(path=[], operation="add", new_value={"name": "Test"})]
        )

        events = detector.generate_events(result, "proj-1", "test-1")

        assert len(events) == 1
        assert events[0].type == SyncEventType.TEST_CREATED

    def test_generate_step_added_event(self):
        """Test event for step addition."""
        detector = ChangeDetector()
        result = DiffResult(
            has_changes=True,
            changes=[Change(path=["steps", "0"], operation="add", new_value={"action": "click"})]
        )

        events = detector.generate_events(result, "proj-1", "test-1")

        assert len(events) == 1
        assert events[0].type == SyncEventType.STEP_ADDED

    def test_generate_step_updated_event(self):
        """Test event for step update."""
        detector = ChangeDetector()
        result = DiffResult(
            has_changes=True,
            changes=[Change(path=["steps", "0", "target"], operation="update")]
        )

        events = detector.generate_events(result, "proj-1", "test-1")

        assert events[0].type == SyncEventType.STEP_UPDATED

    def test_generate_metadata_event(self):
        """Test event for metadata change."""
        detector = ChangeDetector()
        result = DiffResult(
            has_changes=True,
            changes=[Change(path=["metadata", "priority"], operation="update")]
        )

        events = detector.generate_events(result, "proj-1", "test-1")

        assert events[0].type == SyncEventType.METADATA_UPDATED


class TestStepReorderDetection:
    """Tests for step reorder detection."""

    def test_detect_reorder_same_ids(self):
        """Test reorder detection with same IDs."""
        detector = ChangeDetector()
        old_steps = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        new_steps = [{"id": "c"}, {"id": "a"}, {"id": "b"}]

        assert detector.detect_step_reorder(old_steps, new_steps) is True

    def test_detect_not_reorder_different_count(self):
        """Test reorder returns false for different counts."""
        detector = ChangeDetector()
        old_steps = [{"id": "a"}, {"id": "b"}]
        new_steps = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

        assert detector.detect_step_reorder(old_steps, new_steps) is False

    def test_detect_not_reorder_different_ids(self):
        """Test reorder returns false for different IDs."""
        detector = ChangeDetector()
        old_steps = [{"id": "a"}, {"id": "b"}]
        new_steps = [{"id": "c"}, {"id": "d"}]

        assert detector.detect_step_reorder(old_steps, new_steps) is False


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_diff_specs(self):
        """Test diff_specs function."""
        old = {"name": "Old"}
        new = {"name": "New"}

        result = diff_specs(old, new)

        assert result.has_changes is True

    def test_calculate_checksum(self):
        """Test calculate_checksum function."""
        spec = {"name": "Test"}
        checksum = calculate_checksum(spec)

        assert isinstance(checksum, str)
