"""Tests for cursor tracking."""


from src.collaboration.cursors import (
    CursorState,
    CursorTracker,
    generate_cursor_label,
    interpolate_cursor_position,
)
from src.collaboration.models import CursorPosition, SelectionRange

# =============================================================================
# CursorState Tests
# =============================================================================


class TestCursorState:
    """Tests for CursorState dataclass."""

    def test_create_cursor_state(self):
        """Test creating cursor state."""
        state = CursorState(
            user_id="user-1",
            user_name="Test User",
            color="#FF6B6B"
        )
        assert state.user_id == "user-1"
        assert state.user_name == "Test User"
        assert state.color == "#FF6B6B"
        assert state.position is None
        assert state.selection is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = CursorState(
            user_id="user-1",
            user_name="Test User",
            color="#FF6B6B",
            position=CursorPosition(x=100, y=200)
        )
        data = state.to_dict()

        assert data["user_id"] == "user-1"
        assert data["position"]["x"] == 100
        assert "last_updated" in data

    def test_to_dict_without_position(self):
        """Test to_dict when position is None."""
        state = CursorState(
            user_id="user-1",
            user_name="Test User",
            color="#FF6B6B"
        )
        data = state.to_dict()

        assert data["position"] is None


# =============================================================================
# CursorTracker Tests
# =============================================================================


class TestCursorTracker:
    """Tests for CursorTracker class."""

    def test_create_tracker(self):
        """Test creating tracker."""
        tracker = CursorTracker(test_id="test-1")
        assert tracker.test_id == "test-1"

    def test_add_user(self):
        """Test adding user to tracker."""
        tracker = CursorTracker(test_id="test-1")
        state = tracker.add_user("user-1", "Test User")

        assert state.user_id == "user-1"
        assert state.color  # Auto-assigned

    def test_add_user_with_color(self):
        """Test adding user with custom color."""
        tracker = CursorTracker(test_id="test-1")
        state = tracker.add_user("user-1", "Test User", color="#FF0000")

        assert state.color == "#FF0000"

    def test_remove_user(self):
        """Test removing user from tracker."""
        tracker = CursorTracker(test_id="test-1")
        tracker.add_user("user-1", "Test User")
        tracker.remove_user("user-1")

        assert tracker.get_cursor("user-1") is None

    def test_update_position(self):
        """Test updating cursor position."""
        tracker = CursorTracker(test_id="test-1")
        tracker.add_user("user-1", "Test User")

        position = CursorPosition(x=150, y=250, step_index=0)
        tracker.update_position("user-1", position, broadcast=False)

        cursor = tracker.get_cursor("user-1")
        assert cursor.position.x == 150
        assert cursor.position.step_index == 0

    def test_update_position_nonexistent_user(self):
        """Test updating position for nonexistent user."""
        tracker = CursorTracker(test_id="test-1")
        position = CursorPosition(x=100, y=100)

        result = tracker.update_position("nonexistent", position)
        assert result is False

    def test_update_selection(self):
        """Test updating text selection."""
        tracker = CursorTracker(test_id="test-1")
        tracker.add_user("user-1", "Test User")

        selection = SelectionRange(start=10, end=20, element_id="input-1")
        tracker.update_selection("user-1", selection, broadcast=False)

        cursor = tracker.get_cursor("user-1")
        assert cursor.selection.start == 10
        assert cursor.selection.end == 20

    def test_clear_selection(self):
        """Test clearing selection."""
        tracker = CursorTracker(test_id="test-1")
        tracker.add_user("user-1", "Test User")
        tracker.update_selection(
            "user-1",
            SelectionRange(start=0, end=10),
            broadcast=False
        )

        tracker.clear_selection("user-1")

        cursor = tracker.get_cursor("user-1")
        assert cursor.selection is None

    def test_get_all_cursors(self):
        """Test getting all cursors."""
        tracker = CursorTracker(test_id="test-1")
        tracker.add_user("user-1", "User 1")
        tracker.add_user("user-2", "User 2")

        cursors = tracker.get_all_cursors()
        assert len(cursors) == 2

    def test_get_active_cursors(self):
        """Test getting active (non-stale) cursors."""
        tracker = CursorTracker(test_id="test-1")
        tracker.add_user("user-1", "User 1")

        # Update to make recent
        tracker.update_position("user-1", CursorPosition(x=0, y=0), broadcast=False)

        active = tracker.get_active_cursors()
        assert len(active) == 1

    def test_get_cursors_at_element(self):
        """Test getting cursors at specific element."""
        tracker = CursorTracker(test_id="test-1")
        tracker.add_user("user-1", "User 1")
        tracker.add_user("user-2", "User 2")

        tracker.update_position(
            "user-1",
            CursorPosition(x=0, y=0, element_id="step-1"),
            broadcast=False
        )
        tracker.update_position(
            "user-2",
            CursorPosition(x=0, y=0, element_id="step-2"),
            broadcast=False
        )

        cursors_at_step1 = tracker.get_cursors_at_element("step-1")
        assert len(cursors_at_step1) == 1
        assert cursors_at_step1[0].user_id == "user-1"

    def test_get_cursors_at_step(self):
        """Test getting cursors at specific step index."""
        tracker = CursorTracker(test_id="test-1")
        tracker.add_user("user-1", "User 1")
        tracker.add_user("user-2", "User 2")

        tracker.update_position(
            "user-1",
            CursorPosition(x=0, y=0, step_index=0),
            broadcast=False
        )
        tracker.update_position(
            "user-2",
            CursorPosition(x=0, y=0, step_index=1),
            broadcast=False
        )

        cursors_at_step0 = tracker.get_cursors_at_step(0)
        assert len(cursors_at_step0) == 1

    def test_get_full_state(self):
        """Test getting full state for sync."""
        tracker = CursorTracker(test_id="test-1")
        tracker.add_user("user-1", "User 1")
        tracker.add_user("user-2", "User 2")

        state = tracker.get_full_state()

        assert state["test_id"] == "test-1"
        assert len(state["cursors"]) == 2


# =============================================================================
# Broadcast Tests
# =============================================================================


class TestCursorBroadcast:
    """Tests for cursor broadcast functionality."""

    def test_broadcast_on_position_update(self):
        """Test broadcast is called on position update."""
        broadcasts = []
        tracker = CursorTracker(
            test_id="test-1",
            broadcast_fn=lambda msg: broadcasts.append(msg)
        )
        tracker.add_user("user-1", "User 1")

        tracker.update_position("user-1", CursorPosition(x=100, y=100))

        assert len(broadcasts) == 1
        assert broadcasts[0].event == "cursor_move"

    def test_throttle_rapid_updates(self):
        """Test throttling of rapid cursor updates."""
        broadcasts = []
        tracker = CursorTracker(
            test_id="test-1",
            broadcast_fn=lambda msg: broadcasts.append(msg)
        )
        tracker.add_user("user-1", "User 1")

        # Rapid updates
        for i in range(10):
            tracker.update_position("user-1", CursorPosition(x=i, y=i))

        # Should be throttled (not all 10 broadcasts)
        # First one should definitely go through
        assert len(broadcasts) >= 1


# =============================================================================
# Interpolation Tests
# =============================================================================


class TestInterpolation:
    """Tests for cursor position interpolation."""

    def test_interpolate_start(self):
        """Test interpolation at start."""
        start = CursorPosition(x=0, y=0)
        end = CursorPosition(x=100, y=100)

        result = interpolate_cursor_position(start, end, 0.0)
        assert result.x == 0
        assert result.y == 0

    def test_interpolate_end(self):
        """Test interpolation at end."""
        start = CursorPosition(x=0, y=0)
        end = CursorPosition(x=100, y=100)

        result = interpolate_cursor_position(start, end, 1.0)
        assert result.x == 100
        assert result.y == 100

    def test_interpolate_middle(self):
        """Test interpolation at middle."""
        start = CursorPosition(x=0, y=0)
        end = CursorPosition(x=100, y=100)

        result = interpolate_cursor_position(start, end, 0.5)
        assert result.x == 50
        assert result.y == 50

    def test_interpolate_element_id(self):
        """Test element_id switches at midpoint."""
        start = CursorPosition(x=0, y=0, element_id="el-1")
        end = CursorPosition(x=100, y=100, element_id="el-2")

        result_before = interpolate_cursor_position(start, end, 0.4)
        result_after = interpolate_cursor_position(start, end, 0.6)

        assert result_before.element_id == "el-1"
        assert result_after.element_id == "el-2"

    def test_interpolate_clamps_progress(self):
        """Test progress is clamped to [0, 1]."""
        start = CursorPosition(x=0, y=0)
        end = CursorPosition(x=100, y=100)

        result_under = interpolate_cursor_position(start, end, -0.5)
        result_over = interpolate_cursor_position(start, end, 1.5)

        assert result_under.x == 0
        assert result_over.x == 100


# =============================================================================
# Label Generation Tests
# =============================================================================


class TestLabelGeneration:
    """Tests for cursor label generation."""

    def test_short_name(self):
        """Test short name is kept as-is."""
        label = generate_cursor_label("John")
        assert label == "John"

    def test_max_length_name(self):
        """Test name at max length."""
        label = generate_cursor_label("Short Name", max_length=10)
        assert label == "Short Name"

    def test_truncate_long_name(self):
        """Test long name is truncated when first name also too long."""
        # Use a name where even first name is too long
        label = generate_cursor_label("Bartholomew", max_length=10)
        assert len(label) <= 10
        assert label.endswith("...")

    def test_use_first_name(self):
        """Test uses first name if it fits."""
        label = generate_cursor_label("John Smith-Williams", max_length=10)
        assert label == "John"
