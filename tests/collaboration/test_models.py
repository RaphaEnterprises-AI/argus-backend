"""Tests for collaboration models."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from src.collaboration.models import (
    PresenceStatus,
    CollaborationEventType,
    CursorPosition,
    SelectionRange,
    UserPresence,
    CollaborationEvent,
    CollaborativeComment,
    EditOperation,
    BroadcastMessage,
    CURSOR_COLORS,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestPresenceStatus:
    """Tests for PresenceStatus enum."""

    def test_status_values(self):
        """Test presence status values."""
        assert PresenceStatus.ONLINE.value == "online"
        assert PresenceStatus.IDLE.value == "idle"
        assert PresenceStatus.OFFLINE.value == "offline"
        assert PresenceStatus.BUSY.value == "busy"

    def test_status_from_string(self):
        """Test creating status from string."""
        assert PresenceStatus("online") == PresenceStatus.ONLINE
        assert PresenceStatus("idle") == PresenceStatus.IDLE


class TestCollaborationEventType:
    """Tests for CollaborationEventType enum."""

    def test_presence_events(self):
        """Test presence event types."""
        assert CollaborationEventType.USER_JOINED.value == "user_joined"
        assert CollaborationEventType.USER_LEFT.value == "user_left"
        assert CollaborationEventType.USER_IDLE.value == "user_idle"

    def test_cursor_events(self):
        """Test cursor event types."""
        assert CollaborationEventType.CURSOR_MOVE.value == "cursor_move"
        assert CollaborationEventType.CURSOR_SELECT.value == "cursor_select"

    def test_edit_events(self):
        """Test edit event types."""
        assert CollaborationEventType.EDIT_START.value == "edit_start"
        assert CollaborationEventType.EDIT_CHANGE.value == "edit_change"

    def test_comment_events(self):
        """Test comment event types."""
        assert CollaborationEventType.COMMENT_ADDED.value == "comment_added"
        assert CollaborationEventType.COMMENT_RESOLVED.value == "comment_resolved"


# =============================================================================
# CursorPosition Tests
# =============================================================================


class TestCursorPosition:
    """Tests for CursorPosition dataclass."""

    def test_default_values(self):
        """Test default values."""
        cursor = CursorPosition()
        assert cursor.x == 0
        assert cursor.y == 0
        assert cursor.element_id is None
        assert cursor.step_index is None
        assert cursor.field_name is None

    def test_custom_values(self):
        """Test custom values."""
        cursor = CursorPosition(
            x=100.5,
            y=200.5,
            element_id="step-1",
            step_index=0,
            field_name="target"
        )
        assert cursor.x == 100.5
        assert cursor.y == 200.5
        assert cursor.element_id == "step-1"
        assert cursor.step_index == 0
        assert cursor.field_name == "target"


class TestSelectionRange:
    """Tests for SelectionRange dataclass."""

    def test_default_values(self):
        """Test default values."""
        selection = SelectionRange()
        assert selection.start == 0
        assert selection.end == 0
        assert selection.element_id is None

    def test_custom_values(self):
        """Test custom values."""
        selection = SelectionRange(start=10, end=25, element_id="input-1")
        assert selection.start == 10
        assert selection.end == 25
        assert selection.element_id == "input-1"


# =============================================================================
# UserPresence Tests
# =============================================================================


class TestUserPresence:
    """Tests for UserPresence dataclass."""

    def test_create_presence(self):
        """Test creating user presence."""
        presence = UserPresence(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="workspace-1"
        )
        assert presence.user_id == "user-1"
        assert presence.user_name == "Test User"
        assert presence.status == PresenceStatus.ONLINE

    def test_auto_assign_color(self):
        """Test color is auto-assigned."""
        presence = UserPresence(user_id="user-1", workspace_id="ws-1")
        assert presence.color in CURSOR_COLORS

    def test_deterministic_color(self):
        """Test same user gets same color."""
        presence1 = UserPresence(user_id="user-1", workspace_id="ws-1")
        presence2 = UserPresence(user_id="user-1", workspace_id="ws-2")
        assert presence1.color == presence2.color

    def test_custom_color(self):
        """Test custom color is preserved."""
        presence = UserPresence(user_id="user-1", workspace_id="ws-1", color="#FF0000")
        assert presence.color == "#FF0000"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        presence = UserPresence(
            id="pres-1",
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="workspace-1",
            status=PresenceStatus.ONLINE,
            color="#FF6B6B"
        )
        data = presence.to_dict()

        assert data["id"] == "pres-1"
        assert data["user_id"] == "user-1"
        assert data["user_name"] == "Test User"
        assert data["status"] == "online"
        assert data["color"] == "#FF6B6B"
        assert "last_active" in data

    def test_to_dict_with_cursor(self):
        """Test to_dict includes cursor."""
        presence = UserPresence(
            user_id="user-1",
            workspace_id="ws-1",
            cursor=CursorPosition(x=100, y=200)
        )
        data = presence.to_dict()

        assert data["cursor"]["x"] == 100
        assert data["cursor"]["y"] == 200

    def test_to_dict_with_selection(self):
        """Test to_dict includes selection."""
        presence = UserPresence(
            user_id="user-1",
            workspace_id="ws-1",
            selection=SelectionRange(start=5, end=15)
        )
        data = presence.to_dict()

        assert data["selection"]["start"] == 5
        assert data["selection"]["end"] == 15

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "id": "pres-1",
            "user_id": "user-1",
            "user_name": "Test User",
            "user_email": "test@example.com",
            "workspace_id": "ws-1",
            "status": "idle",
            "color": "#4ECDC4"
        }
        presence = UserPresence.from_dict(data)

        assert presence.id == "pres-1"
        assert presence.user_id == "user-1"
        assert presence.status == PresenceStatus.IDLE
        assert presence.color == "#4ECDC4"

    def test_from_dict_with_cursor(self):
        """Test from_dict parses cursor."""
        data = {
            "user_id": "user-1",
            "workspace_id": "ws-1",
            "cursor": {"x": 50, "y": 100, "element_id": "el-1"}
        }
        presence = UserPresence.from_dict(data)

        assert presence.cursor is not None
        assert presence.cursor.x == 50
        assert presence.cursor.y == 100


# =============================================================================
# CollaborationEvent Tests
# =============================================================================


class TestCollaborationEvent:
    """Tests for CollaborationEvent dataclass."""

    def test_create_event(self):
        """Test creating an event."""
        event = CollaborationEvent(
            type=CollaborationEventType.USER_JOINED,
            user_id="user-1",
            workspace_id="ws-1"
        )
        assert event.type == CollaborationEventType.USER_JOINED
        assert event.user_id == "user-1"
        assert event.workspace_id == "ws-1"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = CollaborationEvent(
            id="evt-1",
            type=CollaborationEventType.CURSOR_MOVE,
            user_id="user-1",
            workspace_id="ws-1",
            payload={"x": 100, "y": 200}
        )
        data = event.to_dict()

        assert data["id"] == "evt-1"
        assert data["type"] == "cursor_move"
        assert data["payload"]["x"] == 100


# =============================================================================
# CollaborativeComment Tests
# =============================================================================


class TestCollaborativeComment:
    """Tests for CollaborativeComment dataclass."""

    def test_create_comment(self):
        """Test creating a comment."""
        comment = CollaborativeComment(
            test_id="test-1",
            author_id="user-1",
            author_name="Test User",
            content="This looks good!"
        )
        assert comment.test_id == "test-1"
        assert comment.content == "This looks good!"
        assert comment.resolved is False

    def test_comment_with_mentions(self):
        """Test comment with mentions."""
        comment = CollaborativeComment(
            test_id="test-1",
            author_id="user-1",
            author_name="User 1",
            content="@user-2 please review",
            mentions=["user-2"]
        )
        assert "user-2" in comment.mentions

    def test_threaded_comment(self):
        """Test reply comment."""
        parent = CollaborativeComment(
            test_id="test-1",
            author_id="user-1",
            author_name="User 1",
            content="Parent comment"
        )
        reply = CollaborativeComment(
            test_id="test-1",
            author_id="user-2",
            author_name="User 2",
            content="Reply comment",
            parent_id=parent.id
        )
        assert reply.parent_id == parent.id

    def test_to_dict(self):
        """Test conversion to dictionary."""
        comment = CollaborativeComment(
            id="cmt-1",
            test_id="test-1",
            step_index=2,
            author_id="user-1",
            author_name="Test User",
            content="Test comment"
        )
        data = comment.to_dict()

        assert data["id"] == "cmt-1"
        assert data["test_id"] == "test-1"
        assert data["step_index"] == 2
        assert data["resolved"] is False


# =============================================================================
# EditOperation Tests
# =============================================================================


class TestEditOperation:
    """Tests for EditOperation dataclass."""

    def test_create_operation(self):
        """Test creating an operation."""
        op = EditOperation(
            user_id="user-1",
            test_id="test-1",
            operation="update",
            path=["steps", "0", "target"],
            value="#new-selector"
        )
        assert op.operation == "update"
        assert op.path == ["steps", "0", "target"]
        assert op.value == "#new-selector"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        op = EditOperation(
            id="op-1",
            user_id="user-1",
            test_id="test-1",
            operation="insert",
            path=["steps"],
            value={"action": "click"}
        )
        data = op.to_dict()

        assert data["id"] == "op-1"
        assert data["operation"] == "insert"
        assert data["value"]["action"] == "click"


# =============================================================================
# BroadcastMessage Tests
# =============================================================================


class TestBroadcastMessage:
    """Tests for BroadcastMessage dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = BroadcastMessage(
            channel="workspace:ws-1",
            event="user_joined",
            payload={"user_id": "user-1"}
        )
        assert msg.channel == "workspace:ws-1"
        assert msg.event == "user_joined"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        msg = BroadcastMessage(
            channel="test:test-1",
            event="edit_change",
            payload={"path": ["steps", "0"]}
        )
        data = msg.to_dict()

        assert data["channel"] == "test:test-1"
        assert data["event"] == "edit_change"


# =============================================================================
# CURSOR_COLORS Tests
# =============================================================================


class TestCursorColors:
    """Tests for cursor colors."""

    def test_colors_count(self):
        """Test there are enough colors."""
        assert len(CURSOR_COLORS) >= 10

    def test_colors_are_hex(self):
        """Test colors are valid hex codes."""
        for color in CURSOR_COLORS:
            assert color.startswith("#")
            assert len(color) == 7
