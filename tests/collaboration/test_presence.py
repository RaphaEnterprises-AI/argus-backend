"""Tests for presence manager."""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta

from src.collaboration.presence import PresenceManager
from src.collaboration.models import (
    PresenceStatus,
    CursorPosition,
    SelectionRange,
    BroadcastMessage,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def presence_manager():
    """Create a fresh presence manager."""
    return PresenceManager()


@pytest.fixture
def presence_manager_with_broadcast():
    """Create presence manager with broadcast tracking."""
    broadcasts = []

    def capture_broadcast(msg: BroadcastMessage):
        broadcasts.append(msg)

    manager = PresenceManager(broadcast_fn=capture_broadcast)
    manager._broadcasts = broadcasts
    return manager


# =============================================================================
# User Join/Leave Tests
# =============================================================================


class TestUserJoinLeave:
    """Tests for user join and leave operations."""

    @pytest.mark.asyncio
    async def test_user_join(self, presence_manager):
        """Test user joining workspace."""
        presence = await presence_manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        assert presence.user_id == "user-1"
        assert presence.user_name == "Test User"
        assert presence.status == PresenceStatus.ONLINE
        assert presence.color  # Color should be assigned

    @pytest.mark.asyncio
    async def test_user_join_with_test(self, presence_manager):
        """Test user joining with specific test."""
        presence = await presence_manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1",
            test_id="test-1"
        )

        assert presence.test_id == "test-1"

        # Should be in test presence too
        test_presence = presence_manager.get_test_presence("test-1")
        assert len(test_presence) == 1

    @pytest.mark.asyncio
    async def test_user_leave(self, presence_manager):
        """Test user leaving workspace."""
        await presence_manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        await presence_manager.user_leave("user-1", "ws-1")

        workspace_presence = presence_manager.get_workspace_presence("ws-1")
        assert len(workspace_presence) == 0

    @pytest.mark.asyncio
    async def test_user_leave_cleans_test(self, presence_manager):
        """Test leave also removes from test tracking."""
        await presence_manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1",
            test_id="test-1"
        )

        await presence_manager.user_leave("user-1", "ws-1", "test-1")

        test_presence = presence_manager.get_test_presence("test-1")
        assert len(test_presence) == 0

    @pytest.mark.asyncio
    async def test_multiple_users_join(self, presence_manager):
        """Test multiple users in workspace."""
        await presence_manager.user_join(
            user_id="user-1",
            user_name="User 1",
            user_email="user1@example.com",
            workspace_id="ws-1"
        )
        await presence_manager.user_join(
            user_id="user-2",
            user_name="User 2",
            user_email="user2@example.com",
            workspace_id="ws-1"
        )

        workspace_presence = presence_manager.get_workspace_presence("ws-1")
        assert len(workspace_presence) == 2


# =============================================================================
# Cursor Update Tests
# =============================================================================


class TestCursorUpdates:
    """Tests for cursor position updates."""

    @pytest.mark.asyncio
    async def test_update_cursor(self, presence_manager):
        """Test updating cursor position."""
        await presence_manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        cursor = CursorPosition(x=100, y=200, element_id="step-1")
        await presence_manager.update_cursor("user-1", "ws-1", cursor)

        presence = presence_manager.get_user_presence("user-1", "ws-1")
        assert presence.cursor.x == 100
        assert presence.cursor.y == 200

    @pytest.mark.asyncio
    async def test_cursor_updates_activity(self, presence_manager):
        """Test cursor update refreshes last_active."""
        await presence_manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        presence = presence_manager.get_user_presence("user-1", "ws-1")
        old_active = presence.last_active

        # Small delay
        await asyncio.sleep(0.01)

        cursor = CursorPosition(x=50, y=50)
        await presence_manager.update_cursor("user-1", "ws-1", cursor)

        presence = presence_manager.get_user_presence("user-1", "ws-1")
        assert presence.last_active > old_active

    @pytest.mark.asyncio
    async def test_cursor_reverts_idle(self, presence_manager):
        """Test cursor update reverts idle status."""
        await presence_manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        await presence_manager.set_status("user-1", "ws-1", PresenceStatus.IDLE)

        cursor = CursorPosition(x=50, y=50)
        await presence_manager.update_cursor("user-1", "ws-1", cursor)

        presence = presence_manager.get_user_presence("user-1", "ws-1")
        assert presence.status == PresenceStatus.ONLINE


# =============================================================================
# Selection Update Tests
# =============================================================================


class TestSelectionUpdates:
    """Tests for selection range updates."""

    @pytest.mark.asyncio
    async def test_update_selection(self, presence_manager):
        """Test updating text selection."""
        await presence_manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        selection = SelectionRange(start=10, end=25, element_id="input-1")
        await presence_manager.update_selection("user-1", "ws-1", selection)

        presence = presence_manager.get_user_presence("user-1", "ws-1")
        assert presence.selection.start == 10
        assert presence.selection.end == 25


# =============================================================================
# Status Tests
# =============================================================================


class TestStatusUpdates:
    """Tests for status updates."""

    @pytest.mark.asyncio
    async def test_set_status(self, presence_manager):
        """Test setting user status."""
        await presence_manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        await presence_manager.set_status("user-1", "ws-1", PresenceStatus.BUSY)

        presence = presence_manager.get_user_presence("user-1", "ws-1")
        assert presence.status == PresenceStatus.BUSY

    @pytest.mark.asyncio
    async def test_update_activity(self, presence_manager):
        """Test activity update."""
        await presence_manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        await presence_manager.set_status("user-1", "ws-1", PresenceStatus.IDLE)
        await presence_manager.update_activity("user-1", "ws-1")

        presence = presence_manager.get_user_presence("user-1", "ws-1")
        assert presence.status == PresenceStatus.ONLINE


# =============================================================================
# Broadcast Tests
# =============================================================================


class TestBroadcasts:
    """Tests for broadcast functionality."""

    @pytest.mark.asyncio
    async def test_join_broadcasts(self, presence_manager_with_broadcast):
        """Test join creates broadcast."""
        manager = presence_manager_with_broadcast

        await manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        assert len(manager._broadcasts) == 1
        assert manager._broadcasts[0].event == "user_joined"

    @pytest.mark.asyncio
    async def test_leave_broadcasts(self, presence_manager_with_broadcast):
        """Test leave creates broadcast."""
        manager = presence_manager_with_broadcast

        await manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        await manager.user_leave("user-1", "ws-1")

        assert len(manager._broadcasts) == 2
        assert manager._broadcasts[1].event == "user_left"

    @pytest.mark.asyncio
    async def test_cursor_broadcasts(self, presence_manager_with_broadcast):
        """Test cursor update creates broadcast."""
        manager = presence_manager_with_broadcast

        await manager.user_join(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        cursor = CursorPosition(x=100, y=200)
        await manager.update_cursor("user-1", "ws-1", cursor)

        assert any(b.event == "cursor_move" for b in manager._broadcasts)


# =============================================================================
# Query Tests
# =============================================================================


class TestQueries:
    """Tests for presence queries."""

    @pytest.mark.asyncio
    async def test_get_workspace_presence(self, presence_manager):
        """Test getting workspace presence."""
        await presence_manager.user_join(
            user_id="user-1",
            user_name="User 1",
            user_email="user1@example.com",
            workspace_id="ws-1"
        )
        await presence_manager.user_join(
            user_id="user-2",
            user_name="User 2",
            user_email="user2@example.com",
            workspace_id="ws-1"
        )
        await presence_manager.user_join(
            user_id="user-3",
            user_name="User 3",
            user_email="user3@example.com",
            workspace_id="ws-2"
        )

        ws1_presence = presence_manager.get_workspace_presence("ws-1")
        ws2_presence = presence_manager.get_workspace_presence("ws-2")

        assert len(ws1_presence) == 2
        assert len(ws2_presence) == 1

    @pytest.mark.asyncio
    async def test_get_test_presence(self, presence_manager):
        """Test getting test-specific presence."""
        await presence_manager.user_join(
            user_id="user-1",
            user_name="User 1",
            user_email="user1@example.com",
            workspace_id="ws-1",
            test_id="test-1"
        )
        await presence_manager.user_join(
            user_id="user-2",
            user_name="User 2",
            user_email="user2@example.com",
            workspace_id="ws-1"  # No test_id
        )

        test_presence = presence_manager.get_test_presence("test-1")
        assert len(test_presence) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_user(self, presence_manager):
        """Test getting nonexistent user returns None."""
        presence = presence_manager.get_user_presence("nonexistent", "ws-1")
        assert presence is None

    @pytest.mark.asyncio
    async def test_get_empty_workspace(self, presence_manager):
        """Test getting empty workspace returns empty list."""
        presence = presence_manager.get_workspace_presence("empty-ws")
        assert presence == []
