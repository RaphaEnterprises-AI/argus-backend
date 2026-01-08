"""Tests for real-time collaboration manager."""

import pytest
import asyncio

from src.collaboration.realtime import (
    RealtimeConfig,
    RealtimeSession,
    RealtimeManager,
    create_realtime_manager,
)
from src.collaboration.models import (
    PresenceStatus,
    CursorPosition,
    SelectionRange,
    BroadcastMessage,
)
from src.collaboration.crdt import CRDTOperation


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def realtime_config():
    """Create test configuration."""
    return RealtimeConfig(
        supabase_url="http://localhost:54321",
        supabase_key="test-key"
    )


@pytest.fixture
def realtime_manager(realtime_config):
    """Create real-time manager."""
    return RealtimeManager(realtime_config)


@pytest.fixture
def manager_with_broadcasts():
    """Create manager with broadcast capture."""
    broadcasts = []
    manager = RealtimeManager()
    manager.add_broadcast_handler(lambda msg: broadcasts.append(msg))
    manager._captured_broadcasts = broadcasts
    return manager


# =============================================================================
# Configuration Tests
# =============================================================================


class TestRealtimeConfig:
    """Tests for RealtimeConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RealtimeConfig()
        assert config.workspace_channel_prefix == "workspace"
        assert config.test_channel_prefix == "test"
        assert config.presence_sync_interval_ms == 5000

    def test_custom_config(self):
        """Test custom configuration."""
        config = RealtimeConfig(
            supabase_url="http://example.com",
            max_concurrent_editors=5
        )
        assert config.supabase_url == "http://example.com"
        assert config.max_concurrent_editors == 5


class TestRealtimeSession:
    """Tests for RealtimeSession."""

    def test_create_session(self):
        """Test creating session."""
        session = RealtimeSession(
            user_id="user-1",
            workspace_id="ws-1"
        )
        assert session.user_id == "user-1"
        assert session.workspace_id == "ws-1"
        assert session.test_id is None
        assert session.id  # Auto-generated


# =============================================================================
# Connection Tests
# =============================================================================


class TestConnections:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_connect(self, realtime_manager):
        """Test connecting a user."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        assert session.user_id == "user-1"
        assert session.workspace_id == "ws-1"

    @pytest.mark.asyncio
    async def test_connect_with_test(self, realtime_manager):
        """Test connecting to specific test."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1",
            test_id="test-1"
        )

        assert session.test_id == "test-1"

    @pytest.mark.asyncio
    async def test_disconnect(self, realtime_manager):
        """Test disconnecting a user."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        await realtime_manager.disconnect(session.id)

        users = realtime_manager.get_workspace_users("ws-1")
        assert len(users) == 0

    @pytest.mark.asyncio
    async def test_switch_test(self, realtime_manager):
        """Test switching between tests."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1",
            test_id="test-1"
        )

        await realtime_manager.switch_test(session.id, "test-2")

        test1_users = realtime_manager.get_test_users("test-1")
        test2_users = realtime_manager.get_test_users("test-2")

        assert len(test1_users) == 0
        assert len(test2_users) == 1


# =============================================================================
# Presence Tests
# =============================================================================


class TestPresenceOperations:
    """Tests for presence operations."""

    @pytest.mark.asyncio
    async def test_update_cursor(self, realtime_manager):
        """Test updating cursor position."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        cursor = CursorPosition(x=100, y=200, step_index=0)
        await realtime_manager.update_cursor(session.id, cursor)

        users = realtime_manager.get_workspace_users("ws-1")
        assert users[0].cursor.x == 100

    @pytest.mark.asyncio
    async def test_update_selection(self, realtime_manager):
        """Test updating text selection."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        selection = SelectionRange(start=10, end=20)
        await realtime_manager.update_selection(session.id, selection)

        users = realtime_manager.get_workspace_users("ws-1")
        assert users[0].selection.start == 10

    @pytest.mark.asyncio
    async def test_set_status(self, realtime_manager):
        """Test setting user status."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        await realtime_manager.set_status(session.id, PresenceStatus.BUSY)

        users = realtime_manager.get_workspace_users("ws-1")
        assert users[0].status == PresenceStatus.BUSY

    @pytest.mark.asyncio
    async def test_get_workspace_users(self, realtime_manager):
        """Test getting workspace users."""
        await realtime_manager.connect(
            user_id="user-1",
            user_name="User 1",
            user_email="user1@example.com",
            workspace_id="ws-1"
        )
        await realtime_manager.connect(
            user_id="user-2",
            user_name="User 2",
            user_email="user2@example.com",
            workspace_id="ws-1"
        )

        users = realtime_manager.get_workspace_users("ws-1")
        assert len(users) == 2

    @pytest.mark.asyncio
    async def test_get_test_users(self, realtime_manager):
        """Test getting test users."""
        await realtime_manager.connect(
            user_id="user-1",
            user_name="User 1",
            user_email="user1@example.com",
            workspace_id="ws-1",
            test_id="test-1"
        )
        await realtime_manager.connect(
            user_id="user-2",
            user_name="User 2",
            user_email="user2@example.com",
            workspace_id="ws-1"  # No test
        )

        test_users = realtime_manager.get_test_users("test-1")
        assert len(test_users) == 1


# =============================================================================
# CRDT Tests
# =============================================================================


class TestCRDTOperations:
    """Tests for CRDT operations."""

    @pytest.mark.asyncio
    async def test_load_test_spec(self, realtime_manager):
        """Test loading test spec for editing."""
        test_spec = {
            "id": "test-1",
            "name": "Login Test",
            "steps": [{"action": "goto", "target": "/login"}],
            "assertions": [],
            "metadata": {}
        }

        crdt = await realtime_manager.load_test_spec("test-1", test_spec)

        assert crdt.test_spec["name"] == "Login Test"

    @pytest.mark.asyncio
    async def test_get_test_spec(self, realtime_manager):
        """Test getting test spec."""
        test_spec = {"id": "test-1", "name": "Test", "steps": [], "assertions": [], "metadata": {}}
        await realtime_manager.load_test_spec("test-1", test_spec)

        retrieved = await realtime_manager.get_test_spec("test-1")
        assert retrieved["name"] == "Test"

    @pytest.mark.asyncio
    async def test_get_nonexistent_spec(self, realtime_manager):
        """Test getting nonexistent spec returns None."""
        spec = await realtime_manager.get_test_spec("nonexistent")
        assert spec is None


# =============================================================================
# Comment Tests
# =============================================================================


class TestComments:
    """Tests for comment functionality."""

    @pytest.mark.asyncio
    async def test_add_comment(self, realtime_manager):
        """Test adding a comment."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        comment = await realtime_manager.add_comment(
            session_id=session.id,
            test_id="test-1",
            content="This looks good!"
        )

        assert comment.content == "This looks good!"
        assert comment.author_id == "user-1"

    @pytest.mark.asyncio
    async def test_add_comment_with_step(self, realtime_manager):
        """Test adding comment on specific step."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        comment = await realtime_manager.add_comment(
            session_id=session.id,
            test_id="test-1",
            content="Check this selector",
            step_index=2
        )

        assert comment.step_index == 2

    @pytest.mark.asyncio
    async def test_add_comment_with_mentions(self, realtime_manager):
        """Test adding comment with mentions."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        comment = await realtime_manager.add_comment(
            session_id=session.id,
            test_id="test-1",
            content="@user-2 please review",
            mentions=["user-2"]
        )

        assert "user-2" in comment.mentions

    @pytest.mark.asyncio
    async def test_resolve_comment(self, realtime_manager):
        """Test resolving a comment."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        comment = await realtime_manager.add_comment(
            session_id=session.id,
            test_id="test-1",
            content="Issue to resolve"
        )

        resolved = await realtime_manager.resolve_comment(session.id, comment.id)

        assert resolved
        comments = realtime_manager.get_comments("test-1", include_resolved=True)
        assert comments[0].resolved

    @pytest.mark.asyncio
    async def test_get_comments(self, realtime_manager):
        """Test getting comments."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        await realtime_manager.add_comment(session.id, "test-1", "Comment 1")
        await realtime_manager.add_comment(session.id, "test-1", "Comment 2")

        comments = realtime_manager.get_comments("test-1")
        assert len(comments) == 2

    @pytest.mark.asyncio
    async def test_get_comments_filters_resolved(self, realtime_manager):
        """Test that resolved comments are filtered by default."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        comment = await realtime_manager.add_comment(session.id, "test-1", "To resolve")
        await realtime_manager.add_comment(session.id, "test-1", "Keep open")
        await realtime_manager.resolve_comment(session.id, comment.id)

        comments = realtime_manager.get_comments("test-1")
        assert len(comments) == 1
        assert comments[0].content == "Keep open"

    @pytest.mark.asyncio
    async def test_get_comments_by_step(self, realtime_manager):
        """Test filtering comments by step."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        await realtime_manager.add_comment(session.id, "test-1", "Step 0", step_index=0)
        await realtime_manager.add_comment(session.id, "test-1", "Step 1", step_index=1)
        await realtime_manager.add_comment(session.id, "test-1", "Step 1 again", step_index=1)

        comments = realtime_manager.get_comments("test-1", step_index=1)
        assert len(comments) == 2


# =============================================================================
# State Sync Tests
# =============================================================================


class TestStateSync:
    """Tests for state synchronization."""

    @pytest.mark.asyncio
    async def test_get_full_state(self, realtime_manager):
        """Test getting full state for sync."""
        session = await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1",
            test_id="test-1"
        )

        test_spec = {"id": "test-1", "name": "Test", "steps": [], "assertions": [], "metadata": {}}
        await realtime_manager.load_test_spec("test-1", test_spec)

        await realtime_manager.add_comment(session.id, "test-1", "Test comment")

        state = await realtime_manager.get_full_state(session.id, "test-1")

        assert state["test_id"] == "test-1"
        assert state["test_spec"] is not None
        assert len(state["presence"]) == 1
        assert len(state["comments"]) == 1


# =============================================================================
# Broadcast Tests
# =============================================================================


class TestBroadcasts:
    """Tests for broadcast functionality."""

    @pytest.mark.asyncio
    async def test_add_broadcast_handler(self, realtime_manager):
        """Test adding broadcast handler."""
        messages = []
        realtime_manager.add_broadcast_handler(lambda msg: messages.append(msg))

        await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_remove_broadcast_handler(self, realtime_manager):
        """Test removing broadcast handler."""
        messages = []
        handler = lambda msg: messages.append(msg)

        realtime_manager.add_broadcast_handler(handler)
        realtime_manager.remove_broadcast_handler(handler)

        await realtime_manager.connect(
            user_id="user-1",
            user_name="Test User",
            user_email="test@example.com",
            workspace_id="ws-1"
        )

        assert len(messages) == 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactory:
    """Tests for factory function."""

    def test_create_realtime_manager(self):
        """Test creating manager with factory."""
        manager = create_realtime_manager(
            supabase_url="http://test.supabase.co",
            supabase_key="test-key"
        )

        assert manager.config.supabase_url == "http://test.supabase.co"
        assert manager.config.supabase_key == "test-key"

    def test_create_with_defaults(self):
        """Test creating manager with defaults."""
        manager = create_realtime_manager()

        assert manager.config.supabase_url == ""
        assert manager is not None
