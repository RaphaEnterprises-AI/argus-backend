"""Tests for Time Travel API endpoints (src/api/time_travel.py).

Covers:
- Get checkpoints
- Get state history
- Get state at checkpoint
- Replay from checkpoint
- Fork from checkpoint
- Compare states
- List threads
- Delete thread history
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_checkpointer():
    """Create a mock checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


@pytest.fixture
def mock_state():
    """Create a mock graph state."""
    state = MagicMock()
    state.config = {"configurable": {"checkpoint_id": "cp-123"}}
    state.parent_config = {"configurable": {"checkpoint_id": "cp-122"}}
    state.next = ["execute_test"]
    state.values = {
        "iteration": 5,
        "passed_count": 10,
        "failed_count": 2,
        "current_test_index": 12,
        "error": None,
        "should_continue": True,
        "healing_attempts": 0,
    }
    state.metadata = {"created_at": datetime.now().isoformat()}
    return state


@pytest.fixture
def mock_compiled_app(mock_state):
    """Create a mock compiled LangGraph app."""
    app = AsyncMock()
    app.aget_state = AsyncMock(return_value=mock_state)
    app.aupdate_state = AsyncMock()
    app.ainvoke = AsyncMock(return_value={
        "iteration": 10,
        "passed_count": 15,
        "failed_count": 2,
        "skipped_count": 0,
        "error": None,
    })

    async def mock_history(*args, **kwargs):
        for i in range(5):
            state = MagicMock()
            state.config = {"configurable": {"checkpoint_id": f"cp-{i}"}}
            state.parent_config = {"configurable": {"checkpoint_id": f"cp-{i-1}"}} if i > 0 else None
            state.next = ["execute_test"] if i < 4 else None
            state.values = {
                "iteration": i,
                "passed_count": i,
                "failed_count": 0,
                "current_test_index": i,
                "error": None,
                "should_continue": True,
                "healing_attempts": 0,
            }
            state.created_at = datetime.now()
            yield state

    app.aget_state_history = mock_history
    return app


# ============================================================================
# Get Checkpoints Tests
# ============================================================================


class TestGetCheckpoints:
    """Tests for GET /api/v1/time-travel/checkpoints endpoint."""

    @pytest.mark.asyncio
    async def test_get_checkpoints_success(self, mock_env_vars, mock_compiled_app):
        """Test getting checkpoints successfully."""
        from src.api.time_travel import get_checkpoints

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_compiled_app

            response = await get_checkpoints(thread_id="thread-123", limit=10)

            assert response.thread_id == "thread-123"
            assert len(response.checkpoints) == 5

    @pytest.mark.asyncio
    async def test_get_checkpoints_with_limit(self, mock_env_vars, mock_compiled_app):
        """Test getting checkpoints with limit."""
        from src.api.time_travel import get_checkpoints

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_compiled_app

            response = await get_checkpoints(thread_id="thread-123", limit=3)

            assert len(response.checkpoints) <= 3

    @pytest.mark.asyncio
    async def test_get_checkpoints_error(self, mock_env_vars):
        """Test getting checkpoints handles errors."""
        from src.api.time_travel import get_checkpoints

        mock_app = AsyncMock()
        # Make aget_state_history raise an exception when iterated
        async def raise_error(*args, **kwargs):
            raise Exception("Graph error")
            yield  # Make it a generator

        mock_app.aget_state_history = raise_error

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_app

            with pytest.raises(HTTPException) as exc_info:
                await get_checkpoints(thread_id="thread-123")

            assert exc_info.value.status_code == 500


# ============================================================================
# Get State History Tests
# ============================================================================


class TestGetStateHistory:
    """Tests for GET /api/v1/time-travel/history/{thread_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_history_success(self, mock_env_vars):
        """Test getting state history successfully."""
        from src.api.time_travel import get_state_history

        # Create a proper async generator for state history
        async def mock_history(config):
            for i in range(5):
                state = MagicMock()
                state.config = {"configurable": {"checkpoint_id": f"cp-{i}"}}
                state.parent_config = {"configurable": {"checkpoint_id": f"cp-{i-1}"}} if i > 0 else None
                state.next = ["execute_test"] if i < 4 else None
                state.values = {
                    "iteration": i,
                    "passed_count": i,
                    "failed_count": 0,
                    "current_test_index": i,
                    "error": None,
                    "should_continue": True,
                    "healing_attempts": 0,
                }
                state.created_at = datetime.now()
                yield state

        mock_app = MagicMock()
        mock_app.aget_state_history = mock_history

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_app

            # Pass actual values instead of Query objects
            response = await get_state_history(thread_id="thread-123", limit=50, before_checkpoint=None)

            assert response.thread_id == "thread-123"
            # The mock generates 5 states
            assert len(response.snapshots) == 5

    @pytest.mark.asyncio
    async def test_get_history_with_pagination(self, mock_env_vars, mock_compiled_app):
        """Test getting state history with pagination."""
        from src.api.time_travel import get_state_history

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_compiled_app

            response = await get_state_history(
                thread_id="thread-123",
                limit=10,
                before_checkpoint="cp-2"
            )

            assert response.thread_id == "thread-123"


# ============================================================================
# Get State at Checkpoint Tests
# ============================================================================


class TestGetStateAtCheckpoint:
    """Tests for GET /api/v1/time-travel/state/{thread_id}/{checkpoint_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_state_success(self, mock_env_vars, mock_compiled_app, mock_state):
        """Test getting state at checkpoint successfully."""
        from src.api.time_travel import get_state_at_checkpoint

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_compiled_app

            response = await get_state_at_checkpoint(
                thread_id="thread-123",
                checkpoint_id="cp-123"
            )

            assert response.thread_id == "thread-123"
            assert response.checkpoint_id == "cp-123"
            assert "iteration" in response.state

    @pytest.mark.asyncio
    async def test_get_state_not_found(self, mock_env_vars):
        """Test getting state when checkpoint not found."""
        from src.api.time_travel import get_state_at_checkpoint

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=None)

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_app

            with pytest.raises(HTTPException) as exc_info:
                await get_state_at_checkpoint(
                    thread_id="thread-123",
                    checkpoint_id="nonexistent"
                )

            assert exc_info.value.status_code == 404


# ============================================================================
# Replay from Checkpoint Tests
# ============================================================================


class TestReplayFromCheckpoint:
    """Tests for POST /api/v1/time-travel/replay endpoint."""

    @pytest.mark.asyncio
    async def test_replay_success(self, mock_env_vars, mock_compiled_app):
        """Test replaying from checkpoint successfully."""
        from src.api.time_travel import ReplayRequest, replay_from_checkpoint

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_compiled_app

            request = ReplayRequest(
                thread_id="thread-123",
                checkpoint_id="cp-123"
            )
            response = await replay_from_checkpoint(request)

            assert response.success is True
            assert response.source_checkpoint == "cp-123"
            assert response.target_thread_id == "thread-123"

    @pytest.mark.asyncio
    async def test_replay_to_new_thread(self, mock_env_vars, mock_compiled_app):
        """Test replaying to a new thread."""
        from src.api.time_travel import ReplayRequest, replay_from_checkpoint

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_compiled_app

            request = ReplayRequest(
                thread_id="thread-123",
                checkpoint_id="cp-123",
                new_thread_id="thread-456"
            )
            response = await replay_from_checkpoint(request)

            assert response.success is True
            assert response.target_thread_id == "thread-456"

    @pytest.mark.asyncio
    async def test_replay_checkpoint_not_found(self, mock_env_vars):
        """Test replay when checkpoint not found."""
        from src.api.time_travel import ReplayRequest, replay_from_checkpoint

        mock_app = AsyncMock()
        mock_state = MagicMock()
        mock_state.values = None  # No values = not found
        mock_app.aget_state = AsyncMock(return_value=mock_state)

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_app

            request = ReplayRequest(
                thread_id="thread-123",
                checkpoint_id="nonexistent"
            )

            with pytest.raises(HTTPException) as exc_info:
                await replay_from_checkpoint(request)

            assert exc_info.value.status_code == 404


# ============================================================================
# Fork from Checkpoint Tests
# ============================================================================


class TestForkFromCheckpoint:
    """Tests for POST /api/v1/time-travel/fork endpoint."""

    @pytest.mark.asyncio
    async def test_fork_success(self, mock_env_vars, mock_compiled_app):
        """Test forking from checkpoint successfully."""
        from src.api.time_travel import ForkRequest, fork_from_checkpoint

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_compiled_app

            request = ForkRequest(
                thread_id="thread-123",
                checkpoint_id="cp-123",
                new_thread_id="thread-456"
            )
            response = await fork_from_checkpoint(request)

            assert response.success is True
            assert response.new_thread_id == "thread-456"
            assert response.source_checkpoint == "cp-123"

    @pytest.mark.asyncio
    async def test_fork_with_modifications(self, mock_env_vars, mock_compiled_app):
        """Test forking with state modifications."""
        from src.api.time_travel import ForkRequest, fork_from_checkpoint

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_compiled_app

            request = ForkRequest(
                thread_id="thread-123",
                checkpoint_id="cp-123",
                new_thread_id="thread-456",
                state_modifications={"iteration": 10, "should_continue": False}
            )
            response = await fork_from_checkpoint(request)

            assert response.success is True
            assert "iteration" in response.modifications_applied
            assert "should_continue" in response.modifications_applied

    @pytest.mark.asyncio
    async def test_fork_checkpoint_not_found(self, mock_env_vars):
        """Test fork when checkpoint not found."""
        from src.api.time_travel import ForkRequest, fork_from_checkpoint

        mock_app = AsyncMock()
        mock_state = MagicMock()
        mock_state.values = None
        mock_app.aget_state = AsyncMock(return_value=mock_state)

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_app

            request = ForkRequest(
                thread_id="thread-123",
                checkpoint_id="nonexistent",
                new_thread_id="thread-456"
            )

            with pytest.raises(HTTPException) as exc_info:
                await fork_from_checkpoint(request)

            assert exc_info.value.status_code == 404


# ============================================================================
# Compare States Tests
# ============================================================================


class TestCompareStates:
    """Tests for GET /api/v1/time-travel/compare/{thread_id_1}/{thread_id_2} endpoint."""

    @pytest.mark.asyncio
    async def test_compare_success(self, mock_env_vars):
        """Test comparing states successfully."""
        from src.api.time_travel import compare_states

        # Create different states for comparison
        state1 = MagicMock()
        state1.values = {"iteration": 5, "passed_count": 10, "failed_count": 2}

        state2 = MagicMock()
        state2.values = {"iteration": 8, "passed_count": 15, "failed_count": 1}

        call_count = [0]

        async def mock_get_state(config):
            call_count[0] += 1
            if call_count[0] == 1:
                return state1
            return state2

        mock_app = AsyncMock()
        mock_app.aget_state = mock_get_state

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_app

            # Pass actual values for checkpoint_id params
            response = await compare_states(
                thread_id_1="thread-1",
                thread_id_2="thread-2",
                checkpoint_id_1=None,
                checkpoint_id_2=None
            )

            assert response.thread_1 == "thread-1"
            assert response.thread_2 == "thread-2"
            assert response.difference_count > 0
            assert "iteration" in response.differences

    @pytest.mark.asyncio
    async def test_compare_identical_states(self, mock_env_vars, mock_state):
        """Test comparing identical states."""
        from src.api.time_travel import compare_states

        async def mock_get_state(config):
            return mock_state

        mock_app = AsyncMock()
        mock_app.aget_state = mock_get_state

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_app

            # Pass actual values for checkpoint_id params
            response = await compare_states(
                thread_id_1="thread-1",
                thread_id_2="thread-2",
                checkpoint_id_1=None,
                checkpoint_id_2=None
            )

            assert response.difference_count == 0

    @pytest.mark.asyncio
    async def test_compare_thread_not_found(self, mock_env_vars):
        """Test compare when thread not found."""
        from src.api.time_travel import compare_states

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=None)

        with patch("src.api.time_travel.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_app

            with pytest.raises(HTTPException) as exc_info:
                await compare_states(
                    thread_id_1="thread-1",
                    thread_id_2="thread-2"
                )

            assert exc_info.value.status_code == 404


# ============================================================================
# List Threads Tests
# ============================================================================


class TestListThreads:
    """Tests for GET /api/v1/time-travel/threads endpoint."""

    @pytest.mark.asyncio
    async def test_list_threads_memory_saver(self, mock_env_vars, mock_state):
        """Test listing threads with MemorySaver."""
        from langgraph.checkpoint.memory import MemorySaver
        from src.api.time_travel import list_threads

        # Create a MemorySaver with some data
        memory_saver = MemorySaver()
        memory_saver.storage = {"thread-1": {}, "thread-2": {}}

        async def mock_get_state(config):
            return mock_state

        mock_app = MagicMock()
        mock_app.aget_state = mock_get_state

        with patch("src.api.time_travel.get_checkpointer", return_value=memory_saver), \
             patch("src.api.time_travel.get_settings", return_value=MagicMock()), \
             patch("src.api.time_travel.create_testing_graph") as mock_graph:

            mock_graph.return_value.compile.return_value = mock_app

            # Pass actual values instead of Query objects
            response = await list_threads(limit=50, status=None)

            assert "threads" in response
            assert "total" in response


# ============================================================================
# Delete Thread History Tests
# ============================================================================


class TestDeleteThreadHistory:
    """Tests for DELETE /api/v1/time-travel/thread/{thread_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_thread_memory_saver(self, mock_env_vars):
        """Test deleting thread with MemorySaver."""
        from langgraph.checkpoint.memory import MemorySaver
        from src.api.time_travel import delete_thread_history

        memory_saver = MemorySaver()
        memory_saver.storage = {"thread-123": {"checkpoints": []}}

        with patch("src.api.time_travel.get_checkpointer", return_value=memory_saver):
            response = await delete_thread_history("thread-123")

            assert response["success"] is True
            assert "thread-123" not in memory_saver.storage

    @pytest.mark.asyncio
    async def test_delete_thread_not_found(self, mock_env_vars):
        """Test deleting non-existent thread."""
        from langgraph.checkpoint.memory import MemorySaver
        from src.api.time_travel import delete_thread_history

        memory_saver = MemorySaver()
        memory_saver.storage = {}

        with patch("src.api.time_travel.get_checkpointer", return_value=memory_saver):
            with pytest.raises(HTTPException) as exc_info:
                await delete_thread_history("nonexistent")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_thread_unsupported_checkpointer(self, mock_env_vars):
        """Test deleting with unsupported checkpointer type."""
        from src.api.time_travel import delete_thread_history

        mock_checkpointer = MagicMock()  # Not a MemorySaver

        with patch("src.api.time_travel.get_checkpointer", return_value=mock_checkpointer):
            with pytest.raises(HTTPException) as exc_info:
                await delete_thread_history("thread-123")

            assert exc_info.value.status_code == 501


# ============================================================================
# Model Validation Tests
# ============================================================================


class TestRequestModels:
    """Tests for request model validation."""

    def test_replay_request(self, mock_env_vars):
        """Test ReplayRequest validation."""
        from src.api.time_travel import ReplayRequest

        request = ReplayRequest(
            thread_id="thread-123",
            checkpoint_id="cp-456"
        )

        assert request.thread_id == "thread-123"
        assert request.new_thread_id is None

    def test_fork_request(self, mock_env_vars):
        """Test ForkRequest validation."""
        from src.api.time_travel import ForkRequest

        request = ForkRequest(
            thread_id="thread-123",
            checkpoint_id="cp-456",
            new_thread_id="thread-789",
            state_modifications={"key": "value"}
        )

        assert request.new_thread_id == "thread-789"
        assert request.state_modifications == {"key": "value"}


class TestResponseModels:
    """Tests for response model validation."""

    def test_checkpoint_info(self, mock_env_vars):
        """Test CheckpointInfo model."""
        from src.api.time_travel import CheckpointInfo

        info = CheckpointInfo(
            checkpoint_id="cp-123",
            parent_checkpoint_id="cp-122",
            created_at="2024-01-15T12:00:00Z",
            next_node="execute_test"
        )

        assert info.checkpoint_id == "cp-123"
        assert info.parent_checkpoint_id == "cp-122"

    def test_state_snapshot(self, mock_env_vars):
        """Test StateSnapshot model."""
        from src.api.time_travel import StateSnapshot

        snapshot = StateSnapshot(
            checkpoint_id="cp-123",
            thread_id="thread-456",
            created_at="2024-01-15T12:00:00Z",
            next_node="execute_test",
            state_summary={"iteration": 5, "passed_count": 10}
        )

        assert snapshot.thread_id == "thread-456"
        assert snapshot.state_summary["iteration"] == 5

    def test_replay_response(self, mock_env_vars):
        """Test ReplayResponse model."""
        from src.api.time_travel import ReplayResponse

        response = ReplayResponse(
            success=True,
            source_checkpoint="cp-123",
            target_thread_id="thread-456",
            final_state_summary={"passed": 15, "failed": 2}
        )

        assert response.success is True
        assert response.target_thread_id == "thread-456"

    def test_fork_response(self, mock_env_vars):
        """Test ForkResponse model."""
        from src.api.time_travel import ForkResponse

        response = ForkResponse(
            success=True,
            source_thread="thread-123",
            source_checkpoint="cp-456",
            new_thread_id="thread-789",
            modifications_applied=["iteration", "should_continue"]
        )

        assert len(response.modifications_applied) == 2

    def test_compare_states_response(self, mock_env_vars):
        """Test CompareStatesResponse model."""
        from src.api.time_travel import CompareStatesResponse

        response = CompareStatesResponse(
            thread_1="thread-1",
            thread_2="thread-2",
            checkpoint_1="cp-1",
            checkpoint_2="cp-2",
            differences={"iteration": {"thread_1": "5", "thread_2": "8"}},
            difference_count=1
        )

        assert response.difference_count == 1
        assert "iteration" in response.differences
