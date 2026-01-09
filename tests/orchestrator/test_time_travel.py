"""Tests for time travel API."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime


class TestTimeTravelModels:
    """Tests for time travel request/response models."""

    def test_state_snapshot_model(self):
        """Should create StateSnapshot with required fields."""
        from src.api.time_travel import StateSnapshot

        snapshot = StateSnapshot(
            checkpoint_id="cp-123",
            thread_id="thread-456",
            created_at="2024-01-01T00:00:00Z",
            state_summary={"iteration": 1, "passed_count": 2},
        )

        assert snapshot.checkpoint_id == "cp-123"
        assert snapshot.thread_id == "thread-456"
        assert snapshot.state_summary["iteration"] == 1

    def test_state_snapshot_optional_fields(self):
        """Should handle optional fields correctly."""
        from src.api.time_travel import StateSnapshot

        snapshot = StateSnapshot(
            checkpoint_id="cp-123",
            thread_id="thread-456",
            created_at="2024-01-01T00:00:00Z",
            parent_checkpoint_id="cp-122",
            next_node="execute_test",
            state_summary={},
        )

        assert snapshot.parent_checkpoint_id == "cp-122"
        assert snapshot.next_node == "execute_test"

    def test_replay_request_model(self):
        """Should create ReplayRequest with required fields."""
        from src.api.time_travel import ReplayRequest

        request = ReplayRequest(
            thread_id="thread-123",
            checkpoint_id="cp-456",
        )

        assert request.thread_id == "thread-123"
        assert request.checkpoint_id == "cp-456"
        assert request.new_thread_id is None

    def test_fork_request_model(self):
        """Should create ForkRequest with required fields."""
        from src.api.time_travel import ForkRequest

        request = ForkRequest(
            thread_id="thread-123",
            checkpoint_id="cp-456",
            new_thread_id="forked-thread",
        )

        assert request.thread_id == "thread-123"
        assert request.new_thread_id == "forked-thread"


class TestTimeTravelAPI:
    """Tests for time travel endpoints."""

    @pytest.mark.asyncio
    async def test_get_state_history_returns_snapshots(self):
        """Should return list of state snapshots."""
        from src.api.time_travel import get_state_history
        from datetime import datetime

        mock_state = MagicMock()
        mock_state.config = {"configurable": {"checkpoint_id": "cp-1"}}
        mock_state.parent_config = {"configurable": {"checkpoint_id": "cp-0"}}
        mock_state.next = ["execute_test"]
        mock_state.values = {
            "iteration": 1,
            "passed_count": 2,
            "failed_count": 0,
            "current_test_index": 0,
            "error": None,
            "should_continue": True,
            "healing_attempts": 0,
        }
        # Provide proper created_at attribute
        mock_state.created_at = None  # Will use default datetime.now()

        # Create a proper async generator class
        class AsyncHistoryGenerator:
            def __init__(self, states):
                self.states = states
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.states):
                    raise StopAsyncIteration
                state = self.states[self.index]
                self.index += 1
                return state

        mock_app = MagicMock()
        mock_app.aget_state_history = MagicMock(return_value=AsyncHistoryGenerator([mock_state]))

        with patch("src.api.time_travel.get_checkpointer"):
            with patch("src.api.time_travel.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                with patch("src.api.time_travel.create_testing_graph") as mock_graph:
                    mock_graph.return_value.compile.return_value = mock_app

                    # Call with explicit None for before_checkpoint
                    response = await get_state_history("test-thread", limit=50, before_checkpoint=None)

                    assert response.thread_id == "test-thread"
                    assert len(response.snapshots) > 0
                    assert response.snapshots[0].checkpoint_id == "cp-1"

    @pytest.mark.asyncio
    async def test_get_state_history_empty_thread(self):
        """Should return empty list for thread with no history."""
        from src.api.time_travel import get_state_history

        async def mock_empty_history(*args, **kwargs):
            return
            yield  # Make it an async generator

        with patch("src.api.time_travel.get_checkpointer"):
            with patch("src.api.time_travel.create_testing_graph") as mock_graph:
                with patch("src.api.time_travel.get_settings") as mock_settings:
                    mock_settings.return_value = MagicMock()
                    mock_app = AsyncMock()
                    mock_app.aget_state_history = mock_empty_history
                    mock_graph.return_value.compile.return_value = mock_app

                    response = await get_state_history("empty-thread")

                    assert response.thread_id == "empty-thread"
                    assert len(response.snapshots) == 0

    @pytest.mark.asyncio
    async def test_get_state_at_checkpoint_found(self):
        """Should return full state at checkpoint."""
        from src.api.time_travel import get_state_at_checkpoint

        mock_state = MagicMock()
        mock_state.values = {
            "iteration": 5,
            "passed_count": 3,
            "failed_count": 1,
        }
        mock_state.next = ["report"]
        mock_state.metadata = {"step": 5}

        with patch("src.api.time_travel.get_checkpointer"):
            with patch("src.api.time_travel.create_testing_graph") as mock_graph:
                with patch("src.api.time_travel.get_settings") as mock_settings:
                    mock_settings.return_value = MagicMock()
                    mock_app = AsyncMock()
                    mock_app.aget_state = AsyncMock(return_value=mock_state)
                    mock_graph.return_value.compile.return_value = mock_app

                    response = await get_state_at_checkpoint("test-thread", "cp-123")

                    assert response.checkpoint_id == "cp-123"
                    assert response.thread_id == "test-thread"
                    assert response.next_node == "report"
                    assert response.state["iteration"] == 5

    @pytest.mark.asyncio
    async def test_get_state_at_checkpoint_not_found(self):
        """Should raise 404 for missing checkpoint."""
        from src.api.time_travel import get_state_at_checkpoint
        from fastapi import HTTPException

        with patch("src.api.time_travel.get_checkpointer"):
            with patch("src.api.time_travel.create_testing_graph") as mock_graph:
                with patch("src.api.time_travel.get_settings") as mock_settings:
                    mock_settings.return_value = MagicMock()
                    mock_app = AsyncMock()
                    mock_app.aget_state = AsyncMock(return_value=None)
                    mock_graph.return_value.compile.return_value = mock_app

                    with pytest.raises(HTTPException) as exc_info:
                        await get_state_at_checkpoint("test-thread", "missing-cp")

                    assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_fork_creates_new_thread(self):
        """Should create new thread from checkpoint."""
        from src.api.time_travel import fork_from_checkpoint, ForkRequest

        mock_state = MagicMock()
        mock_state.values = {"iteration": 1, "passed_count": 0}

        with patch("src.api.time_travel.get_checkpointer"):
            with patch("src.api.time_travel.create_testing_graph") as mock_graph:
                with patch("src.api.time_travel.get_settings") as mock_settings:
                    mock_settings.return_value = MagicMock()
                    mock_app = AsyncMock()
                    mock_app.aget_state = AsyncMock(return_value=mock_state)
                    mock_app.aupdate_state = AsyncMock()
                    mock_graph.return_value.compile.return_value = mock_app

                    request = ForkRequest(
                        thread_id="original-thread",
                        checkpoint_id="cp-1",
                        new_thread_id="forked-thread",
                    )

                    response = await fork_from_checkpoint(request)

                    assert response.success is True
                    assert response.new_thread_id == "forked-thread"
                    assert response.source_thread == "original-thread"

    @pytest.mark.asyncio
    async def test_fork_with_modifications(self):
        """Should apply modifications to forked state."""
        from src.api.time_travel import fork_from_checkpoint, ForkRequest

        mock_state = MagicMock()
        mock_state.values = {
            "iteration": 1,
            "passed_count": 0,
            "should_continue": True,
        }

        with patch("src.api.time_travel.get_checkpointer"):
            with patch("src.api.time_travel.create_testing_graph") as mock_graph:
                with patch("src.api.time_travel.get_settings") as mock_settings:
                    mock_settings.return_value = MagicMock()
                    mock_app = AsyncMock()
                    mock_app.aget_state = AsyncMock(return_value=mock_state)
                    mock_app.aupdate_state = AsyncMock()
                    mock_graph.return_value.compile.return_value = mock_app

                    request = ForkRequest(
                        thread_id="original-thread",
                        checkpoint_id="cp-1",
                        new_thread_id="forked-thread",
                        state_modifications={"should_continue": False},
                    )

                    response = await fork_from_checkpoint(request)

                    assert response.success is True
                    assert "should_continue" in response.modifications_applied

    @pytest.mark.asyncio
    async def test_fork_missing_checkpoint(self):
        """Should raise 404 for missing source checkpoint."""
        from src.api.time_travel import fork_from_checkpoint, ForkRequest
        from fastapi import HTTPException

        with patch("src.api.time_travel.get_checkpointer"):
            with patch("src.api.time_travel.create_testing_graph") as mock_graph:
                with patch("src.api.time_travel.get_settings") as mock_settings:
                    mock_settings.return_value = MagicMock()
                    mock_app = AsyncMock()
                    mock_app.aget_state = AsyncMock(return_value=None)
                    mock_graph.return_value.compile.return_value = mock_app

                    request = ForkRequest(
                        thread_id="original-thread",
                        checkpoint_id="missing-cp",
                        new_thread_id="forked-thread",
                    )

                    with pytest.raises(HTTPException) as exc_info:
                        await fork_from_checkpoint(request)

                    assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_replay_from_checkpoint(self):
        """Should replay execution from checkpoint."""
        from src.api.time_travel import replay_from_checkpoint, ReplayRequest

        mock_state = MagicMock()
        mock_state.values = {"iteration": 1, "passed_count": 0}

        final_state = {
            "iteration": 5,
            "passed_count": 3,
            "failed_count": 1,
            "skipped_count": 0,
            "error": None,
        }

        with patch("src.api.time_travel.get_checkpointer"):
            with patch("src.api.time_travel.create_testing_graph") as mock_graph:
                with patch("src.api.time_travel.get_settings") as mock_settings:
                    mock_settings.return_value = MagicMock()
                    mock_app = AsyncMock()
                    mock_app.aget_state = AsyncMock(return_value=mock_state)
                    mock_app.aupdate_state = AsyncMock()
                    mock_app.ainvoke = AsyncMock(return_value=final_state)
                    mock_graph.return_value.compile.return_value = mock_app

                    request = ReplayRequest(
                        thread_id="test-thread",
                        checkpoint_id="cp-1",
                    )

                    response = await replay_from_checkpoint(request)

                    assert response.success is True
                    assert response.source_checkpoint == "cp-1"
                    assert response.final_state_summary["passed_count"] == 3

    @pytest.mark.asyncio
    async def test_replay_to_new_thread(self):
        """Should replay to a new thread if specified."""
        from src.api.time_travel import replay_from_checkpoint, ReplayRequest

        mock_state = MagicMock()
        mock_state.values = {"iteration": 1}

        final_state = {"iteration": 5, "passed_count": 2, "failed_count": 0, "skipped_count": 0}

        with patch("src.api.time_travel.get_checkpointer"):
            with patch("src.api.time_travel.create_testing_graph") as mock_graph:
                with patch("src.api.time_travel.get_settings") as mock_settings:
                    mock_settings.return_value = MagicMock()
                    mock_app = AsyncMock()
                    mock_app.aget_state = AsyncMock(return_value=mock_state)
                    mock_app.aupdate_state = AsyncMock()
                    mock_app.ainvoke = AsyncMock(return_value=final_state)
                    mock_graph.return_value.compile.return_value = mock_app

                    request = ReplayRequest(
                        thread_id="original-thread",
                        checkpoint_id="cp-1",
                        new_thread_id="replay-thread",
                    )

                    response = await replay_from_checkpoint(request)

                    assert response.success is True
                    assert response.target_thread_id == "replay-thread"

    @pytest.mark.asyncio
    async def test_compare_states(self):
        """Should compare states between threads."""
        from src.api.time_travel import compare_states

        mock_state_1 = MagicMock()
        mock_state_1.values = {
            "iteration": 5,
            "passed_count": 3,
            "failed_count": 1,
        }

        mock_state_2 = MagicMock()
        mock_state_2.values = {
            "iteration": 5,
            "passed_count": 4,
            "failed_count": 0,
        }

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(side_effect=[mock_state_1, mock_state_2])

        with patch("src.api.time_travel.get_checkpointer"):
            with patch("src.api.time_travel.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                with patch("src.api.time_travel.create_testing_graph") as mock_graph:
                    mock_graph.return_value.compile.return_value = mock_app

                    # Pass explicit None values for optional parameters
                    response = await compare_states(
                        thread_id_1="thread-1",
                        thread_id_2="thread-2",
                        checkpoint_id_1=None,
                        checkpoint_id_2=None,
                    )

                    assert response.thread_1 == "thread-1"
                    assert response.thread_2 == "thread-2"
                    assert response.difference_count > 0
                    assert "passed_count" in response.differences
                    assert "failed_count" in response.differences

    @pytest.mark.asyncio
    async def test_list_threads(self):
        """Should list all threads."""
        from src.api.time_travel import list_threads
        from langgraph.checkpoint.memory import MemorySaver

        mock_checkpointer = MemorySaver()
        mock_app = AsyncMock()

        with patch("src.api.time_travel.get_checkpointer", return_value=mock_checkpointer):
            with patch("src.api.time_travel.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                with patch("src.api.time_travel.create_testing_graph") as mock_graph:
                    mock_graph.return_value.compile.return_value = mock_app

                    # Pass explicit limit and status values for direct function call
                    response = await list_threads(limit=50, status=None)

                    assert "threads" in response
                    assert "total" in response
                    assert isinstance(response["threads"], list)

    @pytest.mark.asyncio
    async def test_delete_thread_not_found(self):
        """Should raise 404 for missing thread."""
        from src.api.time_travel import delete_thread_history
        from fastapi import HTTPException
        from langgraph.checkpoint.memory import MemorySaver

        mock_checkpointer = MemorySaver()

        with patch("src.api.time_travel.get_checkpointer", return_value=mock_checkpointer):
            with pytest.raises(HTTPException) as exc_info:
                await delete_thread_history("nonexistent-thread")

            assert exc_info.value.status_code == 404


class TestTimeTravelIntegration:
    """Integration tests for time travel functionality."""

    @pytest.mark.asyncio
    async def test_history_and_fork_workflow(self):
        """Should support full history -> fork workflow."""
        from src.api.time_travel import (
            get_state_history,
            fork_from_checkpoint,
            ForkRequest,
        )

        # Mock multiple history states
        mock_states = []
        for i in range(4):
            state = MagicMock()
            state.config = {"configurable": {"checkpoint_id": f"cp-{i}"}}
            state.parent_config = {"configurable": {"checkpoint_id": f"cp-{i-1}"}} if i > 0 else None
            state.next = ["execute_test"] if i < 3 else None
            state.values = {
                "iteration": i,
                "passed_count": i,
                "failed_count": 0,
                "current_test_index": i,
                "error": None,
                "should_continue": True,
                "healing_attempts": 0,
            }
            state.created_at = None  # Will use default datetime.now()
            mock_states.append(state)

        # Create a proper async generator class
        class AsyncHistoryGenerator:
            def __init__(self, states):
                self.states = states
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.states):
                    raise StopAsyncIteration
                state = self.states[self.index]
                self.index += 1
                return state

        mock_app = MagicMock()
        mock_app.aget_state_history = MagicMock(return_value=AsyncHistoryGenerator(mock_states))
        mock_app.aget_state = AsyncMock(return_value=mock_states[2])
        mock_app.aupdate_state = AsyncMock()

        with patch("src.api.time_travel.get_checkpointer"):
            with patch("src.api.time_travel.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                with patch("src.api.time_travel.create_testing_graph") as mock_graph:
                    mock_graph.return_value.compile.return_value = mock_app

                    # Get history
                    history = await get_state_history("test-thread", limit=50, before_checkpoint=None)
                    assert len(history.snapshots) == 4

                    # Fork from checkpoint 2
                    fork_request = ForkRequest(
                        thread_id="test-thread",
                        checkpoint_id="cp-2",
                        new_thread_id="forked-thread",
                    )

                    fork_response = await fork_from_checkpoint(fork_request)
                    assert fork_response.success is True
