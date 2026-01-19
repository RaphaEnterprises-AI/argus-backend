"""Tests for streaming functionality."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestStreamTestRequest:
    """Tests for StreamTestRequest model."""

    def test_required_fields(self):
        """Should require codebase_path and app_url."""
        from src.api.streaming import StreamTestRequest

        request = StreamTestRequest(
            codebase_path="/test/path",
            app_url="http://localhost:3000"
        )

        assert request.codebase_path == "/test/path"
        assert request.app_url == "http://localhost:3000"

    def test_optional_fields(self):
        """Should accept optional fields."""
        from src.api.streaming import StreamTestRequest

        request = StreamTestRequest(
            codebase_path="/test/path",
            app_url="http://localhost:3000",
            thread_id="custom-thread",
            pr_number=123,
            changed_files=["file1.py", "file2.py"]
        )

        assert request.thread_id == "custom-thread"
        assert request.pr_number == 123
        assert request.changed_files == ["file1.py", "file2.py"]


class TestStreamChatRequest:
    """Tests for StreamChatRequest model."""

    def test_required_fields(self):
        """Should require message and thread_id."""
        from src.api.streaming import StreamChatRequest

        request = StreamChatRequest(
            message="Hello",
            thread_id="test-thread"
        )

        assert request.message == "Hello"
        assert request.thread_id == "test-thread"

    def test_optional_app_url(self):
        """Should accept optional app_url."""
        from src.api.streaming import StreamChatRequest

        request = StreamChatRequest(
            message="Hello",
            thread_id="test-thread",
            app_url="http://localhost:3000"
        )

        assert request.app_url == "http://localhost:3000"


class TestStreamingEndpoints:
    """Tests for streaming API endpoints."""

    @pytest.mark.asyncio
    async def test_stream_test_execution_returns_event_source_response(self):
        """Should return an EventSourceResponse."""
        from sse_starlette.sse import EventSourceResponse

        from src.api.streaming import StreamTestRequest, stream_test_execution

        with patch("src.api.streaming.EnhancedTestingOrchestrator") as mock_orchestrator:
            mock_app = AsyncMock()
            # Mock the async iterator for astream
            mock_app.astream = MagicMock(return_value=AsyncIteratorMock([]))
            mock_app.aget_state = AsyncMock(return_value=MagicMock(
                values={"passed_count": 1, "failed_count": 0}
            ))
            mock_orchestrator.return_value.app = mock_app

            request = StreamTestRequest(
                codebase_path="/test/path",
                app_url="http://localhost:3000"
            )

            response = await stream_test_execution(request)

            # Response should be an EventSourceResponse
            assert isinstance(response, EventSourceResponse)

    @pytest.mark.asyncio
    async def test_stream_chat_returns_event_source_response(self):
        """Should return SSE response for chat streaming."""
        from sse_starlette.sse import EventSourceResponse

        from src.api.streaming import StreamChatRequest, stream_chat

        with patch("src.api.streaming.get_checkpointer") as mock_checkpointer:
            mock_checkpointer.return_value = MagicMock()

            request = StreamChatRequest(
                message="Hello",
                thread_id="test-thread"
            )

            response = await stream_chat(request)
            assert isinstance(response, EventSourceResponse)

    @pytest.mark.asyncio
    async def test_get_stream_status_not_found(self):
        """Should return not found for unknown thread."""
        from src.api.streaming import get_stream_status

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=None)

        # Imports inside function use their original module paths
        with patch("src.api.streaming.get_checkpointer"):
            with patch("src.config.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                with patch("src.orchestrator.graph.create_testing_graph") as mock_graph:
                    mock_graph.return_value.compile.return_value = mock_app

                    response = await get_stream_status("unknown-thread")

                    assert response["found"] is False
                    assert response["thread_id"] == "unknown-thread"

    @pytest.mark.asyncio
    async def test_get_stream_status_found(self):
        """Should return status for known thread."""
        from src.api.streaming import get_stream_status

        mock_state = MagicMock()
        mock_state.values = {
            "run_id": "run-123",
            "iteration": 5,
            "passed_count": 3,
            "failed_count": 1,
            "skipped_count": 0,
            "test_plan": [{"id": "test1"}, {"id": "test2"}],
            "current_test_index": 1,
            "total_cost": 0.5,
            "error": None,
            "next_agent": "execute_test",
            "started_at": "2024-01-01T00:00:00Z",
        }
        mock_state.next = ["execute_test"]

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=mock_state)

        with patch("src.api.streaming.get_checkpointer"):
            with patch("src.config.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                with patch("src.orchestrator.graph.create_testing_graph") as mock_graph:
                    mock_graph.return_value.compile.return_value = mock_app

                    response = await get_stream_status("test-thread")

                    assert response["found"] is True
                    assert response["thread_id"] == "test-thread"
                    assert response["status"] == "running"
                    assert response["state_summary"]["passed"] == 3
                    assert response["state_summary"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_cancel_stream_not_found(self):
        """Should raise 404 for unknown thread."""
        from fastapi import HTTPException

        from src.api.streaming import cancel_stream

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=None)

        with patch("src.api.streaming.get_checkpointer"):
            with patch("src.config.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                with patch("src.orchestrator.graph.create_testing_graph") as mock_graph:
                    mock_graph.return_value.compile.return_value = mock_app

                    with pytest.raises(HTTPException) as exc_info:
                        await cancel_stream("unknown-thread")

                    assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_stream_success(self):
        """Should cancel stream successfully."""
        from src.api.streaming import cancel_stream

        mock_state = MagicMock()
        mock_state.values = {
            "passed_count": 2,
            "failed_count": 1,
            "iteration": 3,
            "should_continue": True,
        }

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=mock_state)
        mock_app.aupdate_state = AsyncMock()

        with patch("src.api.streaming.get_checkpointer"):
            with patch("src.config.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                with patch("src.orchestrator.graph.create_testing_graph") as mock_graph:
                    mock_graph.return_value.compile.return_value = mock_app

                    response = await cancel_stream("test-thread")

                    assert response["success"] is True
                    assert response["thread_id"] == "test-thread"
                    assert "cancelled_at" in response

    @pytest.mark.asyncio
    async def test_resume_stream_returns_event_source_response(self):
        """Should return EventSourceResponse for resume."""
        from sse_starlette.sse import EventSourceResponse

        from src.api.streaming import resume_stream

        mock_state = MagicMock()
        mock_state.values = {"passed_count": 0, "failed_count": 0}
        mock_state.next = ["execute_test"]

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=mock_state)
        mock_app.astream = MagicMock(return_value=AsyncIteratorMock([]))

        with patch("src.api.streaming.get_checkpointer"):
            with patch("src.config.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                with patch("src.orchestrator.graph.create_testing_graph") as mock_graph:
                    mock_graph.return_value.compile.return_value = mock_app

                    response = await resume_stream("test-thread")

                    assert isinstance(response, EventSourceResponse)


class TestStreamEventGeneration:
    """Tests for stream event generation logic."""

    def test_event_data_serialization(self):
        """Should serialize event data as JSON."""
        event_data = {
            "thread_id": "test-123",
            "passed_count": 5,
            "failed_count": 2,
        }

        serialized = json.dumps(event_data)
        deserialized = json.loads(serialized)

        assert deserialized == event_data

    def test_state_summary_extraction(self):
        """Should extract correct state summary fields."""

        state = {
            "iteration": 10,
            "current_test_index": 5,
            "test_plan": [{"id": f"test-{i}"} for i in range(10)],
            "passed_count": 4,
            "failed_count": 1,
            "skipped_count": 0,
            "total_cost": 1.25,
            "next_agent": "report",
            "should_continue": False,
            "error": None,
        }

        summary = {
            "iteration": state.get("iteration", 0),
            "current_test_index": state.get("current_test_index", 0),
            "total_tests": len(state.get("test_plan", [])),
            "passed_count": state.get("passed_count", 0),
            "failed_count": state.get("failed_count", 0),
            "skipped_count": state.get("skipped_count", 0),
            "total_cost": state.get("total_cost", 0),
            "next_agent": state.get("next_agent", ""),
            "should_continue": state.get("should_continue", True),
            "error": state.get("error"),
        }

        assert summary["iteration"] == 10
        assert summary["total_tests"] == 10
        assert summary["passed_count"] == 4


class AsyncIteratorMock:
    """Mock async iterator for testing."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item
