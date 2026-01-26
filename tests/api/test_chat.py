"""Tests for the Chat API module (src/api/chat.py)."""

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
def mock_human_message():
    """Create a mock HumanMessage."""
    from langchain_core.messages import HumanMessage
    return HumanMessage(content="Test message")


@pytest.fixture
def mock_ai_message():
    """Create a mock AIMessage."""
    from langchain_core.messages import AIMessage
    return AIMessage(content="AI response")


@pytest.fixture
def mock_ai_message_with_tool_calls():
    """Create a mock AIMessage with tool calls."""
    from langchain_core.messages import AIMessage
    msg = AIMessage(content="Let me search for that")
    msg.tool_calls = [
        {"id": "call-123", "name": "search", "args": {"query": "test"}}
    ]
    return msg


@pytest.fixture
def mock_tool_message():
    """Create a mock ToolMessage."""
    from langchain_core.messages import ToolMessage
    return ToolMessage(
        content='{"result": "search results"}',
        tool_call_id="call-123",
        name="search"
    )


@pytest.fixture
def mock_graph_app():
    """Create a mock graph application."""
    mock_app = AsyncMock()
    return mock_app


# ============================================================================
# Model Tests
# ============================================================================

class TestRequestModels:
    """Tests for request model validation."""

    def test_chat_message_model(self, mock_env_vars):
        """Test ChatMessage model."""
        from src.api.chat import ChatMessage

        message = ChatMessage(role="user", content="Hello")

        assert message.role == "user"
        assert message.content == "Hello"

    def test_chat_request_model(self, mock_env_vars):
        """Test ChatRequest model."""
        from src.api.chat import ChatMessage, ChatRequest

        request = ChatRequest(
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there!"),
            ],
            thread_id="thread-123",
            app_url="http://localhost:3000",
        )

        assert len(request.messages) == 2
        assert request.thread_id == "thread-123"
        assert request.app_url == "http://localhost:3000"

    def test_chat_request_minimal(self, mock_env_vars):
        """Test ChatRequest with minimal fields."""
        from src.api.chat import ChatMessage, ChatRequest

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Test")]
        )

        assert request.thread_id is None
        assert request.app_url is None


class TestResponseModels:
    """Tests for response model validation."""

    def test_chat_response_model(self, mock_env_vars):
        """Test ChatResponse model."""
        from src.api.chat import ChatResponse

        response = ChatResponse(
            message="AI response",
            thread_id="thread-123",
            tool_calls=[{"id": "call-1", "name": "search", "args": {}}],
        )

        assert response.message == "AI response"
        assert response.thread_id == "thread-123"
        assert len(response.tool_calls) == 1

    def test_chat_response_no_tool_calls(self, mock_env_vars):
        """Test ChatResponse without tool calls."""
        from src.api.chat import ChatResponse

        response = ChatResponse(
            message="Simple response",
            thread_id="thread-123",
        )

        assert response.tool_calls is None


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestSerializeMessageContent:
    """Tests for _serialize_message_content helper."""

    def test_serialize_string_content(self, mock_env_vars):
        """Test serializing string content."""
        from src.api.chat import _serialize_message_content

        result = _serialize_message_content("Hello world")
        assert result == "Hello world"

    def test_serialize_json_string_content(self, mock_env_vars):
        """Test serializing JSON string content."""
        from src.api.chat import _serialize_message_content

        result = _serialize_message_content('{"key": "value"}')
        assert result == {"key": "value"}

    def test_serialize_list_content(self, mock_env_vars):
        """Test serializing list content."""
        from src.api.chat import _serialize_message_content

        content = [{"type": "text", "text": "Hello"}]
        result = _serialize_message_content(content)
        assert result == content

    def test_serialize_dict_content(self, mock_env_vars):
        """Test serializing dict content."""
        from src.api.chat import _serialize_message_content

        content = {"result": "data"}
        result = _serialize_message_content(content)
        assert result == content

    def test_serialize_other_content(self, mock_env_vars):
        """Test serializing other types of content."""
        from src.api.chat import _serialize_message_content

        result = _serialize_message_content(12345)
        assert result == "12345"


class TestSerializeMessage:
    """Tests for _serialize_message helper."""

    def test_serialize_human_message(self, mock_env_vars, mock_human_message):
        """Test serializing HumanMessage."""
        from src.api.chat import _serialize_message

        result = _serialize_message(mock_human_message)

        assert result["role"] == "user"
        assert result["content"] == "Test message"

    def test_serialize_ai_message(self, mock_env_vars, mock_ai_message):
        """Test serializing AIMessage."""
        from src.api.chat import _serialize_message

        result = _serialize_message(mock_ai_message)

        assert result["role"] == "assistant"
        assert result["content"] == "AI response"

    def test_serialize_ai_message_with_tool_calls(
        self, mock_env_vars, mock_ai_message_with_tool_calls
    ):
        """Test serializing AIMessage with tool calls."""
        from src.api.chat import _serialize_message

        result = _serialize_message(mock_ai_message_with_tool_calls)

        assert result["role"] == "assistant"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search"

    def test_serialize_tool_message(self, mock_env_vars, mock_tool_message):
        """Test serializing ToolMessage."""
        from src.api.chat import _serialize_message

        result = _serialize_message(mock_tool_message)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call-123"
        assert result["name"] == "search"
        assert result["content"]["result"] == "search results"

    def test_serialize_unknown_message(self, mock_env_vars):
        """Test serializing unknown message type."""
        from src.api.chat import _serialize_message

        mock_msg = MagicMock()
        mock_msg.content = "Unknown content"

        result = _serialize_message(mock_msg)

        assert result["role"] == "unknown"


# ============================================================================
# Send Message Endpoint Tests
# ============================================================================

class TestSendMessageEndpoint:
    """Tests for POST /api/v1/chat/message endpoint."""

    @pytest.fixture
    def mock_user(self):
        """Create a mock user context."""
        from src.api.security.auth import AuthMethod, UserContext
        return UserContext(
            user_id="test-user-123",
            organization_id="test-org-123",
            auth_method=AuthMethod.JWT,
        )

    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_env_vars, mock_ai_message, mock_user):
        """Test successful message send."""
        from src.api.chat import ChatMessage, ChatRequest, send_message

        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(return_value={
            "messages": [mock_ai_message],
        })

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            thread_id="thread-123",
        )

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            response = await send_message(request, user=mock_user)

            assert response.message == "AI response"
            assert response.thread_id == "thread-123"

    @pytest.mark.asyncio
    async def test_send_message_generates_thread_id(self, mock_env_vars, mock_ai_message, mock_user):
        """Test that thread_id is generated if not provided."""
        from src.api.chat import ChatMessage, ChatRequest, send_message

        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(return_value={
            "messages": [mock_ai_message],
        })

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")],
        )

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            response = await send_message(request, user=mock_user)

            assert response.thread_id is not None
            assert len(response.thread_id) > 0

    @pytest.mark.asyncio
    async def test_send_message_with_tool_calls(
        self, mock_env_vars, mock_ai_message_with_tool_calls, mock_user
    ):
        """Test message response includes tool calls."""
        from src.api.chat import ChatMessage, ChatRequest, send_message

        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(return_value={
            "messages": [mock_ai_message_with_tool_calls],
        })

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Search for something")],
        )

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            response = await send_message(request, user=mock_user)

            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_send_message_no_response(self, mock_env_vars, mock_user):
        """Test message when no AI response is generated."""
        from langchain_core.messages import HumanMessage

        from src.api.chat import ChatMessage, ChatRequest, send_message

        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(return_value={
            "messages": [HumanMessage(content="User message only")],
        })

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")],
        )

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            response = await send_message(request, user=mock_user)

            assert response.message == "No response generated"

    @pytest.mark.asyncio
    async def test_send_message_with_app_url(self, mock_env_vars, mock_ai_message, mock_user):
        """Test message includes app_url in state."""
        from src.api.chat import ChatMessage, ChatRequest, send_message

        mock_app = AsyncMock()
        captured_state = None

        async def capture_state(state, config):
            nonlocal captured_state
            captured_state = state
            return {"messages": [mock_ai_message]}

        mock_app.ainvoke = capture_state

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            app_url="http://localhost:3000",
        )

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            await send_message(request, user=mock_user)

            assert captured_state["app_url"] == "http://localhost:3000"


# ============================================================================
# Stream Message Endpoint Tests
# ============================================================================

class TestStreamMessageEndpoint:
    """Tests for POST /api/v1/chat/stream endpoint."""

    @pytest.fixture
    def mock_user(self):
        """Create a mock user context."""
        from src.api.security.auth import AuthMethod, UserContext
        return UserContext(
            user_id="test-user-123",
            organization_id="test-org-123",
            auth_method=AuthMethod.JWT,
        )

    @pytest.mark.asyncio
    async def test_stream_message_returns_streaming_response(self, mock_env_vars, mock_user):
        """Test that stream endpoint returns StreamingResponse."""
        from fastapi.responses import StreamingResponse

        from src.api.chat import ChatMessage, ChatRequest, stream_message

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")],
        )

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=MagicMock()):
            response = await stream_message(request, user=mock_user)

            assert isinstance(response, StreamingResponse)
            assert response.media_type == "text/plain; charset=utf-8"

    @pytest.mark.asyncio
    async def test_stream_message_headers(self, mock_env_vars, mock_user):
        """Test stream response includes correct headers."""
        from src.api.chat import ChatMessage, ChatRequest, stream_message

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            thread_id="thread-123",
        )

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=MagicMock()):
            response = await stream_message(request, user=mock_user)

            assert "X-Thread-Id" in response.headers
            assert response.headers["X-Thread-Id"] == "thread-123"
            assert response.headers["Cache-Control"] == "no-cache"

    @pytest.mark.asyncio
    async def test_stream_generates_thread_id(self, mock_env_vars, mock_user):
        """Test stream generates thread_id if not provided."""
        from src.api.chat import ChatMessage, ChatRequest, stream_message

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")],
        )

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=MagicMock()):
            response = await stream_message(request, user=mock_user)

            assert "X-Thread-Id" in response.headers
            assert len(response.headers["X-Thread-Id"]) > 0


# ============================================================================
# Chat History Endpoint Tests
# ============================================================================

class TestGetChatHistoryEndpoint:
    """Tests for GET /api/v1/chat/history/{thread_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_chat_history_success(
        self, mock_env_vars, mock_human_message, mock_ai_message
    ):
        """Test successful chat history retrieval."""
        from src.api.chat import get_chat_history

        mock_state = MagicMock()
        mock_state.values = {
            "messages": [mock_human_message, mock_ai_message],
        }

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=mock_state)

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            response = await get_chat_history("thread-123")

            assert response["thread_id"] == "thread-123"
            assert len(response["messages"]) == 2
            assert response["messages"][0]["role"] == "user"
            assert response["messages"][1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_chat_history_empty(self, mock_env_vars):
        """Test chat history when thread has no messages."""
        from src.api.chat import get_chat_history

        mock_state = MagicMock()
        mock_state.values = {"messages": []}

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=mock_state)

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            response = await get_chat_history("thread-123")

            assert response["thread_id"] == "thread-123"
            assert len(response["messages"]) == 0

    @pytest.mark.asyncio
    async def test_get_chat_history_no_state(self, mock_env_vars):
        """Test chat history when no state exists."""
        from src.api.chat import get_chat_history

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=None)

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            response = await get_chat_history("thread-123")

            assert response["thread_id"] == "thread-123"
            assert len(response["messages"]) == 0

    @pytest.mark.asyncio
    async def test_get_chat_history_with_tool_messages(
        self, mock_env_vars, mock_human_message, mock_ai_message_with_tool_calls, mock_tool_message
    ):
        """Test chat history includes tool messages."""
        from src.api.chat import get_chat_history

        mock_state = MagicMock()
        mock_state.values = {
            "messages": [
                mock_human_message,
                mock_ai_message_with_tool_calls,
                mock_tool_message,
            ],
        }

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=mock_state)

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            response = await get_chat_history("thread-123")

            assert len(response["messages"]) == 3
            assert response["messages"][2]["role"] == "tool"

    @pytest.mark.asyncio
    async def test_get_chat_history_error(self, mock_env_vars):
        """Test chat history handles errors."""
        from src.api.chat import get_chat_history

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(side_effect=Exception("Database error"))

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            with pytest.raises(HTTPException) as exc_info:
                await get_chat_history("thread-123")

            assert exc_info.value.status_code == 404


# ============================================================================
# List Threads Endpoint Tests
# ============================================================================

class TestListThreadsEndpoint:
    """Tests for GET /api/v1/chat/threads endpoint."""

    @pytest.mark.asyncio
    async def test_list_threads_returns_message(self, mock_env_vars):
        """Test list threads returns informative message."""
        from src.api.chat import list_threads

        response = await list_threads()

        assert "threads" in response
        assert "message" in response
        assert "PostgresSaver" in response["message"]


# ============================================================================
# Delete Chat History Endpoint Tests
# ============================================================================

class TestDeleteChatHistoryEndpoint:
    """Tests for DELETE /api/v1/chat/history/{thread_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_chat_history(self, mock_env_vars):
        """Test delete chat history returns info about support."""
        from src.api.chat import delete_chat_history

        response = await delete_chat_history("thread-123")

        assert response["thread_id"] == "thread-123"
        assert response["deleted"] is False
        assert "MemorySaver" in response["message"]


# ============================================================================
# Cancel Chat Endpoint Tests
# ============================================================================

class TestCancelChatEndpoint:
    """Tests for DELETE /api/v1/chat/cancel/{thread_id} endpoint."""

    @pytest.mark.asyncio
    async def test_cancel_chat_success(self, mock_env_vars):
        """Test successful chat cancellation."""
        from src.api.chat import cancel_chat

        mock_state = MagicMock()
        mock_state.values = {
            "messages": [],
            "should_continue": True,
        }

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=mock_state)
        mock_app.aupdate_state = AsyncMock()

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            response = await cancel_chat("thread-123")

            assert response["success"] is True
            assert response["thread_id"] == "thread-123"
            assert "cancelled_at" in response

            # Verify state was updated
            mock_app.aupdate_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_chat_not_found(self, mock_env_vars):
        """Test cancel chat when thread not found."""
        from src.api.chat import cancel_chat

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=None)

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            with pytest.raises(HTTPException) as exc_info:
                await cancel_chat("nonexistent")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_chat_no_values(self, mock_env_vars):
        """Test cancel chat when state has no values."""
        from src.api.chat import cancel_chat

        mock_state = MagicMock()
        mock_state.values = None

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=mock_state)

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            with pytest.raises(HTTPException) as exc_info:
                await cancel_chat("thread-123")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_chat_error(self, mock_env_vars):
        """Test cancel chat handles errors."""
        from src.api.chat import cancel_chat

        mock_state = MagicMock()
        mock_state.values = {"messages": []}

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=mock_state)
        mock_app.aupdate_state = AsyncMock(side_effect=Exception("Update failed"))

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            with pytest.raises(HTTPException) as exc_info:
                await cancel_chat("thread-123")

            assert exc_info.value.status_code == 500


# ============================================================================
# Integration-style Tests
# ============================================================================

class TestChatIntegration:
    """Integration-style tests for chat functionality."""

    @pytest.fixture
    def mock_user(self):
        """Create a mock user context."""
        from src.api.security.auth import AuthMethod, UserContext
        return UserContext(
            user_id="test-user-123",
            organization_id="test-org-123",
            auth_method=AuthMethod.JWT,
        )

    @pytest.mark.asyncio
    async def test_full_conversation_flow(
        self, mock_env_vars, mock_human_message, mock_ai_message, mock_user
    ):
        """Test a complete conversation flow."""
        from src.api.chat import ChatMessage, ChatRequest, get_chat_history, send_message

        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(return_value={
            "messages": [mock_human_message, mock_ai_message],
        })

        mock_state = MagicMock()
        mock_state.values = {
            "messages": [mock_human_message, mock_ai_message],
        }
        mock_app.aget_state = AsyncMock(return_value=mock_state)

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            # Send a message
            request = ChatRequest(
                messages=[ChatMessage(role="user", content="Hello")],
                thread_id="thread-123",
            )
            send_response = await send_message(request, user=mock_user)

            assert send_response.message == "AI response"

            # Get history
            history = await get_chat_history("thread-123")

            assert len(history["messages"]) == 2

    @pytest.mark.asyncio
    async def test_message_conversion(self, mock_env_vars, mock_user):
        """Test that messages are correctly converted to LangChain format."""
        from langchain_core.messages import AIMessage, HumanMessage

        from src.api.chat import ChatMessage, ChatRequest, send_message

        captured_state = None

        async def capture_and_respond(state, config):
            nonlocal captured_state
            captured_state = state
            return {"messages": [AIMessage(content="Response")]}

        mock_app = AsyncMock()
        mock_app.ainvoke = capture_and_respond

        mock_graph = MagicMock()
        mock_graph.compile = MagicMock(return_value=mock_app)

        request = ChatRequest(
            messages=[
                ChatMessage(role="user", content="User message"),
                ChatMessage(role="assistant", content="Previous response"),
                ChatMessage(role="user", content="Follow up"),
            ],
        )

        with patch("src.api.chat.get_checkpointer", return_value=MagicMock()), \
             patch("src.api.chat.create_chat_graph", return_value=mock_graph):
            await send_message(request, user=mock_user)

            # Verify messages were converted correctly
            assert len(captured_state["messages"]) == 3
            assert isinstance(captured_state["messages"][0], HumanMessage)
            assert isinstance(captured_state["messages"][1], AIMessage)
            assert isinstance(captured_state["messages"][2], HumanMessage)
