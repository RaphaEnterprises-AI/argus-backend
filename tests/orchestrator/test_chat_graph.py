"""Tests for the chat graph orchestrator module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class TestEstimateTokens:
    """Tests for the estimate_tokens function."""

    def test_estimate_tokens_empty_string(self, mock_env_vars):
        """Test token estimation for empty string."""
        from src.orchestrator.chat_graph import estimate_tokens

        result = estimate_tokens("")
        assert result == 0

    def test_estimate_tokens_short_string(self, mock_env_vars):
        """Test token estimation for short string."""
        from src.orchestrator.chat_graph import estimate_tokens

        # 8 characters / 4 = 2 tokens
        result = estimate_tokens("12345678")
        assert result == 2

    def test_estimate_tokens_medium_string(self, mock_env_vars):
        """Test token estimation for medium string."""
        from src.orchestrator.chat_graph import estimate_tokens

        # 400 characters / 4 = 100 tokens
        result = estimate_tokens("a" * 400)
        assert result == 100

    def test_estimate_tokens_long_string(self, mock_env_vars):
        """Test token estimation for long string."""
        from src.orchestrator.chat_graph import estimate_tokens

        # 10000 characters / 4 = 2500 tokens
        result = estimate_tokens("x" * 10000)
        assert result == 2500


class TestStripBase64FromContent:
    """Tests for the strip_base64_from_content function."""

    def test_strip_data_url_image(self, mock_env_vars):
        """Test stripping data URL images."""
        from src.orchestrator.chat_graph import strip_base64_from_content

        content = 'Some text data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg== more text'
        result = strip_base64_from_content(content)

        assert "[IMAGE_REMOVED]" in result
        assert "iVBORw0KGgo" not in result
        assert "Some text" in result
        assert "more text" in result

    def test_strip_json_screenshot_field(self, mock_env_vars):
        """Test stripping screenshot fields from JSON."""
        from src.orchestrator.chat_graph import strip_base64_from_content

        # Create a long base64 string (>1000 chars)
        long_base64 = "A" * 1500
        content = f'{{"screenshot": "{long_base64}"}}'
        result = strip_base64_from_content(content)

        assert "[IMAGE_REMOVED]" in result
        assert long_base64 not in result

    def test_strip_json_image_field(self, mock_env_vars):
        """Test stripping image fields from JSON."""
        from src.orchestrator.chat_graph import strip_base64_from_content

        long_base64 = "B" * 1500
        content = f'{{"image": "{long_base64}"}}'
        result = strip_base64_from_content(content)

        assert "[IMAGE_REMOVED]" in result
        assert long_base64 not in result

    def test_strip_finalScreenshot_field(self, mock_env_vars):
        """Test stripping finalScreenshot fields."""
        from src.orchestrator.chat_graph import strip_base64_from_content

        long_base64 = "C" * 1500
        content = f'{{"finalScreenshot": "{long_base64}"}}'
        result = strip_base64_from_content(content)

        assert "[IMAGE_REMOVED]" in result

    def test_strip_screenshotBase64_field(self, mock_env_vars):
        """Test stripping screenshotBase64 fields."""
        from src.orchestrator.chat_graph import strip_base64_from_content

        long_base64 = "D" * 1500
        content = f'{{"screenshotBase64": "{long_base64}"}}'
        result = strip_base64_from_content(content)

        assert "[IMAGE_REMOVED]" in result

    def test_no_strip_short_base64(self, mock_env_vars):
        """Test that short base64 in JSON is not stripped."""
        from src.orchestrator.chat_graph import strip_base64_from_content

        short_base64 = "ABC123"  # Less than 1000 chars
        content = f'{{"screenshot": "{short_base64}"}}'
        result = strip_base64_from_content(content)

        # Short base64 should remain
        assert short_base64 in result

    def test_no_modification_clean_content(self, mock_env_vars):
        """Test that clean content without images is not modified."""
        from src.orchestrator.chat_graph import strip_base64_from_content

        content = "This is regular text without any base64 images."
        result = strip_base64_from_content(content)

        assert result == content


class TestTruncateToolResult:
    """Tests for the truncate_tool_result function."""

    def test_truncate_short_content(self, mock_env_vars):
        """Test that short content is not truncated."""
        from src.orchestrator.chat_graph import truncate_tool_result

        content = "Short result"
        result = truncate_tool_result(content)

        assert result == content

    def test_truncate_long_content(self, mock_env_vars):
        """Test that long content is truncated."""
        from src.orchestrator.chat_graph import MAX_TOOL_RESULT_TOKENS, truncate_tool_result

        # Create content longer than MAX_TOOL_RESULT_TOKENS * 4 chars
        long_content = "x" * (MAX_TOOL_RESULT_TOKENS * 4 + 1000)
        result = truncate_tool_result(long_content)

        assert len(result) < len(long_content)
        assert "[TRUNCATED - result too long]" in result

    def test_truncate_strips_base64_first(self, mock_env_vars):
        """Test that base64 images are stripped before truncation."""
        from src.orchestrator.chat_graph import truncate_tool_result

        long_base64 = "A" * 2000
        content = f'Start {{"screenshot": "{long_base64}"}} End'
        result = truncate_tool_result(content)

        assert "[IMAGE_REMOVED]" in result
        assert long_base64 not in result

    def test_truncate_with_custom_max_tokens(self, mock_env_vars):
        """Test truncation with custom max tokens."""
        from src.orchestrator.chat_graph import truncate_tool_result

        content = "a" * 500  # 125 tokens
        result = truncate_tool_result(content, max_tokens=50)

        # Should be truncated to ~50*4 = 200 chars
        assert len(result) < 500
        assert "[TRUNCATED - result too long]" in result


class TestPruneMessagesForContext:
    """Tests for the prune_messages_for_context function."""

    def test_prune_empty_messages(self, mock_env_vars):
        """Test pruning empty message list."""
        from src.orchestrator.chat_graph import prune_messages_for_context

        result = prune_messages_for_context([])
        assert result == []

    def test_prune_tool_messages_strips_images(self, mock_env_vars):
        """Test that tool messages have base64 images stripped."""
        from src.orchestrator.chat_graph import prune_messages_for_context

        long_base64 = "X" * 2000
        tool_msg = ToolMessage(
            content=f'{{"screenshot": "{long_base64}"}}',
            tool_call_id="test-call-1",
            name="runTest",
        )

        result = prune_messages_for_context([tool_msg])

        assert len(result) == 1
        assert "[IMAGE_REMOVED]" in result[0].content

    def test_prune_ai_messages_strips_images(self, mock_env_vars):
        """Test that AI messages have base64 images stripped."""
        from src.orchestrator.chat_graph import prune_messages_for_context

        long_base64 = "Y" * 2000
        ai_msg = AIMessage(
            content=f'Here is the result with data:image/png;base64,{long_base64}',
        )

        result = prune_messages_for_context([ai_msg])

        assert len(result) == 1
        assert long_base64 not in result[0].content

    def test_prune_human_messages_preserved(self, mock_env_vars):
        """Test that human messages are preserved as-is."""
        from src.orchestrator.chat_graph import prune_messages_for_context

        human_msg = HumanMessage(content="Run the login test")

        result = prune_messages_for_context([human_msg])

        assert len(result) == 1
        assert result[0].content == "Run the login test"

    def test_prune_drops_old_messages_when_over_limit(self, mock_env_vars):
        """Test that old messages are dropped when over token limit."""
        from src.orchestrator.chat_graph import KEEP_RECENT_MESSAGES, prune_messages_for_context

        # Create many messages to exceed the limit
        messages = []
        for i in range(20):
            messages.append(HumanMessage(content="x" * 30000))  # Large message

        result = prune_messages_for_context(messages, max_tokens=10000)

        # Should keep at least KEEP_RECENT_MESSAGES
        assert len(result) >= min(KEEP_RECENT_MESSAGES, 20)
        # But fewer than original
        assert len(result) <= 20

    def test_prune_keeps_minimum_messages(self, mock_env_vars):
        """Test that minimum messages are always kept."""
        from src.orchestrator.chat_graph import KEEP_RECENT_MESSAGES, prune_messages_for_context

        messages = []
        for i in range(KEEP_RECENT_MESSAGES):
            messages.append(HumanMessage(content="x" * 50000))

        # Even with very low token limit, should keep minimum
        result = prune_messages_for_context(messages, max_tokens=1)

        assert len(result) == KEEP_RECENT_MESSAGES

    def test_prune_preserves_tool_call_id(self, mock_env_vars):
        """Test that tool call IDs are preserved after pruning."""
        from src.orchestrator.chat_graph import prune_messages_for_context

        tool_msg = ToolMessage(
            content="test result",
            tool_call_id="unique-call-id-123",
            name="checkStatus",
        )

        result = prune_messages_for_context([tool_msg])

        assert result[0].tool_call_id == "unique-call-id-123"


class TestCreateSystemPrompt:
    """Tests for the create_system_prompt function."""

    def test_create_system_prompt_with_url(self, mock_env_vars):
        """Test creating system prompt with app URL."""
        from src.orchestrator.chat_graph import create_system_prompt

        result = create_system_prompt("http://localhost:3000")

        assert "Argus" in result
        assert "E2E Testing Agent" in result
        assert "http://localhost:3000" in result
        assert "runTest" in result
        assert "executeAction" in result
        assert "discoverElements" in result
        assert "createTest" in result
        assert "checkStatus" in result

    def test_create_system_prompt_without_url(self, mock_env_vars):
        """Test creating system prompt without app URL."""
        from src.orchestrator.chat_graph import create_system_prompt

        result = create_system_prompt("")

        assert "Not specified" in result

    def test_create_system_prompt_no_argument(self, mock_env_vars):
        """Test creating system prompt with no argument."""
        from src.orchestrator.chat_graph import create_system_prompt

        result = create_system_prompt()

        assert "Not specified" in result

    def test_create_system_prompt_mentions_capabilities(self, mock_env_vars):
        """Test that system prompt mentions key capabilities."""
        from src.orchestrator.chat_graph import create_system_prompt

        result = create_system_prompt("http://localhost")

        assert "self-healing" in result.lower()
        assert "visual" in result.lower()
        assert "LangGraph" in result


class TestChatState:
    """Tests for the ChatState TypedDict."""

    def test_chat_state_structure(self, mock_env_vars):
        """Test ChatState has correct structure."""
        from src.orchestrator.chat_graph import ChatState

        # Verify ChatState has expected keys
        annotations = ChatState.__annotations__

        assert "messages" in annotations
        assert "app_url" in annotations
        assert "current_tool" in annotations
        assert "tool_results" in annotations
        assert "session_id" in annotations
        assert "should_continue" in annotations


class TestChatNode:
    """Tests for the chat_node function."""

    @pytest.mark.asyncio
    async def test_chat_node_missing_api_key(self, mock_env_vars, monkeypatch):
        """Test chat_node raises error when API key is missing."""
        from src.orchestrator.chat_graph import chat_node

        # Set API key to None
        with patch("src.orchestrator.chat_graph.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None

            state = {
                "messages": [HumanMessage(content="Hello")],
                "app_url": "http://localhost",
                "current_tool": None,
                "tool_results": [],
                "session_id": "test-session",
                "should_continue": True,
            }

            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not configured"):
                await chat_node(state, {})

    @pytest.mark.asyncio
    async def test_chat_node_invokes_llm(self, mock_env_vars):
        """Test chat_node invokes the LLM correctly."""
        from src.orchestrator.chat_graph import chat_node

        mock_response = MagicMock()
        mock_response.content = "I can help you run tests."
        mock_response.tool_calls = []

        with patch("src.orchestrator.chat_graph.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("src.orchestrator.chat_graph.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.bind_tools.return_value = mock_llm
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "messages": [HumanMessage(content="Run login test")],
                    "app_url": "http://localhost:3000",
                    "current_tool": None,
                    "tool_results": [],
                    "session_id": "test-session",
                    "should_continue": True,
                }

                result = await chat_node(state, {})

                assert "messages" in result
                assert len(result["messages"]) == 1
                mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_node_prunes_messages(self, mock_env_vars):
        """Test that chat_node prunes messages before sending to LLM."""
        from src.orchestrator.chat_graph import chat_node

        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_response.tool_calls = []

        with patch("src.orchestrator.chat_graph.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("src.orchestrator.chat_graph.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.bind_tools.return_value = mock_llm
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                # Add a tool message with base64 image
                long_base64 = "A" * 2000
                tool_msg = ToolMessage(
                    content=f'{{"screenshot": "{long_base64}"}}',
                    tool_call_id="call-1",
                    name="runTest",
                )

                state = {
                    "messages": [HumanMessage(content="Run test"), tool_msg],
                    "app_url": "http://localhost:3000",
                    "current_tool": None,
                    "tool_results": [],
                    "session_id": "test-session",
                    "should_continue": True,
                }

                await chat_node(state, {})

                # Verify ainvoke was called with pruned messages
                call_args = mock_llm.ainvoke.call_args[0][0]
                # First message is system prompt, then pruned messages
                # The tool message should have had base64 stripped
                for msg in call_args:
                    if isinstance(msg, ToolMessage):
                        assert long_base64 not in msg.content

    @pytest.mark.asyncio
    async def test_chat_node_handles_api_key_secret_value(self, mock_env_vars):
        """Test chat_node handles SecretStr API key correctly."""
        from src.orchestrator.chat_graph import chat_node

        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_response.tool_calls = []

        with patch("src.orchestrator.chat_graph.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-secret-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("src.orchestrator.chat_graph.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.bind_tools.return_value = mock_llm
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "messages": [HumanMessage(content="Test")],
                    "app_url": "http://localhost",
                    "current_tool": None,
                    "tool_results": [],
                    "session_id": "test-session",
                    "should_continue": True,
                }

                await chat_node(state, {})

                # Verify LLM was created with the extracted API key
                mock_llm_class.assert_called_once()
                call_kwargs = mock_llm_class.call_args[1]
                assert call_kwargs["api_key"] == "sk-ant-secret-key"


class TestToolExecutorNode:
    """Tests for the tool_executor_node function."""

    @pytest.mark.asyncio
    async def test_tool_executor_no_tool_calls(self, mock_env_vars):
        """Test tool executor with no tool calls."""
        from src.orchestrator.chat_graph import tool_executor_node

        ai_msg = AIMessage(content="No tools needed.")
        ai_msg.tool_calls = []

        state = {
            "messages": [ai_msg],
            "app_url": "http://localhost",
            "current_tool": None,
            "tool_results": [],
            "session_id": "test-session",
            "should_continue": True,
        }

        result = await tool_executor_node(state, {})

        assert result["messages"] == []

    @pytest.mark.asyncio
    async def test_tool_executor_check_status(self, mock_env_vars):
        """Test tool executor handles checkStatus tool."""
        from src.orchestrator.chat_graph import tool_executor_node

        ai_msg = AIMessage(content="Checking status")
        ai_msg.tool_calls = [
            {"name": "checkStatus", "args": {}, "id": "call-123"}
        ]

        state = {
            "messages": [ai_msg],
            "app_url": "http://localhost",
            "current_tool": None,
            "tool_results": [],
            "session_id": "test-session",
            "should_continue": True,
        }

        with patch("src.orchestrator.chat_graph.get_settings") as mock_settings:
            mock_settings.return_value.browser_worker_url = "https://test-worker.com"
            with patch("src.services.cloudflare_storage.is_cloudflare_configured", return_value=False):
                result = await tool_executor_node(state, {})

        assert len(result["messages"]) == 1
        tool_result = json.loads(result["messages"][0].content)
        assert tool_result["success"] is True
        assert "components" in tool_result

    @pytest.mark.asyncio
    async def test_tool_executor_run_test(self, mock_env_vars):
        """Test tool executor handles runTest tool."""
        from src.orchestrator.chat_graph import tool_executor_node

        ai_msg = AIMessage(content="Running test")
        ai_msg.tool_calls = [
            {
                "name": "runTest",
                "args": {
                    "url": "http://localhost:3000",
                    "steps": ["Click login button", "Enter credentials"],
                },
                "id": "call-456",
            }
        ]

        state = {
            "messages": [ai_msg],
            "app_url": "http://localhost:3000",
            "current_tool": None,
            "tool_results": [],
            "session_id": "test-session",
            "should_continue": True,
        }

        # Mock BrowserPoolClient instead of httpx.AsyncClient
        mock_pool_result = MagicMock()
        mock_pool_result.to_dict.return_value = {
            "success": True,
            "steps": [{"passed": True}],
        }

        with patch("src.browser.pool_client.BrowserPoolClient") as mock_browser_pool_class:
            mock_browser_pool = AsyncMock()
            mock_browser_pool.pool_url = "https://test-pool.com"
            mock_browser_pool.test = AsyncMock(return_value=mock_pool_result)
            mock_browser_pool_class.return_value.__aenter__ = AsyncMock(return_value=mock_browser_pool)
            mock_browser_pool_class.return_value.__aexit__ = AsyncMock(return_value=None)
            with patch("src.services.cloudflare_storage.is_cloudflare_configured", return_value=False):
                result = await tool_executor_node(state, {})

        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_call_id == "call-456"

    @pytest.mark.asyncio
    async def test_tool_executor_execute_action(self, mock_env_vars):
        """Test tool executor handles executeAction tool."""
        from src.orchestrator.chat_graph import tool_executor_node

        ai_msg = AIMessage(content="Executing action")
        ai_msg.tool_calls = [
            {
                "name": "executeAction",
                "args": {
                    "url": "http://localhost:3000",
                    "instruction": "Click the submit button",
                },
                "id": "call-789",
            }
        ]

        state = {
            "messages": [ai_msg],
            "app_url": "http://localhost:3000",
            "current_tool": None,
            "tool_results": [],
            "session_id": "test-session",
            "should_continue": True,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True, "action": "clicked"}

        with patch("src.orchestrator.chat_graph.get_settings") as mock_settings:
            mock_settings.return_value.browser_worker_url = "https://test-worker.com"
            with patch("src.services.cloudflare_storage.is_cloudflare_configured", return_value=False):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

                    result = await tool_executor_node(state, {})

        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_tool_executor_discover_elements(self, mock_env_vars):
        """Test tool executor handles discoverElements tool."""
        from src.orchestrator.chat_graph import tool_executor_node

        ai_msg = AIMessage(content="Discovering elements")
        ai_msg.tool_calls = [
            {
                "name": "discoverElements",
                "args": {"url": "http://localhost:3000"},
                "id": "call-discover",
            }
        ]

        state = {
            "messages": [ai_msg],
            "app_url": "http://localhost:3000",
            "current_tool": None,
            "tool_results": [],
            "session_id": "test-session",
            "should_continue": True,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "elements": [{"selector": "#login-btn", "type": "button"}],
        }

        with patch("src.orchestrator.chat_graph.get_settings") as mock_settings:
            mock_settings.return_value.browser_worker_url = "https://test-worker.com"
            with patch("src.services.cloudflare_storage.is_cloudflare_configured", return_value=False):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

                    result = await tool_executor_node(state, {})

        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_tool_executor_create_test(self, mock_env_vars):
        """Test tool executor handles createTest tool."""
        from src.orchestrator.chat_graph import tool_executor_node

        ai_msg = AIMessage(content="Creating test")
        ai_msg.tool_calls = [
            {
                "name": "createTest",
                "args": {
                    "description": "Test login flow with valid credentials",
                    "app_url": "http://localhost:3000",
                },
                "id": "call-create",
            }
        ]

        state = {
            "messages": [ai_msg],
            "app_url": "http://localhost:3000",
            "current_tool": None,
            "tool_results": [],
            "session_id": "test-session",
            "should_continue": True,
        }

        # Mock the NLPTestCreator
        mock_test = MagicMock()
        mock_test.name = "Login Test"
        mock_test.steps = [MagicMock(action="click", target="#login", description="Click login")]
        mock_test.assertions = [MagicMock()]
        mock_test.estimated_duration_seconds = 10
        mock_test.to_dict.return_value = {"name": "Login Test", "steps": []}

        with patch("src.orchestrator.chat_graph.get_settings") as mock_settings:
            mock_settings.return_value.browser_worker_url = "https://test-worker.com"
            with patch("src.services.cloudflare_storage.is_cloudflare_configured", return_value=False):
                with patch("src.agents.nlp_test_creator.NLPTestCreator") as mock_creator_class:
                    mock_creator = MagicMock()
                    mock_creator.create = AsyncMock(return_value=mock_test)
                    mock_creator_class.return_value = mock_creator

                    result = await tool_executor_node(state, {})

        assert len(result["messages"]) == 1
        tool_result = json.loads(result["messages"][0].content)
        assert tool_result["success"] is True
        assert tool_result["_type"] == "test_preview"

    @pytest.mark.asyncio
    async def test_tool_executor_unknown_tool(self, mock_env_vars):
        """Test tool executor handles unknown tool."""
        from src.orchestrator.chat_graph import tool_executor_node

        ai_msg = AIMessage(content="Using unknown tool")
        ai_msg.tool_calls = [
            {
                "name": "unknownTool",
                "args": {},
                "id": "call-unknown",
            }
        ]

        state = {
            "messages": [ai_msg],
            "app_url": "http://localhost",
            "current_tool": None,
            "tool_results": [],
            "session_id": "test-session",
            "should_continue": True,
        }

        with patch("src.orchestrator.chat_graph.get_settings") as mock_settings:
            mock_settings.return_value.browser_worker_url = "https://test-worker.com"
            with patch("src.services.cloudflare_storage.is_cloudflare_configured", return_value=False):
                result = await tool_executor_node(state, {})

        assert len(result["messages"]) == 1
        tool_result = json.loads(result["messages"][0].content)
        assert "error" in tool_result
        assert "Unknown tool" in tool_result["error"]

    @pytest.mark.asyncio
    async def test_tool_executor_handles_exception(self, mock_env_vars):
        """Test tool executor handles exceptions gracefully."""
        from src.orchestrator.chat_graph import tool_executor_node

        ai_msg = AIMessage(content="Running test")
        ai_msg.tool_calls = [
            {
                "name": "runTest",
                "args": {
                    "url": "http://localhost:3000",
                    "steps": ["Test step"],
                },
                "id": "call-error",
            }
        ]

        state = {
            "messages": [ai_msg],
            "app_url": "http://localhost:3000",
            "current_tool": None,
            "tool_results": [],
            "session_id": "test-session",
            "should_continue": True,
        }

        with patch("src.orchestrator.chat_graph.get_settings") as mock_settings:
            mock_settings.return_value.browser_worker_url = "https://test-worker.com"
            with patch("src.services.cloudflare_storage.is_cloudflare_configured", return_value=False):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(side_effect=Exception("Connection failed"))
                    mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

                    result = await tool_executor_node(state, {})

        assert len(result["messages"]) == 1
        tool_result = json.loads(result["messages"][0].content)
        assert "error" in tool_result
        assert "Connection failed" in tool_result["error"]


class TestShouldContinue:
    """Tests for the should_continue function."""

    def test_should_continue_with_tool_calls(self, mock_env_vars):
        """Test should_continue returns 'tools' when tool calls present."""
        from src.orchestrator.chat_graph import should_continue

        ai_msg = AIMessage(content="Using tool")
        ai_msg.tool_calls = [{"name": "runTest", "args": {}, "id": "call-1"}]

        state = {
            "messages": [ai_msg],
            "should_continue": True,
        }

        result = should_continue(state)
        assert result == "tools"

    def test_should_continue_no_tool_calls(self, mock_env_vars):
        """Test should_continue returns 'end' when no tool calls."""
        from src.orchestrator.chat_graph import should_continue

        ai_msg = AIMessage(content="Done")
        ai_msg.tool_calls = []

        state = {
            "messages": [ai_msg],
            "should_continue": True,
        }

        result = should_continue(state)
        assert result == "end"

    def test_should_continue_cancelled(self, mock_env_vars):
        """Test should_continue returns 'end' when cancelled."""
        from src.orchestrator.chat_graph import should_continue

        ai_msg = AIMessage(content="Using tool")
        ai_msg.tool_calls = [{"name": "runTest", "args": {}, "id": "call-1"}]

        state = {
            "messages": [ai_msg],
            "should_continue": False,  # Cancelled
        }

        result = should_continue(state)
        assert result == "end"

    def test_should_continue_missing_flag(self, mock_env_vars):
        """Test should_continue defaults to continuing."""
        from src.orchestrator.chat_graph import should_continue

        ai_msg = AIMessage(content="Using tool")
        ai_msg.tool_calls = [{"name": "runTest", "args": {}, "id": "call-1"}]

        state = {
            "messages": [ai_msg],
            # No should_continue key
        }

        result = should_continue(state)
        assert result == "tools"

    def test_should_continue_no_tool_calls_attribute(self, mock_env_vars):
        """Test should_continue when message has no tool_calls attribute."""
        from src.orchestrator.chat_graph import should_continue

        human_msg = HumanMessage(content="Hello")  # No tool_calls attribute

        state = {
            "messages": [human_msg],
            "should_continue": True,
        }

        result = should_continue(state)
        assert result == "end"


class TestCreateChatGraph:
    """Tests for the create_chat_graph function."""

    def test_create_chat_graph_returns_state_graph(self, mock_env_vars):
        """Test create_chat_graph returns a StateGraph."""
        from langgraph.graph import StateGraph

        from src.orchestrator.chat_graph import create_chat_graph

        graph = create_chat_graph()

        assert isinstance(graph, StateGraph)

    def test_create_chat_graph_has_nodes(self, mock_env_vars):
        """Test create_chat_graph has expected nodes."""
        from src.orchestrator.chat_graph import create_chat_graph

        graph = create_chat_graph()

        # Check nodes exist
        assert "chat" in graph.nodes
        assert "tools" in graph.nodes

    def test_create_chat_graph_compiles(self, mock_env_vars):
        """Test create_chat_graph compiles successfully."""
        from langgraph.checkpoint.memory import MemorySaver

        from src.orchestrator.chat_graph import create_chat_graph

        graph = create_chat_graph()
        checkpointer = MemorySaver()

        # Should compile without error
        compiled = graph.compile(checkpointer=checkpointer)

        assert compiled is not None
