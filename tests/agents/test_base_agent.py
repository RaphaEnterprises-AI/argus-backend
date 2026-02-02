"""Tests for the base agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    @pytest.mark.parametrize(
        "success,data,error,input_tokens,output_tokens,cost,expected_data,expected_error",
        [
            # Success case with data
            (True, {"key": "value"}, None, 100, 50, 0.01, {"key": "value"}, None),
            # Success case with empty data
            (True, {}, None, 0, 0, 0.0, {}, None),
            # Success case with nested data
            (True, {"nested": {"key": "value"}}, None, 200, 100, 0.02, {"nested": {"key": "value"}}, None),
            # Failure case with error message
            (False, None, "Something went wrong", 0, 0, 0.0, None, "Something went wrong"),
            # Failure case with empty error
            (False, None, "", 0, 0, 0.0, None, ""),
            # Failure case with detailed error
            (False, None, "Connection timeout after 30s", 50, 0, 0.005, None, "Connection timeout after 30s"),
        ],
        ids=[
            "success_with_data",
            "success_empty_data",
            "success_nested_data",
            "failure_with_error",
            "failure_empty_error",
            "failure_detailed_error",
        ],
    )
    def test_agent_result_creation(
        self,
        mock_env_vars,
        success,
        data,
        error,
        input_tokens,
        output_tokens,
        cost,
        expected_data,
        expected_error,
    ):
        """Test creating AgentResult with various configurations."""
        from src.agents.base import AgentResult

        result = AgentResult(
            success=success,
            data=data,
            error=error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

        assert result.success is success
        assert result.data == expected_data
        assert result.error == expected_error
        assert result.input_tokens == input_tokens
        assert result.output_tokens == output_tokens
        assert result.cost == cost


class TestUsageStats:
    """Tests for UsageStats dataclass."""

    def test_usage_stats_defaults(self, mock_env_vars):
        """Test UsageStats default values."""
        from src.agents.base import UsageStats

        stats = UsageStats()

        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_cost == 0.0
        assert stats.total_calls == 0

    @pytest.mark.parametrize(
        "input_tokens,output_tokens,cost,calls",
        [
            (0, 0, 0.0, 0),
            (100, 50, 0.01, 1),
            (1000, 500, 0.10, 10),
            (10000, 5000, 1.0, 100),
        ],
        ids=["zero_usage", "single_call", "moderate_usage", "high_usage"],
    )
    def test_usage_stats_custom_values(
        self, mock_env_vars, input_tokens, output_tokens, cost, calls
    ):
        """Test UsageStats with custom values."""
        from src.agents.base import UsageStats

        stats = UsageStats(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cost=cost,
            total_calls=calls,
        )

        assert stats.total_input_tokens == input_tokens
        assert stats.total_output_tokens == output_tokens
        assert stats.total_cost == cost
        assert stats.total_calls == calls


class TestBaseAgent:
    """Tests for BaseAgent class."""

    def test_base_agent_initialization(self, mock_env_vars):
        """Test BaseAgent initialization."""
        from src.agents.base import BaseAgent

        # Create a concrete implementation
        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent()

        assert agent.settings is not None
        assert agent.config is not None
        assert agent.model is not None

    def test_base_agent_with_config(self, mock_env_vars):
        """Test BaseAgent with custom config."""
        from src.agents.base import BaseAgent
        from src.config import AgentConfig, ModelName

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        config = AgentConfig(name="custom_agent", model=ModelName.OPUS)
        agent = TestAgent(config=config)

        assert agent.config.name == "custom_agent"

    def test_model_router_lazy_initialization(self, mock_env_vars):
        """Test that model router is lazily initialized."""
        from src.agents.base import BaseAgent

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent()
        assert agent._model_router is None

        # Access property triggers initialization
        router = agent.model_router
        assert router is not None
        assert agent._model_router is not None

    def test_client_lazy_initialization(self, mock_env_vars):
        """Test that Anthropic client is lazily initialized."""
        from src.agents.base import BaseAgent

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent()
        assert agent._client is None

        # Access property triggers initialization
        with patch('anthropic.Anthropic'):
            client = agent.client
            assert client is not None

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            # Plain JSON
            ('{"key": "value"}', {"key": "value"}),
            # JSON in markdown code block
            ('```json\n{"key": "value"}\n```', {"key": "value"}),
            # JSON in code block without language
            ('```\n{"key": "value"}\n```', {"key": "value"}),
            # Empty object
            ('{}', {}),
            # Array
            ('[1, 2, 3]', [1, 2, 3]),
            # Nested JSON
            ('{"outer": {"inner": "value"}}', {"outer": {"inner": "value"}}),
            # JSON with whitespace
            ('  \n  {"key": "value"}  \n  ', {"key": "value"}),
            # Boolean values
            ('{"flag": true, "other": false}', {"flag": True, "other": False}),
            # Null value
            ('{"key": null}', {"key": None}),
            # Numeric values
            ('{"int": 42, "float": 3.14}', {"int": 42, "float": 3.14}),
            # String with special characters
            ('{"msg": "Hello\\nWorld"}', {"msg": "Hello\nWorld"}),
        ],
        ids=[
            "plain_json",
            "markdown_code_block",
            "code_block_no_lang",
            "empty_object",
            "array",
            "nested_json",
            "whitespace_padded",
            "boolean_values",
            "null_value",
            "numeric_values",
            "special_characters",
        ],
    )
    def test_parse_json_response(self, mock_env_vars, input_text, expected):
        """Test JSON response parsing with various input formats."""
        from src.agents.base import BaseAgent

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent()
        result = agent._parse_json_response(input_text)
        assert result == expected

    @pytest.mark.parametrize(
        "invalid_input,fallback,expected",
        [
            # Invalid JSON with dict fallback
            ("invalid json", {"default": True}, {"default": True}),
            # Invalid JSON with None fallback
            ("not json at all", None, None),
            # Empty string with fallback
            ("", {"empty": True}, {"empty": True}),
            # Partial JSON with fallback
            ('{"incomplete":', {"fallback": "value"}, {"fallback": "value"}),
            # Plain text with fallback
            ("just some text", [], []),
        ],
        ids=[
            "invalid_with_dict_fallback",
            "invalid_with_none_fallback",
            "empty_string",
            "partial_json",
            "plain_text_with_list_fallback",
        ],
    )
    def test_parse_json_response_fallback(
        self, mock_env_vars, invalid_input, fallback, expected
    ):
        """Test JSON response parsing with fallback for invalid inputs."""
        from src.agents.base import BaseAgent

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent()
        result = agent._parse_json_response(invalid_input, fallback=fallback)
        assert result == expected

    @pytest.mark.parametrize(
        "response_text,expected",
        [
            ("Hello, world!", "Hello, world!"),
            ("", ""),
            ("Multi\nLine\nResponse", "Multi\nLine\nResponse"),
            ("   Whitespace   ", "   Whitespace   "),
            ('{"json": "response"}', '{"json": "response"}'),
            ("Unicode: \u00e9\u00e0\u00fc\u00f1", "Unicode: \u00e9\u00e0\u00fc\u00f1"),
        ],
        ids=[
            "simple_text",
            "empty_text",
            "multiline",
            "whitespace",
            "json_as_text",
            "unicode",
        ],
    )
    def test_extract_text_response(
        self, mock_env_vars, mock_anthropic_client, response_text, expected
    ):
        """Test extracting text from Claude response with various content."""
        from src.agents.base import BaseAgent

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent()

        # Create mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=response_text)]

        result = agent._extract_text_response(mock_response)
        assert result == expected

    @pytest.mark.parametrize(
        "current_cost,expected_within_limit",
        [
            (0.0, True),       # Zero cost - within limit
            (0.01, True),      # Small cost - within limit
            (1.0, True),       # Moderate cost - within limit
            (9.99, True),      # Just under default limit (10.0)
            (10.0, False),     # At default limit - exceeds
            (10.01, False),    # Just over limit - exceeds
            (100.0, False),    # Well over limit - exceeds
            (0.001, True),     # Very small cost - within limit
        ],
        ids=[
            "zero_cost",
            "small_cost",
            "moderate_cost",
            "just_under_limit",
            "at_limit",
            "just_over_limit",
            "well_over_limit",
            "tiny_cost",
        ],
    )
    def test_check_cost_limit(self, mock_env_vars, current_cost, expected_within_limit):
        """Test cost limit checking with various cost values."""
        from src.agents.base import BaseAgent

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent()
        agent._usage.total_cost = current_cost
        assert agent._check_cost_limit() is expected_within_limit

    @pytest.mark.parametrize(
        "initial_input_tokens,initial_output_tokens,initial_cost,initial_calls",
        [
            (1000, 500, 0.05, 5),
            (0, 0, 0.0, 0),
            (10000, 5000, 1.0, 100),
            (1, 1, 0.001, 1),
        ],
        ids=["moderate_usage", "zero_usage", "high_usage", "minimal_usage"],
    )
    def test_reset_usage(
        self,
        mock_env_vars,
        initial_input_tokens,
        initial_output_tokens,
        initial_cost,
        initial_calls,
    ):
        """Test resetting usage statistics from various initial states."""
        from src.agents.base import BaseAgent

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent()

        # Set initial usage
        agent._usage.total_input_tokens = initial_input_tokens
        agent._usage.total_output_tokens = initial_output_tokens
        agent._usage.total_cost = initial_cost
        agent._usage.total_calls = initial_calls

        # Reset
        agent.reset_usage()

        # All values should be reset to zero
        assert agent._usage.total_input_tokens == 0
        assert agent._usage.total_output_tokens == 0
        assert agent._usage.total_cost == 0.0
        assert agent._usage.total_calls == 0

    def test_call_claude_with_retry(self, mock_env_vars, mock_anthropic_client):
        """Test Claude API call with retry logic."""
        from src.agents.base import BaseAgent

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent()
        agent._client = mock_anthropic_client

        response = agent._call_claude(
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response is not None
        mock_anthropic_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_model_multi_model_mode(self, mock_env_vars):
        """Test _call_model with multi-model routing."""
        from src.agents.base import BaseAgent
        from src.core.model_router import TaskType

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent(use_multi_model=True)

        # Mock the model router
        mock_result = {
            "content": "Test response",
            "model": "test-model",
            "model_name": "sonnet",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.01,
        }

        with patch.object(agent.model_router, 'complete', new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_result

            result = await agent._call_model(
                messages=[{"role": "user", "content": "Hello"}],
                task_type=TaskType.GENERAL,
            )

            assert result["content"] == "Test response"
            mock_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_model_claude_only_mode(self, mock_env_vars, mock_anthropic_client):
        """Test _call_model in Claude-only mode."""
        from src.agents.base import BaseAgent

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent(use_multi_model=False)
        agent._client = mock_anthropic_client

        result = await agent._call_model(
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert "content" in result
        assert result["provider"] == "anthropic"


class TestUsageTracking:
    """Tests for usage tracking in BaseAgent."""

    @pytest.mark.parametrize(
        "input_tokens,output_tokens",
        [
            (100, 50),
            (0, 0),
            (1000, 500),
            (1, 1),
            (10000, 5000),
        ],
        ids=["typical", "zero", "moderate", "minimal", "large"],
    )
    def test_track_usage(
        self, mock_env_vars, mock_anthropic_client, input_tokens, output_tokens
    ):
        """Test that usage is tracked correctly with various token counts."""
        from src.agents.base import BaseAgent

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent()

        # Create mock response
        mock_response = MagicMock()
        mock_response.usage = MagicMock(
            input_tokens=input_tokens, output_tokens=output_tokens
        )

        agent._track_usage(mock_response)

        assert agent._usage.total_input_tokens == input_tokens
        assert agent._usage.total_output_tokens == output_tokens
        if input_tokens > 0 or output_tokens > 0:
            assert agent._usage.total_cost > 0
        assert agent._usage.total_calls == 1

    @pytest.mark.parametrize(
        "call_sequence",
        [
            [(100, 50), (200, 100)],
            [(50, 25), (50, 25), (50, 25)],
            [(1000, 500), (1, 1)],
        ],
        ids=["two_calls", "three_equal_calls", "large_then_small"],
    )
    def test_track_usage_cumulative(
        self, mock_env_vars, mock_anthropic_client, call_sequence
    ):
        """Test that usage accumulates correctly across multiple calls."""
        from src.agents.base import BaseAgent

        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None

            def _get_system_prompt(self):
                return "Test prompt"

        agent = TestAgent()

        expected_input = 0
        expected_output = 0

        for input_tokens, output_tokens in call_sequence:
            mock_response = MagicMock()
            mock_response.usage = MagicMock(
                input_tokens=input_tokens, output_tokens=output_tokens
            )
            agent._track_usage(mock_response)

            expected_input += input_tokens
            expected_output += output_tokens

        assert agent._usage.total_input_tokens == expected_input
        assert agent._usage.total_output_tokens == expected_output
        assert agent._usage.total_calls == len(call_sequence)
