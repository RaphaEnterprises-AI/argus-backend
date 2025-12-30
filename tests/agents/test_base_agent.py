"""Tests for the base agent module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_agent_result_success(self, mock_env_vars):
        """Test creating a successful AgentResult."""
        from src.agents.base import AgentResult
        
        result = AgentResult(
            success=True,
            data={"key": "value"},
            input_tokens=100,
            output_tokens=50,
            cost=0.01,
        )
        
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_agent_result_failure(self, mock_env_vars):
        """Test creating a failed AgentResult."""
        from src.agents.base import AgentResult
        
        result = AgentResult(
            success=False,
            error="Something went wrong",
        )
        
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data is None


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
        with patch('anthropic.Anthropic') as mock_anthropic:
            client = agent.client
            assert client is not None

    def test_parse_json_response(self, mock_env_vars):
        """Test JSON response parsing."""
        from src.agents.base import BaseAgent
        
        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None
            
            def _get_system_prompt(self):
                return "Test prompt"
        
        agent = TestAgent()
        
        # Test plain JSON
        result = agent._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}
        
        # Test JSON in markdown code block
        result = agent._parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}
        
        # Test invalid JSON with fallback
        result = agent._parse_json_response('invalid json', fallback={"default": True})
        assert result == {"default": True}

    def test_extract_text_response(self, mock_env_vars, mock_anthropic_client):
        """Test extracting text from Claude response."""
        from src.agents.base import BaseAgent
        
        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None
            
            def _get_system_prompt(self):
                return "Test prompt"
        
        agent = TestAgent()
        
        # Create mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello, world!")]
        
        result = agent._extract_text_response(mock_response)
        assert result == "Hello, world!"

    def test_check_cost_limit(self, mock_env_vars):
        """Test cost limit checking."""
        from src.agents.base import BaseAgent
        
        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None
            
            def _get_system_prompt(self):
                return "Test prompt"
        
        agent = TestAgent()
        
        # Should be within budget initially
        assert agent._check_cost_limit() is True
        
        # Set usage above limit
        agent._usage.total_cost = 100.0
        assert agent._check_cost_limit() is False

    def test_reset_usage(self, mock_env_vars):
        """Test resetting usage statistics."""
        from src.agents.base import BaseAgent
        
        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None
            
            def _get_system_prompt(self):
                return "Test prompt"
        
        agent = TestAgent()
        
        # Add some usage
        agent._usage.total_input_tokens = 1000
        agent._usage.total_cost = 0.05
        
        # Reset
        agent.reset_usage()
        
        assert agent._usage.total_input_tokens == 0
        assert agent._usage.total_cost == 0.0

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

    def test_track_usage(self, mock_env_vars, mock_anthropic_client):
        """Test that usage is tracked correctly."""
        from src.agents.base import BaseAgent
        
        class TestAgent(BaseAgent):
            async def execute(self, **kwargs):
                return None
            
            def _get_system_prompt(self):
                return "Test prompt"
        
        agent = TestAgent()
        
        # Create mock response
        mock_response = MagicMock()
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        
        agent._track_usage(mock_response)
        
        assert agent._usage.total_input_tokens == 100
        assert agent._usage.total_output_tokens == 50
        assert agent._usage.total_cost > 0
        assert agent._usage.total_calls == 1
