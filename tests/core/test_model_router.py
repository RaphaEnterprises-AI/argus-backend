"""Tests for the model router module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_creation(self, mock_env_vars):
        """Test creating a ModelConfig."""
        from src.core.model_router import ModelConfig, ModelProvider
        
        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-sonnet-4-5-20250514",
            input_cost_per_1m=3.00,
            output_cost_per_1m=15.00,
            max_tokens=8192,
            supports_vision=True,
            supports_tools=True,
        )
        
        assert config.provider == ModelProvider.ANTHROPIC
        assert config.model_id == "claude-sonnet-4-5-20250514"
        assert config.supports_vision is True

    def test_avg_cost_per_1k(self, mock_env_vars):
        """Test average cost calculation."""
        from src.core.model_router import ModelConfig, ModelProvider
        
        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="test-model",
            input_cost_per_1m=3.00,
            output_cost_per_1m=15.00,
            max_tokens=8192,
        )
        
        # Expected: (3.00 * 0.6 + 15.00 * 0.4) / 1000 = 0.0078
        expected = (3.00 * 0.6 + 15.00 * 0.4) / 1000
        assert abs(config.avg_cost_per_1k - expected) < 0.0001


class TestTaskTypes:
    """Tests for TaskType enum."""

    def test_task_types_exist(self, mock_env_vars):
        """Test that all expected task types exist."""
        from src.core.model_router import TaskType
        
        assert TaskType.ELEMENT_CLASSIFICATION
        assert TaskType.CODE_ANALYSIS
        assert TaskType.VISUAL_COMPARISON
        assert TaskType.SELF_HEALING
        assert TaskType.COMPUTER_USE_SIMPLE
        assert TaskType.GENERAL


class TestTaskModelMapping:
    """Tests for task to model mapping."""

    def test_task_model_mapping_exists(self, mock_env_vars):
        """Test that all task types have model mappings."""
        from src.core.model_router import TASK_MODEL_MAPPING, TaskType
        
        for task_type in TaskType:
            assert task_type in TASK_MODEL_MAPPING
            assert len(TASK_MODEL_MAPPING[task_type]) > 0

    def test_trivial_tasks_use_cheap_models(self, mock_env_vars):
        """Test that trivial tasks prefer cheaper models."""
        from src.core.model_router import TASK_MODEL_MAPPING, TaskType
        
        # Trivial tasks should start with cheap models
        cheap_models = ["llama-3.1-8b", "gemini-flash", "gpt-4o-mini"]
        
        for model in TASK_MODEL_MAPPING[TaskType.ELEMENT_CLASSIFICATION]:
            if model in cheap_models:
                break
        else:
            pytest.fail("Trivial tasks should prefer cheap models")


class TestModelRouter:
    """Tests for ModelRouter class."""

    def test_router_initialization(self, mock_env_vars):
        """Test ModelRouter initialization."""
        from src.core.model_router import ModelRouter, ModelProvider
        
        router = ModelRouter(
            prefer_provider=ModelProvider.ANTHROPIC,
            cost_limit_per_call=0.10,
            enable_fallback=True,
        )
        
        assert router.prefer_provider == ModelProvider.ANTHROPIC
        assert router.cost_limit == 0.10
        assert router.enable_fallback is True

    def test_select_model_for_task(self, mock_env_vars):
        """Test model selection for a task."""
        from src.core.model_router import ModelRouter, TaskType
        
        router = ModelRouter()
        model_name, config = router.select_model(TaskType.ELEMENT_CLASSIFICATION)
        
        assert model_name is not None
        assert config is not None
        assert config.model_id is not None

    def test_select_model_with_vision_requirement(self, mock_env_vars):
        """Test model selection when vision is required."""
        from src.core.model_router import ModelRouter, TaskType
        
        router = ModelRouter()
        model_name, config = router.select_model(
            TaskType.VISUAL_COMPARISON,
            requires_vision=True,
        )
        
        assert config.supports_vision is True

    def test_select_model_with_tools_requirement(self, mock_env_vars):
        """Test model selection when tools are required."""
        from src.core.model_router import ModelRouter, TaskType
        
        router = ModelRouter()
        model_name, config = router.select_model(
            TaskType.CODE_ANALYSIS,
            requires_tools=True,
        )
        
        assert config.supports_tools is True

    def test_select_model_fallback_to_sonnet(self, mock_env_vars):
        """Test that model selection falls back to Sonnet."""
        from src.core.model_router import ModelRouter, TaskType
        
        # Create router with very low cost limit
        router = ModelRouter(cost_limit_per_call=0.00001)
        model_name, config = router.select_model(TaskType.GENERAL)
        
        # Should still return something (fallback)
        assert model_name is not None

    @pytest.mark.asyncio
    async def test_complete_basic(self, mock_env_vars):
        """Test basic completion call."""
        from src.core.model_router import ModelRouter, TaskType
        
        router = ModelRouter()
        
        # Mock the client
        with patch.object(router, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value={
                "content": "Test response",
                "input_tokens": 100,
                "output_tokens": 50,
                "model": "test-model",
            })
            mock_get_client.return_value = mock_client
            
            result = await router.complete(
                task_type=TaskType.GENERAL,
                messages=[{"role": "user", "content": "Hello"}],
            )
            
            assert "content" in result
            assert "cost" in result

    def test_get_cost_report(self, mock_env_vars):
        """Test cost report generation."""
        from src.core.model_router import ModelRouter
        
        router = ModelRouter()
        
        # Manually add some usage stats
        router.usage_stats["sonnet"] = {
            "calls": 10,
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
            "total_cost": 0.05,
        }
        
        report = router.get_cost_report()
        
        assert "total_cost" in report
        assert "by_model" in report
        assert "potential_savings" in report


class TestModelClients:
    """Tests for model client implementations."""

    @pytest.mark.asyncio
    async def test_anthropic_client_complete(self, mock_env_vars, mock_async_anthropic_client):
        """Test AnthropicClient.complete method."""
        from src.core.model_router import AnthropicClient, ModelConfig, ModelProvider

        # Create a mock for the anthropic module that returns our mock client
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_anthropic_client

        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            # Need to reload to get the patched module
            client = AnthropicClient()
            # Override the client directly
            client.client = mock_async_anthropic_client

            config = ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_id="claude-sonnet-4-5-20250514",
                input_cost_per_1m=3.00,
                output_cost_per_1m=15.00,
                max_tokens=8192,
            )

            result = await client.complete(
                messages=[{"role": "user", "content": "Hello"}],
                model_config=config,
            )

            assert "content" in result
            assert "input_tokens" in result
            assert "output_tokens" in result

    @pytest.mark.asyncio
    async def test_anthropic_client_complete_with_temperature(self, mock_env_vars, mock_async_anthropic_client):
        """Test AnthropicClient.complete with temperature."""
        from src.core.model_router import AnthropicClient, ModelConfig, ModelProvider

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_anthropic_client

        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            client = AnthropicClient()
            client.client = mock_async_anthropic_client

            config = ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_id="claude-sonnet-4-5-20250514",
                input_cost_per_1m=3.00,
                output_cost_per_1m=15.00,
                max_tokens=8192,
            )

            result = await client.complete(
                messages=[{"role": "user", "content": "Hello"}],
                model_config=config,
                temperature=0.7,
            )

            assert "content" in result

    @pytest.mark.asyncio
    async def test_anthropic_client_complete_with_tools(self, mock_env_vars, mock_async_anthropic_client):
        """Test AnthropicClient.complete with tools."""
        from src.core.model_router import AnthropicClient, ModelConfig, ModelProvider

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_anthropic_client

        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            client = AnthropicClient()
            client.client = mock_async_anthropic_client

            config = ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_id="claude-sonnet-4-5-20250514",
                input_cost_per_1m=3.00,
                output_cost_per_1m=15.00,
                max_tokens=8192,
                supports_tools=True,
            )

            tools = [{"name": "test_tool", "description": "A test tool"}]

            result = await client.complete(
                messages=[{"role": "user", "content": "Hello"}],
                model_config=config,
                tools=tools,
            )

            assert "content" in result

    @pytest.mark.asyncio
    async def test_anthropic_client_complete_with_vision(self, mock_env_vars, mock_async_anthropic_client):
        """Test AnthropicClient.complete_with_vision method."""
        from src.core.model_router import AnthropicClient, ModelConfig, ModelProvider

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_anthropic_client

        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            client = AnthropicClient()
            client.client = mock_async_anthropic_client

            config = ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_id="claude-sonnet-4-5-20250514",
                input_cost_per_1m=3.00,
                output_cost_per_1m=15.00,
                max_tokens=8192,
                supports_vision=True,
            )

            result = await client.complete_with_vision(
                messages=[{"role": "user", "content": "Describe this image"}],
                images=[b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'],
                model_config=config,
            )

            assert "content" in result

    @pytest.mark.asyncio
    async def test_anthropic_client_vision_with_existing_content_list(self, mock_env_vars, mock_async_anthropic_client):
        """Test AnthropicClient.complete_with_vision with existing content list."""
        from src.core.model_router import AnthropicClient, ModelConfig, ModelProvider

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_anthropic_client

        with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
            client = AnthropicClient()
            client.client = mock_async_anthropic_client

            config = ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_id="claude-sonnet-4-5-20250514",
                input_cost_per_1m=3.00,
                output_cost_per_1m=15.00,
                max_tokens=8192,
                supports_vision=True,
            )

            # Messages with content already as list
            result = await client.complete_with_vision(
                messages=[{"role": "user", "content": [{"type": "text", "text": "Describe"}]}],
                images=[b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'],
                model_config=config,
            )

            assert "content" in result


try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
class TestOpenAIClient:
    """Tests for OpenAI client implementation."""

    @pytest.mark.asyncio
    async def test_openai_client_complete(self, mock_env_vars):
        """Test OpenAIClient.complete method."""
        from src.core.model_router import OpenAIClient, ModelConfig, ModelProvider

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            client = OpenAIClient()
            client.client = mock_openai_client

            config = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_id="gpt-4o",
                input_cost_per_1m=2.50,
                output_cost_per_1m=10.00,
                max_tokens=16384,
                supports_json_mode=True,
            )

            result = await client.complete(
                messages=[{"role": "user", "content": "Hello"}],
                model_config=config,
            )

            assert result["content"] == "Test response"
            assert result["input_tokens"] == 100
            assert result["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_openai_client_complete_with_json_mode(self, mock_env_vars):
        """Test OpenAIClient.complete with json_mode."""
        from src.core.model_router import OpenAIClient, ModelConfig, ModelProvider

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"result": "test"}'))]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            client = OpenAIClient()
            client.client = mock_openai_client

            config = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_id="gpt-4o",
                input_cost_per_1m=2.50,
                output_cost_per_1m=10.00,
                max_tokens=16384,
                supports_json_mode=True,
            )

            result = await client.complete(
                messages=[{"role": "user", "content": "Return JSON"}],
                model_config=config,
                json_mode=True,
            )

            assert "result" in result["content"]

    @pytest.mark.asyncio
    async def test_openai_client_complete_with_tools(self, mock_env_vars):
        """Test OpenAIClient.complete with tools."""
        from src.core.model_router import OpenAIClient, ModelConfig, ModelProvider

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            client = OpenAIClient()
            client.client = mock_openai_client

            config = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_id="gpt-4o",
                input_cost_per_1m=2.50,
                output_cost_per_1m=10.00,
                max_tokens=16384,
                supports_tools=True,
            )

            tools = [{"name": "test_tool", "description": "Test", "input_schema": {}}]

            result = await client.complete(
                messages=[{"role": "user", "content": "Use tool"}],
                model_config=config,
                tools=tools,
            )

            assert "content" in result

    @pytest.mark.asyncio
    async def test_openai_client_complete_with_vision(self, mock_env_vars):
        """Test OpenAIClient.complete_with_vision method."""
        from src.core.model_router import OpenAIClient, ModelConfig, ModelProvider

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Image description"))]
        mock_response.usage = MagicMock(prompt_tokens=200, completion_tokens=100)

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            client = OpenAIClient()
            client.client = mock_openai_client

            config = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_id="gpt-4o",
                input_cost_per_1m=2.50,
                output_cost_per_1m=10.00,
                max_tokens=16384,
                supports_vision=True,
            )

            result = await client.complete_with_vision(
                messages=[{"role": "user", "content": "Describe this image"}],
                images=[b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'],
                model_config=config,
            )

            assert result["content"] == "Image description"

    def test_openai_convert_tools(self, mock_env_vars):
        """Test OpenAIClient tool conversion."""
        from src.core.model_router import OpenAIClient

        with patch("openai.AsyncOpenAI"):
            client = OpenAIClient()

            anthropic_tools = [
                {"name": "test_tool", "description": "Test description", "input_schema": {"type": "object"}}
            ]

            openai_tools = client._convert_tools(anthropic_tools)

            assert len(openai_tools) == 1
            assert openai_tools[0]["type"] == "function"
            assert openai_tools[0]["function"]["name"] == "test_tool"


@pytest.mark.skipif(not HAS_GROQ, reason="groq not installed")
class TestGroqClient:
    """Tests for Groq client implementation."""

    @pytest.mark.asyncio
    async def test_groq_client_complete(self, mock_env_vars):
        """Test GroqClient.complete method."""
        from src.core.model_router import GroqClient, ModelConfig, ModelProvider

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Fast response"))]
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=25)

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("groq.AsyncGroq", return_value=mock_groq_client):
            client = GroqClient()
            client.client = mock_groq_client

            config = ModelConfig(
                provider=ModelProvider.GROQ,
                model_id="llama-3.1-8b-instant",
                input_cost_per_1m=0.05,
                output_cost_per_1m=0.08,
                max_tokens=8192,
                supports_json_mode=True,
            )

            result = await client.complete(
                messages=[{"role": "user", "content": "Hello"}],
                model_config=config,
            )

            assert result["content"] == "Fast response"

    @pytest.mark.asyncio
    async def test_groq_client_complete_with_json_mode(self, mock_env_vars):
        """Test GroqClient.complete with json_mode."""
        from src.core.model_router import GroqClient, ModelConfig, ModelProvider

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"result": "ok"}'))]
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=25)

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("groq.AsyncGroq", return_value=mock_groq_client):
            client = GroqClient()
            client.client = mock_groq_client

            config = ModelConfig(
                provider=ModelProvider.GROQ,
                model_id="llama-3.1-8b-instant",
                input_cost_per_1m=0.05,
                output_cost_per_1m=0.08,
                max_tokens=8192,
                supports_json_mode=True,
            )

            result = await client.complete(
                messages=[{"role": "user", "content": "Return JSON"}],
                model_config=config,
                json_mode=True,
            )

            assert "result" in result["content"]

    @pytest.mark.asyncio
    async def test_groq_client_vision_not_supported(self, mock_env_vars):
        """Test GroqClient.complete_with_vision raises NotImplementedError."""
        from src.core.model_router import GroqClient, ModelConfig, ModelProvider

        with patch("groq.AsyncGroq"):
            client = GroqClient()

            config = ModelConfig(
                provider=ModelProvider.GROQ,
                model_id="llama-3.1-8b-instant",
                input_cost_per_1m=0.05,
                output_cost_per_1m=0.08,
                max_tokens=8192,
            )

            with pytest.raises(NotImplementedError, match="Groq doesn't support vision"):
                await client.complete_with_vision(
                    messages=[{"role": "user", "content": "Hello"}],
                    images=[b'\x89PNG'],
                    model_config=config,
                )


try:
    import google.generativeai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False


class TestModelRouterAdvanced:
    """Advanced tests for ModelRouter class."""

    def test_get_client_anthropic(self, mock_env_vars):
        """Test _get_client for Anthropic provider."""
        from src.core.model_router import ModelRouter, ModelProvider, AnthropicClient

        with patch("anthropic.AsyncAnthropic"):
            router = ModelRouter()
            client = router._get_client(ModelProvider.ANTHROPIC)

            assert isinstance(client, AnthropicClient)
            # Should cache the client
            assert router._clients[ModelProvider.ANTHROPIC] is client

    @pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
    def test_get_client_openai(self, mock_env_vars):
        """Test _get_client for OpenAI provider."""
        from src.core.model_router import ModelRouter, ModelProvider, OpenAIClient

        with patch("openai.AsyncOpenAI"):
            router = ModelRouter()
            client = router._get_client(ModelProvider.OPENAI)

            assert isinstance(client, OpenAIClient)

    @pytest.mark.skipif(not HAS_GOOGLE, reason="google-generativeai not installed")
    def test_get_client_google(self, mock_env_vars):
        """Test _get_client for Google provider."""
        from src.core.model_router import ModelRouter, ModelProvider, GoogleClient

        with patch("google.generativeai.configure"):
            router = ModelRouter()
            client = router._get_client(ModelProvider.GOOGLE)

            assert isinstance(client, GoogleClient)

    @pytest.mark.skipif(not HAS_GROQ, reason="groq not installed")
    def test_get_client_groq(self, mock_env_vars):
        """Test _get_client for Groq provider."""
        from src.core.model_router import ModelRouter, ModelProvider, GroqClient

        with patch("groq.AsyncGroq"):
            router = ModelRouter()
            client = router._get_client(ModelProvider.GROQ)

            assert isinstance(client, GroqClient)

    def test_get_client_unsupported(self, mock_env_vars):
        """Test _get_client for unsupported provider."""
        from src.core.model_router import ModelRouter, ModelProvider

        router = ModelRouter()

        with pytest.raises(ValueError, match="Unsupported provider"):
            router._get_client(ModelProvider.LOCAL)

    def test_select_model_with_latency_constraint(self, mock_env_vars):
        """Test model selection with latency constraint."""
        from src.core.model_router import ModelRouter, TaskType

        router = ModelRouter()

        # Request very low latency - should find fast models
        model_name, config = router.select_model(
            TaskType.ELEMENT_CLASSIFICATION,
            max_latency_ms=200,
        )

        # Should select a fast model or fallback
        assert config.latency_ms <= 200 or model_name == "sonnet"

    def test_select_model_prefer_provider(self, mock_env_vars):
        """Test model selection with provider preference."""
        from src.core.model_router import ModelRouter, TaskType, ModelProvider

        router = ModelRouter(prefer_provider=ModelProvider.ANTHROPIC)
        model_name, config = router.select_model(TaskType.GENERAL)

        # Should still work, provider preference is soft
        assert model_name is not None

    @pytest.mark.asyncio
    async def test_complete_with_vision(self, mock_env_vars):
        """Test complete with vision images."""
        from src.core.model_router import ModelRouter, TaskType

        router = ModelRouter()

        mock_client = AsyncMock()
        mock_client.complete_with_vision = AsyncMock(return_value={
            "content": "Image analyzed",
            "input_tokens": 200,
            "output_tokens": 50,
            "model": "gpt-4o",
        })

        with patch.object(router, '_get_client', return_value=mock_client):
            result = await router.complete(
                task_type=TaskType.VISUAL_COMPARISON,
                messages=[{"role": "user", "content": "Describe"}],
                images=[b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'],
            )

            assert result["content"] == "Image analyzed"
            mock_client.complete_with_vision.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_with_tools(self, mock_env_vars):
        """Test complete with tools."""
        from src.core.model_router import ModelRouter, TaskType

        router = ModelRouter()

        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value={
            "content": "Tool used",
            "input_tokens": 100,
            "output_tokens": 50,
            "model": "sonnet",
        })

        with patch.object(router, '_get_client', return_value=mock_client):
            tools = [{"name": "test_tool", "description": "Test"}]
            result = await router.complete(
                task_type=TaskType.CODE_ANALYSIS,
                messages=[{"role": "user", "content": "Use tool"}],
                tools=tools,
            )

            assert result["content"] == "Tool used"

    @pytest.mark.asyncio
    async def test_complete_fallback_on_error(self, mock_env_vars):
        """Test complete falls back on error."""
        from src.core.model_router import ModelRouter, TaskType

        router = ModelRouter(enable_fallback=True)

        mock_client = AsyncMock()
        # First call fails, second (fallback) succeeds
        mock_client.complete = AsyncMock(side_effect=[
            Exception("Primary failed"),
            {"content": "Fallback", "input_tokens": 100, "output_tokens": 50, "model": "sonnet"}
        ])

        with patch.object(router, '_get_client', return_value=mock_client):
            result = await router.complete(
                task_type=TaskType.GENERAL,
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert result["content"] == "Fallback"
            assert result.get("fallback") is True

    @pytest.mark.asyncio
    async def test_complete_no_fallback(self, mock_env_vars):
        """Test complete without fallback raises error."""
        from src.core.model_router import ModelRouter, TaskType

        router = ModelRouter(enable_fallback=False)

        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(side_effect=Exception("API Error"))

        with patch.object(router, '_get_client', return_value=mock_client):
            with pytest.raises(Exception, match="API Error"):
                await router.complete(
                    task_type=TaskType.GENERAL,
                    messages=[{"role": "user", "content": "Hello"}],
                )

    @pytest.mark.asyncio
    async def test_fallback_complete_with_vision(self, mock_env_vars):
        """Test _fallback_complete with vision."""
        from src.core.model_router import ModelRouter

        router = ModelRouter()

        mock_client = AsyncMock()
        mock_client.complete_with_vision = AsyncMock(return_value={
            "content": "Fallback vision",
            "input_tokens": 200,
            "output_tokens": 100,
            "model": "sonnet",
        })

        with patch.object(router, '_get_client', return_value=mock_client):
            from src.core.model_router import TaskType
            result = await router._fallback_complete(
                task_type=TaskType.VISUAL_COMPARISON,
                messages=[{"role": "user", "content": "Describe"}],
                images=[b'\x89PNG'],
                max_tokens=4096,
                temperature=0.0,
                json_mode=False,
                tools=None,
                original_error="Primary failed",
            )

            assert result["content"] == "Fallback vision"
            assert result["fallback"] is True
            assert result["original_error"] == "Primary failed"

    @pytest.mark.asyncio
    async def test_fallback_complete_without_vision(self, mock_env_vars):
        """Test _fallback_complete without vision."""
        from src.core.model_router import ModelRouter

        router = ModelRouter()

        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value={
            "content": "Fallback text",
            "input_tokens": 100,
            "output_tokens": 50,
            "model": "sonnet",
        })

        with patch.object(router, '_get_client', return_value=mock_client):
            from src.core.model_router import TaskType
            result = await router._fallback_complete(
                task_type=TaskType.TEXT_EXTRACTION,
                messages=[{"role": "user", "content": "Hello"}],
                images=None,
                max_tokens=4096,
                temperature=0.5,
                json_mode=True,
                tools=[{"name": "tool"}],
                original_error="Primary failed",
            )

            assert result["content"] == "Fallback text"
            assert result["fallback"] is True

    def test_track_usage_new_model(self, mock_env_vars):
        """Test _track_usage for a new model."""
        from src.core.model_router import ModelRouter

        router = ModelRouter()

        result = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.005,
        }

        router._track_usage("test-model", result)

        assert "test-model" in router.usage_stats
        assert router.usage_stats["test-model"]["calls"] == 1
        assert router.usage_stats["test-model"]["total_input_tokens"] == 100

    def test_track_usage_existing_model(self, mock_env_vars):
        """Test _track_usage for an existing model."""
        from src.core.model_router import ModelRouter

        router = ModelRouter()

        # First call
        router._track_usage("test-model", {
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.005,
        })

        # Second call
        router._track_usage("test-model", {
            "input_tokens": 200,
            "output_tokens": 100,
            "cost": 0.01,
        })

        assert router.usage_stats["test-model"]["calls"] == 2
        assert router.usage_stats["test-model"]["total_input_tokens"] == 300
        assert router.usage_stats["test-model"]["total_output_tokens"] == 150
        assert router.usage_stats["test-model"]["total_cost"] == 0.015

    def test_calculate_savings_no_usage(self, mock_env_vars):
        """Test _calculate_savings with no usage."""
        from src.core.model_router import ModelRouter

        router = ModelRouter()
        savings = router._calculate_savings()

        assert savings["actual_cost"] == 0.0
        assert savings["sonnet_equivalent"] == 0.0
        assert savings["saved"] == 0.0

    def test_calculate_savings_with_usage(self, mock_env_vars):
        """Test _calculate_savings with usage data."""
        from src.core.model_router import ModelRouter

        router = ModelRouter()

        # Add cheap model usage
        router.usage_stats["llama-3.1-8b"] = {
            "calls": 100,
            "total_input_tokens": 400000,  # 400K tokens
            "total_output_tokens": 200000,
            "total_cost": 0.036,  # Very cheap
        }

        savings = router._calculate_savings()

        # Should show savings vs using Sonnet
        assert savings["actual_cost"] == 0.036
        assert savings["sonnet_equivalent"] > 0
        assert savings["saved"] > 0
        assert savings["savings_percent"] > 0


class TestCostEstimation:
    """Tests for cost estimation utilities."""

    def test_estimate_monthly_costs(self, mock_env_vars):
        """Test monthly cost estimation."""
        from src.core.model_router import estimate_monthly_costs

        costs = estimate_monthly_costs(
            daily_tests=100,
            tests_with_vision=0.3,
            avg_tokens_per_test=6000,
        )

        assert "all_sonnet" in costs
        assert "multi_model_aggressive" in costs
        assert "multi_model_balanced" in costs

        # Multi-model should be cheaper than all Sonnet
        assert costs["multi_model_aggressive"]["total"] < costs["all_sonnet"]["total"]

    def test_estimate_monthly_costs_zero_tests(self, mock_env_vars):
        """Test monthly cost estimation with zero tests."""
        from src.core.model_router import estimate_monthly_costs

        costs = estimate_monthly_costs(daily_tests=0)

        assert costs["all_sonnet"]["total"] == 0
        assert costs["multi_model_aggressive"]["savings_percent"] == 0

    def test_create_router(self, mock_env_vars):
        """Test router creation helper."""
        from src.core.model_router import create_router, ModelProvider

        router = create_router(prefer_anthropic=True, cost_limit=0.05)

        assert router.prefer_provider == ModelProvider.ANTHROPIC
        assert router.cost_limit == 0.05

    def test_create_router_default(self, mock_env_vars):
        """Test router creation with defaults."""
        from src.core.model_router import create_router

        router = create_router()

        assert router.prefer_provider is None
        assert router.cost_limit == 0.10
        assert router.enable_fallback is True
