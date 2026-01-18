"""Tests for RouterAgent - Intelligent Model Routing."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.router_agent import (
    RouterAgent,
    RouterAgentConfig,
    RoutingDecision,
    TaskContext,
    route_with_agent,
)
from src.core.model_router import MODELS, TaskComplexity, TaskType


class TestRouterAgentConfig:
    """Tests for RouterAgentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RouterAgentConfig()

        assert config.router_model == "llama-small"
        assert "gemini-flash" in config.router_model_fallbacks
        assert config.custom_system_prompt is None
        assert config.trivial_threshold_tokens == 500
        assert config.max_routing_cost == 0.001
        assert config.enable_learning is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = RouterAgentConfig(
            router_model="gemini-flash",
            router_model_fallbacks=["haiku"],
            custom_system_prompt="Custom prompt",
            trivial_threshold_tokens=1000,
            enable_learning=False,
        )

        assert config.router_model == "gemini-flash"
        assert config.router_model_fallbacks == ["haiku"]
        assert config.custom_system_prompt == "Custom prompt"
        assert config.trivial_threshold_tokens == 1000
        assert config.enable_learning is False


class TestTaskContext:
    """Tests for TaskContext dataclass."""

    def test_minimal_context(self):
        """Test minimal task context."""
        context = TaskContext(prompt="Test prompt")

        assert context.prompt == "Test prompt"
        assert context.task_type is None
        assert context.has_images is False
        assert context.has_tools is False
        assert context.max_cost is None

    def test_full_context(self):
        """Test full task context."""
        context = TaskContext(
            prompt="Complex test prompt",
            task_type=TaskType.CODE_ANALYSIS,
            has_images=True,
            has_tools=True,
            num_tokens_estimate=8000,
            max_latency_ms=2000,
            max_cost=0.10,
        )

        assert context.task_type == TaskType.CODE_ANALYSIS
        assert context.has_images is True
        assert context.has_tools is True
        assert context.num_tokens_estimate == 8000


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_routing_decision(self):
        """Test RoutingDecision creation."""
        decision = RoutingDecision(
            model_name="sonnet",
            model_config=MODELS["sonnet"],
            reasoning="Complex task needs strong reasoning",
            confidence=0.95,
            fallback_model="opus",
            estimated_cost=0.05,
        )

        assert decision.model_name == "sonnet"
        assert decision.confidence == 0.95
        assert decision.fallback_model == "opus"


class TestRouterAgent:
    """Tests for RouterAgent class."""

    @pytest.fixture
    def router(self, mock_env_vars):
        """Create a RouterAgent for testing."""
        return RouterAgent()

    @pytest.fixture
    def custom_router(self, mock_env_vars):
        """Create a RouterAgent with custom config."""
        return RouterAgent(config=RouterAgentConfig(
            router_model="haiku",
            enable_learning=False,
        ))

    def test_locked_tasks(self, router):
        """Test that locked tasks are defined correctly."""
        assert TaskType.COMPUTER_USE_SIMPLE in router.LOCKED_TASKS
        assert TaskType.COMPUTER_USE_COMPLEX in router.LOCKED_TASKS
        assert router.LOCKED_TASKS[TaskType.COMPUTER_USE_SIMPLE] == "claude-computer-use"

    def test_complexity_tiers(self, router):
        """Test complexity tier definitions."""
        assert TaskComplexity.TRIVIAL in router.COMPLEXITY_TIERS
        assert TaskComplexity.EXPERT in router.COMPLEXITY_TIERS
        assert "llama-small" in router.COMPLEXITY_TIERS[TaskComplexity.TRIVIAL]
        assert "opus" in router.COMPLEXITY_TIERS[TaskComplexity.EXPERT]

    def test_is_trivial_task_short_prompt(self, router):
        """Test trivial task detection with short prompt."""
        context = TaskContext(
            prompt="Extract the email from this text",
            task_type=TaskType.TEXT_EXTRACTION,
        )
        assert router._is_trivial_task(context) is True

    def test_is_trivial_task_long_complex(self, router):
        """Test trivial task detection with long complex prompt."""
        context = TaskContext(
            prompt="x" * 1000,  # Long prompt
            task_type=TaskType.SELF_HEALING,  # Complex task type
        )
        assert router._is_trivial_task(context) is False

    def test_is_trivial_task_keywords(self, router):
        """Test trivial task detection with keywords."""
        context = TaskContext(prompt="classify this element as button or link")
        assert router._is_trivial_task(context) is True

        context = TaskContext(prompt="parse this JSON response")
        assert router._is_trivial_task(context) is True

    @pytest.mark.asyncio
    async def test_execute_locked_task(self, router, mock_env_vars):
        """Test that locked tasks return immediately without LLM call."""
        context = TaskContext(
            prompt="Click the login button",
            task_type=TaskType.COMPUTER_USE_SIMPLE,
        )

        result = await router.execute(context)

        assert result.success is True
        assert result.data.model_name == "claude-computer-use"
        assert result.data.confidence == 1.0
        assert "requires" in result.data.reasoning.lower()

    @pytest.mark.asyncio
    async def test_execute_trivial_task_no_images(self, router, mock_env_vars):
        """Test trivial task without images uses cheap text model."""
        context = TaskContext(
            prompt="Extract the name",
            task_type=TaskType.TEXT_EXTRACTION,
        )

        result = await router.execute(context)

        assert result.success is True
        assert result.data.model_name in ["llama-small", "gemini-flash"]
        assert result.data.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_execute_trivial_task_with_images(self, router, mock_env_vars):
        """Test trivial task with images uses vision model."""
        context = TaskContext(
            prompt="Extract the text",
            task_type=TaskType.TEXT_EXTRACTION,
            has_images=True,
        )

        result = await router.execute(context)

        assert result.success is True
        # Should use gemini-flash (has vision) not llama
        assert result.data.model_name == "gemini-flash"

    @pytest.mark.asyncio
    async def test_execute_complex_task_llm_decision(self, router, mock_env_vars):
        """Test complex task triggers LLM-based routing."""
        # Mock the router client by setting _router_client directly
        mock_result = {
            "content": json.dumps({
                "model": "sonnet",
                "complexity": "complex",
                "reasoning": "Needs strong reasoning for debugging",
                "confidence": 0.9,
                "fallback": "opus",
            }),
            "input_tokens": 100,
            "output_tokens": 50,
        }

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_result)
        router._router_client = mock_client
        router._router_model_name = "llama-3.1-8b"

        context = TaskContext(
            prompt="Debug this complex authentication flow that's failing intermittently " * 20,
            task_type=TaskType.COMPLEX_DEBUGGING,
            has_tools=True,
        )

        result = await router.execute(context)

        assert result.success is True
        assert result.data.model_name == "sonnet"
        assert result.data.confidence == 0.9
        assert result.data.fallback_model == "opus"

    @pytest.mark.asyncio
    async def test_execute_handles_invalid_model_response(self, router, mock_env_vars):
        """Test handling of invalid model name in LLM response."""
        mock_result = {
            "content": json.dumps({
                "model": "nonexistent-model",
                "complexity": "moderate",
                "reasoning": "Test",
                "confidence": 0.8,
            }),
            "input_tokens": 100,
            "output_tokens": 50,
        }

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_result)
        router._router_client = mock_client
        router._router_model_name = "llama-3.1-8b"

        context = TaskContext(
            prompt="Complex task " * 100,
            task_type=TaskType.CODE_ANALYSIS,
        )

        result = await router.execute(context)

        # Should fallback to sonnet for invalid model
        assert result.success is True
        assert result.data.model_name == "sonnet"

    @pytest.mark.asyncio
    async def test_execute_handles_llm_error(self, router, mock_env_vars):
        """Test handling of LLM errors during routing."""
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(side_effect=Exception("API Error"))
        router._router_client = mock_client
        router._router_model_name = "llama-3.1-8b"

        context = TaskContext(
            prompt="Complex task " * 100,
            task_type=TaskType.ROOT_CAUSE_ANALYSIS,
        )

        result = await router.execute(context)

        # Should return safe fallback
        assert result.success is True
        assert result.data.model_name == "sonnet"
        assert "error" in result.data.reasoning.lower() or "fallback" in result.data.reasoning.lower()
        assert result.data.confidence == 0.5

    def test_get_system_prompt_default(self, router):
        """Test default system prompt generation."""
        prompt = router._get_system_prompt()

        # Enhanced prompt uses different terminology
        assert "routing" in prompt.lower()
        assert "agent" in prompt.lower()
        assert "JSON" in prompt or "json" in prompt

    def test_get_system_prompt_custom(self, mock_env_vars):
        """Test custom system prompt is overridden by enhanced prompt."""
        # With enhanced prompts enabled, custom_system_prompt is ignored
        # because enhanced prompts take priority
        custom_prompt = "You are a custom router."
        router = RouterAgent(config=RouterAgentConfig(
            custom_system_prompt=custom_prompt
        ))

        # Enhanced prompt overrides custom prompt
        prompt = router._get_system_prompt()
        assert "routing" in prompt.lower()

    def test_estimate_cost(self, router):
        """Test cost estimation."""
        config = MODELS["sonnet"]
        cost = router._estimate_cost(config, 10000)

        # sonnet: $3/1M input, $15/1M output
        # 10000 tokens * 0.6 input = 6000 → $0.018
        # 10000 tokens * 0.4 output = 4000 → $0.06
        expected = (6000 / 1_000_000) * 3.0 + (4000 / 1_000_000) * 15.0
        assert abs(cost - expected) < 0.001

    def test_track_decision(self, router):
        """Test decision tracking."""
        context = TaskContext(
            prompt="Test",
            task_type=TaskType.CODE_ANALYSIS,
            has_images=True,
        )
        decision = RoutingDecision(
            model_name="gemini-pro",
            model_config=MODELS["gemini-pro"],
            reasoning="Test",
            confidence=0.9,
            estimated_cost=0.05,
        )

        router._track_decision(context, decision)

        assert len(router.decision_history) == 1
        assert router.decision_history[0]["model_selected"] == "gemini-pro"
        assert router.decision_history[0]["has_images"] is True

    def test_record_outcome(self, router):
        """Test outcome recording for learning."""
        router.record_outcome(
            model_name="sonnet",
            task_type=TaskType.CODE_ANALYSIS,
            success=True,
            actual_cost=0.05,
            latency_ms=1200,
        )

        key = "sonnet:code_analysis"
        assert key in router.performance_tracker
        assert router.performance_tracker[key]["calls"] == 1
        assert router.performance_tracker[key]["successes"] == 1

        # Record another outcome
        router.record_outcome(
            model_name="sonnet",
            task_type=TaskType.CODE_ANALYSIS,
            success=False,
            actual_cost=0.04,
            latency_ms=1100,
        )

        assert router.performance_tracker[key]["calls"] == 2
        assert router.performance_tracker[key]["successes"] == 1

    def test_get_performance_report(self, router):
        """Test performance report generation."""
        # Record some outcomes
        router.record_outcome("sonnet", TaskType.CODE_ANALYSIS, True, 0.05, 1000)
        router.record_outcome("sonnet", TaskType.CODE_ANALYSIS, True, 0.04, 900)
        router.record_outcome("haiku", TaskType.TEXT_EXTRACTION, True, 0.001, 500)

        report = router.get_performance_report()

        assert "sonnet:code_analysis" in report
        assert report["sonnet:code_analysis"]["success_rate"] == 1.0
        assert report["sonnet:code_analysis"]["total_calls"] == 2

    def test_find_vision_model(self, router):
        """Test finding vision-capable models by complexity."""
        assert router._find_vision_model("trivial") == "gemini-flash"
        assert router._find_vision_model("simple") == "gpt-4o-mini"
        assert router._find_vision_model("expert") == "sonnet"
        assert router._find_vision_model("unknown") == "gemini-pro"

    def test_build_routing_prompt(self, router):
        """Test routing prompt construction."""
        context = TaskContext(
            prompt="Analyze this code for bugs",
            task_type=TaskType.CODE_ANALYSIS,
            has_images=False,
            has_tools=True,
            num_tokens_estimate=5000,
            max_cost=0.10,
        )

        prompt = router._build_routing_prompt(context)

        assert "code_analysis" in prompt.lower()
        assert "has images: false" in prompt.lower()
        assert "needs tools: true" in prompt.lower()
        assert "5000" in prompt


class TestRouterClientSelection:
    """Tests for router's own model selection."""

    def test_get_router_client_openrouter_available(self, mock_env_vars):
        """Test OpenRouter client selection when available."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "src.core.model_router.OpenRouterClient",
                return_value=MagicMock()
            ):
                router = RouterAgent()
                client, model_name = router._get_router_client()

                # Default router_model is "llama-small" which uses OpenRouter
                assert model_name == "llama-small"

    def test_get_router_client_fallback_models(self, mock_env_vars):
        """Test fallback to other models when primary unavailable."""
        # Set OPENROUTER_API_KEY so OpenRouter models work
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "src.core.model_router.OpenRouterClient",
                return_value=MagicMock()
            ):
                # Use a custom config with different primary model
                config = RouterAgentConfig(
                    router_model="gemini-flash",  # Also OpenRouter
                    router_model_fallbacks=["llama-small"],
                )
                router = RouterAgent(config=config)
                client, model_name = router._get_router_client()

                # Should select gemini-flash as it's the configured primary
                assert model_name == "gemini-flash"

    def test_get_router_client_ultimate_fallback(self, mock_env_vars):
        """Test ultimate fallback to Anthropic when no other providers available."""
        # Create env with only ANTHROPIC_API_KEY (no OPENROUTER_API_KEY)
        test_env = {"ANTHROPIC_API_KEY": "test-key"}

        with patch.dict(os.environ, test_env, clear=True):
            with patch(
                "src.core.model_router.AnthropicClient",
                return_value=MagicMock()
            ):
                router = RouterAgent()
                client, model_name = router._get_router_client()

                # Should fall back to haiku (Anthropic model)
                assert model_name == "haiku"


class TestRouteWithAgentHelper:
    """Tests for route_with_agent helper function."""

    @pytest.mark.asyncio
    async def test_route_with_agent_basic(self, mock_env_vars):
        """Test basic usage of route_with_agent helper."""
        router = RouterAgent()

        # Mock execute to return a known decision
        with patch.object(router, 'execute') as mock_execute:
            mock_execute.return_value = MagicMock(
                success=True,
                data=RoutingDecision(
                    model_name="sonnet",
                    model_config=MODELS["sonnet"],
                    reasoning="Test",
                    confidence=0.9,
                ),
            )

            decision = await route_with_agent(
                router=router,
                prompt="Test prompt",
                task_type=TaskType.CODE_ANALYSIS,
            )

            assert decision.model_name == "sonnet"

    @pytest.mark.asyncio
    async def test_route_with_agent_with_images(self, mock_env_vars):
        """Test route_with_agent with images."""
        router = RouterAgent()

        with patch.object(router, 'execute') as mock_execute:
            mock_execute.return_value = MagicMock(
                success=True,
                data=RoutingDecision(
                    model_name="gemini-pro",
                    model_config=MODELS["gemini-pro"],
                    reasoning="Vision required",
                    confidence=0.95,
                ),
            )

            await route_with_agent(
                router=router,
                prompt="Analyze this screenshot",
                images=[b"fake-image-data"],
            )

            # Verify context was built correctly
            call_args = mock_execute.call_args
            context = call_args[0][0]
            assert context.has_images is True
