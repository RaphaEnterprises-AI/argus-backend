"""Comprehensive tests for AI Client and Core Infrastructure.

Tests the following components:
1. AI Client Factory (src/core/ai_client.py)
   - Langfuse instrumentation
   - Client factory functions
   - Tracing decorators

2. Model Router (src/core/model_router.py)
   - Task-based model routing
   - Provider failover
   - Budget enforcement
   - Cost tracking

3. Model Discovery (src/core/model_discovery.py)
   - Dynamic model fetching
   - Cache management
   - Provider detection

4. Model Registry (src/core/model_registry.py)
   - Model configuration
   - Task type mapping
   - Capability filtering

5. Provider Base (src/core/providers/base.py)
   - Base provider abstractions
   - Error types
   - Data structures
"""

import asyncio
import os
import time
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ==============================================================================
# AI Client Factory Tests
# ==============================================================================


class TestLangfuseSetup:
    """Test Langfuse initialization and configuration."""

    def test_langfuse_disabled_via_env(self, monkeypatch):
        """Test that Langfuse is disabled when LANGFUSE_ENABLED=false."""
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")

        # Reset the module-level state
        from src.core import ai_client
        ai_client._langfuse_enabled = None
        ai_client._langfuse_client = None

        assert ai_client._is_langfuse_enabled() is False

    def test_langfuse_disabled_missing_credentials(self, monkeypatch):
        """Test that Langfuse is disabled when credentials are missing."""
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        from src.core import ai_client
        ai_client._langfuse_enabled = None
        ai_client._langfuse_client = None

        assert ai_client._is_langfuse_enabled() is False


class TestAnthropicClientFactory:
    """Test Anthropic client factory functions."""

    def test_get_anthropic_client_missing_key_raises(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from src.core.ai_client import get_anthropic_client

        with pytest.raises(ValueError, match="No Anthropic API key"):
            get_anthropic_client(api_key=None)

    def test_get_anthropic_client_with_explicit_key(self, mock_env_vars):
        """Test client creation with explicit API key."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            from src.core.ai_client import get_anthropic_client

            client = get_anthropic_client(api_key="sk-ant-explicit-key")

            mock_anthropic.assert_called_once_with(api_key="sk-ant-explicit-key")
            assert client is not None

    def test_get_async_anthropic_client_missing_key_raises(self, monkeypatch):
        """Test that missing API key raises ValueError for async client."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from src.core.ai_client import get_async_anthropic_client

        with pytest.raises(ValueError, match="No Anthropic API key"):
            get_async_anthropic_client(api_key=None)


class TestOpenAIClientFactory:
    """Test OpenAI client factory functions."""

    def test_get_openai_client_with_custom_base_url(self, mock_env_vars):
        """Test OpenAI client with custom base URL (for OpenRouter, etc.)."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            from src.core.ai_client import get_openai_client

            client = get_openai_client(
                api_key="sk-openai-key",
                base_url="https://custom-api.example.com/v1"
            )

            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "sk-openai-key"
            assert call_kwargs["base_url"] == "https://custom-api.example.com/v1"

    def test_get_openrouter_client(self, mock_env_vars, monkeypatch):
        """Test OpenRouter client factory."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            from src.core.ai_client import get_openrouter_client

            client = get_openrouter_client()

            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"


class TestTraceDecorator:
    """Test the trace_ai_call decorator."""

    @pytest.mark.asyncio
    async def test_trace_decorator_without_langfuse(self, monkeypatch):
        """Test trace decorator when Langfuse is disabled."""
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")

        from src.core import ai_client
        ai_client._langfuse_enabled = None
        ai_client._langfuse_client = None

        from src.core.ai_client import trace_ai_call

        @trace_ai_call(name="test_function")
        async def my_async_func(x):
            return x * 2

        result = await my_async_func(5)
        assert result == 10

    def test_sync_trace_decorator_without_langfuse(self, monkeypatch):
        """Test sync trace decorator when Langfuse is disabled."""
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")

        from src.core import ai_client
        ai_client._langfuse_enabled = None
        ai_client._langfuse_client = None

        from src.core.ai_client import trace_ai_call

        @trace_ai_call(name="test_sync_function")
        def my_sync_func(x):
            return x + 10

        result = my_sync_func(5)
        assert result == 15


# ==============================================================================
# Model Discovery Tests
# ==============================================================================


class TestModelDiscovery:
    """Test dynamic model discovery functionality."""

    def test_model_cache_validity(self):
        """Test ModelCache.is_valid() method."""
        from src.core.model_discovery import ModelCache

        cache = ModelCache()
        assert cache.is_valid() is False  # Empty cache is invalid

        # Set last_updated to now
        cache.last_updated = time.time()
        assert cache.is_valid() is True

        # Set last_updated to 2 hours ago (beyond TTL)
        cache.last_updated = time.time() - 7200
        assert cache.is_valid() is False

    def test_model_cache_update(self):
        """Test ModelCache.update() method."""
        from src.core.model_discovery import DiscoveredModel, ModelCache, ModelProvider

        cache = ModelCache()

        models = [
            DiscoveredModel(
                id="claude-sonnet-4-5",
                provider=ModelProvider.ANTHROPIC,
                display_name="Claude Sonnet 4.5"
            ),
            DiscoveredModel(
                id="gpt-4o",
                provider=ModelProvider.OPENAI,
                display_name="GPT-4o"
            ),
        ]

        cache.update(models)

        assert len(cache.models) == 2
        assert "claude-sonnet-4-5" in cache.models
        assert "gpt-4o" in cache.models
        assert cache.is_valid() is True

    def test_discovered_model_short_id(self):
        """Test DiscoveredModel.short_id property."""
        from src.core.model_discovery import DiscoveredModel, ModelProvider

        # Model with date suffix
        model1 = DiscoveredModel(
            id="claude-sonnet-4-5-20241022",
            provider=ModelProvider.ANTHROPIC,
            display_name="Claude Sonnet"
        )
        assert model1.short_id == "claude-sonnet-4-5"

        # Model without date suffix
        model2 = DiscoveredModel(
            id="gpt-4o",
            provider=ModelProvider.OPENAI,
            display_name="GPT-4o"
        )
        assert model2.short_id == "gpt-4o"

    def test_detect_provider(self):
        """Test _detect_provider function."""
        from src.core.model_discovery import ModelProvider, _detect_provider

        assert _detect_provider("anthropic/claude-sonnet-4") == ModelProvider.ANTHROPIC
        assert _detect_provider("claude-3-opus") == ModelProvider.ANTHROPIC
        assert _detect_provider("openai/gpt-4o") == ModelProvider.OPENAI
        assert _detect_provider("gpt-4-turbo") == ModelProvider.OPENAI
        assert _detect_provider("google/gemini-pro") == ModelProvider.GOOGLE
        assert _detect_provider("groq/llama-3") == ModelProvider.GROQ
        assert _detect_provider("deepseek-chat") == ModelProvider.DEEPSEEK
        assert _detect_provider("mistral-large") == ModelProvider.MISTRAL
        assert _detect_provider("meta-llama/llama-3") == ModelProvider.META
        assert _detect_provider("cohere/command-r") == ModelProvider.COHERE
        assert _detect_provider("xai/grok-1") == ModelProvider.XAI

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        from src.core.model_discovery import DiscoveredModel, ModelProvider, _model_cache, clear_cache

        # Add something to cache first
        _model_cache.models = {"test": MagicMock()}
        _model_cache.last_updated = time.time()

        clear_cache()

        from src.core.model_discovery import _model_cache as fresh_cache
        assert len(fresh_cache.models) == 0
        assert fresh_cache.last_updated == 0

    @pytest.mark.asyncio
    async def test_fetch_anthropic_models_no_key(self, monkeypatch):
        """Test that missing Anthropic key returns empty list."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from src.core.model_discovery import fetch_anthropic_models

        result = await fetch_anthropic_models(api_key=None)
        assert result == []


# ==============================================================================
# Model Registry Tests
# ==============================================================================


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def test_registry_singleton(self, mock_env_vars):
        """Test that get_model_registry returns singleton."""
        from src.core.model_registry import get_model_registry

        registry1 = get_model_registry()
        registry2 = get_model_registry()

        assert registry1 is registry2

    def test_get_model_config(self, mock_env_vars):
        """Test getting a model configuration."""
        from src.core.model_registry import get_model_registry

        registry = get_model_registry()
        model = registry.get_model("claude-sonnet-4-5")

        assert model is not None
        assert model.model_id == "claude-sonnet-4-5"
        assert model.input_price == 3.00
        assert model.output_price == 15.00

    def test_get_model_nonexistent(self, mock_env_vars):
        """Test getting a nonexistent model returns None."""
        from src.core.model_registry import get_model_registry

        registry = get_model_registry()
        model = registry.get_model("nonexistent-model")

        assert model is None

    def test_get_model_for_task_with_capability(self, mock_env_vars):
        """Test getting model for task with required capability."""
        from src.core.model_registry import Capability, TaskType, get_model_registry

        registry = get_model_registry()
        model = registry.get_model_for_task(
            TaskType.VISUAL_COMPARISON,
            required_capability=Capability.VISION
        )

        assert model is not None
        assert model.has_capability(Capability.VISION)

    def test_get_model_for_task_with_budget(self, mock_env_vars):
        """Test getting model for task with budget constraint."""
        from src.core.model_registry import TaskType, get_model_registry

        registry = get_model_registry()

        # Very low budget - should still return something
        model = registry.get_model_for_task(
            TaskType.CLASSIFICATION,
            budget_remaining=0.0001
        )

        assert model is not None

    def test_get_computer_use_models(self, mock_env_vars):
        """Test getting models that support computer use."""
        from src.core.model_registry import Capability, get_model_registry

        registry = get_model_registry()
        models = registry.get_computer_use_models()

        assert len(models) > 0
        for model in models:
            assert model.has_capability(Capability.COMPUTER_USE)

    def test_get_api_model_id(self, mock_env_vars):
        """Test getting full API model ID."""
        from src.core.model_registry import get_api_model_id

        api_id = get_api_model_id("claude-sonnet-4-5")

        # Should return versioned ID
        assert "claude-sonnet-4-5" in api_id
        assert len(api_id) > len("claude-sonnet-4-5")

    def test_model_config_cost_calculation(self, mock_env_vars):
        """Test ModelConfig.cost_per_1k_tokens calculation."""
        from src.core.model_registry import get_model_registry

        registry = get_model_registry()
        model = registry.get_model("claude-sonnet-4-5")

        # 1000 input, 1000 output tokens
        cost = model.cost_per_1k_tokens(1000, 1000)

        # (1000 * 3.00 + 1000 * 15.00) / 1000 = 18.00
        expected = (1000 * 3.00 + 1000 * 15.00) / 1000
        assert cost == expected


# ==============================================================================
# Provider Base Tests
# ==============================================================================


class TestProviderBase:
    """Test provider base classes and utilities."""

    def test_model_info_calculate_cost(self):
        """Test ModelInfo.calculate_cost method."""
        from src.core.providers.base import ModelInfo, ModelTier

        model_info = ModelInfo(
            model_id="test-model",
            provider="test",
            display_name="Test Model",
            input_price_per_1m=3.00,
            output_price_per_1m=15.00,
            context_window=128000,
            max_output=4096,
            tier=ModelTier.PREMIUM
        )

        # Calculate cost for 1M input, 500K output
        cost = model_info.calculate_cost(1_000_000, 500_000)
        expected = (1_000_000 / 1_000_000) * 3.00 + (500_000 / 1_000_000) * 15.00
        assert cost == expected

    def test_chat_message_factory_methods(self):
        """Test ChatMessage factory methods."""
        from src.core.providers.base import ChatMessage

        user_msg = ChatMessage.user("Hello")
        assert user_msg.role == "user"
        assert user_msg.content == "Hello"

        assistant_msg = ChatMessage.assistant("Hi there")
        assert assistant_msg.role == "assistant"
        assert assistant_msg.content == "Hi there"

        system_msg = ChatMessage.system("You are helpful")
        assert system_msg.role == "system"
        assert system_msg.content == "You are helpful"

    def test_chat_response_total_tokens(self):
        """Test ChatResponse.total_tokens property."""
        from src.core.providers.base import ChatResponse

        response = ChatResponse(
            content="Test response",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            finish_reason="stop"
        )

        assert response.total_tokens == 150

    def test_validate_temperature(self):
        """Test temperature validation function."""
        from src.core.providers.base import validate_temperature

        # Normal case
        assert validate_temperature(0.7) == 0.7

        # Below minimum - clamped
        assert validate_temperature(-0.5) == 0.0

        # Above maximum - clamped
        assert validate_temperature(3.0) == 2.0

        # Custom range
        assert validate_temperature(0.5, min_temp=1.0, max_temp=2.0) == 1.0

    def test_mask_api_key(self):
        """Test API key masking function."""
        from src.core.providers.base import mask_api_key

        # Normal key
        assert mask_api_key("sk-ant-api-key-12345678") == "sk-a...5678"

        # Short key - fully masked
        assert mask_api_key("short") == "*****"

        # None - returns placeholder
        assert mask_api_key(None) == "<not set>"

    def test_provider_errors(self):
        """Test provider error classes."""
        from src.core.providers.base import (
            AuthenticationError,
            ContentFilterError,
            ContextLengthError,
            ModelNotFoundError,
            ProviderError,
            QuotaExceededError,
            RateLimitError,
        )

        # Base error
        error = ProviderError("Test error")
        assert str(error) == "Test error"

        # Rate limit with retry_after
        rate_error = RateLimitError("Rate limited", retry_after=30.0)
        assert rate_error.retry_after == 30.0

        # Quota exceeded with reset time
        quota_error = QuotaExceededError("Quota exceeded", reset_at="2024-01-01T00:00:00Z")
        assert quota_error.reset_at == "2024-01-01T00:00:00Z"


# ==============================================================================
# Model Router Budget Tests
# ==============================================================================


class TestModelRouterBudget:
    """Test ModelRouter budget enforcement."""

    @pytest.mark.asyncio
    async def test_budget_exceeded_daily_raises(self, mock_env_vars):
        """Test that exceeding daily budget raises BudgetExceededError."""
        from src.core.model_router import BudgetExceededError, ModelRouter, TaskType

        # Mock the cost tracker
        with patch("src.core.model_router.get_cost_tracker") as mock_tracker:
            mock_tracker_instance = MagicMock()
            mock_tracker.return_value = mock_tracker_instance

            # Mock check_budget to return exceeded status
            from src.services.ai_cost_tracker import BudgetStatus
            mock_tracker_instance.check_budget = AsyncMock(return_value=BudgetStatus(
                has_daily_budget=False,
                has_monthly_budget=True,
                daily_remaining=0,
                monthly_remaining=100,
                daily_limit=10,
                monthly_limit=300,
                daily_used=10,
                monthly_used=200,
            ))

            router = ModelRouter(
                organization_id="org-123",
                enforce_budget=True
            )

            with pytest.raises(BudgetExceededError, match="Daily AI budget exceeded"):
                await router.complete(
                    task_type=TaskType.GENERAL,
                    messages=[{"role": "user", "content": "test"}]
                )

    @pytest.mark.asyncio
    async def test_budget_exceeded_monthly_raises(self, mock_env_vars):
        """Test that exceeding monthly budget raises BudgetExceededError."""
        from src.core.model_router import BudgetExceededError, ModelRouter, TaskType

        with patch("src.core.model_router.get_cost_tracker") as mock_tracker:
            mock_tracker_instance = MagicMock()
            mock_tracker.return_value = mock_tracker_instance

            from src.services.ai_cost_tracker import BudgetStatus
            mock_tracker_instance.check_budget = AsyncMock(return_value=BudgetStatus(
                has_daily_budget=True,
                has_monthly_budget=False,
                daily_remaining=5,
                monthly_remaining=0,
                daily_limit=10,
                monthly_limit=300,
                daily_used=5,
                monthly_used=300,
            ))

            router = ModelRouter(
                organization_id="org-123",
                enforce_budget=True
            )

            with pytest.raises(BudgetExceededError, match="Monthly AI budget exceeded"):
                await router.complete(
                    task_type=TaskType.GENERAL,
                    messages=[{"role": "user", "content": "test"}]
                )

    @pytest.mark.asyncio
    async def test_skip_budget_check(self, mock_env_vars):
        """Test that skip_budget_check bypasses enforcement."""
        from src.core.model_router import ModelRouter, TaskType

        with patch("src.core.model_router.get_cost_tracker") as mock_tracker:
            mock_tracker_instance = MagicMock()
            mock_tracker.return_value = mock_tracker_instance
            mock_tracker_instance.record_usage = AsyncMock()

            router = ModelRouter(
                organization_id="org-123",
                enforce_budget=True
            )

            # Mock the client
            with patch.object(router, '_get_client') as mock_get_client:
                mock_client = AsyncMock()
                mock_client.complete = AsyncMock(return_value={
                    "content": "Response",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "model": "test-model",
                })
                mock_get_client.return_value = mock_client

                # Should not check budget when skip_budget_check=True
                result = await router.complete(
                    task_type=TaskType.GENERAL,
                    messages=[{"role": "user", "content": "test"}],
                    skip_budget_check=True
                )

                assert "content" in result
                # check_budget should not have been called
                mock_tracker_instance.check_budget.assert_not_called()


# ==============================================================================
# Model Router Task Type Mapping Tests
# ==============================================================================


class TestTaskTypeMapping:
    """Test task type to model mapping."""

    def test_all_task_types_have_mappings(self, mock_env_vars):
        """Test that all TaskType values have model mappings."""
        from src.core.model_router import TASK_MODEL_MAPPING, TaskType

        for task_type in TaskType:
            assert task_type in TASK_MODEL_MAPPING, f"Missing mapping for {task_type}"
            assert len(TASK_MODEL_MAPPING[task_type]) > 0, f"Empty mapping for {task_type}"

    def test_trivial_tasks_use_cheap_models_first(self, mock_env_vars):
        """Test that trivial tasks prioritize cheap models."""
        from src.core.model_router import MODELS, TASK_MODEL_MAPPING, TaskType

        trivial_tasks = [
            TaskType.ELEMENT_CLASSIFICATION,
            TaskType.ACTION_EXTRACTION,
            TaskType.SELECTOR_VALIDATION,
            TaskType.TEXT_EXTRACTION,
            TaskType.JSON_PARSING,
        ]

        for task_type in trivial_tasks:
            first_model = TASK_MODEL_MAPPING[task_type][0]
            config = MODELS.get(first_model)
            if config:
                # First model should be relatively cheap (< $1/1M input)
                assert config.input_cost_per_1m < 1.0, \
                    f"Task {task_type} first model {first_model} is expensive"

    def test_computer_use_tasks_use_capable_models(self, mock_env_vars):
        """Test that computer use tasks prioritize capable models."""
        from src.core.model_router import MODELS, TASK_MODEL_MAPPING, TaskType

        computer_use_tasks = [
            TaskType.COMPUTER_USE_SIMPLE,
            TaskType.COMPUTER_USE_COMPLEX,
            TaskType.COMPUTER_USE_MOBILE,
        ]

        for task_type in computer_use_tasks:
            models = TASK_MODEL_MAPPING[task_type]
            # At least one model should support computer use
            has_capable_model = any(
                MODELS.get(m, MagicMock(supports_computer_use=False)).supports_computer_use
                for m in models
            )
            assert has_capable_model, f"Task {task_type} has no computer use capable models"


# ==============================================================================
# Model Config Tests
# ==============================================================================


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_model_config_avg_cost_calculation(self, mock_env_vars):
        """Test average cost per 1K tokens calculation."""
        from src.core.model_router import ModelConfig, ModelProvider

        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="test-model",
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
            max_tokens=4096,
        )

        # Expected: (10 * 0.6 + 30 * 0.4) / 1000 = (6 + 12) / 1000 = 0.018
        expected = (10.0 * 0.6 + 30.0 * 0.4) / 1000
        assert abs(config.avg_cost_per_1k - expected) < 0.0001

    def test_model_config_all_providers(self, mock_env_vars):
        """Test that all providers have at least one model."""
        from src.core.model_router import MODELS, ModelProvider

        # These providers should have models defined
        expected_providers = [
            ModelProvider.OPENROUTER,
            ModelProvider.LOCAL,
            ModelProvider.AWS_BEDROCK,
            ModelProvider.AZURE_OPENAI,
        ]

        for provider in expected_providers:
            has_model = any(m.provider == provider for m in MODELS.values())
            assert has_model, f"No models defined for provider {provider}"


# ==============================================================================
# Integration: Model Selection Flow Tests
# ==============================================================================


class TestModelSelectionFlow:
    """Test the complete model selection flow."""

    def test_select_model_respects_vision_requirement(self, mock_env_vars):
        """Test that vision requirement filters out non-vision models."""
        from src.core.model_router import ModelRouter, TaskType

        router = ModelRouter()

        model_name, config = router.select_model(
            TaskType.VISUAL_COMPARISON,
            requires_vision=True
        )

        assert config.supports_vision is True

    def test_select_model_respects_tools_requirement(self, mock_env_vars):
        """Test that tools requirement filters out non-tool models."""
        from src.core.model_router import ModelRouter, TaskType

        router = ModelRouter()

        model_name, config = router.select_model(
            TaskType.CODE_ANALYSIS,
            requires_tools=True
        )

        assert config.supports_tools is True

    def test_select_model_respects_latency_constraint(self, mock_env_vars):
        """Test that latency constraint is respected."""
        from src.core.model_router import ModelRouter, TaskType

        router = ModelRouter()

        model_name, config = router.select_model(
            TaskType.ELEMENT_CLASSIFICATION,
            max_latency_ms=100  # Very low latency requirement
        )

        # Should either find a fast model or fall back to sonnet
        assert model_name is not None

    def test_select_model_respects_cost_limit(self, mock_env_vars):
        """Test that cost limit is respected."""
        from src.core.model_router import ModelRouter, TaskType

        # Very low cost limit
        router = ModelRouter(cost_limit_per_call=0.0001)

        model_name, config = router.select_model(TaskType.GENERAL)

        # Should still return something (fallback to sonnet if needed)
        assert model_name is not None


# ==============================================================================
# Provider Client Tests
# ==============================================================================


class TestProviderClients:
    """Test provider client instantiation."""

    def test_get_client_openrouter(self, mock_env_vars, monkeypatch):
        """Test getting OpenRouter client."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        from src.core.model_router import ModelProvider, ModelRouter, OpenRouterClient

        with patch("openai.AsyncOpenAI"):
            router = ModelRouter()
            client = router._get_client(ModelProvider.OPENROUTER)

            assert isinstance(client, OpenRouterClient)

    def test_get_client_caching(self, mock_env_vars):
        """Test that clients are cached."""
        from src.core.model_router import ModelProvider, ModelRouter

        with patch("anthropic.AsyncAnthropic"):
            router = ModelRouter()

            client1 = router._get_client(ModelProvider.ANTHROPIC)
            client2 = router._get_client(ModelProvider.ANTHROPIC)

            assert client1 is client2


# ==============================================================================
# Cost Report Tests
# ==============================================================================


class TestCostReport:
    """Test cost reporting functionality."""

    def test_cost_report_empty_usage(self, mock_env_vars):
        """Test cost report with no usage."""
        from src.core.model_router import ModelRouter

        router = ModelRouter()
        report = router.get_cost_report()

        assert report["total_cost"] == 0.0
        assert report["by_model"] == {}

    def test_cost_report_with_usage(self, mock_env_vars):
        """Test cost report with tracked usage."""
        from src.core.model_router import ModelRouter

        router = ModelRouter()

        # Simulate usage
        router._track_usage("sonnet", {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cost": 0.01,
        })
        router._track_usage("deepseek", {
            "input_tokens": 2000,
            "output_tokens": 1000,
            "cost": 0.002,
        })

        report = router.get_cost_report()

        assert report["total_cost"] == 0.012
        assert len(report["by_model"]) == 2
        assert "sonnet" in report["by_model"]
        assert "deepseek" in report["by_model"]

    def test_savings_calculation(self, mock_env_vars):
        """Test savings calculation vs Sonnet baseline."""
        from src.core.model_router import ModelRouter

        router = ModelRouter()

        # Track usage of a cheap model
        router.usage_stats["llama-small"] = {
            "calls": 100,
            "total_input_tokens": 100000,
            "total_output_tokens": 50000,
            "total_cost": 0.01,  # Very cheap
        }

        savings = router._calculate_savings()

        # Should show savings vs Sonnet
        assert savings["actual_cost"] == 0.01
        assert savings["sonnet_equivalent"] > savings["actual_cost"]
        assert savings["saved"] > 0
        assert savings["savings_percent"] > 0


# ==============================================================================
# Estimate Monthly Costs Tests
# ==============================================================================


class TestEstimateMonthyCosts:
    """Test monthly cost estimation utility."""

    def test_estimate_costs_basic(self, mock_env_vars):
        """Test basic monthly cost estimation."""
        from src.core.model_router import estimate_monthly_costs

        costs = estimate_monthly_costs(
            daily_tests=100,
            tests_with_vision=0.2,
            avg_tokens_per_test=5000
        )

        assert "all_sonnet" in costs
        assert "multi_model_aggressive" in costs
        assert "multi_model_balanced" in costs

        # Multi-model should be cheaper than all Sonnet
        assert costs["multi_model_aggressive"]["total"] < costs["all_sonnet"]["total"]

    def test_estimate_costs_zero_tests(self, mock_env_vars):
        """Test estimation with zero daily tests."""
        from src.core.model_router import estimate_monthly_costs

        costs = estimate_monthly_costs(daily_tests=0)

        assert costs["all_sonnet"]["total"] == 0
        assert costs["multi_model_aggressive"]["savings_percent"] == 0
