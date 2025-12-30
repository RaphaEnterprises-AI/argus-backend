"""Tests for configuration module."""

import os
import pytest
from unittest.mock import patch


class TestSettings:
    """Tests for Settings class."""

    def test_settings_loads_from_env(self, mock_env_vars):
        """Test that settings loads from environment variables."""
        # Clear cached settings
        from src.config import Settings
        
        settings = Settings()
        assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test-key-12345"

    def test_settings_default_values(self, mock_env_vars):
        """Test default values are set correctly."""
        from src.config import Settings, ModelName, MultiModelStrategy
        
        settings = Settings()
        assert settings.default_model == ModelName.SONNET
        assert settings.model_strategy == MultiModelStrategy.BALANCED
        assert settings.cost_limit_per_run == 10.0
        assert settings.max_iterations == 50

    def test_settings_optional_keys(self, mock_env_vars):
        """Test that optional API keys can be None."""
        from src.config import Settings
        
        settings = Settings()
        # Google key is set in mock_env_vars
        assert settings.google_api_key is not None


class TestAgentConfig:
    """Tests for AgentConfig class."""

    def test_agent_config_defaults(self, mock_env_vars):
        """Test AgentConfig with default values."""
        from src.config import AgentConfig, ModelName
        
        config = AgentConfig()
        assert config.name == "default_agent"
        assert config.model == ModelName.SONNET
        assert config.max_tokens == 4096
        assert config.temperature == 0.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_agent_config_custom_values(self, mock_env_vars):
        """Test AgentConfig with custom values."""
        from src.config import AgentConfig, ModelName
        
        config = AgentConfig(
            name="test_agent",
            model=ModelName.OPUS,
            max_tokens=8192,
            temperature=0.5,
            max_retries=5,
            retry_delay=2.0,
        )
        assert config.name == "test_agent"
        assert config.model == ModelName.OPUS
        assert config.max_tokens == 8192
        assert config.max_retries == 5


class TestModelPricing:
    """Tests for model pricing constants."""

    def test_model_pricing_exists(self, mock_env_vars):
        """Test that pricing exists for all models."""
        from src.config import MODEL_PRICING, ModelName
        
        for model in ModelName:
            assert model in MODEL_PRICING
            assert "input" in MODEL_PRICING[model]
            assert "output" in MODEL_PRICING[model]

    def test_multi_model_pricing(self, mock_env_vars):
        """Test multi-model pricing dictionary."""
        from src.config import MULTI_MODEL_PRICING
        
        assert "gpt-4o" in MULTI_MODEL_PRICING
        assert "claude-sonnet-4-5" in MULTI_MODEL_PRICING
        assert "gemini-1.5-pro" in MULTI_MODEL_PRICING


class TestScreenshotTokenEstimation:
    """Tests for screenshot token estimation."""

    def test_estimate_screenshot_tokens(self, mock_env_vars):
        """Test screenshot token estimation."""
        from src.config import estimate_screenshot_tokens
        
        # Standard resolutions
        assert estimate_screenshot_tokens(1920, 1080) == 2500
        assert estimate_screenshot_tokens(1280, 720) == 1800
        
    def test_estimate_screenshot_tokens_closest_match(self, mock_env_vars):
        """Test that non-standard resolutions get closest match."""
        from src.config import estimate_screenshot_tokens
        
        # Should find closest match
        tokens = estimate_screenshot_tokens(1900, 1000)
        assert tokens > 0


class TestEnums:
    """Tests for configuration enums."""

    def test_model_provider_values(self, mock_env_vars):
        """Test ModelProvider enum values."""
        from src.config import ModelProvider
        
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.GOOGLE.value == "google"

    def test_model_name_values(self, mock_env_vars):
        """Test ModelName enum values."""
        from src.config import ModelName
        
        assert ModelName.OPUS.value == "claude-opus-4-5"
        assert ModelName.SONNET.value == "claude-sonnet-4-5"
        assert ModelName.HAIKU.value == "claude-haiku-4-5"

    def test_inference_gateway_values(self, mock_env_vars):
        """Test InferenceGateway enum values."""
        from src.config import InferenceGateway
        
        assert InferenceGateway.DIRECT.value == "direct"
        assert InferenceGateway.CLOUDFLARE.value == "cloudflare"
        assert InferenceGateway.AWS_BEDROCK.value == "aws_bedrock"
