"""Configuration management for E2E Testing Agent."""

from enum import Enum
from typing import Optional, Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProvider(str, Enum):
    """Supported model providers."""
    ANTHROPIC = "anthropic"
    VERTEX_AI = "vertex_ai"  # Claude via Google Cloud (unified GCP billing)
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"
    TOGETHER = "together"


class ModelName(str, Enum):
    """Available Claude models (legacy, for backwards compatibility)."""
    OPUS = "claude-opus-4-5"
    SONNET = "claude-sonnet-4-5"
    HAIKU = "claude-haiku-4-5"


class MultiModelStrategy(str, Enum):
    """Model routing strategies."""
    ANTHROPIC_ONLY = "anthropic_only"  # Use only Claude models
    COST_OPTIMIZED = "cost_optimized"  # Use cheapest model for each task
    BALANCED = "balanced"  # Balance cost and quality
    QUALITY_FIRST = "quality_first"  # Use best model, fallback to cheaper


class InferenceGateway(str, Enum):
    """Inference gateway/platform options."""
    DIRECT = "direct"  # Direct API calls to providers
    CLOUDFLARE = "cloudflare"  # Cloudflare AI Gateway (recommended - unified billing, edge caching)
    AWS_BEDROCK = "aws_bedrock"  # AWS Bedrock (enterprise, AWS ecosystem)
    AZURE = "azure"  # Azure OpenAI (Microsoft ecosystem)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys - Multi-Provider Support
    anthropic_api_key: SecretStr = Field(..., description="Anthropic API key")
    openai_api_key: Optional[SecretStr] = Field(None, description="OpenAI API key (for GPT-4)")
    google_api_key: Optional[SecretStr] = Field(None, description="Google API key (for Gemini)")
    groq_api_key: Optional[SecretStr] = Field(None, description="Groq API key (for fast Llama)")
    together_api_key: Optional[SecretStr] = Field(None, description="Together API key (for open models)")

    # Vertex AI Configuration (Claude via Google Cloud)
    # Benefits: Unified GCP billing, committed spend, enterprise features, Computer Use supported
    use_vertex_ai: bool = Field(False, description="Use Vertex AI for Claude models instead of direct API")
    google_cloud_project: Optional[str] = Field(None, description="GCP project ID for Vertex AI")
    vertex_ai_region: str = Field("global", description="Vertex AI region ('global' or specific like 'us-east1')")

    # Integration API Keys
    github_token: Optional[SecretStr] = Field(None, description="GitHub token for PR integration")

    # Inference Gateway Configuration (Cloudflare AI Gateway recommended)
    inference_gateway: InferenceGateway = Field(
        InferenceGateway.CLOUDFLARE,
        description="Which gateway to route AI requests through"
    )
    cloudflare_account_id: Optional[str] = Field(None, description="Cloudflare account ID for AI Gateway")
    cloudflare_gateway_id: Optional[str] = Field(None, description="Cloudflare AI Gateway ID")
    aws_region: Optional[str] = Field("us-east-1", description="AWS region for Bedrock")

    # Database (optional)
    database_url: Optional[str] = Field(None, description="PostgreSQL connection string")

    # Notifications (optional)
    slack_webhook_url: Optional[str] = Field(None, description="Slack webhook for notifications")

    # Multi-Model Configuration
    model_strategy: MultiModelStrategy = Field(
        MultiModelStrategy.BALANCED,
        description="Model routing strategy"
    )
    prefer_provider: Optional[ModelProvider] = Field(
        None,
        description="Preferred provider (if available)"
    )
    enable_model_fallback: bool = Field(
        True,
        description="Fall back to Claude if other providers fail"
    )

    # Legacy Model Configuration (backwards compatibility)
    default_model: ModelName = Field(ModelName.SONNET, description="Default model for testing")
    verification_model: ModelName = Field(ModelName.HAIKU, description="Fast model for verifications")
    debugging_model: ModelName = Field(ModelName.OPUS, description="Powerful model for complex debugging")
    
    # Computer Use Settings
    screenshot_width: int = Field(1920, description="Screenshot width in pixels")
    screenshot_height: int = Field(1080, description="Screenshot height in pixels")
    max_iterations: int = Field(50, description="Max iterations per test")
    action_timeout_ms: int = Field(10000, description="Timeout for UI actions")
    
    # Cost Control
    cost_limit_per_run: float = Field(10.0, description="Max cost per test run in USD")
    cost_limit_per_test: float = Field(1.0, description="Max cost per individual test in USD")
    
    # Execution Settings
    parallel_tests: int = Field(1, description="Number of tests to run in parallel")
    retry_failed_tests: int = Field(2, description="Number of retries for failed tests")
    self_heal_enabled: bool = Field(True, description="Enable automatic test healing")
    self_heal_confidence_threshold: float = Field(0.8, description="Min confidence for auto-heal")
    
    # Paths
    output_dir: str = Field("./test-results", description="Directory for test outputs")
    screenshot_dir: str = Field("./test-results/screenshots", description="Directory for screenshots")
    
    # Server Settings (for webhook mode)
    server_host: str = Field("0.0.0.0", description="Server host")
    server_port: int = Field(8000, description="Server port")


class AgentConfig(BaseSettings):
    """Configuration for individual agents."""

    model_config = SettingsConfigDict(extra="ignore")

    name: str = Field("default_agent", description="Agent name for logging")
    model: ModelName = Field(ModelName.SONNET, description="Model to use")
    max_tokens: int = Field(4096, description="Maximum response tokens")
    temperature: float = Field(0.0, description="Sampling temperature")
    timeout_seconds: int = Field(120, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries on failure")
    retry_delay: float = Field(1.0, description="Base delay between retries in seconds")


# Model pricing (per million tokens) - January 2025
# Claude models (legacy)
MODEL_PRICING = {
    ModelName.OPUS: {"input": 15.00, "output": 75.00},
    ModelName.SONNET: {"input": 3.00, "output": 15.00},
    ModelName.HAIKU: {"input": 0.80, "output": 4.00},
}

# All provider pricing for multi-model routing
MULTI_MODEL_PRICING = {
    # Anthropic
    "claude-opus-4-5": {"input": 15.00, "output": 75.00, "provider": "anthropic"},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00, "provider": "anthropic"},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00, "provider": "anthropic"},

    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00, "provider": "openai"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "provider": "openai"},
    "o1": {"input": 15.00, "output": 60.00, "provider": "openai"},
    "o1-mini": {"input": 3.00, "output": 12.00, "provider": "openai"},

    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00, "provider": "google"},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30, "provider": "google"},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40, "provider": "google"},

    # Groq (fast Llama inference)
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08, "provider": "groq"},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79, "provider": "groq"},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79, "provider": "groq"},

    # Together (open models)
    "deepseek-v3": {"input": 0.27, "output": 1.10, "provider": "together"},
    "qwen-2.5-72b": {"input": 0.60, "output": 0.60, "provider": "together"},
}

# Screenshot token estimates by resolution
SCREENSHOT_TOKENS = {
    (1024, 768): 1500,
    (1280, 720): 1800,
    (1920, 1080): 2500,
    (2560, 1440): 4000,
}


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def estimate_screenshot_tokens(width: int, height: int) -> int:
    """Estimate tokens for a screenshot at given resolution."""
    # Find closest match
    closest = min(
        SCREENSHOT_TOKENS.keys(),
        key=lambda res: abs(res[0] * res[1] - width * height)
    )
    return SCREENSHOT_TOKENS[closest]
