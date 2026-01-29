"""Configuration management for E2E Testing Agent."""

from enum import Enum

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProvider(str, Enum):
    """Supported model providers."""
    # Primary providers
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"

    # Cloud platform providers
    VERTEX_AI = "vertex_ai"  # Claude via Google Cloud (unified GCP billing)
    AZURE_OPENAI = "azure_openai"  # OpenAI via Azure
    AWS_BEDROCK = "aws_bedrock"  # Claude/Llama via AWS Bedrock

    # Inference providers (fast/cheap)
    GROQ = "groq"  # Fast Llama inference
    TOGETHER = "together"  # Open models
    FIREWORKS = "fireworks"  # Fast inference
    CEREBRAS = "cerebras"  # Ultra-fast inference

    # Specialized providers
    OPENROUTER = "openrouter"  # Multi-model router
    DEEPSEEK = "deepseek"  # DeepSeek models
    MISTRAL = "mistral"  # Mistral models
    PERPLEXITY = "perplexity"  # Search-augmented models
    COHERE = "cohere"  # Enterprise NLP
    XAI = "xai"  # Grok models


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
        extra="ignore",
        populate_by_name=True
    )

    # ==========================================================================
    # API Keys - Multi-Provider Support (RAP-202)
    # ==========================================================================
    # Made optional for health checks; validated at runtime when needed

    # Primary providers
    anthropic_api_key: SecretStr | None = Field(None, description="Anthropic API key")
    openai_api_key: SecretStr | None = Field(None, description="OpenAI API key (for GPT-4)")
    google_api_key: SecretStr | None = Field(None, description="Google API key (for Gemini)")

    # Inference providers (fast/cheap)
    groq_api_key: SecretStr | None = Field(None, description="Groq API key (for fast Llama)")
    together_api_key: SecretStr | None = Field(None, description="Together API key (for open models)")
    fireworks_api_key: SecretStr | None = Field(None, description="Fireworks API key (for fast inference)")
    cerebras_api_key: SecretStr | None = Field(None, description="Cerebras API key (for ultra-fast inference)")

    # Multi-model router
    openrouter_api_key: SecretStr | None = Field(None, description="OpenRouter API key (multi-model router)")

    # Specialized providers
    deepseek_api_key: SecretStr | None = Field(None, description="DeepSeek API key")
    mistral_api_key: SecretStr | None = Field(None, description="Mistral API key")
    perplexity_api_key: SecretStr | None = Field(None, description="Perplexity API key (search-augmented)")
    cohere_api_key: SecretStr | None = Field(None, description="Cohere API key (enterprise NLP)")
    xai_api_key: SecretStr | None = Field(None, description="xAI API key (Grok models)")

    # Cloud platform providers
    azure_openai_api_key: SecretStr | None = Field(None, description="Azure OpenAI API key")
    azure_openai_endpoint: str | None = Field(None, description="Azure OpenAI endpoint URL")
    aws_bedrock_region: str | None = Field(None, description="AWS Bedrock region (e.g., us-east-1, us-west-2)")

    # Vertex AI Configuration (Claude via Google Cloud)
    # Benefits: Unified GCP billing, committed spend, enterprise features, Computer Use supported
    use_vertex_ai: bool = Field(False, description="Use Vertex AI for Claude models instead of direct API")
    google_cloud_project: str | None = Field(None, description="GCP project ID for Vertex AI")
    vertex_ai_region: str = Field("global", description="Vertex AI region ('global' or specific like 'us-east1')")

    # Integration API Keys
    github_token: SecretStr | None = Field(None, description="GitHub token for PR integration")

    # Inference Gateway Configuration (Cloudflare AI Gateway recommended)
    inference_gateway: InferenceGateway = Field(
        InferenceGateway.CLOUDFLARE,
        description="Which gateway to route AI requests through"
    )
    cloudflare_account_id: str | None = Field(None, description="Cloudflare account ID for AI Gateway")
    cloudflare_gateway_id: str | None = Field(None, description="Cloudflare AI Gateway ID")
    aws_region: str | None = Field("us-east-1", description="AWS region for Bedrock")

    # Database (optional)
    database_url: str | None = Field(None, description="PostgreSQL connection string")

    # Supabase Configuration (for Quality Intelligence)
    supabase_url: str | None = Field(None, description="Supabase project URL")
    supabase_service_key: SecretStr | None = Field(None, description="Supabase service role key")

    # Cloudflare Storage Configuration (for artifacts, caching, memory)
    cloudflare_api_token: SecretStr | None = Field(None, description="Cloudflare API token for R2/KV/Vectorize access")
    cloudflare_r2_bucket: str | None = Field(
        "argus-artifacts",
        description="Cloudflare R2 bucket for screenshots and artifacts"
    )
    cloudflare_kv_namespace_id: str | None = Field(
        "e1f3cdefb05c43b88528adb515bde16a",
        description="Cloudflare KV namespace ID (from wrangler.toml)"
    )
    cloudflare_vectorize_index: str | None = Field(
        "argus-patterns",
        description="Cloudflare Vectorize index name for self-healing memory"
    )
    cloudflare_d1_database_id: str | None = Field(
        None,
        description="Cloudflare D1 database ID for test history"
    )
    # R2 S3-compatible credentials for presigned URLs
    cloudflare_r2_access_key_id: str | None = Field(
        None,
        description="Cloudflare R2 access key ID for S3-compatible API (presigned URLs)"
    )
    cloudflare_r2_secret_access_key: SecretStr | None = Field(
        None,
        description="Cloudflare R2 secret access key for S3-compatible API"
    )
    cloudflare_r2_presigned_url_expiry: int = Field(
        3600,
        description="Presigned URL expiry in seconds (default: 1 hour)"
    )
    cache_enabled: bool = Field(True, description="Enable caching layer")
    cache_ttl_quality_scores: int = Field(300, description="TTL for quality scores in seconds (default: 5 min)")
    cache_ttl_llm_responses: int = Field(86400, description="TTL for LLM responses in seconds (default: 24 hours)")
    cache_ttl_healing_patterns: int = Field(604800, description="TTL for healing patterns in seconds (default: 7 days)")

    # Media URL Signing (screenshots/videos)
    # Must match MEDIA_SIGNING_SECRET in Cloudflare Worker
    cloudflare_media_signing_secret: SecretStr | None = Field(
        None,
        description="HMAC secret for signed media URLs (screenshots/videos)"
    )
    cloudflare_media_url_expiry: int = Field(
        900,
        description="Signed URL expiry in seconds (default: 15 minutes)"
    )
    cloudflare_worker_url: str = Field(
        "https://argus-api.samuelvinay-kumar.workers.dev",
        description="Cloudflare Worker URL for screenshot/video access"
    )

    # Notifications (optional)
    slack_webhook_url: str | None = Field(None, description="Slack webhook for notifications")

    # Multi-Model Configuration
    model_strategy: MultiModelStrategy = Field(
        MultiModelStrategy.BALANCED,
        description="Model routing strategy"
    )
    prefer_provider: ModelProvider | None = Field(
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
    parallel_threshold: int = Field(5, description="Minimum tests to trigger parallel execution")
    retry_failed_tests: int = Field(2, description="Number of retries for failed tests")
    self_heal_enabled: bool = Field(True, description="Enable automatic test healing")
    self_heal_confidence_threshold: float = Field(0.8, description="Min confidence for auto-heal")

    # Human-in-the-Loop Approval Settings
    require_healing_approval: bool = Field(
        default=False,
        description="Require human approval before applying self-healing"
    )
    require_test_plan_approval: bool = Field(
        default=False,
        description="Require human approval of test plan before execution"
    )
    require_human_approval_for_healing: bool = Field(
        default=False,
        description="Alias for require_healing_approval (backwards compatibility)"
    )
    approval_timeout_seconds: int = Field(
        default=300,
        description="Timeout in seconds for approval requests"
    )

    # Paths
    output_dir: str = Field("./test-results", description="Directory for test outputs")
    screenshot_dir: str = Field("./test-results/screenshots", description="Directory for screenshots")

    # Server Settings (for webhook mode)
    server_host: str = Field("0.0.0.0", description="Server host")
    server_port: int = Field(8000, description="Server port")

    # ==========================================================================
    # Browser Pool Configuration (Vultr K8s - Primary)
    # ==========================================================================
    # The BrowserPoolClient checks URLs in this order:
    # 1. BROWSER_POOL_URL (Vultr K8s) - Primary, production-grade, scalable
    # 2. BROWSER_WORKER_URL (Cloudflare) - Legacy fallback
    # 3. http://localhost:8080 - Local development

    browser_pool_url: str | None = Field(
        None,
        description="URL of the Vultr K8s browser pool (primary). Set via BROWSER_POOL_URL env var."
    )
    browser_pool_jwt_secret: SecretStr | None = Field(
        None,
        description="JWT secret for browser pool authentication. Set via BROWSER_POOL_JWT_SECRET env var."
    )
    browser_timeout_ms: int = Field(
        60000,
        description="Default timeout for browser operations in milliseconds"
    )
    browser_retry_count: int = Field(
        3,
        description="Number of retries for failed browser operations"
    )
    vision_fallback_enabled: bool = Field(
        True,
        description="Enable Claude Computer Use as fallback when DOM execution fails"
    )

    # Legacy: Cloudflare Worker (fallback if BROWSER_POOL_URL not set)
    browser_worker_url: str = Field(
        "https://argus-api.samuelvinay-kumar.workers.dev",
        description="URL of the Cloudflare browser worker (legacy fallback). Use BROWSER_POOL_URL instead."
    )

    # ==========================================================================
    # Selenium Grid Configuration (Video Recording)
    # ==========================================================================
    # Selenium Grid is a separate system from browser-pool, used specifically for
    # video recording. It runs with ffmpeg sidecars that auto-upload to R2.
    selenium_grid_url: str | None = Field(
        None,
        description="URL of the Selenium Grid hub (e.g., http://65.20.71.218:4444). Used for video recording."
    )

    # ==========================================================================
    # Security Settings (SOC2 Compliance)
    # ==========================================================================

    # Authentication
    enforce_authentication: bool = Field(
        False,  # Set to True in production
        description="Enforce authentication on all endpoints"
    )
    jwt_secret_key: SecretStr | None = Field(
        None,
        description="Secret key for JWT token signing (required for auth)"
    )
    jwt_expiration_hours: int = Field(24, description="JWT token expiration in hours")
    jwt_refresh_expiration_days: int = Field(30, description="JWT refresh token expiration in days")

    # Rate Limiting
    rate_limiting_enabled: bool = Field(True, description="Enable rate limiting")
    rate_limit_requests: int = Field(60, description="Max requests per window")
    rate_limit_window_seconds: int = Field(60, description="Rate limit window in seconds")

    # CORS Security - accepts comma-separated string or JSON array
    cors_allowed_origins_raw: str = Field(
        default="*",
        alias="cors_allowed_origins",
        description="Allowed CORS origins (comma-separated or JSON array)"
    )

    @property
    def cors_allowed_origins(self) -> list[str]:
        """Parse CORS origins from comma-separated string or JSON array."""
        v = self.cors_allowed_origins_raw
        if not v or not v.strip():
            return ["*"]
        # If it looks like JSON array, try to parse it
        if v.strip().startswith("["):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        # Otherwise treat as comma-separated
        return [origin.strip() for origin in v.split(",") if origin.strip()]

    # Security Headers
    enable_hsts: bool = Field(True, description="Enable HSTS header")
    enable_csp: bool = Field(True, description="Enable Content-Security-Policy header")
    hsts_max_age: int = Field(31536000, description="HSTS max-age in seconds (1 year)")

    # Audit Logging
    audit_logging_enabled: bool = Field(True, description="Enable comprehensive audit logging")
    audit_log_request_body: bool = Field(False, description="Log request bodies (sensitive)")
    audit_log_response_body: bool = Field(False, description="Log response bodies (sensitive)")
    audit_log_retention_days: int = Field(365, description="Audit log retention in days (SOC2: min 1 year)")

    # Input Validation
    max_request_body_size: int = Field(10485760, description="Max request body size in bytes (10MB)")
    max_string_length: int = Field(10000, description="Max string field length")
    max_array_length: int = Field(1000, description="Max array length in requests")

    # Session Security
    session_timeout_minutes: int = Field(60, description="Session timeout in minutes")
    max_concurrent_sessions: int = Field(5, description="Max concurrent sessions per user")

    # API Versioning
    api_version: str = Field("v1", description="Current API version")
    api_version_header: str = Field("X-API-Version", description="API version header name")
    deprecation_warning_enabled: bool = Field(True, description="Enable deprecation warnings")

    # ==========================================================================
    # Sentry Configuration (Error Monitoring & Performance)
    # ==========================================================================
    sentry_dsn: str | None = Field(
        None,
        description="Sentry DSN for error tracking. Get from Sentry project settings."
    )
    sentry_environment: str = Field(
        "development",
        description="Sentry environment (development, staging, production)"
    )
    sentry_traces_sample_rate: float = Field(
        0.1,
        description="Sample rate for performance monitoring (0.0 to 1.0). Use 0.1 for production."
    )
    sentry_profiles_sample_rate: float = Field(
        0.1,
        description="Sample rate for profiling (0.0 to 1.0). Requires traces to be enabled."
    )
    sentry_send_default_pii: bool = Field(
        False,
        description="Send Personally Identifiable Information to Sentry. Disable for GDPR compliance."
    )
    sentry_debug: bool = Field(
        False,
        description="Enable Sentry debug mode for troubleshooting integration issues."
    )

    # ==========================================================================
    # OAuth Configuration (Third-party Integrations)
    # ==========================================================================

    # GitHub OAuth App
    github_client_id: str | None = Field(
        None,
        description="GitHub OAuth App Client ID"
    )
    github_client_secret: SecretStr | None = Field(
        None,
        description="GitHub OAuth App Client Secret"
    )

    # Slack OAuth App
    slack_client_id: str | None = Field(
        None,
        description="Slack OAuth App Client ID"
    )
    slack_client_secret: SecretStr | None = Field(
        None,
        description="Slack OAuth App Client Secret"
    )

    # Jira OAuth 2.0 (3LO)
    jira_client_id: str | None = Field(
        None,
        description="Jira OAuth 2.0 (3LO) Client ID"
    )
    jira_client_secret: SecretStr | None = Field(
        None,
        description="Jira OAuth 2.0 (3LO) Client Secret"
    )

    # Linear OAuth
    linear_client_id: str | None = Field(
        None,
        description="Linear OAuth App Client ID"
    )
    linear_client_secret: SecretStr | None = Field(
        None,
        description="Linear OAuth App Client Secret"
    )

    # OAuth Token Encryption Key (AES-256-GCM)
    oauth_encryption_key: SecretStr | None = Field(
        None,
        description="32-byte key for encrypting OAuth tokens (base64 encoded). Generate with: openssl rand -base64 32"
    )

    # OAuth Redirect Base URL (for constructing callback URLs)
    oauth_redirect_base_url: str = Field(
        "http://localhost:3000",
        description="Base URL for OAuth redirect callbacks (e.g., https://app.heyargus.com)"
    )

    # ==========================================================================
    # Redpanda Configuration (Event Streaming)
    # ==========================================================================
    redpanda_brokers: str = Field(
        "redpanda-kafka.argus-data.svc.cluster.local:9092",
        description="Comma-separated list of Redpanda broker addresses"
    )
    redpanda_sasl_username: str | None = Field(
        None,
        description="SASL username for Redpanda authentication"
    )
    redpanda_sasl_password: SecretStr | None = Field(
        None,
        description="SASL password for Redpanda authentication"
    )
    redpanda_schema_registry_url: str | None = Field(
        None,
        description="URL of the Redpanda Schema Registry"
    )

    # ==========================================================================
    # FalkorDB Configuration (Knowledge Graphs)
    # ==========================================================================
    falkordb_host: str = Field(
        "falkordb.argus-data.svc.cluster.local",
        description="FalkorDB host address"
    )
    falkordb_port: int = Field(
        6379,
        description="FalkorDB port"
    )
    falkordb_password: SecretStr | None = Field(
        None,
        description="FalkorDB password"
    )

    # ==========================================================================
    # Valkey Configuration (Cache - Redis Replacement)
    # ==========================================================================
    valkey_url: str = Field(
        "redis://valkey.argus-data.svc.cluster.local:6379",
        description="Valkey connection URL (Redis-compatible)"
    )

    # ==========================================================================
    # Upstash Redis Configuration (Preferred fallback - 227x cheaper than KV)
    # ==========================================================================
    # Upstash Redis is used when Valkey (K8s) is unreachable (e.g., from Railway)
    # Much better than Cloudflare KV: 10-30ms latency, full Redis features, cheap
    # Cost comparison (1M reads + 100K writes): KV=$500/mo, Upstash=$2.20/mo
    upstash_redis_rest_url: str | None = Field(
        None,
        description="Upstash Redis REST URL (e.g., https://xxx.upstash.io)"
    )
    upstash_redis_rest_token: str | None = Field(
        None,
        description="Upstash Redis REST API token"
    )

    # ==========================================================================
    # Cognee Configuration (AI Memory)
    # ==========================================================================
    cognee_env: str = Field(
        "production",
        description="Cognee environment (development, staging, production)"
    )
    vector_db_provider: str = Field(
        "pgvector",
        description="Vector database provider for Cognee (pgvector, qdrant, etc.)"
    )
    graph_database_provider: str = Field(
        "falkordb",
        description="Graph database provider for Cognee (falkordb, neo4j, etc.)"
    )
    llm_model: str = Field(
        "anthropic/claude-sonnet-4",
        description="LLM model for Cognee analysis (via OpenRouter)"
    )


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
