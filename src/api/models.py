"""Models API endpoints.

Provides unified access to AI models across all configured providers:
- GET /api/v1/models - List all models with filtering
- GET /api/v1/providers - List all providers with status
- GET /api/v1/providers/{provider_id}/status - Get single provider status

This API allows the dashboard to display available models and providers,
helping users select the right model for their tasks.
"""

import os
import threading
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.core.model_registry import (
    Capability,
    ModelConfig,
    ModelRegistry,
    Provider,
    get_model_registry,
)
from src.core.model_router import MODELS as ROUTER_MODELS
from src.core.model_router import ModelConfig as RouterModelConfig
from src.core.model_router import ModelProvider

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["Models"])

# Cache for model list (refreshed periodically)
_model_cache: dict[str, Any] = {}
_cache_timestamp: datetime | None = None
_cache_lock = threading.Lock()  # Thread-safe cache updates
CACHE_TTL_SECONDS = 300  # 5 minutes


# ============================================================================
# Response Models
# ============================================================================


class ModelCapability(BaseModel):
    """Model capability information."""

    id: str
    name: str
    description: str


class ModelTier(BaseModel):
    """Model tier classification."""

    id: str
    name: str
    description: str
    price_range: str


class ModelResponse(BaseModel):
    """Model information for API response."""

    id: str
    model_id: str
    display_name: str
    provider: str
    provider_name: str

    # Pricing (per 1M tokens)
    input_price: float
    output_price: float

    # Tier classification
    tier: str
    tier_name: str

    # Capabilities
    capabilities: list[str]
    supports_vision: bool
    supports_tools: bool
    supports_computer_use: bool
    supports_json_mode: bool
    supports_thinking: bool

    # Limits
    max_tokens: int
    context_window: int

    # Metadata
    latency_ms: int
    is_deprecated: bool = False
    is_available: bool = True  # Based on platform/user keys


class ModelListResponse(BaseModel):
    """Paginated list of models."""

    models: list[ModelResponse]
    total: int
    limit: int
    offset: int
    providers: list[str]
    capabilities: list[str]
    tiers: list[str]


class ProviderResponse(BaseModel):
    """Provider information for API response."""

    id: str
    name: str
    description: str
    website: str
    key_url: str
    models_count: int
    has_platform_key: bool
    has_user_key: bool = False  # Populated when user context available
    env_var_name: str


class ProviderListResponse(BaseModel):
    """List of providers."""

    providers: list[ProviderResponse]
    total: int


class ProviderStatusResponse(BaseModel):
    """Detailed provider status."""

    id: str
    name: str
    is_configured: bool
    has_platform_key: bool
    has_user_key: bool
    key_source: str | None  # "platform", "byok", or None
    models_count: int
    available_models: list[str]
    last_validated_at: str | None = None
    validation_error: str | None = None


# ============================================================================
# Provider Metadata
# ============================================================================

PROVIDER_METADATA = {
    "anthropic": {
        "name": "Anthropic",
        "description": "Creator of Claude AI models with industry-leading safety and capabilities",
        "website": "https://anthropic.com",
        "key_url": "https://console.anthropic.com/settings/keys",
        "env_var": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "name": "OpenAI",
        "description": "Pioneering AI research lab behind GPT and o1 models",
        "website": "https://openai.com",
        "key_url": "https://platform.openai.com/api-keys",
        "env_var": "OPENAI_API_KEY",
    },
    "google": {
        "name": "Google",
        "description": "Google DeepMind's Gemini family of multimodal models",
        "website": "https://ai.google.dev",
        "key_url": "https://aistudio.google.com/app/apikey",
        "env_var": "GOOGLE_API_KEY",
    },
    "groq": {
        "name": "Groq",
        "description": "Ultra-fast inference with custom LPU hardware",
        "website": "https://groq.com",
        "key_url": "https://console.groq.com/keys",
        "env_var": "GROQ_API_KEY",
    },
    "together": {
        "name": "Together AI",
        "description": "Open source model hosting with competitive pricing",
        "website": "https://together.ai",
        "key_url": "https://api.together.xyz/settings/api-keys",
        "env_var": "TOGETHER_API_KEY",
    },
    "openrouter": {
        "name": "OpenRouter",
        "description": "Unified API gateway for 300+ models from all providers",
        "website": "https://openrouter.ai",
        "key_url": "https://openrouter.ai/keys",
        "env_var": "OPENROUTER_API_KEY",
    },
    "deepseek": {
        "name": "DeepSeek",
        "description": "High-quality reasoning models at exceptional value",
        "website": "https://deepseek.com",
        "key_url": "https://platform.deepseek.com/api_keys",
        "env_var": "DEEPSEEK_API_KEY",
    },
    "cerebras": {
        "name": "Cerebras",
        "description": "World's fastest AI inference using wafer-scale chips",
        "website": "https://cerebras.ai",
        "key_url": "https://cloud.cerebras.ai/platform",
        "env_var": "CEREBRAS_API_KEY",
    },
    "vertex_ai": {
        "name": "Vertex AI",
        "description": "Claude models via Google Cloud with enterprise features",
        "website": "https://cloud.google.com/vertex-ai",
        "key_url": "https://console.cloud.google.com/apis/credentials",
        "env_var": "GOOGLE_CLOUD_PROJECT",
    },
}

TIER_METADATA = {
    "flash": {
        "name": "Flash",
        "description": "Fastest and cheapest models for simple tasks",
        "price_range": "$0.05-0.30 per 1M tokens",
    },
    "value": {
        "name": "Value",
        "description": "Great quality at competitive prices",
        "price_range": "$0.14-0.80 per 1M tokens",
    },
    "standard": {
        "name": "Standard",
        "description": "Balanced performance for most tasks",
        "price_range": "$1.00-3.00 per 1M tokens",
    },
    "premium": {
        "name": "Premium",
        "description": "High-capability models for complex tasks",
        "price_range": "$3.00-15.00 per 1M tokens",
    },
    "expert": {
        "name": "Expert",
        "description": "Most powerful models for demanding tasks",
        "price_range": "$15.00+ per 1M tokens",
    },
}

CAPABILITY_METADATA = {
    "vision": {
        "name": "Vision",
        "description": "Can analyze and understand images",
    },
    "tool_use": {
        "name": "Tool Use",
        "description": "Can use tools and function calling",
    },
    "computer_use": {
        "name": "Computer Use",
        "description": "Can control browsers and desktop applications",
    },
    "json_mode": {
        "name": "JSON Mode",
        "description": "Guaranteed valid JSON output",
    },
    "thinking": {
        "name": "Extended Thinking",
        "description": "Deep reasoning with chain-of-thought",
    },
    "extended_context": {
        "name": "Extended Context",
        "description": "200K+ token context window",
    },
    "code_generation": {
        "name": "Code Generation",
        "description": "Optimized for writing code",
    },
    "fast_inference": {
        "name": "Fast Inference",
        "description": "Sub-second response times",
    },
}


# ============================================================================
# Helper Functions
# ============================================================================


def _get_tier_from_price(input_price: float, output_price: float) -> str:
    """Classify model tier based on pricing."""
    avg_price = (input_price + output_price) / 2

    if avg_price < 0.30:
        return "flash"
    elif avg_price < 1.00:
        return "value"
    elif avg_price < 3.50:
        return "standard"
    elif avg_price < 15.00:
        return "premium"
    else:
        return "expert"


def _normalize_provider(provider: str | ModelProvider | Provider) -> str:
    """Normalize provider to string ID."""
    if isinstance(provider, (ModelProvider, Provider)):
        return provider.value.lower()
    return str(provider).lower()


def _get_provider_name(provider_id: str) -> str:
    """Get display name for provider."""
    meta = PROVIDER_METADATA.get(provider_id, {})
    return meta.get("name", provider_id.title())


def _check_platform_key(provider_id: str) -> bool:
    """Check if platform has API key configured for provider."""
    meta = PROVIDER_METADATA.get(provider_id, {})
    env_var = meta.get("env_var", "")
    return bool(os.environ.get(env_var))


def _build_model_list() -> list[ModelResponse]:
    """Build unified model list from both registries."""
    models: list[ModelResponse] = []
    seen_model_ids: set[str] = set()

    # Get models from model_router.py (primary source with more models)
    for model_key, config in ROUTER_MODELS.items():
        if config.model_id in seen_model_ids:
            continue
        seen_model_ids.add(config.model_id)

        provider_id = _normalize_provider(config.provider)
        tier = _get_tier_from_price(config.input_cost_per_1m, config.output_cost_per_1m)

        # Build capabilities list
        capabilities = []
        if config.supports_vision:
            capabilities.append("vision")
        if config.supports_tools:
            capabilities.append("tool_use")
        if config.supports_computer_use:
            capabilities.append("computer_use")
        if config.supports_json_mode:
            capabilities.append("json_mode")
        if config.supports_thinking:
            capabilities.append("thinking")
        if config.context_window >= 200000:
            capabilities.append("extended_context")
        if config.latency_ms < 500:
            capabilities.append("fast_inference")

        models.append(
            ModelResponse(
                id=model_key,
                model_id=config.model_id,
                display_name=model_key.replace("-", " ").replace("_", " ").title(),
                provider=provider_id,
                provider_name=_get_provider_name(provider_id),
                input_price=config.input_cost_per_1m,
                output_price=config.output_cost_per_1m,
                tier=tier,
                tier_name=TIER_METADATA.get(tier, {}).get("name", tier.title()),
                capabilities=capabilities,
                supports_vision=config.supports_vision,
                supports_tools=config.supports_tools,
                supports_computer_use=config.supports_computer_use,
                supports_json_mode=config.supports_json_mode,
                supports_thinking=config.supports_thinking,
                max_tokens=config.max_tokens,
                context_window=config.context_window,
                latency_ms=config.latency_ms,
                is_available=_check_platform_key(provider_id),
            )
        )

    # Add any models from model_registry.py not already included
    try:
        registry = get_model_registry()
        for model in registry.list_all_models():
            if model.model_id in seen_model_ids:
                continue
            seen_model_ids.add(model.model_id)

            provider_id = _normalize_provider(model.provider)
            tier = _get_tier_from_price(model.input_price, model.output_price)

            capabilities = [cap.value for cap in model.capabilities]

            models.append(
                ModelResponse(
                    id=model.model_id.replace(".", "-").replace("/", "-"),
                    model_id=model.model_id,
                    display_name=model.display_name,
                    provider=provider_id,
                    provider_name=_get_provider_name(provider_id),
                    input_price=model.input_price,
                    output_price=model.output_price,
                    tier=tier,
                    tier_name=TIER_METADATA.get(tier, {}).get("name", tier.title()),
                    capabilities=capabilities,
                    supports_vision=Capability.VISION in model.capabilities,
                    supports_tools=Capability.TOOL_USE in model.capabilities,
                    supports_computer_use=Capability.COMPUTER_USE in model.capabilities,
                    supports_json_mode=Capability.JSON_MODE in model.capabilities,
                    supports_thinking=False,  # Registry doesn't track this
                    max_tokens=model.max_tokens,
                    context_window=model.context_window,
                    latency_ms=1000,  # Registry doesn't track this
                    is_deprecated=model.is_deprecated,
                    is_available=_check_platform_key(provider_id),
                )
            )
    except Exception as e:
        logger.warning("Failed to load models from registry", error=str(e))

    return models


def _get_cached_models() -> list[ModelResponse]:
    """Get models from cache or rebuild.

    Uses a threading lock to prevent race conditions when multiple
    concurrent requests trigger cache rebuild simultaneously.
    """
    global _model_cache, _cache_timestamp

    now = datetime.now(UTC)

    # Check if cache is valid without acquiring lock (fast path)
    if (
        _cache_timestamp is not None
        and (now - _cache_timestamp).total_seconds() <= CACHE_TTL_SECONDS
        and "models" in _model_cache
    ):
        return _model_cache["models"]

    # Cache miss - acquire lock and rebuild
    with _cache_lock:
        # Double-check after acquiring lock (another thread may have rebuilt)
        if (
            _cache_timestamp is not None
            and (now - _cache_timestamp).total_seconds() <= CACHE_TTL_SECONDS
            and "models" in _model_cache
        ):
            return _model_cache["models"]

        _model_cache["models"] = _build_model_list()
        _cache_timestamp = now
        logger.debug("Rebuilt model cache", model_count=len(_model_cache["models"]))

    return _model_cache["models"]


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    request: Request,
    provider: str | None = Query(None, description="Filter by provider ID"),
    capability: str | None = Query(None, description="Filter by capability"),
    tier: str | None = Query(None, description="Filter by tier"),
    search: str | None = Query(None, description="Search by name or model ID"),
    supports_vision: bool | None = Query(None, description="Filter for vision support"),
    supports_computer_use: bool | None = Query(None, description="Filter for computer use"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Results offset"),
):
    """List all available AI models with optional filtering.

    Returns a paginated list of models from all configured providers,
    with filtering options for provider, capabilities, tier, and search.

    The response includes:
    - Model details (ID, name, provider, pricing)
    - Capabilities (vision, tools, computer use, etc.)
    - Tier classification (flash, value, standard, premium, expert)
    - Availability based on configured API keys
    """
    models = _get_cached_models()

    # Apply filters
    filtered = models

    if provider:
        provider_lower = provider.lower()
        filtered = [m for m in filtered if m.provider == provider_lower]

    if capability:
        capability_lower = capability.lower()
        filtered = [m for m in filtered if capability_lower in m.capabilities]

    if tier:
        tier_lower = tier.lower()
        filtered = [m for m in filtered if m.tier == tier_lower]

    if search:
        search_lower = search.lower()
        filtered = [
            m for m in filtered
            if search_lower in m.display_name.lower()
            or search_lower in m.model_id.lower()
            or search_lower in m.provider_name.lower()
        ]

    if supports_vision is not None:
        filtered = [m for m in filtered if m.supports_vision == supports_vision]

    if supports_computer_use is not None:
        filtered = [m for m in filtered if m.supports_computer_use == supports_computer_use]

    # Sort by tier (cheapest first), then by name
    tier_order = {"flash": 0, "value": 1, "standard": 2, "premium": 3, "expert": 4}
    filtered.sort(key=lambda m: (tier_order.get(m.tier, 5), m.display_name))

    # Get unique values for filter options
    all_providers = sorted(set(m.provider for m in models))
    all_capabilities = sorted(set(cap for m in models for cap in m.capabilities))
    all_tiers = sorted(set(m.tier for m in models), key=lambda t: tier_order.get(t, 5))

    # Paginate
    total = len(filtered)
    paginated = filtered[offset:offset + limit]

    return ModelListResponse(
        models=paginated,
        total=total,
        limit=limit,
        offset=offset,
        providers=all_providers,
        capabilities=all_capabilities,
        tiers=all_tiers,
    )


@router.get("/providers", response_model=ProviderListResponse)
async def list_providers(request: Request):
    """List all AI providers with their status.

    Returns information about each configured provider including:
    - Provider metadata (name, description, website)
    - Number of available models
    - Whether platform API key is configured
    - Link to get API keys
    """
    models = _get_cached_models()

    # Count models per provider
    provider_model_counts: dict[str, int] = {}
    for model in models:
        provider_model_counts[model.provider] = provider_model_counts.get(model.provider, 0) + 1

    providers: list[ProviderResponse] = []

    for provider_id, meta in PROVIDER_METADATA.items():
        providers.append(
            ProviderResponse(
                id=provider_id,
                name=meta["name"],
                description=meta["description"],
                website=meta["website"],
                key_url=meta["key_url"],
                models_count=provider_model_counts.get(provider_id, 0),
                has_platform_key=_check_platform_key(provider_id),
                env_var_name=meta["env_var"],
            )
        )

    # Sort: configured providers first, then by model count
    providers.sort(key=lambda p: (not p.has_platform_key, -p.models_count, p.name))

    return ProviderListResponse(
        providers=providers,
        total=len(providers),
    )


@router.get("/providers/{provider_id}/status", response_model=ProviderStatusResponse)
async def get_provider_status(request: Request, provider_id: str):
    """Get detailed status for a specific provider.

    Returns:
    - Configuration status (platform key, user key)
    - List of available models from this provider
    - Validation status if applicable
    """
    provider_id = provider_id.lower()

    if provider_id not in PROVIDER_METADATA:
        raise HTTPException(
            status_code=404,
            detail=f"Provider '{provider_id}' not found. Available providers: {list(PROVIDER_METADATA.keys())}",
        )

    meta = PROVIDER_METADATA[provider_id]
    models = _get_cached_models()

    # Filter models for this provider
    provider_models = [m for m in models if m.provider == provider_id]
    available_model_ids = [m.model_id for m in provider_models]

    has_platform_key = _check_platform_key(provider_id)

    # Determine key source
    key_source = None
    if has_platform_key:
        key_source = "platform"

    return ProviderStatusResponse(
        id=provider_id,
        name=meta["name"],
        is_configured=has_platform_key,
        has_platform_key=has_platform_key,
        has_user_key=False,  # Would need user context to check
        key_source=key_source,
        models_count=len(provider_models),
        available_models=available_model_ids,
    )


@router.get("/models/capabilities", response_model=list[ModelCapability])
async def list_capabilities():
    """List all model capabilities with descriptions.

    Returns the list of capabilities that can be used for filtering models.
    """
    return [
        ModelCapability(
            id=cap_id,
            name=meta["name"],
            description=meta["description"],
        )
        for cap_id, meta in CAPABILITY_METADATA.items()
    ]


@router.get("/models/tiers", response_model=list[ModelTier])
async def list_tiers():
    """List all model tiers with descriptions.

    Returns the tier classifications used for categorizing models.
    """
    return [
        ModelTier(
            id=tier_id,
            name=meta["name"],
            description=meta["description"],
            price_range=meta["price_range"],
        )
        for tier_id, meta in TIER_METADATA.items()
    ]


@router.get("/models/{model_id}")
async def get_model(request: Request, model_id: str):
    """Get details for a specific model.

    The model_id can be either the short key (e.g., "sonnet") or
    the full model ID (e.g., "anthropic/claude-sonnet-4").
    """
    models = _get_cached_models()

    # Try exact match on id or model_id
    model_id_lower = model_id.lower()
    for model in models:
        if model.id.lower() == model_id_lower or model.model_id.lower() == model_id_lower:
            return model

    # Try partial match
    for model in models:
        if model_id_lower in model.model_id.lower():
            return model

    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_id}' not found",
    )
