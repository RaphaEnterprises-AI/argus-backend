"""
Dynamic Model Discovery Service - Fetches models from provider APIs.

This module provides real-time model discovery from:
- Anthropic API (Claude models)
- OpenRouter API (400+ models from all providers)

The service caches results and falls back to static definitions if APIs are unavailable.

Usage:
    from src.core.model_discovery import get_available_models, discover_models

    # Get cached models (instant, uses cache)
    models = await get_available_models()

    # Force refresh from APIs
    models = await discover_models(force_refresh=True)
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import httpx

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Cache TTL (1 hour)
CACHE_TTL_SECONDS = 3600


class ModelProvider(str, Enum):
    """Model provider identifiers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    META = "meta"
    COHERE = "cohere"
    XAI = "xai"


@dataclass
class DiscoveredModel:
    """A model discovered from a provider API."""
    id: str  # Full API model ID (e.g., "claude-sonnet-4-5-20250514")
    provider: ModelProvider
    display_name: str
    created_at: datetime | None = None
    context_length: int = 128000
    max_output_tokens: int = 4096

    # Pricing per million tokens (if available)
    input_price: float | None = None
    output_price: float | None = None

    # Capabilities
    supports_vision: bool = False
    supports_tools: bool = False
    supports_computer_use: bool = False
    supports_json_mode: bool = False

    # Additional metadata
    description: str = ""
    is_deprecated: bool = False

    @property
    def short_id(self) -> str:
        """Get short model ID without version/date suffix."""
        # claude-sonnet-4-5-20250514 -> claude-sonnet-4-5
        # gpt-4o-2024-08-06 -> gpt-4o
        parts = self.id.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) >= 6:
            return parts[0]
        return self.id


@dataclass
class ModelCache:
    """Cache for discovered models."""
    models: dict[str, DiscoveredModel] = field(default_factory=dict)
    last_updated: float = 0

    def is_valid(self) -> bool:
        """Check if cache is still valid."""
        return time.time() - self.last_updated < CACHE_TTL_SECONDS

    def update(self, models: list[DiscoveredModel]) -> None:
        """Update cache with new models."""
        self.models = {m.id: m for m in models}
        self.last_updated = time.time()


# Global cache instance
_model_cache = ModelCache()


async def fetch_anthropic_models(api_key: str | None = None) -> list[DiscoveredModel]:
    """
    Fetch available models from Anthropic API.

    API: GET https://api.anthropic.com/v1/models
    Docs: https://docs.anthropic.com/en/api/models

    Returns:
        List of discovered Anthropic models
    """
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, skipping Anthropic model discovery")
        return []

    models: list[DiscoveredModel] = []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                params={"limit": 100},
            )
            response.raise_for_status()
            data = response.json()

            for model_data in data.get("data", []):
                model_id = model_data.get("id", "")
                display_name = model_data.get("display_name", model_id)

                # Parse created_at timestamp
                created_at = None
                if model_data.get("created_at"):
                    try:
                        created_at = datetime.fromisoformat(
                            model_data["created_at"].replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass

                # Determine capabilities based on model name
                is_claude_4 = "claude-4" in model_id or "claude-sonnet-4" in model_id or "claude-opus-4" in model_id or "claude-haiku-4" in model_id
                is_claude_3_5 = "claude-3-5" in model_id or "claude-3.5" in model_id

                model = DiscoveredModel(
                    id=model_id,
                    provider=ModelProvider.ANTHROPIC,
                    display_name=display_name,
                    created_at=created_at,
                    context_length=200000 if is_claude_4 or is_claude_3_5 else 100000,
                    max_output_tokens=8192 if is_claude_4 else 4096,
                    supports_vision=is_claude_4 or is_claude_3_5,
                    supports_tools=is_claude_4 or is_claude_3_5,
                    supports_computer_use=is_claude_4 or "sonnet" in model_id,
                    supports_json_mode=True,
                )
                models.append(model)
                logger.debug(f"Discovered Anthropic model: {model_id}")

            logger.info(f"Discovered {len(models)} Anthropic models")

    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch Anthropic models: {e}")
    except Exception as e:
        logger.error(f"Error processing Anthropic models response: {e}")

    return models


async def fetch_openrouter_models(api_key: str | None = None) -> list[DiscoveredModel]:
    """
    Fetch available models from OpenRouter API.

    API: GET https://openrouter.ai/api/v1/models
    Docs: https://openrouter.ai/docs/api/api-reference/models/get-models

    Returns:
        List of discovered models from all providers via OpenRouter
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")

    models: list[DiscoveredModel] = []

    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            for model_data in data.get("data", []):
                model_id = model_data.get("id", "")

                # Determine provider from model ID
                provider = _detect_provider(model_id)

                # Parse pricing (OpenRouter uses per-token pricing)
                pricing = model_data.get("pricing", {})
                input_price = float(pricing.get("prompt", 0)) * 1_000_000  # Per million
                output_price = float(pricing.get("completion", 0)) * 1_000_000

                model = DiscoveredModel(
                    id=model_id,
                    provider=provider,
                    display_name=model_data.get("name", model_id),
                    context_length=model_data.get("context_length", 128000),
                    max_output_tokens=model_data.get("top_provider", {}).get("max_completion_tokens", 4096),
                    input_price=input_price,
                    output_price=output_price,
                    supports_vision="image" in str(model_data.get("architecture", {}).get("modality", "")),
                    supports_tools=model_data.get("supported_generation_methods", []) != [],
                    description=model_data.get("description", ""),
                )
                models.append(model)

            logger.info(f"Discovered {len(models)} models from OpenRouter")

    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch OpenRouter models: {e}")
    except Exception as e:
        logger.error(f"Error processing OpenRouter models response: {e}")

    return models


def _detect_provider(model_id: str) -> ModelProvider:
    """Detect provider from model ID."""
    model_id_lower = model_id.lower()

    if "anthropic" in model_id_lower or "claude" in model_id_lower:
        return ModelProvider.ANTHROPIC
    elif "openai" in model_id_lower or "gpt" in model_id_lower or model_id_lower.startswith("o1"):
        return ModelProvider.OPENAI
    elif "google" in model_id_lower or "gemini" in model_id_lower:
        return ModelProvider.GOOGLE
    elif "groq" in model_id_lower:
        return ModelProvider.GROQ
    elif "together" in model_id_lower:
        return ModelProvider.TOGETHER
    elif "fireworks" in model_id_lower:
        return ModelProvider.FIREWORKS
    elif "deepseek" in model_id_lower:
        return ModelProvider.DEEPSEEK
    elif "mistral" in model_id_lower:
        return ModelProvider.MISTRAL
    elif "meta" in model_id_lower or "llama" in model_id_lower:
        return ModelProvider.META
    elif "cohere" in model_id_lower or "command" in model_id_lower:
        return ModelProvider.COHERE
    elif "xai" in model_id_lower or "grok" in model_id_lower:
        return ModelProvider.XAI
    else:
        return ModelProvider.OPENAI  # Default fallback


async def discover_models(
    force_refresh: bool = False,
    include_openrouter: bool = True,
) -> dict[str, DiscoveredModel]:
    """
    Discover available models from provider APIs.

    Args:
        force_refresh: Force refresh even if cache is valid
        include_openrouter: Include OpenRouter models (400+ models)

    Returns:
        Dictionary of model_id -> DiscoveredModel
    """
    global _model_cache

    # Return cache if valid and not forcing refresh
    if not force_refresh and _model_cache.is_valid():
        logger.debug("Returning cached models")
        return _model_cache.models

    logger.info("Discovering models from provider APIs...")

    # Fetch from APIs in parallel
    tasks = [fetch_anthropic_models()]
    if include_openrouter:
        tasks.append(fetch_openrouter_models())

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_models: list[DiscoveredModel] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Model discovery failed: {result}")
        elif isinstance(result, list):
            all_models.extend(result)

    # Update cache
    _model_cache.update(all_models)

    logger.info(f"Total models discovered: {len(all_models)}")
    return _model_cache.models


async def get_available_models() -> dict[str, DiscoveredModel]:
    """
    Get available models (uses cache, refreshes if stale).

    This is the main function to call for model listings.
    """
    return await discover_models(force_refresh=False)


async def get_anthropic_models() -> list[DiscoveredModel]:
    """Get only Anthropic Claude models."""
    models = await get_available_models()
    return [m for m in models.values() if m.provider == ModelProvider.ANTHROPIC]


async def get_model_by_id(model_id: str) -> DiscoveredModel | None:
    """Get a specific model by ID."""
    models = await get_available_models()
    return models.get(model_id)


async def get_latest_model(
    provider: ModelProvider = ModelProvider.ANTHROPIC,
    tier: str = "sonnet",
) -> DiscoveredModel | None:
    """
    Get the latest model for a provider and tier.

    Args:
        provider: Model provider
        tier: Model tier (e.g., "sonnet", "opus", "haiku" for Anthropic)

    Returns:
        Latest model matching criteria, or None
    """
    models = await get_available_models()

    matching = [
        m for m in models.values()
        if m.provider == provider and tier.lower() in m.id.lower()
    ]

    if not matching:
        return None

    # Sort by creation date (newest first) or by ID (higher versions first)
    matching.sort(
        key=lambda m: (m.created_at or datetime.min, m.id),
        reverse=True,
    )

    return matching[0]


def get_cached_models() -> dict[str, DiscoveredModel]:
    """Get currently cached models (synchronous, may be empty/stale)."""
    return _model_cache.models


def clear_cache() -> None:
    """Clear the model cache."""
    global _model_cache
    _model_cache = ModelCache()
    logger.info("Model cache cleared")


# Startup helper
async def warm_cache() -> None:
    """Warm the model cache on application startup."""
    logger.info("Warming model discovery cache...")
    try:
        await discover_models(force_refresh=True)
    except Exception as e:
        logger.error(f"Failed to warm model cache: {e}")
