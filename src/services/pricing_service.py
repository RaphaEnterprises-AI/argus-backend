"""Pricing Service for AI models with OpenRouter integration.

Provides real-time pricing information by:
1. Fetching live pricing from OpenRouter API (400+ models)
2. Caching results for 24 hours
3. Falling back to hardcoded pricing if API unavailable

This hybrid approach ensures:
- Pricing stays current as providers change rates
- Service remains available during API outages
- No blocking on startup
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
import structlog

from src.services.ai_cost_tracker import MODEL_PRICING, ModelPricing

logger = structlog.get_logger()


@dataclass
class ModelInfo:
    """Complete model information including pricing."""

    model_id: str
    display_name: str
    provider: str
    input_price: Decimal  # Per 1M tokens
    output_price: Decimal  # Per 1M tokens
    context_window: int
    max_tokens: int
    capabilities: list[str]
    is_available: bool = True


class PricingService:
    """Service for fetching and caching AI model pricing."""

    OPENROUTER_API = "https://openrouter.ai/api/v1/models"
    CACHE_TTL = timedelta(hours=24)

    # Provider prefixes in OpenRouter model IDs
    PROVIDER_PREFIXES = {
        "anthropic/": "anthropic",
        "openai/": "openai",
        "google/": "google",
        "meta-llama/": "groq",  # Llama models often via Groq
        "deepseek/": "together",
        "groq/": "groq",
        "together/": "together",
    }

    def __init__(self):
        self._cached_models: dict[str, ModelInfo] = {}
        self._last_fetch: datetime | None = None
        self._fetch_lock = asyncio.Lock()
        self._initialized = False

    @property
    def is_cache_fresh(self) -> bool:
        """Check if cache is still valid."""
        if not self._last_fetch:
            return False
        return datetime.now(UTC) - self._last_fetch < self.CACHE_TTL

    async def get_pricing(self, model: str) -> ModelInfo | None:
        """Get pricing for a specific model.

        Args:
            model: Model ID (e.g., "claude-sonnet-4-5", "gpt-4o")

        Returns:
            ModelInfo with pricing, or None if unknown
        """
        await self._ensure_initialized()

        # Try exact match first
        if model in self._cached_models:
            return self._cached_models[model]

        # Try with common prefixes
        for prefix in ["anthropic/", "openai/", "google/"]:
            full_id = f"{prefix}{model}"
            if full_id in self._cached_models:
                return self._cached_models[full_id]

        # Fallback to hardcoded pricing
        if model in MODEL_PRICING:
            pricing = MODEL_PRICING[model]
            return ModelInfo(
                model_id=model,
                display_name=model.replace("-", " ").title(),
                provider=pricing.provider,
                input_price=pricing.input_price,
                output_price=pricing.output_price,
                context_window=128000,  # Default
                max_tokens=4096,
                capabilities=[],
            )

        return None

    async def get_all_models(self) -> list[ModelInfo]:
        """Get all available models with pricing.

        Returns:
            List of ModelInfo for all known models
        """
        await self._ensure_initialized()
        return list(self._cached_models.values())

    async def get_models_by_provider(self, provider: str) -> list[ModelInfo]:
        """Get all models for a specific provider.

        Args:
            provider: Provider name (e.g., "anthropic", "openai")

        Returns:
            List of ModelInfo for that provider
        """
        await self._ensure_initialized()
        return [m for m in self._cached_models.values() if m.provider == provider]

    async def _ensure_initialized(self) -> None:
        """Ensure pricing cache is initialized."""
        if self._initialized and self.is_cache_fresh:
            return

        async with self._fetch_lock:
            # Double-check after acquiring lock
            if self._initialized and self.is_cache_fresh:
                return

            # First load hardcoded pricing as baseline
            self._load_hardcoded_pricing()

            # Then try to fetch from OpenRouter
            await self._fetch_openrouter_pricing()

            self._initialized = True

    def _load_hardcoded_pricing(self) -> None:
        """Load hardcoded pricing as fallback."""
        for model_id, pricing in MODEL_PRICING.items():
            self._cached_models[model_id] = ModelInfo(
                model_id=model_id,
                display_name=self._format_display_name(model_id),
                provider=pricing.provider,
                input_price=pricing.input_price,
                output_price=pricing.output_price,
                context_window=128000,
                max_tokens=4096,
                capabilities=self._infer_capabilities(model_id),
            )

    async def _fetch_openrouter_pricing(self) -> None:
        """Fetch live pricing from OpenRouter API."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(self.OPENROUTER_API)
                response.raise_for_status()
                data = response.json()

                models = data.get("data", [])
                logger.info(
                    "Fetched OpenRouter pricing", model_count=len(models)
                )

                for model in models:
                    try:
                        model_info = self._parse_openrouter_model(model)
                        if model_info:
                            # Store with full OpenRouter ID
                            self._cached_models[model["id"]] = model_info

                            # Also store with short ID for easy lookup
                            short_id = self._extract_short_id(model["id"])
                            if short_id and short_id not in self._cached_models:
                                self._cached_models[short_id] = model_info

                    except Exception as e:
                        logger.debug(
                            "Failed to parse model",
                            model_id=model.get("id"),
                            error=str(e),
                        )

                self._last_fetch = datetime.now(UTC)

        except Exception as e:
            logger.warning(
                "OpenRouter fetch failed, using hardcoded pricing",
                error=str(e),
            )
            # Keep using hardcoded pricing

    def _parse_openrouter_model(self, model: dict[str, Any]) -> ModelInfo | None:
        """Parse OpenRouter model data into ModelInfo."""
        model_id = model.get("id", "")
        pricing = model.get("pricing", {})

        # OpenRouter returns price per token, we store per 1M tokens
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")

        if prompt_price is None or completion_price is None:
            return None

        # Convert to Decimal and scale to per-1M
        try:
            input_price = Decimal(str(prompt_price)) * Decimal("1000000")
            output_price = Decimal(str(completion_price)) * Decimal("1000000")
        except Exception:
            return None

        # Extract provider
        provider = self._extract_provider(model_id)

        # Get context window
        context_window = model.get("context_length", 128000)

        # Get capabilities from model metadata
        capabilities = []
        if model.get("architecture", {}).get("supports_vision"):
            capabilities.append("vision")
        if model.get("architecture", {}).get("supports_tools"):
            capabilities.append("tool_use")

        return ModelInfo(
            model_id=model_id,
            display_name=model.get("name", model_id),
            provider=provider,
            input_price=input_price,
            output_price=output_price,
            context_window=context_window,
            max_tokens=model.get("top_p", 4096),  # OpenRouter uses different field
            capabilities=capabilities,
        )

    def _extract_provider(self, model_id: str) -> str:
        """Extract provider from OpenRouter model ID."""
        for prefix, provider in self.PROVIDER_PREFIXES.items():
            if model_id.startswith(prefix):
                return provider

        # Try to infer from model name
        lower_id = model_id.lower()
        if "claude" in lower_id:
            return "anthropic"
        if "gpt" in lower_id or "o1" in lower_id:
            return "openai"
        if "gemini" in lower_id:
            return "google"
        if "llama" in lower_id:
            return "groq"
        if "deepseek" in lower_id:
            return "together"

        return "unknown"

    def _extract_short_id(self, full_id: str) -> str | None:
        """Extract short model ID from full OpenRouter ID."""
        # "anthropic/claude-sonnet-4-5" -> "claude-sonnet-4-5"
        if "/" in full_id:
            return full_id.split("/", 1)[1]
        return None

    def _format_display_name(self, model_id: str) -> str:
        """Format model ID into display name."""
        # "claude-sonnet-4-5" -> "Claude Sonnet 4.5"
        name = model_id.replace("-", " ").replace("_", " ")
        # Title case but preserve version numbers
        parts = name.split()
        formatted = []
        for part in parts:
            if part.isdigit() or (len(part) <= 3 and any(c.isdigit() for c in part)):
                formatted.append(part)
            else:
                formatted.append(part.title())
        return " ".join(formatted)

    def _infer_capabilities(self, model_id: str) -> list[str]:
        """Infer capabilities from model ID."""
        capabilities = []
        lower_id = model_id.lower()

        # Vision capability
        if any(x in lower_id for x in ["claude", "gpt-4o", "gemini"]):
            capabilities.append("vision")

        # Tool use
        if any(x in lower_id for x in ["claude", "gpt-4", "gemini"]):
            capabilities.append("tool_use")

        # Computer use
        if any(x in lower_id for x in ["claude-sonnet", "claude-opus", "computer-use"]):
            capabilities.append("computer_use")

        # Fast inference
        if any(x in lower_id for x in ["haiku", "flash", "mini", "instant"]):
            capabilities.append("fast")

        return capabilities


# Singleton instance
_pricing_service: PricingService | None = None


def get_pricing_service() -> PricingService:
    """Get or create global pricing service."""
    global _pricing_service
    if _pricing_service is None:
        _pricing_service = PricingService()
    return _pricing_service


async def get_model_pricing(model: str) -> ModelInfo | None:
    """Get pricing for a model (convenience function)."""
    return await get_pricing_service().get_pricing(model)


async def get_all_model_pricing() -> list[ModelInfo]:
    """Get all models with pricing (convenience function)."""
    return await get_pricing_service().get_all_models()
