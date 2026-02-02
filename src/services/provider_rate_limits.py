"""Provider Rate Limit Handling and Fallback.

This module handles rate limits from LLM providers (Anthropic, OpenAI, etc.)
which are separate from our application-level rate limits.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    PROVIDER RATE LIMIT FLOW                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Request                                                        │
│       ↓                                                              │
│  Our Rate Limit (60/min) ✓                                          │
│       ↓                                                              │
│  Provider Rate Limit Check (pre-flight)                             │
│       ↓                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Try Primary Provider (e.g., Anthropic)                      │    │
│  │       ↓                                                      │    │
│  │  429 Rate Limit? ──Yes──→ Parse Retry-After                 │    │
│  │       ↓                   Mark provider cooling              │    │
│  │  Success ✓                      ↓                           │    │
│  │       ↓                 Try Fallback Provider                │    │
│  │  Return Response               (OpenRouter → OpenAI → ...)  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       ↓                                                              │
│  All Providers Exhausted?                                           │
│       ↓                                                              │
│  Return 503 with Retry-After from fastest cooling provider          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Provider Rate Limits (as of 2026):
- Anthropic: 4000 RPM, 400K TPM (tokens/min)
- OpenAI: Varies by tier (500-10K RPM)
- Google Gemini: 60 RPM
- Groq: 30 RPM (free), 6000 RPM (paid)
- Together: 60 RPM (free tier)
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.services.cache import ValkeyClient

logger = structlog.get_logger(__name__)


class Provider(str, Enum):
    """LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"
    TOGETHER = "together"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    FIREWORKS = "fireworks"
    PERPLEXITY = "perplexity"
    COHERE = "cohere"
    XAI = "xai"
    CEREBRAS = "cerebras"


@dataclass
class ProviderQuota:
    """Known rate limits for a provider.

    Used for pre-flight checks to avoid hitting limits.
    """
    rpm: int  # Requests per minute
    tpm: int  # Tokens per minute
    rpd: int | None = None  # Requests per day (some providers)
    concurrent: int | None = None  # Max concurrent requests


# Known provider quotas (platform tier defaults)
PROVIDER_QUOTAS: dict[str, ProviderQuota] = {
    Provider.ANTHROPIC.value: ProviderQuota(rpm=4000, tpm=400000),
    Provider.OPENAI.value: ProviderQuota(rpm=500, tpm=200000),  # Tier 1
    Provider.GOOGLE.value: ProviderQuota(rpm=60, tpm=4000000),
    Provider.GROQ.value: ProviderQuota(rpm=30, tpm=15000),  # Free tier
    Provider.TOGETHER.value: ProviderQuota(rpm=60, tpm=100000),
    Provider.OPENROUTER.value: ProviderQuota(rpm=200, tpm=500000),
    Provider.DEEPSEEK.value: ProviderQuota(rpm=60, tpm=100000),
    Provider.MISTRAL.value: ProviderQuota(rpm=120, tpm=500000),
    Provider.FIREWORKS.value: ProviderQuota(rpm=600, tpm=600000),
    Provider.CEREBRAS.value: ProviderQuota(rpm=60, tpm=60000),
}


@dataclass
class ProviderRateLimitInfo:
    """Information about a provider rate limit event."""
    provider: str
    limit_type: str  # "rpm", "tpm", "rpd", "concurrent"
    retry_after: float  # Seconds until retry
    message: str
    timestamp: float = field(default_factory=time.time)

    @property
    def cooling_until(self) -> float:
        """Unix timestamp when provider can be retried."""
        return self.timestamp + self.retry_after

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "limit_type": self.limit_type,
            "retry_after": self.retry_after,
            "message": self.message,
            "cooling_until": self.cooling_until,
        }


@dataclass
class FallbackResult:
    """Result of a provider fallback attempt."""
    success: bool
    provider_used: str
    original_provider: str
    fallback_reason: str | None = None
    providers_tried: list[str] = field(default_factory=list)


class ProviderRateLimitError(Exception):
    """Raised when a provider rate limit is hit.

    This is distinct from our application rate limits.
    """

    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: float,
        limit_type: str = "unknown",
        is_byok: bool = False,
    ):
        super().__init__(message)
        self.provider = provider
        self.retry_after = retry_after
        self.limit_type = limit_type
        self.is_byok = is_byok

    def to_response_dict(self) -> dict:
        """Convert to API response format."""
        return {
            "error": "provider_rate_limit",
            "detail": str(self),
            "provider": self.provider,
            "retry_after": int(self.retry_after),
            "limit_type": self.limit_type,
            "is_byok": self.is_byok,
            "suggestion": (
                "Your API key has hit its rate limit. Wait and retry."
                if self.is_byok else
                "The AI provider is temporarily rate limited. Please retry shortly."
            ),
        }


class ProviderRateLimitTracker:
    """Tracks provider rate limit states for intelligent fallback.

    Features:
    - Tracks cooling periods per provider per organization
    - Enables pre-flight checks before making requests
    - Supports distributed tracking via Redis/Valkey
    - Provides fallback provider recommendations
    """

    # Fallback order when primary provider is rate limited
    FALLBACK_ORDER: dict[str, list[str]] = {
        # Claude models → try other Claude-capable providers
        Provider.ANTHROPIC.value: [
            Provider.OPENROUTER.value,  # Has Claude via API
            Provider.OPENAI.value,      # Similar capability
            Provider.GOOGLE.value,      # Alternative
        ],
        Provider.OPENAI.value: [
            Provider.OPENROUTER.value,
            Provider.ANTHROPIC.value,
            Provider.GOOGLE.value,
        ],
        Provider.GOOGLE.value: [
            Provider.OPENROUTER.value,
            Provider.ANTHROPIC.value,
            Provider.OPENAI.value,
        ],
        # Default fallback for others
        "default": [
            Provider.OPENROUTER.value,
            Provider.ANTHROPIC.value,
            Provider.OPENAI.value,
        ],
    }

    def __init__(self, key_prefix: str = "provider_limit"):
        """Initialize the tracker.

        Args:
            key_prefix: Prefix for Redis keys
        """
        self._key_prefix = key_prefix
        self._valkey_client: "ValkeyClient | None" = None
        self._valkey_healthy: bool | None = None
        # In-memory fallback: {org_id: {provider: ProviderRateLimitInfo}}
        self._cooling_providers: dict[str, dict[str, ProviderRateLimitInfo]] = defaultdict(dict)
        self._lock = asyncio.Lock()
        self._log = logger.bind(component="provider_rate_tracker")

    async def _get_valkey_client(self) -> "ValkeyClient | None":
        """Get Valkey client with health caching."""
        if self._valkey_client is None:
            from src.services.cache import get_valkey_client
            self._valkey_client = get_valkey_client()

        if self._valkey_client is None:
            return None

        if self._valkey_healthy is False:
            return None

        try:
            healthy = await self._valkey_client.ping()
            self._valkey_healthy = healthy
            return self._valkey_client if healthy else None
        except Exception as e:
            self._valkey_healthy = False
            self._log.warning("Valkey unhealthy", error=str(e))
            return None

    def _get_redis_key(self, org_id: str, provider: str) -> str:
        """Generate Redis key for provider cooling state."""
        return f"{self._key_prefix}:{org_id}:{provider}"

    async def mark_rate_limited(
        self,
        provider: str,
        org_id: str,
        retry_after: float,
        limit_type: str = "unknown",
        message: str = "",
    ) -> None:
        """Mark a provider as rate limited for an organization.

        Args:
            provider: Provider that hit rate limit
            org_id: Organization ID
            retry_after: Seconds until retry
            limit_type: Type of limit hit (rpm, tpm, etc.)
            message: Error message from provider
        """
        info = ProviderRateLimitInfo(
            provider=provider,
            limit_type=limit_type,
            retry_after=retry_after,
            message=message,
        )

        # Store in Redis if available
        client = await self._get_valkey_client()
        if client is not None:
            try:
                redis_client = await client._get_client()
                key = self._get_redis_key(org_id, provider)
                await redis_client.setex(
                    key,
                    int(retry_after) + 5,  # Extra buffer
                    f"{limit_type}:{retry_after}:{message}"
                )
            except Exception as e:
                self._log.warning("Failed to store rate limit in Redis", error=str(e))

        # Always store in memory as fallback
        async with self._lock:
            self._cooling_providers[org_id][provider] = info

        self._log.warning(
            "Provider rate limited",
            provider=provider,
            org_id=org_id,
            retry_after=retry_after,
            limit_type=limit_type,
        )

    async def is_cooling(self, provider: str, org_id: str) -> ProviderRateLimitInfo | None:
        """Check if a provider is currently in cooling period.

        Args:
            provider: Provider to check
            org_id: Organization ID

        Returns:
            ProviderRateLimitInfo if cooling, None if available
        """
        now = time.time()

        # Check Redis first
        client = await self._get_valkey_client()
        if client is not None:
            try:
                redis_client = await client._get_client()
                key = self._get_redis_key(org_id, provider)
                value = await redis_client.get(key)
                if value:
                    parts = value.split(":", 2)
                    if len(parts) == 3:
                        limit_type, retry_after, message = parts
                        ttl = await redis_client.ttl(key)
                        return ProviderRateLimitInfo(
                            provider=provider,
                            limit_type=limit_type,
                            retry_after=float(ttl),
                            message=message,
                        )
            except Exception as e:
                self._log.warning("Failed to check Redis", error=str(e))

        # Check memory fallback
        async with self._lock:
            info = self._cooling_providers.get(org_id, {}).get(provider)
            if info and info.cooling_until > now:
                return info
            elif info:
                # Clean up expired entry
                del self._cooling_providers[org_id][provider]

        return None

    async def get_available_providers(
        self,
        org_id: str,
        preferred: str | None = None,
    ) -> list[str]:
        """Get list of available providers (not cooling).

        Args:
            org_id: Organization ID
            preferred: Preferred provider (put first if available)

        Returns:
            List of available provider names in recommended order
        """
        all_providers = list(PROVIDER_QUOTAS.keys())
        available = []

        for provider in all_providers:
            cooling = await self.is_cooling(provider, org_id)
            if cooling is None:
                available.append(provider)

        # Put preferred first if available
        if preferred and preferred in available:
            available.remove(preferred)
            available.insert(0, preferred)

        return available

    async def get_fallback_providers(
        self,
        failed_provider: str,
        org_id: str,
    ) -> list[str]:
        """Get fallback providers when primary fails.

        Args:
            failed_provider: Provider that failed
            org_id: Organization ID

        Returns:
            List of fallback providers in order
        """
        fallback_order = self.FALLBACK_ORDER.get(
            failed_provider,
            self.FALLBACK_ORDER["default"]
        )

        # Filter out providers that are also cooling
        available = []
        for provider in fallback_order:
            if provider != failed_provider:
                cooling = await self.is_cooling(provider, org_id)
                if cooling is None:
                    available.append(provider)

        return available

    async def get_earliest_retry(self, org_id: str) -> float | None:
        """Get the earliest retry time across all cooling providers.

        Useful for 503 Retry-After header when all providers are limited.

        Args:
            org_id: Organization ID

        Returns:
            Seconds until earliest provider is available, or None if all available
        """
        now = time.time()
        earliest = float("inf")

        async with self._lock:
            for provider, info in self._cooling_providers.get(org_id, {}).items():
                remaining = info.cooling_until - now
                if remaining > 0:
                    earliest = min(earliest, remaining)

        return earliest if earliest != float("inf") else None

    async def clear_cooling(self, provider: str, org_id: str) -> bool:
        """Clear cooling state for a provider (e.g., after successful request).

        Args:
            provider: Provider to clear
            org_id: Organization ID

        Returns:
            True if cleared
        """
        # Clear in Redis
        client = await self._get_valkey_client()
        if client is not None:
            try:
                redis_client = await client._get_client()
                key = self._get_redis_key(org_id, provider)
                await redis_client.delete(key)
            except Exception:
                pass

        # Clear in memory
        async with self._lock:
            if org_id in self._cooling_providers:
                self._cooling_providers[org_id].pop(provider, None)

        return True


def parse_provider_rate_limit_error(
    exception: Exception,
    provider: str,
) -> ProviderRateLimitInfo | None:
    """Parse rate limit information from provider exception.

    Different providers return rate limit info differently:
    - Anthropic: headers["retry-after"]
    - OpenAI: headers["x-ratelimit-reset-requests"]
    - Google: error message parsing

    Args:
        exception: The exception from the provider
        provider: Provider name

    Returns:
        ProviderRateLimitInfo if rate limit detected, None otherwise
    """
    error_str = str(exception).lower()

    # Check if this is a rate limit error
    is_rate_limit = (
        "rate limit" in error_str or
        "rate_limit" in error_str or
        "too many requests" in error_str or
        "429" in error_str or
        "quota" in error_str or
        getattr(exception, "status_code", 0) == 429 or
        getattr(exception, "status", 0) == 429
    )

    if not is_rate_limit:
        return None

    # Extract retry_after
    retry_after = 60.0  # Default fallback

    # Try to get from exception attributes
    if hasattr(exception, "retry_after"):
        retry_after = float(exception.retry_after)
    elif hasattr(exception, "headers"):
        headers = exception.headers
        if isinstance(headers, dict):
            # Different header names by provider
            for header in ["retry-after", "x-ratelimit-reset", "x-ratelimit-reset-requests"]:
                if header in headers:
                    try:
                        retry_after = float(headers[header])
                        break
                    except (ValueError, TypeError):
                        pass

    # Determine limit type from message
    limit_type = "unknown"
    if "token" in error_str or "tpm" in error_str:
        limit_type = "tpm"
    elif "request" in error_str or "rpm" in error_str:
        limit_type = "rpm"
    elif "concurrent" in error_str:
        limit_type = "concurrent"
    elif "daily" in error_str or "rpd" in error_str:
        limit_type = "rpd"

    return ProviderRateLimitInfo(
        provider=provider,
        limit_type=limit_type,
        retry_after=retry_after,
        message=str(exception)[:200],
    )


# Global tracker instance
_tracker: ProviderRateLimitTracker | None = None


def get_provider_rate_tracker() -> ProviderRateLimitTracker:
    """Get the global provider rate limit tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ProviderRateLimitTracker()
    return _tracker


def reset_provider_rate_tracker() -> None:
    """Reset the global tracker (for testing)."""
    global _tracker
    _tracker = None
