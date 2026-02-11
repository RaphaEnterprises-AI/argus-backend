"""
Configuration-driven rate limiting for API endpoints.

This module provides:
- Externalized endpoint-specific rate limits
- Tier-based multipliers for different subscription levels
- Environment-variable overrides for dynamic configuration
- Type-safe configuration with dataclasses

The configuration allows operators to adjust rate limits without code changes
by setting environment variables like:
    RATE_LIMIT_CHAT_REQUESTS=100
    RATE_LIMIT_CHAT_WINDOW=60

References:
- https://zuplo.com/learning-center/10-best-practices-for-api-rate-limiting-in-2025
"""

import os
from dataclasses import dataclass, field
from enum import Enum


class UserTier(str, Enum):
    """User subscription tiers for rate limit multipliers."""

    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"


@dataclass(frozen=True)
class EndpointLimit:
    """Rate limit configuration for a specific endpoint pattern.

    Attributes:
        pattern: URL path prefix to match (e.g., "/api/v1/chat")
        requests: Base number of requests allowed per window
        window: Time window in seconds
        tier_multipliers: Multipliers for each subscription tier
        description: Human-readable description for documentation
    """

    pattern: str
    requests: int
    window: int = 60
    tier_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            UserTier.FREE.value: 1.0,
            UserTier.STARTER.value: 2.0,
            UserTier.PRO.value: 10.0,
            UserTier.ENTERPRISE.value: 33.3,
            UserTier.UNLIMITED.value: float("inf"),
        }
    )
    description: str = ""

    def get_limit_for_tier(self, tier: str | None) -> int:
        """Get the adjusted limit for a specific tier.

        Args:
            tier: User's subscription tier

        Returns:
            Adjusted request limit (tier multiplier * base)
        """
        if tier is None:
            tier = UserTier.FREE.value

        multiplier = self.tier_multipliers.get(tier, 1.0)

        if multiplier == float("inf"):
            return int(1e9)  # Effectively unlimited

        return int(self.requests * multiplier)


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable with fallback."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


# =============================================================================
# Default Rate Limits (can be overridden via environment variables)
# =============================================================================

# General API limits
DEFAULT_LIMIT = _get_env_int("RATE_LIMIT_DEFAULT_REQUESTS", 120)
DEFAULT_WINDOW = _get_env_int("RATE_LIMIT_DEFAULT_WINDOW", 60)

# LLM-heavy endpoint limits
CHAT_LIMIT = _get_env_int("RATE_LIMIT_CHAT_REQUESTS", 60)
CHAT_WINDOW = _get_env_int("RATE_LIMIT_CHAT_WINDOW", 60)

TEST_GEN_LIMIT = _get_env_int("RATE_LIMIT_TEST_GEN_REQUESTS", 10)
VISUAL_AI_LIMIT = _get_env_int("RATE_LIMIT_VISUAL_AI_REQUESTS", 10)
HEALING_LIMIT = _get_env_int("RATE_LIMIT_HEALING_REQUESTS", 10)

# Auth endpoint limits (brute force protection)
AUTH_LIMIT = _get_env_int("RATE_LIMIT_AUTH_REQUESTS", 10)

# Health endpoint limits (monitoring)
HEALTH_LIMIT = _get_env_int("RATE_LIMIT_HEALTH_REQUESTS", 1000)


# =============================================================================
# Tier-based limits (base requests per minute per tier)
# =============================================================================

TIER_LIMITS: dict[str, dict[str, int | float]] = {
    UserTier.FREE.value: {
        "requests": _get_env_int("RATE_LIMIT_TIER_FREE", 60),
        "window": 60,
    },
    UserTier.STARTER.value: {
        "requests": _get_env_int("RATE_LIMIT_TIER_STARTER", 120),
        "window": 60,
    },
    UserTier.PRO.value: {
        "requests": _get_env_int("RATE_LIMIT_TIER_PRO", 600),
        "window": 60,
    },
    UserTier.ENTERPRISE.value: {
        "requests": _get_env_int("RATE_LIMIT_TIER_ENTERPRISE", 2000),
        "window": 60,
    },
    UserTier.UNLIMITED.value: {
        "requests": float("inf"),
        "window": 60,
    },
}


# =============================================================================
# Endpoint-specific limits
# =============================================================================
# NOTE: Order matters! More specific paths must come before less specific ones
# because matching uses startswith().

ENDPOINT_LIMITS: list[EndpointLimit] = [
    # =========================================================================
    # LLM-HEAVY ENDPOINTS (Expensive API calls - strict limits)
    # =========================================================================
    EndpointLimit(
        pattern="/api/v1/chat/stream",
        requests=CHAT_LIMIT,
        window=CHAT_WINDOW,
        description="Streaming chat endpoint - high LLM cost",
    ),
    EndpointLimit(
        pattern="/api/v1/chat/message",
        requests=CHAT_LIMIT,
        window=CHAT_WINDOW,
        description="Chat message endpoint - high LLM cost",
    ),
    EndpointLimit(
        pattern="/api/v1/stream/test",
        requests=5,
        window=60,
        description="Test streaming - developer use only",
    ),
    # Test generation
    EndpointLimit(
        pattern="/api/v1/quality/generate-test",
        requests=TEST_GEN_LIMIT,
        window=60,
        description="AI test generation - high LLM cost",
    ),
    EndpointLimit(
        pattern="/api/v1/quality/batch-generate",
        requests=3,
        window=60,
        description="Batch test generation - very high LLM cost",
    ),
    # Visual AI
    EndpointLimit(
        pattern="/api/v1/visual-ai/analyze",
        requests=VISUAL_AI_LIMIT,
        window=60,
        description="Visual analysis - includes screenshot processing",
    ),
    EndpointLimit(
        pattern="/api/v1/visual-ai/compare",
        requests=VISUAL_AI_LIMIT,
        window=60,
        description="Visual comparison - includes screenshot processing",
    ),
    EndpointLimit(
        pattern="/api/v1/visual-ai/accessibility",
        requests=VISUAL_AI_LIMIT,
        window=60,
        description="Accessibility analysis",
    ),
    # API test generation
    EndpointLimit(
        pattern="/api/v1/api-tests/generate",
        requests=5,
        window=60,
        description="API test generation - high LLM cost",
    ),
    # Self-healing
    EndpointLimit(
        pattern="/api/v1/healing/analyze",
        requests=HEALING_LIMIT,
        window=60,
        description="Self-healing analysis - AI-powered",
    ),
    EndpointLimit(
        pattern="/api/v1/healing/suggest",
        requests=HEALING_LIMIT,
        window=60,
        description="Self-healing suggestions - AI-powered",
    ),
    # SAST analysis
    EndpointLimit(
        pattern="/api/v1/sast/analyze",
        requests=5,
        window=60,
        description="SAST security analysis - expensive",
    ),
    # Insights generation
    EndpointLimit(
        pattern="/api/v1/insights/generate",
        requests=10,
        window=60,
        description="AI insights generation",
    ),
    EndpointLimit(
        pattern="/api/v1/correlations/insights",
        requests=10,
        window=60,
        description="Correlation insights - AI-powered",
    ),
    # Reports
    EndpointLimit(
        pattern="/api/v1/reports/generate",
        requests=5,
        window=60,
        description="Report generation with AI summaries",
    ),
    # =========================================================================
    # AUTHENTICATION ENDPOINTS (Brute force protection)
    # =========================================================================
    EndpointLimit(
        pattern="/api/v1/auth/device/authorize",
        requests=AUTH_LIMIT,
        window=60,
        description="Device authorization - brute force protection",
    ),
    EndpointLimit(
        pattern="/api/v1/auth/device/token",
        requests=AUTH_LIMIT,
        window=60,
        description="Device token exchange - brute force protection",
    ),
    EndpointLimit(
        pattern="/api/v1/auth/device/verify",
        requests=AUTH_LIMIT,
        window=60,
        description="Device verification - brute force protection",
    ),
    EndpointLimit(
        pattern="/api/v1/api-keys",
        requests=AUTH_LIMIT,
        window=60,
        description="API key management - security sensitive",
    ),
    EndpointLimit(
        pattern="/api/v1/oauth/",
        requests=20,
        window=60,
        description="OAuth flows",
    ),
    # =========================================================================
    # DATA MODIFICATION ENDPOINTS (Prevent abuse)
    # =========================================================================
    EndpointLimit(
        pattern="/api/v1/organizations",
        requests=5,
        window=60,
        description="Organization management - prevent spam creation",
    ),
    EndpointLimit(
        pattern="/api/v1/projects",
        requests=20,
        window=60,
        description="Project management",
    ),
    EndpointLimit(
        pattern="/api/v1/tests",
        requests=30,
        window=60,
        description="Test management",
    ),
    EndpointLimit(
        pattern="/api/v1/artifacts",
        requests=20,
        window=60,
        description="Artifact uploads",
    ),
    # =========================================================================
    # WEBHOOK ENDPOINTS (External integrations)
    # =========================================================================
    EndpointLimit(
        pattern="/api/v1/webhooks/",
        requests=100,
        window=60,
        description="Webhook endpoints - higher limit for integrations",
    ),
    # =========================================================================
    # DISCOVERY & CRAWLING (Resource intensive)
    # =========================================================================
    # Discovery API endpoints - higher limits for normal API usage
    # Must come BEFORE /api/v1/discover to avoid false matches
    EndpointLimit(
        pattern="/api/v1/discovery/",
        requests=100,
        window=60,
        description="Discovery API - normal usage",
    ),
    # Auto-discover endpoint (intensive crawling operation)
    EndpointLimit(
        pattern="/api/v1/discover",
        requests=10,
        window=60,
        description="Auto-discovery crawling - resource intensive",
    ),
    # =========================================================================
    # BROWSER EXECUTION (Resource intensive)
    # =========================================================================
    EndpointLimit(
        pattern="/api/v1/browser/execute",
        requests=10,
        window=60,
        description="Browser execution - resource intensive",
    ),
    EndpointLimit(
        pattern="/api/v1/browser/action",
        requests=30,
        window=60,
        description="Browser actions",
    ),
    # =========================================================================
    # HEALTH & MONITORING (High limits)
    # =========================================================================
    EndpointLimit(
        pattern="/health",
        requests=HEALTH_LIMIT,
        window=60,
        description="Health checks - monitoring systems",
    ),
    EndpointLimit(
        pattern="/api/v1/health",
        requests=HEALTH_LIMIT,
        window=60,
        description="API health checks - monitoring systems",
    ),
]


# Exempt endpoints (bypass rate limiting entirely)
EXEMPT_ENDPOINTS: set[str] = {
    "/health",
    "/openapi.json",
    "/docs",
    "/redoc",
}

# Exempt organizations (bypass rate limiting entirely)
# Comma-separated org IDs via environment variable
EXEMPT_ORGS: set[str] = set(
    filter(None, os.getenv("RATE_LIMIT_EXEMPT_ORGS", "").split(","))
)


# =============================================================================
# CONCURRENT CONNECTION LIMITS (for SSE/streaming endpoints)
# =============================================================================
# These endpoints hold open connections for extended periods.
# Unlike rate limiting (requests/minute), this limits simultaneous connections.


@dataclass(frozen=True)
class ConcurrentLimit:
    """Concurrent connection limit for streaming endpoints.

    Attributes:
        pattern: URL path prefix to match
        max_concurrent: Maximum simultaneous connections per user/org
        description: Human-readable description
    """

    pattern: str
    max_concurrent: int
    tier_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            UserTier.FREE.value: 1.0,
            UserTier.STARTER.value: 2.0,
            UserTier.PRO.value: 5.0,
            UserTier.ENTERPRISE.value: 10.0,
            UserTier.UNLIMITED.value: float("inf"),
        }
    )
    description: str = ""

    def get_limit_for_tier(self, tier: str | None) -> int:
        """Get adjusted concurrent limit for a tier."""
        if tier is None:
            tier = UserTier.FREE.value

        multiplier = self.tier_multipliers.get(tier, 1.0)

        if multiplier == float("inf"):
            return int(1e6)  # Effectively unlimited

        return max(1, int(self.max_concurrent * multiplier))


# Base concurrent limits (can be overridden via env vars)
CHAT_CONCURRENT = _get_env_int("CONCURRENT_CHAT_STREAMS", 3)
TEST_CONCURRENT = _get_env_int("CONCURRENT_TEST_STREAMS", 2)
STREAM_CONCURRENT = _get_env_int("CONCURRENT_GENERIC_STREAMS", 5)


CONCURRENT_LIMITS: list[ConcurrentLimit] = [
    # Chat streams - users can have multiple active conversations
    ConcurrentLimit(
        pattern="/api/v1/chat/stream",
        max_concurrent=CHAT_CONCURRENT,
        description="Chat streaming - AI conversation streams",
    ),
    ConcurrentLimit(
        pattern="/api/v1/chat/message",
        max_concurrent=CHAT_CONCURRENT,
        description="Chat messages - non-streaming but LLM-heavy",
    ),
    # Test execution streams - resource intensive
    ConcurrentLimit(
        pattern="/api/v1/stream/test",
        max_concurrent=TEST_CONCURRENT,
        description="Test execution streaming - very resource intensive",
    ),
    ConcurrentLimit(
        pattern="/api/v1/stream/",
        max_concurrent=STREAM_CONCURRENT,
        description="Generic streaming endpoints",
    ),
    # Browser automation - holds Selenium session
    ConcurrentLimit(
        pattern="/api/v1/browser/execute",
        max_concurrent=2,
        description="Browser execution - holds browser session",
    ),
    # Discovery crawling - intensive operation
    ConcurrentLimit(
        pattern="/api/v1/discover",
        max_concurrent=1,
        description="Auto-discovery - crawls entire application",
    ),
]


def get_concurrent_limit_for_endpoint(
    path: str,
    user_tier: str | None = None,
) -> int | None:
    """Get concurrent connection limit for an endpoint.

    Args:
        path: Request path
        user_tier: User's subscription tier

    Returns:
        Concurrent limit or None if no limit applies
    """
    for limit in CONCURRENT_LIMITS:
        if path.startswith(limit.pattern):
            return limit.get_limit_for_tier(user_tier)

    return None  # No concurrent limit for this endpoint


def is_streaming_endpoint(path: str) -> bool:
    """Check if an endpoint is a streaming endpoint that needs concurrent limiting.

    Args:
        path: Request path

    Returns:
        True if this is a streaming endpoint
    """
    streaming_patterns = [
        "/api/v1/chat/stream",
        "/api/v1/stream/",
        "/api/v1/browser/execute",
        "/api/v1/discover",
    ]
    return any(path.startswith(pattern) for pattern in streaming_patterns)


def get_limit_for_endpoint(
    path: str,
    user_tier: str | None = None,
) -> tuple[int, int]:
    """Get the rate limit configuration for an endpoint.

    Searches endpoint limits in order (most specific first).

    Args:
        path: The request path
        user_tier: Optional user tier for multipliers

    Returns:
        Tuple of (requests_limit, window_seconds)
    """
    for limit in ENDPOINT_LIMITS:
        if path.startswith(limit.pattern):
            requests = limit.get_limit_for_tier(user_tier)
            return requests, limit.window

    # Fall back to tier-based default
    if user_tier and user_tier in TIER_LIMITS:
        tier_limit = TIER_LIMITS[user_tier]
        requests = tier_limit["requests"]
        if requests == float("inf"):
            requests = int(1e9)
        return int(requests), int(tier_limit["window"])

    return DEFAULT_LIMIT, DEFAULT_WINDOW


def is_exempt_endpoint(path: str) -> bool:
    """Check if an endpoint is exempt from rate limiting.

    Args:
        path: The request path

    Returns:
        True if exempt
    """
    return path in EXEMPT_ENDPOINTS


def is_exempt_org(org_id: str | None) -> bool:
    """Check if an organization is exempt from rate limiting.

    Args:
        org_id: Organization ID

    Returns:
        True if exempt
    """
    return org_id is not None and org_id in EXEMPT_ORGS
