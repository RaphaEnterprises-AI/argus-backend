"""
Enterprise-grade distributed rate limiting service.

Implements sliding window counter algorithm using Redis/Valkey for:
- Distributed rate limiting across multiple instances
- Smooth rate limiting without fixed-window burst vulnerabilities
- Automatic fallback chain: Valkey -> Upstash -> In-memory

Algorithm: Sliding Window Counter
---------------------------------
The sliding window algorithm provides smoother rate limiting than fixed windows
by calculating a weighted sum of the current and previous window counts:

    count = prev_window_count * (1 - elapsed/window) + current_window_count

This prevents the "burst at window boundary" problem where users could make
2x requests at the exact moment a fixed window resets.

Example with 60 requests/minute limit:
- Fixed window: User could make 60 requests at 0:59 and 60 more at 1:01
- Sliding window: Requests in the previous window are weighted down as time passes

Fallback Chain:
1. Valkey (K8s internal) - 1-5ms, primary backend
2. Upstash Redis (REST) - 10-30ms, external fallback
3. In-memory - 0ms, emergency fallback (per-instance only)

References:
- https://redis.io/glossary/rate-limiting/
- https://foojay.io/today/rate-limiting-with-redis-an-essential-guide/
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import structlog

from src.config import get_settings

if TYPE_CHECKING:
    from src.services.cache import UpstashRedisClient, ValkeyClient

logger = structlog.get_logger(__name__)


class RateLimitBackend(str, Enum):
    """Rate limiting backend options."""

    REDIS = "redis"
    MEMORY = "memory"


@dataclass
class RateLimitResult:
    """Result of a rate limit check.

    Attributes:
        allowed: Whether the request should be allowed
        remaining: Number of requests remaining in the current window
        limit: The maximum number of requests allowed
        reset_at: Unix timestamp when the rate limit window resets
        retry_after: Seconds until the request can be retried (0 if allowed)
        backend: Which backend was used for the check
    """

    allowed: bool
    remaining: int
    limit: int
    reset_at: int
    retry_after: int
    backend: str


class SlidingWindowRateLimiter:
    """Distributed rate limiter using sliding window counter algorithm.

    The sliding window counter algorithm provides better accuracy than fixed
    windows by weighting the previous window's count based on elapsed time.

    Formula:
        effective_count = prev_count * (1 - elapsed_ratio) + current_count

    Where:
        - prev_count: requests in the previous window
        - elapsed_ratio: (current_time - window_start) / window_size
        - current_count: requests in the current window

    This implementation uses Redis MULTI/EXEC for atomic operations and
    falls back gracefully through the cache hierarchy.

    Example:
        ```python
        limiter = SlidingWindowRateLimiter()

        result = await limiter.is_allowed(
            key="org:acme_corp",
            limit=60,
            window_seconds=60,
        )

        if not result.allowed:
            return Response(
                status_code=429,
                headers={"Retry-After": str(result.retry_after)},
            )
        ```
    """

    def __init__(
        self,
        key_prefix: str = "ratelimit",
        fallback_to_memory: bool = True,
    ):
        """Initialize the rate limiter.

        Args:
            key_prefix: Prefix for all rate limit keys in Redis
            fallback_to_memory: If True, use in-memory fallback when Redis unavailable
        """
        self._key_prefix = key_prefix
        self._fallback_to_memory = fallback_to_memory
        self._valkey_client: "ValkeyClient | None" = None
        self._upstash_client: "UpstashRedisClient | None" = None
        self._valkey_healthy: bool | None = None
        self._upstash_healthy: bool | None = None
        self._memory_storage: dict[str, list[float]] = defaultdict(list)
        self._memory_lock = asyncio.Lock()
        self._log = logger.bind(component="rate_limiter")

    def _get_window_id(self, window_seconds: int) -> tuple[int, int]:
        """Get current and previous window IDs.

        Args:
            window_seconds: Size of each window in seconds

        Returns:
            Tuple of (current_window_id, previous_window_id)
        """
        now = int(time.time())
        current_window = now // window_seconds
        return current_window, current_window - 1

    def _get_key(self, base_key: str, window_id: int) -> str:
        """Generate Redis key for a specific window.

        Args:
            base_key: The rate limit key (e.g., "org:acme_corp")
            window_id: The window identifier

        Returns:
            Full Redis key
        """
        return f"{self._key_prefix}:{base_key}:{window_id}"

    async def _get_valkey_client(self) -> "ValkeyClient | None":
        """Get Valkey client with health caching."""
        if self._valkey_client is None:
            from src.services.cache import get_valkey_client

            self._valkey_client = get_valkey_client()

        if self._valkey_client is None:
            return None

        # Skip if previously unhealthy (reset via reset_health_cache)
        if self._valkey_healthy is False:
            return None

        try:
            healthy = await self._valkey_client.ping()
            self._valkey_healthy = healthy
            return self._valkey_client if healthy else None
        except Exception as e:
            self._valkey_healthy = False
            self._log.warning("Valkey unhealthy for rate limiting", error=str(e))
            return None

    async def _get_upstash_client(self) -> "UpstashRedisClient | None":
        """Get Upstash client with health caching."""
        if self._upstash_client is None:
            from src.services.cache import get_upstash_client

            self._upstash_client = get_upstash_client()

        if self._upstash_client is None:
            return None

        # Skip if previously unhealthy
        if self._upstash_healthy is False:
            return None

        try:
            healthy = await self._upstash_client.ping()
            self._upstash_healthy = healthy
            return self._upstash_client if healthy else None
        except Exception as e:
            self._upstash_healthy = False
            self._log.warning("Upstash unhealthy for rate limiting", error=str(e))
            return None

    def reset_health_cache(self) -> None:
        """Reset health status cache to force re-checking backends."""
        self._valkey_healthy = None
        self._upstash_healthy = None

    async def _check_with_valkey(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult | None:
        """Check rate limit using Valkey (Redis-compatible).

        Uses atomic Redis operations for accurate distributed counting.

        Args:
            key: Rate limit key
            limit: Maximum requests allowed
            window_seconds: Window size in seconds

        Returns:
            RateLimitResult if successful, None if Valkey unavailable
        """
        client = await self._get_valkey_client()
        if client is None:
            return None

        try:
            now = time.time()
            current_window_id, prev_window_id = self._get_window_id(window_seconds)

            current_key = self._get_key(key, current_window_id)
            prev_key = self._get_key(key, prev_window_id)

            # Get the underlying redis client for pipeline operations
            redis_client = await client._get_client()

            # Use pipeline for atomic operations
            pipe = redis_client.pipeline()
            pipe.get(prev_key)
            pipe.incr(current_key)
            pipe.expire(current_key, window_seconds * 2)  # Keep 2 windows for sliding
            results = await pipe.execute()

            prev_count = int(results[0] or 0)
            current_count = int(results[1])

            # Calculate sliding window count
            window_start = current_window_id * window_seconds
            elapsed_ratio = (now - window_start) / window_seconds
            effective_count = prev_count * (1 - elapsed_ratio) + current_count

            allowed = effective_count <= limit
            remaining = max(0, limit - int(effective_count))
            reset_at = (current_window_id + 1) * window_seconds

            if not allowed:
                retry_after = int(reset_at - now)
            else:
                retry_after = 0

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                limit=limit,
                reset_at=reset_at,
                retry_after=retry_after,
                backend="valkey",
            )

        except Exception as e:
            self._log.warning("Valkey rate limit check failed", error=str(e), key=key)
            self._valkey_healthy = False
            return None

    async def _check_with_upstash(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult | None:
        """Check rate limit using Upstash Redis REST API.

        Uses individual REST calls since Upstash doesn't support pipelines.

        Args:
            key: Rate limit key
            limit: Maximum requests allowed
            window_seconds: Window size in seconds

        Returns:
            RateLimitResult if successful, None if Upstash unavailable
        """
        client = await self._get_upstash_client()
        if client is None:
            return None

        try:
            now = time.time()
            current_window_id, prev_window_id = self._get_window_id(window_seconds)

            current_key = self._get_key(key, current_window_id)
            prev_key = self._get_key(key, prev_window_id)

            # Get previous window count
            prev_value = await client.get(prev_key)
            prev_count = int(prev_value) if prev_value else 0

            # Increment current window (atomic)
            current_count = await client.incr(current_key)
            if current_count is None:
                current_count = 1

            # Set expiry
            await client.expire(current_key, window_seconds * 2)

            # Calculate sliding window count
            window_start = current_window_id * window_seconds
            elapsed_ratio = (now - window_start) / window_seconds
            effective_count = prev_count * (1 - elapsed_ratio) + current_count

            allowed = effective_count <= limit
            remaining = max(0, limit - int(effective_count))
            reset_at = (current_window_id + 1) * window_seconds

            if not allowed:
                retry_after = int(reset_at - now)
            else:
                retry_after = 0

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                limit=limit,
                reset_at=reset_at,
                retry_after=retry_after,
                backend="upstash",
            )

        except Exception as e:
            self._log.warning("Upstash rate limit check failed", error=str(e), key=key)
            self._upstash_healthy = False
            return None

    async def _check_with_memory(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        """Check rate limit using in-memory storage.

        This is a per-instance fallback when Redis is unavailable.
        Note: This does NOT provide distributed rate limiting.

        Args:
            key: Rate limit key
            limit: Maximum requests allowed
            window_seconds: Window size in seconds

        Returns:
            RateLimitResult
        """
        now = time.time()
        window_start = now - window_seconds

        async with self._memory_lock:
            # Clean old requests outside the window
            self._memory_storage[key] = [
                ts for ts in self._memory_storage[key] if ts > window_start
            ]

            current_count = len(self._memory_storage[key])

            if current_count >= limit:
                # Find when oldest request will expire
                oldest = min(self._memory_storage[key]) if self._memory_storage[key] else now
                retry_after = int(oldest + window_seconds - now)
                retry_after = max(1, retry_after)

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=limit,
                    reset_at=int(now + retry_after),
                    retry_after=retry_after,
                    backend="memory",
                )

            # Record this request
            self._memory_storage[key].append(now)
            remaining = limit - len(self._memory_storage[key])

            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                limit=limit,
                reset_at=int(now + window_seconds),
                retry_after=0,
                backend="memory",
            )

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int = 60,
    ) -> RateLimitResult:
        """Check if a request is allowed under the rate limit.

        Tries backends in order: Valkey -> Upstash -> Memory

        Args:
            key: Unique identifier for the rate limit bucket
                 (e.g., "org:acme_corp", "user:12345", "ip:192.168.1.1")
            limit: Maximum number of requests allowed in the window
            window_seconds: Size of the rate limiting window in seconds

        Returns:
            RateLimitResult with allowed status and metadata for response headers
        """
        # Try Valkey first (fastest)
        result = await self._check_with_valkey(key, limit, window_seconds)
        if result is not None:
            self._log.debug(
                "Rate limit checked via Valkey",
                key=key,
                allowed=result.allowed,
                remaining=result.remaining,
            )
            return result

        # Try Upstash second
        result = await self._check_with_upstash(key, limit, window_seconds)
        if result is not None:
            self._log.debug(
                "Rate limit checked via Upstash",
                key=key,
                allowed=result.allowed,
                remaining=result.remaining,
            )
            return result

        # Fall back to in-memory if allowed
        if self._fallback_to_memory:
            self._log.warning(
                "Using in-memory rate limiting (no Redis available)",
                key=key,
            )
            return await self._check_with_memory(key, limit, window_seconds)

        # If no fallback, allow by default (fail-open)
        self._log.error(
            "No rate limiting backend available, allowing request",
            key=key,
        )
        return RateLimitResult(
            allowed=True,
            remaining=limit,
            limit=limit,
            reset_at=int(time.time() + window_seconds),
            retry_after=0,
            backend="none",
        )

    async def get_remaining(
        self,
        key: str,
        limit: int,
        window_seconds: int = 60,
    ) -> int:
        """Get remaining requests without incrementing counter.

        Useful for displaying rate limit info without consuming quota.

        Args:
            key: Rate limit key
            limit: Maximum requests allowed
            window_seconds: Window size in seconds

        Returns:
            Number of remaining requests
        """
        # For read-only check, we need custom logic that doesn't INCR
        client = await self._get_valkey_client()
        if client is not None:
            try:
                now = time.time()
                current_window_id, prev_window_id = self._get_window_id(window_seconds)

                current_key = self._get_key(key, current_window_id)
                prev_key = self._get_key(key, prev_window_id)

                redis_client = await client._get_client()

                # Get both window counts
                pipe = redis_client.pipeline()
                pipe.get(prev_key)
                pipe.get(current_key)
                results = await pipe.execute()

                prev_count = int(results[0] or 0)
                current_count = int(results[1] or 0)

                # Calculate sliding window count
                window_start = current_window_id * window_seconds
                elapsed_ratio = (now - window_start) / window_seconds
                effective_count = prev_count * (1 - elapsed_ratio) + current_count

                return max(0, limit - int(effective_count))

            except Exception as e:
                self._log.warning("Failed to get remaining count", error=str(e))

        # Fallback to memory
        async with self._memory_lock:
            window_start = time.time() - window_seconds
            requests = [
                ts for ts in self._memory_storage.get(key, []) if ts > window_start
            ]
            return max(0, limit - len(requests))

    async def reset(self, key: str) -> bool:
        """Reset rate limit for a specific key.

        Useful for admin operations or testing.

        Args:
            key: Rate limit key to reset

        Returns:
            True if reset was successful
        """
        success = False

        # Reset in Valkey
        client = await self._get_valkey_client()
        if client is not None:
            try:
                redis_client = await client._get_client()
                # Delete all windows for this key
                pattern = f"{self._key_prefix}:{key}:*"
                cursor = 0
                while True:
                    cursor, keys = await redis_client.scan(
                        cursor=cursor, match=pattern, count=100
                    )
                    if keys:
                        await redis_client.delete(*keys)
                    if cursor == 0:
                        break
                success = True
            except Exception as e:
                self._log.warning("Failed to reset in Valkey", error=str(e), key=key)

        # Reset in memory
        async with self._memory_lock:
            if key in self._memory_storage:
                del self._memory_storage[key]
                success = True

        return success

    async def health_check(self) -> dict:
        """Check health of rate limiting backends.

        Returns:
            Health status dictionary
        """
        valkey_healthy = False
        upstash_healthy = False

        valkey = await self._get_valkey_client()
        if valkey is not None:
            valkey_healthy = True

        upstash = await self._get_upstash_client()
        if upstash is not None:
            upstash_healthy = True

        # Determine active backend
        if valkey_healthy:
            active = "valkey"
        elif upstash_healthy:
            active = "upstash"
        elif self._fallback_to_memory:
            active = "memory"
        else:
            active = "none"

        return {
            "healthy": valkey_healthy or upstash_healthy or self._fallback_to_memory,
            "valkey_available": valkey_healthy,
            "upstash_available": upstash_healthy,
            "memory_fallback_enabled": self._fallback_to_memory,
            "active_backend": active,
        }


# Global rate limiter instance (lazy initialized)
_rate_limiter: SlidingWindowRateLimiter | None = None


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Get or create the global rate limiter instance.

    Returns:
        SlidingWindowRateLimiter instance
    """
    global _rate_limiter

    if _rate_limiter is None:
        settings = get_settings()

        # Check if rate limiting should use Redis or fall back to memory-only
        backend = getattr(settings, "rate_limit_backend", "redis")

        _rate_limiter = SlidingWindowRateLimiter(
            key_prefix=getattr(settings, "rate_limit_key_prefix", "ratelimit"),
            fallback_to_memory=(backend != "redis_only"),
        )

    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter instance."""
    global _rate_limiter
    _rate_limiter = None


# Utility function for generating consistent rate limit keys
def generate_rate_limit_key(
    org_id: str | None = None,
    user_id: str | None = None,
    ip_address: str | None = None,
    endpoint_hash: str | None = None,
) -> str:
    """Generate a rate limit key from identifiers.

    Priority: org_id > user_id > ip_address

    Args:
        org_id: Organization ID (highest priority)
        user_id: User ID
        ip_address: Client IP address (lowest priority)
        endpoint_hash: Optional endpoint path hash for endpoint-specific limiting

    Returns:
        Rate limit key string
    """
    if org_id:
        base = f"org:{org_id}"
    elif user_id:
        base = f"user:{user_id}"
    elif ip_address:
        base = f"ip:{ip_address}"
    else:
        base = "global"

    if endpoint_hash:
        return f"{base}:{endpoint_hash}"
    return base


def hash_endpoint(path: str) -> str:
    """Generate a short hash for an endpoint path.

    Args:
        path: API endpoint path

    Returns:
        8-character hash
    """
    return hashlib.sha256(path.encode()).hexdigest()[:8]
