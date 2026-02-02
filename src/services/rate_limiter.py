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


# =============================================================================
# CONCURRENT CONNECTION LIMITER
# =============================================================================
# For SSE/streaming endpoints that hold connections for extended periods


@dataclass
class ConcurrentResult:
    """Result of a concurrent connection check.

    Attributes:
        allowed: Whether the connection should be allowed
        active: Number of currently active connections
        limit: Maximum concurrent connections allowed
        connection_id: Unique ID for this connection (needed for release)
        backend: Which backend was used
    """

    allowed: bool
    active: int
    limit: int
    connection_id: str
    backend: str


class ConcurrentConnectionLimiter:
    """Distributed concurrent connection limiter for SSE/streaming endpoints.

    Unlike rate limiting which counts requests per time window, this limiter
    tracks the number of **active** connections at any moment. This is essential
    for SSE streams that can run for minutes.

    Use Case:
        A user starts a chat stream that takes 5 minutes. During this time,
        they should not be able to open unlimited additional streams. This
        limiter ensures at most N concurrent streams per user/org.

    Implementation:
        Uses Redis sorted sets with timestamp scores for automatic expiration.
        Each active connection is stored with its start timestamp. A background
        cleanup removes connections older than the TTL (in case of ungraceful
        disconnects).

    Example:
        ```python
        limiter = ConcurrentConnectionLimiter()

        result = await limiter.acquire(
            key="org:acme_corp:chat",
            limit=3,
        )

        if not result.allowed:
            return Response(status_code=429, content="Too many concurrent streams")

        try:
            async for chunk in stream_chat():
                yield chunk
        finally:
            # CRITICAL: Always release the connection
            await limiter.release(key, result.connection_id)
        ```
    """

    def __init__(
        self,
        key_prefix: str = "concurrent",
        connection_ttl: int = 600,  # 10 minute max connection
    ):
        """Initialize the concurrent connection limiter.

        Args:
            key_prefix: Prefix for Redis keys
            connection_ttl: Max lifetime of a connection in seconds (auto-cleanup)
        """
        self._key_prefix = key_prefix
        self._connection_ttl = connection_ttl
        self._valkey_client: "ValkeyClient | None" = None
        self._valkey_healthy: bool | None = None
        self._memory_connections: dict[str, set[str]] = defaultdict(set)
        self._memory_lock = asyncio.Lock()
        self._log = logger.bind(component="concurrent_limiter")

    def _get_key(self, base_key: str) -> str:
        """Generate Redis key for connection tracking."""
        return f"{self._key_prefix}:{base_key}"

    def _generate_connection_id(self) -> str:
        """Generate unique connection ID."""
        import uuid

        return str(uuid.uuid4())

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
            self._log.warning("Valkey unhealthy for concurrent limiting", error=str(e))
            return None

    async def acquire(
        self,
        key: str,
        limit: int,
    ) -> ConcurrentResult:
        """Acquire a connection slot.

        Args:
            key: Unique identifier for the connection bucket
            limit: Maximum concurrent connections allowed

        Returns:
            ConcurrentResult with allowed status and connection_id
        """
        connection_id = self._generate_connection_id()
        now = time.time()

        # Try Valkey first
        client = await self._get_valkey_client()
        if client is not None:
            try:
                redis_client = await client._get_client()
                redis_key = self._get_key(key)

                # Clean up expired connections and check count atomically
                pipe = redis_client.pipeline()
                # Remove connections older than TTL
                pipe.zremrangebyscore(redis_key, 0, now - self._connection_ttl)
                # Get current count
                pipe.zcard(redis_key)
                results = await pipe.execute()

                current_count = results[1]

                if current_count >= limit:
                    self._log.info(
                        "Concurrent limit reached",
                        key=key,
                        active=current_count,
                        limit=limit,
                    )
                    return ConcurrentResult(
                        allowed=False,
                        active=current_count,
                        limit=limit,
                        connection_id="",
                        backend="valkey",
                    )

                # Add this connection
                await redis_client.zadd(redis_key, {connection_id: now})
                await redis_client.expire(redis_key, self._connection_ttl * 2)

                self._log.debug(
                    "Connection acquired",
                    key=key,
                    connection_id=connection_id,
                    active=current_count + 1,
                )

                return ConcurrentResult(
                    allowed=True,
                    active=current_count + 1,
                    limit=limit,
                    connection_id=connection_id,
                    backend="valkey",
                )

            except Exception as e:
                self._log.warning(
                    "Valkey concurrent check failed, using memory",
                    error=str(e),
                )
                self._valkey_healthy = False

        # Fallback to in-memory
        async with self._memory_lock:
            current_count = len(self._memory_connections[key])

            if current_count >= limit:
                return ConcurrentResult(
                    allowed=False,
                    active=current_count,
                    limit=limit,
                    connection_id="",
                    backend="memory",
                )

            self._memory_connections[key].add(connection_id)

            return ConcurrentResult(
                allowed=True,
                active=current_count + 1,
                limit=limit,
                connection_id=connection_id,
                backend="memory",
            )

    async def release(self, key: str, connection_id: str) -> bool:
        """Release a connection slot.

        CRITICAL: Must be called when stream ends (success or error).

        Args:
            key: Connection bucket key
            connection_id: ID returned from acquire()

        Returns:
            True if released successfully
        """
        if not connection_id:
            return False

        # Try Valkey first
        client = await self._get_valkey_client()
        if client is not None:
            try:
                redis_client = await client._get_client()
                redis_key = self._get_key(key)
                removed = await redis_client.zrem(redis_key, connection_id)

                self._log.debug(
                    "Connection released",
                    key=key,
                    connection_id=connection_id,
                    removed=removed,
                )
                return removed > 0

            except Exception as e:
                self._log.warning(
                    "Valkey release failed",
                    error=str(e),
                    connection_id=connection_id,
                )

        # Fallback to memory
        async with self._memory_lock:
            if connection_id in self._memory_connections.get(key, set()):
                self._memory_connections[key].discard(connection_id)
                return True

        return False

    async def get_active_count(self, key: str) -> int:
        """Get current active connection count (for monitoring).

        Args:
            key: Connection bucket key

        Returns:
            Number of active connections
        """
        client = await self._get_valkey_client()
        if client is not None:
            try:
                redis_client = await client._get_client()
                redis_key = self._get_key(key)

                # Clean expired and count
                now = time.time()
                await redis_client.zremrangebyscore(
                    redis_key, 0, now - self._connection_ttl
                )
                return await redis_client.zcard(redis_key)

            except Exception as e:
                self._log.warning("Failed to get active count", error=str(e))

        async with self._memory_lock:
            return len(self._memory_connections.get(key, set()))

    async def health_check(self) -> dict:
        """Check health of concurrent connection limiter."""
        valkey_healthy = False

        client = await self._get_valkey_client()
        if client is not None:
            valkey_healthy = True

        return {
            "healthy": valkey_healthy or True,  # Memory fallback always works
            "valkey_available": valkey_healthy,
            "memory_fallback": True,
            "connection_ttl_seconds": self._connection_ttl,
        }


# Global concurrent limiter instance (lazy initialized)
_concurrent_limiter: ConcurrentConnectionLimiter | None = None


def get_concurrent_limiter() -> ConcurrentConnectionLimiter:
    """Get or create the global concurrent connection limiter.

    Returns:
        ConcurrentConnectionLimiter instance
    """
    global _concurrent_limiter

    if _concurrent_limiter is None:
        settings = get_settings()
        _concurrent_limiter = ConcurrentConnectionLimiter(
            key_prefix=getattr(settings, "rate_limit_key_prefix", "ratelimit")
            + ":concurrent",
            connection_ttl=getattr(settings, "concurrent_connection_ttl", 600),
        )

    return _concurrent_limiter


def reset_concurrent_limiter() -> None:
    """Reset the global concurrent limiter instance."""
    global _concurrent_limiter
    _concurrent_limiter = None


class ConcurrentConnectionContext:
    """Context manager for concurrent connection limiting.

    Automatically acquires and releases connection slots, ensuring proper
    cleanup even if the stream fails.

    Example:
        ```python
        from src.services.rate_limiter import ConcurrentConnectionContext

        @router.post("/chat/stream")
        async def stream_chat(request: Request):
            async with ConcurrentConnectionContext(
                key=f"org:{request.state.user.organization_id}:chat",
                limit=3,
            ) as conn:
                if not conn.allowed:
                    raise HTTPException(429, "Too many concurrent streams")

                async def generate():
                    async for chunk in chat_stream():
                        yield chunk

                return EventSourceResponse(generate())
        ```
    """

    def __init__(
        self,
        key: str,
        limit: int,
        limiter: ConcurrentConnectionLimiter | None = None,
    ):
        """Initialize the context.

        Args:
            key: Connection bucket key (e.g., "org:acme:chat")
            limit: Maximum concurrent connections
            limiter: Optional limiter instance (uses global if not provided)
        """
        self.key = key
        self.limit = limit
        self._limiter = limiter
        self._result: ConcurrentResult | None = None

    @property
    def allowed(self) -> bool:
        """Whether the connection was allowed."""
        return self._result.allowed if self._result else False

    @property
    def active(self) -> int:
        """Number of active connections."""
        return self._result.active if self._result else 0

    async def __aenter__(self) -> "ConcurrentConnectionContext":
        if self._limiter is None:
            self._limiter = get_concurrent_limiter()

        self._result = await self._limiter.acquire(self.key, self.limit)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._result and self._result.connection_id:
            await self._limiter.release(self.key, self._result.connection_id)
        return None  # Don't suppress exceptions


async def check_concurrent_limit(
    request: "Request",
    path: str,
    user_tier: str | None = None,
) -> ConcurrentResult | None:
    """Check concurrent connection limit for a streaming endpoint.

    This is a simpler API for middleware-level checking.

    Args:
        request: FastAPI request
        path: Request path
        user_tier: User's subscription tier

    Returns:
        ConcurrentResult if limit applies, None if no limit for this endpoint
    """
    from src.api.security.rate_limit_config import (
        get_concurrent_limit_for_endpoint,
        is_streaming_endpoint,
    )

    # Only check for streaming endpoints
    if not is_streaming_endpoint(path):
        return None

    limit = get_concurrent_limit_for_endpoint(path, user_tier)
    if limit is None:
        return None

    # Generate key based on user/org
    if hasattr(request.state, "user") and request.state.user:
        user = request.state.user
        if user.organization_id:
            key = f"org:{user.organization_id}"
        else:
            key = f"user:{user.user_id}"
    else:
        from src.api.security.auth import get_client_ip

        key = f"ip:{get_client_ip(request)}"

    # Add path to key for endpoint-specific limits
    key = f"{key}:{hash_endpoint(path)}"

    limiter = get_concurrent_limiter()
    return await limiter.acquire(key, limit)
