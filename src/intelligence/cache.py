"""
Intelligence Cache Module - High-performance caching for the Unified Instant Intelligence Layer (UIIL).

This module provides a caching layer that transforms 2-5s LLM calls into <100ms responses
by caching intelligence results in Valkey (Redis-compatible).

Key Features:
- get_or_search: Check cache before invoking expensive search/LLM operations
- Intent-based TTL: Different TTLs for different intent types
- Cache key generation with SHA256 hashing
- Latency tracking for performance monitoring
- Pattern-based cache invalidation

Usage:
    ```python
    from src.intelligence import IntelligenceCache, CachedResult

    cache = IntelligenceCache()

    # Get cached result or execute search function
    result = await cache.get_or_search(
        query="how to fix login button selector",
        intent="self_healing",
        search_fn=expensive_cognee_search
    )

    # Check if result came from cache
    if result.source == "valkey":
        print(f"Cache hit! Latency: {result.latency_ms}ms")
    ```
"""

import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import structlog

from src.services.cache import (
    CloudflareKVClient,
    UpstashRedisClient,
    ValkeyClient,
    get_kv_client,
    get_upstash_client,
    get_valkey_client,
)

logger = structlog.get_logger(__name__)

# Cache client type that supports all backends
CacheClient = ValkeyClient | UpstashRedisClient | CloudflareKVClient

# Type variable for generic search function return type
T = TypeVar("T")


class CacheSource(str, Enum):
    """Source of cached result."""

    VALKEY = "valkey"
    COGNEE = "cognee"
    LLM = "llm"


@dataclass
class CachedResult:
    """Result from intelligence cache lookup.

    Attributes:
        data: The cached or computed data
        source: Where the data came from (valkey, cognee, or llm)
        latency_ms: Time taken to retrieve/compute the result in milliseconds
    """

    data: Any
    source: CacheSource | str
    latency_ms: float

    def __post_init__(self):
        """Normalize source to CacheSource enum if string."""
        if isinstance(self.source, str):
            try:
                self.source = CacheSource(self.source)
            except ValueError:
                # Keep as string if not a known source
                pass


# Default TTL configurations (in seconds)
DEFAULT_TTL = 3600  # 1 hour for standard intents
REALTIME_TTL = 300  # 5 minutes for real-time intents

# Intent-specific TTL overrides
INTENT_TTL_MAP: dict[str, int] = {
    # Real-time intents (shorter TTL)
    "realtime": REALTIME_TTL,
    "live_analysis": REALTIME_TTL,
    "current_state": REALTIME_TTL,
    "session": REALTIME_TTL,
    # Standard intents (default TTL)
    "self_healing": DEFAULT_TTL,
    "pattern_match": DEFAULT_TTL,
    "code_analysis": DEFAULT_TTL,
    "test_generation": DEFAULT_TTL,
    "failure_analysis": DEFAULT_TTL,
    # Long-lived intents (extended TTL)
    "static_knowledge": 86400,  # 24 hours
    "documentation": 86400,
    "selector_mapping": 7200,  # 2 hours
}


class IntelligenceCache:
    """High-performance caching layer for intelligence operations.

    Wraps ValkeyClient to provide:
    - get_or_search: Cache-through pattern for expensive operations
    - Intent-based TTL management
    - Cache invalidation by pattern or project

    Example:
        ```python
        cache = IntelligenceCache()

        # Cache a search result
        result = await cache.get_or_search(
            query="find similar failures for timeout error",
            intent="self_healing",
            search_fn=lambda: cognee_client.find_similar_failures("timeout error")
        )

        # Invalidate all self_healing cache entries
        await cache.invalidate("intel:self_healing:*")

        # Invalidate cache for a specific project
        await cache.invalidate_for_project("org123", "proj456")
        ```
    """

    def __init__(
        self,
        valkey_client: ValkeyClient | None = None,
        upstash_client: UpstashRedisClient | None = None,
        kv_client: CloudflareKVClient | None = None,
        default_ttl: int = DEFAULT_TTL,
        key_prefix: str = "intel",
    ):
        """Initialize the intelligence cache.

        Cache priority (fastest to slowest, cheapest for mixed workloads):
        1. Valkey (K8s internal) - 1-5ms latency
        2. Upstash Redis (REST) - 10-30ms latency, 227x cheaper than KV for writes
        3. Cloudflare KV (REST) - 115ms+ latency, expensive writes ($5/1M)

        Args:
            valkey_client: Optional ValkeyClient for K8s deployments.
            upstash_client: Optional UpstashRedisClient for external deployments.
            kv_client: Optional CloudflareKVClient as last-resort fallback.
            default_ttl: Default TTL in seconds for cache entries.
            key_prefix: Prefix for all cache keys.
        """
        self._valkey_client = valkey_client
        self._upstash_client = upstash_client
        self._kv_client = kv_client
        self._default_ttl = default_ttl
        self._key_prefix = key_prefix
        self._log = logger.bind(component="intelligence_cache")
        self._valkey_healthy: bool | None = None  # Cache health status
        self._upstash_healthy: bool | None = None  # Cache health status

    async def _get_client(self) -> CacheClient | None:
        """Get the best available cache client with automatic fallback.

        Priority (fastest/cheapest first):
        1. Valkey (K8s internal) - 1-5ms latency
        2. Upstash Redis (REST) - 10-30ms, 227x cheaper than KV for writes
        3. Cloudflare KV (REST) - 115ms+, expensive ($5/1M writes), last resort

        Returns:
            Cache client or None if no cache is available.
        """
        # Try Valkey first (fastest for K8s deployments)
        if self._valkey_client is None:
            self._valkey_client = get_valkey_client()

        if self._valkey_client is not None:
            # Check if we've already determined Valkey is unhealthy
            if self._valkey_healthy is False:
                self._log.debug("Skipping Valkey (previously unhealthy)")
            else:
                try:
                    healthy = await self._valkey_client.ping()
                    self._valkey_healthy = healthy
                    if healthy:
                        return self._valkey_client
                    else:
                        self._log.warning("Valkey ping returned False, trying Upstash")
                except Exception as e:
                    self._valkey_healthy = False
                    self._log.warning(
                        "Valkey unreachable, trying Upstash Redis",
                        error=str(e),
                    )

        # Try Upstash Redis second (10-30ms, much cheaper than Cloudflare KV)
        if self._upstash_client is None:
            self._upstash_client = get_upstash_client()

        if self._upstash_client is not None:
            if self._upstash_healthy is False:
                self._log.debug("Skipping Upstash (previously unhealthy)")
            else:
                try:
                    healthy = await self._upstash_client.ping()
                    self._upstash_healthy = healthy
                    if healthy:
                        self._log.debug("Using Upstash Redis as cache backend")
                        return self._upstash_client
                    else:
                        self._log.warning("Upstash ping returned False, trying Cloudflare KV")
                except Exception as e:
                    self._upstash_healthy = False
                    self._log.warning(
                        "Upstash unreachable, falling back to Cloudflare KV",
                        error=str(e),
                    )

        # Fallback to Cloudflare KV (last resort - expensive and slow)
        if self._kv_client is None:
            self._kv_client = get_kv_client()

        if self._kv_client is not None:
            self._log.debug("Using Cloudflare KV as cache backend (last resort)")
            return self._kv_client

        self._log.warning("No cache backend available")
        return None

    def reset_health_cache(self) -> None:
        """Reset the cached health status to force re-check on next call."""
        self._valkey_healthy = None
        self._upstash_healthy = None

    def _generate_cache_key(self, query: str, intent: str) -> str:
        """Generate a cache key from query and intent.

        Format: intel:{intent}:{sha256(query)[:16]}

        Args:
            query: The search query
            intent: The intent type (e.g., "self_healing", "pattern_match")

        Returns:
            Cache key string
        """
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        return f"{self._key_prefix}:{intent}:{query_hash}"

    def _get_ttl_for_intent(self, intent: str) -> int:
        """Get the TTL for a given intent type.

        Args:
            intent: The intent type

        Returns:
            TTL in seconds
        """
        return INTENT_TTL_MAP.get(intent, self._default_ttl)

    def _serialize(self, value: Any) -> str:
        """Serialize a value for cache storage.

        Args:
            value: Value to serialize

        Returns:
            JSON string
        """
        return json.dumps(value, default=str)

    def _deserialize(self, value: str) -> Any:
        """Deserialize a value from cache storage.

        Args:
            value: JSON string

        Returns:
            Deserialized value
        """
        return json.loads(value)

    async def get_or_search(
        self,
        query: str,
        intent: str,
        search_fn: Callable[[], Awaitable[T]],
        ttl: int | None = None,
    ) -> CachedResult:
        """Get cached result or execute search function.

        This is the primary method for cache-through access. It:
        1. Checks Valkey cache for existing result
        2. If cache miss, executes the search function
        3. Stores the result in cache with appropriate TTL
        4. Returns CachedResult with source and latency info

        Args:
            query: The search query string
            intent: The intent type for TTL selection
            search_fn: Async function to execute on cache miss
            ttl: Optional TTL override (uses intent-based TTL if not provided)

        Returns:
            CachedResult containing data, source, and latency_ms

        Example:
            ```python
            result = await cache.get_or_search(
                query="login button selector",
                intent="pattern_match",
                search_fn=lambda: pattern_service.find_patterns("login button")
            )
            ```
        """
        start_time = time.perf_counter()
        cache_key = self._generate_cache_key(query, intent)
        effective_ttl = ttl if ttl is not None else self._get_ttl_for_intent(intent)

        client = await self._get_client()

        # Try cache lookup
        if client is not None:
            try:
                cached_value = await client.get(cache_key)
                if cached_value is not None:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    self._log.debug(
                        "Cache hit",
                        cache_key=cache_key,
                        intent=intent,
                        latency_ms=round(latency_ms, 2),
                    )
                    return CachedResult(
                        data=self._deserialize(cached_value),
                        source=CacheSource.VALKEY,
                        latency_ms=latency_ms,
                    )
            except Exception as e:
                self._log.warning(
                    "Cache lookup failed, proceeding with search",
                    cache_key=cache_key,
                    error=str(e),
                )

        # Cache miss - execute search function
        self._log.debug(
            "Cache miss, executing search",
            cache_key=cache_key,
            intent=intent,
        )

        try:
            result = await search_fn()
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Determine source based on result type or default to LLM
            # This is a heuristic; callers can override by returning annotated results
            source = CacheSource.LLM
            if isinstance(result, dict) and "_source" in result:
                source_str = result.pop("_source", "llm")
                try:
                    source = CacheSource(source_str)
                except ValueError:
                    source = CacheSource.LLM

            # Store in cache
            if client is not None and result is not None:
                try:
                    await client.set(
                        cache_key,
                        self._serialize(result),
                        ex=effective_ttl,
                    )
                    self._log.debug(
                        "Cached search result",
                        cache_key=cache_key,
                        ttl=effective_ttl,
                    )
                except Exception as e:
                    self._log.warning(
                        "Failed to cache result",
                        cache_key=cache_key,
                        error=str(e),
                    )

            return CachedResult(
                data=result,
                source=source,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._log.error(
                "Search function failed",
                cache_key=cache_key,
                intent=intent,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def get(self, query: str, intent: str) -> CachedResult | None:
        """Get a cached result without executing a search function.

        Args:
            query: The search query string
            intent: The intent type

        Returns:
            CachedResult if found, None otherwise
        """
        start_time = time.perf_counter()
        cache_key = self._generate_cache_key(query, intent)
        client = await self._get_client()

        if client is None:
            return None

        try:
            cached_value = await client.get(cache_key)
            if cached_value is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                # Determine source based on client type
                source = CacheSource.VALKEY if isinstance(client, ValkeyClient) else CacheSource.VALKEY
                return CachedResult(
                    data=self._deserialize(cached_value),
                    source=source,
                    latency_ms=latency_ms,
                )
        except Exception as e:
            self._log.warning(
                "Cache get failed",
                cache_key=cache_key,
                error=str(e),
            )

        return None

    async def set(
        self,
        query: str,
        intent: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Explicitly set a cache entry.

        Args:
            query: The search query string
            intent: The intent type
            value: Value to cache
            ttl: Optional TTL override

        Returns:
            True if cached successfully, False otherwise
        """
        cache_key = self._generate_cache_key(query, intent)
        effective_ttl = ttl if ttl is not None else self._get_ttl_for_intent(intent)
        client = await self._get_client()

        if client is None:
            return False

        try:
            await client.set(cache_key, self._serialize(value), ex=effective_ttl)
            self._log.debug(
                "Set cache entry",
                cache_key=cache_key,
                ttl=effective_ttl,
            )
            return True
        except Exception as e:
            self._log.warning(
                "Failed to set cache entry",
                cache_key=cache_key,
                error=str(e),
            )
            return False

    async def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern.

        Uses Redis SCAN + DELETE for pattern matching (e.g., "intel:self_healing:*").
        Only works with Valkey/Redis backend. For Cloudflare KV, pattern invalidation
        is not supported - use delete() for individual keys.

        Note: Pattern matching requires iterating through keys, which may be
        slow for large datasets. For single-key invalidation, use delete().

        Args:
            pattern: Redis glob pattern (e.g., "intel:*", "intel:self_healing:*")

        Returns:
            Number of keys deleted
        """
        client = await self._get_client()
        if client is None:
            self._log.warning("Cannot invalidate: no cache client available")
            return 0

        # Pattern invalidation only works with Valkey/Redis
        if not isinstance(client, ValkeyClient):
            self._log.warning(
                "Pattern invalidation not supported with current cache backend",
                backend=type(client).__name__,
            )
            return 0

        try:
            redis_client = await client._get_client()
            deleted_count = 0
            cursor = 0

            # Use SCAN to find matching keys
            while True:
                cursor, keys = await redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100,
                )

                if keys:
                    await redis_client.delete(*keys)
                    deleted_count += len(keys)

                if cursor == 0:
                    break

            self._log.info(
                "Invalidated cache entries",
                pattern=pattern,
                deleted_count=deleted_count,
            )
            return deleted_count

        except Exception as e:
            self._log.error(
                "Cache invalidation failed",
                pattern=pattern,
                error=str(e),
                error_type=type(e).__name__,
            )
            return 0

    async def invalidate_for_project(self, org_id: str, project_id: str) -> int:
        """Invalidate all cache entries for a specific project.

        This is useful when project data changes and cached intelligence
        results may be stale.

        Args:
            org_id: Organization ID
            project_id: Project ID

        Returns:
            Number of keys deleted
        """
        # Generate a pattern that matches all intents for this project
        # We store project-scoped keys with an optional project suffix
        project_key_part = hashlib.sha256(
            f"{org_id}:{project_id}".encode("utf-8")
        ).hexdigest()[:8]

        # Invalidate any keys containing the project identifier
        # This uses a broad pattern that may need refinement based on actual key structure
        pattern = f"{self._key_prefix}:*:{project_key_part}*"

        deleted = await self.invalidate(pattern)

        # Also try direct project-prefixed pattern
        direct_pattern = f"{self._key_prefix}:{org_id}:{project_id}:*"
        deleted += await self.invalidate(direct_pattern)

        self._log.info(
            "Invalidated project cache",
            org_id=org_id,
            project_id=project_id,
            deleted_count=deleted,
        )

        return deleted

    async def delete(self, query: str, intent: str) -> bool:
        """Delete a specific cache entry.

        Args:
            query: The search query string
            intent: The intent type

        Returns:
            True if deleted, False otherwise
        """
        cache_key = self._generate_cache_key(query, intent)
        client = await self._get_client()

        if client is None:
            return False

        try:
            await client.delete(cache_key)
            self._log.debug("Deleted cache entry", cache_key=cache_key)
            return True
        except Exception as e:
            self._log.warning(
                "Failed to delete cache entry",
                cache_key=cache_key,
                error=str(e),
            )
            return False

    async def health_check(self) -> dict[str, Any]:
        """Check the health of the intelligence cache.

        Returns:
            Health status dictionary with details about which backend is in use.
        """
        # Check Valkey first (K8s internal)
        valkey_healthy = False
        valkey_error = None
        if self._valkey_client is None:
            self._valkey_client = get_valkey_client()

        if self._valkey_client is not None:
            try:
                valkey_healthy = await self._valkey_client.ping()
            except Exception as e:
                valkey_error = str(e)

        # Check Upstash Redis (preferred fallback)
        upstash_healthy = False
        upstash_error = None
        if self._upstash_client is None:
            self._upstash_client = get_upstash_client()

        if self._upstash_client is not None:
            try:
                upstash_healthy = await self._upstash_client.ping()
            except Exception as e:
                upstash_error = str(e)

        # Check Cloudflare KV (last resort)
        kv_healthy = False
        if self._kv_client is None:
            self._kv_client = get_kv_client()

        if self._kv_client is not None:
            try:
                # KV doesn't have ping, but if client exists it's configured
                kv_healthy = True
            except Exception:
                pass

        # Determine overall health and backend
        if valkey_healthy:
            return {
                "healthy": True,
                "component": "intelligence_cache",
                "backend": "valkey",
                "fallback_available": upstash_healthy or kv_healthy,
                "upstash_available": upstash_healthy,
                "kv_available": kv_healthy,
            }
        elif upstash_healthy:
            return {
                "healthy": True,
                "component": "intelligence_cache",
                "backend": "upstash_redis",
                "valkey_error": valkey_error,
                "fallback_available": kv_healthy,
                "note": "Using Upstash Redis (Valkey unreachable)",
            }
        elif kv_healthy:
            return {
                "healthy": True,
                "component": "intelligence_cache",
                "backend": "cloudflare_kv",
                "valkey_error": valkey_error,
                "upstash_error": upstash_error,
                "note": "Using Cloudflare KV (last resort - consider configuring Upstash)",
            }
        else:
            return {
                "healthy": False,
                "reason": valkey_error or upstash_error or "No cache backend available",
                "component": "intelligence_cache",
            }


# Global instance (lazy initialized)
_intelligence_cache: IntelligenceCache | None = None


def get_intelligence_cache() -> IntelligenceCache:
    """Get or create the global IntelligenceCache instance.

    Returns:
        IntelligenceCache instance
    """
    global _intelligence_cache

    if _intelligence_cache is None:
        _intelligence_cache = IntelligenceCache()

    return _intelligence_cache


def reset_intelligence_cache() -> None:
    """Reset the global IntelligenceCache instance."""
    global _intelligence_cache
    _intelligence_cache = None
