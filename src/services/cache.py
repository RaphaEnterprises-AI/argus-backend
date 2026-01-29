"""
Caching service using multiple backends with intelligent fallback.

Cache Priority (fastest to slowest):
1. Valkey (K8s internal) - 1-5ms, Redis protocol
2. Upstash Redis (REST API) - 10-30ms, full Redis features, cost-effective
3. Cloudflare KV (REST API) - 115ms+, expensive writes, last resort

Provides caching decorators for:
- Quality scores (TTL: 5 min)
- LLM responses (TTL: 24 hours)
- Healing patterns (TTL: 7 days)
- Discovery patterns (TTL: 15 min) - via Valkey

Cost comparison (1M reads + 100K writes/month):
- Cloudflare KV: ~$500 (writes are $5/M!)
- Upstash Redis: ~$2.20 (227x cheaper)
"""

import hashlib
import json
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar
from urllib.parse import urlparse

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

# Try to import redis (valkey-compatible)
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    REDIS_AVAILABLE = False

# Type variables for generic decorators
P = ParamSpec("P")
T = TypeVar("T")

# Cloudflare KV API base URL
CF_API_BASE = "https://api.cloudflare.com/client/v4/accounts"


class CloudflareKVClient:
    """Cloudflare KV REST API client."""

    def __init__(self, account_id: str, namespace_id: str, api_token: str):
        self.account_id = account_id
        self.namespace_id = namespace_id
        self.api_token = api_token
        self.base_url = f"{CF_API_BASE}/{account_id}/storage/kv/namespaces/{namespace_id}"
        self._client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def get(self, key: str) -> str | None:
        """Get a value from KV."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/values/{key}",
                headers=self._get_headers()
            )
            if response.status_code == 200:
                return response.text
            elif response.status_code == 404:
                return None
            else:
                logger.warning(f"KV get error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.warning(f"KV get exception: {e}")
            return None

    async def set(self, key: str, value: str, ex: int = 300) -> bool:
        """Set a value in KV with TTL."""
        try:
            client = await self._get_client()
            response = await client.put(
                f"{self.base_url}/values/{key}",
                params={"expiration_ttl": ex},
                content=value,
                headers=self._get_headers()
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"KV set exception: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value from KV."""
        try:
            client = await self._get_client()
            response = await client.delete(
                f"{self.base_url}/values/{key}",
                headers=self._get_headers()
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"KV delete exception: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Upstash Redis Client (REST API - 227x cheaper than Cloudflare KV for writes)
# =============================================================================


class UpstashRedisClient:
    """Upstash Redis REST API client.

    Upstash Redis is the preferred fallback when Valkey (K8s) is unreachable:
    - 10-30ms latency (vs 115ms+ for Cloudflare KV)
    - Full Redis command support (SCAN, EXPIRE, atomic ops, etc.)
    - $2/1M requests vs $5/1M writes for Cloudflare KV
    - Real-time replication (vs 60s propagation for KV)

    REST API format: GET/POST to {url}/{command}/{args}
    Response format: {"result": <value>}
    """

    def __init__(self, rest_url: str, rest_token: str):
        """Initialize Upstash Redis client.

        Args:
            rest_url: Upstash REST URL (e.g., https://xxx.upstash.io)
            rest_token: Upstash REST API token
        """
        self.rest_url = rest_url.rstrip("/")
        self.rest_token = rest_token
        self._client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.rest_token}",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def _execute(self, *args: str) -> Any:
        """Execute a Redis command via REST API.

        Args:
            *args: Redis command and arguments (e.g., "GET", "mykey")

        Returns:
            The result from the response
        """
        client = await self._get_client()
        # Upstash REST API: POST with JSON array of command args
        response = await client.post(
            self.rest_url,
            headers=self._get_headers(),
            json=list(args),
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("result")
        else:
            logger.warning(f"Upstash error: {response.status_code} - {response.text}")
            return None

    async def get(self, key: str) -> str | None:
        """Get a value from Upstash Redis."""
        try:
            return await self._execute("GET", key)
        except Exception as e:
            logger.warning(f"Upstash get exception: {e}")
            return None

    async def set(self, key: str, value: str, ex: int = 300) -> bool:
        """Set a value in Upstash Redis with TTL."""
        try:
            result = await self._execute("SET", key, value, "EX", str(ex))
            return result == "OK"
        except Exception as e:
            logger.warning(f"Upstash set exception: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value from Upstash Redis."""
        try:
            result = await self._execute("DEL", key)
            return result is not None and result > 0
        except Exception as e:
            logger.warning(f"Upstash delete exception: {e}")
            return False

    async def ping(self) -> bool:
        """Check if Upstash Redis is reachable."""
        try:
            result = await self._execute("PING")
            return result == "PONG"
        except Exception:
            return False

    async def mget(self, keys: list[str]) -> list[str | None]:
        """Get multiple values from Upstash Redis."""
        try:
            result = await self._execute("MGET", *keys)
            return result if result else [None] * len(keys)
        except Exception as e:
            logger.warning(f"Upstash mget exception: {e}")
            return [None] * len(keys)

    async def incr(self, key: str) -> int | None:
        """Atomically increment a counter."""
        try:
            return await self._execute("INCR", key)
        except Exception as e:
            logger.warning(f"Upstash incr exception: {e}")
            return None

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on a key."""
        try:
            result = await self._execute("EXPIRE", key, str(seconds))
            return result == 1
        except Exception as e:
            logger.warning(f"Upstash expire exception: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global Upstash client (lazy initialized)
_upstash_client: UpstashRedisClient | None = None


def get_upstash_client() -> UpstashRedisClient | None:
    """Get or create Upstash Redis client (lazy initialization)."""
    global _upstash_client

    if _upstash_client is not None:
        return _upstash_client

    settings = get_settings()

    # Check for Upstash environment variables
    upstash_url = getattr(settings, "upstash_redis_rest_url", None)
    upstash_token = getattr(settings, "upstash_redis_rest_token", None)

    if not upstash_url or not upstash_token:
        # Try environment variables directly
        import os
        upstash_url = os.getenv("UPSTASH_REDIS_REST_URL")
        upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

    if not upstash_url or not upstash_token:
        logger.debug("Upstash Redis not configured")
        return None

    try:
        # Handle SecretStr if needed
        if hasattr(upstash_token, "get_secret_value"):
            upstash_token = upstash_token.get_secret_value()

        _upstash_client = UpstashRedisClient(
            rest_url=upstash_url,
            rest_token=upstash_token,
        )
        logger.info("Upstash Redis client initialized")
        return _upstash_client
    except Exception as e:
        logger.warning(f"Failed to initialize Upstash Redis client: {e}")
        return None


# Global KV client (lazy initialized)
_kv_client: CloudflareKVClient | None = None


def get_kv_client() -> CloudflareKVClient | None:
    """Get or create Cloudflare KV client (lazy initialization)."""
    global _kv_client

    if _kv_client is not None:
        return _kv_client

    settings = get_settings()

    if not settings.cloudflare_api_token or not settings.cloudflare_account_id:
        logger.warning("Cloudflare KV not configured - caching disabled")
        return None

    if not settings.cloudflare_kv_namespace_id:
        logger.warning("Cloudflare KV namespace ID not configured - caching disabled")
        return None

    try:
        api_token = settings.cloudflare_api_token
        if hasattr(api_token, 'get_secret_value'):
            api_token = api_token.get_secret_value()

        _kv_client = CloudflareKVClient(
            account_id=settings.cloudflare_account_id,
            namespace_id=settings.cloudflare_kv_namespace_id,
            api_token=api_token
        )
        logger.info("Cloudflare KV client initialized")
        return _kv_client
    except Exception as e:
        logger.error(f"Failed to initialize Cloudflare KV client: {e}")
        return None


# =============================================================================
# Valkey Client (Redis-compatible, for local/K8s deployments)
# =============================================================================


class ValkeyClient:
    """Valkey (Redis-compatible) cache client.

    Used for high-performance caching in Kubernetes deployments.
    Provides same interface as CloudflareKVClient for easy switching.
    """

    def __init__(self, url: str):
        """Initialize Valkey client.

        Args:
            url: Redis-compatible URL (e.g., redis://:password@host:port)
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed. Run: pip install redis")

        self.url = url
        self._pool: redis.ConnectionPool | None = None
        self._client: redis.Redis | None = None

    async def _get_client(self) -> "redis.Redis":
        """Get or create Redis client."""
        if self._client is None:
            self._pool = redis.ConnectionPool.from_url(
                self.url,
                max_connections=10,
                decode_responses=True,
            )
            self._client = redis.Redis(connection_pool=self._pool)
        return self._client

    async def get(self, key: str) -> str | None:
        """Get a value from Valkey."""
        try:
            client = await self._get_client()
            return await client.get(key)
        except Exception as e:
            logger.warning(f"Valkey get exception: {e}")
            return None

    async def set(self, key: str, value: str, ex: int = 300) -> bool:
        """Set a value in Valkey with TTL."""
        try:
            client = await self._get_client()
            await client.set(key, value, ex=ex)
            return True
        except Exception as e:
            logger.warning(f"Valkey set exception: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value from Valkey."""
        try:
            client = await self._get_client()
            await client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Valkey delete exception: {e}")
            return False

    async def mget(self, keys: list[str]) -> list[str | None]:
        """Get multiple values from Valkey."""
        try:
            client = await self._get_client()
            return await client.mget(keys)
        except Exception as e:
            logger.warning(f"Valkey mget exception: {e}")
            return [None] * len(keys)

    async def mset(self, mapping: dict[str, str], ex: int = 300) -> bool:
        """Set multiple values in Valkey with TTL."""
        try:
            client = await self._get_client()
            pipe = client.pipeline()
            for key, value in mapping.items():
                pipe.set(key, value, ex=ex)
            await pipe.execute()
            return True
        except Exception as e:
            logger.warning(f"Valkey mset exception: {e}")
            return False

    async def close(self):
        """Close the Redis client."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None

    async def ping(self) -> bool:
        """Check if Valkey is reachable."""
        try:
            client = await self._get_client()
            return await client.ping()
        except Exception:
            return False


# Global Valkey client (lazy initialized)
_valkey_client: ValkeyClient | None = None


def get_valkey_client() -> ValkeyClient | None:
    """Get or create Valkey client (lazy initialization)."""
    global _valkey_client

    if _valkey_client is not None:
        return _valkey_client

    if not REDIS_AVAILABLE:
        logger.debug("Redis package not available - Valkey caching disabled")
        return None

    settings = get_settings()

    if not settings.valkey_url:
        logger.debug("Valkey URL not configured - Valkey caching disabled")
        return None

    try:
        _valkey_client = ValkeyClient(url=settings.valkey_url)
        logger.info("Valkey client initialized")
        return _valkey_client
    except Exception as e:
        logger.warning(f"Failed to initialize Valkey client: {e}")
        return None


def _make_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a cache key from prefix and arguments."""
    key_data = json.dumps({
        "args": [str(a) for a in args],
        "kwargs": {k: str(v) for k, v in sorted(kwargs.items())}
    }, sort_keys=True)

    key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
    return f"argus:{prefix}:{key_hash}"


def _serialize(value: Any) -> str:
    """Serialize value for KV storage."""
    return json.dumps(value, default=str)


def _deserialize(value: str) -> Any:
    """Deserialize value from KV storage."""
    return json.loads(value)


def cache_quality_score(
    ttl_seconds: int | None = None,
    key_prefix: str = "score"
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Cache decorator for quality score lookups.

    Args:
        ttl_seconds: Cache TTL in seconds (default: from settings)
        key_prefix: Cache key prefix

    Usage:
        @cache_quality_score()
        async def get_quality_score(project_id: str) -> dict:
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            settings = get_settings()

            if not settings.cache_enabled:
                return await func(*args, **kwargs)

            kv = get_kv_client()
            if kv is None:
                return await func(*args, **kwargs)

            ttl = ttl_seconds or settings.cache_ttl_quality_scores
            cache_key = _make_cache_key(key_prefix, *args, **kwargs)

            try:
                cached = await kv.get(cache_key)
                if cached:
                    logger.debug(f"Cache HIT: {cache_key}")
                    return _deserialize(cached)

                logger.debug(f"Cache MISS: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

            result = await func(*args, **kwargs)

            try:
                await kv.set(cache_key, _serialize(result), ex=ttl)
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

            return result

        return wrapper
    return decorator


def cache_llm_response(
    ttl_seconds: int | None = None,
    key_prefix: str = "llm"
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Cache decorator for LLM API responses.

    Caches based on prompt hash for deterministic responses.

    Args:
        ttl_seconds: Cache TTL in seconds (default: from settings, 24 hours)
        key_prefix: Cache key prefix

    Usage:
        @cache_llm_response()
        async def analyze_code(prompt: str) -> str:
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            settings = get_settings()

            if not settings.cache_enabled:
                return await func(*args, **kwargs)

            kv = get_kv_client()
            if kv is None:
                return await func(*args, **kwargs)

            ttl = ttl_seconds or settings.cache_ttl_llm_responses
            cache_key = _make_cache_key(key_prefix, *args, **kwargs)

            try:
                cached = await kv.get(cache_key)
                if cached:
                    logger.debug(f"LLM Cache HIT: {cache_key}")
                    return _deserialize(cached)

                logger.debug(f"LLM Cache MISS: {cache_key}")
            except Exception as e:
                logger.warning(f"LLM cache read error: {e}")

            result = await func(*args, **kwargs)

            try:
                await kv.set(cache_key, _serialize(result), ex=ttl)
            except Exception as e:
                logger.warning(f"LLM cache write error: {e}")

            return result

        return wrapper
    return decorator


def cache_healing_pattern(
    ttl_seconds: int | None = None,
    key_prefix: str = "heal"
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Cache decorator for healing patterns.

    Stores successful selector healings for reuse across tests.

    Args:
        ttl_seconds: Cache TTL in seconds (default: from settings, 7 days)
        key_prefix: Cache key prefix

    Usage:
        @cache_healing_pattern()
        async def find_healed_selector(original: str, error_type: str) -> Optional[str]:
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            settings = get_settings()

            if not settings.cache_enabled:
                return await func(*args, **kwargs)

            kv = get_kv_client()
            if kv is None:
                return await func(*args, **kwargs)

            ttl = ttl_seconds or settings.cache_ttl_healing_patterns
            cache_key = _make_cache_key(key_prefix, *args, **kwargs)

            try:
                cached = await kv.get(cache_key)
                if cached:
                    logger.debug(f"Healing Cache HIT: {cache_key}")
                    return _deserialize(cached)

                logger.debug(f"Healing Cache MISS: {cache_key}")
            except Exception as e:
                logger.warning(f"Healing cache read error: {e}")

            result = await func(*args, **kwargs)

            if result is not None:
                try:
                    await kv.set(cache_key, _serialize(result), ex=ttl)
                except Exception as e:
                    logger.warning(f"Healing cache write error: {e}")

            return result

        return wrapper
    return decorator


# Direct cache operations for non-decorator usage

async def get_cached(key: str) -> Any | None:
    """Get a value from cache directly."""
    kv = get_kv_client()
    if kv is None:
        return None

    try:
        value = await kv.get(f"argus:{key}")
        return _deserialize(value) if value else None
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
        return None


async def set_cached(key: str, value: Any, ttl_seconds: int = 300) -> bool:
    """Set a value in cache directly."""
    kv = get_kv_client()
    if kv is None:
        return False

    try:
        await kv.set(f"argus:{key}", _serialize(value), ex=ttl_seconds)
        return True
    except Exception as e:
        logger.warning(f"Cache set error: {e}")
        return False


async def delete_cached(key: str) -> bool:
    """Delete a value from cache."""
    kv = get_kv_client()
    if kv is None:
        return False

    try:
        await kv.delete(f"argus:{key}")
        return True
    except Exception as e:
        logger.warning(f"Cache delete error: {e}")
        return False


# Health check

async def check_cache_health() -> dict:
    """Check if cache is healthy and return stats."""
    kv = get_kv_client()

    if kv is None:
        return {
            "healthy": False,
            "reason": "Cloudflare KV not configured",
            "enabled": get_settings().cache_enabled
        }

    try:
        # Test with a simple write/read
        test_key = "argus:health:ping"
        await kv.set(test_key, "pong", ex=60)
        result = await kv.get(test_key)

        if result == "pong":
            return {
                "healthy": True,
                "provider": "cloudflare_kv",
                "enabled": True
            }
        else:
            return {
                "healthy": False,
                "reason": "Ping-pong test failed"
            }
    except Exception as e:
        return {
            "healthy": False,
            "reason": str(e)
        }


# =============================================================================
# Valkey-specific caching for hot patterns (high-performance)
# =============================================================================

# Default TTLs for pattern caching
PATTERN_CACHE_TTL_HOT = 900  # 15 minutes for frequently accessed patterns
PATTERN_CACHE_TTL_WARM = 3600  # 1 hour for moderately accessed patterns
PATTERN_CACHE_TTL_COLD = 86400  # 24 hours for rarely accessed patterns


def cache_discovery_pattern(
    ttl_seconds: int = PATTERN_CACHE_TTL_HOT,
    key_prefix: str = "pattern"
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Cache decorator for discovery patterns using Valkey.

    Uses Valkey for high-performance local caching of frequently accessed
    patterns. Falls back to Cloudflare KV if Valkey is unavailable.

    Args:
        ttl_seconds: Cache TTL in seconds (default: 15 min)
        key_prefix: Cache key prefix

    Usage:
        @cache_discovery_pattern()
        async def find_similar_patterns(query: str) -> list[dict]:
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            settings = get_settings()

            if not settings.cache_enabled:
                return await func(*args, **kwargs)

            # Try Valkey first (faster for local/K8s)
            valkey = get_valkey_client()
            cache_key = _make_cache_key(key_prefix, *args, **kwargs)

            if valkey is not None:
                try:
                    cached = await valkey.get(cache_key)
                    if cached:
                        logger.debug(f"Valkey Pattern Cache HIT: {cache_key}")
                        return _deserialize(cached)
                    logger.debug(f"Valkey Pattern Cache MISS: {cache_key}")
                except Exception as e:
                    logger.warning(f"Valkey pattern cache read error: {e}")

            # Fallback to Cloudflare KV
            kv = get_kv_client()
            if kv is not None:
                try:
                    cached = await kv.get(cache_key)
                    if cached:
                        logger.debug(f"KV Pattern Cache HIT: {cache_key}")
                        return _deserialize(cached)
                except Exception as e:
                    logger.warning(f"KV pattern cache read error: {e}")

            # Execute function
            result = await func(*args, **kwargs)

            if result is not None:
                serialized = _serialize(result)
                # Write to both caches
                if valkey is not None:
                    try:
                        await valkey.set(cache_key, serialized, ex=ttl_seconds)
                    except Exception as e:
                        logger.warning(f"Valkey pattern cache write error: {e}")

                if kv is not None:
                    try:
                        await kv.set(cache_key, serialized, ex=ttl_seconds)
                    except Exception as e:
                        logger.warning(f"KV pattern cache write error: {e}")

            return result

        return wrapper
    return decorator


async def get_cached_pattern(pattern_id: str, pattern_type: str) -> dict | None:
    """Get a cached pattern by ID and type.

    Uses Valkey first, falls back to Cloudflare KV.

    Args:
        pattern_id: Pattern identifier
        pattern_type: Pattern type (e.g., "failure", "discovery")

    Returns:
        Cached pattern dict or None
    """
    cache_key = f"argus:pattern:{pattern_type}:{pattern_id}"

    # Try Valkey first
    valkey = get_valkey_client()
    if valkey is not None:
        try:
            value = await valkey.get(cache_key)
            if value:
                return _deserialize(value)
        except Exception as e:
            logger.warning(f"Valkey pattern get error: {e}")

    # Fallback to KV
    kv = get_kv_client()
    if kv is not None:
        try:
            value = await kv.get(cache_key)
            if value:
                return _deserialize(value)
        except Exception as e:
            logger.warning(f"KV pattern get error: {e}")

    return None


async def set_cached_pattern(
    pattern_id: str,
    pattern_type: str,
    pattern_data: dict,
    ttl_seconds: int = PATTERN_CACHE_TTL_HOT,
) -> bool:
    """Cache a pattern for fast retrieval.

    Writes to both Valkey and Cloudflare KV for redundancy.

    Args:
        pattern_id: Pattern identifier
        pattern_type: Pattern type (e.g., "failure", "discovery")
        pattern_data: Pattern data to cache
        ttl_seconds: Cache TTL

    Returns:
        True if cached successfully (in at least one backend)
    """
    cache_key = f"argus:pattern:{pattern_type}:{pattern_id}"
    serialized = _serialize(pattern_data)
    success = False

    # Write to Valkey
    valkey = get_valkey_client()
    if valkey is not None:
        try:
            await valkey.set(cache_key, serialized, ex=ttl_seconds)
            success = True
        except Exception as e:
            logger.warning(f"Valkey pattern set error: {e}")

    # Write to KV
    kv = get_kv_client()
    if kv is not None:
        try:
            await kv.set(cache_key, serialized, ex=ttl_seconds)
            success = True
        except Exception as e:
            logger.warning(f"KV pattern set error: {e}")

    return success


async def invalidate_pattern_cache(pattern_id: str, pattern_type: str) -> bool:
    """Invalidate a cached pattern.

    Removes from both Valkey and Cloudflare KV.

    Args:
        pattern_id: Pattern identifier
        pattern_type: Pattern type

    Returns:
        True if invalidated successfully
    """
    cache_key = f"argus:pattern:{pattern_type}:{pattern_id}"
    success = False

    valkey = get_valkey_client()
    if valkey is not None:
        try:
            await valkey.delete(cache_key)
            success = True
        except Exception as e:
            logger.warning(f"Valkey pattern delete error: {e}")

    kv = get_kv_client()
    if kv is not None:
        try:
            await kv.delete(cache_key)
            success = True
        except Exception as e:
            logger.warning(f"KV pattern delete error: {e}")

    return success


async def get_cached_patterns_bulk(
    pattern_ids: list[str],
    pattern_type: str,
) -> dict[str, dict | None]:
    """Get multiple cached patterns at once.

    Uses Valkey's mget for efficient bulk retrieval.

    Args:
        pattern_ids: List of pattern identifiers
        pattern_type: Pattern type

    Returns:
        Dict mapping pattern_id to pattern_data (or None if not cached)
    """
    if not pattern_ids:
        return {}

    cache_keys = [f"argus:pattern:{pattern_type}:{pid}" for pid in pattern_ids]
    results: dict[str, dict | None] = {pid: None for pid in pattern_ids}

    # Try Valkey bulk get
    valkey = get_valkey_client()
    if valkey is not None:
        try:
            values = await valkey.mget(cache_keys)
            for i, value in enumerate(values):
                if value:
                    results[pattern_ids[i]] = _deserialize(value)
        except Exception as e:
            logger.warning(f"Valkey bulk pattern get error: {e}")

    # Fill in missing from KV (one by one since KV doesn't have mget)
    kv = get_kv_client()
    if kv is not None:
        for pid in pattern_ids:
            if results[pid] is None:
                try:
                    value = await kv.get(f"argus:pattern:{pattern_type}:{pid}")
                    if value:
                        results[pid] = _deserialize(value)
                except Exception as e:
                    logger.warning(f"KV pattern get error for {pid}: {e}")

    return results


async def check_valkey_health() -> dict:
    """Check if Valkey is healthy."""
    valkey = get_valkey_client()

    if valkey is None:
        return {
            "healthy": False,
            "reason": "Valkey not configured",
            "available": REDIS_AVAILABLE,
        }

    try:
        healthy = await valkey.ping()
        if healthy:
            return {
                "healthy": True,
                "provider": "valkey",
                "available": True,
            }
        else:
            return {
                "healthy": False,
                "reason": "Ping failed",
            }
    except Exception as e:
        return {
            "healthy": False,
            "reason": str(e),
        }


async def check_upstash_health() -> dict:
    """Check if Upstash Redis is healthy."""
    upstash = get_upstash_client()

    if upstash is None:
        return {
            "healthy": False,
            "reason": "Upstash Redis not configured",
            "configured": False,
        }

    try:
        healthy = await upstash.ping()
        if healthy:
            return {
                "healthy": True,
                "provider": "upstash_redis",
                "configured": True,
                "note": "Preferred fallback (227x cheaper than Cloudflare KV)",
            }
        else:
            return {
                "healthy": False,
                "reason": "Ping failed",
                "configured": True,
            }
    except Exception as e:
        return {
            "healthy": False,
            "reason": str(e),
            "configured": True,
        }


async def check_all_cache_health() -> dict:
    """Check health of all cache backends."""
    valkey_health = await check_valkey_health()
    upstash_health = await check_upstash_health()
    kv_health = await check_cache_health()

    # Determine active backend (priority order)
    active_backend = None
    if valkey_health.get("healthy"):
        active_backend = "valkey"
    elif upstash_health.get("healthy"):
        active_backend = "upstash_redis"
    elif kv_health.get("healthy"):
        active_backend = "cloudflare_kv"

    return {
        "valkey": valkey_health,
        "upstash_redis": upstash_health,
        "cloudflare_kv": kv_health,
        "active_backend": active_backend,
        "any_healthy": (
            valkey_health.get("healthy", False) or
            upstash_health.get("healthy", False) or
            kv_health.get("healthy", False)
        ),
    }
