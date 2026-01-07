"""
Caching service using Cloudflare KV REST API.

Provides caching decorators for:
- Quality scores (TTL: 5 min)
- LLM responses (TTL: 24 hours)
- Healing patterns (TTL: 7 days)

Uses Cloudflare KV via REST API for Python backend on Railway.
"""

import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, ParamSpec

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

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
        self._client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def get(self, key: str) -> Optional[str]:
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


# Global KV client (lazy initialized)
_kv_client: Optional[CloudflareKVClient] = None


def get_kv_client() -> Optional[CloudflareKVClient]:
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
    ttl_seconds: Optional[int] = None,
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
    ttl_seconds: Optional[int] = None,
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
    ttl_seconds: Optional[int] = None,
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

async def get_cached(key: str) -> Optional[Any]:
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
