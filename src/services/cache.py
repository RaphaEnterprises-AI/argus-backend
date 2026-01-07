"""
Caching service using Upstash Redis.

Provides caching decorators for:
- Quality scores (TTL: 5 min)
- LLM responses (TTL: 24 hours)
- Healing patterns (TTL: 7 days)
"""

import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, ParamSpec

from src.config import get_settings

logger = logging.getLogger(__name__)

# Type variables for generic decorators
P = ParamSpec("P")
T = TypeVar("T")

# Global Redis client (lazy initialized)
_redis_client: Optional[Any] = None


def get_redis_client():
    """Get or create Redis client (lazy initialization)."""
    global _redis_client

    if _redis_client is not None:
        return _redis_client

    settings = get_settings()

    if not settings.upstash_redis_rest_url or not settings.upstash_redis_rest_token:
        logger.warning("Upstash Redis not configured - caching disabled")
        return None

    try:
        from upstash_redis import Redis

        token = settings.upstash_redis_rest_token
        if hasattr(token, 'get_secret_value'):
            token = token.get_secret_value()

        _redis_client = Redis(
            url=settings.upstash_redis_rest_url,
            token=token
        )
        logger.info("Upstash Redis client initialized")
        return _redis_client
    except ImportError:
        logger.warning("upstash-redis not installed - run: pip install upstash-redis")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Redis client: {e}")
        return None


def _make_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a cache key from prefix and arguments."""
    # Create a deterministic hash of the arguments
    key_data = json.dumps({
        "args": [str(a) for a in args],
        "kwargs": {k: str(v) for k, v in sorted(kwargs.items())}
    }, sort_keys=True)

    key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
    return f"argus:{prefix}:{key_hash}"


def _serialize(value: Any) -> str:
    """Serialize value for Redis storage."""
    return json.dumps(value, default=str)


def _deserialize(value: str) -> Any:
    """Deserialize value from Redis storage."""
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

            redis = get_redis_client()
            if redis is None:
                return await func(*args, **kwargs)

            ttl = ttl_seconds or settings.cache_ttl_quality_scores
            cache_key = _make_cache_key(key_prefix, *args, **kwargs)

            try:
                # Try to get from cache
                cached = redis.get(cache_key)
                if cached:
                    logger.debug(f"Cache HIT: {cache_key}")
                    return _deserialize(cached)

                logger.debug(f"Cache MISS: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            try:
                redis.set(cache_key, _serialize(result), ex=ttl)
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

            redis = get_redis_client()
            if redis is None:
                return await func(*args, **kwargs)

            ttl = ttl_seconds or settings.cache_ttl_llm_responses
            cache_key = _make_cache_key(key_prefix, *args, **kwargs)

            try:
                cached = redis.get(cache_key)
                if cached:
                    logger.debug(f"LLM Cache HIT: {cache_key}")
                    return _deserialize(cached)

                logger.debug(f"LLM Cache MISS: {cache_key}")
            except Exception as e:
                logger.warning(f"LLM cache read error: {e}")

            result = await func(*args, **kwargs)

            try:
                redis.set(cache_key, _serialize(result), ex=ttl)
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

            redis = get_redis_client()
            if redis is None:
                return await func(*args, **kwargs)

            ttl = ttl_seconds or settings.cache_ttl_healing_patterns
            cache_key = _make_cache_key(key_prefix, *args, **kwargs)

            try:
                cached = redis.get(cache_key)
                if cached:
                    logger.debug(f"Healing Cache HIT: {cache_key}")
                    return _deserialize(cached)

                logger.debug(f"Healing Cache MISS: {cache_key}")
            except Exception as e:
                logger.warning(f"Healing cache read error: {e}")

            result = await func(*args, **kwargs)

            # Only cache if we found a healing pattern
            if result is not None:
                try:
                    redis.set(cache_key, _serialize(result), ex=ttl)
                except Exception as e:
                    logger.warning(f"Healing cache write error: {e}")

            return result

        return wrapper
    return decorator


# Direct cache operations for non-decorator usage

async def get_cached(key: str) -> Optional[Any]:
    """Get a value from cache directly."""
    redis = get_redis_client()
    if redis is None:
        return None

    try:
        value = redis.get(f"argus:{key}")
        return _deserialize(value) if value else None
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
        return None


async def set_cached(key: str, value: Any, ttl_seconds: int = 300) -> bool:
    """Set a value in cache directly."""
    redis = get_redis_client()
    if redis is None:
        return False

    try:
        redis.set(f"argus:{key}", _serialize(value), ex=ttl_seconds)
        return True
    except Exception as e:
        logger.warning(f"Cache set error: {e}")
        return False


async def delete_cached(key: str) -> bool:
    """Delete a value from cache."""
    redis = get_redis_client()
    if redis is None:
        return False

    try:
        redis.delete(f"argus:{key}")
        return True
    except Exception as e:
        logger.warning(f"Cache delete error: {e}")
        return False


async def invalidate_pattern(pattern: str) -> int:
    """
    Invalidate all keys matching a pattern.

    Args:
        pattern: Key pattern (e.g., "score:*" to invalidate all scores)

    Returns:
        Number of keys deleted
    """
    redis = get_redis_client()
    if redis is None:
        return 0

    try:
        # Note: SCAN is safer for production, but Upstash supports KEYS for small datasets
        keys = redis.keys(f"argus:{pattern}")
        if keys:
            redis.delete(*keys)
            return len(keys)
        return 0
    except Exception as e:
        logger.warning(f"Cache invalidate error: {e}")
        return 0


# Health check

async def check_cache_health() -> dict:
    """Check if cache is healthy and return stats."""
    redis = get_redis_client()

    if redis is None:
        return {
            "healthy": False,
            "reason": "Redis not configured",
            "enabled": get_settings().cache_enabled
        }

    try:
        # Test with a simple ping
        redis.set("argus:health:ping", "pong", ex=60)
        result = redis.get("argus:health:ping")

        if result == "pong":
            return {
                "healthy": True,
                "provider": "upstash",
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
