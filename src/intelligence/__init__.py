"""
Argus Intelligence Layer - Unified Instant Intelligence for <100ms responses.

This module provides the caching and intelligence infrastructure that transforms
slow LLM calls (2-5s) into fast cached responses (<100ms).

Components:
- IntelligenceCache: High-performance Valkey-backed cache for intelligence results
- CachedResult: Dataclass representing cached/computed results with latency info
- CacheSource: Enum indicating result source (valkey, cognee, llm)
- QueryRouter: Routes queries through cache, vector search, or LLM
- PrecomputedReader: Access precomputed results from background jobs

Usage:
    ```python
    from src.intelligence import IntelligenceCache, CachedResult, get_intelligence_cache

    # Get global cache instance
    cache = get_intelligence_cache()

    # Cache-through pattern for expensive operations
    result = await cache.get_or_search(
        query="how to fix login button selector",
        intent="self_healing",
        search_fn=expensive_cognee_search
    )

    # Check source and latency
    print(f"Source: {result.source}, Latency: {result.latency_ms}ms")

    # Invalidate stale cache entries
    await cache.invalidate("intel:self_healing:*")
    await cache.invalidate_for_project("org123", "proj456")
    ```

TTL Configuration:
- Default: 3600s (1 hour) for standard intents
- Real-time intents: 300s (5 minutes)
- Static knowledge: 86400s (24 hours)

Intent Types:
- self_healing: Failure pattern matching
- pattern_match: UI pattern discovery
- code_analysis: Code understanding
- test_generation: Test case generation
- realtime: Current state queries
- static_knowledge: Documentation, stable data
"""

from .cache import (
    DEFAULT_TTL,
    INTENT_TTL_MAP,
    REALTIME_TTL,
    CachedResult,
    CacheSource,
    IntelligenceCache,
    get_intelligence_cache,
    reset_intelligence_cache,
)
from .precomputed import PrecomputedReader, PrecomputedResult, get_precomputed_reader
from .query_router import (
    INTENT_KEYWORDS,
    QueryIntent,
    QueryResult,
    QueryRouter,
    get_query_router,
    reset_query_router,
    route_query,
)

__all__ = [
    # Cache classes and dataclasses
    "IntelligenceCache",
    "CachedResult",
    "CacheSource",
    # Factory functions
    "get_intelligence_cache",
    "reset_intelligence_cache",
    # TTL constants
    "DEFAULT_TTL",
    "REALTIME_TTL",
    "INTENT_TTL_MAP",
    # Query routing
    "QueryIntent",
    "QueryResult",
    "QueryRouter",
    "INTENT_KEYWORDS",
    "get_query_router",
    "reset_query_router",
    "route_query",
    # Precomputed results
    "PrecomputedReader",
    "PrecomputedResult",
    "get_precomputed_reader",
]
