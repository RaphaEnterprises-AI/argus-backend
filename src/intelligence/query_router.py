"""
Query Router with Intent Detection for the Intelligence Layer.

This module implements a tiered query routing system that optimizes for:
1. Latency - Check cache first, then precomputed, then vector, finally LLM
2. Cost - Avoid expensive LLM calls when simpler methods suffice
3. Accuracy - Use confidence scoring to determine when to escalate

The routing flow is:
    Query -> Intent Detection -> Cache Check -> Precomputed Check -> Vector Search -> LLM

RAP-238: Query router with intent detection
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
from prometheus_client import Counter, Histogram, Gauge

from src.knowledge.cognee_client import (
    CogneeKnowledgeClient,
    CogneeSearchError,
    get_cognee_client,
)
from src.intelligence.cache import (
    IntelligenceCache,
    CachedResult,
    CacheSource,
    get_intelligence_cache,
)
from src.intelligence.precomputed import (
    PrecomputedReader,
    get_precomputed_reader,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Prometheus Metrics for Intelligence Layer
# =============================================================================

INTELLIGENCE_QUERIES_TOTAL = Counter(
    "intelligence_queries_total",
    "Total intelligence queries processed",
    ["intent", "tier"],
)

INTELLIGENCE_CACHE_HITS = Counter(
    "intelligence_cache_hits_total",
    "Total cache hits in intelligence layer",
    ["intent"],
)

INTELLIGENCE_CACHE_MISSES = Counter(
    "intelligence_cache_misses_total",
    "Total cache misses in intelligence layer",
    ["intent"],
)

INTELLIGENCE_LLM_FALLBACK = Counter(
    "intelligence_llm_fallback_total",
    "Total queries that fell back to LLM",
    ["intent"],
)

INTELLIGENCE_QUERY_DURATION = Histogram(
    "intelligence_query_duration_seconds",
    "Time spent processing intelligence queries",
    ["intent", "tier"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

INTELLIGENCE_QUERY_CONFIDENCE = Gauge(
    "intelligence_query_confidence",
    "Confidence score of the last query result",
    ["intent"],
)

INTELLIGENCE_QUERY_ERRORS = Counter(
    "intelligence_query_errors_total",
    "Total errors in intelligence query processing",
    ["intent", "error_type"],
)

INTELLIGENCE_PRECOMPUTED_VALID_UNTIL = Gauge(
    "intelligence_precomputed_valid_until",
    "Unix timestamp when precomputed data expires",
    ["type"],
)


class QueryIntent(str, Enum):
    """
    Detected intent of a query, used to route to appropriate data source.

    Each intent maps to optimal retrieval strategies:
    - SIMILAR_ERRORS: Vector search for semantic similarity
    - TEST_IMPACT: Precomputed impact analysis data
    - ROOT_CAUSE: LLM reasoning with context from vector search
    - DOCUMENTATION: Vector search in docs namespace
    - CODE_CONTEXT: Vector search with code-specific embeddings
    - COVERAGE_GAPS: Precomputed coverage analysis
    - SECURITY_IMPACT: LLM reasoning with security context
    """

    SIMILAR_ERRORS = "similar_errors"
    TEST_IMPACT = "test_impact"
    ROOT_CAUSE = "root_cause"
    DOCUMENTATION = "documentation"
    CODE_CONTEXT = "code_context"
    COVERAGE_GAPS = "coverage_gaps"
    SECURITY_IMPACT = "security_impact"


# Keyword patterns for intent detection
# Each intent has associated keywords and phrases
INTENT_KEYWORDS: dict[QueryIntent, list[str]] = {
    QueryIntent.SIMILAR_ERRORS: [
        "similar error",
        "same error",
        "like this error",
        "matching failure",
        "related failure",
        "error pattern",
        "failure pattern",
        "seen before",
        "happened before",
        "previous error",
        "past failure",
        "element not found",
        "timeout",
        "assertion failed",
        "exception",
    ],
    QueryIntent.TEST_IMPACT: [
        "test impact",
        "affected tests",
        "impacted tests",
        "which tests",
        "tests to run",
        "test selection",
        "tests affected by",
        "impact analysis",
        "change impact",
        "blast radius",
        "downstream tests",
        "dependent tests",
    ],
    QueryIntent.ROOT_CAUSE: [
        "root cause",
        "why did",
        "what caused",
        "reason for",
        "explain why",
        "diagnosis",
        "analyze failure",
        "understand error",
        "investigate",
        "debug",
        "troubleshoot",
        "deep dive",
    ],
    QueryIntent.DOCUMENTATION: [
        "documentation",
        "docs",
        "how to",
        "guide",
        "tutorial",
        "example",
        "usage",
        "api reference",
        "help with",
        "explain how",
        "what is",
        "definition",
    ],
    QueryIntent.CODE_CONTEXT: [
        "code context",
        "source code",
        "implementation",
        "function",
        "method",
        "class",
        "module",
        "file",
        "codebase",
        "where is",
        "find code",
        "code for",
        "code that",
    ],
    QueryIntent.COVERAGE_GAPS: [
        "coverage gap",
        "uncovered",
        "not tested",
        "missing tests",
        "test coverage",
        "untested code",
        "coverage report",
        "low coverage",
        "coverage analysis",
        "what needs tests",
        "lacking tests",
    ],
    QueryIntent.SECURITY_IMPACT: [
        "security",
        "vulnerability",
        "cve",
        "security risk",
        "security impact",
        "exploit",
        "injection",
        "xss",
        "csrf",
        "authentication",
        "authorization",
        "sensitive data",
        "encryption",
    ],
}


@dataclass
class QueryResult:
    """
    Result of a routed query with metadata about the retrieval.

    Attributes:
        data: The actual query result data (structure depends on intent)
        source: Where the data came from (cache, precomputed, vector, llm)
        confidence: Confidence score (0.0-1.0) for the result
        latency_ms: Time taken to process the query in milliseconds
        intent: Detected intent of the query
        cache_key: Cache key used (if applicable)
        metadata: Additional metadata about the query execution
    """

    data: Any
    source: str  # "cache" | "precomputed" | "vector" | "llm"
    confidence: float
    latency_ms: int
    intent: QueryIntent
    cache_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "data": self.data,
            "source": self.source,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "intent": self.intent.value,
            "cache_key": self.cache_key,
            "metadata": self.metadata,
        }


class QueryRouter:
    """
    Routes queries to the appropriate tier based on intent detection.

    The router implements a tiered approach:
    1. Cache (fastest) - For repeated queries
    2. Precomputed (fast) - For TEST_IMPACT and COVERAGE_GAPS
    3. Vector Search (medium) - For semantic similarity queries
    4. LLM (slowest) - For complex reasoning when confidence < 0.7

    Example:
        ```python
        router = QueryRouter(org_id="org123", project_id="proj456")

        result = await router.route(
            query="Find similar errors to 'Element not found: #submit-btn'",
            org_id="org123",
            project_id="proj456"
        )

        print(f"Source: {result.source}, Confidence: {result.confidence}")
        print(f"Data: {result.data}")
        ```
    """

    # Cache TTLs by intent type (seconds)
    CACHE_TTLS: dict[QueryIntent, int] = {
        QueryIntent.SIMILAR_ERRORS: 300,       # 5 minutes
        QueryIntent.TEST_IMPACT: 600,          # 10 minutes (changes less often)
        QueryIntent.ROOT_CAUSE: 900,           # 15 minutes (analysis is expensive)
        QueryIntent.DOCUMENTATION: 3600,       # 1 hour (docs rarely change)
        QueryIntent.CODE_CONTEXT: 300,         # 5 minutes
        QueryIntent.COVERAGE_GAPS: 600,        # 10 minutes
        QueryIntent.SECURITY_IMPACT: 900,      # 15 minutes
    }

    # Minimum confidence threshold for accepting non-LLM results
    CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        org_id: str | None = None,
        project_id: str | None = None,
        cognee_client: CogneeKnowledgeClient | None = None,
        intelligence_cache: IntelligenceCache | None = None,
        precomputed_reader: PrecomputedReader | None = None,
        confidence_threshold: float | None = None,
    ):
        """
        Initialize the query router.

        Args:
            org_id: Organization ID for multi-tenant isolation
            project_id: Project ID for multi-tenant isolation
            cognee_client: Optional pre-configured Cognee client
            intelligence_cache: Optional pre-configured IntelligenceCache
            precomputed_reader: Optional pre-configured PrecomputedReader
            confidence_threshold: Optional custom confidence threshold
        """
        self.org_id = org_id or "default"
        self.project_id = project_id or "default"
        self._cognee_client = cognee_client
        self._intelligence_cache = intelligence_cache
        self._precomputed_reader = precomputed_reader
        self._log = logger.bind(
            component="query_router",
            org_id=self.org_id,
            project_id=self.project_id,
        )
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.CONFIDENCE_THRESHOLD
        )

    @property
    def cognee_client(self) -> CogneeKnowledgeClient:
        """Get or create the Cognee client."""
        if self._cognee_client is None:
            self._cognee_client = get_cognee_client(
                org_id=self.org_id,
                project_id=self.project_id,
            )
        return self._cognee_client

    @property
    def intelligence_cache(self) -> IntelligenceCache:
        """Get or create the IntelligenceCache instance."""
        if self._intelligence_cache is None:
            self._intelligence_cache = get_intelligence_cache()
        return self._intelligence_cache

    @property
    def precomputed_reader(self) -> PrecomputedReader:
        """Get or create the PrecomputedReader instance."""
        if self._precomputed_reader is None:
            self._precomputed_reader = get_precomputed_reader()
        return self._precomputed_reader

    def detect_intent(self, query: str) -> QueryIntent:
        """
        Detect the intent of a query using keyword matching.

        Uses a scoring approach where each intent gets points based on
        matching keywords. The intent with the highest score wins.

        Args:
            query: The user's query string

        Returns:
            Detected QueryIntent enum value
        """
        query_lower = query.lower()
        scores: dict[QueryIntent, int] = {intent: 0 for intent in QueryIntent}

        for intent, keywords in INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    # Longer keywords are more specific, give more weight
                    scores[intent] += len(keyword.split())

        # Find the intent with highest score
        best_intent = max(scores, key=lambda k: scores[k])

        # If no keywords matched, default based on query characteristics
        if scores[best_intent] == 0:
            # Default heuristics
            if "?" in query:
                best_intent = QueryIntent.ROOT_CAUSE
            elif "error" in query_lower or "fail" in query_lower:
                best_intent = QueryIntent.SIMILAR_ERRORS
            else:
                best_intent = QueryIntent.CODE_CONTEXT

        self._log.debug(
            "Detected intent",
            query=query[:50] if len(query) > 50 else query,
            intent=best_intent.value,
            score=scores[best_intent],
        )

        return best_intent

    def _get_intent_string(self, intent: QueryIntent) -> str:
        """
        Map QueryIntent to cache intent string.

        The IntelligenceCache uses string intents for TTL lookup.

        Args:
            intent: QueryIntent enum value

        Returns:
            Intent string for cache operations
        """
        intent_map = {
            QueryIntent.SIMILAR_ERRORS: "self_healing",
            QueryIntent.TEST_IMPACT: "realtime",  # Short TTL for impact data
            QueryIntent.ROOT_CAUSE: "failure_analysis",
            QueryIntent.DOCUMENTATION: "documentation",
            QueryIntent.CODE_CONTEXT: "code_analysis",
            QueryIntent.COVERAGE_GAPS: "realtime",
            QueryIntent.SECURITY_IMPACT: "failure_analysis",
        }
        return intent_map.get(intent, "self_healing")

    async def _check_cache(
        self,
        query: str,
        intent: QueryIntent,
    ) -> tuple[Any, bool, float]:
        """
        Check if a query result is cached using IntelligenceCache.

        Args:
            query: The query string
            intent: Detected intent

        Returns:
            Tuple of (cached_data, hit, latency_ms) where hit is True if found
        """
        intent_str = self._get_intent_string(intent)

        try:
            result = await self.intelligence_cache.get(query, intent_str)
            if result is not None:
                self._log.debug(
                    "Cache hit",
                    intent=intent.value,
                    source=result.source,
                    latency_ms=result.latency_ms,
                )
                return result.data, True, result.latency_ms
        except Exception as e:
            self._log.warning("Cache check failed", error=str(e))

        return None, False, 0.0

    async def _set_cache(
        self,
        query: str,
        data: Any,
        intent: QueryIntent,
    ) -> None:
        """
        Cache a query result using IntelligenceCache.

        Args:
            query: The query string
            data: Data to cache
            intent: Query intent (determines TTL via intent mapping)
        """
        intent_str = self._get_intent_string(intent)
        ttl = self.CACHE_TTLS.get(intent, 300)

        try:
            success = await self.intelligence_cache.set(query, intent_str, data, ttl=ttl)
            if success:
                self._log.debug("Cached result", intent=intent.value, ttl=ttl)
            else:
                self._log.warning("Cache set returned False")
        except Exception as e:
            self._log.warning("Cache set failed", error=str(e))

    async def _check_precomputed(
        self,
        query: str,
        intent: QueryIntent,
        org_id: str,
        project_id: str,
    ) -> tuple[Any, float] | None:
        """
        Check for precomputed results for certain intents using PrecomputedReader.

        TEST_IMPACT and COVERAGE_GAPS often have precomputed data
        stored in the database from background jobs.

        Args:
            query: The query string (unused but kept for API consistency)
            intent: Detected intent
            org_id: Organization ID
            project_id: Project ID

        Returns:
            Tuple of (data, confidence) if found, None otherwise
        """
        if intent not in (QueryIntent.TEST_IMPACT, QueryIntent.COVERAGE_GAPS):
            return None

        try:
            if intent == QueryIntent.TEST_IMPACT:
                # Check for precomputed test impact matrix
                data = await self.precomputed_reader.get_test_impact_matrix(
                    org_id=org_id,
                    project_id=project_id,
                )
                if data:
                    self._log.debug(
                        "Found precomputed test impact data",
                        org_id=org_id,
                        project_id=project_id,
                        file_count=len(data) if isinstance(data, dict) else 0,
                    )
                    # High confidence for precomputed data
                    # Slightly lower if data might be stale
                    is_stale = await self.precomputed_reader.is_stale(
                        org_id, project_id, "test_impact_matrix"
                    )
                    confidence = 0.85 if is_stale else 0.95
                    return data, confidence

            elif intent == QueryIntent.COVERAGE_GAPS:
                # Check for precomputed coverage gaps
                data = await self.precomputed_reader.get_coverage_gaps(
                    org_id=org_id,
                    project_id=project_id,
                )
                if data:
                    self._log.debug(
                        "Found precomputed coverage gap data",
                        org_id=org_id,
                        project_id=project_id,
                        gap_count=len(data) if isinstance(data, list) else 0,
                    )
                    is_stale = await self.precomputed_reader.is_stale(
                        org_id, project_id, "coverage_gaps"
                    )
                    confidence = 0.85 if is_stale else 0.95
                    return data, confidence

        except Exception as e:
            self._log.warning(
                "Precomputed check failed",
                intent=intent.value,
                error=str(e),
                error_type=type(e).__name__,
            )

        return None

    async def _vector_search(
        self,
        query: str,
        intent: QueryIntent,
        limit: int = 5,
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Perform vector search using Cognee.

        Args:
            query: The query string
            intent: Detected intent (determines namespace)
            limit: Maximum results

        Returns:
            Tuple of (results, average_confidence)
        """
        # Map intent to search namespace
        namespace_map = {
            QueryIntent.SIMILAR_ERRORS: ["failure_patterns"],
            QueryIntent.ROOT_CAUSE: ["failure_patterns"],
            QueryIntent.DOCUMENTATION: ["documentation"],
            QueryIntent.CODE_CONTEXT: ["codebase"],
            QueryIntent.SECURITY_IMPACT: ["security"],
            QueryIntent.TEST_IMPACT: ["test_impact"],
            QueryIntent.COVERAGE_GAPS: ["coverage"],
        }

        namespace = namespace_map.get(intent, ["general"])

        try:
            results = await self.cognee_client.search(
                namespace=namespace,
                query=query,
                limit=limit,
                threshold=0.5,  # Lower threshold, we'll filter by confidence
            )

            if not results:
                return [], 0.0

            # Calculate confidence from similarity scores
            # Cognee returns results ordered by relevance
            confidences = [r.get("similarity", 0.5) for r in results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            self._log.debug(
                "Vector search completed",
                intent=intent.value,
                num_results=len(results),
                avg_confidence=avg_confidence,
            )

            return results, avg_confidence

        except CogneeSearchError as e:
            self._log.error("Vector search failed", error=str(e))
            return [], 0.0
        except Exception as e:
            self._log.error("Vector search unexpected error", error=str(e))
            return [], 0.0

    async def _llm_fallback(
        self,
        query: str,
        intent: QueryIntent,
        context: list[dict[str, Any]] | None = None,
    ) -> tuple[Any, float]:
        """
        Fall back to LLM for complex reasoning when vector search confidence is low.

        Args:
            query: The query string
            intent: Detected intent
            context: Optional context from vector search

        Returns:
            Tuple of (result, confidence)
        """
        try:
            from anthropic import Anthropic
            from src.config import get_settings
            from src.core.model_registry import get_model_id

            settings = get_settings()
            api_key = settings.anthropic_api_key
            if api_key is None:
                self._log.warning("Anthropic API key not configured for LLM fallback")
                return {"error": "LLM not configured"}, 0.0

            if hasattr(api_key, 'get_secret_value'):
                api_key = api_key.get_secret_value()

            client = Anthropic(api_key=api_key)

            # Build prompt based on intent
            context_str = ""
            if context:
                context_str = "\n\nRelevant context:\n" + "\n".join(
                    f"- {c.get('content', str(c))[:200]}" for c in context[:5]
                )

            intent_prompts = {
                QueryIntent.ROOT_CAUSE: f"Analyze this error and provide the root cause:\n\n{query}{context_str}",
                QueryIntent.SIMILAR_ERRORS: f"Find patterns in this error and suggest similar issues:\n\n{query}{context_str}",
                QueryIntent.SECURITY_IMPACT: f"Analyze the security implications of this issue:\n\n{query}{context_str}",
                QueryIntent.DOCUMENTATION: f"Provide guidance on:\n\n{query}{context_str}",
                QueryIntent.CODE_CONTEXT: f"Explain this code context:\n\n{query}{context_str}",
                QueryIntent.TEST_IMPACT: f"Analyze test impact for:\n\n{query}{context_str}",
                QueryIntent.COVERAGE_GAPS: f"Identify coverage gaps for:\n\n{query}{context_str}",
            }

            prompt = intent_prompts.get(
                intent,
                f"Answer this query:\n\n{query}{context_str}"
            )

            response = client.messages.create(
                model=get_model_id("claude-sonnet-4-5"),
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            result_text = response.content[0].text if response.content else ""

            # LLM responses get high confidence since they're reasoning-based
            confidence = 0.85

            self._log.info(
                "LLM fallback completed",
                intent=intent.value,
                response_length=len(result_text),
            )

            return {
                "analysis": result_text,
                "intent": intent.value,
                "context_used": len(context) if context else 0,
            }, confidence

        except Exception as e:
            self._log.error("LLM fallback failed", error=str(e))
            return {"error": str(e)}, 0.0

    async def route(
        self,
        query: str,
        org_id: str | None = None,
        project_id: str | None = None,
        skip_cache: bool = False,
        force_llm: bool = False,
    ) -> QueryResult:
        """
        Route a query through the tiered system.

        Routing logic:
        1. Detect intent from query
        2. Check cache (unless skip_cache=True)
        3. Check precomputed data for TEST_IMPACT/COVERAGE_GAPS
        4. Perform vector search via Cognee
        5. Fall back to LLM if confidence < 0.7 (or force_llm=True)

        Args:
            query: The query string to process
            org_id: Organization ID (overrides instance default)
            project_id: Project ID (overrides instance default)
            skip_cache: Skip cache lookup
            force_llm: Force LLM processing regardless of confidence

        Returns:
            QueryResult with data, source, confidence, and metadata
        """
        start_time = time.time()

        # Use provided IDs or fall back to instance defaults
        org = org_id or self.org_id
        proj = project_id or self.project_id

        # Step 1: Detect intent
        intent = self.detect_intent(query)
        intent_str = self._get_intent_string(intent)

        # Step 2: Check cache
        if not skip_cache:
            cached_data, cache_hit, cache_latency = await self._check_cache(query, intent)
            if cache_hit:
                latency_ms = int((time.time() - start_time) * 1000)
                # Record metrics
                INTELLIGENCE_CACHE_HITS.labels(intent=intent_str).inc()
                INTELLIGENCE_QUERIES_TOTAL.labels(intent=intent_str, tier="cache").inc()
                INTELLIGENCE_QUERY_DURATION.labels(intent=intent_str, tier="cache").observe(latency_ms / 1000)
                INTELLIGENCE_QUERY_CONFIDENCE.labels(intent=intent_str).set(1.0)
                return QueryResult(
                    data=cached_data,
                    source="cache",
                    confidence=1.0,  # Cached results are trusted
                    latency_ms=latency_ms,
                    intent=intent,
                    cache_key=f"{intent_str}:{query[:32]}",
                    metadata={"from_cache": True, "cache_latency_ms": cache_latency},
                )
            else:
                INTELLIGENCE_CACHE_MISSES.labels(intent=intent_str).inc()

        # Step 3: Check precomputed for specific intents
        precomputed = await self._check_precomputed(query, intent, org, proj)
        if precomputed is not None:
            data, confidence = precomputed
            latency_ms = int((time.time() - start_time) * 1000)

            # Cache the precomputed result for faster future access
            await self._set_cache(query, data, intent)

            # Record metrics
            INTELLIGENCE_QUERIES_TOTAL.labels(intent=intent_str, tier="precomputed").inc()
            INTELLIGENCE_QUERY_DURATION.labels(intent=intent_str, tier="precomputed").observe(latency_ms / 1000)
            INTELLIGENCE_QUERY_CONFIDENCE.labels(intent=intent_str).set(confidence)

            return QueryResult(
                data=data,
                source="precomputed",
                confidence=confidence,
                latency_ms=latency_ms,
                intent=intent,
                cache_key=f"{intent_str}:{query[:32]}",
                metadata={"precomputed": True},
            )

        # Step 4: Vector search
        vector_results, vector_confidence = await self._vector_search(
            query=query,
            intent=intent,
            limit=5,
        )

        # Step 5: Decide whether to use vector results or fall back to LLM
        if not force_llm and vector_confidence >= self.confidence_threshold:
            # Vector results are good enough
            latency_ms = int((time.time() - start_time) * 1000)

            result_data = {
                "results": vector_results,
                "intent": intent.value,
                "query": query,
            }

            # Cache the result
            await self._set_cache(query, result_data, intent)

            # Record metrics
            INTELLIGENCE_QUERIES_TOTAL.labels(intent=intent_str, tier="vector").inc()
            INTELLIGENCE_QUERY_DURATION.labels(intent=intent_str, tier="vector").observe(latency_ms / 1000)
            INTELLIGENCE_QUERY_CONFIDENCE.labels(intent=intent_str).set(vector_confidence)

            return QueryResult(
                data=result_data,
                source="vector",
                confidence=vector_confidence,
                latency_ms=latency_ms,
                intent=intent,
                cache_key=f"{intent_str}:{query[:32]}",
                metadata={
                    "num_results": len(vector_results),
                    "vector_confidence": vector_confidence,
                },
            )

        # Step 6: Fall back to LLM
        INTELLIGENCE_LLM_FALLBACK.labels(intent=intent_str).inc()

        llm_result, llm_confidence = await self._llm_fallback(
            query=query,
            intent=intent,
            context=vector_results if vector_results else None,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Cache the LLM result
        await self._set_cache(query, llm_result, intent)

        # Record metrics
        INTELLIGENCE_QUERIES_TOTAL.labels(intent=intent_str, tier="llm").inc()
        INTELLIGENCE_QUERY_DURATION.labels(intent=intent_str, tier="llm").observe(latency_ms / 1000)
        INTELLIGENCE_QUERY_CONFIDENCE.labels(intent=intent_str).set(llm_confidence)

        return QueryResult(
            data=llm_result,
            source="llm",
            confidence=llm_confidence,
            latency_ms=latency_ms,
            intent=intent,
            cache_key=f"{intent_str}:{query[:32]}",
            metadata={
                "vector_confidence": vector_confidence,
                "vector_results_count": len(vector_results),
                "used_context": len(vector_results) > 0,
            },
        )


# =============================================================================
# Global Instance Management
# =============================================================================

_query_router: QueryRouter | None = None


def get_query_router(
    org_id: str | None = None,
    project_id: str | None = None,
) -> QueryRouter:
    """
    Get or create the global query router instance.

    Args:
        org_id: Organization ID (only used on first call)
        project_id: Project ID (only used on first call)

    Returns:
        QueryRouter instance
    """
    global _query_router

    if _query_router is None:
        _query_router = QueryRouter(
            org_id=org_id,
            project_id=project_id,
        )

    return _query_router


def reset_query_router() -> None:
    """Reset the global query router instance."""
    global _query_router
    _query_router = None


async def route_query(
    query: str,
    org_id: str | None = None,
    project_id: str | None = None,
    skip_cache: bool = False,
    force_llm: bool = False,
) -> QueryResult:
    """
    Convenience function to route a query using the global router.

    Args:
        query: The query string
        org_id: Organization ID
        project_id: Project ID
        skip_cache: Skip cache lookup
        force_llm: Force LLM processing

    Returns:
        QueryResult
    """
    router = get_query_router(org_id=org_id, project_id=project_id)
    return await router.route(
        query=query,
        org_id=org_id,
        project_id=project_id,
        skip_cache=skip_cache,
        force_llm=force_llm,
    )
