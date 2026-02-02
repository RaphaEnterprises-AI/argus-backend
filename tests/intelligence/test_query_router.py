"""Tests for QueryRouter - Intent Detection and Tiered Query Routing.

Tests the query routing system that optimizes for:
1. Latency - Check cache first, then precomputed, then vector, finally LLM
2. Cost - Avoid expensive LLM calls when simpler methods suffice
3. Accuracy - Use confidence scoring to determine when to escalate

RAP-238: Query router with intent detection tests
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_cognee_client():
    """Mock the CogneeKnowledgeClient."""
    mock = MagicMock()
    mock.search = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_intelligence_cache():
    """Mock the IntelligenceCache."""
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_precomputed_reader():
    """Mock the PrecomputedReader."""
    mock = MagicMock()
    mock.get_test_impact_matrix = AsyncMock(return_value=None)
    mock.get_coverage_gaps = AsyncMock(return_value=None)
    mock.is_stale = AsyncMock(return_value=True)
    return mock


class TestQueryIntent:
    """Tests for QueryIntent enum."""

    def test_query_intent_values(self):
        """Test QueryIntent has expected values."""
        from src.intelligence.query_router import QueryIntent

        assert QueryIntent.SIMILAR_ERRORS.value == "similar_errors"
        assert QueryIntent.TEST_IMPACT.value == "test_impact"
        assert QueryIntent.ROOT_CAUSE.value == "root_cause"
        assert QueryIntent.DOCUMENTATION.value == "documentation"
        assert QueryIntent.CODE_CONTEXT.value == "code_context"
        assert QueryIntent.COVERAGE_GAPS.value == "coverage_gaps"
        assert QueryIntent.SECURITY_IMPACT.value == "security_impact"

    def test_query_intent_count(self):
        """Test QueryIntent has 7 values."""
        from src.intelligence.query_router import QueryIntent

        assert len(QueryIntent) == 7


class TestIntentKeywords:
    """Tests for INTENT_KEYWORDS configuration."""

    def test_intent_keywords_has_all_intents(self):
        """Test INTENT_KEYWORDS covers all QueryIntent values."""
        from src.intelligence.query_router import INTENT_KEYWORDS, QueryIntent

        for intent in QueryIntent:
            assert intent in INTENT_KEYWORDS
            assert len(INTENT_KEYWORDS[intent]) > 0

    def test_intent_keywords_similar_errors(self):
        """Test SIMILAR_ERRORS keywords include expected terms."""
        from src.intelligence.query_router import INTENT_KEYWORDS, QueryIntent

        keywords = INTENT_KEYWORDS[QueryIntent.SIMILAR_ERRORS]

        assert "similar error" in keywords
        assert "error pattern" in keywords
        assert "element not found" in keywords

    def test_intent_keywords_test_impact(self):
        """Test TEST_IMPACT keywords include expected terms."""
        from src.intelligence.query_router import INTENT_KEYWORDS, QueryIntent

        keywords = INTENT_KEYWORDS[QueryIntent.TEST_IMPACT]

        assert "test impact" in keywords
        assert "affected tests" in keywords
        assert "blast radius" in keywords


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result_creation(self):
        """Test QueryResult can be created with required fields."""
        from src.intelligence.query_router import QueryIntent, QueryResult

        result = QueryResult(
            data={"results": []},
            source="cache",
            confidence=0.95,
            latency_ms=5,
            intent=QueryIntent.SIMILAR_ERRORS,
        )

        assert result.data == {"results": []}
        assert result.source == "cache"
        assert result.confidence == 0.95

    def test_query_result_to_dict(self):
        """Test QueryResult to_dict includes intent value."""
        from src.intelligence.query_router import QueryIntent, QueryResult

        result = QueryResult(
            data={"results": []},
            source="vector",
            confidence=0.8,
            latency_ms=25,
            intent=QueryIntent.ROOT_CAUSE,
            cache_key="test_key",
            metadata={"num_results": 5},
        )

        d = result.to_dict()

        assert d["intent"] == "root_cause"  # String value
        assert d["source"] == "vector"
        assert d["cache_key"] == "test_key"
        assert d["metadata"]["num_results"] == 5


class TestQueryRouterIntentDetection:
    """Tests for QueryRouter.detect_intent."""

    def test_detect_intent_similar_errors(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test detect_intent identifies similar_errors queries."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        queries = [
            "Find similar errors to 'Element not found'",
            "Have we seen this error pattern before?",
            "Show me past failures like this timeout",
        ]

        for query in queries:
            intent = router.detect_intent(query)
            assert intent == QueryIntent.SIMILAR_ERRORS, f"Failed for: {query}"

    def test_detect_intent_test_impact(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test detect_intent identifies test_impact queries."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        queries = [
            "What is the test impact of this change?",
            "Which tests are affected by modifying auth.py?",
            "Show me the blast radius for this commit",
        ]

        for query in queries:
            intent = router.detect_intent(query)
            assert intent == QueryIntent.TEST_IMPACT, f"Failed for: {query}"

    def test_detect_intent_root_cause(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test detect_intent identifies root_cause queries."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        queries = [
            "What is the root cause of this failure?",
            "Why did the login test fail?",
            "Can you diagnose this issue?",
        ]

        for query in queries:
            intent = router.detect_intent(query)
            assert intent == QueryIntent.ROOT_CAUSE, f"Failed for: {query}"

    def test_detect_intent_documentation(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test detect_intent identifies documentation queries."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        # Use queries that strongly match documentation keywords
        queries = [
            "Show me the documentation for this feature",
            "I need the API reference please",
            "Give me a tutorial on testing",
        ]

        for query in queries:
            intent = router.detect_intent(query)
            assert intent == QueryIntent.DOCUMENTATION, f"Failed for: {query}"

    def test_detect_intent_code_context(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test detect_intent identifies code_context queries."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        queries = [
            "Show me the source code for the login function",
            "Where is the UserService class implemented?",
            "Find code that handles authentication",
        ]

        for query in queries:
            intent = router.detect_intent(query)
            assert intent == QueryIntent.CODE_CONTEXT, f"Failed for: {query}"

    def test_detect_intent_coverage_gaps(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test detect_intent identifies coverage_gaps queries."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        # Use queries that strongly match coverage_gaps keywords
        queries = [
            "What are the coverage gaps in this module?",
            "Which code is not tested?",
            "Show me the test coverage report",
        ]

        for query in queries:
            intent = router.detect_intent(query)
            assert intent == QueryIntent.COVERAGE_GAPS, f"Failed for: {query}"

    def test_detect_intent_security_impact(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test detect_intent identifies security_impact queries."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        queries = [
            "What is the security impact of this vulnerability?",
            "Is there a CVE related to this?",
            "Check for XSS vulnerabilities",
        ]

        for query in queries:
            intent = router.detect_intent(query)
            assert intent == QueryIntent.SECURITY_IMPACT, f"Failed for: {query}"

    def test_detect_intent_default_with_question(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test detect_intent handles generic questions."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        # Query without specific keywords should use heuristics
        # "What is" matches documentation keyword, so check for valid intent
        intent = router.detect_intent("xyz happening xyz?")

        # With no keywords and a question mark, defaults to ROOT_CAUSE
        assert intent == QueryIntent.ROOT_CAUSE

    def test_detect_intent_default_with_error(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test detect_intent defaults to SIMILAR_ERRORS for error mentions."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        intent = router.detect_intent("login failed unexpectedly")

        assert intent == QueryIntent.SIMILAR_ERRORS

    def test_detect_intent_case_insensitive(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test detect_intent is case insensitive."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        intent1 = router.detect_intent("SHOW ME SIMILAR ERRORS")
        intent2 = router.detect_intent("show me similar errors")

        assert intent1 == intent2


class TestQueryRouterCacheTTL:
    """Tests for QueryRouter cache TTL configuration."""

    def test_cache_ttls_exist_for_all_intents(self):
        """Test CACHE_TTLS has entry for all intents."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        for intent in QueryIntent:
            assert intent in QueryRouter.CACHE_TTLS

    def test_cache_ttls_are_reasonable(self):
        """Test cache TTLs are within reasonable bounds."""
        from src.intelligence.query_router import QueryRouter

        for intent, ttl in QueryRouter.CACHE_TTLS.items():
            assert ttl >= 300  # At least 5 minutes
            assert ttl <= 3600  # At most 1 hour

    def test_documentation_has_longest_ttl(self):
        """Test documentation intent has longest TTL."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        doc_ttl = QueryRouter.CACHE_TTLS[QueryIntent.DOCUMENTATION]

        for intent, ttl in QueryRouter.CACHE_TTLS.items():
            assert doc_ttl >= ttl


class TestQueryRouterCacheOperations:
    """Tests for QueryRouter cache hit/miss logic."""

    @pytest.mark.asyncio
    async def test_check_cache_returns_hit_on_found(
        self, mock_cognee_client, mock_precomputed_reader
    ):
        """Test _check_cache returns hit when cached."""
        from src.intelligence.cache import CachedResult, CacheSource
        from src.intelligence.query_router import QueryIntent, QueryRouter

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(
            return_value=CachedResult(
                data={"results": []},
                source=CacheSource.VALKEY,
                latency_ms=2.5,
            )
        )

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        data, hit, latency = await router._check_cache(
            "test query", QueryIntent.SIMILAR_ERRORS
        )

        assert hit is True
        assert data == {"results": []}

    @pytest.mark.asyncio
    async def test_check_cache_returns_miss_on_not_found(
        self, mock_cognee_client, mock_precomputed_reader
    ):
        """Test _check_cache returns miss when not cached."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        data, hit, latency = await router._check_cache(
            "test query", QueryIntent.SIMILAR_ERRORS
        )

        assert hit is False
        assert data is None

    @pytest.mark.asyncio
    async def test_set_cache_calls_cache_set(
        self, mock_cognee_client, mock_precomputed_reader
    ):
        """Test _set_cache calls cache.set with correct parameters."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        mock_cache = MagicMock()
        mock_cache.set = AsyncMock(return_value=True)

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        await router._set_cache(
            "test query", {"results": []}, QueryIntent.SIMILAR_ERRORS
        )

        mock_cache.set.assert_called_once()


class TestQueryRouterPrecomputedOperations:
    """Tests for QueryRouter precomputed data access."""

    @pytest.mark.asyncio
    async def test_check_precomputed_returns_none_for_non_precomputed_intents(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test _check_precomputed returns None for non-precomputed intents."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        # ROOT_CAUSE is not precomputed
        result = await router._check_precomputed(
            "test query", QueryIntent.ROOT_CAUSE, "org1", "proj1"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_check_precomputed_returns_data_for_test_impact(
        self, mock_cognee_client, mock_intelligence_cache
    ):
        """Test _check_precomputed returns data for TEST_IMPACT."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        mock_reader = MagicMock()
        mock_reader.get_test_impact_matrix = AsyncMock(
            return_value={"src/auth.py": ["test_login", "test_logout"]}
        )
        mock_reader.is_stale = AsyncMock(return_value=False)

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_reader,
        )

        result = await router._check_precomputed(
            "test query", QueryIntent.TEST_IMPACT, "org1", "proj1"
        )

        assert result is not None
        data, confidence = result
        assert "src/auth.py" in data
        assert confidence >= 0.85

    @pytest.mark.asyncio
    async def test_check_precomputed_returns_data_for_coverage_gaps(
        self, mock_cognee_client, mock_intelligence_cache
    ):
        """Test _check_precomputed returns data for COVERAGE_GAPS."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        mock_reader = MagicMock()
        mock_reader.get_coverage_gaps = AsyncMock(
            return_value=[{"file": "src/api.py", "coverage": 45}]
        )
        mock_reader.is_stale = AsyncMock(return_value=False)

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_reader,
        )

        result = await router._check_precomputed(
            "test query", QueryIntent.COVERAGE_GAPS, "org1", "proj1"
        )

        assert result is not None
        data, confidence = result
        assert len(data) == 1
        assert confidence >= 0.85


class TestQueryRouterVectorSearch:
    """Tests for QueryRouter vector search operations."""

    @pytest.mark.asyncio
    async def test_vector_search_returns_results(
        self, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test _vector_search returns results with confidence."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        mock_cognee = MagicMock()
        mock_cognee.search = AsyncMock(
            return_value=[
                {"content": "Result 1", "similarity": 0.9},
                {"content": "Result 2", "similarity": 0.8},
            ]
        )

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        results, confidence = await router._vector_search(
            "test query", QueryIntent.SIMILAR_ERRORS
        )

        assert len(results) == 2
        assert confidence > 0

    @pytest.mark.asyncio
    async def test_vector_search_handles_empty_results(
        self, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test _vector_search handles empty results."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        mock_cognee = MagicMock()
        mock_cognee.search = AsyncMock(return_value=[])

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        results, confidence = await router._vector_search(
            "test query", QueryIntent.SIMILAR_ERRORS
        )

        assert len(results) == 0
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_vector_search_handles_errors(
        self, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test _vector_search handles Cognee errors gracefully."""
        from src.knowledge.cognee_client import CogneeSearchError
        from src.intelligence.query_router import QueryIntent, QueryRouter

        mock_cognee = MagicMock()
        mock_cognee.search = AsyncMock(side_effect=CogneeSearchError("Search failed"))

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        results, confidence = await router._vector_search(
            "test query", QueryIntent.SIMILAR_ERRORS
        )

        assert len(results) == 0
        assert confidence == 0.0


class TestQueryRouterRouteFunction:
    """Tests for QueryRouter.route main routing function."""

    @pytest.mark.asyncio
    async def test_route_returns_cache_hit(
        self, mock_cognee_client, mock_precomputed_reader
    ):
        """Test route returns cached result on cache hit."""
        from src.intelligence.cache import CachedResult, CacheSource
        from src.intelligence.query_router import QueryIntent, QueryRouter

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(
            return_value=CachedResult(
                data={"cached": True},
                source=CacheSource.VALKEY,
                latency_ms=2.0,
            )
        )

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        result = await router.route("Find similar errors")

        assert result.source == "cache"
        assert result.confidence == 1.0
        assert result.data == {"cached": True}

    @pytest.mark.asyncio
    async def test_route_skips_cache_when_requested(
        self, mock_cognee_client, mock_precomputed_reader
    ):
        """Test route skips cache when skip_cache=True."""
        from src.intelligence.cache import CachedResult, CacheSource
        from src.intelligence.query_router import QueryRouter

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(
            return_value=CachedResult(
                data={"cached": True},
                source=CacheSource.VALKEY,
                latency_ms=2.0,
            )
        )
        mock_cache.set = AsyncMock(return_value=True)

        mock_cognee_client.search = AsyncMock(return_value=[{"result": "fresh"}])

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        result = await router.route("Find similar errors", skip_cache=True)

        # Should not return cached result
        assert result.source != "cache"

    @pytest.mark.asyncio
    async def test_route_uses_precomputed_for_test_impact(
        self, mock_cognee_client, mock_intelligence_cache
    ):
        """Test route uses precomputed data for TEST_IMPACT."""
        mock_intelligence_cache.get = AsyncMock(return_value=None)
        mock_intelligence_cache.set = AsyncMock(return_value=True)

        mock_reader = MagicMock()
        mock_reader.get_test_impact_matrix = AsyncMock(
            return_value={"src/api.py": ["test1", "test2"]}
        )
        mock_reader.is_stale = AsyncMock(return_value=False)

        from src.intelligence.query_router import QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_reader,
        )

        result = await router.route("What is the test impact?")

        assert result.source == "precomputed"
        assert "src/api.py" in result.data

    @pytest.mark.asyncio
    async def test_route_falls_back_to_vector_search(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test route falls back to vector search when no cache/precomputed."""
        mock_intelligence_cache.get = AsyncMock(return_value=None)
        mock_intelligence_cache.set = AsyncMock(return_value=True)

        mock_cognee_client.search = AsyncMock(
            return_value=[
                {"content": "Result 1", "similarity": 0.85},
            ]
        )

        from src.intelligence.query_router import QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        result = await router.route("Find similar errors to element not found")

        assert result.source == "vector"

    @pytest.mark.asyncio
    async def test_route_includes_latency(
        self, mock_cognee_client, mock_precomputed_reader
    ):
        """Test route includes latency in result."""
        from src.intelligence.cache import CachedResult, CacheSource
        from src.intelligence.query_router import QueryRouter

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(
            return_value=CachedResult(
                data={"cached": True},
                source=CacheSource.VALKEY,
                latency_ms=2.0,
            )
        )

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        result = await router.route("test query")

        assert result.latency_ms >= 0


class TestQueryRouterConfidenceThreshold:
    """Tests for QueryRouter confidence threshold behavior."""

    def test_default_confidence_threshold(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test default confidence threshold is 0.7."""
        from src.intelligence.query_router import QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        assert router.confidence_threshold == 0.7

    def test_custom_confidence_threshold(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test custom confidence threshold can be set."""
        from src.intelligence.query_router import QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
            confidence_threshold=0.5,
        )

        assert router.confidence_threshold == 0.5


class TestGlobalInstanceManagement:
    """Tests for global QueryRouter instance management."""

    def test_get_query_router_creates_instance(self):
        """Test get_query_router creates new instance."""
        from src.intelligence.query_router import get_query_router, reset_query_router

        reset_query_router()

        router = get_query_router(org_id="org1", project_id="proj1")

        assert router is not None
        assert router.org_id == "org1"

    def test_get_query_router_reuses_instance(self):
        """Test get_query_router reuses existing instance."""
        from src.intelligence.query_router import get_query_router, reset_query_router

        reset_query_router()

        r1 = get_query_router(org_id="org1", project_id="proj1")
        r2 = get_query_router(org_id="org1", project_id="proj1")

        assert r1 is r2

    def test_reset_query_router(self):
        """Test reset_query_router clears global state."""
        from src.intelligence.query_router import get_query_router, reset_query_router

        r1 = get_query_router(org_id="org1", project_id="proj1")
        reset_query_router()
        r2 = get_query_router(org_id="org1", project_id="proj1")

        assert r1 is not r2


class TestRouteQueryConvenienceFunction:
    """Tests for route_query convenience function."""

    @pytest.mark.asyncio
    async def test_route_query_uses_global_router(self):
        """Test route_query function uses global router."""
        from unittest.mock import patch

        from src.intelligence.query_router import reset_query_router, route_query

        reset_query_router()

        # This will use the global router
        with patch(
            "src.intelligence.query_router.get_query_router"
        ) as mock_get_router:
            mock_router = MagicMock()
            mock_router.route = AsyncMock(
                return_value=MagicMock(data={"test": True})
            )
            mock_get_router.return_value = mock_router

            result = await route_query("test query")

            mock_router.route.assert_called_once()


class TestIntentStringMapping:
    """Tests for intent to cache string mapping."""

    def test_get_intent_string_maps_all_intents(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test _get_intent_string maps all QueryIntent values."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        for intent in QueryIntent:
            intent_str = router._get_intent_string(intent)
            assert isinstance(intent_str, str)
            assert len(intent_str) > 0

    def test_get_intent_string_maps_similar_errors(
        self, mock_cognee_client, mock_intelligence_cache, mock_precomputed_reader
    ):
        """Test SIMILAR_ERRORS maps to self_healing."""
        from src.intelligence.query_router import QueryIntent, QueryRouter

        router = QueryRouter(
            org_id="test",
            project_id="test",
            cognee_client=mock_cognee_client,
            intelligence_cache=mock_intelligence_cache,
            precomputed_reader=mock_precomputed_reader,
        )

        intent_str = router._get_intent_string(QueryIntent.SIMILAR_ERRORS)

        assert intent_str == "self_healing"
