"""Tests for CogneeKnowledgeClient.

This tests the unified knowledge layer that replaces:
- src/orchestrator/memory_store.py (MemoryStore)
- src/knowledge_graph/graph_store.py (GraphStore)

See RAP-132 for migration details.
"""

from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import exception types for testing
# Note: These will be imported via the mock system since we patch sys.modules


@pytest.fixture
def mock_cognee():
    """Mock the cognee module."""
    mock = MagicMock()
    mock.add = AsyncMock()
    mock.cognify = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    with patch.dict("sys.modules", {"cognee": mock}):
        yield mock


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up required environment variables."""
    monkeypatch.setenv("COGNEE_API_KEY", "test-api-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")


class TestCogneeKnowledgeClientInit:
    """Tests for CogneeKnowledgeClient initialization."""

    def test_init_with_org_and_project(self, mock_cognee, mock_env_vars):
        """Test initialization with org_id and project_id."""
        from src.knowledge.cognee_client import CogneeKnowledgeClient

        client = CogneeKnowledgeClient(org_id="org123", project_id="proj456")

        assert client.org_id == "org123"
        assert client.project_id == "proj456"
        assert client._namespace_prefix == "org123:proj456"

    def test_init_default_namespace(self, mock_cognee, mock_env_vars):
        """Test initialization creates correct namespace prefix."""
        from src.knowledge.cognee_client import CogneeKnowledgeClient

        client = CogneeKnowledgeClient(org_id="acme", project_id="webapp")

        assert client._namespace_prefix == "acme:webapp"


class TestCogneeKnowledgeClientGetClient:
    """Tests for get_cognee_client factory function."""

    def test_get_cognee_client_creates_instance(self, mock_cognee, mock_env_vars):
        """Test get_cognee_client creates new instance."""
        from src.knowledge.cognee_client import get_cognee_client, reset_cognee_client

        reset_cognee_client()

        client = get_cognee_client(org_id="org1", project_id="proj1")

        assert client is not None
        assert client.org_id == "org1"
        assert client.project_id == "proj1"

    def test_get_cognee_client_reuses_instance(self, mock_cognee, mock_env_vars):
        """Test get_cognee_client reuses existing instance for same org/project."""
        from src.knowledge.cognee_client import get_cognee_client, reset_cognee_client

        reset_cognee_client()

        client1 = get_cognee_client(org_id="org1", project_id="proj1")
        client2 = get_cognee_client(org_id="org1", project_id="proj1")

        assert client1 is client2

    def test_get_cognee_client_different_tenants(self, mock_cognee, mock_env_vars):
        """Test get_cognee_client creates different instances for different tenants."""
        from src.knowledge.cognee_client import get_cognee_client, reset_cognee_client

        reset_cognee_client()

        client1 = get_cognee_client(org_id="org1", project_id="proj1")
        client2 = get_cognee_client(org_id="org2", project_id="proj2")

        assert client1 is not client2
        assert client1.org_id == "org1"
        assert client2.org_id == "org2"

    def test_reset_cognee_client(self, mock_cognee, mock_env_vars):
        """Test reset_cognee_client clears all instances."""
        from src.knowledge.cognee_client import get_cognee_client, reset_cognee_client

        client1 = get_cognee_client(org_id="org1", project_id="proj1")
        reset_cognee_client()
        client2 = get_cognee_client(org_id="org1", project_id="proj1")

        assert client1 is not client2


class TestSimilarFailureDataclass:
    """Tests for SimilarFailure dataclass."""

    def test_similar_failure_creation(self, mock_cognee, mock_env_vars):
        """Test SimilarFailure dataclass can be created."""
        from src.knowledge.cognee_client import SimilarFailure

        failure = SimilarFailure(
            id="pattern-123",
            error_message="Element not found: #submit",
            error_type="selector_changed",
            original_selector="#submit",
            healed_selector="#submit-button",
            healing_method="semantic_match",
            success_count=5,
            failure_count=1,
            success_rate=0.833,
            similarity=0.92,
            metadata={"source": "test"},
        )

        assert failure.id == "pattern-123"
        assert failure.healed_selector == "#submit-button"
        assert failure.success_rate == 0.833
        assert failure.similarity == 0.92

    def test_similar_failure_attribute_access(self, mock_cognee, mock_env_vars):
        """Test SimilarFailure attributes are accessible (not dict-style)."""
        from src.knowledge.cognee_client import SimilarFailure

        failure = SimilarFailure(
            id="p1",
            error_message="Error",
            error_type="type1",
            original_selector="#old",
            healed_selector="#new",
            healing_method="method1",
            success_count=10,
            failure_count=2,
            success_rate=0.833,
            similarity=0.95,
            metadata={},
        )

        # This is the key test - attribute access instead of dict access
        assert failure.success_rate == 0.833
        assert failure.similarity == 0.95
        assert failure.healing_method == "method1"
        assert failure.id == "p1"

    def test_similar_failure_to_dict(self, mock_cognee, mock_env_vars):
        """Test SimilarFailure can be converted to dict."""
        from src.knowledge.cognee_client import SimilarFailure

        failure = SimilarFailure(
            id="p1",
            error_message="Error",
            error_type="type1",
            original_selector="#old",
            healed_selector="#new",
            healing_method="method1",
            success_count=10,
            failure_count=2,
            success_rate=0.833,
            similarity=0.95,
            metadata={"key": "value"},
        )

        d = asdict(failure)

        assert d["id"] == "p1"
        assert d["success_rate"] == 0.833
        assert d["metadata"] == {"key": "value"}


class TestCogneeKnowledgeClientNamespacing:
    """Tests for namespace functionality."""

    def test_build_namespace(self, mock_cognee, mock_env_vars):
        """Test _build_namespace creates correct namespaces."""
        from src.knowledge.cognee_client import CogneeKnowledgeClient

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        ns = client._build_namespace(["patterns", "selectors"])

        assert ns == "org1:proj1:patterns:selectors"

    def test_build_namespace_empty(self, mock_cognee, mock_env_vars):
        """Test _build_namespace with empty parts."""
        from src.knowledge.cognee_client import CogneeKnowledgeClient

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        ns = client._build_namespace([])

        assert ns == "org1:proj1"


class TestCogneeKnowledgeClientPutGet:
    """Tests for put/get operations."""

    @pytest.mark.asyncio
    async def test_put_calls_cognee_add(self, mock_env_vars):
        """Test put calls cognee.add correctly."""
        mock_cognee_module = MagicMock()
        mock_cognee_module.add = AsyncMock()
        mock_cognee_module.cognify = AsyncMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            # Force reimport to get the mocked version
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            await client.put(
                namespace=["patterns"],
                key="test-key",
                value={"data": "test"},
            )

            # Verify cognee.add was called
            mock_cognee_module.add.assert_called_once()
            mock_cognee_module.cognify.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_calls_cognee_search(self, mock_env_vars):
        """Test get calls cognee.search correctly."""
        mock_cognee_module = MagicMock()
        mock_cognee_module.search = AsyncMock(return_value=[{"data": "test"}])

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            result = await client.get(namespace=["patterns"], key="test-key")

            # Verify cognee.search was called
            mock_cognee_module.search.assert_called_once()


class TestCogneeKnowledgeClientFailurePatterns:
    """Tests for failure pattern operations."""

    @pytest.mark.asyncio
    async def test_store_failure_pattern_calls_cognee(self, mock_env_vars):
        """Test store_failure_pattern calls cognee.add."""
        mock_cognee_module = MagicMock()
        mock_cognee_module.add = AsyncMock()
        mock_cognee_module.cognify = AsyncMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            pattern_id = await client.store_failure_pattern(
                error_message="Element not found: #submit",
                healed_selector="#submit-button",
                healing_method="semantic_match",
                error_type="selector_changed",
            )

            assert pattern_id is not None
            mock_cognee_module.add.assert_called_once()
            mock_cognee_module.cognify.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_similar_failures_returns_dataclass(self, mock_env_vars):
        """Test find_similar_failures returns SimilarFailure objects."""
        mock_results = [
            {
                "id": "p1",
                "error_message": "Element not found",
                "error_type": "selector_changed",
                "original_selector": "#old",
                "healed_selector": "#new",
                "healing_method": "semantic",
                "success_count": 5,
                "failure_count": 1,
                "success_rate": 0.833,
                "metadata": {},
                "_similarity": 0.92,
            }
        ]

        mock_cognee_module = MagicMock()
        mock_cognee_module.search = AsyncMock(return_value=mock_results)

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient, SimilarFailure

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            results = await client.find_similar_failures(
                error_message="Element not found: #button"
            )

            # Verify cognee.search was called
            mock_cognee_module.search.assert_called_once()
            assert len(results) == 1
            assert isinstance(results[0], SimilarFailure)

    @pytest.mark.asyncio
    async def test_record_healing_outcome_success(self, mock_env_vars):
        """Test record_healing_outcome updates pattern correctly."""
        # This test verifies the method exists and can be called
        mock_cognee_module = MagicMock()
        mock_cognee_module.search = AsyncMock(return_value=[{"id": "p1", "success_count": 5}])
        mock_cognee_module.add = AsyncMock()
        mock_cognee_module.cognify = AsyncMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            # This should not raise
            await client.record_healing_outcome("pattern-123", success=True)


class TestCogneeKnowledgeClientKnowledgeGraph:
    """Tests for knowledge graph operations."""

    @pytest.mark.asyncio
    async def test_add_to_knowledge_graph(self, mock_env_vars):
        """Test add_to_knowledge_graph adds content correctly."""
        mock_cognee_module = MagicMock()
        mock_cognee_module.add = AsyncMock()
        mock_cognee_module.cognify = AsyncMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            await client.add_to_knowledge_graph(
                content="Test failed because button moved",
                content_type="failure_analysis",
            )

            mock_cognee_module.add.assert_called_once()
            mock_cognee_module.cognify.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_knowledge_graph(self, mock_env_vars):
        """Test query_knowledge_graph returns results."""
        mock_results = [{"content": "Login button is used in 5 tests", "score": 0.95}]

        mock_cognee_module = MagicMock()
        mock_cognee_module.search = AsyncMock(return_value=mock_results)

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            results = await client.query_knowledge_graph(
                query="What tests use the login button?"
            )

            # query_knowledge_graph searches multiple namespaces
            assert mock_cognee_module.search.call_count >= 1
            # Results are aggregated from all namespaces
            assert len(results) >= 1


class TestDeprecationWarnings:
    """Tests for deprecation warnings."""

    def test_memory_store_deprecation_warning(self, mock_env_vars):
        """Test MemoryStore import emits deprecation warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from src.orchestrator.memory_store import MemoryStore  # noqa: F401

            # Check that a deprecation warning was issued
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1

    def test_graph_store_deprecation_warning(self, mock_env_vars):
        """Test GraphStore import emits deprecation warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from src.knowledge_graph.graph_store import GraphStore  # noqa: F401

            # Check that a deprecation warning was issued
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1

    def test_orchestrator_get_memory_store_deprecation(self, mock_env_vars):
        """Test orchestrator.get_memory_store emits deprecation warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from src.orchestrator import get_memory_store

            get_memory_store()

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1


class TestCogneeExceptions:
    """Tests for Cognee exception types."""

    def test_cognee_error_is_base_class(self, mock_env_vars):
        """Test CogneeError is the base exception class."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import (
                CogneeError,
                CogneeGraphError,
                CogneeRetrievalError,
                CogneeSearchError,
                CogneeStorageError,
            )

            # All exceptions should inherit from CogneeError
            assert issubclass(CogneeStorageError, CogneeError)
            assert issubclass(CogneeRetrievalError, CogneeError)
            assert issubclass(CogneeSearchError, CogneeError)
            assert issubclass(CogneeGraphError, CogneeError)

            # CogneeError should inherit from Exception
            assert issubclass(CogneeError, Exception)

    @pytest.mark.asyncio
    async def test_put_raises_storage_error_on_failure(self, mock_env_vars):
        """Test put raises CogneeStorageError when cognee.add fails."""
        mock_cognee_module = MagicMock()
        mock_cognee_module.add = AsyncMock(side_effect=RuntimeError("Storage failed"))

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import (
                CogneeKnowledgeClient,
                CogneeStorageError,
            )

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            with pytest.raises(CogneeStorageError) as exc_info:
                await client.put(
                    namespace=["test"],
                    key="key1",
                    value={"data": "test"},
                )

            assert "Storage failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_raises_retrieval_error_on_failure(self, mock_env_vars):
        """Test get raises CogneeRetrievalError when cognee.search fails."""
        mock_cognee_module = MagicMock()
        mock_cognee_module.search = AsyncMock(side_effect=RuntimeError("Search failed"))

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import (
                CogneeKnowledgeClient,
                CogneeRetrievalError,
            )

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            with pytest.raises(CogneeRetrievalError) as exc_info:
                await client.get(namespace=["test"], key="key1")

            assert "Search failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_raises_search_error_on_failure(self, mock_env_vars):
        """Test search raises CogneeSearchError when cognee.search fails."""
        mock_cognee_module = MagicMock()
        mock_cognee_module.search = AsyncMock(side_effect=RuntimeError("Search failed"))

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import (
                CogneeKnowledgeClient,
                CogneeSearchError,
            )

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            with pytest.raises(CogneeSearchError) as exc_info:
                await client.search(
                    namespace=["test"],
                    query="test query",
                )

            assert "Search failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_add_to_knowledge_graph_raises_graph_error(self, mock_env_vars):
        """Test add_to_knowledge_graph raises CogneeGraphError on failure."""
        mock_cognee_module = MagicMock()
        mock_cognee_module.add = AsyncMock(side_effect=RuntimeError("Graph error"))

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import (
                CogneeGraphError,
                CogneeKnowledgeClient,
            )

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            with pytest.raises(CogneeGraphError) as exc_info:
                await client.add_to_knowledge_graph(
                    content="Test content",
                    content_type="test",
                )

            assert "Graph error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing_key(self, mock_env_vars):
        """Test get returns None for missing keys (not an error)."""
        mock_cognee_module = MagicMock()
        mock_cognee_module.search = AsyncMock(return_value=[])  # Empty results

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            # Should return None, not raise an error
            result = await client.get(namespace=["test"], key="missing-key")

            assert result is None


class TestClassifySelectorType:
    """Tests for classify_selector_type function (RAP-355)."""

    def test_classify_data_testid_attribute_selector(self, mock_env_vars):
        """Test classification of data-testid attribute selectors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import classify_selector_type

            assert classify_selector_type('[data-testid="login-button"]') == "data_testid"
            assert classify_selector_type("[data-testid='submit']") == "data_testid"
            assert classify_selector_type('[data-testid^="prefix-"]') == "data_testid"

    def test_classify_data_cy_selector(self, mock_env_vars):
        """Test classification of Cypress data-cy selectors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import classify_selector_type

            assert classify_selector_type('[data-cy="login"]') == "data_cy"

    def test_classify_id_selectors(self, mock_env_vars):
        """Test classification of ID-based selectors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import classify_selector_type

            assert classify_selector_type("#login-button") == "id_attribute"
            assert classify_selector_type("#submit_btn") == "id_attribute"
            assert classify_selector_type('[id="main-content"]') == "id_attribute"

    def test_classify_class_selectors(self, mock_env_vars):
        """Test classification of class-based selectors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import classify_selector_type

            assert classify_selector_type(".btn-primary") == "class_single"
            assert classify_selector_type(".btn.btn-primary") == "class_multiple"

    def test_classify_aria_selectors(self, mock_env_vars):
        """Test classification of ARIA-based selectors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import classify_selector_type

            assert classify_selector_type('[aria-label="Submit form"]') == "aria_label"
            assert classify_selector_type('[aria-labelledby="header"]') == "aria_labelledby"

    def test_classify_xpath_selectors(self, mock_env_vars):
        """Test classification of XPath selectors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import classify_selector_type

            assert classify_selector_type('//button[@id="submit"]') == "xpath_id"
            assert classify_selector_type('//button[text()="Submit"]') == "xpath_text"
            assert classify_selector_type('//div[@class="container"]') == "xpath_attribute"
            assert classify_selector_type("//div[3]") == "xpath_positional"

    def test_classify_role_selectors(self, mock_env_vars):
        """Test classification of role-based selectors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import classify_selector_type

            assert classify_selector_type('[role="button"]') == "role"

    def test_classify_playwright_text_selectors(self, mock_env_vars):
        """Test classification of Playwright text selectors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import classify_selector_type

            assert classify_selector_type('text="Submit"') == "text_exact"
            assert classify_selector_type("*text=Submit") == "text_partial"

    def test_classify_empty_and_none(self, mock_env_vars):
        """Test classification handles empty/None values."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import classify_selector_type

            assert classify_selector_type("") == "unknown"
            assert classify_selector_type(None) == "unknown"

    def test_classify_tag_only(self, mock_env_vars):
        """Test classification of tag-only selectors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import classify_selector_type

            assert classify_selector_type("button") == "tag_only"
            assert classify_selector_type("input") == "tag_only"


class TestStoreSuccessfulHeal:
    """Tests for store_successful_heal method (RAP-353)."""

    @pytest.mark.asyncio
    async def test_store_successful_heal_basic(self, mock_env_vars):
        """Test storing a successful heal pattern."""
        mock_cognee_module = MagicMock()
        mock_cognee_module.add = AsyncMock()
        mock_cognee_module.cognify = AsyncMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            pattern_id = await client.store_successful_heal(
                org_id="org1",
                test_id="test-123",
                failure_context={
                    "error_type": "selector_not_found",
                    "error_message": "Element not found: #login-btn",
                    "failed_selector": "#login-btn",
                    "page_url": "https://app.example.com/login",
                },
                fix_applied={
                    "new_selector": "[data-testid='login-button']",
                    "fix_type": "selector_update",
                    "confidence": 0.95,
                    "healing_method": "code_aware",
                },
            )

            assert pattern_id is not None
            assert len(pattern_id) == 32  # SHA256 truncated to 32 chars
            mock_cognee_module.add.assert_called_once()
            mock_cognee_module.cognify.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_successful_heal_anonymizes_selector(self, mock_env_vars):
        """Test that store_successful_heal anonymizes selectors."""
        mock_cognee_module = MagicMock()
        mock_cognee_module.add = AsyncMock()
        mock_cognee_module.cognify = AsyncMock()

        stored_data = {}

        async def capture_add(data, dataset_name):
            import json

            stored_data["content"] = json.loads(data)
            stored_data["dataset"] = dataset_name

        mock_cognee_module.add = AsyncMock(side_effect=capture_add)

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            await client.store_successful_heal(
                org_id="org1",
                test_id="test-123",
                failure_context={
                    "error_type": "selector_not_found",
                    "error_message": "Element not found: #login-btn",
                    "failed_selector": "#login-btn",
                    "page_url": "https://app.example.com/login",
                },
                fix_applied={
                    "new_selector": "[data-testid='login-button']",
                    "fix_type": "selector_update",
                    "confidence": 0.95,
                    "healing_method": "code_aware",
                },
            )

            # Verify the stored data has anonymized selectors
            content = stored_data.get("content", {})
            # Should NOT contain actual selector values
            assert "#login-btn" not in str(content)
            assert "login-button" not in str(content)
            # Should contain selector TYPES
            assert content.get("original_selector_type") == "id_attribute"
            assert content.get("new_selector_type") == "data_testid"


class TestSearchSimilarFixes:
    """Tests for search_similar_fixes method (RAP-354)."""

    @pytest.mark.asyncio
    async def test_search_similar_fixes_returns_results(self, mock_env_vars):
        """Test searching for similar fixes returns properly formatted results."""
        mock_results = [
            {
                "pattern_id": "abc123",
                "error_type": "selector_not_found",
                "error_category": "element_not_found",
                "original_selector_type": "id_attribute",
                "new_selector_type": "data_testid",
                "selector_transition": "id_attribute -> data_testid",
                "fix_type": "selector_update",
                "healing_method": "code_aware",
                "confidence": 0.95,
                "success_count": 10,
                "failure_count": 1,
                "total_applications": 11,
                "similarity": 0.92,
            }
        ]

        mock_cognee_module = MagicMock()
        mock_cognee_module.search = AsyncMock(return_value=mock_results)

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            fixes = await client.search_similar_fixes(
                org_id="org1",
                error_type="selector_not_found",
                selector_type="id_attribute",
                top_k=5,
            )

            assert len(fixes) == 1
            assert fixes[0]["pattern_id"] == "abc123"
            assert fixes[0]["selector_transition"] == "id_attribute -> data_testid"
            assert fixes[0]["success_rate"] > 0.9  # 10/11 = 0.909

    @pytest.mark.asyncio
    async def test_search_similar_fixes_filters_low_success_rate(self, mock_env_vars):
        """Test that search_similar_fixes filters patterns with low success rate."""
        mock_results = [
            {
                "pattern_id": "abc123",
                "error_type": "selector_not_found",
                "success_count": 2,
                "failure_count": 8,  # Only 20% success rate
                "similarity": 0.9,
            }
        ]

        mock_cognee_module = MagicMock()
        mock_cognee_module.search = AsyncMock(return_value=mock_results)

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            fixes = await client.search_similar_fixes(
                org_id="org1",
                error_type="selector_not_found",
                top_k=5,
            )

            # Should be filtered out due to low success rate (< 30%)
            assert len(fixes) == 0


class TestCategorizeErrorMessage:
    """Tests for _categorize_error_message private method."""

    def test_categorize_element_not_found(self, mock_env_vars):
        """Test categorization of element not found errors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            assert client._categorize_error_message("Element not found: #button") == "element_not_found"
            assert client._categorize_error_message("Unable to locate element") == "element_not_found"
            assert client._categorize_error_message("Cannot find selector") == "element_not_found"

    def test_categorize_timeout_errors(self, mock_env_vars):
        """Test categorization of timeout errors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            assert client._categorize_error_message("Navigation timeout exceeded") == "navigation_timeout"
            assert client._categorize_error_message("Wait timed out") == "wait_timeout"
            assert client._categorize_error_message("Request timeout") == "general_timeout"

    def test_categorize_visibility_errors(self, mock_env_vars):
        """Test categorization of visibility errors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            assert client._categorize_error_message("Element not visible") == "element_not_visible"
            assert client._categorize_error_message("Element is hidden") == "element_not_visible"

    def test_categorize_assertion_errors(self, mock_env_vars):
        """Test categorization of assertion errors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            assert client._categorize_error_message("Assertion failed: expected text") == "text_assertion_failed"
            assert client._categorize_error_message("Expected element to be visible") == "visibility_assertion_failed"
            assert client._categorize_error_message("Expected count to equal 5") == "count_assertion_failed"

    def test_categorize_unknown(self, mock_env_vars):
        """Test categorization returns 'other' for unknown errors."""
        mock_cognee_module = MagicMock()

        with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
            import importlib

            import src.knowledge.cognee_client

            importlib.reload(src.knowledge.cognee_client)
            from src.knowledge.cognee_client import CogneeKnowledgeClient

            client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

            assert client._categorize_error_message("Some random error") == "other"
            assert client._categorize_error_message("") == "unknown"
            assert client._categorize_error_message(None) == "unknown"
