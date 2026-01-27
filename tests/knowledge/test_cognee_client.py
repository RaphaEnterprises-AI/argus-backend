"""Tests for CogneeKnowledgeClient.

This tests the unified knowledge layer that replaces:
- src/orchestrator/memory_store.py (MemoryStore)
- src/knowledge_graph/graph_store.py (GraphStore)

See RAP-132 for migration details.
"""

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_cognee():
    """Mock the cognee module."""
    with patch.dict("sys.modules", {"cognee": MagicMock()}):
        yield


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
    async def test_put_stores_value(self, mock_cognee, mock_env_vars):
        """Test put stores value correctly."""
        from src.knowledge.cognee_client import CogneeKnowledgeClient

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        # Mock the internal storage
        client._store = {}

        async def mock_put(ns, key, value, embed_text=None):
            client._store[f"{ns}:{key}"] = value

        with patch.object(client, "_cognee_put", mock_put):
            await client.put(
                namespace=["patterns"],
                key="test-key",
                value={"data": "test"},
            )

            assert "org1:proj1:patterns:test-key" in client._store

    @pytest.mark.asyncio
    async def test_get_retrieves_value(self, mock_cognee, mock_env_vars):
        """Test get retrieves stored value."""
        from src.knowledge.cognee_client import CogneeKnowledgeClient

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        expected_value = {"data": "test"}

        async def mock_get(ns, key):
            return expected_value

        with patch.object(client, "_cognee_get", mock_get):
            result = await client.get(namespace=["patterns"], key="test-key")

            assert result == expected_value


class TestCogneeKnowledgeClientFailurePatterns:
    """Tests for failure pattern operations."""

    @pytest.mark.asyncio
    async def test_store_failure_pattern(self, mock_cognee, mock_env_vars):
        """Test store_failure_pattern creates pattern correctly."""
        from src.knowledge.cognee_client import CogneeKnowledgeClient

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        stored_patterns = []

        async def mock_store(ns, key, value, embed_text=None):
            stored_patterns.append({"ns": ns, "key": key, "value": value})
            return key

        with patch.object(client, "_cognee_put", mock_store):
            pattern_id = await client.store_failure_pattern(
                error_message="Element not found: #submit",
                healed_selector="#submit-button",
                healing_method="semantic_match",
                error_type="selector_changed",
            )

            assert pattern_id is not None
            assert len(stored_patterns) == 1
            assert stored_patterns[0]["value"]["healed_selector"] == "#submit-button"

    @pytest.mark.asyncio
    async def test_find_similar_failures_returns_dataclass(
        self, mock_cognee, mock_env_vars
    ):
        """Test find_similar_failures returns SimilarFailure objects."""
        from src.knowledge.cognee_client import CogneeKnowledgeClient, SimilarFailure

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

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
                "metadata": {},
                "similarity": 0.92,
            }
        ]

        async def mock_search(ns, query, limit):
            return mock_results

        with patch.object(client, "_cognee_search", mock_search):
            results = await client.find_similar_failures(
                error_message="Element not found: #button"
            )

            assert len(results) == 1
            assert isinstance(results[0], SimilarFailure)
            # Key assertion: attribute access works
            assert results[0].similarity == 0.92
            assert results[0].healed_selector == "#new"

    @pytest.mark.asyncio
    async def test_record_healing_outcome_success(self, mock_cognee, mock_env_vars):
        """Test record_healing_outcome updates pattern correctly."""
        from src.knowledge.cognee_client import CogneeKnowledgeClient

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        updated_patterns = []

        async def mock_update(pattern_id, success):
            updated_patterns.append({"id": pattern_id, "success": success})

        with patch.object(client, "_cognee_update_outcome", mock_update):
            await client.record_healing_outcome("pattern-123", success=True)

            assert len(updated_patterns) == 1
            assert updated_patterns[0]["success"] is True


class TestCogneeKnowledgeClientKnowledgeGraph:
    """Tests for knowledge graph operations."""

    @pytest.mark.asyncio
    async def test_add_to_knowledge_graph(self, mock_cognee, mock_env_vars):
        """Test add_to_knowledge_graph adds content correctly."""
        from src.knowledge.cognee_client import CogneeKnowledgeClient

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        added_content = []

        async def mock_add(content, content_type, metadata):
            added_content.append(
                {"content": content, "type": content_type, "metadata": metadata}
            )

        with patch.object(client, "_cognee_add", mock_add):
            await client.add_to_knowledge_graph(
                content="Test failed because button moved",
                content_type="failure_analysis",
                metadata={"test_id": "t1"},
            )

            assert len(added_content) == 1
            assert added_content[0]["type"] == "failure_analysis"

    @pytest.mark.asyncio
    async def test_query_knowledge_graph(self, mock_cognee, mock_env_vars):
        """Test query_knowledge_graph returns results."""
        from src.knowledge.cognee_client import CogneeKnowledgeClient

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        mock_results = [
            {"content": "Login button is used in 5 tests", "score": 0.95}
        ]

        async def mock_query(query):
            return mock_results

        with patch.object(client, "_cognee_query", mock_query):
            results = await client.query_knowledge_graph(
                query="What tests use the login button?"
            )

            assert len(results) == 1
            assert results[0]["score"] == 0.95


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
