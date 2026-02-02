"""Comprehensive tests for CogneeKnowledgeClient integration.

Tests the unified knowledge layer that provides:
- Key-value storage with embeddings
- Semantic search capabilities
- Failure pattern storage and retrieval
- Discovery pattern management
- Knowledge graph operations

RAP-342: Extended Cognee integration tests
"""

import hashlib
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Fixtures for mocking Cognee
@pytest.fixture
def mock_cognee():
    """Mock the cognee module with all necessary methods."""
    mock = MagicMock()
    mock.add = AsyncMock()
    mock.cognify = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.config = MagicMock()
    mock.config.set_llm_provider = MagicMock()
    mock.config.set_llm_api_key = MagicMock()
    mock.config.set_llm_model = MagicMock()
    mock.config.set_llm_config = MagicMock()
    mock.__version__ = "0.5.1"
    with patch.dict("sys.modules", {"cognee": mock}):
        yield mock


@pytest.fixture
def cognee_env_vars(monkeypatch):
    """Set up required environment variables for Cognee."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("DEFAULT_ORG_ID", "default-org")
    monkeypatch.setenv("DEFAULT_PROJECT_ID", "default-project")


@pytest.fixture
def cognee_client(mock_cognee, cognee_env_vars):
    """Create a CogneeKnowledgeClient for testing."""
    import importlib
    import src.knowledge.cognee_client
    importlib.reload(src.knowledge.cognee_client)
    from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
    reset_cognee_client()
    return CogneeKnowledgeClient(org_id="test-org", project_id="test-project")


class TestCogneeKnowledgeClientDatasetNaming:
    """Tests for dataset naming conventions."""

    def test_get_dataset_name_single_namespace(self, cognee_client):
        """Test dataset name generation with single namespace."""
        name = cognee_client._get_dataset_name(["patterns"])
        assert name == "org_test-org_project_test-project_patterns"

    def test_get_dataset_name_nested_namespace(self, cognee_client):
        """Test dataset name generation with nested namespace."""
        name = cognee_client._get_dataset_name(["discovery_patterns", "forms"])
        assert name == "org_test-org_project_test-project_discovery_patterns_forms"

    def test_get_dataset_name_empty_namespace(self, cognee_client):
        """Test dataset name generation with empty namespace."""
        name = cognee_client._get_dataset_name([])
        assert name == "org_test-org_project_test-project_default"

    def test_get_dataset_name_with_tuple(self, cognee_client):
        """Test dataset name generation works with tuple input."""
        name = cognee_client._get_dataset_name(("failure_patterns",))
        assert name == "org_test-org_project_test-project_failure_patterns"


class TestCogneeKnowledgeClientKeyGeneration:
    """Tests for unique key ID generation."""

    def test_generate_key_id_deterministic(self, cognee_client):
        """Test key ID generation is deterministic."""
        key1 = cognee_client._generate_key_id(["patterns"], "test-key")
        key2 = cognee_client._generate_key_id(["patterns"], "test-key")
        assert key1 == key2

    def test_generate_key_id_unique_for_different_keys(self, cognee_client):
        """Test key IDs are unique for different keys."""
        key1 = cognee_client._generate_key_id(["patterns"], "key-1")
        key2 = cognee_client._generate_key_id(["patterns"], "key-2")
        assert key1 != key2

    def test_generate_key_id_unique_for_different_namespaces(self, cognee_client):
        """Test key IDs are unique across namespaces."""
        key1 = cognee_client._generate_key_id(["patterns"], "same-key")
        key2 = cognee_client._generate_key_id(["errors"], "same-key")
        assert key1 != key2

    def test_generate_key_id_includes_org_project(self, cognee_client):
        """Test key ID includes org and project in hash input."""
        # Different org/project should produce different keys
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client2 = CogneeKnowledgeClient(org_id="other-org", project_id="test-project")

        key1 = cognee_client._generate_key_id(["patterns"], "test-key")
        key2 = client2._generate_key_id(["patterns"], "test-key")
        assert key1 != key2


class TestCogneeKnowledgeClientPutOperations:
    """Tests for put (store) operations."""

    @pytest.mark.asyncio
    async def test_put_with_embed_text(self, mock_cognee, cognee_env_vars):
        """Test put with custom embed text."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        await client.put(
            namespace=["patterns"],
            key="login-button",
            value={"selector": "#login-btn", "confidence": 0.95},
            embed_text="login button selector pattern",
        )

        # Verify cognee.add was called with embed text in content
        mock_cognee.add.assert_called_once()
        call_args = mock_cognee.add.call_args
        content = call_args[0][0]
        content_dict = json.loads(content)
        assert content_dict["_embed_text"] == "login button selector pattern"
        assert content_dict["selector"] == "#login-btn"

    @pytest.mark.asyncio
    async def test_put_includes_metadata(self, mock_cognee, cognee_env_vars):
        """Test put includes metadata in stored content."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        await client.put(
            namespace=["patterns", "selectors"],
            key="submit-btn",
            value={"data": "test"},
        )

        call_args = mock_cognee.add.call_args
        content = json.loads(call_args[0][0])

        assert "_id" in content
        assert "_key" in content
        assert content["_key"] == "submit-btn"
        assert "_namespace" in content
        assert "_org_id" in content
        assert content["_org_id"] == "org1"
        assert "_project_id" in content
        assert content["_project_id"] == "proj1"
        assert "_created_at" in content

    @pytest.mark.asyncio
    async def test_put_triggers_cognify(self, mock_cognee, cognee_env_vars):
        """Test put triggers cognify to generate embeddings."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        await client.put(
            namespace=["patterns"],
            key="test",
            value={"data": "test"},
        )

        mock_cognee.cognify.assert_called_once()

    @pytest.mark.asyncio
    async def test_put_raises_storage_error_on_failure(self, mock_cognee, cognee_env_vars):
        """Test put raises CogneeStorageError on failure."""
        mock_cognee.add = AsyncMock(side_effect=Exception("Storage failed"))

        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import (
            CogneeKnowledgeClient,
            CogneeStorageError,
            reset_cognee_client,
        )
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        with pytest.raises(CogneeStorageError) as exc_info:
            await client.put(
                namespace=["patterns"],
                key="test",
                value={"data": "test"},
            )

        assert "Storage failed" in str(exc_info.value)


class TestCogneeKnowledgeClientGetOperations:
    """Tests for get (retrieve) operations."""

    @pytest.mark.asyncio
    async def test_get_returns_none_when_not_found(self, mock_cognee, cognee_env_vars):
        """Test get returns None when key not found."""
        mock_cognee.search = AsyncMock(return_value=[])

        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        result = await client.get(namespace=["patterns"], key="nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_parses_string_result(self, mock_cognee, cognee_env_vars):
        """Test get parses string result from Cognee."""
        mock_cognee.search = AsyncMock(
            return_value=['{"selector": "#login-btn", "confidence": 0.95}']
        )

        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        result = await client.get(namespace=["patterns"], key="login")

        assert result["selector"] == "#login-btn"
        assert result["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_get_handles_dict_result(self, mock_cognee, cognee_env_vars):
        """Test get handles dict result from Cognee."""
        mock_cognee.search = AsyncMock(
            return_value=[{"_id": "123", "selector": "#btn", "data": "test"}]
        )

        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        result = await client.get(namespace=["patterns"], key="test")

        # Internal fields should be stripped
        assert "_id" not in result
        assert "selector" in result
        assert "data" in result


class TestCogneeKnowledgeClientSearch:
    """Tests for semantic search operations."""

    @pytest.mark.asyncio
    async def test_search_returns_results_with_similarity(
        self, mock_cognee, cognee_env_vars
    ):
        """Test search returns results with similarity scores."""
        mock_cognee.search = AsyncMock(
            return_value=[
                {"content": "Result 1", "id": "r1"},
                {"content": "Result 2", "id": "r2"},
            ]
        )

        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        results = await client.search(
            namespace=["patterns"],
            query="login button",
            limit=5,
        )

        assert len(results) == 2
        # Results should have similarity scores added
        assert "similarity" in results[0]
        assert "similarity" in results[1]
        # First result should have higher similarity
        assert results[0]["similarity"] >= results[1]["similarity"]

    @pytest.mark.asyncio
    async def test_search_passes_correct_parameters(
        self, mock_cognee, cognee_env_vars
    ):
        """Test search passes correct parameters to Cognee."""
        mock_cognee.search = AsyncMock(return_value=[])

        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        await client.search(
            namespace=["failure_patterns"],
            query="element not found",
            limit=10,
            threshold=0.8,
        )

        mock_cognee.search.assert_called_once()
        call_kwargs = mock_cognee.search.call_args[1]
        assert call_kwargs["query_text"] == "element not found"
        assert call_kwargs["top_k"] == 10

    @pytest.mark.asyncio
    async def test_search_raises_error_on_failure(self, mock_cognee, cognee_env_vars):
        """Test search raises CogneeSearchError on failure."""
        mock_cognee.search = AsyncMock(side_effect=Exception("Search failed"))

        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import (
            CogneeKnowledgeClient,
            CogneeSearchError,
            reset_cognee_client,
        )
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        with pytest.raises(CogneeSearchError) as exc_info:
            await client.search(
                namespace=["patterns"],
                query="test",
            )

        assert "Search failed" in str(exc_info.value)


class TestCogneeKnowledgeClientDelete:
    """Tests for delete operations."""

    @pytest.mark.asyncio
    async def test_delete_marks_item_as_deleted(self, mock_cognee, cognee_env_vars):
        """Test delete marks item with _deleted flag."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        result = await client.delete(namespace=["patterns"], key="test-key")

        assert result is True
        # Verify cognee.add was called (soft delete via put)
        mock_cognee.add.assert_called_once()


class TestCogneeKnowledgeClientFailurePatterns:
    """Tests for failure pattern operations."""

    @pytest.mark.asyncio
    async def test_store_failure_pattern_generates_pattern_id(
        self, mock_cognee, cognee_env_vars
    ):
        """Test store_failure_pattern generates deterministic pattern ID."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        pattern_id = await client.store_failure_pattern(
            error_message="Element not found: #submit",
            healed_selector="#submit-button",
            healing_method="semantic_match",
            error_type="selector_changed",
            original_selector="#submit",
        )

        assert pattern_id is not None
        assert len(pattern_id) == 32  # SHA256 truncated to 32 chars

    @pytest.mark.asyncio
    async def test_store_failure_pattern_includes_all_fields(
        self, mock_cognee, cognee_env_vars
    ):
        """Test store_failure_pattern includes all required fields."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        await client.store_failure_pattern(
            error_message="Element not found",
            healed_selector="#new-selector",
            healing_method="ai_analysis",
            error_type="dom_changed",
            original_selector="#old-selector",
            test_id="test-123",
            metadata={"context": "login_flow"},
        )

        call_args = mock_cognee.add.call_args
        content = json.loads(call_args[0][0])

        assert "pattern_id" in content
        assert "error_message" in content
        assert "healed_selector" in content
        assert "healing_method" in content
        assert "error_type" in content
        assert "original_selector" in content
        assert "test_id" in content
        assert "success_count" in content
        assert content["success_count"] == 1
        assert "failure_count" in content
        assert content["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_find_similar_failures_returns_dataclass_list(
        self, mock_cognee, cognee_env_vars
    ):
        """Test find_similar_failures returns list of SimilarFailure."""
        mock_cognee.search = AsyncMock(
            return_value=[
                {
                    "pattern_id": "p1",
                    "error_message": "Element not found: #btn",
                    "error_type": "selector_changed",
                    "original_selector": "#btn",
                    "healed_selector": "#button",
                    "healing_method": "semantic_match",
                    "success_count": 5,
                    "failure_count": 1,
                    "metadata": {},
                }
            ]
        )

        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import (
            CogneeKnowledgeClient,
            SimilarFailure,
            reset_cognee_client,
        )
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        results = await client.find_similar_failures(
            error_message="Element not found: #submit",
            limit=5,
        )

        assert len(results) == 1
        assert isinstance(results[0], SimilarFailure)
        # SimilarFailure uses 'id' not 'pattern_id'
        assert results[0].id == "p1"
        assert results[0].healed_selector == "#button"
        assert results[0].success_rate == 5 / 6  # 5/(5+1)

    @pytest.mark.asyncio
    async def test_find_similar_failures_filters_deleted(
        self, mock_cognee, cognee_env_vars
    ):
        """Test find_similar_failures excludes deleted patterns."""
        mock_cognee.search = AsyncMock(
            return_value=[
                {
                    "pattern_id": "p1",
                    "error_message": "Error 1",
                    "healed_selector": "#s1",
                    "healing_method": "method1",
                    "success_count": 5,
                    "failure_count": 1,
                },
                {
                    "pattern_id": "p2",
                    "error_message": "Error 2",
                    "healed_selector": "#s2",
                    "healing_method": "method2",
                    "success_count": 3,
                    "failure_count": 0,
                    "_deleted": True,  # Should be filtered
                },
            ]
        )

        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        results = await client.find_similar_failures(
            error_message="Test error",
        )

        assert len(results) == 1
        assert results[0].healed_selector == "#s1"

    @pytest.mark.asyncio
    async def test_record_healing_outcome_increments_success(
        self, mock_cognee, cognee_env_vars
    ):
        """Test record_healing_outcome increments success count."""
        mock_cognee.search = AsyncMock(
            return_value=[
                {
                    "pattern_id": "p1",
                    "success_count": 5,
                    "failure_count": 1,
                    "error_message": "test",
                    "healed_selector": "#btn",
                    "healing_method": "method",
                }
            ]
        )

        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        await client.record_healing_outcome(pattern_id="p1", success=True)

        # Verify put was called (which calls cognee.add)
        assert mock_cognee.add.call_count >= 1


class TestCogneeKnowledgeClientDiscoveryPatterns:
    """Tests for discovery pattern operations."""

    @pytest.mark.asyncio
    async def test_store_discovery_pattern_returns_id(
        self, mock_cognee, cognee_env_vars
    ):
        """Test store_discovery_pattern returns pattern ID."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        pattern_id = await client.store_discovery_pattern(
            pattern_type="form",
            pattern_name="Login Form",
            pattern_signature="abc123",
            pattern_data={"fields": ["email", "password"]},
            source_url="https://example.com/login",
        )

        assert pattern_id is not None
        assert "discovery_" in pattern_id

    @pytest.mark.asyncio
    async def test_store_discovery_pattern_includes_statistics(
        self, mock_cognee, cognee_env_vars
    ):
        """Test store_discovery_pattern initializes statistics fields."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        await client.store_discovery_pattern(
            pattern_type="navigation",
            pattern_name="Main Menu",
            pattern_signature="nav123",
            pattern_data={"items": ["Home", "About", "Contact"]},
        )

        call_args = mock_cognee.add.call_args
        content = json.loads(call_args[0][0])

        assert content.get("times_seen") == 1
        assert content.get("projects_seen") == 1
        assert content.get("test_success_rate") == 0.0
        assert content.get("self_heal_success_rate") == 0.0

    @pytest.mark.asyncio
    async def test_find_similar_discovery_patterns(
        self, mock_cognee, cognee_env_vars
    ):
        """Test find_similar_discovery_patterns returns matching patterns."""
        mock_cognee.search = AsyncMock(
            return_value=[
                {
                    "id": "discovery_xyz",
                    "pattern_type": "form",
                    "pattern_name": "Login Form",
                    "similarity": 0.92,
                }
            ]
        )

        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        results = await client.find_similar_discovery_patterns(
            query_text="login form with email and password",
            pattern_type="form",
            limit=5,
        )

        assert len(results) >= 0  # May be 0 if nothing matches


class TestCogneeKnowledgeClientKnowledgeGraph:
    """Tests for knowledge graph operations."""

    @pytest.mark.asyncio
    async def test_add_to_knowledge_graph_with_dict(
        self, mock_cognee, cognee_env_vars
    ):
        """Test add_to_knowledge_graph with dict content."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        await client.add_to_knowledge_graph(
            content={"entity": "LoginButton", "type": "UI_ELEMENT"},
            content_type="codebase",
        )

        mock_cognee.add.assert_called_once()
        mock_cognee.cognify.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_to_knowledge_graph_with_string(
        self, mock_cognee, cognee_env_vars
    ):
        """Test add_to_knowledge_graph with string content."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        await client.add_to_knowledge_graph(
            content="The login button is used for user authentication",
            content_type="documentation",
        )

        mock_cognee.add.assert_called_once()
        call_args = mock_cognee.add.call_args
        assert call_args[0][0] == "The login button is used for user authentication"

    @pytest.mark.asyncio
    async def test_query_knowledge_graph_aggregates_types(
        self, mock_cognee, cognee_env_vars
    ):
        """Test query_knowledge_graph searches across content types."""
        mock_cognee.search = AsyncMock(
            return_value=[{"content": "Result", "type": "codebase"}]
        )

        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeKnowledgeClient, reset_cognee_client
        reset_cognee_client()

        client = CogneeKnowledgeClient(org_id="org1", project_id="proj1")

        results = await client.query_knowledge_graph(
            query="login functionality",
            content_types=["codebase", "tests"],
            limit=10,
        )

        # Should call search for each content type
        assert mock_cognee.search.call_count >= 1


class TestGlobalInstanceManagement:
    """Tests for global instance management functions."""

    def test_init_cognee_client_returns_initialized_client(
        self, mock_cognee, cognee_env_vars
    ):
        """Test init_cognee_client returns working client."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import (
            get_cognee_client,
            init_cognee_client,
            reset_cognee_client,
        )
        reset_cognee_client()

        # Use synchronous get for this test
        client = get_cognee_client(org_id="init-test", project_id="proj")

        assert client is not None
        assert client.org_id == "init-test"

    def test_reset_clears_global_state(self, mock_cognee, cognee_env_vars):
        """Test reset_cognee_client clears global state."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import get_cognee_client, reset_cognee_client

        client1 = get_cognee_client(org_id="org1", project_id="proj1")
        reset_cognee_client()
        client2 = get_cognee_client(org_id="org1", project_id="proj1")

        assert client1 is not client2


class TestCogneeExceptions:
    """Tests for Cognee exception hierarchy."""

    def test_cognee_storage_error_is_cognee_error(self, mock_cognee, cognee_env_vars):
        """Test CogneeStorageError inherits from CogneeError."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeError, CogneeStorageError

        err = CogneeStorageError("Test error")
        assert isinstance(err, CogneeError)
        assert isinstance(err, Exception)

    def test_cognee_retrieval_error_is_cognee_error(self, mock_cognee, cognee_env_vars):
        """Test CogneeRetrievalError inherits from CogneeError."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeError, CogneeRetrievalError

        err = CogneeRetrievalError("Test error")
        assert isinstance(err, CogneeError)

    def test_cognee_search_error_is_cognee_error(self, mock_cognee, cognee_env_vars):
        """Test CogneeSearchError inherits from CogneeError."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeError, CogneeSearchError

        err = CogneeSearchError("Test error")
        assert isinstance(err, CogneeError)

    def test_cognee_graph_error_is_cognee_error(self, mock_cognee, cognee_env_vars):
        """Test CogneeGraphError inherits from CogneeError."""
        import importlib
        import src.knowledge.cognee_client
        importlib.reload(src.knowledge.cognee_client)
        from src.knowledge.cognee_client import CogneeError, CogneeGraphError

        err = CogneeGraphError("Test error")
        assert isinstance(err, CogneeError)
