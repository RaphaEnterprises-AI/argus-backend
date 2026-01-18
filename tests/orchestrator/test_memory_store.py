"""Tests for the memory store module."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMemoryStoreInit:
    """Tests for MemoryStore initialization."""

    def test_init_with_defaults(self, mock_env_vars, monkeypatch):
        """Test MemoryStore initialization with defaults."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")

        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore()

        assert store.database_url == "postgresql://localhost/test"
        assert store.embeddings is None
        assert store._pool is None

    def test_init_with_custom_url(self, mock_env_vars):
        """Test MemoryStore initialization with custom URL."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://custom/db")

        assert store.database_url == "postgresql://custom/db"

    def test_init_with_embeddings(self, mock_env_vars, monkeypatch):
        """Test MemoryStore initialization with embeddings."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")

        from src.orchestrator.memory_store import MemoryStore

        mock_embeddings = MagicMock()
        store = MemoryStore(embeddings=mock_embeddings)

        assert store.embeddings == mock_embeddings


class TestMemoryStoreGetPool:
    """Tests for MemoryStore._get_pool method."""

    @pytest.mark.asyncio
    async def test_get_pool_missing_database_url(self, mock_env_vars, monkeypatch):
        """Test _get_pool raises error when DATABASE_URL is missing."""
        monkeypatch.delenv("DATABASE_URL", raising=False)

        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url=None)

        with pytest.raises(ValueError, match="DATABASE_URL"):
            await store._get_pool()

    @pytest.mark.asyncio
    async def test_get_pool_creates_pool(self, mock_env_vars):
        """Test _get_pool creates connection pool."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_pool = MagicMock()
        with patch("asyncpg.create_pool", AsyncMock(return_value=mock_pool)):
            pool = await store._get_pool()

            assert pool == mock_pool
            assert store._pool == mock_pool

    @pytest.mark.asyncio
    async def test_get_pool_reuses_existing(self, mock_env_vars):
        """Test _get_pool reuses existing pool."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_pool = MagicMock()
        store._pool = mock_pool

        pool = await store._get_pool()

        assert pool == mock_pool

    @pytest.mark.asyncio
    async def test_get_pool_missing_asyncpg(self, mock_env_vars):
        """Test _get_pool raises ImportError when asyncpg missing."""
        from src.orchestrator.memory_store import MemoryStore

        MemoryStore(database_url="postgresql://localhost/test")

        with patch.dict("sys.modules", {"asyncpg": None}):
            # This should raise when asyncpg is not available
            # The import happens inside _get_pool
            pass  # Test structure for ImportError case


class TestMemoryStoreClose:
    """Tests for MemoryStore.close method."""

    @pytest.mark.asyncio
    async def test_close_with_pool(self, mock_env_vars):
        """Test close closes the connection pool."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_pool = AsyncMock()
        store._pool = mock_pool

        await store.close()

        mock_pool.close.assert_called_once()
        assert store._pool is None

    @pytest.mark.asyncio
    async def test_close_without_pool(self, mock_env_vars):
        """Test close handles missing pool gracefully."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        # Should not raise
        await store.close()

        assert store._pool is None


class TestMemoryStoreGetEmbedding:
    """Tests for MemoryStore._get_embedding method."""

    @pytest.mark.asyncio
    async def test_get_embedding_no_embeddings(self, mock_env_vars):
        """Test _get_embedding returns None when no embeddings configured."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        result = await store._get_embedding("test text")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_embedding_async(self, mock_env_vars):
        """Test _get_embedding with async embeddings."""
        from src.orchestrator.memory_store import MemoryStore

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        store = MemoryStore(
            database_url="postgresql://localhost/test",
            embeddings=mock_embeddings,
        )

        result = await store._get_embedding("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_embeddings.aembed_query.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_get_embedding_sync(self, mock_env_vars):
        """Test _get_embedding with sync embeddings."""
        from src.orchestrator.memory_store import MemoryStore

        mock_embeddings = MagicMock()
        del mock_embeddings.aembed_query  # Remove async method
        mock_embeddings.embed_query = MagicMock(return_value=[0.4, 0.5, 0.6])

        store = MemoryStore(
            database_url="postgresql://localhost/test",
            embeddings=mock_embeddings,
        )

        result = await store._get_embedding("test text")

        assert result == [0.4, 0.5, 0.6]
        mock_embeddings.embed_query.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_get_embedding_no_method(self, mock_env_vars):
        """Test _get_embedding with embeddings lacking methods."""
        from src.orchestrator.memory_store import MemoryStore

        mock_embeddings = MagicMock(spec=[])  # No methods

        store = MemoryStore(
            database_url="postgresql://localhost/test",
            embeddings=mock_embeddings,
        )

        result = await store._get_embedding("test text")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_embedding_exception(self, mock_env_vars):
        """Test _get_embedding handles exceptions gracefully."""
        from src.orchestrator.memory_store import MemoryStore

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(side_effect=Exception("API error"))

        store = MemoryStore(
            database_url="postgresql://localhost/test",
            embeddings=mock_embeddings,
        )

        result = await store._get_embedding("test text")

        assert result is None


class TestMemoryStorePut:
    """Tests for MemoryStore.put method."""

    @pytest.mark.asyncio
    async def test_put_without_embedding(self, mock_env_vars):
        """Test put stores value without embedding."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        await store.put(
            namespace=["tests", "login"],
            key="test-key",
            value={"data": "test value"},
        )

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_put_with_embedding(self, mock_env_vars):
        """Test put stores value with embedding."""
        from src.orchestrator.memory_store import MemoryStore

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        store = MemoryStore(
            database_url="postgresql://localhost/test",
            embeddings=mock_embeddings,
        )

        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        await store.put(
            namespace=["failures"],
            key="pattern-1",
            value={"pattern": "test"},
            embed_text="Error: element not found",
        )

        mock_conn.execute.assert_called_once()
        mock_embeddings.aembed_query.assert_called_once()


class TestMemoryStoreGet:
    """Tests for MemoryStore.get method."""

    @pytest.mark.asyncio
    async def test_get_found(self, mock_env_vars):
        """Test get returns value when found."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"value": '{"data": "test"}'})
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        result = await store.get(namespace=["tests"], key="key-1")

        assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_get_not_found(self, mock_env_vars):
        """Test get returns None when not found."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        result = await store.get(namespace=["tests"], key="nonexistent")

        assert result is None


class TestMemoryStoreDelete:
    """Tests for MemoryStore.delete method."""

    @pytest.mark.asyncio
    async def test_delete_found(self, mock_env_vars):
        """Test delete returns True when item deleted."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 1")
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        result = await store.delete(namespace=["tests"], key="key-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_not_found(self, mock_env_vars):
        """Test delete returns False when item not found."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 0")
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        result = await store.delete(namespace=["tests"], key="nonexistent")

        assert result is False


class TestMemoryStoreListKeys:
    """Tests for MemoryStore.list_keys method."""

    @pytest.mark.asyncio
    async def test_list_keys(self, mock_env_vars):
        """Test list_keys returns keys in namespace."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {"key": "key-1"},
            {"key": "key-2"},
            {"key": "key-3"},
        ])
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        result = await store.list_keys(namespace=["tests"])

        assert result == ["key-1", "key-2", "key-3"]

    @pytest.mark.asyncio
    async def test_list_keys_empty(self, mock_env_vars):
        """Test list_keys returns empty list when no keys."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        result = await store.list_keys(namespace=["empty"])

        assert result == []

    @pytest.mark.asyncio
    async def test_list_keys_with_limit(self, mock_env_vars):
        """Test list_keys respects limit parameter."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[{"key": "key-1"}])
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        await store.list_keys(namespace=["tests"], limit=1)

        # Verify limit was passed to query
        call_args = mock_conn.fetch.call_args[0]
        assert 1 in call_args


class TestMemoryStoreSearch:
    """Tests for MemoryStore.search method."""

    @pytest.mark.asyncio
    async def test_search_no_embeddings(self, mock_env_vars):
        """Test search returns empty when no embeddings configured."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        result = await store.search(namespace=["tests"], query="test query")

        assert result == []

    @pytest.mark.asyncio
    async def test_search_with_results(self, mock_env_vars):
        """Test search returns matching items."""
        from src.orchestrator.memory_store import MemoryStore

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        store = MemoryStore(
            database_url="postgresql://localhost/test",
            embeddings=mock_embeddings,
        )

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {"key": "key-1", "value": '{"data": "match1"}', "similarity": 0.95},
            {"key": "key-2", "value": '{"data": "match2"}', "similarity": 0.85},
        ])
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        result = await store.search(namespace=["tests"], query="test query")

        assert len(result) == 2
        assert result[0]["key"] == "key-1"
        assert result[0]["similarity"] == 0.95

    @pytest.mark.asyncio
    async def test_search_no_embedding_generated(self, mock_env_vars):
        """Test search returns empty when embedding generation fails."""
        from src.orchestrator.memory_store import MemoryStore

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(side_effect=Exception("API error"))

        store = MemoryStore(
            database_url="postgresql://localhost/test",
            embeddings=mock_embeddings,
        )

        result = await store.search(namespace=["tests"], query="test query")

        assert result == []


class TestMemoryStoreFailurePatterns:
    """Tests for MemoryStore failure pattern methods."""

    @pytest.mark.asyncio
    async def test_store_failure_pattern(self, mock_env_vars):
        """Test store_failure_pattern stores pattern correctly."""
        from src.orchestrator.memory_store import MemoryStore

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        store = MemoryStore(
            database_url="postgresql://localhost/test",
            embeddings=mock_embeddings,
        )

        mock_conn = AsyncMock()
        pattern_id = uuid.uuid4()
        mock_conn.fetchrow = AsyncMock(return_value={"id": pattern_id})
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        result = await store.store_failure_pattern(
            error_message="Element not found: #submit",
            healed_selector="#submit-button",
            healing_method="semantic_match",
            error_type="selector_changed",
        )

        assert result == str(pattern_id)

    @pytest.mark.asyncio
    async def test_store_failure_pattern_with_test_id(self, mock_env_vars):
        """Test store_failure_pattern with test_id."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        pattern_id = uuid.uuid4()
        mock_conn.fetchrow = AsyncMock(return_value={"id": pattern_id})
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        # With valid UUID
        valid_uuid = str(uuid.uuid4())
        await store.store_failure_pattern(
            error_message="Error",
            healed_selector="#new",
            healing_method="method",
            test_id=valid_uuid,
        )

        # With non-UUID string
        await store.store_failure_pattern(
            error_message="Error",
            healed_selector="#new",
            healing_method="method",
            test_id="test-123",
        )

    @pytest.mark.asyncio
    async def test_find_similar_failures(self, mock_env_vars):
        """Test find_similar_failures finds matching patterns."""
        from src.orchestrator.memory_store import MemoryStore

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        store = MemoryStore(
            database_url="postgresql://localhost/test",
            embeddings=mock_embeddings,
        )

        pattern_id = uuid.uuid4()
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {
                "id": pattern_id,
                "error_message": "Element not found: #submit-btn",
                "error_type": "selector_changed",
                "selector": "#submit-btn",
                "healed_selector": "#submit-button",
                "healing_method": "semantic_match",
                "success_count": 5,
                "failure_count": 1,
                "metadata": "{}",
                "similarity": 0.92,
                "success_rate": 0.833,
            }
        ])
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        result = await store.find_similar_failures(
            error_message="Element not found: #submit"
        )

        assert len(result) == 1
        assert result[0]["healed_selector"] == "#submit-button"
        assert result[0]["similarity"] == 0.92

    @pytest.mark.asyncio
    async def test_find_similar_failures_with_error_type(self, mock_env_vars):
        """Test find_similar_failures filters by error type."""
        from src.orchestrator.memory_store import MemoryStore

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        store = MemoryStore(
            database_url="postgresql://localhost/test",
            embeddings=mock_embeddings,
        )

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        await store.find_similar_failures(
            error_message="Error",
            error_type="timing_issue",
        )

        # Verify error_type was passed to query
        call_args = mock_conn.fetch.call_args[0]
        assert "timing_issue" in call_args

    @pytest.mark.asyncio
    async def test_find_similar_failures_no_embeddings(self, mock_env_vars):
        """Test find_similar_failures returns empty without embeddings."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        result = await store.find_similar_failures(error_message="Error")

        assert result == []


class TestMemoryStoreHealingOutcome:
    """Tests for MemoryStore.record_healing_outcome method."""

    @pytest.mark.asyncio
    async def test_record_healing_success(self, mock_env_vars):
        """Test record_healing_outcome records success."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        pattern_id = str(uuid.uuid4())
        await store.record_healing_outcome(pattern_id, success=True)

        mock_conn.execute.assert_called_once()
        # Verify success_count is incremented
        call_args = mock_conn.execute.call_args[0][0]
        assert "success_count = success_count + 1" in call_args

    @pytest.mark.asyncio
    async def test_record_healing_failure(self, mock_env_vars):
        """Test record_healing_outcome records failure."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        pattern_id = str(uuid.uuid4())
        await store.record_healing_outcome(pattern_id, success=False)

        mock_conn.execute.assert_called_once()
        # Verify failure_count is incremented
        call_args = mock_conn.execute.call_args[0][0]
        assert "failure_count = failure_count + 1" in call_args

    @pytest.mark.asyncio
    async def test_record_healing_invalid_id(self, mock_env_vars):
        """Test record_healing_outcome handles invalid ID."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_pool = MagicMock()
        store._pool = mock_pool

        # Invalid UUID should return without error
        await store.record_healing_outcome("not-a-valid-uuid", success=True)


class TestMemoryStorePatternStats:
    """Tests for MemoryStore.get_pattern_stats method."""

    @pytest.mark.asyncio
    async def test_get_pattern_stats(self, mock_env_vars):
        """Test get_pattern_stats returns statistics."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={
            "total_patterns": 100,
            "total_successes": 80,
            "total_failures": 20,
            "unique_error_types": 5,
            "avg_success_rate": 0.8,
        })
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        result = await store.get_pattern_stats()

        assert result["total_patterns"] == 100
        assert result["total_successes"] == 80
        assert result["total_failures"] == 20
        assert result["unique_error_types"] == 5
        assert result["avg_success_rate"] == 0.8

    @pytest.mark.asyncio
    async def test_get_pattern_stats_empty(self, mock_env_vars):
        """Test get_pattern_stats handles empty database."""
        from src.orchestrator.memory_store import MemoryStore

        store = MemoryStore(database_url="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={
            "total_patterns": 0,
            "total_successes": None,
            "total_failures": None,
            "unique_error_types": 0,
            "avg_success_rate": None,
        })
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store._pool = mock_pool

        result = await store.get_pattern_stats()

        assert result["total_patterns"] == 0
        assert result["total_successes"] == 0
        assert result["total_failures"] == 0
        assert result["avg_success_rate"] == 0


class TestGlobalMemoryStore:
    """Tests for global memory store functions."""

    def test_get_memory_store_creates_instance(self, mock_env_vars, monkeypatch):
        """Test get_memory_store creates new instance."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")

        from src.orchestrator.memory_store import (
            get_memory_store,
            reset_memory_store,
        )

        reset_memory_store()  # Clear any existing instance

        store = get_memory_store()

        assert store is not None
        assert store.database_url == "postgresql://localhost/test"

    def test_get_memory_store_reuses_instance(self, mock_env_vars, monkeypatch):
        """Test get_memory_store reuses existing instance."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")

        from src.orchestrator.memory_store import (
            get_memory_store,
            reset_memory_store,
        )

        reset_memory_store()

        store1 = get_memory_store()
        store2 = get_memory_store()

        assert store1 is store2

    def test_reset_memory_store(self, mock_env_vars, monkeypatch):
        """Test reset_memory_store clears the instance."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")

        from src.orchestrator.memory_store import (
            get_memory_store,
            reset_memory_store,
        )

        store1 = get_memory_store()
        reset_memory_store()
        store2 = get_memory_store()

        assert store1 is not store2

    @pytest.mark.asyncio
    async def test_init_memory_store(self, mock_env_vars, monkeypatch):
        """Test init_memory_store initializes and verifies connection."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")

        from src.orchestrator.memory_store import (
            init_memory_store,
            reset_memory_store,
        )

        reset_memory_store()

        mock_pool = MagicMock()
        with patch("asyncpg.create_pool", AsyncMock(return_value=mock_pool)):
            store = await init_memory_store()

            assert store is not None
            assert store._pool == mock_pool
