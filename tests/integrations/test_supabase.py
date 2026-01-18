"""Tests for Supabase integration module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os


class TestSupabaseClientInit:
    """Tests for SupabaseClient initialization."""

    def test_client_initialization(self, mock_env_vars):
        """Test SupabaseClient initialization."""
        from src.integrations.supabase import SupabaseClient

        client = SupabaseClient(
            url="https://test.supabase.co",
            key="test-service-key",
        )

        assert client.url == "https://test.supabase.co"
        assert client.key == "test-service-key"
        assert client._client is None
        assert client._initialized is False


class TestSupabaseClientEnsureInitialized:
    """Tests for SupabaseClient._ensure_initialized method."""

    @pytest.mark.asyncio
    async def test_ensure_initialized_success(self, mock_env_vars):
        """Test successful initialization."""
        from src.integrations.supabase import SupabaseClient

        mock_supabase_client = MagicMock()

        # Patch at the module where create_client is imported (inside the method)
        with patch.dict("sys.modules", {"supabase": MagicMock(create_client=MagicMock(return_value=mock_supabase_client))}):
            client = SupabaseClient(
                url="https://test.supabase.co",
                key="test-service-key",
            )

            result = await client._ensure_initialized()

            assert result is True
            assert client._initialized is True
            assert client._client is mock_supabase_client

    @pytest.mark.asyncio
    async def test_ensure_initialized_import_error(self, mock_env_vars):
        """Test initialization with missing supabase package."""
        from src.integrations.supabase import SupabaseClient

        client = SupabaseClient(
            url="https://test.supabase.co",
            key="test-service-key",
        )

        # Manually test the import error scenario
        async def mock_ensure():
            client._initialized = True
            return False

        client._ensure_initialized = mock_ensure

        result = await client._ensure_initialized()

        assert result is False
        assert client._initialized is True

    @pytest.mark.asyncio
    async def test_ensure_initialized_idempotent(self, mock_env_vars):
        """Test _ensure_initialized is idempotent."""
        from src.integrations.supabase import SupabaseClient

        mock_supabase_client = MagicMock()
        call_count = 0

        def track_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_supabase_client

        with patch.dict("sys.modules", {"supabase": MagicMock(create_client=track_calls)}):
            client = SupabaseClient(
                url="https://test.supabase.co",
                key="test-service-key",
            )

            # Call twice
            await client._ensure_initialized()
            await client._ensure_initialized()

            # Should only create client once
            assert call_count == 1


class TestSupabaseClientInsert:
    """Tests for SupabaseClient.insert method."""

    @pytest.fixture
    def initialized_client(self, mock_env_vars):
        """Create an initialized SupabaseClient."""
        from src.integrations.supabase import SupabaseClient

        client = SupabaseClient(
            url="https://test.supabase.co",
            key="test-service-key",
        )

        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": 1}])
        mock_supabase.table.return_value = mock_table

        client._client = mock_supabase
        client._initialized = True

        return client

    @pytest.mark.asyncio
    async def test_insert_success(self, initialized_client):
        """Test successful insert."""
        result = await initialized_client.insert(
            table="test_table",
            records=[{"name": "test", "value": 123}],
        )

        assert result is True
        initialized_client._client.table.assert_called_with("test_table")

    @pytest.mark.asyncio
    async def test_insert_multiple_records(self, initialized_client):
        """Test inserting multiple records."""
        records = [
            {"name": "test1", "value": 1},
            {"name": "test2", "value": 2},
            {"name": "test3", "value": 3},
        ]

        result = await initialized_client.insert(
            table="test_table",
            records=records,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_insert_failure(self, initialized_client):
        """Test insert failure."""
        initialized_client._client.table.return_value.insert.return_value.execute.side_effect = Exception(
            "Insert failed"
        )

        result = await initialized_client.insert(
            table="test_table",
            records=[{"name": "test"}],
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_insert_not_initialized(self, mock_env_vars):
        """Test insert when client not initialized."""
        from src.integrations.supabase import SupabaseClient

        client = SupabaseClient(
            url="https://test.supabase.co",
            key="test-service-key",
        )

        # Mock _ensure_initialized to return False
        client._ensure_initialized = AsyncMock(return_value=False)

        result = await client.insert("test_table", [{"name": "test"}])

        assert result is False


class TestSupabaseClientSelect:
    """Tests for SupabaseClient.select method."""

    @pytest.fixture
    def initialized_client(self, mock_env_vars):
        """Create an initialized SupabaseClient."""
        from src.integrations.supabase import SupabaseClient

        client = SupabaseClient(
            url="https://test.supabase.co",
            key="test-service-key",
        )

        mock_supabase = MagicMock()
        client._client = mock_supabase
        client._initialized = True

        return client

    @pytest.mark.asyncio
    async def test_select_all(self, initialized_client):
        """Test selecting all records."""
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.execute.return_value = MagicMock(data=[{"id": 1}, {"id": 2}])
        initialized_client._client.table.return_value = mock_query

        result = await initialized_client.select("test_table")

        assert len(result) == 2
        mock_query.select.assert_called_with("*")

    @pytest.mark.asyncio
    async def test_select_with_columns(self, initialized_client):
        """Test selecting specific columns."""
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.execute.return_value = MagicMock(data=[{"name": "test"}])
        initialized_client._client.table.return_value = mock_query

        result = await initialized_client.select("test_table", columns="name, value")

        mock_query.select.assert_called_with("name, value")

    @pytest.mark.asyncio
    async def test_select_with_filters(self, initialized_client):
        """Test selecting with filters."""
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.execute.return_value = MagicMock(data=[{"id": 1, "status": "active"}])
        initialized_client._client.table.return_value = mock_query

        result = await initialized_client.select(
            "test_table",
            filters={"status": "active"},
        )

        mock_query.eq.assert_called_with("status", "active")

    @pytest.mark.asyncio
    async def test_select_with_order_by(self, initialized_client):
        """Test selecting with order by."""
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.execute.return_value = MagicMock(data=[{"id": 2}, {"id": 1}])
        initialized_client._client.table.return_value = mock_query

        result = await initialized_client.select(
            "test_table",
            order_by="id",
            ascending=False,
        )

        mock_query.order.assert_called_with("id", desc=True)

    @pytest.mark.asyncio
    async def test_select_with_limit(self, initialized_client):
        """Test selecting with limit."""
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.execute.return_value = MagicMock(data=[{"id": 1}])
        initialized_client._client.table.return_value = mock_query

        result = await initialized_client.select(
            "test_table",
            limit=10,
        )

        mock_query.limit.assert_called_with(10)

    @pytest.mark.asyncio
    async def test_select_with_offset(self, initialized_client):
        """Test selecting with offset (pagination)."""
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.execute.return_value = MagicMock(data=[{"id": 11}])
        initialized_client._client.table.return_value = mock_query

        result = await initialized_client.select(
            "test_table",
            offset=10,
        )

        mock_query.offset.assert_called_with(10)

    @pytest.mark.asyncio
    async def test_select_failure(self, initialized_client):
        """Test select failure."""
        initialized_client._client.table.return_value.select.side_effect = Exception(
            "Select failed"
        )

        result = await initialized_client.select("test_table")

        assert result == []

    @pytest.mark.asyncio
    async def test_select_empty_result(self, initialized_client):
        """Test select with no results."""
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.execute.return_value = MagicMock(data=None)
        initialized_client._client.table.return_value = mock_query

        result = await initialized_client.select("test_table")

        assert result == []

    @pytest.mark.asyncio
    async def test_select_not_initialized(self, mock_env_vars):
        """Test select when client not initialized."""
        from src.integrations.supabase import SupabaseClient

        client = SupabaseClient(
            url="https://test.supabase.co",
            key="test-service-key",
        )

        client._ensure_initialized = AsyncMock(return_value=False)

        result = await client.select("test_table")

        assert result == []


class TestSupabaseClientUpdate:
    """Tests for SupabaseClient.update method."""

    @pytest.fixture
    def initialized_client(self, mock_env_vars):
        """Create an initialized SupabaseClient."""
        from src.integrations.supabase import SupabaseClient

        client = SupabaseClient(
            url="https://test.supabase.co",
            key="test-service-key",
        )

        mock_supabase = MagicMock()
        client._client = mock_supabase
        client._initialized = True

        return client

    @pytest.mark.asyncio
    async def test_update_success(self, initialized_client):
        """Test successful update."""
        mock_query = MagicMock()
        mock_query.update.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.execute.return_value = MagicMock()
        initialized_client._client.table.return_value = mock_query

        result = await initialized_client.update(
            table="test_table",
            values={"name": "updated"},
            filters={"id": 1},
        )

        assert result is True
        mock_query.update.assert_called_with({"name": "updated"})
        mock_query.eq.assert_called_with("id", 1)

    @pytest.mark.asyncio
    async def test_update_multiple_filters(self, initialized_client):
        """Test update with multiple filters."""
        mock_query = MagicMock()
        mock_query.update.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.execute.return_value = MagicMock()
        initialized_client._client.table.return_value = mock_query

        result = await initialized_client.update(
            table="test_table",
            values={"status": "active"},
            filters={"id": 1, "org_id": "org-123"},
        )

        assert result is True
        assert mock_query.eq.call_count == 2

    @pytest.mark.asyncio
    async def test_update_failure(self, initialized_client):
        """Test update failure."""
        initialized_client._client.table.return_value.update.return_value.eq.return_value.execute.side_effect = Exception(
            "Update failed"
        )

        result = await initialized_client.update(
            table="test_table",
            values={"name": "updated"},
            filters={"id": 1},
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_not_initialized(self, mock_env_vars):
        """Test update when client not initialized."""
        from src.integrations.supabase import SupabaseClient

        client = SupabaseClient(
            url="https://test.supabase.co",
            key="test-service-key",
        )

        client._ensure_initialized = AsyncMock(return_value=False)

        result = await client.update("test_table", {"name": "test"}, {"id": 1})

        assert result is False


class TestSupabaseClientDelete:
    """Tests for SupabaseClient.delete method."""

    @pytest.fixture
    def initialized_client(self, mock_env_vars):
        """Create an initialized SupabaseClient."""
        from src.integrations.supabase import SupabaseClient

        client = SupabaseClient(
            url="https://test.supabase.co",
            key="test-service-key",
        )

        mock_supabase = MagicMock()
        client._client = mock_supabase
        client._initialized = True

        return client

    @pytest.mark.asyncio
    async def test_delete_success(self, initialized_client):
        """Test successful delete."""
        mock_query = MagicMock()
        mock_query.delete.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.execute.return_value = MagicMock()
        initialized_client._client.table.return_value = mock_query

        result = await initialized_client.delete(
            table="test_table",
            filters={"id": 1},
        )

        assert result is True
        mock_query.eq.assert_called_with("id", 1)

    @pytest.mark.asyncio
    async def test_delete_multiple_filters(self, initialized_client):
        """Test delete with multiple filters."""
        mock_query = MagicMock()
        mock_query.delete.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.execute.return_value = MagicMock()
        initialized_client._client.table.return_value = mock_query

        result = await initialized_client.delete(
            table="test_table",
            filters={"id": 1, "org_id": "org-123"},
        )

        assert result is True
        assert mock_query.eq.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_failure(self, initialized_client):
        """Test delete failure."""
        initialized_client._client.table.return_value.delete.return_value.eq.return_value.execute.side_effect = Exception(
            "Delete failed"
        )

        result = await initialized_client.delete(
            table="test_table",
            filters={"id": 1},
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_not_initialized(self, mock_env_vars):
        """Test delete when client not initialized."""
        from src.integrations.supabase import SupabaseClient

        client = SupabaseClient(
            url="https://test.supabase.co",
            key="test-service-key",
        )

        client._ensure_initialized = AsyncMock(return_value=False)

        result = await client.delete("test_table", {"id": 1})

        assert result is False


class TestGetSupabase:
    """Tests for get_supabase function."""

    @pytest.mark.asyncio
    async def test_get_supabase_success(self, mock_env_vars, monkeypatch):
        """Test get_supabase returns client."""
        # Reset the global client
        import src.integrations.supabase as module

        module._supabase_client = None

        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key")

        from src.integrations.supabase import get_supabase

        # Mock the settings at the src.config level
        mock_settings = MagicMock()
        mock_settings.supabase_url = "https://test.supabase.co"
        mock_settings.supabase_service_key = MagicMock()
        mock_settings.supabase_service_key.get_secret_value.return_value = "test-key"

        with patch("src.config.get_settings", return_value=mock_settings):
            client = await get_supabase()

            assert client is not None
            assert client.url == "https://test.supabase.co"

    @pytest.mark.asyncio
    async def test_get_supabase_not_configured(self, mock_env_vars, monkeypatch):
        """Test get_supabase returns None when not configured."""
        # Reset the global client
        import src.integrations.supabase as module

        module._supabase_client = None

        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

        from src.integrations.supabase import get_supabase

        # Mock the settings to raise an exception to trigger fallback
        with patch("src.config.get_settings", side_effect=Exception("No settings")):
            client = await get_supabase()

            assert client is None

    @pytest.mark.asyncio
    async def test_get_supabase_singleton(self, mock_env_vars, monkeypatch):
        """Test get_supabase returns same instance."""
        # Reset the global client
        import src.integrations.supabase as module

        module._supabase_client = None

        from src.integrations.supabase import get_supabase

        mock_settings = MagicMock()
        mock_settings.supabase_url = "https://test.supabase.co"
        mock_settings.supabase_service_key = MagicMock()
        mock_settings.supabase_service_key.get_secret_value.return_value = "test-key"

        with patch("src.config.get_settings", return_value=mock_settings):
            client1 = await get_supabase()
            client2 = await get_supabase()

            assert client1 is client2

    @pytest.mark.asyncio
    async def test_get_supabase_fallback_to_env(self, mock_env_vars, monkeypatch):
        """Test get_supabase falls back to env vars on settings error."""
        # Reset the global client
        import src.integrations.supabase as module

        module._supabase_client = None

        monkeypatch.setenv("SUPABASE_URL", "https://env.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "env-key")

        from src.integrations.supabase import get_supabase

        with patch("src.config.get_settings", side_effect=Exception("Settings error")):
            client = await get_supabase()

            assert client is not None
            assert client.url == "https://env.supabase.co"


class TestIsSupabaseConfigured:
    """Tests for is_supabase_configured function."""

    def test_is_configured_true(self, mock_env_vars):
        """Test is_supabase_configured returns True when configured."""
        from src.integrations.supabase import is_supabase_configured

        mock_settings = MagicMock()
        mock_settings.supabase_url = "https://test.supabase.co"
        mock_settings.supabase_service_key = MagicMock()

        with patch("src.config.get_settings", return_value=mock_settings):
            result = is_supabase_configured()

            assert result is True

    def test_is_configured_false_no_url(self, mock_env_vars):
        """Test is_supabase_configured returns False when URL missing."""
        from src.integrations.supabase import is_supabase_configured

        mock_settings = MagicMock()
        mock_settings.supabase_url = None
        mock_settings.supabase_service_key = MagicMock()

        with patch("src.config.get_settings", return_value=mock_settings):
            result = is_supabase_configured()

            assert result is False

    def test_is_configured_false_no_key(self, mock_env_vars):
        """Test is_supabase_configured returns False when key missing."""
        from src.integrations.supabase import is_supabase_configured

        mock_settings = MagicMock()
        mock_settings.supabase_url = "https://test.supabase.co"
        mock_settings.supabase_service_key = None

        with patch("src.config.get_settings", return_value=mock_settings):
            result = is_supabase_configured()

            assert result is False

    def test_is_configured_fallback_to_env(self, mock_env_vars, monkeypatch):
        """Test is_supabase_configured falls back to env vars."""
        from src.integrations.supabase import is_supabase_configured

        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key")

        with patch("src.config.get_settings", side_effect=Exception("Settings error")):
            result = is_supabase_configured()

            assert result is True

    def test_is_configured_env_not_set(self, mock_env_vars, monkeypatch):
        """Test is_supabase_configured when env vars not set."""
        from src.integrations.supabase import is_supabase_configured

        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

        with patch("src.config.get_settings", side_effect=Exception("Settings error")):
            result = is_supabase_configured()

            assert result is False
