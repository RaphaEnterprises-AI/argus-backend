"""Tests for the Cache service using Cloudflare KV.

This module tests:
- CloudflareKVClient operations (get, set, delete)
- Cache key generation
- Cache decorators (cache_quality_score, cache_llm_response, cache_healing_pattern)
- Direct cache operations (get_cached, set_cached, delete_cached)
- Cache health checking
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


class TestCloudflareKVClient:
    """Tests for CloudflareKVClient class."""

    @pytest.fixture
    def kv_client(self):
        """Create a CloudflareKVClient instance."""
        from src.services.cache import CloudflareKVClient
        return CloudflareKVClient(
            account_id="test-account",
            namespace_id="test-namespace",
            api_token="test-token",
        )

    def test_client_initialization(self, mock_env_vars):
        """Test CloudflareKVClient initialization."""
        from src.services.cache import CloudflareKVClient

        client = CloudflareKVClient(
            account_id="acc-123",
            namespace_id="ns-456",
            api_token="token-789",
        )

        assert client.account_id == "acc-123"
        assert client.namespace_id == "ns-456"
        assert client.api_token == "token-789"
        assert "acc-123" in client.base_url
        assert "ns-456" in client.base_url
        assert client._client is None

    def test_get_headers(self, mock_env_vars, kv_client):
        """Test that _get_headers returns correct headers."""
        headers = kv_client._get_headers()

        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, mock_env_vars, kv_client):
        """Test that _get_client creates an httpx client."""
        assert kv_client._client is None

        client = await kv_client._get_client()

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        assert kv_client._client is client

        # Cleanup
        await kv_client.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self, mock_env_vars, kv_client):
        """Test that _get_client reuses existing client."""
        client1 = await kv_client._get_client()
        client2 = await kv_client._get_client()

        assert client1 is client2

        # Cleanup
        await kv_client.close()

    @pytest.mark.asyncio
    async def test_get_success(self, mock_env_vars, kv_client):
        """Test successful get operation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"data": "test-value"}'

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        kv_client._client = mock_http_client

        result = await kv_client.get("test-key")

        assert result == '{"data": "test-value"}'
        mock_http_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_not_found(self, mock_env_vars, kv_client):
        """Test get operation when key not found."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        kv_client._client = mock_http_client

        result = await kv_client.get("nonexistent-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_error_status(self, mock_env_vars, kv_client):
        """Test get operation with error status code."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        kv_client._client = mock_http_client

        result = await kv_client.get("test-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_exception(self, mock_env_vars, kv_client):
        """Test get operation with exception."""
        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(side_effect=Exception("Network error"))
        kv_client._client = mock_http_client

        result = await kv_client.get("test-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_success(self, mock_env_vars, kv_client):
        """Test successful set operation."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_http_client = AsyncMock()
        mock_http_client.put = AsyncMock(return_value=mock_response)
        kv_client._client = mock_http_client

        result = await kv_client.set("test-key", "test-value", ex=600)

        assert result is True
        mock_http_client.put.assert_called_once()
        call_args = mock_http_client.put.call_args
        assert call_args.kwargs["params"] == {"expiration_ttl": 600}
        assert call_args.kwargs["content"] == "test-value"

    @pytest.mark.asyncio
    async def test_set_failure(self, mock_env_vars, kv_client):
        """Test set operation with failure."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_http_client = AsyncMock()
        mock_http_client.put = AsyncMock(return_value=mock_response)
        kv_client._client = mock_http_client

        result = await kv_client.set("test-key", "test-value")

        assert result is False

    @pytest.mark.asyncio
    async def test_set_exception(self, mock_env_vars, kv_client):
        """Test set operation with exception."""
        mock_http_client = AsyncMock()
        mock_http_client.put = AsyncMock(side_effect=Exception("Network error"))
        kv_client._client = mock_http_client

        result = await kv_client.set("test-key", "test-value")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_success(self, mock_env_vars, kv_client):
        """Test successful delete operation."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_http_client = AsyncMock()
        mock_http_client.delete = AsyncMock(return_value=mock_response)
        kv_client._client = mock_http_client

        result = await kv_client.delete("test-key")

        assert result is True
        mock_http_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_failure(self, mock_env_vars, kv_client):
        """Test delete operation with failure."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_http_client = AsyncMock()
        mock_http_client.delete = AsyncMock(return_value=mock_response)
        kv_client._client = mock_http_client

        result = await kv_client.delete("nonexistent-key")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_exception(self, mock_env_vars, kv_client):
        """Test delete operation with exception."""
        mock_http_client = AsyncMock()
        mock_http_client.delete = AsyncMock(side_effect=Exception("Network error"))
        kv_client._client = mock_http_client

        result = await kv_client.delete("test-key")

        assert result is False

    @pytest.mark.asyncio
    async def test_close(self, mock_env_vars, kv_client):
        """Test closing the HTTP client."""
        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()
        kv_client._client = mock_http_client

        await kv_client.close()

        mock_http_client.aclose.assert_called_once()
        assert kv_client._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self, mock_env_vars, kv_client):
        """Test close when no client exists."""
        assert kv_client._client is None

        # Should not raise
        await kv_client.close()

        assert kv_client._client is None


class TestGetKVClient:
    """Tests for get_kv_client function."""

    def test_get_kv_client_returns_none_when_not_configured(self, mock_env_vars):
        """Test that get_kv_client returns None when not configured."""
        mock_settings = MagicMock()
        mock_settings.cloudflare_api_token = None
        mock_settings.cloudflare_account_id = None
        mock_settings.cloudflare_kv_namespace_id = None

        import src.services.cache as module
        module._kv_client = None

        with patch("src.services.cache.get_settings", return_value=mock_settings):
            result = module.get_kv_client()

        assert result is None

    def test_get_kv_client_returns_none_when_namespace_not_configured(self, mock_env_vars):
        """Test that get_kv_client returns None when namespace ID is missing."""
        mock_settings = MagicMock()
        mock_settings.cloudflare_api_token = "test-token"
        mock_settings.cloudflare_account_id = "test-account"
        mock_settings.cloudflare_kv_namespace_id = None

        import src.services.cache as module
        module._kv_client = None

        with patch("src.services.cache.get_settings", return_value=mock_settings):
            result = module.get_kv_client()

        assert result is None

    def test_get_kv_client_creates_client(self, mock_env_vars):
        """Test that get_kv_client creates a new client when configured."""
        mock_settings = MagicMock()
        mock_settings.cloudflare_api_token = "test-token"
        mock_settings.cloudflare_account_id = "test-account"
        mock_settings.cloudflare_kv_namespace_id = "test-namespace"

        import src.services.cache as module
        module._kv_client = None

        with patch("src.services.cache.get_settings", return_value=mock_settings):
            result = module.get_kv_client()

        assert result is not None
        from src.services.cache import CloudflareKVClient
        assert isinstance(result, CloudflareKVClient)

    def test_get_kv_client_returns_singleton(self, mock_env_vars):
        """Test that get_kv_client returns the same instance."""
        mock_settings = MagicMock()
        mock_settings.cloudflare_api_token = "test-token"
        mock_settings.cloudflare_account_id = "test-account"
        mock_settings.cloudflare_kv_namespace_id = "test-namespace"

        import src.services.cache as module
        module._kv_client = None

        with patch("src.services.cache.get_settings", return_value=mock_settings):
            client1 = module.get_kv_client()
            client2 = module.get_kv_client()

        assert client1 is client2

    def test_get_kv_client_handles_secret_value(self, mock_env_vars):
        """Test that get_kv_client handles SecretStr for api_token."""
        mock_secret = MagicMock()
        mock_secret.get_secret_value.return_value = "secret-token"

        mock_settings = MagicMock()
        mock_settings.cloudflare_api_token = mock_secret
        mock_settings.cloudflare_account_id = "test-account"
        mock_settings.cloudflare_kv_namespace_id = "test-namespace"

        import src.services.cache as module
        module._kv_client = None

        with patch("src.services.cache.get_settings", return_value=mock_settings):
            result = module.get_kv_client()

        assert result is not None
        assert result.api_token == "secret-token"


class TestCacheKeyGeneration:
    """Tests for _make_cache_key function."""

    def test_make_cache_key_with_args(self, mock_env_vars):
        """Test cache key generation with positional args."""
        from src.services.cache import _make_cache_key

        key = _make_cache_key("test", "arg1", "arg2")

        assert key.startswith("argus:test:")
        assert len(key.split(":")[-1]) == 16  # SHA256 hash truncated to 16 chars

    def test_make_cache_key_with_kwargs(self, mock_env_vars):
        """Test cache key generation with keyword args."""
        from src.services.cache import _make_cache_key

        key = _make_cache_key("test", key1="value1", key2="value2")

        assert key.startswith("argus:test:")
        assert len(key.split(":")[-1]) == 16

    def test_make_cache_key_deterministic(self, mock_env_vars):
        """Test that cache key generation is deterministic."""
        from src.services.cache import _make_cache_key

        key1 = _make_cache_key("test", "arg", key="value")
        key2 = _make_cache_key("test", "arg", key="value")

        assert key1 == key2

    def test_make_cache_key_different_args_different_keys(self, mock_env_vars):
        """Test that different args produce different keys."""
        from src.services.cache import _make_cache_key

        key1 = _make_cache_key("test", "arg1")
        key2 = _make_cache_key("test", "arg2")

        assert key1 != key2

    def test_make_cache_key_different_prefixes(self, mock_env_vars):
        """Test that different prefixes produce different keys."""
        from src.services.cache import _make_cache_key

        key1 = _make_cache_key("prefix1", "arg")
        key2 = _make_cache_key("prefix2", "arg")

        assert key1 != key2
        assert "prefix1" in key1
        assert "prefix2" in key2


class TestSerializationFunctions:
    """Tests for _serialize and _deserialize functions."""

    def test_serialize_dict(self, mock_env_vars):
        """Test serializing a dictionary."""
        from src.services.cache import _serialize

        data = {"key": "value", "number": 42}
        result = _serialize(data)

        assert isinstance(result, str)
        assert json.loads(result) == data

    def test_serialize_list(self, mock_env_vars):
        """Test serializing a list."""
        from src.services.cache import _serialize

        data = [1, 2, 3, "test"]
        result = _serialize(data)

        assert isinstance(result, str)
        assert json.loads(result) == data

    def test_serialize_with_default_str(self, mock_env_vars):
        """Test serializing objects that need str conversion."""
        from datetime import datetime

        from src.services.cache import _serialize

        data = {"timestamp": datetime(2024, 1, 15, 12, 0, 0)}
        result = _serialize(data)

        assert isinstance(result, str)
        assert "2024" in result

    def test_deserialize_json(self, mock_env_vars):
        """Test deserializing JSON string."""
        from src.services.cache import _deserialize

        json_str = '{"key": "value", "number": 42}'
        result = _deserialize(json_str)

        assert result == {"key": "value", "number": 42}

    def test_deserialize_list(self, mock_env_vars):
        """Test deserializing JSON list."""
        from src.services.cache import _deserialize

        json_str = '[1, 2, 3, "test"]'
        result = _deserialize(json_str)

        assert result == [1, 2, 3, "test"]


class TestCacheQualityScoreDecorator:
    """Tests for cache_quality_score decorator."""

    @pytest.fixture
    def mock_kv_client(self):
        """Create a mock KV client."""
        client = MagicMock()
        client.get = AsyncMock()
        client.set = AsyncMock()
        return client

    @pytest.fixture
    def mock_settings_cache_enabled(self):
        """Create mock settings with cache enabled."""
        settings = MagicMock()
        settings.cache_enabled = True
        settings.cache_ttl_quality_scores = 300
        return settings

    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_env_vars, mock_kv_client, mock_settings_cache_enabled):
        """Test cache decorator with cache hit."""
        mock_kv_client.get = AsyncMock(return_value='{"score": 95}')

        with patch("src.services.cache.get_settings", return_value=mock_settings_cache_enabled):
            with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
                from src.services.cache import cache_quality_score

                @cache_quality_score()
                async def get_score(project_id: str):
                    return {"score": 100}  # Should not be called on cache hit

                result = await get_score("proj-123")

        assert result == {"score": 95}

    @pytest.mark.asyncio
    async def test_cache_miss(self, mock_env_vars, mock_kv_client, mock_settings_cache_enabled):
        """Test cache decorator with cache miss."""
        mock_kv_client.get = AsyncMock(return_value=None)
        mock_kv_client.set = AsyncMock(return_value=True)

        with patch("src.services.cache.get_settings", return_value=mock_settings_cache_enabled):
            with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
                from src.services.cache import cache_quality_score

                @cache_quality_score()
                async def get_score(project_id: str):
                    return {"score": 100}

                result = await get_score("proj-123")

        assert result == {"score": 100}
        mock_kv_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_env_vars):
        """Test decorator when cache is disabled."""
        mock_settings = MagicMock()
        mock_settings.cache_enabled = False

        with patch("src.services.cache.get_settings", return_value=mock_settings):
            from src.services.cache import cache_quality_score

            @cache_quality_score()
            async def get_score(project_id: str):
                return {"score": 100}

            result = await get_score("proj-123")

        assert result == {"score": 100}

    @pytest.mark.asyncio
    async def test_kv_client_not_available(self, mock_env_vars, mock_settings_cache_enabled):
        """Test decorator when KV client is not available."""
        with patch("src.services.cache.get_settings", return_value=mock_settings_cache_enabled):
            with patch("src.services.cache.get_kv_client", return_value=None):
                from src.services.cache import cache_quality_score

                @cache_quality_score()
                async def get_score(project_id: str):
                    return {"score": 100}

                result = await get_score("proj-123")

        assert result == {"score": 100}

    @pytest.mark.asyncio
    async def test_custom_ttl(self, mock_env_vars, mock_kv_client, mock_settings_cache_enabled):
        """Test decorator with custom TTL."""
        mock_kv_client.get = AsyncMock(return_value=None)
        mock_kv_client.set = AsyncMock(return_value=True)

        with patch("src.services.cache.get_settings", return_value=mock_settings_cache_enabled):
            with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
                from src.services.cache import cache_quality_score

                @cache_quality_score(ttl_seconds=600)
                async def get_score(project_id: str):
                    return {"score": 100}

                await get_score("proj-123")

        call_args = mock_kv_client.set.call_args
        assert call_args[1]["ex"] == 600

    @pytest.mark.asyncio
    async def test_cache_read_error(self, mock_env_vars, mock_kv_client, mock_settings_cache_enabled):
        """Test decorator handles cache read errors gracefully."""
        mock_kv_client.get = AsyncMock(side_effect=Exception("Read error"))
        mock_kv_client.set = AsyncMock(return_value=True)

        with patch("src.services.cache.get_settings", return_value=mock_settings_cache_enabled):
            with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
                from src.services.cache import cache_quality_score

                @cache_quality_score()
                async def get_score(project_id: str):
                    return {"score": 100}

                result = await get_score("proj-123")

        assert result == {"score": 100}

    @pytest.mark.asyncio
    async def test_cache_write_error(self, mock_env_vars, mock_kv_client, mock_settings_cache_enabled):
        """Test decorator handles cache write errors gracefully."""
        mock_kv_client.get = AsyncMock(return_value=None)
        mock_kv_client.set = AsyncMock(side_effect=Exception("Write error"))

        with patch("src.services.cache.get_settings", return_value=mock_settings_cache_enabled):
            with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
                from src.services.cache import cache_quality_score

                @cache_quality_score()
                async def get_score(project_id: str):
                    return {"score": 100}

                result = await get_score("proj-123")

        # Should still return the result despite write error
        assert result == {"score": 100}


class TestCacheLLMResponseDecorator:
    """Tests for cache_llm_response decorator."""

    @pytest.fixture
    def mock_kv_client(self):
        """Create a mock KV client."""
        client = MagicMock()
        client.get = AsyncMock()
        client.set = AsyncMock()
        return client

    @pytest.fixture
    def mock_settings_cache_enabled(self):
        """Create mock settings with cache enabled."""
        settings = MagicMock()
        settings.cache_enabled = True
        settings.cache_ttl_llm_responses = 86400
        return settings

    @pytest.mark.asyncio
    async def test_llm_cache_hit(self, mock_env_vars, mock_kv_client, mock_settings_cache_enabled):
        """Test LLM cache decorator with cache hit."""
        mock_kv_client.get = AsyncMock(return_value='"cached response"')

        with patch("src.services.cache.get_settings", return_value=mock_settings_cache_enabled):
            with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
                from src.services.cache import cache_llm_response

                @cache_llm_response()
                async def analyze_code(prompt: str):
                    return "new response"

                result = await analyze_code("analyze this code")

        assert result == "cached response"

    @pytest.mark.asyncio
    async def test_llm_cache_miss(self, mock_env_vars, mock_kv_client, mock_settings_cache_enabled):
        """Test LLM cache decorator with cache miss."""
        mock_kv_client.get = AsyncMock(return_value=None)
        mock_kv_client.set = AsyncMock(return_value=True)

        with patch("src.services.cache.get_settings", return_value=mock_settings_cache_enabled):
            with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
                from src.services.cache import cache_llm_response

                @cache_llm_response()
                async def analyze_code(prompt: str):
                    return "new response"

                result = await analyze_code("analyze this code")

        assert result == "new response"
        mock_kv_client.set.assert_called_once()


class TestCacheHealingPatternDecorator:
    """Tests for cache_healing_pattern decorator."""

    @pytest.fixture
    def mock_kv_client(self):
        """Create a mock KV client."""
        client = MagicMock()
        client.get = AsyncMock()
        client.set = AsyncMock()
        return client

    @pytest.fixture
    def mock_settings_cache_enabled(self):
        """Create mock settings with cache enabled."""
        settings = MagicMock()
        settings.cache_enabled = True
        settings.cache_ttl_healing_patterns = 604800
        return settings

    @pytest.mark.asyncio
    async def test_healing_cache_hit(self, mock_env_vars, mock_kv_client, mock_settings_cache_enabled):
        """Test healing cache decorator with cache hit."""
        mock_kv_client.get = AsyncMock(return_value='"healed-selector"')

        with patch("src.services.cache.get_settings", return_value=mock_settings_cache_enabled):
            with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
                from src.services.cache import cache_healing_pattern

                @cache_healing_pattern()
                async def find_healed_selector(original: str, error_type: str):
                    return "new-selector"

                result = await find_healed_selector(".old-selector", "element_not_found")

        assert result == "healed-selector"

    @pytest.mark.asyncio
    async def test_healing_cache_miss_with_result(self, mock_env_vars, mock_kv_client, mock_settings_cache_enabled):
        """Test healing cache decorator caches non-None results."""
        mock_kv_client.get = AsyncMock(return_value=None)
        mock_kv_client.set = AsyncMock(return_value=True)

        with patch("src.services.cache.get_settings", return_value=mock_settings_cache_enabled):
            with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
                from src.services.cache import cache_healing_pattern

                @cache_healing_pattern()
                async def find_healed_selector(original: str, error_type: str):
                    return "new-selector"

                result = await find_healed_selector(".old-selector", "element_not_found")

        assert result == "new-selector"
        mock_kv_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_healing_cache_miss_with_none_result(self, mock_env_vars, mock_kv_client, mock_settings_cache_enabled):
        """Test healing cache decorator does not cache None results."""
        mock_kv_client.get = AsyncMock(return_value=None)
        mock_kv_client.set = AsyncMock(return_value=True)

        with patch("src.services.cache.get_settings", return_value=mock_settings_cache_enabled):
            with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
                from src.services.cache import cache_healing_pattern

                @cache_healing_pattern()
                async def find_healed_selector(original: str, error_type: str):
                    return None

                result = await find_healed_selector(".old-selector", "element_not_found")

        assert result is None
        mock_kv_client.set.assert_not_called()


class TestDirectCacheOperations:
    """Tests for direct cache operations (get_cached, set_cached, delete_cached)."""

    @pytest.fixture
    def mock_kv_client(self):
        """Create a mock KV client."""
        client = MagicMock()
        client.get = AsyncMock()
        client.set = AsyncMock()
        client.delete = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_get_cached_success(self, mock_env_vars, mock_kv_client):
        """Test get_cached with successful retrieval."""
        mock_kv_client.get = AsyncMock(return_value='{"data": "value"}')

        with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
            from src.services.cache import get_cached

            result = await get_cached("test-key")

        assert result == {"data": "value"}
        mock_kv_client.get.assert_called_once_with("argus:test-key")

    @pytest.mark.asyncio
    async def test_get_cached_not_found(self, mock_env_vars, mock_kv_client):
        """Test get_cached when key not found."""
        mock_kv_client.get = AsyncMock(return_value=None)

        with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
            from src.services.cache import get_cached

            result = await get_cached("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_no_kv_client(self, mock_env_vars):
        """Test get_cached when KV client is not available."""
        with patch("src.services.cache.get_kv_client", return_value=None):
            from src.services.cache import get_cached

            result = await get_cached("test-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_exception(self, mock_env_vars, mock_kv_client):
        """Test get_cached handles exceptions."""
        mock_kv_client.get = AsyncMock(side_effect=Exception("Error"))

        with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
            from src.services.cache import get_cached

            result = await get_cached("test-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_cached_success(self, mock_env_vars, mock_kv_client):
        """Test set_cached with successful write."""
        mock_kv_client.set = AsyncMock(return_value=True)

        with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
            from src.services.cache import set_cached

            result = await set_cached("test-key", {"data": "value"}, ttl_seconds=600)

        assert result is True
        mock_kv_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_cached_no_kv_client(self, mock_env_vars):
        """Test set_cached when KV client is not available."""
        with patch("src.services.cache.get_kv_client", return_value=None):
            from src.services.cache import set_cached

            result = await set_cached("test-key", {"data": "value"})

        assert result is False

    @pytest.mark.asyncio
    async def test_set_cached_exception(self, mock_env_vars, mock_kv_client):
        """Test set_cached handles exceptions."""
        mock_kv_client.set = AsyncMock(side_effect=Exception("Error"))

        with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
            from src.services.cache import set_cached

            result = await set_cached("test-key", {"data": "value"})

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_cached_success(self, mock_env_vars, mock_kv_client):
        """Test delete_cached with successful deletion."""
        mock_kv_client.delete = AsyncMock(return_value=True)

        with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
            from src.services.cache import delete_cached

            result = await delete_cached("test-key")

        assert result is True
        mock_kv_client.delete.assert_called_once_with("argus:test-key")

    @pytest.mark.asyncio
    async def test_delete_cached_no_kv_client(self, mock_env_vars):
        """Test delete_cached when KV client is not available."""
        with patch("src.services.cache.get_kv_client", return_value=None):
            from src.services.cache import delete_cached

            result = await delete_cached("test-key")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_cached_exception(self, mock_env_vars, mock_kv_client):
        """Test delete_cached handles exceptions."""
        mock_kv_client.delete = AsyncMock(side_effect=Exception("Error"))

        with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
            from src.services.cache import delete_cached

            result = await delete_cached("test-key")

        assert result is False


class TestCacheHealthCheck:
    """Tests for check_cache_health function."""

    @pytest.fixture
    def mock_kv_client(self):
        """Create a mock KV client."""
        client = MagicMock()
        client.get = AsyncMock()
        client.set = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_env_vars, mock_kv_client):
        """Test health check with successful ping-pong."""
        mock_kv_client.set = AsyncMock(return_value=True)
        mock_kv_client.get = AsyncMock(return_value="pong")

        mock_settings = MagicMock()
        mock_settings.cache_enabled = True

        with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
            with patch("src.services.cache.get_settings", return_value=mock_settings):
                from src.services.cache import check_cache_health

                result = await check_cache_health()

        assert result["healthy"] is True
        assert result["provider"] == "cloudflare_kv"
        assert result["enabled"] is True

    @pytest.mark.asyncio
    async def test_health_check_no_kv_client(self, mock_env_vars):
        """Test health check when KV client is not available."""
        mock_settings = MagicMock()
        mock_settings.cache_enabled = True

        with patch("src.services.cache.get_kv_client", return_value=None):
            with patch("src.services.cache.get_settings", return_value=mock_settings):
                from src.services.cache import check_cache_health

                result = await check_cache_health()

        assert result["healthy"] is False
        assert "not configured" in result["reason"]

    @pytest.mark.asyncio
    async def test_health_check_ping_pong_failure(self, mock_env_vars, mock_kv_client):
        """Test health check when ping-pong test fails."""
        mock_kv_client.set = AsyncMock(return_value=True)
        mock_kv_client.get = AsyncMock(return_value="wrong")

        mock_settings = MagicMock()
        mock_settings.cache_enabled = True

        with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
            with patch("src.services.cache.get_settings", return_value=mock_settings):
                from src.services.cache import check_cache_health

                result = await check_cache_health()

        assert result["healthy"] is False
        assert "Ping-pong test failed" in result["reason"]

    @pytest.mark.asyncio
    async def test_health_check_exception(self, mock_env_vars, mock_kv_client):
        """Test health check handles exceptions."""
        mock_kv_client.set = AsyncMock(side_effect=Exception("Connection error"))

        mock_settings = MagicMock()
        mock_settings.cache_enabled = True

        with patch("src.services.cache.get_kv_client", return_value=mock_kv_client):
            with patch("src.services.cache.get_settings", return_value=mock_settings):
                from src.services.cache import check_cache_health

                result = await check_cache_health()

        assert result["healthy"] is False
        assert "Connection error" in result["reason"]
