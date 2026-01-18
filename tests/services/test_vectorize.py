"""Tests for the Cloudflare Vectorize service."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services.vectorize import (
    CF_API_BASE,
    CloudflareVectorizeClient,
    get_vectorize_client,
    index_error_pattern,
    index_production_event,
    semantic_search_errors,
)


class TestCloudflareVectorizeClientInit:
    """Tests for CloudflareVectorizeClient initialization."""

    def test_init_with_parameters(self):
        """Test initialization with explicit parameters."""
        client = CloudflareVectorizeClient(
            account_id="test-account",
            index_name="test-index",
            api_token="test-token"
        )

        assert client.account_id == "test-account"
        assert client.index_name == "test-index"
        assert client.api_token == "test-token"
        assert client._client is None

    def test_base_url_format(self):
        """Test that base URL is correctly formatted."""
        client = CloudflareVectorizeClient(
            account_id="my-account",
            index_name="my-index",
            api_token="token"
        )

        expected_base = f"{CF_API_BASE}/my-account/vectorize/v2/indexes/my-index"
        assert client.base_url == expected_base

    def test_ai_url_format(self):
        """Test that AI URL is correctly formatted."""
        client = CloudflareVectorizeClient(
            account_id="my-account",
            index_name="my-index",
            api_token="token"
        )

        expected_ai = f"{CF_API_BASE}/my-account/ai/run/@cf/baai/bge-large-en-v1.5"
        assert client.ai_url == expected_ai

    def test_get_headers(self):
        """Test that headers are correctly formatted."""
        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="my-api-token"
        )

        headers = client._get_headers()

        assert headers["Authorization"] == "Bearer my-api-token"
        assert headers["Content-Type"] == "application/json"


class TestCloudflareVectorizeClientHttpClient:
    """Tests for HTTP client management."""

    @pytest.mark.asyncio
    async def test_get_client_creates_new(self):
        """Test that _get_client creates a new client."""
        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )

        http_client = await client._get_client()

        assert http_client is not None
        assert isinstance(http_client, httpx.AsyncClient)

        await client.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self):
        """Test that _get_client reuses existing client."""
        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )

        http_client1 = await client._get_client()
        http_client2 = await client._get_client()

        assert http_client1 is http_client2

        await client.close()

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing the HTTP client."""
        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )

        await client._get_client()
        assert client._client is not None

        await client.close()

        assert client._client is None


class TestCloudflareVectorizeClientGenerateEmbedding:
    """Tests for generate_embedding method."""

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, mock_httpx_client):
        """Test successful embedding generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "result": {
                "data": [[0.1, 0.2, 0.3, 0.4, 0.5]]  # Sample embedding
            }
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        embedding = await client.generate_embedding("test text")

        assert embedding is not None
        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

        # Verify the request was made correctly
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert call_args[1]["json"] == {"text": ["test text"]}

    @pytest.mark.asyncio
    async def test_generate_embedding_api_error(self, mock_httpx_client):
        """Test embedding generation with API error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        embedding = await client.generate_embedding("test text")

        assert embedding is None

    @pytest.mark.asyncio
    async def test_generate_embedding_missing_data(self, mock_httpx_client):
        """Test embedding generation with missing data in response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "result": {}  # Missing data
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        embedding = await client.generate_embedding("test text")

        assert embedding is None

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_data(self, mock_httpx_client):
        """Test embedding generation with empty data array."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "result": {"data": []}  # Empty data
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        embedding = await client.generate_embedding("test text")

        assert embedding is None

    @pytest.mark.asyncio
    async def test_generate_embedding_exception(self, mock_httpx_client):
        """Test embedding generation with exception."""
        mock_httpx_client.post = AsyncMock(side_effect=Exception("Network error"))

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        embedding = await client.generate_embedding("test text")

        assert embedding is None

    @pytest.mark.asyncio
    async def test_generate_embedding_not_success(self, mock_httpx_client):
        """Test embedding generation when success is False."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "errors": ["Rate limit exceeded"]
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        embedding = await client.generate_embedding("test text")

        assert embedding is None


class TestCloudflareVectorizeClientQuery:
    """Tests for query method."""

    @pytest.mark.asyncio
    async def test_query_success(self, mock_httpx_client):
        """Test successful vector query."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "result": {
                "matches": [
                    {"id": "error-1", "score": 0.95, "metadata": {"title": "Error 1"}},
                    {"id": "error-2", "score": 0.85, "metadata": {"title": "Error 2"}}
                ]
            }
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        vector = [0.1] * 1024  # 1024-dim vector
        results = await client.query(vector, top_k=5)

        assert len(results) == 2
        assert results[0]["id"] == "error-1"
        assert results[0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_query_with_filter(self, mock_httpx_client):
        """Test vector query with filter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "result": {"matches": []}
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        vector = [0.1] * 1024
        await client.query(vector, top_k=5, filter={"severity": "error"})

        call_args = mock_httpx_client.post.call_args
        assert "filter" in call_args[1]["json"]
        assert call_args[1]["json"]["filter"] == {"severity": "error"}

    @pytest.mark.asyncio
    async def test_query_api_error(self, mock_httpx_client):
        """Test vector query with API error."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        vector = [0.1] * 1024
        results = await client.query(vector)

        assert results == []

    @pytest.mark.asyncio
    async def test_query_not_success(self, mock_httpx_client):
        """Test vector query when success is False."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "errors": ["Query failed"]
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        vector = [0.1] * 1024
        results = await client.query(vector)

        assert results == []

    @pytest.mark.asyncio
    async def test_query_exception(self, mock_httpx_client):
        """Test vector query with exception."""
        mock_httpx_client.post = AsyncMock(side_effect=Exception("Network error"))

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        vector = [0.1] * 1024
        results = await client.query(vector)

        assert results == []


class TestCloudflareVectorizeClientInsert:
    """Tests for insert method."""

    @pytest.mark.asyncio
    async def test_insert_success(self, mock_httpx_client):
        """Test successful vector insert."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        vectors = [
            {"id": "vec-1", "values": [0.1] * 1024, "metadata": {"title": "Test"}}
        ]
        result = await client.insert(vectors)

        assert result is True

        call_args = mock_httpx_client.post.call_args
        assert "/insert" in call_args[0][0]
        assert call_args[1]["json"]["vectors"] == vectors

    @pytest.mark.asyncio
    async def test_insert_api_error(self, mock_httpx_client):
        """Test vector insert with API error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        vectors = [{"id": "vec-1", "values": [0.1] * 1024}]
        result = await client.insert(vectors)

        assert result is False

    @pytest.mark.asyncio
    async def test_insert_not_success(self, mock_httpx_client):
        """Test vector insert when success is False."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": False}
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        vectors = [{"id": "vec-1", "values": [0.1] * 1024}]
        result = await client.insert(vectors)

        assert result is False

    @pytest.mark.asyncio
    async def test_insert_exception(self, mock_httpx_client):
        """Test vector insert with exception."""
        mock_httpx_client.post = AsyncMock(side_effect=Exception("Network error"))

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        vectors = [{"id": "vec-1", "values": [0.1] * 1024}]
        result = await client.insert(vectors)

        assert result is False


class TestCloudflareVectorizeClientUpsert:
    """Tests for upsert method."""

    @pytest.mark.asyncio
    async def test_upsert_success(self, mock_httpx_client):
        """Test successful vector upsert."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        vectors = [
            {"id": "vec-1", "values": [0.1] * 1024, "metadata": {"updated": True}}
        ]
        result = await client.upsert(vectors)

        assert result is True

        call_args = mock_httpx_client.post.call_args
        assert "/upsert" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_upsert_api_error(self, mock_httpx_client):
        """Test vector upsert with API error."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        vectors = [{"id": "vec-1", "values": [0.1] * 1024}]
        result = await client.upsert(vectors)

        assert result is False

    @pytest.mark.asyncio
    async def test_upsert_exception(self, mock_httpx_client):
        """Test vector upsert with exception."""
        mock_httpx_client.post = AsyncMock(side_effect=Exception("Network error"))

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        vectors = [{"id": "vec-1", "values": [0.1] * 1024}]
        result = await client.upsert(vectors)

        assert result is False


class TestCloudflareVectorizeClientDeleteByIds:
    """Tests for delete_by_ids method."""

    @pytest.mark.asyncio
    async def test_delete_by_ids_success(self, mock_httpx_client):
        """Test successful delete by IDs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        result = await client.delete_by_ids(["id-1", "id-2", "id-3"])

        assert result is True

        call_args = mock_httpx_client.post.call_args
        assert "/delete-by-ids" in call_args[0][0]
        assert call_args[1]["json"]["ids"] == ["id-1", "id-2", "id-3"]

    @pytest.mark.asyncio
    async def test_delete_by_ids_api_error(self, mock_httpx_client):
        """Test delete by IDs with API error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        result = await client.delete_by_ids(["nonexistent"])

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_by_ids_exception(self, mock_httpx_client):
        """Test delete by IDs with exception."""
        mock_httpx_client.post = AsyncMock(side_effect=Exception("Network error"))

        client = CloudflareVectorizeClient(
            account_id="test",
            index_name="test",
            api_token="token"
        )
        client._client = mock_httpx_client

        result = await client.delete_by_ids(["id-1"])

        assert result is False


class TestGetVectorizeClient:
    """Tests for get_vectorize_client factory function."""

    def test_get_vectorize_client_returns_cached(self, mock_env_vars):
        """Test that get_vectorize_client returns cached instance."""
        import src.services.vectorize as module

        mock_client = MagicMock()
        module._vectorize_client = mock_client

        result = get_vectorize_client()

        assert result is mock_client

        # Reset for other tests
        module._vectorize_client = None

    def test_get_vectorize_client_not_configured_missing_token(self, mock_env_vars):
        """Test get_vectorize_client returns None when token is missing."""
        import src.services.vectorize as module

        module._vectorize_client = None

        with patch("src.services.vectorize.get_settings") as mock_settings:
            mock_settings.return_value.cloudflare_api_token = None
            mock_settings.return_value.cloudflare_account_id = "test-account"
            mock_settings.return_value.cloudflare_vectorize_index = "test-index"

            result = get_vectorize_client()

            assert result is None

    def test_get_vectorize_client_not_configured_missing_account(self, mock_env_vars):
        """Test get_vectorize_client returns None when account is missing."""
        import src.services.vectorize as module

        module._vectorize_client = None

        with patch("src.services.vectorize.get_settings") as mock_settings:
            mock_settings.return_value.cloudflare_api_token = "test-token"
            mock_settings.return_value.cloudflare_account_id = None
            mock_settings.return_value.cloudflare_vectorize_index = "test-index"

            result = get_vectorize_client()

            assert result is None

    def test_get_vectorize_client_not_configured_missing_index(self, mock_env_vars):
        """Test get_vectorize_client returns None when index is missing."""
        import src.services.vectorize as module

        module._vectorize_client = None

        with patch("src.services.vectorize.get_settings") as mock_settings:
            mock_settings.return_value.cloudflare_api_token = "test-token"
            mock_settings.return_value.cloudflare_account_id = "test-account"
            mock_settings.return_value.cloudflare_vectorize_index = None

            result = get_vectorize_client()

            assert result is None

    def test_get_vectorize_client_success(self, mock_env_vars):
        """Test successful creation of vectorize client."""
        import src.services.vectorize as module

        module._vectorize_client = None

        with patch("src.services.vectorize.get_settings") as mock_settings:
            mock_settings.return_value.cloudflare_api_token = "test-token"
            mock_settings.return_value.cloudflare_account_id = "test-account"
            mock_settings.return_value.cloudflare_vectorize_index = "test-index"

            result = get_vectorize_client()

            assert result is not None
            assert isinstance(result, CloudflareVectorizeClient)

        # Reset for other tests
        module._vectorize_client = None

    def test_get_vectorize_client_with_secret_value(self, mock_env_vars):
        """Test get_vectorize_client with SecretStr token."""
        import src.services.vectorize as module

        module._vectorize_client = None

        with patch("src.services.vectorize.get_settings") as mock_settings:
            mock_token = MagicMock()
            mock_token.get_secret_value.return_value = "secret-token"

            mock_settings.return_value.cloudflare_api_token = mock_token
            mock_settings.return_value.cloudflare_account_id = "test-account"
            mock_settings.return_value.cloudflare_vectorize_index = "test-index"

            result = get_vectorize_client()

            assert result is not None
            assert result.api_token == "secret-token"

        module._vectorize_client = None

    def test_get_vectorize_client_creation_error(self, mock_env_vars):
        """Test get_vectorize_client handles creation errors."""
        import src.services.vectorize as module

        module._vectorize_client = None

        with patch("src.services.vectorize.get_settings") as mock_settings:
            mock_settings.return_value.cloudflare_api_token = "test-token"
            mock_settings.return_value.cloudflare_account_id = "test-account"
            mock_settings.return_value.cloudflare_vectorize_index = "test-index"

            with patch("src.services.vectorize.CloudflareVectorizeClient", side_effect=Exception("Init error")):
                result = get_vectorize_client()

                assert result is None


class TestSemanticSearchErrors:
    """Tests for semantic_search_errors function."""

    @pytest.mark.asyncio
    async def test_semantic_search_no_client(self, mock_env_vars):
        """Test semantic search when vectorize client is not available."""
        with patch("src.services.vectorize.get_vectorize_client", return_value=None):
            results = await semantic_search_errors("test error")

            assert results == []

    @pytest.mark.asyncio
    async def test_semantic_search_embedding_fails(self, mock_env_vars):
        """Test semantic search when embedding generation fails."""
        mock_client = AsyncMock()
        mock_client.generate_embedding = AsyncMock(return_value=None)

        with patch("src.services.vectorize.get_vectorize_client", return_value=mock_client):
            results = await semantic_search_errors("test error")

            assert results == []

    @pytest.mark.asyncio
    async def test_semantic_search_success(self, mock_env_vars):
        """Test successful semantic search."""
        mock_client = AsyncMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.query = AsyncMock(return_value=[
            {"id": "err-1", "score": 0.9, "metadata": {"title": "Error 1"}},
            {"id": "err-2", "score": 0.8, "metadata": {"title": "Error 2"}},
            {"id": "err-3", "score": 0.5, "metadata": {"title": "Error 3"}}  # Below min_score
        ])

        with patch("src.services.vectorize.get_vectorize_client", return_value=mock_client):
            results = await semantic_search_errors("test error", limit=5, min_score=0.7)

            assert len(results) == 2  # Only scores >= 0.7
            assert results[0]["id"] == "err-1"
            assert results[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_semantic_search_respects_limit(self, mock_env_vars):
        """Test that semantic search respects the limit parameter."""
        mock_client = AsyncMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.query = AsyncMock(return_value=[
            {"id": "err-1", "score": 0.9, "metadata": {}},
            {"id": "err-2", "score": 0.85, "metadata": {}},
            {"id": "err-3", "score": 0.8, "metadata": {}},
        ])

        with patch("src.services.vectorize.get_vectorize_client", return_value=mock_client):
            results = await semantic_search_errors("test error", limit=2, min_score=0.7)

            assert len(results) == 2


class TestIndexErrorPattern:
    """Tests for index_error_pattern function."""

    @pytest.mark.asyncio
    async def test_index_error_pattern_no_client(self, mock_env_vars):
        """Test indexing when vectorize client is not available."""
        with patch("src.services.vectorize.get_vectorize_client", return_value=None):
            result = await index_error_pattern("error-1", "Test error")

            assert result is False

    @pytest.mark.asyncio
    async def test_index_error_pattern_embedding_fails(self, mock_env_vars):
        """Test indexing when embedding generation fails."""
        mock_client = AsyncMock()
        mock_client.generate_embedding = AsyncMock(return_value=None)

        with patch("src.services.vectorize.get_vectorize_client", return_value=mock_client):
            result = await index_error_pattern("error-1", "Test error")

            assert result is False

    @pytest.mark.asyncio
    async def test_index_error_pattern_success(self, mock_env_vars):
        """Test successful error pattern indexing."""
        mock_client = AsyncMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.upsert = AsyncMock(return_value=True)

        with patch("src.services.vectorize.get_vectorize_client", return_value=mock_client):
            result = await index_error_pattern(
                "error-1",
                "Test error message",
                metadata={"severity": "high"}
            )

            assert result is True

            mock_client.upsert.assert_called_once()
            call_args = mock_client.upsert.call_args[0][0]
            assert call_args[0]["id"] == "error-1"
            assert call_args[0]["metadata"] == {"severity": "high"}

    @pytest.mark.asyncio
    async def test_index_error_pattern_no_metadata(self, mock_env_vars):
        """Test indexing with no metadata."""
        mock_client = AsyncMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.upsert = AsyncMock(return_value=True)

        with patch("src.services.vectorize.get_vectorize_client", return_value=mock_client):
            result = await index_error_pattern("error-1", "Test error")

            assert result is True

            call_args = mock_client.upsert.call_args[0][0]
            assert call_args[0]["metadata"] == {}


class TestIndexProductionEvent:
    """Tests for index_production_event function."""

    @pytest.mark.asyncio
    async def test_index_production_event_no_id(self, mock_env_vars):
        """Test indexing event without ID."""
        result = await index_production_event({})

        assert result is False

    @pytest.mark.asyncio
    async def test_index_production_event_no_text(self, mock_env_vars):
        """Test indexing event with no searchable text."""
        result = await index_production_event({"id": "event-1"})

        assert result is False

    @pytest.mark.asyncio
    async def test_index_production_event_success(self, mock_env_vars):
        """Test successful production event indexing."""
        mock_client = AsyncMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.upsert = AsyncMock(return_value=True)

        with patch("src.services.vectorize.get_vectorize_client", return_value=mock_client):
            event = {
                "id": "event-1",
                "title": "Application Error",
                "message": "Failed to connect to database",
                "component": "api-server",
                "severity": "critical",
                "source": "production",
                "url": "https://example.com/errors/1",
                "fingerprint": "abc123",
                "project_id": "proj-1"
            }

            result = await index_production_event(event)

            assert result is True

    @pytest.mark.asyncio
    async def test_index_production_event_with_stack_trace(self, mock_env_vars):
        """Test indexing event with stack trace."""
        mock_client = AsyncMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.upsert = AsyncMock(return_value=True)

        with patch("src.services.vectorize.get_vectorize_client", return_value=mock_client):
            event = {
                "id": "event-1",
                "title": "Error",
                "message": "Something failed",
                "stack_trace": "Error: Something failed\n    at Function.run (/app/index.js:10:5)\n" * 100  # Long stack trace
            }

            result = await index_production_event(event)

            assert result is True

    @pytest.mark.asyncio
    async def test_index_production_event_handles_none_values(self, mock_env_vars):
        """Test indexing event handles None values gracefully."""
        mock_client = AsyncMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.upsert = AsyncMock(return_value=True)

        with patch("src.services.vectorize.get_vectorize_client", return_value=mock_client):
            event = {
                "id": "event-1",
                "title": "Error",
                "message": None,
                "component": None,
                "url": None,
                "severity": None,
                "source": None,
                "fingerprint": None,
                "project_id": None
            }

            result = await index_production_event(event)

            assert result is True

    @pytest.mark.asyncio
    async def test_index_production_event_no_vectorize(self, mock_env_vars):
        """Test indexing when vectorize is not available."""
        with patch("src.services.vectorize.get_vectorize_client", return_value=None):
            event = {
                "id": "event-1",
                "title": "Error",
                "message": "Test"
            }

            result = await index_production_event(event)

            assert result is False

    @pytest.mark.asyncio
    async def test_index_production_event_embedding_fails(self, mock_env_vars):
        """Test indexing when embedding fails."""
        mock_client = AsyncMock()
        mock_client.generate_embedding = AsyncMock(return_value=None)

        with patch("src.services.vectorize.get_vectorize_client", return_value=mock_client):
            event = {
                "id": "event-1",
                "title": "Error",
                "message": "Test"
            }

            result = await index_production_event(event)

            assert result is False

    @pytest.mark.asyncio
    async def test_index_production_event_truncates_long_fields(self, mock_env_vars):
        """Test that long fields are truncated in metadata."""
        mock_client = AsyncMock()
        mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
        mock_client.upsert = AsyncMock(return_value=True)

        with patch("src.services.vectorize.get_vectorize_client", return_value=mock_client):
            long_title = "A" * 500  # Longer than 200 char limit
            long_message = "B" * 1000  # Longer than 500 char limit

            event = {
                "id": "event-1",
                "title": long_title,
                "message": long_message
            }

            result = await index_production_event(event)

            assert result is True

            call_args = mock_client.upsert.call_args[0][0]
            metadata = call_args[0]["metadata"]
            assert len(metadata["title"]) == 200
            assert len(metadata["message"]) == 500
