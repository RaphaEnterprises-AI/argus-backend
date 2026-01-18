"""Tests for the Cloudflare Storage service.

This module tests:
- CloudflareConfig from environment and settings
- R2Storage for screenshots and artifacts
- VectorizeMemory for failure patterns
- D1Database for test history
- KVCache for session data
- AIGateway for LLM routing
- CloudflareClient unified interface
"""

import pytest
import base64
import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


class TestCloudflareConfig:
    """Tests for CloudflareConfig dataclass."""

    def test_config_creation(self, mock_env_vars):
        """Test creating a CloudflareConfig instance."""
        from src.services.cloudflare_storage import CloudflareConfig

        config = CloudflareConfig(
            account_id="acc-123",
            api_token="token-456",
            r2_bucket="my-bucket",
            vectorize_index="my-index",
            d1_database_id="db-789",
            kv_namespace_id="kv-012",
            ai_gateway_id="gateway-345",
        )

        assert config.account_id == "acc-123"
        assert config.api_token == "token-456"
        assert config.r2_bucket == "my-bucket"
        assert config.vectorize_index == "my-index"
        assert config.d1_database_id == "db-789"
        assert config.kv_namespace_id == "kv-012"
        assert config.ai_gateway_id == "gateway-345"

    def test_config_defaults(self, mock_env_vars):
        """Test CloudflareConfig default values."""
        from src.services.cloudflare_storage import CloudflareConfig

        config = CloudflareConfig(
            account_id="acc-123",
            api_token="token-456",
        )

        assert config.r2_bucket == "argus-artifacts"
        assert config.vectorize_index == "argus-patterns"
        assert config.d1_database_id == ""
        assert config.kv_namespace_id == ""
        assert config.ai_gateway_id == "argus-gateway"

    def test_config_from_env(self, mock_env_vars, monkeypatch):
        """Test CloudflareConfig.from_env() method."""
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "env-account")
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "env-token")
        monkeypatch.setenv("CLOUDFLARE_R2_BUCKET", "env-bucket")
        monkeypatch.setenv("CLOUDFLARE_VECTORIZE_INDEX", "env-index")

        from src.services.cloudflare_storage import CloudflareConfig

        config = CloudflareConfig.from_env()

        assert config.account_id == "env-account"
        assert config.api_token == "env-token"
        assert config.r2_bucket == "env-bucket"
        assert config.vectorize_index == "env-index"

    def test_config_from_env_with_defaults(self, mock_env_vars, monkeypatch):
        """Test CloudflareConfig.from_env() uses defaults when env vars not set."""
        # Clear relevant env vars
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)

        from src.services.cloudflare_storage import CloudflareConfig

        config = CloudflareConfig.from_env()

        assert config.account_id == ""
        assert config.api_token == ""
        assert config.r2_bucket == "argus-artifacts"

    def test_config_from_settings(self, mock_env_vars):
        """Test CloudflareConfig.from_settings() method."""
        mock_secret = MagicMock()
        mock_secret.get_secret_value.return_value = "settings-token"

        mock_settings = MagicMock()
        mock_settings.cloudflare_account_id = "settings-account"
        mock_settings.cloudflare_api_token = mock_secret
        mock_settings.cloudflare_r2_bucket = "settings-bucket"
        mock_settings.cloudflare_vectorize_index = "settings-index"
        mock_settings.cloudflare_d1_database_id = "settings-db"
        mock_settings.cloudflare_kv_namespace_id = "settings-kv"
        mock_settings.cloudflare_gateway_id = "settings-gateway"

        with patch("src.config.get_settings", return_value=mock_settings):
            from src.services.cloudflare_storage import CloudflareConfig

            config = CloudflareConfig.from_settings()

        assert config.account_id == "settings-account"
        assert config.api_token == "settings-token"
        assert config.r2_bucket == "settings-bucket"

    def test_config_from_settings_none_values(self, mock_env_vars):
        """Test CloudflareConfig.from_settings() handles None values."""
        mock_settings = MagicMock()
        mock_settings.cloudflare_account_id = None
        mock_settings.cloudflare_api_token = None
        mock_settings.cloudflare_r2_bucket = None
        mock_settings.cloudflare_vectorize_index = None
        mock_settings.cloudflare_d1_database_id = None
        mock_settings.cloudflare_kv_namespace_id = None
        mock_settings.cloudflare_gateway_id = None

        with patch("src.config.get_settings", return_value=mock_settings):
            from src.services.cloudflare_storage import CloudflareConfig

            config = CloudflareConfig.from_settings()

        assert config.account_id == ""
        assert config.api_token == ""
        assert config.r2_bucket == "argus-artifacts"


class TestR2Storage:
    """Tests for R2Storage class."""

    @pytest.fixture
    def r2_config(self):
        """Create a test config for R2."""
        from src.services.cloudflare_storage import CloudflareConfig

        return CloudflareConfig(
            account_id="test-account",
            api_token="test-token",
            r2_bucket="test-bucket",
        )

    @pytest.fixture
    def r2_storage(self, r2_config):
        """Create R2Storage instance."""
        from src.services.cloudflare_storage import R2Storage

        return R2Storage(r2_config)

    def test_r2_initialization(self, mock_env_vars, r2_config):
        """Test R2Storage initialization."""
        from src.services.cloudflare_storage import R2Storage

        storage = R2Storage(r2_config)

        assert storage.config == r2_config
        assert "test-account" in storage.base_url
        assert "test-bucket" in storage.base_url
        assert storage.headers["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_store_screenshot_success(self, mock_env_vars, r2_storage):
        """Test successful screenshot storage."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        # Create a valid base64 PNG
        test_data = b"fake png data"
        base64_data = base64.b64encode(test_data).decode()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await r2_storage.store_screenshot(
                base64_data,
                metadata={"step": 1, "test_id": "test-001"},
            )

        assert result["type"] == "screenshot"
        assert result["storage"] == "r2"
        assert "artifact_id" in result
        assert "key" in result
        assert "url" in result
        assert result["metadata"]["step"] == 1

    @pytest.mark.asyncio
    async def test_store_screenshot_with_data_url(self, mock_env_vars, r2_storage):
        """Test storing screenshot with data URL prefix."""
        mock_response = MagicMock()
        mock_response.status_code = 201

        test_data = b"fake png data"
        base64_data = f"data:image/png;base64,{base64.b64encode(test_data).decode()}"

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await r2_storage.store_screenshot(base64_data)

        assert result["storage"] == "r2"
        assert result["type"] == "screenshot"

    @pytest.mark.asyncio
    async def test_store_screenshot_failure(self, mock_env_vars, r2_storage):
        """Test screenshot storage failure falls back to inline."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        test_data = b"fake png data"
        base64_data = base64.b64encode(test_data).decode()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await r2_storage.store_screenshot(base64_data)

        assert result["storage"] == "inline"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_store_screenshot_exception(self, mock_env_vars, r2_storage):
        """Test screenshot storage handles exceptions."""
        test_data = b"fake png data"
        base64_data = base64.b64encode(test_data).decode()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(side_effect=Exception("Network error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await r2_storage.store_screenshot(base64_data)

        assert result["storage"] == "inline"
        assert "error" in result
        assert "Network error" in result["error"]

    @pytest.mark.asyncio
    async def test_get_screenshot_success(self, mock_env_vars, r2_storage):
        """Test successful screenshot retrieval."""
        test_content = b"png content"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = test_content

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await r2_storage.get_screenshot("screenshot_abc123_20240115")

        assert result == base64.b64encode(test_content).decode()

    @pytest.mark.asyncio
    async def test_get_screenshot_not_found(self, mock_env_vars, r2_storage):
        """Test screenshot retrieval when not found."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await r2_storage.get_screenshot("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_screenshot_exception(self, mock_env_vars, r2_storage):
        """Test screenshot retrieval handles exceptions."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Network error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await r2_storage.get_screenshot("test-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_store_test_result_extracts_screenshots(self, mock_env_vars, r2_storage):
        """Test that store_test_result extracts and stores screenshots."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        # Create a large base64 string (>1000 chars)
        large_screenshot = base64.b64encode(b"x" * 1000).decode()

        test_result = {
            "success": True,
            "screenshot": large_screenshot,
            "other_data": "preserved",
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await r2_storage.store_test_result(test_result, "test-123")

        assert result["_r2_stored"] is True
        assert "_artifact_refs" in result
        assert result["other_data"] == "preserved"
        # Screenshot should be replaced with artifact ID
        assert result["screenshot"] != large_screenshot

    @pytest.mark.asyncio
    async def test_store_test_result_preserves_small_screenshots(self, mock_env_vars, r2_storage):
        """Test that small screenshots are preserved inline."""
        test_result = {
            "success": True,
            "screenshot": "small",  # Less than 1000 chars
        }

        result = await r2_storage.store_test_result(test_result, "test-123")

        assert result["screenshot"] == "small"
        assert result["_r2_stored"] is True


class TestVectorizeMemory:
    """Tests for VectorizeMemory class."""

    @pytest.fixture
    def vectorize_config(self):
        """Create a test config for Vectorize."""
        from src.services.cloudflare_storage import CloudflareConfig

        return CloudflareConfig(
            account_id="test-account",
            api_token="test-token",
            vectorize_index="test-index",
        )

    @pytest.fixture
    def vectorize_memory(self, vectorize_config):
        """Create VectorizeMemory instance."""
        from src.services.cloudflare_storage import VectorizeMemory

        return VectorizeMemory(vectorize_config)

    def test_vectorize_initialization(self, mock_env_vars, vectorize_config):
        """Test VectorizeMemory initialization."""
        from src.services.cloudflare_storage import VectorizeMemory

        memory = VectorizeMemory(vectorize_config)

        assert memory.config == vectorize_config
        assert "test-account" in memory.base_url
        assert "test-index" in memory.base_url
        assert memory.headers["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_get_embedding_success(self, mock_env_vars, vectorize_memory):
        """Test successful embedding generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"data": [[0.1, 0.2, 0.3] * 341 + [0.1]]}  # 1024 dimensions
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await vectorize_memory._get_embedding("test text")

        assert isinstance(result, list)
        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_get_embedding_failure(self, mock_env_vars, vectorize_memory):
        """Test embedding generation failure."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            with pytest.raises(Exception) as exc_info:
                await vectorize_memory._get_embedding("test text")

            assert "Embedding generation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_store_failure_pattern_success(self, mock_env_vars, vectorize_memory):
        """Test storing a failure pattern successfully."""
        embedding_response = MagicMock()
        embedding_response.status_code = 200
        embedding_response.json.return_value = {
            "result": {"data": [[0.1] * 1024]}
        }

        insert_response = MagicMock()
        insert_response.status_code = 200

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[embedding_response, insert_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            pattern_id = await vectorize_memory.store_failure_pattern(
                error_message="Element not found",
                failed_selector=".login-btn",
                healed_selector="#login-button",
                context={"url": "https://example.com/login"},
            )

        assert pattern_id.startswith("pattern_")

    @pytest.mark.asyncio
    async def test_store_failure_pattern_without_healing(self, mock_env_vars, vectorize_memory):
        """Test storing a failure pattern without a healed selector."""
        embedding_response = MagicMock()
        embedding_response.status_code = 200
        embedding_response.json.return_value = {
            "result": {"data": [[0.1] * 1024]}
        }

        insert_response = MagicMock()
        insert_response.status_code = 201

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[embedding_response, insert_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            pattern_id = await vectorize_memory.store_failure_pattern(
                error_message="Timeout",
                failed_selector=".submit",
            )

        assert pattern_id.startswith("pattern_")

    @pytest.mark.asyncio
    async def test_store_failure_pattern_insert_failure(self, mock_env_vars, vectorize_memory):
        """Test storing a failure pattern when insert fails."""
        embedding_response = MagicMock()
        embedding_response.status_code = 200
        embedding_response.json.return_value = {
            "result": {"data": [[0.1] * 1024]}
        }

        insert_response = MagicMock()
        insert_response.status_code = 500

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[embedding_response, insert_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            with pytest.raises(Exception) as exc_info:
                await vectorize_memory.store_failure_pattern(
                    error_message="Error",
                    failed_selector=".btn",
                )

            assert "Vectorize insert failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_find_similar_failures_success(self, mock_env_vars, vectorize_memory):
        """Test finding similar failures successfully."""
        embedding_response = MagicMock()
        embedding_response.status_code = 200
        embedding_response.json.return_value = {
            "result": {"data": [[0.1] * 1024]}
        }

        query_response = MagicMock()
        query_response.status_code = 200
        query_response.json.return_value = {
            "result": {
                "matches": [
                    {
                        "id": "pattern_abc",
                        "score": 0.9,
                        "metadata": {
                            "error_message": "Similar error",
                            "failed_selector": ".old-btn",
                            "healed_selector": "#new-btn",
                            "success_count": 5,
                        },
                    }
                ]
            }
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[embedding_response, query_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            suggestions = await vectorize_memory.find_similar_failures(
                error_message="Element not found",
                selector=".btn",
            )

        assert len(suggestions) == 1
        assert suggestions[0]["pattern_id"] == "pattern_abc"
        assert suggestions[0]["score"] == 0.9
        assert suggestions[0]["healed_selector"] == "#new-btn"

    @pytest.mark.asyncio
    async def test_find_similar_failures_filters_low_scores(self, mock_env_vars, vectorize_memory):
        """Test that low-score matches are filtered out."""
        embedding_response = MagicMock()
        embedding_response.status_code = 200
        embedding_response.json.return_value = {
            "result": {"data": [[0.1] * 1024]}
        }

        query_response = MagicMock()
        query_response.status_code = 200
        query_response.json.return_value = {
            "result": {
                "matches": [
                    {"id": "pattern_low", "score": 0.5, "metadata": {}},  # Below threshold
                    {"id": "pattern_high", "score": 0.8, "metadata": {
                        "healed_selector": "#btn",
                        "success_count": 3,
                    }},
                ]
            }
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[embedding_response, query_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            suggestions = await vectorize_memory.find_similar_failures(
                error_message="Error",
                selector=".btn",
                min_score=0.7,
            )

        assert len(suggestions) == 1
        assert suggestions[0]["pattern_id"] == "pattern_high"

    @pytest.mark.asyncio
    async def test_find_similar_failures_query_failure(self, mock_env_vars, vectorize_memory):
        """Test finding similar failures when query fails."""
        embedding_response = MagicMock()
        embedding_response.status_code = 200
        embedding_response.json.return_value = {
            "result": {"data": [[0.1] * 1024]}
        }

        query_response = MagicMock()
        query_response.status_code = 500

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[embedding_response, query_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            suggestions = await vectorize_memory.find_similar_failures(
                error_message="Error",
                selector=".btn",
            )

        assert suggestions == []

    @pytest.mark.asyncio
    async def test_record_healing_success(self, mock_env_vars, vectorize_memory):
        """Test recording a healing success."""
        # This method currently just logs, so we verify it doesn't raise
        await vectorize_memory.record_healing_success("pattern_abc")


class TestD1Database:
    """Tests for D1Database class."""

    @pytest.fixture
    def d1_config(self):
        """Create a test config for D1."""
        from src.services.cloudflare_storage import CloudflareConfig

        return CloudflareConfig(
            account_id="test-account",
            api_token="test-token",
            d1_database_id="test-db",
        )

    @pytest.fixture
    def d1_database(self, d1_config):
        """Create D1Database instance."""
        from src.services.cloudflare_storage import D1Database

        return D1Database(d1_config)

    def test_d1_initialization(self, mock_env_vars, d1_config):
        """Test D1Database initialization."""
        from src.services.cloudflare_storage import D1Database

        db = D1Database(d1_config)

        assert db.config == d1_config
        assert "test-account" in db.base_url
        assert "test-db" in db.base_url

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_env_vars, d1_database):
        """Test successful SQL execution."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": [{"id": 1, "name": "test"}]}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await d1_database.execute("SELECT * FROM tests")

        assert result["result"] == [{"id": 1, "name": "test"}]

    @pytest.mark.asyncio
    async def test_execute_with_params(self, mock_env_vars, d1_database):
        """Test SQL execution with parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": []}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            await d1_database.execute(
                "SELECT * FROM tests WHERE id = ?",
                params=[123],
            )

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["params"] == [123]

    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_env_vars, d1_database):
        """Test SQL execution failure."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            with pytest.raises(Exception) as exc_info:
                await d1_database.execute("SELECT * FROM tests")

            assert "D1 query failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_store_test_run(self, mock_env_vars, d1_database):
        """Test storing a test run."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": []}

        test_result = {
            "success": True,
            "steps": [{"success": True}, {"success": True}],
            "duration_ms": 1500,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await d1_database.store_test_run(
                test_id="test-123",
                project_id="proj-456",
                result=test_result,
            )

        assert result == "test-123"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_test_history(self, mock_env_vars, d1_database):
        """Test getting test history."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": [
                {"id": "test-1", "success": True},
                {"id": "test-2", "success": False},
            ]
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await d1_database.get_test_history("proj-456", limit=10)

        assert len(result) == 2


class TestKVCache:
    """Tests for KVCache class."""

    @pytest.fixture
    def kv_config(self):
        """Create a test config for KV."""
        from src.services.cloudflare_storage import CloudflareConfig

        return CloudflareConfig(
            account_id="test-account",
            api_token="test-token",
            kv_namespace_id="test-ns",
        )

    @pytest.fixture
    def kv_cache(self, kv_config):
        """Create KVCache instance."""
        from src.services.cloudflare_storage import KVCache

        return KVCache(kv_config)

    def test_kv_initialization(self, mock_env_vars, kv_config):
        """Test KVCache initialization."""
        from src.services.cloudflare_storage import KVCache

        cache = KVCache(kv_config)

        assert cache.config == kv_config
        assert "test-account" in cache.base_url
        assert "test-ns" in cache.base_url

    @pytest.mark.asyncio
    async def test_set_success(self, mock_env_vars, kv_cache):
        """Test successful KV set."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await kv_cache.set("test-key", {"data": "value"}, ttl_seconds=3600)

        assert result is True

    @pytest.mark.asyncio
    async def test_set_string_value(self, mock_env_vars, kv_cache):
        """Test KV set with string value."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await kv_cache.set("test-key", "string-value")

        assert result is True

    @pytest.mark.asyncio
    async def test_get_json_value(self, mock_env_vars, kv_cache):
        """Test KV get returning JSON value."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"data": "value"}'

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await kv_cache.get("test-key")

        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_get_plain_string(self, mock_env_vars, kv_cache):
        """Test KV get returning plain string."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "plain string"

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await kv_cache.get("test-key")

        assert result == "plain string"

    @pytest.mark.asyncio
    async def test_get_not_found(self, mock_env_vars, kv_cache):
        """Test KV get when key not found."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await kv_cache.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_page_elements(self, mock_env_vars, kv_cache):
        """Test caching page elements."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        elements = [
            {"selector": "#btn1", "type": "button"},
            {"selector": "#input1", "type": "input"},
        ]

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await kv_cache.cache_page_elements("https://example.com", elements)

        assert result is True

    @pytest.mark.asyncio
    async def test_get_cached_elements(self, mock_env_vars, kv_cache):
        """Test getting cached page elements."""
        elements = [{"selector": "#btn1", "type": "button"}]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(elements)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await kv_cache.get_cached_elements("https://example.com")

        assert result == elements


class TestAIGateway:
    """Tests for AIGateway class."""

    @pytest.fixture
    def gateway_config(self):
        """Create a test config for AI Gateway."""
        from src.services.cloudflare_storage import CloudflareConfig

        return CloudflareConfig(
            account_id="test-account",
            api_token="test-token",
            ai_gateway_id="test-gateway",
        )

    def test_ai_gateway_initialization(self, mock_env_vars, gateway_config):
        """Test AIGateway initialization."""
        from src.services.cloudflare_storage import AIGateway

        gateway = AIGateway(gateway_config)

        assert gateway.config == gateway_config
        assert "test-account" in gateway.gateway_url
        assert "test-gateway" in gateway.gateway_url

    def test_get_anthropic_url(self, mock_env_vars, gateway_config):
        """Test getting Anthropic gateway URL."""
        from src.services.cloudflare_storage import AIGateway

        gateway = AIGateway(gateway_config)
        url = gateway.get_anthropic_url()

        assert "test-account" in url
        assert "test-gateway" in url
        assert "/anthropic/v1/messages" in url

    def test_get_openai_url(self, mock_env_vars, gateway_config):
        """Test getting OpenAI gateway URL."""
        from src.services.cloudflare_storage import AIGateway

        gateway = AIGateway(gateway_config)
        url = gateway.get_openai_url()

        assert "test-account" in url
        assert "test-gateway" in url
        assert "/openai/v1/chat/completions" in url


class TestCloudflareClient:
    """Tests for unified CloudflareClient."""

    @pytest.fixture
    def cf_config(self):
        """Create a test config."""
        from src.services.cloudflare_storage import CloudflareConfig

        return CloudflareConfig(
            account_id="test-account",
            api_token="test-token",
            r2_bucket="test-bucket",
            vectorize_index="test-index",
            d1_database_id="test-db",
            kv_namespace_id="test-kv",
        )

    def test_client_initialization(self, mock_env_vars, cf_config):
        """Test CloudflareClient initialization."""
        from src.services.cloudflare_storage import CloudflareClient

        client = CloudflareClient(cf_config)

        assert client.config == cf_config
        assert client.r2 is not None
        assert client.vectorize is not None
        assert client.d1 is not None
        assert client.kv is not None
        assert client.ai_gateway is not None

    def test_client_default_config(self, mock_env_vars, monkeypatch):
        """Test CloudflareClient with default config from env."""
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "env-account")
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "env-token")

        from src.services.cloudflare_storage import CloudflareClient

        client = CloudflareClient()

        assert client.config.account_id == "env-account"

    @pytest.mark.asyncio
    async def test_get_healing_suggestions_no_vectorize(self, mock_env_vars):
        """Test get_healing_suggestions when vectorize not configured."""
        from src.services.cloudflare_storage import CloudflareConfig, CloudflareClient

        config = CloudflareConfig(
            account_id="test",
            api_token="test",
            vectorize_index="",  # Not configured
        )
        client = CloudflareClient(config)

        result = await client.get_healing_suggestions("error", ".selector")

        assert result == []


class TestGlobalFunctions:
    """Tests for module-level functions."""

    def test_get_cloudflare_client_singleton(self, mock_env_vars):
        """Test that get_cloudflare_client returns singleton."""
        mock_settings = MagicMock()
        mock_settings.cloudflare_account_id = "test-account"
        mock_settings.cloudflare_api_token = MagicMock()
        mock_settings.cloudflare_api_token.get_secret_value.return_value = "test-token"
        mock_settings.cloudflare_r2_bucket = None
        mock_settings.cloudflare_vectorize_index = None
        mock_settings.cloudflare_d1_database_id = None
        mock_settings.cloudflare_kv_namespace_id = None
        mock_settings.cloudflare_gateway_id = None

        import src.services.cloudflare_storage as module
        module._cloudflare_client = None

        with patch("src.config.get_settings", return_value=mock_settings):
            client1 = module.get_cloudflare_client()
            client2 = module.get_cloudflare_client()

        assert client1 is client2

    def test_is_cloudflare_configured_true(self, mock_env_vars):
        """Test is_cloudflare_configured returns True when configured."""
        mock_settings = MagicMock()
        mock_settings.cloudflare_account_id = "test-account"
        mock_settings.cloudflare_api_token = MagicMock()
        mock_settings.cloudflare_api_token.get_secret_value.return_value = "test-token"

        with patch("src.config.get_settings", return_value=mock_settings):
            from src.services.cloudflare_storage import is_cloudflare_configured

            result = is_cloudflare_configured()

        assert result is True

    def test_is_cloudflare_configured_false(self, mock_env_vars):
        """Test is_cloudflare_configured returns False when not configured."""
        mock_settings = MagicMock()
        mock_settings.cloudflare_account_id = None
        mock_settings.cloudflare_api_token = None

        with patch("src.config.get_settings", return_value=mock_settings):
            from src.services.cloudflare_storage import is_cloudflare_configured

            result = is_cloudflare_configured()

        assert result is False

    def test_is_cloudflare_configured_exception_fallback(self, mock_env_vars, monkeypatch):
        """Test is_cloudflare_configured falls back to env vars on exception."""
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "env-account")
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "env-token")

        with patch("src.config.get_settings", side_effect=Exception("Settings error")):
            from src.services.cloudflare_storage import is_cloudflare_configured

            result = is_cloudflare_configured()

        assert result is True
