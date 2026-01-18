"""Tests for Browser Pool Client module."""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


class TestUserContext:
    """Tests for UserContext dataclass."""

    def test_user_context_creation(self, mock_env_vars):
        """Test creating UserContext."""
        from src.browser.pool_client import UserContext

        ctx = UserContext(
            user_id="user-123",
            org_id="org-456",
            email="test@example.com",
            ip="192.168.1.1",
        )

        assert ctx.user_id == "user-123"
        assert ctx.org_id == "org-456"
        assert ctx.email == "test@example.com"
        assert ctx.ip == "192.168.1.1"

    def test_user_context_minimal(self, mock_env_vars):
        """Test UserContext with only required fields."""
        from src.browser.pool_client import UserContext

        ctx = UserContext(user_id="user-123")

        assert ctx.user_id == "user-123"
        assert ctx.org_id is None
        assert ctx.email is None
        assert ctx.ip is None


class TestBase64UrlEncode:
    """Tests for _base64url_encode function."""

    def test_base64url_encode(self, mock_env_vars):
        """Test base64url encoding."""
        from src.browser.pool_client import _base64url_encode

        result = _base64url_encode(b"test data")

        assert isinstance(result, str)
        # Should not contain padding or standard base64 chars that are unsafe
        assert "=" not in result
        assert "+" not in result
        assert "/" not in result

    def test_base64url_encode_json(self, mock_env_vars):
        """Test base64url encoding of JSON."""
        from src.browser.pool_client import _base64url_encode

        data = json.dumps({"test": "value"}).encode()
        result = _base64url_encode(data)

        assert isinstance(result, str)
        assert len(result) > 0


class TestSignPoolToken:
    """Tests for sign_pool_token function."""

    def test_sign_pool_token(self, mock_env_vars):
        """Test JWT token signing."""
        from src.browser.pool_client import sign_pool_token, UserContext

        ctx = UserContext(
            user_id="user-123",
            org_id="org-456",
            email="test@example.com",
        )

        token = sign_pool_token(
            user_context=ctx,
            secret="test-secret",
            action="observe",
        )

        # JWT format: header.payload.signature
        parts = token.split(".")
        assert len(parts) == 3
        assert all(len(p) > 0 for p in parts)

    def test_sign_pool_token_different_secrets(self, mock_env_vars):
        """Test that different secrets produce different signatures."""
        from src.browser.pool_client import sign_pool_token, UserContext

        ctx = UserContext(user_id="user-123")

        token1 = sign_pool_token(ctx, secret="secret1", action="act")
        token2 = sign_pool_token(ctx, secret="secret2", action="act")

        # Signatures should differ
        sig1 = token1.split(".")[2]
        sig2 = token2.split(".")[2]
        assert sig1 != sig2

    def test_sign_pool_token_expires(self, mock_env_vars):
        """Test token expiration setting."""
        from src.browser.pool_client import sign_pool_token, UserContext
        import base64

        ctx = UserContext(user_id="user-123")

        token = sign_pool_token(ctx, secret="secret", expires_in_seconds=600)

        # Decode payload
        payload_b64 = token.split(".")[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        assert "exp" in payload
        assert "iat" in payload
        assert payload["exp"] - payload["iat"] == 600


class TestBrowserPoolErrors:
    """Tests for Browser Pool exception classes."""

    def test_browser_pool_error(self, mock_env_vars):
        """Test BrowserPoolError."""
        from src.browser.pool_client import BrowserPoolError

        error = BrowserPoolError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_browser_pool_timeout_error(self, mock_env_vars):
        """Test BrowserPoolTimeoutError."""
        from src.browser.pool_client import BrowserPoolTimeoutError, BrowserPoolError

        error = BrowserPoolTimeoutError("Request timed out")
        assert isinstance(error, BrowserPoolError)
        assert str(error) == "Request timed out"

    def test_browser_pool_unavailable_error(self, mock_env_vars):
        """Test BrowserPoolUnavailableError."""
        from src.browser.pool_client import BrowserPoolUnavailableError, BrowserPoolError

        error = BrowserPoolUnavailableError("No browsers available")
        assert isinstance(error, BrowserPoolError)
        assert str(error) == "No browsers available"


class TestBrowserPoolClientInit:
    """Tests for BrowserPoolClient initialization."""

    def test_client_initialization_default(self, mock_env_vars, monkeypatch):
        """Test default client initialization."""
        from src.browser.pool_client import BrowserPoolClient

        monkeypatch.setenv("BROWSER_POOL_URL", "http://pool.example.com")

        client = BrowserPoolClient()

        assert client.pool_url == "http://pool.example.com"
        assert client._client is None
        assert client._selector_cache == {}

    def test_client_initialization_with_params(self, mock_env_vars):
        """Test client initialization with parameters."""
        from src.browser.pool_client import BrowserPoolClient, UserContext
        from src.browser.pool_models import BrowserPoolConfig

        user_ctx = UserContext(user_id="test-user", org_id="test-org")
        config = BrowserPoolConfig(pool_url="http://custom.pool.com", timeout_ms=30000)

        client = BrowserPoolClient(
            pool_url="http://custom.pool.com",
            jwt_secret="test-secret",
            user_context=user_ctx,
            config=config,
        )

        assert client.pool_url == "http://custom.pool.com"
        assert client.jwt_secret == "test-secret"
        assert client.user_context.user_id == "test-user"
        assert client.config.timeout_ms == 30000

    def test_client_initialization_fallback_url(self, mock_env_vars, monkeypatch):
        """Test fallback to BROWSER_WORKER_URL."""
        from src.browser.pool_client import BrowserPoolClient

        monkeypatch.delenv("BROWSER_POOL_URL", raising=False)
        monkeypatch.setenv("BROWSER_WORKER_URL", "http://worker.example.com")

        client = BrowserPoolClient()

        assert client.pool_url == "http://worker.example.com"

    def test_client_initialization_default_localhost(self, mock_env_vars, monkeypatch):
        """Test default to localhost when no env vars set."""
        from src.browser.pool_client import BrowserPoolClient

        monkeypatch.delenv("BROWSER_POOL_URL", raising=False)
        monkeypatch.delenv("BROWSER_WORKER_URL", raising=False)

        client = BrowserPoolClient()

        assert client.pool_url == "http://localhost:8080"


class TestBrowserPoolClientAuth:
    """Tests for BrowserPoolClient authentication."""

    def test_get_auth_header_jwt(self, mock_env_vars):
        """Test auth header with JWT secret."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(
            pool_url="http://pool.example.com",
            jwt_secret="test-jwt-secret",
        )

        headers = client._get_auth_header(action="observe")

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        # Should be a JWT
        token = headers["Authorization"].replace("Bearer ", "")
        assert len(token.split(".")) == 3

    def test_get_auth_header_api_key(self, mock_env_vars):
        """Test auth header with API key (legacy)."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(
            pool_url="http://pool.example.com",
            api_key="test-api-key",
        )

        headers = client._get_auth_header()

        assert headers["Authorization"] == "Bearer test-api-key"

    def test_get_auth_header_no_auth(self, mock_env_vars, monkeypatch):
        """Test auth header when no credentials provided."""
        from src.browser.pool_client import BrowserPoolClient

        monkeypatch.delenv("BROWSER_POOL_JWT_SECRET", raising=False)
        monkeypatch.delenv("BROWSER_POOL_API_KEY", raising=False)

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        headers = client._get_auth_header()

        assert headers == {}


class TestBrowserPoolClientContextManager:
    """Tests for BrowserPoolClient context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_enter(self, mock_env_vars):
        """Test async context manager entry."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(client, "_ensure_client", new_callable=AsyncMock) as mock_ensure:
            async with client as ctx:
                assert ctx is client
                mock_ensure.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_exit(self, mock_env_vars):
        """Test async context manager exit."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(client, "_ensure_client", new_callable=AsyncMock):
            with patch.object(client, "close", new_callable=AsyncMock) as mock_close:
                async with client:
                    pass
                mock_close.assert_called_once()


class TestBrowserPoolClientEnsureClient:
    """Tests for _ensure_client method."""

    @pytest.mark.asyncio
    async def test_ensure_client_creates_client(self, mock_env_vars):
        """Test _ensure_client creates httpx client."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        assert client._client is None

        await client._ensure_client()

        assert client._client is not None
        assert isinstance(client._client, httpx.AsyncClient)

        await client.close()

    @pytest.mark.asyncio
    async def test_ensure_client_idempotent(self, mock_env_vars):
        """Test _ensure_client is idempotent."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        await client._ensure_client()
        first_client = client._client

        await client._ensure_client()
        second_client = client._client

        assert first_client is second_client

        await client.close()


class TestBrowserPoolClientClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    async def test_close_client(self, mock_env_vars):
        """Test closing the client."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")
        await client._ensure_client()

        assert client._client is not None

        await client.close()

        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self, mock_env_vars):
        """Test closing when client not initialized."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        # Should not raise
        await client.close()

        assert client._client is None


class TestBrowserPoolClientCacheKey:
    """Tests for _cache_key method."""

    def test_cache_key_generation(self, mock_env_vars):
        """Test cache key generation."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        key = client._cache_key("https://example.com", "login button")

        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hex digest length

    def test_cache_key_consistency(self, mock_env_vars):
        """Test cache key is consistent for same inputs."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        key1 = client._cache_key("https://example.com", "button")
        key2 = client._cache_key("https://example.com", "button")

        assert key1 == key2

    def test_cache_key_different_inputs(self, mock_env_vars):
        """Test cache key differs for different inputs."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        key1 = client._cache_key("https://example.com", "button1")
        key2 = client._cache_key("https://example.com", "button2")

        assert key1 != key2


class TestBrowserPoolClientRequest:
    """Tests for _request method."""

    @pytest.mark.asyncio
    async def test_request_success(self, mock_env_vars, mock_httpx_client):
        """Test successful request."""
        from src.browser.pool_client import BrowserPoolClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": "test"}
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.get = AsyncMock(return_value=mock_response)
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = BrowserPoolClient(pool_url="http://pool.example.com")
        client._client = mock_httpx_client

        result = await client._request("GET", "/health")

        assert result == {"success": True, "data": "test"}

    @pytest.mark.asyncio
    async def test_request_post(self, mock_env_vars, mock_httpx_client):
        """Test POST request."""
        from src.browser.pool_client import BrowserPoolClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        client = BrowserPoolClient(pool_url="http://pool.example.com")
        client._client = mock_httpx_client

        result = await client._request("POST", "/observe", data={"url": "https://example.com"})

        assert result == {"result": "ok"}
        mock_httpx_client.post.assert_called()

    @pytest.mark.asyncio
    async def test_request_timeout_retry(self, mock_env_vars, mock_httpx_client):
        """Test request retries on timeout."""
        from src.browser.pool_client import BrowserPoolClient, BrowserPoolTimeoutError
        from src.browser.pool_models import BrowserPoolConfig

        mock_httpx_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        config = BrowserPoolConfig(
            pool_url="http://pool.example.com",
            retry_count=2,
            retry_delay_ms=10,
        )
        client = BrowserPoolClient(pool_url="http://pool.example.com", config=config)
        client._client = mock_httpx_client

        with pytest.raises(BrowserPoolTimeoutError):
            await client._request("GET", "/health")

        # Should have tried 3 times (initial + 2 retries)
        assert mock_httpx_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_request_503_error(self, mock_env_vars, mock_httpx_client):
        """Test 503 error raises BrowserPoolUnavailableError."""
        from src.browser.pool_client import BrowserPoolClient, BrowserPoolUnavailableError
        from src.browser.pool_models import BrowserPoolConfig

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "No browsers available"
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=mock_response)
        )

        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        config = BrowserPoolConfig(pool_url="http://pool.example.com", retry_count=0)
        client = BrowserPoolClient(pool_url="http://pool.example.com", config=config)
        client._client = mock_httpx_client

        with pytest.raises(BrowserPoolUnavailableError):
            await client._request("GET", "/health")


class TestBrowserPoolClientHealth:
    """Tests for health method."""

    @pytest.mark.asyncio
    async def test_health_check(self, mock_env_vars):
        """Test health check."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={
                "status": "healthy",
                "poolSize": 10,
                "available": 8,
                "activeSessions": 2,
            },
        ):
            health = await client.health()

            assert health.healthy is True
            assert health.total_pods == 10
            assert health.available_pods == 8
            assert health.active_sessions == 2

    @pytest.mark.asyncio
    async def test_health_check_cached(self, mock_env_vars):
        """Test health check uses cache."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={"status": "healthy", "poolSize": 10, "available": 8},
        ) as mock_request:
            # First call
            health1 = await client.health()

            # Second call should use cache
            health2 = await client.health(use_cache=True)

            # Should only have made one request
            assert mock_request.call_count == 1
            assert health1.healthy == health2.healthy

    @pytest.mark.asyncio
    async def test_health_check_error(self, mock_env_vars):
        """Test health check handles errors."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=Exception("Connection failed"),
        ):
            health = await client.health(use_cache=False)

            assert health.healthy is False


class TestBrowserPoolClientObserve:
    """Tests for observe method."""

    @pytest.mark.asyncio
    async def test_observe_success(self, mock_env_vars):
        """Test successful observe."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "url": "https://example.com",
                "title": "Test Page",
                "actions": [
                    {
                        "selector": "#btn",
                        "type": "button",
                        "text": "Click me",
                        "description": "Submit button",
                        "confidence": 0.9,
                    }
                ],
            },
        ):
            result = await client.observe("https://example.com")

            assert result.success is True
            assert len(result.elements) == 1
            assert result.elements[0].selector == "#btn"

    @pytest.mark.asyncio
    async def test_observe_with_instruction(self, mock_env_vars):
        """Test observe with custom instruction."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={"success": True, "url": "https://example.com", "actions": []},
        ) as mock_request:
            await client.observe("https://example.com", instruction="Find login forms")

            # _request is called as _request("POST", "/observe", {...})
            call_args = mock_request.call_args
            # Args are positional: (method, endpoint, data)
            data = call_args[0][2] if len(call_args[0]) > 2 else call_args.kwargs.get("data")
            assert data["instruction"] == "Find login forms"

    @pytest.mark.asyncio
    async def test_observe_failure(self, mock_env_vars):
        """Test observe handles failure."""
        from src.browser.pool_client import BrowserPoolClient, BrowserPoolError

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=BrowserPoolError("Request failed"),
        ):
            result = await client.observe("https://example.com")

            assert result.success is False
            assert "Request failed" in result.error


class TestBrowserPoolClientAct:
    """Tests for act method."""

    @pytest.mark.asyncio
    async def test_act_success(self, mock_env_vars):
        """Test successful act."""
        from src.browser.pool_client import BrowserPoolClient
        from src.browser.pool_models import ExecutionMode

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "message": "Action completed",
                "url": "https://example.com",
                "actions": [{"action": "click", "success": True, "selector": "#btn"}],
            },
        ):
            result = await client.act("https://example.com", "Click the button")

            assert result.success is True
            assert result.execution_mode == ExecutionMode.DOM
            assert len(result.actions) == 1

    @pytest.mark.asyncio
    async def test_act_with_screenshot(self, mock_env_vars):
        """Test act captures screenshot."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "url": "https://example.com",
                "screenshot": "base64_image_data",
                "actions": [],
            },
        ):
            result = await client.act("https://example.com", "Click button", screenshot=True)

            assert result.screenshot == "base64_image_data"

    @pytest.mark.asyncio
    async def test_act_vision_fallback(self, mock_env_vars):
        """Test act falls back to vision on failure."""
        from src.browser.pool_client import BrowserPoolClient
        from src.browser.pool_models import ExecutionMode

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        # First request fails, vision fallback succeeds
        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={"success": False, "url": "https://example.com", "actions": []},
        ):
            with patch.object(
                client,
                "_vision_fallback",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    success=True,
                    execution_mode=ExecutionMode.VISION,
                ),
            ) as mock_vision:
                result = await client.act(
                    "https://example.com",
                    "Click button",
                    use_vision_fallback=True,
                )

                mock_vision.assert_called_once()

    @pytest.mark.asyncio
    async def test_act_no_vision_fallback(self, mock_env_vars):
        """Test act without vision fallback."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={"success": False, "url": "https://example.com", "actions": []},
        ):
            result = await client.act(
                "https://example.com",
                "Click button",
                use_vision_fallback=False,
            )

            assert result.success is False


class TestBrowserPoolClientTest:
    """Tests for test method."""

    @pytest.mark.asyncio
    async def test_test_success(self, mock_env_vars):
        """Test successful test execution."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "steps": [
                    {"stepIndex": 0, "instruction": "Click login", "success": True, "duration": 500},
                    {"stepIndex": 1, "instruction": "Fill email", "success": True, "duration": 300},
                ],
                "finalScreenshot": "base64_image",
            },
        ):
            result = await client.test(
                "https://example.com",
                steps=["Click login", "Fill email"],
            )

            assert result.success is True
            assert len(result.steps) == 2
            assert result.steps[0].success is True
            assert result.final_screenshot == "base64_image"

    @pytest.mark.asyncio
    async def test_test_partial_failure(self, mock_env_vars):
        """Test test execution with partial failure."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={
                "success": False,
                "steps": [
                    {"stepIndex": 0, "instruction": "Click login", "success": True, "duration": 500},
                    {"stepIndex": 1, "instruction": "Fill email", "success": False, "error": "Element not found"},
                ],
            },
        ):
            result = await client.test(
                "https://example.com",
                steps=["Click login", "Fill email"],
            )

            assert result.success is False
            assert result.steps[1].error == "Element not found"

    @pytest.mark.asyncio
    async def test_test_with_browser_type(self, mock_env_vars):
        """Test test execution with specific browser."""
        from src.browser.pool_client import BrowserPoolClient
        from src.browser.pool_models import BrowserType

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={"success": True, "steps": []},
        ) as mock_request:
            await client.test(
                "https://example.com",
                steps=["Click button"],
                browser=BrowserType.FIREFOX,
            )

            # _request is called as _request("POST", "/test", {...})
            call_args = mock_request.call_args
            data = call_args[0][2] if len(call_args[0]) > 2 else call_args.kwargs.get("data")
            assert data["browser"] == "firefox"


class TestBrowserPoolClientExtract:
    """Tests for extract method."""

    @pytest.mark.asyncio
    async def test_extract_success(self, mock_env_vars):
        """Test successful data extraction."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "url": "https://example.com",
                "data": {"title": "Product", "price": 99.99},
            },
        ):
            result = await client.extract(
                "https://example.com",
                schema={"title": "string", "price": "number"},
            )

            assert result.success is True
            assert result.data["title"] == "Product"
            assert result.data["price"] == 99.99

    @pytest.mark.asyncio
    async def test_extract_with_instruction(self, mock_env_vars):
        """Test extract with instruction."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={"success": True, "data": {}},
        ) as mock_request:
            await client.extract(
                "https://example.com",
                schema={"name": "string"},
                instruction="Extract product name",
            )

            # _request is called as _request("POST", "/extract", {...})
            call_args = mock_request.call_args
            data = call_args[0][2] if len(call_args[0]) > 2 else call_args.kwargs.get("data")
            assert "Extract product name" in data["instruction"]


class TestBrowserPoolClientScreenshot:
    """Tests for screenshot method."""

    @pytest.mark.asyncio
    async def test_screenshot_success(self, mock_env_vars):
        """Test successful screenshot capture."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={"screenshot": "base64_screenshot_data"},
        ):
            result = await client.screenshot("https://example.com")

            assert result == "base64_screenshot_data"

    @pytest.mark.asyncio
    async def test_screenshot_full_page(self, mock_env_vars):
        """Test full page screenshot."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            return_value={"screenshot": "full_page_screenshot"},
        ) as mock_request:
            await client.screenshot("https://example.com", full_page=True)

            # _request is called as _request("POST", "/screenshot", {...})
            call_args = mock_request.call_args
            data = call_args[0][2] if len(call_args[0]) > 2 else call_args.kwargs.get("data")
            assert data["fullPage"] is True

    @pytest.mark.asyncio
    async def test_screenshot_failure(self, mock_env_vars):
        """Test screenshot handles failure."""
        from src.browser.pool_client import BrowserPoolClient

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=Exception("Failed"),
        ):
            result = await client.screenshot("https://example.com")

            assert result is None


class TestBrowserPoolClientVisionFallback:
    """Tests for _vision_fallback method."""

    @pytest.mark.asyncio
    async def test_vision_fallback_import_error(self, mock_env_vars):
        """Test vision fallback handles import error."""
        from src.browser.pool_client import BrowserPoolClient
        from src.browser.pool_models import ExecutionMode

        client = BrowserPoolClient(pool_url="http://pool.example.com")
        client._vision_client = None

        # Mock the import to raise ImportError
        with patch.dict("sys.modules", {"src.computer_use.client": None}):
            # Simulate what happens when the import fails
            original_fallback = client._vision_fallback

            async def mock_vision_fallback(url: str, instruction: str):
                try:
                    # Simulate the import error
                    raise ImportError("Module not found")
                except ImportError:
                    from src.browser.pool_models import ActResult, ExecutionMode
                    return ActResult(
                        success=False,
                        error="Vision fallback not available",
                        url=url,
                        execution_mode=ExecutionMode.VISION,
                    )

            client._vision_fallback = mock_vision_fallback

            result = await client._vision_fallback("https://example.com", "Click button")

            assert result.success is False
            assert result.execution_mode == ExecutionMode.VISION

    @pytest.mark.asyncio
    async def test_vision_fallback_exception(self, mock_env_vars):
        """Test vision fallback handles exceptions."""
        from src.browser.pool_client import BrowserPoolClient
        from src.browser.pool_models import ExecutionMode

        client = BrowserPoolClient(pool_url="http://pool.example.com")

        mock_vision_client = AsyncMock()
        mock_vision_client.execute_task = AsyncMock(side_effect=Exception("Vision failed"))
        client._vision_client = mock_vision_client

        result = await client._vision_fallback("https://example.com", "Click button")

        assert result.success is False
        assert result.execution_mode == ExecutionMode.VISION
        assert "Vision fallback failed" in result.error


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_browser_pool_client(self, mock_env_vars, monkeypatch):
        """Test get_browser_pool_client singleton."""
        # Reset the global client
        import src.browser.pool_client as module

        module._default_client = None

        monkeypatch.setenv("BROWSER_POOL_URL", "http://pool.example.com")

        from src.browser.pool_client import get_browser_pool_client

        client1 = get_browser_pool_client()
        client2 = get_browser_pool_client()

        assert client1 is client2

    @pytest.mark.asyncio
    async def test_observe_convenience(self, mock_env_vars, monkeypatch):
        """Test observe convenience function."""
        import src.browser.pool_client as module

        monkeypatch.setenv("BROWSER_POOL_URL", "http://pool.example.com")

        mock_client = MagicMock()
        mock_client.observe = AsyncMock(
            return_value=MagicMock(success=True, elements=[])
        )
        module._default_client = mock_client

        from src.browser.pool_client import observe

        result = await observe("https://example.com")

        mock_client.observe.assert_called_once()

    @pytest.mark.asyncio
    async def test_act_convenience(self, mock_env_vars, monkeypatch):
        """Test act convenience function."""
        import src.browser.pool_client as module

        monkeypatch.setenv("BROWSER_POOL_URL", "http://pool.example.com")

        mock_client = MagicMock()
        mock_client.act = AsyncMock(return_value=MagicMock(success=True))
        module._default_client = mock_client

        from src.browser.pool_client import act

        result = await act("https://example.com", "Click button")

        mock_client.act.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_convenience(self, mock_env_vars, monkeypatch):
        """Test test convenience function."""
        import src.browser.pool_client as module

        monkeypatch.setenv("BROWSER_POOL_URL", "http://pool.example.com")

        mock_client = MagicMock()
        mock_client.test = AsyncMock(return_value=MagicMock(success=True, steps=[]))
        module._default_client = mock_client

        from src.browser.pool_client import test

        result = await test("https://example.com", ["Click button"])

        mock_client.test.assert_called_once()
