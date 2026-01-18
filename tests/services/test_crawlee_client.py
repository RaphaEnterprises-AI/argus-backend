"""Tests for the Crawlee Service Client.

This module tests:
- CrawleeResponse dataclass
- CrawleeServiceUnavailable exception
- CrawleeClient HTTP operations
- Discovery crawling
- Screenshot capture
- Test execution
- Element extraction
- Health checking
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


class TestCrawleeResponse:
    """Tests for CrawleeResponse dataclass."""

    def test_response_creation_success(self, mock_env_vars):
        """Test creating a successful CrawleeResponse."""
        from src.services.crawlee_client import CrawleeResponse

        response = CrawleeResponse(
            success=True,
            request_id="req-123",
            duration=1500,
            data={"pages": [], "elements": []},
        )

        assert response.success is True
        assert response.request_id == "req-123"
        assert response.duration == 1500
        assert response.data == {"pages": [], "elements": []}
        assert response.error is None

    def test_response_creation_failure(self, mock_env_vars):
        """Test creating a failed CrawleeResponse."""
        from src.services.crawlee_client import CrawleeResponse

        response = CrawleeResponse(
            success=False,
            request_id="req-456",
            duration=0,
            data={},
            error="Connection timeout",
        )

        assert response.success is False
        assert response.error == "Connection timeout"


class TestCrawleeServiceUnavailable:
    """Tests for CrawleeServiceUnavailable exception."""

    def test_exception_creation(self, mock_env_vars):
        """Test creating CrawleeServiceUnavailable exception."""
        from src.services.crawlee_client import CrawleeServiceUnavailable

        exc = CrawleeServiceUnavailable("Cannot connect to service")

        assert str(exc) == "Cannot connect to service"
        assert isinstance(exc, Exception)


class TestCrawleeClient:
    """Tests for CrawleeClient class."""

    @pytest.fixture
    def crawlee_client(self):
        """Create a CrawleeClient instance."""
        from src.services.crawlee_client import CrawleeClient

        return CrawleeClient(base_url="http://localhost:3000")

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        client = AsyncMock()
        client.is_closed = False
        client.get = AsyncMock()
        client.post = AsyncMock()
        client.aclose = AsyncMock()
        return client

    def test_client_initialization_default(self, mock_env_vars):
        """Test CrawleeClient initialization with defaults."""
        from src.services.crawlee_client import CRAWLEE_SERVICE_URL, CrawleeClient

        client = CrawleeClient()

        assert client.base_url == CRAWLEE_SERVICE_URL
        assert client._client is None

    def test_client_initialization_custom_url(self, mock_env_vars):
        """Test CrawleeClient initialization with custom URL."""
        from src.services.crawlee_client import CrawleeClient

        client = CrawleeClient(base_url="http://custom:8080")

        assert client.base_url == "http://custom:8080"

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, mock_env_vars, crawlee_client):
        """Test that _get_client creates an httpx client."""
        assert crawlee_client._client is None

        client = await crawlee_client._get_client()

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

        # Cleanup
        await crawlee_client.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self, mock_env_vars, crawlee_client):
        """Test that _get_client reuses existing client."""
        client1 = await crawlee_client._get_client()
        client2 = await crawlee_client._get_client()

        assert client1 is client2

        # Cleanup
        await crawlee_client.close()

    @pytest.mark.asyncio
    async def test_get_client_creates_new_if_closed(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test that _get_client creates new client if previous was closed."""
        mock_http_client.is_closed = True
        crawlee_client._client = mock_http_client

        # Should create a new client since the old one is closed
        client = await crawlee_client._get_client()

        assert client is not crawlee_client._client or client.is_closed is False

        # Cleanup
        await crawlee_client.close()

    @pytest.mark.asyncio
    async def test_close_client(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test closing the HTTP client."""
        crawlee_client._client = mock_http_client

        await crawlee_client.close()

        mock_http_client.aclose.assert_called_once()
        assert crawlee_client._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self, mock_env_vars, crawlee_client):
        """Test close when no client exists."""
        assert crawlee_client._client is None

        # Should not raise
        await crawlee_client.close()

        assert crawlee_client._client is None


class TestCrawleeClientHealthCheck:
    """Tests for health check functionality."""

    @pytest.fixture
    def crawlee_client(self):
        """Create a CrawleeClient instance."""
        from src.services.crawlee_client import CrawleeClient

        return CrawleeClient(base_url="http://localhost:3000")

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        client = AsyncMock()
        client.is_closed = False
        client.get = AsyncMock()
        client.aclose = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy", "version": "1.0.0"}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        result = await crawlee_client.health_check()

        assert result == {"status": "healthy", "version": "1.0.0"}
        mock_http_client.get.assert_called_once_with("/health")

    @pytest.mark.asyncio
    async def test_health_check_connect_error(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test health check with connection error."""
        from src.services.crawlee_client import CrawleeServiceUnavailable

        mock_http_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        crawlee_client._client = mock_http_client

        with pytest.raises(CrawleeServiceUnavailable) as exc_info:
            await crawlee_client.health_check()

        assert "Cannot connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_health_check_other_error(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test health check with other errors."""
        from src.services.crawlee_client import CrawleeServiceUnavailable

        mock_http_client.get = AsyncMock(side_effect=Exception("Unknown error"))
        crawlee_client._client = mock_http_client

        with pytest.raises(CrawleeServiceUnavailable) as exc_info:
            await crawlee_client.health_check()

        assert "service error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_is_available_true(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test is_available returns True when service is healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        result = await crawlee_client.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_false(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test is_available returns False when service is unavailable."""
        mock_http_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        crawlee_client._client = mock_http_client

        result = await crawlee_client.is_available()

        assert result is False


class TestCrawleeClientDiscovery:
    """Tests for discovery crawling functionality."""

    @pytest.fixture
    def crawlee_client(self):
        """Create a CrawleeClient instance."""
        from src.services.crawlee_client import CrawleeClient

        return CrawleeClient(base_url="http://localhost:3000")

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        client = AsyncMock()
        client.is_closed = False
        client.post = AsyncMock()
        client.aclose = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_run_discovery_success(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test successful discovery crawl."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "requestId": "disc-123",
            "duration": 5000,
            "result": {
                "pages": [
                    {"url": "https://example.com", "title": "Example"},
                    {"url": "https://example.com/about", "title": "About"},
                ],
                "elements": [],
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        result = await crawlee_client.run_discovery(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
        )

        assert result.success is True
        assert result.request_id == "disc-123"
        assert result.duration == 5000
        assert len(result.data.get("pages", [])) == 2

    @pytest.mark.asyncio
    async def test_run_discovery_with_all_options(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test discovery crawl with all options."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "requestId": "disc-456",
            "duration": 3000,
            "result": {},
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        await crawlee_client.run_discovery(
            start_url="https://example.com",
            max_pages=50,
            max_depth=3,
            include_patterns=["*/products/*"],
            exclude_patterns=["*/admin/*"],
            capture_screenshots=True,
            viewport={"width": 1280, "height": 720},
            auth_config={"type": "basic", "username": "user", "password": "pass"},
        )

        call_args = mock_http_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["startUrl"] == "https://example.com"
        assert payload["maxPages"] == 50
        assert payload["maxDepth"] == 3
        assert payload["includePatterns"] == ["*/products/*"]
        assert payload["excludePatterns"] == ["*/admin/*"]
        assert payload["captureScreenshots"] is True
        assert payload["viewport"]["width"] == 1280
        assert "authConfig" in payload

    @pytest.mark.asyncio
    async def test_run_discovery_connect_error(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test discovery crawl with connection error."""
        from src.services.crawlee_client import CrawleeServiceUnavailable

        mock_http_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        crawlee_client._client = mock_http_client

        with pytest.raises(CrawleeServiceUnavailable):
            await crawlee_client.run_discovery(start_url="https://example.com")

    @pytest.mark.asyncio
    async def test_run_discovery_http_error(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test discovery crawl with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("Server error", request=MagicMock(), response=mock_response)
        )
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        result = await crawlee_client.run_discovery(start_url="https://example.com")

        assert result.success is False
        assert result.error is not None


class TestCrawleeClientScreenshot:
    """Tests for screenshot capture functionality."""

    @pytest.fixture
    def crawlee_client(self):
        """Create a CrawleeClient instance."""
        from src.services.crawlee_client import CrawleeClient

        return CrawleeClient(base_url="http://localhost:3000")

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        client = AsyncMock()
        client.is_closed = False
        client.post = AsyncMock()
        client.aclose = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_capture_screenshot_success(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test successful screenshot capture."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "requestId": "cap-123",
            "duration": 2000,
            "result": {
                "screenshot": "base64encodeddata",
                "dom": {"html": "<html>...</html>"},
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        result = await crawlee_client.capture_screenshot(
            url="https://example.com",
        )

        assert result.success is True
        assert result.request_id == "cap-123"
        assert "screenshot" in result.data

    @pytest.mark.asyncio
    async def test_capture_screenshot_with_options(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test screenshot capture with all options."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "requestId": "cap-456",
            "duration": 3000,
            "result": {},
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        await crawlee_client.capture_screenshot(
            url="https://example.com",
            viewport={"width": 1440, "height": 900},
            full_page=True,
            selector="#main-content",
            wait_for_selector=".loaded",
            wait_for_timeout=10000,
            capture_dom=True,
        )

        call_args = mock_http_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["url"] == "https://example.com"
        assert payload["viewport"]["width"] == 1440
        assert payload["fullPage"] is True
        assert payload["selector"] == "#main-content"
        assert payload["waitForSelector"] == ".loaded"
        assert payload["waitForTimeout"] == 10000
        assert payload["captureDom"] is True

    @pytest.mark.asyncio
    async def test_capture_screenshot_connect_error(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test screenshot capture with connection error."""
        from src.services.crawlee_client import CrawleeServiceUnavailable

        mock_http_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        crawlee_client._client = mock_http_client

        with pytest.raises(CrawleeServiceUnavailable):
            await crawlee_client.capture_screenshot(url="https://example.com")

    @pytest.mark.asyncio
    async def test_capture_screenshot_http_error(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test screenshot capture with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("Bad request", request=MagicMock(), response=mock_response)
        )
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        result = await crawlee_client.capture_screenshot(url="https://example.com")

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_capture_responsive_success(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test responsive screenshot capture."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "requestId": "resp-123",
            "duration": 5000,
            "results": {
                "desktop": {"screenshot": "base64..."},
                "tablet": {"screenshot": "base64..."},
                "mobile": {"screenshot": "base64..."},
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        viewports = [
            {"name": "desktop", "width": 1920, "height": 1080},
            {"name": "tablet", "width": 768, "height": 1024},
            {"name": "mobile", "width": 375, "height": 667},
        ]

        result = await crawlee_client.capture_responsive(
            url="https://example.com",
            viewports=viewports,
        )

        assert result.success is True
        assert "desktop" in result.data

    @pytest.mark.asyncio
    async def test_capture_responsive_connect_error(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test responsive capture with connection error."""
        from src.services.crawlee_client import CrawleeServiceUnavailable

        mock_http_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        crawlee_client._client = mock_http_client

        with pytest.raises(CrawleeServiceUnavailable):
            await crawlee_client.capture_responsive(url="https://example.com")


class TestCrawleeClientTestExecution:
    """Tests for test execution functionality."""

    @pytest.fixture
    def crawlee_client(self):
        """Create a CrawleeClient instance."""
        from src.services.crawlee_client import CrawleeClient

        return CrawleeClient(base_url="http://localhost:3000")

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        client = AsyncMock()
        client.is_closed = False
        client.post = AsyncMock()
        client.aclose = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_execute_test_success(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test successful test execution."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "requestId": "test-123",
            "duration": 10000,
            "result": {
                "passed": True,
                "steps": [
                    {"action": "goto", "success": True},
                    {"action": "click", "success": True},
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        steps = [
            {"action": "goto", "target": "/login"},
            {"action": "click", "target": "#login-btn"},
        ]

        result = await crawlee_client.execute_test(
            test_id="test-001",
            steps=steps,
            base_url="https://example.com",
        )

        assert result.success is True
        assert result.request_id == "test-123"
        assert result.data.get("passed") is True

    @pytest.mark.asyncio
    async def test_execute_test_with_all_options(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test test execution with all options."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "requestId": "test-456",
            "duration": 15000,
            "result": {"passed": True},
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        steps = [{"action": "goto", "target": "/"}]

        await crawlee_client.execute_test(
            test_id="test-002",
            steps=steps,
            base_url="https://example.com",
            viewport={"width": 1280, "height": 720},
            timeout=60000,
            capture_screenshots=True,
            capture_video=True,
        )

        call_args = mock_http_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["testId"] == "test-002"
        assert payload["steps"] == steps
        assert payload["baseUrl"] == "https://example.com"
        assert payload["viewport"]["width"] == 1280
        assert payload["timeout"] == 60000
        assert payload["captureScreenshots"] is True
        assert payload["captureVideo"] is True

    @pytest.mark.asyncio
    async def test_execute_test_failure(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test test execution with test failure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "requestId": "test-789",
            "duration": 5000,
            "result": {"passed": False},
            "error": "Step 2 failed: Element not found",
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        result = await crawlee_client.execute_test(
            test_id="test-003",
            steps=[{"action": "click", "target": "#nonexistent"}],
            base_url="https://example.com",
        )

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_execute_test_connect_error(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test test execution with connection error."""
        from src.services.crawlee_client import CrawleeServiceUnavailable

        mock_http_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        crawlee_client._client = mock_http_client

        with pytest.raises(CrawleeServiceUnavailable):
            await crawlee_client.execute_test(
                test_id="test-004",
                steps=[],
                base_url="https://example.com",
            )

    @pytest.mark.asyncio
    async def test_execute_test_http_error(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test test execution with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("Server error", request=MagicMock(), response=mock_response)
        )
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        result = await crawlee_client.execute_test(
            test_id="test-005",
            steps=[],
            base_url="https://example.com",
        )

        assert result.success is False


class TestCrawleeClientElementExtraction:
    """Tests for element extraction functionality."""

    @pytest.fixture
    def crawlee_client(self):
        """Create a CrawleeClient instance."""
        from src.services.crawlee_client import CrawleeClient

        return CrawleeClient(base_url="http://localhost:3000")

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        client = AsyncMock()
        client.is_closed = False
        client.post = AsyncMock()
        client.aclose = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_extract_elements_success(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test successful element extraction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "requestId": "ext-123",
            "duration": 1500,
            "elements": [
                {"selector": "#btn1", "type": "button", "text": "Submit"},
                {"selector": "#input1", "type": "input", "placeholder": "Enter text"},
            ],
            "forms": [
                {"selector": "form#login", "inputs": ["#username", "#password"]},
            ],
            "links": [
                {"href": "/about", "text": "About Us"},
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        result = await crawlee_client.extract_elements(url="https://example.com")

        assert result.success is True
        assert result.request_id == "ext-123"
        assert len(result.data.get("elements", [])) == 2
        assert len(result.data.get("forms", [])) == 1
        assert len(result.data.get("links", [])) == 1

    @pytest.mark.asyncio
    async def test_extract_elements_with_selectors(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test element extraction with specific selectors."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "requestId": "ext-456",
            "duration": 1000,
            "elements": [{"selector": "#specific", "type": "div"}],
            "forms": [],
            "links": [],
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        await crawlee_client.extract_elements(
            url="https://example.com",
            selectors=["#specific", ".class-name"],
        )

        call_args = mock_http_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["selectors"] == ["#specific", ".class-name"]

    @pytest.mark.asyncio
    async def test_extract_elements_connect_error(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test element extraction with connection error."""
        from src.services.crawlee_client import CrawleeServiceUnavailable

        mock_http_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        crawlee_client._client = mock_http_client

        with pytest.raises(CrawleeServiceUnavailable):
            await crawlee_client.extract_elements(url="https://example.com")

    @pytest.mark.asyncio
    async def test_extract_elements_http_error(self, mock_env_vars, crawlee_client, mock_http_client):
        """Test element extraction with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_response)
        )
        mock_http_client.post = AsyncMock(return_value=mock_response)
        crawlee_client._client = mock_http_client

        result = await crawlee_client.extract_elements(url="https://example.com/nonexistent")

        assert result.success is False
        assert result.error is not None


class TestGlobalFunctions:
    """Tests for module-level functions."""

    def test_get_crawlee_client_singleton(self, mock_env_vars):
        """Test that get_crawlee_client returns singleton."""
        import src.services.crawlee_client as module

        # Reset global instance
        module._client = None

        client1 = module.get_crawlee_client()
        client2 = module.get_crawlee_client()

        assert client1 is client2

    @pytest.mark.asyncio
    async def test_check_crawlee_service_available(self, mock_env_vars):
        """Test check_crawlee_service when available."""
        import src.services.crawlee_client as module

        module._client = None

        mock_client = MagicMock()
        mock_client.is_available = AsyncMock(return_value=True)

        with patch.object(module, "get_crawlee_client", return_value=mock_client):
            result = await module.check_crawlee_service()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_crawlee_service_unavailable(self, mock_env_vars):
        """Test check_crawlee_service when unavailable."""
        import src.services.crawlee_client as module

        module._client = None

        mock_client = MagicMock()
        mock_client.is_available = AsyncMock(return_value=False)

        with patch.object(module, "get_crawlee_client", return_value=mock_client):
            result = await module.check_crawlee_service()

        assert result is False


class TestCrawleeClientConfiguration:
    """Tests for configuration handling."""

    def test_environment_variables(self, mock_env_vars, monkeypatch):
        """Test that environment variables are respected."""
        monkeypatch.setenv("CRAWLEE_SERVICE_URL", "http://custom:9000")
        monkeypatch.setenv("CRAWLEE_TIMEOUT", "300")

        # Reload module to pick up new env vars
        import importlib

        import src.services.crawlee_client as module
        importlib.reload(module)

        assert module.CRAWLEE_SERVICE_URL == "http://custom:9000"
        assert module.CRAWLEE_TIMEOUT == 300

    def test_default_configuration(self, mock_env_vars, monkeypatch):
        """Test default configuration values."""
        monkeypatch.delenv("CRAWLEE_SERVICE_URL", raising=False)
        monkeypatch.delenv("CRAWLEE_TIMEOUT", raising=False)

        import importlib

        import src.services.crawlee_client as module
        importlib.reload(module)

        assert module.CRAWLEE_SERVICE_URL == "http://localhost:3000"
        assert module.CRAWLEE_TIMEOUT == 120
