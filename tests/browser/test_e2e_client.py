"""Tests for E2E Browser Client module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


class TestBrowserAction:
    """Tests for BrowserAction enum."""

    def test_browser_action_values(self, mock_env_vars):
        """Test BrowserAction enum values."""
        from src.browser.e2e_client import BrowserAction

        assert BrowserAction.ACT.value == "act"
        assert BrowserAction.EXTRACT.value == "extract"
        assert BrowserAction.OBSERVE.value == "observe"

    def test_browser_action_is_str_enum(self, mock_env_vars):
        """Test BrowserAction inherits from str."""
        from src.browser.e2e_client import BrowserAction

        assert isinstance(BrowserAction.ACT, str)


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_action_result_success(self, mock_env_vars):
        """Test creating a successful ActionResult."""
        from src.browser.e2e_client import ActionResult, BrowserAction

        result = ActionResult(
            success=True,
            action=BrowserAction.ACT,
            instruction="Click button",
            result={"clicked": True},
            cached=True,
            healed=False,
            duration_ms=100,
            tokens_used=50,
            model_used="claude-sonnet",
        )

        assert result.success is True
        assert result.cached is True
        assert result.healed is False
        assert result.tokens_used == 50

    def test_action_result_failure(self, mock_env_vars):
        """Test creating a failed ActionResult."""
        from src.browser.e2e_client import ActionResult, BrowserAction

        result = ActionResult(
            success=False,
            action=BrowserAction.ACT,
            instruction="Click button",
            error="Element not found",
        )

        assert result.success is False
        assert result.error == "Element not found"

    def test_action_result_defaults(self, mock_env_vars):
        """Test ActionResult default values."""
        from src.browser.e2e_client import ActionResult, BrowserAction

        result = ActionResult(
            success=True,
            action=BrowserAction.OBSERVE,
            instruction="Observe page",
        )

        assert result.cached is False
        assert result.healed is False
        assert result.duration_ms == 0
        assert result.tokens_used == 0
        assert result.model_used is None


class TestExtractionSchema:
    """Tests for ExtractionSchema class."""

    def test_extraction_schema_creation(self, mock_env_vars):
        """Test creating ExtractionSchema."""
        from src.browser.e2e_client import ExtractionSchema

        schema = ExtractionSchema(
            fields={
                "title": "string",
                "price": "number",
            }
        )

        assert schema.fields["title"] == "string"
        assert schema.fields["price"] == "number"

    def test_extraction_schema_to_zod(self, mock_env_vars):
        """Test converting schema to Zod format."""
        from src.browser.e2e_client import ExtractionSchema

        schema = ExtractionSchema(
            fields={
                "title": "string",
                "price": "number",
                "in_stock": "boolean",
                "tags": "array",
            }
        )

        zod_schema = schema.to_zod_schema()

        assert zod_schema["type"] == "object"
        assert "properties" in zod_schema
        assert zod_schema["properties"]["title"]["type"] == "string"
        assert zod_schema["properties"]["price"]["type"] == "number"
        assert zod_schema["properties"]["in_stock"]["type"] == "boolean"
        assert zod_schema["properties"]["tags"]["type"] == "array"

    def test_extraction_schema_unknown_type(self, mock_env_vars):
        """Test schema with unknown type defaults to string."""
        from src.browser.e2e_client import ExtractionSchema

        schema = ExtractionSchema(fields={"custom": "unknown_type"})
        zod_schema = schema.to_zod_schema()

        assert zod_schema["properties"]["custom"]["type"] == "string"


class TestPageState:
    """Tests for PageState dataclass."""

    def test_page_state_creation(self, mock_env_vars):
        """Test creating PageState."""
        from src.browser.e2e_client import PageState

        state = PageState(
            url="https://example.com",
            title="Test Page",
            visible_text="Hello World",
        )

        assert state.url == "https://example.com"
        assert state.title == "Test Page"
        assert state.visible_text == "Hello World"
        assert state.forms == []
        assert state.buttons == []
        assert state.links == []

    def test_page_state_with_elements(self, mock_env_vars):
        """Test PageState with form elements."""
        from src.browser.e2e_client import PageState

        state = PageState(
            url="https://example.com",
            title="Test",
            forms=[{"id": "login-form"}],
            buttons=["Submit", "Cancel"],
            links=[{"href": "/about", "text": "About"}],
        )

        assert len(state.forms) == 1
        assert len(state.buttons) == 2
        assert len(state.links) == 1


class TestBrowserPage:
    """Tests for BrowserPage class."""

    @pytest.fixture
    def mock_client(self, mock_env_vars):
        """Create mock E2EBrowserClient."""
        client = MagicMock()
        client._execute_action = AsyncMock(
            return_value={
                "success": True,
                "action_taken": "clicked",
                "cached": False,
                "healed": False,
            }
        )
        client._get_page_state = AsyncMock(
            return_value={
                "url": "https://example.com",
                "title": "Test Page",
            }
        )
        client._capture_screenshot = AsyncMock(return_value=b"screenshot_data")
        return client

    @pytest.mark.asyncio
    async def test_page_act_success(self, mock_env_vars, mock_client):
        """Test page.act method succeeds."""
        from src.browser.e2e_client import BrowserPage, BrowserAction

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        result = await page.act("Click the login button")

        assert result.success is True
        assert result.action == BrowserAction.ACT
        assert result.instruction == "Click the login button"
        mock_client._execute_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_page_act_failure(self, mock_env_vars, mock_client):
        """Test page.act handling failure."""
        from src.browser.e2e_client import BrowserPage

        mock_client._execute_action = AsyncMock(side_effect=Exception("Network error"))

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        result = await page.act("Click button")

        assert result.success is False
        assert "Network error" in result.error

    @pytest.mark.asyncio
    async def test_page_act_healed(self, mock_env_vars, mock_client):
        """Test page.act tracks self-healing."""
        from src.browser.e2e_client import BrowserPage

        mock_client._execute_action = AsyncMock(
            return_value={
                "success": True,
                "action_taken": "clicked",
                "cached": False,
                "healed": True,
                "original_selector": "#old-btn",
                "healed_selector": "#new-btn",
            }
        )

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        result = await page.act("Click button")

        assert result.healed is True

    @pytest.mark.asyncio
    async def test_page_extract_dict_schema(self, mock_env_vars, mock_client):
        """Test page.extract with dict schema."""
        from src.browser.e2e_client import BrowserPage, BrowserAction

        mock_client._execute_action = AsyncMock(
            return_value={
                "success": True,
                "data": {"title": "Product", "price": 99.99},
            }
        )

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        result = await page.extract({"title": "string", "price": "number"})

        assert result.success is True
        assert result.action == BrowserAction.EXTRACT
        assert result.result == {"title": "Product", "price": 99.99}

    @pytest.mark.asyncio
    async def test_page_extract_with_instruction(self, mock_env_vars, mock_client):
        """Test page.extract with custom instruction."""
        from src.browser.e2e_client import BrowserPage

        mock_client._execute_action = AsyncMock(
            return_value={"success": True, "data": {"items": ["a", "b"]}}
        )

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        result = await page.extract(
            {"items": "array"},
            instruction="Extract all product names",
        )

        assert result.success is True
        # Verify instruction was passed
        call_args = mock_client._execute_action.call_args
        assert call_args.kwargs["instruction"] == "Extract all product names"

    @pytest.mark.asyncio
    async def test_page_extract_failure(self, mock_env_vars, mock_client):
        """Test page.extract handles failure."""
        from src.browser.e2e_client import BrowserPage

        mock_client._execute_action = AsyncMock(side_effect=Exception("Extraction failed"))

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        result = await page.extract({"title": "string"})

        assert result.success is False
        assert "Extraction failed" in result.error

    @pytest.mark.asyncio
    async def test_page_observe(self, mock_env_vars, mock_client):
        """Test page.observe method."""
        from src.browser.e2e_client import BrowserPage, BrowserAction

        mock_client._execute_action = AsyncMock(
            return_value={
                "success": True,
                "observation": "I see a login form with email and password fields",
            }
        )

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        result = await page.observe("What forms are visible?")

        assert result.success is True
        assert result.action == BrowserAction.OBSERVE
        assert "login form" in result.result

    @pytest.mark.asyncio
    async def test_page_observe_failure(self, mock_env_vars, mock_client):
        """Test page.observe handles failure."""
        from src.browser.e2e_client import BrowserPage

        mock_client._execute_action = AsyncMock(side_effect=Exception("Observe failed"))

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        result = await page.observe()

        assert result.success is False
        assert "Observe failed" in result.error

    @pytest.mark.asyncio
    async def test_page_get_state(self, mock_env_vars, mock_client):
        """Test page.get_state method."""
        from src.browser.e2e_client import BrowserPage, PageState

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        state = await page.get_state()

        assert isinstance(state, PageState)
        assert state.url == "https://example.com"
        assert state.title == "Test Page"

    @pytest.mark.asyncio
    async def test_page_screenshot(self, mock_env_vars, mock_client):
        """Test page.screenshot method."""
        from src.browser.e2e_client import BrowserPage

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        screenshot = await page.screenshot()

        assert screenshot == b"screenshot_data"

    @pytest.mark.asyncio
    async def test_page_goto(self, mock_env_vars, mock_client):
        """Test page.goto method."""
        from src.browser.e2e_client import BrowserPage

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        result = await page.goto("https://other.com")

        # goto calls act internally
        assert mock_client._execute_action.called

    def test_page_action_history(self, mock_env_vars, mock_client):
        """Test page.action_history property."""
        from src.browser.e2e_client import BrowserPage, ActionResult, BrowserAction

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        # Add some action history
        page._action_history = [
            ActionResult(True, BrowserAction.ACT, "action1"),
            ActionResult(True, BrowserAction.ACT, "action2"),
        ]

        history = page.action_history

        # Should return a copy
        assert len(history) == 2
        assert history is not page._action_history

    def test_page_get_stats(self, mock_env_vars, mock_client):
        """Test page.get_stats method."""
        from src.browser.e2e_client import BrowserPage, ActionResult, BrowserAction

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        # Add some action history
        page._action_history = [
            ActionResult(True, BrowserAction.ACT, "action1", cached=True, healed=False, tokens_used=100),
            ActionResult(True, BrowserAction.ACT, "action2", cached=False, healed=True, tokens_used=150),
            ActionResult(True, BrowserAction.ACT, "action3", cached=True, healed=False, tokens_used=100),
        ]

        stats = page.get_stats()

        assert stats["total_actions"] == 3
        assert stats["cached_actions"] == 2
        assert stats["cache_hit_rate"] == pytest.approx(2 / 3)
        assert stats["healed_actions"] == 1
        assert stats["heal_rate"] == pytest.approx(1 / 3)
        assert stats["total_tokens"] == 350

    def test_page_get_stats_empty(self, mock_env_vars, mock_client):
        """Test page.get_stats with no actions."""
        from src.browser.e2e_client import BrowserPage

        page = BrowserPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )

        stats = page.get_stats()

        assert stats["total_actions"] == 0
        assert stats["cache_hit_rate"] == 0
        assert stats["heal_rate"] == 0


class TestE2EBrowserClient:
    """Tests for E2EBrowserClient class."""

    def test_client_initialization(self, mock_env_vars, monkeypatch):
        """Test E2EBrowserClient initialization."""
        monkeypatch.setenv("E2E_WORKER_URL", "https://test-worker.workers.dev")

        from src.browser.e2e_client import E2EBrowserClient

        client = E2EBrowserClient()

        assert client.endpoint == "https://test-worker.workers.dev"
        assert client.cache_enabled is True
        assert client.self_healing_enabled is True

    def test_client_initialization_with_params(self, mock_env_vars):
        """Test client initialization with parameters."""
        from src.browser.e2e_client import E2EBrowserClient

        client = E2EBrowserClient(
            endpoint="https://custom-worker.workers.dev",
            api_token="test-token",
            cache_enabled=False,
            self_healing_enabled=False,
            timeout_seconds=120,
        )

        assert client.endpoint == "https://custom-worker.workers.dev"
        assert client.api_token == "test-token"
        assert client.cache_enabled is False
        assert client.self_healing_enabled is False
        assert client.timeout_seconds == 120

    def test_client_resolve_endpoint_error(self, mock_env_vars, monkeypatch):
        """Test error when no endpoint configured."""
        monkeypatch.delenv("E2E_WORKER_URL", raising=False)
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("STAGEHAND_WORKER_URL", raising=False)

        from src.browser.e2e_client import E2EBrowserClient

        # Need to also mock get_settings to return None for cloudflare_account_id
        mock_settings = MagicMock()
        mock_settings.cloudflare_account_id = None

        with patch("src.browser.e2e_client.get_settings", return_value=mock_settings):
            with pytest.raises(ValueError, match="E2E Browser Worker endpoint not configured"):
                E2EBrowserClient()

    def test_client_get_headers_with_token(self, mock_env_vars):
        """Test header generation with API token."""
        from src.browser.e2e_client import E2EBrowserClient

        client = E2EBrowserClient(
            endpoint="https://test-worker.workers.dev",
            api_token="test-token",
        )

        headers = client._get_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-token"

    def test_client_get_headers_no_token(self, mock_env_vars):
        """Test headers without API token."""
        from src.browser.e2e_client import E2EBrowserClient

        client = E2EBrowserClient(endpoint="https://test-worker.workers.dev")

        headers = client._get_headers()

        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_client_connect(self, mock_env_vars, mock_httpx_client):
        """Test client connection."""
        from src.browser.e2e_client import E2EBrowserClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        client = E2EBrowserClient(endpoint="https://test-worker.workers.dev")

        with patch("httpx.AsyncClient", return_value=mock_httpx_client):
            await client._connect()

            assert client._connected is True
            mock_httpx_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_connect_health_check_failure(self, mock_env_vars):
        """Test connection failure on health check."""
        from src.browser.e2e_client import E2EBrowserClient

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client.get = AsyncMock(return_value=mock_response)

        client = E2EBrowserClient(endpoint="https://test-worker.workers.dev")

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ConnectionError, match="health check failed"):
                await client._connect()

    @pytest.mark.asyncio
    async def test_client_connect_network_error(self, mock_env_vars):
        """Test connection network error."""
        from src.browser.e2e_client import E2EBrowserClient

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("Connection failed"))

        client = E2EBrowserClient(endpoint="https://test-worker.workers.dev")

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                await client._connect()

    @pytest.mark.asyncio
    async def test_client_disconnect(self, mock_env_vars):
        """Test client disconnection."""
        from src.browser.e2e_client import E2EBrowserClient

        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        client = E2EBrowserClient(endpoint="https://test-worker.workers.dev")
        client._http_client = mock_client
        client._connected = True

        await client._disconnect()

        assert client._connected is False
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_new_page(self, mock_env_vars, mock_httpx_client):
        """Test creating a new page."""
        from src.browser.e2e_client import E2EBrowserClient, BrowserPage

        client = E2EBrowserClient(endpoint="https://test-worker.workers.dev")
        client._connected = True
        client._http_client = mock_httpx_client

        page = await client.new_page("https://example.com")

        assert isinstance(page, BrowserPage)
        assert page.url == "https://example.com"
        assert page.page_id in client._pages

    @pytest.mark.asyncio
    async def test_client_new_page_not_connected(self, mock_env_vars):
        """Test new_page when not connected."""
        from src.browser.e2e_client import E2EBrowserClient

        client = E2EBrowserClient(endpoint="https://test-worker.workers.dev")

        with pytest.raises(RuntimeError, match="not connected"):
            await client.new_page("https://example.com")

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_env_vars, mock_httpx_client):
        """Test async context manager."""
        from src.browser.e2e_client import E2EBrowserClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_httpx_client):
            async with E2EBrowserClient(endpoint="https://test-worker.workers.dev") as client:
                assert client._connected is True

            # After exit, should be disconnected
            assert client._connected is False


class TestE2EBrowserClientExecuteAction:
    """Tests for E2EBrowserClient._execute_action method."""

    @pytest.fixture
    def connected_client(self, mock_env_vars, mock_httpx_client):
        """Create a connected E2EBrowserClient."""
        from src.browser.e2e_client import E2EBrowserClient, BrowserPage

        client = E2EBrowserClient(endpoint="https://test-worker.workers.dev")
        client._connected = True
        client._http_client = mock_httpx_client

        # Create a page
        page = BrowserPage(client=client, page_id="test-page", url="https://example.com")
        client._pages["test-page"] = page

        return client

    @pytest.mark.asyncio
    async def test_execute_action_act(self, connected_client, mock_httpx_client):
        """Test execute action for ACT."""
        from src.browser.e2e_client import BrowserAction

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "steps": [{"success": True, "instruction": "Click login"}],
            "stats": {"cachedActions": 1, "healedActions": 0},
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        result = await connected_client._execute_action(
            page_id="test-page",
            action=BrowserAction.ACT,
            instruction="Click the login button",
        )

        assert result["success"] is True
        assert result["cached"] is True
        mock_httpx_client.post.assert_called()

    @pytest.mark.asyncio
    async def test_execute_action_extract(self, connected_client, mock_httpx_client):
        """Test execute action for EXTRACT."""
        from src.browser.e2e_client import BrowserAction

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {"title": "Product"},
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        result = await connected_client._execute_action(
            page_id="test-page",
            action=BrowserAction.EXTRACT,
            instruction="Extract title",
            schema={"type": "object", "properties": {"title": {"type": "string"}}},
        )

        assert result["success"] is True
        assert result["data"]["title"] == "Product"

    @pytest.mark.asyncio
    async def test_execute_action_observe(self, connected_client, mock_httpx_client):
        """Test execute action for OBSERVE."""
        from src.browser.e2e_client import BrowserAction

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "observation": "I see a form",
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        result = await connected_client._execute_action(
            page_id="test-page",
            action=BrowserAction.OBSERVE,
            instruction="What is visible?",
        )

        assert result["success"] is True
        assert "form" in result["observation"]

    @pytest.mark.asyncio
    async def test_execute_action_no_url(self, connected_client):
        """Test execute action with no URL available."""
        from src.browser.e2e_client import BrowserAction

        result = await connected_client._execute_action(
            page_id="non-existent-page",
            action=BrowserAction.ACT,
            instruction="Click",
        )

        assert result["success"] is False
        assert "No URL" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_action_timeout(self, connected_client, mock_httpx_client):
        """Test execute action timeout."""
        from src.browser.e2e_client import BrowserAction

        mock_httpx_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        result = await connected_client._execute_action(
            page_id="test-page",
            action=BrowserAction.ACT,
            instruction="Click button",
            timeout_ms=5000,
        )

        assert result["success"] is False
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_action_request_error(self, connected_client, mock_httpx_client):
        """Test execute action request error."""
        from src.browser.e2e_client import BrowserAction

        mock_httpx_client.post = AsyncMock(
            side_effect=httpx.RequestError("Connection failed")
        )

        result = await connected_client._execute_action(
            page_id="test-page",
            action=BrowserAction.ACT,
            instruction="Click button",
        )

        assert result["success"] is False
        assert "Request failed" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_action_http_error(self, connected_client, mock_httpx_client):
        """Test execute action HTTP error."""
        from src.browser.e2e_client import BrowserAction

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        result = await connected_client._execute_action(
            page_id="test-page",
            action=BrowserAction.ACT,
            instruction="Click button",
        )

        assert result["success"] is False


class TestE2EBrowserClientHelperMethods:
    """Tests for E2EBrowserClient helper methods."""

    @pytest.fixture
    def connected_client(self, mock_env_vars, mock_httpx_client):
        """Create a connected E2EBrowserClient with a page."""
        from src.browser.e2e_client import E2EBrowserClient, BrowserPage

        client = E2EBrowserClient(endpoint="https://test-worker.workers.dev")
        client._connected = True
        client._http_client = mock_httpx_client

        page = BrowserPage(client=client, page_id="test-page", url="https://example.com")
        client._pages["test-page"] = page

        return client

    @pytest.mark.asyncio
    async def test_get_page_state(self, connected_client, mock_httpx_client):
        """Test _get_page_state method."""
        from src.browser.e2e_client import BrowserAction

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "observation": "Page has login form",
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        result = await connected_client._get_page_state("test-page")

        assert result["url"] == "https://example.com"
        assert "login form" in result["visible_text"]

    @pytest.mark.asyncio
    async def test_get_page_state_no_page(self, connected_client):
        """Test _get_page_state with non-existent page."""
        result = await connected_client._get_page_state("non-existent")

        assert result == {}

    @pytest.mark.asyncio
    async def test_capture_screenshot(self, connected_client, mock_httpx_client):
        """Test _capture_screenshot method."""
        import base64

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "screenshot": base64.b64encode(b"screenshot_data").decode(),
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        result = await connected_client._capture_screenshot("test-page")

        assert result == b"screenshot_data"

    @pytest.mark.asyncio
    async def test_capture_screenshot_no_page(self, connected_client):
        """Test _capture_screenshot with non-existent page."""
        result = await connected_client._capture_screenshot("non-existent")

        assert result == b""

    @pytest.mark.asyncio
    async def test_capture_screenshot_error(self, connected_client, mock_httpx_client):
        """Test _capture_screenshot handles errors."""
        mock_httpx_client.post = AsyncMock(side_effect=Exception("Failed"))

        result = await connected_client._capture_screenshot("test-page")

        assert result == b""


class TestE2EBrowserClientRunTest:
    """Tests for E2EBrowserClient.run_test method."""

    @pytest.fixture
    def connected_client(self, mock_env_vars, mock_httpx_client):
        """Create a connected E2EBrowserClient."""
        from src.browser.e2e_client import E2EBrowserClient

        client = E2EBrowserClient(endpoint="https://test-worker.workers.dev")
        client._connected = True
        client._http_client = mock_httpx_client
        return client

    @pytest.mark.asyncio
    async def test_run_test_success(self, connected_client, mock_httpx_client):
        """Test run_test success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "steps": [
                {"instruction": "Click login", "success": True},
                {"instruction": "Fill email", "success": True},
            ],
            "stats": {"totalDuration": 2000},
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        result = await connected_client.run_test(
            url="https://example.com",
            steps=["Click login", "Fill email"],
        )

        assert result["success"] is True
        assert len(result["steps"]) == 2

    @pytest.mark.asyncio
    async def test_run_test_with_extract(self, connected_client, mock_httpx_client):
        """Test run_test with extraction schema."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "steps": [],
            "extracted": {"title": "Product"},
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        result = await connected_client.run_test(
            url="https://example.com",
            steps=["Click product"],
            extract_schema={"title": "string"},
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_run_test_with_screenshot(self, connected_client, mock_httpx_client):
        """Test run_test with screenshot capture."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "steps": [],
            "screenshot": "base64_image",
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        result = await connected_client.run_test(
            url="https://example.com",
            steps=["Click button"],
            screenshot=True,
        )

        # Verify screenshot param was passed
        call_args = mock_httpx_client.post.call_args
        assert call_args.kwargs["json"]["screenshot"] is True

    @pytest.mark.asyncio
    async def test_run_test_timeout(self, connected_client, mock_httpx_client):
        """Test run_test handles timeout."""
        mock_httpx_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        result = await connected_client.run_test(
            url="https://example.com",
            steps=["Click button"],
        )

        assert result["success"] is False
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_run_test_request_error(self, connected_client, mock_httpx_client):
        """Test run_test handles request error."""
        mock_httpx_client.post = AsyncMock(
            side_effect=httpx.RequestError("Connection failed")
        )

        result = await connected_client.run_test(
            url="https://example.com",
            steps=["Click button"],
        )

        assert result["success"] is False


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_run_test_with_e2e_client(self, mock_env_vars, mock_httpx_client):
        """Test run_test_with_e2e_client helper."""
        from src.browser.e2e_client import run_test_with_e2e_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "steps": [{"instruction": "test", "success": True}],
            "stats": {},
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_httpx_client):
            result = await run_test_with_e2e_client(
                url="https://example.com",
                steps=["Click button"],
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_run_test_with_e2e_client_extraction(self, mock_env_vars, mock_httpx_client):
        """Test run_test_with_e2e_client with extraction."""
        from src.browser.e2e_client import run_test_with_e2e_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "steps": [{"instruction": "test", "success": True}],
            "data": {"name": "Product"},
            "stats": {},
        }
        mock_httpx_client.post = AsyncMock(return_value=mock_response)
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_httpx_client):
            result = await run_test_with_e2e_client(
                url="https://example.com",
                steps=["Click product"],
                extract_schema={"name": "string"},
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_run_test_with_e2e_client_step_failure(self, mock_env_vars, mock_httpx_client):
        """Test run_test_with_e2e_client stops on step failure."""
        from src.browser.e2e_client import run_test_with_e2e_client

        # First call is health check (success), subsequent calls for steps
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200

        mock_step_response = MagicMock()
        mock_step_response.status_code = 200
        mock_step_response.json.return_value = {
            "success": False,
            "steps": [{"instruction": "test", "success": False, "error": "Element not found"}],
            "stats": {},
        }

        mock_httpx_client.get = AsyncMock(return_value=mock_health_response)
        mock_httpx_client.post = AsyncMock(return_value=mock_step_response)

        with patch("httpx.AsyncClient", return_value=mock_httpx_client):
            result = await run_test_with_e2e_client(
                url="https://example.com",
                steps=["Click login", "Fill email"],  # Second step won't run
            )

            assert result["success"] is False
