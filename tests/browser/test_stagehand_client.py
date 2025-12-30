"""Tests for the Stagehand client module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


class TestStagehandAction:
    """Tests for StagehandAction enum."""

    def test_stagehand_actions(self, mock_env_vars):
        """Test StagehandAction enum values."""
        from src.browser.stagehand_client import StagehandAction
        
        assert StagehandAction.ACT.value == "act"
        assert StagehandAction.EXTRACT.value == "extract"
        assert StagehandAction.OBSERVE.value == "observe"


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_action_result_success(self, mock_env_vars):
        """Test creating a successful ActionResult."""
        from src.browser.stagehand_client import ActionResult, StagehandAction
        
        result = ActionResult(
            success=True,
            action=StagehandAction.ACT,
            instruction="Click button",
            result={"clicked": True},
            cached=True,
            healed=False,
            duration_ms=100,
        )
        
        assert result.success is True
        assert result.cached is True
        assert result.healed is False

    def test_action_result_failure(self, mock_env_vars):
        """Test creating a failed ActionResult."""
        from src.browser.stagehand_client import ActionResult, StagehandAction
        
        result = ActionResult(
            success=False,
            action=StagehandAction.ACT,
            instruction="Click button",
            error="Element not found",
        )
        
        assert result.success is False
        assert result.error == "Element not found"


class TestExtractionSchema:
    """Tests for ExtractionSchema."""

    def test_extraction_schema_to_zod(self, mock_env_vars):
        """Test converting schema to Zod format."""
        from src.browser.stagehand_client import ExtractionSchema
        
        schema = ExtractionSchema(fields={
            "title": "string",
            "price": "number",
            "in_stock": "boolean",
            "tags": "array",
        })
        
        zod_schema = schema.to_zod_schema()
        
        assert zod_schema["type"] == "object"
        assert "properties" in zod_schema
        assert zod_schema["properties"]["title"]["type"] == "string"
        assert zod_schema["properties"]["price"]["type"] == "number"
        assert zod_schema["properties"]["in_stock"]["type"] == "boolean"
        assert zod_schema["properties"]["tags"]["type"] == "array"


class TestStagehandPage:
    """Tests for StagehandPage class."""

    @pytest.fixture
    def mock_client(self, mock_env_vars):
        """Create mock StagehandClient."""
        client = MagicMock()
        client._execute_action = AsyncMock(return_value={
            "success": True,
            "action_taken": "clicked",
            "cached": False,
            "healed": False,
        })
        client._get_page_state = AsyncMock(return_value={
            "url": "https://example.com",
            "title": "Test Page",
        })
        client._capture_screenshot = AsyncMock(return_value=b"screenshot_data")
        return client

    @pytest.mark.asyncio
    async def test_page_act(self, mock_env_vars, mock_client):
        """Test page.act method."""
        from src.browser.stagehand_client import StagehandPage
        
        page = StagehandPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )
        
        result = await page.act("Click the login button")
        
        assert result.success is True
        assert result.instruction == "Click the login button"
        mock_client._execute_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_page_act_failure(self, mock_env_vars, mock_client):
        """Test page.act handling failure."""
        from src.browser.stagehand_client import StagehandPage
        
        mock_client._execute_action = AsyncMock(side_effect=Exception("Network error"))
        
        page = StagehandPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )
        
        result = await page.act("Click button")
        
        assert result.success is False
        assert "Network error" in result.error

    @pytest.mark.asyncio
    async def test_page_extract(self, mock_env_vars, mock_client):
        """Test page.extract method."""
        from src.browser.stagehand_client import StagehandPage, StagehandAction
        
        mock_client._execute_action = AsyncMock(return_value={
            "success": True,
            "data": {"title": "Product", "price": 99.99},
        })
        
        page = StagehandPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )
        
        result = await page.extract({"title": "string", "price": "number"})
        
        assert result.success is True
        assert result.result == {"title": "Product", "price": 99.99}

    @pytest.mark.asyncio
    async def test_page_observe(self, mock_env_vars, mock_client):
        """Test page.observe method."""
        from src.browser.stagehand_client import StagehandPage
        
        mock_client._execute_action = AsyncMock(return_value={
            "success": True,
            "observation": "I see a login form with email and password fields",
        })
        
        page = StagehandPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )
        
        result = await page.observe("What forms are visible?")
        
        assert result.success is True
        assert "login form" in result.result

    def test_page_get_stats(self, mock_env_vars, mock_client):
        """Test page statistics collection."""
        from src.browser.stagehand_client import StagehandPage, ActionResult, StagehandAction
        
        page = StagehandPage(
            client=mock_client,
            page_id="test-page-id",
            url="https://example.com",
        )
        
        # Add some action history
        page._action_history = [
            ActionResult(True, StagehandAction.ACT, "action1", cached=True, healed=False, tokens_used=100),
            ActionResult(True, StagehandAction.ACT, "action2", cached=False, healed=True, tokens_used=150),
            ActionResult(True, StagehandAction.ACT, "action3", cached=True, healed=False, tokens_used=100),
        ]
        
        stats = page.get_stats()
        
        assert stats["total_actions"] == 3
        assert stats["cached_actions"] == 2
        assert stats["healed_actions"] == 1
        assert stats["total_tokens"] == 350


class TestStagehandClient:
    """Tests for StagehandClient class."""

    def test_client_initialization(self, mock_env_vars):
        """Test StagehandClient initialization."""
        from src.browser.stagehand_client import StagehandClient
        
        client = StagehandClient(
            endpoint="https://test-worker.workers.dev",
            api_token="test-token",
        )
        
        assert client.endpoint == "https://test-worker.workers.dev"
        assert client.api_token == "test-token"

    def test_client_get_headers(self, mock_env_vars):
        """Test header generation."""
        from src.browser.stagehand_client import StagehandClient
        
        client = StagehandClient(
            endpoint="https://test-worker.workers.dev",
            api_token="test-token",
        )
        
        headers = client._get_headers()
        
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-token"

    def test_client_get_headers_no_token(self, mock_env_vars):
        """Test headers without API token."""
        from src.browser.stagehand_client import StagehandClient
        
        client = StagehandClient(
            endpoint="https://test-worker.workers.dev",
        )
        
        headers = client._get_headers()
        
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_client_connect(self, mock_env_vars, mock_httpx_client):
        """Test client connection."""
        from src.browser.stagehand_client import StagehandClient
        
        client = StagehandClient(
            endpoint="https://test-worker.workers.dev",
        )
        
        with patch('httpx.AsyncClient', return_value=mock_httpx_client):
            await client._connect()
            
            assert client._connected is True
            mock_httpx_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_connect_failure(self, mock_env_vars):
        """Test client connection failure."""
        from src.browser.stagehand_client import StagehandClient
        
        client = StagehandClient(
            endpoint="https://test-worker.workers.dev",
        )
        
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx.RequestError("Connection failed"))
        
        with patch('httpx.AsyncClient', return_value=mock_http):
            with pytest.raises(ConnectionError):
                await client._connect()

    @pytest.mark.asyncio
    async def test_client_new_page(self, mock_env_vars, mock_httpx_client):
        """Test creating a new page."""
        from src.browser.stagehand_client import StagehandClient, StagehandPage
        
        client = StagehandClient(
            endpoint="https://test-worker.workers.dev",
        )
        client._connected = True
        client._http_client = mock_httpx_client
        
        page = await client.new_page("https://example.com")
        
        assert isinstance(page, StagehandPage)
        assert page.url == "https://example.com"

    @pytest.mark.asyncio
    async def test_client_new_page_not_connected(self, mock_env_vars):
        """Test new_page when not connected."""
        from src.browser.stagehand_client import StagehandClient
        
        client = StagehandClient(
            endpoint="https://test-worker.workers.dev",
        )
        
        with pytest.raises(RuntimeError, match="not connected"):
            await client.new_page("https://example.com")

    @pytest.mark.asyncio
    async def test_client_run_test(self, mock_env_vars, mock_httpx_client):
        """Test run_test method."""
        from src.browser.stagehand_client import StagehandClient
        
        mock_httpx_client.post = AsyncMock(return_value=MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "success": True,
                "steps": [
                    {"instruction": "Click login", "success": True},
                ],
                "stats": {"totalDuration": 1000},
            }),
        ))
        
        client = StagehandClient(
            endpoint="https://test-worker.workers.dev",
        )
        client._connected = True
        client._http_client = mock_httpx_client
        
        result = await client.run_test(
            url="https://example.com",
            steps=["Click the login button"],
        )
        
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_env_vars, mock_httpx_client):
        """Test async context manager."""
        from src.browser.stagehand_client import StagehandClient
        
        with patch('httpx.AsyncClient', return_value=mock_httpx_client):
            async with StagehandClient(endpoint="https://test-worker.workers.dev") as client:
                assert client._connected is True
            
            # After exit, should be disconnected
            assert client._connected is False


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_run_test_with_stagehand(self, mock_env_vars, mock_httpx_client):
        """Test run_test_with_stagehand helper."""
        from src.browser.stagehand_client import run_test_with_stagehand
        
        mock_httpx_client.post = AsyncMock(return_value=MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "success": True,
                "steps": [{"instruction": "test", "success": True}],
                "stats": {},
            }),
        ))
        
        with patch('httpx.AsyncClient', return_value=mock_httpx_client):
            result = await run_test_with_stagehand(
                url="https://example.com",
                steps=["Click button"],
            )
            
            assert result["success"] is True
