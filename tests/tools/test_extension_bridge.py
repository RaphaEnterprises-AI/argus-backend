"""Tests for extension bridge module."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.browser_abstraction import AutomationFramework, BrowserConfig
from src.tools.extension_bridge import (
    ExtensionBridge,
    ExtensionMessage,
    ExtensionResponse,
    create_extension_bridge,
)


class TestExtensionMessage:
    """Tests for ExtensionMessage dataclass."""

    def test_create_message(self):
        """Test creating an extension message."""
        msg = ExtensionMessage(
            action="click",
            requestId="req_123",
            tabId=1,
            params={"selector": "#button"},
        )
        assert msg.action == "click"
        assert msg.requestId == "req_123"
        assert msg.tabId == 1
        assert msg.params == {"selector": "#button"}

    def test_default_request_id(self):
        """Test that requestId is auto-generated."""
        msg = ExtensionMessage(action="navigate")
        assert msg.requestId is not None
        assert len(msg.requestId) > 0

    def test_to_dict(self):
        """Test converting message to dict."""
        msg = ExtensionMessage(
            action="fill",
            requestId="req_456",
            tabId=2,
            params={"selector": "#input", "value": "test"},
        )
        result = msg.to_dict()

        assert result["action"] == "fill"
        assert result["requestId"] == "req_456"
        assert result["tabId"] == 2
        assert result["params"]["selector"] == "#input"
        assert result["params"]["value"] == "test"


class TestExtensionResponse:
    """Tests for ExtensionResponse dataclass."""

    def test_create_success_response(self):
        """Test creating a success response."""
        resp = ExtensionResponse(
            requestId="req_123",
            success=True,
            data={"url": "https://example.com"},
        )
        assert resp.requestId == "req_123"
        assert resp.success is True
        assert resp.data["url"] == "https://example.com"
        assert resp.error is None

    def test_create_error_response(self):
        """Test creating an error response."""
        resp = ExtensionResponse(
            requestId="req_456",
            success=False,
            error="Element not found",
        )
        assert resp.success is False
        assert resp.error == "Element not found"
        assert resp.data is None


class TestExtensionBridge:
    """Tests for ExtensionBridge class."""

    def test_init_default(self):
        """Test default initialization."""
        bridge = ExtensionBridge()

        assert bridge.port == 8765
        assert bridge.host == "localhost"
        assert bridge._server is None
        assert len(bridge._connections) == 0
        assert len(bridge._pending_requests) == 0
        assert bridge._active_tab_id is None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = BrowserConfig(
            framework=AutomationFramework.CUSTOM,
            timeout_ms=60000,
        )
        bridge = ExtensionBridge(config=config, port=9000, host="0.0.0.0")

        assert bridge.port == 9000
        assert bridge.host == "0.0.0.0"
        assert bridge.config.timeout_ms == 60000

    @pytest.mark.asyncio
    async def test_start_without_websockets(self):
        """Test start raises ImportError when websockets not available."""
        bridge = ExtensionBridge()

        with patch.object(bridge, "_server", None):
            with patch("src.tools.extension_bridge.websockets", None):
                with pytest.raises(ImportError, match="websockets package required"):
                    await bridge.start()

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stopping the bridge."""
        bridge = ExtensionBridge()
        bridge._server = MagicMock()
        bridge._server.close = MagicMock()
        bridge._server.wait_closed = AsyncMock()

        await bridge.stop()

        bridge._server.close.assert_called_once()
        bridge._server.wait_closed.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_no_server(self):
        """Test stopping when server isn't running."""
        bridge = ExtensionBridge()
        await bridge.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_handle_message_connected(self):
        """Test handling connected message."""
        bridge = ExtensionBridge()

        message = {
            "type": "connected",
            "capabilities": ["screenshot", "click"],
        }

        await bridge._handle_message(message)
        # Just verify no error is raised

    @pytest.mark.asyncio
    async def test_handle_message_response(self):
        """Test handling response message."""
        bridge = ExtensionBridge()

        # Create a pending request
        future = asyncio.get_event_loop().create_future()
        bridge._pending_requests["req_123"] = future

        message = {
            "type": "response",
            "requestId": "req_123",
            "success": True,
            "data": {"url": "https://example.com"},
        }

        await bridge._handle_message(message)

        response = future.result()
        assert response.requestId == "req_123"
        assert response.success is True
        assert response.data["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_handle_message_console_log(self):
        """Test handling console log message."""
        bridge = ExtensionBridge()

        message = {
            "type": "consoleLog",
            "tabId": 1,
            "level": "info",
            "args": ["Test log message"],
            "timestamp": 1234567890,
        }

        await bridge._handle_message(message)

        assert len(bridge._console_logs) == 1
        assert bridge._console_logs[0]["level"] == "info"

    @pytest.mark.asyncio
    async def test_handle_message_console_log_with_handler(self):
        """Test handling console log with event handler."""
        bridge = ExtensionBridge()

        handler = AsyncMock()
        bridge.on_console_log(handler)

        message = {
            "type": "consoleLog",
            "tabId": 1,
            "level": "error",
            "args": ["Error occurred"],
            "timestamp": 1234567890,
        }

        await bridge._handle_message(message)

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_error(self):
        """Test handling error message."""
        bridge = ExtensionBridge()

        message = {
            "type": "error",
            "error": "Something went wrong",
        }

        await bridge._handle_message(message)
        # Just verify no error is raised

    @pytest.mark.asyncio
    async def test_send_command_no_connection(self):
        """Test sending command without connection raises error."""
        bridge = ExtensionBridge()

        with pytest.raises(RuntimeError, match="No extension connected"):
            await bridge._send_command("click", {"selector": "#btn"})

    @pytest.mark.asyncio
    async def test_get_console_logs(self):
        """Test getting console logs."""
        bridge = ExtensionBridge()
        bridge._console_logs = [
            {"level": "info", "args": ["Log 1"]},
            {"level": "error", "args": ["Log 2"]},
        ]

        logs = await bridge.get_console_logs(clear=True)

        assert len(logs) == 2
        assert logs[0]["level"] == "info"
        assert len(bridge._console_logs) == 0  # Cleared

    @pytest.mark.asyncio
    async def test_get_console_logs_no_clear(self):
        """Test getting console logs without clearing."""
        bridge = ExtensionBridge()
        bridge._console_logs = [{"level": "info", "args": ["Log 1"]}]

        logs = await bridge.get_console_logs(clear=False)

        assert len(logs) == 1
        assert len(bridge._console_logs) == 1  # Not cleared

    def test_on_console_log(self):
        """Test registering console log handler."""
        bridge = ExtensionBridge()

        async def handler(log):
            pass

        bridge.on_console_log(handler)

        assert "consoleLog" in bridge._event_handlers
        assert handler in bridge._event_handlers["consoleLog"]

    @pytest.mark.asyncio
    async def test_goto_success(self):
        """Test goto action with success."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data={"tabId": 1, "url": "https://example.com"},
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.goto("https://example.com")

        assert result.success is True
        assert result.action == "goto"
        assert bridge._active_tab_id == 1

    @pytest.mark.asyncio
    async def test_goto_failure(self):
        """Test goto action with failure."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}
        bridge._send_command = AsyncMock(side_effect=Exception("Navigation failed"))

        result = await bridge.goto("https://example.com")

        assert result.success is False
        assert "Navigation failed" in result.error

    @pytest.mark.asyncio
    async def test_click_success(self):
        """Test click action with success."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data={"clicked": True},
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.click("#button")

        assert result.success is True
        assert result.action == "click"

    @pytest.mark.asyncio
    async def test_click_failure(self):
        """Test click action with failure."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}
        bridge._send_command = AsyncMock(side_effect=Exception("Click failed"))

        result = await bridge.click("#button")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_fill_success(self):
        """Test fill action with success."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.fill("#input", "test value")

        assert result.success is True
        assert result.action == "fill"

    @pytest.mark.asyncio
    async def test_fill_failure(self):
        """Test fill action with failure."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}
        bridge._send_command = AsyncMock(side_effect=Exception("Fill failed"))

        result = await bridge.fill("#input", "value")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_type_text_success(self):
        """Test type_text action with success."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.type_text("#input", "test", delay_ms=100)

        assert result.success is True
        assert result.action == "type"

    @pytest.mark.asyncio
    async def test_type_text_failure(self):
        """Test type_text action with failure."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}
        bridge._send_command = AsyncMock(side_effect=Exception("Type failed"))

        result = await bridge.type_text("#input", "test")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_screenshot_success(self):
        """Test screenshot action with success."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        test_image = b"fake image data"
        b64_image = base64.b64encode(test_image).decode()

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data={"dataUrl": f"data:image/png;base64,{b64_image}"},
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.screenshot()

        assert result == test_image

    @pytest.mark.asyncio
    async def test_screenshot_failure(self):
        """Test screenshot action with failure."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=False,
            error="Screenshot failed",
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Screenshot failed"):
            await bridge.screenshot()

    @pytest.mark.asyncio
    async def test_get_text_success(self):
        """Test get_text action with success."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data={"textContent": "Hello World"},
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.get_text("#element")

        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_get_text_failure(self):
        """Test get_text action with failure."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=False,
            error="Element not found",
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Get text failed"):
            await bridge.get_text("#element")

    @pytest.mark.asyncio
    async def test_is_visible_true(self):
        """Test is_visible returns True."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data={"isVisible": True},
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.is_visible("#element")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_visible_false(self):
        """Test is_visible returns False when element not found."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=False,
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.is_visible("#element")

        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_selector_success(self):
        """Test wait_for_selector with success."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data={"found": True},
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.wait_for_selector("#element")

        assert result.success is True
        assert result.action == "wait_for_selector"

    @pytest.mark.asyncio
    async def test_wait_for_selector_failure(self):
        """Test wait_for_selector with failure."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}
        bridge._send_command = AsyncMock(side_effect=Exception("Timeout"))

        result = await bridge.wait_for_selector("#element")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_get_current_url_success(self):
        """Test get_current_url with success."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data={"url": "https://example.com/page"},
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.get_current_url()

        assert result == "https://example.com/page"

    @pytest.mark.asyncio
    async def test_get_current_url_failure(self):
        """Test get_current_url with failure."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=False,
            error="Tab not found",
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Get URL failed"):
            await bridge.get_current_url()

    @pytest.mark.asyncio
    async def test_hover_success(self):
        """Test hover action."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(requestId="req_1", success=True)
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.hover("#element")

        assert result.success is True
        assert result.action == "hover"

    @pytest.mark.asyncio
    async def test_select_option_success(self):
        """Test select_option action."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(requestId="req_1", success=True)
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.select_option("#select", "option1")

        assert result.success is True
        assert result.action == "select"

    @pytest.mark.asyncio
    async def test_press_key_success(self):
        """Test press_key action."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(requestId="req_1", success=True)
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.press_key("Enter")

        assert result.success is True
        assert result.action == "press_key"

    @pytest.mark.asyncio
    async def test_scroll_success(self):
        """Test scroll action."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(requestId="req_1", success=True)
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.scroll(x=0, y=500)

        assert result.success is True
        assert result.action == "scroll"

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        """Test evaluate JavaScript."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data={"result": 42},
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.evaluate("return 42")

        assert result == {"result": 42}

    @pytest.mark.asyncio
    async def test_evaluate_failure(self):
        """Test evaluate JavaScript with failure."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=False,
            error="Syntax error",
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Evaluate failed"):
            await bridge.evaluate("invalid javascript")

    @pytest.mark.asyncio
    async def test_create_tab_success(self):
        """Test create_tab action."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data={"tabId": 5},
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.create_tab("https://example.com")

        assert result == 5

    @pytest.mark.asyncio
    async def test_create_tab_failure(self):
        """Test create_tab failure."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=False,
            error="Could not create tab",
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Create tab failed"):
            await bridge.create_tab()

    @pytest.mark.asyncio
    async def test_close_tab(self):
        """Test close_tab action."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}
        bridge._active_tab_id = 3

        mock_response = ExtensionResponse(requestId="req_1", success=True)
        bridge._send_command = AsyncMock(return_value=mock_response)

        await bridge.close_tab()

        bridge._send_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_tabs_success(self):
        """Test get_tabs action."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data=[{"id": 1, "url": "https://example.com"}],
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.get_tabs()

        assert len(result) == 1
        assert result[0]["id"] == 1

    @pytest.mark.asyncio
    async def test_get_tabs_failure(self):
        """Test get_tabs with failure returns empty list."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(requestId="req_1", success=False)
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.get_tabs()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_page_info_success(self):
        """Test get_page_info action."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data={"url": "https://example.com", "title": "Example", "readyState": "complete"},
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.get_page_info()

        assert result["title"] == "Example"
        assert result["readyState"] == "complete"

    @pytest.mark.asyncio
    async def test_get_page_info_failure(self):
        """Test get_page_info with failure."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=False,
            error="Page not accessible",
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Get page info failed"):
            await bridge.get_page_info()

    @pytest.mark.asyncio
    async def test_query_selector_success(self):
        """Test query_selector action."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data={"tagName": "div", "id": "element"},
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.query_selector("#element")

        assert result["tagName"] == "div"

    @pytest.mark.asyncio
    async def test_query_selector_not_found(self):
        """Test query_selector when element not found."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(requestId="req_1", success=False)
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.query_selector("#nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_query_selector_all_success(self):
        """Test query_selector_all action."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(
            requestId="req_1",
            success=True,
            data=[{"tagName": "div"}, {"tagName": "span"}],
        )
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.query_selector_all("div, span")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_query_selector_all_empty(self):
        """Test query_selector_all with no matches."""
        bridge = ExtensionBridge()
        bridge._connections = {MagicMock()}

        mock_response = ExtensionResponse(requestId="req_1", success=False)
        bridge._send_command = AsyncMock(return_value=mock_response)

        result = await bridge.query_selector_all(".nonexistent")

        assert result == []


class TestCreateExtensionBridge:
    """Tests for create_extension_bridge function."""

    @pytest.mark.asyncio
    async def test_create_extension_bridge(self):
        """Test factory function creates and starts bridge."""
        with patch("src.tools.extension_bridge.ExtensionBridge") as MockBridge:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock()
            MockBridge.return_value = mock_instance

            result = await create_extension_bridge(port=9000, host="localhost")

            MockBridge.assert_called_once()
            mock_instance.start.assert_called_once()
            assert result == mock_instance
