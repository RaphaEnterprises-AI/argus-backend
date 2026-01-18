"""Chrome Extension Bridge for E2E Testing Agent.

This module provides a WebSocket server that connects to the E2E Testing Agent
Chrome extension, enabling browser automation through a real browser instance.

Benefits over Playwright/Selenium:
- Works with user's existing browser session (cookies, auth, extensions)
- No bot detection (real user agent, real browser)
- Direct DOM access with console log capture
- Framework-agnostic - works with any web app
- Same approach as Claude in Chrome / Google Antigravity

Architecture:
    Python Agent ←→ WebSocket Server ←→ Chrome Extension ←→ Web Page

Usage:
    async with ExtensionBridge() as bridge:
        await bridge.navigate("https://example.com")
        await bridge.click("#login-button")
        await bridge.fill("#email", "test@example.com")
        screenshot = await bridge.screenshot()

References:
    - Claude in Chrome: https://www.anthropic.com/news/claude-for-chrome
    - Google Antigravity: https://developers.googleblog.com/build-with-google-antigravity-our-new-agentic-development-platform/
"""

import asyncio
import base64
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

try:
    import websockets
    from websockets.server import serve
except ImportError:
    websockets = None
    serve = None

from .browser_abstraction import ActionResult, AutomationFramework, BrowserAutomation, BrowserConfig

logger = structlog.get_logger()


@dataclass
class ExtensionMessage:
    """Message to/from the Chrome extension."""

    action: str
    requestId: str = field(default_factory=lambda: str(uuid.uuid4()))
    tabId: int | None = None
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "requestId": self.requestId,
            "tabId": self.tabId,
            "params": self.params,
        }


@dataclass
class ExtensionResponse:
    """Response from the Chrome extension."""

    requestId: str
    success: bool
    data: Any = None
    error: str | None = None


class ExtensionBridge(BrowserAutomation):
    """Browser automation via Chrome Extension.

    This provides the same interface as PlaywrightAutomation but uses
    a Chrome extension for browser control. This has several advantages:

    1. **Real Browser Session**: Uses the user's existing Chrome with all
       cookies, authentication, and extensions intact.

    2. **No Bot Detection**: Unlike Playwright/Selenium, this is a real
       browser that websites cannot distinguish from a human user.

    3. **Direct DOM Access**: The extension can read and manipulate the
       DOM directly, capture console logs, and monitor network requests.

    4. **Framework Agnostic**: Works with any web application regardless
       of the frontend framework used.

    Similar to:
    - Claude in Chrome (Anthropic)
    - Google Antigravity browser extension
    """

    def __init__(
        self,
        config: BrowserConfig | None = None,
        port: int = 8765,
        host: str = "localhost",
    ):
        """Initialize the extension bridge.

        Args:
            config: Browser configuration
            port: WebSocket server port
            host: WebSocket server host
        """
        super().__init__(config or BrowserConfig(framework=AutomationFramework.CUSTOM))
        self.port = port
        self.host = host
        self._server = None
        self._connections: set = set()
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._active_tab_id: int | None = None
        self._console_logs: list[dict] = []
        self._event_handlers: dict[str, list[Callable]] = {}
        self.log = logger.bind(component="extension_bridge")

    async def start(self) -> None:
        """Start the WebSocket server and wait for extension connection."""
        if websockets is None:
            raise ImportError(
                "websockets package required. Install with: pip install websockets"
            )

        self._server = await serve(
            self._handle_connection,
            self.host,
            self.port,
        )

        self.log.info(
            "Extension bridge started",
            host=self.host,
            port=self.port,
        )

        # Wait for initial connection (with timeout)
        try:
            await asyncio.wait_for(
                self._wait_for_connection(),
                timeout=30.0,
            )
            self.log.info("Chrome extension connected")
        except TimeoutError:
            self.log.warning(
                "No extension connection within timeout. "
                "Make sure the E2E Testing Agent extension is installed and enabled."
            )

    async def _wait_for_connection(self) -> None:
        """Wait for at least one extension to connect."""
        while not self._connections:
            await asyncio.sleep(0.1)

    async def _handle_connection(self, websocket) -> None:
        """Handle a WebSocket connection from the extension."""
        self._connections.add(websocket)
        self.log.info("Extension connected", remote=websocket.remote_address)

        try:
            async for message in websocket:
                await self._handle_message(json.loads(message))
        except Exception as e:
            self.log.error("Connection error", error=str(e))
        finally:
            self._connections.discard(websocket)
            self.log.info("Extension disconnected")

    async def _handle_message(self, message: dict) -> None:
        """Handle a message from the extension."""
        msg_type = message.get("type")

        if msg_type == "connected":
            self.log.info("Extension ready", capabilities=message.get("capabilities"))

        elif msg_type == "response":
            request_id = message.get("requestId")
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                future.set_result(
                    ExtensionResponse(
                        requestId=request_id,
                        success=message.get("success", False),
                        data=message.get("data"),
                        error=message.get("error"),
                    )
                )

        elif msg_type == "consoleLog":
            log_entry = {
                "tabId": message.get("tabId"),
                "level": message.get("level"),
                "args": message.get("args"),
                "timestamp": message.get("timestamp"),
            }
            self._console_logs.append(log_entry)

            # Trigger event handlers
            for handler in self._event_handlers.get("consoleLog", []):
                try:
                    await handler(log_entry)
                except Exception as e:
                    self.log.warning("Console log handler error", error=str(e))

        elif msg_type == "error":
            self.log.error("Extension error", error=message.get("error"))

    async def _send_command(
        self,
        action: str,
        params: dict = None,
        tab_id: int | None = None,
        timeout: float = 30.0,
    ) -> ExtensionResponse:
        """Send a command to the extension and wait for response."""
        if not self._connections:
            raise RuntimeError("No extension connected")

        message = ExtensionMessage(
            action=action,
            tabId=tab_id or self._active_tab_id,
            params=params or {},
        )

        # Create future for response
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[message.requestId] = future

        # Send to all connections (usually just one)
        data = json.dumps(message.to_dict())
        await asyncio.gather(
            *[ws.send(data) for ws in self._connections],
            return_exceptions=True,
        )

        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout)
            return response
        except TimeoutError:
            self._pending_requests.pop(message.requestId, None)
            raise TimeoutError(f"Command '{action}' timed out after {timeout}s")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self.log.info("Extension bridge stopped")

    # BrowserAutomation interface implementation

    async def goto(self, url: str, wait_until: str = "load") -> ActionResult:
        start = time.time()
        try:
            response = await self._send_command(
                "navigate",
                {"url": url, "waitUntil": wait_until},
            )

            if response.success:
                self._active_tab_id = response.data.get("tabId")

            return ActionResult(
                success=response.success,
                action="goto",
                duration_ms=int((time.time() - start) * 1000),
                error=response.error,
                data=response.data,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="goto",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def click(self, selector: str, timeout_ms: int | None = None) -> ActionResult:
        start = time.time()
        try:
            response = await self._send_command(
                "click",
                {"selector": selector},
                timeout=(timeout_ms or self.config.timeout_ms) / 1000,
            )
            return ActionResult(
                success=response.success,
                action="click",
                duration_ms=int((time.time() - start) * 1000),
                error=response.error,
                data=response.data,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="click",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def fill(self, selector: str, value: str) -> ActionResult:
        start = time.time()
        try:
            response = await self._send_command(
                "fill",
                {"selector": selector, "value": value},
            )
            return ActionResult(
                success=response.success,
                action="fill",
                duration_ms=int((time.time() - start) * 1000),
                error=response.error,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="fill",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def type_text(self, selector: str, text: str, delay_ms: int = 50) -> ActionResult:
        start = time.time()
        try:
            response = await self._send_command(
                "type",
                {"selector": selector, "text": text, "delay": delay_ms},
            )
            return ActionResult(
                success=response.success,
                action="type",
                duration_ms=int((time.time() - start) * 1000),
                error=response.error,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="type",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def screenshot(self, full_page: bool = False) -> bytes:
        response = await self._send_command(
            "screenshot",
            {"fullPage": full_page},
        )

        if response.success and response.data:
            # Decode base64 data URL
            data_url = response.data.get("dataUrl", "")
            if data_url.startswith("data:image/png;base64,"):
                return base64.b64decode(data_url.split(",")[1])

        raise RuntimeError(f"Screenshot failed: {response.error}")

    async def get_text(self, selector: str) -> str:
        response = await self._send_command(
            "querySelector",
            {"selector": selector},
        )

        if response.success and response.data:
            return response.data.get("textContent", "")

        raise RuntimeError(f"Get text failed: {response.error}")

    async def is_visible(self, selector: str) -> bool:
        response = await self._send_command(
            "querySelector",
            {"selector": selector},
        )

        if response.success and response.data:
            return response.data.get("isVisible", False)

        return False

    async def wait_for_selector(self, selector: str, timeout_ms: int | None = None) -> ActionResult:
        start = time.time()
        try:
            response = await self._send_command(
                "waitForSelector",
                {"selector": selector, "timeout": timeout_ms or self.config.timeout_ms},
                timeout=(timeout_ms or self.config.timeout_ms) / 1000 + 1,
            )
            return ActionResult(
                success=response.success,
                action="wait_for_selector",
                duration_ms=int((time.time() - start) * 1000),
                error=response.error,
                data=response.data,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="wait_for_selector",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def get_current_url(self) -> str:
        response = await self._send_command("getPageInfo")

        if response.success and response.data:
            return response.data.get("url", "")

        raise RuntimeError(f"Get URL failed: {response.error}")

    async def hover(self, selector: str) -> ActionResult:
        start = time.time()
        response = await self._send_command("hover", {"selector": selector})
        return ActionResult(
            success=response.success,
            action="hover",
            duration_ms=int((time.time() - start) * 1000),
            error=response.error,
        )

    async def select_option(self, selector: str, value: str) -> ActionResult:
        start = time.time()
        response = await self._send_command("select", {"selector": selector, "value": value})
        return ActionResult(
            success=response.success,
            action="select",
            duration_ms=int((time.time() - start) * 1000),
            error=response.error,
        )

    async def press_key(self, key: str) -> ActionResult:
        start = time.time()
        response = await self._send_command("pressKey", {"key": key})
        return ActionResult(
            success=response.success,
            action="press_key",
            duration_ms=int((time.time() - start) * 1000),
            error=response.error,
        )

    async def scroll(self, x: int = 0, y: int = 0) -> ActionResult:
        start = time.time()
        response = await self._send_command("scroll", {"x": x, "y": y})
        return ActionResult(
            success=response.success,
            action="scroll",
            duration_ms=int((time.time() - start) * 1000),
            error=response.error,
        )

    async def evaluate(self, script: str) -> Any:
        response = await self._send_command("evaluate", {"script": script})

        if response.success:
            return response.data

        raise RuntimeError(f"Evaluate failed: {response.error}")

    # Extension-specific methods

    async def get_console_logs(self, clear: bool = True) -> list[dict]:
        """Get captured console logs.

        Args:
            clear: Whether to clear logs after retrieval

        Returns:
            List of console log entries
        """
        logs = self._console_logs.copy()
        if clear:
            self._console_logs.clear()
        return logs

    async def create_tab(self, url: str | None = None) -> int:
        """Create a new browser tab.

        Args:
            url: Optional URL to open in the new tab

        Returns:
            Tab ID
        """
        response = await self._send_command("createTab", {"url": url})

        if response.success:
            return response.data.get("tabId")

        raise RuntimeError(f"Create tab failed: {response.error}")

    async def close_tab(self, tab_id: int | None = None) -> None:
        """Close a browser tab.

        Args:
            tab_id: Tab ID to close (defaults to active tab)
        """
        await self._send_command("closeTab", tab_id=tab_id or self._active_tab_id)

    async def get_tabs(self) -> list[dict]:
        """Get all open tabs.

        Returns:
            List of tab info dicts
        """
        response = await self._send_command("getTabs")

        if response.success:
            return response.data

        return []

    def on_console_log(self, handler: Callable) -> None:
        """Register a handler for console log events.

        Args:
            handler: Async function to call with log entries
        """
        if "consoleLog" not in self._event_handlers:
            self._event_handlers["consoleLog"] = []
        self._event_handlers["consoleLog"].append(handler)

    async def get_page_info(self) -> dict:
        """Get information about the current page.

        Returns:
            Dict with url, title, readyState, etc.
        """
        response = await self._send_command("getPageInfo")

        if response.success:
            return response.data

        raise RuntimeError(f"Get page info failed: {response.error}")

    async def query_selector(self, selector: str) -> dict | None:
        """Query for an element.

        Args:
            selector: CSS selector

        Returns:
            Element info dict or None
        """
        response = await self._send_command("querySelector", {"selector": selector})

        if response.success:
            return response.data

        return None

    async def query_selector_all(self, selector: str) -> list[dict]:
        """Query for all matching elements.

        Args:
            selector: CSS selector

        Returns:
            List of element info dicts
        """
        response = await self._send_command("querySelectorAll", {"selector": selector})

        if response.success:
            return response.data

        return []


# Convenience function
async def create_extension_bridge(
    port: int = 8765,
    host: str = "localhost",
    config: BrowserConfig | None = None,
) -> ExtensionBridge:
    """Create and start an extension bridge.

    Args:
        port: WebSocket port
        host: WebSocket host
        config: Browser configuration

    Returns:
        Connected ExtensionBridge instance

    Example:
        async with await create_extension_bridge() as browser:
            await browser.goto("https://example.com")
            await browser.click("#login")
    """
    bridge = ExtensionBridge(config=config, port=port, host=host)
    await bridge.start()
    return bridge
