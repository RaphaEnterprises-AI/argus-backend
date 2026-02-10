"""Chrome DevTools MCP (Model Context Protocol) integration.

This module provides integration with the Chrome DevTools MCP server for deep
browser debugging and diagnostics. It complements Playwright MCP by providing:

- Network request/response BODIES (not just URLs)
- Source-mapped console stack traces
- Core Web Vitals performance tracing
- Emulation (geolocation, device, network throttling)

The Chrome DevTools MCP server connects to Chrome via CDP (Chrome DevTools Protocol),
giving access to capabilities that Playwright MCP does not expose.

Usage:
    1. Install: npx @anthropic-ai/chrome-devtools-mcp@latest
    2. Use programmatically or configure in MCP settings

See: https://developer.chrome.com/blog/chrome-devtools-mcp
Tool reference: https://anthropic-ai.github.io/chrome-devtools-mcp/tool-reference
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class DevToolsMCPConfig:
    """Configuration for Chrome DevTools MCP server.

    Source: https://www.npmjs.com/package/@anthropic-ai/chrome-devtools-mcp
    """

    command: str = "npx"
    args: list[str] = field(
        default_factory=lambda: ["-y", "@anthropic-ai/chrome-devtools-mcp@latest"]
    )
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class DevToolsToolResult:
    """Result from a Chrome DevTools MCP tool call."""

    success: bool
    content: Any
    error: str | None = None


@dataclass
class NetworkRequest:
    """Parsed network request from DevTools."""

    url: str
    method: str
    status: int | None = None
    response_body: str | None = None
    request_headers: dict[str, str] = field(default_factory=dict)
    response_headers: dict[str, str] = field(default_factory=dict)
    duration_ms: float | None = None
    resource_type: str | None = None


@dataclass
class ConsoleMessage:
    """Parsed console message with source-mapped stack trace."""

    level: str  # log, warn, error, info
    text: str
    url: str | None = None
    line: int | None = None
    column: int | None = None
    stack_trace: str | None = None  # Source-mapped


@dataclass
class PerformanceMetrics:
    """Core Web Vitals and performance data."""

    lcp_ms: float | None = None  # Largest Contentful Paint
    fid_ms: float | None = None  # First Input Delay
    cls: float | None = None  # Cumulative Layout Shift
    fcp_ms: float | None = None  # First Contentful Paint
    ttfb_ms: float | None = None  # Time to First Byte
    dom_content_loaded_ms: float | None = None
    load_ms: float | None = None
    raw_trace: dict | None = None


class DevToolsMCPClient:
    """Client for interacting with Chrome DevTools MCP server.

    The Chrome DevTools MCP server provides 26 tools across 6 categories:

    Input Automation (8):
        - devtools_click: Click element by CSS/XPath/text
        - devtools_fill: Fill form input
        - devtools_select: Select dropdown option
        - devtools_check / devtools_uncheck: Checkbox control
        - devtools_hover: Hover over element
        - devtools_type: Type text with keyboard events
        - devtools_press_key: Press keyboard key

    Navigation (6):
        - devtools_navigate: Navigate to URL
        - devtools_go_back / devtools_go_forward: History navigation
        - devtools_wait: Wait for selector/time
        - devtools_screenshot: Take screenshot
        - devtools_snapshot: Get accessibility tree snapshot

    Emulation (2):
        - devtools_set_geolocation: Set GPS coordinates
        - devtools_set_device: Emulate device (mobile, tablet)

    Performance (3):
        - devtools_start_performance_trace: Begin tracing
        - devtools_stop_performance_trace: End tracing + Core Web Vitals
        - devtools_get_performance_metrics: Get runtime metrics

    Network (2):
        - devtools_get_network_requests: List captured requests
        - devtools_get_network_request: Get request/response BODY

    Debugging (5):
        - devtools_evaluate: Execute JavaScript
        - devtools_get_console_messages: Get console with stack traces
        - devtools_get_page_info: Get page metadata
        - devtools_find_element: Find element by selector
        - devtools_get_styles: Get computed styles

    Example:
        async with DevToolsMCPClient() as client:
            await client.navigate("https://example.com")
            await client.start_performance_trace()
            await client.click("button#submit")
            metrics = await client.stop_performance_trace()
            requests = await client.get_network_requests()
            console_msgs = await client.get_console_messages()
    """

    def __init__(self, config: DevToolsMCPConfig | None = None):
        self.config = config or DevToolsMCPConfig()
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._request_id = 0
        self.log = logger.bind(component="devtools_mcp")

    async def start(self) -> None:
        """Start the Chrome DevTools MCP server process."""
        self.log.info("Starting Chrome DevTools MCP server")

        try:
            import os

            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **self.config.env},
            )

            self._reader = self._process.stdout
            self._writer = self._process.stdin

            await self._send_initialize()
            self.log.info("Chrome DevTools MCP server started")

        except FileNotFoundError:
            raise RuntimeError(
                "Chrome DevTools MCP server not found. Install with: "
                "npx @anthropic-ai/chrome-devtools-mcp@latest"
            )

    async def _send_initialize(self) -> None:
        """Send MCP initialization handshake."""
        init_message = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "argus-e2e-agent", "version": "1.0.0"},
            },
        }
        await self._send(init_message)
        response = await self._receive()
        self.log.debug("DevTools MCP initialized", response=response)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send(self, message: dict) -> None:
        """Send JSON-RPC message to the MCP server."""
        if not self._writer:
            raise RuntimeError("DevTools MCP server not started")

        data = json.dumps(message) + "\n"
        self._writer.write(data.encode())
        await self._writer.drain()

    async def _receive(self) -> dict:
        """Receive JSON-RPC message from the MCP server."""
        if not self._reader:
            raise RuntimeError("DevTools MCP server not started")

        line = await self._reader.readline()
        if not line:
            raise RuntimeError("DevTools MCP server closed connection")

        return json.loads(line.decode())

    async def call_tool(self, name: str, arguments: dict | None = None) -> DevToolsToolResult:
        """Call a Chrome DevTools MCP tool."""
        message = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }

        await self._send(message)
        response = await self._receive()

        if "error" in response:
            return DevToolsToolResult(
                success=False,
                content=None,
                error=response["error"].get("message", "Unknown error"),
            )

        return DevToolsToolResult(
            success=True,
            content=response.get("result", {}).get("content", []),
        )

    # ── Navigation ──────────────────────────────────────────────

    async def navigate(self, url: str) -> DevToolsToolResult:
        """Navigate to a URL."""
        return await self.call_tool("devtools_navigate", {"url": url})

    async def go_back(self) -> DevToolsToolResult:
        """Navigate back in history."""
        return await self.call_tool("devtools_go_back")

    async def go_forward(self) -> DevToolsToolResult:
        """Navigate forward in history."""
        return await self.call_tool("devtools_go_forward")

    async def wait(
        self,
        selector: str | None = None,
        timeout_ms: int | None = None,
    ) -> DevToolsToolResult:
        """Wait for selector to appear or specified time."""
        args: dict[str, Any] = {}
        if selector:
            args["selector"] = selector
        if timeout_ms:
            args["timeout"] = timeout_ms
        return await self.call_tool("devtools_wait", args)

    async def screenshot(self) -> DevToolsToolResult:
        """Take a screenshot of the current page."""
        return await self.call_tool("devtools_screenshot")

    async def snapshot(self) -> DevToolsToolResult:
        """Get accessibility tree snapshot of the page."""
        return await self.call_tool("devtools_snapshot")

    # ── Input Automation ────────────────────────────────────────

    async def click(self, selector: str) -> DevToolsToolResult:
        """Click an element by CSS selector, XPath, or text content."""
        return await self.call_tool("devtools_click", {"selector": selector})

    async def fill(self, selector: str, value: str) -> DevToolsToolResult:
        """Fill a form input field."""
        return await self.call_tool("devtools_fill", {"selector": selector, "value": value})

    async def select(self, selector: str, value: str) -> DevToolsToolResult:
        """Select from dropdown."""
        return await self.call_tool("devtools_select", {"selector": selector, "value": value})

    async def check(self, selector: str) -> DevToolsToolResult:
        """Check a checkbox."""
        return await self.call_tool("devtools_check", {"selector": selector})

    async def uncheck(self, selector: str) -> DevToolsToolResult:
        """Uncheck a checkbox."""
        return await self.call_tool("devtools_uncheck", {"selector": selector})

    async def hover(self, selector: str) -> DevToolsToolResult:
        """Hover over an element."""
        return await self.call_tool("devtools_hover", {"selector": selector})

    async def type_text(self, text: str) -> DevToolsToolResult:
        """Type text with keyboard events."""
        return await self.call_tool("devtools_type", {"text": text})

    async def press_key(self, key: str) -> DevToolsToolResult:
        """Press a keyboard key (e.g., 'Enter', 'Tab', 'Escape')."""
        return await self.call_tool("devtools_press_key", {"key": key})

    # ── Network Inspection ──────────────────────────────────────

    async def get_network_requests(self) -> list[NetworkRequest]:
        """Get all captured network requests.

        Returns parsed NetworkRequest objects with URLs, methods, status codes.
        Use get_network_request_detail() to get full request/response bodies.
        """
        result = await self.call_tool("devtools_get_network_requests")
        if not result.success:
            return []

        requests = []
        for item in self._extract_content(result):
            if isinstance(item, dict):
                requests.append(
                    NetworkRequest(
                        url=item.get("url", ""),
                        method=item.get("method", "GET"),
                        status=item.get("status"),
                        duration_ms=item.get("duration"),
                        resource_type=item.get("resourceType"),
                    )
                )
        return requests

    async def get_network_request_detail(self, request_id: str) -> NetworkRequest | None:
        """Get full request/response details including BODY.

        This is the key differentiator from Playwright MCP -- it returns
        the actual response body content, not just headers.
        """
        result = await self.call_tool("devtools_get_network_request", {"requestId": request_id})
        if not result.success:
            return None

        data = self._extract_first_content(result)
        if not data or not isinstance(data, dict):
            return None

        return NetworkRequest(
            url=data.get("url", ""),
            method=data.get("method", "GET"),
            status=data.get("status"),
            response_body=data.get("responseBody"),
            request_headers=data.get("requestHeaders", {}),
            response_headers=data.get("responseHeaders", {}),
            duration_ms=data.get("duration"),
            resource_type=data.get("resourceType"),
        )

    # ── Console & Debugging ─────────────────────────────────────

    async def get_console_messages(self) -> list[ConsoleMessage]:
        """Get console messages with source-mapped stack traces.

        Chrome DevTools MCP v0.16.0+ provides source-mapped stack traces,
        showing original file/line from TypeScript/JSX source maps.
        """
        result = await self.call_tool("devtools_get_console_messages")
        if not result.success:
            return []

        messages = []
        for item in self._extract_content(result):
            if isinstance(item, dict):
                messages.append(
                    ConsoleMessage(
                        level=item.get("level", "log"),
                        text=item.get("text", ""),
                        url=item.get("url"),
                        line=item.get("line"),
                        column=item.get("column"),
                        stack_trace=item.get("stackTrace"),
                    )
                )
        return messages

    async def evaluate(self, expression: str) -> DevToolsToolResult:
        """Execute JavaScript in the browser context."""
        return await self.call_tool("devtools_evaluate", {"expression": expression})

    async def get_page_info(self) -> DevToolsToolResult:
        """Get current page metadata (title, URL, etc.)."""
        return await self.call_tool("devtools_get_page_info")

    async def find_element(self, selector: str) -> DevToolsToolResult:
        """Find element by CSS selector and return its properties."""
        return await self.call_tool("devtools_find_element", {"selector": selector})

    async def get_styles(self, selector: str) -> DevToolsToolResult:
        """Get computed styles of an element."""
        return await self.call_tool("devtools_get_styles", {"selector": selector})

    # ── Performance ─────────────────────────────────────────────

    async def start_performance_trace(self) -> DevToolsToolResult:
        """Start a performance trace (captures Core Web Vitals)."""
        return await self.call_tool("devtools_start_performance_trace")

    async def stop_performance_trace(self) -> PerformanceMetrics:
        """Stop performance trace and return Core Web Vitals.

        Returns parsed PerformanceMetrics with LCP, FID, CLS, FCP, TTFB.
        """
        result = await self.call_tool("devtools_stop_performance_trace")
        if not result.success:
            return PerformanceMetrics()

        data = self._extract_first_content(result)
        if not data or not isinstance(data, dict):
            return PerformanceMetrics(raw_trace=data)

        return PerformanceMetrics(
            lcp_ms=data.get("lcp"),
            fid_ms=data.get("fid"),
            cls=data.get("cls"),
            fcp_ms=data.get("fcp"),
            ttfb_ms=data.get("ttfb"),
            dom_content_loaded_ms=data.get("domContentLoaded"),
            load_ms=data.get("load"),
            raw_trace=data,
        )

    async def get_performance_metrics(self) -> DevToolsToolResult:
        """Get runtime performance metrics (JS heap, nodes, etc.)."""
        return await self.call_tool("devtools_get_performance_metrics")

    # ── Emulation ───────────────────────────────────────────────

    async def set_geolocation(self, latitude: float, longitude: float) -> DevToolsToolResult:
        """Set device geolocation for testing location-based features."""
        return await self.call_tool(
            "devtools_set_geolocation",
            {"latitude": latitude, "longitude": longitude},
        )

    async def set_device(self, device: str) -> DevToolsToolResult:
        """Emulate a device (e.g., 'iPhone 14', 'Pixel 7', 'iPad')."""
        return await self.call_tool("devtools_set_device", {"device": device})

    # ── Helpers ──────────────────────────────────────────────────

    def _extract_content(self, result: DevToolsToolResult) -> list[Any]:
        """Extract content items from MCP tool result."""
        if not result.content:
            return []
        if isinstance(result.content, list):
            parsed = []
            for item in result.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    try:
                        parsed.append(json.loads(item["text"]))
                    except (json.JSONDecodeError, KeyError):
                        parsed.append(item.get("text", item))
                else:
                    parsed.append(item)
            return parsed
        return [result.content]

    def _extract_first_content(self, result: DevToolsToolResult) -> Any:
        """Extract first content item from MCP tool result."""
        items = self._extract_content(result)
        return items[0] if items else None

    async def stop(self) -> None:
        """Stop the Chrome DevTools MCP server."""
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None
            self.log.info("Chrome DevTools MCP server stopped")

    async def __aenter__(self) -> "DevToolsMCPClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


def create_devtools_mcp_tools() -> list[dict]:
    """Create tool definitions for Claude's tool use API.

    These definitions allow Claude to call Chrome DevTools MCP tools
    when used in the tool use API.
    """
    return [
        {
            "name": "devtools_navigate",
            "description": "Navigate the browser to a URL",
            "input_schema": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to navigate to"}},
                "required": ["url"],
            },
        },
        {
            "name": "devtools_click",
            "description": "Click an element by CSS selector, XPath, or text content",
            "input_schema": {
                "type": "object",
                "properties": {"selector": {"type": "string", "description": "Element selector"}},
                "required": ["selector"],
            },
        },
        {
            "name": "devtools_fill",
            "description": "Fill a form input field",
            "input_schema": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "Input selector"},
                    "value": {"type": "string", "description": "Value to fill"},
                },
                "required": ["selector", "value"],
            },
        },
        {
            "name": "devtools_screenshot",
            "description": "Take a screenshot of the current page",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "devtools_get_network_requests",
            "description": "Get all captured network requests with URLs, methods, and status codes",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "devtools_get_network_request",
            "description": "Get full request/response details including body content",
            "input_schema": {
                "type": "object",
                "properties": {
                    "requestId": {"type": "string", "description": "Network request ID"},
                },
                "required": ["requestId"],
            },
        },
        {
            "name": "devtools_get_console_messages",
            "description": "Get console messages with source-mapped stack traces",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "devtools_evaluate",
            "description": "Execute JavaScript in the browser context",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "JavaScript to execute"},
                },
                "required": ["expression"],
            },
        },
        {
            "name": "devtools_start_performance_trace",
            "description": "Start performance tracing (captures Core Web Vitals)",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "devtools_stop_performance_trace",
            "description": "Stop trace and return Core Web Vitals (LCP, FID, CLS, FCP, TTFB)",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "devtools_set_device",
            "description": "Emulate a device (e.g., 'iPhone 14', 'Pixel 7')",
            "input_schema": {
                "type": "object",
                "properties": {"device": {"type": "string", "description": "Device name"}},
                "required": ["device"],
            },
        },
        {
            "name": "devtools_hover",
            "description": "Hover over an element",
            "input_schema": {
                "type": "object",
                "properties": {"selector": {"type": "string", "description": "Element selector"}},
                "required": ["selector"],
            },
        },
    ]


# MCP configuration for Claude Code / MCP clients
MCP_CONFIG = {
    "chrome-devtools": {
        "command": "npx",
        "args": ["-y", "@anthropic-ai/chrome-devtools-mcp@latest"],
        "env": {},
    }
}


def generate_mcp_config() -> str:
    """Generate MCP configuration for Claude Code settings."""
    return json.dumps({"mcpServers": MCP_CONFIG}, indent=2)
