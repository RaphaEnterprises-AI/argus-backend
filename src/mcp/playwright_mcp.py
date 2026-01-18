"""Playwright MCP (Model Context Protocol) integration.

This module provides integration with Playwright's MCP server for browser automation.
The Playwright MCP server allows Claude to interact with browsers through a standardized
protocol.

Usage:
    1. Install Playwright MCP: npx @anthropic/mcp-server-playwright
    2. Configure in your Claude Code settings or use programmatically

See: https://github.com/anthropics/mcp-servers
"""

import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class MCPServerConfig:
    """Configuration for MCP server.

    Uses official Microsoft Playwright MCP server.
    Source: https://www.npmjs.com/package/@playwright/mcp
    """
    command: str = "npx"
    args: list[str] = field(default_factory=lambda: ["-y", "@playwright/mcp@latest"])
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class MCPToolCall:
    """Represents a call to an MCP tool."""
    name: str
    arguments: dict[str, Any]


@dataclass
class MCPToolResult:
    """Result from an MCP tool call."""
    success: bool
    content: Any
    error: str | None = None


class PlaywrightMCPClient:
    """
    Client for interacting with Playwright MCP server.

    The Playwright MCP server provides these tools:
    - browser_navigate: Navigate to a URL
    - browser_screenshot: Take a screenshot
    - browser_click: Click an element
    - browser_fill: Fill a form field
    - browser_select: Select from dropdown
    - browser_hover: Hover over element
    - browser_evaluate: Run JavaScript

    Example:
        async with PlaywrightMCPClient() as client:
            await client.navigate("https://example.com")
            await client.click("button#submit")
            screenshot = await client.screenshot()
    """

    def __init__(self, config: MCPServerConfig | None = None):
        self.config = config or MCPServerConfig()
        self._process: subprocess.Popen | None = None
        self._reader = None
        self._writer = None
        self.log = logger.bind(component="playwright_mcp")

    async def start(self) -> None:
        """Start the MCP server process."""
        self.log.info("Starting Playwright MCP server")

        try:
            # Start the MCP server
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**dict(__import__("os").environ), **self.config.env},
            )

            self._reader = self._process.stdout
            self._writer = self._process.stdin

            # Initialize the connection
            await self._send_initialize()

            self.log.info("Playwright MCP server started")

        except FileNotFoundError:
            raise RuntimeError(
                "Playwright MCP server not found. Install with: "
                "npm install -g @anthropic/mcp-server-playwright"
            )

    async def _send_initialize(self) -> None:
        """Send initialization message to MCP server."""
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "e2e-testing-agent",
                    "version": "1.0.0"
                }
            }
        }
        await self._send(init_message)
        response = await self._receive()
        self.log.debug("MCP initialized", response=response)

    async def _send(self, message: dict) -> None:
        """Send a message to the MCP server."""
        if not self._writer:
            raise RuntimeError("MCP server not started")

        data = json.dumps(message) + "\n"
        self._writer.write(data.encode())
        await self._writer.drain()

    async def _receive(self) -> dict:
        """Receive a message from the MCP server."""
        if not self._reader:
            raise RuntimeError("MCP server not started")

        line = await self._reader.readline()
        if not line:
            raise RuntimeError("MCP server closed connection")

        return json.loads(line.decode())

    async def call_tool(self, name: str, arguments: dict) -> MCPToolResult:
        """Call an MCP tool."""
        message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }

        await self._send(message)
        response = await self._receive()

        if "error" in response:
            return MCPToolResult(
                success=False,
                content=None,
                error=response["error"].get("message", "Unknown error")
            )

        return MCPToolResult(
            success=True,
            content=response.get("result", {}).get("content", [])
        )

    # High-level browser actions

    async def navigate(self, url: str) -> MCPToolResult:
        """Navigate to a URL."""
        return await self.call_tool("browser_navigate", {"url": url})

    async def screenshot(self) -> MCPToolResult:
        """Take a screenshot."""
        return await self.call_tool("browser_screenshot", {})

    async def click(self, selector: str) -> MCPToolResult:
        """Click an element."""
        return await self.call_tool("browser_click", {"selector": selector})

    async def fill(self, selector: str, value: str) -> MCPToolResult:
        """Fill a form field."""
        return await self.call_tool("browser_fill", {
            "selector": selector,
            "value": value
        })

    async def select(self, selector: str, value: str) -> MCPToolResult:
        """Select from dropdown."""
        return await self.call_tool("browser_select", {
            "selector": selector,
            "value": value
        })

    async def hover(self, selector: str) -> MCPToolResult:
        """Hover over element."""
        return await self.call_tool("browser_hover", {"selector": selector})

    async def evaluate(self, script: str) -> MCPToolResult:
        """Run JavaScript in browser."""
        return await self.call_tool("browser_evaluate", {"script": script})

    async def stop(self) -> None:
        """Stop the MCP server."""
        if self._process:
            self._process.terminate()
            await self._process.wait()
            self._process = None
            self.log.info("Playwright MCP server stopped")

    async def __aenter__(self) -> "PlaywrightMCPClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


def create_playwright_mcp_tools() -> list[dict]:
    """
    Create tool definitions for Claude to use Playwright MCP.

    These tools can be passed to Claude's tool use API.

    Returns:
        List of tool definitions
    """
    return [
        {
            "name": "browser_navigate",
            "description": "Navigate the browser to a URL",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to"
                    }
                },
                "required": ["url"]
            }
        },
        {
            "name": "browser_screenshot",
            "description": "Take a screenshot of the current page",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "browser_click",
            "description": "Click on an element in the page",
            "input_schema": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the element to click"
                    }
                },
                "required": ["selector"]
            }
        },
        {
            "name": "browser_fill",
            "description": "Fill a form field with text",
            "input_schema": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the input field"
                    },
                    "value": {
                        "type": "string",
                        "description": "The text to fill in"
                    }
                },
                "required": ["selector", "value"]
            }
        },
        {
            "name": "browser_select",
            "description": "Select an option from a dropdown",
            "input_schema": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the select element"
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to select"
                    }
                },
                "required": ["selector", "value"]
            }
        },
        {
            "name": "browser_hover",
            "description": "Hover over an element",
            "input_schema": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the element to hover over"
                    }
                },
                "required": ["selector"]
            }
        },
        {
            "name": "browser_evaluate",
            "description": "Execute JavaScript in the browser",
            "input_schema": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "JavaScript code to execute"
                    }
                },
                "required": ["script"]
            }
        }
    ]


# Claude Code MCP configuration for mcp_servers.json
# Source: https://www.npmjs.com/package/@playwright/mcp
MCP_CONFIG = {
    "playwright": {
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest"],
        "env": {}
    }
}


def generate_mcp_config() -> str:
    """Generate MCP configuration for Claude Code settings."""
    return json.dumps({"mcpServers": MCP_CONFIG}, indent=2)
