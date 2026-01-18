"""Tests for Playwright MCP client module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_default_config(self, mock_env_vars):
        """Test default MCPServerConfig values."""
        from src.mcp.playwright_mcp import MCPServerConfig

        config = MCPServerConfig()

        assert config.command == "npx"
        assert config.args == ["-y", "@playwright/mcp@latest"]
        assert config.env == {}

    def test_custom_config(self, mock_env_vars):
        """Test MCPServerConfig with custom values."""
        from src.mcp.playwright_mcp import MCPServerConfig

        config = MCPServerConfig(
            command="node",
            args=["./custom-mcp.js"],
            env={"DEBUG": "true"},
        )

        assert config.command == "node"
        assert config.args == ["./custom-mcp.js"]
        assert config.env == {"DEBUG": "true"}


class TestMCPToolCall:
    """Tests for MCPToolCall dataclass."""

    def test_tool_call_creation(self, mock_env_vars):
        """Test MCPToolCall creation."""
        from src.mcp.playwright_mcp import MCPToolCall

        call = MCPToolCall(
            name="browser_click",
            arguments={"selector": "#submit-button"},
        )

        assert call.name == "browser_click"
        assert call.arguments == {"selector": "#submit-button"}


class TestMCPToolResult:
    """Tests for MCPToolResult dataclass."""

    def test_successful_result(self, mock_env_vars):
        """Test successful MCPToolResult."""
        from src.mcp.playwright_mcp import MCPToolResult

        result = MCPToolResult(
            success=True,
            content={"screenshot": "base64data"},
        )

        assert result.success is True
        assert result.content == {"screenshot": "base64data"}
        assert result.error is None

    def test_failed_result(self, mock_env_vars):
        """Test failed MCPToolResult."""
        from src.mcp.playwright_mcp import MCPToolResult

        result = MCPToolResult(
            success=False,
            content=None,
            error="Element not found",
        )

        assert result.success is False
        assert result.content is None
        assert result.error == "Element not found"


class TestPlaywrightMCPClient:
    """Tests for PlaywrightMCPClient class."""

    def test_client_creation_default(self, mock_env_vars):
        """Test PlaywrightMCPClient creation with defaults."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()

        assert client.config is not None
        assert client.config.command == "npx"
        assert client._process is None
        assert client._reader is None
        assert client._writer is None

    def test_client_creation_custom_config(self, mock_env_vars):
        """Test PlaywrightMCPClient creation with custom config."""
        from src.mcp.playwright_mcp import MCPServerConfig, PlaywrightMCPClient

        config = MCPServerConfig(command="custom", args=["--test"])
        client = PlaywrightMCPClient(config=config)

        assert client.config.command == "custom"
        assert client.config.args == ["--test"]

    @pytest.mark.asyncio
    async def test_start_success(self, mock_env_vars):
        """Test successful MCP server start."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        mock_process = MagicMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.drain = AsyncMock()

        # Mock readline to return init response
        init_response = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}})
        mock_process.stdout.readline = AsyncMock(
            return_value=init_response.encode() + b"\n"
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            client = PlaywrightMCPClient()
            await client.start()

            assert client._process is mock_process
            assert client._reader is mock_process.stdout
            assert client._writer is mock_process.stdin

    @pytest.mark.asyncio
    async def test_start_not_found(self, mock_env_vars):
        """Test MCP server not found error."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("npx not found"),
        ):
            client = PlaywrightMCPClient()

            with pytest.raises(RuntimeError) as exc_info:
                await client.start()

            assert "Playwright MCP server not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_without_start(self, mock_env_vars):
        """Test _send raises error when not started."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()

        with pytest.raises(RuntimeError) as exc_info:
            await client._send({"test": "message"})

        assert "MCP server not started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_receive_without_start(self, mock_env_vars):
        """Test _receive raises error when not started."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()

        with pytest.raises(RuntimeError) as exc_info:
            await client._receive()

        assert "MCP server not started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_receive_connection_closed(self, mock_env_vars):
        """Test _receive when connection is closed."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()
        client._reader = AsyncMock()
        client._reader.readline = AsyncMock(return_value=b"")

        with pytest.raises(RuntimeError) as exc_info:
            await client._receive()

        assert "MCP server closed connection" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_tool_success(self, mock_env_vars):
        """Test successful tool call."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()
        client._writer = AsyncMock()
        client._writer.drain = AsyncMock()

        response = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"content": [{"type": "text", "text": "Clicked"}]},
        }
        client._reader = AsyncMock()
        client._reader.readline = AsyncMock(
            return_value=json.dumps(response).encode()
        )

        result = await client.call_tool("browser_click", {"selector": "#btn"})

        assert result.success is True
        assert result.content == [{"type": "text", "text": "Clicked"}]
        assert result.error is None

    @pytest.mark.asyncio
    async def test_call_tool_error(self, mock_env_vars):
        """Test tool call with error response."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()
        client._writer = AsyncMock()
        client._writer.drain = AsyncMock()

        response = {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"message": "Element not found"},
        }
        client._reader = AsyncMock()
        client._reader.readline = AsyncMock(
            return_value=json.dumps(response).encode()
        )

        result = await client.call_tool("browser_click", {"selector": "#missing"})

        assert result.success is False
        assert result.content is None
        assert result.error == "Element not found"

    @pytest.mark.asyncio
    async def test_navigate(self, mock_env_vars):
        """Test navigate convenience method."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()
        client.call_tool = AsyncMock(return_value=MagicMock(success=True))

        await client.navigate("https://example.com")

        client.call_tool.assert_called_once_with(
            "browser_navigate", {"url": "https://example.com"}
        )

    @pytest.mark.asyncio
    async def test_screenshot(self, mock_env_vars):
        """Test screenshot convenience method."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()
        client.call_tool = AsyncMock(return_value=MagicMock(success=True))

        await client.screenshot()

        client.call_tool.assert_called_once_with("browser_screenshot", {})

    @pytest.mark.asyncio
    async def test_click(self, mock_env_vars):
        """Test click convenience method."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()
        client.call_tool = AsyncMock(return_value=MagicMock(success=True))

        await client.click("#submit-btn")

        client.call_tool.assert_called_once_with(
            "browser_click", {"selector": "#submit-btn"}
        )

    @pytest.mark.asyncio
    async def test_fill(self, mock_env_vars):
        """Test fill convenience method."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()
        client.call_tool = AsyncMock(return_value=MagicMock(success=True))

        await client.fill("#email", "test@example.com")

        client.call_tool.assert_called_once_with(
            "browser_fill", {"selector": "#email", "value": "test@example.com"}
        )

    @pytest.mark.asyncio
    async def test_select(self, mock_env_vars):
        """Test select convenience method."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()
        client.call_tool = AsyncMock(return_value=MagicMock(success=True))

        await client.select("#country", "USA")

        client.call_tool.assert_called_once_with(
            "browser_select", {"selector": "#country", "value": "USA"}
        )

    @pytest.mark.asyncio
    async def test_hover(self, mock_env_vars):
        """Test hover convenience method."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()
        client.call_tool = AsyncMock(return_value=MagicMock(success=True))

        await client.hover("#menu")

        client.call_tool.assert_called_once_with(
            "browser_hover", {"selector": "#menu"}
        )

    @pytest.mark.asyncio
    async def test_evaluate(self, mock_env_vars):
        """Test evaluate convenience method."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()
        client.call_tool = AsyncMock(return_value=MagicMock(success=True))

        await client.evaluate("document.title")

        client.call_tool.assert_called_once_with(
            "browser_evaluate", {"script": "document.title"}
        )

    @pytest.mark.asyncio
    async def test_stop(self, mock_env_vars):
        """Test stop method."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()

        client = PlaywrightMCPClient()
        client._process = mock_process

        await client.stop()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        assert client._process is None

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, mock_env_vars):
        """Test stop when process is not running."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()
        # Should not raise any error
        await client.stop()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_env_vars):
        """Test async context manager."""
        from src.mcp.playwright_mcp import PlaywrightMCPClient

        client = PlaywrightMCPClient()
        client.start = AsyncMock()
        client.stop = AsyncMock()

        async with client as c:
            assert c is client
            client.start.assert_called_once()

        client.stop.assert_called_once()


class TestCreatePlaywrightMCPTools:
    """Tests for create_playwright_mcp_tools function."""

    def test_returns_all_tools(self, mock_env_vars):
        """Test that all expected tools are returned."""
        from src.mcp.playwright_mcp import create_playwright_mcp_tools

        tools = create_playwright_mcp_tools()

        assert len(tools) == 7
        tool_names = [t["name"] for t in tools]
        assert "browser_navigate" in tool_names
        assert "browser_screenshot" in tool_names
        assert "browser_click" in tool_names
        assert "browser_fill" in tool_names
        assert "browser_select" in tool_names
        assert "browser_hover" in tool_names
        assert "browser_evaluate" in tool_names

    def test_navigate_tool_schema(self, mock_env_vars):
        """Test browser_navigate tool schema."""
        from src.mcp.playwright_mcp import create_playwright_mcp_tools

        tools = create_playwright_mcp_tools()
        navigate_tool = next(t for t in tools if t["name"] == "browser_navigate")

        assert "description" in navigate_tool
        assert navigate_tool["input_schema"]["type"] == "object"
        assert "url" in navigate_tool["input_schema"]["properties"]
        assert "url" in navigate_tool["input_schema"]["required"]

    def test_screenshot_tool_schema(self, mock_env_vars):
        """Test browser_screenshot tool schema."""
        from src.mcp.playwright_mcp import create_playwright_mcp_tools

        tools = create_playwright_mcp_tools()
        screenshot_tool = next(t for t in tools if t["name"] == "browser_screenshot")

        assert screenshot_tool["input_schema"]["required"] == []

    def test_click_tool_schema(self, mock_env_vars):
        """Test browser_click tool schema."""
        from src.mcp.playwright_mcp import create_playwright_mcp_tools

        tools = create_playwright_mcp_tools()
        click_tool = next(t for t in tools if t["name"] == "browser_click")

        assert "selector" in click_tool["input_schema"]["properties"]
        assert "selector" in click_tool["input_schema"]["required"]

    def test_fill_tool_schema(self, mock_env_vars):
        """Test browser_fill tool schema."""
        from src.mcp.playwright_mcp import create_playwright_mcp_tools

        tools = create_playwright_mcp_tools()
        fill_tool = next(t for t in tools if t["name"] == "browser_fill")

        assert "selector" in fill_tool["input_schema"]["properties"]
        assert "value" in fill_tool["input_schema"]["properties"]
        assert "selector" in fill_tool["input_schema"]["required"]
        assert "value" in fill_tool["input_schema"]["required"]


class TestMCPConfig:
    """Tests for MCP_CONFIG constant."""

    def test_mcp_config_structure(self, mock_env_vars):
        """Test MCP_CONFIG has expected structure."""
        from src.mcp.playwright_mcp import MCP_CONFIG

        assert "playwright" in MCP_CONFIG
        assert MCP_CONFIG["playwright"]["command"] == "npx"
        assert "-y" in MCP_CONFIG["playwright"]["args"]
        assert "@playwright/mcp@latest" in MCP_CONFIG["playwright"]["args"]


class TestGenerateMCPConfig:
    """Tests for generate_mcp_config function."""

    def test_generates_valid_json(self, mock_env_vars):
        """Test that generate_mcp_config returns valid JSON."""
        from src.mcp.playwright_mcp import generate_mcp_config

        config_str = generate_mcp_config()

        # Should be valid JSON
        config = json.loads(config_str)
        assert "mcpServers" in config
        assert "playwright" in config["mcpServers"]

    def test_config_structure(self, mock_env_vars):
        """Test generated config structure."""
        from src.mcp.playwright_mcp import generate_mcp_config

        config_str = generate_mcp_config()
        config = json.loads(config_str)

        playwright = config["mcpServers"]["playwright"]
        assert playwright["command"] == "npx"
        assert "args" in playwright
        assert "env" in playwright
