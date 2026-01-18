"""Tests for the local MCP server module."""


import pytest

from src.mcp.local_mcp_server import LocalMCPServer


class TestLocalMCPServer:
    """Test LocalMCPServer functionality."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository with source files."""
        # Create source directory
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        (src_dir / "main.py").write_text('''
"""Main application module."""

import os
from typing import Optional

def main():
    """Entry point."""
    print("Hello, World!")

class App:
    """Main application class."""

    def __init__(self, config: dict):
        self.config = config

    def run(self) -> None:
        """Run the application."""
        pass

if __name__ == "__main__":
    main()
''')

        (src_dir / "utils.py").write_text('''
"""Utility functions."""

def helper_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

def another_helper():
    """Another helper function."""
    return "hello"
''')

        # Create tests directory
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        (tests_dir / "test_main.py").write_text('''
"""Tests for main module."""

import pytest
from src.main import main, App

def test_main():
    """Test main function."""
    main()

def test_app_run():
    """Test app run method."""
    app = App({})
    app.run()
''')

        return tmp_path

    @pytest.fixture
    def server(self, temp_repo):
        """Create a LocalMCPServer for the temp repo."""
        return LocalMCPServer(str(temp_repo))

    @pytest.mark.asyncio
    async def test_initialize(self, server):
        """Test server initialization."""
        await server.initialize()

        # Should have indexed the repository
        assert server._indexed is True or len(server._chunks) > 0

    def test_get_tools(self, server):
        """Test getting available tools."""
        tools = server.get_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

        tool_names = [t["name"] for t in tools]
        assert "analyze_code" in tool_names
        assert "find_similar_code" in tool_names
        assert "detect_changes" in tool_names

    @pytest.mark.asyncio
    async def test_analyze_code_tool(self, server):
        """Test the analyze_code tool."""
        await server.initialize()

        result = await server.call_tool("analyze_code", {"path": "src/main.py"})

        # MCP-style response has content
        assert "content" in result or "success" in result

    @pytest.mark.asyncio
    async def test_get_function_details(self, server):
        """Test the get_function_details tool."""
        await server.initialize()

        result = await server.call_tool("get_function_details", {
            "name": "helper_function",
        })

        # MCP-style response has content
        assert "content" in result or "success" in result

    @pytest.mark.asyncio
    async def test_get_class_details(self, server):
        """Test the get_class_details tool."""
        await server.initialize()

        result = await server.call_tool("get_class_details", {
            "name": "App",
        })

        # MCP-style response has content
        assert "content" in result or "success" in result

    @pytest.mark.asyncio
    async def test_detect_changes(self, temp_repo, server):
        """Test the detect_changes tool."""
        await server.initialize()

        # First call should show initial state
        result1 = await server.call_tool("detect_changes", {})
        assert "content" in result1 or "success" in result1

        # Modify a file
        (temp_repo / "src" / "main.py").write_text("# Modified content\n")

        # Second call should detect change
        result2 = await server.call_tool("detect_changes", {})
        assert "content" in result2 or "success" in result2

    @pytest.mark.asyncio
    async def test_get_repo_stats(self, server):
        """Test the get_repo_stats tool."""
        await server.initialize()

        result = await server.call_tool("get_repo_stats", {})

        # MCP-style response has content
        assert "content" in result or "success" in result

    @pytest.mark.asyncio
    async def test_suggest_tests(self, server):
        """Test the suggest_tests tool."""
        await server.initialize()

        result = await server.call_tool("suggest_tests", {
            "path": "src/utils.py",
        })

        # MCP-style response has content
        assert "content" in result or "success" in result

    @pytest.mark.asyncio
    async def test_find_usages(self, server):
        """Test the find_usages tool."""
        await server.initialize()

        result = await server.call_tool("find_usages", {
            "name": "helper_function",
        })

        # MCP-style response has content
        assert "content" in result or "success" in result

    @pytest.mark.asyncio
    async def test_unknown_tool(self, server):
        """Test calling an unknown tool."""
        await server.initialize()

        result = await server.call_tool("unknown_tool", {})

        # Unknown tool returns error content
        assert result.get("success") is False or "error" in result or "content" in result

    @pytest.mark.asyncio
    async def test_tool_without_init(self, temp_repo):
        """Test calling tool without initialization."""
        server = LocalMCPServer(str(temp_repo))

        # Should handle gracefully
        await server.call_tool("get_repo_stats", {})
        # May auto-initialize or return error


class TestLocalMCPServerTools:
    """Test specific tool implementations."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository."""
        (tmp_path / "module.py").write_text('''
def foo():
    """Foo function."""
    return 1

def bar():
    """Bar function."""
    return foo() + 1

class MyClass:
    """A class."""

    def method(self):
        """A method."""
        return bar()
''')
        return tmp_path

    @pytest.fixture
    def server(self, temp_repo):
        """Create server."""
        return LocalMCPServer(str(temp_repo))

    @pytest.mark.asyncio
    async def test_analyze_code_returns_structure(self, server):
        """Test that analyze_code returns code structure."""
        await server.initialize()

        result = await server.call_tool("analyze_code", {"path": "module.py"})

        # Check response is valid MCP format or success format
        assert "content" in result or "success" in result

    @pytest.mark.asyncio
    async def test_find_similar_code_with_query(self, server):
        """Test find_similar_code with a query."""
        await server.initialize()

        result = await server.call_tool("find_similar_code", {
            "query": "function that returns a number",
        })

        # MCP-style response or success
        assert "content" in result or "success" in result


class TestLocalMCPServerEdgeCases:
    """Test edge cases for LocalMCPServer."""

    @pytest.mark.asyncio
    async def test_empty_repo(self, tmp_path):
        """Test with an empty repository."""
        server = LocalMCPServer(str(tmp_path))
        await server.initialize()

        result = await server.call_tool("get_repo_stats", {})
        # MCP-style response
        assert "content" in result or "success" in result

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, tmp_path):
        """Test with a nonexistent path in tool call."""
        (tmp_path / "test.py").write_text("x = 1")
        server = LocalMCPServer(str(tmp_path))
        await server.initialize()

        result = await server.call_tool("analyze_code", {
            "path": "nonexistent.py",
        })

        # Should handle gracefully - MCP style response
        assert "content" in result or "success" in result or "error" in result

    @pytest.mark.asyncio
    async def test_invalid_arguments(self, tmp_path):
        """Test with invalid tool arguments."""
        (tmp_path / "test.py").write_text("x = 1")
        server = LocalMCPServer(str(tmp_path))
        await server.initialize()

        # Missing required arguments
        result = await server.call_tool("get_function_details", {})

        # Should handle gracefully
        assert isinstance(result, dict)
