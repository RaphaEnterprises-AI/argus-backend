"""MCP (Model Context Protocol) integration for E2E testing and quality intelligence."""

from .playwright_mcp import PlaywrightMCPClient, create_playwright_mcp_tools
from .langgraph_mcp import (
    create_testing_agent_with_mcp,
    execute_test_with_mcp,
    MCPTestingOrchestrator,
    MCP_SERVER_CONFIGS,
)
from .quality_mcp import (
    QualityMCPServer,
    create_quality_mcp_tools,
    QUALITY_TOOLS,
)

__all__ = [
    # Low-level Playwright MCP client
    "PlaywrightMCPClient",
    "create_playwright_mcp_tools",
    # LangGraph + MCP integration (using langchain-mcp-adapters)
    "create_testing_agent_with_mcp",
    "execute_test_with_mcp",
    "MCPTestingOrchestrator",
    "MCP_SERVER_CONFIGS",
    # Quality Intelligence MCP tools
    "QualityMCPServer",
    "create_quality_mcp_tools",
    "QUALITY_TOOLS",
]
