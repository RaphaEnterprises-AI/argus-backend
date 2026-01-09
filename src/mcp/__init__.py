"""MCP (Model Context Protocol) integration for E2E testing and quality intelligence.

Authentication:
    MCP servers use OAuth2 Device Flow for authentication.
    On first run, users are prompted to authenticate via browser.
    Tokens are cached in ~/.argus/tokens.json.

Usage:
    # In your MCP client config (e.g., ~/.claude/mcp_servers.json):
    {
        "argus-quality": {
            "command": "python",
            "args": ["-m", "src.mcp.quality_mcp"]
        }
    }

    # CLI authentication commands:
    python -m src.mcp.auth login    # Authenticate with Argus
    python -m src.mcp.auth status   # Check auth status
    python -m src.mcp.auth logout   # Clear tokens
"""

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
from .auth import (
    MCPAuthenticator,
    AuthenticatedClient,
    AuthenticationError,
    load_cached_tokens,
    clear_tokens,
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
    # Authentication
    "MCPAuthenticator",
    "AuthenticatedClient",
    "AuthenticationError",
    "load_cached_tokens",
    "clear_tokens",
]
