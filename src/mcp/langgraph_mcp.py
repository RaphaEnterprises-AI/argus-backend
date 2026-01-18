"""LangGraph + MCP integration using langchain-mcp-adapters.

This module uses the official langchain-mcp-adapters library to integrate
MCP servers with LangGraph for E2E testing.

Documentation: https://github.com/langchain-ai/langchain-mcp-adapters
Docs: https://docs.langchain.com/oss/python/langchain/mcp

Requirements:
    pip install langchain-mcp-adapters langgraph langchain-anthropic

Usage:
    from src.mcp.langgraph_mcp import create_testing_agent_with_mcp

    async with create_testing_agent_with_mcp() as agent:
        result = await agent.ainvoke({
            "messages": [("user", "Navigate to example.com and click login")]
        })
"""

from contextlib import asynccontextmanager

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

logger = structlog.get_logger()


# MCP Server configurations for E2E testing
# Sources:
#   - Playwright: https://www.npmjs.com/package/@playwright/mcp (Microsoft official)
#   - Filesystem: https://www.npmjs.com/package/@modelcontextprotocol/server-filesystem
#   - GitHub: https://www.npmjs.com/package/@modelcontextprotocol/server-github
MCP_SERVER_CONFIGS = {
    "playwright": {
        # Official Microsoft Playwright MCP server
        # Uses accessibility tree, no vision models needed
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest"],
    },
    "filesystem": {
        # Official MCP filesystem server with configurable access controls
        # Paths after package name are allowed directories
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
    },
    "github": {
        # Official MCP GitHub server (requires GITHUB_PERSONAL_ACCESS_TOKEN env var)
        # Note: Development moved to github/github-mcp-server repo
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
    },
}


@asynccontextmanager
async def create_testing_agent_with_mcp(
    servers: list[str] = ["playwright"],
    model: str = "claude-sonnet-4-5",
    system_prompt: str | None = None,
):
    """
    Create a LangGraph agent with MCP tools using langchain-mcp-adapters.

    This uses the official MultiServerMCPClient from langchain-mcp-adapters.

    Args:
        servers: List of MCP server names (e.g., ["playwright", "filesystem"])
        model: Claude model to use
        system_prompt: Optional custom system prompt

    Yields:
        LangGraph agent with MCP tools

    Example:
        async with create_testing_agent_with_mcp(["playwright"]) as agent:
            result = await agent.ainvoke({
                "messages": [("user", "Navigate to example.com")]
            })
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # Build server config for MultiServerMCPClient
    server_config = {}
    for name in servers:
        if name in MCP_SERVER_CONFIGS:
            server_config[name] = {
                "transport": "stdio",
                **MCP_SERVER_CONFIGS[name],
            }
        else:
            logger.warning(f"Unknown MCP server: {name}")

    if not server_config:
        raise ValueError("No valid MCP servers specified")

    # Create MCP client
    async with MultiServerMCPClient(server_config) as client:
        # Get tools from all connected MCP servers
        tools = await client.get_tools()

        logger.info(
            "MCP tools loaded",
            server_count=len(server_config),
            tool_count=len(tools),
        )

        # Create LLM
        llm = ChatAnthropic(
            model=model,
            max_tokens=4096,
        )

        # Default system prompt for testing
        default_prompt = system_prompt or """You are an autonomous E2E testing agent.

You have access to browser automation tools through MCP.

When executing tests:
1. Navigate to the target URL first
2. Wait for page to load before interacting
3. Execute each test step carefully and verify results
4. Take screenshots to document state
5. Report any issues or failures clearly

Be precise and methodical. After each action, verify the expected outcome."""

        # Create ReAct agent with MCP tools
        agent = create_react_agent(
            llm,
            tools,
            state_modifier=default_prompt,
        )

        yield agent


async def execute_test_with_mcp(
    test_spec: dict,
    app_url: str,
    servers: list[str] = ["playwright"],
    model: str = "claude-sonnet-4-5",
) -> dict:
    """
    Execute a single test using MCP tools.

    Args:
        test_spec: Test specification with steps and assertions
        app_url: Base URL of the application
        servers: MCP servers to use
        model: Claude model to use

    Returns:
        Test result with status and details
    """
    # Format test as prompt
    steps_text = "\n".join(
        f"  {i+1}. {s.get('action')}: {s.get('target', '')} {s.get('value', '')}"
        for i, s in enumerate(test_spec.get("steps", []))
    )

    assertions_text = "\n".join(
        f"  - {a.get('type')}: {a.get('target', '')} = {a.get('expected', '')}"
        for a in test_spec.get("assertions", [])
    )

    prompt = f"""Execute this E2E test:

Test: {test_spec.get('name', 'Unnamed Test')}
App URL: {app_url}

Steps:
{steps_text}

Assertions to verify:
{assertions_text}

Execute each step, then verify all assertions.
Report your findings in JSON format:
{{
    "status": "passed" or "failed",
    "steps_executed": [list of steps with results],
    "assertions_checked": [list of assertions with pass/fail],
    "error_message": null or description of any failure
}}
"""

    async with create_testing_agent_with_mcp(servers, model) as agent:
        try:
            result = await agent.ainvoke({
                "messages": [HumanMessage(content=prompt)]
            })

            # Extract final response
            messages = result.get("messages", [])
            if messages:
                final_message = messages[-1]
                content = final_message.content if hasattr(final_message, 'content') else str(final_message)

                # Try to parse JSON from response
                import json
                if "{" in content and "}" in content:
                    try:
                        start = content.index("{")
                        end = content.rindex("}") + 1
                        return json.loads(content[start:end])
                    except (json.JSONDecodeError, ValueError):
                        pass

                # Fallback parsing
                if "passed" in content.lower():
                    return {"status": "passed", "raw_response": content}
                else:
                    return {"status": "failed", "raw_response": content}

            return {"status": "unknown", "error": "No response from agent"}

        except Exception as e:
            logger.error("MCP test execution failed", error=str(e))
            return {"status": "failed", "error_message": str(e)}


class MCPTestingOrchestrator:
    """
    Orchestrator for running tests using MCP tools with LangGraph.

    This integrates with the main testing orchestrator to provide
    MCP-based test execution.

    Usage:
        orchestrator = MCPTestingOrchestrator(app_url="http://localhost:3000")
        await orchestrator.start()

        result = await orchestrator.execute_test(test_spec)

        await orchestrator.stop()
    """

    def __init__(
        self,
        app_url: str,
        servers: list[str] = ["playwright"],
        model: str = "claude-sonnet-4-5",
    ):
        self.app_url = app_url
        self.servers = servers
        self.model = model
        self._client = None
        self._agent = None
        self.log = logger.bind(component="mcp_orchestrator")

    async def start(self) -> None:
        """Start the MCP client and create agent."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

        server_config = {}
        for name in self.servers:
            if name in MCP_SERVER_CONFIGS:
                server_config[name] = {
                    "transport": "stdio",
                    **MCP_SERVER_CONFIGS[name],
                }

        self._client = MultiServerMCPClient(server_config)
        await self._client.__aenter__()

        tools = await self._client.get_tools()

        llm = ChatAnthropic(model=self.model, max_tokens=4096)
        self._agent = create_react_agent(llm, tools)

        self.log.info("MCP orchestrator started", tools=len(tools))

    async def execute_test(self, test_spec: dict) -> dict:
        """Execute a test using MCP agent."""
        if not self._agent:
            await self.start()

        return await execute_test_with_mcp(
            test_spec,
            self.app_url,
            self.servers,
            self.model,
        )

    async def stop(self) -> None:
        """Stop the MCP client."""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
            self._agent = None
            self.log.info("MCP orchestrator stopped")

    async def __aenter__(self) -> "MCPTestingOrchestrator":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
