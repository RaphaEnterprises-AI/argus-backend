"""Tests for LangGraph MCP integration module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMCPServerConfigs:
    """Tests for MCP_SERVER_CONFIGS constant."""

    def test_playwright_config(self, mock_env_vars):
        """Test Playwright server configuration."""
        from src.mcp.langgraph_mcp import MCP_SERVER_CONFIGS

        assert "playwright" in MCP_SERVER_CONFIGS
        assert MCP_SERVER_CONFIGS["playwright"]["command"] == "npx"
        assert "-y" in MCP_SERVER_CONFIGS["playwright"]["args"]
        assert "@playwright/mcp@latest" in MCP_SERVER_CONFIGS["playwright"]["args"]

    def test_filesystem_config(self, mock_env_vars):
        """Test Filesystem server configuration."""
        from src.mcp.langgraph_mcp import MCP_SERVER_CONFIGS

        assert "filesystem" in MCP_SERVER_CONFIGS
        assert MCP_SERVER_CONFIGS["filesystem"]["command"] == "npx"
        assert "@modelcontextprotocol/server-filesystem" in MCP_SERVER_CONFIGS["filesystem"]["args"]

    def test_github_config(self, mock_env_vars):
        """Test GitHub server configuration."""
        from src.mcp.langgraph_mcp import MCP_SERVER_CONFIGS

        assert "github" in MCP_SERVER_CONFIGS
        assert MCP_SERVER_CONFIGS["github"]["command"] == "npx"
        assert "@modelcontextprotocol/server-github" in MCP_SERVER_CONFIGS["github"]["args"]


class TestCreateTestingAgentWithMCP:
    """Tests for create_testing_agent_with_mcp async context manager."""

    @pytest.mark.asyncio
    async def test_no_valid_servers_raises_error(self, mock_env_vars):
        """Test error when no valid servers specified."""
        from src.mcp.langgraph_mcp import create_testing_agent_with_mcp

        with pytest.raises(ValueError) as exc_info:
            async with create_testing_agent_with_mcp(servers=["invalid_server"]):
                pass

        assert "No valid MCP servers specified" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unknown_server_warning(self, mock_env_vars):
        """Test warning logged for unknown servers."""
        from src.mcp.langgraph_mcp import create_testing_agent_with_mcp

        # Mix of valid and invalid servers - should warn about invalid but still work
        with patch("src.mcp.langgraph_mcp.logger") as mock_logger:
            with pytest.raises(ValueError):
                # All invalid servers will raise ValueError
                async with create_testing_agent_with_mcp(servers=["unknown"]):
                    pass

            # Check warning was logged
            mock_logger.warning.assert_called_with("Unknown MCP server: unknown")

    @pytest.mark.asyncio
    async def test_creates_agent_with_valid_servers(self, mock_env_vars):
        """Test agent creation with valid servers."""
        from src.mcp.langgraph_mcp import create_testing_agent_with_mcp

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get_tools = AsyncMock(return_value=[
            MagicMock(name="browser_navigate"),
            MagicMock(name="browser_click"),
        ])

        mock_agent = MagicMock()

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=mock_client,
        ):
            with patch(
                "src.mcp.langgraph_mcp.ChatAnthropic",
                return_value=MagicMock(),
            ):
                with patch(
                    "src.mcp.langgraph_mcp.create_react_agent",
                    return_value=mock_agent,
                ):
                    async with create_testing_agent_with_mcp(
                        servers=["playwright"],
                        model="claude-sonnet-4-5",
                    ) as agent:
                        assert agent is mock_agent

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, mock_env_vars):
        """Test custom system prompt is used."""
        from src.mcp.langgraph_mcp import create_testing_agent_with_mcp

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get_tools = AsyncMock(return_value=[])

        custom_prompt = "You are a custom testing agent."

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=mock_client,
        ):
            with patch(
                "src.mcp.langgraph_mcp.ChatAnthropic",
                return_value=MagicMock(),
            ):
                with patch(
                    "src.mcp.langgraph_mcp.create_react_agent",
                ) as mock_create:
                    mock_create.return_value = MagicMock()

                    async with create_testing_agent_with_mcp(
                        servers=["playwright"],
                        system_prompt=custom_prompt,
                    ):
                        # Verify create_react_agent was called with custom prompt
                        call_args = mock_create.call_args
                        assert call_args.kwargs["state_modifier"] == custom_prompt


class TestExecuteTestWithMCP:
    """Tests for execute_test_with_mcp function."""

    @pytest.mark.asyncio
    async def test_execute_test_success_with_json(self, mock_env_vars):
        """Test successful test execution with JSON response."""
        from src.mcp.langgraph_mcp import execute_test_with_mcp

        test_spec = {
            "name": "Login Test",
            "steps": [
                {"action": "navigate", "target": "/login"},
                {"action": "fill", "target": "#email", "value": "test@example.com"},
                {"action": "click", "target": "#submit"},
            ],
            "assertions": [
                {"type": "url", "expected": "/dashboard"},
            ],
        }

        mock_message = MagicMock()
        mock_message.content = '{"status": "passed", "steps_executed": []}'

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_message]})

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "src.mcp.langgraph_mcp.create_testing_agent_with_mcp",
            return_value=mock_context,
        ):
            result = await execute_test_with_mcp(
                test_spec,
                "http://localhost:3000",
            )

            assert result["status"] == "passed"

    @pytest.mark.asyncio
    async def test_execute_test_fallback_parsing_passed(self, mock_env_vars):
        """Test fallback parsing when JSON parsing fails - passed."""
        from src.mcp.langgraph_mcp import execute_test_with_mcp

        test_spec = {"name": "Test", "steps": [], "assertions": []}

        mock_message = MagicMock()
        mock_message.content = "The test has passed successfully."

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_message]})

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "src.mcp.langgraph_mcp.create_testing_agent_with_mcp",
            return_value=mock_context,
        ):
            result = await execute_test_with_mcp(test_spec, "http://localhost:3000")

            assert result["status"] == "passed"
            assert "raw_response" in result

    @pytest.mark.asyncio
    async def test_execute_test_fallback_parsing_failed(self, mock_env_vars):
        """Test fallback parsing when JSON parsing fails - failed."""
        from src.mcp.langgraph_mcp import execute_test_with_mcp

        test_spec = {"name": "Test", "steps": [], "assertions": []}

        mock_message = MagicMock()
        mock_message.content = "The test encountered an error and could not complete."

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_message]})

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "src.mcp.langgraph_mcp.create_testing_agent_with_mcp",
            return_value=mock_context,
        ):
            result = await execute_test_with_mcp(test_spec, "http://localhost:3000")

            assert result["status"] == "failed"
            assert "raw_response" in result

    @pytest.mark.asyncio
    async def test_execute_test_no_messages(self, mock_env_vars):
        """Test execution with no response messages."""
        from src.mcp.langgraph_mcp import execute_test_with_mcp

        test_spec = {"name": "Test", "steps": [], "assertions": []}

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": []})

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "src.mcp.langgraph_mcp.create_testing_agent_with_mcp",
            return_value=mock_context,
        ):
            result = await execute_test_with_mcp(test_spec, "http://localhost:3000")

            assert result["status"] == "unknown"
            assert result["error"] == "No response from agent"

    @pytest.mark.asyncio
    async def test_execute_test_exception(self, mock_env_vars):
        """Test execution with exception."""
        from src.mcp.langgraph_mcp import execute_test_with_mcp

        test_spec = {"name": "Test", "steps": [], "assertions": []}

        # Mock the agent to raise an exception when ainvoke is called
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(side_effect=Exception("Connection failed"))

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "src.mcp.langgraph_mcp.create_testing_agent_with_mcp",
            return_value=mock_context,
        ):
            result = await execute_test_with_mcp(test_spec, "http://localhost:3000")

            assert result["status"] == "failed"
            assert "Connection failed" in result["error_message"]

    @pytest.mark.asyncio
    async def test_execute_test_message_without_content_attr(self, mock_env_vars):
        """Test execution when message has no content attribute."""
        from src.mcp.langgraph_mcp import execute_test_with_mcp

        test_spec = {"name": "Test", "steps": [], "assertions": []}

        # Message that converts to string with "passed"
        mock_message = "Test passed successfully"

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_message]})

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "src.mcp.langgraph_mcp.create_testing_agent_with_mcp",
            return_value=mock_context,
        ):
            result = await execute_test_with_mcp(test_spec, "http://localhost:3000")

            assert result["status"] == "passed"

    @pytest.mark.asyncio
    async def test_execute_test_invalid_json_in_response(self, mock_env_vars):
        """Test execution with invalid JSON in response."""
        from src.mcp.langgraph_mcp import execute_test_with_mcp

        test_spec = {"name": "Test", "steps": [], "assertions": []}

        mock_message = MagicMock()
        mock_message.content = 'Result: {"status": passed} - invalid json'

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_message]})

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "src.mcp.langgraph_mcp.create_testing_agent_with_mcp",
            return_value=mock_context,
        ):
            result = await execute_test_with_mcp(test_spec, "http://localhost:3000")

            # Should fall back to string parsing
            assert result["status"] == "passed"  # "passed" is in the content


class TestMCPTestingOrchestrator:
    """Tests for MCPTestingOrchestrator class."""

    def test_orchestrator_init(self, mock_env_vars):
        """Test orchestrator initialization."""
        from src.mcp.langgraph_mcp import MCPTestingOrchestrator

        orchestrator = MCPTestingOrchestrator(
            app_url="http://localhost:3000",
            servers=["playwright", "filesystem"],
            model="claude-sonnet-4-5",
        )

        assert orchestrator.app_url == "http://localhost:3000"
        assert orchestrator.servers == ["playwright", "filesystem"]
        assert orchestrator.model == "claude-sonnet-4-5"
        assert orchestrator._client is None
        assert orchestrator._agent is None

    def test_orchestrator_default_values(self, mock_env_vars):
        """Test orchestrator default values."""
        from src.mcp.langgraph_mcp import MCPTestingOrchestrator

        orchestrator = MCPTestingOrchestrator(app_url="http://localhost:3000")

        assert orchestrator.servers == ["playwright"]
        assert orchestrator.model == "claude-sonnet-4-5"

    @pytest.mark.asyncio
    async def test_start(self, mock_env_vars):
        """Test orchestrator start method."""
        from src.mcp.langgraph_mcp import MCPTestingOrchestrator

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.get_tools = AsyncMock(return_value=[MagicMock(), MagicMock()])

        mock_agent = MagicMock()

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=mock_client,
        ):
            with patch(
                "src.mcp.langgraph_mcp.ChatAnthropic",
                return_value=MagicMock(),
            ):
                with patch(
                    "src.mcp.langgraph_mcp.create_react_agent",
                    return_value=mock_agent,
                ):
                    orchestrator = MCPTestingOrchestrator(
                        app_url="http://localhost:3000"
                    )
                    await orchestrator.start()

                    assert orchestrator._client is mock_client
                    assert orchestrator._agent is mock_agent

    @pytest.mark.asyncio
    async def test_execute_test(self, mock_env_vars):
        """Test orchestrator execute_test method."""
        from src.mcp.langgraph_mcp import MCPTestingOrchestrator

        orchestrator = MCPTestingOrchestrator(app_url="http://localhost:3000")
        orchestrator._agent = MagicMock()  # Simulate already started

        test_spec = {"name": "Test", "steps": [], "assertions": []}

        with patch(
            "src.mcp.langgraph_mcp.execute_test_with_mcp",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = {"status": "passed"}

            result = await orchestrator.execute_test(test_spec)

            assert result["status"] == "passed"
            mock_execute.assert_called_once_with(
                test_spec,
                "http://localhost:3000",
                ["playwright"],
                "claude-sonnet-4-5",
            )

    @pytest.mark.asyncio
    async def test_execute_test_auto_start(self, mock_env_vars):
        """Test orchestrator auto-starts if not started."""
        from src.mcp.langgraph_mcp import MCPTestingOrchestrator

        orchestrator = MCPTestingOrchestrator(app_url="http://localhost:3000")
        orchestrator.start = AsyncMock()

        test_spec = {"name": "Test", "steps": [], "assertions": []}

        with patch(
            "src.mcp.langgraph_mcp.execute_test_with_mcp",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = {"status": "passed"}

            await orchestrator.execute_test(test_spec)

            orchestrator.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop(self, mock_env_vars):
        """Test orchestrator stop method."""
        from src.mcp.langgraph_mcp import MCPTestingOrchestrator

        mock_client = MagicMock()
        mock_client.__aexit__ = AsyncMock(return_value=None)

        orchestrator = MCPTestingOrchestrator(app_url="http://localhost:3000")
        orchestrator._client = mock_client
        orchestrator._agent = MagicMock()

        await orchestrator.stop()

        mock_client.__aexit__.assert_called_once_with(None, None, None)
        assert orchestrator._client is None
        assert orchestrator._agent is None

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, mock_env_vars):
        """Test stop when not running."""
        from src.mcp.langgraph_mcp import MCPTestingOrchestrator

        orchestrator = MCPTestingOrchestrator(app_url="http://localhost:3000")

        # Should not raise
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_env_vars):
        """Test orchestrator async context manager."""
        from src.mcp.langgraph_mcp import MCPTestingOrchestrator

        orchestrator = MCPTestingOrchestrator(app_url="http://localhost:3000")
        orchestrator.start = AsyncMock()
        orchestrator.stop = AsyncMock()

        async with orchestrator as o:
            assert o is orchestrator
            orchestrator.start.assert_called_once()

        orchestrator.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, mock_env_vars):
        """Test context manager cleanup on exception."""
        from src.mcp.langgraph_mcp import MCPTestingOrchestrator

        orchestrator = MCPTestingOrchestrator(app_url="http://localhost:3000")
        orchestrator.start = AsyncMock()
        orchestrator.stop = AsyncMock()

        with pytest.raises(ValueError):
            async with orchestrator:
                raise ValueError("Test error")

        orchestrator.stop.assert_called_once()
