"""Tests for the Tool Discovery Agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestToolDiscoveryDataClasses:
    """Tests for Tool Discovery data classes."""

    def test_tool_parameter_creation(self, mock_env_vars):
        """Test ToolParameter dataclass."""
        from src.agents.tool_discovery_agent import ToolParameter, ParameterType

        param = ToolParameter(
            name="user_id",
            type=ParameterType.STRING,
            description="The unique user identifier",
            required=True,
            default=None,
        )

        assert param.name == "user_id"
        assert param.type == ParameterType.STRING
        assert param.required is True

    def test_discovered_tool_creation(self, mock_env_vars):
        """Test DiscoveredTool dataclass."""
        from src.agents.tool_discovery_agent import (
            DiscoveredTool,
            ToolType,
            ToolParameter,
            ToolReturn,
            ParameterType,
            AuthType,
        )

        tool = DiscoveredTool(
            name="get_user",
            description="Retrieves user information by ID",
            tool_type=ToolType.REST,
            source_url="https://api.example.com",
            parameters=[
                ToolParameter(
                    name="user_id",
                    type=ParameterType.STRING,
                    description="User ID",
                    required=True,
                )
            ],
            returns=ToolReturn(
                type=ParameterType.OBJECT,
                description="User object",
            ),
            auth_type=AuthType.BEARER,
        )

        assert tool.name == "get_user"
        assert tool.tool_type == ToolType.REST
        assert len(tool.parameters) == 1
        assert tool.auth_type == AuthType.BEARER

    def test_generated_tool_creation(self, mock_env_vars):
        """Test GeneratedTool dataclass."""
        from src.agents.tool_discovery_agent import GeneratedTool, DiscoveredTool, ToolType, ToolParameter, ParameterType

        discovered = DiscoveredTool(
            name="test_tool",
            description="Test tool",
            tool_type=ToolType.PYTHON,
            source_url="inline",
            parameters=[
                ToolParameter(
                    name="arg1",
                    type=ParameterType.STRING,
                    description="First argument",
                    required=True,
                )
            ],
        )

        generated = GeneratedTool(
            tool=discovered,
            wrapper_code="async def test_tool(): pass",
            language="python",
            dependencies=["httpx"],
            test_cases=["test_basic()"],
        )

        assert generated.tool.name == "test_tool"
        assert "async def" in generated.wrapper_code

    def test_tool_verification_creation(self, mock_env_vars):
        """Test ToolVerification dataclass."""
        from src.agents.tool_discovery_agent import ToolVerification

        verification = ToolVerification(
            tool_name="get_user",
            syntax_valid=True,
            security_passed=True,
            runtime_tested=False,
            test_results=[],
            security_issues=[],
            syntax_errors=[],
        )

        assert verification.syntax_valid is True
        assert verification.security_passed is True
        assert len(verification.test_results) == 0


class TestToolDiscoveryEnums:
    """Tests for Tool Discovery enums."""

    def test_tool_type_values(self, mock_env_vars):
        """Test ToolType enum values."""
        from src.agents.tool_discovery_agent import ToolType

        assert ToolType.OPENAPI.value == "openapi"
        assert ToolType.GRAPHQL.value == "graphql"
        assert ToolType.MCP.value == "mcp"
        assert ToolType.REST.value == "rest"
        assert ToolType.PYTHON.value == "python"

    def test_parameter_type_values(self, mock_env_vars):
        """Test ParameterType enum values."""
        from src.agents.tool_discovery_agent import ParameterType

        assert ParameterType.STRING.value == "string"
        assert ParameterType.INTEGER.value == "integer"
        assert ParameterType.BOOLEAN.value == "boolean"
        assert ParameterType.OBJECT.value == "object"
        assert ParameterType.ARRAY.value == "array"

    def test_auth_type_values(self, mock_env_vars):
        """Test AuthType enum values."""
        from src.agents.tool_discovery_agent import AuthType

        assert AuthType.NONE.value == "none"
        assert AuthType.API_KEY.value == "api_key"
        assert AuthType.BEARER.value == "bearer"
        assert AuthType.OAUTH2.value == "oauth2"


class TestToolDiscoveryAgent:
    """Tests for ToolDiscoveryAgent class."""

    def test_tool_discovery_agent_initialization(self, mock_env_vars):
        """Test ToolDiscoveryAgent initialization."""
        from src.agents.tool_discovery_agent import ToolDiscoveryAgent

        agent = ToolDiscoveryAgent()

        assert agent is not None
        assert hasattr(agent, 'discover_tools_from_docs')
        assert hasattr(agent, 'generate_tool_wrapper')
        assert hasattr(agent, 'verify_tool')
        assert hasattr(agent, 'integrate_tool')

    @pytest.mark.asyncio
    async def test_discover_tools_from_openapi(self, mock_env_vars):
        """Test discovering tools from OpenAPI spec."""
        from src.agents.tool_discovery_agent import ToolDiscoveryAgent

        agent = ToolDiscoveryAgent()

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='{"tools": []}'
            )

            result = await agent.discover_tools_from_docs(
                doc_content='{"openapi": "3.0.0", "paths": {}}',
                doc_type="openapi",
                source_url="https://api.example.com/openapi.json",
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_verify_tool_syntax(self, mock_env_vars):
        """Test tool verification for syntax."""
        from src.agents.tool_discovery_agent import ToolDiscoveryAgent, DiscoveredTool, GeneratedTool, ToolType, ToolParameter, ParameterType

        agent = ToolDiscoveryAgent()

        # verify_tool expects a GeneratedTool, not DiscoveredTool
        discovered = DiscoveredTool(
            name="valid_tool",
            description="A valid tool",
            tool_type=ToolType.PYTHON,
            source_url="inline",
            parameters=[
                ToolParameter(
                    name="arg1",
                    type=ParameterType.STRING,
                    description="First argument",
                    required=True,
                )
            ],
        )

        generated = GeneratedTool(
            tool=discovered,
            wrapper_code="async def valid_tool(arg1: str) -> str:\n    return arg1",
            language="python",
            dependencies=[],
            test_cases=[],
        )

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='{"syntax_valid": true, "security_passed": true}'
            )

            result = await agent.verify_tool(generated)

            assert result is not None


class TestToolCreatorAgent:
    """Tests for ToolCreatorAgent class."""

    def test_tool_creator_initialization(self, mock_env_vars):
        """Test ToolCreatorAgent initialization."""
        from src.agents.tool_discovery_agent import ToolCreatorAgent

        agent = ToolCreatorAgent()

        assert agent is not None
        assert hasattr(agent, 'identify_capability_gap')
        assert hasattr(agent, 'design_tool')
        assert hasattr(agent, 'implement_tool')

    @pytest.mark.asyncio
    async def test_design_tool(self, mock_env_vars):
        """Test designing a new tool."""
        from src.agents.tool_discovery_agent import ToolCreatorAgent, CapabilityGap

        agent = ToolCreatorAgent()

        gap = CapabilityGap(
            description="JWT validation needed",
            required_capability="jwt_validation",
            context="Authentication flow",
            priority=1,
        )

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='''{
                    "tool_name": "jwt_validator",
                    "description": "Validates JWT tokens",
                    "parameters": [
                        {"name": "token", "type": "string", "required": true}
                    ],
                    "returns": {"type": "object", "description": "Decoded claims"}
                }'''
            )

            result = await agent.design_tool(gap)

            assert result is not None


class TestDFSDTToolSelector:
    """Tests for DFSDTToolSelector class."""

    def test_dfsdt_selector_initialization(self, mock_env_vars):
        """Test DFSDTToolSelector initialization."""
        from src.agents.tool_discovery_agent import DFSDTToolSelector, ToolRegistry

        registry = ToolRegistry()
        selector = DFSDTToolSelector(registry=registry)

        assert selector is not None
        assert hasattr(selector, 'select_tools')
        assert hasattr(selector, 'build_tool_tree')

    @pytest.mark.asyncio
    async def test_select_tool_from_registry(self, mock_env_vars):
        """Test selecting a tool using DFSDT algorithm."""
        from src.agents.tool_discovery_agent import DFSDTToolSelector, DiscoveredTool, ToolType, ToolRegistry, ToolParameter, ParameterType

        registry = ToolRegistry()
        selector = DFSDTToolSelector(registry=registry)

        tools = [
            DiscoveredTool(
                name="tool_a",
                description="Does A",
                tool_type=ToolType.REST,
                source_url="https://api.example.com",
                parameters=[
                    ToolParameter(name="arg1", type=ParameterType.STRING, description="Arg", required=True)
                ],
            ),
            DiscoveredTool(
                name="tool_b",
                description="Does B",
                tool_type=ToolType.REST,
                source_url="https://api.example.com",
                parameters=[
                    ToolParameter(name="arg1", type=ParameterType.STRING, description="Arg", required=True)
                ],
            ),
        ]

        # DFSDTToolSelector doesn't have _call_ai, it uses internal decision logic
        # Mock the _evaluate_tool_relevance method instead if needed
        # For basic test, just verify select_tools returns something
        result = await selector.select_tools(
            task="Get user data",
            available_tools=tools,
        )

        assert result is not None


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_tool_registry_initialization(self, mock_env_vars):
        """Test ToolRegistry initialization."""
        from src.agents.tool_discovery_agent import ToolRegistry

        registry = ToolRegistry()

        assert registry is not None
        assert hasattr(registry, 'register')
        assert hasattr(registry, 'search')
        assert hasattr(registry, 'get')


class TestToolLearning:
    """Tests for ToolLearning dataclass."""

    def test_tool_learning_creation(self, mock_env_vars):
        """Test ToolLearning dataclass."""
        from src.agents.tool_discovery_agent import ToolLearning

        learning = ToolLearning(
            tool_name="get_user",
            success_rate=0.95,
            avg_latency_ms=150.0,
            error_patterns=["timeout", "rate_limit"],
            usage_patterns=["fetch_by_id", "batch_fetch"],
            common_parameters={"limit": 100, "offset": 0},
        )

        assert learning.tool_name == "get_user"
        assert learning.success_rate == 0.95
        assert len(learning.error_patterns) == 2


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_tool_discovery_agent(self, mock_env_vars):
        """Test create_tool_discovery_agent factory."""
        from src.agents.tool_discovery_agent import create_tool_discovery_agent

        agent = create_tool_discovery_agent()

        assert agent is not None

    def test_create_tool_creator_agent(self, mock_env_vars):
        """Test create_tool_creator_agent factory."""
        from src.agents.tool_discovery_agent import create_tool_creator_agent

        agent = create_tool_creator_agent()

        assert agent is not None
