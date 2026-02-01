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
            tool_type=ToolType.REST_API,
            endpoint="/api/users/{user_id}",
            method="GET",
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
            auth_type=AuthType.BEARER_TOKEN,
            source_doc="openapi.json",
        )

        assert tool.name == "get_user"
        assert tool.tool_type == ToolType.REST_API
        assert len(tool.parameters) == 1
        assert tool.auth_type == AuthType.BEARER_TOKEN

    def test_generated_tool_creation(self, mock_env_vars):
        """Test GeneratedTool dataclass."""
        from src.agents.tool_discovery_agent import GeneratedTool, DiscoveredTool, ToolType

        discovered = DiscoveredTool(
            name="test_tool",
            description="Test tool",
            tool_type=ToolType.REST_API,
        )

        generated = GeneratedTool(
            tool=discovered,
            wrapper_code="async def test_tool(): pass",
            test_code="async def test_test_tool(): pass",
            documentation="# Test Tool\n\nA test tool.",
        )

        assert generated.tool.name == "test_tool"
        assert "async def" in generated.wrapper_code

    def test_tool_verification_creation(self, mock_env_vars):
        """Test ToolVerification dataclass."""
        from src.agents.tool_discovery_agent import ToolVerification

        verification = ToolVerification(
            tool_name="get_user",
            is_valid=True,
            syntax_valid=True,
            security_valid=True,
            runtime_tested=False,
            issues=[],
            suggestions=["Add rate limiting"],
        )

        assert verification.is_valid is True
        assert verification.syntax_valid is True
        assert len(verification.suggestions) == 1


class TestToolDiscoveryEnums:
    """Tests for Tool Discovery enums."""

    def test_tool_type_values(self, mock_env_vars):
        """Test ToolType enum values."""
        from src.agents.tool_discovery_agent import ToolType

        assert ToolType.REST_API.value == "rest_api"
        assert ToolType.GRAPHQL.value == "graphql"
        assert ToolType.MCP_TOOL.value == "mcp_tool"
        assert ToolType.PYTHON_FUNCTION.value == "python_function"

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
        assert AuthType.BEARER_TOKEN.value == "bearer_token"
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

        with patch.object(agent, '_fetch_documentation', new_callable=AsyncMock) as mock_fetch:
            with patch.object(agent, '_parse_openapi', new_callable=AsyncMock) as mock_parse:
                mock_fetch.return_value = '{"openapi": "3.0.0", "paths": {}}'
                mock_parse.return_value = []

                result = await agent.discover_tools_from_docs(
                    doc_url="https://api.example.com/openapi.json",
                    org_id="test_org",
                    project_id="test_project",
                )

                assert result is not None

    @pytest.mark.asyncio
    async def test_verify_tool_syntax(self, mock_env_vars):
        """Test tool verification for syntax."""
        from src.agents.tool_discovery_agent import ToolDiscoveryAgent, DiscoveredTool, ToolType

        agent = ToolDiscoveryAgent()

        tool = DiscoveredTool(
            name="valid_tool",
            description="A valid tool",
            tool_type=ToolType.PYTHON_FUNCTION,
        )

        with patch.object(agent, '_check_syntax', return_value=True):
            with patch.object(agent, '_check_security', return_value=True):
                result = await agent.verify_tool(tool)

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
    async def test_identify_capability_gap(self, mock_env_vars):
        """Test identifying capability gaps."""
        from src.agents.tool_discovery_agent import ToolCreatorAgent

        agent = ToolCreatorAgent()

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='''{
                    "gap_identified": true,
                    "description": "Need JWT validation tool",
                    "required_capabilities": ["token_decode", "signature_verify"],
                    "existing_partial_tools": []
                }'''
            )

            result = await agent.identify_capability_gap(
                task_description="Validate JWT tokens and extract claims",
            )

            assert result is not None
            mock_ai.assert_called()

    @pytest.mark.asyncio
    async def test_design_tool(self, mock_env_vars):
        """Test designing a new tool."""
        from src.agents.tool_discovery_agent import ToolCreatorAgent, CapabilityGap

        agent = ToolCreatorAgent()

        gap = CapabilityGap(
            description="JWT validation needed",
            required_capabilities=["decode", "verify"],
            suggested_approach="Use PyJWT library",
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
        from src.agents.tool_discovery_agent import DFSDTToolSelector

        selector = DFSDTToolSelector()

        assert selector is not None
        assert hasattr(selector, 'select_tool')
        assert hasattr(selector, 'build_decision_tree')

    @pytest.mark.asyncio
    async def test_select_tool_from_registry(self, mock_env_vars):
        """Test selecting a tool using DFSDT algorithm."""
        from src.agents.tool_discovery_agent import DFSDTToolSelector, DiscoveredTool, ToolType

        selector = DFSDTToolSelector()

        tools = [
            DiscoveredTool(name="tool_a", description="Does A", tool_type=ToolType.REST_API),
            DiscoveredTool(name="tool_b", description="Does B", tool_type=ToolType.REST_API),
        ]

        with patch.object(selector, '_build_decision_tree', new_callable=AsyncMock) as mock_tree:
            with patch.object(selector, '_traverse_tree', new_callable=AsyncMock) as mock_traverse:
                mock_tree.return_value = {"root": "task_type"}
                mock_traverse.return_value = tools[0]

                result = await selector.select_tool(
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
        assert hasattr(registry, 'get_by_capability')

    def test_get_tool_registry_singleton(self, mock_env_vars):
        """Test get_tool_registry returns singleton."""
        from src.agents.tool_discovery_agent import get_tool_registry

        registry1 = get_tool_registry()
        registry2 = get_tool_registry()

        assert registry1 is registry2


class TestToolLearning:
    """Tests for ToolLearning dataclass."""

    def test_tool_learning_creation(self, mock_env_vars):
        """Test ToolLearning dataclass."""
        from src.agents.tool_discovery_agent import ToolLearning

        learning = ToolLearning(
            tool_name="get_user",
            success_rate=0.95,
            avg_latency_ms=150.0,
            common_errors=["timeout", "rate_limit"],
            usage_patterns=["fetch_by_id", "batch_fetch"],
            optimization_suggestions=["Add caching", "Implement retry"],
        )

        assert learning.tool_name == "get_user"
        assert learning.success_rate == 0.95
        assert len(learning.common_errors) == 2


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
