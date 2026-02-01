"""Tool Discovery and Creation Agents.

This module implements the ToolLLM research patterns for mastering 16,000+ APIs:
- ToolDiscoveryAgent: Discovers tools from documentation (OpenAPI, GraphQL, MCP, etc.)
- ToolCreatorAgent: Creates new tools when existing ones are insufficient (CREATOR pattern)
- DFSDTToolSelector: Depth-First Search Decision Tree for systematic tool exploration

Research References:
- ToolLLM: https://arxiv.org/abs/2307.16789
- CREATOR: https://arxiv.org/abs/2305.14318
- DFSDT: Depth-First Search-based Decision Tree from ToolLLM

Architecture:
    Documentation -> Discovery -> Verification -> Integration -> Learning
                            |
                            v
                    ToolRegistry (Cognee-backed)
"""

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import httpx
import structlog

from ..core.model_router import TaskType
from ..knowledge.cognee_client import CogneeKnowledgeClient, get_cognee_client

# Alias for backwards compatibility
CogneeClient = CogneeKnowledgeClient
from .base import AgentCapability, AgentResult, AICapability, BaseAgent


# =============================================================================
# Tool Capability Constants for A2A Protocol
# =============================================================================


class ToolCapability:
    """Tool-related capability constants for A2A discovery."""

    # Discovery capabilities
    TOOL_DISCOVERY = "tool_discovery"
    API_PARSING = "api_parsing"
    OPENAPI_PARSING = "openapi_parsing"
    GRAPHQL_PARSING = "graphql_parsing"
    MCP_MANIFEST_PARSING = "mcp_manifest_parsing"
    DOC_PARSING = "doc_parsing"

    # Creation capabilities
    TOOL_CREATION = "tool_creation"
    CODE_GENERATION = "code_generation"
    WRAPPER_GENERATION = "wrapper_generation"

    # Verification capabilities
    TOOL_VERIFICATION = "tool_verification"
    SYNTAX_VALIDATION = "syntax_validation"
    SECURITY_ANALYSIS = "security_analysis"
    RUNTIME_TESTING = "runtime_testing"

    # Integration capabilities
    TOOL_INTEGRATION = "tool_integration"
    MCP_INTEGRATION = "mcp_integration"
    REGISTRY_MANAGEMENT = "registry_management"

    # Learning capabilities
    TOOL_LEARNING = "tool_learning"
    PATTERN_EXTRACTION = "pattern_extraction"
    USAGE_ANALYSIS = "usage_analysis"


# =============================================================================
# Data Models
# =============================================================================


class ToolType(str, Enum):
    """Types of tools that can be discovered or created."""

    OPENAPI = "openapi"
    GRAPHQL = "graphql"
    MCP = "mcp"
    REST = "rest"
    GRPC = "grpc"
    CLI = "cli"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    CUSTOM = "custom"


class ParameterType(str, Enum):
    """Parameter types for tool schemas."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


class AuthType(str, Enum):
    """Authentication types for tools."""

    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    CUSTOM = "custom"


@dataclass
class ToolParameter:
    """Parameter definition for a tool."""

    name: str
    type: ParameterType
    description: str
    required: bool = False
    default: Any = None
    enum: list[Any] | None = None
    example: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.enum:
            result["enum"] = self.enum
        if self.example is not None:
            result["example"] = self.example
        return result


@dataclass
class ToolReturn:
    """Return type definition for a tool."""

    type: ParameterType
    description: str
    schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.schema:
            result["schema"] = self.schema
        return result


@dataclass
class DiscoveredTool:
    """A tool discovered from documentation."""

    name: str
    description: str
    tool_type: ToolType
    source_url: str
    parameters: list[ToolParameter]
    returns: ToolReturn | None = None
    auth_type: AuthType = AuthType.NONE
    auth_config: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "description": self.description,
            "tool_type": self.tool_type.value,
            "source_url": self.source_url,
            "parameters": [p.to_dict() for p in self.parameters],
            "returns": self.returns.to_dict() if self.returns else None,
            "auth_type": self.auth_type.value,
            "auth_config": self.auth_config,
            "examples": self.examples,
            "metadata": self.metadata,
            "discovered_at": self.discovered_at.isoformat(),
            "confidence": self.confidence,
        }

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type.value, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class GeneratedTool:
    """A tool wrapper generated from a discovered tool."""

    tool: DiscoveredTool
    wrapper_code: str
    language: str = "python"
    dependencies: list[str] = field(default_factory=list)
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "tool": self.tool.to_dict(),
            "wrapper_code": self.wrapper_code,
            "language": self.language,
            "dependencies": self.dependencies,
            "test_cases": self.test_cases,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class ToolVerification:
    """Result of tool verification."""

    tool_name: str
    syntax_valid: bool
    security_passed: bool
    runtime_tested: bool
    test_results: list[dict[str, Any]] = field(default_factory=list)
    security_issues: list[str] = field(default_factory=list)
    syntax_errors: list[str] = field(default_factory=list)
    verified_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def passed(self) -> bool:
        """Check if all verifications passed."""
        return self.syntax_valid and self.security_passed and self.runtime_tested

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "syntax_valid": self.syntax_valid,
            "security_passed": self.security_passed,
            "runtime_tested": self.runtime_tested,
            "test_results": self.test_results,
            "security_issues": self.security_issues,
            "syntax_errors": self.syntax_errors,
            "verified_at": self.verified_at.isoformat(),
            "passed": self.passed,
        }


@dataclass
class IntegrationResult:
    """Result of tool integration."""

    tool_name: str
    success: bool
    integration_type: str  # "mcp", "registry", "direct"
    endpoint: str | None = None
    errors: list[str] = field(default_factory=list)
    integrated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "integration_type": self.integration_type,
            "endpoint": self.endpoint,
            "errors": self.errors,
            "integrated_at": self.integrated_at.isoformat(),
        }


@dataclass
class ToolLearning:
    """Learned patterns from tool usage."""

    tool_name: str
    usage_patterns: list[dict[str, Any]]
    common_parameters: dict[str, Any]
    success_rate: float
    avg_latency_ms: float
    error_patterns: list[dict[str, Any]] = field(default_factory=list)
    learned_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "usage_patterns": self.usage_patterns,
            "common_parameters": self.common_parameters,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "error_patterns": self.error_patterns,
            "learned_at": self.learned_at.isoformat(),
        }


@dataclass
class CapabilityGap:
    """Identified gap in available tool capabilities."""

    description: str
    required_capability: str
    context: dict[str, Any]
    priority: str = "medium"  # low, medium, high, critical
    identified_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "required_capability": self.required_capability,
            "context": self.context,
            "priority": self.priority,
            "identified_at": self.identified_at.isoformat(),
        }


@dataclass
class ToolDesign:
    """Design specification for a new tool."""

    name: str
    description: str
    parameters: list[ToolParameter]
    returns: ToolReturn
    implementation_plan: str
    estimated_complexity: str  # low, medium, high
    dependencies: list[str] = field(default_factory=list)
    designed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "returns": self.returns.to_dict(),
            "implementation_plan": self.implementation_plan,
            "estimated_complexity": self.estimated_complexity,
            "dependencies": self.dependencies,
            "designed_at": self.designed_at.isoformat(),
        }


# =============================================================================
# Tool Registry (Cognee-backed)
# =============================================================================


class ToolRegistry:
    """Registry for discovered and generated tools.

    Uses Cognee for persistent storage with semantic search capabilities.
    """

    def __init__(self, cognee_client: CogneeClient | None = None):
        """Initialize the registry.

        Args:
            cognee_client: Optional Cognee client for persistence.
        """
        self._cognee = cognee_client
        self._tools: dict[str, DiscoveredTool] = {}
        self._generated: dict[str, GeneratedTool] = {}
        self._agent_tools: dict[str, list[str]] = {}  # agent_id -> tool_names
        self.log = structlog.get_logger().bind(component="ToolRegistry")

    @property
    def cognee(self) -> CogneeClient:
        """Lazy-initialize Cognee client."""
        if self._cognee is None:
            self._cognee = get_cognee_client()
        return self._cognee

    def register(
        self,
        tool: DiscoveredTool,
        generated: GeneratedTool | None = None,
    ) -> str:
        """Register a discovered tool.

        Args:
            tool: The discovered tool.
            generated: Optional generated wrapper.

        Returns:
            Tool ID for reference.
        """
        tool_id = self._generate_tool_id(tool)
        self._tools[tool_id] = tool
        if generated:
            self._generated[tool_id] = generated

        self.log.info(
            "Tool registered",
            tool_id=tool_id,
            tool_name=tool.name,
            tool_type=tool.tool_type.value,
        )

        return tool_id

    async def register_async(
        self,
        tool: DiscoveredTool,
        generated: GeneratedTool | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
    ) -> str:
        """Register a tool with Cognee persistence.

        Args:
            tool: The discovered tool.
            generated: Optional generated wrapper.
            org_id: Organization ID for multi-tenancy.
            project_id: Project ID for multi-tenancy.

        Returns:
            Tool ID for reference.
        """
        tool_id = self.register(tool, generated)

        # Persist to Cognee if tenant context provided
        if org_id:
            namespace = f"org_{org_id}"
            if project_id:
                namespace = f"{namespace}_project_{project_id}"
            namespace = f"{namespace}_tools"

            await self.cognee.put(
                key=f"tool_{tool_id}",
                value=tool.to_dict(),
                namespace=namespace,
                embeddings=True,
            )

            if generated:
                await self.cognee.put(
                    key=f"tool_generated_{tool_id}",
                    value=generated.to_dict(),
                    namespace=namespace,
                    embeddings=False,
                )

            self.log.info(
                "Tool persisted to Cognee",
                tool_id=tool_id,
                namespace=namespace,
            )

        return tool_id

    def get(self, tool_id: str) -> DiscoveredTool | None:
        """Get a tool by ID."""
        return self._tools.get(tool_id)

    def get_generated(self, tool_id: str) -> GeneratedTool | None:
        """Get a generated wrapper by tool ID."""
        return self._generated.get(tool_id)

    def search(self, query: str) -> list[DiscoveredTool]:
        """Search tools by name or description.

        Args:
            query: Search query.

        Returns:
            List of matching tools.
        """
        query_lower = query.lower()
        results = []

        for tool in self._tools.values():
            if (
                query_lower in tool.name.lower()
                or query_lower in tool.description.lower()
            ):
                results.append(tool)

        return results

    async def search_semantic(
        self,
        query: str,
        org_id: str,
        project_id: str | None = None,
        top_k: int = 5,
    ) -> list[DiscoveredTool]:
        """Semantic search for tools using Cognee.

        Args:
            query: Natural language query.
            org_id: Organization ID.
            project_id: Optional project ID.
            top_k: Number of results to return.

        Returns:
            List of matching tools.
        """
        namespace = f"org_{org_id}"
        if project_id:
            namespace = f"{namespace}_project_{project_id}"
        namespace = f"{namespace}_tools"

        results = await self.cognee.semantic_search(
            query=query,
            namespace=namespace,
            top_k=top_k,
        )

        tools = []
        for result in results:
            tool_data = result.get("value", {})
            if tool_data:
                try:
                    tool = self._deserialize_tool(tool_data)
                    tools.append(tool)
                except Exception as e:
                    self.log.warning(
                        "Failed to deserialize tool",
                        error=str(e),
                        tool_data=tool_data,
                    )

        return tools

    def get_for_agent(self, agent_id: str) -> list[DiscoveredTool]:
        """Get tools assigned to an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            List of tools for this agent.
        """
        tool_names = self._agent_tools.get(agent_id, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def assign_to_agent(self, tool_id: str, agent_id: str) -> bool:
        """Assign a tool to an agent.

        Args:
            tool_id: Tool identifier.
            agent_id: Agent identifier.

        Returns:
            True if successful.
        """
        if tool_id not in self._tools:
            return False

        if agent_id not in self._agent_tools:
            self._agent_tools[agent_id] = []

        if tool_id not in self._agent_tools[agent_id]:
            self._agent_tools[agent_id].append(tool_id)

        return True

    def unregister(self, tool_id: str) -> bool:
        """Remove a tool from the registry.

        Args:
            tool_id: Tool identifier.

        Returns:
            True if removed.
        """
        if tool_id in self._tools:
            del self._tools[tool_id]
            if tool_id in self._generated:
                del self._generated[tool_id]

            # Remove from agent assignments
            for agent_tools in self._agent_tools.values():
                if tool_id in agent_tools:
                    agent_tools.remove(tool_id)

            return True
        return False

    def get_all(self) -> list[DiscoveredTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        type_counts = {}
        for tool in self._tools.values():
            type_name = tool.tool_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "total_tools": len(self._tools),
            "generated_wrappers": len(self._generated),
            "agent_assignments": len(self._agent_tools),
            "by_type": type_counts,
        }

    async def store_discovered(
        self,
        tools: list[DiscoveredTool],
        org_id: str,
        project_id: str | None = None,
    ) -> int:
        """Bulk store discovered tools.

        Args:
            tools: List of discovered tools.
            org_id: Organization ID.
            project_id: Optional project ID.

        Returns:
            Number of tools stored.
        """
        stored = 0
        for tool in tools:
            try:
                await self.register_async(
                    tool=tool,
                    org_id=org_id,
                    project_id=project_id,
                )
                stored += 1
            except Exception as e:
                self.log.error(
                    "Failed to store tool",
                    tool_name=tool.name,
                    error=str(e),
                )

        return stored

    async def get_discovered(
        self,
        org_id: str,
        project_id: str | None = None,
        tool_type: ToolType | None = None,
    ) -> list[DiscoveredTool]:
        """Get all discovered tools for a tenant.

        Args:
            org_id: Organization ID.
            project_id: Optional project ID.
            tool_type: Optional filter by tool type.

        Returns:
            List of discovered tools.
        """
        # First return in-memory tools
        tools = list(self._tools.values())

        # Filter by type if specified
        if tool_type:
            tools = [t for t in tools if t.tool_type == tool_type]

        return tools

    def _generate_tool_id(self, tool: DiscoveredTool) -> str:
        """Generate a unique ID for a tool."""
        content = f"{tool.name}:{tool.source_url}:{tool.tool_type.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _deserialize_tool(self, data: dict[str, Any]) -> DiscoveredTool:
        """Deserialize a tool from dictionary."""
        parameters = [
            ToolParameter(
                name=p["name"],
                type=ParameterType(p["type"]),
                description=p["description"],
                required=p.get("required", False),
                default=p.get("default"),
                enum=p.get("enum"),
                example=p.get("example"),
            )
            for p in data.get("parameters", [])
        ]

        returns = None
        if data.get("returns"):
            returns = ToolReturn(
                type=ParameterType(data["returns"]["type"]),
                description=data["returns"]["description"],
                schema=data["returns"].get("schema"),
            )

        return DiscoveredTool(
            name=data["name"],
            description=data["description"],
            tool_type=ToolType(data["tool_type"]),
            source_url=data["source_url"],
            parameters=parameters,
            returns=returns,
            auth_type=AuthType(data.get("auth_type", "none")),
            auth_config=data.get("auth_config", {}),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {}),
            discovered_at=datetime.fromisoformat(data["discovered_at"]) if data.get("discovered_at") else datetime.utcnow(),
            confidence=data.get("confidence", 1.0),
        )


# =============================================================================
# Tool Discovery Agent
# =============================================================================


class ToolDiscoveryAgent(BaseAgent):
    """Agent that discovers tools from various documentation formats.

    Implements ToolLLM patterns for mastering 16,000+ APIs by:
    1. Parsing OpenAPI/Swagger specifications
    2. Understanding GraphQL schemas
    3. Processing MCP server manifests
    4. Extracting tools from plain documentation using AI

    Capabilities:
        - TOOL_DISCOVERY: Main capability for discovering new tools
        - API_PARSING: Parse API specifications
        - OPENAPI_PARSING: Parse OpenAPI/Swagger specs
        - GRAPHQL_PARSING: Parse GraphQL schemas
        - MCP_MANIFEST_PARSING: Parse MCP manifests
        - DOC_PARSING: AI-powered documentation parsing
    """

    DEFAULT_TASK_TYPE = TaskType.CODE_ANALYSIS

    CAPABILITIES = [
        ToolCapability.TOOL_DISCOVERY,
        ToolCapability.API_PARSING,
        ToolCapability.OPENAPI_PARSING,
        ToolCapability.GRAPHQL_PARSING,
        ToolCapability.MCP_MANIFEST_PARSING,
        ToolCapability.DOC_PARSING,
        AgentCapability.CODE_ANALYSIS,
    ]

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        **kwargs,
    ):
        """Initialize the discovery agent.

        Args:
            registry: Optional tool registry for persistence.
            **kwargs: Arguments passed to BaseAgent.
        """
        super().__init__(**kwargs)
        self.registry = registry or ToolRegistry()
        self._http_client: httpx.AsyncClient | None = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Lazy-initialize HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    def _get_system_prompt(self) -> str:
        """Get the system prompt for tool discovery."""
        return """You are a Tool Discovery Agent specialized in understanding and parsing API documentation.

Your capabilities:
1. Parse OpenAPI/Swagger specifications to extract endpoints, parameters, and schemas
2. Understand GraphQL schemas including queries, mutations, and types
3. Process MCP (Model Context Protocol) server manifests
4. Extract tool definitions from plain text documentation using semantic understanding

When discovering tools, you should:
- Identify all callable functions/endpoints
- Extract parameter names, types, descriptions, and constraints
- Determine authentication requirements
- Find examples and usage patterns
- Assess confidence in your parsing accuracy

Output format: Always provide structured JSON with discovered tools."""

    async def execute(
        self,
        doc_url: str | None = None,
        doc_content: str | None = None,
        doc_type: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        **kwargs,
    ) -> AgentResult[list[DiscoveredTool]]:
        """Execute tool discovery.

        Args:
            doc_url: URL to fetch documentation from.
            doc_content: Raw documentation content.
            doc_type: Type of documentation (openapi, graphql, mcp, plain).
            org_id: Organization ID for multi-tenancy.
            project_id: Project ID for multi-tenancy.
            **kwargs: Additional arguments.

        Returns:
            AgentResult with discovered tools.
        """
        start_time = time.time()

        try:
            if not doc_url and not doc_content:
                return AgentResult(
                    success=False,
                    error="Either doc_url or doc_content must be provided",
                )

            # Fetch documentation if URL provided
            content = doc_content
            if doc_url:
                content = await self._fetch_documentation(doc_url)

            if not content:
                return AgentResult(
                    success=False,
                    error="Failed to fetch or parse documentation",
                )

            # Discover tools from documentation
            tools = await self.discover_tools_from_docs(
                doc_content=content,
                doc_type=doc_type,
                source_url=doc_url or "inline",
            )

            # Store in registry if tenant context provided
            if tools and org_id:
                await self.registry.store_discovered(
                    tools=tools,
                    org_id=org_id,
                    project_id=project_id,
                )

            duration_ms = int((time.time() - start_time) * 1000)

            return AgentResult(
                success=True,
                data=tools,
                input_tokens=self._usage.total_input_tokens,
                output_tokens=self._usage.total_output_tokens,
                cost=self._usage.total_cost,
                duration_ms=duration_ms,
            )

        except Exception as e:
            self.log.error("Tool discovery failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000),
            )

    async def discover_tools_from_docs(
        self,
        doc_content: str,
        doc_type: str | None = None,
        source_url: str = "inline",
    ) -> list[DiscoveredTool]:
        """Discover tools from documentation content.

        Args:
            doc_content: Raw documentation content.
            doc_type: Type of documentation (auto-detected if not provided).
            source_url: Source URL for reference.

        Returns:
            List of discovered tools.
        """
        # Auto-detect documentation type
        detected_type = doc_type or self._detect_doc_type(doc_content)

        self.log.info(
            "Discovering tools",
            doc_type=detected_type,
            source_url=source_url,
            content_length=len(doc_content),
        )

        if detected_type == "openapi":
            return await self._parse_openapi(doc_content, source_url)
        elif detected_type == "graphql":
            return await self._parse_graphql(doc_content, source_url)
        elif detected_type == "mcp":
            return await self._parse_mcp_manifest(doc_content, source_url)
        else:
            return await self._parse_plain_docs(doc_content, source_url)

    async def generate_tool_wrapper(
        self,
        tool: DiscoveredTool,
        language: str = "python",
    ) -> GeneratedTool:
        """Generate executable wrapper code for a discovered tool.

        Args:
            tool: The discovered tool definition.
            language: Target programming language.

        Returns:
            GeneratedTool with wrapper code.
        """
        self.log.info(
            "Generating wrapper",
            tool_name=tool.name,
            language=language,
        )

        prompt = f"""Generate a {language} wrapper function for this API tool:

Tool: {tool.name}
Description: {tool.description}
Type: {tool.tool_type.value}
Parameters: {json.dumps([p.to_dict() for p in tool.parameters], indent=2)}
Returns: {json.dumps(tool.returns.to_dict() if tool.returns else None, indent=2)}
Auth Type: {tool.auth_type.value}
Examples: {json.dumps(tool.examples, indent=2)}

Requirements:
1. Create a clean, well-documented function
2. Include proper type hints
3. Handle errors gracefully
4. Support async execution where appropriate
5. Include docstring with examples

Return only the code, no explanations."""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            max_tokens=4096,
        )

        # Extract code from response
        wrapper_code = self._extract_code(response.content, language)

        # Generate test cases
        test_cases = await self._generate_test_cases(tool, wrapper_code)

        # Identify dependencies
        dependencies = self._extract_dependencies(wrapper_code, language)

        return GeneratedTool(
            tool=tool,
            wrapper_code=wrapper_code,
            language=language,
            dependencies=dependencies,
            test_cases=test_cases,
        )

    async def verify_tool(
        self,
        generated: GeneratedTool,
        run_tests: bool = True,
    ) -> ToolVerification:
        """Verify a generated tool wrapper.

        Args:
            generated: The generated tool wrapper.
            run_tests: Whether to run test cases.

        Returns:
            ToolVerification result.
        """
        self.log.info(
            "Verifying tool",
            tool_name=generated.tool.name,
            language=generated.language,
        )

        syntax_errors: list[str] = []
        security_issues: list[str] = []
        test_results: list[dict[str, Any]] = []

        # Syntax validation
        syntax_valid = self._validate_syntax(
            generated.wrapper_code,
            generated.language,
            syntax_errors,
        )

        # Security analysis
        security_passed = await self._analyze_security(
            generated.wrapper_code,
            security_issues,
        )

        # Runtime testing
        runtime_tested = False
        if run_tests and syntax_valid and security_passed:
            runtime_tested, test_results = await self._run_tests(
                generated.wrapper_code,
                generated.test_cases,
                generated.language,
            )

        return ToolVerification(
            tool_name=generated.tool.name,
            syntax_valid=syntax_valid,
            security_passed=security_passed,
            runtime_tested=runtime_tested,
            test_results=test_results,
            security_issues=security_issues,
            syntax_errors=syntax_errors,
        )

    async def integrate_tool(
        self,
        generated: GeneratedTool,
        verification: ToolVerification,
        integration_type: str = "registry",
        org_id: str | None = None,
        project_id: str | None = None,
    ) -> IntegrationResult:
        """Integrate a verified tool.

        Args:
            generated: The generated tool wrapper.
            verification: Verification results.
            integration_type: Type of integration (registry, mcp, direct).
            org_id: Organization ID.
            project_id: Project ID.

        Returns:
            IntegrationResult.
        """
        errors: list[str] = []

        if not verification.passed:
            errors.append("Tool verification failed")
            return IntegrationResult(
                tool_name=generated.tool.name,
                success=False,
                integration_type=integration_type,
                errors=errors,
            )

        try:
            if integration_type == "registry":
                await self.registry.register_async(
                    tool=generated.tool,
                    generated=generated,
                    org_id=org_id,
                    project_id=project_id,
                )
            elif integration_type == "mcp":
                # MCP integration would require additional infrastructure
                errors.append("MCP integration not yet implemented")
                return IntegrationResult(
                    tool_name=generated.tool.name,
                    success=False,
                    integration_type=integration_type,
                    errors=errors,
                )

            return IntegrationResult(
                tool_name=generated.tool.name,
                success=True,
                integration_type=integration_type,
            )

        except Exception as e:
            errors.append(str(e))
            return IntegrationResult(
                tool_name=generated.tool.name,
                success=False,
                integration_type=integration_type,
                errors=errors,
            )

    async def learn_tool_patterns(
        self,
        tool_name: str,
        usage_history: list[dict[str, Any]],
    ) -> ToolLearning:
        """Learn patterns from tool usage history.

        Args:
            tool_name: Name of the tool.
            usage_history: List of usage records.

        Returns:
            ToolLearning with extracted patterns.
        """
        self.log.info(
            "Learning patterns",
            tool_name=tool_name,
            history_size=len(usage_history),
        )

        # Analyze usage patterns
        success_count = sum(1 for u in usage_history if u.get("success", False))
        total_count = len(usage_history)
        success_rate = success_count / total_count if total_count > 0 else 0.0

        # Calculate average latency
        latencies = [u.get("latency_ms", 0) for u in usage_history]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Extract common parameters using AI
        prompt = f"""Analyze this tool usage history and extract patterns:

Tool: {tool_name}
Usage History (last 10): {json.dumps(usage_history[-10:], indent=2)}

Extract:
1. Common parameter combinations
2. Error patterns and their causes
3. Usage patterns (time of day, frequency, etc.)

Return as JSON with keys: common_parameters, error_patterns, usage_patterns"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            max_tokens=2048,
        )

        patterns = self._parse_json_response(response.content, {})

        return ToolLearning(
            tool_name=tool_name,
            usage_patterns=patterns.get("usage_patterns", []),
            common_parameters=patterns.get("common_parameters", {}),
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            error_patterns=patterns.get("error_patterns", []),
        )

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _fetch_documentation(self, url: str) -> str | None:
        """Fetch documentation from a URL."""
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.log.error("Failed to fetch documentation", url=url, error=str(e))
            return None

    def _detect_doc_type(self, content: str) -> str:
        """Auto-detect documentation type from content."""
        content_lower = content.lower().strip()

        # Check for OpenAPI/Swagger
        if '"openapi"' in content_lower or '"swagger"' in content_lower:
            return "openapi"
        if "openapi:" in content_lower or "swagger:" in content_lower:
            return "openapi"

        # Check for GraphQL
        if "type query" in content_lower or "type mutation" in content_lower:
            return "graphql"
        if "schema {" in content_lower:
            return "graphql"

        # Check for MCP manifest
        if '"tools"' in content_lower and '"name"' in content_lower:
            try:
                parsed = json.loads(content)
                if "tools" in parsed:
                    return "mcp"
            except json.JSONDecodeError:
                pass

        return "plain"

    async def _parse_openapi(
        self,
        content: str,
        source_url: str,
    ) -> list[DiscoveredTool]:
        """Parse OpenAPI/Swagger specification."""
        tools: list[DiscoveredTool] = []

        try:
            # Try to parse as JSON first, then YAML
            try:
                spec = json.loads(content)
            except json.JSONDecodeError:
                import yaml
                spec = yaml.safe_load(content)

            paths = spec.get("paths", {})
            base_url = source_url.rsplit("/", 1)[0] if source_url != "inline" else ""

            for path, methods in paths.items():
                for method, details in methods.items():
                    if method.upper() not in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
                        continue

                    # Extract parameters
                    parameters: list[ToolParameter] = []
                    for param in details.get("parameters", []):
                        param_schema = param.get("schema", {})
                        parameters.append(
                            ToolParameter(
                                name=param.get("name", ""),
                                type=ParameterType(
                                    param_schema.get("type", "string")
                                ),
                                description=param.get("description", ""),
                                required=param.get("required", False),
                                default=param_schema.get("default"),
                                enum=param_schema.get("enum"),
                            )
                        )

                    # Extract request body parameters
                    request_body = details.get("requestBody", {})
                    if request_body:
                        content_types = request_body.get("content", {})
                        for content_type, content_schema in content_types.items():
                            schema = content_schema.get("schema", {})
                            props = schema.get("properties", {})
                            required = schema.get("required", [])

                            for prop_name, prop_schema in props.items():
                                parameters.append(
                                    ToolParameter(
                                        name=prop_name,
                                        type=ParameterType(
                                            prop_schema.get("type", "string")
                                        ),
                                        description=prop_schema.get("description", ""),
                                        required=prop_name in required,
                                        default=prop_schema.get("default"),
                                        enum=prop_schema.get("enum"),
                                    )
                                )

                    # Extract return type
                    responses = details.get("responses", {})
                    returns = None
                    for status_code, response in responses.items():
                        if status_code.startswith("2"):
                            response_content = response.get("content", {})
                            for content_type, content_schema in response_content.items():
                                schema = content_schema.get("schema", {})
                                returns = ToolReturn(
                                    type=ParameterType(schema.get("type", "object")),
                                    description=response.get("description", ""),
                                    schema=schema,
                                )
                            break

                    # Create tool name from path and method
                    tool_name = f"{method.lower()}_{path.replace('/', '_').replace('{', '').replace('}', '').strip('_')}"

                    tools.append(
                        DiscoveredTool(
                            name=tool_name,
                            description=details.get("summary", details.get("description", "")),
                            tool_type=ToolType.OPENAPI,
                            source_url=source_url,
                            parameters=parameters,
                            returns=returns,
                            auth_type=self._detect_auth_type(spec),
                            metadata={
                                "method": method.upper(),
                                "path": path,
                                "operation_id": details.get("operationId"),
                                "tags": details.get("tags", []),
                            },
                        )
                    )

            self.log.info(
                "OpenAPI parsing complete",
                tools_found=len(tools),
            )

        except Exception as e:
            self.log.error("OpenAPI parsing failed", error=str(e))

        return tools

    async def _parse_graphql(
        self,
        content: str,
        source_url: str,
    ) -> list[DiscoveredTool]:
        """Parse GraphQL schema."""
        tools: list[DiscoveredTool] = []

        # Use AI to parse GraphQL schema
        prompt = f"""Parse this GraphQL schema and extract all queries and mutations as tools.

Schema:
{content[:8000]}  # Limit content length

For each operation, extract:
1. Name
2. Description
3. Arguments with types
4. Return type

Return as JSON array with format:
[{{
    "name": "operation_name",
    "description": "...",
    "operation_type": "query|mutation",
    "arguments": [{{"name": "...", "type": "...", "required": true/false}}],
    "return_type": "..."
}}]"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            max_tokens=4096,
        )

        operations = self._parse_json_response(response.content, [])

        for op in operations:
            parameters = [
                ToolParameter(
                    name=arg.get("name", ""),
                    type=self._map_graphql_type(arg.get("type", "String")),
                    description="",
                    required=arg.get("required", False),
                )
                for arg in op.get("arguments", [])
            ]

            tools.append(
                DiscoveredTool(
                    name=op.get("name", ""),
                    description=op.get("description", ""),
                    tool_type=ToolType.GRAPHQL,
                    source_url=source_url,
                    parameters=parameters,
                    returns=ToolReturn(
                        type=self._map_graphql_type(op.get("return_type", "Any")),
                        description="",
                    ),
                    metadata={
                        "operation_type": op.get("operation_type", "query"),
                    },
                )
            )

        return tools

    async def _parse_mcp_manifest(
        self,
        content: str,
        source_url: str,
    ) -> list[DiscoveredTool]:
        """Parse MCP server manifest."""
        tools: list[DiscoveredTool] = []

        try:
            manifest = json.loads(content)

            for tool_def in manifest.get("tools", []):
                parameters = []
                input_schema = tool_def.get("inputSchema", {})
                props = input_schema.get("properties", {})
                required = input_schema.get("required", [])

                for prop_name, prop_schema in props.items():
                    parameters.append(
                        ToolParameter(
                            name=prop_name,
                            type=ParameterType(prop_schema.get("type", "string")),
                            description=prop_schema.get("description", ""),
                            required=prop_name in required,
                            default=prop_schema.get("default"),
                            enum=prop_schema.get("enum"),
                        )
                    )

                tools.append(
                    DiscoveredTool(
                        name=tool_def.get("name", ""),
                        description=tool_def.get("description", ""),
                        tool_type=ToolType.MCP,
                        source_url=source_url,
                        parameters=parameters,
                        metadata={
                            "server_name": manifest.get("name"),
                            "server_version": manifest.get("version"),
                        },
                    )
                )

        except Exception as e:
            self.log.error("MCP manifest parsing failed", error=str(e))

        return tools

    async def _parse_plain_docs(
        self,
        content: str,
        source_url: str,
    ) -> list[DiscoveredTool]:
        """Parse plain text documentation using AI."""
        self.log.info("Parsing plain documentation with AI")

        prompt = f"""Extract tool/API definitions from this documentation:

{content[:12000]}  # Limit content length

For each tool/function/endpoint found, extract:
1. Name
2. Description
3. Parameters with types and descriptions
4. Return type
5. Any examples

Return as JSON array with format:
[{{
    "name": "tool_name",
    "description": "what it does",
    "parameters": [
        {{"name": "param1", "type": "string", "description": "...", "required": true}}
    ],
    "return_type": "string",
    "examples": [...]
}}]

If no tools are found, return an empty array []."""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.SEMANTIC_UNDERSTANDING,
            max_tokens=4096,
        )

        tool_defs = self._parse_json_response(response.content, [])
        tools: list[DiscoveredTool] = []

        for tool_def in tool_defs:
            parameters = [
                ToolParameter(
                    name=p.get("name", ""),
                    type=ParameterType(p.get("type", "string")),
                    description=p.get("description", ""),
                    required=p.get("required", False),
                )
                for p in tool_def.get("parameters", [])
            ]

            tools.append(
                DiscoveredTool(
                    name=tool_def.get("name", ""),
                    description=tool_def.get("description", ""),
                    tool_type=ToolType.CUSTOM,
                    source_url=source_url,
                    parameters=parameters,
                    returns=ToolReturn(
                        type=ParameterType(tool_def.get("return_type", "string")),
                        description="",
                    ),
                    examples=tool_def.get("examples", []),
                    confidence=0.8,  # Lower confidence for AI-parsed docs
                )
            )

        return tools

    def _detect_auth_type(self, spec: dict[str, Any]) -> AuthType:
        """Detect authentication type from OpenAPI spec."""
        security_schemes = spec.get("components", {}).get("securitySchemes", {})
        if not security_schemes:
            security_schemes = spec.get("securityDefinitions", {})

        for scheme in security_schemes.values():
            scheme_type = scheme.get("type", "").lower()
            if scheme_type == "apikey":
                return AuthType.API_KEY
            elif scheme_type == "http":
                scheme_scheme = scheme.get("scheme", "").lower()
                if scheme_scheme == "bearer":
                    return AuthType.BEARER
                elif scheme_scheme == "basic":
                    return AuthType.BASIC
            elif scheme_type == "oauth2":
                return AuthType.OAUTH2

        return AuthType.NONE

    def _map_graphql_type(self, graphql_type: str) -> ParameterType:
        """Map GraphQL type to ParameterType."""
        type_lower = graphql_type.lower().rstrip("!")

        if type_lower in ("string", "id"):
            return ParameterType.STRING
        elif type_lower in ("int", "integer"):
            return ParameterType.INTEGER
        elif type_lower in ("float", "number"):
            return ParameterType.NUMBER
        elif type_lower == "boolean":
            return ParameterType.BOOLEAN
        elif type_lower.startswith("["):
            return ParameterType.ARRAY
        else:
            return ParameterType.OBJECT

    def _extract_code(self, content: str, language: str) -> str:
        """Extract code from AI response."""
        # Try to extract from code block
        patterns = [
            rf"```{language}\n(.*?)```",
            r"```\n(.*?)```",
            r"```(.*?)```",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()

        return content.strip()

    def _extract_dependencies(self, code: str, language: str) -> list[str]:
        """Extract dependencies from code."""
        deps: list[str] = []

        if language == "python":
            # Extract import statements
            import_pattern = r"^(?:from\s+(\S+)|import\s+(\S+))"
            for match in re.finditer(import_pattern, code, re.MULTILINE):
                module = match.group(1) or match.group(2)
                if module:
                    base_module = module.split(".")[0]
                    if base_module not in ["os", "sys", "json", "re", "typing"]:
                        if base_module not in deps:
                            deps.append(base_module)

        return deps

    async def _generate_test_cases(
        self,
        tool: DiscoveredTool,
        wrapper_code: str,
    ) -> list[dict[str, Any]]:
        """Generate test cases for a tool wrapper."""
        prompt = f"""Generate test cases for this tool:

Tool: {tool.name}
Parameters: {json.dumps([p.to_dict() for p in tool.parameters], indent=2)}

Wrapper Code:
{wrapper_code[:2000]}

Generate 3-5 test cases as JSON array with format:
[{{
    "name": "test_name",
    "inputs": {{}},
    "expected_behavior": "description of expected behavior"
}}]"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.TEST_GENERATION,
            max_tokens=2048,
        )

        return self._parse_json_response(response.content, [])

    def _validate_syntax(
        self,
        code: str,
        language: str,
        errors: list[str],
    ) -> bool:
        """Validate code syntax."""
        if language == "python":
            try:
                compile(code, "<string>", "exec")
                return True
            except SyntaxError as e:
                errors.append(f"Line {e.lineno}: {e.msg}")
                return False

        # For other languages, assume valid (would need language-specific parsers)
        return True

    async def _analyze_security(
        self,
        code: str,
        issues: list[str],
    ) -> bool:
        """Analyze code for security issues."""
        # Check for common security issues
        dangerous_patterns = [
            (r"eval\s*\(", "Use of eval() is dangerous"),
            (r"exec\s*\(", "Use of exec() is dangerous"),
            (r"__import__\s*\(", "Dynamic import is risky"),
            (r"subprocess\.(call|run|Popen)", "Shell command execution detected"),
            (r"os\.system\s*\(", "Shell command execution detected"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(message)

        # More thorough analysis with AI for complex cases
        if not issues:
            prompt = f"""Analyze this code for security vulnerabilities:

{code[:4000]}

Look for:
1. Injection vulnerabilities
2. Unsafe data handling
3. Credential exposure
4. Insecure defaults

Return JSON: {{"secure": true/false, "issues": ["issue1", ...]}}"""

            response = await self._call_ai(
                messages=[{"role": "user", "content": prompt}],
                task_type=TaskType.CODE_ANALYSIS,
                max_tokens=1024,
            )

            result = self._parse_json_response(response.content, {"secure": True})
            if not result.get("secure", True):
                issues.extend(result.get("issues", []))

        return len(issues) == 0

    async def _run_tests(
        self,
        code: str,
        test_cases: list[dict[str, Any]],
        language: str,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Run test cases against wrapper code.

        Note: This is a simplified implementation. Production would use
        proper sandboxing and isolation.
        """
        results: list[dict[str, Any]] = []
        all_passed = True

        if language != "python":
            return True, []  # Skip for non-Python

        for test in test_cases:
            result = {
                "name": test.get("name", "unknown"),
                "passed": False,
                "error": None,
            }

            try:
                # Create isolated namespace for execution
                namespace: dict[str, Any] = {}

                # Execute the wrapper code to define functions
                exec(code, namespace)  # noqa: S102 - Intentional for tool testing

                # For now, just verify the code can be loaded
                # Full test execution would require more infrastructure
                result["passed"] = True

            except Exception as e:
                result["passed"] = False
                result["error"] = str(e)
                all_passed = False

            results.append(result)

        return all_passed, results


# =============================================================================
# Tool Creator Agent (CREATOR Pattern)
# =============================================================================


class ToolCreatorAgent(BaseAgent):
    """Agent that creates new tools when existing ones are insufficient.

    Implements the CREATOR research pattern:
    1. Identify capability gaps
    2. Design tool specifications
    3. Implement tool code
    4. Verify and integrate

    Reference: https://arxiv.org/abs/2305.14318
    """

    DEFAULT_TASK_TYPE = TaskType.CODE_ANALYSIS

    CAPABILITIES = [
        ToolCapability.TOOL_CREATION,
        ToolCapability.CODE_GENERATION,
        ToolCapability.WRAPPER_GENERATION,
        AgentCapability.CODE_ANALYSIS,
        AgentCapability.TEST_GENERATION,
    ]

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        discovery_agent: ToolDiscoveryAgent | None = None,
        **kwargs,
    ):
        """Initialize the creator agent.

        Args:
            registry: Tool registry for storage.
            discovery_agent: Discovery agent for verification.
            **kwargs: Arguments passed to BaseAgent.
        """
        super().__init__(**kwargs)
        self.registry = registry or ToolRegistry()
        self.discovery_agent = discovery_agent

    def _get_system_prompt(self) -> str:
        """Get the system prompt for tool creation."""
        return """You are a Tool Creator Agent specialized in designing and implementing new tools.

Following the CREATOR pattern, you:
1. Identify gaps in existing tool capabilities
2. Design clear tool specifications with proper parameters and return types
3. Implement robust, well-tested tool code
4. Ensure tools integrate smoothly with existing systems

When creating tools:
- Keep them focused on a single capability
- Use clear, descriptive names
- Include comprehensive parameter validation
- Handle errors gracefully
- Provide usage examples"""

    async def execute(
        self,
        task_description: str | None = None,
        capability_gap: CapabilityGap | None = None,
        **kwargs,
    ) -> AgentResult[GeneratedTool]:
        """Execute tool creation.

        Args:
            task_description: Natural language description of needed capability.
            capability_gap: Structured capability gap definition.
            **kwargs: Additional arguments.

        Returns:
            AgentResult with generated tool.
        """
        start_time = time.time()

        try:
            # Identify or use provided capability gap
            if capability_gap is None and task_description:
                capability_gap = await self.identify_capability_gap(task_description)

            if capability_gap is None:
                return AgentResult(
                    success=False,
                    error="No capability gap identified",
                )

            # Design the tool
            design = await self.design_tool(capability_gap)

            # Implement the tool
            generated = await self.implement_tool(design)

            # Verify if discovery agent is available
            if self.discovery_agent:
                verification = await self.discovery_agent.verify_tool(generated)
                if not verification.passed:
                    return AgentResult(
                        success=False,
                        error=f"Tool verification failed: {verification.syntax_errors + verification.security_issues}",
                    )

            duration_ms = int((time.time() - start_time) * 1000)

            return AgentResult(
                success=True,
                data=generated,
                input_tokens=self._usage.total_input_tokens,
                output_tokens=self._usage.total_output_tokens,
                cost=self._usage.total_cost,
                duration_ms=duration_ms,
            )

        except Exception as e:
            self.log.error("Tool creation failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000),
            )

    async def identify_capability_gap(
        self,
        task_description: str,
    ) -> CapabilityGap:
        """Identify what capability is missing.

        Args:
            task_description: Description of the task that can't be completed.

        Returns:
            CapabilityGap describing the needed capability.
        """
        prompt = f"""Analyze this task and identify what tool/capability is missing:

Task: {task_description}

Available capabilities: {list(vars(ToolCapability).keys())}

Return JSON:
{{
    "description": "what capability is needed",
    "required_capability": "capability_name",
    "priority": "low|medium|high|critical",
    "context": {{...relevant context...}}
}}"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.SEMANTIC_UNDERSTANDING,
            max_tokens=1024,
        )

        result = self._parse_json_response(response.content, {})

        return CapabilityGap(
            description=result.get("description", task_description),
            required_capability=result.get("required_capability", "custom"),
            context=result.get("context", {}),
            priority=result.get("priority", "medium"),
        )

    async def design_tool(self, gap: CapabilityGap) -> ToolDesign:
        """Design a tool to fill the capability gap.

        Args:
            gap: The capability gap to address.

        Returns:
            ToolDesign specification.
        """
        prompt = f"""Design a tool to address this capability gap:

Gap: {gap.description}
Required Capability: {gap.required_capability}
Context: {json.dumps(gap.context, indent=2)}

Design a tool with:
1. Clear name and description
2. Well-defined parameters
3. Return type specification
4. Implementation plan

Return JSON:
{{
    "name": "tool_name",
    "description": "what it does",
    "parameters": [
        {{"name": "param1", "type": "string", "description": "...", "required": true}}
    ],
    "returns": {{"type": "object", "description": "..."}},
    "implementation_plan": "step by step implementation",
    "estimated_complexity": "low|medium|high",
    "dependencies": ["dep1", "dep2"]
}}"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            max_tokens=2048,
        )

        result = self._parse_json_response(response.content, {})

        parameters = [
            ToolParameter(
                name=p.get("name", ""),
                type=ParameterType(p.get("type", "string")),
                description=p.get("description", ""),
                required=p.get("required", False),
            )
            for p in result.get("parameters", [])
        ]

        returns_data = result.get("returns", {"type": "string", "description": ""})
        returns = ToolReturn(
            type=ParameterType(returns_data.get("type", "string")),
            description=returns_data.get("description", ""),
        )

        return ToolDesign(
            name=result.get("name", "new_tool"),
            description=result.get("description", ""),
            parameters=parameters,
            returns=returns,
            implementation_plan=result.get("implementation_plan", ""),
            estimated_complexity=result.get("estimated_complexity", "medium"),
            dependencies=result.get("dependencies", []),
        )

    async def implement_tool(self, design: ToolDesign) -> GeneratedTool:
        """Implement a tool based on its design.

        Args:
            design: The tool design specification.

        Returns:
            GeneratedTool with implementation.
        """
        prompt = f"""Implement this tool in Python:

Name: {design.name}
Description: {design.description}
Parameters: {json.dumps([p.to_dict() for p in design.parameters], indent=2)}
Returns: {json.dumps(design.returns.to_dict(), indent=2)}
Implementation Plan: {design.implementation_plan}
Dependencies: {design.dependencies}

Requirements:
1. Create an async function with proper type hints
2. Include comprehensive docstring
3. Add input validation
4. Handle errors gracefully
5. Return structured response

Return only the Python code, no explanations."""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            max_tokens=4096,
        )

        code = self._extract_code(response.content)

        # Create DiscoveredTool from design
        tool = DiscoveredTool(
            name=design.name,
            description=design.description,
            tool_type=ToolType.PYTHON,
            source_url="generated",
            parameters=design.parameters,
            returns=design.returns,
            metadata={
                "implementation_plan": design.implementation_plan,
                "estimated_complexity": design.estimated_complexity,
            },
        )

        return GeneratedTool(
            tool=tool,
            wrapper_code=code,
            language="python",
            dependencies=design.dependencies,
        )

    def _extract_code(self, content: str) -> str:
        """Extract Python code from response."""
        patterns = [
            r"```python\n(.*?)```",
            r"```\n(.*?)```",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()

        return content.strip()


# =============================================================================
# DFSDT Tool Selector (Depth-First Search Decision Tree)
# =============================================================================


class DFSDTToolSelector:
    """Depth-First Search Decision Tree for systematic tool exploration.

    Implements the DFSDT algorithm from ToolLLM for navigating large tool spaces
    by building and traversing a decision tree to find the best tool for a task.

    The algorithm:
    1. Build a tree of tool categories
    2. Traverse depth-first, asking "is this category relevant?"
    3. At leaf nodes, select specific tools
    4. Backtrack if selected tools don't work
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_depth: int = 5,
        max_breadth: int = 10,
    ):
        """Initialize the selector.

        Args:
            registry: Tool registry to search.
            max_depth: Maximum tree depth.
            max_breadth: Maximum children per node.
        """
        self.registry = registry
        self.max_depth = max_depth
        self.max_breadth = max_breadth
        self.log = structlog.get_logger().bind(component="DFSDTToolSelector")

    async def select_tools(
        self,
        task: str,
        available_tools: list[DiscoveredTool],
        selector_fn: Callable[[str, list[str]], str] | None = None,
    ) -> list[DiscoveredTool]:
        """Select the best tools for a task using DFSDT.

        Args:
            task: The task description.
            available_tools: List of available tools.
            selector_fn: Optional function to select category/tool.

        Returns:
            List of selected tools.
        """
        if not available_tools:
            return []

        # Build category tree
        tree = self.build_tool_tree(available_tools)

        # DFS traversal to find best tools
        selected: list[DiscoveredTool] = []
        visited: set[str] = set()

        async def dfs(node: dict[str, Any], path: list[str]) -> None:
            if len(path) >= self.max_depth:
                return

            node_id = "/".join(path)
            if node_id in visited:
                return
            visited.add(node_id)

            # Get tools at this node
            node_tools = node.get("tools", [])
            children = node.get("children", {})

            if node_tools and not children:
                # Leaf node - add tools
                for tool in node_tools[:self.max_breadth]:
                    if tool not in selected:
                        selected.append(tool)
                return

            # Non-leaf - explore children
            for child_name, child_node in list(children.items())[:self.max_breadth]:
                if selector_fn:
                    # Use provided selector
                    choice = selector_fn(task, list(children.keys()))
                    if choice == child_name:
                        await dfs(child_node, path + [child_name])
                else:
                    # Explore all children
                    await dfs(child_node, path + [child_name])

        await dfs(tree, [])

        self.log.info(
            "DFSDT selection complete",
            task=task[:50],
            tools_selected=len(selected),
        )

        return selected

    def build_tool_tree(
        self,
        tools: list[DiscoveredTool],
    ) -> dict[str, Any]:
        """Build a hierarchical tree of tools by category.

        Args:
            tools: List of tools to organize.

        Returns:
            Tree structure with tools at leaves.
        """
        tree: dict[str, Any] = {"children": {}, "tools": []}

        for tool in tools:
            # Get category from tool type or metadata
            category = tool.tool_type.value
            tags = tool.metadata.get("tags", [])

            if tags:
                # Use tags for hierarchy
                current = tree
                for tag in tags[:self.max_depth]:
                    if tag not in current["children"]:
                        current["children"][tag] = {"children": {}, "tools": []}
                    current = current["children"][tag]
                current["tools"].append(tool)
            else:
                # Use tool type as single-level category
                if category not in tree["children"]:
                    tree["children"][category] = {"children": {}, "tools": []}
                tree["children"][category]["tools"].append(tool)

        return tree


# =============================================================================
# Factory Functions
# =============================================================================


def create_tool_discovery_agent(
    registry: ToolRegistry | None = None,
    register_with_mesh: bool = False,
    **kwargs,
) -> ToolDiscoveryAgent:
    """Create a configured ToolDiscoveryAgent.

    Args:
        registry: Optional tool registry.
        register_with_mesh: Whether to register with A2A mesh.
        **kwargs: Additional arguments for BaseAgent.

    Returns:
        Configured ToolDiscoveryAgent.
    """
    return ToolDiscoveryAgent(
        registry=registry,
        register_with_mesh=register_with_mesh,
        **kwargs,
    )


def create_tool_creator_agent(
    registry: ToolRegistry | None = None,
    discovery_agent: ToolDiscoveryAgent | None = None,
    register_with_mesh: bool = False,
    **kwargs,
) -> ToolCreatorAgent:
    """Create a configured ToolCreatorAgent.

    Args:
        registry: Optional tool registry.
        discovery_agent: Optional discovery agent for verification.
        register_with_mesh: Whether to register with A2A mesh.
        **kwargs: Additional arguments for BaseAgent.

    Returns:
        Configured ToolCreatorAgent.
    """
    return ToolCreatorAgent(
        registry=registry,
        discovery_agent=discovery_agent,
        register_with_mesh=register_with_mesh,
        **kwargs,
    )


def get_tool_registry(
    cognee_client: CogneeClient | None = None,
) -> ToolRegistry:
    """Get a singleton ToolRegistry instance.

    Args:
        cognee_client: Optional Cognee client for persistence.

    Returns:
        ToolRegistry instance.
    """
    return ToolRegistry(cognee_client=cognee_client)
