"""Claude Code integration plugin.

This plugin enables Claude Code CLI to access Argus test data and insights
through the Model Context Protocol (MCP). It provides:

- Test results as MCP resources for context
- Test generation tools for creating new tests
- Failure debugging assistance
- Self-healing pattern suggestions

Claude Code connects to Argus MCP server which exposes tools and resources
that Claude can use during coding sessions via the CLI.
"""

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from src.integrations.base import (
    AuthType,
    ConnectionTestResult,
    FieldDefinition,
    IntegrationCategory,
    IntegrationCredentials,
    IntegrationMetadata,
    IntegrationPlugin,
    SyncResult,
)

logger = logging.getLogger(__name__)


# Default Argus MCP server URL (Cloudflare Workers)
DEFAULT_MCP_SERVER_URL = "https://argus-mcp.samuelvinay-kumar.workers.dev"

# Local development MCP server
LOCAL_MCP_SERVER_URL = "http://localhost:8787"


@dataclass
class MCPToolCall:
    """Represents a tool call made through MCP."""

    tool_name: str
    arguments: dict[str, Any]
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass
class MCPResource:
    """Represents an MCP resource."""

    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    contents: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestContext:
    """Test context for Claude Code assistance."""

    test_id: str
    test_name: str
    test_code: Optional[str] = None
    last_result: Optional[str] = None  # passed, failed, skipped
    error_message: Optional[str] = None
    screenshot_url: Optional[str] = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    related_files: list[str] = field(default_factory=list)


class ClaudeCodePlugin(IntegrationPlugin):
    """Claude Code CLI integration via Model Context Protocol.

    This plugin enables Claude Code to access Argus test data through MCP.
    It connects to the Argus MCP server (Cloudflare Workers) to:

    1. Expose test results as resources (read-only context)
    2. Provide test generation tools
    3. Share failure patterns for debugging
    4. Enable self-healing suggestions

    Claude Code can then use this context to:
    - Generate new tests based on production errors
    - Debug failing tests with context
    - Suggest selector fixes for broken tests
    - Understand test coverage gaps

    Example:
        async with ClaudeCodePlugin() as plugin:
            # Test MCP server connection
            result = await plugin.test_connection(IntegrationCredentials())

            # Sync test data to expose via MCP
            sync_result = await plugin.sync_data(
                IntegrationCredentials(),
                resources=["test_results", "failure_patterns"]
            )

    Configuration for Claude Code (~/.config/claude-code/config.json):
        {
            "mcpServers": {
                "argus": {
                    "url": "https://argus-mcp.samuelvinay-kumar.workers.dev",
                    "transport": "sse"
                }
            }
        }
    """

    def __init__(
        self,
        http_client: Optional[httpx.AsyncClient] = None,
        mcp_server_url: Optional[str] = None,
    ):
        """Initialize the Claude Code plugin.

        Args:
            http_client: Optional HTTP client for API requests.
            mcp_server_url: Optional MCP server URL override.
        """
        super().__init__(http_client)
        self._mcp_server_url = mcp_server_url or os.environ.get(
            "ARGUS_MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL
        )
        self._cached_contexts: dict[str, TestContext] = {}
        self._tool_history: list[MCPToolCall] = []

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Claude Code plugin metadata."""
        return IntegrationMetadata(
            slug="claude-code",
            name="Claude Code",
            description="Anthropic's Claude Code CLI with MCP support. Provides test context, generation, and debugging assistance.",
            category=IntegrationCategory.AI_AGENTS,
            auth_type=AuthType.MCP,
            icon_url="https://www.anthropic.com/favicon.ico",
            docs_url="https://docs.anthropic.com/claude-code/mcp",
            website_url="https://claude.ai/claude-code",
            features=[
                "test_results_as_resources",
                "test_generation_tools",
                "failure_debugging",
                "self_healing_suggestions",
                "coverage_gap_analysis",
                "production_error_context",
            ],
            required_fields=[],  # No credentials required for local MCP
            optional_fields=[
                FieldDefinition(
                    name="mcp_server_url",
                    label="MCP Server URL",
                    type="url",
                    required=False,
                    placeholder=DEFAULT_MCP_SERVER_URL,
                    help_text="URL of the Argus MCP server (uses cloud server by default)",
                ),
                FieldDefinition(
                    name="project_id",
                    label="Project ID",
                    type="text",
                    required=False,
                    placeholder="proj_xxx",
                    help_text="Argus project ID to sync test data from",
                ),
                FieldDefinition(
                    name="api_key",
                    label="Argus API Key",
                    type="password",
                    required=False,
                    placeholder="sk_argus_xxx",
                    help_text="Optional API key for authenticated access",
                ),
            ],
            supports_webhooks=False,
            supports_sync=True,
            sync_resources=[
                "test_results",
                "failure_patterns",
                "test_suggestions",
                "coverage_gaps",
                "production_errors",
            ],
        )

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the MCP server connection.

        For Claude Code, this checks if the Argus MCP server is available.
        The server provides tools and resources for AI-assisted testing.

        Args:
            credentials: Integration credentials (may contain mcp_server_url).

        Returns:
            ConnectionTestResult with server status.
        """
        start_time = datetime.utcnow()

        try:
            # Get MCP server URL from credentials or use default
            mcp_url = credentials.extra.get("mcp_server_url") or self._mcp_server_url

            # Check MCP server health
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Try health endpoint first
                try:
                    response = await client.get(f"{mcp_url}/health")
                    if response.status_code == 200:
                        data = response.json()
                        latency = (datetime.utcnow() - start_time).total_seconds() * 1000

                        return ConnectionTestResult(
                            success=True,
                            message="Argus MCP server is available and responding",
                            latency_ms=latency,
                            api_version=data.get("version", "1.0.0"),
                            permissions=self._get_available_permissions(credentials),
                            account_info={
                                "server_url": mcp_url,
                                "transport": "sse",
                                "status": data.get("status", "healthy"),
                                "tools_available": self._get_available_tools(),
                            },
                        )
                except httpx.HTTPStatusError:
                    pass

                # Try SSE endpoint (MCP transport)
                response = await client.get(
                    f"{mcp_url}/sse",
                    headers={"Accept": "text/event-stream"},
                )

                if response.status_code in (200, 204):
                    latency = (datetime.utcnow() - start_time).total_seconds() * 1000

                    return ConnectionTestResult(
                        success=True,
                        message="Argus MCP server is available (SSE transport)",
                        latency_ms=latency,
                        permissions=self._get_available_permissions(credentials),
                        account_info={
                            "server_url": mcp_url,
                            "transport": "sse",
                            "tools_available": self._get_available_tools(),
                        },
                    )
                else:
                    return ConnectionTestResult(
                        success=False,
                        message=f"MCP server returned status {response.status_code}",
                        error_code="MCP_SERVER_ERROR",
                        error_details={
                            "status_code": response.status_code,
                            "url": mcp_url,
                        },
                    )

        except httpx.ConnectError:
            return ConnectionTestResult(
                success=False,
                message="Cannot connect to Argus MCP server. Check your network connection.",
                error_code="MCP_CONNECTION_FAILED",
                error_details={
                    "url": self._mcp_server_url,
                    "hint": "Ensure the MCP server URL is correct in your Claude Code config",
                },
            )
        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection to MCP server timed out",
                error_code="MCP_TIMEOUT",
                error_details={"timeout_seconds": 15},
            )
        except Exception as e:
            logger.exception("Error testing Claude Code MCP connection")
            return ConnectionTestResult(
                success=False,
                message=f"Connection test failed: {str(e)}",
                error_code="MCP_TEST_ERROR",
                error_details={"error_type": type(e).__name__},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync Argus data to expose via MCP for Claude Code.

        This method fetches test data from Argus and prepares it for
        MCP resource exposure. The data includes:

        - Test results (recent executions)
        - Failure patterns (for self-healing)
        - Test suggestions (coverage gaps)
        - Production errors (for test generation)

        Args:
            credentials: Integration credentials with project_id and optional api_key.
            resources: Optional list of resources to sync.
            since: Optional datetime for incremental sync.
            limit: Maximum items to sync per resource type.

        Returns:
            SyncResult with synced data statistics.
        """
        started_at = datetime.utcnow()
        sync_id = f"claude_code_sync_{started_at.strftime('%Y%m%d_%H%M%S')}"

        resources_to_sync = resources or [
            "test_results",
            "failure_patterns",
            "test_suggestions",
            "production_errors",
        ]

        synced_data: dict[str, Any] = {}
        total_synced = 0
        errors: list[str] = []
        warnings: list[str] = []

        project_id = credentials.extra.get("project_id")
        api_key = credentials.api_key

        if not project_id:
            warnings.append("No project_id specified - some features may be limited")

        try:
            # Sync test results
            if "test_results" in resources_to_sync:
                results, count = await self._sync_test_results(
                    project_id, api_key, since, limit
                )
                synced_data["test_results"] = results
                total_synced += count
                logger.info(f"Synced {count} test results for Claude Code")

            # Sync failure patterns
            if "failure_patterns" in resources_to_sync:
                patterns, count = await self._sync_failure_patterns(
                    project_id, api_key, limit
                )
                synced_data["failure_patterns"] = patterns
                total_synced += count
                logger.info(f"Synced {count} failure patterns for Claude Code")

            # Sync test suggestions
            if "test_suggestions" in resources_to_sync:
                suggestions, count = await self._sync_test_suggestions(
                    project_id, api_key, limit
                )
                synced_data["test_suggestions"] = suggestions
                total_synced += count
                logger.info(f"Synced {count} test suggestions for Claude Code")

            # Sync production errors
            if "production_errors" in resources_to_sync:
                events, count = await self._sync_production_errors(
                    project_id, api_key, since, limit
                )
                synced_data["production_errors"] = events
                total_synced += count
                logger.info(f"Synced {count} production errors for Claude Code")

            # Generate MCP configuration
            synced_data["mcp_config"] = self.get_claude_code_config()

            # Generate tool manifest
            synced_data["tools_manifest"] = self._generate_tools_manifest()

            return SyncResult(
                success=True,
                message=f"Successfully synced {total_synced} items for Claude Code MCP",
                data_points_synced=total_synced,
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=synced_data,
            )

        except Exception as e:
            logger.exception("Error syncing data for Claude Code")
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                data=synced_data,
            )

    async def _sync_test_results(
        self,
        project_id: Optional[str],
        api_key: Optional[str],
        since: Optional[datetime],
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Sync test results from Argus API.

        Args:
            project_id: Argus project ID.
            api_key: Optional API key for authentication.
            since: Fetch results since this datetime.
            limit: Maximum results to fetch.

        Returns:
            Tuple of (list of MCP resources, count).
        """
        if not project_id:
            # Return demo data for testing/preview
            demo_results = [
                {
                    "uri": "argus://tests/demo_1",
                    "name": "User Authentication Flow",
                    "description": "Test result: failed (2.5s)",
                    "mimeType": "application/json",
                    "contents": {
                        "test_id": "demo_1",
                        "name": "User Authentication Flow",
                        "status": "failed",
                        "duration_ms": 2500,
                        "error": "Element not found: button[data-testid='login-submit']",
                        "steps": [
                            {"action": "navigate", "target": "/login", "status": "passed"},
                            {"action": "fill", "target": "#email", "value": "test@example.com", "status": "passed"},
                            {"action": "fill", "target": "#password", "value": "***", "status": "passed"},
                            {"action": "click", "target": "button[data-testid='login-submit']", "status": "failed"},
                        ],
                        "screenshot_url": None,
                        "executed_at": datetime.utcnow().isoformat(),
                    },
                },
                {
                    "uri": "argus://tests/demo_2",
                    "name": "Product Search",
                    "description": "Test result: passed (1.2s)",
                    "mimeType": "application/json",
                    "contents": {
                        "test_id": "demo_2",
                        "name": "Product Search",
                        "status": "passed",
                        "duration_ms": 1200,
                        "steps": [
                            {"action": "navigate", "target": "/products", "status": "passed"},
                            {"action": "fill", "target": "#search", "value": "laptop", "status": "passed"},
                            {"action": "click", "target": "button[type='submit']", "status": "passed"},
                            {"action": "assert", "target": ".results", "expected": "contains: laptop", "status": "passed"},
                        ],
                        "executed_at": datetime.utcnow().isoformat(),
                    },
                },
            ]
            return demo_results, len(demo_results)

        # Production: Fetch from Argus Brain API
        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["X-API-Key"] = api_key

            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {"project_id": project_id, "limit": limit}
                if since:
                    params["since"] = since.isoformat()

                response = await client.get(
                    f"{os.environ.get('ARGUS_BRAIN_URL', 'https://argus-brain-production.up.railway.app')}/api/v1/tests/runs",
                    params=params,
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    resources = []
                    for run in data.get("runs", []):
                        resources.append({
                            "uri": f"argus://tests/{run['id']}",
                            "name": run.get("test_name", run["id"]),
                            "description": f"Test result: {run['status']} ({run.get('duration_ms', 0)}ms)",
                            "mimeType": "application/json",
                            "contents": run,
                        })
                    return resources, len(resources)

        except Exception as e:
            logger.warning(f"Failed to fetch test results: {e}")

        return [], 0

    async def _sync_failure_patterns(
        self,
        project_id: Optional[str],
        api_key: Optional[str],
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Sync failure patterns for self-healing context.

        Args:
            project_id: Argus project ID.
            api_key: Optional API key.
            limit: Maximum patterns to fetch.

        Returns:
            Tuple of (list of MCP resources, count).
        """
        if not project_id:
            # Demo patterns
            demo_patterns = [
                {
                    "uri": "argus://patterns/p1",
                    "name": "Login Button Selector Change",
                    "description": "High confidence fix (95%)",
                    "mimeType": "application/json",
                    "contents": {
                        "pattern_id": "p1",
                        "error_type": "element_not_found",
                        "original_selector": "button[data-testid='login-submit']",
                        "healed_selector": "button[type='submit'][class*='login']",
                        "confidence": 0.95,
                        "occurrences": 12,
                        "last_seen": datetime.utcnow().isoformat(),
                        "context": {
                            "page_url": "/login",
                            "component": "LoginForm",
                        },
                    },
                },
            ]
            return demo_patterns, len(demo_patterns)

        # Production: Fetch from API
        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["X-API-Key"] = api_key

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{os.environ.get('ARGUS_BRAIN_URL', 'https://argus-brain-production.up.railway.app')}/api/v1/healing/patterns",
                    params={"project_id": project_id, "limit": limit},
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    resources = []
                    for pattern in data.get("patterns", []):
                        resources.append({
                            "uri": f"argus://patterns/{pattern['id']}",
                            "name": f"Pattern: {pattern['error_type']}",
                            "description": f"Confidence: {pattern.get('confidence', 0):.0%}",
                            "mimeType": "application/json",
                            "contents": pattern,
                        })
                    return resources, len(resources)

        except Exception as e:
            logger.warning(f"Failed to fetch failure patterns: {e}")

        return [], 0

    async def _sync_test_suggestions(
        self,
        project_id: Optional[str],
        api_key: Optional[str],
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Sync test suggestions based on coverage gaps.

        Args:
            project_id: Argus project ID.
            api_key: Optional API key.
            limit: Maximum suggestions to generate.

        Returns:
            Tuple of (list of suggestions, count).
        """
        if not project_id:
            # Demo suggestions
            demo_suggestions = [
                {
                    "id": "sug_1",
                    "type": "coverage_gap",
                    "priority": "high",
                    "title": "Add test for password reset flow",
                    "description": "The /forgot-password endpoint has production traffic but no test coverage",
                    "suggested_steps": [
                        "Navigate to /forgot-password",
                        "Enter email address",
                        "Click submit",
                        "Verify success message",
                    ],
                    "risk_score": 0.85,
                    "affected_users": 150,
                },
                {
                    "id": "sug_2",
                    "type": "production_error",
                    "priority": "critical",
                    "title": "Add test for checkout payment failure",
                    "description": "Payment failures occurring in production with no test coverage",
                    "error_sample": "PaymentError: Card declined",
                    "suggested_steps": [
                        "Navigate to checkout",
                        "Enter invalid card",
                        "Verify error handling",
                    ],
                    "risk_score": 0.95,
                    "affected_users": 45,
                },
            ]
            return demo_suggestions, len(demo_suggestions)

        # Production: Fetch from API
        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["X-API-Key"] = api_key

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{os.environ.get('ARGUS_BRAIN_URL', 'https://argus-brain-production.up.railway.app')}/api/v1/intelligence/what-to-test",
                    params={"project_id": project_id, "limit": limit},
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("recommendations", []), len(data.get("recommendations", []))

        except Exception as e:
            logger.warning(f"Failed to fetch test suggestions: {e}")

        return [], 0

    async def _sync_production_errors(
        self,
        project_id: Optional[str],
        api_key: Optional[str],
        since: Optional[datetime],
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Sync production errors for test generation context.

        Args:
            project_id: Argus project ID.
            api_key: Optional API key.
            since: Fetch errors since this datetime.
            limit: Maximum errors to fetch.

        Returns:
            Tuple of (list of error events, count).
        """
        if not project_id:
            # Demo errors
            demo_errors = [
                {
                    "id": "err_1",
                    "title": "TypeError: Cannot read property 'map' of undefined",
                    "severity": "error",
                    "url": "/dashboard",
                    "component": "UserList",
                    "occurrences": 45,
                    "affected_users": 23,
                    "first_seen": datetime.utcnow().isoformat(),
                    "stack_trace": "at UserList.render (UserList.tsx:42)",
                    "context": {
                        "user_agent": "Chrome/120",
                        "viewport": "1920x1080",
                    },
                },
            ]
            return demo_errors, len(demo_errors)

        # Production: Fetch from API
        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["X-API-Key"] = api_key

            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {"project_id": project_id, "limit": limit}
                if since:
                    params["since"] = since.isoformat()

                response = await client.get(
                    f"{os.environ.get('ARGUS_BRAIN_URL', 'https://argus-brain-production.up.railway.app')}/api/v1/events",
                    params=params,
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("events", []), len(data.get("events", []))

        except Exception as e:
            logger.warning(f"Failed to fetch production errors: {e}")

        return [], 0

    def _get_available_permissions(
        self,
        credentials: IntegrationCredentials,
    ) -> list[str]:
        """Get available permissions based on credentials.

        Args:
            credentials: Integration credentials.

        Returns:
            List of permission strings.
        """
        # Base permissions (always available)
        permissions = [
            "read_test_results",
            "read_failure_patterns",
            "call_tools",
        ]

        # Additional permissions with API key
        if credentials.api_key:
            permissions.extend([
                "write_tests",
                "approve_healing",
                "access_production_data",
            ])

        return permissions

    def _get_available_tools(self) -> list[str]:
        """Get list of available MCP tools.

        Returns:
            List of tool names.
        """
        return [
            "argus_discover",
            "argus_act",
            "argus_test",
            "argus_extract",
            "argus_agent",
            "argus_generate_test",
            "argus_health",
            "argus_events",
            "argus_what_to_test",
            "argus_healing_patterns",
            "argus_coverage_gaps",
        ]

    def _generate_tools_manifest(self) -> dict[str, Any]:
        """Generate MCP tools manifest for Claude Code.

        This describes all available tools that Claude Code can use.

        Returns:
            Tools manifest dictionary.
        """
        return {
            "tools": [
                {
                    "name": "argus_discover",
                    "description": "Discover interactive elements on a web page for testing",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "format": "uri",
                                "description": "URL of the page to analyze",
                            },
                        },
                        "required": ["url"],
                    },
                },
                {
                    "name": "argus_test",
                    "description": "Run a multi-step E2E test with screenshots",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "format": "uri",
                                "description": "Starting URL",
                            },
                            "steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Test step instructions",
                            },
                        },
                        "required": ["url", "steps"],
                    },
                },
                {
                    "name": "argus_generate_test",
                    "description": "Generate test steps from a natural language description",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "format": "uri",
                                "description": "URL of the page",
                            },
                            "description": {
                                "type": "string",
                                "description": "Natural language description of the test",
                            },
                        },
                        "required": ["url", "description"],
                    },
                },
                {
                    "name": "argus_what_to_test",
                    "description": "Get AI-powered recommendations on what to test based on production data",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "project_id": {
                                "type": "string",
                                "description": "Argus project ID",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum recommendations",
                                "default": 5,
                            },
                        },
                    },
                },
                {
                    "name": "argus_events",
                    "description": "Get production error events from Sentry/other integrations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "project_id": {
                                "type": "string",
                                "description": "Argus project ID",
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["error", "warning", "info"],
                                "description": "Filter by severity",
                            },
                            "limit": {
                                "type": "integer",
                                "default": 10,
                            },
                        },
                    },
                },
                {
                    "name": "argus_healing_patterns",
                    "description": "Get known failure patterns and their fixes",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "project_id": {
                                "type": "string",
                            },
                            "error_type": {
                                "type": "string",
                                "enum": ["element_not_found", "timeout", "assertion_failed"],
                            },
                        },
                    },
                },
                {
                    "name": "argus_coverage_gaps",
                    "description": "Identify test coverage gaps based on production usage",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "project_id": {
                                "type": "string",
                            },
                        },
                    },
                },
            ],
            "resources": [
                {
                    "uriTemplate": "argus://tests/{test_id}",
                    "name": "Test Results",
                    "description": "Recent test execution results with steps and screenshots",
                    "mimeType": "application/json",
                },
                {
                    "uriTemplate": "argus://patterns/{pattern_id}",
                    "name": "Failure Patterns",
                    "description": "Known failure patterns and self-healing suggestions",
                    "mimeType": "application/json",
                },
                {
                    "uriTemplate": "argus://errors/{error_id}",
                    "name": "Production Errors",
                    "description": "Production error events from monitoring",
                    "mimeType": "application/json",
                },
            ],
        }

    def get_claude_code_config(self) -> dict[str, Any]:
        """Generate Claude Code MCP configuration.

        This returns the configuration that users should add to their
        Claude Code config file (~/.config/claude-code/config.json).

        Returns:
            Claude Code MCP configuration dictionary.
        """
        return {
            "mcpServers": {
                "argus": {
                    "url": self._mcp_server_url,
                    "transport": "sse",
                    "description": "Argus E2E Testing - AI-powered test context and insights",
                    "capabilities": {
                        "tools": True,
                        "resources": True,
                        "prompts": False,
                    },
                },
            },
        }

    def get_installation_instructions(self) -> str:
        """Get installation instructions for Claude Code integration.

        Returns:
            Markdown-formatted installation instructions.
        """
        return f"""## Claude Code + Argus Integration

### Quick Setup

Add Argus to your Claude Code configuration:

1. Open your Claude Code config:
   ```bash
   claude config edit
   ```

2. Add the Argus MCP server:
   ```json
   {{
     "mcpServers": {{
       "argus": {{
         "url": "{self._mcp_server_url}",
         "transport": "sse"
       }}
     }}
   }}
   ```

3. Restart Claude Code

### Available Commands

Once configured, Claude Code can use these Argus tools:

- `argus_test` - Run E2E tests with natural language
- `argus_discover` - Discover page elements
- `argus_what_to_test` - Get test recommendations
- `argus_events` - View production errors
- `argus_healing_patterns` - Get self-healing suggestions

### Example Usage

Ask Claude:
> "Run a test that logs in to example.com and checks the dashboard"

Claude will use `argus_test` to execute the test and show results.

### Authentication (Optional)

For full access to your project data, authenticate:

1. Run `argus_auth` in Claude Code
2. Complete sign-in in your browser
3. Run `argus_auth_complete`

Your session will be saved for future use.
"""
