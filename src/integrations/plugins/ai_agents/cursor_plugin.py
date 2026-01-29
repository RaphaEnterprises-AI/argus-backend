"""Cursor AI integration plugin.

This plugin enables Cursor to access Argus test data and insights through
the Model Context Protocol (MCP). It provides:

- Test results as MCP resources
- Test suggestions and failure patterns as tools
- Context sharing for AI-assisted debugging

Cursor connects to Argus MCP server which exposes tools and resources
that the AI can use during coding sessions.
"""

import asyncio
import json
import logging
import socket
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


@dataclass
class MCPServerConfig:
    """Configuration for the MCP server connection."""

    host: str = "127.0.0.1"
    port: int = 3000
    protocol: str = "stdio"  # stdio, sse, or websocket
    server_path: Optional[str] = None  # Path to MCP server executable
    timeout: float = 30.0


@dataclass
class TestResult:
    """A test result exposed as an MCP resource."""

    test_id: str
    test_name: str
    status: str  # passed, failed, skipped
    duration_ms: int
    error_message: Optional[str] = None
    screenshot_url: Optional[str] = None
    executed_at: Optional[datetime] = None
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FailurePattern:
    """A failure pattern for self-healing context."""

    pattern_id: str
    error_type: str
    original_selector: str
    healed_selector: Optional[str] = None
    confidence: float = 0.0
    occurrence_count: int = 0
    last_seen_at: Optional[datetime] = None


class CursorPlugin(IntegrationPlugin):
    """Cursor AI integration via Model Context Protocol.

    This plugin enables Cursor to access Argus test data through MCP.
    It connects to the Argus MCP server (either local or remote) to:

    1. Expose test results as resources (read-only context)
    2. Provide test suggestions as tools (actionable insights)
    3. Share failure patterns for debugging assistance

    Cursor's AI can then use this context to provide better code suggestions
    and help debug test failures.

    Example:
        async with CursorPlugin() as plugin:
            # Test MCP server connection
            result = await plugin.test_connection(IntegrationCredentials())
            if result.success:
                # Sync test data to expose via MCP
                sync_result = await plugin.sync_data(IntegrationCredentials())
    """

    def __init__(
        self,
        http_client: Optional[httpx.AsyncClient] = None,
        mcp_config: Optional[MCPServerConfig] = None,
    ):
        """Initialize the Cursor plugin.

        Args:
            http_client: Optional HTTP client for API requests.
            mcp_config: Optional MCP server configuration.
        """
        super().__init__(http_client)
        self._mcp_config = mcp_config or MCPServerConfig()
        self._cached_test_results: list[TestResult] = []
        self._cached_failure_patterns: list[FailurePattern] = []

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Cursor plugin metadata."""
        return IntegrationMetadata(
            slug="cursor",
            name="Cursor",
            description="AI-powered code editor with MCP support. Provides test context and insights for AI-assisted development.",
            category=IntegrationCategory.AI_AGENTS,
            auth_type=AuthType.MCP,
            icon_url="https://cursor.sh/favicon.ico",
            docs_url="https://docs.cursor.com/context/model-context-protocol",
            website_url="https://cursor.sh",
            features=[
                "test_results_as_resources",
                "test_suggestions_as_tools",
                "failure_pattern_sharing",
                "ai_debugging_context",
                "real_time_sync",
            ],
            required_fields=[],  # No credentials required for local MCP
            optional_fields=[
                FieldDefinition(
                    name="mcp_server_url",
                    label="MCP Server URL",
                    type="url",
                    required=False,
                    placeholder="http://localhost:3000",
                    help_text="URL of the Argus MCP server (default: local stdio)",
                ),
                FieldDefinition(
                    name="project_id",
                    label="Project ID",
                    type="text",
                    required=False,
                    placeholder="proj_xxx",
                    help_text="Argus project ID to sync test data from",
                ),
            ],
            supports_webhooks=False,
            supports_sync=True,
            sync_resources=[
                "test_results",
                "failure_patterns",
                "test_suggestions",
                "coverage_gaps",
            ],
        )

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the MCP server connection.

        For Cursor, this checks if the MCP server is available and responding.
        The MCP server can be:
        - Local stdio process (argus-mcp-server)
        - Remote HTTP/SSE endpoint
        - WebSocket connection

        Args:
            credentials: Integration credentials (may contain mcp_server_url).

        Returns:
            ConnectionTestResult with server status.
        """
        start_time = datetime.utcnow()

        try:
            # Get MCP server URL from credentials or use default
            mcp_url = credentials.extra.get(
                "mcp_server_url",
                f"http://{self._mcp_config.host}:{self._mcp_config.port}",
            )

            # Check if this is a local stdio server
            if self._is_local_stdio():
                return await self._test_stdio_connection()

            # For HTTP-based MCP servers, check health endpoint
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{mcp_url}/health")

                if response.status_code == 200:
                    data = response.json()
                    latency = (datetime.utcnow() - start_time).total_seconds() * 1000

                    return ConnectionTestResult(
                        success=True,
                        message="MCP server is available and responding",
                        latency_ms=latency,
                        api_version=data.get("version", "unknown"),
                        permissions=["read_resources", "call_tools"],
                        account_info={
                            "server_type": "mcp",
                            "protocol": self._mcp_config.protocol,
                            "status": data.get("status", "healthy"),
                        },
                    )
                else:
                    return ConnectionTestResult(
                        success=False,
                        message=f"MCP server returned status {response.status_code}",
                        error_code="MCP_SERVER_ERROR",
                        error_details={"status_code": response.status_code},
                    )

        except httpx.ConnectError:
            return ConnectionTestResult(
                success=False,
                message="Cannot connect to MCP server. Ensure Cursor has MCP enabled.",
                error_code="MCP_CONNECTION_FAILED",
                error_details={
                    "hint": "Run 'cursor --mcp-server argus' or configure MCP in Cursor settings"
                },
            )
        except Exception as e:
            logger.exception("Error testing Cursor MCP connection")
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
        """Sync Argus data to expose via MCP for Cursor.

        This method fetches test data from Argus and prepares it for
        MCP resource exposure. The data is cached locally and served
        to Cursor when it queries the MCP server.

        Args:
            credentials: Integration credentials with project_id.
            resources: Optional list of resources to sync.
                      Options: test_results, failure_patterns, test_suggestions
            since: Optional datetime for incremental sync.
            limit: Maximum items to sync per resource type.

        Returns:
            SyncResult with synced data statistics.
        """
        started_at = datetime.utcnow()
        sync_id = f"cursor_sync_{started_at.strftime('%Y%m%d_%H%M%S')}"

        resources_to_sync = resources or [
            "test_results",
            "failure_patterns",
            "test_suggestions",
        ]

        synced_data: dict[str, Any] = {}
        total_synced = 0
        errors: list[str] = []
        warnings: list[str] = []

        project_id = credentials.extra.get("project_id")
        if not project_id:
            warnings.append("No project_id specified - using demo data")

        try:
            # Sync test results
            if "test_results" in resources_to_sync:
                test_results = await self._fetch_test_results(
                    project_id, since, limit
                )
                self._cached_test_results = test_results
                synced_data["test_results"] = [
                    self._test_result_to_mcp_resource(tr) for tr in test_results
                ]
                total_synced += len(test_results)
                logger.info(f"Synced {len(test_results)} test results for Cursor")

            # Sync failure patterns
            if "failure_patterns" in resources_to_sync:
                patterns = await self._fetch_failure_patterns(project_id, limit)
                self._cached_failure_patterns = patterns
                synced_data["failure_patterns"] = [
                    self._pattern_to_mcp_resource(p) for p in patterns
                ]
                total_synced += len(patterns)
                logger.info(f"Synced {len(patterns)} failure patterns for Cursor")

            # Sync test suggestions
            if "test_suggestions" in resources_to_sync:
                suggestions = await self._generate_test_suggestions(project_id)
                synced_data["test_suggestions"] = suggestions
                total_synced += len(suggestions)
                logger.info(f"Generated {len(suggestions)} test suggestions for Cursor")

            # Generate MCP manifest
            synced_data["mcp_manifest"] = self._generate_mcp_manifest()

            return SyncResult(
                success=True,
                message=f"Successfully synced {total_synced} items for Cursor MCP",
                data_points_synced=total_synced,
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=synced_data,
            )

        except Exception as e:
            logger.exception("Error syncing data for Cursor")
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                data=synced_data,
            )

    def _is_local_stdio(self) -> bool:
        """Check if using local stdio-based MCP server."""
        return self._mcp_config.protocol == "stdio"

    async def _test_stdio_connection(self) -> ConnectionTestResult:
        """Test stdio-based MCP server connection.

        For stdio MCP servers, we check if the server process can be started
        and responds to the initialize request.
        """
        try:
            # Check if argus-mcp-server is available
            server_path = self._mcp_config.server_path or self._find_mcp_server()

            if not server_path:
                return ConnectionTestResult(
                    success=False,
                    message="Argus MCP server not found. Install via 'npx @anthropic/claude-code add argus'",
                    error_code="MCP_SERVER_NOT_FOUND",
                    error_details={
                        "hint": "Configure MCP server in Cursor settings"
                    },
                )

            # Verify the server file exists
            if not Path(server_path).exists():
                return ConnectionTestResult(
                    success=False,
                    message=f"MCP server not found at {server_path}",
                    error_code="MCP_SERVER_NOT_FOUND",
                )

            # For stdio, we just verify the server can be accessed
            # Actual protocol communication happens via Cursor
            return ConnectionTestResult(
                success=True,
                message="MCP server is available (stdio mode)",
                permissions=["read_resources", "call_tools"],
                account_info={
                    "server_type": "mcp",
                    "protocol": "stdio",
                    "server_path": server_path,
                },
            )

        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Failed to test stdio connection: {str(e)}",
                error_code="MCP_STDIO_ERROR",
            )

    def _find_mcp_server(self) -> Optional[str]:
        """Find the Argus MCP server executable.

        Looks in common locations for the MCP server.
        """
        possible_paths = [
            # Local project
            "./argus-mcp-server/dist/index.js",
            # Global npm install
            "/usr/local/bin/argus-mcp",
            # npx location
            "~/.npm/_npx/argus-mcp-server/index.js",
        ]

        for path in possible_paths:
            expanded = Path(path).expanduser()
            if expanded.exists():
                return str(expanded)

        return None

    async def _fetch_test_results(
        self,
        project_id: Optional[str],
        since: Optional[datetime],
        limit: int,
    ) -> list[TestResult]:
        """Fetch test results from Argus API.

        Args:
            project_id: Argus project ID.
            since: Fetch results since this datetime.
            limit: Maximum results to fetch.

        Returns:
            List of TestResult objects.
        """
        if not project_id:
            # Return demo data for testing
            return [
                TestResult(
                    test_id="demo_test_1",
                    test_name="Login Flow Test",
                    status="failed",
                    duration_ms=2500,
                    error_message="Element not found: #submit-button",
                    executed_at=datetime.utcnow(),
                    steps=[
                        {"action": "navigate", "target": "/login", "status": "passed"},
                        {"action": "type", "target": "#email", "value": "test@example.com", "status": "passed"},
                        {"action": "click", "target": "#submit-button", "status": "failed"},
                    ],
                ),
            ]

        # In production, fetch from Argus API
        # TODO: Implement actual API call
        return []

    async def _fetch_failure_patterns(
        self,
        project_id: Optional[str],
        limit: int,
    ) -> list[FailurePattern]:
        """Fetch failure patterns for self-healing context.

        Args:
            project_id: Argus project ID.
            limit: Maximum patterns to fetch.

        Returns:
            List of FailurePattern objects.
        """
        if not project_id:
            # Return demo data
            return [
                FailurePattern(
                    pattern_id="pattern_1",
                    error_type="element_not_found",
                    original_selector="#submit-button",
                    healed_selector="button[type='submit']",
                    confidence=0.95,
                    occurrence_count=5,
                    last_seen_at=datetime.utcnow(),
                ),
            ]

        # TODO: Implement actual API call
        return []

    async def _generate_test_suggestions(
        self,
        project_id: Optional[str],
    ) -> list[dict[str, Any]]:
        """Generate test suggestions based on coverage gaps.

        Args:
            project_id: Argus project ID.

        Returns:
            List of test suggestion dictionaries.
        """
        if not project_id:
            return [
                {
                    "type": "coverage_gap",
                    "priority": "high",
                    "description": "Add test for user registration flow",
                    "suggested_steps": [
                        "Navigate to /register",
                        "Fill registration form",
                        "Submit and verify success",
                    ],
                    "reason": "Registration endpoint has high traffic but no test coverage",
                },
            ]

        # TODO: Implement actual suggestions based on Argus data
        return []

    def _test_result_to_mcp_resource(self, result: TestResult) -> dict[str, Any]:
        """Convert TestResult to MCP resource format.

        MCP resources are read-only data that Cursor can use as context.

        Args:
            result: The TestResult to convert.

        Returns:
            MCP resource dictionary.
        """
        return {
            "uri": f"argus://tests/{result.test_id}",
            "mimeType": "application/json",
            "name": result.test_name,
            "description": f"Test result: {result.status}",
            "contents": {
                "test_id": result.test_id,
                "name": result.test_name,
                "status": result.status,
                "duration_ms": result.duration_ms,
                "error": result.error_message,
                "screenshot_url": result.screenshot_url,
                "executed_at": result.executed_at.isoformat() if result.executed_at else None,
                "steps": result.steps,
            },
        }

    def _pattern_to_mcp_resource(self, pattern: FailurePattern) -> dict[str, Any]:
        """Convert FailurePattern to MCP resource format.

        Args:
            pattern: The FailurePattern to convert.

        Returns:
            MCP resource dictionary.
        """
        return {
            "uri": f"argus://patterns/{pattern.pattern_id}",
            "mimeType": "application/json",
            "name": f"Pattern: {pattern.error_type}",
            "description": f"Failure pattern with {pattern.confidence:.0%} confidence fix",
            "contents": {
                "pattern_id": pattern.pattern_id,
                "error_type": pattern.error_type,
                "original_selector": pattern.original_selector,
                "healed_selector": pattern.healed_selector,
                "confidence": pattern.confidence,
                "occurrence_count": pattern.occurrence_count,
                "last_seen_at": pattern.last_seen_at.isoformat() if pattern.last_seen_at else None,
            },
        }

    def _generate_mcp_manifest(self) -> dict[str, Any]:
        """Generate MCP server manifest for Cursor configuration.

        This manifest describes the tools and resources available from Argus.

        Returns:
            MCP manifest dictionary.
        """
        return {
            "name": "argus",
            "version": "1.0.0",
            "description": "Argus E2E Testing - AI-powered test context and insights",
            "tools": [
                {
                    "name": "argus_test_suggestions",
                    "description": "Get AI-generated test suggestions based on coverage gaps and production errors",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of suggestions",
                                "default": 5,
                            },
                        },
                    },
                },
                {
                    "name": "argus_debug_failure",
                    "description": "Get debugging context for a specific test failure",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "test_id": {
                                "type": "string",
                                "description": "ID of the failed test",
                            },
                            "error_message": {
                                "type": "string",
                                "description": "The error message to analyze",
                            },
                        },
                        "required": ["error_message"],
                    },
                },
                {
                    "name": "argus_healing_suggestion",
                    "description": "Get selector healing suggestions for element not found errors",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "The broken selector",
                            },
                            "page_url": {
                                "type": "string",
                                "description": "URL of the page where the selector failed",
                            },
                        },
                        "required": ["selector"],
                    },
                },
            ],
            "resources": [
                {
                    "uri": "argus://tests/*",
                    "name": "Test Results",
                    "description": "Recent test execution results",
                    "mimeType": "application/json",
                },
                {
                    "uri": "argus://patterns/*",
                    "name": "Failure Patterns",
                    "description": "Known failure patterns and their fixes",
                    "mimeType": "application/json",
                },
            ],
        }

    def get_cursor_config(self) -> dict[str, Any]:
        """Generate Cursor MCP configuration.

        This returns the configuration that users should add to their
        Cursor settings to enable Argus MCP integration.

        Returns:
            Cursor MCP configuration dictionary.
        """
        return {
            "mcpServers": {
                "argus": {
                    "command": "npx",
                    "args": ["-y", "@anthropic/argus-mcp-server"],
                    "env": {
                        "ARGUS_API_URL": "https://argus-brain-production.up.railway.app",
                    },
                },
            },
        }
