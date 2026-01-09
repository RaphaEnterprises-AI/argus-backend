"""Argus Quality Intelligence MCP Server.

This module provides an MCP server that exposes quality intelligence tools
for use by Claude Code and other MCP clients.

Tools provided:
- get_quality_score: Get overall quality score and metrics for a project
- get_recommendations: Get AI-powered recommendations for improving quality
- get_risk_files: Get files/components with highest risk scores

Authentication:
    On first run, the server will prompt you to authenticate via browser.
    Tokens are cached in ~/.argus/tokens.json for future use.

Usage with Claude Code:
    Add to ~/.claude/mcp_servers.json:
    {
        "argus-quality": {
            "command": "python",
            "args": ["-m", "src.mcp.quality_mcp"],
            "env": {
                "ARGUS_API_URL": "https://api.heyargus.ai"
            }
        }
    }

    No credentials needed - authentication happens via browser!
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

logger = structlog.get_logger()

# API URL configuration
ARGUS_API_URL = os.getenv("ARGUS_API_URL", "https://api.heyargus.ai")


# Tool definitions for MCP
QUALITY_TOOLS = [
    {
        "name": "get_quality_score",
        "description": (
            "Get the overall quality score and metrics for a project. "
            "Returns quality score (0-100), risk level, test coverage, "
            "and breakdown by component scores (coverage, error, CI, flaky)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "The project ID to get quality score for"
                }
            },
            "required": ["project_id"]
        }
    },
    {
        "name": "get_recommendations",
        "description": (
            "Get AI-powered recommendations for improving code quality. "
            "Analyzes production errors, test coverage gaps, and risk scores "
            "to provide actionable suggestions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "The project ID to get recommendations for"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of recommendations (default: 5)",
                    "default": 5
                }
            },
            "required": ["project_id"]
        }
    },
    {
        "name": "get_risk_files",
        "description": (
            "Get files and components with the highest risk scores. "
            "Risk is calculated based on error frequency, severity, "
            "test coverage, user impact, and recency."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "The project ID to get risk files for"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10
                },
                "entity_type": {
                    "type": "string",
                    "description": "Filter by entity type: 'component', 'page', 'flow', 'endpoint'",
                    "enum": ["component", "page", "flow", "endpoint"]
                }
            },
            "required": ["project_id"]
        }
    },
    {
        "name": "get_recent_errors",
        "description": (
            "Get recent production errors from Sentry or other error tracking. "
            "Returns errors with their severity, occurrence count, and status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "The project ID to get errors for"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of errors (default: 20)",
                    "default": 20
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status: 'new', 'analyzing', 'test_pending_review', 'test_generated', 'resolved'",
                    "enum": ["new", "analyzing", "test_pending_review", "test_generated", "resolved"]
                },
                "severity": {
                    "type": "string",
                    "description": "Filter by severity: 'fatal', 'error', 'warning', 'info'",
                    "enum": ["fatal", "error", "warning", "info"]
                }
            },
            "required": ["project_id"]
        }
    },
    {
        "name": "get_coverage_report",
        "description": (
            "Get the latest code coverage report for a project. "
            "Returns line, branch, and function coverage percentages."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "The project ID to get coverage for"
                },
                "branch": {
                    "type": "string",
                    "description": "Git branch name (default: 'main')",
                    "default": "main"
                }
            },
            "required": ["project_id"]
        }
    }
]


class QualityMCPServer:
    """MCP Server for Argus Quality Intelligence tools.

    Uses OAuth2 Device Flow for authentication - no credentials needed.
    On first run, opens browser for user to sign in.
    """

    def __init__(self, use_local_supabase: bool = False):
        """Initialize the MCP server.

        Args:
            use_local_supabase: If True, use direct Supabase access (requires env vars).
                               If False (default), use authenticated Argus API.
        """
        self.log = logger.bind(component="quality_mcp")
        self._supabase = None
        self._api_client = None
        self._message_id = 0
        self._use_local_supabase = use_local_supabase
        self._authenticated = False

    async def _ensure_authenticated(self):
        """Ensure we have valid authentication."""
        if self._authenticated:
            return

        if self._use_local_supabase:
            # Direct Supabase access (for development)
            from src.services.supabase_client import get_supabase_client
            self._supabase = get_supabase_client()
        else:
            # OAuth2 Device Flow authentication
            from src.mcp.auth import MCPAuthenticator, AuthenticatedClient
            self.log.info("Authenticating with Argus...")
            self._api_client = AuthenticatedClient(api_url=ARGUS_API_URL)
            await self._api_client.__aenter__()
            # Trigger authentication (will open browser if needed)
            await self._api_client.auth.get_token()
            self.log.info("Authentication successful")

        self._authenticated = True

    def _get_supabase(self):
        """Get Supabase client (for local mode only)."""
        if self._supabase is None and self._use_local_supabase:
            from src.services.supabase_client import get_supabase_client
            self._supabase = get_supabase_client()
        return self._supabase

    async def _api_get(self, path: str) -> dict:
        """Make authenticated GET request to Argus API."""
        if self._use_local_supabase:
            # Direct Supabase query
            supabase = self._get_supabase()
            return await supabase.request("GET", path)
        else:
            # Authenticated API call
            await self._ensure_authenticated()
            response = await self._api_client.get(f"/api/v1{path}")
            if response.status_code == 200:
                return {"data": response.json()}
            return {"error": response.text}

    async def handle_message(self, message: dict) -> dict:
        """Handle an incoming MCP message."""
        method = message.get("method", "")
        msg_id = message.get("id")

        if method == "initialize":
            return self._response(msg_id, {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "argus-quality",
                    "version": "1.0.0"
                },
                "capabilities": {
                    "tools": {}
                }
            })

        elif method == "tools/list":
            return self._response(msg_id, {"tools": QUALITY_TOOLS})

        elif method == "tools/call":
            params = message.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            try:
                result = await self._call_tool(tool_name, arguments)
                return self._response(msg_id, {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                })
            except Exception as e:
                self.log.error("Tool call failed", tool=tool_name, error=str(e))
                return self._error(msg_id, -32603, str(e))

        elif method == "notifications/initialized":
            # Client initialized notification, no response needed
            return None

        else:
            return self._error(msg_id, -32601, f"Method not found: {method}")

    def _response(self, msg_id: Any, result: dict) -> dict:
        """Create a JSON-RPC response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result
        }

    def _error(self, msg_id: Any, code: int, message: str) -> dict:
        """Create a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message}
        }

    async def _call_tool(self, name: str, arguments: dict) -> dict:
        """Execute a tool and return results."""
        if name == "get_quality_score":
            return await self._get_quality_score(arguments)
        elif name == "get_recommendations":
            return await self._get_recommendations(arguments)
        elif name == "get_risk_files":
            return await self._get_risk_files(arguments)
        elif name == "get_recent_errors":
            return await self._get_recent_errors(arguments)
        elif name == "get_coverage_report":
            return await self._get_coverage_report(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _get_quality_score(self, args: dict) -> dict:
        """Get quality score for a project."""
        project_id = args["project_id"]
        await self._ensure_authenticated()

        # Use API in authenticated mode
        if not self._use_local_supabase:
            response = await self._api_client.get(f"/api/v1/quality/score?project_id={project_id}")
            if response.status_code == 200:
                return response.json()
            self.log.warning("Quality score API failed", status=response.status_code)
            # Fall through to local calculation

        supabase = self._get_supabase()
        if not supabase:
            return {"error": "Not authenticated", "project_id": project_id}

        # Get from quality_scores table
        result = await supabase.request(
            f"/quality_scores?project_id=eq.{project_id}&limit=1"
        )

        if result.get("data") and len(result["data"]) > 0:
            score_data = result["data"][0]
            return {
                "project_id": project_id,
                "overall_score": score_data.get("overall_score", 50),
                "risk_level": (
                    "high" if score_data.get("overall_score", 50) < 40 else
                    "medium" if score_data.get("overall_score", 50) < 70 else "low"
                ),
                "component_scores": {
                    "coverage": score_data.get("coverage_score", 50),
                    "error": score_data.get("error_score", 50),
                    "ci": score_data.get("ci_score", 50),
                    "flaky": score_data.get("flaky_score", 50)
                },
                "metrics": {
                    "coverage_percent": float(score_data.get("coverage_percent", 0)),
                    "error_count": score_data.get("error_count", 0),
                    "error_count_24h": score_data.get("error_count_24h", 0),
                    "ci_success_rate": float(score_data.get("ci_success_rate", 0))
                },
                "trend": score_data.get("trend", "stable"),
                "calculated_at": score_data.get("calculated_at")
            }

        # Fallback: calculate from raw data
        events_result = await supabase.request(
            f"/production_events?project_id=eq.{project_id}&select=id,status,severity"
        )
        events = events_result.get("data", [])

        coverage_result = await supabase.request(
            f"/coverage_reports?project_id=eq.{project_id}&order=created_at.desc&limit=1"
        )
        coverage = coverage_result.get("data", [{}])[0] if coverage_result.get("data") else {}

        # Calculate basic score
        error_count = len(events)
        coverage_pct = float(coverage.get("lines_percent", 0))
        base_score = max(0, min(100, (coverage_pct * 0.4) + (100 - min(error_count * 5, 100)) * 0.6))

        return {
            "project_id": project_id,
            "overall_score": round(base_score),
            "risk_level": "high" if base_score < 40 else "medium" if base_score < 70 else "low",
            "component_scores": {
                "coverage": round(coverage_pct),
                "error": max(0, 100 - error_count * 10),
                "ci": 50,
                "flaky": 100
            },
            "metrics": {
                "coverage_percent": coverage_pct,
                "error_count": error_count,
                "error_count_24h": 0,
                "ci_success_rate": 0
            },
            "trend": "stable",
            "calculated_at": datetime.utcnow().isoformat()
        }

    async def _get_recommendations(self, args: dict) -> dict:
        """Get AI-powered recommendations for improving quality."""
        project_id = args["project_id"]
        limit = args.get("limit", 5)
        await self._ensure_authenticated()

        # API mode doesn't have a dedicated recommendations endpoint yet
        # So we always compute locally for now
        supabase = self._get_supabase()
        if not supabase and not self._use_local_supabase:
            # In API mode without direct supabase, return basic recommendations
            return {
                "project_id": project_id,
                "recommendations": [
                    {
                        "priority": "medium",
                        "category": "best_practices",
                        "title": "Generate Quality Report",
                        "description": "Run quality score calculation to get detailed recommendations.",
                        "action": "Use the Argus dashboard to view detailed quality metrics.",
                        "impact": "Better visibility into quality"
                    }
                ],
                "quality_score": 50,
                "generated_at": datetime.utcnow().isoformat()
            }

        recommendations = []

        # Get quality score
        score_result = await supabase.request(
            f"/quality_scores?project_id=eq.{project_id}&limit=1"
        )
        score_data = score_result.get("data", [{}])[0] if score_result.get("data") else {}

        # Get recent errors
        errors_result = await supabase.request(
            f"/production_events?project_id=eq.{project_id}&status=eq.new&order=severity.desc,occurrence_count.desc&limit=5"
        )
        errors = errors_result.get("data", [])

        # Get coverage
        coverage_result = await supabase.request(
            f"/coverage_reports?project_id=eq.{project_id}&order=created_at.desc&limit=1"
        )
        coverage = coverage_result.get("data", [{}])[0] if coverage_result.get("data") else {}

        # Get risk scores
        risk_result = await supabase.request(
            f"/risk_scores?project_id=eq.{project_id}&order=overall_risk_score.desc&limit=5"
        )
        risks = risk_result.get("data", []) if not risk_result.get("error") else []

        # Generate recommendations based on data
        coverage_pct = float(coverage.get("lines_percent", 0))
        if coverage_pct < 80:
            recommendations.append({
                "priority": "high" if coverage_pct < 50 else "medium",
                "category": "coverage",
                "title": "Increase Test Coverage",
                "description": f"Current line coverage is {coverage_pct:.1f}%. Aim for at least 80% coverage.",
                "action": "Focus on adding tests for critical paths and uncovered code.",
                "impact": f"+{min(30, 80 - coverage_pct):.0f} quality score points"
            })

        # Recommendations based on errors
        if errors:
            fatal_errors = [e for e in errors if e.get("severity") == "fatal"]
            if fatal_errors:
                recommendations.append({
                    "priority": "critical",
                    "category": "errors",
                    "title": "Fix Critical Production Errors",
                    "description": f"{len(fatal_errors)} fatal error(s) detected in production.",
                    "action": f"Investigate: {fatal_errors[0].get('title', 'Unknown error')}",
                    "impact": "+20 quality score points per fix"
                })
            else:
                recommendations.append({
                    "priority": "high",
                    "category": "errors",
                    "title": "Address Production Errors",
                    "description": f"{len(errors)} unresolved error(s) in production.",
                    "action": "Generate tests for these errors to prevent regression.",
                    "impact": "+10 quality score points per resolution"
                })

        # Recommendations based on risk scores
        high_risk = [r for r in risks if r.get("overall_risk_score", 0) > 70]
        if high_risk:
            recommendations.append({
                "priority": "high",
                "category": "risk",
                "title": "Reduce High-Risk Components",
                "description": f"{len(high_risk)} component(s) have high risk scores.",
                "action": f"Add tests for: {high_risk[0].get('entity_identifier', 'Unknown')}",
                "impact": "Reduces regression risk significantly"
            })

        # CI/CD recommendations
        ci_success = float(score_data.get("ci_success_rate", 100))
        if ci_success < 90:
            recommendations.append({
                "priority": "medium",
                "category": "ci",
                "title": "Improve CI Success Rate",
                "description": f"CI success rate is {ci_success:.1f}%. Target 95%+.",
                "action": "Investigate flaky tests and fix failing builds.",
                "impact": "+15 quality score points"
            })

        # General best practices
        if len(recommendations) < limit:
            recommendations.append({
                "priority": "low",
                "category": "best_practices",
                "title": "Enable Visual Regression Testing",
                "description": "Catch UI regressions before they reach production.",
                "action": "Set up Argus Visual for screenshot comparison.",
                "impact": "Prevents visual bugs"
            })

        return {
            "project_id": project_id,
            "recommendations": recommendations[:limit],
            "quality_score": score_data.get("overall_score", 50),
            "generated_at": datetime.utcnow().isoformat()
        }

    async def _get_risk_files(self, args: dict) -> dict:
        """Get files/components with highest risk scores."""
        project_id = args["project_id"]
        limit = args.get("limit", 10)
        entity_type = args.get("entity_type")
        await self._ensure_authenticated()

        # Use API in authenticated mode
        if not self._use_local_supabase:
            params = f"project_id={project_id}&limit={limit}"
            if entity_type:
                params += f"&entity_type={entity_type}"
            response = await self._api_client.get(f"/api/v1/quality/risk-scores?{params}")
            if response.status_code == 200:
                return response.json()
            self.log.warning("Risk scores API failed", status=response.status_code)

        supabase = self._get_supabase()
        if not supabase:
            return {"project_id": project_id, "risk_files": [], "message": "Not authenticated"}

        query = f"/risk_scores?project_id=eq.{project_id}"
        if entity_type:
            query += f"&entity_type=eq.{entity_type}"
        query += f"&order=overall_risk_score.desc&limit={limit}"

        result = await supabase.request(query)

        if result.get("error"):
            # Table might not exist yet
            return {
                "project_id": project_id,
                "risk_files": [],
                "message": "No risk scores calculated yet. Run risk calculation first."
            }

        risk_files = []
        for r in result.get("data", []):
            risk_files.append({
                "entity": r.get("entity_identifier"),
                "type": r.get("entity_type"),
                "risk_score": r.get("overall_risk_score", 0),
                "risk_level": (
                    "critical" if r.get("overall_risk_score", 0) > 80 else
                    "high" if r.get("overall_risk_score", 0) > 60 else
                    "medium" if r.get("overall_risk_score", 0) > 40 else "low"
                ),
                "factors": r.get("factors", {}),
                "error_count": r.get("error_count", 0),
                "affected_users": r.get("affected_users", 0),
                "trend": r.get("trend", "stable")
            })

        return {
            "project_id": project_id,
            "risk_files": risk_files,
            "total_count": len(risk_files)
        }

    async def _get_recent_errors(self, args: dict) -> dict:
        """Get recent production errors."""
        project_id = args["project_id"]
        limit = args.get("limit", 20)
        status = args.get("status")
        severity = args.get("severity")
        await self._ensure_authenticated()

        # Use API in authenticated mode
        if not self._use_local_supabase:
            params = f"project_id={project_id}&limit={limit}"
            if status:
                params += f"&status={status}"
            if severity:
                params += f"&severity={severity}"
            response = await self._api_client.get(f"/api/v1/quality/events?{params}")
            if response.status_code == 200:
                return response.json()
            self.log.warning("Events API failed", status=response.status_code)

        supabase = self._get_supabase()
        if not supabase:
            return {"project_id": project_id, "errors": [], "message": "Not authenticated"}

        query = f"/production_events?project_id=eq.{project_id}"
        if status:
            query += f"&status=eq.{status}"
        if severity:
            query += f"&severity=eq.{severity}"
        query += f"&order=created_at.desc&limit={limit}"

        result = await supabase.request(query)

        errors = []
        for e in result.get("data", []):
            errors.append({
                "id": e.get("id"),
                "title": e.get("title"),
                "message": e.get("message"),
                "severity": e.get("severity"),
                "status": e.get("status"),
                "url": e.get("url"),
                "component": e.get("component"),
                "occurrence_count": e.get("occurrence_count", 1),
                "affected_users": e.get("affected_users", 0),
                "first_seen": e.get("first_seen_at"),
                "last_seen": e.get("last_seen_at"),
                "source": e.get("source")
            })

        return {
            "project_id": project_id,
            "errors": errors,
            "total_count": len(errors)
        }

    async def _get_coverage_report(self, args: dict) -> dict:
        """Get latest coverage report."""
        project_id = args["project_id"]
        branch = args.get("branch", "main")
        await self._ensure_authenticated()

        # Use API in authenticated mode
        if not self._use_local_supabase:
            response = await self._api_client.get(
                f"/api/v1/quality/stats?project_id={project_id}"
            )
            if response.status_code == 200:
                data = response.json()
                # Extract coverage data from stats
                return {
                    "project_id": project_id,
                    "branch": branch,
                    "coverage": {
                        "lines": {
                            "percent": data.get("coverage_percent", 0)
                        }
                    },
                    "message": "Coverage data from quality stats"
                }

        supabase = self._get_supabase()
        if not supabase:
            return {"project_id": project_id, "branch": branch, "coverage": None, "message": "Not authenticated"}

        result = await supabase.request(
            f"/coverage_reports?project_id=eq.{project_id}&branch=eq.{branch}&order=created_at.desc&limit=1"
        )

        if not result.get("data") or len(result["data"]) == 0:
            return {
                "project_id": project_id,
                "branch": branch,
                "coverage": None,
                "message": "No coverage report found for this branch"
            }

        report = result["data"][0]

        return {
            "project_id": project_id,
            "branch": branch,
            "commit_sha": report.get("commit_sha"),
            "coverage": {
                "lines": {
                    "total": report.get("lines_total", 0),
                    "covered": report.get("lines_covered", 0),
                    "percent": float(report.get("lines_percent", 0))
                },
                "branches": {
                    "total": report.get("branches_total", 0),
                    "covered": report.get("branches_covered", 0),
                    "percent": float(report.get("branches_percent", 0))
                },
                "functions": {
                    "total": report.get("functions_total", 0),
                    "covered": report.get("functions_covered", 0),
                    "percent": float(report.get("functions_percent", 0))
                }
            },
            "format": report.get("format"),
            "created_at": report.get("created_at")
        }

    async def run(self):
        """Run the MCP server, reading from stdin and writing to stdout."""
        self.log.info("Argus Quality MCP server starting")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break

                message = json.loads(line.decode())
                response = await self.handle_message(message)

                if response:
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()

            except json.JSONDecodeError as e:
                self.log.error("Invalid JSON", error=str(e))
            except Exception as e:
                self.log.exception("Error handling message", error=str(e))


def create_quality_mcp_tools() -> list[dict]:
    """
    Create tool definitions for Claude to use Argus Quality tools.

    These tools can be passed to Claude's tool use API for direct integration.
    """
    return QUALITY_TOOLS


# MCP server configuration for Claude Code
# No credentials needed - authentication happens via browser on first use!
# Add to ~/.claude/mcp_servers.json or your IDE's MCP config
MCP_CONFIG = {
    "argus-quality": {
        "command": "python",
        "args": ["-m", "src.mcp.quality_mcp"],
        "env": {
            # Optional: Override API URL (default: https://api.heyargus.ai)
            # "ARGUS_API_URL": "https://api.heyargus.ai"
        }
    }
}


async def main():
    """Main entry point for the MCP server."""
    server = QualityMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
