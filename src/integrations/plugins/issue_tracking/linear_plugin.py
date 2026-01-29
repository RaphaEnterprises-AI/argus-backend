"""Linear integration plugin.

Linear uses GraphQL API with API key authentication.
API Documentation: https://developers.linear.app/docs/graphql/working-with-the-graphql-api
"""

import time
from datetime import datetime
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


class LinearPlugin(IntegrationPlugin):
    """Linear integration for modern issue tracking and project management.

    Linear provides a fast, streamlined interface for software project management.
    Uses GraphQL API with API key authentication.

    API Documentation: https://developers.linear.app/docs/graphql/working-with-the-graphql-api
    """

    GRAPHQL_ENDPOINT = "https://api.linear.app/graphql"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return Linear integration metadata."""
        return IntegrationMetadata(
            slug="linear",
            name="Linear",
            description="Connect to Linear for modern issue tracking, project management, and roadmaps. Built for speed and efficiency.",
            category=IntegrationCategory.ISSUE_TRACKING,
            auth_type=AuthType.API_KEY,
            icon_url="https://linear.app/static/favicon-32x32.png",
            docs_url="https://developers.linear.app/docs/graphql/working-with-the-graphql-api",
            website_url="https://linear.app",
            features=[
                "issues",
                "projects",
                "cycles",
                "roadmaps",
                "teams",
                "labels",
                "workflows",
                "integrations",
                "webhooks",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key",
                    type="password",
                    required=True,
                    placeholder="lin_api_...",
                    help_text="Generate at Linear Settings > API > Personal API keys",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="team_id",
                    label="Team ID",
                    type="text",
                    required=False,
                    placeholder="TEAM-123",
                    help_text="Optional: Limit sync to a specific team",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["issues", "projects", "cycles", "teams", "users", "labels"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get authentication headers for GraphQL requests."""
        return {
            "Authorization": credentials.api_key,
            "Content-Type": "application/json",
        }

    async def _graphql_request(
        self,
        credentials: IntegrationCredentials,
        query: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a GraphQL request."""
        headers = self._get_headers(credentials)
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = await self.http_client.post(
            self.GRAPHQL_ENDPOINT,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        if "errors" in result:
            raise Exception(f"GraphQL errors: {result['errors']}")

        return result.get("data", {})

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Linear API.

        Uses query { viewer { id } } to verify credentials.
        """
        start_time = time.time()

        try:
            # Query current user and organization info
            query = """
            query {
                viewer {
                    id
                    name
                    email
                    active
                    admin
                    createdAt
                    organization {
                        id
                        name
                        urlKey
                        createdAt
                    }
                }
            }
            """

            data = await self._graphql_request(credentials, query)
            latency_ms = (time.time() - start_time) * 1000

            viewer = data.get("viewer", {})
            org = viewer.get("organization", {})

            if viewer:
                # Get team count for permissions context
                teams_query = """
                query {
                    teams {
                        nodes {
                            id
                            name
                        }
                    }
                }
                """
                try:
                    teams_data = await self._graphql_request(credentials, teams_query)
                    team_count = len(teams_data.get("teams", {}).get("nodes", []))
                    permissions = ["read:issues", "read:projects", "read:teams", f"access:{team_count}_teams"]
                except Exception:
                    permissions = ["read:issues", "read:projects"]

                return ConnectionTestResult(
                    success=True,
                    message=f"Connected as {viewer.get('name', viewer.get('email', 'Unknown'))}",
                    latency_ms=latency_ms,
                    api_version="GraphQL",
                    permissions=permissions,
                    account_info={
                        "user_id": viewer.get("id"),
                        "email": viewer.get("email"),
                        "name": viewer.get("name"),
                        "active": viewer.get("active"),
                        "admin": viewer.get("admin"),
                        "organization_id": org.get("id"),
                        "organization_name": org.get("name"),
                        "organization_url_key": org.get("urlKey"),
                    },
                )

            return ConnectionTestResult(
                success=False,
                message="No viewer data returned",
                latency_ms=latency_ms,
                error_code="NO_DATA",
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            if e.response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed. Check your API key.",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )
            return ConnectionTestResult(
                success=False,
                message=f"HTTP error: {e.response.status_code}",
                latency_ms=latency_ms,
                error_code="HTTP_ERROR",
                error_details={"status_code": e.response.status_code},
            )
        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection test failed: {str(e)}",
                error_code="UNKNOWN_ERROR",
                error_details={"exception": str(e)},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Linear.

        Syncs issues, projects, cycles, teams, and labels.
        """
        started_at = datetime.utcnow()
        sync_id = f"linear_sync_{int(started_at.timestamp())}"

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_items = 0

        # Default to all resources if none specified
        resources_to_sync = resources or ["issues", "projects", "cycles", "teams"]

        try:
            # Sync teams
            if "teams" in resources_to_sync:
                try:
                    teams_query = """
                    query($first: Int) {
                        teams(first: $first) {
                            nodes {
                                id
                                name
                                key
                                description
                                private
                                createdAt
                                issueCount
                            }
                        }
                    }
                    """
                    teams_data = await self._graphql_request(
                        credentials, teams_query, {"first": min(limit, 50)}
                    )
                    teams = teams_data.get("teams", {}).get("nodes", [])
                    data["teams"] = [
                        {
                            "id": t.get("id"),
                            "name": t.get("name"),
                            "key": t.get("key"),
                            "description": t.get("description"),
                            "private": t.get("private"),
                            "created_at": t.get("createdAt"),
                            "issue_count": t.get("issueCount"),
                        }
                        for t in teams
                    ]
                    total_items += len(teams)
                except Exception as e:
                    errors.append(f"Error fetching teams: {str(e)}")

            # Sync projects
            if "projects" in resources_to_sync:
                try:
                    projects_query = """
                    query($first: Int) {
                        projects(first: $first, orderBy: updatedAt) {
                            nodes {
                                id
                                name
                                description
                                state
                                progress
                                targetDate
                                startDate
                                createdAt
                                updatedAt
                                lead {
                                    name
                                }
                                teams {
                                    nodes {
                                        name
                                    }
                                }
                            }
                        }
                    }
                    """
                    projects_data = await self._graphql_request(
                        credentials, projects_query, {"first": min(limit, 50)}
                    )
                    projects = projects_data.get("projects", {}).get("nodes", [])
                    data["projects"] = [
                        {
                            "id": p.get("id"),
                            "name": p.get("name"),
                            "description": p.get("description"),
                            "state": p.get("state"),
                            "progress": p.get("progress"),
                            "target_date": p.get("targetDate"),
                            "start_date": p.get("startDate"),
                            "created_at": p.get("createdAt"),
                            "updated_at": p.get("updatedAt"),
                            "lead": p.get("lead", {}).get("name") if p.get("lead") else None,
                            "teams": [t.get("name") for t in p.get("teams", {}).get("nodes", [])],
                        }
                        for p in projects
                    ]
                    total_items += len(projects)
                except Exception as e:
                    errors.append(f"Error fetching projects: {str(e)}")

            # Sync cycles
            if "cycles" in resources_to_sync:
                try:
                    cycles_query = """
                    query($first: Int) {
                        cycles(first: $first, orderBy: updatedAt) {
                            nodes {
                                id
                                name
                                number
                                description
                                startsAt
                                endsAt
                                progress
                                completedAt
                                team {
                                    name
                                }
                                issues {
                                    nodes {
                                        id
                                    }
                                }
                            }
                        }
                    }
                    """
                    cycles_data = await self._graphql_request(
                        credentials, cycles_query, {"first": min(limit, 50)}
                    )
                    cycles = cycles_data.get("cycles", {}).get("nodes", [])
                    data["cycles"] = [
                        {
                            "id": c.get("id"),
                            "name": c.get("name"),
                            "number": c.get("number"),
                            "description": c.get("description"),
                            "starts_at": c.get("startsAt"),
                            "ends_at": c.get("endsAt"),
                            "progress": c.get("progress"),
                            "completed_at": c.get("completedAt"),
                            "team": c.get("team", {}).get("name") if c.get("team") else None,
                            "issue_count": len(c.get("issues", {}).get("nodes", [])),
                        }
                        for c in cycles
                    ]
                    total_items += len(cycles)
                except Exception as e:
                    errors.append(f"Error fetching cycles: {str(e)}")

            # Sync recent issues
            if "issues" in resources_to_sync:
                try:
                    # Build filter for incremental sync
                    filter_arg = ""
                    variables: dict[str, Any] = {"first": min(limit, 100)}

                    if since:
                        filter_arg = ", filter: { updatedAt: { gte: $since } }"
                        variables["since"] = since.isoformat()

                    issues_query = f"""
                    query($first: Int{', $since: DateTime' if since else ''}) {{
                        issues(first: $first, orderBy: updatedAt{filter_arg}) {{
                            nodes {{
                                id
                                identifier
                                title
                                description
                                priority
                                priorityLabel
                                state {{
                                    name
                                    type
                                }}
                                assignee {{
                                    name
                                }}
                                creator {{
                                    name
                                }}
                                team {{
                                    name
                                    key
                                }}
                                project {{
                                    name
                                }}
                                labels {{
                                    nodes {{
                                        name
                                    }}
                                }}
                                createdAt
                                updatedAt
                                completedAt
                            }}
                        }}
                    }}
                    """
                    issues_data = await self._graphql_request(
                        credentials, issues_query, variables
                    )
                    issues = issues_data.get("issues", {}).get("nodes", [])
                    data["issues"] = [
                        {
                            "id": i.get("id"),
                            "identifier": i.get("identifier"),
                            "title": i.get("title"),
                            "description": i.get("description", "")[:500] if i.get("description") else None,
                            "priority": i.get("priority"),
                            "priority_label": i.get("priorityLabel"),
                            "state": i.get("state", {}).get("name") if i.get("state") else None,
                            "state_type": i.get("state", {}).get("type") if i.get("state") else None,
                            "assignee": i.get("assignee", {}).get("name") if i.get("assignee") else None,
                            "creator": i.get("creator", {}).get("name") if i.get("creator") else None,
                            "team": i.get("team", {}).get("name") if i.get("team") else None,
                            "team_key": i.get("team", {}).get("key") if i.get("team") else None,
                            "project": i.get("project", {}).get("name") if i.get("project") else None,
                            "labels": [l.get("name") for l in i.get("labels", {}).get("nodes", [])],
                            "created_at": i.get("createdAt"),
                            "updated_at": i.get("updatedAt"),
                            "completed_at": i.get("completedAt"),
                        }
                        for i in issues
                    ]
                    total_items += len(issues)
                except Exception as e:
                    errors.append(f"Error fetching issues: {str(e)}")

            # Sync labels
            if "labels" in resources_to_sync:
                try:
                    labels_query = """
                    query($first: Int) {
                        issueLabels(first: $first) {
                            nodes {
                                id
                                name
                                description
                                color
                                team {
                                    name
                                }
                            }
                        }
                    }
                    """
                    labels_data = await self._graphql_request(
                        credentials, labels_query, {"first": min(limit, 100)}
                    )
                    labels = labels_data.get("issueLabels", {}).get("nodes", [])
                    data["labels"] = [
                        {
                            "id": l.get("id"),
                            "name": l.get("name"),
                            "description": l.get("description"),
                            "color": l.get("color"),
                            "team": l.get("team", {}).get("name") if l.get("team") else None,
                        }
                        for l in labels
                    ]
                    total_items += len(labels)
                except Exception as e:
                    errors.append(f"Error fetching labels: {str(e)}")

            completed_at = datetime.utcnow()

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_items} items from Linear" if not errors else f"Partial sync: {total_items} items with {len(errors)} errors",
                data_points_synced=total_items,
                sync_id=sync_id,
                started_at=started_at,
                completed_at=completed_at,
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                errors=[str(e)],
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
