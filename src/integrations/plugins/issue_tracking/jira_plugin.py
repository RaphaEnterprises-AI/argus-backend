"""Jira integration plugin.

Supports both Jira Cloud and Jira Server/Data Center with REST API v3.
Authentication via email + API token (Cloud) or username + password (Server).
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


class JiraPlugin(IntegrationPlugin):
    """Jira integration for issue tracking, projects, and sprints.

    Supports:
    - Jira Cloud: Uses email + API token authentication
    - Jira Server/Data Center: Uses username + password (basic auth)

    API Documentation: https://developer.atlassian.com/cloud/jira/platform/rest/v3/
    """

    # Default cloud URL (can be overridden with instance_url)
    DEFAULT_CLOUD_URL = "https://api.atlassian.com"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return Jira integration metadata."""
        return IntegrationMetadata(
            slug="jira",
            name="Jira",
            description="Connect to Jira for issue tracking, project management, and sprint planning. Supports both Cloud and Server/Data Center deployments.",
            category=IntegrationCategory.ISSUE_TRACKING,
            auth_type=AuthType.BASIC_AUTH,
            icon_url="https://cdn.worldvectorlogo.com/logos/jira-1.svg",
            docs_url="https://developer.atlassian.com/cloud/jira/platform/rest/v3/",
            website_url="https://www.atlassian.com/software/jira",
            features=[
                "issues",
                "projects",
                "sprints",
                "boards",
                "workflows",
                "custom_fields",
                "attachments",
                "comments",
                "webhooks",
            ],
            required_fields=[
                FieldDefinition(
                    name="email",
                    label="Email",
                    type="email",
                    required=True,
                    placeholder="user@example.com",
                    help_text="Your Atlassian account email (Cloud) or username (Server)",
                ),
                FieldDefinition(
                    name="api_key",
                    label="API Token",
                    type="password",
                    required=True,
                    placeholder="ATATT3xFfGF0...",
                    help_text="Generate at https://id.atlassian.com/manage-profile/security/api-tokens (Cloud) or use password (Server)",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="instance_url",
                    label="Instance URL",
                    type="url",
                    required=False,
                    placeholder="https://your-domain.atlassian.net",
                    help_text="Your Jira instance URL. Required for Cloud and Server deployments.",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["projects", "issues", "sprints", "boards", "users"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the base URL for API requests."""
        if credentials.instance_url:
            # Remove trailing slash if present
            return credentials.instance_url.rstrip("/")
        raise ValueError("instance_url is required for Jira integration")

    def _get_auth_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get authentication headers for API requests."""
        import base64

        # Create basic auth header
        auth_string = f"{credentials.email}:{credentials.api_key}"
        auth_bytes = base64.b64encode(auth_string.encode()).decode()

        return {
            "Authorization": f"Basic {auth_bytes}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Jira API.

        Uses GET /rest/api/3/myself to verify credentials.
        """
        start_time = time.time()

        try:
            base_url = self._get_base_url(credentials)
            headers = self._get_auth_headers(credentials)

            # Test connection with /myself endpoint
            response = await self.http_client.get(
                f"{base_url}/rest/api/3/myself",
                headers=headers,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                user_data = response.json()

                # Get server info for API version
                server_response = await self.http_client.get(
                    f"{base_url}/rest/api/3/serverInfo",
                    headers=headers,
                )
                server_info = {}
                api_version = "3"
                if server_response.status_code == 200:
                    server_info = server_response.json()
                    api_version = server_info.get("version", "3")

                # Get user permissions
                permissions_response = await self.http_client.get(
                    f"{base_url}/rest/api/3/mypermissions",
                    headers=headers,
                )
                permissions = []
                if permissions_response.status_code == 200:
                    perms_data = permissions_response.json()
                    permissions = [
                        key
                        for key, value in perms_data.get("permissions", {}).items()
                        if value.get("havePermission", False)
                    ]

                return ConnectionTestResult(
                    success=True,
                    message=f"Connected as {user_data.get('displayName', user_data.get('emailAddress', 'Unknown'))}",
                    latency_ms=latency_ms,
                    api_version=api_version,
                    permissions=permissions[:20],  # Limit to first 20 permissions
                    account_info={
                        "account_id": user_data.get("accountId"),
                        "email": user_data.get("emailAddress"),
                        "display_name": user_data.get("displayName"),
                        "active": user_data.get("active"),
                        "account_type": user_data.get("accountType"),
                        "time_zone": user_data.get("timeZone"),
                        "server_title": server_info.get("serverTitle"),
                        "base_url": server_info.get("baseUrl"),
                        "deployment_type": server_info.get("deploymentType"),
                    },
                )

            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed. Check your email and API token.",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )
            elif response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden. Your account may not have API access.",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                    error_details={"status_code": 403},
                )
            else:
                return ConnectionTestResult(
                    success=False,
                    message=f"Unexpected response: {response.status_code}",
                    latency_ms=latency_ms,
                    error_code="UNEXPECTED_ERROR",
                    error_details={
                        "status_code": response.status_code,
                        "response": response.text[:500],
                    },
                )

        except ValueError as e:
            return ConnectionTestResult(
                success=False,
                message=str(e),
                error_code="CONFIGURATION_ERROR",
            )
        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out. Check your instance URL.",
                error_code="TIMEOUT",
            )
        except httpx.ConnectError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Failed to connect: {str(e)}",
                error_code="CONNECTION_ERROR",
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
        """Sync data from Jira.

        Syncs projects, issues, sprints, and boards.
        """
        started_at = datetime.utcnow()
        sync_id = f"jira_sync_{int(started_at.timestamp())}"

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_items = 0

        # Default to all resources if none specified
        resources_to_sync = resources or ["projects", "issues", "sprints"]

        try:
            base_url = self._get_base_url(credentials)
            headers = self._get_auth_headers(credentials)

            # Sync projects
            if "projects" in resources_to_sync:
                try:
                    projects_response = await self.http_client.get(
                        f"{base_url}/rest/api/3/project/search",
                        headers=headers,
                        params={"maxResults": min(limit, 50)},
                    )

                    if projects_response.status_code == 200:
                        projects_data = projects_response.json()
                        projects = projects_data.get("values", [])
                        data["projects"] = [
                            {
                                "id": p.get("id"),
                                "key": p.get("key"),
                                "name": p.get("name"),
                                "project_type": p.get("projectTypeKey"),
                                "style": p.get("style"),
                                "is_private": p.get("isPrivate"),
                                "lead": p.get("lead", {}).get("displayName") if p.get("lead") else None,
                            }
                            for p in projects
                        ]
                        total_items += len(projects)
                    else:
                        errors.append(f"Failed to fetch projects: {projects_response.status_code}")
                except Exception as e:
                    errors.append(f"Error fetching projects: {str(e)}")

            # Sync recent issues
            if "issues" in resources_to_sync:
                try:
                    jql = "ORDER BY updated DESC"
                    if since:
                        since_str = since.strftime("%Y-%m-%d %H:%M")
                        jql = f"updated >= '{since_str}' ORDER BY updated DESC"

                    issues_response = await self.http_client.get(
                        f"{base_url}/rest/api/3/search",
                        headers=headers,
                        params={
                            "jql": jql,
                            "maxResults": min(limit, 100),
                            "fields": "summary,status,priority,assignee,reporter,created,updated,issuetype,project",
                        },
                    )

                    if issues_response.status_code == 200:
                        issues_data = issues_response.json()
                        issues = issues_data.get("issues", [])
                        data["issues"] = [
                            {
                                "id": i.get("id"),
                                "key": i.get("key"),
                                "summary": i.get("fields", {}).get("summary"),
                                "status": i.get("fields", {}).get("status", {}).get("name"),
                                "priority": i.get("fields", {}).get("priority", {}).get("name"),
                                "issue_type": i.get("fields", {}).get("issuetype", {}).get("name"),
                                "project_key": i.get("fields", {}).get("project", {}).get("key"),
                                "assignee": i.get("fields", {}).get("assignee", {}).get("displayName") if i.get("fields", {}).get("assignee") else None,
                                "reporter": i.get("fields", {}).get("reporter", {}).get("displayName") if i.get("fields", {}).get("reporter") else None,
                                "created": i.get("fields", {}).get("created"),
                                "updated": i.get("fields", {}).get("updated"),
                            }
                            for i in issues
                        ]
                        total_items += len(issues)
                        data["issues_total"] = issues_data.get("total", len(issues))
                    else:
                        errors.append(f"Failed to fetch issues: {issues_response.status_code}")
                except Exception as e:
                    errors.append(f"Error fetching issues: {str(e)}")

            # Sync sprints (requires Jira Software)
            if "sprints" in resources_to_sync:
                try:
                    # First get all boards
                    boards_response = await self.http_client.get(
                        f"{base_url}/rest/agile/1.0/board",
                        headers=headers,
                        params={"maxResults": 50},
                    )

                    if boards_response.status_code == 200:
                        boards_data = boards_response.json()
                        boards = boards_data.get("values", [])
                        data["boards"] = [
                            {
                                "id": b.get("id"),
                                "name": b.get("name"),
                                "type": b.get("type"),
                                "project_key": b.get("location", {}).get("projectKey"),
                            }
                            for b in boards
                        ]
                        total_items += len(boards)

                        # Get sprints for scrum boards
                        all_sprints = []
                        for board in boards[:5]:  # Limit to first 5 boards
                            if board.get("type") == "scrum":
                                try:
                                    sprints_response = await self.http_client.get(
                                        f"{base_url}/rest/agile/1.0/board/{board.get('id')}/sprint",
                                        headers=headers,
                                        params={"maxResults": 10, "state": "active,future"},
                                    )
                                    if sprints_response.status_code == 200:
                                        sprints_data = sprints_response.json()
                                        for s in sprints_data.get("values", []):
                                            all_sprints.append({
                                                "id": s.get("id"),
                                                "name": s.get("name"),
                                                "state": s.get("state"),
                                                "start_date": s.get("startDate"),
                                                "end_date": s.get("endDate"),
                                                "board_id": board.get("id"),
                                                "board_name": board.get("name"),
                                            })
                                except Exception:
                                    pass  # Skip individual board errors

                        data["sprints"] = all_sprints
                        total_items += len(all_sprints)
                    elif boards_response.status_code == 404:
                        warnings.append("Agile API not available (Jira Software may not be installed)")
                    else:
                        warnings.append(f"Failed to fetch boards: {boards_response.status_code}")
                except Exception as e:
                    warnings.append(f"Error fetching sprints: {str(e)}")

            completed_at = datetime.utcnow()

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_items} items from Jira" if not errors else f"Partial sync: {total_items} items with {len(errors)} errors",
                data_points_synced=total_items,
                sync_id=sync_id,
                started_at=started_at,
                completed_at=completed_at,
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except ValueError as e:
            return SyncResult(
                success=False,
                message=str(e),
                errors=[str(e)],
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
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
