"""Asana integration plugin.

Asana REST API with Personal Access Token authentication.
API Documentation: https://developers.asana.com/docs
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


class AsanaPlugin(IntegrationPlugin):
    """Asana integration for task management and project tracking.

    Asana provides work management for teams with projects, tasks, timelines,
    and collaboration features.

    Uses Personal Access Token (PAT) authentication or OAuth2.
    API Documentation: https://developers.asana.com/docs
    """

    BASE_URL = "https://app.asana.com/api/1.0"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return Asana integration metadata."""
        return IntegrationMetadata(
            slug="asana",
            name="Asana",
            description="Connect to Asana for task management, project tracking, and team collaboration. Track work across projects, portfolios, and goals.",
            category=IntegrationCategory.ISSUE_TRACKING,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://luna1.co/eb0e4e.png",
            docs_url="https://developers.asana.com/docs",
            website_url="https://asana.com",
            features=[
                "tasks",
                "projects",
                "workspaces",
                "portfolios",
                "goals",
                "sections",
                "tags",
                "custom_fields",
                "attachments",
                "comments",
                "webhooks",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Personal Access Token",
                    type="password",
                    required=True,
                    placeholder="1/1234567890:abcdef...",
                    help_text="Generate at Asana Settings > Apps > Developer Apps > Personal Access Token",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="workspace_id",
                    label="Workspace ID",
                    type="text",
                    required=False,
                    placeholder="1234567890",
                    help_text="Optional: Limit sync to a specific workspace",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["workspaces", "projects", "tasks", "sections", "tags", "users"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get authentication headers for API requests."""
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Asana API.

        Uses GET /users/me to verify credentials.
        """
        start_time = time.time()

        try:
            headers = self._get_headers(credentials)

            # Test connection with /users/me endpoint
            response = await self.http_client.get(
                f"{self.BASE_URL}/users/me",
                headers=headers,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                user_data = result.get("data", {})

                # Get workspaces for permissions context
                workspaces_response = await self.http_client.get(
                    f"{self.BASE_URL}/workspaces",
                    headers=headers,
                    params={"limit": 100},
                )
                workspaces = []
                if workspaces_response.status_code == 200:
                    ws_result = workspaces_response.json()
                    workspaces = ws_result.get("data", [])

                permissions = [
                    "read:users",
                    "read:workspaces",
                    "read:projects",
                    "read:tasks",
                    f"access:{len(workspaces)}_workspaces",
                ]

                return ConnectionTestResult(
                    success=True,
                    message=f"Connected as {user_data.get('name', user_data.get('email', 'Unknown'))}",
                    latency_ms=latency_ms,
                    api_version="1.0",
                    permissions=permissions,
                    account_info={
                        "user_id": user_data.get("gid"),
                        "name": user_data.get("name"),
                        "email": user_data.get("email"),
                        "photo": user_data.get("photo", {}).get("image_128x128") if user_data.get("photo") else None,
                        "workspaces": [
                            {"id": ws.get("gid"), "name": ws.get("name")}
                            for ws in workspaces[:10]  # Limit to first 10
                        ],
                    },
                )

            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed. Check your Personal Access Token.",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )
            elif response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden. Your token may not have required permissions.",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                    error_details={"status_code": 403},
                )
            else:
                error_data = response.json() if response.content else {}
                return ConnectionTestResult(
                    success=False,
                    message=f"Unexpected response: {response.status_code}",
                    latency_ms=latency_ms,
                    error_code="UNEXPECTED_ERROR",
                    error_details={
                        "status_code": response.status_code,
                        "errors": error_data.get("errors", []),
                    },
                )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
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
        """Sync data from Asana.

        Syncs workspaces, projects, and tasks.
        """
        started_at = datetime.utcnow()
        sync_id = f"asana_sync_{int(started_at.timestamp())}"

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_items = 0

        # Default to all resources if none specified
        resources_to_sync = resources or ["workspaces", "projects", "tasks"]

        try:
            headers = self._get_headers(credentials)

            # Sync workspaces first (needed for other resources)
            workspaces = []
            if "workspaces" in resources_to_sync or credentials.workspace_id is None:
                try:
                    workspaces_response = await self.http_client.get(
                        f"{self.BASE_URL}/workspaces",
                        headers=headers,
                        params={"limit": 100},
                    )

                    if workspaces_response.status_code == 200:
                        ws_result = workspaces_response.json()
                        workspaces = ws_result.get("data", [])
                        data["workspaces"] = [
                            {
                                "id": ws.get("gid"),
                                "name": ws.get("name"),
                                "is_organization": ws.get("is_organization"),
                            }
                            for ws in workspaces
                        ]
                        total_items += len(workspaces)
                    else:
                        errors.append(f"Failed to fetch workspaces: {workspaces_response.status_code}")
                except Exception as e:
                    errors.append(f"Error fetching workspaces: {str(e)}")

            # Filter to specific workspace if provided
            target_workspaces = workspaces
            if credentials.workspace_id:
                target_workspaces = [{"gid": credentials.workspace_id}]

            # Sync projects
            if "projects" in resources_to_sync:
                all_projects = []
                for ws in target_workspaces[:5]:  # Limit to first 5 workspaces
                    try:
                        params: dict[str, Any] = {
                            "workspace": ws.get("gid"),
                            "limit": min(limit, 100),
                            "opt_fields": "name,owner,owner.name,current_status,current_status_update,due_on,start_on,created_at,modified_at,archived,public,team,team.name",
                        }

                        if since:
                            params["modified_since"] = since.isoformat()

                        projects_response = await self.http_client.get(
                            f"{self.BASE_URL}/projects",
                            headers=headers,
                            params=params,
                        )

                        if projects_response.status_code == 200:
                            projects_result = projects_response.json()
                            for p in projects_result.get("data", []):
                                all_projects.append({
                                    "id": p.get("gid"),
                                    "name": p.get("name"),
                                    "workspace_id": ws.get("gid"),
                                    "owner": p.get("owner", {}).get("name") if p.get("owner") else None,
                                    "team": p.get("team", {}).get("name") if p.get("team") else None,
                                    "due_on": p.get("due_on"),
                                    "start_on": p.get("start_on"),
                                    "created_at": p.get("created_at"),
                                    "modified_at": p.get("modified_at"),
                                    "archived": p.get("archived"),
                                    "public": p.get("public"),
                                    "status": p.get("current_status"),
                                })
                    except Exception as e:
                        warnings.append(f"Error fetching projects for workspace {ws.get('gid')}: {str(e)}")

                data["projects"] = all_projects
                total_items += len(all_projects)

            # Sync tasks
            if "tasks" in resources_to_sync:
                all_tasks = []
                projects_for_tasks = data.get("projects", [])[:10]  # Limit to first 10 projects

                for project in projects_for_tasks:
                    try:
                        params: dict[str, Any] = {
                            "project": project.get("id"),
                            "limit": min(limit // len(projects_for_tasks) if projects_for_tasks else limit, 50),
                            "opt_fields": "name,assignee,assignee.name,completed,completed_at,due_on,start_on,created_at,modified_at,notes,tags,tags.name,custom_fields",
                        }

                        if since:
                            params["modified_since"] = since.isoformat()

                        tasks_response = await self.http_client.get(
                            f"{self.BASE_URL}/tasks",
                            headers=headers,
                            params=params,
                        )

                        if tasks_response.status_code == 200:
                            tasks_result = tasks_response.json()
                            for t in tasks_result.get("data", []):
                                all_tasks.append({
                                    "id": t.get("gid"),
                                    "name": t.get("name"),
                                    "project_id": project.get("id"),
                                    "project_name": project.get("name"),
                                    "assignee": t.get("assignee", {}).get("name") if t.get("assignee") else None,
                                    "completed": t.get("completed"),
                                    "completed_at": t.get("completed_at"),
                                    "due_on": t.get("due_on"),
                                    "start_on": t.get("start_on"),
                                    "created_at": t.get("created_at"),
                                    "modified_at": t.get("modified_at"),
                                    "notes": t.get("notes", "")[:500] if t.get("notes") else None,
                                    "tags": [tag.get("name") for tag in t.get("tags", [])],
                                })
                    except Exception as e:
                        warnings.append(f"Error fetching tasks for project {project.get('id')}: {str(e)}")

                data["tasks"] = all_tasks
                total_items += len(all_tasks)

            # Sync tags
            if "tags" in resources_to_sync:
                all_tags = []
                for ws in target_workspaces[:5]:
                    try:
                        tags_response = await self.http_client.get(
                            f"{self.BASE_URL}/workspaces/{ws.get('gid')}/tags",
                            headers=headers,
                            params={"limit": min(limit, 100)},
                        )

                        if tags_response.status_code == 200:
                            tags_result = tags_response.json()
                            for tag in tags_result.get("data", []):
                                all_tags.append({
                                    "id": tag.get("gid"),
                                    "name": tag.get("name"),
                                    "workspace_id": ws.get("gid"),
                                    "color": tag.get("color"),
                                })
                    except Exception as e:
                        warnings.append(f"Error fetching tags for workspace {ws.get('gid')}: {str(e)}")

                data["tags"] = all_tags
                total_items += len(all_tags)

            # Sync users/team members
            if "users" in resources_to_sync:
                all_users = []
                for ws in target_workspaces[:5]:
                    try:
                        users_response = await self.http_client.get(
                            f"{self.BASE_URL}/workspaces/{ws.get('gid')}/users",
                            headers=headers,
                            params={
                                "limit": min(limit, 100),
                                "opt_fields": "name,email,photo,photo.image_128x128",
                            },
                        )

                        if users_response.status_code == 200:
                            users_result = users_response.json()
                            for user in users_result.get("data", []):
                                all_users.append({
                                    "id": user.get("gid"),
                                    "name": user.get("name"),
                                    "email": user.get("email"),
                                    "workspace_id": ws.get("gid"),
                                    "photo": user.get("photo", {}).get("image_128x128") if user.get("photo") else None,
                                })
                    except Exception as e:
                        warnings.append(f"Error fetching users for workspace {ws.get('gid')}: {str(e)}")

                data["users"] = all_users
                total_items += len(all_users)

            completed_at = datetime.utcnow()

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_items} items from Asana" if not errors else f"Partial sync: {total_items} items with {len(errors)} errors",
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
