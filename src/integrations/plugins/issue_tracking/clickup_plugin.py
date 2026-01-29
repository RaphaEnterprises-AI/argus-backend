"""ClickUp integration plugin.

ClickUp REST API v2 with Personal API Token authentication.
API Documentation: https://clickup.com/api/
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


class ClickUpPlugin(IntegrationPlugin):
    """ClickUp integration for project management and task tracking.

    ClickUp provides an all-in-one productivity platform with tasks, docs,
    goals, whiteboards, and more.

    Uses Personal API Token authentication.
    API Documentation: https://clickup.com/api/
    """

    BASE_URL = "https://api.clickup.com/api/v2"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return ClickUp integration metadata."""
        return IntegrationMetadata(
            slug="clickup",
            name="ClickUp",
            description="Connect to ClickUp for comprehensive project management with tasks, docs, goals, and whiteboards. Organize work across spaces, folders, and lists.",
            category=IntegrationCategory.ISSUE_TRACKING,
            auth_type=AuthType.API_KEY,
            icon_url="https://clickup.com/landing/images/clickup-favicon.png",
            docs_url="https://clickup.com/api/",
            website_url="https://clickup.com",
            features=[
                "tasks",
                "spaces",
                "folders",
                "lists",
                "goals",
                "docs",
                "whiteboards",
                "time_tracking",
                "custom_fields",
                "comments",
                "webhooks",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="Personal API Token",
                    type="password",
                    required=True,
                    placeholder="pk_...",
                    help_text="Generate at ClickUp Settings > Apps > API Token",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="workspace_id",
                    label="Workspace ID",
                    type="text",
                    required=False,
                    placeholder="12345678",
                    help_text="Optional: Limit sync to a specific workspace (Team ID)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["workspaces", "spaces", "folders", "lists", "tasks", "goals", "members"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get authentication headers for API requests."""
        return {
            "Authorization": credentials.api_key,
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to ClickUp API.

        Uses GET /user to verify credentials.
        """
        start_time = time.time()

        try:
            headers = self._get_headers(credentials)

            # Test connection with /user endpoint
            response = await self.http_client.get(
                f"{self.BASE_URL}/user",
                headers=headers,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                user_data = result.get("user", {})

                # Get teams/workspaces for permissions context
                teams_response = await self.http_client.get(
                    f"{self.BASE_URL}/team",
                    headers=headers,
                )
                teams = []
                if teams_response.status_code == 200:
                    teams_result = teams_response.json()
                    teams = teams_result.get("teams", [])

                permissions = [
                    "read:user",
                    "read:teams",
                    "read:spaces",
                    "read:tasks",
                    f"access:{len(teams)}_workspaces",
                ]

                return ConnectionTestResult(
                    success=True,
                    message=f"Connected as {user_data.get('username', user_data.get('email', 'Unknown'))}",
                    latency_ms=latency_ms,
                    api_version="v2",
                    permissions=permissions,
                    account_info={
                        "user_id": user_data.get("id"),
                        "username": user_data.get("username"),
                        "email": user_data.get("email"),
                        "color": user_data.get("color"),
                        "profile_picture": user_data.get("profilePicture"),
                        "initials": user_data.get("initials"),
                        "workspaces": [
                            {"id": t.get("id"), "name": t.get("name")}
                            for t in teams[:10]
                        ],
                    },
                )

            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed. Check your API token.",
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
                        "error": error_data.get("err"),
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
        """Sync data from ClickUp.

        Syncs workspaces (teams), spaces, folders, lists, and tasks.
        """
        started_at = datetime.utcnow()
        sync_id = f"clickup_sync_{int(started_at.timestamp())}"

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_items = 0

        # Default to all resources if none specified
        resources_to_sync = resources or ["workspaces", "spaces", "lists", "tasks"]

        try:
            headers = self._get_headers(credentials)

            # Sync workspaces (teams)
            workspaces = []
            if "workspaces" in resources_to_sync or credentials.workspace_id is None:
                try:
                    teams_response = await self.http_client.get(
                        f"{self.BASE_URL}/team",
                        headers=headers,
                    )

                    if teams_response.status_code == 200:
                        teams_result = teams_response.json()
                        workspaces = teams_result.get("teams", [])
                        data["workspaces"] = [
                            {
                                "id": ws.get("id"),
                                "name": ws.get("name"),
                                "color": ws.get("color"),
                                "avatar": ws.get("avatar"),
                                "members_count": len(ws.get("members", [])),
                            }
                            for ws in workspaces
                        ]
                        total_items += len(workspaces)
                    else:
                        errors.append(f"Failed to fetch workspaces: {teams_response.status_code}")
                except Exception as e:
                    errors.append(f"Error fetching workspaces: {str(e)}")

            # Filter to specific workspace if provided
            target_workspaces = workspaces
            if credentials.workspace_id:
                target_workspaces = [{"id": credentials.workspace_id}]

            # Sync spaces
            all_spaces = []
            if "spaces" in resources_to_sync:
                for ws in target_workspaces[:5]:  # Limit to first 5 workspaces
                    try:
                        spaces_response = await self.http_client.get(
                            f"{self.BASE_URL}/team/{ws.get('id')}/space",
                            headers=headers,
                            params={"archived": "false"},
                        )

                        if spaces_response.status_code == 200:
                            spaces_result = spaces_response.json()
                            for space in spaces_result.get("spaces", []):
                                all_spaces.append({
                                    "id": space.get("id"),
                                    "name": space.get("name"),
                                    "workspace_id": ws.get("id"),
                                    "color": space.get("color"),
                                    "private": space.get("private"),
                                    "archived": space.get("archived"),
                                    "statuses": [
                                        {"name": s.get("status"), "type": s.get("type"), "color": s.get("color")}
                                        for s in space.get("statuses", [])
                                    ],
                                })
                    except Exception as e:
                        warnings.append(f"Error fetching spaces for workspace {ws.get('id')}: {str(e)}")

                data["spaces"] = all_spaces
                total_items += len(all_spaces)

            # Sync folders
            all_folders = []
            if "folders" in resources_to_sync:
                for space in all_spaces[:10]:  # Limit to first 10 spaces
                    try:
                        folders_response = await self.http_client.get(
                            f"{self.BASE_URL}/space/{space.get('id')}/folder",
                            headers=headers,
                            params={"archived": "false"},
                        )

                        if folders_response.status_code == 200:
                            folders_result = folders_response.json()
                            for folder in folders_result.get("folders", []):
                                all_folders.append({
                                    "id": folder.get("id"),
                                    "name": folder.get("name"),
                                    "space_id": space.get("id"),
                                    "space_name": space.get("name"),
                                    "hidden": folder.get("hidden"),
                                    "lists_count": len(folder.get("lists", [])),
                                })
                    except Exception as e:
                        warnings.append(f"Error fetching folders for space {space.get('id')}: {str(e)}")

                data["folders"] = all_folders
                total_items += len(all_folders)

            # Sync lists
            all_lists = []
            if "lists" in resources_to_sync:
                # Get folderless lists from spaces
                for space in all_spaces[:10]:
                    try:
                        lists_response = await self.http_client.get(
                            f"{self.BASE_URL}/space/{space.get('id')}/list",
                            headers=headers,
                            params={"archived": "false"},
                        )

                        if lists_response.status_code == 200:
                            lists_result = lists_response.json()
                            for lst in lists_result.get("lists", []):
                                all_lists.append({
                                    "id": lst.get("id"),
                                    "name": lst.get("name"),
                                    "space_id": space.get("id"),
                                    "space_name": space.get("name"),
                                    "folder_id": None,
                                    "archived": lst.get("archived"),
                                    "task_count": lst.get("task_count"),
                                    "due_date": lst.get("due_date"),
                                    "start_date": lst.get("start_date"),
                                    "status": lst.get("status", {}).get("status") if lst.get("status") else None,
                                })
                    except Exception as e:
                        warnings.append(f"Error fetching lists for space {space.get('id')}: {str(e)}")

                # Get lists from folders
                for folder in all_folders[:10]:
                    try:
                        folder_lists_response = await self.http_client.get(
                            f"{self.BASE_URL}/folder/{folder.get('id')}/list",
                            headers=headers,
                            params={"archived": "false"},
                        )

                        if folder_lists_response.status_code == 200:
                            folder_lists_result = folder_lists_response.json()
                            for lst in folder_lists_result.get("lists", []):
                                all_lists.append({
                                    "id": lst.get("id"),
                                    "name": lst.get("name"),
                                    "space_id": folder.get("space_id"),
                                    "space_name": folder.get("space_name"),
                                    "folder_id": folder.get("id"),
                                    "folder_name": folder.get("name"),
                                    "archived": lst.get("archived"),
                                    "task_count": lst.get("task_count"),
                                    "due_date": lst.get("due_date"),
                                    "start_date": lst.get("start_date"),
                                    "status": lst.get("status", {}).get("status") if lst.get("status") else None,
                                })
                    except Exception as e:
                        warnings.append(f"Error fetching lists for folder {folder.get('id')}: {str(e)}")

                data["lists"] = all_lists
                total_items += len(all_lists)

            # Sync tasks
            if "tasks" in resources_to_sync:
                all_tasks = []
                lists_for_tasks = all_lists[:10]  # Limit to first 10 lists

                for lst in lists_for_tasks:
                    try:
                        params: dict[str, Any] = {
                            "archived": "false",
                            "page": 0,
                            "subtasks": "true",
                            "include_closed": "true",
                        }

                        # ClickUp uses Unix timestamps in milliseconds for date_updated_gt
                        if since:
                            params["date_updated_gt"] = int(since.timestamp() * 1000)

                        tasks_response = await self.http_client.get(
                            f"{self.BASE_URL}/list/{lst.get('id')}/task",
                            headers=headers,
                            params=params,
                        )

                        if tasks_response.status_code == 200:
                            tasks_result = tasks_response.json()
                            for task in tasks_result.get("tasks", [])[:limit // len(lists_for_tasks) if lists_for_tasks else limit]:
                                all_tasks.append({
                                    "id": task.get("id"),
                                    "custom_id": task.get("custom_id"),
                                    "name": task.get("name"),
                                    "description": task.get("description", "")[:500] if task.get("description") else None,
                                    "status": task.get("status", {}).get("status") if task.get("status") else None,
                                    "status_type": task.get("status", {}).get("type") if task.get("status") else None,
                                    "priority": task.get("priority", {}).get("priority") if task.get("priority") else None,
                                    "list_id": lst.get("id"),
                                    "list_name": lst.get("name"),
                                    "space_id": lst.get("space_id"),
                                    "space_name": lst.get("space_name"),
                                    "folder_id": lst.get("folder_id"),
                                    "assignees": [
                                        {"id": a.get("id"), "username": a.get("username"), "email": a.get("email")}
                                        for a in task.get("assignees", [])
                                    ],
                                    "creator": {
                                        "id": task.get("creator", {}).get("id"),
                                        "username": task.get("creator", {}).get("username"),
                                    } if task.get("creator") else None,
                                    "tags": [
                                        {"name": t.get("name"), "tag_bg": t.get("tag_bg")}
                                        for t in task.get("tags", [])
                                    ],
                                    "due_date": task.get("due_date"),
                                    "start_date": task.get("start_date"),
                                    "time_estimate": task.get("time_estimate"),
                                    "time_spent": task.get("time_spent"),
                                    "created_at": task.get("date_created"),
                                    "updated_at": task.get("date_updated"),
                                    "closed_at": task.get("date_closed"),
                                    "url": task.get("url"),
                                })
                    except Exception as e:
                        warnings.append(f"Error fetching tasks for list {lst.get('id')}: {str(e)}")

                data["tasks"] = all_tasks
                total_items += len(all_tasks)

            # Sync goals
            if "goals" in resources_to_sync:
                all_goals = []
                for ws in target_workspaces[:5]:
                    try:
                        goals_response = await self.http_client.get(
                            f"{self.BASE_URL}/team/{ws.get('id')}/goal",
                            headers=headers,
                        )

                        if goals_response.status_code == 200:
                            goals_result = goals_response.json()
                            for goal in goals_result.get("goals", []):
                                all_goals.append({
                                    "id": goal.get("id"),
                                    "name": goal.get("name"),
                                    "workspace_id": ws.get("id"),
                                    "color": goal.get("color"),
                                    "due_date": goal.get("due_date"),
                                    "date_created": goal.get("date_created"),
                                    "creator": goal.get("creator"),
                                    "owners": [
                                        {"id": o.get("id"), "username": o.get("username")}
                                        for o in goal.get("owners", [])
                                    ],
                                    "percent_completed": goal.get("percent_completed"),
                                    "key_result_count": goal.get("key_result_count"),
                                })
                    except Exception as e:
                        warnings.append(f"Error fetching goals for workspace {ws.get('id')}: {str(e)}")

                data["goals"] = all_goals
                total_items += len(all_goals)

            # Sync members
            if "members" in resources_to_sync:
                all_members = []
                seen_members = set()
                for ws in target_workspaces[:5]:
                    # Members are included in the team response
                    for workspace in workspaces:
                        if workspace.get("id") == ws.get("id"):
                            for member in workspace.get("members", []):
                                user = member.get("user", {})
                                if user.get("id") not in seen_members:
                                    seen_members.add(user.get("id"))
                                    all_members.append({
                                        "id": user.get("id"),
                                        "username": user.get("username"),
                                        "email": user.get("email"),
                                        "color": user.get("color"),
                                        "profile_picture": user.get("profilePicture"),
                                        "initials": user.get("initials"),
                                        "workspace_id": ws.get("id"),
                                        "role": member.get("role"),
                                    })

                data["members"] = all_members
                total_items += len(all_members)

            completed_at = datetime.utcnow()

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_items} items from ClickUp" if not errors else f"Partial sync: {total_items} items with {len(errors)} errors",
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
