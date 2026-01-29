"""Height integration plugin.

This plugin provides integration with Height's REST API for managing
workspaces, lists, and tasks.
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


class HeightPlugin(IntegrationPlugin):
    """Height integration plugin.

    Provides access to Height's REST API for managing:
    - Workspaces
    - Lists (projects/collections)
    - Tasks (issues/items)

    Authentication:
        Uses API Key authentication. Generate a key at:
        Height Settings > Integrations > API

    API Documentation:
        https://height.notion.site/API-documentation-643aea5bf01742de9232e5971cb4afda
    """

    BASE_URL = "https://api.height.app"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Height plugin metadata."""
        return IntegrationMetadata(
            slug="height",
            name="Height",
            description="The autonomous project collaboration tool for teams",
            category=IntegrationCategory.ISSUE_TRACKING,
            auth_type=AuthType.API_KEY,
            icon_url="https://height.app/favicon.ico",
            docs_url="https://height.notion.site/API-documentation-643aea5bf01742de9232e5971cb4afda",
            website_url="https://height.app",
            features=[
                "workspaces",
                "lists",
                "tasks",
                "users",
                "activities",
                "field_templates",
                "search",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key",
                    type="password",
                    required=True,
                    placeholder="secret_xxxxxxxxxxxxxxxxxxxxxxxx",
                    help_text="Generate at Height Settings > Integrations > API",
                ),
            ],
            optional_fields=[],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["workspaces", "lists", "tasks"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for Height API requests."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"api-key {credentials.api_key or ''}",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Height API.

        Uses the /users/me endpoint to verify the API key is valid
        and retrieve information about the authenticated user.

        Args:
            credentials: Credentials containing the API key.

        Returns:
            ConnectionTestResult with success status and account info.
        """
        start_time = time.time()

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/users/me",
                headers=self._get_headers(credentials),
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid API key",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="API key does not have required permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            if response.status_code == 429:
                return ConnectionTestResult(
                    success=False,
                    message="Rate limit exceeded. Please try again later.",
                    latency_ms=latency_ms,
                    error_code="RATE_LIMIT",
                )

            response.raise_for_status()
            data = response.json()

            return ConnectionTestResult(
                success=True,
                message=f"Connected as {data.get('username', 'Unknown')}",
                latency_ms=latency_ms,
                api_version="v1",
                permissions=[],
                account_info={
                    "user_id": data.get("id"),
                    "username": data.get("username"),
                    "email": data.get("email"),
                    "firstname": data.get("firstname"),
                    "lastname": data.get("lastname"),
                    "pictureUrl": data.get("pictureUrl"),
                    "state": data.get("state"),
                    "created_at": data.get("createdAt"),
                },
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="TIMEOUT",
            )
        except httpx.HTTPStatusError as e:
            return ConnectionTestResult(
                success=False,
                message=f"HTTP error: {e.response.status_code}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code=f"HTTP_{e.response.status_code}",
                error_details={"response": e.response.text[:500]},
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
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
        """Sync data from Height.

        Syncs workspaces, lists, and tasks from the workspace.

        Args:
            credentials: Credentials containing the API key.
            resources: Optional list of resources to sync (workspaces, lists, tasks).
            since: Optional datetime to sync data modified since.
            limit: Maximum number of items to sync per resource.

        Returns:
            SyncResult with synced data and statistics.
        """
        started_at = datetime.utcnow()
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        # Default to all resources if not specified
        resources_to_sync = resources or self.metadata.sync_resources

        headers = self._get_headers(credentials)

        try:
            # Sync workspaces
            if "workspaces" in resources_to_sync:
                workspaces_result = await self._sync_workspaces(headers, limit)
                if workspaces_result["success"]:
                    data["workspaces"] = workspaces_result["data"]
                    total_synced += len(workspaces_result["data"])
                else:
                    errors.append(workspaces_result["error"])

            # Sync lists
            if "lists" in resources_to_sync:
                lists_result = await self._sync_lists(headers, limit)
                if lists_result["success"]:
                    data["lists"] = lists_result["data"]
                    total_synced += len(lists_result["data"])
                else:
                    errors.append(lists_result["error"])

            # Sync tasks
            if "tasks" in resources_to_sync:
                tasks_result = await self._sync_tasks(headers, since, limit)
                if tasks_result["success"]:
                    data["tasks"] = tasks_result["data"]
                    total_synced += len(tasks_result["data"])
                else:
                    errors.append(tasks_result["error"])

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Height"
                if len(errors) == 0
                else f"Sync completed with {len(errors)} errors",
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                data=data,
            )

    async def _sync_workspaces(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync workspaces from Height."""
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/workspace",
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()

            # Height returns a single workspace for the API key
            workspace = result
            workspaces = [{
                "id": workspace.get("id"),
                "model": workspace.get("model"),
                "name": workspace.get("name"),
                "url": workspace.get("url"),
                "created_at": workspace.get("createdAt"),
            }]

            return {"success": True, "data": workspaces[:limit]}

        except Exception as e:
            return {"success": False, "error": f"Failed to sync workspaces: {str(e)}", "data": []}

    async def _sync_lists(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync lists from Height."""
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/lists",
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()

            lists = []
            for lst in result.get("list", [])[:limit]:
                lists.append({
                    "id": lst.get("id"),
                    "model": lst.get("model"),
                    "name": lst.get("name"),
                    "description": lst.get("description"),
                    "key": lst.get("key"),
                    "type": lst.get("type"),
                    "visualization": lst.get("visualization"),
                    "appearance": lst.get("appearance"),
                    "url": lst.get("url"),
                    "archived_at": lst.get("archivedAt"),
                    "created_at": lst.get("createdAt"),
                    "created_by_id": lst.get("createdUserId"),
                })

            return {"success": True, "data": lists}

        except Exception as e:
            return {"success": False, "error": f"Failed to sync lists: {str(e)}", "data": []}

    async def _sync_tasks(
        self,
        headers: dict[str, str],
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync tasks from Height using search endpoint."""
        try:
            # Build filters
            filters: dict[str, Any] = {
                "status": {"values": ["backLog", "inProgress"]},  # Active tasks
            }

            if since:
                filters["lastActivityAt"] = {
                    "gt": {"date": since.isoformat()},
                }

            payload = {
                "filters": filters,
                "limit": min(limit, 50),  # Height max per request
            }

            response = await self.http_client.post(
                f"{self.BASE_URL}/tasks/search",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            tasks = []
            for task in result.get("list", [])[:limit]:
                tasks.append({
                    "id": task.get("id"),
                    "model": task.get("model"),
                    "index": task.get("index"),
                    "name": task.get("name"),
                    "description": task.get("description"),
                    "status": task.get("status"),
                    "list_ids": task.get("listIds", []),
                    "parent_task_id": task.get("parentTaskId"),
                    "assignee_ids": task.get("assigneesIds", []),
                    "created_by_id": task.get("createdUserId"),
                    "fields": task.get("fields", []),
                    "url": task.get("url"),
                    "completed_at": task.get("completedAt"),
                    "deleted_at": task.get("deletedAt"),
                    "trashed_at": task.get("trashedAt"),
                    "created_at": task.get("createdAt"),
                    "last_activity_at": task.get("lastActivityAt"),
                })

            return {"success": True, "data": tasks}

        except Exception as e:
            return {"success": False, "error": f"Failed to sync tasks: {str(e)}", "data": []}
