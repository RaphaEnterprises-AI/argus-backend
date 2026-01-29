"""Shortcut (formerly Clubhouse) integration plugin.

This plugin provides integration with Shortcut's REST API for managing
stories, epics, and iterations.
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


class ShortcutPlugin(IntegrationPlugin):
    """Shortcut (formerly Clubhouse) integration plugin.

    Provides access to Shortcut's REST API for managing:
    - Stories (tasks/issues)
    - Epics (groups of stories)
    - Iterations (sprints)

    Authentication:
        Uses API Token authentication. Generate a token at:
        https://app.shortcut.com/settings/account/api-tokens

    API Documentation:
        https://developer.shortcut.com/api/rest/v3
    """

    BASE_URL = "https://api.app.shortcut.com/api/v3"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Shortcut plugin metadata."""
        return IntegrationMetadata(
            slug="shortcut",
            name="Shortcut",
            description="Project management for software teams (formerly Clubhouse)",
            category=IntegrationCategory.ISSUE_TRACKING,
            auth_type=AuthType.API_KEY,
            icon_url="https://www.shortcut.com/favicon.ico",
            docs_url="https://developer.shortcut.com/api/rest/v3",
            website_url="https://www.shortcut.com",
            features=[
                "stories",
                "epics",
                "iterations",
                "projects",
                "workflows",
                "labels",
                "members",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Token",
                    type="password",
                    required=True,
                    placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    help_text="Generate at Shortcut Settings > API Tokens",
                ),
            ],
            optional_fields=[],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["stories", "epics", "iterations"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for Shortcut API requests."""
        return {
            "Content-Type": "application/json",
            "Shortcut-Token": credentials.api_key or "",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Shortcut API.

        Uses the /member endpoint to verify the API token is valid
        and retrieve information about the authenticated user.

        Args:
            credentials: Credentials containing the API token.

        Returns:
            ConnectionTestResult with success status and account info.
        """
        start_time = time.time()

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/member",
                headers=self._get_headers(credentials),
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid API token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
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
                message=f"Connected as {data.get('profile', {}).get('name', 'Unknown')}",
                latency_ms=latency_ms,
                api_version="v3",
                permissions=data.get("role", "member").split(","),
                account_info={
                    "member_id": data.get("id"),
                    "name": data.get("profile", {}).get("name"),
                    "email": data.get("profile", {}).get("email_address"),
                    "role": data.get("role"),
                    "workspace": data.get("workspace2", {}).get("name"),
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
        """Sync data from Shortcut.

        Syncs stories, epics, and iterations from the workspace.

        Args:
            credentials: Credentials containing the API token.
            resources: Optional list of resources to sync (stories, epics, iterations).
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
            # Sync stories
            if "stories" in resources_to_sync:
                stories_result = await self._sync_stories(
                    headers, since, limit
                )
                if stories_result["success"]:
                    data["stories"] = stories_result["data"]
                    total_synced += len(stories_result["data"])
                else:
                    errors.append(stories_result["error"])

            # Sync epics
            if "epics" in resources_to_sync:
                epics_result = await self._sync_epics(headers, limit)
                if epics_result["success"]:
                    data["epics"] = epics_result["data"]
                    total_synced += len(epics_result["data"])
                else:
                    errors.append(epics_result["error"])

            # Sync iterations
            if "iterations" in resources_to_sync:
                iterations_result = await self._sync_iterations(headers, limit)
                if iterations_result["success"]:
                    data["iterations"] = iterations_result["data"]
                    total_synced += len(iterations_result["data"])
                else:
                    errors.append(iterations_result["error"])

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Shortcut"
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

    async def _sync_stories(
        self,
        headers: dict[str, str],
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync stories from Shortcut using search endpoint."""
        try:
            # Build search query
            query = "!is:archived"
            if since:
                query += f" updated:>{since.strftime('%Y-%m-%d')}"

            response = await self.http_client.get(
                f"{self.BASE_URL}/search/stories",
                headers=headers,
                params={
                    "query": query,
                    "page_size": min(limit, 25),  # Shortcut max is 25
                },
            )
            response.raise_for_status()
            result = response.json()

            stories = []
            for story in result.get("data", [])[:limit]:
                stories.append({
                    "id": story.get("id"),
                    "name": story.get("name"),
                    "description": story.get("description"),
                    "story_type": story.get("story_type"),
                    "state": story.get("workflow_state_id"),
                    "epic_id": story.get("epic_id"),
                    "iteration_id": story.get("iteration_id"),
                    "estimate": story.get("estimate"),
                    "labels": [l.get("name") for l in story.get("labels", [])],
                    "owners": story.get("owner_ids", []),
                    "created_at": story.get("created_at"),
                    "updated_at": story.get("updated_at"),
                    "completed_at": story.get("completed_at"),
                    "external_id": story.get("external_id"),
                    "app_url": story.get("app_url"),
                })

            return {"success": True, "data": stories}

        except Exception as e:
            return {"success": False, "error": f"Failed to sync stories: {str(e)}", "data": []}

    async def _sync_epics(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync epics from Shortcut."""
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/epics",
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()

            epics = []
            for epic in result[:limit]:
                epics.append({
                    "id": epic.get("id"),
                    "name": epic.get("name"),
                    "description": epic.get("description"),
                    "state": epic.get("state"),
                    "milestone_id": epic.get("milestone_id"),
                    "planned_start_date": epic.get("planned_start_date"),
                    "deadline": epic.get("deadline"),
                    "labels": [l.get("name") for l in epic.get("labels", [])],
                    "owners": epic.get("owner_ids", []),
                    "created_at": epic.get("created_at"),
                    "updated_at": epic.get("updated_at"),
                    "completed_at": epic.get("completed_at"),
                    "stats": epic.get("stats", {}),
                    "app_url": epic.get("app_url"),
                })

            return {"success": True, "data": epics}

        except Exception as e:
            return {"success": False, "error": f"Failed to sync epics: {str(e)}", "data": []}

    async def _sync_iterations(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync iterations (sprints) from Shortcut."""
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/iterations",
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()

            iterations = []
            for iteration in result[:limit]:
                iterations.append({
                    "id": iteration.get("id"),
                    "name": iteration.get("name"),
                    "description": iteration.get("description"),
                    "status": iteration.get("status"),
                    "start_date": iteration.get("start_date"),
                    "end_date": iteration.get("end_date"),
                    "labels": [l.get("name") for l in iteration.get("labels", [])],
                    "created_at": iteration.get("created_at"),
                    "updated_at": iteration.get("updated_at"),
                    "stats": iteration.get("stats", {}),
                    "app_url": iteration.get("app_url"),
                })

            return {"success": True, "data": iterations}

        except Exception as e:
            return {"success": False, "error": f"Failed to sync iterations: {str(e)}", "data": []}
