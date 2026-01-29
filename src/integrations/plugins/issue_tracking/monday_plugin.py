"""Monday.com integration plugin.

This plugin provides integration with Monday.com's GraphQL API v2 for managing
boards, items, and groups.
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


class MondayPlugin(IntegrationPlugin):
    """Monday.com integration plugin.

    Provides access to Monday.com's GraphQL API v2 for managing:
    - Boards (projects/workspaces)
    - Items (tasks/records)
    - Groups (columns/categories within boards)

    Authentication:
        Uses API Token authentication. Generate a token at:
        https://monday.com/apps/manage -> Your Apps -> Create New App

    API Documentation:
        https://developer.monday.com/api-reference/docs
    """

    GRAPHQL_URL = "https://api.monday.com/v2"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Monday.com plugin metadata."""
        return IntegrationMetadata(
            slug="monday",
            name="Monday.com",
            description="Work OS platform for teams to manage projects and workflows",
            category=IntegrationCategory.ISSUE_TRACKING,
            auth_type=AuthType.API_KEY,
            icon_url="https://monday.com/favicon.ico",
            docs_url="https://developer.monday.com/api-reference/docs",
            website_url="https://monday.com",
            features=[
                "boards",
                "items",
                "groups",
                "columns",
                "subitems",
                "updates",
                "users",
                "workspaces",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Token",
                    type="password",
                    required=True,
                    placeholder="eyJhbGciOiJIUzI1NiJ9...",
                    help_text="Generate at monday.com Apps > Your Apps > Developer Token",
                ),
            ],
            optional_fields=[],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["boards", "items", "groups"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for Monday.com API requests."""
        return {
            "Content-Type": "application/json",
            "Authorization": credentials.api_key or "",
            "API-Version": "2024-01",
        }

    async def _execute_graphql(
        self,
        credentials: IntegrationCredentials,
        query: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a GraphQL query against Monday.com API.

        Args:
            credentials: API credentials.
            query: GraphQL query string.
            variables: Optional query variables.

        Returns:
            GraphQL response data.

        Raises:
            httpx.HTTPStatusError: On HTTP errors.
            ValueError: On GraphQL errors.
        """
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        response = await self.http_client.post(
            self.GRAPHQL_URL,
            headers=self._get_headers(credentials),
            json=payload,
        )
        response.raise_for_status()

        result = response.json()

        if "errors" in result:
            error_messages = [e.get("message", "Unknown error") for e in result["errors"]]
            raise ValueError(f"GraphQL errors: {'; '.join(error_messages)}")

        return result.get("data", {})

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Monday.com API.

        Uses the { me { id } } query to verify the API token is valid
        and retrieve information about the authenticated user.

        Args:
            credentials: Credentials containing the API token.

        Returns:
            ConnectionTestResult with success status and account info.
        """
        start_time = time.time()

        query = """
        query {
            me {
                id
                name
                email
                account {
                    id
                    name
                    plan {
                        tier
                    }
                }
            }
        }
        """

        try:
            data = await self._execute_graphql(credentials, query)
            latency_ms = (time.time() - start_time) * 1000

            me = data.get("me", {})
            account = me.get("account", {})

            return ConnectionTestResult(
                success=True,
                message=f"Connected as {me.get('name', 'Unknown')}",
                latency_ms=latency_ms,
                api_version="2024-01",
                permissions=[],  # Monday doesn't expose permissions via API
                account_info={
                    "user_id": me.get("id"),
                    "name": me.get("name"),
                    "email": me.get("email"),
                    "account_id": account.get("id"),
                    "account_name": account.get("name"),
                    "plan_tier": account.get("plan", {}).get("tier"),
                },
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000

            if e.response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid API token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            if e.response.status_code == 429:
                return ConnectionTestResult(
                    success=False,
                    message="Rate limit exceeded. Please try again later.",
                    latency_ms=latency_ms,
                    error_code="RATE_LIMIT",
                )

            return ConnectionTestResult(
                success=False,
                message=f"HTTP error: {e.response.status_code}",
                latency_ms=latency_ms,
                error_code=f"HTTP_{e.response.status_code}",
                error_details={"response": e.response.text[:500]},
            )

        except ValueError as e:
            return ConnectionTestResult(
                success=False,
                message=str(e),
                latency_ms=(time.time() - start_time) * 1000,
                error_code="GRAPHQL_ERROR",
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="TIMEOUT",
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
        """Sync data from Monday.com.

        Syncs boards, items, and groups from the workspace.

        Args:
            credentials: Credentials containing the API token.
            resources: Optional list of resources to sync (boards, items, groups).
            since: Optional datetime to sync data modified since (limited support).
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

        try:
            # Sync boards (required for items and groups)
            boards_data: list[dict[str, Any]] = []
            if "boards" in resources_to_sync or "items" in resources_to_sync or "groups" in resources_to_sync:
                boards_result = await self._sync_boards(credentials, limit)
                if boards_result["success"]:
                    boards_data = boards_result["data"]
                    if "boards" in resources_to_sync:
                        data["boards"] = boards_data
                        total_synced += len(boards_data)
                else:
                    errors.append(boards_result["error"])

            # Sync groups (from boards)
            if "groups" in resources_to_sync and boards_data:
                groups_result = await self._sync_groups(credentials, boards_data, limit)
                if groups_result["success"]:
                    data["groups"] = groups_result["data"]
                    total_synced += len(groups_result["data"])
                else:
                    errors.append(groups_result["error"])

            # Sync items
            if "items" in resources_to_sync and boards_data:
                items_result = await self._sync_items(credentials, boards_data, limit)
                if items_result["success"]:
                    data["items"] = items_result["data"]
                    total_synced += len(items_result["data"])
                else:
                    errors.append(items_result["error"])

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Monday.com"
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

    async def _sync_boards(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> dict[str, Any]:
        """Sync boards from Monday.com."""
        query = """
        query ($limit: Int!) {
            boards(limit: $limit) {
                id
                name
                description
                state
                board_kind
                workspace {
                    id
                    name
                }
                owner {
                    id
                    name
                }
                permissions
                columns {
                    id
                    title
                    type
                }
                groups {
                    id
                    title
                    color
                    position
                }
                items_count
                updated_at
            }
        }
        """

        try:
            data = await self._execute_graphql(
                credentials, query, {"limit": min(limit, 100)}
            )

            boards = []
            for board in data.get("boards", []):
                boards.append({
                    "id": board.get("id"),
                    "name": board.get("name"),
                    "description": board.get("description"),
                    "state": board.get("state"),
                    "board_kind": board.get("board_kind"),
                    "workspace_id": board.get("workspace", {}).get("id"),
                    "workspace_name": board.get("workspace", {}).get("name"),
                    "owner_id": board.get("owner", {}).get("id"),
                    "owner_name": board.get("owner", {}).get("name"),
                    "permissions": board.get("permissions"),
                    "columns": board.get("columns", []),
                    "groups": board.get("groups", []),
                    "items_count": board.get("items_count"),
                    "updated_at": board.get("updated_at"),
                })

            return {"success": True, "data": boards}

        except Exception as e:
            return {"success": False, "error": f"Failed to sync boards: {str(e)}", "data": []}

    async def _sync_groups(
        self,
        credentials: IntegrationCredentials,
        boards: list[dict[str, Any]],
        limit: int,
    ) -> dict[str, Any]:
        """Extract groups from boards data (already fetched with boards)."""
        try:
            groups = []
            for board in boards[:limit]:
                for group in board.get("groups", []):
                    groups.append({
                        "id": group.get("id"),
                        "title": group.get("title"),
                        "color": group.get("color"),
                        "position": group.get("position"),
                        "board_id": board.get("id"),
                        "board_name": board.get("name"),
                    })

            return {"success": True, "data": groups[:limit]}

        except Exception as e:
            return {"success": False, "error": f"Failed to sync groups: {str(e)}", "data": []}

    async def _sync_items(
        self,
        credentials: IntegrationCredentials,
        boards: list[dict[str, Any]],
        limit: int,
    ) -> dict[str, Any]:
        """Sync items from Monday.com boards."""
        query = """
        query ($board_id: ID!, $limit: Int!) {
            boards(ids: [$board_id]) {
                items_page(limit: $limit) {
                    items {
                        id
                        name
                        state
                        group {
                            id
                            title
                        }
                        column_values {
                            id
                            text
                            value
                            type
                        }
                        creator {
                            id
                            name
                        }
                        subscribers {
                            id
                            name
                        }
                        created_at
                        updated_at
                    }
                }
            }
        }
        """

        try:
            items = []
            items_per_board = max(1, limit // len(boards)) if boards else limit

            for board in boards:
                if len(items) >= limit:
                    break

                data = await self._execute_graphql(
                    credentials,
                    query,
                    {
                        "board_id": board.get("id"),
                        "limit": min(items_per_board, limit - len(items)),
                    },
                )

                board_data = data.get("boards", [{}])[0]
                items_page = board_data.get("items_page", {})

                for item in items_page.get("items", []):
                    items.append({
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "state": item.get("state"),
                        "board_id": board.get("id"),
                        "board_name": board.get("name"),
                        "group_id": item.get("group", {}).get("id"),
                        "group_title": item.get("group", {}).get("title"),
                        "column_values": item.get("column_values", []),
                        "creator_id": item.get("creator", {}).get("id"),
                        "creator_name": item.get("creator", {}).get("name"),
                        "subscribers": [
                            {"id": s.get("id"), "name": s.get("name")}
                            for s in item.get("subscribers", [])
                        ],
                        "created_at": item.get("created_at"),
                        "updated_at": item.get("updated_at"),
                    })

            return {"success": True, "data": items[:limit]}

        except Exception as e:
            return {"success": False, "error": f"Failed to sync items: {str(e)}", "data": []}
