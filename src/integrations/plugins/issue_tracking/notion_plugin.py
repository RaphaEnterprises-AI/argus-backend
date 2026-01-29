"""Notion integration plugin.

This plugin provides integration with Notion's REST API for managing
databases and pages.
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


class NotionPlugin(IntegrationPlugin):
    """Notion integration plugin.

    Provides access to Notion's REST API for managing:
    - Databases (structured collections)
    - Pages (documents/wiki)

    Authentication:
        Uses Internal Integration Token. Create an integration at:
        https://www.notion.so/my-integrations

    API Documentation:
        https://developers.notion.com/reference
    """

    BASE_URL = "https://api.notion.com/v1"
    API_VERSION = "2022-06-28"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Notion plugin metadata."""
        return IntegrationMetadata(
            slug="notion",
            name="Notion",
            description="All-in-one workspace for notes, docs, and project management",
            category=IntegrationCategory.ISSUE_TRACKING,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://www.notion.so/images/favicon.ico",
            docs_url="https://developers.notion.com/reference",
            website_url="https://www.notion.so",
            features=[
                "databases",
                "pages",
                "blocks",
                "users",
                "search",
                "comments",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Internal Integration Token",
                    type="password",
                    required=True,
                    placeholder="secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    help_text="Create an integration at notion.so/my-integrations",
                ),
            ],
            optional_fields=[],
            supports_webhooks=False,  # Notion uses polling
            supports_sync=True,
            sync_resources=["databases", "pages"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for Notion API requests."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credentials.access_token or ''}",
            "Notion-Version": self.API_VERSION,
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Notion API.

        Uses the /users/me endpoint to verify the integration token is valid
        and retrieve information about the bot user.

        Args:
            credentials: Credentials containing the integration token.

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
                    message="Invalid integration token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Integration does not have required permissions",
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

            bot_info = data.get("bot", {})
            owner = bot_info.get("owner", {})

            return ConnectionTestResult(
                success=True,
                message=f"Connected as {data.get('name', 'Bot')}",
                latency_ms=latency_ms,
                api_version=self.API_VERSION,
                permissions=[],  # Notion scopes are set at integration creation
                account_info={
                    "bot_id": data.get("id"),
                    "name": data.get("name"),
                    "type": data.get("type"),
                    "avatar_url": data.get("avatar_url"),
                    "owner_type": owner.get("type"),
                    "workspace_name": owner.get("workspace", {}).get("name")
                    if owner.get("type") == "workspace"
                    else None,
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
        """Sync data from Notion.

        Syncs databases and pages from the workspace.

        Args:
            credentials: Credentials containing the integration token.
            resources: Optional list of resources to sync (databases, pages).
            since: Optional datetime to filter by last edited time.
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
            # Sync databases
            if "databases" in resources_to_sync:
                databases_result = await self._sync_databases(
                    credentials, since, limit
                )
                if databases_result["success"]:
                    data["databases"] = databases_result["data"]
                    total_synced += len(databases_result["data"])
                else:
                    errors.append(databases_result["error"])

            # Sync pages
            if "pages" in resources_to_sync:
                pages_result = await self._sync_pages(credentials, since, limit)
                if pages_result["success"]:
                    data["pages"] = pages_result["data"]
                    total_synced += len(pages_result["data"])
                else:
                    errors.append(pages_result["error"])

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Notion"
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

    async def _sync_databases(
        self,
        credentials: IntegrationCredentials,
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync databases from Notion using search endpoint."""
        try:
            payload: dict[str, Any] = {
                "filter": {"property": "object", "value": "database"},
                "page_size": min(limit, 100),  # Notion max is 100
            }

            # Add filter for last edited time if specified
            if since:
                payload["filter"] = {
                    "and": [
                        {"property": "object", "value": "database"},
                        {
                            "timestamp": "last_edited_time",
                            "last_edited_time": {"after": since.isoformat()},
                        },
                    ]
                }

            response = await self.http_client.post(
                f"{self.BASE_URL}/search",
                headers=self._get_headers(credentials),
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            databases = []
            for db in result.get("results", [])[:limit]:
                # Extract title from title property
                title_parts = db.get("title", [])
                title = "".join(
                    part.get("plain_text", "") for part in title_parts
                )

                # Extract properties schema
                properties = {}
                for prop_name, prop_data in db.get("properties", {}).items():
                    properties[prop_name] = {
                        "id": prop_data.get("id"),
                        "type": prop_data.get("type"),
                        "name": prop_data.get("name", prop_name),
                    }

                databases.append({
                    "id": db.get("id"),
                    "title": title,
                    "description": self._extract_rich_text(db.get("description", [])),
                    "icon": self._extract_icon(db.get("icon")),
                    "cover": db.get("cover"),
                    "properties": properties,
                    "parent": db.get("parent"),
                    "url": db.get("url"),
                    "public_url": db.get("public_url"),
                    "archived": db.get("archived", False),
                    "is_inline": db.get("is_inline", False),
                    "created_time": db.get("created_time"),
                    "last_edited_time": db.get("last_edited_time"),
                    "created_by": db.get("created_by", {}).get("id"),
                    "last_edited_by": db.get("last_edited_by", {}).get("id"),
                })

            return {"success": True, "data": databases}

        except Exception as e:
            return {"success": False, "error": f"Failed to sync databases: {str(e)}", "data": []}

    async def _sync_pages(
        self,
        credentials: IntegrationCredentials,
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync pages from Notion using search endpoint."""
        try:
            payload: dict[str, Any] = {
                "filter": {"property": "object", "value": "page"},
                "page_size": min(limit, 100),  # Notion max is 100
            }

            # Add filter for last edited time if specified
            if since:
                payload["filter"] = {
                    "and": [
                        {"property": "object", "value": "page"},
                        {
                            "timestamp": "last_edited_time",
                            "last_edited_time": {"after": since.isoformat()},
                        },
                    ]
                }

            response = await self.http_client.post(
                f"{self.BASE_URL}/search",
                headers=self._get_headers(credentials),
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            pages = []
            for page in result.get("results", [])[:limit]:
                # Extract title from properties (usually "title" or "Name")
                title = self._extract_page_title(page.get("properties", {}))

                # Extract property values
                property_values = {}
                for prop_name, prop_data in page.get("properties", {}).items():
                    property_values[prop_name] = self._extract_property_value(prop_data)

                pages.append({
                    "id": page.get("id"),
                    "title": title,
                    "icon": self._extract_icon(page.get("icon")),
                    "cover": page.get("cover"),
                    "properties": property_values,
                    "parent": page.get("parent"),
                    "url": page.get("url"),
                    "public_url": page.get("public_url"),
                    "archived": page.get("archived", False),
                    "created_time": page.get("created_time"),
                    "last_edited_time": page.get("last_edited_time"),
                    "created_by": page.get("created_by", {}).get("id"),
                    "last_edited_by": page.get("last_edited_by", {}).get("id"),
                })

            return {"success": True, "data": pages}

        except Exception as e:
            return {"success": False, "error": f"Failed to sync pages: {str(e)}", "data": []}

    def _extract_rich_text(self, rich_text_array: list[dict[str, Any]]) -> str:
        """Extract plain text from Notion rich text array."""
        return "".join(
            part.get("plain_text", "") for part in rich_text_array
        )

    def _extract_icon(self, icon: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Extract icon information."""
        if not icon:
            return None

        icon_type = icon.get("type")
        if icon_type == "emoji":
            return {"type": "emoji", "emoji": icon.get("emoji")}
        elif icon_type == "external":
            return {"type": "external", "url": icon.get("external", {}).get("url")}
        elif icon_type == "file":
            return {"type": "file", "url": icon.get("file", {}).get("url")}
        return None

    def _extract_page_title(self, properties: dict[str, Any]) -> str:
        """Extract page title from properties."""
        # Try common title property names
        for prop_name in ["title", "Title", "Name", "name"]:
            if prop_name in properties:
                prop = properties[prop_name]
                if prop.get("type") == "title":
                    return self._extract_rich_text(prop.get("title", []))

        # Fallback: look for any title-type property
        for prop in properties.values():
            if prop.get("type") == "title":
                return self._extract_rich_text(prop.get("title", []))

        return "Untitled"

    def _extract_property_value(self, prop: dict[str, Any]) -> Any:
        """Extract value from a Notion property."""
        prop_type = prop.get("type")

        if prop_type == "title":
            return self._extract_rich_text(prop.get("title", []))
        elif prop_type == "rich_text":
            return self._extract_rich_text(prop.get("rich_text", []))
        elif prop_type == "number":
            return prop.get("number")
        elif prop_type == "select":
            select = prop.get("select")
            return select.get("name") if select else None
        elif prop_type == "multi_select":
            return [s.get("name") for s in prop.get("multi_select", [])]
        elif prop_type == "date":
            date = prop.get("date")
            if date:
                return {
                    "start": date.get("start"),
                    "end": date.get("end"),
                    "time_zone": date.get("time_zone"),
                }
            return None
        elif prop_type == "people":
            return [p.get("id") for p in prop.get("people", [])]
        elif prop_type == "files":
            files = []
            for f in prop.get("files", []):
                file_type = f.get("type")
                if file_type == "external":
                    files.append({"name": f.get("name"), "url": f.get("external", {}).get("url")})
                elif file_type == "file":
                    files.append({"name": f.get("name"), "url": f.get("file", {}).get("url")})
            return files
        elif prop_type == "checkbox":
            return prop.get("checkbox")
        elif prop_type == "url":
            return prop.get("url")
        elif prop_type == "email":
            return prop.get("email")
        elif prop_type == "phone_number":
            return prop.get("phone_number")
        elif prop_type == "formula":
            formula = prop.get("formula", {})
            formula_type = formula.get("type")
            return formula.get(formula_type)
        elif prop_type == "relation":
            return [r.get("id") for r in prop.get("relation", [])]
        elif prop_type == "rollup":
            rollup = prop.get("rollup", {})
            rollup_type = rollup.get("type")
            if rollup_type == "array":
                return [self._extract_property_value(item) for item in rollup.get("array", [])]
            return rollup.get(rollup_type)
        elif prop_type == "status":
            status = prop.get("status")
            return status.get("name") if status else None
        elif prop_type == "created_time":
            return prop.get("created_time")
        elif prop_type == "created_by":
            return prop.get("created_by", {}).get("id")
        elif prop_type == "last_edited_time":
            return prop.get("last_edited_time")
        elif prop_type == "last_edited_by":
            return prop.get("last_edited_by", {}).get("id")

        return None
