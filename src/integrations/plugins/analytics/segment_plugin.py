"""Segment Analytics integration plugin.

Segment is a customer data platform (CDP) that collects, cleans, and
routes customer data to hundreds of tools. This plugin provides access
to sources, destinations, tracking plans, and event sending.

API Docs: https://segment.com/docs/api/
Config API: https://segment.com/docs/api/config-api/
Tracking API: https://segment.com/docs/connections/sources/catalog/libraries/server/http-api/
"""

import base64
import json
import time
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

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


class SegmentPlugin(IntegrationPlugin):
    """Segment Analytics integration plugin.

    Provides access to:
    - Sources (data collection endpoints)
    - Destinations (where data is sent)
    - Tracking Plans (schema definitions)
    - Event tracking via HTTP API

    Authentication:
    - Write Key for tracking data (from source settings)
    - Access Token for Config API (workspace-level, optional)

    Usage:
        plugin = SegmentPlugin()
        credentials = IntegrationCredentials(
            api_key="your_write_key",
            access_token="your_access_token"  # Optional, for Config API
        )

        # Test connection
        result = await plugin.test_connection(credentials)
        if result.success:
            # Sync sources, destinations, and tracking plans
            sync_result = await plugin.sync_data(credentials)

            # Track an event
            await plugin.track(credentials, "Page Viewed", {"url": "/home"}, "user123")
    """

    # Segment API endpoints
    TRACKING_API_URL = "https://api.segment.io/v1"
    CONFIG_API_URL = "https://api.segmentapis.com"

    # Write key format validation (base64-encoded)
    WRITE_KEY_MIN_LENGTH = 20

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Segment plugin metadata."""
        return IntegrationMetadata(
            slug="segment",
            name="Segment",
            description="Customer data platform for collecting and routing analytics data",
            category=IntegrationCategory.ANALYTICS,
            auth_type=AuthType.API_KEY,
            icon_url="https://segment.com/favicon.ico",
            docs_url="https://segment.com/docs/api/",
            website_url="https://segment.com",
            features=[
                "Event tracking and collection",
                "Source management",
                "Destination routing",
                "Tracking plan enforcement",
                "Data schema validation",
                "Real-time event streaming",
                "Customer data unification",
                "Privacy controls (GDPR/CCPA)",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="Write Key",
                    type="password",
                    required=True,
                    placeholder="Enter your Segment Write Key",
                    help_text="Found in Segment Source Settings > API Keys",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Access Token",
                    type="password",
                    required=False,
                    placeholder="Enter your Segment Access Token",
                    help_text="Required for Config API access. Found in Workspace Settings > Tokens.",
                ),
                FieldDefinition(
                    name="extra.workspace_slug",
                    label="Workspace Slug",
                    type="text",
                    required=False,
                    placeholder="your-workspace",
                    help_text="Workspace slug for Config API calls",
                ),
                FieldDefinition(
                    name="extra.source_slug",
                    label="Source Slug",
                    type="text",
                    required=False,
                    placeholder="your-source",
                    help_text="Source slug for source-specific operations",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["sources", "destinations", "tracking_plans", "functions"],
        )

    def _get_tracking_auth_header(self, credentials: IntegrationCredentials) -> str:
        """Generate Basic Auth header for tracking API (write key only)."""
        # Segment tracking API uses write_key as username with empty password
        auth_string = f"{credentials.api_key}:"
        encoded = base64.b64encode(auth_string.encode()).decode()
        return f"Basic {encoded}"

    def _get_config_auth_header(self, credentials: IntegrationCredentials) -> str:
        """Generate Bearer auth header for Config API."""
        if credentials.access_token:
            return f"Bearer {credentials.access_token}"
        return ""

    def _validate_write_key_format(self, write_key: str) -> tuple[bool, str]:
        """Validate write key format.

        Write keys are typically base64-encoded and have a specific format.
        """
        if not write_key:
            return False, "Write key is required"

        if len(write_key) < self.WRITE_KEY_MIN_LENGTH:
            return False, f"Write key is too short (minimum {self.WRITE_KEY_MIN_LENGTH} characters)"

        # Try to validate base64 format (Segment keys are usually base64)
        # But also allow raw keys as Segment format varies
        try:
            # Check if it's valid base64 (informational only, not a hard requirement)
            base64.b64decode(write_key + "==")  # Add padding for check
        except Exception:
            # Not base64, but might still be valid - Segment has different key formats
            pass

        return True, "Write key format is valid"

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Segment API.

        Tests the write key by attempting a batch endpoint call.
        If access token is provided, also tests Config API access.

        Args:
            credentials: IntegrationCredentials with api_key (write key)
                        and optional access_token

        Returns:
            ConnectionTestResult with success status and capabilities
        """
        start_time = time.time()

        # Validate write key format first
        is_valid, validation_message = self._validate_write_key_format(
            credentials.api_key or ""
        )
        if not is_valid:
            return ConnectionTestResult(
                success=False,
                message=validation_message,
                error_code="INVALID_WRITE_KEY_FORMAT",
            )

        try:
            headers = {
                "Authorization": self._get_tracking_auth_header(credentials),
                "Content-Type": "application/json",
            }

            # Test tracking API with a batch call containing a minimal identify
            # Using batch endpoint as it's the most reliable for testing
            test_payload = {
                "batch": [
                    {
                        "type": "identify",
                        "userId": "_segment_connection_test_",
                        "traits": {"_test": True},
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "messageId": str(uuid4()),
                    }
                ],
            }

            response = await self.http_client.post(
                f"{self.TRACKING_API_URL}/batch",
                headers=headers,
                json=test_payload,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                # Success - write key is valid
                permissions = ["write:events"]
                account_info: dict[str, Any] = {
                    "write_key_valid": True,
                }

                # If access token is provided, test Config API
                if credentials.access_token:
                    config_test = await self._test_config_api(credentials)
                    if config_test["success"]:
                        permissions.extend(["read:sources", "read:destinations", "read:tracking_plans"])
                        account_info["config_api_access"] = True
                        account_info["workspace"] = config_test.get("workspace")
                    else:
                        account_info["config_api_access"] = False
                        account_info["config_api_error"] = config_test.get("error")

                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to Segment",
                    latency_ms=latency_ms,
                    api_version="v1",
                    permissions=permissions,
                    account_info=account_info,
                )
            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed. Check your Write Key.",
                    latency_ms=latency_ms,
                    error_code="AUTH_FAILED",
                    error_details={"status_code": 401},
                )
            elif response.status_code == 400:
                # Bad request might mean the key format is wrong
                error_body = response.json() if response.text else {}
                return ConnectionTestResult(
                    success=False,
                    message=f"Invalid request: {error_body.get('message', 'Bad request')}",
                    latency_ms=latency_ms,
                    error_code="BAD_REQUEST",
                    error_details={"status_code": 400, "response": error_body},
                )
            elif response.status_code == 429:
                return ConnectionTestResult(
                    success=False,
                    message="Rate limited. Please try again later.",
                    latency_ms=latency_ms,
                    error_code="RATE_LIMITED",
                    error_details={"status_code": 429},
                )
            else:
                return ConnectionTestResult(
                    success=False,
                    message=f"Connection failed with status {response.status_code}",
                    latency_ms=latency_ms,
                    error_code="API_ERROR",
                    error_details={
                        "status_code": response.status_code,
                        "response": response.text[:500],
                    },
                )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out. Segment API may be slow or unreachable.",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="TIMEOUT",
            )
        except httpx.RequestError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection error: {str(e)}",
                error_code="CONNECTION_ERROR",
                error_details={"error": str(e)},
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                error_code="UNKNOWN_ERROR",
                error_details={"error": str(e)},
            )

    async def _test_config_api(self, credentials: IntegrationCredentials) -> dict:
        """Test Config API access with the access token."""
        if not credentials.access_token:
            return {"success": False, "error": "No access token provided"}

        try:
            headers = {
                "Authorization": self._get_config_auth_header(credentials),
                "Content-Type": "application/json",
            }

            # Get workspace info
            response = await self.http_client.get(
                f"{self.CONFIG_API_URL}/workspaces",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                workspaces = data.get("data", {}).get("workspaces", [])
                workspace_name = workspaces[0].get("name") if workspaces else "Unknown"
                return {"success": True, "workspace": workspace_name}
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Segment Config API.

        Syncs the following resources:
        - sources: Data collection sources
        - destinations: Where data is sent
        - tracking_plans: Schema definitions
        - functions: Custom functions (if available)

        Note: Requires access_token for Config API access.
        Without it, only write key validation is available.

        Args:
            credentials: IntegrationCredentials with api_key and access_token
            resources: Optional list of resources to sync (defaults to all)
            since: Optional datetime for incremental sync (not supported)
            limit: Maximum items per resource

        Returns:
            SyncResult with synced data
        """
        started_at = datetime.utcnow()
        sync_id = f"segment_{started_at.strftime('%Y%m%d_%H%M%S')}"

        # Config API requires access token
        if not credentials.access_token:
            return SyncResult(
                success=False,
                message="Access Token is required for sync. Write Key only allows event tracking.",
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=["Access Token not provided - cannot access Config API"],
            )

        resources_to_sync = resources or ["sources", "destinations", "tracking_plans"]
        synced_data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_points = 0

        headers = {
            "Authorization": self._get_config_auth_header(credentials),
            "Content-Type": "application/json",
        }

        workspace_slug = credentials.extra.get("workspace_slug")

        # Sync sources
        if "sources" in resources_to_sync:
            sources_data, sources_count, sources_errors = await self._sync_sources(
                headers, workspace_slug, limit
            )
            synced_data["sources"] = sources_data
            total_points += sources_count
            errors.extend(sources_errors)

        # Sync destinations
        if "destinations" in resources_to_sync:
            dests_data, dests_count, dests_errors = await self._sync_destinations(
                headers, workspace_slug, limit
            )
            synced_data["destinations"] = dests_data
            total_points += dests_count
            errors.extend(dests_errors)

        # Sync tracking plans
        if "tracking_plans" in resources_to_sync:
            plans_data, plans_count, plans_errors = await self._sync_tracking_plans(
                headers, workspace_slug, limit
            )
            synced_data["tracking_plans"] = plans_data
            total_points += plans_count
            errors.extend(plans_errors)

        # Sync functions (if requested)
        if "functions" in resources_to_sync:
            funcs_data, funcs_count, funcs_errors = await self._sync_functions(
                headers, workspace_slug, limit
            )
            synced_data["functions"] = funcs_data
            total_points += funcs_count
            errors.extend(funcs_errors)

        completed_at = datetime.utcnow()

        return SyncResult(
            success=len(errors) == 0,
            message=(
                f"Synced {total_points} data points from Segment"
                if len(errors) == 0
                else f"Sync completed with {len(errors)} errors"
            ),
            data_points_synced=total_points,
            sync_id=sync_id,
            started_at=started_at,
            completed_at=completed_at,
            errors=errors,
            warnings=warnings,
            data=synced_data,
        )

    async def _sync_sources(
        self,
        headers: dict[str, str],
        workspace_slug: Optional[str],
        limit: int,
    ) -> tuple[list[dict], int, list[str]]:
        """Sync sources from Segment Config API."""
        sources = []
        errors = []

        try:
            params = {"pagination.count": min(limit, 200)}

            response = await self.http_client.get(
                f"{self.CONFIG_API_URL}/sources",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                source_list = data.get("data", {}).get("sources", [])

                for source in source_list[:limit]:
                    sources.append({
                        "id": source.get("id"),
                        "slug": source.get("slug"),
                        "name": source.get("name"),
                        "catalog_name": source.get("catalogName"),
                        "workspace_id": source.get("workspaceId"),
                        "enabled": source.get("enabled", True),
                        "write_keys": source.get("writeKeys", []),
                        "settings": source.get("settings", {}),
                        "labels": source.get("labels", []),
                        "created_at": source.get("createdAt"),
                        "updated_at": source.get("updatedAt"),
                    })
            elif response.status_code == 401:
                errors.append("Authentication failed for Config API")
            elif response.status_code == 403:
                errors.append("Access denied to sources")
            else:
                errors.append(f"Failed to fetch sources: HTTP {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing sources: {str(e)}")

        return sources, len(sources), errors

    async def _sync_destinations(
        self,
        headers: dict[str, str],
        workspace_slug: Optional[str],
        limit: int,
    ) -> tuple[list[dict], int, list[str]]:
        """Sync destinations from Segment Config API."""
        destinations = []
        errors = []

        try:
            params = {"pagination.count": min(limit, 200)}

            response = await self.http_client.get(
                f"{self.CONFIG_API_URL}/destinations",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                dest_list = data.get("data", {}).get("destinations", [])

                for dest in dest_list[:limit]:
                    destinations.append({
                        "id": dest.get("id"),
                        "name": dest.get("name"),
                        "catalog_name": dest.get("catalogName"),
                        "source_id": dest.get("sourceId"),
                        "workspace_id": dest.get("workspaceId"),
                        "enabled": dest.get("enabled", True),
                        "settings": dest.get("settings", {}),
                        "labels": dest.get("labels", []),
                        "created_at": dest.get("createdAt"),
                        "updated_at": dest.get("updatedAt"),
                    })
            elif response.status_code == 401:
                errors.append("Authentication failed for Config API")
            elif response.status_code == 403:
                errors.append("Access denied to destinations")
            else:
                errors.append(f"Failed to fetch destinations: HTTP {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing destinations: {str(e)}")

        return destinations, len(destinations), errors

    async def _sync_tracking_plans(
        self,
        headers: dict[str, str],
        workspace_slug: Optional[str],
        limit: int,
    ) -> tuple[list[dict], int, list[str]]:
        """Sync tracking plans from Segment Config API."""
        plans = []
        errors = []

        try:
            params = {"pagination.count": min(limit, 200)}

            response = await self.http_client.get(
                f"{self.CONFIG_API_URL}/tracking-plans",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                plan_list = data.get("data", {}).get("trackingPlans", [])

                for plan in plan_list[:limit]:
                    # Get plan details including rules
                    plan_details = await self._get_tracking_plan_details(
                        headers, plan.get("id")
                    )

                    plans.append({
                        "id": plan.get("id"),
                        "name": plan.get("name"),
                        "slug": plan.get("slug"),
                        "description": plan.get("description"),
                        "workspace_id": plan.get("workspaceId"),
                        "rules": plan_details.get("rules", []),
                        "created_at": plan.get("createdAt"),
                        "updated_at": plan.get("updatedAt"),
                    })
            elif response.status_code == 404:
                # Tracking plans may not be available for all plans
                pass
            elif response.status_code == 401:
                errors.append("Authentication failed for Config API")
            else:
                errors.append(f"Failed to fetch tracking plans: HTTP {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing tracking plans: {str(e)}")

        return plans, len(plans), errors

    async def _get_tracking_plan_details(
        self,
        headers: dict[str, str],
        plan_id: str,
    ) -> dict:
        """Get detailed tracking plan including rules."""
        try:
            response = await self.http_client.get(
                f"{self.CONFIG_API_URL}/tracking-plans/{plan_id}",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                plan = data.get("data", {}).get("trackingPlan", {})
                return {
                    "rules": plan.get("rules", {}).get("events", []),
                }
            return {}

        except Exception:
            return {}

    async def _sync_functions(
        self,
        headers: dict[str, str],
        workspace_slug: Optional[str],
        limit: int,
    ) -> tuple[list[dict], int, list[str]]:
        """Sync custom functions from Segment Config API."""
        functions = []
        errors = []

        try:
            params = {"pagination.count": min(limit, 200)}

            response = await self.http_client.get(
                f"{self.CONFIG_API_URL}/functions",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                func_list = data.get("data", {}).get("functions", [])

                for func in func_list[:limit]:
                    functions.append({
                        "id": func.get("id"),
                        "resource_type": func.get("resourceType"),
                        "display_name": func.get("displayName"),
                        "description": func.get("description"),
                        "logo_url": func.get("logoUrl"),
                        "catalog_id": func.get("catalogId"),
                        "workspace_id": func.get("workspaceId"),
                        "created_at": func.get("createdAt"),
                        "updated_at": func.get("updatedAt"),
                    })
            elif response.status_code == 404:
                # Functions may not be available for all plans
                pass
            else:
                errors.append(f"Failed to fetch functions: HTTP {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing functions: {str(e)}")

        return functions, len(functions), errors

    async def track(
        self,
        credentials: IntegrationCredentials,
        event: str,
        properties: dict[str, Any],
        user_id: Optional[str] = None,
        anonymous_id: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Track an event via Segment Tracking API.

        Args:
            credentials: IntegrationCredentials with api_key (write key)
            event: Event name
            properties: Event properties
            user_id: User identifier (required if no anonymous_id)
            anonymous_id: Anonymous identifier (required if no user_id)

        Returns:
            Tuple of (success boolean, error message or None)
        """
        if not user_id and not anonymous_id:
            return False, "Either user_id or anonymous_id is required"

        try:
            headers = {
                "Authorization": self._get_tracking_auth_header(credentials),
                "Content-Type": "application/json",
            }

            payload = {
                "event": event,
                "properties": properties,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "messageId": str(uuid4()),
            }

            if user_id:
                payload["userId"] = user_id
            if anonymous_id:
                payload["anonymousId"] = anonymous_id

            response = await self.http_client.post(
                f"{self.TRACKING_API_URL}/track",
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                return True, None
            else:
                return False, f"Track failed: HTTP {response.status_code}"

        except Exception as e:
            return False, f"Track error: {str(e)}"

    async def identify(
        self,
        credentials: IntegrationCredentials,
        user_id: str,
        traits: dict[str, Any],
        anonymous_id: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Identify a user via Segment Tracking API.

        Args:
            credentials: IntegrationCredentials with api_key (write key)
            user_id: User identifier
            traits: User traits/properties
            anonymous_id: Optional anonymous identifier to link

        Returns:
            Tuple of (success boolean, error message or None)
        """
        try:
            headers = {
                "Authorization": self._get_tracking_auth_header(credentials),
                "Content-Type": "application/json",
            }

            payload = {
                "userId": user_id,
                "traits": traits,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "messageId": str(uuid4()),
            }

            if anonymous_id:
                payload["anonymousId"] = anonymous_id

            response = await self.http_client.post(
                f"{self.TRACKING_API_URL}/identify",
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                return True, None
            else:
                return False, f"Identify failed: HTTP {response.status_code}"

        except Exception as e:
            return False, f"Identify error: {str(e)}"

    async def page(
        self,
        credentials: IntegrationCredentials,
        name: str,
        properties: dict[str, Any],
        user_id: Optional[str] = None,
        anonymous_id: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Track a page view via Segment Tracking API.

        Args:
            credentials: IntegrationCredentials with api_key (write key)
            name: Page name
            properties: Page properties (url, referrer, etc.)
            user_id: User identifier (required if no anonymous_id)
            anonymous_id: Anonymous identifier (required if no user_id)

        Returns:
            Tuple of (success boolean, error message or None)
        """
        if not user_id and not anonymous_id:
            return False, "Either user_id or anonymous_id is required"

        try:
            headers = {
                "Authorization": self._get_tracking_auth_header(credentials),
                "Content-Type": "application/json",
            }

            payload = {
                "name": name,
                "properties": properties,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "messageId": str(uuid4()),
            }

            if user_id:
                payload["userId"] = user_id
            if anonymous_id:
                payload["anonymousId"] = anonymous_id

            response = await self.http_client.post(
                f"{self.TRACKING_API_URL}/page",
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                return True, None
            else:
                return False, f"Page failed: HTTP {response.status_code}"

        except Exception as e:
            return False, f"Page error: {str(e)}"

    async def batch(
        self,
        credentials: IntegrationCredentials,
        messages: list[dict[str, Any]],
    ) -> tuple[bool, Optional[str]]:
        """Send a batch of messages via Segment Tracking API.

        This is more efficient for sending multiple events at once.

        Args:
            credentials: IntegrationCredentials with api_key (write key)
            messages: List of message objects (track, identify, page, etc.)
                     Each must include 'type' and appropriate fields

        Returns:
            Tuple of (success boolean, error message or None)
        """
        if not messages:
            return False, "No messages to send"

        try:
            headers = {
                "Authorization": self._get_tracking_auth_header(credentials),
                "Content-Type": "application/json",
            }

            # Add messageId and timestamp to messages that don't have them
            for msg in messages:
                if "messageId" not in msg:
                    msg["messageId"] = str(uuid4())
                if "timestamp" not in msg:
                    msg["timestamp"] = datetime.utcnow().isoformat() + "Z"

            response = await self.http_client.post(
                f"{self.TRACKING_API_URL}/batch",
                headers=headers,
                json={"batch": messages},
            )

            if response.status_code == 200:
                return True, None
            else:
                return False, f"Batch failed: HTTP {response.status_code}"

        except Exception as e:
            return False, f"Batch error: {str(e)}"
