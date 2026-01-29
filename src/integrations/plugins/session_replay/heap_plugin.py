"""Heap integration plugin for product analytics and session replay.

Heap automatically captures all user interactions without requiring manual event
tracking. This plugin integrates with Heap's APIs to retrieve event data, user
information, and session details for correlation with test results.

IMPORTANT API INFORMATION:
--------------------------
Heap provides several APIs with different purposes:

1. Track API (server-side):
   - POST events to Heap from server
   - Add/update user properties
   - Used for data ingestion, not retrieval

2. Query API (formerly Export API):
   - Export raw event data
   - Export user data
   - Requires Enterprise plan
   - Has rate limits

3. Connect API (data export):
   - Scheduled data exports to data warehouses
   - BigQuery, Snowflake, Redshift integrations
   - Requires Enterprise plan

4. Session Replay:
   - Available in Growth and Pro plans
   - Viewing is through dashboard only
   - No API access to replay videos

The Track API is the most commonly available. The Query API requires
Enterprise access. This plugin supports both with appropriate error handling.

API Docs: https://developers.heap.io/reference
"""

import time
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
import structlog

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

logger = structlog.get_logger()


class HeapPlugin(IntegrationPlugin):
    """Heap integration plugin for product analytics.

    Heap provides auto-capture analytics that records all user interactions
    without requiring manual event tracking. This plugin supports:

    - Connection testing via Track API
    - User property updates (Track API)
    - Event export (Query API - Enterprise)
    - User export (Query API - Enterprise)
    - Session data export (Query API - Enterprise)

    Note: Session replay videos are only viewable in the Heap dashboard.

    Plan Requirements:
    - Free/Growth: Track API only
    - Pro: Track API + limited Query API
    - Enterprise: Full API access including Query API

    Usage:
        async with HeapPlugin() as plugin:
            result = await plugin.test_connection(credentials)
            if result.success:
                data = await plugin.sync_data(
                    credentials,
                    resources=["events", "users", "sessions"]
                )
    """

    # Heap API endpoints
    TRACK_API_URL = "https://heapanalytics.com/api"
    QUERY_API_URL = "https://heapanalytics.com/api/query"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Heap integration metadata."""
        return IntegrationMetadata(
            slug="heap",
            name="Heap",
            description=(
                "Product analytics with auto-capture technology. Heap automatically "
                "records all user interactions for complete behavioral data. "
                "Query API requires Enterprise plan."
            ),
            category=IntegrationCategory.SESSION_REPLAY,
            auth_type=AuthType.API_KEY,
            icon_url="https://heap.io/favicon.ico",
            docs_url="https://developers.heap.io/reference",
            website_url="https://heap.io",
            features=[
                "auto_capture_events",
                "user_tracking",
                "session_analytics",
                "funnel_analysis",
                "retention_analysis",
                "session_replay_dashboard_only",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="Heap App ID",
                    type="text",
                    required=True,
                    placeholder="e.g., 1234567890",
                    help_text=(
                        "Your Heap App ID (also called Environment ID). "
                        "Found in Heap dashboard under Account > Environment."
                    ),
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="api_secret",
                    label="API Secret (Enterprise)",
                    type="password",
                    required=False,
                    placeholder="Enter your Heap API secret",
                    help_text=(
                        "Required for Query API access (Enterprise plan). "
                        "Found in Account > API."
                    ),
                ),
            ],
            supports_webhooks=False,
            supports_sync=True,
            sync_resources=[
                "events",
                "users",
                "sessions",
                "track_event",
                "identify_user",
            ],
        )

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Heap API connection.

        Tests the Track API endpoint which is available on all plans.
        If API secret is provided, also tests Query API access.

        Args:
            credentials: IntegrationCredentials with:
                - api_key: Heap App ID (required)
                - api_secret: API secret for Query API (optional, Enterprise)

        Returns:
            ConnectionTestResult indicating success or failure.
        """
        log = logger.bind(integration="heap")
        start_time = time.time()

        # Validate App ID (api_key)
        if not credentials.api_key:
            return ConnectionTestResult(
                success=False,
                message="Heap App ID is required",
                error_code="MISSING_APP_ID",
            )

        # App ID should be numeric
        if not credentials.api_key.isdigit():
            return ConnectionTestResult(
                success=False,
                message="Invalid App ID format. Heap App IDs are numeric.",
                error_code="INVALID_APP_ID_FORMAT",
            )

        try:
            # Test Track API with a minimal identify call
            # This validates the App ID is correct
            track_response = await self._test_track_api(credentials.api_key)
            latency_ms = (time.time() - start_time) * 1000

            if not track_response["success"]:
                return ConnectionTestResult(
                    success=False,
                    message=track_response["message"],
                    latency_ms=latency_ms,
                    error_code=track_response.get("error_code", "TRACK_API_FAILED"),
                    error_details=track_response.get("details"),
                )

            # Track API works, now check Query API if secret provided
            permissions = ["track:events", "track:users"]
            api_access = ["Track API"]

            if credentials.api_secret:
                query_response = await self._test_query_api(
                    credentials.api_key,
                    credentials.api_secret,
                )
                if query_response["success"]:
                    permissions.extend([
                        "query:events",
                        "query:users",
                        "query:sessions",
                        "query:export",
                    ])
                    api_access.append("Query API (Enterprise)")
                else:
                    # Query API failed but Track API worked
                    log.warning(
                        "Query API access failed",
                        error=query_response.get("message"),
                    )

            log.info("Heap connection successful", api_access=api_access)

            return ConnectionTestResult(
                success=True,
                message=f"Successfully connected to Heap ({', '.join(api_access)})",
                latency_ms=latency_ms,
                api_version="v1",
                permissions=permissions,
                account_info={
                    "app_id": credentials.api_key,
                    "has_query_api": credentials.api_secret is not None,
                    "api_access": api_access,
                },
            )

        except httpx.TimeoutException:
            log.error("Heap connection timeout")
            return ConnectionTestResult(
                success=False,
                message="Connection timed out. Heap may be experiencing issues.",
                error_code="TIMEOUT",
            )
        except httpx.RequestError as e:
            log.error("Heap connection error", error=str(e))
            return ConnectionTestResult(
                success=False,
                message=f"Connection error: {str(e)}",
                error_code="REQUEST_ERROR",
                error_details={"error": str(e)},
            )

    async def _test_track_api(self, app_id: str) -> dict[str, Any]:
        """Test Track API with a minimal request.

        The Track API accepts event tracking requests. We send a minimal
        identify request to verify the App ID is valid.

        Args:
            app_id: Heap App ID.

        Returns:
            Dict with success status and message.
        """
        # Heap Track API accepts events via POST
        # We use the identify endpoint with a test identity
        url = f"{self.TRACK_API_URL}/track"

        payload = {
            "app_id": app_id,
            "identity": f"_test_connection_{int(time.time())}",
            "event": "_argus_connection_test",
            "properties": {
                "_test": True,
                "_timestamp": datetime.utcnow().isoformat(),
            },
        }

        response = await self.http_client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        # Heap returns 200 for valid requests, 4xx for invalid
        if response.status_code == 200:
            return {"success": True, "message": "Track API accessible"}
        elif response.status_code == 400:
            # 400 could mean invalid app_id
            return {
                "success": False,
                "message": "Invalid App ID or request format",
                "error_code": "INVALID_APP_ID",
                "details": {"response": response.text[:200]},
            }
        elif response.status_code == 401:
            return {
                "success": False,
                "message": "App ID not recognized",
                "error_code": "AUTH_FAILED",
            }
        else:
            return {
                "success": False,
                "message": f"Unexpected response: {response.status_code}",
                "error_code": "UNEXPECTED_STATUS",
                "details": {
                    "status_code": response.status_code,
                    "response": response.text[:200],
                },
            }

    async def _test_query_api(self, app_id: str, api_secret: str) -> dict[str, Any]:
        """Test Query API access (Enterprise feature).

        The Query API allows exporting event and user data. This requires
        an Enterprise plan and API secret.

        Args:
            app_id: Heap App ID.
            api_secret: API secret for authentication.

        Returns:
            Dict with success status and message.
        """
        # Query API uses Basic Auth with app_id:secret
        auth = (app_id, api_secret)

        # Try a simple query to verify access
        # We'll query for a minimal time range to avoid large responses
        url = f"{self.QUERY_API_URL}/events"

        # Query for events in the last minute (should return minimal data)
        params = {
            "from": (datetime.utcnow() - timedelta(minutes=1)).isoformat() + "Z",
            "to": datetime.utcnow().isoformat() + "Z",
            "limit": 1,
        }

        response = await self.http_client.get(
            url,
            auth=auth,
            params=params,
        )

        if response.status_code == 200:
            return {"success": True, "message": "Query API accessible"}
        elif response.status_code == 401:
            return {
                "success": False,
                "message": "Query API authentication failed. Check API secret.",
                "error_code": "QUERY_AUTH_FAILED",
            }
        elif response.status_code == 403:
            return {
                "success": False,
                "message": "Query API access denied. Enterprise plan may be required.",
                "error_code": "QUERY_ACCESS_DENIED",
            }
        else:
            return {
                "success": False,
                "message": f"Query API returned {response.status_code}",
                "error_code": "QUERY_UNEXPECTED_STATUS",
            }

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Heap.

        Available sync operations:
        - events: Export event data (Query API - Enterprise)
        - users: Export user data (Query API - Enterprise)
        - sessions: Export session data (Query API - Enterprise)

        The Track API (identify_user, track_event) is write-only and
        cannot be used for data sync.

        Args:
            credentials: IntegrationCredentials with api_key and optional api_secret.
            resources: List of resources to sync. Options: ["events", "users", "sessions"]
            since: Optional datetime for incremental sync.
            limit: Maximum items per resource.

        Returns:
            SyncResult with synced data and statistics.
        """
        log = logger.bind(integration="heap")
        start_time = datetime.utcnow()

        # Default resources
        if resources is None:
            resources = ["events", "users", "sessions"]

        synced_data: dict[str, Any] = {}
        total_points = 0
        errors: list[str] = []
        warnings: list[str] = []

        # Check for Query API access
        if not credentials.api_secret:
            warnings.append(
                "No API secret provided. Query API access requires Enterprise plan. "
                "Only basic account validation is possible without it."
            )
            # Can only return basic info without Query API
            synced_data["info"] = {
                "app_id": credentials.api_key,
                "note": "Query API access requires api_secret (Enterprise plan)",
            }
            return SyncResult(
                success=True,
                message="Limited sync - Query API requires Enterprise plan",
                data_points_synced=0,
                started_at=start_time,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=synced_data,
            )

        # Session replay note
        if "session_replay" in resources or "recordings" in resources:
            warnings.append(
                "Session replay videos are only viewable in the Heap dashboard. "
                "They cannot be exported via API."
            )

        # Set up auth for Query API
        auth = (credentials.api_key, credentials.api_secret)

        # Default time range
        end_time = datetime.utcnow()
        start_range = since or (end_time - timedelta(days=7))

        # Sync events
        if "events" in resources:
            events_result = await self._sync_events(
                auth, start_range, end_time, limit, log
            )
            if events_result.get("data"):
                synced_data["events"] = events_result["data"]
                total_points += events_result.get("count", 0)
            if events_result.get("error"):
                errors.append(events_result["error"])
            if events_result.get("warning"):
                warnings.append(events_result["warning"])

        # Sync users
        if "users" in resources:
            users_result = await self._sync_users(
                auth, start_range, end_time, limit, log
            )
            if users_result.get("data"):
                synced_data["users"] = users_result["data"]
                total_points += users_result.get("count", 0)
            if users_result.get("error"):
                errors.append(users_result["error"])
            if users_result.get("warning"):
                warnings.append(users_result["warning"])

        # Sync sessions
        if "sessions" in resources:
            sessions_result = await self._sync_sessions(
                auth, start_range, end_time, limit, log
            )
            if sessions_result.get("data"):
                synced_data["sessions"] = sessions_result["data"]
                total_points += sessions_result.get("count", 0)
            if sessions_result.get("error"):
                errors.append(sessions_result["error"])
            if sessions_result.get("warning"):
                warnings.append(sessions_result["warning"])

        success = len(errors) == 0
        message = (
            f"Synced {total_points} data points from Heap"
            if success
            else f"Sync completed with {len(errors)} error(s)"
        )

        return SyncResult(
            success=success,
            message=message,
            data_points_synced=total_points,
            started_at=start_time,
            completed_at=datetime.utcnow(),
            errors=errors,
            warnings=warnings,
            data=synced_data,
        )

    async def _sync_events(
        self,
        auth: tuple[str, str],
        from_time: datetime,
        to_time: datetime,
        limit: int,
        log: Any,
    ) -> dict[str, Any]:
        """Sync events from Heap Query API.

        Args:
            auth: Tuple of (app_id, api_secret) for basic auth.
            from_time: Start of time range.
            to_time: End of time range.
            limit: Maximum events to return.
            log: Logger instance.

        Returns:
            Dict with data, count, and optional error/warning.
        """
        try:
            url = f"{self.QUERY_API_URL}/events"
            params = {
                "from": from_time.isoformat() + "Z",
                "to": to_time.isoformat() + "Z",
                "limit": limit,
            }

            response = await self.http_client.get(url, auth=auth, params=params)

            if response.status_code == 200:
                data = response.json()
                events = data if isinstance(data, list) else data.get("events", [])
                log.info("Synced Heap events", count=len(events))
                return {"data": events, "count": len(events)}
            elif response.status_code == 403:
                return {
                    "data": None,
                    "warning": "Events export requires Enterprise plan",
                }
            else:
                return {
                    "data": None,
                    "error": f"Failed to fetch events: {response.status_code}",
                }

        except Exception as e:
            log.error("Error syncing Heap events", error=str(e))
            return {"data": None, "error": f"Error fetching events: {str(e)}"}

    async def _sync_users(
        self,
        auth: tuple[str, str],
        from_time: datetime,
        to_time: datetime,
        limit: int,
        log: Any,
    ) -> dict[str, Any]:
        """Sync users from Heap Query API.

        Args:
            auth: Tuple of (app_id, api_secret) for basic auth.
            from_time: Start of time range (users active since).
            to_time: End of time range.
            limit: Maximum users to return.
            log: Logger instance.

        Returns:
            Dict with data, count, and optional error/warning.
        """
        try:
            url = f"{self.QUERY_API_URL}/users"
            params = {
                "from": from_time.isoformat() + "Z",
                "to": to_time.isoformat() + "Z",
                "limit": limit,
            }

            response = await self.http_client.get(url, auth=auth, params=params)

            if response.status_code == 200:
                data = response.json()
                users = data if isinstance(data, list) else data.get("users", [])
                log.info("Synced Heap users", count=len(users))
                return {"data": users, "count": len(users)}
            elif response.status_code == 403:
                return {
                    "data": None,
                    "warning": "Users export requires Enterprise plan",
                }
            else:
                return {
                    "data": None,
                    "error": f"Failed to fetch users: {response.status_code}",
                }

        except Exception as e:
            log.error("Error syncing Heap users", error=str(e))
            return {"data": None, "error": f"Error fetching users: {str(e)}"}

    async def _sync_sessions(
        self,
        auth: tuple[str, str],
        from_time: datetime,
        to_time: datetime,
        limit: int,
        log: Any,
    ) -> dict[str, Any]:
        """Sync sessions from Heap Query API.

        Args:
            auth: Tuple of (app_id, api_secret) for basic auth.
            from_time: Start of time range.
            to_time: End of time range.
            limit: Maximum sessions to return.
            log: Logger instance.

        Returns:
            Dict with data, count, and optional error/warning.
        """
        try:
            url = f"{self.QUERY_API_URL}/sessions"
            params = {
                "from": from_time.isoformat() + "Z",
                "to": to_time.isoformat() + "Z",
                "limit": limit,
            }

            response = await self.http_client.get(url, auth=auth, params=params)

            if response.status_code == 200:
                data = response.json()
                sessions = data if isinstance(data, list) else data.get("sessions", [])
                log.info("Synced Heap sessions", count=len(sessions))
                return {"data": sessions, "count": len(sessions)}
            elif response.status_code == 403:
                return {
                    "data": None,
                    "warning": "Sessions export requires Enterprise plan",
                }
            else:
                return {
                    "data": None,
                    "error": f"Failed to fetch sessions: {response.status_code}",
                }

        except Exception as e:
            log.error("Error syncing Heap sessions", error=str(e))
            return {"data": None, "error": f"Error fetching sessions: {str(e)}"}

    async def track_event(
        self,
        credentials: IntegrationCredentials,
        event_name: str,
        identity: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Track a custom event via Heap Track API.

        This is a write operation to send events to Heap from server-side.
        Useful for tracking test execution events in Heap.

        Args:
            credentials: IntegrationCredentials with api_key (App ID).
            event_name: Name of the event to track.
            identity: User identity (email or user ID).
            properties: Optional event properties.

        Returns:
            True if event was tracked successfully.
        """
        log = logger.bind(integration="heap", event=event_name)

        payload = {
            "app_id": credentials.api_key,
            "identity": identity,
            "event": event_name,
            "properties": properties or {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        try:
            response = await self.http_client.post(
                f"{self.TRACK_API_URL}/track",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                log.info("Event tracked successfully")
                return True
            else:
                log.warning(
                    "Failed to track event",
                    status_code=response.status_code,
                )
                return False

        except Exception as e:
            log.error("Error tracking event", error=str(e))
            return False

    async def identify_user(
        self,
        credentials: IntegrationCredentials,
        identity: str,
        properties: dict[str, Any],
    ) -> bool:
        """Add or update user properties via Heap Track API.

        This allows setting user-level properties that will be
        associated with the user's events.

        Args:
            credentials: IntegrationCredentials with api_key (App ID).
            identity: User identity (email or user ID).
            properties: User properties to set.

        Returns:
            True if user was identified successfully.
        """
        log = logger.bind(integration="heap", identity=identity)

        payload = {
            "app_id": credentials.api_key,
            "identity": identity,
            "properties": properties,
        }

        try:
            response = await self.http_client.post(
                f"{self.TRACK_API_URL}/add_user_properties",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                log.info("User identified successfully")
                return True
            else:
                log.warning(
                    "Failed to identify user",
                    status_code=response.status_code,
                )
                return False

        except Exception as e:
            log.error("Error identifying user", error=str(e))
            return False

    def get_dashboard_url(
        self,
        credentials: IntegrationCredentials,
        resource_type: str = "events",
    ) -> str:
        """Get the Heap dashboard URL for a specific resource.

        Since session replays are only viewable in the dashboard,
        this helper provides direct links.

        Args:
            credentials: IntegrationCredentials (not used, for signature consistency).
            resource_type: One of "events", "users", "sessions", "replay".

        Returns:
            Dashboard URL.
        """
        base_url = "https://heapanalytics.com"

        url_map = {
            "events": f"{base_url}/app/definitions/events",
            "users": f"{base_url}/app/segments",
            "sessions": f"{base_url}/app/analysis/sessions",
            "replay": f"{base_url}/app/analysis/replay",
            "funnels": f"{base_url}/app/analysis/funnels",
            "retention": f"{base_url}/app/analysis/retention",
        }

        return url_map.get(resource_type, base_url)


def create_heap_plugin(
    http_client: Optional[httpx.AsyncClient] = None,
) -> HeapPlugin:
    """Factory function for HeapPlugin.

    Args:
        http_client: Optional HTTP client for making API requests.

    Returns:
        Configured HeapPlugin instance.
    """
    return HeapPlugin(http_client=http_client)
