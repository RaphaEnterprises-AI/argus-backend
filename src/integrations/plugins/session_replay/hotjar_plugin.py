"""Hotjar integration plugin for session replay and heatmaps.

Hotjar provides behavior analytics tools including heatmaps, session recordings,
and user feedback surveys. This plugin integrates with Hotjar's API to retrieve
session replay data and correlate user behavior with test results.

IMPORTANT API LIMITATIONS:
--------------------------
Hotjar's public API is extremely limited compared to other session replay tools:
1. The API is primarily read-only and focuses on surveys/feedback
2. Session recordings are NOT accessible via API - only viewable in dashboard
3. Heatmaps are NOT accessible via API - only viewable in dashboard
4. The main API endpoints are for:
   - Survey responses (read-only)
   - Feedback widget responses (read-only)
   - Some user/account information

For full session replay and heatmap data, users must access the Hotjar dashboard
directly. This plugin provides what integration is possible given these limitations.

API Docs: https://help.hotjar.com/hc/en-us/articles/360037898754-Hotjar-API-Reference
"""

import re
import time
from datetime import datetime
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


class HotjarPlugin(IntegrationPlugin):
    """Hotjar integration plugin for session replay and feedback data.

    Hotjar is a behavior analytics company that provides heatmaps, session recordings,
    and user feedback tools. However, the API is limited and does not provide direct
    access to session recordings or heatmaps.

    Key Capabilities:
    - Retrieve survey responses
    - Retrieve feedback widget responses
    - Basic site/account information
    - API key validation

    Limitations:
    - No programmatic access to session recordings (view in dashboard only)
    - No programmatic access to heatmaps (view in dashboard only)
    - No real-time event streaming
    - Limited to survey/feedback data export

    Usage:
        async with HotjarPlugin() as plugin:
            result = await plugin.test_connection(credentials)
            if result.success:
                data = await plugin.sync_data(credentials, resources=["surveys"])
    """

    # Hotjar API base URL
    BASE_URL = "https://insights.hotjar.com/api/v2"

    # Known Hotjar site ID pattern (numeric)
    SITE_ID_PATTERN = re.compile(r"^\d{6,10}$")

    # API key pattern (Hotjar uses alphanumeric keys with hyphens)
    API_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9\-]{20,64}$")

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Hotjar integration metadata."""
        return IntegrationMetadata(
            slug="hotjar",
            name="Hotjar",
            description=(
                "Behavior analytics with heatmaps, session recordings, and surveys. "
                "Note: API access is limited - recordings and heatmaps are only "
                "viewable in the Hotjar dashboard."
            ),
            category=IntegrationCategory.SESSION_REPLAY,
            auth_type=AuthType.API_KEY,
            icon_url="https://www.hotjar.com/favicon.ico",
            docs_url="https://help.hotjar.com/hc/en-us/articles/360037898754-Hotjar-API-Reference",
            website_url="https://www.hotjar.com",
            features=[
                "surveys",
                "feedback_widgets",
                "heatmaps_dashboard_only",
                "recordings_dashboard_only",
                "user_feedback",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="Hotjar API Key",
                    type="password",
                    required=True,
                    placeholder="Enter your Hotjar API key",
                    help_text=(
                        "Personal API key from Hotjar Settings > Integrations > Personal API Key. "
                        "Required scopes: Read access to surveys and feedback."
                    ),
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra",
                    label="Site ID",
                    type="text",
                    required=False,
                    placeholder="e.g., 1234567",
                    help_text=(
                        "Your Hotjar Site ID (found in tracking code). "
                        "Store in credentials.extra['site_id']."
                    ),
                ),
            ],
            supports_webhooks=False,
            supports_sync=True,
            sync_resources=[
                "surveys",
                "feedback",
                "account_info",
            ],
        )

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Hotjar API connection by validating the API key.

        Hotjar doesn't have a dedicated health/ping endpoint, so we validate
        the API key format and attempt to fetch account information.

        Args:
            credentials: IntegrationCredentials with api_key set.

        Returns:
            ConnectionTestResult indicating success or failure.
        """
        log = logger.bind(integration="hotjar")
        start_time = time.time()

        # Validate API key format
        if not credentials.api_key:
            return ConnectionTestResult(
                success=False,
                message="API key is required",
                error_code="MISSING_API_KEY",
            )

        if not self.API_KEY_PATTERN.match(credentials.api_key):
            return ConnectionTestResult(
                success=False,
                message=(
                    "Invalid API key format. Hotjar API keys are typically "
                    "20-64 alphanumeric characters with hyphens."
                ),
                error_code="INVALID_API_KEY_FORMAT",
            )

        # Validate site ID if provided
        site_id = credentials.extra.get("site_id") if credentials.extra else None
        if site_id and not self.SITE_ID_PATTERN.match(str(site_id)):
            return ConnectionTestResult(
                success=False,
                message="Invalid Site ID format. Site IDs are 6-10 digit numbers.",
                error_code="INVALID_SITE_ID_FORMAT",
            )

        try:
            # Attempt to validate the API key by making a request
            # Note: Hotjar's API is very limited, we try to fetch account/user info
            headers = {
                "Authorization": f"Bearer {credentials.api_key}",
                "Content-Type": "application/json",
            }

            # Try to get account information
            response = await self.http_client.get(
                f"{self.BASE_URL}/users/me",
                headers=headers,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                log.info("Hotjar connection successful")
                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to Hotjar",
                    latency_ms=latency_ms,
                    api_version="v2",
                    permissions=["read:surveys", "read:feedback"],
                    account_info={
                        "user_id": data.get("id"),
                        "email": data.get("email"),
                        "name": data.get("name"),
                        "site_id": site_id,
                    },
                )
            elif response.status_code == 401:
                log.warning("Hotjar authentication failed")
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed. Please check your API key.",
                    latency_ms=latency_ms,
                    error_code="AUTH_FAILED",
                    error_details={"status_code": 401},
                )
            elif response.status_code == 403:
                log.warning("Hotjar access forbidden")
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden. Your API key may lack required permissions.",
                    latency_ms=latency_ms,
                    error_code="ACCESS_FORBIDDEN",
                    error_details={"status_code": 403},
                )
            else:
                log.error(
                    "Hotjar connection failed",
                    status_code=response.status_code,
                    response=response.text[:500],
                )
                return ConnectionTestResult(
                    success=False,
                    message=f"Connection failed with status {response.status_code}",
                    latency_ms=latency_ms,
                    error_code="CONNECTION_FAILED",
                    error_details={
                        "status_code": response.status_code,
                        "response": response.text[:200],
                    },
                )

        except httpx.TimeoutException:
            log.error("Hotjar connection timeout")
            return ConnectionTestResult(
                success=False,
                message="Connection timed out. Hotjar may be experiencing issues.",
                error_code="TIMEOUT",
            )
        except httpx.RequestError as e:
            log.error("Hotjar connection error", error=str(e))
            return ConnectionTestResult(
                success=False,
                message=f"Connection error: {str(e)}",
                error_code="REQUEST_ERROR",
                error_details={"error": str(e)},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Hotjar.

        Due to API limitations, this can only sync:
        - Survey responses
        - Feedback widget responses
        - Account information

        Session recordings and heatmaps are NOT available via API.

        Args:
            credentials: IntegrationCredentials with api_key and optional site_id.
            resources: List of resources to sync. Options: ["surveys", "feedback", "account_info"]
            since: Optional datetime for incremental sync (limited support).
            limit: Maximum items per resource.

        Returns:
            SyncResult with synced data and statistics.
        """
        log = logger.bind(integration="hotjar")
        start_time = datetime.utcnow()

        # Default to all available resources
        if resources is None:
            resources = ["surveys", "feedback", "account_info"]

        headers = {
            "Authorization": f"Bearer {credentials.api_key}",
            "Content-Type": "application/json",
        }

        site_id = credentials.extra.get("site_id") if credentials.extra else None
        synced_data: dict[str, Any] = {}
        total_points = 0
        errors: list[str] = []
        warnings: list[str] = []

        # Always add a warning about API limitations
        warnings.append(
            "Hotjar API is limited. Session recordings and heatmaps are only "
            "viewable in the Hotjar dashboard - they cannot be exported via API."
        )

        # Sync account info
        if "account_info" in resources:
            try:
                response = await self.http_client.get(
                    f"{self.BASE_URL}/users/me",
                    headers=headers,
                )
                if response.status_code == 200:
                    synced_data["account_info"] = response.json()
                    total_points += 1
                    log.info("Synced Hotjar account info")
                else:
                    errors.append(f"Failed to fetch account info: {response.status_code}")
            except Exception as e:
                errors.append(f"Error fetching account info: {str(e)}")

        # Sync surveys
        if "surveys" in resources and site_id:
            try:
                response = await self.http_client.get(
                    f"{self.BASE_URL}/sites/{site_id}/surveys",
                    headers=headers,
                    params={"limit": limit},
                )
                if response.status_code == 200:
                    surveys = response.json()
                    synced_data["surveys"] = surveys
                    total_points += len(surveys) if isinstance(surveys, list) else 1
                    log.info("Synced Hotjar surveys", count=total_points)

                    # Fetch responses for each survey
                    survey_responses = []
                    for survey in (surveys if isinstance(surveys, list) else []):
                        survey_id = survey.get("id")
                        if survey_id:
                            resp = await self._fetch_survey_responses(
                                headers, site_id, survey_id, limit, since
                            )
                            if resp:
                                survey_responses.extend(resp)

                    if survey_responses:
                        synced_data["survey_responses"] = survey_responses
                        total_points += len(survey_responses)

                elif response.status_code == 404:
                    warnings.append(f"No surveys found for site {site_id}")
                else:
                    errors.append(f"Failed to fetch surveys: {response.status_code}")
            except Exception as e:
                errors.append(f"Error fetching surveys: {str(e)}")
        elif "surveys" in resources and not site_id:
            warnings.append("Site ID required to fetch surveys. Provide site_id in credentials.extra.")

        # Sync feedback
        if "feedback" in resources and site_id:
            try:
                response = await self.http_client.get(
                    f"{self.BASE_URL}/sites/{site_id}/feedback",
                    headers=headers,
                    params={"limit": limit},
                )
                if response.status_code == 200:
                    feedback = response.json()
                    synced_data["feedback"] = feedback
                    total_points += len(feedback) if isinstance(feedback, list) else 1
                    log.info("Synced Hotjar feedback", count=len(feedback) if isinstance(feedback, list) else 1)
                elif response.status_code == 404:
                    warnings.append(f"No feedback found for site {site_id}")
                else:
                    errors.append(f"Failed to fetch feedback: {response.status_code}")
            except Exception as e:
                errors.append(f"Error fetching feedback: {str(e)}")
        elif "feedback" in resources and not site_id:
            warnings.append("Site ID required to fetch feedback. Provide site_id in credentials.extra.")

        # Note about unavailable resources
        unavailable = [
            r for r in resources
            if r in ["recordings", "heatmaps", "session_replay"]
        ]
        if unavailable:
            warnings.append(
                f"Requested resources {unavailable} are not available via Hotjar API. "
                "These can only be viewed in the Hotjar dashboard."
            )

        return SyncResult(
            success=len(errors) == 0,
            message=(
                f"Synced {total_points} data points from Hotjar"
                if len(errors) == 0
                else f"Sync completed with {len(errors)} error(s)"
            ),
            data_points_synced=total_points,
            started_at=start_time,
            completed_at=datetime.utcnow(),
            errors=errors,
            warnings=warnings,
            data=synced_data,
        )

    async def _fetch_survey_responses(
        self,
        headers: dict[str, str],
        site_id: str,
        survey_id: str,
        limit: int,
        since: Optional[datetime],
    ) -> list[dict[str, Any]]:
        """Fetch responses for a specific survey.

        Args:
            headers: HTTP headers with authorization.
            site_id: Hotjar site ID.
            survey_id: Survey ID to fetch responses for.
            limit: Maximum number of responses.
            since: Optional datetime filter.

        Returns:
            List of survey response dictionaries.
        """
        try:
            params: dict[str, Any] = {"limit": limit}
            if since:
                params["from"] = since.isoformat()

            response = await self.http_client.get(
                f"{self.BASE_URL}/sites/{site_id}/surveys/{survey_id}/responses",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    async def get_dashboard_url(
        self,
        credentials: IntegrationCredentials,
        resource_type: str = "recordings",
    ) -> Optional[str]:
        """Get the Hotjar dashboard URL for a specific resource.

        Since recordings and heatmaps are not available via API, this helper
        provides direct links to view them in the Hotjar dashboard.

        Args:
            credentials: IntegrationCredentials with site_id in extra.
            resource_type: One of "recordings", "heatmaps", "surveys", "feedback".

        Returns:
            Dashboard URL or None if site_id not available.
        """
        site_id = credentials.extra.get("site_id") if credentials.extra else None
        if not site_id:
            return None

        base_dashboard = f"https://insights.hotjar.com/sites/{site_id}"

        url_map = {
            "recordings": f"{base_dashboard}/recordings/list",
            "heatmaps": f"{base_dashboard}/heatmaps/list",
            "surveys": f"{base_dashboard}/polls/list",
            "feedback": f"{base_dashboard}/feedback/list",
        }

        return url_map.get(resource_type)


def create_hotjar_plugin(
    http_client: Optional[httpx.AsyncClient] = None,
) -> HotjarPlugin:
    """Factory function for HotjarPlugin.

    Args:
        http_client: Optional HTTP client for making API requests.

    Returns:
        Configured HotjarPlugin instance.
    """
    return HotjarPlugin(http_client=http_client)
