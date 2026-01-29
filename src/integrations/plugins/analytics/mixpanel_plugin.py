"""Mixpanel Analytics integration plugin.

Mixpanel is a product analytics platform that helps teams analyze
user behavior with events, funnels, and cohorts. This plugin provides
access to events, user profiles, funnels, and JQL queries.

API Docs: https://developer.mixpanel.com/reference/overview
"""

import base64
import json
import time
from datetime import datetime, timedelta
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


class MixpanelPlugin(IntegrationPlugin):
    """Mixpanel Analytics integration plugin.

    Provides access to:
    - Events and event properties
    - User profiles (Engage)
    - Funnels and conversion data
    - JQL queries for custom data retrieval

    Authentication:
    - Project Token for tracking (write)
    - API Secret for data export (read) - optional but recommended
    - Service Account for full API access

    Usage:
        plugin = MixpanelPlugin()
        credentials = IntegrationCredentials(
            api_key="project_token",
            api_secret="api_secret"  # Optional
        )

        # Test connection
        result = await plugin.test_connection(credentials)
        if result.success:
            # Sync events, profiles, and funnels
            sync_result = await plugin.sync_data(credentials)
    """

    # Mixpanel API endpoints
    DATA_API_URL = "https://data.mixpanel.com/api/2.0"
    QUERY_API_URL = "https://mixpanel.com/api/2.0"
    ENGAGE_API_URL = "https://mixpanel.com/api/2.0/engage"

    # EU data residency endpoints
    EU_DATA_API_URL = "https://data-eu.mixpanel.com/api/2.0"
    EU_QUERY_API_URL = "https://eu.mixpanel.com/api/2.0"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Mixpanel plugin metadata."""
        return IntegrationMetadata(
            slug="mixpanel",
            name="Mixpanel",
            description="Product analytics platform for tracking user behavior and engagement",
            category=IntegrationCategory.ANALYTICS,
            auth_type=AuthType.API_KEY,
            icon_url="https://mixpanel.com/favicon.ico",
            docs_url="https://developer.mixpanel.com/reference/overview",
            website_url="https://mixpanel.com",
            features=[
                "Event tracking and analytics",
                "User profile management",
                "Funnel analysis",
                "Retention analysis",
                "Cohort segmentation",
                "JQL queries for custom data",
                "Real-time event streaming",
                "A/B testing insights",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="Project Token",
                    type="password",
                    required=True,
                    placeholder="Enter your Mixpanel Project Token",
                    help_text="Found in Mixpanel Project Settings",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="api_secret",
                    label="API Secret",
                    type="password",
                    required=False,
                    placeholder="Enter your Mixpanel API Secret",
                    help_text="Required for data export APIs. Found in Project Settings.",
                ),
                FieldDefinition(
                    name="extra.project_id",
                    label="Project ID",
                    type="text",
                    required=False,
                    placeholder="Your Mixpanel Project ID",
                    help_text="Numeric project ID for API calls",
                ),
                FieldDefinition(
                    name="extra.eu_residency",
                    label="EU Data Residency",
                    type="select",
                    required=False,
                    help_text="Enable if your project uses EU data residency",
                    options=[
                        {"value": "false", "label": "US (Default)"},
                        {"value": "true", "label": "EU Data Residency"},
                    ],
                ),
            ],
            supports_webhooks=False,
            supports_sync=True,
            sync_resources=["events", "profiles", "funnels", "cohorts"],
        )

    def _get_base_urls(self, credentials: IntegrationCredentials) -> tuple[str, str]:
        """Get the appropriate base URLs based on data residency."""
        eu_residency = credentials.extra.get("eu_residency", "false") == "true"
        if eu_residency:
            return self.EU_DATA_API_URL, self.EU_QUERY_API_URL
        return self.DATA_API_URL, self.QUERY_API_URL

    def _get_auth_header(self, credentials: IntegrationCredentials) -> str:
        """Generate Basic Auth header from API secret."""
        if credentials.api_secret:
            # Service account or API secret auth
            auth_string = f"{credentials.api_secret}:"
            encoded = base64.b64encode(auth_string.encode()).decode()
            return f"Basic {encoded}"
        else:
            # Project token in query params (handled separately)
            return ""

    def _format_date(self, dt: datetime) -> str:
        """Format datetime for Mixpanel API (YYYY-MM-DD)."""
        return dt.strftime("%Y-%m-%d")

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Mixpanel API.

        Tests the connection by fetching user profiles (Engage API),
        which validates the credentials.

        Args:
            credentials: IntegrationCredentials with api_key (project token)
                        and optional api_secret

        Returns:
            ConnectionTestResult with success status and account info
        """
        start_time = time.time()

        if not credentials.api_key:
            return ConnectionTestResult(
                success=False,
                message="Project Token is required",
                error_code="MISSING_CREDENTIALS",
            )

        _, query_url = self._get_base_urls(credentials)

        try:
            # If we have API secret, test with Engage API
            if credentials.api_secret:
                headers = {
                    "Authorization": self._get_auth_header(credentials),
                    "Accept": "application/json",
                }

                # Test with engage endpoint - get a single profile
                response = await self.http_client.post(
                    f"{query_url}/engage",
                    headers=headers,
                    json={
                        "filter_by_cohort": None,
                        "page_size": 1,
                    },
                )
            else:
                # Without API secret, we can only verify token format
                # Real validation would require trying to track an event
                if len(credentials.api_key) < 20:
                    return ConnectionTestResult(
                        success=False,
                        message="Invalid project token format",
                        latency_ms=(time.time() - start_time) * 1000,
                        error_code="INVALID_TOKEN_FORMAT",
                    )

                return ConnectionTestResult(
                    success=True,
                    message="Project token format validated. Add API Secret for full API access.",
                    latency_ms=(time.time() - start_time) * 1000,
                    api_version="2.0",
                    permissions=["write:events"],
                    account_info={
                        "token_validated": True,
                        "api_secret_provided": False,
                        "note": "Limited to event tracking without API Secret",
                    },
                )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                profile_count = data.get("total", 0)

                return ConnectionTestResult(
                    success=True,
                    message=f"Successfully connected to Mixpanel. Found {profile_count} user profiles.",
                    latency_ms=latency_ms,
                    api_version="2.0",
                    permissions=[
                        "read:profiles",
                        "read:events",
                        "read:funnels",
                        "write:events",
                    ],
                    account_info={
                        "profile_count": profile_count,
                        "project_id": credentials.extra.get("project_id"),
                        "eu_residency": credentials.extra.get("eu_residency", "false"),
                    },
                )
            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed. Check your API Secret.",
                    latency_ms=latency_ms,
                    error_code="AUTH_FAILED",
                    error_details={"status_code": 401},
                )
            elif response.status_code == 402:
                return ConnectionTestResult(
                    success=False,
                    message="API access requires a paid Mixpanel plan.",
                    latency_ms=latency_ms,
                    error_code="PAYMENT_REQUIRED",
                    error_details={"status_code": 402},
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
                message="Connection timed out. Mixpanel API may be slow or unreachable.",
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

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Mixpanel.

        Syncs the following resources:
        - events: Event names and properties schema
        - profiles: User profiles from Engage
        - funnels: Saved funnel definitions
        - cohorts: Saved cohort definitions

        Note: Full sync requires API Secret. Without it, only limited
        data is available.

        Args:
            credentials: IntegrationCredentials with api_key and optional api_secret
            resources: Optional list of resources to sync (defaults to all)
            since: Optional datetime for incremental sync
            limit: Maximum items per resource

        Returns:
            SyncResult with synced data
        """
        started_at = datetime.utcnow()
        sync_id = f"mixpanel_{started_at.strftime('%Y%m%d_%H%M%S')}"

        # Without API secret, we can't sync data
        if not credentials.api_secret:
            return SyncResult(
                success=False,
                message="API Secret is required for data sync",
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=["API Secret not provided - cannot access data APIs"],
            )

        resources_to_sync = resources or ["events", "profiles", "funnels"]
        synced_data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_points = 0

        data_url, query_url = self._get_base_urls(credentials)
        headers = {
            "Authorization": self._get_auth_header(credentials),
            "Accept": "application/json",
        }

        # Sync events
        if "events" in resources_to_sync:
            events_data, events_count, events_errors = await self._sync_events(
                headers, query_url, since, limit
            )
            synced_data["events"] = events_data
            total_points += events_count
            errors.extend(events_errors)

        # Sync profiles
        if "profiles" in resources_to_sync:
            profiles_data, profiles_count, profiles_errors = await self._sync_profiles(
                headers, query_url, limit
            )
            synced_data["profiles"] = profiles_data
            total_points += profiles_count
            errors.extend(profiles_errors)

        # Sync funnels
        if "funnels" in resources_to_sync:
            funnels_data, funnels_count, funnels_errors = await self._sync_funnels(
                headers, query_url, credentials.extra.get("project_id"), limit
            )
            synced_data["funnels"] = funnels_data
            total_points += funnels_count
            errors.extend(funnels_errors)

        # Sync cohorts
        if "cohorts" in resources_to_sync:
            cohorts_data, cohorts_count, cohorts_errors = await self._sync_cohorts(
                headers, query_url, credentials.extra.get("project_id"), limit
            )
            synced_data["cohorts"] = cohorts_data
            total_points += cohorts_count
            errors.extend(cohorts_errors)

        completed_at = datetime.utcnow()

        return SyncResult(
            success=len(errors) == 0,
            message=(
                f"Synced {total_points} data points from Mixpanel"
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

    async def _sync_events(
        self,
        headers: dict[str, str],
        query_url: str,
        since: Optional[datetime],
        limit: int,
    ) -> tuple[list[dict], int, list[str]]:
        """Sync event names and their properties from Mixpanel."""
        events = []
        errors = []

        try:
            # Get event names using the events/names endpoint
            start_date = since or (datetime.utcnow() - timedelta(days=30))
            end_date = datetime.utcnow()

            params = {
                "type": "general",
                "unit": "day",
                "from_date": self._format_date(start_date),
                "to_date": self._format_date(end_date),
                "limit": limit,
            }

            response = await self.http_client.get(
                f"{query_url}/events/names",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                event_names = data if isinstance(data, list) else data.get("data", [])

                for event_name in event_names[:limit]:
                    # Get event properties
                    props = await self._get_event_properties(
                        headers, query_url, event_name
                    )

                    events.append({
                        "event_name": event_name,
                        "properties": props,
                        "period_start": start_date.isoformat(),
                        "period_end": end_date.isoformat(),
                    })

            elif response.status_code == 402:
                errors.append("Events API requires a paid Mixpanel plan")
            else:
                errors.append(f"Failed to fetch events: HTTP {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing events: {str(e)}")

        return events, len(events), errors

    async def _get_event_properties(
        self,
        headers: dict[str, str],
        query_url: str,
        event_name: str,
    ) -> list[dict]:
        """Get properties for a specific event."""
        properties = []

        try:
            response = await self.http_client.get(
                f"{query_url}/events/properties",
                headers=headers,
                params={
                    "event": event_name,
                    "type": "general",
                    "unit": "day",
                    "limit": 20,
                },
            )

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    properties = [{"name": p, "type": "unknown"} for p in data]
                elif isinstance(data, dict):
                    for prop_name, prop_data in data.items():
                        properties.append({
                            "name": prop_name,
                            "type": prop_data.get("type", "unknown") if isinstance(prop_data, dict) else "unknown",
                        })

        except Exception:
            # Silently fail - properties are supplementary
            pass

        return properties

    async def _sync_profiles(
        self,
        headers: dict[str, str],
        query_url: str,
        limit: int,
    ) -> tuple[list[dict], int, list[str]]:
        """Sync user profiles from Mixpanel Engage."""
        profiles = []
        errors = []

        try:
            response = await self.http_client.post(
                f"{query_url}/engage",
                headers=headers,
                json={
                    "filter_by_cohort": None,
                    "page_size": min(limit, 1000),  # Max 1000 per page
                },
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                for profile in results[:limit]:
                    distinct_id = profile.get("$distinct_id")
                    properties = profile.get("$properties", {})

                    profiles.append({
                        "distinct_id": distinct_id,
                        "email": properties.get("$email"),
                        "name": properties.get("$name"),
                        "created": properties.get("$created"),
                        "last_seen": properties.get("$last_seen"),
                        "city": properties.get("$city"),
                        "region": properties.get("$region"),
                        "country": properties.get("mp_country_code"),
                        "properties": {
                            k: v for k, v in properties.items()
                            if not k.startswith("$")
                        },
                    })

            elif response.status_code == 402:
                errors.append("Engage API requires a paid Mixpanel plan")
            else:
                errors.append(f"Failed to fetch profiles: HTTP {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing profiles: {str(e)}")

        return profiles, len(profiles), errors

    async def _sync_funnels(
        self,
        headers: dict[str, str],
        query_url: str,
        project_id: Optional[str],
        limit: int,
    ) -> tuple[list[dict], int, list[str]]:
        """Sync saved funnels from Mixpanel."""
        funnels = []
        errors = []

        try:
            params = {}
            if project_id:
                params["project_id"] = project_id

            response = await self.http_client.get(
                f"{query_url}/funnels/list",
                headers=headers,
                params=params if params else None,
            )

            if response.status_code == 200:
                data = response.json()
                funnel_list = data if isinstance(data, list) else data.get("data", [])

                for funnel in funnel_list[:limit]:
                    funnels.append({
                        "funnel_id": funnel.get("funnel_id"),
                        "name": funnel.get("name"),
                        "steps": funnel.get("steps", []),
                        "created_at": funnel.get("created"),
                    })

            elif response.status_code == 404:
                # No funnels or endpoint not available
                pass
            elif response.status_code == 402:
                errors.append("Funnels API requires a paid Mixpanel plan")
            else:
                errors.append(f"Failed to fetch funnels: HTTP {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing funnels: {str(e)}")

        return funnels, len(funnels), errors

    async def _sync_cohorts(
        self,
        headers: dict[str, str],
        query_url: str,
        project_id: Optional[str],
        limit: int,
    ) -> tuple[list[dict], int, list[str]]:
        """Sync saved cohorts from Mixpanel."""
        cohorts = []
        errors = []

        try:
            params = {}
            if project_id:
                params["project_id"] = project_id

            response = await self.http_client.get(
                f"{query_url}/cohorts/list",
                headers=headers,
                params=params if params else None,
            )

            if response.status_code == 200:
                data = response.json()
                cohort_list = data if isinstance(data, list) else data.get("data", [])

                for cohort in cohort_list[:limit]:
                    cohorts.append({
                        "id": cohort.get("id"),
                        "name": cohort.get("name"),
                        "description": cohort.get("description"),
                        "count": cohort.get("count"),
                        "is_visible": cohort.get("is_visible", True),
                        "created_at": cohort.get("created"),
                    })

            elif response.status_code == 404:
                # No cohorts or endpoint not available
                pass
            elif response.status_code == 402:
                errors.append("Cohorts API requires a paid Mixpanel plan")
            else:
                errors.append(f"Failed to fetch cohorts: HTTP {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing cohorts: {str(e)}")

        return cohorts, len(cohorts), errors

    async def execute_jql(
        self,
        credentials: IntegrationCredentials,
        query: str,
    ) -> tuple[Optional[list[dict]], Optional[str]]:
        """Execute a JQL (JavaScript Query Language) query.

        JQL allows custom data retrieval using JavaScript-like syntax.

        Args:
            credentials: IntegrationCredentials with api_key and api_secret
            query: JQL query string

        Returns:
            Tuple of (results list, error message or None)
        """
        if not credentials.api_secret:
            return None, "API Secret is required for JQL queries"

        _, query_url = self._get_base_urls(credentials)
        headers = {
            "Authorization": self._get_auth_header(credentials),
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            response = await self.http_client.post(
                f"{query_url}/jql",
                headers=headers,
                data={"script": query},
                timeout=60.0,  # JQL queries can take time
            )

            if response.status_code == 200:
                return response.json(), None
            elif response.status_code == 400:
                return None, f"Invalid JQL query: {response.text}"
            elif response.status_code == 402:
                return None, "JQL requires a paid Mixpanel plan"
            else:
                return None, f"JQL query failed: HTTP {response.status_code}"

        except Exception as e:
            return None, f"JQL error: {str(e)}"

    async def track_event(
        self,
        credentials: IntegrationCredentials,
        event_name: str,
        properties: dict[str, Any],
        distinct_id: str,
    ) -> tuple[bool, Optional[str]]:
        """Track an event using the Mixpanel tracking API.

        Args:
            credentials: IntegrationCredentials with api_key (project token)
            event_name: Name of the event to track
            properties: Event properties
            distinct_id: Unique identifier for the user

        Returns:
            Tuple of (success boolean, error message or None)
        """
        try:
            event_data = {
                "event": event_name,
                "properties": {
                    **properties,
                    "token": credentials.api_key,
                    "distinct_id": distinct_id,
                    "time": int(datetime.utcnow().timestamp()),
                },
            }

            # Base64 encode the event data
            encoded_data = base64.b64encode(json.dumps(event_data).encode()).decode()

            response = await self.http_client.get(
                "https://api.mixpanel.com/track",
                params={"data": encoded_data},
            )

            if response.status_code == 200 and response.text.strip() == "1":
                return True, None
            else:
                return False, f"Track failed: {response.text}"

        except Exception as e:
            return False, f"Track error: {str(e)}"
