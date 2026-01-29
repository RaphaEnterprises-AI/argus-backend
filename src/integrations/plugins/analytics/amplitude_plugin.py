"""Amplitude Analytics integration plugin.

Amplitude is a product analytics platform that tracks user behavior
to help teams build better products. This plugin provides access to
events, cohorts, user properties, and chart data.

API Docs: https://www.docs.developers.amplitude.com/analytics/apis/
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


class AmplitudePlugin(IntegrationPlugin):
    """Amplitude Analytics integration plugin.

    Provides access to:
    - Event taxonomy and metrics
    - User properties and cohorts
    - Charts and segmentation data
    - Export API for bulk data retrieval

    Authentication:
    - Uses API Key + Secret Key with Basic Auth
    - API keys are project-scoped

    Usage:
        plugin = AmplitudePlugin()
        credentials = IntegrationCredentials(
            api_key="your_api_key",
            api_secret="your_secret_key"
        )

        # Test connection
        result = await plugin.test_connection(credentials)
        if result.success:
            # Sync events, cohorts, and user properties
            sync_result = await plugin.sync_data(credentials)
    """

    BASE_URL = "https://amplitude.com/api/2"
    EXPORT_BASE_URL = "https://amplitude.com/api/2/export"
    TAXONOMY_BASE_URL = "https://amplitude.com/api/2/taxonomy"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Amplitude plugin metadata."""
        return IntegrationMetadata(
            slug="amplitude",
            name="Amplitude",
            description="Product analytics platform for tracking user behavior and engagement",
            category=IntegrationCategory.ANALYTICS,
            auth_type=AuthType.BASIC_AUTH,
            icon_url="https://amplitude.com/favicon.ico",
            docs_url="https://www.docs.developers.amplitude.com/analytics/apis/",
            website_url="https://amplitude.com",
            features=[
                "Event tracking and analytics",
                "User behavior analysis",
                "Cohort management",
                "Custom charts and segmentation",
                "Funnel analysis",
                "Retention analysis",
                "User property tracking",
                "Bulk data export",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key",
                    type="password",
                    required=True,
                    placeholder="Enter your Amplitude API Key",
                    help_text="Found in Amplitude Settings > Projects > API Keys",
                ),
                FieldDefinition(
                    name="api_secret",
                    label="Secret Key",
                    type="password",
                    required=True,
                    placeholder="Enter your Amplitude Secret Key",
                    help_text="Found alongside your API Key in project settings",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra.project_id",
                    label="Project ID",
                    type="text",
                    required=False,
                    placeholder="Optional project ID for multi-project orgs",
                    help_text="Leave blank to use the default project for the API key",
                ),
            ],
            supports_webhooks=False,
            supports_sync=True,
            sync_resources=["events", "cohorts", "user_properties", "charts"],
        )

    def _get_auth_header(self, credentials: IntegrationCredentials) -> str:
        """Generate Basic Auth header from API key and secret."""
        auth_string = f"{credentials.api_key}:{credentials.api_secret}"
        encoded = base64.b64encode(auth_string.encode()).decode()
        return f"Basic {encoded}"

    def _format_date(self, dt: datetime) -> str:
        """Format datetime for Amplitude API (YYYYMMDD)."""
        return dt.strftime("%Y%m%d")

    def _format_datetime_hour(self, dt: datetime) -> str:
        """Format datetime with hour for Export API (YYYYMMDDTHH)."""
        return dt.strftime("%Y%m%dT%H")

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Amplitude API.

        Tests the connection by fetching event taxonomy, which requires
        valid API credentials.

        Args:
            credentials: IntegrationCredentials with api_key and api_secret

        Returns:
            ConnectionTestResult with success status and account info
        """
        start_time = time.time()

        if not credentials.api_key or not credentials.api_secret:
            return ConnectionTestResult(
                success=False,
                message="API Key and Secret Key are required",
                error_code="MISSING_CREDENTIALS",
            )

        try:
            headers = {
                "Authorization": self._get_auth_header(credentials),
                "Content-Type": "application/json",
            }

            # Test with taxonomy/event endpoint
            response = await self.http_client.get(
                f"{self.TAXONOMY_BASE_URL}/event",
                headers=headers,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                event_count = len(data.get("data", []))

                return ConnectionTestResult(
                    success=True,
                    message=f"Successfully connected to Amplitude. Found {event_count} event types.",
                    latency_ms=latency_ms,
                    api_version="v2",
                    permissions=["read:events", "read:cohorts", "read:charts"],
                    account_info={
                        "event_types_count": event_count,
                    },
                )
            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed. Check your API Key and Secret Key.",
                    latency_ms=latency_ms,
                    error_code="AUTH_FAILED",
                    error_details={"status_code": 401},
                )
            elif response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access denied. Your API key may not have sufficient permissions.",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                    error_details={"status_code": 403},
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
                message="Connection timed out. Amplitude API may be slow or unreachable.",
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
        """Sync data from Amplitude.

        Syncs the following resources:
        - events: Event types and their metrics
        - cohorts: Saved cohorts/segments
        - user_properties: User property definitions
        - charts: Saved charts and dashboards

        Args:
            credentials: IntegrationCredentials with api_key and api_secret
            resources: Optional list of resources to sync (defaults to all)
            since: Optional datetime for incremental sync (events only)
            limit: Maximum items per resource

        Returns:
            SyncResult with synced data
        """
        started_at = datetime.utcnow()
        sync_id = f"amplitude_{started_at.strftime('%Y%m%d_%H%M%S')}"

        resources_to_sync = resources or ["events", "cohorts", "user_properties"]
        synced_data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_points = 0

        headers = {
            "Authorization": self._get_auth_header(credentials),
            "Content-Type": "application/json",
        }

        # Sync events
        if "events" in resources_to_sync:
            events_data, events_count, events_errors = await self._sync_events(
                headers, since, limit
            )
            synced_data["events"] = events_data
            total_points += events_count
            errors.extend(events_errors)

        # Sync cohorts
        if "cohorts" in resources_to_sync:
            cohorts_data, cohorts_count, cohorts_errors = await self._sync_cohorts(
                headers, limit
            )
            synced_data["cohorts"] = cohorts_data
            total_points += cohorts_count
            errors.extend(cohorts_errors)

        # Sync user properties
        if "user_properties" in resources_to_sync:
            props_data, props_count, props_errors = await self._sync_user_properties(
                headers, limit
            )
            synced_data["user_properties"] = props_data
            total_points += props_count
            errors.extend(props_errors)

        # Sync charts (if requested)
        if "charts" in resources_to_sync:
            charts_data, charts_count, charts_errors = await self._sync_charts(
                headers, limit
            )
            synced_data["charts"] = charts_data
            total_points += charts_count
            errors.extend(charts_errors)

        completed_at = datetime.utcnow()

        return SyncResult(
            success=len(errors) == 0,
            message=(
                f"Synced {total_points} data points from Amplitude"
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
        since: Optional[datetime],
        limit: int,
    ) -> tuple[list[dict], int, list[str]]:
        """Sync event types and metrics from Amplitude."""
        events = []
        errors = []

        try:
            # Get event taxonomy
            response = await self.http_client.get(
                f"{self.TAXONOMY_BASE_URL}/event",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                event_types = data.get("data", [])[:limit]

                for event in event_types:
                    events.append({
                        "event_type": event.get("event_type"),
                        "display_name": event.get("display_name"),
                        "description": event.get("description"),
                        "category": event.get("category"),
                        "is_active": event.get("is_active", True),
                        "is_hidden": event.get("is_hidden", False),
                    })

                # If since is provided, get event metrics for the time range
                if since:
                    metrics = await self._get_event_metrics(headers, since, limit)
                    for event in events:
                        event_metrics = metrics.get(event["event_type"], {})
                        event.update(event_metrics)

            else:
                errors.append(f"Failed to fetch events: HTTP {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing events: {str(e)}")

        return events, len(events), errors

    async def _get_event_metrics(
        self,
        headers: dict[str, str],
        since: datetime,
        limit: int,
    ) -> dict[str, dict]:
        """Get event metrics using segmentation API."""
        metrics = {}

        try:
            end = datetime.utcnow()
            params = {
                "e": json.dumps({"event_type": "_all"}),
                "m": "uniques",
                "start": self._format_date(since),
                "end": self._format_date(end),
                "i": "1",
            }

            response = await self.http_client.get(
                f"{self.BASE_URL}/events/segmentation",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                series_data = data.get("data", {}).get("series", [])
                series_labels = data.get("data", {}).get("seriesLabels", [])

                for idx, label in enumerate(series_labels[:limit]):
                    if idx < len(series_data):
                        total_uniques = sum(v or 0 for v in series_data[idx])
                        metrics[label] = {
                            "unique_users": total_uniques,
                            "period_start": since.isoformat(),
                            "period_end": end.isoformat(),
                        }

        except Exception:
            # Silently fail for metrics - main event data is more important
            pass

        return metrics

    async def _sync_cohorts(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> tuple[list[dict], int, list[str]]:
        """Sync cohorts from Amplitude."""
        cohorts = []
        errors = []

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/cohorts",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                cohort_list = data.get("cohorts", [])[:limit]

                for cohort in cohort_list:
                    cohorts.append({
                        "id": cohort.get("id"),
                        "name": cohort.get("name"),
                        "description": cohort.get("description"),
                        "app_id": cohort.get("appId"),
                        "archived": cohort.get("archived", False),
                        "created_at": cohort.get("createdAt"),
                        "last_computed": cohort.get("lastComputed"),
                        "size": cohort.get("size"),
                        "type": cohort.get("type"),
                        "is_predictive": cohort.get("isPredictive", False),
                    })
            elif response.status_code == 404:
                # Cohorts endpoint may not be available for all plans
                errors.append("Cohorts API not available for this account")
            else:
                errors.append(f"Failed to fetch cohorts: HTTP {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing cohorts: {str(e)}")

        return cohorts, len(cohorts), errors

    async def _sync_user_properties(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> tuple[list[dict], int, list[str]]:
        """Sync user property definitions from Amplitude."""
        properties = []
        errors = []

        try:
            response = await self.http_client.get(
                f"{self.TAXONOMY_BASE_URL}/user-property",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                prop_list = data.get("data", [])[:limit]

                for prop in prop_list:
                    properties.append({
                        "user_property": prop.get("user_property"),
                        "display_name": prop.get("display_name"),
                        "description": prop.get("description"),
                        "type": prop.get("type"),
                        "is_active": prop.get("is_active", True),
                        "is_hidden": prop.get("is_hidden", False),
                    })
            else:
                errors.append(
                    f"Failed to fetch user properties: HTTP {response.status_code}"
                )

        except Exception as e:
            errors.append(f"Error syncing user properties: {str(e)}")

        return properties, len(properties), errors

    async def _sync_charts(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> tuple[list[dict], int, list[str]]:
        """Sync saved charts from Amplitude."""
        charts = []
        errors = []

        try:
            # Note: Charts API endpoint may vary by Amplitude plan
            response = await self.http_client.get(
                f"{self.BASE_URL}/charts",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                chart_list = data.get("charts", data.get("data", []))[:limit]

                for chart in chart_list:
                    charts.append({
                        "id": chart.get("id"),
                        "name": chart.get("name"),
                        "description": chart.get("description"),
                        "chart_type": chart.get("chartType", chart.get("type")),
                        "created_at": chart.get("createdAt"),
                        "updated_at": chart.get("updatedAt"),
                    })
            elif response.status_code == 404:
                # Charts endpoint may not be available
                pass  # Silently skip if not available
            else:
                errors.append(f"Failed to fetch charts: HTTP {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing charts: {str(e)}")

        return charts, len(charts), errors

    async def export_events(
        self,
        credentials: IntegrationCredentials,
        start: datetime,
        end: datetime,
    ) -> tuple[list[dict], list[str]]:
        """Export raw event data using the Export API.

        This is useful for bulk data retrieval for analysis.

        Args:
            credentials: IntegrationCredentials with api_key and api_secret
            start: Start datetime (hour granularity)
            end: End datetime (hour granularity)

        Returns:
            Tuple of (events list, errors list)
        """
        events = []
        errors = []

        headers = {
            "Authorization": self._get_auth_header(credentials),
        }

        try:
            params = {
                "start": self._format_datetime_hour(start),
                "end": self._format_datetime_hour(end),
            }

            response = await self.http_client.get(
                self.EXPORT_BASE_URL,
                headers=headers,
                params=params,
                timeout=120.0,  # Export can take longer
            )

            if response.status_code == 200:
                # Export API returns gzipped JSON lines
                # Each line is a separate event JSON object
                content = response.text
                for line in content.strip().split("\n"):
                    if line:
                        try:
                            event = json.loads(line)
                            events.append(event)
                        except json.JSONDecodeError:
                            continue
            elif response.status_code == 404:
                errors.append("No data available for the specified time range")
            else:
                errors.append(f"Export failed: HTTP {response.status_code}")

        except httpx.TimeoutException:
            errors.append("Export request timed out")
        except Exception as e:
            errors.append(f"Export error: {str(e)}")

        return events, errors

    async def get_chart_data(
        self,
        credentials: IntegrationCredentials,
        chart_id: str,
    ) -> tuple[Optional[dict], Optional[str]]:
        """Get data from a saved chart.

        Args:
            credentials: IntegrationCredentials with api_key and api_secret
            chart_id: The chart ID to fetch data for

        Returns:
            Tuple of (chart data dict, error message or None)
        """
        headers = {
            "Authorization": self._get_auth_header(credentials),
            "Content-Type": "application/json",
        }

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/charts/{chart_id}/query",
                headers=headers,
            )

            if response.status_code == 200:
                return response.json(), None
            elif response.status_code == 404:
                return None, f"Chart {chart_id} not found"
            else:
                return None, f"Failed to fetch chart data: HTTP {response.status_code}"

        except Exception as e:
            return None, f"Error fetching chart data: {str(e)}"
