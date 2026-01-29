"""Datadog integration plugin.

This module provides integration with Datadog for infrastructure monitoring,
APM, logs, and alerting. Datadog is a comprehensive monitoring and security
platform for cloud applications.

Features:
- Fetch monitors and their states
- Query metrics and time-series data
- Sync events and incidents
- Support for multiple Datadog sites (US, EU, etc.)
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


class DatadogPlugin(IntegrationPlugin):
    """Datadog integration for infrastructure and application monitoring.

    Datadog provides comprehensive monitoring for cloud-scale applications,
    including infrastructure monitoring, APM, logs, and security.

    Authentication:
        Uses API Key + Application Key authentication.
        - API Key: For sending data to Datadog (DD-API-KEY header)
        - Application Key: For reading data from Datadog (DD-APPLICATION-KEY header)

        Keys can be created at: https://app.datadoghq.com/organization-settings/api-keys

    Supported Sites:
        - datadoghq.com (US1 - default)
        - us3.datadoghq.com (US3)
        - us5.datadoghq.com (US5)
        - datadoghq.eu (EU1)
        - ap1.datadoghq.com (AP1)
        - ddog-gov.com (US1-FED)

    Example:
        async with DatadogPlugin() as plugin:
            credentials = IntegrationCredentials(
                api_key="your-api-key",
                api_secret="your-application-key",
                site="datadoghq.com",
            )
            result = await plugin.test_connection(credentials)
            if result.success:
                data = await plugin.sync_data(credentials)
    """

    # Site to base URL mapping
    SITE_URLS = {
        "datadoghq.com": "https://api.datadoghq.com",
        "us3.datadoghq.com": "https://api.us3.datadoghq.com",
        "us5.datadoghq.com": "https://api.us5.datadoghq.com",
        "datadoghq.eu": "https://api.datadoghq.eu",
        "ap1.datadoghq.com": "https://api.ap1.datadoghq.com",
        "ddog-gov.com": "https://api.ddog-gov.com",
    }

    DEFAULT_SITE = "datadoghq.com"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Datadog integration metadata."""
        return IntegrationMetadata(
            slug="datadog",
            name="Datadog",
            description="Cloud monitoring and security platform for infrastructure and applications",
            category=IntegrationCategory.OBSERVABILITY,
            auth_type=AuthType.API_KEY,
            icon_url="https://www.datadoghq.com/favicon.ico",
            docs_url="https://docs.datadoghq.com/api/",
            website_url="https://www.datadoghq.com",
            features=[
                "infrastructure_monitoring",
                "apm",
                "log_management",
                "monitors_alerts",
                "dashboards",
                "metrics",
                "events",
                "incidents",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key",
                    type="password",
                    required=True,
                    placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    help_text="Datadog API Key. Create at Organization Settings > API Keys",
                ),
                FieldDefinition(
                    name="api_secret",
                    label="Application Key",
                    type="password",
                    required=True,
                    placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    help_text="Datadog Application Key. Create at Organization Settings > Application Keys",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="site",
                    label="Datadog Site",
                    type="select",
                    required=False,
                    placeholder="datadoghq.com",
                    help_text="Your Datadog site (check your browser URL)",
                    options=[
                        {"value": "datadoghq.com", "label": "US1 (datadoghq.com)"},
                        {"value": "us3.datadoghq.com", "label": "US3 (us3.datadoghq.com)"},
                        {"value": "us5.datadoghq.com", "label": "US5 (us5.datadoghq.com)"},
                        {"value": "datadoghq.eu", "label": "EU1 (datadoghq.eu)"},
                        {"value": "ap1.datadoghq.com", "label": "AP1 (ap1.datadoghq.com)"},
                        {"value": "ddog-gov.com", "label": "US1-FED (ddog-gov.com)"},
                    ],
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["monitors", "events", "metrics", "dashboards", "service_level_objectives"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the base URL for API calls based on site."""
        site = credentials.site or self.DEFAULT_SITE
        return self.SITE_URLS.get(site, self.SITE_URLS[self.DEFAULT_SITE])

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "DD-API-KEY": credentials.api_key or "",
            "DD-APPLICATION-KEY": credentials.api_secret or "",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the Datadog connection.

        Validates API and Application keys using the validate endpoint.

        Args:
            credentials: Must contain api_key and api_secret (application key).

        Returns:
            ConnectionTestResult with validation status.
        """
        is_valid, missing = self.validate_credentials(credentials)
        if not is_valid:
            return ConnectionTestResult(
                success=False,
                message=f"Missing required fields: {', '.join(missing)}",
                error_code="MISSING_CREDENTIALS",
            )

        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)
        url = f"{base_url}/api/v1/validate"

        start_time = time.time()
        try:
            response = await self.http_client.get(url, headers=headers)
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 403:
                result = response.json()
                return ConnectionTestResult(
                    success=False,
                    message=f"Invalid credentials: {result.get('errors', ['Unknown error'])[0]}",
                    latency_ms=latency_ms,
                    error_code="INVALID_CREDENTIALS",
                )

            response.raise_for_status()
            result = response.json()

            # Get additional organization info
            org_info = {}
            try:
                org_url = f"{base_url}/api/v1/org"
                org_response = await self.http_client.get(org_url, headers=headers)
                if org_response.status_code == 200:
                    org_data = org_response.json()
                    org = org_data.get("org", {})
                    org_info = {
                        "organization_name": org.get("name"),
                        "organization_public_id": org.get("public_id"),
                        "subscription_plan": org.get("subscription", {}).get("type"),
                    }
            except Exception:
                pass

            return ConnectionTestResult(
                success=result.get("valid", False),
                message="API key and Application key are valid"
                if result.get("valid")
                else "Validation failed",
                latency_ms=latency_ms,
                api_version="v1",
                permissions=[],  # Datadog doesn't expose scopes via API
                account_info={
                    "site": credentials.site or self.DEFAULT_SITE,
                    **org_info,
                },
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except httpx.HTTPStatusError as e:
            return ConnectionTestResult(
                success=False,
                message=f"HTTP error: {e.response.status_code}",
                error_code=f"HTTP_{e.response.status_code}",
                error_details={"response": e.response.text},
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection failed: {str(e)}",
                error_code="CONNECTION_ERROR",
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Datadog.

        Fetches monitors, events, metrics summaries, and dashboards.

        Args:
            credentials: Must contain api_key and api_secret.
            resources: List of resources to sync: "monitors", "events", "metrics", "dashboards", "service_level_objectives".
            since: Only fetch data since this datetime (for events).
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced data organized by resource type.
        """
        started_at = datetime.utcnow()
        sync_resources = resources or self.metadata.sync_resources
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        # Sync monitors
        if "monitors" in sync_resources:
            try:
                monitors_data = await self._sync_monitors(
                    base_url, headers, limit
                )
                data["monitors"] = monitors_data
                total_synced += len(monitors_data)
            except Exception as e:
                errors.append(f"Failed to sync monitors: {str(e)}")

        # Sync events
        if "events" in sync_resources:
            try:
                events_data = await self._sync_events(
                    base_url, headers, since, limit
                )
                data["events"] = events_data
                total_synced += len(events_data)
            except Exception as e:
                errors.append(f"Failed to sync events: {str(e)}")

        # Sync metrics (list available metrics)
        if "metrics" in sync_resources:
            try:
                metrics_data = await self._sync_metrics(
                    base_url, headers, since, limit
                )
                data["metrics"] = metrics_data
                total_synced += len(metrics_data)
            except Exception as e:
                errors.append(f"Failed to sync metrics: {str(e)}")

        # Sync dashboards
        if "dashboards" in sync_resources:
            try:
                dashboards_data = await self._sync_dashboards(
                    base_url, headers, limit
                )
                data["dashboards"] = dashboards_data
                total_synced += len(dashboards_data)
            except Exception as e:
                errors.append(f"Failed to sync dashboards: {str(e)}")

        # Sync SLOs
        if "service_level_objectives" in sync_resources:
            try:
                slos_data = await self._sync_slos(
                    base_url, headers, limit
                )
                data["service_level_objectives"] = slos_data
                total_synced += len(slos_data)
            except Exception as e:
                errors.append(f"Failed to sync SLOs: {str(e)}")

        completed_at = datetime.utcnow()

        return SyncResult(
            success=len(errors) == 0,
            message=f"Synced {total_synced} items from Datadog"
            if len(errors) == 0
            else f"Sync completed with {len(errors)} error(s)",
            data_points_synced=total_synced,
            started_at=started_at,
            completed_at=completed_at,
            errors=errors,
            warnings=warnings,
            data=data,
        )

    async def _sync_monitors(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync monitors from Datadog."""
        url = f"{base_url}/api/v1/monitor"

        params: dict[str, Any] = {
            "page_size": min(limit, 100),
        }

        response = await self.http_client.get(url, headers=headers, params=params)
        response.raise_for_status()

        monitors = response.json()
        return [
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "type": m.get("type"),
                "query": m.get("query"),
                "message": m.get("message"),
                "overall_state": m.get("overall_state"),
                "state": {
                    "status": m.get("state", {}).get("overall_state"),
                    "groups": m.get("state", {}).get("groups", {}),
                },
                "tags": m.get("tags", []),
                "priority": m.get("priority"),
                "created": m.get("created"),
                "created_at": m.get("created"),
                "modified": m.get("modified"),
                "options": {
                    "thresholds": m.get("options", {}).get("thresholds", {}),
                    "notify_no_data": m.get("options", {}).get("notify_no_data", False),
                    "renotify_interval": m.get("options", {}).get("renotify_interval"),
                    "escalation_message": m.get("options", {}).get("escalation_message"),
                },
            }
            for m in monitors[:limit]
        ]

    async def _sync_events(
        self,
        base_url: str,
        headers: dict[str, str],
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync events from Datadog."""
        url = f"{base_url}/api/v1/events"

        # Default to last 24 hours if no since specified
        if since:
            start_time = int(since.timestamp())
        else:
            start_time = int((datetime.utcnow().timestamp()) - 86400)  # 24 hours ago

        end_time = int(datetime.utcnow().timestamp())

        params: dict[str, Any] = {
            "start": start_time,
            "end": end_time,
        }

        response = await self.http_client.get(url, headers=headers, params=params)
        response.raise_for_status()

        result = response.json()
        events = result.get("events", [])

        return [
            {
                "id": e.get("id"),
                "title": e.get("title"),
                "text": e.get("text"),
                "date_happened": e.get("date_happened"),
                "date_happened_iso": datetime.fromtimestamp(
                    e.get("date_happened", 0)
                ).isoformat()
                if e.get("date_happened")
                else None,
                "priority": e.get("priority"),
                "alert_type": e.get("alert_type"),
                "source_type_name": e.get("source_type_name"),
                "host": e.get("host"),
                "device_name": e.get("device_name"),
                "tags": e.get("tags", []),
                "is_aggregate": e.get("is_aggregate", False),
                "url": e.get("url"),
            }
            for e in events[:limit]
        ]

    async def _sync_metrics(
        self,
        base_url: str,
        headers: dict[str, str],
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync available metrics from Datadog."""
        url = f"{base_url}/api/v1/metrics"

        # Get metric names from a time window
        if since:
            from_time = int(since.timestamp())
        else:
            from_time = int((datetime.utcnow().timestamp()) - 3600)  # Last hour

        params: dict[str, Any] = {
            "from": from_time,
        }

        response = await self.http_client.get(url, headers=headers, params=params)
        response.raise_for_status()

        result = response.json()
        metrics = result.get("metrics", [])

        # Return just the metric names with metadata where available
        metrics_data: list[dict[str, Any]] = []
        for metric_name in metrics[:limit]:
            metrics_data.append(
                {
                    "name": metric_name,
                    "type": "gauge",  # Default, actual type requires another API call
                }
            )

        return metrics_data

    async def _sync_dashboards(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync dashboards from Datadog."""
        url = f"{base_url}/api/v1/dashboard"

        response = await self.http_client.get(url, headers=headers)
        response.raise_for_status()

        result = response.json()
        dashboards = result.get("dashboards", [])

        return [
            {
                "id": d.get("id"),
                "title": d.get("title"),
                "description": d.get("description"),
                "layout_type": d.get("layout_type"),
                "author_handle": d.get("author_handle"),
                "author_name": d.get("author_name"),
                "created_at": d.get("created_at"),
                "modified_at": d.get("modified_at"),
                "is_read_only": d.get("is_read_only", False),
                "url": d.get("url"),
            }
            for d in dashboards[:limit]
        ]

    async def _sync_slos(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync Service Level Objectives from Datadog."""
        url = f"{base_url}/api/v1/slo"

        params: dict[str, Any] = {
            "limit": min(limit, 1000),
        }

        response = await self.http_client.get(url, headers=headers, params=params)
        response.raise_for_status()

        result = response.json()
        slos = result.get("data", [])

        return [
            {
                "id": s.get("id"),
                "name": s.get("name"),
                "description": s.get("description"),
                "type": s.get("type"),
                "tags": s.get("tags", []),
                "creator": s.get("creator", {}).get("handle"),
                "thresholds": s.get("thresholds", []),
                "target_threshold": s.get("target_threshold"),
                "timeframe": s.get("timeframe"),
                "created_at": s.get("created_at"),
                "modified_at": s.get("modified_at"),
            }
            for s in slos[:limit]
        ]

    async def query_metrics(
        self,
        credentials: IntegrationCredentials,
        query: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Query time-series metrics from Datadog.

        Args:
            credentials: Datadog credentials.
            query: Datadog metrics query string (e.g., "avg:system.cpu.user{*}").
            start: Start time for the query.
            end: End time for the query. Defaults to now.

        Returns:
            Time-series data for the query.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)
        url = f"{base_url}/api/v1/query"

        end_time = end or datetime.utcnow()

        params = {
            "from": int(start.timestamp()),
            "to": int(end_time.timestamp()),
            "query": query,
        }

        response = await self.http_client.get(url, headers=headers, params=params)
        response.raise_for_status()

        result = response.json()

        # Process the series data
        series_data = []
        for series in result.get("series", []):
            series_data.append(
                {
                    "metric": series.get("metric"),
                    "display_name": series.get("display_name"),
                    "unit": series.get("unit"),
                    "scope": series.get("scope"),
                    "expression": series.get("expression"),
                    "start": series.get("start"),
                    "end": series.get("end"),
                    "interval": series.get("interval"),
                    "pointlist": series.get("pointlist", []),  # [[timestamp, value], ...]
                    "length": series.get("length"),
                }
            )

        return {
            "status": result.get("status"),
            "query": result.get("query"),
            "from_date": result.get("from_date"),
            "to_date": result.get("to_date"),
            "series": series_data,
        }

    async def get_monitor_details(
        self,
        credentials: IntegrationCredentials,
        monitor_id: int,
    ) -> dict[str, Any]:
        """Get detailed information about a specific monitor.

        Args:
            credentials: Datadog credentials.
            monitor_id: The monitor ID.

        Returns:
            Detailed monitor data including thresholds and notification settings.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)
        url = f"{base_url}/api/v1/monitor/{monitor_id}"

        response = await self.http_client.get(url, headers=headers)
        response.raise_for_status()

        return response.json()

    async def search_hosts(
        self,
        credentials: IntegrationCredentials,
        filter_query: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search for hosts in Datadog.

        Args:
            credentials: Datadog credentials.
            filter_query: Optional filter string.
            limit: Maximum number of hosts to return.

        Returns:
            List of hosts matching the filter.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)
        url = f"{base_url}/api/v1/hosts"

        params: dict[str, Any] = {
            "count": min(limit, 1000),
            "include_muted_hosts_data": True,
            "include_hosts_metadata": True,
        }

        if filter_query:
            params["filter"] = filter_query

        response = await self.http_client.get(url, headers=headers, params=params)
        response.raise_for_status()

        result = response.json()
        hosts = result.get("host_list", [])

        return [
            {
                "name": h.get("name"),
                "id": h.get("id"),
                "host_name": h.get("host_name"),
                "up": h.get("up"),
                "is_muted": h.get("is_muted", False),
                "mute_timeout": h.get("mute_timeout"),
                "sources": h.get("sources", []),
                "apps": h.get("apps", []),
                "tags_by_source": h.get("tags_by_source", {}),
                "metrics": h.get("metrics", {}),
                "last_reported_time": h.get("last_reported_time"),
                "meta": h.get("meta", {}),
            }
            for h in hosts
        ]

    def parse_webhook_payload(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Parse and normalize a Datadog webhook payload.

        Datadog webhooks are typically configured in monitor notifications.

        Args:
            payload: The webhook payload from Datadog.
            headers: Request headers.

        Returns:
            Parsed event data with normalized structure.
        """
        # Datadog sends monitor alerts with custom payload templates
        # Common fields in default template
        alert_type = payload.get("alert_type", "")
        event_type = payload.get("event_type", "")

        # Normalize to common event type
        normalized_type = "datadog.alert"
        if "recovered" in alert_type.lower():
            normalized_type = "datadog.alert.recovered"
        elif "triggered" in alert_type.lower() or "alert" in alert_type.lower():
            normalized_type = "datadog.alert.triggered"
        elif "warn" in alert_type.lower():
            normalized_type = "datadog.alert.warning"
        elif "no_data" in alert_type.lower():
            normalized_type = "datadog.alert.no_data"

        return {
            "event_type": normalized_type,
            "timestamp": payload.get("date"),
            "alert_id": payload.get("alert_id"),
            "alert_type": alert_type,
            "event_title": payload.get("event_title"),
            "event_message": payload.get("event_message"),
            "monitor": {
                "id": payload.get("id"),
                "name": payload.get("monitor_name") or payload.get("title"),
                "type": payload.get("alert_type"),
                "query": payload.get("alert_query"),
            },
            "host": payload.get("host"),
            "hostname": payload.get("hostname"),
            "tags": payload.get("tags", "").split(",") if payload.get("tags") else [],
            "link": payload.get("link"),
            "snapshot": payload.get("snapshot"),
            "logs_sample": payload.get("logs_sample"),
            "raw_payload": payload,
        }
