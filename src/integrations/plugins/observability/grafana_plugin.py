"""Grafana integration plugin.

This plugin provides integration with Grafana for dashboard monitoring,
alerting, and observability data synchronization.

Grafana is an open-source analytics and monitoring solution that supports
multiple data sources and provides powerful visualization capabilities.
"""

import logging
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

logger = logging.getLogger(__name__)


class GrafanaPlugin(IntegrationPlugin):
    """Grafana integration plugin.

    Supports:
    - Dashboard management and sync
    - Alert rules and notifications
    - Annotations
    - Data source configuration

    Authentication:
    - API Key (Service Account Token) with Bearer authentication
    - Requires instance_url for self-hosted or cloud instances

    Example:
        plugin = GrafanaPlugin()
        credentials = IntegrationCredentials(
            api_key="glsa_xxxxxxxxxxxxxxxxxxxx",
            instance_url="https://your-org.grafana.net",
        )
        result = await plugin.test_connection(credentials)
    """

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Grafana integration metadata."""
        return IntegrationMetadata(
            slug="grafana",
            name="Grafana",
            description="Open-source analytics and monitoring platform with powerful dashboards and alerting",
            category=IntegrationCategory.OBSERVABILITY,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/grafana/grafana-original.svg",
            docs_url="https://grafana.com/docs/grafana/latest/developers/http_api/",
            website_url="https://grafana.com",
            features=[
                "dashboards",
                "alerts",
                "annotations",
                "datasources",
                "folders",
                "teams",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key / Service Account Token",
                    type="password",
                    required=True,
                    placeholder="glsa_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    help_text="Generate a Service Account Token from Grafana Administration > Service accounts",
                ),
                FieldDefinition(
                    name="instance_url",
                    label="Grafana Instance URL",
                    type="url",
                    required=True,
                    placeholder="https://your-org.grafana.net",
                    help_text="Your Grafana Cloud URL or self-hosted instance URL",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="org_slug",
                    label="Organization ID",
                    type="text",
                    required=False,
                    placeholder="1",
                    help_text="Organization ID (defaults to current org)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["dashboards", "alerts", "annotations", "datasources"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the base API URL from credentials."""
        instance_url = credentials.instance_url or ""
        # Remove trailing slash if present
        return instance_url.rstrip("/")

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get the request headers with authentication."""
        return {
            "Authorization": f"Bearer {credentials.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Grafana API.

        Uses GET /api/org to verify authentication and retrieve
        organization information.

        Args:
            credentials: Grafana credentials with api_key and instance_url.

        Returns:
            ConnectionTestResult with organization details on success.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        start_time = time.time()

        try:
            # Test connection using /api/org endpoint
            response = await self.http_client.get(
                f"{base_url}/api/org",
                headers=headers,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                org_data = response.json()

                # Get additional info about the user/token
                user_response = await self.http_client.get(
                    f"{base_url}/api/user",
                    headers=headers,
                )
                user_data = user_response.json() if user_response.status_code == 200 else {}

                # Get Grafana version info
                health_response = await self.http_client.get(
                    f"{base_url}/api/health",
                    headers=headers,
                )
                health_data = health_response.json() if health_response.status_code == 200 else {}

                return ConnectionTestResult(
                    success=True,
                    message=f"Connected to Grafana organization: {org_data.get('name', 'Unknown')}",
                    latency_ms=latency_ms,
                    api_version=health_data.get("version", "unknown"),
                    permissions=self._extract_permissions(user_data),
                    account_info={
                        "organization_id": org_data.get("id"),
                        "organization_name": org_data.get("name"),
                        "user_id": user_data.get("id"),
                        "user_name": user_data.get("name"),
                        "user_login": user_data.get("login"),
                        "user_email": user_data.get("email"),
                        "is_grafana_admin": user_data.get("isGrafanaAdmin", False),
                        "grafana_version": health_data.get("version"),
                        "database": health_data.get("database", "unknown"),
                    },
                )

            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed. Please check your API key.",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )

            elif response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden. The API key may lack required permissions.",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                    error_details={"status_code": 403},
                )

            else:
                return ConnectionTestResult(
                    success=False,
                    message=f"Unexpected response from Grafana: {response.status_code}",
                    latency_ms=latency_ms,
                    error_code="UNEXPECTED_RESPONSE",
                    error_details={
                        "status_code": response.status_code,
                        "response_body": response.text[:500],
                    },
                )

        except httpx.ConnectError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Failed to connect to Grafana at {base_url}. Please verify the URL.",
                error_code="CONNECTION_ERROR",
                error_details={"error": str(e)},
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection to Grafana timed out.",
                error_code="TIMEOUT",
            )

        except Exception as e:
            logger.exception("Unexpected error testing Grafana connection")
            return ConnectionTestResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                error_code="UNKNOWN_ERROR",
                error_details={"error": str(e)},
            )

    def _extract_permissions(self, user_data: dict) -> list[str]:
        """Extract permissions from user data."""
        permissions = []

        if user_data.get("isGrafanaAdmin"):
            permissions.append("grafana_admin")

        # Check organization role
        org_role = user_data.get("orgRole", "")
        if org_role:
            permissions.append(f"org:{org_role.lower()}")

        return permissions

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Grafana.

        Syncs dashboards, alerts, annotations, and datasources.

        Args:
            credentials: Grafana credentials.
            resources: Optional list of resources to sync.
                      Options: dashboards, alerts, annotations, datasources
            since: Optional datetime for incremental sync.
            limit: Maximum items to sync per resource.

        Returns:
            SyncResult with synced data.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        started_at = datetime.utcnow()
        sync_resources = resources or ["dashboards", "alerts", "annotations", "datasources"]

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        try:
            # Sync dashboards
            if "dashboards" in sync_resources:
                dashboards_result = await self._sync_dashboards(
                    base_url, headers, limit
                )
                data["dashboards"] = dashboards_result["data"]
                total_synced += dashboards_result["count"]
                errors.extend(dashboards_result.get("errors", []))

            # Sync alerts
            if "alerts" in sync_resources:
                alerts_result = await self._sync_alerts(base_url, headers, limit)
                data["alerts"] = alerts_result["data"]
                total_synced += alerts_result["count"]
                errors.extend(alerts_result.get("errors", []))

            # Sync annotations
            if "annotations" in sync_resources:
                annotations_result = await self._sync_annotations(
                    base_url, headers, since, limit
                )
                data["annotations"] = annotations_result["data"]
                total_synced += annotations_result["count"]
                errors.extend(annotations_result.get("errors", []))

            # Sync datasources
            if "datasources" in sync_resources:
                datasources_result = await self._sync_datasources(base_url, headers)
                data["datasources"] = datasources_result["data"]
                total_synced += datasources_result["count"]
                errors.extend(datasources_result.get("errors", []))

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Grafana"
                if len(errors) == 0
                else f"Synced {total_synced} items with {len(errors)} errors",
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            logger.exception("Error syncing Grafana data")
            return SyncResult(
                success=False,
                message=f"Failed to sync Grafana data: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                data=data,
            )

    async def _sync_dashboards(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync dashboards from Grafana."""
        dashboards = []
        errors = []

        try:
            # Search for all dashboards
            response = await self.http_client.get(
                f"{base_url}/api/search",
                headers=headers,
                params={"type": "dash-db", "limit": limit},
            )

            if response.status_code == 200:
                search_results = response.json()

                for item in search_results[:limit]:
                    # Get full dashboard details
                    uid = item.get("uid")
                    if uid:
                        detail_response = await self.http_client.get(
                            f"{base_url}/api/dashboards/uid/{uid}",
                            headers=headers,
                        )
                        if detail_response.status_code == 200:
                            detail = detail_response.json()
                            dashboards.append({
                                "uid": uid,
                                "id": item.get("id"),
                                "title": item.get("title"),
                                "url": item.get("url"),
                                "type": item.get("type"),
                                "tags": item.get("tags", []),
                                "folder_id": detail.get("meta", {}).get("folderId"),
                                "folder_title": detail.get("meta", {}).get("folderTitle"),
                                "created": detail.get("meta", {}).get("created"),
                                "updated": detail.get("meta", {}).get("updated"),
                                "version": detail.get("meta", {}).get("version"),
                                "panels_count": len(
                                    detail.get("dashboard", {}).get("panels", [])
                                ),
                            })
            else:
                errors.append(f"Failed to fetch dashboards: {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing dashboards: {str(e)}")

        return {"data": dashboards, "count": len(dashboards), "errors": errors}

    async def _sync_alerts(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync alerts from Grafana."""
        alerts = []
        errors = []

        try:
            # Get alert rules (Grafana 8+ unified alerting)
            response = await self.http_client.get(
                f"{base_url}/api/v1/provisioning/alert-rules",
                headers=headers,
            )

            if response.status_code == 200:
                alert_rules = response.json()
                for rule in alert_rules[:limit]:
                    alerts.append({
                        "uid": rule.get("uid"),
                        "title": rule.get("title"),
                        "condition": rule.get("condition"),
                        "folder_uid": rule.get("folderUID"),
                        "rule_group": rule.get("ruleGroup"),
                        "for": rule.get("for"),
                        "labels": rule.get("labels", {}),
                        "annotations": rule.get("annotations", {}),
                        "is_paused": rule.get("isPaused", False),
                    })
            elif response.status_code == 404:
                # Try legacy alerting API
                legacy_response = await self.http_client.get(
                    f"{base_url}/api/alerts",
                    headers=headers,
                    params={"limit": limit},
                )
                if legacy_response.status_code == 200:
                    legacy_alerts = legacy_response.json()
                    for alert in legacy_alerts:
                        alerts.append({
                            "id": alert.get("id"),
                            "dashboard_id": alert.get("dashboardId"),
                            "panel_id": alert.get("panelId"),
                            "name": alert.get("name"),
                            "state": alert.get("state"),
                            "new_state_date": alert.get("newStateDate"),
                            "eval_data": alert.get("evalData"),
                        })
            else:
                errors.append(f"Failed to fetch alerts: {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing alerts: {str(e)}")

        return {"data": alerts, "count": len(alerts), "errors": errors}

    async def _sync_annotations(
        self,
        base_url: str,
        headers: dict[str, str],
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync annotations from Grafana."""
        annotations = []
        errors = []

        try:
            params: dict[str, Any] = {"limit": limit}

            if since:
                # Convert datetime to epoch milliseconds
                params["from"] = int(since.timestamp() * 1000)

            response = await self.http_client.get(
                f"{base_url}/api/annotations",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                annotation_list = response.json()
                for annotation in annotation_list:
                    annotations.append({
                        "id": annotation.get("id"),
                        "alert_id": annotation.get("alertId"),
                        "dashboard_id": annotation.get("dashboardId"),
                        "panel_id": annotation.get("panelId"),
                        "text": annotation.get("text"),
                        "tags": annotation.get("tags", []),
                        "time": annotation.get("time"),
                        "time_end": annotation.get("timeEnd"),
                        "created": annotation.get("created"),
                        "updated": annotation.get("updated"),
                    })
            else:
                errors.append(f"Failed to fetch annotations: {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing annotations: {str(e)}")

        return {"data": annotations, "count": len(annotations), "errors": errors}

    async def _sync_datasources(
        self,
        base_url: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Sync datasources from Grafana."""
        datasources = []
        errors = []

        try:
            response = await self.http_client.get(
                f"{base_url}/api/datasources",
                headers=headers,
            )

            if response.status_code == 200:
                datasource_list = response.json()
                for ds in datasource_list:
                    datasources.append({
                        "id": ds.get("id"),
                        "uid": ds.get("uid"),
                        "name": ds.get("name"),
                        "type": ds.get("type"),
                        "type_name": ds.get("typeName"),
                        "access": ds.get("access"),
                        "url": ds.get("url"),
                        "is_default": ds.get("isDefault", False),
                        "database": ds.get("database"),
                        "read_only": ds.get("readOnly", False),
                    })
            else:
                errors.append(f"Failed to fetch datasources: {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing datasources: {str(e)}")

        return {"data": datasources, "count": len(datasources), "errors": errors}

    async def get_dashboard(
        self,
        credentials: IntegrationCredentials,
        uid: str,
    ) -> Optional[dict[str, Any]]:
        """Get a specific dashboard by UID.

        Args:
            credentials: Grafana credentials.
            uid: Dashboard UID.

        Returns:
            Dashboard data or None if not found.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        try:
            response = await self.http_client.get(
                f"{base_url}/api/dashboards/uid/{uid}",
                headers=headers,
            )

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"Error fetching dashboard {uid}: {e}")
            return None

    async def create_annotation(
        self,
        credentials: IntegrationCredentials,
        text: str,
        tags: Optional[list[str]] = None,
        dashboard_uid: Optional[str] = None,
        panel_id: Optional[int] = None,
        time_from: Optional[int] = None,
        time_to: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Create an annotation in Grafana.

        Args:
            credentials: Grafana credentials.
            text: Annotation text.
            tags: Optional list of tags.
            dashboard_uid: Optional dashboard UID.
            panel_id: Optional panel ID.
            time_from: Start time in epoch milliseconds (defaults to now).
            time_to: End time in epoch milliseconds.

        Returns:
            Created annotation data or None on failure.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        payload: dict[str, Any] = {
            "text": text,
            "tags": tags or [],
        }

        if dashboard_uid:
            payload["dashboardUID"] = dashboard_uid
        if panel_id:
            payload["panelId"] = panel_id
        if time_from:
            payload["time"] = time_from
        else:
            payload["time"] = int(datetime.utcnow().timestamp() * 1000)
        if time_to:
            payload["timeEnd"] = time_to

        try:
            response = await self.http_client.post(
                f"{base_url}/api/annotations",
                headers=headers,
                json=payload,
            )

            if response.status_code in (200, 201):
                return response.json()

            logger.error(f"Failed to create annotation: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error creating annotation: {e}")
            return None
