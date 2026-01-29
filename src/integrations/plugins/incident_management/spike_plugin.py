"""Spike.sh integration plugin for incident management.

Spike.sh is a modern incident management platform that helps teams respond to
incidents faster with intelligent alerting, on-call scheduling, and escalation policies.

API Documentation: https://docs.spike.sh/api
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


class SpikePlugin(IntegrationPlugin):
    """Integration plugin for Spike.sh incident management.

    Spike.sh provides incident alerting, on-call scheduling, and escalation
    management for DevOps and SRE teams.

    Features:
    - Incident management and alerting
    - On-call scheduling and rotations
    - Service catalog integration
    - Alert deduplication and grouping

    Authentication:
    - API Key authentication via X-API-Key header

    API Base URL: https://api.spike.sh
    """

    BASE_URL = "https://api.spike.sh"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Spike.sh integration metadata."""
        return IntegrationMetadata(
            slug="spike",
            name="Spike.sh",
            description="Modern incident management with intelligent alerting and on-call scheduling",
            category=IntegrationCategory.INCIDENT_MANAGEMENT,
            auth_type=AuthType.API_KEY,
            icon_url="https://spike.sh/favicon.ico",
            docs_url="https://docs.spike.sh/api",
            website_url="https://spike.sh",
            features=[
                "incidents",
                "alerts",
                "services",
                "on-call",
                "escalations",
                "integrations",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key",
                    type="password",
                    required=True,
                    placeholder="spike_api_key_...",
                    help_text="Generate an API key from Spike.sh Settings > API Keys",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="instance_url",
                    label="API Base URL",
                    type="url",
                    required=False,
                    placeholder="https://api.spike.sh",
                    help_text="Override the default API URL (for self-hosted instances)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["incidents", "alerts", "services"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the base URL for API requests."""
        return credentials.instance_url or self.BASE_URL

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "X-API-Key": credentials.api_key or "",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Spike.sh API.

        Tests the connection by fetching the incidents list, which validates
        that the API key is valid and has appropriate permissions.

        Args:
            credentials: IntegrationCredentials containing the API key.

        Returns:
            ConnectionTestResult with success status and account information.
        """
        start_time = time.time()

        try:
            base_url = self._get_base_url(credentials)
            headers = self._get_headers(credentials)

            # Test connection by fetching incidents (lightweight request)
            response = await self.http_client.get(
                f"{base_url}/v1/incidents",
                headers=headers,
                params={"limit": 1},
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to Spike.sh",
                    latency_ms=latency_ms,
                    api_version="v1",
                    permissions=["read:incidents", "read:alerts", "read:services"],
                    account_info={
                        "total_incidents": data.get("total", 0),
                    },
                )
            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed: Invalid API key",
                    latency_ms=latency_ms,
                    error_code="INVALID_API_KEY",
                    error_details={"status_code": 401},
                )
            elif response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Authorization failed: Insufficient permissions",
                    latency_ms=latency_ms,
                    error_code="INSUFFICIENT_PERMISSIONS",
                    error_details={"status_code": 403},
                )
            else:
                return ConnectionTestResult(
                    success=False,
                    message=f"Unexpected response: {response.status_code}",
                    latency_ms=latency_ms,
                    error_code="UNEXPECTED_RESPONSE",
                    error_details={
                        "status_code": response.status_code,
                        "body": response.text[:500],
                    },
                )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timeout: Spike.sh API is not responding",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="TIMEOUT",
            )
        except httpx.RequestError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="CONNECTION_ERROR",
                error_details={"exception": str(e)},
            )
        except Exception as e:
            logger.exception("Unexpected error testing Spike.sh connection")
            return ConnectionTestResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="UNEXPECTED_ERROR",
                error_details={"exception": str(e)},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Spike.sh.

        Syncs incidents, alerts, and services from Spike.sh.

        Args:
            credentials: IntegrationCredentials containing the API key.
            resources: Optional list of resources to sync (incidents, alerts, services).
                      If None, syncs all available resources.
            since: Optional datetime to sync data since (incremental sync).
            limit: Maximum number of items to sync per resource.

        Returns:
            SyncResult with synced data and statistics.
        """
        started_at = datetime.utcnow()
        sync_resources = resources or ["incidents", "alerts", "services"]
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        try:
            # Sync incidents
            if "incidents" in sync_resources:
                incidents_result = await self._sync_incidents(
                    base_url, headers, since, limit
                )
                if incidents_result["success"]:
                    data["incidents"] = incidents_result["data"]
                    total_synced += len(incidents_result["data"])
                else:
                    errors.append(f"Failed to sync incidents: {incidents_result['error']}")

            # Sync alerts
            if "alerts" in sync_resources:
                alerts_result = await self._sync_alerts(
                    base_url, headers, since, limit
                )
                if alerts_result["success"]:
                    data["alerts"] = alerts_result["data"]
                    total_synced += len(alerts_result["data"])
                else:
                    errors.append(f"Failed to sync alerts: {alerts_result['error']}")

            # Sync services
            if "services" in sync_resources:
                services_result = await self._sync_services(base_url, headers, limit)
                if services_result["success"]:
                    data["services"] = services_result["data"]
                    total_synced += len(services_result["data"])
                else:
                    errors.append(f"Failed to sync services: {services_result['error']}")

            completed_at = datetime.utcnow()

            return SyncResult(
                success=len(errors) == 0,
                message=(
                    f"Successfully synced {total_synced} items from Spike.sh"
                    if len(errors) == 0
                    else f"Sync completed with {len(errors)} errors"
                ),
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=completed_at,
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            logger.exception("Unexpected error syncing data from Spike.sh")
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

    async def _sync_incidents(
        self,
        base_url: str,
        headers: dict[str, str],
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync incidents from Spike.sh."""
        try:
            params: dict[str, Any] = {"limit": min(limit, 100)}
            if since:
                params["created_after"] = since.isoformat()

            response = await self.http_client.get(
                f"{base_url}/v1/incidents",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            incidents = data.get("incidents", data.get("data", []))
            if isinstance(incidents, list):
                return {
                    "success": True,
                    "data": [self._normalize_incident(inc) for inc in incidents],
                }
            return {"success": True, "data": []}

        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}", "data": []}
        except Exception as e:
            return {"success": False, "error": str(e), "data": []}

    async def _sync_alerts(
        self,
        base_url: str,
        headers: dict[str, str],
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync alerts from Spike.sh."""
        try:
            params: dict[str, Any] = {"limit": min(limit, 100)}
            if since:
                params["created_after"] = since.isoformat()

            response = await self.http_client.get(
                f"{base_url}/v1/alerts",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            alerts = data.get("alerts", data.get("data", []))
            if isinstance(alerts, list):
                return {
                    "success": True,
                    "data": [self._normalize_alert(alert) for alert in alerts],
                }
            return {"success": True, "data": []}

        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}", "data": []}
        except Exception as e:
            return {"success": False, "error": str(e), "data": []}

    async def _sync_services(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync services from Spike.sh."""
        try:
            params: dict[str, Any] = {"limit": min(limit, 100)}

            response = await self.http_client.get(
                f"{base_url}/v1/services",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            services = data.get("services", data.get("data", []))
            if isinstance(services, list):
                return {
                    "success": True,
                    "data": [self._normalize_service(svc) for svc in services],
                }
            return {"success": True, "data": []}

        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}", "data": []}
        except Exception as e:
            return {"success": False, "error": str(e), "data": []}

    def _normalize_incident(self, incident: dict[str, Any]) -> dict[str, Any]:
        """Normalize incident data to a standard format."""
        return {
            "id": incident.get("id") or incident.get("_id"),
            "title": incident.get("title") or incident.get("message"),
            "description": incident.get("description") or incident.get("details"),
            "status": incident.get("status"),
            "severity": incident.get("severity") or incident.get("priority"),
            "service_id": incident.get("service_id") or incident.get("serviceId"),
            "service_name": incident.get("service_name") or incident.get("serviceName"),
            "created_at": incident.get("created_at") or incident.get("createdAt"),
            "updated_at": incident.get("updated_at") or incident.get("updatedAt"),
            "resolved_at": incident.get("resolved_at") or incident.get("resolvedAt"),
            "acknowledged_at": incident.get("acknowledged_at") or incident.get("acknowledgedAt"),
            "assigned_to": incident.get("assigned_to") or incident.get("assignedTo"),
            "escalation_policy": incident.get("escalation_policy") or incident.get("escalationPolicy"),
            "alert_count": incident.get("alert_count") or incident.get("alertCount", 0),
            "source": "spike",
            "raw": incident,
        }

    def _normalize_alert(self, alert: dict[str, Any]) -> dict[str, Any]:
        """Normalize alert data to a standard format."""
        return {
            "id": alert.get("id") or alert.get("_id"),
            "title": alert.get("title") or alert.get("message"),
            "description": alert.get("description") or alert.get("details"),
            "status": alert.get("status"),
            "severity": alert.get("severity") or alert.get("priority"),
            "incident_id": alert.get("incident_id") or alert.get("incidentId"),
            "service_id": alert.get("service_id") or alert.get("serviceId"),
            "created_at": alert.get("created_at") or alert.get("createdAt"),
            "acknowledged_at": alert.get("acknowledged_at") or alert.get("acknowledgedAt"),
            "resolved_at": alert.get("resolved_at") or alert.get("resolvedAt"),
            "integration": alert.get("integration") or alert.get("source"),
            "dedup_key": alert.get("dedup_key") or alert.get("dedupKey"),
            "source": "spike",
            "raw": alert,
        }

    def _normalize_service(self, service: dict[str, Any]) -> dict[str, Any]:
        """Normalize service data to a standard format."""
        return {
            "id": service.get("id") or service.get("_id"),
            "name": service.get("name"),
            "description": service.get("description"),
            "status": service.get("status"),
            "escalation_policy_id": (
                service.get("escalation_policy_id") or service.get("escalationPolicyId")
            ),
            "created_at": service.get("created_at") or service.get("createdAt"),
            "updated_at": service.get("updated_at") or service.get("updatedAt"),
            "team_id": service.get("team_id") or service.get("teamId"),
            "integration_keys": service.get("integration_keys") or service.get("integrationKeys", []),
            "source": "spike",
            "raw": service,
        }

    async def create_incident(
        self,
        credentials: IntegrationCredentials,
        title: str,
        description: str,
        severity: str = "critical",
        service_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new incident in Spike.sh.

        Args:
            credentials: IntegrationCredentials containing the API key.
            title: Incident title.
            description: Incident description.
            severity: Severity level (critical, high, medium, low).
            service_id: Optional service ID to associate the incident with.

        Returns:
            Created incident data or error information.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        payload: dict[str, Any] = {
            "title": title,
            "description": description,
            "severity": severity,
        }
        if service_id:
            payload["service_id"] = service_id

        try:
            response = await self.http_client.post(
                f"{base_url}/v1/incidents",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return {"success": True, "incident": response.json()}
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def acknowledge_incident(
        self,
        credentials: IntegrationCredentials,
        incident_id: str,
    ) -> dict[str, Any]:
        """Acknowledge an incident in Spike.sh.

        Args:
            credentials: IntegrationCredentials containing the API key.
            incident_id: ID of the incident to acknowledge.

        Returns:
            Updated incident data or error information.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        try:
            response = await self.http_client.post(
                f"{base_url}/v1/incidents/{incident_id}/acknowledge",
                headers=headers,
            )
            response.raise_for_status()
            return {"success": True, "incident": response.json()}
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def resolve_incident(
        self,
        credentials: IntegrationCredentials,
        incident_id: str,
        resolution_note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Resolve an incident in Spike.sh.

        Args:
            credentials: IntegrationCredentials containing the API key.
            incident_id: ID of the incident to resolve.
            resolution_note: Optional note about the resolution.

        Returns:
            Updated incident data or error information.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        payload: dict[str, Any] = {}
        if resolution_note:
            payload["resolution_note"] = resolution_note

        try:
            response = await self.http_client.post(
                f"{base_url}/v1/incidents/{incident_id}/resolve",
                headers=headers,
                json=payload if payload else None,
            )
            response.raise_for_status()
            return {"success": True, "incident": response.json()}
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
