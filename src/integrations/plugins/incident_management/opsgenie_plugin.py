"""Opsgenie integration plugin.

This module provides integration with Opsgenie for alert management,
incident response, and on-call scheduling.

API Documentation:
- REST API: https://docs.opsgenie.com/docs/api-overview
- Alert API: https://docs.opsgenie.com/docs/alert-api
- Incident API: https://docs.opsgenie.com/docs/incident-api
"""

import time
import uuid
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


class OpsgeniePlugin(IntegrationPlugin):
    """Opsgenie integration plugin.

    Provides access to:
    - Alerts: Create, list, acknowledge, close alerts
    - Incidents: Create, list, resolve incidents
    - Schedules: View on-call schedules
    - Users: List and manage users
    - Teams: List teams and their members
    - Escalations: View escalation policies
    """

    # Base URLs - Opsgenie has different URLs for US and EU
    US_API_URL = "https://api.opsgenie.com"
    EU_API_URL = "https://api.eu.opsgenie.com"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get metadata about this integration."""
        return IntegrationMetadata(
            slug="opsgenie",
            name="Opsgenie",
            description="Alert management, incident response, and on-call scheduling by Atlassian",
            category=IntegrationCategory.INCIDENT_MANAGEMENT,
            auth_type=AuthType.API_KEY,
            icon_url="https://www.atlassian.com/favicon.ico",
            docs_url="https://docs.opsgenie.com/docs/api-overview",
            website_url="https://www.atlassian.com/software/opsgenie",
            features=[
                "alerts",
                "incidents",
                "schedules",
                "users",
                "teams",
                "escalations",
                "integrations",
                "heartbeats",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key (GenieKey)",
                    type="password",
                    required=True,
                    placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    help_text="Generate from Settings > API Key Management in Opsgenie",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra.region",
                    label="Region",
                    type="select",
                    required=False,
                    options=[
                        {"value": "us", "label": "US (api.opsgenie.com)"},
                        {"value": "eu", "label": "EU (api.eu.opsgenie.com)"},
                    ],
                    help_text="Select your Opsgenie region (US or EU)",
                ),
                FieldDefinition(
                    name="extra.integration_api_key",
                    label="Integration API Key",
                    type="password",
                    required=False,
                    placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    help_text="Integration-specific API key for creating alerts",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["alerts", "incidents", "users", "schedules", "teams"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the appropriate base URL based on region."""
        region = credentials.extra.get("region", "us").lower()
        return self.EU_API_URL if region == "eu" else self.US_API_URL

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"GenieKey {credentials.api_key}",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the Opsgenie connection by getting account info.

        Uses GET /v2/account endpoint to verify API key validity.
        """
        start_time = time.time()
        base_url = self._get_base_url(credentials)

        try:
            response = await self.http_client.get(
                f"{base_url}/v2/account",
                headers=self._get_headers(credentials),
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json().get("data", {})

                return ConnectionTestResult(
                    success=True,
                    message=f"Connected to {data.get('name', 'Unknown')} account",
                    latency_ms=latency_ms,
                    api_version="2",
                    permissions=self._extract_permissions(data),
                    account_info={
                        "account_name": data.get("name"),
                        "plan": data.get("plan", {}).get("name"),
                        "user_count": data.get("userCount"),
                        "is_limited": data.get("plan", {}).get("isYearly", False),
                    },
                )
            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid API key (GenieKey)",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status": 401},
                )
            elif response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="API key lacks permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                    error_details={"status": 403},
                )
            else:
                error_data = response.json() if response.content else {}
                return ConnectionTestResult(
                    success=False,
                    message=f"API error: {response.status_code}",
                    latency_ms=latency_ms,
                    error_code=str(response.status_code),
                    error_details=error_data,
                )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except httpx.ConnectError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Failed to connect: {str(e)}",
                error_code="CONNECTION_ERROR",
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                error_code="UNKNOWN_ERROR",
            )

    def _extract_permissions(self, account: dict) -> list[str]:
        """Extract permissions from account data."""
        permissions = ["read"]

        plan = account.get("plan", {}).get("name", "").lower()
        if plan in ["enterprise", "team"]:
            permissions.extend(["write", "delete", "admin"])
        elif plan in ["standard", "essentials"]:
            permissions.append("write")

        return permissions

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Opsgenie.

        Supports syncing:
        - alerts: Open and closed alerts
        - incidents: Active and resolved incidents
        - users: Account users
        - schedules: On-call schedules
        - teams: Teams and members
        """
        start_time = datetime.utcnow()
        sync_id = str(uuid.uuid4())

        if resources is None:
            resources = ["alerts", "incidents", "users", "schedules"]

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        for resource in resources:
            try:
                if resource == "alerts":
                    result = await self._sync_alerts(base_url, headers, since, limit)
                    data["alerts"] = result
                    total_synced += len(result)
                elif resource == "incidents":
                    result = await self._sync_incidents(base_url, headers, since, limit)
                    data["incidents"] = result
                    total_synced += len(result)
                elif resource == "users":
                    result = await self._sync_users(base_url, headers, limit)
                    data["users"] = result
                    total_synced += len(result)
                elif resource == "schedules":
                    result = await self._sync_schedules(base_url, headers, limit)
                    data["schedules"] = result
                    total_synced += len(result)
                elif resource == "teams":
                    result = await self._sync_teams(base_url, headers, limit)
                    data["teams"] = result
                    total_synced += len(result)
                else:
                    warnings.append(f"Unknown resource type: {resource}")
            except httpx.HTTPStatusError as e:
                errors.append(f"Failed to sync {resource}: HTTP {e.response.status_code}")
            except Exception as e:
                errors.append(f"Failed to sync {resource}: {str(e)}")

        return SyncResult(
            success=len(errors) == 0,
            message=f"Synced {total_synced} items from Opsgenie"
            if len(errors) == 0
            else f"Sync completed with {len(errors)} errors",
            data_points_synced=total_synced,
            sync_id=sync_id,
            started_at=start_time,
            completed_at=datetime.utcnow(),
            errors=errors,
            warnings=warnings,
            data=data,
        )

    async def _sync_alerts(
        self,
        base_url: str,
        headers: dict[str, str],
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync alerts from Opsgenie."""
        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "sort": "createdAt",
            "order": "desc",
        }

        if since:
            # Convert to milliseconds timestamp
            params["query"] = f"createdAt >= {int(since.timestamp() * 1000)}"

        alerts = []
        offset = 0

        while len(alerts) < limit:
            params["offset"] = offset
            response = await self.http_client.get(
                f"{base_url}/v2/alerts",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            batch = data.get("data", [])
            if not batch:
                break

            for alert in batch:
                alerts.append(self._transform_alert(alert))
                if len(alerts) >= limit:
                    break

            # Check for pagination
            paging = data.get("paging", {})
            if not paging.get("next"):
                break

            offset += len(batch)

        return alerts

    def _transform_alert(self, alert: dict) -> dict[str, Any]:
        """Transform Opsgenie alert to standard format."""
        return {
            "id": alert.get("id"),
            "tiny_id": alert.get("tinyId"),
            "alias": alert.get("alias"),
            "message": alert.get("message"),
            "description": alert.get("description"),
            "status": alert.get("status"),
            "acknowledged": alert.get("acknowledged"),
            "is_seen": alert.get("isSeen"),
            "tags": alert.get("tags", []),
            "priority": alert.get("priority"),
            "source": alert.get("source"),
            "owner": alert.get("owner"),
            "responders": alert.get("responders", []),
            "teams": alert.get("teams", []),
            "integration": alert.get("integration", {}),
            "report": alert.get("report", {}),
            "created_at": alert.get("createdAt"),
            "updated_at": alert.get("updatedAt"),
            "acknowledged_at": alert.get("report", {}).get("ackTime"),
            "closed_at": alert.get("report", {}).get("closeTime"),
            "count": alert.get("count", 1),
        }

    async def _sync_incidents(
        self,
        base_url: str,
        headers: dict[str, str],
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync incidents from Opsgenie."""
        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "sort": "createdAt",
            "order": "desc",
        }

        if since:
            params["query"] = f"createdAt >= {int(since.timestamp() * 1000)}"

        incidents = []
        offset = 0

        while len(incidents) < limit:
            params["offset"] = offset
            response = await self.http_client.get(
                f"{base_url}/v1/incidents",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            batch = data.get("data", [])
            if not batch:
                break

            for incident in batch:
                incidents.append(self._transform_incident(incident))
                if len(incidents) >= limit:
                    break

            paging = data.get("paging", {})
            if not paging.get("next"):
                break

            offset += len(batch)

        return incidents

    def _transform_incident(self, incident: dict) -> dict[str, Any]:
        """Transform Opsgenie incident to standard format."""
        return {
            "id": incident.get("id"),
            "tiny_id": incident.get("tinyId"),
            "message": incident.get("message"),
            "description": incident.get("description"),
            "status": incident.get("status"),
            "priority": incident.get("priority"),
            "tags": incident.get("tags", []),
            "impacted_services": incident.get("impactedServices", []),
            "responders": incident.get("responders", []),
            "owner": incident.get("ownerTeam"),
            "created_at": incident.get("createdAt"),
            "updated_at": incident.get("updatedAt"),
            "extra_properties": incident.get("extraProperties", {}),
        }

    async def _sync_users(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync users from Opsgenie."""
        response = await self.http_client.get(
            f"{base_url}/v2/users",
            headers=headers,
            params={"limit": min(limit, 100)},
        )
        response.raise_for_status()
        data = response.json()

        users = []
        for user in data.get("data", []):
            users.append({
                "id": user.get("id"),
                "username": user.get("username"),
                "full_name": user.get("fullName"),
                "role": user.get("role", {}).get("name"),
                "time_zone": user.get("timeZone"),
                "locale": user.get("locale"),
                "verified": user.get("verified"),
                "blocked": user.get("blocked"),
                "user_address": user.get("userAddress", {}),
                "created_at": user.get("createdAt"),
            })

        return users

    async def _sync_schedules(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync schedules from Opsgenie."""
        response = await self.http_client.get(
            f"{base_url}/v2/schedules",
            headers=headers,
            params={"limit": min(limit, 100)},
        )
        response.raise_for_status()
        data = response.json()

        schedules = []
        for schedule in data.get("data", []):
            schedules.append({
                "id": schedule.get("id"),
                "name": schedule.get("name"),
                "description": schedule.get("description"),
                "timezone": schedule.get("timezone"),
                "enabled": schedule.get("enabled"),
                "owner_team": schedule.get("ownerTeam", {}),
                "rotations": schedule.get("rotations", []),
            })

        return schedules

    async def _sync_teams(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync teams from Opsgenie."""
        response = await self.http_client.get(
            f"{base_url}/v2/teams",
            headers=headers,
            params={"limit": min(limit, 100)},
        )
        response.raise_for_status()
        data = response.json()

        teams = []
        for team in data.get("data", []):
            teams.append({
                "id": team.get("id"),
                "name": team.get("name"),
                "description": team.get("description"),
                "members": team.get("members", []),
            })

        return teams

    # Alert Operations

    async def create_alert(
        self,
        credentials: IntegrationCredentials,
        message: str,
        priority: str = "P3",
        description: Optional[str] = None,
        alias: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[list[str]] = None,
        entity: Optional[str] = None,
        responders: Optional[list[dict[str, str]]] = None,
        visible_to: Optional[list[dict[str, str]]] = None,
        actions: Optional[list[str]] = None,
        details: Optional[dict[str, str]] = None,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new alert.

        Args:
            credentials: Integration credentials
            message: Alert message (limited to 130 characters)
            priority: P1, P2, P3, P4, P5 (P1 is highest)
            description: Detailed description
            alias: User-defined identifier for deduplication
            source: Source of alert
            tags: List of tags
            entity: Domain or service affected
            responders: List of responders [{type, id/name}]
            visible_to: List of teams/users who can see alert
            actions: Custom actions
            details: Key-value details
            note: Additional note

        Returns:
            Created alert data
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        # Use integration API key if available
        integration_key = credentials.extra.get("integration_api_key")
        if integration_key:
            headers["Authorization"] = f"GenieKey {integration_key}"

        alert_data: dict[str, Any] = {
            "message": message[:130],  # API limit
            "priority": priority,
        }

        if description:
            alert_data["description"] = description
        if alias:
            alert_data["alias"] = alias
        if source:
            alert_data["source"] = source
        if tags:
            alert_data["tags"] = tags
        if entity:
            alert_data["entity"] = entity
        if responders:
            alert_data["responders"] = responders
        if visible_to:
            alert_data["visibleTo"] = visible_to
        if actions:
            alert_data["actions"] = actions
        if details:
            alert_data["details"] = details
        if note:
            alert_data["note"] = note

        response = await self.http_client.post(
            f"{base_url}/v2/alerts",
            headers=headers,
            json=alert_data,
        )
        response.raise_for_status()

        result = response.json()
        return {
            "request_id": result.get("requestId"),
            "took": result.get("took"),
            "result": result.get("result"),
            "alias": alias,
        }

    async def get_alert(
        self,
        credentials: IntegrationCredentials,
        identifier: str,
        identifier_type: str = "id",
    ) -> dict[str, Any]:
        """Get a specific alert.

        Args:
            credentials: Integration credentials
            identifier: Alert ID or alias
            identifier_type: Type of identifier (id, alias, tiny)

        Returns:
            Alert data
        """
        base_url = self._get_base_url(credentials)

        response = await self.http_client.get(
            f"{base_url}/v2/alerts/{identifier}",
            headers=self._get_headers(credentials),
            params={"identifierType": identifier_type},
        )
        response.raise_for_status()

        return self._transform_alert(response.json().get("data", {}))

    async def acknowledge_alert(
        self,
        credentials: IntegrationCredentials,
        identifier: str,
        identifier_type: str = "id",
        user: Optional[str] = None,
        source: Optional[str] = None,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Acknowledge an alert.

        Args:
            credentials: Integration credentials
            identifier: Alert ID or alias
            identifier_type: Type of identifier (id, alias, tiny)
            user: User acknowledging the alert
            source: Source of acknowledgment
            note: Note to add

        Returns:
            Request result
        """
        base_url = self._get_base_url(credentials)

        data: dict[str, Any] = {}
        if user:
            data["user"] = user
        if source:
            data["source"] = source
        if note:
            data["note"] = note

        response = await self.http_client.post(
            f"{base_url}/v2/alerts/{identifier}/acknowledge",
            headers=self._get_headers(credentials),
            params={"identifierType": identifier_type},
            json=data if data else None,
        )
        response.raise_for_status()

        return response.json()

    async def close_alert(
        self,
        credentials: IntegrationCredentials,
        identifier: str,
        identifier_type: str = "id",
        user: Optional[str] = None,
        source: Optional[str] = None,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Close an alert.

        Args:
            credentials: Integration credentials
            identifier: Alert ID or alias
            identifier_type: Type of identifier (id, alias, tiny)
            user: User closing the alert
            source: Source of close action
            note: Note to add

        Returns:
            Request result
        """
        base_url = self._get_base_url(credentials)

        data: dict[str, Any] = {}
        if user:
            data["user"] = user
        if source:
            data["source"] = source
        if note:
            data["note"] = note

        response = await self.http_client.post(
            f"{base_url}/v2/alerts/{identifier}/close",
            headers=self._get_headers(credentials),
            params={"identifierType": identifier_type},
            json=data if data else None,
        )
        response.raise_for_status()

        return response.json()

    async def add_alert_note(
        self,
        credentials: IntegrationCredentials,
        identifier: str,
        note: str,
        identifier_type: str = "id",
        user: Optional[str] = None,
        source: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add a note to an alert.

        Args:
            credentials: Integration credentials
            identifier: Alert ID or alias
            note: Note content
            identifier_type: Type of identifier (id, alias, tiny)
            user: User adding the note
            source: Source of the note

        Returns:
            Request result
        """
        base_url = self._get_base_url(credentials)

        data: dict[str, Any] = {"note": note}
        if user:
            data["user"] = user
        if source:
            data["source"] = source

        response = await self.http_client.post(
            f"{base_url}/v2/alerts/{identifier}/notes",
            headers=self._get_headers(credentials),
            params={"identifierType": identifier_type},
            json=data,
        )
        response.raise_for_status()

        return response.json()

    # Incident Operations

    async def create_incident(
        self,
        credentials: IntegrationCredentials,
        message: str,
        priority: str = "P3",
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        impacted_services: Optional[list[str]] = None,
        responders: Optional[list[dict[str, str]]] = None,
        status_page_entry: Optional[dict[str, Any]] = None,
        notify_stakeholders: bool = False,
        details: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Create a new incident.

        Args:
            credentials: Integration credentials
            message: Incident message
            priority: P1-P5 priority
            description: Detailed description
            tags: List of tags
            impacted_services: List of service IDs
            responders: List of responders
            status_page_entry: Status page configuration
            notify_stakeholders: Whether to notify stakeholders
            details: Additional key-value details

        Returns:
            Created incident data
        """
        base_url = self._get_base_url(credentials)

        incident_data: dict[str, Any] = {
            "message": message,
            "priority": priority,
        }

        if description:
            incident_data["description"] = description
        if tags:
            incident_data["tags"] = tags
        if impacted_services:
            incident_data["impactedServices"] = impacted_services
        if responders:
            incident_data["responders"] = responders
        if status_page_entry:
            incident_data["statusPageEntry"] = status_page_entry
        if notify_stakeholders:
            incident_data["notifyStakeholders"] = notify_stakeholders
        if details:
            incident_data["details"] = details

        response = await self.http_client.post(
            f"{base_url}/v1/incidents/create",
            headers=self._get_headers(credentials),
            json=incident_data,
        )
        response.raise_for_status()

        return response.json()

    async def get_incident(
        self,
        credentials: IntegrationCredentials,
        identifier: str,
        identifier_type: str = "id",
    ) -> dict[str, Any]:
        """Get a specific incident.

        Args:
            credentials: Integration credentials
            identifier: Incident ID
            identifier_type: Type of identifier (id, tiny)

        Returns:
            Incident data
        """
        base_url = self._get_base_url(credentials)

        response = await self.http_client.get(
            f"{base_url}/v1/incidents/{identifier}",
            headers=self._get_headers(credentials),
            params={"identifierType": identifier_type},
        )
        response.raise_for_status()

        return self._transform_incident(response.json().get("data", {}))

    async def resolve_incident(
        self,
        credentials: IntegrationCredentials,
        identifier: str,
        identifier_type: str = "id",
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Resolve an incident.

        Args:
            credentials: Integration credentials
            identifier: Incident ID
            identifier_type: Type of identifier (id, tiny)
            note: Optional resolution note

        Returns:
            Request result
        """
        base_url = self._get_base_url(credentials)

        data: dict[str, Any] = {}
        if note:
            data["note"] = note

        response = await self.http_client.post(
            f"{base_url}/v1/incidents/{identifier}/resolve",
            headers=self._get_headers(credentials),
            params={"identifierType": identifier_type},
            json=data if data else None,
        )
        response.raise_for_status()

        return response.json()

    async def close_incident(
        self,
        credentials: IntegrationCredentials,
        identifier: str,
        identifier_type: str = "id",
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Close an incident.

        Args:
            credentials: Integration credentials
            identifier: Incident ID
            identifier_type: Type of identifier (id, tiny)
            note: Optional closing note

        Returns:
            Request result
        """
        base_url = self._get_base_url(credentials)

        data: dict[str, Any] = {}
        if note:
            data["note"] = note

        response = await self.http_client.post(
            f"{base_url}/v1/incidents/{identifier}/close",
            headers=self._get_headers(credentials),
            params={"identifierType": identifier_type},
            json=data if data else None,
        )
        response.raise_for_status()

        return response.json()

    async def add_incident_note(
        self,
        credentials: IntegrationCredentials,
        identifier: str,
        note: str,
        identifier_type: str = "id",
    ) -> dict[str, Any]:
        """Add a note to an incident.

        Args:
            credentials: Integration credentials
            identifier: Incident ID
            note: Note content
            identifier_type: Type of identifier (id, tiny)

        Returns:
            Request result
        """
        base_url = self._get_base_url(credentials)

        response = await self.http_client.post(
            f"{base_url}/v1/incidents/{identifier}/notes",
            headers=self._get_headers(credentials),
            params={"identifierType": identifier_type},
            json={"note": note},
        )
        response.raise_for_status()

        return response.json()

    async def get_on_call(
        self,
        credentials: IntegrationCredentials,
        schedule_identifier: str,
        schedule_identifier_type: str = "id",
        flat: bool = True,
    ) -> list[dict[str, Any]]:
        """Get current on-call participants for a schedule.

        Args:
            credentials: Integration credentials
            schedule_identifier: Schedule ID or name
            schedule_identifier_type: Type of identifier (id, name)
            flat: Return flat list (True) or nested by period (False)

        Returns:
            List of on-call participants
        """
        base_url = self._get_base_url(credentials)

        response = await self.http_client.get(
            f"{base_url}/v2/schedules/{schedule_identifier}/on-calls",
            headers=self._get_headers(credentials),
            params={
                "scheduleIdentifierType": schedule_identifier_type,
                "flat": str(flat).lower(),
            },
        )
        response.raise_for_status()

        data = response.json().get("data", {})
        on_calls = data.get("onCallParticipants", []) if flat else data.get("_parent", {}).get("onCallRecipients", [])

        return on_calls
