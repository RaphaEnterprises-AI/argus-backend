"""PagerDuty integration plugin.

This module provides integration with PagerDuty for incident management,
on-call scheduling, and alerting.

API Documentation:
- REST API v2: https://developer.pagerduty.com/api-reference/
- Events API v2: https://developer.pagerduty.com/docs/events-api-v2/overview/
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


class PagerDutyPlugin(IntegrationPlugin):
    """PagerDuty integration plugin.

    Provides access to:
    - Incidents: List, create, update, resolve incidents
    - Services: List services and their integrations
    - On-Calls: View current on-call schedules
    - Escalation Policies: View and manage escalation policies
    - Events: Trigger, acknowledge, resolve events via Events API v2
    """

    REST_API_BASE_URL = "https://api.pagerduty.com"
    EVENTS_API_URL = "https://events.pagerduty.com/v2/enqueue"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get metadata about this integration."""
        return IntegrationMetadata(
            slug="pagerduty",
            name="PagerDuty",
            description="Incident management, on-call scheduling, and alerting platform",
            category=IntegrationCategory.INCIDENT_MANAGEMENT,
            auth_type=AuthType.API_KEY,
            icon_url="https://www.pagerduty.com/favicon.ico",
            docs_url="https://developer.pagerduty.com/docs/",
            website_url="https://www.pagerduty.com",
            features=[
                "incidents",
                "services",
                "on_calls",
                "escalation_policies",
                "event_triggering",
                "schedules",
                "teams",
                "users",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="REST API Token",
                    type="password",
                    required=True,
                    placeholder="u+abcd1234...",
                    help_text="Generate from Configuration > API Access Keys in PagerDuty",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra.routing_key",
                    label="Events API Routing Key",
                    type="password",
                    required=False,
                    placeholder="R0123456789ABCDEF...",
                    help_text="Integration key for triggering events (from a service integration)",
                ),
                FieldDefinition(
                    name="extra.default_from_email",
                    label="Default From Email",
                    type="email",
                    required=False,
                    placeholder="alerts@yourcompany.com",
                    help_text="Email of a valid user for API actions requiring 'from' header",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["incidents", "services", "on_calls", "escalation_policies", "schedules"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for REST API requests."""
        headers = {
            "Authorization": f"Token token={credentials.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.pagerduty+json;version=2",
        }
        # Add From header if provided (required for some write operations)
        from_email = credentials.extra.get("default_from_email")
        if from_email:
            headers["From"] = from_email
        return headers

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the PagerDuty connection by getting current user info.

        Uses GET /users/me endpoint to verify API key validity.
        """
        start_time = time.time()

        try:
            response = await self.http_client.get(
                f"{self.REST_API_BASE_URL}/users/me",
                headers=self._get_headers(credentials),
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                user = data.get("user", {})

                return ConnectionTestResult(
                    success=True,
                    message=f"Connected as {user.get('name', 'Unknown')}",
                    latency_ms=latency_ms,
                    api_version="2",
                    permissions=self._extract_permissions(user),
                    account_info={
                        "user_id": user.get("id"),
                        "user_name": user.get("name"),
                        "user_email": user.get("email"),
                        "role": user.get("role"),
                        "time_zone": user.get("time_zone"),
                        "avatar_url": user.get("avatar_url"),
                    },
                )
            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid API key",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status": 401},
                )
            elif response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="API key lacks permissions to access user info",
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

    def _extract_permissions(self, user: dict) -> list[str]:
        """Extract permissions from user role."""
        role = user.get("role", "")
        permissions = ["read"]

        if role in ["admin", "owner", "account_owner"]:
            permissions.extend(["write", "delete", "admin"])
        elif role in ["manager", "responder"]:
            permissions.append("write")
        elif role == "limited_user":
            pass  # Read only

        return permissions

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from PagerDuty.

        Supports syncing:
        - incidents: Active and resolved incidents
        - services: PagerDuty services
        - on_calls: Current on-call schedules
        - escalation_policies: Escalation policies
        - schedules: On-call schedules
        """
        start_time = datetime.utcnow()
        sync_id = str(uuid.uuid4())

        # Default to all resources if not specified
        if resources is None:
            resources = ["incidents", "services", "on_calls"]

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        headers = self._get_headers(credentials)

        for resource in resources:
            try:
                if resource == "incidents":
                    result = await self._sync_incidents(headers, since, limit)
                    data["incidents"] = result
                    total_synced += len(result)
                elif resource == "services":
                    result = await self._sync_services(headers, limit)
                    data["services"] = result
                    total_synced += len(result)
                elif resource == "on_calls":
                    result = await self._sync_on_calls(headers)
                    data["on_calls"] = result
                    total_synced += len(result)
                elif resource == "escalation_policies":
                    result = await self._sync_escalation_policies(headers, limit)
                    data["escalation_policies"] = result
                    total_synced += len(result)
                elif resource == "schedules":
                    result = await self._sync_schedules(headers, limit)
                    data["schedules"] = result
                    total_synced += len(result)
                else:
                    warnings.append(f"Unknown resource type: {resource}")
            except httpx.HTTPStatusError as e:
                errors.append(f"Failed to sync {resource}: HTTP {e.response.status_code}")
            except Exception as e:
                errors.append(f"Failed to sync {resource}: {str(e)}")

        return SyncResult(
            success=len(errors) == 0,
            message=f"Synced {total_synced} items from PagerDuty"
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

    async def _sync_incidents(
        self,
        headers: dict[str, str],
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync incidents from PagerDuty."""
        params: dict[str, Any] = {
            "limit": min(limit, 100),  # PagerDuty max is 100
            "sort_by": "created_at:desc",
            "statuses[]": ["triggered", "acknowledged", "resolved"],
        }

        if since:
            params["since"] = since.isoformat()

        incidents = []
        offset = 0

        while len(incidents) < limit:
            params["offset"] = offset
            response = await self.http_client.get(
                f"{self.REST_API_BASE_URL}/incidents",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            batch = data.get("incidents", [])
            if not batch:
                break

            for incident in batch:
                incidents.append(self._transform_incident(incident))
                if len(incidents) >= limit:
                    break

            if not data.get("more", False):
                break

            offset += len(batch)

        return incidents

    def _transform_incident(self, incident: dict) -> dict[str, Any]:
        """Transform PagerDuty incident to standard format."""
        return {
            "id": incident.get("id"),
            "incident_number": incident.get("incident_number"),
            "title": incident.get("title"),
            "description": incident.get("description"),
            "status": incident.get("status"),
            "urgency": incident.get("urgency"),
            "priority": incident.get("priority", {}).get("name") if incident.get("priority") else None,
            "service": {
                "id": incident.get("service", {}).get("id"),
                "name": incident.get("service", {}).get("summary"),
            },
            "assigned_to": [
                {"id": a.get("assignee", {}).get("id"), "name": a.get("assignee", {}).get("summary")}
                for a in incident.get("assignments", [])
            ],
            "escalation_policy": {
                "id": incident.get("escalation_policy", {}).get("id"),
                "name": incident.get("escalation_policy", {}).get("summary"),
            },
            "created_at": incident.get("created_at"),
            "updated_at": incident.get("updated_at"),
            "resolved_at": incident.get("last_status_change_at")
            if incident.get("status") == "resolved"
            else None,
            "html_url": incident.get("html_url"),
            "alert_counts": incident.get("alert_counts", {}),
        }

    async def _sync_services(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync services from PagerDuty."""
        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "include[]": ["integrations", "escalation_policies"],
        }

        services = []
        offset = 0

        while len(services) < limit:
            params["offset"] = offset
            response = await self.http_client.get(
                f"{self.REST_API_BASE_URL}/services",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            batch = data.get("services", [])
            if not batch:
                break

            for service in batch:
                services.append({
                    "id": service.get("id"),
                    "name": service.get("name"),
                    "description": service.get("description"),
                    "status": service.get("status"),
                    "created_at": service.get("created_at"),
                    "updated_at": service.get("updated_at"),
                    "escalation_policy": {
                        "id": service.get("escalation_policy", {}).get("id"),
                        "name": service.get("escalation_policy", {}).get("summary"),
                    },
                    "integrations": [
                        {
                            "id": i.get("id"),
                            "type": i.get("type"),
                            "name": i.get("name"),
                            "integration_key": i.get("integration_key"),
                        }
                        for i in service.get("integrations", [])
                    ],
                    "html_url": service.get("html_url"),
                    "alert_creation": service.get("alert_creation"),
                    "auto_resolve_timeout": service.get("auto_resolve_timeout"),
                    "acknowledgement_timeout": service.get("acknowledgement_timeout"),
                })
                if len(services) >= limit:
                    break

            if not data.get("more", False):
                break

            offset += len(batch)

        return services

    async def _sync_on_calls(
        self,
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Sync current on-call schedules from PagerDuty."""
        response = await self.http_client.get(
            f"{self.REST_API_BASE_URL}/oncalls",
            headers=headers,
            params={
                "limit": 100,
                "include[]": ["users", "schedules", "escalation_policies"],
            },
        )
        response.raise_for_status()
        data = response.json()

        on_calls = []
        for on_call in data.get("oncalls", []):
            on_calls.append({
                "user": {
                    "id": on_call.get("user", {}).get("id"),
                    "name": on_call.get("user", {}).get("summary"),
                    "email": on_call.get("user", {}).get("email"),
                },
                "schedule": {
                    "id": on_call.get("schedule", {}).get("id"),
                    "name": on_call.get("schedule", {}).get("summary"),
                } if on_call.get("schedule") else None,
                "escalation_policy": {
                    "id": on_call.get("escalation_policy", {}).get("id"),
                    "name": on_call.get("escalation_policy", {}).get("summary"),
                },
                "escalation_level": on_call.get("escalation_level"),
                "start": on_call.get("start"),
                "end": on_call.get("end"),
            })

        return on_calls

    async def _sync_escalation_policies(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync escalation policies from PagerDuty."""
        response = await self.http_client.get(
            f"{self.REST_API_BASE_URL}/escalation_policies",
            headers=headers,
            params={"limit": min(limit, 100)},
        )
        response.raise_for_status()
        data = response.json()

        policies = []
        for policy in data.get("escalation_policies", []):
            policies.append({
                "id": policy.get("id"),
                "name": policy.get("name"),
                "description": policy.get("description"),
                "num_loops": policy.get("num_loops"),
                "escalation_rules": [
                    {
                        "escalation_delay_in_minutes": rule.get("escalation_delay_in_minutes"),
                        "targets": [
                            {
                                "id": t.get("id"),
                                "type": t.get("type"),
                                "name": t.get("summary"),
                            }
                            for t in rule.get("targets", [])
                        ],
                    }
                    for rule in policy.get("escalation_rules", [])
                ],
                "services": [
                    {"id": s.get("id"), "name": s.get("summary")}
                    for s in policy.get("services", [])
                ],
            })

        return policies

    async def _sync_schedules(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync schedules from PagerDuty."""
        response = await self.http_client.get(
            f"{self.REST_API_BASE_URL}/schedules",
            headers=headers,
            params={"limit": min(limit, 100)},
        )
        response.raise_for_status()
        data = response.json()

        schedules = []
        for schedule in data.get("schedules", []):
            schedules.append({
                "id": schedule.get("id"),
                "name": schedule.get("name"),
                "description": schedule.get("description"),
                "time_zone": schedule.get("time_zone"),
                "escalation_policies": [
                    {"id": ep.get("id"), "name": ep.get("summary")}
                    for ep in schedule.get("escalation_policies", [])
                ],
                "users": [
                    {"id": u.get("id"), "name": u.get("summary")}
                    for u in schedule.get("users", [])
                ],
            })

        return schedules

    # Incident Management Operations

    async def create_incident(
        self,
        credentials: IntegrationCredentials,
        title: str,
        service_id: str,
        urgency: str = "high",
        body: Optional[str] = None,
        escalation_policy_id: Optional[str] = None,
        incident_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new incident via REST API.

        Args:
            credentials: Integration credentials
            title: Incident title
            service_id: ID of the service to create incident for
            urgency: Urgency level (high or low)
            body: Incident body/description
            escalation_policy_id: Optional escalation policy override
            incident_key: Optional unique key for deduplication

        Returns:
            Created incident data
        """
        headers = self._get_headers(credentials)

        incident_data: dict[str, Any] = {
            "type": "incident",
            "title": title,
            "service": {
                "id": service_id,
                "type": "service_reference",
            },
            "urgency": urgency,
        }

        if body:
            incident_data["body"] = {"type": "incident_body", "details": body}

        if escalation_policy_id:
            incident_data["escalation_policy"] = {
                "id": escalation_policy_id,
                "type": "escalation_policy_reference",
            }

        if incident_key:
            incident_data["incident_key"] = incident_key

        response = await self.http_client.post(
            f"{self.REST_API_BASE_URL}/incidents",
            headers=headers,
            json={"incident": incident_data},
        )
        response.raise_for_status()

        return self._transform_incident(response.json().get("incident", {}))

    async def update_incident(
        self,
        credentials: IntegrationCredentials,
        incident_id: str,
        status: Optional[str] = None,
        title: Optional[str] = None,
        urgency: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update an existing incident.

        Args:
            credentials: Integration credentials
            incident_id: ID of the incident to update
            status: New status (triggered, acknowledged, resolved)
            title: New title
            urgency: New urgency (high or low)
            resolution: Resolution note (when resolving)

        Returns:
            Updated incident data
        """
        headers = self._get_headers(credentials)

        incident_data: dict[str, Any] = {
            "id": incident_id,
            "type": "incident",
        }

        if status:
            incident_data["status"] = status
        if title:
            incident_data["title"] = title
        if urgency:
            incident_data["urgency"] = urgency
        if resolution:
            incident_data["resolution"] = resolution

        response = await self.http_client.put(
            f"{self.REST_API_BASE_URL}/incidents/{incident_id}",
            headers=headers,
            json={"incident": incident_data},
        )
        response.raise_for_status()

        return self._transform_incident(response.json().get("incident", {}))

    async def resolve_incident(
        self,
        credentials: IntegrationCredentials,
        incident_id: str,
        resolution: Optional[str] = None,
    ) -> dict[str, Any]:
        """Resolve an incident.

        Args:
            credentials: Integration credentials
            incident_id: ID of the incident to resolve
            resolution: Optional resolution note

        Returns:
            Updated incident data
        """
        return await self.update_incident(
            credentials=credentials,
            incident_id=incident_id,
            status="resolved",
            resolution=resolution,
        )

    async def acknowledge_incident(
        self,
        credentials: IntegrationCredentials,
        incident_id: str,
    ) -> dict[str, Any]:
        """Acknowledge an incident.

        Args:
            credentials: Integration credentials
            incident_id: ID of the incident to acknowledge

        Returns:
            Updated incident data
        """
        return await self.update_incident(
            credentials=credentials,
            incident_id=incident_id,
            status="acknowledged",
        )

    # Events API v2 Operations

    async def trigger_event(
        self,
        credentials: IntegrationCredentials,
        summary: str,
        source: str,
        severity: str = "error",
        routing_key: Optional[str] = None,
        dedup_key: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        component: Optional[str] = None,
        group: Optional[str] = None,
        event_class: Optional[str] = None,
        custom_details: Optional[dict[str, Any]] = None,
        links: Optional[list[dict[str, str]]] = None,
        images: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """Trigger an event via Events API v2.

        Args:
            credentials: Integration credentials
            summary: A brief text summary of the event
            source: The source of the event (e.g., hostname)
            severity: Event severity (critical, error, warning, info)
            routing_key: Integration key (defaults to credentials.extra.routing_key)
            dedup_key: Deduplication key for grouping alerts
            timestamp: Event timestamp (defaults to now)
            component: Component affected
            group: Logical grouping
            event_class: Class/type of the event
            custom_details: Additional details
            links: Related links [{href, text}]
            images: Related images [{src, href, alt}]

        Returns:
            Event response with dedup_key and status
        """
        key = routing_key or credentials.extra.get("routing_key")
        if not key:
            raise ValueError("Routing key required for Events API. Set routing_key or credentials.extra.routing_key")

        payload: dict[str, Any] = {
            "summary": summary,
            "source": source,
            "severity": severity,
        }

        if timestamp:
            payload["timestamp"] = timestamp.isoformat()
        if component:
            payload["component"] = component
        if group:
            payload["group"] = group
        if event_class:
            payload["class"] = event_class
        if custom_details:
            payload["custom_details"] = custom_details

        event_data: dict[str, Any] = {
            "routing_key": key,
            "event_action": "trigger",
            "payload": payload,
        }

        if dedup_key:
            event_data["dedup_key"] = dedup_key
        if links:
            event_data["links"] = links
        if images:
            event_data["images"] = images

        response = await self.http_client.post(
            self.EVENTS_API_URL,
            json=event_data,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        return response.json()

    async def acknowledge_event(
        self,
        credentials: IntegrationCredentials,
        dedup_key: str,
        routing_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Acknowledge an event via Events API v2.

        Args:
            credentials: Integration credentials
            dedup_key: The dedup_key of the alert to acknowledge
            routing_key: Integration key (defaults to credentials.extra.routing_key)

        Returns:
            Event response
        """
        key = routing_key or credentials.extra.get("routing_key")
        if not key:
            raise ValueError("Routing key required for Events API")

        response = await self.http_client.post(
            self.EVENTS_API_URL,
            json={
                "routing_key": key,
                "event_action": "acknowledge",
                "dedup_key": dedup_key,
            },
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        return response.json()

    async def resolve_event(
        self,
        credentials: IntegrationCredentials,
        dedup_key: str,
        routing_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Resolve an event via Events API v2.

        Args:
            credentials: Integration credentials
            dedup_key: The dedup_key of the alert to resolve
            routing_key: Integration key (defaults to credentials.extra.routing_key)

        Returns:
            Event response
        """
        key = routing_key or credentials.extra.get("routing_key")
        if not key:
            raise ValueError("Routing key required for Events API")

        response = await self.http_client.post(
            self.EVENTS_API_URL,
            json={
                "routing_key": key,
                "event_action": "resolve",
                "dedup_key": dedup_key,
            },
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        return response.json()

    async def get_incident(
        self,
        credentials: IntegrationCredentials,
        incident_id: str,
    ) -> dict[str, Any]:
        """Get a specific incident by ID.

        Args:
            credentials: Integration credentials
            incident_id: The incident ID

        Returns:
            Incident data
        """
        response = await self.http_client.get(
            f"{self.REST_API_BASE_URL}/incidents/{incident_id}",
            headers=self._get_headers(credentials),
        )
        response.raise_for_status()

        return self._transform_incident(response.json().get("incident", {}))

    async def add_incident_note(
        self,
        credentials: IntegrationCredentials,
        incident_id: str,
        content: str,
    ) -> dict[str, Any]:
        """Add a note to an incident.

        Args:
            credentials: Integration credentials
            incident_id: The incident ID
            content: Note content

        Returns:
            Created note data
        """
        response = await self.http_client.post(
            f"{self.REST_API_BASE_URL}/incidents/{incident_id}/notes",
            headers=self._get_headers(credentials),
            json={"note": {"content": content}},
        )
        response.raise_for_status()

        note = response.json().get("note", {})
        return {
            "id": note.get("id"),
            "content": note.get("content"),
            "created_at": note.get("created_at"),
            "user": {
                "id": note.get("user", {}).get("id"),
                "name": note.get("user", {}).get("summary"),
            },
        }
