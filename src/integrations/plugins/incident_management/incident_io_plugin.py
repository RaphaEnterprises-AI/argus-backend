"""incident.io integration plugin.

This module provides integration with incident.io for incident management,
post-mortems, and status page updates.

API Documentation:
- API Reference: https://api-docs.incident.io/
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


class IncidentIOPlugin(IntegrationPlugin):
    """incident.io integration plugin.

    Provides access to:
    - Incidents: List, create, update incidents
    - Actions: Manage incident actions
    - Announcements: Create and manage announcements
    - Custom Fields: Work with custom field values
    - Severities: List severity levels
    - Incident Types: List incident types
    - Roles: List incident roles
    - Statuses: List incident statuses
    """

    API_BASE_URL = "https://api.incident.io"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get metadata about this integration."""
        return IntegrationMetadata(
            slug="incident_io",
            name="incident.io",
            description="Modern incident management platform for fast-moving teams",
            category=IntegrationCategory.INCIDENT_MANAGEMENT,
            auth_type=AuthType.API_KEY,
            icon_url="https://incident.io/favicon.ico",
            docs_url="https://api-docs.incident.io/",
            website_url="https://incident.io",
            features=[
                "incidents",
                "actions",
                "announcements",
                "custom_fields",
                "severities",
                "incident_types",
                "roles",
                "statuses",
                "post_mortems",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key",
                    type="password",
                    required=True,
                    placeholder="inc_xxxxxxxxxxxxxxxx",
                    help_text="Generate from Settings > API keys in incident.io",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra.organization_id",
                    label="Organization ID",
                    type="text",
                    required=False,
                    placeholder="org_xxxxxxxx",
                    help_text="Your incident.io organization ID (optional)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["incidents", "actions", "announcements", "severities", "incident_types"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {credentials.api_key}",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the incident.io connection by listing incidents.

        Uses GET /v2/incidents endpoint to verify API key validity.
        """
        start_time = time.time()

        try:
            # incident.io doesn't have a dedicated /me endpoint,
            # so we list incidents with a limit of 1 to verify connectivity
            response = await self.http_client.get(
                f"{self.API_BASE_URL}/v2/incidents",
                headers=self._get_headers(credentials),
                params={"page_size": 1},
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                pagination = data.get("pagination_meta", {})

                return ConnectionTestResult(
                    success=True,
                    message="Connected to incident.io",
                    latency_ms=latency_ms,
                    api_version="2",
                    permissions=["read", "write"],  # API key typically has full access
                    account_info={
                        "total_incidents": pagination.get("total_record_count", 0),
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

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from incident.io.

        Supports syncing:
        - incidents: Active and resolved incidents
        - actions: Incident actions
        - announcements: Status announcements
        - severities: Severity levels
        - incident_types: Incident types
        """
        start_time = datetime.utcnow()
        sync_id = str(uuid.uuid4())

        if resources is None:
            resources = ["incidents", "actions", "severities", "incident_types"]

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
                elif resource == "actions":
                    result = await self._sync_actions(headers, limit)
                    data["actions"] = result
                    total_synced += len(result)
                elif resource == "announcements":
                    result = await self._sync_announcements(headers, limit)
                    data["announcements"] = result
                    total_synced += len(result)
                elif resource == "severities":
                    result = await self._sync_severities(headers)
                    data["severities"] = result
                    total_synced += len(result)
                elif resource == "incident_types":
                    result = await self._sync_incident_types(headers)
                    data["incident_types"] = result
                    total_synced += len(result)
                elif resource == "roles":
                    result = await self._sync_roles(headers)
                    data["roles"] = result
                    total_synced += len(result)
                elif resource == "statuses":
                    result = await self._sync_statuses(headers)
                    data["statuses"] = result
                    total_synced += len(result)
                else:
                    warnings.append(f"Unknown resource type: {resource}")
            except httpx.HTTPStatusError as e:
                errors.append(f"Failed to sync {resource}: HTTP {e.response.status_code}")
            except Exception as e:
                errors.append(f"Failed to sync {resource}: {str(e)}")

        return SyncResult(
            success=len(errors) == 0,
            message=f"Synced {total_synced} items from incident.io"
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
        """Sync incidents from incident.io."""
        params: dict[str, Any] = {
            "page_size": min(limit, 100),
        }

        if since:
            params["created_at[gte]"] = since.isoformat()

        incidents = []
        after_cursor = None

        while len(incidents) < limit:
            if after_cursor:
                params["after"] = after_cursor

            response = await self.http_client.get(
                f"{self.API_BASE_URL}/v2/incidents",
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

            # Check pagination
            pagination = data.get("pagination_meta", {})
            after_cursor = pagination.get("after")
            if not after_cursor:
                break

        return incidents

    def _transform_incident(self, incident: dict) -> dict[str, Any]:
        """Transform incident.io incident to standard format."""
        return {
            "id": incident.get("id"),
            "reference": incident.get("reference"),
            "name": incident.get("name"),
            "summary": incident.get("summary"),
            "status": {
                "id": incident.get("incident_status", {}).get("id"),
                "name": incident.get("incident_status", {}).get("name"),
                "category": incident.get("incident_status", {}).get("category"),
            },
            "severity": {
                "id": incident.get("severity", {}).get("id"),
                "name": incident.get("severity", {}).get("name"),
            } if incident.get("severity") else None,
            "incident_type": {
                "id": incident.get("incident_type", {}).get("id"),
                "name": incident.get("incident_type", {}).get("name"),
            } if incident.get("incident_type") else None,
            "mode": incident.get("mode"),
            "visibility": incident.get("visibility"),
            "created_at": incident.get("created_at"),
            "updated_at": incident.get("updated_at"),
            "call_url": incident.get("call_url"),
            "slack_channel_id": incident.get("slack_channel_id"),
            "slack_channel_name": incident.get("slack_channel_name"),
            "incident_role_assignments": [
                {
                    "role": {
                        "id": ra.get("role", {}).get("id"),
                        "name": ra.get("role", {}).get("name"),
                    },
                    "assignee": {
                        "id": ra.get("assignee", {}).get("id"),
                        "name": ra.get("assignee", {}).get("name"),
                        "email": ra.get("assignee", {}).get("email"),
                    } if ra.get("assignee") else None,
                }
                for ra in incident.get("incident_role_assignments", [])
            ],
            "custom_field_entries": [
                {
                    "custom_field": {
                        "id": cfe.get("custom_field", {}).get("id"),
                        "name": cfe.get("custom_field", {}).get("name"),
                    },
                    "values": cfe.get("values", []),
                }
                for cfe in incident.get("custom_field_entries", [])
            ],
            "timestamps": incident.get("incident_timestamps", []),
            "permalink": incident.get("permalink"),
            "postmortem_document_url": incident.get("postmortem_document_url"),
        }

    async def _sync_actions(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync actions from incident.io."""
        params: dict[str, Any] = {
            "page_size": min(limit, 100),
        }

        actions = []
        after_cursor = None

        while len(actions) < limit:
            if after_cursor:
                params["after"] = after_cursor

            response = await self.http_client.get(
                f"{self.API_BASE_URL}/v2/actions",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            batch = data.get("actions", [])
            if not batch:
                break

            for action in batch:
                actions.append({
                    "id": action.get("id"),
                    "incident_id": action.get("incident_id"),
                    "status": action.get("status"),
                    "description": action.get("description"),
                    "follow_up": action.get("follow_up"),
                    "assignee": {
                        "id": action.get("assignee", {}).get("id"),
                        "name": action.get("assignee", {}).get("name"),
                        "email": action.get("assignee", {}).get("email"),
                    } if action.get("assignee") else None,
                    "created_at": action.get("created_at"),
                    "updated_at": action.get("updated_at"),
                    "completed_at": action.get("completed_at"),
                })
                if len(actions) >= limit:
                    break

            pagination = data.get("pagination_meta", {})
            after_cursor = pagination.get("after")
            if not after_cursor:
                break

        return actions

    async def _sync_announcements(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync announcements from incident.io."""
        response = await self.http_client.get(
            f"{self.API_BASE_URL}/v2/incident_updates",
            headers=headers,
            params={"page_size": min(limit, 100)},
        )
        response.raise_for_status()
        data = response.json()

        announcements = []
        for update in data.get("incident_updates", []):
            announcements.append({
                "id": update.get("id"),
                "incident_id": update.get("incident_id"),
                "message": update.get("message"),
                "new_status": {
                    "id": update.get("new_incident_status", {}).get("id"),
                    "name": update.get("new_incident_status", {}).get("name"),
                } if update.get("new_incident_status") else None,
                "new_severity": {
                    "id": update.get("new_severity", {}).get("id"),
                    "name": update.get("new_severity", {}).get("name"),
                } if update.get("new_severity") else None,
                "creator": {
                    "id": update.get("creator", {}).get("id"),
                    "name": update.get("creator", {}).get("name"),
                } if update.get("creator") else None,
                "created_at": update.get("created_at"),
            })

        return announcements

    async def _sync_severities(
        self,
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Sync severities from incident.io."""
        response = await self.http_client.get(
            f"{self.API_BASE_URL}/v1/severities",
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        severities = []
        for severity in data.get("severities", []):
            severities.append({
                "id": severity.get("id"),
                "name": severity.get("name"),
                "description": severity.get("description"),
                "rank": severity.get("rank"),
            })

        return severities

    async def _sync_incident_types(
        self,
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Sync incident types from incident.io."""
        response = await self.http_client.get(
            f"{self.API_BASE_URL}/v1/incident_types",
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        incident_types = []
        for incident_type in data.get("incident_types", []):
            incident_types.append({
                "id": incident_type.get("id"),
                "name": incident_type.get("name"),
                "description": incident_type.get("description"),
                "is_default": incident_type.get("is_default"),
                "created_at": incident_type.get("created_at"),
                "updated_at": incident_type.get("updated_at"),
            })

        return incident_types

    async def _sync_roles(
        self,
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Sync incident roles from incident.io."""
        response = await self.http_client.get(
            f"{self.API_BASE_URL}/v1/incident_roles",
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        roles = []
        for role in data.get("incident_roles", []):
            roles.append({
                "id": role.get("id"),
                "name": role.get("name"),
                "description": role.get("description"),
                "shortform": role.get("shortform"),
                "required": role.get("required"),
                "role_type": role.get("role_type"),
                "created_at": role.get("created_at"),
                "updated_at": role.get("updated_at"),
            })

        return roles

    async def _sync_statuses(
        self,
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Sync incident statuses from incident.io."""
        response = await self.http_client.get(
            f"{self.API_BASE_URL}/v1/incident_statuses",
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        statuses = []
        for status in data.get("incident_statuses", []):
            statuses.append({
                "id": status.get("id"),
                "name": status.get("name"),
                "description": status.get("description"),
                "category": status.get("category"),
                "rank": status.get("rank"),
                "created_at": status.get("created_at"),
                "updated_at": status.get("updated_at"),
            })

        return statuses

    # Incident Operations

    async def create_incident(
        self,
        credentials: IntegrationCredentials,
        name: str,
        summary: Optional[str] = None,
        severity_id: Optional[str] = None,
        incident_type_id: Optional[str] = None,
        incident_status_id: Optional[str] = None,
        mode: str = "standard",
        visibility: str = "public",
        idempotency_key: Optional[str] = None,
        custom_field_entries: Optional[list[dict[str, Any]]] = None,
        incident_role_assignments: Optional[list[dict[str, str]]] = None,
        slack_team_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new incident.

        Args:
            credentials: Integration credentials
            name: Incident name
            summary: Brief summary
            severity_id: Severity ID
            incident_type_id: Incident type ID
            incident_status_id: Initial status ID
            mode: Incident mode (standard, retrospective, test, tutorial)
            visibility: Visibility (public, private)
            idempotency_key: Unique key for idempotent creation
            custom_field_entries: Custom field values
            incident_role_assignments: Role assignments
            slack_team_id: Slack team ID for channel creation

        Returns:
            Created incident data
        """
        headers = self._get_headers(credentials)

        incident_data: dict[str, Any] = {
            "name": name,
            "mode": mode,
            "visibility": visibility,
        }

        if summary:
            incident_data["summary"] = summary
        if severity_id:
            incident_data["severity_id"] = severity_id
        if incident_type_id:
            incident_data["incident_type_id"] = incident_type_id
        if incident_status_id:
            incident_data["incident_status_id"] = incident_status_id
        if idempotency_key:
            incident_data["idempotency_key"] = idempotency_key
        if custom_field_entries:
            incident_data["custom_field_entries"] = custom_field_entries
        if incident_role_assignments:
            incident_data["incident_role_assignments"] = incident_role_assignments
        if slack_team_id:
            incident_data["slack_team_id"] = slack_team_id

        response = await self.http_client.post(
            f"{self.API_BASE_URL}/v2/incidents",
            headers=headers,
            json=incident_data,
        )
        response.raise_for_status()

        return self._transform_incident(response.json().get("incident", {}))

    async def get_incident(
        self,
        credentials: IntegrationCredentials,
        incident_id: str,
    ) -> dict[str, Any]:
        """Get a specific incident.

        Args:
            credentials: Integration credentials
            incident_id: Incident ID

        Returns:
            Incident data
        """
        response = await self.http_client.get(
            f"{self.API_BASE_URL}/v2/incidents/{incident_id}",
            headers=self._get_headers(credentials),
        )
        response.raise_for_status()

        return self._transform_incident(response.json().get("incident", {}))

    async def update_incident(
        self,
        credentials: IntegrationCredentials,
        incident_id: str,
        name: Optional[str] = None,
        summary: Optional[str] = None,
        severity_id: Optional[str] = None,
        incident_status_id: Optional[str] = None,
        incident_type_id: Optional[str] = None,
        custom_field_entries: Optional[list[dict[str, Any]]] = None,
        incident_role_assignments: Optional[list[dict[str, str]]] = None,
        call_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update an incident.

        Args:
            credentials: Integration credentials
            incident_id: Incident ID
            name: New name
            summary: New summary
            severity_id: New severity ID
            incident_status_id: New status ID
            incident_type_id: New incident type ID
            custom_field_entries: Updated custom field values
            incident_role_assignments: Updated role assignments
            call_url: Call/meeting URL

        Returns:
            Updated incident data
        """
        headers = self._get_headers(credentials)

        # First get existing incident to merge changes
        existing = await self.get_incident(credentials, incident_id)

        # Build update payload - only include changed fields
        incident_data: dict[str, Any] = {"incident": {}}

        if name is not None:
            incident_data["incident"]["name"] = name
        if summary is not None:
            incident_data["incident"]["summary"] = summary
        if severity_id is not None:
            incident_data["incident"]["severity_id"] = severity_id
        if incident_status_id is not None:
            incident_data["incident"]["incident_status_id"] = incident_status_id
        if incident_type_id is not None:
            incident_data["incident"]["incident_type_id"] = incident_type_id
        if custom_field_entries is not None:
            incident_data["incident"]["custom_field_entries"] = custom_field_entries
        if incident_role_assignments is not None:
            incident_data["incident"]["incident_role_assignments"] = incident_role_assignments
        if call_url is not None:
            incident_data["incident"]["call_url"] = call_url

        # Use PATCH for partial updates
        response = await self.http_client.patch(
            f"{self.API_BASE_URL}/v2/incidents/{incident_id}",
            headers=headers,
            json=incident_data,
        )
        response.raise_for_status()

        return self._transform_incident(response.json().get("incident", {}))

    async def close_incident(
        self,
        credentials: IntegrationCredentials,
        incident_id: str,
        status_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Close an incident by setting its status to closed.

        Args:
            credentials: Integration credentials
            incident_id: Incident ID
            status_id: Closed status ID (if not provided, will try to find one)

        Returns:
            Updated incident data
        """
        # If no status_id provided, try to find a closed status
        if not status_id:
            statuses = await self._sync_statuses(self._get_headers(credentials))
            closed_statuses = [s for s in statuses if s.get("category") == "closed"]
            if closed_statuses:
                status_id = closed_statuses[0]["id"]
            else:
                raise ValueError("No closed status found. Please provide a status_id.")

        return await self.update_incident(
            credentials=credentials,
            incident_id=incident_id,
            incident_status_id=status_id,
        )

    # Action Operations

    async def create_action(
        self,
        credentials: IntegrationCredentials,
        incident_id: str,
        description: str,
        assignee_id: Optional[str] = None,
        follow_up: bool = False,
    ) -> dict[str, Any]:
        """Create an action for an incident.

        Args:
            credentials: Integration credentials
            incident_id: Incident ID
            description: Action description
            assignee_id: User ID to assign the action to
            follow_up: Whether this is a follow-up action

        Returns:
            Created action data
        """
        headers = self._get_headers(credentials)

        action_data: dict[str, Any] = {
            "incident_id": incident_id,
            "description": description,
            "follow_up": follow_up,
        }

        if assignee_id:
            action_data["assignee_id"] = assignee_id

        response = await self.http_client.post(
            f"{self.API_BASE_URL}/v1/actions",
            headers=headers,
            json=action_data,
        )
        response.raise_for_status()

        action = response.json().get("action", {})
        return {
            "id": action.get("id"),
            "incident_id": action.get("incident_id"),
            "description": action.get("description"),
            "status": action.get("status"),
            "follow_up": action.get("follow_up"),
            "assignee": action.get("assignee"),
            "created_at": action.get("created_at"),
        }

    async def update_action(
        self,
        credentials: IntegrationCredentials,
        action_id: str,
        description: Optional[str] = None,
        assignee_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update an action.

        Args:
            credentials: Integration credentials
            action_id: Action ID
            description: New description
            assignee_id: New assignee ID
            status: New status (outstanding, completed, deleted, not_doing)

        Returns:
            Updated action data
        """
        headers = self._get_headers(credentials)

        action_data: dict[str, Any] = {}
        if description is not None:
            action_data["description"] = description
        if assignee_id is not None:
            action_data["assignee_id"] = assignee_id
        if status is not None:
            action_data["status"] = status

        response = await self.http_client.put(
            f"{self.API_BASE_URL}/v1/actions/{action_id}",
            headers=headers,
            json=action_data,
        )
        response.raise_for_status()

        action = response.json().get("action", {})
        return {
            "id": action.get("id"),
            "incident_id": action.get("incident_id"),
            "description": action.get("description"),
            "status": action.get("status"),
            "follow_up": action.get("follow_up"),
            "assignee": action.get("assignee"),
            "updated_at": action.get("updated_at"),
        }

    async def complete_action(
        self,
        credentials: IntegrationCredentials,
        action_id: str,
    ) -> dict[str, Any]:
        """Mark an action as completed.

        Args:
            credentials: Integration credentials
            action_id: Action ID

        Returns:
            Updated action data
        """
        return await self.update_action(
            credentials=credentials,
            action_id=action_id,
            status="completed",
        )

    # Incident Update (Announcement) Operations

    async def create_incident_update(
        self,
        credentials: IntegrationCredentials,
        incident_id: str,
        message: str,
        new_severity_id: Optional[str] = None,
        new_status_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create an incident update (announcement).

        Args:
            credentials: Integration credentials
            incident_id: Incident ID
            message: Update message
            new_severity_id: New severity ID (optional status change)
            new_status_id: New status ID (optional status change)

        Returns:
            Created update data
        """
        headers = self._get_headers(credentials)

        update_data: dict[str, Any] = {
            "incident_id": incident_id,
            "message": message,
        }

        if new_severity_id:
            update_data["new_severity_id"] = new_severity_id
        if new_status_id:
            update_data["new_incident_status_id"] = new_status_id

        response = await self.http_client.post(
            f"{self.API_BASE_URL}/v2/incident_updates",
            headers=headers,
            json=update_data,
        )
        response.raise_for_status()

        update = response.json().get("incident_update", {})
        return {
            "id": update.get("id"),
            "incident_id": update.get("incident_id"),
            "message": update.get("message"),
            "new_status": update.get("new_incident_status"),
            "new_severity": update.get("new_severity"),
            "created_at": update.get("created_at"),
        }

    async def list_users(
        self,
        credentials: IntegrationCredentials,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List users in the organization.

        Args:
            credentials: Integration credentials
            limit: Maximum number of users to return

        Returns:
            List of users
        """
        response = await self.http_client.get(
            f"{self.API_BASE_URL}/v2/users",
            headers=self._get_headers(credentials),
            params={"page_size": min(limit, 100)},
        )
        response.raise_for_status()

        users = []
        for user in response.json().get("users", []):
            users.append({
                "id": user.get("id"),
                "name": user.get("name"),
                "email": user.get("email"),
                "role": user.get("role"),
                "slack_user_id": user.get("slack_user_id"),
            })

        return users

    async def get_severities(
        self,
        credentials: IntegrationCredentials,
    ) -> list[dict[str, Any]]:
        """Get all severity levels.

        Args:
            credentials: Integration credentials

        Returns:
            List of severities
        """
        return await self._sync_severities(self._get_headers(credentials))

    async def get_incident_types(
        self,
        credentials: IntegrationCredentials,
    ) -> list[dict[str, Any]]:
        """Get all incident types.

        Args:
            credentials: Integration credentials

        Returns:
            List of incident types
        """
        return await self._sync_incident_types(self._get_headers(credentials))

    async def get_incident_statuses(
        self,
        credentials: IntegrationCredentials,
    ) -> list[dict[str, Any]]:
        """Get all incident statuses.

        Args:
            credentials: Integration credentials

        Returns:
            List of incident statuses
        """
        return await self._sync_statuses(self._get_headers(credentials))
