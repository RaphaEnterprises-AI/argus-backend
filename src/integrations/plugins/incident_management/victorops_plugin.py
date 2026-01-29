"""Splunk On-Call (VictorOps) integration plugin for incident management.

Splunk On-Call (formerly VictorOps) is an incident management platform that helps
DevOps teams detect, respond to, and resolve incidents faster.

API Documentation: https://portal.victorops.com/public/api-docs.html
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


class VictorOpsPlugin(IntegrationPlugin):
    """Integration plugin for Splunk On-Call (VictorOps) incident management.

    Splunk On-Call provides incident response automation, on-call scheduling,
    and post-incident analysis for DevOps and IT operations teams.

    Features:
    - Incident management and alerting
    - On-call scheduling and rotations
    - Team management
    - Escalation policies
    - Post-incident reviews

    Authentication:
    - API Key + API ID authentication via headers
    - X-VO-Api-Id: Your API ID
    - X-VO-Api-Key: Your API Key

    API Base URL: https://api.victorops.com
    """

    BASE_URL = "https://api.victorops.com"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Splunk On-Call (VictorOps) integration metadata."""
        return IntegrationMetadata(
            slug="victorops",
            name="Splunk On-Call (VictorOps)",
            description="Incident response automation with on-call scheduling and team management",
            category=IntegrationCategory.INCIDENT_MANAGEMENT,
            auth_type=AuthType.API_KEY,
            icon_url="https://www.splunk.com/favicon.ico",
            docs_url="https://portal.victorops.com/public/api-docs.html",
            website_url="https://www.splunk.com/en_us/products/on-call.html",
            features=[
                "incidents",
                "teams",
                "on-call",
                "rotations",
                "escalations",
                "users",
                "routing-keys",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key",
                    type="password",
                    required=True,
                    placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    help_text="Your VictorOps API Key from Settings > API",
                ),
                FieldDefinition(
                    name="api_secret",
                    label="API ID",
                    type="password",
                    required=True,
                    placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    help_text="Your VictorOps API ID from Settings > API",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="instance_url",
                    label="API Base URL",
                    type="url",
                    required=False,
                    placeholder="https://api.victorops.com",
                    help_text="Override the default API URL (rarely needed)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["incidents", "teams", "oncall", "users", "rotations"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the base URL for API requests."""
        return credentials.instance_url or self.BASE_URL

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "X-VO-Api-Id": credentials.api_secret or "",  # API ID stored in api_secret
            "X-VO-Api-Key": credentials.api_key or "",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Splunk On-Call (VictorOps) API.

        Tests the connection by fetching the current user's information,
        which validates that the API credentials are valid.

        Args:
            credentials: IntegrationCredentials containing API key and API ID.

        Returns:
            ConnectionTestResult with success status and account information.
        """
        start_time = time.time()

        try:
            base_url = self._get_base_url(credentials)
            headers = self._get_headers(credentials)

            # Test connection by fetching current user
            response = await self.http_client.get(
                f"{base_url}/api-public/v1/user",
                headers=headers,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                users = data.get("users", [])
                user_info = users[0] if users else {}

                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to Splunk On-Call (VictorOps)",
                    latency_ms=latency_ms,
                    api_version="v1",
                    permissions=[
                        "read:incidents",
                        "read:teams",
                        "read:oncall",
                        "read:users",
                    ],
                    account_info={
                        "username": user_info.get("username"),
                        "first_name": user_info.get("firstName"),
                        "last_name": user_info.get("lastName"),
                        "email": user_info.get("email"),
                        "user_count": len(users),
                    },
                )
            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed: Invalid API Key or API ID",
                    latency_ms=latency_ms,
                    error_code="INVALID_CREDENTIALS",
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
                message="Connection timeout: VictorOps API is not responding",
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
            logger.exception("Unexpected error testing VictorOps connection")
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
        """Sync data from Splunk On-Call (VictorOps).

        Syncs incidents, teams, on-call schedules, users, and rotations.

        Args:
            credentials: IntegrationCredentials containing API key and API ID.
            resources: Optional list of resources to sync.
                      If None, syncs all available resources.
            since: Optional datetime to sync data since (incremental sync).
            limit: Maximum number of items to sync per resource.

        Returns:
            SyncResult with synced data and statistics.
        """
        started_at = datetime.utcnow()
        sync_resources = resources or ["incidents", "teams", "oncall", "users", "rotations"]
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

            # Sync teams
            if "teams" in sync_resources:
                teams_result = await self._sync_teams(base_url, headers, limit)
                if teams_result["success"]:
                    data["teams"] = teams_result["data"]
                    total_synced += len(teams_result["data"])
                else:
                    errors.append(f"Failed to sync teams: {teams_result['error']}")

            # Sync on-call schedules
            if "oncall" in sync_resources:
                oncall_result = await self._sync_oncall(base_url, headers)
                if oncall_result["success"]:
                    data["oncall"] = oncall_result["data"]
                    total_synced += len(oncall_result["data"])
                else:
                    errors.append(f"Failed to sync on-call schedules: {oncall_result['error']}")

            # Sync users
            if "users" in sync_resources:
                users_result = await self._sync_users(base_url, headers, limit)
                if users_result["success"]:
                    data["users"] = users_result["data"]
                    total_synced += len(users_result["data"])
                else:
                    errors.append(f"Failed to sync users: {users_result['error']}")

            # Sync rotations
            if "rotations" in sync_resources:
                rotations_result = await self._sync_rotations(base_url, headers)
                if rotations_result["success"]:
                    data["rotations"] = rotations_result["data"]
                    total_synced += len(rotations_result["data"])
                else:
                    errors.append(f"Failed to sync rotations: {rotations_result['error']}")

            completed_at = datetime.utcnow()

            return SyncResult(
                success=len(errors) == 0,
                message=(
                    f"Successfully synced {total_synced} items from VictorOps"
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
            logger.exception("Unexpected error syncing data from VictorOps")
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
        """Sync incidents from VictorOps."""
        try:
            params: dict[str, Any] = {}
            if since:
                # VictorOps uses epoch timestamps
                params["startedAfter"] = int(since.timestamp())

            response = await self.http_client.get(
                f"{base_url}/api-public/v1/incidents",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            incidents = data.get("incidents", [])
            if isinstance(incidents, list):
                # Apply limit
                incidents = incidents[:limit]
                return {
                    "success": True,
                    "data": [self._normalize_incident(inc) for inc in incidents],
                }
            return {"success": True, "data": []}

        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}", "data": []}
        except Exception as e:
            return {"success": False, "error": str(e), "data": []}

    async def _sync_teams(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync teams from VictorOps."""
        try:
            response = await self.http_client.get(
                f"{base_url}/api-public/v1/team",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            teams = data if isinstance(data, list) else data.get("teams", [])
            if isinstance(teams, list):
                teams = teams[:limit]
                return {
                    "success": True,
                    "data": [self._normalize_team(team) for team in teams],
                }
            return {"success": True, "data": []}

        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}", "data": []}
        except Exception as e:
            return {"success": False, "error": str(e), "data": []}

    async def _sync_oncall(
        self,
        base_url: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Sync on-call schedules from VictorOps."""
        try:
            response = await self.http_client.get(
                f"{base_url}/api-public/v1/oncall/current",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            oncall = data.get("teamsOnCall", [])
            if isinstance(oncall, list):
                return {
                    "success": True,
                    "data": [self._normalize_oncall(oc) for oc in oncall],
                }
            return {"success": True, "data": []}

        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}", "data": []}
        except Exception as e:
            return {"success": False, "error": str(e), "data": []}

    async def _sync_users(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync users from VictorOps."""
        try:
            response = await self.http_client.get(
                f"{base_url}/api-public/v1/user",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            users = data.get("users", [])
            if isinstance(users, list):
                users = users[:limit]
                return {
                    "success": True,
                    "data": [self._normalize_user(user) for user in users],
                }
            return {"success": True, "data": []}

        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}", "data": []}
        except Exception as e:
            return {"success": False, "error": str(e), "data": []}

    async def _sync_rotations(
        self,
        base_url: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Sync rotations from VictorOps.

        Note: VictorOps rotations are accessed per-team, so we need to fetch
        teams first and then get rotations for each team.
        """
        try:
            # First get all teams
            teams_response = await self.http_client.get(
                f"{base_url}/api-public/v1/team",
                headers=headers,
            )
            teams_response.raise_for_status()
            teams_data = teams_response.json()

            teams = teams_data if isinstance(teams_data, list) else teams_data.get("teams", [])
            all_rotations: list[dict[str, Any]] = []

            # Get rotations for each team
            for team in teams:
                team_slug = team.get("slug") or team.get("name")
                if not team_slug:
                    continue

                try:
                    rotation_response = await self.http_client.get(
                        f"{base_url}/api-public/v1/team/{team_slug}/oncall/schedule",
                        headers=headers,
                    )
                    if rotation_response.status_code == 200:
                        rotation_data = rotation_response.json()
                        schedules = rotation_data.get("schedules", [])
                        for schedule in schedules:
                            schedule["team_slug"] = team_slug
                            all_rotations.append(self._normalize_rotation(schedule))
                except Exception:
                    # Continue with other teams if one fails
                    pass

            return {"success": True, "data": all_rotations}

        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}", "data": []}
        except Exception as e:
            return {"success": False, "error": str(e), "data": []}

    def _normalize_incident(self, incident: dict[str, Any]) -> dict[str, Any]:
        """Normalize incident data to a standard format."""
        # Convert epoch timestamps to ISO format
        started_at = incident.get("startTime")
        if started_at and isinstance(started_at, (int, float)):
            started_at = datetime.fromtimestamp(started_at / 1000).isoformat()

        last_alert_time = incident.get("lastAlertTime")
        if last_alert_time and isinstance(last_alert_time, (int, float)):
            last_alert_time = datetime.fromtimestamp(last_alert_time / 1000).isoformat()

        return {
            "id": incident.get("incidentNumber"),
            "title": incident.get("entityDisplayName") or incident.get("entityId"),
            "description": incident.get("stateMessage"),
            "status": incident.get("currentPhase"),
            "severity": incident.get("monitorType"),
            "service": incident.get("service"),
            "host": incident.get("host"),
            "started_at": started_at,
            "last_alert_time": last_alert_time,
            "alert_count": incident.get("alertCount", 0),
            "ack_user": incident.get("ackUserId"),
            "ack_time": incident.get("ackTime"),
            "entity_id": incident.get("entityId"),
            "entity_type": incident.get("entityType"),
            "routing_key": incident.get("routingKey"),
            "teams": incident.get("teams", []),
            "transitions": incident.get("transitions", []),
            "source": "victorops",
            "raw": incident,
        }

    def _normalize_team(self, team: dict[str, Any]) -> dict[str, Any]:
        """Normalize team data to a standard format."""
        return {
            "id": team.get("slug") or team.get("_teamSlug"),
            "name": team.get("name"),
            "slug": team.get("slug") or team.get("_teamSlug"),
            "description": team.get("description"),
            "member_count": team.get("memberCount", 0),
            "members": team.get("members", []),
            "admins": team.get("admins", []),
            "default_escalation_policy": team.get("defaultPolicy"),
            "is_default_team": team.get("isDefaultTeam", False),
            "source": "victorops",
            "raw": team,
        }

    def _normalize_oncall(self, oncall: dict[str, Any]) -> dict[str, Any]:
        """Normalize on-call schedule data to a standard format."""
        return {
            "team": oncall.get("team", {}).get("name") or oncall.get("team", {}).get("slug"),
            "team_slug": oncall.get("team", {}).get("slug"),
            "oncall_now": oncall.get("oncallNow", []),
            "overrideoncall_now": oncall.get("overrideoncallNow", []),
            "source": "victorops",
            "raw": oncall,
        }

    def _normalize_user(self, user: dict[str, Any]) -> dict[str, Any]:
        """Normalize user data to a standard format."""
        return {
            "id": user.get("username"),
            "username": user.get("username"),
            "first_name": user.get("firstName"),
            "last_name": user.get("lastName"),
            "email": user.get("email"),
            "admin": user.get("admin", False),
            "created_at": user.get("createdAt"),
            "verified": user.get("verified", False),
            "password_last_updated": user.get("passwordLastUpdated"),
            "source": "victorops",
            "raw": user,
        }

    def _normalize_rotation(self, rotation: dict[str, Any]) -> dict[str, Any]:
        """Normalize rotation data to a standard format."""
        return {
            "id": rotation.get("policySlug") or rotation.get("scheduleSlug"),
            "name": rotation.get("policyName") or rotation.get("scheduleName"),
            "team_slug": rotation.get("team_slug"),
            "schedule": rotation.get("schedule", {}),
            "overrides": rotation.get("overrides", []),
            "policy_slug": rotation.get("policySlug"),
            "rotation_group_slug": rotation.get("rotationGroupSlug"),
            "source": "victorops",
            "raw": rotation,
        }

    async def create_incident(
        self,
        credentials: IntegrationCredentials,
        routing_key: str,
        message_type: str = "CRITICAL",
        entity_id: str = "",
        entity_display_name: str = "",
        state_message: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new incident in VictorOps.

        Args:
            credentials: IntegrationCredentials containing API key and API ID.
            routing_key: The routing key for the alert.
            message_type: Alert type (CRITICAL, WARNING, ACKNOWLEDGEMENT, INFO, RECOVERY).
            entity_id: Unique identifier for the entity this alert relates to.
            entity_display_name: Display name for the entity.
            state_message: Description of the alert state.
            **kwargs: Additional fields to include in the alert payload.

        Returns:
            Created incident data or error information.
        """
        base_url = self._get_base_url(credentials)

        # REST endpoint for creating alerts
        alert_url = f"{base_url}/api-public/v1/incidents/trigger"

        payload = {
            "routing_key": routing_key,
            "message_type": message_type,
            "entity_id": entity_id or f"argus-{int(time.time())}",
            "entity_display_name": entity_display_name,
            "state_message": state_message,
            **kwargs,
        }

        try:
            response = await self.http_client.post(
                alert_url,
                headers=self._get_headers(credentials),
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
        incident_number: str,
        message: Optional[str] = None,
    ) -> dict[str, Any]:
        """Acknowledge an incident in VictorOps.

        Args:
            credentials: IntegrationCredentials containing API key and API ID.
            incident_number: The incident number to acknowledge.
            message: Optional acknowledgement message.

        Returns:
            Updated incident data or error information.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        payload: dict[str, Any] = {
            "userName": "argus-integration",
            "incidentNames": [incident_number],
        }
        if message:
            payload["message"] = message

        try:
            response = await self.http_client.patch(
                f"{base_url}/api-public/v1/incidents/ack",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return {"success": True, "result": response.json()}
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
        incident_number: str,
        message: Optional[str] = None,
    ) -> dict[str, Any]:
        """Resolve an incident in VictorOps.

        Args:
            credentials: IntegrationCredentials containing API key and API ID.
            incident_number: The incident number to resolve.
            message: Optional resolution message.

        Returns:
            Updated incident data or error information.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        payload: dict[str, Any] = {
            "userName": "argus-integration",
            "incidentNames": [incident_number],
        }
        if message:
            payload["message"] = message

        try:
            response = await self.http_client.patch(
                f"{base_url}/api-public/v1/incidents/resolve",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return {"success": True, "result": response.json()}
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_team_oncall(
        self,
        credentials: IntegrationCredentials,
        team_slug: str,
    ) -> dict[str, Any]:
        """Get current on-call users for a specific team.

        Args:
            credentials: IntegrationCredentials containing API key and API ID.
            team_slug: The team slug to get on-call information for.

        Returns:
            On-call information or error.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        try:
            response = await self.http_client.get(
                f"{base_url}/api-public/v1/team/{team_slug}/oncall/current",
                headers=headers,
            )
            response.raise_for_status()
            return {"success": True, "oncall": response.json()}
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_user(
        self,
        credentials: IntegrationCredentials,
        username: str,
    ) -> dict[str, Any]:
        """Get a specific user's information.

        Args:
            credentials: IntegrationCredentials containing API key and API ID.
            username: The username to look up.

        Returns:
            User information or error.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        try:
            response = await self.http_client.get(
                f"{base_url}/api-public/v1/user/{username}",
                headers=headers,
            )
            response.raise_for_status()
            return {"success": True, "user": self._normalize_user(response.json())}
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
