"""Split integration plugin.

This plugin provides integration with Split.io for feature flag management.
It supports syncing splits (feature flags), environments, segments, and
traffic types via the Split Admin API.
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


class SplitPlugin(IntegrationPlugin):
    """Split.io integration for feature flag management.

    Split.io is a feature delivery platform that provides controlled rollouts,
    experimentation, and feature flags. This plugin provides access to splits,
    environments, segments, and traffic types.

    API Reference: https://docs.split.io/reference/introduction-to-admin-api

    Authentication:
        Uses Admin API Key for authentication.
        Keys can be created at: https://app.split.io/admin/settings/api-keys

    Example:
        async with SplitPlugin() as plugin:
            credentials = IntegrationCredentials(
                api_key="admin-api-key",
                extra={"workspace_id": "ws-xxx"}
            )
            result = await plugin.test_connection(credentials)
    """

    BASE_URL = "https://api.split.io/internal/api/v2"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return metadata about the Split integration."""
        return IntegrationMetadata(
            slug="split",
            name="Split",
            description="Feature delivery platform for controlled rollouts and experimentation",
            category=IntegrationCategory.FEATURE_FLAGS,
            auth_type=AuthType.API_KEY,
            icon_url="https://www.split.io/favicon.ico",
            docs_url="https://docs.split.io/",
            website_url="https://www.split.io/",
            features=[
                "feature_flags",
                "splits",
                "targeting_rules",
                "segments",
                "environments",
                "traffic_types",
                "treatments",
                "rollout_percentages",
                "experimentation",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="Admin API Key",
                    type="password",
                    required=True,
                    placeholder="xxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx",
                    help_text="Split Admin API key from Admin Settings > API Keys",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="workspace_id",
                    label="Workspace ID",
                    type="text",
                    required=False,
                    placeholder="ws-xxxxxxxx",
                    help_text="Filter to a specific workspace (optional)",
                ),
                FieldDefinition(
                    name="environment_id",
                    label="Environment ID",
                    type="text",
                    required=False,
                    placeholder="env-xxxxxxxx",
                    help_text="Filter to a specific environment (optional)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["splits", "environments", "segments", "traffic_types"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Build authentication headers for Split API requests."""
        return {
            "Authorization": f"Bearer {credentials.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get_workspace_id(self, credentials: IntegrationCredentials) -> Optional[str]:
        """Extract workspace ID from credentials."""
        return credentials.workspace_id or credentials.extra.get("workspace_id")

    def _get_environment_id(self, credentials: IntegrationCredentials) -> Optional[str]:
        """Extract environment ID from credentials."""
        return credentials.extra.get("environment_id")

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the Split API connection.

        Tests connectivity by fetching splits from the API.
        This validates the API key and access permissions.

        Args:
            credentials: Must include api_key.

        Returns:
            ConnectionTestResult with success status and account info.
        """
        start_time = time.time()

        try:
            # Test by fetching workspaces first
            response = await self.http_client.get(
                f"{self.BASE_URL}/workspaces",
                headers=self._get_headers(credentials),
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid Admin API key",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="API key does not have required permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            response.raise_for_status()
            data = response.json()

            # Extract account info from workspaces
            workspaces = data.get("objects", [])
            account_info = {
                "workspace_count": len(workspaces),
                "workspaces": [
                    {"id": ws.get("id"), "name": ws.get("name")}
                    for ws in workspaces[:5]  # Limit to first 5
                ],
            }

            # If workspace_id specified, verify it exists
            workspace_id = self._get_workspace_id(credentials)
            if workspace_id:
                ws_exists = any(ws.get("id") == workspace_id for ws in workspaces)
                if not ws_exists:
                    return ConnectionTestResult(
                        success=False,
                        message=f"Workspace '{workspace_id}' not found",
                        latency_ms=latency_ms,
                        error_code="WORKSPACE_NOT_FOUND",
                        account_info=account_info,
                    )
                account_info["selected_workspace"] = workspace_id

            return ConnectionTestResult(
                success=True,
                message="Successfully connected to Split",
                latency_ms=latency_ms,
                api_version="v2",
                permissions=["read:splits", "read:environments", "read:segments"],
                account_info=account_info,
            )

        except httpx.HTTPStatusError as e:
            return ConnectionTestResult(
                success=False,
                message=f"HTTP error: {e.response.status_code}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="HTTP_ERROR",
                error_details={"status_code": e.response.status_code},
            )
        except httpx.RequestError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="CONNECTION_ERROR",
            )
        except Exception as e:
            logger.exception("Split connection test failed")
            return ConnectionTestResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="UNEXPECTED_ERROR",
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Split.

        Syncs splits, environments, segments, and traffic types.

        Args:
            credentials: Must include api_key.
            resources: List of resources to sync. Options: splits, environments,
                      segments, traffic_types. If None, syncs all.
            since: Not used by Split API (included for interface compatibility).
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced data organized by resource type.
        """
        started_at = datetime.utcnow()
        workspace_id = self._get_workspace_id(credentials)
        environment_id = self._get_environment_id(credentials)

        # Default to all resources if none specified
        resources_to_sync = resources or ["splits", "environments", "segments", "traffic_types"]

        synced_data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_points = 0

        try:
            # Get workspace(s) to sync
            workspaces = await self._get_workspaces(credentials, workspace_id)
            if not workspaces:
                return SyncResult(
                    success=False,
                    message="No workspaces found or accessible",
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    errors=["No accessible workspaces"],
                )

            # Sync traffic types
            if "traffic_types" in resources_to_sync:
                traffic_result = await self._sync_traffic_types(
                    credentials, workspaces, limit
                )
                if traffic_result.get("error"):
                    errors.append(traffic_result["error"])
                else:
                    synced_data["traffic_types"] = traffic_result.get("data", [])
                    total_points += len(synced_data["traffic_types"])

            # Sync environments
            if "environments" in resources_to_sync:
                envs_result = await self._sync_environments(credentials, workspaces)
                if envs_result.get("error"):
                    errors.append(envs_result["error"])
                else:
                    synced_data["environments"] = envs_result.get("data", [])
                    total_points += len(synced_data["environments"])

            # Sync splits (feature flags)
            if "splits" in resources_to_sync:
                splits_result = await self._sync_splits(
                    credentials, workspaces, environment_id, limit
                )
                if splits_result.get("error"):
                    errors.append(splits_result["error"])
                else:
                    synced_data["splits"] = splits_result.get("data", [])
                    total_points += len(synced_data["splits"])

            # Sync segments
            if "segments" in resources_to_sync:
                segments_result = await self._sync_segments(
                    credentials, workspaces, environment_id, limit
                )
                if segments_result.get("error"):
                    errors.append(segments_result["error"])
                else:
                    synced_data["segments"] = segments_result.get("data", [])
                    total_points += len(synced_data["segments"])

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_points} items from Split"
                if not errors
                else f"Completed with {len(errors)} errors",
                data_points_synced=total_points,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=synced_data,
            )

        except Exception as e:
            logger.exception("Split sync failed")
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

    async def _get_workspaces(
        self,
        credentials: IntegrationCredentials,
        workspace_id: Optional[str],
    ) -> list[dict]:
        """Get workspaces to sync from."""
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/workspaces",
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            data = response.json()

            workspaces = data.get("objects", [])
            if workspace_id:
                workspaces = [ws for ws in workspaces if ws.get("id") == workspace_id]

            return workspaces
        except Exception:
            return []

    async def _sync_splits(
        self,
        credentials: IntegrationCredentials,
        workspaces: list[dict],
        environment_id: Optional[str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync splits (feature flags) from Split."""
        try:
            all_splits = []

            for workspace in workspaces:
                ws_id = workspace.get("id")

                # Get environments for this workspace
                envs_response = await self.http_client.get(
                    f"{self.BASE_URL}/environments/ws/{ws_id}",
                    headers=self._get_headers(credentials),
                )
                if envs_response.status_code != 200:
                    continue

                envs_data = envs_response.json()
                environments = envs_data.get("objects", [])
                if environment_id:
                    environments = [
                        env for env in environments if env.get("id") == environment_id
                    ]

                for env in environments[:3]:  # Limit environments to avoid rate limits
                    env_id = env.get("id")

                    response = await self.http_client.get(
                        f"{self.BASE_URL}/splits/ws/{ws_id}/environments/{env_id}",
                        headers=self._get_headers(credentials),
                        params={"limit": limit, "offset": 0},
                    )
                    if response.status_code != 200:
                        continue

                    data = response.json()
                    for item in data.get("objects", []):
                        split = {
                            "name": item.get("name"),
                            "description": item.get("description"),
                            "workspace_id": ws_id,
                            "workspace_name": workspace.get("name"),
                            "environment_id": env_id,
                            "environment_name": env.get("name"),
                            "traffic_type_id": item.get("trafficTypeId"),
                            "traffic_type_name": item.get("trafficTypeName"),
                            "killed": item.get("killed", False),
                            "treatments": item.get("treatments", []),
                            "default_treatment": item.get("defaultTreatment"),
                            "base_treatment": item.get("baseTreatment"),
                            "rules": self._extract_split_rules(item),
                            "default_rule": item.get("defaultRule", []),
                            "creation_time": item.get("creationTime"),
                            "last_update_time": item.get("lastUpdateTime"),
                            "tags": item.get("tags", []),
                        }
                        all_splits.append(split)

            return {"data": all_splits}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch splits: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch splits: {str(e)}"}

    def _extract_split_rules(self, split_data: dict) -> list[dict]:
        """Extract targeting rules from split data."""
        rules = []
        for rule in split_data.get("rules", []):
            rules.append({
                "condition": rule.get("condition", {}),
                "buckets": rule.get("buckets", []),
            })
        return rules

    async def _sync_environments(
        self,
        credentials: IntegrationCredentials,
        workspaces: list[dict],
    ) -> dict[str, Any]:
        """Sync environments from Split."""
        try:
            all_environments = []

            for workspace in workspaces:
                ws_id = workspace.get("id")
                response = await self.http_client.get(
                    f"{self.BASE_URL}/environments/ws/{ws_id}",
                    headers=self._get_headers(credentials),
                )
                if response.status_code != 200:
                    continue

                data = response.json()
                for item in data.get("objects", []):
                    env = {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "workspace_id": ws_id,
                        "workspace_name": workspace.get("name"),
                        "production": item.get("production", False),
                        "creation_time": item.get("creationTime"),
                        "status": item.get("status"),
                        "permission_type": item.get("permissionType"),
                        "org_id": item.get("orgId"),
                    }
                    all_environments.append(env)

            return {"data": all_environments}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch environments: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch environments: {str(e)}"}

    async def _sync_segments(
        self,
        credentials: IntegrationCredentials,
        workspaces: list[dict],
        environment_id: Optional[str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync segments from Split."""
        try:
            all_segments = []

            for workspace in workspaces:
                ws_id = workspace.get("id")

                # Get environments
                envs_response = await self.http_client.get(
                    f"{self.BASE_URL}/environments/ws/{ws_id}",
                    headers=self._get_headers(credentials),
                )
                if envs_response.status_code != 200:
                    continue

                envs_data = envs_response.json()
                environments = envs_data.get("objects", [])
                if environment_id:
                    environments = [
                        env for env in environments if env.get("id") == environment_id
                    ]

                for env in environments[:3]:
                    env_id = env.get("id")

                    response = await self.http_client.get(
                        f"{self.BASE_URL}/segments/ws/{ws_id}/environments/{env_id}",
                        headers=self._get_headers(credentials),
                        params={"limit": limit, "offset": 0},
                    )
                    if response.status_code != 200:
                        continue

                    data = response.json()
                    for item in data.get("objects", []):
                        segment = {
                            "name": item.get("name"),
                            "description": item.get("description"),
                            "workspace_id": ws_id,
                            "workspace_name": workspace.get("name"),
                            "environment_id": env_id,
                            "environment_name": env.get("name"),
                            "traffic_type_id": item.get("trafficTypeId"),
                            "traffic_type_name": item.get("trafficTypeName"),
                            "creation_time": item.get("creationTime"),
                            "tags": item.get("tags", []),
                        }
                        all_segments.append(segment)

            return {"data": all_segments}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch segments: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch segments: {str(e)}"}

    async def _sync_traffic_types(
        self,
        credentials: IntegrationCredentials,
        workspaces: list[dict],
        limit: int,
    ) -> dict[str, Any]:
        """Sync traffic types from Split."""
        try:
            all_traffic_types = []

            for workspace in workspaces:
                ws_id = workspace.get("id")
                response = await self.http_client.get(
                    f"{self.BASE_URL}/trafficTypes/ws/{ws_id}",
                    headers=self._get_headers(credentials),
                )
                if response.status_code != 200:
                    continue

                data = response.json()
                for item in data.get("objects", []):
                    traffic_type = {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "display_attribute_id": item.get("displayAttributeId"),
                        "workspace_id": ws_id,
                        "workspace_name": workspace.get("name"),
                    }
                    all_traffic_types.append(traffic_type)

            return {"data": all_traffic_types}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch traffic types: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch traffic types: {str(e)}"}

    def validate_credentials(
        self,
        credentials: IntegrationCredentials,
    ) -> tuple[bool, list[str]]:
        """Validate that required Split credentials are provided.

        Args:
            credentials: The credentials to validate.

        Returns:
            Tuple of (is_valid, list_of_missing_fields).
        """
        missing = []

        if not credentials.api_key:
            missing.append("api_key")

        return (len(missing) == 0, missing)
