"""LaunchDarkly integration plugin.

This plugin provides integration with LaunchDarkly for feature flag management.
It supports syncing flags, environments, segments, and audit logs via the
LaunchDarkly REST API v2.
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


class LaunchDarklyPlugin(IntegrationPlugin):
    """LaunchDarkly integration for feature flag management.

    LaunchDarkly is a feature management platform that enables teams to
    safely deploy and control feature rollouts. This plugin provides
    access to flags, environments, segments, and audit logs.

    API Reference: https://apidocs.launchdarkly.com/reference

    Authentication:
        Uses API Access Token (Bearer token) for authentication.
        Tokens can be created at: https://app.launchdarkly.com/settings/authorization

    Example:
        async with LaunchDarklyPlugin() as plugin:
            credentials = IntegrationCredentials(
                access_token="api-xxx",
                extra={"project_key": "my-project"}
            )
            result = await plugin.test_connection(credentials)
    """

    BASE_URL = "https://app.launchdarkly.com/api/v2"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return metadata about the LaunchDarkly integration."""
        return IntegrationMetadata(
            slug="launchdarkly",
            name="LaunchDarkly",
            description="Feature management platform for safe deployments and controlled rollouts",
            category=IntegrationCategory.FEATURE_FLAGS,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://launchdarkly.com/favicon.ico",
            docs_url="https://docs.launchdarkly.com/",
            website_url="https://launchdarkly.com/",
            features=[
                "feature_flags",
                "targeting_rules",
                "segments",
                "environments",
                "audit_log",
                "flag_variations",
                "flag_prerequisites",
                "rollout_percentages",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Access Token",
                    type="password",
                    required=True,
                    placeholder="api-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    help_text="LaunchDarkly API access token from Settings > Authorization",
                ),
                FieldDefinition(
                    name="project_key",
                    label="Project Key",
                    type="text",
                    required=True,
                    placeholder="my-project",
                    help_text="The key of the LaunchDarkly project to access",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="environment_key",
                    label="Environment Key",
                    type="text",
                    required=False,
                    placeholder="production",
                    help_text="Filter to a specific environment (optional)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["flags", "environments", "segments", "audit_log"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Build authentication headers for LaunchDarkly API requests."""
        return {
            "Authorization": credentials.access_token or "",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get_project_key(self, credentials: IntegrationCredentials) -> str:
        """Extract project key from credentials."""
        return credentials.extra.get("project_key", "default")

    def _get_environment_key(self, credentials: IntegrationCredentials) -> Optional[str]:
        """Extract optional environment key from credentials."""
        return credentials.extra.get("environment_key")

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the LaunchDarkly API connection.

        Tests connectivity by fetching flags for the specified project.
        This validates both the API token and project access.

        Args:
            credentials: Must include access_token and project_key in extra.

        Returns:
            ConnectionTestResult with success status and account info.
        """
        start_time = time.time()
        project_key = self._get_project_key(credentials)

        try:
            # Test by fetching flags for the project
            response = await self.http_client.get(
                f"{self.BASE_URL}/flags/{project_key}",
                headers=self._get_headers(credentials),
                params={"limit": 1},  # Just need to verify access
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid API access token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            if response.status_code == 404:
                return ConnectionTestResult(
                    success=False,
                    message=f"Project '{project_key}' not found",
                    latency_ms=latency_ms,
                    error_code="PROJECT_NOT_FOUND",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access token does not have permission to access this project",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            response.raise_for_status()
            data = response.json()

            # Get additional account info
            account_response = await self.http_client.get(
                f"{self.BASE_URL}/projects/{project_key}",
                headers=self._get_headers(credentials),
            )
            account_info = {}
            if account_response.status_code == 200:
                project_data = account_response.json()
                account_info = {
                    "project_name": project_data.get("name"),
                    "project_key": project_key,
                    "environment_count": len(project_data.get("environments", [])),
                    "environments": [
                        env.get("key") for env in project_data.get("environments", [])
                    ],
                }

            return ConnectionTestResult(
                success=True,
                message="Successfully connected to LaunchDarkly",
                latency_ms=latency_ms,
                api_version="v2",
                permissions=["read:flags", "read:environments", "read:segments"],
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
            logger.exception("LaunchDarkly connection test failed")
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
        """Sync data from LaunchDarkly.

        Syncs flags, environments, segments, and audit log entries
        from the specified project.

        Args:
            credentials: Must include access_token and project_key.
            resources: List of resources to sync. Options: flags, environments,
                      segments, audit_log. If None, syncs all.
            since: Sync data modified after this datetime (for audit_log).
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced data organized by resource type.
        """
        started_at = datetime.utcnow()
        project_key = self._get_project_key(credentials)
        environment_key = self._get_environment_key(credentials)

        # Default to all resources if none specified
        resources_to_sync = resources or ["flags", "environments", "segments", "audit_log"]

        synced_data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_points = 0

        try:
            # Sync flags
            if "flags" in resources_to_sync:
                flags_result = await self._sync_flags(
                    credentials, project_key, environment_key, limit
                )
                if flags_result.get("error"):
                    errors.append(flags_result["error"])
                else:
                    synced_data["flags"] = flags_result.get("data", [])
                    total_points += len(synced_data["flags"])

            # Sync environments
            if "environments" in resources_to_sync:
                envs_result = await self._sync_environments(credentials, project_key)
                if envs_result.get("error"):
                    errors.append(envs_result["error"])
                else:
                    synced_data["environments"] = envs_result.get("data", [])
                    total_points += len(synced_data["environments"])

            # Sync segments
            if "segments" in resources_to_sync:
                segments_result = await self._sync_segments(
                    credentials, project_key, environment_key, limit
                )
                if segments_result.get("error"):
                    errors.append(segments_result["error"])
                else:
                    synced_data["segments"] = segments_result.get("data", [])
                    total_points += len(synced_data["segments"])

            # Sync audit log
            if "audit_log" in resources_to_sync:
                audit_result = await self._sync_audit_log(
                    credentials, since, limit
                )
                if audit_result.get("error"):
                    errors.append(audit_result["error"])
                else:
                    synced_data["audit_log"] = audit_result.get("data", [])
                    total_points += len(synced_data["audit_log"])

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_points} items from LaunchDarkly"
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
            logger.exception("LaunchDarkly sync failed")
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

    async def _sync_flags(
        self,
        credentials: IntegrationCredentials,
        project_key: str,
        environment_key: Optional[str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync feature flags from LaunchDarkly."""
        try:
            params: dict[str, Any] = {"limit": limit}
            if environment_key:
                params["env"] = environment_key

            response = await self.http_client.get(
                f"{self.BASE_URL}/flags/{project_key}",
                headers=self._get_headers(credentials),
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            flags = []
            for item in data.get("items", []):
                flag = {
                    "key": item.get("key"),
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "kind": item.get("kind"),  # boolean, multivariate, etc.
                    "temporary": item.get("temporary", False),
                    "tags": item.get("tags", []),
                    "creation_date": item.get("creationDate"),
                    "variations": item.get("variations", []),
                    "defaults": item.get("defaults"),
                    "environments": self._extract_flag_environments(item),
                }
                flags.append(flag)

            return {"data": flags}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch flags: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch flags: {str(e)}"}

    def _extract_flag_environments(self, flag_data: dict) -> list[dict]:
        """Extract environment-specific flag configuration."""
        environments = []
        for env_key, env_config in flag_data.get("environments", {}).items():
            env_info = {
                "key": env_key,
                "on": env_config.get("on", False),
                "targeting_rules": self._extract_targeting_rules(env_config),
                "fallthrough": env_config.get("fallthrough"),
                "off_variation": env_config.get("offVariation"),
                "prerequisites": env_config.get("prerequisites", []),
                "targets": self._extract_targets(env_config),
            }
            environments.append(env_info)
        return environments

    def _extract_targeting_rules(self, env_config: dict) -> list[dict]:
        """Extract targeting rules from environment config."""
        rules = []
        for rule in env_config.get("rules", []):
            rules.append({
                "id": rule.get("_id"),
                "description": rule.get("description"),
                "clauses": rule.get("clauses", []),
                "variation": rule.get("variation"),
                "rollout": rule.get("rollout"),
                "track_events": rule.get("trackEvents", False),
            })
        return rules

    def _extract_targets(self, env_config: dict) -> list[dict]:
        """Extract individual user targets from environment config."""
        targets = []
        for target in env_config.get("targets", []):
            targets.append({
                "variation": target.get("variation"),
                "values": target.get("values", []),
            })
        return targets

    async def _sync_environments(
        self,
        credentials: IntegrationCredentials,
        project_key: str,
    ) -> dict[str, Any]:
        """Sync environments from LaunchDarkly."""
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/projects/{project_key}/environments",
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            data = response.json()

            environments = []
            for item in data.get("items", []):
                env = {
                    "key": item.get("key"),
                    "name": item.get("name"),
                    "color": item.get("color"),
                    "default_ttl": item.get("defaultTtl"),
                    "secure_mode": item.get("secureMode", False),
                    "default_track_events": item.get("defaultTrackEvents", False),
                    "require_comments": item.get("requireComments", False),
                    "confirm_changes": item.get("confirmChanges", False),
                    "tags": item.get("tags", []),
                    "api_key": item.get("apiKey"),  # SDK key (may be masked)
                    "mobile_key": item.get("mobileKey"),
                }
                environments.append(env)

            return {"data": environments}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch environments: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch environments: {str(e)}"}

    async def _sync_segments(
        self,
        credentials: IntegrationCredentials,
        project_key: str,
        environment_key: Optional[str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync user segments from LaunchDarkly."""
        try:
            # If no specific environment, get all environments first
            if not environment_key:
                envs_response = await self.http_client.get(
                    f"{self.BASE_URL}/projects/{project_key}/environments",
                    headers=self._get_headers(credentials),
                )
                envs_response.raise_for_status()
                envs_data = envs_response.json()
                env_keys = [env.get("key") for env in envs_data.get("items", [])]
            else:
                env_keys = [environment_key]

            all_segments = []
            for env_key in env_keys[:3]:  # Limit to first 3 envs to avoid rate limits
                response = await self.http_client.get(
                    f"{self.BASE_URL}/segments/{project_key}/{env_key}",
                    headers=self._get_headers(credentials),
                    params={"limit": limit},
                )
                if response.status_code != 200:
                    continue

                data = response.json()
                for item in data.get("items", []):
                    segment = {
                        "key": item.get("key"),
                        "name": item.get("name"),
                        "description": item.get("description"),
                        "environment": env_key,
                        "tags": item.get("tags", []),
                        "creation_date": item.get("creationDate"),
                        "included": item.get("included", []),  # Included user keys
                        "excluded": item.get("excluded", []),  # Excluded user keys
                        "rules": item.get("rules", []),  # Segment rules
                        "unbounded": item.get("unbounded", False),
                    }
                    all_segments.append(segment)

            return {"data": all_segments}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch segments: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch segments: {str(e)}"}

    async def _sync_audit_log(
        self,
        credentials: IntegrationCredentials,
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync audit log entries from LaunchDarkly."""
        try:
            params: dict[str, Any] = {"limit": limit}
            if since:
                # LaunchDarkly uses milliseconds since epoch
                params["after"] = int(since.timestamp() * 1000)

            response = await self.http_client.get(
                f"{self.BASE_URL}/auditlog",
                headers=self._get_headers(credentials),
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            entries = []
            for item in data.get("items", []):
                entry = {
                    "id": item.get("_id"),
                    "date": item.get("date"),
                    "kind": item.get("kind"),  # flag, project, environment, etc.
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "short_description": item.get("shortDescription"),
                    "member": item.get("member", {}).get("email"),
                    "title_verb": item.get("titleVerb"),
                    "title": item.get("title"),
                    "target": item.get("target"),
                    "parent": item.get("parent"),
                }
                entries.append(entry)

            return {"data": entries}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch audit log: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch audit log: {str(e)}"}

    def validate_credentials(
        self,
        credentials: IntegrationCredentials,
    ) -> tuple[bool, list[str]]:
        """Validate that required LaunchDarkly credentials are provided.

        Args:
            credentials: The credentials to validate.

        Returns:
            Tuple of (is_valid, list_of_missing_fields).
        """
        missing = []

        if not credentials.access_token:
            missing.append("access_token")

        if not credentials.extra.get("project_key"):
            missing.append("project_key")

        return (len(missing) == 0, missing)
