"""Unleash integration plugin.

This plugin provides integration with Unleash for feature flag management.
Unleash requires a self-hosted instance URL. Syncs features, strategies,
and environments via the Unleash Admin API.
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


class UnleashPlugin(IntegrationPlugin):
    """Unleash integration for feature flag management.

    Unleash is an open-source feature management solution. It is typically
    self-hosted, so an instance URL is required. This plugin provides access
    to features (toggles), strategies, environments, and projects.

    API Reference: https://docs.getunleash.io/reference/api/unleash

    Authentication:
        Uses Admin API Token for authentication.
        Tokens can be created in the Unleash admin UI under API Access.

    Example:
        async with UnleashPlugin() as plugin:
            credentials = IntegrationCredentials(
                access_token="*:*.xxxxxxxxxxxxxxxxxxxx",
                instance_url="https://unleash.mycompany.com",
            )
            result = await plugin.test_connection(credentials)
    """

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return metadata about the Unleash integration."""
        return IntegrationMetadata(
            slug="unleash",
            name="Unleash",
            description="Open-source feature management solution",
            category=IntegrationCategory.FEATURE_FLAGS,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://www.getunleash.io/favicon.ico",
            docs_url="https://docs.getunleash.io/",
            website_url="https://www.getunleash.io/",
            features=[
                "feature_flags",
                "feature_toggles",
                "strategies",
                "targeting_rules",
                "environments",
                "projects",
                "constraints",
                "variants",
                "self_hosted",
                "open_source",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Admin API Token",
                    type="password",
                    required=True,
                    placeholder="*:*.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    help_text="Unleash Admin API token from API Access settings",
                ),
                FieldDefinition(
                    name="instance_url",
                    label="Instance URL",
                    type="url",
                    required=True,
                    placeholder="https://unleash.mycompany.com",
                    help_text="URL of your Unleash instance",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="project_id",
                    label="Project ID",
                    type="text",
                    required=False,
                    placeholder="default",
                    help_text="Filter to a specific project (optional)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["features", "strategies", "environments"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the base URL for Unleash API requests."""
        if not credentials.instance_url:
            raise ValueError("Unleash requires instance_url")
        base = credentials.instance_url.rstrip("/")
        return f"{base}/api/admin"

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Build authentication headers for Unleash API requests."""
        return {
            "Authorization": credentials.access_token or "",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get_project_id(self, credentials: IntegrationCredentials) -> str:
        """Extract project ID from credentials."""
        return credentials.extra.get("project_id", "default")

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the Unleash API connection.

        Tests connectivity by fetching features from the Admin API.
        This validates the API token and instance URL.

        Args:
            credentials: Must include access_token and instance_url.

        Returns:
            ConnectionTestResult with success status and account info.
        """
        start_time = time.time()

        try:
            base_url = self._get_base_url(credentials)
        except ValueError as e:
            return ConnectionTestResult(
                success=False,
                message=str(e),
                error_code="MISSING_INSTANCE_URL",
            )

        try:
            # Test by fetching features
            response = await self.http_client.get(
                f"{base_url}/features",
                headers=self._get_headers(credentials),
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid Admin API token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="API token does not have required permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            if response.status_code == 404:
                return ConnectionTestResult(
                    success=False,
                    message="API endpoint not found. Check your instance URL",
                    latency_ms=latency_ms,
                    error_code="NOT_FOUND",
                )

            response.raise_for_status()
            data = response.json()

            # Get instance info
            account_info = {
                "instance_url": credentials.instance_url,
                "feature_count": len(data.get("features", [])),
            }

            # Try to get version info
            try:
                version_response = await self.http_client.get(
                    f"{credentials.instance_url.rstrip('/')}/api",
                    headers=self._get_headers(credentials),
                    timeout=5.0,
                )
                if version_response.status_code == 200:
                    version_data = version_response.json()
                    account_info["version"] = version_data.get("version")
            except Exception:
                pass  # Version info is optional

            # Try to get projects
            try:
                projects_response = await self.http_client.get(
                    f"{base_url}/projects",
                    headers=self._get_headers(credentials),
                )
                if projects_response.status_code == 200:
                    projects_data = projects_response.json()
                    projects = projects_data.get("projects", [])
                    account_info["project_count"] = len(projects)
                    account_info["projects"] = [
                        {"id": p.get("id"), "name": p.get("name")}
                        for p in projects[:5]
                    ]
            except Exception:
                pass  # Project info is optional

            return ConnectionTestResult(
                success=True,
                message="Successfully connected to Unleash",
                latency_ms=latency_ms,
                api_version="admin",
                permissions=["read:features", "read:strategies", "read:environments"],
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
                message=f"Connection error: {str(e)} (instance: {credentials.instance_url})",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="CONNECTION_ERROR",
            )
        except Exception as e:
            logger.exception("Unleash connection test failed")
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
        """Sync data from Unleash.

        Syncs features, strategies, and environments.

        Args:
            credentials: Must include access_token and instance_url.
            resources: List of resources to sync. Options: features, strategies,
                      environments. If None, syncs all.
            since: Not used by Unleash API (included for interface compatibility).
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced data organized by resource type.
        """
        started_at = datetime.utcnow()

        try:
            base_url = self._get_base_url(credentials)
        except ValueError as e:
            return SyncResult(
                success=False,
                message=str(e),
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

        project_id = self._get_project_id(credentials)

        # Default to all resources if none specified
        resources_to_sync = resources or ["features", "strategies", "environments"]

        synced_data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_points = 0

        try:
            # Sync features
            if "features" in resources_to_sync:
                features_result = await self._sync_features(
                    credentials, base_url, project_id, limit
                )
                if features_result.get("error"):
                    errors.append(features_result["error"])
                else:
                    synced_data["features"] = features_result.get("data", [])
                    total_points += len(synced_data["features"])

            # Sync strategies
            if "strategies" in resources_to_sync:
                strategies_result = await self._sync_strategies(credentials, base_url)
                if strategies_result.get("error"):
                    errors.append(strategies_result["error"])
                else:
                    synced_data["strategies"] = strategies_result.get("data", [])
                    total_points += len(synced_data["strategies"])

            # Sync environments
            if "environments" in resources_to_sync:
                envs_result = await self._sync_environments(credentials, base_url)
                if envs_result.get("error"):
                    errors.append(envs_result["error"])
                else:
                    synced_data["environments"] = envs_result.get("data", [])
                    total_points += len(synced_data["environments"])

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_points} items from Unleash"
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
            logger.exception("Unleash sync failed")
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

    async def _sync_features(
        self,
        credentials: IntegrationCredentials,
        base_url: str,
        project_id: str,
        limit: int,
    ) -> dict[str, Any]:
        """Sync features (toggles) from Unleash."""
        try:
            # Try project-specific endpoint first
            response = await self.http_client.get(
                f"{base_url}/projects/{project_id}/features",
                headers=self._get_headers(credentials),
            )

            # Fall back to general features endpoint if project endpoint fails
            if response.status_code == 404:
                response = await self.http_client.get(
                    f"{base_url}/features",
                    headers=self._get_headers(credentials),
                )

            response.raise_for_status()
            data = response.json()

            features = []
            items = data.get("features", [])[:limit]

            for item in items:
                feature = {
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "type": item.get("type", "release"),  # release, experiment, etc.
                    "project": item.get("project", project_id),
                    "enabled": item.get("enabled", False),
                    "stale": item.get("stale", False),
                    "impression_data": item.get("impressionData", False),
                    "created_at": item.get("createdAt"),
                    "last_seen_at": item.get("lastSeenAt"),
                    "strategies": self._extract_strategies(item),
                    "variants": self._extract_variants(item),
                    "environments": self._extract_feature_environments(item),
                    "tags": item.get("tags", []),
                    "dependencies": item.get("dependencies", []),
                }
                features.append(feature)

            return {"data": features}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch features: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch features: {str(e)}"}

    def _extract_strategies(self, feature: dict) -> list[dict]:
        """Extract strategies from feature data."""
        strategies = []
        for strategy in feature.get("strategies", []):
            strategies.append({
                "id": strategy.get("id"),
                "name": strategy.get("name"),
                "title": strategy.get("title"),
                "disabled": strategy.get("disabled", False),
                "sort_order": strategy.get("sortOrder"),
                "parameters": strategy.get("parameters", {}),
                "constraints": self._extract_constraints(strategy),
                "segments": strategy.get("segments", []),
                "variants": strategy.get("variants", []),
            })
        return strategies

    def _extract_constraints(self, strategy: dict) -> list[dict]:
        """Extract constraints from strategy data."""
        constraints = []
        for constraint in strategy.get("constraints", []):
            constraints.append({
                "context_name": constraint.get("contextName"),
                "operator": constraint.get("operator"),
                "value": constraint.get("value"),
                "values": constraint.get("values", []),
                "inverted": constraint.get("inverted", False),
                "case_insensitive": constraint.get("caseInsensitive", False),
            })
        return constraints

    def _extract_variants(self, feature: dict) -> list[dict]:
        """Extract variants from feature data."""
        variants = []
        for variant in feature.get("variants", []):
            variants.append({
                "name": variant.get("name"),
                "weight": variant.get("weight"),
                "weight_type": variant.get("weightType", "variable"),
                "stickiness": variant.get("stickiness"),
                "payload": variant.get("payload"),
                "overrides": variant.get("overrides", []),
            })
        return variants

    def _extract_feature_environments(self, feature: dict) -> list[dict]:
        """Extract environment-specific feature configuration."""
        environments = []
        for env in feature.get("environments", []):
            environments.append({
                "name": env.get("name"),
                "type": env.get("type"),
                "enabled": env.get("enabled", False),
                "sort_order": env.get("sortOrder"),
                "variant_count": len(env.get("variants", [])),
                "strategy_count": len(env.get("strategies", [])),
                "last_seen_at": env.get("lastSeenAt"),
            })
        return environments

    async def _sync_strategies(
        self,
        credentials: IntegrationCredentials,
        base_url: str,
    ) -> dict[str, Any]:
        """Sync strategies from Unleash."""
        try:
            response = await self.http_client.get(
                f"{base_url}/strategies",
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            data = response.json()

            strategies = []
            for item in data.get("strategies", []):
                strategy = {
                    "name": item.get("name"),
                    "display_name": item.get("displayName"),
                    "description": item.get("description"),
                    "editable": item.get("editable", True),
                    "deprecated": item.get("deprecated", False),
                    "built_in": not item.get("editable", True),
                    "parameters": self._extract_strategy_parameters(item),
                }
                strategies.append(strategy)

            return {"data": strategies}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch strategies: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch strategies: {str(e)}"}

    def _extract_strategy_parameters(self, strategy: dict) -> list[dict]:
        """Extract parameter definitions from strategy data."""
        parameters = []
        for param in strategy.get("parameters", []):
            parameters.append({
                "name": param.get("name"),
                "type": param.get("type"),
                "description": param.get("description"),
                "required": param.get("required", False),
            })
        return parameters

    async def _sync_environments(
        self,
        credentials: IntegrationCredentials,
        base_url: str,
    ) -> dict[str, Any]:
        """Sync environments from Unleash."""
        try:
            response = await self.http_client.get(
                f"{base_url}/environments",
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            data = response.json()

            environments = []
            for item in data.get("environments", []):
                env = {
                    "name": item.get("name"),
                    "type": item.get("type"),  # development, test, preproduction, production
                    "enabled": item.get("enabled", True),
                    "protected": item.get("protected", False),
                    "sort_order": item.get("sortOrder"),
                    "project_count": item.get("projectCount"),
                }
                environments.append(env)

            return {"data": environments}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch environments: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch environments: {str(e)}"}

    async def get_feature_by_name(
        self,
        credentials: IntegrationCredentials,
        feature_name: str,
        project_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get a specific feature by name.

        Args:
            credentials: Unleash credentials.
            feature_name: The name of the feature to retrieve.
            project_id: Optional project ID (defaults to credential's project_id).

        Returns:
            Dictionary with feature details.
        """
        try:
            base_url = self._get_base_url(credentials)
            project = project_id or self._get_project_id(credentials)

            response = await self.http_client.get(
                f"{base_url}/projects/{project}/features/{feature_name}",
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            item = response.json()

            return {
                "name": item.get("name"),
                "description": item.get("description"),
                "type": item.get("type"),
                "project": item.get("project"),
                "enabled": item.get("enabled", False),
                "stale": item.get("stale", False),
                "strategies": self._extract_strategies(item),
                "variants": self._extract_variants(item),
                "environments": self._extract_feature_environments(item),
                "tags": item.get("tags", []),
            }

        except Exception as e:
            return {"error": f"Failed to get feature: {str(e)}"}

    def validate_credentials(
        self,
        credentials: IntegrationCredentials,
    ) -> tuple[bool, list[str]]:
        """Validate that required Unleash credentials are provided.

        Args:
            credentials: The credentials to validate.

        Returns:
            Tuple of (is_valid, list_of_missing_fields).
        """
        missing = []

        if not credentials.access_token:
            missing.append("access_token")

        if not credentials.instance_url:
            missing.append("instance_url")

        return (len(missing) == 0, missing)
