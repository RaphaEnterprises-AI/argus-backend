"""Flagsmith integration plugin.

This plugin provides integration with Flagsmith for feature flag management.
It supports both cloud and self-hosted Flagsmith instances. Syncs flags,
environments, and identities via the Flagsmith REST API.
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


class FlagsmithPlugin(IntegrationPlugin):
    """Flagsmith integration for feature flag management.

    Flagsmith is an open-source feature flag and remote config service.
    It can be used as a cloud service or self-hosted. This plugin provides
    access to flags, environments, and identities.

    API Reference: https://docs.flagsmith.com/clients/rest

    Authentication:
        Uses Server-side Environment Key for authentication.
        Keys are available in the Flagsmith dashboard under Environment settings.

    Example:
        async with FlagsmithPlugin() as plugin:
            credentials = IntegrationCredentials(
                api_key="ser.xxxxx",  # Server-side environment key
            )
            result = await plugin.test_connection(credentials)

        # For self-hosted:
        credentials = IntegrationCredentials(
            api_key="ser.xxxxx",
            instance_url="https://flagsmith.mycompany.com",
        )
    """

    DEFAULT_BASE_URL = "https://edge.api.flagsmith.com/api/v1"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return metadata about the Flagsmith integration."""
        return IntegrationMetadata(
            slug="flagsmith",
            name="Flagsmith",
            description="Open-source feature flag and remote config service",
            category=IntegrationCategory.FEATURE_FLAGS,
            auth_type=AuthType.API_KEY,
            icon_url="https://www.flagsmith.com/favicon.ico",
            docs_url="https://docs.flagsmith.com/",
            website_url="https://www.flagsmith.com/",
            features=[
                "feature_flags",
                "remote_config",
                "targeting_rules",
                "segments",
                "environments",
                "identities",
                "traits",
                "self_hosted",
                "open_source",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="Server-side Environment Key",
                    type="password",
                    required=True,
                    placeholder="ser.xxxxxxxxxxxxxxxxxxxx",
                    help_text="Server-side environment key from Flagsmith dashboard",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="instance_url",
                    label="Instance URL",
                    type="url",
                    required=False,
                    placeholder="https://flagsmith.mycompany.com",
                    help_text="URL for self-hosted Flagsmith instance (leave blank for cloud)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["flags", "environments", "identities"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the base URL for Flagsmith API requests."""
        if credentials.instance_url:
            # For self-hosted instances
            base = credentials.instance_url.rstrip("/")
            return f"{base}/api/v1"
        return self.DEFAULT_BASE_URL

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Build authentication headers for Flagsmith API requests."""
        return {
            "X-Environment-Key": credentials.api_key or "",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the Flagsmith API connection.

        Tests connectivity by fetching environments.
        This validates the environment key and access permissions.

        Args:
            credentials: Must include api_key. Optional instance_url for self-hosted.

        Returns:
            ConnectionTestResult with success status and account info.
        """
        start_time = time.time()
        base_url = self._get_base_url(credentials)

        try:
            # Test by fetching flags for the environment
            response = await self.http_client.get(
                f"{base_url}/flags/",
                headers=self._get_headers(credentials),
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid environment key",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Environment key does not have required permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            if response.status_code == 404:
                return ConnectionTestResult(
                    success=False,
                    message="Environment not found or API endpoint not available",
                    latency_ms=latency_ms,
                    error_code="NOT_FOUND",
                )

            response.raise_for_status()
            flags_data = response.json()

            # Get environment info if possible
            account_info = {
                "instance_type": "self-hosted" if credentials.instance_url else "cloud",
                "flag_count": len(flags_data) if isinstance(flags_data, list) else 0,
            }

            # Try to get environment details
            env_response = await self.http_client.get(
                f"{base_url}/environments/",
                headers=self._get_headers(credentials),
            )
            if env_response.status_code == 200:
                env_data = env_response.json()
                if isinstance(env_data, list) and len(env_data) > 0:
                    account_info["environment_name"] = env_data[0].get("name")
                    account_info["project_name"] = env_data[0].get("project", {}).get("name")

            return ConnectionTestResult(
                success=True,
                message="Successfully connected to Flagsmith",
                latency_ms=latency_ms,
                api_version="v1",
                permissions=["read:flags", "read:environments", "read:identities"],
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
            error_message = f"Connection error: {str(e)}"
            if credentials.instance_url:
                error_message += f" (instance: {credentials.instance_url})"
            return ConnectionTestResult(
                success=False,
                message=error_message,
                latency_ms=(time.time() - start_time) * 1000,
                error_code="CONNECTION_ERROR",
            )
        except Exception as e:
            logger.exception("Flagsmith connection test failed")
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
        """Sync data from Flagsmith.

        Syncs flags, environments, and identities.

        Args:
            credentials: Must include api_key.
            resources: List of resources to sync. Options: flags, environments,
                      identities. If None, syncs all.
            since: Not used by Flagsmith API (included for interface compatibility).
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced data organized by resource type.
        """
        started_at = datetime.utcnow()
        base_url = self._get_base_url(credentials)

        # Default to all resources if none specified
        resources_to_sync = resources or ["flags", "environments", "identities"]

        synced_data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_points = 0

        try:
            # Sync flags
            if "flags" in resources_to_sync:
                flags_result = await self._sync_flags(credentials, base_url)
                if flags_result.get("error"):
                    errors.append(flags_result["error"])
                else:
                    synced_data["flags"] = flags_result.get("data", [])
                    total_points += len(synced_data["flags"])

            # Sync environments
            if "environments" in resources_to_sync:
                envs_result = await self._sync_environments(credentials, base_url)
                if envs_result.get("error"):
                    errors.append(envs_result["error"])
                else:
                    synced_data["environments"] = envs_result.get("data", [])
                    total_points += len(synced_data["environments"])

            # Sync identities
            if "identities" in resources_to_sync:
                identities_result = await self._sync_identities(
                    credentials, base_url, limit
                )
                if identities_result.get("error"):
                    errors.append(identities_result["error"])
                else:
                    synced_data["identities"] = identities_result.get("data", [])
                    total_points += len(synced_data["identities"])

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_points} items from Flagsmith"
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
            logger.exception("Flagsmith sync failed")
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
        base_url: str,
    ) -> dict[str, Any]:
        """Sync feature flags from Flagsmith."""
        try:
            response = await self.http_client.get(
                f"{base_url}/flags/",
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            data = response.json()

            flags = []
            items = data if isinstance(data, list) else data.get("results", [])

            for item in items:
                feature = item.get("feature", {})
                flag = {
                    "id": item.get("id"),
                    "feature_id": feature.get("id"),
                    "feature_name": feature.get("name"),
                    "description": feature.get("description"),
                    "type": feature.get("type"),  # FLAG or CONFIG
                    "enabled": item.get("enabled", False),
                    "value": item.get("feature_state_value"),
                    "initial_value": feature.get("initial_value"),
                    "default_enabled": feature.get("default_enabled", False),
                    "is_archived": feature.get("is_archived", False),
                    "created_date": feature.get("created_date"),
                    "tags": feature.get("tags", []),
                    "owners": feature.get("owners", []),
                    "multivariate_options": self._extract_multivariate_options(feature),
                }
                flags.append(flag)

            return {"data": flags}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch flags: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch flags: {str(e)}"}

    def _extract_multivariate_options(self, feature: dict) -> list[dict]:
        """Extract multivariate options from feature data."""
        options = []
        for opt in feature.get("multivariate_options", []):
            options.append({
                "id": opt.get("id"),
                "value": opt.get("string_value"),
                "percentage_allocation": opt.get("default_percentage_allocation"),
            })
        return options

    async def _sync_environments(
        self,
        credentials: IntegrationCredentials,
        base_url: str,
    ) -> dict[str, Any]:
        """Sync environments from Flagsmith."""
        try:
            response = await self.http_client.get(
                f"{base_url}/environments/",
                headers=self._get_headers(credentials),
            )

            if response.status_code == 404:
                # This endpoint may not be available for all API keys
                return {"data": [], "warning": "Environments endpoint not available"}

            response.raise_for_status()
            data = response.json()

            environments = []
            items = data if isinstance(data, list) else data.get("results", [])

            for item in items:
                env = {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "api_key": item.get("api_key"),  # May be masked
                    "project": {
                        "id": item.get("project", {}).get("id"),
                        "name": item.get("project", {}).get("name"),
                    },
                    "description": item.get("description"),
                    "hide_disabled_flags": item.get("hide_disabled_flags", False),
                    "hide_sensitive_data": item.get("hide_sensitive_data", False),
                    "allow_client_traits": item.get("allow_client_traits", True),
                    "minimum_change_request_approvals": item.get(
                        "minimum_change_request_approvals"
                    ),
                    "use_identity_composite_key_for_hashing": item.get(
                        "use_identity_composite_key_for_hashing", False
                    ),
                }
                environments.append(env)

            return {"data": environments}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch environments: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch environments: {str(e)}"}

    async def _sync_identities(
        self,
        credentials: IntegrationCredentials,
        base_url: str,
        limit: int,
    ) -> dict[str, Any]:
        """Sync identities from Flagsmith."""
        try:
            response = await self.http_client.get(
                f"{base_url}/identities/",
                headers=self._get_headers(credentials),
                params={"page_size": limit},
            )

            if response.status_code == 404:
                # This endpoint may not be available for all API keys
                return {"data": [], "warning": "Identities endpoint not available"}

            response.raise_for_status()
            data = response.json()

            identities = []
            items = data.get("results", []) if isinstance(data, dict) else data

            for item in items:
                identity = {
                    "id": item.get("id"),
                    "identifier": item.get("identifier"),
                    "environment": item.get("environment"),
                    "created_date": item.get("created_date"),
                    "flags": self._extract_identity_flags(item),
                    "traits": self._extract_identity_traits(item),
                }
                identities.append(identity)

            return {"data": identities}

        except httpx.HTTPStatusError as e:
            return {"error": f"Failed to fetch identities: HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch identities: {str(e)}"}

    def _extract_identity_flags(self, identity: dict) -> list[dict]:
        """Extract flag overrides for an identity."""
        flags = []
        for flag in identity.get("flags", []):
            flags.append({
                "feature_id": flag.get("feature", {}).get("id"),
                "feature_name": flag.get("feature", {}).get("name"),
                "enabled": flag.get("enabled"),
                "value": flag.get("feature_state_value"),
            })
        return flags

    def _extract_identity_traits(self, identity: dict) -> list[dict]:
        """Extract traits for an identity."""
        traits = []
        for trait in identity.get("traits", []):
            traits.append({
                "trait_key": trait.get("trait_key"),
                "trait_value": trait.get("trait_value"),
            })
        return traits

    async def get_identity_flags(
        self,
        credentials: IntegrationCredentials,
        identifier: str,
        traits: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Get flags for a specific identity.

        This method retrieves the flags for a specific user/identity,
        taking into account any targeting rules and traits.

        Args:
            credentials: Flagsmith credentials.
            identifier: The identity identifier (user ID, etc.).
            traits: Optional list of traits to pass for evaluation.

        Returns:
            Dictionary with identity flags and values.
        """
        base_url = self._get_base_url(credentials)

        try:
            # Use POST to include traits
            payload: dict[str, Any] = {"identifier": identifier}
            if traits:
                payload["traits"] = traits

            response = await self.http_client.post(
                f"{base_url}/identities/",
                headers=self._get_headers(credentials),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return {
                "identifier": data.get("identifier"),
                "flags": self._extract_identity_flags(data),
                "traits": self._extract_identity_traits(data),
            }

        except Exception as e:
            return {"error": f"Failed to get identity flags: {str(e)}"}

    def validate_credentials(
        self,
        credentials: IntegrationCredentials,
    ) -> tuple[bool, list[str]]:
        """Validate that required Flagsmith credentials are provided.

        Args:
            credentials: The credentials to validate.

        Returns:
            Tuple of (is_valid, list_of_missing_fields).
        """
        missing = []

        if not credentials.api_key:
            missing.append("api_key")

        return (len(missing) == 0, missing)
