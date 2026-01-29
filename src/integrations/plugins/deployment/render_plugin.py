"""Render deployment integration plugin.

This module provides integration with Render for deployment tracking,
service management, and deploy monitoring.

API Documentation: https://api-docs.render.com/reference/introduction
"""

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


class RenderPlugin(IntegrationPlugin):
    """Render deployment platform integration.

    Render is a unified cloud platform for hosting web applications,
    static sites, background workers, cron jobs, databases, and more.

    Features:
    - Track service deployments
    - Monitor deployment status
    - List services and their configurations
    - View deployment logs and history

    Authentication:
    - API Key (Bearer token)
    - Get your API key from: https://dashboard.render.com/account/api-keys

    Usage:
        plugin = RenderPlugin()
        credentials = IntegrationCredentials(api_key="rnd_xxx")
        result = await plugin.test_connection(credentials)
    """

    BASE_URL = "https://api.render.com/v1"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Render plugin metadata."""
        return IntegrationMetadata(
            slug="render",
            name="Render",
            description="Deploy and manage web services, static sites, and databases on Render",
            category=IntegrationCategory.DEPLOYMENT,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://render.com/favicon.ico",
            docs_url="https://api-docs.render.com/reference/introduction",
            website_url="https://render.com",
            features=[
                "service_tracking",
                "deployment_monitoring",
                "deployment_history",
                "service_management",
                "environment_variables",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key",
                    type="password",
                    required=True,
                    placeholder="rnd_xxxxxxxxxxxx",
                    help_text="Get your API key from https://dashboard.render.com/account/api-keys",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra.owner_id",
                    label="Owner ID",
                    type="text",
                    required=False,
                    placeholder="usr_xxxx or tea_xxxx",
                    help_text="Filter services by owner (user or team ID)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["services", "deploys", "environment_groups"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for Render API requests."""
        return {
            "Authorization": f"Bearer {credentials.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the Render API connection.

        Tests by fetching the owner information using GET /owners endpoint.

        Args:
            credentials: Credentials containing the API key.

        Returns:
            ConnectionTestResult with success status and owner info.
        """
        if not credentials.api_key:
            return ConnectionTestResult(
                success=False,
                message="API key is required",
                error_code="MISSING_API_KEY",
            )

        start_time = time.time()

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/owners",
                headers=self._get_headers(credentials),
                params={"limit": 1},
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid API key",
                    latency_ms=latency_ms,
                    error_code="INVALID_API_KEY",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="API key does not have required permissions",
                    latency_ms=latency_ms,
                    error_code="INSUFFICIENT_PERMISSIONS",
                )

            response.raise_for_status()
            data = response.json()

            owners = data if isinstance(data, list) else data.get("owners", [])
            owner_info = {}
            permissions = []

            if owners:
                owner = owners[0].get("owner", owners[0])
                owner_info = {
                    "id": owner.get("id"),
                    "name": owner.get("name"),
                    "email": owner.get("email"),
                    "type": owner.get("type"),  # user or team
                }
                permissions = ["read:owners", "read:services", "read:deploys"]

            return ConnectionTestResult(
                success=True,
                message="Successfully connected to Render",
                latency_ms=latency_ms,
                api_version="v1",
                permissions=permissions,
                account_info=owner_info,
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except httpx.HTTPStatusError as e:
            return ConnectionTestResult(
                success=False,
                message=f"HTTP error: {e.response.status_code}",
                error_code="HTTP_ERROR",
                error_details={"status_code": e.response.status_code},
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection failed: {str(e)}",
                error_code="CONNECTION_ERROR",
                error_details={"error": str(e)},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Render.

        Syncs services and deployments from Render.

        Args:
            credentials: Credentials containing the API key.
            resources: Optional list of resources to sync ("services", "deploys").
            since: Optional datetime to sync data since (for incremental sync).
            limit: Maximum number of items to sync per resource.

        Returns:
            SyncResult with synced services and deployments.
        """
        if not credentials.api_key:
            return SyncResult(
                success=False,
                message="API key is required",
            )

        started_at = datetime.utcnow()
        resources_to_sync = resources or self.metadata.sync_resources
        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        try:
            # Sync services
            if "services" in resources_to_sync:
                services_data = await self._sync_services(credentials, limit)
                data["services"] = services_data["services"]
                total_synced += len(services_data["services"])
                if services_data.get("errors"):
                    errors.extend(services_data["errors"])

            # Sync deploys (requires services first)
            if "deploys" in resources_to_sync:
                if "services" not in data:
                    services_data = await self._sync_services(credentials, limit)
                    data["services"] = services_data["services"]

                deploys_data = await self._sync_deploys(
                    credentials,
                    data["services"],
                    since,
                    limit,
                )
                data["deploys"] = deploys_data["deploys"]
                total_synced += len(deploys_data["deploys"])
                if deploys_data.get("errors"):
                    errors.extend(deploys_data["errors"])

            # Sync environment groups
            if "environment_groups" in resources_to_sync:
                env_groups_data = await self._sync_env_groups(credentials, limit)
                data["environment_groups"] = env_groups_data["environment_groups"]
                total_synced += len(env_groups_data["environment_groups"])
                if env_groups_data.get("errors"):
                    errors.extend(env_groups_data["errors"])

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Render"
                if len(errors) == 0
                else f"Synced with {len(errors)} errors",
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                data=data,
            )

    async def _sync_services(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> dict[str, Any]:
        """Sync services from Render."""
        services: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            params: dict[str, Any] = {"limit": min(limit, 100)}
            owner_id = credentials.extra.get("owner_id")
            if owner_id:
                params["ownerId"] = owner_id

            response = await self.http_client.get(
                f"{self.BASE_URL}/services",
                headers=self._get_headers(credentials),
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            raw_services = data if isinstance(data, list) else data.get("services", [])

            for item in raw_services:
                service = item.get("service", item)
                services.append(
                    {
                        "id": service.get("id"),
                        "name": service.get("name"),
                        "type": service.get("type"),  # web_service, static_site, etc.
                        "repo": service.get("repo"),
                        "branch": service.get("branch"),
                        "auto_deploy": service.get("autoDeploy"),
                        "suspended": service.get("suspended"),
                        "created_at": service.get("createdAt"),
                        "updated_at": service.get("updatedAt"),
                        "dashboard_url": service.get("dashboardUrl"),
                        "service_details": service.get("serviceDetails", {}),
                    }
                )

        except httpx.HTTPStatusError as e:
            errors.append(f"Failed to fetch services: HTTP {e.response.status_code}")
        except Exception as e:
            errors.append(f"Failed to fetch services: {str(e)}")

        return {"services": services, "errors": errors}

    async def _sync_deploys(
        self,
        credentials: IntegrationCredentials,
        services: list[dict[str, Any]],
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync deployments for all services."""
        deploys: list[dict[str, Any]] = []
        errors: list[str] = []

        # Limit deploys per service to avoid too many requests
        deploys_per_service = max(1, limit // max(len(services), 1))

        for service in services:
            service_id = service.get("id")
            if not service_id:
                continue

            try:
                params: dict[str, Any] = {"limit": min(deploys_per_service, 20)}

                response = await self.http_client.get(
                    f"{self.BASE_URL}/services/{service_id}/deploys",
                    headers=self._get_headers(credentials),
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                raw_deploys = data if isinstance(data, list) else data.get("deploys", [])

                for item in raw_deploys:
                    deploy = item.get("deploy", item)

                    # Filter by since date if provided
                    created_at_str = deploy.get("createdAt")
                    if since and created_at_str:
                        try:
                            created_at = datetime.fromisoformat(
                                created_at_str.replace("Z", "+00:00")
                            )
                            if created_at.replace(tzinfo=None) < since:
                                continue
                        except ValueError:
                            pass

                    deploys.append(
                        {
                            "id": deploy.get("id"),
                            "service_id": service_id,
                            "service_name": service.get("name"),
                            "commit": {
                                "id": deploy.get("commit", {}).get("id"),
                                "message": deploy.get("commit", {}).get("message"),
                                "created_at": deploy.get("commit", {}).get("createdAt"),
                            },
                            "status": deploy.get("status"),
                            "trigger": deploy.get("trigger"),
                            "created_at": deploy.get("createdAt"),
                            "updated_at": deploy.get("updatedAt"),
                            "finished_at": deploy.get("finishedAt"),
                        }
                    )

            except httpx.HTTPStatusError as e:
                errors.append(
                    f"Failed to fetch deploys for service {service_id}: HTTP {e.response.status_code}"
                )
            except Exception as e:
                errors.append(f"Failed to fetch deploys for service {service_id}: {str(e)}")

        return {"deploys": deploys, "errors": errors}

    async def _sync_env_groups(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> dict[str, Any]:
        """Sync environment groups from Render."""
        env_groups: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            params: dict[str, Any] = {"limit": min(limit, 100)}
            owner_id = credentials.extra.get("owner_id")
            if owner_id:
                params["ownerId"] = owner_id

            response = await self.http_client.get(
                f"{self.BASE_URL}/env-groups",
                headers=self._get_headers(credentials),
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            raw_groups = data if isinstance(data, list) else data.get("envGroups", [])

            for item in raw_groups:
                group = item.get("envGroup", item)
                env_groups.append(
                    {
                        "id": group.get("id"),
                        "name": group.get("name"),
                        "owner_id": group.get("ownerId"),
                        "created_at": group.get("createdAt"),
                        "updated_at": group.get("updatedAt"),
                        "service_links": group.get("serviceLinks", []),
                    }
                )

        except httpx.HTTPStatusError as e:
            errors.append(f"Failed to fetch environment groups: HTTP {e.response.status_code}")
        except Exception as e:
            errors.append(f"Failed to fetch environment groups: {str(e)}")

        return {"environment_groups": env_groups, "errors": errors}

    async def get_service(
        self,
        credentials: IntegrationCredentials,
        service_id: str,
    ) -> Optional[dict[str, Any]]:
        """Get a specific service by ID.

        Args:
            credentials: Credentials containing the API key.
            service_id: The service ID.

        Returns:
            Service details or None if not found.
        """
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/services/{service_id}",
                headers=self._get_headers(credentials),
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()
            return data.get("service", data)

        except Exception:
            return None

    async def get_deploy(
        self,
        credentials: IntegrationCredentials,
        service_id: str,
        deploy_id: str,
    ) -> Optional[dict[str, Any]]:
        """Get a specific deployment by ID.

        Args:
            credentials: Credentials containing the API key.
            service_id: The service ID.
            deploy_id: The deployment ID.

        Returns:
            Deployment details or None if not found.
        """
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/services/{service_id}/deploys/{deploy_id}",
                headers=self._get_headers(credentials),
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()
            return data.get("deploy", data)

        except Exception:
            return None

    async def trigger_deploy(
        self,
        credentials: IntegrationCredentials,
        service_id: str,
        clear_cache: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Trigger a new deployment for a service.

        Args:
            credentials: Credentials containing the API key.
            service_id: The service ID to deploy.
            clear_cache: Whether to clear the build cache.

        Returns:
            New deployment details or None on failure.
        """
        try:
            response = await self.http_client.post(
                f"{self.BASE_URL}/services/{service_id}/deploys",
                headers=self._get_headers(credentials),
                json={"clearCache": "clear" if clear_cache else "do_not_clear"},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("deploy", data)

        except Exception:
            return None
