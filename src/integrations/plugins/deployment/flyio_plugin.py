"""Fly.io deployment integration plugin.

This module provides integration with Fly.io for deployment tracking,
application management, and machine monitoring.

API Documentation: https://fly.io/docs/machines/api/
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


class FlyioPlugin(IntegrationPlugin):
    """Fly.io deployment platform integration.

    Fly.io is a platform for running applications close to users worldwide
    using lightweight VMs called Fly Machines.

    Features:
    - Track application deployments
    - Monitor machine status and health
    - List applications and their configurations
    - View volumes and storage

    Authentication:
    - Access Token (Bearer token)
    - Get your token from: https://fly.io/user/personal_access_tokens

    Usage:
        plugin = FlyioPlugin()
        credentials = IntegrationCredentials(access_token="fo1_xxx")
        result = await plugin.test_connection(credentials)
    """

    BASE_URL = "https://api.machines.dev/v1"
    # Alternative: https://api.fly.io/graphql for GraphQL API

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Fly.io plugin metadata."""
        return IntegrationMetadata(
            slug="flyio",
            name="Fly.io",
            description="Deploy and manage applications globally on Fly.io",
            category=IntegrationCategory.DEPLOYMENT,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://fly.io/favicon.ico",
            docs_url="https://fly.io/docs/machines/api/",
            website_url="https://fly.io",
            features=[
                "app_tracking",
                "machine_monitoring",
                "deployment_status",
                "volume_management",
                "secrets_management",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Access Token",
                    type="password",
                    required=True,
                    placeholder="fo1_xxxxxxxxxxxx",
                    help_text="Get your token from https://fly.io/user/personal_access_tokens",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="org_slug",
                    label="Organization",
                    type="text",
                    required=False,
                    placeholder="personal or org-slug",
                    help_text="Filter apps by organization (default: all accessible)",
                ),
            ],
            supports_webhooks=False,
            supports_sync=True,
            sync_resources=["apps", "machines", "volumes"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for Fly.io API requests."""
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the Fly.io API connection.

        Tests by fetching user information using the GraphQL API.

        Args:
            credentials: Credentials containing the access token.

        Returns:
            ConnectionTestResult with success status and user info.
        """
        if not credentials.access_token:
            return ConnectionTestResult(
                success=False,
                message="Access token is required",
                error_code="MISSING_ACCESS_TOKEN",
            )

        start_time = time.time()

        try:
            # Use GraphQL API to get viewer info (more reliable for auth check)
            graphql_url = "https://api.fly.io/graphql"
            query = """
            query {
                viewer {
                    id
                    email
                    name
                    organizations {
                        nodes {
                            id
                            slug
                            name
                            type
                        }
                    }
                }
            }
            """

            response = await self.http_client.post(
                graphql_url,
                headers=self._get_headers(credentials),
                json={"query": query},
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid access token",
                    latency_ms=latency_ms,
                    error_code="INVALID_ACCESS_TOKEN",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access token does not have required permissions",
                    latency_ms=latency_ms,
                    error_code="INSUFFICIENT_PERMISSIONS",
                )

            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                error_msg = data["errors"][0].get("message", "Unknown GraphQL error")
                return ConnectionTestResult(
                    success=False,
                    message=f"API error: {error_msg}",
                    latency_ms=latency_ms,
                    error_code="GRAPHQL_ERROR",
                )

            viewer = data.get("data", {}).get("viewer", {})
            orgs = viewer.get("organizations", {}).get("nodes", [])

            account_info = {
                "id": viewer.get("id"),
                "email": viewer.get("email"),
                "name": viewer.get("name"),
                "organizations": [
                    {"slug": org.get("slug"), "name": org.get("name")} for org in orgs
                ],
            }

            permissions = [
                "read:apps",
                "read:machines",
                "read:volumes",
                "read:organizations",
            ]

            return ConnectionTestResult(
                success=True,
                message="Successfully connected to Fly.io",
                latency_ms=latency_ms,
                api_version="v1",
                permissions=permissions,
                account_info=account_info,
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
        """Sync data from Fly.io.

        Syncs applications, machines, and volumes from Fly.io.

        Args:
            credentials: Credentials containing the access token.
            resources: Optional list of resources to sync ("apps", "machines", "volumes").
            since: Optional datetime to sync data since (for incremental sync).
            limit: Maximum number of items to sync per resource.

        Returns:
            SyncResult with synced applications and machines.
        """
        if not credentials.access_token:
            return SyncResult(
                success=False,
                message="Access token is required",
            )

        started_at = datetime.utcnow()
        resources_to_sync = resources or self.metadata.sync_resources
        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        try:
            # Sync apps
            if "apps" in resources_to_sync:
                apps_data = await self._sync_apps(credentials, limit)
                data["apps"] = apps_data["apps"]
                total_synced += len(apps_data["apps"])
                if apps_data.get("errors"):
                    errors.extend(apps_data["errors"])

            # Sync machines (requires apps first)
            if "machines" in resources_to_sync:
                if "apps" not in data:
                    apps_data = await self._sync_apps(credentials, limit)
                    data["apps"] = apps_data["apps"]

                machines_data = await self._sync_machines(
                    credentials,
                    data["apps"],
                    limit,
                )
                data["machines"] = machines_data["machines"]
                total_synced += len(machines_data["machines"])
                if machines_data.get("errors"):
                    errors.extend(machines_data["errors"])

            # Sync volumes (requires apps first)
            if "volumes" in resources_to_sync:
                if "apps" not in data:
                    apps_data = await self._sync_apps(credentials, limit)
                    data["apps"] = apps_data["apps"]

                volumes_data = await self._sync_volumes(
                    credentials,
                    data["apps"],
                    limit,
                )
                data["volumes"] = volumes_data["volumes"]
                total_synced += len(volumes_data["volumes"])
                if volumes_data.get("errors"):
                    errors.extend(volumes_data["errors"])

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Fly.io"
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

    async def _sync_apps(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> dict[str, Any]:
        """Sync applications from Fly.io using GraphQL API."""
        apps: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            # Use GraphQL for better app listing
            graphql_url = "https://api.fly.io/graphql"

            # Build query with optional org filter
            org_slug = credentials.org_slug
            if org_slug:
                query = """
                query($orgSlug: String!) {
                    organization(slug: $orgSlug) {
                        apps {
                            nodes {
                                id
                                name
                                status
                                hostname
                                organization {
                                    slug
                                    name
                                }
                                currentRelease {
                                    id
                                    version
                                    createdAt
                                    status
                                }
                                machines {
                                    nodes {
                                        id
                                    }
                                }
                                regions {
                                    code
                                    name
                                }
                                createdAt
                            }
                        }
                    }
                }
                """
                variables = {"orgSlug": org_slug}
            else:
                query = """
                query {
                    apps {
                        nodes {
                            id
                            name
                            status
                            hostname
                            organization {
                                slug
                                name
                            }
                            currentRelease {
                                id
                                version
                                createdAt
                                status
                            }
                            machines {
                                nodes {
                                    id
                                }
                            }
                            regions {
                                code
                                name
                            }
                            createdAt
                        }
                    }
                }
                """
                variables = {}

            response = await self.http_client.post(
                graphql_url,
                headers=self._get_headers(credentials),
                json={"query": query, "variables": variables},
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                error_msg = data["errors"][0].get("message", "Unknown error")
                errors.append(f"Failed to fetch apps: {error_msg}")
                return {"apps": apps, "errors": errors}

            # Extract apps from response
            if org_slug:
                raw_apps = (
                    data.get("data", {})
                    .get("organization", {})
                    .get("apps", {})
                    .get("nodes", [])
                )
            else:
                raw_apps = data.get("data", {}).get("apps", {}).get("nodes", [])

            for app in raw_apps[:limit]:
                current_release = app.get("currentRelease", {}) or {}
                machines = app.get("machines", {}).get("nodes", [])
                regions = app.get("regions", [])

                apps.append(
                    {
                        "id": app.get("id"),
                        "name": app.get("name"),
                        "status": app.get("status"),
                        "hostname": app.get("hostname"),
                        "organization": {
                            "slug": app.get("organization", {}).get("slug"),
                            "name": app.get("organization", {}).get("name"),
                        },
                        "current_release": {
                            "id": current_release.get("id"),
                            "version": current_release.get("version"),
                            "status": current_release.get("status"),
                            "created_at": current_release.get("createdAt"),
                        }
                        if current_release
                        else None,
                        "machine_count": len(machines),
                        "regions": [r.get("code") for r in regions],
                        "created_at": app.get("createdAt"),
                    }
                )

        except httpx.HTTPStatusError as e:
            errors.append(f"Failed to fetch apps: HTTP {e.response.status_code}")
        except Exception as e:
            errors.append(f"Failed to fetch apps: {str(e)}")

        return {"apps": apps, "errors": errors}

    async def _sync_machines(
        self,
        credentials: IntegrationCredentials,
        apps: list[dict[str, Any]],
        limit: int,
    ) -> dict[str, Any]:
        """Sync machines for all applications."""
        machines: list[dict[str, Any]] = []
        errors: list[str] = []

        # Limit machines per app to avoid too many requests
        machines_per_app = max(1, limit // max(len(apps), 1))

        for app in apps:
            app_name = app.get("name")
            if not app_name:
                continue

            try:
                response = await self.http_client.get(
                    f"{self.BASE_URL}/apps/{app_name}/machines",
                    headers=self._get_headers(credentials),
                )
                response.raise_for_status()
                data = response.json()

                raw_machines = data if isinstance(data, list) else []

                for machine in raw_machines[:machines_per_app]:
                    config = machine.get("config", {}) or {}
                    checks = machine.get("checks", []) or []

                    machines.append(
                        {
                            "id": machine.get("id"),
                            "name": machine.get("name"),
                            "app_name": app_name,
                            "state": machine.get("state"),
                            "region": machine.get("region"),
                            "instance_id": machine.get("instance_id"),
                            "private_ip": machine.get("private_ip"),
                            "config": {
                                "image": config.get("image"),
                                "size": config.get("size"),
                                "env": list(config.get("env", {}).keys()),  # Only keys, not values
                                "services": [
                                    {
                                        "protocol": s.get("protocol"),
                                        "internal_port": s.get("internal_port"),
                                    }
                                    for s in config.get("services", [])
                                ],
                            },
                            "checks": [
                                {
                                    "name": c.get("name"),
                                    "status": c.get("status"),
                                }
                                for c in checks
                            ],
                            "created_at": machine.get("created_at"),
                            "updated_at": machine.get("updated_at"),
                        }
                    )

            except httpx.HTTPStatusError as e:
                if e.response.status_code != 404:  # Skip 404s for apps without machines
                    errors.append(
                        f"Failed to fetch machines for app {app_name}: HTTP {e.response.status_code}"
                    )
            except Exception as e:
                errors.append(f"Failed to fetch machines for app {app_name}: {str(e)}")

        return {"machines": machines, "errors": errors}

    async def _sync_volumes(
        self,
        credentials: IntegrationCredentials,
        apps: list[dict[str, Any]],
        limit: int,
    ) -> dict[str, Any]:
        """Sync volumes for all applications."""
        volumes: list[dict[str, Any]] = []
        errors: list[str] = []

        # Limit volumes per app
        volumes_per_app = max(1, limit // max(len(apps), 1))

        for app in apps:
            app_name = app.get("name")
            if not app_name:
                continue

            try:
                response = await self.http_client.get(
                    f"{self.BASE_URL}/apps/{app_name}/volumes",
                    headers=self._get_headers(credentials),
                )
                response.raise_for_status()
                data = response.json()

                raw_volumes = data if isinstance(data, list) else []

                for volume in raw_volumes[:volumes_per_app]:
                    volumes.append(
                        {
                            "id": volume.get("id"),
                            "name": volume.get("name"),
                            "app_name": app_name,
                            "state": volume.get("state"),
                            "region": volume.get("region"),
                            "size_gb": volume.get("size_gb"),
                            "encrypted": volume.get("encrypted"),
                            "attached_machine_id": volume.get("attached_machine_id"),
                            "created_at": volume.get("created_at"),
                        }
                    )

            except httpx.HTTPStatusError as e:
                if e.response.status_code != 404:  # Skip 404s for apps without volumes
                    errors.append(
                        f"Failed to fetch volumes for app {app_name}: HTTP {e.response.status_code}"
                    )
            except Exception as e:
                errors.append(f"Failed to fetch volumes for app {app_name}: {str(e)}")

        return {"volumes": volumes, "errors": errors}

    async def get_app(
        self,
        credentials: IntegrationCredentials,
        app_name: str,
    ) -> Optional[dict[str, Any]]:
        """Get a specific application by name.

        Args:
            credentials: Credentials containing the access token.
            app_name: The application name.

        Returns:
            Application details or None if not found.
        """
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/apps/{app_name}",
                headers=self._get_headers(credentials),
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return response.json()

        except Exception:
            return None

    async def get_machine(
        self,
        credentials: IntegrationCredentials,
        app_name: str,
        machine_id: str,
    ) -> Optional[dict[str, Any]]:
        """Get a specific machine by ID.

        Args:
            credentials: Credentials containing the access token.
            app_name: The application name.
            machine_id: The machine ID.

        Returns:
            Machine details or None if not found.
        """
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/apps/{app_name}/machines/{machine_id}",
                headers=self._get_headers(credentials),
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return response.json()

        except Exception:
            return None

    async def start_machine(
        self,
        credentials: IntegrationCredentials,
        app_name: str,
        machine_id: str,
    ) -> bool:
        """Start a stopped machine.

        Args:
            credentials: Credentials containing the access token.
            app_name: The application name.
            machine_id: The machine ID.

        Returns:
            True if machine was started successfully.
        """
        try:
            response = await self.http_client.post(
                f"{self.BASE_URL}/apps/{app_name}/machines/{machine_id}/start",
                headers=self._get_headers(credentials),
            )
            return response.status_code == 200

        except Exception:
            return False

    async def stop_machine(
        self,
        credentials: IntegrationCredentials,
        app_name: str,
        machine_id: str,
    ) -> bool:
        """Stop a running machine.

        Args:
            credentials: Credentials containing the access token.
            app_name: The application name.
            machine_id: The machine ID.

        Returns:
            True if machine was stopped successfully.
        """
        try:
            response = await self.http_client.post(
                f"{self.BASE_URL}/apps/{app_name}/machines/{machine_id}/stop",
                headers=self._get_headers(credentials),
            )
            return response.status_code == 200

        except Exception:
            return False
