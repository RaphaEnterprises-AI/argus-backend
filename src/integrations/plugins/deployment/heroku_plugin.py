"""Heroku deployment integration plugin.

This plugin provides integration with Heroku's Platform API,
allowing sync of apps, dynos, and releases.
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


class HerokuPlugin(IntegrationPlugin):
    """Heroku deployment integration plugin.

    Integrates with Heroku's Platform API for:
    - Apps: List and manage Heroku applications
    - Dynos: Monitor and scale dynos
    - Releases: Access release history and rollback

    Authentication: Bearer token (API Key)
    API Documentation: https://devcenter.heroku.com/articles/platform-api-reference
    """

    BASE_URL = "https://api.heroku.com"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Heroku integration metadata."""
        return IntegrationMetadata(
            slug="heroku",
            name="Heroku",
            description="Cloud platform for building, running, and scaling applications",
            category=IntegrationCategory.DEPLOYMENT,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://www.herokucdn.com/favicons/favicon.ico",
            docs_url="https://devcenter.heroku.com/articles/platform-api-reference",
            website_url="https://www.heroku.com",
            features=[
                "apps",
                "dynos",
                "releases",
                "addons",
                "config_vars",
                "domains",
                "formation",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key",
                    type="password",
                    required=True,
                    placeholder="Enter your Heroku API key",
                    help_text="Generate from Heroku Dashboard > Account Settings > API Key",
                ),
            ],
            optional_fields=[],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["apps", "releases", "dynos"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get authorization headers for Heroku API requests."""
        return {
            "Authorization": f"Bearer {credentials.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.heroku+json; version=3",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Heroku API.

        Tests authentication by fetching the current account information.

        Args:
            credentials: IntegrationCredentials with api_key.

        Returns:
            ConnectionTestResult indicating success or failure.
        """
        start_time = time.time()

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/account",
                headers=self._get_headers(credentials),
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()

                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to Heroku",
                    latency_ms=latency_ms,
                    api_version="v3",
                    permissions=["read", "write"],
                    account_info={
                        "id": data.get("id"),
                        "email": data.get("email"),
                        "name": data.get("name"),
                        "verified": data.get("verified"),
                        "two_factor_authentication": data.get("two_factor_authentication"),
                        "created_at": data.get("created_at"),
                        "last_login": data.get("last_login"),
                    },
                )
            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid or expired API key",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )
            elif response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden - check API key permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                    error_details={"status_code": 403},
                )
            else:
                return ConnectionTestResult(
                    success=False,
                    message=f"Unexpected response: {response.status_code}",
                    latency_ms=latency_ms,
                    error_code="UNEXPECTED_ERROR",
                    error_details={
                        "status_code": response.status_code,
                        "body": response.text[:500],
                    },
                )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except httpx.RequestError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection error: {str(e)}",
                error_code="CONNECTION_ERROR",
                error_details={"exception": str(e)},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Heroku.

        Syncs apps, releases, and dynos from Heroku.

        Args:
            credentials: IntegrationCredentials with api_key.
            resources: List of resources to sync (apps, releases, dynos).
            since: Optional datetime for incremental sync.
            limit: Maximum number of items per resource.

        Returns:
            SyncResult with synced data.
        """
        started_at = datetime.utcnow()
        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_points = 0

        # Default to all resources if none specified
        if resources is None:
            resources = ["apps", "releases", "dynos"]

        headers = self._get_headers(credentials)

        try:
            # Sync apps first (needed for releases and dynos)
            apps_list = []
            if "apps" in resources or "releases" in resources or "dynos" in resources:
                try:
                    apps_list = await self._sync_apps(headers, limit)
                    if "apps" in resources:
                        data["apps"] = apps_list
                        total_points += len(apps_list)
                except Exception as e:
                    errors.append(f"Failed to sync apps: {str(e)}")

            # Sync releases for each app
            if "releases" in resources and apps_list:
                try:
                    all_releases = []
                    releases_per_app = max(1, limit // len(apps_list)) if apps_list else limit

                    for app in apps_list[:20]:  # Limit to 20 apps to avoid rate limits
                        try:
                            app_releases = await self._sync_app_releases(
                                headers, app["id"], releases_per_app, since
                            )
                            all_releases.extend(app_releases)
                        except Exception as e:
                            warnings.append(
                                f"Failed to sync releases for app {app.get('name', app['id'])}: {str(e)}"
                            )

                    data["releases"] = all_releases
                    total_points += len(all_releases)
                except Exception as e:
                    errors.append(f"Failed to sync releases: {str(e)}")

            # Sync dynos for each app
            if "dynos" in resources and apps_list:
                try:
                    all_dynos = []

                    for app in apps_list[:20]:  # Limit to 20 apps to avoid rate limits
                        try:
                            app_dynos = await self._sync_app_dynos(headers, app["id"])
                            all_dynos.extend(app_dynos)
                        except Exception as e:
                            warnings.append(
                                f"Failed to sync dynos for app {app.get('name', app['id'])}: {str(e)}"
                            )

                    data["dynos"] = all_dynos
                    total_points += len(all_dynos)
                except Exception as e:
                    errors.append(f"Failed to sync dynos: {str(e)}")

            return SyncResult(
                success=len(errors) == 0,
                message=(
                    f"Synced {total_points} items from Heroku"
                    if len(errors) == 0
                    else f"Partial sync completed with {len(errors)} errors"
                ),
                data_points_synced=total_points,
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
            )

    async def _sync_apps(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync apps from Heroku."""
        response = await self.http_client.get(
            f"{self.BASE_URL}/apps",
            headers=headers,
        )
        response.raise_for_status()

        apps = response.json()

        return [
            {
                "id": app.get("id"),
                "name": app.get("name"),
                "web_url": app.get("web_url"),
                "git_url": app.get("git_url"),
                "region": app.get("region", {}).get("name"),
                "stack": app.get("build_stack", {}).get("name"),
                "created_at": app.get("created_at"),
                "updated_at": app.get("updated_at"),
                "released_at": app.get("released_at"),
                "maintenance": app.get("maintenance"),
                "owner": {
                    "id": app.get("owner", {}).get("id"),
                    "email": app.get("owner", {}).get("email"),
                },
                "organization": app.get("organization", {}).get("name") if app.get("organization") else None,
                "team": app.get("team", {}).get("name") if app.get("team") else None,
                "space": app.get("space", {}).get("name") if app.get("space") else None,
                "internal_routing": app.get("internal_routing"),
                "acm": app.get("acm"),
            }
            for app in apps[:limit]
        ]

    async def _sync_app_releases(
        self,
        headers: dict[str, str],
        app_id: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """Sync releases for a specific app."""
        response = await self.http_client.get(
            f"{self.BASE_URL}/apps/{app_id}/releases",
            headers=headers,
        )
        response.raise_for_status()

        releases = response.json()

        # Filter by since date if provided
        if since:
            since_ts = since.isoformat()
            releases = [r for r in releases if r.get("created_at", "") >= since_ts]

        return [
            {
                "id": release.get("id"),
                "app_id": app_id,
                "version": release.get("version"),
                "description": release.get("description"),
                "status": release.get("status"),
                "current": release.get("current"),
                "created_at": release.get("created_at"),
                "updated_at": release.get("updated_at"),
                "user": {
                    "id": release.get("user", {}).get("id"),
                    "email": release.get("user", {}).get("email"),
                },
                "slug": {
                    "id": release.get("slug", {}).get("id") if release.get("slug") else None,
                    "commit": release.get("slug", {}).get("commit") if release.get("slug") else None,
                },
                "output_stream_url": release.get("output_stream_url"),
            }
            for release in releases[:limit]
        ]

    async def _sync_app_dynos(
        self,
        headers: dict[str, str],
        app_id: str,
    ) -> list[dict[str, Any]]:
        """Sync dynos for a specific app."""
        response = await self.http_client.get(
            f"{self.BASE_URL}/apps/{app_id}/dynos",
            headers=headers,
        )
        response.raise_for_status()

        dynos = response.json()

        return [
            {
                "id": dyno.get("id"),
                "app_id": app_id,
                "name": dyno.get("name"),
                "type": dyno.get("type"),
                "size": dyno.get("size"),
                "state": dyno.get("state"),
                "created_at": dyno.get("created_at"),
                "updated_at": dyno.get("updated_at"),
                "command": dyno.get("command"),
                "release": {
                    "id": dyno.get("release", {}).get("id") if dyno.get("release") else None,
                    "version": dyno.get("release", {}).get("version") if dyno.get("release") else None,
                },
                "attach_url": dyno.get("attach_url"),
            }
            for dyno in dynos
        ]

    async def get_app_status(
        self,
        credentials: IntegrationCredentials,
        app_id: str,
    ) -> dict[str, Any]:
        """Get detailed status of a specific app.

        Args:
            credentials: IntegrationCredentials with api_key.
            app_id: The app ID or name to check.

        Returns:
            Dictionary with app status details.
        """
        headers = self._get_headers(credentials)

        # Get app info
        app_response = await self.http_client.get(
            f"{self.BASE_URL}/apps/{app_id}",
            headers=headers,
        )
        app_response.raise_for_status()
        app = app_response.json()

        # Get formation (dyno configuration)
        formation_response = await self.http_client.get(
            f"{self.BASE_URL}/apps/{app_id}/formation",
            headers=headers,
        )
        formation_response.raise_for_status()
        formation = formation_response.json()

        # Get dynos (running instances)
        dynos_response = await self.http_client.get(
            f"{self.BASE_URL}/apps/{app_id}/dynos",
            headers=headers,
        )
        dynos_response.raise_for_status()
        dynos = dynos_response.json()

        return {
            "id": app.get("id"),
            "name": app.get("name"),
            "web_url": app.get("web_url"),
            "maintenance": app.get("maintenance"),
            "released_at": app.get("released_at"),
            "formation": [
                {
                    "type": f.get("type"),
                    "quantity": f.get("quantity"),
                    "size": f.get("size"),
                    "command": f.get("command"),
                }
                for f in formation
            ],
            "dynos": [
                {
                    "name": d.get("name"),
                    "type": d.get("type"),
                    "state": d.get("state"),
                    "size": d.get("size"),
                }
                for d in dynos
            ],
            "dyno_summary": {
                "total": len(dynos),
                "running": len([d for d in dynos if d.get("state") == "up"]),
                "crashed": len([d for d in dynos if d.get("state") == "crashed"]),
            },
        }

    async def scale_dynos(
        self,
        credentials: IntegrationCredentials,
        app_id: str,
        dyno_type: str,
        quantity: int,
    ) -> dict[str, Any]:
        """Scale dynos for an app.

        Args:
            credentials: IntegrationCredentials with api_key.
            app_id: The app ID or name.
            dyno_type: The dyno type to scale (e.g., 'web', 'worker').
            quantity: The number of dynos to run.

        Returns:
            Dictionary with the updated formation.
        """
        headers = self._get_headers(credentials)

        response = await self.http_client.patch(
            f"{self.BASE_URL}/apps/{app_id}/formation/{dyno_type}",
            headers=headers,
            json={"quantity": quantity},
        )
        response.raise_for_status()

        data = response.json()
        return {
            "type": data.get("type"),
            "quantity": data.get("quantity"),
            "size": data.get("size"),
            "command": data.get("command"),
        }

    async def rollback(
        self,
        credentials: IntegrationCredentials,
        app_id: str,
        release_version: int,
    ) -> dict[str, Any]:
        """Rollback an app to a previous release.

        Args:
            credentials: IntegrationCredentials with api_key.
            app_id: The app ID or name.
            release_version: The release version to rollback to.

        Returns:
            Dictionary with the new release information.
        """
        headers = self._get_headers(credentials)

        response = await self.http_client.post(
            f"{self.BASE_URL}/apps/{app_id}/releases",
            headers=headers,
            json={"release": f"v{release_version}"},
        )
        response.raise_for_status()

        data = response.json()
        return {
            "id": data.get("id"),
            "version": data.get("version"),
            "description": data.get("description"),
            "status": data.get("status"),
            "created_at": data.get("created_at"),
        }
