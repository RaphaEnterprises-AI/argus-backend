"""Netlify deployment integration plugin.

This plugin provides integration with Netlify's deployment platform,
allowing sync of sites, deploys, and builds.
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


class NetlifyPlugin(IntegrationPlugin):
    """Netlify deployment integration plugin.

    Integrates with Netlify's REST API for:
    - Sites: List and manage Netlify sites
    - Deploys: Access deployment history and status
    - Builds: Monitor build processes

    Authentication: Personal Access Token
    API Documentation: https://docs.netlify.com/api/get-started/
    """

    BASE_URL = "https://api.netlify.com/api/v1"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Netlify integration metadata."""
        return IntegrationMetadata(
            slug="netlify",
            name="Netlify",
            description="Build, deploy, and scale modern web projects",
            category=IntegrationCategory.DEPLOYMENT,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://www.netlify.com/v3/img/components/logomark.png",
            docs_url="https://docs.netlify.com/api/get-started/",
            website_url="https://www.netlify.com",
            features=[
                "sites",
                "deploys",
                "builds",
                "forms",
                "functions",
                "deploy_previews",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Personal Access Token",
                    type="password",
                    required=True,
                    placeholder="Enter your Netlify access token",
                    help_text="Generate from Netlify Dashboard > User Settings > Applications > Personal access tokens",
                ),
            ],
            optional_fields=[],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["sites", "deploys"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get authorization headers for Netlify API requests."""
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Netlify API.

        Tests authentication by fetching the current user's information.

        Args:
            credentials: IntegrationCredentials with access_token.

        Returns:
            ConnectionTestResult indicating success or failure.
        """
        start_time = time.time()

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/user",
                headers=self._get_headers(credentials),
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()

                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to Netlify",
                    latency_ms=latency_ms,
                    api_version="v1",
                    permissions=["read", "write"],
                    account_info={
                        "user_id": data.get("id"),
                        "email": data.get("email"),
                        "full_name": data.get("full_name"),
                        "avatar_url": data.get("avatar_url"),
                        "slug": data.get("slug"),
                        "created_at": data.get("created_at"),
                    },
                )
            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid or expired access token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )
            elif response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden - check token permissions",
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
        """Sync data from Netlify.

        Syncs sites and deploys from Netlify.

        Args:
            credentials: IntegrationCredentials with access_token.
            resources: List of resources to sync (sites, deploys).
            since: Optional datetime for incremental sync (not used for sites).
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
            resources = ["sites", "deploys"]

        headers = self._get_headers(credentials)

        try:
            # Sync sites first (needed for deploys)
            sites_list = []
            if "sites" in resources or "deploys" in resources:
                try:
                    sites_list = await self._sync_sites(headers, limit)
                    if "sites" in resources:
                        data["sites"] = sites_list
                        total_points += len(sites_list)
                except Exception as e:
                    errors.append(f"Failed to sync sites: {str(e)}")

            # Sync deploys for each site
            if "deploys" in resources and sites_list:
                try:
                    all_deploys = []
                    deploys_per_site = max(1, limit // len(sites_list)) if sites_list else limit

                    for site in sites_list[:10]:  # Limit to 10 sites to avoid rate limits
                        try:
                            site_deploys = await self._sync_site_deploys(
                                headers, site["id"], deploys_per_site, since
                            )
                            all_deploys.extend(site_deploys)
                        except Exception as e:
                            warnings.append(
                                f"Failed to sync deploys for site {site.get('name', site['id'])}: {str(e)}"
                            )

                    data["deploys"] = all_deploys
                    total_points += len(all_deploys)
                except Exception as e:
                    errors.append(f"Failed to sync deploys: {str(e)}")

            return SyncResult(
                success=len(errors) == 0,
                message=(
                    f"Synced {total_points} items from Netlify"
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

    async def _sync_sites(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync sites from Netlify."""
        response = await self.http_client.get(
            f"{self.BASE_URL}/sites",
            headers=headers,
            params={"per_page": min(limit, 100)},
        )
        response.raise_for_status()

        sites = response.json()

        return [
            {
                "id": site.get("id"),
                "name": site.get("name"),
                "url": site.get("url"),
                "ssl_url": site.get("ssl_url"),
                "admin_url": site.get("admin_url"),
                "screenshot_url": site.get("screenshot_url"),
                "created_at": site.get("created_at"),
                "updated_at": site.get("updated_at"),
                "state": site.get("state"),
                "build_settings": {
                    "repo_url": site.get("build_settings", {}).get("repo_url"),
                    "repo_branch": site.get("build_settings", {}).get("repo_branch"),
                    "cmd": site.get("build_settings", {}).get("cmd"),
                    "dir": site.get("build_settings", {}).get("dir"),
                },
                "published_deploy": {
                    "id": site.get("published_deploy", {}).get("id") if site.get("published_deploy") else None,
                    "state": site.get("published_deploy", {}).get("state") if site.get("published_deploy") else None,
                },
                "custom_domain": site.get("custom_domain"),
                "default_domain": site.get("default_domain"),
                "branch_deploy_custom_domain": site.get("branch_deploy_custom_domain"),
                "deploy_preview_custom_domain": site.get("deploy_preview_custom_domain"),
            }
            for site in sites
        ]

    async def _sync_site_deploys(
        self,
        headers: dict[str, str],
        site_id: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """Sync deploys for a specific site."""
        response = await self.http_client.get(
            f"{self.BASE_URL}/sites/{site_id}/deploys",
            headers=headers,
            params={"per_page": min(limit, 100)},
        )
        response.raise_for_status()

        deploys = response.json()

        # Filter by since date if provided
        if since:
            since_ts = since.isoformat()
            deploys = [d for d in deploys if d.get("created_at", "") >= since_ts]

        return [
            {
                "id": deploy.get("id"),
                "site_id": site_id,
                "state": deploy.get("state"),
                "name": deploy.get("name"),
                "url": deploy.get("url"),
                "ssl_url": deploy.get("ssl_url"),
                "deploy_url": deploy.get("deploy_url"),
                "deploy_ssl_url": deploy.get("deploy_ssl_url"),
                "admin_url": deploy.get("admin_url"),
                "created_at": deploy.get("created_at"),
                "updated_at": deploy.get("updated_at"),
                "published_at": deploy.get("published_at"),
                "title": deploy.get("title"),
                "context": deploy.get("context"),
                "branch": deploy.get("branch"),
                "commit_ref": deploy.get("commit_ref"),
                "commit_url": deploy.get("commit_url"),
                "review_id": deploy.get("review_id"),
                "review_url": deploy.get("review_url"),
                "locked": deploy.get("locked"),
                "log_access_attributes": {
                    "type": deploy.get("log_access_attributes", {}).get("type"),
                    "url": deploy.get("log_access_attributes", {}).get("url"),
                },
                "error_message": deploy.get("error_message"),
                "deploy_time": deploy.get("deploy_time"),
            }
            for deploy in deploys
        ]

    async def get_deploy_status(
        self,
        credentials: IntegrationCredentials,
        site_id: str,
        deploy_id: str,
    ) -> dict[str, Any]:
        """Get detailed status of a specific deploy.

        Args:
            credentials: IntegrationCredentials with access_token.
            site_id: The site ID.
            deploy_id: The deploy ID to check.

        Returns:
            Dictionary with deploy status details.
        """
        headers = self._get_headers(credentials)

        response = await self.http_client.get(
            f"{self.BASE_URL}/sites/{site_id}/deploys/{deploy_id}",
            headers=headers,
        )
        response.raise_for_status()

        deploy = response.json()
        return {
            "id": deploy.get("id"),
            "site_id": site_id,
            "state": deploy.get("state"),
            "url": deploy.get("ssl_url") or deploy.get("url"),
            "created_at": deploy.get("created_at"),
            "published_at": deploy.get("published_at"),
            "deploy_time": deploy.get("deploy_time"),
            "error_message": deploy.get("error_message"),
            "summary": deploy.get("summary", {}),
            "screenshot_url": deploy.get("screenshot_url"),
            "skipped": deploy.get("skipped"),
            "review_id": deploy.get("review_id"),
        }

    async def trigger_build(
        self,
        credentials: IntegrationCredentials,
        site_id: str,
        clear_cache: bool = False,
    ) -> dict[str, Any]:
        """Trigger a new build for a site.

        Args:
            credentials: IntegrationCredentials with access_token.
            site_id: The site ID to build.
            clear_cache: Whether to clear the build cache.

        Returns:
            Dictionary with the new deploy information.
        """
        headers = self._get_headers(credentials)

        params = {}
        if clear_cache:
            params["clear_cache"] = "true"

        response = await self.http_client.post(
            f"{self.BASE_URL}/sites/{site_id}/builds",
            headers=headers,
            params=params,
        )
        response.raise_for_status()

        data = response.json()
        return {
            "build_id": data.get("id"),
            "deploy_id": data.get("deploy_id"),
            "done": data.get("done"),
            "error": data.get("error"),
            "created_at": data.get("created_at"),
        }
