"""Vercel deployment integration plugin.

This plugin provides integration with Vercel's deployment platform,
allowing sync of deployments, projects, and domains.
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


class VercelPlugin(IntegrationPlugin):
    """Vercel deployment integration plugin.

    Integrates with Vercel's REST API for:
    - Deployments: List and monitor deployment status
    - Projects: Access project configuration and settings
    - Domains: Manage custom domains

    Authentication: Bearer token (API Token)
    API Documentation: https://vercel.com/docs/rest-api
    """

    BASE_URL = "https://api.vercel.com"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Vercel integration metadata."""
        return IntegrationMetadata(
            slug="vercel",
            name="Vercel",
            description="Deploy and host web applications with zero configuration",
            category=IntegrationCategory.DEPLOYMENT,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://assets.vercel.com/image/upload/v1588805858/repositories/vercel/logo.png",
            docs_url="https://vercel.com/docs/rest-api",
            website_url="https://vercel.com",
            features=[
                "deployments",
                "projects",
                "domains",
                "environment_variables",
                "deployment_status",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="API Token",
                    type="password",
                    required=True,
                    placeholder="Enter your Vercel API token",
                    help_text="Generate from Vercel Dashboard > Settings > Tokens",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="team_id",
                    label="Team ID",
                    type="text",
                    required=False,
                    placeholder="team_xxxxxxxxxxxx",
                    help_text="Required for team projects. Found in team settings.",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["deployments", "projects", "domains"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get authorization headers for Vercel API requests."""
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
        }

    def _get_team_params(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get team query parameters if team_id is provided."""
        if credentials.team_id:
            return {"teamId": credentials.team_id}
        return {}

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Vercel API.

        Tests authentication by fetching the current user's information.

        Args:
            credentials: IntegrationCredentials with access_token.

        Returns:
            ConnectionTestResult indicating success or failure.
        """
        start_time = time.time()

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/v2/user",
                headers=self._get_headers(credentials),
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                user = data.get("user", {})

                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to Vercel",
                    latency_ms=latency_ms,
                    api_version="v2",
                    permissions=["read", "write"] if credentials.access_token else [],
                    account_info={
                        "user_id": user.get("id"),
                        "username": user.get("username"),
                        "email": user.get("email"),
                        "name": user.get("name"),
                    },
                )
            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid or expired API token",
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
        """Sync data from Vercel.

        Syncs deployments, projects, and domains from Vercel.

        Args:
            credentials: IntegrationCredentials with access_token.
            resources: List of resources to sync (deployments, projects, domains).
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
            resources = ["deployments", "projects", "domains"]

        headers = self._get_headers(credentials)
        team_params = self._get_team_params(credentials)

        try:
            # Sync projects
            if "projects" in resources:
                try:
                    projects_data = await self._sync_projects(
                        headers, team_params, limit
                    )
                    data["projects"] = projects_data
                    total_points += len(projects_data)
                except Exception as e:
                    errors.append(f"Failed to sync projects: {str(e)}")

            # Sync deployments
            if "deployments" in resources:
                try:
                    deployments_data = await self._sync_deployments(
                        headers, team_params, limit, since
                    )
                    data["deployments"] = deployments_data
                    total_points += len(deployments_data)
                except Exception as e:
                    errors.append(f"Failed to sync deployments: {str(e)}")

            # Sync domains
            if "domains" in resources:
                try:
                    domains_data = await self._sync_domains(headers, team_params, limit)
                    data["domains"] = domains_data
                    total_points += len(domains_data)
                except Exception as e:
                    errors.append(f"Failed to sync domains: {str(e)}")

            return SyncResult(
                success=len(errors) == 0,
                message=(
                    f"Synced {total_points} items from Vercel"
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

    async def _sync_projects(
        self,
        headers: dict[str, str],
        team_params: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync projects from Vercel."""
        response = await self.http_client.get(
            f"{self.BASE_URL}/v9/projects",
            headers=headers,
            params={**team_params, "limit": min(limit, 100)},
        )
        response.raise_for_status()

        data = response.json()
        projects = data.get("projects", [])

        return [
            {
                "id": project.get("id"),
                "name": project.get("name"),
                "framework": project.get("framework"),
                "created_at": project.get("createdAt"),
                "updated_at": project.get("updatedAt"),
                "node_version": project.get("nodeVersion"),
                "build_command": project.get("buildCommand"),
                "output_directory": project.get("outputDirectory"),
                "root_directory": project.get("rootDirectory"),
                "git_repository": project.get("link", {}).get("repo") if project.get("link") else None,
            }
            for project in projects
        ]

    async def _sync_deployments(
        self,
        headers: dict[str, str],
        team_params: dict[str, str],
        limit: int,
        since: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """Sync deployments from Vercel."""
        params = {**team_params, "limit": min(limit, 100)}
        if since:
            params["since"] = int(since.timestamp() * 1000)

        response = await self.http_client.get(
            f"{self.BASE_URL}/v6/deployments",
            headers=headers,
            params=params,
        )
        response.raise_for_status()

        data = response.json()
        deployments = data.get("deployments", [])

        return [
            {
                "id": deployment.get("uid"),
                "name": deployment.get("name"),
                "url": deployment.get("url"),
                "state": deployment.get("state"),
                "ready_state": deployment.get("readyState"),
                "created_at": deployment.get("createdAt"),
                "building_at": deployment.get("buildingAt"),
                "ready_at": deployment.get("ready"),
                "creator": deployment.get("creator", {}).get("username"),
                "meta": deployment.get("meta", {}),
                "target": deployment.get("target"),
                "alias_error": deployment.get("aliasError"),
                "alias_assigned": deployment.get("aliasAssigned"),
            }
            for deployment in deployments
        ]

    async def _sync_domains(
        self,
        headers: dict[str, str],
        team_params: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync domains from Vercel."""
        response = await self.http_client.get(
            f"{self.BASE_URL}/v5/domains",
            headers=headers,
            params={**team_params, "limit": min(limit, 100)},
        )
        response.raise_for_status()

        data = response.json()
        domains = data.get("domains", [])

        return [
            {
                "id": domain.get("id"),
                "name": domain.get("name"),
                "service_type": domain.get("serviceType"),
                "ns_verified_at": domain.get("nsVerifiedAt"),
                "txt_verified_at": domain.get("txtVerifiedAt"),
                "cdn_enabled": domain.get("cdnEnabled"),
                "created_at": domain.get("createdAt"),
                "bought_at": domain.get("boughtAt"),
                "expires_at": domain.get("expiresAt"),
                "verified": domain.get("verified"),
            }
            for domain in domains
        ]

    async def get_deployment_status(
        self,
        credentials: IntegrationCredentials,
        deployment_id: str,
    ) -> dict[str, Any]:
        """Get detailed status of a specific deployment.

        Args:
            credentials: IntegrationCredentials with access_token.
            deployment_id: The deployment ID to check.

        Returns:
            Dictionary with deployment status details.
        """
        headers = self._get_headers(credentials)
        team_params = self._get_team_params(credentials)

        response = await self.http_client.get(
            f"{self.BASE_URL}/v13/deployments/{deployment_id}",
            headers=headers,
            params=team_params,
        )
        response.raise_for_status()

        data = response.json()
        return {
            "id": data.get("id"),
            "url": data.get("url"),
            "state": data.get("state"),
            "ready_state": data.get("readyState"),
            "ready_state_reason": data.get("readyStateReason"),
            "created_at": data.get("createdAt"),
            "building_at": data.get("buildingAt"),
            "ready_at": data.get("ready"),
            "error_message": data.get("errorMessage"),
            "error_code": data.get("errorCode"),
            "build_logs_url": f"https://vercel.com/{data.get('name')}/{deployment_id}/logs",
        }
