"""Azure Pipelines integration plugin.

This module provides integration with Azure DevOps Pipelines for CI/CD management.
Supports pipelines, builds, releases, and deployment tracking.
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


class AzurePipelinesPlugin(IntegrationPlugin):
    """Azure Pipelines integration plugin.

    Provides integration with Azure DevOps Pipelines for:
    - Pipeline management and monitoring
    - Build history and status tracking
    - Release management and deployments
    - Agent pool monitoring

    Authentication:
        Uses Personal Access Token (PAT) with Basic Auth.
        Token should have appropriate scopes for Build, Release, and Pipeline access.

    API Reference:
        https://docs.microsoft.com/en-us/rest/api/azure/devops/
    """

    DEFAULT_API_VERSION = "7.1"
    BASE_URL_TEMPLATE = "https://dev.azure.com/{organization}"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Azure Pipelines integration metadata."""
        return IntegrationMetadata(
            slug="azure-pipelines",
            name="Azure Pipelines",
            description="Azure DevOps Pipelines for CI/CD automation and deployment management",
            category=IntegrationCategory.CICD,
            auth_type=AuthType.BASIC_AUTH,
            icon_url="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/azure/azure-original.svg",
            docs_url="https://docs.microsoft.com/en-us/azure/devops/pipelines/",
            website_url="https://azure.microsoft.com/en-us/services/devops/pipelines/",
            features=[
                "pipeline_management",
                "build_monitoring",
                "release_management",
                "deployment_tracking",
                "agent_monitoring",
                "artifact_management",
                "yaml_pipelines",
                "classic_pipelines",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Personal Access Token",
                    type="password",
                    required=True,
                    placeholder="Enter your Azure DevOps PAT",
                    help_text="Create a PAT at https://dev.azure.com/{org}/_usersSettings/tokens",
                ),
                FieldDefinition(
                    name="org_slug",
                    label="Organization",
                    type="text",
                    required=True,
                    placeholder="myorganization",
                    help_text="Your Azure DevOps organization name",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="project_slug",
                    label="Project",
                    type="text",
                    required=False,
                    placeholder="MyProject",
                    help_text="Limit to a specific project (optional)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["pipelines", "builds", "releases", "agents"],
        )

    def _get_base_url(self, organization: str) -> str:
        """Get the base API URL for the organization."""
        return self.BASE_URL_TEMPLATE.format(organization=organization)

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for API requests.

        Azure DevOps uses Basic Auth with empty username and PAT as password.
        """
        import base64

        # Azure DevOps PAT uses empty username with PAT as password
        auth_string = f":{credentials.access_token}"
        encoded = base64.b64encode(auth_string.encode()).decode()

        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the Azure Pipelines connection.

        Uses the /_apis/connectionData endpoint to verify authentication
        and retrieve account information.

        Args:
            credentials: IntegrationCredentials containing access_token and org_slug.

        Returns:
            ConnectionTestResult with connection status and account details.
        """
        if not credentials.access_token:
            return ConnectionTestResult(
                success=False,
                message="Personal Access Token is required",
                error_code="MISSING_TOKEN",
            )

        if not credentials.org_slug:
            return ConnectionTestResult(
                success=False,
                message="Organization name is required",
                error_code="MISSING_ORG",
            )

        base_url = self._get_base_url(credentials.org_slug)
        headers = self._get_headers(credentials)

        start_time = time.time()

        try:
            # Test connection using connection data endpoint
            response = await self.http_client.get(
                f"{base_url}/_apis/connectionData",
                headers=headers,
                params={"api-version": self.DEFAULT_API_VERSION},
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid Personal Access Token",
                    latency_ms=latency_ms,
                    error_code="INVALID_TOKEN",
                )

            if response.status_code == 404:
                return ConnectionTestResult(
                    success=False,
                    message=f"Organization '{credentials.org_slug}' not found",
                    latency_ms=latency_ms,
                    error_code="ORG_NOT_FOUND",
                )

            response.raise_for_status()
            data = response.json()

            # Extract account info
            authenticated_user = data.get("authenticatedUser", {})
            instance_id = data.get("instanceId", "")

            # Determine permissions based on authorized scopes
            permissions = []
            authorized_user = data.get("authorizedUser", {})
            if authorized_user:
                permissions.append("authenticated")

            # Try to get projects to verify project access
            projects_response = await self.http_client.get(
                f"{base_url}/_apis/projects",
                headers=headers,
                params={"api-version": self.DEFAULT_API_VERSION, "$top": 1},
            )
            if projects_response.status_code == 200:
                permissions.append("read_projects")
                projects_data = projects_response.json()
                project_count = projects_data.get("count", 0)
            else:
                project_count = 0

            return ConnectionTestResult(
                success=True,
                message=f"Connected to Azure DevOps organization '{credentials.org_slug}'",
                latency_ms=latency_ms,
                api_version=self.DEFAULT_API_VERSION,
                permissions=permissions,
                account_info={
                    "organization": credentials.org_slug,
                    "instance_id": instance_id,
                    "user_id": authenticated_user.get("id", ""),
                    "user_name": authenticated_user.get("providerDisplayName", ""),
                    "project_count": project_count,
                },
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="TIMEOUT",
            )
        except httpx.HTTPStatusError as e:
            return ConnectionTestResult(
                success=False,
                message=f"HTTP error: {e.response.status_code}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code=f"HTTP_{e.response.status_code}",
                error_details={"response": e.response.text[:500]},
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
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
        """Sync data from Azure Pipelines.

        Syncs pipelines, builds, releases, and agent information.

        Args:
            credentials: IntegrationCredentials for authentication.
            resources: List of resources to sync. Defaults to all.
            since: Only sync items modified after this time.
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced data and statistics.
        """
        started_at = datetime.utcnow()
        sync_id = f"azure-pipelines-{started_at.isoformat()}"

        if not credentials.access_token or not credentials.org_slug:
            return SyncResult(
                success=False,
                message="Missing required credentials",
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=["access_token and org_slug are required"],
            )

        base_url = self._get_base_url(credentials.org_slug)
        headers = self._get_headers(credentials)

        resources_to_sync = resources or ["pipelines", "builds", "releases", "agents"]
        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        # Get project list first
        try:
            projects = await self._get_projects(base_url, headers, limit)
        except Exception as e:
            return SyncResult(
                success=False,
                message=f"Failed to fetch projects: {str(e)}",
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

        data["projects"] = projects

        # Sync pipelines for each project
        if "pipelines" in resources_to_sync:
            all_pipelines = []
            for project in projects[:10]:  # Limit to first 10 projects
                project_name = project.get("name", "")
                try:
                    pipelines = await self._get_pipelines(
                        base_url, headers, project_name, limit
                    )
                    for pipeline in pipelines:
                        pipeline["project"] = project_name
                    all_pipelines.extend(pipelines)
                except Exception as e:
                    warnings.append(f"Failed to fetch pipelines for {project_name}: {str(e)}")

            data["pipelines"] = all_pipelines
            total_synced += len(all_pipelines)

        # Sync builds
        if "builds" in resources_to_sync:
            all_builds = []
            for project in projects[:10]:
                project_name = project.get("name", "")
                try:
                    builds = await self._get_builds(
                        base_url, headers, project_name, since, limit
                    )
                    for build in builds:
                        build["project"] = project_name
                    all_builds.extend(builds)
                except Exception as e:
                    warnings.append(f"Failed to fetch builds for {project_name}: {str(e)}")

            data["builds"] = all_builds
            total_synced += len(all_builds)

        # Sync releases
        if "releases" in resources_to_sync:
            all_releases = []
            # Releases use a different API base URL
            vsrm_url = f"https://vsrm.dev.azure.com/{credentials.org_slug}"
            for project in projects[:10]:
                project_name = project.get("name", "")
                try:
                    releases = await self._get_releases(
                        vsrm_url, headers, project_name, since, limit
                    )
                    for release in releases:
                        release["project"] = project_name
                    all_releases.extend(releases)
                except Exception as e:
                    warnings.append(f"Failed to fetch releases for {project_name}: {str(e)}")

            data["releases"] = all_releases
            total_synced += len(all_releases)

        # Sync agents (organization-level)
        if "agents" in resources_to_sync:
            try:
                agents = await self._get_agents(base_url, headers, limit)
                data["agents"] = agents
                total_synced += len(agents)
            except Exception as e:
                warnings.append(f"Failed to fetch agents: {str(e)}")
                data["agents"] = []

        return SyncResult(
            success=len(errors) == 0,
            message=f"Synced {total_synced} items from Azure Pipelines",
            data_points_synced=total_synced,
            sync_id=sync_id,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            errors=errors,
            warnings=warnings,
            data=data,
        )

    async def _get_projects(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch projects from Azure DevOps."""
        response = await self.http_client.get(
            f"{base_url}/_apis/projects",
            headers=headers,
            params={
                "api-version": self.DEFAULT_API_VERSION,
                "$top": min(limit, 100),
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("value", [])

    async def _get_pipelines(
        self,
        base_url: str,
        headers: dict[str, str],
        project: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch pipelines for a project."""
        response = await self.http_client.get(
            f"{base_url}/{project}/_apis/pipelines",
            headers=headers,
            params={
                "api-version": self.DEFAULT_API_VERSION,
                "$top": min(limit, 100),
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("value", [])

    async def _get_builds(
        self,
        base_url: str,
        headers: dict[str, str],
        project: str,
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch builds for a project."""
        params: dict[str, Any] = {
            "api-version": self.DEFAULT_API_VERSION,
            "$top": min(limit, 100),
        }

        if since:
            params["minTime"] = since.isoformat()

        response = await self.http_client.get(
            f"{base_url}/{project}/_apis/build/builds",
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("value", [])

    async def _get_releases(
        self,
        vsrm_url: str,
        headers: dict[str, str],
        project: str,
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch releases for a project.

        Note: Releases use a different API endpoint (vsrm.dev.azure.com).
        """
        params: dict[str, Any] = {
            "api-version": "7.1",
            "$top": min(limit, 100),
        }

        if since:
            params["minCreatedTime"] = since.isoformat()

        response = await self.http_client.get(
            f"{vsrm_url}/{project}/_apis/release/releases",
            headers=headers,
            params=params,
        )

        # Releases API might return 404 if not enabled for project
        if response.status_code == 404:
            return []

        response.raise_for_status()
        data = response.json()
        return data.get("value", [])

    async def _get_agents(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch agent pools and agents."""
        # First get agent pools
        pools_response = await self.http_client.get(
            f"{base_url}/_apis/distributedtask/pools",
            headers=headers,
            params={"api-version": self.DEFAULT_API_VERSION},
        )
        pools_response.raise_for_status()
        pools_data = pools_response.json()
        pools = pools_data.get("value", [])

        all_agents = []
        for pool in pools[:5]:  # Limit to first 5 pools
            pool_id = pool.get("id")
            if pool_id:
                try:
                    agents_response = await self.http_client.get(
                        f"{base_url}/_apis/distributedtask/pools/{pool_id}/agents",
                        headers=headers,
                        params={"api-version": self.DEFAULT_API_VERSION},
                    )
                    if agents_response.status_code == 200:
                        agents_data = agents_response.json()
                        agents = agents_data.get("value", [])
                        for agent in agents:
                            agent["pool_id"] = pool_id
                            agent["pool_name"] = pool.get("name", "")
                        all_agents.extend(agents)
                except Exception:
                    continue

        return all_agents[:limit]
