"""TeamCity integration plugin.

This module provides integration with JetBrains TeamCity for CI/CD management.
Supports builds, projects, build types, and agent monitoring.
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


class TeamCityPlugin(IntegrationPlugin):
    """TeamCity integration plugin.

    Provides integration with JetBrains TeamCity for:
    - Build monitoring and history
    - Project and build configuration management
    - Agent pool monitoring
    - Build queue management

    Authentication:
        Uses Bearer token authentication.
        Token can be created in TeamCity under User Settings > Access Tokens.

    API Reference:
        https://www.jetbrains.com/help/teamcity/rest-api.html
    """

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get TeamCity integration metadata."""
        return IntegrationMetadata(
            slug="teamcity",
            name="TeamCity",
            description="JetBrains TeamCity for CI/CD build automation and management",
            category=IntegrationCategory.CICD,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://resources.jetbrains.com/storage/products/teamcity/img/meta/teamcity_logo_300x300.png",
            docs_url="https://www.jetbrains.com/help/teamcity/rest-api.html",
            website_url="https://www.jetbrains.com/teamcity/",
            features=[
                "build_monitoring",
                "project_management",
                "build_configuration",
                "agent_monitoring",
                "build_queue",
                "artifact_management",
                "vcs_triggers",
                "build_chains",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Access Token",
                    type="password",
                    required=True,
                    placeholder="Enter your TeamCity access token",
                    help_text="Create at User Settings > Access Tokens in TeamCity",
                ),
                FieldDefinition(
                    name="instance_url",
                    label="Server URL",
                    type="url",
                    required=True,
                    placeholder="https://teamcity.example.com",
                    help_text="Your TeamCity server URL",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="project_slug",
                    label="Project ID",
                    type="text",
                    required=False,
                    placeholder="MyProject",
                    help_text="Limit to a specific project (optional)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["builds", "projects", "build_types", "agents"],
        )

    def _get_base_url(self, instance_url: str) -> str:
        """Get the base API URL."""
        # Remove trailing slash if present
        base = instance_url.rstrip("/")
        return f"{base}/app/rest"

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for API requests using Bearer token."""
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the TeamCity connection.

        Uses the /app/rest/server endpoint to verify authentication
        and retrieve server information.

        Args:
            credentials: IntegrationCredentials containing access_token and instance_url.

        Returns:
            ConnectionTestResult with connection status and server details.
        """
        if not credentials.access_token:
            return ConnectionTestResult(
                success=False,
                message="Access Token is required",
                error_code="MISSING_TOKEN",
            )

        if not credentials.instance_url:
            return ConnectionTestResult(
                success=False,
                message="Server URL is required",
                error_code="MISSING_URL",
            )

        base_url = self._get_base_url(credentials.instance_url)
        headers = self._get_headers(credentials)

        start_time = time.time()

        try:
            # Test connection using /server endpoint
            response = await self.http_client.get(
                f"{base_url}/server",
                headers=headers,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid access token",
                    latency_ms=latency_ms,
                    error_code="INVALID_TOKEN",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Insufficient permissions for this token",
                    latency_ms=latency_ms,
                    error_code="INSUFFICIENT_PERMISSIONS",
                )

            if response.status_code == 404:
                return ConnectionTestResult(
                    success=False,
                    message="TeamCity REST API not found. Check server URL.",
                    latency_ms=latency_ms,
                    error_code="API_NOT_FOUND",
                )

            response.raise_for_status()
            server_data = response.json()

            # Get authenticated user info
            user_response = await self.http_client.get(
                f"{base_url}/users/current",
                headers=headers,
            )

            user_info = {}
            permissions = ["authenticated"]

            if user_response.status_code == 200:
                user_data = user_response.json()
                user_info = {
                    "user_id": user_data.get("id", ""),
                    "username": user_data.get("username", ""),
                    "name": user_data.get("name", ""),
                    "email": user_data.get("email", ""),
                }
                permissions.append("read_user")

            # Check project access
            projects_response = await self.http_client.get(
                f"{base_url}/projects",
                headers=headers,
                params={"locator": "count:1"},
            )

            project_count = 0
            if projects_response.status_code == 200:
                permissions.append("read_projects")
                projects_data = projects_response.json()
                project_count = projects_data.get("count", 0)

            # Check build access
            builds_response = await self.http_client.get(
                f"{base_url}/builds",
                headers=headers,
                params={"locator": "count:1"},
            )

            if builds_response.status_code == 200:
                permissions.append("read_builds")

            # Check agent access
            agents_response = await self.http_client.get(
                f"{base_url}/agents",
                headers=headers,
                params={"locator": "count:1"},
            )

            agent_count = 0
            if agents_response.status_code == 200:
                permissions.append("read_agents")
                agents_data = agents_response.json()
                agent_count = agents_data.get("count", 0)

            return ConnectionTestResult(
                success=True,
                message=f"Connected to TeamCity at {credentials.instance_url}",
                latency_ms=latency_ms,
                api_version=server_data.get("version", "unknown"),
                permissions=permissions,
                account_info={
                    "server_url": credentials.instance_url,
                    "version": server_data.get("version", ""),
                    "version_major": server_data.get("versionMajor", ""),
                    "build_number": server_data.get("buildNumber", ""),
                    "project_count": project_count,
                    "agent_count": agent_count,
                    **user_info,
                },
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="TIMEOUT",
            )
        except httpx.ConnectError:
            return ConnectionTestResult(
                success=False,
                message=f"Cannot connect to {credentials.instance_url}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="CONNECTION_ERROR",
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
        """Sync data from TeamCity.

        Syncs builds, projects, build types, and agents.

        Args:
            credentials: IntegrationCredentials for authentication.
            resources: List of resources to sync. Defaults to all.
            since: Only sync items modified after this time.
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced data and statistics.
        """
        started_at = datetime.utcnow()
        sync_id = f"teamcity-{started_at.isoformat()}"

        if not credentials.access_token or not credentials.instance_url:
            return SyncResult(
                success=False,
                message="Missing required credentials",
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=["access_token and instance_url are required"],
            )

        base_url = self._get_base_url(credentials.instance_url)
        headers = self._get_headers(credentials)

        resources_to_sync = resources or ["builds", "projects", "build_types", "agents"]
        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        # Sync projects
        if "projects" in resources_to_sync:
            try:
                projects = await self._get_projects(
                    base_url, headers, credentials.project_slug, limit
                )
                data["projects"] = projects
                total_synced += len(projects)
            except Exception as e:
                errors.append(f"Failed to fetch projects: {str(e)}")
                data["projects"] = []

        # Sync build types (build configurations)
        if "build_types" in resources_to_sync:
            try:
                build_types = await self._get_build_types(
                    base_url, headers, credentials.project_slug, limit
                )
                data["build_types"] = build_types
                total_synced += len(build_types)
            except Exception as e:
                warnings.append(f"Failed to fetch build types: {str(e)}")
                data["build_types"] = []

        # Sync builds
        if "builds" in resources_to_sync:
            try:
                builds = await self._get_builds(
                    base_url, headers, credentials.project_slug, since, limit
                )
                data["builds"] = builds
                total_synced += len(builds)
            except Exception as e:
                warnings.append(f"Failed to fetch builds: {str(e)}")
                data["builds"] = []

        # Sync agents
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
            message=f"Synced {total_synced} items from TeamCity",
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
        project_id: Optional[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch projects from TeamCity."""
        locator = f"count:{min(limit, 100)}"
        if project_id:
            locator = f"id:{project_id}"

        response = await self.http_client.get(
            f"{base_url}/projects",
            headers=headers,
            params={"locator": locator, "fields": "project(id,name,description,archived,parentProjectId,buildTypes(buildType(id,name)))"},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("project", [])

    async def _get_build_types(
        self,
        base_url: str,
        headers: dict[str, str],
        project_id: Optional[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch build types (build configurations) from TeamCity."""
        locator = f"count:{min(limit, 100)}"
        if project_id:
            locator = f"project:(id:{project_id}),count:{min(limit, 100)}"

        response = await self.http_client.get(
            f"{base_url}/buildTypes",
            headers=headers,
            params={
                "locator": locator,
                "fields": "buildType(id,name,description,projectId,projectName,paused,vcsRootEntries(vcsRootEntry(vcsRoot(name))))",
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("buildType", [])

    async def _get_builds(
        self,
        base_url: str,
        headers: dict[str, str],
        project_id: Optional[str],
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch builds from TeamCity."""
        locator_parts = [f"count:{min(limit, 100)}"]

        if project_id:
            locator_parts.append(f"project:(id:{project_id})")

        if since:
            # TeamCity uses format: yyyyMMddTHHmmss+0000
            since_str = since.strftime("%Y%m%dT%H%M%S") + "+0000"
            locator_parts.append(f"sinceDate:{since_str}")

        locator = ",".join(locator_parts)

        response = await self.http_client.get(
            f"{base_url}/builds",
            headers=headers,
            params={
                "locator": locator,
                "fields": "build(id,buildTypeId,number,status,state,branchName,startDate,finishDate,queuedDate,agent(name),triggered(type,user(username)))",
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("build", [])

    async def _get_agents(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch build agents from TeamCity."""
        response = await self.http_client.get(
            f"{base_url}/agents",
            headers=headers,
            params={
                "locator": f"count:{min(limit, 100)}",
                "fields": "agent(id,name,connected,enabled,authorized,uptodate,ip,pool(id,name),build(id,buildTypeId,status))",
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("agent", [])

    async def trigger_build(
        self,
        credentials: IntegrationCredentials,
        build_type_id: str,
        branch: Optional[str] = None,
        properties: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Trigger a new build in TeamCity.

        Args:
            credentials: IntegrationCredentials for authentication.
            build_type_id: The build configuration ID to trigger.
            branch: Optional branch name to build.
            properties: Optional build parameters.

        Returns:
            Build information dict.
        """
        base_url = self._get_base_url(credentials.instance_url)
        headers = self._get_headers(credentials)

        build_request: dict[str, Any] = {
            "buildType": {"id": build_type_id},
        }

        if branch:
            build_request["branchName"] = branch

        if properties:
            build_request["properties"] = {
                "property": [
                    {"name": k, "value": v} for k, v in properties.items()
                ]
            }

        response = await self.http_client.post(
            f"{base_url}/buildQueue",
            headers=headers,
            json=build_request,
        )
        response.raise_for_status()
        return response.json()

    async def get_build_details(
        self,
        credentials: IntegrationCredentials,
        build_id: int,
    ) -> dict[str, Any]:
        """Get detailed information about a specific build.

        Args:
            credentials: IntegrationCredentials for authentication.
            build_id: The build ID to retrieve.

        Returns:
            Detailed build information.
        """
        base_url = self._get_base_url(credentials.instance_url)
        headers = self._get_headers(credentials)

        response = await self.http_client.get(
            f"{base_url}/builds/id:{build_id}",
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def get_build_log(
        self,
        credentials: IntegrationCredentials,
        build_id: int,
    ) -> str:
        """Get the build log for a specific build.

        Args:
            credentials: IntegrationCredentials for authentication.
            build_id: The build ID to retrieve logs for.

        Returns:
            Build log as plain text.
        """
        base_url = self._get_base_url(credentials.instance_url)
        headers = self._get_headers(credentials)
        headers["Accept"] = "text/plain"

        response = await self.http_client.get(
            f"{base_url}/builds/id:{build_id}/log",
            headers=headers,
        )
        response.raise_for_status()
        return response.text
