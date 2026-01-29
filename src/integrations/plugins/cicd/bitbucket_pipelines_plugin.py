"""Bitbucket Pipelines integration plugin.

This module provides integration with Bitbucket Pipelines for CI/CD management.
Supports pipeline runs, steps, and deployment tracking.
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


class BitbucketPipelinesPlugin(IntegrationPlugin):
    """Bitbucket Pipelines integration plugin.

    Provides integration with Bitbucket Cloud Pipelines for:
    - Pipeline run management and monitoring
    - Step-level status tracking
    - Deployment environment tracking
    - Repository pipeline configuration

    Authentication:
        Uses App Password with Basic Auth.
        App Password should have appropriate permissions for Pipelines.

    API Reference:
        https://developer.atlassian.com/cloud/bitbucket/rest/api-group-pipelines/
    """

    BASE_URL = "https://api.bitbucket.org/2.0"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Bitbucket Pipelines integration metadata."""
        return IntegrationMetadata(
            slug="bitbucket-pipelines",
            name="Bitbucket Pipelines",
            description="Bitbucket Cloud Pipelines for CI/CD automation",
            category=IntegrationCategory.CICD,
            auth_type=AuthType.APP_PASSWORD,
            icon_url="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/bitbucket/bitbucket-original.svg",
            docs_url="https://support.atlassian.com/bitbucket-cloud/docs/get-started-with-bitbucket-pipelines/",
            website_url="https://bitbucket.org/product/features/pipelines",
            features=[
                "pipeline_runs",
                "step_monitoring",
                "deployment_tracking",
                "branch_pipelines",
                "pull_request_pipelines",
                "custom_pipelines",
                "scheduled_pipelines",
                "artifact_management",
            ],
            required_fields=[
                FieldDefinition(
                    name="username",
                    label="Username",
                    type="text",
                    required=True,
                    placeholder="your-bitbucket-username",
                    help_text="Your Bitbucket username (not email)",
                ),
                FieldDefinition(
                    name="password",
                    label="App Password",
                    type="password",
                    required=True,
                    placeholder="Enter your Bitbucket App Password",
                    help_text="Create at https://bitbucket.org/account/settings/app-passwords/",
                ),
                FieldDefinition(
                    name="workspace_id",
                    label="Workspace",
                    type="text",
                    required=True,
                    placeholder="my-workspace",
                    help_text="Your Bitbucket workspace slug",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="project_slug",
                    label="Repository Slug",
                    type="text",
                    required=False,
                    placeholder="my-repo",
                    help_text="Limit to a specific repository (optional)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["pipelines", "steps", "deployments", "variables"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for API requests using Basic Auth."""
        import base64

        auth_string = f"{credentials.username}:{credentials.password}"
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
        """Test the Bitbucket connection.

        Uses the /user endpoint to verify authentication
        and retrieve account information.

        Args:
            credentials: IntegrationCredentials containing username, password (app password),
                        and workspace_id.

        Returns:
            ConnectionTestResult with connection status and account details.
        """
        if not credentials.username:
            return ConnectionTestResult(
                success=False,
                message="Username is required",
                error_code="MISSING_USERNAME",
            )

        if not credentials.password:
            return ConnectionTestResult(
                success=False,
                message="App Password is required",
                error_code="MISSING_PASSWORD",
            )

        if not credentials.workspace_id:
            return ConnectionTestResult(
                success=False,
                message="Workspace ID is required",
                error_code="MISSING_WORKSPACE",
            )

        headers = self._get_headers(credentials)
        start_time = time.time()

        try:
            # Test connection using /user endpoint
            response = await self.http_client.get(
                f"{self.BASE_URL}/user",
                headers=headers,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid username or App Password",
                    latency_ms=latency_ms,
                    error_code="INVALID_CREDENTIALS",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Insufficient permissions. Check App Password scopes.",
                    latency_ms=latency_ms,
                    error_code="INSUFFICIENT_PERMISSIONS",
                )

            response.raise_for_status()
            user_data = response.json()

            # Verify workspace access
            workspace_response = await self.http_client.get(
                f"{self.BASE_URL}/workspaces/{credentials.workspace_id}",
                headers=headers,
            )

            if workspace_response.status_code == 404:
                return ConnectionTestResult(
                    success=False,
                    message=f"Workspace '{credentials.workspace_id}' not found",
                    latency_ms=latency_ms,
                    error_code="WORKSPACE_NOT_FOUND",
                )

            if workspace_response.status_code != 200:
                return ConnectionTestResult(
                    success=False,
                    message=f"Cannot access workspace '{credentials.workspace_id}'",
                    latency_ms=latency_ms,
                    error_code="WORKSPACE_ACCESS_DENIED",
                )

            workspace_data = workspace_response.json()

            # Check pipelines access
            permissions = ["authenticated", "read_user"]

            # Try to list repositories to check repo access
            repos_response = await self.http_client.get(
                f"{self.BASE_URL}/repositories/{credentials.workspace_id}",
                headers=headers,
                params={"pagelen": 1},
            )
            if repos_response.status_code == 200:
                permissions.append("read_repositories")
                repos_data = repos_response.json()
                repo_count = repos_data.get("size", 0)
            else:
                repo_count = 0

            return ConnectionTestResult(
                success=True,
                message=f"Connected to Bitbucket workspace '{credentials.workspace_id}'",
                latency_ms=latency_ms,
                api_version="2.0",
                permissions=permissions,
                account_info={
                    "username": user_data.get("username", ""),
                    "display_name": user_data.get("display_name", ""),
                    "account_id": user_data.get("account_id", ""),
                    "workspace": workspace_data.get("name", ""),
                    "workspace_slug": credentials.workspace_id,
                    "repository_count": repo_count,
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
        """Sync data from Bitbucket Pipelines.

        Syncs pipelines, steps, deployments, and variables.

        Args:
            credentials: IntegrationCredentials for authentication.
            resources: List of resources to sync. Defaults to all.
            since: Only sync items modified after this time.
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced data and statistics.
        """
        started_at = datetime.utcnow()
        sync_id = f"bitbucket-pipelines-{started_at.isoformat()}"

        if not credentials.username or not credentials.password or not credentials.workspace_id:
            return SyncResult(
                success=False,
                message="Missing required credentials",
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=["username, password, and workspace_id are required"],
            )

        headers = self._get_headers(credentials)
        workspace = credentials.workspace_id

        resources_to_sync = resources or ["pipelines", "steps", "deployments", "variables"]
        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        # Get repository list first
        try:
            repositories = await self._get_repositories(headers, workspace, limit)
        except Exception as e:
            return SyncResult(
                success=False,
                message=f"Failed to fetch repositories: {str(e)}",
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

        # Filter to specific repo if provided
        if credentials.project_slug:
            repositories = [r for r in repositories if r.get("slug") == credentials.project_slug]
            if not repositories:
                return SyncResult(
                    success=False,
                    message=f"Repository '{credentials.project_slug}' not found",
                    sync_id=sync_id,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    errors=[f"Repository '{credentials.project_slug}' not found in workspace"],
                )

        data["repositories"] = repositories

        # Sync pipelines for each repository
        if "pipelines" in resources_to_sync:
            all_pipelines = []
            for repo in repositories[:10]:  # Limit to first 10 repos
                repo_slug = repo.get("slug", "")
                try:
                    pipelines = await self._get_pipelines(
                        headers, workspace, repo_slug, since, limit
                    )
                    for pipeline in pipelines:
                        pipeline["repository"] = repo_slug
                    all_pipelines.extend(pipelines)
                except Exception as e:
                    warnings.append(f"Failed to fetch pipelines for {repo_slug}: {str(e)}")

            data["pipelines"] = all_pipelines
            total_synced += len(all_pipelines)

            # Sync steps for recent pipelines
            if "steps" in resources_to_sync:
                all_steps = []
                for pipeline in all_pipelines[:20]:  # Limit to first 20 pipelines
                    repo_slug = pipeline.get("repository", "")
                    pipeline_uuid = pipeline.get("uuid", "")
                    if repo_slug and pipeline_uuid:
                        try:
                            steps = await self._get_pipeline_steps(
                                headers, workspace, repo_slug, pipeline_uuid
                            )
                            for step in steps:
                                step["pipeline_uuid"] = pipeline_uuid
                                step["repository"] = repo_slug
                            all_steps.extend(steps)
                        except Exception as e:
                            warnings.append(
                                f"Failed to fetch steps for pipeline {pipeline_uuid}: {str(e)}"
                            )

                data["steps"] = all_steps
                total_synced += len(all_steps)

        # Sync deployments
        if "deployments" in resources_to_sync:
            all_deployments = []
            for repo in repositories[:10]:
                repo_slug = repo.get("slug", "")
                try:
                    deployments = await self._get_deployments(
                        headers, workspace, repo_slug, limit
                    )
                    for deployment in deployments:
                        deployment["repository"] = repo_slug
                    all_deployments.extend(deployments)
                except Exception as e:
                    warnings.append(f"Failed to fetch deployments for {repo_slug}: {str(e)}")

            data["deployments"] = all_deployments
            total_synced += len(all_deployments)

        # Sync pipeline variables
        if "variables" in resources_to_sync:
            all_variables = []
            for repo in repositories[:10]:
                repo_slug = repo.get("slug", "")
                try:
                    variables = await self._get_pipeline_variables(
                        headers, workspace, repo_slug, limit
                    )
                    for var in variables:
                        var["repository"] = repo_slug
                    all_variables.extend(variables)
                except Exception as e:
                    warnings.append(f"Failed to fetch variables for {repo_slug}: {str(e)}")

            data["variables"] = all_variables
            total_synced += len(all_variables)

        return SyncResult(
            success=len(errors) == 0,
            message=f"Synced {total_synced} items from Bitbucket Pipelines",
            data_points_synced=total_synced,
            sync_id=sync_id,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            errors=errors,
            warnings=warnings,
            data=data,
        )

    async def _get_repositories(
        self,
        headers: dict[str, str],
        workspace: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch repositories from the workspace."""
        response = await self.http_client.get(
            f"{self.BASE_URL}/repositories/{workspace}",
            headers=headers,
            params={"pagelen": min(limit, 100)},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("values", [])

    async def _get_pipelines(
        self,
        headers: dict[str, str],
        workspace: str,
        repo_slug: str,
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch pipeline runs for a repository."""
        params: dict[str, Any] = {
            "pagelen": min(limit, 100),
            "sort": "-created_on",  # Most recent first
        }

        response = await self.http_client.get(
            f"{self.BASE_URL}/repositories/{workspace}/{repo_slug}/pipelines",
            headers=headers,
            params=params,
        )

        # Pipelines might not be enabled for all repos
        if response.status_code == 404:
            return []

        response.raise_for_status()
        data = response.json()
        pipelines = data.get("values", [])

        # Filter by date if since is provided
        if since:
            filtered = []
            for pipeline in pipelines:
                created_on = pipeline.get("created_on", "")
                if created_on:
                    try:
                        created_dt = datetime.fromisoformat(created_on.replace("Z", "+00:00"))
                        if created_dt.replace(tzinfo=None) >= since:
                            filtered.append(pipeline)
                    except ValueError:
                        filtered.append(pipeline)
            pipelines = filtered

        return pipelines

    async def _get_pipeline_steps(
        self,
        headers: dict[str, str],
        workspace: str,
        repo_slug: str,
        pipeline_uuid: str,
    ) -> list[dict[str, Any]]:
        """Fetch steps for a pipeline run."""
        # Remove curly braces from UUID if present
        clean_uuid = pipeline_uuid.strip("{}")

        response = await self.http_client.get(
            f"{self.BASE_URL}/repositories/{workspace}/{repo_slug}/pipelines/{{{clean_uuid}}}/steps",
            headers=headers,
            params={"pagelen": 100},
        )

        if response.status_code == 404:
            return []

        response.raise_for_status()
        data = response.json()
        return data.get("values", [])

    async def _get_deployments(
        self,
        headers: dict[str, str],
        workspace: str,
        repo_slug: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch deployment environments for a repository."""
        response = await self.http_client.get(
            f"{self.BASE_URL}/repositories/{workspace}/{repo_slug}/environments",
            headers=headers,
            params={"pagelen": min(limit, 100)},
        )

        if response.status_code == 404:
            return []

        response.raise_for_status()
        data = response.json()
        return data.get("values", [])

    async def _get_pipeline_variables(
        self,
        headers: dict[str, str],
        workspace: str,
        repo_slug: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch pipeline variables for a repository."""
        response = await self.http_client.get(
            f"{self.BASE_URL}/repositories/{workspace}/{repo_slug}/pipelines_config/variables",
            headers=headers,
            params={"pagelen": min(limit, 100)},
        )

        if response.status_code == 404:
            return []

        response.raise_for_status()
        data = response.json()

        # Mask secure variable values
        variables = data.get("values", [])
        for var in variables:
            if var.get("secured", False):
                var["value"] = "***SECURED***"

        return variables
