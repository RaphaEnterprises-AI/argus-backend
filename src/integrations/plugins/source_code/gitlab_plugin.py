"""GitLab integration plugin.

Provides integration with GitLab (cloud and self-hosted) for projects,
merge requests, pipelines, and webhook events.
"""

import time
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote_plus

import httpx

from src.integrations.base import (
    AuthType,
    ConnectionTestResult,
    FieldDefinition,
    IntegrationCategory,
    IntegrationCredentials,
    IntegrationMetadata,
    IntegrationPlugin,
    OAuthConfig,
    SyncResult,
)


class GitLabPlugin(IntegrationPlugin):
    """GitLab integration using OAuth2 or Personal Access Token.

    Features:
    - Support for GitLab.com and self-hosted instances
    - OAuth2 flow for secure authentication
    - Personal Access Token (PAT) support
    - Project/repository management
    - Merge request tracking
    - CI/CD pipeline integration
    - Webhook support for MR, push, pipeline events

    API Documentation: https://docs.gitlab.com/ee/api/
    """

    DEFAULT_BASE_URL = "https://gitlab.com"

    # Rate limit tracking
    _rate_limit_remaining: int = 2000
    _rate_limit_reset: Optional[float] = None

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the GitLab instance URL."""
        if credentials.instance_url:
            # Remove trailing slash
            return credentials.instance_url.rstrip("/")
        return self.DEFAULT_BASE_URL

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug="gitlab",
            name="GitLab",
            description="Source code hosting, merge requests, CI/CD pipelines, and DevOps platform",
            category=IntegrationCategory.SOURCE_CODE,
            auth_type=AuthType.OAUTH2,
            icon_url="https://about.gitlab.com/images/press/logo/png/gitlab-icon-rgb.png",
            docs_url="https://docs.gitlab.com/ee/api/",
            website_url="https://gitlab.com",
            features=[
                "projects",
                "merge_requests",
                "pipelines",
                "issues",
                "webhooks",
                "ci_cd",
                "commits",
                "branches",
                "self_hosted",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Access Token",
                    type="password",
                    required=True,
                    placeholder="glpat-xxxxxxxxxxxxxxxxxxxx",
                    help_text="Personal access token or OAuth token",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="instance_url",
                    label="GitLab Instance URL",
                    type="url",
                    required=False,
                    placeholder="https://gitlab.example.com",
                    help_text="For self-hosted GitLab. Leave empty for GitLab.com",
                ),
                FieldDefinition(
                    name="org_slug",
                    label="Group/Namespace",
                    type="text",
                    required=False,
                    placeholder="my-group",
                    help_text="Optional: Filter to a specific group/namespace",
                ),
            ],
            oauth_config=OAuthConfig(
                authorize_url="https://gitlab.com/oauth/authorize",
                token_url="https://gitlab.com/oauth/token",
                scopes=["api", "read_api", "read_user", "read_repository"],
                client_id_env_var="GITLAB_CLIENT_ID",
                client_secret_env_var="GITLAB_CLIENT_SECRET",
                redirect_uri_path="/api/v1/integrations/oauth/callback/gitlab",
                pkce_required=False,
                state_required=True,
            ),
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["projects", "merge_requests", "pipelines"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for GitLab API requests."""
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Argus-E2E-Testing-Agent/1.0",
        }

    def _update_rate_limits(self, response: httpx.Response) -> None:
        """Update rate limit tracking from response headers."""
        if "RateLimit-Remaining" in response.headers:
            self._rate_limit_remaining = int(response.headers["RateLimit-Remaining"])
        if "RateLimit-Reset" in response.headers:
            self._rate_limit_reset = float(response.headers["RateLimit-Reset"])

    async def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit is exhausted."""
        if self._rate_limit_remaining <= 1 and self._rate_limit_reset:
            wait_time = self._rate_limit_reset - time.time()
            if wait_time > 0:
                wait_time = min(wait_time, 60)
                import asyncio
                await asyncio.sleep(wait_time)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        credentials: IntegrationCredentials,
        params: Optional[dict[str, Any]] = None,
        json_data: Optional[dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make an API request with rate limit handling."""
        await self._wait_for_rate_limit()

        base_url = self._get_base_url(credentials)
        url = f"{base_url}/api/v4{endpoint}"
        headers = self._get_headers(credentials)

        response = await self.http_client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_data,
        )

        self._update_rate_limits(response)
        return response

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test GitLab connection by fetching authenticated user info.

        Uses GET /user endpoint to verify token validity and permissions.
        """
        start_time = time.time()

        try:
            response = await self._make_request(
                method="GET",
                endpoint="/user",
                credentials=credentials,
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid or expired access token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden - insufficient permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            response.raise_for_status()
            user_data = response.json()

            # Fetch version info for API version
            api_version = "v4"
            try:
                version_response = await self._make_request(
                    method="GET",
                    endpoint="/version",
                    credentials=credentials,
                )
                if version_response.status_code == 200:
                    version_data = version_response.json()
                    api_version = f"v4 (GitLab {version_data.get('version', 'unknown')})"
            except Exception:
                pass  # Version endpoint may not be accessible

            return ConnectionTestResult(
                success=True,
                message=f"Connected as {user_data.get('username', 'unknown')}",
                latency_ms=latency_ms,
                api_version=api_version,
                permissions=[],  # GitLab doesn't expose token scopes easily
                account_info={
                    "id": user_data.get("id"),
                    "username": user_data.get("username"),
                    "name": user_data.get("name"),
                    "email": user_data.get("email"),
                    "avatar_url": user_data.get("avatar_url"),
                    "web_url": user_data.get("web_url"),
                    "state": user_data.get("state"),
                    "is_admin": user_data.get("is_admin", False),
                    "instance_url": self._get_base_url(credentials),
                    "rate_limit_remaining": self._rate_limit_remaining,
                },
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="TIMEOUT",
            )
        except httpx.ConnectError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Failed to connect to GitLab API: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="CONNECTION_ERROR",
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="UNKNOWN_ERROR",
                error_details={"exception": type(e).__name__},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from GitLab.

        Syncs projects, merge requests, and pipelines based on the requested resources.

        Args:
            credentials: GitLab credentials with access token
            resources: List of resources to sync (projects, merge_requests, pipelines)
            since: Only fetch data updated since this datetime
            limit: Maximum items per resource type

        Returns:
            SyncResult containing synced data
        """
        started_at = datetime.utcnow()
        sync_id = f"gitlab_{started_at.timestamp()}"

        if resources is None:
            resources = ["projects", "merge_requests", "pipelines"]

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_points = 0

        try:
            # Sync projects
            if "projects" in resources:
                projects_result = await self._sync_projects(
                    credentials, since, limit
                )
                data["projects"] = projects_result["data"]
                total_points += projects_result["count"]
                errors.extend(projects_result.get("errors", []))
                warnings.extend(projects_result.get("warnings", []))

            # Sync merge requests
            if "merge_requests" in resources:
                mrs_result = await self._sync_merge_requests(
                    credentials,
                    projects=data.get("projects", []),
                    since=since,
                    limit=limit,
                )
                data["merge_requests"] = mrs_result["data"]
                total_points += mrs_result["count"]
                errors.extend(mrs_result.get("errors", []))
                warnings.extend(mrs_result.get("warnings", []))

            # Sync pipelines
            if "pipelines" in resources:
                pipelines_result = await self._sync_pipelines(
                    credentials,
                    projects=data.get("projects", []),
                    since=since,
                    limit=limit,
                )
                data["pipelines"] = pipelines_result["data"]
                total_points += pipelines_result["count"]
                errors.extend(pipelines_result.get("errors", []))
                warnings.extend(pipelines_result.get("warnings", []))

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_points} data points from GitLab",
                data_points_synced=total_points,
                sync_id=sync_id,
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
                data_points_synced=total_points,
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                warnings=warnings,
                data=data,
            )

    async def _sync_projects(
        self,
        credentials: IntegrationCredentials,
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Fetch projects for the authenticated user."""
        projects: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []

        try:
            params: dict[str, Any] = {
                "per_page": min(limit, 100),
                "order_by": "updated_at",
                "sort": "desc",
                "membership": "true",
            }

            if since:
                params["updated_after"] = since.isoformat() + "Z"

            # Fetch from specific group if specified
            if credentials.org_slug:
                endpoint = f"/groups/{quote_plus(credentials.org_slug)}/projects"
            else:
                endpoint = "/projects"

            response = await self._make_request(
                method="GET",
                endpoint=endpoint,
                credentials=credentials,
                params=params,
            )

            if response.status_code != 200:
                errors.append(f"Failed to fetch projects: {response.status_code}")
            else:
                projects_data = response.json()
                for project in projects_data[:limit]:
                    projects.append({
                        "id": project["id"],
                        "name": project["name"],
                        "path_with_namespace": project["path_with_namespace"],
                        "description": project.get("description"),
                        "visibility": project.get("visibility"),
                        "web_url": project["web_url"],
                        "http_url_to_repo": project.get("http_url_to_repo"),
                        "ssh_url_to_repo": project.get("ssh_url_to_repo"),
                        "default_branch": project.get("default_branch", "main"),
                        "topics": project.get("topics", []),
                        "updated_at": project.get("last_activity_at"),
                        "created_at": project.get("created_at"),
                        "open_issues_count": project.get("open_issues_count", 0),
                        "forks_count": project.get("forks_count", 0),
                        "star_count": project.get("star_count", 0),
                    })

        except Exception as e:
            errors.append(f"Error syncing projects: {str(e)}")

        return {
            "data": projects,
            "count": len(projects),
            "errors": errors,
            "warnings": warnings,
        }

    async def _sync_merge_requests(
        self,
        credentials: IntegrationCredentials,
        projects: list[dict[str, Any]],
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Fetch recent merge requests from projects."""
        merge_requests: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []

        # Limit projects to check
        projects_to_check = projects[:10] if len(projects) > 10 else projects

        if len(projects) > 10:
            warnings.append(f"Limited MR sync to first 10 projects out of {len(projects)}")

        for project in projects_to_check:
            try:
                params: dict[str, Any] = {
                    "state": "all",
                    "per_page": min(limit // max(len(projects_to_check), 1) + 1, 30),
                    "order_by": "updated_at",
                    "sort": "desc",
                }

                if since:
                    params["updated_after"] = since.isoformat() + "Z"

                response = await self._make_request(
                    method="GET",
                    endpoint=f"/projects/{project['id']}/merge_requests",
                    credentials=credentials,
                    params=params,
                )

                if response.status_code != 200:
                    warnings.append(f"Failed to fetch MRs for {project['path_with_namespace']}")
                    continue

                mrs_data = response.json()
                for mr in mrs_data:
                    merge_requests.append({
                        "id": mr["id"],
                        "iid": mr["iid"],
                        "title": mr["title"],
                        "state": mr["state"],
                        "web_url": mr["web_url"],
                        "project_id": project["id"],
                        "project_path": project["path_with_namespace"],
                        "author": mr.get("author", {}).get("username"),
                        "created_at": mr.get("created_at"),
                        "updated_at": mr.get("updated_at"),
                        "merged_at": mr.get("merged_at"),
                        "source_branch": mr.get("source_branch"),
                        "target_branch": mr.get("target_branch"),
                        "draft": mr.get("draft", False),
                        "work_in_progress": mr.get("work_in_progress", False),
                        "labels": mr.get("labels", []),
                    })

                    if len(merge_requests) >= limit:
                        break

            except Exception as e:
                warnings.append(f"Error fetching MRs for {project['path_with_namespace']}: {str(e)}")

        return {
            "data": merge_requests[:limit],
            "count": len(merge_requests[:limit]),
            "errors": errors,
            "warnings": warnings,
        }

    async def _sync_pipelines(
        self,
        credentials: IntegrationCredentials,
        projects: list[dict[str, Any]],
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Fetch recent pipelines from projects."""
        pipelines: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []

        # Limit projects to check
        projects_to_check = projects[:10] if len(projects) > 10 else projects

        if len(projects) > 10:
            warnings.append(f"Limited pipeline sync to first 10 projects out of {len(projects)}")

        for project in projects_to_check:
            try:
                params: dict[str, Any] = {
                    "per_page": min(limit // max(len(projects_to_check), 1) + 1, 20),
                    "order_by": "updated_at",
                    "sort": "desc",
                }

                if since:
                    params["updated_after"] = since.isoformat() + "Z"

                response = await self._make_request(
                    method="GET",
                    endpoint=f"/projects/{project['id']}/pipelines",
                    credentials=credentials,
                    params=params,
                )

                if response.status_code != 200:
                    warnings.append(f"Failed to fetch pipelines for {project['path_with_namespace']}")
                    continue

                pipelines_data = response.json()
                for pipeline in pipelines_data:
                    pipelines.append({
                        "id": pipeline["id"],
                        "iid": pipeline.get("iid"),
                        "status": pipeline["status"],
                        "web_url": pipeline.get("web_url"),
                        "project_id": project["id"],
                        "project_path": project["path_with_namespace"],
                        "ref": pipeline.get("ref"),
                        "sha": pipeline.get("sha"),
                        "source": pipeline.get("source"),
                        "created_at": pipeline.get("created_at"),
                        "updated_at": pipeline.get("updated_at"),
                        "started_at": pipeline.get("started_at"),
                        "finished_at": pipeline.get("finished_at"),
                    })

                    if len(pipelines) >= limit:
                        break

            except Exception as e:
                warnings.append(f"Error fetching pipelines for {project['path_with_namespace']}: {str(e)}")

        return {
            "data": pipelines[:limit],
            "count": len(pipelines[:limit]),
            "errors": errors,
            "warnings": warnings,
        }

    async def get_project(
        self,
        credentials: IntegrationCredentials,
        project_id: int,
    ) -> Optional[dict[str, Any]]:
        """Fetch a specific project by ID."""
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/projects/{project_id}",
                credentials=credentials,
            )

            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    async def get_merge_request(
        self,
        credentials: IntegrationCredentials,
        project_id: int,
        mr_iid: int,
    ) -> Optional[dict[str, Any]]:
        """Fetch a specific merge request."""
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/projects/{project_id}/merge_requests/{mr_iid}",
                credentials=credentials,
            )

            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    async def create_webhook(
        self,
        credentials: IntegrationCredentials,
        project_id: int,
        webhook_url: str,
        events: dict[str, bool],
        secret: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Create a webhook for a project.

        Args:
            credentials: GitLab credentials
            project_id: Project ID
            webhook_url: URL to receive webhook events
            events: Dict of event types to enable (e.g., {"push_events": True})
            secret: Optional webhook secret for verification

        Returns:
            Webhook configuration if created successfully
        """
        try:
            payload: dict[str, Any] = {
                "url": webhook_url,
                "enable_ssl_verification": True,
                **events,
            }
            if secret:
                payload["token"] = secret

            response = await self._make_request(
                method="POST",
                endpoint=f"/projects/{project_id}/hooks",
                credentials=credentials,
                json_data=payload,
            )

            if response.status_code == 201:
                return response.json()
            return None
        except Exception:
            return None

    async def trigger_pipeline(
        self,
        credentials: IntegrationCredentials,
        project_id: int,
        ref: str,
        variables: Optional[dict[str, str]] = None,
    ) -> Optional[dict[str, Any]]:
        """Trigger a new pipeline for a project.

        Args:
            credentials: GitLab credentials
            project_id: Project ID
            ref: Branch or tag name
            variables: Optional CI/CD variables

        Returns:
            Pipeline data if triggered successfully
        """
        try:
            payload: dict[str, Any] = {"ref": ref}
            if variables:
                payload["variables"] = [
                    {"key": k, "value": v} for k, v in variables.items()
                ]

            response = await self._make_request(
                method="POST",
                endpoint=f"/projects/{project_id}/pipeline",
                credentials=credentials,
                json_data=payload,
            )

            if response.status_code == 201:
                return response.json()
            return None
        except Exception:
            return None
