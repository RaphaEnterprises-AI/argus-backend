"""GitLab CI integration plugin.

Provides integration with GitLab CI/CD for pipeline management,
job monitoring, and artifact access. Supports both GitLab.com
and self-hosted GitLab instances.
"""

import time
from datetime import datetime
from typing import Any, Optional

import structlog

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

logger = structlog.get_logger()


class GitLabCIPlugin(IntegrationPlugin):
    """GitLab CI integration plugin.

    Features:
    - Monitor pipelines and their status
    - View and manage jobs
    - Access job artifacts and logs
    - Trigger pipelines manually
    - Track pipeline schedules
    - Support for self-hosted instances

    Authentication:
    - Personal Access Token with 'api' or 'read_api' scope
    - Project Access Token with appropriate permissions

    API Reference:
    - https://docs.gitlab.com/ee/api/pipelines.html
    - https://docs.gitlab.com/ee/api/jobs.html
    """

    DEFAULT_API_BASE = "https://gitlab.com/api/v4"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return plugin metadata."""
        return IntegrationMetadata(
            slug="gitlab_ci",
            name="GitLab CI",
            description="CI/CD pipelines integrated with GitLab repositories",
            category=IntegrationCategory.CICD,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://about.gitlab.com/images/press/press-kit-icon.svg",
            docs_url="https://docs.gitlab.com/ee/ci/",
            website_url="https://about.gitlab.com/stages-devops-lifecycle/continuous-integration/",
            features=[
                "pipelines",
                "jobs",
                "artifacts",
                "schedules",
                "triggers",
                "environments",
                "self_hosted",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Personal Access Token",
                    type="password",
                    required=True,
                    placeholder="glpat-xxxxxxxxxxxxxxxxxxxx",
                    help_text="GitLab PAT with 'api' or 'read_api' scope",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="instance_url",
                    label="GitLab Instance URL",
                    type="url",
                    required=False,
                    placeholder="https://gitlab.example.com",
                    help_text="For self-hosted GitLab instances. Leave empty for GitLab.com",
                ),
                FieldDefinition(
                    name="org_slug",
                    label="Group/Namespace",
                    type="text",
                    required=False,
                    placeholder="my-group",
                    help_text="GitLab group or namespace to scope operations",
                ),
                FieldDefinition(
                    name="extra.project_id",
                    label="Project ID or Path",
                    type="text",
                    required=False,
                    placeholder="my-group/my-project or 12345",
                    help_text="Specific project ID or URL-encoded path",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["pipelines", "jobs", "schedules"],
        )

    def _get_api_base(self, credentials: IntegrationCredentials) -> str:
        """Get the API base URL, supporting self-hosted instances."""
        if credentials.instance_url:
            base = credentials.instance_url.rstrip("/")
            return f"{base}/api/v4"
        return self.DEFAULT_API_BASE

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for GitLab API requests."""
        return {
            "PRIVATE-TOKEN": credentials.access_token or "",
            "Content-Type": "application/json",
        }

    def _encode_project_path(self, project_path: str) -> str:
        """URL-encode a project path for API calls."""
        from urllib.parse import quote
        return quote(project_path, safe="")

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the GitLab CI connection.

        Validates the token by fetching the authenticated user info.

        Args:
            credentials: Integration credentials with access_token.

        Returns:
            ConnectionTestResult with success status and account info.
        """
        start_time = time.time()
        api_base = self._get_api_base(credentials)

        try:
            # Test authentication by getting user info
            response = await self.http_client.get(
                f"{api_base}/user",
                headers=self._get_headers(credentials),
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid or expired access token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden - check token scopes",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            response.raise_for_status()
            user_data = response.json()

            # Determine permissions based on token capabilities
            permissions = ["read:user"]

            # Check if we can access pipelines
            project_id = credentials.extra.get("project_id")
            if project_id:
                encoded_id = self._encode_project_path(project_id) if "/" in str(project_id) else project_id
                pipelines_resp = await self.http_client.get(
                    f"{api_base}/projects/{encoded_id}/pipelines",
                    headers=self._get_headers(credentials),
                    params={"per_page": 1},
                )
                if pipelines_resp.status_code == 200:
                    permissions.append("read:pipelines")

            # Get GitLab version info
            version_resp = await self.http_client.get(
                f"{api_base}/version",
                headers=self._get_headers(credentials),
            )
            api_version = None
            if version_resp.status_code == 200:
                version_data = version_resp.json()
                api_version = version_data.get("version")

            return ConnectionTestResult(
                success=True,
                message=f"Connected as {user_data.get('username', 'unknown')}",
                latency_ms=latency_ms,
                api_version=api_version,
                permissions=permissions,
                account_info={
                    "id": user_data.get("id"),
                    "username": user_data.get("username"),
                    "name": user_data.get("name"),
                    "email": user_data.get("email"),
                    "avatar_url": user_data.get("avatar_url"),
                    "web_url": user_data.get("web_url"),
                    "state": user_data.get("state"),
                    "is_admin": user_data.get("is_admin", False),
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("GitLab CI connection test failed", error=str(e))
            return ConnectionTestResult(
                success=False,
                message=f"Connection failed: {str(e)}",
                latency_ms=latency_ms,
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
        """Sync data from GitLab CI.

        Fetches pipelines, jobs, and schedules.

        Args:
            credentials: Integration credentials.
            resources: Optional list of resources to sync (pipelines, jobs, schedules).
            since: Optional datetime to fetch data since.
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced CI/CD data.
        """
        started_at = datetime.utcnow()
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        data_points = 0

        resources_to_sync = resources or self.metadata.sync_resources
        api_base = self._get_api_base(credentials)
        headers = self._get_headers(credentials)

        try:
            # Get projects to sync
            projects_to_sync: list[dict[str, Any]] = []
            project_id = credentials.extra.get("project_id")

            if project_id:
                # Single project specified
                encoded_id = self._encode_project_path(project_id) if "/" in str(project_id) else project_id
                resp = await self.http_client.get(
                    f"{api_base}/projects/{encoded_id}",
                    headers=headers,
                )
                if resp.status_code == 200:
                    project_data = resp.json()
                    projects_to_sync.append({
                        "id": project_data.get("id"),
                        "path_with_namespace": project_data.get("path_with_namespace"),
                    })
                else:
                    errors.append(f"Could not access project {project_id}: {resp.status_code}")
            elif credentials.org_slug:
                # Get projects from group
                resp = await self.http_client.get(
                    f"{api_base}/groups/{self._encode_project_path(credentials.org_slug)}/projects",
                    headers=headers,
                    params={"per_page": 20, "order_by": "last_activity_at"},
                )
                if resp.status_code == 200:
                    projects = resp.json()
                    for proj in projects[:10]:
                        projects_to_sync.append({
                            "id": proj.get("id"),
                            "path_with_namespace": proj.get("path_with_namespace"),
                        })
                else:
                    warnings.append(f"Could not list group projects: {resp.status_code}")
            else:
                # Get user's projects
                resp = await self.http_client.get(
                    f"{api_base}/projects",
                    headers=headers,
                    params={"membership": True, "per_page": 20, "order_by": "last_activity_at"},
                )
                if resp.status_code == 200:
                    projects = resp.json()
                    for proj in projects[:10]:
                        projects_to_sync.append({
                            "id": proj.get("id"),
                            "path_with_namespace": proj.get("path_with_namespace"),
                        })
                else:
                    warnings.append(f"Could not list projects: {resp.status_code}")

            if not projects_to_sync:
                return SyncResult(
                    success=False,
                    message="No projects found to sync",
                    errors=errors,
                    warnings=warnings,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )

            # Sync pipelines
            if "pipelines" in resources_to_sync:
                data["pipelines"] = []
                for project in projects_to_sync:
                    try:
                        params: dict[str, Any] = {"per_page": min(limit, 100)}
                        if since:
                            params["updated_after"] = since.strftime("%Y-%m-%dT%H:%M:%SZ")

                        resp = await self.http_client.get(
                            f"{api_base}/projects/{project['id']}/pipelines",
                            headers=headers,
                            params=params,
                        )
                        if resp.status_code == 200:
                            pipelines = resp.json()
                            for pipeline in pipelines[:limit]:
                                data["pipelines"].append({
                                    "id": pipeline.get("id"),
                                    "iid": pipeline.get("iid"),
                                    "project_id": project["id"],
                                    "project_path": project["path_with_namespace"],
                                    "sha": pipeline.get("sha"),
                                    "ref": pipeline.get("ref"),
                                    "status": pipeline.get("status"),
                                    "source": pipeline.get("source"),
                                    "created_at": pipeline.get("created_at"),
                                    "updated_at": pipeline.get("updated_at"),
                                    "web_url": pipeline.get("web_url"),
                                })
                                data_points += 1
                    except Exception as e:
                        warnings.append(f"Failed to sync pipelines for {project['path_with_namespace']}: {str(e)}")

            # Sync jobs
            if "jobs" in resources_to_sync:
                data["jobs"] = []
                for project in projects_to_sync:
                    try:
                        params = {"per_page": min(limit, 100)}

                        resp = await self.http_client.get(
                            f"{api_base}/projects/{project['id']}/jobs",
                            headers=headers,
                            params=params,
                        )
                        if resp.status_code == 200:
                            jobs = resp.json()
                            for job in jobs[:limit]:
                                # Calculate duration if available
                                duration = None
                                if job.get("started_at") and job.get("finished_at"):
                                    try:
                                        start = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
                                        end = datetime.fromisoformat(job["finished_at"].replace("Z", "+00:00"))
                                        duration = (end - start).total_seconds()
                                    except Exception:
                                        pass

                                data["jobs"].append({
                                    "id": job.get("id"),
                                    "name": job.get("name"),
                                    "stage": job.get("stage"),
                                    "status": job.get("status"),
                                    "project_id": project["id"],
                                    "project_path": project["path_with_namespace"],
                                    "pipeline_id": job.get("pipeline", {}).get("id"),
                                    "ref": job.get("ref"),
                                    "tag": job.get("tag"),
                                    "created_at": job.get("created_at"),
                                    "started_at": job.get("started_at"),
                                    "finished_at": job.get("finished_at"),
                                    "duration_seconds": duration or job.get("duration"),
                                    "queued_duration": job.get("queued_duration"),
                                    "allow_failure": job.get("allow_failure"),
                                    "web_url": job.get("web_url"),
                                    "runner": job.get("runner", {}).get("description") if job.get("runner") else None,
                                    "artifacts": [
                                        {
                                            "filename": a.get("filename"),
                                            "size": a.get("size"),
                                            "file_type": a.get("file_type"),
                                        }
                                        for a in job.get("artifacts", [])
                                    ],
                                })
                                data_points += 1
                    except Exception as e:
                        warnings.append(f"Failed to sync jobs for {project['path_with_namespace']}: {str(e)}")

            # Sync schedules
            if "schedules" in resources_to_sync:
                data["schedules"] = []
                for project in projects_to_sync:
                    try:
                        resp = await self.http_client.get(
                            f"{api_base}/projects/{project['id']}/pipeline_schedules",
                            headers=headers,
                        )
                        if resp.status_code == 200:
                            schedules = resp.json()
                            for schedule in schedules[:limit]:
                                data["schedules"].append({
                                    "id": schedule.get("id"),
                                    "description": schedule.get("description"),
                                    "ref": schedule.get("ref"),
                                    "cron": schedule.get("cron"),
                                    "cron_timezone": schedule.get("cron_timezone"),
                                    "active": schedule.get("active"),
                                    "project_id": project["id"],
                                    "project_path": project["path_with_namespace"],
                                    "next_run_at": schedule.get("next_run_at"),
                                    "created_at": schedule.get("created_at"),
                                    "updated_at": schedule.get("updated_at"),
                                    "owner": schedule.get("owner", {}).get("username") if schedule.get("owner") else None,
                                })
                                data_points += 1
                    except Exception as e:
                        warnings.append(f"Failed to sync schedules for {project['path_with_namespace']}: {str(e)}")

            return SyncResult(
                success=True,
                message=f"Synced {data_points} items from GitLab CI",
                data_points_synced=data_points,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            logger.error("GitLab CI sync failed", error=str(e))
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

    async def trigger_pipeline(
        self,
        credentials: IntegrationCredentials,
        project_id: str | int,
        ref: str = "main",
        variables: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """Trigger a new pipeline.

        Args:
            credentials: Integration credentials.
            project_id: Project ID or URL-encoded path.
            ref: Git reference (branch, tag, or SHA).
            variables: Optional pipeline variables as list of {"key": "", "value": ""}.

        Returns:
            Dict with pipeline info or error.
        """
        api_base = self._get_api_base(credentials)

        try:
            encoded_id = self._encode_project_path(str(project_id)) if "/" in str(project_id) else project_id
            payload: dict[str, Any] = {"ref": ref}
            if variables:
                payload["variables"] = variables

            response = await self.http_client.post(
                f"{api_base}/projects/{encoded_id}/pipeline",
                headers=self._get_headers(credentials),
                json=payload,
            )

            if response.status_code == 201:
                pipeline = response.json()
                return {
                    "success": True,
                    "message": f"Pipeline #{pipeline.get('iid')} triggered on {ref}",
                    "pipeline_id": pipeline.get("id"),
                    "pipeline_iid": pipeline.get("iid"),
                    "web_url": pipeline.get("web_url"),
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to trigger pipeline: {response.status_code}",
                    "error": response.text,
                }

        except Exception as e:
            logger.error("Failed to trigger pipeline", error=str(e))
            return {
                "success": False,
                "message": f"Failed to trigger pipeline: {str(e)}",
            }

    async def get_job_log(
        self,
        credentials: IntegrationCredentials,
        project_id: str | int,
        job_id: int,
    ) -> Optional[str]:
        """Get the log output for a job.

        Args:
            credentials: Integration credentials.
            project_id: Project ID or URL-encoded path.
            job_id: Job ID.

        Returns:
            Job log as string, or None on failure.
        """
        api_base = self._get_api_base(credentials)

        try:
            encoded_id = self._encode_project_path(str(project_id)) if "/" in str(project_id) else project_id
            response = await self.http_client.get(
                f"{api_base}/projects/{encoded_id}/jobs/{job_id}/trace",
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            return response.text

        except Exception as e:
            logger.error("Failed to get job log", error=str(e))
            return None

    async def retry_job(
        self,
        credentials: IntegrationCredentials,
        project_id: str | int,
        job_id: int,
    ) -> dict[str, Any]:
        """Retry a failed job.

        Args:
            credentials: Integration credentials.
            project_id: Project ID or URL-encoded path.
            job_id: Job ID.

        Returns:
            Dict with retry result.
        """
        api_base = self._get_api_base(credentials)

        try:
            encoded_id = self._encode_project_path(str(project_id)) if "/" in str(project_id) else project_id
            response = await self.http_client.post(
                f"{api_base}/projects/{encoded_id}/jobs/{job_id}/retry",
                headers=self._get_headers(credentials),
            )

            if response.status_code == 201:
                job = response.json()
                return {
                    "success": True,
                    "message": f"Job {job.get('name')} retried",
                    "job_id": job.get("id"),
                    "web_url": job.get("web_url"),
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to retry job: {response.status_code}",
                    "error": response.text,
                }

        except Exception as e:
            logger.error("Failed to retry job", error=str(e))
            return {
                "success": False,
                "message": f"Failed to retry job: {str(e)}",
            }

    async def download_artifact(
        self,
        credentials: IntegrationCredentials,
        project_id: str | int,
        job_id: int,
        artifact_path: Optional[str] = None,
    ) -> Optional[bytes]:
        """Download job artifacts.

        Args:
            credentials: Integration credentials.
            project_id: Project ID or URL-encoded path.
            job_id: Job ID.
            artifact_path: Optional specific artifact path within the archive.

        Returns:
            Artifact content as bytes, or None on failure.
        """
        api_base = self._get_api_base(credentials)

        try:
            encoded_id = self._encode_project_path(str(project_id)) if "/" in str(project_id) else project_id

            if artifact_path:
                # Download specific file
                url = f"{api_base}/projects/{encoded_id}/jobs/{job_id}/artifacts/{artifact_path}"
            else:
                # Download entire artifacts archive
                url = f"{api_base}/projects/{encoded_id}/jobs/{job_id}/artifacts"

            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.content

        except Exception as e:
            logger.error("Failed to download artifact", error=str(e))
            return None
