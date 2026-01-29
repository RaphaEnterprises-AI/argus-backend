"""CircleCI integration plugin.

Provides integration with CircleCI for pipeline and workflow
management, job monitoring, and artifact access.
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


class CircleCIPlugin(IntegrationPlugin):
    """CircleCI integration plugin.

    Features:
    - Monitor pipelines and workflows
    - View job details and logs
    - Access artifacts
    - Trigger pipelines
    - Insights and metrics
    - Support for projects on GitHub and Bitbucket

    Authentication:
    - Personal API Token (generated in User Settings > Personal API Tokens)
    - Project tokens for specific project access

    API Reference:
    - https://circleci.com/docs/api/v2/
    """

    API_BASE = "https://circleci.com/api/v2"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return plugin metadata."""
        return IntegrationMetadata(
            slug="circleci",
            name="CircleCI",
            description="Cloud-native CI/CD platform for automated pipelines",
            category=IntegrationCategory.CICD,
            auth_type=AuthType.API_KEY,
            icon_url="https://circleci.com/favicon.ico",
            docs_url="https://circleci.com/docs/",
            website_url="https://circleci.com/",
            features=[
                "pipelines",
                "workflows",
                "jobs",
                "artifacts",
                "insights",
                "contexts",
                "triggers",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="Personal API Token",
                    type="password",
                    required=True,
                    placeholder="CCIPAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    help_text="Personal API token from CircleCI User Settings",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="org_slug",
                    label="Organization",
                    type="text",
                    required=False,
                    placeholder="gh/my-org or bb/my-org",
                    help_text="Organization slug (e.g., 'gh/my-org' for GitHub, 'bb/my-org' for Bitbucket)",
                ),
                FieldDefinition(
                    name="extra.project_slug",
                    label="Project Slug",
                    type="text",
                    required=False,
                    placeholder="gh/my-org/my-repo",
                    help_text="Full project slug (e.g., 'gh/my-org/my-repo')",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["pipelines", "workflows", "jobs"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for CircleCI API requests."""
        return {
            "Circle-Token": credentials.api_key or "",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the CircleCI connection.

        Validates the token by fetching the authenticated user info.

        Args:
            credentials: Integration credentials with api_key.

        Returns:
            ConnectionTestResult with success status and account info.
        """
        start_time = time.time()

        try:
            # Test authentication by getting user info
            response = await self.http_client.get(
                f"{self.API_BASE}/me",
                headers=self._get_headers(credentials),
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid or expired API token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden - check token permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            response.raise_for_status()
            user_data = response.json()

            # Determine permissions based on token capabilities
            permissions = ["read:user"]

            # Check if we can access projects
            if credentials.org_slug:
                org_id = credentials.org_slug.replace("/", "-")
                # Try to list some projects
                proj_resp = await self.http_client.get(
                    f"{self.API_BASE}/insights/{credentials.org_slug}/summary",
                    headers=self._get_headers(credentials),
                    params={"reporting-window": "last-7-days"},
                )
                if proj_resp.status_code == 200:
                    permissions.append("read:insights")

            return ConnectionTestResult(
                success=True,
                message=f"Connected as {user_data.get('login', user_data.get('name', 'unknown'))}",
                latency_ms=latency_ms,
                api_version="v2",
                permissions=permissions,
                account_info={
                    "id": user_data.get("id"),
                    "login": user_data.get("login"),
                    "name": user_data.get("name"),
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("CircleCI connection test failed", error=str(e))
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
        """Sync data from CircleCI.

        Fetches pipelines, workflows, and jobs.

        Args:
            credentials: Integration credentials.
            resources: Optional list of resources to sync (pipelines, workflows, jobs).
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
        headers = self._get_headers(credentials)

        # Get project slug
        project_slug = credentials.extra.get("project_slug")
        org_slug = credentials.org_slug

        if not project_slug and not org_slug:
            return SyncResult(
                success=False,
                message="Either project_slug or org_slug is required for syncing",
                errors=["Missing project_slug or org_slug in credentials"],
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        try:
            # If only org_slug is provided, list projects first
            projects_to_sync: list[str] = []
            if project_slug:
                projects_to_sync = [project_slug]
            elif org_slug:
                # For CircleCI, we need to know specific projects
                # Try to get insights summary which includes project names
                resp = await self.http_client.get(
                    f"{self.API_BASE}/insights/{org_slug}/summary",
                    headers=headers,
                    params={"reporting-window": "last-30-days"},
                )
                if resp.status_code == 200:
                    summary = resp.json()
                    for item in summary.get("org_data", {}).get("metrics", []):
                        if item.get("project_slug"):
                            projects_to_sync.append(item["project_slug"])
                    if not projects_to_sync:
                        # Fallback: try to get recent pipelines for the org
                        warnings.append("Could not determine projects from insights, trying pipeline list")

            # Sync pipelines
            if "pipelines" in resources_to_sync:
                data["pipelines"] = []

                for project in projects_to_sync[:5]:  # Limit to 5 projects
                    try:
                        params: dict[str, Any] = {}
                        if org_slug and not project:
                            params["org-slug"] = org_slug

                        url = f"{self.API_BASE}/project/{project}/pipeline" if project else f"{self.API_BASE}/pipeline"

                        resp = await self.http_client.get(
                            url,
                            headers=headers,
                            params=params,
                        )
                        if resp.status_code == 200:
                            pipelines_resp = resp.json()
                            pipelines = pipelines_resp.get("items", [])

                            for pipeline in pipelines[:limit]:
                                # Filter by since if provided
                                created_at = pipeline.get("created_at")
                                if since and created_at:
                                    try:
                                        pipeline_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                                        if pipeline_time.replace(tzinfo=None) < since:
                                            continue
                                    except Exception:
                                        pass

                                trigger = pipeline.get("trigger", {})
                                vcs = pipeline.get("vcs", {})

                                data["pipelines"].append({
                                    "id": pipeline.get("id"),
                                    "number": pipeline.get("number"),
                                    "project_slug": pipeline.get("project_slug"),
                                    "state": pipeline.get("state"),
                                    "created_at": created_at,
                                    "updated_at": pipeline.get("updated_at"),
                                    "trigger_type": trigger.get("type"),
                                    "trigger_actor": trigger.get("actor", {}).get("login"),
                                    "vcs_branch": vcs.get("branch"),
                                    "vcs_revision": vcs.get("revision"),
                                    "vcs_tag": vcs.get("tag"),
                                    "vcs_origin_repository_url": vcs.get("origin_repository_url"),
                                })
                                data_points += 1
                    except Exception as e:
                        warnings.append(f"Failed to sync pipelines for {project}: {str(e)}")

            # Sync workflows (requires pipeline IDs)
            if "workflows" in resources_to_sync:
                data["workflows"] = []
                pipeline_ids = [p["id"] for p in data.get("pipelines", [])[:20]]

                for pipeline_id in pipeline_ids:
                    try:
                        resp = await self.http_client.get(
                            f"{self.API_BASE}/pipeline/{pipeline_id}/workflow",
                            headers=headers,
                        )
                        if resp.status_code == 200:
                            workflows_resp = resp.json()
                            workflows = workflows_resp.get("items", [])

                            for workflow in workflows[:limit // max(len(pipeline_ids), 1)]:
                                data["workflows"].append({
                                    "id": workflow.get("id"),
                                    "name": workflow.get("name"),
                                    "pipeline_id": pipeline_id,
                                    "pipeline_number": workflow.get("pipeline_number"),
                                    "project_slug": workflow.get("project_slug"),
                                    "status": workflow.get("status"),
                                    "created_at": workflow.get("created_at"),
                                    "stopped_at": workflow.get("stopped_at"),
                                    "duration": self._calculate_duration(
                                        workflow.get("created_at"),
                                        workflow.get("stopped_at")
                                    ),
                                    "tag": workflow.get("tag"),
                                })
                                data_points += 1
                    except Exception as e:
                        warnings.append(f"Failed to sync workflows for pipeline {pipeline_id}: {str(e)}")

            # Sync jobs (requires workflow IDs)
            if "jobs" in resources_to_sync:
                data["jobs"] = []
                workflow_ids = [w["id"] for w in data.get("workflows", [])[:15]]

                for workflow_id in workflow_ids:
                    try:
                        resp = await self.http_client.get(
                            f"{self.API_BASE}/workflow/{workflow_id}/job",
                            headers=headers,
                        )
                        if resp.status_code == 200:
                            jobs_resp = resp.json()
                            jobs = jobs_resp.get("items", [])

                            for job in jobs[:limit // max(len(workflow_ids), 1)]:
                                data["jobs"].append({
                                    "id": job.get("id"),
                                    "name": job.get("name"),
                                    "job_number": job.get("job_number"),
                                    "workflow_id": workflow_id,
                                    "project_slug": job.get("project_slug"),
                                    "type": job.get("type"),
                                    "status": job.get("status"),
                                    "started_at": job.get("started_at"),
                                    "stopped_at": job.get("stopped_at"),
                                    "duration": self._calculate_duration(
                                        job.get("started_at"),
                                        job.get("stopped_at")
                                    ),
                                    "dependencies": job.get("dependencies", []),
                                })
                                data_points += 1
                    except Exception as e:
                        warnings.append(f"Failed to sync jobs for workflow {workflow_id}: {str(e)}")

            return SyncResult(
                success=True,
                message=f"Synced {data_points} items from CircleCI",
                data_points_synced=data_points,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            logger.error("CircleCI sync failed", error=str(e))
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

    def _calculate_duration(self, start: Optional[str], end: Optional[str]) -> Optional[float]:
        """Calculate duration in seconds between two ISO timestamps."""
        if not start or not end:
            return None
        try:
            start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
            return (end_time - start_time).total_seconds()
        except Exception:
            return None

    async def trigger_pipeline(
        self,
        credentials: IntegrationCredentials,
        project_slug: str,
        branch: Optional[str] = None,
        tag: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Trigger a new pipeline.

        Args:
            credentials: Integration credentials.
            project_slug: Full project slug (e.g., 'gh/my-org/my-repo').
            branch: Git branch to build (mutually exclusive with tag).
            tag: Git tag to build (mutually exclusive with branch).
            parameters: Optional pipeline parameters.

        Returns:
            Dict with pipeline info or error.
        """
        try:
            payload: dict[str, Any] = {}
            if branch:
                payload["branch"] = branch
            if tag:
                payload["tag"] = tag
            if parameters:
                payload["parameters"] = parameters

            response = await self.http_client.post(
                f"{self.API_BASE}/project/{project_slug}/pipeline",
                headers=self._get_headers(credentials),
                json=payload,
            )

            if response.status_code in (200, 201, 202):
                pipeline = response.json()
                return {
                    "success": True,
                    "message": f"Pipeline #{pipeline.get('number')} triggered",
                    "pipeline_id": pipeline.get("id"),
                    "pipeline_number": pipeline.get("number"),
                    "state": pipeline.get("state"),
                    "created_at": pipeline.get("created_at"),
                }
            else:
                error_data = {}
                try:
                    error_data = response.json()
                except Exception:
                    pass
                return {
                    "success": False,
                    "message": f"Failed to trigger pipeline: {response.status_code}",
                    "error": error_data.get("message", response.text),
                }

        except Exception as e:
            logger.error("Failed to trigger pipeline", error=str(e))
            return {
                "success": False,
                "message": f"Failed to trigger pipeline: {str(e)}",
            }

    async def get_workflow(
        self,
        credentials: IntegrationCredentials,
        workflow_id: str,
    ) -> Optional[dict[str, Any]]:
        """Get workflow details.

        Args:
            credentials: Integration credentials.
            workflow_id: Workflow ID.

        Returns:
            Workflow details or None on failure.
        """
        try:
            response = await self.http_client.get(
                f"{self.API_BASE}/workflow/{workflow_id}",
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error("Failed to get workflow", error=str(e))
            return None

    async def cancel_workflow(
        self,
        credentials: IntegrationCredentials,
        workflow_id: str,
    ) -> dict[str, Any]:
        """Cancel a running workflow.

        Args:
            credentials: Integration credentials.
            workflow_id: Workflow ID.

        Returns:
            Dict with cancellation result.
        """
        try:
            response = await self.http_client.post(
                f"{self.API_BASE}/workflow/{workflow_id}/cancel",
                headers=self._get_headers(credentials),
            )

            if response.status_code in (200, 202):
                return {
                    "success": True,
                    "message": f"Workflow {workflow_id} cancelled",
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to cancel workflow: {response.status_code}",
                    "error": response.text,
                }

        except Exception as e:
            logger.error("Failed to cancel workflow", error=str(e))
            return {
                "success": False,
                "message": f"Failed to cancel workflow: {str(e)}",
            }

    async def rerun_workflow(
        self,
        credentials: IntegrationCredentials,
        workflow_id: str,
        from_failed: bool = False,
        sparse_tree: bool = False,
    ) -> dict[str, Any]:
        """Rerun a workflow.

        Args:
            credentials: Integration credentials.
            workflow_id: Workflow ID.
            from_failed: Rerun only from failed jobs.
            sparse_tree: Rerun only failed and downstream jobs.

        Returns:
            Dict with rerun result.
        """
        try:
            payload: dict[str, Any] = {}
            if from_failed:
                payload["from_failed"] = True
            if sparse_tree:
                payload["sparse_tree"] = True

            response = await self.http_client.post(
                f"{self.API_BASE}/workflow/{workflow_id}/rerun",
                headers=self._get_headers(credentials),
                json=payload,
            )

            if response.status_code in (200, 201, 202):
                result = response.json()
                return {
                    "success": True,
                    "message": f"Workflow {workflow_id} rerun initiated",
                    "workflow_id": result.get("workflow_id"),
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to rerun workflow: {response.status_code}",
                    "error": response.text,
                }

        except Exception as e:
            logger.error("Failed to rerun workflow", error=str(e))
            return {
                "success": False,
                "message": f"Failed to rerun workflow: {str(e)}",
            }

    async def get_job_artifacts(
        self,
        credentials: IntegrationCredentials,
        project_slug: str,
        job_number: int,
    ) -> list[dict[str, Any]]:
        """List artifacts for a job.

        Args:
            credentials: Integration credentials.
            project_slug: Full project slug.
            job_number: Job number.

        Returns:
            List of artifact metadata.
        """
        try:
            response = await self.http_client.get(
                f"{self.API_BASE}/project/{project_slug}/{job_number}/artifacts",
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            result = response.json()
            return result.get("items", [])

        except Exception as e:
            logger.error("Failed to get job artifacts", error=str(e))
            return []

    async def download_artifact(
        self,
        credentials: IntegrationCredentials,
        artifact_url: str,
    ) -> Optional[bytes]:
        """Download an artifact by URL.

        Args:
            credentials: Integration credentials.
            artifact_url: Full artifact download URL.

        Returns:
            Artifact content as bytes, or None on failure.
        """
        try:
            response = await self.http_client.get(
                artifact_url,
                headers=self._get_headers(credentials),
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.content

        except Exception as e:
            logger.error("Failed to download artifact", error=str(e))
            return None

    async def get_project_insights(
        self,
        credentials: IntegrationCredentials,
        project_slug: str,
        workflow_name: Optional[str] = None,
        branch: Optional[str] = None,
        reporting_window: str = "last-30-days",
    ) -> dict[str, Any]:
        """Get project insights and metrics.

        Args:
            credentials: Integration credentials.
            project_slug: Full project slug.
            workflow_name: Optional workflow name to filter.
            branch: Optional branch to filter.
            reporting_window: Time window (last-7-days, last-30-days, last-60-days, last-90-days).

        Returns:
            Dict with project insights.
        """
        try:
            params: dict[str, Any] = {
                "reporting-window": reporting_window,
            }
            if branch:
                params["branch"] = branch

            endpoint = f"{self.API_BASE}/insights/{project_slug}/workflows"
            if workflow_name:
                endpoint = f"{endpoint}/{workflow_name}"

            response = await self.http_client.get(
                endpoint,
                headers=self._get_headers(credentials),
                params=params,
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error("Failed to get project insights", error=str(e))
            return {}
