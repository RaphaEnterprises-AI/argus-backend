"""GitHub Actions integration plugin.

Provides integration with GitHub Actions for CI/CD workflow management,
monitoring runs, and accessing artifacts.
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


class GitHubActionsPlugin(IntegrationPlugin):
    """GitHub Actions integration plugin.

    Features:
    - Monitor workflow runs and their status
    - List and download artifacts
    - Trigger workflow dispatches
    - View job logs and timing
    - Track workflow usage and billing

    Authentication:
    - Personal Access Token (PAT) with 'repo' and 'actions' scopes
    - Fine-grained PAT with Actions read/write permissions

    API Reference:
    - https://docs.github.com/en/rest/actions
    """

    API_BASE = "https://api.github.com"
    API_VERSION = "2022-11-28"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return plugin metadata."""
        return IntegrationMetadata(
            slug="github_actions",
            name="GitHub Actions",
            description="CI/CD workflows integrated with GitHub repositories",
            category=IntegrationCategory.CICD,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
            docs_url="https://docs.github.com/en/actions",
            website_url="https://github.com/features/actions",
            features=[
                "workflow_runs",
                "artifacts",
                "workflow_dispatch",
                "job_logs",
                "usage_metrics",
                "check_runs",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Personal Access Token",
                    type="password",
                    required=True,
                    placeholder="ghp_xxxxxxxxxxxxxxxxxxxx",
                    help_text="GitHub PAT with 'repo' and 'actions' scopes",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="org_slug",
                    label="Organization/Owner",
                    type="text",
                    required=False,
                    placeholder="my-organization",
                    help_text="GitHub organization or user name to scope operations",
                ),
                FieldDefinition(
                    name="extra.repository",
                    label="Repository",
                    type="text",
                    required=False,
                    placeholder="my-repo",
                    help_text="Specific repository to monitor (optional)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["workflow_runs", "workflows", "artifacts"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for GitHub API requests."""
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": self.API_VERSION,
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the GitHub Actions connection.

        Validates the token by fetching the authenticated user info.

        Args:
            credentials: Integration credentials with access_token.

        Returns:
            ConnectionTestResult with success status and account info.
        """
        start_time = time.time()

        try:
            # Test authentication by getting user info
            response = await self.http_client.get(
                f"{self.API_BASE}/user",
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
                    message="Access forbidden - check token permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            response.raise_for_status()
            user_data = response.json()

            # Check for Actions-specific permissions by listing workflows
            permissions = ["read:user"]
            if credentials.org_slug:
                # Try to access org workflows
                repo = credentials.extra.get("repository", "")
                if repo:
                    workflows_resp = await self.http_client.get(
                        f"{self.API_BASE}/repos/{credentials.org_slug}/{repo}/actions/workflows",
                        headers=self._get_headers(credentials),
                    )
                    if workflows_resp.status_code == 200:
                        permissions.append("actions:read")

            return ConnectionTestResult(
                success=True,
                message=f"Connected as {user_data.get('login', 'unknown')}",
                latency_ms=latency_ms,
                api_version=self.API_VERSION,
                permissions=permissions,
                account_info={
                    "login": user_data.get("login"),
                    "name": user_data.get("name"),
                    "email": user_data.get("email"),
                    "avatar_url": user_data.get("avatar_url"),
                    "html_url": user_data.get("html_url"),
                    "type": user_data.get("type"),
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("GitHub Actions connection test failed", error=str(e))
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
        """Sync data from GitHub Actions.

        Fetches workflow runs, workflows, and artifacts.

        Args:
            credentials: Integration credentials.
            resources: Optional list of resources to sync (workflow_runs, workflows, artifacts).
            since: Optional datetime to fetch data since.
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced workflow data.
        """
        started_at = datetime.utcnow()
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        data_points = 0

        resources_to_sync = resources or self.metadata.sync_resources

        owner = credentials.org_slug
        repo = credentials.extra.get("repository")

        if not owner:
            return SyncResult(
                success=False,
                message="Organization/owner is required for syncing",
                errors=["Missing org_slug in credentials"],
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        try:
            headers = self._get_headers(credentials)

            # If no specific repo, list repos first
            repos_to_sync = []
            if repo:
                repos_to_sync = [repo]
            else:
                # Get repositories for the org/user
                repos_resp = await self.http_client.get(
                    f"{self.API_BASE}/users/{owner}/repos",
                    headers=headers,
                    params={"per_page": 30, "sort": "updated"},
                )
                if repos_resp.status_code == 200:
                    repos_data = repos_resp.json()
                    repos_to_sync = [r["name"] for r in repos_data[:10]]  # Limit to 10 repos
                else:
                    warnings.append(f"Could not list repos: {repos_resp.status_code}")

            # Sync workflows
            if "workflows" in resources_to_sync:
                data["workflows"] = []
                for repo_name in repos_to_sync:
                    try:
                        resp = await self.http_client.get(
                            f"{self.API_BASE}/repos/{owner}/{repo_name}/actions/workflows",
                            headers=headers,
                        )
                        if resp.status_code == 200:
                            workflows = resp.json().get("workflows", [])
                            for wf in workflows[:limit]:
                                data["workflows"].append({
                                    "id": wf.get("id"),
                                    "name": wf.get("name"),
                                    "path": wf.get("path"),
                                    "state": wf.get("state"),
                                    "repository": f"{owner}/{repo_name}",
                                    "created_at": wf.get("created_at"),
                                    "updated_at": wf.get("updated_at"),
                                    "html_url": wf.get("html_url"),
                                })
                                data_points += 1
                    except Exception as e:
                        warnings.append(f"Failed to sync workflows for {repo_name}: {str(e)}")

            # Sync workflow runs
            if "workflow_runs" in resources_to_sync:
                data["workflow_runs"] = []
                for repo_name in repos_to_sync:
                    try:
                        params: dict[str, Any] = {"per_page": min(limit, 100)}
                        if since:
                            params["created"] = f">={since.strftime('%Y-%m-%dT%H:%M:%SZ')}"

                        resp = await self.http_client.get(
                            f"{self.API_BASE}/repos/{owner}/{repo_name}/actions/runs",
                            headers=headers,
                            params=params,
                        )
                        if resp.status_code == 200:
                            runs = resp.json().get("workflow_runs", [])
                            for run in runs[:limit]:
                                data["workflow_runs"].append({
                                    "id": run.get("id"),
                                    "name": run.get("name"),
                                    "workflow_id": run.get("workflow_id"),
                                    "repository": f"{owner}/{repo_name}",
                                    "head_branch": run.get("head_branch"),
                                    "head_sha": run.get("head_sha"),
                                    "status": run.get("status"),
                                    "conclusion": run.get("conclusion"),
                                    "event": run.get("event"),
                                    "run_number": run.get("run_number"),
                                    "run_attempt": run.get("run_attempt"),
                                    "created_at": run.get("created_at"),
                                    "updated_at": run.get("updated_at"),
                                    "run_started_at": run.get("run_started_at"),
                                    "html_url": run.get("html_url"),
                                    "actor": run.get("actor", {}).get("login"),
                                })
                                data_points += 1
                    except Exception as e:
                        warnings.append(f"Failed to sync runs for {repo_name}: {str(e)}")

            # Sync artifacts
            if "artifacts" in resources_to_sync:
                data["artifacts"] = []
                for repo_name in repos_to_sync:
                    try:
                        resp = await self.http_client.get(
                            f"{self.API_BASE}/repos/{owner}/{repo_name}/actions/artifacts",
                            headers=headers,
                            params={"per_page": min(limit, 100)},
                        )
                        if resp.status_code == 200:
                            artifacts = resp.json().get("artifacts", [])
                            for artifact in artifacts[:limit]:
                                data["artifacts"].append({
                                    "id": artifact.get("id"),
                                    "name": artifact.get("name"),
                                    "size_in_bytes": artifact.get("size_in_bytes"),
                                    "repository": f"{owner}/{repo_name}",
                                    "workflow_run_id": artifact.get("workflow_run", {}).get("id"),
                                    "expired": artifact.get("expired"),
                                    "expires_at": artifact.get("expires_at"),
                                    "created_at": artifact.get("created_at"),
                                    "archive_download_url": artifact.get("archive_download_url"),
                                })
                                data_points += 1
                    except Exception as e:
                        warnings.append(f"Failed to sync artifacts for {repo_name}: {str(e)}")

            return SyncResult(
                success=True,
                message=f"Synced {data_points} items from GitHub Actions",
                data_points_synced=data_points,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            logger.error("GitHub Actions sync failed", error=str(e))
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

    async def trigger_workflow(
        self,
        credentials: IntegrationCredentials,
        owner: str,
        repo: str,
        workflow_id: str | int,
        ref: str = "main",
        inputs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Trigger a workflow dispatch event.

        Args:
            credentials: Integration credentials.
            owner: Repository owner.
            repo: Repository name.
            workflow_id: Workflow ID or filename (e.g., "ci.yml").
            ref: Git reference (branch, tag, or SHA).
            inputs: Optional workflow inputs.

        Returns:
            Dict with trigger status.
        """
        try:
            payload: dict[str, Any] = {"ref": ref}
            if inputs:
                payload["inputs"] = inputs

            response = await self.http_client.post(
                f"{self.API_BASE}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches",
                headers=self._get_headers(credentials),
                json=payload,
            )

            if response.status_code == 204:
                return {
                    "success": True,
                    "message": f"Workflow {workflow_id} triggered on {ref}",
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to trigger workflow: {response.status_code}",
                    "error": response.text,
                }

        except Exception as e:
            logger.error("Failed to trigger workflow", error=str(e))
            return {
                "success": False,
                "message": f"Failed to trigger workflow: {str(e)}",
            }

    async def get_workflow_run_jobs(
        self,
        credentials: IntegrationCredentials,
        owner: str,
        repo: str,
        run_id: int,
    ) -> list[dict[str, Any]]:
        """Get jobs for a workflow run.

        Args:
            credentials: Integration credentials.
            owner: Repository owner.
            repo: Repository name.
            run_id: Workflow run ID.

        Returns:
            List of job details.
        """
        try:
            response = await self.http_client.get(
                f"{self.API_BASE}/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            jobs_data = response.json().get("jobs", [])

            return [
                {
                    "id": job.get("id"),
                    "name": job.get("name"),
                    "status": job.get("status"),
                    "conclusion": job.get("conclusion"),
                    "started_at": job.get("started_at"),
                    "completed_at": job.get("completed_at"),
                    "runner_name": job.get("runner_name"),
                    "runner_group_name": job.get("runner_group_name"),
                    "steps": [
                        {
                            "name": step.get("name"),
                            "status": step.get("status"),
                            "conclusion": step.get("conclusion"),
                            "number": step.get("number"),
                            "started_at": step.get("started_at"),
                            "completed_at": step.get("completed_at"),
                        }
                        for step in job.get("steps", [])
                    ],
                }
                for job in jobs_data
            ]

        except Exception as e:
            logger.error("Failed to get workflow run jobs", error=str(e))
            return []

    async def download_artifact(
        self,
        credentials: IntegrationCredentials,
        owner: str,
        repo: str,
        artifact_id: int,
    ) -> Optional[bytes]:
        """Download an artifact.

        Args:
            credentials: Integration credentials.
            owner: Repository owner.
            repo: Repository name.
            artifact_id: Artifact ID.

        Returns:
            Artifact content as bytes, or None on failure.
        """
        try:
            response = await self.http_client.get(
                f"{self.API_BASE}/repos/{owner}/{repo}/actions/artifacts/{artifact_id}/zip",
                headers=self._get_headers(credentials),
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.content

        except Exception as e:
            logger.error("Failed to download artifact", error=str(e))
            return None
