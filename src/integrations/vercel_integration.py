"""
Vercel Integration for deployment tracking.

Tracks deployments to trigger tests and correlate errors with releases.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import httpx
import structlog

logger = structlog.get_logger()


class DeploymentState(str, Enum):
    """Vercel deployment states."""
    BUILDING = "BUILDING"
    ERROR = "ERROR"
    INITIALIZING = "INITIALIZING"
    QUEUED = "QUEUED"
    READY = "READY"
    CANCELED = "CANCELED"


@dataclass
class VercelDeployment:
    """A Vercel deployment."""
    deployment_id: str
    name: str
    url: str
    state: DeploymentState

    # Git info
    git_source: str | None  # github, gitlab, bitbucket
    git_repo: str | None
    git_branch: str | None
    git_commit_sha: str | None
    git_commit_message: str | None

    # Environment
    target: str | None  # production, preview

    # Timestamps
    created_at: datetime
    ready_at: datetime | None

    # URLs
    inspect_url: str


@dataclass
class VercelProject:
    """A Vercel project."""
    project_id: str
    name: str
    framework: str | None
    created_at: datetime
    updated_at: datetime

    # Git integration
    git_provider: str | None  # github, gitlab, bitbucket
    git_repo: str | None

    # Production info
    production_url: str | None
    latest_deployments: list[str]  # deployment IDs


class VercelIntegration:
    """
    Vercel Integration.

    API Docs: https://vercel.com/docs/rest-api

    Features:
    - List recent deployments
    - Get deployment details
    - Track preview deployments for PRs
    - Webhook support for deployment events

    Usage:
        vercel = VercelIntegration(api_token="xxx", team_id="team_xxx")

        # Get recent deployments
        deployments = await vercel.get_deployments(project_id="prj_xxx")

        # Get preview URL for a branch
        preview_url = await vercel.get_preview_url_for_branch(
            project_id="prj_xxx",
            branch="feature/login",
        )

        # Wait for deployment to be ready
        deployment = await vercel.wait_for_deployment(
            deployment_id="dpl_xxx",
            timeout_seconds=300,
        )
    """

    def __init__(
        self,
        api_token: str | None = None,
        team_id: str | None = None,
    ):
        self.api_token = api_token or os.environ.get("VERCEL_API_TOKEN")
        self.team_id = team_id or os.environ.get("VERCEL_TEAM_ID")
        self.base_url = "https://api.vercel.com"
        self._http: httpx.AsyncClient | None = None
        self.log = logger.bind(component="vercel")

        if not self.api_token:
            self.log.warning("No Vercel API token provided - integration will be disabled")

    @property
    def http(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    @property
    def headers(self) -> dict:
        """Get headers for Vercel API requests."""
        return {"Authorization": f"Bearer {self.api_token}"}

    @property
    def params(self) -> dict:
        """Get default query params (team_id if set)."""
        return {"teamId": self.team_id} if self.team_id else {}

    def _parse_deployment(self, data: dict) -> VercelDeployment:
        """Parse deployment data from API response."""
        # Extract git info from meta
        meta = data.get("meta", {}) or {}
        git_meta = meta.get("githubCommitRef") or meta.get("gitlabCommitRef") or meta.get("bitbucketCommitRef")

        # Determine git source
        git_source = None
        if "githubCommitRef" in meta:
            git_source = "github"
        elif "gitlabCommitRef" in meta:
            git_source = "gitlab"
        elif "bitbucketCommitRef" in meta:
            git_source = "bitbucket"

        # Parse state
        state_str = data.get("state", "").upper()
        try:
            state = DeploymentState(state_str)
        except ValueError:
            # Handle unknown states
            self.log.warning("Unknown deployment state", state=state_str)
            state = DeploymentState.QUEUED

        # Parse timestamps
        created_at = datetime.fromtimestamp(data.get("createdAt", 0) / 1000)
        ready_at = None
        if data.get("ready"):
            ready_at = datetime.fromtimestamp(data.get("ready") / 1000)

        # Get URL - prefer alias URL, fall back to deployment URL
        url = data.get("url", "")
        if data.get("alias"):
            # Use first alias if available
            aliases = data.get("alias", [])
            if aliases:
                url = aliases[0]

        # Ensure URL has protocol
        if url and not url.startswith("http"):
            url = f"https://{url}"

        return VercelDeployment(
            deployment_id=data.get("uid", data.get("id", "")),
            name=data.get("name", ""),
            url=url,
            state=state,
            git_source=git_source,
            git_repo=meta.get("githubRepo") or meta.get("gitlabRepo") or meta.get("bitbucketRepo"),
            git_branch=meta.get("githubCommitRef") or meta.get("gitlabCommitRef") or meta.get("bitbucketCommitRef"),
            git_commit_sha=meta.get("githubCommitSha") or meta.get("gitlabCommitSha") or meta.get("bitbucketCommitSha"),
            git_commit_message=meta.get("githubCommitMessage") or meta.get("gitlabCommitMessage") or meta.get("bitbucketCommitMessage"),
            target=data.get("target"),
            created_at=created_at,
            ready_at=ready_at,
            inspect_url=data.get("inspectorUrl", f"https://vercel.com/{data.get('name')}/{data.get('uid', '')}"),
        )

    def _parse_project(self, data: dict) -> VercelProject:
        """Parse project data from API response."""
        # Parse timestamps
        created_at = datetime.fromtimestamp(data.get("createdAt", 0) / 1000)
        updated_at = datetime.fromtimestamp(data.get("updatedAt", 0) / 1000)

        # Get git info from link
        link = data.get("link", {}) or {}
        git_provider = link.get("type")  # github, gitlab, bitbucket
        git_repo = link.get("repo")

        # Get production URL from alias
        production_url = None
        if data.get("alias"):
            aliases = data.get("alias", [])
            if aliases:
                alias = aliases[0]
                if isinstance(alias, dict):
                    production_url = alias.get("domain")
                else:
                    production_url = alias

        # Ensure URL has protocol
        if production_url and not production_url.startswith("http"):
            production_url = f"https://{production_url}"

        return VercelProject(
            project_id=data.get("id", ""),
            name=data.get("name", ""),
            framework=data.get("framework"),
            created_at=created_at,
            updated_at=updated_at,
            git_provider=git_provider,
            git_repo=git_repo,
            production_url=production_url,
            latest_deployments=[d.get("uid", "") for d in data.get("latestDeployments", [])],
        )

    async def test_connection(self) -> bool:
        """
        Test if API token is valid.

        Returns:
            True if connection is successful
        """
        if not self.api_token:
            return False

        try:
            response = await self.http.get(
                f"{self.base_url}/v2/user",
                headers=self.headers,
            )
            if response.status_code == 200:
                data = response.json()
                self.log.info("Vercel connection successful", user=data.get("user", {}).get("username"))
                return True
            else:
                self.log.warning("Vercel connection failed", status=response.status_code)
                return False
        except Exception as e:
            self.log.error("Vercel connection error", error=str(e))
            return False

    async def get_deployments(
        self,
        project_id: str | None = None,
        limit: int = 20,
        state: DeploymentState | None = None,
    ) -> list[VercelDeployment]:
        """
        Get recent deployments.

        Args:
            project_id: Filter by project ID (optional)
            limit: Maximum number of deployments to return (default 20, max 100)
            state: Filter by deployment state (optional)

        Returns:
            List of deployments
        """
        if not self.api_token:
            self.log.warning("Vercel API token not configured, skipping get_deployments")
            return []

        params = {**self.params, "limit": min(limit, 100)}
        if project_id:
            params["projectId"] = project_id
        if state:
            params["state"] = state.value

        try:
            response = await self.http.get(
                f"{self.base_url}/v6/deployments",
                headers=self.headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            deployments = [
                self._parse_deployment(d)
                for d in data.get("deployments", [])
            ]

            self.log.debug("Fetched deployments", count=len(deployments), project_id=project_id)
            return deployments

        except httpx.HTTPStatusError as e:
            self.log.error("Failed to get deployments", status=e.response.status_code, error=str(e))
            return []
        except Exception as e:
            self.log.error("Failed to get deployments", error=str(e))
            return []

    async def get_deployment(self, deployment_id: str) -> VercelDeployment | None:
        """
        Get a single deployment by ID.

        Args:
            deployment_id: Deployment ID (uid)

        Returns:
            Deployment or None if not found
        """
        if not self.api_token:
            self.log.warning("Vercel API token not configured, skipping get_deployment")
            return None

        try:
            response = await self.http.get(
                f"{self.base_url}/v13/deployments/{deployment_id}",
                headers=self.headers,
                params=self.params,
            )

            if response.status_code == 404:
                self.log.warning("Deployment not found", deployment_id=deployment_id)
                return None

            response.raise_for_status()
            data = response.json()

            deployment = self._parse_deployment(data)
            self.log.debug("Fetched deployment", deployment_id=deployment_id, state=deployment.state)
            return deployment

        except httpx.HTTPStatusError as e:
            self.log.error("Failed to get deployment", deployment_id=deployment_id, status=e.response.status_code)
            return None
        except Exception as e:
            self.log.error("Failed to get deployment", deployment_id=deployment_id, error=str(e))
            return None

    async def get_project_deployments(
        self,
        project_id: str,
        target: str | None = None,  # 'production' or 'preview'
        limit: int = 20,
    ) -> list[VercelDeployment]:
        """
        Get deployments for a specific project.

        Args:
            project_id: Project ID
            target: Filter by target ('production' or 'preview')
            limit: Maximum number of deployments

        Returns:
            List of deployments
        """
        if not self.api_token:
            self.log.warning("Vercel API token not configured, skipping get_project_deployments")
            return []

        params = {**self.params, "projectId": project_id, "limit": min(limit, 100)}
        if target:
            params["target"] = target

        try:
            response = await self.http.get(
                f"{self.base_url}/v6/deployments",
                headers=self.headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            deployments = [
                self._parse_deployment(d)
                for d in data.get("deployments", [])
            ]

            self.log.debug("Fetched project deployments", count=len(deployments), project_id=project_id, target=target)
            return deployments

        except httpx.HTTPStatusError as e:
            self.log.error("Failed to get project deployments", project_id=project_id, status=e.response.status_code)
            return []
        except Exception as e:
            self.log.error("Failed to get project deployments", project_id=project_id, error=str(e))
            return []

    async def get_preview_url_for_branch(
        self,
        project_id: str,
        branch: str,
    ) -> str | None:
        """
        Get the latest preview URL for a branch (useful for PR testing).

        Args:
            project_id: Project ID
            branch: Git branch name

        Returns:
            Preview URL or None if not found
        """
        if not self.api_token:
            self.log.warning("Vercel API token not configured, skipping get_preview_url_for_branch")
            return None

        # Get preview deployments for the project
        deployments = await self.get_project_deployments(
            project_id=project_id,
            target="preview",
            limit=50,  # Check more deployments to find the branch
        )

        # Find the latest ready deployment for this branch
        for deployment in deployments:
            if deployment.git_branch == branch and deployment.state == DeploymentState.READY:
                self.log.info("Found preview URL for branch", branch=branch, url=deployment.url)
                return deployment.url

        # Also check for building deployments if no ready one found
        for deployment in deployments:
            if deployment.git_branch == branch:
                self.log.info(
                    "Found deployment for branch (not ready)",
                    branch=branch,
                    state=deployment.state,
                    url=deployment.url,
                )
                return deployment.url

        self.log.warning("No preview deployment found for branch", branch=branch, project_id=project_id)
        return None

    async def get_preview_url_for_commit(
        self,
        project_id: str,
        commit_sha: str,
    ) -> str | None:
        """
        Get the preview URL for a specific commit.

        Args:
            project_id: Project ID
            commit_sha: Git commit SHA

        Returns:
            Preview URL or None if not found
        """
        if not self.api_token:
            return None

        deployments = await self.get_project_deployments(
            project_id=project_id,
            limit=50,
        )

        # Find deployment for this commit
        for deployment in deployments:
            if deployment.git_commit_sha and deployment.git_commit_sha.startswith(commit_sha[:7]):
                self.log.info("Found deployment for commit", commit_sha=commit_sha[:7], url=deployment.url)
                return deployment.url

        self.log.warning("No deployment found for commit", commit_sha=commit_sha[:7], project_id=project_id)
        return None

    async def list_projects(self) -> list[VercelProject]:
        """
        List all projects.

        Returns:
            List of projects
        """
        if not self.api_token:
            self.log.warning("Vercel API token not configured, skipping list_projects")
            return []

        try:
            response = await self.http.get(
                f"{self.base_url}/v9/projects",
                headers=self.headers,
                params=self.params,
            )
            response.raise_for_status()
            data = response.json()

            projects = [
                self._parse_project(p)
                for p in data.get("projects", [])
            ]

            self.log.debug("Fetched projects", count=len(projects))
            return projects

        except httpx.HTTPStatusError as e:
            self.log.error("Failed to list projects", status=e.response.status_code)
            return []
        except Exception as e:
            self.log.error("Failed to list projects", error=str(e))
            return []

    async def get_project(self, project_id: str) -> VercelProject | None:
        """
        Get a project by ID or name.

        Args:
            project_id: Project ID or name

        Returns:
            Project or None if not found
        """
        if not self.api_token:
            return None

        try:
            response = await self.http.get(
                f"{self.base_url}/v9/projects/{project_id}",
                headers=self.headers,
                params=self.params,
            )

            if response.status_code == 404:
                self.log.warning("Project not found", project_id=project_id)
                return None

            response.raise_for_status()
            data = response.json()

            project = self._parse_project(data)
            self.log.debug("Fetched project", project_id=project_id, name=project.name)
            return project

        except httpx.HTTPStatusError as e:
            self.log.error("Failed to get project", project_id=project_id, status=e.response.status_code)
            return None
        except Exception as e:
            self.log.error("Failed to get project", project_id=project_id, error=str(e))
            return None

    async def wait_for_deployment(
        self,
        deployment_id: str,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 10,
    ) -> VercelDeployment | None:
        """
        Wait for a deployment to reach READY or ERROR state.

        Args:
            deployment_id: Deployment ID
            timeout_seconds: Maximum time to wait (default 5 minutes)
            poll_interval_seconds: Time between checks (default 10 seconds)

        Returns:
            Final deployment state or None on timeout/error
        """
        import asyncio

        if not self.api_token:
            return None

        start_time = datetime.now()
        terminal_states = {DeploymentState.READY, DeploymentState.ERROR, DeploymentState.CANCELED}

        while True:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout_seconds:
                self.log.warning("Deployment wait timeout", deployment_id=deployment_id, elapsed=elapsed)
                return None

            deployment = await self.get_deployment(deployment_id)
            if deployment is None:
                return None

            if deployment.state in terminal_states:
                self.log.info(
                    "Deployment reached terminal state",
                    deployment_id=deployment_id,
                    state=deployment.state,
                    elapsed=elapsed,
                )
                return deployment

            self.log.debug(
                "Waiting for deployment",
                deployment_id=deployment_id,
                state=deployment.state,
                elapsed=elapsed,
            )
            await asyncio.sleep(poll_interval_seconds)

    async def get_deployment_events(
        self,
        deployment_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get build logs/events for a deployment.

        Args:
            deployment_id: Deployment ID
            limit: Maximum number of events

        Returns:
            List of event dictionaries
        """
        if not self.api_token:
            return []

        try:
            response = await self.http.get(
                f"{self.base_url}/v2/deployments/{deployment_id}/events",
                headers=self.headers,
                params={**self.params, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            events = data if isinstance(data, list) else data.get("events", [])
            self.log.debug("Fetched deployment events", deployment_id=deployment_id, count=len(events))
            return events

        except Exception as e:
            self.log.error("Failed to get deployment events", deployment_id=deployment_id, error=str(e))
            return []

    async def close(self):
        """Close the HTTP client."""
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
            self._http = None


def create_vercel_integration(
    api_token: str | None = None,
    team_id: str | None = None,
) -> VercelIntegration:
    """Factory function for VercelIntegration."""
    return VercelIntegration(api_token=api_token, team_id=team_id)
