"""
Proactive CI/CD Monitor Service for GitLab Duo-style Analysis.

RAP-234: Implements a polling-based monitor that watches for new MRs/PRs
across GitHub and GitLab, triggering automated analysis and test suggestions.

Features:
- Background polling for new MRs/PRs
- Support for both GitHub and GitLab
- Integration with MRAnalyzerAgent for intelligent analysis
- Automatic comment posting with test suggestions
- Multi-tenant support with project isolation
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import httpx
import structlog

from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger(__name__)


class VCSPlatform(str, Enum):
    """Supported VCS platforms."""
    GITHUB = "github"
    GITLAB = "gitlab"


class PRState(str, Enum):
    """Pull Request / Merge Request states."""
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"
    DRAFT = "draft"


@dataclass
class FileChange:
    """Represents a changed file in a PR/MR."""
    path: str
    status: str  # added, modified, removed, renamed
    additions: int = 0
    deletions: int = 0
    patch: str | None = None
    previous_path: str | None = None  # For renames


@dataclass
class PRInfo:
    """Information about a GitHub Pull Request."""
    platform: VCSPlatform = VCSPlatform.GITHUB
    repo: str = ""  # owner/repo format
    number: int = 0
    title: str = ""
    description: str = ""
    state: PRState = PRState.OPEN
    author: str = ""
    head_sha: str = ""
    head_branch: str = ""
    base_branch: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    files: list[FileChange] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    is_draft: bool = False
    html_url: str = ""

    @property
    def identifier(self) -> str:
        """Unique identifier for this PR."""
        return f"github:{self.repo}:{self.number}"


@dataclass
class MRInfo:
    """Information about a GitLab Merge Request."""
    platform: VCSPlatform = VCSPlatform.GITLAB
    project_path: str = ""  # namespace/project format
    iid: int = 0  # Internal ID within project
    mr_id: int = 0  # Global MR ID
    title: str = ""
    description: str = ""
    state: PRState = PRState.OPEN
    author: str = ""
    source_branch: str = ""
    target_branch: str = ""
    sha: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    files: list[FileChange] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    is_draft: bool = False
    web_url: str = ""

    @property
    def identifier(self) -> str:
        """Unique identifier for this MR."""
        return f"gitlab:{self.project_path}:{self.iid}"


@dataclass
class MonitoredProject:
    """Configuration for a project being monitored."""
    project_id: str  # Argus project ID
    org_id: str
    platform: VCSPlatform
    repo_identifier: str  # owner/repo for GitHub, project_path for GitLab
    enabled: bool = True
    last_poll_at: datetime | None = None
    poll_interval_seconds: int = 60
    auto_comment: bool = True
    analysis_settings: dict[str, Any] = field(default_factory=dict)


class CICDMonitor:
    """
    Proactive MR/PR monitoring service.

    Polls GitHub and GitLab for new merge requests and pull requests,
    triggering AI-powered analysis and posting test suggestions as comments.

    Usage:
        monitor = CICDMonitor(poll_interval=60)
        await monitor.start()

        # Add projects to monitor
        await monitor.add_project(
            project_id="proj_123",
            org_id="org_456",
            platform=VCSPlatform.GITHUB,
            repo="myorg/myapp",
        )

        # Later...
        await monitor.stop()
    """

    def __init__(
        self,
        poll_interval: int = 60,
        github_token: str | None = None,
        gitlab_token: str | None = None,
        gitlab_url: str = "https://gitlab.com",
        max_concurrent_analyses: int = 5,
    ):
        """
        Initialize the CI/CD monitor.

        Args:
            poll_interval: Default polling interval in seconds
            github_token: GitHub API token (or GITHUB_TOKEN env var)
            gitlab_token: GitLab API token (or GITLAB_TOKEN env var)
            gitlab_url: GitLab instance URL (default: gitlab.com)
            max_concurrent_analyses: Max concurrent analysis tasks
        """
        self.poll_interval = poll_interval
        self._github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self._gitlab_token = gitlab_token or os.environ.get("GITLAB_TOKEN")
        self._gitlab_url = gitlab_url.rstrip("/")
        self._max_concurrent = max_concurrent_analyses

        self._running = False
        self._poll_task: asyncio.Task | None = None
        self._analysis_semaphore = asyncio.Semaphore(max_concurrent_analyses)

        # Track monitored projects
        self._projects: dict[str, MonitoredProject] = {}

        # Track already-processed PRs/MRs to avoid duplicate analysis
        self._processed: set[str] = set()

        self.log = logger.bind(component="cicd_monitor")

    @property
    def github_headers(self) -> dict:
        """Get headers for GitHub API requests."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._github_token:
            headers["Authorization"] = f"Bearer {self._github_token}"
        return headers

    @property
    def gitlab_headers(self) -> dict:
        """Get headers for GitLab API requests."""
        headers = {
            "Accept": "application/json",
        }
        if self._gitlab_token:
            headers["PRIVATE-TOKEN"] = self._gitlab_token
        return headers

    async def start(self) -> None:
        """Start the monitoring loop."""
        if self._running:
            self.log.warning("Monitor already running")
            return

        self._running = True
        self._poll_task = asyncio.create_task(self._polling_loop())
        self.log.info(
            "CI/CD monitor started",
            poll_interval=self.poll_interval,
            max_concurrent=self._max_concurrent,
        )

    async def stop(self) -> None:
        """Stop the monitoring loop."""
        if not self._running:
            self.log.warning("Monitor not running")
            return

        self._running = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        self.log.info(
            "CI/CD monitor stopped",
            processed_count=len(self._processed),
        )

    async def add_project(
        self,
        project_id: str,
        org_id: str,
        platform: VCSPlatform,
        repo: str,
        poll_interval: int | None = None,
        auto_comment: bool = True,
        analysis_settings: dict[str, Any] | None = None,
    ) -> MonitoredProject:
        """
        Add a project to monitor.

        Args:
            project_id: Argus project ID
            org_id: Organization ID
            platform: VCS platform (github or gitlab)
            repo: Repository identifier (owner/repo for GitHub, path for GitLab)
            poll_interval: Custom polling interval (uses default if not set)
            auto_comment: Whether to auto-post analysis comments
            analysis_settings: Custom analysis configuration

        Returns:
            The created MonitoredProject
        """
        project = MonitoredProject(
            project_id=project_id,
            org_id=org_id,
            platform=platform,
            repo_identifier=repo,
            enabled=True,
            poll_interval_seconds=poll_interval or self.poll_interval,
            auto_comment=auto_comment,
            analysis_settings=analysis_settings or {},
        )

        key = f"{platform.value}:{repo}"
        self._projects[key] = project

        self.log.info(
            "Added project to monitor",
            project_id=project_id,
            platform=platform.value,
            repo=repo,
        )

        return project

    async def remove_project(self, platform: VCSPlatform, repo: str) -> bool:
        """Remove a project from monitoring."""
        key = f"{platform.value}:{repo}"
        if key in self._projects:
            del self._projects[key]
            self.log.info("Removed project from monitor", platform=platform.value, repo=repo)
            return True
        return False

    async def poll_github(self, project_id: str, repo: str) -> list[PRInfo]:
        """
        Poll GitHub for open pull requests.

        Args:
            project_id: Argus project ID (for logging/tracking)
            repo: Repository in owner/repo format

        Returns:
            List of PRInfo objects for open PRs
        """
        if not self._github_token:
            self.log.warning("GitHub token not configured, skipping poll")
            return []

        prs: list[PRInfo] = []
        url = f"https://api.github.com/repos/{repo}/pulls"
        params = {
            "state": "open",
            "sort": "updated",
            "direction": "desc",
            "per_page": 30,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers=self.github_headers,
                    params=params,
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                for pr_data in data:
                    pr = self._parse_github_pr(repo, pr_data)

                    # Fetch files for each PR
                    pr.files = await self._fetch_github_pr_files(client, repo, pr.number)
                    prs.append(pr)

                self.log.debug(
                    "Polled GitHub PRs",
                    repo=repo,
                    count=len(prs),
                )

            except httpx.HTTPError as e:
                self.log.error(
                    "Failed to poll GitHub",
                    repo=repo,
                    error=str(e),
                )

        return prs

    async def poll_gitlab(self, project_id: str, repo: str) -> list[MRInfo]:
        """
        Poll GitLab for open merge requests.

        Args:
            project_id: Argus project ID (for logging/tracking)
            repo: Project path (namespace/project)

        Returns:
            List of MRInfo objects for open MRs
        """
        if not self._gitlab_token:
            self.log.warning("GitLab token not configured, skipping poll")
            return []

        mrs: list[MRInfo] = []

        # URL-encode the project path
        import urllib.parse
        encoded_path = urllib.parse.quote(repo, safe="")

        url = f"{self._gitlab_url}/api/v4/projects/{encoded_path}/merge_requests"
        params = {
            "state": "opened",
            "order_by": "updated_at",
            "sort": "desc",
            "per_page": 30,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers=self.gitlab_headers,
                    params=params,
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                for mr_data in data:
                    mr = self._parse_gitlab_mr(repo, mr_data)

                    # Fetch changes for each MR
                    mr.files = await self._fetch_gitlab_mr_changes(client, repo, mr.iid)
                    mrs.append(mr)

                self.log.debug(
                    "Polled GitLab MRs",
                    project=repo,
                    count=len(mrs),
                )

            except httpx.HTTPError as e:
                self.log.error(
                    "Failed to poll GitLab",
                    project=repo,
                    error=str(e),
                )

        return mrs

    async def trigger_analysis(self, pr_or_mr: PRInfo | MRInfo) -> dict[str, Any] | None:
        """
        Trigger analysis for a PR or MR.

        Args:
            pr_or_mr: PRInfo or MRInfo to analyze

        Returns:
            Analysis result dict or None if analysis fails
        """
        # Check if already processed
        identifier = pr_or_mr.identifier
        if identifier in self._processed:
            self.log.debug("Already processed", identifier=identifier)
            return None

        # Use semaphore to limit concurrent analyses
        async with self._analysis_semaphore:
            try:
                # Import here to avoid circular imports
                from src.agents.mr_analyzer import MRAnalyzerAgent

                # Get the project config
                platform = pr_or_mr.platform.value
                repo = pr_or_mr.repo if isinstance(pr_or_mr, PRInfo) else pr_or_mr.project_path
                key = f"{platform}:{repo}"

                project = self._projects.get(key)
                if not project:
                    self.log.warning("No project config found", key=key)
                    return None

                # Create analyzer and run analysis
                analyzer = MRAnalyzerAgent()

                analysis = await analyzer.analyze(
                    changes=pr_or_mr.files,
                    project_id=project.project_id,
                )

                # Get test suggestions
                suggestions = await analyzer.suggest_tests(analysis)

                # Generate comment
                comment = await analyzer.generate_comment(suggestions)

                result = {
                    "identifier": identifier,
                    "analysis": analysis,
                    "suggestions": suggestions,
                    "comment": comment,
                    "analyzed_at": datetime.now(UTC).isoformat(),
                }

                # Post comment if auto_comment enabled
                if project.auto_comment and comment:
                    from src.integrations.comment_poster import CommentPoster

                    poster = CommentPoster()

                    if isinstance(pr_or_mr, PRInfo):
                        await poster.post_to_github(
                            repo=pr_or_mr.repo,
                            pr_number=pr_or_mr.number,
                            comment=comment,
                        )
                    else:
                        await poster.post_to_gitlab(
                            project=pr_or_mr.project_path,
                            mr_iid=pr_or_mr.iid,
                            comment=comment,
                        )

                # Store analysis result
                await self._store_analysis_result(project, pr_or_mr, result)

                # Mark as processed
                self._processed.add(identifier)

                self.log.info(
                    "Analysis completed",
                    identifier=identifier,
                    suggestions_count=len(suggestions),
                )

                return result

            except ImportError as e:
                self.log.error(
                    "Failed to import analyzer",
                    error=str(e),
                )
                return None
            except Exception as e:
                self.log.exception(
                    "Analysis failed",
                    identifier=identifier,
                    error=str(e),
                )
                return None

    async def _polling_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._poll_all_projects()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.exception("Error in polling loop", error=str(e))
                await asyncio.sleep(self.poll_interval)

    async def _poll_all_projects(self) -> None:
        """Poll all registered projects."""
        for key, project in self._projects.items():
            if not project.enabled:
                continue

            # Check if enough time has passed since last poll
            if project.last_poll_at:
                elapsed = (datetime.now(UTC) - project.last_poll_at).total_seconds()
                if elapsed < project.poll_interval_seconds:
                    continue

            try:
                if project.platform == VCSPlatform.GITHUB:
                    prs = await self.poll_github(
                        project.project_id,
                        project.repo_identifier,
                    )
                    for pr in prs:
                        if pr.identifier not in self._processed:
                            await self.trigger_analysis(pr)
                else:
                    mrs = await self.poll_gitlab(
                        project.project_id,
                        project.repo_identifier,
                    )
                    for mr in mrs:
                        if mr.identifier not in self._processed:
                            await self.trigger_analysis(mr)

                project.last_poll_at = datetime.now(UTC)

            except Exception as e:
                self.log.error(
                    "Failed to poll project",
                    project_id=project.project_id,
                    platform=project.platform.value,
                    error=str(e),
                )

    def _parse_github_pr(self, repo: str, data: dict) -> PRInfo:
        """Parse GitHub PR API response into PRInfo."""
        state = PRState.OPEN
        if data.get("merged_at"):
            state = PRState.MERGED
        elif data.get("state") == "closed":
            state = PRState.CLOSED
        elif data.get("draft"):
            state = PRState.DRAFT

        return PRInfo(
            platform=VCSPlatform.GITHUB,
            repo=repo,
            number=data.get("number", 0),
            title=data.get("title", ""),
            description=data.get("body") or "",
            state=state,
            author=data.get("user", {}).get("login", ""),
            head_sha=data.get("head", {}).get("sha", ""),
            head_branch=data.get("head", {}).get("ref", ""),
            base_branch=data.get("base", {}).get("ref", ""),
            created_at=self._parse_datetime(data.get("created_at")),
            updated_at=self._parse_datetime(data.get("updated_at")),
            labels=[l.get("name", "") for l in data.get("labels", [])],
            is_draft=data.get("draft", False),
            html_url=data.get("html_url", ""),
        )

    def _parse_gitlab_mr(self, project_path: str, data: dict) -> MRInfo:
        """Parse GitLab MR API response into MRInfo."""
        state_map = {
            "opened": PRState.OPEN,
            "closed": PRState.CLOSED,
            "merged": PRState.MERGED,
        }
        state = state_map.get(data.get("state", ""), PRState.OPEN)

        if data.get("work_in_progress") or data.get("draft"):
            state = PRState.DRAFT

        return MRInfo(
            platform=VCSPlatform.GITLAB,
            project_path=project_path,
            iid=data.get("iid", 0),
            mr_id=data.get("id", 0),
            title=data.get("title", ""),
            description=data.get("description") or "",
            state=state,
            author=data.get("author", {}).get("username", ""),
            source_branch=data.get("source_branch", ""),
            target_branch=data.get("target_branch", ""),
            sha=data.get("sha", ""),
            created_at=self._parse_datetime(data.get("created_at")),
            updated_at=self._parse_datetime(data.get("updated_at")),
            labels=data.get("labels", []),
            is_draft=data.get("work_in_progress", False) or data.get("draft", False),
            web_url=data.get("web_url", ""),
        )

    async def _fetch_github_pr_files(
        self,
        client: httpx.AsyncClient,
        repo: str,
        pr_number: int,
    ) -> list[FileChange]:
        """Fetch files changed in a GitHub PR."""
        files: list[FileChange] = []
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"

        try:
            response = await client.get(
                url,
                headers=self.github_headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            for file_data in data:
                files.append(FileChange(
                    path=file_data.get("filename", ""),
                    status=file_data.get("status", "modified"),
                    additions=file_data.get("additions", 0),
                    deletions=file_data.get("deletions", 0),
                    patch=file_data.get("patch"),
                    previous_path=file_data.get("previous_filename"),
                ))
        except httpx.HTTPError as e:
            self.log.warning(
                "Failed to fetch PR files",
                repo=repo,
                pr=pr_number,
                error=str(e),
            )

        return files

    async def _fetch_gitlab_mr_changes(
        self,
        client: httpx.AsyncClient,
        project_path: str,
        mr_iid: int,
    ) -> list[FileChange]:
        """Fetch files changed in a GitLab MR."""
        files: list[FileChange] = []

        import urllib.parse
        encoded_path = urllib.parse.quote(project_path, safe="")
        url = f"{self._gitlab_url}/api/v4/projects/{encoded_path}/merge_requests/{mr_iid}/changes"

        try:
            response = await client.get(
                url,
                headers=self.gitlab_headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            for change in data.get("changes", []):
                # Map GitLab change status
                if change.get("new_file"):
                    status = "added"
                elif change.get("deleted_file"):
                    status = "removed"
                elif change.get("renamed_file"):
                    status = "renamed"
                else:
                    status = "modified"

                # Count additions/deletions from diff
                diff = change.get("diff", "")
                additions = diff.count("\n+") - diff.count("\n+++")
                deletions = diff.count("\n-") - diff.count("\n---")

                files.append(FileChange(
                    path=change.get("new_path", ""),
                    status=status,
                    additions=max(0, additions),
                    deletions=max(0, deletions),
                    patch=diff,
                    previous_path=change.get("old_path") if change.get("renamed_file") else None,
                ))
        except httpx.HTTPError as e:
            self.log.warning(
                "Failed to fetch MR changes",
                project=project_path,
                mr=mr_iid,
                error=str(e),
            )

        return files

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        """Parse ISO datetime string."""
        if not value:
            return None
        try:
            # Handle Z suffix
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None

    async def _store_analysis_result(
        self,
        project: MonitoredProject,
        pr_or_mr: PRInfo | MRInfo,
        result: dict[str, Any],
    ) -> None:
        """Store analysis result in database."""
        try:
            supabase = get_supabase_client()

            record = {
                "project_id": project.project_id,
                "org_id": project.org_id,
                "platform": pr_or_mr.platform.value,
                "identifier": pr_or_mr.identifier,
                "pr_number": pr_or_mr.number if isinstance(pr_or_mr, PRInfo) else pr_or_mr.iid,
                "title": pr_or_mr.title,
                "analysis_data": result,
                "analyzed_at": datetime.now(UTC).isoformat(),
            }

            await supabase.insert("mr_analysis_results", record)

        except Exception as e:
            self.log.warning(
                "Failed to store analysis result",
                identifier=pr_or_mr.identifier,
                error=str(e),
            )

    @property
    def is_running(self) -> bool:
        """Check if the monitor is running."""
        return self._running

    @property
    def monitored_projects_count(self) -> int:
        """Get the number of monitored projects."""
        return len(self._projects)

    @property
    def processed_count(self) -> int:
        """Get the number of processed PRs/MRs."""
        return len(self._processed)


# =============================================================================
# Global Instance
# =============================================================================

_monitor: CICDMonitor | None = None


def get_cicd_monitor() -> CICDMonitor:
    """Get the global CICDMonitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = CICDMonitor()
    return _monitor


async def start_cicd_monitoring() -> CICDMonitor:
    """Start the global CI/CD monitor."""
    monitor = get_cicd_monitor()
    await monitor.start()
    return monitor


async def stop_cicd_monitoring() -> None:
    """Stop the global CI/CD monitor."""
    monitor = get_cicd_monitor()
    await monitor.stop()
