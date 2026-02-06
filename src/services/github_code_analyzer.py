"""GitHub Code Analyzer Service - Remote code-aware healing via GitHub API.

This service provides code-aware healing capabilities by fetching code context
from GitHub repositories via the REST API, without requiring a local clone.

Used by HealingConsumer when:
- Project has repository_url configured (e.g., https://github.com/org/repo)
- Organization has GitHub integration connected with valid credentials

Replaces GitAnalyzer for production/Railway environments where local repos
are not available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GitHubCommit:
    """Represents a GitHub commit."""
    sha: str
    short_sha: str
    author: str
    author_email: str
    date: datetime
    message: str
    files_changed: list[str] = field(default_factory=list)


@dataclass
class SelectorChange:
    """Tracks a selector change in git history (API-compatible with GitAnalyzer)."""
    old_selector: str | None
    new_selector: str
    commit: GitHubCommit
    file_path: str
    line_number: int
    change_type: str  # "added", "removed", "modified", "renamed"
    context: str  # Surrounding code context


class GitHubCodeAnalyzer:
    """Analyzes GitHub repositories for code-aware test healing via REST API.

    This is the production alternative to GitAnalyzer - it uses GitHub's REST API
    to fetch commit history, file contents, and diffs without needing a local clone.

    Optimization (RAP-346 to RAP-349):
    - Uses GitHub Compare API for bulk diffs (1 API call instead of 50+)
    - Falls back to per-commit diffs only if bulk diff fails
    - Tracks API call count for metrics

    Usage:
        # Initialize with repo URL and access token
        analyzer = GitHubCodeAnalyzer(
            repo_url="https://github.com/myorg/myapp",
            access_token="ghp_xxxx",
        )

        # Find replacement selector from recent commits
        change = await analyzer.find_replacement_selector(
            failed_selector="#login-btn",
            file_hint="src/components/Login.tsx",
        )

        # Check API call metrics
        print(f"API calls made: {analyzer.api_call_count}")
    """

    # Common selector patterns (same as GitAnalyzer)
    SELECTOR_PATTERNS = [
        # data-testid (React, Vue, etc.)
        r'data-testid=["\']([^"\']+)["\']',
        r'\[data-testid=["\']?([^"\'>\]]+)',
        # id attributes
        r'id=["\']([^"\']+)["\']',
        r'#([a-zA-Z][a-zA-Z0-9_-]+)',
        # class names
        r'class(?:Name)?=["\']([^"\']+)["\']',
        r'\.([a-zA-Z][a-zA-Z0-9_-]+)',
        # aria labels
        r'aria-label=["\']([^"\']+)["\']',
        # Playwright/Cypress selectors
        r'getByTestId\(["\']([^"\']+)["\']',
        r'getByRole\(["\']([^"\']+)["\']',
        r'getByText\(["\']([^"\']+)["\']',
        r'locator\(["\']([^"\']+)["\']',
    ]

    def __init__(
        self,
        repo_url: str,
        access_token: str,
        api_base: str = "https://api.github.com",
    ):
        """Initialize GitHub Code Analyzer.

        Args:
            repo_url: Full GitHub repository URL (e.g., https://github.com/org/repo)
            access_token: GitHub personal access token or OAuth token
            api_base: GitHub API base URL (for GitHub Enterprise)
        """
        self.repo_url = repo_url
        self.access_token = access_token
        self.api_base = api_base

        # Parse owner/repo from URL
        self.owner, self.repo = self._parse_repo_url(repo_url)

        self._client: httpx.AsyncClient | None = None

        # RAP-349: Track API call count for metrics
        self._api_call_count: int = 0

    @property
    def api_call_count(self) -> int:
        """Get the total number of API calls made by this analyzer instance."""
        return self._api_call_count

    def reset_api_call_count(self) -> None:
        """Reset the API call counter to zero."""
        self._api_call_count = 0

    def _parse_repo_url(self, url: str) -> tuple[str, str]:
        """Parse owner and repo name from GitHub URL.

        Args:
            url: GitHub repository URL

        Returns:
            Tuple of (owner, repo)
        """
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub URL: {url}")

        owner = path_parts[0]
        repo = path_parts[1].removesuffix(".git")

        return owner, repo

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _headers(self) -> dict[str, str]:
        """Get headers for GitHub API requests."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Argus-Healing-Agent/1.0",
        }

    async def _api_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        accept_header: str | None = None,
        return_text: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]] | str | None:
        """Make a GitHub API request.

        Args:
            endpoint: API endpoint (e.g., /repos/{owner}/{repo}/commits)
            params: Query parameters
            accept_header: Override Accept header (e.g., for diff format)
            return_text: Return raw text instead of JSON

        Returns:
            JSON response data, raw text, or None on error
        """
        client = await self._get_client()
        url = f"{self.api_base}{endpoint}"

        # RAP-349: Track API calls
        self._api_call_count += 1

        headers = self._headers()
        if accept_header:
            headers["Accept"] = accept_header

        try:
            response = await client.get(
                url,
                headers=headers,
                params=params,
            )

            if response.status_code == 404:
                logger.debug("GitHub resource not found", endpoint=endpoint)
                return None

            response.raise_for_status()

            if return_text:
                return response.text
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                "GitHub API error",
                endpoint=endpoint,
                status=e.response.status_code,
                error=str(e),
            )
            return None
        except Exception as e:
            logger.error("GitHub request failed", endpoint=endpoint, error=str(e))
            return None

    async def get_recent_commits(
        self,
        path: str | None = None,
        since: datetime | None = None,
        limit: int = 30,
    ) -> list[GitHubCommit]:
        """Fetch recent commits, optionally filtered by path.

        Args:
            path: Filter commits to this file/directory path
            since: Only commits after this date
            limit: Maximum number of commits

        Returns:
            List of GitHubCommit objects
        """
        params: dict[str, Any] = {"per_page": min(limit, 100)}

        if path:
            params["path"] = path
        if since:
            params["since"] = since.isoformat() + "Z"

        endpoint = f"/repos/{self.owner}/{self.repo}/commits"
        data = await self._api_request(endpoint, params)

        if not data or not isinstance(data, list):
            return []

        commits = []
        for item in data[:limit]:
            commit_data = item.get("commit", {})
            author_data = commit_data.get("author", {})

            commits.append(GitHubCommit(
                sha=item.get("sha", ""),
                short_sha=item.get("sha", "")[:7],
                author=author_data.get("name", "Unknown"),
                author_email=author_data.get("email", ""),
                date=datetime.fromisoformat(
                    author_data.get("date", "").replace("Z", "+00:00")
                ) if author_data.get("date") else datetime.now(),
                message=commit_data.get("message", ""),
            ))

        return commits

    async def get_commit_diff(self, sha: str) -> str | None:
        """Fetch the diff for a specific commit.

        Args:
            sha: Commit SHA

        Returns:
            Unified diff as string, or None on error
        """
        endpoint = f"/repos/{self.owner}/{self.repo}/commits/{sha}"
        result = await self._api_request(
            endpoint,
            accept_header="application/vnd.github.v3.diff",
            return_text=True,
        )

        if result is None:
            logger.error("Failed to fetch commit diff", sha=sha[:7])
            return None

        return result if isinstance(result, str) else None

    async def get_compare_diff(
        self,
        base: str = "HEAD~30",
        head: str = "HEAD",
    ) -> str | None:
        """Get combined diff using GitHub Compare API (RAP-346).

        This is much more efficient than fetching individual commit diffs:
        - 1 API call instead of N calls for N commits
        - Returns combined diff for all changes between base and head

        Args:
            base: Base ref (commit SHA, branch, or HEAD~N notation)
            head: Head ref (commit SHA, branch, or HEAD)

        Returns:
            Combined unified diff as string, or None on error

        Note:
            GitHub Compare API has a limit of 250 commits between base and head.
            For larger ranges, the API will return a 422 error.
        """
        # GitHub Compare API requires explicit refs, not HEAD~N notation
        # We need to resolve HEAD~N to actual commit SHA first
        if "~" in base:
            # Parse HEAD~N to get the offset
            match = re.match(r"HEAD~(\d+)", base)
            if match:
                offset = int(match.group(1))
                # Get commits to find the base SHA
                commits = await self.get_recent_commits(limit=offset + 1)
                if len(commits) > offset:
                    base = commits[offset].sha
                else:
                    logger.warning(
                        "Not enough commits for base ref",
                        requested_offset=offset,
                        available_commits=len(commits),
                    )
                    return None

        # Resolve HEAD to default branch
        if head == "HEAD":
            repo_info = await self._api_request(
                f"/repos/{self.owner}/{self.repo}"
            )
            if repo_info and isinstance(repo_info, dict):
                head = repo_info.get("default_branch", "main")
            else:
                head = "main"

        endpoint = f"/repos/{self.owner}/{self.repo}/compare/{base}...{head}"

        result = await self._api_request(
            endpoint,
            accept_header="application/vnd.github.v3.diff",
            return_text=True,
        )

        if result is None:
            logger.debug(
                "Failed to fetch compare diff",
                base=base[:7] if len(base) > 7 else base,
                head=head,
            )
            return None

        return result if isinstance(result, str) else None

    async def get_file_content(
        self,
        path: str,
        ref: str = "HEAD",
    ) -> str | None:
        """Fetch file content at a specific ref.

        Args:
            path: File path within repository
            ref: Git ref (branch, tag, or SHA)

        Returns:
            File content as string, or None if not found
        """
        endpoint = f"/repos/{self.owner}/{self.repo}/contents/{path}"
        params = {"ref": ref}

        data = await self._api_request(endpoint, params)

        if not data or isinstance(data, list):
            return None

        # Content is base64 encoded
        import base64
        content = data.get("content", "")
        if content:
            try:
                return base64.b64decode(content).decode("utf-8")
            except Exception:
                return None
        return None

    async def find_selector_in_diff(
        self,
        diff: str,
        target_selector: str,
    ) -> tuple[str | None, str | None, int]:
        """Find old and new selector values in a diff.

        Args:
            diff: Unified diff text
            target_selector: The selector we're looking for

        Returns:
            Tuple of (old_selector, new_selector, line_number)
        """
        old_selector = None
        new_selector = None
        line_number = 0

        # Normalize selector for matching
        normalized_target = target_selector.strip().lower()

        current_line = 0
        for line in diff.split("\n"):
            # Track line numbers from @@ header
            if line.startswith("@@"):
                match = re.search(r"\+(\d+)", line)
                if match:
                    current_line = int(match.group(1))
                continue

            if line.startswith("+") and not line.startswith("+++"):
                current_line += 1
                # Check if this line contains a selector similar to target
                for pattern in self.SELECTOR_PATTERNS:
                    matches = re.findall(pattern, line)
                    for match in matches:
                        if normalized_target in match.lower() or match.lower() in normalized_target:
                            new_selector = match
                            line_number = current_line

            elif line.startswith("-") and not line.startswith("---"):
                # Check removed lines for old selector
                for pattern in self.SELECTOR_PATTERNS:
                    matches = re.findall(pattern, line)
                    for match in matches:
                        if normalized_target in match.lower() or match.lower() in normalized_target:
                            old_selector = match

        return old_selector, new_selector, line_number

    def _find_selector_in_bulk_diff(
        self,
        diff: str,
        target_selector: str,
        file_hint: str | None = None,
    ) -> SelectorChange | None:
        """Parse bulk diff locally to find selector changes (RAP-347).

        This method parses a multi-file diff (from Compare API) and extracts
        selector changes without making additional API calls.

        Args:
            diff: Combined unified diff text (multi-file)
            target_selector: The selector we're looking for
            file_hint: Optional hint to filter files

        Returns:
            SelectorChange if found, or None
        """
        # Normalize selector for matching
        normalized_target = target_selector.strip().lower()

        # Split diff into per-file chunks
        # Each file section starts with "diff --git a/... b/..."
        file_chunks = re.split(r"(?=^diff --git)", diff, flags=re.MULTILINE)

        for chunk in file_chunks:
            if not chunk.strip():
                continue

            # Extract file path from diff header
            file_path_match = re.search(r"^\+\+\+ b/(.+)$", chunk, re.MULTILINE)
            if not file_path_match:
                continue

            file_path = file_path_match.group(1)

            # If file_hint is provided, filter to relevant files
            if file_hint:
                # Allow partial matches (e.g., "Login.tsx" matches "src/components/Login.tsx")
                if file_hint.lower() not in file_path.lower():
                    continue

            # Parse this file's diff for selector changes
            old_selector = None
            new_selector = None
            line_number = 0
            current_line = 0

            for line in chunk.split("\n"):
                # Track line numbers from @@ header
                if line.startswith("@@"):
                    match = re.search(r"\+(\d+)", line)
                    if match:
                        current_line = int(match.group(1))
                    continue

                if line.startswith("+") and not line.startswith("+++"):
                    current_line += 1
                    # Check if this line contains a selector similar to target
                    for pattern in self.SELECTOR_PATTERNS:
                        matches = re.findall(pattern, line)
                        for sel_match in matches:
                            if (
                                normalized_target in sel_match.lower()
                                or sel_match.lower() in normalized_target
                            ):
                                new_selector = sel_match
                                line_number = current_line

                elif line.startswith("-") and not line.startswith("---"):
                    # Check removed lines for old selector
                    for pattern in self.SELECTOR_PATTERNS:
                        matches = re.findall(pattern, line)
                        for sel_match in matches:
                            if (
                                normalized_target in sel_match.lower()
                                or sel_match.lower() in normalized_target
                            ):
                                old_selector = sel_match

            # If we found a change in this file, create a SelectorChange
            if new_selector and old_selector != new_selector:
                # Extract context from this chunk
                context = self._extract_context_from_diff(chunk, line_number)

                # Create a minimal commit object for bulk diff results
                # We don't have individual commit info in bulk diff
                from datetime import timezone

                bulk_commit = GitHubCommit(
                    sha="bulk_compare",
                    short_sha="bulk",
                    author="Unknown (bulk compare)",
                    author_email="",
                    date=datetime.now(timezone.utc),
                    message="Found via GitHub Compare API (bulk diff)",
                )

                logger.info(
                    "Found selector change in bulk diff",
                    old_selector=old_selector,
                    new_selector=new_selector,
                    file=file_path,
                )

                return SelectorChange(
                    old_selector=old_selector,
                    new_selector=new_selector,
                    commit=bulk_commit,
                    file_path=file_path,
                    line_number=line_number,
                    change_type="modified",
                    context=context,
                )

        return None

    async def find_replacement_selector(
        self,
        failed_selector: str,
        file_hint: str | None = None,
        days: int = 30,
        max_commits: int = 50,
    ) -> SelectorChange | None:
        """Find a replacement selector by analyzing recent commits.

        This is the main entry point for code-aware healing. It searches
        recent commits for changes that might have affected the selector.

        RAP-348 Optimization: Uses bulk diff (Compare API) first, falling back
        to per-commit diffs only if bulk diff fails or doesn't find the change.

        Args:
            failed_selector: The selector that failed (e.g., "#login-btn")
            file_hint: Optional hint about which file to look in
            days: How many days of history to search
            max_commits: Maximum commits to search (default 50)

        Returns:
            SelectorChange with old and new selector, or None if not found
        """
        initial_api_calls = self._api_call_count

        logger.info(
            "Searching for selector change",
            selector=failed_selector[:50],
            file_hint=file_hint,
            days=days,
        )

        # RAP-348: Try bulk diff first (1-2 API calls instead of 50+)
        # This is much more efficient for most cases
        bulk_diff = await self.get_compare_diff(
            base=f"HEAD~{max_commits}",
            head="HEAD",
        )

        if bulk_diff:
            logger.debug(
                "Using bulk diff for selector search",
                diff_size=len(bulk_diff),
            )

            result = self._find_selector_in_bulk_diff(
                bulk_diff,
                failed_selector,
                file_hint=file_hint,
            )

            if result:
                logger.info(
                    "Found selector via bulk diff",
                    api_calls=self._api_call_count - initial_api_calls,
                    old_selector=result.old_selector,
                    new_selector=result.new_selector,
                )
                return result

            logger.debug("Bulk diff did not contain selector change")

        # Fallback: Per-commit search if bulk diff fails or doesn't find it
        # This is slower but may find changes in very old commits or
        # when the Compare API has limitations (e.g., >250 commits)
        logger.debug("Falling back to per-commit diff search")

        from datetime import timedelta, timezone

        since = datetime.now(timezone.utc) - timedelta(days=days)

        # Get recent commits, optionally filtered by file
        commits = await self.get_recent_commits(
            path=file_hint,
            since=since,
            limit=max_commits,
        )

        logger.info(
            "Searching commits for selector change (fallback mode)",
            selector=failed_selector[:50],
            commits_count=len(commits),
            file_hint=file_hint,
        )

        for commit in commits:
            # Get the diff for this commit
            diff = await self.get_commit_diff(commit.sha)
            if not diff:
                continue

            # Look for selector changes in the diff
            old_sel, new_sel, line_num = await self.find_selector_in_diff(
                diff, failed_selector
            )

            if new_sel and old_sel != new_sel:
                # Extract file path from diff
                file_path = ""
                for line in diff.split("\n"):
                    if line.startswith("+++ b/"):
                        file_path = line[6:]
                        break

                # Get surrounding context
                context = self._extract_context_from_diff(diff, line_num)

                logger.info(
                    "Found selector change in commit (fallback mode)",
                    commit=commit.short_sha,
                    old_selector=old_sel,
                    new_selector=new_sel,
                    file=file_path,
                    api_calls=self._api_call_count - initial_api_calls,
                )

                return SelectorChange(
                    old_selector=old_sel,
                    new_selector=new_sel,
                    commit=commit,
                    file_path=file_path,
                    line_number=line_num,
                    change_type="modified",
                    context=context,
                )

        logger.debug(
            "No selector change found in recent commits",
            api_calls=self._api_call_count - initial_api_calls,
        )
        return None

    def _extract_context_from_diff(self, diff: str, target_line: int) -> str:
        """Extract context lines around a target line from diff.

        Args:
            diff: Unified diff text
            target_line: Line number to get context for

        Returns:
            Context string (few lines before/after)
        """
        lines = []
        in_hunk = False
        current_line = 0

        for line in diff.split("\n"):
            if line.startswith("@@"):
                match = re.search(r"\+(\d+)", line)
                if match:
                    current_line = int(match.group(1))
                    in_hunk = True
                continue

            if in_hunk and not line.startswith("-"):
                if abs(current_line - target_line) <= 5:
                    lines.append(line)
                current_line += 1

        return "\n".join(lines[-10:])  # Return last 10 lines of context

    async def find_selector_changes(
        self,
        selector: str,
        days: int = 30,
    ) -> list[SelectorChange]:
        """Find all changes to a selector in recent history.

        Args:
            selector: The selector to search for
            days: Days of history to search

        Returns:
            List of SelectorChange objects
        """
        from datetime import timedelta, timezone

        since = datetime.now(timezone.utc) - timedelta(days=days)
        commits = await self.get_recent_commits(since=since, limit=100)

        changes = []
        for commit in commits:
            diff = await self.get_commit_diff(commit.sha)
            if not diff:
                continue

            old_sel, new_sel, line_num = await self.find_selector_in_diff(
                diff, selector
            )

            if old_sel or new_sel:
                file_path = ""
                for line in diff.split("\n"):
                    if line.startswith("+++ b/"):
                        file_path = line[6:]
                        break

                changes.append(SelectorChange(
                    old_selector=old_sel,
                    new_selector=new_sel or "",
                    commit=commit,
                    file_path=file_path,
                    line_number=line_num,
                    change_type="modified" if old_sel and new_sel else ("added" if new_sel else "removed"),
                    context=self._extract_context_from_diff(diff, line_num),
                ))

        return changes

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


async def get_github_analyzer_for_project(
    project_id: str,
    org_id: str,
) -> GitHubCodeAnalyzer | None:
    """Get a GitHubCodeAnalyzer for a project if GitHub is configured.

    This factory function:
    1. Fetches the project's repository_url
    2. Fetches the organization's GitHub integration credentials
    3. Returns a configured analyzer, or None if not available

    Args:
        project_id: The project UUID
        org_id: The organization UUID

    Returns:
        Configured GitHubCodeAnalyzer or None
    """
    from src.services.supabase_client import get_supabase_client

    supabase = get_supabase_client()
    if not supabase.is_configured:
        logger.debug("Supabase not configured, cannot get GitHub analyzer")
        return None

    # Fetch project to get repository_url
    project_result = await supabase.request(
        f"/projects?id=eq.{project_id}&select=repository_url"
    )
    project_data = project_result.get("data", [])

    if not project_data or not project_data[0].get("repository_url"):
        logger.debug("Project has no repository_url configured", project_id=project_id)
        return None

    repo_url = project_data[0]["repository_url"]

    # Fetch GitHub integration for this organization
    integration_result = await supabase.request(
        f"/integrations?organization_id=eq.{org_id}&type=eq.github&status=eq.active&select=credentials"
    )
    integration_data = integration_result.get("data", [])

    if not integration_data:
        logger.debug(
            "No active GitHub integration for organization",
            org_id=org_id,
        )
        return None

    credentials = integration_data[0].get("credentials", {})
    access_token = credentials.get("access_token")

    if not access_token:
        logger.warning(
            "GitHub integration missing access_token",
            org_id=org_id,
        )
        return None

    logger.info(
        "Created GitHub code analyzer",
        repo_url=repo_url,
        org_id=org_id,
    )

    return GitHubCodeAnalyzer(
        repo_url=repo_url,
        access_token=access_token,
    )
