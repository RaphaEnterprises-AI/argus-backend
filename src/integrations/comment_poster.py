"""
Comment Poster - Posts analysis comments to GitHub and GitLab.

RAP-234: Provides a unified interface for posting MR/PR analysis comments
to both GitHub and GitLab, with support for updating existing comments.

Features:
- Post comments to GitHub PRs
- Post comments to GitLab MRs
- Update existing comments (avoid duplicates)
- Rate limiting and error handling
- Comment signature for identification
"""

from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
import structlog

logger = structlog.get_logger(__name__)


# Comment signature to identify our comments
COMMENT_SIGNATURE = "<!-- argus-e2e-testing-agent -->"


@dataclass
class CommentResult:
    """Result of a comment operation."""
    success: bool
    comment_id: str | None = None
    url: str | None = None
    error: str | None = None
    was_updated: bool = False


class CommentPoster:
    """
    Posts analysis comments to GitHub and GitLab.

    Handles the posting of test suggestions and analysis results
    to pull requests (GitHub) and merge requests (GitLab).

    Usage:
        poster = CommentPoster()

        # Post to GitHub
        result = await poster.post_to_github(
            repo="myorg/myapp",
            pr_number=123,
            comment="## Analysis\\n\\nTest suggestions...",
        )

        # Post to GitLab
        result = await poster.post_to_gitlab(
            project="mygroup/myproject",
            mr_iid=456,
            comment="## Analysis\\n\\nTest suggestions...",
        )

        # Update an existing comment
        result = await poster.update_comment(
            comment_id="github:myorg/myapp:123456",
            new_content="Updated analysis...",
        )
    """

    def __init__(
        self,
        github_token: str | None = None,
        gitlab_token: str | None = None,
        gitlab_url: str = "https://gitlab.com",
        update_existing: bool = True,
    ):
        """
        Initialize the CommentPoster.

        Args:
            github_token: GitHub API token (or GITHUB_TOKEN env var)
            gitlab_token: GitLab API token (or GITLAB_TOKEN env var)
            gitlab_url: GitLab instance URL (default: gitlab.com)
            update_existing: If True, update existing comments instead of creating new ones
        """
        self._github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self._gitlab_token = gitlab_token or os.environ.get("GITLAB_TOKEN")
        self._gitlab_url = gitlab_url.rstrip("/")
        self._update_existing = update_existing

        self.log = logger.bind(component="comment_poster")

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
            "Content-Type": "application/json",
        }
        if self._gitlab_token:
            headers["PRIVATE-TOKEN"] = self._gitlab_token
        return headers

    async def post_to_github(
        self,
        repo: str,
        pr_number: int,
        comment: str,
        update_existing: bool | None = None,
    ) -> CommentResult:
        """
        Post a comment to a GitHub pull request.

        Args:
            repo: Repository in owner/repo format
            pr_number: Pull request number
            comment: Comment body (markdown)
            update_existing: Override instance setting for updating

        Returns:
            CommentResult with success status and details
        """
        if not self._github_token:
            self.log.warning("GitHub token not configured")
            return CommentResult(
                success=False,
                error="GitHub token not configured",
            )

        should_update = update_existing if update_existing is not None else self._update_existing

        # Add signature to comment
        full_comment = f"{COMMENT_SIGNATURE}\n{comment}"

        async with httpx.AsyncClient() as client:
            # Check for existing comment if update mode
            existing_comment_id = None
            if should_update:
                existing_comment_id = await self._find_github_comment(
                    client, repo, pr_number
                )

            if existing_comment_id:
                # Update existing comment
                return await self._update_github_comment(
                    client, repo, existing_comment_id, full_comment
                )
            else:
                # Create new comment
                return await self._create_github_comment(
                    client, repo, pr_number, full_comment
                )

    async def post_to_gitlab(
        self,
        project: str,
        mr_iid: int,
        comment: str,
        update_existing: bool | None = None,
    ) -> CommentResult:
        """
        Post a comment to a GitLab merge request.

        Args:
            project: Project path (namespace/project)
            mr_iid: Merge request internal ID
            comment: Comment body (markdown)
            update_existing: Override instance setting for updating

        Returns:
            CommentResult with success status and details
        """
        if not self._gitlab_token:
            self.log.warning("GitLab token not configured")
            return CommentResult(
                success=False,
                error="GitLab token not configured",
            )

        should_update = update_existing if update_existing is not None else self._update_existing

        # Add signature to comment
        full_comment = f"{COMMENT_SIGNATURE}\n{comment}"

        async with httpx.AsyncClient() as client:
            # Check for existing note if update mode
            existing_note_id = None
            if should_update:
                existing_note_id = await self._find_gitlab_note(
                    client, project, mr_iid
                )

            if existing_note_id:
                # Update existing note
                return await self._update_gitlab_note(
                    client, project, mr_iid, existing_note_id, full_comment
                )
            else:
                # Create new note
                return await self._create_gitlab_note(
                    client, project, mr_iid, full_comment
                )

    async def update_comment(
        self,
        comment_id: str,
        new_content: str,
    ) -> CommentResult:
        """
        Update an existing comment by its ID.

        Args:
            comment_id: Comment identifier in format "platform:repo:id"
            new_content: New comment content

        Returns:
            CommentResult with success status
        """
        parts = comment_id.split(":")
        if len(parts) < 3:
            return CommentResult(
                success=False,
                error=f"Invalid comment ID format: {comment_id}",
            )

        platform = parts[0]
        repo_or_project = ":".join(parts[1:-1])
        numeric_id = parts[-1]

        # Add signature to content
        full_content = f"{COMMENT_SIGNATURE}\n{new_content}"

        async with httpx.AsyncClient() as client:
            if platform == "github":
                return await self._update_github_comment(
                    client, repo_or_project, int(numeric_id), full_content
                )
            elif platform == "gitlab":
                # For GitLab, we need both project and MR IID
                # This is a simplified case - real implementation would need more context
                self.log.warning(
                    "GitLab comment update requires MR context",
                    comment_id=comment_id,
                )
                return CommentResult(
                    success=False,
                    error="GitLab comment update requires MR context",
                )
            else:
                return CommentResult(
                    success=False,
                    error=f"Unknown platform: {platform}",
                )

    async def delete_comment(
        self,
        comment_id: str,
    ) -> bool:
        """
        Delete a comment by its ID.

        Args:
            comment_id: Comment identifier in format "platform:repo:id"

        Returns:
            True if deleted successfully
        """
        parts = comment_id.split(":")
        if len(parts) < 3:
            self.log.warning("Invalid comment ID format", comment_id=comment_id)
            return False

        platform = parts[0]
        repo_or_project = ":".join(parts[1:-1])
        numeric_id = parts[-1]

        async with httpx.AsyncClient() as client:
            if platform == "github":
                url = f"https://api.github.com/repos/{repo_or_project}/issues/comments/{numeric_id}"
                try:
                    response = await client.delete(
                        url,
                        headers=self.github_headers,
                        timeout=30.0,
                    )
                    return response.status_code == 204
                except httpx.HTTPError as e:
                    self.log.error("Failed to delete GitHub comment", error=str(e))
                    return False
            else:
                self.log.warning("Delete not implemented for platform", platform=platform)
                return False

    async def _find_github_comment(
        self,
        client: httpx.AsyncClient,
        repo: str,
        pr_number: int,
    ) -> int | None:
        """Find an existing comment by our signature."""
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"

        try:
            response = await client.get(
                url,
                headers=self.github_headers,
                params={"per_page": 100},
                timeout=30.0,
            )
            response.raise_for_status()

            for comment in response.json():
                body = comment.get("body", "")
                if COMMENT_SIGNATURE in body:
                    return comment["id"]

        except httpx.HTTPError as e:
            self.log.warning(
                "Failed to search GitHub comments",
                repo=repo,
                pr=pr_number,
                error=str(e),
            )

        return None

    async def _create_github_comment(
        self,
        client: httpx.AsyncClient,
        repo: str,
        pr_number: int,
        body: str,
    ) -> CommentResult:
        """Create a new GitHub PR comment."""
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"

        try:
            response = await client.post(
                url,
                headers=self.github_headers,
                json={"body": body},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            self.log.info(
                "Created GitHub comment",
                repo=repo,
                pr=pr_number,
                comment_id=data["id"],
            )

            return CommentResult(
                success=True,
                comment_id=f"github:{repo}:{data['id']}",
                url=data.get("html_url"),
                was_updated=False,
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            self.log.error(
                "Failed to create GitHub comment",
                repo=repo,
                pr=pr_number,
                error=error_msg,
            )
            return CommentResult(success=False, error=error_msg)

        except httpx.HTTPError as e:
            self.log.error(
                "Failed to create GitHub comment",
                repo=repo,
                pr=pr_number,
                error=str(e),
            )
            return CommentResult(success=False, error=str(e))

    async def _update_github_comment(
        self,
        client: httpx.AsyncClient,
        repo: str,
        comment_id: int,
        body: str,
    ) -> CommentResult:
        """Update an existing GitHub comment."""
        url = f"https://api.github.com/repos/{repo}/issues/comments/{comment_id}"

        try:
            response = await client.patch(
                url,
                headers=self.github_headers,
                json={"body": body},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            self.log.info(
                "Updated GitHub comment",
                repo=repo,
                comment_id=comment_id,
            )

            return CommentResult(
                success=True,
                comment_id=f"github:{repo}:{comment_id}",
                url=data.get("html_url"),
                was_updated=True,
            )

        except httpx.HTTPError as e:
            self.log.error(
                "Failed to update GitHub comment",
                repo=repo,
                comment_id=comment_id,
                error=str(e),
            )
            return CommentResult(success=False, error=str(e))

    async def _find_gitlab_note(
        self,
        client: httpx.AsyncClient,
        project: str,
        mr_iid: int,
    ) -> int | None:
        """Find an existing note by our signature."""
        encoded_project = urllib.parse.quote(project, safe="")
        url = f"{self._gitlab_url}/api/v4/projects/{encoded_project}/merge_requests/{mr_iid}/notes"

        try:
            response = await client.get(
                url,
                headers=self.gitlab_headers,
                params={"per_page": 100},
                timeout=30.0,
            )
            response.raise_for_status()

            for note in response.json():
                body = note.get("body", "")
                if COMMENT_SIGNATURE in body and not note.get("system", False):
                    return note["id"]

        except httpx.HTTPError as e:
            self.log.warning(
                "Failed to search GitLab notes",
                project=project,
                mr=mr_iid,
                error=str(e),
            )

        return None

    async def _create_gitlab_note(
        self,
        client: httpx.AsyncClient,
        project: str,
        mr_iid: int,
        body: str,
    ) -> CommentResult:
        """Create a new GitLab MR note."""
        encoded_project = urllib.parse.quote(project, safe="")
        url = f"{self._gitlab_url}/api/v4/projects/{encoded_project}/merge_requests/{mr_iid}/notes"

        try:
            response = await client.post(
                url,
                headers=self.gitlab_headers,
                json={"body": body},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            self.log.info(
                "Created GitLab note",
                project=project,
                mr=mr_iid,
                note_id=data["id"],
            )

            # Construct URL to the note
            note_url = f"{self._gitlab_url}/{project}/-/merge_requests/{mr_iid}#note_{data['id']}"

            return CommentResult(
                success=True,
                comment_id=f"gitlab:{project}:{mr_iid}:{data['id']}",
                url=note_url,
                was_updated=False,
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            self.log.error(
                "Failed to create GitLab note",
                project=project,
                mr=mr_iid,
                error=error_msg,
            )
            return CommentResult(success=False, error=error_msg)

        except httpx.HTTPError as e:
            self.log.error(
                "Failed to create GitLab note",
                project=project,
                mr=mr_iid,
                error=str(e),
            )
            return CommentResult(success=False, error=str(e))

    async def _update_gitlab_note(
        self,
        client: httpx.AsyncClient,
        project: str,
        mr_iid: int,
        note_id: int,
        body: str,
    ) -> CommentResult:
        """Update an existing GitLab note."""
        encoded_project = urllib.parse.quote(project, safe="")
        url = f"{self._gitlab_url}/api/v4/projects/{encoded_project}/merge_requests/{mr_iid}/notes/{note_id}"

        try:
            response = await client.put(
                url,
                headers=self.gitlab_headers,
                json={"body": body},
                timeout=30.0,
            )
            response.raise_for_status()

            self.log.info(
                "Updated GitLab note",
                project=project,
                mr=mr_iid,
                note_id=note_id,
            )

            note_url = f"{self._gitlab_url}/{project}/-/merge_requests/{mr_iid}#note_{note_id}"

            return CommentResult(
                success=True,
                comment_id=f"gitlab:{project}:{mr_iid}:{note_id}",
                url=note_url,
                was_updated=True,
            )

        except httpx.HTTPError as e:
            self.log.error(
                "Failed to update GitLab note",
                project=project,
                mr=mr_iid,
                note_id=note_id,
                error=str(e),
            )
            return CommentResult(success=False, error=str(e))


def create_comment_poster(
    github_token: str | None = None,
    gitlab_token: str | None = None,
    gitlab_url: str = "https://gitlab.com",
) -> CommentPoster:
    """Factory function to create a CommentPoster."""
    return CommentPoster(
        github_token=github_token,
        gitlab_token=gitlab_token,
        gitlab_url=gitlab_url,
    )
