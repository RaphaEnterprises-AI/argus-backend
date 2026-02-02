"""Tests for the CI/CD Monitor service.

This module tests:
- CICDMonitor initialization and configuration
- GitHub PR polling
- GitLab MR polling
- Project management (add/remove)
- Analysis triggering
- Datetime parsing utilities
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services.cicd_monitor import (
    CICDMonitor,
    FileChange,
    MRInfo,
    MonitoredProject,
    PRInfo,
    PRState,
    VCSPlatform,
    get_cicd_monitor,
)


class TestDataClasses:
    """Tests for dataclasses used by the monitor."""

    def test_file_change_creation(self):
        """Test FileChange dataclass creation."""
        change = FileChange(
            path="src/auth.py",
            status="modified",
            additions=10,
            deletions=5,
            patch="@@ -1,5 +1,10 @@",
            previous_path=None,
        )
        assert change.path == "src/auth.py"
        assert change.status == "modified"
        assert change.additions == 10
        assert change.deletions == 5
        assert change.patch == "@@ -1,5 +1,10 @@"

    def test_file_change_defaults(self):
        """Test FileChange default values."""
        change = FileChange(path="test.py", status="added")
        assert change.additions == 0
        assert change.deletions == 0
        assert change.patch is None
        assert change.previous_path is None

    def test_pr_info_identifier(self):
        """Test PRInfo identifier property."""
        pr = PRInfo(
            repo="org/repo",
            number=123,
            title="Test PR",
        )
        assert pr.identifier == "github:org/repo:123"

    def test_pr_info_defaults(self):
        """Test PRInfo default values."""
        pr = PRInfo()
        assert pr.platform == VCSPlatform.GITHUB
        assert pr.repo == ""
        assert pr.number == 0
        assert pr.title == ""
        assert pr.state == PRState.OPEN
        assert pr.files == []
        assert pr.labels == []
        assert pr.is_draft is False

    def test_mr_info_identifier(self):
        """Test MRInfo identifier property."""
        mr = MRInfo(
            project_path="namespace/project",
            iid=456,
            title="Test MR",
        )
        assert mr.identifier == "gitlab:namespace/project:456"

    def test_mr_info_defaults(self):
        """Test MRInfo default values."""
        mr = MRInfo()
        assert mr.platform == VCSPlatform.GITLAB
        assert mr.project_path == ""
        assert mr.iid == 0
        assert mr.state == PRState.OPEN
        assert mr.files == []
        assert mr.labels == []

    def test_monitored_project_defaults(self):
        """Test MonitoredProject default values."""
        project = MonitoredProject(
            project_id="proj_123",
            org_id="org_456",
            platform=VCSPlatform.GITHUB,
            repo_identifier="org/repo",
        )
        assert project.enabled is True
        assert project.last_poll_at is None
        assert project.poll_interval_seconds == 60
        assert project.auto_comment is True
        assert project.analysis_settings == {}


class TestCICDMonitorInit:
    """Tests for CICDMonitor initialization."""

    def test_init_default_values(self, mock_env_vars, monkeypatch):
        """Test initialization with default values."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        assert monitor.poll_interval == 60
        assert monitor._max_concurrent == 5
        assert monitor._running is False
        assert monitor._projects == {}
        assert monitor._processed == set()

    def test_init_with_custom_values(self, mock_env_vars, monkeypatch):
        """Test initialization with custom values."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor(
            poll_interval=120,
            github_token="gh_token_123",
            gitlab_token="gl_token_456",
            gitlab_url="https://gitlab.custom.com/",
            max_concurrent_analyses=10,
        )
        assert monitor.poll_interval == 120
        assert monitor._github_token == "gh_token_123"
        assert monitor._gitlab_token == "gl_token_456"
        assert monitor._gitlab_url == "https://gitlab.custom.com"
        assert monitor._max_concurrent == 10

    def test_init_from_env_vars(self, mock_env_vars, monkeypatch):
        """Test initialization from environment variables."""
        monkeypatch.setenv("GITHUB_TOKEN", "env_gh_token")
        monkeypatch.setenv("GITLAB_TOKEN", "env_gl_token")

        monitor = CICDMonitor()
        assert monitor._github_token == "env_gh_token"
        assert monitor._gitlab_token == "env_gl_token"


class TestCICDMonitorHeaders:
    """Tests for API header generation."""

    def test_github_headers_with_token(self, mock_env_vars, monkeypatch):
        """Test GitHub headers with token."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor(github_token="test_token")
        headers = monitor.github_headers

        assert headers["Authorization"] == "Bearer test_token"
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["X-GitHub-Api-Version"] == "2022-11-28"

    def test_github_headers_without_token(self, mock_env_vars, monkeypatch):
        """Test GitHub headers without token."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        headers = monitor.github_headers

        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github.v3+json"

    def test_gitlab_headers_with_token(self, mock_env_vars, monkeypatch):
        """Test GitLab headers with token."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor(gitlab_token="test_token")
        headers = monitor.gitlab_headers

        assert headers["PRIVATE-TOKEN"] == "test_token"
        assert headers["Accept"] == "application/json"

    def test_gitlab_headers_without_token(self, mock_env_vars, monkeypatch):
        """Test GitLab headers without token."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        headers = monitor.gitlab_headers

        assert "PRIVATE-TOKEN" not in headers
        assert headers["Accept"] == "application/json"


class TestCICDMonitorStartStop:
    """Tests for monitor start/stop functionality."""

    @pytest.mark.asyncio
    async def test_start_monitor(self, mock_env_vars, monkeypatch):
        """Test starting the monitor."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        await monitor.start()

        assert monitor._running is True
        assert monitor._poll_task is not None

        # Clean up
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_start_already_running(self, mock_env_vars, monkeypatch):
        """Test starting when already running."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        await monitor.start()
        await monitor.start()  # Should log warning but not fail

        assert monitor._running is True

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_monitor(self, mock_env_vars, monkeypatch):
        """Test stopping the monitor."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        await monitor.start()
        await monitor.stop()

        assert monitor._running is False
        assert monitor._poll_task is None

    @pytest.mark.asyncio
    async def test_stop_not_running(self, mock_env_vars, monkeypatch):
        """Test stopping when not running."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        await monitor.stop()  # Should log warning but not fail

        assert monitor._running is False


class TestCICDMonitorProjectManagement:
    """Tests for project management."""

    @pytest.mark.asyncio
    async def test_add_project(self, mock_env_vars, monkeypatch):
        """Test adding a project to monitor."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        project = await monitor.add_project(
            project_id="proj_123",
            org_id="org_456",
            platform=VCSPlatform.GITHUB,
            repo="myorg/myapp",
            poll_interval=120,
            auto_comment=False,
            analysis_settings={"max_suggestions": 5},
        )

        assert project.project_id == "proj_123"
        assert project.org_id == "org_456"
        assert project.platform == VCSPlatform.GITHUB
        assert project.repo_identifier == "myorg/myapp"
        assert project.poll_interval_seconds == 120
        assert project.auto_comment is False
        assert project.analysis_settings == {"max_suggestions": 5}
        assert monitor.monitored_projects_count == 1

    @pytest.mark.asyncio
    async def test_add_project_default_poll_interval(self, mock_env_vars, monkeypatch):
        """Test adding a project uses default poll interval."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor(poll_interval=90)
        project = await monitor.add_project(
            project_id="proj_123",
            org_id="org_456",
            platform=VCSPlatform.GITHUB,
            repo="org/repo",
        )

        assert project.poll_interval_seconds == 90

    @pytest.mark.asyncio
    async def test_remove_project(self, mock_env_vars, monkeypatch):
        """Test removing a project from monitoring."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        await monitor.add_project(
            project_id="proj_123",
            org_id="org_456",
            platform=VCSPlatform.GITHUB,
            repo="org/repo",
        )

        result = await monitor.remove_project(VCSPlatform.GITHUB, "org/repo")

        assert result is True
        assert monitor.monitored_projects_count == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_project(self, mock_env_vars, monkeypatch):
        """Test removing a project that doesn't exist."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        result = await monitor.remove_project(VCSPlatform.GITHUB, "nonexistent/repo")

        assert result is False


class TestCICDMonitorGitHubPolling:
    """Tests for GitHub PR polling."""

    @pytest.mark.asyncio
    async def test_poll_github_no_token(self, mock_env_vars, monkeypatch):
        """Test polling GitHub without token."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        prs = await monitor.poll_github("proj_123", "org/repo")

        assert prs == []

    @pytest.mark.asyncio
    async def test_poll_github_success(self, mock_env_vars, monkeypatch):
        """Test successful GitHub PR polling."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        mock_pr_response = MagicMock()
        mock_pr_response.status_code = 200
        mock_pr_response.json.return_value = [
            {
                "number": 123,
                "title": "Test PR",
                "body": "PR description",
                "state": "open",
                "user": {"login": "testuser"},
                "head": {"sha": "abc123", "ref": "feature-branch"},
                "base": {"ref": "main"},
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T12:00:00Z",
                "labels": [{"name": "enhancement"}],
                "draft": False,
                "html_url": "https://github.com/org/repo/pull/123",
            }
        ]

        mock_files_response = MagicMock()
        mock_files_response.status_code = 200
        mock_files_response.json.return_value = [
            {
                "filename": "src/auth.py",
                "status": "modified",
                "additions": 10,
                "deletions": 5,
                "patch": "@@ -1,5 +1,10 @@",
            }
        ]

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[mock_pr_response, mock_files_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            monitor = CICDMonitor(github_token="test_token")
            prs = await monitor.poll_github("proj_123", "org/repo")

        assert len(prs) == 1
        assert prs[0].number == 123
        assert prs[0].title == "Test PR"
        assert prs[0].author == "testuser"
        assert len(prs[0].files) == 1
        assert prs[0].files[0].path == "src/auth.py"

    @pytest.mark.asyncio
    async def test_poll_github_http_error(self, mock_env_vars, monkeypatch):
        """Test GitHub polling with HTTP error."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            monitor = CICDMonitor(github_token="test_token")
            prs = await monitor.poll_github("proj_123", "org/repo")

        assert prs == []


class TestCICDMonitorGitLabPolling:
    """Tests for GitLab MR polling."""

    @pytest.mark.asyncio
    async def test_poll_gitlab_no_token(self, mock_env_vars, monkeypatch):
        """Test polling GitLab without token."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        mrs = await monitor.poll_gitlab("proj_123", "namespace/project")

        assert mrs == []

    @pytest.mark.asyncio
    async def test_poll_gitlab_success(self, mock_env_vars, monkeypatch):
        """Test successful GitLab MR polling."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        mock_mr_response = MagicMock()
        mock_mr_response.status_code = 200
        mock_mr_response.json.return_value = [
            {
                "iid": 456,
                "id": 12345,
                "title": "Test MR",
                "description": "MR description",
                "state": "opened",
                "author": {"username": "testuser"},
                "source_branch": "feature-branch",
                "target_branch": "main",
                "sha": "def456",
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T12:00:00Z",
                "labels": ["enhancement"],
                "work_in_progress": False,
                "draft": False,
                "web_url": "https://gitlab.com/namespace/project/-/merge_requests/456",
            }
        ]

        mock_changes_response = MagicMock()
        mock_changes_response.status_code = 200
        mock_changes_response.json.return_value = {
            "changes": [
                {
                    "new_path": "src/auth.py",
                    "old_path": "src/auth.py",
                    "new_file": False,
                    "deleted_file": False,
                    "renamed_file": False,
                    "diff": "@@ -1,5 +1,10 @@\n+added line\n-removed line",
                }
            ]
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[mock_mr_response, mock_changes_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            monitor = CICDMonitor(gitlab_token="test_token")
            mrs = await monitor.poll_gitlab("proj_123", "namespace/project")

        assert len(mrs) == 1
        assert mrs[0].iid == 456
        assert mrs[0].title == "Test MR"
        assert mrs[0].author == "testuser"
        assert len(mrs[0].files) == 1


class TestCICDMonitorAnalysis:
    """Tests for analysis triggering."""

    @pytest.mark.asyncio
    async def test_trigger_analysis_already_processed(self, mock_env_vars, monkeypatch):
        """Test triggering analysis for already processed PR."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        pr = PRInfo(repo="org/repo", number=123)
        monitor._processed.add(pr.identifier)

        result = await monitor.trigger_analysis(pr)

        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_analysis_no_project_config(self, mock_env_vars, monkeypatch):
        """Test triggering analysis without project config."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        pr = PRInfo(repo="org/repo", number=123)

        result = await monitor.trigger_analysis(pr)

        assert result is None


class TestCICDMonitorParsingUtils:
    """Tests for parsing utility methods."""

    def test_parse_datetime_iso(self, mock_env_vars, monkeypatch):
        """Test parsing ISO datetime."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        result = CICDMonitor._parse_datetime("2024-01-15T10:30:00+00:00")

        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_parse_datetime_with_z_suffix(self, mock_env_vars, monkeypatch):
        """Test parsing datetime with Z suffix."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        result = CICDMonitor._parse_datetime("2024-01-15T10:30:00Z")

        assert result is not None
        assert result.year == 2024

    def test_parse_datetime_none(self, mock_env_vars, monkeypatch):
        """Test parsing None datetime."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        result = CICDMonitor._parse_datetime(None)

        assert result is None

    def test_parse_datetime_invalid(self, mock_env_vars, monkeypatch):
        """Test parsing invalid datetime."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        result = CICDMonitor._parse_datetime("invalid-date")

        assert result is None

    def test_parse_github_pr_open(self, mock_env_vars, monkeypatch):
        """Test parsing open GitHub PR."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        data = {
            "number": 123,
            "title": "Test PR",
            "body": "Description",
            "state": "open",
            "user": {"login": "user"},
            "head": {"sha": "abc", "ref": "feature"},
            "base": {"ref": "main"},
            "labels": [],
            "draft": False,
            "html_url": "https://github.com/org/repo/pull/123",
        }

        pr = monitor._parse_github_pr("org/repo", data)

        assert pr.state == PRState.OPEN
        assert pr.is_draft is False

    def test_parse_github_pr_merged(self, mock_env_vars, monkeypatch):
        """Test parsing merged GitHub PR."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        data = {
            "number": 123,
            "title": "Test PR",
            "body": None,
            "state": "closed",
            "merged_at": "2024-01-15T12:00:00Z",
            "user": {"login": "user"},
            "head": {"sha": "abc", "ref": "feature"},
            "base": {"ref": "main"},
            "labels": [],
            "draft": False,
            "html_url": "https://github.com/org/repo/pull/123",
        }

        pr = monitor._parse_github_pr("org/repo", data)

        assert pr.state == PRState.MERGED

    def test_parse_github_pr_draft(self, mock_env_vars, monkeypatch):
        """Test parsing draft GitHub PR."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        data = {
            "number": 123,
            "title": "Test PR",
            "body": "",
            "state": "open",
            "user": {"login": "user"},
            "head": {"sha": "abc", "ref": "feature"},
            "base": {"ref": "main"},
            "labels": [{"name": "WIP"}],
            "draft": True,
            "html_url": "https://github.com/org/repo/pull/123",
        }

        pr = monitor._parse_github_pr("org/repo", data)

        assert pr.state == PRState.DRAFT
        assert pr.is_draft is True

    def test_parse_gitlab_mr_states(self, mock_env_vars, monkeypatch):
        """Test parsing GitLab MR states."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()

        # Test opened state
        data = {
            "iid": 1,
            "id": 100,
            "title": "Test",
            "description": "",
            "state": "opened",
            "author": {"username": "user"},
            "source_branch": "feature",
            "target_branch": "main",
            "sha": "abc",
            "labels": [],
        }
        mr = monitor._parse_gitlab_mr("ns/proj", data)
        assert mr.state == PRState.OPEN

        # Test merged state
        data["state"] = "merged"
        mr = monitor._parse_gitlab_mr("ns/proj", data)
        assert mr.state == PRState.MERGED

        # Test closed state
        data["state"] = "closed"
        mr = monitor._parse_gitlab_mr("ns/proj", data)
        assert mr.state == PRState.CLOSED

        # Test draft state
        data["state"] = "opened"
        data["work_in_progress"] = True
        mr = monitor._parse_gitlab_mr("ns/proj", data)
        assert mr.state == PRState.DRAFT


class TestCICDMonitorProperties:
    """Tests for monitor properties."""

    def test_is_running(self, mock_env_vars, monkeypatch):
        """Test is_running property."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        assert monitor.is_running is False

        monitor._running = True
        assert monitor.is_running is True

    def test_processed_count(self, mock_env_vars, monkeypatch):
        """Test processed_count property."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        monitor = CICDMonitor()
        assert monitor.processed_count == 0

        monitor._processed.add("test1")
        monitor._processed.add("test2")
        assert monitor.processed_count == 2


class TestGlobalFunctions:
    """Tests for module-level functions."""

    def test_get_cicd_monitor_singleton(self, mock_env_vars, monkeypatch):
        """Test get_cicd_monitor returns singleton."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)

        import src.services.cicd_monitor as module
        module._monitor = None

        monitor1 = get_cicd_monitor()
        monitor2 = get_cicd_monitor()

        assert monitor1 is monitor2

        # Clean up
        module._monitor = None
