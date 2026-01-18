"""Tests for GitHub integration module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import httpx


class TestTestSummary:
    """Tests for TestSummary dataclass."""

    def test_test_summary_creation(self, mock_env_vars):
        """Test creating TestSummary."""
        from src.integrations.github_integration import TestSummary

        summary = TestSummary(
            total=100,
            passed=95,
            failed=3,
            skipped=2,
            duration_seconds=120.5,
            cost_usd=0.5,
            failures=[{"test_id": "test-1", "error_message": "Failed"}],
            screenshots=["https://example.com/screenshot.png"],
        )

        assert summary.total == 100
        assert summary.passed == 95
        assert summary.failed == 3
        assert summary.skipped == 2
        assert summary.duration_seconds == 120.5
        assert summary.cost_usd == 0.5
        assert len(summary.failures) == 1


class TestGitHubIntegrationInit:
    """Tests for GitHubIntegration initialization."""

    def test_initialization_with_token(self, mock_env_vars):
        """Test initialization with token."""
        from src.integrations.github_integration import GitHubIntegration

        github = GitHubIntegration(token="ghp_test_token")

        assert github.token == "ghp_test_token"
        assert github.api_base == "https://api.github.com"

    def test_initialization_from_env(self, mock_env_vars, monkeypatch):
        """Test initialization from environment variable."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_env_token")

        from src.integrations.github_integration import GitHubIntegration

        github = GitHubIntegration()

        assert github.token == "ghp_env_token"

    def test_initialization_custom_api_base(self, mock_env_vars):
        """Test initialization with custom API base."""
        from src.integrations.github_integration import GitHubIntegration

        github = GitHubIntegration(
            token="ghp_test",
            api_base="https://github.mycompany.com/api/v3",
        )

        assert github.api_base == "https://github.mycompany.com/api/v3"

    def test_initialization_no_token(self, mock_env_vars, monkeypatch):
        """Test initialization without token."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        from src.integrations.github_integration import GitHubIntegration

        github = GitHubIntegration()

        assert github.token is None


class TestGitHubIntegrationHeaders:
    """Tests for GitHubIntegration headers property."""

    def test_headers_with_token(self, mock_env_vars):
        """Test headers include authorization."""
        from src.integrations.github_integration import GitHubIntegration

        github = GitHubIntegration(token="ghp_test_token")

        headers = github.headers

        assert headers["Authorization"] == "Bearer ghp_test_token"
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["X-GitHub-Api-Version"] == "2022-11-28"


class TestGitHubIntegrationFormatComment:
    """Tests for GitHubIntegration._format_comment method."""

    @pytest.fixture
    def github(self, mock_env_vars):
        """Create GitHubIntegration instance."""
        from src.integrations.github_integration import GitHubIntegration

        return GitHubIntegration(token="ghp_test_token")

    def test_format_comment_success(self, github):
        """Test formatting comment for successful test run."""
        from src.integrations.github_integration import TestSummary

        summary = TestSummary(
            total=10,
            passed=10,
            failed=0,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[],
            screenshots=[],
        )

        comment = github._format_comment(summary)

        assert "## " in comment  # Header
        assert "10" in comment  # Total tests
        assert "100.0%" in comment  # Pass rate
        assert "30.5s" in comment  # Duration
        assert "$0.0500" in comment  # Cost
        assert "E2E Testing Agent" in comment  # Footer

    def test_format_comment_with_failures(self, github):
        """Test formatting comment with failures."""
        from src.integrations.github_integration import TestSummary

        summary = TestSummary(
            total=10,
            passed=8,
            failed=2,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[
                {"test_id": "test-1", "error_message": "Element not found"},
                {"test_id": "test-2", "error_message": "Timeout", "root_cause": "Slow network"},
            ],
            screenshots=[],
        )

        comment = github._format_comment(summary)

        assert "Failed Tests" in comment
        assert "test-1" in comment
        assert "Element not found" in comment
        assert "test-2" in comment
        assert "Root Cause" in comment
        assert "Slow network" in comment

    def test_format_comment_with_screenshots(self, github):
        """Test formatting comment with screenshots."""
        from src.integrations.github_integration import TestSummary

        summary = TestSummary(
            total=10,
            passed=9,
            failed=1,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[],
            screenshots=[
                "https://example.com/screenshot1.png",
                "https://example.com/screenshot2.png",
            ],
        )

        comment = github._format_comment(summary)

        assert "Failure Screenshots" in comment
        assert "screenshot1.png" in comment
        assert "screenshot2.png" in comment

    def test_format_comment_many_failures(self, github):
        """Test formatting comment truncates many failures."""
        from src.integrations.github_integration import TestSummary

        failures = [
            {"test_id": f"test-{i}", "error_message": f"Error {i}"}
            for i in range(15)
        ]

        summary = TestSummary(
            total=15,
            passed=0,
            failed=15,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=failures,
            screenshots=[],
        )

        comment = github._format_comment(summary)

        # Should show first 10 and mention remaining
        assert "test-9" in comment
        assert "5 more failures" in comment

    def test_format_comment_custom_title(self, github):
        """Test formatting comment with custom title."""
        from src.integrations.github_integration import TestSummary

        summary = TestSummary(
            total=5,
            passed=5,
            failed=0,
            skipped=0,
            duration_seconds=10.0,
            cost_usd=0.01,
            failures=[],
            screenshots=[],
        )

        comment = github._format_comment(summary, title="Custom Test Results")

        assert "Custom Test Results" in comment


class TestGitHubIntegrationFormatCheckOutput:
    """Tests for GitHubIntegration._format_check_output method."""

    @pytest.fixture
    def github(self, mock_env_vars):
        """Create GitHubIntegration instance."""
        from src.integrations.github_integration import GitHubIntegration

        return GitHubIntegration(token="ghp_test_token")

    def test_format_check_output_success(self, github):
        """Test formatting check output for success."""
        from src.integrations.github_integration import TestSummary

        summary = TestSummary(
            total=10,
            passed=10,
            failed=0,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[],
            screenshots=[],
        )

        output = github._format_check_output(summary)

        assert "title" in output
        assert "summary" in output
        assert "text" in output
        assert "10/10" in output["title"]

    def test_format_check_output_failure(self, github):
        """Test formatting check output with failures."""
        from src.integrations.github_integration import TestSummary

        summary = TestSummary(
            total=10,
            passed=8,
            failed=2,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[
                {"test_id": "test-1", "error_message": "Error 1", "root_cause": "Network issue"},
            ],
            screenshots=[],
        )

        output = github._format_check_output(summary)

        assert "Failures" in output["text"]
        assert "test-1" in output["text"]
        assert "Network issue" in output["text"]


class TestGitHubIntegrationPostPRComment:
    """Tests for GitHubIntegration.post_pr_comment method."""

    @pytest.fixture
    def github(self, mock_env_vars):
        """Create GitHubIntegration instance."""
        from src.integrations.github_integration import GitHubIntegration

        return GitHubIntegration(token="ghp_test_token")

    @pytest.fixture
    def test_summary(self, mock_env_vars):
        """Create test summary."""
        from src.integrations.github_integration import TestSummary

        return TestSummary(
            total=10,
            passed=10,
            failed=0,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[],
            screenshots=[],
        )

    @pytest.mark.asyncio
    async def test_post_pr_comment_success(self, github, test_summary):
        """Test posting PR comment successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 123}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value=[]),
            ))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await github.post_pr_comment(
                owner="testowner",
                repo="testrepo",
                pr_number=123,
                summary=test_summary,
            )

            assert result is not None
            assert result["id"] == 123

    @pytest.mark.asyncio
    async def test_post_pr_comment_update_existing(self, github, test_summary):
        """Test updating existing PR comment."""
        # Existing comment
        existing_comments = [
            {"id": 456, "body": "<!-- e2e-testing-agent -->\nOld results"},
        ]

        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = existing_comments
        mock_get_response.raise_for_status = MagicMock()

        mock_patch_response = MagicMock()
        mock_patch_response.status_code = 200
        mock_patch_response.json.return_value = {"id": 456}
        mock_patch_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_get_response)
            mock_client.patch = AsyncMock(return_value=mock_patch_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await github.post_pr_comment(
                owner="testowner",
                repo="testrepo",
                pr_number=123,
                summary=test_summary,
                update_existing=True,
            )

            assert result is not None
            assert result["id"] == 456
            mock_client.patch.assert_called()

    @pytest.mark.asyncio
    async def test_post_pr_comment_no_token(self, mock_env_vars, test_summary, monkeypatch):
        """Test post_pr_comment when no token."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        from src.integrations.github_integration import GitHubIntegration

        github = GitHubIntegration()

        result = await github.post_pr_comment(
            owner="testowner",
            repo="testrepo",
            pr_number=123,
            summary=test_summary,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_post_pr_comment_failure(self, github, test_summary):
        """Test post_pr_comment handles failure."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value=[]),
                raise_for_status=MagicMock(),
            ))
            mock_client.post = AsyncMock(side_effect=Exception("API error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await github.post_pr_comment(
                owner="testowner",
                repo="testrepo",
                pr_number=123,
                summary=test_summary,
            )

            assert result is None


class TestGitHubIntegrationFindExistingComment:
    """Tests for GitHubIntegration._find_existing_comment method."""

    @pytest.fixture
    def github(self, mock_env_vars):
        """Create GitHubIntegration instance."""
        from src.integrations.github_integration import GitHubIntegration

        return GitHubIntegration(token="ghp_test_token")

    @pytest.mark.asyncio
    async def test_find_existing_comment_found(self, github):
        """Test finding existing comment."""
        comments = [
            {"id": 1, "body": "Some other comment"},
            {"id": 2, "body": "<!-- e2e-testing-agent -->\nTest results"},
            {"id": 3, "body": "Another comment"},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = comments
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await github._find_existing_comment(
            client=mock_client,
            owner="testowner",
            repo="testrepo",
            pr_number=123,
            signature="<!-- e2e-testing-agent -->",
        )

        assert result is not None
        assert result["id"] == 2

    @pytest.mark.asyncio
    async def test_find_existing_comment_not_found(self, github):
        """Test finding existing comment when not present."""
        comments = [
            {"id": 1, "body": "Some other comment"},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = comments
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await github._find_existing_comment(
            client=mock_client,
            owner="testowner",
            repo="testrepo",
            pr_number=123,
            signature="<!-- e2e-testing-agent -->",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_find_existing_comment_error(self, github):
        """Test finding existing comment handles error."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("API error"))

        result = await github._find_existing_comment(
            client=mock_client,
            owner="testowner",
            repo="testrepo",
            pr_number=123,
            signature="<!-- e2e-testing-agent -->",
        )

        assert result is None


class TestGitHubIntegrationCreateCheckRun:
    """Tests for GitHubIntegration.create_check_run method."""

    @pytest.fixture
    def github(self, mock_env_vars):
        """Create GitHubIntegration instance."""
        from src.integrations.github_integration import GitHubIntegration

        return GitHubIntegration(token="ghp_test_token")

    @pytest.fixture
    def test_summary(self, mock_env_vars):
        """Create test summary."""
        from src.integrations.github_integration import TestSummary

        return TestSummary(
            total=10,
            passed=10,
            failed=0,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[],
            screenshots=[],
        )

    @pytest.mark.asyncio
    async def test_create_check_run_success(self, github, test_summary):
        """Test creating check run successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 789}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await github.create_check_run(
                owner="testowner",
                repo="testrepo",
                sha="abc123def456",
                summary=test_summary,
            )

            assert result is not None
            assert result["id"] == 789

    @pytest.mark.asyncio
    async def test_create_check_run_failure(self, github, test_summary):
        """Test creating check run with failed tests."""
        from src.integrations.github_integration import TestSummary

        failed_summary = TestSummary(
            total=10,
            passed=8,
            failed=2,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[{"test_id": "test-1", "error_message": "Error"}],
            screenshots=[],
        )

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 789, "conclusion": "failure"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await github.create_check_run(
                owner="testowner",
                repo="testrepo",
                sha="abc123def456",
                summary=failed_summary,
            )

            # Verify conclusion is failure
            call_args = mock_client.post.call_args
            assert call_args.kwargs["json"]["conclusion"] == "failure"

    @pytest.mark.asyncio
    async def test_create_check_run_no_token(self, mock_env_vars, test_summary, monkeypatch):
        """Test create_check_run when no token."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        from src.integrations.github_integration import GitHubIntegration

        github = GitHubIntegration()

        result = await github.create_check_run(
            owner="testowner",
            repo="testrepo",
            sha="abc123",
            summary=test_summary,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_create_check_run_custom_name(self, github, test_summary):
        """Test creating check run with custom name."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 789}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await github.create_check_run(
                owner="testowner",
                repo="testrepo",
                sha="abc123",
                summary=test_summary,
                name="Custom E2E Tests",
            )

            call_args = mock_client.post.call_args
            assert call_args.kwargs["json"]["name"] == "Custom E2E Tests"


class TestGitHubIntegrationUpdateCommitStatus:
    """Tests for GitHubIntegration.update_commit_status method."""

    @pytest.fixture
    def github(self, mock_env_vars):
        """Create GitHubIntegration instance."""
        from src.integrations.github_integration import GitHubIntegration

        return GitHubIntegration(token="ghp_test_token")

    @pytest.mark.asyncio
    async def test_update_commit_status_success(self, github):
        """Test updating commit status."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 999, "state": "success"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await github.update_commit_status(
                owner="testowner",
                repo="testrepo",
                sha="abc123",
                state="success",
                description="All tests passed",
            )

            assert result is not None
            assert result["state"] == "success"

    @pytest.mark.asyncio
    async def test_update_commit_status_with_target_url(self, github):
        """Test updating commit status with target URL."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 999}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await github.update_commit_status(
                owner="testowner",
                repo="testrepo",
                sha="abc123",
                state="success",
                description="All tests passed",
                target_url="https://example.com/report",
            )

            call_args = mock_client.post.call_args
            assert call_args.kwargs["json"]["target_url"] == "https://example.com/report"

    @pytest.mark.asyncio
    async def test_update_commit_status_truncates_description(self, github):
        """Test description is truncated to GitHub's limit."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 999}
        mock_response.raise_for_status = MagicMock()

        long_description = "x" * 200  # Longer than 140 char limit

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await github.update_commit_status(
                owner="testowner",
                repo="testrepo",
                sha="abc123",
                state="success",
                description=long_description,
            )

            call_args = mock_client.post.call_args
            assert len(call_args.kwargs["json"]["description"]) <= 140

    @pytest.mark.asyncio
    async def test_update_commit_status_no_token(self, mock_env_vars, monkeypatch):
        """Test update_commit_status when no token."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        from src.integrations.github_integration import GitHubIntegration

        github = GitHubIntegration()

        result = await github.update_commit_status(
            owner="testowner",
            repo="testrepo",
            sha="abc123",
            state="success",
            description="Test",
        )

        assert result is None


class TestGitHubIntegrationGetPRFiles:
    """Tests for GitHubIntegration.get_pr_files method."""

    @pytest.fixture
    def github(self, mock_env_vars):
        """Create GitHubIntegration instance."""
        from src.integrations.github_integration import GitHubIntegration

        return GitHubIntegration(token="ghp_test_token")

    @pytest.mark.asyncio
    async def test_get_pr_files_success(self, github):
        """Test getting PR files."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"filename": "src/app.py"},
            {"filename": "tests/test_app.py"},
            {"filename": "README.md"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await github.get_pr_files(
                owner="testowner",
                repo="testrepo",
                pr_number=123,
            )

            assert len(result) == 3
            assert "src/app.py" in result
            assert "tests/test_app.py" in result

    @pytest.mark.asyncio
    async def test_get_pr_files_no_token(self, mock_env_vars, monkeypatch):
        """Test get_pr_files when no token."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        from src.integrations.github_integration import GitHubIntegration

        github = GitHubIntegration()

        result = await github.get_pr_files(
            owner="testowner",
            repo="testrepo",
            pr_number=123,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_get_pr_files_error(self, github):
        """Test get_pr_files handles error."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("API error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await github.get_pr_files(
                owner="testowner",
                repo="testrepo",
                pr_number=123,
            )

            assert result == []


class TestCreateGitHubIntegration:
    """Tests for create_github_integration factory function."""

    def test_create_with_token(self, mock_env_vars):
        """Test creating integration with token."""
        from src.integrations.github_integration import create_github_integration

        github = create_github_integration(token="ghp_test")

        assert github.token == "ghp_test"

    def test_create_without_token(self, mock_env_vars, monkeypatch):
        """Test creating integration without token."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        from src.integrations.github_integration import create_github_integration

        github = create_github_integration()

        assert github.token is None
