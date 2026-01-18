"""Tests for integration modules (GitHub, Slack, Reporter)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================================
# GitHub Integration Tests
# ============================================================================

class TestGitHubIntegration:
    """Tests for GitHubIntegration class."""

    @pytest.fixture
    def github(self):
        """Create GitHubIntegration with mocked token."""
        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test_token"}):
            from src.integrations.github_integration import GitHubIntegration
            return GitHubIntegration(token="ghp_test_token")

    @pytest.fixture
    def test_summary(self):
        """Create a test summary for testing."""
        from src.integrations.github_integration import TestSummary
        return TestSummary(
            total=10,
            passed=8,
            failed=2,
            skipped=0,
            duration_seconds=45.5,
            cost_usd=0.25,
            failures=[
                {
                    "test_id": "login-test",
                    "error_message": "Element not found: #submit",
                    "root_cause": "Button ID changed",
                }
            ],
            screenshots=["https://example.com/screenshot1.png"],
        )

    def test_init_with_token(self, github):
        """Test initialization with token."""
        assert github.token == "ghp_test_token"
        assert github.api_base == "https://api.github.com"

    def test_init_without_token(self):
        """Test initialization without token logs warning."""
        with patch.dict("os.environ", {}, clear=True):
            from src.integrations.github_integration import GitHubIntegration
            gh = GitHubIntegration(token=None)
            assert gh.token is None

    def test_headers(self, github):
        """Test headers property."""
        headers = github.headers
        assert "Authorization" in headers
        assert "Bearer ghp_test_token" in headers["Authorization"]
        assert headers["Accept"] == "application/vnd.github.v3+json"

    def test_format_comment(self, github, test_summary):
        """Test formatting PR comment."""
        comment = github._format_comment(test_summary)

        assert "E2E Test Results" in comment
        assert "10" in comment  # total
        assert "8" in comment  # passed
        assert "2" in comment  # failed
        assert "80.0%" in comment  # pass rate
        assert "login-test" in comment  # failure

    def test_format_check_output(self, github, test_summary):
        """Test formatting check run output."""
        output = github._format_check_output(test_summary)

        assert "title" in output
        assert "summary" in output
        assert "text" in output
        assert "8/10" in output["title"]

    @pytest.mark.asyncio
    async def test_post_pr_comment(self, github, test_summary):
        """Test posting PR comment."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.json.return_value = {"id": 123, "html_url": "https://..."}
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.get = AsyncMock(return_value=MagicMock(
                json=MagicMock(return_value=[]),
                raise_for_status=MagicMock()
            ))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_instance

            result = await github.post_pr_comment(
                owner="testorg",
                repo="testrepo",
                pr_number=42,
                summary=test_summary,
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_create_check_run(self, github, test_summary):
        """Test creating check run."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.json.return_value = {"id": 456}
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_instance

            result = await github.create_check_run(
                owner="testorg",
                repo="testrepo",
                sha="abc123def456",
                summary=test_summary,
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_update_commit_status(self, github):
        """Test updating commit status."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.json.return_value = {"id": 789}
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_instance

            result = await github.update_commit_status(
                owner="testorg",
                repo="testrepo",
                sha="abc123",
                state="success",
                description="All tests passed",
            )

            assert result is not None


# ============================================================================
# Slack Integration Tests
# ============================================================================

class TestSlackIntegration:
    """Tests for SlackIntegration class."""

    @pytest.fixture
    def slack(self):
        """Create SlackIntegration with mocked webhook."""
        from src.integrations.slack_integration import SlackIntegration
        return SlackIntegration(webhook_url="https://hooks.slack.com/test")

    @pytest.fixture
    def test_summary(self):
        """Create a test summary for testing."""
        from src.integrations.slack_integration import TestSummary
        return TestSummary(
            total=10,
            passed=8,
            failed=2,
            skipped=0,
            duration_seconds=45.5,
            cost_usd=0.25,
            failures=[
                {
                    "test_id": "login-test",
                    "error_message": "Element not found",
                }
            ],
        )

    def test_init_with_webhook(self, slack):
        """Test initialization with webhook URL."""
        assert slack.webhook_url == "https://hooks.slack.com/test"

    def test_init_with_bot_token(self):
        """Test initialization with bot token."""
        from src.integrations.slack_integration import SlackIntegration
        slack = SlackIntegration(bot_token="xoxb-test-token")
        assert slack.bot_token == "xoxb-test-token"

    def test_format_results_blocks(self, slack, test_summary):
        """Test formatting results as Slack blocks."""
        blocks = slack._format_results_blocks(test_summary)

        assert len(blocks) > 0
        assert blocks[0]["type"] == "header"
        # Check for summary section
        section_blocks = [b for b in blocks if b.get("type") == "section"]
        assert len(section_blocks) > 0

    def test_format_results_blocks_with_failures(self, slack, test_summary):
        """Test formatting results with failures."""
        blocks = slack._format_results_blocks(test_summary)

        # Should have failure section
        block_texts = str(blocks)
        assert "Failed Tests" in block_texts or "login-test" in block_texts

    def test_format_failure_blocks(self, slack):
        """Test formatting failure alert blocks."""
        blocks = slack._format_failure_blocks(
            test_id="checkout-test",
            error="Payment form not loading",
            root_cause="API timeout",
        )

        assert len(blocks) > 0
        assert blocks[0]["type"] == "header"
        block_text = str(blocks)
        assert "checkout-test" in block_text

    @pytest.mark.asyncio
    async def test_send_test_results(self, slack, test_summary):
        """Test sending test results."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_instance

            result = await slack.send_test_results(test_summary)

            assert result is True

    @pytest.mark.asyncio
    async def test_send_failure_alert(self, slack):
        """Test sending failure alert."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_instance

            result = await slack.send_failure_alert(
                test_id="critical-test",
                error="Database connection failed",
                root_cause="Connection pool exhausted",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_send_simple_message(self, slack):
        """Test sending simple message."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_instance

            result = await slack.send_simple_message("Test started!")

            assert result is True


class TestSlackNotifier:
    """Tests for SlackNotifier helper class."""

    @pytest.fixture
    def notifier(self):
        """Create SlackNotifier with mocked webhook."""
        from src.integrations.slack_integration import SlackNotifier
        return SlackNotifier(webhook_url="https://hooks.slack.com/test")

    @pytest.mark.asyncio
    async def test_notify_start(self, notifier):
        """Test notifying test start."""
        with patch.object(notifier.slack, "send_test_started", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await notifier.notify_start(test_count=10, pr=42)
            assert result is True
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_complete(self, notifier):
        """Test notifying test completion."""
        with patch.object(notifier.slack, "send_test_completed", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await notifier.notify_complete(
                passed=8,
                failed=2,
                skipped=0,
                duration=45.0,
                cost=0.25,
                failures=[],
            )
            assert result is True


# ============================================================================
# Reporter Tests
# ============================================================================

class TestReportGenerator:
    """Tests for ReportGenerator class."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "test-results"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def reporter(self, temp_output_dir):
        """Create ReportGenerator with temp directory."""
        from src.integrations.reporter import ReportGenerator
        return ReportGenerator(output_dir=str(temp_output_dir))

    @pytest.fixture
    def report_data(self):
        """Create sample report data."""
        from src.integrations.reporter import ReportData, TestResultData
        return ReportData(
            run_id="test-run-123",
            app_url="http://localhost:3000",
            started_at="2024-01-01T10:00:00Z",
            completed_at="2024-01-01T10:05:00Z",
            total_tests=10,
            passed=8,
            failed=2,
            skipped=0,
            duration_seconds=300.0,
            total_cost_usd=0.50,
            test_results=[
                TestResultData(
                    test_id="test-1",
                    name="Login Test",
                    status="passed",
                    duration_seconds=15.0,
                ),
                TestResultData(
                    test_id="test-2",
                    name="Checkout Test",
                    status="failed",
                    duration_seconds=25.0,
                    error_message="Element not found: #checkout-btn",
                ),
            ],
            failures=[
                {
                    "test_id": "test-2",
                    "failure_type": "element_not_found",
                    "root_cause": "Button removed in recent update",
                }
            ],
        )

    def test_init(self, reporter, temp_output_dir):
        """Test reporter initialization."""
        assert reporter.output_dir == temp_output_dir
        assert reporter.include_screenshots is True

    def test_generate_json(self, reporter, report_data, temp_output_dir):
        """Test generating JSON report."""
        output_path = temp_output_dir / "test.json"
        result_path = reporter.generate_json(report_data, output_path)

        assert result_path.exists()
        with open(result_path) as f:
            data = json.load(f)
        assert data["run_id"] == "test-run-123"
        assert data["summary"]["total"] == 10

    def test_generate_html(self, reporter, report_data, temp_output_dir):
        """Test generating HTML report."""
        output_path = temp_output_dir / "test.html"
        result_path = reporter.generate_html(report_data, output_path)

        assert result_path.exists()
        content = result_path.read_text()
        assert "E2E Test Report" in content
        assert "test-run-123" in content

    def test_generate_markdown(self, reporter, report_data, temp_output_dir):
        """Test generating Markdown report."""
        output_path = temp_output_dir / "test.md"
        result_path = reporter.generate_markdown(report_data, output_path)

        assert result_path.exists()
        content = result_path.read_text()
        assert "# E2E Test Report" in content
        assert "test-run-123" in content

    def test_generate_junit(self, reporter, report_data, temp_output_dir):
        """Test generating JUnit XML report."""
        output_path = temp_output_dir / "test.xml"
        result_path = reporter.generate_junit(report_data, output_path)

        assert result_path.exists()
        content = result_path.read_text()
        assert '<?xml version="1.0"' in content
        assert 'testsuite' in content
        assert 'tests="10"' in content

    def test_generate_all(self, reporter, report_data):
        """Test generating all report formats."""
        paths = reporter.generate_all(report_data)

        assert "json" in paths
        assert "html" in paths
        assert "markdown" in paths
        assert "junit" in paths
        assert all(p.exists() for p in paths.values() if isinstance(p, Path))


class TestReportData:
    """Tests for ReportData and TestResultData dataclasses."""

    def test_test_result_data_creation(self):
        """Test creating test result data."""
        from src.integrations.reporter import TestResultData

        result = TestResultData(
            test_id="test-1",
            name="My Test",
            status="passed",
            duration_seconds=10.5,
        )

        assert result.test_id == "test-1"
        assert result.status == "passed"
        assert result.error_message is None

    def test_test_result_data_with_failure(self):
        """Test creating failed test result data."""
        from src.integrations.reporter import TestResultData

        result = TestResultData(
            test_id="test-2",
            name="Failed Test",
            status="failed",
            duration_seconds=5.0,
            error_message="Assertion failed",
            screenshots=["base64data"],
        )

        assert result.status == "failed"
        assert result.error_message == "Assertion failed"
        assert len(result.screenshots) == 1


class TestCreateReportFromState:
    """Tests for create_report_from_state function."""

    def test_create_from_state(self):
        """Test creating report data from orchestrator state."""
        from src.integrations.reporter import create_report_from_state

        state = {
            "run_id": "run-abc",
            "app_url": "http://localhost:3000",
            "started_at": "2024-01-01T00:00:00Z",
            "passed_count": 5,
            "failed_count": 1,
            "skipped_count": 0,
            "total_cost": 0.15,
            "test_results": [
                {
                    "test_id": "test-1",
                    "status": "passed",
                    "duration_seconds": 10.0,
                }
            ],
            "failures": [],
        }

        report_data = create_report_from_state(state)

        assert report_data.run_id == "run-abc"
        assert report_data.total_tests == 6
        assert report_data.passed == 5
        assert len(report_data.test_results) == 1
