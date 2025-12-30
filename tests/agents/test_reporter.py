"""Tests for the reporter module."""

import pytest
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch


# Patch target for Anthropic client in base agent
ANTHROPIC_PATCH = 'anthropic.Anthropic'


class TestTestSummary:
    """Tests for TestSummary dataclass."""

    def test_summary_creation(self, mock_env_vars):
        """Test TestSummary creation."""
        from src.agents.reporter import TestSummary

        summary = TestSummary(
            total_tests=10,
            passed=8,
            failed=2,
            errors=0,
            skipped=0,
            total_duration_ms=5000,
        )

        assert summary.total_tests == 10
        assert summary.passed == 8
        assert summary.pass_rate == 80.0

    def test_summary_zero_tests(self, mock_env_vars):
        """Test TestSummary with zero tests."""
        from src.agents.reporter import TestSummary

        summary = TestSummary(
            total_tests=0,
            passed=0,
            failed=0,
        )

        assert summary.pass_rate == 0.0

    def test_summary_all_passed(self, mock_env_vars):
        """Test TestSummary with all tests passed."""
        from src.agents.reporter import TestSummary

        summary = TestSummary(
            total_tests=5,
            passed=5,
            failed=0,
        )

        assert summary.pass_rate == 100.0


class TestFailureReport:
    """Tests for FailureReport dataclass."""

    def test_failure_report_creation(self, mock_env_vars):
        """Test FailureReport creation."""
        from src.agents.reporter import FailureReport

        report = FailureReport(
            test_id="test-001",
            test_name="Login Test",
            failure_type="assertion",
            error_message="Expected dashboard URL",
        )

        assert report.test_id == "test-001"
        assert report.priority == "medium"  # default

    def test_failure_report_with_all_fields(self, mock_env_vars):
        """Test FailureReport with all fields."""
        from src.agents.reporter import FailureReport

        report = FailureReport(
            test_id="test-002",
            test_name="Checkout Test",
            failure_type="timeout",
            error_message="Element not found",
            step_failed=3,
            screenshot_path="/screenshots/fail.png",
            suggested_fix="Increase wait timeout",
            priority="critical",
        )

        assert report.step_failed == 3
        assert report.priority == "critical"
        assert report.screenshot_path is not None


class TestTestReport:
    """Tests for TestReport dataclass."""

    def test_report_creation(self, mock_env_vars):
        """Test TestReport creation."""
        from src.agents.reporter import TestReport, TestSummary

        summary = TestSummary(total_tests=1, passed=1)

        report = TestReport(
            run_id="run-001",
            timestamp=datetime.now().isoformat(),
            summary=summary,
        )

        assert report.run_id == "run-001"
        assert len(report.failures) == 0

    def test_report_to_dict(self, mock_env_vars):
        """Test TestReport to_dict method."""
        from src.agents.reporter import TestReport, TestSummary, FailureReport

        summary = TestSummary(total_tests=2, passed=1, failed=1)
        failure = FailureReport(
            test_id="test-001",
            test_name="Failing Test",
            failure_type="error",
            error_message="Something went wrong",
            priority="high",
        )

        report = TestReport(
            run_id="run-002",
            timestamp=datetime.now().isoformat(),
            summary=summary,
            failures=[failure],
            recommendations=["Fix the bug"],
        )

        result = report.to_dict()

        assert result["run_id"] == "run-002"
        assert result["summary"]["total_tests"] == 2
        assert result["summary"]["pass_rate"] == 50.0
        assert len(result["failures"]) == 1
        assert result["failures"][0]["priority"] == "high"


class TestReporterAgent:
    """Tests for ReporterAgent class."""

    def test_agent_creation(self, mock_env_vars):
        """Test ReporterAgent creation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.reporter import ReporterAgent

            agent = ReporterAgent()

            assert agent is not None

    def test_get_system_prompt(self, mock_env_vars):
        """Test system prompt generation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.reporter import ReporterAgent

            agent = ReporterAgent()
            prompt = agent._get_system_prompt()

            assert "report" in prompt.lower()
            assert "actionable" in prompt.lower()

    def test_calculate_summary(self, mock_env_vars):
        """Test summary calculation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.reporter import ReporterAgent

            agent = ReporterAgent()

            results = [
                {"status": "passed", "duration_ms": 1000},
                {"status": "passed", "duration_ms": 2000},
                {"status": "failed", "duration_ms": 500},
                {"status": "error", "duration_ms": 100},
                {"status": "skipped", "duration_ms": 0},
            ]

            summary = agent._calculate_summary(results)

            assert summary.total_tests == 5
            assert summary.passed == 2
            assert summary.failed == 1
            assert summary.errors == 1
            assert summary.skipped == 1
            assert summary.total_duration_ms == 3600
            assert summary.pass_rate == 40.0

    def test_extract_failures(self, mock_env_vars):
        """Test failure extraction."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.reporter import ReporterAgent

            agent = ReporterAgent()

            results = [
                {"status": "passed", "test_id": "t1"},
                {
                    "status": "failed",
                    "test_id": "t2",
                    "test_name": "Login Test",
                    "failure_type": "assertion",
                    "error_message": "URL mismatch",
                    "priority": "critical",
                },
                {
                    "status": "error",
                    "test_id": "t3",
                    "test_name": "Checkout Test",
                    "failure_type": "timeout",
                    "error_message": "Timed out",
                    "priority": "high",
                },
            ]

            failures = agent._extract_failures(results)

            assert len(failures) == 2
            # Should be sorted by priority
            assert failures[0].priority == "critical"
            assert failures[1].priority == "high"

    def test_generate_markdown_report(self, mock_env_vars):
        """Test markdown report generation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.reporter import ReporterAgent, TestSummary, FailureReport

            agent = ReporterAgent()

            summary = TestSummary(total_tests=10, passed=8, failed=2)
            failures = [
                FailureReport(
                    test_id="t1",
                    test_name="Login Test",
                    failure_type="assertion",
                    error_message="Failed",
                    priority="critical",
                ),
            ]

            markdown = agent._generate_markdown_report(
                run_id="run-001",
                summary=summary,
                failures=failures,
                results=[],
                metadata={"env": "staging"},
            )

            assert "# E2E Test Report" in markdown
            assert "run-001" in markdown
            assert "80.0%" in markdown
            assert "Login Test" in markdown
            assert "🔴" in markdown  # Critical priority emoji

    def test_generate_markdown_report_all_passed(self, mock_env_vars):
        """Test markdown report with all tests passed."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.reporter import ReporterAgent, TestSummary

            agent = ReporterAgent()

            summary = TestSummary(total_tests=5, passed=5, failed=0)

            markdown = agent._generate_markdown_report(
                run_id="run-002",
                summary=summary,
                failures=[],
                results=[],
                metadata=None,
            )

            assert "✅ **All tests passed!**" in markdown

    def test_generate_html_report(self, mock_env_vars):
        """Test HTML report generation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.reporter import ReporterAgent, TestSummary, FailureReport

            agent = ReporterAgent()

            summary = TestSummary(total_tests=5, passed=4, failed=1)
            failures = [
                FailureReport(
                    test_id="t1",
                    test_name="Test",
                    failure_type="error",
                    error_message="Error",
                    priority="high",
                ),
            ]
            results = [
                {"test_name": "Test 1", "status": "passed", "duration_ms": 100},
                {"test_name": "Test 2", "status": "failed", "duration_ms": 200},
            ]

            html = agent._generate_html_report(
                run_id="run-003",
                summary=summary,
                failures=failures,
                results=results,
                metadata=None,
            )

            assert "<!DOCTYPE html>" in html
            assert "E2E Test Report" in html
            assert "run-003" in html
            assert "80.0%" in html

    @pytest.mark.asyncio
    async def test_execute(self, mock_env_vars):
        """Test full report execution."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.reporter import ReporterAgent

            agent = ReporterAgent()

            results = [
                {"test_id": "t1", "test_name": "Test 1", "status": "passed", "duration_ms": 100},
                {"test_id": "t2", "test_name": "Test 2", "status": "passed", "duration_ms": 200},
            ]

            report = await agent.execute(
                test_results=results,
                execution_metadata={"run_id": "test-run"},
                include_recommendations=False,
            )

            assert report.success is True
            assert report.data.run_id == "test-run"
            assert report.data.summary.total_tests == 2
            assert report.data.summary.pass_rate == 100.0

    @pytest.mark.asyncio
    async def test_execute_with_failures(self, mock_env_vars):
        """Test report execution with failures."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='["Fix the selector", "Add more waits"]')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.reporter import ReporterAgent

            agent = ReporterAgent()

            results = [
                {"test_id": "t1", "status": "passed", "duration_ms": 100},
                {
                    "test_id": "t2",
                    "test_name": "Failing Test",
                    "status": "failed",
                    "failure_type": "assertion",
                    "error_message": "Element not found",
                    "duration_ms": 200,
                },
            ]

            report = await agent.execute(
                test_results=results,
                execution_metadata={"run_id": "fail-run"},
                include_recommendations=True,
            )

            assert report.success is True
            assert len(report.data.failures) == 1
            assert len(report.data.recommendations) == 2

    @pytest.mark.asyncio
    async def test_generate_recommendations(self, mock_env_vars):
        """Test AI recommendation generation."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='["Recommendation 1", "Recommendation 2", "Recommendation 3"]')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.reporter import ReporterAgent, TestSummary, FailureReport

            agent = ReporterAgent()

            summary = TestSummary(total_tests=10, passed=5, failed=5)
            failures = [
                FailureReport(
                    test_id="t1",
                    test_name="Test",
                    failure_type="error",
                    error_message="Error",
                    priority="high",
                ),
            ]

            recommendations = await agent._generate_recommendations(
                summary, failures, []
            )

            assert len(recommendations) == 3

    @pytest.mark.asyncio
    async def test_generate_recommendations_cost_limit(self, mock_env_vars):
        """Test recommendations with cost limit exceeded."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.reporter import ReporterAgent, TestSummary, FailureReport

            agent = ReporterAgent()
            agent._usage.total_cost = 100.0  # Exceed default limit (10.0)

            summary = TestSummary(total_tests=1, failed=1)
            failures = [
                FailureReport("t1", "Test", "error", "Error", priority="high"),
            ]

            recommendations = await agent._generate_recommendations(
                summary, failures, []
            )

            assert "cost limit" in recommendations[0].lower()

    @pytest.mark.asyncio
    async def test_create_github_issue(self, mock_env_vars):
        """Test GitHub issue body generation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.reporter import ReporterAgent, FailureReport

            agent = ReporterAgent()

            failure = FailureReport(
                test_id="test-001",
                test_name="Login Test",
                failure_type="assertion",
                error_message="URL did not match",
                step_failed=5,
                priority="critical",
                suggested_fix="Update URL assertion",
            )

            issue_body = await agent.create_github_issue(
                failure=failure,
                repo="owner/repo",
            )

            assert "## Test Failure Report" in issue_body
            assert "Login Test" in issue_body
            assert "critical" in issue_body
            assert "URL did not match" in issue_body
            assert "Update URL assertion" in issue_body

    def test_save_report(self, mock_env_vars):
        """Test saving reports to files."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.reporter import ReporterAgent, TestReport, TestSummary

            agent = ReporterAgent()

            summary = TestSummary(total_tests=1, passed=1)
            report = TestReport(
                run_id="save-test",
                timestamp=datetime.now().isoformat(),
                summary=summary,
                markdown_report="# Test Report",
                html_report="<html><body>Report</body></html>",
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                saved = agent.save_report(
                    report=report,
                    output_dir=tmpdir,
                    formats=["json", "markdown", "html"],
                )

                assert len(saved) == 3
                assert os.path.exists(saved["json"])
                assert os.path.exists(saved["markdown"])
                assert os.path.exists(saved["html"])

                # Verify JSON content
                with open(saved["json"]) as f:
                    data = json.load(f)
                    assert data["run_id"] == "save-test"

    def test_save_report_selected_formats(self, mock_env_vars):
        """Test saving reports with selected formats."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.reporter import ReporterAgent, TestReport, TestSummary

            agent = ReporterAgent()

            summary = TestSummary(total_tests=1, passed=1)
            report = TestReport(
                run_id="format-test",
                timestamp=datetime.now().isoformat(),
                summary=summary,
                markdown_report="# Report",
                html_report="<html></html>",
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                saved = agent.save_report(
                    report=report,
                    output_dir=tmpdir,
                    formats=["json"],  # Only JSON
                )

                assert len(saved) == 1
                assert "json" in saved
                assert "markdown" not in saved
                assert "html" not in saved
