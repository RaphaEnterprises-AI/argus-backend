"""Extended tests for orchestrator nodes."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json

from src.orchestrator.nodes import (
    _track_usage,
    analyze_code_node,
    plan_tests_node,
    execute_test_node,
    self_heal_node,
    report_node,
    _execute_ui_test,
    _execute_api_test,
    _execute_simulated_test,
)
from src.orchestrator.state import TestingState, TestStatus


class TestTrackUsage:
    """Tests for _track_usage helper."""

    def test_track_usage_with_usage(self):
        """Test tracking usage from response."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost",
            total_input_tokens=100,
            total_output_tokens=50,
            total_cost=0.01,
            iteration=5,
        )

        response = MagicMock()
        response.usage.input_tokens = 1000
        response.usage.output_tokens = 500

        with patch("src.orchestrator.nodes.get_settings") as mock_settings:
            mock_settings.return_value.default_model = "claude-sonnet-4-5-20250514"
            with patch("src.orchestrator.nodes.MODEL_PRICING", {
                "claude-sonnet-4-5-20250514": {"input": 3.0, "output": 15.0}
            }):
                result = _track_usage(state, response)

        assert result["total_input_tokens"] == 1100
        assert result["total_output_tokens"] == 550
        assert result["total_cost"] > 0.01
        assert result["iteration"] == 6

    def test_track_usage_without_usage(self):
        """Test tracking when response has no usage."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost",
            total_input_tokens=100,
            total_output_tokens=50,
            total_cost=0.01,
            iteration=5,
        )

        response = MagicMock(spec=[])  # No usage attribute

        result = _track_usage(state, response)

        assert result["total_input_tokens"] == 100
        assert result["total_output_tokens"] == 50
        assert result["iteration"] == 6


class TestAnalyzeCodeNode:
    """Tests for analyze_code_node."""

    @pytest.mark.asyncio
    async def test_analyze_code_permission_error(self):
        """Test analysis when consent not granted."""
        state = TestingState(
            codebase_path="/test/project",
            app_url="http://localhost:3000",
        )

        with patch("src.orchestrator.nodes.create_secure_reader") as mock_create:
            mock_create.side_effect = PermissionError("Consent required")
            with patch("src.orchestrator.nodes.get_settings"):
                with patch("src.orchestrator.nodes.get_audit_logger"):
                    result = await analyze_code_node(state)

        assert "Consent required" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_analyze_code_api_error(self):
        """Test analysis when API call fails."""
        state = TestingState(
            codebase_path="/test/project",
            app_url="http://localhost:3000",
        )

        mock_reader = MagicMock()
        mock_reader.read_codebase.return_value = []
        mock_reader.get_file_summary.return_value = {
            "readable": 5, "skipped": 0, "secrets_redacted": 0, "by_sensitivity": {}
        }
        mock_reader.get_context_for_ai.return_value = "// Code"

        with patch("src.orchestrator.nodes.create_secure_reader", return_value=mock_reader):
            with patch("src.orchestrator.nodes.get_settings") as mock_settings:
                mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"
                mock_settings.return_value.default_model.value = "claude-sonnet-4-5-20250514"
                with patch("src.orchestrator.nodes.get_audit_logger") as mock_audit:
                    mock_audit.return_value = MagicMock()
                    with patch("anthropic.Anthropic") as mock_anthropic:
                        mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")
                        result = await analyze_code_node(state)

        assert "Code analysis failed" in result.get("error", "")


class TestPlanTestsNode:
    """Tests for plan_tests_node."""

    @pytest.mark.asyncio
    async def test_plan_tests_error(self):
        """Test planning when API fails."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost",
            codebase_summary="App",
            testable_surfaces=[],
        )

        with patch("src.orchestrator.nodes.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"
            mock_settings.return_value.default_model.value = "claude-sonnet-4-5-20250514"
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")
                result = await plan_tests_node(state)

        assert "Test planning failed" in result.get("error", "")


class TestExecuteTestNode:
    """Tests for execute_test_node."""

    @pytest.mark.asyncio
    async def test_execute_all_tests_completed(self):
        """Test when all tests are completed."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost",
            test_plan=[{"id": "test-1"}],
            current_test_index=1,  # Past the end
        )

        result = await execute_test_node(state)

        assert result["should_continue"] is False

    @pytest.mark.asyncio
    async def test_execute_ui_test(self):
        """Test executing a UI test."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost:3000",
            test_plan=[{
                "id": "test-1",
                "name": "Login Test",
                "type": "ui",
                "steps": [{"action": "goto", "target": "/login"}],
                "assertions": [],
            }],
            current_test_index=0,
            test_results=[],
            failures=[],
            healing_queue=[],
            passed_count=0,
            failed_count=0,
            iteration=0,
        )

        with patch("src.orchestrator.nodes._execute_ui_test") as mock_execute:
            from src.orchestrator.state import TestResult, TestStatus
            mock_execute.return_value = TestResult(
                test_id="test-1",
                status=TestStatus.PASSED,
                duration_seconds=1.5,
            )
            with patch("src.orchestrator.nodes.get_settings"):
                result = await execute_test_node(state)

        assert result["passed_count"] == 1
        assert result["current_test_index"] == 1
        assert len(result["test_results"]) == 1

    @pytest.mark.asyncio
    async def test_execute_api_test(self):
        """Test executing an API test."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost:3000",
            test_plan=[{
                "id": "test-1",
                "name": "API Test",
                "type": "api",
                "steps": [],
                "assertions": [],
            }],
            current_test_index=0,
            test_results=[],
            failures=[],
            healing_queue=[],
            passed_count=0,
            failed_count=0,
            iteration=0,
        )

        with patch("src.orchestrator.nodes._execute_api_test") as mock_execute:
            from src.orchestrator.state import TestResult, TestStatus
            mock_execute.return_value = TestResult(
                test_id="test-1",
                status=TestStatus.PASSED,
                duration_seconds=0.5,
            )
            with patch("src.orchestrator.nodes.get_settings"):
                result = await execute_test_node(state)

        assert result["passed_count"] == 1

    @pytest.mark.asyncio
    async def test_execute_db_test_fallback(self):
        """Test executing a DB test with fallback."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost:3000",
            test_plan=[{
                "id": "test-1",
                "name": "DB Test",
                "type": "db",
                "steps": [],
                "assertions": [],
            }],
            current_test_index=0,
            test_results=[],
            failures=[],
            healing_queue=[],
            passed_count=0,
            failed_count=0,
            iteration=0,
        )

        with patch("src.orchestrator.nodes._execute_simulated_test") as mock_execute:
            from src.orchestrator.state import TestResult, TestStatus
            mock_execute.return_value = TestResult(
                test_id="test-1",
                status=TestStatus.PASSED,
                duration_seconds=0.5,
            )
            with patch("src.orchestrator.nodes.get_settings"):
                result = await execute_test_node(state)

        assert result["passed_count"] == 1

    @pytest.mark.asyncio
    async def test_execute_test_failure(self):
        """Test handling test failure."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost:3000",
            test_plan=[{
                "id": "test-1",
                "name": "Failing Test",
                "type": "ui",
                "steps": [],
                "assertions": [],
            }],
            current_test_index=0,
            test_results=[],
            failures=[],
            healing_queue=[],
            passed_count=0,
            failed_count=0,
            iteration=0,
        )

        with patch("src.orchestrator.nodes._execute_ui_test") as mock_execute:
            from src.orchestrator.state import TestResult, TestStatus
            mock_execute.return_value = TestResult(
                test_id="test-1",
                status=TestStatus.FAILED,
                duration_seconds=1.0,
                error_message="Element not found",
                screenshots=["base64screenshot"],
            )
            with patch("src.orchestrator.nodes.get_settings"):
                result = await execute_test_node(state)

        assert result["failed_count"] == 1
        assert "test-1" in result["healing_queue"]
        assert len(result["failures"]) == 1


class TestExecuteAPITest:
    """Tests for _execute_api_test helper."""

    @pytest.mark.asyncio
    async def test_execute_api_test_get_success(self):
        """Test successful GET API test."""
        test = {
            "id": "test-1",
            "name": "API Test",
            "steps": [
                {"action": "api_call", "method": "GET", "target": "/api/users"},
            ],
            "assertions": [
                {"type": "status_code", "expected": "200"},
            ],
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.text = '{"users": []}'

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            result = await _execute_api_test(
                test, "http://localhost", MagicMock(), MagicMock()
            )

        assert result.status == TestStatus.PASSED
        assert result.assertions_passed == 1

    @pytest.mark.asyncio
    async def test_execute_api_test_post(self):
        """Test POST API test."""
        test = {
            "id": "test-1",
            "name": "API Test",
            "steps": [
                {"action": "api_call", "method": "POST", "target": "/api/users", "value": {"name": "Test"}},
            ],
            "assertions": [
                {"type": "status_code", "expected": "201"},
            ],
        }

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.is_success = True

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.post = AsyncMock(return_value=mock_response)

            result = await _execute_api_test(
                test, "http://localhost", MagicMock(), MagicMock()
            )

        assert result.status == TestStatus.PASSED

    @pytest.mark.asyncio
    async def test_execute_api_test_assertion_failure(self):
        """Test API test with assertion failure."""
        test = {
            "id": "test-1",
            "name": "API Test",
            "steps": [
                {"action": "api_call", "method": "GET", "target": "/api/users"},
            ],
            "assertions": [
                {"type": "status_code", "expected": "200"},
            ],
        }

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.is_success = False

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            result = await _execute_api_test(
                test, "http://localhost", MagicMock(), MagicMock()
            )

        assert result.status == TestStatus.FAILED
        assert result.assertions_failed == 1

    @pytest.mark.asyncio
    async def test_execute_api_test_response_contains(self):
        """Test response_contains assertion."""
        test = {
            "id": "test-1",
            "name": "API Test",
            "steps": [
                {"action": "api_call", "method": "GET", "target": "/api/health"},
            ],
            "assertions": [
                {"type": "response_contains", "expected": "healthy"},
            ],
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.text = '{"status": "healthy"}'

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            result = await _execute_api_test(
                test, "http://localhost", MagicMock(), MagicMock()
            )

        assert result.status == TestStatus.PASSED


class TestExecuteSimulatedTest:
    """Tests for _execute_simulated_test helper."""

    @pytest.mark.asyncio
    async def test_execute_simulated_test_pass(self):
        """Test simulated test that passes."""
        test = {
            "id": "test-1",
            "name": "Simulated Test",
            "steps": [],
            "assertions": [],
        }
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost",
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "status": "passed",
            "steps_executed": [{"step": 1, "action": "click", "result": "success"}],
            "assertions_results": [{"assertion": "visible", "passed": True}],
            "error_message": None,
        }))]

        mock_settings = MagicMock()
        mock_settings.anthropic_api_key.get_secret_value.return_value = "key"
        mock_settings.default_model.value = "claude-sonnet-4-5-20250514"

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.return_value = mock_response
            result = await _execute_simulated_test(
                test, state, mock_settings, MagicMock()
            )

        assert result.status == TestStatus.PASSED
        assert result.assertions_passed == 1

    @pytest.mark.asyncio
    async def test_execute_simulated_test_fail(self):
        """Test simulated test that fails."""
        test = {"id": "test-1", "name": "Simulated Test"}
        state = TestingState(codebase_path="/test", app_url="http://localhost")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "status": "failed",
            "steps_executed": [],
            "assertions_results": [{"assertion": "visible", "passed": False}],
            "error_message": "Element not found",
        }))]

        mock_settings = MagicMock()
        mock_settings.anthropic_api_key.get_secret_value.return_value = "key"
        mock_settings.default_model.value = "claude-sonnet-4-5-20250514"

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.return_value = mock_response
            result = await _execute_simulated_test(
                test, state, mock_settings, MagicMock()
            )

        assert result.status == TestStatus.FAILED
        assert result.assertions_failed == 1

    @pytest.mark.asyncio
    async def test_execute_simulated_test_error(self):
        """Test simulated test with API error."""
        test = {"id": "test-1", "name": "Simulated Test"}
        state = TestingState(codebase_path="/test", app_url="http://localhost")

        mock_settings = MagicMock()
        mock_settings.anthropic_api_key.get_secret_value.return_value = "key"
        mock_settings.default_model.value = "claude-sonnet-4-5-20250514"

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")
            result = await _execute_simulated_test(
                test, state, mock_settings, MagicMock()
            )

        assert result.status == TestStatus.FAILED
        assert "API Error" in result.error_message


class TestSelfHealNode:
    """Tests for self_heal_node."""

    @pytest.mark.asyncio
    async def test_self_heal_empty_queue(self):
        """Test healing with empty queue."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost",
            healing_queue=[],
        )

        result = await self_heal_node(state)

        # State should be unchanged
        assert result.get("healing_queue") == []

    @pytest.mark.asyncio
    async def test_self_heal_low_confidence(self):
        """Test healing with low confidence (no auto-fix)."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost",
            healing_queue=["test-1"],
            test_plan=[{
                "id": "test-1",
                "name": "Test",
                "steps": [{"action": "click", "target": "#button"}],
            }],
            failures=[{
                "test_id": "test-1",
                "failure_type": "unknown",
                "root_cause": "Error",
                "confidence": 0.0,
            }],
            test_results=[],
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "diagnosis": "unknown",
            "root_cause": "Unclear",
            "fix": {"type": "none"},
            "confidence": 0.3,
            "should_auto_fix": False,
        }))]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 100

        mock_model = MagicMock()
        mock_model.value = "claude-sonnet-4-5-20250514"

        with patch("src.orchestrator.nodes.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"
            mock_settings.return_value.default_model = mock_model
            mock_settings.return_value.self_heal_confidence_threshold = 0.8
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_anthropic.return_value.messages.create.return_value = mock_response
                with patch("src.orchestrator.nodes.MODEL_PRICING", {
                    "claude-sonnet-4-5-20250514": {"input": 3.0, "output": 15.0}
                }):
                    result = await self_heal_node(state)

        assert result["healing_queue"] == []
        # Step should not be changed
        assert result["test_plan"][0]["steps"][0]["target"] == "#button"

    @pytest.mark.asyncio
    async def test_self_heal_test_not_found(self):
        """Test healing when test not found."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost",
            healing_queue=["test-nonexistent"],
            test_plan=[],
            failures=[],
            test_results=[],
        )

        with patch("src.orchestrator.nodes.get_settings"):
            result = await self_heal_node(state)

        assert result["healing_queue"] == []


class TestReportNode:
    """Tests for report_node."""

    @pytest.mark.asyncio
    async def test_report_node_with_pr_number(self):
        """Test report with GitHub PR comment."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost",
            passed_count=10,
            failed_count=0,
            skipped_count=0,
            failures=[],
            total_cost=0.03,
            iteration=5,
            pr_number=123,
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="All tests passed!")]
        mock_response.usage.input_tokens = 30
        mock_response.usage.output_tokens = 50

        mock_model = MagicMock()
        mock_model.value = "claude-sonnet-4-5-20250514"

        with patch("src.orchestrator.nodes.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"
            mock_settings.return_value.default_model = mock_model
            mock_settings.return_value.output_dir = "/tmp"
            with patch("src.integrations.reporter.create_reporter") as mock_reporter:
                mock_reporter.return_value.generate_all.return_value = {}
                with patch("src.integrations.reporter.create_report_from_state"):
                    with patch("src.integrations.github_integration.GitHubIntegration") as mock_gh:
                        mock_gh.return_value.post_pr_comment = AsyncMock()
                        with patch.dict("os.environ", {
                            "GITHUB_REPOSITORY_OWNER": "owner",
                            "GITHUB_REPOSITORY": "owner/repo",
                        }):
                            with patch("anthropic.Anthropic") as mock_anthropic:
                                mock_anthropic.return_value.messages.create.return_value = mock_response
                                with patch("src.orchestrator.nodes.MODEL_PRICING", {
                                    "claude-sonnet-4-5-20250514": {"input": 3.0, "output": 15.0}
                                }):
                                    result = await report_node(state)

        assert result["should_continue"] is False

    @pytest.mark.asyncio
    async def test_report_node_slack_notification(self):
        """Test report with Slack notification."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost",
            passed_count=5,
            failed_count=5,
            skipped_count=0,
            failures=[],
            total_cost=0.05,
            iteration=10,
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="50% pass rate")]
        mock_response.usage.input_tokens = 30
        mock_response.usage.output_tokens = 50

        mock_model = MagicMock()
        mock_model.value = "claude-sonnet-4-5-20250514"

        with patch("src.orchestrator.nodes.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"
            mock_settings.return_value.default_model = mock_model
            mock_settings.return_value.output_dir = "/tmp"
            with patch("src.integrations.reporter.create_reporter") as mock_reporter:
                mock_reporter.return_value.generate_all.return_value = {}
                with patch("src.integrations.reporter.create_report_from_state"):
                    with patch("src.integrations.slack_integration.SlackIntegration") as mock_slack:
                        mock_slack.return_value.send_test_results = AsyncMock()
                        with patch("anthropic.Anthropic") as mock_anthropic:
                            mock_anthropic.return_value.messages.create.return_value = mock_response
                            with patch("src.orchestrator.nodes.MODEL_PRICING", {
                                "claude-sonnet-4-5-20250514": {"input": 3.0, "output": 15.0}
                            }):
                                result = await report_node(state)

        assert result["should_continue"] is False

    @pytest.mark.asyncio
    async def test_report_node_zero_tests(self):
        """Test report with zero tests."""
        state = TestingState(
            codebase_path="/test",
            app_url="http://localhost",
            passed_count=0,
            failed_count=0,
            skipped_count=0,
            failures=[],
            total_cost=0.0,
            iteration=0,
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="No tests executed")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        mock_model = MagicMock()
        mock_model.value = "claude-sonnet-4-5-20250514"

        with patch("src.orchestrator.nodes.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"
            mock_settings.return_value.default_model = mock_model
            mock_settings.return_value.output_dir = "/tmp"
            with patch("src.integrations.reporter.create_reporter") as mock_reporter:
                mock_reporter.return_value.generate_all.return_value = {}
                with patch("src.integrations.reporter.create_report_from_state"):
                    with patch("anthropic.Anthropic") as mock_anthropic:
                        mock_anthropic.return_value.messages.create.return_value = mock_response
                        with patch("src.orchestrator.nodes.MODEL_PRICING", {
                            "claude-sonnet-4-5-20250514": {"input": 3.0, "output": 15.0}
                        }):
                            result = await report_node(state)

        assert result["should_continue"] is False
