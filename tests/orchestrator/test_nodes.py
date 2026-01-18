"""Tests for the orchestrator nodes module."""

from unittest.mock import MagicMock, patch

import pytest


class TestTrackUsage:
    """Tests for _track_usage helper function."""

    def test_track_usage_updates_tokens(self, mock_env_vars):
        """Test tracking usage updates token counts."""
        from src.orchestrator.nodes import _track_usage

        state = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "iteration": 0,
        }

        response = MagicMock()
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50

        result = _track_usage(state, response)

        assert result["total_input_tokens"] == 100
        assert result["total_output_tokens"] == 50
        assert result["total_cost"] > 0
        assert result["iteration"] == 1

    def test_track_usage_no_usage_attr(self, mock_env_vars):
        """Test tracking when response has no usage attribute."""
        from src.orchestrator.nodes import _track_usage

        state = {
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "total_cost": 0.01,
            "iteration": 1,
        }

        response = MagicMock(spec=[])  # No usage attribute

        result = _track_usage(state, response)

        assert result["total_input_tokens"] == 10  # Unchanged
        assert result["total_output_tokens"] == 5  # Unchanged
        assert result["iteration"] == 2  # Incremented


class TestAnalyzeCodeNode:
    """Tests for analyze_code_node function."""

    @pytest.mark.asyncio
    async def test_analyze_code_success(self, mock_env_vars):
        """Test successful code analysis."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            with patch("src.orchestrator.nodes.create_secure_reader") as mock_reader:
                with patch("src.orchestrator.nodes.get_audit_logger"):
                    from src.orchestrator.nodes import analyze_code_node

                    # Setup mock reader
                    mock_reader_instance = MagicMock()
                    mock_reader_instance.read_codebase.return_value = []
                    mock_reader_instance.get_file_summary.return_value = {
                        "readable": 10,
                        "skipped": 2,
                        "secrets_redacted": 1,
                        "by_sensitivity": {},
                    }
                    mock_reader_instance.get_context_for_ai.return_value = "# Code"
                    mock_reader.return_value = mock_reader_instance

                    # Setup mock Claude response
                    mock_response = MagicMock()
                    mock_response.usage.input_tokens = 100
                    mock_response.usage.output_tokens = 200
                    mock_response.content = [
                        MagicMock(
                            text='{"summary": "Test app", "framework": "React", "testable_surfaces": []}'
                        )
                    ]
                    mock_anthropic.return_value.messages.create.return_value = (
                        mock_response
                    )

                    state = {
                        "codebase_path": "/app",
                        "app_url": "http://localhost:3000",
                        "user_id": "test-user",
                        "session_id": "session-123",
                        "changed_files": [],
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "total_cost": 0.0,
                        "iteration": 0,
                    }

                    result = await analyze_code_node(state)

                    assert result["codebase_summary"] == "Test app"
                    assert result["testable_surfaces"] == []
                    assert "security_summary" in result

    @pytest.mark.asyncio
    async def test_analyze_code_with_changed_files(self, mock_env_vars):
        """Test code analysis with changed files."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            with patch("src.orchestrator.nodes.create_secure_reader") as mock_reader:
                with patch("src.orchestrator.nodes.get_audit_logger"):
                    import tempfile
                    from pathlib import Path

                    from src.orchestrator.nodes import analyze_code_node

                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Create a test file
                        test_file = Path(tmpdir) / "changed.py"
                        test_file.write_text("def test(): pass")

                        mock_reader_instance = MagicMock()
                        mock_reader_instance.read_codebase.return_value = []
                        mock_reader_instance.get_file_summary.return_value = {
                            "readable": 1,
                            "skipped": 0,
                            "secrets_redacted": 0,
                            "by_sensitivity": {},
                        }
                        mock_reader_instance.get_context_for_ai.return_value = "# Code"

                        read_result = MagicMock()
                        read_result.skipped = False
                        mock_reader_instance.read_file.return_value = read_result
                        mock_reader.return_value = mock_reader_instance

                        mock_response = MagicMock()
                        mock_response.usage.input_tokens = 100
                        mock_response.usage.output_tokens = 200
                        mock_response.content = [
                            MagicMock(text='{"summary": "App", "testable_surfaces": []}')
                        ]
                        mock_anthropic.return_value.messages.create.return_value = (
                            mock_response
                        )

                        state = {
                            "codebase_path": tmpdir,
                            "app_url": "http://localhost",
                            "user_id": "user",
                            "session_id": "session",
                            "changed_files": ["changed.py"],
                            "total_input_tokens": 0,
                            "total_output_tokens": 0,
                            "total_cost": 0.0,
                            "iteration": 0,
                        }

                        result = await analyze_code_node(state)

                        assert "error" not in result or result.get("error") is None

    @pytest.mark.asyncio
    async def test_analyze_code_permission_error(self, mock_env_vars):
        """Test code analysis with permission error."""
        with patch("src.orchestrator.nodes.create_secure_reader") as mock_reader:
            with patch("src.orchestrator.nodes.get_audit_logger"):
                from src.orchestrator.nodes import analyze_code_node

                mock_reader.side_effect = PermissionError("Consent required")

                state = {
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                    "user_id": "user",
                    "session_id": "session",
                    "changed_files": [],
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost": 0.0,
                    "iteration": 0,
                }

                result = await analyze_code_node(state)

                assert "error" in result
                assert "Consent" in result["error"]


class TestPlanTestsNode:
    """Tests for plan_tests_node function."""

    @pytest.mark.asyncio
    async def test_plan_tests_success(self, mock_env_vars):
        """Test successful test planning."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            from src.orchestrator.nodes import plan_tests_node

            mock_response = MagicMock()
            mock_response.usage.input_tokens = 200
            mock_response.usage.output_tokens = 500
            mock_response.content = [
                MagicMock(
                    text="""[
                    {"id": "test-001", "name": "Login", "type": "ui", "priority": "critical"},
                    {"id": "test-002", "name": "API", "type": "api", "priority": "high"}
                ]"""
                )
            ]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            state = {
                "codebase_summary": "A web app",
                "testable_surfaces": [{"type": "ui", "name": "Login"}],
                "app_url": "http://localhost",
                "changed_files": [],
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "iteration": 0,
            }

            result = await plan_tests_node(state)

            assert len(result["test_plan"]) == 2
            # Should be sorted by priority
            assert result["test_plan"][0]["priority"] == "critical"
            assert result["current_test_index"] == 0

    @pytest.mark.asyncio
    async def test_plan_tests_error(self, mock_env_vars):
        """Test test planning with error."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            from src.orchestrator.nodes import plan_tests_node

            mock_anthropic.return_value.messages.create.side_effect = Exception(
                "API Error"
            )

            state = {
                "codebase_summary": "App",
                "testable_surfaces": [],
                "app_url": "http://localhost",
                "changed_files": [],
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "iteration": 0,
            }

            result = await plan_tests_node(state)

            assert "error" in result
            assert "planning failed" in result["error"]


class TestExecuteTestNode:
    """Tests for execute_test_node function."""

    @pytest.mark.asyncio
    async def test_execute_test_all_complete(self, mock_env_vars):
        """Test when all tests are complete."""
        from src.orchestrator.nodes import execute_test_node

        state = {
            "test_plan": [{"id": "test-001"}],
            "current_test_index": 1,  # Past the end
            "should_continue": True,
        }

        result = await execute_test_node(state)

        assert result["should_continue"] is False

    @pytest.mark.asyncio
    async def test_execute_test_ui_simulated(self, mock_env_vars):
        """Test executing a UI test with simulated execution."""
        with patch("anthropic.Anthropic"):
            with patch(
                "src.orchestrator.nodes._execute_ui_test"
            ) as mock_ui:
                from src.orchestrator.nodes import execute_test_node
                from src.orchestrator.state import TestResult, TestStatus

                mock_result = TestResult(
                    test_id="test-001",
                    status=TestStatus.PASSED,
                    duration_seconds=2.0,
                )
                mock_ui.return_value = mock_result

                state = {
                    "test_plan": [
                        {
                            "id": "test-001",
                            "name": "Login Test",
                            "type": "ui",
                            "steps": [],
                            "assertions": [],
                        }
                    ],
                    "current_test_index": 0,
                    "app_url": "http://localhost",
                    "test_results": [],
                    "passed_count": 0,
                    "failed_count": 0,
                    "healing_queue": [],
                    "failures": [],
                    "iteration": 0,
                }

                result = await execute_test_node(state)

                assert result["passed_count"] == 1
                assert result["current_test_index"] == 1
                assert len(result["test_results"]) == 1


class TestSelfHealNode:
    """Tests for self_heal_node function."""

    @pytest.mark.asyncio
    async def test_self_heal_no_queue(self, mock_env_vars):
        """Test self heal with empty queue."""
        from src.orchestrator.nodes import self_heal_node

        state = {
            "healing_queue": [],
        }

        result = await self_heal_node(state)

        assert result["healing_queue"] == []

    @pytest.mark.asyncio
    async def test_self_heal_success(self, mock_env_vars):
        """Test successful self healing."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            from src.config import Settings
            from src.orchestrator.nodes import self_heal_node

            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 200
            mock_response.content = [
                MagicMock(
                    text="""{
                    "diagnosis": "selector_changed",
                    "root_cause": "Button ID changed",
                    "fix": {"type": "update_selector", "original": "#old", "replacement": "#new", "step_index": 0},
                    "confidence": 0.9,
                    "should_auto_fix": true
                }"""
                )
            ]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            settings = Settings()
            settings.self_heal_confidence_threshold = 0.8

            with patch("src.orchestrator.nodes.get_settings", return_value=settings):
                state = {
                    "healing_queue": ["test-001"],
                    "test_plan": [
                        {"id": "test-001", "steps": [{"target": "#old"}], "assertions": []}
                    ],
                    "failures": [
                        {"test_id": "test-001", "failure_type": "unknown", "root_cause": ""}
                    ],
                    "test_results": [{"test_id": "test-001"}],
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost": 0.0,
                    "iteration": 0,
                }

                result = await self_heal_node(state)

                # Should remove from healing queue
                assert result["healing_queue"] == []
                # Should update selector
                assert result["test_plan"][0]["steps"][0]["target"] == "#new"

    @pytest.mark.asyncio
    async def test_self_heal_low_confidence(self, mock_env_vars):
        """Test self heal with low confidence."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            from src.config import Settings
            from src.orchestrator.nodes import self_heal_node

            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 200
            mock_response.content = [
                MagicMock(
                    text="""{
                    "diagnosis": "unknown",
                    "root_cause": "Not sure",
                    "confidence": 0.3,
                    "should_auto_fix": false
                }"""
                )
            ]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            settings = Settings()
            settings.self_heal_confidence_threshold = 0.8

            with patch("src.orchestrator.nodes.get_settings", return_value=settings):
                state = {
                    "healing_queue": ["test-001"],
                    "test_plan": [{"id": "test-001", "steps": []}],
                    "failures": [
                        {"test_id": "test-001", "failure_type": "unknown", "root_cause": ""}
                    ],
                    "test_results": [],
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost": 0.0,
                    "iteration": 0,
                }

                result = await self_heal_node(state)

                # Should still remove from queue even if not fixed
                assert result["healing_queue"] == []


class TestReportNode:
    """Tests for report_node function."""

    @pytest.mark.asyncio
    async def test_report_node_success(self, mock_env_vars):
        """Test successful report generation."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            with patch("src.integrations.reporter.create_reporter"):
                with patch("src.integrations.reporter.create_report_from_state"):
                    from src.orchestrator.nodes import report_node

                    mock_response = MagicMock()
                    mock_response.usage.input_tokens = 100
                    mock_response.usage.output_tokens = 200
                    mock_response.content = [
                        MagicMock(text="All tests passed successfully.")
                    ]
                    mock_anthropic.return_value.messages.create.return_value = (
                        mock_response
                    )

                    state = {
                        "passed_count": 10,
                        "failed_count": 0,
                        "skipped_count": 0,
                        "failures": [],
                        "pr_number": None,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "total_cost": 0.01,
                        "iteration": 10,
                        "should_continue": True,
                    }

                    result = await report_node(state)

                    assert result["should_continue"] is False
                    # executive_summary may or may not be present depending on AI call
                    # Just check that the function completed

    @pytest.mark.asyncio
    async def test_report_node_with_failures(self, mock_env_vars):
        """Test report with failures."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            with patch("src.integrations.reporter.create_reporter"):
                with patch("src.integrations.reporter.create_report_from_state"):
                    from src.orchestrator.nodes import report_node

                    mock_response = MagicMock()
                    mock_response.usage.input_tokens = 100
                    mock_response.usage.output_tokens = 200
                    mock_response.content = [
                        MagicMock(text="2 tests failed.")
                    ]
                    mock_anthropic.return_value.messages.create.return_value = (
                        mock_response
                    )

                    state = {
                        "passed_count": 8,
                        "failed_count": 2,
                        "skipped_count": 0,
                        "failures": [
                            {"test_id": "test-001", "failure_type": "real_bug"}
                        ],
                        "pr_number": None,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "total_cost": 0.02,
                        "iteration": 12,
                        "should_continue": True,
                    }

                    result = await report_node(state)

                    assert result["should_continue"] is False
