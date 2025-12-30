"""Tests for the orchestrator graph module."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestRouteAfterAnalysis:
    """Tests for route_after_analysis function."""

    def test_route_to_report_on_error(self, mock_env_vars):
        """Test routing to report when error exists."""
        from src.orchestrator.graph import route_after_analysis

        state = {
            "error": "Something went wrong",
            "testable_surfaces": [],
        }

        result = route_after_analysis(state)

        assert result == "report"

    def test_route_to_report_no_surfaces(self, mock_env_vars):
        """Test routing to report when no testable surfaces."""
        from src.orchestrator.graph import route_after_analysis

        state = {
            "error": None,
            "testable_surfaces": [],
        }

        result = route_after_analysis(state)

        assert result == "report"

    def test_route_to_plan_tests(self, mock_env_vars):
        """Test routing to plan_tests on success."""
        from src.orchestrator.graph import route_after_analysis

        state = {
            "error": None,
            "testable_surfaces": [{"type": "ui", "name": "Login"}],
        }

        result = route_after_analysis(state)

        assert result == "plan_tests"


class TestRouteAfterPlanning:
    """Tests for route_after_planning function."""

    def test_route_to_report_on_error(self, mock_env_vars):
        """Test routing to report when error exists."""
        from src.orchestrator.graph import route_after_planning

        state = {
            "error": "Planning failed",
            "test_plan": [],
        }

        result = route_after_planning(state)

        assert result == "report"

    def test_route_to_report_no_plan(self, mock_env_vars):
        """Test routing to report when no test plan."""
        from src.orchestrator.graph import route_after_planning

        state = {
            "error": None,
            "test_plan": [],
        }

        result = route_after_planning(state)

        assert result == "report"

    def test_route_to_execute_test(self, mock_env_vars):
        """Test routing to execute_test on success."""
        from src.orchestrator.graph import route_after_planning

        state = {
            "error": None,
            "test_plan": [{"id": "test-001"}],
        }

        result = route_after_planning(state)

        assert result == "execute_test"


class TestRouteAfterExecution:
    """Tests for route_after_execution function."""

    def test_route_to_report_on_error(self, mock_env_vars):
        """Test routing to report when error exists."""
        from src.orchestrator.graph import route_after_execution

        state = {
            "error": "Execution failed",
            "healing_queue": [],
            "should_continue": True,
            "current_test_index": 0,
            "test_plan": [],
        }

        result = route_after_execution(state)

        assert result == "report"

    def test_route_to_self_heal(self, mock_env_vars):
        """Test routing to self_heal when healing queue has items."""
        from src.orchestrator.graph import route_after_execution
        from src.config import Settings

        settings = Settings()
        settings.self_heal_enabled = True

        state = {
            "error": None,
            "healing_queue": ["test-001"],
            "should_continue": True,
            "current_test_index": 1,
            "test_plan": [{"id": "test-001"}],
        }

        with patch("src.orchestrator.graph.get_settings", return_value=settings):
            result = route_after_execution(state)

        assert result == "self_heal"

    def test_route_to_execute_test_more_tests(self, mock_env_vars):
        """Test routing to execute_test when more tests remain."""
        from src.orchestrator.graph import route_after_execution

        state = {
            "error": None,
            "healing_queue": [],
            "should_continue": True,
            "current_test_index": 0,
            "test_plan": [{"id": "test-001"}, {"id": "test-002"}],
        }

        result = route_after_execution(state)

        assert result == "execute_test"

    def test_route_to_report_all_done(self, mock_env_vars):
        """Test routing to report when all tests complete."""
        from src.orchestrator.graph import route_after_execution

        state = {
            "error": None,
            "healing_queue": [],
            "should_continue": True,
            "current_test_index": 2,
            "test_plan": [{"id": "test-001"}, {"id": "test-002"}],
        }

        result = route_after_execution(state)

        assert result == "report"


class TestRouteAfterHealing:
    """Tests for route_after_healing function."""

    def test_route_to_execute_test_retry(self, mock_env_vars):
        """Test routing to execute_test to retry healed test."""
        from src.orchestrator.graph import route_after_healing

        state = {
            "healing_queue": ["test-001"],
            "current_test_index": 0,
            "test_plan": [{"id": "test-001"}],
        }

        result = route_after_healing(state)

        assert result == "execute_test"

    def test_route_to_execute_test_more_tests(self, mock_env_vars):
        """Test routing to execute_test when more tests remain."""
        from src.orchestrator.graph import route_after_healing

        state = {
            "healing_queue": [],
            "current_test_index": 0,
            "test_plan": [{"id": "test-001"}, {"id": "test-002"}],
        }

        result = route_after_healing(state)

        assert result == "execute_test"

    def test_route_to_report(self, mock_env_vars):
        """Test routing to report when no more tests."""
        from src.orchestrator.graph import route_after_healing

        state = {
            "healing_queue": [],
            "current_test_index": 2,
            "test_plan": [{"id": "test-001"}, {"id": "test-002"}],
        }

        result = route_after_healing(state)

        assert result == "report"


class TestShouldContinue:
    """Tests for should_continue function."""

    def test_stop_on_max_iterations(self, mock_env_vars):
        """Test stopping when max iterations reached."""
        from src.orchestrator.graph import should_continue

        state = {
            "iteration": 100,
            "max_iterations": 100,
            "total_cost": 0.0,
            "error": None,
            "should_continue": True,
        }

        result = should_continue(state)

        assert result is False

    def test_stop_on_cost_limit(self, mock_env_vars):
        """Test stopping when cost limit reached."""
        from src.orchestrator.graph import should_continue
        from src.config import Settings

        settings = Settings()
        settings.cost_limit_per_run = 5.0

        state = {
            "iteration": 10,
            "max_iterations": 100,
            "total_cost": 10.0,  # Over limit
            "error": None,
            "should_continue": True,
        }

        with patch("src.orchestrator.graph.get_settings", return_value=settings):
            result = should_continue(state)

        assert result is False

    def test_stop_on_error(self, mock_env_vars):
        """Test stopping when error exists."""
        from src.orchestrator.graph import should_continue

        state = {
            "iteration": 10,
            "max_iterations": 100,
            "total_cost": 0.0,
            "error": "Something went wrong",
            "should_continue": True,
        }

        result = should_continue(state)

        assert result is False

    def test_continue_normal(self, mock_env_vars):
        """Test continuing in normal case."""
        from src.orchestrator.graph import should_continue
        from src.config import Settings

        settings = Settings()
        settings.cost_limit_per_run = 10.0

        state = {
            "iteration": 10,
            "max_iterations": 100,
            "total_cost": 1.0,
            "error": None,
            "should_continue": True,
        }

        with patch("src.orchestrator.graph.get_settings", return_value=settings):
            result = should_continue(state)

        assert result is True


class TestCreateTestingGraph:
    """Tests for create_testing_graph function."""

    def test_create_graph(self, mock_env_vars):
        """Test creating the testing graph."""
        from src.orchestrator.graph import create_testing_graph
        from src.config import Settings

        settings = Settings()

        graph = create_testing_graph(settings)

        assert graph is not None


class TestTestingOrchestrator:
    """Tests for TestingOrchestrator class."""

    def test_orchestrator_creation(self, mock_env_vars):
        """Test TestingOrchestrator creation."""
        from src.orchestrator.graph import TestingOrchestrator

        orchestrator = TestingOrchestrator(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
        )

        assert orchestrator.codebase_path == "/path/to/app"
        assert orchestrator.app_url == "http://localhost:3000"
        assert orchestrator.pr_number is None
        assert orchestrator.changed_files == []

    def test_orchestrator_with_pr(self, mock_env_vars):
        """Test TestingOrchestrator with PR info."""
        from src.orchestrator.graph import TestingOrchestrator

        orchestrator = TestingOrchestrator(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
            pr_number=123,
            changed_files=["src/login.py"],
        )

        assert orchestrator.pr_number == 123
        assert orchestrator.changed_files == ["src/login.py"]

    def test_orchestrator_with_settings(self, mock_env_vars):
        """Test TestingOrchestrator with custom settings."""
        from src.orchestrator.graph import TestingOrchestrator
        from src.config import Settings

        settings = Settings()
        settings.max_iterations = 50

        orchestrator = TestingOrchestrator(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
            settings=settings,
        )

        assert orchestrator.settings.max_iterations == 50

    def test_get_run_summary(self, mock_env_vars):
        """Test get_run_summary method."""
        from src.orchestrator.graph import TestingOrchestrator

        orchestrator = TestingOrchestrator(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
        )

        state = {
            "run_id": "test-run-123",
            "started_at": "2024-01-01T00:00:00",
            "passed_count": 8,
            "failed_count": 2,
            "skipped_count": 0,
            "total_cost": 0.05,
            "iteration": 10,
            "error": None,
        }

        summary = orchestrator.get_run_summary(state)

        assert summary["run_id"] == "test-run-123"
        assert summary["total_tests"] == 10
        assert summary["passed"] == 8
        assert summary["failed"] == 2
        assert summary["pass_rate"] == 0.8
        assert summary["total_cost"] == 0.05

    def test_get_run_summary_with_error(self, mock_env_vars):
        """Test get_run_summary with error."""
        from src.orchestrator.graph import TestingOrchestrator

        orchestrator = TestingOrchestrator(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
        )

        state = {
            "run_id": "test-run-456",
            "started_at": "2024-01-01T00:00:00",
            "passed_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "total_cost": 0.0,
            "iteration": 1,
            "error": "Analysis failed",
        }

        summary = orchestrator.get_run_summary(state)

        assert summary["total_tests"] == 0
        assert summary["pass_rate"] == 0
        assert summary["error"] == "Analysis failed"
