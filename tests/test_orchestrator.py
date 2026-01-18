"""Tests for the testing orchestrator."""

from unittest.mock import MagicMock, patch


class TestTestingState:
    """Tests for TestingState and related types."""

    def test_create_initial_state(self):
        """Test creating initial state."""
        from src.orchestrator.state import create_initial_state

        state = create_initial_state(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
        )

        assert state["codebase_path"] == "/path/to/app"
        assert state["app_url"] == "http://localhost:3000"
        assert state["current_test_index"] == 0
        assert state["passed_count"] == 0
        assert state["failed_count"] == 0
        assert state["should_continue"] is True
        assert state["run_id"] is not None

    def test_create_initial_state_with_pr(self):
        """Test creating initial state with PR number."""
        from src.orchestrator.state import create_initial_state

        state = create_initial_state(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
            pr_number=123,
            changed_files=["src/app.py", "src/models.py"],
        )

        assert state["pr_number"] == 123
        assert state["changed_files"] == ["src/app.py", "src/models.py"]


class TestTestSpec:
    """Tests for TestSpec dataclass."""

    def test_test_spec_creation(self):
        """Test creating a test spec."""
        from src.orchestrator.state import Priority, TestSpec, TestType

        spec = TestSpec(
            id="test-1",
            name="Login Test",
            type=TestType.UI,
            priority=Priority.HIGH,
            steps=[{"action": "goto", "target": "/login"}],
            assertions=[{"type": "element_visible", "target": "#dashboard"}],
        )

        assert spec.id == "test-1"
        assert spec.name == "Login Test"
        assert spec.type == TestType.UI
        assert spec.priority == Priority.HIGH

    def test_test_spec_to_dict(self):
        """Test converting test spec to dict."""
        from src.orchestrator.state import Priority, TestSpec, TestType

        spec = TestSpec(
            id="test-1",
            name="Login Test",
            type=TestType.UI,
            priority=Priority.HIGH,
            steps=[{"action": "goto", "target": "/login"}],
            assertions=[{"type": "element_visible", "target": "#dashboard"}],
            tags=["smoke", "auth"],
        )

        result = spec.to_dict()

        assert result["id"] == "test-1"
        assert result["type"] == "ui"
        assert result["priority"] == "high"
        assert result["tags"] == ["smoke", "auth"]


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_test_result_passed(self):
        """Test creating a passed test result."""
        from src.orchestrator.state import TestResult, TestStatus

        result = TestResult(
            test_id="test-1",
            status=TestStatus.PASSED,
            duration_seconds=1.5,
            assertions_passed=3,
            assertions_failed=0,
        )

        assert result.status == TestStatus.PASSED
        assert result.duration_seconds == 1.5
        assert result.assertions_passed == 3
        assert result.assertions_failed == 0
        assert result.error_message is None

    def test_test_result_failed(self):
        """Test creating a failed test result."""
        from src.orchestrator.state import TestResult, TestStatus

        result = TestResult(
            test_id="test-1",
            status=TestStatus.FAILED,
            duration_seconds=2.0,
            error_message="Element not found: #button",
            assertions_passed=2,
            assertions_failed=1,
        )

        assert result.status == TestStatus.FAILED
        assert result.error_message == "Element not found: #button"

    def test_test_result_to_dict(self):
        """Test converting test result to dict."""
        from src.orchestrator.state import TestResult, TestStatus

        result = TestResult(
            test_id="test-1",
            status=TestStatus.PASSED,
            duration_seconds=1.5,
        )

        result_dict = result.to_dict()

        assert result_dict["test_id"] == "test-1"
        assert result_dict["status"] == "passed"
        assert result_dict["duration_seconds"] == 1.5


class TestFailureAnalysis:
    """Tests for FailureAnalysis dataclass."""

    def test_failure_analysis_creation(self):
        """Test creating a failure analysis."""
        from src.orchestrator.state import FailureAnalysis

        analysis = FailureAnalysis(
            test_id="test-1",
            failure_type="selector_changed",
            root_cause="Button ID changed from #submit to #submit-btn",
            confidence=0.85,
        )

        assert analysis.test_id == "test-1"
        assert analysis.failure_type == "selector_changed"
        assert analysis.confidence == 0.85

    def test_failure_analysis_with_fix(self):
        """Test failure analysis with suggested fix."""
        from src.orchestrator.state import FailureAnalysis

        analysis = FailureAnalysis(
            test_id="test-1",
            failure_type="selector_changed",
            root_cause="Button ID changed",
            suggested_fix={
                "type": "update_selector",
                "old": "#submit",
                "new": "#submit-btn",
            },
            confidence=0.9,
        )

        result = analysis.to_dict()

        assert result["suggested_fix"]["type"] == "update_selector"
        assert result["confidence"] == 0.9


class TestRoutingFunctions:
    """Tests for graph routing functions."""

    def test_route_after_analysis_success(self):
        """Test routing after successful analysis."""
        from src.orchestrator.graph import route_after_analysis

        state = {
            "testable_surfaces": [{"name": "login", "type": "ui"}],
            "error": None,
        }

        result = route_after_analysis(state)

        assert result == "plan_tests"

    def test_route_after_analysis_no_surfaces(self):
        """Test routing when no testable surfaces found."""
        from src.orchestrator.graph import route_after_analysis

        state = {
            "testable_surfaces": [],
            "error": None,
        }

        result = route_after_analysis(state)

        assert result == "report"

    def test_route_after_analysis_error(self):
        """Test routing after analysis error."""
        from src.orchestrator.graph import route_after_analysis

        state = {
            "testable_surfaces": [],
            "error": "Failed to analyze codebase",
        }

        result = route_after_analysis(state)

        assert result == "report"

    def test_route_after_execution_more_tests(self):
        """Test routing when more tests remain."""
        from src.orchestrator.graph import route_after_execution

        state = {
            "error": None,
            "healing_queue": [],
            "should_continue": True,
            "current_test_index": 0,
            "test_plan": [{"id": "test-1"}, {"id": "test-2"}],
        }

        result = route_after_execution(state)

        assert result == "execute_test"

    def test_route_after_execution_all_done(self):
        """Test routing when all tests complete."""
        from src.orchestrator.graph import route_after_execution

        state = {
            "error": None,
            "healing_queue": [],
            "should_continue": True,
            "current_test_index": 2,
            "test_plan": [{"id": "test-1"}, {"id": "test-2"}],
        }

        result = route_after_execution(state)

        assert result == "report"

    def test_should_continue_normal(self):
        """Test should_continue under normal conditions."""
        from src.orchestrator.graph import should_continue

        state = {
            "iteration": 5,
            "max_iterations": 100,
            "total_cost": 1.0,
            "error": None,
            "should_continue": True,
        }

        with patch("src.orchestrator.graph.get_settings") as mock_settings:
            mock_settings.return_value.cost_limit_per_run = 10.0
            result = should_continue(state)

        assert result is True

    def test_should_continue_max_iterations(self):
        """Test should_continue at max iterations."""
        from src.orchestrator.graph import should_continue

        state = {
            "iteration": 100,
            "max_iterations": 100,
            "total_cost": 1.0,
            "error": None,
            "should_continue": True,
        }

        result = should_continue(state)

        assert result is False

    def test_should_continue_cost_exceeded(self):
        """Test should_continue when cost limit exceeded."""
        from src.orchestrator.graph import should_continue

        state = {
            "iteration": 5,
            "max_iterations": 100,
            "total_cost": 15.0,
            "error": None,
            "should_continue": True,
        }

        with patch("src.orchestrator.graph.get_settings") as mock_settings:
            mock_settings.return_value.cost_limit_per_run = 10.0
            result = should_continue(state)

        assert result is False


class TestTestingOrchestrator:
    """Tests for TestingOrchestrator class."""

    def test_orchestrator_init(self):
        """Test orchestrator initialization."""
        with patch("src.orchestrator.graph.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()

            from src.orchestrator.graph import TestingOrchestrator

            orchestrator = TestingOrchestrator(
                codebase_path="/path/to/app",
                app_url="http://localhost:3000",
            )

            assert orchestrator.codebase_path == "/path/to/app"
            assert orchestrator.app_url == "http://localhost:3000"

    def test_get_run_summary(self):
        """Test getting run summary."""
        with patch("src.orchestrator.graph.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()

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
                "total_cost": 1.23,
                "iteration": 15,
                "error": None,
            }

            summary = orchestrator.get_run_summary(state)

            assert summary["run_id"] == "test-run-123"
            assert summary["total_tests"] == 10
            assert summary["passed"] == 8
            assert summary["failed"] == 2
            assert summary["pass_rate"] == 0.8
            assert summary["total_cost"] == 1.23
