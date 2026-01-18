"""Tests for the orchestrator state module."""



class TestTestType:
    """Tests for TestType enum."""

    def test_test_types_exist(self, mock_env_vars):
        """Test all test types are defined."""
        from src.orchestrator.state import TestType

        assert TestType.UI == "ui"
        assert TestType.API == "api"
        assert TestType.DATABASE == "db"
        assert TestType.INTEGRATION == "integration"


class TestTestStatus:
    """Tests for TestStatus enum."""

    def test_test_statuses_exist(self, mock_env_vars):
        """Test all test statuses are defined."""
        from src.orchestrator.state import TestStatus

        assert TestStatus.PENDING == "pending"
        assert TestStatus.RUNNING == "running"
        assert TestStatus.PASSED == "passed"
        assert TestStatus.FAILED == "failed"
        assert TestStatus.SKIPPED == "skipped"
        assert TestStatus.HEALED == "healed"


class TestPriority:
    """Tests for Priority enum."""

    def test_priorities_exist(self, mock_env_vars):
        """Test all priorities are defined."""
        from src.orchestrator.state import Priority

        assert Priority.CRITICAL == "critical"
        assert Priority.HIGH == "high"
        assert Priority.MEDIUM == "medium"
        assert Priority.LOW == "low"


class TestTestSpec:
    """Tests for TestSpec dataclass."""

    def test_test_spec_creation(self, mock_env_vars):
        """Test TestSpec creation."""
        from src.orchestrator.state import Priority, TestSpec, TestType

        spec = TestSpec(
            id="test-001",
            name="Login Test",
            type=TestType.UI,
            priority=Priority.CRITICAL,
            steps=[{"action": "click", "target": "#login"}],
            assertions=[{"type": "visible", "target": "#dashboard"}],
        )

        assert spec.id == "test-001"
        assert spec.name == "Login Test"
        assert spec.type == TestType.UI
        assert spec.priority == Priority.CRITICAL
        assert spec.timeout_seconds == 120  # default

    def test_test_spec_with_all_fields(self, mock_env_vars):
        """Test TestSpec with all fields."""
        from src.orchestrator.state import Priority, TestSpec, TestType

        spec = TestSpec(
            id="test-002",
            name="Checkout Test",
            type=TestType.UI,
            priority=Priority.HIGH,
            steps=[{"action": "click"}],
            assertions=[{"type": "visible"}],
            preconditions=["User is logged in"],
            cleanup=["Clear cart"],
            tags=["checkout", "e2e"],
            timeout_seconds=300,
        )

        assert spec.preconditions == ["User is logged in"]
        assert spec.cleanup == ["Clear cart"]
        assert spec.tags == ["checkout", "e2e"]
        assert spec.timeout_seconds == 300

    def test_test_spec_to_dict(self, mock_env_vars):
        """Test TestSpec to_dict method."""
        from src.orchestrator.state import Priority, TestSpec, TestType

        spec = TestSpec(
            id="test-003",
            name="API Test",
            type=TestType.API,
            priority=Priority.MEDIUM,
            steps=[],
            assertions=[],
        )

        result = spec.to_dict()

        assert result["id"] == "test-003"
        assert result["type"] == "api"
        assert result["priority"] == "medium"
        assert result["timeout_seconds"] == 120


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_test_result_creation(self, mock_env_vars):
        """Test TestResult creation."""
        from src.orchestrator.state import TestResult, TestStatus

        result = TestResult(
            test_id="test-001",
            status=TestStatus.PASSED,
            duration_seconds=5.5,
        )

        assert result.test_id == "test-001"
        assert result.status == TestStatus.PASSED
        assert result.duration_seconds == 5.5
        assert result.error_message is None
        assert result.screenshots == []

    def test_test_result_with_error(self, mock_env_vars):
        """Test TestResult with error."""
        from src.orchestrator.state import TestResult, TestStatus

        result = TestResult(
            test_id="test-002",
            status=TestStatus.FAILED,
            duration_seconds=2.0,
            error_message="Element not found",
            assertions_failed=1,
        )

        assert result.status == TestStatus.FAILED
        assert result.error_message == "Element not found"
        assert result.assertions_failed == 1

    def test_test_result_with_screenshots(self, mock_env_vars):
        """Test TestResult with screenshots."""
        from src.orchestrator.state import TestResult, TestStatus

        result = TestResult(
            test_id="test-003",
            status=TestStatus.PASSED,
            duration_seconds=3.0,
            screenshots=["base64_screenshot_1", "base64_screenshot_2"],
        )

        assert len(result.screenshots) == 2

    def test_test_result_to_dict(self, mock_env_vars):
        """Test TestResult to_dict method."""
        from src.orchestrator.state import TestResult, TestStatus

        result = TestResult(
            test_id="test-004",
            status=TestStatus.HEALED,
            duration_seconds=4.0,
            healing_applied={"type": "selector_update"},
        )

        result_dict = result.to_dict()

        assert result_dict["test_id"] == "test-004"
        assert result_dict["status"] == "healed"
        assert result_dict["healing_applied"] == {"type": "selector_update"}


class TestFailureAnalysis:
    """Tests for FailureAnalysis dataclass."""

    def test_failure_analysis_creation(self, mock_env_vars):
        """Test FailureAnalysis creation."""
        from src.orchestrator.state import FailureAnalysis

        analysis = FailureAnalysis(
            test_id="test-001",
            failure_type="selector_changed",
            root_cause="Button ID changed from #login to #signin",
        )

        assert analysis.test_id == "test-001"
        assert analysis.failure_type == "selector_changed"
        assert analysis.confidence == 0.0  # default

    def test_failure_analysis_with_fix(self, mock_env_vars):
        """Test FailureAnalysis with suggested fix."""
        from src.orchestrator.state import FailureAnalysis

        analysis = FailureAnalysis(
            test_id="test-002",
            failure_type="timing_issue",
            root_cause="Element not loaded in time",
            suggested_fix={"type": "add_wait", "value": 2000},
            confidence=0.85,
        )

        assert analysis.suggested_fix is not None
        assert analysis.confidence == 0.85

    def test_failure_analysis_to_dict(self, mock_env_vars):
        """Test FailureAnalysis to_dict method."""
        from src.orchestrator.state import FailureAnalysis

        analysis = FailureAnalysis(
            test_id="test-003",
            failure_type="real_bug",
            root_cause="Actual application bug",
            confidence=0.95,
        )

        result = analysis.to_dict()

        assert result["test_id"] == "test-003"
        assert result["failure_type"] == "real_bug"
        assert result["confidence"] == 0.95


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_create_initial_state_basic(self, mock_env_vars):
        """Test creating initial state with basic params."""
        from src.orchestrator.state import create_initial_state

        state = create_initial_state(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
        )

        assert state["codebase_path"] == "/path/to/app"
        assert state["app_url"] == "http://localhost:3000"
        assert state["messages"] == []
        assert state["test_plan"] == []
        assert state["current_test_index"] == 0
        assert state["passed_count"] == 0
        assert state["failed_count"] == 0
        assert state["should_continue"] is True
        assert state["run_id"] is not None
        assert state["user_id"] == "anonymous"

    def test_create_initial_state_with_pr(self, mock_env_vars):
        """Test creating initial state with PR info."""
        from src.orchestrator.state import create_initial_state

        state = create_initial_state(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
            pr_number=123,
            changed_files=["src/login.py", "src/auth.py"],
        )

        assert state["pr_number"] == 123
        assert state["changed_files"] == ["src/login.py", "src/auth.py"]

    def test_create_initial_state_with_user(self, mock_env_vars):
        """Test creating initial state with user info."""
        from src.orchestrator.state import create_initial_state

        state = create_initial_state(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
            user_id="user-123",
        )

        assert state["user_id"] == "user-123"

    def test_create_initial_state_defaults(self, mock_env_vars):
        """Test initial state default values."""
        from src.orchestrator.state import create_initial_state

        state = create_initial_state(
            codebase_path="/app",
            app_url="http://localhost",
        )

        assert state["codebase_summary"] == ""
        assert state["testable_surfaces"] == []
        assert state["test_priorities"] == {}
        assert state["current_test"] is None
        assert state["test_results"] == []
        assert state["failures"] == []
        assert state["healing_queue"] == []
        assert state["screenshots"] == []
        assert state["total_input_tokens"] == 0
        assert state["total_output_tokens"] == 0
        assert state["total_cost"] == 0.0
        assert state["iteration"] == 0
        assert state["max_iterations"] == 100
        assert state["next_agent"] == "analyze_code"
        assert state["error"] is None
        assert state["security_summary"] is None

    def test_create_initial_state_session_id_matches_run_id(self, mock_env_vars):
        """Test that session_id matches run_id."""
        from src.orchestrator.state import create_initial_state

        state = create_initial_state(
            codebase_path="/app",
            app_url="http://localhost",
        )

        assert state["session_id"] == state["run_id"]
