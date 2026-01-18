"""Tests for Browser Pool Models module."""

from datetime import datetime


class TestActionType:
    """Tests for ActionType enum."""

    def test_action_type_values(self, mock_env_vars):
        """Test ActionType enum has correct values."""
        from src.browser.pool_models import ActionType

        assert ActionType.CLICK.value == "click"
        assert ActionType.FILL.value == "fill"
        assert ActionType.TYPE.value == "type"
        assert ActionType.PRESS.value == "press"
        assert ActionType.NAVIGATE.value == "navigate"
        assert ActionType.HOVER.value == "hover"
        assert ActionType.SELECT.value == "select"
        assert ActionType.WAIT.value == "wait"
        assert ActionType.SCROLL.value == "scroll"
        assert ActionType.SCREENSHOT.value == "screenshot"

    def test_action_type_is_str_enum(self, mock_env_vars):
        """Test ActionType inherits from str."""
        from src.browser.pool_models import ActionType

        assert isinstance(ActionType.CLICK, str)
        assert ActionType.CLICK == "click"


class TestBrowserType:
    """Tests for BrowserType enum."""

    def test_browser_type_values(self, mock_env_vars):
        """Test BrowserType enum values."""
        from src.browser.pool_models import BrowserType

        assert BrowserType.CHROMIUM.value == "chromium"
        assert BrowserType.CHROME.value == "chrome"
        assert BrowserType.FIREFOX.value == "firefox"
        assert BrowserType.WEBKIT.value == "webkit"
        assert BrowserType.EDGE.value == "edge"

    def test_browser_type_is_str_enum(self, mock_env_vars):
        """Test BrowserType inherits from str."""
        from src.browser.pool_models import BrowserType

        assert isinstance(BrowserType.CHROMIUM, str)


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_execution_mode_values(self, mock_env_vars):
        """Test ExecutionMode enum values."""
        from src.browser.pool_models import ExecutionMode

        assert ExecutionMode.DOM.value == "dom"
        assert ExecutionMode.VISION.value == "vision"
        assert ExecutionMode.HYBRID.value == "hybrid"
        assert ExecutionMode.CACHED.value == "cached"


class TestElementInfo:
    """Tests for ElementInfo dataclass."""

    def test_element_info_creation(self, mock_env_vars):
        """Test creating ElementInfo with required fields."""
        from src.browser.pool_models import ElementInfo

        element = ElementInfo(
            selector="#login-btn",
            type="button",
            tag_name="button",
        )

        assert element.selector == "#login-btn"
        assert element.type == "button"
        assert element.tag_name == "button"
        assert element.text == ""
        assert element.confidence == 0.5

    def test_element_info_full_creation(self, mock_env_vars):
        """Test creating ElementInfo with all fields."""
        from src.browser.pool_models import ElementInfo

        element = ElementInfo(
            selector="#email-input",
            type="input",
            tag_name="input",
            text="Enter email",
            value="test@example.com",
            placeholder="Email address",
            description="Email input field",
            confidence=0.95,
            bounds={"x": 100, "y": 200, "width": 300, "height": 40},
            attributes={"id": "email-input", "type": "email"},
        )

        assert element.selector == "#email-input"
        assert element.value == "test@example.com"
        assert element.confidence == 0.95
        assert element.bounds["width"] == 300

    def test_element_info_to_dict(self, mock_env_vars):
        """Test ElementInfo to_dict method."""
        from src.browser.pool_models import ElementInfo

        element = ElementInfo(
            selector="#btn",
            type="button",
            tag_name="button",
            text="Submit",
            confidence=0.9,
        )

        result = element.to_dict()

        assert result["selector"] == "#btn"
        assert result["type"] == "button"
        assert result["tagName"] == "button"
        assert result["text"] == "Submit"
        assert result["confidence"] == 0.9


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_action_result_success(self, mock_env_vars):
        """Test creating a successful ActionResult."""
        from src.browser.pool_models import ActionResult

        result = ActionResult(
            action="click",
            success=True,
            selector="#btn",
            duration_ms=150,
        )

        assert result.action == "click"
        assert result.success is True
        assert result.error is None

    def test_action_result_failure(self, mock_env_vars):
        """Test creating a failed ActionResult."""
        from src.browser.pool_models import ActionResult

        result = ActionResult(
            action="fill",
            success=False,
            selector="#input",
            error="Element not found",
        )

        assert result.success is False
        assert result.error == "Element not found"

    def test_action_result_to_dict(self, mock_env_vars):
        """Test ActionResult to_dict method."""
        from src.browser.pool_models import ActionResult

        result = ActionResult(
            action="navigate",
            success=True,
            url="https://example.com",
        )

        data = result.to_dict()

        assert data["action"] == "navigate"
        assert data["success"] is True
        assert data["url"] == "https://example.com"


class TestObserveResult:
    """Tests for ObserveResult dataclass."""

    def test_observe_result_creation(self, mock_env_vars):
        """Test creating ObserveResult."""
        from src.browser.pool_models import ElementInfo, ObserveResult

        elements = [
            ElementInfo(selector="#btn1", type="button", tag_name="button"),
            ElementInfo(selector="#input1", type="input", tag_name="input"),
        ]

        result = ObserveResult(
            success=True,
            url="https://example.com",
            title="Test Page",
            elements=elements,
        )

        assert result.success is True
        assert len(result.elements) == 2
        assert result.title == "Test Page"

    def test_observe_result_actions_alias(self, mock_env_vars):
        """Test ObserveResult.actions alias for elements."""
        from src.browser.pool_models import ElementInfo, ObserveResult

        elements = [ElementInfo(selector="#btn", type="button", tag_name="button")]
        result = ObserveResult(success=True, url="https://example.com", elements=elements)

        # actions should return the same as elements
        assert result.actions == result.elements

    def test_observe_result_error(self, mock_env_vars):
        """Test ObserveResult with error."""
        from src.browser.pool_models import ObserveResult

        result = ObserveResult(
            success=False,
            url="https://example.com",
            error="Page not found",
        )

        assert result.success is False
        assert result.error == "Page not found"
        assert result.elements == []

    def test_observe_result_to_dict(self, mock_env_vars):
        """Test ObserveResult to_dict method."""
        from src.browser.pool_models import ElementInfo, ObserveResult

        result = ObserveResult(
            success=True,
            url="https://example.com",
            title="Test",
            elements=[ElementInfo(selector="#a", type="link", tag_name="a")],
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["url"] == "https://example.com"
        assert len(data["actions"]) == 1


class TestActResult:
    """Tests for ActResult dataclass."""

    def test_act_result_creation(self, mock_env_vars):
        """Test creating ActResult."""
        from src.browser.pool_models import ActionResult, ActResult, ExecutionMode

        actions = [ActionResult(action="click", success=True, selector="#btn")]

        result = ActResult(
            success=True,
            message="Action completed",
            actions=actions,
            url="https://example.com",
            execution_mode=ExecutionMode.DOM,
        )

        assert result.success is True
        assert len(result.actions) == 1
        assert result.execution_mode == ExecutionMode.DOM

    def test_act_result_with_screenshot(self, mock_env_vars):
        """Test ActResult with screenshot."""
        from src.browser.pool_models import ActResult

        result = ActResult(
            success=True,
            url="https://example.com",
            screenshot="base64_encoded_image_data",
        )

        assert result.screenshot == "base64_encoded_image_data"

    def test_act_result_to_dict(self, mock_env_vars):
        """Test ActResult to_dict method."""
        from src.browser.pool_models import ActResult, ExecutionMode

        result = ActResult(
            success=True,
            message="Done",
            url="https://example.com",
            execution_mode=ExecutionMode.VISION,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["execution_mode"] == "vision"


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_creation(self, mock_env_vars):
        """Test creating StepResult."""
        from src.browser.pool_models import ActionResult, StepResult

        step = StepResult(
            step_index=0,
            instruction="Click the login button",
            success=True,
            actions=[ActionResult(action="click", success=True)],
            duration_ms=500,
        )

        assert step.step_index == 0
        assert step.instruction == "Click the login button"
        assert step.success is True
        assert step.duration_ms == 500

    def test_step_result_failure(self, mock_env_vars):
        """Test StepResult with failure."""
        from src.browser.pool_models import StepResult

        step = StepResult(
            step_index=1,
            instruction="Fill email field",
            success=False,
            error="Element not visible",
        )

        assert step.success is False
        assert step.error == "Element not visible"

    def test_step_result_to_dict(self, mock_env_vars):
        """Test StepResult to_dict method."""
        from src.browser.pool_models import ExecutionMode, StepResult

        step = StepResult(
            step_index=0,
            instruction="Navigate to page",
            success=True,
            duration_ms=1000,
            execution_mode=ExecutionMode.DOM,
        )

        data = step.to_dict()

        assert data["stepIndex"] == 0
        assert data["instruction"] == "Navigate to page"
        assert data["execution_mode"] == "dom"


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_test_result_creation(self, mock_env_vars):
        """Test creating TestResult."""
        from src.browser.pool_models import StepResult, TestResult

        steps = [
            StepResult(step_index=0, instruction="step1", success=True, duration_ms=100),
            StepResult(step_index=1, instruction="step2", success=True, duration_ms=200),
            StepResult(step_index=2, instruction="step3", success=False, duration_ms=150),
        ]

        result = TestResult(success=False, steps=steps)

        # __post_init__ should calculate totals
        assert result.total_steps == 3
        assert result.passed_steps == 2
        assert result.failed_steps == 1
        assert result.total_duration_ms == 450

    def test_test_result_all_passed(self, mock_env_vars):
        """Test TestResult when all steps pass."""
        from src.browser.pool_models import StepResult, TestResult

        steps = [
            StepResult(step_index=0, instruction="step1", success=True, duration_ms=100),
            StepResult(step_index=1, instruction="step2", success=True, duration_ms=100),
        ]

        result = TestResult(success=True, steps=steps)

        assert result.passed_steps == 2
        assert result.failed_steps == 0

    def test_test_result_empty_steps(self, mock_env_vars):
        """Test TestResult with empty steps."""
        from src.browser.pool_models import TestResult

        result = TestResult(success=False, error="No steps provided")

        assert result.total_steps == 0
        assert result.passed_steps == 0
        assert result.failed_steps == 0

    def test_test_result_to_dict(self, mock_env_vars):
        """Test TestResult to_dict method."""
        from src.browser.pool_models import StepResult, TestResult

        steps = [
            StepResult(step_index=0, instruction="step1", success=True, duration_ms=100),
        ]

        result = TestResult(success=True, steps=steps)
        data = result.to_dict()

        assert data["success"] is True
        assert data["summary"]["total"] == 1
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 0


class TestExtractResult:
    """Tests for ExtractResult dataclass."""

    def test_extract_result_creation(self, mock_env_vars):
        """Test creating ExtractResult."""
        from src.browser.pool_models import ExtractResult

        result = ExtractResult(
            success=True,
            data={"title": "Product", "price": 99.99},
            url="https://example.com",
        )

        assert result.success is True
        assert result.data["title"] == "Product"
        assert result.data["price"] == 99.99

    def test_extract_result_failure(self, mock_env_vars):
        """Test ExtractResult with failure."""
        from src.browser.pool_models import ExtractResult

        result = ExtractResult(
            success=False,
            url="https://example.com",
            error="Failed to extract data",
        )

        assert result.success is False
        assert result.error == "Failed to extract data"
        assert result.data == {}

    def test_extract_result_to_dict(self, mock_env_vars):
        """Test ExtractResult to_dict method."""
        from src.browser.pool_models import ExtractResult

        result = ExtractResult(
            success=True,
            data={"name": "Test"},
            url="https://example.com",
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["data"]["name"] == "Test"


class TestSessionInfo:
    """Tests for SessionInfo dataclass."""

    def test_session_info_creation(self, mock_env_vars):
        """Test creating SessionInfo."""
        from src.browser.pool_models import BrowserType, SessionInfo

        session = SessionInfo(
            session_id="session-123",
            pod_name="browser-pod-1",
            pod_ip="10.0.0.1",
        )

        assert session.session_id == "session-123"
        assert session.pod_name == "browser-pod-1"
        assert session.browser_type == BrowserType.CHROMIUM
        assert session.is_active is True

    def test_session_info_custom_browser(self, mock_env_vars):
        """Test SessionInfo with custom browser type."""
        from src.browser.pool_models import BrowserType, SessionInfo

        session = SessionInfo(
            session_id="session-456",
            pod_name="browser-pod-2",
            pod_ip="10.0.0.2",
            browser_type=BrowserType.FIREFOX,
            is_active=False,
        )

        assert session.browser_type == BrowserType.FIREFOX
        assert session.is_active is False


class TestPoolHealth:
    """Tests for PoolHealth dataclass."""

    def test_pool_health_healthy(self, mock_env_vars):
        """Test creating healthy PoolHealth."""
        from src.browser.pool_models import PoolHealth

        health = PoolHealth(
            healthy=True,
            total_pods=10,
            available_pods=8,
            active_sessions=2,
            pool_url="http://pool.example.com",
        )

        assert health.healthy is True
        assert health.total_pods == 10
        assert health.available_pods == 8
        assert health.active_sessions == 2

    def test_pool_health_unhealthy(self, mock_env_vars):
        """Test creating unhealthy PoolHealth."""
        from src.browser.pool_models import PoolHealth

        health = PoolHealth(healthy=False, pool_url="http://pool.example.com")

        assert health.healthy is False
        assert health.total_pods == 0

    def test_pool_health_to_dict(self, mock_env_vars):
        """Test PoolHealth to_dict method."""
        from src.browser.pool_models import PoolHealth

        health = PoolHealth(
            healthy=True,
            total_pods=5,
            available_pods=3,
            active_sessions=2,
            pool_url="http://pool.example.com",
        )

        data = health.to_dict()

        assert data["healthy"] is True
        assert data["totalPods"] == 5
        assert data["availablePods"] == 3
        assert data["activeSessions"] == 2


class TestBrowserPoolConfig:
    """Tests for BrowserPoolConfig dataclass."""

    def test_config_default_values(self, mock_env_vars):
        """Test BrowserPoolConfig with default values."""
        from src.browser.pool_models import BrowserPoolConfig, BrowserType

        config = BrowserPoolConfig(pool_url="http://pool.example.com")

        assert config.pool_url == "http://pool.example.com"
        assert config.timeout_ms == 60000
        assert config.retry_count == 3
        assert config.retry_delay_ms == 1000
        assert config.default_browser == BrowserType.CHROMIUM
        assert config.capture_screenshots is True
        assert config.enable_caching is True
        assert config.enable_self_healing is True
        assert config.vision_fallback_enabled is True

    def test_config_custom_values(self, mock_env_vars):
        """Test BrowserPoolConfig with custom values."""
        from src.browser.pool_models import BrowserPoolConfig, BrowserType

        config = BrowserPoolConfig(
            pool_url="http://custom.pool.com",
            timeout_ms=30000,
            retry_count=5,
            retry_delay_ms=2000,
            default_viewport=(1280, 720),
            default_browser=BrowserType.FIREFOX,
            capture_screenshots=False,
            enable_caching=False,
            enable_self_healing=False,
            vision_fallback_enabled=False,
            vision_fallback_timeout_ms=20000,
        )

        assert config.timeout_ms == 30000
        assert config.retry_count == 5
        assert config.default_viewport == (1280, 720)
        assert config.default_browser == BrowserType.FIREFOX
        assert config.capture_screenshots is False

    def test_config_to_dict(self, mock_env_vars):
        """Test BrowserPoolConfig to_dict method."""
        from src.browser.pool_models import BrowserPoolConfig

        config = BrowserPoolConfig(
            pool_url="http://pool.example.com",
            default_viewport=(1920, 1080),
        )

        data = config.to_dict()

        assert data["poolUrl"] == "http://pool.example.com"
        assert data["timeoutMs"] == 60000
        assert data["retryCount"] == 3
        assert data["defaultViewport"]["width"] == 1920
        assert data["defaultViewport"]["height"] == 1080
        assert data["defaultBrowser"] == "chromium"


class TestTimestampFields:
    """Tests for timestamp fields in dataclasses."""

    def test_observe_result_timestamp(self, mock_env_vars):
        """Test ObserveResult has timestamp."""
        from src.browser.pool_models import ObserveResult

        result = ObserveResult(success=True, url="https://example.com")

        assert result.timestamp is not None
        # Should be ISO format
        datetime.fromisoformat(result.timestamp)

    def test_act_result_timestamp(self, mock_env_vars):
        """Test ActResult has timestamp."""
        from src.browser.pool_models import ActResult

        result = ActResult(success=True)

        assert result.timestamp is not None
        datetime.fromisoformat(result.timestamp)

    def test_test_result_timestamp(self, mock_env_vars):
        """Test TestResult has timestamp."""
        from src.browser.pool_models import TestResult

        result = TestResult(success=True)

        assert result.timestamp is not None
        datetime.fromisoformat(result.timestamp)

    def test_extract_result_timestamp(self, mock_env_vars):
        """Test ExtractResult has timestamp."""
        from src.browser.pool_models import ExtractResult

        result = ExtractResult(success=True)

        assert result.timestamp is not None
        datetime.fromisoformat(result.timestamp)

    def test_pool_health_timestamp(self, mock_env_vars):
        """Test PoolHealth has timestamp."""
        from src.browser.pool_models import PoolHealth

        health = PoolHealth(healthy=True)

        assert health.timestamp is not None
        datetime.fromisoformat(health.timestamp)

    def test_session_info_timestamps(self, mock_env_vars):
        """Test SessionInfo has created_at and last_activity."""
        from src.browser.pool_models import SessionInfo

        session = SessionInfo(
            session_id="test-123",
            pod_name="pod-1",
            pod_ip="10.0.0.1",
        )

        assert session.created_at is not None
        assert session.last_activity is not None
        datetime.fromisoformat(session.created_at)
        datetime.fromisoformat(session.last_activity)
