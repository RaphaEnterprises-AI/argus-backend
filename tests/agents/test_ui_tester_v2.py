"""Tests for the UI Tester V2 agent.

This module tests the UITesterAgentV2 which uses the unified BrowserPoolClient
for browser automation with automatic vision fallback.
"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ==============================================================================
# StepResult Dataclass Tests
# ==============================================================================


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_success(self, mock_env_vars):
        """Test creating a successful StepResult."""
        from src.agents.ui_tester_v2 import StepResult

        result = StepResult(
            step_index=0,
            action="click",
            success=True,
            duration_ms=100,
        )

        assert result.step_index == 0
        assert result.action == "click"
        assert result.success is True
        assert result.duration_ms == 100
        assert result.error is None
        assert result.screenshot is None
        assert result.execution_mode == "dom"
        assert result.fallback_triggered is False

    def test_step_result_failure(self, mock_env_vars):
        """Test creating a failed StepResult."""
        from src.agents.ui_tester_v2 import StepResult

        result = StepResult(
            step_index=1,
            action="fill",
            success=False,
            duration_ms=50,
            error="Element not found",
            execution_mode="vision",
            fallback_triggered=True,
        )

        assert result.success is False
        assert result.error == "Element not found"
        assert result.execution_mode == "vision"
        assert result.fallback_triggered is True

    def test_step_result_with_screenshot(self, mock_env_vars):
        """Test StepResult with screenshot data."""
        from src.agents.ui_tester_v2 import StepResult

        screenshot_data = b"fake_screenshot_data"
        result = StepResult(
            step_index=0,
            action="screenshot",
            success=True,
            duration_ms=200,
            screenshot=screenshot_data,
        )

        assert result.screenshot == screenshot_data

    def test_step_result_to_dict(self, mock_env_vars):
        """Test StepResult serialization to dict."""
        from src.agents.ui_tester_v2 import StepResult

        result = StepResult(
            step_index=0,
            action="click",
            success=True,
            duration_ms=100,
            screenshot=b"fake_data",
            execution_mode="hybrid",
            fallback_triggered=True,
        )

        data = result.to_dict()

        assert data["step_index"] == 0
        assert data["action"] == "click"
        assert data["success"] is True
        assert data["duration_ms"] == 100
        assert data["error"] is None
        assert data["has_screenshot"] is True
        assert data["execution_mode"] == "hybrid"
        assert data["fallback_triggered"] is True

    def test_step_result_to_dict_no_screenshot(self, mock_env_vars):
        """Test StepResult serialization without screenshot."""
        from src.agents.ui_tester_v2 import StepResult

        result = StepResult(
            step_index=0,
            action="click",
            success=True,
            duration_ms=100,
        )

        data = result.to_dict()

        assert data["has_screenshot"] is False


# ==============================================================================
# AssertionResult Dataclass Tests
# ==============================================================================


class TestAssertionResult:
    """Tests for AssertionResult dataclass."""

    def test_assertion_result_passed(self, mock_env_vars):
        """Test passed assertion result."""
        from src.agents.ui_tester_v2 import AssertionResult

        result = AssertionResult(
            type="element_visible",
            target="#button",
            expected="visible",
            actual="visible",
            passed=True,
        )

        assert result.type == "element_visible"
        assert result.target == "#button"
        assert result.expected == "visible"
        assert result.actual == "visible"
        assert result.passed is True
        assert result.error is None

    def test_assertion_result_failed(self, mock_env_vars):
        """Test failed assertion result."""
        from src.agents.ui_tester_v2 import AssertionResult

        result = AssertionResult(
            type="text_equals",
            target="#title",
            expected="Hello",
            actual="Goodbye",
            passed=False,
            error="Text mismatch",
        )

        assert result.passed is False
        assert result.expected == "Hello"
        assert result.actual == "Goodbye"
        assert result.error == "Text mismatch"

    def test_assertion_result_to_dict(self, mock_env_vars):
        """Test AssertionResult serialization."""
        from src.agents.ui_tester_v2 import AssertionResult

        result = AssertionResult(
            type="url_contains",
            target=None,
            expected="/dashboard",
            actual="/dashboard?tab=overview",
            passed=True,
        )

        data = result.to_dict()

        assert data["type"] == "url_contains"
        assert data["target"] is None
        assert data["expected"] == "/dashboard"
        assert data["actual"] == "/dashboard?tab=overview"
        assert data["passed"] is True
        assert data["error"] is None


# ==============================================================================
# UITestResult Dataclass Tests
# ==============================================================================


class TestUITestResult:
    """Tests for UITestResult dataclass."""

    def test_ui_test_result_basic(self, mock_env_vars):
        """Test creating basic UITestResult."""
        from src.agents.ui_tester_v2 import UITestResult

        result = UITestResult(
            test_id="test-001",
            test_name="Login Test",
            status="passed",
            total_duration_ms=5000,
        )

        assert result.test_id == "test-001"
        assert result.test_name == "Login Test"
        assert result.status == "passed"
        assert result.total_duration_ms == 5000
        assert result.step_results == []
        assert result.assertion_results == []
        assert result.error_message is None
        assert result.execution_mode == "pool"
        assert result.fallback_count == 0
        assert result.fallback_rate == 0.0

    def test_ui_test_result_with_steps(self, mock_env_vars):
        """Test UITestResult with step results."""
        from src.agents.ui_tester_v2 import StepResult, UITestResult

        steps = [
            StepResult(0, "goto", True, 100),
            StepResult(1, "fill", True, 50),
            StepResult(2, "click", True, 75),
        ]

        result = UITestResult(
            test_id="test-001",
            test_name="Login Test",
            status="passed",
            step_results=steps,
            total_duration_ms=225,
        )

        assert len(result.step_results) == 3
        assert result.total_duration_ms == 225

    def test_ui_test_result_with_assertions(self, mock_env_vars):
        """Test UITestResult with assertion results."""
        from src.agents.ui_tester_v2 import AssertionResult, UITestResult

        assertions = [
            AssertionResult("url_contains", None, "/dashboard", "/dashboard", True),
            AssertionResult("element_visible", ".welcome", None, None, True),
        ]

        result = UITestResult(
            test_id="test-001",
            test_name="Login Test",
            status="passed",
            assertion_results=assertions,
            total_duration_ms=1000,
        )

        assert len(result.assertion_results) == 2

    def test_ui_test_result_failed(self, mock_env_vars):
        """Test failed UITestResult."""
        from src.agents.ui_tester_v2 import UITestResult

        result = UITestResult(
            test_id="test-002",
            test_name="Checkout Test",
            status="failed",
            total_duration_ms=3000,
            error_message="Button not clickable",
            failure_screenshot=b"failure_screenshot_data",
        )

        assert result.status == "failed"
        assert result.error_message == "Button not clickable"
        assert result.failure_screenshot == b"failure_screenshot_data"

    def test_ui_test_result_error_status(self, mock_env_vars):
        """Test UITestResult with error status."""
        from src.agents.ui_tester_v2 import UITestResult

        result = UITestResult(
            test_id="test-003",
            test_name="Error Test",
            status="error",
            total_duration_ms=500,
            error_message="Connection refused",
        )

        assert result.status == "error"
        assert result.error_message == "Connection refused"

    def test_ui_test_result_with_fallbacks(self, mock_env_vars):
        """Test UITestResult with fallback tracking."""
        from src.agents.ui_tester_v2 import StepResult, UITestResult

        steps = [
            StepResult(0, "goto", True, 100, execution_mode="dom", fallback_triggered=False),
            StepResult(1, "click", True, 200, execution_mode="vision", fallback_triggered=True),
            StepResult(2, "fill", True, 150, execution_mode="vision", fallback_triggered=True),
        ]

        result = UITestResult(
            test_id="test-001",
            test_name="Test with Fallbacks",
            status="passed",
            step_results=steps,
            total_duration_ms=450,
            fallback_count=2,
            fallback_rate=2 / 3,  # ~0.667
        )

        assert result.fallback_count == 2
        assert abs(result.fallback_rate - 0.667) < 0.01

    def test_ui_test_result_to_dict(self, mock_env_vars):
        """Test UITestResult serialization."""
        from src.agents.ui_tester_v2 import StepResult, UITestResult

        result = UITestResult(
            test_id="test-001",
            test_name="Login Test",
            status="passed",
            step_results=[
                StepResult(0, "goto", True, 100),
                StepResult(1, "fill", True, 50),
            ],
            total_duration_ms=150,
            final_screenshot=b"final_screenshot_data",
            execution_mode="pool",
            fallback_count=0,
            fallback_rate=0.0,
        )

        data = result.to_dict()

        assert data["test_id"] == "test-001"
        assert data["test_name"] == "Login Test"
        assert data["status"] == "passed"
        assert len(data["step_results"]) == 2
        assert data["total_duration_ms"] == 150
        assert data["has_final_screenshot"] is True
        assert data["has_failure_screenshot"] is False
        assert data["execution_mode"] == "pool"
        assert data["fallback_count"] == 0
        assert data["fallback_rate"] == 0.0

    def test_ui_test_result_to_dict_empty_results(self, mock_env_vars):
        """Test UITestResult serialization with empty results."""
        from src.agents.ui_tester_v2 import UITestResult

        result = UITestResult(
            test_id="test-001",
            test_name="Empty Test",
            status="passed",
        )

        data = result.to_dict()

        assert data["step_results"] == []
        assert data["assertion_results"] == []
        assert data["has_final_screenshot"] is False


# ==============================================================================
# UITesterAgentV2 Initialization Tests
# ==============================================================================


class TestUITesterAgentV2Initialization:
    """Tests for UITesterAgentV2 initialization."""

    def test_agent_initialization_default(self, mock_env_vars):
        """Test UITesterAgentV2 default initialization."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2()

        assert agent._pool_client is None
        assert agent._owns_client is False

    def test_agent_initialization_with_pool_client(self, mock_env_vars):
        """Test UITesterAgentV2 initialization with provided pool client."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        mock_pool_client = MagicMock()
        agent = UITesterAgentV2(pool_client=mock_pool_client)

        assert agent._pool_client is mock_pool_client
        assert agent._owns_client is False

    def test_agent_capabilities(self, mock_env_vars):
        """Test that agent has correct capabilities declared."""
        from src.agents.base import AgentCapability
        from src.agents.ui_tester_v2 import UITesterAgentV2

        assert AgentCapability.BROWSER_AUTOMATION in UITesterAgentV2.CAPABILITIES
        assert AgentCapability.SCREENSHOT in UITesterAgentV2.CAPABILITIES
        assert AgentCapability.DOM_ANALYSIS in UITesterAgentV2.CAPABILITIES

    def test_get_system_prompt(self, mock_env_vars):
        """Test system prompt generation."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2()
        prompt = agent._get_system_prompt()

        # Should contain UI testing related content
        assert "ui" in prompt.lower() or "visual" in prompt.lower()
        assert "test" in prompt.lower() or "verify" in prompt.lower()

    @pytest.mark.asyncio
    async def test_ensure_client_creates_new_client(self, mock_env_vars):
        """Test _ensure_client creates a new client when none provided."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2()

        # Patch at the source module where it's imported inside the method
        with patch("src.browser.pool_client.BrowserPoolClient") as MockPoolClient:
            mock_instance = AsyncMock()
            MockPoolClient.return_value = mock_instance

            client = await agent._ensure_client()

            MockPoolClient.assert_called_once()
            mock_instance._ensure_client.assert_called_once()
            assert agent._owns_client is True

    @pytest.mark.asyncio
    async def test_ensure_client_reuses_existing_client(self, mock_env_vars):
        """Test _ensure_client reuses existing client."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        mock_pool_client = MagicMock()
        agent = UITesterAgentV2(pool_client=mock_pool_client)

        client = await agent._ensure_client()

        assert client is mock_pool_client

    @pytest.mark.asyncio
    async def test_close_owned_client(self, mock_env_vars):
        """Test close() cleans up owned client."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2()
        mock_client = AsyncMock()
        agent._pool_client = mock_client
        agent._owns_client = True

        await agent.close()

        mock_client.close.assert_called_once()
        assert agent._pool_client is None

    @pytest.mark.asyncio
    async def test_close_non_owned_client(self, mock_env_vars):
        """Test close() does not clean up non-owned client."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        mock_client = AsyncMock()
        agent = UITesterAgentV2(pool_client=mock_client)

        await agent.close()

        mock_client.close.assert_not_called()


# ==============================================================================
# UITesterAgentV2._step_to_instruction Tests
# ==============================================================================


class TestStepToInstruction:
    """Tests for UITesterAgentV2._step_to_instruction method."""

    @pytest.fixture
    def agent(self, mock_env_vars):
        """Create UITesterAgentV2 instance."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        return UITesterAgentV2()

    @pytest.mark.parametrize(
        "action,target,value,app_url,expected",
        [
            # goto action
            ("goto", "/login", None, "https://example.com", "Navigate to https://example.com/login"),
            ("goto", "https://other.com/page", None, "https://example.com", "Navigate to https://other.com/page"),
            ("goto", "/", None, "https://example.com", "Navigate to https://example.com/"),
            # click action
            ("click", "#submit-btn", None, "https://example.com", "Click on #submit-btn"),
            ("click", ".login-button", None, "https://example.com", "Click on .login-button"),
            # double_click action
            ("double_click", "#item", None, "https://example.com", "Double click on #item"),
            # fill action
            ("fill", "#email", "test@example.com", "https://example.com", "Fill 'test@example.com' into #email"),
            ("fill", "#password", "secret123", "https://example.com", "Fill 'secret123' into #password"),
            # type action
            ("type", "#search", "query text", "https://example.com", "Type 'query text' into #search"),
            # select action
            ("select", "#country", "USA", "https://example.com", "Select 'USA' from #country"),
            # hover action
            ("hover", "#menu", None, "https://example.com", "Hover over #menu"),
            # wait action with target
            ("wait", "#loader", None, "https://example.com", "Wait for #loader to be visible"),
            # wait action with value (time)
            ("wait", None, "2000", "https://example.com", "Wait 2000ms"),
            # scroll action
            ("scroll", None, "down", "https://example.com", "Scroll down"),
            ("scroll", None, "up", "https://example.com", "Scroll up"),
            ("scroll", None, None, "https://example.com", "Scroll down"),
            # press_key action
            ("press_key", "Enter", None, "https://example.com", "Press Enter key"),
            ("press_key", None, "Tab", "https://example.com", "Press Tab key"),
            # screenshot action
            ("screenshot", None, None, "https://example.com", "Take a screenshot"),
            # unknown action
            ("custom_action", "#element", "value", "https://example.com", "custom_action #element value"),
        ],
    )
    def test_step_to_instruction(self, agent, action, target, value, app_url, expected, mock_env_vars):
        """Test step to instruction conversion for various action types."""
        from src.agents.test_planner import TestStep

        step = TestStep(action=action, target=target, value=value)
        instruction = agent._step_to_instruction(step, app_url)

        assert instruction == expected

    def test_step_to_instruction_goto_with_trailing_slash(self, agent, mock_env_vars):
        """Test goto instruction with trailing slash in app_url."""
        from src.agents.test_planner import TestStep

        step = TestStep(action="goto", target="/page")
        instruction = agent._step_to_instruction(step, "https://example.com/")

        assert instruction == "Navigate to https://example.com/page"

    def test_step_to_instruction_empty_values(self, agent, mock_env_vars):
        """Test step instruction with empty target and value."""
        from src.agents.test_planner import TestStep

        step = TestStep(action="unknown", target=None, value=None)
        instruction = agent._step_to_instruction(step, "https://example.com")

        assert instruction == "unknown"

    def test_step_to_instruction_case_insensitive(self, agent, mock_env_vars):
        """Test that action matching is case insensitive."""
        from src.agents.test_planner import TestStep

        step = TestStep(action="CLICK", target="#button")
        instruction = agent._step_to_instruction(step, "https://example.com")

        assert instruction == "Click on #button"

        step = TestStep(action="Fill", target="#input", value="text")
        instruction = agent._step_to_instruction(step, "https://example.com")

        assert instruction == "Fill 'text' into #input"


# ==============================================================================
# UITesterAgentV2.execute Tests
# ==============================================================================


class TestUITesterAgentV2Execute:
    """Tests for UITesterAgentV2.execute method."""

    @pytest.fixture
    def mock_pool_client(self):
        """Create a mock BrowserPoolClient."""
        from src.browser.pool_models import ExecutionMode, PoolHealth
        from src.browser.pool_models import StepResult as PoolStepResult
        from src.browser.pool_models import TestResult

        mock_client = AsyncMock()

        # Mock health check
        mock_client.health.return_value = PoolHealth(
            healthy=True,
            total_pods=3,
            available_pods=2,
        )

        # Default successful test result
        mock_client.test.return_value = TestResult(
            success=True,
            steps=[
                PoolStepResult(
                    step_index=0,
                    instruction="Navigate to https://example.com/login",
                    success=True,
                    duration_ms=100,
                    execution_mode=ExecutionMode.DOM,
                ),
                PoolStepResult(
                    step_index=1,
                    instruction="Fill 'test@example.com' into #email",
                    success=True,
                    duration_ms=50,
                    execution_mode=ExecutionMode.DOM,
                ),
            ],
            final_screenshot=base64.b64encode(b"final_screenshot").decode(),
        )

        return mock_client

    @pytest.fixture
    def sample_test_spec_v2(self):
        """Sample test spec for V2 agent."""
        return {
            "id": "test-001",
            "name": "Login Flow Test",
            "type": "ui",
            "priority": "high",
            "description": "Test the login flow",
            "steps": [
                {"action": "goto", "target": "/login", "value": None, "timeout": 5000},
                {"action": "fill", "target": "#email", "value": "test@example.com", "timeout": 5000},
            ],
            "assertions": [],
        }

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_env_vars, mock_pool_client, sample_test_spec_v2):
        """Test successful test execution."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        result = await agent.execute(
            test_spec=sample_test_spec_v2,
            app_url="https://example.com",
        )

        assert result.success is True
        assert result.data is not None
        assert result.data.status == "passed"
        assert result.data.test_id == "test-001"
        assert result.data.test_name == "Login Flow Test"
        assert len(result.data.step_results) == 2

    @pytest.mark.asyncio
    async def test_execute_with_testspec_object(self, mock_env_vars, mock_pool_client):
        """Test execution with TestSpec object instead of dict."""
        from src.agents.test_planner import TestAssertion, TestSpec
        from src.agents.test_planner import TestStep as PlannerTestStep
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        test_spec = TestSpec(
            id="test-002",
            name="Direct TestSpec Test",
            type="ui",
            priority="high",
            description="Test with TestSpec object",
            steps=[
                PlannerTestStep(action="goto", target="/"),
                PlannerTestStep(action="click", target="#btn"),
            ],
            assertions=[],
        )

        result = await agent.execute(
            test_spec=test_spec,
            app_url="https://example.com",
        )

        assert result.success is True
        assert result.data.test_id == "test-002"

    @pytest.mark.asyncio
    async def test_execute_failed_test(self, mock_env_vars, mock_pool_client, sample_test_spec_v2):
        """Test handling of failed test execution."""
        from src.browser.pool_models import ExecutionMode
        from src.browser.pool_models import StepResult as PoolStepResult
        from src.browser.pool_models import TestResult

        from src.agents.ui_tester_v2 import UITesterAgentV2

        # Configure mock for failed test
        mock_pool_client.test.return_value = TestResult(
            success=False,
            steps=[
                PoolStepResult(
                    step_index=0,
                    instruction="Navigate to https://example.com/login",
                    success=True,
                    duration_ms=100,
                    execution_mode=ExecutionMode.DOM,
                ),
                PoolStepResult(
                    step_index=1,
                    instruction="Fill 'test@example.com' into #email",
                    success=False,
                    duration_ms=50,
                    error="Element not found",
                    screenshot=base64.b64encode(b"failure_screenshot").decode(),
                    execution_mode=ExecutionMode.DOM,
                ),
            ],
            error="Step 1 failed: Element not found",
        )

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        result = await agent.execute(
            test_spec=sample_test_spec_v2,
            app_url="https://example.com",
        )

        assert result.success is False
        assert result.data.status == "failed"
        assert result.data.error_message == "Step 1 failed: Element not found"
        assert result.data.failure_screenshot is not None

    @pytest.mark.asyncio
    async def test_execute_with_vision_fallback(self, mock_env_vars, mock_pool_client, sample_test_spec_v2):
        """Test execution with vision fallback triggered."""
        from src.browser.pool_models import ExecutionMode
        from src.browser.pool_models import StepResult as PoolStepResult
        from src.browser.pool_models import TestResult

        from src.agents.ui_tester_v2 import UITesterAgentV2

        # Configure mock with vision fallback
        mock_pool_client.test.return_value = TestResult(
            success=True,
            steps=[
                PoolStepResult(
                    step_index=0,
                    instruction="Navigate to https://example.com/login",
                    success=True,
                    duration_ms=100,
                    execution_mode=ExecutionMode.DOM,
                ),
                PoolStepResult(
                    step_index=1,
                    instruction="Fill 'test@example.com' into #email",
                    success=True,
                    duration_ms=500,  # Longer due to vision
                    execution_mode=ExecutionMode.VISION,
                ),
            ],
        )

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        result = await agent.execute(
            test_spec=sample_test_spec_v2,
            app_url="https://example.com",
        )

        assert result.success is True
        assert result.data.fallback_count == 1
        assert result.data.fallback_rate == 0.5

    @pytest.mark.asyncio
    async def test_execute_unhealthy_pool_warning(self, mock_env_vars, mock_pool_client, sample_test_spec_v2):
        """Test that unhealthy pool triggers warning but continues."""
        from src.browser.pool_models import PoolHealth

        from src.agents.ui_tester_v2 import UITesterAgentV2

        mock_pool_client.health.return_value = PoolHealth(healthy=False)

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        # Should still attempt to execute
        result = await agent.execute(
            test_spec=sample_test_spec_v2,
            app_url="https://example.com",
        )

        # Test should still run (pool may use fallback)
        mock_pool_client.test.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, mock_env_vars, mock_pool_client, sample_test_spec_v2):
        """Test exception handling during execution."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        mock_pool_client.test.side_effect = Exception("Connection refused")

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        result = await agent.execute(
            test_spec=sample_test_spec_v2,
            app_url="https://example.com",
        )

        assert result.success is False
        assert result.data.status == "error"
        assert "Connection refused" in result.data.error_message

    @pytest.mark.asyncio
    async def test_execute_converts_step_instructions(self, mock_env_vars, mock_pool_client, sample_test_spec_v2):
        """Test that steps are converted to natural language instructions."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        await agent.execute(
            test_spec=sample_test_spec_v2,
            app_url="https://example.com",
        )

        # Check that test was called with converted instructions
        call_args = mock_pool_client.test.call_args
        assert call_args is not None
        steps = call_args.kwargs.get("steps") or call_args[1].get("steps")
        assert "Navigate to https://example.com/login" in steps
        assert "Fill 'test@example.com' into #email" in steps

    @pytest.mark.asyncio
    async def test_execute_without_screenshots(self, mock_env_vars, mock_pool_client, sample_test_spec_v2):
        """Test execution without capturing screenshots."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        await agent.execute(
            test_spec=sample_test_spec_v2,
            app_url="https://example.com",
            capture_screenshots=False,
        )

        call_args = mock_pool_client.test.call_args
        capture_screenshots = call_args.kwargs.get("capture_screenshots")
        assert capture_screenshots is False


# ==============================================================================
# UITesterAgentV2.observe Tests
# ==============================================================================


class TestUITesterAgentV2Observe:
    """Tests for UITesterAgentV2.observe method."""

    @pytest.fixture
    def mock_pool_client(self):
        """Create a mock BrowserPoolClient for observe tests."""
        from src.browser.pool_models import ElementInfo, ObserveResult

        mock_client = AsyncMock()

        mock_client.observe.return_value = ObserveResult(
            success=True,
            url="https://example.com",
            title="Example Page",
            elements=[
                ElementInfo(
                    selector="#login-btn",
                    type="button",
                    tag_name="button",
                    text="Login",
                    description="Login button",
                    confidence=0.95,
                ),
                ElementInfo(
                    selector="#email",
                    type="input",
                    tag_name="input",
                    placeholder="Enter email",
                    description="Email input field",
                    confidence=0.92,
                ),
            ],
        )

        return mock_client

    @pytest.mark.asyncio
    async def test_observe_success(self, mock_env_vars, mock_pool_client):
        """Test successful observe operation."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        result = await agent.observe(
            url="https://example.com",
            instruction="Find all interactive elements",
        )

        assert result.success is True
        assert result.data["url"] == "https://example.com"
        assert result.data["title"] == "Example Page"
        assert len(result.data["elements"]) == 2

    @pytest.mark.asyncio
    async def test_observe_without_instruction(self, mock_env_vars, mock_pool_client):
        """Test observe without specific instruction."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        result = await agent.observe(url="https://example.com")

        assert result.success is True
        mock_pool_client.observe.assert_called_once_with("https://example.com", None)

    @pytest.mark.asyncio
    async def test_observe_failure(self, mock_env_vars, mock_pool_client):
        """Test observe failure handling."""
        from src.browser.pool_models import ObserveResult

        from src.agents.ui_tester_v2 import UITesterAgentV2

        mock_pool_client.observe.return_value = ObserveResult(
            success=False,
            url="https://example.com",
            error="Page load timeout",
        )

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        result = await agent.observe(url="https://example.com")

        assert result.success is False
        assert result.data["error"] == "Page load timeout"

    @pytest.mark.asyncio
    async def test_observe_exception(self, mock_env_vars, mock_pool_client):
        """Test observe exception handling."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        mock_pool_client.observe.side_effect = Exception("Network error")

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        result = await agent.observe(url="https://example.com")

        assert result.success is False
        assert "Network error" in result.error


# ==============================================================================
# UITesterAgentV2.act Tests
# ==============================================================================


class TestUITesterAgentV2Act:
    """Tests for UITesterAgentV2.act method."""

    @pytest.fixture
    def mock_pool_client(self):
        """Create a mock BrowserPoolClient for act tests."""
        from src.browser.pool_models import ActionResult, ActResult, ExecutionMode

        mock_client = AsyncMock()

        mock_client.act.return_value = ActResult(
            success=True,
            message="Clicked the login button",
            actions=[
                ActionResult(
                    action="click",
                    success=True,
                    selector="#login-btn",
                ),
            ],
            url="https://example.com",
            screenshot=base64.b64encode(b"screenshot").decode(),
            execution_mode=ExecutionMode.DOM,
        )

        return mock_client

    @pytest.mark.asyncio
    async def test_act_success(self, mock_env_vars, mock_pool_client):
        """Test successful act operation."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        result = await agent.act(
            url="https://example.com",
            instruction="Click the login button",
        )

        assert result.success is True
        assert result.data["success"] is True
        assert result.data["message"] == "Clicked the login button"

    @pytest.mark.asyncio
    async def test_act_with_screenshot(self, mock_env_vars, mock_pool_client):
        """Test act operation with screenshot capture."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        result = await agent.act(
            url="https://example.com",
            instruction="Click the login button",
            capture_screenshot=True,
        )

        mock_pool_client.act.assert_called_once_with(
            "https://example.com",
            "Click the login button",
            screenshot=True,
        )

    @pytest.mark.asyncio
    async def test_act_without_screenshot(self, mock_env_vars, mock_pool_client):
        """Test act operation without screenshot capture."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        await agent.act(
            url="https://example.com",
            instruction="Click the login button",
            capture_screenshot=False,
        )

        mock_pool_client.act.assert_called_once_with(
            "https://example.com",
            "Click the login button",
            screenshot=False,
        )

    @pytest.mark.asyncio
    async def test_act_failure(self, mock_env_vars, mock_pool_client):
        """Test act operation failure."""
        from src.browser.pool_models import ActResult

        from src.agents.ui_tester_v2 import UITesterAgentV2

        mock_pool_client.act.return_value = ActResult(
            success=False,
            error="Element not found",
            url="https://example.com",
        )

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        result = await agent.act(
            url="https://example.com",
            instruction="Click non-existent button",
        )

        assert result.success is False
        assert result.data["error"] == "Element not found"

    @pytest.mark.asyncio
    async def test_act_exception(self, mock_env_vars, mock_pool_client):
        """Test act operation exception handling."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        mock_pool_client.act.side_effect = Exception("Connection timeout")

        agent = UITesterAgentV2(pool_client=mock_pool_client)

        result = await agent.act(
            url="https://example.com",
            instruction="Click the login button",
        )

        assert result.success is False
        assert "Connection timeout" in result.error


# ==============================================================================
# UITesterAgentV2.verify_with_vision Tests
# ==============================================================================


class TestUITesterAgentV2VerifyWithVision:
    """Tests for UITesterAgentV2.verify_with_vision method."""

    @pytest.fixture
    def agent_with_mock_client(self, mock_env_vars, mock_anthropic_client):
        """Create UITesterAgentV2 with mocked Anthropic client."""
        from src.agents.ui_tester_v2 import UITesterAgentV2

        agent = UITesterAgentV2()
        agent._client = mock_anthropic_client

        return agent

    @pytest.mark.asyncio
    async def test_verify_with_vision_success(self, agent_with_mock_client, mock_anthropic_client):
        """Test successful visual verification."""
        mock_anthropic_client.messages.create.return_value.content = [
            MagicMock(
                text='{"passed": true, "observations": ["Login button visible", "Form fields present"], "issues": [], "confidence": 0.95}'
            )
        ]

        result = await agent_with_mock_client.verify_with_vision(
            screenshot=b"fake_screenshot_data",
            verification_prompt="Verify the login page is displayed correctly",
        )

        assert result.success is True
        assert result.data["passed"] is True
        assert len(result.data["observations"]) == 2
        assert result.data["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_verify_with_vision_failure(self, agent_with_mock_client, mock_anthropic_client):
        """Test visual verification that finds issues."""
        mock_anthropic_client.messages.create.return_value.content = [
            MagicMock(
                text='{"passed": false, "observations": ["Error message displayed"], "issues": ["Login button not visible"], "confidence": 0.88}'
            )
        ]

        result = await agent_with_mock_client.verify_with_vision(
            screenshot=b"fake_screenshot_data",
            verification_prompt="Verify login was successful",
        )

        assert result.success is True  # Call succeeded
        assert result.data["passed"] is False  # But verification failed
        assert "Login button not visible" in result.data["issues"]

    @pytest.mark.asyncio
    async def test_verify_with_vision_cost_limit_exceeded(self, agent_with_mock_client):
        """Test verify_with_vision respects cost limits."""
        # Set usage above limit
        agent_with_mock_client._usage.total_cost = 100.0

        result = await agent_with_mock_client.verify_with_vision(
            screenshot=b"fake_screenshot_data",
            verification_prompt="Verify page",
        )

        assert result.success is False
        assert "Cost limit exceeded" in result.error

    @pytest.mark.asyncio
    async def test_verify_with_vision_api_error(self, agent_with_mock_client, mock_anthropic_client):
        """Test verify_with_vision handles API errors."""
        mock_anthropic_client.messages.create.side_effect = Exception("API rate limit exceeded")

        result = await agent_with_mock_client.verify_with_vision(
            screenshot=b"fake_screenshot_data",
            verification_prompt="Verify page",
        )

        assert result.success is False
        assert "API rate limit exceeded" in result.error

    @pytest.mark.asyncio
    async def test_verify_with_vision_invalid_json_response(self, agent_with_mock_client, mock_anthropic_client):
        """Test verify_with_vision handles invalid JSON response."""
        mock_anthropic_client.messages.create.return_value.content = [
            MagicMock(text="This is not valid JSON")
        ]

        result = await agent_with_mock_client.verify_with_vision(
            screenshot=b"fake_screenshot_data",
            verification_prompt="Verify page",
        )

        assert result.success is True
        # Should return fallback data
        assert result.data["passed"] is False

    @pytest.mark.asyncio
    async def test_verify_with_vision_tracks_tokens(self, agent_with_mock_client, mock_anthropic_client):
        """Test that verify_with_vision tracks token usage."""
        mock_anthropic_client.messages.create.return_value.content = [
            MagicMock(text='{"passed": true, "observations": [], "issues": [], "confidence": 0.9}')
        ]
        mock_anthropic_client.messages.create.return_value.usage = MagicMock(
            input_tokens=500,
            output_tokens=100,
        )

        result = await agent_with_mock_client.verify_with_vision(
            screenshot=b"fake_screenshot_data",
            verification_prompt="Verify page",
        )

        assert result.input_tokens == 500
        assert result.output_tokens == 100


# ==============================================================================
# Fallback Rate Calculation Tests
# ==============================================================================


class TestFallbackRateCalculation:
    """Tests for fallback rate calculation."""

    def test_fallback_rate_no_steps(self, mock_env_vars):
        """Test fallback rate with no steps."""
        from src.agents.ui_tester_v2 import UITestResult

        result = UITestResult(
            test_id="test-001",
            test_name="Empty Test",
            status="passed",
            step_results=[],
            fallback_count=0,
            fallback_rate=0.0,
        )

        assert result.fallback_rate == 0.0

    def test_fallback_rate_no_fallbacks(self, mock_env_vars):
        """Test fallback rate with no fallbacks."""
        from src.agents.ui_tester_v2 import StepResult, UITestResult

        steps = [
            StepResult(0, "goto", True, 100, execution_mode="dom", fallback_triggered=False),
            StepResult(1, "click", True, 50, execution_mode="dom", fallback_triggered=False),
            StepResult(2, "fill", True, 75, execution_mode="dom", fallback_triggered=False),
        ]

        result = UITestResult(
            test_id="test-001",
            test_name="Test",
            status="passed",
            step_results=steps,
            fallback_count=0,
            fallback_rate=0.0,
        )

        assert result.fallback_count == 0
        assert result.fallback_rate == 0.0

    def test_fallback_rate_all_fallbacks(self, mock_env_vars):
        """Test fallback rate with all steps using fallback."""
        from src.agents.ui_tester_v2 import StepResult, UITestResult

        steps = [
            StepResult(0, "goto", True, 200, execution_mode="vision", fallback_triggered=True),
            StepResult(1, "click", True, 300, execution_mode="vision", fallback_triggered=True),
        ]

        result = UITestResult(
            test_id="test-001",
            test_name="Test",
            status="passed",
            step_results=steps,
            fallback_count=2,
            fallback_rate=1.0,
        )

        assert result.fallback_count == 2
        assert result.fallback_rate == 1.0

    def test_fallback_rate_mixed(self, mock_env_vars):
        """Test fallback rate with mixed execution modes."""
        from src.agents.ui_tester_v2 import StepResult, UITestResult

        steps = [
            StepResult(0, "goto", True, 100, execution_mode="dom", fallback_triggered=False),
            StepResult(1, "click", True, 200, execution_mode="vision", fallback_triggered=True),
            StepResult(2, "fill", True, 100, execution_mode="dom", fallback_triggered=False),
            StepResult(3, "click", True, 250, execution_mode="vision", fallback_triggered=True),
        ]

        # Calculate expected rate
        expected_rate = 2 / 4  # 0.5

        result = UITestResult(
            test_id="test-001",
            test_name="Test",
            status="passed",
            step_results=steps,
            fallback_count=2,
            fallback_rate=expected_rate,
        )

        assert result.fallback_count == 2
        assert result.fallback_rate == 0.5


# ==============================================================================
# Backward Compatibility Tests
# ==============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with UITesterAgent alias."""

    def test_ui_tester_agent_alias(self, mock_env_vars):
        """Test that UITesterAgent is an alias for UITesterAgentV2."""
        from src.agents.ui_tester_v2 import UITesterAgent, UITesterAgentV2

        assert UITesterAgent is UITesterAgentV2

    def test_ui_tester_agent_instantiation(self, mock_env_vars):
        """Test that UITesterAgent can be instantiated."""
        from src.agents.ui_tester_v2 import UITesterAgent

        agent = UITesterAgent()
        assert agent is not None
