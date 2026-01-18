"""Tests for the UI Tester agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_success(self, mock_env_vars):
        """Test creating a successful StepResult."""
        from src.agents.ui_tester import StepResult

        result = StepResult(
            step_index=0,
            action="click",
            success=True,
            duration_ms=100,
        )

        assert result.success is True
        assert result.error is None

    def test_step_result_to_dict(self, mock_env_vars):
        """Test StepResult serialization."""
        from src.agents.ui_tester import StepResult

        result = StepResult(
            step_index=0,
            action="click",
            success=True,
            duration_ms=100,
            screenshot=b"fake_data",
        )

        data = result.to_dict()

        assert data["step_index"] == 0
        assert data["action"] == "click"
        assert data["has_screenshot"] is True


class TestAssertionResult:
    """Tests for AssertionResult dataclass."""

    def test_assertion_result_passed(self, mock_env_vars):
        """Test passed assertion result."""
        from src.agents.ui_tester import AssertionResult

        result = AssertionResult(
            type="element_visible",
            target="#button",
            expected="visible",
            actual="visible",
            passed=True,
        )

        assert result.passed is True

    def test_assertion_result_to_dict(self, mock_env_vars):
        """Test AssertionResult serialization."""
        from src.agents.ui_tester import AssertionResult

        result = AssertionResult(
            type="text_equals",
            target="#title",
            expected="Hello",
            actual="Hello",
            passed=True,
        )

        data = result.to_dict()

        assert data["type"] == "text_equals"
        assert data["passed"] is True


class TestUITestResult:
    """Tests for UITestResult dataclass."""

    def test_ui_test_result(self, mock_env_vars):
        """Test creating UITestResult."""
        from src.agents.ui_tester import UITestResult

        result = UITestResult(
            test_id="test-001",
            test_name="Login Test",
            status="passed",
            total_duration_ms=5000,
        )

        assert result.test_id == "test-001"
        assert result.status == "passed"

    def test_ui_test_result_to_dict(self, mock_env_vars):
        """Test UITestResult serialization."""
        from src.agents.ui_tester import StepResult, UITestResult

        result = UITestResult(
            test_id="test-001",
            test_name="Login Test",
            status="passed",
            step_results=[
                StepResult(0, "goto", True, 100),
                StepResult(1, "fill", True, 50),
            ],
            total_duration_ms=150,
        )

        data = result.to_dict()

        assert len(data["step_results"]) == 2
        assert data["status"] == "passed"


class TestUITesterAgent:
    """Tests for UITesterAgent class."""

    @pytest.fixture
    def ui_agent(self, mock_env_vars, mock_playwright_tools):
        """Create UI tester agent with mock playwright."""
        from src.agents.ui_tester import UITesterAgent

        # use_worker=False to skip Worker and use mocked Playwright directly
        agent = UITesterAgent(playwright_tools=mock_playwright_tools, use_worker=False)
        return agent

    def test_agent_initialization(self, mock_env_vars, mock_playwright_tools):
        """Test UITesterAgent initialization."""
        from src.agents.ui_tester import UITesterAgent

        agent = UITesterAgent(playwright_tools=mock_playwright_tools)

        assert agent._playwright is mock_playwright_tools

    def test_get_system_prompt(self, ui_agent):
        """Test system prompt generation."""
        prompt = ui_agent._get_system_prompt()

        # Enhanced prompt uses different terminology
        assert "ui" in prompt.lower() or "browser" in prompt.lower() or "automation" in prompt.lower()
        assert "test" in prompt.lower()

    @pytest.mark.asyncio
    async def test_execute_no_playwright(self, mock_env_vars, sample_test_spec):
        """Test execute fails without playwright tools when Worker disabled."""
        from src.agents.ui_tester import UITesterAgent

        agent = UITesterAgent(use_worker=False)  # No playwright tools, no worker

        result = await agent.execute(
            test_spec=sample_test_spec,
            app_url="https://example.com",
            use_worker=False,  # Force local Playwright mode
        )

        assert result.success is False
        assert result.error is not None
        assert "PlaywrightTools" in result.error or "playwright" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_success(self, ui_agent, sample_test_spec, mock_playwright_tools):
        """Test successful test execution."""
        result = await ui_agent.execute(
            test_spec=sample_test_spec,
            app_url="https://example.com",
            playwright_tools=mock_playwright_tools,
        )

        assert result.success is True
        assert result.data is not None
        assert result.data.status == "passed"

    @pytest.mark.asyncio
    async def test_execute_step_failure(self, ui_agent, sample_test_spec, mock_playwright_tools):
        """Test handling step failures."""
        # Make click fail
        mock_playwright_tools.click = AsyncMock(side_effect=Exception("Element not found"))

        result = await ui_agent.execute(
            test_spec=sample_test_spec,
            app_url="https://example.com",
            playwright_tools=mock_playwright_tools,
        )

        assert result.data.status == "failed"
        assert "failed" in result.data.step_results[-1].error.lower() or "not found" in result.data.step_results[-1].error.lower()

    @pytest.mark.asyncio
    async def test_execute_assertion_failure(self, mock_env_vars, mock_playwright_tools):
        """Test handling assertion failures."""
        from src.agents.ui_tester import UITesterAgent

        # use_worker=False to use mocked Playwright directly
        agent = UITesterAgent(playwright_tools=mock_playwright_tools, use_worker=False)

        # Make URL assertion fail
        mock_playwright_tools.get_current_url = AsyncMock(return_value="https://example.com/login")

        test_spec = {
            "id": "test-001",
            "name": "Test",
            "type": "ui",
            "priority": "high",
            "description": "",
            "steps": [],
            "assertions": [
                {"type": "url_contains", "target": None, "expected": "/dashboard"},
            ],
        }

        result = await agent.execute(
            test_spec=test_spec,
            app_url="https://example.com",
            playwright_tools=mock_playwright_tools,
        )

        assert result.data.status == "failed"

    @pytest.mark.asyncio
    async def test_execute_step_goto(self, ui_agent, mock_playwright_tools):
        """Test goto step execution."""
        from src.agents.test_planner import TestStep

        step = TestStep(action="goto", target="/login", value=None, timeout=5000)

        await ui_agent._execute_step(mock_playwright_tools, step, "https://example.com")

        mock_playwright_tools.goto.assert_called_once_with("https://example.com/login")

    @pytest.mark.asyncio
    async def test_execute_step_fill(self, ui_agent, mock_playwright_tools):
        """Test fill step execution."""
        from src.agents.test_planner import TestStep

        step = TestStep(action="fill", target="#email", value="test@example.com", timeout=5000)

        await ui_agent._execute_step(mock_playwright_tools, step, "https://example.com")

        mock_playwright_tools.fill.assert_called_once_with("#email", "test@example.com")

    @pytest.mark.asyncio
    async def test_execute_step_click(self, ui_agent, mock_playwright_tools):
        """Test click step execution."""
        from src.agents.test_planner import TestStep

        step = TestStep(action="click", target="#submit", value=None, timeout=5000)

        await ui_agent._execute_step(mock_playwright_tools, step, "https://example.com")

        mock_playwright_tools.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_step_unknown_action(self, ui_agent, mock_playwright_tools):
        """Test unknown action raises error."""
        from src.agents.test_planner import TestStep

        step = TestStep(action="unknown_action", target="#element", value=None, timeout=5000)

        with pytest.raises(ValueError, match="Unknown action"):
            await ui_agent._execute_step(mock_playwright_tools, step, "https://example.com")

    @pytest.mark.asyncio
    async def test_check_assertion_element_visible(self, ui_agent, mock_playwright_tools):
        """Test element_visible assertion."""
        from src.agents.test_planner import TestAssertion

        assertion = TestAssertion(type="element_visible", target="#button", expected=None)

        result = await ui_agent._check_assertion(mock_playwright_tools, assertion)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_assertion_text_contains(self, ui_agent, mock_playwright_tools):
        """Test text_contains assertion."""
        from src.agents.test_planner import TestAssertion

        mock_playwright_tools.get_text = AsyncMock(return_value="Welcome, Test User!")

        assertion = TestAssertion(type="text_contains", target="#message", expected="Welcome")

        result = await ui_agent._check_assertion(mock_playwright_tools, assertion)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_assertion_url_contains(self, ui_agent, mock_playwright_tools):
        """Test url_contains assertion."""
        from src.agents.test_planner import TestAssertion

        assertion = TestAssertion(type="url_contains", target=None, expected="/dashboard")

        result = await ui_agent._check_assertion(mock_playwright_tools, assertion)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_assertion_unknown_type(self, ui_agent, mock_playwright_tools):
        """Test unknown assertion type fails."""
        from src.agents.test_planner import TestAssertion

        assertion = TestAssertion(type="unknown_type", target="#elem", expected="value")

        result = await ui_agent._check_assertion(mock_playwright_tools, assertion)

        assert result.passed is False
        assert "Unknown assertion type" in result.error

    @pytest.mark.asyncio
    async def test_verify_with_vision(self, mock_env_vars, mock_anthropic_client):
        """Test visual verification with Claude."""
        from src.agents.ui_tester import UITesterAgent

        agent = UITesterAgent()
        agent._client = mock_anthropic_client

        # Mock response
        mock_anthropic_client.messages.create.return_value.content = [
            MagicMock(text='{"passed": true, "observations": ["Button visible"], "issues": [], "confidence": 0.9}')
        ]

        result = await agent.verify_with_vision(
            screenshot=b"fake_screenshot_data",
            verification_prompt="Is the login button visible?",
        )

        assert result.success is True
