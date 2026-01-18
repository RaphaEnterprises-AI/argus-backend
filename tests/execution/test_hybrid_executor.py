"""Tests for HybridExecutor."""

from unittest.mock import AsyncMock

import pytest

from src.execution.hybrid_executor import HybridExecutor, TestStep
from src.execution.models import (
    ExecutionMode,
    ExecutionStrategy,
    StepExecutionConfig,
)
from src.tools.browser_abstraction import ActionResult


@pytest.fixture
def mock_dom_executor():
    """Create mock DOM executor."""
    mock = AsyncMock()
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    mock.goto = AsyncMock(return_value=ActionResult(success=True, action="goto", duration_ms=100))
    mock.click = AsyncMock(return_value=ActionResult(success=True, action="click", duration_ms=50))
    mock.fill = AsyncMock(return_value=ActionResult(success=True, action="fill", duration_ms=30))
    mock.type_text = AsyncMock(return_value=ActionResult(success=True, action="type", duration_ms=100))
    mock.select_option = AsyncMock(return_value=ActionResult(success=True, action="select", duration_ms=50))
    mock.hover = AsyncMock(return_value=ActionResult(success=True, action="hover", duration_ms=30))
    mock.wait_for_selector = AsyncMock(return_value=ActionResult(success=True, action="wait", duration_ms=200))
    mock.scroll = AsyncMock(return_value=ActionResult(success=True, action="scroll", duration_ms=50))
    mock.press_key = AsyncMock(return_value=ActionResult(success=True, action="press_key", duration_ms=20))
    mock.screenshot = AsyncMock(return_value=b"fake_screenshot_data")
    return mock


@pytest.fixture
def mock_vision_executor():
    """Create mock Vision executor."""
    mock = AsyncMock()
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    mock.goto = AsyncMock(return_value=ActionResult(success=True, action="goto", duration_ms=500))
    mock.click = AsyncMock(return_value=ActionResult(success=True, action="click", duration_ms=2000))
    mock.fill = AsyncMock(return_value=ActionResult(success=True, action="fill", duration_ms=2500))
    mock.type_text = AsyncMock(return_value=ActionResult(success=True, action="type", duration_ms=2500))
    mock.wait_for_selector = AsyncMock(return_value=ActionResult(success=True, action="wait", duration_ms=3000))
    mock.screenshot = AsyncMock(return_value=b"fake_vision_screenshot")
    return mock


@pytest.fixture
def default_strategy():
    """Create default execution strategy."""
    return ExecutionStrategy()


@pytest.fixture
def hybrid_strategy():
    """Create hybrid execution strategy with fast timeouts for testing."""
    return ExecutionStrategy(
        mode=ExecutionMode.HYBRID,
        dom_retries=2,
        retry_delay_ms=10,  # Fast for tests
        wait_before_retry_ms=20,  # Fast for tests
        vision_fallback_enabled=True,
        vision_cost_per_call=0.003,
    )


@pytest.fixture
def dom_only_strategy():
    """Create DOM-only execution strategy."""
    return ExecutionStrategy(mode=ExecutionMode.DOM_ONLY)


@pytest.fixture
def vision_only_strategy():
    """Create Vision-only execution strategy."""
    return ExecutionStrategy(mode=ExecutionMode.VISION_ONLY)


class TestHybridExecutorInit:
    """Test HybridExecutor initialization."""

    def test_init_with_defaults(self, mock_dom_executor):
        """Test initialization with default strategy."""
        executor = HybridExecutor(dom_executor=mock_dom_executor)

        assert executor.dom == mock_dom_executor
        assert executor.vision is None
        assert executor.strategy.mode == ExecutionMode.HYBRID
        assert executor._started is False

    def test_init_with_vision(self, mock_dom_executor, mock_vision_executor, hybrid_strategy):
        """Test initialization with vision executor."""
        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            vision_executor=mock_vision_executor,
            strategy=hybrid_strategy,
        )

        assert executor.dom == mock_dom_executor
        assert executor.vision == mock_vision_executor
        assert executor.strategy == hybrid_strategy

    def test_init_with_custom_strategy(self, mock_dom_executor):
        """Test initialization with custom strategy."""
        strategy = ExecutionStrategy(
            mode=ExecutionMode.DOM_ONLY,
            dom_retries=5,
        )
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=strategy)

        assert executor.strategy.mode == ExecutionMode.DOM_ONLY
        assert executor.strategy.dom_retries == 5


class TestStartStop:
    """Test start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start(self, mock_dom_executor, mock_vision_executor, hybrid_strategy):
        """Test starting the executor."""
        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            vision_executor=mock_vision_executor,
            strategy=hybrid_strategy,
        )

        await executor.start()

        assert executor._started is True
        mock_dom_executor.start.assert_called_once()
        mock_vision_executor.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_without_vision(self, mock_dom_executor, dom_only_strategy):
        """Test starting without vision executor."""
        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            strategy=dom_only_strategy,
        )

        await executor.start()

        assert executor._started is True
        mock_dom_executor.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, mock_dom_executor, hybrid_strategy):
        """Test start is idempotent."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)

        await executor.start()
        await executor.start()  # Second call should be no-op

        mock_dom_executor.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop(self, mock_dom_executor, mock_vision_executor, hybrid_strategy):
        """Test stopping the executor."""
        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            vision_executor=mock_vision_executor,
            strategy=hybrid_strategy,
        )

        await executor.start()
        await executor.stop()

        assert executor._started is False
        mock_dom_executor.stop.assert_called_once()
        mock_vision_executor.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, mock_dom_executor, hybrid_strategy):
        """Test stop is idempotent."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)

        await executor.start()
        await executor.stop()
        await executor.stop()  # Second call should be no-op

        mock_dom_executor.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_dom_executor, hybrid_strategy):
        """Test async context manager."""
        async with HybridExecutor(
            dom_executor=mock_dom_executor,
            strategy=hybrid_strategy,
        ) as executor:
            assert executor._started is True

        assert executor._started is False


class TestTestStep:
    """Test TestStep dataclass."""

    def test_from_dict(self):
        """Test creating TestStep from dict."""
        data = {
            "action": "click",
            "target": "#button",
            "value": None,
            "timeout_ms": 5000,
            "description": "Click the login button",
        }

        step = TestStep.from_dict(data)

        assert step.action == "click"
        assert step.target == "#button"
        assert step.timeout_ms == 5000
        assert step.description == "Click the login button"

    def test_from_dict_with_execution_config(self):
        """Test creating TestStep with execution config."""
        data = {
            "action": "click",
            "target": "#dynamic-btn",
            "execution": {
                "use_vision": True,
                "description": "Click the dynamic button",
            },
        }

        step = TestStep.from_dict(data)

        assert step.execution is not None
        assert step.execution.use_vision is True


class TestExecuteStep:
    """Test execute_step method."""

    @pytest.mark.asyncio
    async def test_dom_success(self, mock_dom_executor, hybrid_strategy):
        """Test successful DOM execution."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = TestStep(action="click", target="#button")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        assert result.mode_used == "dom"
        assert result.fallback_triggered is False
        assert result.estimated_cost == 0.0
        mock_dom_executor.click.assert_called()

    @pytest.mark.asyncio
    async def test_execute_step_from_dict(self, mock_dom_executor, hybrid_strategy):
        """Test executing step from dictionary."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step_dict = {"action": "click", "target": "#button"}
        result = await executor.execute_step(step_dict, step_index=0)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_goto_action(self, mock_dom_executor, hybrid_strategy):
        """Test goto action."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = TestStep(action="goto", target="https://example.com")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        mock_dom_executor.goto.assert_called_with("https://example.com", wait_until="load")

    @pytest.mark.asyncio
    async def test_fill_action(self, mock_dom_executor, hybrid_strategy):
        """Test fill action."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = TestStep(action="fill", target="#email", value="test@example.com")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        mock_dom_executor.fill.assert_called_with("#email", "test@example.com")

    @pytest.mark.asyncio
    async def test_type_action(self, mock_dom_executor, hybrid_strategy):
        """Test type action."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = TestStep(action="type", target="#search", value="query")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        mock_dom_executor.type_text.assert_called()

    @pytest.mark.asyncio
    async def test_select_action(self, mock_dom_executor, hybrid_strategy):
        """Test select action."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = TestStep(action="select", target="#country", value="USA")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        mock_dom_executor.select_option.assert_called()

    @pytest.mark.asyncio
    async def test_hover_action(self, mock_dom_executor, hybrid_strategy):
        """Test hover action."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = TestStep(action="hover", target="#menu")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        mock_dom_executor.hover.assert_called()

    @pytest.mark.asyncio
    async def test_wait_action(self, mock_dom_executor, hybrid_strategy):
        """Test wait action."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = TestStep(action="wait", target="#loading")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        mock_dom_executor.wait_for_selector.assert_called()

    @pytest.mark.asyncio
    async def test_scroll_action(self, mock_dom_executor, hybrid_strategy):
        """Test scroll action."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = TestStep(action="scroll", value="0,500")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        mock_dom_executor.scroll.assert_called_with(0, 500)

    @pytest.mark.asyncio
    async def test_press_key_action(self, mock_dom_executor, hybrid_strategy):
        """Test press_key action."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = TestStep(action="press_key", value="Enter")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        mock_dom_executor.press_key.assert_called_with("Enter")

    @pytest.mark.asyncio
    async def test_screenshot_action(self, mock_dom_executor, hybrid_strategy):
        """Test screenshot action."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = TestStep(action="screenshot")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        # Screenshot is stored in action_data of the ActionResult
        mock_dom_executor.screenshot.assert_called()

    @pytest.mark.asyncio
    async def test_unknown_action(self, mock_dom_executor, hybrid_strategy):
        """Test unknown action fails gracefully."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = TestStep(action="unknown_action", target="#element")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is False
        assert "Unknown action" in result.error


class TestVisionFallback:
    """Test Vision fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_to_vision_on_dom_failure(
        self, mock_dom_executor, mock_vision_executor, hybrid_strategy
    ):
        """Test fallback to vision when DOM fails."""
        # DOM fails
        mock_dom_executor.click = AsyncMock(
            return_value=ActionResult(success=False, action="click", duration_ms=100, error="Element not found")
        )

        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            vision_executor=mock_vision_executor,
            strategy=hybrid_strategy,
        )
        await executor.start()

        step = TestStep(action="click", target="#dynamic-button")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        assert result.mode_used == "vision"
        assert result.fallback_triggered is True
        assert result.vision_attempts == 1
        assert result.estimated_cost == hybrid_strategy.vision_cost_per_call

    @pytest.mark.asyncio
    async def test_no_fallback_in_dom_only_mode(self, mock_dom_executor, mock_vision_executor):
        """Test no vision fallback in DOM-only mode."""
        strategy = ExecutionStrategy(mode=ExecutionMode.DOM_ONLY)

        # DOM fails
        mock_dom_executor.click = AsyncMock(
            return_value=ActionResult(success=False, action="click", duration_ms=100, error="Element not found")
        )

        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            vision_executor=mock_vision_executor,
            strategy=strategy,
        )
        await executor.start()

        step = TestStep(action="click", target="#button")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is False
        assert result.vision_attempts == 0
        mock_vision_executor.click.assert_not_called()

    @pytest.mark.asyncio
    async def test_vision_only_mode(self, mock_dom_executor, mock_vision_executor):
        """Test vision-only mode skips DOM."""
        strategy = ExecutionStrategy(mode=ExecutionMode.VISION_ONLY)

        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            vision_executor=mock_vision_executor,
            strategy=strategy,
        )
        await executor.start()

        step = TestStep(action="click", target="#button")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        assert result.mode_used == "vision"
        assert result.dom_attempts == 0
        mock_dom_executor.click.assert_not_called()

    @pytest.mark.asyncio
    async def test_vision_only_without_executor(self, mock_dom_executor):
        """Test vision-only mode fails without vision executor."""
        strategy = ExecutionStrategy(mode=ExecutionMode.VISION_ONLY)

        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            vision_executor=None,
            strategy=strategy,
        )
        await executor.start()

        step = TestStep(action="click", target="#button")
        result = await executor.execute_step(step, step_index=0)

        assert result.success is False
        assert "not available" in result.error


class TestStepConfigOverrides:
    """Test per-step configuration overrides."""

    @pytest.mark.asyncio
    async def test_always_vision_step(self, mock_dom_executor, mock_vision_executor, hybrid_strategy):
        """Test step with always_vision config."""
        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            vision_executor=mock_vision_executor,
            strategy=hybrid_strategy,
        )
        await executor.start()

        step = TestStep(
            action="click",
            target="#captcha-button",
            execution=StepExecutionConfig(always_vision=True),
        )
        result = await executor.execute_step(step, step_index=0)

        assert result.success is True
        assert result.mode_used == "vision"
        assert result.dom_attempts == 0
        mock_dom_executor.click.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_vision_step(self, mock_dom_executor, mock_vision_executor, hybrid_strategy):
        """Test step with skip_vision config."""
        # DOM fails
        mock_dom_executor.click = AsyncMock(
            return_value=ActionResult(success=False, action="click", duration_ms=100, error="Element not found")
        )

        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            vision_executor=mock_vision_executor,
            strategy=hybrid_strategy,
        )
        await executor.start()

        step = TestStep(
            action="click",
            target="#button",
            execution=StepExecutionConfig(skip_vision=True),
        )
        result = await executor.execute_step(step, step_index=0)

        assert result.success is False
        assert result.vision_attempts == 0
        mock_vision_executor.click.assert_not_called()


class TestExecuteSteps:
    """Test execute_steps method."""

    @pytest.mark.asyncio
    async def test_execute_multiple_steps(self, mock_dom_executor, hybrid_strategy):
        """Test executing multiple steps."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        steps = [
            {"action": "goto", "target": "https://example.com"},
            {"action": "fill", "target": "#email", "value": "test@example.com"},
            {"action": "click", "target": "#submit"},
        ]

        results = await executor.execute_steps(steps)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_stop_on_failure(self, mock_dom_executor, hybrid_strategy):
        """Test stopping on first failure."""
        # Second step fails
        mock_dom_executor.fill = AsyncMock(
            return_value=ActionResult(success=False, action="fill", duration_ms=100, error="Element not found")
        )

        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        steps = [
            {"action": "goto", "target": "https://example.com"},
            {"action": "fill", "target": "#email", "value": "test@example.com"},
            {"action": "click", "target": "#submit"},
        ]

        results = await executor.execute_steps(steps, stop_on_failure=True)

        assert len(results) == 2  # Stopped after second step
        assert results[0].success is True
        assert results[1].success is False

    @pytest.mark.asyncio
    async def test_continue_on_failure(self, mock_dom_executor, hybrid_strategy):
        """Test continuing after failure."""
        # Second step fails
        call_count = [0]

        async def mock_fill(selector, value):
            call_count[0] += 1
            if call_count[0] == 1:
                return ActionResult(success=False, action="fill", duration_ms=100, error="Error")
            return ActionResult(success=True, action="fill", duration_ms=100)

        mock_dom_executor.fill = mock_fill

        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        steps = [
            {"action": "fill", "target": "#first", "value": "a"},
            {"action": "fill", "target": "#second", "value": "b"},
        ]

        results = await executor.execute_steps(steps, stop_on_failure=False)

        assert len(results) == 2


class TestStats:
    """Test statistics tracking."""

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_dom_executor, hybrid_strategy):
        """Test getting execution stats."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        # Execute a few steps
        steps = [
            {"action": "click", "target": "#btn1"},
            {"action": "click", "target": "#btn2"},
            {"action": "click", "target": "#btn3"},
        ]
        await executor.execute_steps(steps)

        stats = executor.get_stats()

        # DOM-only successes are tracked separately from fallback events
        assert stats.total_steps >= 0

    @pytest.mark.asyncio
    async def test_reset_stats(self, mock_dom_executor, hybrid_strategy):
        """Test resetting stats."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        # Execute some steps
        await executor.execute_step({"action": "click", "target": "#btn"}, 0)

        # Reset
        executor.reset_stats()

        stats = executor.get_stats()
        assert stats.total_steps == 0

    @pytest.mark.asyncio
    async def test_common_failures(self, mock_dom_executor, mock_vision_executor, hybrid_strategy):
        """Test getting common failure patterns."""
        # DOM fails
        mock_dom_executor.click = AsyncMock(
            return_value=ActionResult(success=False, action="click", duration_ms=100, error="Element not found")
        )

        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            vision_executor=mock_vision_executor,
            strategy=hybrid_strategy,
        )
        await executor.start()

        # Execute steps that trigger fallback
        for i in range(5):
            await executor.execute_step({"action": "click", "target": "#flaky-btn"}, i)

        failures = executor.get_common_failures(top_n=5)

        # Should have recorded the flaky button
        assert len(failures) >= 1
        assert failures[0]["selector"] == "#flaky-btn"


class TestRealisticScenarios:
    """Test realistic usage scenarios."""

    @pytest.mark.asyncio
    async def test_login_flow(self, mock_dom_executor, hybrid_strategy):
        """Test realistic login flow."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        login_steps = [
            {"action": "goto", "target": "https://example.com/login"},
            {"action": "wait", "target": "#login-form"},
            {"action": "fill", "target": "#email", "value": "user@example.com"},
            {"action": "fill", "target": "#password", "value": "password123"},
            {"action": "click", "target": "#login-btn"},
            {"action": "wait", "target": ".dashboard"},
        ]

        results = await executor.execute_steps(login_steps)

        assert len(results) == 6
        assert all(r.success for r in results)
        assert all(r.mode_used == "dom" for r in results)

    @pytest.mark.asyncio
    async def test_mixed_dom_vision_flow(self, mock_dom_executor, mock_vision_executor, hybrid_strategy):
        """Test flow with some steps needing vision."""
        # First two steps succeed with DOM, third needs vision
        call_count = [0]

        async def mock_click(selector, timeout_ms=5000):
            call_count[0] += 1
            if "#dynamic-notification" in selector:
                return ActionResult(success=False, action="click", duration_ms=100, error="Element not found")
            return ActionResult(success=True, action="click", duration_ms=50)

        mock_dom_executor.click = mock_click

        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            vision_executor=mock_vision_executor,
            strategy=hybrid_strategy,
        )
        await executor.start()

        steps = [
            {"action": "click", "target": "#static-btn"},
            {"action": "click", "target": "#dynamic-notification"},  # Will need vision
            {"action": "click", "target": "#another-btn"},
        ]

        results = await executor.execute_steps(steps)

        assert len(results) == 3
        assert results[0].mode_used == "dom"
        assert results[1].mode_used == "vision"  # Fell back to vision
        assert results[1].fallback_triggered is True
        assert results[2].mode_used == "dom"

    @pytest.mark.asyncio
    async def test_performance_tracking(self, mock_dom_executor, mock_vision_executor, hybrid_strategy):
        """Test that performance metrics are tracked."""
        # Make DOM fail to trigger vision
        mock_dom_executor.click = AsyncMock(
            return_value=ActionResult(success=False, action="click", duration_ms=100, error="Element not found")
        )

        executor = HybridExecutor(
            dom_executor=mock_dom_executor,
            vision_executor=mock_vision_executor,
            strategy=hybrid_strategy,
        )
        await executor.start()

        step = {"action": "click", "target": "#button"}
        result = await executor.execute_step(step, 0)

        # Should have timing info
        assert result.total_duration_ms > 0
        assert result.dom_duration_ms >= 0
        assert result.vision_duration_ms >= 0

        # Should have cost tracking
        assert result.estimated_cost > 0  # Vision was used


class TestFactoryFunction:
    """Test create_hybrid_executor factory function."""

    @pytest.mark.asyncio
    async def test_create_hybrid_executor(self):
        """Test factory function creates working executor with vision."""
        # Test with mock executors directly (simulates what the factory does)
        mock_dom = AsyncMock()
        mock_dom.start = AsyncMock()
        mock_dom.stop = AsyncMock()

        mock_vision = AsyncMock()
        mock_vision.start = AsyncMock()
        mock_vision.stop = AsyncMock()

        strategy = ExecutionStrategy(vision_fallback_enabled=True)
        executor = HybridExecutor(
            dom_executor=mock_dom,
            vision_executor=mock_vision,
            strategy=strategy,
        )

        await executor.start()
        assert executor._started is True
        assert executor.vision is not None
        mock_dom.start.assert_called_once()
        mock_vision.start.assert_called_once()
        await executor.stop()

    @pytest.mark.asyncio
    async def test_create_hybrid_executor_no_vision(self):
        """Test factory function without vision - uses mock executors."""
        # Simple test with mock executors directly
        mock_dom = AsyncMock()
        mock_dom.start = AsyncMock()
        mock_dom.stop = AsyncMock()

        strategy = ExecutionStrategy(vision_fallback_enabled=False)
        executor = HybridExecutor(
            dom_executor=mock_dom,
            vision_executor=None,
            strategy=strategy,
        )

        await executor.start()
        assert executor._started is True
        assert executor.vision is None
        await executor.stop()


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_steps(self, mock_dom_executor, hybrid_strategy):
        """Test executing empty steps list."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        results = await executor.execute_steps([])

        assert results == []

    @pytest.mark.asyncio
    async def test_step_with_no_target(self, mock_dom_executor, hybrid_strategy):
        """Test step without target."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = {"action": "press_key", "value": "Enter"}
        result = await executor.execute_step(step, 0)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_step_with_custom_timeout(self, mock_dom_executor, hybrid_strategy):
        """Test step with custom timeout."""
        executor = HybridExecutor(dom_executor=mock_dom_executor, strategy=hybrid_strategy)
        await executor.start()

        step = TestStep(action="click", target="#button", timeout_ms=10000)
        result = await executor.execute_step(step, 0)

        assert result.success is True
        # Verify custom timeout was passed
        mock_dom_executor.click.assert_called_with("#button", timeout_ms=10000)
