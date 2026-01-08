"""Hybrid Executor for E2E Testing.

This module provides the main HybridExecutor class that combines
DOM-based and Vision-based test execution for optimal speed and reliability.

The HybridExecutor:
1. Uses DOM execution first (fast: 50-200ms)
2. Automatically escalates through retry levels
3. Falls back to Vision AI when DOM fails (reliable: 2-5s)
4. Records all fallback events for analytics
5. Tracks estimated costs for Vision API calls

Example:
    from src.execution import HybridExecutor, ExecutionStrategy
    from src.tools.browser_abstraction import PlaywrightAutomation, ComputerUseAutomation

    # Create executors
    dom_executor = PlaywrightAutomation()
    vision_executor = ComputerUseAutomation()

    # Create hybrid executor
    executor = HybridExecutor(
        dom_executor=dom_executor,
        vision_executor=vision_executor,
        strategy=ExecutionStrategy(
            mode=ExecutionMode.HYBRID,
            dom_retries=2,
            vision_fallback_enabled=True,
        )
    )

    # Start browsers
    await executor.start()

    # Execute steps
    for i, step in enumerate(test_steps):
        result = await executor.execute_step(step, step_index=i)
        if not result.success:
            print(f"Step {i} failed: {result.error}")
            break

    # Get stats
    stats = executor.get_stats()
    print(f"Fallback rate: {stats.fallback_rate:.1%}")
    print(f"Total cost: ${stats.total_cost:.3f}")

    # Stop browsers
    await executor.stop()
"""

import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import structlog

from src.tools.browser_abstraction import ActionResult, BrowserAutomation

from .fallback_manager import FallbackManager
from .models import (
    ExecutionMode,
    ExecutionStats,
    ExecutionStrategy,
    FallbackLevel,
    HybridStepResult,
    StepExecutionConfig,
)

logger = structlog.get_logger()


@dataclass
class TestStep:
    """A single test step to execute.

    This is a simplified version of the TestStep from test_planner.py.
    It's defined here to avoid circular imports.
    """

    action: str
    target: Optional[str] = None
    value: Optional[str] = None
    timeout_ms: Optional[int] = None
    description: Optional[str] = None
    execution: Optional[StepExecutionConfig] = None

    @classmethod
    def from_dict(cls, data: dict) -> "TestStep":
        """Create a TestStep from a dictionary."""
        execution_data = data.get("execution")
        execution = StepExecutionConfig(**execution_data) if execution_data else None

        return cls(
            action=data.get("action", ""),
            target=data.get("target"),
            value=data.get("value"),
            timeout_ms=data.get("timeout_ms"),
            description=data.get("description"),
            execution=execution,
        )


class HybridExecutor:
    """Hybrid execution engine combining DOM and Vision strategies.

    This is the main class for executing test steps with automatic
    fallback from DOM to Vision when needed.

    Attributes:
        dom_executor: BrowserAutomation for DOM-based execution
        vision_executor: BrowserAutomation for Vision-based execution (optional)
        strategy: ExecutionStrategy configuration
        fallback_manager: FallbackManager for escalation and tracking
    """

    def __init__(
        self,
        dom_executor: BrowserAutomation,
        vision_executor: Optional[BrowserAutomation] = None,
        strategy: Optional[ExecutionStrategy] = None,
    ):
        """Initialize the HybridExecutor.

        Args:
            dom_executor: BrowserAutomation for DOM-based execution
            vision_executor: BrowserAutomation for Vision-based execution (optional)
            strategy: ExecutionStrategy configuration (uses defaults if not provided)
        """
        self.dom = dom_executor
        self.vision = vision_executor
        self.strategy = strategy or ExecutionStrategy()
        self.fallback_manager = FallbackManager(self.strategy)
        self.log = logger.bind(component="hybrid_executor")
        self._started = False

    async def start(self) -> None:
        """Start the browser automation sessions."""
        if self._started:
            return

        self.log.info("Starting HybridExecutor")

        await self.dom.start()

        if self.vision and self.strategy.vision_fallback_enabled:
            await self.vision.start()

        self._started = True
        self.log.info(
            "HybridExecutor started",
            mode=self.strategy.mode.value,
            vision_enabled=self.vision is not None and self.strategy.vision_fallback_enabled,
        )

    async def stop(self) -> None:
        """Stop the browser automation sessions."""
        if not self._started:
            return

        self.log.info("Stopping HybridExecutor")

        await self.dom.stop()

        if self.vision:
            await self.vision.stop()

        self._started = False
        self.log.info("HybridExecutor stopped")

    async def execute_step(
        self,
        step: TestStep | dict,
        step_index: int,
    ) -> HybridStepResult:
        """Execute a single test step with hybrid strategy.

        This is the main execution method. It:
        1. Determines the execution strategy for this step
        2. Creates DOM and Vision execution functions
        3. Delegates to FallbackManager for escalation
        4. Returns detailed execution result

        Args:
            step: The test step to execute (TestStep or dict)
            step_index: Index of this step in the test

        Returns:
            HybridStepResult with execution details
        """
        if isinstance(step, dict):
            step = TestStep.from_dict(step)

        start_time = time.time()

        # Check if this step should always use Vision
        if self._should_use_vision_only(step):
            return await self._execute_vision_only(step, step_index)

        # Check if Vision should be skipped entirely
        if self._should_skip_vision(step):
            return await self._execute_dom_only(step, step_index)

        # Hybrid execution with fallback
        return await self._execute_hybrid(step, step_index)

    def _should_use_vision_only(self, step: TestStep) -> bool:
        """Check if this step should use Vision only."""
        if step.execution and step.execution.always_vision:
            return True

        if step.action in self.strategy.always_use_vision_for:
            return True

        if self.strategy.mode == ExecutionMode.VISION_ONLY:
            return True

        return False

    def _should_skip_vision(self, step: TestStep) -> bool:
        """Check if Vision should be skipped for this step."""
        if step.execution and step.execution.skip_vision:
            return True

        if step.action in self.strategy.never_use_vision_for:
            return True

        if self.strategy.mode == ExecutionMode.DOM_ONLY:
            return True

        if not self.strategy.vision_fallback_enabled:
            return True

        if not self.vision:
            return True

        return False

    async def _execute_hybrid(
        self,
        step: TestStep,
        step_index: int,
    ) -> HybridStepResult:
        """Execute step with DOM first, Vision fallback."""
        # Create DOM execution function
        async def dom_fn() -> ActionResult:
            return await self._execute_dom_action(step)

        # Create Vision execution function
        async def vision_fn() -> ActionResult:
            return await self._execute_vision_action(step)

        # Delegate to FallbackManager
        result = await self.fallback_manager.execute_with_escalation(
            dom_fn=dom_fn,
            vision_fn=vision_fn if self.vision else None,
            step_index=step_index,
            step_action=step.action,
            step_target=step.target or "",
            step_config=step.execution,
        )

        # Take screenshot if configured
        if result.success and self.strategy.auto_screenshot_on_success:
            result.screenshot = await self.dom.screenshot()
        elif not result.success and self.strategy.auto_screenshot_on_failure:
            result.screenshot = await self.dom.screenshot()

        return result

    async def _execute_dom_only(
        self,
        step: TestStep,
        step_index: int,
    ) -> HybridStepResult:
        """Execute step using DOM only, no Vision fallback."""
        start_time = time.time()
        dom_attempts = 0
        last_error: Optional[str] = None

        # Get DOM-only escalation levels
        levels = [FallbackLevel.NONE]
        for _ in range(self.strategy.dom_retries):
            levels.append(FallbackLevel.RETRY)
        levels.append(FallbackLevel.WAIT_AND_RETRY)

        import asyncio

        for level in levels:
            dom_attempts += 1

            # Apply delays based on level
            if level == FallbackLevel.RETRY and dom_attempts > 1:
                await asyncio.sleep(self.strategy.retry_delay_ms / 1000)
            elif level == FallbackLevel.WAIT_AND_RETRY:
                await asyncio.sleep(self.strategy.wait_before_retry_ms / 1000)

            try:
                result = await self._execute_dom_action(step)

                if result.success:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self.fallback_manager.update_stats_for_dom_success(
                        dom_attempts=dom_attempts,
                        duration_ms=duration_ms,
                    )

                    return HybridStepResult(
                        success=True,
                        mode_used="dom",
                        fallback_triggered=level != FallbackLevel.NONE,
                        dom_attempts=dom_attempts,
                        vision_attempts=0,
                        total_duration_ms=duration_ms,
                        dom_duration_ms=duration_ms,
                        vision_duration_ms=0,
                        estimated_cost=0.0,
                        fallback_level=level,
                    )
                else:
                    last_error = result.error

            except Exception as e:
                last_error = str(e)

        # All attempts failed
        duration_ms = int((time.time() - start_time) * 1000)
        return HybridStepResult(
            success=False,
            error=last_error,
            mode_used="dom",
            fallback_triggered=False,
            dom_attempts=dom_attempts,
            vision_attempts=0,
            total_duration_ms=duration_ms,
            dom_duration_ms=duration_ms,
            vision_duration_ms=0,
            estimated_cost=0.0,
            fallback_level=levels[-1] if levels else FallbackLevel.NONE,
        )

    async def _execute_vision_only(
        self,
        step: TestStep,
        step_index: int,
    ) -> HybridStepResult:
        """Execute step using Vision only, skip DOM."""
        if not self.vision:
            return HybridStepResult(
                success=False,
                error="Vision executor not available",
                mode_used="vision",
                fallback_triggered=False,
                dom_attempts=0,
                vision_attempts=0,
                total_duration_ms=0,
                dom_duration_ms=0,
                vision_duration_ms=0,
                estimated_cost=0.0,
            )

        start_time = time.time()

        try:
            result = await self._execute_vision_action(step)
            duration_ms = int((time.time() - start_time) * 1000)

            self.fallback_manager.update_stats_for_vision_only(
                success=result.success,
                duration_ms=duration_ms,
                cost=self.strategy.vision_cost_per_call,
            )

            return HybridStepResult(
                success=result.success,
                error=result.error if not result.success else None,
                mode_used="vision",
                fallback_triggered=False,
                dom_attempts=0,
                vision_attempts=1,
                total_duration_ms=duration_ms,
                dom_duration_ms=0,
                vision_duration_ms=duration_ms,
                estimated_cost=self.strategy.vision_cost_per_call,
                action_data=result.data,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return HybridStepResult(
                success=False,
                error=str(e),
                mode_used="vision",
                fallback_triggered=False,
                dom_attempts=0,
                vision_attempts=1,
                total_duration_ms=duration_ms,
                dom_duration_ms=0,
                vision_duration_ms=duration_ms,
                estimated_cost=self.strategy.vision_cost_per_call,
            )

    async def _execute_dom_action(self, step: TestStep) -> ActionResult:
        """Execute a step action using DOM.

        Args:
            step: The test step to execute

        Returns:
            ActionResult from the DOM executor
        """
        timeout = step.timeout_ms or self.strategy.dom_timeout_ms

        if step.action == "goto":
            return await self.dom.goto(step.target or "", wait_until="load")

        elif step.action == "click":
            return await self.dom.click(step.target or "", timeout_ms=timeout)

        elif step.action == "fill":
            return await self.dom.fill(step.target or "", step.value or "")

        elif step.action == "type":
            return await self.dom.type_text(step.target or "", step.value or "")

        elif step.action == "select":
            return await self.dom.select_option(step.target or "", step.value or "")

        elif step.action == "hover":
            return await self.dom.hover(step.target or "")

        elif step.action == "wait":
            return await self.dom.wait_for_selector(step.target or "", timeout_ms=timeout)

        elif step.action == "scroll":
            # Parse scroll value (e.g., "0,500" for x,y)
            x, y = 0, 0
            if step.value:
                parts = step.value.split(",")
                x = int(parts[0]) if len(parts) > 0 else 0
                y = int(parts[1]) if len(parts) > 1 else 0
            return await self.dom.scroll(x, y)

        elif step.action == "press_key":
            return await self.dom.press_key(step.value or "")

        elif step.action == "screenshot":
            screenshot = await self.dom.screenshot(full_page=step.value == "full")
            return ActionResult(success=True, action="screenshot", duration_ms=0, data=screenshot)

        else:
            return ActionResult(
                success=False,
                action=step.action,
                duration_ms=0,
                error=f"Unknown action: {step.action}",
            )

    async def _execute_vision_action(self, step: TestStep) -> ActionResult:
        """Execute a step action using Vision.

        For Vision execution, selectors become visual descriptions.
        The Vision executor uses Claude's Computer Use to identify
        and interact with elements based on their appearance.

        Args:
            step: The test step to execute

        Returns:
            ActionResult from the Vision executor
        """
        if not self.vision:
            return ActionResult(
                success=False,
                action=step.action,
                duration_ms=0,
                error="Vision executor not available",
            )

        # For Vision, use description if available, otherwise use target
        description = step.description or step.target or step.action
        timeout = step.timeout_ms or self.strategy.vision_timeout_ms

        if step.action == "goto":
            return await self.vision.goto(step.target or "", wait_until="load")

        elif step.action == "click":
            return await self.vision.click(description, timeout_ms=timeout)

        elif step.action == "fill":
            return await self.vision.fill(description, step.value or "")

        elif step.action == "type":
            return await self.vision.type_text(description, step.value or "")

        elif step.action == "select":
            # Vision: describe dropdown and value
            select_description = f"{description} and select '{step.value}'"
            return await self.vision.click(select_description, timeout_ms=timeout)

        elif step.action == "hover":
            return await self.vision.click(f"hover over {description}")

        elif step.action == "wait":
            return await self.vision.wait_for_selector(description, timeout_ms=timeout)

        elif step.action == "scroll":
            # Vision: describe scroll action
            x, y = 0, 0
            if step.value:
                parts = step.value.split(",")
                y = int(parts[1]) if len(parts) > 1 else 0
            direction = "down" if y > 0 else "up"
            return await self.vision.click(f"scroll {direction}")

        elif step.action == "press_key":
            # Vision: describe key press
            return await self.vision.click(f"press {step.value}")

        elif step.action == "screenshot":
            screenshot = await self.vision.screenshot()
            return ActionResult(success=True, action="screenshot", duration_ms=0, data=screenshot)

        else:
            return ActionResult(
                success=False,
                action=step.action,
                duration_ms=0,
                error=f"Unknown action: {step.action}",
            )

    async def execute_steps(
        self,
        steps: list[TestStep | dict],
        stop_on_failure: bool = True,
    ) -> list[HybridStepResult]:
        """Execute multiple test steps.

        Args:
            steps: List of test steps to execute
            stop_on_failure: Whether to stop on first failure

        Returns:
            List of HybridStepResult for each step
        """
        results = []

        for i, step in enumerate(steps):
            result = await self.execute_step(step, step_index=i)
            results.append(result)

            if not result.success and stop_on_failure:
                self.log.warning(
                    "Stopping execution due to failure",
                    step_index=i,
                    error=result.error,
                )
                break

        return results

    def get_stats(self) -> ExecutionStats:
        """Get execution statistics.

        Returns:
            ExecutionStats with aggregate metrics
        """
        return self.fallback_manager.get_stats()

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.fallback_manager.reset_stats()

    def get_common_failures(self, top_n: int = 10) -> list[dict]:
        """Get the most common failure patterns.

        Args:
            top_n: Number of top failures to return

        Returns:
            List of failure patterns with counts
        """
        return self.fallback_manager.get_common_failures(top_n)

    # Context manager support

    async def __aenter__(self) -> "HybridExecutor":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


# Factory function for easy creation


async def create_hybrid_executor(
    headless: bool = True,
    vision_enabled: bool = True,
    strategy: Optional[ExecutionStrategy] = None,
) -> HybridExecutor:
    """Create and start a HybridExecutor with default configuration.

    This is a convenience function that creates a HybridExecutor
    with Playwright for DOM and Computer Use for Vision.

    Args:
        headless: Whether to run browser headlessly
        vision_enabled: Whether to enable Vision fallback
        strategy: Optional execution strategy

    Returns:
        Started HybridExecutor instance

    Example:
        async with await create_hybrid_executor() as executor:
            result = await executor.execute_step(step, 0)
    """
    from src.tools.browser_abstraction import (
        BrowserConfig,
        ComputerUseAutomation,
        PlaywrightAutomation,
    )

    config = BrowserConfig(headless=headless)

    dom_executor = PlaywrightAutomation(config)
    vision_executor = ComputerUseAutomation(config) if vision_enabled else None

    if strategy is None:
        strategy = ExecutionStrategy(
            vision_fallback_enabled=vision_enabled,
        )

    executor = HybridExecutor(
        dom_executor=dom_executor,
        vision_executor=vision_executor,
        strategy=strategy,
    )

    await executor.start()
    return executor
