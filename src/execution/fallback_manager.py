"""Fallback Manager for Hybrid Execution.

This module manages the escalation logic for the HybridExecutor:
- Determines which fallback level to try next
- Tracks fallback events for analytics
- Calculates estimated costs

The escalation flow is:
1. DOM attempt (fast, ~100ms)
2. DOM retry (transient failure recovery)
3. DOM with wait (element stabilization)
4. DOM with refresh (stale element recovery) - optional
5. Vision fallback (reliable, ~2-5s)
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

from .models import (
    ExecutionMode,
    ExecutionStats,
    ExecutionStrategy,
    FallbackEvent,
    FallbackLevel,
    HybridStepResult,
    StepExecutionConfig,
)

logger = structlog.get_logger()


@dataclass
class AttemptResult:
    """Result from a single execution attempt."""

    success: bool
    error: str | None = None
    duration_ms: int = 0
    data: Any = None


class FallbackManager:
    """Manages fallback escalation and tracking.

    This class handles:
    - Determining which fallback levels to try
    - Executing attempts with proper delays
    - Recording fallback events
    - Calculating costs

    Example:
        manager = FallbackManager(strategy)

        # Get escalation levels for a step
        levels = manager.get_escalation_levels(step_config)

        # Record a fallback event
        manager.record_fallback(event)

        # Get stats
        stats = manager.get_stats()
    """

    def __init__(self, strategy: ExecutionStrategy):
        """Initialize the FallbackManager.

        Args:
            strategy: Execution strategy configuration
        """
        self.strategy = strategy
        self.stats = ExecutionStats()
        self.log = logger.bind(component="fallback_manager")

    def get_escalation_levels(
        self,
        step_action: str,
        step_config: StepExecutionConfig | None = None,
    ) -> list[FallbackLevel]:
        """Get the escalation levels to try for a step.

        This determines which fallback levels should be attempted
        based on the strategy and step configuration.

        Args:
            step_action: The action type (click, fill, etc.)
            step_config: Optional per-step configuration

        Returns:
            List of FallbackLevel to try in order
        """
        # Check for per-step overrides
        if step_config:
            if step_config.always_vision:
                # Skip DOM entirely, use Vision only
                return [FallbackLevel.VISION_FALLBACK]

            if step_config.skip_vision:
                # Never use Vision, DOM only with retries
                return self._get_dom_only_levels()

            if step_config.use_vision:
                # DOM first, but include Vision fallback
                return self._get_full_escalation_levels()

        # Check strategy-level action overrides
        if step_action in self.strategy.always_use_vision_for:
            return [FallbackLevel.VISION_FALLBACK]

        if step_action in self.strategy.never_use_vision_for:
            return self._get_dom_only_levels()

        # Default behavior based on mode
        if self.strategy.mode == ExecutionMode.DOM_ONLY:
            return self._get_dom_only_levels()

        if self.strategy.mode == ExecutionMode.VISION_ONLY:
            return [FallbackLevel.VISION_FALLBACK]

        if self.strategy.mode in (ExecutionMode.HYBRID, ExecutionMode.HYBRID_VERIFY):
            return self._get_full_escalation_levels()

        return self._get_dom_only_levels()

    def _get_dom_only_levels(self) -> list[FallbackLevel]:
        """Get DOM-only escalation levels."""
        levels = [FallbackLevel.NONE]  # First attempt

        for _ in range(self.strategy.dom_retries):
            levels.append(FallbackLevel.RETRY)

        levels.append(FallbackLevel.WAIT_AND_RETRY)

        if self.strategy.refresh_on_failure:
            levels.append(FallbackLevel.REFRESH_AND_RETRY)

        return levels

    def _get_full_escalation_levels(self) -> list[FallbackLevel]:
        """Get full escalation levels including Vision."""
        levels = self._get_dom_only_levels()

        if self.strategy.vision_fallback_enabled:
            levels.append(FallbackLevel.VISION_FALLBACK)

        return levels

    async def execute_with_escalation(
        self,
        dom_fn: Callable[[], Any],
        vision_fn: Callable[[], Any] | None,
        step_index: int,
        step_action: str,
        step_target: str,
        step_config: StepExecutionConfig | None = None,
    ) -> HybridStepResult:
        """Execute a step with escalation through fallback levels.

        This is the main execution method. It tries DOM execution first,
        then escalates through retry levels, and finally falls back to
        Vision if needed.

        Args:
            dom_fn: Async function to execute via DOM
            vision_fn: Async function to execute via Vision (optional)
            step_index: Index of the current step
            step_action: Action type (click, fill, etc.)
            step_target: Target selector or description
            step_config: Optional per-step configuration

        Returns:
            HybridStepResult with execution details
        """
        start_time = time.time()
        levels = self.get_escalation_levels(step_action, step_config)

        dom_attempts = 0
        vision_attempts = 0
        dom_duration_ms = 0
        vision_duration_ms = 0
        last_error: str | None = None
        fallback_event: FallbackEvent | None = None

        for level in levels:
            if level == FallbackLevel.VISION_FALLBACK:
                # Vision fallback
                if not vision_fn:
                    self.log.warning(
                        "Vision fallback requested but no vision executor provided",
                        step_index=step_index,
                    )
                    break

                vision_attempts += 1
                vision_start = time.time()

                if self.strategy.log_fallback_events:
                    self.log.info(
                        "Escalating to Vision fallback",
                        step_index=step_index,
                        step_action=step_action,
                        target=step_target,
                        dom_attempts=dom_attempts,
                        dom_error=last_error,
                    )

                try:
                    result = await vision_fn()
                    vision_duration_ms = int((time.time() - vision_start) * 1000)

                    if result.success:
                        # Vision succeeded
                        fallback_event = self._create_fallback_event(
                            step_index=step_index,
                            step_action=step_action,
                            original_selector=step_target,
                            fallback_level=level,
                            dom_attempts=dom_attempts,
                            dom_error=last_error,
                            success=True,
                            dom_duration_ms=dom_duration_ms,
                            vision_duration_ms=vision_duration_ms,
                        )
                        self.record_fallback(fallback_event)

                        return HybridStepResult(
                            success=True,
                            mode_used="vision",
                            fallback_triggered=True,
                            dom_attempts=dom_attempts,
                            vision_attempts=vision_attempts,
                            total_duration_ms=int((time.time() - start_time) * 1000),
                            dom_duration_ms=dom_duration_ms,
                            vision_duration_ms=vision_duration_ms,
                            estimated_cost=self.strategy.vision_cost_per_call,
                            fallback_event=fallback_event,
                            fallback_level=level,
                            action_data=result.data if hasattr(result, "data") else None,
                        )
                    else:
                        last_error = result.error if hasattr(result, "error") else "Vision execution failed"

                except Exception as e:
                    vision_duration_ms = int((time.time() - vision_start) * 1000)
                    last_error = str(e)
                    self.log.error(
                        "Vision fallback failed",
                        step_index=step_index,
                        error=str(e),
                    )

            else:
                # DOM attempt
                dom_attempts += 1
                attempt_start = time.time()

                # Apply delays based on level
                if level == FallbackLevel.RETRY and dom_attempts > 1:
                    await asyncio.sleep(self.strategy.retry_delay_ms / 1000)

                elif level == FallbackLevel.WAIT_AND_RETRY:
                    if self.strategy.log_all_attempts:
                        self.log.debug(
                            "Waiting before retry",
                            step_index=step_index,
                            wait_ms=self.strategy.wait_before_retry_ms,
                        )
                    await asyncio.sleep(self.strategy.wait_before_retry_ms / 1000)

                elif level == FallbackLevel.REFRESH_AND_RETRY:
                    # TODO: Implement page refresh
                    if self.strategy.log_all_attempts:
                        self.log.debug(
                            "Refresh and retry (not implemented)",
                            step_index=step_index,
                        )

                if self.strategy.log_all_attempts:
                    self.log.debug(
                        "DOM attempt",
                        step_index=step_index,
                        attempt=dom_attempts,
                        level=level.value,
                    )

                try:
                    result = await dom_fn()
                    attempt_duration = int((time.time() - attempt_start) * 1000)
                    dom_duration_ms += attempt_duration

                    if result.success:
                        # DOM succeeded
                        return HybridStepResult(
                            success=True,
                            mode_used="dom",
                            fallback_triggered=level != FallbackLevel.NONE,
                            dom_attempts=dom_attempts,
                            vision_attempts=vision_attempts,
                            total_duration_ms=int((time.time() - start_time) * 1000),
                            dom_duration_ms=dom_duration_ms,
                            vision_duration_ms=vision_duration_ms,
                            estimated_cost=0.0,
                            fallback_level=level,
                            action_data=result.data if hasattr(result, "data") else None,
                        )
                    else:
                        last_error = result.error if hasattr(result, "error") else "DOM execution failed"

                except Exception as e:
                    dom_duration_ms += int((time.time() - attempt_start) * 1000)
                    last_error = str(e)

        # All attempts failed
        total_duration_ms = int((time.time() - start_time) * 1000)

        # Record failed fallback if Vision was attempted
        if vision_attempts > 0:
            fallback_event = self._create_fallback_event(
                step_index=step_index,
                step_action=step_action,
                original_selector=step_target,
                fallback_level=FallbackLevel.VISION_FALLBACK,
                dom_attempts=dom_attempts,
                dom_error=last_error,
                success=False,
                final_error=last_error,
                dom_duration_ms=dom_duration_ms,
                vision_duration_ms=vision_duration_ms,
            )
            self.record_fallback(fallback_event)

        return HybridStepResult(
            success=False,
            error=last_error,
            mode_used="hybrid" if vision_attempts > 0 else "dom",
            fallback_triggered=vision_attempts > 0,
            dom_attempts=dom_attempts,
            vision_attempts=vision_attempts,
            total_duration_ms=total_duration_ms,
            dom_duration_ms=dom_duration_ms,
            vision_duration_ms=vision_duration_ms,
            estimated_cost=self.strategy.vision_cost_per_call * vision_attempts,
            fallback_event=fallback_event,
            fallback_level=levels[-1] if levels else FallbackLevel.NONE,
        )

    def _create_fallback_event(
        self,
        step_index: int,
        step_action: str,
        original_selector: str,
        fallback_level: FallbackLevel,
        dom_attempts: int,
        dom_error: str | None,
        success: bool,
        dom_duration_ms: int = 0,
        vision_duration_ms: int = 0,
        final_error: str | None = None,
    ) -> FallbackEvent:
        """Create a FallbackEvent for recording."""
        return FallbackEvent(
            step_index=step_index,
            step_action=step_action,
            original_selector=original_selector,
            fallback_level=fallback_level,
            dom_attempts=dom_attempts,
            dom_error=dom_error,
            success=success,
            final_error=final_error,
            dom_duration_ms=dom_duration_ms,
            vision_duration_ms=vision_duration_ms,
            total_duration_ms=dom_duration_ms + vision_duration_ms,
            estimated_cost=self.strategy.vision_cost_per_call if fallback_level == FallbackLevel.VISION_FALLBACK else 0.0,
        )

    def record_fallback(self, event: FallbackEvent) -> None:
        """Record a fallback event for analytics.

        Args:
            event: The FallbackEvent to record
        """
        self.stats.fallback_events.append(event)

        if event.success:
            if event.fallback_level == FallbackLevel.VISION_FALLBACK:
                self.stats.vision_fallback_steps += 1
                self.stats.total_vision_duration_ms += event.vision_duration_ms
            self.stats.successful_steps += 1
        else:
            self.stats.failed_steps += 1

        self.stats.total_steps += 1
        self.stats.total_dom_attempts += event.dom_attempts
        self.stats.total_dom_duration_ms += event.dom_duration_ms
        self.stats.total_duration_ms += event.total_duration_ms
        self.stats.total_cost += event.estimated_cost

        if self.strategy.log_fallback_events:
            self.log.info(
                "Fallback event recorded",
                step_index=event.step_index,
                fallback_level=event.fallback_level.value,
                success=event.success,
                dom_attempts=event.dom_attempts,
                estimated_cost=event.estimated_cost,
            )

    def update_stats_for_dom_success(
        self,
        dom_attempts: int,
        duration_ms: int,
    ) -> None:
        """Update stats for a DOM-only success (no fallback).

        Args:
            dom_attempts: Number of DOM attempts
            duration_ms: Total duration in milliseconds
        """
        self.stats.total_steps += 1
        self.stats.successful_steps += 1
        self.stats.dom_only_steps += 1
        self.stats.total_dom_attempts += dom_attempts
        self.stats.total_dom_duration_ms += duration_ms
        self.stats.total_duration_ms += duration_ms

    def update_stats_for_vision_only(
        self,
        success: bool,
        duration_ms: int,
        cost: float,
    ) -> None:
        """Update stats for a Vision-only execution.

        Args:
            success: Whether execution succeeded
            duration_ms: Total duration in milliseconds
            cost: Estimated cost
        """
        self.stats.total_steps += 1
        self.stats.vision_only_steps += 1
        self.stats.total_vision_attempts += 1
        self.stats.total_vision_duration_ms += duration_ms
        self.stats.total_duration_ms += duration_ms
        self.stats.total_cost += cost

        if success:
            self.stats.successful_steps += 1
        else:
            self.stats.failed_steps += 1

    def get_stats(self) -> ExecutionStats:
        """Get current execution statistics.

        Returns:
            ExecutionStats with aggregate metrics
        """
        return self.stats

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats = ExecutionStats()

    def get_common_failures(self, top_n: int = 10) -> list[dict]:
        """Get the most common failure patterns.

        This is useful for identifying which selectors frequently
        require Vision fallback.

        Args:
            top_n: Number of top failures to return

        Returns:
            List of failure patterns with counts
        """
        from collections import Counter

        selector_counts = Counter(
            event.original_selector
            for event in self.stats.fallback_events
        )

        return [
            {"selector": selector, "count": count}
            for selector, count in selector_counts.most_common(top_n)
        ]
