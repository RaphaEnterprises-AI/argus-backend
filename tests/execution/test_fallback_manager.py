"""Tests for FallbackManager."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.fallback_manager import FallbackManager
from src.execution.models import (
    ExecutionMode,
    ExecutionStrategy,
    FallbackEvent,
    FallbackLevel,
    StepExecutionConfig,
)


@pytest.fixture
def default_strategy():
    """Create default execution strategy."""
    return ExecutionStrategy()


@pytest.fixture
def dom_only_strategy():
    """Create DOM-only execution strategy."""
    return ExecutionStrategy(mode=ExecutionMode.DOM_ONLY)


@pytest.fixture
def vision_only_strategy():
    """Create Vision-only execution strategy."""
    return ExecutionStrategy(mode=ExecutionMode.VISION_ONLY)


@pytest.fixture
def hybrid_strategy():
    """Create hybrid execution strategy with custom settings."""
    return ExecutionStrategy(
        mode=ExecutionMode.HYBRID,
        dom_retries=3,
        retry_delay_ms=100,  # Faster for tests
        wait_before_retry_ms=200,  # Faster for tests
        vision_fallback_enabled=True,
        vision_cost_per_call=0.003,
    )


class TestGetEscalationLevels:
    """Test get_escalation_levels method."""

    def test_hybrid_mode_full_escalation(self, hybrid_strategy):
        """Test full escalation levels in hybrid mode."""
        manager = FallbackManager(hybrid_strategy)

        levels = manager.get_escalation_levels("click")

        # Should include: NONE, RETRY x3, WAIT_AND_RETRY, VISION_FALLBACK
        assert FallbackLevel.NONE in levels
        assert FallbackLevel.RETRY in levels
        assert FallbackLevel.WAIT_AND_RETRY in levels
        assert FallbackLevel.VISION_FALLBACK in levels
        assert levels[0] == FallbackLevel.NONE
        assert levels[-1] == FallbackLevel.VISION_FALLBACK

    def test_dom_only_mode_no_vision(self, dom_only_strategy):
        """Test DOM-only mode never includes vision."""
        manager = FallbackManager(dom_only_strategy)

        levels = manager.get_escalation_levels("click")

        assert FallbackLevel.VISION_FALLBACK not in levels
        assert FallbackLevel.NONE in levels
        assert FallbackLevel.WAIT_AND_RETRY in levels

    def test_vision_only_mode_only_vision(self, vision_only_strategy):
        """Test Vision-only mode only includes vision."""
        manager = FallbackManager(vision_only_strategy)

        levels = manager.get_escalation_levels("click")

        assert levels == [FallbackLevel.VISION_FALLBACK]

    def test_always_vision_actions(self, hybrid_strategy):
        """Test actions that always use vision."""
        manager = FallbackManager(hybrid_strategy)

        # These actions should always use vision per default strategy
        for action in ["captcha", "visual_assertion", "canvas", "slider"]:
            levels = manager.get_escalation_levels(action)
            assert levels == [FallbackLevel.VISION_FALLBACK], f"Failed for {action}"

    def test_never_vision_actions(self, hybrid_strategy):
        """Test actions that never use vision."""
        manager = FallbackManager(hybrid_strategy)

        # These actions should never use vision per default strategy
        for action in ["goto", "evaluate"]:
            levels = manager.get_escalation_levels(action)
            assert FallbackLevel.VISION_FALLBACK not in levels, f"Failed for {action}"

    def test_step_config_always_vision(self, hybrid_strategy):
        """Test step config forcing vision-only."""
        manager = FallbackManager(hybrid_strategy)
        step_config = StepExecutionConfig(always_vision=True)

        levels = manager.get_escalation_levels("click", step_config)

        assert levels == [FallbackLevel.VISION_FALLBACK]

    def test_step_config_skip_vision(self, hybrid_strategy):
        """Test step config skipping vision."""
        manager = FallbackManager(hybrid_strategy)
        step_config = StepExecutionConfig(skip_vision=True)

        levels = manager.get_escalation_levels("click", step_config)

        assert FallbackLevel.VISION_FALLBACK not in levels

    def test_step_config_use_vision(self, hybrid_strategy):
        """Test step config enabling vision fallback."""
        manager = FallbackManager(hybrid_strategy)
        step_config = StepExecutionConfig(use_vision=True)

        levels = manager.get_escalation_levels("click", step_config)

        assert FallbackLevel.VISION_FALLBACK in levels

    def test_refresh_on_failure_included(self):
        """Test refresh level is included when enabled."""
        strategy = ExecutionStrategy(refresh_on_failure=True)
        manager = FallbackManager(strategy)

        levels = manager.get_escalation_levels("click")

        assert FallbackLevel.REFRESH_AND_RETRY in levels

    def test_refresh_on_failure_excluded(self, default_strategy):
        """Test refresh level is excluded by default."""
        manager = FallbackManager(default_strategy)

        levels = manager.get_escalation_levels("click")

        assert FallbackLevel.REFRESH_AND_RETRY not in levels


class TestExecuteWithEscalation:
    """Test execute_with_escalation method."""

    @pytest.mark.asyncio
    async def test_dom_success_first_attempt(self, hybrid_strategy):
        """Test DOM succeeds on first attempt."""
        manager = FallbackManager(hybrid_strategy)

        dom_result = MagicMock(success=True, error=None)
        dom_fn = AsyncMock(return_value=dom_result)
        vision_fn = AsyncMock()

        result = await manager.execute_with_escalation(
            dom_fn=dom_fn,
            vision_fn=vision_fn,
            step_index=0,
            step_action="click",
            step_target="#button",
        )

        assert result.success is True
        assert result.mode_used == "dom"
        assert result.fallback_triggered is False
        assert result.dom_attempts == 1
        assert result.vision_attempts == 0
        assert result.estimated_cost == 0.0
        dom_fn.assert_called_once()
        vision_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_dom_success_after_retry(self, hybrid_strategy):
        """Test DOM succeeds after retry."""
        manager = FallbackManager(hybrid_strategy)

        # First call fails, second succeeds
        dom_results = [
            MagicMock(success=False, error="Timeout"),
            MagicMock(success=True, error=None),
        ]
        dom_fn = AsyncMock(side_effect=dom_results)
        vision_fn = AsyncMock()

        result = await manager.execute_with_escalation(
            dom_fn=dom_fn,
            vision_fn=vision_fn,
            step_index=0,
            step_action="click",
            step_target="#button",
        )

        assert result.success is True
        assert result.mode_used == "dom"
        assert result.fallback_triggered is True  # Retry was needed
        assert result.dom_attempts == 2
        assert result.vision_attempts == 0
        vision_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_vision_fallback_on_dom_failure(self, hybrid_strategy):
        """Test Vision fallback when DOM always fails."""
        manager = FallbackManager(hybrid_strategy)

        # DOM always fails
        dom_fn = AsyncMock(return_value=MagicMock(success=False, error="Element not found"))
        # Vision succeeds
        vision_result = MagicMock(success=True, error=None)
        vision_fn = AsyncMock(return_value=vision_result)

        result = await manager.execute_with_escalation(
            dom_fn=dom_fn,
            vision_fn=vision_fn,
            step_index=0,
            step_action="click",
            step_target="#button",
        )

        assert result.success is True
        assert result.mode_used == "vision"
        assert result.fallback_triggered is True
        assert result.dom_attempts >= 1
        assert result.vision_attempts == 1
        assert result.estimated_cost == hybrid_strategy.vision_cost_per_call

    @pytest.mark.asyncio
    async def test_both_fail(self, hybrid_strategy):
        """Test both DOM and Vision fail."""
        manager = FallbackManager(hybrid_strategy)

        # Both fail
        dom_fn = AsyncMock(return_value=MagicMock(success=False, error="DOM error"))
        vision_fn = AsyncMock(return_value=MagicMock(success=False, error="Vision error"))

        result = await manager.execute_with_escalation(
            dom_fn=dom_fn,
            vision_fn=vision_fn,
            step_index=0,
            step_action="click",
            step_target="#button",
        )

        assert result.success is False
        assert result.error is not None
        assert result.fallback_triggered is True
        assert result.vision_attempts == 1

    @pytest.mark.asyncio
    async def test_dom_exception_handling(self, hybrid_strategy):
        """Test DOM exception is caught and escalates."""
        manager = FallbackManager(hybrid_strategy)

        # DOM throws exception
        dom_fn = AsyncMock(side_effect=Exception("Network error"))
        # Vision succeeds
        vision_fn = AsyncMock(return_value=MagicMock(success=True, error=None))

        result = await manager.execute_with_escalation(
            dom_fn=dom_fn,
            vision_fn=vision_fn,
            step_index=0,
            step_action="click",
            step_target="#button",
        )

        assert result.success is True
        assert result.mode_used == "vision"

    @pytest.mark.asyncio
    async def test_vision_exception_handling(self, hybrid_strategy):
        """Test Vision exception results in failure."""
        manager = FallbackManager(hybrid_strategy)

        # Both fail
        dom_fn = AsyncMock(return_value=MagicMock(success=False, error="DOM error"))
        vision_fn = AsyncMock(side_effect=Exception("Vision API error"))

        result = await manager.execute_with_escalation(
            dom_fn=dom_fn,
            vision_fn=vision_fn,
            step_index=0,
            step_action="click",
            step_target="#button",
        )

        assert result.success is False
        assert "Vision API error" in result.error

    @pytest.mark.asyncio
    async def test_no_vision_executor_provided(self, hybrid_strategy):
        """Test when no vision executor is available."""
        manager = FallbackManager(hybrid_strategy)

        # DOM fails
        dom_fn = AsyncMock(return_value=MagicMock(success=False, error="DOM error"))

        result = await manager.execute_with_escalation(
            dom_fn=dom_fn,
            vision_fn=None,  # No vision
            step_index=0,
            step_action="click",
            step_target="#button",
        )

        assert result.success is False
        assert result.vision_attempts == 0

    @pytest.mark.asyncio
    async def test_timing_tracking(self, hybrid_strategy):
        """Test duration tracking is accurate."""
        manager = FallbackManager(hybrid_strategy)

        async def slow_dom():
            await asyncio.sleep(0.1)  # 100ms
            return MagicMock(success=True, error=None)

        result = await manager.execute_with_escalation(
            dom_fn=slow_dom,
            vision_fn=None,
            step_index=0,
            step_action="click",
            step_target="#button",
        )

        assert result.success is True
        assert result.total_duration_ms >= 100
        assert result.dom_duration_ms >= 100


class TestRecordFallback:
    """Test fallback event recording."""

    def test_successful_fallback_recorded(self, hybrid_strategy):
        """Test successful fallback is recorded in stats."""
        manager = FallbackManager(hybrid_strategy)

        event = FallbackEvent(
            step_index=0,
            step_action="click",
            original_selector="#btn",
            fallback_level=FallbackLevel.VISION_FALLBACK,
            dom_attempts=3,
            dom_error="Timeout",
            success=True,
            dom_duration_ms=500,
            vision_duration_ms=2000,
            total_duration_ms=2500,
            estimated_cost=0.003,
        )

        manager.record_fallback(event)
        stats = manager.get_stats()

        assert stats.total_steps == 1
        assert stats.successful_steps == 1
        assert stats.failed_steps == 0
        assert stats.vision_fallback_steps == 1
        assert stats.total_dom_attempts == 3
        assert stats.total_dom_duration_ms == 500
        assert stats.total_vision_duration_ms == 2000
        assert stats.total_cost == 0.003
        assert len(stats.fallback_events) == 1

    def test_failed_fallback_recorded(self, hybrid_strategy):
        """Test failed fallback is recorded in stats."""
        manager = FallbackManager(hybrid_strategy)

        event = FallbackEvent(
            step_index=0,
            step_action="click",
            original_selector="#btn",
            fallback_level=FallbackLevel.VISION_FALLBACK,
            dom_attempts=3,
            success=False,
            final_error="Vision failed",
            estimated_cost=0.003,
        )

        manager.record_fallback(event)
        stats = manager.get_stats()

        assert stats.total_steps == 1
        assert stats.successful_steps == 0
        assert stats.failed_steps == 1
        assert len(stats.fallback_events) == 1


class TestUpdateStats:
    """Test stats update methods."""

    def test_update_dom_success(self, default_strategy):
        """Test updating stats for DOM success."""
        manager = FallbackManager(default_strategy)

        manager.update_stats_for_dom_success(dom_attempts=1, duration_ms=150)
        manager.update_stats_for_dom_success(dom_attempts=2, duration_ms=350)

        stats = manager.get_stats()

        assert stats.total_steps == 2
        assert stats.successful_steps == 2
        assert stats.dom_only_steps == 2
        assert stats.total_dom_attempts == 3
        assert stats.total_dom_duration_ms == 500

    def test_update_vision_only_success(self, default_strategy):
        """Test updating stats for vision-only success."""
        manager = FallbackManager(default_strategy)

        manager.update_stats_for_vision_only(success=True, duration_ms=3000, cost=0.003)

        stats = manager.get_stats()

        assert stats.total_steps == 1
        assert stats.successful_steps == 1
        assert stats.vision_only_steps == 1
        assert stats.total_vision_attempts == 1
        assert stats.total_vision_duration_ms == 3000
        assert stats.total_cost == 0.003

    def test_update_vision_only_failure(self, default_strategy):
        """Test updating stats for vision-only failure."""
        manager = FallbackManager(default_strategy)

        manager.update_stats_for_vision_only(success=False, duration_ms=5000, cost=0.003)

        stats = manager.get_stats()

        assert stats.total_steps == 1
        assert stats.successful_steps == 0
        assert stats.failed_steps == 1


class TestResetStats:
    """Test stats reset."""

    def test_reset_clears_all_stats(self, default_strategy):
        """Test reset clears all stats."""
        manager = FallbackManager(default_strategy)

        # Add some stats
        manager.update_stats_for_dom_success(dom_attempts=1, duration_ms=100)
        manager.update_stats_for_vision_only(success=True, duration_ms=2000, cost=0.003)

        # Verify stats exist
        stats = manager.get_stats()
        assert stats.total_steps == 2

        # Reset
        manager.reset_stats()

        # Verify stats cleared
        stats = manager.get_stats()
        assert stats.total_steps == 0
        assert stats.successful_steps == 0
        assert stats.total_cost == 0.0


class TestCommonFailures:
    """Test common failure pattern analysis."""

    def test_get_common_failures(self, default_strategy):
        """Test getting common failure patterns."""
        manager = FallbackManager(default_strategy)

        # Record some fallback events
        for i in range(5):
            manager.record_fallback(FallbackEvent(
                step_index=i,
                original_selector="#dynamic-btn",
                success=True,
            ))

        for i in range(3):
            manager.record_fallback(FallbackEvent(
                step_index=i,
                original_selector="#spinner-overlay",
                success=True,
            ))

        for i in range(1):
            manager.record_fallback(FallbackEvent(
                step_index=i,
                original_selector="#rare-element",
                success=True,
            ))

        failures = manager.get_common_failures(top_n=3)

        assert len(failures) == 3
        assert failures[0]["selector"] == "#dynamic-btn"
        assert failures[0]["count"] == 5
        assert failures[1]["selector"] == "#spinner-overlay"
        assert failures[1]["count"] == 3

    def test_get_common_failures_empty(self, default_strategy):
        """Test getting common failures when none recorded."""
        manager = FallbackManager(default_strategy)

        failures = manager.get_common_failures()

        assert failures == []


class TestRealisticScenarios:
    """Test realistic execution scenarios."""

    @pytest.mark.asyncio
    async def test_e2e_login_flow(self, hybrid_strategy):
        """Simulate e2e login flow with varied success rates."""
        manager = FallbackManager(hybrid_strategy)

        # Simulate 10 steps:
        # - 7 succeed immediately with DOM (tracked via update_stats_for_dom_success)
        # - 2 need vision fallback (dynamic elements)
        # - 1 fails entirely

        steps = [
            ("goto", "/login", True, False),  # DOM success
            ("fill", "#email", True, False),  # DOM success
            ("fill", "#password", True, False),  # DOM success
            ("click", "#login-btn", True, False),  # DOM success
            ("wait", ".dashboard", True, False),  # DOM success
            ("click", "#dynamic-notification", True, True),  # Vision fallback
            ("fill", "#search", True, False),  # DOM success
            ("click", "#results-item-dynamic", True, True),  # Vision fallback
            ("click", "#missing-element", False, True),  # Both fail
            ("click", "#logout", True, False),  # DOM success
        ]

        for i, (action, target, should_succeed, needs_vision) in enumerate(steps):
            if needs_vision and should_succeed:
                # DOM fails, vision succeeds
                dom_fn = AsyncMock(return_value=MagicMock(success=False, error="Element not found"))
                vision_fn = AsyncMock(return_value=MagicMock(success=True, error=None))
                await manager.execute_with_escalation(
                    dom_fn=dom_fn,
                    vision_fn=vision_fn,
                    step_index=i,
                    step_action=action,
                    step_target=target,
                )
            elif needs_vision and not should_succeed:
                # Both fail
                dom_fn = AsyncMock(return_value=MagicMock(success=False, error="DOM error"))
                vision_fn = AsyncMock(return_value=MagicMock(success=False, error="Vision error"))
                await manager.execute_with_escalation(
                    dom_fn=dom_fn,
                    vision_fn=vision_fn,
                    step_index=i,
                    step_action=action,
                    step_target=target,
                )
            else:
                # DOM succeeds - track directly
                manager.update_stats_for_dom_success(dom_attempts=1, duration_ms=100)

        stats = manager.get_stats()

        # Total: 7 DOM success + 2 vision fallback success + 1 failure = 10 steps
        assert stats.total_steps == 10
        assert stats.dom_only_steps == 7
        assert stats.vision_fallback_steps == 2  # 2 successful vision fallbacks
        assert stats.failed_steps == 1  # 1 total failure
        assert stats.successful_steps == 9  # 7 DOM + 2 vision
        assert abs(stats.total_cost - 0.009) < 0.0001  # 3 vision attempts at $0.003 each

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, hybrid_strategy):
        """Test batch processing of many steps."""
        manager = FallbackManager(hybrid_strategy)

        # Simulate 100 steps - 95% DOM success, 5% need vision
        for i in range(100):
            needs_vision = i % 20 == 0  # Every 20th step needs vision

            if needs_vision:
                dom_fn = AsyncMock(return_value=MagicMock(success=False, error="Flaky"))
                vision_fn = AsyncMock(return_value=MagicMock(success=True, error=None))
                await manager.execute_with_escalation(
                    dom_fn=dom_fn,
                    vision_fn=vision_fn,
                    step_index=i,
                    step_action="click",
                    step_target=f"#element-{i}",
                )
            else:
                # DOM succeeds - track directly
                manager.update_stats_for_dom_success(dom_attempts=1, duration_ms=50)

        stats = manager.get_stats()

        # Total: 95 DOM success + 5 vision fallback = 100 steps
        assert stats.total_steps == 100
        assert stats.dom_only_steps == 95
        assert stats.vision_fallback_steps == 5
        assert stats.total_cost == 0.015  # 5 vision calls at $0.003
        assert stats.fallback_rate == 0.05  # 5/100 = 5%
