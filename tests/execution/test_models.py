"""Tests for execution data models."""

from datetime import datetime, timedelta

from src.execution.models import (
    ExecutionMode,
    ExecutionStats,
    ExecutionStrategy,
    FallbackEvent,
    FallbackLevel,
    HybridStepResult,
    StepExecutionConfig,
)


class TestExecutionMode:
    """Test ExecutionMode enum."""

    def test_enum_values(self):
        """Test all execution mode values."""
        assert ExecutionMode.DOM_ONLY.value == "dom_only"
        assert ExecutionMode.VISION_ONLY.value == "vision_only"
        assert ExecutionMode.HYBRID.value == "hybrid"
        assert ExecutionMode.HYBRID_VERIFY.value == "hybrid_verify"

    def test_enum_string_comparison(self):
        """Test enum can be compared as string."""
        assert ExecutionMode.HYBRID == "hybrid"
        assert ExecutionMode.DOM_ONLY != ExecutionMode.VISION_ONLY


class TestFallbackLevel:
    """Test FallbackLevel enum."""

    def test_enum_values(self):
        """Test all fallback level values."""
        assert FallbackLevel.NONE.value == "none"
        assert FallbackLevel.RETRY.value == "retry"
        assert FallbackLevel.WAIT_AND_RETRY.value == "wait_and_retry"
        assert FallbackLevel.REFRESH_AND_RETRY.value == "refresh_and_retry"
        assert FallbackLevel.VISION_FALLBACK.value == "vision_fallback"

    def test_escalation_ordering(self):
        """Test escalation levels are in expected order."""
        levels = [
            FallbackLevel.NONE,
            FallbackLevel.RETRY,
            FallbackLevel.WAIT_AND_RETRY,
            FallbackLevel.REFRESH_AND_RETRY,
            FallbackLevel.VISION_FALLBACK,
        ]
        # Verify they're all unique
        assert len(levels) == len(set(levels))


class TestExecutionStrategy:
    """Test ExecutionStrategy dataclass."""

    def test_default_values(self):
        """Test default strategy configuration."""
        strategy = ExecutionStrategy()

        assert strategy.mode == ExecutionMode.HYBRID
        assert strategy.dom_timeout_ms == 5000
        assert strategy.dom_retries == 2
        assert strategy.retry_delay_ms == 500
        assert strategy.wait_before_retry_ms == 1000
        assert strategy.refresh_on_failure is False
        assert strategy.vision_fallback_enabled is True
        assert strategy.vision_timeout_ms == 30000
        assert strategy.vision_model == "claude-sonnet-4-5"
        assert strategy.vision_cost_per_call == 0.003
        assert strategy.track_costs is True
        assert strategy.auto_screenshot_on_failure is True
        assert strategy.auto_screenshot_on_success is False
        assert strategy.log_fallback_events is True
        assert strategy.log_all_attempts is False

    def test_custom_strategy(self):
        """Test custom strategy configuration."""
        strategy = ExecutionStrategy(
            mode=ExecutionMode.DOM_ONLY,
            dom_retries=5,
            vision_fallback_enabled=False,
            dom_timeout_ms=10000,
        )

        assert strategy.mode == ExecutionMode.DOM_ONLY
        assert strategy.dom_retries == 5
        assert strategy.vision_fallback_enabled is False
        assert strategy.dom_timeout_ms == 10000

    def test_always_use_vision_for_defaults(self):
        """Test default actions that always use vision."""
        strategy = ExecutionStrategy()

        assert "visual_assertion" in strategy.always_use_vision_for
        assert "captcha" in strategy.always_use_vision_for
        assert "canvas" in strategy.always_use_vision_for
        assert "drag_and_drop" in strategy.always_use_vision_for
        assert "slider" in strategy.always_use_vision_for

    def test_never_use_vision_for_defaults(self):
        """Test default actions that never use vision."""
        strategy = ExecutionStrategy()

        assert "goto" in strategy.never_use_vision_for
        assert "evaluate" in strategy.never_use_vision_for


class TestStepExecutionConfig:
    """Test StepExecutionConfig dataclass."""

    def test_default_values(self):
        """Test default step config."""
        config = StepExecutionConfig()

        assert config.use_vision is False
        assert config.always_vision is False
        assert config.skip_vision is False
        assert config.timeout_ms is None
        assert config.retries is None
        assert config.description is None

    def test_force_vision(self):
        """Test forcing vision for a step."""
        config = StepExecutionConfig(always_vision=True)

        assert config.always_vision is True
        assert config.use_vision is False  # These are different flags

    def test_skip_vision(self):
        """Test skipping vision for a step."""
        config = StepExecutionConfig(skip_vision=True)

        assert config.skip_vision is True

    def test_custom_description(self):
        """Test custom description for vision."""
        config = StepExecutionConfig(
            use_vision=True,
            description="Click the blue login button at the top right",
        )

        assert config.use_vision is True
        assert "blue login button" in config.description


class TestFallbackEvent:
    """Test FallbackEvent dataclass."""

    def test_default_values(self):
        """Test default fallback event."""
        event = FallbackEvent(step_index=0)

        assert event.step_index == 0
        assert event.step_action == ""
        assert event.original_selector == ""
        assert event.fallback_level == FallbackLevel.NONE
        assert event.dom_attempts == 0
        assert event.dom_error is None
        assert event.success is False
        assert event.estimated_cost == 0.0

    def test_successful_vision_fallback(self):
        """Test successful vision fallback event."""
        event = FallbackEvent(
            step_index=3,
            step_action="click",
            original_selector="#dynamic-button",
            fallback_level=FallbackLevel.VISION_FALLBACK,
            dom_attempts=3,
            dom_error="Element not found",
            success=True,
            dom_duration_ms=1500,
            vision_duration_ms=2500,
            total_duration_ms=4000,
            estimated_cost=0.003,
        )

        assert event.step_index == 3
        assert event.step_action == "click"
        assert event.fallback_level == FallbackLevel.VISION_FALLBACK
        assert event.dom_attempts == 3
        assert event.success is True
        assert event.estimated_cost == 0.003

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = FallbackEvent(
            step_index=1,
            step_action="click",
            original_selector="#btn",
            fallback_level=FallbackLevel.VISION_FALLBACK,
            dom_attempts=2,
            dom_error="Timeout",
            success=True,
            dom_duration_ms=1000,
            vision_duration_ms=2000,
            total_duration_ms=3000,
            estimated_cost=0.003,
        )

        d = event.to_dict()

        assert d["step_index"] == 1
        assert d["step_action"] == "click"
        assert d["original_selector"] == "#btn"
        assert d["fallback_level"] == "vision_fallback"
        assert d["dom_attempts"] == 2
        assert d["success"] is True
        assert d["estimated_cost"] == 0.003
        assert "timestamp" in d

    def test_timestamp_auto_generated(self):
        """Test timestamp is auto-generated."""
        event = FallbackEvent(step_index=0)

        assert event.timestamp is not None
        # Timestamp should be recent (within 1 minute)
        assert datetime.utcnow() - event.timestamp < timedelta(minutes=1)


class TestHybridStepResult:
    """Test HybridStepResult dataclass."""

    def test_successful_dom_result(self):
        """Test successful DOM execution result."""
        result = HybridStepResult(
            success=True,
            mode_used="dom",
            fallback_triggered=False,
            dom_attempts=1,
            vision_attempts=0,
            total_duration_ms=150,
            dom_duration_ms=150,
            vision_duration_ms=0,
            estimated_cost=0.0,
            fallback_level=FallbackLevel.NONE,
        )

        assert result.success is True
        assert result.mode_used == "dom"
        assert result.fallback_triggered is False
        assert result.dom_attempts == 1
        assert result.estimated_cost == 0.0

    def test_successful_vision_fallback_result(self):
        """Test successful vision fallback result."""
        result = HybridStepResult(
            success=True,
            mode_used="vision",
            fallback_triggered=True,
            dom_attempts=3,
            vision_attempts=1,
            total_duration_ms=4000,
            dom_duration_ms=1500,
            vision_duration_ms=2500,
            estimated_cost=0.003,
            fallback_level=FallbackLevel.VISION_FALLBACK,
        )

        assert result.success is True
        assert result.mode_used == "vision"
        assert result.fallback_triggered is True
        assert result.dom_attempts == 3
        assert result.vision_attempts == 1
        assert result.estimated_cost == 0.003

    def test_failed_result(self):
        """Test failed execution result."""
        result = HybridStepResult(
            success=False,
            error="Element not found after all retries",
            mode_used="hybrid",
            fallback_triggered=True,
            dom_attempts=3,
            vision_attempts=1,
            total_duration_ms=5000,
            dom_duration_ms=2000,
            vision_duration_ms=3000,
            estimated_cost=0.003,
            fallback_level=FallbackLevel.VISION_FALLBACK,
        )

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = HybridStepResult(
            success=True,
            mode_used="dom",
            fallback_triggered=False,
            dom_attempts=1,
            vision_attempts=0,
            total_duration_ms=100,
            dom_duration_ms=100,
            vision_duration_ms=0,
            estimated_cost=0.0,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["mode_used"] == "dom"
        assert d["fallback_triggered"] is False
        assert d["dom_attempts"] == 1
        assert d["vision_attempts"] == 0
        assert d["estimated_cost"] == 0.0


class TestExecutionStats:
    """Test ExecutionStats dataclass."""

    def test_default_values(self):
        """Test default stats."""
        stats = ExecutionStats()

        assert stats.total_steps == 0
        assert stats.successful_steps == 0
        assert stats.failed_steps == 0
        assert stats.dom_only_steps == 0
        assert stats.vision_fallback_steps == 0
        assert stats.vision_only_steps == 0
        assert stats.total_cost == 0.0
        assert len(stats.fallback_events) == 0

    def test_success_rate_calculation(self):
        """Test success rate property."""
        stats = ExecutionStats(
            total_steps=10,
            successful_steps=8,
            failed_steps=2,
        )

        assert stats.success_rate == 0.8

    def test_success_rate_zero_steps(self):
        """Test success rate with zero steps."""
        stats = ExecutionStats()

        assert stats.success_rate == 0.0

    def test_fallback_rate_calculation(self):
        """Test fallback rate property."""
        stats = ExecutionStats(
            total_steps=20,
            vision_fallback_steps=2,
        )

        assert stats.fallback_rate == 0.1

    def test_fallback_rate_zero_steps(self):
        """Test fallback rate with zero steps."""
        stats = ExecutionStats()

        assert stats.fallback_rate == 0.0

    def test_average_dom_duration(self):
        """Test average DOM duration property."""
        stats = ExecutionStats(
            dom_only_steps=5,
            total_dom_duration_ms=500,
        )

        assert stats.average_dom_duration_ms == 100.0

    def test_average_dom_duration_zero_steps(self):
        """Test average DOM duration with zero steps."""
        stats = ExecutionStats()

        assert stats.average_dom_duration_ms == 0.0

    def test_average_vision_duration(self):
        """Test average Vision duration property."""
        stats = ExecutionStats(
            vision_fallback_steps=3,
            vision_only_steps=2,
            total_vision_duration_ms=10000,
        )

        # 10000 / (3 + 2) = 2000
        assert stats.average_vision_duration_ms == 2000.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = ExecutionStats(
            total_steps=10,
            successful_steps=9,
            failed_steps=1,
            dom_only_steps=7,
            vision_fallback_steps=2,
            vision_only_steps=0,
            total_dom_duration_ms=700,
            total_vision_duration_ms=5000,
            total_cost=0.006,
        )

        d = stats.to_dict()

        assert d["total_steps"] == 10
        assert d["successful_steps"] == 9
        assert d["failed_steps"] == 1
        assert d["success_rate"] == 0.9
        assert d["fallback_rate"] == 0.2
        assert d["total_cost"] == 0.006
        assert "average_dom_duration_ms" in d
        assert "average_vision_duration_ms" in d

    def test_realistic_scenario(self):
        """Test with realistic test execution data."""
        # Simulate running 50 test steps:
        # - 40 pass with DOM only
        # - 5 need vision fallback (all succeed)
        # - 3 fail entirely
        # - 2 use vision-only steps
        stats = ExecutionStats(
            total_steps=50,
            successful_steps=47,
            failed_steps=3,
            dom_only_steps=40,
            vision_fallback_steps=5,
            vision_only_steps=2,
            total_dom_attempts=55,  # Some retries
            total_vision_attempts=7,
            total_dom_duration_ms=5000,  # ~125ms avg
            total_vision_duration_ms=21000,  # ~3s avg
            total_duration_ms=26000,
            total_cost=0.021,  # 7 vision calls at $0.003 each
        )

        assert stats.success_rate == 0.94
        assert stats.fallback_rate == 0.1  # 5/50
        assert stats.average_dom_duration_ms == 125.0
        assert stats.average_vision_duration_ms == 3000.0
        assert stats.total_cost == 0.021
