"""Tests for the token counting and cost tracking module."""

import pytest
from datetime import datetime
from unittest.mock import patch


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_estimate_tokens_empty_text(self, mock_env_vars):
        """Test token estimation for empty text."""
        from src.utils.tokens import estimate_tokens

        assert estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self, mock_env_vars):
        """Test token estimation for short text."""
        from src.utils.tokens import estimate_tokens

        # "Hello" is 5 chars, should be ~2 tokens (5/4 + 1)
        tokens = estimate_tokens("Hello")
        assert tokens >= 1

    def test_estimate_tokens_long_text(self, mock_env_vars):
        """Test token estimation for longer text."""
        from src.utils.tokens import estimate_tokens

        text = "This is a longer piece of text that should have more tokens."
        tokens = estimate_tokens(text)

        # Approximately len/4 + 1
        expected = len(text) // 4 + 1
        assert tokens == expected

    def test_estimate_tokens_with_model(self, mock_env_vars):
        """Test token estimation with model parameter."""
        from src.utils.tokens import estimate_tokens

        tokens = estimate_tokens("Test text", model="claude-opus-4-5")
        assert tokens > 0


class TestEstimateImageTokens:
    """Tests for estimate_image_tokens function."""

    def test_small_image(self, mock_env_vars):
        """Test token estimation for small image."""
        from src.utils.tokens import estimate_image_tokens

        tokens = estimate_image_tokens(width=640, height=480)
        assert tokens > 0

    def test_medium_image(self, mock_env_vars):
        """Test token estimation for medium image."""
        from src.utils.tokens import estimate_image_tokens

        tokens = estimate_image_tokens(width=1280, height=720)
        assert tokens > 0

    def test_large_image(self, mock_env_vars):
        """Test token estimation for large image (1920x1080)."""
        from src.utils.tokens import estimate_image_tokens

        tokens = estimate_image_tokens(width=1920, height=1080)
        assert tokens > 0

    def test_extra_large_image(self, mock_env_vars):
        """Test token estimation for extra large image."""
        from src.utils.tokens import estimate_image_tokens

        # Should cap at 4x base tokens
        tokens = estimate_image_tokens(width=3840, height=2160)
        assert tokens > 0

    def test_with_detail_parameter(self, mock_env_vars):
        """Test token estimation with detail parameter."""
        from src.utils.tokens import estimate_image_tokens

        tokens_auto = estimate_image_tokens(width=1920, height=1080, detail="auto")
        tokens_high = estimate_image_tokens(width=1920, height=1080, detail="high")

        assert tokens_auto > 0
        assert tokens_high > 0


class TestEstimateScreenshotTokens:
    """Tests for estimate_screenshot_tokens function."""

    def test_default_resolution(self, mock_env_vars):
        """Test screenshot tokens with default resolution."""
        from src.utils.tokens import estimate_screenshot_tokens

        tokens = estimate_screenshot_tokens(b"fake_screenshot_data")
        assert tokens > 0

    def test_custom_resolution(self, mock_env_vars):
        """Test screenshot tokens with custom resolution."""
        from src.utils.tokens import estimate_screenshot_tokens

        tokens = estimate_screenshot_tokens(
            b"fake_screenshot_data",
            resolution=(2560, 1440),
        )
        assert tokens > 0


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_token_usage_creation(self, mock_env_vars):
        """Test TokenUsage creation."""
        from src.utils.tokens import TokenUsage

        usage = TokenUsage(
            timestamp=datetime.now(),
            model="claude-sonnet-4-5",
            input_tokens=1000,
            output_tokens=500,
            cost=0.015,
        )

        assert usage.model == "claude-sonnet-4-5"
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.operation == ""
        assert usage.metadata == {}

    def test_token_usage_with_metadata(self, mock_env_vars):
        """Test TokenUsage with metadata."""
        from src.utils.tokens import TokenUsage

        usage = TokenUsage(
            timestamp=datetime.now(),
            model="claude-sonnet-4-5",
            input_tokens=1000,
            output_tokens=500,
            cost=0.015,
            operation="test_execution",
            metadata={"test_id": "123"},
        )

        assert usage.operation == "test_execution"
        assert usage.metadata["test_id"] == "123"


class TestUsageSummary:
    """Tests for UsageSummary dataclass."""

    def test_usage_summary_defaults(self, mock_env_vars):
        """Test UsageSummary defaults."""
        from src.utils.tokens import UsageSummary

        summary = UsageSummary()

        assert summary.total_input_tokens == 0
        assert summary.total_output_tokens == 0
        assert summary.total_cost == 0.0
        assert summary.call_count == 0
        assert summary.by_model == {}
        assert summary.by_operation == {}


class TestTokenCounter:
    """Tests for TokenCounter class."""

    def test_token_counter_creation(self, mock_env_vars):
        """Test TokenCounter creation."""
        from src.utils.tokens import TokenCounter

        counter = TokenCounter()

        assert counter.total_cost == 0.0
        assert counter.total_tokens == 0

    def test_add_usage_known_model(self, mock_env_vars):
        """Test add_usage with known model."""
        from src.utils.tokens import TokenCounter

        counter = TokenCounter()
        cost = counter.add_usage(
            model="claude-sonnet-4-5-20250514",
            input_tokens=1000,
            output_tokens=500,
        )

        assert cost > 0
        assert counter.total_cost > 0
        assert counter.total_tokens == 1500

    def test_add_usage_unknown_model(self, mock_env_vars):
        """Test add_usage with unknown model (falls back to Sonnet pricing)."""
        from src.utils.tokens import TokenCounter

        counter = TokenCounter()
        cost = counter.add_usage(
            model="unknown-model-xyz",
            input_tokens=1000,
            output_tokens=500,
        )

        assert cost > 0

    def test_add_usage_with_operation(self, mock_env_vars):
        """Test add_usage with operation name."""
        from src.utils.tokens import TokenCounter

        counter = TokenCounter()
        counter.add_usage(
            model="claude-sonnet-4-5-20250514",
            input_tokens=1000,
            output_tokens=500,
            operation="test_execution",
        )

        summary = counter.get_summary()
        assert "test_execution" in summary.by_operation

    def test_add_usage_with_metadata(self, mock_env_vars):
        """Test add_usage with metadata."""
        from src.utils.tokens import TokenCounter

        counter = TokenCounter()
        counter.add_usage(
            model="claude-sonnet-4-5-20250514",
            input_tokens=1000,
            output_tokens=500,
            test_id="123",
            step="login",
        )

        log = counter.get_usage_log()
        assert len(log) == 1

    def test_get_summary(self, mock_env_vars):
        """Test get_summary method."""
        from src.utils.tokens import TokenCounter

        counter = TokenCounter()
        counter.add_usage(
            model="claude-sonnet-4-5-20250514",
            input_tokens=1000,
            output_tokens=500,
            operation="op1",
        )
        counter.add_usage(
            model="claude-sonnet-4-5-20250514",
            input_tokens=2000,
            output_tokens=1000,
            operation="op2",
        )

        summary = counter.get_summary()

        assert summary.total_input_tokens == 3000
        assert summary.total_output_tokens == 1500
        assert summary.call_count == 2
        assert "claude-sonnet-4-5-20250514" in summary.by_model
        assert "op1" in summary.by_operation
        assert "op2" in summary.by_operation

    def test_total_cost_property(self, mock_env_vars):
        """Test total_cost property."""
        from src.utils.tokens import TokenCounter

        counter = TokenCounter()
        counter.add_usage(
            model="claude-sonnet-4-5-20250514",
            input_tokens=1000,
            output_tokens=500,
        )

        assert counter.total_cost > 0

    def test_total_tokens_property(self, mock_env_vars):
        """Test total_tokens property."""
        from src.utils.tokens import TokenCounter

        counter = TokenCounter()
        counter.add_usage(
            model="claude-sonnet-4-5-20250514",
            input_tokens=1000,
            output_tokens=500,
        )

        assert counter.total_tokens == 1500

    def test_reset(self, mock_env_vars):
        """Test reset method."""
        from src.utils.tokens import TokenCounter

        counter = TokenCounter()
        counter.add_usage(
            model="claude-sonnet-4-5-20250514",
            input_tokens=1000,
            output_tokens=500,
        )

        counter.reset()

        assert counter.total_cost == 0.0
        assert counter.total_tokens == 0

    def test_get_usage_log(self, mock_env_vars):
        """Test get_usage_log method."""
        from src.utils.tokens import TokenCounter

        counter = TokenCounter()
        counter.add_usage(
            model="claude-sonnet-4-5-20250514",
            input_tokens=1000,
            output_tokens=500,
            operation="test",
        )

        log = counter.get_usage_log()

        assert len(log) == 1
        assert log[0]["model"] == "claude-sonnet-4-5-20250514"
        assert log[0]["input_tokens"] == 1000
        assert log[0]["output_tokens"] == 500
        assert log[0]["operation"] == "test"
        assert "timestamp" in log[0]
        assert "cost" in log[0]


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_cost_tracker_creation(self, mock_env_vars):
        """Test CostTracker creation."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker()

        assert tracker.budget_limit == 10.0
        assert tracker.warning_threshold == 0.8
        assert tracker.total_cost == 0.0

    def test_cost_tracker_custom_limit(self, mock_env_vars):
        """Test CostTracker with custom limits."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker(budget_limit=50.0, warning_threshold=0.9)

        assert tracker.budget_limit == 50.0
        assert tracker.warning_threshold == 0.9

    def test_record_cost(self, mock_env_vars):
        """Test record_cost method."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker()
        tracker.record_cost(1.5)

        assert tracker.total_cost == 1.5

    def test_record_cost_with_operation(self, mock_env_vars):
        """Test record_cost with operation."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker()
        tracker.record_cost(1.5, operation="test_execution")

        breakdown = tracker.get_cost_breakdown()
        assert breakdown["test_execution"] == 1.5

    def test_can_afford_within_budget(self, mock_env_vars):
        """Test can_afford when within budget."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker(budget_limit=10.0)
        tracker.record_cost(5.0)

        assert tracker.can_afford(3.0) is True

    def test_can_afford_exceeds_budget(self, mock_env_vars):
        """Test can_afford when would exceed budget."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker(budget_limit=10.0)
        tracker.record_cost(8.0)

        assert tracker.can_afford(5.0) is False

    def test_is_budget_exceeded_false(self, mock_env_vars):
        """Test is_budget_exceeded when under budget."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker(budget_limit=10.0)
        tracker.record_cost(5.0)

        assert tracker.is_budget_exceeded() is False

    def test_is_budget_exceeded_true(self, mock_env_vars):
        """Test is_budget_exceeded when at/over budget."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker(budget_limit=10.0)
        tracker.record_cost(10.0)

        assert tracker.is_budget_exceeded() is True

    def test_is_warning_threshold_reached_false(self, mock_env_vars):
        """Test is_warning_threshold_reached when below threshold."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker(budget_limit=10.0, warning_threshold=0.8)
        tracker.record_cost(5.0)

        assert tracker.is_warning_threshold_reached() is False

    def test_is_warning_threshold_reached_true(self, mock_env_vars):
        """Test is_warning_threshold_reached when at/above threshold."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker(budget_limit=10.0, warning_threshold=0.8)
        tracker.record_cost(8.5)

        assert tracker.is_warning_threshold_reached() is True

    def test_remaining_budget(self, mock_env_vars):
        """Test remaining_budget property."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker(budget_limit=10.0)
        tracker.record_cost(3.5)

        assert tracker.remaining_budget == 6.5

    def test_remaining_budget_negative(self, mock_env_vars):
        """Test remaining_budget when over budget (should be 0)."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker(budget_limit=10.0)
        tracker.record_cost(15.0)

        assert tracker.remaining_budget == 0

    def test_budget_usage_percent(self, mock_env_vars):
        """Test budget_usage_percent property."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker(budget_limit=10.0)
        tracker.record_cost(5.0)

        assert tracker.budget_usage_percent == 50.0

    def test_budget_usage_percent_zero_budget(self, mock_env_vars):
        """Test budget_usage_percent with zero budget."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker(budget_limit=0.0)

        assert tracker.budget_usage_percent == 100.0

    def test_reset(self, mock_env_vars):
        """Test reset method."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker()
        tracker.record_cost(5.0)
        tracker.reset()

        assert tracker.total_cost == 0.0
        assert len(tracker.get_cost_breakdown()) == 0

    def test_get_cost_breakdown(self, mock_env_vars):
        """Test get_cost_breakdown method."""
        from src.utils.tokens import CostTracker

        tracker = CostTracker()
        tracker.record_cost(1.0, operation="op1")
        tracker.record_cost(2.0, operation="op1")
        tracker.record_cost(3.0, operation="op2")
        tracker.record_cost(0.5)  # No operation

        breakdown = tracker.get_cost_breakdown()

        assert breakdown["op1"] == 3.0
        assert breakdown["op2"] == 3.0
        assert breakdown["unknown"] == 0.5


class TestBudgetExceeded:
    """Tests for BudgetExceeded exception."""

    def test_budget_exceeded_creation(self, mock_env_vars):
        """Test BudgetExceeded exception creation."""
        from src.utils.tokens import BudgetExceeded

        exc = BudgetExceeded(current_cost=15.5, budget_limit=10.0)

        assert exc.current_cost == 15.5
        assert exc.budget_limit == 10.0
        assert "15.5" in str(exc)
        assert "10.0" in str(exc)

    def test_budget_exceeded_raise(self, mock_env_vars):
        """Test raising BudgetExceeded."""
        from src.utils.tokens import BudgetExceeded

        with pytest.raises(BudgetExceeded) as exc_info:
            raise BudgetExceeded(current_cost=20.0, budget_limit=10.0)

        assert exc_info.value.current_cost == 20.0
        assert exc_info.value.budget_limit == 10.0
