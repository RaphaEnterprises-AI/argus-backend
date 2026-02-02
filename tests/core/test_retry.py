"""Tests for centralized retry policy (RAP-333).

Tests the RetryPolicy class including:
- Basic retry behavior
- Exponential backoff
- Jitter strategies
- Custom retry predicates
- Async and sync execution
- Metrics collection
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.core.retry import (
    DEFAULT_RETRY_POLICY,
    FAST_RETRY_POLICY,
    LLM_RETRY_POLICY,
    NETWORK_RETRY_POLICY,
    JitterStrategy,
    RetryMetrics,
    RetryPolicy,
    create_retry_policy,
    is_rate_limit_error,
    is_server_error,
    is_transient_error,
    retry_async,
    retry_sync,
)


class TestRetryPolicy:
    """Test RetryPolicy class."""

    def test_default_initialization(self):
        """Test default policy configuration."""
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.exponential_base == 2.0
        assert policy.jitter == JitterStrategy.EQUAL

    def test_custom_initialization(self):
        """Test custom policy configuration."""
        policy = RetryPolicy(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=JitterStrategy.DECORRELATED,
        )
        assert policy.max_retries == 5
        assert policy.base_delay == 0.5
        assert policy.max_delay == 30.0
        assert policy.exponential_base == 3.0
        assert policy.jitter == JitterStrategy.DECORRELATED

    def test_validation_negative_retries(self):
        """Test validation of negative max_retries."""
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            RetryPolicy(max_retries=-1)

    def test_validation_zero_base_delay(self):
        """Test validation of zero base_delay."""
        with pytest.raises(ValueError, match="base_delay must be > 0"):
            RetryPolicy(base_delay=0)

    def test_validation_max_delay_less_than_base(self):
        """Test validation of max_delay < base_delay."""
        with pytest.raises(ValueError, match="max_delay must be >= base_delay"):
            RetryPolicy(base_delay=10.0, max_delay=5.0)

    def test_validation_exponential_base_less_than_one(self):
        """Test validation of exponential_base < 1."""
        with pytest.raises(ValueError, match="exponential_base must be >= 1"):
            RetryPolicy(exponential_base=0.5)


class TestDelayCalculation:
    """Test delay calculation with different jitter strategies."""

    def test_no_jitter(self):
        """Test delay calculation without jitter."""
        policy = RetryPolicy(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=JitterStrategy.NONE,
        )
        assert policy.calculate_delay(1) == 1.0  # 1.0 * 2^0
        assert policy.calculate_delay(2) == 2.0  # 1.0 * 2^1
        assert policy.calculate_delay(3) == 4.0  # 1.0 * 2^2
        assert policy.calculate_delay(4) == 8.0  # 1.0 * 2^3

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        policy = RetryPolicy(
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=JitterStrategy.NONE,
        )
        assert policy.calculate_delay(5) == 5.0  # 1.0 * 2^4 = 16, capped to 5

    def test_full_jitter(self):
        """Test full jitter produces values in expected range."""
        policy = RetryPolicy(
            base_delay=10.0,
            jitter=JitterStrategy.FULL,
        )
        delays = [policy.calculate_delay(1) for _ in range(100)]
        assert all(0 <= d <= 10.0 for d in delays)
        # Should have some variance (not all same value)
        assert len(set(delays)) > 1

    def test_equal_jitter(self):
        """Test equal jitter produces values in expected range."""
        policy = RetryPolicy(
            base_delay=10.0,
            jitter=JitterStrategy.EQUAL,
        )
        delays = [policy.calculate_delay(1) for _ in range(100)]
        # Equal jitter: half + random(0, half), so range is [5, 10]
        assert all(5.0 <= d <= 10.0 for d in delays)
        assert len(set(delays)) > 1

    def test_decorrelated_jitter(self):
        """Test decorrelated jitter produces reasonable values."""
        policy = RetryPolicy(
            base_delay=1.0,
            max_delay=30.0,
            jitter=JitterStrategy.DECORRELATED,
        )
        delays = [policy.calculate_delay(1) for _ in range(100)]
        assert all(0 < d <= 30.0 for d in delays)


class TestShouldRetry:
    """Test retry predicate logic."""

    def test_default_predicate_timeout(self):
        """Test default predicate retries TimeoutError."""
        policy = RetryPolicy()
        assert policy.should_retry(TimeoutError())
        assert policy.should_retry(asyncio.TimeoutError())

    def test_default_predicate_connection_error(self):
        """Test default predicate retries ConnectionError."""
        policy = RetryPolicy()
        assert policy.should_retry(ConnectionError())

    def test_default_predicate_value_error(self):
        """Test default predicate does not retry ValueError."""
        policy = RetryPolicy()
        assert not policy.should_retry(ValueError())

    def test_custom_predicate(self):
        """Test custom retry predicate."""
        policy = RetryPolicy(
            retry_on=lambda e: isinstance(e, ValueError)
        )
        assert policy.should_retry(ValueError())
        assert not policy.should_retry(TimeoutError())


class TestSyncExecution:
    """Test synchronous execution."""

    def test_success_no_retry(self):
        """Test successful execution without retry."""
        policy = RetryPolicy(max_retries=3)
        func = MagicMock(return_value="success")

        result = policy.execute_sync(func)

        assert result == "success"
        assert func.call_count == 1

    def test_success_after_retry(self):
        """Test successful execution after one retry."""
        policy = RetryPolicy(max_retries=3, base_delay=0.01)
        func = MagicMock(side_effect=[TimeoutError(), "success"])

        result = policy.execute_sync(func)

        assert result == "success"
        assert func.call_count == 2

    def test_all_retries_exhausted(self):
        """Test failure after all retries exhausted."""
        policy = RetryPolicy(max_retries=2, base_delay=0.01)
        func = MagicMock(side_effect=TimeoutError("timeout"))

        with pytest.raises(TimeoutError):
            policy.execute_sync(func)

        assert func.call_count == 3  # initial + 2 retries

    def test_non_retryable_error(self):
        """Test non-retryable error raises immediately."""
        policy = RetryPolicy(max_retries=3)
        func = MagicMock(side_effect=ValueError("invalid"))

        with pytest.raises(ValueError):
            policy.execute_sync(func)

        assert func.call_count == 1  # no retries


class TestAsyncExecution:
    """Test asynchronous execution."""

    @pytest.mark.asyncio
    async def test_async_success_no_retry(self):
        """Test successful async execution without retry."""
        policy = RetryPolicy(max_retries=3)

        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await policy.execute_async(func)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_success_after_retry(self):
        """Test successful async execution after retry."""
        policy = RetryPolicy(max_retries=3, base_delay=0.01)

        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError()
            return "success"

        result = await policy.execute_async(func)

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_all_retries_exhausted(self):
        """Test async failure after all retries exhausted."""
        policy = RetryPolicy(max_retries=2, base_delay=0.01)

        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("timeout")

        with pytest.raises(TimeoutError):
            await policy.execute_async(func)

        assert call_count == 3  # initial + 2 retries


class TestDecorators:
    """Test retry decorators."""

    def test_sync_decorator(self):
        """Test sync retry decorator."""
        call_count = 0

        @retry_sync(max_retries=2, base_delay=0.01)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError()
            return "success"

        result = func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test async retry decorator."""
        call_count = 0

        @retry_async(max_retries=2, base_delay=0.01)
        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError()
            return "success"

        result = await func()

        assert result == "success"
        assert call_count == 2


class TestRetryPredicates:
    """Test retry predicate helpers."""

    def test_is_rate_limit_error_by_name(self):
        """Test rate limit detection by exception name."""
        class RateLimitError(Exception):
            pass

        assert is_rate_limit_error(RateLimitError())

    def test_is_rate_limit_error_by_status(self):
        """Test rate limit detection by status code."""
        class HttpError(Exception):
            status_code = 429

        assert is_rate_limit_error(HttpError())

    def test_is_rate_limit_error_by_message(self):
        """Test rate limit detection by message."""
        assert is_rate_limit_error(Exception("rate limit exceeded"))
        assert is_rate_limit_error(Exception("too many requests"))

    def test_is_server_error_by_status(self):
        """Test server error detection by status code."""
        class HttpError(Exception):
            status_code = 500

        assert is_server_error(HttpError())

        class HttpError502(Exception):
            status_code = 502

        assert is_server_error(HttpError502())

    def test_is_server_error_by_message(self):
        """Test server error detection by message."""
        assert is_server_error(Exception("internal server error"))
        assert is_server_error(Exception("service unavailable"))
        assert is_server_error(Exception("bad gateway"))

    def test_is_transient_error(self):
        """Test combined transient error detection."""
        assert is_transient_error(TimeoutError())
        assert is_transient_error(ConnectionError())

        class RateLimitError(Exception):
            pass

        assert is_transient_error(RateLimitError())


class TestPreConfiguredPolicies:
    """Test pre-configured policy instances."""

    def test_default_policy(self):
        """Test DEFAULT_RETRY_POLICY configuration."""
        assert DEFAULT_RETRY_POLICY.max_retries == 3
        assert DEFAULT_RETRY_POLICY.base_delay == 1.0
        assert DEFAULT_RETRY_POLICY.max_delay == 30.0

    def test_network_policy(self):
        """Test NETWORK_RETRY_POLICY configuration."""
        assert NETWORK_RETRY_POLICY.max_retries == 5
        assert NETWORK_RETRY_POLICY.jitter == JitterStrategy.DECORRELATED

    def test_llm_policy(self):
        """Test LLM_RETRY_POLICY configuration."""
        assert LLM_RETRY_POLICY.max_retries == 2
        assert LLM_RETRY_POLICY.base_delay == 2.0

    def test_fast_policy(self):
        """Test FAST_RETRY_POLICY configuration."""
        assert FAST_RETRY_POLICY.base_delay == 0.1
        assert FAST_RETRY_POLICY.max_delay == 2.0


class TestFactoryFunction:
    """Test create_retry_policy factory function."""

    def test_default_operation_type(self):
        """Test factory with default operation type."""
        policy = create_retry_policy("default")
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0

    def test_network_operation_type(self):
        """Test factory with network operation type."""
        policy = create_retry_policy("network")
        assert policy.max_retries == 5
        assert policy.jitter == JitterStrategy.DECORRELATED

    def test_llm_operation_type(self):
        """Test factory with llm operation type."""
        policy = create_retry_policy("llm")
        assert policy.max_retries == 2
        assert policy.base_delay == 2.0

    def test_overrides(self):
        """Test factory with overrides."""
        policy = create_retry_policy("default", max_retries=10)
        assert policy.max_retries == 10

    def test_unknown_operation_type(self):
        """Test factory with unknown operation type falls back to default."""
        policy = create_retry_policy("unknown")
        assert policy.max_retries == 3  # default


class TestRetryMetrics:
    """Test RetryMetrics dataclass."""

    def test_success_metrics(self):
        """Test metrics for successful operation."""
        metrics = RetryMetrics()
        metrics.total_attempts = 1
        metrics.successful_attempt = 1

        assert metrics.success is True
        assert metrics.retries_needed == 0

    def test_retry_metrics(self):
        """Test metrics for operation with retries."""
        metrics = RetryMetrics()
        metrics.total_attempts = 3
        metrics.successful_attempt = 3

        assert metrics.success is True
        assert metrics.retries_needed == 2

    def test_failure_metrics(self):
        """Test metrics for failed operation."""
        metrics = RetryMetrics()
        metrics.total_attempts = 4
        metrics.successful_attempt = None
        metrics.errors = [
            (1, "TimeoutError", "timeout"),
            (2, "TimeoutError", "timeout"),
            (3, "TimeoutError", "timeout"),
            (4, "TimeoutError", "timeout"),
        ]

        assert metrics.success is False
        assert metrics.retries_needed == 3

    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = RetryMetrics()
        metrics.total_attempts = 2
        metrics.successful_attempt = 2
        metrics.total_delay_seconds = 1.5
        metrics.errors = [(1, "TimeoutError", "timeout message")]

        data = metrics.to_dict()

        assert data["total_attempts"] == 2
        assert data["successful_attempt"] == 2
        assert data["total_delay_seconds"] == 1.5
        assert data["success"] is True
        assert len(data["errors"]) == 1


class TestOnRetryCallback:
    """Test on_retry callback functionality."""

    def test_callback_called_on_retry(self):
        """Test that callback is called before each retry."""
        callback_calls = []

        def on_retry(attempt, exc, delay):
            callback_calls.append((attempt, type(exc).__name__, delay))

        policy = RetryPolicy(
            max_retries=2,
            base_delay=0.01,
            on_retry=on_retry,
        )

        func = MagicMock(side_effect=[TimeoutError(), TimeoutError(), "success"])

        result = policy.execute_sync(func)

        assert result == "success"
        assert len(callback_calls) == 2
        assert callback_calls[0][0] == 1  # First retry attempt
        assert callback_calls[0][1] == "TimeoutError"
        assert callback_calls[1][0] == 2  # Second retry attempt
