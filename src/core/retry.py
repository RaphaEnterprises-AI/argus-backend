"""Centralized Retry Policy with Exponential Backoff and Jitter.

RAP-333: Provides a unified retry mechanism for the entire codebase.

This module implements a centralized retry policy that:
- Uses exponential backoff with configurable base and multiplier
- Adds jitter to prevent thundering herd problems
- Supports both sync and async operations
- Provides configurable retry predicates for different error types
- Tracks retry metrics for observability

Usage:
    from src.core.retry import RetryPolicy, retry_async, retry_sync

    # Using decorator
    @retry_async(max_retries=3, base_delay=1.0)
    async def my_api_call():
        ...

    # Using policy directly
    policy = RetryPolicy(max_retries=3, base_delay=1.0)
    result = await policy.execute_async(my_api_call)

    # With custom retry predicate
    policy = RetryPolicy(
        max_retries=3,
        retry_on=lambda e: isinstance(e, (RateLimitError, TimeoutError))
    )
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, ParamSpec, TypeVar

import structlog

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class JitterStrategy(str, Enum):
    """Jitter strategies for retry delays.

    - NONE: No jitter, use exact calculated delay
    - FULL: Uniform random between 0 and calculated delay
    - EQUAL: Half fixed, half random (delay/2 + random(0, delay/2))
    - DECORRELATED: AWS-style decorrelated jitter (better distribution)
    """
    NONE = "none"
    FULL = "full"
    EQUAL = "equal"
    DECORRELATED = "decorrelated"


@dataclass
class RetryMetrics:
    """Metrics collected during retry execution.

    Useful for observability and debugging retry behavior.
    """
    total_attempts: int = 0
    successful_attempt: int | None = None
    total_delay_seconds: float = 0.0
    errors: list[tuple[int, str, str]] = field(default_factory=list)  # (attempt, error_type, message)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    @property
    def success(self) -> bool:
        """Whether the operation ultimately succeeded."""
        return self.successful_attempt is not None

    @property
    def retries_needed(self) -> int:
        """Number of retries (attempts - 1 if successful on first try)."""
        if self.successful_attempt is not None:
            return self.successful_attempt - 1
        return self.total_attempts - 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "total_attempts": self.total_attempts,
            "successful_attempt": self.successful_attempt,
            "total_delay_seconds": round(self.total_delay_seconds, 3),
            "retries_needed": self.retries_needed,
            "success": self.success,
            "errors": [
                {"attempt": a, "type": t, "message": m[:200]}
                for a, t, m in self.errors
            ],
        }


@dataclass
class RetryPolicy:
    """Centralized retry policy with exponential backoff and jitter.

    This is the primary class for retry configuration. It encapsulates all
    retry behavior in a reusable, testable way.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries, just initial attempt)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay cap in seconds (prevents excessive waits)
        exponential_base: Multiplier for exponential backoff (default 2.0)
        jitter: Jitter strategy to prevent thundering herd
        retry_on: Callable that takes an exception and returns True if should retry
        on_retry: Optional callback called before each retry with (attempt, exception, delay)

    Example:
        # Simple usage
        policy = RetryPolicy(max_retries=3, base_delay=1.0)
        result = await policy.execute_async(my_api_call)

        # With custom configuration
        policy = RetryPolicy(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=JitterStrategy.DECORRELATED,
            retry_on=lambda e: isinstance(e, (RateLimitError, TimeoutError)),
        )
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: JitterStrategy = JitterStrategy.EQUAL
    retry_on: Callable[[Exception], bool] | None = None
    on_retry: Callable[[int, Exception, float], None] | None = None

    # Internal state for decorrelated jitter
    _last_delay: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be > 0")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.exponential_base < 1:
            raise ValueError("exponential_base must be >= 1")

        # Reset decorrelated jitter state
        self._last_delay = self.base_delay

    def should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry.

        If no retry_on predicate is provided, defaults to retrying on:
        - TimeoutError / asyncio.TimeoutError
        - ConnectionError
        - OSError with network-related errno
        """
        if self.retry_on is not None:
            return self.retry_on(exception)

        # Default: retry on common transient errors
        if isinstance(exception, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
            return True

        # Retry on network-related OSError
        if isinstance(exception, OSError) and exception.errno in (
            110,  # Connection timed out
            111,  # Connection refused
            113,  # No route to host
        ):
            return True

        return False

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (1-indexed).

        Uses exponential backoff with configurable jitter strategy.

        Args:
            attempt: Current attempt number (1 = first retry after initial failure)

        Returns:
            Delay in seconds before the next retry
        """
        # Calculate base exponential delay
        exponential_delay = self.base_delay * (self.exponential_base ** (attempt - 1))

        # Apply jitter based on strategy
        if self.jitter == JitterStrategy.NONE:
            delay = exponential_delay

        elif self.jitter == JitterStrategy.FULL:
            # Full jitter: random between 0 and calculated delay
            delay = random.uniform(0, exponential_delay)

        elif self.jitter == JitterStrategy.EQUAL:
            # Equal jitter: half fixed, half random
            half = exponential_delay / 2
            delay = half + random.uniform(0, half)

        elif self.jitter == JitterStrategy.DECORRELATED:
            # AWS-style decorrelated jitter
            # sleep = min(cap, random_between(base, sleep * 3))
            delay = random.uniform(self.base_delay, self._last_delay * 3)
            self._last_delay = delay

        else:
            delay = exponential_delay

        # Apply max delay cap
        return min(delay, self.max_delay)

    async def execute_async(
        self,
        func: Callable[P, Awaitable[T]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Execute an async function with retry logic.

        Args:
            func: Async callable to execute
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result from successful function execution

        Raises:
            Last exception if all retries exhausted
        """
        metrics = RetryMetrics()
        last_exception: Exception | None = None

        # Reset decorrelated jitter state
        self._last_delay = self.base_delay

        for attempt in range(self.max_retries + 1):
            metrics.total_attempts = attempt + 1

            try:
                result = await func(*args, **kwargs)
                metrics.successful_attempt = attempt + 1
                metrics.completed_at = datetime.now(UTC)

                if attempt > 0:
                    logger.info(
                        "Operation succeeded after retries",
                        attempt=attempt + 1,
                        total_delay=round(metrics.total_delay_seconds, 3),
                    )

                return result

            except Exception as e:
                last_exception = e
                metrics.errors.append((attempt + 1, type(e).__name__, str(e)))

                # Check if we should retry
                if attempt >= self.max_retries or not self.should_retry(e):
                    metrics.completed_at = datetime.now(UTC)
                    logger.warning(
                        "Operation failed - not retrying",
                        attempt=attempt + 1,
                        error_type=type(e).__name__,
                        error=str(e)[:200],
                        should_retry=self.should_retry(e),
                        max_retries_reached=attempt >= self.max_retries,
                    )
                    raise

                # Calculate delay for next retry
                delay = self.calculate_delay(attempt + 1)
                metrics.total_delay_seconds += delay

                logger.debug(
                    "Retrying operation",
                    attempt=attempt + 1,
                    next_attempt=attempt + 2,
                    delay_seconds=round(delay, 3),
                    error_type=type(e).__name__,
                    error=str(e)[:200],
                )

                # Call on_retry callback if provided
                if self.on_retry:
                    self.on_retry(attempt + 1, e, delay)

                await asyncio.sleep(delay)

        # Should never reach here, but just in case
        metrics.completed_at = datetime.now(UTC)
        raise last_exception or RuntimeError("Retry loop completed unexpectedly")

    def execute_sync(
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Execute a sync function with retry logic.

        Args:
            func: Callable to execute
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result from successful function execution

        Raises:
            Last exception if all retries exhausted
        """
        metrics = RetryMetrics()
        last_exception: Exception | None = None

        # Reset decorrelated jitter state
        self._last_delay = self.base_delay

        for attempt in range(self.max_retries + 1):
            metrics.total_attempts = attempt + 1

            try:
                result = func(*args, **kwargs)
                metrics.successful_attempt = attempt + 1
                metrics.completed_at = datetime.now(UTC)

                if attempt > 0:
                    logger.info(
                        "Operation succeeded after retries",
                        attempt=attempt + 1,
                        total_delay=round(metrics.total_delay_seconds, 3),
                    )

                return result

            except Exception as e:
                last_exception = e
                metrics.errors.append((attempt + 1, type(e).__name__, str(e)))

                # Check if we should retry
                if attempt >= self.max_retries or not self.should_retry(e):
                    metrics.completed_at = datetime.now(UTC)
                    logger.warning(
                        "Operation failed - not retrying",
                        attempt=attempt + 1,
                        error_type=type(e).__name__,
                        error=str(e)[:200],
                        should_retry=self.should_retry(e),
                        max_retries_reached=attempt >= self.max_retries,
                    )
                    raise

                # Calculate delay for next retry
                delay = self.calculate_delay(attempt + 1)
                metrics.total_delay_seconds += delay

                logger.debug(
                    "Retrying operation",
                    attempt=attempt + 1,
                    next_attempt=attempt + 2,
                    delay_seconds=round(delay, 3),
                    error_type=type(e).__name__,
                    error=str(e)[:200],
                )

                # Call on_retry callback if provided
                if self.on_retry:
                    self.on_retry(attempt + 1, e, delay)

                time.sleep(delay)

        # Should never reach here, but just in case
        metrics.completed_at = datetime.now(UTC)
        raise last_exception or RuntimeError("Retry loop completed unexpectedly")


# =============================================================================
# Pre-configured Policies for Common Use Cases
# =============================================================================

# Default policy for general API calls
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    jitter=JitterStrategy.EQUAL,
)

# Aggressive retry for transient network issues
NETWORK_RETRY_POLICY = RetryPolicy(
    max_retries=5,
    base_delay=0.5,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=JitterStrategy.DECORRELATED,
    retry_on=lambda e: isinstance(e, (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)),
)

# Conservative retry for expensive operations (e.g., LLM calls)
LLM_RETRY_POLICY = RetryPolicy(
    max_retries=2,
    base_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=JitterStrategy.EQUAL,
)

# Fast retry for quick operations
FAST_RETRY_POLICY = RetryPolicy(
    max_retries=3,
    base_delay=0.1,
    max_delay=2.0,
    jitter=JitterStrategy.FULL,
)


# =============================================================================
# Decorator Functions
# =============================================================================

def retry_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: JitterStrategy = JitterStrategy.EQUAL,
    retry_on: Callable[[Exception], bool] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to add retry logic to async functions.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Multiplier for exponential backoff
        jitter: Jitter strategy to use
        retry_on: Predicate for retryable exceptions
        on_retry: Callback called before each retry

    Example:
        @retry_async(max_retries=3, base_delay=1.0)
        async def call_api():
            ...
    """
    policy = RetryPolicy(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_on=retry_on,
        on_retry=on_retry,
    )

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await policy.execute_async(func, *args, **kwargs)
        return wrapper

    return decorator


def retry_sync(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: JitterStrategy = JitterStrategy.EQUAL,
    retry_on: Callable[[Exception], bool] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to add retry logic to sync functions.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Multiplier for exponential backoff
        jitter: Jitter strategy to use
        retry_on: Predicate for retryable exceptions
        on_retry: Callback called before each retry

    Example:
        @retry_sync(max_retries=3, base_delay=1.0)
        def call_api():
            ...
    """
    policy = RetryPolicy(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_on=retry_on,
        on_retry=on_retry,
    )

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return policy.execute_sync(func, *args, **kwargs)
        return wrapper

    return decorator


# =============================================================================
# Retry Predicates for Common Error Types
# =============================================================================

def is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception is a rate limit error.

    Supports common patterns across different libraries.
    """
    # Check by exception name (works for dynamically imported classes)
    error_name = type(exception).__name__.lower()
    if "ratelimit" in error_name or "rate_limit" in error_name:
        return True

    # Check for HTTP 429 status
    if hasattr(exception, "status_code") and exception.status_code == 429:
        return True
    if hasattr(exception, "status") and exception.status == 429:
        return True

    # Check error message
    message = str(exception).lower()
    if "rate limit" in message or "too many requests" in message:
        return True

    return False


def is_server_error(exception: Exception) -> bool:
    """Check if exception indicates a server error (5xx).

    These are typically transient and worth retrying.
    """
    # Check HTTP status code
    status = None
    if hasattr(exception, "status_code"):
        status = exception.status_code
    elif hasattr(exception, "status"):
        status = exception.status

    if status and 500 <= status < 600:
        return True

    # Check error message for common server error indicators
    message = str(exception).lower()
    if any(term in message for term in ["internal server error", "service unavailable", "bad gateway"]):
        return True

    return False


def is_transient_error(exception: Exception) -> bool:
    """Check if exception is a transient error worth retrying.

    Combines rate limits, server errors, and network issues.
    """
    return (
        is_rate_limit_error(exception) or
        is_server_error(exception) or
        isinstance(exception, (TimeoutError, asyncio.TimeoutError, ConnectionError))
    )


# =============================================================================
# Factory Functions
# =============================================================================

def create_retry_policy(
    operation_type: str = "default",
    **overrides,
) -> RetryPolicy:
    """Create a retry policy based on operation type.

    Args:
        operation_type: Type of operation - "default", "network", "llm", "fast"
        **overrides: Override any policy parameter

    Returns:
        Configured RetryPolicy

    Example:
        policy = create_retry_policy("llm", max_retries=4)
    """
    base_configs = {
        "default": {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "jitter": JitterStrategy.EQUAL,
        },
        "network": {
            "max_retries": 5,
            "base_delay": 0.5,
            "max_delay": 30.0,
            "jitter": JitterStrategy.DECORRELATED,
            "retry_on": is_transient_error,
        },
        "llm": {
            "max_retries": 2,
            "base_delay": 2.0,
            "max_delay": 60.0,
            "jitter": JitterStrategy.EQUAL,
            "retry_on": is_transient_error,
        },
        "fast": {
            "max_retries": 3,
            "base_delay": 0.1,
            "max_delay": 2.0,
            "jitter": JitterStrategy.FULL,
        },
    }

    config = base_configs.get(operation_type, base_configs["default"]).copy()
    config.update(overrides)

    return RetryPolicy(**config)
