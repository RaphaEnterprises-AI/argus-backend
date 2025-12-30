"""Structured logging configuration for E2E testing agent.

Provides:
- Structured logging with structlog
- Context-aware logging
- Log levels and formatting
- Integration with test execution
"""

import logging
import sys
from contextlib import contextmanager
from typing import Any, Optional

import structlog


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    include_timestamp: bool = True,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Output logs as JSON
        include_timestamp: Include timestamps in logs
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Build processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso") if include_timestamp else structlog.processors.TimeStamper(fmt=None),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None, **context) -> structlog.BoundLogger:
    """Get a configured logger with optional context.

    Args:
        name: Logger name
        **context: Additional context to bind

    Returns:
        Configured structlog logger
    """
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger


class LogContext:
    """Context manager for scoped logging context.

    Usage:
        with LogContext(test_id="test-123", component="ui_tester"):
            logger.info("Running test")
            # All logs within this block have test_id and component bound
    """

    def __init__(self, **context):
        """Initialize with context to bind.

        Args:
            **context: Key-value pairs to bind to logs
        """
        self.context = context
        self._token = None

    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


@contextmanager
def log_operation(
    operation: str,
    logger: Optional[structlog.BoundLogger] = None,
    **context,
):
    """Context manager for logging operation start/end.

    Args:
        operation: Name of the operation
        logger: Optional logger to use
        **context: Additional context

    Yields:
        Dict to store operation results

    Example:
        with log_operation("execute_test", test_id="123") as op:
            result = run_test()
            op["result"] = result.status
    """
    log = logger or get_logger()
    log = log.bind(operation=operation, **context)

    log.info(f"{operation} started")
    result = {"success": False, "error": None}

    try:
        yield result
        result["success"] = True
        log.info(f"{operation} completed", **result)
    except Exception as e:
        result["error"] = str(e)
        log.error(f"{operation} failed", **result)
        raise


class TestExecutionLogger:
    """Logger specialized for test execution tracking.

    Provides structured logging for:
    - Test start/end
    - Step execution
    - Assertions
    - Screenshots
    - Failures
    """

    def __init__(self, test_id: str, test_name: str):
        """Initialize test logger.

        Args:
            test_id: Unique test identifier
            test_name: Human-readable test name
        """
        self.log = get_logger().bind(
            test_id=test_id,
            test_name=test_name,
        )
        self.step_count = 0
        self.assertion_count = 0

    def test_started(self, metadata: Optional[dict] = None) -> None:
        """Log test start."""
        self.log.info("Test started", **(metadata or {}))

    def test_completed(self, status: str, duration_ms: int) -> None:
        """Log test completion."""
        self.log.info(
            "Test completed",
            status=status,
            duration_ms=duration_ms,
            steps_executed=self.step_count,
            assertions_checked=self.assertion_count,
        )

    def step_started(self, step_index: int, action: str, target: Optional[str] = None) -> None:
        """Log step start."""
        self.step_count = step_index + 1
        self.log.debug(
            "Step started",
            step_index=step_index,
            action=action,
            target=target,
        )

    def step_completed(self, step_index: int, action: str, duration_ms: int) -> None:
        """Log step completion."""
        self.log.debug(
            "Step completed",
            step_index=step_index,
            action=action,
            duration_ms=duration_ms,
        )

    def step_failed(self, step_index: int, action: str, error: str) -> None:
        """Log step failure."""
        self.log.error(
            "Step failed",
            step_index=step_index,
            action=action,
            error=error,
        )

    def assertion_checked(self, assertion_type: str, passed: bool, details: Optional[dict] = None) -> None:
        """Log assertion check."""
        self.assertion_count += 1
        level = self.log.debug if passed else self.log.warning
        level(
            "Assertion checked",
            assertion_type=assertion_type,
            passed=passed,
            **(details or {}),
        )

    def screenshot_taken(self, path: Optional[str] = None, size_bytes: Optional[int] = None) -> None:
        """Log screenshot capture."""
        self.log.debug(
            "Screenshot taken",
            path=path,
            size_bytes=size_bytes,
        )

    def warning(self, message: str, **context) -> None:
        """Log a warning."""
        self.log.warning(message, **context)

    def error(self, message: str, **context) -> None:
        """Log an error."""
        self.log.error(message, **context)
