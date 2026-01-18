"""Tests for the logging utility module."""


import pytest


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_default(self, mock_env_vars):
        """Test configure_logging with defaults."""
        from src.utils.logging import configure_logging

        # Should not raise
        configure_logging()

    def test_configure_logging_debug_level(self, mock_env_vars):
        """Test configure_logging with DEBUG level."""
        from src.utils.logging import configure_logging

        configure_logging(level="DEBUG")

    def test_configure_logging_warning_level(self, mock_env_vars):
        """Test configure_logging with WARNING level."""
        from src.utils.logging import configure_logging

        configure_logging(level="WARNING")

    def test_configure_logging_json_format(self, mock_env_vars):
        """Test configure_logging with JSON output."""
        from src.utils.logging import configure_logging

        configure_logging(json_format=True)

    def test_configure_logging_no_timestamp(self, mock_env_vars):
        """Test configure_logging without timestamps."""
        from src.utils.logging import configure_logging

        configure_logging(include_timestamp=False)

    def test_configure_logging_all_options(self, mock_env_vars):
        """Test configure_logging with all options."""
        from src.utils.logging import configure_logging

        configure_logging(
            level="ERROR",
            json_format=True,
            include_timestamp=True,
        )


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_default(self, mock_env_vars):
        """Test get_logger without name."""
        from src.utils.logging import configure_logging, get_logger

        configure_logging()
        logger = get_logger()

        assert logger is not None

    def test_get_logger_with_name(self, mock_env_vars):
        """Test get_logger with name."""
        from src.utils.logging import configure_logging, get_logger

        configure_logging()
        logger = get_logger("test_logger")

        assert logger is not None

    def test_get_logger_with_context(self, mock_env_vars):
        """Test get_logger with context."""
        from src.utils.logging import configure_logging, get_logger

        configure_logging()
        logger = get_logger("test_logger", test_id="123", component="ui")

        assert logger is not None

    def test_get_logger_multiple_context(self, mock_env_vars):
        """Test get_logger with multiple context values."""
        from src.utils.logging import configure_logging, get_logger

        configure_logging()
        logger = get_logger(
            "test_logger",
            test_id="test-001",
            component="api_tester",
            user="test_user",
            environment="test",
        )

        assert logger is not None


class TestLogContext:
    """Tests for LogContext class."""

    def test_log_context_creation(self, mock_env_vars):
        """Test LogContext creation."""
        from src.utils.logging import LogContext

        context = LogContext(test_id="123", component="test")

        assert context.context == {"test_id": "123", "component": "test"}

    def test_log_context_enter_exit(self, mock_env_vars):
        """Test LogContext as context manager."""
        from src.utils.logging import LogContext, configure_logging

        configure_logging()

        with LogContext(test_id="456", action="test"):
            pass  # Context should be bound and then unbound

    def test_log_context_nested(self, mock_env_vars):
        """Test nested LogContext."""
        from src.utils.logging import LogContext, configure_logging

        configure_logging()

        with LogContext(level1="a"):
            with LogContext(level2="b"):
                pass

    def test_log_context_empty(self, mock_env_vars):
        """Test LogContext with no context."""
        from src.utils.logging import LogContext, configure_logging

        configure_logging()

        with LogContext():
            pass


class TestLogOperation:
    """Tests for log_operation context manager."""

    def test_log_operation_success(self, mock_env_vars):
        """Test log_operation on success."""
        from src.utils.logging import configure_logging, log_operation

        configure_logging()

        with log_operation("test_op") as op:
            op["result"] = "success"

        assert op["success"] is True
        assert op["error"] is None

    def test_log_operation_with_context(self, mock_env_vars):
        """Test log_operation with context."""
        from src.utils.logging import configure_logging, log_operation

        configure_logging()

        with log_operation("test_op", test_id="123") as op:
            op["data"] = "value"

        assert op["success"] is True

    def test_log_operation_with_logger(self, mock_env_vars):
        """Test log_operation with custom logger."""
        from src.utils.logging import configure_logging, get_logger, log_operation

        configure_logging()
        logger = get_logger("custom")

        with log_operation("test_op", logger=logger) as op:
            pass

        assert op["success"] is True

    def test_log_operation_failure(self, mock_env_vars):
        """Test log_operation on failure."""
        from src.utils.logging import configure_logging, log_operation

        configure_logging()

        with pytest.raises(ValueError):
            with log_operation("failing_op"):
                raise ValueError("Test error")


class TestTestExecutionLogger:
    """Tests for TestExecutionLogger class."""

    def test_logger_creation(self, mock_env_vars):
        """Test TestExecutionLogger creation."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        assert logger.step_count == 0
        assert logger.assertion_count == 0

    def test_test_started(self, mock_env_vars):
        """Test test_started method."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        logger.test_started()
        logger.test_started(metadata={"browser": "chrome", "viewport": "1920x1080"})

    def test_test_completed(self, mock_env_vars):
        """Test test_completed method."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        logger.test_completed(status="passed", duration_ms=5000)

    def test_step_started(self, mock_env_vars):
        """Test step_started method."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        logger.step_started(step_index=0, action="goto", target="/login")

        assert logger.step_count == 1

    def test_step_started_no_target(self, mock_env_vars):
        """Test step_started without target."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        logger.step_started(step_index=0, action="wait")

    def test_step_completed(self, mock_env_vars):
        """Test step_completed method."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        logger.step_completed(step_index=0, action="click", duration_ms=150)

    def test_step_failed(self, mock_env_vars):
        """Test step_failed method."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        logger.step_failed(step_index=0, action="click", error="Element not found")

    def test_assertion_checked_passed(self, mock_env_vars):
        """Test assertion_checked when passed."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        logger.assertion_checked(assertion_type="element_visible", passed=True)

        assert logger.assertion_count == 1

    def test_assertion_checked_failed(self, mock_env_vars):
        """Test assertion_checked when failed."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        logger.assertion_checked(
            assertion_type="url_contains",
            passed=False,
            details={"expected": "/dashboard", "actual": "/login"},
        )

        assert logger.assertion_count == 1

    def test_screenshot_taken(self, mock_env_vars):
        """Test screenshot_taken method."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        logger.screenshot_taken()
        logger.screenshot_taken(path="/tmp/screenshot.png", size_bytes=50000)

    def test_warning(self, mock_env_vars):
        """Test warning method."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        logger.warning("Element slow to respond", delay_ms=5000)

    def test_error(self, mock_env_vars):
        """Test error method."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        logger.error("Test failed", reason="Element not found")

    def test_full_test_lifecycle(self, mock_env_vars):
        """Test full test execution lifecycle."""
        from src.utils.logging import TestExecutionLogger, configure_logging

        configure_logging()
        logger = TestExecutionLogger(test_id="test-001", test_name="Login Test")

        # Start test
        logger.test_started(metadata={"browser": "chrome"})

        # Execute steps
        logger.step_started(0, "goto", "/login")
        logger.step_completed(0, "goto", 200)

        logger.step_started(1, "fill", "#email")
        logger.step_completed(1, "fill", 50)

        logger.step_started(2, "click", "#submit")
        logger.step_completed(2, "click", 100)

        # Check assertions
        logger.assertion_checked("url_contains", True)
        logger.assertion_checked("element_visible", True)

        # Take screenshot
        logger.screenshot_taken(path="/tmp/final.png")

        # Complete test
        logger.test_completed("passed", 5000)

        assert logger.step_count == 3
        assert logger.assertion_count == 2
