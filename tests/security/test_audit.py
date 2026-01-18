"""Tests for the security audit module."""

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_event_types_exist(self, mock_env_vars):
        """Test all event types are defined."""
        from src.security.audit import AuditEventType

        # AI Interactions
        assert AuditEventType.AI_REQUEST == "ai_request"
        assert AuditEventType.AI_RESPONSE == "ai_response"
        assert AuditEventType.AI_ERROR == "ai_error"

        # Data Access
        assert AuditEventType.FILE_READ == "file_read"
        assert AuditEventType.FILE_WRITE == "file_write"
        assert AuditEventType.CODE_SANITIZED == "code_sanitized"
        assert AuditEventType.SECRET_DETECTED == "secret_detected"

        # Test Execution
        assert AuditEventType.TEST_STARTED == "test_started"
        assert AuditEventType.TEST_COMPLETED == "test_completed"
        assert AuditEventType.TEST_FAILED == "test_failed"
        assert AuditEventType.BROWSER_ACTION == "browser_action"

        # Authentication
        assert AuditEventType.USER_LOGIN == "user_login"
        assert AuditEventType.USER_LOGOUT == "user_logout"

        # System
        assert AuditEventType.CONFIG_CHANGED == "config_changed"
        assert AuditEventType.INTEGRATION_CONNECTED == "integration_connected"
        assert AuditEventType.INTEGRATION_ERROR == "integration_error"


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_audit_event_creation_defaults(self, mock_env_vars):
        """Test AuditEvent creation with defaults."""
        from src.security.audit import AuditEvent, AuditEventType

        event = AuditEvent()

        assert event.id is not None
        assert event.timestamp is not None
        assert event.event_type == AuditEventType.AI_REQUEST
        assert event.user_id is None
        assert event.success is True
        assert event.data_classification == "internal"
        assert event.retention_days == 90

    def test_audit_event_creation_custom(self, mock_env_vars):
        """Test AuditEvent creation with custom values."""
        from src.security.audit import AuditEvent, AuditEventType

        event = AuditEvent(
            event_type=AuditEventType.FILE_READ,
            user_id="user-123",
            session_id="session-456",
            action="read",
            resource="/path/to/file.py",
            resource_type="file",
            model="claude-sonnet",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.05,
            success=True,
            data_classification="confidential",
        )

        assert event.event_type == AuditEventType.FILE_READ
        assert event.user_id == "user-123"
        assert event.session_id == "session-456"
        assert event.input_tokens == 100
        assert event.output_tokens == 200
        assert event.cost_usd == 0.05

    def test_audit_event_to_dict(self, mock_env_vars):
        """Test AuditEvent to_dict method."""
        from src.security.audit import AuditEvent, AuditEventType

        event = AuditEvent(
            event_type=AuditEventType.AI_REQUEST,
            user_id="user-123",
            action="analyze",
        )

        result = event.to_dict()

        assert isinstance(result, dict)
        assert result["user_id"] == "user-123"
        assert result["action"] == "analyze"

    def test_audit_event_to_json(self, mock_env_vars):
        """Test AuditEvent to_json method."""
        from src.security.audit import AuditEvent, AuditEventType

        event = AuditEvent(
            event_type=AuditEventType.AI_REQUEST,
            user_id="user-123",
        )

        json_str = event.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["event_type"] == "ai_request"
        assert data["user_id"] == "user-123"

    def test_audit_event_from_dict(self, mock_env_vars):
        """Test AuditEvent from_dict method."""
        from src.security.audit import AuditEvent, AuditEventType

        data = {
            "id": "test-id",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "event_type": "file_read",
            "user_id": "user-123",
            "action": "read",
            "resource": "/file.py",
            "resource_type": "file",
            "metadata": {},
            "success": True,
            "data_classification": "internal",
            "retention_days": 90,
        }

        event = AuditEvent.from_dict(data)

        assert event.id == "test-id"
        assert event.event_type == AuditEventType.FILE_READ
        assert event.user_id == "user-123"


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_audit_logger_creation(self, mock_env_vars):
        """Test AuditLogger creation."""
        from src.security.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            assert logger.output_dir == Path(tmpdir)
            assert logger.log_to_file is True
            assert logger.log_to_stdout is False

    def test_audit_logger_creation_with_options(self, mock_env_vars):
        """Test AuditLogger creation with custom options."""
        from src.security.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(
                output_dir=tmpdir,
                log_to_stdout=True,
                log_to_file=True,
                max_file_size_mb=50,
                retention_days=30,
            )

            assert logger.log_to_stdout is True
            assert logger.retention_days == 30

    def test_log_event(self, mock_env_vars):
        """Test log_event method."""
        from src.security.audit import AuditEvent, AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            event = AuditEvent(
                event_type=AuditEventType.AI_REQUEST,
                user_id="user-123",
            )

            logger.log_event(event)

            # Check that log file was created
            log_files = list(Path(tmpdir).glob("audit-*.jsonl"))
            assert len(log_files) == 1

            # Check content
            with open(log_files[0]) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["user_id"] == "user-123"

    def test_log_event_to_stdout(self, mock_env_vars, capsys):
        """Test log_event to stdout."""
        from src.security.audit import AuditEvent, AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir, log_to_stdout=True)

            event = AuditEvent(
                event_type=AuditEventType.AI_REQUEST,
                user_id="user-123",
            )

            logger.log_event(event)

            captured = capsys.readouterr()
            assert "AUDIT:" in captured.out
            assert "user-123" in captured.out

    def test_log_ai_request(self, mock_env_vars):
        """Test log_ai_request method."""
        from src.security.audit import AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            event = logger.log_ai_request(
                user_id="user-123",
                model="claude-sonnet",
                action="analyze",
                prompt_hash="abc123",
                input_tokens=100,
            )

            assert event.event_type == AuditEventType.AI_REQUEST
            assert event.user_id == "user-123"
            assert event.model == "claude-sonnet"
            assert event.input_tokens == 100
            assert event.content_hash == "abc123"

    def test_log_ai_response(self, mock_env_vars):
        """Test log_ai_response method."""
        from src.security.audit import AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            event = logger.log_ai_response(
                request_id="req-123",
                user_id="user-123",
                model="claude-sonnet",
                output_tokens=200,
                cost_usd=0.05,
                success=True,
            )

            assert event.event_type == AuditEventType.AI_RESPONSE
            assert event.output_tokens == 200
            assert event.cost_usd == 0.05

    def test_log_ai_response_error(self, mock_env_vars):
        """Test log_ai_response with error."""
        from src.security.audit import AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            event = logger.log_ai_response(
                request_id="req-123",
                user_id="user-123",
                model="claude-sonnet",
                output_tokens=0,
                cost_usd=0.0,
                success=False,
                error_message="API error",
            )

            assert event.event_type == AuditEventType.AI_ERROR
            assert event.success is False
            assert event.error_message == "API error"

    def test_log_file_read(self, mock_env_vars):
        """Test log_file_read method."""
        from src.security.audit import AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            event = logger.log_file_read(
                user_id="user-123",
                file_path="/path/to/file.py",
                file_hash="hash123",
                classification="internal",
                was_sanitized=True,
                secrets_redacted=2,
            )

            assert event.event_type == AuditEventType.FILE_READ
            assert event.resource == "/path/to/file.py"
            assert event.metadata["secrets_redacted"] == 2

    def test_log_secret_detected(self, mock_env_vars):
        """Test log_secret_detected method."""
        from src.security.audit import AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            event = logger.log_secret_detected(
                user_id="user-123",
                file_path="/path/to/file.py",
                secret_type="api_key",
                line_number=10,
            )

            assert event.event_type == AuditEventType.SECRET_DETECTED
            assert event.data_classification == "restricted"
            assert event.metadata["secret_type"] == "api_key"
            assert event.metadata["line_number"] == 10

    def test_log_test_execution(self, mock_env_vars):
        """Test log_test_execution method."""
        from src.security.audit import AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            event = logger.log_test_execution(
                user_id="user-123",
                test_id="test-001",
                test_name="Login Test",
                status="passed",
                duration_seconds=5.5,
            )

            assert event.event_type == AuditEventType.TEST_COMPLETED
            assert event.resource == "test-001"
            assert event.success is True
            assert event.metadata["duration_seconds"] == 5.5

    def test_log_test_execution_failed(self, mock_env_vars):
        """Test log_test_execution with failure."""
        from src.security.audit import AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            event = logger.log_test_execution(
                user_id="user-123",
                test_id="test-001",
                test_name="Login Test",
                status="failed",
                duration_seconds=2.0,
                error_message="Element not found",
            )

            assert event.event_type == AuditEventType.TEST_FAILED
            assert event.success is False
            assert event.error_message == "Element not found"

    def test_log_browser_action(self, mock_env_vars):
        """Test log_browser_action method."""
        from src.security.audit import AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            event = logger.log_browser_action(
                user_id="user-123",
                session_id="session-456",
                action_type="click",
                target="#login-button",
                success=True,
                duration_ms=50.0,
            )

            assert event.event_type == AuditEventType.BROWSER_ACTION
            assert event.action == "click"
            assert event.resource == "#login-button"
            assert event.metadata["duration_ms"] == 50.0

    def test_log_integration_event_success(self, mock_env_vars):
        """Test log_integration_event with success."""
        from src.security.audit import AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            event = logger.log_integration_event(
                user_id="user-123",
                integration="github",
                action="connect",
                success=True,
            )

            assert event.event_type == AuditEventType.INTEGRATION_CONNECTED
            assert event.resource == "github"

    def test_log_integration_event_error(self, mock_env_vars):
        """Test log_integration_event with error."""
        from src.security.audit import AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            event = logger.log_integration_event(
                user_id="user-123",
                integration="slack",
                action="send_message",
                success=False,
                error_message="Authentication failed",
            )

            assert event.event_type == AuditEventType.INTEGRATION_ERROR
            assert event.success is False

    def test_query_events(self, mock_env_vars):
        """Test query_events method."""
        from src.security.audit import AuditEventType, AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            # Log some events
            logger.log_ai_request(
                user_id="user-123",
                model="claude-sonnet",
                action="analyze",
                prompt_hash="hash1",
            )
            logger.log_file_read(
                user_id="user-123",
                file_path="/file.py",
            )
            logger.log_file_read(
                user_id="user-456",
                file_path="/other.py",
            )

            # Query all events
            events = logger.query_events()
            assert len(events) == 3

            # Query by user
            events = logger.query_events(user_id="user-123")
            assert len(events) == 2

            # Query by event type
            events = logger.query_events(event_type=AuditEventType.FILE_READ)
            assert len(events) == 2

    def test_query_events_with_limit(self, mock_env_vars):
        """Test query_events with limit."""
        from src.security.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            for i in range(10):
                logger.log_ai_request(
                    user_id=f"user-{i}",
                    model="claude-sonnet",
                    action="analyze",
                    prompt_hash=f"hash{i}",
                )

            events = logger.query_events(limit=5)
            assert len(events) == 5

    def test_generate_compliance_report(self, mock_env_vars):
        """Test generate_compliance_report method."""
        from src.security.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(output_dir=tmpdir)

            # Log various events
            logger.log_ai_request(
                user_id="user-123",
                model="claude-sonnet",
                action="analyze",
                prompt_hash="hash1",
            )
            logger.log_file_read(
                user_id="user-123",
                file_path="/file.py",
            )
            logger.log_secret_detected(
                user_id="user-123",
                file_path="/config.py",
                secret_type="api_key",
                line_number=10,
            )
            logger.log_test_execution(
                user_id="user-123",
                test_id="test-001",
                test_name="Login Test",
                status="passed",
                duration_seconds=5.0,
            )

            # Generate report
            start_date = datetime.now(UTC) - timedelta(hours=1)
            end_date = datetime.now(UTC) + timedelta(hours=1)

            report = logger.generate_compliance_report(start_date, end_date)

            assert report["summary"]["total_events"] == 4
            assert report["summary"]["ai_requests"] == 1
            assert report["summary"]["files_accessed"] == 1
            assert report["summary"]["secrets_detected"] == 1
            assert report["summary"]["tests_run"] == 1
            assert "user-123" in report["by_user"]


class TestHashContent:
    """Tests for hash_content function."""

    def test_hash_content(self, mock_env_vars):
        """Test hash_content function."""
        from src.security.audit import hash_content

        result = hash_content("test content")

        assert isinstance(result, str)
        assert len(result) == 16  # First 16 chars of sha256

    def test_hash_content_deterministic(self, mock_env_vars):
        """Test hash_content is deterministic."""
        from src.security.audit import hash_content

        result1 = hash_content("same content")
        result2 = hash_content("same content")

        assert result1 == result2

    def test_hash_content_different_inputs(self, mock_env_vars):
        """Test hash_content produces different results for different inputs."""
        from src.security.audit import hash_content

        result1 = hash_content("content 1")
        result2 = hash_content("content 2")

        assert result1 != result2


class TestGetAuditLogger:
    """Tests for get_audit_logger function."""

    def test_get_audit_logger_default(self, mock_env_vars):
        """Test get_audit_logger with default settings."""
        import src.security.audit as audit_module
        from src.security.audit import get_audit_logger

        # Reset global logger
        audit_module._audit_logger = None

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                logger = get_audit_logger()

                assert logger is not None
                assert logger.output_dir == Path(tmpdir)

    def test_get_audit_logger_singleton(self, mock_env_vars):
        """Test get_audit_logger returns singleton."""
        import src.security.audit as audit_module
        from src.security.audit import get_audit_logger

        # Reset global logger
        audit_module._audit_logger = None

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                logger1 = get_audit_logger()
                logger2 = get_audit_logger()

                assert logger1 is logger2
