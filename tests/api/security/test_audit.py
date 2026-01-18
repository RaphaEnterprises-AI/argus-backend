"""Comprehensive tests for security audit logging module (audit.py).

Tests cover:
- AuditEventType enum
- AuditSeverity enum
- AuditEvent model
- SecurityAuditLogger class
- Event logging methods
- Buffer flushing
- Convenience logging methods
- Global audit logger
- Configuration masking
"""

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# =============================================================================
# AuditEventType Enum Tests
# =============================================================================

class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_auth_event_types(self):
        """Test authentication event types exist."""
        from src.api.security.audit import AuditEventType

        auth_events = [
            AuditEventType.AUTH_LOGIN_SUCCESS,
            AuditEventType.AUTH_LOGIN_FAILED,
            AuditEventType.AUTH_LOGOUT,
            AuditEventType.AUTH_TOKEN_ISSUED,
            AuditEventType.AUTH_TOKEN_REVOKED,
            AuditEventType.AUTH_TOKEN_EXPIRED,
            AuditEventType.AUTH_MFA_ENABLED,
            AuditEventType.AUTH_MFA_DISABLED,
            AuditEventType.AUTH_MFA_CHALLENGE,
        ]
        assert all(event for event in auth_events)

    def test_authz_event_types(self):
        """Test authorization event types exist."""
        from src.api.security.audit import AuditEventType

        authz_events = [
            AuditEventType.AUTHZ_PERMISSION_GRANTED,
            AuditEventType.AUTHZ_PERMISSION_DENIED,
            AuditEventType.AUTHZ_ROLE_ASSIGNED,
            AuditEventType.AUTHZ_ROLE_REMOVED,
        ]
        assert all(event for event in authz_events)

    def test_api_key_event_types(self):
        """Test API key event types exist."""
        from src.api.security.audit import AuditEventType

        api_key_events = [
            AuditEventType.API_KEY_CREATED,
            AuditEventType.API_KEY_REVOKED,
            AuditEventType.API_KEY_ROTATED,
            AuditEventType.API_KEY_USED,
        ]
        assert all(event for event in api_key_events)

    def test_data_event_types(self):
        """Test data access event types exist."""
        from src.api.security.audit import AuditEventType

        data_events = [
            AuditEventType.DATA_READ,
            AuditEventType.DATA_CREATED,
            AuditEventType.DATA_UPDATED,
            AuditEventType.DATA_DELETED,
            AuditEventType.DATA_EXPORTED,
        ]
        assert all(event for event in data_events)

    def test_security_event_types(self):
        """Test security event types exist."""
        from src.api.security.audit import AuditEventType

        security_events = [
            AuditEventType.SECURITY_ALERT,
            AuditEventType.SECURITY_RATE_LIMIT,
            AuditEventType.SECURITY_SUSPICIOUS_ACTIVITY,
            AuditEventType.SECURITY_BRUTE_FORCE,
        ]
        assert all(event for event in security_events)

    def test_event_type_values(self):
        """Test event type values follow naming convention."""
        from src.api.security.audit import AuditEventType

        assert AuditEventType.AUTH_LOGIN_SUCCESS.value == "auth.login.success"
        assert AuditEventType.DATA_READ.value == "data.read"
        assert AuditEventType.SECURITY_ALERT.value == "security.alert"


# =============================================================================
# AuditSeverity Enum Tests
# =============================================================================

class TestAuditSeverity:
    """Tests for AuditSeverity enum."""

    def test_severity_levels(self):
        """Test all severity levels exist."""
        from src.api.security.audit import AuditSeverity

        severities = [
            AuditSeverity.DEBUG,
            AuditSeverity.INFO,
            AuditSeverity.WARNING,
            AuditSeverity.ERROR,
            AuditSeverity.CRITICAL,
        ]
        assert all(sev for sev in severities)

    def test_severity_values(self):
        """Test severity values."""
        from src.api.security.audit import AuditSeverity

        assert AuditSeverity.DEBUG.value == "debug"
        assert AuditSeverity.INFO.value == "info"
        assert AuditSeverity.WARNING.value == "warning"
        assert AuditSeverity.ERROR.value == "error"
        assert AuditSeverity.CRITICAL.value == "critical"


# =============================================================================
# AuditEvent Model Tests
# =============================================================================

class TestAuditEventModel:
    """Tests for AuditEvent model."""

    def test_audit_event_creation_minimal(self):
        """Test creating AuditEvent with minimal fields."""
        from src.api.security.audit import AuditEvent, AuditEventType

        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN_SUCCESS,
            action="User logged in",
        )

        assert event.event_type == AuditEventType.AUTH_LOGIN_SUCCESS
        assert event.action == "User logged in"
        assert event.id is not None
        assert event.timestamp is not None

    def test_audit_event_creation_full(self):
        """Test creating AuditEvent with all fields."""
        from src.api.security.audit import AuditEvent, AuditEventType, AuditSeverity

        event = AuditEvent(
            event_type=AuditEventType.DATA_UPDATED,
            severity=AuditSeverity.INFO,
            user_id="user_123",
            organization_id="org_456",
            session_id="session_789",
            api_key_id="key_abc",
            request_id="req_def",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0",
            method="PUT",
            path="/api/v1/users/123",
            resource_type="user",
            resource_id="123",
            action="User profile updated",
            description="Changed email address",
            outcome="success",
            metadata={"field": "email", "old_value": "old@example.com"},
            retention_days=365,
            is_sensitive=False,
        )

        assert event.user_id == "user_123"
        assert event.organization_id == "org_456"
        assert event.resource_type == "user"
        assert event.metadata["field"] == "email"

    def test_audit_event_defaults(self):
        """Test AuditEvent default values."""
        from src.api.security.audit import AuditEvent, AuditEventType, AuditSeverity

        event = AuditEvent(
            event_type=AuditEventType.DATA_READ,
            action="Read data",
        )

        assert event.severity == AuditSeverity.INFO
        assert event.outcome == "success"
        assert event.metadata == {}
        assert event.retention_days == 365
        assert event.is_sensitive is False

    def test_audit_event_id_generation(self):
        """Test that AuditEvent IDs are unique."""
        from src.api.security.audit import AuditEvent, AuditEventType

        events = [
            AuditEvent(event_type=AuditEventType.DATA_READ, action="Read 1"),
            AuditEvent(event_type=AuditEventType.DATA_READ, action="Read 2"),
            AuditEvent(event_type=AuditEventType.DATA_READ, action="Read 3"),
        ]

        ids = [e.id for e in events]
        assert len(set(ids)) == 3  # All unique

    def test_audit_event_timestamp_generation(self):
        """Test that AuditEvent timestamps are generated."""
        from src.api.security.audit import AuditEvent, AuditEventType

        before = datetime.now(timezone.utc)
        event = AuditEvent(event_type=AuditEventType.DATA_READ, action="Read")
        after = datetime.now(timezone.utc)

        assert before <= event.timestamp <= after


# =============================================================================
# SecurityAuditLogger Tests
# =============================================================================

class TestSecurityAuditLogger:
    """Tests for SecurityAuditLogger class."""

    @pytest.fixture
    def logger(self):
        """Create SecurityAuditLogger instance."""
        from src.api.security.audit import SecurityAuditLogger
        return SecurityAuditLogger(persist_to_db=False)

    @pytest.fixture
    def logger_with_db(self):
        """Create SecurityAuditLogger with DB persistence."""
        from src.api.security.audit import SecurityAuditLogger
        return SecurityAuditLogger(persist_to_db=True)

    @pytest.mark.asyncio
    async def test_log_basic(self, logger):
        """Test basic event logging."""
        from src.api.security.audit import AuditEventType

        event = await logger.log(
            event_type=AuditEventType.AUTH_LOGIN_SUCCESS,
            action="User logged in",
            user_id="user_123",
        )

        assert event.event_type == AuditEventType.AUTH_LOGIN_SUCCESS
        assert event.action == "User logged in"
        assert event.user_id == "user_123"

    @pytest.mark.asyncio
    async def test_log_with_all_params(self, logger):
        """Test logging with all parameters."""
        from src.api.security.audit import AuditEventType, AuditSeverity

        event = await logger.log(
            event_type=AuditEventType.DATA_UPDATED,
            action="Updated record",
            user_id="user_123",
            organization_id="org_456",
            resource_type="test",
            resource_id="test_789",
            description="Test description",
            outcome="success",
            severity=AuditSeverity.INFO,
            metadata={"key": "value"},
            request_id="req_abc",
            ip_address="10.0.0.1",
            user_agent="TestAgent/1.0",
            session_id="sess_123",
            api_key_id="key_456",
        )

        assert event.organization_id == "org_456"
        assert event.resource_type == "test"
        assert event.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_log_buffers_events(self, logger_with_db):
        """Test that events are buffered when DB persistence is enabled."""
        from src.api.security.audit import AuditEventType

        await logger_with_db.log(
            event_type=AuditEventType.DATA_READ,
            action="Read data",
        )

        assert len(logger_with_db._event_buffer) == 1

    @pytest.mark.asyncio
    async def test_log_triggers_flush_at_buffer_size(self, logger_with_db):
        """Test that buffer is flushed when it reaches max size."""
        from src.api.security.audit import AuditEventType

        # Set small buffer size for testing
        logger_with_db._buffer_size = 5

        with patch.object(logger_with_db, "_flush_buffer", new_callable=AsyncMock) as mock_flush:
            for i in range(6):
                await logger_with_db.log(
                    event_type=AuditEventType.DATA_READ,
                    action=f"Read {i}",
                )

            # Should have triggered flush
            mock_flush.assert_called()

    @pytest.mark.asyncio
    async def test_flush_buffer_empty(self, logger_with_db):
        """Test flushing empty buffer."""
        # Should not raise error
        await logger_with_db._flush_buffer()

    @pytest.mark.asyncio
    async def test_flush_buffer_to_supabase(self, logger_with_db):
        """Test flushing buffer to Supabase."""
        from src.api.security.audit import AuditEvent, AuditEventType

        # Add events to buffer
        logger_with_db._event_buffer = [
            AuditEvent(event_type=AuditEventType.DATA_READ, action="Read 1"),
            AuditEvent(event_type=AuditEventType.DATA_READ, action="Read 2"),
        ]

        with patch("src.integrations.supabase.get_supabase", new_callable=AsyncMock) as mock_get_supabase:
            mock_supabase = AsyncMock()
            mock_supabase.insert = AsyncMock()
            mock_get_supabase.return_value = mock_supabase

            await logger_with_db._flush_buffer()

            mock_supabase.insert.assert_called_once()
            assert len(logger_with_db._event_buffer) == 0

    @pytest.mark.asyncio
    async def test_flush_buffer_requeues_on_error(self, logger_with_db):
        """Test that events are re-queued on flush error."""
        from src.api.security.audit import AuditEvent, AuditEventType

        # Add events to buffer
        events = [
            AuditEvent(event_type=AuditEventType.DATA_READ, action="Read 1"),
            AuditEvent(event_type=AuditEventType.DATA_READ, action="Read 2"),
        ]
        logger_with_db._event_buffer = events.copy()

        with patch("src.integrations.supabase.get_supabase", new_callable=AsyncMock) as mock_get_supabase:
            mock_supabase = AsyncMock()
            mock_supabase.insert = AsyncMock(side_effect=Exception("DB error"))
            mock_get_supabase.return_value = mock_supabase

            await logger_with_db._flush_buffer()

            # Events should be re-queued
            assert len(logger_with_db._event_buffer) == 2


# =============================================================================
# Convenience Method Tests
# =============================================================================

class TestSecurityAuditLoggerConvenienceMethods:
    """Tests for SecurityAuditLogger convenience methods."""

    @pytest.fixture
    def logger(self):
        """Create SecurityAuditLogger instance."""
        from src.api.security.audit import SecurityAuditLogger
        return SecurityAuditLogger(persist_to_db=False)

    @pytest.mark.asyncio
    async def test_log_auth_success(self, logger):
        """Test log_auth_success convenience method."""
        from src.api.security.audit import AuditEventType

        event = await logger.log_auth_success(
            user_id="user_123",
            auth_method="jwt",
            ip_address="192.168.1.1",
        )

        assert event.event_type == AuditEventType.AUTH_LOGIN_SUCCESS
        assert event.user_id == "user_123"
        assert event.metadata["auth_method"] == "jwt"

    @pytest.mark.asyncio
    async def test_log_auth_failure(self, logger):
        """Test log_auth_failure convenience method."""
        from src.api.security.audit import AuditEventType, AuditSeverity

        event = await logger.log_auth_failure(
            attempted_user="attacker@example.com",
            reason="Invalid credentials",
            ip_address="10.0.0.5",
        )

        assert event.event_type == AuditEventType.AUTH_LOGIN_FAILED
        assert event.outcome == "failure"
        assert event.severity == AuditSeverity.WARNING
        assert event.metadata["reason"] == "Invalid credentials"

    @pytest.mark.asyncio
    async def test_log_permission_denied(self, logger):
        """Test log_permission_denied convenience method."""
        from src.api.security.audit import AuditEventType, AuditSeverity

        event = await logger.log_permission_denied(
            user_id="user_123",
            required_permission="admin:full_access",
            resource_type="settings",
            resource_id="global",
        )

        assert event.event_type == AuditEventType.AUTHZ_PERMISSION_DENIED
        assert event.outcome == "failure"
        assert event.severity == AuditSeverity.WARNING
        assert event.metadata["required_permission"] == "admin:full_access"

    @pytest.mark.asyncio
    async def test_log_data_access_read(self, logger):
        """Test log_data_access for read operations."""
        from src.api.security.audit import AuditEventType

        event = await logger.log_data_access(
            user_id="user_123",
            resource_type="test",
            resource_id="test_456",
            action="read",
        )

        assert event.event_type == AuditEventType.DATA_READ
        assert event.action == "Data read"

    @pytest.mark.asyncio
    async def test_log_data_access_create(self, logger):
        """Test log_data_access for create operations."""
        from src.api.security.audit import AuditEventType

        event = await logger.log_data_access(
            user_id="user_123",
            resource_type="test",
            resource_id="test_456",
            action="create",
        )

        assert event.event_type == AuditEventType.DATA_CREATED

    @pytest.mark.asyncio
    async def test_log_data_access_update(self, logger):
        """Test log_data_access for update operations."""
        from src.api.security.audit import AuditEventType

        event = await logger.log_data_access(
            user_id="user_123",
            resource_type="test",
            resource_id="test_456",
            action="update",
        )

        assert event.event_type == AuditEventType.DATA_UPDATED

    @pytest.mark.asyncio
    async def test_log_data_access_delete(self, logger):
        """Test log_data_access for delete operations."""
        from src.api.security.audit import AuditEventType

        event = await logger.log_data_access(
            user_id="user_123",
            resource_type="test",
            resource_id="test_456",
            action="delete",
        )

        assert event.event_type == AuditEventType.DATA_DELETED

    @pytest.mark.asyncio
    async def test_log_data_access_export(self, logger):
        """Test log_data_access for export operations."""
        from src.api.security.audit import AuditEventType

        event = await logger.log_data_access(
            user_id="user_123",
            resource_type="report",
            resource_id="report_789",
            action="export",
        )

        assert event.event_type == AuditEventType.DATA_EXPORTED

    @pytest.mark.asyncio
    async def test_log_security_alert(self, logger):
        """Test log_security_alert convenience method."""
        from src.api.security.audit import AuditEventType, AuditSeverity

        event = await logger.log_security_alert(
            alert_type="suspicious_login",
            description="Login from unusual location",
            user_id="user_123",
            ip_address="203.0.113.50",
            severity=AuditSeverity.WARNING,
        )

        assert event.event_type == AuditEventType.SECURITY_ALERT
        assert event.metadata["alert_type"] == "suspicious_login"

    @pytest.mark.asyncio
    async def test_log_rate_limit(self, logger):
        """Test log_rate_limit convenience method."""
        from src.api.security.audit import AuditEventType, AuditSeverity

        event = await logger.log_rate_limit(
            ip_address="192.168.1.100",
            endpoint="/api/v1/tests",
            user_id="user_123",
        )

        assert event.event_type == AuditEventType.SECURITY_RATE_LIMIT
        assert event.severity == AuditSeverity.WARNING
        assert event.metadata["endpoint"] == "/api/v1/tests"

    @pytest.mark.asyncio
    async def test_log_api_key_event_created(self, logger):
        """Test log_api_key_event for key creation."""
        from src.api.security.audit import AuditEventType

        event = await logger.log_api_key_event(
            event_type=AuditEventType.API_KEY_CREATED,
            api_key_id="key_123",
            user_id="user_456",
            organization_id="org_789",
            action="API key created",
        )

        assert event.event_type == AuditEventType.API_KEY_CREATED
        assert event.api_key_id == "key_123"
        assert event.resource_type == "api_key"

    @pytest.mark.asyncio
    async def test_log_api_key_event_revoked(self, logger):
        """Test log_api_key_event for key revocation."""
        from src.api.security.audit import AuditEventType

        event = await logger.log_api_key_event(
            event_type=AuditEventType.API_KEY_REVOKED,
            api_key_id="key_123",
            user_id="admin_456",
            organization_id="org_789",
            action="API key revoked",
        )

        assert event.event_type == AuditEventType.API_KEY_REVOKED

    @pytest.mark.asyncio
    async def test_log_config_change(self, logger):
        """Test log_config_change convenience method."""
        from src.api.security.audit import AuditEventType

        event = await logger.log_config_change(
            user_id="admin_123",
            config_key="max_test_runs",
            old_value=10,
            new_value=20,
            organization_id="org_456",
        )

        assert event.event_type == AuditEventType.CONFIG_CHANGED
        assert "max_test_runs" in event.action
        assert event.resource_type == "config"

    @pytest.mark.asyncio
    async def test_log_config_change_masks_secrets(self, logger):
        """Test that sensitive config values are masked."""
        from src.api.security.audit import AuditEventType

        event = await logger.log_config_change(
            user_id="admin_123",
            config_key="api_secret_key",
            old_value="old_secret_12345",
            new_value="new_secret_67890",
        )

        assert event.metadata["old_value"] == "***REDACTED***"
        assert event.metadata["new_value"] == "***REDACTED***"

    @pytest.mark.asyncio
    async def test_log_config_change_masks_password(self, logger):
        """Test that password config values are masked."""
        event = await logger.log_config_change(
            user_id="admin_123",
            config_key="database_password",
            old_value="old_password",
            new_value="new_password",
        )

        assert event.metadata["old_value"] == "***REDACTED***"
        assert event.metadata["new_value"] == "***REDACTED***"

    @pytest.mark.asyncio
    async def test_log_config_change_masks_token(self, logger):
        """Test that token config values are masked."""
        event = await logger.log_config_change(
            user_id="admin_123",
            config_key="auth_token",
            old_value="old_token",
            new_value="new_token",
        )

        assert event.metadata["old_value"] == "***REDACTED***"
        assert event.metadata["new_value"] == "***REDACTED***"


# =============================================================================
# Global Audit Logger Tests
# =============================================================================

class TestGetAuditLogger:
    """Tests for get_audit_logger function."""

    def test_get_audit_logger_returns_instance(self):
        """Test that get_audit_logger returns a SecurityAuditLogger."""
        from src.api.security.audit import get_audit_logger, SecurityAuditLogger

        logger = get_audit_logger()
        assert isinstance(logger, SecurityAuditLogger)

    def test_get_audit_logger_singleton(self):
        """Test that get_audit_logger returns same instance."""
        from src.api.security.audit import get_audit_logger

        logger1 = get_audit_logger()
        logger2 = get_audit_logger()

        assert logger1 is logger2


class TestAuditLogFunction:
    """Tests for audit_log convenience function."""

    @pytest.mark.asyncio
    async def test_audit_log_function(self):
        """Test audit_log convenience function."""
        from src.api.security.audit import audit_log, AuditEventType

        # Reset global logger to ensure clean state
        import src.api.security.audit as audit_module
        audit_module._audit_logger = None

        with patch.object(
            audit_module.SecurityAuditLogger,
            "log",
            new_callable=AsyncMock
        ) as mock_log:
            mock_event = MagicMock()
            mock_log.return_value = mock_event

            result = await audit_log(
                event_type=AuditEventType.DATA_READ,
                action="Test read",
                user_id="user_123",
            )

            # Reset for other tests
            audit_module._audit_logger = None


# =============================================================================
# Integration Tests
# =============================================================================

class TestAuditLoggingIntegration:
    """Integration tests for audit logging."""

    @pytest.mark.asyncio
    async def test_full_audit_flow(self):
        """Test complete audit logging flow."""
        from src.api.security.audit import (
            SecurityAuditLogger,
            AuditEventType,
            AuditSeverity,
        )

        logger = SecurityAuditLogger(persist_to_db=False)

        # Simulate user authentication
        login_event = await logger.log_auth_success(
            user_id="user_123",
            auth_method="jwt",
            ip_address="192.168.1.100",
        )
        assert login_event.event_type == AuditEventType.AUTH_LOGIN_SUCCESS

        # Simulate data access
        read_event = await logger.log_data_access(
            user_id="user_123",
            resource_type="test",
            resource_id="test_456",
            action="read",
        )
        assert read_event.event_type == AuditEventType.DATA_READ

        # Simulate permission denied
        denied_event = await logger.log_permission_denied(
            user_id="user_123",
            required_permission="admin:full_access",
            resource_type="settings",
            resource_id="global",
        )
        assert denied_event.outcome == "failure"

    @pytest.mark.asyncio
    async def test_audit_severity_logging(self):
        """Test that severity affects logging."""
        from src.api.security.audit import (
            SecurityAuditLogger,
            AuditEventType,
            AuditSeverity,
        )

        logger = SecurityAuditLogger(persist_to_db=False)

        # Log at different severity levels
        with patch("src.api.security.audit.logger") as mock_logger:
            mock_logger.debug = MagicMock()
            mock_logger.info = MagicMock()
            mock_logger.warning = MagicMock()
            mock_logger.error = MagicMock()
            mock_logger.critical = MagicMock()

            await logger.log(
                event_type=AuditEventType.DATA_READ,
                action="Debug action",
                severity=AuditSeverity.DEBUG,
            )
            mock_logger.debug.assert_called()

            await logger.log(
                event_type=AuditEventType.DATA_READ,
                action="Info action",
                severity=AuditSeverity.INFO,
            )
            mock_logger.info.assert_called()

            await logger.log(
                event_type=AuditEventType.SECURITY_ALERT,
                action="Warning action",
                severity=AuditSeverity.WARNING,
            )
            mock_logger.warning.assert_called()

            await logger.log(
                event_type=AuditEventType.SYSTEM_ERROR,
                action="Error action",
                severity=AuditSeverity.ERROR,
            )
            mock_logger.error.assert_called()

            await logger.log(
                event_type=AuditEventType.SECURITY_BRUTE_FORCE,
                action="Critical action",
                severity=AuditSeverity.CRITICAL,
            )
            mock_logger.critical.assert_called()

    @pytest.mark.asyncio
    async def test_audit_metadata_preservation(self):
        """Test that metadata is preserved through logging."""
        from src.api.security.audit import SecurityAuditLogger, AuditEventType

        logger = SecurityAuditLogger(persist_to_db=False)

        event = await logger.log(
            event_type=AuditEventType.DATA_UPDATED,
            action="Update with metadata",
            metadata={
                "field": "email",
                "old_value": "old@example.com",
                "new_value": "new@example.com",
                "reason": "User requested change",
            },
        )

        assert event.metadata["field"] == "email"
        assert event.metadata["old_value"] == "old@example.com"
        assert event.metadata["reason"] == "User requested change"

    @pytest.mark.asyncio
    async def test_soc2_compliance_retention(self):
        """Test SOC2 compliance retention period."""
        from src.api.security.audit import AuditEvent, AuditEventType

        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN_SUCCESS,
            action="User logged in",
        )

        # SOC2 requires minimum 1 year retention
        assert event.retention_days >= 365
