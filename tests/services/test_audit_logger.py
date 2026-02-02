"""Tests for the Audit Logger service.

This module tests:
- AuditLogger initialization
- Core log() method with various parameters
- Specialized logging methods (log_error, log_chat_message, etc.)
- Error handling and graceful degradation
- Safe serialization utilities
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.audit_logger import (
    AuditAction,
    AuditLogger,
    AuditStatus,
    ResourceType,
    _safe_serialize,
    get_audit_logger,
)


class TestEnums:
    """Tests for audit logger enums."""

    def test_audit_action_values(self):
        """Test AuditAction enum values."""
        assert AuditAction.MEMBER_INVITE.value == "member.invite"
        assert AuditAction.API_KEY_CREATE.value == "api_key.create"
        assert AuditAction.PROJECT_CREATE.value == "project.create"
        assert AuditAction.TEST_RUN.value == "test.run"
        assert AuditAction.AUTH_LOGIN.value == "auth.login"
        assert AuditAction.SYSTEM_ERROR.value == "system.error"
        assert AuditAction.SECURITY_PROMPT_INJECTION_BLOCKED.value == "security.prompt_injection_blocked"

    def test_resource_type_values(self):
        """Test ResourceType enum values."""
        assert ResourceType.ORGANIZATION.value == "organization"
        assert ResourceType.PROJECT.value == "project"
        assert ResourceType.API_KEY.value == "api_key"
        assert ResourceType.TEST.value == "test"
        assert ResourceType.SECURITY.value == "security"

    def test_audit_status_values(self):
        """Test AuditStatus enum values."""
        assert AuditStatus.SUCCESS.value == "success"
        assert AuditStatus.FAILURE.value == "failure"
        assert AuditStatus.PENDING.value == "pending"


class TestAuditLoggerInit:
    """Tests for AuditLogger initialization."""

    def test_init(self, mock_env_vars):
        """Test AuditLogger initialization."""
        logger = AuditLogger()
        assert logger._supabase is None

    @pytest.mark.asyncio
    async def test_get_supabase_lazy_init(self, mock_env_vars):
        """Test lazy Supabase client initialization."""
        mock_client = MagicMock()

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            supabase = await logger._get_supabase()

            assert supabase is mock_client
            assert logger._supabase is mock_client


class TestAuditLoggerLogMethod:
    """Tests for the core log() method."""

    @pytest.mark.asyncio
    async def test_log_success(self, mock_env_vars):
        """Test successful audit logging."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            audit_id = await logger.log(
                action=AuditAction.TEST_RUN,
                resource_type=ResourceType.TEST,
                description="Test executed",
                user_id="user_123",
                organization_id="org_456",
                resource_id="test_789",
            )

        assert audit_id is not None
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "/audit_logs"
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["body"]["action"] == "test.run"

    @pytest.mark.asyncio
    async def test_log_with_string_action(self, mock_env_vars):
        """Test logging with string action instead of enum."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            audit_id = await logger.log(
                action="custom.action",
                resource_type="custom_resource",
                description="Custom action",
            )

        assert audit_id is not None
        call_args = mock_client.request.call_args
        assert call_args[1]["body"]["action"] == "custom.action"
        assert call_args[1]["body"]["resource_type"] == "custom_resource"

    @pytest.mark.asyncio
    async def test_log_with_all_parameters(self, mock_env_vars):
        """Test logging with all parameters."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            await logger.log(
                action=AuditAction.API_KEY_USE,
                resource_type=ResourceType.API_KEY,
                description="API key used",
                user_id="user_123",
                organization_id="org_456",
                resource_id="key_789",
                user_email="test@example.com",
                metadata={"key": "value"},
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0",
                request_id="req_abc",
                status=AuditStatus.SUCCESS,
                error_message=None,
                duration_ms=150,
            )

        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert body["user_id"] == "user_123"
        assert body["organization_id"] == "org_456"
        assert body["user_email"] == "test@example.com"
        assert body["ip_address"] == "192.168.1.1"
        assert body["user_agent"] == "Mozilla/5.0"
        assert body["request_id"] == "req_abc"
        assert body["metadata"]["key"] == "value"
        assert body["metadata"]["duration_ms"] == 150

    @pytest.mark.asyncio
    async def test_log_supabase_not_configured(self, mock_env_vars):
        """Test logging when Supabase is not configured."""
        mock_client = MagicMock()
        mock_client.is_configured = False

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            audit_id = await logger.log(
                action=AuditAction.TEST_RUN,
                resource_type=ResourceType.TEST,
                description="Test",
            )

        assert audit_id is None

    @pytest.mark.asyncio
    async def test_log_request_error(self, mock_env_vars):
        """Test logging when request returns error."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"error": "Database error"})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            audit_id = await logger.log(
                action=AuditAction.TEST_RUN,
                resource_type=ResourceType.TEST,
                description="Test",
            )

        assert audit_id is None

    @pytest.mark.asyncio
    async def test_log_exception_handled(self, mock_env_vars):
        """Test that exceptions in logging don't break the application."""
        with patch("src.services.audit_logger.get_supabase_client", side_effect=Exception("Connection failed")):
            logger = AuditLogger()
            logger._supabase = None  # Force reinitialization

            audit_id = await logger.log(
                action=AuditAction.TEST_RUN,
                resource_type=ResourceType.TEST,
                description="Test",
            )

        assert audit_id is None  # Should not raise


class TestAuditLoggerSpecializedMethods:
    """Tests for specialized logging methods."""

    @pytest.mark.asyncio
    async def test_log_error_with_exception(self, mock_env_vars):
        """Test logging an error with exception object."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()

            try:
                raise ValueError("Test error message")
            except ValueError as e:
                audit_id = await logger.log_error(
                    error=e,
                    context="test_function",
                    user_id="user_123",
                    metadata={"extra": "info"},
                )

        assert audit_id is not None
        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert body["action"] == "system.error"
        assert body["status"] == "failure"
        assert "ValueError" in body["metadata"]["error_type"]
        assert "stack_trace" in body["metadata"]

    @pytest.mark.asyncio
    async def test_log_error_with_string(self, mock_env_vars):
        """Test logging an error with string message."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            audit_id = await logger.log_error(
                error="Something went wrong",
                context="api_endpoint",
            )

        assert audit_id is not None
        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert "Error" in body["metadata"]["error_type"]
        assert "stack_trace" not in body["metadata"]

    @pytest.mark.asyncio
    async def test_log_chat_message(self, mock_env_vars):
        """Test logging a chat message."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            audit_id = await logger.log_chat_message(
                thread_id="thread_123",
                role="user",
                content="Hello, this is a test message",
                user_id="user_123",
                tool_calls=[{"name": "test_tool"}],
            )

        assert audit_id is not None
        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert body["metadata"]["thread_id"] == "thread_123"
        assert body["metadata"]["role"] == "user"
        assert body["metadata"]["has_tool_calls"] is True
        assert body["metadata"]["tool_call_count"] == 1

    @pytest.mark.asyncio
    async def test_log_tool_execution_success(self, mock_env_vars):
        """Test logging successful tool execution."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            audit_id = await logger.log_tool_execution(
                tool_name="browser_click",
                tool_args={"selector": "#submit"},
                result={"success": True},
                success=True,
                duration_ms=250,
                thread_id="thread_123",
            )

        assert audit_id is not None
        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert body["status"] == "success"
        assert body["metadata"]["tool_name"] == "browser_click"
        assert body["metadata"]["duration_ms"] == 250

    @pytest.mark.asyncio
    async def test_log_tool_execution_failure(self, mock_env_vars):
        """Test logging failed tool execution."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            await logger.log_tool_execution(
                tool_name="browser_click",
                tool_args={"selector": "#missing"},
                result=None,
                success=False,
                duration_ms=1000,
                error="Element not found",
            )

        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert body["status"] == "failure"
        assert body["error_message"] == "Element not found"

    @pytest.mark.asyncio
    async def test_log_tool_execution_large_result(self, mock_env_vars):
        """Test logging tool execution with large result is truncated."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        large_result = {"data": "x" * 10000}

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            await logger.log_tool_execution(
                tool_name="fetch_data",
                tool_args={},
                result=large_result,
                success=True,
                duration_ms=500,
            )

        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert "result_preview" in body["metadata"]
        assert "truncated" in body["metadata"]["result_preview"]
        assert body["metadata"]["result_size"] > 5000

    @pytest.mark.asyncio
    async def test_log_auth_event_login(self, mock_env_vars):
        """Test logging login auth event."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            audit_id = await logger.log_auth_event(
                event_type="login",
                user_id="user_123",
                success=True,
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0",
            )

        assert audit_id is not None
        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert body["action"] == "auth.login"
        assert body["status"] == "success"

    @pytest.mark.asyncio
    async def test_log_auth_event_api_key(self, mock_env_vars):
        """Test logging API key auth event."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            await logger.log_auth_event(
                event_type="api_key",
                user_id="user_123",
                success=True,
            )

        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert body["action"] == "api_key.use"

    @pytest.mark.asyncio
    async def test_log_auth_event_failed(self, mock_env_vars):
        """Test logging failed auth event."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            await logger.log_auth_event(
                event_type="login",
                user_id="user_123",
                success=False,
                error="Invalid credentials",
            )

        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert body["status"] == "failure"
        assert body["error_message"] == "Invalid credentials"

    @pytest.mark.asyncio
    async def test_log_browser_operation_success(self, mock_env_vars):
        """Test logging successful browser operation."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            await logger.log_browser_operation(
                operation="screenshot",
                url="https://example.com",
                success=True,
                duration_ms=350,
                result={"width": 1920, "height": 1080},
            )

        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert body["status"] == "success"
        assert body["metadata"]["operation"] == "screenshot"
        assert body["metadata"]["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_log_browser_operation_large_result(self, mock_env_vars):
        """Test browser operation logging truncates large results."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        large_result = {"html": "x" * 5000}

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            await logger.log_browser_operation(
                operation="get_html",
                url="https://example.com",
                success=True,
                duration_ms=200,
                result=large_result,
            )

        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert "result_preview" in body["metadata"]
        assert "truncated" in body["metadata"]["result_preview"]

    @pytest.mark.asyncio
    async def test_log_api_request_success(self, mock_env_vars):
        """Test logging successful API request."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            await logger.log_api_request(
                method="GET",
                path="/api/v1/tests",
                status_code=200,
                duration_ms=45,
                user_id="user_123",
                ip_address="192.168.1.1",
            )

        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert body["action"] == "api_key.use"
        assert body["status"] == "success"
        assert body["metadata"]["method"] == "GET"
        assert body["metadata"]["path"] == "/api/v1/tests"
        assert body["metadata"]["status_code"] == 200

    @pytest.mark.asyncio
    async def test_log_api_request_error(self, mock_env_vars):
        """Test logging failed API request."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            await logger.log_api_request(
                method="POST",
                path="/api/v1/tests",
                status_code=500,
                duration_ms=100,
                error="Internal server error",
            )

        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert body["action"] == "system.error"
        assert body["status"] == "failure"

    @pytest.mark.asyncio
    async def test_log_api_request_client_error(self, mock_env_vars):
        """Test logging client error API request (4xx)."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.request = AsyncMock(return_value={"data": [{"id": "audit_123"}]})

        with patch("src.services.audit_logger.get_supabase_client", return_value=mock_client):
            logger = AuditLogger()
            await logger.log_api_request(
                method="POST",
                path="/api/v1/tests",
                status_code=400,
                duration_ms=20,
            )

        call_args = mock_client.request.call_args
        body = call_args[1]["body"]
        assert body["status"] == "failure"


class TestSafeSerialize:
    """Tests for the _safe_serialize utility function."""

    def test_serialize_dict(self):
        """Test serializing a dictionary."""
        data = {"key": "value", "number": 123}
        result = _safe_serialize(data)
        assert result == json.dumps(data)

    def test_serialize_list(self):
        """Test serializing a list."""
        data = [1, 2, 3, "four"]
        result = _safe_serialize(data)
        assert result == json.dumps(data)

    def test_serialize_datetime(self):
        """Test serializing datetime object."""
        data = {"timestamp": datetime(2024, 1, 15, 10, 30, 0)}
        result = _safe_serialize(data)
        assert "2024" in result

    def test_serialize_non_serializable(self):
        """Test serializing non-JSON-serializable object."""

        class CustomObject:
            pass

        obj = CustomObject()
        result = _safe_serialize(obj)
        assert "CustomObject" in result

    def test_serialize_string(self):
        """Test serializing a plain string."""
        data = "Hello, world!"
        result = _safe_serialize(data)
        assert result == json.dumps(data)


class TestGlobalFunctions:
    """Tests for module-level functions."""

    def test_get_audit_logger_singleton(self, mock_env_vars):
        """Test get_audit_logger returns singleton."""
        import src.services.audit_logger as module
        module._audit_logger = None

        logger1 = get_audit_logger()
        logger2 = get_audit_logger()

        assert logger1 is logger2

        # Clean up
        module._audit_logger = None

    def test_audit_actions_coverage(self):
        """Test that common audit actions are defined."""
        # Team actions
        assert AuditAction.MEMBER_INVITE
        assert AuditAction.MEMBER_REMOVE

        # API Key actions
        assert AuditAction.API_KEY_CREATE
        assert AuditAction.API_KEY_REVOKE

        # Test actions
        assert AuditAction.TEST_GENERATE
        assert AuditAction.TEST_RUN

        # Security actions
        assert AuditAction.AUTH_LOGIN
        assert AuditAction.SECURITY_PROMPT_INJECTION_BLOCKED
