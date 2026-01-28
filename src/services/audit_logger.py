"""
Centralized Audit Logging Service

Provides comprehensive logging for all system operations:
- Chat conversations and errors
- Tool executions and results
- Authentication events
- API requests and responses
- Browser pool operations
- System errors and exceptions

All logs are persisted to Supabase audit_logs table for:
- Debugging and troubleshooting
- Compliance and audit trails
- Performance monitoring
- Error tracking and alerting
"""

import json
import traceback
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger(__name__)


class AuditAction(str, Enum):
    """Audit log action types."""
    # Team actions
    MEMBER_INVITE = "member.invite"
    MEMBER_ACCEPT = "member.accept"
    MEMBER_REMOVE = "member.remove"
    MEMBER_ROLE_CHANGE = "member.role_change"

    # API Key actions
    API_KEY_CREATE = "api_key.create"
    API_KEY_ROTATE = "api_key.rotate"
    API_KEY_REVOKE = "api_key.revoke"
    API_KEY_USE = "api_key.use"

    # Project actions
    PROJECT_CREATE = "project.create"
    PROJECT_UPDATE = "project.update"
    PROJECT_DELETE = "project.delete"
    PROJECT_SETTINGS_CHANGE = "project.settings_change"

    # Test actions
    TEST_GENERATE = "test.generate"
    TEST_APPROVE = "test.approve"
    TEST_REJECT = "test.reject"
    TEST_RUN = "test.run"

    # Webhook actions
    WEBHOOK_RECEIVE = "webhook.receive"
    WEBHOOK_PROCESS = "webhook.process"
    WEBHOOK_FAIL = "webhook.fail"

    # Self-healing actions
    HEALING_APPLY = "healing.apply"
    HEALING_LEARN = "healing.learn"
    HEALING_REJECT = "healing.reject"

    # Security actions
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_MFA_ENABLE = "auth.mfa_enable"
    AUTH_MFA_DISABLE = "auth.mfa_disable"
    AUTH_PASSWORD_CHANGE = "auth.password_change"

    # Organization actions
    ORG_CREATE = "org.create"
    ORG_UPDATE = "org.update"
    ORG_PLAN_CHANGE = "org.plan_change"
    ORG_SETTINGS_CHANGE = "org.settings_change"

    # System actions
    SYSTEM_ERROR = "system.error"
    SYSTEM_CONFIG_CHANGE = "system.config_change"


class ResourceType(str, Enum):
    """Resource types for audit logs."""
    ORGANIZATION = "organization"
    MEMBER = "member"
    PROJECT = "project"
    API_KEY = "api_key"
    TEST = "test"
    EVENT = "event"
    WEBHOOK = "webhook"
    HEALING_PATTERN = "healing_pattern"
    SETTINGS = "settings"
    SYSTEM = "system"


class AuditStatus(str, Enum):
    """Audit log status."""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"


class AuditLogger:
    """
    Centralized audit logging service.

    Usage:
        audit = AuditLogger()

        # Log a successful operation
        await audit.log(
            action=AuditAction.TEST_RUN,
            resource_type=ResourceType.TEST,
            resource_id="test-123",
            description="Test executed successfully",
            user_id="user_abc",
            metadata={"duration_ms": 1500}
        )

        # Log an error
        await audit.log_error(
            error=exception,
            context="chat_streaming",
            user_id="user_abc",
            metadata={"thread_id": "thread-123"}
        )
    """

    def __init__(self):
        self._supabase = None

    async def _get_supabase(self):
        """Get Supabase client lazily."""
        if self._supabase is None:
            self._supabase = get_supabase_client()
        return self._supabase

    async def log(
        self,
        action: AuditAction | str,
        resource_type: ResourceType | str,
        description: str,
        user_id: str = "system",
        organization_id: str | None = None,
        resource_id: str | None = None,
        user_email: str | None = None,
        metadata: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
        status: AuditStatus | str = AuditStatus.SUCCESS,
        error_message: str | None = None,
        duration_ms: int | None = None,
    ) -> str | None:
        """
        Log an audit event to Supabase.

        Returns the audit log ID if successful, None otherwise.
        """
        try:
            supabase = await self._get_supabase()
            if not supabase.is_configured:
                logger.warning("Supabase not configured, skipping audit log")
                return None

            # Convert enums to strings
            action_str = action.value if isinstance(action, AuditAction) else action
            resource_type_str = resource_type.value if isinstance(resource_type, ResourceType) else resource_type
            status_str = status.value if isinstance(status, AuditStatus) else status

            # Build metadata with duration if provided
            full_metadata = metadata or {}
            if duration_ms is not None:
                full_metadata["duration_ms"] = duration_ms

            audit_id = str(uuid4())

            result = await supabase.request(
                "/audit_logs",
                method="POST",
                body={
                    "id": audit_id,
                    "organization_id": organization_id,
                    "user_id": user_id,
                    "user_email": user_email,
                    "action": action_str,
                    "resource_type": resource_type_str,
                    "resource_id": resource_id,
                    "description": description,
                    "metadata": full_metadata,
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "request_id": request_id,
                    "status": status_str,
                    "error_message": error_message,
                },
            )

            if result.get("error"):
                logger.error(
                    "Failed to write audit log",
                    error=result["error"],
                    action=action_str,
                )
                return None

            logger.debug(
                "Audit log created",
                audit_id=audit_id,
                action=action_str,
                resource_type=resource_type_str,
            )
            return audit_id

        except Exception as e:
            # Don't let audit logging failures break the application
            logger.error("Audit logging error", error=str(e), action=str(action))
            return None

    async def log_error(
        self,
        error: Exception | str,
        context: str,
        user_id: str = "system",
        organization_id: str | None = None,
        resource_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        ip_address: str | None = None,
        request_id: str | None = None,
    ) -> str | None:
        """
        Log an error to the audit logs.

        Args:
            error: The exception or error message
            context: Where the error occurred (e.g., "chat_streaming", "tool_executor")
            user_id: User who triggered the operation
            organization_id: Organization context
            resource_id: Related resource ID
            metadata: Additional context
            ip_address: Client IP
            request_id: Request tracking ID
        """
        # Extract error details
        if isinstance(error, Exception):
            error_message = str(error)
            error_type = type(error).__name__
            stack_trace = traceback.format_exc()
        else:
            error_message = str(error)
            error_type = "Error"
            stack_trace = None

        # Build metadata
        full_metadata = metadata or {}
        full_metadata.update({
            "context": context,
            "error_type": error_type,
            "timestamp": datetime.now(UTC).isoformat(),
        })
        if stack_trace:
            full_metadata["stack_trace"] = stack_trace

        return await self.log(
            action=AuditAction.SYSTEM_ERROR,
            resource_type=ResourceType.SYSTEM,
            resource_id=resource_id,
            description=f"Error in {context}: {error_message[:500]}",
            user_id=user_id,
            organization_id=organization_id,
            metadata=full_metadata,
            ip_address=ip_address,
            request_id=request_id,
            status=AuditStatus.FAILURE,
            error_message=error_message[:2000],  # Truncate long errors
        )

    async def log_chat_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        user_id: str = "system",
        organization_id: str | None = None,
        tool_calls: list[dict] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Log a chat message event."""
        full_metadata = metadata or {}
        full_metadata.update({
            "thread_id": thread_id,
            "role": role,
            "content_length": len(content),
            "has_tool_calls": bool(tool_calls),
            "tool_call_count": len(tool_calls) if tool_calls else 0,
        })

        # Don't log full content to audit logs (privacy), just metadata
        return await self.log(
            action=AuditAction.TEST_GENERATE if role == "assistant" else AuditAction.TEST_RUN,
            resource_type=ResourceType.TEST,
            resource_id=thread_id,
            description=f"Chat message ({role}): {content[:100]}...",
            user_id=user_id,
            organization_id=organization_id,
            metadata=full_metadata,
        )

    async def log_tool_execution(
        self,
        tool_name: str,
        tool_args: dict,
        result: dict | str | None,
        success: bool,
        duration_ms: int,
        user_id: str = "system",
        organization_id: str | None = None,
        thread_id: str | None = None,
        error: str | None = None,
    ) -> str | None:
        """Log a tool execution event."""
        metadata = {
            "tool_name": tool_name,
            "tool_args": _safe_serialize(tool_args),
            "thread_id": thread_id,
            "duration_ms": duration_ms,
        }

        if result:
            # Truncate large results
            result_str = _safe_serialize(result)
            if len(result_str) > 5000:
                metadata["result_preview"] = result_str[:5000] + "...(truncated)"
                metadata["result_size"] = len(result_str)
            else:
                metadata["result"] = result_str

        return await self.log(
            action=AuditAction.TEST_RUN,
            resource_type=ResourceType.TEST,
            resource_id=thread_id,
            description=f"Tool executed: {tool_name}",
            user_id=user_id,
            organization_id=organization_id,
            metadata=metadata,
            status=AuditStatus.SUCCESS if success else AuditStatus.FAILURE,
            error_message=error,
            duration_ms=duration_ms,
        )

    async def log_auth_event(
        self,
        event_type: str,
        user_id: str,
        success: bool,
        ip_address: str | None = None,
        user_agent: str | None = None,
        organization_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> str | None:
        """Log an authentication event."""
        action_map = {
            "login": AuditAction.AUTH_LOGIN,
            "logout": AuditAction.AUTH_LOGOUT,
            "api_key": AuditAction.API_KEY_USE,
            "jwt": AuditAction.AUTH_LOGIN,
            "device_auth": AuditAction.AUTH_LOGIN,
        }
        action = action_map.get(event_type, AuditAction.AUTH_LOGIN)

        return await self.log(
            action=action,
            resource_type=ResourceType.MEMBER,
            resource_id=user_id,
            description=f"Auth event: {event_type} ({'success' if success else 'failed'})",
            user_id=user_id,
            organization_id=organization_id,
            metadata=metadata or {},
            ip_address=ip_address,
            user_agent=user_agent,
            status=AuditStatus.SUCCESS if success else AuditStatus.FAILURE,
            error_message=error,
        )

    async def log_browser_operation(
        self,
        operation: str,
        url: str,
        success: bool,
        duration_ms: int,
        user_id: str = "system",
        organization_id: str | None = None,
        result: dict | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Log a browser pool operation."""
        full_metadata = metadata or {}
        full_metadata.update({
            "operation": operation,
            "url": url,
            "duration_ms": duration_ms,
        })

        if result:
            result_str = _safe_serialize(result)
            if len(result_str) > 3000:
                full_metadata["result_preview"] = result_str[:3000] + "...(truncated)"
            else:
                full_metadata["result"] = result_str

        return await self.log(
            action=AuditAction.TEST_RUN,
            resource_type=ResourceType.TEST,
            description=f"Browser {operation}: {url[:100]}",
            user_id=user_id,
            organization_id=organization_id,
            metadata=full_metadata,
            status=AuditStatus.SUCCESS if success else AuditStatus.FAILURE,
            error_message=error,
            duration_ms=duration_ms,
        )

    async def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: int,
        user_id: str = "system",
        organization_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
        error: str | None = None,
    ) -> str | None:
        """Log an API request."""
        success = 200 <= status_code < 400

        return await self.log(
            action=AuditAction.API_KEY_USE if success else AuditAction.SYSTEM_ERROR,
            resource_type=ResourceType.SYSTEM,
            resource_id=request_id,
            description=f"{method} {path} -> {status_code}",
            user_id=user_id,
            organization_id=organization_id,
            metadata={
                "method": method,
                "path": path,
                "status_code": status_code,
            },
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            status=AuditStatus.SUCCESS if success else AuditStatus.FAILURE,
            error_message=error,
            duration_ms=duration_ms,
        )


def _safe_serialize(obj: Any) -> str:
    """Safely serialize an object to JSON string."""
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)


# Global singleton instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
