"""Security audit logging for SOC2 compliance.

Provides comprehensive audit logging for:
- Authentication events
- Authorization decisions
- Data access
- Configuration changes
- Security incidents
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from enum import Enum

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


# =============================================================================
# Audit Event Types
# =============================================================================

class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILED = "auth.login.failed"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_ISSUED = "auth.token.issued"
    AUTH_TOKEN_REVOKED = "auth.token.revoked"
    AUTH_TOKEN_EXPIRED = "auth.token.expired"
    AUTH_MFA_ENABLED = "auth.mfa.enabled"
    AUTH_MFA_DISABLED = "auth.mfa.disabled"
    AUTH_MFA_CHALLENGE = "auth.mfa.challenge"

    # Authorization events
    AUTHZ_PERMISSION_GRANTED = "authz.permission.granted"
    AUTHZ_PERMISSION_DENIED = "authz.permission.denied"
    AUTHZ_ROLE_ASSIGNED = "authz.role.assigned"
    AUTHZ_ROLE_REMOVED = "authz.role.removed"

    # API Key events
    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"
    API_KEY_ROTATED = "api_key.rotated"
    API_KEY_USED = "api_key.used"

    # Data access events
    DATA_READ = "data.read"
    DATA_CREATED = "data.created"
    DATA_UPDATED = "data.updated"
    DATA_DELETED = "data.deleted"
    DATA_EXPORTED = "data.exported"

    # Configuration events
    CONFIG_CHANGED = "config.changed"
    CONFIG_SECURITY_CHANGED = "config.security.changed"

    # User management events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_SUSPENDED = "user.suspended"
    USER_ACTIVATED = "user.activated"

    # Organization events
    ORG_CREATED = "org.created"
    ORG_UPDATED = "org.updated"
    ORG_DELETED = "org.deleted"
    ORG_MEMBER_ADDED = "org.member.added"
    ORG_MEMBER_REMOVED = "org.member.removed"

    # Security events
    SECURITY_ALERT = "security.alert"
    SECURITY_RATE_LIMIT = "security.rate_limit"
    SECURITY_SUSPICIOUS_ACTIVITY = "security.suspicious_activity"
    SECURITY_BRUTE_FORCE = "security.brute_force"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Audit Event Model
# =============================================================================

class AuditEvent(BaseModel):
    """Audit event record."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType
    severity: AuditSeverity = AuditSeverity.INFO

    # Actor information
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    session_id: Optional[str] = None
    api_key_id: Optional[str] = None

    # Request context
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    method: Optional[str] = None
    path: Optional[str] = None

    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None

    # Event details
    action: str
    description: Optional[str] = None
    outcome: str = "success"  # success, failure, error
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # For compliance
    retention_days: int = 365  # SOC2 requires minimum 1 year
    is_sensitive: bool = False

    class Config:
        use_enum_values = True


# =============================================================================
# Security Audit Logger
# =============================================================================

class SecurityAuditLogger:
    """Comprehensive security audit logger for SOC2 compliance."""

    def __init__(self, persist_to_db: bool = True):
        self.persist_to_db = persist_to_db
        self._event_buffer: List[AuditEvent] = []
        self._buffer_size = 100
        self._flush_interval = 5  # seconds

    async def log(
        self,
        event_type: AuditEventType,
        action: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        description: Optional[str] = None,
        outcome: str = "success",
        severity: AuditSeverity = AuditSeverity.INFO,
        metadata: Dict[str, Any] = None,
        request_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
    ) -> AuditEvent:
        """Log an audit event."""
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            organization_id=organization_id,
            session_id=session_id,
            api_key_id=api_key_id,
            request_id=request_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            description=description,
            outcome=outcome,
            metadata=metadata or {},
        )

        # Log to structured logger
        log_method = getattr(logger, severity.value, logger.info)
        log_method(
            f"Audit: {action}",
            event_id=event.id,
            event_type=event_type,
            user_id=user_id,
            organization_id=organization_id,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            ip_address=ip_address,
        )

        # Buffer for batch persistence
        if self.persist_to_db:
            self._event_buffer.append(event)
            if len(self._event_buffer) >= self._buffer_size:
                asyncio.create_task(self._flush_buffer())

        return event

    async def _flush_buffer(self) -> None:
        """Flush event buffer to database."""
        if not self._event_buffer:
            return

        events_to_persist = self._event_buffer.copy()
        self._event_buffer.clear()

        try:
            from src.integrations.supabase import get_supabase

            supabase = await get_supabase()
            if not supabase:
                return

            # Batch insert
            records = [
                {
                    "id": event.id,
                    "event_type": event.event_type,
                    "severity": event.severity,
                    "user_id": event.user_id,
                    "organization_id": event.organization_id,
                    "session_id": event.session_id,
                    "api_key_id": event.api_key_id,
                    "request_id": event.request_id,
                    "ip_address": event.ip_address,
                    "user_agent": event.user_agent,
                    "resource_type": event.resource_type,
                    "resource_id": event.resource_id,
                    "action": event.action,
                    "description": event.description,
                    "outcome": event.outcome,
                    "metadata": event.metadata,
                    "created_at": event.timestamp.isoformat(),
                }
                for event in events_to_persist
            ]

            await supabase.insert("security_audit_logs", records)

        except Exception as e:
            logger.error("Failed to persist audit events", error=str(e))
            # Re-add to buffer for retry
            self._event_buffer.extend(events_to_persist)

    # ==========================================================================
    # Convenience Methods
    # ==========================================================================

    async def log_auth_success(
        self,
        user_id: str,
        auth_method: str,
        ip_address: str = None,
        **kwargs,
    ) -> AuditEvent:
        """Log successful authentication."""
        return await self.log(
            event_type=AuditEventType.AUTH_LOGIN_SUCCESS,
            action="User authenticated",
            user_id=user_id,
            ip_address=ip_address,
            metadata={"auth_method": auth_method, **kwargs},
        )

    async def log_auth_failure(
        self,
        attempted_user: str,
        reason: str,
        ip_address: str = None,
        **kwargs,
    ) -> AuditEvent:
        """Log failed authentication."""
        return await self.log(
            event_type=AuditEventType.AUTH_LOGIN_FAILED,
            action="Authentication failed",
            description=reason,
            outcome="failure",
            severity=AuditSeverity.WARNING,
            ip_address=ip_address,
            metadata={"attempted_user": attempted_user, "reason": reason, **kwargs},
        )

    async def log_permission_denied(
        self,
        user_id: str,
        required_permission: str,
        resource_type: str = None,
        resource_id: str = None,
        **kwargs,
    ) -> AuditEvent:
        """Log permission denied."""
        return await self.log(
            event_type=AuditEventType.AUTHZ_PERMISSION_DENIED,
            action="Permission denied",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome="failure",
            severity=AuditSeverity.WARNING,
            metadata={"required_permission": required_permission, **kwargs},
        )

    async def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        **kwargs,
    ) -> AuditEvent:
        """Log data access."""
        event_type_map = {
            "read": AuditEventType.DATA_READ,
            "create": AuditEventType.DATA_CREATED,
            "update": AuditEventType.DATA_UPDATED,
            "delete": AuditEventType.DATA_DELETED,
            "export": AuditEventType.DATA_EXPORTED,
        }

        return await self.log(
            event_type=event_type_map.get(action, AuditEventType.DATA_READ),
            action=f"Data {action}",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            **kwargs,
        )

    async def log_security_alert(
        self,
        alert_type: str,
        description: str,
        user_id: str = None,
        ip_address: str = None,
        severity: AuditSeverity = AuditSeverity.WARNING,
        **kwargs,
    ) -> AuditEvent:
        """Log security alert."""
        return await self.log(
            event_type=AuditEventType.SECURITY_ALERT,
            action=f"Security alert: {alert_type}",
            description=description,
            user_id=user_id,
            ip_address=ip_address,
            severity=severity,
            metadata={"alert_type": alert_type, **kwargs},
        )

    async def log_rate_limit(
        self,
        ip_address: str,
        endpoint: str,
        user_id: str = None,
        **kwargs,
    ) -> AuditEvent:
        """Log rate limit event."""
        return await self.log(
            event_type=AuditEventType.SECURITY_RATE_LIMIT,
            action="Rate limit exceeded",
            user_id=user_id,
            ip_address=ip_address,
            severity=AuditSeverity.WARNING,
            metadata={"endpoint": endpoint, **kwargs},
        )

    async def log_api_key_event(
        self,
        event_type: AuditEventType,
        api_key_id: str,
        user_id: str,
        organization_id: str,
        action: str,
        **kwargs,
    ) -> AuditEvent:
        """Log API key event."""
        return await self.log(
            event_type=event_type,
            action=action,
            user_id=user_id,
            organization_id=organization_id,
            api_key_id=api_key_id,
            resource_type="api_key",
            resource_id=api_key_id,
            **kwargs,
        )

    async def log_config_change(
        self,
        user_id: str,
        config_key: str,
        old_value: Any,
        new_value: Any,
        organization_id: str = None,
        **kwargs,
    ) -> AuditEvent:
        """Log configuration change."""
        # Mask sensitive values
        if any(s in config_key.lower() for s in ["secret", "key", "password", "token"]):
            old_value = "***REDACTED***"
            new_value = "***REDACTED***"

        return await self.log(
            event_type=AuditEventType.CONFIG_CHANGED,
            action=f"Configuration changed: {config_key}",
            user_id=user_id,
            organization_id=organization_id,
            resource_type="config",
            resource_id=config_key,
            metadata={"old_value": old_value, "new_value": new_value, **kwargs},
        )


# =============================================================================
# Global Audit Logger Instance
# =============================================================================

_audit_logger: Optional[SecurityAuditLogger] = None


def get_audit_logger() -> SecurityAuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = SecurityAuditLogger()
    return _audit_logger


async def audit_log(
    event_type: AuditEventType,
    action: str,
    **kwargs,
) -> AuditEvent:
    """Convenience function to log an audit event."""
    return await get_audit_logger().log(event_type, action, **kwargs)
