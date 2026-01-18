"""Audit logging for enterprise compliance.

Tracks all AI interactions for:
- SOC2/ISO27001 compliance
- Security investigations
- Cost attribution
- Usage analytics
"""

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import structlog

logger = structlog.get_logger()


class AuditEventType(str, Enum):
    """Types of auditable events."""
    # AI Interactions
    AI_REQUEST = "ai_request"
    AI_RESPONSE = "ai_response"
    AI_ERROR = "ai_error"

    # Data Access
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    CODE_SANITIZED = "code_sanitized"
    SECRET_DETECTED = "secret_detected"

    # Test Execution
    TEST_STARTED = "test_started"
    TEST_COMPLETED = "test_completed"
    TEST_FAILED = "test_failed"
    BROWSER_ACTION = "browser_action"

    # Authentication
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    TOKEN_CREATED = "token_created"
    TOKEN_REVOKED = "token_revoked"

    # System
    CONFIG_CHANGED = "config_changed"
    INTEGRATION_CONNECTED = "integration_connected"
    INTEGRATION_ERROR = "integration_error"


@dataclass
class AuditEvent:
    """An auditable event in the system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    event_type: AuditEventType = AuditEventType.AI_REQUEST

    # Actor information
    user_id: str | None = None
    session_id: str | None = None
    client_ip: str | None = None

    # Event details
    action: str = ""
    resource: str = ""
    resource_type: str = ""

    # Data (sanitized - never raw secrets)
    metadata: dict = field(default_factory=dict)

    # For AI interactions
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    # Hashes for integrity (not the actual content)
    content_hash: str | None = None

    # Outcome
    success: bool = True
    error_message: str | None = None

    # Compliance
    data_classification: str = "internal"  # public, internal, confidential, restricted
    retention_days: int = 90

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = self.to_dict()
        data["event_type"] = data["event_type"].value if isinstance(data["event_type"], AuditEventType) else data["event_type"]
        return json.dumps(data)

    @classmethod
    def from_dict(cls, data: dict) -> "AuditEvent":
        """Create from dictionary."""
        if isinstance(data.get("event_type"), str):
            data["event_type"] = AuditEventType(data["event_type"])
        return cls(**data)


class AuditLogger:
    """
    Enterprise audit logger.

    Features:
    - Immutable audit trail
    - Content hashing (never logs actual secrets)
    - Multiple output destinations
    - Rotation and retention policies
    - Compliance-ready format

    Usage:
        audit = AuditLogger(output_dir="./audit-logs")

        # Log an AI request
        audit.log_ai_request(
            user_id="user-123",
            model="claude-sonnet-4-5",
            prompt_hash=hash(prompt),  # Never log actual prompt
            input_tokens=1500,
        )

        # Log file access
        audit.log_file_read(
            user_id="user-123",
            file_path="/app/src/main.py",
            classification="internal",
        )
    """

    def __init__(
        self,
        output_dir: str = "./audit-logs",
        log_to_stdout: bool = False,
        log_to_file: bool = True,
        max_file_size_mb: int = 100,
        retention_days: int = 90,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_to_stdout = log_to_stdout
        self.log_to_file = log_to_file
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.retention_days = retention_days

        self.log = logger.bind(component="audit")
        self._current_file = None
        self._current_size = 0

    def _get_log_file(self) -> Path:
        """Get current log file, rotating if needed."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        base_path = self.output_dir / f"audit-{today}.jsonl"

        if self._current_file != base_path:
            self._current_file = base_path
            self._current_size = base_path.stat().st_size if base_path.exists() else 0

        # Rotate if too large
        if self._current_size >= self.max_file_size:
            index = 1
            while True:
                rotated = self.output_dir / f"audit-{today}-{index:03d}.jsonl"
                if not rotated.exists():
                    self._current_file = rotated
                    self._current_size = 0
                    break
                index += 1

        return self._current_file

    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        event_json = event.to_json()

        if self.log_to_stdout:
            print(f"AUDIT: {event_json}")

        if self.log_to_file:
            log_file = self._get_log_file()
            with open(log_file, "a") as f:
                f.write(event_json + "\n")
            self._current_size += len(event_json) + 1

    def log_ai_request(
        self,
        user_id: str,
        model: str,
        action: str,
        prompt_hash: str,
        input_tokens: int = 0,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> AuditEvent:
        """Log an AI API request."""
        event = AuditEvent(
            event_type=AuditEventType.AI_REQUEST,
            user_id=user_id,
            session_id=session_id,
            action=action,
            model=model,
            input_tokens=input_tokens,
            content_hash=prompt_hash,
            metadata=metadata or {},
            data_classification="confidential",
        )
        self.log_event(event)
        return event

    def log_ai_response(
        self,
        request_id: str,
        user_id: str,
        model: str,
        output_tokens: int,
        cost_usd: float,
        success: bool = True,
        error_message: str | None = None,
    ) -> AuditEvent:
        """Log an AI API response."""
        event = AuditEvent(
            event_type=AuditEventType.AI_RESPONSE if success else AuditEventType.AI_ERROR,
            user_id=user_id,
            model=model,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            success=success,
            error_message=error_message,
            metadata={"request_id": request_id},
        )
        self.log_event(event)
        return event

    def log_file_read(
        self,
        user_id: str,
        file_path: str,
        file_hash: str | None = None,
        classification: str = "internal",
        was_sanitized: bool = False,
        secrets_redacted: int = 0,
    ) -> AuditEvent:
        """Log a file read operation."""
        event = AuditEvent(
            event_type=AuditEventType.FILE_READ,
            user_id=user_id,
            action="read",
            resource=file_path,
            resource_type="file",
            content_hash=file_hash,
            data_classification=classification,
            metadata={
                "was_sanitized": was_sanitized,
                "secrets_redacted": secrets_redacted,
            },
        )
        self.log_event(event)
        return event

    def log_secret_detected(
        self,
        user_id: str,
        file_path: str,
        secret_type: str,
        line_number: int,
    ) -> AuditEvent:
        """Log that a secret was detected (and redacted)."""
        event = AuditEvent(
            event_type=AuditEventType.SECRET_DETECTED,
            user_id=user_id,
            action="secret_redacted",
            resource=file_path,
            resource_type="file",
            data_classification="restricted",
            metadata={
                "secret_type": secret_type,
                "line_number": line_number,
                "action_taken": "redacted",
            },
        )
        self.log_event(event)
        return event

    def log_test_execution(
        self,
        user_id: str,
        test_id: str,
        test_name: str,
        status: str,
        duration_seconds: float,
        error_message: str | None = None,
    ) -> AuditEvent:
        """Log test execution."""
        event_type = AuditEventType.TEST_COMPLETED if status == "passed" else AuditEventType.TEST_FAILED
        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            action=f"test_{status}",
            resource=test_id,
            resource_type="test",
            success=status == "passed",
            error_message=error_message,
            metadata={
                "test_name": test_name,
                "duration_seconds": duration_seconds,
            },
        )
        self.log_event(event)
        return event

    def log_browser_action(
        self,
        user_id: str,
        session_id: str,
        action_type: str,
        target: str,
        success: bool,
        duration_ms: float,
    ) -> AuditEvent:
        """Log browser automation action."""
        event = AuditEvent(
            event_type=AuditEventType.BROWSER_ACTION,
            user_id=user_id,
            session_id=session_id,
            action=action_type,
            resource=target,
            resource_type="browser_element",
            success=success,
            metadata={"duration_ms": duration_ms},
        )
        self.log_event(event)
        return event

    def log_integration_event(
        self,
        user_id: str,
        integration: str,
        action: str,
        success: bool,
        error_message: str | None = None,
    ) -> AuditEvent:
        """Log integration (GitHub, Slack, etc.) event."""
        event = AuditEvent(
            event_type=AuditEventType.INTEGRATION_CONNECTED if success else AuditEventType.INTEGRATION_ERROR,
            user_id=user_id,
            action=action,
            resource=integration,
            resource_type="integration",
            success=success,
            error_message=error_message,
        )
        self.log_event(event)
        return event

    def query_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        event_type: AuditEventType | None = None,
        user_id: str | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """Query audit events (for compliance reporting)."""
        events = []

        for log_file in sorted(self.output_dir.glob("audit-*.jsonl")):
            with open(log_file) as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        event = AuditEvent.from_dict(data)

                        # Apply filters
                        if event_type and event.event_type != event_type:
                            continue
                        if user_id and event.user_id != user_id:
                            continue

                        event_time = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
                        if start_date and event_time < start_date:
                            continue
                        if end_date and event_time > end_date:
                            continue

                        events.append(event)

                        if len(events) >= limit:
                            return events

                    except Exception as e:
                        self.log.warning("Failed to parse audit event", error=str(e))

        return events

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        """Generate a compliance report for the given period."""
        events = self.query_events(start_date=start_date, end_date=end_date, limit=100000)

        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_events": len(events),
                "ai_requests": 0,
                "files_accessed": 0,
                "secrets_detected": 0,
                "tests_run": 0,
                "total_cost_usd": 0.0,
            },
            "by_user": {},
            "secrets_by_type": {},
            "errors": [],
        }

        for event in events:
            # Update summary
            if event.event_type == AuditEventType.AI_REQUEST:
                report["summary"]["ai_requests"] += 1
            elif event.event_type == AuditEventType.FILE_READ:
                report["summary"]["files_accessed"] += 1
            elif event.event_type == AuditEventType.SECRET_DETECTED:
                report["summary"]["secrets_detected"] += 1
                secret_type = event.metadata.get("secret_type", "unknown")
                report["secrets_by_type"][secret_type] = report["secrets_by_type"].get(secret_type, 0) + 1
            elif event.event_type in (AuditEventType.TEST_COMPLETED, AuditEventType.TEST_FAILED):
                report["summary"]["tests_run"] += 1

            report["summary"]["total_cost_usd"] += event.cost_usd

            # Track by user
            if event.user_id:
                if event.user_id not in report["by_user"]:
                    report["by_user"][event.user_id] = {"events": 0, "cost_usd": 0.0}
                report["by_user"][event.user_id]["events"] += 1
                report["by_user"][event.user_id]["cost_usd"] += event.cost_usd

            # Track errors
            if not event.success and event.error_message:
                report["errors"].append({
                    "timestamp": event.timestamp,
                    "event_type": event.event_type.value,
                    "error": event.error_message,
                })

        return report


def hash_content(content: str) -> str:
    """Create a hash of content for audit logging (never log raw content)."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        output_dir = os.environ.get("AUDIT_LOG_DIR", "./audit-logs")
        _audit_logger = AuditLogger(output_dir=output_dir)
    return _audit_logger
