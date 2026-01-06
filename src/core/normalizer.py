"""
Event Normalizer - Unified Event Format

Converts events from various sources (Sentry, GitHub, Datadog, etc.)
into a standardized internal format for correlation and analysis.
"""

import hashlib
import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


class EventSource(str, Enum):
    """Supported event sources."""
    SENTRY = "sentry"
    DATADOG = "datadog"
    FULLSTORY = "fullstory"
    LOGROCKET = "logrocket"
    NEWRELIC = "newrelic"
    BUGSNAG = "bugsnag"
    ROLLBAR = "rollbar"
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    CIRCLECI = "circleci"
    JENKINS = "jenkins"
    COVERAGE = "coverage"


class EventType(str, Enum):
    """Types of events."""
    ERROR = "error"
    EXCEPTION = "exception"
    PERFORMANCE = "performance"
    SESSION = "session"
    RAGE_CLICK = "rage_click"
    DEAD_CLICK = "dead_click"
    CI_RUN = "ci_run"
    COVERAGE_REPORT = "coverage_report"
    CODE_CHANGE = "code_change"


class Severity(str, Enum):
    """Event severity levels."""
    FATAL = "fatal"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class StackFrame:
    """A single stack frame."""
    filename: Optional[str] = None
    function: Optional[str] = None
    lineno: Optional[int] = None
    colno: Optional[int] = None
    context: Optional[str] = None  # Code snippet around the line
    in_app: bool = True  # Is this app code vs library code?

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "function": self.function,
            "lineno": self.lineno,
            "colno": self.colno,
            "context": self.context,
            "in_app": self.in_app,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StackFrame":
        return cls(
            filename=data.get("filename"),
            function=data.get("function"),
            lineno=data.get("lineno"),
            colno=data.get("colno"),
            context=data.get("context"),
            in_app=data.get("in_app", True),
        )


@dataclass
class NormalizedEvent:
    """
    Unified event format for all sources.

    This is the internal format used for correlation, analysis, and storage.
    """
    # Identity
    id: str
    source: EventSource
    external_id: str
    external_url: Optional[str] = None

    # Classification
    event_type: EventType = EventType.ERROR
    severity: Severity = Severity.ERROR

    # Content
    title: str = ""
    message: Optional[str] = None
    error_type: Optional[str] = None  # e.g., "TypeError", "ValueError"

    # Stack trace (parsed)
    stack_frames: list[StackFrame] = field(default_factory=list)
    raw_stack_trace: Optional[str] = None

    # Location
    file_path: Optional[str] = None  # Primary file where error occurred
    function_name: Optional[str] = None  # Primary function
    line_number: Optional[int] = None
    component: Optional[str] = None  # UI component (React/Vue/Angular)

    # Context
    url: Optional[str] = None  # Page URL where error occurred
    browser: Optional[str] = None
    os: Optional[str] = None
    device_type: Optional[str] = None  # desktop, mobile, tablet
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Metrics
    occurrence_count: int = 1
    affected_users: int = 1

    # Timing
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Grouping
    fingerprint: str = ""  # For deduplication
    tags: list[str] = field(default_factory=list)

    # Raw data
    raw_payload: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source.value,
            "external_id": self.external_id,
            "external_url": self.external_url,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "error_type": self.error_type,
            "stack_frames": [f.to_dict() for f in self.stack_frames],
            "raw_stack_trace": self.raw_stack_trace,
            "file_path": self.file_path,
            "function_name": self.function_name,
            "line_number": self.line_number,
            "component": self.component,
            "url": self.url,
            "browser": self.browser,
            "os": self.os,
            "device_type": self.device_type,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "occurrence_count": self.occurrence_count,
            "affected_users": self.affected_users,
            "first_seen_at": self.first_seen_at.isoformat() if self.first_seen_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "created_at": self.created_at.isoformat(),
            "fingerprint": self.fingerprint,
            "tags": self.tags,
            "raw_payload": self.raw_payload,
            "metadata": self.metadata,
        }


class EventNormalizer:
    """
    Normalizes events from various sources into a unified format.

    This is the first step in the Quality Intelligence pipeline:
    Raw Webhook → Normalized Event → Correlation → Risk Scoring
    """

    def __init__(self):
        self.log = logger.bind(component="event_normalizer")

    def normalize(self, source: EventSource, raw_data: dict) -> NormalizedEvent:
        """
        Normalize a raw event from any source.

        Args:
            source: The event source (sentry, datadog, etc.)
            raw_data: The raw webhook payload

        Returns:
            NormalizedEvent in unified format
        """
        normalizers = {
            EventSource.SENTRY: self._normalize_sentry,
            EventSource.DATADOG: self._normalize_datadog,
            EventSource.FULLSTORY: self._normalize_fullstory,
            EventSource.LOGROCKET: self._normalize_logrocket,
            EventSource.NEWRELIC: self._normalize_newrelic,
            EventSource.BUGSNAG: self._normalize_bugsnag,
            EventSource.ROLLBAR: self._normalize_rollbar,
            EventSource.GITHUB_ACTIONS: self._normalize_github_actions,
        }

        normalizer = normalizers.get(source)
        if not normalizer:
            self.log.warning("Unknown source, using generic normalizer", source=source)
            return self._normalize_generic(source, raw_data)

        try:
            event = normalizer(raw_data)
            self.log.debug("Event normalized", source=source, fingerprint=event.fingerprint)
            return event
        except Exception as e:
            self.log.error("Normalization failed", source=source, error=str(e))
            return self._normalize_generic(source, raw_data)

    def _normalize_sentry(self, data: dict) -> NormalizedEvent:
        """Normalize Sentry webhook payload."""
        import uuid

        issue_data = data.get("data", {}).get("issue", {})
        event_data = data.get("data", {}).get("event", {})

        # Extract exception info
        exception_values = event_data.get("event", {}).get("exception", {}).get("values", [])
        exception = exception_values[0] if exception_values else {}

        # Parse stack trace
        stack_frames = []
        raw_frames = exception.get("stacktrace", {}).get("frames", [])
        for frame in reversed(raw_frames):  # Sentry frames are bottom-up
            stack_frames.append(StackFrame(
                filename=frame.get("filename"),
                function=frame.get("function"),
                lineno=frame.get("lineno"),
                colno=frame.get("colno"),
                context=frame.get("context_line"),
                in_app=frame.get("in_app", True),
            ))

        # Build raw stack trace string
        raw_stack = "\n".join(
            f"  at {f.function or 'anonymous'} ({f.filename}:{f.lineno}:{f.colno})"
            for f in stack_frames if f.filename
        )

        # Extract primary location
        primary_frame = next((f for f in stack_frames if f.in_app), stack_frames[0] if stack_frames else None)

        # Extract context info
        contexts = event_data.get("event", {}).get("contexts", {})
        browser_info = contexts.get("browser", {})
        os_info = contexts.get("os", {})
        device_info = contexts.get("device", {})

        # Determine device type
        device_type = None
        if device_info.get("family"):
            family = device_info["family"].lower()
            if "iphone" in family or "android" in family:
                device_type = "mobile"
            elif "ipad" in family or "tablet" in family:
                device_type = "tablet"
            else:
                device_type = "desktop"

        title = issue_data.get("title") or exception.get("type") or "Unknown Error"
        message = issue_data.get("message") or exception.get("value") or ""
        error_type = exception.get("type")

        # Extract component from stack
        component = self._extract_component(raw_stack)

        # Generate fingerprint
        page_url = event_data.get("event", {}).get("request", {}).get("url")
        fingerprint = self._generate_fingerprint(error_type or "Error", message, component, page_url)

        return NormalizedEvent(
            id=str(uuid.uuid4()),
            source=EventSource.SENTRY,
            external_id=issue_data.get("id") or event_data.get("event_id") or str(uuid.uuid4()),
            external_url=issue_data.get("url") or event_data.get("issue_url"),
            event_type=EventType.ERROR,
            severity=self._parse_severity(issue_data.get("level", "error")),
            title=title,
            message=message,
            error_type=error_type,
            stack_frames=stack_frames,
            raw_stack_trace=raw_stack if raw_stack else None,
            file_path=primary_frame.filename if primary_frame else None,
            function_name=primary_frame.function if primary_frame else None,
            line_number=primary_frame.lineno if primary_frame else None,
            component=component,
            url=page_url,
            browser=f"{browser_info.get('name', '')} {browser_info.get('version', '')}".strip() or None,
            os=f"{os_info.get('name', '')} {os_info.get('version', '')}".strip() or None,
            device_type=device_type,
            occurrence_count=int(issue_data.get("count", "1")),
            affected_users=issue_data.get("userCount", 1),
            first_seen_at=self._parse_datetime(issue_data.get("firstSeen")),
            last_seen_at=self._parse_datetime(issue_data.get("lastSeen")),
            fingerprint=fingerprint,
            tags=[f"{t['key']}:{t['value']}" for t in issue_data.get("tags", [])],
            raw_payload=data,
            metadata={
                "sentry_project": event_data.get("project_name") or issue_data.get("project"),
                "sentry_platform": event_data.get("platform") or issue_data.get("platform"),
                "sentry_short_id": issue_data.get("shortId"),
            },
        )

    def _normalize_datadog(self, data: dict) -> NormalizedEvent:
        """Normalize Datadog webhook payload."""
        import uuid

        # Datadog can send single event or array
        events = data if isinstance(data, list) else [data]
        event = events[0]  # Take first event

        error = event.get("error", {})
        view = event.get("view", {})

        # Parse stack if available
        stack_frames = []
        if error.get("stack"):
            stack_frames = self._parse_generic_stack(error["stack"])

        title = event.get("title", "Datadog Event")
        message = error.get("message") or event.get("message") or ""
        error_type = error.get("type")

        page_url = view.get("url") or error.get("source") or event.get("url")
        component = self._extract_component(error.get("stack"))
        fingerprint = self._generate_fingerprint(error_type or "Error", message, component, page_url)

        # Determine event type
        event_type = EventType.ERROR
        if error_type:
            event_type = EventType.EXCEPTION
        elif "performance" in event.get("event_type", "") or "apm" in event.get("source_type_name", ""):
            event_type = EventType.PERFORMANCE

        return NormalizedEvent(
            id=str(uuid.uuid4()),
            source=EventSource.DATADOG,
            external_id=event.get("id") or event.get("aggregation_key") or str(uuid.uuid4()),
            external_url=event.get("url"),
            event_type=event_type,
            severity=self._parse_severity(event.get("alert_type", "error")),
            title=title,
            message=message,
            error_type=error_type,
            stack_frames=stack_frames,
            raw_stack_trace=error.get("stack"),
            file_path=stack_frames[0].filename if stack_frames else None,
            function_name=stack_frames[0].function if stack_frames else None,
            line_number=stack_frames[0].lineno if stack_frames else None,
            component=component,
            url=page_url,
            fingerprint=fingerprint,
            tags=event.get("tags", []),
            raw_payload=data,
            metadata={
                "datadog_host": event.get("host"),
                "datadog_source": event.get("source_type_name"),
                "datadog_priority": event.get("priority"),
            },
        )

    def _normalize_fullstory(self, data: dict) -> NormalizedEvent:
        """Normalize FullStory webhook payload."""
        import uuid

        event_type_raw = data.get("type", "error")
        event_type = EventType.ERROR
        if "rage" in event_type_raw.lower():
            event_type = EventType.RAGE_CLICK
        elif "dead" in event_type_raw.lower():
            event_type = EventType.DEAD_CLICK

        session_url = data.get("session", {}).get("url") or data.get("sessionUrl")
        page_url = data.get("page", {}).get("url") or data.get("pageUrl")
        element_selector = data.get("element", {}).get("selector") or data.get("selector")

        title = data.get("title") or f"FullStory {event_type_raw.replace('_', ' ').title()}"
        message = data.get("message") or element_selector or ""

        fingerprint = self._generate_fingerprint(event_type_raw, message, element_selector, page_url)

        return NormalizedEvent(
            id=str(uuid.uuid4()),
            source=EventSource.FULLSTORY,
            external_id=data.get("id") or str(uuid.uuid4()),
            external_url=session_url,
            event_type=event_type,
            severity=Severity.WARNING if event_type in (EventType.RAGE_CLICK, EventType.DEAD_CLICK) else Severity.ERROR,
            title=title,
            message=message,
            component=element_selector,
            url=page_url,
            occurrence_count=data.get("count", 1),
            affected_users=data.get("userCount", 1),
            fingerprint=fingerprint,
            raw_payload=data,
            metadata={
                "fullstory_session_url": session_url,
                "element_selector": element_selector,
            },
        )

    def _normalize_logrocket(self, data: dict) -> NormalizedEvent:
        """Normalize LogRocket webhook payload."""
        import uuid

        error = data.get("error", {})
        session = data.get("session", {})

        title = error.get("type") or error.get("name") or data.get("title") or "LogRocket Error"
        message = error.get("message") or data.get("message") or ""
        stack_trace = error.get("stack") or error.get("stackTrace")

        stack_frames = self._parse_generic_stack(stack_trace) if stack_trace else []
        component = self._extract_component(stack_trace)
        page_url = session.get("url") or data.get("url")

        fingerprint = self._generate_fingerprint(error.get("type", "Error"), message, component, page_url)

        return NormalizedEvent(
            id=str(uuid.uuid4()),
            source=EventSource.LOGROCKET,
            external_id=data.get("id") or session.get("id") or str(uuid.uuid4()),
            external_url=session.get("sessionUrl"),
            event_type=EventType.ERROR,
            severity=self._parse_severity(data.get("severity", "error")),
            title=title,
            message=message,
            error_type=error.get("type"),
            stack_frames=stack_frames,
            raw_stack_trace=stack_trace,
            file_path=stack_frames[0].filename if stack_frames else None,
            function_name=stack_frames[0].function if stack_frames else None,
            line_number=stack_frames[0].lineno if stack_frames else None,
            component=component,
            url=page_url,
            browser=session.get("browser"),
            os=session.get("os"),
            fingerprint=fingerprint,
            raw_payload=data,
            metadata={
                "logrocket_session_url": session.get("sessionUrl"),
                "logrocket_app_id": data.get("appId"),
            },
        )

    def _normalize_newrelic(self, data: dict) -> NormalizedEvent:
        """Normalize NewRelic webhook payload."""
        import uuid

        incident = data.get("incident", data)
        title = incident.get("incident_title") or incident.get("condition_name") or data.get("title") or "NewRelic Alert"
        message = incident.get("details") or data.get("message") or ""

        targets = incident.get("targets", [])
        page_url = targets[0].get("link") if targets else data.get("url")

        fingerprint = self._generate_fingerprint("newrelic_alert", message, None, page_url)

        priority = incident.get("priority", "HIGH")
        severity = Severity.ERROR
        if priority == "CRITICAL":
            severity = Severity.FATAL
        elif priority == "HIGH":
            severity = Severity.ERROR
        elif priority == "MEDIUM":
            severity = Severity.WARNING
        else:
            severity = Severity.INFO

        return NormalizedEvent(
            id=str(uuid.uuid4()),
            source=EventSource.NEWRELIC,
            external_id=str(incident.get("incident_id") or data.get("id") or uuid.uuid4()),
            external_url=incident.get("incident_url"),
            event_type=EventType.ERROR,
            severity=severity,
            title=title,
            message=message,
            url=page_url,
            fingerprint=fingerprint,
            raw_payload=data,
            metadata={
                "newrelic_account_id": incident.get("account_id"),
                "newrelic_condition_name": incident.get("condition_name"),
                "newrelic_policy_name": incident.get("policy_name"),
            },
        )

    def _normalize_bugsnag(self, data: dict) -> NormalizedEvent:
        """Normalize Bugsnag webhook payload."""
        import uuid

        error = data.get("error", {})
        trigger = data.get("trigger", {})

        title = error.get("errorClass") or error.get("exceptionClass") or data.get("title") or "Bugsnag Error"
        message = error.get("message") or ""

        # Parse stack trace
        stacktrace = error.get("stacktrace", [])
        stack_frames = []
        for frame in stacktrace:
            stack_frames.append(StackFrame(
                filename=frame.get("file"),
                function=frame.get("method"),
                lineno=frame.get("lineNumber"),
                in_app=frame.get("inProject", True),
            ))

        raw_stack = "\n".join(
            f"  at {frame.get('method', 'anonymous')} ({frame.get('file')}:{frame.get('lineNumber')})"
            for frame in stacktrace
        )

        page_url = error.get("context") or error.get("url")
        component = self._extract_component(raw_stack)
        fingerprint = self._generate_fingerprint(error.get("errorClass", "Error"), message, component, page_url)

        return NormalizedEvent(
            id=str(uuid.uuid4()),
            source=EventSource.BUGSNAG,
            external_id=error.get("id") or str(uuid.uuid4()),
            external_url=error.get("url"),
            event_type=EventType.ERROR,
            severity=self._parse_severity(error.get("severity", "error")),
            title=title,
            message=message,
            error_type=error.get("errorClass"),
            stack_frames=stack_frames,
            raw_stack_trace=raw_stack if raw_stack else None,
            file_path=stack_frames[0].filename if stack_frames else None,
            function_name=stack_frames[0].function if stack_frames else None,
            line_number=stack_frames[0].lineno if stack_frames else None,
            component=component,
            url=page_url,
            occurrence_count=error.get("eventsCount", 1),
            affected_users=error.get("usersCount", 1),
            first_seen_at=self._parse_datetime(error.get("firstSeen")),
            last_seen_at=self._parse_datetime(error.get("lastSeen")),
            fingerprint=fingerprint,
            raw_payload=data,
            metadata={
                "bugsnag_project_id": data.get("project", {}).get("id"),
                "bugsnag_project_name": data.get("project", {}).get("name"),
                "bugsnag_trigger": trigger.get("type"),
            },
        )

    def _normalize_rollbar(self, data: dict) -> NormalizedEvent:
        """Normalize Rollbar webhook payload."""
        import uuid

        event_name = data.get("event_name", "new_item")
        item = data.get("data", {}).get("item", {})
        occurrence = data.get("data", {}).get("occurrence", item.get("last_occurrence", {}))

        title = item.get("title") or occurrence.get("exception", {}).get("class") or "Rollbar Error"
        message = occurrence.get("exception", {}).get("message") or item.get("message") or ""

        # Parse stack trace
        frames = occurrence.get("exception", {}).get("frames", [])
        stack_frames = []
        for frame in reversed(frames):
            stack_frames.append(StackFrame(
                filename=frame.get("filename"),
                function=frame.get("method"),
                lineno=frame.get("lineno"),
            ))

        raw_stack = "\n".join(
            f"  at {frame.get('method', 'anonymous')} ({frame.get('filename')}:{frame.get('lineno')})"
            for frame in reversed(frames)
        )

        page_url = occurrence.get("request", {}).get("url") or occurrence.get("context")
        component = self._extract_component(raw_stack)
        fingerprint = self._generate_fingerprint(
            occurrence.get("exception", {}).get("class", "Error"),
            message, component, page_url
        )

        return NormalizedEvent(
            id=str(uuid.uuid4()),
            source=EventSource.ROLLBAR,
            external_id=str(item.get("id") or occurrence.get("id") or uuid.uuid4()),
            external_url=item.get("public_item_handle"),
            event_type=EventType.ERROR,
            severity=self._parse_severity(item.get("level", "error")),
            title=title,
            message=message,
            error_type=occurrence.get("exception", {}).get("class"),
            stack_frames=stack_frames,
            raw_stack_trace=raw_stack if raw_stack else None,
            file_path=stack_frames[0].filename if stack_frames else None,
            function_name=stack_frames[0].function if stack_frames else None,
            line_number=stack_frames[0].lineno if stack_frames else None,
            component=component,
            url=page_url,
            browser=occurrence.get("client", {}).get("browser"),
            os=occurrence.get("client", {}).get("os"),
            occurrence_count=item.get("total_occurrences", 1),
            affected_users=item.get("unique_occurrences", 1),
            first_seen_at=self._parse_datetime(item.get("first_occurrence_timestamp")),
            last_seen_at=self._parse_datetime(item.get("last_occurrence_timestamp")),
            fingerprint=fingerprint,
            raw_payload=data,
            metadata={
                "rollbar_environment": item.get("environment"),
                "rollbar_framework": item.get("framework"),
                "rollbar_event": event_name,
            },
        )

    def _normalize_github_actions(self, data: dict) -> NormalizedEvent:
        """Normalize GitHub Actions webhook payload."""
        import uuid

        action = data.get("action", "")
        run = data.get("workflow_run", {})
        repo = data.get("repository", {})

        # Map status
        status = "pending"
        if run.get("status") == "completed":
            conclusion = run.get("conclusion", "").lower()
            if conclusion == "success":
                status = "success"
            elif conclusion in ("failure", "timed_out"):
                status = "failure"
            elif conclusion == "cancelled":
                status = "cancelled"
        elif run.get("status") == "in_progress":
            status = "running"

        title = f"{run.get('name', 'Workflow')}: {status}"
        message = f"Branch: {run.get('head_branch')} | Commit: {run.get('head_sha', '')[:7]}"

        return NormalizedEvent(
            id=str(uuid.uuid4()),
            source=EventSource.GITHUB_ACTIONS,
            external_id=str(run.get("id", uuid.uuid4())),
            external_url=run.get("html_url"),
            event_type=EventType.CI_RUN,
            severity=Severity.ERROR if status == "failure" else Severity.INFO,
            title=title,
            message=message,
            fingerprint=f"gh-{repo.get('full_name')}-{run.get('workflow_id')}-{run.get('run_number')}",
            tags=[f"branch:{run.get('head_branch')}", f"status:{status}"],
            raw_payload=data,
            metadata={
                "repository": repo.get("full_name"),
                "workflow_id": run.get("workflow_id"),
                "workflow_name": run.get("name"),
                "run_number": run.get("run_number"),
                "branch": run.get("head_branch"),
                "commit_sha": run.get("head_sha"),
                "status": status,
                "conclusion": run.get("conclusion"),
                "actor": run.get("actor", {}).get("login") if run.get("actor") else None,
            },
        )

    def _normalize_generic(self, source: EventSource, data: dict) -> NormalizedEvent:
        """Generic normalizer for unknown sources."""
        import uuid

        return NormalizedEvent(
            id=str(uuid.uuid4()),
            source=source,
            external_id=data.get("id") or str(uuid.uuid4()),
            event_type=EventType.ERROR,
            severity=Severity.ERROR,
            title=data.get("title") or data.get("message", "Unknown Event")[:100],
            message=data.get("message") or str(data)[:500],
            fingerprint=hashlib.sha256(str(data).encode()).hexdigest()[:12],
            raw_payload=data,
        )

    # Helper methods

    def _parse_severity(self, level: str) -> Severity:
        """Parse severity level."""
        level_lower = level.lower()
        if level_lower == "fatal":
            return Severity.FATAL
        elif level_lower in ("error", "high"):
            return Severity.ERROR
        elif level_lower in ("warning", "warn", "normal"):
            return Severity.WARNING
        return Severity.INFO

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            try:
                # ISO format
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        return None

    def _generate_fingerprint(
        self,
        error_type: str,
        message: str,
        component: Optional[str],
        url: Optional[str],
    ) -> str:
        """Generate fingerprint for error grouping."""
        parts = [error_type, message[:100] if message else ""]
        if component:
            parts.append(component)
        if url:
            # Normalize URL
            normalized_url = re.sub(r"\?.*$", "", url)
            normalized_url = re.sub(r"/\d+", "/:id", normalized_url)
            normalized_url = re.sub(r"/[a-f0-9-]{36}", "/:uuid", normalized_url)
            parts.append(normalized_url)

        combined = "|".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:12]

    def _extract_component(self, stack_trace: Optional[str]) -> Optional[str]:
        """Extract UI component name from stack trace."""
        if not stack_trace:
            return None

        # React component pattern
        react_match = re.search(r"at\s+([A-Z][a-zA-Z0-9]*)\s+\(", stack_trace)
        if react_match:
            return react_match.group(1)

        # Vue component pattern
        vue_match = re.search(r"VueComponent\.([a-zA-Z0-9_]+)", stack_trace)
        if vue_match:
            return vue_match.group(1)

        # Angular component pattern
        angular_match = re.search(r"([A-Z][a-zA-Z0-9]*Component)\.", stack_trace)
        if angular_match:
            return angular_match.group(1)

        return None

    def _parse_generic_stack(self, stack_trace: str) -> list[StackFrame]:
        """Parse a generic stack trace string into frames."""
        frames = []

        # Match patterns like "at functionName (file.js:10:5)"
        pattern = r"at\s+([^\s(]+)\s*\(([^:]+):(\d+):?(\d+)?\)"

        for match in re.finditer(pattern, stack_trace):
            frames.append(StackFrame(
                function=match.group(1),
                filename=match.group(2),
                lineno=int(match.group(3)),
                colno=int(match.group(4)) if match.group(4) else None,
            ))

        return frames
