"""Tests for the error normalizer module."""



class TestEventSource:
    """Tests for EventSource enum."""

    def test_event_sources_exist(self, mock_env_vars):
        """Test that all event sources are defined."""
        from src.core.normalizer import EventSource

        assert EventSource.SENTRY == "sentry"
        assert EventSource.DATADOG == "datadog"
        assert EventSource.FULLSTORY == "fullstory"
        assert EventSource.LOGROCKET == "logrocket"
        assert EventSource.NEWRELIC == "newrelic"
        assert EventSource.BUGSNAG == "bugsnag"
        assert EventSource.ROLLBAR == "rollbar"
        assert EventSource.GITHUB_ACTIONS == "github_actions"


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_exist(self, mock_env_vars):
        """Test that all event types are defined."""
        from src.core.normalizer import EventType

        assert EventType.ERROR == "error"
        assert EventType.EXCEPTION == "exception"
        assert EventType.PERFORMANCE == "performance"
        assert EventType.SESSION == "session"
        assert EventType.RAGE_CLICK == "rage_click"
        assert EventType.DEAD_CLICK == "dead_click"
        assert EventType.CI_RUN == "ci_run"


class TestSeverity:
    """Tests for Severity enum."""

    def test_severities_exist(self, mock_env_vars):
        """Test that all severities are defined."""
        from src.core.normalizer import Severity

        assert Severity.FATAL == "fatal"
        assert Severity.ERROR == "error"
        assert Severity.WARNING == "warning"
        assert Severity.INFO == "info"


class TestStackFrame:
    """Tests for StackFrame dataclass."""

    def test_stack_frame_creation(self, mock_env_vars):
        """Test creating a StackFrame."""
        from src.core.normalizer import StackFrame

        frame = StackFrame(
            filename="src/auth/login.py",
            function="validate_user",
            lineno=42,
            colno=10,
            context="    if user is None:",
            in_app=True,
        )

        assert frame.filename == "src/auth/login.py"
        assert frame.function == "validate_user"
        assert frame.lineno == 42
        assert frame.in_app is True

    def test_stack_frame_minimal(self, mock_env_vars):
        """Test StackFrame with minimal fields."""
        from src.core.normalizer import StackFrame

        frame = StackFrame(
            filename="test.py",
            function="test",
            lineno=1,
        )

        assert frame.filename == "test.py"
        assert frame.colno is None
        assert frame.in_app is True  # Default

    def test_stack_frame_to_dict(self, mock_env_vars):
        """Test StackFrame to_dict method."""
        from src.core.normalizer import StackFrame

        frame = StackFrame(
            filename="test.py",
            function="main",
            lineno=10,
            context="print('hello')",
            in_app=True,
        )

        result = frame.to_dict()

        assert result["filename"] == "test.py"
        assert result["function"] == "main"
        assert result["lineno"] == 10
        assert result["context"] == "print('hello')"

    def test_stack_frame_from_dict(self, mock_env_vars):
        """Test StackFrame from_dict class method."""
        from src.core.normalizer import StackFrame

        data = {
            "filename": "app.js",
            "function": "handleClick",
            "lineno": 42,
            "colno": 10,
            "in_app": False,
        }

        frame = StackFrame.from_dict(data)

        assert frame.filename == "app.js"
        assert frame.function == "handleClick"
        assert frame.lineno == 42
        assert frame.in_app is False


class TestNormalizedEvent:
    """Tests for NormalizedEvent dataclass."""

    def test_normalized_event_creation(self, mock_env_vars):
        """Test creating a NormalizedEvent."""
        from src.core.normalizer import (
            EventSource,
            EventType,
            NormalizedEvent,
            Severity,
            StackFrame,
        )

        event = NormalizedEvent(
            id="event-001",
            source=EventSource.SENTRY,
            external_id="ext-001",
            event_type=EventType.ERROR,
            title="TypeError: Cannot read property 'name'",
            message="Cannot read property 'name' of undefined",
            severity=Severity.ERROR,
            error_type="TypeError",
            stack_frames=[
                StackFrame(filename="app.js", function="getUser", lineno=42),
            ],
            browser="Chrome 120",
            os="Windows 10",
            url="https://example.com/users",
            component="UserProfile",
            user_id="user-123",
        )

        assert event.id == "event-001"
        assert event.source == EventSource.SENTRY
        assert event.event_type == EventType.ERROR
        assert event.severity == Severity.ERROR
        assert len(event.stack_frames) == 1

    def test_normalized_event_to_dict(self, mock_env_vars):
        """Test NormalizedEvent to_dict method."""
        from src.core.normalizer import (
            EventSource,
            EventType,
            NormalizedEvent,
            Severity,
            StackFrame,
        )

        frame = StackFrame(filename="test.py", function="main", lineno=10)
        event = NormalizedEvent(
            id="event-002",
            source=EventSource.DATADOG,
            external_id="ext-002",
            event_type=EventType.EXCEPTION,
            title="Deprecation Warning",
            message="Method X is deprecated",
            severity=Severity.WARNING,
            stack_frames=[frame],
            tags=["env:production"],
        )

        result = event.to_dict()

        assert result["id"] == "event-002"
        assert result["source"] == "datadog"
        assert result["event_type"] == "exception"
        assert result["severity"] == "warning"
        assert len(result["stack_frames"]) == 1
        assert "env:production" in result["tags"]


class TestEventNormalizer:
    """Tests for EventNormalizer class."""

    def test_normalizer_initialization(self, mock_env_vars):
        """Test EventNormalizer initialization."""
        from src.core.normalizer import EventNormalizer

        normalizer = EventNormalizer()

        assert normalizer is not None

    def test_normalize_sentry_error(self, mock_env_vars):
        """Test normalizing a Sentry error event."""
        from src.core.normalizer import EventNormalizer, EventSource, EventType

        normalizer = EventNormalizer()

        sentry_payload = {
            "data": {
                "issue": {
                    "id": "issue-123",
                    "title": "TypeError: Cannot read property",
                    "level": "error",
                    "count": "5",
                    "userCount": 3,
                    "firstSeen": "2024-01-15T10:00:00Z",
                    "lastSeen": "2024-01-15T12:00:00Z",
                },
                "event": {
                    "event_id": "event-456",
                    "platform": "javascript",
                    "event": {
                        "exception": {
                            "values": [
                                {
                                    "type": "TypeError",
                                    "value": "Cannot read property 'x'",
                                    "stacktrace": {
                                        "frames": [
                                            {
                                                "filename": "app.js",
                                                "function": "handleClick",
                                                "lineno": 42,
                                                "colno": 10,
                                                "in_app": True,
                                            }
                                        ]
                                    },
                                }
                            ]
                        },
                        "request": {"url": "https://example.com/users"},
                        "contexts": {
                            "browser": {"name": "Chrome", "version": "120"},
                            "os": {"name": "Windows", "version": "10"},
                        },
                    },
                },
            },
        }

        event = normalizer.normalize(EventSource.SENTRY, sentry_payload)

        assert event.source == EventSource.SENTRY
        assert event.event_type == EventType.ERROR
        assert event.error_type == "TypeError"
        assert len(event.stack_frames) == 1
        assert event.occurrence_count == 5

    def test_normalize_sentry_minimal(self, mock_env_vars):
        """Test normalizing minimal Sentry event."""
        from src.core.normalizer import EventNormalizer, EventSource

        normalizer = EventNormalizer()

        sentry_payload = {
            "data": {
                "issue": {
                    "id": "minimal-001",
                    "title": "Simple Error",
                },
            },
        }

        event = normalizer.normalize(EventSource.SENTRY, sentry_payload)

        assert event.source == EventSource.SENTRY
        assert event.title == "Simple Error"

    def test_normalize_datadog_error(self, mock_env_vars):
        """Test normalizing a Datadog error event."""
        from src.core.normalizer import EventNormalizer, EventSource

        normalizer = EventNormalizer()

        datadog_payload = {
            "id": "dd-event-001",
            "title": "Application Error",
            "message": "Error occurred",
            "alert_type": "error",
            "error": {
                "type": "ValueError",
                "message": "Invalid input",
                "stack": "Error: Invalid input\n    at handleRequest (app.js:10:5)",
            },
            "view": {"url": "https://example.com"},
            "host": "server-01",
        }

        event = normalizer.normalize(EventSource.DATADOG, datadog_payload)

        assert event.source == EventSource.DATADOG
        assert event.error_type == "ValueError"

    def test_normalize_fullstory_rage_click(self, mock_env_vars):
        """Test normalizing FullStory rage click event."""
        from src.core.normalizer import EventNormalizer, EventSource, EventType

        normalizer = EventNormalizer()

        fullstory_payload = {
            "type": "rage_click",
            "id": "fs-001",
            "title": "Rage Click Detected",
            "session": {"url": "https://app.fullstory.com/session/123"},
            "page": {"url": "https://example.com/checkout"},
            "element": {"selector": "#submit-btn"},
            "count": 5,
            "userCount": 2,
        }

        event = normalizer.normalize(EventSource.FULLSTORY, fullstory_payload)

        assert event.source == EventSource.FULLSTORY
        assert event.event_type == EventType.RAGE_CLICK
        assert event.occurrence_count == 5

    def test_normalize_github_actions_workflow(self, mock_env_vars):
        """Test normalizing GitHub Actions workflow event."""
        from src.core.normalizer import EventNormalizer, EventSource, EventType, Severity

        normalizer = EventNormalizer()

        gh_payload = {
            "action": "completed",
            "workflow_run": {
                "id": 12345,
                "name": "CI Pipeline",
                "status": "completed",
                "conclusion": "failure",
                "head_branch": "main",
                "head_sha": "abc123def456",
                "html_url": "https://github.com/owner/repo/actions/runs/12345",
            },
            "repository": {
                "full_name": "owner/repo",
            },
        }

        event = normalizer.normalize(EventSource.GITHUB_ACTIONS, gh_payload)

        assert event.source == EventSource.GITHUB_ACTIONS
        assert event.event_type == EventType.CI_RUN
        assert event.severity == Severity.ERROR
        assert "failure" in event.title

    def test_normalize_generic_unknown_source(self, mock_env_vars):
        """Test normalizing with unknown source falls back to generic."""
        from src.core.normalizer import EventNormalizer, EventSource

        normalizer = EventNormalizer()

        # Use a source without a specific normalizer
        unknown_payload = {
            "id": "unknown-001",
            "title": "Unknown Event",
            "message": "Something happened",
        }

        event = normalizer.normalize(EventSource.COVERAGE, unknown_payload)

        assert event.title == "Unknown Event"

    def test_parse_severity(self, mock_env_vars):
        """Test _parse_severity helper."""
        from src.core.normalizer import EventNormalizer, Severity

        normalizer = EventNormalizer()

        assert normalizer._parse_severity("fatal") == Severity.FATAL
        assert normalizer._parse_severity("error") == Severity.ERROR
        assert normalizer._parse_severity("high") == Severity.ERROR
        assert normalizer._parse_severity("warning") == Severity.WARNING
        assert normalizer._parse_severity("warn") == Severity.WARNING
        assert normalizer._parse_severity("info") == Severity.INFO
        assert normalizer._parse_severity("unknown") == Severity.INFO

    def test_parse_datetime_iso(self, mock_env_vars):
        """Test _parse_datetime with ISO format."""
        from src.core.normalizer import EventNormalizer

        normalizer = EventNormalizer()

        dt = normalizer._parse_datetime("2024-01-15T12:30:00Z")

        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_parse_datetime_timestamp(self, mock_env_vars):
        """Test _parse_datetime with Unix timestamp."""
        from src.core.normalizer import EventNormalizer

        normalizer = EventNormalizer()

        dt = normalizer._parse_datetime(1705320000)  # Unix timestamp

        assert dt is not None
        assert dt.year >= 2024

    def test_parse_datetime_none(self, mock_env_vars):
        """Test _parse_datetime with None value."""
        from src.core.normalizer import EventNormalizer

        normalizer = EventNormalizer()

        dt = normalizer._parse_datetime(None)

        assert dt is None

    def test_parse_datetime_invalid(self, mock_env_vars):
        """Test _parse_datetime with invalid value."""
        from src.core.normalizer import EventNormalizer

        normalizer = EventNormalizer()

        dt = normalizer._parse_datetime("not a timestamp")

        assert dt is None

    def test_generate_fingerprint(self, mock_env_vars):
        """Test _generate_fingerprint helper."""
        from src.core.normalizer import EventNormalizer

        normalizer = EventNormalizer()

        fp1 = normalizer._generate_fingerprint(
            error_type="TypeError",
            message="Cannot read property",
            component="UserComponent",
            url="https://example.com/users",
        )

        fp2 = normalizer._generate_fingerprint(
            error_type="TypeError",
            message="Cannot read property",
            component="UserComponent",
            url="https://example.com/users",
        )

        # Same inputs should produce same fingerprint
        assert fp1 == fp2
        assert len(fp1) == 12  # SHA256 truncated

    def test_generate_fingerprint_normalizes_urls(self, mock_env_vars):
        """Test that fingerprint normalizes URLs (removes IDs)."""
        from src.core.normalizer import EventNormalizer

        normalizer = EventNormalizer()

        fp1 = normalizer._generate_fingerprint(
            error_type="Error",
            message="Error",
            component=None,
            url="https://example.com/users/123",
        )

        fp2 = normalizer._generate_fingerprint(
            error_type="Error",
            message="Error",
            component=None,
            url="https://example.com/users/456",  # Different ID
        )

        # Should be same after URL normalization
        assert fp1 == fp2

    def test_extract_component_react(self, mock_env_vars):
        """Test _extract_component finds React components."""
        from src.core.normalizer import EventNormalizer

        normalizer = EventNormalizer()

        stack_trace = """Error: Something went wrong
    at UserProfile (components/UserProfile.jsx:42:10)
    at render (react-dom.js:100:5)"""

        component = normalizer._extract_component(stack_trace)

        assert component == "UserProfile"

    def test_extract_component_angular(self, mock_env_vars):
        """Test _extract_component finds Angular components."""
        from src.core.normalizer import EventNormalizer

        normalizer = EventNormalizer()

        stack_trace = """Error: Failed
    at UserProfileComponent.ngOnInit (user-profile.component.ts:25:10)"""

        component = normalizer._extract_component(stack_trace)

        assert component == "UserProfileComponent"

    def test_extract_component_none(self, mock_env_vars):
        """Test _extract_component returns None when no component found."""
        from src.core.normalizer import EventNormalizer

        normalizer = EventNormalizer()

        component = normalizer._extract_component(None)

        assert component is None

    def test_parse_generic_stack(self, mock_env_vars):
        """Test _parse_generic_stack helper."""
        from src.core.normalizer import EventNormalizer

        normalizer = EventNormalizer()

        stack_trace = """Error: Something went wrong
    at handleClick (app.js:42:10)
    at processEvent (utils.js:15:5)"""

        frames = normalizer._parse_generic_stack(stack_trace)

        assert len(frames) == 2
        assert frames[0].function == "handleClick"
        assert frames[0].filename == "app.js"
        assert frames[0].lineno == 42


class TestEventNormalizerIntegration:
    """Integration tests for EventNormalizer with various payloads."""

    def test_normalize_logrocket_error(self, mock_env_vars):
        """Test normalizing LogRocket error event."""
        from src.core.normalizer import EventNormalizer, EventSource

        normalizer = EventNormalizer()

        logrocket_payload = {
            "error": {
                "type": "ReferenceError",
                "message": "x is not defined",
                "stack": "ReferenceError: x is not defined\n    at render (App.js:10:5)",
            },
            "session": {
                "id": "session-123",
                "url": "https://example.com",
                "sessionUrl": "https://app.logrocket.com/session/123",
                "browser": "Chrome 120",
            },
        }

        event = normalizer.normalize(EventSource.LOGROCKET, logrocket_payload)

        assert event.source == EventSource.LOGROCKET
        assert event.error_type == "ReferenceError"

    def test_normalize_newrelic_alert(self, mock_env_vars):
        """Test normalizing NewRelic alert event."""
        from src.core.normalizer import EventNormalizer, EventSource, Severity

        normalizer = EventNormalizer()

        newrelic_payload = {
            "incident": {
                "incident_id": 12345,
                "incident_title": "High Error Rate",
                "condition_name": "Error rate > 5%",
                "priority": "CRITICAL",
                "details": "Error rate exceeded threshold",
                "incident_url": "https://alerts.newrelic.com/incidents/12345",
            },
        }

        event = normalizer.normalize(EventSource.NEWRELIC, newrelic_payload)

        assert event.source == EventSource.NEWRELIC
        assert event.severity == Severity.FATAL

    def test_normalize_bugsnag_error(self, mock_env_vars):
        """Test normalizing Bugsnag error event."""
        from src.core.normalizer import EventNormalizer, EventSource

        normalizer = EventNormalizer()

        bugsnag_payload = {
            "error": {
                "id": "error-123",
                "errorClass": "NullPointerException",
                "message": "Attempt to invoke method on null object",
                "severity": "error",
                "eventsCount": 10,
                "usersCount": 5,
                "stacktrace": [
                    {
                        "file": "MainActivity.java",
                        "method": "onClick",
                        "lineNumber": 42,
                        "inProject": True,
                    }
                ],
            },
            "trigger": {"type": "new"},
        }

        event = normalizer.normalize(EventSource.BUGSNAG, bugsnag_payload)

        assert event.source == EventSource.BUGSNAG
        assert event.error_type == "NullPointerException"
        assert event.occurrence_count == 10

    def test_normalize_rollbar_error(self, mock_env_vars):
        """Test normalizing Rollbar error event."""
        from src.core.normalizer import EventNormalizer, EventSource

        normalizer = EventNormalizer()

        rollbar_payload = {
            "event_name": "new_item",
            "data": {
                "item": {
                    "id": "item-123",
                    "title": "ValueError: Invalid argument",
                    "level": "error",
                    "total_occurrences": 15,
                },
                "occurrence": {
                    "exception": {
                        "class": "ValueError",
                        "message": "Invalid argument",
                        "frames": [
                            {
                                "filename": "app.py",
                                "method": "validate",
                                "lineno": 42,
                            }
                        ],
                    },
                    "request": {"url": "https://api.example.com/users"},
                },
            },
        }

        event = normalizer.normalize(EventSource.ROLLBAR, rollbar_payload)

        assert event.source == EventSource.ROLLBAR
        assert event.error_type == "ValueError"
        assert event.occurrence_count == 15

    def test_normalize_handles_exception(self, mock_env_vars):
        """Test that normalization handles exceptions gracefully."""
        from src.core.normalizer import EventNormalizer, EventSource

        normalizer = EventNormalizer()

        # Malformed payload that might cause parsing errors
        malformed_payload = {
            "completely": "unexpected",
            "structure": [1, 2, 3],
        }

        # Should fall back to generic without raising
        event = normalizer.normalize(EventSource.SENTRY, malformed_payload)

        assert event is not None
        assert event.source == EventSource.SENTRY
