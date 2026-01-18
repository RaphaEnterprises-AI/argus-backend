"""Tests for the session to test conversion module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


class TestSessionEventType:
    """Tests for SessionEventType enum."""

    def test_event_types(self, mock_env_vars):
        """Test SessionEventType enum values."""
        from src.agents.session_to_test import SessionEventType

        assert SessionEventType.CLICK.value == "click"
        assert SessionEventType.INPUT.value == "input"
        assert SessionEventType.SCROLL.value == "scroll"
        assert SessionEventType.NAVIGATION.value == "navigation"
        assert SessionEventType.FORM_SUBMIT.value == "form_submit"
        assert SessionEventType.ERROR.value == "error"
        assert SessionEventType.NETWORK_REQUEST.value == "network_request"
        assert SessionEventType.PAGE_LOAD.value == "page_load"
        assert SessionEventType.CUSTOM.value == "custom"


class TestSessionEvent:
    """Tests for SessionEvent dataclass."""

    def test_event_creation(self, mock_env_vars):
        """Test SessionEvent creation."""
        from src.agents.session_to_test import SessionEvent, SessionEventType

        event = SessionEvent(
            timestamp=datetime.now(),
            type=SessionEventType.CLICK,
            target="#submit-btn",
        )

        assert event.type == SessionEventType.CLICK
        assert event.target == "#submit-btn"
        assert event.value is None

    def test_event_with_all_fields(self, mock_env_vars):
        """Test SessionEvent with all fields."""
        from src.agents.session_to_test import SessionEvent, SessionEventType

        event = SessionEvent(
            timestamp=datetime.now(),
            type=SessionEventType.INPUT,
            target="#email",
            value="test@example.com",
            url="https://example.com/login",
            metadata={"field_type": "email"},
        )

        assert event.value == "test@example.com"
        assert event.url == "https://example.com/login"


class TestUserSession:
    """Tests for UserSession dataclass."""

    def test_session_creation(self, mock_env_vars):
        """Test UserSession creation."""
        from src.agents.session_to_test import UserSession

        session = UserSession(
            session_id="sess-001",
            user_id="user-123",
            started_at=datetime.now(),
            ended_at=None,
        )

        assert session.session_id == "sess-001"
        assert session.events == []
        assert session.errors == []

    def test_session_with_events(self, mock_env_vars):
        """Test UserSession with events."""
        from src.agents.session_to_test import SessionEvent, SessionEventType, UserSession

        events = [
            SessionEvent(
                timestamp=datetime.now(),
                type=SessionEventType.NAVIGATION,
                url="/login",
            ),
            SessionEvent(
                timestamp=datetime.now(),
                type=SessionEventType.CLICK,
                target="#submit",
            ),
        ]

        session = UserSession(
            session_id="sess-001",
            user_id="user-123",
            started_at=datetime.now(),
            ended_at=datetime.now() + timedelta(minutes=5),
            events=events,
            outcome="conversion",
        )

        assert len(session.events) == 2
        assert session.outcome == "conversion"


class TestGeneratedTest:
    """Tests for GeneratedTest dataclass."""

    def test_test_creation(self, mock_env_vars):
        """Test GeneratedTest creation."""
        from src.agents.session_to_test import GeneratedTest

        test = GeneratedTest(
            id="test-001",
            name="Login Test",
            description="Test login flow",
            source_session_ids=["sess-001"],
            priority="high",
            steps=[{"action": "click", "target": "#btn"}],
            assertions=[{"type": "url_matches", "expected": "/dashboard"}],
            preconditions=["User must be logged out"],
            rationale="Generated from session",
            confidence=0.85,
            user_journey="Login",
        )

        assert test.id == "test-001"
        assert test.confidence == 0.85


class TestSessionAnalyzer:
    """Tests for SessionAnalyzer class."""

    def test_analyzer_creation(self, mock_env_vars):
        """Test SessionAnalyzer creation."""
        with patch('src.agents.session_to_test.Anthropic'):
            from src.agents.session_to_test import SessionAnalyzer

            analyzer = SessionAnalyzer()

            assert analyzer.client is not None

    def test_is_sensitive(self, mock_env_vars):
        """Test sensitive field detection."""
        with patch('src.agents.session_to_test.Anthropic'):
            from src.agents.session_to_test import SessionAnalyzer

            analyzer = SessionAnalyzer()

            assert analyzer._is_sensitive("#password") is True
            assert analyzer._is_sensitive("#password-input") is True
            assert analyzer._is_sensitive("#credit-card") is True
            assert analyzer._is_sensitive("#email") is False
            assert analyzer._is_sensitive("#username") is False
            assert analyzer._is_sensitive(None) is False

    def test_summarize_events(self, mock_env_vars):
        """Test event summarization."""
        with patch('src.agents.session_to_test.Anthropic'):
            from src.agents.session_to_test import SessionAnalyzer, SessionEvent, SessionEventType

            analyzer = SessionAnalyzer()

            events = [
                SessionEvent(
                    timestamp=datetime(2024, 1, 1, 10, 0, 0),
                    type=SessionEventType.NAVIGATION,
                    url="/login",
                ),
                SessionEvent(
                    timestamp=datetime(2024, 1, 1, 10, 0, 5),
                    type=SessionEventType.CLICK,
                    target="#submit-btn",
                ),
                SessionEvent(
                    timestamp=datetime(2024, 1, 1, 10, 0, 10),
                    type=SessionEventType.INPUT,
                    target="#email",
                    value="test@example.com",
                ),
                SessionEvent(
                    timestamp=datetime(2024, 1, 1, 10, 0, 15),
                    type=SessionEventType.INPUT,
                    target="#password",
                    value="secret123",
                ),
            ]

            summary = analyzer._summarize_events(events)

            assert "NAVIGATE to /login" in summary
            assert "CLICK on #submit-btn" in summary
            assert "[MASKED]" in summary  # Password should be masked

    @pytest.mark.asyncio
    async def test_analyze_session(self, mock_env_vars):
        """Test session analysis."""
        with patch('src.agents.session_to_test.Anthropic') as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='''
            {
                "user_intent": "Login to account",
                "journey_name": "Login Flow",
                "success": true,
                "test_worthy": true,
                "test_priority": "high",
                "suggested_assertions": ["URL contains /dashboard"]
            }
            ''')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.session_to_test import SessionAnalyzer, UserSession

            analyzer = SessionAnalyzer()

            session = UserSession(
                session_id="sess-001",
                user_id="user-123",
                started_at=datetime.now(),
                ended_at=datetime.now() + timedelta(minutes=5),
            )

            result = await analyzer.analyze_session(session)

            assert result["user_intent"] == "Login to account"
            assert result["test_worthy"] is True


class TestSessionToTestConverter:
    """Tests for SessionToTestConverter class."""

    def test_converter_creation(self, mock_env_vars):
        """Test SessionToTestConverter creation."""
        with patch('src.agents.session_to_test.Anthropic'):
            from src.agents.session_to_test import SessionToTestConverter

            converter = SessionToTestConverter()

            assert converter.analyzer is not None

    def test_extract_steps(self, mock_env_vars):
        """Test step extraction from events."""
        with patch('src.agents.session_to_test.Anthropic'):
            from src.agents.session_to_test import (
                SessionEvent,
                SessionEventType,
                SessionToTestConverter,
            )

            converter = SessionToTestConverter()

            events = [
                SessionEvent(
                    timestamp=datetime.now(),
                    type=SessionEventType.NAVIGATION,
                    url="/login",
                ),
                SessionEvent(
                    timestamp=datetime.now(),
                    type=SessionEventType.CLICK,
                    target="#submit-btn",
                ),
                SessionEvent(
                    timestamp=datetime.now(),
                    type=SessionEventType.INPUT,
                    target="#email",
                    value="test@example.com",
                ),
            ]

            steps = converter._extract_steps(events, generalize=False)

            assert len(steps) == 3
            assert steps[0]["action"] == "navigate"
            assert steps[1]["action"] == "click"
            assert steps[2]["action"] == "type"

    def test_generalize_url(self, mock_env_vars):
        """Test URL generalization."""
        with patch('src.agents.session_to_test.Anthropic'):
            from src.agents.session_to_test import SessionToTestConverter

            converter = SessionToTestConverter()

            # Test ID replacement
            assert "/{id}/" in converter._generalize_url("/products/123/details")

            # Test UUID replacement
            assert "/{uuid}/" in converter._generalize_url("/users/550e8400-e29b-41d4-a716-446655440000/profile")

            # Test query param removal
            assert "?" not in converter._generalize_url("/search?q=test")

            # Test None
            assert converter._generalize_url(None) is None

    def test_generalize_value(self, mock_env_vars):
        """Test value generalization."""
        with patch('src.agents.session_to_test.Anthropic'):
            from src.agents.session_to_test import SessionToTestConverter

            converter = SessionToTestConverter()

            assert converter._generalize_value("#email", "test@example.com") == "{{test_email}}"
            assert converter._generalize_value("#password", "secret") == "{{test_password}}"
            assert converter._generalize_value("#name", "John") == "{{test_name}}"
            assert converter._generalize_value("#phone", "123-456-7890") == "{{test_phone}}"
            assert converter._generalize_value("#search", "query") == "{{search_query}}"
            assert converter._generalize_value("#other", "value") == "value"

    def test_infer_preconditions(self, mock_env_vars):
        """Test precondition inference."""
        with patch('src.agents.session_to_test.Anthropic'):
            from src.agents.session_to_test import (
                SessionEvent,
                SessionEventType,
                SessionToTestConverter,
                UserSession,
            )

            converter = SessionToTestConverter()

            # Session starting with login
            session1 = UserSession(
                session_id="sess-001",
                user_id=None,
                started_at=datetime.now(),
                ended_at=None,
                events=[
                    SessionEvent(
                        timestamp=datetime.now(),
                        type=SessionEventType.NAVIGATION,
                        url="/login",
                    ),
                ],
            )

            precond1 = converter._infer_preconditions(session1)
            assert any("logged out" in p.lower() for p in precond1)

            # Session accessing account
            session2 = UserSession(
                session_id="sess-002",
                user_id="user-123",
                started_at=datetime.now(),
                ended_at=None,
                events=[
                    SessionEvent(
                        timestamp=datetime.now(),
                        type=SessionEventType.NAVIGATION,
                        url="/account/settings",
                    ),
                ],
            )

            precond2 = converter._infer_preconditions(session2)
            assert any("logged in" in p.lower() for p in precond2)

    @pytest.mark.asyncio
    async def test_convert_session(self, mock_env_vars):
        """Test full session conversion."""
        with patch('src.agents.session_to_test.Anthropic') as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='''
            {
                "user_intent": "Login to account",
                "journey_name": "Login Flow",
                "success": true,
                "test_worthy": true,
                "test_priority": "high",
                "suggested_assertions": []
            }
            ''')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.session_to_test import (
                SessionEvent,
                SessionEventType,
                SessionToTestConverter,
                UserSession,
            )

            converter = SessionToTestConverter()

            session = UserSession(
                session_id="sess-001",
                user_id="user-123",
                started_at=datetime.now(),
                ended_at=datetime.now() + timedelta(minutes=5),
                events=[
                    SessionEvent(
                        timestamp=datetime.now(),
                        type=SessionEventType.NAVIGATION,
                        url="/login",
                    ),
                ],
                outcome="conversion",
            )

            test = await converter.convert_session(session)

            assert test.id == "session-sess-001"
            assert test.user_journey == "Login Flow"


class TestErrorToTestConverter:
    """Tests for ErrorToTestConverter class."""

    def test_converter_creation(self, mock_env_vars):
        """Test ErrorToTestConverter creation."""
        with patch('src.agents.session_to_test.Anthropic'):
            from src.agents.session_to_test import ErrorToTestConverter

            converter = ErrorToTestConverter()

            assert converter.session_converter is not None

    @pytest.mark.asyncio
    async def test_convert_error_with_session(self, mock_env_vars):
        """Test converting error with session."""
        with patch('src.agents.session_to_test.Anthropic') as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='''
            {
                "user_intent": "Buy product",
                "journey_name": "Checkout",
                "success": false,
                "test_worthy": true,
                "test_priority": "critical"
            }
            ''')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.session_to_test import ErrorToTestConverter, UserSession

            converter = ErrorToTestConverter()

            error_event = {
                "id": "err-001",
                "message": "Payment failed: Invalid card",
            }

            session = UserSession(
                session_id="sess-001",
                user_id="user-123",
                started_at=datetime.now(),
                ended_at=datetime.now() + timedelta(minutes=5),
                outcome="error",
            )

            test = await converter.convert_error(error_event, session)

            assert "Regression" in test.name
            assert test.priority == "critical"

    @pytest.mark.asyncio
    async def test_convert_error_without_session(self, mock_env_vars):
        """Test converting error without session."""
        with patch('src.agents.session_to_test.Anthropic') as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='''
            {
                "name": "Payment Error Test",
                "steps": [{"action": "click", "target": "#pay-btn"}],
                "assertions": [],
                "preconditions": []
            }
            ''')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.session_to_test import ErrorToTestConverter

            converter = ErrorToTestConverter()

            error_event = {
                "id": "err-001",
                "message": "Payment failed: Invalid card",
            }

            test = await converter._generate_minimal_test(error_event)

            assert test.id == "error-err-001"
            assert test.priority == "critical"

    @pytest.mark.asyncio
    async def test_generate_minimal_test_fallback(self, mock_env_vars):
        """Test minimal test fallback on parse error."""
        with patch('src.agents.session_to_test.Anthropic') as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Invalid response")]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.session_to_test import ErrorToTestConverter

            converter = ErrorToTestConverter()

            error_event = {
                "id": "err-001",
                "message": "Some error",
            }

            test = await converter._generate_minimal_test(error_event)

            assert test.confidence == 0.0
            assert "Manual Review" in test.name


class TestRUMIntegration:
    """Tests for RUMIntegration class."""

    @pytest.mark.asyncio
    async def test_fetch_sessions_from_fullstory(self, mock_env_vars):
        """Test FullStory integration stub."""
        from src.agents.session_to_test import RUMIntegration

        rum = RUMIntegration()
        sessions = await rum.fetch_sessions_from_fullstory("api-key")

        assert sessions == []

    @pytest.mark.asyncio
    async def test_fetch_sessions_from_logrocket(self, mock_env_vars):
        """Test LogRocket integration stub."""
        from src.agents.session_to_test import RUMIntegration

        rum = RUMIntegration()
        sessions = await rum.fetch_sessions_from_logrocket("api-key")

        assert sessions == []

    @pytest.mark.asyncio
    async def test_fetch_errors_from_sentry(self, mock_env_vars):
        """Test Sentry integration stub."""
        from src.agents.session_to_test import RUMIntegration

        rum = RUMIntegration()
        errors = await rum.fetch_errors_from_sentry("api-key", "project")

        assert errors == []

    @pytest.mark.asyncio
    async def test_fetch_errors_from_datadog(self, mock_env_vars):
        """Test Datadog integration stub."""
        from src.agents.session_to_test import RUMIntegration

        rum = RUMIntegration()
        errors = await rum.fetch_errors_from_datadog("api-key", "app-key")

        assert errors == []
