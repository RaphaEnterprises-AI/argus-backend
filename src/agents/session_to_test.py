"""
Session Replay to Test Conversion

REVOLUTIONARY FEATURE: Convert real user sessions into executable tests.

This solves the fundamental problem of testing:
"How do we know what to test?"

Answer: Test what REAL USERS actually do, not what we think they do.

Integration points:
- FullStory / LogRocket / Hotjar session recordings
- Real User Monitoring (RUM) data
- Error tracking (Sentry, Datadog, etc.)
- Analytics events (Amplitude, Mixpanel, etc.)
"""

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from anthropic import Anthropic

from src.config import get_settings


class SessionEventType(str, Enum):
    CLICK = "click"
    INPUT = "input"
    SCROLL = "scroll"
    NAVIGATION = "navigation"
    FORM_SUBMIT = "form_submit"
    ERROR = "error"
    NETWORK_REQUEST = "network_request"
    PAGE_LOAD = "page_load"
    CUSTOM = "custom"


@dataclass
class SessionEvent:
    """A single event from a user session."""
    timestamp: datetime
    type: SessionEventType
    target: str | None = None  # CSS selector or description
    value: str | None = None
    url: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class UserSession:
    """A recorded user session."""
    session_id: str
    user_id: str | None
    started_at: datetime
    ended_at: datetime | None
    events: list[SessionEvent] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    device_info: dict = field(default_factory=dict)
    geo_info: dict = field(default_factory=dict)
    outcome: str | None = None  # "conversion", "abandonment", "error"


@dataclass
class GeneratedTest:
    """A test generated from session analysis."""
    id: str
    name: str
    description: str
    source_session_ids: list[str]
    priority: str
    steps: list[dict]
    assertions: list[dict]
    preconditions: list[str]
    rationale: str
    confidence: float
    user_journey: str


class SessionAnalyzer:
    """
    Analyzes user sessions to understand behavior patterns.

    This is NOT just recording and replaying. We UNDERSTAND what users
    are trying to accomplish and generate intelligent tests from that understanding.
    """

    def __init__(self):
        self.settings = get_settings()
        api_key = self.settings.anthropic_api_key
        if hasattr(api_key, 'get_secret_value'):
            api_key = api_key.get_secret_value()
        self.client = Anthropic(api_key=api_key)

    async def analyze_session(
        self,
        session: UserSession
    ) -> dict:
        """
        Deeply analyze a user session to understand:
        - What was the user trying to accomplish?
        - Did they succeed or fail?
        - What obstacles did they encounter?
        - What's the test-worthy behavior here?
        """
        events_summary = self._summarize_events(session.events)

        prompt = f"""Analyze this user session to understand the user's intent and experience.

SESSION OVERVIEW:
- Duration: {(session.ended_at - session.started_at).total_seconds() if session.ended_at else 'ongoing'} seconds
- Device: {json.dumps(session.device_info)}
- Location: {json.dumps(session.geo_info)}
- Outcome: {session.outcome or 'unknown'}

EVENTS TIMELINE:
{events_summary}

ERRORS ENCOUNTERED:
{json.dumps(session.errors[:10], indent=2)}

Analyze and provide:
{{
    "user_intent": "What the user was trying to accomplish",
    "journey_name": "Name for this user journey (e.g., 'Product Search and Purchase')",
    "success": true/false,
    "obstacles": ["Issues the user encountered"],
    "key_actions": ["Most important actions in the session"],
    "test_worthy": true/false,
    "test_priority": "critical|high|medium|low",
    "suggested_assertions": ["What should we verify works correctly"],
    "edge_cases_revealed": ["Edge cases this session revealed"],
    "ux_insights": ["UX improvement opportunities"]
}}"""

        from src.core.model_registry import get_model_id
        response = self.client.messages.create(
            model=get_model_id("claude-sonnet-4-5"),
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.content[0].text
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0:
                return json.loads(text[json_start:json_end])
        except:
            pass

        return {"user_intent": "Unknown", "test_worthy": False}

    def _summarize_events(self, events: list[SessionEvent]) -> str:
        """Create a human-readable summary of events."""
        lines = []
        for i, event in enumerate(events[:100]):  # Limit to first 100 events
            timestamp = event.timestamp.strftime("%H:%M:%S")
            if event.type == SessionEventType.CLICK:
                lines.append(f"{timestamp} CLICK on {event.target}")
            elif event.type == SessionEventType.INPUT:
                # Mask sensitive data
                value = "[MASKED]" if self._is_sensitive(event.target) else event.value
                lines.append(f"{timestamp} TYPE '{value}' into {event.target}")
            elif event.type == SessionEventType.NAVIGATION:
                lines.append(f"{timestamp} NAVIGATE to {event.url}")
            elif event.type == SessionEventType.FORM_SUBMIT:
                lines.append(f"{timestamp} SUBMIT form {event.target}")
            elif event.type == SessionEventType.ERROR:
                lines.append(f"{timestamp} ERROR: {event.value}")
            elif event.type == SessionEventType.SCROLL:
                continue  # Skip scroll events for brevity
            else:
                lines.append(f"{timestamp} {event.type.value}: {event.value}")

        return "\n".join(lines)

    def _is_sensitive(self, target: str) -> bool:
        """Check if a field likely contains sensitive data."""
        sensitive_patterns = [
            "password", "passwd", "secret", "token",
            "ssn", "credit", "card", "cvv", "cvc"
        ]
        if target:
            target_lower = target.lower()
            return any(p in target_lower for p in sensitive_patterns)
        return False


class SessionToTestConverter:
    """
    Converts user sessions into executable tests.

    The magic here is that we don't just record and replay.
    We UNDERSTAND the session and generate a PROPER test with:
    - Meaningful assertions (not just "did it not crash")
    - Proper test data handling
    - Edge case coverage
    - Maintainable structure
    """

    def __init__(self):
        self.settings = get_settings()
        api_key = self.settings.anthropic_api_key
        if hasattr(api_key, 'get_secret_value'):
            api_key = api_key.get_secret_value()
        self.client = Anthropic(api_key=api_key)
        self.analyzer = SessionAnalyzer()

    async def convert_session(
        self,
        session: UserSession,
        include_assertions: bool = True,
        generalize: bool = True
    ) -> GeneratedTest:
        """
        Convert a single session into a test.

        Args:
            session: The user session to convert
            include_assertions: Whether to add intelligent assertions
            generalize: Whether to generalize test data for reuse
        """
        # First, analyze to understand the session
        analysis = await self.analyzer.analyze_session(session)

        if not analysis.get("test_worthy", False):
            # Session isn't worth testing, but we can still generate if forced
            pass

        # Generate test steps
        steps = self._extract_steps(session.events, generalize)

        # Generate intelligent assertions
        assertions = []
        if include_assertions:
            assertions = await self._generate_assertions(
                session, analysis, steps
            )

        test = GeneratedTest(
            id=f"session-{session.session_id[:8]}",
            name=analysis.get("journey_name", f"User Journey {session.session_id[:8]}"),
            description=analysis.get("user_intent", ""),
            source_session_ids=[session.session_id],
            priority=analysis.get("test_priority", "medium"),
            steps=steps,
            assertions=assertions,
            preconditions=self._infer_preconditions(session),
            rationale=f"Generated from real user session. {analysis.get('user_intent', '')}",
            confidence=0.8 if analysis.get("success") else 0.6,
            user_journey=analysis.get("journey_name", "Unknown")
        )

        return test

    def _extract_steps(
        self,
        events: list[SessionEvent],
        generalize: bool
    ) -> list[dict]:
        """Extract test steps from session events."""
        steps = []

        for event in events:
            if event.type == SessionEventType.NAVIGATION:
                steps.append({
                    "action": "navigate",
                    "url": self._generalize_url(event.url) if generalize else event.url
                })
            elif event.type == SessionEventType.CLICK:
                steps.append({
                    "action": "click",
                    "target": event.target,
                    "description": f"Click on {event.target}"
                })
            elif event.type == SessionEventType.INPUT:
                value = event.value
                if generalize:
                    value = self._generalize_value(event.target, event.value)
                steps.append({
                    "action": "type",
                    "target": event.target,
                    "value": value,
                    "description": f"Enter value in {event.target}"
                })
            elif event.type == SessionEventType.FORM_SUBMIT:
                steps.append({
                    "action": "submit",
                    "target": event.target
                })
            elif event.type == SessionEventType.SCROLL:
                # Only include significant scrolls
                continue

        return steps

    def _generalize_url(self, url: str) -> str:
        """Generalize URL to be reusable."""
        if not url:
            return url

        # Replace IDs with placeholders
        url = re.sub(r'/\d+/', '/{id}/', url)
        url = re.sub(r'/[a-f0-9-]{36}/', '/{uuid}/', url)

        # Replace query params with placeholders
        url = re.sub(r'\?.*$', '', url)

        return url

    def _generalize_value(self, target: str, value: str) -> str:
        """Generalize input values for test data."""
        if not target or not value:
            return value

        target_lower = target.lower()

        # Map field types to test data variables
        if "email" in target_lower:
            return "{{test_email}}"
        elif "password" in target_lower:
            return "{{test_password}}"
        elif "name" in target_lower:
            return "{{test_name}}"
        elif "phone" in target_lower:
            return "{{test_phone}}"
        elif "address" in target_lower:
            return "{{test_address}}"
        elif "search" in target_lower:
            return "{{search_query}}"

        return value

    def _infer_preconditions(self, session: UserSession) -> list[str]:
        """Infer test preconditions from session."""
        preconditions = []

        # Check if user was logged in
        for event in session.events:
            if event.url and "/login" in event.url:
                preconditions.append("User must be logged out initially")
                break
            if event.url and ("/account" in event.url or "/profile" in event.url):
                preconditions.append("User must be logged in")
                break

        return preconditions

    async def _generate_assertions(
        self,
        session: UserSession,
        analysis: dict,
        steps: list[dict]
    ) -> list[dict]:
        """Generate intelligent assertions based on session analysis."""
        assertions = []

        # Use Claude to generate meaningful assertions
        prompt = f"""Generate test assertions for this user journey.

USER INTENT: {analysis.get('user_intent', 'Unknown')}

TEST STEPS:
{json.dumps(steps[:20], indent=2)}

SUGGESTED ASSERTIONS FROM ANALYSIS:
{json.dumps(analysis.get('suggested_assertions', []), indent=2)}

SESSION OUTCOME: {session.outcome or 'unknown'}

Generate assertions that verify:
1. The user's goal was achieved
2. Expected UI elements are present
3. Data was saved/processed correctly
4. No errors occurred

Format as JSON array:
[
    {{
        "type": "element_visible|text_contains|url_matches|element_enabled|custom",
        "target": "selector or URL pattern",
        "expected": "expected value",
        "description": "what this assertion verifies"
    }}
]"""

        from src.core.model_registry import get_model_id
        response = self.client.messages.create(
            model=get_model_id("claude-sonnet-4-5"),
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.content[0].text
            json_start = text.find("[")
            json_end = text.rfind("]") + 1
            if json_start >= 0:
                assertions = json.loads(text[json_start:json_end])
        except:
            pass

        return assertions

    async def convert_multiple_sessions(
        self,
        sessions: list[UserSession],
        deduplicate: bool = True
    ) -> AsyncIterator[GeneratedTest]:
        """
        Convert multiple sessions into tests, optionally deduplicating similar journeys.

        When deduplicate=True, we:
        1. Group sessions by user journey type
        2. Create ONE test per journey type
        3. Use the most common path as the test
        4. Include variations as parameterized test cases
        """
        if not deduplicate:
            for session in sessions:
                yield await self.convert_session(session)
            return

        # Group sessions by journey
        journey_groups = await self._group_by_journey(sessions)

        for journey_name, group_sessions in journey_groups.items():
            # Create a composite test from the group
            test = await self._create_composite_test(journey_name, group_sessions)
            yield test

    async def _group_by_journey(
        self,
        sessions: list[UserSession]
    ) -> dict[str, list[UserSession]]:
        """Group sessions by their user journey."""
        groups = {}

        for session in sessions:
            analysis = await self.analyzer.analyze_session(session)
            journey = analysis.get("journey_name", "unknown")

            if journey not in groups:
                groups[journey] = []
            groups[journey].append(session)

        return groups

    async def _create_composite_test(
        self,
        journey_name: str,
        sessions: list[UserSession]
    ) -> GeneratedTest:
        """Create a single test from multiple similar sessions."""
        # Find the most common path
        # For now, use the first successful session
        best_session = None
        for session in sessions:
            if session.outcome == "conversion" or not session.errors:
                best_session = session
                break

        if not best_session:
            best_session = sessions[0]

        test = await self.convert_session(best_session)
        test.source_session_ids = [s.session_id for s in sessions]
        test.name = f"{journey_name} (from {len(sessions)} sessions)"
        test.confidence = min(0.95, 0.5 + len(sessions) * 0.05)  # Higher confidence with more sessions

        return test


class ErrorToTestConverter:
    """
    Convert production errors into regression tests.

    When an error occurs in production, we:
    1. Capture the session that led to it
    2. Generate a test that reproduces the error
    3. Add it to the test suite to prevent regression
    """

    def __init__(self):
        self.settings = get_settings()
        api_key = self.settings.anthropic_api_key
        if hasattr(api_key, 'get_secret_value'):
            api_key = api_key.get_secret_value()
        self.client = Anthropic(api_key=api_key)
        self.session_converter = SessionToTestConverter()

    async def convert_error(
        self,
        error_event: dict,
        session: UserSession | None = None
    ) -> GeneratedTest:
        """
        Convert a production error into a regression test.

        Args:
            error_event: The error event from Sentry/Datadog/etc
            session: The user session that led to the error (if available)
        """
        # If we have the session, use it
        if session:
            test = await self.session_converter.convert_session(session)
            test.name = f"Regression: {error_event.get('message', 'Unknown Error')[:50]}"
            test.priority = "critical"  # Errors that happened in prod are critical
            test.rationale = f"Regression test for production error: {error_event.get('message')}"

            # Add assertion that the error should NOT occur
            test.assertions.append({
                "type": "no_console_errors",
                "target": "console",
                "expected": f"No error matching: {error_event.get('message', '')[:50]}",
                "description": "Verify the original error does not recur"
            })

            return test

        # Without session, generate a minimal reproduction test
        return await self._generate_minimal_test(error_event)

    async def _generate_minimal_test(self, error_event: dict) -> GeneratedTest:
        """Generate a minimal test from error context."""
        prompt = f"""Generate a test to reproduce this production error.

ERROR DETAILS:
{json.dumps(error_event, indent=2)}

Create a minimal test that would trigger this error:
{{
    "name": "Descriptive test name",
    "steps": [
        {{"action": "...", "target": "...", "value": "..."}}
    ],
    "assertions": [
        {{"type": "...", "target": "...", "expected": "..."}}
    ],
    "preconditions": ["..."]
}}

Be specific about what user actions would trigger this error."""

        from src.core.model_registry import get_model_id
        response = self.client.messages.create(
            model=get_model_id("claude-sonnet-4-5"),
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.content[0].text
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0:
                test_data = json.loads(text[json_start:json_end])
                return GeneratedTest(
                    id=f"error-{error_event.get('id', 'unknown')[:8]}",
                    name=test_data.get("name", "Error Regression Test"),
                    description=f"Regression test for: {error_event.get('message', '')}",
                    source_session_ids=[],
                    priority="critical",
                    steps=test_data.get("steps", []),
                    assertions=test_data.get("assertions", []),
                    preconditions=test_data.get("preconditions", []),
                    rationale="Generated from production error",
                    confidence=0.6,
                    user_journey="Error Reproduction"
                )
        except:
            pass

        # Fallback
        return GeneratedTest(
            id=f"error-{error_event.get('id', 'unknown')[:8]}",
            name="Manual Review Required",
            description=f"Could not auto-generate test for: {error_event.get('message', '')}",
            source_session_ids=[],
            priority="high",
            steps=[],
            assertions=[],
            preconditions=[],
            rationale="Requires manual test creation",
            confidence=0.0,
            user_journey="Unknown"
        )


class RUMIntegration:
    """
    Real User Monitoring integration.

    Connect to RUM providers to get real user sessions
    and automatically generate tests from them.
    """

    async def fetch_sessions_from_fullstory(
        self,
        api_key: str,
        filters: dict = None
    ) -> list[UserSession]:
        """Fetch sessions from FullStory."""
        # Would integrate with FullStory API
        # https://developer.fullstory.com/
        return []

    async def fetch_sessions_from_logrocket(
        self,
        api_key: str,
        filters: dict = None
    ) -> list[UserSession]:
        """Fetch sessions from LogRocket."""
        # Would integrate with LogRocket API
        return []

    async def fetch_errors_from_sentry(
        self,
        api_key: str,
        project: str,
        filters: dict = None
    ) -> list[dict]:
        """Fetch errors from Sentry."""
        # Would integrate with Sentry API
        # https://docs.sentry.io/api/
        return []

    async def fetch_errors_from_datadog(
        self,
        api_key: str,
        app_key: str,
        filters: dict = None
    ) -> list[dict]:
        """Fetch errors from Datadog."""
        # Would integrate with Datadog API
        return []
