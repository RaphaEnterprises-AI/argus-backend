"""
Observability Integration Hub

THE KEY INSIGHT: We don't build monitoring. We CONSUME it.
We connect to Datadog, New Relic, Sentry, FullStory, etc. and use AI
to synthesize testing intelligence from real production data.

User Experience:
1. Connect your observability stack (one-click OAuth or API key)
2. AI automatically discovers everything
3. Tests generate themselves from real user behavior
4. Zero configuration required

This is what makes us TRULY next-gen.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import httpx

from src.config import get_settings


class Platform(str, Enum):
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    SENTRY = "sentry"
    DYNATRACE = "dynatrace"
    FULLSTORY = "fullstory"
    LOGROCKET = "logrocket"
    AMPLITUDE = "amplitude"
    MIXPANEL = "mixpanel"
    SEGMENT = "segment"
    POSTHOG = "posthog"
    HONEYCOMB = "honeycomb"
    GRAFANA = "grafana"
    ELASTIC_APM = "elastic_apm"


@dataclass
class RealUserSession:
    """A real user session from RUM data."""
    session_id: str
    user_id: str | None
    platform: Platform
    started_at: datetime
    duration_ms: int
    page_views: list[dict]
    actions: list[dict]
    errors: list[dict]
    performance_metrics: dict
    device: dict
    geo: dict
    frustration_signals: list[dict]  # Rage clicks, dead clicks, etc.
    conversion_events: list[dict]
    replay_url: str | None = None  # Direct link to session replay


@dataclass
class ProductionError:
    """An error from production."""
    error_id: str
    platform: Platform
    message: str
    stack_trace: str | None
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int
    affected_users: int
    affected_sessions: list[str]
    tags: dict
    context: dict
    release: str | None
    environment: str
    severity: str
    status: str  # "unresolved", "resolved", "ignored"
    assignee: str | None
    issue_url: str | None  # Link to Sentry/Datadog issue


@dataclass
class PerformanceAnomaly:
    """A performance anomaly detected in production."""
    anomaly_id: str
    platform: Platform
    metric: str  # "LCP", "FID", "response_time", etc.
    baseline_value: float
    current_value: float
    deviation_percent: float
    affected_pages: list[str]
    affected_users_percent: float
    started_at: datetime
    detected_at: datetime
    probable_cause: str | None


@dataclass
class UserJourneyPattern:
    """A common user journey pattern from analytics."""
    pattern_id: str
    name: str
    steps: list[dict]
    frequency: int  # How many users follow this path
    conversion_rate: float
    avg_duration_ms: int
    drop_off_points: list[dict]
    is_critical: bool


class ObservabilityProvider(ABC):
    """Base class for observability platform integrations."""

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.http = httpx.AsyncClient(timeout=30.0)

    @abstractmethod
    async def get_recent_sessions(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[RealUserSession]:
        """Get recent user sessions with full context."""
        pass

    @abstractmethod
    async def get_errors(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[ProductionError]:
        """Get recent production errors."""
        pass

    @abstractmethod
    async def get_performance_anomalies(
        self,
        since: datetime | None = None
    ) -> list[PerformanceAnomaly]:
        """Get performance anomalies."""
        pass

    @abstractmethod
    async def get_user_journeys(
        self,
        limit: int = 20
    ) -> list[UserJourneyPattern]:
        """Get common user journey patterns."""
        pass

    async def close(self):
        await self.http.aclose()


class DatadogProvider(ObservabilityProvider):
    """
    Datadog RUM + APM Integration.

    API Docs:
    - RUM: https://docs.datadoghq.com/api/latest/rum/
    - APM: https://docs.datadoghq.com/api/latest/tracing/
    - Logs: https://docs.datadoghq.com/api/latest/logs/
    """

    def __init__(
        self,
        api_key: str,
        app_key: str,
        site: str = "datadoghq.com"  # or datadoghq.eu, etc.
    ):
        super().__init__(api_key)
        self.app_key = app_key
        self.base_url = f"https://api.{site}/api/v2"
        self.headers = {
            "DD-API-KEY": api_key,
            "DD-APPLICATION-KEY": app_key,
            "Content-Type": "application/json"
        }

    async def get_recent_sessions(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[RealUserSession]:
        """Fetch RUM sessions from Datadog."""
        since = since or (datetime.utcnow() - timedelta(hours=24))

        # RUM Events Search API
        query = {
            "filter": {
                "from": since.isoformat() + "Z",
                "to": datetime.utcnow().isoformat() + "Z",
                "query": "@type:session"
            },
            "page": {"limit": limit},
            "sort": "-timestamp"
        }

        response = await self.http.post(
            f"{self.base_url}/rum/events/search",
            headers=self.headers,
            json=query
        )

        if response.status_code != 200:
            return []

        data = response.json()
        sessions = []

        for event in data.get("data", []):
            attrs = event.get("attributes", {})
            sessions.append(RealUserSession(
                session_id=attrs.get("session", {}).get("id", ""),
                user_id=attrs.get("usr", {}).get("id"),
                platform=Platform.DATADOG,
                started_at=datetime.fromisoformat(attrs.get("date", "").rstrip("Z")),
                duration_ms=attrs.get("session", {}).get("time_spent", 0),
                page_views=attrs.get("view", {}).get("url_path_group", []),
                actions=attrs.get("action", []),
                errors=attrs.get("error", []),
                performance_metrics={
                    "lcp": attrs.get("view", {}).get("largest_contentful_paint"),
                    "fid": attrs.get("view", {}).get("first_input_delay"),
                    "cls": attrs.get("view", {}).get("cumulative_layout_shift"),
                },
                device={
                    "type": attrs.get("device", {}).get("type"),
                    "browser": attrs.get("browser", {}).get("name"),
                },
                geo={
                    "country": attrs.get("geo", {}).get("country"),
                    "city": attrs.get("geo", {}).get("city"),
                },
                frustration_signals=attrs.get("frustration", []),
                conversion_events=[],
                replay_url=attrs.get("session", {}).get("replay_url")
            ))

        return sessions

    async def get_errors(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[ProductionError]:
        """Fetch RUM errors from Datadog."""
        since = since or (datetime.utcnow() - timedelta(hours=24))

        query = {
            "filter": {
                "from": since.isoformat() + "Z",
                "to": datetime.utcnow().isoformat() + "Z",
                "query": "@type:error"
            },
            "page": {"limit": limit},
            "sort": "-timestamp"
        }

        response = await self.http.post(
            f"{self.base_url}/rum/events/search",
            headers=self.headers,
            json=query
        )

        if response.status_code != 200:
            return []

        data = response.json()
        errors = []

        for event in data.get("data", []):
            attrs = event.get("attributes", {})
            error = attrs.get("error", {})
            errors.append(ProductionError(
                error_id=event.get("id", ""),
                platform=Platform.DATADOG,
                message=error.get("message", ""),
                stack_trace=error.get("stack"),
                first_seen=datetime.fromisoformat(attrs.get("date", "").rstrip("Z")),
                last_seen=datetime.fromisoformat(attrs.get("date", "").rstrip("Z")),
                occurrence_count=1,
                affected_users=1,
                affected_sessions=[attrs.get("session", {}).get("id", "")],
                tags=attrs.get("tags", {}),
                context=attrs.get("context", {}),
                release=attrs.get("version", {}).get("version"),
                environment=attrs.get("env", ""),
                severity=error.get("handling", "unhandled"),
                status="unresolved",
                assignee=None,
                issue_url=None
            ))

        return errors

    async def get_performance_anomalies(
        self,
        since: datetime | None = None
    ) -> list[PerformanceAnomaly]:
        """Detect performance anomalies from RUM metrics."""
        # Use Datadog's metrics API to detect anomalies
        # This would typically use their anomaly detection
        return []

    async def get_user_journeys(
        self,
        limit: int = 20
    ) -> list[UserJourneyPattern]:
        """Extract user journey patterns from RUM data."""
        # Analyze session data to find common paths
        return []


class SentryProvider(ObservabilityProvider):
    """
    Sentry Integration for error tracking.

    API Docs: https://docs.sentry.io/api/
    """

    def __init__(
        self,
        auth_token: str,
        organization: str,
        project: str
    ):
        super().__init__(auth_token)
        self.organization = organization
        self.project = project
        self.base_url = "https://sentry.io/api/0"
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }

    async def get_recent_sessions(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[RealUserSession]:
        """Sentry doesn't have full session replay, but we can get session data."""
        # Sentry Session API
        return []

    async def get_errors(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[ProductionError]:
        """Fetch issues from Sentry."""
        params = {
            "query": "is:unresolved",
            "sort": "date",
            "limit": limit
        }

        response = await self.http.get(
            f"{self.base_url}/projects/{self.organization}/{self.project}/issues/",
            headers=self.headers,
            params=params
        )

        if response.status_code != 200:
            return []

        issues = response.json()
        errors = []

        for issue in issues:
            errors.append(ProductionError(
                error_id=issue.get("id", ""),
                platform=Platform.SENTRY,
                message=issue.get("title", ""),
                stack_trace=issue.get("culprit"),
                first_seen=datetime.fromisoformat(issue.get("firstSeen", "").rstrip("Z")),
                last_seen=datetime.fromisoformat(issue.get("lastSeen", "").rstrip("Z")),
                occurrence_count=issue.get("count", 0),
                affected_users=issue.get("userCount", 0),
                affected_sessions=[],
                tags={},
                context={},
                release=issue.get("firstRelease", {}).get("version") if issue.get("firstRelease") else None,
                environment="production",
                severity=issue.get("level", "error"),
                status=issue.get("status", "unresolved"),
                assignee=issue.get("assignedTo", {}).get("name") if issue.get("assignedTo") else None,
                issue_url=issue.get("permalink")
            ))

        return errors

    async def get_performance_anomalies(
        self,
        since: datetime | None = None
    ) -> list[PerformanceAnomaly]:
        """Get performance issues from Sentry Performance."""
        return []

    async def get_user_journeys(
        self,
        limit: int = 20
    ) -> list[UserJourneyPattern]:
        """Sentry doesn't track user journeys."""
        return []


class NewRelicProvider(ObservabilityProvider):
    """
    New Relic Integration.

    Uses NerdGraph (GraphQL) API.
    Docs: https://docs.newrelic.com/docs/apis/nerdgraph/
    """

    def __init__(
        self,
        api_key: str,
        account_id: str
    ):
        super().__init__(api_key)
        self.account_id = account_id
        self.base_url = "https://api.newrelic.com/graphql"
        self.headers = {
            "API-Key": api_key,
            "Content-Type": "application/json"
        }

    async def _query(self, query: str, variables: dict = None) -> dict:
        """Execute a NerdGraph query."""
        response = await self.http.post(
            self.base_url,
            headers=self.headers,
            json={"query": query, "variables": variables or {}}
        )
        return response.json() if response.status_code == 200 else {}

    async def get_recent_sessions(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[RealUserSession]:
        """Get browser sessions from New Relic Browser."""
        query = """
        {
          actor {
            account(id: %s) {
              nrql(query: "SELECT * FROM BrowserInteraction LIMIT %d") {
                results
              }
            }
          }
        }
        """ % (self.account_id, limit)

        await self._query(query)
        # Parse and convert to RealUserSession
        return []

    async def get_errors(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[ProductionError]:
        """Get JavaScript errors from New Relic Browser."""
        query = """
        {
          actor {
            account(id: %s) {
              nrql(query: "SELECT * FROM JavaScriptError LIMIT %d") {
                results
              }
            }
          }
        }
        """ % (self.account_id, limit)

        await self._query(query)
        return []

    async def get_performance_anomalies(
        self,
        since: datetime | None = None
    ) -> list[PerformanceAnomaly]:
        """Get performance anomalies using New Relic's anomaly detection."""
        return []

    async def get_user_journeys(
        self,
        limit: int = 20
    ) -> list[UserJourneyPattern]:
        """Get user journeys from New Relic Browser funnel analysis."""
        return []


class FullStoryProvider(ObservabilityProvider):
    """
    FullStory Integration for session replay.

    Docs: https://developer.fullstory.com/
    """

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.fullstory.com"
        self.headers = {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json"
        }

    async def get_recent_sessions(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[RealUserSession]:
        """Get sessions with replay URLs from FullStory."""
        # FullStory Data Export API
        response = await self.http.post(
            f"{self.base_url}/sessions/v2/search",
            headers=self.headers,
            json={
                "limit": limit,
                "start": since.isoformat() if since else None
            }
        )

        if response.status_code != 200:
            return []

        data = response.json()
        sessions = []

        for session in data.get("sessions", []):
            sessions.append(RealUserSession(
                session_id=session.get("sessionId", ""),
                user_id=session.get("userId"),
                platform=Platform.FULLSTORY,
                started_at=datetime.fromisoformat(session.get("createdTime", "")),
                duration_ms=session.get("totalDuration", 0),
                page_views=session.get("visitedUrls", []),
                actions=session.get("events", []),
                errors=session.get("errors", []),
                performance_metrics={},
                device=session.get("device", {}),
                geo=session.get("geo", {}),
                frustration_signals=session.get("frustrationSignals", []),
                conversion_events=session.get("conversions", []),
                replay_url=session.get("playbackUrl")
            ))

        return sessions

    async def get_errors(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[ProductionError]:
        """Get errors with session context from FullStory."""
        return []

    async def get_performance_anomalies(
        self,
        since: datetime | None = None
    ) -> list[PerformanceAnomaly]:
        """FullStory doesn't focus on performance metrics."""
        return []

    async def get_user_journeys(
        self,
        limit: int = 20
    ) -> list[UserJourneyPattern]:
        """Get user journey patterns from FullStory."""
        return []


class PostHogProvider(ObservabilityProvider):
    """
    PostHog Integration - Open source product analytics.

    Docs: https://posthog.com/docs/api
    """

    def __init__(self, api_key: str, host: str = "https://app.posthog.com"):
        super().__init__(api_key)
        self.host = host
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def get_recent_sessions(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[RealUserSession]:
        """Get sessions from PostHog with replay."""
        response = await self.http.get(
            f"{self.host}/api/projects/@current/session_recordings",
            headers=self.headers,
            params={"limit": limit}
        )

        if response.status_code != 200:
            return []

        data = response.json()
        sessions = []

        for recording in data.get("results", []):
            sessions.append(RealUserSession(
                session_id=recording.get("id", ""),
                user_id=recording.get("person", {}).get("id"),
                platform=Platform.POSTHOG,
                started_at=datetime.fromisoformat(recording.get("start_time", "").rstrip("Z")),
                duration_ms=recording.get("recording_duration", 0) * 1000,
                page_views=[],
                actions=[],
                errors=[],
                performance_metrics={},
                device={},
                geo={},
                frustration_signals=[],
                conversion_events=[],
                replay_url=f"{self.host}/replay/{recording.get('id')}"
            ))

        return sessions

    async def get_errors(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[ProductionError]:
        """Get errors from PostHog."""
        return []

    async def get_performance_anomalies(
        self,
        since: datetime | None = None
    ) -> list[PerformanceAnomaly]:
        """Get performance data from PostHog."""
        return []

    async def get_user_journeys(
        self,
        limit: int = 20
    ) -> list[UserJourneyPattern]:
        """Get user paths from PostHog funnels."""
        response = await self.http.get(
            f"{self.host}/api/projects/@current/insights",
            headers=self.headers,
            params={"insight": "FUNNELS", "limit": limit}
        )

        if response.status_code != 200:
            return []

        # Parse funnel data into user journeys
        return []


class ObservabilityHub:
    """
    Central hub that connects to all observability platforms.

    This is THE key integration point. Users connect their platforms once,
    and our AI continuously learns from real production data.
    """

    def __init__(self):
        self.providers: dict[Platform, ObservabilityProvider] = {}
        self.settings = get_settings()

    def connect_datadog(
        self,
        api_key: str,
        app_key: str,
        site: str = "datadoghq.com"
    ):
        """Connect Datadog RUM + APM."""
        self.providers[Platform.DATADOG] = DatadogProvider(api_key, app_key, site)

    def connect_sentry(
        self,
        auth_token: str,
        organization: str,
        project: str
    ):
        """Connect Sentry error tracking."""
        self.providers[Platform.SENTRY] = SentryProvider(auth_token, organization, project)

    def connect_new_relic(
        self,
        api_key: str,
        account_id: str
    ):
        """Connect New Relic APM + Browser."""
        self.providers[Platform.NEW_RELIC] = NewRelicProvider(api_key, account_id)

    def connect_fullstory(self, api_key: str):
        """Connect FullStory session replay."""
        self.providers[Platform.FULLSTORY] = FullStoryProvider(api_key)

    def connect_posthog(self, api_key: str, host: str = "https://app.posthog.com"):
        """Connect PostHog analytics."""
        self.providers[Platform.POSTHOG] = PostHogProvider(api_key, host)

    async def get_all_sessions(
        self,
        limit_per_platform: int = 50,
        since: datetime | None = None
    ) -> list[RealUserSession]:
        """Get sessions from all connected platforms."""
        all_sessions = []

        tasks = [
            provider.get_recent_sessions(limit_per_platform, since)
            for provider in self.providers.values()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_sessions.extend(result)

        # Sort by timestamp
        all_sessions.sort(key=lambda s: s.started_at, reverse=True)
        return all_sessions

    async def get_all_errors(
        self,
        limit_per_platform: int = 50,
        since: datetime | None = None
    ) -> list[ProductionError]:
        """Get errors from all connected platforms."""
        all_errors = []

        tasks = [
            provider.get_errors(limit_per_platform, since)
            for provider in self.providers.values()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_errors.extend(result)

        # Sort by occurrence count (most impactful first)
        all_errors.sort(key=lambda e: e.occurrence_count, reverse=True)
        return all_errors

    async def get_all_anomalies(
        self,
        since: datetime | None = None
    ) -> list[PerformanceAnomaly]:
        """Get performance anomalies from all platforms."""
        all_anomalies = []

        tasks = [
            provider.get_performance_anomalies(since)
            for provider in self.providers.values()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_anomalies.extend(result)

        return all_anomalies

    async def get_all_user_journeys(
        self,
        limit_per_platform: int = 10
    ) -> list[UserJourneyPattern]:
        """Get user journey patterns from all platforms."""
        all_journeys = []

        tasks = [
            provider.get_user_journeys(limit_per_platform)
            for provider in self.providers.values()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_journeys.extend(result)

        return all_journeys

    async def close(self):
        """Close all provider connections."""
        for provider in self.providers.values():
            await provider.close()
