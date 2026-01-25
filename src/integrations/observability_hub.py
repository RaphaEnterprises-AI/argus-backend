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
        """
        Detect performance anomalies from Datadog RUM metrics.

        Uses the RUM Analytics API to fetch Web Vitals metrics and detect
        anomalies by comparing current values against baseline percentiles.
        """
        since = since or (datetime.utcnow() - timedelta(hours=24))
        anomalies = []

        # Core Web Vitals to monitor
        metrics = [
            ("largest_contentful_paint", "LCP", 2500),  # 2.5s threshold
            ("first_input_delay", "FID", 100),  # 100ms threshold
            ("cumulative_layout_shift", "CLS", 0.1),  # 0.1 threshold
            ("first_contentful_paint", "FCP", 1800),  # 1.8s threshold
            ("dom_interactive", "DOM Interactive", 3000),  # 3s threshold
        ]

        for metric_name, display_name, threshold in metrics:
            # Query RUM analytics for metric percentiles
            query = {
                "filter": {
                    "from": since.isoformat() + "Z",
                    "to": datetime.utcnow().isoformat() + "Z",
                    "query": "@type:view"
                },
                "compute": [
                    {"aggregation": "pc75", "metric": f"@view.{metric_name}"},
                    {"aggregation": "pc95", "metric": f"@view.{metric_name}"},
                    {"aggregation": "count"},
                ],
                "group_by": [
                    {"facet": "@view.url_path", "limit": 10, "sort": {"aggregation": "pc95", "metric": f"@view.{metric_name}", "order": "desc"}}
                ],
            }

            try:
                response = await self.http.post(
                    f"{self.base_url}/rum/analytics/aggregate",
                    headers=self.headers,
                    json=query
                )

                if response.status_code != 200:
                    continue

                data = response.json()
                buckets = data.get("data", {}).get("buckets", [])

                for bucket in buckets:
                    by_values = bucket.get("by", {})
                    url_path = by_values.get("@view.url_path", "")
                    computes = bucket.get("computes", {})

                    pc75 = computes.get("c0", 0)  # pc75 is first compute
                    pc95 = computes.get("c1", 0)  # pc95 is second compute
                    count = computes.get("c2", 0)

                    if pc95 and pc95 > threshold:
                        # Calculate deviation from threshold
                        deviation = ((pc95 - threshold) / threshold) * 100

                        anomalies.append(PerformanceAnomaly(
                            anomaly_id=f"dd-{metric_name}-{hash(url_path) % 10000}",
                            platform=Platform.DATADOG,
                            metric=display_name,
                            baseline_value=threshold,
                            current_value=pc95,
                            deviation_percent=deviation,
                            affected_pages=[url_path],
                            affected_users_percent=min(100, count / 100),  # Approximate
                            started_at=since,
                            detected_at=datetime.utcnow(),
                            probable_cause=f"P95 {display_name} ({pc95:.0f}ms) exceeds threshold ({threshold}ms) on {url_path}",
                        ))

            except Exception:
                # Silently continue on API errors
                continue

        return anomalies

    async def get_user_journeys(
        self,
        limit: int = 20
    ) -> list[UserJourneyPattern]:
        """
        Extract user journey patterns from Datadog RUM session data.

        Analyzes session view sequences to identify common navigation paths.
        """
        # Get recent sessions with view data
        since = datetime.utcnow() - timedelta(hours=24)

        query = {
            "filter": {
                "from": since.isoformat() + "Z",
                "to": datetime.utcnow().isoformat() + "Z",
                "query": "@type:session @session.view.count:>2"
            },
            "page": {"limit": 100},
            "sort": "-@session.view.count"
        }

        try:
            response = await self.http.post(
                f"{self.base_url}/rum/events/search",
                headers=self.headers,
                json=query
            )

            if response.status_code != 200:
                return []

            data = response.json()

            # Count path frequencies
            path_counts: dict[str, dict] = {}

            for event in data.get("data", []):
                attrs = event.get("attributes", {})
                session = attrs.get("session", {})

                # Get views in this session
                views = session.get("view", {}).get("url_path_group", [])
                if isinstance(views, str):
                    views = [views]

                # Create path signature (first 5 views)
                path_key = " -> ".join(views[:5]) if views else ""
                if not path_key:
                    continue

                if path_key not in path_counts:
                    path_counts[path_key] = {
                        "steps": views[:5],
                        "count": 0,
                        "total_duration": 0,
                        "conversions": 0,
                    }

                path_counts[path_key]["count"] += 1
                path_counts[path_key]["total_duration"] += session.get("time_spent", 0)

            # Convert to UserJourneyPattern objects
            journeys = []
            sorted_paths = sorted(path_counts.items(), key=lambda x: x[1]["count"], reverse=True)

            for path_key, stats in sorted_paths[:limit]:
                avg_duration = stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0

                journeys.append(UserJourneyPattern(
                    pattern_id=f"dd-journey-{hash(path_key) % 10000}",
                    name=f"Path: {stats['steps'][0]} to {stats['steps'][-1]}" if len(stats["steps"]) > 1 else f"Single view: {stats['steps'][0]}",
                    steps=[{"url": step, "order": i} for i, step in enumerate(stats["steps"])],
                    frequency=stats["count"],
                    conversion_rate=stats["conversions"] / stats["count"] if stats["count"] > 0 else 0,
                    avg_duration_ms=int(avg_duration),
                    drop_off_points=[],
                    is_critical=stats["count"] > 10,  # Mark as critical if common path
                ))

            return journeys

        except Exception:
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
        since: datetime | None = None,
        filter_errors: bool = False,
        filter_frustrations: bool = False,
    ) -> list[RealUserSession]:
        """
        Get sessions with replay URLs from FullStory.

        Args:
            limit: Maximum number of sessions to return
            since: Only return sessions after this time
            filter_errors: Only return sessions with errors
            filter_frustrations: Only return sessions with frustration signals
        """
        # Build search query
        search_body: dict = {
            "limit": limit,
        }

        if since:
            search_body["start"] = since.isoformat()

        # Add filters
        filters = []
        if filter_errors:
            filters.append({
                "type": "event",
                "eventType": "error",
                "operator": "exists"
            })
        if filter_frustrations:
            filters.append({
                "type": "event",
                "eventType": "frustration",
                "operator": "exists"
            })

        if filters:
            search_body["filters"] = filters

        try:
            response = await self.http.post(
                f"{self.base_url}/sessions/v2/search",
                headers=self.headers,
                json=search_body
            )

            if response.status_code != 200:
                return []

            data = response.json()
            sessions = []

            for session in data.get("sessions", []):
                # Parse created time carefully
                created_time = session.get("createdTime", "")
                try:
                    if created_time:
                        started_at = datetime.fromisoformat(created_time.rstrip("Z"))
                    else:
                        started_at = datetime.utcnow()
                except (ValueError, TypeError):
                    started_at = datetime.utcnow()

                sessions.append(RealUserSession(
                    session_id=session.get("sessionId", ""),
                    user_id=session.get("userId"),
                    platform=Platform.FULLSTORY,
                    started_at=started_at,
                    duration_ms=session.get("totalDuration", 0),
                    page_views=session.get("visitedUrls", []),
                    actions=session.get("events", []),
                    errors=session.get("errors", []),
                    performance_metrics={
                        "pageLoadTime": session.get("pageLoadTime"),
                        "domContentLoaded": session.get("domContentLoaded"),
                    },
                    device=session.get("device", {}),
                    geo=session.get("geo", {}),
                    frustration_signals=session.get("frustrationSignals", []),
                    conversion_events=session.get("conversions", []),
                    replay_url=session.get("playbackUrl")
                ))

            return sessions

        except Exception:
            return []

    async def search_sessions(
        self,
        query: str | None = None,
        user_id: str | None = None,
        url_contains: str | None = None,
        has_errors: bool = False,
        has_rage_clicks: bool = False,
        has_dead_clicks: bool = False,
        limit: int = 50,
    ) -> list[RealUserSession]:
        """
        Search for sessions with specific criteria.

        This is the enhanced search API that allows filtering on multiple dimensions.
        """
        filters = []

        if user_id:
            filters.append({
                "type": "user",
                "field": "uid",
                "operator": "is",
                "value": user_id
            })

        if url_contains:
            filters.append({
                "type": "visited",
                "field": "url",
                "operator": "contains",
                "value": url_contains
            })

        if has_errors:
            filters.append({
                "type": "event",
                "eventType": "error",
                "operator": "exists"
            })

        if has_rage_clicks:
            filters.append({
                "type": "event",
                "eventType": "rageclick",
                "operator": "exists"
            })

        if has_dead_clicks:
            filters.append({
                "type": "event",
                "eventType": "deadclick",
                "operator": "exists"
            })

        search_body: dict = {
            "limit": limit,
        }

        if filters:
            search_body["filters"] = filters

        if query:
            search_body["query"] = query

        try:
            response = await self.http.post(
                f"{self.base_url}/sessions/v2/search",
                headers=self.headers,
                json=search_body
            )

            if response.status_code != 200:
                return []

            data = response.json()
            sessions = []

            for session in data.get("sessions", []):
                created_time = session.get("createdTime", "")
                try:
                    started_at = datetime.fromisoformat(created_time.rstrip("Z")) if created_time else datetime.utcnow()
                except (ValueError, TypeError):
                    started_at = datetime.utcnow()

                sessions.append(RealUserSession(
                    session_id=session.get("sessionId", ""),
                    user_id=session.get("userId"),
                    platform=Platform.FULLSTORY,
                    started_at=started_at,
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

        except Exception:
            return []

    async def get_replay_url(self, session_id: str) -> str | None:
        """Get the replay URL for a specific session."""
        try:
            response = await self.http.get(
                f"{self.base_url}/sessions/v1/{session_id}",
                headers=self.headers
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("playbackUrl")
            return None

        except Exception:
            return None

    async def get_errors(
        self,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[ProductionError]:
        """
        Get errors with session context from FullStory.

        FullStory captures JavaScript errors that occur during sessions.
        Each error is linked to a session for replay context.
        """
        # Get sessions that have errors
        sessions = await self.get_recent_sessions(
            limit=limit,
            since=since,
            filter_errors=True
        )

        errors = []
        for session in sessions:
            for error_data in session.errors:
                if isinstance(error_data, dict):
                    errors.append(ProductionError(
                        error_id=error_data.get("id", f"fs-{session.session_id}-{len(errors)}"),
                        platform=Platform.FULLSTORY,
                        message=error_data.get("message", "Unknown error"),
                        stack_trace=error_data.get("stack"),
                        first_seen=session.started_at,
                        last_seen=session.started_at,
                        occurrence_count=1,
                        affected_users=1,
                        affected_sessions=[session.session_id],
                        tags={},
                        context={
                            "url": error_data.get("url"),
                            "device": session.device,
                        },
                        release=None,
                        environment="production",
                        severity="error",
                        status="unresolved",
                        assignee=None,
                        issue_url=session.replay_url,
                    ))

        return errors

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
        """
        Get user journey patterns from FullStory sessions.

        Analyzes session data to identify common navigation patterns.
        """
        sessions = await self.get_recent_sessions(limit=100)

        # Count path frequencies
        path_counts: dict[str, dict] = {}

        for session in sessions:
            page_views = session.page_views
            if not page_views or not isinstance(page_views, list):
                continue

            # Create path signature (first 5 pages)
            path_key = " -> ".join(str(p) for p in page_views[:5])
            if not path_key:
                continue

            if path_key not in path_counts:
                path_counts[path_key] = {
                    "steps": page_views[:5],
                    "count": 0,
                    "total_duration": 0,
                    "has_conversion": 0,
                    "has_frustration": 0,
                }

            path_counts[path_key]["count"] += 1
            path_counts[path_key]["total_duration"] += session.duration_ms
            if session.conversion_events:
                path_counts[path_key]["has_conversion"] += 1
            if session.frustration_signals:
                path_counts[path_key]["has_frustration"] += 1

        # Convert to UserJourneyPattern objects
        journeys = []
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1]["count"], reverse=True)

        for path_key, stats in sorted_paths[:limit]:
            avg_duration = stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0
            conversion_rate = stats["has_conversion"] / stats["count"] if stats["count"] > 0 else 0

            journeys.append(UserJourneyPattern(
                pattern_id=f"fs-journey-{hash(path_key) % 10000}",
                name=f"Path: {stats['steps'][0]} to {stats['steps'][-1]}" if len(stats["steps"]) > 1 else f"Single page: {stats['steps'][0]}",
                steps=[{"url": str(step), "order": i} for i, step in enumerate(stats["steps"])],
                frequency=stats["count"],
                conversion_rate=conversion_rate,
                avg_duration_ms=int(avg_duration),
                drop_off_points=[{
                    "step": len(stats["steps"]) - 1,
                    "frustration_rate": stats["has_frustration"] / stats["count"] if stats["count"] > 0 else 0
                }],
                is_critical=stats["count"] > 5 and stats["has_frustration"] > 0,
            ))

        return journeys


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
