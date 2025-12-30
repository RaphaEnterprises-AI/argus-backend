"""Tests for observability hub module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.integrations.observability_hub import (
    Platform,
    RealUserSession,
    ProductionError,
    PerformanceAnomaly,
    UserJourneyPattern,
    ObservabilityProvider,
    DatadogProvider,
    SentryProvider,
    NewRelicProvider,
    FullStoryProvider,
    PostHogProvider,
    ObservabilityHub,
)


class TestPlatformEnum:
    """Tests for Platform enum."""

    def test_platform_values(self):
        """Test all platform enum values exist."""
        assert Platform.DATADOG == "datadog"
        assert Platform.NEW_RELIC == "new_relic"
        assert Platform.SENTRY == "sentry"
        assert Platform.DYNATRACE == "dynatrace"
        assert Platform.FULLSTORY == "fullstory"
        assert Platform.LOGROCKET == "logrocket"
        assert Platform.AMPLITUDE == "amplitude"
        assert Platform.MIXPANEL == "mixpanel"
        assert Platform.SEGMENT == "segment"
        assert Platform.POSTHOG == "posthog"
        assert Platform.HONEYCOMB == "honeycomb"
        assert Platform.GRAFANA == "grafana"
        assert Platform.ELASTIC_APM == "elastic_apm"

    def test_platform_is_string_enum(self):
        """Test that Platform inherits from str."""
        assert isinstance(Platform.DATADOG, str)
        assert Platform.DATADOG == "datadog"


class TestRealUserSession:
    """Tests for RealUserSession dataclass."""

    def test_create_real_user_session(self):
        """Test creating a real user session."""
        session = RealUserSession(
            session_id="sess_123",
            user_id="user_456",
            platform=Platform.DATADOG,
            started_at=datetime(2024, 1, 1, 10, 0),
            duration_ms=5000,
            page_views=[{"url": "/home"}],
            actions=[{"type": "click"}],
            errors=[{"message": "Error"}],
            performance_metrics={"lcp": 1500},
            device={"type": "desktop"},
            geo={"country": "US"},
            frustration_signals=[{"type": "rage_click"}],
            conversion_events=[{"type": "purchase"}],
        )
        assert session.session_id == "sess_123"
        assert session.user_id == "user_456"
        assert session.platform == Platform.DATADOG
        assert session.duration_ms == 5000
        assert session.replay_url is None

    def test_real_user_session_with_replay_url(self):
        """Test session with replay URL."""
        session = RealUserSession(
            session_id="sess_123",
            user_id=None,
            platform=Platform.FULLSTORY,
            started_at=datetime(2024, 1, 1),
            duration_ms=3000,
            page_views=[],
            actions=[],
            errors=[],
            performance_metrics={},
            device={},
            geo={},
            frustration_signals=[],
            conversion_events=[],
            replay_url="https://fullstory.com/replay/123",
        )
        assert session.replay_url == "https://fullstory.com/replay/123"
        assert session.user_id is None


class TestProductionError:
    """Tests for ProductionError dataclass."""

    def test_create_production_error(self):
        """Test creating a production error."""
        error = ProductionError(
            error_id="err_123",
            platform=Platform.SENTRY,
            message="NullPointerException",
            stack_trace="at line 10...",
            first_seen=datetime(2024, 1, 1),
            last_seen=datetime(2024, 1, 2),
            occurrence_count=50,
            affected_users=25,
            affected_sessions=["sess_1", "sess_2"],
            tags={"env": "production"},
            context={"path": "/api/users"},
            release="v1.2.3",
            environment="production",
            severity="error",
            status="unresolved",
            assignee="dev@example.com",
            issue_url="https://sentry.io/issues/123",
        )
        assert error.error_id == "err_123"
        assert error.platform == Platform.SENTRY
        assert error.occurrence_count == 50
        assert error.affected_users == 25
        assert error.severity == "error"
        assert error.status == "unresolved"

    def test_production_error_optional_fields(self):
        """Test production error with optional fields."""
        error = ProductionError(
            error_id="err_456",
            platform=Platform.DATADOG,
            message="Error",
            stack_trace=None,
            first_seen=datetime(2024, 1, 1),
            last_seen=datetime(2024, 1, 1),
            occurrence_count=1,
            affected_users=1,
            affected_sessions=[],
            tags={},
            context={},
            release=None,
            environment="staging",
            severity="warning",
            status="resolved",
            assignee=None,
            issue_url=None,
        )
        assert error.stack_trace is None
        assert error.release is None
        assert error.assignee is None
        assert error.issue_url is None


class TestPerformanceAnomaly:
    """Tests for PerformanceAnomaly dataclass."""

    def test_create_performance_anomaly(self):
        """Test creating a performance anomaly."""
        anomaly = PerformanceAnomaly(
            anomaly_id="anom_123",
            platform=Platform.DATADOG,
            metric="LCP",
            baseline_value=1500.0,
            current_value=3000.0,
            deviation_percent=100.0,
            affected_pages=["/home", "/products"],
            affected_users_percent=25.0,
            started_at=datetime(2024, 1, 1, 10, 0),
            detected_at=datetime(2024, 1, 1, 10, 30),
            probable_cause="High server load",
        )
        assert anomaly.anomaly_id == "anom_123"
        assert anomaly.metric == "LCP"
        assert anomaly.deviation_percent == 100.0
        assert len(anomaly.affected_pages) == 2
        assert anomaly.probable_cause == "High server load"


class TestUserJourneyPattern:
    """Tests for UserJourneyPattern dataclass."""

    def test_create_user_journey_pattern(self):
        """Test creating a user journey pattern."""
        journey = UserJourneyPattern(
            pattern_id="journey_123",
            name="Checkout Flow",
            steps=[{"page": "/cart"}, {"page": "/checkout"}],
            frequency=1000,
            conversion_rate=0.75,
            avg_duration_ms=60000,
            drop_off_points=[{"step": 2, "rate": 0.15}],
            is_critical=True,
        )
        assert journey.pattern_id == "journey_123"
        assert journey.name == "Checkout Flow"
        assert journey.frequency == 1000
        assert journey.conversion_rate == 0.75
        assert journey.is_critical is True


class TestDatadogProvider:
    """Tests for DatadogProvider class."""

    def test_init(self):
        """Test DatadogProvider initialization."""
        provider = DatadogProvider(
            api_key="dd_api_key",
            app_key="dd_app_key",
            site="datadoghq.com",
        )
        assert provider.api_key == "dd_api_key"
        assert provider.app_key == "dd_app_key"
        assert provider.base_url == "https://api.datadoghq.com/api/v2"
        assert "DD-API-KEY" in provider.headers
        assert "DD-APPLICATION-KEY" in provider.headers

    def test_init_eu_site(self):
        """Test DatadogProvider with EU site."""
        provider = DatadogProvider(
            api_key="key",
            app_key="app_key",
            site="datadoghq.eu",
        )
        assert provider.base_url == "https://api.datadoghq.eu/api/v2"

    @pytest.mark.asyncio
    async def test_get_recent_sessions_success(self):
        """Test getting recent sessions from Datadog."""
        provider = DatadogProvider("api_key", "app_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "session": {"id": "sess_1", "time_spent": 5000, "replay_url": None},
                        "usr": {"id": "user_1"},
                        "date": "2024-01-01T10:00:00Z",
                        "view": {"url_path_group": ["/home"], "largest_contentful_paint": 1500},
                        "action": [],
                        "error": [],
                        "device": {"type": "desktop"},
                        "browser": {"name": "Chrome"},
                        "geo": {"country": "US", "city": "NYC"},
                        "frustration": [],
                    }
                }
            ]
        }

        provider.http.post = AsyncMock(return_value=mock_response)

        sessions = await provider.get_recent_sessions(limit=10)

        assert len(sessions) == 1
        assert sessions[0].session_id == "sess_1"
        assert sessions[0].platform == Platform.DATADOG

    @pytest.mark.asyncio
    async def test_get_recent_sessions_failure(self):
        """Test handling API failure."""
        provider = DatadogProvider("api_key", "app_key")

        mock_response = MagicMock()
        mock_response.status_code = 500

        provider.http.post = AsyncMock(return_value=mock_response)

        sessions = await provider.get_recent_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_get_errors_success(self):
        """Test getting errors from Datadog."""
        provider = DatadogProvider("api_key", "app_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "err_1",
                    "attributes": {
                        "error": {"message": "Error message", "stack": "stack trace", "handling": "unhandled"},
                        "date": "2024-01-01T10:00:00Z",
                        "session": {"id": "sess_1"},
                        "tags": {},
                        "context": {},
                        "version": {"version": "1.0.0"},
                        "env": "production",
                    }
                }
            ]
        }

        provider.http.post = AsyncMock(return_value=mock_response)

        errors = await provider.get_errors(limit=10)

        assert len(errors) == 1
        assert errors[0].error_id == "err_1"
        assert errors[0].platform == Platform.DATADOG

    @pytest.mark.asyncio
    async def test_get_errors_failure(self):
        """Test handling error API failure."""
        provider = DatadogProvider("api_key", "app_key")

        mock_response = MagicMock()
        mock_response.status_code = 401

        provider.http.post = AsyncMock(return_value=mock_response)

        errors = await provider.get_errors()
        assert errors == []

    @pytest.mark.asyncio
    async def test_get_performance_anomalies(self):
        """Test getting performance anomalies."""
        provider = DatadogProvider("api_key", "app_key")
        anomalies = await provider.get_performance_anomalies()
        assert anomalies == []

    @pytest.mark.asyncio
    async def test_get_user_journeys(self):
        """Test getting user journeys."""
        provider = DatadogProvider("api_key", "app_key")
        journeys = await provider.get_user_journeys()
        assert journeys == []

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing provider connection."""
        provider = DatadogProvider("api_key", "app_key")
        provider.http.aclose = AsyncMock()
        await provider.close()
        provider.http.aclose.assert_called_once()


class TestSentryProvider:
    """Tests for SentryProvider class."""

    def test_init(self):
        """Test SentryProvider initialization."""
        provider = SentryProvider(
            auth_token="sentry_token",
            organization="my-org",
            project="my-project",
        )
        assert provider.api_key == "sentry_token"
        assert provider.organization == "my-org"
        assert provider.project == "my-project"
        assert provider.base_url == "https://sentry.io/api/0"
        assert "Bearer sentry_token" in provider.headers["Authorization"]

    @pytest.mark.asyncio
    async def test_get_recent_sessions(self):
        """Test getting sessions from Sentry (empty)."""
        provider = SentryProvider("token", "org", "project")
        sessions = await provider.get_recent_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_get_errors_success(self):
        """Test getting errors from Sentry."""
        provider = SentryProvider("token", "org", "project")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "issue_1",
                "title": "TypeError: Cannot read property",
                "culprit": "src/app.js",
                "firstSeen": "2024-01-01T10:00:00Z",
                "lastSeen": "2024-01-01T12:00:00Z",
                "count": 100,
                "userCount": 50,
                "firstRelease": {"version": "1.0.0"},
                "level": "error",
                "status": "unresolved",
                "assignedTo": {"name": "John"},
                "permalink": "https://sentry.io/issue/1",
            }
        ]

        provider.http.get = AsyncMock(return_value=mock_response)

        errors = await provider.get_errors(limit=10)

        assert len(errors) == 1
        assert errors[0].error_id == "issue_1"
        assert errors[0].platform == Platform.SENTRY
        assert errors[0].occurrence_count == 100
        assert errors[0].affected_users == 50

    @pytest.mark.asyncio
    async def test_get_errors_failure(self):
        """Test handling Sentry API failure."""
        provider = SentryProvider("token", "org", "project")

        mock_response = MagicMock()
        mock_response.status_code = 403

        provider.http.get = AsyncMock(return_value=mock_response)

        errors = await provider.get_errors()
        assert errors == []

    @pytest.mark.asyncio
    async def test_get_errors_with_none_values(self):
        """Test handling Sentry errors with None values."""
        provider = SentryProvider("token", "org", "project")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "issue_2",
                "title": "Error",
                "culprit": None,
                "firstSeen": "2024-01-01T10:00:00Z",
                "lastSeen": "2024-01-01T10:00:00Z",
                "count": 1,
                "userCount": 1,
                "firstRelease": None,
                "level": "warning",
                "status": "resolved",
                "assignedTo": None,
                "permalink": None,
            }
        ]

        provider.http.get = AsyncMock(return_value=mock_response)

        errors = await provider.get_errors()

        assert len(errors) == 1
        assert errors[0].release is None
        assert errors[0].assignee is None

    @pytest.mark.asyncio
    async def test_get_performance_anomalies(self):
        """Test getting performance anomalies from Sentry."""
        provider = SentryProvider("token", "org", "project")
        anomalies = await provider.get_performance_anomalies()
        assert anomalies == []

    @pytest.mark.asyncio
    async def test_get_user_journeys(self):
        """Test getting user journeys from Sentry."""
        provider = SentryProvider("token", "org", "project")
        journeys = await provider.get_user_journeys()
        assert journeys == []


class TestNewRelicProvider:
    """Tests for NewRelicProvider class."""

    def test_init(self):
        """Test NewRelicProvider initialization."""
        provider = NewRelicProvider(
            api_key="nr_api_key",
            account_id="12345",
        )
        assert provider.api_key == "nr_api_key"
        assert provider.account_id == "12345"
        assert provider.base_url == "https://api.newrelic.com/graphql"
        assert "API-Key" in provider.headers

    @pytest.mark.asyncio
    async def test_query_success(self):
        """Test NerdGraph query."""
        provider = NewRelicProvider("api_key", "12345")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"actor": {"account": {}}}}

        provider.http.post = AsyncMock(return_value=mock_response)

        result = await provider._query("{ actor { account(id: 12345) { nrql(query: \"SELECT * FROM Transaction\") { results } } } }")

        assert "data" in result

    @pytest.mark.asyncio
    async def test_query_failure(self):
        """Test NerdGraph query failure."""
        provider = NewRelicProvider("api_key", "12345")

        mock_response = MagicMock()
        mock_response.status_code = 500

        provider.http.post = AsyncMock(return_value=mock_response)

        result = await provider._query("{ actor { ... } }")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_recent_sessions(self):
        """Test getting sessions from New Relic."""
        provider = NewRelicProvider("api_key", "12345")
        provider._query = AsyncMock(return_value={})

        sessions = await provider.get_recent_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_get_errors(self):
        """Test getting errors from New Relic."""
        provider = NewRelicProvider("api_key", "12345")
        provider._query = AsyncMock(return_value={})

        errors = await provider.get_errors()
        assert errors == []

    @pytest.mark.asyncio
    async def test_get_performance_anomalies(self):
        """Test getting anomalies from New Relic."""
        provider = NewRelicProvider("api_key", "12345")
        anomalies = await provider.get_performance_anomalies()
        assert anomalies == []

    @pytest.mark.asyncio
    async def test_get_user_journeys(self):
        """Test getting user journeys from New Relic."""
        provider = NewRelicProvider("api_key", "12345")
        journeys = await provider.get_user_journeys()
        assert journeys == []


class TestFullStoryProvider:
    """Tests for FullStoryProvider class."""

    def test_init(self):
        """Test FullStoryProvider initialization."""
        provider = FullStoryProvider(api_key="fs_api_key")
        assert provider.api_key == "fs_api_key"
        assert provider.base_url == "https://api.fullstory.com"
        assert "Basic fs_api_key" in provider.headers["Authorization"]

    @pytest.mark.asyncio
    async def test_get_recent_sessions_success(self):
        """Test getting sessions from FullStory."""
        provider = FullStoryProvider("api_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sessions": [
                {
                    "sessionId": "sess_fs_1",
                    "userId": "user_1",
                    "createdTime": "2024-01-01T10:00:00",
                    "totalDuration": 5000,
                    "visitedUrls": ["/home"],
                    "events": [{"type": "click"}],
                    "errors": [],
                    "device": {"type": "mobile"},
                    "geo": {"country": "UK"},
                    "frustrationSignals": [],
                    "conversions": [],
                    "playbackUrl": "https://fullstory.com/replay/1",
                }
            ]
        }

        provider.http.post = AsyncMock(return_value=mock_response)

        sessions = await provider.get_recent_sessions(limit=10)

        assert len(sessions) == 1
        assert sessions[0].session_id == "sess_fs_1"
        assert sessions[0].platform == Platform.FULLSTORY
        assert sessions[0].replay_url == "https://fullstory.com/replay/1"

    @pytest.mark.asyncio
    async def test_get_recent_sessions_failure(self):
        """Test handling FullStory API failure."""
        provider = FullStoryProvider("api_key")

        mock_response = MagicMock()
        mock_response.status_code = 401

        provider.http.post = AsyncMock(return_value=mock_response)

        sessions = await provider.get_recent_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_get_errors(self):
        """Test getting errors from FullStory."""
        provider = FullStoryProvider("api_key")
        errors = await provider.get_errors()
        assert errors == []

    @pytest.mark.asyncio
    async def test_get_performance_anomalies(self):
        """Test getting anomalies from FullStory."""
        provider = FullStoryProvider("api_key")
        anomalies = await provider.get_performance_anomalies()
        assert anomalies == []

    @pytest.mark.asyncio
    async def test_get_user_journeys(self):
        """Test getting user journeys from FullStory."""
        provider = FullStoryProvider("api_key")
        journeys = await provider.get_user_journeys()
        assert journeys == []


class TestPostHogProvider:
    """Tests for PostHogProvider class."""

    def test_init_default_host(self):
        """Test PostHogProvider initialization with default host."""
        provider = PostHogProvider(api_key="ph_api_key")
        assert provider.api_key == "ph_api_key"
        assert provider.host == "https://app.posthog.com"
        assert "Bearer ph_api_key" in provider.headers["Authorization"]

    def test_init_custom_host(self):
        """Test PostHogProvider with custom host."""
        provider = PostHogProvider(api_key="key", host="https://posthog.mycompany.com")
        assert provider.host == "https://posthog.mycompany.com"

    @pytest.mark.asyncio
    async def test_get_recent_sessions_success(self):
        """Test getting sessions from PostHog."""
        provider = PostHogProvider("api_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "rec_1",
                    "person": {"id": "person_1"},
                    "start_time": "2024-01-01T10:00:00Z",
                    "recording_duration": 60,
                }
            ]
        }

        provider.http.get = AsyncMock(return_value=mock_response)

        sessions = await provider.get_recent_sessions(limit=10)

        assert len(sessions) == 1
        assert sessions[0].session_id == "rec_1"
        assert sessions[0].platform == Platform.POSTHOG
        assert "posthog.com/replay/rec_1" in sessions[0].replay_url

    @pytest.mark.asyncio
    async def test_get_recent_sessions_failure(self):
        """Test handling PostHog API failure."""
        provider = PostHogProvider("api_key")

        mock_response = MagicMock()
        mock_response.status_code = 500

        provider.http.get = AsyncMock(return_value=mock_response)

        sessions = await provider.get_recent_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_get_errors(self):
        """Test getting errors from PostHog."""
        provider = PostHogProvider("api_key")
        errors = await provider.get_errors()
        assert errors == []

    @pytest.mark.asyncio
    async def test_get_performance_anomalies(self):
        """Test getting anomalies from PostHog."""
        provider = PostHogProvider("api_key")
        anomalies = await provider.get_performance_anomalies()
        assert anomalies == []

    @pytest.mark.asyncio
    async def test_get_user_journeys_success(self):
        """Test getting user journeys from PostHog."""
        provider = PostHogProvider("api_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}

        provider.http.get = AsyncMock(return_value=mock_response)

        journeys = await provider.get_user_journeys()
        assert journeys == []

    @pytest.mark.asyncio
    async def test_get_user_journeys_failure(self):
        """Test handling PostHog user journeys API failure."""
        provider = PostHogProvider("api_key")

        mock_response = MagicMock()
        mock_response.status_code = 404

        provider.http.get = AsyncMock(return_value=mock_response)

        journeys = await provider.get_user_journeys()
        assert journeys == []


class TestObservabilityHub:
    """Tests for ObservabilityHub class."""

    def test_init(self):
        """Test ObservabilityHub initialization."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()
            assert hub.providers == {}

    def test_connect_datadog(self):
        """Test connecting Datadog provider."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()
            hub.connect_datadog("api_key", "app_key", "datadoghq.com")

            assert Platform.DATADOG in hub.providers
            assert isinstance(hub.providers[Platform.DATADOG], DatadogProvider)

    def test_connect_sentry(self):
        """Test connecting Sentry provider."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()
            hub.connect_sentry("auth_token", "org", "project")

            assert Platform.SENTRY in hub.providers
            assert isinstance(hub.providers[Platform.SENTRY], SentryProvider)

    def test_connect_new_relic(self):
        """Test connecting New Relic provider."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()
            hub.connect_new_relic("api_key", "12345")

            assert Platform.NEW_RELIC in hub.providers
            assert isinstance(hub.providers[Platform.NEW_RELIC], NewRelicProvider)

    def test_connect_fullstory(self):
        """Test connecting FullStory provider."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()
            hub.connect_fullstory("api_key")

            assert Platform.FULLSTORY in hub.providers
            assert isinstance(hub.providers[Platform.FULLSTORY], FullStoryProvider)

    def test_connect_posthog(self):
        """Test connecting PostHog provider."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()
            hub.connect_posthog("api_key", "https://custom.posthog.com")

            assert Platform.POSTHOG in hub.providers
            assert isinstance(hub.providers[Platform.POSTHOG], PostHogProvider)

    @pytest.mark.asyncio
    async def test_get_all_sessions(self):
        """Test getting sessions from all providers."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()

            mock_provider1 = AsyncMock()
            mock_provider1.get_recent_sessions = AsyncMock(return_value=[
                RealUserSession(
                    session_id="sess_1",
                    user_id=None,
                    platform=Platform.DATADOG,
                    started_at=datetime(2024, 1, 1, 12, 0),
                    duration_ms=1000,
                    page_views=[],
                    actions=[],
                    errors=[],
                    performance_metrics={},
                    device={},
                    geo={},
                    frustration_signals=[],
                    conversion_events=[],
                )
            ])

            mock_provider2 = AsyncMock()
            mock_provider2.get_recent_sessions = AsyncMock(return_value=[
                RealUserSession(
                    session_id="sess_2",
                    user_id=None,
                    platform=Platform.SENTRY,
                    started_at=datetime(2024, 1, 1, 10, 0),
                    duration_ms=2000,
                    page_views=[],
                    actions=[],
                    errors=[],
                    performance_metrics={},
                    device={},
                    geo={},
                    frustration_signals=[],
                    conversion_events=[],
                )
            ])

            hub.providers = {
                Platform.DATADOG: mock_provider1,
                Platform.SENTRY: mock_provider2,
            }

            sessions = await hub.get_all_sessions(limit_per_platform=10)

            assert len(sessions) == 2
            # Should be sorted by started_at descending
            assert sessions[0].session_id == "sess_1"
            assert sessions[1].session_id == "sess_2"

    @pytest.mark.asyncio
    async def test_get_all_sessions_with_exception(self):
        """Test handling provider exceptions."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()

            mock_provider = AsyncMock()
            mock_provider.get_recent_sessions = AsyncMock(side_effect=Exception("API Error"))

            hub.providers = {Platform.DATADOG: mock_provider}

            sessions = await hub.get_all_sessions()
            assert sessions == []

    @pytest.mark.asyncio
    async def test_get_all_errors(self):
        """Test getting errors from all providers."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()

            mock_provider = AsyncMock()
            mock_provider.get_errors = AsyncMock(return_value=[
                ProductionError(
                    error_id="err_1",
                    platform=Platform.SENTRY,
                    message="Error 1",
                    stack_trace=None,
                    first_seen=datetime(2024, 1, 1),
                    last_seen=datetime(2024, 1, 1),
                    occurrence_count=50,
                    affected_users=25,
                    affected_sessions=[],
                    tags={},
                    context={},
                    release=None,
                    environment="production",
                    severity="error",
                    status="unresolved",
                    assignee=None,
                    issue_url=None,
                ),
                ProductionError(
                    error_id="err_2",
                    platform=Platform.SENTRY,
                    message="Error 2",
                    stack_trace=None,
                    first_seen=datetime(2024, 1, 1),
                    last_seen=datetime(2024, 1, 1),
                    occurrence_count=100,
                    affected_users=50,
                    affected_sessions=[],
                    tags={},
                    context={},
                    release=None,
                    environment="production",
                    severity="error",
                    status="unresolved",
                    assignee=None,
                    issue_url=None,
                ),
            ])

            hub.providers = {Platform.SENTRY: mock_provider}

            errors = await hub.get_all_errors(limit_per_platform=10)

            assert len(errors) == 2
            # Should be sorted by occurrence_count descending
            assert errors[0].error_id == "err_2"
            assert errors[1].error_id == "err_1"

    @pytest.mark.asyncio
    async def test_get_all_errors_with_exception(self):
        """Test handling provider exceptions for errors."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()

            mock_provider = AsyncMock()
            mock_provider.get_errors = AsyncMock(side_effect=Exception("API Error"))

            hub.providers = {Platform.DATADOG: mock_provider}

            errors = await hub.get_all_errors()
            assert errors == []

    @pytest.mark.asyncio
    async def test_get_all_anomalies(self):
        """Test getting anomalies from all providers."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()

            mock_provider = AsyncMock()
            mock_provider.get_performance_anomalies = AsyncMock(return_value=[
                PerformanceAnomaly(
                    anomaly_id="anom_1",
                    platform=Platform.DATADOG,
                    metric="LCP",
                    baseline_value=1500,
                    current_value=3000,
                    deviation_percent=100,
                    affected_pages=["/home"],
                    affected_users_percent=25,
                    started_at=datetime(2024, 1, 1),
                    detected_at=datetime(2024, 1, 1),
                    probable_cause=None,
                )
            ])

            hub.providers = {Platform.DATADOG: mock_provider}

            anomalies = await hub.get_all_anomalies()

            assert len(anomalies) == 1
            assert anomalies[0].anomaly_id == "anom_1"

    @pytest.mark.asyncio
    async def test_get_all_anomalies_with_exception(self):
        """Test handling provider exceptions for anomalies."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()

            mock_provider = AsyncMock()
            mock_provider.get_performance_anomalies = AsyncMock(side_effect=Exception("Error"))

            hub.providers = {Platform.DATADOG: mock_provider}

            anomalies = await hub.get_all_anomalies()
            assert anomalies == []

    @pytest.mark.asyncio
    async def test_get_all_user_journeys(self):
        """Test getting user journeys from all providers."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()

            mock_provider = AsyncMock()
            mock_provider.get_user_journeys = AsyncMock(return_value=[
                UserJourneyPattern(
                    pattern_id="journey_1",
                    name="Checkout",
                    steps=[],
                    frequency=1000,
                    conversion_rate=0.75,
                    avg_duration_ms=60000,
                    drop_off_points=[],
                    is_critical=True,
                )
            ])

            hub.providers = {Platform.POSTHOG: mock_provider}

            journeys = await hub.get_all_user_journeys()

            assert len(journeys) == 1
            assert journeys[0].pattern_id == "journey_1"

    @pytest.mark.asyncio
    async def test_get_all_user_journeys_with_exception(self):
        """Test handling provider exceptions for user journeys."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()

            mock_provider = AsyncMock()
            mock_provider.get_user_journeys = AsyncMock(side_effect=Exception("Error"))

            hub.providers = {Platform.POSTHOG: mock_provider}

            journeys = await hub.get_all_user_journeys()
            assert journeys == []

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing all provider connections."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()

            mock_provider1 = AsyncMock()
            mock_provider1.close = AsyncMock()

            mock_provider2 = AsyncMock()
            mock_provider2.close = AsyncMock()

            hub.providers = {
                Platform.DATADOG: mock_provider1,
                Platform.SENTRY: mock_provider2,
            }

            await hub.close()

            mock_provider1.close.assert_called_once()
            mock_provider2.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_hub(self):
        """Test hub with no providers."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = ObservabilityHub()

            sessions = await hub.get_all_sessions()
            errors = await hub.get_all_errors()
            anomalies = await hub.get_all_anomalies()
            journeys = await hub.get_all_user_journeys()

            assert sessions == []
            assert errors == []
            assert anomalies == []
            assert journeys == []
