"""Tests for AI synthesis module."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.integrations.ai_synthesis import (
    ActionPriority,
    AISynthesizer,
    CoverageGap,
    ErrorInsight,
    FailurePrediction,
    InsightType,
    SynthesisReport,
    TestSuggestion,
    create_ai_synthesizer,
)
from src.integrations.observability_hub import (
    ObservabilityHub,
    PerformanceAnomaly,
    Platform,
    ProductionError,
    RealUserSession,
    UserJourneyPattern,
)


class TestInsightTypeEnum:
    """Tests for InsightType enum."""

    def test_insight_type_values(self):
        """Test all insight type enum values."""
        assert InsightType.TEST_SUGGESTION == "test_suggestion"
        assert InsightType.ERROR_PRIORITY == "error_priority"
        assert InsightType.FAILURE_PREDICTION == "failure_prediction"
        assert InsightType.COVERAGE_GAP == "coverage_gap"
        assert InsightType.USER_PATTERN == "user_pattern"
        assert InsightType.PERFORMANCE_ALERT == "performance_alert"
        assert InsightType.FLAKY_DETECTION == "flaky_detection"

    def test_insight_type_is_string_enum(self):
        """Test that InsightType inherits from str."""
        assert isinstance(InsightType.TEST_SUGGESTION, str)


class TestActionPriorityEnum:
    """Tests for ActionPriority enum."""

    def test_action_priority_values(self):
        """Test all action priority enum values."""
        assert ActionPriority.CRITICAL == "critical"
        assert ActionPriority.HIGH == "high"
        assert ActionPriority.MEDIUM == "medium"
        assert ActionPriority.LOW == "low"
        assert ActionPriority.INFO == "info"

    def test_action_priority_is_string_enum(self):
        """Test that ActionPriority inherits from str."""
        assert isinstance(ActionPriority.CRITICAL, str)


class TestTestSuggestion:
    """Tests for TestSuggestion dataclass."""

    def test_create_test_suggestion(self):
        """Test creating a test suggestion."""
        suggestion = TestSuggestion(
            id="ts_123",
            name="Login Test",
            description="Test user login flow",
            source="session_replay",
            source_platform=Platform.DATADOG,
            source_id="sess_123",
            priority=ActionPriority.HIGH,
            confidence=0.85,
            steps=[{"action": "navigate", "target": "/login"}],
            expected_outcomes=["User is logged in"],
            tags=["auth", "e2e"],
            estimated_coverage=0.15,
        )
        assert suggestion.id == "ts_123"
        assert suggestion.name == "Login Test"
        assert suggestion.priority == ActionPriority.HIGH
        assert suggestion.confidence == 0.85
        assert len(suggestion.steps) == 1
        assert len(suggestion.tags) == 2


class TestErrorInsight:
    """Tests for ErrorInsight dataclass."""

    def test_create_error_insight(self):
        """Test creating an error insight."""
        error = ProductionError(
            error_id="err_123",
            platform=Platform.SENTRY,
            message="NullPointerException",
            stack_trace="at line 10...",
            first_seen=datetime(2024, 1, 1),
            last_seen=datetime(2024, 1, 2),
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
            issue_url="https://sentry.io/issue/123",
        )

        insight = ErrorInsight(
            error=error,
            priority=ActionPriority.CRITICAL,
            impact_score=85.0,
            root_cause_hypothesis="Null reference error",
            suggested_test=None,
            related_errors=["err_456", "err_789"],
            affected_user_journeys=["checkout"],
            recommended_actions=["Fix immediately"],
        )
        assert insight.error.error_id == "err_123"
        assert insight.priority == ActionPriority.CRITICAL
        assert insight.impact_score == 85.0
        assert insight.suggested_test is None
        assert len(insight.related_errors) == 2


class TestFailurePrediction:
    """Tests for FailurePrediction dataclass."""

    def test_create_failure_prediction(self):
        """Test creating a failure prediction."""
        prediction = FailurePrediction(
            id="fp_123",
            prediction_type="error_rate_increase",
            confidence=0.8,
            affected_area="API Endpoints",
            description="Error rate increasing",
            evidence=[{"type": "trend", "data": {}}],
            recommended_actions=["Monitor closely"],
            predicted_timeframe="Next 2 hours",
            prevention_tests=[],
        )
        assert prediction.id == "fp_123"
        assert prediction.prediction_type == "error_rate_increase"
        assert prediction.confidence == 0.8
        assert len(prediction.evidence) == 1


class TestCoverageGap:
    """Tests for CoverageGap dataclass."""

    def test_create_coverage_gap(self):
        """Test creating a coverage gap."""
        gap = CoverageGap(
            id="cg_123",
            area="/checkout",
            description="High-traffic page needs coverage",
            user_traffic_percent=35.0,
            current_coverage_percent=0.0,
            priority=ActionPriority.HIGH,
            suggested_tests=[],
        )
        assert gap.id == "cg_123"
        assert gap.area == "/checkout"
        assert gap.user_traffic_percent == 35.0
        assert gap.priority == ActionPriority.HIGH


class TestSynthesisReport:
    """Tests for SynthesisReport dataclass."""

    def test_create_synthesis_report(self):
        """Test creating a synthesis report."""
        report = SynthesisReport(
            generated_at=datetime(2024, 1, 1),
            platforms_analyzed=[Platform.DATADOG, Platform.SENTRY],
            sessions_analyzed=100,
            errors_analyzed=50,
            test_suggestions=[],
            error_insights=[],
            failure_predictions=[],
            coverage_gaps=[],
            overall_health_score=85.0,
            test_coverage_score=70.0,
            error_trend="stable",
            summary="Analysis complete",
            top_actions=[],
        )
        assert report.sessions_analyzed == 100
        assert report.errors_analyzed == 50
        assert report.overall_health_score == 85.0
        assert report.error_trend == "stable"
        assert len(report.platforms_analyzed) == 2


class TestAISynthesizer:
    """Tests for AISynthesizer class."""

    def create_mock_session(self, session_id: str = "sess_1", **kwargs) -> RealUserSession:
        """Create a mock session for testing."""
        defaults = {
            "session_id": session_id,
            "user_id": "user_1",
            "platform": Platform.DATADOG,
            "started_at": datetime(2024, 1, 1),
            "duration_ms": 5000,
            "page_views": ["/home", "/products"],
            "actions": [{"type": "click"}],
            "errors": [],
            "performance_metrics": {},
            "device": {},
            "geo": {},
            "frustration_signals": [],
            "conversion_events": [],
        }
        defaults.update(kwargs)
        return RealUserSession(**defaults)

    def create_mock_error(self, error_id: str = "err_1", **kwargs) -> ProductionError:
        """Create a mock error for testing."""
        defaults = {
            "error_id": error_id,
            "platform": Platform.SENTRY,
            "message": "Test error",
            "stack_trace": "at line 10",
            "first_seen": datetime(2024, 1, 1, 10, 0),
            "last_seen": datetime(2024, 1, 1, 12, 0),
            "occurrence_count": 10,
            "affected_users": 5,
            "affected_sessions": [],
            "tags": {},
            "context": {},
            "release": None,
            "environment": "production",
            "severity": "error",
            "status": "unresolved",
            "assignee": None,
            "issue_url": None,
        }
        defaults.update(kwargs)
        return ProductionError(**defaults)

    def create_mock_journey(self, pattern_id: str = "journey_1") -> UserJourneyPattern:
        """Create a mock user journey for testing."""
        return UserJourneyPattern(
            pattern_id=pattern_id,
            name="Checkout Flow",
            steps=[{"page": "/cart"}, {"page": "/checkout"}],
            frequency=1000,
            conversion_rate=0.75,
            avg_duration_ms=60000,
            drop_off_points=[],
            is_critical=True,
        )

    @pytest.fixture
    def mock_hub(self):
        """Create a mock ObservabilityHub."""
        with patch("src.integrations.observability_hub.get_settings"):
            hub = MagicMock(spec=ObservabilityHub)
            hub.providers = {}
            return hub

    @pytest.fixture
    def synthesizer(self, mock_hub):
        """Create an AISynthesizer with mocked dependencies."""
        with patch("src.integrations.ai_synthesis.get_settings") as mock_settings, \
             patch("src.integrations.ai_synthesis.anthropic.Anthropic"):
            mock_settings.return_value.anthropic_api_key = "test_key"
            return AISynthesizer(mock_hub)

    def test_init(self, mock_hub):
        """Test AISynthesizer initialization."""
        with patch("src.integrations.ai_synthesis.get_settings") as mock_settings, \
             patch("src.integrations.ai_synthesis.anthropic.Anthropic") as mock_anthropic:
            mock_settings.return_value.anthropic_api_key = "test_key"

            synthesizer = AISynthesizer(mock_hub)

            assert synthesizer.hub == mock_hub
            mock_anthropic.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize(self, synthesizer, mock_hub):
        """Test main synthesize function."""
        mock_hub.get_all_sessions = AsyncMock(return_value=[])
        mock_hub.get_all_errors = AsyncMock(return_value=[])
        mock_hub.get_all_anomalies = AsyncMock(return_value=[])
        mock_hub.get_all_user_journeys = AsyncMock(return_value=[])

        report = await synthesizer.synthesize(lookback_hours=24)

        assert isinstance(report, SynthesisReport)
        assert report.sessions_analyzed == 0
        assert report.errors_analyzed == 0

    @pytest.mark.asyncio
    async def test_synthesize_with_data(self, synthesizer, mock_hub):
        """Test synthesize with actual data."""
        sessions = [self.create_mock_session()]
        errors = [self.create_mock_error()]

        mock_hub.get_all_sessions = AsyncMock(return_value=sessions)
        mock_hub.get_all_errors = AsyncMock(return_value=errors)
        mock_hub.get_all_anomalies = AsyncMock(return_value=[])
        mock_hub.get_all_user_journeys = AsyncMock(return_value=[])

        report = await synthesizer.synthesize()

        assert report.sessions_analyzed == 1
        assert report.errors_analyzed == 1

    @pytest.mark.asyncio
    async def test_synthesize_handles_exceptions(self, synthesizer, mock_hub):
        """Test synthesize handles API exceptions gracefully."""
        mock_hub.get_all_sessions = AsyncMock(side_effect=Exception("API Error"))
        mock_hub.get_all_errors = AsyncMock(return_value=[])
        mock_hub.get_all_anomalies = AsyncMock(return_value=[])
        mock_hub.get_all_user_journeys = AsyncMock(return_value=[])

        report = await synthesizer.synthesize()

        # Should handle exception and continue
        assert report.sessions_analyzed == 0

    def test_identify_high_value_sessions(self, synthesizer):
        """Test identifying high-value sessions."""
        sessions = [
            self.create_mock_session("sess_1", conversion_events=[{"type": "purchase"}]),
            self.create_mock_session("sess_2", errors=[{"message": "error"}]),
            self.create_mock_session("sess_3", actions=[{"type": "click"}] * 15),
            self.create_mock_session("sess_4", frustration_signals=[{"type": "rage_click"}]),
            self.create_mock_session("sess_5"),
        ]

        result = synthesizer._identify_high_value_sessions(sessions)

        # Session with conversion should be first (score 50)
        assert result[0].session_id == "sess_1"

    def test_identify_high_value_sessions_empty(self, synthesizer):
        """Test with empty sessions."""
        result = synthesizer._identify_high_value_sessions([])
        assert result == []

    @pytest.mark.asyncio
    async def test_session_to_test_no_actions(self, synthesizer):
        """Test session_to_test with no actions."""
        session = self.create_mock_session(actions=[])

        result = await synthesizer._session_to_test(session)

        assert result is None

    @pytest.mark.asyncio
    async def test_session_to_test_success(self, synthesizer):
        """Test session_to_test with Claude response."""
        session = self.create_mock_session(actions=[{"type": "click"}])

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"name": "Test", "description": "Desc", "priority": "high", "confidence": 0.8, "steps": [], "expected_outcomes": [], "tags": [], "estimated_coverage": 0.1}')]

        synthesizer.client.messages.create = MagicMock(return_value=mock_response)

        result = await synthesizer._session_to_test(session)

        assert result is not None
        assert result.name == "Test"
        assert result.priority == ActionPriority.HIGH

    @pytest.mark.asyncio
    async def test_session_to_test_api_error(self, synthesizer):
        """Test session_to_test handles API errors."""
        session = self.create_mock_session(actions=[{"type": "click"}])

        synthesizer.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        result = await synthesizer._session_to_test(session)

        assert result is None

    @pytest.mark.asyncio
    async def test_error_to_test_success(self, synthesizer):
        """Test error_to_test with Claude response."""
        error = self.create_mock_error()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"name": "Regression Test", "description": "Prevent error", "priority": "high", "confidence": 0.85, "steps": [], "expected_outcomes": [], "tags": ["regression"], "estimated_coverage": 0.05}')]

        synthesizer.client.messages.create = MagicMock(return_value=mock_response)

        result = await synthesizer._error_to_test(error)

        assert result is not None
        assert "regression" in result.tags

    @pytest.mark.asyncio
    async def test_error_to_test_api_error(self, synthesizer):
        """Test error_to_test handles API errors."""
        error = self.create_mock_error()

        synthesizer.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        result = await synthesizer._error_to_test(error)

        assert result is None

    @pytest.mark.asyncio
    async def test_journey_to_test_no_steps(self, synthesizer):
        """Test journey_to_test with no steps."""
        journey = UserJourneyPattern(
            pattern_id="journey_1",
            name="Empty Journey",
            steps=[],
            frequency=100,
            conversion_rate=0.5,
            avg_duration_ms=30000,
            drop_off_points=[],
            is_critical=False,
        )

        result = await synthesizer._journey_to_test(journey)

        assert result is None

    @pytest.mark.asyncio
    async def test_journey_to_test_success(self, synthesizer):
        """Test journey_to_test with Claude response."""
        journey = self.create_mock_journey()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"name": "E2E Test", "description": "Journey test", "priority": "high", "confidence": 0.9, "steps": [], "expected_outcomes": [], "tags": ["e2e"], "estimated_coverage": 0.2}')]

        synthesizer.client.messages.create = MagicMock(return_value=mock_response)

        result = await synthesizer._journey_to_test(journey)

        assert result is not None
        # Critical journey should have critical priority
        assert result.priority == ActionPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_journey_to_test_api_error(self, synthesizer):
        """Test journey_to_test handles API errors."""
        journey = self.create_mock_journey()

        synthesizer.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        result = await synthesizer._journey_to_test(journey)

        assert result is None

    def test_group_similar_errors(self, synthesizer):
        """Test grouping similar errors."""
        errors = [
            self.create_mock_error("err_1", message="TypeError: undefined", stack_trace="at module.js:10"),
            self.create_mock_error("err_2", message="TypeError: undefined", stack_trace="at module.js:10"),
            self.create_mock_error("err_3", message="NetworkError", stack_trace="at network.js:20"),
        ]

        groups = synthesizer._group_similar_errors(errors)

        assert len(groups) == 2
        # First group should have the matching errors
        assert len(groups[0]) == 2

    def test_group_similar_errors_empty(self, synthesizer):
        """Test grouping with empty errors."""
        groups = synthesizer._group_similar_errors([])
        assert groups == []

    def test_group_similar_errors_by_stack_trace(self, synthesizer):
        """Test grouping by stack trace."""
        errors = [
            self.create_mock_error("err_1", message="Error 1", stack_trace="at module.js:10\nat app.js:20"),
            self.create_mock_error("err_2", message="Error 2", stack_trace="at module.js:10\nat app.js:20"),
        ]

        groups = synthesizer._group_similar_errors(errors)

        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_generate_root_cause_hypothesis_network(self, synthesizer):
        """Test root cause hypothesis for network errors."""
        error = self.create_mock_error(message="fetch failed: network error")
        result = synthesizer._generate_root_cause_hypothesis(error)
        assert "Network" in result

    def test_generate_root_cause_hypothesis_null(self, synthesizer):
        """Test root cause hypothesis for null errors."""
        error = self.create_mock_error(message="Cannot read property 'x' of undefined")
        result = synthesizer._generate_root_cause_hypothesis(error)
        assert "Null" in result

    def test_generate_root_cause_hypothesis_timeout(self, synthesizer):
        """Test root cause hypothesis for timeout errors."""
        error = self.create_mock_error(message="Request timeout exceeded")
        result = synthesizer._generate_root_cause_hypothesis(error)
        assert "timeout" in result

    def test_generate_root_cause_hypothesis_permission(self, synthesizer):
        """Test root cause hypothesis for permission errors."""
        error = self.create_mock_error(message="403 Forbidden")
        result = synthesizer._generate_root_cause_hypothesis(error)
        assert "authorization" in result.lower()

    def test_generate_root_cause_hypothesis_syntax(self, synthesizer):
        """Test root cause hypothesis for syntax errors."""
        error = self.create_mock_error(message="JSON parse error")
        result = synthesizer._generate_root_cause_hypothesis(error)
        assert "parsing" in result.lower()

    def test_generate_root_cause_hypothesis_unknown(self, synthesizer):
        """Test root cause hypothesis for unknown errors."""
        error = self.create_mock_error(message="Something went wrong")
        result = synthesizer._generate_root_cause_hypothesis(error)
        assert "investigation" in result.lower()

    def test_generate_error_actions_critical(self, synthesizer):
        """Test generating actions for critical errors."""
        error = self.create_mock_error(affected_sessions=["sess_1"])
        actions = synthesizer._generate_error_actions(error, impact_score=90)

        assert any("IMMEDIATE" in a for a in actions)
        assert any("Notify" in a for a in actions)
        assert any("regression" in a for a in actions)

    def test_generate_error_actions_low_impact(self, synthesizer):
        """Test generating actions for low impact errors."""
        error = self.create_mock_error(stack_trace="at line 10")
        actions = synthesizer._generate_error_actions(error, impact_score=20)

        assert not any("IMMEDIATE" in a for a in actions)
        assert any("stack trace" in a for a in actions)

    def test_analyze_error_trend_empty(self, synthesizer):
        """Test error trend analysis with no errors."""
        result = synthesizer._analyze_error_trend([])
        assert result["is_increasing"] is False

    def test_analyze_error_trend_single_hour(self, synthesizer):
        """Test error trend analysis with single hour."""
        errors = [
            self.create_mock_error("err_1", first_seen=datetime(2024, 1, 1, 10, 0)),
        ]
        result = synthesizer._analyze_error_trend(errors)
        assert result["is_increasing"] is False

    def test_analyze_error_trend_increasing(self, synthesizer):
        """Test error trend analysis with increasing trend."""
        errors = [
            self.create_mock_error("err_1", first_seen=datetime(2024, 1, 1, 10, 0), occurrence_count=10),
            self.create_mock_error("err_2", first_seen=datetime(2024, 1, 1, 11, 0), occurrence_count=50),
        ]
        result = synthesizer._analyze_error_trend(errors)

        assert result["is_increasing"] is True
        assert result["increase_percent"] > 20

    def test_analyze_error_trend_stable(self, synthesizer):
        """Test error trend analysis with stable trend."""
        errors = [
            self.create_mock_error("err_1", first_seen=datetime(2024, 1, 1, 10, 0), occurrence_count=10),
            self.create_mock_error("err_2", first_seen=datetime(2024, 1, 1, 11, 0), occurrence_count=10),
        ]
        result = synthesizer._analyze_error_trend(errors)

        assert result["is_increasing"] is False

    @pytest.mark.asyncio
    async def test_predict_failures_error_trend(self, synthesizer):
        """Test failure prediction for error rate increase."""
        errors = [
            self.create_mock_error("err_1", first_seen=datetime(2024, 1, 1, 10, 0), occurrence_count=10),
            self.create_mock_error("err_2", first_seen=datetime(2024, 1, 1, 11, 0), occurrence_count=100),
        ]

        predictions = await synthesizer._predict_failures([], errors, [])

        assert len(predictions) >= 1
        assert any(p.prediction_type == "error_rate_increase" for p in predictions)

    @pytest.mark.asyncio
    async def test_predict_failures_performance_degradation(self, synthesizer):
        """Test failure prediction for performance degradation."""
        anomalies = [
            PerformanceAnomaly(
                anomaly_id="anom_1",
                platform=Platform.DATADOG,
                metric="LCP",
                baseline_value=1500,
                current_value=4000,
                deviation_percent=166,
                affected_pages=["/home"],
                affected_users_percent=25,
                started_at=datetime(2024, 1, 1),
                detected_at=datetime(2024, 1, 1),
                probable_cause=None,
            )
        ]

        predictions = await synthesizer._predict_failures([], [], anomalies)

        assert len(predictions) >= 1
        assert any(p.prediction_type == "performance_degradation" for p in predictions)

    @pytest.mark.asyncio
    async def test_predict_failures_frustration(self, synthesizer):
        """Test failure prediction for high frustration."""
        sessions = [
            self.create_mock_session(f"sess_{i}", frustration_signals=[{"type": "rage_click"}] * 10)
            for i in range(10)
        ]

        predictions = await synthesizer._predict_failures(sessions, [], [])

        assert len(predictions) >= 1
        assert any(p.prediction_type == "user_experience_degradation" for p in predictions)

    @pytest.mark.asyncio
    async def test_identify_coverage_gaps(self, synthesizer):
        """Test identifying coverage gaps."""
        # Need more than 5 sessions for traffic threshold
        sessions = [
            self.create_mock_session(f"sess_{i}", page_views=["/home", "/products"])
            for i in range(10)
        ]

        gaps = await synthesizer._identify_coverage_gaps(sessions, [])

        # /home should have highest traffic
        assert len(gaps) >= 1
        assert any("/home" in g.area for g in gaps)

    @pytest.mark.asyncio
    async def test_identify_coverage_gaps_with_dict_pages(self, synthesizer):
        """Test coverage gaps with dict page views."""
        # Need more than 5 sessions for traffic threshold
        sessions = [
            self.create_mock_session(f"sess_{i}", page_views=[{"url": "/home"}, {"url": "/products"}])
            for i in range(10)
        ]

        gaps = await synthesizer._identify_coverage_gaps(sessions, [])

        assert len(gaps) >= 1

    @pytest.mark.asyncio
    async def test_identify_coverage_gaps_empty(self, synthesizer):
        """Test coverage gaps with no sessions."""
        gaps = await synthesizer._identify_coverage_gaps([], [])
        assert gaps == []

    def test_calculate_health_score_perfect(self, synthesizer):
        """Test health score with no issues."""
        score = synthesizer._calculate_health_score([], [])
        assert score == 100.0

    def test_calculate_health_score_with_errors(self, synthesizer):
        """Test health score with errors."""
        errors = [
            self.create_mock_error("err_1", severity="critical"),
            self.create_mock_error("err_2", severity="error"),
        ]

        score = synthesizer._calculate_health_score(errors, [])

        assert score < 100
        assert score >= 0

    def test_calculate_health_score_with_anomalies(self, synthesizer):
        """Test health score with performance anomalies."""
        anomalies = [
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
        ]

        score = synthesizer._calculate_health_score([], anomalies)

        assert score < 100
        assert score >= 0

    def test_calculate_coverage_score_no_gaps(self, synthesizer):
        """Test coverage score with no gaps."""
        score = synthesizer._calculate_coverage_score([])
        assert score == 100.0

    def test_calculate_coverage_score_with_gaps(self, synthesizer):
        """Test coverage score with gaps."""
        gaps = [
            CoverageGap(
                id="cg_1",
                area="/checkout",
                description="Gap",
                user_traffic_percent=40,
                current_coverage_percent=0,
                priority=ActionPriority.CRITICAL,
                suggested_tests=[],
            ),
            CoverageGap(
                id="cg_2",
                area="/cart",
                description="Gap",
                user_traffic_percent=20,
                current_coverage_percent=0,
                priority=ActionPriority.HIGH,
                suggested_tests=[],
            ),
        ]

        score = synthesizer._calculate_coverage_score(gaps)

        assert score < 100
        assert score >= 0

    def test_calculate_error_trend_degrading(self, synthesizer):
        """Test error trend calculation - degrading."""
        errors = [
            self.create_mock_error("err_1", first_seen=datetime(2024, 1, 1, 10, 0), occurrence_count=10),
            self.create_mock_error("err_2", first_seen=datetime(2024, 1, 1, 11, 0), occurrence_count=100),
        ]

        trend = synthesizer._calculate_error_trend(errors)
        assert trend in ["degrading", "slightly_degrading"]

    def test_calculate_error_trend_stable(self, synthesizer):
        """Test error trend calculation - stable."""
        errors = [
            self.create_mock_error("err_1", first_seen=datetime(2024, 1, 1, 10, 0), occurrence_count=10),
            self.create_mock_error("err_2", first_seen=datetime(2024, 1, 1, 11, 0), occurrence_count=10),
        ]

        trend = synthesizer._calculate_error_trend(errors)
        assert trend == "stable"

    def test_calculate_error_trend_improving(self, synthesizer):
        """Test error trend calculation - improving."""
        errors = [
            self.create_mock_error("err_1", first_seen=datetime(2024, 1, 1, 10, 0), occurrence_count=100),
            self.create_mock_error("err_2", first_seen=datetime(2024, 1, 1, 11, 0), occurrence_count=10),
        ]

        trend = synthesizer._calculate_error_trend(errors)
        assert trend == "improving"

    @pytest.mark.asyncio
    async def test_generate_summary(self, synthesizer):
        """Test generating executive summary."""
        summary = await synthesizer._generate_summary(
            test_suggestions=[],
            error_insights=[],
            predictions=[],
            coverage_gaps=[],
            health_score=85.0,
            sessions_analyzed=100,
            errors_analyzed=50,
        )

        assert "100 sessions" in summary
        assert "50 errors" in summary
        assert "85" in summary

    @pytest.mark.asyncio
    async def test_generate_summary_with_critical_errors(self, synthesizer):
        """Test summary with critical errors."""
        error = self.create_mock_error()
        insight = ErrorInsight(
            error=error,
            priority=ActionPriority.CRITICAL,
            impact_score=90,
            root_cause_hypothesis="Hypothesis",
            suggested_test=None,
            related_errors=[],
            affected_user_journeys=[],
            recommended_actions=[],
        )

        summary = await synthesizer._generate_summary(
            test_suggestions=[],
            error_insights=[insight],
            predictions=[],
            coverage_gaps=[],
            health_score=50.0,
            sessions_analyzed=10,
            errors_analyzed=5,
        )

        assert "ALERT" in summary
        assert "critical" in summary.lower()

    @pytest.mark.asyncio
    async def test_generate_summary_with_predictions(self, synthesizer):
        """Test summary with high confidence predictions."""
        prediction = FailurePrediction(
            id="fp_1",
            prediction_type="error_rate_increase",
            confidence=0.9,
            affected_area="Global",
            description="Error rate increasing",
            evidence=[],
            recommended_actions=[],
            predicted_timeframe="Next 2 hours",
            prevention_tests=[],
        )

        summary = await synthesizer._generate_summary(
            test_suggestions=[],
            error_insights=[],
            predictions=[prediction],
            coverage_gaps=[],
            health_score=70.0,
            sessions_analyzed=10,
            errors_analyzed=5,
        )

        assert "WARNING" in summary
        assert "prediction" in summary.lower()

    def test_prioritize_actions(self, synthesizer):
        """Test prioritizing actions."""
        error = self.create_mock_error()
        insight = ErrorInsight(
            error=error,
            priority=ActionPriority.CRITICAL,
            impact_score=90,
            root_cause_hypothesis="Hypothesis",
            suggested_test=None,
            related_errors=[],
            affected_user_journeys=[],
            recommended_actions=[],
        )

        suggestion = TestSuggestion(
            id="ts_1",
            name="Test",
            description="Desc",
            source="session_replay",
            source_platform=Platform.DATADOG,
            source_id="sess_1",
            priority=ActionPriority.HIGH,
            confidence=0.9,
            steps=[],
            expected_outcomes=[],
            tags=[],
            estimated_coverage=0.1,
        )

        prediction = FailurePrediction(
            id="fp_1",
            prediction_type="error_rate",
            confidence=0.85,
            affected_area="API",
            description="Rate increasing",
            evidence=[],
            recommended_actions=["Monitor"],
            predicted_timeframe="Soon",
            prevention_tests=[],
        )

        actions = synthesizer._prioritize_actions(
            test_suggestions=[suggestion],
            error_insights=[insight],
            predictions=[prediction],
            coverage_gaps=[],
        )

        # Critical errors should come first
        assert actions[0]["priority"] == "critical"
        assert len(actions) <= 10

    def test_prioritize_actions_empty(self, synthesizer):
        """Test prioritizing with no actions."""
        actions = synthesizer._prioritize_actions([], [], [], [])
        assert actions == []


class TestCreateAISynthesizer:
    """Tests for create_ai_synthesizer function."""

    @pytest.mark.asyncio
    async def test_create_with_hub(self):
        """Test creating synthesizer with provided hub."""
        with patch("src.integrations.ai_synthesis.get_settings") as mock_settings, \
             patch("src.integrations.ai_synthesis.anthropic.Anthropic"), \
             patch("src.integrations.observability_hub.get_settings"):
            mock_settings.return_value.anthropic_api_key = "test_key"

            hub = ObservabilityHub()
            synthesizer = await create_ai_synthesizer(hub)

            assert synthesizer.hub == hub

    @pytest.mark.asyncio
    async def test_create_without_hub(self):
        """Test creating synthesizer without hub."""
        with patch("src.integrations.ai_synthesis.get_settings") as mock_settings, \
             patch("src.integrations.ai_synthesis.anthropic.Anthropic"), \
             patch("src.integrations.ai_synthesis.ObservabilityHub") as mock_hub_class:
            mock_settings.return_value.anthropic_api_key = "test_key"

            synthesizer = await create_ai_synthesizer()

            mock_hub_class.assert_called_once()
            assert synthesizer.hub is not None


class TestAnalyzeErrors:
    """Tests for _analyze_errors method."""

    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer for testing."""
        with patch("src.integrations.ai_synthesis.get_settings") as mock_settings, \
             patch("src.integrations.ai_synthesis.anthropic.Anthropic"), \
             patch("src.integrations.observability_hub.get_settings"):
            mock_settings.return_value.anthropic_api_key = "test_key"
            hub = MagicMock(spec=ObservabilityHub)
            return AISynthesizer(hub)

    @pytest.mark.asyncio
    async def test_analyze_errors_empty(self, synthesizer):
        """Test analyzing empty error list."""
        insights = await synthesizer._analyze_errors([], [])
        assert insights == []

    @pytest.mark.asyncio
    async def test_analyze_errors_with_data(self, synthesizer):
        """Test analyzing errors with data."""
        errors = [
            ProductionError(
                error_id="err_1",
                platform=Platform.SENTRY,
                message="Error 1",
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
                severity="critical",
                status="unresolved",
                assignee=None,
                issue_url=None,
            )
        ]

        synthesizer._error_to_test = AsyncMock(return_value=None)

        insights = await synthesizer._analyze_errors(errors, [])

        assert len(insights) == 1
        assert insights[0].priority == ActionPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_analyze_errors_priority_levels(self, synthesizer):
        """Test different priority levels based on impact."""
        errors = [
            ProductionError(
                error_id="err_high",
                platform=Platform.SENTRY,
                message="High impact",
                stack_trace=None,
                first_seen=datetime(2024, 1, 1),
                last_seen=datetime(2024, 1, 1),
                occurrence_count=50,
                affected_users=12,
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
                error_id="err_medium",
                platform=Platform.SENTRY,
                message="Medium impact",
                stack_trace=None,
                first_seen=datetime(2024, 1, 1),
                last_seen=datetime(2024, 1, 1),
                occurrence_count=10,
                affected_users=5,
                affected_sessions=[],
                tags={},
                context={},
                release=None,
                environment="production",
                severity="warning",
                status="unresolved",
                assignee=None,
                issue_url=None,
            ),
        ]

        synthesizer._error_to_test = AsyncMock(return_value=None)

        insights = await synthesizer._analyze_errors(errors, [])

        # Should have different priorities
        priorities = [i.priority for i in insights]
        assert ActionPriority.HIGH in priorities or ActionPriority.MEDIUM in priorities


class TestGenerateTestSuggestions:
    """Tests for _generate_test_suggestions method."""

    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer for testing."""
        with patch("src.integrations.ai_synthesis.get_settings") as mock_settings, \
             patch("src.integrations.ai_synthesis.anthropic.Anthropic"), \
             patch("src.integrations.observability_hub.get_settings"):
            mock_settings.return_value.anthropic_api_key = "test_key"
            hub = MagicMock(spec=ObservabilityHub)
            return AISynthesizer(hub)

    @pytest.mark.asyncio
    async def test_generate_empty(self, synthesizer):
        """Test generating with no data."""
        synthesizer._session_to_test = AsyncMock(return_value=None)
        synthesizer._error_to_test = AsyncMock(return_value=None)
        synthesizer._journey_to_test = AsyncMock(return_value=None)

        suggestions = await synthesizer._generate_test_suggestions([], [], [])
        assert suggestions == []

    @pytest.mark.asyncio
    async def test_generate_from_sessions(self, synthesizer):
        """Test generating suggestions from sessions."""
        session = RealUserSession(
            session_id="sess_1",
            user_id="user_1",
            platform=Platform.DATADOG,
            started_at=datetime(2024, 1, 1),
            duration_ms=5000,
            page_views=["/home"],
            actions=[{"type": "click"}],
            errors=[],
            performance_metrics={},
            device={},
            geo={},
            frustration_signals=[],
            conversion_events=[{"type": "purchase"}],
        )

        suggestion = TestSuggestion(
            id="ts_1",
            name="Test",
            description="Desc",
            source="session_replay",
            source_platform=Platform.DATADOG,
            source_id="sess_1",
            priority=ActionPriority.HIGH,
            confidence=0.9,
            steps=[],
            expected_outcomes=[],
            tags=[],
            estimated_coverage=0.1,
        )

        synthesizer._session_to_test = AsyncMock(return_value=suggestion)
        synthesizer._error_to_test = AsyncMock(return_value=None)
        synthesizer._journey_to_test = AsyncMock(return_value=None)

        suggestions = await synthesizer._generate_test_suggestions([session], [], [])

        assert len(suggestions) == 1
        assert suggestions[0].id == "ts_1"
