"""Tests for the Correlation Engine service.

This module tests:
- CorrelationEngine initialization
- Event retrieval methods
- Correlation algorithms (time proximity, key overlap)
- Pattern detection (failure clusters, deployment risks)
- Commit impact analysis
- AI insight generation
- Insight persistence
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.correlation_engine import (
    CorrelationEngine,
    CorrelationKey,
    CorrelationResult,
    GeneratedInsight,
    InsightType,
    SDLCEventData,
    Severity,
    get_correlation_engine,
)


class TestDataClasses:
    """Tests for dataclasses used by the correlation engine."""

    def test_correlation_key_hash_and_equality(self):
        """Test CorrelationKey hashing and equality."""
        key1 = CorrelationKey(key_type="commit_sha", value="abc123")
        key2 = CorrelationKey(key_type="commit_sha", value="abc123")
        key3 = CorrelationKey(key_type="commit_sha", value="def456")

        assert key1 == key2
        assert key1 != key3
        assert hash(key1) == hash(key2)
        assert hash(key1) != hash(key3)

    def test_correlation_key_string_conversion(self):
        """Test CorrelationKey value conversion to string."""
        key1 = CorrelationKey(key_type="pr_number", value=123)
        key2 = CorrelationKey(key_type="pr_number", value="123")

        assert key1 == key2

    def test_sdlc_event_data_from_db_row(self):
        """Test SDLCEventData.from_db_row creation."""
        row = {
            "id": "evt_123",
            "project_id": "proj_456",
            "event_type": "commit",
            "source_platform": "github",
            "external_id": "abc123",
            "external_url": "https://github.com/org/repo/commit/abc123",
            "title": "Fix bug",
            "occurred_at": "2024-01-15T10:00:00Z",
            "commit_sha": "abc123",
            "pr_number": 45,
            "jira_key": "PROJ-123",
            "deploy_id": None,
            "branch_name": "main",
            "data": {"additions": 10, "deletions": 5},
        }

        event = SDLCEventData.from_db_row(row)

        assert event.id == "evt_123"
        assert event.project_id == "proj_456"
        assert event.event_type == "commit"
        assert event.commit_sha == "abc123"
        assert event.pr_number == 45
        assert event.jira_key == "PROJ-123"
        assert event.data == {"additions": 10, "deletions": 5}

    def test_sdlc_event_data_from_db_row_iso_datetime(self):
        """Test SDLCEventData handles various datetime formats."""
        row = {
            "id": "evt_123",
            "project_id": "proj_456",
            "event_type": "commit",
            "source_platform": "github",
            "external_id": "abc123",
            "occurred_at": "2024-01-15T10:00:00+00:00",
        }

        event = SDLCEventData.from_db_row(row)
        assert event.occurred_at.year == 2024

    def test_sdlc_event_data_get_correlation_keys(self):
        """Test SDLCEventData.get_correlation_keys."""
        event = SDLCEventData(
            id="evt_123",
            project_id="proj_456",
            event_type="commit",
            source_platform="github",
            external_id="abc123",
            external_url=None,
            title="Fix bug",
            occurred_at=datetime.now(UTC),
            commit_sha="abc123",
            pr_number=45,
            jira_key="PROJ-123",
            deploy_id=None,
            branch_name="main",
        )

        keys = event.get_correlation_keys()

        assert len(keys) == 4
        key_types = {k.key_type for k in keys}
        assert "commit_sha" in key_types
        assert "pr_number" in key_types
        assert "jira_key" in key_types
        assert "branch_name" in key_types

    def test_generated_insight_creation(self):
        """Test GeneratedInsight creation."""
        insight = GeneratedInsight(
            insight_type=InsightType.FAILURE_CLUSTER,
            severity=Severity.HIGH,
            title="Test Insight",
            description="Detailed description",
            recommendations=[{"action": "investigate", "description": "Do X"}],
            event_ids=["evt_1", "evt_2"],
            confidence=0.85,
        )

        assert insight.insight_type == InsightType.FAILURE_CLUSTER
        assert insight.severity == Severity.HIGH
        assert insight.confidence == 0.85


class TestCorrelationEngineInit:
    """Tests for CorrelationEngine initialization."""

    def test_init(self, mock_env_vars):
        """Test engine initialization."""
        with patch("src.services.correlation_engine.get_supabase_client") as mock_supabase:
            mock_supabase.return_value = MagicMock()
            with patch("src.services.correlation_engine.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                engine = CorrelationEngine()

                assert engine.supabase is not None
                assert engine.settings is not None


class TestCorrelationEngineEventRetrieval:
    """Tests for event retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_events_in_window_success(self, mock_env_vars):
        """Test getting events in a time window."""
        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={
            "data": [
                {
                    "id": "evt_1",
                    "project_id": "proj_123",
                    "event_type": "commit",
                    "source_platform": "github",
                    "external_id": "abc",
                    "occurred_at": "2024-01-15T10:00:00Z",
                },
                {
                    "id": "evt_2",
                    "project_id": "proj_123",
                    "event_type": "deploy",
                    "source_platform": "github",
                    "external_id": "def",
                    "occurred_at": "2024-01-15T11:00:00Z",
                },
            ]
        })

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                engine = CorrelationEngine()

                events = await engine.get_events_in_window(
                    project_id="proj_123",
                    start_time=datetime.now(UTC) - timedelta(days=1),
                    end_time=datetime.now(UTC),
                )

        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_get_events_in_window_with_filter(self, mock_env_vars):
        """Test getting events with event type filter."""
        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={"data": []})

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                engine = CorrelationEngine()

                await engine.get_events_in_window(
                    project_id="proj_123",
                    start_time=datetime.now(UTC) - timedelta(days=1),
                    end_time=datetime.now(UTC),
                    event_types=["error", "incident"],
                )

        call_args = mock_supabase.request.call_args[0][0]
        assert "error" in call_args
        assert "incident" in call_args

    @pytest.mark.asyncio
    async def test_get_events_in_window_error(self, mock_env_vars):
        """Test getting events when error occurs."""
        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={"error": "Database error"})

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                engine = CorrelationEngine()

                events = await engine.get_events_in_window(
                    project_id="proj_123",
                    start_time=datetime.now(UTC) - timedelta(days=1),
                    end_time=datetime.now(UTC),
                )

        assert events == []

    @pytest.mark.asyncio
    async def test_get_events_by_key(self, mock_env_vars):
        """Test getting events by correlation key."""
        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={
            "data": [
                {
                    "id": "evt_1",
                    "project_id": "proj_123",
                    "event_type": "commit",
                    "source_platform": "github",
                    "external_id": "abc",
                    "occurred_at": "2024-01-15T10:00:00Z",
                    "commit_sha": "abc123",
                }
            ]
        })

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                engine = CorrelationEngine()

                key = CorrelationKey("commit_sha", "abc123")
                events = await engine.get_events_by_key("proj_123", key)

        assert len(events) == 1


class TestCorrelationAlgorithms:
    """Tests for correlation algorithms."""

    def test_calculate_time_proximity_score_close(self, mock_env_vars):
        """Test time proximity score for close events."""
        with patch("src.services.correlation_engine.get_supabase_client"):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                now = datetime.now(UTC)
                event1 = SDLCEventData(
                    id="evt_1", project_id="p", event_type="commit",
                    source_platform="github", external_id="a",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha=None, pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )
                event2 = SDLCEventData(
                    id="evt_2", project_id="p", event_type="deploy",
                    source_platform="github", external_id="b",
                    external_url=None, title=None, occurred_at=now + timedelta(hours=1),
                    commit_sha=None, pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )

                score = engine.calculate_time_proximity_score(event1, event2)

        # 1 hour apart in 24 hour window = ~0.287
        assert 0.2 < score < 0.3

    def test_calculate_time_proximity_score_far(self, mock_env_vars):
        """Test time proximity score for far events."""
        with patch("src.services.correlation_engine.get_supabase_client"):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                now = datetime.now(UTC)
                event1 = SDLCEventData(
                    id="evt_1", project_id="p", event_type="commit",
                    source_platform="github", external_id="a",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha=None, pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )
                event2 = SDLCEventData(
                    id="evt_2", project_id="p", event_type="deploy",
                    source_platform="github", external_id="b",
                    external_url=None, title=None, occurred_at=now + timedelta(hours=25),
                    commit_sha=None, pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )

                score = engine.calculate_time_proximity_score(event1, event2)

        # Beyond max_hours window
        assert score == 0.0

    def test_calculate_key_overlap_score_shared_commit(self, mock_env_vars):
        """Test key overlap score with shared commit SHA."""
        with patch("src.services.correlation_engine.get_supabase_client"):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                now = datetime.now(UTC)
                event1 = SDLCEventData(
                    id="evt_1", project_id="p", event_type="commit",
                    source_platform="github", external_id="a",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha="abc123", pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )
                event2 = SDLCEventData(
                    id="evt_2", project_id="p", event_type="deploy",
                    source_platform="github", external_id="b",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha="abc123", pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )

                score, shared = engine.calculate_key_overlap_score(event1, event2)

        # commit_sha weight is 0.5
        assert score == 0.5
        assert len(shared) == 1
        assert shared[0].key_type == "commit_sha"

    def test_calculate_key_overlap_score_no_shared(self, mock_env_vars):
        """Test key overlap score with no shared keys."""
        with patch("src.services.correlation_engine.get_supabase_client"):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                now = datetime.now(UTC)
                event1 = SDLCEventData(
                    id="evt_1", project_id="p", event_type="commit",
                    source_platform="github", external_id="a",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha="abc123", pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )
                event2 = SDLCEventData(
                    id="evt_2", project_id="p", event_type="deploy",
                    source_platform="github", external_id="b",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha="def456", pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )

                score, shared = engine.calculate_key_overlap_score(event1, event2)

        assert score == 0.0
        assert shared == []

    def test_calculate_key_overlap_score_multiple_shared(self, mock_env_vars):
        """Test key overlap score with multiple shared keys."""
        with patch("src.services.correlation_engine.get_supabase_client"):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                now = datetime.now(UTC)
                event1 = SDLCEventData(
                    id="evt_1", project_id="p", event_type="commit",
                    source_platform="github", external_id="a",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha="abc123", pr_number=45, jira_key=None,
                    deploy_id=None, branch_name="main",
                )
                event2 = SDLCEventData(
                    id="evt_2", project_id="p", event_type="deploy",
                    source_platform="github", external_id="b",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha="abc123", pr_number=45, jira_key=None,
                    deploy_id=None, branch_name="main",
                )

                score, shared = engine.calculate_key_overlap_score(event1, event2)

        # Capped at 0.5
        assert score == 0.5
        assert len(shared) == 3

    def test_calculate_correlation_confidence(self, mock_env_vars):
        """Test overall correlation confidence calculation."""
        with patch("src.services.correlation_engine.get_supabase_client"):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                now = datetime.now(UTC)
                event1 = SDLCEventData(
                    id="evt_1", project_id="p", event_type="commit",
                    source_platform="github", external_id="a",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha="abc123", pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )
                event2 = SDLCEventData(
                    id="evt_2", project_id="p", event_type="deploy",
                    source_platform="github", external_id="b",
                    external_url=None, title=None, occurred_at=now + timedelta(hours=1),
                    commit_sha="abc123", pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )

                result = engine.calculate_correlation_confidence(event1, event2)

        assert isinstance(result, CorrelationResult)
        assert result.source_event_id == "evt_1"
        assert result.target_event_id == "evt_2"
        assert 0 < result.confidence <= 1.0
        assert len(result.factors) > 0

    def test_event_type_relationship_score(self, mock_env_vars):
        """Test event type relationship scoring."""
        with patch("src.services.correlation_engine.get_supabase_client"):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                now = datetime.now(UTC)
                commit_event = SDLCEventData(
                    id="evt_1", project_id="p", event_type="commit",
                    source_platform="github", external_id="a",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha=None, pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )
                build_event = SDLCEventData(
                    id="evt_2", project_id="p", event_type="build",
                    source_platform="github", external_id="b",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha=None, pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )

                score = engine._calculate_event_type_relationship_score(commit_event, build_event)

        # commit -> build is a known relationship
        assert score == 0.2

    def test_determine_correlation_type(self, mock_env_vars):
        """Test correlation type determination."""
        with patch("src.services.correlation_engine.get_supabase_client"):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                now = datetime.now(UTC)
                commit_event = SDLCEventData(
                    id="evt_1", project_id="p", event_type="commit",
                    source_platform="github", external_id="a",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha=None, pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )
                error_event = SDLCEventData(
                    id="evt_2", project_id="p", event_type="error",
                    source_platform="github", external_id="b",
                    external_url=None, title=None, occurred_at=now,
                    commit_sha=None, pr_number=None, jira_key=None,
                    deploy_id=None, branch_name=None,
                )

                corr_type = engine._determine_correlation_type(commit_event, error_event)

        assert corr_type == "introduced_by"


class TestPatternDetection:
    """Tests for pattern detection methods."""

    @pytest.mark.asyncio
    async def test_detect_failure_clusters_insufficient_events(self, mock_env_vars):
        """Test failure cluster detection with insufficient events."""
        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={"data": []})

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                clusters = await engine.detect_failure_clusters("proj_123", days=7, min_cluster_size=3)

        assert clusters == []

    @pytest.mark.asyncio
    async def test_detect_failure_clusters_found(self, mock_env_vars):
        """Test failure cluster detection finding clusters."""
        events_data = [
            {"id": f"evt_{i}", "project_id": "proj_123", "event_type": "error",
             "source_platform": "github", "external_id": str(i),
             "occurred_at": (datetime.now(UTC) - timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "commit_sha": "common_sha"}
            for i in range(5)
        ]

        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={"data": events_data})

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                clusters = await engine.detect_failure_clusters("proj_123", days=7, min_cluster_size=3)

        assert len(clusters) >= 1
        assert clusters[0]["event_count"] >= 3
        assert clusters[0]["key_type"] == "commit_sha"
        assert clusters[0]["key_value"] == "common_sha"

    @pytest.mark.asyncio
    async def test_detect_deployment_risks_no_deploys(self, mock_env_vars):
        """Test deployment risk detection with no deployments."""
        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={"data": []})

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                risks = await engine.detect_deployment_risks("proj_123")

        assert risks == []

    @pytest.mark.asyncio
    async def test_analyze_commit_impact_not_found(self, mock_env_vars):
        """Test commit impact analysis when not found."""
        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={"data": []})

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                result = await engine.analyze_commit_impact("proj_123", "nonexistent_sha")

        assert result["found"] is False
        assert "No events found" in result["message"]

    @pytest.mark.asyncio
    async def test_analyze_commit_impact_found(self, mock_env_vars):
        """Test commit impact analysis with results."""
        events_data = [
            {"id": "evt_1", "project_id": "proj_123", "event_type": "commit",
             "source_platform": "github", "external_id": "a",
             "occurred_at": "2024-01-15T10:00:00Z", "commit_sha": "abc123"},
            {"id": "evt_2", "project_id": "proj_123", "event_type": "error",
             "source_platform": "github", "external_id": "b",
             "occurred_at": "2024-01-15T11:00:00Z", "commit_sha": "abc123"},
        ]

        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={"data": events_data})

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                result = await engine.analyze_commit_impact("proj_123", "abc123")

        assert result["found"] is True
        assert result["total_events"] == 2
        assert "error" in result["events_by_type"]
        assert result["risk_score"] > 0


class TestInsightGeneration:
    """Tests for AI insight generation."""

    @pytest.mark.asyncio
    async def test_generate_insights_with_clusters(self, mock_env_vars):
        """Test insight generation with failure clusters."""
        events_data = [
            {"id": f"evt_{i}", "project_id": "proj_123", "event_type": "error",
             "source_platform": "github", "external_id": str(i),
             "occurred_at": (datetime.now(UTC) - timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "commit_sha": "common_sha"}
            for i in range(5)
        ]

        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={"data": events_data})

        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = None  # Disable AI insights for this test

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings", return_value=mock_settings):
                engine = CorrelationEngine()

                insights = await engine.generate_insights("proj_123", days=7, max_insights=5)

        # Should find cluster-based insights
        assert len(insights) >= 1
        assert insights[0].insight_type == InsightType.FAILURE_CLUSTER

    @pytest.mark.asyncio
    async def test_generate_ai_insights_no_api_key(self, mock_env_vars):
        """Test AI insight generation without API key."""
        mock_supabase = MagicMock()
        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = None

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings", return_value=mock_settings):
                engine = CorrelationEngine()

                insights = await engine._generate_ai_insights("proj_123", days=7, max_insights=3)

        assert insights == []

    def test_summarize_events_for_ai(self, mock_env_vars):
        """Test event summarization for AI."""
        with patch("src.services.correlation_engine.get_supabase_client"):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                now = datetime.now(UTC)
                events = [
                    SDLCEventData(
                        id=f"evt_{i}", project_id="p", event_type="error" if i % 2 == 0 else "commit",
                        source_platform="github", external_id=str(i),
                        external_url=None, title=f"Event {i}",
                        occurred_at=now - timedelta(hours=i),
                        commit_sha=None, pr_number=None, jira_key=None,
                        deploy_id=None, branch_name=None,
                    )
                    for i in range(10)
                ]

                summary = engine._summarize_events_for_ai(events)

        assert "Total Events: 10" in summary
        assert "Events by Type:" in summary
        assert "error" in summary.lower()
        assert "commit" in summary.lower()


class TestInsightPersistence:
    """Tests for insight persistence methods."""

    @pytest.mark.asyncio
    async def test_save_insight_success(self, mock_env_vars):
        """Test saving insight successfully."""
        mock_supabase = MagicMock()
        mock_supabase.insert = AsyncMock(return_value={"data": [{"id": "insight_123"}]})

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                insight = GeneratedInsight(
                    insight_type=InsightType.FAILURE_CLUSTER,
                    severity=Severity.HIGH,
                    title="Test",
                    description="Description",
                    recommendations=[],
                    event_ids=["evt_1"],
                    confidence=0.8,
                )

                result = await engine.save_insight("proj_123", insight)

        assert result == "insight_123"

    @pytest.mark.asyncio
    async def test_save_insight_error(self, mock_env_vars):
        """Test saving insight with error."""
        mock_supabase = MagicMock()
        mock_supabase.insert = AsyncMock(return_value={"error": "Database error"})

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                insight = GeneratedInsight(
                    insight_type=InsightType.FAILURE_CLUSTER,
                    severity=Severity.HIGH,
                    title="Test",
                    description="Description",
                    recommendations=[],
                    event_ids=[],
                    confidence=0.8,
                )

                result = await engine.save_insight("proj_123", insight)

        assert result is None

    @pytest.mark.asyncio
    async def test_save_correlation_success(self, mock_env_vars):
        """Test saving correlation successfully."""
        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={"data": [{"id": "corr_123"}]})

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                correlation = CorrelationResult(
                    source_event_id="evt_1",
                    target_event_id="evt_2",
                    correlation_type="related_to",
                    confidence=0.85,
                )

                result = await engine.save_correlation(correlation)

        assert result == "corr_123"

    @pytest.mark.asyncio
    async def test_save_correlation_duplicate_ignored(self, mock_env_vars):
        """Test saving duplicate correlation is handled gracefully."""
        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={"error": "duplicate key"})

        with patch("src.services.correlation_engine.get_supabase_client", return_value=mock_supabase):
            with patch("src.services.correlation_engine.get_settings"):
                engine = CorrelationEngine()

                correlation = CorrelationResult(
                    source_event_id="evt_1",
                    target_event_id="evt_2",
                    correlation_type="related_to",
                    confidence=0.85,
                )

                result = await engine.save_correlation(correlation)

        assert result is None


class TestGlobalFunctions:
    """Tests for module-level functions."""

    def test_get_correlation_engine_singleton(self, mock_env_vars):
        """Test get_correlation_engine returns singleton."""
        import src.services.correlation_engine as module
        module._engine = None

        with patch("src.services.correlation_engine.get_supabase_client"):
            with patch("src.services.correlation_engine.get_settings"):
                engine1 = get_correlation_engine()
                engine2 = get_correlation_engine()

        assert engine1 is engine2

        # Clean up
        module._engine = None
