"""Tests for the Correlations API module (src/api/correlations.py).

Covers:
- Timeline endpoints (get timeline, get event)
- Impact analysis endpoints (commit impact)
- Root cause analysis endpoints
- Insights endpoints (list, generate, acknowledge, resolve, dismiss)
- Natural language query endpoint
- Event correlation endpoints (by-commit, by-pr, by-jira, by-deploy)
- Statistics endpoint
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from fastapi import HTTPException


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    mock = MagicMock()
    mock.request = AsyncMock()
    mock.insert = AsyncMock()
    mock.update = AsyncMock()
    mock.rpc = AsyncMock()
    return mock


@pytest.fixture
def mock_request():
    """Create a mock HTTP request."""
    request = MagicMock()
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers = {"user-agent": "test-agent"}
    request.query_params = {}
    return request


@pytest.fixture
def mock_user_context():
    """Create a mock UserContext."""
    from src.api.security.auth import UserContext, AuthMethod

    return UserContext(
        user_id="user-123",
        organization_id=str(uuid.uuid4()),
        email="test@example.com",
        auth_method=AuthMethod.API_KEY,
    )


@pytest.fixture
def sample_sdlc_event():
    """Create sample SDLC event data."""
    return {
        "id": str(uuid.uuid4()),
        "event_type": "commit",
        "source_platform": "github",
        "external_id": "abc123",
        "external_url": "https://github.com/org/repo/commit/abc123",
        "title": "Fix bug in auth module",
        "occurred_at": datetime.now(UTC).isoformat(),
        "commit_sha": "abc123def456",
        "pr_number": 42,
        "jira_key": "PROJ-123",
        "deploy_id": None,
        "data": {"author": "dev@example.com"},
    }


@pytest.fixture
def sample_correlation_insight():
    """Create sample correlation insight data."""
    return {
        "id": str(uuid.uuid4()),
        "insight_type": "failure_cluster",
        "severity": "high",
        "title": "Multiple failures in auth module",
        "description": "Several tests in auth module failed after recent commit",
        "recommendations": [{"action": "Review auth changes"}],
        "event_ids": [str(uuid.uuid4())],
        "status": "active",
        "created_at": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def mock_correlation_engine():
    """Create a mock correlation engine."""
    mock_engine = MagicMock()

    # Mock generate_insights
    mock_insight = MagicMock()
    mock_insight.insight_type = MagicMock()
    mock_insight.insight_type.value = "failure_cluster"
    mock_insight.severity = MagicMock()
    mock_insight.severity.value = "high"
    mock_insight.title = "Test Insight"
    mock_insight.description = "Test description"
    mock_insight.recommendations = []
    mock_insight.event_ids = []

    mock_engine.generate_insights = AsyncMock(return_value=[mock_insight])
    mock_engine.save_insight = AsyncMock(return_value=str(uuid.uuid4()))
    mock_engine.analyze_commit_impact = AsyncMock(return_value={
        "commit_sha": "abc123",
        "found": True,
        "total_events": 5,
        "events_by_type": {"commit": 1, "deploy": 2, "error": 2},
        "event_ids": [str(uuid.uuid4())],
        "risk_score": 0.75,
        "risk_factors": [{"factor": "errors", "impact": "high"}],
        "time_span_hours": 24.0,
        "first_event": datetime.now(UTC).isoformat(),
        "last_event": datetime.now(UTC).isoformat(),
    })

    return mock_engine


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestRowToSdlcEvent:
    """Tests for _row_to_sdlc_event helper function."""

    def test_convert_complete_row(self, mock_env_vars, sample_sdlc_event):
        """Test converting a complete event row."""
        from src.api.correlations import _row_to_sdlc_event

        event = _row_to_sdlc_event(sample_sdlc_event)

        assert event.id == sample_sdlc_event["id"]
        assert event.event_type == "commit"
        assert event.source_platform == "github"
        assert event.commit_sha == "abc123def456"

    def test_convert_minimal_row(self, mock_env_vars):
        """Test converting a minimal event row."""
        from src.api.correlations import _row_to_sdlc_event

        minimal_row = {
            "id": str(uuid.uuid4()),
            "event_type": "commit",
            "source_platform": "github",
            "external_id": "123",
            "occurred_at": datetime.now(UTC).isoformat(),
        }

        event = _row_to_sdlc_event(minimal_row)

        assert event.id == minimal_row["id"]
        assert event.data == {}


class TestRowToInsight:
    """Tests for _row_to_insight helper function."""

    def test_convert_complete_row(self, mock_env_vars, sample_correlation_insight):
        """Test converting a complete insight row."""
        from src.api.correlations import _row_to_insight

        insight = _row_to_insight(sample_correlation_insight)

        assert insight.id == sample_correlation_insight["id"]
        assert insight.severity == "high"
        assert insight.status == "active"


class TestCalculateImpactRiskScore:
    """Tests for _calculate_impact_risk_score helper function."""

    @pytest.mark.asyncio
    async def test_risk_score_with_errors(self, mock_env_vars):
        """Test risk score calculation with errors."""
        from src.api.correlations import _calculate_impact_risk_score

        events = [
            {"event_type": "error", "data": {"severity": "critical"}},
            {"event_type": "commit"},
        ]

        score = await _calculate_impact_risk_score(events)

        assert 0 <= score <= 1
        assert score > 0.2  # Errors should increase score

    @pytest.mark.asyncio
    async def test_risk_score_empty(self, mock_env_vars):
        """Test risk score with no events."""
        from src.api.correlations import _calculate_impact_risk_score

        score = await _calculate_impact_risk_score([])

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_risk_score_with_incident(self, mock_env_vars):
        """Test risk score with incident events."""
        from src.api.correlations import _calculate_impact_risk_score

        events = [
            {"event_type": "incident", "data": {}},
        ]

        score = await _calculate_impact_risk_score(events)

        assert score > 0.15  # Incidents have high weight


class TestGetPotentialImpacts:
    """Tests for _get_potential_impacts helper function."""

    @pytest.mark.asyncio
    async def test_impacts_with_errors(self, mock_env_vars):
        """Test impact descriptions with errors."""
        from src.api.correlations import _get_potential_impacts

        events = [
            {"event_type": "error"},
            {"event_type": "error"},
        ]

        impacts = await _get_potential_impacts(events)

        assert any("error" in i.lower() for i in impacts)

    @pytest.mark.asyncio
    async def test_impacts_with_failed_tests(self, mock_env_vars):
        """Test impact descriptions with failed tests."""
        from src.api.correlations import _get_potential_impacts

        events = [
            {"event_type": "test_run", "data": {"status": "failed"}},
        ]

        impacts = await _get_potential_impacts(events)

        assert any("test" in i.lower() for i in impacts)

    @pytest.mark.asyncio
    async def test_impacts_no_events(self, mock_env_vars):
        """Test impact descriptions with no significant events."""
        from src.api.correlations import _get_potential_impacts

        impacts = await _get_potential_impacts([])

        assert any("no significant" in i.lower() for i in impacts)


# ============================================================================
# Timeline Endpoint Tests
# ============================================================================

class TestGetTimeline:
    """Tests for get_timeline endpoint."""

    @pytest.mark.asyncio
    async def test_get_timeline_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_sdlc_event):
        """Test successful timeline retrieval."""
        from src.api.correlations import get_timeline

        mock_supabase.request.return_value = {"data": [sample_sdlc_event], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_timeline(
                mock_request,
                project_id=None,
                event_types=None,
                days=7,
                limit=100,
                offset=0,
                user=mock_user_context
            )

        assert len(result.events) == 1
        assert result.events[0].event_type == "commit"

    @pytest.mark.asyncio
    async def test_get_timeline_with_filters(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_sdlc_event):
        """Test timeline with event type filter."""
        from src.api.correlations import get_timeline

        mock_supabase.request.return_value = {"data": [sample_sdlc_event], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_timeline(
                mock_request,
                project_id=None,
                event_types=["commit", "deploy"],
                days=30,
                limit=100,
                offset=0,
                user=mock_user_context,
            )

        assert isinstance(result.events, list)

    @pytest.mark.asyncio
    async def test_get_timeline_table_not_found(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test graceful handling when table doesn't exist."""
        from src.api.correlations import get_timeline

        mock_supabase.request.return_value = {"data": None, "error": "42P01: table does not exist"}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_timeline(
                mock_request,
                project_id=None,
                event_types=None,
                days=7,
                limit=100,
                offset=0,
                user=mock_user_context
            )

        assert result.events == []
        assert result.total_count == 0


class TestGetEvent:
    """Tests for get_event endpoint."""

    @pytest.mark.asyncio
    async def test_get_event_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_sdlc_event):
        """Test successful event retrieval."""
        from src.api.correlations import get_event

        event_id = sample_sdlc_event["id"]
        mock_supabase.request.return_value = {"data": [sample_sdlc_event], "error": None}
        mock_supabase.rpc.return_value = {"data": [], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_event(event_id, mock_request, user=mock_user_context)

        assert result["event"].id == event_id

    @pytest.mark.asyncio
    async def test_get_event_not_found(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test event not found error."""
        from src.api.correlations import get_event

        event_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await get_event(event_id, mock_request, user=mock_user_context)

            assert exc_info.value.status_code == 404


# ============================================================================
# Impact Analysis Endpoint Tests
# ============================================================================

class TestGetCommitImpact:
    """Tests for get_commit_impact endpoint."""

    @pytest.mark.asyncio
    async def test_get_commit_impact_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_sdlc_event):
        """Test successful commit impact analysis."""
        from src.api.correlations import get_commit_impact

        commit_sha = "abc123"
        mock_supabase.request.return_value = {"data": [sample_sdlc_event], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_commit_impact(commit_sha, mock_request, user=mock_user_context)

        assert result.commit_sha == commit_sha
        assert 0 <= result.risk_score <= 1
        assert isinstance(result.potential_impacts, list)

    @pytest.mark.asyncio
    async def test_get_commit_impact_table_not_found(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test graceful handling when table doesn't exist."""
        from src.api.correlations import get_commit_impact

        commit_sha = "abc123"
        mock_supabase.request.return_value = {"data": None, "error": "42P01: table does not exist"}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_commit_impact(commit_sha, mock_request, user=mock_user_context)

        assert result.related_events == []


class TestGetCommitImpactAnalysis:
    """Tests for get_commit_impact_analysis endpoint using correlation engine."""

    @pytest.mark.asyncio
    async def test_get_impact_analysis_success(self, mock_env_vars, mock_request, mock_user_context):
        """Test successful impact analysis via correlation engine."""
        from src.api.correlations import get_commit_impact_analysis

        commit_sha = "abc123"
        project_id = str(uuid.uuid4())

        # Create a fresh mock correlation engine for this test
        mock_engine = MagicMock()
        mock_engine.analyze_commit_impact = AsyncMock(return_value={
            "commit_sha": "abc123",
            "found": True,
            "total_events": 5,
            "events_by_type": {"commit": 1, "deploy": 2, "error": 2},
            "event_ids": [str(uuid.uuid4())],
            "risk_score": 0.75,
            "risk_factors": [{"factor": "errors", "impact": "high"}],
            "time_span_hours": 24.0,
            "first_event": datetime.now(UTC).isoformat(),
            "last_event": datetime.now(UTC).isoformat(),
        })

        with patch("src.services.correlation_engine.get_correlation_engine", return_value=mock_engine):
            result = await get_commit_impact_analysis(
                mock_request,
                commit_sha=commit_sha,
                project_id=project_id,
                user=mock_user_context,
            )

        assert result.commit_sha == "abc123"
        assert result.found is True
        assert result.risk_score == 0.75


# ============================================================================
# Root Cause Analysis Endpoint Tests
# ============================================================================

class TestGetRootCause:
    """Tests for get_root_cause endpoint."""

    @pytest.mark.asyncio
    async def test_get_root_cause_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_sdlc_event):
        """Test successful root cause analysis."""
        from src.api.correlations import get_root_cause

        event_id = sample_sdlc_event["id"]

        # Mock multiple queries
        mock_supabase.request.side_effect = [
            {"data": [sample_sdlc_event], "error": None},  # Target event
            {"data": [], "error": None},  # Correlations
        ]
        mock_supabase.rpc.return_value = {"data": [], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_root_cause(event_id, mock_request, user=mock_user_context)

        assert result.target_event.id == event_id
        assert isinstance(result.analysis_summary, str)

    @pytest.mark.asyncio
    async def test_get_root_cause_not_found(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test root cause analysis for non-existent event."""
        from src.api.correlations import get_root_cause

        event_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await get_root_cause(event_id, mock_request, user=mock_user_context)

            assert exc_info.value.status_code == 404


# ============================================================================
# Insights Endpoint Tests
# ============================================================================

class TestGetInsights:
    """Tests for get_insights endpoint."""

    @pytest.mark.asyncio
    async def test_get_insights_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_correlation_insight):
        """Test successful insights retrieval."""
        from src.api.correlations import get_insights

        mock_supabase.request.return_value = {"data": [sample_correlation_insight], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_insights(
                mock_request,
                project_id=None,
                insight_types=None,
                status="active",
                limit=20,
                user=mock_user_context
            )

        assert len(result) == 1
        assert result[0].severity == "high"

    @pytest.mark.asyncio
    async def test_get_insights_with_filters(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_correlation_insight):
        """Test insights with status filter."""
        from src.api.correlations import get_insights

        mock_supabase.request.return_value = {"data": [sample_correlation_insight], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_insights(
                mock_request,
                project_id=None,
                insight_types=["failure_cluster"],
                status="active",
                limit=20,
                user=mock_user_context,
            )

        assert isinstance(result, list)


class TestGenerateInsights:
    """Tests for generate_insights endpoint."""

    @pytest.mark.asyncio
    async def test_generate_insights_success(self, mock_env_vars, mock_request, mock_user_context):
        """Test successful insight generation."""
        from src.api.correlations import generate_insights, GenerateInsightsRequest

        project_id = str(uuid.uuid4())
        request_body = GenerateInsightsRequest(
            days=7,
            max_insights=5,
            save_to_database=True,
        )

        # Create mock insight
        mock_insight = MagicMock()
        mock_insight.insight_type = MagicMock()
        mock_insight.insight_type.value = "failure_cluster"
        mock_insight.severity = MagicMock()
        mock_insight.severity.value = "high"
        mock_insight.title = "Test Insight"
        mock_insight.description = "Test description"
        mock_insight.recommendations = []
        mock_insight.event_ids = []

        mock_engine = MagicMock()
        mock_engine.generate_insights = AsyncMock(return_value=[mock_insight])
        mock_engine.save_insight = AsyncMock(return_value=str(uuid.uuid4()))

        with patch("src.services.correlation_engine.get_correlation_engine", return_value=mock_engine), \
             patch("src.api.correlations.get_settings", return_value=MagicMock()):

            result = await generate_insights(
                request_body,
                mock_request,
                project_id=project_id,
                user=mock_user_context,
            )

        assert len(result.insights) > 0
        assert "project_id" in result.analysis_summary


class TestAcknowledgeInsight:
    """Tests for acknowledge_insight endpoint."""

    @pytest.mark.asyncio
    async def test_acknowledge_insight_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test successful insight acknowledgment."""
        from src.api.correlations import acknowledge_insight

        insight_id = str(uuid.uuid4())
        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await acknowledge_insight(insight_id, mock_request, user=mock_user_context)

        assert result["success"] is True


class TestResolveInsight:
    """Tests for resolve_insight endpoint."""

    @pytest.mark.asyncio
    async def test_resolve_insight_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test successful insight resolution."""
        from src.api.correlations import resolve_insight

        insight_id = str(uuid.uuid4())
        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await resolve_insight(
                insight_id,
                mock_request,
                resolution_notes="Fixed the issue",
                user=mock_user_context,
            )

        assert result["success"] is True


class TestDismissInsight:
    """Tests for dismiss_insight endpoint."""

    @pytest.mark.asyncio
    async def test_dismiss_insight_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test successful insight dismissal."""
        from src.api.correlations import dismiss_insight

        insight_id = str(uuid.uuid4())
        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await dismiss_insight(
                insight_id,
                mock_request,
                reason="Not relevant",
                user=mock_user_context,
            )

        assert result["success"] is True


# ============================================================================
# Event Correlation Endpoint Tests
# ============================================================================

class TestGetEventsByCommit:
    """Tests for get_events_by_commit endpoint."""

    @pytest.mark.asyncio
    async def test_get_events_by_commit_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_sdlc_event):
        """Test successful events by commit retrieval."""
        from src.api.correlations import get_events_by_commit

        commit_sha = "abc123"
        mock_supabase.request.return_value = {"data": [sample_sdlc_event], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_events_by_commit(commit_sha, mock_request, user=mock_user_context)

        assert result["commit_sha"] == commit_sha
        assert len(result["events"]) == 1


class TestGetEventsByPR:
    """Tests for get_events_by_pr endpoint."""

    @pytest.mark.asyncio
    async def test_get_events_by_pr_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_sdlc_event):
        """Test successful events by PR retrieval."""
        from src.api.correlations import get_events_by_pr

        pr_number = 42
        mock_supabase.request.return_value = {"data": [sample_sdlc_event], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_events_by_pr(
                pr_number,
                mock_request,
                project_id=None,
                user=mock_user_context
            )

        assert result["pr_number"] == pr_number


class TestGetEventsByJira:
    """Tests for get_events_by_jira endpoint."""

    @pytest.mark.asyncio
    async def test_get_events_by_jira_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_sdlc_event):
        """Test successful events by Jira key retrieval."""
        from src.api.correlations import get_events_by_jira

        jira_key = "PROJ-123"
        mock_supabase.request.return_value = {"data": [sample_sdlc_event], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_events_by_jira(jira_key, mock_request, user=mock_user_context)

        assert result["jira_key"] == jira_key


class TestGetEventsByDeploy:
    """Tests for get_events_by_deploy endpoint."""

    @pytest.mark.asyncio
    async def test_get_events_by_deploy_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_sdlc_event):
        """Test successful events by deploy ID retrieval."""
        from src.api.correlations import get_events_by_deploy

        deploy_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": [sample_sdlc_event], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_events_by_deploy(deploy_id, mock_request, user=mock_user_context)

        assert result["deploy_id"] == deploy_id


# ============================================================================
# Statistics Endpoint Tests
# ============================================================================

class TestGetCorrelationStats:
    """Tests for get_correlation_stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_sdlc_event, sample_correlation_insight):
        """Test successful stats retrieval."""
        from src.api.correlations import get_correlation_stats

        mock_supabase.request.side_effect = [
            {"data": [sample_sdlc_event], "error": None},  # Events
            {"data": [sample_correlation_insight], "error": None},  # Insights
            {"data": [], "error": None},  # Correlations
        ]

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase):
            result = await get_correlation_stats(
                mock_request,
                project_id=None,
                days=30,
                user=mock_user_context
            )

        assert "events" in result
        assert "insights" in result
        assert "correlations" in result


# ============================================================================
# Natural Language Query Endpoint Tests
# ============================================================================

class TestNaturalLanguageQuery:
    """Tests for natural_language_query endpoint."""

    @pytest.mark.asyncio
    async def test_nl_query_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_sdlc_event):
        """Test successful natural language query."""
        from src.api.correlations import natural_language_query

        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = MagicMock()
        mock_settings.anthropic_api_key.get_secret_value.return_value = "test-key"

        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"interpreted_as": "Find commits", "event_types": ["commit"], "time_range_days": 7, "insights": [], "suggested_actions": []}')]
        mock_anthropic.messages.create.return_value = mock_response

        mock_supabase.request.return_value = {"data": [sample_sdlc_event], "error": None}

        with patch("src.api.correlations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.correlations.get_settings", return_value=mock_settings), \
             patch("anthropic.Anthropic", return_value=mock_anthropic):

            result = await natural_language_query(
                mock_request,
                query="Show me recent commits",
                project_id=None,
                user=mock_user_context,
            )

        assert result.query == "Show me recent commits"
        assert isinstance(result.events, list)

    @pytest.mark.asyncio
    async def test_nl_query_no_api_key(self, mock_env_vars, mock_request, mock_user_context):
        """Test error when Anthropic API key not configured."""
        from src.api.correlations import natural_language_query

        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = None

        with patch("src.api.correlations.get_settings", return_value=mock_settings):
            with pytest.raises(HTTPException) as exc_info:
                await natural_language_query(
                    mock_request,
                    query="Show me recent commits",
                    project_id=None,
                    user=mock_user_context,
                )

            assert exc_info.value.status_code == 500
            assert "API key not configured" in exc_info.value.detail


# ============================================================================
# Model Validation Tests
# ============================================================================

class TestSDLCEventModel:
    """Tests for SDLCEvent model validation."""

    def test_valid_event(self, mock_env_vars):
        """Test creating a valid SDLC event."""
        from src.api.correlations import SDLCEvent

        event = SDLCEvent(
            id=str(uuid.uuid4()),
            event_type="commit",
            source_platform="github",
            external_id="abc123",
            occurred_at=datetime.now(UTC),
        )

        assert event.event_type == "commit"
        assert event.source_platform == "github"


class TestCorrelationInsightModel:
    """Tests for CorrelationInsight model validation."""

    def test_valid_insight(self, mock_env_vars):
        """Test creating a valid correlation insight."""
        from src.api.correlations import CorrelationInsight

        insight = CorrelationInsight(
            id=str(uuid.uuid4()),
            insight_type="failure_cluster",
            severity="high",
            title="Test insight",
            description="Test description",
            created_at=datetime.now(UTC),
        )

        assert insight.severity == "high"
        assert insight.status == "active"  # Default value


class TestRootCauseChainModel:
    """Tests for RootCauseChain model validation."""

    def test_valid_chain(self, mock_env_vars):
        """Test creating a valid root cause chain."""
        from src.api.correlations import RootCauseChain, SDLCEvent

        event = SDLCEvent(
            id=str(uuid.uuid4()),
            event_type="commit",
            source_platform="github",
            external_id="abc123",
            occurred_at=datetime.now(UTC),
        )

        chain = RootCauseChain(
            event=event,
            correlation_type="caused_by",
            confidence=0.9,
        )

        assert chain.correlation_type == "caused_by"
        assert chain.confidence == 0.9


class TestGenerateInsightsRequestModel:
    """Tests for GenerateInsightsRequest model validation."""

    def test_valid_request(self, mock_env_vars):
        """Test creating a valid request."""
        from src.api.correlations import GenerateInsightsRequest

        request = GenerateInsightsRequest(
            days=14,
            max_insights=10,
            save_to_database=False,
        )

        assert request.days == 14
        assert request.save_to_database is False

    def test_request_defaults(self, mock_env_vars):
        """Test request default values."""
        from src.api.correlations import GenerateInsightsRequest

        request = GenerateInsightsRequest()

        assert request.days == 7
        assert request.max_insights == 5
        assert request.save_to_database is True
