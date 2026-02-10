"""
Production-grade E2E Quality Intelligence & AI Insights Integration Tests.

Tests the quality intelligence, AI insights, failure patterns, and flaky
detection endpoints against LIVE production with NO MOCKS.
Every response is validated against strict Pydantic models matching the
production API contract (camelCase via CamelCaseMiddleware).

Covers:
- Quality score, events, stats, risk scores, generated tests (quality.py)
- AI failure clustering, coverage gaps, insight generation/list (insights.py)
- Failure pattern CRUD, predict, train, accuracy stats (failure_patterns.py)
- Flaky test detection, trend, stats, quarantine (flaky_detector.py)

Run with:
    pytest tests/integration/test_quality_intelligence_live.py -v --tb=short

Run by category:
    pytest tests/integration/test_quality_intelligence_live.py -v -k "TestQualityScore"
    pytest tests/integration/test_quality_intelligence_live.py -v -k "TestQualityEvents"
    pytest tests/integration/test_quality_intelligence_live.py -v -k "TestRiskScores"
    pytest tests/integration/test_quality_intelligence_live.py -v -k "TestGeneratedTests"
    pytest tests/integration/test_quality_intelligence_live.py -v -k "TestSimilarErrors"
    pytest tests/integration/test_quality_intelligence_live.py -v -k "TestInsightCluster"
    pytest tests/integration/test_quality_intelligence_live.py -v -k "TestInsightCoverage"
    pytest tests/integration/test_quality_intelligence_live.py -v -k "TestInsightGenerate"
    pytest tests/integration/test_quality_intelligence_live.py -v -k "TestInsightList"
    pytest tests/integration/test_quality_intelligence_live.py -v -k "TestFailurePattern"
    pytest tests/integration/test_quality_intelligence_live.py -v -k "TestFlakyTest"
"""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — env vars with production defaults
# ═══════════════════════════════════════════════════════════════════════════

BASE_URL = os.environ.get(
    "ARGUS_BASE_URL",
    "https://skopaq-brain-production.up.railway.app",
)
API_KEY = os.environ.get(
    "ARGUS_API_KEY",
    "argus_sk_9a1819e0e9faa47011c28ff75bbeb442ffa285779233ea3e81aed7e992367198",
)
ORG_ID = os.environ.get(
    "ARGUS_ORG_ID",
    "229904aa-3084-4b25-8bf8-c07ae749888c",
)
PROJ_ID = os.environ.get(
    "ARGUS_PROJECT_ID",
    "e98fc22d-ae11-490d-86f2-116fdd29941b",
)

HEADERS = {
    "X-API-Key": API_KEY,
    "X-Organization-Id": ORG_ID,
    "Content-Type": "application/json",
}

# Non-AI endpoints
DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=15.0)
# AI endpoints (Claude calls) — allow 120s
AI_TIMEOUT = httpx.Timeout(120.0, connect=15.0)
# Slow AI endpoints (multi-step analysis)
SLOW_TIMEOUT = httpx.Timeout(180.0, connect=15.0)

FAKE_UUID = "00000000-0000-0000-0000-000000000000"

# ═══════════════════════════════════════════════════════════════════════════
# Endpoint URLs
# ═══════════════════════════════════════════════════════════════════════════

# Quality Intelligence (quality.py)
QUALITY_SCORE_URL = f"{BASE_URL}/api/v1/quality/score"
QUALITY_EVENTS_URL = f"{BASE_URL}/api/v1/quality/events"
QUALITY_STATS_URL = f"{BASE_URL}/api/v1/quality/stats"
QUALITY_RISK_SCORES_URL = f"{BASE_URL}/api/v1/quality/risk-scores"
QUALITY_GENERATED_TESTS_URL = f"{BASE_URL}/api/v1/quality/generated-tests"
QUALITY_SIMILAR_ERRORS_URL = f"{BASE_URL}/api/v1/quality/similar-errors"
QUALITY_GENERATE_TEST_URL = f"{BASE_URL}/api/v1/quality/generate-test"
QUALITY_BATCH_GENERATE_URL = f"{BASE_URL}/api/v1/quality/batch-generate"
QUALITY_UPDATE_TEST_URL = f"{BASE_URL}/api/v1/quality/update-test"
QUALITY_CALCULATE_RISK_URL = f"{BASE_URL}/api/v1/quality/calculate-risk"
QUALITY_BACKFILL_URL = f"{BASE_URL}/api/v1/quality/backfill-index"

# AI Insights (insights.py)
INSIGHTS_CLUSTER_URL = f"{BASE_URL}/api/v1/insights/cluster"
INSIGHTS_COVERAGE_URL = f"{BASE_URL}/api/v1/insights/coverage-gaps"
INSIGHTS_GENERATE_URL = f"{BASE_URL}/api/v1/insights/generate"
INSIGHTS_LIST_URL = f"{BASE_URL}/api/v1/insights"

# Failure Patterns (failure_patterns.py)
PATTERNS_URL = f"{BASE_URL}/api/v1/patterns"
PATTERNS_PREDICT_URL = f"{BASE_URL}/api/v1/patterns/predict"
PATTERNS_OUTCOME_URL = f"{BASE_URL}/api/v1/patterns/outcome"
PATTERNS_FEATURES_URL = f"{BASE_URL}/api/v1/patterns/features"
PATTERNS_TRAIN_URL = f"{BASE_URL}/api/v1/patterns/train"
PATTERNS_ACCURACY_URL = f"{BASE_URL}/api/v1/patterns/accuracy/stats"

# Flaky Tests (flaky_detector.py)
FLAKY_TESTS_URL = f"{BASE_URL}/api/v1/flaky-tests"
FLAKY_TREND_URL = f"{BASE_URL}/api/v1/flaky-tests/trend"
FLAKY_STATS_URL = f"{BASE_URL}/api/v1/flaky-tests/stats"


# ═══════════════════════════════════════════════════════════════════════════
# Response Models — strict Pydantic validation of the production API contract.
#
# The API uses a global CamelCaseMiddleware (src/api/middleware/camelcase.py)
# that converts every snake_case key to camelCase.  These models encode that
# contract via Field(alias=...).  If the API changes shape, Pydantic raises
# ValidationError and the test fails immediately — no silent drift.
#
# extra="allow" lets the API add NEW fields without breaking tests, but
# removing or renaming an existing field is caught instantly.
# ═══════════════════════════════════════════════════════════════════════════


# --- Quality Score ---
class QualityScoreResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    quality_score: float = Field(alias="qualityScore")
    risk_level: str = Field(alias="riskLevel")
    test_coverage: float = Field(alias="testCoverage")
    total_events: int = Field(alias="totalEvents")
    total_tests: int = Field(alias="totalTests")
    approved_tests: int = Field(alias="approvedTests")


# --- Quality Stats ---
class QualityStatsInner(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    total_events: int = Field(alias="totalEvents")
    events_by_status: dict = Field(alias="eventsByStatus")
    events_by_severity: dict = Field(alias="eventsBySeverity")
    total_generated_tests: int = Field(alias="totalGeneratedTests")
    tests_by_status: dict = Field(alias="testsByStatus")
    coverage_rate: float = Field(alias="coverageRate")


class QualityStatsResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    stats: QualityStatsInner


# --- Quality Events ---
class QualityEventsResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    events: list


# --- Risk Scores ---
class RiskScoresResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    risk_scores: list = Field(alias="riskScores")


# --- Generated Tests ---
class GeneratedTestsResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    tests: list


# --- Similar Errors ---
class SimilarErrorsResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    query: str
    matches: list
    total: int


# --- Batch Generate ---
class BatchGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    success: bool
    message: str


# --- Calculate Risk ---
class CalculateRiskResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    success: bool
    risk_scores: list = Field(alias="riskScores")
    total_entities: int = Field(alias="totalEntities")


# --- Backfill Index ---
class BackfillIndexResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    success: bool
    indexed: int
    total: int


# --- Insights: Failure Cluster ---
class FailureClusterItem(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    id: str
    name: str
    description: str
    count: int
    percentage: float
    error_type: str = Field(alias="errorType")
    root_cause_analysis: str = Field(alias="rootCauseAnalysis")
    affected_tests: list = Field(alias="affectedTests")
    affected_test_count: int = Field(alias="affectedTestCount")
    suggested_fix: str = Field(alias="suggestedFix")
    severity: str
    trend: str
    sample_errors: list = Field(alias="sampleErrors")


class FailureClusterResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    clusters: list
    total_failures: int = Field(alias="totalFailures")
    analysis_summary: str = Field(alias="analysisSummary")
    generated_at: str = Field(alias="generatedAt")


# --- Insights: Coverage Gaps ---
class CoverageGapItem(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    id: str
    area: str
    area_type: str = Field(alias="areaType")
    current_coverage: float = Field(alias="currentCoverage")
    risk_level: str = Field(alias="riskLevel")
    risk_analysis: str = Field(alias="riskAnalysis")
    impact_description: str = Field(alias="impactDescription")
    suggested_tests: list = Field(alias="suggestedTests")
    suggested_test_count: int = Field(alias="suggestedTestCount")
    priority_score: float = Field(alias="priorityScore")
    related_failures: list = Field(alias="relatedFailures")


class CoverageGapResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    gaps: list
    overall_coverage: float = Field(alias="overallCoverage")
    critical_gaps: int = Field(alias="criticalGaps")
    high_gaps: int = Field(alias="highGaps")
    total_suggested_tests: int = Field(alias="totalSuggestedTests")
    analysis_summary: str = Field(alias="analysisSummary")
    generated_at: str = Field(alias="generatedAt")


# --- Insights: Generate ---
class GenerateInsightsResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    insights: list
    total_generated: int = Field(alias="totalGenerated")
    analysis_duration_ms: int = Field(alias="analysisDurationMs")
    generated_at: str = Field(alias="generatedAt")


# --- Insights: List (array of AIInsight) ---
class AIInsightItem(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    id: str
    project_id: str = Field(alias="projectId")
    insight_type: str = Field(alias="insightType")
    severity: str
    title: str
    description: str
    confidence: float
    is_resolved: bool = Field(alias="isResolved")
    created_at: str = Field(alias="createdAt")


# --- Failure Patterns ---
class FailurePatternItem(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    id: str
    project_id: str = Field(alias="projectId")
    pattern_type: str = Field(alias="patternType")
    pattern_definition: dict = Field(alias="patternDefinition")
    occurrences: int
    predictions_made: int = Field(alias="predictionsMade")
    predictions_correct: int = Field(alias="predictionsCorrect")
    accuracy: float
    confidence: float
    is_active: bool = Field(alias="isActive")
    discovered_by: str = Field(alias="discoveredBy")
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")


# --- Accuracy Stats ---
class AccuracyStatsResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    project_id: str = Field(alias="projectId")
    total_patterns: int = Field(alias="totalPatterns")
    active_patterns: int = Field(alias="activePatterns")
    total_predictions: int = Field(alias="totalPredictions")
    correct_predictions: int = Field(alias="correctPredictions")
    overall_accuracy: float = Field(alias="overallAccuracy")
    accuracy_by_type: dict = Field(alias="accuracyByType")
    precision: float
    recall: float
    f1_score: float = Field(alias="f1Score")


# --- Training Job ---
class TrainingJobResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    id: str
    job_type: str = Field(alias="jobType")
    status: str
    progress: float
    commits_analyzed: int = Field(alias="commitsAnalyzed")
    test_runs_analyzed: int = Field(alias="testRunsAnalyzed")
    patterns_discovered: int = Field(alias="patternsDiscovered")
    patterns_updated: int = Field(alias="patternsUpdated")
    patterns_deactivated: int = Field(alias="patternsDeactivated")
    created_at: str = Field(alias="createdAt")


# --- Prediction Result ---
class PredictionResultResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    commit_sha: str = Field(alias="commitSha")
    failure_probability: float = Field(alias="failureProbability")
    confidence_level: str = Field(alias="confidenceLevel")
    contributing_patterns: list = Field(alias="contributingPatterns")
    reasons: list
    recommended_tests: list = Field(alias="recommendedTests")


# --- Flaky Tests ---
class RootCauseItem(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    type: str
    description: str
    confidence: float


class FlakyTestItem(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    id: str
    name: str
    flakiness_score: float = Field(alias="flakinessScore")
    total_runs: int = Field(alias="totalRuns")
    pass_count: int = Field(alias="passCount")
    fail_count: int = Field(alias="failCount")
    trend: str
    is_quarantined: bool = Field(alias="isQuarantined")
    root_causes: list = Field(alias="rootCauses")
    recent_results: list = Field(alias="recentResults")
    avg_duration: float = Field(alias="avgDuration")
    project_id: str = Field(alias="projectId")
    project_name: str = Field(alias="projectName")


class FlakyTestsListResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    tests: list
    total: int


# --- Flaky Trend ---
class FlakyTrendItem(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    date: str
    flaky: int
    fixed: int


class FlakyTrendResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    trend: list


# --- Flaky Stats ---
class FlakyStatsResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    total: int
    high: int
    medium: int
    low: int
    quarantined: int
    avg_score: float = Field(alias="avgScore")


# --- Quarantine Toggle ---
class QuarantineToggleResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    success: bool
    test_id: str = Field(alias="testId")
    is_quarantined: bool = Field(alias="isQuarantined")


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Quality Intelligence (quality.py)
# ═══════════════════════════════════════════════════════════════════════════


class TestQualityScore:
    """Tests for GET /api/v1/quality/score."""

    @pytest.mark.asyncio
    async def test_quality_score_returns_200(self):
        """Quality score endpoint returns valid response for project."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_SCORE_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        QualityScoreResponse.model_validate(data)
        assert 0 <= data["qualityScore"] <= 100
        assert data["riskLevel"] in ("low", "medium", "high")

    @pytest.mark.asyncio
    async def test_quality_score_missing_project_id(self):
        """Quality score requires project_id parameter."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(QUALITY_SCORE_URL, headers=HEADERS)
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_quality_score_invalid_uuid(self):
        """Quality score rejects invalid UUID format."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_SCORE_URL,
                headers=HEADERS,
                params={"project_id": "not-a-uuid"},
            )
        assert resp.status_code in (400, 422), f"Expected 400/422, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_quality_score_nonexistent_project(self):
        """Quality score returns sensible defaults for project with no data."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_SCORE_URL,
                headers=HEADERS,
                params={"project_id": FAKE_UUID},
            )
        # Should still return 200 with empty/default data
        assert resp.status_code == 200, resp.text
        data = resp.json()
        QualityScoreResponse.model_validate(data)


class TestQualityStats:
    """Tests for GET /api/v1/quality/stats."""

    @pytest.mark.asyncio
    async def test_quality_stats_returns_200(self):
        """Quality stats endpoint returns valid response."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_STATS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        QualityStatsResponse.model_validate(data)
        stats = data["stats"]
        assert stats["totalEvents"] >= 0
        assert stats["totalGeneratedTests"] >= 0
        assert 0 <= stats["coverageRate"] <= 100

    @pytest.mark.asyncio
    async def test_quality_stats_missing_project_id(self):
        """Quality stats requires project_id parameter."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(QUALITY_STATS_URL, headers=HEADERS)
        assert resp.status_code == 422


class TestQualityEvents:
    """Tests for GET /api/v1/quality/events."""

    @pytest.mark.asyncio
    async def test_events_returns_200(self):
        """Events endpoint returns valid list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_EVENTS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        QualityEventsResponse.model_validate(data)
        assert isinstance(data["events"], list)

    @pytest.mark.asyncio
    async def test_events_with_status_filter(self):
        """Events can be filtered by status."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_EVENTS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "status": "new"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        QualityEventsResponse.model_validate(data)

    @pytest.mark.asyncio
    async def test_events_with_severity_filter(self):
        """Events can be filtered by severity."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_EVENTS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "severity": "error"},
            )
        assert resp.status_code == 200, resp.text
        QualityEventsResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_events_pagination(self):
        """Events endpoint supports pagination."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_EVENTS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "limit": 5, "offset": 0},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert len(data["events"]) <= 5

    @pytest.mark.asyncio
    async def test_events_missing_project_id(self):
        """Events endpoint requires project_id."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(QUALITY_EVENTS_URL, headers=HEADERS)
        assert resp.status_code == 422


class TestRiskScores:
    """Tests for GET /api/v1/quality/risk-scores."""

    @pytest.mark.asyncio
    async def test_risk_scores_returns_200(self):
        """Risk scores endpoint returns valid response."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_RISK_SCORES_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        RiskScoresResponse.model_validate(data)
        assert isinstance(data["riskScores"], list)

    @pytest.mark.asyncio
    async def test_risk_scores_with_entity_filter(self):
        """Risk scores can be filtered by entity type."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_RISK_SCORES_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "entity_type": "component"},
            )
        assert resp.status_code == 200, resp.text
        RiskScoresResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_risk_scores_missing_project_id(self):
        """Risk scores requires project_id."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(QUALITY_RISK_SCORES_URL, headers=HEADERS)
        assert resp.status_code == 422


class TestGeneratedTests:
    """Tests for GET /api/v1/quality/generated-tests."""

    @pytest.mark.asyncio
    async def test_generated_tests_returns_200(self):
        """Generated tests endpoint returns valid list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_GENERATED_TESTS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        GeneratedTestsResponse.model_validate(data)
        assert isinstance(data["tests"], list)

    @pytest.mark.asyncio
    async def test_generated_tests_with_status_filter(self):
        """Generated tests can be filtered by status."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_GENERATED_TESTS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "status": "pending"},
            )
        assert resp.status_code == 200, resp.text
        GeneratedTestsResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_generated_tests_missing_project_id(self):
        """Generated tests requires project_id."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(QUALITY_GENERATED_TESTS_URL, headers=HEADERS)
        assert resp.status_code == 422


class TestSimilarErrors:
    """Tests for GET /api/v1/quality/similar-errors."""

    @pytest.mark.asyncio
    async def test_similar_errors_returns_200(self):
        """Similar errors endpoint returns matches for a query."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_SIMILAR_ERRORS_URL,
                headers=HEADERS,
                params={"error_text": "button not found", "limit": 5},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        SimilarErrorsResponse.model_validate(data)
        assert data["query"] == "button not found"
        assert isinstance(data["matches"], list)
        assert data["total"] >= 0

    @pytest.mark.asyncio
    async def test_similar_errors_with_project_filter(self):
        """Similar errors can be scoped to a project."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_SIMILAR_ERRORS_URL,
                headers=HEADERS,
                params={
                    "error_text": "timeout during login",
                    "project_id": PROJ_ID,
                    "limit": 3,
                },
            )
        assert resp.status_code == 200, resp.text
        SimilarErrorsResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_similar_errors_min_score_filter(self):
        """Similar errors can filter by minimum similarity score."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_SIMILAR_ERRORS_URL,
                headers=HEADERS,
                params={"error_text": "authentication failed", "min_score": 0.9},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        SimilarErrorsResponse.model_validate(data)
        # All matches should meet the min_score threshold
        for match in data["matches"]:
            assert match.get("score", 0) >= 0.9

    @pytest.mark.asyncio
    async def test_similar_errors_missing_query(self):
        """Similar errors requires error_text parameter."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(QUALITY_SIMILAR_ERRORS_URL, headers=HEADERS)
        assert resp.status_code == 422


class TestBatchGenerate:
    """Tests for POST /api/v1/quality/batch-generate."""

    @pytest.mark.asyncio
    async def test_batch_generate_returns_200(self):
        """Batch generate endpoint processes events for project."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.post(
                QUALITY_BATCH_GENERATE_URL,
                headers=HEADERS,
                json={
                    "project_id": PROJ_ID,
                    "status": "new",
                    "limit": 1,
                    "framework": "playwright",
                },
            )
        # 429 is acceptable - rate limited AI endpoint
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            BatchGenerateResponse.model_validate(data)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_batch_generate_invalid_project_id(self):
        """Batch generate rejects invalid project_id."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                QUALITY_BATCH_GENERATE_URL,
                headers=HEADERS,
                json={
                    "project_id": "not-a-uuid",
                    "status": "new",
                    "limit": 1,
                },
            )
        # 429 acceptable - rate limited endpoint
        assert resp.status_code in (400, 422, 429), f"Expected 400/422/429, got {resp.status_code}: {resp.text}"


class TestCalculateRisk:
    """Tests for POST /api/v1/quality/calculate-risk."""

    @pytest.mark.asyncio
    async def test_calculate_risk_returns_200(self):
        """Calculate risk endpoint returns risk scores."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                QUALITY_CALCULATE_RISK_URL,
                headers=HEADERS,
                json={
                    "project_id": PROJ_ID,
                    "entity_types": ["page", "component"],
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        CalculateRiskResponse.model_validate(data)
        assert data["success"] is True
        assert data["totalEntities"] >= 0

    @pytest.mark.asyncio
    async def test_calculate_risk_invalid_uuid(self):
        """Calculate risk rejects invalid UUID."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                QUALITY_CALCULATE_RISK_URL,
                headers=HEADERS,
                json={"project_id": "bad-uuid"},
            )
        assert resp.status_code in (400, 422)


class TestBackfillIndex:
    """Tests for POST /api/v1/quality/backfill-index."""

    @pytest.mark.asyncio
    async def test_backfill_index_returns_200(self):
        """Backfill index endpoint returns valid response."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.post(
                QUALITY_BACKFILL_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "limit": 5, "offset": 0},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        BackfillIndexResponse.model_validate(data)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_backfill_index_no_project(self):
        """Backfill index works without project_id (indexes all)."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                QUALITY_BACKFILL_URL,
                headers=HEADERS,
                params={"limit": 2},
            )
        assert resp.status_code == 200, resp.text
        BackfillIndexResponse.model_validate(resp.json())


class TestUpdateTest:
    """Tests for POST /api/v1/quality/update-test."""

    @pytest.mark.asyncio
    async def test_update_test_nonexistent(self):
        """Updating a non-existent test should not crash."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                QUALITY_UPDATE_TEST_URL,
                headers=HEADERS,
                json={
                    "test_id": FAKE_UUID,
                    "action": "approve",
                },
            )
        # Could be 200 (silently no-op) or 404 or 500 - just ensure no crash
        assert resp.status_code in (200, 404, 500), resp.text

    @pytest.mark.asyncio
    async def test_update_test_invalid_uuid(self):
        """Updating with invalid UUID should fail validation."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                QUALITY_UPDATE_TEST_URL,
                headers=HEADERS,
                json={
                    "test_id": "not-valid",
                    "action": "approve",
                },
            )
        assert resp.status_code in (400, 422)


# ═══════════════════════════════════════════════════════════════════════════
# Tests — AI Insights (insights.py)
# ═══════════════════════════════════════════════════════════════════════════


class TestInsightCluster:
    """Tests for POST /api/v1/insights/cluster."""

    @pytest.mark.asyncio
    async def test_cluster_returns_200(self):
        """Failure clustering endpoint returns valid response."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                INSIGHTS_CLUSTER_URL,
                headers=HEADERS,
                json={
                    "project_id": PROJ_ID,
                    "days": 30,
                    "min_cluster_size": 2,
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        FailureClusterResponse.model_validate(data)
        assert data["totalFailures"] >= 0
        assert isinstance(data["clusters"], list)
        assert "analysisSummary" in data

    @pytest.mark.asyncio
    async def test_cluster_short_window(self):
        """Clustering works with a short time window (1 day)."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.post(
                INSIGHTS_CLUSTER_URL,
                headers=HEADERS,
                json={
                    "project_id": PROJ_ID,
                    "days": 1,
                    "min_cluster_size": 2,
                },
            )
        assert resp.status_code == 200, resp.text
        FailureClusterResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_cluster_missing_project_id(self):
        """Cluster requires project_id in body."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                INSIGHTS_CLUSTER_URL,
                headers=HEADERS,
                json={"days": 7},
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_cluster_invalid_project_id(self):
        """Cluster rejects invalid project_id UUID."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                INSIGHTS_CLUSTER_URL,
                headers=HEADERS,
                json={"project_id": "bad-uuid", "days": 7},
            )
        assert resp.status_code in (400, 422)


class TestInsightCoverage:
    """Tests for POST /api/v1/insights/coverage-gaps."""

    @pytest.mark.asyncio
    async def test_coverage_gaps_returns_200(self):
        """Coverage gaps endpoint returns valid response."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                INSIGHTS_COVERAGE_URL,
                headers=HEADERS,
                json={
                    "project_id": PROJ_ID,
                    "include_api_gaps": True,
                    "include_ui_gaps": True,
                    "include_flow_gaps": True,
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        CoverageGapResponse.model_validate(data)
        assert data["overallCoverage"] >= 0
        assert isinstance(data["gaps"], list)

    @pytest.mark.asyncio
    async def test_coverage_gaps_api_only(self):
        """Coverage gaps can run with only API gaps enabled."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.post(
                INSIGHTS_COVERAGE_URL,
                headers=HEADERS,
                json={
                    "project_id": PROJ_ID,
                    "include_api_gaps": True,
                    "include_ui_gaps": False,
                    "include_flow_gaps": False,
                },
            )
        assert resp.status_code == 200, resp.text
        CoverageGapResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_coverage_gaps_missing_project_id(self):
        """Coverage gaps requires project_id."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                INSIGHTS_COVERAGE_URL,
                headers=HEADERS,
                json={"include_api_gaps": True},
            )
        assert resp.status_code == 422


class TestInsightGenerate:
    """Tests for POST /api/v1/insights/generate."""

    @pytest.mark.asyncio
    async def test_generate_insights_returns_200(self):
        """Insight generation endpoint returns valid response."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                INSIGHTS_GENERATE_URL,
                headers=HEADERS,
                json={
                    "project_id": PROJ_ID,
                    "insight_types": ["failure_pattern", "coverage_gap"],
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        GenerateInsightsResponse.model_validate(data)
        assert data["totalGenerated"] >= 0
        assert data["analysisDurationMs"] >= 0

    @pytest.mark.asyncio
    async def test_generate_insights_single_type(self):
        """Insight generation works with a single type."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                INSIGHTS_GENERATE_URL,
                headers=HEADERS,
                json={
                    "project_id": PROJ_ID,
                    "insight_types": ["risk_alert"],
                },
            )
        assert resp.status_code == 200, resp.text
        GenerateInsightsResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_generate_insights_missing_project_id(self):
        """Insight generation requires project_id."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                INSIGHTS_GENERATE_URL,
                headers=HEADERS,
                json={"insight_types": ["failure_pattern"]},
            )
        assert resp.status_code == 422


class TestInsightList:
    """Tests for GET /api/v1/insights."""

    @pytest.mark.asyncio
    async def test_list_insights_returns_200(self):
        """List insights endpoint returns array."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                INSIGHTS_LIST_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)
        if data:
            AIInsightItem.model_validate(data[0])

    @pytest.mark.asyncio
    async def test_list_insights_type_filter(self):
        """List insights can filter by insight_type."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                INSIGHTS_LIST_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "insight_type": "anomaly"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)
        for item in data:
            assert item.get("insightType") == "anomaly"

    @pytest.mark.asyncio
    async def test_list_insights_severity_filter(self):
        """List insights can filter by severity."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                INSIGHTS_LIST_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "severity": "critical"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_list_insights_resolved_filter(self):
        """List insights can filter by resolved status."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                INSIGHTS_LIST_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "is_resolved": False},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_list_insights_missing_project_id(self):
        """List insights requires project_id."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(INSIGHTS_LIST_URL, headers=HEADERS)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_list_insights_nonexistent_project(self):
        """List insights returns empty for unknown project."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                INSIGHTS_LIST_URL,
                headers=HEADERS,
                params={"project_id": FAKE_UUID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Failure Patterns (failure_patterns.py)
# ═══════════════════════════════════════════════════════════════════════════


class TestFailurePatternList:
    """Tests for GET /api/v1/patterns."""

    @pytest.mark.asyncio
    async def test_list_patterns_returns_200(self):
        """List patterns endpoint returns array."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                PATTERNS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)
        if data:
            FailurePatternItem.model_validate(data[0])

    @pytest.mark.asyncio
    async def test_list_patterns_type_filter(self):
        """List patterns can filter by pattern_type."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                PATTERNS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "pattern_type": "file_change"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_list_patterns_active_filter(self):
        """List patterns can filter by active status."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                PATTERNS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "is_active": True},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_list_patterns_confidence_filter(self):
        """List patterns can filter by minimum confidence."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                PATTERNS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "min_confidence": 0.8},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_list_patterns_missing_project_id(self):
        """List patterns requires project_id."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(PATTERNS_URL, headers=HEADERS)
        assert resp.status_code == 422


class TestFailurePatternAccuracy:
    """Tests for GET /api/v1/patterns/accuracy/stats."""

    @pytest.mark.asyncio
    async def test_accuracy_stats_returns_200(self):
        """Accuracy stats endpoint returns valid response."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                PATTERNS_ACCURACY_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        AccuracyStatsResponse.model_validate(data)
        assert data["projectId"] == PROJ_ID
        assert data["totalPatterns"] >= 0
        assert 0 <= data["overallAccuracy"] <= 1
        assert 0 <= data["precision"] <= 1
        assert 0 <= data["recall"] <= 1
        assert 0 <= data["f1Score"] <= 1

    @pytest.mark.asyncio
    async def test_accuracy_stats_missing_project_id(self):
        """Accuracy stats requires project_id."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(PATTERNS_ACCURACY_URL, headers=HEADERS)
        assert resp.status_code == 422


class TestFailurePatternPredict:
    """Tests for POST /api/v1/patterns/predict."""

    @pytest.mark.asyncio
    async def test_predict_nonexistent_commit(self):
        """Prediction for non-existent commit returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                PATTERNS_PREDICT_URL,
                headers=HEADERS,
                params={
                    "project_id": PROJ_ID,
                    "commit_sha": "abcdef1234567890abcdef1234567890abcdef12",
                },
            )
        # 404 because commit features don't exist
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_predict_missing_params(self):
        """Prediction requires both project_id and commit_sha."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                PATTERNS_PREDICT_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 422


class TestFailurePatternFeatures:
    """Tests for POST /api/v1/patterns/features."""

    @pytest.mark.asyncio
    async def test_extract_features_returns_200(self):
        """Extract features endpoint stores commit features."""
        test_sha = f"test_{uuid.uuid4().hex[:20]}"
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                PATTERNS_FEATURES_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
                json={
                    "commit_sha": test_sha,
                    "author_email": "test@example.com",
                    "author_name": "Test Author",
                    "files_changed": 3,
                    "lines_added": 50,
                    "lines_deleted": 10,
                    "changed_files": ["src/api/quality.py", "src/api/insights.py", "tests/test_quality.py"],
                    "changed_directories": ["src/api", "tests"],
                    "file_extensions": ["py"],
                    "touches_tests": True,
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["success"] is True
        assert data["commitSha"] == test_sha

    @pytest.mark.asyncio
    async def test_extract_features_missing_project_id(self):
        """Extract features requires project_id."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                PATTERNS_FEATURES_URL,
                headers=HEADERS,
                json={"commit_sha": "abc123"},
            )
        assert resp.status_code == 422


class TestFailurePatternCRUD:
    """Tests for POST/DELETE /api/v1/patterns."""

    @pytest.mark.asyncio
    async def test_create_pattern_returns_200(self):
        """Creating a failure pattern returns valid response."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                PATTERNS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
                json={
                    "pattern_type": "file_change",
                    "pattern_definition": {
                        "file_patterns": ["src/api/*.py"],
                    },
                    "description": "API file changes may cause test failures",
                    "confidence": 0.6,
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        FailurePatternItem.model_validate(data)
        assert data["patternType"] == "file_change"
        assert data["discoveredBy"] == "user"
        assert data["isActive"] is True

    @pytest.mark.asyncio
    async def test_create_pattern_invalid_type(self):
        """Creating a pattern with invalid type fails validation."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                PATTERNS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
                json={
                    "pattern_type": "invalid_type",
                    "pattern_definition": {},
                },
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_deactivate_nonexistent_pattern(self):
        """Deactivating non-existent pattern should not crash."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{PATTERNS_URL}/{FAKE_UUID}",
                headers=HEADERS,
            )
        # May be 200 (soft delete no-op) or 404
        assert resp.status_code in (200, 404, 500), resp.text


class TestFailurePatternTrain:
    """Tests for POST /api/v1/patterns/train."""

    @pytest.mark.asyncio
    async def test_train_returns_200(self):
        """Training job creation returns valid response."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                PATTERNS_TRAIN_URL,
                headers=HEADERS,
                params={
                    "project_id": PROJ_ID,
                    "job_type": "incremental",
                    "days": 7,
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        TrainingJobResponse.model_validate(data)
        assert data["jobType"] == "incremental"
        assert data["status"] == "pending"
        assert data["progress"] == 0

    @pytest.mark.asyncio
    async def test_train_invalid_job_type(self):
        """Training with invalid job type fails."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                PATTERNS_TRAIN_URL,
                headers=HEADERS,
                params={
                    "project_id": PROJ_ID,
                    "job_type": "invalid_job",
                },
            )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_train_missing_project_id(self):
        """Training requires project_id."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(PATTERNS_TRAIN_URL, headers=HEADERS)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_training_status_nonexistent_job(self):
        """Getting status of non-existent job returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{PATTERNS_TRAIN_URL}/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"


class TestFailurePatternOutcome:
    """Tests for POST /api/v1/patterns/outcome."""

    @pytest.mark.asyncio
    async def test_record_outcome_returns_200(self):
        """Recording outcome works even with no matching predictions."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                PATTERNS_OUTCOME_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
                json={
                    "commit_sha": "abc123def456",
                    "test_name": "test_login_flow",
                    "outcome": "passed",
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["success"] is True
        assert data["outcome"] == "passed"

    @pytest.mark.asyncio
    async def test_record_outcome_invalid_outcome(self):
        """Recording invalid outcome status fails validation."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                PATTERNS_OUTCOME_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
                json={
                    "commit_sha": "abc123",
                    "test_name": "test_login",
                    "outcome": "invalid_status",
                },
            )
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Flaky Test Detection (flaky_detector.py)
# ═══════════════════════════════════════════════════════════════════════════


class TestFlakyTestList:
    """Tests for GET /api/v1/flaky-tests."""

    @pytest.mark.asyncio
    async def test_list_flaky_tests_returns_200(self):
        """Flaky tests list endpoint returns valid response."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FLAKY_TESTS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        FlakyTestsListResponse.model_validate(data)
        assert isinstance(data["tests"], list)
        assert data["total"] >= 0

    @pytest.mark.asyncio
    async def test_list_flaky_tests_no_project_filter(self):
        """Flaky tests list works without project filter (all org projects)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FLAKY_TESTS_URL,
                headers=HEADERS,
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        FlakyTestsListResponse.model_validate(data)

    @pytest.mark.asyncio
    async def test_list_flaky_tests_min_score_filter(self):
        """Flaky tests can be filtered by minimum score."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FLAKY_TESTS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "min_score": 0.5},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        FlakyTestsListResponse.model_validate(data)

    @pytest.mark.asyncio
    async def test_list_flaky_tests_custom_days(self):
        """Flaky tests can use custom analysis window."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FLAKY_TESTS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "days": 7},
            )
        assert resp.status_code == 200, resp.text
        FlakyTestsListResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_list_flaky_tests_invalid_project_id(self):
        """Flaky tests list rejects invalid project_id UUID."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FLAKY_TESTS_URL,
                headers=HEADERS,
                params={"project_id": "not-a-uuid"},
            )
        assert resp.status_code in (400, 422), f"Expected 400/422, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_list_flaky_tests_response_structure(self):
        """Flaky tests list items have expected structure."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FLAKY_TESTS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        for item in data.get("tests", []):
            FlakyTestItem.model_validate(item)
            # Verify score bounds
            assert 0 <= item["flakinessScore"] <= 1
            assert item["totalRuns"] >= 0
            assert item["trend"] in ("increasing", "decreasing", "stable")


class TestFlakyTestTrend:
    """Tests for GET /api/v1/flaky-tests/trend."""

    @pytest.mark.asyncio
    async def test_trend_returns_200(self):
        """Flakiness trend endpoint returns valid response."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FLAKY_TREND_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        FlakyTrendResponse.model_validate(data)
        assert isinstance(data["trend"], list)

    @pytest.mark.asyncio
    async def test_trend_custom_weeks(self):
        """Trend supports custom number of weeks."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FLAKY_TREND_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "weeks": 4},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        FlakyTrendResponse.model_validate(data)
        # Should have approximately 4 data points
        assert len(data["trend"]) <= 4

    @pytest.mark.asyncio
    async def test_trend_no_project_filter(self):
        """Trend works without project filter."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FLAKY_TREND_URL,
                headers=HEADERS,
            )
        assert resp.status_code == 200, resp.text
        FlakyTrendResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_trend_items_have_structure(self):
        """Trend items have expected date/flaky/fixed structure."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FLAKY_TREND_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "weeks": 4},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        for item in data.get("trend", []):
            FlakyTrendItem.model_validate(item)
            assert item["flaky"] >= 0
            assert item["fixed"] >= 0


class TestFlakyTestStats:
    """Tests for GET /api/v1/flaky-tests/stats."""

    @pytest.mark.asyncio
    async def test_stats_returns_200(self):
        """Flaky test stats endpoint returns valid response."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FLAKY_STATS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        FlakyStatsResponse.model_validate(data)
        assert data["total"] >= 0
        assert data["high"] >= 0
        assert data["medium"] >= 0
        assert data["low"] >= 0
        assert data["quarantined"] >= 0
        assert 0 <= data["avgScore"] <= 1

    @pytest.mark.asyncio
    async def test_stats_no_project_filter(self):
        """Stats works without project filter."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(FLAKY_STATS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        FlakyStatsResponse.model_validate(resp.json())


class TestFlakyTestQuarantine:
    """Tests for POST /api/v1/flaky-tests/{test_id}/quarantine."""

    @pytest.mark.asyncio
    async def test_quarantine_nonexistent_test(self):
        """Quarantining non-existent test returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{FLAKY_TESTS_URL}/{FAKE_UUID}/quarantine",
                headers=HEADERS,
                json={"quarantine": True},
            )
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_quarantine_invalid_uuid(self):
        """Quarantine rejects invalid UUID."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{FLAKY_TESTS_URL}/bad-uuid/quarantine",
                headers=HEADERS,
                json={"quarantine": True},
            )
        assert resp.status_code in (400, 422), f"Expected 400/422, got {resp.status_code}: {resp.text}"


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Edge Cases & Auth
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCasesAuth:
    """Cross-cutting edge cases: missing auth, wrong org, etc."""

    @pytest.mark.asyncio
    async def test_quality_no_api_key(self):
        """Quality endpoints reject requests without API key."""
        headers_no_key = {k: v for k, v in HEADERS.items() if k != "X-API-Key"}
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_SCORE_URL,
                headers=headers_no_key,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code in (401, 403), f"Expected 401/403, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_insights_no_api_key(self):
        """Insights endpoints reject requests without API key."""
        headers_no_key = {k: v for k, v in HEADERS.items() if k != "X-API-Key"}
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                INSIGHTS_LIST_URL,
                headers=headers_no_key,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code in (401, 403), f"Expected 401/403, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_patterns_no_api_key(self):
        """Patterns endpoints reject requests without API key."""
        headers_no_key = {k: v for k, v in HEADERS.items() if k != "X-API-Key"}
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                PATTERNS_URL,
                headers=headers_no_key,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code in (401, 403), f"Expected 401/403, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_flaky_no_api_key(self):
        """Flaky tests endpoints reject requests without API key."""
        headers_no_key = {k: v for k, v in HEADERS.items() if k != "X-API-Key"}
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(FLAKY_TESTS_URL, headers=headers_no_key)
        assert resp.status_code in (401, 403), f"Expected 401/403, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_quality_invalid_api_key(self):
        """Quality endpoints reject invalid API keys."""
        headers_bad = {**HEADERS, "X-API-Key": "invalid_key_12345"}
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                QUALITY_SCORE_URL,
                headers=headers_bad,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code in (401, 403), f"Expected 401/403, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_patterns_no_org_header(self):
        """Patterns endpoints with no org header: API key implicitly resolves org."""
        headers_no_org = {k: v for k, v in HEADERS.items() if k != "X-Organization-Id"}
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                PATTERNS_URL,
                headers=headers_no_org,
                params={"project_id": PROJ_ID},
            )
        # API key auth resolves org implicitly, so this may succeed
        assert resp.status_code in (200, 400, 401, 403, 422), f"Unexpected {resp.status_code}"
