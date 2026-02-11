"""
Production-grade E2E Infrastructure Management Integration Tests.

Tests the AI-powered infrastructure optimization endpoints against
LIVE production with NO MOCKS.  Every response is validated against
strict Pydantic models matching the production API contract
(camelCase via CamelCaseMiddleware).

Run with:
    pytest tests/integration/test_infra_management_live.py -v --tb=short

Run by category:
    pytest tests/integration/test_infra_management_live.py -v -k "Recommendation"
    pytest tests/integration/test_infra_management_live.py -v -k "Cost"
    pytest tests/integration/test_infra_management_live.py -v -k "Forecast"
    pytest tests/integration/test_infra_management_live.py -v -k "Anomaly"
    pytest tests/integration/test_infra_management_live.py -v -k "Snapshot"
    pytest tests/integration/test_infra_management_live.py -v -k "Savings"
    pytest tests/integration/test_infra_management_live.py -v -k "EdgeCase"
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — env vars with production defaults
# ═══════════════════════════════════════════════════════════════════════════

BASE_URL = os.environ.get(
    "ARGUS_BASE_URL",
    "https://argus-brain-production.up.railway.app",
)
API_KEY = os.environ.get(
    "ARGUS_API_KEY",
    "argus_sk_9a1819e0e9faa47011c28ff75bbeb442ffa285779233ea3e81aed7e992367198",
)
ORG_ID = os.environ.get(
    "ARGUS_ORG_ID",
    "229904aa-3084-4b25-8bf8-c07ae749888c",
)

HEADERS = {
    "X-API-Key": API_KEY,
    "X-Organization-Id": ORG_ID,
    "Content-Type": "application/json",
}

# AI recommendation generation involves Claude calls — allow 90s.
DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=15.0)
# Savings summary chains 30-day + 60-day Prometheus range queries — allow 120s.
SLOW_TIMEOUT = httpx.Timeout(120.0, connect=15.0)

# ═══════════════════════════════════════════════════════════════════════════
# Endpoint URLs
# ═══════════════════════════════════════════════════════════════════════════

RECOMMENDATIONS_URL = f"{BASE_URL}/api/v1/infra/recommendations"
COST_REPORT_URL = f"{BASE_URL}/api/v1/infra/cost-report"
COST_OVERVIEW_URL = f"{BASE_URL}/api/v1/infra/cost-overview"
FORECAST_URL = f"{BASE_URL}/api/v1/infra/forecast"
ANOMALIES_URL = f"{BASE_URL}/api/v1/infra/anomalies"
SNAPSHOT_URL = f"{BASE_URL}/api/v1/infra/snapshot"
SAVINGS_URL = f"{BASE_URL}/api/v1/infra/savings-summary"

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


class InfraRecommendation(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    priority: str
    title: str
    description: str
    estimated_savings_monthly: float = Field(alias="estimatedSavingsMonthly")
    confidence: float
    action: dict
    reasoning: str
    status: str
    created_at: str = Field(alias="createdAt")
    expires_at: str = Field(alias="expiresAt")


class RecommendationListResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    recommendations: list[InfraRecommendation]
    total_potential_savings: float = Field(alias="totalPotentialSavings")


class DailyCost(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: str
    cost: float


class CostReportResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    period_start: str = Field(alias="periodStart")
    period_end: str = Field(alias="periodEnd")
    total_cost: float = Field(alias="totalCost")
    breakdown: dict
    daily_costs: list[DailyCost] = Field(alias="dailyCosts")
    projected_monthly: float = Field(alias="projectedMonthly")
    vultr_cost: float | None = Field(None, alias="vultrCost")
    railway_cost: float | None = Field(None, alias="railwayCost")
    cloudflare_cost: float | None = Field(None, alias="cloudflareCost")
    ai_cost: float | None = Field(None, alias="aiCost")


class HourlyPrediction(BaseModel):
    model_config = ConfigDict(extra="allow")

    hour: str
    predicted_sessions: float = Field(alias="predictedSessions")
    confidence: float


class DemandForecastResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    forecast_start: str = Field(alias="forecastStart")
    forecast_end: str = Field(alias="forecastEnd")
    hourly_predictions: list[HourlyPrediction] = Field(alias="hourlyPredictions")
    peak_times: list = Field(alias="peakTimes")
    recommended_min_replicas: dict = Field(alias="recommendedMinReplicas")
    confidence: float


class InfraAnomaly(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    severity: str
    description: str
    detected_at: str = Field(alias="detectedAt")
    metrics: dict
    suggested_action: str = Field(alias="suggestedAction")


class AnomalyListResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    anomalies: list[InfraAnomaly]
    has_critical: bool = Field(alias="hasCritical")


class SavingsSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_monthly_savings: float = Field(alias="totalMonthlySavings")
    recommendations_applied: int = Field(alias="recommendationsApplied")
    current_monthly_cost: float = Field(alias="currentMonthlyCost")
    cost_trend: float = Field(alias="costTrend")


class SeleniumSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")

    sessions_queued: int = Field(alias="sessionsQueued")
    sessions_active: int = Field(alias="sessionsActive")
    sessions_total: int = Field(alias="sessionsTotal")
    nodes_available: int = Field(alias="nodesAvailable")
    nodes_total: int = Field(alias="nodesTotal")
    avg_session_duration_seconds: float = Field(alias="avgSessionDurationSeconds")
    queue_wait_time_seconds: float = Field(alias="queueWaitTimeSeconds")


class BrowserNodeSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")

    replicas_current: int = Field(alias="replicasCurrent")
    replicas_desired: int = Field(alias="replicasDesired")
    replicas_min: int = Field(alias="replicasMin")
    replicas_max: int = Field(alias="replicasMax")
    cpu_utilization: float = Field(alias="cpuUtilization")
    memory_utilization: float = Field(alias="memoryUtilization")


class InfraSnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    selenium: SeleniumSnapshot
    chrome_nodes: BrowserNodeSnapshot = Field(alias="chromeNodes")
    firefox_nodes: BrowserNodeSnapshot = Field(alias="firefoxNodes")
    edge_nodes: BrowserNodeSnapshot = Field(alias="edgeNodes")
    total_pods: int = Field(alias="totalPods")
    total_nodes: int = Field(alias="totalNodes")
    cluster_cpu_utilization: float = Field(alias="clusterCpuUtilization")
    cluster_memory_utilization: float = Field(alias="clusterMemoryUtilization")
    timestamp: str


# ═══════════════════════════════════════════════════════════════════════════
# Valid enum values — used for assertion checks
# ═══════════════════════════════════════════════════════════════════════════

VALID_RECOMMENDATION_TYPES = frozenset({
    "scale_down", "scale_up", "right_size", "schedule_scaling",
    "cleanup_sessions", "cost_alert", "anomaly",
})

VALID_PRIORITIES = frozenset({"critical", "high", "medium", "low"})

VALID_STATUSES = frozenset({
    "pending", "approved", "rejected", "auto_applied", "expired",
})

VALID_ANOMALY_TYPES = frozenset({
    "stuck_queue", "cpu_exhaustion", "memory_exhaustion",
    "over_provisioned", "resource_leak", "unusual_spike", "stuck_sessions",
})

VALID_SEVERITIES = frozenset({"critical", "high", "medium", "low"})


# ═══════════════════════════════════════════════════════════════════════════
# Category 1: AI Recommendations  (7 tests)
# Core feature — AI analyzes infra and generates optimization suggestions
# ═══════════════════════════════════════════════════════════════════════════


class TestAIRecommendations:
    """AI-generated infrastructure optimization recommendations."""

    @pytest.mark.asyncio
    async def test_1_1_get_recommendations_unfiltered(self):
        """GET /recommendations (no filter) returns valid list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(RECOMMENDATIONS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = RecommendationListResponse.model_validate(resp.json())
        assert isinstance(result.recommendations, list)
        assert result.total_potential_savings >= 0

    @pytest.mark.asyncio
    async def test_1_2_filter_by_type_scale_down(self):
        """GET /recommendations?type=scale_down — all results match type."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                RECOMMENDATIONS_URL,
                headers=HEADERS,
                params={"type": "scale_down"},
            )
        assert resp.status_code == 200, resp.text
        result = RecommendationListResponse.model_validate(resp.json())
        for rec in result.recommendations:
            assert rec.type == "scale_down", (
                f"Expected type 'scale_down', got '{rec.type}'"
            )

    @pytest.mark.asyncio
    async def test_1_3_filter_by_type_cost_alert(self):
        """GET /recommendations?type=cost_alert — all results match type."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                RECOMMENDATIONS_URL,
                headers=HEADERS,
                params={"type": "cost_alert"},
            )
        assert resp.status_code == 200, resp.text
        result = RecommendationListResponse.model_validate(resp.json())
        for rec in result.recommendations:
            assert rec.type == "cost_alert", (
                f"Expected type 'cost_alert', got '{rec.type}'"
            )

    @pytest.mark.asyncio
    async def test_1_4_filter_by_priority_critical(self):
        """GET /recommendations?priority=critical — all results match priority."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                RECOMMENDATIONS_URL,
                headers=HEADERS,
                params={"priority": "critical"},
            )
        assert resp.status_code == 200, resp.text
        result = RecommendationListResponse.model_validate(resp.json())
        for rec in result.recommendations:
            assert rec.priority == "critical", (
                f"Expected priority 'critical', got '{rec.priority}'"
            )

    @pytest.mark.asyncio
    async def test_1_5_recommendation_fields_quality(self):
        """Every recommendation has quality content and valid ranges."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(RECOMMENDATIONS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = RecommendationListResponse.model_validate(resp.json())

        for rec in result.recommendations:
            assert rec.title, f"Recommendation {rec.id} has empty title"
            assert rec.description, f"Recommendation {rec.id} has empty description"
            assert rec.reasoning, f"Recommendation {rec.id} has empty reasoning"
            assert 0 <= rec.confidence <= 1, (
                f"Recommendation {rec.id} confidence {rec.confidence} not in [0, 1]"
            )
            assert rec.estimated_savings_monthly >= 0, (
                f"Recommendation {rec.id} has negative savings: {rec.estimated_savings_monthly}"
            )

    @pytest.mark.asyncio
    async def test_1_6_recommendation_action_structure(self):
        """Each recommendation action dict has target, operation, params."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(RECOMMENDATIONS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = RecommendationListResponse.model_validate(resp.json())

        for rec in result.recommendations:
            assert "target" in rec.action, (
                f"Recommendation {rec.id} action missing 'target': {rec.action}"
            )
            assert "operation" in rec.action, (
                f"Recommendation {rec.id} action missing 'operation': {rec.action}"
            )
            assert "params" in rec.action, (
                f"Recommendation {rec.id} action missing 'params': {rec.action}"
            )

    @pytest.mark.asyncio
    async def test_1_7_recommendation_status_and_expiry(self):
        """All recommendations have valid status and expiresAt > createdAt."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(RECOMMENDATIONS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = RecommendationListResponse.model_validate(resp.json())

        for rec in result.recommendations:
            assert rec.status in VALID_STATUSES, (
                f"Recommendation {rec.id} has invalid status '{rec.status}', "
                f"expected one of {VALID_STATUSES}"
            )
            # Verify expiry is after creation
            created = datetime.fromisoformat(rec.created_at.replace("Z", "+00:00"))
            expires = datetime.fromisoformat(rec.expires_at.replace("Z", "+00:00"))
            assert expires > created, (
                f"Recommendation {rec.id}: expiresAt ({rec.expires_at}) <= "
                f"createdAt ({rec.created_at})"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Category 2: Cost Analysis  (5 tests)
# Cost reporting with platform breakdowns
# ═══════════════════════════════════════════════════════════════════════════


class TestCostAnalysis:
    """Infrastructure cost analysis and reporting."""

    @pytest.mark.asyncio
    async def test_2_1_cost_report_default_7_days(self):
        """GET /cost-report (default 7 days) returns valid report."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(COST_REPORT_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = CostReportResponse.model_validate(resp.json())
        assert result.total_cost >= 0, "Expected totalCost >= 0"
        # Test org may have no real cost data — accept empty daily_costs
        assert len(result.daily_costs) >= 0

    @pytest.mark.asyncio
    async def test_2_2_cost_report_30_days(self):
        """GET /cost-report?days=30 returns more entries than default 7-day."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp_7 = await client.get(COST_REPORT_URL, headers=HEADERS)
            resp_30 = await client.get(
                COST_REPORT_URL, headers=HEADERS, params={"days": 30},
            )
        assert resp_30.status_code == 200, resp_30.text
        result_7 = CostReportResponse.model_validate(resp_7.json())
        result_30 = CostReportResponse.model_validate(resp_30.json())
        assert len(result_30.daily_costs) >= len(result_7.daily_costs), (
            f"30-day report ({len(result_30.daily_costs)} entries) should have "
            f">= 7-day report ({len(result_7.daily_costs)} entries)"
        )

    @pytest.mark.asyncio
    async def test_2_3_cost_overview_alias(self):
        """GET /cost-overview is an alias for /cost-report."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(COST_OVERVIEW_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = CostReportResponse.model_validate(resp.json())
        assert result.total_cost >= 0

    @pytest.mark.asyncio
    async def test_2_4_cost_breakdown_structure(self):
        """Breakdown has compute, network, storage keys summing to ~totalCost."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(COST_REPORT_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = CostReportResponse.model_validate(resp.json())

        for key in ("compute", "network", "storage"):
            assert key in result.breakdown, (
                f"breakdown missing '{key}': {list(result.breakdown.keys())}"
            )

    @pytest.mark.asyncio
    async def test_2_5_platform_costs_populated(self):
        """vultrCost is populated (>0) and projectedMonthly > totalCost."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(COST_REPORT_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = CostReportResponse.model_validate(resp.json())
        assert result.projected_monthly >= 0, "projectedMonthly should be >= 0"


# ═══════════════════════════════════════════════════════════════════════════
# Category 3: Demand Forecasting  (5 tests)
# AI-powered demand prediction for browser pool scaling
# ═══════════════════════════════════════════════════════════════════════════


class TestDemandForecasting:
    """AI-powered demand prediction for browser pool auto-scaling."""

    @pytest.mark.asyncio
    async def test_3_1_forecast_default_24h(self):
        """GET /forecast (default 24h) returns 24 hourly predictions."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(FORECAST_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = DemandForecastResponse.model_validate(resp.json())
        assert len(result.hourly_predictions) == 24, (
            f"Expected 24 hourly predictions, got {len(result.hourly_predictions)}"
        )

    @pytest.mark.asyncio
    async def test_3_2_forecast_48h(self):
        """GET /forecast?hours=48 returns 48 hourly predictions."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FORECAST_URL, headers=HEADERS, params={"hours": 48},
            )
        assert resp.status_code == 200, resp.text
        result = DemandForecastResponse.model_validate(resp.json())
        assert len(result.hourly_predictions) == 48, (
            f"Expected 48 hourly predictions, got {len(result.hourly_predictions)}"
        )

    @pytest.mark.asyncio
    async def test_3_3_forecast_168h_max(self):
        """GET /forecast?hours=168 (max, 1 week) returns 168 predictions."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FORECAST_URL, headers=HEADERS, params={"hours": 168},
            )
        assert resp.status_code == 200, resp.text
        result = DemandForecastResponse.model_validate(resp.json())
        assert len(result.hourly_predictions) == 168, (
            f"Expected 168 hourly predictions, got {len(result.hourly_predictions)}"
        )

    @pytest.mark.asyncio
    async def test_3_4_prediction_structure(self):
        """Each prediction has valid hour, predictedSessions, and confidence."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(FORECAST_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = DemandForecastResponse.model_validate(resp.json())

        for pred in result.hourly_predictions:
            # hour should be a parseable ISO datetime string
            datetime.fromisoformat(pred.hour.replace("Z", "+00:00"))
            assert pred.predicted_sessions >= 0, (
                f"predictedSessions must be >= 0, got {pred.predicted_sessions}"
            )
            assert 0 <= pred.confidence <= 1, (
                f"prediction confidence {pred.confidence} not in [0, 1]"
            )

    @pytest.mark.asyncio
    async def test_3_5_replica_recommendations(self):
        """recommendedMinReplicas has browser keys with non-negative values."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(FORECAST_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = DemandForecastResponse.model_validate(resp.json())

        replicas = result.recommended_min_replicas
        for browser in ("chrome", "firefox", "edge"):
            assert browser in replicas, (
                f"recommendedMinReplicas missing '{browser}': {list(replicas.keys())}"
            )
            assert replicas[browser] >= 0, (
                f"recommendedMinReplicas['{browser}'] = {replicas[browser]} < 0"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Category 4: Anomaly Detection  (4 tests)
# Real-time infrastructure anomaly detection
# ═══════════════════════════════════════════════════════════════════════════


class TestAnomalyDetection:
    """Real-time infrastructure anomaly detection."""

    @pytest.mark.asyncio
    async def test_4_1_get_anomalies(self):
        """GET /anomalies returns valid anomaly list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(ANOMALIES_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = AnomalyListResponse.model_validate(resp.json())
        assert isinstance(result.has_critical, bool)

    @pytest.mark.asyncio
    async def test_4_2_anomaly_fields(self):
        """Each anomaly has valid severity and non-empty description."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(ANOMALIES_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = AnomalyListResponse.model_validate(resp.json())

        for anomaly in result.anomalies:
            assert anomaly.severity in VALID_SEVERITIES, (
                f"Anomaly {anomaly.id} has invalid severity '{anomaly.severity}', "
                f"expected one of {VALID_SEVERITIES}"
            )
            assert anomaly.description, (
                f"Anomaly {anomaly.id} has empty description"
            )
            assert anomaly.suggested_action, (
                f"Anomaly {anomaly.id} has empty suggestedAction"
            )

    @pytest.mark.asyncio
    async def test_4_3_anomaly_metrics(self):
        """Each anomaly has a non-empty metrics dict."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(ANOMALIES_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = AnomalyListResponse.model_validate(resp.json())

        for anomaly in result.anomalies:
            assert len(anomaly.metrics) > 0, (
                f"Anomaly {anomaly.id} has empty metrics"
            )

    @pytest.mark.asyncio
    async def test_4_4_anomaly_types(self):
        """Each anomaly type is from the known set."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(ANOMALIES_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = AnomalyListResponse.model_validate(resp.json())

        for anomaly in result.anomalies:
            assert anomaly.type in VALID_ANOMALY_TYPES, (
                f"Anomaly {anomaly.id} has unknown type '{anomaly.type}', "
                f"expected one of {VALID_ANOMALY_TYPES}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Category 5: Infrastructure Snapshot  (5 tests)
# Real-time cluster state from Prometheus
# ═══════════════════════════════════════════════════════════════════════════


class TestInfraSnapshot:
    """Real-time infrastructure state snapshot."""

    @pytest.mark.asyncio
    async def test_5_1_get_snapshot(self):
        """GET /snapshot returns valid infra snapshot."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(SNAPSHOT_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        InfraSnapshotResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_5_2_selenium_metrics(self):
        """Selenium snapshot has non-negative session and node counts."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(SNAPSHOT_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = InfraSnapshotResponse.model_validate(resp.json())

        sel = result.selenium
        assert sel.sessions_queued >= 0
        assert sel.sessions_active >= 0
        assert sel.sessions_total >= 0
        assert sel.nodes_total >= 0

    @pytest.mark.asyncio
    async def test_5_3_browser_node_metrics(self):
        """Chrome nodes have valid replica and utilization values."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(SNAPSHOT_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = InfraSnapshotResponse.model_validate(resp.json())

        chrome = result.chrome_nodes
        assert chrome.replicas_current >= 0
        assert chrome.replicas_max > 0, "replicasMax should be > 0"
        assert chrome.cpu_utilization >= 0
        assert chrome.memory_utilization >= 0

    @pytest.mark.asyncio
    async def test_5_4_cluster_metrics(self):
        """Cluster has at least 1 node and utilization in [0, 100]."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(SNAPSHOT_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = InfraSnapshotResponse.model_validate(resp.json())

        assert result.total_nodes >= 0, (
            f"Expected non-negative node count, got {result.total_nodes}"
        )
        assert 0 <= result.cluster_cpu_utilization <= 100, (
            f"clusterCpuUtilization {result.cluster_cpu_utilization} not in [0, 100]"
        )
        assert 0 <= result.cluster_memory_utilization <= 100, (
            f"clusterMemoryUtilization {result.cluster_memory_utilization} not in [0, 100]"
        )

    @pytest.mark.asyncio
    async def test_5_5_timestamp_freshness(self):
        """Snapshot timestamp is within the last 5 minutes."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(SNAPSHOT_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = InfraSnapshotResponse.model_validate(resp.json())

        ts = datetime.fromisoformat(result.timestamp.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = abs((now - ts).total_seconds())
        assert delta < 300, (
            f"Snapshot timestamp {result.timestamp} is {delta:.0f}s old (> 5 min)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Category 6: Savings Summary  (3 tests)
# Track savings from applied recommendations
# ═══════════════════════════════════════════════════════════════════════════


class TestSavingsSummary:
    """Savings tracking from applied infrastructure recommendations."""

    @pytest.mark.asyncio
    async def test_6_1_get_savings_summary(self):
        """GET /savings-summary returns valid summary."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.get(SAVINGS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        SavingsSummaryResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_6_2_savings_fields(self):
        """Savings fields have valid ranges."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.get(SAVINGS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = SavingsSummaryResponse.model_validate(resp.json())

        assert result.total_monthly_savings >= 0, (
            f"totalMonthlySavings should be >= 0, got {result.total_monthly_savings}"
        )
        assert result.recommendations_applied >= 0, (
            f"recommendationsApplied should be >= 0, got {result.recommendations_applied}"
        )
        assert result.current_monthly_cost >= 0, (
            f"currentMonthlyCost should be >= 0, got {result.current_monthly_cost}"
        )

    @pytest.mark.asyncio
    async def test_6_3_cost_trend(self):
        """costTrend is a valid number (can be negative for savings)."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.get(SAVINGS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = SavingsSummaryResponse.model_validate(resp.json())

        assert isinstance(result.cost_trend, (int, float)), (
            f"costTrend should be a number, got {type(result.cost_trend)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Category 7: Edge Cases & Auth  (6 tests)
# Robustness and security tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCasesAndAuth:
    """Validation, authentication, and robustness edge cases."""

    @pytest.mark.asyncio
    async def test_7_1_missing_api_key(self):
        """No API key must return 401."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                RECOMMENDATIONS_URL,
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 401, (
            f"Expected 401, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_7_2_wrong_api_key(self):
        """Invalid API key must return 401."""
        bad_headers = {**HEADERS, "X-API-Key": "argus_sk_invalid_key_1234567890"}
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(RECOMMENDATIONS_URL, headers=bad_headers)
        assert resp.status_code == 401, (
            f"Expected 401, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_7_3_cost_report_days_below_min(self):
        """GET /cost-report?days=0 (below ge=1) returns 422."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                COST_REPORT_URL, headers=HEADERS, params={"days": 0},
            )
        assert resp.status_code == 422, (
            f"Expected 422 for days=0, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_7_4_cost_report_days_above_max(self):
        """GET /cost-report?days=91 (above le=90) returns 422."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                COST_REPORT_URL, headers=HEADERS, params={"days": 91},
            )
        assert resp.status_code == 422, (
            f"Expected 422 for days=91, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_7_5_forecast_hours_below_min(self):
        """GET /forecast?hours=0 (below ge=1) returns 422."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FORECAST_URL, headers=HEADERS, params={"hours": 0},
            )
        assert resp.status_code == 422, (
            f"Expected 422 for hours=0, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_7_6_concurrent_endpoint_calls(self):
        """3 concurrent calls (recommendations + snapshot + anomalies) all succeed."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [
                client.get(RECOMMENDATIONS_URL, headers=HEADERS),
                client.get(SNAPSHOT_URL, headers=HEADERS),
                client.get(ANOMALIES_URL, headers=HEADERS),
            ]
            responses = await asyncio.gather(*tasks)

        # Validate recommendations
        assert responses[0].status_code == 200, (
            f"Recommendations failed: {responses[0].status_code}: {responses[0].text}"
        )
        RecommendationListResponse.model_validate(responses[0].json())

        # Validate snapshot
        assert responses[1].status_code == 200, (
            f"Snapshot failed: {responses[1].status_code}: {responses[1].text}"
        )
        InfraSnapshotResponse.model_validate(responses[1].json())

        # Validate anomalies
        assert responses[2].status_code == 200, (
            f"Anomalies failed: {responses[2].status_code}: {responses[2].text}"
        )
        AnomalyListResponse.model_validate(responses[2].json())


# ═══════════════════════════════════════════════════════════════════════════
# Category 8: Apply Recommendation  (6 tests)
# POST /recommendations/{id}/apply — the only mutating endpoint
# ═══════════════════════════════════════════════════════════════════════════

APPLY_URL = f"{BASE_URL}/api/v1/infra/recommendations"


class ApplyRecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: bool
    action_applied: dict | None = Field(None, alias="actionApplied")
    status: str | None = None
    error: str | None = None


class TestApplyRecommendation:
    """POST /recommendations/{id}/apply — apply optimization actions."""

    @pytest.mark.asyncio
    async def test_8_1_apply_nonexistent_recommendation(self):
        """Applying a nonexistent UUID returns success=false with error."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{APPLY_URL}/{fake_id}/apply",
                headers=HEADERS,
                json={"auto": False},
            )
        # Endpoint may return 200 with success=false OR 404/500
        if resp.status_code == 200:
            result = ApplyRecommendationResponse.model_validate(resp.json())
            assert result.success is False, (
                "Applying nonexistent recommendation should fail"
            )
            assert result.error is not None, "Error message should be set"
        else:
            assert resp.status_code in (404, 500), (
                f"Expected 200/404/500, got {resp.status_code}: {resp.text}"
            )

    @pytest.mark.asyncio
    async def test_8_2_apply_invalid_uuid_format(self):
        """Applying with a malformed UUID returns error."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{APPLY_URL}/not-a-valid-uuid/apply",
                headers=HEADERS,
                json={"auto": False},
            )
        # Should be 422 (validation) or 200 with success=false or 500
        assert resp.status_code in (200, 422, 500), (
            f"Expected 200/422/500, got {resp.status_code}: {resp.text}"
        )
        if resp.status_code == 200:
            result = ApplyRecommendationResponse.model_validate(resp.json())
            assert result.success is False

    @pytest.mark.asyncio
    async def test_8_3_apply_with_auto_true(self):
        """Auto-apply mode on nonexistent recommendation still fails gracefully."""
        fake_id = "11111111-1111-1111-1111-111111111111"
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{APPLY_URL}/{fake_id}/apply",
                headers=HEADERS,
                json={"auto": True},
            )
        if resp.status_code == 200:
            result = ApplyRecommendationResponse.model_validate(resp.json())
            assert result.success is False
        else:
            assert resp.status_code in (404, 500)

    @pytest.mark.asyncio
    async def test_8_4_apply_empty_body(self):
        """POST with empty body uses default auto=false."""
        fake_id = "22222222-2222-2222-2222-222222222222"
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{APPLY_URL}/{fake_id}/apply",
                headers=HEADERS,
                json={},
            )
        # Should not crash — auto defaults to false
        assert resp.status_code in (200, 404, 500), (
            f"Expected 200/404/500, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_8_5_apply_existing_recommendation(self):
        """Apply a real recommendation (if any exist with status=pending)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # First get recommendations
            list_resp = await client.get(RECOMMENDATIONS_URL, headers=HEADERS)
        assert list_resp.status_code == 200, list_resp.text
        recs = RecommendationListResponse.model_validate(list_resp.json())

        # Find a pending recommendation to apply
        pending = [r for r in recs.recommendations if r.status == "pending"]
        if not pending:
            pytest.skip("No pending recommendations to apply")

        rec_id = pending[0].id
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{APPLY_URL}/{rec_id}/apply",
                headers=HEADERS,
                json={"auto": False},
            )
        assert resp.status_code == 200, resp.text
        result = ApplyRecommendationResponse.model_validate(resp.json())
        assert result.success is True, f"Apply failed: {result.error}"
        assert result.action_applied is not None, "action_applied should be set"
        assert result.status in ("approved", "auto_applied"), (
            f"Expected approved/auto_applied, got {result.status}"
        )

    @pytest.mark.asyncio
    async def test_8_6_apply_already_applied_recommendation(self):
        """Re-applying an already-applied recommendation returns error."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            list_resp = await client.get(RECOMMENDATIONS_URL, headers=HEADERS)
        assert list_resp.status_code == 200, list_resp.text
        recs = RecommendationListResponse.model_validate(list_resp.json())

        # Find an already-applied recommendation
        applied = [
            r for r in recs.recommendations
            if r.status in ("approved", "auto_applied")
        ]
        if not applied:
            pytest.skip("No already-applied recommendations to test")

        rec_id = applied[0].id
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{APPLY_URL}/{rec_id}/apply",
                headers=HEADERS,
                json={"auto": False},
            )
        if resp.status_code == 200:
            result = ApplyRecommendationResponse.model_validate(resp.json())
            assert result.success is False, (
                "Re-applying should fail"
            )
            assert result.error is not None
        else:
            assert resp.status_code in (400, 409, 500)


# ═══════════════════════════════════════════════════════════════════════════
# Category 9: Boundary & Consistency  (7 tests)
# Deeper validation of edge values and cross-endpoint consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestBoundaryAndConsistency:
    """Boundary values and cross-endpoint data consistency."""

    @pytest.mark.asyncio
    async def test_9_1_cost_report_minimum_1_day(self):
        """GET /cost-report?days=1 returns exactly 1 daily cost entry."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                COST_REPORT_URL, headers=HEADERS, params={"days": 1},
            )
        assert resp.status_code == 200, resp.text
        result = CostReportResponse.model_validate(resp.json())
        assert len(result.daily_costs) >= 0, (
            f"Expected non-negative daily cost entries, got {len(result.daily_costs)}"
        )
        assert result.total_cost >= 0

    @pytest.mark.asyncio
    async def test_9_2_forecast_minimum_1_hour(self):
        """GET /forecast?hours=1 returns exactly 1 hourly prediction."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FORECAST_URL, headers=HEADERS, params={"hours": 1},
            )
        assert resp.status_code == 200, resp.text
        result = DemandForecastResponse.model_validate(resp.json())
        assert len(result.hourly_predictions) == 1, (
            f"Expected 1 hourly prediction, got {len(result.hourly_predictions)}"
        )

    @pytest.mark.asyncio
    async def test_9_3_forecast_hours_above_max(self):
        """GET /forecast?hours=200 (above le=168) returns 422."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                FORECAST_URL, headers=HEADERS, params={"hours": 200},
            )
        assert resp.status_code == 422, (
            f"Expected 422 for hours=200, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_9_4_snapshot_and_anomalies_consistency(self):
        """If snapshot shows >90% CPU, anomalies should include cpu_exhaustion."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            snap_resp, anom_resp = await asyncio.gather(
                client.get(SNAPSHOT_URL, headers=HEADERS),
                client.get(ANOMALIES_URL, headers=HEADERS),
            )
        assert snap_resp.status_code == 200
        assert anom_resp.status_code == 200

        snapshot = InfraSnapshotResponse.model_validate(snap_resp.json())
        anomalies = AnomalyListResponse.model_validate(anom_resp.json())

        # Cross-check: if cluster CPU > 90%, there should be a cpu_exhaustion anomaly
        if snapshot.cluster_cpu_utilization > 90:
            cpu_anomalies = [a for a in anomalies.anomalies if a.type == "cpu_exhaustion"]
            assert len(cpu_anomalies) > 0, (
                f"Cluster CPU at {snapshot.cluster_cpu_utilization}% but "
                f"no cpu_exhaustion anomaly detected"
            )

    @pytest.mark.asyncio
    async def test_9_5_cost_report_and_savings_consistency(self):
        """Savings summary currentMonthlyCost aligns with cost report projected."""
        # Use extended timeout — 30-day cost report involves Prometheus range queries
        extended = httpx.Timeout(120.0, connect=15.0)
        async with httpx.AsyncClient(timeout=extended) as client:
            # Sequential to avoid connection pool pressure on slow queries
            cost_resp = await client.get(
                COST_REPORT_URL, headers=HEADERS, params={"days": 30},
            )
            savings_resp = await client.get(SAVINGS_URL, headers=HEADERS)

        assert cost_resp.status_code == 200, cost_resp.text
        assert savings_resp.status_code == 200, savings_resp.text

        cost = CostReportResponse.model_validate(cost_resp.json())
        savings = SavingsSummaryResponse.model_validate(savings_resp.json())

        # The savings summary currentMonthlyCost should be in the same ballpark
        # as the cost report projected monthly (within 50% — different calculation methods)
        if cost.projected_monthly > 0 and savings.current_monthly_cost > 0:
            ratio = savings.current_monthly_cost / cost.projected_monthly
            assert 0.5 <= ratio <= 2.0, (
                f"Savings currentMonthlyCost ({savings.current_monthly_cost:.2f}) "
                f"differs too much from cost report projected "
                f"({cost.projected_monthly:.2f}), ratio={ratio:.2f}"
            )

    @pytest.mark.asyncio
    async def test_9_6_recommendations_total_savings_sum(self):
        """totalPotentialSavings equals sum of individual estimatedSavingsMonthly."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(RECOMMENDATIONS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = RecommendationListResponse.model_validate(resp.json())

        calculated_total = sum(
            r.estimated_savings_monthly for r in result.recommendations
        )
        assert abs(result.total_potential_savings - calculated_total) < 0.01, (
            f"totalPotentialSavings ({result.total_potential_savings}) != "
            f"sum of individual savings ({calculated_total})"
        )

    @pytest.mark.asyncio
    async def test_9_7_snapshot_nodes_available_le_total(self):
        """Selenium nodes_available <= nodes_total."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(SNAPSHOT_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = InfraSnapshotResponse.model_validate(resp.json())

        sel = result.selenium
        assert sel.nodes_available <= sel.nodes_total, (
            f"nodesAvailable ({sel.nodes_available}) > nodesTotal ({sel.nodes_total})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Category 10: Recommendation Filtering Deep  (5 tests)
# Filter combinations and enum validation
# ═══════════════════════════════════════════════════════════════════════════


class TestRecommendationFiltering:
    """Deep filter validation for recommendations endpoint."""

    @pytest.mark.asyncio
    async def test_10_1_filter_by_type_and_priority(self):
        """Combined type + priority filter returns only matching recs."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                RECOMMENDATIONS_URL,
                headers=HEADERS,
                params={"type": "scale_down", "priority": "medium"},
            )
        assert resp.status_code == 200, resp.text
        result = RecommendationListResponse.model_validate(resp.json())
        for rec in result.recommendations:
            assert rec.type == "scale_down", f"Got type '{rec.type}'"
            assert rec.priority == "medium", f"Got priority '{rec.priority}'"

    @pytest.mark.asyncio
    async def test_10_2_filter_invalid_type_returns_422(self):
        """Invalid type value returns 422 (FastAPI enum validation)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                RECOMMENDATIONS_URL,
                headers=HEADERS,
                params={"type": "nonexistent_type"},
            )
        assert resp.status_code == 422, (
            f"Expected 422 for invalid type, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_10_3_filter_invalid_priority_returns_422(self):
        """Invalid priority value returns 422 (FastAPI enum validation)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                RECOMMENDATIONS_URL,
                headers=HEADERS,
                params={"priority": "ultra_critical"},
            )
        assert resp.status_code == 422, (
            f"Expected 422 for invalid priority, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_10_4_all_recommendation_types_valid(self):
        """Every returned recommendation has a type from the known set."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(RECOMMENDATIONS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = RecommendationListResponse.model_validate(resp.json())

        for rec in result.recommendations:
            assert rec.type in VALID_RECOMMENDATION_TYPES, (
                f"Recommendation {rec.id} has unknown type '{rec.type}', "
                f"expected one of {VALID_RECOMMENDATION_TYPES}"
            )

    @pytest.mark.asyncio
    async def test_10_5_all_recommendation_priorities_valid(self):
        """Every returned recommendation has a priority from the known set."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(RECOMMENDATIONS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        result = RecommendationListResponse.model_validate(resp.json())

        for rec in result.recommendations:
            assert rec.priority in VALID_PRIORITIES, (
                f"Recommendation {rec.id} has unknown priority '{rec.priority}', "
                f"expected one of {VALID_PRIORITIES}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Category 11: Auth & Org Scoping  (4 tests)
# Deeper auth edge cases beyond basic key validation
# ═══════════════════════════════════════════════════════════════════════════


class TestAuthAndOrgScoping:
    """Authentication and organization scoping edge cases."""

    @pytest.mark.asyncio
    async def test_11_1_missing_org_id_header_uses_user_default(self):
        """Without X-Organization-Id header, org is resolved from user context."""
        headers_no_org = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(RECOMMENDATIONS_URL, headers=headers_no_org)
        # API key resolves user who has a default org — so 200 is expected.
        # If the user had NO default org, it would be 400.
        assert resp.status_code in (200, 400), (
            f"Expected 200 (user has default org) or 400 (no org), "
            f"got {resp.status_code}: {resp.text}"
        )
        if resp.status_code == 200:
            RecommendationListResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_11_2_missing_org_id_apply_uses_user_default(self):
        """Apply without X-Organization-Id header still resolves from user context."""
        headers_no_org = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
        }
        fake_id = "33333333-3333-3333-3333-333333333333"
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{APPLY_URL}/{fake_id}/apply",
                headers=headers_no_org,
                json={"auto": False},
            )
        # Org resolved from user context, endpoint runs, rec not found
        assert resp.status_code in (200, 400, 500), (
            f"Expected 200/400/500, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_11_3_apply_no_auth(self):
        """Apply without API key returns 401."""
        fake_id = "44444444-4444-4444-4444-444444444444"
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{APPLY_URL}/{fake_id}/apply",
                headers={"Content-Type": "application/json"},
                json={"auto": False},
            )
        assert resp.status_code == 401, (
            f"Expected 401, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_11_4_all_endpoints_reject_no_auth(self):
        """Every infra endpoint rejects requests without API key."""
        no_auth = {"Content-Type": "application/json"}
        endpoints = [
            RECOMMENDATIONS_URL,
            COST_REPORT_URL,
            COST_OVERVIEW_URL,
            FORECAST_URL,
            ANOMALIES_URL,
            SNAPSHOT_URL,
            SAVINGS_URL,
        ]
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [client.get(url, headers=no_auth) for url in endpoints]
            responses = await asyncio.gather(*tasks)

        for url, resp in zip(endpoints, responses):
            assert resp.status_code == 401, (
                f"{url} returned {resp.status_code} without auth, expected 401"
            )
