"""Infrastructure Optimization API endpoints.

Provides endpoints for AI-driven infrastructure optimization:
- GET /api/v1/infra/recommendations - Get optimization recommendations
- GET /api/v1/infra/cost-report - Get cost analysis report
- GET /api/v1/infra/forecast - Get demand forecast
- GET /api/v1/infra/anomalies - Get detected anomalies
- POST /api/v1/infra/recommendations/{id}/apply - Apply a recommendation
- GET /api/v1/infra/savings-summary - Get savings summary
- GET /api/v1/infra/snapshot - Get current infrastructure state
"""


import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.context import require_organization_id
from src.services.infra_optimizer import (
    AIInfraOptimizer,
    RecommendationPriority,
    RecommendationType,
    create_infra_optimizer,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/infra", tags=["Infrastructure Optimization"])


# ============================================================================
# Response Models
# ============================================================================

class RecommendationResponse(BaseModel):
    """Infrastructure recommendation response."""

    id: str
    type: str
    priority: str
    title: str
    description: str
    estimated_savings_monthly: float
    confidence: float
    action: dict
    reasoning: str
    status: str
    created_at: str
    expires_at: str


class RecommendationListResponse(BaseModel):
    """List of recommendations."""

    recommendations: list[RecommendationResponse]
    total_potential_savings: float


class CostReportResponse(BaseModel):
    """Cost report response."""

    period_start: str
    period_end: str
    total_cost: float
    breakdown: dict[str, float]
    daily_costs: list[dict]
    projected_monthly: float
    # Platform-specific costs
    vultr_cost: float | None = None
    railway_cost: float | None = None
    cloudflare_cost: float | None = None
    ai_cost: float | None = None


class DemandForecastResponse(BaseModel):
    """Demand forecast response."""

    forecast_start: str
    forecast_end: str
    hourly_predictions: list[dict]
    peak_times: list[str]
    recommended_min_replicas: dict[str, int]
    confidence: float


class AnomalyResponse(BaseModel):
    """Anomaly detection response."""

    id: str
    type: str
    severity: str
    description: str
    detected_at: str
    metrics: dict
    suggested_action: str


class AnomalyListResponse(BaseModel):
    """List of anomalies."""

    anomalies: list[AnomalyResponse]
    has_critical: bool


class SavingsSummaryResponse(BaseModel):
    """Savings summary response."""

    total_monthly_savings: float
    recommendations_applied: int
    current_monthly_cost: float
    cost_trend: float  # Percentage change vs last period


class InfraSnapshotResponse(BaseModel):
    """Infrastructure snapshot response."""

    selenium: dict
    chrome_nodes: dict
    firefox_nodes: dict
    edge_nodes: dict
    total_pods: int
    total_nodes: int
    cluster_cpu_utilization: float
    cluster_memory_utilization: float
    timestamp: str


class ApplyRecommendationRequest(BaseModel):
    """Request to apply a recommendation."""

    auto: bool = Field(False, description="Whether to auto-apply without approval")


class ApplyRecommendationResponse(BaseModel):
    """Response from applying a recommendation."""

    success: bool
    action_applied: dict | None = None
    status: str | None = None
    error: str | None = None


# ============================================================================
# Dependency Injection
# ============================================================================

def get_optimizer() -> AIInfraOptimizer:
    """Get optimizer instance."""
    return create_infra_optimizer()


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/recommendations", response_model=RecommendationListResponse)
async def get_recommendations(
    org_id: str = Depends(require_organization_id),
    type_filter: RecommendationType | None = Query(None, alias="type"),
    priority_filter: RecommendationPriority | None = Query(None, alias="priority"),
    optimizer: AIInfraOptimizer = Depends(get_optimizer),
):
    """Get AI-generated infrastructure optimization recommendations.

    Analyzes current infrastructure state, usage patterns, and costs
    to generate actionable recommendations for cost savings and
    performance improvements.

    **Recommendation Types:**
    - `scale_down` - Reduce minimum replicas during off-hours
    - `scale_up` - Increase replicas for predicted demand
    - `right_size` - Adjust resource requests/limits
    - `schedule_scaling` - Pre-scale for predicted peaks
    - `cleanup_sessions` - Clean stuck sessions
    - `cost_alert` - Alert on cost threshold breach
    - `anomaly` - Detected anomaly requiring attention

    **Priority Levels:**
    - `critical` - Requires immediate action
    - `high` - Should be addressed soon
    - `medium` - Optimize when convenient
    - `low` - Nice to have
    """
    logger.info("getting_recommendations", org_id=org_id)

    try:
        recommendations = await optimizer.analyze_and_recommend(org_id)

        # Apply filters
        if type_filter:
            recommendations = [r for r in recommendations if r.type == type_filter]
        if priority_filter:
            recommendations = [r for r in recommendations if r.priority == priority_filter]

        # Calculate total potential savings
        total_savings = sum(float(r.estimated_savings_monthly) for r in recommendations)

        response_recs = [
            RecommendationResponse(
                id=r.id,
                type=r.type.value,
                priority=r.priority.value,
                title=r.title,
                description=r.description,
                estimated_savings_monthly=float(r.estimated_savings_monthly),
                confidence=r.confidence,
                action=r.action,
                reasoning=r.reasoning,
                status=r.status.value,
                created_at=r.created_at.isoformat(),
                expires_at=r.expires_at.isoformat(),
            )
            for r in recommendations
        ]

        return RecommendationListResponse(
            recommendations=response_recs,
            total_potential_savings=total_savings,
        )

    except Exception as e:
        logger.error("recommendations_error", error=str(e), org_id=org_id)
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


async def _get_cost_report_impl(
    org_id: str,
    days: int,
    optimizer: AIInfraOptimizer,
) -> CostReportResponse:
    """Implementation for cost report endpoints."""
    try:
        report = await optimizer.get_cost_report(org_id, days=days)

        return CostReportResponse(
            period_start=report.period_start.isoformat(),
            period_end=report.period_end.isoformat(),
            total_cost=float(report.total_cost),
            breakdown={k: float(v) for k, v in report.breakdown.items()},
            daily_costs=[
                {"date": dt.isoformat(), "cost": float(cost)}
                for dt, cost in report.daily_costs
            ],
            projected_monthly=float(report.projected_monthly),
            vultr_cost=float(report.vultr_cost),
            railway_cost=float(report.railway_cost),
            cloudflare_cost=float(report.cloudflare_cost),
            ai_cost=float(report.ai_cost),
        )

    except Exception as e:
        logger.error("cost_report_error", error=str(e), org_id=org_id)
        raise HTTPException(status_code=500, detail=f"Failed to get cost report: {str(e)}")


@router.get("/cost-overview", response_model=CostReportResponse)
async def get_cost_overview(
    org_id: str = Depends(require_organization_id),
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    optimizer: AIInfraOptimizer = Depends(get_optimizer),
):
    """Get infrastructure cost overview.

    Alias for /cost-report. Provides detailed cost breakdown including:
    - Total cost for the period
    - Daily cost breakdown
    - Cost by resource type (compute, network, storage, ai_inference)
    - Projected monthly cost
    - Platform-specific costs (Vultr, Railway, Cloudflare, AI)
    """
    logger.info("getting_cost_overview", org_id=org_id, days=days)
    return await _get_cost_report_impl(org_id, days, optimizer)


@router.get("/cost-report", response_model=CostReportResponse)
async def get_cost_report(
    org_id: str = Depends(require_organization_id),
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    optimizer: AIInfraOptimizer = Depends(get_optimizer),
):
    """Get infrastructure cost analysis report.

    Provides detailed cost breakdown including:
    - Total cost for the period
    - Daily cost breakdown
    - Cost by resource type (compute, network, storage, ai_inference)
    - Projected monthly cost
    - Platform-specific costs (Vultr K8s, Railway, Cloudflare, AI)

    **Platform breakdown:**
    - `vultr_cost`: Kubernetes node costs from Vultr VKE
    - `railway_cost`: Backend service costs (API, Crawlee)
    - `cloudflare_cost`: R2 storage and Workers
    - `ai_cost`: LLM inference costs (Claude, embeddings)
    """
    logger.info("getting_cost_report", org_id=org_id, days=days)
    return await _get_cost_report_impl(org_id, days, optimizer)


@router.get("/forecast", response_model=DemandForecastResponse)
async def get_demand_forecast(
    org_id: str = Depends(require_organization_id),
    hours: int = Query(24, ge=1, le=168, description="Forecast horizon in hours"),
    optimizer: AIInfraOptimizer = Depends(get_optimizer),
):
    """Get demand forecast for browser pool.

    Analyzes historical usage patterns to predict future demand:
    - Hourly predictions for the forecast period
    - Identified peak times
    - Recommended minimum replica counts per browser type

    The forecast uses:
    - Last 7 days of session data
    - Hour-of-day patterns
    - Day-of-week patterns
    - CI/CD schedule awareness (if configured)
    """
    logger.info("getting_forecast", org_id=org_id, hours=hours)

    try:
        forecast = await optimizer.predict_demand(org_id, horizon_hours=hours)

        return DemandForecastResponse(
            forecast_start=forecast.forecast_start.isoformat(),
            forecast_end=forecast.forecast_end.isoformat(),
            hourly_predictions=forecast.hourly_predictions,
            peak_times=[t.isoformat() for t in forecast.peak_times],
            recommended_min_replicas=forecast.recommended_min_replicas,
            confidence=forecast.confidence,
        )

    except Exception as e:
        logger.error("forecast_error", error=str(e), org_id=org_id)
        raise HTTPException(status_code=500, detail=f"Failed to get forecast: {str(e)}")


@router.get("/anomalies", response_model=AnomalyListResponse)
async def get_anomalies(
    org_id: str = Depends(require_organization_id),
    optimizer: AIInfraOptimizer = Depends(get_optimizer),
):
    """Get detected infrastructure anomalies.

    Detects anomalies such as:
    - **stuck_queue** - Session queue not draining
    - **cpu_exhaustion** - Nodes at >90% CPU
    - **memory_exhaustion** - Nodes at >90% memory
    - **over_provisioned** - Cluster running at <20% utilization
    - **resource_leak** - Memory usage increasing over time
    - **unusual_spike** - Sudden usage increase
    """
    logger.info("getting_anomalies", org_id=org_id)

    try:
        anomalies = await optimizer.detect_anomalies(org_id)

        has_critical = any(a.severity == RecommendationPriority.CRITICAL for a in anomalies)

        return AnomalyListResponse(
            anomalies=[
                AnomalyResponse(
                    id=a.id,
                    type=a.type,
                    severity=a.severity.value,
                    description=a.description,
                    detected_at=a.detected_at.isoformat(),
                    metrics=a.metrics,
                    suggested_action=a.suggested_action,
                )
                for a in anomalies
            ],
            has_critical=has_critical,
        )

    except Exception as e:
        logger.error("anomalies_error", error=str(e), org_id=org_id)
        raise HTTPException(status_code=500, detail=f"Failed to get anomalies: {str(e)}")


@router.post("/recommendations/{recommendation_id}/apply", response_model=ApplyRecommendationResponse)
async def apply_recommendation(
    recommendation_id: str,
    request: ApplyRecommendationRequest,
    org_id: str = Depends(require_organization_id),
    optimizer: AIInfraOptimizer = Depends(get_optimizer),
):
    """Apply an infrastructure optimization recommendation.

    **Auto-apply mode:**
    When `auto=true`, the recommendation is applied without requiring
    explicit approval. Use this for scheduled/automated optimization.

    **Manual approval mode (default):**
    When `auto=false`, the recommendation is marked as approved and
    the action is applied with an audit trail.

    **Actions that can be applied:**
    - Set minimum/maximum replica counts
    - Schedule scaling rules
    - Trigger session cleanup
    - Adjust resource requests/limits
    """
    logger.info(
        "applying_recommendation",
        recommendation_id=recommendation_id,
        org_id=org_id,
        auto=request.auto
    )

    try:
        result = await optimizer.apply_recommendation(
            recommendation_id=recommendation_id,
            auto=request.auto
        )

        return ApplyRecommendationResponse(
            success=result["success"],
            action_applied=result.get("action_applied"),
            status=result.get("status"),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error("apply_recommendation_error", error=str(e), recommendation_id=recommendation_id)
        raise HTTPException(status_code=500, detail=f"Failed to apply recommendation: {str(e)}")


@router.get("/savings-summary", response_model=SavingsSummaryResponse)
async def get_savings_summary(
    org_id: str = Depends(require_organization_id),
    optimizer: AIInfraOptimizer = Depends(get_optimizer),
):
    """Get summary of savings achieved through optimization.

    Returns:
    - Total monthly savings from applied recommendations
    - Count of recommendations applied
    - Current infrastructure cost
    - Cost trend (percentage change vs previous period)
    """
    logger.info("getting_savings_summary", org_id=org_id)

    try:
        summary = await optimizer.get_savings_summary(org_id)

        return SavingsSummaryResponse(**summary)

    except Exception as e:
        logger.error("savings_summary_error", error=str(e), org_id=org_id)
        raise HTTPException(status_code=500, detail=f"Failed to get savings summary: {str(e)}")


@router.get("/snapshot", response_model=InfraSnapshotResponse)
async def get_infrastructure_snapshot(
    org_id: str = Depends(require_organization_id),
    optimizer: AIInfraOptimizer = Depends(get_optimizer),
):
    """Get current infrastructure state snapshot.

    Returns real-time metrics for:
    - Selenium Grid status (sessions, queue, nodes)
    - Browser node metrics (Chrome, Firefox, Edge)
    - Cluster-wide utilization
    """
    logger.info("getting_snapshot", org_id=org_id)

    try:
        snapshot = await optimizer.prometheus.get_infrastructure_snapshot()

        return InfraSnapshotResponse(
            selenium={
                "sessions_queued": snapshot.selenium.sessions_queued,
                "sessions_active": snapshot.selenium.sessions_active,
                "sessions_total": snapshot.selenium.sessions_total,
                "nodes_available": snapshot.selenium.nodes_available,
                "nodes_total": snapshot.selenium.nodes_total,
                "avg_session_duration_seconds": snapshot.selenium.avg_session_duration_seconds,
                "queue_wait_time_seconds": snapshot.selenium.queue_wait_time_seconds,
            },
            chrome_nodes={
                "replicas_current": snapshot.chrome_nodes.replicas_current,
                "replicas_desired": snapshot.chrome_nodes.replicas_desired,
                "replicas_min": snapshot.chrome_nodes.replicas_min,
                "replicas_max": snapshot.chrome_nodes.replicas_max,
                "cpu_utilization": snapshot.chrome_nodes.cpu_utilization.cpu_usage_percent,
                "memory_utilization": snapshot.chrome_nodes.memory_utilization.memory_usage_percent,
            },
            firefox_nodes={
                "replicas_current": snapshot.firefox_nodes.replicas_current,
                "replicas_desired": snapshot.firefox_nodes.replicas_desired,
                "replicas_min": snapshot.firefox_nodes.replicas_min,
                "replicas_max": snapshot.firefox_nodes.replicas_max,
                "cpu_utilization": snapshot.firefox_nodes.cpu_utilization.cpu_usage_percent,
                "memory_utilization": snapshot.firefox_nodes.memory_utilization.memory_usage_percent,
            },
            edge_nodes={
                "replicas_current": snapshot.edge_nodes.replicas_current,
                "replicas_desired": snapshot.edge_nodes.replicas_desired,
                "replicas_min": snapshot.edge_nodes.replicas_min,
                "replicas_max": snapshot.edge_nodes.replicas_max,
                "cpu_utilization": snapshot.edge_nodes.cpu_utilization.cpu_usage_percent,
                "memory_utilization": snapshot.edge_nodes.memory_utilization.memory_usage_percent,
            },
            total_pods=snapshot.total_pods,
            total_nodes=snapshot.total_nodes,
            cluster_cpu_utilization=snapshot.cluster_cpu_utilization,
            cluster_memory_utilization=snapshot.cluster_memory_utilization,
            timestamp=snapshot.timestamp.isoformat(),
        )

    except Exception as e:
        logger.error("snapshot_error", error=str(e), org_id=org_id)
        raise HTTPException(status_code=500, detail=f"Failed to get snapshot: {str(e)}")
