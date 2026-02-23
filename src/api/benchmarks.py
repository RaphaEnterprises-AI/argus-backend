"""Agent Reliability Benchmarks API.

Exposes CLEAR framework metrics and Pass@k reliability data
for the dashboard and public trust page.

Endpoints:
- GET /benchmarks         → Latest benchmarks for the org (all agents)
- GET /benchmarks/public  → Global anonymized benchmarks (no auth)
- GET /benchmarks/{agent} → Time-series data for one agent
- GET /evaluations/recent → Last 100 evaluations for the org
- POST /benchmarks/aggregate → Trigger on-demand aggregation
"""

from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.teams import get_current_user
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Benchmarks"])


# ============================================================================
# Response Models
# ============================================================================


class BenchmarkResponse(BaseModel):
    """Single agent benchmark record."""

    agent_type: str = Field(alias="agentType")
    period_start: str = Field(alias="periodStart")
    period_end: str = Field(alias="periodEnd")
    total_executions: int = Field(alias="totalExecutions")
    pass_at_1: float | None = Field(None, alias="passAt1")
    pass_at_3: float | None = Field(None, alias="passAt3")
    pass_at_8: float | None = Field(None, alias="passAt8")
    avg_efficacy: float | None = Field(None, alias="avgEfficacy")
    avg_latency_ms: float | None = Field(None, alias="avgLatencyMs")
    p50_latency_ms: int | None = Field(None, alias="p50LatencyMs")
    p95_latency_ms: int | None = Field(None, alias="p95LatencyMs")
    p99_latency_ms: int | None = Field(None, alias="p99LatencyMs")
    avg_cost_usd: float | None = Field(None, alias="avgCostUsd")
    cost_normalized_accuracy: float | None = Field(None, alias="costNormalizedAccuracy")
    avg_assurance: float | None = Field(None, alias="avgAssurance")
    self_correction_rate: float | None = Field(None, alias="selfCorrectionRate")
    human_escalation_rate: float | None = Field(None, alias="humanEscalationRate")

    model_config = {"populate_by_name": True}


class EvaluationResponse(BaseModel):
    """Single agent evaluation record."""

    id: str
    agent_type: str = Field(alias="agentType")
    task_type: str = Field(alias="taskType")
    task_completed: bool = Field(alias="taskCompleted")
    efficacy_score: float | None = Field(None, alias="efficacyScore")
    latency_ms: int | None = Field(None, alias="latencyMs")
    cost_usd: float | None = Field(None, alias="costUsd")
    assurance_score: float | None = Field(None, alias="assuranceScore")
    self_corrections: int = Field(0, alias="selfCorrections")
    human_escalated: bool = Field(False, alias="humanEscalated")
    error_type: str | None = Field(None, alias="errorType")
    created_at: str = Field(alias="createdAt")

    model_config = {"populate_by_name": True}


class AggregateResponse(BaseModel):
    """Response for on-demand aggregation."""

    benchmarks_created: int = Field(alias="benchmarksCreated")
    agent_types: list[str] = Field(alias="agentTypes")

    model_config = {"populate_by_name": True}


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/benchmarks/public")
async def get_public_benchmarks():
    """Return global (anonymized) benchmarks for the public trust page.

    No authentication required. Returns global benchmarks (sentinel org or NULL).
    """
    BENCHMARK_ORG_UUID = "00000000-0000-0000-0000-000000000000"
    supabase = get_supabase_client()

    response = await supabase.request(
        f"/agent_benchmarks?or=(organization_id.is.null,organization_id.eq.{BENCHMARK_ORG_UUID})"
        "&order=period_start.desc"
        "&select=agent_type,period_start,period_end,total_executions,"
        "pass_at_1,pass_at_3,pass_at_8,avg_efficacy,avg_latency_ms,"
        "p50_latency_ms,p95_latency_ms,p99_latency_ms,avg_cost_usd,"
        "cost_normalized_accuracy,avg_assurance,self_correction_rate,"
        "human_escalation_rate"
        "&limit=50"
    )

    rows = response.get("data", [])
    return {"data": rows, "total": len(rows)}


@router.get("/benchmarks/{agent_type}")
async def get_agent_benchmarks(
    request: Request,
    agent_type: str,
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
):
    """Return time-series benchmark data for a single agent type."""
    user = await get_current_user(request)
    org_id = user.get("organization_id")

    supabase = get_supabase_client()

    since = (datetime.now(UTC) - timedelta(days=days)).date().isoformat()

    filters = (
        f"agent_type=eq.{agent_type}"
        f"&period_start=gte.{since}"
        "&order=period_start.asc"
        "&select=agent_type,period_start,period_end,total_executions,"
        "pass_at_1,pass_at_3,pass_at_8,avg_efficacy,avg_latency_ms,"
        "p50_latency_ms,p95_latency_ms,p99_latency_ms,avg_cost_usd,"
        "cost_normalized_accuracy,avg_assurance,self_correction_rate,"
        "human_escalation_rate"
    )

    if org_id:
        filters += f"&organization_id=eq.{org_id}"

    response = await supabase.request(f"/agent_benchmarks?{filters}")

    rows = response.get("data", [])
    if not rows:
        raise HTTPException(status_code=404, detail=f"No benchmarks found for agent '{agent_type}'")

    return {"data": rows, "agent_type": agent_type, "total": len(rows)}


@router.get("/benchmarks")
async def get_benchmarks(
    request: Request,
    days: int = Query(7, ge=1, le=365, description="Number of days of history"),
):
    """Return latest benchmarks for all agents in the user's org + global benchmarks."""
    BENCHMARK_ORG_UUID = "00000000-0000-0000-0000-000000000000"
    user = await get_current_user(request)
    org_id = user.get("organization_id")

    supabase = get_supabase_client()

    since = (datetime.now(UTC) - timedelta(days=days)).date().isoformat()

    select = (
        "&select=agent_type,period_start,period_end,total_executions,"
        "pass_at_1,pass_at_3,pass_at_8,avg_efficacy,avg_latency_ms,"
        "p50_latency_ms,p95_latency_ms,p99_latency_ms,avg_cost_usd,"
        "cost_normalized_accuracy,avg_assurance,self_correction_rate,"
        "human_escalation_rate"
    )

    # Include both org-specific and global (sentinel UUID) benchmarks
    if org_id:
        org_filter = f"or=(organization_id.eq.{org_id},organization_id.eq.{BENCHMARK_ORG_UUID},organization_id.is.null)"
    else:
        org_filter = f"or=(organization_id.eq.{BENCHMARK_ORG_UUID},organization_id.is.null)"

    response = await supabase.request(
        f"/agent_benchmarks?period_start=gte.{since}&order=period_start.desc{select}&{org_filter}"
    )

    rows = response.get("data", [])
    return {"data": rows, "total": len(rows)}


@router.get("/evaluations/recent")
async def get_recent_evaluations(
    request: Request,
    limit: int = Query(100, ge=1, le=500, description="Number of evaluations"),
    agent_type: str | None = Query(None, description="Filter by agent type"),
):
    """Return the most recent agent evaluations for the org + global benchmarks."""
    BENCHMARK_ORG_UUID = "00000000-0000-0000-0000-000000000000"
    user = await get_current_user(request)
    org_id = user.get("organization_id")

    supabase = get_supabase_client()

    select = (
        "&select=id,agent_type,task_type,task_completed,"
        "efficacy_score,latency_ms,cost_usd,assurance_score,"
        "self_corrections,human_escalated,error_type,created_at"
    )

    # Include both org-specific and global benchmark evaluations
    if org_id:
        org_filter = f"&or=(organization_id.eq.{org_id},organization_id.eq.{BENCHMARK_ORG_UUID})"
    else:
        org_filter = f"&organization_id=eq.{BENCHMARK_ORG_UUID}"

    agent_filter = f"&agent_type=eq.{agent_type}" if agent_type else ""

    response = await supabase.request(
        f"/agent_evaluations?order=created_at.desc&limit={limit}{select}{org_filter}{agent_filter}"
    )

    rows = response.get("data", [])
    return {"data": rows, "total": len(rows)}


@router.post("/benchmarks/aggregate")
async def trigger_aggregation(
    request: Request,
    period_days: int = Query(1, ge=1, le=90, description="Days to aggregate"),
):
    """Trigger on-demand benchmark aggregation for the org."""
    user = await get_current_user(request)
    org_id = user.get("organization_id")

    from src.services.benchmark_aggregator import aggregate_benchmarks

    benchmarks = await aggregate_benchmarks(
        period_days=period_days,
        organization_id=org_id,
    )

    return {
        "benchmarksCreated": len(benchmarks),
        "agentTypes": [b["agent_type"] for b in benchmarks],
    }
