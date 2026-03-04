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


class RecordEvaluationRequest(BaseModel):
    """Request to record a single agent evaluation."""

    agent_type: str = Field(description="Agent type identifier (e.g., TestGenAgent)")
    task_type: str = Field(description="Task type (e.g., requirements_analysis, test_generation)")
    task_completed: bool = Field(description="Whether the task completed successfully")
    efficacy_score: float | None = Field(None, description="Quality/efficacy score 0.0-1.0")
    latency_ms: int | None = Field(None, description="Execution time in milliseconds")
    cost_usd: float | None = Field(None, description="AI cost in USD")
    assurance_score: float | None = Field(None, description="Assurance/confidence score 0.0-1.0")
    input_tokens: int | None = Field(None, description="Input tokens consumed")
    output_tokens: int | None = Field(None, description="Output tokens generated")
    tool_calls_count: int | None = Field(None, description="Number of tool calls made")
    self_corrections: int = Field(0, description="Number of self-correction iterations")
    human_escalated: bool = Field(False, description="Whether human intervention was needed")
    error_type: str | None = Field(None, description="Error category if failed")
    error_message: str | None = Field(None, description="Error details if failed")
    trigger_source: str = Field("benchmark", description="What triggered this evaluation")
    metadata: dict | None = Field(None, description="Additional metadata (framework, domain, etc.)")


class RecordBatchRequest(BaseModel):
    """Request to record multiple evaluations at once."""

    evaluations: list[RecordEvaluationRequest] = Field(
        ..., description="List of evaluations to record", min_length=1, max_length=100
    )
    global_benchmark: bool = Field(
        False,
        description="If true, store with the global sentinel org UUID so results are visible to all users",
    )


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
    include_global: bool = Query(False, description="Also aggregate global (sentinel org) benchmarks"),
):
    """Trigger on-demand benchmark aggregation for the org."""
    BENCHMARK_ORG_UUID = "00000000-0000-0000-0000-000000000000"
    user = await get_current_user(request)
    org_id = user.get("organization_id")

    from src.services.benchmark_aggregator import aggregate_benchmarks

    all_benchmarks = []

    # Aggregate org-specific benchmarks
    if org_id:
        benchmarks = await aggregate_benchmarks(
            period_days=period_days,
            organization_id=org_id,
        )
        all_benchmarks.extend(benchmarks)

    # Also aggregate global benchmarks (sentinel org)
    if include_global or not org_id:
        global_benchmarks = await aggregate_benchmarks(
            period_days=period_days,
            organization_id=BENCHMARK_ORG_UUID,
        )
        all_benchmarks.extend(global_benchmarks)

    return {
        "benchmarksCreated": len(all_benchmarks),
        "agentTypes": list({b["agent_type"] for b in all_benchmarks}),
    }


@router.post("/benchmarks/run")
async def trigger_benchmark_run(
    request: Request,
    suites: list[str] | None = Query(None, description="Suites to run (default: all)"),
    runs_per_scenario: int = Query(1, ge=1, le=8, description="Runs per scenario (for Pass@k)"),
):
    """Trigger an on-demand benchmark run.

    Runs the specified suites (or all) and records real evaluations.
    Each run actually executes the agent against test data.

    WARNING: This costs real money (AI API calls) and can take minutes.
    """
    import asyncio

    from src.services.benchmark_runner import SUITES, run_all_benchmarks

    user = await get_current_user(request)

    available = list(SUITES.keys())
    if suites:
        invalid = [s for s in suites if s not in available]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown suites: {invalid}. Available: {available}",
            )

    logger.info(
        "benchmark_run_triggered",
        user=user.get("user_id"),
        suites=suites or "all",
        runs_per_scenario=runs_per_scenario,
    )

    # Run in background so the request returns immediately
    async def _run():
        try:
            results = await run_all_benchmarks(
                runs_per_scenario=runs_per_scenario,
                suites=suites,
                aggregate=True,
            )
            logger.info(
                "benchmark_run_complete",
                suites_run=len(results),
                total_cost=sum(r.total_cost for r in results.values()),
            )
        except Exception as e:
            logger.error("benchmark_run_failed", error=str(e))

    asyncio.create_task(_run())

    return {
        "status": "started",
        "suites": suites or available,
        "runs_per_scenario": runs_per_scenario,
        "message": "Benchmark run started in background. Check /benchmarks/public for results.",
    }


@router.post("/benchmarks/reset")
async def reset_benchmark_data(request: Request):
    """Delete ALL benchmark evaluations and aggregated benchmarks.

    This removes fabricated/old data so real benchmark runs can start fresh.
    Only deletes records with the sentinel benchmark org_id.
    """
    user = await get_current_user(request)

    supabase = get_supabase_client()
    BENCHMARK_ORG = "00000000-0000-0000-0000-000000000000"

    # Delete evaluations with benchmark org
    try:
        await supabase.request(
            f"/agent_evaluations?organization_id=eq.{BENCHMARK_ORG}",
            method="DELETE",
        )
        logger.info("benchmark_evaluations_deleted", org_id=BENCHMARK_ORG)
    except Exception as e:
        logger.error("delete_evaluations_failed", error=str(e))

    # Delete aggregated benchmarks with benchmark org
    try:
        await supabase.request(
            f"/agent_benchmarks?organization_id=eq.{BENCHMARK_ORG}",
            method="DELETE",
        )
        logger.info("benchmark_aggregations_deleted", org_id=BENCHMARK_ORG)
    except Exception as e:
        logger.error("delete_benchmarks_failed", error=str(e))

    return {
        "status": "reset_complete",
        "message": "All benchmark evaluations and aggregations deleted. Run POST /benchmarks/run to generate real data.",
    }


@router.post("/evaluations/record")
async def record_evaluation(
    request: Request,
    body: RecordEvaluationRequest,
):
    """Record a single agent evaluation result.

    Used to persist benchmark results, ad-hoc test results, or manual evaluations.
    """
    user = await get_current_user(request)
    org_id = user.get("organization_id") or "00000000-0000-0000-0000-000000000000"

    supabase = get_supabase_client()

    record = {
        "organization_id": org_id,
        "agent_type": body.agent_type,
        "task_type": body.task_type,
        "task_completed": body.task_completed,
        "efficacy_score": body.efficacy_score,
        "latency_ms": body.latency_ms,
        "cost_usd": body.cost_usd,
        "assurance_score": body.assurance_score,
        "input_tokens": body.input_tokens,
        "output_tokens": body.output_tokens,
        "tool_calls_count": body.tool_calls_count,
        "self_corrections": body.self_corrections,
        "human_escalated": body.human_escalated,
        "error_type": body.error_type,
        "error_message": body.error_message,
        "trigger_source": body.trigger_source,
        "metadata": body.metadata,
    }

    result = await supabase.insert("agent_evaluations", record)

    logger.info(
        "evaluation_recorded",
        agent_type=body.agent_type,
        task_type=body.task_type,
        task_completed=body.task_completed,
    )

    return {"id": result.get("id") if isinstance(result, dict) else None, "status": "recorded"}


@router.post("/evaluations/record-batch")
async def record_evaluation_batch(
    request: Request,
    body: RecordBatchRequest,
):
    """Record multiple agent evaluations in a single request.

    Efficiently inserts up to 100 evaluations at once.
    Set global_benchmark=true to store with sentinel org UUID (visible to all users).
    """
    BENCHMARK_ORG_UUID = "00000000-0000-0000-0000-000000000000"
    user = await get_current_user(request)
    org_id = BENCHMARK_ORG_UUID if body.global_benchmark else (
        user.get("organization_id") or BENCHMARK_ORG_UUID
    )

    supabase = get_supabase_client()

    records = []
    for ev in body.evaluations:
        records.append({
            "organization_id": org_id,
            "agent_type": ev.agent_type,
            "task_type": ev.task_type,
            "task_completed": ev.task_completed,
            "efficacy_score": ev.efficacy_score,
            "latency_ms": ev.latency_ms,
            "cost_usd": ev.cost_usd,
            "assurance_score": ev.assurance_score,
            "input_tokens": ev.input_tokens,
            "output_tokens": ev.output_tokens,
            "tool_calls_count": ev.tool_calls_count,
            "self_corrections": ev.self_corrections,
            "human_escalated": ev.human_escalated,
            "error_type": ev.error_type,
            "error_message": ev.error_message,
            "trigger_source": ev.trigger_source,
            "metadata": ev.metadata,
        })

    # Batch insert
    inserted = 0
    for record in records:
        try:
            await supabase.insert("agent_evaluations", record)
            inserted += 1
        except Exception as e:
            logger.error("batch_insert_failed", agent_type=record["agent_type"], error=str(e))

    logger.info(
        "evaluation_batch_recorded",
        total=len(records),
        inserted=inserted,
    )

    return {
        "inserted": inserted,
        "total": len(records),
        "status": "recorded",
    }
