"""QA Engineer API endpoints.

Provides endpoints for:
- Starting async repository analysis (full, PR review, coverage, incremental)
- Listing and retrieving analysis results
- Quality score trending over time
- Discovering coverage gaps and user flows
- Comparing two analyses side by side
- SSE streaming for analysis progress
"""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.api.context import require_organization_id
from src.api.middleware.tenant import validate_uuid
from src.api.projects import verify_project_access
from src.api.teams import get_current_user
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/qa-engineer", tags=["QA Engineer"])


def now_z() -> str:
    """Return current UTC timestamp in ISO format with Z suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


# =============================================================================
# Request / Response models
# =============================================================================


class QAAnalysisRequest(BaseModel):
    """Request to start a QA analysis."""

    analysis_type: str = Field(
        ...,
        description="Type of analysis: full, pr_review, coverage_only, incremental",
    )
    trigger_context: dict = Field(
        default_factory=dict,
        description="Additional context (e.g., PR URL, changed files, branch)",
    )


class CompareRequest(BaseModel):
    """Request to compare two analyses."""

    job_id_a: str = Field(..., description="First analysis job ID")
    job_id_b: str = Field(..., description="Second analysis job ID")


# =============================================================================
# POST /{org_id}/{project_id}/analyze - Start async repo analysis
# =============================================================================


VALID_ANALYSIS_TYPES = {"full", "pr_review", "coverage_only", "incremental"}


@router.post("/{org_id}/{project_id}/analyze")
async def start_qa_analysis(
    org_id: str,
    project_id: str,
    body: QAAnalysisRequest,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Start an async QA analysis for the project.

    Launches QAEngineerAgent in the background to analyze the repository,
    discover coverage gaps, user flows, and produce quality recommendations.
    """
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")

    if body.analysis_type not in VALID_ANALYSIS_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid analysis_type '{body.analysis_type}'. Must be one of: {', '.join(sorted(VALID_ANALYSIS_TYPES))}",
        )

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    job_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    await supabase.insert("qa_analysis_jobs", {
        "id": job_id,
        "organization_id": org_id,
        "project_id": project_id,
        "analysis_type": body.analysis_type,
        "trigger_context": body.trigger_context,
        "status": "pending",
        "progress": 0,
        "created_by": user_id,
        "created_at": now_z(),
    })

    # Launch background task
    asyncio.create_task(
        _run_qa_analysis(job_id, org_id, project_id, body.analysis_type, body.trigger_context, user_id)
    )

    logger.info(
        "QA analysis started",
        job_id=job_id,
        org_id=org_id,
        project_id=project_id,
        analysis_type=body.analysis_type,
    )

    return {"jobId": job_id, "status": "pending"}


# =============================================================================
# Background task
# =============================================================================


async def _run_qa_analysis(
    job_id: str,
    org_id: str,
    project_id: str,
    analysis_type: str,
    trigger_context: dict,
    user_id: str,
):
    """Run the QA analysis in the background."""
    from src.agents.qa_engineer_agent import QAEngineerAgent

    supabase = get_supabase_client()

    try:
        await supabase.update(
            "qa_analysis_jobs",
            {"id": f"eq.{job_id}"},
            {"status": "running", "started_at": now_z()},
        )

        agent = QAEngineerAgent()
        result = await agent.analyze(org_id, project_id, analysis_type, trigger_context, job_id)

        await supabase.update(
            "qa_analysis_jobs",
            {"id": f"eq.{job_id}"},
            {
                "status": "completed",
                "progress": 100,
                "quality_score": result.get("quality_score"),
                "components_analyzed": result.get("components", []),
                "coverage_gaps": result.get("gaps", []),
                "user_flows_discovered": result.get("user_flows", []),
                "recommendations": result.get("recommendations", []),
                "ai_cost": result.get("ai_cost", 0),
                "completed_at": now_z(),
            },
        )

        logger.info(
            "QA analysis completed",
            job_id=job_id,
            quality_score=result.get("quality_score"),
            gaps=len(result.get("gaps", [])),
        )

    except Exception as e:
        logger.exception("QA analysis failed", job_id=job_id, error=str(e))
        await supabase.update(
            "qa_analysis_jobs",
            {"id": f"eq.{job_id}"},
            {
                "status": "failed",
                "error_message": str(e)[:2000],
                "completed_at": now_z(),
            },
        )


# =============================================================================
# GET /{org_id}/{project_id}/analyses - List analyses
# =============================================================================


@router.get("/{org_id}/{project_id}/analyses")
async def list_qa_analyses(
    org_id: str,
    project_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
    analysis_type: str | None = Query(None, alias="analysisType"),
    status: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List QA analysis jobs for a project."""
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    filters: dict = {
        "organization_id": f"eq.{org_id}",
        "project_id": f"eq.{project_id}",
        "order": "created_at.desc",
        "limit": str(limit),
        "offset": str(offset),
    }
    if analysis_type:
        filters["analysis_type"] = f"eq.{analysis_type}"
    if status:
        filters["status"] = f"eq.{status}"

    supabase = get_supabase_client()
    result = await supabase.select("qa_analysis_jobs", filters=filters)
    rows = result.get("data", [])

    analyses = [
        {
            "jobId": r["id"],
            "analysisType": r.get("analysis_type"),
            "status": r.get("status"),
            "progress": r.get("progress", 0),
            "qualityScore": r.get("quality_score"),
            "aiCost": r.get("ai_cost"),
            "createdAt": r.get("created_at"),
            "startedAt": r.get("started_at"),
            "completedAt": r.get("completed_at"),
        }
        for r in rows
    ]

    return {"analyses": analyses, "total": len(analyses)}


# =============================================================================
# GET /{org_id}/{project_id}/analyses/{job_id} - Get analysis results
# =============================================================================


@router.get("/{org_id}/{project_id}/analyses/{job_id}")
async def get_qa_analysis(
    org_id: str,
    project_id: str,
    job_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Get the full analysis record including components, gaps, flows, and recommendations."""
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")
    validate_uuid(job_id, "job_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    supabase = get_supabase_client()
    result = await supabase.select(
        "qa_analysis_jobs",
        filters={
            "id": f"eq.{job_id}",
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
        },
    )
    rows = result.get("data", [])

    if not rows:
        raise HTTPException(status_code=404, detail="Analysis not found")

    r = rows[0]

    # Parse JSONB fields that may come back as strings
    def _parse_jsonb(val, default=None):
        if default is None:
            default = []
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default
        return val if val is not None else default

    return {
        "jobId": r["id"],
        "organizationId": r.get("organization_id"),
        "projectId": r.get("project_id"),
        "analysisType": r.get("analysis_type"),
        "triggerContext": r.get("trigger_context"),
        "status": r.get("status"),
        "progress": r.get("progress", 0),
        "qualityScore": r.get("quality_score"),
        "componentsAnalyzed": _parse_jsonb(r.get("components_analyzed")),
        "coverageGaps": _parse_jsonb(r.get("coverage_gaps")),
        "userFlowsDiscovered": _parse_jsonb(r.get("user_flows_discovered")),
        "recommendations": _parse_jsonb(r.get("recommendations")),
        "aiCost": r.get("ai_cost"),
        "errorMessage": r.get("error_message"),
        "createdBy": r.get("created_by"),
        "createdAt": r.get("created_at"),
        "startedAt": r.get("started_at"),
        "completedAt": r.get("completed_at"),
    }


# =============================================================================
# GET /{org_id}/{project_id}/quality-trend - Quality score over time
# =============================================================================


@router.get("/{org_id}/{project_id}/quality-trend")
async def get_quality_trend(
    org_id: str,
    project_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(30, ge=1, le=200),
):
    """Get quality score time-series from qa_quality_scores."""
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    since = (datetime.now(UTC) - timedelta(days=days)).isoformat().replace("+00:00", "Z")

    supabase = get_supabase_client()
    result = await supabase.select(
        "qa_quality_scores",
        filters={
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
            "recorded_at": f"gte.{since}",
            "order": "recorded_at.desc",
            "limit": str(limit),
        },
    )
    rows = result.get("data", [])

    scores = [
        {
            "id": r.get("id"),
            "score": r.get("score"),
            "breakdown": r.get("breakdown"),
            "recordedAt": r.get("recorded_at"),
        }
        for r in rows
    ]

    return {"scores": scores, "total": len(scores)}


# =============================================================================
# GET /{org_id}/{project_id}/coverage-gaps - Current coverage gaps
# =============================================================================


@router.get("/{org_id}/{project_id}/coverage-gaps")
async def get_coverage_gaps(
    org_id: str,
    project_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Return coverage gaps from the most recent completed analysis."""
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    supabase = get_supabase_client()
    result = await supabase.select(
        "qa_analysis_jobs",
        filters={
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
            "status": "eq.completed",
            "order": "completed_at.desc",
            "limit": "1",
        },
    )
    rows = result.get("data", [])

    if not rows:
        return {
            "coverageGaps": [],
            "analysisJobId": None,
            "completedAt": None,
            "message": "No completed analysis found",
        }

    r = rows[0]
    gaps = r.get("coverage_gaps", [])
    if isinstance(gaps, str):
        try:
            gaps = json.loads(gaps)
        except (json.JSONDecodeError, TypeError):
            gaps = []

    return {
        "coverageGaps": gaps,
        "analysisJobId": r["id"],
        "completedAt": r.get("completed_at"),
    }


# =============================================================================
# GET /{org_id}/{project_id}/user-flows - Discovered user flows
# =============================================================================


@router.get("/{org_id}/{project_id}/user-flows")
async def get_user_flows(
    org_id: str,
    project_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Return discovered user flows from the most recent completed analysis."""
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    supabase = get_supabase_client()
    result = await supabase.select(
        "qa_analysis_jobs",
        filters={
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
            "status": "eq.completed",
            "order": "completed_at.desc",
            "limit": "1",
        },
    )
    rows = result.get("data", [])

    if not rows:
        return {
            "userFlows": [],
            "analysisJobId": None,
            "completedAt": None,
            "message": "No completed analysis found",
        }

    r = rows[0]
    flows = r.get("user_flows_discovered", [])
    if isinstance(flows, str):
        try:
            flows = json.loads(flows)
        except (json.JSONDecodeError, TypeError):
            flows = []

    return {
        "userFlows": flows,
        "analysisJobId": r["id"],
        "completedAt": r.get("completed_at"),
    }


# =============================================================================
# POST /{org_id}/{project_id}/compare - Compare two analyses
# =============================================================================


@router.post("/{org_id}/{project_id}/compare")
async def compare_analyses(
    org_id: str,
    project_id: str,
    body: CompareRequest,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Compare two analysis jobs and return the delta."""
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")
    validate_uuid(body.job_id_a, "job_id_a")
    validate_uuid(body.job_id_b, "job_id_b")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    supabase = get_supabase_client()

    # Fetch both jobs (scoped to org + project)
    result_a = await supabase.select(
        "qa_analysis_jobs",
        filters={
            "id": f"eq.{body.job_id_a}",
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
        },
    )
    result_b = await supabase.select(
        "qa_analysis_jobs",
        filters={
            "id": f"eq.{body.job_id_b}",
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
        },
    )

    rows_a = result_a.get("data", [])
    rows_b = result_b.get("data", [])

    if not rows_a:
        raise HTTPException(status_code=404, detail=f"Analysis {body.job_id_a} not found")
    if not rows_b:
        raise HTTPException(status_code=404, detail=f"Analysis {body.job_id_b} not found")

    a = rows_a[0]
    b = rows_b[0]

    def _parse_jsonb(val, default=None):
        if default is None:
            default = []
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default
        return val if val is not None else default

    score_a = a.get("quality_score")
    score_b = b.get("quality_score")
    score_delta = None
    if score_a is not None and score_b is not None:
        score_delta = round(score_b - score_a, 2)

    gaps_a = set(json.dumps(g, sort_keys=True) for g in _parse_jsonb(a.get("coverage_gaps")))
    gaps_b = set(json.dumps(g, sort_keys=True) for g in _parse_jsonb(b.get("coverage_gaps")))

    new_gaps = [json.loads(g) for g in (gaps_b - gaps_a)]
    resolved_gaps = [json.loads(g) for g in (gaps_a - gaps_b)]

    flows_a = set(json.dumps(f, sort_keys=True) for f in _parse_jsonb(a.get("user_flows_discovered")))
    flows_b = set(json.dumps(f, sort_keys=True) for f in _parse_jsonb(b.get("user_flows_discovered")))

    new_flows = [json.loads(f) for f in (flows_b - flows_a)]
    removed_flows = [json.loads(f) for f in (flows_a - flows_b)]

    return {
        "jobIdA": body.job_id_a,
        "jobIdB": body.job_id_b,
        "qualityScoreA": score_a,
        "qualityScoreB": score_b,
        "qualityScoreDelta": score_delta,
        "newCoverageGaps": new_gaps,
        "resolvedCoverageGaps": resolved_gaps,
        "newUserFlows": new_flows,
        "removedUserFlows": removed_flows,
        "completedAtA": a.get("completed_at"),
        "completedAtB": b.get("completed_at"),
    }


# =============================================================================
# GET /{org_id}/{project_id}/analyses/{job_id}/stream - SSE progress
# =============================================================================


@router.get("/{org_id}/{project_id}/analyses/{job_id}/stream")
async def stream_analysis_progress(
    org_id: str,
    project_id: str,
    job_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Stream analysis progress via Server-Sent Events.

    Polls the analysis record periodically and emits status updates
    until the job reaches a terminal state.
    """
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")
    validate_uuid(job_id, "job_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    # Verify job exists
    supabase = get_supabase_client()
    result = await supabase.select(
        "qa_analysis_jobs",
        filters={
            "id": f"eq.{job_id}",
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
        },
    )
    rows = result.get("data", [])
    if not rows:
        raise HTTPException(status_code=404, detail="Analysis not found")

    async def event_generator() -> AsyncGenerator[dict, None]:
        terminal_statuses = {"completed", "failed", "cancelled"}

        try:
            # Emit initial state
            yield {
                "event": "start",
                "data": json.dumps({
                    "jobId": job_id,
                    "status": rows[0].get("status"),
                    "progress": rows[0].get("progress", 0),
                }),
            }

            # If already terminal, close immediately
            if rows[0].get("status") in terminal_statuses:
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "jobId": job_id,
                        "status": rows[0].get("status"),
                        "progress": rows[0].get("progress", 0),
                    }),
                }
                return

            # Poll for updates
            last_progress = rows[0].get("progress", 0)
            last_status = rows[0].get("status")

            while True:
                await asyncio.sleep(3)

                poll_result = await supabase.select(
                    "qa_analysis_jobs",
                    filters={"id": f"eq.{job_id}"},
                )
                poll_rows = poll_result.get("data", [])
                if not poll_rows:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": "Analysis record not found"}),
                    }
                    break

                current = poll_rows[0]
                current_status = current.get("status")
                current_progress = current.get("progress", 0)

                # Emit update if something changed
                if current_status != last_status or current_progress != last_progress:
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "jobId": job_id,
                            "status": current_status,
                            "progress": current_progress,
                            "qualityScore": current.get("quality_score"),
                        }),
                    }
                    last_status = current_status
                    last_progress = current_progress

                # Check terminal
                if current_status in terminal_statuses:
                    yield {
                        "event": "complete",
                        "data": json.dumps({
                            "jobId": job_id,
                            "status": current_status,
                            "progress": current_progress,
                            "qualityScore": current.get("quality_score"),
                            "completedAt": current.get("completed_at"),
                            "errorMessage": current.get("error_message"),
                        }),
                    }
                    break

        except asyncio.CancelledError:
            logger.info("SSE stream cancelled by client", job_id=job_id)
        except Exception as e:
            logger.exception("SSE streaming error", job_id=job_id, error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "timestamp": now_z(),
                }),
            }

    return EventSourceResponse(event_generator())
