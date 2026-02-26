"""TestGen API endpoints.

Provides endpoints for:
- Triggering async test generation jobs
- Listing and retrieving job results
- Analyzing requirements for ambiguity (sync)
- Requirement-to-test coverage matrix
- SSE streaming for job progress
"""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

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
router = APIRouter(prefix="/api/v1/testgen", tags=["Test Generation"])


def _now_z() -> str:
    """Return current UTC timestamp in ISO format with Z suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


# =============================================================================
# Request / Response models
# =============================================================================


class TestGenRequest(BaseModel):
    """Request to trigger a test generation job."""

    source_type: str = Field(
        ...,
        description="Source type: requirements_text, jira, github_pr, figma, confluence, codebase_analysis",
    )
    source_data: dict = Field(
        ...,
        description="Source data payload (varies by source_type)",
    )
    config: dict = Field(
        default_factory=dict,
        description="Optional generation config (framework, language, etc.)",
    )


class RequirementsAnalysisRequest(BaseModel):
    """Request to analyze requirements for ambiguity."""

    text: str = Field(..., description="Raw requirements text to analyze")
    source_type: str = Field(
        default="requirements_text",
        description="Source type hint for parsing",
    )


class TestGenJobResponse(BaseModel):
    """Response schema for a test generation job."""

    id: str
    status: str
    progress: int = 0
    source_type: str
    config: dict = Field(default_factory=dict)
    results: dict | None = None
    quality_score: float | None = None
    ai_cost: float = 0
    error_message: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


# =============================================================================
# POST /{org_id}/{project_id}/generate - Create async test generation job
# =============================================================================


VALID_SOURCE_TYPES = {
    "requirements_text",
    "jira",
    "github_pr",
    "figma",
    "confluence",
    "codebase_analysis",
    "user_story",
    "openapi_spec",
}


@router.post("/{org_id}/{project_id}/generate")
async def create_testgen_job(
    org_id: str,
    project_id: str,
    body: TestGenRequest,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Create an async test generation job.

    Inserts a job record, launches a background task using TestGenAgent,
    and returns the job ID for polling or SSE streaming.
    """
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")

    if body.source_type not in VALID_SOURCE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source_type '{body.source_type}'. Must be one of: {', '.join(sorted(VALID_SOURCE_TYPES))}",
        )

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    job_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    await supabase.insert("testgen_jobs", {
        "id": job_id,
        "organization_id": org_id,
        "project_id": project_id,
        "source_type": body.source_type,
        "config": body.config,
        "source_data": body.source_data,
        "status": "pending",
        "progress": 0,
        "created_by": user_id,
        "created_at": _now_z(),
    })

    # Launch background task
    asyncio.create_task(
        _run_testgen_job(
            job_id, org_id, project_id,
            body.source_type, body.source_data, body.config, user_id,
        )
    )

    logger.info(
        "TestGen job created",
        job_id=job_id,
        org_id=org_id,
        project_id=project_id,
        source_type=body.source_type,
    )

    return {"id": job_id, "status": "pending"}


# =============================================================================
# Background task
# =============================================================================


async def _run_testgen_job(
    job_id: str,
    org_id: str,
    project_id: str,
    source_type: str,
    source_data: dict,
    config: dict,
    user_id: str,
):
    """Run the test generation pipeline in the background."""
    from src.agents.testgen_agent import TestGenAgent

    supabase = get_supabase_client()

    try:
        await supabase.update(
            "testgen_jobs",
            {"id": f"eq.{job_id}"},
            {"status": "running", "started_at": _now_z()},
        )

        agent = TestGenAgent()
        result = await agent.generate(
            org_id=org_id,
            project_id=project_id,
            source_type=source_type,
            source_data=source_data,
            config=config,
            job_id=job_id,
        )

        result_dict = (
            result.to_dict()
            if hasattr(result, "to_dict")
            else (result if isinstance(result, dict) else {})
        )

        await supabase.update(
            "testgen_jobs",
            {"id": f"eq.{job_id}"},
            {
                "status": "completed",
                "progress": 100,
                "results": result_dict,
                "quality_score": result_dict.get("quality_score"),
                "ai_cost": result_dict.get("ai_cost", 0),
                "completed_at": _now_z(),
            },
        )

        logger.info(
            "TestGen job completed",
            job_id=job_id,
            tests_count=result_dict.get("summary", {}).get("tests_count", 0),
            quality_score=result_dict.get("quality_score"),
        )

    except Exception as e:
        logger.error("TestGen job failed", job_id=job_id, error=str(e))
        await supabase.update(
            "testgen_jobs",
            {"id": f"eq.{job_id}"},
            {
                "status": "failed",
                "error_message": str(e)[:2000],
                "completed_at": _now_z(),
            },
        )


# =============================================================================
# GET /{org_id}/{project_id}/jobs - List generation jobs
# =============================================================================


@router.get("/{org_id}/{project_id}/jobs")
async def list_testgen_jobs(
    org_id: str,
    project_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
    status: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List test generation jobs for a project."""
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
    if status:
        filters["status"] = f"eq.{status}"

    supabase = get_supabase_client()
    result = await supabase.select("testgen_jobs", filters=filters)
    rows = result.get("data", [])

    jobs = []
    for r in rows:
        # Parse JSONB fields that PostgREST may return as strings
        config_val = r.get("config", {})
        if isinstance(config_val, str):
            try:
                config_val = json.loads(config_val)
            except (json.JSONDecodeError, TypeError):
                config_val = {}

        results_val = r.get("results")
        if isinstance(results_val, str):
            try:
                results_val = json.loads(results_val)
            except (json.JSONDecodeError, TypeError):
                results_val = None

        jobs.append({
            "id": r["id"],
            "status": r.get("status"),
            "progress": r.get("progress", 0),
            "sourceType": r.get("source_type"),
            "config": config_val,
            "qualityScore": r.get("quality_score"),
            "aiCost": r.get("ai_cost", 0),
            "errorMessage": r.get("error_message"),
            "createdAt": r.get("created_at"),
            "startedAt": r.get("started_at"),
            "completedAt": r.get("completed_at"),
        })

    return {"jobs": jobs, "total": len(jobs)}


# =============================================================================
# GET /{org_id}/{project_id}/jobs/{job_id} - Get job status + results
# =============================================================================


@router.get("/{org_id}/{project_id}/jobs/{job_id}")
async def get_testgen_job(
    org_id: str,
    project_id: str,
    job_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Get full job record including generated test results."""
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")
    validate_uuid(job_id, "job_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    supabase = get_supabase_client()
    result = await supabase.select(
        "testgen_jobs",
        filters={
            "id": f"eq.{job_id}",
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
        },
    )
    rows = result.get("data", [])

    if not rows:
        raise HTTPException(status_code=404, detail="TestGen job not found")

    r = rows[0]

    # Parse JSONB fields that PostgREST may return as strings
    config_val = r.get("config", {})
    if isinstance(config_val, str):
        try:
            config_val = json.loads(config_val)
        except (json.JSONDecodeError, TypeError):
            config_val = {}

    results_val = r.get("results")
    if isinstance(results_val, str):
        try:
            results_val = json.loads(results_val)
        except (json.JSONDecodeError, TypeError):
            results_val = None

    source_data_val = r.get("source_data", {})
    if isinstance(source_data_val, str):
        try:
            source_data_val = json.loads(source_data_val)
        except (json.JSONDecodeError, TypeError):
            source_data_val = {}

    return {
        "id": r["id"],
        "organizationId": r.get("organization_id"),
        "projectId": r.get("project_id"),
        "status": r.get("status"),
        "progress": r.get("progress", 0),
        "sourceType": r.get("source_type"),
        "sourceData": source_data_val,
        "config": config_val,
        "results": results_val,
        "qualityScore": r.get("quality_score"),
        "aiCost": r.get("ai_cost", 0),
        "errorMessage": r.get("error_message"),
        "createdBy": r.get("created_by"),
        "createdAt": r.get("created_at"),
        "startedAt": r.get("started_at"),
        "completedAt": r.get("completed_at"),
    }


# =============================================================================
# POST /{org_id}/{project_id}/analyze-requirements - Sync requirements analysis
# =============================================================================


@router.post("/{org_id}/{project_id}/analyze-requirements")
async def analyze_requirements(
    org_id: str,
    project_id: str,
    body: RequirementsAnalysisRequest,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Analyze requirements text for ambiguity (synchronous, fast).

    Returns parsed requirements with ambiguity scores directly
    without creating a background job.
    """
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Requirements text must not be empty")

    from src.agents.testgen_agent import TestGenAgent, SourceType

    agent = TestGenAgent()

    try:
        source_type = SourceType(body.source_type)
    except ValueError:
        source_type = SourceType.REQUIREMENTS_TEXT

    requirements = await agent.analyze_requirements(
        requirements_text=body.text,
        source_type=source_type,
    )

    parsed = [
        {
            "id": req.id,
            "title": req.title,
            "description": req.description,
            "acceptanceCriteria": req.acceptance_criteria,
            "ambiguityScore": req.ambiguity_score,
            "ambiguityDetails": req.ambiguity_details,
            "priority": req.priority,
            "tags": req.tags,
            "sourceRef": req.source_ref,
        }
        for req in requirements
    ]

    avg_ambiguity = (
        sum(r.ambiguity_score for r in requirements) / len(requirements)
        if requirements
        else 0.0
    )

    return {
        "requirements": parsed,
        "total": len(parsed),
        "avgAmbiguityScore": round(avg_ambiguity, 3),
    }


# =============================================================================
# GET /{org_id}/{project_id}/coverage-matrix/{job_id} - Traceability matrix
# =============================================================================


@router.get("/{org_id}/{project_id}/coverage-matrix/{job_id}")
async def get_coverage_matrix(
    org_id: str,
    project_id: str,
    job_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Get requirement-to-test traceability matrix for a generation job."""
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")
    validate_uuid(job_id, "job_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    supabase = get_supabase_client()
    result = await supabase.select(
        "testgen_coverage_matrix",
        filters={
            "job_id": f"eq.{job_id}",
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
            "order": "requirement_id.asc",
        },
    )
    rows = result.get("data", [])

    entries = [
        {
            "id": r.get("id"),
            "requirementId": r.get("requirement_id"),
            "requirementTitle": r.get("requirement_title"),
            "codeEntity": r.get("code_entity"),
            "codeEntityType": r.get("code_entity_type"),
            "testName": r.get("test_name"),
            "testType": r.get("test_type"),
            "coverageType": r.get("coverage_type"),
            "confidence": r.get("confidence"),
        }
        for r in rows
    ]

    return {"entries": entries, "total": len(entries), "jobId": job_id}


# =============================================================================
# GET /{org_id}/{project_id}/jobs/{job_id}/stream - SSE for job progress
# =============================================================================


@router.get("/{org_id}/{project_id}/jobs/{job_id}/stream")
async def stream_job_progress(
    org_id: str,
    project_id: str,
    job_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Stream job progress via Server-Sent Events.

    Polls the job record periodically and emits status updates
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
        "testgen_jobs",
        filters={
            "id": f"eq.{job_id}",
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
        },
    )
    rows = result.get("data", [])
    if not rows:
        raise HTTPException(status_code=404, detail="TestGen job not found")

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
                    "testgen_jobs",
                    filters={"id": f"eq.{job_id}"},
                )
                poll_rows = poll_result.get("data", [])
                if not poll_rows:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": "Job record not found"}),
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
                            "aiCost": current.get("ai_cost", 0),
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
                            "aiCost": current.get("ai_cost", 0),
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
                    "timestamp": _now_z(),
                }),
            }

    return EventSourceResponse(event_generator())
