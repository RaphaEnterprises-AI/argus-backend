"""Healing CI/CD API endpoints.

Provides endpoints for:
- Triggering proactive and CI-failure healing scans
- Listing and retrieving scan results with fix proposals
- Health score trending over time
- GitHub Action failure callbacks
- SSE streaming for scan progress
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
router = APIRouter(prefix="/api/v1/healing-cicd", tags=["Healing CI/CD"])


def now_z() -> str:
    """Return current UTC timestamp in ISO format with Z suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


# =============================================================================
# Request / Response models
# =============================================================================


class HealingScanRequest(BaseModel):
    """Request to trigger a healing scan."""

    scan_type: str = Field(
        ...,
        description="Type of scan: proactive, ci_failure, pr_review, manual",
    )
    trigger_context: dict = Field(
        default_factory=dict,
        description="Additional context for the scan trigger",
    )


class CICallbackRequest(BaseModel):
    """GitHub Action failure callback payload."""

    run_url: str = Field(..., description="URL of the failed CI run")
    commit_sha: str = Field(..., description="Commit SHA that triggered the run")
    branch: str = Field(..., description="Branch name")
    pr_number: int | None = Field(None, description="PR number if applicable")
    test_framework: str | None = Field(None, description="Test framework (pytest, jest, etc.)")
    test_results_path: str | None = Field(None, description="Path to test results artifact")
    failed_tests: list[dict] = Field(
        default_factory=list,
        description="List of failed test details",
    )


# =============================================================================
# POST /{org_id}/{project_id}/scan - Trigger healing scan
# =============================================================================


VALID_SCAN_TYPES = {"proactive", "ci_failure", "pr_review", "manual"}


@router.post("/{org_id}/{project_id}/scan")
async def trigger_healing_scan(
    org_id: str,
    project_id: str,
    body: HealingScanRequest,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Trigger a healing scan for the project.

    Launches an async background task that uses TestHealerCICDAgent
    to scan for broken/flaky tests and propose fixes.
    """
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")

    if body.scan_type not in VALID_SCAN_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scan_type '{body.scan_type}'. Must be one of: {', '.join(sorted(VALID_SCAN_TYPES))}",
        )

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    scan_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    await supabase.insert("healing_scan_jobs", {
        "id": scan_id,
        "organization_id": org_id,
        "project_id": project_id,
        "scan_type": body.scan_type,
        "trigger_context": body.trigger_context,
        "status": "pending",
        "progress": 0,
        "created_by": user_id,
        "created_at": now_z(),
    })

    # Launch background task
    asyncio.create_task(
        _run_healing_scan(scan_id, org_id, project_id, body.scan_type, body.trigger_context, user_id)
    )

    logger.info(
        "Healing scan triggered",
        scan_id=scan_id,
        org_id=org_id,
        project_id=project_id,
        scan_type=body.scan_type,
    )

    return {"scanId": scan_id, "status": "pending"}


# =============================================================================
# Background task
# =============================================================================


async def _run_healing_scan(
    scan_id: str,
    org_id: str,
    project_id: str,
    scan_type: str,
    trigger_context: dict,
    user_id: str,
):
    """Run the healing scan in the background."""
    from src.agents.testhealer_cicd_agent import TestHealerCICDAgent

    supabase = get_supabase_client()

    try:
        await supabase.update(
            "healing_scan_jobs",
            {"id": f"eq.{scan_id}"},
            {"status": "running", "started_at": now_z()},
        )

        agent = TestHealerCICDAgent()
        result = await agent.scan(org_id, project_id, scan_type, trigger_context, scan_id)

        await supabase.update(
            "healing_scan_jobs",
            {"id": f"eq.{scan_id}"},
            {
                "status": "completed",
                "progress": 100,
                "tests_scanned": result.get("tests_scanned", 0),
                "issues_found": result.get("issues_found", 0),
                "fixes_proposed": result.get("fixes_proposed", 0),
                "fix_proposals": result.get("fix_proposals", []),
                "health_score_after": result.get("health_score_after"),
                "ai_cost": result.get("ai_cost", 0),
                "completed_at": now_z(),
            },
        )

        logger.info(
            "Healing scan completed",
            scan_id=scan_id,
            tests_scanned=result.get("tests_scanned", 0),
            issues_found=result.get("issues_found", 0),
        )

    except Exception as e:
        logger.exception("Healing scan failed", scan_id=scan_id, error=str(e))
        await supabase.update(
            "healing_scan_jobs",
            {"id": f"eq.{scan_id}"},
            {
                "status": "failed",
                "error_message": str(e)[:2000],
                "completed_at": now_z(),
            },
        )


# =============================================================================
# GET /{org_id}/{project_id}/scans - List scan jobs
# =============================================================================


@router.get("/{org_id}/{project_id}/scans")
async def list_healing_scans(
    org_id: str,
    project_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
    scan_type: str | None = Query(None, alias="scanType"),
    status: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List healing scan jobs for a project."""
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
    if scan_type:
        filters["scan_type"] = f"eq.{scan_type}"
    if status:
        filters["status"] = f"eq.{status}"

    supabase = get_supabase_client()
    result = await supabase.select("healing_scan_jobs", filters=filters)
    rows = result.get("data", [])

    scans = [
        {
            "scanId": r["id"],
            "scanType": r.get("scan_type"),
            "status": r.get("status"),
            "progress": r.get("progress", 0),
            "testsScanned": r.get("tests_scanned", 0),
            "issuesFound": r.get("issues_found", 0),
            "fixesProposed": r.get("fixes_proposed", 0),
            "healthScoreAfter": r.get("health_score_after"),
            "aiCost": r.get("ai_cost"),
            "createdAt": r.get("created_at"),
            "startedAt": r.get("started_at"),
            "completedAt": r.get("completed_at"),
        }
        for r in rows
    ]

    return {"scans": scans, "total": len(scans)}


# =============================================================================
# GET /{org_id}/{project_id}/scans/{scan_id} - Get scan results
# =============================================================================


@router.get("/{org_id}/{project_id}/scans/{scan_id}")
async def get_healing_scan(
    org_id: str,
    project_id: str,
    scan_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Get full scan record including fix proposals."""
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")
    validate_uuid(scan_id, "scan_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    supabase = get_supabase_client()
    result = await supabase.select(
        "healing_scan_jobs",
        filters={
            "id": f"eq.{scan_id}",
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
        },
    )
    rows = result.get("data", [])

    if not rows:
        raise HTTPException(status_code=404, detail="Scan not found")

    r = rows[0]

    # Parse fix_proposals if returned as string (PostgREST JSONB)
    fix_proposals = r.get("fix_proposals", [])
    if isinstance(fix_proposals, str):
        try:
            fix_proposals = json.loads(fix_proposals)
        except (json.JSONDecodeError, TypeError):
            fix_proposals = []

    return {
        "scanId": r["id"],
        "organizationId": r.get("organization_id"),
        "projectId": r.get("project_id"),
        "scanType": r.get("scan_type"),
        "triggerContext": r.get("trigger_context"),
        "status": r.get("status"),
        "progress": r.get("progress", 0),
        "testsScanned": r.get("tests_scanned", 0),
        "issuesFound": r.get("issues_found", 0),
        "fixesProposed": r.get("fixes_proposed", 0),
        "fixProposals": fix_proposals,
        "healthScoreAfter": r.get("health_score_after"),
        "aiCost": r.get("ai_cost"),
        "errorMessage": r.get("error_message"),
        "createdBy": r.get("created_by"),
        "createdAt": r.get("created_at"),
        "startedAt": r.get("started_at"),
        "completedAt": r.get("completed_at"),
    }


# =============================================================================
# GET /{org_id}/{project_id}/health-scores - Health score trending
# =============================================================================


@router.get("/{org_id}/{project_id}/health-scores")
async def get_health_scores(
    org_id: str,
    project_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(30, ge=1, le=200),
):
    """Get health score time-series for trending."""
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    since = datetime.now(UTC)
    from datetime import timedelta

    since = (since - timedelta(days=days)).isoformat().replace("+00:00", "Z")

    supabase = get_supabase_client()
    result = await supabase.select(
        "healing_health_scores",
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
            "components": r.get("components"),
            "recordedAt": r.get("recorded_at"),
        }
        for r in rows
    ]

    return {"scores": scores, "total": len(scores)}


# =============================================================================
# GET /{org_id}/{project_id}/health-score/latest - Current health score
# =============================================================================


@router.get("/{org_id}/{project_id}/health-score/latest")
async def get_latest_health_score(
    org_id: str,
    project_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Get the most recent health score for the project."""
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    supabase = get_supabase_client()
    result = await supabase.select(
        "healing_health_scores",
        filters={
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
            "order": "recorded_at.desc",
            "limit": "1",
        },
    )
    rows = result.get("data", [])

    if not rows:
        return {
            "score": None,
            "components": None,
            "recordedAt": None,
            "message": "No health scores recorded yet",
        }

    r = rows[0]
    return {
        "id": r.get("id"),
        "score": r.get("score"),
        "components": r.get("components"),
        "recordedAt": r.get("recorded_at"),
    }


# =============================================================================
# POST /{org_id}/{project_id}/ci-callback - GitHub Action failure callback
# =============================================================================


@router.post("/{org_id}/{project_id}/ci-callback")
async def ci_failure_callback(
    org_id: str,
    project_id: str,
    body: CICallbackRequest,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Receive a GitHub Action failure callback and trigger a ci_failure scan.

    Designed to be called from a GitHub Action step on test failure,
    passing along the failed test details for automatic healing.
    """
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    scan_id = str(uuid.uuid4())
    trigger_context = {
        "run_url": body.run_url,
        "commit_sha": body.commit_sha,
        "branch": body.branch,
        "pr_number": body.pr_number,
        "test_framework": body.test_framework,
        "test_results_path": body.test_results_path,
        "failed_tests": body.failed_tests,
    }

    supabase = get_supabase_client()
    await supabase.insert("healing_scan_jobs", {
        "id": scan_id,
        "organization_id": org_id,
        "project_id": project_id,
        "scan_type": "ci_failure",
        "trigger_context": trigger_context,
        "status": "pending",
        "progress": 0,
        "created_by": user_id,
        "created_at": now_z(),
    })

    # Emit Kafka event
    try:
        from src.events.producer import get_event_producer
        from src.events.schemas import EventMetadata, TenantInfo

        producer = get_event_producer()
        await producer.send(
            topic="argus.healing.requested",
            value={
                "event_type": "healing.ci_callback",
                "tenant": TenantInfo(
                    org_id=org_id,
                    project_id=project_id,
                    user_id=user_id,
                ).model_dump(mode="json"),
                "metadata": EventMetadata(source="ci_callback").model_dump(mode="json"),
                "scan_id": scan_id,
                "commit_sha": body.commit_sha,
                "branch": body.branch,
                "failed_tests_count": len(body.failed_tests),
            },
        )
    except Exception as exc:
        logger.warning("Failed to emit healing.ci_callback event", error=str(exc))

    # Launch background scan
    asyncio.create_task(
        _run_healing_scan(scan_id, org_id, project_id, "ci_failure", trigger_context, user_id)
    )

    logger.info(
        "CI failure callback received",
        scan_id=scan_id,
        org_id=org_id,
        project_id=project_id,
        commit_sha=body.commit_sha,
        failed_tests=len(body.failed_tests),
    )

    return {"scanId": scan_id, "status": "pending"}


# =============================================================================
# GET /{org_id}/{project_id}/scans/{scan_id}/stream - SSE for scan progress
# =============================================================================


@router.get("/{org_id}/{project_id}/scans/{scan_id}/stream")
async def stream_scan_progress(
    org_id: str,
    project_id: str,
    scan_id: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Stream scan progress via Server-Sent Events.

    Polls the scan record periodically and emits status updates
    until the scan reaches a terminal state.
    """
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")
    validate_uuid(scan_id, "scan_id")

    user = await get_current_user(request)
    user_id = user.get("id", "") if isinstance(user, dict) else getattr(user, "id", "")
    user_email = user.get("email") if isinstance(user, dict) else getattr(user, "email", None)

    await verify_project_access(project_id, user_id, user_email, request=request)

    # Verify scan exists
    supabase = get_supabase_client()
    result = await supabase.select(
        "healing_scan_jobs",
        filters={
            "id": f"eq.{scan_id}",
            "organization_id": f"eq.{org_id}",
            "project_id": f"eq.{project_id}",
        },
    )
    rows = result.get("data", [])
    if not rows:
        raise HTTPException(status_code=404, detail="Scan not found")

    async def event_generator() -> AsyncGenerator[dict, None]:
        terminal_statuses = {"completed", "failed", "cancelled"}

        try:
            # Emit initial state
            yield {
                "event": "start",
                "data": json.dumps({
                    "scanId": scan_id,
                    "status": rows[0].get("status"),
                    "progress": rows[0].get("progress", 0),
                }),
            }

            # If already terminal, close immediately
            if rows[0].get("status") in terminal_statuses:
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "scanId": scan_id,
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
                    "healing_scan_jobs",
                    filters={"id": f"eq.{scan_id}"},
                )
                poll_rows = poll_result.get("data", [])
                if not poll_rows:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": "Scan record not found"}),
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
                            "scanId": scan_id,
                            "status": current_status,
                            "progress": current_progress,
                            "testsScanned": current.get("tests_scanned", 0),
                            "issuesFound": current.get("issues_found", 0),
                            "fixesProposed": current.get("fixes_proposed", 0),
                        }),
                    }
                    last_status = current_status
                    last_progress = current_progress

                # Check terminal
                if current_status in terminal_statuses:
                    yield {
                        "event": "complete",
                        "data": json.dumps({
                            "scanId": scan_id,
                            "status": current_status,
                            "progress": current_progress,
                            "testsScanned": current.get("tests_scanned", 0),
                            "issuesFound": current.get("issues_found", 0),
                            "fixesProposed": current.get("fixes_proposed", 0),
                            "healthScoreAfter": current.get("health_score_after"),
                            "completedAt": current.get("completed_at"),
                            "errorMessage": current.get("error_message"),
                        }),
                    }
                    break

        except asyncio.CancelledError:
            logger.info("SSE stream cancelled by client", scan_id=scan_id)
        except Exception as e:
            logger.exception("SSE streaming error", scan_id=scan_id, error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "timestamp": now_z(),
                }),
            }

    return EventSourceResponse(event_generator())
