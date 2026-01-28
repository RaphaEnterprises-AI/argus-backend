"""GitLab Webhook Handler for VCS Integration.

Handles GitLab webhook events:
- merge_request: opened, updated, merged, closed
- push: commits pushed to branches
- pipeline: CI pipeline status updates (optional)

Security:
- X-Gitlab-Token header verification
- Secrets stored in environment variables
"""

import os
import secrets
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.services.event_gateway import ArgusEvent, EventGateway, EventType
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/webhooks/vcs/gitlab", tags=["VCS Webhooks - GitLab"])


# =============================================================================
# Models
# =============================================================================


class WebhookResponse(BaseModel):
    """Response for webhook processing."""

    success: bool
    message: str
    event_type: str
    delivery_id: str
    sdlc_event_id: str | None = None
    impact_analysis_triggered: bool = False


class ImpactAnalysisResult(BaseModel):
    """Result of test impact analysis."""

    files_changed: list[str] = Field(default_factory=list)
    tests_affected: int = 0
    risk_score: float = 0.0
    recommended_tests: list[str] = Field(default_factory=list)


# =============================================================================
# Token Verification
# =============================================================================


def verify_gitlab_token(
    token_header: str | None,
    expected_token: str,
) -> bool:
    """Verify GitLab webhook token.

    GitLab uses a simple token-based verification where the token
    is sent in the X-Gitlab-Token header.

    Args:
        token_header: X-Gitlab-Token header value
        expected_token: Expected webhook token from settings

    Returns:
        True if token is valid, False otherwise
    """
    if not token_header:
        logger.warning("Missing X-Gitlab-Token header")
        return False

    # Use constant-time comparison to prevent timing attacks
    return secrets.compare_digest(token_header, expected_token)


def get_webhook_token() -> str | None:
    """Get GitLab webhook token from environment."""
    return os.environ.get("GITLAB_WEBHOOK_TOKEN")


# =============================================================================
# Helper Functions
# =============================================================================


async def store_vcs_event(
    project_id: str,
    platform: str,
    event_type: str,
    delivery_id: str,
    payload: dict[str, Any],
) -> dict:
    """Store a VCS event in the database.

    Args:
        project_id: Project ID
        platform: VCS platform (github, gitlab)
        event_type: Event type (push, merge_request, pipeline)
        delivery_id: Webhook delivery ID (generated)
        payload: Event payload

    Returns:
        Created record
    """
    supabase = get_supabase_client()

    record = {
        "project_id": project_id,
        "platform": platform,
        "event_type": event_type,
        "delivery_id": delivery_id,
        "payload": payload,
        "status": "received",
        "received_at": datetime.now(UTC).isoformat(),
    }

    result = await supabase.insert("vcs_webhook_events", record)

    if result.get("error"):
        logger.error(
            "Failed to store VCS event",
            delivery_id=delivery_id,
            error=result["error"],
        )
        return {}

    return result.get("data", [{}])[0]


async def update_vcs_event_status(
    delivery_id: str,
    status: str,
    sdlc_event_id: str | None = None,
    error_message: str | None = None,
) -> None:
    """Update VCS event processing status."""
    supabase = get_supabase_client()

    update_data: dict[str, Any] = {
        "status": status,
        "processed_at": datetime.now(UTC).isoformat(),
    }

    if sdlc_event_id:
        update_data["sdlc_event_id"] = sdlc_event_id
    if error_message:
        update_data["error_message"] = error_message

    await supabase.update(
        "vcs_webhook_events",
        {"delivery_id": f"eq.{delivery_id}"},
        update_data,
    )


async def emit_internal_event(
    event_type: EventType,
    org_id: str,
    project_id: str,
    data: dict[str, Any],
    correlation_id: str | None = None,
) -> None:
    """Emit an internal event to the event gateway.

    Args:
        event_type: Type of event (using INTEGRATION_GITHUB for GitLab too)
        org_id: Organization ID
        project_id: Project ID
        data: Event payload data
        correlation_id: Optional correlation ID for tracing
    """
    try:
        gateway = EventGateway()
        await gateway.start()

        event = ArgusEvent(
            event_type=event_type,
            org_id=org_id,
            project_id=project_id,
            data=data,
            correlation_id=correlation_id,
            source="gitlab-webhook",
        )

        await gateway.publish(
            event_type=event_type,
            data=event.model_dump(),
            org_id=org_id,
            project_id=project_id,
        )

        await gateway.stop()

        logger.info(
            "Emitted internal event",
            event_type=event_type.value,
            project_id=project_id,
        )

    except ImportError:
        # Event gateway not available (aiokafka not installed)
        logger.debug(
            "Event gateway not available, skipping internal event emission",
            event_type=event_type.value,
        )
    except Exception as e:
        # Don't fail webhook processing if event emission fails
        logger.warning(
            "Failed to emit internal event",
            event_type=event_type.value,
            error=str(e),
        )


async def trigger_impact_analysis(
    project_id: str,
    commit_sha: str,
    files_changed: list[str],
    mr_iid: int | None = None,
    branch_name: str | None = None,
    repository: str | None = None,
) -> ImpactAnalysisResult:
    """Trigger test impact analysis for changed files.

    Args:
        project_id: Project ID
        commit_sha: Git commit SHA
        files_changed: List of changed file paths
        mr_iid: Optional MR internal ID
        branch_name: Optional branch name
        repository: Repository path

    Returns:
        Impact analysis result
    """
    supabase = get_supabase_client()

    logger.info(
        "Triggering impact analysis",
        project_id=project_id,
        commit_sha=commit_sha[:8] if commit_sha else "unknown",
        files_count=len(files_changed),
        mr_iid=mr_iid,
    )

    # Query impact graph for affected tests
    result = await supabase.rpc(
        "get_affected_tests",
        {
            "p_project_id": project_id,
            "p_file_paths": files_changed,
            "p_min_score": 0.3,
            "p_limit": 50,
        },
    )

    affected_tests = []
    if not result.get("error"):
        affected_tests = result.get("data") or []

    # Calculate risk score based on impact
    risk_score = 0.0
    if affected_tests:
        test_count_risk = min(0.3, len(affected_tests) / 100)
        max_impact = max((t.get("total_impact_score", 0) for t in affected_tests), default=0)
        impact_risk = min(0.4, float(max_impact))
        risk_score = min(1.0, test_count_risk + impact_risk)

    # Store impact analysis result
    analysis_record = {
        "project_id": project_id,
        "commit_sha": commit_sha,
        "mr_iid": mr_iid,
        "branch_name": branch_name,
        "files_changed": len(files_changed),
        "tests_affected": len(affected_tests),
        "risk_score": risk_score,
        "affected_test_ids": [t.get("test_id") for t in affected_tests if t.get("test_id")],
        "analyzed_at": datetime.now(UTC).isoformat(),
    }

    await supabase.request(
        "/commit_impact_analyses",
        method="POST",
        body=analysis_record,
        headers={"Prefer": "resolution=merge-duplicates"},
    )

    return ImpactAnalysisResult(
        files_changed=files_changed,
        tests_affected=len(affected_tests),
        risk_score=round(risk_score, 2),
        recommended_tests=[t.get("test_name", "") for t in affected_tests[:10]],
    )


def extract_files_from_commits(commits: list[dict]) -> list[str]:
    """Extract changed files from a list of commits (GitLab format)."""
    files = set()
    for commit in commits:
        files.update(commit.get("added", []))
        files.update(commit.get("removed", []))
        files.update(commit.get("modified", []))
    return list(files)


def generate_delivery_id(payload: dict) -> str:
    """Generate a unique delivery ID for GitLab webhooks.

    GitLab doesn't include a delivery ID like GitHub, so we generate one
    based on the payload content.
    """
    import hashlib
    import json

    # Create a deterministic ID from key fields
    key_parts = []

    # For push events
    if "checkout_sha" in payload:
        key_parts.append(payload.get("checkout_sha", ""))

    # For merge request events
    if "object_attributes" in payload:
        attrs = payload["object_attributes"]
        key_parts.append(str(attrs.get("id", "")))
        key_parts.append(str(attrs.get("iid", "")))
        key_parts.append(attrs.get("action", ""))

    # For pipeline events
    if "object_kind" in payload:
        key_parts.append(payload["object_kind"])

    # Add timestamp for uniqueness
    key_parts.append(str(datetime.now(UTC).timestamp()))

    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


# =============================================================================
# Event Handlers
# =============================================================================


async def handle_push_event(
    project_id: str,
    org_id: str,
    payload: dict,
    delivery_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    """Handle GitLab push event.

    GitLab push webhook payload structure:
    - ref: refs/heads/branch-name
    - checkout_sha: HEAD commit SHA
    - commits: list of commit objects
    - project: project info
    """
    ref = payload.get("ref", "")
    branch_name = ref.replace("refs/heads/", "")
    checkout_sha = payload.get("checkout_sha", "")
    project_info = payload.get("project", {})
    repo_path = project_info.get("path_with_namespace", "")

    commits = payload.get("commits", [])

    logger.info(
        "Processing GitLab push event",
        delivery_id=delivery_id,
        repository=repo_path,
        branch=branch_name,
        commits_count=len(commits),
    )

    # Extract changed files
    files_changed = extract_files_from_commits(commits)

    # Emit internal event
    await emit_internal_event(
        event_type=EventType.INTEGRATION_GITHUB,  # Reuse GitHub event type
        org_id=org_id,
        project_id=project_id,
        data={
            "source": "gitlab",
            "event_type": "push",
            "repository": repo_path,
            "branch": branch_name,
            "commits_count": len(commits),
            "head_sha": checkout_sha,
            "files_changed": files_changed[:100],
        },
        correlation_id=delivery_id,
    )

    # Trigger impact analysis
    impact_result = None
    if checkout_sha and files_changed:
        impact_result = await trigger_impact_analysis(
            project_id=project_id,
            commit_sha=checkout_sha,
            files_changed=files_changed,
            branch_name=branch_name,
            repository=repo_path,
        )

    return {
        "sdlc_event_id": None,
        "impact_analysis_triggered": impact_result is not None,
        "tests_affected": impact_result.tests_affected if impact_result else 0,
    }


async def handle_merge_request_event(
    project_id: str,
    org_id: str,
    payload: dict,
    delivery_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    """Handle GitLab merge_request event.

    GitLab MR webhook payload structure:
    - object_attributes: MR details (iid, action, state, etc.)
    - project: project info
    - user: user who triggered the event
    - changes: what changed (optional)
    """
    attrs = payload.get("object_attributes", {})
    action = attrs.get("action", "")
    mr_iid = attrs.get("iid")
    mr_id = attrs.get("id")
    state = attrs.get("state", "")
    project_info = payload.get("project", {})
    repo_path = project_info.get("path_with_namespace", "")

    # Extract branch info
    source_branch = attrs.get("source_branch", "")
    target_branch = attrs.get("target_branch", "")
    last_commit = attrs.get("last_commit", {})
    head_sha = last_commit.get("id", "")

    logger.info(
        "Processing GitLab merge_request event",
        delivery_id=delivery_id,
        action=action,
        mr_iid=mr_iid,
        state=state,
        repository=repo_path,
    )

    # Emit internal event
    await emit_internal_event(
        event_type=EventType.INTEGRATION_GITHUB,  # Reuse GitHub event type
        org_id=org_id,
        project_id=project_id,
        data={
            "source": "gitlab",
            "event_type": "merge_request",
            "action": action,
            "mr_iid": mr_iid,
            "mr_id": mr_id,
            "state": state,
            "repository": repo_path,
            "source_branch": source_branch,
            "target_branch": target_branch,
            "head_sha": head_sha,
            "title": attrs.get("title"),
            "author": payload.get("user", {}).get("username"),
            "draft": attrs.get("work_in_progress", False) or attrs.get("draft", False),
            "url": attrs.get("url"),
        },
        correlation_id=delivery_id,
    )

    # Trigger impact analysis for opened, updated, or reopened MRs
    impact_result = None
    if action in ("open", "update", "reopen") and head_sha:
        # Get changed files from MR changes (if available)
        # GitLab may include changes in the payload
        changes = payload.get("changes", {})
        files_changed: list[str] = []

        # GitLab sometimes includes file changes in the payload
        if "files" in changes:
            files_changed = [f.get("path", "") for f in changes.get("files", []) if f.get("path")]

        impact_result = await trigger_impact_analysis(
            project_id=project_id,
            commit_sha=head_sha,
            files_changed=files_changed,
            mr_iid=mr_iid,
            branch_name=source_branch,
            repository=repo_path,
        )

    return {
        "sdlc_event_id": None,
        "impact_analysis_triggered": impact_result is not None,
        "tests_affected": impact_result.tests_affected if impact_result else 0,
    }


async def handle_pipeline_event(
    project_id: str,
    org_id: str,
    payload: dict,
    delivery_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    """Handle GitLab pipeline event (CI pipeline status).

    GitLab pipeline webhook payload structure:
    - object_attributes: pipeline details (id, status, ref, sha)
    - project: project info
    - builds: list of job details (optional)
    """
    attrs = payload.get("object_attributes", {})
    pipeline_id = attrs.get("id")
    status = attrs.get("status", "")
    ref = attrs.get("ref", "")
    sha = attrs.get("sha", "")
    project_info = payload.get("project", {})
    repo_path = project_info.get("path_with_namespace", "")

    logger.info(
        "Processing GitLab pipeline event",
        delivery_id=delivery_id,
        pipeline_id=pipeline_id,
        status=status,
        repository=repo_path,
    )

    # Emit internal event
    await emit_internal_event(
        event_type=EventType.INTEGRATION_GITHUB,  # Reuse GitHub event type
        org_id=org_id,
        project_id=project_id,
        data={
            "source": "gitlab",
            "event_type": "pipeline",
            "pipeline_id": pipeline_id,
            "status": status,
            "ref": ref,
            "sha": sha,
            "repository": repo_path,
            "duration": attrs.get("duration"),
            "created_at": attrs.get("created_at"),
            "finished_at": attrs.get("finished_at"),
            "url": f"{project_info.get('web_url', '')}/-/pipelines/{pipeline_id}",
        },
        correlation_id=delivery_id,
    )

    return {
        "sdlc_event_id": None,
        "impact_analysis_triggered": False,
    }


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("", response_model=WebhookResponse)
async def receive_gitlab_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_gitlab_event: str | None = Header(None, alias="X-Gitlab-Event"),
    x_gitlab_token: str | None = Header(None, alias="X-Gitlab-Token"),
    project_id: str = Query(..., description="Project ID for this webhook"),
    org_id: str = Query(..., description="Organization ID for multi-tenancy"),
):
    """Receive and process GitLab webhook events.

    Supports events:
    - Push Hook: Commits pushed to branches
    - Merge Request Hook: MR opened, updated, merged, closed
    - Pipeline Hook: CI pipeline status updates

    Security: Verifies X-Gitlab-Token header using GITLAB_WEBHOOK_TOKEN env var.

    Args:
        request: FastAPI request object
        background_tasks: Background task runner
        x_gitlab_event: GitLab event type header
        x_gitlab_token: Webhook token for verification
        project_id: Project ID to associate events with
        org_id: Organization ID for multi-tenancy

    Returns:
        WebhookResponse with processing status
    """
    # Validate required headers
    if not x_gitlab_event:
        raise HTTPException(status_code=400, detail="Missing X-Gitlab-Event header")

    # Parse payload
    try:
        payload = await request.json()
    except Exception as e:
        logger.error("Failed to parse webhook payload", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Generate delivery ID (GitLab doesn't provide one)
    delivery_id = generate_delivery_id(payload)

    # Verify token if configured
    webhook_token = get_webhook_token()
    if webhook_token:
        if not x_gitlab_token:
            logger.warning(
                "Missing token header",
                delivery_id=delivery_id,
            )
            raise HTTPException(status_code=401, detail="Missing token")

        if not verify_gitlab_token(x_gitlab_token, webhook_token):
            logger.warning(
                "Invalid GitLab webhook token",
                delivery_id=delivery_id,
                event_type=x_gitlab_event,
            )
            raise HTTPException(status_code=401, detail="Invalid token")
    else:
        logger.debug(
            "Webhook token verification skipped (no token configured)",
            delivery_id=delivery_id,
        )

    # Normalize event type
    # GitLab uses "Push Hook", "Merge Request Hook", etc.
    event_type_normalized = x_gitlab_event.lower().replace(" hook", "").replace(" ", "_")

    logger.info(
        "Received GitLab webhook",
        delivery_id=delivery_id,
        event_type=x_gitlab_event,
        event_type_normalized=event_type_normalized,
        project_id=project_id,
        org_id=org_id,
    )

    # Store raw event
    await store_vcs_event(
        project_id=project_id,
        platform="gitlab",
        event_type=event_type_normalized,
        delivery_id=delivery_id,
        payload=payload,
    )

    # Route to appropriate handler
    handler_map = {
        "push": handle_push_event,
        "merge_request": handle_merge_request_event,
        "pipeline": handle_pipeline_event,
    }

    handler = handler_map.get(event_type_normalized)

    if not handler:
        logger.info(
            "Unsupported webhook event type",
            event_type=x_gitlab_event,
            event_type_normalized=event_type_normalized,
            delivery_id=delivery_id,
        )
        await update_vcs_event_status(delivery_id, "skipped")
        return WebhookResponse(
            success=True,
            message=f"Event type '{x_gitlab_event}' not processed",
            event_type=x_gitlab_event,
            delivery_id=delivery_id,
        )

    try:
        result = await handler(
            project_id=project_id,
            org_id=org_id,
            payload=payload,
            delivery_id=delivery_id,
            background_tasks=background_tasks,
        )

        await update_vcs_event_status(
            delivery_id=delivery_id,
            status="processed",
            sdlc_event_id=result.get("sdlc_event_id"),
        )

        return WebhookResponse(
            success=True,
            message=f"Successfully processed {x_gitlab_event}",
            event_type=x_gitlab_event,
            delivery_id=delivery_id,
            sdlc_event_id=result.get("sdlc_event_id"),
            impact_analysis_triggered=result.get("impact_analysis_triggered", False),
        )

    except Exception as e:
        logger.exception(
            "Failed to process GitLab webhook",
            event_type=x_gitlab_event,
            delivery_id=delivery_id,
            error=str(e),
        )

        await update_vcs_event_status(
            delivery_id=delivery_id,
            status="failed",
            error_message=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process webhook: {str(e)}"
        )


@router.get("/events")
async def list_gitlab_webhook_events(
    project_id: str = Query(..., description="Project ID"),
    event_type: str | None = Query(None, description="Filter by event type"),
    status: str | None = Query(None, description="Filter by processing status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum events to return"),
):
    """List recent GitLab webhook events for a project."""
    supabase = get_supabase_client()

    query_path = (
        f"/vcs_webhook_events?project_id=eq.{project_id}"
        f"&platform=eq.gitlab"
    )

    if event_type:
        query_path += f"&event_type=eq.{event_type}"
    if status:
        query_path += f"&status=eq.{status}"

    query_path += f"&order=received_at.desc&limit={limit}"

    result = await supabase.request(query_path)

    if result.get("error"):
        error_msg = str(result.get("error", ""))
        if "does not exist" in error_msg or "42P01" in error_msg:
            return {"events": [], "total": 0}
        raise HTTPException(status_code=500, detail="Failed to fetch events")

    events = result.get("data") or []

    return {
        "events": events,
        "total": len(events),
    }
