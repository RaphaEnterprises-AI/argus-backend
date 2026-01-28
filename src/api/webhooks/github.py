"""GitHub Webhook Handler for VCS Integration.

Handles GitHub webhook events:
- pull_request: opened, synchronize, reopened, closed
- push: commits pushed to branches
- check_run: CI check status updates

Security:
- HMAC SHA256 signature verification
- Secrets stored in environment variables
"""

import hashlib
import hmac
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

router = APIRouter(prefix="/api/v1/webhooks/vcs/github", tags=["VCS Webhooks - GitHub"])


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
# Signature Verification
# =============================================================================


def verify_github_signature(
    payload_body: bytes,
    signature_header: str | None,
    webhook_secret: str,
) -> bool:
    """Verify GitHub webhook signature using HMAC SHA256.

    Args:
        payload_body: Raw request body bytes
        signature_header: X-Hub-Signature-256 header value
        webhook_secret: Webhook secret from GitHub settings

    Returns:
        True if signature is valid, False otherwise
    """
    if not signature_header:
        logger.warning("Missing X-Hub-Signature-256 header")
        return False

    if not signature_header.startswith("sha256="):
        logger.warning("Invalid signature format", header=signature_header[:20])
        return False

    expected_signature = signature_header[7:]  # Remove "sha256=" prefix

    # Calculate HMAC SHA256
    mac = hmac.new(
        webhook_secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256,
    )
    calculated_signature = mac.hexdigest()

    # Use constant-time comparison to prevent timing attacks
    return secrets.compare_digest(expected_signature, calculated_signature)


def get_webhook_secret() -> str | None:
    """Get GitHub webhook secret from environment."""
    return os.environ.get("GITHUB_WEBHOOK_SECRET")


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
        event_type: Event type (push, pr, check_run)
        delivery_id: Webhook delivery ID
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
        event_type: Type of event
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
            source="github-webhook",
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
    pr_number: int | None = None,
    branch_name: str | None = None,
    repository: str | None = None,
) -> ImpactAnalysisResult:
    """Trigger test impact analysis for changed files.

    This queries the impact graph to determine which tests are affected
    by the changed files and calculates a risk score.

    Args:
        project_id: Project ID
        commit_sha: Git commit SHA
        files_changed: List of changed file paths
        pr_number: Optional PR number
        branch_name: Optional branch name
        repository: Repository in owner/repo format

    Returns:
        Impact analysis result
    """
    supabase = get_supabase_client()

    logger.info(
        "Triggering impact analysis",
        project_id=project_id,
        commit_sha=commit_sha[:8],
        files_count=len(files_changed),
        pr_number=pr_number,
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
        # Higher risk if many tests affected
        test_count_risk = min(0.3, len(affected_tests) / 100)
        # Higher risk if high-impact tests affected
        max_impact = max((t.get("total_impact_score", 0) for t in affected_tests), default=0)
        impact_risk = min(0.4, float(max_impact))
        risk_score = min(1.0, test_count_risk + impact_risk)

    # Store impact analysis result
    analysis_record = {
        "project_id": project_id,
        "commit_sha": commit_sha,
        "pr_number": pr_number,
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
    """Extract changed files from a list of commits."""
    files = set()
    for commit in commits:
        files.update(commit.get("added", []))
        files.update(commit.get("removed", []))
        files.update(commit.get("modified", []))
    return list(files)


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
    """Handle GitHub push event.

    Creates SDLC events for commits and triggers impact analysis.
    """
    ref = payload.get("ref", "")
    branch_name = ref.replace("refs/heads/", "")
    repository = payload.get("repository", {})
    repo_full_name = repository.get("full_name", "")

    commits = payload.get("commits", [])
    head_commit = payload.get("head_commit", {})

    logger.info(
        "Processing GitHub push event",
        delivery_id=delivery_id,
        repository=repo_full_name,
        branch=branch_name,
        commits_count=len(commits),
    )

    # Extract changed files
    files_changed = extract_files_from_commits(commits)

    # Emit internal event
    await emit_internal_event(
        event_type=EventType.INTEGRATION_GITHUB,
        org_id=org_id,
        project_id=project_id,
        data={
            "event_type": "push",
            "repository": repo_full_name,
            "branch": branch_name,
            "commits_count": len(commits),
            "head_sha": head_commit.get("id"),
            "files_changed": files_changed[:100],  # Limit for event size
        },
        correlation_id=delivery_id,
    )

    # Trigger impact analysis for head commit
    impact_result = None
    if head_commit.get("id") and files_changed:
        impact_result = await trigger_impact_analysis(
            project_id=project_id,
            commit_sha=head_commit["id"],
            files_changed=files_changed,
            branch_name=branch_name,
            repository=repo_full_name,
        )

    return {
        "sdlc_event_id": None,
        "impact_analysis_triggered": impact_result is not None,
        "tests_affected": impact_result.tests_affected if impact_result else 0,
    }


async def handle_pull_request_event(
    project_id: str,
    org_id: str,
    payload: dict,
    delivery_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    """Handle GitHub pull_request event."""
    action = payload.get("action")
    pr = payload.get("pull_request", {})
    pr_number = pr.get("number")
    repository = payload.get("repository", {})
    repo_full_name = repository.get("full_name", "")

    logger.info(
        "Processing GitHub pull_request event",
        delivery_id=delivery_id,
        action=action,
        pr_number=pr_number,
        repository=repo_full_name,
    )

    # Extract commit info
    head_sha = pr.get("head", {}).get("sha")
    head_branch = pr.get("head", {}).get("ref")
    base_branch = pr.get("base", {}).get("ref")

    # Emit internal event
    await emit_internal_event(
        event_type=EventType.INTEGRATION_GITHUB,
        org_id=org_id,
        project_id=project_id,
        data={
            "event_type": "pull_request",
            "action": action,
            "pr_number": pr_number,
            "repository": repo_full_name,
            "head_sha": head_sha,
            "head_branch": head_branch,
            "base_branch": base_branch,
            "title": pr.get("title"),
            "author": pr.get("user", {}).get("login"),
            "draft": pr.get("draft", False),
            "html_url": pr.get("html_url"),
        },
        correlation_id=delivery_id,
    )

    # Trigger impact analysis for opened, synchronize, or reopened PRs
    impact_result = None
    if action in ("opened", "synchronize", "reopened") and head_sha:
        # Get changed files from PR (if available in payload)
        # GitHub doesn't include files in PR webhook, need to fetch separately
        # For now, use an empty list - the impact analysis will fetch files
        files_changed: list[str] = []

        impact_result = await trigger_impact_analysis(
            project_id=project_id,
            commit_sha=head_sha,
            files_changed=files_changed,
            pr_number=pr_number,
            branch_name=head_branch,
            repository=repo_full_name,
        )

    return {
        "sdlc_event_id": None,
        "impact_analysis_triggered": impact_result is not None,
        "tests_affected": impact_result.tests_affected if impact_result else 0,
    }


async def handle_check_run_event(
    project_id: str,
    org_id: str,
    payload: dict,
    delivery_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    """Handle GitHub check_run event."""
    action = payload.get("action")
    check_run = payload.get("check_run", {})
    repository = payload.get("repository", {})
    repo_full_name = repository.get("full_name", "")

    logger.info(
        "Processing GitHub check_run event",
        delivery_id=delivery_id,
        action=action,
        name=check_run.get("name"),
        conclusion=check_run.get("conclusion"),
    )

    # Emit internal event
    await emit_internal_event(
        event_type=EventType.INTEGRATION_GITHUB,
        org_id=org_id,
        project_id=project_id,
        data={
            "event_type": "check_run",
            "action": action,
            "check_run_id": check_run.get("id"),
            "name": check_run.get("name"),
            "status": check_run.get("status"),
            "conclusion": check_run.get("conclusion"),
            "head_sha": check_run.get("head_sha"),
            "repository": repo_full_name,
            "html_url": check_run.get("html_url"),
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
async def receive_github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str | None = Header(None, alias="X-GitHub-Event"),
    x_github_delivery: str | None = Header(None, alias="X-GitHub-Delivery"),
    x_hub_signature_256: str | None = Header(None, alias="X-Hub-Signature-256"),
    project_id: str = Query(..., description="Project ID for this webhook"),
    org_id: str = Query(..., description="Organization ID for multi-tenancy"),
):
    """Receive and process GitHub webhook events.

    Supports events:
    - push: Commits pushed to branches
    - pull_request: PR opened, updated, closed
    - check_run: CI check status updates

    Security: Verifies HMAC SHA256 signature using GITHUB_WEBHOOK_SECRET env var.

    Args:
        request: FastAPI request object
        background_tasks: Background task runner
        x_github_event: GitHub event type header
        x_github_delivery: Unique delivery ID
        x_hub_signature_256: HMAC signature for verification
        project_id: Project ID to associate events with
        org_id: Organization ID for multi-tenancy

    Returns:
        WebhookResponse with processing status
    """
    # Validate required headers
    if not x_github_event:
        raise HTTPException(status_code=400, detail="Missing X-GitHub-Event header")

    if not x_github_delivery:
        raise HTTPException(status_code=400, detail="Missing X-GitHub-Delivery header")

    # Read raw body for signature verification
    body = await request.body()

    # Verify signature if webhook secret is configured
    webhook_secret = get_webhook_secret()
    if webhook_secret:
        if not x_hub_signature_256:
            logger.warning(
                "Missing signature header",
                delivery_id=x_github_delivery,
            )
            raise HTTPException(status_code=401, detail="Missing signature")

        if not verify_github_signature(body, x_hub_signature_256, webhook_secret):
            logger.warning(
                "Invalid GitHub webhook signature",
                delivery_id=x_github_delivery,
                event_type=x_github_event,
            )
            raise HTTPException(status_code=401, detail="Invalid signature")
    else:
        logger.debug(
            "Webhook signature verification skipped (no secret configured)",
            delivery_id=x_github_delivery,
        )

    # Parse payload
    try:
        payload = await request.json()
    except Exception as e:
        logger.error("Failed to parse webhook payload", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    logger.info(
        "Received GitHub webhook",
        delivery_id=x_github_delivery,
        event_type=x_github_event,
        project_id=project_id,
        org_id=org_id,
    )

    # Store raw event
    await store_vcs_event(
        project_id=project_id,
        platform="github",
        event_type=x_github_event,
        delivery_id=x_github_delivery,
        payload=payload,
    )

    # Route to appropriate handler
    handler_map = {
        "push": handle_push_event,
        "pull_request": handle_pull_request_event,
        "check_run": handle_check_run_event,
    }

    handler = handler_map.get(x_github_event)

    if not handler:
        logger.info(
            "Unsupported webhook event type",
            event_type=x_github_event,
            delivery_id=x_github_delivery,
        )
        await update_vcs_event_status(x_github_delivery, "skipped")
        return WebhookResponse(
            success=True,
            message=f"Event type '{x_github_event}' not processed",
            event_type=x_github_event,
            delivery_id=x_github_delivery,
        )

    try:
        result = await handler(
            project_id=project_id,
            org_id=org_id,
            payload=payload,
            delivery_id=x_github_delivery,
            background_tasks=background_tasks,
        )

        await update_vcs_event_status(
            delivery_id=x_github_delivery,
            status="processed",
            sdlc_event_id=result.get("sdlc_event_id"),
        )

        return WebhookResponse(
            success=True,
            message=f"Successfully processed {x_github_event} event",
            event_type=x_github_event,
            delivery_id=x_github_delivery,
            sdlc_event_id=result.get("sdlc_event_id"),
            impact_analysis_triggered=result.get("impact_analysis_triggered", False),
        )

    except Exception as e:
        logger.exception(
            "Failed to process GitHub webhook",
            event_type=x_github_event,
            delivery_id=x_github_delivery,
            error=str(e),
        )

        await update_vcs_event_status(
            delivery_id=x_github_delivery,
            status="failed",
            error_message=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process webhook: {str(e)}"
        )


@router.get("/events")
async def list_github_webhook_events(
    project_id: str = Query(..., description="Project ID"),
    event_type: str | None = Query(None, description="Filter by event type"),
    status: str | None = Query(None, description="Filter by processing status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum events to return"),
):
    """List recent GitHub webhook events for a project."""
    supabase = get_supabase_client()

    query_path = (
        f"/vcs_webhook_events?project_id=eq.{project_id}"
        f"&platform=eq.github"
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
