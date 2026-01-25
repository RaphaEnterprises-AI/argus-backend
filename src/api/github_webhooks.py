"""GitHub Webhook Handler for Commit Impact Analysis.

Receives GitHub webhook events (push, pull_request, check_run, deployment,
deployment_status) and triggers commit analysis using the Impact Graph and
Failure Pattern systems.

Security:
- HMAC SHA256 signature verification
- Secrets stored in environment variables
"""

import hashlib
import hmac
import secrets
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Header, Query, Request
from pydantic import BaseModel, Field

from src.config import get_settings
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/webhooks/github", tags=["GitHub Webhooks"])


# =============================================================================
# Models
# =============================================================================


class GitHubWebhookResponse(BaseModel):
    """Response for webhook processing."""

    success: bool
    message: str
    event_type: str
    delivery_id: str
    sdlc_event_id: str | None = None
    commit_analysis_id: str | None = None


class CommitAnalysisResponse(BaseModel):
    """Response for commit analysis."""

    id: str
    commit_sha: str
    pr_number: int | None = None
    risk_score: float
    deployment_strategy: str
    tests_to_run_count: int
    predicted_failures_count: int
    security_issues_count: int
    recommendations_count: int


class CommitAnalysisRequest(BaseModel):
    """Request to analyze a specific commit."""

    commit_sha: str = Field(..., description="Git commit SHA to analyze")
    pr_number: int | None = Field(None, description="Optional PR number")
    branch_name: str | None = Field(None, description="Branch name")
    repository: str | None = Field(None, description="Repository in owner/repo format")


# =============================================================================
# Helper Functions
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


async def store_webhook_event(
    delivery_id: str,
    event_type: str,
    payload: dict,
    project_id: str | None = None,
) -> dict:
    """Store raw webhook event for auditing.

    Args:
        delivery_id: X-GitHub-Delivery header
        event_type: X-GitHub-Event header
        payload: Webhook payload
        project_id: Optional project ID if identified

    Returns:
        Created record
    """
    supabase = get_supabase_client()

    result = await supabase.insert(
        "github_webhook_events",
        {
            "delivery_id": delivery_id,
            "event_type": event_type,
            "payload": payload,
            "project_id": project_id,
            "status": "pending",
        },
    )

    if result.get("error"):
        logger.error(
            "Failed to store webhook event",
            delivery_id=delivery_id,
            error=result["error"],
        )
        return {}

    return result.get("data", [{}])[0]


async def update_webhook_status(
    delivery_id: str,
    status: str,
    sdlc_event_id: str | None = None,
    commit_analysis_id: str | None = None,
    error_message: str | None = None,
):
    """Update webhook event processing status."""
    supabase = get_supabase_client()

    update_data: dict[str, Any] = {
        "status": status,
        "processed_at": datetime.now(UTC).isoformat(),
    }

    if sdlc_event_id:
        update_data["sdlc_event_id"] = sdlc_event_id
    if commit_analysis_id:
        update_data["commit_analysis_id"] = commit_analysis_id
    if error_message:
        update_data["error_message"] = error_message

    await supabase.update(
        "github_webhook_events",
        {"delivery_id": f"eq.{delivery_id}"},
        update_data,
    )


async def store_sdlc_event(
    project_id: str,
    event_type: str,
    external_id: str,
    payload: dict,
    occurred_at: datetime | None = None,
) -> dict:
    """Store an SDLC event from GitHub webhook.

    Args:
        project_id: Project ID
        event_type: Event type (push, pr, commit, etc.)
        external_id: External event ID
        payload: Event data
        occurred_at: Event timestamp

    Returns:
        Created SDLC event record
    """
    supabase = get_supabase_client()

    # Extract common fields from payload
    title = payload.get("title")
    description = payload.get("description") or payload.get("body")
    commit_sha = payload.get("commit_sha") or payload.get("head_sha") or payload.get("sha")
    pr_number = payload.get("pr_number") or payload.get("number")
    branch_name = payload.get("branch_name") or payload.get("ref", "").replace("refs/heads/", "")

    # Build external URL
    external_url = payload.get("html_url") or payload.get("url")

    record = {
        "project_id": project_id,
        "event_type": event_type,
        "source_platform": "github",
        "external_id": external_id,
        "external_url": external_url,
        "title": title,
        "description": description,
        "commit_sha": commit_sha,
        "pr_number": pr_number,
        "branch_name": branch_name,
        "occurred_at": (occurred_at or datetime.now(UTC)).isoformat(),
        "data": payload,
    }

    result = await supabase.insert("sdlc_events", record)

    if result.get("error"):
        # Check if it's a duplicate - return existing record
        if "duplicate" in str(result["error"]).lower():
            existing = await supabase.request(
                f"/sdlc_events?project_id=eq.{project_id}"
                f"&source_platform=eq.github"
                f"&external_id=eq.{external_id}"
            )
            if existing.get("data"):
                return existing["data"][0]

        logger.error("Failed to store SDLC event", error=result["error"])
        return {}

    return result.get("data", [{}])[0]


async def fetch_github_diff(
    owner: str,
    repo: str,
    commit_sha: str,
    github_token: str | None = None,
) -> dict:
    """Fetch commit diff from GitHub API.

    Args:
        owner: Repository owner
        repo: Repository name
        commit_sha: Commit SHA
        github_token: Optional GitHub token for authentication

    Returns:
        Commit data with files and stats
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Argus-Quality-Intelligence",
    }

    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)

            if response.status_code == 404:
                logger.warning("Commit not found", commit_sha=commit_sha)
                return {}

            if not response.is_success:
                logger.error(
                    "GitHub API error",
                    status=response.status_code,
                    error=response.text[:200],
                )
                return {}

            return response.json()

    except Exception as e:
        logger.exception("Failed to fetch GitHub diff", error=str(e))
        return {}


def parse_commit_files(commit_data: dict) -> list[str]:
    """Extract changed files from commit data."""
    files = commit_data.get("files", [])
    return [f.get("filename", "") for f in files if f.get("filename")]


def parse_commit_stats(commit_data: dict) -> dict:
    """Extract commit statistics."""
    stats = commit_data.get("stats", {})
    return {
        "lines_added": stats.get("additions", 0),
        "lines_deleted": stats.get("deletions", 0),
        "files_changed": len(commit_data.get("files", [])),
    }


def extract_components(files: list[str]) -> list[str]:
    """Extract component names from file paths."""
    components = set()

    for file_path in files:
        parts = file_path.split("/")
        if len(parts) > 1:
            # First directory is often the component
            component = parts[0]
            if component not in (".", "..", "src", "lib", "tests", "test"):
                components.add(component)

            # Check for known component patterns
            if "src" in parts:
                idx = parts.index("src")
                if idx + 1 < len(parts):
                    components.add(parts[idx + 1])

    return list(components)


async def get_affected_tests(
    project_id: str,
    files: list[str],
    min_score: float = 0.3,
    limit: int = 50,
) -> list[dict]:
    """Query impact graph for affected tests.

    Args:
        project_id: Project ID
        files: List of changed files
        min_score: Minimum impact score
        limit: Maximum tests to return

    Returns:
        List of affected tests with scores
    """
    supabase = get_supabase_client()

    result = await supabase.rpc(
        "get_affected_tests",
        {
            "p_project_id": project_id,
            "p_file_paths": files,
            "p_min_score": min_score,
            "p_limit": limit,
        },
    )

    if result.get("error"):
        error_msg = str(result.get("error", ""))
        if "does not exist" in error_msg or "42P01" in error_msg:
            return []
        logger.error("Failed to get affected tests", error=error_msg)
        return []

    return result.get("data") or []


async def get_predicted_failures(
    project_id: str,
    commit_sha: str,
    commit_features: dict,
) -> list[dict]:
    """Get failure predictions from pattern matching.

    Args:
        project_id: Project ID
        commit_sha: Commit SHA
        commit_features: Extracted commit features

    Returns:
        List of predicted failures
    """
    supabase = get_supabase_client()

    # First, store commit features
    features_record = {
        "project_id": project_id,
        "commit_sha": commit_sha,
        **commit_features,
    }

    await supabase.request(
        "/commit_features",
        method="POST",
        body=features_record,
        headers={"Prefer": "resolution=merge-duplicates"},
    )

    # Then query for matching patterns
    result = await supabase.rpc(
        "find_matching_patterns",
        {
            "p_project_id": project_id,
            "p_commit_sha": commit_sha,
        },
    )

    if result.get("error"):
        error_msg = str(result.get("error", ""))
        if "does not exist" in error_msg or "42P01" in error_msg:
            return []
        logger.error("Failed to get predicted failures", error=error_msg)
        return []

    patterns = result.get("data") or []

    # Convert to predictions
    predictions = []
    for pattern in patterns:
        if pattern.get("match_score", 0) > 0:
            predictions.append({
                "pattern_id": pattern["pattern_id"],
                "pattern_type": pattern["pattern_type"],
                "failure_probability": float(pattern["confidence"]) * float(pattern["match_score"]),
                "reason": pattern.get("match_reason", "Pattern matched"),
            })

    return predictions


def calculate_risk_score(
    stats: dict,
    predictions: list[dict],
    security_issues: list[dict],
    affected_tests: list[dict],
) -> tuple[float, list[dict]]:
    """Calculate overall risk score for a commit.

    Args:
        stats: Commit statistics
        predictions: Failure predictions
        security_issues: Security vulnerabilities found
        affected_tests: Tests that may be affected

    Returns:
        Tuple of (risk_score, risk_factors)
    """
    risk_factors = []
    total_risk = 0.0

    # Size-based risk
    lines_changed = stats.get("lines_added", 0) + stats.get("lines_deleted", 0)
    files_changed = stats.get("files_changed", 0)

    if lines_changed > 500:
        factor_score = min(0.3, lines_changed / 2000)
        total_risk += factor_score
        risk_factors.append({
            "factor": "large_change",
            "score": round(factor_score, 2),
            "description": f"Large commit with {lines_changed} lines changed",
        })

    if files_changed > 20:
        factor_score = min(0.2, files_changed / 100)
        total_risk += factor_score
        risk_factors.append({
            "factor": "many_files",
            "score": round(factor_score, 2),
            "description": f"Many files changed ({files_changed})",
        })

    # Prediction-based risk
    if predictions:
        max_failure_prob = max(p.get("failure_probability", 0) for p in predictions)
        if max_failure_prob > 0.5:
            factor_score = max_failure_prob * 0.3
            total_risk += factor_score
            risk_factors.append({
                "factor": "predicted_failures",
                "score": round(factor_score, 2),
                "description": f"Pattern analysis predicts {len(predictions)} potential failures",
            })

    # Security-based risk
    if security_issues:
        critical_count = sum(1 for i in security_issues if i.get("severity") == "critical")
        high_count = sum(1 for i in security_issues if i.get("severity") == "high")

        if critical_count > 0:
            factor_score = min(0.5, critical_count * 0.25)
            total_risk += factor_score
            risk_factors.append({
                "factor": "critical_security",
                "score": round(factor_score, 2),
                "description": f"{critical_count} critical security issue(s) detected",
            })
        elif high_count > 0:
            factor_score = min(0.3, high_count * 0.15)
            total_risk += factor_score
            risk_factors.append({
                "factor": "high_security",
                "score": round(factor_score, 2),
                "description": f"{high_count} high severity security issue(s) detected",
            })

    # Test coverage risk
    if affected_tests and len(affected_tests) > 10:
        # Many affected tests suggests broad changes
        factor_score = min(0.15, len(affected_tests) / 200)
        total_risk += factor_score
        risk_factors.append({
            "factor": "broad_impact",
            "score": round(factor_score, 2),
            "description": f"Changes affect {len(affected_tests)} tests",
        })

    # Normalize to 0-1 range
    final_risk = min(1.0, total_risk)

    return round(final_risk, 2), risk_factors


def determine_deployment_strategy(
    risk_score: float,
    security_issues: list[dict],
) -> tuple[str, str]:
    """Determine recommended deployment strategy based on risk.

    Args:
        risk_score: Overall risk score
        security_issues: Security vulnerabilities found

    Returns:
        Tuple of (strategy, notes)
    """
    # Critical security issues block deployment
    critical_security = [i for i in security_issues if i.get("severity") == "critical"]
    if critical_security:
        return (
            "blocked",
            f"Blocked due to {len(critical_security)} critical security issue(s). "
            "Resolve before deploying."
        )

    high_security = [i for i in security_issues if i.get("severity") == "high"]
    if high_security:
        return (
            "manual_review",
            f"Manual review required due to {len(high_security)} high severity "
            "security issue(s)."
        )

    if risk_score >= 0.7:
        return (
            "staged_rollout",
            "High risk commit. Use staged rollout with 5-10% initial traffic. "
            "Monitor error rates closely."
        )

    if risk_score >= 0.4:
        return (
            "deploy_with_monitoring",
            "Medium risk commit. Deploy with enhanced monitoring. "
            "Watch for test failures and error rate increases."
        )

    return (
        "safe_to_deploy",
        "Low risk commit. Safe to deploy using standard process."
    )


async def analyze_commit(
    project_id: str,
    commit_sha: str,
    pr_number: int | None = None,
    branch_name: str | None = None,
    repository: str | None = None,
) -> dict:
    """Analyze a commit for risk, tests to run, and predicted failures.

    Args:
        project_id: Project ID
        commit_sha: Git commit SHA
        pr_number: Optional PR number
        branch_name: Branch name
        repository: Repository in owner/repo format

    Returns:
        Analysis result
    """
    supabase = get_supabase_client()
    settings = get_settings()

    logger.info(
        "Starting commit analysis",
        project_id=project_id,
        commit_sha=commit_sha,
        pr_number=pr_number,
    )

    # Fetch commit diff from GitHub
    github_token = settings.github_token.get_secret_value() if settings.github_token else None
    commit_data = {}
    files_changed: list[str] = []
    stats = {"lines_added": 0, "lines_deleted": 0, "files_changed": 0}

    if repository:
        owner, repo = repository.split("/", 1) if "/" in repository else (repository, "")
        if owner and repo:
            commit_data = await fetch_github_diff(owner, repo, commit_sha, github_token)
            files_changed = parse_commit_files(commit_data)
            stats = parse_commit_stats(commit_data)

    # Extract components
    components = extract_components(files_changed)

    # Build commit features for pattern matching
    commit_timestamp = datetime.now(UTC)
    commit_features = {
        "author_email": commit_data.get("commit", {}).get("author", {}).get("email"),
        "author_name": commit_data.get("commit", {}).get("author", {}).get("name"),
        "commit_timestamp": commit_timestamp.isoformat(),
        "day_of_week": commit_timestamp.weekday(),
        "hour_of_day": commit_timestamp.hour,
        "files_changed": stats["files_changed"],
        "lines_added": stats["lines_added"],
        "lines_deleted": stats["lines_deleted"],
        "total_lines_changed": stats["lines_added"] + stats["lines_deleted"],
        "changed_files": files_changed,
        "changed_directories": list(set(
            "/".join(f.split("/")[:-1]) for f in files_changed if "/" in f
        )),
        "components_affected": components,
        "is_dependency_update": any(
            f in ["package.json", "requirements.txt", "Pipfile", "go.mod", "Cargo.toml"]
            for f in files_changed
        ),
        "touches_tests": any("test" in f.lower() or "spec" in f.lower() for f in files_changed),
        "touches_config": any(
            f.endswith((".yml", ".yaml", ".toml", ".ini", ".env")) or "config" in f.lower()
            for f in files_changed
        ),
        "touches_migrations": any(
            "migration" in f.lower() or "migrate" in f.lower()
            for f in files_changed
        ),
    }

    # Get affected tests from impact graph
    affected_tests = await get_affected_tests(project_id, files_changed)

    # Get failure predictions from pattern matching
    predictions = await get_predicted_failures(project_id, commit_sha, commit_features)

    # TODO: Add security analysis (e.g., using Semgrep or custom rules)
    security_issues: list[dict] = []

    # Calculate risk score
    risk_score, risk_factors = calculate_risk_score(
        stats, predictions, security_issues, affected_tests
    )

    # Determine deployment strategy
    deployment_strategy, deployment_notes = determine_deployment_strategy(
        risk_score, security_issues
    )

    # Build recommendations
    recommendations = []

    if predictions:
        recommendations.append({
            "type": "run_tests",
            "priority": "high",
            "description": f"Pattern analysis found {len(predictions)} potential failure patterns",
            "suggested_action": "Run the suggested test suite before merging",
        })

    if not affected_tests and files_changed:
        recommendations.append({
            "type": "add_coverage",
            "priority": "medium",
            "description": "No tests found for changed files",
            "suggested_action": "Consider adding tests for the changed code",
        })

    if stats["lines_added"] + stats["lines_deleted"] > 500:
        recommendations.append({
            "type": "split_commit",
            "priority": "low",
            "description": "Large commit with many changes",
            "suggested_action": "Consider splitting into smaller, focused commits",
        })

    # Prepare tests to run
    tests_to_run = [
        {
            "test_name": t.get("test_name"),
            "test_file": t.get("test_file_path"),
            "impact_score": float(t.get("total_impact_score", 0)),
            "reason": f"Affected by changes to: {', '.join(t.get('affected_files', [])[:3])}",
        }
        for t in affected_tests
    ]

    # Prepare predicted failures
    predicted_failures = [
        {
            "pattern_id": p.get("pattern_id"),
            "failure_probability": float(p.get("failure_probability", 0)),
            "reason": p.get("reason"),
        }
        for p in predictions
    ]

    # Store analysis result
    analysis_record = {
        "project_id": project_id,
        "commit_sha": commit_sha,
        "pr_number": pr_number,
        "branch_name": branch_name,
        "files_changed": stats["files_changed"],
        "lines_added": stats["lines_added"],
        "lines_deleted": stats["lines_deleted"],
        "affected_components": components,
        "predicted_test_failures": predicted_failures,
        "tests_to_run_suggested": tests_to_run,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "security_vulnerabilities": security_issues,
        "security_risk_score": 0.0,  # TODO: Calculate from security_issues
        "recommendations": recommendations,
        "deployment_strategy": deployment_strategy,
        "deployment_notes": deployment_notes,
        "analyzed_at": datetime.now(UTC).isoformat(),
    }

    result = await supabase.request(
        "/commit_analyses",
        method="POST",
        body=analysis_record,
        headers={"Prefer": "resolution=merge-duplicates,return=representation"},
    )

    if result.get("error"):
        # Try to get existing analysis
        existing = await supabase.request(
            f"/commit_analyses?project_id=eq.{project_id}&commit_sha=eq.{commit_sha}"
        )
        if existing.get("data"):
            analysis_record = existing["data"][0]
        else:
            logger.error("Failed to store commit analysis", error=result["error"])
            analysis_record["id"] = None
    else:
        analysis_record = result.get("data", [analysis_record])[0]

    logger.info(
        "Commit analysis completed",
        commit_sha=commit_sha,
        risk_score=risk_score,
        deployment_strategy=deployment_strategy,
        tests_to_run=len(tests_to_run),
        predicted_failures=len(predicted_failures),
    )

    return analysis_record


# =============================================================================
# Event Handlers
# =============================================================================


async def handle_push_event(
    project_id: str,
    payload: dict,
    delivery_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    """Handle GitHub push event.

    Creates SDLC events for each commit and triggers analysis for head commit.
    """
    ref = payload.get("ref", "")
    branch_name = ref.replace("refs/heads/", "")
    repository = payload.get("repository", {})
    repo_full_name = repository.get("full_name", "")

    commits = payload.get("commits", [])
    head_commit = payload.get("head_commit", {})

    logger.info(
        "Processing push event",
        delivery_id=delivery_id,
        repository=repo_full_name,
        branch=branch_name,
        commits_count=len(commits),
    )

    sdlc_event_id = None

    # Store SDLC event for the push
    push_event = await store_sdlc_event(
        project_id=project_id,
        event_type="push",
        external_id=delivery_id,
        payload={
            "ref": ref,
            "branch_name": branch_name,
            "repository": repo_full_name,
            "commits_count": len(commits),
            "head_sha": head_commit.get("id"),
            "pusher": payload.get("pusher", {}).get("name"),
            "html_url": repository.get("html_url"),
        },
        occurred_at=datetime.fromisoformat(
            head_commit.get("timestamp", datetime.now(UTC).isoformat()).replace("Z", "+00:00")
        ) if head_commit.get("timestamp") else None,
    )

    if push_event:
        sdlc_event_id = push_event.get("id")

    # Trigger analysis for head commit if present
    commit_analysis_id = None
    if head_commit.get("id"):
        analysis = await analyze_commit(
            project_id=project_id,
            commit_sha=head_commit["id"],
            branch_name=branch_name,
            repository=repo_full_name,
        )
        commit_analysis_id = analysis.get("id")

    return {
        "sdlc_event_id": sdlc_event_id,
        "commit_analysis_id": commit_analysis_id,
    }


async def handle_pull_request_event(
    project_id: str,
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
        "Processing pull_request event",
        delivery_id=delivery_id,
        action=action,
        pr_number=pr_number,
    )

    # Store SDLC event
    sdlc_event = await store_sdlc_event(
        project_id=project_id,
        event_type="pr",
        external_id=f"{repo_full_name}/pull/{pr_number}#{action}",
        payload={
            "action": action,
            "pr_number": pr_number,
            "title": pr.get("title"),
            "body": pr.get("body"),
            "state": pr.get("state"),
            "draft": pr.get("draft"),
            "head_sha": pr.get("head", {}).get("sha"),
            "base_branch": pr.get("base", {}).get("ref"),
            "head_branch": pr.get("head", {}).get("ref"),
            "author": pr.get("user", {}).get("login"),
            "html_url": pr.get("html_url"),
            "branch_name": pr.get("head", {}).get("ref"),
            "commit_sha": pr.get("head", {}).get("sha"),
        },
        occurred_at=datetime.fromisoformat(
            pr.get("updated_at", datetime.now(UTC).isoformat()).replace("Z", "+00:00")
        ) if pr.get("updated_at") else None,
    )

    sdlc_event_id = sdlc_event.get("id") if sdlc_event else None

    # Trigger analysis for opened, synchronize, or reopened PRs
    commit_analysis_id = None
    if action in ("opened", "synchronize", "reopened"):
        head_sha = pr.get("head", {}).get("sha")
        head_branch = pr.get("head", {}).get("ref")

        if head_sha:
            analysis = await analyze_commit(
                project_id=project_id,
                commit_sha=head_sha,
                pr_number=pr_number,
                branch_name=head_branch,
                repository=repo_full_name,
            )
            commit_analysis_id = analysis.get("id")

    return {
        "sdlc_event_id": sdlc_event_id,
        "commit_analysis_id": commit_analysis_id,
    }


async def handle_check_run_event(
    project_id: str,
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
        "Processing check_run event",
        delivery_id=delivery_id,
        action=action,
        name=check_run.get("name"),
    )

    # Store SDLC event
    sdlc_event = await store_sdlc_event(
        project_id=project_id,
        event_type="check_run",
        external_id=f"{repo_full_name}/check/{check_run.get('id')}",
        payload={
            "action": action,
            "check_run_id": check_run.get("id"),
            "name": check_run.get("name"),
            "status": check_run.get("status"),
            "conclusion": check_run.get("conclusion"),
            "head_sha": check_run.get("head_sha"),
            "html_url": check_run.get("html_url"),
            "commit_sha": check_run.get("head_sha"),
        },
        occurred_at=datetime.fromisoformat(
            check_run.get("completed_at", check_run.get("started_at", datetime.now(UTC).isoformat()))
            .replace("Z", "+00:00")
        ),
    )

    return {
        "sdlc_event_id": sdlc_event.get("id") if sdlc_event else None,
        "commit_analysis_id": None,
    }


async def handle_deployment_event(
    project_id: str,
    payload: dict,
    delivery_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    """Handle GitHub deployment event."""
    deployment = payload.get("deployment", {})
    repository = payload.get("repository", {})
    repo_full_name = repository.get("full_name", "")

    logger.info(
        "Processing deployment event",
        delivery_id=delivery_id,
        environment=deployment.get("environment"),
    )

    # Store SDLC event
    sdlc_event = await store_sdlc_event(
        project_id=project_id,
        event_type="deploy",
        external_id=f"{repo_full_name}/deployment/{deployment.get('id')}",
        payload={
            "deployment_id": deployment.get("id"),
            "environment": deployment.get("environment"),
            "description": deployment.get("description"),
            "sha": deployment.get("sha"),
            "ref": deployment.get("ref"),
            "task": deployment.get("task"),
            "creator": deployment.get("creator", {}).get("login"),
            "commit_sha": deployment.get("sha"),
            "deploy_id": str(deployment.get("id")),
        },
        occurred_at=datetime.fromisoformat(
            deployment.get("created_at", datetime.now(UTC).isoformat()).replace("Z", "+00:00")
        ) if deployment.get("created_at") else None,
    )

    return {
        "sdlc_event_id": sdlc_event.get("id") if sdlc_event else None,
        "commit_analysis_id": None,
    }


async def handle_deployment_status_event(
    project_id: str,
    payload: dict,
    delivery_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    """Handle GitHub deployment_status event."""
    deployment_status = payload.get("deployment_status", {})
    deployment = payload.get("deployment", {})
    repository = payload.get("repository", {})
    repo_full_name = repository.get("full_name", "")

    logger.info(
        "Processing deployment_status event",
        delivery_id=delivery_id,
        state=deployment_status.get("state"),
    )

    # Store SDLC event
    sdlc_event = await store_sdlc_event(
        project_id=project_id,
        event_type="deployment_status",
        external_id=f"{repo_full_name}/deployment_status/{deployment_status.get('id')}",
        payload={
            "deployment_status_id": deployment_status.get("id"),
            "deployment_id": deployment.get("id"),
            "state": deployment_status.get("state"),
            "description": deployment_status.get("description"),
            "environment": deployment_status.get("environment"),
            "target_url": deployment_status.get("target_url"),
            "sha": deployment.get("sha"),
            "commit_sha": deployment.get("sha"),
            "deploy_id": str(deployment.get("id")),
        },
        occurred_at=datetime.fromisoformat(
            deployment_status.get("created_at", datetime.now(UTC).isoformat())
            .replace("Z", "+00:00")
        ) if deployment_status.get("created_at") else None,
    )

    return {
        "sdlc_event_id": sdlc_event.get("id") if sdlc_event else None,
        "commit_analysis_id": None,
    }


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("", response_model=GitHubWebhookResponse)
async def receive_github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str | None = Header(None, alias="X-GitHub-Event"),
    x_github_delivery: str | None = Header(None, alias="X-GitHub-Delivery"),
    x_hub_signature_256: str | None = Header(None, alias="X-Hub-Signature-256"),
    project_id: str = Query(..., description="Project ID for this webhook"),
):
    """Receive and process GitHub webhook events.

    Supports events: push, pull_request, check_run, deployment, deployment_status

    Security: Verifies HMAC SHA256 signature using GITHUB_WEBHOOK_SECRET env var.
    """
    settings = get_settings()

    # Validate required headers
    if not x_github_event:
        raise HTTPException(status_code=400, detail="Missing X-GitHub-Event header")

    if not x_github_delivery:
        raise HTTPException(status_code=400, detail="Missing X-GitHub-Delivery header")

    # Read raw body for signature verification
    body = await request.body()

    # Verify signature if webhook secret is configured
    webhook_secret = settings.github_token.get_secret_value() if settings.github_token else None

    # Allow GitHub webhook secret to be different from GitHub token
    # Check for GITHUB_WEBHOOK_SECRET env var first
    import os
    webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET", webhook_secret)

    if webhook_secret and x_hub_signature_256:
        if not verify_github_signature(body, x_hub_signature_256, webhook_secret):
            logger.warning(
                "Invalid GitHub webhook signature",
                delivery_id=x_github_delivery,
                event_type=x_github_event,
            )
            raise HTTPException(status_code=401, detail="Invalid signature")

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
    )

    # Store raw event
    await store_webhook_event(
        delivery_id=x_github_delivery,
        event_type=x_github_event,
        payload=payload,
        project_id=project_id,
    )

    # Route to appropriate handler
    handler_map = {
        "push": handle_push_event,
        "pull_request": handle_pull_request_event,
        "check_run": handle_check_run_event,
        "deployment": handle_deployment_event,
        "deployment_status": handle_deployment_status_event,
    }

    handler = handler_map.get(x_github_event)

    if not handler:
        logger.info(
            "Unsupported webhook event type",
            event_type=x_github_event,
            delivery_id=x_github_delivery,
        )
        await update_webhook_status(x_github_delivery, "skipped")
        return GitHubWebhookResponse(
            success=True,
            message=f"Event type '{x_github_event}' not processed",
            event_type=x_github_event,
            delivery_id=x_github_delivery,
        )

    try:
        result = await handler(project_id, payload, x_github_delivery, background_tasks)

        await update_webhook_status(
            delivery_id=x_github_delivery,
            status="completed",
            sdlc_event_id=result.get("sdlc_event_id"),
            commit_analysis_id=result.get("commit_analysis_id"),
        )

        return GitHubWebhookResponse(
            success=True,
            message=f"Successfully processed {x_github_event} event",
            event_type=x_github_event,
            delivery_id=x_github_delivery,
            sdlc_event_id=result.get("sdlc_event_id"),
            commit_analysis_id=result.get("commit_analysis_id"),
        )

    except Exception as e:
        logger.exception(
            "Failed to process webhook",
            event_type=x_github_event,
            delivery_id=x_github_delivery,
            error=str(e),
        )

        await update_webhook_status(
            delivery_id=x_github_delivery,
            status="failed",
            error_message=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process webhook: {str(e)}"
        )


@router.post("/analyze", response_model=CommitAnalysisResponse)
async def analyze_commit_endpoint(
    request_body: CommitAnalysisRequest,
    project_id: str = Query(..., description="Project ID"),
):
    """Manually trigger commit analysis.

    Use this endpoint to analyze a specific commit without waiting for a webhook.
    """
    analysis = await analyze_commit(
        project_id=project_id,
        commit_sha=request_body.commit_sha,
        pr_number=request_body.pr_number,
        branch_name=request_body.branch_name,
        repository=request_body.repository,
    )

    if not analysis.get("id"):
        raise HTTPException(status_code=500, detail="Failed to create commit analysis")

    return CommitAnalysisResponse(
        id=analysis["id"],
        commit_sha=analysis["commit_sha"],
        pr_number=analysis.get("pr_number"),
        risk_score=float(analysis.get("risk_score", 0)),
        deployment_strategy=analysis.get("deployment_strategy", "safe_to_deploy"),
        tests_to_run_count=len(analysis.get("tests_to_run_suggested", [])),
        predicted_failures_count=len(analysis.get("predicted_test_failures", [])),
        security_issues_count=len(analysis.get("security_vulnerabilities", [])),
        recommendations_count=len(analysis.get("recommendations", [])),
    )


@router.get("/analysis/{commit_sha}")
async def get_commit_analysis(
    commit_sha: str,
    project_id: str = Query(..., description="Project ID"),
):
    """Get analysis results for a specific commit."""
    supabase = get_supabase_client()

    result = await supabase.request(
        f"/commit_analyses?project_id=eq.{project_id}&commit_sha=eq.{commit_sha}"
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch analysis")

    if not result.get("data"):
        raise HTTPException(status_code=404, detail="Analysis not found")

    return result["data"][0]


@router.get("/events")
async def list_webhook_events(
    project_id: str = Query(..., description="Project ID"),
    event_type: str | None = Query(None, description="Filter by event type"),
    status: str | None = Query(None, description="Filter by processing status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum events to return"),
):
    """List recent GitHub webhook events."""
    supabase = get_supabase_client()

    query_path = f"/github_webhook_events?project_id=eq.{project_id}"

    if event_type:
        query_path += f"&event_type=eq.{event_type}"
    if status:
        query_path += f"&status=eq.{status}"

    query_path += f"&order=created_at.desc&limit={limit}"

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
