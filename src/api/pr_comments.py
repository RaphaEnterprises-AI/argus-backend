"""PR Comment Bot for Argus Quality Intelligence Platform.

Provides endpoints and utilities for:
- Formatting commit analysis as rich GitHub PR comments
- Posting/updating Argus comments on PRs via GitHub REST API
- Manual trigger endpoint for posting analysis to associated PRs

RAP-91: PR Comment Bot implementation.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.context import get_current_organization_id
from src.api.teams import get_current_user, verify_org_access
from src.services.key_encryption import decrypt_api_key
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/pr-comments", tags=["PR Comments"])

# Argus comment marker for identification
ARGUS_COMMENT_MARKER = "<!-- argus-quality-intelligence -->"

# GitHub API base URL
GITHUB_API_BASE = "https://api.github.com"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TestPrediction:
    """Predicted test failure."""

    test_name: str
    failure_probability: float
    reason: str


@dataclass
class SecurityVulnerability:
    """Detected security vulnerability."""

    severity: Literal["critical", "high", "medium", "low", "info"]
    message: str
    file: str | None = None
    line: int | None = None


@dataclass
class CommitAnalysis:
    """Analysis results for a commit."""

    commit_sha: str
    pr_number: int
    files_changed: int
    risk_score: float
    predicted_test_failures: list[TestPrediction] = field(default_factory=list)
    security_vulnerabilities: list[SecurityVulnerability] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    deployment_strategy: Literal["canary", "blue_green", "direct"] = "direct"
    analysis_url: str | None = None


# =============================================================================
# Request/Response Models
# =============================================================================


class PostToPRRequest(BaseModel):
    """Request to post analysis to a PR."""

    github_token: str | None = Field(
        None,
        description="GitHub token for API access. If not provided, uses project integration.",
    )
    repo_owner: str | None = Field(
        None,
        description="Repository owner. If not provided, uses project settings.",
    )
    repo_name: str | None = Field(
        None,
        description="Repository name. If not provided, uses project settings.",
    )
    pr_number: int | None = Field(
        None,
        description="PR number. If not provided, attempts to find PR for commit.",
    )


class PostToPRResponse(BaseModel):
    """Response after posting to PR."""

    success: bool
    message: str
    comment_id: int | None = None
    comment_url: str | None = None
    action: Literal["created", "updated", "skipped"] | None = None


class CommitAnalysisResponse(BaseModel):
    """Response containing commit analysis data."""

    commit_sha: str
    pr_number: int | None
    files_changed: int
    risk_score: float
    predicted_test_failures: list[dict]
    security_vulnerabilities: list[dict]
    recommendations: list[str]
    deployment_strategy: str
    created_at: str | None = None


# =============================================================================
# Comment Formatting
# =============================================================================


def _get_risk_emoji(probability: float) -> str:
    """Get emoji for risk probability."""
    if probability >= 0.7:
        return "\U0001f534"  # Red circle
    elif probability >= 0.4:
        return "\U0001f7e1"  # Yellow circle
    else:
        return "\U0001f7e2"  # Green circle


def _get_risk_label(score: float) -> tuple[str, str]:
    """Get risk label and emoji for overall risk score."""
    if score >= 0.7:
        return "High", "\U0001f534"
    elif score >= 0.4:
        return "Medium", "\U0001f7e1"
    else:
        return "Low", "\U0001f7e2"


def _get_severity_emoji(severity: str) -> str:
    """Get emoji for security severity."""
    severity_map = {
        "critical": "\U0001f6a8",  # Rotating light
        "high": "\u26a0\ufe0f",  # Warning
        "medium": "\u26a0\ufe0f",  # Warning
        "low": "\u2139\ufe0f",  # Info
        "info": "\u2139\ufe0f",  # Info
    }
    return severity_map.get(severity.lower(), "\u2139\ufe0f")


def _format_deployment_strategy(strategy: str) -> str:
    """Format deployment strategy for display."""
    strategy_map = {
        "canary": "Canary 10%",
        "blue_green": "Blue/Green",
        "direct": "Direct Deploy",
    }
    return strategy_map.get(strategy, strategy.replace("_", " ").title())


def format_pr_comment(analysis: CommitAnalysis) -> str:
    """Format analysis as a rich GitHub PR comment.

    Args:
        analysis: The commit analysis to format

    Returns:
        Formatted markdown string for GitHub PR comment
    """
    lines = [
        ARGUS_COMMENT_MARKER,
        "",
        "## \U0001f52e Argus Commit Impact Analysis",
        "",
    ]

    # Test Predictions section
    if analysis.predicted_test_failures:
        lines.extend([
            "### \U0001f9ea Test Predictions",
            "",
            "| Test | Failure Risk | Reason |",
            "|------|-------------|--------|",
        ])

        for pred in analysis.predicted_test_failures:
            emoji = _get_risk_emoji(pred.failure_probability)
            risk_percent = int(pred.failure_probability * 100)
            # Escape pipe characters in test name and reason
            test_name = pred.test_name.replace("|", "\\|")
            reason = pred.reason.replace("|", "\\|")
            lines.append(f"| `{test_name}` | {emoji} {risk_percent}% | {reason} |")

        # Calculate recommended tests to run
        high_risk_count = sum(
            1 for p in analysis.predicted_test_failures if p.failure_probability >= 0.4
        )
        if high_risk_count > 0:
            estimated_time = high_risk_count * 2  # Rough estimate: 2 min per test
            lines.extend([
                "",
                f"**Recommended**: Run {high_risk_count} tests before merge (~{estimated_time} min)",
            ])
        lines.append("")
    else:
        lines.extend([
            "### \U0001f9ea Test Predictions",
            "",
            "\u2705 No high-risk test failures predicted.",
            "",
        ])

    # Security Analysis section
    lines.append("### \U0001f512 Security Analysis")
    lines.append("")

    if analysis.security_vulnerabilities:
        for vuln in analysis.security_vulnerabilities:
            emoji = _get_severity_emoji(vuln.severity)
            severity_display = vuln.severity.capitalize()
            location = ""
            if vuln.file:
                location = f" in `{vuln.file}"
                if vuln.line:
                    location += f":{vuln.line}"
                location += "`"
            lines.append(f"- {emoji} **{severity_display}**: {vuln.message}{location}")
    else:
        lines.append("\u2705 No security issues detected")

    lines.append("")

    # Production Risk Assessment section
    risk_label, risk_emoji = _get_risk_label(analysis.risk_score)
    deployment_display = _format_deployment_strategy(analysis.deployment_strategy)

    lines.extend([
        "### \U0001f4c8 Production Risk Assessment",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Overall Risk | {risk_emoji} {risk_label} ({analysis.risk_score:.2f}) |",
        f"| Files Changed | {analysis.files_changed} |",
        f"| Recommended Strategy | {deployment_display} |",
        "",
    ])

    # Recommendations section
    if analysis.recommendations:
        lines.extend([
            "### \U0001f4a1 Recommendations",
            "",
        ])
        for i, rec in enumerate(analysis.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    # Footer
    lines.extend([
        "---",
    ])

    if analysis.analysis_url:
        lines.append(
            f"<sub>\U0001f916 Powered by [Argus Quality Intelligence]({analysis.analysis_url})</sub>"
        )
    else:
        lines.append(
            "<sub>\U0001f916 Powered by [Argus Quality Intelligence](https://heyargus.com)</sub>"
        )

    return "\n".join(lines)


# =============================================================================
# GitHub API Integration
# =============================================================================


async def find_argus_comment(
    repo_owner: str,
    repo_name: str,
    pr_number: int,
    github_token: str,
) -> int | None:
    """Find existing Argus comment ID on a PR.

    Args:
        repo_owner: Repository owner
        repo_name: Repository name
        pr_number: Pull request number
        github_token: GitHub API token

    Returns:
        Comment ID if found, None otherwise
    """
    url = f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Paginate through comments to find ours
            page = 1
            while True:
                response = await client.get(
                    url,
                    headers=headers,
                    params={"page": page, "per_page": 100},
                )

                if response.status_code != 200:
                    logger.warning(
                        "Failed to fetch PR comments",
                        status=response.status_code,
                        repo=f"{repo_owner}/{repo_name}",
                        pr=pr_number,
                    )
                    return None

                comments = response.json()
                if not comments:
                    break

                for comment in comments:
                    body = comment.get("body", "")
                    if ARGUS_COMMENT_MARKER in body:
                        logger.debug(
                            "Found existing Argus comment",
                            comment_id=comment["id"],
                            pr=pr_number,
                        )
                        return comment["id"]

                # Check if there are more pages
                if len(comments) < 100:
                    break
                page += 1

        except httpx.RequestError as e:
            logger.error(
                "HTTP error fetching PR comments",
                error=str(e),
                repo=f"{repo_owner}/{repo_name}",
                pr=pr_number,
            )

    return None


async def post_analysis_to_pr(
    repo_owner: str,
    repo_name: str,
    pr_number: int,
    analysis: CommitAnalysis,
    github_token: str,
) -> dict:
    """Post or update Argus comment on a PR.

    Args:
        repo_owner: Repository owner
        repo_name: Repository name
        pr_number: Pull request number
        analysis: Commit analysis to post
        github_token: GitHub API token

    Returns:
        Dict with success status, message, comment_id, and action taken
    """
    comment_body = format_pr_comment(analysis)

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check for existing comment
        existing_comment_id = await find_argus_comment(
            repo_owner, repo_name, pr_number, github_token
        )

        try:
            if existing_comment_id:
                # Update existing comment
                url = f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}/issues/comments/{existing_comment_id}"
                response = await client.patch(
                    url,
                    headers=headers,
                    json={"body": comment_body},
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(
                        "Updated Argus PR comment",
                        comment_id=existing_comment_id,
                        repo=f"{repo_owner}/{repo_name}",
                        pr=pr_number,
                    )
                    return {
                        "success": True,
                        "message": "Comment updated successfully",
                        "comment_id": existing_comment_id,
                        "comment_url": result.get("html_url"),
                        "action": "updated",
                    }
                else:
                    error_msg = response.text
                    logger.error(
                        "Failed to update PR comment",
                        status=response.status_code,
                        error=error_msg,
                        repo=f"{repo_owner}/{repo_name}",
                        pr=pr_number,
                    )
                    return {
                        "success": False,
                        "message": f"Failed to update comment: {error_msg}",
                        "comment_id": None,
                        "comment_url": None,
                        "action": None,
                    }
            else:
                # Create new comment
                url = f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}/issues/{pr_number}/comments"
                response = await client.post(
                    url,
                    headers=headers,
                    json={"body": comment_body},
                )

                if response.status_code == 201:
                    result = response.json()
                    comment_id = result.get("id")
                    logger.info(
                        "Created Argus PR comment",
                        comment_id=comment_id,
                        repo=f"{repo_owner}/{repo_name}",
                        pr=pr_number,
                    )
                    return {
                        "success": True,
                        "message": "Comment created successfully",
                        "comment_id": comment_id,
                        "comment_url": result.get("html_url"),
                        "action": "created",
                    }
                else:
                    error_msg = response.text
                    logger.error(
                        "Failed to create PR comment",
                        status=response.status_code,
                        error=error_msg,
                        repo=f"{repo_owner}/{repo_name}",
                        pr=pr_number,
                    )
                    return {
                        "success": False,
                        "message": f"Failed to create comment: {error_msg}",
                        "comment_id": None,
                        "comment_url": None,
                        "action": None,
                    }

        except httpx.RequestError as e:
            logger.error(
                "HTTP error posting PR comment",
                error=str(e),
                repo=f"{repo_owner}/{repo_name}",
                pr=pr_number,
            )
            return {
                "success": False,
                "message": f"HTTP error: {str(e)}",
                "comment_id": None,
                "comment_url": None,
                "action": None,
            }


async def find_pr_for_commit(
    repo_owner: str,
    repo_name: str,
    commit_sha: str,
    github_token: str,
) -> int | None:
    """Find PR number associated with a commit.

    Args:
        repo_owner: Repository owner
        repo_name: Repository name
        commit_sha: Commit SHA
        github_token: GitHub API token

    Returns:
        PR number if found, None otherwise
    """
    url = f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}/commits/{commit_sha}/pulls"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, headers=headers)

            if response.status_code == 200:
                pulls = response.json()
                if pulls:
                    # Return the first (most recent) PR
                    pr_number = pulls[0].get("number")
                    logger.debug(
                        "Found PR for commit",
                        commit=commit_sha[:8],
                        pr=pr_number,
                    )
                    return pr_number

            logger.debug(
                "No PR found for commit",
                commit=commit_sha[:8],
                status=response.status_code,
            )

        except httpx.RequestError as e:
            logger.error(
                "HTTP error finding PR for commit",
                error=str(e),
                commit=commit_sha[:8],
            )

    return None


# =============================================================================
# Helper Functions
# =============================================================================


async def get_github_credentials(
    supabase,
    project_id: str,
    provided_token: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Get GitHub credentials from project integration or provided token.

    Args:
        supabase: Supabase client
        project_id: Project ID
        provided_token: Optionally provided GitHub token

    Returns:
        Tuple of (token, repo_owner, repo_name)
    """
    if provided_token:
        return provided_token, None, None

    # Try to get from project's GitHub integration
    result = await supabase.request(
        f"/integrations?project_id=eq.{project_id}&type=eq.github&status=eq.connected&select=credentials,settings"
    )

    if result.get("data") and len(result["data"]) > 0:
        integration = result["data"][0]
        encrypted_creds = integration.get("credentials", {})
        settings = integration.get("settings", {})

        # Decrypt token
        token = None
        if encrypted_creds.get("token"):
            try:
                token = decrypt_api_key(encrypted_creds["token"])
            except Exception as e:
                logger.warning("Failed to decrypt GitHub token", error=str(e))

        repo_owner = settings.get("repo_owner")
        repo_name = settings.get("repo_name")

        return token, repo_owner, repo_name

    return None, None, None


async def get_commit_analysis(
    supabase,
    project_id: str,
    commit_sha: str,
) -> CommitAnalysis | None:
    """Fetch commit analysis from database.

    Args:
        supabase: Supabase client
        project_id: Project ID
        commit_sha: Commit SHA

    Returns:
        CommitAnalysis if found, None otherwise
    """
    result = await supabase.request(
        f"/commit_analyses?project_id=eq.{project_id}&commit_sha=eq.{commit_sha}&select=*&limit=1"
    )

    if not result.get("data") or len(result["data"]) == 0:
        return None

    data = result["data"][0]

    # Parse predictions
    predictions = []
    for pred in data.get("predicted_test_failures", []):
        predictions.append(
            TestPrediction(
                test_name=pred.get("test_name", "Unknown"),
                failure_probability=pred.get("failure_probability", 0.0),
                reason=pred.get("reason", "No reason provided"),
            )
        )

    # Parse vulnerabilities
    vulnerabilities = []
    for vuln in data.get("security_vulnerabilities", []):
        vulnerabilities.append(
            SecurityVulnerability(
                severity=vuln.get("severity", "info"),
                message=vuln.get("message", ""),
                file=vuln.get("file"),
                line=vuln.get("line"),
            )
        )

    return CommitAnalysis(
        commit_sha=data.get("commit_sha", commit_sha),
        pr_number=data.get("pr_number", 0),
        files_changed=data.get("files_changed", 0),
        risk_score=data.get("risk_score", 0.0),
        predicted_test_failures=predictions,
        security_vulnerabilities=vulnerabilities,
        recommendations=data.get("recommendations", []),
        deployment_strategy=data.get("deployment_strategy", "direct"),
        analysis_url=data.get("analysis_url"),
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/analysis/{project_id}/{commit_sha}", response_model=CommitAnalysisResponse)
async def get_analysis(
    project_id: str,
    commit_sha: str,
    request: Request,
):
    """
    Get commit analysis for a specific commit.

    Returns the analysis data that would be posted to a PR comment.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Verify project access
    project_result = await supabase.request(
        f"/projects?id=eq.{project_id}&select=organization_id"
    )
    if not project_result.get("data"):
        raise HTTPException(status_code=404, detail="Project not found")

    org_id = project_result["data"][0]["organization_id"]
    await verify_org_access(
        org_id,
        user["user_id"],
        user_email=user.get("email"),
        request=request,
    )

    # Get analysis
    analysis = await get_commit_analysis(supabase, project_id, commit_sha)

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found for commit {commit_sha[:8]}",
        )

    logger.info(
        "Retrieved commit analysis",
        project_id=project_id,
        commit=commit_sha[:8],
        user_id=user["user_id"],
    )

    return CommitAnalysisResponse(
        commit_sha=analysis.commit_sha,
        pr_number=analysis.pr_number if analysis.pr_number > 0 else None,
        files_changed=analysis.files_changed,
        risk_score=analysis.risk_score,
        predicted_test_failures=[
            {
                "test_name": p.test_name,
                "failure_probability": p.failure_probability,
                "reason": p.reason,
            }
            for p in analysis.predicted_test_failures
        ],
        security_vulnerabilities=[
            {
                "severity": v.severity,
                "message": v.message,
                "file": v.file,
                "line": v.line,
            }
            for v in analysis.security_vulnerabilities
        ],
        recommendations=analysis.recommendations,
        deployment_strategy=analysis.deployment_strategy,
    )


@router.post(
    "/projects/{project_id}/commits/{commit_sha}/post-to-pr",
    response_model=PostToPRResponse,
)
async def post_commit_analysis_to_pr(
    project_id: str,
    commit_sha: str,
    body: PostToPRRequest | None = None,
    request: Request = None,
):
    """
    Manually trigger posting analysis to associated PR.

    This endpoint:
    1. Fetches the commit analysis from the database
    2. Determines the associated PR (from analysis or by querying GitHub)
    3. Posts or updates the Argus comment on the PR

    If no GitHub token is provided in the request, it will use the
    project's configured GitHub integration.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Verify project access
    project_result = await supabase.request(
        f"/projects?id=eq.{project_id}&select=organization_id,name,settings"
    )
    if not project_result.get("data"):
        raise HTTPException(status_code=404, detail="Project not found")

    org_id = project_result["data"][0]["organization_id"]
    project_settings = project_result["data"][0].get("settings", {})
    await verify_org_access(
        org_id,
        user["user_id"],
        user_email=user.get("email"),
        request=request,
    )

    # Get commit analysis
    analysis = await get_commit_analysis(supabase, project_id, commit_sha)

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found for commit {commit_sha[:8]}. Run analysis first.",
        )

    # Get GitHub credentials
    body = body or PostToPRRequest()
    github_token, default_owner, default_name = await get_github_credentials(
        supabase, project_id, body.github_token
    )

    if not github_token:
        raise HTTPException(
            status_code=400,
            detail="No GitHub token available. Provide one in the request or configure GitHub integration.",
        )

    # Determine repo owner and name
    repo_owner = body.repo_owner or default_owner or project_settings.get("github_owner")
    repo_name = body.repo_name or default_name or project_settings.get("github_repo")

    if not repo_owner or not repo_name:
        raise HTTPException(
            status_code=400,
            detail="Repository owner and name required. Provide in request or configure in project settings.",
        )

    # Determine PR number
    pr_number = body.pr_number or analysis.pr_number

    if not pr_number or pr_number == 0:
        # Try to find PR for this commit
        pr_number = await find_pr_for_commit(
            repo_owner, repo_name, commit_sha, github_token
        )

    if not pr_number:
        raise HTTPException(
            status_code=404,
            detail=f"No PR found for commit {commit_sha[:8]}. Provide pr_number explicitly.",
        )

    # Update analysis with PR number if we found it
    analysis.pr_number = pr_number

    # Post to PR
    result = await post_analysis_to_pr(
        repo_owner,
        repo_name,
        pr_number,
        analysis,
        github_token,
    )

    if result["success"]:
        # Log the action
        logger.info(
            "Posted analysis to PR",
            project_id=project_id,
            commit=commit_sha[:8],
            pr=pr_number,
            action=result["action"],
            user_id=user["user_id"],
        )

        # Update analysis record with posted status
        await supabase.update(
            "commit_analyses",
            {"project_id": f"eq.{project_id}", "commit_sha": f"eq.{commit_sha}"},
            {
                "pr_comment_posted_at": datetime.now(UTC).isoformat(),
                "pr_comment_id": result["comment_id"],
                "pr_number": pr_number,
            },
        )

    return PostToPRResponse(
        success=result["success"],
        message=result["message"],
        comment_id=result["comment_id"],
        comment_url=result["comment_url"],
        action=result["action"],
    )


@router.get("/preview/{project_id}/{commit_sha}")
async def preview_pr_comment(
    project_id: str,
    commit_sha: str,
    request: Request,
):
    """
    Preview the PR comment that would be posted.

    Returns the formatted markdown without actually posting to GitHub.
    Useful for testing and verification.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Verify project access
    project_result = await supabase.request(
        f"/projects?id=eq.{project_id}&select=organization_id"
    )
    if not project_result.get("data"):
        raise HTTPException(status_code=404, detail="Project not found")

    org_id = project_result["data"][0]["organization_id"]
    await verify_org_access(
        org_id,
        user["user_id"],
        user_email=user.get("email"),
        request=request,
    )

    # Get analysis
    analysis = await get_commit_analysis(supabase, project_id, commit_sha)

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found for commit {commit_sha[:8]}",
        )

    # Format comment
    comment_body = format_pr_comment(analysis)

    logger.debug(
        "Generated PR comment preview",
        project_id=project_id,
        commit=commit_sha[:8],
    )

    return {
        "commit_sha": commit_sha,
        "pr_number": analysis.pr_number if analysis.pr_number > 0 else None,
        "markdown": comment_body,
        "character_count": len(comment_body),
    }


@router.post("/format")
async def format_comment(
    analysis_data: dict,
    request: Request,
):
    """
    Format arbitrary analysis data as a PR comment.

    This endpoint is useful for testing comment formatting without
    requiring stored analysis data.
    """
    await get_current_user(request)

    # Parse predictions
    predictions = []
    for pred in analysis_data.get("predicted_test_failures", []):
        predictions.append(
            TestPrediction(
                test_name=pred.get("test_name", "Unknown"),
                failure_probability=pred.get("failure_probability", 0.0),
                reason=pred.get("reason", "No reason provided"),
            )
        )

    # Parse vulnerabilities
    vulnerabilities = []
    for vuln in analysis_data.get("security_vulnerabilities", []):
        vulnerabilities.append(
            SecurityVulnerability(
                severity=vuln.get("severity", "info"),
                message=vuln.get("message", ""),
                file=vuln.get("file"),
                line=vuln.get("line"),
            )
        )

    analysis = CommitAnalysis(
        commit_sha=analysis_data.get("commit_sha", "unknown"),
        pr_number=analysis_data.get("pr_number", 0),
        files_changed=analysis_data.get("files_changed", 0),
        risk_score=analysis_data.get("risk_score", 0.0),
        predicted_test_failures=predictions,
        security_vulnerabilities=vulnerabilities,
        recommendations=analysis_data.get("recommendations", []),
        deployment_strategy=analysis_data.get("deployment_strategy", "direct"),
        analysis_url=analysis_data.get("analysis_url"),
    )

    comment_body = format_pr_comment(analysis)

    return {
        "markdown": comment_body,
        "character_count": len(comment_body),
    }
