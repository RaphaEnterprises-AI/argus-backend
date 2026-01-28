"""CI/CD Build, Pipeline, and Deployment Management API endpoints.

Provides endpoints for:
- Listing builds for a project
- Getting build details with test results
- Listing pipelines for a project (GitHub Actions)
- Getting pipeline details with stages
- Retriggering failed/completed pipelines
- Cancelling running pipelines
- Listing deployments with filtering
- Getting deployment details
- Rolling back deployments
- CI/CD statistics aggregation
- AI-powered test impact analysis (Claude + Cognee)
- AI-powered deployment risk calculation (Claude)

Integrates with GitHub Actions API for pipeline operations.
Uses Claude for semantic code analysis and Cognee for knowledge graph queries.
"""

import json
import re
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Optional

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.context import require_organization_id
from src.api.projects import verify_project_access
from src.api.teams import get_current_user, log_audit
from src.config import get_settings

# Event schemas for Kafka/Redpanda integration (AI learning loop)
from src.events.schemas import (
    EventMetadata,
    EventType,
    HealingRequestedEvent,
    TestExecutedEvent,
    TestFailedEvent,
)
from src.services.supabase_client import get_supabase_client

# Note: EventProducer and CogneeClient are imported lazily to avoid startup failures
# when Kafka/Cognee infrastructure is not available

# RAP-250: Intelligence modules imported lazily to avoid cognee dependency at startup
# from src.intelligence.precomputed import PrecomputedReader, get_precomputed_reader
# from src.intelligence.query_router import QueryRouter, QueryIntent, get_query_router
from src.services.cache import cache_discovery_pattern

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/cicd", tags=["CI/CD"])


# ============================================================================
# Pydantic Models
# ============================================================================


class Build(BaseModel):
    """CI/CD build information."""

    id: str
    project_id: str
    pipeline_id: str | None = None
    provider: str
    build_number: int
    name: str
    branch: str
    status: str  # pending, running, success, failed, cancelled, skipped
    commit_sha: str
    commit_message: str | None = None
    commit_author: str | None = None
    tests_total: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    coverage_percent: float | None = None
    artifact_urls: list[str] = Field(default_factory=list)
    logs_url: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: str


class BuildListResponse(BaseModel):
    """Response for listing builds."""

    builds: list[Build]
    total: int


# ============================================================================
# Pipeline Models
# ============================================================================


class PipelineStage(BaseModel):
    """A stage within a pipeline."""

    id: str
    name: str
    status: str  # pending, running, success, failure, cancelled, skipped
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: int | None = None
    jobs: list[dict] = Field(default_factory=list)


class Pipeline(BaseModel):
    """Pipeline details."""

    id: str
    project_id: str
    workflow_id: str | None = None
    workflow_name: str | None = None
    run_number: int | None = None
    branch: str | None = None
    commit_sha: str | None = None
    commit_message: str | None = None
    status: str  # queued, in_progress, completed, cancelled
    conclusion: str | None = None  # success, failure, cancelled, skipped, timed_out
    event: str | None = None  # push, pull_request, workflow_dispatch, etc.
    actor: str | None = None
    html_url: str | None = None
    created_at: str
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    stages: list[PipelineStage] = Field(default_factory=list)


class PipelineListResponse(BaseModel):
    """Response for listing pipelines."""

    pipelines: list[Pipeline]
    total: int


class RetriggerResponse(BaseModel):
    """Response for retrigger operation."""

    success: bool
    message: str
    pipeline: Pipeline | None = None


class CancelResponse(BaseModel):
    """Response for cancel operation."""

    success: bool
    message: str
    pipeline: Pipeline | None = None


# ============================================================================
# Deployment Models
# ============================================================================


class RiskFactor(BaseModel):
    """A risk factor contributing to deployment risk score."""

    category: str = Field(
        ...,
        description="Risk category: test_coverage, flaky_tests, change_size, time_since_deploy, error_rate, dependencies",
    )
    severity: str = Field(
        ...,
        description="Severity level: low, medium, high, critical",
    )
    description: str = Field(..., description="Human-readable description of the risk")
    score: int = Field(..., ge=0, le=100, description="Risk score contribution (0-100)")


class Deployment(BaseModel):
    """Deployment model with full details."""

    id: str
    project_id: str
    build_id: str | None = None
    environment: str = Field(
        ..., description="Environment: development, staging, production, preview"
    )
    status: str = Field(
        ..., description="Status: pending, in_progress, success, failed, rolled_back"
    )
    version: str | None = None
    commit_sha: str | None = None
    deployed_by: str | None = None
    deployment_url: str | None = None
    preview_url: str | None = None
    risk_score: int | None = Field(
        None, ge=0, le=100, description="Overall risk score (0-100)"
    )
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    health_check_status: str = Field(
        default="unknown",
        description="Health status: healthy, degraded, unhealthy, unknown",
    )
    rollback_available: bool = False
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: str


class DeploymentListResponse(BaseModel):
    """Response for listing deployments."""

    deployments: list[Deployment]
    total: int


class RollbackResponse(BaseModel):
    """Response for rollback operation."""

    success: bool
    message: str
    deployment: Deployment


# ============================================================================
# Helper Functions
# ============================================================================


def _build_from_row(row: dict) -> Build:
    """Convert a database row to a Build model."""
    return Build(
        id=row["id"],
        project_id=row["project_id"],
        pipeline_id=row.get("pipeline_id"),
        provider=row.get("provider", "unknown"),
        build_number=row.get("build_number", 0),
        name=row.get("name", ""),
        branch=row.get("branch", ""),
        status=row.get("status", "pending"),
        commit_sha=row.get("commit_sha", ""),
        commit_message=row.get("commit_message"),
        commit_author=row.get("commit_author"),
        tests_total=row.get("tests_total", 0),
        tests_passed=row.get("tests_passed", 0),
        tests_failed=row.get("tests_failed", 0),
        tests_skipped=row.get("tests_skipped", 0),
        coverage_percent=row.get("coverage_percent"),
        artifact_urls=row.get("artifact_urls") or [],
        logs_url=row.get("logs_url"),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        duration_ms=row.get("duration_ms"),
        metadata=row.get("metadata") or {},
        created_at=row["created_at"],
    )


def _deployment_from_row(row: dict) -> Deployment:
    """Convert a database row to a Deployment model."""
    # Parse risk_factors from JSON if present
    risk_factors_data = row.get("risk_factors") or []
    risk_factors = []
    for rf in risk_factors_data:
        if isinstance(rf, dict):
            risk_factors.append(
                RiskFactor(
                    category=rf.get("category", "unknown"),
                    severity=rf.get("severity", "low"),
                    description=rf.get("description", ""),
                    score=rf.get("score", 0),
                )
            )

    return Deployment(
        id=row["id"],
        project_id=row["project_id"],
        build_id=row.get("build_id"),
        environment=row.get("environment", "preview"),
        status=row.get("status", "pending"),
        version=row.get("version"),
        commit_sha=row.get("commit_sha"),
        deployed_by=row.get("deployed_by"),
        deployment_url=row.get("deployment_url"),
        preview_url=row.get("preview_url"),
        risk_score=row.get("risk_score"),
        risk_factors=risk_factors,
        health_check_status=row.get("health_check_status", "unknown"),
        rollback_available=row.get("rollback_available", False),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        duration_ms=row.get("duration_ms"),
        metadata=row.get("metadata") or {},
        created_at=row["created_at"],
    )


# ============================================================================
# Build Endpoints
# ============================================================================


@router.get("/builds", response_model=BuildListResponse)
async def list_builds(
    request: Request,
    project_id: str = Query(..., description="Project ID (required)"),
    branch: str | None = Query(None, description="Filter by branch name"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of builds to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """List builds for a project.

    Returns paginated list of builds with optional filtering by branch and status.
    Requires membership in the project's organization.
    """
    user = await get_current_user(request)
    await verify_project_access(project_id, user["user_id"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Build query for builds
    query = f"/ci_builds?project_id=eq.{project_id}&select=*&order=created_at.desc"

    if branch:
        query += f"&branch=eq.{branch}"

    if status:
        query += f"&status=eq.{status}"

    query += f"&limit={limit}&offset={offset}"

    builds_result = await supabase.request(query)

    if builds_result.get("error"):
        logger.error("Failed to fetch builds", error=builds_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch builds")

    builds_data = builds_result.get("data", [])

    # Get total count for pagination
    count_query = f"/ci_builds?project_id=eq.{project_id}&select=id"
    if branch:
        count_query += f"&branch=eq.{branch}"
    if status:
        count_query += f"&status=eq.{status}"

    count_result = await supabase.request(
        count_query,
        headers={"Prefer": "count=exact"}
    )

    # Extract count from response headers or fallback to data length
    total = len(count_result.get("data", [])) if count_result.get("data") is not None else len(builds_data)

    builds = [_build_from_row(row) for row in builds_data]

    return BuildListResponse(builds=builds, total=total)


@router.get("/builds/{build_id}", response_model=Build)
async def get_build(
    build_id: str,
    request: Request,
):
    """Get a single build by ID.

    Returns full build object with test results.
    Requires membership in the build's project organization.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # First, get the build to find its project_id
    build_result = await supabase.request(
        f"/ci_builds?id=eq.{build_id}&select=*"
    )

    if build_result.get("error"):
        logger.error("Failed to fetch build", error=build_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch build")

    if not build_result.get("data"):
        raise HTTPException(status_code=404, detail="Build not found")

    build_data = build_result["data"][0]

    # Verify user has access to the project
    await verify_project_access(
        build_data["project_id"],
        user["user_id"],
        user.get("email"),
        request=request
    )

    return _build_from_row(build_data)


# ============================================================================
# Deployment Endpoints
# ============================================================================


@router.get("/deployments", response_model=DeploymentListResponse)
async def list_deployments(
    request: Request,
    project_id: str = Query(..., description="Project ID (required)"),
    environment: str | None = Query(None, description="Filter by environment"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    List deployments for a project.

    Returns deployments with optional filtering by environment and status.
    Requires membership in the project's organization.
    """
    user = await get_current_user(request)
    await verify_project_access(
        project_id, user["user_id"], user.get("email"), request=request
    )

    supabase = get_supabase_client()

    # Build query
    query = f"/ci_deployments?project_id=eq.{project_id}&select=*"

    if environment:
        query += f"&environment=eq.{environment}"
    if status:
        query += f"&status=eq.{status}"

    query += f"&order=created_at.desc&limit={limit}&offset={offset}"

    result = await supabase.request(query)

    if result.get("error"):
        error_msg = str(result.get("error", ""))
        # Handle missing table gracefully
        if "does not exist" in error_msg or "42703" in error_msg or "42P01" in error_msg:
            logger.warning("ci_deployments table not found - returning empty list")
            return DeploymentListResponse(deployments=[], total=0)
        logger.error("Failed to fetch deployments", error=error_msg)
        raise HTTPException(status_code=500, detail="Failed to fetch deployments")

    deployments_data = result.get("data") or []

    # Get total count
    count_query = f"/ci_deployments?project_id=eq.{project_id}&select=id"
    if environment:
        count_query += f"&environment=eq.{environment}"
    if status:
        count_query += f"&status=eq.{status}"

    count_result = await supabase.request(count_query)
    total = len(count_result.get("data", []))

    deployments = [_deployment_from_row(row) for row in deployments_data]

    return DeploymentListResponse(
        deployments=deployments,
        total=total,
    )


@router.get("/deployments/{deployment_id}", response_model=Deployment)
async def get_deployment(
    deployment_id: str,
    request: Request,
):
    """
    Get a single deployment by ID.

    Returns full deployment details including risk factors and metadata.
    Requires membership in the project's organization.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Get the deployment
    result = await supabase.request(f"/ci_deployments?id=eq.{deployment_id}&select=*")

    if result.get("error"):
        error_msg = str(result.get("error", ""))
        if "does not exist" in error_msg or "42703" in error_msg or "42P01" in error_msg:
            raise HTTPException(status_code=404, detail="Deployment not found")
        logger.error("Failed to fetch deployment", error=error_msg)
        raise HTTPException(status_code=500, detail="Failed to fetch deployment")

    deployments = result.get("data", [])
    if not deployments:
        raise HTTPException(status_code=404, detail="Deployment not found")

    deployment_row = deployments[0]

    # Verify user has access to the project
    await verify_project_access(
        deployment_row["project_id"],
        user["user_id"],
        user.get("email"),
        request=request,
    )

    return _deployment_from_row(deployment_row)


@router.post("/deployments/{deployment_id}/rollback", response_model=RollbackResponse)
async def rollback_deployment(
    deployment_id: str,
    request: Request,
):
    """
    Rollback a deployment.

    Creates a new deployment with status "rolled_back" pointing to the previous version.
    The original deployment is marked as rolled back and a new deployment record is created.

    Requires membership in the project's organization.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Get the deployment to rollback
    result = await supabase.request(f"/ci_deployments?id=eq.{deployment_id}&select=*")

    if result.get("error"):
        error_msg = str(result.get("error", ""))
        if "does not exist" in error_msg or "42703" in error_msg or "42P01" in error_msg:
            raise HTTPException(status_code=404, detail="Deployment not found")
        logger.error("Failed to fetch deployment", error=error_msg)
        raise HTTPException(status_code=500, detail="Failed to fetch deployment")

    deployments = result.get("data", [])
    if not deployments:
        raise HTTPException(status_code=404, detail="Deployment not found")

    original_deployment = deployments[0]
    project_id = original_deployment["project_id"]

    # Verify user has access to the project
    await verify_project_access(
        project_id,
        user["user_id"],
        user.get("email"),
        request=request,
    )

    # Check if rollback is available
    if not original_deployment.get("rollback_available", True):
        raise HTTPException(
            status_code=400, detail="Rollback not available for this deployment"
        )

    # Find the previous successful deployment for this environment
    previous_result = await supabase.request(
        f"/ci_deployments?project_id=eq.{project_id}"
        f"&environment=eq.{original_deployment.get('environment', 'production')}"
        f"&status=eq.success"
        f"&id=neq.{deployment_id}"
        f"&order=created_at.desc&limit=1"
    )

    previous_version = None
    previous_commit = None
    if previous_result.get("data"):
        prev = previous_result["data"][0]
        previous_version = prev.get("version")
        previous_commit = prev.get("commit_sha")

    # Create new rollback deployment
    now = datetime.now(UTC)
    new_deployment_id = str(uuid.uuid4())

    rollback_data = {
        "id": new_deployment_id,
        "project_id": project_id,
        "build_id": original_deployment.get("build_id"),
        "environment": original_deployment.get("environment", "production"),
        "status": "rolled_back",
        "version": previous_version or original_deployment.get("version"),
        "commit_sha": previous_commit or original_deployment.get("commit_sha"),
        "deployed_by": user.get("email") or user.get("user_id"),
        "deployment_url": original_deployment.get("deployment_url"),
        "preview_url": original_deployment.get("preview_url"),
        "risk_score": 0,  # Rollbacks have low risk
        "risk_factors": [],
        "health_check_status": "unknown",
        "rollback_available": False,  # Can't rollback a rollback
        "started_at": now.isoformat(),
        "completed_at": now.isoformat(),
        "duration_ms": 0,
        "metadata": {
            "rollback_from": deployment_id,
            "original_version": original_deployment.get("version"),
            "original_commit": original_deployment.get("commit_sha"),
            "rolled_back_to_version": previous_version,
            "rolled_back_to_commit": previous_commit,
            "triggered_by": user.get("email") or user.get("user_id"),
        },
        "created_at": now.isoformat(),
    }

    insert_result = await supabase.insert("ci_deployments", rollback_data)

    if insert_result.get("error"):
        logger.error(
            "Failed to create rollback deployment", error=insert_result.get("error")
        )
        raise HTTPException(status_code=500, detail="Failed to create rollback deployment")

    # Update original deployment to mark it as rolled back
    await supabase.update(
        "ci_deployments",
        {"id": f"eq.{deployment_id}"},
        {
            "rollback_available": False,
            "metadata": {
                **(original_deployment.get("metadata") or {}),
                "rolled_back_at": now.isoformat(),
                "rolled_back_by": user.get("email") or user.get("user_id"),
                "rollback_deployment_id": new_deployment_id,
            },
        },
    )

    # Audit log
    await log_audit(
        organization_id=None,  # Will be looked up from project
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="deployment.rollback",
        resource_type="deployment",
        resource_id=deployment_id,
        description=f"Rolled back deployment {deployment_id} in {original_deployment.get('environment', 'production')}",
        metadata={
            "original_deployment_id": deployment_id,
            "new_deployment_id": new_deployment_id,
            "environment": original_deployment.get("environment"),
            "rolled_back_to_version": previous_version,
        },
        request=request,
    )

    logger.info(
        "Deployment rolled back",
        original_deployment_id=deployment_id,
        new_deployment_id=new_deployment_id,
        project_id=project_id,
        environment=original_deployment.get("environment"),
    )

    new_deployment = _deployment_from_row(rollback_data)

    return RollbackResponse(
        success=True,
        message=f"Successfully rolled back deployment in {original_deployment.get('environment', 'production')}",
        deployment=new_deployment,
    )


# ============================================================================
# Pipeline Helper Functions
# ============================================================================


async def get_project_github_config(project_id: str) -> dict | None:
    """Get GitHub configuration for a project.

    Returns dict with owner, repo, and token if configured.
    """
    supabase = get_supabase_client()

    # Get project with its settings
    result = await supabase.request(
        f"/projects?id=eq.{project_id}&select=id,organization_id,repository_url,settings"
    )

    if result.get("error") or not result.get("data"):
        return None

    project = result["data"][0]
    settings = project.get("settings") or {}

    # Try to get GitHub token from project settings or integrations
    github_token = settings.get("github_token")

    if not github_token:
        # Try to get from organization integrations
        org_id = project.get("organization_id")
        if org_id:
            integration_result = await supabase.request(
                f"/integrations?organization_id=eq.{org_id}&provider=eq.github&status=eq.active&select=credentials"
            )
            if integration_result.get("data"):
                credentials = integration_result["data"][0].get("credentials") or {}
                github_token = credentials.get("access_token")

    # Parse repository URL to get owner/repo
    repo_url = project.get("repository_url")
    if not repo_url:
        return None

    # Parse GitHub URL: https://github.com/owner/repo or git@github.com:owner/repo.git
    owner = None
    repo = None

    if "github.com" in repo_url:
        # Remove .git suffix if present
        repo_url = repo_url.rstrip("/").removesuffix(".git")

        if repo_url.startswith("https://github.com/"):
            parts = repo_url.replace("https://github.com/", "").split("/")
            if len(parts) >= 2:
                owner, repo = parts[0], parts[1]
        elif repo_url.startswith("git@github.com:"):
            parts = repo_url.replace("git@github.com:", "").split("/")
            if len(parts) >= 2:
                owner, repo = parts[0], parts[1]

    if not owner or not repo:
        return None

    return {
        "owner": owner,
        "repo": repo,
        "token": github_token,
        "project_id": project_id,
        "organization_id": project.get("organization_id"),
    }


async def fetch_github_workflow_runs(
    owner: str,
    repo: str,
    token: str | None,
    branch: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Fetch workflow runs from GitHub Actions API."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    params = {"per_page": min(limit, 100)}
    if branch:
        params["branch"] = branch
    if status:
        # GitHub API uses different status values
        params["status"] = status

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/actions/runs",
            headers=headers,
            params=params,
            timeout=30.0,
        )

        if response.status_code == 401:
            logger.error("GitHub API authentication failed", owner=owner, repo=repo)
            raise HTTPException(status_code=401, detail="GitHub authentication failed")

        if response.status_code == 404:
            logger.error("GitHub repository not found", owner=owner, repo=repo)
            raise HTTPException(status_code=404, detail="GitHub repository not found")

        if not response.is_success:
            logger.error(
                "GitHub API error",
                status=response.status_code,
                body=response.text[:500],
            )
            raise HTTPException(
                status_code=502,
                detail=f"GitHub API error: {response.status_code}",
            )

        data = response.json()
        return data.get("workflow_runs", [])


async def fetch_github_workflow_run(
    owner: str,
    repo: str,
    run_id: int,
    token: str | None,
) -> dict | None:
    """Fetch a single workflow run with jobs from GitHub Actions API."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient() as client:
        # Get workflow run
        run_response = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}",
            headers=headers,
            timeout=30.0,
        )

        if not run_response.is_success:
            return None

        run_data = run_response.json()

        # Get jobs for the run
        jobs_response = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
            headers=headers,
            timeout=30.0,
        )

        jobs_data = []
        if jobs_response.is_success:
            jobs_data = jobs_response.json().get("jobs", [])

        run_data["jobs"] = jobs_data
        return run_data


def github_run_to_pipeline(run: dict, project_id: str) -> Pipeline:
    """Convert GitHub workflow run to Pipeline model."""
    # Group jobs by workflow job name to create stages
    jobs = run.get("jobs", [])
    stages = []

    for job in jobs:
        stage = PipelineStage(
            id=str(job.get("id", "")),
            name=job.get("name", "Unknown"),
            status=map_github_status(job.get("status"), job.get("conclusion")),
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at"),
            duration_seconds=calculate_pipeline_duration(
                job.get("started_at"),
                job.get("completed_at"),
            ),
            jobs=[
                {
                    "id": str(job.get("id")),
                    "name": job.get("name"),
                    "status": job.get("status"),
                    "conclusion": job.get("conclusion"),
                    "steps": job.get("steps", []),
                }
            ],
        )
        stages.append(stage)

    return Pipeline(
        id=str(run.get("id", "")),
        project_id=project_id,
        workflow_id=str(run.get("workflow_id", "")),
        workflow_name=run.get("name"),
        run_number=run.get("run_number"),
        branch=run.get("head_branch"),
        commit_sha=run.get("head_sha"),
        commit_message=run.get("head_commit", {}).get("message")
        if run.get("head_commit")
        else None,
        status=run.get("status", "unknown"),
        conclusion=run.get("conclusion"),
        event=run.get("event"),
        actor=run.get("actor", {}).get("login") if run.get("actor") else None,
        html_url=run.get("html_url"),
        created_at=run.get("created_at", datetime.now(UTC).isoformat()),
        updated_at=run.get("updated_at"),
        started_at=run.get("run_started_at"),
        completed_at=run.get("updated_at") if run.get("status") == "completed" else None,
        stages=stages,
    )


def map_github_status(status: str | None, conclusion: str | None) -> str:
    """Map GitHub status/conclusion to unified status."""
    if status == "queued":
        return "pending"
    if status == "in_progress":
        return "running"
    if status == "completed":
        if conclusion == "success":
            return "success"
        if conclusion == "failure":
            return "failure"
        if conclusion == "cancelled":
            return "cancelled"
        if conclusion == "skipped":
            return "skipped"
        return conclusion or "unknown"
    return status or "unknown"


def calculate_pipeline_duration(started_at: str | None, completed_at: str | None) -> int | None:
    """Calculate duration in seconds between two ISO timestamps."""
    if not started_at or not completed_at:
        return None

    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        return int((end - start).total_seconds())
    except (ValueError, TypeError):
        return None


# ============================================================================
# Pipeline Endpoints
# ============================================================================


@router.get("/pipelines", response_model=PipelineListResponse)
async def list_pipelines(
    request: Request,
    project_id: str = Query(..., description="Project ID (required)"),
    branch: str | None = Query(None, description="Filter by branch name"),
    status: str | None = Query(None, description="Filter by status (queued, in_progress, completed)"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of pipelines to return"),
):
    """List pipelines for a project.

    Fetches workflow runs from GitHub Actions for the project's repository.

    Query Parameters:
    - project_id: Required. The project to list pipelines for.
    - branch: Optional. Filter by branch name.
    - status: Optional. Filter by status (queued, in_progress, completed).
    - limit: Optional. Maximum results (default 50, max 100).

    Returns:
    - pipelines: List of pipeline objects
    - total: Total count of returned pipelines
    """
    # Get GitHub config for project
    github_config = await get_project_github_config(project_id)

    if not github_config:
        logger.warning("GitHub not configured for project", project_id=project_id)
        # Return empty list if GitHub not configured
        return PipelineListResponse(pipelines=[], total=0)

    # Fetch workflow runs from GitHub
    try:
        runs = await fetch_github_workflow_runs(
            owner=github_config["owner"],
            repo=github_config["repo"],
            token=github_config.get("token"),
            branch=branch,
            status=status,
            limit=limit,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch GitHub workflow runs", error=str(e))
        raise HTTPException(status_code=502, detail="Failed to fetch pipelines from GitHub")

    # Convert to Pipeline models
    pipelines = [github_run_to_pipeline(run, project_id) for run in runs]

    logger.info(
        "Listed pipelines",
        project_id=project_id,
        count=len(pipelines),
        branch=branch,
        status=status,
    )

    return PipelineListResponse(pipelines=pipelines, total=len(pipelines))


@router.get("/pipelines/{pipeline_id}", response_model=Pipeline)
async def get_pipeline(
    pipeline_id: str,
    request: Request,
    project_id: str = Query(..., description="Project ID"),
):
    """Get a single pipeline with full details including stages.

    Path Parameters:
    - pipeline_id: The GitHub workflow run ID.

    Query Parameters:
    - project_id: Required. The project the pipeline belongs to.

    Returns:
    - Full pipeline object with stages and jobs.
    """
    # Get GitHub config for project
    github_config = await get_project_github_config(project_id)

    if not github_config:
        raise HTTPException(
            status_code=400,
            detail="GitHub not configured for this project",
        )

    # Fetch workflow run with jobs from GitHub
    try:
        run_id = int(pipeline_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline ID format")

    try:
        run = await fetch_github_workflow_run(
            owner=github_config["owner"],
            repo=github_config["repo"],
            run_id=run_id,
            token=github_config.get("token"),
        )
    except Exception as e:
        logger.exception("Failed to fetch GitHub workflow run", error=str(e))
        raise HTTPException(status_code=502, detail="Failed to fetch pipeline from GitHub")

    if not run:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    pipeline = github_run_to_pipeline(run, project_id)

    logger.info("Retrieved pipeline", pipeline_id=pipeline_id, project_id=project_id)

    return pipeline


@router.post("/pipelines/{pipeline_id}/retrigger", response_model=RetriggerResponse)
async def retrigger_pipeline(
    pipeline_id: str,
    request: Request,
    project_id: str = Query(..., description="Project ID"),
):
    """Retrigger a pipeline by re-running the GitHub Actions workflow.

    Path Parameters:
    - pipeline_id: The GitHub workflow run ID to retrigger.

    Query Parameters:
    - project_id: Required. The project the pipeline belongs to.

    Returns:
    - success: Whether the retrigger was initiated
    - message: Status message
    - pipeline: The new pipeline object if successful
    """
    # Get GitHub config for project
    github_config = await get_project_github_config(project_id)

    if not github_config:
        raise HTTPException(
            status_code=400,
            detail="GitHub not configured for this project",
        )

    if not github_config.get("token"):
        raise HTTPException(
            status_code=400,
            detail="GitHub token required for retrigger operation",
        )

    try:
        run_id = int(pipeline_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline ID format")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_config['token']}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient() as client:
        # Re-run the workflow
        response = await client.post(
            f"https://api.github.com/repos/{github_config['owner']}/{github_config['repo']}/actions/runs/{run_id}/rerun",
            headers=headers,
            timeout=30.0,
        )

        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="GitHub authentication failed")

        if response.status_code == 403:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions to retrigger workflow",
            )

        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        if not response.is_success:
            logger.error(
                "Failed to retrigger workflow",
                status=response.status_code,
                body=response.text[:500],
            )
            raise HTTPException(
                status_code=502,
                detail=f"Failed to retrigger pipeline: {response.status_code}",
            )

    logger.info(
        "Pipeline retriggered",
        pipeline_id=pipeline_id,
        project_id=project_id,
        owner=github_config["owner"],
        repo=github_config["repo"],
    )

    # Fetch the updated run
    try:
        run = await fetch_github_workflow_run(
            owner=github_config["owner"],
            repo=github_config["repo"],
            run_id=run_id,
            token=github_config.get("token"),
        )
        pipeline = github_run_to_pipeline(run, project_id) if run else None
    except Exception:
        pipeline = None

    return RetriggerResponse(
        success=True,
        message="Pipeline retrigger initiated successfully",
        pipeline=pipeline,
    )


@router.post("/pipelines/{pipeline_id}/cancel", response_model=CancelResponse)
async def cancel_pipeline(
    pipeline_id: str,
    request: Request,
    project_id: str = Query(..., description="Project ID"),
):
    """Cancel a running pipeline.

    Path Parameters:
    - pipeline_id: The GitHub workflow run ID to cancel.

    Query Parameters:
    - project_id: Required. The project the pipeline belongs to.

    Returns:
    - success: Whether the cancellation was initiated
    - message: Status message
    - pipeline: The updated pipeline object if successful
    """
    # Get GitHub config for project
    github_config = await get_project_github_config(project_id)

    if not github_config:
        raise HTTPException(
            status_code=400,
            detail="GitHub not configured for this project",
        )

    if not github_config.get("token"):
        raise HTTPException(
            status_code=400,
            detail="GitHub token required for cancel operation",
        )

    try:
        run_id = int(pipeline_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline ID format")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_config['token']}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient() as client:
        # Cancel the workflow run
        response = await client.post(
            f"https://api.github.com/repos/{github_config['owner']}/{github_config['repo']}/actions/runs/{run_id}/cancel",
            headers=headers,
            timeout=30.0,
        )

        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="GitHub authentication failed")

        if response.status_code == 403:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions to cancel workflow",
            )

        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        if response.status_code == 409:
            # Workflow is not running (already completed or cancelled)
            return CancelResponse(
                success=False,
                message="Pipeline is not running or has already completed",
                pipeline=None,
            )

        if not response.is_success:
            logger.error(
                "Failed to cancel workflow",
                status=response.status_code,
                body=response.text[:500],
            )
            raise HTTPException(
                status_code=502,
                detail=f"Failed to cancel pipeline: {response.status_code}",
            )

    logger.info(
        "Pipeline cancelled",
        pipeline_id=pipeline_id,
        project_id=project_id,
        owner=github_config["owner"],
        repo=github_config["repo"],
    )

    # Fetch the updated run
    try:
        run = await fetch_github_workflow_run(
            owner=github_config["owner"],
            repo=github_config["repo"],
            run_id=run_id,
            token=github_config.get("token"),
        )
        pipeline = github_run_to_pipeline(run, project_id) if run else None
    except Exception:
        pipeline = None

    return CancelResponse(
        success=True,
        message="Pipeline cancellation initiated successfully",
        pipeline=pipeline,
    )


# ============================================================================
# Stats & Test Impact Models (RAP-198)
# ============================================================================


class CICDStats(BaseModel):
    """Aggregate CI/CD statistics for a project."""

    total_pipelines: int = Field(default=0, description="Total CI pipeline runs")
    pipelines_last_24h: int = Field(default=0, description="Pipeline runs in last 24 hours")
    success_rate: float = Field(default=0.0, description="Overall pipeline success rate (0-100)")
    avg_pipeline_duration_ms: int = Field(default=0, description="Average pipeline duration in milliseconds")
    total_builds: int = Field(default=0, description="Total build count")
    builds_last_24h: int = Field(default=0, description="Builds in last 24 hours")
    build_success_rate: float = Field(default=0.0, description="Build success rate (0-100)")
    avg_build_duration_ms: int = Field(default=0, description="Average build duration in milliseconds")
    total_deployments: int = Field(default=0, description="Total deployment count")
    deployments_last_24h: int = Field(default=0, description="Deployments in last 24 hours")
    deployment_success_rate: float = Field(default=0.0, description="Deployment success rate (0-100)")
    avg_deployment_duration_ms: int = Field(default=0, description="Average deployment duration in milliseconds")
    current_risk_score: int = Field(default=50, description="Current deployment risk score (0-100)")
    tests_impacted_by_recent_changes: int = Field(default=0, description="Tests impacted by recent code changes")


class ChangedFile(BaseModel):
    """A file that was changed in a commit."""

    path: str = Field(..., description="File path")
    change_type: str = Field(
        default="modified",
        description="Type of change: added, modified, deleted, renamed"
    )
    additions: int = Field(default=0, description="Lines added")
    deletions: int = Field(default=0, description="Lines deleted")
    impact_score: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Impact score based on file type and location"
    )


class ImpactedTest(BaseModel):
    """A test impacted by file changes."""

    test_id: str = Field(..., description="Test ID")
    test_name: str = Field(..., description="Test name")
    impact_reason: str = Field(..., description="Reason for impact (e.g., 'direct import', 'path similarity')")
    confidence: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Confidence score of impact prediction"
    )
    priority: Literal["critical", "high", "medium", "low"] = Field(
        default="medium",
        description="Test priority"
    )


class TestImpactAnalysis(BaseModel):
    """Result of test impact analysis for a commit."""

    id: str = Field(..., description="Analysis ID")
    project_id: str = Field(..., description="Project ID")
    commit_sha: str = Field(..., description="Git commit SHA")
    branch: str = Field(default="main", description="Git branch")
    base_sha: str | None = Field(None, description="Base commit SHA for comparison")
    changed_files: list[ChangedFile] = Field(default_factory=list, description="List of changed files")
    impacted_tests: list[ImpactedTest] = Field(default_factory=list, description="List of impacted tests")
    total_files_changed: int = Field(default=0, description="Total number of files changed")
    total_tests_impacted: int = Field(default=0, description="Total number of tests impacted")
    recommended_tests: list[str] = Field(default_factory=list, description="Test IDs recommended to run")
    skip_candidates: list[str] = Field(default_factory=list, description="Test IDs safe to skip")
    confidence_score: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Overall confidence of the analysis"
    )
    analysis_time_ms: int = Field(default=0, description="Time taken for analysis in milliseconds")
    created_at: str = Field(..., description="ISO timestamp of when analysis was created")


class TestImpactAnalyzeRequest(BaseModel):
    """Request body for running test impact analysis."""

    project_id: str = Field(..., description="Project ID")
    commit_sha: str = Field(..., description="Git commit SHA to analyze")
    base_sha: str | None = Field(None, description="Base commit SHA for comparison (defaults to parent)")
    branch: str = Field(default="main", description="Git branch name")
    changed_files: list[ChangedFile] | None = Field(
        None,
        description="Pre-computed list of changed files (if not provided, will be inferred)"
    )


class DeploymentRiskResponse(BaseModel):
    """Deployment risk assessment response."""

    project_id: str = Field(..., description="Project ID")
    commit_sha: str | None = Field(None, description="Commit SHA being assessed")
    risk_score: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Overall risk score (0-100, higher = riskier)"
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Risk level classification"
    )
    factors: dict = Field(default_factory=dict, description="Risk factor breakdown")
    recommendations: list[str] = Field(default_factory=list, description="Risk mitigation recommendations")
    tests_to_run: int = Field(default=0, description="Number of tests recommended to run")
    estimated_test_time_ms: int = Field(default=0, description="Estimated time to run recommended tests")
    calculated_at: str = Field(..., description="ISO timestamp of calculation")


# ============================================================================
# AI-Powered Analysis Helpers
# ============================================================================


def _calculate_risk_level(score: int) -> Literal["low", "medium", "high", "critical"]:
    """Convert risk score to risk level (simple threshold, not AI)."""
    if score >= 80:
        return "critical"
    elif score >= 60:
        return "high"
    elif score >= 40:
        return "medium"
    else:
        return "low"


async def _run_ai_test_impact_analysis(
    changed_files: list[dict],
    all_tests: list[dict],
    commit_sha: str,
    branch: str,
) -> tuple[list, list, list, float]:
    """Run AI-powered test impact analysis using Claude directly.

    Returns:
        Tuple of (impacted_tests, recommended_tests, skip_candidates, confidence)
    """
    try:
        from src.config import get_settings
        settings = get_settings()

        logger.info("Starting AI test impact analysis", commit_sha=commit_sha, branch=branch)

        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")

        import anthropic
        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )

        # Build file changes summary
        files_summary = "\n".join([
            f"- {cf.get('path', 'unknown')}: +{cf.get('additions', 0)}/-{cf.get('deletions', 0)} ({cf.get('change_type', 'modified')})"
            for cf in changed_files
        ])

        # Build tests summary (limit to 50 for context)
        tests_summary = "\n".join([
            f"- {t.get('id', '')}: {t.get('name', '')} (priority: {t.get('priority', 'medium')})"
            for t in all_tests[:50]
        ])

        prompt = f"""Analyze which tests are likely affected by these code changes.

CHANGED FILES:
{files_summary}

AVAILABLE TESTS:
{tests_summary}

Based on the file paths and test names, determine:
1. Which tests are LIKELY to be affected by these changes
2. Which tests can SAFELY be skipped
3. Overall confidence in your assessment

Consider:
- Tests with names/paths matching changed files are likely affected
- Changes to auth/security files affect authentication tests
- Changes to API files affect API tests
- Changes to utils/helpers may affect many tests

Respond in JSON format:
{{
    "affected_test_ids": ["id1", "id2", ...],
    "skip_test_ids": ["id3", "id4", ...],
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}"""

        logger.info("Calling Claude API for test impact analysis")

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse the AI response
        text = response.content[0].text
        json_start = text.find("{")
        json_end = text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            result = json.loads(text[json_start:json_end])

            affected_ids = set(result.get("affected_test_ids", []))
            skip_ids = set(result.get("skip_test_ids", []))
            confidence = float(result.get("confidence", 0.7))

            # Build impacted tests list
            impacted_tests = []
            for test in all_tests:
                test_id = test.get("id", "")
                if test_id in affected_ids:
                    impacted_tests.append({
                        "test_id": test_id,
                        "test_name": test.get("name", ""),
                        "impact_reason": "AI semantic analysis",
                        "confidence": confidence,
                        "priority": test.get("priority", "medium"),
                    })

            # Recommended = affected tests
            recommended_tests = list(affected_ids)

            # Skip candidates = skip_ids + tests not in affected_ids
            all_test_ids = {t.get("id") for t in all_tests}
            skip_candidates = list(skip_ids | (all_test_ids - affected_ids))

            logger.info(
                "AI test impact analysis completed",
                affected_count=len(impacted_tests),
                skip_count=len(skip_candidates),
                confidence=confidence,
                reasoning=result.get("reasoning", ""),
            )

            return impacted_tests, recommended_tests, skip_candidates, confidence

        raise ValueError("Could not parse AI response for test impact")

    except Exception as e:
        logger.warning(
            "AI test impact analysis failed, will use fallback",
            error=str(e),
        )
        raise


# =============================================================================
# RAP-250: Precomputed Test Impact Analysis
# =============================================================================


def _lookup_tests_in_matrix(
    matrix: dict[str, list[str]],
    changed_files: list[str],
) -> list[str]:
    """Look up affected tests in the precomputed impact matrix.

    Supports both exact path matches and partial path matches to handle
    relative vs absolute path differences.

    Args:
        matrix: File path -> test IDs mapping from precomputed data
        changed_files: List of changed file paths

    Returns:
        Deduplicated list of affected test IDs
    """
    affected = set()

    for file_path in changed_files:
        # Normalize path separators
        normalized_path = file_path.replace("\\", "/")

        # Try exact match first
        if normalized_path in matrix:
            affected.update(matrix[normalized_path])
            continue

        # Try partial path matches (handle relative vs absolute paths)
        file_name = normalized_path.split("/")[-1]
        for matrix_path, tests in matrix.items():
            # Check if one path ends with the other
            if matrix_path.endswith(normalized_path) or normalized_path.endswith(matrix_path):
                affected.update(tests)
                break
            # Match by filename if paths don't align
            matrix_name = matrix_path.split("/")[-1]
            if file_name == matrix_name:
                affected.update(tests)

    return list(affected)


@cache_discovery_pattern(ttl_seconds=300, key_prefix="test_impact")
async def get_affected_tests(
    org_id: str,
    project_id: str,
    changed_files: list[str],
) -> tuple[list[str], str]:
    """Get affected tests using precomputed matrix with LLM fallback.

    RAP-250: This function provides ~20-50ms response times for projects
    with precomputed data, falling back to LLM calls (~2-3s) only when
    precomputed data is unavailable.

    Args:
        org_id: Organization ID for tenant isolation
        project_id: Project ID
        changed_files: List of changed file paths

    Returns:
        Tuple of (affected_test_ids, source) where source indicates
        whether data came from 'precomputed', 'query_router', or 'llm'
    """
    # RAP-250: Lazy imports to avoid cognee dependency at startup
    from src.intelligence.precomputed import get_precomputed_reader
    from src.intelligence.query_router import get_query_router

    start_time = time.time()

    # Step 1: Check precomputed matrix (fastest path, ~20-50ms)
    precomputed = get_precomputed_reader()
    matrix = await precomputed.get_test_impact_matrix(org_id, project_id)

    if matrix is not None:
        affected_tests = _lookup_tests_in_matrix(matrix, changed_files)
        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Precomputed test impact lookup completed",
            org_id=org_id,
            project_id=project_id,
            files_checked=len(changed_files),
            tests_found=len(affected_tests),
            latency_ms=latency_ms,
            source="precomputed",
        )

        return affected_tests, "precomputed"

    # Step 2: Try QueryRouter for intelligent routing (medium path, ~100-500ms)
    try:
        router = get_query_router()
        result = await router.route(
            query=f"test impact for files: {', '.join(changed_files[:10])}",
            org_id=org_id,
            project_id=project_id,
        )

        if result.data and result.source != "llm":
            # Extract test IDs from router result
            data = result.data
            if isinstance(data, dict) and "results" in data:
                affected_tests = []
                for r in data["results"]:
                    if isinstance(r, dict) and "test_id" in r:
                        affected_tests.append(r["test_id"])
                    elif isinstance(r, str):
                        affected_tests.append(r)

                latency_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    "Query router test impact lookup completed",
                    org_id=org_id,
                    project_id=project_id,
                    tests_found=len(affected_tests),
                    latency_ms=latency_ms,
                    source=result.source,
                )
                return affected_tests, f"query_router:{result.source}"

    except Exception as router_error:
        logger.debug(
            "Query router lookup failed, falling back to LLM",
            error=str(router_error),
        )

    # Step 3: No precomputed data available, return empty to trigger LLM fallback
    logger.info(
        "No precomputed test impact data available",
        org_id=org_id,
        project_id=project_id,
        files_checked=len(changed_files),
    )

    return [], "none"


async def _run_ai_deployment_risk_analysis(
    project_id: str,
    commit_sha: str | None,
    ci_failure_rate: float | None,
    deploy_failure_rate: float | None,
    tests_impacted: int,
    tests_total: int,
    quality_score: int | None,
) -> tuple[int, str, list[str]]:
    """Run AI-powered deployment risk analysis using Claude.

    Returns:
        Tuple of (risk_score, risk_level, recommendations)
    """
    try:
        from src.config import get_settings
        settings = get_settings()

        logger.info("Starting AI deployment risk analysis", project_id=project_id, commit_sha=commit_sha)

        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")

        import anthropic
        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )

        logger.info("Calling Claude API for risk analysis")

        # Build context for AI analysis
        context = f"""Analyze the deployment risk for this project and provide a risk assessment.

Project Context:
- Commit SHA: {commit_sha or 'Not specified'}
- CI Failure Rate (last 7 days): {f'{ci_failure_rate:.1%}' if ci_failure_rate is not None else 'No data'}
- Deployment Failure Rate (last 7 days): {f'{deploy_failure_rate:.1%}' if deploy_failure_rate is not None else 'No data'}
- Tests Impacted: {tests_impacted} out of {tests_total} total tests
- Code Quality Score: {quality_score if quality_score is not None else 'No data'}/100

Based on this data, provide:
1. A risk score from 0-100 (0 = safe to deploy, 100 = extremely risky)
2. A risk level: "low", "medium", "high", or "critical"
3. 3-5 specific, actionable recommendations

Consider:
- Historical failure patterns suggest future issues
- Low test coverage increases uncertainty
- Recent failures indicate instability
- Quality metrics reflect code health

Respond in JSON format:
{{
    "risk_score": <number 0-100>,
    "risk_level": "<low|medium|high|critical>",
    "recommendations": ["recommendation 1", "recommendation 2", ...],
    "reasoning": "Brief explanation of the assessment"
}}"""

        # Use actual Anthropic model ID directly for reliability
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",  # Fast model for quick risk assessment
            max_tokens=1024,
            temperature=0.3,
            messages=[{"role": "user", "content": context}]
        )

        # Parse the AI response
        text = response.content[0].text

        # Extract JSON from response
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(text[json_start:json_end])

            risk_score = int(result.get("risk_score", 50))
            risk_level = result.get("risk_level", "medium")
            recommendations = result.get("recommendations", [])

            # Validate risk_level
            if risk_level not in ("low", "medium", "high", "critical"):
                risk_level = _calculate_risk_level(risk_score)

            logger.info(
                "AI deployment risk analysis completed",
                risk_score=risk_score,
                risk_level=risk_level,
                reasoning=result.get("reasoning", ""),
            )

            return risk_score, risk_level, recommendations

        raise ValueError("Could not parse AI response")

    except Exception as e:
        logger.warning(
            "AI deployment risk analysis failed, using fallback",
            error=str(e),
        )
        raise


# ============================================================================
# AI Learning Loop Integration (Event Emission + Cognee Storage)
# ============================================================================


async def _emit_test_event(
    event_type: str,
    org_id: str,
    project_id: str,
    test_id: str,
    test_name: str,
    status: str,
    error_message: str | None = None,
    error_type: str | None = None,
    duration_ms: int | None = None,
    metadata: dict | None = None,
) -> bool:
    """Emit test execution/failure event to Kafka for AI learning loop.

    This enables:
    - Cognee worker to process failures and build knowledge graph
    - Self-healing pipeline to be triggered
    - Cross-project pattern learning
    """
    try:
        from src.events.producer import EventProducer
        from src.events.topics import get_topic_for_event
        settings = get_settings()

        if not settings.redpanda_brokers:
            logger.debug("Redpanda not configured, skipping event emission")
            return False

        event_metadata = EventMetadata(
            request_id=str(uuid.uuid4()),
            source="cicd_pipeline",
            triggered_by="api",
        )

        if event_type == "test_failed":
            event = TestFailedEvent(
                org_id=org_id,
                project_id=project_id,
                metadata=event_metadata,
                test_id=test_id,
                test_name=test_name,
                error_message=error_message or "Unknown error",
                error_type=error_type or "unknown",
                failed_step_index=0,
                failed_selector=metadata.get("failed_selector") if metadata else None,
                page_url=metadata.get("page_url") if metadata else None,
                screenshot_url=metadata.get("screenshot_url") if metadata else None,
                dom_snapshot=metadata.get("dom_snapshot") if metadata else None,
            )
        else:
            event = TestExecutedEvent(
                org_id=org_id,
                project_id=project_id,
                metadata=event_metadata,
                test_id=test_id,
                run_id=str(uuid.uuid4()),
                status=status,
                duration_ms=duration_ms or 0,
                steps_executed=metadata.get("steps_executed", 0) if metadata else 0,
                assertions_passed=metadata.get("assertions_passed", 0) if metadata else 0,
                assertions_failed=metadata.get("assertions_failed", 0) if metadata else 0,
            )

        topic = get_topic_for_event(event.event_type)

        async with EventProducer.create(
            bootstrap_servers=settings.redpanda_brokers,
            sasl_username=settings.redpanda_sasl_username,
            sasl_password=settings.redpanda_sasl_password.get_secret_value() if settings.redpanda_sasl_password else None,
        ) as producer:
            await producer.send(topic, event)

        logger.info(
            "Test event emitted to Kafka",
            event_type=event_type,
            test_id=test_id,
            topic=topic,
        )
        return True

    except Exception as e:
        logger.warning(
            "Failed to emit test event (non-blocking)",
            error=str(e),
            event_type=event_type,
            test_id=test_id,
        )
        return False


async def _emit_healing_request(
    org_id: str,
    project_id: str,
    test_id: str,
    test_name: str,
    error_message: str,
    error_type: str,
    screenshot_urls: list[str] | None = None,
) -> bool:
    """Emit healing request to trigger self-healing pipeline.

    This starts the AI learning loop:
    1. Event goes to Kafka (argus.healing.requested)
    2. Cognee worker receives it
    3. Searches knowledge graph for similar failures
    4. Produces healing suggestions
    5. Results stored for future learning
    """
    try:
        from src.events.producer import EventProducer
        from src.events.topics import TOPIC_HEALING_REQUESTED
        settings = get_settings()

        if not settings.redpanda_brokers:
            logger.debug("Redpanda not configured, skipping healing request")
            return False

        event = HealingRequestedEvent(
            org_id=org_id,
            project_id=project_id,
            metadata=EventMetadata(
                request_id=str(uuid.uuid4()),
                source="cicd_pipeline",
                triggered_by="test_failure",
            ),
            test_id=test_id,
            run_id=str(uuid.uuid4()),
            failure_type=error_type,
            error_message=error_message,
            failed_step_index=0,
            screenshot_urls=screenshot_urls or [],
            previous_healing_attempts=0,
        )

        async with EventProducer.create(
            bootstrap_servers=settings.redpanda_brokers,
            sasl_username=settings.redpanda_sasl_username,
            sasl_password=settings.redpanda_sasl_password.get_secret_value() if settings.redpanda_sasl_password else None,
        ) as producer:
            await producer.send(TOPIC_HEALING_REQUESTED, event)

        logger.info(
            "Healing request emitted to Kafka",
            test_id=test_id,
            error_type=error_type,
        )
        return True

    except Exception as e:
        logger.warning(
            "Failed to emit healing request (non-blocking)",
            error=str(e),
            test_id=test_id,
        )
        return False


async def _store_analysis_in_cognee(
    org_id: str,
    project_id: str,
    analysis_type: str,
    data: dict,
) -> bool:
    """Store analysis results in Cognee knowledge graph for learning.

    This enables:
    - Semantic search for similar code change patterns
    - Cross-project learning
    - Historical trend analysis
    """
    try:
        from src.knowledge.cognee_client import get_cognee_client

        cognee = get_cognee_client()
        if not cognee:
            logger.debug("Cognee not configured, skipping knowledge storage")
            return False

        # Create unique key for this analysis
        analysis_key = f"{analysis_type}_{data.get('commit_sha', 'unknown')}_{uuid.uuid4().hex[:8]}"

        # Store with multi-tenant namespace
        namespace = f"org_{org_id}_project_{project_id}_{analysis_type}"

        await cognee.put(
            key=analysis_key,
            value={
                "type": analysis_type,
                "timestamp": datetime.now(UTC).isoformat(),
                **data,
            },
            namespace=namespace,
            embeddings=True,  # Enable semantic search
        )

        logger.info(
            "Analysis stored in Cognee knowledge graph",
            analysis_type=analysis_type,
            namespace=namespace,
            key=analysis_key,
        )
        return True

    except Exception as e:
        logger.warning(
            "Failed to store in Cognee (non-blocking)",
            error=str(e),
            analysis_type=analysis_type,
        )
        return False


# ============================================================================
# Stats Endpoint (RAP-198)
# ============================================================================


@router.get("/stats", response_model=CICDStats)
async def get_cicd_stats(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
):
    """Get aggregated CI/CD statistics for a project.

    Aggregates data from:
    - ci_events table (pipelines/builds)
    - deployment_events table (deployments)
    - quality_scores table (risk score)
    - test_impact_graph table (impacted tests)
    """
    user = await get_current_user(request)
    await verify_project_access(project_id, user["user_id"], user.get("email"), request=request)

    supabase = get_supabase_client()

    now = datetime.now(UTC)
    yesterday = now - timedelta(hours=24)
    yesterday_iso = yesterday.isoformat()

    try:
        # Get CI events (pipelines)
        ci_result = await supabase.request(
            f"/ci_events?project_id=eq.{project_id}&select=id,status,duration_seconds,created_at"
        )
        ci_events = ci_result.get("data") or []

        # Calculate pipeline stats
        total_pipelines = len(ci_events)
        pipelines_last_24h = sum(
            1 for e in ci_events
            if e.get("created_at", "") > yesterday_iso
        )
        successful_pipelines = sum(1 for e in ci_events if e.get("status") == "success")
        success_rate = round(
            (successful_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0.0, 1
        )

        # Average duration (convert seconds to ms)
        durations = [e.get("duration_seconds", 0) or 0 for e in ci_events if e.get("duration_seconds")]
        avg_duration_ms = int((sum(durations) / len(durations) * 1000) if durations else 0)

        # Get deployment events
        deploy_result = await supabase.request(
            f"/deployment_events?project_id=eq.{project_id}&select=id,status,duration_seconds,created_at"
        )
        deployments = deploy_result.get("data") or []

        total_deployments = len(deployments)
        deployments_last_24h = sum(
            1 for d in deployments
            if d.get("created_at", "") > yesterday_iso
        )
        successful_deployments = sum(1 for d in deployments if d.get("status") == "ready")
        deployment_success_rate = round(
            (successful_deployments / total_deployments * 100) if total_deployments > 0 else 0.0, 1
        )

        deploy_durations = [d.get("duration_seconds", 0) or 0 for d in deployments if d.get("duration_seconds")]
        avg_deploy_duration_ms = int((sum(deploy_durations) / len(deploy_durations) * 1000) if deploy_durations else 0)

        # Get quality/risk score
        quality_result = await supabase.request(
            f"/quality_scores?project_id=eq.{project_id}&select=overall_score&limit=1"
        )
        quality_data = quality_result.get("data") or []
        current_risk_score = 100 - (quality_data[0].get("overall_score", 50) if quality_data else 50)

        # Get impacted tests count (from recent impact graph activity)
        impact_result = await supabase.request(
            f"/test_impact_graph?project_id=eq.{project_id}"
            f"&updated_at=gte.{yesterday_iso}&select=id"
        )
        # Handle table not existing gracefully
        if impact_result.get("error"):
            tests_impacted = 0
        else:
            tests_impacted = len(impact_result.get("data") or [])

        logger.info(
            "CI/CD stats retrieved",
            project_id=project_id,
            total_pipelines=total_pipelines,
            total_deployments=total_deployments,
            user_id=user["user_id"],
        )

        return CICDStats(
            total_pipelines=total_pipelines,
            pipelines_last_24h=pipelines_last_24h,
            success_rate=success_rate,
            avg_pipeline_duration_ms=avg_duration_ms,
            total_builds=total_pipelines,  # For now, builds = pipelines
            builds_last_24h=pipelines_last_24h,
            build_success_rate=success_rate,
            avg_build_duration_ms=avg_duration_ms,
            total_deployments=total_deployments,
            deployments_last_24h=deployments_last_24h,
            deployment_success_rate=deployment_success_rate,
            avg_deployment_duration_ms=avg_deploy_duration_ms,
            current_risk_score=int(current_risk_score),
            tests_impacted_by_recent_changes=tests_impacted,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get CI/CD stats", error=str(e), project_id=project_id)
        raise HTTPException(status_code=500, detail=f"Failed to get CI/CD stats: {str(e)}")


# ============================================================================
# Test Impact Endpoints (RAP-198)
# ============================================================================


@router.get("/test-impact", response_model=TestImpactAnalysis | None)
async def get_test_impact(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    commit_sha: str | None = Query(None, description="Specific commit SHA (latest if not provided)"),
):
    """Get the latest or specific test impact analysis for a project.

    Returns the most recent test impact analysis, or a specific one if commit_sha is provided.
    """
    user = await get_current_user(request)
    await verify_project_access(project_id, user["user_id"], user.get("email"), request=request)

    supabase = get_supabase_client()

    try:
        # Build query
        query = f"/ci_test_impact?project_id=eq.{project_id}"
        if commit_sha:
            query += f"&commit_sha=eq.{commit_sha}"
        query += "&order=created_at.desc&limit=1"

        result = await supabase.request(query)

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            # Handle missing table gracefully
            if "does not exist" in error_msg or "42P01" in error_msg:
                logger.warning("ci_test_impact table not found")
                return None
            raise HTTPException(status_code=500, detail="Failed to fetch test impact analysis")

        data = result.get("data")
        if not data:
            return None

        analysis = data[0]

        return TestImpactAnalysis(
            id=analysis["id"],
            project_id=analysis["project_id"],
            commit_sha=analysis["commit_sha"],
            branch=analysis.get("branch", "main"),
            base_sha=analysis.get("base_sha"),
            changed_files=[ChangedFile(**f) for f in (analysis.get("changed_files") or [])],
            impacted_tests=[ImpactedTest(**t) for t in (analysis.get("impacted_tests") or [])],
            total_files_changed=analysis.get("total_files_changed", 0),
            total_tests_impacted=analysis.get("total_tests_impacted", 0),
            recommended_tests=analysis.get("recommended_tests") or [],
            skip_candidates=analysis.get("skip_candidates") or [],
            confidence_score=float(analysis.get("confidence_score", 0.5)),
            analysis_time_ms=analysis.get("analysis_time_ms", 0),
            created_at=analysis["created_at"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get test impact analysis", error=str(e), project_id=project_id)
        raise HTTPException(status_code=500, detail=f"Failed to get test impact analysis: {str(e)}")


@router.post("/test-impact/analyze", response_model=TestImpactAnalysis)
async def analyze_test_impact(
    request: Request,
    body: TestImpactAnalyzeRequest,
):
    """Run test impact analysis for a commit.

    RAP-250: Optimized to use precomputed test impact matrix first (~20-50ms),
    falling back to LLM analysis only when precomputed data is unavailable (~2-3s).

    Analyzes changed files and determines which tests are likely impacted.
    Priority order:
    1. Precomputed matrix lookup (fastest)
    2. Cached query results (fast)
    3. LLM analysis (slowest, most accurate)
    """
    start_time = time.time()
    user = await get_current_user(request)
    await verify_project_access(body.project_id, user["user_id"], user.get("email"), request=request)

    supabase = get_supabase_client()

    try:
        # Use provided changed files or create placeholder
        changed_files = body.changed_files or []

        # If no files provided, we'd typically fetch from git
        # For now, create a placeholder analysis
        if not changed_files:
            logger.info(
                "No changed files provided for analysis",
                project_id=body.project_id,
                commit_sha=body.commit_sha,
            )

        # Set impact scores for each file (simple heuristic for display only)
        for cf in changed_files:
            if cf.impact_score == 0.5:  # Default value
                path_lower = cf.path.lower()
                if any(p in path_lower for p in ["auth", "payment", "security", "api/"]):
                    cf.impact_score = 0.9
                elif any(p in path_lower for p in ["service", "handler", "model"]):
                    cf.impact_score = 0.7
                else:
                    cf.impact_score = 0.5

        # Get all tests for the project
        tests_result = await supabase.request(
            f"/tests?project_id=eq.{body.project_id}&is_active=eq.true&select=id,name,steps,tags,priority"
        )
        tests = tests_result.get("data") or []

        # Create lookup for test details
        tests_by_id = {t["id"]: t for t in tests}

        # RAP-250: Use precomputed test impact matrix first
        impacted_tests: list[ImpactedTest] = []
        recommended_tests: list[str] = []
        skip_candidates: list[str] = []
        overall_confidence = 0.8
        impact_source = "llm"  # Default to LLM

        # Get org_id for precomputed lookup
        try:
            from src.api.tests import get_project_org_id
            org_id = await get_project_org_id(body.project_id)
        except Exception:
            org_id = None

        # Try precomputed lookup first (RAP-250 optimization)
        precomputed_test_ids: list[str] = []
        if org_id and changed_files:
            file_paths = [cf.path for cf in changed_files]
            precomputed_test_ids, impact_source = await get_affected_tests(
                org_id=org_id,
                project_id=body.project_id,
                changed_files=file_paths,
            )

            if precomputed_test_ids:
                # Build ImpactedTest objects from precomputed results
                for test_id in precomputed_test_ids:
                    test_info = tests_by_id.get(test_id)
                    if test_info:
                        impacted_tests.append(ImpactedTest(
                            test_id=test_id,
                            test_name=test_info.get("name", ""),
                            impact_reason=f"Precomputed impact matrix ({impact_source})",
                            confidence=0.95,  # High confidence for precomputed data
                            priority=test_info.get("priority", "medium"),
                        ))
                    else:
                        # Test ID from matrix but not in current tests list
                        impacted_tests.append(ImpactedTest(
                            test_id=test_id,
                            test_name=test_id,
                            impact_reason=f"Precomputed impact matrix ({impact_source})",
                            confidence=0.90,
                            priority="medium",
                        ))

                recommended_tests = precomputed_test_ids
                all_test_ids = {t["id"] for t in tests}
                skip_candidates = list(all_test_ids - set(precomputed_test_ids))
                overall_confidence = 0.95

                precomputed_time_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    "Precomputed test impact analysis completed",
                    project_id=body.project_id,
                    impacted_count=len(impacted_tests),
                    source=impact_source,
                    latency_ms=precomputed_time_ms,
                )

        # Fall back to LLM analysis if precomputed data unavailable or incomplete
        if not impacted_tests:
            try:
                # Run AI analysis (slow path, ~2-3s)
                ai_impacted, ai_recommended, ai_skip, ai_confidence = await _run_ai_test_impact_analysis(
                    changed_files=[cf.model_dump() for cf in changed_files],
                    all_tests=tests,
                    commit_sha=body.commit_sha,
                    branch=body.branch,
                )

                # Convert AI results to ImpactedTest objects
                for impact in ai_impacted:
                    impacted_tests.append(ImpactedTest(
                        test_id=impact["test_id"],
                        test_name=impact["test_name"],
                        impact_reason=impact["impact_reason"],
                        confidence=impact["confidence"],
                        priority=impact["priority"],
                    ))

                recommended_tests = ai_recommended
                skip_candidates = ai_skip
                overall_confidence = ai_confidence
                impact_source = "llm"

                logger.info(
                    "LLM test impact analysis completed (precomputed unavailable)",
                    impacted_count=len(impacted_tests),
                    recommended_count=len(recommended_tests),
                )

            except Exception as ai_error:
                # Fallback: mark all tests as potentially impacted if AI fails
                logger.warning(
                    "AI analysis unavailable, using conservative fallback",
                    error=str(ai_error),
                )
                for test in tests:
                    impacted_tests.append(ImpactedTest(
                        test_id=test["id"],
                        test_name=test.get("name", ""),
                        impact_reason="AI unavailable - conservative selection",
                        confidence=0.5,
                        priority=test.get("priority", "medium"),
                    ))
                recommended_tests = [t["id"] for t in tests]
                skip_candidates = []
                overall_confidence = 0.5
                impact_source = "fallback"

        # Sort by confidence and priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        impacted_tests.sort(
            key=lambda t: (priority_order.get(t.priority, 2), -t.confidence)
        )

        # Calculate analysis time
        analysis_time_ms = int((time.time() - start_time) * 1000)

        # Create analysis record
        analysis_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        analysis_data = {
            "id": analysis_id,
            "project_id": body.project_id,
            "commit_sha": body.commit_sha,
            "branch": body.branch,
            "base_sha": body.base_sha,
            "changed_files": [cf.model_dump() for cf in changed_files],
            "impacted_tests": [t.model_dump() for t in impacted_tests],
            "total_files_changed": len(changed_files),
            "total_tests_impacted": len(impacted_tests),
            "recommended_tests": recommended_tests,
            "skip_candidates": skip_candidates[:50],  # Limit size
            "confidence_score": round(overall_confidence, 2),
            "analysis_time_ms": analysis_time_ms,
            "created_at": now,
            # RAP-250: Track source for performance monitoring
            "metadata": {"impact_source": impact_source},
        }

        # Store analysis (handle missing table gracefully)
        insert_result = await supabase.insert("ci_test_impact", analysis_data)
        if insert_result.get("error"):
            error_msg = str(insert_result.get("error", ""))
            if "does not exist" not in error_msg and "42P01" not in error_msg:
                logger.warning("Failed to store test impact analysis", error=insert_result.get("error"))

        # Log audit event
        try:
            from src.api.tests import get_project_org_id
            org_id = await get_project_org_id(body.project_id)
            await log_audit(
                organization_id=org_id,
                user_id=user["user_id"],
                user_email=user.get("email"),
                action="cicd.test_impact_analysis",
                resource_type="project",
                resource_id=body.project_id,
                description=f"Ran test impact analysis for commit {body.commit_sha[:8]}",
                metadata={
                    "commit_sha": body.commit_sha,
                    "files_changed": len(changed_files),
                    "tests_impacted": len(impacted_tests),
                },
                request=request,
            )
        except Exception as audit_error:
            logger.warning("Failed to log audit event", error=str(audit_error))

        # =====================================================================
        # AI Learning Loop Integration: Store in Cognee for cross-project learning
        # =====================================================================
        try:
            await _store_analysis_in_cognee(
                org_id=org_id,
                project_id=body.project_id,
                analysis_type="test_impact",
                data={
                    "commit_sha": body.commit_sha,
                    "branch": body.branch,
                    "changed_files": [cf.path for cf in changed_files],
                    "impacted_test_count": len(impacted_tests),
                    "impacted_test_ids": [t.test_id for t in impacted_tests],
                    "confidence_score": overall_confidence,
                    "recommended_tests": recommended_tests,
                    "skip_candidates": skip_candidates[:20],  # Limit for storage
                },
            )
        except Exception as cognee_error:
            logger.debug("Cognee storage skipped", error=str(cognee_error))

        logger.info(
            "Test impact analysis completed",
            project_id=body.project_id,
            commit_sha=body.commit_sha,
            files_changed=len(changed_files),
            tests_impacted=len(impacted_tests),
            analysis_time_ms=analysis_time_ms,
            user_id=user["user_id"],
        )

        return TestImpactAnalysis(
            id=analysis_id,
            project_id=body.project_id,
            commit_sha=body.commit_sha,
            branch=body.branch,
            base_sha=body.base_sha,
            changed_files=changed_files,
            impacted_tests=impacted_tests,
            total_files_changed=len(changed_files),
            total_tests_impacted=len(impacted_tests),
            recommended_tests=recommended_tests,
            skip_candidates=skip_candidates[:50],
            confidence_score=round(overall_confidence, 2),
            analysis_time_ms=analysis_time_ms,
            created_at=now,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to run test impact analysis", error=str(e), project_id=body.project_id)
        raise HTTPException(status_code=500, detail=f"Failed to run test impact analysis: {str(e)}")


# ============================================================================
# Deployment Risk Endpoint (RAP-198)
# ============================================================================


@router.get("/deployment-risk", response_model=DeploymentRiskResponse)
async def get_deployment_risk(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    commit_sha: str | None = Query(None, description="Specific commit SHA to assess"),
):
    """Calculate deployment risk score for a project or specific commit.

    Risk is calculated based on:
    - Recent CI/CD failure rate
    - Test coverage of changed files
    - Code complexity of changes
    - Historical deployment success rate
    - Time since last successful deployment
    """
    user = await get_current_user(request)
    await verify_project_access(project_id, user["user_id"], user.get("email"), request=request)

    supabase = get_supabase_client()

    now = datetime.now(UTC)
    now_iso = now.isoformat()
    week_ago = (now - timedelta(days=7)).isoformat()

    try:
        factors: dict = {}
        recommendations: list[str] = []

        # Factor 1: Recent CI/CD failure rate (0-25 points)
        ci_result = await supabase.request(
            f"/ci_events?project_id=eq.{project_id}"
            f"&created_at=gte.{week_ago}&select=status"
        )
        ci_events = ci_result.get("data") or []
        if ci_events:
            failures = sum(1 for e in ci_events if e.get("status") in ("failure", "error"))
            failure_rate = failures / len(ci_events)
            ci_risk = int(failure_rate * 25)
            factors["ci_failure_rate"] = round(failure_rate * 100, 1)
            if failure_rate > 0.3:
                recommendations.append("Fix failing CI pipelines before deploying")
        else:
            ci_risk = 12  # Medium risk if no CI data
            factors["ci_failure_rate"] = None

        # Factor 2: Test coverage / impacted tests (0-25 points)
        tests_impacted = 0
        tests_total = 0

        if commit_sha:
            # Check for existing impact analysis
            impact_result = await supabase.request(
                f"/ci_test_impact?project_id=eq.{project_id}"
                f"&commit_sha=eq.{commit_sha}&limit=1"
            )
            if not impact_result.get("error"):
                impact_data = impact_result.get("data")
                if impact_data:
                    tests_impacted = impact_data[0].get("total_tests_impacted", 0)

        # Get total test count
        tests_result = await supabase.request(
            f"/tests?project_id=eq.{project_id}&is_active=eq.true&select=id"
        )
        tests_total = len(tests_result.get("data") or [])

        if tests_total > 0:
            coverage_ratio = tests_impacted / tests_total if tests_impacted else 0.5
            test_risk = int(coverage_ratio * 25)
            factors["tests_impacted_ratio"] = round(coverage_ratio * 100, 1)
            if coverage_ratio > 0.5:
                recommendations.append(f"Run full test suite - {tests_impacted} tests potentially impacted")
        else:
            test_risk = 20  # High risk if no tests
            factors["tests_impacted_ratio"] = None
            recommendations.append("Add automated tests to reduce deployment risk")

        # Factor 3: Deployment history (0-25 points)
        deploy_result = await supabase.request(
            f"/deployment_events?project_id=eq.{project_id}"
            f"&created_at=gte.{week_ago}&select=status,created_at"
        )
        deployments = deploy_result.get("data") or []

        if deployments:
            failed_deploys = sum(1 for d in deployments if d.get("status") == "error")
            deploy_failure_rate = failed_deploys / len(deployments)
            deploy_risk = int(deploy_failure_rate * 25)
            factors["deployment_failure_rate"] = round(deploy_failure_rate * 100, 1)
            if deploy_failure_rate > 0.2:
                recommendations.append("Review recent deployment failures before proceeding")
        else:
            deploy_risk = 15  # Medium-high risk if no deployment history
            factors["deployment_failure_rate"] = None

        # Factor 4: Quality score (0-25 points)
        quality_result = await supabase.request(
            f"/quality_scores?project_id=eq.{project_id}&limit=1"
        )
        quality_data = quality_result.get("data")
        quality_score = None

        if quality_data:
            quality_score = quality_data[0].get("overall_score", 50)
            quality_risk = int((100 - quality_score) * 0.25)
            factors["quality_score"] = quality_score
        else:
            quality_risk = 12  # Medium risk if no quality data
            factors["quality_score"] = None

        # Try AI-powered risk analysis for more intelligent assessment
        try:
            ai_risk_score, ai_risk_level, ai_recommendations = await _run_ai_deployment_risk_analysis(
                project_id=project_id,
                commit_sha=commit_sha,
                ci_failure_rate=factors.get("ci_failure_rate"),
                deploy_failure_rate=factors.get("deployment_failure_rate"),
                tests_impacted=tests_impacted,
                tests_total=tests_total,
                quality_score=quality_score,
            )

            # Use AI results
            total_risk = ai_risk_score
            risk_level = ai_risk_level
            recommendations = ai_recommendations

            logger.info(
                "AI-powered deployment risk analysis completed",
                risk_score=total_risk,
                risk_level=risk_level,
            )

        except Exception as ai_error:
            # Fallback to simple calculation if AI fails
            logger.warning(
                "AI risk analysis unavailable, using fallback calculation",
                error=str(ai_error),
            )

            # Calculate total risk score (fallback)
            total_risk = ci_risk + test_risk + deploy_risk + quality_risk
            total_risk = min(100, max(0, total_risk))  # Clamp to 0-100

            risk_level = _calculate_risk_level(total_risk)

            # Add risk-level specific recommendations (fallback)
            if quality_score is not None and quality_score < 50:
                recommendations.append("Improve code quality score before deploying")
            if risk_level == "critical":
                recommendations.insert(0, "CRITICAL: Address critical issues before deploying to production")
            elif risk_level == "high":
                recommendations.insert(0, "Consider additional testing and review before deployment")

        # Estimate test time (rough estimate: 30s per test)
        estimated_test_time_ms = tests_impacted * 30 * 1000 if tests_impacted else tests_total * 30 * 1000

        logger.info(
            "Deployment risk calculated",
            project_id=project_id,
            commit_sha=commit_sha,
            risk_score=total_risk,
            risk_level=risk_level,
            user_id=user["user_id"],
        )

        return DeploymentRiskResponse(
            project_id=project_id,
            commit_sha=commit_sha,
            risk_score=total_risk,
            risk_level=risk_level,
            factors=factors,
            recommendations=recommendations[:5],  # Limit to top 5
            tests_to_run=tests_impacted or tests_total,
            estimated_test_time_ms=estimated_test_time_ms,
            calculated_at=now_iso,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to calculate deployment risk", error=str(e), project_id=project_id)
        raise HTTPException(status_code=500, detail=f"Failed to calculate deployment risk: {str(e)}")
