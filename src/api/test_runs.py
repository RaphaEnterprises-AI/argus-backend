"""Test Runs CRUD API endpoints.

Provides standard REST endpoints for:
- Listing test runs (with optional project_id filter)
- Getting a single test run by ID (with optional results)
- Creating a test run
- Updating a test run status
- Deleting a test run

Note: Test execution is handled by /api/v1/browser/test endpoint.
This module handles the data management for test runs.
"""

import urllib.parse
from datetime import UTC, datetime
from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.middleware.tenant import validate_uuid, validate_uuid_optional
from src.api.projects import verify_project_access
from src.api.teams import get_current_user, log_audit
from src.api.tests import get_project_org_id, get_project_org_ids_batch
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["Test Runs"])


# ============================================================================
# Request/Response Models
# ============================================================================

TestRunStatus = Literal["pending", "running", "passed", "failed", "cancelled", "error"]
TriggerType = Literal["manual", "scheduled", "ci", "webhook", "api"]


class TestResultSummary(BaseModel):
    """Summary of a test result within a run."""
    id: str
    test_id: str | None
    name: str
    status: str
    duration_ms: int | None
    error_message: str | None
    created_at: str


class TestRunResponse(BaseModel):
    """Test run details response."""
    id: str
    project_id: str
    name: str | None
    app_url: str | None
    browser: str | None
    status: TestRunStatus
    trigger: TriggerType | None
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    duration_ms: int | None
    started_at: str | None
    completed_at: str | None
    created_by: str | None
    created_at: str
    updated_at: str | None


class TestRunWithResultsResponse(TestRunResponse):
    """Test run with embedded results."""
    results: list[TestResultSummary]


class TestRunListResponse(BaseModel):
    """Test run list item response."""
    id: str
    project_id: str
    name: str | None
    status: TestRunStatus
    trigger: TriggerType | None
    total_tests: int
    passed_tests: int
    failed_tests: int
    duration_ms: int | None
    completed_at: str | None
    created_at: str


class CreateTestRunRequest(BaseModel):
    """Request to create a new test run."""
    project_id: str = Field(..., description="Project ID")
    name: str | None = Field(None, max_length=255, description="Optional run name")
    app_url: str | None = Field(None, max_length=500, description="Application URL being tested")
    browser: str | None = Field(None, max_length=50, description="Browser used for testing")
    trigger: TriggerType = Field(default="api", description="What triggered this run")
    total_tests: int = Field(default=0, ge=0, description="Total number of tests to run")


class UpdateTestRunRequest(BaseModel):
    """Request to update a test run."""
    name: str | None = Field(None, max_length=255, description="Run name")
    status: TestRunStatus | None = Field(None, description="Run status")
    passed_tests: int | None = Field(None, ge=0, description="Number of passed tests")
    failed_tests: int | None = Field(None, ge=0, description="Number of failed tests")
    skipped_tests: int | None = Field(None, ge=0, description="Number of skipped tests")
    duration_ms: int | None = Field(None, ge=0, description="Duration in milliseconds")
    completed_at: str | None = Field(None, description="Completion timestamp")


class TestRunListPaginatedResponse(BaseModel):
    """Paginated test run list response."""
    runs: list[TestRunListResponse]
    total: int
    limit: int
    offset: int


# ============================================================================
# Test Result Models
# ============================================================================

class TestResultResponse(BaseModel):
    """Full test result response."""
    id: str
    test_run_id: str
    test_id: str | None
    name: str
    status: str
    duration_ms: int | None
    steps_total: int | None
    steps_completed: int | None
    error_message: str | None
    error_screenshot: str | None
    step_results: list[dict] | None
    completed_at: str | None
    created_at: str


class CreateTestResultRequest(BaseModel):
    """Request to add a test result to a run."""
    test_id: str | None = Field(None, description="ID of the test that was run")
    name: str = Field(..., min_length=1, max_length=255, description="Test name")
    status: str = Field(..., description="Result status (passed, failed, skipped, error)")
    duration_ms: int | None = Field(None, ge=0, description="Duration in milliseconds")
    steps_total: int | None = Field(None, ge=0, description="Total steps in test")
    steps_completed: int | None = Field(None, ge=0, description="Steps completed")
    error_message: str | None = Field(None, description="Error message if failed")
    error_screenshot: str | None = Field(None, description="Screenshot on failure")
    step_results: list[dict] | None = Field(None, description="Individual step results")


# ============================================================================
# Helper Functions
# ============================================================================

async def verify_test_run_access(run_id: str, user_id: str, user_email: str = None, request: Request = None) -> dict:
    """Verify user has access to the test run via project organization membership.

    Returns the test run data if access is granted.
    """
    supabase = get_supabase_client()

    # Get test run
    run_result = await supabase.request(
        f"/test_runs?id=eq.{run_id}&select=*"
    )

    if not run_result.get("data"):
        raise HTTPException(status_code=404, detail="Test run not found")

    run = run_result["data"][0]

    # Verify user has access to the project
    await verify_project_access(run["project_id"], user_id, user_email, request)

    return run


# ============================================================================
# Test Run Endpoints
# ============================================================================

@router.get("/test-runs", response_model=TestRunListPaginatedResponse)
async def list_test_runs(
    request: Request,
    project_id: str | None = None,
    status: TestRunStatus | None = None,
    trigger: TriggerType | None = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Maximum results")] = 50,
    offset: Annotated[int, Query(ge=0, le=10000, description="Offset for pagination")] = 0,
):
    """List test runs with optional filters.

    Args:
        project_id: Filter by project ID (required for non-admin users)
        status: Filter by run status
        trigger: Filter by trigger type
        limit: Maximum results (default 50, max 500)
        offset: Pagination offset (max 10000)

    Returns:
        Paginated list of test runs
    """
    # RAP-292: UUID Validation
    validate_uuid_optional(project_id, "project_id")

    user = await get_current_user(request)
    supabase = get_supabase_client()

    if project_id:
        # Verify access to the specific project
        await verify_project_access(project_id, user["user_id"], user.get("email"), request)

        # Build query for specific project
        query = f"/test_runs?project_id=eq.{project_id}&select=*&order=created_at.desc"
    else:
        # Get runs from all projects the user has access to via organizations
        memberships = await supabase.request(
            f"/organization_members?user_id=eq.{user['user_id']}&status=eq.active&select=organization_id"
        )

        if not memberships.get("data"):
            return TestRunListPaginatedResponse(runs=[], total=0, limit=limit, offset=offset)

        org_ids = [m["organization_id"] for m in memberships["data"]]

        # Get all projects from user's organizations
        projects_result = await supabase.request(
            f"/projects?organization_id=in.({','.join(org_ids)})&select=id"
        )

        if not projects_result.get("data"):
            return TestRunListPaginatedResponse(runs=[], total=0, limit=limit, offset=offset)

        project_ids = [p["id"] for p in projects_result["data"]]
        query = f"/test_runs?project_id=in.({','.join(project_ids)})&select=*&order=created_at.desc"

    # Apply filters
    if status:
        query += f"&status=eq.{status}"

    if trigger:
        query += f"&trigger=eq.{trigger}"

    # Get total count first (without pagination)
    count_query = query.replace("&select=*", "&select=id")
    count_result = await supabase.request(count_query)
    total = len(count_result.get("data", []))

    # Apply pagination
    query += f"&limit={limit}&offset={offset}"

    runs_result = await supabase.request(query)

    if runs_result.get("error"):
        logger.error("Failed to fetch test runs", error=runs_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch test runs")

    runs = runs_result.get("data", [])

    result = [
        TestRunListResponse(
            id=run["id"],
            project_id=run["project_id"],
            name=run.get("name"),
            status=run.get("status", "pending"),
            trigger=run.get("trigger"),
            total_tests=run.get("total_tests", 0) or 0,
            passed_tests=run.get("passed_tests", 0) or 0,
            failed_tests=run.get("failed_tests", 0) or 0,
            duration_ms=run.get("duration_ms"),
            completed_at=run.get("completed_at"),
            created_at=run["created_at"],
        )
        for run in runs
    ]

    return TestRunListPaginatedResponse(
        runs=result,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/test-runs", response_model=TestRunResponse)
async def create_test_run(body: CreateTestRunRequest, request: Request):
    """Create a new test run.

    Requires membership in the project's organization.
    """
    # RAP-292: UUID Validation
    validate_uuid(body.project_id, "project_id")

    user = await get_current_user(request)

    # Verify access to the project
    await verify_project_access(body.project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Create test run
    run_data = {
        "project_id": body.project_id,
        "name": body.name,
        "app_url": body.app_url,
        "browser": body.browser,
        "trigger": body.trigger,
        "status": "pending",
        "total_tests": body.total_tests,
        "passed_tests": 0,
        "failed_tests": 0,
        "skipped_tests": 0,
        "created_by": user["user_id"],
    }

    result = await supabase.insert("test_runs", run_data)

    if result.get("error"):
        logger.error("Failed to create test run", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create test run")

    run = result["data"][0]

    # Audit log
    org_id = await get_project_org_id(body.project_id)
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="test_run.create",
        resource_type="test_run",
        resource_id=run["id"],
        description=f"Created test run '{body.name or 'Unnamed'}'",
        metadata={"project_id": body.project_id, "trigger": body.trigger},
        request=request,
    )

    logger.info("Test run created", run_id=run["id"], project_id=body.project_id)

    return TestRunResponse(
        id=run["id"],
        project_id=run["project_id"],
        name=run.get("name"),
        app_url=run.get("app_url"),
        browser=run.get("browser"),
        status=run.get("status", "pending"),
        trigger=run.get("trigger"),
        total_tests=run.get("total_tests", 0) or 0,
        passed_tests=run.get("passed_tests", 0) or 0,
        failed_tests=run.get("failed_tests", 0) or 0,
        skipped_tests=run.get("skipped_tests", 0) or 0,
        duration_ms=run.get("duration_ms"),
        started_at=run.get("started_at"),
        completed_at=run.get("completed_at"),
        created_by=run.get("created_by"),
        created_at=run["created_at"],
        updated_at=run.get("updated_at"),
    )


@router.get("/test-runs/{run_id}", response_model=TestRunWithResultsResponse)
async def get_test_run(
    run_id: str,
    request: Request,
    include_results: Annotated[bool, Query(description="Include test results")] = True,
):
    """Get a single test run by ID.

    Optionally includes test results.
    Requires membership in the run's project organization.
    """
    # RAP-292: UUID Validation
    validate_uuid(run_id, "run_id")

    user = await get_current_user(request)
    run = await verify_test_run_access(run_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Fetch results if requested
    results = []
    if include_results:
        results_result = await supabase.request(
            f"/test_results?test_run_id=eq.{run_id}&select=*&order=created_at.asc"
        )

        if results_result.get("data"):
            results = [
                TestResultSummary(
                    id=r["id"],
                    test_id=r.get("test_id"),
                    name=r["name"],
                    status=r.get("status", "unknown"),
                    duration_ms=r.get("duration_ms"),
                    error_message=r.get("error_message"),
                    created_at=r["created_at"],
                )
                for r in results_result["data"]
            ]

    return TestRunWithResultsResponse(
        id=run["id"],
        project_id=run["project_id"],
        name=run.get("name"),
        app_url=run.get("app_url"),
        browser=run.get("browser"),
        status=run.get("status", "pending"),
        trigger=run.get("trigger"),
        total_tests=run.get("total_tests", 0) or 0,
        passed_tests=run.get("passed_tests", 0) or 0,
        failed_tests=run.get("failed_tests", 0) or 0,
        skipped_tests=run.get("skipped_tests", 0) or 0,
        duration_ms=run.get("duration_ms"),
        started_at=run.get("started_at"),
        completed_at=run.get("completed_at"),
        created_by=run.get("created_by"),
        created_at=run["created_at"],
        updated_at=run.get("updated_at"),
        results=results,
    )


@router.put("/test-runs/{run_id}", response_model=TestRunResponse)
@router.patch("/test-runs/{run_id}", response_model=TestRunResponse)
async def update_test_run(run_id: str, body: UpdateTestRunRequest, request: Request):
    """Update a test run.

    PUT: Full update (all fields can be provided)
    PATCH: Partial update (only provided fields are updated)

    Requires membership in the run's project organization.
    """
    # RAP-292: UUID Validation
    validate_uuid(run_id, "run_id")

    user = await get_current_user(request)
    run = await verify_test_run_access(run_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Build update data
    update_data = {"updated_at": datetime.now(UTC).isoformat()}

    if body.name is not None:
        update_data["name"] = body.name
    if body.status is not None:
        update_data["status"] = body.status
    if body.passed_tests is not None:
        update_data["passed_tests"] = body.passed_tests
    if body.failed_tests is not None:
        update_data["failed_tests"] = body.failed_tests
    if body.skipped_tests is not None:
        update_data["skipped_tests"] = body.skipped_tests
    if body.duration_ms is not None:
        update_data["duration_ms"] = body.duration_ms
    if body.completed_at is not None:
        update_data["completed_at"] = body.completed_at

    result = await supabase.update("test_runs", {"id": f"eq.{run_id}"}, update_data)

    if result.get("error"):
        logger.error("Failed to update test run", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to update test run")

    # Audit log
    org_id = await get_project_org_id(run["project_id"])
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="test_run.update",
        resource_type="test_run",
        resource_id=run_id,
        description=f"Updated test run '{run.get('name') or 'Unnamed'}'",
        metadata={"changes": {k: v for k, v in update_data.items() if k != "updated_at"}},
        request=request,
    )

    logger.info("Test run updated", run_id=run_id)

    # Fetch updated run
    updated_result = await supabase.request(f"/test_runs?id=eq.{run_id}&select=*")
    updated_run = updated_result["data"][0]

    return TestRunResponse(
        id=updated_run["id"],
        project_id=updated_run["project_id"],
        name=updated_run.get("name"),
        app_url=updated_run.get("app_url"),
        browser=updated_run.get("browser"),
        status=updated_run.get("status", "pending"),
        trigger=updated_run.get("trigger"),
        total_tests=updated_run.get("total_tests", 0) or 0,
        passed_tests=updated_run.get("passed_tests", 0) or 0,
        failed_tests=updated_run.get("failed_tests", 0) or 0,
        skipped_tests=updated_run.get("skipped_tests", 0) or 0,
        duration_ms=updated_run.get("duration_ms"),
        started_at=updated_run.get("started_at"),
        completed_at=updated_run.get("completed_at"),
        created_by=updated_run.get("created_by"),
        created_at=updated_run["created_at"],
        updated_at=updated_run.get("updated_at"),
    )


@router.delete("/test-runs/{run_id}")
async def delete_test_run(run_id: str, request: Request):
    """Delete a test run and all its results.

    Requires membership in the run's project organization.
    """
    # RAP-292: UUID Validation
    validate_uuid(run_id, "run_id")

    user = await get_current_user(request)
    run = await verify_test_run_access(run_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Delete test results first (cascade)
    await supabase.request(
        f"/test_results?test_run_id=eq.{run_id}",
        method="DELETE"
    )

    # Delete test run
    delete_result = await supabase.request(
        f"/test_runs?id=eq.{run_id}",
        method="DELETE"
    )

    if delete_result.get("error"):
        logger.error("Failed to delete test run", error=delete_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to delete test run")

    # Audit log
    org_id = await get_project_org_id(run["project_id"])
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="test_run.delete",
        resource_type="test_run",
        resource_id=run_id,
        description=f"Deleted test run '{run.get('name') or 'Unnamed'}'",
        metadata={"project_id": run["project_id"]},
        request=request,
    )

    logger.info("Test run deleted", run_id=run_id)

    return {"success": True, "message": f"Test run has been deleted"}


# ============================================================================
# Test Result Endpoints
# ============================================================================

@router.get("/test-runs/{run_id}/results", response_model=list[TestResultResponse])
async def get_test_run_results(run_id: str, request: Request):
    """Get all results for a test run.

    Requires membership in the run's project organization.
    """
    # RAP-292: UUID Validation
    validate_uuid(run_id, "run_id")

    user = await get_current_user(request)
    await verify_test_run_access(run_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    results_result = await supabase.request(
        f"/test_results?test_run_id=eq.{run_id}&select=*&order=created_at.asc"
    )

    if results_result.get("error"):
        logger.error("Failed to fetch test results", error=results_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch test results")

    return [
        TestResultResponse(
            id=r["id"],
            test_run_id=r["test_run_id"],
            test_id=r.get("test_id"),
            name=r["name"],
            status=r.get("status", "unknown"),
            duration_ms=r.get("duration_ms"),
            steps_total=r.get("steps_total"),
            steps_completed=r.get("steps_completed"),
            error_message=r.get("error_message"),
            error_screenshot=r.get("error_screenshot"),
            step_results=r.get("step_results"),
            completed_at=r.get("completed_at"),
            created_at=r["created_at"],
        )
        for r in results_result.get("data", [])
    ]


@router.post("/test-runs/{run_id}/results", response_model=TestResultResponse)
async def add_test_result(run_id: str, body: CreateTestResultRequest, request: Request):
    """Add a test result to a run.

    Requires membership in the run's project organization.
    """
    # RAP-292: UUID Validation
    validate_uuid(run_id, "run_id")
    validate_uuid_optional(body.test_id, "test_id")

    user = await get_current_user(request)
    run = await verify_test_run_access(run_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Create result
    result_data = {
        "test_run_id": run_id,
        "test_id": body.test_id,
        "name": body.name,
        "status": body.status,
        "duration_ms": body.duration_ms,
        "steps_total": body.steps_total,
        "steps_completed": body.steps_completed,
        "error_message": body.error_message,
        "error_screenshot": body.error_screenshot,
        "step_results": body.step_results,
        "completed_at": datetime.now(UTC).isoformat() if body.status in ["passed", "failed", "skipped", "error"] else None,
    }

    result = await supabase.insert("test_results", result_data)

    if result.get("error"):
        logger.error("Failed to add test result", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to add test result")

    test_result = result["data"][0]

    logger.info("Test result added", run_id=run_id, result_id=test_result["id"], status=body.status)

    return TestResultResponse(
        id=test_result["id"],
        test_run_id=test_result["test_run_id"],
        test_id=test_result.get("test_id"),
        name=test_result["name"],
        status=test_result.get("status", "unknown"),
        duration_ms=test_result.get("duration_ms"),
        steps_total=test_result.get("steps_total"),
        steps_completed=test_result.get("steps_completed"),
        error_message=test_result.get("error_message"),
        error_screenshot=test_result.get("error_screenshot"),
        step_results=test_result.get("step_results"),
        completed_at=test_result.get("completed_at"),
        created_at=test_result["created_at"],
    )


# ============================================================================
# Comparison Endpoint
# ============================================================================

class TestRunComparisonResponse(BaseModel):
    """Test run comparison response."""
    current_run: TestRunListResponse | None
    previous_run: TestRunListResponse | None
    deltas: dict
    has_previous_run: bool


@router.get("/test-runs/comparison/{project_id}", response_model=TestRunComparisonResponse)
async def get_test_run_comparison(project_id: str, request: Request):
    """Compare the latest test run with the previous one.

    Returns delta values for passed/failed tests and duration.
    Requires membership in the project's organization.
    """
    # RAP-292: UUID Validation
    validate_uuid(project_id, "project_id")

    user = await get_current_user(request)
    await verify_project_access(project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Fetch the two most recent completed test runs
    runs_result = await supabase.request(
        f"/test_runs?project_id=eq.{project_id}&status=in.(passed,failed)&select=*&order=created_at.desc&limit=2"
    )

    if runs_result.get("error"):
        logger.error("Failed to fetch test runs for comparison", error=runs_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch test runs")

    runs = runs_result.get("data", [])
    current_run = runs[0] if len(runs) > 0 else None
    previous_run = runs[1] if len(runs) > 1 else None

    # Calculate deltas
    deltas = {
        "passed_delta": None,
        "failed_delta": None,
        "duration_delta": None,
        "pass_rate_delta": None,
    }

    if current_run and previous_run:
        deltas["passed_delta"] = (current_run.get("passed_tests", 0) or 0) - (previous_run.get("passed_tests", 0) or 0)
        deltas["failed_delta"] = (current_run.get("failed_tests", 0) or 0) - (previous_run.get("failed_tests", 0) or 0)

        if current_run.get("duration_ms") and previous_run.get("duration_ms"):
            deltas["duration_delta"] = current_run["duration_ms"] - previous_run["duration_ms"]

        # Calculate pass rate delta
        current_total = (current_run.get("passed_tests", 0) or 0) + (current_run.get("failed_tests", 0) or 0)
        previous_total = (previous_run.get("passed_tests", 0) or 0) + (previous_run.get("failed_tests", 0) or 0)

        if current_total > 0 and previous_total > 0:
            current_pass_rate = ((current_run.get("passed_tests", 0) or 0) / current_total) * 100
            previous_pass_rate = ((previous_run.get("passed_tests", 0) or 0) / previous_total) * 100
            deltas["pass_rate_delta"] = current_pass_rate - previous_pass_rate

    def to_list_response(run: dict | None) -> TestRunListResponse | None:
        if not run:
            return None
        return TestRunListResponse(
            id=run["id"],
            project_id=run["project_id"],
            name=run.get("name"),
            status=run.get("status", "pending"),
            trigger=run.get("trigger"),
            total_tests=run.get("total_tests", 0) or 0,
            passed_tests=run.get("passed_tests", 0) or 0,
            failed_tests=run.get("failed_tests", 0) or 0,
            duration_ms=run.get("duration_ms"),
            completed_at=run.get("completed_at"),
            created_at=run["created_at"],
        )

    return TestRunComparisonResponse(
        current_run=to_list_response(current_run),
        previous_run=to_list_response(previous_run),
        deltas=deltas,
        has_previous_run=previous_run is not None,
    )
