"""Performance Tests CRUD API endpoints.

Provides REST endpoints for:
- Listing performance tests (with optional project_id filter)
- Getting a single performance test by ID
- Getting the latest performance test for a project
- Getting performance trends over time
- Running a new performance test
- Deleting a performance test
"""

from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.middleware.tenant import validate_uuid, validate_uuid_optional
from src.api.projects import verify_project_access
from src.api.teams import get_current_user, log_audit
from src.api.tests import get_project_org_id
from src.services.supabase_client import get_supabase_client
from src.utils import safe_datetime

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/performance", tags=["Performance"])


# ============================================================================
# Request/Response Models
# ============================================================================

DeviceType = Literal["mobile", "desktop"]
StatusType = Literal["pending", "running", "completed", "failed"]
GradeType = Literal["excellent", "good", "needs_work", "poor"]
IssueSeverityType = Literal["critical", "high", "medium", "low"]


class PerformanceIssue(BaseModel):
    """A specific performance issue."""

    category: str
    severity: IssueSeverityType
    title: str
    description: str
    savings_ms: float | None = None
    savings_kb: float | None = None
    fix_suggestion: str | None = None


class PerformanceTestResponse(BaseModel):
    """Performance test details response."""

    id: str
    project_id: str
    url: str
    device: DeviceType
    status: StatusType

    # Core Web Vitals
    lcp_ms: float | None = None
    fid_ms: float | None = None
    cls: float | None = None
    inp_ms: float | None = None

    # Additional timing metrics
    ttfb_ms: float | None = None
    fcp_ms: float | None = None
    speed_index: float | None = None
    tti_ms: float | None = None
    tbt_ms: float | None = None

    # Resource metrics
    total_requests: int | None = None
    total_transfer_size_kb: float | None = None
    js_execution_time_ms: float | None = None
    dom_content_loaded_ms: float | None = None
    load_time_ms: float | None = None

    # Scores (0-100)
    performance_score: int | None = None
    accessibility_score: int | None = None
    best_practices_score: int | None = None
    seo_score: int | None = None

    # Overall grade
    overall_grade: GradeType | None = None

    # AI Analysis
    recommendations: list[str] = []
    issues: list[PerformanceIssue] = []
    summary: str | None = None

    # Metadata
    started_at: str | None = None
    completed_at: str | None = None
    triggered_by: str | None = None
    created_at: str


class PerformanceTestListResponse(BaseModel):
    """Paginated performance test list response."""

    tests: list[PerformanceTestResponse]
    total: int
    limit: int
    offset: int


class PerformanceTrendPoint(BaseModel):
    """A single point in the performance trend."""

    date: str
    lcp_ms: float
    fid_ms: float
    cls: float
    performance_score: float


class PerformanceTrendsResponse(BaseModel):
    """Performance trends over time."""

    trends: list[PerformanceTrendPoint]
    days: int


class RunPerformanceTestRequest(BaseModel):
    """Request to run a new performance test."""

    project_id: str = Field(..., description="Project ID")
    url: str = Field(..., description="URL to analyze")
    device: DeviceType = Field(default="mobile", description="Device type")


class PerformanceAverages(BaseModel):
    """Average metrics from recent tests."""

    avg_lcp: float
    avg_fid: float
    avg_cls: float
    avg_score: float


class PerformanceMetricsSummaryResponse(BaseModel):
    """Summary of performance metrics for a project."""

    latest_test: PerformanceTestResponse | None
    trends: list[PerformanceTrendPoint]
    averages: PerformanceAverages | None
    total_tests: int


# ============================================================================
# Helper Functions
# ============================================================================


def _db_to_response(test: dict) -> PerformanceTestResponse:
    """Convert database row to response model."""
    # Parse issues from JSONB
    raw_issues = test.get("issues", []) or []
    issues = []
    for issue in raw_issues:
        if isinstance(issue, dict):
            issues.append(
                PerformanceIssue(
                    category=issue.get("category", "unknown"),
                    severity=issue.get("severity", "medium"),
                    title=issue.get("title", ""),
                    description=issue.get("description", ""),
                    savings_ms=issue.get("savings_ms"),
                    savings_kb=issue.get("savings_kb"),
                    fix_suggestion=issue.get("fix_suggestion"),
                )
            )

    return PerformanceTestResponse(
        id=test["id"],
        project_id=test["project_id"],
        url=test["url"],
        device=test.get("device", "mobile"),
        status=test.get("status", "pending"),
        lcp_ms=test.get("lcp_ms"),
        fid_ms=test.get("fid_ms"),
        cls=test.get("cls"),
        inp_ms=test.get("inp_ms"),
        ttfb_ms=test.get("ttfb_ms"),
        fcp_ms=test.get("fcp_ms"),
        speed_index=test.get("speed_index"),
        tti_ms=test.get("tti_ms"),
        tbt_ms=test.get("tbt_ms"),
        total_requests=test.get("total_requests"),
        total_transfer_size_kb=test.get("total_transfer_size_kb"),
        js_execution_time_ms=test.get("js_execution_time_ms"),
        dom_content_loaded_ms=test.get("dom_content_loaded_ms"),
        load_time_ms=test.get("load_time_ms"),
        performance_score=test.get("performance_score"),
        accessibility_score=test.get("accessibility_score"),
        best_practices_score=test.get("best_practices_score"),
        seo_score=test.get("seo_score"),
        overall_grade=test.get("overall_grade"),
        recommendations=test.get("recommendations", []) or [],
        issues=issues,
        summary=test.get("summary"),
        started_at=test.get("started_at"),
        completed_at=test.get("completed_at"),
        triggered_by=test.get("triggered_by"),
        created_at=safe_datetime(test.get("created_at")),
    )


async def verify_performance_test_access(
    test_id: str, user_id: str, user_email: str = None, request: Request = None
) -> dict:
    """Verify user has access to the performance test via project organization membership."""
    supabase = get_supabase_client()

    test_result = await supabase.request(f"/performance_tests?id=eq.{test_id}&select=*")

    if not test_result.get("data"):
        raise HTTPException(status_code=404, detail="Performance test not found")

    test = test_result["data"][0]

    # Verify user has access to the project
    await verify_project_access(test["project_id"], user_id, user_email, request)

    return test


# ============================================================================
# Performance Test Endpoints
# ============================================================================


@router.get("/tests", response_model=PerformanceTestListResponse)
async def list_performance_tests(
    request: Request,
    project_id: str | None = None,
    status: StatusType | None = None,
    device: DeviceType | None = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum results")] = 10,
    offset: Annotated[int, Query(ge=0, le=10000, description="Offset")] = 0,
):
    """List performance tests with optional filters.

    Args:
        project_id: Filter by project ID (required for access control)
        status: Filter by status
        device: Filter by device type
        limit: Maximum number of results (default 10, max 100)
        offset: Offset for pagination (max 10000)

    Returns:
        Paginated list of performance tests
    """
    # RAP-292: UUID Validation
    validate_uuid_optional(project_id, "project_id")

    user = await get_current_user(request)
    supabase = get_supabase_client()

    if project_id:
        # Verify access to the specific project
        await verify_project_access(project_id, user["user_id"], user.get("email"), request)

        query = f"/performance_tests?project_id=eq.{project_id}&select=*&order=created_at.desc"
    else:
        # Get tests from all projects the user has access to
        memberships = await supabase.request(
            f"/organization_members?user_id=eq.{user['user_id']}&status=eq.active&select=organization_id"
        )

        if not memberships.get("data"):
            return PerformanceTestListResponse(tests=[], total=0, limit=limit, offset=offset)

        org_ids = [m["organization_id"] for m in memberships["data"]]

        projects_result = await supabase.request(
            f"/projects?organization_id=in.({','.join(org_ids)})&select=id"
        )

        if not projects_result.get("data"):
            return PerformanceTestListResponse(tests=[], total=0, limit=limit, offset=offset)

        project_ids = [p["id"] for p in projects_result["data"]]
        query = f"/performance_tests?project_id=in.({','.join(project_ids)})&select=*&order=created_at.desc"

    # Apply filters
    if status:
        query += f"&status=eq.{status}"

    if device:
        query += f"&device=eq.{device}"

    # Get total count first
    count_query = query.replace("&select=*", "&select=id")
    count_result = await supabase.request(count_query)
    total = len(count_result.get("data", []))

    # Apply pagination
    query += f"&limit={limit}&offset={offset}"

    tests_result = await supabase.request(query)

    if tests_result.get("error"):
        logger.error("Failed to fetch performance tests", error=tests_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch performance tests")

    tests = [_db_to_response(t) for t in tests_result.get("data", [])]

    return PerformanceTestListResponse(
        tests=tests,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/tests/latest", response_model=PerformanceTestResponse | None)
async def get_latest_performance_test(
    request: Request,
    project_id: str,
):
    """Get the latest completed performance test for a project.

    Args:
        project_id: Project ID

    Returns:
        Latest completed performance test or null
    """
    # RAP-292: UUID Validation
    validate_uuid(project_id, "project_id")

    user = await get_current_user(request)

    # Verify access to the project
    await verify_project_access(project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    result = await supabase.request(
        f"/performance_tests?project_id=eq.{project_id}&status=eq.completed"
        f"&select=*&order=created_at.desc&limit=1"
    )

    if result.get("error"):
        logger.error("Failed to fetch latest performance test", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch performance test")

    data = result.get("data", [])
    if not data:
        return None

    return _db_to_response(data[0])


@router.get("/trends", response_model=PerformanceTrendsResponse)
async def get_performance_trends(
    request: Request,
    project_id: str,
    days: Annotated[int, Query(ge=1, le=365, description="Number of days")] = 30,
):
    """Get performance trends over time.

    Args:
        project_id: Project ID
        days: Number of days to look back (default 30, max 365)

    Returns:
        Performance trends as time series
    """
    # RAP-292: UUID Validation
    validate_uuid(project_id, "project_id")

    user = await get_current_user(request)

    # Verify access to the project
    await verify_project_access(project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    start_date = datetime.now(UTC) - timedelta(days=days)

    result = await supabase.request(
        f"/performance_tests?project_id=eq.{project_id}&status=eq.completed"
        f"&created_at=gte.{start_date.isoformat()}"
        f"&select=created_at,lcp_ms,fid_ms,cls,performance_score"
        f"&order=created_at.asc"
    )

    if result.get("error"):
        logger.error("Failed to fetch performance trends", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch performance trends")

    trends = []
    for item in result.get("data", []):
        # Format date as "Jan 15" style
        created_at = safe_datetime(item.get("created_at"))
        date_str = datetime.fromisoformat(created_at).strftime("%b %d") if created_at else "Unknown"

        trends.append(
            PerformanceTrendPoint(
                date=date_str,
                lcp_ms=item.get("lcp_ms") or 0,
                fid_ms=item.get("fid_ms") or 0,
                cls=item.get("cls") or 0,
                performance_score=item.get("performance_score") or 0,
            )
        )

    return PerformanceTrendsResponse(trends=trends, days=days)


@router.get("/summary", response_model=PerformanceMetricsSummaryResponse)
async def get_performance_summary(
    request: Request,
    project_id: str,
    limit: Annotated[int, Query(ge=1, le=100, description="Tests for averages")] = 50,
):
    """Get performance metrics summary for a project.

    Combines latest test, trends, and averages in a single call.

    Args:
        project_id: Project ID
        limit: Number of tests to use for averages (default 50)

    Returns:
        Performance summary with latest, trends, and averages
    """
    # RAP-292: UUID Validation
    validate_uuid(project_id, "project_id")

    user = await get_current_user(request)

    # Verify access to the project
    await verify_project_access(project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Fetch recent tests for averages
    tests_result = await supabase.request(
        f"/performance_tests?project_id=eq.{project_id}&status=eq.completed"
        f"&select=*&order=created_at.desc&limit={limit}"
    )

    if tests_result.get("error"):
        logger.error("Failed to fetch performance summary", error=tests_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch performance summary")

    tests = tests_result.get("data", [])

    # Get latest test
    latest_test = _db_to_response(tests[0]) if tests else None

    # Calculate averages
    averages = None
    if tests:
        avg_lcp = sum(t.get("lcp_ms") or 0 for t in tests) / len(tests)
        avg_fid = sum(t.get("fid_ms") or 0 for t in tests) / len(tests)
        avg_cls = sum(t.get("cls") or 0 for t in tests) / len(tests)
        avg_score = sum(t.get("performance_score") or 0 for t in tests) / len(tests)

        averages = PerformanceAverages(
            avg_lcp=round(avg_lcp, 2),
            avg_fid=round(avg_fid, 2),
            avg_cls=round(avg_cls, 4),
            avg_score=round(avg_score, 1),
        )

    # Get 30-day trends
    start_date = datetime.now(UTC) - timedelta(days=30)
    trends_result = await supabase.request(
        f"/performance_tests?project_id=eq.{project_id}&status=eq.completed"
        f"&created_at=gte.{start_date.isoformat()}"
        f"&select=created_at,lcp_ms,fid_ms,cls,performance_score"
        f"&order=created_at.asc"
    )

    trends = []
    for item in trends_result.get("data", []):
        created_at = safe_datetime(item.get("created_at"))
        date_str = datetime.fromisoformat(created_at).strftime("%b %d") if created_at else "Unknown"

        trends.append(
            PerformanceTrendPoint(
                date=date_str,
                lcp_ms=item.get("lcp_ms") or 0,
                fid_ms=item.get("fid_ms") or 0,
                cls=item.get("cls") or 0,
                performance_score=item.get("performance_score") or 0,
            )
        )

    return PerformanceMetricsSummaryResponse(
        latest_test=latest_test,
        trends=trends,
        averages=averages,
        total_tests=len(tests),
    )


@router.get("/tests/{test_id}", response_model=PerformanceTestResponse)
async def get_performance_test(test_id: str, request: Request):
    """Get a single performance test by ID.

    Args:
        test_id: Performance test ID

    Returns:
        Performance test details
    """
    # RAP-292: UUID Validation
    validate_uuid(test_id, "test_id")

    user = await get_current_user(request)
    test = await verify_performance_test_access(test_id, user["user_id"], user.get("email"), request)

    return _db_to_response(test)


@router.post("/tests", response_model=PerformanceTestResponse)
async def run_performance_test(body: RunPerformanceTestRequest, request: Request):
    """Run a new performance test.

    Creates a performance test record and triggers analysis.

    Args:
        body: Performance test parameters

    Returns:
        Created performance test (status will be 'running')
    """
    # RAP-292: UUID Validation
    validate_uuid(body.project_id, "project_id")

    user = await get_current_user(request)

    # Verify access to the project
    await verify_project_access(body.project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Create performance test record
    test_data = {
        "project_id": body.project_id,
        "url": body.url,
        "device": body.device,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "triggered_by": user["user_id"],
        "recommendations": [],
        "issues": [],
    }

    result = await supabase.insert("performance_tests", test_data)

    if result.get("error"):
        logger.error("Failed to create performance test", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create performance test")

    test = result["data"][0]

    # Audit log
    org_id = await get_project_org_id(body.project_id)
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="performance_test.create",
        resource_type="performance_test",
        resource_id=test["id"],
        description=f"Started performance test for {body.url}",
        metadata={"url": body.url, "device": body.device, "project_id": body.project_id},
        request=request,
    )

    logger.info(
        "Performance test created",
        test_id=test["id"],
        url=body.url,
        device=body.device,
        project_id=body.project_id,
    )

    # TODO: Trigger async performance analysis via PerformanceAnalyzerAgent
    # For now, the test is created in 'running' status and will be updated
    # when the analysis completes (either via a background task or webhook)

    return _db_to_response(test)


@router.patch("/tests/{test_id}", response_model=PerformanceTestResponse)
async def update_performance_test(test_id: str, request: Request):
    """Update a performance test with results.

    This endpoint is used internally by the performance analyzer to update
    test results. The body should contain the analysis results.
    """
    # RAP-292: UUID Validation
    validate_uuid(test_id, "test_id")

    user = await get_current_user(request)
    test = await verify_performance_test_access(test_id, user["user_id"], user.get("email"), request)

    # Parse request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    supabase = get_supabase_client()

    # Build update data from body
    update_data = {}
    allowed_fields = [
        "status",
        "lcp_ms",
        "fid_ms",
        "cls",
        "inp_ms",
        "ttfb_ms",
        "fcp_ms",
        "speed_index",
        "tti_ms",
        "tbt_ms",
        "total_requests",
        "total_transfer_size_kb",
        "js_execution_time_ms",
        "dom_content_loaded_ms",
        "load_time_ms",
        "performance_score",
        "accessibility_score",
        "best_practices_score",
        "seo_score",
        "overall_grade",
        "recommendations",
        "issues",
        "summary",
        "completed_at",
    ]

    for field in allowed_fields:
        if field in body:
            update_data[field] = body[field]

    if not update_data:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    result = await supabase.update("performance_tests", {"id": f"eq.{test_id}"}, update_data)

    if result.get("error"):
        logger.error("Failed to update performance test", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to update performance test")

    # Fetch updated test
    updated_test = await verify_performance_test_access(
        test_id, user["user_id"], user.get("email"), request
    )

    logger.info("Performance test updated", test_id=test_id, status=update_data.get("status"))

    return _db_to_response(updated_test)


@router.delete("/tests/{test_id}")
async def delete_performance_test(test_id: str, request: Request):
    """Delete a performance test.

    Args:
        test_id: Performance test ID

    Returns:
        Success message
    """
    # RAP-292: UUID Validation
    validate_uuid(test_id, "test_id")

    user = await get_current_user(request)
    test = await verify_performance_test_access(test_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    delete_result = await supabase.request(f"/performance_tests?id=eq.{test_id}", method="DELETE")

    if delete_result.get("error"):
        logger.error("Failed to delete performance test", error=delete_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to delete performance test")

    # Audit log
    org_id = await get_project_org_id(test["project_id"])
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="performance_test.delete",
        resource_type="performance_test",
        resource_id=test_id,
        description=f"Deleted performance test for {test['url']}",
        metadata={"url": test["url"], "project_id": test["project_id"]},
        request=request,
    )

    logger.info("Performance test deleted", test_id=test_id)

    return {"success": True, "message": "Performance test deleted"}
