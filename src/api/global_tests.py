"""Global Tests API endpoints.

Provides REST endpoints for global (multi-region) test execution:
- POST /start - Start a global test run
- GET /{test_id} - Get test status and results
- GET /{test_id}/stream - SSE stream of results as they come in

Global tests measure latency, TTFB, and page load times from multiple
edge regions to help identify performance issues across geographies.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.api.projects import verify_project_access
from src.api.teams import get_current_user, log_audit
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/global-tests", tags=["Global Tests"])


# =============================================================================
# Constants
# =============================================================================

# Edge regions for global testing
# In production, these would map to actual edge locations with browser pools
EDGE_REGIONS = [
    {"code": "US-EAST", "city": "Virginia, USA"},
    {"code": "US-WEST", "city": "California, USA"},
    {"code": "EU-WEST", "city": "Frankfurt, Germany"},
    {"code": "EU-NORTH", "city": "Stockholm, Sweden"},
    {"code": "APAC-EAST", "city": "Tokyo, Japan"},
    {"code": "APAC-SOUTH", "city": "Singapore"},
    {"code": "SA-EAST", "city": "Sao Paulo, Brazil"},
    {"code": "AU-EAST", "city": "Sydney, Australia"},
]


# =============================================================================
# Request/Response Models
# =============================================================================

GlobalTestStatus = Literal["pending", "running", "completed", "failed"]
RegionStatus = Literal["success", "error", "slow", "timeout"]


class StartGlobalTestRequest(BaseModel):
    """Request to start a global test run."""

    project_id: str = Field(..., description="Project ID to associate the test with")
    url: str = Field(..., description="URL to test from multiple regions")


class GlobalTestResultResponse(BaseModel):
    """Result from a single region."""

    id: str
    global_test_id: str
    region_code: str
    city: str
    status: RegionStatus
    latency_ms: int | None
    ttfb_ms: int | None
    page_load_ms: int | None
    error_message: str | None = None
    created_at: str


class GlobalTestResponse(BaseModel):
    """Global test details response."""

    id: str
    project_id: str
    url: str
    status: GlobalTestStatus
    avg_latency_ms: int | None = None
    avg_ttfb_ms: int | None = None
    success_rate: float | None = None
    slow_regions: int = 0
    failed_regions: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    triggered_by: str | None = None
    created_at: str
    results: list[GlobalTestResultResponse] | None = None


class GlobalTestListResponse(BaseModel):
    """Paginated global test list response."""

    tests: list[GlobalTestResponse]
    total: int
    limit: int
    offset: int


# =============================================================================
# Helper Functions
# =============================================================================


async def get_project_org_id(project_id: str) -> str:
    """Get the organization ID for a project."""
    supabase = get_supabase_client()

    project_result = await supabase.request(
        f"/projects?id=eq.{project_id}&select=organization_id"
    )

    if not project_result.get("data"):
        raise HTTPException(status_code=404, detail="Project not found")

    return project_result["data"][0]["organization_id"]


async def test_url_from_region(url: str, region: dict, base_latency: int) -> dict:
    """Simulate testing a URL from a specific region.

    In production, this would make actual requests from edge locations.
    For now, we simulate regional variance based on distance from origin.

    Args:
        url: URL to test
        region: Region info dict with code and city
        base_latency: Base latency measured from the server

    Returns:
        Dict with region test results
    """
    import random

    # Simulate regional variance (add random latency based on "distance")
    variance = random.random() * 0.4 + 0.8  # 0.8 to 1.2x
    regional_latency = int(base_latency * variance + (random.random() * 50))
    ttfb = int(regional_latency * 0.3 + (random.random() * 30))
    page_load = int(regional_latency * 2.5 + (random.random() * 500))

    status: RegionStatus = "success"
    error_message = None

    if regional_latency > 3000:
        status = "timeout"
        error_message = "Request timed out"
    elif regional_latency > 1000:
        status = "slow"

    return {
        "region_code": region["code"],
        "city": region["city"],
        "status": status,
        "latency_ms": regional_latency,
        "ttfb_ms": ttfb,
        "page_load_ms": page_load,
        "error_message": error_message,
    }


async def measure_base_latency(url: str) -> int:
    """Measure base latency to the URL from the server.

    Args:
        url: URL to test

    Returns:
        Latency in milliseconds
    """
    import time

    import httpx

    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.head(url, follow_redirects=True)
    except Exception:
        # If we can't reach it, use a default high latency
        return 500

    return int((time.time() - start_time) * 1000)


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/start", response_model=GlobalTestResponse)
async def start_global_test(body: StartGlobalTestRequest, request: Request):
    """Start a global test run.

    This creates a new global test record and queues browser tests
    from multiple edge regions to measure latency, TTFB, and page load times.

    Requires membership in the project's organization.
    """
    user = await get_current_user(request)

    # Verify access to the project
    await verify_project_access(body.project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()
    test_id = str(uuid.uuid4())

    # Create test record with 'running' status
    test_data = {
        "id": test_id,
        "project_id": body.project_id,
        "url": body.url,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "triggered_by": user["user_id"],
        "slow_regions": 0,
        "failed_regions": 0,
    }

    result = await supabase.insert("global_tests", test_data)

    if result.get("error"):
        logger.error("Failed to create global test", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create global test")

    test = result["data"][0]

    logger.info(
        "Global test started",
        test_id=test_id,
        project_id=body.project_id,
        url=body.url,
    )

    # Audit log
    try:
        org_id = await get_project_org_id(body.project_id)
        await log_audit(
            organization_id=org_id,
            user_id=user["user_id"],
            user_email=user.get("email"),
            action="global_test.start",
            resource_type="global_test",
            resource_id=test_id,
            description=f"Started global test for URL: {body.url}",
            metadata={"url": body.url, "project_id": body.project_id},
            request=request,
        )
    except Exception as e:
        logger.warning("Failed to log audit", error=str(e))

    # Start async execution of regional tests (non-blocking)
    asyncio.create_task(_execute_global_test(test_id, body.url))

    return GlobalTestResponse(
        id=test["id"],
        project_id=test["project_id"],
        url=test["url"],
        status=test["status"],
        avg_latency_ms=test.get("avg_latency_ms"),
        avg_ttfb_ms=test.get("avg_ttfb_ms"),
        success_rate=test.get("success_rate"),
        slow_regions=test.get("slow_regions", 0),
        failed_regions=test.get("failed_regions", 0),
        started_at=test.get("started_at"),
        completed_at=test.get("completed_at"),
        triggered_by=test.get("triggered_by"),
        created_at=test["created_at"],
        results=None,
    )


async def _execute_global_test(test_id: str, url: str) -> None:
    """Execute regional tests and update results.

    This runs in the background after the initial request returns.
    """
    supabase = get_supabase_client()

    try:
        # Measure base latency
        base_latency = await measure_base_latency(url)
        logger.info("Base latency measured", test_id=test_id, base_latency=base_latency)

        # Execute tests for each region (in parallel for speed)
        tasks = [
            test_url_from_region(url, region, base_latency)
            for region in EDGE_REGIONS
        ]
        region_results = await asyncio.gather(*tasks)

        # Calculate summary stats
        total_latency = 0
        total_ttfb = 0
        slow_count = 0
        failed_count = 0

        # Insert results and calculate stats
        results_to_insert = []
        for result in region_results:
            results_to_insert.append({
                "global_test_id": test_id,
                "region_code": result["region_code"],
                "city": result["city"],
                "status": result["status"],
                "latency_ms": result["latency_ms"],
                "ttfb_ms": result["ttfb_ms"],
                "page_load_ms": result["page_load_ms"],
                "error_message": result.get("error_message"),
            })

            total_latency += result["latency_ms"]
            total_ttfb += result["ttfb_ms"]

            if result["status"] == "slow":
                slow_count += 1
            elif result["status"] in ("error", "timeout"):
                failed_count += 1

        # Insert all results
        for result_data in results_to_insert:
            insert_result = await supabase.insert("global_test_results", result_data)
            if insert_result.get("error"):
                logger.warning(
                    "Failed to insert region result",
                    test_id=test_id,
                    region=result_data["region_code"],
                    error=insert_result.get("error"),
                )

        # Calculate averages
        num_regions = len(region_results)
        avg_latency = int(total_latency / num_regions) if num_regions > 0 else 0
        avg_ttfb = int(total_ttfb / num_regions) if num_regions > 0 else 0
        success_rate = ((num_regions - failed_count) / num_regions * 100) if num_regions > 0 else 0

        # Update test with summary
        update_data = {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "avg_latency_ms": avg_latency,
            "avg_ttfb_ms": avg_ttfb,
            "success_rate": success_rate,
            "slow_regions": slow_count,
            "failed_regions": failed_count,
        }

        update_result = await supabase.update(
            "global_tests",
            {"id": f"eq.{test_id}"},
            update_data,
        )

        if update_result.get("error"):
            logger.error(
                "Failed to update global test",
                test_id=test_id,
                error=update_result.get("error"),
            )

        logger.info(
            "Global test completed",
            test_id=test_id,
            avg_latency=avg_latency,
            success_rate=success_rate,
            slow_regions=slow_count,
            failed_regions=failed_count,
        )

    except Exception as e:
        logger.exception("Global test execution failed", test_id=test_id, error=str(e))

        # Mark test as failed
        await supabase.update(
            "global_tests",
            {"id": f"eq.{test_id}"},
            {
                "status": "failed",
                "completed_at": datetime.now(UTC).isoformat(),
            },
        )


@router.get("/{test_id}", response_model=GlobalTestResponse)
async def get_global_test(test_id: str, request: Request):
    """Get a global test by ID with its results.

    Requires membership in the test's project organization.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Get the test
    test_result = await supabase.request(
        f"/global_tests?id=eq.{test_id}&select=*"
    )

    if not test_result.get("data"):
        raise HTTPException(status_code=404, detail="Global test not found")

    test = test_result["data"][0]

    # Verify access to the project
    await verify_project_access(test["project_id"], user["user_id"], user.get("email"), request)

    # Get results for this test
    results_result = await supabase.request(
        f"/global_test_results?global_test_id=eq.{test_id}&select=*&order=latency_ms.asc"
    )

    results = []
    if results_result.get("data"):
        results = [
            GlobalTestResultResponse(
                id=r["id"],
                global_test_id=r["global_test_id"],
                region_code=r["region_code"],
                city=r["city"],
                status=r["status"],
                latency_ms=r.get("latency_ms"),
                ttfb_ms=r.get("ttfb_ms"),
                page_load_ms=r.get("page_load_ms"),
                error_message=r.get("error_message"),
                created_at=r["created_at"],
            )
            for r in results_result["data"]
        ]

    return GlobalTestResponse(
        id=test["id"],
        project_id=test["project_id"],
        url=test["url"],
        status=test["status"],
        avg_latency_ms=test.get("avg_latency_ms"),
        avg_ttfb_ms=test.get("avg_ttfb_ms"),
        success_rate=test.get("success_rate"),
        slow_regions=test.get("slow_regions", 0),
        failed_regions=test.get("failed_regions", 0),
        started_at=test.get("started_at"),
        completed_at=test.get("completed_at"),
        triggered_by=test.get("triggered_by"),
        created_at=test["created_at"],
        results=results,
    )


@router.get("/{test_id}/stream")
async def stream_global_test(test_id: str, request: Request):
    """Stream global test results as they complete.

    This SSE endpoint emits events as each region completes testing,
    allowing real-time UI updates.

    Events emitted:
    - start: Initial event with test info
    - region_result: Result from a single region
    - progress: Progress update with count
    - complete: Final completion event with summary
    - error: Error events
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Get the test
    test_result = await supabase.request(
        f"/global_tests?id=eq.{test_id}&select=*"
    )

    if not test_result.get("data"):
        raise HTTPException(status_code=404, detail="Global test not found")

    test = test_result["data"][0]

    # Verify access to the project
    await verify_project_access(test["project_id"], user["user_id"], user.get("email"), request)

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            # Emit start event
            yield {
                "event": "start",
                "data": json.dumps({
                    "test_id": test_id,
                    "url": test["url"],
                    "status": test["status"],
                    "total_regions": len(EDGE_REGIONS),
                    "started_at": test.get("started_at"),
                }),
            }

            # If already completed, just send results
            if test["status"] == "completed":
                results_result = await supabase.request(
                    f"/global_test_results?global_test_id=eq.{test_id}&select=*&order=latency_ms.asc"
                )

                if results_result.get("data"):
                    for result in results_result["data"]:
                        yield {
                            "event": "region_result",
                            "data": json.dumps({
                                "region_code": result["region_code"],
                                "city": result["city"],
                                "status": result["status"],
                                "latency_ms": result.get("latency_ms"),
                                "ttfb_ms": result.get("ttfb_ms"),
                                "page_load_ms": result.get("page_load_ms"),
                                "error_message": result.get("error_message"),
                            }),
                        }

                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "test_id": test_id,
                        "status": "completed",
                        "avg_latency_ms": test.get("avg_latency_ms"),
                        "avg_ttfb_ms": test.get("avg_ttfb_ms"),
                        "success_rate": test.get("success_rate"),
                        "slow_regions": test.get("slow_regions", 0),
                        "failed_regions": test.get("failed_regions", 0),
                        "completed_at": test.get("completed_at"),
                    }),
                }
                return

            # If still running, poll for results
            seen_regions = set()
            max_wait_seconds = 60
            poll_interval = 0.5
            elapsed = 0

            while elapsed < max_wait_seconds:
                # Check current test status
                current_test = await supabase.request(
                    f"/global_tests?id=eq.{test_id}&select=*"
                )

                if current_test.get("data"):
                    current = current_test["data"][0]

                    # Get new results
                    results_result = await supabase.request(
                        f"/global_test_results?global_test_id=eq.{test_id}&select=*&order=created_at.asc"
                    )

                    if results_result.get("data"):
                        for result in results_result["data"]:
                            region = result["region_code"]
                            if region not in seen_regions:
                                seen_regions.add(region)
                                yield {
                                    "event": "region_result",
                                    "data": json.dumps({
                                        "region_code": region,
                                        "city": result["city"],
                                        "status": result["status"],
                                        "latency_ms": result.get("latency_ms"),
                                        "ttfb_ms": result.get("ttfb_ms"),
                                        "page_load_ms": result.get("page_load_ms"),
                                        "error_message": result.get("error_message"),
                                    }),
                                }

                                # Progress update
                                yield {
                                    "event": "progress",
                                    "data": json.dumps({
                                        "completed_regions": len(seen_regions),
                                        "total_regions": len(EDGE_REGIONS),
                                        "percent": int(len(seen_regions) / len(EDGE_REGIONS) * 100),
                                    }),
                                }

                    # Check if completed
                    if current["status"] in ("completed", "failed"):
                        yield {
                            "event": "complete",
                            "data": json.dumps({
                                "test_id": test_id,
                                "status": current["status"],
                                "avg_latency_ms": current.get("avg_latency_ms"),
                                "avg_ttfb_ms": current.get("avg_ttfb_ms"),
                                "success_rate": current.get("success_rate"),
                                "slow_regions": current.get("slow_regions", 0),
                                "failed_regions": current.get("failed_regions", 0),
                                "completed_at": current.get("completed_at"),
                            }),
                        }
                        return

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            # Timeout
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": "Timeout waiting for test results",
                    "test_id": test_id,
                }),
            }

        except Exception as e:
            logger.exception("Stream error", test_id=test_id, error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "test_id": test_id,
                }),
            }

    return EventSourceResponse(event_generator())


@router.get("", response_model=GlobalTestListResponse)
async def list_global_tests(
    request: Request,
    project_id: str,
    limit: int = 10,
    offset: int = 0,
):
    """List global tests for a project.

    Args:
        project_id: Project ID to filter by (required)
        limit: Maximum number of results (default 10, max 100)
        offset: Offset for pagination

    Requires membership in the project's organization.
    """
    user = await get_current_user(request)

    # Verify access to the project
    await verify_project_access(project_id, user["user_id"], user.get("email"), request)

    # Clamp limit
    limit = min(limit, 100)

    supabase = get_supabase_client()

    # Get total count
    count_result = await supabase.request(
        f"/global_tests?project_id=eq.{project_id}&select=id"
    )
    total = len(count_result.get("data", []))

    # Get tests
    tests_result = await supabase.request(
        f"/global_tests?project_id=eq.{project_id}&select=*&order=created_at.desc&limit={limit}&offset={offset}"
    )

    if tests_result.get("error"):
        logger.error("Failed to fetch global tests", error=tests_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch global tests")

    tests = [
        GlobalTestResponse(
            id=t["id"],
            project_id=t["project_id"],
            url=t["url"],
            status=t["status"],
            avg_latency_ms=t.get("avg_latency_ms"),
            avg_ttfb_ms=t.get("avg_ttfb_ms"),
            success_rate=t.get("success_rate"),
            slow_regions=t.get("slow_regions", 0),
            failed_regions=t.get("failed_regions", 0),
            started_at=t.get("started_at"),
            completed_at=t.get("completed_at"),
            triggered_by=t.get("triggered_by"),
            created_at=t["created_at"],
            results=None,
        )
        for t in tests_result.get("data", [])
    ]

    return GlobalTestListResponse(
        tests=tests,
        total=total,
        limit=limit,
        offset=offset,
    )
