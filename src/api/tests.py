"""Tests CRUD API endpoints.

Provides standard REST endpoints for:
- Listing tests (with optional project_id filter)
- Getting a single test by ID
- Updating a test
- Deleting a test

Note: Test creation via NLP is in server.py at /api/v1/tests/create
"""

import urllib.parse
from datetime import UTC, datetime
from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.projects import verify_project_access
from src.api.teams import get_current_user, log_audit
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["Tests"])


# ============================================================================
# Request/Response Models
# ============================================================================

PriorityType = Literal["critical", "high", "medium", "low"]
SourceType = Literal["manual", "discovered", "generated", "imported"]


class TestStep(BaseModel):
    """A single test step."""
    action: str = Field(..., description="Action to perform (e.g., 'click', 'type', 'navigate')")
    target: str | None = Field(None, description="Target selector or URL")
    value: str | None = Field(None, description="Value to use (e.g., text to type)")
    description: str | None = Field(None, description="Human-readable step description")


class TestResponse(BaseModel):
    """Test details response."""
    id: str
    project_id: str
    name: str
    description: str | None
    steps: list[dict]
    tags: list[str]
    priority: PriorityType
    is_active: bool
    source: SourceType
    created_by: str | None
    created_at: str
    updated_at: str | None


class TestListResponse(BaseModel):
    """Test list item response."""
    id: str
    project_id: str
    name: str
    description: str | None
    tags: list[str]
    priority: PriorityType
    is_active: bool
    source: SourceType
    step_count: int
    created_at: str


class CreateTestRequest(BaseModel):
    """Request to create a new test."""
    project_id: str = Field(..., description="Project ID to associate the test with")
    name: str = Field(..., min_length=1, max_length=255, description="Test name")
    description: str | None = Field(None, max_length=2000, description="Test description")
    steps: list[dict] = Field(default_factory=list, description="Test steps")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    priority: PriorityType = Field(default="medium", description="Test priority")
    is_active: bool = Field(default=True, description="Whether the test is active")
    source: SourceType = Field(default="manual", description="How the test was created")


class UpdateTestRequest(BaseModel):
    """Request to update a test."""
    name: str | None = Field(None, min_length=1, max_length=255, description="Test name")
    description: str | None = Field(None, max_length=2000, description="Test description")
    steps: list[dict] | None = Field(None, description="Test steps")
    tags: list[str] | None = Field(None, description="Tags for categorization")
    priority: PriorityType | None = Field(None, description="Test priority")
    is_active: bool | None = Field(None, description="Whether the test is active")


class TestListPaginatedResponse(BaseModel):
    """Paginated test list response."""
    tests: list[TestListResponse]
    total: int
    limit: int
    offset: int


# ============================================================================
# Helper Functions
# ============================================================================

async def verify_test_access(test_id: str, user_id: str, user_email: str = None, request: Request = None) -> dict:
    """Verify user has access to the test via project organization membership.

    Returns the test data if access is granted.
    """
    supabase = get_supabase_client()

    # Get test
    test_result = await supabase.request(
        f"/tests?id=eq.{test_id}&select=*"
    )

    if not test_result.get("data"):
        raise HTTPException(status_code=404, detail="Test not found")

    test = test_result["data"][0]

    # Verify user has access to the project (which checks org membership)
    await verify_project_access(test["project_id"], user_id, user_email, request)

    return test


async def get_project_org_id(project_id: str) -> str:
    """Get the organization ID for a project."""
    supabase = get_supabase_client()

    project_result = await supabase.request(
        f"/projects?id=eq.{project_id}&select=organization_id"
    )

    if not project_result.get("data"):
        raise HTTPException(status_code=404, detail="Project not found")

    return project_result["data"][0]["organization_id"]


async def get_project_org_ids_batch(project_ids: list[str]) -> dict[str, str]:
    """Get organization IDs for multiple projects in a single query.

    Uses Supabase RPC function to avoid N+1 query problem.
    Returns a dict mapping project_id -> organization_id.
    """
    if not project_ids:
        return {}

    supabase = get_supabase_client()
    result = await supabase.rpc("get_project_org_ids", {"project_ids": project_ids})

    if result.get("error"):
        logger.warning("Batch project org_id query failed, falling back to direct query", error=result.get("error"))
        # Fallback to direct query
        query_result = await supabase.request(
            f"/projects?id=in.({','.join(project_ids)})&select=id,organization_id"
        )
        return {p["id"]: p["organization_id"] for p in query_result.get("data", [])}

    return {str(row["project_id"]): str(row["organization_id"]) for row in result.get("data", [])}


async def batch_verify_test_access(
    test_ids: list[str],
    user_id: str,
    user_email: str = None
) -> tuple[dict[str, dict], dict[str, str]]:
    """Verify user access to multiple tests in batch.

    Returns:
        - accessible_tests: dict mapping test_id -> test data (tests user can access)
        - failed_tests: dict mapping test_id -> error message (tests user cannot access)

    This function batches database queries to avoid N+1 problems:
    1. Fetch all tests in a single query
    2. Get all unique project_ids and their org_ids in one query
    3. Get user's memberships for all relevant orgs in one query
    4. Filter based on membership
    """
    supabase = get_supabase_client()
    accessible_tests = {}
    failed_tests = {}

    if not test_ids:
        return accessible_tests, failed_tests

    # Step 1: Fetch all tests in a single query
    tests_result = await supabase.request(
        f"/tests?id=in.({','.join(test_ids)})&select=*"
    )

    if tests_result.get("error"):
        for test_id in test_ids:
            failed_tests[test_id] = "Failed to fetch tests"
        return accessible_tests, failed_tests

    tests_by_id = {t["id"]: t for t in tests_result.get("data", [])}

    # Mark missing tests as failed
    for test_id in test_ids:
        if test_id not in tests_by_id:
            failed_tests[test_id] = "Test not found"

    if not tests_by_id:
        return accessible_tests, failed_tests

    # Step 2: Get all unique project_ids and their org_ids
    project_ids = list(set(t["project_id"] for t in tests_by_id.values()))
    project_org_map = await get_project_org_ids_batch(project_ids)

    # Step 3: Get user's memberships for all relevant orgs in one query
    org_ids = list(set(project_org_map.values()))
    if not org_ids:
        for test_id in tests_by_id:
            failed_tests[test_id] = "Project organization not found"
        return accessible_tests, failed_tests

    membership_result = await supabase.request(
        f"/organization_members?organization_id=in.({','.join(org_ids)})&user_id=eq.{user_id}&status=eq.active&select=organization_id"
    )

    user_org_ids = set()
    if membership_result.get("data"):
        user_org_ids = {m["organization_id"] for m in membership_result["data"]}

    # Step 4: Filter tests based on membership
    for test_id, test in tests_by_id.items():
        project_id = test["project_id"]
        org_id = project_org_map.get(project_id)

        if not org_id:
            failed_tests[test_id] = "Project organization not found"
        elif org_id not in user_org_ids:
            failed_tests[test_id] = "Access denied - not a member of project organization"
        else:
            accessible_tests[test_id] = test

    return accessible_tests, failed_tests


async def batch_insert_audit_logs(audit_entries: list[dict]) -> None:
    """Insert multiple audit log entries in a single batch operation."""
    if not audit_entries:
        return

    supabase = get_supabase_client()
    result = await supabase.insert("audit_logs", audit_entries)

    if result.get("error"):
        logger.warning("Batch audit log insert failed", error=result.get("error"), count=len(audit_entries))


# ============================================================================
# Test Endpoints
# ============================================================================

@router.get("/tests", response_model=TestListPaginatedResponse)
async def list_tests(
    request: Request,
    project_id: str | None = None,
    is_active: bool | None = None,
    priority: PriorityType | None = None,
    source: SourceType | None = None,
    tags: Annotated[str | None, Query(max_length=500, description="Tags to filter by (comma-separated)")] = None,
    search: Annotated[str | None, Query(max_length=200, description="Search term for name/description")] = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Maximum results to return")] = 50,
    offset: Annotated[int, Query(ge=0, le=10000, description="Offset for pagination")] = 0,
):
    """List tests with optional filters.

    Args:
        project_id: Filter by project ID (required for access control)
        is_active: Filter by active status
        priority: Filter by priority level
        source: Filter by test source
        tags: Filter by tag (comma-separated for multiple, max 500 chars)
        search: Search in name and description (max 200 chars)
        limit: Maximum number of results (default 50, max 500)
        offset: Offset for pagination (max 10000)

    Returns:
        Paginated list of tests
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Note: limit and offset are validated by FastAPI Query constraints
    # limit: 1-500, offset: 0-10000

    if project_id:
        # Verify access to the specific project
        await verify_project_access(project_id, user["user_id"], user.get("email"), request)

        # Build query for specific project
        query = f"/tests?project_id=eq.{project_id}&select=*&order=created_at.desc"
    else:
        # Get tests from all projects the user has access to via organizations
        memberships = await supabase.request(
            f"/organization_members?user_id=eq.{user['user_id']}&status=eq.active&select=organization_id"
        )

        if not memberships.get("data"):
            return TestListPaginatedResponse(tests=[], total=0, limit=limit, offset=offset)

        org_ids = [m["organization_id"] for m in memberships["data"]]

        # Get all projects from user's organizations
        projects_result = await supabase.request(
            f"/projects?organization_id=in.({','.join(org_ids)})&select=id"
        )

        if not projects_result.get("data"):
            return TestListPaginatedResponse(tests=[], total=0, limit=limit, offset=offset)

        project_ids = [p["id"] for p in projects_result["data"]]
        query = f"/tests?project_id=in.({','.join(project_ids)})&select=*&order=created_at.desc"

    # Apply filters
    if is_active is not None:
        query += f"&is_active=eq.{str(is_active).lower()}"

    if priority:
        query += f"&priority=eq.{priority}"

    if source:
        query += f"&source=eq.{source}"

    if tags:
        # Filter tests that contain any of the specified tags
        # URL-encode each tag to prevent injection
        tag_list = [urllib.parse.quote(t.strip(), safe='') for t in tags.split(",")]
        query += f"&tags=ov.{{{','.join(tag_list)}}}"

    if search:
        # Search in name or description (case-insensitive)
        # URL-encode the search term to prevent PostgREST injection
        safe_search = urllib.parse.quote(search, safe='')
        query += f"&or=(name.ilike.*{safe_search}*,description.ilike.*{safe_search}*)"

    # Get total count first (without pagination)
    count_query = query.replace("&select=*", "&select=id")
    count_result = await supabase.request(count_query)
    total = len(count_result.get("data", []))

    # Apply pagination
    query += f"&limit={limit}&offset={offset}"

    tests_result = await supabase.request(query)

    if tests_result.get("error"):
        logger.error("Failed to fetch tests", error=tests_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch tests")

    tests = tests_result.get("data", [])

    result = [
        TestListResponse(
            id=test["id"],
            project_id=test["project_id"],
            name=test["name"],
            description=test.get("description"),
            tags=test.get("tags", []) or [],
            priority=test.get("priority", "medium"),
            is_active=test.get("is_active", True),
            source=test.get("source", "manual"),
            step_count=len(test.get("steps", []) or []),
            created_at=test["created_at"],
        )
        for test in tests
    ]

    return TestListPaginatedResponse(
        tests=result,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/tests", response_model=TestResponse)
async def create_test(body: CreateTestRequest, request: Request):
    """Create a new test.

    Requires membership in the project's organization.
    """
    user = await get_current_user(request)

    # Verify access to the project
    await verify_project_access(body.project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Create test
    test_data = {
        "project_id": body.project_id,
        "name": body.name,
        "description": body.description,
        "steps": body.steps,
        "tags": body.tags,
        "priority": body.priority,
        "is_active": body.is_active,
        "source": body.source,
        "created_by": user["user_id"],
    }

    result = await supabase.insert("tests", test_data)

    if result.get("error"):
        logger.error("Failed to create test", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create test")

    test = result["data"][0]

    # Audit log
    org_id = await get_project_org_id(body.project_id)
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="test.create",
        resource_type="test",
        resource_id=test["id"],
        description=f"Created test '{body.name}'",
        metadata={"name": body.name, "project_id": body.project_id},
        request=request,
    )

    logger.info("Test created", test_id=test["id"], name=body.name, project_id=body.project_id)

    # Emit TEST_CREATED event to Redpanda for downstream processing
    try:
        from src.services.event_gateway import emit_test_created, get_event_gateway
        event_gateway = get_event_gateway()
        if event_gateway.is_running:
            await emit_test_created(
                test_id=test["id"],
                test_name=body.name,
                test_type="manual",  # API-created tests default to manual type
                org_id=org_id,
                project_id=body.project_id,
                user_id=user["user_id"],
                source=body.source,
                priority=body.priority,
                tags=body.tags or [],
                steps_count=len(body.steps) if body.steps else 0,
            )
    except Exception as e:
        # Event emission is non-critical - log but don't fail the request
        logger.warning("Failed to emit TEST_CREATED event", error=str(e), test_id=test["id"])

    return TestResponse(
        id=test["id"],
        project_id=test["project_id"],
        name=test["name"],
        description=test.get("description"),
        steps=test.get("steps", []) or [],
        tags=test.get("tags", []) or [],
        priority=test.get("priority", "medium"),
        is_active=test.get("is_active", True),
        source=test.get("source", "manual"),
        created_by=test.get("created_by"),
        created_at=test["created_at"],
        updated_at=test.get("updated_at"),
    )


@router.get("/tests/{test_id}", response_model=TestResponse)
async def get_test(test_id: str, request: Request):
    """Get a single test by ID.

    Requires membership in the test's project organization.
    """
    user = await get_current_user(request)
    test = await verify_test_access(test_id, user["user_id"], user.get("email"), request)

    return TestResponse(
        id=test["id"],
        project_id=test["project_id"],
        name=test["name"],
        description=test.get("description"),
        steps=test.get("steps", []) or [],
        tags=test.get("tags", []) or [],
        priority=test.get("priority", "medium"),
        is_active=test.get("is_active", True),
        source=test.get("source", "manual"),
        created_by=test.get("created_by"),
        created_at=test["created_at"],
        updated_at=test.get("updated_at"),
    )


@router.put("/tests/{test_id}", response_model=TestResponse)
@router.patch("/tests/{test_id}", response_model=TestResponse)
async def update_test(test_id: str, body: UpdateTestRequest, request: Request):
    """Update a test.

    PUT: Full update (all fields can be provided)
    PATCH: Partial update (only provided fields are updated)

    Both methods support partial updates since all fields are optional.
    Requires membership in the test's project organization.
    """
    user = await get_current_user(request)
    test = await verify_test_access(test_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Build update data
    update_data = {"updated_at": datetime.now(UTC).isoformat()}

    if body.name is not None:
        update_data["name"] = body.name
    if body.description is not None:
        update_data["description"] = body.description
    if body.steps is not None:
        update_data["steps"] = body.steps
    if body.tags is not None:
        update_data["tags"] = body.tags
    if body.priority is not None:
        update_data["priority"] = body.priority
    if body.is_active is not None:
        update_data["is_active"] = body.is_active

    result = await supabase.update("tests", {"id": f"eq.{test_id}"}, update_data)

    if result.get("error"):
        logger.error("Failed to update test", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to update test")

    # Audit log
    org_id = await get_project_org_id(test["project_id"])
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="test.update",
        resource_type="test",
        resource_id=test_id,
        description=f"Updated test '{test['name']}'",
        metadata={"changes": {k: v for k, v in update_data.items() if k != "updated_at"}},
        request=request,
    )

    logger.info("Test updated", test_id=test_id)

    # Return updated test
    return await get_test(test_id, request)


@router.delete("/tests/{test_id}")
async def delete_test(test_id: str, request: Request):
    """Delete a test.

    Requires membership in the test's project organization.
    """
    user = await get_current_user(request)
    test = await verify_test_access(test_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Delete test
    delete_result = await supabase.request(
        f"/tests?id=eq.{test_id}",
        method="DELETE"
    )

    if delete_result.get("error"):
        logger.error("Failed to delete test", error=delete_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to delete test")

    # Audit log
    org_id = await get_project_org_id(test["project_id"])
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="test.delete",
        resource_type="test",
        resource_id=test_id,
        description=f"Deleted test '{test['name']}'",
        metadata={"name": test["name"], "project_id": test["project_id"]},
        request=request,
    )

    logger.info("Test deleted", test_id=test_id, name=test["name"])

    return {"success": True, "message": f"Test '{test['name']}' has been deleted"}


# ============================================================================
# Bulk Operations
# ============================================================================

class BulkDeleteRequest(BaseModel):
    """Request for bulk delete operation."""
    test_ids: list[str] = Field(..., min_length=1, max_length=100, description="Test IDs to delete")


class BulkUpdateRequest(BaseModel):
    """Request for bulk update operation."""
    test_ids: list[str] = Field(..., min_length=1, max_length=100, description="Test IDs to update")
    is_active: bool | None = Field(None, description="Set active status for all tests")
    priority: PriorityType | None = Field(None, description="Set priority for all tests")
    tags_add: list[str] | None = Field(None, description="Tags to add to all tests")
    tags_remove: list[str] | None = Field(None, description="Tags to remove from all tests")


@router.post("/tests/bulk-delete")
async def bulk_delete_tests(body: BulkDeleteRequest, request: Request):
    """Delete multiple tests at once.

    Requires membership in each test's project organization.

    Optimized to use batch queries:
    - Single query to fetch all tests
    - Single query to verify all org memberships
    - Single DELETE query for all accessible tests
    - Single batch INSERT for audit logs
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Step 1: Batch verify access to all tests (3 queries total instead of N*2)
    accessible_tests, access_failures = await batch_verify_test_access(
        body.test_ids, user["user_id"], user.get("email")
    )

    # Build failed list from access failures
    failed = [{"id": test_id, "error": error} for test_id, error in access_failures.items()]

    if not accessible_tests:
        return {
            "success": len(failed) == 0,
            "deleted": [],
            "failed": failed,
            "deleted_count": 0,
            "failed_count": len(failed),
        }

    # Step 2: Single DELETE query for all accessible tests
    accessible_ids = list(accessible_tests.keys())
    delete_result = await supabase.request(
        f"/tests?id=in.({','.join(accessible_ids)})",
        method="DELETE"
    )

    deleted = []
    if delete_result.get("error"):
        # If batch delete failed, mark all as failed
        for test_id in accessible_ids:
            failed.append({"id": test_id, "error": "Delete failed"})
    else:
        deleted = accessible_ids

    # Step 3: Batch get org_ids for audit logs (1 query instead of N)
    if deleted:
        project_ids = list(set(accessible_tests[tid]["project_id"] for tid in deleted))
        project_org_map = await get_project_org_ids_batch(project_ids)

        # Step 4: Build and batch insert audit logs (1 query instead of N)
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        now = datetime.now(UTC).isoformat()

        audit_entries = []
        for test_id in deleted:
            test = accessible_tests[test_id]
            org_id = project_org_map.get(test["project_id"])
            if org_id:
                audit_entries.append({
                    "organization_id": org_id,
                    "user_id": user["user_id"],
                    "user_email": user.get("email"),
                    "action": "test.delete",
                    "resource_type": "test",
                    "resource_id": test_id,
                    "description": f"Deleted test '{test['name']}' (bulk operation)",
                    "metadata": {"name": test["name"], "project_id": test["project_id"], "bulk": True},
                    "ip_address": client_ip,
                    "user_agent": user_agent,
                    "created_at": now,
                })

        await batch_insert_audit_logs(audit_entries)

    logger.info("Bulk delete completed", deleted=len(deleted), failed=len(failed))

    return {
        "success": len(failed) == 0,
        "deleted": deleted,
        "failed": failed,
        "deleted_count": len(deleted),
        "failed_count": len(failed),
    }


@router.post("/tests/bulk-update")
async def bulk_update_tests(body: BulkUpdateRequest, request: Request):
    """Update multiple tests at once.

    Requires membership in each test's project organization.

    Optimized to use batch queries:
    - Single query to fetch all tests
    - Single query to verify all org memberships
    - Single UPDATE query when no tag modifications (or per-test for tags)
    - Single batch INSERT for audit logs
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Step 1: Batch verify access to all tests (3 queries total instead of N*2)
    accessible_tests, access_failures = await batch_verify_test_access(
        body.test_ids, user["user_id"], user.get("email")
    )

    # Build failed list from access failures
    failed = [{"id": test_id, "error": error} for test_id, error in access_failures.items()]

    if not accessible_tests:
        return {
            "success": len(failed) == 0,
            "updated": [],
            "failed": failed,
            "updated_count": 0,
            "failed_count": len(failed),
        }

    accessible_ids = list(accessible_tests.keys())
    now = datetime.now(UTC).isoformat()
    updated = []
    update_metadata = {}  # Track changes for audit log

    # Step 2: Perform updates
    # If no tag modifications, we can do a single bulk UPDATE
    if not body.tags_add and not body.tags_remove:
        update_data = {"updated_at": now}
        if body.is_active is not None:
            update_data["is_active"] = body.is_active
        if body.priority is not None:
            update_data["priority"] = body.priority

        result = await supabase.request(
            f"/tests?id=in.({','.join(accessible_ids)})",
            method="PATCH",
            body=update_data
        )

        if result.get("error"):
            for test_id in accessible_ids:
                failed.append({"id": test_id, "error": "Update failed"})
        else:
            updated = accessible_ids
            # Store the same update metadata for all tests
            changes = {k: v for k, v in update_data.items() if k != "updated_at"}
            for test_id in updated:
                update_metadata[test_id] = changes
    else:
        # Tag modifications require per-test updates (each test has different existing tags)
        for test_id in accessible_ids:
            test = accessible_tests[test_id]
            update_data = {"updated_at": now}

            if body.is_active is not None:
                update_data["is_active"] = body.is_active
            if body.priority is not None:
                update_data["priority"] = body.priority

            # Handle tag modifications
            current_tags = set(test.get("tags", []) or [])
            if body.tags_add:
                current_tags.update(body.tags_add)
            if body.tags_remove:
                current_tags -= set(body.tags_remove)
            update_data["tags"] = list(current_tags)

            result = await supabase.update("tests", {"id": f"eq.{test_id}"}, update_data)

            if result.get("error"):
                failed.append({"id": test_id, "error": "Update failed"})
            else:
                updated.append(test_id)
                update_metadata[test_id] = {k: v for k, v in update_data.items() if k != "updated_at"}

    # Step 3: Batch get org_ids for audit logs (1 query instead of N)
    if updated:
        project_ids = list(set(accessible_tests[tid]["project_id"] for tid in updated))
        project_org_map = await get_project_org_ids_batch(project_ids)

        # Step 4: Build and batch insert audit logs (1 query instead of N)
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

        audit_entries = []
        for test_id in updated:
            test = accessible_tests[test_id]
            org_id = project_org_map.get(test["project_id"])
            if org_id:
                audit_entries.append({
                    "organization_id": org_id,
                    "user_id": user["user_id"],
                    "user_email": user.get("email"),
                    "action": "test.update",
                    "resource_type": "test",
                    "resource_id": test_id,
                    "description": f"Updated test '{test['name']}' (bulk operation)",
                    "metadata": {"changes": update_metadata.get(test_id, {}), "bulk": True},
                    "ip_address": client_ip,
                    "user_agent": user_agent,
                    "created_at": now,
                })

        await batch_insert_audit_logs(audit_entries)

    logger.info("Bulk update completed", updated=len(updated), failed=len(failed))

    return {
        "success": len(failed) == 0,
        "updated": updated,
        "failed": failed,
        "updated_count": len(updated),
        "failed_count": len(failed),
    }
