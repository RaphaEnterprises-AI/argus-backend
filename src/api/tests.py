"""Tests CRUD API endpoints.

Provides standard REST endpoints for:
- Listing tests (with optional project_id filter)
- Getting a single test by ID
- Updating a test
- Deleting a test

Note: Test creation via NLP is in server.py at /api/v1/tests/create
"""

from datetime import datetime, timezone
from typing import Optional, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import structlog

from src.services.supabase_client import get_supabase_client
from src.api.teams import get_current_user, verify_org_access, log_audit
from src.api.projects import verify_project_access

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
    target: Optional[str] = Field(None, description="Target selector or URL")
    value: Optional[str] = Field(None, description="Value to use (e.g., text to type)")
    description: Optional[str] = Field(None, description="Human-readable step description")


class TestResponse(BaseModel):
    """Test details response."""
    id: str
    project_id: str
    name: str
    description: Optional[str]
    steps: list[dict]
    tags: list[str]
    priority: PriorityType
    is_active: bool
    source: SourceType
    created_by: Optional[str]
    created_at: str
    updated_at: Optional[str]


class TestListResponse(BaseModel):
    """Test list item response."""
    id: str
    project_id: str
    name: str
    description: Optional[str]
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
    description: Optional[str] = Field(None, max_length=2000, description="Test description")
    steps: list[dict] = Field(default_factory=list, description="Test steps")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    priority: PriorityType = Field(default="medium", description="Test priority")
    is_active: bool = Field(default=True, description="Whether the test is active")
    source: SourceType = Field(default="manual", description="How the test was created")


class UpdateTestRequest(BaseModel):
    """Request to update a test."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Test name")
    description: Optional[str] = Field(None, max_length=2000, description="Test description")
    steps: Optional[list[dict]] = Field(None, description="Test steps")
    tags: Optional[list[str]] = Field(None, description="Tags for categorization")
    priority: Optional[PriorityType] = Field(None, description="Test priority")
    is_active: Optional[bool] = Field(None, description="Whether the test is active")


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


# ============================================================================
# Test Endpoints
# ============================================================================

@router.get("/tests", response_model=TestListPaginatedResponse)
async def list_tests(
    request: Request,
    project_id: Optional[str] = None,
    is_active: Optional[bool] = None,
    priority: Optional[PriorityType] = None,
    source: Optional[SourceType] = None,
    tags: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List tests with optional filters.

    Args:
        project_id: Filter by project ID (required for access control)
        is_active: Filter by active status
        priority: Filter by priority level
        source: Filter by test source
        tags: Filter by tag (comma-separated for multiple)
        search: Search in name and description
        limit: Maximum number of results (default 50, max 100)
        offset: Offset for pagination

    Returns:
        Paginated list of tests
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Limit max results
    limit = min(limit, 100)

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
        tag_list = [t.strip() for t in tags.split(",")]
        query += f"&tags=ov.{{{','.join(tag_list)}}}"

    if search:
        # Search in name or description (case-insensitive)
        query += f"&or=(name.ilike.*{search}*,description.ilike.*{search}*)"

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
    update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}

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
    is_active: Optional[bool] = Field(None, description="Set active status for all tests")
    priority: Optional[PriorityType] = Field(None, description="Set priority for all tests")
    tags_add: Optional[list[str]] = Field(None, description="Tags to add to all tests")
    tags_remove: Optional[list[str]] = Field(None, description="Tags to remove from all tests")


@router.post("/tests/bulk-delete")
async def bulk_delete_tests(body: BulkDeleteRequest, request: Request):
    """Delete multiple tests at once.

    Requires membership in each test's project organization.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    deleted = []
    failed = []

    for test_id in body.test_ids:
        try:
            test = await verify_test_access(test_id, user["user_id"], user.get("email"), request)

            delete_result = await supabase.request(
                f"/tests?id=eq.{test_id}",
                method="DELETE"
            )

            if delete_result.get("error"):
                failed.append({"id": test_id, "error": "Delete failed"})
            else:
                deleted.append(test_id)

                # Audit log
                org_id = await get_project_org_id(test["project_id"])
                await log_audit(
                    organization_id=org_id,
                    user_id=user["user_id"],
                    user_email=user.get("email"),
                    action="test.delete",
                    resource_type="test",
                    resource_id=test_id,
                    description=f"Deleted test '{test['name']}' (bulk operation)",
                    metadata={"name": test["name"], "project_id": test["project_id"], "bulk": True},
                    request=request,
                )
        except HTTPException as e:
            failed.append({"id": test_id, "error": e.detail})

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
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    updated = []
    failed = []

    for test_id in body.test_ids:
        try:
            test = await verify_test_access(test_id, user["user_id"], user.get("email"), request)

            update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}

            if body.is_active is not None:
                update_data["is_active"] = body.is_active
            if body.priority is not None:
                update_data["priority"] = body.priority

            # Handle tag modifications
            if body.tags_add or body.tags_remove:
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

                # Audit log
                org_id = await get_project_org_id(test["project_id"])
                await log_audit(
                    organization_id=org_id,
                    user_id=user["user_id"],
                    user_email=user.get("email"),
                    action="test.update",
                    resource_type="test",
                    resource_id=test_id,
                    description=f"Updated test '{test['name']}' (bulk operation)",
                    metadata={"changes": {k: v for k, v in update_data.items() if k != "updated_at"}, "bulk": True},
                    request=request,
                )
        except HTTPException as e:
            failed.append({"id": test_id, "error": e.detail})

    logger.info("Bulk update completed", updated=len(updated), failed=len(failed))

    return {
        "success": len(failed) == 0,
        "updated": updated,
        "failed": failed,
        "updated_count": len(updated),
        "failed_count": len(failed),
    }
