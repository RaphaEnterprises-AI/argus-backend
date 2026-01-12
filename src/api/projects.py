"""Project Management API endpoints.

Provides endpoints for:
- Creating projects within organizations
- Listing projects (scoped to organization)
- Getting project details
- Updating projects
- Deleting projects
- Managing project settings
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field
import structlog

from src.services.supabase_client import get_supabase_client
from src.api.teams import get_current_user, verify_org_access, log_audit
from src.api.context import get_current_organization_id, require_organization_id

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["Projects"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateProjectRequest(BaseModel):
    """Request to create a new project."""
    name: str = Field(..., min_length=2, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    app_url: Optional[str] = Field(None, max_length=500)
    codebase_path: Optional[str] = Field(None, max_length=500)
    repository_url: Optional[str] = Field(None, max_length=500)
    settings: Optional[dict] = None


class UpdateProjectRequest(BaseModel):
    """Request to update a project."""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    app_url: Optional[str] = Field(None, max_length=500)
    codebase_path: Optional[str] = Field(None, max_length=500)
    repository_url: Optional[str] = Field(None, max_length=500)
    settings: Optional[dict] = None
    is_active: Optional[bool] = None


class ProjectResponse(BaseModel):
    """Project details response."""
    id: str
    organization_id: str
    name: str
    description: Optional[str]
    app_url: Optional[str]
    codebase_path: Optional[str]
    repository_url: Optional[str]
    settings: Optional[dict]
    is_active: bool
    test_count: int
    last_run_at: Optional[str]
    created_at: str
    updated_at: Optional[str]


class ProjectListResponse(BaseModel):
    """Project list item response."""
    id: str
    organization_id: str
    name: str
    description: Optional[str]
    app_url: Optional[str]
    is_active: bool
    test_count: int
    last_run_at: Optional[str]
    created_at: str


# ============================================================================
# Helper Functions
# ============================================================================

async def get_project_test_count(project_id: str) -> int:
    """Get the count of tests in a project."""
    supabase = get_supabase_client()
    tests = await supabase.request(
        f"/tests?project_id=eq.{project_id}&select=id"
    )
    return len(tests.get("data", []))


async def verify_project_access(project_id: str, user_id: str, user_email: str = None, request: Request = None) -> dict:
    """Verify user has access to the project via organization membership.

    Returns the project data if access is granted.
    """
    supabase = get_supabase_client()

    # Get project
    project_result = await supabase.request(
        f"/projects?id=eq.{project_id}&select=*"
    )

    if not project_result.get("data"):
        raise HTTPException(status_code=404, detail="Project not found")

    project = project_result["data"][0]

    # Verify user has access to the organization
    await verify_org_access(project["organization_id"], user_id, user_email=user_email, request=request)

    return project


# ============================================================================
# Organization-Scoped Project Endpoints
# ============================================================================

@router.get("/organizations/{org_id}/projects", response_model=list[ProjectListResponse])
async def list_organization_projects(
    org_id: str,
    request: Request,
    is_active: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List all projects for an organization.

    Requires membership in the organization.
    """
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

    supabase = get_supabase_client()

    # Build query
    query = f"/projects?organization_id=eq.{org_id}&select=*&order=created_at.desc"

    if is_active is not None:
        query += f"&is_active=eq.{str(is_active).lower()}"

    query += f"&limit={limit}&offset={offset}"

    projects_result = await supabase.request(query)

    if projects_result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch projects")

    projects = projects_result.get("data", [])

    # Build response with test counts
    result = []
    for project in projects:
        test_count = await get_project_test_count(project["id"])
        result.append(ProjectListResponse(
            id=project["id"],
            organization_id=project["organization_id"],
            name=project["name"],
            description=project.get("description"),
            app_url=project.get("app_url"),
            is_active=project.get("is_active", True),
            test_count=test_count,
            last_run_at=project.get("last_run_at"),
            created_at=project["created_at"],
        ))

    return result


@router.post("/organizations/{org_id}/projects", response_model=ProjectResponse)
async def create_organization_project(
    org_id: str,
    body: CreateProjectRequest,
    request: Request,
):
    """Create a new project in the organization.

    Requires admin or owner role in the organization.
    """
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Create project
    project_data = {
        "organization_id": org_id,
        "name": body.name,
        "description": body.description,
        "app_url": body.app_url,
        "codebase_path": body.codebase_path,
        "repository_url": body.repository_url,
        "settings": body.settings or {},
        "is_active": True,
    }

    project_result = await supabase.insert("projects", project_data)

    if project_result.get("error"):
        logger.error("Failed to create project", error=project_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create project")

    project = project_result["data"][0]

    # Audit log
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="project.create",
        resource_type="project",
        resource_id=project["id"],
        description=f"Created project '{body.name}'",
        metadata={"name": body.name, "app_url": body.app_url},
        request=request,
    )

    logger.info("Project created", org_id=org_id, project_id=project["id"], name=body.name)

    return ProjectResponse(
        id=project["id"],
        organization_id=project["organization_id"],
        name=project["name"],
        description=project.get("description"),
        app_url=project.get("app_url"),
        codebase_path=project.get("codebase_path"),
        repository_url=project.get("repository_url"),
        settings=project.get("settings"),
        is_active=project.get("is_active", True),
        test_count=0,
        last_run_at=None,
        created_at=project["created_at"],
        updated_at=project.get("updated_at"),
    )


# ============================================================================
# Project Endpoints (with org context from header/query)
# ============================================================================

@router.get("/projects", response_model=list[ProjectListResponse])
async def list_projects(
    request: Request,
    is_active: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List projects for the current organization context.

    Organization is determined from:
    1. X-Organization-ID header
    2. org_id query parameter
    3. User's default organization
    """
    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    if not org_id:
        # Fall back to listing all projects the user has access to
        supabase = get_supabase_client()

        # Get user's organizations
        memberships = await supabase.request(
            f"/organization_members?user_id=eq.{user['user_id']}&status=eq.active&select=organization_id"
        )

        if not memberships.get("data"):
            return []

        org_ids = [m["organization_id"] for m in memberships["data"]]

        # Get projects from all user's organizations
        query = f"/projects?organization_id=in.({','.join(org_ids)})&select=*&order=created_at.desc"

        if is_active is not None:
            query += f"&is_active=eq.{str(is_active).lower()}"

        query += f"&limit={limit}&offset={offset}"

        projects_result = await supabase.request(query)

        if projects_result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to fetch projects")

        projects = projects_result.get("data", [])

        result = []
        for project in projects:
            test_count = await get_project_test_count(project["id"])
            result.append(ProjectListResponse(
                id=project["id"],
                organization_id=project["organization_id"],
                name=project["name"],
                description=project.get("description"),
                app_url=project.get("app_url"),
                is_active=project.get("is_active", True),
                test_count=test_count,
                last_run_at=project.get("last_run_at"),
                created_at=project["created_at"],
            ))

        return result

    # Specific organization context provided
    return await list_organization_projects(org_id, request, is_active, limit, offset)


@router.post("/projects", response_model=ProjectResponse)
async def create_project(body: CreateProjectRequest, request: Request):
    """Create a new project in the current organization context.

    Organization is determined from:
    1. X-Organization-ID header
    2. org_id query parameter
    3. User's default organization

    Requires admin or owner role in the organization.
    """
    org_id = await require_organization_id(request)
    return await create_organization_project(org_id, body, request)


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, request: Request):
    """Get project details.

    Requires membership in the project's organization.
    """
    user = await get_current_user(request)
    project = await verify_project_access(project_id, user["user_id"], user.get("email"), request=request)

    test_count = await get_project_test_count(project_id)

    return ProjectResponse(
        id=project["id"],
        organization_id=project["organization_id"],
        name=project["name"],
        description=project.get("description"),
        app_url=project.get("app_url"),
        codebase_path=project.get("codebase_path"),
        repository_url=project.get("repository_url"),
        settings=project.get("settings"),
        is_active=project.get("is_active", True),
        test_count=test_count,
        last_run_at=project.get("last_run_at"),
        created_at=project["created_at"],
        updated_at=project.get("updated_at"),
    )


@router.put("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, body: UpdateProjectRequest, request: Request):
    """Update a project.

    Requires admin or owner role in the project's organization.
    """
    user = await get_current_user(request)
    project = await verify_project_access(project_id, user["user_id"], user.get("email"), request=request)

    # Verify admin/owner access
    await verify_org_access(project["organization_id"], user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Build update data
    update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}

    if body.name is not None:
        update_data["name"] = body.name
    if body.description is not None:
        update_data["description"] = body.description
    if body.app_url is not None:
        update_data["app_url"] = body.app_url
    if body.codebase_path is not None:
        update_data["codebase_path"] = body.codebase_path
    if body.repository_url is not None:
        update_data["repository_url"] = body.repository_url
    if body.settings is not None:
        update_data["settings"] = body.settings
    if body.is_active is not None:
        update_data["is_active"] = body.is_active

    result = await supabase.update("projects", {"id": f"eq.{project_id}"}, update_data)

    if result.get("error"):
        logger.error("Failed to update project", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to update project")

    # Audit log
    await log_audit(
        organization_id=project["organization_id"],
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="project.update",
        resource_type="project",
        resource_id=project_id,
        description=f"Updated project '{project['name']}'",
        metadata={"changes": {k: v for k, v in update_data.items() if k != "updated_at"}},
        request=request,
    )

    logger.info("Project updated", project_id=project_id)

    return await get_project(project_id, request)


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str, request: Request):
    """Delete a project.

    Requires owner role in the project's organization.
    This permanently deletes the project and all associated data.
    """
    user = await get_current_user(request)
    project = await verify_project_access(project_id, user["user_id"], user.get("email"), request=request)

    # Verify owner access
    await verify_org_access(project["organization_id"], user["user_id"], ["owner"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Delete project
    delete_result = await supabase.request(
        f"/projects?id=eq.{project_id}",
        method="DELETE"
    )

    if delete_result.get("error"):
        logger.error("Failed to delete project", error=delete_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to delete project")

    # Audit log
    await log_audit(
        organization_id=project["organization_id"],
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="project.delete",
        resource_type="project",
        resource_id=project_id,
        description=f"Deleted project '{project['name']}'",
        metadata={"name": project["name"]},
        request=request,
    )

    logger.info("Project deleted", project_id=project_id, name=project["name"])

    return {"success": True, "message": f"Project '{project['name']}' has been deleted"}
