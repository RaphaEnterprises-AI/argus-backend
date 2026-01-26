"""
Multi-Tenant Organization API Routes

Provides tenant-scoped endpoints using the new TenantContext system.
All routes under /api/v1/orgs/{org_id} automatically validate organization access.

URL Structure:
    /api/v1/orgs                           - List user's organizations
    /api/v1/orgs/{org_id}                  - Organization details
    /api/v1/orgs/{org_id}/projects         - List projects
    /api/v1/orgs/{org_id}/projects/{id}    - Project details
    /api/v1/orgs/{org_id}/projects/{id}/tests - Tests for project
"""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.middleware.tenant import (
    TenantDep,
    ProjectTenantDep,
    get_tenant_context,
    require_tenant_context,
)
from src.api.teams import get_current_user, log_audit, verify_org_access
from src.core.tenant import TenantContext
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()

# Main router for /api/v1/orgs
router = APIRouter(prefix="/api/v1/orgs", tags=["Organizations (Multi-Tenant)"])


# =============================================================================
# Response Models
# =============================================================================

class OrgSummary(BaseModel):
    """Organization summary for list responses."""
    id: str
    name: str
    slug: str
    plan: str
    logo_url: str | None = None
    member_count: int = 0
    role: str = "member"


class OrgDetails(BaseModel):
    """Full organization details."""
    id: str
    name: str
    slug: str
    plan: str
    logo_url: str | None = None
    domain: str | None = None
    settings: dict | None = None
    features: dict | None = None
    member_count: int = 0
    ai_budget_daily: float = 1.0
    ai_budget_monthly: float = 25.0
    created_at: str
    updated_at: str | None = None


class ProjectSummary(BaseModel):
    """Project summary for list responses."""
    id: str
    org_id: str
    name: str
    description: str | None = None
    app_url: str | None = None
    repository_url: str | None = None
    is_active: bool = True
    test_count: int = 0
    created_at: str


class ProjectDetails(BaseModel):
    """Full project details."""
    id: str
    org_id: str
    name: str
    description: str | None = None
    app_url: str | None = None
    codebase_path: str | None = None
    repository_url: str | None = None
    settings: dict | None = None
    is_active: bool = True
    test_count: int = 0
    last_run_at: str | None = None
    created_at: str
    updated_at: str | None = None


class CreateProjectRequest(BaseModel):
    """Request to create a project."""
    name: str = Field(..., min_length=2, max_length=100)
    description: str | None = Field(None, max_length=500)
    app_url: str | None = None
    repository_url: str | None = None


# =============================================================================
# Organization Routes (no org_id in path - list user's orgs)
# =============================================================================

@router.get("", response_model=list[OrgSummary])
async def list_my_organizations(request: Request):
    """List all organizations the current user belongs to.

    Returns organizations with the user's role in each.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Get user's memberships
    memberships = await supabase.request(
        f"/organization_members?user_id=eq.{user['user_id']}&status=eq.active&select=organization_id,role"
    )

    if not memberships.get("data"):
        return []

    org_roles = {m["organization_id"]: m["role"] for m in memberships["data"]}
    org_ids = list(org_roles.keys())

    # Get organization details
    orgs = await supabase.request(
        f"/organizations?id=in.({','.join(org_ids)})&select=*&order=created_at.desc"
    )

    if orgs.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch organizations")

    # Build response
    result = []
    for org in orgs.get("data", []):
        result.append(OrgSummary(
            id=org["id"],
            name=org["name"],
            slug=org["slug"],
            plan=org["plan"],
            logo_url=org.get("logo_url"),
            role=org_roles.get(org["id"], "member"),
        ))

    return result


# =============================================================================
# Organization Routes (with org_id)
# =============================================================================

@router.get("/{org_id}", response_model=OrgDetails)
async def get_organization(org_id: str, request: Request):
    """Get organization details.

    Validates user has access to this organization.
    """
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(
        org_id, user["user_id"], user_email=user.get("email"), request=request
    )

    supabase = get_supabase_client()
    org = await supabase.request(f"/organizations?id=eq.{supabase_org_id}&select=*")

    if not org.get("data"):
        raise HTTPException(status_code=404, detail="Organization not found")

    org_data = org["data"][0]

    # Get member count
    members = await supabase.request(
        f"/organization_members?organization_id=eq.{supabase_org_id}&status=eq.active&select=id"
    )
    member_count = len(members.get("data", []))

    return OrgDetails(
        id=org_data["id"],
        name=org_data["name"],
        slug=org_data["slug"],
        plan=org_data["plan"],
        logo_url=org_data.get("logo_url"),
        domain=org_data.get("domain"),
        settings=org_data.get("settings"),
        features=org_data.get("features"),
        member_count=member_count,
        ai_budget_daily=float(org_data.get("ai_budget_daily", 1.0)),
        ai_budget_monthly=float(org_data.get("ai_budget_monthly", 25.0)),
        created_at=org_data["created_at"],
        updated_at=org_data.get("updated_at"),
    )


# =============================================================================
# Project Routes (scoped to organization)
# =============================================================================

@router.get("/{org_id}/projects", response_model=list[ProjectSummary])
async def list_org_projects(
    org_id: str,
    request: Request,
    is_active: bool | None = None,
    limit: int = Query(default=100, le=500),
    offset: int = 0,
):
    """List all projects for an organization.

    Validates user has access to this organization.
    """
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(
        org_id, user["user_id"], user_email=user.get("email"), request=request
    )

    supabase = get_supabase_client()

    query = f"/projects?organization_id=eq.{supabase_org_id}&select=*&order=created_at.desc"
    if is_active is not None:
        query += f"&is_active=eq.{str(is_active).lower()}"
    query += f"&limit={limit}&offset={offset}"

    projects = await supabase.request(query)

    if projects.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch projects")

    # Get test counts
    project_ids = [p["id"] for p in projects.get("data", [])]
    test_counts = {}
    if project_ids:
        for pid in project_ids:
            tests = await supabase.request(f"/tests?project_id=eq.{pid}&select=id")
            test_counts[pid] = len(tests.get("data", []))

    result = []
    for project in projects.get("data", []):
        result.append(ProjectSummary(
            id=project["id"],
            org_id=project["organization_id"],
            name=project["name"],
            description=project.get("description"),
            app_url=project.get("app_url"),
            repository_url=project.get("repository_url"),
            is_active=project.get("is_active", True),
            test_count=test_counts.get(project["id"], 0),
            created_at=project["created_at"],
        ))

    return result


@router.post("/{org_id}/projects", response_model=ProjectDetails, status_code=201)
async def create_project(
    org_id: str,
    body: CreateProjectRequest,
    request: Request,
):
    """Create a new project in the organization.

    Requires admin or owner role.
    """
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(
        org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request
    )

    supabase = get_supabase_client()

    project_data = {
        "organization_id": supabase_org_id,
        "name": body.name,
        "description": body.description,
        "app_url": body.app_url,
        "repository_url": body.repository_url,
        "settings": {},
        "is_active": True,
    }

    result = await supabase.insert("projects", project_data)

    if result.get("error"):
        logger.error("Failed to create project", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create project")

    project = result["data"][0]

    await log_audit(
        organization_id=supabase_org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="project.create",
        resource_type="project",
        resource_id=project["id"],
        description=f"Created project '{body.name}'",
        metadata={"name": body.name},
        request=request,
    )

    logger.info("Project created", org_id=supabase_org_id, project_id=project["id"])

    return ProjectDetails(
        id=project["id"],
        org_id=project["organization_id"],
        name=project["name"],
        description=project.get("description"),
        app_url=project.get("app_url"),
        repository_url=project.get("repository_url"),
        settings=project.get("settings"),
        is_active=True,
        test_count=0,
        created_at=project["created_at"],
    )


@router.get("/{org_id}/projects/{project_id}", response_model=ProjectDetails)
async def get_project(org_id: str, project_id: str, request: Request):
    """Get project details.

    Validates user has access to the organization and project exists in it.
    """
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(
        org_id, user["user_id"], user_email=user.get("email"), request=request
    )

    supabase = get_supabase_client()

    # Get project and verify it belongs to this org
    project = await supabase.request(
        f"/projects?id=eq.{project_id}&organization_id=eq.{supabase_org_id}&select=*"
    )

    if not project.get("data"):
        raise HTTPException(status_code=404, detail="Project not found in this organization")

    project_data = project["data"][0]

    # Get test count
    tests = await supabase.request(f"/tests?project_id=eq.{project_id}&select=id")
    test_count = len(tests.get("data", []))

    return ProjectDetails(
        id=project_data["id"],
        org_id=project_data["organization_id"],
        name=project_data["name"],
        description=project_data.get("description"),
        app_url=project_data.get("app_url"),
        codebase_path=project_data.get("codebase_path"),
        repository_url=project_data.get("repository_url"),
        settings=project_data.get("settings"),
        is_active=project_data.get("is_active", True),
        test_count=test_count,
        last_run_at=project_data.get("last_run_at"),
        created_at=project_data["created_at"],
        updated_at=project_data.get("updated_at"),
    )


# =============================================================================
# Tenant Context Helper
# =============================================================================

async def get_tenant_from_path(
    org_id: str,
    project_id: str | None,
    request: Request,
) -> TenantContext:
    """Build TenantContext from URL path parameters.

    This validates access and builds a proper tenant context.
    Use this in routes that need full tenant context for events.
    """
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(
        org_id, user["user_id"], user_email=user.get("email"), request=request
    )

    return TenantContext(
        org_id=supabase_org_id,
        project_id=project_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
    )
