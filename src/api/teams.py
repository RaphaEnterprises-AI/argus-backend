"""Team Management API endpoints.

Provides endpoints for:
- Organization management
- Team member invitations and roles
- RBAC operations
"""

import hashlib
import re
import secrets
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, EmailStr
import structlog

from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/teams", tags=["Team Management"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateOrganizationRequest(BaseModel):
    """Request to create a new organization."""
    name: str = Field(..., min_length=2, max_length=100)
    slug: Optional[str] = Field(None, min_length=2, max_length=50, pattern="^[a-z0-9-]+$")


class UpdateOrganizationRequest(BaseModel):
    """Request to update organization settings."""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    ai_budget_daily: Optional[float] = Field(None, ge=0, le=10000)
    ai_budget_monthly: Optional[float] = Field(None, ge=0, le=100000)
    settings: Optional[dict] = None


class InviteMemberRequest(BaseModel):
    """Request to invite a member to the organization."""
    email: EmailStr
    role: str = Field("member", pattern="^(admin|member|viewer)$")


class UpdateMemberRoleRequest(BaseModel):
    """Request to update a member's role."""
    role: str = Field(..., pattern="^(admin|member|viewer)$")


class OrganizationResponse(BaseModel):
    """Organization details response."""
    id: str
    name: str
    slug: str
    plan: str
    ai_budget_daily: float
    ai_budget_monthly: float
    ai_spend_today: float
    ai_spend_this_month: float
    features: dict
    member_count: int
    created_at: str


class MemberResponse(BaseModel):
    """Organization member response."""
    id: str
    user_id: str
    email: str
    role: str
    status: str
    invited_at: Optional[str]
    accepted_at: Optional[str]
    created_at: str


# ============================================================================
# Helper Functions
# ============================================================================

async def get_current_user(request: Request) -> dict:
    """Extract current user from request state (set by auth middleware).

    The AuthenticationMiddleware stores authenticated user info in request.state.user
    as a UserContext object. This function extracts it for use in route handlers.
    """
    # Check if user was authenticated by middleware
    if hasattr(request.state, "user") and request.state.user:
        user = request.state.user
        # Handle both UserContext objects and dict-like objects
        if hasattr(user, "user_id"):
            return {
                "user_id": user.user_id,
                "email": getattr(user, "email", None),
                "organization_id": getattr(user, "organization_id", None),
                "roles": getattr(user, "roles", []),
            }
        elif isinstance(user, dict):
            return {
                "user_id": user.get("user_id"),
                "email": user.get("email"),
                "organization_id": user.get("organization_id"),
                "roles": user.get("roles", []),
            }

    # SECURITY: No header fallback - authentication must come from middleware
    # The old "legacy support" for x-user-id headers was an auth bypass vulnerability
    raise HTTPException(status_code=401, detail="Authentication required")


def is_clerk_org_id(org_id: str) -> bool:
    """Check if an organization ID is from Clerk (format: org_xxx)."""
    return org_id and org_id.startswith("org_")


async def ensure_clerk_org_synced(
    clerk_org_id: str,
    user_id: str,
    user_email: str = None
) -> str:
    """Ensure a Clerk organization is synced to Supabase.

    If the Clerk org doesn't exist in Supabase, auto-create it and add
    the current user as owner. Returns the Supabase organization ID
    (a UUID, not the Clerk org ID).

    This bridges Clerk's organization system with Supabase's storage.
    """
    supabase = get_supabase_client()

    # Check if org already exists in Supabase (by clerk_org_id)
    existing = await supabase.request(
        f"/organizations?clerk_org_id=eq.{clerk_org_id}&select=id"
    )

    if existing.get("data") and len(existing["data"]) > 0:
        # Org already exists, return its Supabase UUID
        return existing["data"][0]["id"]

    # Create new organization synced from Clerk
    # Generate a slug from the clerk_org_id (remove prefix, lowercase)
    slug_base = clerk_org_id.lower().replace("org_", "").replace("_", "-")
    slug = f"clerk-{slug_base[:20]}"  # Keep it short

    # Create the organization (Supabase will generate a UUID for id)
    org_result = await supabase.insert("organizations", {
        "name": "My Organization",  # Default name, user can update later
        "slug": slug,
        "plan": "free",
        "clerk_org_id": clerk_org_id,  # Store Clerk reference for lookups
    })

    if org_result.get("error"):
        # If insert fails (e.g., duplicate slug), try with a unique slug
        unique_slug = f"{slug}-{secrets.token_hex(4)}"
        org_result = await supabase.insert("organizations", {
            "name": "My Organization",
            "slug": unique_slug,
            "plan": "free",
            "clerk_org_id": clerk_org_id,
        })

        if org_result.get("error"):
            logger.error("Failed to create Clerk org in Supabase",
                        clerk_org_id=clerk_org_id,
                        error=org_result.get("error"))
            raise HTTPException(
                status_code=500,
                detail="Failed to sync organization from Clerk"
            )

    org = org_result["data"][0]

    # Add the current user as owner
    member_result = await supabase.insert("organization_members", {
        "organization_id": org["id"],
        "user_id": user_id,
        "email": user_email or "",
        "role": "owner",
        "status": "active",
        "accepted_at": datetime.now(timezone.utc).isoformat(),
    })

    if member_result.get("error"):
        logger.warning("Failed to add owner to synced org",
                      org_id=org["id"],
                      error=member_result.get("error"))

    # Create default healing config (ignore errors - not critical)
    try:
        await supabase.insert("self_healing_config", {
            "organization_id": org["id"],
        })
    except Exception as e:
        logger.warning("Failed to create healing config for synced org",
                      org_id=org["id"], error=str(e))

    logger.info("Synced Clerk organization to Supabase",
               clerk_org_id=clerk_org_id,
               supabase_org_id=org["id"],
               user_id=user_id)

    return org["id"]


async def translate_clerk_org_id(org_id: str, user_id: str = None, user_email: str = None) -> str:
    """Translate a Clerk organization ID to its Supabase UUID.

    If the org_id is already a Supabase UUID (not starting with 'org_'),
    returns it unchanged. If it's a Clerk org ID, looks up or creates
    the corresponding Supabase organization and returns its UUID.

    Args:
        org_id: The organization ID (Clerk format org_xxx or Supabase UUID)
        user_id: User ID for syncing new orgs (optional)
        user_email: User email for syncing new orgs (optional)

    Returns:
        The Supabase organization UUID
    """
    if not is_clerk_org_id(org_id):
        return org_id

    # Look up existing mapping
    supabase = get_supabase_client()
    result = await supabase.request(
        f"/organizations?clerk_org_id=eq.{org_id}&select=id"
    )

    if result.get("data") and len(result["data"]) > 0:
        return result["data"][0]["id"]

    # If no mapping exists and we have user info, sync the org
    if user_id:
        return await ensure_clerk_org_synced(org_id, user_id, user_email)

    # No mapping and no user info - can't proceed
    logger.error("Clerk org not synced and no user info to sync", clerk_org_id=org_id)
    raise HTTPException(
        status_code=404,
        detail=f"Organization {org_id} not found"
    )


async def verify_org_access(
    organization_id: str,
    user_id: str,
    required_roles: list[str] = None,
    user_email: str = None,
    request: "Request" = None
) -> tuple[dict, str]:
    """Verify user has access to the organization with required role.

    Checks membership by user_id first, then falls back to email.
    For API key authentication, trusts the API key's organization_id claim.

    For Clerk organizations (org_xxx format), auto-syncs to Supabase if needed.

    Returns:
        tuple[dict, str]: A tuple of (member_record, translated_org_id) where
        translated_org_id is the Supabase UUID (translated from Clerk org ID if needed).
    """
    # Store original for reference
    original_org_id = organization_id

    # Handle API key authentication - trust the API key's organization_id
    if request and hasattr(request.state, "user") and request.state.user:
        user = request.state.user
        auth_method = getattr(user, "auth_method", None)
        user_org_id = getattr(user, "organization_id", None)

        # API key auth: if the API key's org_id matches the requested org_id, grant access
        # str(AuthMethod.API_KEY) returns "api_key" for StrEnum
        if auth_method and str(auth_method) == "api_key":
            if user_org_id == organization_id:
                # Return synthetic member record for API key access
                # Translate org ID for API key auth as well
                translated_org_id = await translate_clerk_org_id(organization_id, user_id, user_email) if is_clerk_org_id(organization_id) else organization_id
                return {
                    "user_id": user_id,
                    "organization_id": translated_org_id,
                    "role": "admin",  # API keys get admin role by default
                    "status": "active",
                    "auth_method": "api_key"
                }, translated_org_id
            else:
                raise HTTPException(
                    status_code=403,
                    detail="API key not authorized for this organization"
                )

    supabase = get_supabase_client()

    # For Clerk org IDs, ensure the org is synced to Supabase first
    if is_clerk_org_id(organization_id):
        try:
            # This will create the org in Supabase if it doesn't exist
            organization_id = await ensure_clerk_org_synced(
                organization_id,
                user_id,
                user_email
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to sync Clerk org", org_id=organization_id, error=str(e))
            # Continue with verification - might still work if already synced

    # Try by user_id first (only active members)
    result = await supabase.request(
        f"/organization_members?organization_id=eq.{organization_id}&user_id=eq.{user_id}&status=eq.active&select=*"
    )

    # If not found by user_id, try by email (only active members)
    if (not result.get("data") or result.get("error")) and user_email:
        result = await supabase.request(
            f"/organization_members?organization_id=eq.{organization_id}&email=eq.{user_email}&status=eq.active&select=*"
        )

    if result.get("error") or not result.get("data"):
        raise HTTPException(status_code=403, detail="Access denied to this organization")

    member = result["data"][0]

    if required_roles and member["role"] not in required_roles:
        raise HTTPException(
            status_code=403,
            detail=f"Requires {' or '.join(required_roles)} role"
        )

    # Return both member record and the translated organization ID
    return member, organization_id


async def log_audit(
    organization_id: str,
    user_id: str,
    user_email: str,
    action: str,
    resource_type: str,
    resource_id: str,
    description: str,
    metadata: dict = None,
    request: Request = None
):
    """Create an audit log entry."""
    supabase = get_supabase_client()

    ip_address = None
    user_agent = None

    if request:
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

    await supabase.insert("audit_logs", {
        "organization_id": organization_id,
        "user_id": user_id,
        "user_email": user_email,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "description": description,
        "metadata": metadata or {},
        "ip_address": ip_address,
        "user_agent": user_agent,
        "status": "success",
    })


# ============================================================================
# Organization Endpoints
# ============================================================================

@router.get("/organizations", response_model=list[OrganizationResponse])
async def list_organizations(request: Request):
    """List all organizations the current user belongs to."""
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Get user's memberships
    memberships = await supabase.request(
        f"/organization_members?user_id=eq.{user['user_id']}&status=eq.active&select=organization_id"
    )

    if memberships.get("error") or not memberships.get("data"):
        return []

    org_ids = [m["organization_id"] for m in memberships["data"]]

    # Get organization details
    orgs = await supabase.request(
        f"/organizations?id=in.({','.join(org_ids)})&select=*"
    )

    if orgs.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch organizations")

    # Get member counts
    result = []
    for org in orgs.get("data", []):
        members = await supabase.request(
            f"/organization_members?organization_id=eq.{org['id']}&status=eq.active&select=id"
        )
        member_count = len(members.get("data", []))

        result.append(OrganizationResponse(
            id=org["id"],
            name=org["name"],
            slug=org["slug"],
            plan=org["plan"],
            ai_budget_daily=float(org.get("ai_budget_daily", 1.0)),
            ai_budget_monthly=float(org.get("ai_budget_monthly", 25.0)),
            ai_spend_today=float(org.get("ai_spend_today", 0)),
            ai_spend_this_month=float(org.get("ai_spend_this_month", 0)),
            features=org.get("features", {}),
            member_count=member_count,
            created_at=org["created_at"],
        ))

    return result


@router.post("/organizations", response_model=OrganizationResponse)
async def create_organization(body: CreateOrganizationRequest, request: Request):
    """Create a new organization."""
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Generate slug from name if not provided
    if body.slug:
        slug = body.slug
    else:
        # Auto-generate slug from name
        slug = body.name.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)  # Remove special chars
        slug = re.sub(r'\s+', '-', slug)          # Spaces to hyphens
        slug = re.sub(r'-+', '-', slug)           # Multiple hyphens to single
        slug = slug[:50].strip('-')               # Limit length, trim hyphens

    # Check if slug is available
    existing = await supabase.request(
        f"/organizations?slug=eq.{slug}&select=id"
    )

    # If slug taken, append random suffix
    if existing.get("data"):
        slug = f"{slug[:42]}-{secrets.token_hex(4)}"

    # Create organization
    org_result = await supabase.insert("organizations", {
        "name": body.name,
        "slug": slug,
        "plan": "free",
    })

    if org_result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to create organization")

    org = org_result["data"][0]

    # Add creator as owner
    await supabase.insert("organization_members", {
        "organization_id": org["id"],
        "user_id": user["user_id"],
        "email": user["email"] or "",
        "role": "owner",
        "status": "active",
        "accepted_at": datetime.now(timezone.utc).isoformat(),
    })

    # Create default healing config
    await supabase.insert("self_healing_config", {
        "organization_id": org["id"],
    })

    # Audit log
    await log_audit(
        organization_id=org["id"],
        user_id=user["user_id"],
        user_email=user["email"],
        action="org.create",
        resource_type="organization",
        resource_id=org["id"],
        description=f"Created organization '{body.name}'",
        request=request,
    )

    logger.info("Organization created", org_id=org["id"], name=body.name)

    return OrganizationResponse(
        id=org["id"],
        name=org["name"],
        slug=org["slug"],
        plan=org["plan"],
        ai_budget_daily=float(org.get("ai_budget_daily", 1.0)),
        ai_budget_monthly=float(org.get("ai_budget_monthly", 25.0)),
        ai_spend_today=0,
        ai_spend_this_month=0,
        features=org.get("features", {}),
        member_count=1,
        created_at=org["created_at"],
    )


@router.get("/organizations/{org_id}", response_model=OrganizationResponse)
async def get_organization(org_id: str, request: Request):
    """Get organization details."""
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

    supabase = get_supabase_client()

    org = await supabase.request(f"/organizations?id=eq.{supabase_org_id}&select=*")

    if org.get("error") or not org.get("data"):
        raise HTTPException(status_code=404, detail="Organization not found")

    org_data = org["data"][0]

    # Get member count
    members = await supabase.request(
        f"/organization_members?organization_id=eq.{supabase_org_id}&status=eq.active&select=id"
    )

    return OrganizationResponse(
        id=org_data["id"],
        name=org_data["name"],
        slug=org_data["slug"],
        plan=org_data["plan"],
        ai_budget_daily=float(org_data.get("ai_budget_daily", 1.0)),
        ai_budget_monthly=float(org_data.get("ai_budget_monthly", 25.0)),
        ai_spend_today=float(org_data.get("ai_spend_today", 0)),
        ai_spend_this_month=float(org_data.get("ai_spend_this_month", 0)),
        features=org_data.get("features", {}),
        member_count=len(members.get("data", [])),
        created_at=org_data["created_at"],
    )


@router.patch("/organizations/{org_id}", response_model=OrganizationResponse)
async def update_organization(org_id: str, body: UpdateOrganizationRequest, request: Request):
    """Update organization settings (admin/owner only)."""
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}

    if body.name is not None:
        update_data["name"] = body.name
    if body.ai_budget_daily is not None:
        update_data["ai_budget_daily"] = body.ai_budget_daily
    if body.ai_budget_monthly is not None:
        update_data["ai_budget_monthly"] = body.ai_budget_monthly
    if body.settings is not None:
        update_data["settings"] = body.settings

    result = await supabase.update("organizations", {"id": f"eq.{supabase_org_id}"}, update_data)

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to update organization")

    # Audit log
    await log_audit(
        organization_id=supabase_org_id,
        user_id=user["user_id"],
        user_email=user["email"],
        action="org.update",
        resource_type="organization",
        resource_id=supabase_org_id,
        description="Updated organization settings",
        metadata={"changes": update_data},
        request=request,
    )

    return await get_organization(supabase_org_id, request)


# ============================================================================
# Member Management Endpoints
# ============================================================================

@router.get("/organizations/{org_id}/members", response_model=list[MemberResponse])
async def list_members(org_id: str, request: Request):
    """List all members of an organization."""
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

    supabase = get_supabase_client()

    members = await supabase.request(
        f"/organization_members?organization_id=eq.{supabase_org_id}&select=*&order=created_at.asc"
    )

    if members.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch members")

    return [
        MemberResponse(
            id=m["id"],
            user_id=m["user_id"],
            email=m["email"],
            role=m["role"],
            status=m["status"],
            invited_at=m.get("invited_at"),
            accepted_at=m.get("accepted_at"),
            created_at=m["created_at"],
        )
        for m in members.get("data", [])
    ]


@router.post("/organizations/{org_id}/members/invite", response_model=MemberResponse)
async def invite_member(org_id: str, body: InviteMemberRequest, request: Request):
    """Invite a new member to the organization (admin/owner only)."""
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Check if already a member
    existing = await supabase.request(
        f"/organization_members?organization_id=eq.{supabase_org_id}&email=eq.{body.email}&select=id,status"
    )

    if existing.get("data"):
        member = existing["data"][0]
        if member["status"] == "active":
            raise HTTPException(status_code=400, detail="User is already a member")
        elif member["status"] == "pending":
            raise HTTPException(status_code=400, detail="Invitation already pending")

    # Get inviter's member record
    inviter = await supabase.request(
        f"/organization_members?organization_id=eq.{supabase_org_id}&user_id=eq.{user['user_id']}&select=id"
    )
    inviter_id = inviter["data"][0]["id"] if inviter.get("data") else None

    # Create pending membership
    member_result = await supabase.insert("organization_members", {
        "organization_id": supabase_org_id,
        "user_id": f"pending_{secrets.token_hex(8)}",  # Placeholder until they accept
        "email": body.email,
        "role": body.role,
        "status": "pending",
        "invited_by": inviter_id,
        "invited_at": datetime.now(timezone.utc).isoformat(),
    })

    if member_result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to create invitation")

    member = member_result["data"][0]

    # TODO: Send invitation email

    # Audit log
    await log_audit(
        organization_id=supabase_org_id,
        user_id=user["user_id"],
        user_email=user["email"],
        action="member.invite",
        resource_type="member",
        resource_id=member["id"],
        description=f"Invited {body.email} as {body.role}",
        metadata={"invited_email": body.email, "role": body.role},
        request=request,
    )

    logger.info("Member invited", org_id=supabase_org_id, email=body.email, role=body.role)

    return MemberResponse(
        id=member["id"],
        user_id=member["user_id"],
        email=member["email"],
        role=member["role"],
        status=member["status"],
        invited_at=member.get("invited_at"),
        accepted_at=member.get("accepted_at"),
        created_at=member["created_at"],
    )


@router.patch("/organizations/{org_id}/members/{member_id}/role")
async def update_member_role(
    org_id: str,
    member_id: str,
    body: UpdateMemberRoleRequest,
    request: Request
):
    """Update a member's role (owner only)."""
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], ["owner"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Get target member
    member = await supabase.request(
        f"/organization_members?id=eq.{member_id}&organization_id=eq.{supabase_org_id}&select=*"
    )

    if not member.get("data"):
        raise HTTPException(status_code=404, detail="Member not found")

    target_member = member["data"][0]

    # Cannot change owner's role
    if target_member["role"] == "owner":
        raise HTTPException(status_code=400, detail="Cannot change owner's role")

    old_role = target_member["role"]

    # Update role
    await supabase.update(
        "organization_members",
        {"id": f"eq.{member_id}"},
        {"role": body.role, "updated_at": datetime.now(timezone.utc).isoformat()}
    )

    # Audit log
    await log_audit(
        organization_id=supabase_org_id,
        user_id=user["user_id"],
        user_email=user["email"],
        action="member.role_change",
        resource_type="member",
        resource_id=member_id,
        description=f"Changed role from {old_role} to {body.role}",
        metadata={"old_role": old_role, "new_role": body.role, "member_email": target_member["email"]},
        request=request,
    )

    logger.info("Member role updated", org_id=supabase_org_id, member_id=member_id, new_role=body.role)

    return {"success": True, "message": f"Role updated to {body.role}"}


@router.delete("/organizations/{org_id}/members/{member_id}")
async def remove_member(org_id: str, member_id: str, request: Request):
    """Remove a member from the organization (admin/owner only)."""
    user = await get_current_user(request)
    current_member, supabase_org_id = await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Get target member
    member = await supabase.request(
        f"/organization_members?id=eq.{member_id}&organization_id=eq.{supabase_org_id}&select=*"
    )

    if not member.get("data"):
        raise HTTPException(status_code=404, detail="Member not found")

    target_member = member["data"][0]

    # Cannot remove owner
    if target_member["role"] == "owner":
        raise HTTPException(status_code=400, detail="Cannot remove organization owner")

    # Admin cannot remove other admins
    if current_member["role"] == "admin" and target_member["role"] == "admin":
        raise HTTPException(status_code=403, detail="Admins cannot remove other admins")

    # Delete member
    await supabase.request(
        f"/organization_members?id=eq.{member_id}",
        method="DELETE"
    )

    # Audit log
    await log_audit(
        organization_id=supabase_org_id,
        user_id=user["user_id"],
        user_email=user["email"],
        action="member.remove",
        resource_type="member",
        resource_id=member_id,
        description=f"Removed {target_member['email']} from organization",
        metadata={"removed_email": target_member["email"], "removed_role": target_member["role"]},
        request=request,
    )

    logger.info("Member removed", org_id=supabase_org_id, member_email=target_member["email"])

    return {"success": True, "message": "Member removed"}
