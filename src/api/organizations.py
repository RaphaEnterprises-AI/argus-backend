"""Organization Management API endpoints.

Provides endpoints for:
- Creating organizations
- Listing user's organizations
- Getting organization details
- Updating organizations
- Deleting organizations
- Transferring ownership
"""

import re
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import structlog

from src.services.supabase_client import get_supabase_client
from src.api.teams import get_current_user, verify_org_access, log_audit

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/organizations", tags=["Organizations"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateOrganizationRequest(BaseModel):
    """Request to create a new organization."""
    name: str = Field(..., min_length=2, max_length=100)


class UpdateOrganizationRequest(BaseModel):
    """Request to update organization settings."""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    logo_url: Optional[str] = Field(None, max_length=500)
    domain: Optional[str] = Field(None, max_length=255)
    ai_budget_daily: Optional[float] = Field(None, ge=0, le=10000)
    ai_budget_monthly: Optional[float] = Field(None, ge=0, le=100000)
    settings: Optional[dict] = None
    features: Optional[dict] = None
    sso_enabled: Optional[bool] = None


class TransferOwnershipRequest(BaseModel):
    """Request to transfer organization ownership."""
    new_owner_user_id: str = Field(..., min_length=1)


class OrganizationResponse(BaseModel):
    """Organization details response."""
    id: str
    name: str
    slug: str
    plan: str
    ai_budget_daily: float
    ai_budget_monthly: float
    settings: Optional[dict]
    features: Optional[dict]
    stripe_customer_id: Optional[str]
    stripe_subscription_id: Optional[str]
    logo_url: Optional[str]
    domain: Optional[str]
    sso_enabled: bool
    member_count: int
    created_at: str
    updated_at: Optional[str]


class OrganizationListResponse(BaseModel):
    """Organization list item response."""
    id: str
    name: str
    slug: str
    plan: str
    logo_url: Optional[str]
    member_count: int
    role: str  # User's role in this organization
    created_at: str


# ============================================================================
# Helper Functions
# ============================================================================

def generate_slug(name: str) -> str:
    """Generate a URL-friendly slug from organization name.

    Converts to lowercase, replaces spaces and special chars with hyphens,
    removes consecutive hyphens, and trims hyphens from ends.
    """
    # Convert to lowercase
    slug = name.lower()
    # Replace spaces and special characters with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    # Remove consecutive hyphens
    slug = re.sub(r'-+', '-', slug)
    # Trim hyphens from ends
    slug = slug.strip('-')
    return slug


async def ensure_unique_slug(base_slug: str) -> str:
    """Ensure the slug is unique by appending random chars if needed."""
    supabase = get_supabase_client()
    slug = base_slug

    # Check if slug exists
    existing = await supabase.request(
        f"/organizations?slug=eq.{slug}&select=id"
    )

    if not existing.get("data"):
        return slug

    # Slug exists, append random chars and try again
    for _ in range(10):  # Max 10 attempts
        random_suffix = secrets.token_hex(3)  # 6 chars
        slug = f"{base_slug}-{random_suffix}"

        existing = await supabase.request(
            f"/organizations?slug=eq.{slug}&select=id"
        )

        if not existing.get("data"):
            return slug

    # If still not unique after 10 attempts, use a longer suffix
    return f"{base_slug}-{secrets.token_hex(8)}"


async def get_member_count(org_id: str) -> int:
    """Get the count of active members in an organization."""
    supabase = get_supabase_client()
    members = await supabase.request(
        f"/organization_members?organization_id=eq.{org_id}&status=eq.active&select=id"
    )
    return len(members.get("data", []))


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/", response_model=OrganizationResponse)
async def create_organization(body: CreateOrganizationRequest, request: Request):
    """Create a new organization.

    The current user becomes the owner of the organization.
    A URL-friendly slug is automatically generated from the name.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Generate unique slug from name
    base_slug = generate_slug(body.name)
    if not base_slug:
        raise HTTPException(status_code=400, detail="Organization name must contain at least one alphanumeric character")

    slug = await ensure_unique_slug(base_slug)

    # Create organization
    org_result = await supabase.insert("organizations", {
        "name": body.name,
        "slug": slug,
        "plan": "free",
        "ai_budget_daily": 1.0,
        "ai_budget_monthly": 25.0,
        "settings": {},
        "features": {},
        "sso_enabled": False,
    })

    if org_result.get("error"):
        logger.error("Failed to create organization", error=org_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create organization")

    org = org_result["data"][0]

    # Add creator as owner
    member_result = await supabase.insert("organization_members", {
        "organization_id": org["id"],
        "user_id": user["user_id"],
        "email": user.get("email") or "",
        "role": "owner",
        "status": "active",
        "accepted_at": datetime.now(timezone.utc).isoformat(),
    })

    if member_result.get("error"):
        # Rollback: delete the organization
        await supabase.request(
            f"/organizations?id=eq.{org['id']}",
            method="DELETE"
        )
        logger.error("Failed to add owner membership", error=member_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create organization membership")

    # Audit log
    await log_audit(
        organization_id=org["id"],
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="org.create",
        resource_type="organization",
        resource_id=org["id"],
        description=f"Created organization '{body.name}'",
        metadata={"name": body.name, "slug": slug},
        request=request,
    )

    logger.info("Organization created", org_id=org["id"], name=body.name, slug=slug)

    return OrganizationResponse(
        id=org["id"],
        name=org["name"],
        slug=org["slug"],
        plan=org["plan"],
        ai_budget_daily=float(org.get("ai_budget_daily", 1.0)),
        ai_budget_monthly=float(org.get("ai_budget_monthly", 25.0)),
        settings=org.get("settings"),
        features=org.get("features"),
        stripe_customer_id=org.get("stripe_customer_id"),
        stripe_subscription_id=org.get("stripe_subscription_id"),
        logo_url=org.get("logo_url"),
        domain=org.get("domain"),
        sso_enabled=org.get("sso_enabled", False),
        member_count=1,
        created_at=org["created_at"],
        updated_at=org.get("updated_at"),
    )


@router.get("/", response_model=list[OrganizationListResponse])
async def list_organizations(request: Request):
    """List all organizations the current user has access to."""
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Get user's memberships with organization data
    memberships = await supabase.request(
        f"/organization_members?user_id=eq.{user['user_id']}&status=eq.active&select=organization_id,role"
    )

    if memberships.get("error") or not memberships.get("data"):
        return []

    # Build lookup of org_id -> role
    org_roles = {m["organization_id"]: m["role"] for m in memberships["data"]}
    org_ids = list(org_roles.keys())

    if not org_ids:
        return []

    # Get organization details
    orgs = await supabase.request(
        f"/organizations?id=in.({','.join(org_ids)})&select=*&order=created_at.desc"
    )

    if orgs.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch organizations")

    # Build response with member counts
    result = []
    for org in orgs.get("data", []):
        member_count = await get_member_count(org["id"])

        result.append(OrganizationListResponse(
            id=org["id"],
            name=org["name"],
            slug=org["slug"],
            plan=org["plan"],
            logo_url=org.get("logo_url"),
            member_count=member_count,
            role=org_roles.get(org["id"], "member"),
            created_at=org["created_at"],
        ))

    return result


@router.get("/{org_id}", response_model=OrganizationResponse)
async def get_organization(org_id: str, request: Request):
    """Get organization details."""
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], user_email=user.get("email"))

    supabase = get_supabase_client()

    org = await supabase.request(f"/organizations?id=eq.{org_id}&select=*")

    if org.get("error") or not org.get("data"):
        raise HTTPException(status_code=404, detail="Organization not found")

    org_data = org["data"][0]
    member_count = await get_member_count(org_id)

    return OrganizationResponse(
        id=org_data["id"],
        name=org_data["name"],
        slug=org_data["slug"],
        plan=org_data["plan"],
        ai_budget_daily=float(org_data.get("ai_budget_daily", 1.0)),
        ai_budget_monthly=float(org_data.get("ai_budget_monthly", 25.0)),
        settings=org_data.get("settings"),
        features=org_data.get("features"),
        stripe_customer_id=org_data.get("stripe_customer_id"),
        stripe_subscription_id=org_data.get("stripe_subscription_id"),
        logo_url=org_data.get("logo_url"),
        domain=org_data.get("domain"),
        sso_enabled=org_data.get("sso_enabled", False),
        member_count=member_count,
        created_at=org_data["created_at"],
        updated_at=org_data.get("updated_at"),
    )


@router.put("/{org_id}", response_model=OrganizationResponse)
async def update_organization(org_id: str, body: UpdateOrganizationRequest, request: Request):
    """Update organization settings (admin/owner only)."""
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"))

    supabase = get_supabase_client()

    # Build update data from non-null fields
    update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}

    if body.name is not None:
        update_data["name"] = body.name
    if body.logo_url is not None:
        update_data["logo_url"] = body.logo_url
    if body.domain is not None:
        update_data["domain"] = body.domain
    if body.ai_budget_daily is not None:
        update_data["ai_budget_daily"] = body.ai_budget_daily
    if body.ai_budget_monthly is not None:
        update_data["ai_budget_monthly"] = body.ai_budget_monthly
    if body.settings is not None:
        update_data["settings"] = body.settings
    if body.features is not None:
        update_data["features"] = body.features
    if body.sso_enabled is not None:
        update_data["sso_enabled"] = body.sso_enabled

    result = await supabase.update("organizations", {"id": f"eq.{org_id}"}, update_data)

    if result.get("error"):
        logger.error("Failed to update organization", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to update organization")

    # Audit log
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="org.update",
        resource_type="organization",
        resource_id=org_id,
        description="Updated organization settings",
        metadata={"changes": {k: v for k, v in update_data.items() if k != "updated_at"}},
        request=request,
    )

    logger.info("Organization updated", org_id=org_id)

    return await get_organization(org_id, request)


@router.delete("/{org_id}")
async def delete_organization(org_id: str, request: Request):
    """Delete an organization (owner only).

    This permanently deletes the organization and all associated data.
    """
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], ["owner"], user.get("email"))

    supabase = get_supabase_client()

    # Get organization details for audit log
    org = await supabase.request(f"/organizations?id=eq.{org_id}&select=name,slug")

    if org.get("error") or not org.get("data"):
        raise HTTPException(status_code=404, detail="Organization not found")

    org_data = org["data"][0]

    # Delete all organization members first
    await supabase.request(
        f"/organization_members?organization_id=eq.{org_id}",
        method="DELETE"
    )

    # Delete the organization
    delete_result = await supabase.request(
        f"/organizations?id=eq.{org_id}",
        method="DELETE"
    )

    if delete_result.get("error"):
        logger.error("Failed to delete organization", error=delete_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to delete organization")

    # Note: Audit log for deleted org is stored but org reference will be orphaned
    # In production, consider a soft delete pattern instead
    logger.info("Organization deleted", org_id=org_id, name=org_data["name"])

    return {"success": True, "message": f"Organization '{org_data['name']}' has been deleted"}


@router.post("/{org_id}/transfer")
async def transfer_ownership(org_id: str, body: TransferOwnershipRequest, request: Request):
    """Transfer organization ownership to another member (owner only).

    The new owner must already be a member of the organization.
    The current owner will be demoted to admin.
    """
    user = await get_current_user(request)
    current_member = await verify_org_access(org_id, user["user_id"], ["owner"], user.get("email"))

    supabase = get_supabase_client()

    # Verify new owner is a member of the organization
    new_owner_member = await supabase.request(
        f"/organization_members?organization_id=eq.{org_id}&user_id=eq.{body.new_owner_user_id}&status=eq.active&select=*"
    )

    if not new_owner_member.get("data"):
        raise HTTPException(
            status_code=400,
            detail="The specified user is not an active member of this organization"
        )

    new_owner = new_owner_member["data"][0]

    # Cannot transfer to self
    if new_owner["user_id"] == user["user_id"]:
        raise HTTPException(status_code=400, detail="Cannot transfer ownership to yourself")

    # Update new owner to owner role
    await supabase.update(
        "organization_members",
        {"id": f"eq.{new_owner['id']}"},
        {"role": "owner", "updated_at": datetime.now(timezone.utc).isoformat()}
    )

    # Demote current owner to admin
    await supabase.update(
        "organization_members",
        {"id": f"eq.{current_member['id']}"},
        {"role": "admin", "updated_at": datetime.now(timezone.utc).isoformat()}
    )

    # Audit log
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="org.transfer_ownership",
        resource_type="organization",
        resource_id=org_id,
        description=f"Transferred ownership to user {body.new_owner_user_id}",
        metadata={
            "previous_owner_id": user["user_id"],
            "new_owner_id": body.new_owner_user_id,
            "new_owner_email": new_owner.get("email"),
        },
        request=request,
    )

    logger.info(
        "Organization ownership transferred",
        org_id=org_id,
        from_user=user["user_id"],
        to_user=body.new_owner_user_id
    )

    return {
        "success": True,
        "message": "Ownership transferred successfully",
        "new_owner_id": body.new_owner_user_id,
        "previous_owner_role": "admin",
    }
