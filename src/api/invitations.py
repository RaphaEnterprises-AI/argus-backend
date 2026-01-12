"""Invitation Management API endpoints.

Provides endpoints for:
- Sending invitations to join organizations
- Listing pending invitations
- Revoking invitations
- Accepting invitations
- Validating invitation tokens
"""

import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, EmailStr
import structlog

from src.services.supabase_client import get_supabase_client
from src.services.email_service import get_email_service
from src.api.teams import get_current_user, verify_org_access, log_audit

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/invitations", tags=["Invitation Management"])


# ============================================================================
# Request/Response Models
# ============================================================================

class SendInvitationRequest(BaseModel):
    """Request to send an invitation."""
    email: EmailStr
    role: str = Field("member", pattern="^(admin|member|viewer)$")
    message: Optional[str] = Field(None, max_length=500, description="Optional personal message")


class InvitationResponse(BaseModel):
    """Invitation details response."""
    id: str
    organization_id: str
    email: str
    role: str
    status: str
    message: Optional[str]
    token_expires_at: str
    invited_by: Optional[str]
    created_at: str


class InvitationListResponse(BaseModel):
    """List of invitations response."""
    invitations: list[InvitationResponse]
    total: int


class AcceptInvitationResponse(BaseModel):
    """Response after accepting an invitation."""
    success: bool
    message: str
    organization_id: str
    organization_name: str
    role: str


class ValidateTokenResponse(BaseModel):
    """Response for token validation."""
    valid: bool
    email: Optional[str] = None
    organization_name: Optional[str] = None
    role: Optional[str] = None
    expires_at: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def generate_invitation_token() -> str:
    """Generate a secure invitation token."""
    return secrets.token_urlsafe(32)


def get_token_expiration() -> str:
    """Get expiration datetime 7 days from now."""
    return (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()


# ============================================================================
# Organization Invitation Endpoints (Authenticated)
# ============================================================================

@router.post("/organizations/{org_id}/invitations", response_model=InvitationResponse)
async def send_invitation(org_id: str, body: SendInvitationRequest, request: Request):
    """Send an invitation to join the organization (admin/owner only).

    Creates a new invitation with a secure token that expires in 7 days.
    The invitation email should be sent separately (e.g., via email service).
    """
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Check if email is already a member
    existing_member = await supabase.request(
        f"/organization_members?organization_id=eq.{org_id}&email=eq.{body.email}&status=eq.active&select=id"
    )

    if existing_member.get("data"):
        raise HTTPException(status_code=400, detail="User is already a member of this organization")

    # Check if there's already a pending invitation for this email
    existing_invitation = await supabase.request(
        f"/invitations?organization_id=eq.{org_id}&email=eq.{body.email}&status=eq.pending&select=id"
    )

    if existing_invitation.get("data"):
        raise HTTPException(status_code=400, detail="An invitation is already pending for this email")

    # Get inviter's member record for invited_by reference
    inviter_member = await supabase.request(
        f"/organization_members?organization_id=eq.{org_id}&user_id=eq.{user['user_id']}&select=id"
    )
    inviter_id = inviter_member["data"][0]["id"] if inviter_member.get("data") else None

    # Generate secure token and expiration
    token = generate_invitation_token()
    token_expires_at = get_token_expiration()

    # Create invitation
    invitation_result = await supabase.insert("invitations", {
        "organization_id": org_id,
        "email": body.email,
        "role": body.role,
        "token": token,
        "token_expires_at": token_expires_at,
        "message": body.message,
        "status": "pending",
        "invited_by": inviter_id,
    })

    if invitation_result.get("error"):
        logger.error("Failed to create invitation", error=invitation_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create invitation")

    invitation = invitation_result["data"][0]

    # Get organization name for the email
    org_result = await supabase.request(f"/organizations?id=eq.{org_id}&select=name")
    org_name = org_result["data"][0]["name"] if org_result.get("data") else "the organization"

    # Send invitation email
    email_service = get_email_service()
    email_sent = await email_service.send_invitation(
        to=body.email,
        org_name=org_name,
        inviter_email=user.get("email", "A team member"),
        token=token,
        role=body.role,
    )

    if not email_sent:
        logger.warning("Failed to send invitation email", email=body.email)
        # Don't fail the request - invitation is created, email can be resent

    # Audit log
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="invitation.send",
        resource_type="invitation",
        resource_id=invitation["id"],
        description=f"Sent invitation to {body.email} as {body.role}",
        metadata={"invited_email": body.email, "role": body.role, "email_sent": email_sent},
        request=request,
    )

    logger.info("Invitation sent", org_id=org_id, email=body.email, role=body.role, email_sent=email_sent)

    return InvitationResponse(
        id=invitation["id"],
        organization_id=invitation["organization_id"],
        email=invitation["email"],
        role=invitation["role"],
        status=invitation["status"],
        message=invitation.get("message"),
        token_expires_at=invitation["token_expires_at"],
        invited_by=invitation.get("invited_by"),
        created_at=invitation["created_at"],
    )


@router.get("/organizations/{org_id}/invitations", response_model=InvitationListResponse)
async def list_pending_invitations(org_id: str, request: Request):
    """List all pending invitations for the organization.

    Returns invitations with status 'pending' that haven't expired.
    """
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

    supabase = get_supabase_client()

    # Get pending invitations (not expired)
    # Use 'Z' suffix instead of '+00:00' to avoid URL encoding issues with '+'
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    invitations_result = await supabase.request(
        f"/invitations?organization_id=eq.{org_id}&status=eq.pending"
        f"&token_expires_at=gt.{now}&select=*&order=created_at.desc"
    )

    if invitations_result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch invitations")

    invitations = invitations_result.get("data", [])

    return InvitationListResponse(
        invitations=[
            InvitationResponse(
                id=inv["id"],
                organization_id=inv["organization_id"],
                email=inv["email"],
                role=inv["role"],
                status=inv["status"],
                message=inv.get("message"),
                token_expires_at=inv["token_expires_at"],
                invited_by=inv.get("invited_by"),
                created_at=inv["created_at"],
            )
            for inv in invitations
        ],
        total=len(invitations),
    )


@router.delete("/organizations/{org_id}/invitations/{invite_id}")
async def revoke_invitation(org_id: str, invite_id: str, request: Request):
    """Revoke a pending invitation (admin/owner only).

    Sets the invitation status to 'revoked', preventing it from being accepted.
    """
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Get the invitation
    invitation_result = await supabase.request(
        f"/invitations?id=eq.{invite_id}&organization_id=eq.{org_id}&select=*"
    )

    if not invitation_result.get("data"):
        raise HTTPException(status_code=404, detail="Invitation not found")

    invitation = invitation_result["data"][0]

    if invitation["status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot revoke invitation with status '{invitation['status']}'"
        )

    # Update invitation status to revoked
    update_result = await supabase.update(
        "invitations",
        {"id": f"eq.{invite_id}"},
        {"status": "revoked", "updated_at": datetime.now(timezone.utc).isoformat()}
    )

    if update_result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to revoke invitation")

    # Audit log
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="invitation.revoke",
        resource_type="invitation",
        resource_id=invite_id,
        description=f"Revoked invitation for {invitation['email']}",
        metadata={"revoked_email": invitation["email"]},
        request=request,
    )

    logger.info("Invitation revoked", org_id=org_id, invite_id=invite_id, email=invitation["email"])

    return {"success": True, "message": "Invitation revoked successfully"}


# ============================================================================
# Public Invitation Endpoints (Token-based)
# ============================================================================

@router.get("/validate/{token}", response_model=ValidateTokenResponse)
async def validate_invitation_token(token: str):
    """Validate an invitation token (public endpoint).

    Checks if the token exists, is pending, and hasn't expired.
    Returns basic invitation details without accepting it.
    """
    supabase = get_supabase_client()

    # Get invitation by token
    invitation_result = await supabase.request(
        f"/invitations?token=eq.{token}&select=*,organizations(name)"
    )

    if not invitation_result.get("data"):
        return ValidateTokenResponse(valid=False, error="Invalid invitation token")

    invitation = invitation_result["data"][0]

    # Check if already accepted or revoked
    if invitation["status"] != "pending":
        return ValidateTokenResponse(
            valid=False,
            error=f"Invitation has already been {invitation['status']}"
        )

    # Check if expired
    expires_at = datetime.fromisoformat(invitation["token_expires_at"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) > expires_at:
        return ValidateTokenResponse(valid=False, error="Invitation has expired")

    # Get organization name
    org_name = None
    if invitation.get("organizations"):
        org_name = invitation["organizations"].get("name")

    return ValidateTokenResponse(
        valid=True,
        email=invitation["email"],
        organization_name=org_name,
        role=invitation["role"],
        expires_at=invitation["token_expires_at"],
    )


@router.post("/accept/{token}", response_model=AcceptInvitationResponse)
async def accept_invitation(token: str, request: Request):
    """Accept an invitation and join the organization.

    This endpoint requires authentication. The authenticated user must match
    the invitation email, or be a new user accepting their first invitation.

    Creates an organization_members record and updates the invitation status.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Get invitation by token with organization details
    invitation_result = await supabase.request(
        f"/invitations?token=eq.{token}&select=*,organizations(id,name,slug)"
    )

    if not invitation_result.get("data"):
        raise HTTPException(status_code=404, detail="Invalid invitation token")

    invitation = invitation_result["data"][0]

    # Validate invitation status
    if invitation["status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Invitation has already been {invitation['status']}"
        )

    # Check if expired
    expires_at = datetime.fromisoformat(invitation["token_expires_at"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=400, detail="Invitation has expired")

    # Verify the accepting user matches the invitation email (if email available)
    if user.get("email") and user["email"].lower() != invitation["email"].lower():
        raise HTTPException(
            status_code=403,
            detail="This invitation was sent to a different email address"
        )

    org_id = invitation["organization_id"]
    org_name = invitation.get("organizations", {}).get("name", "Unknown")

    # Check if user is already a member
    existing_member = await supabase.request(
        f"/organization_members?organization_id=eq.{org_id}&user_id=eq.{user['user_id']}&select=id"
    )

    if existing_member.get("data"):
        raise HTTPException(status_code=400, detail="You are already a member of this organization")

    # Create organization membership
    now = datetime.now(timezone.utc).isoformat()

    member_result = await supabase.insert("organization_members", {
        "organization_id": org_id,
        "user_id": user["user_id"],
        "email": invitation["email"],
        "role": invitation["role"],
        "status": "active",
        "invited_by": invitation.get("invited_by"),
        "invited_at": invitation["created_at"],
        "accepted_at": now,
    })

    if member_result.get("error"):
        logger.error("Failed to create membership", error=member_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create membership")

    # Update invitation status to accepted
    await supabase.update(
        "invitations",
        {"id": f"eq.{invitation['id']}"},
        {
            "status": "accepted",
            "accepted_by": user["user_id"],
            "accepted_at": now,
            "updated_at": now,
        }
    )

    # Audit log
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email") or invitation["email"],
        action="invitation.accept",
        resource_type="invitation",
        resource_id=invitation["id"],
        description=f"Accepted invitation and joined as {invitation['role']}",
        metadata={"role": invitation["role"]},
        request=request,
    )

    logger.info(
        "Invitation accepted",
        org_id=org_id,
        user_id=user["user_id"],
        role=invitation["role"]
    )

    return AcceptInvitationResponse(
        success=True,
        message="Successfully joined the organization",
        organization_id=org_id,
        organization_name=org_name,
        role=invitation["role"],
    )
