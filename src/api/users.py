"""User Profile Management API endpoints.

Provides endpoints for:
- Getting and updating user profile
- Managing notification preferences
- Setting default organization
- Listing user's organizations
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, EmailStr
import structlog

from src.services.supabase_client import get_supabase_client
from src.api.teams import get_current_user

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/users", tags=["User Profile"])


# ============================================================================
# Request/Response Models
# ============================================================================

class NotificationPreferences(BaseModel):
    """User notification preferences."""
    email_test_failures: bool = True
    email_test_completions: bool = False
    email_weekly_digest: bool = True
    slack_test_failures: bool = False
    slack_test_completions: bool = False
    in_app_test_failures: bool = True
    in_app_test_completions: bool = True


class UpdateProfileRequest(BaseModel):
    """Request to update user profile."""
    display_name: Optional[str] = Field(None, min_length=1, max_length=100)
    avatar_url: Optional[str] = Field(None, max_length=500)
    bio: Optional[str] = Field(None, max_length=500)
    timezone: Optional[str] = Field(None, max_length=50)
    language: Optional[str] = Field(None, max_length=10)
    theme: Optional[str] = Field(None, pattern="^(light|dark|system)$")


class UpdateNotificationPreferencesRequest(BaseModel):
    """Request to update notification preferences."""
    email_test_failures: Optional[bool] = None
    email_test_completions: Optional[bool] = None
    email_weekly_digest: Optional[bool] = None
    slack_test_failures: Optional[bool] = None
    slack_test_completions: Optional[bool] = None
    in_app_test_failures: Optional[bool] = None
    in_app_test_completions: Optional[bool] = None


class SetDefaultOrganizationRequest(BaseModel):
    """Request to set default organization."""
    organization_id: str = Field(..., min_length=1)
    project_id: Optional[str] = Field(None, min_length=1)


class UserProfileResponse(BaseModel):
    """User profile response."""
    id: str
    user_id: str
    email: Optional[str]
    display_name: Optional[str]
    avatar_url: Optional[str]
    bio: Optional[str]
    timezone: Optional[str]
    language: Optional[str]
    theme: Optional[str]
    notification_preferences: NotificationPreferences
    default_organization_id: Optional[str]
    default_project_id: Optional[str]
    onboarding_completed: bool
    onboarding_step: Optional[str]
    last_login_at: Optional[str]
    last_active_at: Optional[str]
    login_count: int
    created_at: str
    updated_at: str


class OrganizationSummary(BaseModel):
    """Summary of an organization for listing."""
    id: str
    name: str
    slug: str
    role: str
    plan: str
    member_count: int
    is_default: bool


# ============================================================================
# Helper Functions
# ============================================================================

def get_default_notification_preferences() -> dict:
    """Get default notification preferences."""
    return {
        "email_test_failures": True,
        "email_test_completions": False,
        "email_weekly_digest": True,
        "slack_test_failures": False,
        "slack_test_completions": False,
        "in_app_test_failures": True,
        "in_app_test_completions": True,
    }


async def get_or_create_profile(user_id: str, email: Optional[str] = None) -> dict:
    """Get user profile, creating it if it doesn't exist.

    Args:
        user_id: The authenticated user's ID
        email: The user's email (optional, used for profile creation)

    Returns:
        The user profile dict
    """
    supabase = get_supabase_client()

    # Try to get existing profile
    result = await supabase.request(
        f"/user_profiles?user_id=eq.{user_id}&select=*"
    )

    if result.get("data") and len(result["data"]) > 0:
        return result["data"][0]

    # Profile doesn't exist, create it
    now = datetime.now(timezone.utc).isoformat()

    new_profile = {
        "user_id": user_id,
        "email": email,
        "notification_preferences": get_default_notification_preferences(),
        "onboarding_completed": False,
        "login_count": 1,
        "last_login_at": now,
        "last_active_at": now,
        "theme": "system",
        "language": "en",
    }

    create_result = await supabase.insert("user_profiles", new_profile)

    if create_result.get("error"):
        logger.error(
            "Failed to create user profile",
            user_id=user_id,
            error=create_result.get("error")
        )
        raise HTTPException(status_code=500, detail="Failed to create user profile")

    logger.info("User profile created", user_id=user_id)

    return create_result["data"][0]


# ============================================================================
# Profile Endpoints
# ============================================================================

@router.get("/me", response_model=UserProfileResponse)
async def get_my_profile(request: Request):
    """Get the current user's profile.

    Creates the profile automatically if it doesn't exist.
    """
    user = await get_current_user(request)

    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    # Update last_active_at
    supabase = get_supabase_client()
    await supabase.update(
        "user_profiles",
        {"id": f"eq.{profile['id']}"},
        {"last_active_at": datetime.now(timezone.utc).isoformat()}
    )

    # Parse notification preferences
    notification_prefs = profile.get("notification_preferences") or get_default_notification_preferences()

    return UserProfileResponse(
        id=profile["id"],
        user_id=profile["user_id"],
        email=profile.get("email"),
        display_name=profile.get("display_name"),
        avatar_url=profile.get("avatar_url"),
        bio=profile.get("bio"),
        timezone=profile.get("timezone"),
        language=profile.get("language"),
        theme=profile.get("theme"),
        notification_preferences=NotificationPreferences(**notification_prefs),
        default_organization_id=profile.get("default_organization_id"),
        default_project_id=profile.get("default_project_id"),
        onboarding_completed=profile.get("onboarding_completed", False),
        onboarding_step=profile.get("onboarding_step"),
        last_login_at=profile.get("last_login_at"),
        last_active_at=profile.get("last_active_at"),
        login_count=profile.get("login_count", 0),
        created_at=profile["created_at"],
        updated_at=profile["updated_at"],
    )


@router.put("/me", response_model=UserProfileResponse)
async def update_my_profile(body: UpdateProfileRequest, request: Request):
    """Update the current user's profile."""
    user = await get_current_user(request)

    # Ensure profile exists
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    supabase = get_supabase_client()

    # Build update data
    update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}

    if body.display_name is not None:
        update_data["display_name"] = body.display_name
    if body.avatar_url is not None:
        update_data["avatar_url"] = body.avatar_url
    if body.bio is not None:
        update_data["bio"] = body.bio
    if body.timezone is not None:
        update_data["timezone"] = body.timezone
    if body.language is not None:
        update_data["language"] = body.language
    if body.theme is not None:
        update_data["theme"] = body.theme

    result = await supabase.update(
        "user_profiles",
        {"id": f"eq.{profile['id']}"},
        update_data
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to update profile")

    logger.info("User profile updated", user_id=user["user_id"])

    # Return updated profile
    return await get_my_profile(request)


@router.put("/me/preferences", response_model=UserProfileResponse)
async def update_notification_preferences(
    body: UpdateNotificationPreferencesRequest,
    request: Request
):
    """Update notification preferences."""
    user = await get_current_user(request)

    # Ensure profile exists
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    supabase = get_supabase_client()

    # Get current preferences
    current_prefs = profile.get("notification_preferences") or get_default_notification_preferences()

    # Merge updates
    if body.email_test_failures is not None:
        current_prefs["email_test_failures"] = body.email_test_failures
    if body.email_test_completions is not None:
        current_prefs["email_test_completions"] = body.email_test_completions
    if body.email_weekly_digest is not None:
        current_prefs["email_weekly_digest"] = body.email_weekly_digest
    if body.slack_test_failures is not None:
        current_prefs["slack_test_failures"] = body.slack_test_failures
    if body.slack_test_completions is not None:
        current_prefs["slack_test_completions"] = body.slack_test_completions
    if body.in_app_test_failures is not None:
        current_prefs["in_app_test_failures"] = body.in_app_test_failures
    if body.in_app_test_completions is not None:
        current_prefs["in_app_test_completions"] = body.in_app_test_completions

    result = await supabase.update(
        "user_profiles",
        {"id": f"eq.{profile['id']}"},
        {
            "notification_preferences": current_prefs,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to update preferences")

    logger.info("User notification preferences updated", user_id=user["user_id"])

    # Return updated profile
    return await get_my_profile(request)


@router.post("/me/default-organization", response_model=UserProfileResponse)
async def set_default_organization(
    body: SetDefaultOrganizationRequest,
    request: Request
):
    """Set the user's default organization."""
    user = await get_current_user(request)

    # Ensure profile exists
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    supabase = get_supabase_client()

    # Verify user has access to the organization
    membership = await supabase.request(
        f"/organization_members?organization_id=eq.{body.organization_id}&user_id=eq.{user['user_id']}&status=eq.active&select=id"
    )

    # Also check by email if not found by user_id
    if (not membership.get("data") or len(membership["data"]) == 0) and user.get("email"):
        membership = await supabase.request(
            f"/organization_members?organization_id=eq.{body.organization_id}&email=eq.{user['email']}&status=eq.active&select=id"
        )

    if not membership.get("data") or len(membership["data"]) == 0:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this organization"
        )

    # If project_id provided, verify it belongs to the organization
    if body.project_id:
        project = await supabase.request(
            f"/projects?id=eq.{body.project_id}&organization_id=eq.{body.organization_id}&select=id"
        )

        if not project.get("data") or len(project["data"]) == 0:
            raise HTTPException(
                status_code=404,
                detail="Project not found in this organization"
            )

    # Update profile
    update_data = {
        "default_organization_id": body.organization_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    if body.project_id:
        update_data["default_project_id"] = body.project_id

    result = await supabase.update(
        "user_profiles",
        {"id": f"eq.{profile['id']}"},
        update_data
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to set default organization")

    logger.info(
        "User default organization set",
        user_id=user["user_id"],
        organization_id=body.organization_id,
        project_id=body.project_id
    )

    # Return updated profile
    return await get_my_profile(request)


@router.get("/me/organizations", response_model=list[OrganizationSummary])
async def list_my_organizations(request: Request):
    """List all organizations the current user belongs to."""
    user = await get_current_user(request)

    # Ensure profile exists to get default_organization_id
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    supabase = get_supabase_client()

    # Get user's memberships by user_id
    memberships = await supabase.request(
        f"/organization_members?user_id=eq.{user['user_id']}&status=eq.active&select=organization_id,role"
    )

    # Also get memberships by email if available
    memberships_by_email = []
    if user.get("email"):
        email_result = await supabase.request(
            f"/organization_members?email=eq.{user['email']}&status=eq.active&select=organization_id,role"
        )
        memberships_by_email = email_result.get("data", [])

    # Combine memberships, avoiding duplicates
    all_memberships = memberships.get("data", [])
    seen_org_ids = {m["organization_id"] for m in all_memberships}

    for m in memberships_by_email:
        if m["organization_id"] not in seen_org_ids:
            all_memberships.append(m)
            seen_org_ids.add(m["organization_id"])

    if not all_memberships:
        return []

    # Create role lookup
    role_by_org = {m["organization_id"]: m["role"] for m in all_memberships}
    org_ids = list(role_by_org.keys())

    # Get organization details
    orgs = await supabase.request(
        f"/organizations?id=in.({','.join(org_ids)})&select=*"
    )

    if orgs.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch organizations")

    # Build response with member counts
    result = []
    default_org_id = profile.get("default_organization_id")

    for org in orgs.get("data", []):
        # Get member count
        members = await supabase.request(
            f"/organization_members?organization_id=eq.{org['id']}&status=eq.active&select=id"
        )

        result.append(OrganizationSummary(
            id=org["id"],
            name=org["name"],
            slug=org["slug"],
            role=role_by_org.get(org["id"], "member"),
            plan=org.get("plan", "free"),
            member_count=len(members.get("data", [])),
            is_default=(org["id"] == default_org_id),
        ))

    # Sort with default org first, then by name
    result.sort(key=lambda x: (not x.is_default, x.name.lower()))

    return result
