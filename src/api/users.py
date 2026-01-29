"""User Profile Management API endpoints.

Provides endpoints for:
- Getting and updating user profile
- Managing notification preferences
- Setting default organization
- Listing user's organizations
"""

import base64
import io
import secrets
import uuid
from datetime import UTC, datetime
from typing import Optional

import httpx
import structlog
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from src.api.teams import get_current_user
from src.config import settings
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/users", tags=["User Profile"])


# ============================================================================
# Request/Response Models
# ============================================================================

class NotificationPreferences(BaseModel):
    """User notification preferences."""
    # Master toggles for each channel
    email_notifications: bool = True
    slack_notifications: bool = False
    in_app_notifications: bool = True

    # Email-specific settings
    email_test_failures: bool = True
    email_test_completions: bool = False
    email_weekly_digest: bool = True

    # Slack-specific settings
    slack_test_failures: bool = False
    slack_test_completions: bool = False

    # In-app specific settings
    in_app_test_failures: bool = True
    in_app_test_completions: bool = True

    # Alert settings
    test_failure_alerts: bool = True
    daily_digest: bool = False
    weekly_report: bool = True
    alert_threshold: int = 80  # Percentage threshold for alerting


class UpdateProfileRequest(BaseModel):
    """Request to update user profile."""
    display_name: str | None = Field(None, min_length=1, max_length=100)
    avatar_url: str | None = Field(None, max_length=500)
    bio: str | None = Field(None, max_length=500)
    timezone: str | None = Field(None, max_length=50)
    language: str | None = Field(None, max_length=10)
    theme: str | None = Field(None, pattern="^(light|dark|system)$")
    # Professional fields
    job_title: str | None = Field(None, max_length=100)
    company: str | None = Field(None, max_length=100)
    department: str | None = Field(None, max_length=100)
    phone: str | None = Field(None, max_length=20)
    # Social links
    github_username: str | None = Field(None, max_length=50)
    linkedin_url: str | None = Field(None, max_length=200)
    twitter_handle: str | None = Field(None, max_length=50)
    website_url: str | None = Field(None, max_length=200)


class UpdateNotificationPreferencesRequest(BaseModel):
    """Request to update notification preferences."""
    # Master toggles
    email_notifications: bool | None = None
    slack_notifications: bool | None = None
    in_app_notifications: bool | None = None

    # Email-specific settings
    email_test_failures: bool | None = None
    email_test_completions: bool | None = None
    email_weekly_digest: bool | None = None

    # Slack-specific settings
    slack_test_failures: bool | None = None
    slack_test_completions: bool | None = None

    # In-app specific settings
    in_app_test_failures: bool | None = None
    in_app_test_completions: bool | None = None

    # Alert settings
    test_failure_alerts: bool | None = None
    daily_digest: bool | None = None
    weekly_report: bool | None = None
    alert_threshold: int | None = None


class TestDefaults(BaseModel):
    """User test execution defaults."""
    default_browser: str = "chromium"  # chromium, firefox, webkit
    default_timeout: int = 30000  # Timeout in milliseconds
    parallel_execution: bool = True
    retry_failed_tests: bool = True
    max_retries: int = 2
    screenshot_on_failure: bool = True
    video_recording: bool = False


class UpdateTestDefaultsRequest(BaseModel):
    """Request to update test defaults."""
    default_browser: str | None = Field(None, pattern="^(chromium|firefox|webkit)$")
    default_timeout: int | None = Field(None, ge=1000, le=300000)
    parallel_execution: bool | None = None
    retry_failed_tests: bool | None = None
    max_retries: int | None = Field(None, ge=0, le=10)
    screenshot_on_failure: bool | None = None
    video_recording: bool | None = None


class DiscoveryPreferences(BaseModel):
    """User discovery configuration preferences."""
    mode: str = Field(default="standard", pattern="^(quick|standard|deep|focused|autonomous)$")
    strategy: str = Field(default="bfs", pattern="^(bfs|dfs|priority|ai-guided)$")
    maxPages: int = Field(default=50, ge=1, le=500)
    maxDepth: int = Field(default=3, ge=1, le=10)
    includePatterns: str = ""
    excludePatterns: str = "/api/*, /static/*, *.pdf, *.jpg, *.png"
    focusAreas: list[str] = Field(default_factory=list)
    captureScreenshots: bool = True
    useVisionAi: bool = False
    authRequired: bool | None = None
    authConfig: dict | None = None


class UpdateDiscoveryPreferencesRequest(BaseModel):
    """Request to update discovery preferences."""
    mode: str | None = Field(None, pattern="^(quick|standard|deep|focused|autonomous)$")
    strategy: str | None = Field(None, pattern="^(bfs|dfs|priority|ai-guided)$")
    maxPages: int | None = Field(None, ge=1, le=500)
    maxDepth: int | None = Field(None, ge=1, le=10)
    includePatterns: str | None = None
    excludePatterns: str | None = None
    focusAreas: list[str] | None = None
    captureScreenshots: bool | None = None
    useVisionAi: bool | None = None
    authRequired: bool | None = None
    authConfig: dict | None = None


class SetDefaultOrganizationRequest(BaseModel):
    """Request to set default organization."""
    organization_id: str = Field(..., min_length=1)
    project_id: str | None = Field(None, min_length=1)


class UserProfileResponse(BaseModel):
    """User profile response."""
    id: str
    user_id: str
    email: str | None
    display_name: str | None
    avatar_url: str | None
    bio: str | None
    timezone: str | None
    language: str | None
    theme: str | None
    # Professional fields
    job_title: str | None = None
    company: str | None = None
    department: str | None = None
    phone: str | None = None
    # Social links
    github_username: str | None = None
    linkedin_url: str | None = None
    twitter_handle: str | None = None
    website_url: str | None = None
    # Preferences
    notification_preferences: NotificationPreferences
    test_defaults: TestDefaults
    discovery_preferences: DiscoveryPreferences
    default_organization_id: str | None
    default_project_id: str | None
    onboarding_completed: bool
    onboarding_step: int | None
    last_login_at: str | None
    last_active_at: str | None
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
    is_personal: bool = False


class UserPreferencesResponse(BaseModel):
    """User preferences response."""
    notification_preferences: NotificationPreferences
    test_defaults: TestDefaults
    theme: str | None
    timezone: str | None
    language: str | None


# ============================================================================
# Helper Functions
# ============================================================================

def get_default_notification_preferences() -> dict:
    """Get default notification preferences."""
    return {
        # Master toggles
        "email_notifications": True,
        "slack_notifications": False,
        "in_app_notifications": True,
        # Email-specific
        "email_test_failures": True,
        "email_test_completions": False,
        "email_weekly_digest": True,
        # Slack-specific
        "slack_test_failures": False,
        "slack_test_completions": False,
        # In-app specific
        "in_app_test_failures": True,
        "in_app_test_completions": True,
        # Alert settings
        "test_failure_alerts": True,
        "daily_digest": False,
        "weekly_report": True,
        "alert_threshold": 80,
    }


def get_default_test_defaults() -> dict:
    """Get default test execution settings."""
    return {
        "default_browser": "chromium",
        "default_timeout": 30000,
        "parallel_execution": True,
        "retry_failed_tests": True,
        "max_retries": 2,
        "screenshot_on_failure": True,
        "video_recording": False,
    }


def get_default_discovery_preferences() -> dict:
    """Get default discovery preferences."""
    return {
        "mode": "standard",
        "strategy": "bfs",
        "maxPages": 50,
        "maxDepth": 3,
        "includePatterns": "",
        "excludePatterns": "/api/*, /static/*, *.pdf, *.jpg, *.png",
        "focusAreas": [],
        "captureScreenshots": True,
        "useVisionAi": False,
    }


async def get_or_create_profile(user_id: str, email: str | None = None) -> dict:
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
    now = datetime.now(UTC).isoformat()

    new_profile = {
        "user_id": user_id,
        "email": email,
        "notification_preferences": get_default_notification_preferences(),
        "test_defaults": get_default_test_defaults(),
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
        {"last_active_at": datetime.now(UTC).isoformat()}
    )

    # Parse notification preferences, test defaults, and discovery preferences
    notification_prefs = profile.get("notification_preferences") or get_default_notification_preferences()
    test_defaults = profile.get("test_defaults") or get_default_test_defaults()
    discovery_prefs = profile.get("discovery_preferences") or get_default_discovery_preferences()

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
        # Professional fields
        job_title=profile.get("job_title"),
        company=profile.get("company"),
        department=profile.get("department"),
        phone=profile.get("phone"),
        # Social links
        github_username=profile.get("github_username"),
        linkedin_url=profile.get("linkedin_url"),
        twitter_handle=profile.get("twitter_handle"),
        website_url=profile.get("website_url"),
        # Preferences
        notification_preferences=NotificationPreferences(**notification_prefs),
        test_defaults=TestDefaults(**test_defaults),
        discovery_preferences=DiscoveryPreferences(**discovery_prefs),
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
    """Update the current user's profile (full update)."""
    user = await get_current_user(request)

    # Ensure profile exists
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    supabase = get_supabase_client()

    # Build update data
    update_data = {"updated_at": datetime.now(UTC).isoformat()}

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
    # Professional fields
    if body.job_title is not None:
        update_data["job_title"] = body.job_title
    if body.company is not None:
        update_data["company"] = body.company
    if body.department is not None:
        update_data["department"] = body.department
    if body.phone is not None:
        update_data["phone"] = body.phone
    # Social links
    if body.github_username is not None:
        update_data["github_username"] = body.github_username
    if body.linkedin_url is not None:
        update_data["linkedin_url"] = body.linkedin_url
    if body.twitter_handle is not None:
        update_data["twitter_handle"] = body.twitter_handle
    if body.website_url is not None:
        update_data["website_url"] = body.website_url

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


@router.patch("/me", response_model=UserProfileResponse)
async def patch_my_profile(body: UpdateProfileRequest, request: Request):
    """Update the current user's profile (partial update).

    This endpoint allows partial updates - only the fields provided will be updated.
    Omitted fields will remain unchanged.
    """
    user = await get_current_user(request)

    # Ensure profile exists
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    supabase = get_supabase_client()

    # Build update data - only include fields that were explicitly provided
    update_data = {"updated_at": datetime.now(UTC).isoformat()}

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
    # Professional fields
    if body.job_title is not None:
        update_data["job_title"] = body.job_title
    if body.company is not None:
        update_data["company"] = body.company
    if body.department is not None:
        update_data["department"] = body.department
    if body.phone is not None:
        update_data["phone"] = body.phone
    # Social links
    if body.github_username is not None:
        update_data["github_username"] = body.github_username
    if body.linkedin_url is not None:
        update_data["linkedin_url"] = body.linkedin_url
    if body.twitter_handle is not None:
        update_data["twitter_handle"] = body.twitter_handle
    if body.website_url is not None:
        update_data["website_url"] = body.website_url

    result = await supabase.update(
        "user_profiles",
        {"id": f"eq.{profile['id']}"},
        update_data
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to update profile")

    logger.info("User profile patched", user_id=user["user_id"])

    # Return updated profile
    return await get_my_profile(request)


@router.get("/me/preferences", response_model=UserPreferencesResponse)
async def get_my_preferences(request: Request):
    """Get the current user's preferences.

    Returns notification preferences, test defaults, theme, timezone, and language.
    """
    user = await get_current_user(request)

    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    # Parse notification preferences and test defaults
    notification_prefs = profile.get("notification_preferences") or get_default_notification_preferences()
    test_defaults = profile.get("test_defaults") or get_default_test_defaults()

    return UserPreferencesResponse(
        notification_preferences=NotificationPreferences(**notification_prefs),
        test_defaults=TestDefaults(**test_defaults),
        theme=profile.get("theme"),
        timezone=profile.get("timezone"),
        language=profile.get("language"),
    )


@router.put("/me/preferences", response_model=UserPreferencesResponse)
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

    # Merge updates - Master toggles
    if body.email_notifications is not None:
        current_prefs["email_notifications"] = body.email_notifications
    if body.slack_notifications is not None:
        current_prefs["slack_notifications"] = body.slack_notifications
    if body.in_app_notifications is not None:
        current_prefs["in_app_notifications"] = body.in_app_notifications

    # Email-specific settings
    if body.email_test_failures is not None:
        current_prefs["email_test_failures"] = body.email_test_failures
    if body.email_test_completions is not None:
        current_prefs["email_test_completions"] = body.email_test_completions
    if body.email_weekly_digest is not None:
        current_prefs["email_weekly_digest"] = body.email_weekly_digest

    # Slack-specific settings
    if body.slack_test_failures is not None:
        current_prefs["slack_test_failures"] = body.slack_test_failures
    if body.slack_test_completions is not None:
        current_prefs["slack_test_completions"] = body.slack_test_completions

    # In-app specific settings
    if body.in_app_test_failures is not None:
        current_prefs["in_app_test_failures"] = body.in_app_test_failures
    if body.in_app_test_completions is not None:
        current_prefs["in_app_test_completions"] = body.in_app_test_completions

    # Alert settings
    if body.test_failure_alerts is not None:
        current_prefs["test_failure_alerts"] = body.test_failure_alerts
    if body.daily_digest is not None:
        current_prefs["daily_digest"] = body.daily_digest
    if body.weekly_report is not None:
        current_prefs["weekly_report"] = body.weekly_report
    if body.alert_threshold is not None:
        current_prefs["alert_threshold"] = body.alert_threshold

    result = await supabase.update(
        "user_profiles",
        {"id": f"eq.{profile['id']}"},
        {
            "notification_preferences": current_prefs,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to update preferences")

    logger.info("User notification preferences updated", user_id=user["user_id"])

    # Return updated preferences
    return await get_my_preferences(request)


@router.put("/me/test-defaults", response_model=UserProfileResponse)
async def update_test_defaults(
    body: UpdateTestDefaultsRequest,
    request: Request
):
    """Update test execution defaults."""
    user = await get_current_user(request)

    # Ensure profile exists
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    supabase = get_supabase_client()

    # Get current test defaults
    current_defaults = profile.get("test_defaults") or get_default_test_defaults()

    # Merge updates
    if body.default_browser is not None:
        current_defaults["default_browser"] = body.default_browser
    if body.default_timeout is not None:
        current_defaults["default_timeout"] = body.default_timeout
    if body.parallel_execution is not None:
        current_defaults["parallel_execution"] = body.parallel_execution
    if body.retry_failed_tests is not None:
        current_defaults["retry_failed_tests"] = body.retry_failed_tests
    if body.max_retries is not None:
        current_defaults["max_retries"] = body.max_retries
    if body.screenshot_on_failure is not None:
        current_defaults["screenshot_on_failure"] = body.screenshot_on_failure
    if body.video_recording is not None:
        current_defaults["video_recording"] = body.video_recording

    result = await supabase.update(
        "user_profiles",
        {"id": f"eq.{profile['id']}"},
        {
            "test_defaults": current_defaults,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to update test defaults")

    logger.info("User test defaults updated", user_id=user["user_id"])

    # Return updated profile
    return await get_my_profile(request)


@router.put("/me/discovery-preferences", response_model=UserProfileResponse)
async def update_discovery_preferences(
    body: UpdateDiscoveryPreferencesRequest,
    request: Request
):
    """Update discovery preferences."""
    user = await get_current_user(request)

    # Ensure profile exists
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    supabase = get_supabase_client()

    # Get current discovery preferences
    current_prefs = profile.get("discovery_preferences") or get_default_discovery_preferences()

    # Merge updates
    if body.mode is not None:
        current_prefs["mode"] = body.mode
    if body.strategy is not None:
        current_prefs["strategy"] = body.strategy
    if body.maxPages is not None:
        current_prefs["maxPages"] = body.maxPages
    if body.maxDepth is not None:
        current_prefs["maxDepth"] = body.maxDepth
    if body.includePatterns is not None:
        current_prefs["includePatterns"] = body.includePatterns
    if body.excludePatterns is not None:
        current_prefs["excludePatterns"] = body.excludePatterns
    if body.focusAreas is not None:
        current_prefs["focusAreas"] = body.focusAreas
    if body.captureScreenshots is not None:
        current_prefs["captureScreenshots"] = body.captureScreenshots
    if body.useVisionAi is not None:
        current_prefs["useVisionAi"] = body.useVisionAi
    if body.authRequired is not None:
        current_prefs["authRequired"] = body.authRequired
    if body.authConfig is not None:
        current_prefs["authConfig"] = body.authConfig

    result = await supabase.update(
        "user_profiles",
        {"id": f"eq.{profile['id']}"},
        {
            "discovery_preferences": current_prefs,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to update discovery preferences")

    logger.info("User discovery preferences updated", user_id=user["user_id"])

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
        "updated_at": datetime.now(UTC).isoformat(),
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


async def _create_personal_organization(
    user_id: str,
    user_email: str | None = None
) -> Optional["OrganizationSummary"]:
    """Auto-create a personal organization for users without any org membership.

    This ensures every user has at least one organization to work in,
    providing a seamless experience for individual users.

    IMPORTANT: This function includes race condition protection:
    1. Application-level check for existing personal org
    2. Database trigger prevents duplicate inserts
    3. Unique constraint on slug prevents duplicate orgs

    Args:
        user_id: The user's ID from Clerk
        user_email: The user's email address (optional)

    Returns:
        OrganizationSummary for the new personal org, or None if creation fails
    """
    try:
        supabase = get_supabase_client()

        # RACE CONDITION PROTECTION: Check if user already has a personal org
        # This is the first line of defense (DB trigger is backup)
        existing_personal = await supabase.request(
            f"/organization_members?user_id=eq.{user_id}&status=eq.active"
            f"&select=organization_id,organizations!inner(id,name,slug,plan,is_personal)"
        )

        if existing_personal.get("data"):
            for membership in existing_personal["data"]:
                org = membership.get("organizations", {})
                if org.get("is_personal"):
                    logger.info(
                        "User already has personal organization, returning existing",
                        user_id=user_id,
                        org_id=org["id"]
                    )
                    # Return existing personal org instead of creating new one
                    return OrganizationSummary(
                        id=org["id"],
                        name=org["name"],
                        slug=org["slug"],
                        role="owner",
                        plan=org.get("plan", "free"),
                        member_count=1,
                        is_default=True,
                        is_personal=True,
                    )

        # Generate unique slug
        slug_base = "personal"
        if user_email:
            # Use email prefix for more meaningful slug
            email_prefix = user_email.split("@")[0][:20]
            slug_base = f"{email_prefix}-workspace"

        # Ensure slug uniqueness with random suffix
        slug = f"{slug_base}-{secrets.token_hex(4)}"

        # Determine org name
        if user_email:
            org_name = f"{user_email.split('@')[0]}'s Workspace"
        else:
            org_name = "Personal Workspace"

        # Create the organization
        org_result = await supabase.insert("organizations", {
            "name": org_name,
            "slug": slug,
            "plan": "free",
            "is_personal": True,  # Mark as personal/individual org
        })

        if org_result.get("error"):
            logger.error(
                "Failed to create personal organization",
                user_id=user_id,
                error=org_result.get("error")
            )
            return None

        org = org_result["data"][0]

        # Add user as owner
        member_result = await supabase.insert("organization_members", {
            "organization_id": org["id"],
            "user_id": user_id,
            "email": user_email or "",
            "role": "owner",
            "status": "active",
            "accepted_at": datetime.now(UTC).isoformat(),
        })

        if member_result.get("error"):
            logger.error(
                "Failed to add user to personal organization",
                user_id=user_id,
                org_id=org["id"],
                error=member_result.get("error")
            )
            # Clean up org if member creation failed
            await supabase.request(
                f"/organizations?id=eq.{org['id']}",
                method="DELETE"
            )
            return None

        # Create default healing config for the org
        await supabase.insert("self_healing_config", {
            "organization_id": org["id"],
        })

        # Update user profile with default org
        profile = await get_or_create_profile(user_id, user_email)
        if profile:
            await supabase.update(
                "user_profiles",
                {"id": f"eq.{profile['id']}"},
                {"default_organization_id": org["id"]}
            )

        logger.info(
            "Personal organization created",
            user_id=user_id,
            org_id=org["id"],
            org_name=org_name
        )

        return OrganizationSummary(
            id=org["id"],
            name=org["name"],
            slug=org["slug"],
            role="owner",
            plan="free",
            member_count=1,
            is_default=True,
            is_personal=True,
        )

    except Exception as e:
        logger.error(
            "Exception creating personal organization",
            user_id=user_id,
            error=str(e)
        )
        return None


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
        # Auto-create personal organization for users without any org
        personal_org = await _create_personal_organization(
            user_id=user["user_id"],
            user_email=user.get("email")
        )
        if personal_org:
            return [personal_org]
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
            is_personal=org.get("is_personal", False),
        ))

    # Sort with default org first, then by name
    result.sort(key=lambda x: (not x.is_default, x.name.lower()))

    return result


class SwitchOrganizationResponse(BaseModel):
    """Response for organization switch."""
    success: bool
    organization_id: str
    organization_name: str
    message: str


@router.post("/me/organizations/{org_id}/switch", response_model=SwitchOrganizationResponse)
async def switch_organization(org_id: str, request: Request):
    """Switch the user's active organization.

    This endpoint updates the user's default organization and is intended
    to be called when the user switches organizations in the UI.
    """
    user = await get_current_user(request)

    # Ensure profile exists
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    supabase = get_supabase_client()

    # Verify user has access to the organization
    membership = await supabase.request(
        f"/organization_members?organization_id=eq.{org_id}&user_id=eq.{user['user_id']}&status=eq.active&select=id,role"
    )

    # Also check by email if not found by user_id
    if (not membership.get("data") or len(membership["data"]) == 0) and user.get("email"):
        membership = await supabase.request(
            f"/organization_members?organization_id=eq.{org_id}&email=eq.{user['email']}&status=eq.active&select=id,role"
        )

    if not membership.get("data") or len(membership["data"]) == 0:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this organization"
        )

    # Get organization details
    org_result = await supabase.request(
        f"/organizations?id=eq.{org_id}&select=id,name,slug"
    )

    if not org_result.get("data") or len(org_result["data"]) == 0:
        raise HTTPException(
            status_code=404,
            detail="Organization not found"
        )

    org = org_result["data"][0]

    # Update profile with new default organization
    result = await supabase.update(
        "user_profiles",
        {"id": f"eq.{profile['id']}"},
        {
            "default_organization_id": org_id,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to switch organization")

    logger.info(
        "User switched organization",
        user_id=user["user_id"],
        organization_id=org_id,
        organization_name=org["name"]
    )

    return SwitchOrganizationResponse(
        success=True,
        organization_id=org_id,
        organization_name=org["name"],
        message=f"Successfully switched to {org['name']}"
    )


# ============================================================================
# Avatar Upload
# ============================================================================

class AvatarUploadResponse(BaseModel):
    """Response for avatar upload."""
    success: bool
    avatar_url: str
    message: str


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_AVATAR_SIZE = 5 * 1024 * 1024  # 5MB


@router.post("/me/avatar", response_model=AvatarUploadResponse)
async def upload_avatar(
    request: Request,
    file: UploadFile = File(...),
):
    """Upload user avatar to Supabase Storage.

    Validates file type and size, then uploads to the avatars bucket.
    The avatar URL is automatically updated in the user's profile.

    Supported formats: JPEG, PNG, WebP, GIF
    Max size: 5MB
    """
    user = await get_current_user(request)

    # Validate file type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: JPEG, PNG, WebP, GIF"
        )

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > MAX_AVATAR_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is 5MB"
        )

    # Ensure profile exists
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    supabase = get_supabase_client()

    # Generate unique filename
    file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    filename = f"{user['user_id']}/{uuid.uuid4()}.{file_ext}"

    # Upload to Supabase Storage
    try:
        # Use Supabase Storage API
        storage_url = f"{settings.SUPABASE_URL}/storage/v1/object/avatars/{filename}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                storage_url,
                content=content,
                headers={
                    "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
                    "Content-Type": file.content_type,
                },
            )

            if response.status_code not in (200, 201):
                logger.error(
                    "Failed to upload avatar to storage",
                    user_id=user["user_id"],
                    status_code=response.status_code,
                    response=response.text
                )
                raise HTTPException(
                    status_code=500,
                    detail="Failed to upload avatar"
                )

        # Build public URL
        avatar_url = f"{settings.SUPABASE_URL}/storage/v1/object/public/avatars/{filename}"

        # Update user profile with new avatar URL
        result = await supabase.update(
            "user_profiles",
            {"id": f"eq.{profile['id']}"},
            {
                "avatar_url": avatar_url,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to update profile")

        logger.info(
            "Avatar uploaded successfully",
            user_id=user["user_id"],
            avatar_url=avatar_url
        )

        return AvatarUploadResponse(
            success=True,
            avatar_url=avatar_url,
            message="Avatar uploaded successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Exception uploading avatar",
            user_id=user["user_id"],
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Failed to upload avatar")


# ============================================================================
# Account Activity
# ============================================================================

class AccountActivityResponse(BaseModel):
    """Response for account activity."""
    member_since: str
    last_login_at: str | None
    last_active_at: str | None
    login_count: int
    organizations_count: int
    api_keys_count: int
    api_requests_30d: int
    organizations: list[OrganizationSummary]


@router.get("/me/activity", response_model=AccountActivityResponse)
async def get_account_activity(request: Request):
    """Get user's account activity summary.

    Returns account statistics including login count, API usage,
    and organization memberships.
    """
    user = await get_current_user(request)

    # Ensure profile exists
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    supabase = get_supabase_client()

    # Get organizations
    orgs = await list_my_organizations(request)

    # Get API keys count
    api_keys_result = await supabase.request(
        f"/api_keys?user_id=eq.{user['user_id']}&is_active=eq.true&select=id"
    )
    api_keys_count = len(api_keys_result.get("data", []))

    # Get API requests in last 30 days (from ai_usage_logs if available)
    # This is a simplified count - in production you'd query a proper usage table
    api_requests_30d = 0
    try:
        usage_result = await supabase.request(
            f"/ai_usage_logs?user_id=eq.{user['user_id']}"
            f"&created_at=gte.{(datetime.now(UTC).replace(day=1)).isoformat()}"
            f"&select=id"
        )
        api_requests_30d = len(usage_result.get("data", []))
    except Exception:
        # Table might not exist, that's okay
        pass

    return AccountActivityResponse(
        member_since=profile["created_at"],
        last_login_at=profile.get("last_login_at"),
        last_active_at=profile.get("last_active_at"),
        login_count=profile.get("login_count", 0),
        organizations_count=len(orgs),
        api_keys_count=api_keys_count,
        api_requests_30d=api_requests_30d,
        organizations=orgs,
    )


# ============================================================================
# Connected Accounts
# ============================================================================

class ConnectedAccount(BaseModel):
    """A connected OAuth provider."""
    provider: str
    provider_name: str
    email: str | None
    connected_at: str | None


class ConnectedAccountsResponse(BaseModel):
    """Response for connected accounts."""
    accounts: list[ConnectedAccount]
    api_keys_active: int
    api_keys_total: int


@router.get("/me/connected-accounts", response_model=ConnectedAccountsResponse)
async def get_connected_accounts(request: Request):
    """Get OAuth providers linked to user account.

    Returns list of connected OAuth providers (via Clerk) and API key summary.
    Note: OAuth provider info comes from Clerk user metadata.
    """
    user = await get_current_user(request)

    supabase = get_supabase_client()

    # Get API keys summary
    api_keys_result = await supabase.request(
        f"/api_keys?user_id=eq.{user['user_id']}&select=id,is_active"
    )
    api_keys = api_keys_result.get("data", [])
    api_keys_active = sum(1 for k in api_keys if k.get("is_active"))
    api_keys_total = len(api_keys)

    # Build connected accounts from Clerk user data
    # Clerk provides external accounts in user metadata
    accounts: list[ConnectedAccount] = []

    # The user dict from get_current_user typically includes provider info
    # In a real implementation, you'd query Clerk API for external accounts
    # For now, we infer from the email domain or auth method

    user_email = user.get("email", "")

    # Primary email is always "connected"
    if user_email:
        # Detect provider from email domain
        provider = "email"
        provider_name = "Email"

        if "gmail.com" in user_email or "googlemail.com" in user_email:
            provider = "google"
            provider_name = "Google"
        elif "github" in user.get("user_id", "").lower():
            provider = "github"
            provider_name = "GitHub"

        accounts.append(ConnectedAccount(
            provider=provider,
            provider_name=provider_name,
            email=user_email,
            connected_at=None,  # Would come from Clerk API
        ))

    # Note: For full OAuth provider listing, integrate with Clerk Admin API
    # GET https://api.clerk.dev/v1/users/{user_id}
    # The external_accounts field contains all linked providers

    return ConnectedAccountsResponse(
        accounts=accounts,
        api_keys_active=api_keys_active,
        api_keys_total=api_keys_total,
    )
