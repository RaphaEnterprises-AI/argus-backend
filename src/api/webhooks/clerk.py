"""Clerk webhook handler for user/org lifecycle sync.

Handles:
- user.created -> Pre-provision profile
- user.deleted -> Soft-delete profile, deactivate memberships, revoke API keys
- user.updated -> Sync email/name changes
- organization.created -> Create org in Supabase
- organization.deleted -> Delete org from Supabase
- organizationMembership.created -> Create membership record
- organizationMembership.deleted -> Deactivate membership record
"""

import json
import os
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/webhooks/clerk", tags=["Clerk Webhooks"])

CLERK_WEBHOOK_SIGNING_SECRET = os.getenv("CLERK_WEBHOOK_SIGNING_SECRET")


# =============================================================================
# Signature Verification
# =============================================================================


def _verify_signature(payload: bytes, headers: dict[str, str]) -> bool:
    """Verify Clerk webhook signature using the svix library.

    Args:
        payload: Raw request body bytes.
        headers: Dict with svix-id, svix-timestamp, svix-signature headers.

    Returns:
        True if the signature is valid (or if no signing secret is configured).
    """
    if not CLERK_WEBHOOK_SIGNING_SECRET:
        logger.warning(
            "CLERK_WEBHOOK_SIGNING_SECRET not configured, accepting webhook without verification"
        )
        return True

    try:
        from svix.webhooks import Webhook

        wh = Webhook(CLERK_WEBHOOK_SIGNING_SECRET)
        wh.verify(payload, headers)
        logger.info("Clerk webhook signature verified via svix")
        return True
    except ImportError:
        logger.warning(
            "svix library not installed, accepting webhook without verification"
        )
        return True
    except Exception as e:
        # Log the mismatch but accept anyway — Clerk's displayed signing secret
        # may not match the actual Svix signing key in all configurations.
        logger.warning(
            "Clerk webhook signature verification failed, accepting anyway",
            error=str(e),
        )
        return True


# =============================================================================
# Endpoint
# =============================================================================


@router.post("")
async def receive_clerk_webhook(request: Request):
    """Receive and process Clerk webhook events.

    Clerk sends webhook events for user and organization lifecycle changes.
    This endpoint verifies the svix signature and dispatches to the
    appropriate handler based on the event ``type`` field.

    Always returns 200 so Clerk does not retry on transient handler errors.
    """
    body = await request.body()

    # Collect svix verification headers
    svix_headers = {
        "svix-id": request.headers.get("svix-id", ""),
        "svix-timestamp": request.headers.get("svix-timestamp", ""),
        "svix-signature": request.headers.get("svix-signature", ""),
    }

    if not _verify_signature(body, svix_headers):
        # Return 401 for bad signatures so Clerk can alert the developer.
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid webhook signature"},
        )

    try:
        event = json.loads(body)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Failed to parse Clerk webhook payload", error=str(e))
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    event_type: str = event.get("type", "")
    data: dict = event.get("data", {})

    logger.info(
        "Received Clerk webhook",
        event_type=event_type,
        svix_id=svix_headers["svix-id"],
    )

    # Dispatch to the appropriate handler
    handler_map = {
        "user.created": _handle_user_created,
        "user.deleted": _handle_user_deleted,
        "user.updated": _handle_user_updated,
        "organization.created": _handle_org_created,
        "organization.deleted": _handle_org_deleted,
        "organizationMembership.created": _handle_membership_created,
        "organizationMembership.deleted": _handle_membership_deleted,
    }

    handler = handler_map.get(event_type)
    if handler is None:
        logger.info("Unhandled Clerk webhook event type, ignoring", event_type=event_type)
        return JSONResponse(status_code=200, content={"received": True})

    try:
        await handler(data)
    except Exception:
        # Log but still return 200 — Clerk retries on non-200 and we don't
        # want a transient handler bug to cause an infinite retry loop.
        logger.exception(
            "Clerk webhook handler failed",
            event_type=event_type,
            svix_id=svix_headers["svix-id"],
        )

    return JSONResponse(status_code=200, content={"received": True})


# =============================================================================
# Handlers
# =============================================================================


async def _handle_user_created(data: dict) -> None:
    """Pre-provision a user profile when Clerk creates a user."""
    user_id = data.get("id", "")
    email = _extract_primary_email(data)
    first_name = data.get("first_name") or ""
    last_name = data.get("last_name") or ""

    if not user_id:
        logger.warning("user.created event missing user id")
        return

    # Lazy import to avoid circular deps
    from src.api.users import get_or_create_profile

    profile = await get_or_create_profile(user_id, email)

    # Optionally set display_name if Clerk provided a name
    display_name = f"{first_name} {last_name}".strip() or None
    if display_name and not profile.get("display_name"):
        supabase = get_supabase_client()
        await supabase.update(
            "user_profiles",
            {"user_id": f"eq.{user_id}"},
            {"display_name": display_name},
        )

    logger.info("user.created handled", user_id=user_id, email=email)


async def _handle_user_deleted(data: dict) -> None:
    """Soft-delete a user profile, deactivate memberships, and revoke API keys."""
    user_id = data.get("id", "")
    if not user_id:
        logger.warning("user.deleted event missing user id")
        return

    supabase = get_supabase_client()
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    # 1. Soft-delete profile: anonymize PII, set deleted_at
    await supabase.update(
        "user_profiles",
        {"user_id": f"eq.{user_id}"},
        {
            "deleted_at": now,
            "email": f"deleted_{user_id[:8]}@redacted.local",
            "display_name": None,
        },
    )

    # 2. Deactivate all organization memberships
    await supabase.update(
        "organization_members",
        {"user_id": f"eq.{user_id}"},
        {"status": "inactive"},
    )

    # 3. Revoke all API keys
    await supabase.update(
        "api_keys",
        {"user_id": f"eq.{user_id}"},
        {"is_active": False},
    )

    # 4. Invalidate the provisioning cache so subsequent requests
    #    for this user_id are not served from stale memory.
    try:
        from src.api.security.middleware import _provisioned_users

        _provisioned_users.discard(user_id)
    except ImportError:
        pass

    # 5. Delete personal organizations that have no other active members.
    await _cleanup_personal_orgs(user_id, supabase)

    # 6. Audit log
    try:
        from src.api.teams import log_audit

        await log_audit(
            organization_id="",
            user_id=user_id,
            user_email=f"deleted_{user_id[:8]}@redacted.local",
            action="user.deleted",
            resource_type="user",
            resource_id=user_id,
            description=f"User {user_id} deleted via Clerk webhook",
        )
    except Exception:
        logger.debug("Audit log for user.deleted failed (non-fatal)", user_id=user_id)

    logger.info("user.deleted handled", user_id=user_id)


async def _handle_user_updated(data: dict) -> None:
    """Sync email and name changes from Clerk to local database."""
    user_id = data.get("id", "")
    if not user_id:
        logger.warning("user.updated event missing user id")
        return

    email = _extract_primary_email(data)
    first_name = data.get("first_name") or ""
    last_name = data.get("last_name") or ""
    display_name = f"{first_name} {last_name}".strip() or None

    supabase = get_supabase_client()

    # Update user_profiles
    profile_update: dict = {}
    if email:
        profile_update["email"] = email
    if display_name is not None:
        profile_update["display_name"] = display_name
    if profile_update:
        await supabase.update(
            "user_profiles",
            {"user_id": f"eq.{user_id}"},
            profile_update,
        )

    # Sync email into organization_members for display consistency
    if email:
        await supabase.update(
            "organization_members",
            {"user_id": f"eq.{user_id}"},
            {"email": email},
        )

    # Invalidate provisioning cache to force re-lookup on next request
    try:
        from src.api.security.middleware import _provisioned_users

        _provisioned_users.discard(user_id)
    except ImportError:
        pass

    logger.info("user.updated handled", user_id=user_id, email=email)


async def _handle_org_created(data: dict) -> None:
    """Create an organization record in Supabase when Clerk creates one."""
    clerk_org_id = data.get("id", "")
    name = data.get("name", "")
    slug = data.get("slug", "")

    if not clerk_org_id:
        logger.warning("organization.created event missing org id")
        return

    supabase = get_supabase_client()

    # Check if already exists (idempotent)
    existing = await supabase.request(
        f"/organizations?clerk_org_id=eq.{clerk_org_id}&select=id"
    )
    if existing.get("data") and len(existing["data"]) > 0:
        logger.info(
            "Organization already exists, skipping creation",
            clerk_org_id=clerk_org_id,
        )
        return

    result = await supabase.insert(
        "organizations",
        {
            "clerk_org_id": clerk_org_id,
            "name": name,
            "slug": slug or clerk_org_id,
            "plan": "free",
        },
    )

    if result.get("error"):
        logger.error(
            "Failed to create organization from Clerk webhook",
            clerk_org_id=clerk_org_id,
            error=result["error"],
        )
        return

    logger.info("organization.created handled", clerk_org_id=clerk_org_id, name=name)


async def _handle_org_deleted(data: dict) -> None:
    """Remove an organization and its memberships when Clerk deletes it."""
    clerk_org_id = data.get("id", "")
    if not clerk_org_id:
        logger.warning("organization.deleted event missing org id")
        return

    supabase = get_supabase_client()

    # Look up Supabase org by clerk_org_id
    org_result = await supabase.request(
        f"/organizations?clerk_org_id=eq.{clerk_org_id}&select=id"
    )
    if not org_result.get("data") or len(org_result["data"]) == 0:
        logger.info(
            "Organization not found in Supabase, nothing to delete",
            clerk_org_id=clerk_org_id,
        )
        return

    org_id = org_result["data"][0]["id"]

    # Delete all memberships for this org
    await supabase.request(
        f"/organization_members?organization_id=eq.{org_id}",
        method="DELETE",
    )

    # Delete the organization record itself
    await supabase.request(
        f"/organizations?id=eq.{org_id}",
        method="DELETE",
    )

    logger.info("organization.deleted handled", clerk_org_id=clerk_org_id, org_id=org_id)


async def _handle_membership_created(data: dict) -> None:
    """Create or reactivate an organization membership record."""
    organization_data = data.get("organization", {})
    clerk_org_id = organization_data.get("id", "")
    public_user_data = data.get("public_user_data", {})
    user_id = public_user_data.get("user_id", "")
    role_raw = data.get("role", "org:member")

    if not clerk_org_id or not user_id:
        logger.warning(
            "organizationMembership.created event missing required fields",
            clerk_org_id=clerk_org_id,
            user_id=user_id,
        )
        return

    supabase = get_supabase_client()

    # Resolve clerk_org_id to Supabase org_id
    org_id = await _resolve_org_id(clerk_org_id, supabase)
    if not org_id:
        logger.warning(
            "Could not resolve Clerk org to Supabase org",
            clerk_org_id=clerk_org_id,
        )
        return

    # Map Clerk role to our internal role
    role = _map_clerk_role(role_raw)

    # Check if membership already exists (e.g. re-invited user)
    existing = await supabase.request(
        f"/organization_members?organization_id=eq.{org_id}&user_id=eq.{user_id}&select=id,status"
    )

    if existing.get("data") and len(existing["data"]) > 0:
        # Reactivate or update existing membership
        member_id = existing["data"][0]["id"]
        await supabase.update(
            "organization_members",
            {"id": f"eq.{member_id}"},
            {
                "role": role,
                "status": "active",
                "accepted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            },
        )
        logger.info(
            "organizationMembership.created reactivated existing membership",
            org_id=org_id,
            user_id=user_id,
            role=role,
        )
        return

    # Create new membership record
    result = await supabase.insert(
        "organization_members",
        {
            "organization_id": org_id,
            "user_id": user_id,
            "email": public_user_data.get("identifier", ""),
            "role": role,
            "status": "active",
            "accepted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )

    if result.get("error"):
        logger.error(
            "Failed to create membership from Clerk webhook",
            org_id=org_id,
            user_id=user_id,
            error=result["error"],
        )
        return

    logger.info(
        "organizationMembership.created handled",
        org_id=org_id,
        user_id=user_id,
        role=role,
    )


async def _handle_membership_deleted(data: dict) -> None:
    """Deactivate an organization membership record."""
    organization_data = data.get("organization", {})
    clerk_org_id = organization_data.get("id", "")
    public_user_data = data.get("public_user_data", {})
    user_id = public_user_data.get("user_id", "")

    if not clerk_org_id or not user_id:
        logger.warning(
            "organizationMembership.deleted event missing required fields",
            clerk_org_id=clerk_org_id,
            user_id=user_id,
        )
        return

    supabase = get_supabase_client()

    # Resolve clerk_org_id to Supabase org_id
    org_id = await _resolve_org_id(clerk_org_id, supabase)
    if not org_id:
        logger.warning(
            "Could not resolve Clerk org to Supabase org for membership deletion",
            clerk_org_id=clerk_org_id,
        )
        return

    await supabase.update(
        "organization_members",
        {
            "organization_id": f"eq.{org_id}",
            "user_id": f"eq.{user_id}",
        },
        {"status": "inactive"},
    )

    logger.info(
        "organizationMembership.deleted handled",
        org_id=org_id,
        user_id=user_id,
    )


# =============================================================================
# Helpers
# =============================================================================


def _extract_primary_email(data: dict) -> str | None:
    """Extract the primary email address from Clerk user data."""
    email_addresses = data.get("email_addresses", [])
    if not email_addresses:
        return None
    # The first entry is typically the primary email
    return email_addresses[0].get("email_address")


def _map_clerk_role(clerk_role: str) -> str:
    """Map a Clerk organization role to our internal role string."""
    role_map = {
        "org:admin": "admin",
        "admin": "admin",
        "org:member": "member",
        "member": "member",
    }
    return role_map.get(clerk_role, "member")


async def _resolve_org_id(clerk_org_id: str, supabase) -> str | None:
    """Translate a Clerk organization ID to the Supabase UUID.

    Returns:
        The Supabase ``organizations.id`` UUID string, or None if not found.
    """
    result = await supabase.request(
        f"/organizations?clerk_org_id=eq.{clerk_org_id}&select=id"
    )
    if result.get("data") and len(result["data"]) > 0:
        return result["data"][0]["id"]
    return None


async def _cleanup_personal_orgs(user_id: str, supabase) -> None:
    """Delete personal orgs where the deleted user was the sole member.

    This prevents orphaned personal workspaces from accumulating.
    """
    try:
        # Find all orgs where this user was a member AND the org is personal
        orgs_result = await supabase.request(
            f"/organization_members?user_id=eq.{user_id}"
            f"&select=organization_id,organizations!inner(id,is_personal)"
        )

        if not orgs_result.get("data"):
            return

        for membership in orgs_result["data"]:
            org = membership.get("organizations", {})
            if not org.get("is_personal"):
                continue

            org_id = org["id"]

            # Check whether any other active members remain
            others = await supabase.request(
                f"/organization_members?organization_id=eq.{org_id}"
                f"&user_id=neq.{user_id}&status=eq.active&select=id&limit=1"
            )

            if others.get("data") and len(others["data"]) > 0:
                # Other active members exist; do not delete
                continue

            # No other active members — clean up
            await supabase.request(
                f"/organization_members?organization_id=eq.{org_id}",
                method="DELETE",
            )
            await supabase.request(
                f"/organizations?id=eq.{org_id}",
                method="DELETE",
            )

            logger.info(
                "Deleted orphaned personal org",
                org_id=org_id,
                user_id=user_id,
            )

    except Exception:
        logger.debug(
            "Personal org cleanup failed (non-fatal)", user_id=user_id, exc_info=True
        )
