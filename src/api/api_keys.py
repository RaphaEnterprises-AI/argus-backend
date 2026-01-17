"""API Key Management endpoints.

Provides endpoints for:
- Creating API keys
- Rotating keys
- Revoking keys
- Listing keys
"""

import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
import structlog

from src.services.supabase_client import get_supabase_client
from src.api.teams import get_current_user, verify_org_access, log_audit, translate_clerk_org_id

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/api-keys", tags=["API Keys"])


# ============================================================================
# Constants
# ============================================================================

KEY_PREFIX = "argus_sk_"
KEY_LENGTH = 32  # 32 bytes = 64 hex chars


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateAPIKeyRequest(BaseModel):
    """Request to create a new API key."""
    name: str = Field(..., min_length=1, max_length=100)
    scopes: list[str] = Field(default=["read", "write"])
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)

    @field_validator("scopes")
    @classmethod
    def scopes_must_not_be_empty(cls, v: list[str]) -> list[str]:
        """SECURITY: Ensure scopes is not empty - empty scopes means no API access."""
        if not v:
            raise ValueError("At least one scope is required. Empty scopes would result in no API access.")
        return v


class APIKeyResponse(BaseModel):
    """API key response (without the secret)."""
    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    last_used_at: Optional[str]
    request_count: int
    expires_at: Optional[str]
    revoked_at: Optional[str]
    created_at: str
    is_active: bool


class APIKeyCreatedResponse(APIKeyResponse):
    """Response when creating a new API key (includes the secret)."""
    key: str  # Only shown once at creation


class RotateKeyResponse(BaseModel):
    """Response when rotating an API key."""
    old_key_id: str
    new_key: APIKeyCreatedResponse
    message: str


# ============================================================================
# Helper Functions
# ============================================================================

def generate_api_key() -> tuple[str, str]:
    """Generate a new API key and its hash.

    Returns:
        Tuple of (plaintext_key, key_hash)
    """
    # Generate random bytes
    random_bytes = secrets.token_hex(KEY_LENGTH)

    # Create the full key with prefix
    full_key = f"{KEY_PREFIX}{random_bytes}"

    # Create SHA-256 hash for storage
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()

    return full_key, key_hash


def hash_api_key(key: str) -> str:
    """Hash an API key for comparison."""
    return hashlib.sha256(key.encode()).hexdigest()


# ============================================================================
# Endpoints
# ============================================================================

async def check_api_key_auth(user: dict, supabase_org_id: str) -> bool:
    """Check if the user is authenticated via API key for the given organization.

    Compares the user's organization ID (from API key auth context) with the
    requested organization ID after translating both to Supabase UUIDs.

    Returns:
        True if authenticated via API key for this organization, False otherwise.
    """
    if "api_user" not in user.get("roles", []):
        return False

    user_org_id = user.get("organization_id")
    if not user_org_id:
        return False

    try:
        user_org_id_translated = await translate_clerk_org_id(
            user_org_id,
            user.get("user_id"),
            user.get("email")
        )
        return user_org_id_translated == supabase_org_id
    except HTTPException:
        return False


async def resolve_org_id(org_id: str, user: dict) -> str:
    """Resolve 'default' org_id to user's actual organization.

    If org_id is 'default', tries to find user's organization from:
    1. User's organization_id from auth context (Clerk)
    2. First organization user is a member of (by user_id)
    3. First organization user is a member of (by email)
    """
    if org_id != "default":
        return org_id

    # Try to get from user's auth context (set by Clerk)
    if user.get("organization_id"):
        return user["organization_id"]

    supabase = get_supabase_client()

    # Try looking up by user_id first
    if user.get("user_id"):
        result = await supabase.request(
            f"/organization_members?user_id=eq.{user['user_id']}&status=eq.active&select=organization_id&limit=1"
        )
        if result.get("data") and len(result["data"]) > 0:
            return result["data"][0]["organization_id"]

    # Fall back to looking up by email
    if user.get("email"):
        result = await supabase.request(
            f"/organization_members?email=eq.{user['email']}&status=eq.active&select=organization_id&limit=1"
        )
        if result.get("data") and len(result["data"]) > 0:
            return result["data"][0]["organization_id"]

    # No organization found - this will cause a 403 in verify_org_access
    return org_id


@router.get("/organizations/{org_id}/keys", response_model=list[APIKeyResponse])
async def list_api_keys(org_id: str, request: Request, include_revoked: bool = False):
    """List all API keys for an organization."""
    user = await get_current_user(request)

    # Resolve 'default' to actual organization ID
    resolved_org_id = await resolve_org_id(org_id, user)

    # Translate Clerk org ID to Supabase UUID if needed
    supabase_org_id = await translate_clerk_org_id(
        resolved_org_id,
        user.get("user_id"),
        user.get("email")
    )

    # For API key auth, the key's organization_id IS the authorization
    # Only verify org membership for JWT/session auth
    is_api_key_auth = await check_api_key_auth(user, supabase_org_id)
    if not is_api_key_auth:
        _, supabase_org_id = await verify_org_access(supabase_org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    query = f"/api_keys?organization_id=eq.{supabase_org_id}&select=*&order=created_at.desc"

    if not include_revoked:
        query += "&revoked_at=is.null"

    try:
        result = await supabase.request(query)
    except Exception as e:
        logger.error("Failed to fetch API keys", org_id=supabase_org_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch API keys")

    if result.get("error"):
        logger.error("Supabase error fetching API keys", org_id=supabase_org_id, error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch API keys")

    now = datetime.now(timezone.utc)

    return [
        APIKeyResponse(
            id=key["id"],
            name=key["name"],
            key_prefix=key["key_prefix"],
            scopes=key.get("scopes", ["read", "write"]),
            last_used_at=key.get("last_used_at"),
            request_count=key.get("request_count") or 0,
            expires_at=key.get("expires_at"),
            revoked_at=key.get("revoked_at"),
            created_at=key["created_at"],
            is_active=(
                key.get("revoked_at") is None and
                (key.get("expires_at") is None or
                 datetime.fromisoformat(key["expires_at"].replace("Z", "+00:00")) > now)
            ),
        )
        for key in result.get("data", [])
    ]


@router.post("/organizations/{org_id}/keys", response_model=APIKeyCreatedResponse)
async def create_api_key(org_id: str, body: CreateAPIKeyRequest, request: Request):
    """Create a new API key for the organization."""
    user = await get_current_user(request)

    # Resolve 'default' to actual organization ID
    resolved_org_id = await resolve_org_id(org_id, user)

    # Translate Clerk org ID to Supabase UUID if needed
    supabase_org_id = await translate_clerk_org_id(
        resolved_org_id,
        user.get("user_id"),
        user.get("email")
    )

    # For API key auth, the key's organization_id IS the authorization
    is_api_key_auth = await check_api_key_auth(user, supabase_org_id)
    if is_api_key_auth:
        # API key auth - use the user_id as created_by (it's the original key creator's member ID)
        member = {"id": user["user_id"]}
    else:
        member, supabase_org_id = await verify_org_access(supabase_org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Validate scopes
    # SECURITY: Require at least one scope - empty scopes means no access
    if not body.scopes:
        raise HTTPException(
            status_code=400,
            detail="At least one scope is required. Empty scopes would result in no API access."
        )

    valid_scopes = {"read", "write", "admin", "webhooks", "tests"}
    for scope in body.scopes:
        if scope not in valid_scopes:
            raise HTTPException(status_code=400, detail=f"Invalid scope: {scope}")

    # Generate the key
    plaintext_key, key_hash = generate_api_key()
    key_prefix = plaintext_key[:16]  # argus_sk_ + first 8 chars of random

    # Calculate expiration
    expires_at = None
    if body.expires_in_days:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)).isoformat()

    # Store the key
    key_result = await supabase.insert("api_keys", {
        "organization_id": supabase_org_id,
        "name": body.name,
        "key_hash": key_hash,
        "key_prefix": key_prefix,
        "scopes": body.scopes,
        "expires_at": expires_at,
        "created_by": member["id"],
    })

    if key_result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to create API key")

    key_data = key_result["data"][0]

    # Audit log
    await log_audit(
        organization_id=supabase_org_id,
        user_id=user["user_id"],
        user_email=user["email"],
        action="api_key.create",
        resource_type="api_key",
        resource_id=key_data["id"],
        description=f"Created API key '{body.name}'",
        metadata={"key_name": body.name, "scopes": body.scopes, "expires_in_days": body.expires_in_days},
        request=request,
    )

    logger.info("API key created", org_id=supabase_org_id, key_name=body.name)

    return APIKeyCreatedResponse(
        id=key_data["id"],
        name=key_data["name"],
        key_prefix=key_data["key_prefix"],
        key=plaintext_key,  # Only shown once!
        scopes=key_data.get("scopes", ["read", "write"]),
        last_used_at=None,
        request_count=0,
        expires_at=key_data.get("expires_at"),
        revoked_at=None,
        created_at=key_data["created_at"],
        is_active=True,
    )


@router.post("/organizations/{org_id}/keys/{key_id}/rotate", response_model=RotateKeyResponse)
async def rotate_api_key(org_id: str, key_id: str, request: Request):
    """Rotate an API key (revoke old, create new with same settings)."""
    user = await get_current_user(request)

    # Resolve 'default' to actual organization ID
    resolved_org_id = await resolve_org_id(org_id, user)

    # Translate Clerk org ID to Supabase UUID if needed
    supabase_org_id = await translate_clerk_org_id(
        resolved_org_id,
        user.get("user_id"),
        user.get("email")
    )

    # For API key auth, the key's organization_id IS the authorization
    is_api_key_auth = await check_api_key_auth(user, supabase_org_id)
    if is_api_key_auth:
        member = {"id": user["user_id"]}
    else:
        member, supabase_org_id = await verify_org_access(supabase_org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Get the existing key
    try:
        existing = await supabase.request(
            f"/api_keys?id=eq.{key_id}&organization_id=eq.{supabase_org_id}&select=*"
        )
    except Exception as e:
        logger.error("Failed to fetch API key for rotation", key_id=key_id, org_id=supabase_org_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch API key")

    if not existing.get("data") or len(existing["data"]) == 0:
        raise HTTPException(status_code=404, detail="API key not found")

    old_key = existing["data"][0]

    if old_key.get("revoked_at"):
        raise HTTPException(status_code=400, detail="Cannot rotate a revoked key")

    # SECURITY: Validate old key scopes before rotation
    # If old key has empty/missing scopes, default to ["read", "write"]
    # to ensure the rotated key is usable
    old_scopes = old_key.get("scopes")
    if not old_scopes or not isinstance(old_scopes, list) or len(old_scopes) == 0:
        logger.warning(
            "Rotating key with empty/invalid scopes, using default scopes",
            key_id=key_id,
            old_scopes=old_scopes
        )
        old_scopes = ["read", "write"]

    # Generate new key with same settings
    plaintext_key, key_hash = generate_api_key()
    key_prefix = plaintext_key[:16]

    # Create new key first, before revoking old one (to avoid partial failure state)
    try:
        new_key_result = await supabase.insert("api_keys", {
            "organization_id": supabase_org_id,
            "name": f"{old_key['name']} (rotated)",
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "scopes": old_scopes,
            "expires_at": old_key.get("expires_at"),
            "created_by": member["id"],
        })
    except Exception as e:
        logger.error("Failed to create new key during rotation", key_id=key_id, org_id=supabase_org_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create new key during rotation")

    if new_key_result.get("error"):
        logger.error("Supabase error creating new key", key_id=key_id, error=new_key_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create new key")

    if not new_key_result.get("data") or len(new_key_result["data"]) == 0:
        logger.error("No data returned when creating new key", key_id=key_id)
        raise HTTPException(status_code=500, detail="Failed to create new key - no data returned")

    new_key_data = new_key_result["data"][0]

    # Revoke old key only after new key is successfully created
    try:
        revoke_result = await supabase.update(
            "api_keys",
            {"id": f"eq.{key_id}"},
            {"revoked_at": datetime.now(timezone.utc).isoformat()}
        )
        if revoke_result.get("error"):
            logger.warning("Failed to revoke old key after rotation, but new key was created",
                         key_id=key_id, new_key_id=new_key_data["id"], error=revoke_result.get("error"))
    except Exception as e:
        logger.warning("Exception revoking old key after rotation, but new key was created",
                      key_id=key_id, new_key_id=new_key_data["id"], error=str(e))

    # Audit log
    await log_audit(
        organization_id=supabase_org_id,
        user_id=user["user_id"],
        user_email=user["email"],
        action="api_key.rotate",
        resource_type="api_key",
        resource_id=key_id,
        description=f"Rotated API key '{old_key['name']}'",
        metadata={"old_key_id": key_id, "new_key_id": new_key_data["id"]},
        request=request,
    )

    logger.info("API key rotated", org_id=supabase_org_id, old_key_id=key_id, new_key_id=new_key_data["id"])

    return RotateKeyResponse(
        old_key_id=key_id,
        new_key=APIKeyCreatedResponse(
            id=new_key_data["id"],
            name=new_key_data["name"],
            key_prefix=new_key_data["key_prefix"],
            key=plaintext_key,
            scopes=new_key_data.get("scopes", ["read", "write"]),
            last_used_at=None,
            request_count=0,
            expires_at=new_key_data.get("expires_at"),
            revoked_at=None,
            created_at=new_key_data["created_at"],
            is_active=True,
        ),
        message="Key rotated successfully. The old key has been revoked.",
    )


@router.delete("/organizations/{org_id}/keys/{key_id}")
async def revoke_api_key(org_id: str, key_id: str, request: Request):
    """Revoke an API key."""
    user = await get_current_user(request)

    # Resolve 'default' to actual organization ID
    resolved_org_id = await resolve_org_id(org_id, user)

    # Translate Clerk org ID to Supabase UUID if needed
    supabase_org_id = await translate_clerk_org_id(
        resolved_org_id,
        user.get("user_id"),
        user.get("email")
    )

    # For API key auth, the key's organization_id IS the authorization
    is_api_key_auth = await check_api_key_auth(user, supabase_org_id)
    if not is_api_key_auth:
        _, supabase_org_id = await verify_org_access(supabase_org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Get the key
    try:
        existing = await supabase.request(
            f"/api_keys?id=eq.{key_id}&organization_id=eq.{supabase_org_id}&select=*"
        )
    except Exception as e:
        logger.error("Failed to fetch API key for revocation", key_id=key_id, org_id=supabase_org_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch API key")

    if not existing.get("data") or len(existing["data"]) == 0:
        raise HTTPException(status_code=404, detail="API key not found")

    key_data = existing["data"][0]

    if key_data.get("revoked_at"):
        raise HTTPException(status_code=400, detail="Key is already revoked")

    # Revoke the key
    try:
        revoke_result = await supabase.update(
            "api_keys",
            {"id": f"eq.{key_id}"},
            {"revoked_at": datetime.now(timezone.utc).isoformat()}
        )
        if revoke_result.get("error"):
            logger.error("Supabase error revoking API key", key_id=key_id, error=revoke_result.get("error"))
            raise HTTPException(status_code=500, detail="Failed to revoke API key")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to revoke API key", key_id=key_id, org_id=supabase_org_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to revoke API key")

    # Audit log
    try:
        await log_audit(
            organization_id=supabase_org_id,
            user_id=user["user_id"],
            user_email=user["email"],
            action="api_key.revoke",
            resource_type="api_key",
            resource_id=key_id,
            description=f"Revoked API key '{key_data['name']}'",
            metadata={"key_name": key_data["name"]},
            request=request,
        )
    except Exception as e:
        # Don't fail the operation if audit logging fails
        logger.warning("Failed to create audit log for key revocation", key_id=key_id, error=str(e))

    logger.info("API key revoked", org_id=supabase_org_id, key_id=key_id)

    return {"success": True, "message": "API key revoked"}


# ============================================================================
# Key Verification (for auth middleware)
# ============================================================================

async def verify_api_key(api_key: str) -> Optional[dict]:
    """Verify an API key and return organization/scopes if valid.

    Returns:
        Dict with organization_id and scopes, or None if invalid
    """
    if not api_key or not api_key.startswith(KEY_PREFIX):
        return None

    key_hash = hash_api_key(api_key)

    supabase = get_supabase_client()

    result = await supabase.request(
        f"/api_keys?key_hash=eq.{key_hash}&revoked_at=is.null&select=*"
    )

    if not result.get("data"):
        return None

    key_data = result["data"][0]

    # Check expiration
    if key_data.get("expires_at"):
        expires_at = datetime.fromisoformat(key_data["expires_at"].replace("Z", "+00:00"))
        if expires_at < datetime.now(timezone.utc):
            return None

    # Update last used
    await supabase.update(
        "api_keys",
        {"id": f"eq.{key_data['id']}"},
        {
            "last_used_at": datetime.now(timezone.utc).isoformat(),
            "request_count": (key_data.get("request_count", 0) or 0) + 1,
        }
    )

    return {
        "key_id": key_data["id"],
        "organization_id": key_data["organization_id"],
        "scopes": key_data.get("scopes", ["read", "write"]),
        "name": key_data["name"],
    }
