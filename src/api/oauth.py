"""OAuth 2.0 Integration Endpoints.

Provides OAuth flows for:
- GitHub: OAuth App flow with scopes `repo`, `read:org`
- Slack: Bot installation with scopes `channels:read`, `chat:write`
- Jira: OAuth 2.0 (3LO) flow
- Linear: OAuth 2.0 flow

Security features:
- PKCE (Proof Key for Code Exchange) where supported
- Cryptographically secure state parameter
- Encrypted token storage (AES-256-GCM)
- Automatic cleanup of expired OAuth states
"""

import base64
import hashlib
import os
import secrets
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from urllib.parse import urlencode

import httpx
import structlog
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.api.teams import get_current_user, log_audit, translate_clerk_org_id
from src.config import get_settings
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/oauth", tags=["OAuth"])


# =============================================================================
# Constants and Configuration
# =============================================================================

class OAuthPlatform(str, Enum):
    """Supported OAuth platforms."""
    GITHUB = "github"
    SLACK = "slack"
    JIRA = "jira"
    LINEAR = "linear"


# OAuth endpoints for each platform
OAUTH_CONFIG = {
    OAuthPlatform.GITHUB: {
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "user_url": "https://api.github.com/user",
        "default_scopes": ["repo", "read:org"],
        "supports_pkce": False,  # GitHub doesn't support PKCE
    },
    OAuthPlatform.SLACK: {
        "authorize_url": "https://slack.com/oauth/v2/authorize",
        "token_url": "https://slack.com/api/oauth.v2.access",
        "user_url": "https://slack.com/api/auth.test",
        "default_scopes": ["channels:read", "chat:write", "users:read"],
        "supports_pkce": False,
    },
    OAuthPlatform.JIRA: {
        "authorize_url": "https://auth.atlassian.com/authorize",
        "token_url": "https://auth.atlassian.com/oauth/token",
        "user_url": "https://api.atlassian.com/me",
        "accessible_resources_url": "https://api.atlassian.com/oauth/token/accessible-resources",
        "default_scopes": [
            "read:jira-work",
            "write:jira-work",
            "read:jira-user",
            "offline_access",
        ],
        "supports_pkce": True,  # Jira recommends PKCE
        "audience": "api.atlassian.com",
    },
    OAuthPlatform.LINEAR: {
        "authorize_url": "https://linear.app/oauth/authorize",
        "token_url": "https://api.linear.app/oauth/token",
        "user_url": "https://api.linear.app/graphql",
        "default_scopes": ["read", "write", "issues:create", "comments:create"],
        "supports_pkce": True,  # Linear supports PKCE
    },
}


# =============================================================================
# Models
# =============================================================================

class OAuthAuthorizeResponse(BaseModel):
    """Response containing the authorization URL."""
    authorize_url: str
    state: str
    platform: str


class OAuthCallbackResponse(BaseModel):
    """Response after successful OAuth callback."""
    success: bool
    platform: str
    account_name: str | None = None
    account_id: str | None = None
    scopes: list[str] = Field(default_factory=list)
    message: str


class OAuthStatusResponse(BaseModel):
    """Status of an OAuth integration."""
    platform: str
    connected: bool
    account_name: str | None = None
    account_id: str | None = None
    scopes: list[str] = Field(default_factory=list)
    connected_at: str | None = None
    token_expires_at: str | None = None


class DisconnectResponse(BaseModel):
    """Response after disconnecting an integration."""
    success: bool
    platform: str
    message: str


# =============================================================================
# Encryption Utilities
# =============================================================================

def get_encryption_key() -> bytes:
    """Get the AES-256 encryption key from settings.

    Returns:
        32-byte encryption key

    Raises:
        ValueError: If encryption key is not configured
    """
    settings = get_settings()
    if not settings.oauth_encryption_key:
        raise ValueError("OAuth encryption key not configured (OAUTH_ENCRYPTION_KEY)")

    key_b64 = settings.oauth_encryption_key.get_secret_value()
    key = base64.b64decode(key_b64)

    if len(key) != 32:
        raise ValueError("OAuth encryption key must be 32 bytes (256 bits)")

    return key


def encrypt_token(token: str) -> str:
    """Encrypt an OAuth token using AES-256-GCM.

    Args:
        token: Plaintext token to encrypt

    Returns:
        Base64-encoded encrypted token (nonce + ciphertext + tag)
    """
    key = get_encryption_key()
    aesgcm = AESGCM(key)

    # Generate a random 12-byte nonce
    nonce = os.urandom(12)

    # Encrypt the token
    ciphertext = aesgcm.encrypt(nonce, token.encode("utf-8"), None)

    # Combine nonce + ciphertext and base64 encode
    encrypted = nonce + ciphertext
    return base64.b64encode(encrypted).decode("utf-8")


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt an OAuth token.

    Args:
        encrypted_token: Base64-encoded encrypted token

    Returns:
        Plaintext token

    Raises:
        ValueError: If decryption fails
    """
    key = get_encryption_key()
    aesgcm = AESGCM(key)

    # Decode from base64
    encrypted = base64.b64decode(encrypted_token)

    # Extract nonce (first 12 bytes) and ciphertext
    nonce = encrypted[:12]
    ciphertext = encrypted[12:]

    # Decrypt
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


# =============================================================================
# PKCE Utilities
# =============================================================================

def generate_pkce_pair() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge.

    Returns:
        Tuple of (code_verifier, code_challenge)
    """
    # Generate a random code verifier (43-128 characters)
    code_verifier = secrets.token_urlsafe(64)

    # Create SHA-256 hash and base64url encode (without padding)
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    return code_verifier, code_challenge


def generate_state() -> str:
    """Generate a cryptographically secure state parameter."""
    return secrets.token_urlsafe(32)


# =============================================================================
# OAuth State Management
# =============================================================================

async def create_oauth_state(
    user_id: str,
    platform: OAuthPlatform,
    redirect_uri: str,
    code_verifier: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Create and store an OAuth state parameter.

    Args:
        user_id: User initiating the OAuth flow
        platform: Target platform
        redirect_uri: Where to redirect after auth
        code_verifier: PKCE code verifier (if using PKCE)
        metadata: Additional metadata to store

    Returns:
        The generated state parameter
    """
    supabase = get_supabase_client()
    state = generate_state()

    result = await supabase.insert("oauth_states", {
        "user_id": user_id,
        "platform": platform.value,
        "state": state,
        "code_verifier": code_verifier,
        "redirect_uri": redirect_uri,
        "metadata": metadata or {},
    })

    if result.get("error"):
        logger.error("Failed to create OAuth state", error=result["error"])
        raise HTTPException(status_code=500, detail="Failed to initiate OAuth flow")

    return state


async def get_and_delete_oauth_state(state: str) -> dict | None:
    """Retrieve and delete an OAuth state (one-time use).

    Args:
        state: The state parameter from the callback

    Returns:
        The state data or None if not found/expired
    """
    supabase = get_supabase_client()

    # Get the state
    result = await supabase.request(
        f"/oauth_states?state=eq.{state}&expires_at=gt.{datetime.now(UTC).isoformat()}&select=*"
    )

    if result.get("error") or not result.get("data"):
        return None

    state_data = result["data"][0]

    # Delete the state (one-time use)
    await supabase.request(
        f"/oauth_states?state=eq.{state}",
        method="DELETE"
    )

    return state_data


async def cleanup_expired_states() -> int:
    """Clean up expired OAuth states.

    Returns:
        Number of states deleted
    """
    supabase = get_supabase_client()

    result = await supabase.request(
        f"/oauth_states?expires_at=lt.{datetime.now(UTC).isoformat()}",
        method="DELETE"
    )

    # Supabase doesn't return count easily, so we just log success
    logger.info("Cleaned up expired OAuth states")
    return 0


# =============================================================================
# Token Exchange Functions
# =============================================================================

async def exchange_github_code(code: str, redirect_uri: str) -> dict:
    """Exchange GitHub authorization code for access token."""
    settings = get_settings()

    if not settings.github_client_id or not settings.github_client_secret:
        raise HTTPException(status_code=500, detail="GitHub OAuth not configured")

    config = OAUTH_CONFIG[OAuthPlatform.GITHUB]

    async with httpx.AsyncClient() as client:
        # Exchange code for token
        response = await client.post(
            config["token_url"],
            data={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret.get_secret_value(),
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
        )

        if response.status_code != 200:
            logger.error("GitHub token exchange failed", status=response.status_code, body=response.text)
            raise HTTPException(status_code=400, detail="Failed to exchange authorization code")

        token_data = response.json()

        if "error" in token_data:
            logger.error("GitHub token error", error=token_data.get("error_description", token_data["error"]))
            raise HTTPException(status_code=400, detail=token_data.get("error_description", "OAuth error"))

        # Get user info
        user_response = await client.get(
            config["user_url"],
            headers={
                "Authorization": f"Bearer {token_data['access_token']}",
                "Accept": "application/json",
            },
        )

        user_data = {}
        if user_response.status_code == 200:
            user_data = user_response.json()

        return {
            "access_token": token_data["access_token"],
            "token_type": token_data.get("token_type", "bearer"),
            "scope": token_data.get("scope", "").split(","),
            "account_id": str(user_data.get("id", "")),
            "account_name": user_data.get("login", user_data.get("name", "")),
        }


async def exchange_slack_code(code: str, redirect_uri: str) -> dict:
    """Exchange Slack authorization code for access token."""
    settings = get_settings()

    if not settings.slack_client_id or not settings.slack_client_secret:
        raise HTTPException(status_code=500, detail="Slack OAuth not configured")

    config = OAUTH_CONFIG[OAuthPlatform.SLACK]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            config["token_url"],
            data={
                "client_id": settings.slack_client_id,
                "client_secret": settings.slack_client_secret.get_secret_value(),
                "code": code,
                "redirect_uri": redirect_uri,
            },
        )

        if response.status_code != 200:
            logger.error("Slack token exchange failed", status=response.status_code)
            raise HTTPException(status_code=400, detail="Failed to exchange authorization code")

        token_data = response.json()

        if not token_data.get("ok"):
            logger.error("Slack token error", error=token_data.get("error"))
            raise HTTPException(status_code=400, detail=token_data.get("error", "OAuth error"))

        # Extract bot token and team info
        return {
            "access_token": token_data.get("access_token"),
            "token_type": "bearer",
            "scope": token_data.get("scope", "").split(","),
            "account_id": token_data.get("team", {}).get("id", ""),
            "account_name": token_data.get("team", {}).get("name", ""),
            "bot_user_id": token_data.get("bot_user_id"),
        }


async def exchange_jira_code(code: str, redirect_uri: str, code_verifier: str | None = None) -> dict:
    """Exchange Jira authorization code for access token."""
    settings = get_settings()

    if not settings.jira_client_id or not settings.jira_client_secret:
        raise HTTPException(status_code=500, detail="Jira OAuth not configured")

    config = OAUTH_CONFIG[OAuthPlatform.JIRA]

    data: dict[str, Any] = {
        "grant_type": "authorization_code",
        "client_id": settings.jira_client_id,
        "client_secret": settings.jira_client_secret.get_secret_value(),
        "code": code,
        "redirect_uri": redirect_uri,
    }

    if code_verifier:
        data["code_verifier"] = code_verifier

    async with httpx.AsyncClient() as client:
        response = await client.post(
            config["token_url"],
            json=data,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            logger.error("Jira token exchange failed", status=response.status_code, body=response.text)
            raise HTTPException(status_code=400, detail="Failed to exchange authorization code")

        token_data = response.json()

        if "error" in token_data:
            logger.error("Jira token error", error=token_data.get("error_description", token_data["error"]))
            raise HTTPException(status_code=400, detail=token_data.get("error_description", "OAuth error"))

        # Get user info
        user_response = await client.get(
            config["user_url"],
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )

        user_data = {}
        if user_response.status_code == 200:
            user_data = user_response.json()

        # Get accessible resources (Jira sites)
        resources_response = await client.get(
            config["accessible_resources_url"],
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )

        resources = []
        if resources_response.status_code == 200:
            resources = resources_response.json()

        return {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "token_type": token_data.get("token_type", "Bearer"),
            "expires_in": token_data.get("expires_in"),
            "scope": token_data.get("scope", "").split(" "),
            "account_id": user_data.get("account_id", ""),
            "account_name": user_data.get("name", user_data.get("email", "")),
            "resources": resources,  # List of accessible Jira sites
        }


async def exchange_linear_code(code: str, redirect_uri: str, code_verifier: str | None = None) -> dict:
    """Exchange Linear authorization code for access token."""
    settings = get_settings()

    if not settings.linear_client_id or not settings.linear_client_secret:
        raise HTTPException(status_code=500, detail="Linear OAuth not configured")

    config = OAUTH_CONFIG[OAuthPlatform.LINEAR]

    data: dict[str, Any] = {
        "grant_type": "authorization_code",
        "client_id": settings.linear_client_id,
        "client_secret": settings.linear_client_secret.get_secret_value(),
        "code": code,
        "redirect_uri": redirect_uri,
    }

    if code_verifier:
        data["code_verifier"] = code_verifier

    async with httpx.AsyncClient() as client:
        response = await client.post(
            config["token_url"],
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            logger.error("Linear token exchange failed", status=response.status_code, body=response.text)
            raise HTTPException(status_code=400, detail="Failed to exchange authorization code")

        token_data = response.json()

        if "error" in token_data:
            logger.error("Linear token error", error=token_data.get("error_description", token_data["error"]))
            raise HTTPException(status_code=400, detail=token_data.get("error_description", "OAuth error"))

        # Get user info via GraphQL
        user_response = await client.post(
            config["user_url"],
            json={"query": "{ viewer { id name email } }"},
            headers={
                "Authorization": f"Bearer {token_data['access_token']}",
                "Content-Type": "application/json",
            },
        )

        user_data = {}
        if user_response.status_code == 200:
            gql_data = user_response.json()
            user_data = gql_data.get("data", {}).get("viewer", {})

        return {
            "access_token": token_data["access_token"],
            "token_type": token_data.get("token_type", "Bearer"),
            "expires_in": token_data.get("expires_in"),
            "scope": token_data.get("scope", "").split(",") if token_data.get("scope") else [],
            "account_id": user_data.get("id", ""),
            "account_name": user_data.get("name", user_data.get("email", "")),
        }


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/{platform}/authorize", response_model=OAuthAuthorizeResponse)
async def oauth_authorize(
    platform: OAuthPlatform,
    request: Request,
    scopes: str | None = Query(None, description="Comma-separated list of scopes"),
):
    """Generate OAuth authorization URL and state.

    Initiates the OAuth flow by generating:
    - A cryptographically secure state parameter
    - PKCE code verifier/challenge (where supported)
    - The authorization URL to redirect the user to

    The frontend should redirect the user to the returned `authorize_url`.
    """
    user = await get_current_user(request)
    settings = get_settings()
    config = OAUTH_CONFIG[platform]

    # Validate platform is configured
    if platform == OAuthPlatform.GITHUB and not settings.github_client_id:
        raise HTTPException(status_code=400, detail="GitHub OAuth not configured")
    elif platform == OAuthPlatform.SLACK and not settings.slack_client_id:
        raise HTTPException(status_code=400, detail="Slack OAuth not configured")
    elif platform == OAuthPlatform.JIRA and not settings.jira_client_id:
        raise HTTPException(status_code=400, detail="Jira OAuth not configured")
    elif platform == OAuthPlatform.LINEAR and not settings.linear_client_id:
        raise HTTPException(status_code=400, detail="Linear OAuth not configured")

    # Determine scopes
    requested_scopes = scopes.split(",") if scopes else config["default_scopes"]

    # Build redirect URI
    redirect_uri = f"{settings.oauth_redirect_base_url}/api/v1/oauth/{platform.value}/callback"

    # Generate PKCE if supported
    code_verifier = None
    code_challenge = None
    if config.get("supports_pkce"):
        code_verifier, code_challenge = generate_pkce_pair()

    # Create and store state
    state = await create_oauth_state(
        user_id=user["user_id"],
        platform=platform,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
        metadata={"scopes": requested_scopes},
    )

    # Build authorization URL
    params: dict[str, Any] = {
        "client_id": getattr(settings, f"{platform.value}_client_id"),
        "redirect_uri": redirect_uri,
        "state": state,
        "response_type": "code",
    }

    # Platform-specific parameters
    if platform == OAuthPlatform.GITHUB:
        params["scope"] = " ".join(requested_scopes)
    elif platform == OAuthPlatform.SLACK:
        params["scope"] = ",".join(requested_scopes)
        # Slack uses user_scope for user tokens
        params["user_scope"] = ""
    elif platform == OAuthPlatform.JIRA:
        params["scope"] = " ".join(requested_scopes)
        params["audience"] = config["audience"]
        params["prompt"] = "consent"
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"
    elif platform == OAuthPlatform.LINEAR:
        params["scope"] = ",".join(requested_scopes)
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

    authorize_url = f"{config['authorize_url']}?{urlencode(params)}"

    logger.info(
        "OAuth authorization initiated",
        platform=platform.value,
        user_id=user["user_id"],
        scopes=requested_scopes,
    )

    return OAuthAuthorizeResponse(
        authorize_url=authorize_url,
        state=state,
        platform=platform.value,
    )


@router.get("/{platform}/callback")
async def oauth_callback(
    platform: OAuthPlatform,
    request: Request,
    code: str = Query(..., description="Authorization code from OAuth provider"),
    state: str = Query(..., description="State parameter for CSRF protection"),
    error: str | None = Query(None, description="Error from OAuth provider"),
    error_description: str | None = Query(None, description="Error description"),
):
    """Handle OAuth callback from provider.

    This endpoint:
    1. Validates the state parameter
    2. Exchanges the authorization code for tokens
    3. Encrypts and stores the tokens
    4. Redirects to the dashboard

    The frontend should not call this directly - it's called by the OAuth provider.
    """
    settings = get_settings()

    # Handle OAuth errors
    if error:
        logger.warning("OAuth callback error", platform=platform.value, error=error, description=error_description)
        # Redirect to dashboard with error
        return RedirectResponse(
            url=f"{settings.oauth_redirect_base_url}/settings/integrations?error={error}&platform={platform.value}",
            status_code=302,
        )

    # Validate state and get stored data
    state_data = await get_and_delete_oauth_state(state)
    if not state_data:
        logger.warning("Invalid or expired OAuth state", state=state[:16])
        return RedirectResponse(
            url=f"{settings.oauth_redirect_base_url}/settings/integrations?error=invalid_state&platform={platform.value}",
            status_code=302,
        )

    user_id = state_data["user_id"]
    redirect_uri = state_data["redirect_uri"]
    code_verifier = state_data.get("code_verifier")
    requested_scopes = state_data.get("metadata", {}).get("scopes", [])

    try:
        # Exchange code for tokens
        if platform == OAuthPlatform.GITHUB:
            token_result = await exchange_github_code(code, redirect_uri)
        elif platform == OAuthPlatform.SLACK:
            token_result = await exchange_slack_code(code, redirect_uri)
        elif platform == OAuthPlatform.JIRA:
            token_result = await exchange_jira_code(code, redirect_uri, code_verifier)
        elif platform == OAuthPlatform.LINEAR:
            token_result = await exchange_linear_code(code, redirect_uri, code_verifier)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")

        # Encrypt tokens
        access_token_encrypted = encrypt_token(token_result["access_token"])
        refresh_token_encrypted = None
        if token_result.get("refresh_token"):
            refresh_token_encrypted = encrypt_token(token_result["refresh_token"])

        # Calculate token expiration
        token_expires_at = None
        if token_result.get("expires_in"):
            token_expires_at = (datetime.now(UTC) + timedelta(seconds=token_result["expires_in"])).isoformat()

        # Store or update integration
        supabase = get_supabase_client()

        # Check if integration already exists
        existing = await supabase.request(
            f"/integrations?user_id=eq.{user_id}&platform=eq.{platform.value}&select=id"
        )

        integration_data = {
            "user_id": user_id,
            "platform": platform.value,
            "platform_type": "cicd" if platform == OAuthPlatform.GITHUB else "notification",
            "is_connected": True,
            "access_token_encrypted": access_token_encrypted,
            "refresh_token_encrypted": refresh_token_encrypted,
            "token_expires_at": token_expires_at,
            "oauth_scopes": token_result.get("scope", requested_scopes),
            "external_account_id": token_result.get("account_id"),
            "external_account_name": token_result.get("account_name"),
            "last_sync_at": datetime.now(UTC).isoformat(),
            "sync_status": "connected",
            "config": {
                "connected_at": datetime.now(UTC).isoformat(),
                **({"resources": token_result.get("resources")} if token_result.get("resources") else {}),
            },
        }

        if existing.get("data") and len(existing["data"]) > 0:
            # Update existing
            integration_id = existing["data"][0]["id"]
            await supabase.update(
                "integrations",
                {"id": f"eq.{integration_id}"},
                integration_data,
            )
        else:
            # Create new
            result = await supabase.insert("integrations", integration_data)
            if result.get("error"):
                raise HTTPException(status_code=500, detail="Failed to save integration")

        # Log audit event
        await log_audit(
            organization_id=None,  # User-level integration
            user_id=user_id,
            user_email=None,
            action="oauth.connect",
            resource_type="integration",
            resource_id=platform.value,
            description=f"Connected {platform.value} account: {token_result.get('account_name', 'unknown')}",
            metadata={
                "platform": platform.value,
                "account_name": token_result.get("account_name"),
                "scopes": token_result.get("scope", []),
            },
            request=request,
        )

        logger.info(
            "OAuth integration connected",
            platform=platform.value,
            user_id=user_id,
            account_name=token_result.get("account_name"),
        )

        # Redirect to dashboard success page
        return RedirectResponse(
            url=f"{settings.oauth_redirect_base_url}/settings/integrations?success=true&platform={platform.value}",
            status_code=302,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("OAuth callback error", platform=platform.value, error=str(e))
        return RedirectResponse(
            url=f"{settings.oauth_redirect_base_url}/settings/integrations?error=token_exchange_failed&platform={platform.value}",
            status_code=302,
        )


@router.get("/{platform}/status", response_model=OAuthStatusResponse)
async def oauth_status(platform: OAuthPlatform, request: Request):
    """Get the status of an OAuth integration."""
    user = await get_current_user(request)
    supabase = get_supabase_client()

    result = await supabase.request(
        f"/integrations?user_id=eq.{user['user_id']}&platform=eq.{platform.value}&select=*"
    )

    if not result.get("data") or len(result["data"]) == 0:
        return OAuthStatusResponse(
            platform=platform.value,
            connected=False,
        )

    integration = result["data"][0]

    return OAuthStatusResponse(
        platform=platform.value,
        connected=integration.get("is_connected", False),
        account_name=integration.get("external_account_name"),
        account_id=integration.get("external_account_id"),
        scopes=integration.get("oauth_scopes", []),
        connected_at=integration.get("config", {}).get("connected_at"),
        token_expires_at=integration.get("token_expires_at"),
    )


@router.delete("/{platform}/disconnect", response_model=DisconnectResponse)
async def oauth_disconnect(platform: OAuthPlatform, request: Request):
    """Disconnect an OAuth integration.

    This will:
    - Revoke the access token (if supported by the platform)
    - Delete the stored tokens
    - Mark the integration as disconnected
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Get the integration
    result = await supabase.request(
        f"/integrations?user_id=eq.{user['user_id']}&platform=eq.{platform.value}&select=*"
    )

    if not result.get("data") or len(result["data"]) == 0:
        raise HTTPException(status_code=404, detail=f"No {platform.value} integration found")

    integration = result["data"][0]

    # TODO: Revoke token with provider if supported
    # GitHub and Slack support token revocation

    # Update integration to disconnected state
    await supabase.update(
        "integrations",
        {"id": f"eq.{integration['id']}"},
        {
            "is_connected": False,
            "access_token_encrypted": None,
            "refresh_token_encrypted": None,
            "token_expires_at": None,
            "sync_status": "disconnected",
        },
    )

    # Log audit event
    await log_audit(
        organization_id=None,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="oauth.disconnect",
        resource_type="integration",
        resource_id=platform.value,
        description=f"Disconnected {platform.value} account",
        metadata={"platform": platform.value},
        request=request,
    )

    logger.info(
        "OAuth integration disconnected",
        platform=platform.value,
        user_id=user["user_id"],
    )

    return DisconnectResponse(
        success=True,
        platform=platform.value,
        message=f"{platform.value.title()} integration disconnected successfully",
    )


@router.get("/status", response_model=list[OAuthStatusResponse])
async def oauth_status_all(request: Request):
    """Get the status of all OAuth integrations for the current user."""
    user = await get_current_user(request)
    supabase = get_supabase_client()

    result = await supabase.request(
        f"/integrations?user_id=eq.{user['user_id']}&select=*"
    )

    integrations_by_platform = {}
    for integration in result.get("data", []):
        integrations_by_platform[integration["platform"]] = integration

    statuses = []
    for platform in OAuthPlatform:
        integration = integrations_by_platform.get(platform.value)
        if integration:
            statuses.append(OAuthStatusResponse(
                platform=platform.value,
                connected=integration.get("is_connected", False),
                account_name=integration.get("external_account_name"),
                account_id=integration.get("external_account_id"),
                scopes=integration.get("oauth_scopes", []),
                connected_at=integration.get("config", {}).get("connected_at"),
                token_expires_at=integration.get("token_expires_at"),
            ))
        else:
            statuses.append(OAuthStatusResponse(
                platform=platform.value,
                connected=False,
            ))

    return statuses


# =============================================================================
# Token Refresh (for background jobs)
# =============================================================================

async def refresh_oauth_token(user_id: str, platform: OAuthPlatform) -> str | None:
    """Refresh an OAuth token if it's expired or about to expire.

    Args:
        user_id: User ID
        platform: OAuth platform

    Returns:
        New access token or None if refresh failed
    """
    settings = get_settings()
    supabase = get_supabase_client()

    # Get the integration
    result = await supabase.request(
        f"/integrations?user_id=eq.{user_id}&platform=eq.{platform.value}&select=*"
    )

    if not result.get("data") or len(result["data"]) == 0:
        return None

    integration = result["data"][0]

    if not integration.get("refresh_token_encrypted"):
        # No refresh token available
        return None

    # Check if token needs refresh (expires in less than 5 minutes)
    if integration.get("token_expires_at"):
        expires_at = datetime.fromisoformat(integration["token_expires_at"].replace("Z", "+00:00"))
        if expires_at > datetime.now(UTC) + timedelta(minutes=5):
            # Token still valid, decrypt and return
            return decrypt_token(integration["access_token_encrypted"])

    # Need to refresh
    refresh_token = decrypt_token(integration["refresh_token_encrypted"])

    # Only Jira supports refresh tokens in our implementation
    if platform != OAuthPlatform.JIRA:
        logger.warning(f"Token refresh not supported for {platform.value}")
        return None

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OAUTH_CONFIG[OAuthPlatform.JIRA]["token_url"],
                json={
                    "grant_type": "refresh_token",
                    "client_id": settings.jira_client_id,
                    "client_secret": settings.jira_client_secret.get_secret_value() if settings.jira_client_secret else "",
                    "refresh_token": refresh_token,
                },
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                logger.error("Token refresh failed", platform=platform.value, status=response.status_code)
                return None

            token_data = response.json()

            # Update stored tokens
            access_token_encrypted = encrypt_token(token_data["access_token"])
            refresh_token_encrypted = None
            if token_data.get("refresh_token"):
                refresh_token_encrypted = encrypt_token(token_data["refresh_token"])

            token_expires_at = None
            if token_data.get("expires_in"):
                token_expires_at = (datetime.now(UTC) + timedelta(seconds=token_data["expires_in"])).isoformat()

            await supabase.update(
                "integrations",
                {"id": f"eq.{integration['id']}"},
                {
                    "access_token_encrypted": access_token_encrypted,
                    "refresh_token_encrypted": refresh_token_encrypted or integration["refresh_token_encrypted"],
                    "token_expires_at": token_expires_at,
                },
            )

            logger.info("OAuth token refreshed", platform=platform.value, user_id=user_id)
            return token_data["access_token"]

    except Exception as e:
        logger.exception("Token refresh error", platform=platform.value, error=str(e))
        return None


async def get_access_token(user_id: str, platform: OAuthPlatform) -> str | None:
    """Get a valid access token for a user's OAuth integration.

    This will automatically refresh the token if needed.

    Args:
        user_id: User ID
        platform: OAuth platform

    Returns:
        Access token or None if not available
    """
    supabase = get_supabase_client()

    # Get the integration
    result = await supabase.request(
        f"/integrations?user_id=eq.{user_id}&platform=eq.{platform.value}&is_connected=eq.true&select=*"
    )

    if not result.get("data") or len(result["data"]) == 0:
        return None

    integration = result["data"][0]

    if not integration.get("access_token_encrypted"):
        return None

    # Check if token needs refresh
    if integration.get("token_expires_at"):
        expires_at = datetime.fromisoformat(integration["token_expires_at"].replace("Z", "+00:00"))
        if expires_at <= datetime.now(UTC) + timedelta(minutes=5):
            # Try to refresh
            new_token = await refresh_oauth_token(user_id, platform)
            if new_token:
                return new_token
            # If refresh failed and token is expired, return None
            if expires_at <= datetime.now(UTC):
                return None

    # Return decrypted token
    return decrypt_token(integration["access_token_encrypted"])
