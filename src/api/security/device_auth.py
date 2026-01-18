"""OAuth2 Device Authorization Grant for MCP and CLI tools.

Implements RFC 8628 - OAuth 2.0 Device Authorization Grant.
This allows CLI tools and MCP servers to authenticate users via browser.

Flow:
1. CLI calls POST /api/v1/auth/device/authorize
2. Gets device_code, user_code, and verification_uri
3. User opens verification_uri in browser, enters user_code
4. CLI polls POST /api/v1/auth/device/token until approved
5. CLI receives access_token and refresh_token

Similar to how Vercel CLI, GitHub CLI, and other tools authenticate.
"""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from enum import Enum

import structlog
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .auth import UserContext, generate_jwt_token

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/auth/device", tags=["Device Auth"])

# =============================================================================
# Configuration
# =============================================================================

DEVICE_CODE_EXPIRATION = 600  # 10 minutes
POLLING_INTERVAL = 5  # seconds
USER_CODE_LENGTH = 8  # characters

# In-memory storage for device codes (should use Redis in production)
# Format: device_code_hash -> DeviceAuthRequest
_device_codes: dict[str, "DeviceAuthRequest"] = {}

# User codes mapping: user_code -> device_code_hash
_user_codes: dict[str, str] = {}


# =============================================================================
# Models
# =============================================================================

class DeviceCodeStatus(str, Enum):
    """Status of a device authorization request."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class DeviceAuthRequest(BaseModel):
    """Device authorization request stored in memory/cache."""
    device_code_hash: str
    user_code: str
    client_id: str
    scope: list[str]
    created_at: datetime
    expires_at: datetime
    status: DeviceCodeStatus = DeviceCodeStatus.PENDING
    user_id: str | None = None
    organization_id: str | None = None
    email: str | None = None
    name: str | None = None


class DeviceAuthorizationResponse(BaseModel):
    """Response to device authorization request."""
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int = POLLING_INTERVAL


class TokenResponse(BaseModel):
    """OAuth2 token response."""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_token: str | None = None
    scope: str


class TokenErrorResponse(BaseModel):
    """OAuth2 error response."""
    error: str
    error_description: str | None = None


# =============================================================================
# Helpers
# =============================================================================

def generate_device_code() -> str:
    """Generate a cryptographically secure device code."""
    return secrets.token_urlsafe(32)


def generate_user_code() -> str:
    """Generate a user-friendly code for verification.

    Format: XXXX-XXXX (letters only, no ambiguous characters)
    """
    # Use only unambiguous uppercase letters
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # No I, O (look like 1, 0)
    first_part = ''.join(secrets.choice(alphabet) for _ in range(4))
    second_part = ''.join(secrets.choice(alphabet) for _ in range(4))
    return f"{first_part}-{second_part}"


def hash_device_code(code: str) -> str:
    """Hash a device code for storage."""
    return hashlib.sha256(code.encode()).hexdigest()


def get_verification_url(request: Request) -> str:
    """Get the base verification URL."""
    # Use the frontend URL from environment or construct from request
    import os
    frontend_url = os.getenv("FRONTEND_URL", "https://heyargus.ai")
    return f"{frontend_url}/auth/device"


def cleanup_expired_codes():
    """Remove expired device codes from memory."""
    now = datetime.now(UTC)
    expired = [
        code_hash for code_hash, req in _device_codes.items()
        if req.expires_at < now
    ]
    for code_hash in expired:
        req = _device_codes.pop(code_hash, None)
        if req:
            _user_codes.pop(req.user_code, None)


# =============================================================================
# Device Authorization Endpoint
# =============================================================================

@router.post("/authorize", response_model=DeviceAuthorizationResponse)
async def device_authorize(
    request: Request,
    client_id: str = Form(default="argus-mcp"),
    scope: str = Form(default="read write"),
):
    """Initialize device authorization flow.

    This endpoint is called by CLI/MCP tools to start the authentication flow.
    Returns a device_code (secret) and user_code (shown to user).
    """
    cleanup_expired_codes()

    # Generate codes
    device_code = generate_device_code()
    user_code = generate_user_code()
    device_code_hash = hash_device_code(device_code)

    # Ensure user_code is unique
    while user_code in _user_codes:
        user_code = generate_user_code()

    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=DEVICE_CODE_EXPIRATION)

    # Store the request
    auth_request = DeviceAuthRequest(
        device_code_hash=device_code_hash,
        user_code=user_code,
        client_id=client_id,
        scope=scope.split(),
        created_at=now,
        expires_at=expires_at,
    )

    _device_codes[device_code_hash] = auth_request
    _user_codes[user_code] = device_code_hash

    verification_uri = get_verification_url(request)

    logger.info(
        "Device authorization started",
        client_id=client_id,
        user_code=user_code,
    )

    return DeviceAuthorizationResponse(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        verification_uri_complete=f"{verification_uri}?code={user_code}",
        expires_in=DEVICE_CODE_EXPIRATION,
        interval=POLLING_INTERVAL,
    )


# =============================================================================
# Token Endpoint (Polling)
# =============================================================================

@router.post("/token")
async def device_token(
    grant_type: str = Form(...),
    device_code: str = Form(...),
    client_id: str = Form(default="argus-mcp"),
):
    """Exchange device code for access token.

    CLI/MCP tools poll this endpoint until the user approves or denies.

    Possible responses:
    - authorization_pending: User hasn't completed auth yet
    - slow_down: Polling too fast
    - expired_token: Device code expired
    - access_denied: User denied
    - 200 with tokens: User approved
    """
    cleanup_expired_codes()

    if grant_type != "urn:ietf:params:oauth:grant-type:device_code":
        raise HTTPException(
            status_code=400,
            detail=TokenErrorResponse(
                error="unsupported_grant_type",
                error_description="Expected grant_type=urn:ietf:params:oauth:grant-type:device_code"
            ).model_dump()
        )

    device_code_hash = hash_device_code(device_code)
    auth_request = _device_codes.get(device_code_hash)

    if not auth_request:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_grant", "error_description": "Invalid device code"}
        )

    if auth_request.client_id != client_id:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_client", "error_description": "Client ID mismatch"}
        )

    if auth_request.expires_at < datetime.now(UTC):
        # Clean up expired request
        _device_codes.pop(device_code_hash, None)
        _user_codes.pop(auth_request.user_code, None)
        raise HTTPException(
            status_code=400,
            detail={"error": "expired_token", "error_description": "Device code has expired"}
        )

    if auth_request.status == DeviceCodeStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail={"error": "authorization_pending", "error_description": "User has not yet authorized"}
        )

    if auth_request.status == DeviceCodeStatus.DENIED:
        # Clean up denied request
        _device_codes.pop(device_code_hash, None)
        _user_codes.pop(auth_request.user_code, None)
        raise HTTPException(
            status_code=400,
            detail={"error": "access_denied", "error_description": "User denied the authorization"}
        )

    if auth_request.status == DeviceCodeStatus.APPROVED:
        # Generate tokens
        access_token = generate_jwt_token(
            user_id=auth_request.user_id,
            organization_id=auth_request.organization_id,
            email=auth_request.email,
            name=auth_request.name,
            roles=["api_user"],
            scopes=auth_request.scope,
            token_type="access",
        )

        refresh_token = generate_jwt_token(
            user_id=auth_request.user_id,
            organization_id=auth_request.organization_id,
            email=auth_request.email,
            name=auth_request.name,
            roles=["api_user"],
            scopes=auth_request.scope,
            token_type="refresh",
        )

        # Clean up used request
        _device_codes.pop(device_code_hash, None)
        _user_codes.pop(auth_request.user_code, None)

        logger.info(
            "Device authorization completed",
            user_id=auth_request.user_id,
            organization_id=auth_request.organization_id,
        )

        return TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=24 * 60 * 60,  # 24 hours
            refresh_token=refresh_token,
            scope=" ".join(auth_request.scope),
        )

    raise HTTPException(status_code=500, detail="Unknown authorization status")


# =============================================================================
# User Verification Endpoints
# =============================================================================

@router.get("/verify", response_class=HTMLResponse)
async def device_verify_page(code: str | None = None):
    """Display device verification page.

    This page is shown to users when they visit the verification_uri.
    They enter the user_code shown by the CLI tool.
    """
    # This would typically redirect to a frontend page
    # For now, return a simple HTML form
    pre_filled = f'value="{code}"' if code else 'placeholder="XXXX-XXXX"'

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Argus - Device Authorization</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
                color: white;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0;
            }}
            .container {{
                background: rgba(255,255,255,0.1);
                padding: 40px;
                border-radius: 16px;
                max-width: 400px;
                text-align: center;
            }}
            h1 {{
                color: #8b5cf6;
                margin-bottom: 10px;
            }}
            p {{
                color: #a0a0a0;
                margin-bottom: 30px;
            }}
            input {{
                padding: 15px 20px;
                font-size: 24px;
                text-align: center;
                letter-spacing: 4px;
                border: 2px solid #333;
                border-radius: 8px;
                background: #1a1a2e;
                color: white;
                width: 200px;
                text-transform: uppercase;
            }}
            input:focus {{
                outline: none;
                border-color: #8b5cf6;
            }}
            button {{
                margin-top: 20px;
                padding: 15px 40px;
                font-size: 16px;
                background: #8b5cf6;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                width: 100%;
            }}
            button:hover {{
                background: #7c3aed;
            }}
            .logo {{
                font-size: 48px;
                margin-bottom: 20px;
            }}
            .error {{
                color: #ef4444;
                margin-top: 10px;
            }}
            .success {{
                color: #22c55e;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">🔐</div>
            <h1>Argus</h1>
            <p>Enter the code displayed by your CLI tool</p>
            <form action="/api/v1/auth/device/verify" method="post" id="verify-form">
                <input type="text" name="user_code" {pre_filled}
                       pattern="[A-Za-z]{{4}}-[A-Za-z]{{4}}"
                       maxlength="9" required
                       autocomplete="off" autocapitalize="characters">
                <button type="submit">Verify Device</button>
            </form>
            <p id="message"></p>
        </div>
        <script>
            // Auto-format input
            document.querySelector('input').addEventListener('input', function(e) {{
                let val = e.target.value.toUpperCase().replace(/[^A-Z]/g, '');
                if (val.length > 4) {{
                    val = val.slice(0, 4) + '-' + val.slice(4, 8);
                }}
                e.target.value = val;
            }});
        </script>
    </body>
    </html>
    """


@router.post("/verify")
async def device_verify_submit(
    request: Request,
    user_code: str = Form(...),
):
    """Handle device verification form submission.

    In production, this would:
    1. Check if user is logged in (redirect to login if not)
    2. Show confirmation screen with scopes
    3. Allow approve/deny

    For now, it auto-approves if user is authenticated.
    """
    # Normalize user code
    user_code = user_code.upper().strip()
    if "-" not in user_code and len(user_code) == 8:
        user_code = f"{user_code[:4]}-{user_code[4:]}"

    # Find the device code
    device_code_hash = _user_codes.get(user_code)
    if not device_code_hash:
        return HTMLResponse(
            content="""
            <html>
            <head>
                <style>
                    body { font-family: sans-serif; text-align: center; padding: 50px; background: #1a1a2e; color: white; }
                    .error { color: #ef4444; }
                    a { color: #8b5cf6; }
                </style>
            </head>
            <body>
                <h1 class="error">Invalid Code</h1>
                <p>The code you entered is invalid or has expired.</p>
                <a href="/api/v1/auth/device/verify">Try Again</a>
            </body>
            </html>
            """,
            status_code=400
        )

    auth_request = _device_codes.get(device_code_hash)
    if not auth_request:
        return HTMLResponse(
            content="<html><body>Code not found</body></html>",
            status_code=400
        )

    if auth_request.expires_at < datetime.now(UTC):
        return HTMLResponse(
            content="""
            <html>
            <head>
                <style>
                    body { font-family: sans-serif; text-align: center; padding: 50px; background: #1a1a2e; color: white; }
                    .error { color: #ef4444; }
                </style>
            </head>
            <body>
                <h1 class="error">Code Expired</h1>
                <p>Please request a new code from your CLI tool.</p>
            </body>
            </html>
            """,
            status_code=400
        )

    # Check if user is authenticated via Clerk (check cookies/session)
    # For simplicity, we'll check for a user in request.state (set by middleware)
    user: UserContext | None = getattr(request.state, "user", None)

    if user and user.user_id != "anonymous":
        # User is authenticated - approve the device
        auth_request.status = DeviceCodeStatus.APPROVED
        auth_request.user_id = user.user_id
        auth_request.organization_id = user.organization_id
        auth_request.email = user.email
        auth_request.name = user.name

        logger.info(
            "Device authorization approved",
            user_id=user.user_id,
            user_code=user_code,
        )

        return HTMLResponse(
            content=f"""
            <html>
            <head>
                <style>
                    body {{ font-family: sans-serif; text-align: center; padding: 50px; background: #1a1a2e; color: white; }}
                    .success {{ color: #22c55e; }}
                    .logo {{ font-size: 64px; }}
                </style>
            </head>
            <body>
                <div class="logo">✓</div>
                <h1 class="success">Device Authorized!</h1>
                <p>You can now close this window and return to your CLI tool.</p>
                <p>Logged in as: {user.email or user.user_id}</p>
            </body>
            </html>
            """
        )

    # User not authenticated - show login redirect
    # In production, redirect to Clerk sign-in with a return URL
    import os
    frontend_url = os.getenv("FRONTEND_URL", "https://heyargus.ai")
    return_url = f"/api/v1/auth/device/verify?code={user_code}"

    return HTMLResponse(
        content=f"""
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; text-align: center; padding: 50px; background: #1a1a2e; color: white; }}
                a {{
                    display: inline-block;
                    padding: 15px 40px;
                    background: #8b5cf6;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    margin-top: 20px;
                }}
                a:hover {{ background: #7c3aed; }}
            </style>
        </head>
        <body>
            <h1>Sign In Required</h1>
            <p>Please sign in to authorize this device.</p>
            <a href="{frontend_url}/sign-in?redirect_url={return_url}">Sign In with Argus</a>
        </body>
        </html>
        """
    )


@router.post("/approve")
async def device_approve(
    request: Request,
    user_code: str = Form(...),
    approve: bool = Form(default=True),
):
    """Approve or deny a device authorization (API endpoint).

    Called by the frontend after user makes their decision.
    """
    user: UserContext | None = getattr(request.state, "user", None)

    if not user or user.user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required")

    # Normalize user code
    user_code = user_code.upper().strip()
    if "-" not in user_code and len(user_code) == 8:
        user_code = f"{user_code[:4]}-{user_code[4:]}"

    device_code_hash = _user_codes.get(user_code)
    if not device_code_hash:
        raise HTTPException(status_code=404, detail="Device code not found")

    auth_request = _device_codes.get(device_code_hash)
    if not auth_request:
        raise HTTPException(status_code=404, detail="Device code not found")

    if auth_request.expires_at < datetime.now(UTC):
        raise HTTPException(status_code=400, detail="Device code expired")

    if approve:
        auth_request.status = DeviceCodeStatus.APPROVED
        auth_request.user_id = user.user_id
        auth_request.organization_id = user.organization_id
        auth_request.email = user.email
        auth_request.name = user.name
    else:
        auth_request.status = DeviceCodeStatus.DENIED

    return {"status": "approved" if approve else "denied"}


# =============================================================================
# Token Refresh
# =============================================================================

@router.post("/refresh")
async def refresh_token(
    refresh_token: str = Form(...),
    grant_type: str = Form(default="refresh_token"),
):
    """Refresh an access token using a refresh token."""
    from .auth import verify_jwt_token

    if grant_type != "refresh_token":
        raise HTTPException(
            status_code=400,
            detail={"error": "unsupported_grant_type"}
        )

    payload = verify_jwt_token(refresh_token)
    if not payload or payload.type != "refresh":
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_grant", "error_description": "Invalid refresh token"}
        )

    # Generate new access token
    access_token = generate_jwt_token(
        user_id=payload.sub,
        organization_id=payload.org,
        email=payload.email,
        name=payload.name,
        roles=payload.roles,
        scopes=payload.scopes,
        token_type="access",
    )

    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_in=24 * 60 * 60,
        scope=" ".join(payload.scopes),
    )
