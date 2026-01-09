"""Authentication and Authorization module for SOC2 compliance.

Supports multiple authentication methods:
- API Keys (X-API-Key header)
- JWT Bearer tokens (Authorization: Bearer <token>)
- Clerk JWTs (with JWKS verification)
- OAuth2 (for third-party integrations)
- Service Account tokens (for internal services)
"""

import hashlib
import hmac
import secrets
import time
import os
from datetime import datetime, timezone, timedelta
from functools import wraps
from typing import Optional, Callable, Any
from enum import Enum

from fastapi import Request, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import structlog
import jwt
import httpx

logger = structlog.get_logger()

# =============================================================================
# Clerk Configuration
# =============================================================================

# Clerk JWKS URL (for token verification)
CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL", "https://api.clerk.com/v1/jwks")
CLERK_ISSUER = os.getenv("CLERK_ISSUER")  # e.g., "https://clerk.your-app.com"
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")  # For Backend API calls

# Cache for Clerk JWKS
_clerk_jwks_cache: dict = {}
_clerk_jwks_cache_time: float = 0
CLERK_JWKS_CACHE_TTL = 3600  # 1 hour

# Cache for Clerk user info (email lookups)
_clerk_user_cache: dict[str, dict] = {}
_clerk_user_cache_time: dict[str, float] = {}
CLERK_USER_CACHE_TTL = 300  # 5 minutes

# =============================================================================
# Configuration
# =============================================================================

# Public endpoints that don't require authentication
PUBLIC_ENDPOINTS = {
    "/health",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/favicon.ico",
    # Chat endpoints - temporarily public for debugging
    # TODO: Remove after fixing Clerk auth forwarding
    "/api/v1/chat/",
    "/api/v1/chat/message",
    "/api/v1/chat/stream",
    # Device auth endpoints (must be public for MCP/CLI to authenticate)
    "/api/v1/auth/device/authorize",
    "/api/v1/auth/device/token",
    "/api/v1/auth/device/verify",
    "/api/v1/auth/device/refresh",
}

# Endpoints that only require API key (not full auth)
API_KEY_ONLY_ENDPOINTS = {
    "/api/v1/webhooks/",
    "/api/v1/webhooks/github",
    "/api/v1/webhooks/gitlab",
}

# JWT Configuration
JWT_SECRET_KEY = None  # Set from environment
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
JWT_REFRESH_EXPIRATION_DAYS = 30

# API Key prefix
API_KEY_PREFIX = "argus_"

# =============================================================================
# Security Schemes
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


# =============================================================================
# Models
# =============================================================================

class AuthMethod(str, Enum):
    """Authentication methods."""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    SERVICE_ACCOUNT = "service_account"
    ANONYMOUS = "anonymous"


class UserContext(BaseModel):
    """Authenticated user context."""
    user_id: str
    organization_id: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    auth_method: AuthMethod = AuthMethod.ANONYMOUS
    api_key_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    authenticated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        use_enum_values = True


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # user_id
    org: Optional[str] = None  # organization_id
    email: Optional[str] = None
    name: Optional[str] = None
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    iat: int  # issued at
    exp: int  # expiration
    jti: str  # JWT ID for revocation
    type: str = "access"  # access or refresh


# =============================================================================
# Token Management
# =============================================================================

def generate_jwt_token(
    user_id: str,
    organization_id: Optional[str] = None,
    email: Optional[str] = None,
    name: Optional[str] = None,
    roles: list[str] = None,
    scopes: list[str] = None,
    token_type: str = "access",
    secret_key: str = None,
) -> str:
    """Generate a JWT token."""
    from src.config import get_settings
    settings = get_settings()

    secret = secret_key or settings.jwt_secret_key
    if not secret:
        raise ValueError("JWT_SECRET_KEY not configured")

    now = datetime.now(timezone.utc)

    if token_type == "access":
        exp = now + timedelta(hours=JWT_EXPIRATION_HOURS)
    else:
        exp = now + timedelta(days=JWT_REFRESH_EXPIRATION_DAYS)

    payload = TokenPayload(
        sub=user_id,
        org=organization_id,
        email=email,
        name=name,
        roles=roles or [],
        scopes=scopes or [],
        iat=int(now.timestamp()),
        exp=int(exp.timestamp()),
        jti=secrets.token_urlsafe(16),
        type=token_type,
    )

    return jwt.encode(payload.model_dump(), secret, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str, secret_key: str = None) -> Optional[TokenPayload]:
    """Verify and decode a JWT token."""
    from src.config import get_settings
    settings = get_settings()

    secret = secret_key or settings.jwt_secret_key
    if not secret:
        return None

    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid JWT token", error=str(e))
        return None


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key and its hash.

    Returns:
        Tuple of (plaintext_key, key_hash)
    """
    random_bytes = secrets.token_bytes(32)
    plaintext_key = f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()
    return plaintext_key, key_hash


def hash_api_key(key: str) -> str:
    """Hash an API key for comparison."""
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key_signature(key: str, signature: str, payload: str) -> bool:
    """Verify HMAC signature for request signing."""
    expected = hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


# =============================================================================
# Authentication Functions
# =============================================================================

async def authenticate_api_key(api_key: str, request: Request) -> Optional[UserContext]:
    """Authenticate using API key."""
    if not api_key or not api_key.startswith(API_KEY_PREFIX):
        return None

    from src.integrations.supabase import get_supabase

    try:
        supabase = await get_supabase()
        if not supabase:
            logger.warning("Supabase not available for API key verification")
            return None

        key_hash = hash_api_key(api_key)
        result = await supabase.fetch_one(
            f"/api_keys?key_hash=eq.{key_hash}&revoked_at=is.null&select=*,organizations(id,name)"
        )

        if not result:
            logger.warning("Invalid API key", key_prefix=api_key[:12])
            return None

        # Check expiration
        if result.get("expires_at"):
            expires = datetime.fromisoformat(result["expires_at"].replace("Z", "+00:00"))
            if expires < datetime.now(timezone.utc):
                logger.warning("Expired API key", key_id=result["id"])
                return None

        # Update last used
        await supabase.update(
            "api_keys",
            {"last_used_at": datetime.now(timezone.utc).isoformat()},
            f"id=eq.{result['id']}"
        )

        return UserContext(
            user_id=result.get("created_by", "system"),
            organization_id=result["organization_id"],
            roles=["api_user"],
            scopes=result.get("scopes", ["read"]),
            auth_method=AuthMethod.API_KEY,
            api_key_id=result["id"],
            ip_address=get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )

    except Exception as e:
        logger.error("API key authentication error", error=str(e))
        return None


async def get_clerk_jwks() -> dict:
    """Fetch and cache Clerk JWKS for token verification."""
    global _clerk_jwks_cache, _clerk_jwks_cache_time

    current_time = time.time()

    # Return cached JWKS if still valid
    if _clerk_jwks_cache and (current_time - _clerk_jwks_cache_time) < CLERK_JWKS_CACHE_TTL:
        return _clerk_jwks_cache

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(CLERK_JWKS_URL, timeout=10.0)
            if response.status_code == 200:
                _clerk_jwks_cache = response.json()
                _clerk_jwks_cache_time = current_time
                logger.info("Clerk JWKS fetched successfully")
                return _clerk_jwks_cache
    except Exception as e:
        logger.error("Failed to fetch Clerk JWKS", error=str(e))

    return _clerk_jwks_cache or {}


# Cache for auto-detected JWKS URLs (keyed by issuer)
_issuer_jwks_cache: dict[str, dict] = {}
_issuer_jwks_cache_time: dict[str, float] = {}


async def get_jwks_for_issuer(issuer: str) -> dict:
    """Fetch JWKS for a specific issuer (auto-detected from token)."""
    global _issuer_jwks_cache, _issuer_jwks_cache_time

    current_time = time.time()

    # Return cached JWKS if still valid
    if issuer in _issuer_jwks_cache and (current_time - _issuer_jwks_cache_time.get(issuer, 0)) < CLERK_JWKS_CACHE_TTL:
        return _issuer_jwks_cache[issuer]

    # Derive JWKS URL from issuer
    jwks_url = f"{issuer.rstrip('/')}/.well-known/jwks.json"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(jwks_url, timeout=10.0)
            if response.status_code == 200:
                _issuer_jwks_cache[issuer] = response.json()
                _issuer_jwks_cache_time[issuer] = current_time
                logger.info("JWKS fetched successfully", issuer=issuer)
                return _issuer_jwks_cache[issuer]
            else:
                logger.warning("Failed to fetch JWKS", issuer=issuer, status=response.status_code)
    except Exception as e:
        logger.error("Failed to fetch JWKS for issuer", issuer=issuer, error=str(e))

    return _issuer_jwks_cache.get(issuer, {})


async def verify_clerk_jwt(token: str) -> Optional[dict]:
    """Verify a Clerk JWT token using JWKS.

    Supports two modes:
    1. Auto-detection: Extracts issuer from token and derives JWKS URL
    2. Configured: Uses CLERK_JWKS_URL if properly set
    """
    try:
        # Import PyJWK - may not be available in all environments
        try:
            from jwt import PyJWK
        except ImportError:
            logger.warning("PyJWK not available, skipping Clerk verification")
            return None

        # Get the unverified header to find the key ID
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        if not kid:
            # Not a Clerk/JWKS token (no kid), skip silently
            return None

        # Get unverified claims to extract issuer for auto-detection
        unverified_claims = jwt.decode(token, options={"verify_signature": False})
        token_issuer = unverified_claims.get("iss", "")

        # Determine which JWKS to use:
        # 1. If CLERK_JWKS_URL is properly configured, use it
        # 2. Otherwise, auto-detect from token's issuer claim
        jwks = {}

        if CLERK_JWKS_URL and CLERK_JWKS_URL != "https://api.clerk.com/v1/jwks":
            # Use configured JWKS URL
            jwks = await get_clerk_jwks()
        elif token_issuer and token_issuer.startswith("https://"):
            # Auto-detect: derive JWKS URL from token's issuer
            # Clerk issuers look like: https://proven-pug-84.clerk.accounts.dev
            # JWKS URL is: https://proven-pug-84.clerk.accounts.dev/.well-known/jwks.json
            logger.debug("Auto-detecting JWKS from token issuer", issuer=token_issuer)
            jwks = await get_jwks_for_issuer(token_issuer)
        else:
            # No way to get JWKS
            logger.debug("Cannot determine JWKS URL, skipping verification")
            return None

        keys = jwks.get("keys", [])

        if not keys:
            logger.warning("No keys in JWKS")
            return None

        # Find the matching key
        key_data = None
        for key in keys:
            if key.get("kid") == kid:
                key_data = key
                break

        if not key_data:
            # Key not found - might be an internal JWT, skip silently
            logger.debug("Key ID not found in JWKS", kid=kid)
            return None

        # Construct the public key
        public_key = PyJWK.from_dict(key_data).key

        # Verify and decode the token
        options = {
            "verify_signature": True,
            "verify_exp": True,
            "verify_iat": True,
            "require": ["exp", "iat", "sub"],
        }

        # Determine issuer for verification
        verify_issuer = CLERK_ISSUER or token_issuer

        if verify_issuer:
            options["verify_iss"] = True
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                options=options,
                issuer=verify_issuer,
            )
        else:
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                options=options,
            )

        logger.info("Clerk JWT verified successfully", user_id=payload.get("sub"))
        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("Clerk JWT expired")
        return None
    except jwt.InvalidTokenError as e:
        # Could be an internal JWT, not a Clerk JWT - skip silently
        logger.debug("JWT not a valid Clerk token", error=str(e))
        return None
    except Exception as e:
        logger.error("Clerk JWT verification error", error=str(e))
        return None


async def get_clerk_user_info(user_id: str) -> Optional[dict]:
    """Fetch user info from Clerk Backend API.

    This is used to get the user's email when it's not included in the JWT.
    Requires CLERK_SECRET_KEY to be configured.

    Returns:
        Dict with user info (email, name, etc.) or None if unavailable
    """
    global _clerk_user_cache, _clerk_user_cache_time

    if not CLERK_SECRET_KEY:
        logger.debug("CLERK_SECRET_KEY not configured, cannot fetch user info")
        return None

    if not user_id or not user_id.startswith("user_"):
        return None

    current_time = time.time()

    # Return cached user info if still valid
    if user_id in _clerk_user_cache and (current_time - _clerk_user_cache_time.get(user_id, 0)) < CLERK_USER_CACHE_TTL:
        return _clerk_user_cache[user_id]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.clerk.com/v1/users/{user_id}",
                headers={
                    "Authorization": f"Bearer {CLERK_SECRET_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )

            if response.status_code == 200:
                user_data = response.json()

                # Extract primary email
                email = None
                email_addresses = user_data.get("email_addresses", [])
                if email_addresses:
                    # Find primary email or use first one
                    for addr in email_addresses:
                        if addr.get("id") == user_data.get("primary_email_address_id"):
                            email = addr.get("email_address")
                            break
                    if not email and email_addresses:
                        email = email_addresses[0].get("email_address")

                # Build user info dict
                user_info = {
                    "id": user_data.get("id"),
                    "email": email,
                    "first_name": user_data.get("first_name"),
                    "last_name": user_data.get("last_name"),
                    "image_url": user_data.get("image_url"),
                }

                # Cache the result
                _clerk_user_cache[user_id] = user_info
                _clerk_user_cache_time[user_id] = current_time

                logger.debug("Fetched Clerk user info", user_id=user_id, email=email)
                return user_info
            else:
                logger.warning("Failed to fetch Clerk user", user_id=user_id, status=response.status_code)
    except Exception as e:
        logger.error("Error fetching Clerk user info", user_id=user_id, error=str(e))

    return None


async def authenticate_jwt(token: str, request: Request) -> Optional[UserContext]:
    """Authenticate using JWT token (supports both internal JWTs and Clerk JWTs)."""

    # First, try to verify as a Clerk JWT (RS256)
    clerk_payload = await verify_clerk_jwt(token)
    if clerk_payload:
        # Clerk JWT verified successfully
        # Log available claims for debugging
        logger.debug("Clerk JWT claims", claims=list(clerk_payload.keys()))

        # Extract user info from Clerk claims
        user_id = clerk_payload.get("sub", "")
        org_id = clerk_payload.get("org_id")
        org_role = clerk_payload.get("org_role")

        # Build roles from Clerk metadata
        roles = []
        if org_role:
            roles.append(org_role)

        # Check for custom claims (set in Clerk Dashboard → Sessions → Customize)
        metadata = clerk_payload.get("public_metadata", {}) or {}
        if metadata.get("roles"):
            roles.extend(metadata["roles"])

        # Extract email - check multiple possible claim locations
        # Clerk can include these via session token customization
        email = (
            clerk_payload.get("email") or
            clerk_payload.get("primary_email") or
            clerk_payload.get("email_address") or
            (clerk_payload.get("user", {}) or {}).get("email") or
            (clerk_payload.get("user", {}) or {}).get("primary_email_address")
        )

        # Extract name - check multiple possible claim locations
        name = (
            clerk_payload.get("name") or
            clerk_payload.get("full_name") or
            clerk_payload.get("first_name") or
            (clerk_payload.get("user", {}) or {}).get("first_name")
        )

        # If we have first_name and last_name, combine them
        if not name and clerk_payload.get("first_name"):
            first = clerk_payload.get("first_name", "")
            last = clerk_payload.get("last_name", "")
            name = f"{first} {last}".strip() or None

        # If email/name not in JWT, fetch from Clerk Backend API
        # This is common when using default Clerk JWT (no custom template)
        if (not email or not name) and user_id:
            clerk_user = await get_clerk_user_info(user_id)
            if clerk_user:
                if not email:
                    email = clerk_user.get("email")
                if not name:
                    first = clerk_user.get("first_name", "")
                    last = clerk_user.get("last_name", "")
                    name = f"{first} {last}".strip() or None

        return UserContext(
            user_id=user_id,
            organization_id=org_id,
            email=email,
            name=name,
            roles=roles or ["member"],
            scopes=["read", "write", "execute"],  # Default scopes for Clerk users
            auth_method=AuthMethod.JWT,
            session_id=clerk_payload.get("sid") or clerk_payload.get("jti"),
            ip_address=get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )

    # Fall back to internal JWT verification (HS256)
    payload = verify_jwt_token(token)
    if not payload:
        return None

    # Check if token is revoked (optional - requires Redis/DB)
    # if await is_token_revoked(payload.jti):
    #     return None

    return UserContext(
        user_id=payload.sub,
        organization_id=payload.org,
        email=payload.email,
        name=payload.name,
        roles=payload.roles,
        scopes=payload.scopes,
        auth_method=AuthMethod.JWT,
        session_id=payload.jti,
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )


async def authenticate_service_account(token: str, request: Request) -> Optional[UserContext]:
    """Authenticate internal service account."""
    from src.config import get_settings
    settings = get_settings()

    # Service accounts use a special prefix
    if not token.startswith("svc_"):
        return None

    # Verify against configured service accounts
    service_accounts = getattr(settings, "service_accounts", {})

    for svc_name, svc_config in service_accounts.items():
        if hmac.compare_digest(token, svc_config.get("token", "")):
            return UserContext(
                user_id=f"service:{svc_name}",
                roles=["service_account"],
                scopes=svc_config.get("scopes", ["read", "write"]),
                auth_method=AuthMethod.SERVICE_ACCOUNT,
                ip_address=get_client_ip(request),
            )

    return None


# =============================================================================
# Request Helpers
# =============================================================================

def get_client_ip(request: Request) -> str:
    """Get client IP address, handling proxies."""
    # Check X-Forwarded-For header (set by reverse proxies)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()

    # Check X-Real-IP header
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip

    # Fall back to direct client IP
    if request.client:
        return request.client.host

    return "unknown"


def is_public_endpoint(path: str) -> bool:
    """Check if endpoint is public (no auth required)."""
    # Exact matches
    if path in PUBLIC_ENDPOINTS:
        return True

    # Prefix matches
    for prefix in PUBLIC_ENDPOINTS:
        if prefix.endswith("/") and path.startswith(prefix):
            return True

    return False


def is_api_key_only_endpoint(path: str) -> bool:
    """Check if endpoint only requires API key (not full auth)."""
    for prefix in API_KEY_ONLY_ENDPOINTS:
        if path.startswith(prefix):
            return True
    return False


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def get_current_user(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> UserContext:
    """Get current authenticated user.

    This is a FastAPI dependency that can be used in route handlers.
    """
    # Check if already authenticated by middleware
    if hasattr(request.state, "user") and request.state.user:
        return request.state.user

    # Skip auth for public endpoints
    if is_public_endpoint(request.url.path):
        return UserContext(
            user_id="anonymous",
            auth_method=AuthMethod.ANONYMOUS,
            ip_address=get_client_ip(request),
        )

    # Try API key authentication
    if api_key:
        user = await authenticate_api_key(api_key, request)
        if user:
            return user

    # Try Bearer token (JWT)
    if bearer and bearer.credentials:
        token = bearer.credentials

        # Try JWT
        user = await authenticate_jwt(token, request)
        if user:
            return user

        # Try service account
        user = await authenticate_service_account(token, request)
        if user:
            return user

    # No valid authentication
    raise HTTPException(
        status_code=401,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer, ApiKey"},
    )


def require_auth(func: Callable = None):
    """Decorator to require authentication on a route."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request and not hasattr(request.state, "user"):
                raise HTTPException(status_code=401, detail="Authentication required")

            return await f(*args, **kwargs)
        return wrapper

    if func:
        return decorator(func)
    return decorator


def require_roles(*required_roles: str):
    """Decorator to require specific roles."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, user: UserContext = Depends(get_current_user), **kwargs):
            if not any(role in user.roles for role in required_roles):
                raise HTTPException(
                    status_code=403,
                    detail=f"Required roles: {', '.join(required_roles)}",
                )
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator


def require_scopes(*required_scopes: str):
    """Decorator to require specific scopes."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, user: UserContext = Depends(get_current_user), **kwargs):
            if not all(scope in user.scopes for scope in required_scopes):
                raise HTTPException(
                    status_code=403,
                    detail=f"Required scopes: {', '.join(required_scopes)}",
                )
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Token Revocation (requires Redis/DB)
# =============================================================================

_revoked_tokens: set[str] = set()  # In-memory for development


async def revoke_token(jti: str) -> None:
    """Revoke a JWT token by its ID."""
    _revoked_tokens.add(jti)
    logger.info("Token revoked", jti=jti)


async def is_token_revoked(jti: str) -> bool:
    """Check if a token is revoked."""
    return jti in _revoked_tokens
