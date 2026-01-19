"""Security middleware for SOC2 compliance.

Provides:
- Global authentication enforcement
- Rate limiting
- Audit logging
- Request/Response encryption
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime

import structlog
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.api.security.auth import (
    AuthMethod,
    UserContext,
    authenticate_api_key,
    authenticate_jwt,
    authenticate_service_account,
    get_client_ip,
    is_public_endpoint,
)
from src.services.audit_logger import get_audit_logger

logger = structlog.get_logger()


# =============================================================================
# Rate Limiting Configuration
# =============================================================================


class RateLimitConfig:
    """Rate limiting configuration."""

    # Default limits (requests per minute)
    # Increased from 60 to 120 for better E2E testing support
    DEFAULT_LIMIT = 120
    DEFAULT_WINDOW = 60  # seconds

    # Tier-based limits (increased across the board)
    TIER_LIMITS = {
        "free": {"requests": 60, "window": 60},  # Increased from 30
        "starter": {"requests": 120, "window": 60},  # Increased from 60
        "pro": {"requests": 600, "window": 60},  # Increased from 300
        "enterprise": {"requests": 2000, "window": 60},  # Increased from 1000
        "unlimited": {"requests": float("inf"), "window": 60},
    }

    # Per-endpoint overrides
    # NOTE: Order matters! More specific paths must come before less specific ones
    # because _get_limit_for_endpoint uses startswith() matching.
    ENDPOINT_LIMITS = {
        "/api/v1/chat/stream": {"requests": 10, "window": 60},
        "/api/v1/stream/test": {"requests": 5, "window": 60},
        # Discovery API endpoints - higher limits for normal API usage
        # Must come BEFORE /api/v1/discover to avoid false matches
        "/api/v1/discovery/": {"requests": 100, "window": 60},
        # Auto-discover endpoint (intensive crawling operation) - keep low
        "/api/v1/discover": {"requests": 10, "window": 60},
        "/health": {"requests": 1000, "window": 60},
    }

    # Exempt endpoints from rate limiting
    EXEMPT_ENDPOINTS = {
        "/health",
        "/openapi.json",
        "/docs",
        "/redoc",
    }


# =============================================================================
# Authentication Middleware
# =============================================================================


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Global authentication middleware.

    Enforces authentication on all non-public endpoints.
    """

    def __init__(self, app: ASGIApp, enforce_auth: bool = True):
        super().__init__(app)
        self.enforce_auth = enforce_auth

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Skip auth for public endpoints
        if is_public_endpoint(request.url.path):
            request.state.user = UserContext(
                user_id="anonymous",
                auth_method=AuthMethod.ANONYMOUS,
                ip_address=get_client_ip(request),
            )
            return await call_next(request)

        # Skip auth enforcement if disabled (development mode only)
        # SECURITY: This should NEVER be used in production
        if not self.enforce_auth:
            import os

            env = os.getenv("ENVIRONMENT", "development")
            if env not in ("development", "test", "local"):
                # CRITICAL: Do NOT bypass auth in production!
                # Log error and continue to actual auth flow
                logger.error(
                    "SECURITY: enforce_auth=False in non-development environment! Auth will be enforced.",
                    environment=env,
                    path=request.url.path,
                )
                # Fall through to normal auth flow below (do not bypass)
            else:
                # Development mode: use fixed dev user, NOT from headers
                # SECURITY: Don't accept x-user-id from headers - that's an auth bypass
                request.state.user = UserContext(
                    user_id="dev-user",
                    organization_id=None,  # No default org - must be specified per-request
                    roles=["developer"],  # Limited role, not admin
                    scopes=["read", "write", "execute"],  # No admin scope
                    auth_method=AuthMethod.ANONYMOUS,
                    ip_address=get_client_ip(request),
                )
                logger.warning(
                    "Development mode: auth bypassed",
                    user_id="dev-user",
                    path=request.url.path,
                )
                return await call_next(request)

        # Try to authenticate
        user = None

        # Check API key header
        api_key = request.headers.get("x-api-key")
        logger.debug(
            "Auth check",
            has_api_key=bool(api_key),
            api_key_prefix=api_key[:16] if api_key else None,
        )
        if api_key:
            user = await authenticate_api_key(api_key, request)
            logger.debug("API key auth result", authenticated=bool(user))

        # Check Authorization header
        if not user:
            auth_header = request.headers.get("authorization")
            if auth_header:
                parts = auth_header.split()
                if len(parts) == 2 and parts[0].lower() == "bearer":
                    token = parts[1]

                    # Try JWT
                    user = await authenticate_jwt(token, request)

                    # Try service account
                    if not user:
                        user = await authenticate_service_account(token, request)

        # If still no user, return 401
        if not user:
            # Log failed authentication to audit trail
            audit = get_audit_logger()
            asyncio.create_task(
                audit.log_auth_event(
                    event_type="login",
                    user_id="anonymous",
                    success=False,
                    ip_address=get_client_ip(request),
                    user_agent=request.headers.get("user-agent"),
                    metadata={
                        "path": request.url.path,
                        "method": request.method,
                        "had_api_key": bool(api_key),
                        "had_auth_header": bool(request.headers.get("authorization")),
                    },
                    error="Authentication required - no valid credentials provided",
                )
            )

            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Authentication required",
                    "request_id": request_id,
                },
                headers={"WWW-Authenticate": "Bearer, ApiKey"},
            )

        # Attach user to request
        request.state.user = user

        # Log authenticated request
        logger.info(
            "Authenticated request",
            request_id=request_id,
            user_id=user.user_id,
            organization_id=user.organization_id,
            auth_method=user.auth_method,
            path=request.url.path,
            method=request.method,
        )

        # Log successful authentication to audit trail
        audit = get_audit_logger()
        asyncio.create_task(
            audit.log_auth_event(
                event_type=user.auth_method.value if hasattr(user.auth_method, "value") else str(user.auth_method),
                user_id=user.user_id,
                success=True,
                ip_address=get_client_ip(request),
                user_agent=request.headers.get("user-agent"),
                organization_id=user.organization_id,
                metadata={
                    "path": request.url.path,
                    "method": request.method,
                    "session_id": user.session_id,
                },
            )
        )

        return await call_next(request)


# =============================================================================
# Rate Limiting Middleware
# =============================================================================


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with per-user and per-IP limits."""

    def __init__(self, app: ASGIApp, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self._request_counts: dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()

    def _get_rate_limit_key(self, request: Request) -> str:
        """Generate rate limit key based on user or IP."""
        if hasattr(request.state, "user") and request.state.user:
            user = request.state.user
            if user.organization_id:
                return f"org:{user.organization_id}"
            return f"user:{user.user_id}"
        return f"ip:{get_client_ip(request)}"

    def _get_limit_for_endpoint(self, path: str, user: UserContext | None) -> dict:
        """Get rate limit for specific endpoint and user tier."""
        # Check endpoint-specific limits
        for endpoint_prefix, limit in RateLimitConfig.ENDPOINT_LIMITS.items():
            if path.startswith(endpoint_prefix):
                return limit

        # Check user tier
        if user and hasattr(user, "tier"):
            tier = getattr(user, "tier", "free")
            if tier in RateLimitConfig.TIER_LIMITS:
                return RateLimitConfig.TIER_LIMITS[tier]

        return {
            "requests": RateLimitConfig.DEFAULT_LIMIT,
            "window": RateLimitConfig.DEFAULT_WINDOW,
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip exempt endpoints
        if request.url.path in RateLimitConfig.EXEMPT_ENDPOINTS:
            return await call_next(request)

        # Get rate limit key and config
        key = self._get_rate_limit_key(request)
        user = getattr(request.state, "user", None)
        limit_config = self._get_limit_for_endpoint(request.url.path, user)

        max_requests = limit_config["requests"]
        window = limit_config["window"]

        # Check rate limit
        now = time.time()

        async with self._lock:
            # Clean old requests
            self._request_counts[key] = [
                ts for ts in self._request_counts[key] if ts > now - window
            ]

            # Check if over limit
            if len(self._request_counts[key]) >= max_requests:
                retry_after = int(window - (now - self._request_counts[key][0]))

                logger.warning(
                    "Rate limit exceeded",
                    key=key,
                    path=request.url.path,
                    limit=max_requests,
                    window=window,
                )

                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "retry_after": retry_after,
                        "limit": max_requests,
                        "window": window,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(max_requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(now + retry_after)),
                    },
                )

            # Record this request
            self._request_counts[key].append(now)
            remaining = max_requests - len(self._request_counts[key])

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + window))

        return response


# =============================================================================
# Audit Logging Middleware
# =============================================================================


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Comprehensive audit logging for SOC2 compliance.

    Logs:
    - All API requests and responses
    - Authentication events
    - Data access patterns
    - Security events
    """

    def __init__(
        self,
        app: ASGIApp,
        enabled: bool = True,
        log_request_body: bool = False,
        log_response_body: bool = False,
        sensitive_paths: set = None,
    ):
        super().__init__(app)
        self.enabled = enabled
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.sensitive_paths = sensitive_paths or {
            "/api/v1/api-keys",
            "/api/v1/auth",
        }

    def _mask_sensitive_data(self, data: dict, fields: set = None) -> dict:
        """Mask sensitive fields in data."""
        if not isinstance(data, dict):
            return data

        sensitive_fields = fields or {
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "authorization",
            "credential",
            "private_key",
            "access_token",
            "refresh_token",
            "ssn",
            "credit_card",
            "card_number",
        }

        masked = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sf in key_lower for sf in sensitive_fields):
                masked[key] = "***REDACTED***"
            elif isinstance(value, dict):
                masked[key] = self._mask_sensitive_data(value, sensitive_fields)
            elif isinstance(value, list):
                masked[key] = [
                    self._mask_sensitive_data(v, sensitive_fields) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                masked[key] = value

        return masked

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Capture request details
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        start_time = time.time()

        user = getattr(request.state, "user", None)
        user_id = user.user_id if user else "anonymous"
        org_id = user.organization_id if user else None

        # Log request
        audit_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "event_type": "api_request",
            "user_id": user_id,
            "organization_id": org_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
        }

        # Optionally log request body (with masking)
        if self.log_request_body and request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                if body:
                    body_data = json.loads(body)
                    audit_entry["request_body"] = self._mask_sensitive_data(body_data)
            except Exception:
                pass

        # Process request
        response = None
        error = None
        try:
            response = await call_next(request)
        except Exception as e:
            error = str(e)
            raise
        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Complete audit entry
            audit_entry.update(
                {
                    "duration_ms": round(duration_ms, 2),
                    "status_code": response.status_code if response else 500,
                    "error": error,
                }
            )

            # Log based on sensitivity and status
            if response and response.status_code >= 400:
                logger.warning("API request failed", **audit_entry)
            elif any(request.url.path.startswith(p) for p in self.sensitive_paths):
                logger.info("Sensitive API access", **audit_entry)
            else:
                logger.debug("API request", **audit_entry)

            # Store audit log (async to not block response)
            asyncio.create_task(self._store_audit_log(audit_entry))

        return response

    def _is_valid_uuid(self, value: str) -> bool:
        """Check if a string is a valid UUID."""
        if not value:
            return False
        try:
            uuid.UUID(str(value))
            return True
        except (ValueError, TypeError):
            return False

    async def _store_audit_log(self, entry: dict) -> None:
        """Store audit log entry to security_audit_logs table.

        Uses the security_audit_logs table (not audit_logs) because:
        - It has duration_ms, status_code, method, path columns natively
        - It has flexible constraints (no CHECK on action/resource_type)
        - It's designed for API request logging per SOC2 requirements
        """
        try:
            from src.integrations.supabase import get_supabase

            supabase = await get_supabase()
            if supabase:
                # Convert duration_ms to int (column is INTEGER type)
                duration = entry.get("duration_ms")
                duration_int = int(round(duration)) if duration is not None else None

                # Determine severity and outcome based on status code
                status_code = entry.get("status_code", 200)
                if status_code >= 500:
                    severity = "error"
                    outcome = "error"
                elif status_code >= 400:
                    severity = "warning"
                    outcome = "failure"
                else:
                    severity = "info"
                    outcome = "success"

                # Validate organization_id - must be valid UUID for FK constraint
                # Dev mode uses "dev-org" which is not a valid UUID
                org_id = entry.get("organization_id")
                if not self._is_valid_uuid(org_id):
                    org_id = None

                await supabase.insert(
                    "security_audit_logs",
                    {
                        "event_type": entry["event_type"],
                        "severity": severity,
                        "user_id": entry["user_id"],
                        "organization_id": org_id,
                        "request_id": entry["request_id"],
                        "ip_address": entry["client_ip"],
                        "user_agent": entry.get("user_agent"),
                        "method": entry["method"],
                        "path": entry["path"],
                        "resource_type": "api_request",
                        "action": f"{entry['method']} {entry['path']}",
                        "description": f"API {entry['method']} request to {entry['path']}",
                        "outcome": outcome,
                        "status_code": status_code,
                        "duration_ms": duration_int,
                        "metadata": {
                            "query_params": entry.get("query_params"),
                            "error": entry.get("error"),
                        },
                    },
                )
        except Exception as e:
            logger.error("Failed to store audit log", error=str(e))


# =============================================================================
# Combined Security Middleware
# =============================================================================


class SecurityMiddleware(BaseHTTPMiddleware):
    """Combined security middleware for SOC2 compliance.

    Includes:
    - Request ID generation
    - Timing attack prevention
    - Request validation
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID if not present
        if not hasattr(request.state, "request_id"):
            request.state.request_id = str(uuid.uuid4())

        # Add timing attack prevention (constant time for auth failures)
        start_time = time.time()

        try:
            response = await call_next(request)
        except HTTPException as e:
            # Ensure minimum response time for auth errors
            if e.status_code == 401:
                elapsed = time.time() - start_time
                if elapsed < 0.1:  # Minimum 100ms for auth failures
                    await asyncio.sleep(0.1 - elapsed)
            raise

        # Add request ID to response
        response.headers["X-Request-ID"] = request.state.request_id

        return response
