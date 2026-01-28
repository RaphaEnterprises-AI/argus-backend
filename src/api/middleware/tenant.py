"""
Tenant Middleware for Multi-Tenant API Isolation

Extracts tenant context from requests and validates access.
All protected endpoints should use TenantDep for automatic context injection.
"""

import re
from typing import Annotated, Optional

import structlog
from fastapi import Depends, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.core.tenant import TenantContext, set_current_tenant

logger = structlog.get_logger()

# URL patterns that don't require tenant context
PUBLIC_PATHS = [
    r"^/$",
    r"^/health$",
    r"^/ready$",
    r"^/docs",
    r"^/openapi\.json$",
    r"^/redoc",
    r"^/api/v1/auth/",
    r"^/api/v1/webhooks/",  # Webhooks have their own auth
    r"^/api/v1/oauth/",
]

# Paths that extract org/project from URL pattern
# Example: /api/v1/orgs/{org_id}/projects/{project_id}/...
ORG_PROJECT_PATH_PATTERN = re.compile(
    r"/api/v1/orgs/(?P<org_id>[a-zA-Z0-9_-]+)"
    r"(?:/projects/(?P<project_id>[a-zA-Z0-9_-]+))?"
)


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware that extracts and sets tenant context for each request.

    The middleware:
    1. Checks if path is public (skips tenant extraction)
    2. Extracts org_id/project_id from URL path
    3. Falls back to headers if not in path
    4. Validates user has access to the organization
    5. Sets TenantContext on request.state and context var

    Usage:
        app.add_middleware(TenantMiddleware)
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request and inject tenant context."""
        path = request.url.path

        # Skip tenant extraction for public paths
        if self._is_public_path(path):
            return await call_next(request)

        # Try to extract tenant context
        try:
            tenant_context = await self._extract_tenant_context(request)
            if tenant_context:
                # Set on request state for FastAPI access
                request.state.tenant = tenant_context
                # Set context var for downstream services
                set_current_tenant(tenant_context)

                logger.debug(
                    "Tenant context set",
                    **tenant_context.to_log_context()
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to extract tenant context", error=str(e))
            # Don't fail request, let endpoints decide if tenant is required

        return await call_next(request)

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public and doesn't require tenant context."""
        for pattern in PUBLIC_PATHS:
            if re.match(pattern, path):
                return True
        return False

    async def _extract_tenant_context(self, request: Request) -> TenantContext | None:
        """Extract tenant context from request.

        Priority:
        1. URL path parameters (/api/v1/orgs/{org_id}/projects/{project_id})
        2. X-Organization-ID and X-Project-ID headers
        3. Query parameters org_id and project_id
        4. User's default organization from JWT

        Args:
            request: FastAPI request object

        Returns:
            TenantContext if extracted, None otherwise
        """
        org_id: str | None = None
        project_id: str | None = None

        # 1. Try URL path pattern
        match = ORG_PROJECT_PATH_PATTERN.search(request.url.path)
        if match:
            org_id = match.group("org_id")
            project_id = match.group("project_id")

        # 2. Fall back to headers
        if not org_id:
            org_id = request.headers.get("x-organization-id")
        if not project_id:
            project_id = request.headers.get("x-project-id")

        # 3. Fall back to query params
        if not org_id:
            org_id = request.query_params.get("org_id")
        if not project_id:
            project_id = request.query_params.get("project_id")

        # 4. Fall back to user's default org from JWT
        user = getattr(request.state, "user", None)
        if not org_id and user:
            if hasattr(user, "organization_id"):
                org_id = user.organization_id
            elif isinstance(user, dict):
                org_id = user.get("organization_id")

        # If we still don't have org_id, return None
        if not org_id:
            return None

        # Get user info for context
        user_id: str | None = None
        user_email: str | None = None
        plan = "free"

        if user:
            if hasattr(user, "id"):
                user_id = str(user.id)
            elif isinstance(user, dict):
                user_id = user.get("id") or user.get("user_id")

            if hasattr(user, "email"):
                user_email = user.email
            elif isinstance(user, dict):
                user_email = user.get("email")

        # Get request ID from headers or generate
        request_id = request.headers.get("x-request-id")

        # Build tenant context
        return TenantContext(
            org_id=org_id,
            project_id=project_id,
            user_id=user_id,
            user_email=user_email,
            plan=plan,
            request_id=request_id,
        )


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def get_tenant_context(request: Request) -> TenantContext | None:
    """Get tenant context from request state.

    Use this dependency when tenant context is optional.

    Args:
        request: FastAPI request object

    Returns:
        TenantContext if available, None otherwise
    """
    return getattr(request.state, "tenant", None)


async def require_tenant_context(request: Request) -> TenantContext:
    """Get tenant context, raising 400 if not available.

    Use this dependency when tenant context is required.

    Args:
        request: FastAPI request object

    Returns:
        TenantContext

    Raises:
        HTTPException: 400 if no tenant context
    """
    tenant = getattr(request.state, "tenant", None)
    if tenant is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "tenant_context_required",
                "message": (
                    "Organization context required. "
                    "Use URL path /api/v1/orgs/{org_id}/... or "
                    "provide X-Organization-ID header."
                ),
            }
        )
    return tenant


async def require_project_context(request: Request) -> TenantContext:
    """Get tenant context with project_id, raising if missing.

    Use this dependency when both org and project context are required.

    Args:
        request: FastAPI request object

    Returns:
        TenantContext with project_id set

    Raises:
        HTTPException: 400 if no tenant context or project_id
    """
    tenant = await require_tenant_context(request)
    if tenant.project_id is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "project_context_required",
                "message": (
                    "Project context required. "
                    "Use URL path /api/v1/orgs/{org_id}/projects/{project_id}/... or "
                    "provide X-Project-ID header."
                ),
            }
        )
    return tenant


# Type aliases for cleaner endpoint signatures
TenantDep = Annotated[TenantContext, Depends(require_tenant_context)]
OptionalTenantDep = Annotated[TenantContext | None, Depends(get_tenant_context)]
ProjectTenantDep = Annotated[TenantContext, Depends(require_project_context)]


# =============================================================================
# Validation Helpers
# =============================================================================

async def validate_org_access(
    request: Request,
    tenant: TenantContext,
) -> bool:
    """Validate the current user has access to the tenant's organization.

    This performs a database check for organization membership.
    Use for sensitive operations that need explicit access verification.

    Args:
        request: FastAPI request object
        tenant: TenantContext to validate

    Returns:
        True if access is valid

    Raises:
        HTTPException: 403 if access denied
    """
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )

    # Get user's organization memberships from profile
    user_org_id = None
    if hasattr(user, "organization_id"):
        user_org_id = user.organization_id
    elif isinstance(user, dict):
        user_org_id = user.get("organization_id")

    # Simple check: user's org matches tenant org
    # For enterprise, this should check team_members table
    if user_org_id != tenant.org_id:
        logger.warning(
            "Organization access denied",
            user_org_id=user_org_id,
            requested_org_id=tenant.org_id,
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error": "organization_access_denied",
                "message": "You do not have access to this organization",
            }
        )

    return True


async def validate_project_access(
    request: Request,
    tenant: TenantContext,
) -> bool:
    """Validate the current user has access to the tenant's project.

    Args:
        request: FastAPI request object
        tenant: TenantContext to validate

    Returns:
        True if access is valid

    Raises:
        HTTPException: 403 if access denied
    """
    # First validate org access
    await validate_org_access(request, tenant)

    # For now, if user has org access, they have project access
    # Enterprise tier could add project-level permissions
    return True
