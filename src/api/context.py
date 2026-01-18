"""Request context helpers for organization scoping.

Provides utilities to extract and manage organization context
from incoming requests for multi-tenant API operations.
"""


import structlog
from fastapi import Request

logger = structlog.get_logger()


async def get_current_organization_id(request: Request) -> str | None:
    """Get the current organization ID from request context.

    Priority:
    1. X-Organization-ID header
    2. Query param ?org_id=
    3. User's default organization from profile

    Args:
        request: The FastAPI request object

    Returns:
        Organization ID string or None if not found
    """
    # Check header first (highest priority)
    org_id = request.headers.get("x-organization-id")
    if org_id:
        logger.debug("Organization ID from header", org_id=org_id)
        return org_id

    # Check query param
    org_id = request.query_params.get("org_id")
    if org_id:
        logger.debug("Organization ID from query param", org_id=org_id)
        return org_id

    # Get from user profile (set by auth middleware)
    user = getattr(request.state, "user", None)
    if user:
        # Handle UserContext objects
        if hasattr(user, "organization_id") and user.organization_id:
            logger.debug("Organization ID from user context", org_id=user.organization_id)
            return user.organization_id
        # Handle dict-like user objects
        elif isinstance(user, dict) and user.get("organization_id"):
            logger.debug("Organization ID from user dict", org_id=user["organization_id"])
            return user["organization_id"]

    return None


async def require_organization_id(request: Request) -> str:
    """Get the current organization ID, raising an error if not found.

    Args:
        request: The FastAPI request object

    Returns:
        Organization ID string

    Raises:
        HTTPException: If no organization context is found
    """
    from fastapi import HTTPException

    org_id = await get_current_organization_id(request)
    if not org_id:
        raise HTTPException(
            status_code=400,
            detail="Organization context required. Provide X-Organization-ID header or org_id query param."
        )
    return org_id


def set_organization_context(request: Request, org_id: str) -> None:
    """Set the organization context on the request state.

    Useful for middleware or dependencies that need to
    explicitly set the organization context.

    Args:
        request: The FastAPI request object
        org_id: The organization ID to set
    """
    if not hasattr(request.state, "organization_id"):
        request.state.organization_id = org_id


async def get_organization_from_user(request: Request) -> str | None:
    """Get organization ID specifically from the authenticated user's profile.

    This ignores headers and query params, useful when you need
    to verify the user's actual organization membership.

    Args:
        request: The FastAPI request object

    Returns:
        Organization ID string or None if not found
    """
    user = getattr(request.state, "user", None)
    if not user:
        return None

    # Handle UserContext objects
    if hasattr(user, "organization_id"):
        return user.organization_id

    # Handle dict-like user objects
    if isinstance(user, dict):
        return user.get("organization_id")

    return None


async def verify_organization_access(request: Request, org_id: str) -> bool:
    """Verify that the current user has access to the specified organization.

    This performs a lightweight check based on the user's context.
    For full RBAC verification, use verify_org_access from teams.py.

    Args:
        request: The FastAPI request object
        org_id: The organization ID to verify access for

    Returns:
        True if access is allowed, False otherwise
    """
    user = getattr(request.state, "user", None)
    if not user:
        return False

    # Get user's organization ID
    user_org_id = None
    if hasattr(user, "organization_id"):
        user_org_id = user.organization_id
    elif isinstance(user, dict):
        user_org_id = user.get("organization_id")

    # Simple check: user's org matches requested org
    if user_org_id and user_org_id == org_id:
        return True

    # For more complex RBAC, the caller should use verify_org_access
    # from teams.py which checks the database
    return False
