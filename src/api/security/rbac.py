"""Role-Based Access Control (RBAC) for SOC2 compliance.

Implements a hierarchical permission system with:
- Predefined roles with specific permissions
- Organization-level and project-level access control
- Resource-level permissions
- Audit trail for permission changes
"""

from enum import Enum
from typing import Optional, Set, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps

from fastapi import HTTPException, Depends, Request
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


# =============================================================================
# Permission Definitions
# =============================================================================

class Permission(str, Enum):
    """All available permissions in the system."""

    # Organization permissions
    ORG_READ = "org:read"
    ORG_WRITE = "org:write"
    ORG_DELETE = "org:delete"
    ORG_MANAGE_MEMBERS = "org:manage_members"
    ORG_MANAGE_BILLING = "org:manage_billing"
    ORG_MANAGE_SETTINGS = "org:manage_settings"

    # Project permissions
    PROJECT_READ = "project:read"
    PROJECT_WRITE = "project:write"
    PROJECT_DELETE = "project:delete"
    PROJECT_MANAGE_MEMBERS = "project:manage_members"
    PROJECT_MANAGE_SETTINGS = "project:manage_settings"

    # Test permissions
    TEST_READ = "test:read"
    TEST_WRITE = "test:write"
    TEST_DELETE = "test:delete"
    TEST_EXECUTE = "test:execute"
    TEST_APPROVE = "test:approve"

    # Results permissions
    RESULTS_READ = "results:read"
    RESULTS_EXPORT = "results:export"
    RESULTS_DELETE = "results:delete"

    # API Key permissions
    API_KEY_READ = "api_key:read"
    API_KEY_CREATE = "api_key:create"
    API_KEY_REVOKE = "api_key:revoke"
    API_KEY_ROTATE = "api_key:rotate"

    # Audit permissions
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"

    # Healing permissions
    HEALING_READ = "healing:read"
    HEALING_APPROVE = "healing:approve"
    HEALING_CONFIGURE = "healing:configure"

    # Integration permissions
    INTEGRATION_READ = "integration:read"
    INTEGRATION_CONFIGURE = "integration:configure"
    INTEGRATION_DELETE = "integration:delete"

    # Admin permissions
    ADMIN_FULL_ACCESS = "admin:full_access"
    ADMIN_IMPERSONATE = "admin:impersonate"
    ADMIN_SYSTEM_CONFIG = "admin:system_config"


class Role(str, Enum):
    """Predefined roles with associated permissions."""

    # Organization-level roles
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

    # Project-level roles
    PROJECT_ADMIN = "project_admin"
    PROJECT_MEMBER = "project_member"
    PROJECT_VIEWER = "project_viewer"

    # Special roles
    BILLING_ADMIN = "billing_admin"
    SECURITY_ADMIN = "security_admin"
    SERVICE_ACCOUNT = "service_account"
    API_USER = "api_user"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.OWNER: {
        Permission.ORG_READ, Permission.ORG_WRITE, Permission.ORG_DELETE,
        Permission.ORG_MANAGE_MEMBERS, Permission.ORG_MANAGE_BILLING, Permission.ORG_MANAGE_SETTINGS,
        Permission.PROJECT_READ, Permission.PROJECT_WRITE, Permission.PROJECT_DELETE,
        Permission.PROJECT_MANAGE_MEMBERS, Permission.PROJECT_MANAGE_SETTINGS,
        Permission.TEST_READ, Permission.TEST_WRITE, Permission.TEST_DELETE,
        Permission.TEST_EXECUTE, Permission.TEST_APPROVE,
        Permission.RESULTS_READ, Permission.RESULTS_EXPORT, Permission.RESULTS_DELETE,
        Permission.API_KEY_READ, Permission.API_KEY_CREATE, Permission.API_KEY_REVOKE, Permission.API_KEY_ROTATE,
        Permission.AUDIT_READ, Permission.AUDIT_EXPORT,
        Permission.HEALING_READ, Permission.HEALING_APPROVE, Permission.HEALING_CONFIGURE,
        Permission.INTEGRATION_READ, Permission.INTEGRATION_CONFIGURE, Permission.INTEGRATION_DELETE,
        Permission.ADMIN_FULL_ACCESS,
    },
    Role.ADMIN: {
        Permission.ORG_READ, Permission.ORG_WRITE,
        Permission.ORG_MANAGE_MEMBERS, Permission.ORG_MANAGE_SETTINGS,
        Permission.PROJECT_READ, Permission.PROJECT_WRITE, Permission.PROJECT_DELETE,
        Permission.PROJECT_MANAGE_MEMBERS, Permission.PROJECT_MANAGE_SETTINGS,
        Permission.TEST_READ, Permission.TEST_WRITE, Permission.TEST_DELETE,
        Permission.TEST_EXECUTE, Permission.TEST_APPROVE,
        Permission.RESULTS_READ, Permission.RESULTS_EXPORT,
        Permission.API_KEY_READ, Permission.API_KEY_CREATE, Permission.API_KEY_REVOKE, Permission.API_KEY_ROTATE,
        Permission.AUDIT_READ,
        Permission.HEALING_READ, Permission.HEALING_APPROVE, Permission.HEALING_CONFIGURE,
        Permission.INTEGRATION_READ, Permission.INTEGRATION_CONFIGURE,
    },
    Role.MEMBER: {
        Permission.ORG_READ,
        Permission.PROJECT_READ, Permission.PROJECT_WRITE,
        Permission.TEST_READ, Permission.TEST_WRITE, Permission.TEST_EXECUTE,
        Permission.RESULTS_READ, Permission.RESULTS_EXPORT,
        Permission.HEALING_READ,
        Permission.INTEGRATION_READ,
    },
    Role.VIEWER: {
        Permission.ORG_READ,
        Permission.PROJECT_READ,
        Permission.TEST_READ,
        Permission.RESULTS_READ,
    },
    Role.PROJECT_ADMIN: {
        Permission.PROJECT_READ, Permission.PROJECT_WRITE,
        Permission.PROJECT_MANAGE_MEMBERS, Permission.PROJECT_MANAGE_SETTINGS,
        Permission.TEST_READ, Permission.TEST_WRITE, Permission.TEST_DELETE,
        Permission.TEST_EXECUTE, Permission.TEST_APPROVE,
        Permission.RESULTS_READ, Permission.RESULTS_EXPORT,
        Permission.HEALING_READ, Permission.HEALING_APPROVE,
    },
    Role.PROJECT_MEMBER: {
        Permission.PROJECT_READ, Permission.PROJECT_WRITE,
        Permission.TEST_READ, Permission.TEST_WRITE, Permission.TEST_EXECUTE,
        Permission.RESULTS_READ, Permission.RESULTS_EXPORT,
        Permission.HEALING_READ,
    },
    Role.PROJECT_VIEWER: {
        Permission.PROJECT_READ,
        Permission.TEST_READ,
        Permission.RESULTS_READ,
    },
    Role.BILLING_ADMIN: {
        Permission.ORG_READ,
        Permission.ORG_MANAGE_BILLING,
    },
    Role.SECURITY_ADMIN: {
        Permission.ORG_READ,
        Permission.AUDIT_READ, Permission.AUDIT_EXPORT,
        Permission.API_KEY_READ, Permission.API_KEY_REVOKE,
    },
    Role.SERVICE_ACCOUNT: {
        Permission.TEST_READ, Permission.TEST_WRITE, Permission.TEST_EXECUTE,
        Permission.RESULTS_READ,
        Permission.HEALING_READ,
    },
    Role.API_USER: {
        Permission.TEST_READ, Permission.TEST_EXECUTE,
        Permission.RESULTS_READ,
    },
}


# =============================================================================
# RBAC Manager
# =============================================================================

class RBACManager:
    """Role-Based Access Control manager."""

    def __init__(self):
        self._role_permissions = ROLE_PERMISSIONS.copy()
        self._custom_permissions: Dict[str, Set[Permission]] = {}

    def get_permissions_for_role(self, role: Role) -> Set[Permission]:
        """Get all permissions for a role."""
        return self._role_permissions.get(role, set())

    def get_permissions_for_roles(self, roles: List[str]) -> Set[Permission]:
        """Get combined permissions for multiple roles."""
        permissions = set()
        for role_name in roles:
            try:
                role = Role(role_name)
                permissions.update(self.get_permissions_for_role(role))
            except ValueError:
                # Unknown role, skip
                pass
        return permissions

    def has_permission(
        self,
        roles: List[str],
        required_permission: Permission,
        scopes: List[str] = None,
    ) -> bool:
        """Check if roles have a required permission."""
        # Check scopes first (API key scopes)
        if scopes:
            scope_map = {
                "read": {Permission.TEST_READ, Permission.RESULTS_READ, Permission.PROJECT_READ},
                "write": {Permission.TEST_WRITE, Permission.PROJECT_WRITE},
                "execute": {Permission.TEST_EXECUTE},
                "admin": {Permission.ADMIN_FULL_ACCESS},
            }
            allowed = set()
            for scope in scopes:
                allowed.update(scope_map.get(scope, set()))
            if required_permission not in allowed:
                return False

        # Check role permissions
        permissions = self.get_permissions_for_roles(roles)
        return required_permission in permissions

    def add_custom_permission(self, user_id: str, permission: Permission) -> None:
        """Add a custom permission for a specific user."""
        if user_id not in self._custom_permissions:
            self._custom_permissions[user_id] = set()
        self._custom_permissions[user_id].add(permission)

    def remove_custom_permission(self, user_id: str, permission: Permission) -> None:
        """Remove a custom permission from a user."""
        if user_id in self._custom_permissions:
            self._custom_permissions[user_id].discard(permission)


# Global RBAC manager instance
_rbac_manager = RBACManager()


def get_rbac_manager() -> RBACManager:
    """Get the global RBAC manager."""
    return _rbac_manager


# =============================================================================
# Permission Decorators
# =============================================================================

def require_permission(*required_permissions: Permission):
    """Decorator to require specific permissions on a route.

    Usage:
        @router.get("/resource")
        @require_permission(Permission.RESOURCE_READ)
        async def get_resource(user: UserContext = Depends(get_current_user)):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get user from kwargs or request
            user = kwargs.get("user")
            if not user:
                request = kwargs.get("request")
                if request and hasattr(request.state, "user"):
                    user = request.state.user

            if not user:
                raise HTTPException(status_code=401, detail="Authentication required")

            # Check permissions
            rbac = get_rbac_manager()
            for permission in required_permissions:
                if not rbac.has_permission(user.roles, permission, user.scopes):
                    logger.warning(
                        "Permission denied",
                        user_id=user.user_id,
                        required=permission.value,
                        roles=user.roles,
                    )
                    raise HTTPException(
                        status_code=403,
                        detail=f"Permission denied: {permission.value}",
                    )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_any_permission(*required_permissions: Permission):
    """Decorator to require any of the specified permissions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("user")
            if not user:
                request = kwargs.get("request")
                if request and hasattr(request.state, "user"):
                    user = request.state.user

            if not user:
                raise HTTPException(status_code=401, detail="Authentication required")

            rbac = get_rbac_manager()
            has_any = any(
                rbac.has_permission(user.roles, perm, user.scopes)
                for perm in required_permissions
            )

            if not has_any:
                raise HTTPException(
                    status_code=403,
                    detail=f"Requires one of: {', '.join(p.value for p in required_permissions)}",
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Resource-Level Access Control
# =============================================================================

@dataclass
class ResourceAccess:
    """Resource access control entry."""
    resource_type: str
    resource_id: str
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    granted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    granted_by: Optional[str] = None
    expires_at: Optional[datetime] = None


class ResourceACL:
    """Access Control List for resources."""

    def __init__(self):
        self._acl: Dict[str, List[ResourceAccess]] = {}

    def _get_key(self, resource_type: str, resource_id: str) -> str:
        return f"{resource_type}:{resource_id}"

    def grant_access(
        self,
        resource_type: str,
        resource_id: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        permissions: Set[Permission] = None,
        granted_by: str = None,
        expires_at: datetime = None,
    ) -> None:
        """Grant access to a resource."""
        key = self._get_key(resource_type, resource_id)
        if key not in self._acl:
            self._acl[key] = []

        entry = ResourceAccess(
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            organization_id=organization_id,
            permissions=permissions or set(),
            granted_by=granted_by,
            expires_at=expires_at,
        )
        self._acl[key].append(entry)

        logger.info(
            "Resource access granted",
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            organization_id=organization_id,
            permissions=[p.value for p in permissions] if permissions else [],
        )

    def revoke_access(
        self,
        resource_type: str,
        resource_id: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> None:
        """Revoke access to a resource."""
        key = self._get_key(resource_type, resource_id)
        if key not in self._acl:
            return

        self._acl[key] = [
            entry for entry in self._acl[key]
            if not (
                (user_id and entry.user_id == user_id) or
                (organization_id and entry.organization_id == organization_id)
            )
        ]

        logger.info(
            "Resource access revoked",
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            organization_id=organization_id,
        )

    def has_access(
        self,
        resource_type: str,
        resource_id: str,
        user_id: str,
        organization_id: Optional[str],
        required_permission: Permission,
    ) -> bool:
        """Check if user has access to a resource."""
        key = self._get_key(resource_type, resource_id)
        if key not in self._acl:
            return False

        now = datetime.now(timezone.utc)

        for entry in self._acl[key]:
            # Check expiration
            if entry.expires_at and entry.expires_at < now:
                continue

            # Check user match
            if entry.user_id and entry.user_id == user_id:
                if required_permission in entry.permissions:
                    return True

            # Check organization match
            if entry.organization_id and entry.organization_id == organization_id:
                if required_permission in entry.permissions:
                    return True

        return False


# Global resource ACL
_resource_acl = ResourceACL()


def get_resource_acl() -> ResourceACL:
    """Get the global resource ACL."""
    return _resource_acl
