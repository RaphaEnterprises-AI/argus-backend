"""Comprehensive tests for Role-Based Access Control module (rbac.py).

Tests cover:
- Permission and Role enums
- Role-to-permission mapping
- RBACManager functionality
- Permission checking with scopes
- Custom permissions
- Permission decorators
- Resource-level access control (ACL)
- Expiration handling
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import pytest


# =============================================================================
# Permission Enum Tests
# =============================================================================

class TestPermissionEnum:
    """Tests for Permission enum."""

    def test_permission_values(self):
        """Test that Permission enum has expected values."""
        from src.api.security.rbac import Permission

        assert Permission.ORG_READ.value == "org:read"
        assert Permission.PROJECT_WRITE.value == "project:write"
        assert Permission.TEST_EXECUTE.value == "test:execute"
        assert Permission.ADMIN_FULL_ACCESS.value == "admin:full_access"

    def test_permission_organization_permissions(self):
        """Test organization-level permissions exist."""
        from src.api.security.rbac import Permission

        org_permissions = [
            Permission.ORG_READ,
            Permission.ORG_WRITE,
            Permission.ORG_DELETE,
            Permission.ORG_MANAGE_MEMBERS,
            Permission.ORG_MANAGE_BILLING,
            Permission.ORG_MANAGE_SETTINGS,
        ]
        assert all(perm for perm in org_permissions)

    def test_permission_project_permissions(self):
        """Test project-level permissions exist."""
        from src.api.security.rbac import Permission

        project_permissions = [
            Permission.PROJECT_READ,
            Permission.PROJECT_WRITE,
            Permission.PROJECT_DELETE,
            Permission.PROJECT_MANAGE_MEMBERS,
            Permission.PROJECT_MANAGE_SETTINGS,
        ]
        assert all(perm for perm in project_permissions)

    def test_permission_test_permissions(self):
        """Test test-related permissions exist."""
        from src.api.security.rbac import Permission

        test_permissions = [
            Permission.TEST_READ,
            Permission.TEST_WRITE,
            Permission.TEST_DELETE,
            Permission.TEST_EXECUTE,
            Permission.TEST_APPROVE,
        ]
        assert all(perm for perm in test_permissions)


# =============================================================================
# Role Enum Tests
# =============================================================================

class TestRoleEnum:
    """Tests for Role enum."""

    def test_role_values(self):
        """Test that Role enum has expected values."""
        from src.api.security.rbac import Role

        assert Role.OWNER.value == "owner"
        assert Role.ADMIN.value == "admin"
        assert Role.MEMBER.value == "member"
        assert Role.VIEWER.value == "viewer"

    def test_project_roles(self):
        """Test project-level roles exist."""
        from src.api.security.rbac import Role

        project_roles = [
            Role.PROJECT_ADMIN,
            Role.PROJECT_MEMBER,
            Role.PROJECT_VIEWER,
        ]
        assert all(role for role in project_roles)

    def test_special_roles(self):
        """Test special roles exist."""
        from src.api.security.rbac import Role

        special_roles = [
            Role.BILLING_ADMIN,
            Role.SECURITY_ADMIN,
            Role.SERVICE_ACCOUNT,
            Role.API_USER,
        ]
        assert all(role for role in special_roles)


# =============================================================================
# Role-Permission Mapping Tests
# =============================================================================

class TestRolePermissionMapping:
    """Tests for role to permission mapping."""

    def test_owner_has_all_org_permissions(self):
        """Test that owner role has all organization permissions."""
        from src.api.security.rbac import ROLE_PERMISSIONS, Role, Permission

        owner_perms = ROLE_PERMISSIONS[Role.OWNER]

        assert Permission.ORG_READ in owner_perms
        assert Permission.ORG_WRITE in owner_perms
        assert Permission.ORG_DELETE in owner_perms
        assert Permission.ORG_MANAGE_MEMBERS in owner_perms
        assert Permission.ORG_MANAGE_BILLING in owner_perms

    def test_admin_has_write_but_not_delete_org(self):
        """Test that admin can write but not delete organization."""
        from src.api.security.rbac import ROLE_PERMISSIONS, Role, Permission

        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]

        assert Permission.ORG_READ in admin_perms
        assert Permission.ORG_WRITE in admin_perms
        assert Permission.ORG_DELETE not in admin_perms

    def test_member_permissions(self):
        """Test member role permissions."""
        from src.api.security.rbac import ROLE_PERMISSIONS, Role, Permission

        member_perms = ROLE_PERMISSIONS[Role.MEMBER]

        assert Permission.ORG_READ in member_perms
        assert Permission.ORG_WRITE not in member_perms
        assert Permission.TEST_READ in member_perms
        assert Permission.TEST_WRITE in member_perms
        assert Permission.TEST_EXECUTE in member_perms
        assert Permission.TEST_DELETE not in member_perms

    def test_viewer_read_only(self):
        """Test viewer role has only read permissions."""
        from src.api.security.rbac import ROLE_PERMISSIONS, Role, Permission

        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]

        assert Permission.ORG_READ in viewer_perms
        assert Permission.PROJECT_READ in viewer_perms
        assert Permission.TEST_READ in viewer_perms
        assert Permission.RESULTS_READ in viewer_perms

        # Should not have write permissions
        assert Permission.ORG_WRITE not in viewer_perms
        assert Permission.PROJECT_WRITE not in viewer_perms
        assert Permission.TEST_WRITE not in viewer_perms

    def test_billing_admin_permissions(self):
        """Test billing admin has only billing-related permissions."""
        from src.api.security.rbac import ROLE_PERMISSIONS, Role, Permission

        billing_perms = ROLE_PERMISSIONS[Role.BILLING_ADMIN]

        assert Permission.ORG_READ in billing_perms
        assert Permission.ORG_MANAGE_BILLING in billing_perms
        assert Permission.ORG_WRITE not in billing_perms
        assert Permission.TEST_EXECUTE not in billing_perms

    def test_security_admin_permissions(self):
        """Test security admin has audit and key permissions."""
        from src.api.security.rbac import ROLE_PERMISSIONS, Role, Permission

        security_perms = ROLE_PERMISSIONS[Role.SECURITY_ADMIN]

        assert Permission.AUDIT_READ in security_perms
        assert Permission.AUDIT_EXPORT in security_perms
        assert Permission.API_KEY_READ in security_perms
        assert Permission.API_KEY_REVOKE in security_perms
        # Should not have create permission
        assert Permission.API_KEY_CREATE not in security_perms

    def test_api_user_minimal_permissions(self):
        """Test API user has minimal execution permissions."""
        from src.api.security.rbac import ROLE_PERMISSIONS, Role, Permission

        api_user_perms = ROLE_PERMISSIONS[Role.API_USER]

        assert Permission.TEST_READ in api_user_perms
        assert Permission.TEST_EXECUTE in api_user_perms
        assert Permission.RESULTS_READ in api_user_perms
        assert Permission.TEST_WRITE not in api_user_perms
        assert Permission.ORG_WRITE not in api_user_perms


# =============================================================================
# RBACManager Tests
# =============================================================================

class TestRBACManager:
    """Tests for RBACManager class."""

    @pytest.fixture
    def rbac_manager(self):
        """Create RBACManager instance."""
        from src.api.security.rbac import RBACManager
        return RBACManager()

    def test_get_permissions_for_role_owner(self, rbac_manager):
        """Test getting permissions for owner role."""
        from src.api.security.rbac import Role, Permission

        perms = rbac_manager.get_permissions_for_role(Role.OWNER)

        assert isinstance(perms, set)
        assert Permission.ADMIN_FULL_ACCESS in perms
        assert Permission.ORG_DELETE in perms

    def test_get_permissions_for_role_unknown(self, rbac_manager):
        """Test getting permissions for unknown role returns empty set."""
        perms = rbac_manager.get_permissions_for_role("unknown_role")
        assert perms == set()

    def test_get_permissions_for_roles_multiple(self, rbac_manager):
        """Test getting combined permissions for multiple roles."""
        from src.api.security.rbac import Permission

        perms = rbac_manager.get_permissions_for_roles(["member", "billing_admin"])

        # Should have member permissions
        assert Permission.TEST_READ in perms
        assert Permission.TEST_WRITE in perms
        # Should also have billing_admin permissions
        assert Permission.ORG_MANAGE_BILLING in perms

    def test_get_permissions_for_roles_with_unknown(self, rbac_manager):
        """Test that unknown roles are skipped."""
        from src.api.security.rbac import Permission

        perms = rbac_manager.get_permissions_for_roles(["member", "unknown_role"])

        # Should still have member permissions
        assert Permission.TEST_READ in perms

    def test_get_permissions_for_roles_empty(self, rbac_manager):
        """Test getting permissions for empty roles list."""
        perms = rbac_manager.get_permissions_for_roles([])
        assert perms == set()


class TestRBACManagerHasPermission:
    """Tests for RBACManager.has_permission method."""

    @pytest.fixture
    def rbac_manager(self):
        """Create RBACManager instance."""
        from src.api.security.rbac import RBACManager
        return RBACManager()

    def test_has_permission_with_role(self, rbac_manager):
        """Test permission check based on roles."""
        from src.api.security.rbac import Permission

        # Admin should have TEST_EXECUTE
        assert rbac_manager.has_permission(["admin"], Permission.TEST_EXECUTE) is True

        # Viewer should not have TEST_EXECUTE
        assert rbac_manager.has_permission(["viewer"], Permission.TEST_EXECUTE) is False

    def test_has_permission_scopes_none(self, rbac_manager):
        """Test permission check when scopes is None (no restriction)."""
        from src.api.security.rbac import Permission

        # With scopes=None, should fall through to role check
        result = rbac_manager.has_permission(
            roles=["admin"],
            required_permission=Permission.TEST_EXECUTE,
            scopes=None,
        )
        assert result is True

    def test_has_permission_scopes_empty_denies(self, rbac_manager):
        """Test that empty scopes list denies all permissions."""
        from src.api.security.rbac import Permission

        # Empty scopes should deny everything
        result = rbac_manager.has_permission(
            roles=["admin"],  # Admin role normally has this permission
            required_permission=Permission.TEST_READ,
            scopes=[],  # Empty scopes = no permissions
        )
        assert result is False

    def test_has_permission_scope_allows(self, rbac_manager):
        """Test that scopes can allow permissions."""
        from src.api.security.rbac import Permission

        # Read scope should allow TEST_READ
        result = rbac_manager.has_permission(
            roles=["api_user"],
            required_permission=Permission.TEST_READ,
            scopes=["read"],
        )
        assert result is True

    def test_has_permission_scope_restricts(self, rbac_manager):
        """Test that scopes restrict available permissions."""
        from src.api.security.rbac import Permission

        # Read scope should not allow TEST_WRITE
        result = rbac_manager.has_permission(
            roles=["admin"],  # Admin can normally write
            required_permission=Permission.TEST_WRITE,
            scopes=["read"],  # But scope only allows read
        )
        assert result is False

    def test_has_permission_execute_scope(self, rbac_manager):
        """Test execute scope allows test execution."""
        from src.api.security.rbac import Permission

        result = rbac_manager.has_permission(
            roles=["api_user"],
            required_permission=Permission.TEST_EXECUTE,
            scopes=["execute"],
        )
        assert result is True

    def test_has_permission_admin_scope(self, rbac_manager):
        """Test admin scope allows full access (requires role with permission too)."""
        from src.api.security.rbac import Permission

        # Owner has ADMIN_FULL_ACCESS permission, and admin scope allows it
        result = rbac_manager.has_permission(
            roles=["owner"],
            required_permission=Permission.ADMIN_FULL_ACCESS,
            scopes=["admin"],
        )
        assert result is True

        # Member doesn't have ADMIN_FULL_ACCESS, so even with admin scope it fails
        result = rbac_manager.has_permission(
            roles=["member"],
            required_permission=Permission.ADMIN_FULL_ACCESS,
            scopes=["admin"],
        )
        assert result is False

    def test_has_permission_multiple_scopes(self, rbac_manager):
        """Test multiple scopes combine permissions (requires role with permission too)."""
        from src.api.security.rbac import Permission

        # Member has TEST_WRITE permission, and write scope allows it
        result = rbac_manager.has_permission(
            roles=["member"],
            required_permission=Permission.TEST_WRITE,
            scopes=["read", "write"],  # Combined scopes
        )
        assert result is True

        # api_user doesn't have TEST_WRITE, so even with write scope it fails
        result = rbac_manager.has_permission(
            roles=["api_user"],
            required_permission=Permission.TEST_WRITE,
            scopes=["read", "write"],
        )
        assert result is False


class TestRBACManagerCustomPermissions:
    """Tests for custom permission management."""

    @pytest.fixture
    def rbac_manager(self):
        """Create RBACManager instance."""
        from src.api.security.rbac import RBACManager
        return RBACManager()

    def test_add_custom_permission(self, rbac_manager):
        """Test adding custom permission to user."""
        from src.api.security.rbac import Permission

        rbac_manager.add_custom_permission("user_123", Permission.ADMIN_FULL_ACCESS)

        assert "user_123" in rbac_manager._custom_permissions
        assert Permission.ADMIN_FULL_ACCESS in rbac_manager._custom_permissions["user_123"]

    def test_add_multiple_custom_permissions(self, rbac_manager):
        """Test adding multiple custom permissions to same user."""
        from src.api.security.rbac import Permission

        rbac_manager.add_custom_permission("user_123", Permission.ADMIN_FULL_ACCESS)
        rbac_manager.add_custom_permission("user_123", Permission.AUDIT_READ)

        assert len(rbac_manager._custom_permissions["user_123"]) == 2

    def test_remove_custom_permission(self, rbac_manager):
        """Test removing custom permission from user."""
        from src.api.security.rbac import Permission

        rbac_manager.add_custom_permission("user_123", Permission.ADMIN_FULL_ACCESS)
        rbac_manager.remove_custom_permission("user_123", Permission.ADMIN_FULL_ACCESS)

        assert Permission.ADMIN_FULL_ACCESS not in rbac_manager._custom_permissions.get("user_123", set())

    def test_remove_custom_permission_nonexistent_user(self, rbac_manager):
        """Test removing permission from user that doesn't exist."""
        from src.api.security.rbac import Permission

        # Should not raise error
        rbac_manager.remove_custom_permission("nonexistent_user", Permission.ADMIN_FULL_ACCESS)


# =============================================================================
# Global RBAC Manager Tests
# =============================================================================

class TestGetRBACManager:
    """Tests for get_rbac_manager function."""

    def test_get_rbac_manager_returns_instance(self):
        """Test that get_rbac_manager returns an RBACManager."""
        from src.api.security.rbac import get_rbac_manager, RBACManager

        manager = get_rbac_manager()
        assert isinstance(manager, RBACManager)

    def test_get_rbac_manager_singleton(self):
        """Test that get_rbac_manager returns same instance."""
        from src.api.security.rbac import get_rbac_manager

        manager1 = get_rbac_manager()
        manager2 = get_rbac_manager()

        assert manager1 is manager2


# =============================================================================
# Permission Decorator Tests
# =============================================================================

class TestRequirePermissionDecorator:
    """Tests for require_permission decorator."""

    @pytest.fixture
    def mock_user(self):
        """Create mock user context."""
        from src.api.security.auth import UserContext, AuthMethod

        return UserContext(
            user_id="user_123",
            organization_id="org_456",
            roles=["member"],
            scopes=["read", "write", "execute"],
            auth_method=AuthMethod.JWT,
        )

    @pytest.fixture
    def admin_user(self):
        """Create admin user context."""
        from src.api.security.auth import UserContext, AuthMethod

        return UserContext(
            user_id="admin_123",
            organization_id="org_456",
            roles=["admin"],
            scopes=["read", "write", "execute", "admin"],
            auth_method=AuthMethod.JWT,
        )

    @pytest.mark.asyncio
    async def test_require_permission_has_permission(self, admin_user):
        """Test require_permission when user has permission."""
        from src.api.security.rbac import require_permission, Permission

        @require_permission(Permission.TEST_WRITE)
        async def protected_endpoint(user=None):
            return {"message": "success"}

        result = await protected_endpoint(user=admin_user)
        assert result["message"] == "success"

    @pytest.mark.asyncio
    async def test_require_permission_missing_permission(self, mock_user):
        """Test require_permission when user lacks permission."""
        from src.api.security.rbac import require_permission, Permission
        from fastapi import HTTPException

        @require_permission(Permission.ORG_DELETE)  # Member doesn't have this
        async def protected_endpoint(user=None):
            return {"message": "success"}

        with pytest.raises(HTTPException) as exc_info:
            await protected_endpoint(user=mock_user)

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_require_permission_no_user(self):
        """Test require_permission when no user provided."""
        from src.api.security.rbac import require_permission, Permission
        from fastapi import HTTPException

        @require_permission(Permission.TEST_READ)
        async def protected_endpoint(user=None):
            return {"message": "success"}

        with pytest.raises(HTTPException) as exc_info:
            await protected_endpoint()

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_require_permission_multiple_permissions(self, admin_user):
        """Test require_permission with multiple required permissions."""
        from src.api.security.rbac import require_permission, Permission

        @require_permission(Permission.TEST_READ, Permission.TEST_WRITE)
        async def protected_endpoint(user=None):
            return {"message": "success"}

        result = await protected_endpoint(user=admin_user)
        assert result["message"] == "success"

    @pytest.mark.asyncio
    async def test_require_permission_user_from_request(self, admin_user):
        """Test require_permission gets user from request state."""
        from src.api.security.rbac import require_permission, Permission

        @require_permission(Permission.TEST_READ)
        async def protected_endpoint(request=None):
            return {"message": "success"}

        mock_request = MagicMock()
        mock_request.state = MagicMock()
        mock_request.state.user = admin_user

        result = await protected_endpoint(request=mock_request)
        assert result["message"] == "success"


class TestRequireAnyPermissionDecorator:
    """Tests for require_any_permission decorator."""

    @pytest.fixture
    def viewer_user(self):
        """Create viewer user context."""
        from src.api.security.auth import UserContext, AuthMethod

        return UserContext(
            user_id="viewer_123",
            roles=["viewer"],
            scopes=["read"],
            auth_method=AuthMethod.JWT,
        )

    @pytest.mark.asyncio
    async def test_require_any_permission_has_one(self, viewer_user):
        """Test require_any_permission when user has one of the permissions."""
        from src.api.security.rbac import require_any_permission, Permission

        @require_any_permission(Permission.TEST_READ, Permission.TEST_WRITE)
        async def protected_endpoint(user=None):
            return {"message": "success"}

        # Viewer has TEST_READ but not TEST_WRITE
        result = await protected_endpoint(user=viewer_user)
        assert result["message"] == "success"

    @pytest.mark.asyncio
    async def test_require_any_permission_has_none(self, viewer_user):
        """Test require_any_permission when user has none of the permissions."""
        from src.api.security.rbac import require_any_permission, Permission
        from fastapi import HTTPException

        @require_any_permission(Permission.TEST_WRITE, Permission.TEST_DELETE)
        async def protected_endpoint(user=None):
            return {"message": "success"}

        with pytest.raises(HTTPException) as exc_info:
            await protected_endpoint(user=viewer_user)

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_require_any_permission_no_user(self):
        """Test require_any_permission when no user provided."""
        from src.api.security.rbac import require_any_permission, Permission
        from fastapi import HTTPException

        @require_any_permission(Permission.TEST_READ)
        async def protected_endpoint(user=None):
            return {"message": "success"}

        with pytest.raises(HTTPException) as exc_info:
            await protected_endpoint()

        assert exc_info.value.status_code == 401


# =============================================================================
# ResourceAccess Tests
# =============================================================================

class TestResourceAccess:
    """Tests for ResourceAccess dataclass."""

    def test_resource_access_creation(self):
        """Test creating ResourceAccess."""
        from src.api.security.rbac import ResourceAccess, Permission

        access = ResourceAccess(
            resource_type="test",
            resource_id="test_123",
            user_id="user_456",
            organization_id="org_789",
            permissions={Permission.TEST_READ, Permission.TEST_WRITE},
            granted_by="admin_user",
        )

        assert access.resource_type == "test"
        assert access.resource_id == "test_123"
        assert access.user_id == "user_456"
        assert Permission.TEST_READ in access.permissions

    def test_resource_access_defaults(self):
        """Test ResourceAccess default values."""
        from src.api.security.rbac import ResourceAccess

        access = ResourceAccess(
            resource_type="project",
            resource_id="proj_123",
        )

        assert access.user_id is None
        assert access.organization_id is None
        assert access.permissions == set()
        assert access.expires_at is None

    def test_resource_access_with_expiration(self):
        """Test ResourceAccess with expiration."""
        from src.api.security.rbac import ResourceAccess

        expiry = datetime.now(timezone.utc) + timedelta(days=7)
        access = ResourceAccess(
            resource_type="test",
            resource_id="test_123",
            expires_at=expiry,
        )

        assert access.expires_at == expiry


# =============================================================================
# ResourceACL Tests
# =============================================================================

class TestResourceACL:
    """Tests for ResourceACL class."""

    @pytest.fixture
    def acl(self):
        """Create ResourceACL instance."""
        from src.api.security.rbac import ResourceACL
        return ResourceACL()

    def test_grant_access_user(self, acl):
        """Test granting access to a user."""
        from src.api.security.rbac import Permission

        acl.grant_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_456",
            permissions={Permission.TEST_READ, Permission.TEST_EXECUTE},
            granted_by="admin",
        )

        key = acl._get_key("test", "test_123")
        assert key in acl._acl
        assert len(acl._acl[key]) == 1
        assert acl._acl[key][0].user_id == "user_456"

    def test_grant_access_organization(self, acl):
        """Test granting access to an organization."""
        from src.api.security.rbac import Permission

        acl.grant_access(
            resource_type="project",
            resource_id="proj_123",
            organization_id="org_456",
            permissions={Permission.PROJECT_READ},
        )

        key = acl._get_key("project", "proj_123")
        assert acl._acl[key][0].organization_id == "org_456"

    def test_revoke_access_user(self, acl):
        """Test revoking access from a user."""
        from src.api.security.rbac import Permission

        acl.grant_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_456",
            permissions={Permission.TEST_READ},
        )

        acl.revoke_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_456",
        )

        key = acl._get_key("test", "test_123")
        assert len(acl._acl[key]) == 0

    def test_revoke_access_nonexistent(self, acl):
        """Test revoking access from nonexistent resource."""
        # Should not raise error
        acl.revoke_access(
            resource_type="nonexistent",
            resource_id="none_123",
            user_id="user_456",
        )

    def test_has_access_user_granted(self, acl):
        """Test checking access for user with granted permission."""
        from src.api.security.rbac import Permission

        acl.grant_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_456",
            permissions={Permission.TEST_READ, Permission.TEST_EXECUTE},
        )

        result = acl.has_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_456",
            organization_id=None,
            required_permission=Permission.TEST_READ,
        )
        assert result is True

    def test_has_access_user_not_granted(self, acl):
        """Test checking access for user without granted permission."""
        from src.api.security.rbac import Permission

        acl.grant_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_456",
            permissions={Permission.TEST_READ},
        )

        result = acl.has_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_456",
            organization_id=None,
            required_permission=Permission.TEST_WRITE,  # Not granted
        )
        assert result is False

    def test_has_access_organization_granted(self, acl):
        """Test checking access via organization membership."""
        from src.api.security.rbac import Permission

        acl.grant_access(
            resource_type="project",
            resource_id="proj_123",
            organization_id="org_456",
            permissions={Permission.PROJECT_READ},
        )

        result = acl.has_access(
            resource_type="project",
            resource_id="proj_123",
            user_id="some_user",
            organization_id="org_456",
            required_permission=Permission.PROJECT_READ,
        )
        assert result is True

    def test_has_access_expired(self, acl):
        """Test that expired access is denied."""
        from src.api.security.rbac import Permission

        past_time = datetime.now(timezone.utc) - timedelta(days=1)
        acl.grant_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_456",
            permissions={Permission.TEST_READ},
            expires_at=past_time,
        )

        result = acl.has_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_456",
            organization_id=None,
            required_permission=Permission.TEST_READ,
        )
        assert result is False

    def test_has_access_not_expired(self, acl):
        """Test that non-expired access is allowed."""
        from src.api.security.rbac import Permission

        future_time = datetime.now(timezone.utc) + timedelta(days=7)
        acl.grant_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_456",
            permissions={Permission.TEST_READ},
            expires_at=future_time,
        )

        result = acl.has_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_456",
            organization_id=None,
            required_permission=Permission.TEST_READ,
        )
        assert result is True

    def test_has_access_no_acl_entry(self, acl):
        """Test checking access when no ACL entry exists."""
        from src.api.security.rbac import Permission

        result = acl.has_access(
            resource_type="nonexistent",
            resource_id="none_123",
            user_id="user_456",
            organization_id=None,
            required_permission=Permission.TEST_READ,
        )
        assert result is False

    def test_multiple_grants_same_resource(self, acl):
        """Test multiple grants to same resource."""
        from src.api.security.rbac import Permission

        acl.grant_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_1",
            permissions={Permission.TEST_READ},
        )
        acl.grant_access(
            resource_type="test",
            resource_id="test_123",
            user_id="user_2",
            permissions={Permission.TEST_WRITE},
        )

        key = acl._get_key("test", "test_123")
        assert len(acl._acl[key]) == 2


# =============================================================================
# Global Resource ACL Tests
# =============================================================================

class TestGetResourceACL:
    """Tests for get_resource_acl function."""

    def test_get_resource_acl_returns_instance(self):
        """Test that get_resource_acl returns a ResourceACL."""
        from src.api.security.rbac import get_resource_acl, ResourceACL

        acl = get_resource_acl()
        assert isinstance(acl, ResourceACL)

    def test_get_resource_acl_singleton(self):
        """Test that get_resource_acl returns same instance."""
        from src.api.security.rbac import get_resource_acl

        acl1 = get_resource_acl()
        acl2 = get_resource_acl()

        assert acl1 is acl2
