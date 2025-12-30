"""Tests for team collaboration module."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta


class TestRole:
    """Tests for Role enum."""

    def test_role_values(self, mock_env_vars):
        """Test role enum values."""
        from src.collaboration.team import Role

        assert Role.ADMIN == "admin"
        assert Role.MANAGER == "manager"
        assert Role.DEVELOPER == "developer"
        assert Role.VIEWER == "viewer"
        assert Role.CI_BOT == "ci_bot"


class TestPermission:
    """Tests for Permission enum."""

    def test_permission_values(self, mock_env_vars):
        """Test permission enum values."""
        from src.collaboration.team import Permission

        assert Permission.CREATE_TEST == "create_test"
        assert Permission.RUN_TEST == "run_test"
        assert Permission.VIEW_RESULTS == "view_results"
        assert Permission.MANAGE_SETTINGS == "manage_settings"
        assert Permission.VIEW_AUDIT_LOG == "view_audit_log"


class TestRolePermissions:
    """Tests for ROLE_PERMISSIONS mapping."""

    def test_admin_has_all_permissions(self, mock_env_vars):
        """Test admin has all permissions."""
        from src.collaboration.team import Role, Permission, ROLE_PERMISSIONS

        admin_permissions = ROLE_PERMISSIONS[Role.ADMIN]
        assert len(admin_permissions) == len(Permission)

    def test_viewer_has_limited_permissions(self, mock_env_vars):
        """Test viewer has limited permissions."""
        from src.collaboration.team import Role, Permission, ROLE_PERMISSIONS

        viewer_permissions = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.VIEW_TEST in viewer_permissions
        assert Permission.VIEW_RESULTS in viewer_permissions
        assert Permission.CREATE_TEST not in viewer_permissions
        assert Permission.MANAGE_SETTINGS not in viewer_permissions

    def test_ci_bot_permissions(self, mock_env_vars):
        """Test CI bot has appropriate permissions."""
        from src.collaboration.team import Role, Permission, ROLE_PERMISSIONS

        ci_permissions = ROLE_PERMISSIONS[Role.CI_BOT]
        assert Permission.RUN_TEST in ci_permissions
        assert Permission.VIEW_TEST in ci_permissions
        assert Permission.CREATE_TEST not in ci_permissions


class TestUser:
    """Tests for User dataclass."""

    def test_user_creation(self, mock_env_vars):
        """Test User creation."""
        from src.collaboration.team import User, Role

        user = User(
            id="user-123",
            email="test@example.com",
            name="Test User",
            role=Role.DEVELOPER,
            team_id="team-456",
        )

        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.role == Role.DEVELOPER
        assert user.avatar_url is None
        assert user.last_login is None
        assert user.mfa_enabled is False

    def test_user_with_sso(self, mock_env_vars):
        """Test User with SSO provider."""
        from src.collaboration.team import User, Role

        user = User(
            id="user-789",
            email="sso@example.com",
            name="SSO User",
            role=Role.MANAGER,
            team_id="team-456",
            sso_provider="okta",
            mfa_enabled=True,
        )

        assert user.sso_provider == "okta"
        assert user.mfa_enabled is True


class TestTeam:
    """Tests for Team dataclass."""

    def test_team_creation(self, mock_env_vars):
        """Test Team creation."""
        from src.collaboration.team import Team

        team = Team(
            id="team-123",
            name="Test Team",
            slug="test-team",
            owner_id="owner-456",
        )

        assert team.id == "team-123"
        assert team.name == "Test Team"
        assert team.slug == "test-team"
        assert team.plan == "free"
        assert team.settings == {}
        assert team.sso_config is None


class TestWorkspace:
    """Tests for Workspace dataclass."""

    def test_workspace_creation(self, mock_env_vars):
        """Test Workspace creation."""
        from src.collaboration.team import Workspace

        workspace = Workspace(
            id="ws-123",
            team_id="team-456",
            name="Frontend Tests",
            description="Tests for frontend components",
            created_by="user-789",
        )

        assert workspace.id == "ws-123"
        assert workspace.team_id == "team-456"
        assert workspace.name == "Frontend Tests"


class TestComment:
    """Tests for Comment dataclass."""

    def test_comment_creation(self, mock_env_vars):
        """Test Comment creation."""
        from src.collaboration.team import Comment

        comment = Comment(
            id="comment-123",
            user_id="user-456",
            target_type="test_result",
            target_id="result-789",
            content="This test is flaky!",
        )

        assert comment.id == "comment-123"
        assert comment.content == "This test is flaky!"
        assert comment.mentions == []
        assert comment.reactions == {}


class TestApprovalRequest:
    """Tests for ApprovalRequest dataclass."""

    def test_approval_request_creation(self, mock_env_vars):
        """Test ApprovalRequest creation."""
        from src.collaboration.team import ApprovalRequest

        request = ApprovalRequest(
            id="req-123",
            type="baseline_update",
            requester_id="user-456",
            target_id="baseline-789",
            description="Update login baseline",
        )

        assert request.id == "req-123"
        assert request.status == "pending"
        assert request.reviewer_id is None
        assert request.resolved_at is None


class TestAuditLogEntry:
    """Tests for AuditLogEntry dataclass."""

    def test_audit_log_entry_creation(self, mock_env_vars):
        """Test AuditLogEntry creation."""
        from src.collaboration.team import AuditLogEntry

        entry = AuditLogEntry(
            id="log-123",
            user_id="user-456",
            action="create_test",
            resource_type="test",
            resource_id="test-789",
            details={"test_name": "Login Test"},
        )

        assert entry.id == "log-123"
        assert entry.action == "create_test"
        assert entry.ip_address is None


class TestTeamManager:
    """Tests for TeamManager class."""

    def test_manager_init(self, mock_env_vars):
        """Test TeamManager initialization."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()

        assert manager.users == {}
        assert manager.teams == {}
        assert manager.workspaces == {}
        assert manager.comments == {}
        assert manager.approvals == {}
        assert manager.audit_log == []
        assert manager.sessions == {}

    def test_create_user(self, mock_env_vars):
        """Test create_user method."""
        from src.collaboration.team import TeamManager, Role

        manager = TeamManager()
        user = manager.create_user(
            email="test@example.com",
            name="Test User",
            team_id="team-123",
            role=Role.DEVELOPER,
        )

        assert user.email == "test@example.com"
        assert user.role == Role.DEVELOPER
        assert user.id in manager.users
        assert len(manager.audit_log) == 1

    def test_get_user(self, mock_env_vars):
        """Test get_user method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        user = manager.create_user("test@example.com", "Test", "team-1")

        found = manager.get_user(user.id)
        assert found is user

        not_found = manager.get_user("nonexistent")
        assert not_found is None

    def test_get_user_by_email(self, mock_env_vars):
        """Test get_user_by_email method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        user = manager.create_user("test@example.com", "Test", "team-1")

        found = manager.get_user_by_email("test@example.com")
        assert found is user

        not_found = manager.get_user_by_email("other@example.com")
        assert not_found is None

    def test_update_user_role(self, mock_env_vars):
        """Test update_user_role method."""
        from src.collaboration.team import TeamManager, Role

        manager = TeamManager()
        user = manager.create_user("test@example.com", "Test", "team-1")
        admin = manager.create_user("admin@example.com", "Admin", "team-1", Role.ADMIN)

        success = manager.update_user_role(user.id, Role.MANAGER, admin.id)

        assert success is True
        assert user.role == Role.MANAGER

    def test_update_user_role_not_found(self, mock_env_vars):
        """Test update_user_role for nonexistent user."""
        from src.collaboration.team import TeamManager, Role

        manager = TeamManager()
        success = manager.update_user_role("nonexistent", Role.ADMIN, "updater")

        assert success is False

    def test_has_permission(self, mock_env_vars):
        """Test has_permission method."""
        from src.collaboration.team import TeamManager, Role, Permission

        manager = TeamManager()
        dev = manager.create_user("dev@example.com", "Dev", "team-1", Role.DEVELOPER)
        viewer = manager.create_user("viewer@example.com", "Viewer", "team-1", Role.VIEWER)

        assert manager.has_permission(dev.id, Permission.CREATE_TEST) is True
        assert manager.has_permission(viewer.id, Permission.CREATE_TEST) is False
        assert manager.has_permission(viewer.id, Permission.VIEW_TEST) is True

    def test_has_permission_nonexistent_user(self, mock_env_vars):
        """Test has_permission for nonexistent user."""
        from src.collaboration.team import TeamManager, Permission

        manager = TeamManager()
        assert manager.has_permission("nonexistent", Permission.VIEW_TEST) is False

    def test_check_permission_success(self, mock_env_vars):
        """Test check_permission when user has permission."""
        from src.collaboration.team import TeamManager, Role, Permission

        manager = TeamManager()
        admin = manager.create_user("admin@example.com", "Admin", "team-1", Role.ADMIN)

        # Should not raise
        manager.check_permission(admin.id, Permission.MANAGE_SETTINGS)

    def test_check_permission_failure(self, mock_env_vars):
        """Test check_permission when user lacks permission."""
        from src.collaboration.team import TeamManager, Role, Permission

        manager = TeamManager()
        viewer = manager.create_user("viewer@example.com", "Viewer", "team-1", Role.VIEWER)

        with pytest.raises(PermissionError):
            manager.check_permission(viewer.id, Permission.CREATE_TEST)

    def test_create_team(self, mock_env_vars):
        """Test create_team method."""
        from src.collaboration.team import TeamManager, Role

        manager = TeamManager()
        team, owner = manager.create_team(
            name="My Team",
            owner_email="owner@example.com",
            owner_name="Owner",
        )

        assert team.name == "My Team"
        assert team.slug == "my-team"
        assert team.owner_id == owner.id
        assert owner.role == Role.ADMIN
        assert team.id in manager.teams

    def test_get_team(self, mock_env_vars):
        """Test get_team method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, _ = manager.create_team("Test Team", "owner@example.com", "Owner")

        found = manager.get_team(team.id)
        assert found is team

        not_found = manager.get_team("nonexistent")
        assert not_found is None

    def test_get_team_members(self, mock_env_vars):
        """Test get_team_members method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")
        member1 = manager.create_user("member1@example.com", "Member 1", team.id)
        member2 = manager.create_user("member2@example.com", "Member 2", team.id)

        members = manager.get_team_members(team.id)

        assert len(members) == 3  # owner + 2 members
        assert owner in members
        assert member1 in members
        assert member2 in members

    def test_invite_member(self, mock_env_vars):
        """Test invite_member method."""
        from src.collaboration.team import TeamManager, Role

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")

        token = manager.invite_member(
            team.id, "newmember@example.com", Role.DEVELOPER, owner.id
        )

        assert token is not None
        assert len(token) > 0

    def test_invite_member_without_permission(self, mock_env_vars):
        """Test invite_member without permission."""
        from src.collaboration.team import TeamManager, Role

        manager = TeamManager()
        team, _ = manager.create_team("Test Team", "owner@example.com", "Owner")
        viewer = manager.create_user("viewer@example.com", "Viewer", team.id, Role.VIEWER)

        with pytest.raises(PermissionError):
            manager.invite_member(team.id, "new@example.com", Role.DEVELOPER, viewer.id)

    def test_create_workspace(self, mock_env_vars):
        """Test create_workspace method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")

        workspace = manager.create_workspace(
            team.id, "API Tests", "Tests for API endpoints", owner.id
        )

        assert workspace.name == "API Tests"
        assert workspace.team_id == team.id
        assert workspace.id in manager.workspaces

    def test_get_team_workspaces(self, mock_env_vars):
        """Test get_team_workspaces method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")
        ws1 = manager.create_workspace(team.id, "WS1", "", owner.id)
        ws2 = manager.create_workspace(team.id, "WS2", "", owner.id)

        workspaces = manager.get_team_workspaces(team.id)

        assert len(workspaces) == 2
        assert ws1 in workspaces
        assert ws2 in workspaces

    def test_add_comment(self, mock_env_vars):
        """Test add_comment method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")

        comment = manager.add_comment(
            owner.id, "test_result", "result-123",
            "This test is flaky!", mentions=["user-456"]
        )

        assert comment.content == "This test is flaky!"
        assert comment.mentions == ["user-456"]
        assert comment.id in manager.comments

    def test_get_comments(self, mock_env_vars):
        """Test get_comments method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")

        manager.add_comment(owner.id, "test_result", "result-123", "Comment 1")
        manager.add_comment(owner.id, "test_result", "result-123", "Comment 2")
        manager.add_comment(owner.id, "test_result", "result-456", "Other comment")

        comments = manager.get_comments("test_result", "result-123")

        assert len(comments) == 2

    def test_add_reaction(self, mock_env_vars):
        """Test add_reaction method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")
        comment = manager.add_comment(owner.id, "test_result", "result-123", "Great!")

        success = manager.add_reaction(comment.id, owner.id, "👍")

        assert success is True
        assert "👍" in comment.reactions
        assert owner.id in comment.reactions["👍"]

    def test_add_reaction_nonexistent_comment(self, mock_env_vars):
        """Test add_reaction for nonexistent comment."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        success = manager.add_reaction("nonexistent", "user-123", "👍")

        assert success is False

    def test_request_approval(self, mock_env_vars):
        """Test request_approval method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")

        request = manager.request_approval(
            owner.id, "baseline_update", "baseline-123",
            "Update login baseline screenshot"
        )

        assert request.type == "baseline_update"
        assert request.status == "pending"
        assert request.id in manager.approvals

    def test_approve_request(self, mock_env_vars):
        """Test approve_request method."""
        from src.collaboration.team import TeamManager, Role

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")
        manager_user = manager.create_user("mgr@example.com", "Manager", team.id, Role.MANAGER)

        request = manager.request_approval(
            owner.id, "baseline_update", "baseline-123", "Update baseline"
        )

        success = manager.approve_request(request.id, manager_user.id, "LGTM")

        assert success is True
        assert request.status == "approved"
        assert request.reviewer_id == manager_user.id
        assert "LGTM" in request.comments

    def test_approve_request_not_found(self, mock_env_vars):
        """Test approve_request for nonexistent request."""
        from src.collaboration.team import TeamManager, Role

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")

        success = manager.approve_request("nonexistent", owner.id)

        assert success is False

    def test_approve_request_already_resolved(self, mock_env_vars):
        """Test approve_request for already resolved request."""
        from src.collaboration.team import TeamManager, Role

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")
        request = manager.request_approval(owner.id, "baseline_update", "b-1", "desc")

        manager.approve_request(request.id, owner.id)
        success = manager.approve_request(request.id, owner.id)  # Try again

        assert success is False

    def test_reject_request(self, mock_env_vars):
        """Test reject_request method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")

        request = manager.request_approval(
            owner.id, "baseline_update", "baseline-123", "Update baseline"
        )

        success = manager.reject_request(request.id, owner.id, "Not needed")

        assert success is True
        assert request.status == "rejected"
        assert "Rejected: Not needed" in request.comments

    def test_get_pending_approvals(self, mock_env_vars):
        """Test get_pending_approvals method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")

        req1 = manager.request_approval(owner.id, "type1", "t1", "desc1")
        req2 = manager.request_approval(owner.id, "type2", "t2", "desc2")
        manager.approve_request(req1.id, owner.id)

        pending = manager.get_pending_approvals(team.id)

        assert len(pending) == 1
        assert req2 in pending

    def test_get_audit_log(self, mock_env_vars):
        """Test get_audit_log method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")

        # Create some activity
        manager.create_user("test@example.com", "Test", team.id)
        manager.create_workspace(team.id, "WS", "", owner.id)

        entries = manager.get_audit_log(owner.id, limit=10)

        assert len(entries) >= 2

    def test_get_audit_log_filter_by_action(self, mock_env_vars):
        """Test get_audit_log filtered by action."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")
        manager.create_workspace(team.id, "WS1", "", owner.id)
        manager.create_workspace(team.id, "WS2", "", owner.id)

        entries = manager.get_audit_log(owner.id, action="create_workspace")

        assert len(entries) == 2
        assert all(e.action == "create_workspace" for e in entries)

    def test_get_audit_log_without_permission(self, mock_env_vars):
        """Test get_audit_log without permission."""
        from src.collaboration.team import TeamManager, Role

        manager = TeamManager()
        team, _ = manager.create_team("Test Team", "owner@example.com", "Owner")
        viewer = manager.create_user("viewer@example.com", "Viewer", team.id, Role.VIEWER)

        with pytest.raises(PermissionError):
            manager.get_audit_log(viewer.id)

    def test_audit_log_max_entries(self, mock_env_vars):
        """Test audit log max entries limit."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")

        # Create many entries
        for i in range(15000):
            manager._log_action(owner.id, "test", "test", str(i), {})

        # Should keep only 10000
        assert len(manager.audit_log) == 10000

    def test_create_session(self, mock_env_vars):
        """Test create_session method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")

        token = manager.create_session(owner.id)

        assert token in manager.sessions
        assert manager.sessions[token]["user_id"] == owner.id

    def test_validate_session_success(self, mock_env_vars):
        """Test validate_session for valid session."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")
        token = manager.create_session(owner.id)

        user_id = manager.validate_session(token)

        assert user_id == owner.id

    def test_validate_session_not_found(self, mock_env_vars):
        """Test validate_session for nonexistent session."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        user_id = manager.validate_session("invalid-token")

        assert user_id is None

    def test_validate_session_expired(self, mock_env_vars):
        """Test validate_session for expired session."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")
        token = manager.create_session(owner.id)

        # Expire the session
        manager.sessions[token]["expires_at"] = datetime.utcnow() - timedelta(hours=1)

        user_id = manager.validate_session(token)

        assert user_id is None
        assert token not in manager.sessions  # Should be cleaned up

    def test_invalidate_session(self, mock_env_vars):
        """Test invalidate_session method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        team, owner = manager.create_team("Test Team", "owner@example.com", "Owner")
        token = manager.create_session(owner.id)

        manager.invalidate_session(token)

        assert token not in manager.sessions

    def test_invalidate_session_not_found(self, mock_env_vars):
        """Test invalidate_session for nonexistent session."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()
        # Should not raise
        manager.invalidate_session("invalid-token")

    def test_slugify(self, mock_env_vars):
        """Test _slugify method."""
        from src.collaboration.team import TeamManager

        manager = TeamManager()

        assert manager._slugify("Test Team") == "test-team"
        assert manager._slugify("My Amazing Team!") == "my-amazing-team"
        assert manager._slugify("Team 123") == "team-123"
        assert manager._slugify("---Test---") == "test"


class TestSSOProviders:
    """Tests for SSO provider classes."""

    def test_okta_provider_init(self, mock_env_vars):
        """Test OktaSSOProvider initialization."""
        from src.collaboration.team import OktaSSOProvider

        provider = OktaSSOProvider(
            domain="example.okta.com",
            client_id="client123",
            client_secret="secret456",
        )

        assert provider.domain == "example.okta.com"
        assert provider.client_id == "client123"
        assert provider.client_secret == "secret456"

    def test_azure_ad_provider_init(self, mock_env_vars):
        """Test AzureADSSOProvider initialization."""
        from src.collaboration.team import AzureADSSOProvider

        provider = AzureADSSOProvider(
            tenant_id="tenant123",
            client_id="client456",
            client_secret="secret789",
        )

        assert provider.tenant_id == "tenant123"
        assert provider.client_id == "client456"
        assert provider.client_secret == "secret789"

    @pytest.mark.asyncio
    async def test_okta_authenticate(self, mock_env_vars):
        """Test OktaSSOProvider authenticate method."""
        from src.collaboration.team import OktaSSOProvider

        provider = OktaSSOProvider("example.okta.com", "client", "secret")
        result = await provider.authenticate("code123")

        # Returns None as it's not implemented
        assert result is None

    @pytest.mark.asyncio
    async def test_okta_get_user_info(self, mock_env_vars):
        """Test OktaSSOProvider get_user_info method."""
        from src.collaboration.team import OktaSSOProvider

        provider = OktaSSOProvider("example.okta.com", "client", "secret")
        result = await provider.get_user_info("access_token")

        assert result is None

    @pytest.mark.asyncio
    async def test_azure_authenticate(self, mock_env_vars):
        """Test AzureADSSOProvider authenticate method."""
        from src.collaboration.team import AzureADSSOProvider

        provider = AzureADSSOProvider("tenant", "client", "secret")
        result = await provider.authenticate("code123")

        assert result is None

    @pytest.mark.asyncio
    async def test_azure_get_user_info(self, mock_env_vars):
        """Test AzureADSSOProvider get_user_info method."""
        from src.collaboration.team import AzureADSSOProvider

        provider = AzureADSSOProvider("tenant", "client", "secret")
        result = await provider.get_user_info("access_token")

        assert result is None
