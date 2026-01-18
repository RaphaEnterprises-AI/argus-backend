"""
Team Collaboration Features

Enterprise-grade collaboration for testing teams:
1. Role-Based Access Control (RBAC)
2. Team workspaces and projects
3. Test sharing and approval workflows
4. Comments and annotations on failures
5. Audit logging
6. SSO/SAML integration support
"""

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4


class Role(str, Enum):
    ADMIN = "admin"           # Full access
    MANAGER = "manager"       # Manage team, approve changes
    DEVELOPER = "developer"   # Create/edit tests, view results
    VIEWER = "viewer"         # View only
    CI_BOT = "ci_bot"         # Automated system access


class Permission(str, Enum):
    # Test permissions
    CREATE_TEST = "create_test"
    EDIT_TEST = "edit_test"
    DELETE_TEST = "delete_test"
    RUN_TEST = "run_test"
    VIEW_TEST = "view_test"

    # Results permissions
    VIEW_RESULTS = "view_results"
    EXPORT_RESULTS = "export_results"
    APPROVE_BASELINE = "approve_baseline"

    # Team permissions
    INVITE_MEMBER = "invite_member"
    REMOVE_MEMBER = "remove_member"
    MANAGE_ROLES = "manage_roles"

    # Admin permissions
    MANAGE_SETTINGS = "manage_settings"
    VIEW_AUDIT_LOG = "view_audit_log"
    MANAGE_INTEGRATIONS = "manage_integrations"


# Role to permissions mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: list(Permission),  # All permissions
    Role.MANAGER: [
        Permission.CREATE_TEST, Permission.EDIT_TEST, Permission.RUN_TEST,
        Permission.VIEW_TEST, Permission.VIEW_RESULTS, Permission.EXPORT_RESULTS,
        Permission.APPROVE_BASELINE, Permission.INVITE_MEMBER
    ],
    Role.DEVELOPER: [
        Permission.CREATE_TEST, Permission.EDIT_TEST, Permission.RUN_TEST,
        Permission.VIEW_TEST, Permission.VIEW_RESULTS
    ],
    Role.VIEWER: [
        Permission.VIEW_TEST, Permission.VIEW_RESULTS
    ],
    Role.CI_BOT: [
        Permission.RUN_TEST, Permission.VIEW_TEST, Permission.VIEW_RESULTS
    ]
}


@dataclass
class User:
    """A user in the system."""
    id: str
    email: str
    name: str
    role: Role
    team_id: str
    avatar_url: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: datetime | None = None
    sso_provider: str | None = None  # "okta", "azure_ad", "google"
    mfa_enabled: bool = False


@dataclass
class Team:
    """A team/organization."""
    id: str
    name: str
    slug: str
    owner_id: str
    plan: str = "free"  # "free", "pro", "enterprise"
    created_at: datetime = field(default_factory=datetime.utcnow)
    settings: dict = field(default_factory=dict)
    sso_config: dict | None = None


@dataclass
class Workspace:
    """A workspace/project within a team."""
    id: str
    team_id: str
    name: str
    description: str = ""
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    settings: dict = field(default_factory=dict)


@dataclass
class Comment:
    """A comment on a test result or failure."""
    id: str
    user_id: str
    target_type: str  # "test_result", "failure", "baseline"
    target_id: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None
    mentions: list[str] = field(default_factory=list)
    reactions: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ApprovalRequest:
    """Request for approval (e.g., baseline update)."""
    id: str
    type: str  # "baseline_update", "test_deletion", "config_change"
    requester_id: str
    target_id: str
    description: str
    status: str = "pending"  # "pending", "approved", "rejected"
    reviewer_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None
    comments: list[str] = field(default_factory=list)


@dataclass
class AuditLogEntry:
    """An entry in the audit log."""
    id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: dict
    ip_address: str | None = None
    user_agent: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TeamManager:
    """Manages teams and collaboration features."""

    def __init__(self):
        self.users: dict[str, User] = {}
        self.teams: dict[str, Team] = {}
        self.workspaces: dict[str, Workspace] = {}
        self.comments: dict[str, Comment] = {}
        self.approvals: dict[str, ApprovalRequest] = {}
        self.audit_log: list[AuditLogEntry] = []
        self.sessions: dict[str, dict] = {}  # token -> session

    # User Management
    def create_user(
        self,
        email: str,
        name: str,
        team_id: str,
        role: Role = Role.DEVELOPER,
        sso_provider: str | None = None
    ) -> User:
        """Create a new user."""
        user = User(
            id=str(uuid4()),
            email=email,
            name=name,
            role=role,
            team_id=team_id,
            sso_provider=sso_provider
        )
        self.users[user.id] = user
        self._log_action(user.id, "create_user", "user", user.id, {"email": email})
        return user

    def get_user(self, user_id: str) -> User | None:
        """Get a user by ID."""
        return self.users.get(user_id)

    def get_user_by_email(self, email: str) -> User | None:
        """Get a user by email."""
        for user in self.users.values():
            if user.email == email:
                return user
        return None

    def update_user_role(
        self,
        user_id: str,
        new_role: Role,
        updated_by: str
    ) -> bool:
        """Update a user's role."""
        user = self.users.get(user_id)
        if not user:
            return False

        old_role = user.role
        user.role = new_role

        self._log_action(
            updated_by, "update_role", "user", user_id,
            {"old_role": old_role.value, "new_role": new_role.value}
        )
        return True

    # Permission Checking
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        user = self.users.get(user_id)
        if not user:
            return False

        allowed_permissions = ROLE_PERMISSIONS.get(user.role, [])
        return permission in allowed_permissions

    def check_permission(self, user_id: str, permission: Permission) -> None:
        """Check permission and raise exception if denied."""
        if not self.has_permission(user_id, permission):
            raise PermissionError(
                f"User {user_id} does not have permission: {permission.value}"
            )

    # Team Management
    def create_team(
        self,
        name: str,
        owner_email: str,
        owner_name: str
    ) -> tuple[Team, User]:
        """Create a new team with owner."""
        team = Team(
            id=str(uuid4()),
            name=name,
            slug=self._slugify(name),
            owner_id=""  # Will be set after user creation
        )

        owner = self.create_user(
            email=owner_email,
            name=owner_name,
            team_id=team.id,
            role=Role.ADMIN
        )
        team.owner_id = owner.id

        self.teams[team.id] = team
        self._log_action(owner.id, "create_team", "team", team.id, {"name": name})

        return team, owner

    def get_team(self, team_id: str) -> Team | None:
        """Get a team by ID."""
        return self.teams.get(team_id)

    def get_team_members(self, team_id: str) -> list[User]:
        """Get all members of a team."""
        return [u for u in self.users.values() if u.team_id == team_id]

    def invite_member(
        self,
        team_id: str,
        email: str,
        role: Role,
        invited_by: str
    ) -> str:
        """Create an invitation for a new team member."""
        self.check_permission(invited_by, Permission.INVITE_MEMBER)

        invite_token = secrets.token_urlsafe(32)

        # Store invitation (in real implementation, this would be in a database)
        self._log_action(
            invited_by, "invite_member", "team", team_id,
            {"email": email, "role": role.value}
        )

        return invite_token

    # Workspace Management
    def create_workspace(
        self,
        team_id: str,
        name: str,
        description: str,
        created_by: str
    ) -> Workspace:
        """Create a new workspace."""
        workspace = Workspace(
            id=str(uuid4()),
            team_id=team_id,
            name=name,
            description=description,
            created_by=created_by
        )
        self.workspaces[workspace.id] = workspace

        self._log_action(
            created_by, "create_workspace", "workspace", workspace.id,
            {"name": name}
        )
        return workspace

    def get_team_workspaces(self, team_id: str) -> list[Workspace]:
        """Get all workspaces for a team."""
        return [w for w in self.workspaces.values() if w.team_id == team_id]

    # Comments and Annotations
    def add_comment(
        self,
        user_id: str,
        target_type: str,
        target_id: str,
        content: str,
        mentions: list[str] = None
    ) -> Comment:
        """Add a comment to a test result or failure."""
        comment = Comment(
            id=str(uuid4()),
            user_id=user_id,
            target_type=target_type,
            target_id=target_id,
            content=content,
            mentions=mentions or []
        )
        self.comments[comment.id] = comment

        self._log_action(
            user_id, "add_comment", target_type, target_id,
            {"comment_id": comment.id}
        )

        # In real implementation, notify mentioned users
        return comment

    def get_comments(self, target_type: str, target_id: str) -> list[Comment]:
        """Get all comments for a target."""
        return [
            c for c in self.comments.values()
            if c.target_type == target_type and c.target_id == target_id
        ]

    def add_reaction(
        self,
        comment_id: str,
        user_id: str,
        reaction: str  # emoji like "👍", "❤️", "🎉"
    ) -> bool:
        """Add a reaction to a comment."""
        comment = self.comments.get(comment_id)
        if not comment:
            return False

        if reaction not in comment.reactions:
            comment.reactions[reaction] = []

        if user_id not in comment.reactions[reaction]:
            comment.reactions[reaction].append(user_id)

        return True

    # Approval Workflows
    def request_approval(
        self,
        requester_id: str,
        approval_type: str,
        target_id: str,
        description: str
    ) -> ApprovalRequest:
        """Create an approval request."""
        request = ApprovalRequest(
            id=str(uuid4()),
            type=approval_type,
            requester_id=requester_id,
            target_id=target_id,
            description=description
        )
        self.approvals[request.id] = request

        self._log_action(
            requester_id, "request_approval", approval_type, target_id,
            {"request_id": request.id}
        )
        return request

    def approve_request(
        self,
        request_id: str,
        reviewer_id: str,
        comment: str | None = None
    ) -> bool:
        """Approve an approval request."""
        self.check_permission(reviewer_id, Permission.APPROVE_BASELINE)

        request = self.approvals.get(request_id)
        if not request or request.status != "pending":
            return False

        request.status = "approved"
        request.reviewer_id = reviewer_id
        request.resolved_at = datetime.utcnow()

        if comment:
            request.comments.append(comment)

        self._log_action(
            reviewer_id, "approve_request", request.type, request.target_id,
            {"request_id": request_id}
        )
        return True

    def reject_request(
        self,
        request_id: str,
        reviewer_id: str,
        reason: str
    ) -> bool:
        """Reject an approval request."""
        self.check_permission(reviewer_id, Permission.APPROVE_BASELINE)

        request = self.approvals.get(request_id)
        if not request or request.status != "pending":
            return False

        request.status = "rejected"
        request.reviewer_id = reviewer_id
        request.resolved_at = datetime.utcnow()
        request.comments.append(f"Rejected: {reason}")

        self._log_action(
            reviewer_id, "reject_request", request.type, request.target_id,
            {"request_id": request_id, "reason": reason}
        )
        return True

    def get_pending_approvals(self, team_id: str) -> list[ApprovalRequest]:
        """Get all pending approvals for a team."""
        return [
            a for a in self.approvals.values()
            if a.status == "pending"
        ]

    # Audit Logging
    def _log_action(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: dict,
        ip_address: str | None = None,
        user_agent: str | None = None
    ):
        """Log an action to the audit log."""
        entry = AuditLogEntry(
            id=str(uuid4()),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
        self.audit_log.append(entry)

        # Keep only last 10000 entries in memory
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]

    def get_audit_log(
        self,
        user_id: str,
        team_id: str | None = None,
        action: str | None = None,
        since: datetime | None = None,
        limit: int = 100
    ) -> list[AuditLogEntry]:
        """Get audit log entries."""
        self.check_permission(user_id, Permission.VIEW_AUDIT_LOG)

        entries = self.audit_log

        if action:
            entries = [e for e in entries if e.action == action]

        if since:
            entries = [e for e in entries if e.timestamp >= since]

        return entries[-limit:]

    # Session Management
    def create_session(self, user_id: str) -> str:
        """Create a new session for a user."""
        token = secrets.token_urlsafe(64)
        self.sessions[token] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=24)
        }
        return token

    def validate_session(self, token: str) -> str | None:
        """Validate a session token and return user ID."""
        session = self.sessions.get(token)
        if not session:
            return None

        if datetime.utcnow() > session["expires_at"]:
            del self.sessions[token]
            return None

        return session["user_id"]

    def invalidate_session(self, token: str):
        """Invalidate a session."""
        if token in self.sessions:
            del self.sessions[token]

    # Utilities
    def _slugify(self, text: str) -> str:
        """Create a URL-safe slug from text."""
        import re
        slug = text.lower()
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        return slug


class SSOProvider:
    """Base class for SSO providers."""

    async def authenticate(self, token: str) -> dict | None:
        """Authenticate with SSO provider."""
        raise NotImplementedError

    async def get_user_info(self, token: str) -> dict | None:
        """Get user info from SSO provider."""
        raise NotImplementedError


class OktaSSOProvider(SSOProvider):
    """Okta SSO integration."""

    def __init__(self, domain: str, client_id: str, client_secret: str):
        self.domain = domain
        self.client_id = client_id
        self.client_secret = client_secret

    async def authenticate(self, code: str) -> dict | None:
        """Exchange code for tokens."""
        # Implementation would use httpx to call Okta
        pass

    async def get_user_info(self, access_token: str) -> dict | None:
        """Get user info from Okta."""
        # Implementation would call Okta userinfo endpoint
        pass


class AzureADSSOProvider(SSOProvider):
    """Azure AD SSO integration."""

    def __init__(self, tenant_id: str, client_id: str, client_secret: str):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret

    async def authenticate(self, code: str) -> dict | None:
        """Exchange code for tokens."""
        pass

    async def get_user_info(self, access_token: str) -> dict | None:
        """Get user info from Azure AD."""
        pass
