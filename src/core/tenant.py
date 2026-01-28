"""
Multi-Tenant Context Management

Provides TenantContext dataclass and utilities for organization/project isolation.
Every API request must have a valid TenantContext to ensure data isolation.
"""

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


@dataclass(frozen=True)
class TenantContext:
    """Immutable tenant context for request-scoped isolation.

    This context flows through all API requests and downstream service calls
    to ensure proper data isolation between organizations and projects.

    Attributes:
        org_id: Organization ID (required, primary tenant identifier)
        org_name: Organization display name (for logging/display)
        project_id: Project ID (optional, for project-scoped operations)
        project_name: Project display name
        user_id: Authenticated user ID
        user_email: User email (for audit logging)
        plan: Organization plan tier (free, pro, enterprise)
        request_id: Correlation ID for request tracing
    """

    org_id: str
    org_name: str | None = None
    project_id: str | None = None
    project_name: str | None = None
    user_id: str | None = None
    user_email: str | None = None
    plan: str = "free"
    request_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        """Validate required fields."""
        if not self.org_id:
            raise ValueError("org_id is required for TenantContext")

    # =========================================================================
    # Cognee Dataset Names
    # =========================================================================

    @property
    def cognee_dataset_prefix(self) -> str:
        """Generate Cognee dataset prefix for this tenant.

        Format: org_{org_id}_project_{project_id}
        If no project_id, returns org-level prefix.

        Returns:
            Dataset prefix string
        """
        if self.project_id:
            return f"org_{self.org_id}_project_{self.project_id}"
        return f"org_{self.org_id}"

    @property
    def codebase_dataset(self) -> str:
        """Cognee dataset name for codebase knowledge.

        Returns:
            Dataset name like 'org_abc123_project_xyz789_codebase'
        """
        return f"{self.cognee_dataset_prefix}_codebase"

    @property
    def tests_dataset(self) -> str:
        """Cognee dataset name for test knowledge.

        Returns:
            Dataset name like 'org_abc123_project_xyz789_tests'
        """
        return f"{self.cognee_dataset_prefix}_tests"

    @property
    def failures_dataset(self) -> str:
        """Cognee dataset name for failure patterns.

        Returns:
            Dataset name like 'org_abc123_project_xyz789_failures'
        """
        return f"{self.cognee_dataset_prefix}_failures"

    # =========================================================================
    # Plan Checks
    # =========================================================================

    @property
    def is_free_plan(self) -> bool:
        """Check if organization is on free plan."""
        return self.plan == "free"

    @property
    def is_pro_plan(self) -> bool:
        """Check if organization is on pro plan or higher."""
        return self.plan in ("pro", "enterprise")

    @property
    def is_enterprise_plan(self) -> bool:
        """Check if organization is on enterprise plan."""
        return self.plan == "enterprise"

    # =========================================================================
    # Logging and Display
    # =========================================================================

    def to_log_context(self) -> dict:
        """Convert to dict for structured logging.

        Returns:
            Dict with non-null values for logging
        """
        ctx = {
            "org_id": self.org_id,
            "request_id": self.request_id,
        }
        if self.project_id:
            ctx["project_id"] = self.project_id
        if self.user_id:
            ctx["user_id"] = self.user_id
        return ctx

    def to_kafka_tenant_info(self) -> dict:
        """Convert to TenantInfo dict for Kafka events.

        Returns:
            Dict compatible with events.schemas.TenantInfo
        """
        return {
            "org_id": self.org_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
        }

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def for_system(cls, org_id: str, project_id: str | None = None) -> "TenantContext":
        """Create context for system/background operations.

        Args:
            org_id: Organization ID
            project_id: Optional project ID

        Returns:
            TenantContext for system operations
        """
        return cls(
            org_id=org_id,
            project_id=project_id,
            user_id="system",
        )

    @classmethod
    def for_testing(
        cls,
        org_id: str = "test-org-123",
        project_id: str = "test-project-456",
    ) -> "TenantContext":
        """Create context for unit tests.

        Args:
            org_id: Test organization ID
            project_id: Test project ID

        Returns:
            TenantContext for testing
        """
        return cls(
            org_id=org_id,
            org_name="Test Organization",
            project_id=project_id,
            project_name="Test Project",
            user_id="test-user-789",
            user_email="test@example.com",
            plan="pro",
        )


# =============================================================================
# Context Variable for Request Scope
# =============================================================================

from contextvars import ContextVar

# Context variable to hold tenant context for current request
_tenant_context_var: ContextVar[TenantContext | None] = ContextVar(
    "tenant_context",
    default=None
)


def get_current_tenant() -> TenantContext | None:
    """Get the current request's tenant context.

    Returns:
        TenantContext if set, None otherwise
    """
    return _tenant_context_var.get()


def set_current_tenant(context: TenantContext) -> None:
    """Set the tenant context for the current request.

    Args:
        context: TenantContext to set
    """
    _tenant_context_var.set(context)


def require_tenant() -> TenantContext:
    """Get current tenant context, raising if not set.

    Returns:
        TenantContext

    Raises:
        RuntimeError: If no tenant context is set
    """
    ctx = get_current_tenant()
    if ctx is None:
        raise RuntimeError("No tenant context set for current request")
    return ctx
