"""
API Middleware

Provides middleware components for request processing:
- Tenant middleware: Extracts and validates tenant context from requests
"""

from src.api.middleware.tenant import (
    TenantMiddleware,
    get_tenant_context,
    require_tenant_context,
    TenantDep,
    OptionalTenantDep,
)

__all__ = [
    "TenantMiddleware",
    "get_tenant_context",
    "require_tenant_context",
    "TenantDep",
    "OptionalTenantDep",
]
