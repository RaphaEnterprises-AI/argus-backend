"""
API Middleware

Provides middleware components for request processing:
- Tenant middleware: Extracts and validates tenant context from requests
"""

from src.api.middleware.tenant import (
    OptionalTenantDep,
    TenantDep,
    TenantMiddleware,
    get_tenant_context,
    require_tenant_context,
)

__all__ = [
    "TenantMiddleware",
    "get_tenant_context",
    "require_tenant_context",
    "TenantDep",
    "OptionalTenantDep",
]
