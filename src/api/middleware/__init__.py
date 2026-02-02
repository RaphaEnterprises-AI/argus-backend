"""
API Middleware

Provides middleware components for request processing:
- Tenant middleware: Extracts and validates tenant context from requests
- Prompt injection middleware: Detects and blocks prompt injection attacks (RAP-331)
"""

from src.api.middleware.prompt_injection import (
    PromptInjectionMiddleware,
    PromptProtectionDep,
    PromptProtectionResult,
    check_prompt_injection,
    validate_chat_messages,
)
from src.api.middleware.tenant import (
    OptionalTenantDep,
    TenantDep,
    TenantMiddleware,
    get_tenant_context,
    require_tenant_context,
)

__all__ = [
    # Tenant middleware
    "TenantMiddleware",
    "get_tenant_context",
    "require_tenant_context",
    "TenantDep",
    "OptionalTenantDep",
    # Prompt injection protection (RAP-331)
    "PromptInjectionMiddleware",
    "PromptProtectionDep",
    "PromptProtectionResult",
    "check_prompt_injection",
    "validate_chat_messages",
]
