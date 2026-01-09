"""Security module for SOC2 compliance.

This module provides enterprise-grade security features:
- API Authentication (API Keys, JWT, OAuth2)
- Rate Limiting and Throttling
- Role-Based Access Control (RBAC)
- Security Headers (OWASP)
- Audit Logging
- Input Validation and Sanitization
- Request Encryption
"""

from src.api.security.middleware import (
    SecurityMiddleware,
    RateLimitMiddleware,
    AuditLogMiddleware,
    AuthenticationMiddleware,
)
from src.api.security.auth import (
    get_current_user,
    require_auth,
    require_roles,
    require_scopes,
)
from src.api.security.rbac import (
    Permission,
    Role,
    RBACManager,
)
from src.api.security.headers import SecurityHeadersMiddleware
from src.api.security.validation import InputValidator, sanitize_input
from src.api.security.audit import SecurityAuditLogger

__all__ = [
    # Middleware
    "SecurityMiddleware",
    "RateLimitMiddleware",
    "AuditLogMiddleware",
    "AuthenticationMiddleware",
    "SecurityHeadersMiddleware",
    # Auth
    "get_current_user",
    "require_auth",
    "require_roles",
    "require_scopes",
    # RBAC
    "Permission",
    "Role",
    "RBACManager",
    # Validation
    "InputValidator",
    "sanitize_input",
    # Audit
    "SecurityAuditLogger",
]
