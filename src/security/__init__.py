"""Security module for enterprise-grade E2E testing.

This module provides:
- Code sanitization (remove secrets before sending to AI)
- Audit logging (track all AI interactions)
- Consent management (user approves what's sent)
- Data classification (identify sensitive files)

Enterprise features:
- GDPR/CCPA compliant consent management
- SOC2/ISO27001 audit logging
- PII detection and redaction
- Secret detection (API keys, passwords, tokens)
- Multi-level data classification (public, internal, confidential, restricted)
"""

from .sanitizer import (
    CodeSanitizer,
    SanitizationResult,
    SecretType,
    RedactedSecret,
    create_sanitizer,
    FORBIDDEN_FILES,
    FORBIDDEN_DIRS,
)
from .audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    get_audit_logger,
    hash_content,
)
from .consent import (
    ConsentManager,
    ConsentScope,
    ConsentStatus,
    ConsentBundle,
    ConsentRecord,
    CONSENT_REQUIREMENTS,
    get_consent_manager,
)
from .classifier import (
    DataClassifier,
    SensitivityLevel,
    DataCategory,
    Classification,
    create_classifier,
)
from .secure_reader import (
    SecureCodeReader,
    SecureReadResult,
    create_secure_reader,
)

__all__ = [
    # Sanitizer
    "CodeSanitizer",
    "SanitizationResult",
    "SecretType",
    "RedactedSecret",
    "create_sanitizer",
    "FORBIDDEN_FILES",
    "FORBIDDEN_DIRS",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "get_audit_logger",
    "hash_content",
    # Consent
    "ConsentManager",
    "ConsentScope",
    "ConsentStatus",
    "ConsentBundle",
    "ConsentRecord",
    "CONSENT_REQUIREMENTS",
    "get_consent_manager",
    # Classifier
    "DataClassifier",
    "SensitivityLevel",
    "DataCategory",
    "Classification",
    "create_classifier",
    # Secure Reader
    "SecureCodeReader",
    "SecureReadResult",
    "create_secure_reader",
]
