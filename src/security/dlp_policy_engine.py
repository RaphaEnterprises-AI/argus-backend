"""Data Loss Prevention (DLP) Policy Engine.

This module implements a formalized DLP policy engine with:
- Policy-as-code for data loss prevention rules
- PII/PHI detection patterns
- Action handlers (mask, block, alert, log)
- Compliance reporting

References:
- GDPR Article 32 (Security of Processing)
- HIPAA Security Rule
- PCI-DSS Requirement 3
- Google DLP API patterns
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Pattern
import hashlib
import json
import logging
import re

from src.config import get_settings

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels."""

    PUBLIC = "public"  # No restrictions
    INTERNAL = "internal"  # Internal use only
    CONFIDENTIAL = "confidential"  # Limited access
    RESTRICTED = "restricted"  # Highly restricted (PII, PHI, PCI)
    TOP_SECRET = "top_secret"  # Maximum protection


class PIIType(Enum):
    """Types of Personally Identifiable Information."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"  # Social Security Number
    CREDIT_CARD = "credit_card"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    MEDICAL_RECORD = "medical_record"  # PHI
    FINANCIAL_ACCOUNT = "financial_account"
    BIOMETRIC = "biometric"
    GENETIC = "genetic"
    CUSTOM = "custom"


class DLPAction(Enum):
    """Actions to take when a policy violation is detected."""

    ALLOW = "allow"  # Allow data through
    LOG = "log"  # Log and allow
    ALERT = "alert"  # Alert and allow
    MASK = "mask"  # Mask sensitive data
    REDACT = "redact"  # Remove sensitive data
    BLOCK = "block"  # Block the operation
    QUARANTINE = "quarantine"  # Quarantine for review
    ENCRYPT = "encrypt"  # Encrypt sensitive data


class PolicyScope(Enum):
    """Scope of policy application."""

    INBOUND = "inbound"  # Data coming into the system
    OUTBOUND = "outbound"  # Data leaving the system
    INTERNAL = "internal"  # Data within the system
    ALL = "all"  # All data flows


@dataclass
class PIIPattern:
    """Pattern for detecting PII."""

    pii_type: PIIType
    pattern: str  # Regex pattern
    description: str
    confidence: float = 0.9  # Detection confidence threshold
    enabled: bool = True

    # Validation function (optional)
    validator: Optional[Callable[[str], bool]] = None

    def __post_init__(self):
        """Compile the regex pattern."""
        self._compiled = re.compile(self.pattern, re.IGNORECASE)

    def match(self, text: str) -> list[tuple[str, int, int]]:
        """Find all matches in text.

        Returns:
            List of (matched_text, start_pos, end_pos)
        """
        matches = []
        for m in self._compiled.finditer(text):
            matched = m.group()
            # Apply validator if present
            if self.validator is None or self.validator(matched):
                matches.append((matched, m.start(), m.end()))
        return matches


@dataclass
class DLPRule:
    """A single DLP rule."""

    rule_id: str
    name: str
    description: str

    # What to detect
    pii_types: list[PIIType] = field(default_factory=list)
    custom_patterns: list[str] = field(default_factory=list)  # Additional regex patterns
    keywords: list[str] = field(default_factory=list)  # Keyword matching

    # When to apply
    scope: PolicyScope = PolicyScope.ALL
    data_classifications: list[DataClassification] = field(
        default_factory=lambda: [DataClassification.RESTRICTED]
    )

    # What to do
    action: DLPAction = DLPAction.BLOCK
    fallback_action: DLPAction = DLPAction.LOG  # If primary action fails

    # Rule settings
    enabled: bool = True
    priority: int = 100  # Lower = higher priority
    threshold: int = 1  # Minimum matches to trigger

    # Exemptions
    exempt_roles: list[str] = field(default_factory=list)
    exempt_contexts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "pii_types": [p.value for p in self.pii_types],
            "custom_patterns": self.custom_patterns,
            "keywords": self.keywords,
            "scope": self.scope.value,
            "data_classifications": [d.value for d in self.data_classifications],
            "action": self.action.value,
            "fallback_action": self.fallback_action.value,
            "enabled": self.enabled,
            "priority": self.priority,
            "threshold": self.threshold,
            "exempt_roles": self.exempt_roles,
            "exempt_contexts": self.exempt_contexts,
        }


@dataclass
class DLPPolicy:
    """A DLP policy containing multiple rules."""

    policy_id: str
    name: str
    description: str
    version: str = "1.0.0"

    rules: list[DLPRule] = field(default_factory=list)

    # Policy metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    updated_at: Optional[datetime] = None

    # Compliance frameworks
    compliance_frameworks: list[str] = field(default_factory=list)  # GDPR, HIPAA, PCI-DSS

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "rules": [r.to_dict() for r in self.rules],
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "compliance_frameworks": self.compliance_frameworks,
        }


@dataclass
class DLPViolation:
    """A detected DLP violation."""

    violation_id: str
    rule_id: str
    rule_name: str

    # Detection details
    pii_type: Optional[PIIType] = None
    matched_pattern: str = ""
    matched_text: str = ""  # Masked version
    original_hash: str = ""  # Hash of original for audit

    # Location
    field_path: str = ""  # JSON path or field name
    position: tuple[int, int] = (0, 0)  # Start, end position

    # Context
    data_classification: DataClassification = DataClassification.RESTRICTED
    scope: PolicyScope = PolicyScope.ALL

    # Action taken
    action_taken: DLPAction = DLPAction.LOG
    action_details: str = ""

    # Metadata
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "violation_id": self.violation_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "pii_type": self.pii_type.value if self.pii_type else None,
            "matched_pattern": self.matched_pattern,
            "matched_text": self.matched_text,
            "original_hash": self.original_hash,
            "field_path": self.field_path,
            "position": self.position,
            "data_classification": self.data_classification.value,
            "scope": self.scope.value,
            "action_taken": self.action_taken.value,
            "action_details": self.action_details,
            "detected_at": self.detected_at.isoformat(),
            "confidence": self.confidence,
        }


@dataclass
class DLPScanResult:
    """Result of a DLP scan."""

    scan_id: str
    scanned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Results
    violations: list[DLPViolation] = field(default_factory=list)
    blocked: bool = False
    transformed_data: Optional[Any] = None  # Data after masking/redaction

    # Summary
    total_fields_scanned: int = 0
    total_violations: int = 0
    violations_by_type: dict[str, int] = field(default_factory=dict)
    actions_taken: dict[str, int] = field(default_factory=dict)

    # Performance
    scan_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "scan_id": self.scan_id,
            "scanned_at": self.scanned_at.isoformat(),
            "violations": [v.to_dict() for v in self.violations],
            "blocked": self.blocked,
            "total_fields_scanned": self.total_fields_scanned,
            "total_violations": self.total_violations,
            "violations_by_type": self.violations_by_type,
            "actions_taken": self.actions_taken,
            "scan_duration_ms": self.scan_duration_ms,
        }


class DLPPolicyEngine:
    """Data Loss Prevention Policy Engine.

    Features:
    - Policy-as-code management
    - PII/PHI detection with configurable patterns
    - Multiple action handlers (mask, redact, block, etc.)
    - Compliance reporting
    - Audit logging
    """

    def __init__(self):
        self.settings = get_settings()

        # PII patterns
        self._pii_patterns: dict[PIIType, PIIPattern] = {}

        # Policies
        self._policies: dict[str, DLPPolicy] = {}

        # Action handlers
        self._action_handlers: dict[DLPAction, Callable] = {}

        # Violation handlers (callbacks)
        self._violation_handlers: list[Callable] = []

        # Initialize defaults
        self._register_default_patterns()
        self._register_default_handlers()
        self._register_default_policies()

    def _register_default_patterns(self):
        """Register default PII detection patterns."""

        # Email
        self._pii_patterns[PIIType.EMAIL] = PIIPattern(
            pii_type=PIIType.EMAIL,
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            description="Email address",
            confidence=0.95,
        )

        # Phone (US format)
        self._pii_patterns[PIIType.PHONE] = PIIPattern(
            pii_type=PIIType.PHONE,
            pattern=r'\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            description="Phone number (US format)",
            confidence=0.85,
        )

        # SSN
        self._pii_patterns[PIIType.SSN] = PIIPattern(
            pii_type=PIIType.SSN,
            pattern=r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            description="Social Security Number",
            confidence=0.9,
            validator=self._validate_ssn,
        )

        # Credit Card (Luhn validation)
        self._pii_patterns[PIIType.CREDIT_CARD] = PIIPattern(
            pii_type=PIIType.CREDIT_CARD,
            pattern=r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            description="Credit card number",
            confidence=0.95,
            validator=self._validate_luhn,
        )

        # IP Address
        self._pii_patterns[PIIType.IP_ADDRESS] = PIIPattern(
            pii_type=PIIType.IP_ADDRESS,
            pattern=r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            description="IP address",
            confidence=0.99,
        )

        # Date of Birth (various formats)
        self._pii_patterns[PIIType.DATE_OF_BIRTH] = PIIPattern(
            pii_type=PIIType.DATE_OF_BIRTH,
            pattern=r'\b(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12][0-9]|3[01])[/\-.](?:19|20)\d{2}\b',
            description="Date of birth",
            confidence=0.7,  # Lower confidence due to false positives
        )

        # Passport (US format)
        self._pii_patterns[PIIType.PASSPORT] = PIIPattern(
            pii_type=PIIType.PASSPORT,
            pattern=r'\b[A-Z]{1,2}[0-9]{6,9}\b',
            description="Passport number",
            confidence=0.6,
        )

        # Medical Record Number (generic)
        self._pii_patterns[PIIType.MEDICAL_RECORD] = PIIPattern(
            pii_type=PIIType.MEDICAL_RECORD,
            pattern=r'\b(?:MRN|MR#|Medical Record)[\s:]*[A-Z0-9]{6,12}\b',
            description="Medical record number",
            confidence=0.85,
        )

    def _register_default_handlers(self):
        """Register default action handlers."""

        self._action_handlers[DLPAction.ALLOW] = self._handle_allow
        self._action_handlers[DLPAction.LOG] = self._handle_log
        self._action_handlers[DLPAction.ALERT] = self._handle_alert
        self._action_handlers[DLPAction.MASK] = self._handle_mask
        self._action_handlers[DLPAction.REDACT] = self._handle_redact
        self._action_handlers[DLPAction.BLOCK] = self._handle_block
        self._action_handlers[DLPAction.QUARANTINE] = self._handle_quarantine
        self._action_handlers[DLPAction.ENCRYPT] = self._handle_encrypt

    def _register_default_policies(self):
        """Register default DLP policies."""

        # GDPR policy
        gdpr_policy = DLPPolicy(
            policy_id="gdpr_standard",
            name="GDPR Standard Policy",
            description="Standard GDPR compliance policy for EU data subjects",
            compliance_frameworks=["GDPR"],
            rules=[
                DLPRule(
                    rule_id="gdpr_pii_outbound",
                    name="GDPR PII Outbound Protection",
                    description="Block PII in outbound data",
                    pii_types=[
                        PIIType.EMAIL, PIIType.PHONE, PIIType.SSN,
                        PIIType.DATE_OF_BIRTH, PIIType.ADDRESS,
                    ],
                    scope=PolicyScope.OUTBOUND,
                    action=DLPAction.MASK,
                    priority=10,
                ),
                DLPRule(
                    rule_id="gdpr_sensitive_data",
                    name="GDPR Sensitive Data",
                    description="Block sensitive categories (Article 9)",
                    pii_types=[
                        PIIType.MEDICAL_RECORD, PIIType.BIOMETRIC, PIIType.GENETIC,
                    ],
                    scope=PolicyScope.ALL,
                    action=DLPAction.BLOCK,
                    priority=5,
                ),
            ],
        )
        self._policies["gdpr_standard"] = gdpr_policy

        # HIPAA policy
        hipaa_policy = DLPPolicy(
            policy_id="hipaa_standard",
            name="HIPAA Standard Policy",
            description="Standard HIPAA compliance policy for PHI",
            compliance_frameworks=["HIPAA"],
            rules=[
                DLPRule(
                    rule_id="hipaa_phi_protection",
                    name="HIPAA PHI Protection",
                    description="Protect Protected Health Information",
                    pii_types=[
                        PIIType.MEDICAL_RECORD, PIIType.SSN, PIIType.DATE_OF_BIRTH,
                    ],
                    scope=PolicyScope.ALL,
                    action=DLPAction.BLOCK,
                    priority=1,
                ),
            ],
        )
        self._policies["hipaa_standard"] = hipaa_policy

        # PCI-DSS policy
        pci_policy = DLPPolicy(
            policy_id="pci_dss_standard",
            name="PCI-DSS Standard Policy",
            description="PCI-DSS compliance for cardholder data",
            compliance_frameworks=["PCI-DSS"],
            rules=[
                DLPRule(
                    rule_id="pci_pan_protection",
                    name="PCI PAN Protection",
                    description="Protect Primary Account Numbers",
                    pii_types=[PIIType.CREDIT_CARD],
                    scope=PolicyScope.ALL,
                    action=DLPAction.MASK,
                    fallback_action=DLPAction.BLOCK,
                    priority=1,
                ),
            ],
        )
        self._policies["pci_dss_standard"] = pci_policy

        # Default catch-all policy
        default_policy = DLPPolicy(
            policy_id="default",
            name="Default DLP Policy",
            description="Default policy for general data protection",
            rules=[
                DLPRule(
                    rule_id="default_pii_logging",
                    name="Default PII Logging",
                    description="Log all PII detections",
                    pii_types=list(PIIType),
                    scope=PolicyScope.ALL,
                    action=DLPAction.LOG,
                    priority=100,
                ),
            ],
        )
        self._policies["default"] = default_policy

    # Validators

    def _validate_ssn(self, value: str) -> bool:
        """Validate SSN format and common invalid patterns."""
        # Remove separators
        ssn = re.sub(r'[-\s]', '', value)

        # Check length
        if len(ssn) != 9:
            return False

        # Check for invalid patterns
        invalid_patterns = [
            '000', '666', '9',  # Invalid area numbers
        ]
        if ssn[:3] in invalid_patterns or ssn[:3].startswith('9'):
            return False

        # Check for all same digits
        if len(set(ssn)) == 1:
            return False

        return True

    def _validate_luhn(self, value: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        # Remove spaces and dashes
        number = re.sub(r'[-\s]', '', value)

        if not number.isdigit():
            return False

        # Luhn algorithm
        total = 0
        for i, digit in enumerate(reversed(number)):
            d = int(digit)
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d

        return total % 10 == 0

    # Action handlers

    async def _handle_allow(
        self, data: str, violation: DLPViolation, context: dict
    ) -> tuple[str, str]:
        """Allow data through unchanged."""
        return data, "Allowed"

    async def _handle_log(
        self, data: str, violation: DLPViolation, context: dict
    ) -> tuple[str, str]:
        """Log violation and allow."""
        logger.info(
            f"DLP violation logged: {violation.rule_name} - "
            f"{violation.pii_type.value if violation.pii_type else 'custom'} "
            f"in {violation.field_path}"
        )
        return data, "Logged"

    async def _handle_alert(
        self, data: str, violation: DLPViolation, context: dict
    ) -> tuple[str, str]:
        """Alert on violation and allow."""
        # In production, this would send to alerting system
        logger.warning(
            f"DLP ALERT: {violation.rule_name} - "
            f"{violation.pii_type.value if violation.pii_type else 'custom'} detected"
        )

        # Notify violation handlers
        for handler in self._violation_handlers:
            try:
                await handler(violation, context)
            except Exception as e:
                logger.error(f"Violation handler error: {e}")

        return data, "Alerted"

    async def _handle_mask(
        self, data: str, violation: DLPViolation, context: dict
    ) -> tuple[str, str]:
        """Mask sensitive data."""
        start, end = violation.position
        matched = data[start:end]

        # Mask based on type
        if violation.pii_type == PIIType.EMAIL:
            # Show first char and domain
            parts = matched.split('@')
            masked = f"{parts[0][0]}***@{parts[1]}" if len(parts) == 2 else "***@***"
        elif violation.pii_type == PIIType.CREDIT_CARD:
            # Show last 4 digits
            masked = f"****-****-****-{matched[-4:]}"
        elif violation.pii_type == PIIType.SSN:
            # Show last 4 digits
            masked = f"***-**-{matched[-4:]}"
        elif violation.pii_type == PIIType.PHONE:
            # Show last 4 digits
            masked = f"(***) ***-{matched[-4:]}"
        else:
            # Generic masking
            if len(matched) > 4:
                masked = f"{matched[:2]}{'*' * (len(matched) - 4)}{matched[-2:]}"
            else:
                masked = '*' * len(matched)

        result = data[:start] + masked + data[end:]
        return result, f"Masked: {matched} → {masked}"

    async def _handle_redact(
        self, data: str, violation: DLPViolation, context: dict
    ) -> tuple[str, str]:
        """Completely redact sensitive data."""
        start, end = violation.position
        result = data[:start] + "[REDACTED]" + data[end:]
        return result, "Redacted"

    async def _handle_block(
        self, data: str, violation: DLPViolation, context: dict
    ) -> tuple[str, str]:
        """Block the operation."""
        raise DLPBlockedException(
            f"Operation blocked by DLP: {violation.rule_name} - "
            f"{violation.pii_type.value if violation.pii_type else 'policy violation'}"
        )

    async def _handle_quarantine(
        self, data: str, violation: DLPViolation, context: dict
    ) -> tuple[str, str]:
        """Quarantine data for review."""
        # In production, this would store to quarantine queue
        logger.warning(f"DLP Quarantine: {violation.violation_id}")
        return "", "Quarantined for review"

    async def _handle_encrypt(
        self, data: str, violation: DLPViolation, context: dict
    ) -> tuple[str, str]:
        """Encrypt sensitive data."""
        # In production, use proper encryption
        start, end = violation.position
        sensitive = data[start:end]
        encrypted = f"[ENCRYPTED:{hashlib.sha256(sensitive.encode()).hexdigest()[:16]}]"
        result = data[:start] + encrypted + data[end:]
        return result, "Encrypted"

    # Public API

    def register_pattern(self, pattern: PIIPattern) -> None:
        """Register a custom PII pattern."""
        self._pii_patterns[pattern.pii_type] = pattern
        logger.info(f"Registered PII pattern: {pattern.pii_type.value}")

    def register_policy(self, policy: DLPPolicy) -> None:
        """Register a DLP policy."""
        self._policies[policy.policy_id] = policy
        logger.info(f"Registered DLP policy: {policy.policy_id}")

    def add_violation_handler(self, handler: Callable) -> None:
        """Add a violation notification handler."""
        self._violation_handlers.append(handler)

    async def scan(
        self,
        data: Any,
        policy_ids: Optional[list[str]] = None,
        scope: PolicyScope = PolicyScope.ALL,
        context: Optional[dict] = None,
    ) -> DLPScanResult:
        """Scan data for policy violations.

        Args:
            data: Data to scan (string, dict, or list)
            policy_ids: Specific policies to apply (None = all)
            scope: Scope of the scan
            context: Additional context (user, role, etc.)

        Returns:
            DLPScanResult with violations and transformed data
        """
        import time
        import uuid

        start_time = time.time()
        scan_id = f"scan_{uuid.uuid4().hex[:12]}"
        context = context or {}

        result = DLPScanResult(scan_id=scan_id)

        # Get applicable policies
        policies = [
            self._policies[pid]
            for pid in (policy_ids or self._policies.keys())
            if pid in self._policies
        ]

        # Sort rules by priority
        all_rules = []
        for policy in policies:
            for rule in policy.rules:
                if rule.enabled and (rule.scope == PolicyScope.ALL or rule.scope == scope):
                    # Check exemptions
                    user_role = context.get("user_role", "")
                    if user_role in rule.exempt_roles:
                        continue
                    scan_context = context.get("context", "")
                    if scan_context in rule.exempt_contexts:
                        continue
                    all_rules.append(rule)

        all_rules.sort(key=lambda r: r.priority)

        # Scan data
        transformed_data = data
        try:
            transformed_data, violations, fields_scanned = await self._scan_value(
                data, "", all_rules, context
            )
            result.violations = violations
            result.total_fields_scanned = fields_scanned
            result.total_violations = len(violations)
            result.transformed_data = transformed_data

            # Calculate summaries
            for v in violations:
                pii_type = v.pii_type.value if v.pii_type else "custom"
                result.violations_by_type[pii_type] = result.violations_by_type.get(pii_type, 0) + 1
                action = v.action_taken.value
                result.actions_taken[action] = result.actions_taken.get(action, 0) + 1

        except DLPBlockedException:
            result.blocked = True
            result.transformed_data = None

        result.scan_duration_ms = (time.time() - start_time) * 1000

        return result

    async def _scan_value(
        self,
        value: Any,
        path: str,
        rules: list[DLPRule],
        context: dict,
    ) -> tuple[Any, list[DLPViolation], int]:
        """Recursively scan a value."""
        violations = []
        fields_scanned = 0

        if isinstance(value, str):
            fields_scanned = 1
            transformed, new_violations = await self._scan_string(value, path, rules, context)
            violations.extend(new_violations)
            return transformed, violations, fields_scanned

        elif isinstance(value, dict):
            transformed = {}
            for key, val in value.items():
                new_path = f"{path}.{key}" if path else key
                t_val, v, f = await self._scan_value(val, new_path, rules, context)
                transformed[key] = t_val
                violations.extend(v)
                fields_scanned += f
            return transformed, violations, fields_scanned

        elif isinstance(value, list):
            transformed = []
            for i, item in enumerate(value):
                new_path = f"{path}[{i}]"
                t_val, v, f = await self._scan_value(item, new_path, rules, context)
                transformed.append(t_val)
                violations.extend(v)
                fields_scanned += f
            return transformed, violations, fields_scanned

        else:
            return value, violations, fields_scanned

    async def _scan_string(
        self,
        value: str,
        path: str,
        rules: list[DLPRule],
        context: dict,
    ) -> tuple[str, list[DLPViolation]]:
        """Scan a string value for violations."""
        import uuid

        violations = []
        transformed = value

        for rule in rules:
            matches = []

            # Check PII patterns
            for pii_type in rule.pii_types:
                pattern = self._pii_patterns.get(pii_type)
                if pattern and pattern.enabled:
                    for matched, start, end in pattern.match(transformed):
                        matches.append((pii_type, pattern.pattern, matched, start, end, pattern.confidence))

            # Check custom patterns
            for custom_pattern in rule.custom_patterns:
                try:
                    for m in re.finditer(custom_pattern, transformed):
                        matches.append((PIIType.CUSTOM, custom_pattern, m.group(), m.start(), m.end(), 0.8))
                except re.error:
                    logger.warning(f"Invalid custom pattern: {custom_pattern}")

            # Check keywords
            for keyword in rule.keywords:
                for m in re.finditer(re.escape(keyword), transformed, re.IGNORECASE):
                    matches.append((PIIType.CUSTOM, keyword, m.group(), m.start(), m.end(), 0.7))

            # Process matches if threshold met
            if len(matches) >= rule.threshold:
                # Sort by position (reverse) to process from end
                matches.sort(key=lambda m: m[3], reverse=True)

                for pii_type, pattern, matched, start, end, confidence in matches:
                    # Create violation
                    violation = DLPViolation(
                        violation_id=f"v_{uuid.uuid4().hex[:12]}",
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        pii_type=pii_type,
                        matched_pattern=pattern,
                        matched_text=self._mask_for_log(matched),
                        original_hash=hashlib.sha256(matched.encode()).hexdigest(),
                        field_path=path,
                        position=(start, end),
                        confidence=confidence,
                    )

                    # Execute action
                    handler = self._action_handlers.get(rule.action)
                    if handler:
                        try:
                            # Update position for current transformed string
                            violation.position = (start, end)
                            transformed, action_details = await handler(
                                transformed, violation, context
                            )
                            violation.action_taken = rule.action
                            violation.action_details = action_details
                        except DLPBlockedException:
                            violation.action_taken = DLPAction.BLOCK
                            violation.action_details = "Blocked"
                            violations.append(violation)
                            raise

                    violations.append(violation)

        return transformed, violations

    def _mask_for_log(self, value: str) -> str:
        """Mask value for safe logging."""
        if len(value) <= 4:
            return '*' * len(value)
        return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"

    def get_compliance_report(
        self,
        scan_results: list[DLPScanResult],
        framework: str,
    ) -> dict:
        """Generate compliance report for a framework.

        Args:
            scan_results: List of scan results to include
            framework: Compliance framework (GDPR, HIPAA, PCI-DSS)

        Returns:
            Compliance report dictionary
        """
        total_scans = len(scan_results)
        total_violations = sum(r.total_violations for r in scan_results)
        blocked_operations = sum(1 for r in scan_results if r.blocked)

        # Group violations by type
        violations_by_type: dict[str, int] = {}
        for result in scan_results:
            for vtype, count in result.violations_by_type.items():
                violations_by_type[vtype] = violations_by_type.get(vtype, 0) + count

        # Get framework-specific policy
        framework_policies = [
            p for p in self._policies.values()
            if framework in p.compliance_frameworks
        ]

        return {
            "framework": framework,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_scans": total_scans,
                "total_violations": total_violations,
                "blocked_operations": blocked_operations,
                "compliance_rate": 1 - (total_violations / max(total_scans, 1)),
            },
            "violations_by_type": violations_by_type,
            "policies_applied": [p.policy_id for p in framework_policies],
            "recommendations": self._get_recommendations(framework, violations_by_type),
        }

    def _get_recommendations(
        self,
        framework: str,
        violations_by_type: dict[str, int],
    ) -> list[str]:
        """Generate recommendations based on violations."""
        recommendations = []

        if violations_by_type.get("email", 0) > 10:
            recommendations.append("Consider implementing email anonymization for non-essential use cases")

        if violations_by_type.get("credit_card", 0) > 0:
            recommendations.append("Review PCI-DSS compliance: credit card data should not be stored without proper tokenization")

        if violations_by_type.get("medical_record", 0) > 0:
            recommendations.append("HIPAA Alert: PHI detected - ensure proper Business Associate Agreements are in place")

        if framework == "GDPR" and violations_by_type.get("ip_address", 0) > 0:
            recommendations.append("GDPR: IP addresses are considered personal data in EU - implement anonymization")

        return recommendations


class DLPBlockedException(Exception):
    """Exception raised when DLP blocks an operation."""
    pass


# Singleton instance
_dlp_engine: Optional[DLPPolicyEngine] = None


def get_dlp_engine() -> DLPPolicyEngine:
    """Get the singleton DLP engine instance."""
    global _dlp_engine
    if _dlp_engine is None:
        _dlp_engine = DLPPolicyEngine()
    return _dlp_engine


# Middleware integration
async def dlp_middleware(data: Any, context: dict) -> tuple[Any, DLPScanResult]:
    """DLP middleware for API endpoints.

    Args:
        data: Data to scan
        context: Request context (user, role, scope, etc.)

    Returns:
        Tuple of (transformed_data, scan_result)
    """
    engine = get_dlp_engine()
    scope = PolicyScope(context.get("scope", "all"))
    result = await engine.scan(data, scope=scope, context=context)
    return result.transformed_data, result
