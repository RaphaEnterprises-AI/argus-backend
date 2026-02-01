"""Tests for DLP Policy Engine."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDataClassificationEnum:
    """Tests for DataClassification enum."""

    def test_classification_values(self, mock_env_vars):
        """Test DataClassification enum values."""
        from src.security.dlp_policy_engine import DataClassification

        assert DataClassification.PUBLIC.value == "public"
        assert DataClassification.INTERNAL.value == "internal"
        assert DataClassification.CONFIDENTIAL.value == "confidential"
        assert DataClassification.RESTRICTED.value == "restricted"
        assert DataClassification.TOP_SECRET.value == "top_secret"


class TestPIITypeEnum:
    """Tests for PIIType enum."""

    def test_pii_type_values(self, mock_env_vars):
        """Test PIIType enum values."""
        from src.security.dlp_policy_engine import PIIType

        assert PIIType.EMAIL.value == "email"
        assert PIIType.PHONE.value == "phone"
        assert PIIType.SSN.value == "ssn"
        assert PIIType.CREDIT_CARD.value == "credit_card"
        assert PIIType.MEDICAL_RECORD.value == "medical_record"


class TestDLPActionEnum:
    """Tests for DLPAction enum."""

    def test_dlp_action_values(self, mock_env_vars):
        """Test DLPAction enum values."""
        from src.security.dlp_policy_engine import DLPAction

        assert DLPAction.ALLOW.value == "allow"
        assert DLPAction.LOG.value == "log"
        assert DLPAction.MASK.value == "mask"
        assert DLPAction.REDACT.value == "redact"
        assert DLPAction.BLOCK.value == "block"


class TestPIIPattern:
    """Tests for PIIPattern dataclass."""

    def test_email_pattern_match(self, mock_env_vars):
        """Test email pattern matching."""
        from src.security.dlp_policy_engine import PIIPattern, PIIType

        pattern = PIIPattern(
            pii_type=PIIType.EMAIL,
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            description="Email address",
        )

        matches = pattern.match("Contact us at support@example.com or sales@test.org")

        assert len(matches) == 2
        assert matches[0][0] == "support@example.com"
        assert matches[1][0] == "sales@test.org"

    def test_pattern_no_match(self, mock_env_vars):
        """Test pattern with no matches."""
        from src.security.dlp_policy_engine import PIIPattern, PIIType

        pattern = PIIPattern(
            pii_type=PIIType.SSN,
            pattern=r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            description="SSN",
        )

        matches = pattern.match("No SSN here, just some text")

        assert len(matches) == 0


class TestDLPRule:
    """Tests for DLPRule dataclass."""

    def test_dlp_rule_creation(self, mock_env_vars):
        """Test DLPRule creation."""
        from src.security.dlp_policy_engine import DLPRule, PIIType, DLPAction, PolicyScope

        rule = DLPRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            pii_types=[PIIType.EMAIL, PIIType.PHONE],
            scope=PolicyScope.OUTBOUND,
            action=DLPAction.MASK,
        )

        assert rule.rule_id == "test_rule"
        assert len(rule.pii_types) == 2
        assert rule.action == DLPAction.MASK

    def test_dlp_rule_to_dict(self, mock_env_vars):
        """Test DLPRule serialization."""
        from src.security.dlp_policy_engine import DLPRule, PIIType, DLPAction

        rule = DLPRule(
            rule_id="test",
            name="Test",
            description="Test rule",
            pii_types=[PIIType.SSN],
            action=DLPAction.BLOCK,
        )

        data = rule.to_dict()

        assert data["rule_id"] == "test"
        assert "ssn" in data["pii_types"]
        assert data["action"] == "block"


class TestDLPPolicyEngine:
    """Tests for DLPPolicyEngine class."""

    def test_engine_initialization(self, mock_env_vars):
        """Test engine initialization with default patterns and policies."""
        from src.security.dlp_policy_engine import DLPPolicyEngine, PIIType

        engine = DLPPolicyEngine()

        assert engine is not None
        assert PIIType.EMAIL in engine._pii_patterns
        assert PIIType.SSN in engine._pii_patterns
        assert PIIType.CREDIT_CARD in engine._pii_patterns
        assert "gdpr_standard" in engine._policies
        assert "hipaa_standard" in engine._policies
        assert "pci_dss_standard" in engine._policies

    def test_validate_ssn(self, mock_env_vars):
        """Test SSN validation."""
        from src.security.dlp_policy_engine import DLPPolicyEngine

        engine = DLPPolicyEngine()

        # Valid SSN
        assert engine._validate_ssn("123-45-6789") is True

        # Invalid - starts with 000
        assert engine._validate_ssn("000-12-3456") is False

        # Invalid - starts with 666
        assert engine._validate_ssn("666-12-3456") is False

        # Invalid - starts with 9
        assert engine._validate_ssn("900-12-3456") is False

    def test_validate_luhn(self, mock_env_vars):
        """Test Luhn credit card validation."""
        from src.security.dlp_policy_engine import DLPPolicyEngine

        engine = DLPPolicyEngine()

        # Valid Visa test number
        assert engine._validate_luhn("4532015112830366") is True

        # Invalid number
        assert engine._validate_luhn("1234567890123456") is False

    @pytest.mark.asyncio
    async def test_scan_string_with_email(self, mock_env_vars):
        """Test scanning string containing email."""
        from src.security.dlp_policy_engine import DLPPolicyEngine, PolicyScope

        engine = DLPPolicyEngine()

        result = await engine.scan(
            data="Contact: user@example.com",
            policy_ids=["default"],
            scope=PolicyScope.ALL,
        )

        assert result.total_violations >= 1
        assert "email" in result.violations_by_type

    @pytest.mark.asyncio
    async def test_scan_dict_recursive(self, mock_env_vars):
        """Test scanning nested dictionary."""
        from src.security.dlp_policy_engine import DLPPolicyEngine

        engine = DLPPolicyEngine()

        data = {
            "user": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "555-123-4567",
            },
            "notes": "Call at 555-987-6543",
        }

        result = await engine.scan(data, policy_ids=["default"])

        assert result.total_fields_scanned >= 4
        assert result.total_violations >= 2  # email + phones

    @pytest.mark.asyncio
    async def test_scan_with_masking(self, mock_env_vars):
        """Test scanning with mask action."""
        from src.security.dlp_policy_engine import (
            DLPPolicyEngine,
            DLPPolicy,
            DLPRule,
            PIIType,
            DLPAction,
        )

        engine = DLPPolicyEngine()

        # Register a policy with masking
        policy = DLPPolicy(
            policy_id="mask_test",
            name="Mask Test",
            description="Test masking",
            rules=[
                DLPRule(
                    rule_id="mask_email",
                    name="Mask Emails",
                    description="Mask all emails",
                    pii_types=[PIIType.EMAIL],
                    action=DLPAction.MASK,
                )
            ],
        )
        engine.register_policy(policy)

        result = await engine.scan(
            data="Email: user@example.com",
            policy_ids=["mask_test"],
        )

        # Data should be masked
        assert "***" in result.transformed_data
        assert "user@example.com" not in result.transformed_data

    @pytest.mark.asyncio
    async def test_scan_with_blocking(self, mock_env_vars):
        """Test scanning with block action."""
        from src.security.dlp_policy_engine import (
            DLPPolicyEngine,
            DLPPolicy,
            DLPRule,
            PIIType,
            DLPAction,
            DLPBlockedException,
        )

        engine = DLPPolicyEngine()

        # Register a policy with blocking
        policy = DLPPolicy(
            policy_id="block_test",
            name="Block Test",
            description="Test blocking",
            rules=[
                DLPRule(
                    rule_id="block_ssn",
                    name="Block SSN",
                    description="Block all SSNs",
                    pii_types=[PIIType.SSN],
                    action=DLPAction.BLOCK,
                )
            ],
        )
        engine.register_policy(policy)

        result = await engine.scan(
            data="SSN: 123-45-6789",
            policy_ids=["block_test"],
        )

        assert result.blocked is True

    @pytest.mark.asyncio
    async def test_scan_list_data(self, mock_env_vars):
        """Test scanning list data."""
        from src.security.dlp_policy_engine import DLPPolicyEngine

        engine = DLPPolicyEngine()

        data = [
            "user1@example.com",
            "user2@test.org",
            "No PII here",
        ]

        result = await engine.scan(data, policy_ids=["default"])

        assert result.total_fields_scanned == 3
        assert result.violations_by_type.get("email", 0) >= 2

    def test_mask_for_log(self, mock_env_vars):
        """Test value masking for logging."""
        from src.security.dlp_policy_engine import DLPPolicyEngine

        engine = DLPPolicyEngine()

        # Long value
        assert engine._mask_for_log("123456789") == "12*****89"

        # Short value
        assert engine._mask_for_log("abc") == "***"

    def test_get_compliance_report(self, mock_env_vars):
        """Test compliance report generation."""
        from src.security.dlp_policy_engine import DLPPolicyEngine, DLPScanResult

        engine = DLPPolicyEngine()

        scan_results = [
            DLPScanResult(
                scan_id="scan1",
                total_violations=5,
                violations_by_type={"email": 3, "phone": 2},
            ),
            DLPScanResult(
                scan_id="scan2",
                total_violations=2,
                violations_by_type={"email": 2},
                blocked=True,
            ),
        ]

        report = engine.get_compliance_report(scan_results, "GDPR")

        assert report["framework"] == "GDPR"
        assert report["summary"]["total_scans"] == 2
        assert report["summary"]["total_violations"] == 7
        assert report["summary"]["blocked_operations"] == 1
        assert report["violations_by_type"]["email"] == 5


class TestDLPViolation:
    """Tests for DLPViolation dataclass."""

    def test_violation_creation(self, mock_env_vars):
        """Test DLPViolation creation."""
        from src.security.dlp_policy_engine import DLPViolation, PIIType, DLPAction

        violation = DLPViolation(
            violation_id="v_123",
            rule_id="rule_1",
            rule_name="Test Rule",
            pii_type=PIIType.EMAIL,
            matched_text="u***@example.com",
            field_path="user.email",
            action_taken=DLPAction.MASK,
        )

        assert violation.violation_id == "v_123"
        assert violation.pii_type == PIIType.EMAIL

    def test_violation_to_dict(self, mock_env_vars):
        """Test DLPViolation serialization."""
        from src.security.dlp_policy_engine import DLPViolation, PIIType, DLPAction

        violation = DLPViolation(
            violation_id="v_456",
            rule_id="rule_2",
            rule_name="Another Rule",
            pii_type=PIIType.CREDIT_CARD,
            action_taken=DLPAction.REDACT,
        )

        data = violation.to_dict()

        assert data["violation_id"] == "v_456"
        assert data["pii_type"] == "credit_card"
        assert data["action_taken"] == "redact"


class TestDLPScanResult:
    """Tests for DLPScanResult dataclass."""

    def test_scan_result_creation(self, mock_env_vars):
        """Test DLPScanResult creation."""
        from src.security.dlp_policy_engine import DLPScanResult

        result = DLPScanResult(
            scan_id="scan_123",
            blocked=False,
            total_fields_scanned=10,
            total_violations=2,
        )

        assert result.scan_id == "scan_123"
        assert result.blocked is False
        assert result.total_violations == 2

    def test_scan_result_to_dict(self, mock_env_vars):
        """Test DLPScanResult serialization."""
        from src.security.dlp_policy_engine import DLPScanResult

        result = DLPScanResult(
            scan_id="scan_456",
            violations_by_type={"email": 3, "ssn": 1},
            actions_taken={"mask": 3, "block": 1},
        )

        data = result.to_dict()

        assert data["scan_id"] == "scan_456"
        assert data["violations_by_type"]["email"] == 3


class TestDLPMiddleware:
    """Tests for DLP middleware integration."""

    @pytest.mark.asyncio
    async def test_dlp_middleware(self, mock_env_vars):
        """Test DLP middleware function."""
        from src.security.dlp_policy_engine import dlp_middleware

        data = {"email": "test@example.com"}
        context = {"scope": "outbound", "user_role": "user"}

        transformed, result = await dlp_middleware(data, context)

        assert result is not None
        assert result.scan_id is not None


class TestDLPBlockedException:
    """Tests for DLPBlockedException."""

    def test_exception_message(self, mock_env_vars):
        """Test exception message."""
        from src.security.dlp_policy_engine import DLPBlockedException

        exc = DLPBlockedException("Operation blocked: SSN detected")

        assert "SSN detected" in str(exc)


class TestGetDLPEngineSingleton:
    """Tests for get_dlp_engine singleton."""

    def test_returns_singleton(self, mock_env_vars):
        """Test singleton pattern."""
        from src.security.dlp_policy_engine import get_dlp_engine

        # Reset singleton
        import src.security.dlp_policy_engine as module
        module._dlp_engine = None

        engine1 = get_dlp_engine()
        engine2 = get_dlp_engine()

        assert engine1 is engine2


class TestDefaultPolicies:
    """Tests for default DLP policies."""

    def test_gdpr_policy_exists(self, mock_env_vars):
        """Test GDPR policy is registered."""
        from src.security.dlp_policy_engine import DLPPolicyEngine

        engine = DLPPolicyEngine()

        policy = engine._policies.get("gdpr_standard")

        assert policy is not None
        assert "GDPR" in policy.compliance_frameworks
        assert len(policy.rules) >= 2

    def test_hipaa_policy_exists(self, mock_env_vars):
        """Test HIPAA policy is registered."""
        from src.security.dlp_policy_engine import DLPPolicyEngine

        engine = DLPPolicyEngine()

        policy = engine._policies.get("hipaa_standard")

        assert policy is not None
        assert "HIPAA" in policy.compliance_frameworks

    def test_pci_policy_exists(self, mock_env_vars):
        """Test PCI-DSS policy is registered."""
        from src.security.dlp_policy_engine import DLPPolicyEngine

        engine = DLPPolicyEngine()

        policy = engine._policies.get("pci_dss_standard")

        assert policy is not None
        assert "PCI-DSS" in policy.compliance_frameworks
