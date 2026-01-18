"""Tests for the security consent module."""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest


class TestConsentScope:
    """Tests for ConsentScope enum."""

    def test_consent_scopes_exist(self, mock_env_vars):
        """Test all consent scopes are defined."""
        from src.security.consent import ConsentScope

        # Code analysis
        assert ConsentScope.SOURCE_CODE == "source_code"
        assert ConsentScope.TEST_FILES == "test_files"
        assert ConsentScope.CONFIG_FILES == "config_files"

        # Runtime data
        assert ConsentScope.SCREENSHOTS == "screenshots"
        assert ConsentScope.BROWSER_ACTIONS == "browser_actions"
        assert ConsentScope.API_RESPONSES == "api_responses"

        # External services
        assert ConsentScope.SEND_TO_ANTHROPIC == "send_to_anthropic"
        assert ConsentScope.SEND_TO_GITHUB == "send_to_github"
        assert ConsentScope.SEND_TO_SLACK == "send_to_slack"

        # Storage
        assert ConsentScope.STORE_LOCALLY == "store_locally"
        assert ConsentScope.STORE_AUDIT_LOGS == "store_audit_logs"


class TestConsentStatus:
    """Tests for ConsentStatus enum."""

    def test_consent_statuses_exist(self, mock_env_vars):
        """Test all consent statuses are defined."""
        from src.security.consent import ConsentStatus

        assert ConsentStatus.GRANTED == "granted"
        assert ConsentStatus.DENIED == "denied"
        assert ConsentStatus.PENDING == "pending"
        assert ConsentStatus.EXPIRED == "expired"
        assert ConsentStatus.REVOKED == "revoked"


class TestConsentRecord:
    """Tests for ConsentRecord dataclass."""

    def test_consent_record_creation(self, mock_env_vars):
        """Test ConsentRecord creation."""
        from src.security.consent import ConsentRecord, ConsentScope, ConsentStatus

        record = ConsentRecord(
            scope=ConsentScope.SOURCE_CODE,
            status=ConsentStatus.GRANTED,
            granted_at="2024-01-01T00:00:00+00:00",
            granted_by="user",
            reason="User approved",
        )

        assert record.scope == ConsentScope.SOURCE_CODE
        assert record.status == ConsentStatus.GRANTED
        assert record.granted_by == "user"

    def test_consent_record_is_valid_granted(self, mock_env_vars):
        """Test ConsentRecord is_valid for granted consent."""
        from src.security.consent import ConsentRecord, ConsentScope, ConsentStatus

        record = ConsentRecord(
            scope=ConsentScope.SOURCE_CODE,
            status=ConsentStatus.GRANTED,
            granted_at=datetime.now(UTC).isoformat(),
        )

        assert record.is_valid() is True

    def test_consent_record_is_valid_denied(self, mock_env_vars):
        """Test ConsentRecord is_valid for denied consent."""
        from src.security.consent import ConsentRecord, ConsentScope, ConsentStatus

        record = ConsentRecord(
            scope=ConsentScope.SOURCE_CODE,
            status=ConsentStatus.DENIED,
        )

        assert record.is_valid() is False

    def test_consent_record_is_valid_expired(self, mock_env_vars):
        """Test ConsentRecord is_valid for expired consent."""
        from src.security.consent import ConsentRecord, ConsentScope, ConsentStatus

        expired_time = (datetime.now(UTC) - timedelta(hours=1)).isoformat()

        record = ConsentRecord(
            scope=ConsentScope.SOURCE_CODE,
            status=ConsentStatus.GRANTED,
            granted_at=datetime.now(UTC).isoformat(),
            expires_at=expired_time,
        )

        assert record.is_valid() is False

    def test_consent_record_is_valid_not_expired(self, mock_env_vars):
        """Test ConsentRecord is_valid for not expired consent."""
        from src.security.consent import ConsentRecord, ConsentScope, ConsentStatus

        future_time = (datetime.now(UTC) + timedelta(hours=1)).isoformat()

        record = ConsentRecord(
            scope=ConsentScope.SOURCE_CODE,
            status=ConsentStatus.GRANTED,
            granted_at=datetime.now(UTC).isoformat(),
            expires_at=future_time,
        )

        assert record.is_valid() is True


class TestConsentBundle:
    """Tests for ConsentBundle dataclass."""

    def test_consent_bundle_creation(self, mock_env_vars):
        """Test ConsentBundle creation."""
        from src.security.consent import ConsentBundle

        bundle = ConsentBundle(session_id="session-123")

        assert bundle.session_id == "session-123"
        assert bundle.consents == {}
        assert bundle.created_at is not None

    def test_consent_bundle_grant(self, mock_env_vars):
        """Test ConsentBundle grant method."""
        from src.security.consent import ConsentBundle, ConsentScope, ConsentStatus

        bundle = ConsentBundle(session_id="session-123")
        bundle.grant(ConsentScope.SOURCE_CODE, granted_by="user", reason="Approved")

        assert ConsentScope.SOURCE_CODE.value in bundle.consents
        assert bundle.consents[ConsentScope.SOURCE_CODE.value].status == ConsentStatus.GRANTED

    def test_consent_bundle_grant_with_expiry(self, mock_env_vars):
        """Test ConsentBundle grant with expiry."""
        from src.security.consent import ConsentBundle, ConsentScope

        bundle = ConsentBundle(session_id="session-123")
        bundle.grant(ConsentScope.SOURCE_CODE, expires_in_hours=24)

        record = bundle.consents[ConsentScope.SOURCE_CODE.value]
        assert record.expires_at is not None

    def test_consent_bundle_deny(self, mock_env_vars):
        """Test ConsentBundle deny method."""
        from src.security.consent import ConsentBundle, ConsentScope, ConsentStatus

        bundle = ConsentBundle(session_id="session-123")
        bundle.deny(ConsentScope.SEND_TO_SLACK, reason="User declined")

        assert bundle.consents[ConsentScope.SEND_TO_SLACK.value].status == ConsentStatus.DENIED
        assert bundle.consents[ConsentScope.SEND_TO_SLACK.value].reason == "User declined"

    def test_consent_bundle_revoke(self, mock_env_vars):
        """Test ConsentBundle revoke method."""
        from src.security.consent import ConsentBundle, ConsentScope, ConsentStatus

        bundle = ConsentBundle(session_id="session-123")
        bundle.grant(ConsentScope.SOURCE_CODE)
        bundle.revoke(ConsentScope.SOURCE_CODE, reason="Changed mind")

        assert bundle.consents[ConsentScope.SOURCE_CODE.value].status == ConsentStatus.REVOKED
        assert bundle.consents[ConsentScope.SOURCE_CODE.value].reason == "Changed mind"

    def test_consent_bundle_has_consent(self, mock_env_vars):
        """Test ConsentBundle has_consent method."""
        from src.security.consent import ConsentBundle, ConsentScope

        bundle = ConsentBundle(session_id="session-123")

        assert bundle.has_consent(ConsentScope.SOURCE_CODE) is False

        bundle.grant(ConsentScope.SOURCE_CODE)
        assert bundle.has_consent(ConsentScope.SOURCE_CODE) is True

        bundle.revoke(ConsentScope.SOURCE_CODE)
        assert bundle.has_consent(ConsentScope.SOURCE_CODE) is False

    def test_consent_bundle_get_missing_consents(self, mock_env_vars):
        """Test ConsentBundle get_missing_consents method."""
        from src.security.consent import ConsentBundle, ConsentScope

        bundle = ConsentBundle(session_id="session-123")
        bundle.grant(ConsentScope.SOURCE_CODE)

        required = [ConsentScope.SOURCE_CODE, ConsentScope.SEND_TO_ANTHROPIC]
        missing = bundle.get_missing_consents(required)

        assert ConsentScope.SEND_TO_ANTHROPIC in missing
        assert ConsentScope.SOURCE_CODE not in missing

    def test_consent_bundle_to_dict(self, mock_env_vars):
        """Test ConsentBundle to_dict method."""
        from src.security.consent import ConsentBundle, ConsentScope

        bundle = ConsentBundle(session_id="session-123", user_id="user-456")
        bundle.grant(ConsentScope.SOURCE_CODE)

        result = bundle.to_dict()

        assert result["session_id"] == "session-123"
        assert result["user_id"] == "user-456"
        assert "source_code" in result["consents"]

    def test_consent_bundle_from_dict(self, mock_env_vars):
        """Test ConsentBundle from_dict method."""
        from src.security.consent import ConsentBundle, ConsentScope

        data = {
            "session_id": "session-123",
            "user_id": "user-456",
            "created_at": "2024-01-01T00:00:00+00:00",
            "consents": {
                "source_code": {
                    "scope": "source_code",
                    "status": "granted",
                    "granted_at": "2024-01-01T00:00:00+00:00",
                    "expires_at": None,
                    "granted_by": "user",
                    "reason": None,
                }
            },
        }

        bundle = ConsentBundle.from_dict(data)

        assert bundle.session_id == "session-123"
        assert bundle.user_id == "user-456"
        assert bundle.has_consent(ConsentScope.SOURCE_CODE)


class TestConsentManager:
    """Tests for ConsentManager class."""

    def test_consent_manager_creation(self, mock_env_vars):
        """Test ConsentManager creation."""
        from src.security.consent import ConsentManager

        manager = ConsentManager()

        assert manager.session_id is not None
        assert manager.bundle is not None

    def test_consent_manager_with_session_id(self, mock_env_vars):
        """Test ConsentManager with session ID."""
        from src.security.consent import ConsentManager

        manager = ConsentManager(session_id="session-123", user_id="user-456")

        assert manager.session_id == "session-123"
        assert manager.user_id == "user-456"

    def test_consent_manager_auto_grant_mode(self, mock_env_vars):
        """Test ConsentManager with auto grant mode."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager(auto_grant_mode="minimal")

        assert manager.has_consent(ConsentScope.SOURCE_CODE)
        assert manager.has_consent(ConsentScope.SEND_TO_ANTHROPIC)
        assert manager.has_consent(ConsentScope.STORE_LOCALLY)

    def test_consent_manager_auto_grant_standard(self, mock_env_vars):
        """Test ConsentManager with standard auto grant mode."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager(auto_grant_mode="standard")

        assert manager.has_consent(ConsentScope.SOURCE_CODE)
        assert manager.has_consent(ConsentScope.TEST_FILES)
        assert manager.has_consent(ConsentScope.SCREENSHOTS)
        assert manager.has_consent(ConsentScope.BROWSER_ACTIONS)

    def test_consent_manager_grant(self, mock_env_vars):
        """Test ConsentManager grant method."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager()
        manager.grant(ConsentScope.SOURCE_CODE, granted_by="test")

        assert manager.has_consent(ConsentScope.SOURCE_CODE)

    def test_consent_manager_grant_all(self, mock_env_vars):
        """Test ConsentManager grant_all method."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager()
        manager.grant_all([ConsentScope.SOURCE_CODE, ConsentScope.TEST_FILES])

        assert manager.has_consent(ConsentScope.SOURCE_CODE)
        assert manager.has_consent(ConsentScope.TEST_FILES)

    def test_consent_manager_deny(self, mock_env_vars):
        """Test ConsentManager deny method."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager()
        manager.deny(ConsentScope.SEND_TO_SLACK, reason="Not needed")

        assert manager.has_consent(ConsentScope.SEND_TO_SLACK) is False

    def test_consent_manager_revoke(self, mock_env_vars):
        """Test ConsentManager revoke method."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager()
        manager.grant(ConsentScope.SOURCE_CODE)
        assert manager.has_consent(ConsentScope.SOURCE_CODE)

        manager.revoke(ConsentScope.SOURCE_CODE, reason="Changed mind")
        assert manager.has_consent(ConsentScope.SOURCE_CODE) is False

    def test_consent_manager_revoke_all(self, mock_env_vars):
        """Test ConsentManager revoke_all method."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager(auto_grant_mode="minimal")
        manager.revoke_all(reason="Reset")

        assert manager.has_consent(ConsentScope.SOURCE_CODE) is False
        assert manager.has_consent(ConsentScope.SEND_TO_ANTHROPIC) is False

    def test_consent_manager_require_consent_success(self, mock_env_vars):
        """Test ConsentManager require_consent with consent."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager()
        manager.grant(ConsentScope.SOURCE_CODE)

        # Should not raise
        manager.require_consent(ConsentScope.SOURCE_CODE)

    def test_consent_manager_require_consent_failure(self, mock_env_vars):
        """Test ConsentManager require_consent without consent."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager()

        with pytest.raises(PermissionError) as excinfo:
            manager.require_consent(ConsentScope.SEND_TO_SLACK)

        assert "Consent required" in str(excinfo.value)

    def test_consent_manager_require_all_success(self, mock_env_vars):
        """Test ConsentManager require_all with all consents."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager()
        manager.grant_all([ConsentScope.SOURCE_CODE, ConsentScope.SEND_TO_ANTHROPIC])

        # Should not raise
        manager.require_all([ConsentScope.SOURCE_CODE, ConsentScope.SEND_TO_ANTHROPIC])

    def test_consent_manager_require_all_failure(self, mock_env_vars):
        """Test ConsentManager require_all with missing consents."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager()
        manager.grant(ConsentScope.SOURCE_CODE)

        with pytest.raises(PermissionError) as excinfo:
            manager.require_all([ConsentScope.SOURCE_CODE, ConsentScope.SEND_TO_ANTHROPIC])

        assert "Missing required consents" in str(excinfo.value)

    def test_consent_manager_get_consent_summary(self, mock_env_vars):
        """Test ConsentManager get_consent_summary method."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager()
        manager.grant(ConsentScope.SOURCE_CODE)
        manager.deny(ConsentScope.SEND_TO_SLACK)

        summary = manager.get_consent_summary()

        assert ConsentScope.SOURCE_CODE.value in summary["granted"]
        assert ConsentScope.SEND_TO_SLACK.value in summary["denied"]
        assert len(summary["pending"]) > 0

    def test_consent_manager_save_and_load(self, mock_env_vars):
        """Test ConsentManager save and load."""
        from src.security.consent import ConsentManager, ConsentScope

        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "consent.json"

            # Create and save
            manager1 = ConsentManager(
                session_id="session-123",
                persist_path=persist_path,
            )
            manager1.grant(ConsentScope.SOURCE_CODE)
            manager1.save()

            # Load
            manager2 = ConsentManager(
                session_id="session-123",
                persist_path=persist_path,
            )

            assert manager2.has_consent(ConsentScope.SOURCE_CODE)

    def test_consent_manager_prompt_not_interactive(self, mock_env_vars):
        """Test ConsentManager prompt_for_consent in non-interactive mode."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager()

        result = manager.prompt_for_consent(
            [ConsentScope.SOURCE_CODE],
            interactive=False,
        )

        assert result is False

    def test_consent_manager_prompt_already_granted(self, mock_env_vars):
        """Test ConsentManager prompt_for_consent with already granted."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager()
        manager.grant(ConsentScope.SOURCE_CODE)

        result = manager.prompt_for_consent(
            [ConsentScope.SOURCE_CODE],
            interactive=False,
        )

        assert result is True

    def test_consent_manager_get_scope_description(self, mock_env_vars):
        """Test ConsentManager _get_scope_description method."""
        from src.security.consent import ConsentManager, ConsentScope

        manager = ConsentManager()

        desc = manager._get_scope_description(ConsentScope.SOURCE_CODE)
        assert "source code" in desc.lower()

        desc = manager._get_scope_description(ConsentScope.SEND_TO_ANTHROPIC)
        assert "claude" in desc.lower() or "anthropic" in desc.lower()


class TestConsentRequirements:
    """Tests for CONSENT_REQUIREMENTS constant."""

    def test_consent_requirements_exist(self, mock_env_vars):
        """Test CONSENT_REQUIREMENTS contains expected operations."""
        from src.security.consent import CONSENT_REQUIREMENTS, ConsentScope

        assert "analyze_code" in CONSENT_REQUIREMENTS
        assert "plan_tests" in CONSENT_REQUIREMENTS
        assert "execute_ui_test" in CONSENT_REQUIREMENTS
        assert "execute_api_test" in CONSENT_REQUIREMENTS
        assert "generate_report" in CONSENT_REQUIREMENTS

        # Check required consents for analyze_code
        assert ConsentScope.SOURCE_CODE in CONSENT_REQUIREMENTS["analyze_code"]
        assert ConsentScope.SEND_TO_ANTHROPIC in CONSENT_REQUIREMENTS["analyze_code"]


class TestGetConsentManager:
    """Tests for get_consent_manager factory function."""

    def test_get_consent_manager_default(self, mock_env_vars):
        """Test get_consent_manager with default settings."""
        from src.security.consent import get_consent_manager

        manager = get_consent_manager()

        assert manager is not None
        assert manager.session_id is not None

    def test_get_consent_manager_with_session(self, mock_env_vars):
        """Test get_consent_manager with session ID."""
        from src.security.consent import get_consent_manager

        manager = get_consent_manager(session_id="session-123")

        assert manager.session_id == "session-123"

    def test_get_consent_manager_with_auto_mode(self, mock_env_vars):
        """Test get_consent_manager with auto mode."""
        from src.security.consent import ConsentScope, get_consent_manager

        manager = get_consent_manager(auto_mode="minimal")

        assert manager.has_consent(ConsentScope.SOURCE_CODE)

    def test_get_consent_manager_env_override(self, mock_env_vars):
        """Test get_consent_manager with environment override."""
        from src.security.consent import ConsentScope, get_consent_manager

        with patch.dict("os.environ", {"CONSENT_MODE": "standard"}):
            manager = get_consent_manager()

            assert manager.has_consent(ConsentScope.SCREENSHOTS)
            assert manager.has_consent(ConsentScope.BROWSER_ACTIONS)
