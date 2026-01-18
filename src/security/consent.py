"""User consent management for enterprise compliance.

Ensures users explicitly approve what data is sent to AI services.
Required for GDPR, CCPA, and enterprise data governance.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import structlog

logger = structlog.get_logger()


class ConsentScope(str, Enum):
    """What the user is consenting to share."""
    # Code analysis
    SOURCE_CODE = "source_code"  # Read and analyze source files
    TEST_FILES = "test_files"  # Read existing test files
    CONFIG_FILES = "config_files"  # Read configuration (sanitized)

    # Runtime data
    SCREENSHOTS = "screenshots"  # Capture and analyze screenshots
    BROWSER_ACTIONS = "browser_actions"  # Execute browser automation
    API_RESPONSES = "api_responses"  # Capture API response data

    # External services
    SEND_TO_ANTHROPIC = "send_to_anthropic"  # Send data to Claude API
    SEND_TO_GITHUB = "send_to_github"  # Post PR comments
    SEND_TO_SLACK = "send_to_slack"  # Send Slack notifications

    # Storage
    STORE_LOCALLY = "store_locally"  # Save results to disk
    STORE_AUDIT_LOGS = "store_audit_logs"  # Keep audit trail


class ConsentStatus(str, Enum):
    """Status of consent."""
    GRANTED = "granted"
    DENIED = "denied"
    PENDING = "pending"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class ConsentRecord:
    """A record of user consent."""
    scope: ConsentScope
    status: ConsentStatus
    granted_at: str | None = None
    expires_at: str | None = None
    granted_by: str | None = None  # User ID or "cli", "env", "config"
    reason: str | None = None  # Why consent was given/denied

    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.status != ConsentStatus.GRANTED:
            return False

        if self.expires_at:
            expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            if datetime.now(UTC) > expiry:
                return False

        return True


@dataclass
class ConsentBundle:
    """Collection of all consents for a session."""
    session_id: str
    user_id: str | None = None
    consents: dict[str, ConsentRecord] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def grant(
        self,
        scope: ConsentScope,
        granted_by: str = "cli",
        expires_in_hours: int | None = None,
        reason: str | None = None,
    ) -> None:
        """Grant consent for a scope."""
        expires_at = None
        if expires_in_hours:
            from datetime import timedelta
            expires_at = (datetime.now(UTC) + timedelta(hours=expires_in_hours)).isoformat()

        self.consents[scope.value] = ConsentRecord(
            scope=scope,
            status=ConsentStatus.GRANTED,
            granted_at=datetime.now(UTC).isoformat(),
            expires_at=expires_at,
            granted_by=granted_by,
            reason=reason,
        )

    def deny(self, scope: ConsentScope, reason: str | None = None) -> None:
        """Explicitly deny consent for a scope."""
        self.consents[scope.value] = ConsentRecord(
            scope=scope,
            status=ConsentStatus.DENIED,
            reason=reason,
        )

    def revoke(self, scope: ConsentScope, reason: str | None = None) -> None:
        """Revoke previously granted consent."""
        if scope.value in self.consents:
            self.consents[scope.value].status = ConsentStatus.REVOKED
            self.consents[scope.value].reason = reason

    def has_consent(self, scope: ConsentScope) -> bool:
        """Check if valid consent exists for a scope."""
        record = self.consents.get(scope.value)
        return record is not None and record.is_valid()

    def get_missing_consents(self, required: list[ConsentScope]) -> list[ConsentScope]:
        """Get list of required consents that are missing."""
        return [scope for scope in required if not self.has_consent(scope)]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "consents": {k: asdict(v) for k, v in self.consents.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConsentBundle":
        """Create from dictionary."""
        bundle = cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
        )
        for scope_str, record_data in data.get("consents", {}).items():
            record_data["scope"] = ConsentScope(record_data["scope"])
            record_data["status"] = ConsentStatus(record_data["status"])
            bundle.consents[scope_str] = ConsentRecord(**record_data)
        return bundle


class ConsentManager:
    """
    Manages user consent for data sharing.

    Enterprise requirements:
    - Explicit opt-in required before sending any data
    - Granular control over what is shared
    - Audit trail of all consent decisions
    - Ability to revoke consent at any time
    - Session-based or persistent consent

    Usage:
        consent = ConsentManager()

        # Check if we can proceed
        if not consent.has_consent(ConsentScope.SEND_TO_ANTHROPIC):
            # Ask user for consent
            if user_approves:
                consent.grant(ConsentScope.SEND_TO_ANTHROPIC)
            else:
                raise PermissionError("Cannot proceed without consent")

        # Now safe to send data
        send_to_claude(data)
    """

    # Default consents for different modes
    DEFAULT_CONSENTS = {
        "minimal": [
            ConsentScope.SOURCE_CODE,
            ConsentScope.SEND_TO_ANTHROPIC,
            ConsentScope.STORE_LOCALLY,
        ],
        "standard": [
            ConsentScope.SOURCE_CODE,
            ConsentScope.TEST_FILES,
            ConsentScope.CONFIG_FILES,
            ConsentScope.SCREENSHOTS,
            ConsentScope.BROWSER_ACTIONS,
            ConsentScope.SEND_TO_ANTHROPIC,
            ConsentScope.STORE_LOCALLY,
            ConsentScope.STORE_AUDIT_LOGS,
        ],
        "full": list(ConsentScope),  # All scopes
    }

    def __init__(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        persist_path: Path | None = None,
        auto_grant_mode: str | None = None,  # "minimal", "standard", "full"
    ):
        import uuid
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.persist_path = persist_path
        self.log = logger.bind(component="consent", session_id=self.session_id)

        # Load existing consent or create new bundle
        self.bundle = self._load_or_create(auto_grant_mode)

    def _load_or_create(self, auto_grant_mode: str | None) -> ConsentBundle:
        """Load existing consent from disk or create new bundle."""
        if self.persist_path and self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                bundle = ConsentBundle.from_dict(data)
                self.log.info("Loaded existing consent", consents=len(bundle.consents))
                return bundle
            except Exception as e:
                self.log.warning("Failed to load consent file", error=str(e))

        # Create new bundle
        bundle = ConsentBundle(
            session_id=self.session_id,
            user_id=self.user_id,
        )

        # Auto-grant if mode specified
        if auto_grant_mode and auto_grant_mode in self.DEFAULT_CONSENTS:
            for scope in self.DEFAULT_CONSENTS[auto_grant_mode]:
                bundle.grant(scope, granted_by="auto", reason=f"Auto-granted via {auto_grant_mode} mode")
            self.log.info("Auto-granted consents", mode=auto_grant_mode, count=len(bundle.consents))

        return bundle

    def save(self) -> None:
        """Persist consent bundle to disk."""
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps(self.bundle.to_dict(), indent=2))
            self.log.info("Saved consent", path=str(self.persist_path))

    def grant(
        self,
        scope: ConsentScope,
        granted_by: str = "user",
        expires_in_hours: int | None = None,
        reason: str | None = None,
    ) -> None:
        """Grant consent for a scope."""
        self.bundle.grant(scope, granted_by, expires_in_hours, reason)
        self.log.info("Consent granted", scope=scope.value, by=granted_by, reason=reason)
        self.save()

    def grant_all(self, scopes: list[ConsentScope], granted_by: str = "user") -> None:
        """Grant consent for multiple scopes at once."""
        for scope in scopes:
            self.bundle.grant(scope, granted_by)
        self.log.info("Multiple consents granted", count=len(scopes), by=granted_by)
        self.save()

    def deny(self, scope: ConsentScope, reason: str | None = None) -> None:
        """Explicitly deny consent."""
        self.bundle.deny(scope, reason)
        self.log.info("Consent denied", scope=scope.value, reason=reason)
        self.save()

    def revoke(self, scope: ConsentScope, reason: str | None = None) -> None:
        """Revoke previously granted consent."""
        self.bundle.revoke(scope, reason)
        self.log.warning("Consent revoked", scope=scope.value, reason=reason)
        self.save()

    def revoke_all(self, reason: str | None = None) -> None:
        """Revoke all consents."""
        for scope_str in list(self.bundle.consents.keys()):
            self.bundle.consents[scope_str].status = ConsentStatus.REVOKED
            self.bundle.consents[scope_str].reason = reason
        self.log.warning("All consents revoked", reason=reason)
        self.save()

    def has_consent(self, scope: ConsentScope) -> bool:
        """Check if valid consent exists for a scope."""
        return self.bundle.has_consent(scope)

    def require_consent(self, scope: ConsentScope) -> None:
        """Raise exception if consent not granted."""
        if not self.has_consent(scope):
            raise PermissionError(
                f"Consent required for '{scope.value}'. "
                f"Call consent.grant(ConsentScope.{scope.name}) to proceed."
            )

    def require_all(self, scopes: list[ConsentScope]) -> None:
        """Raise exception if any required consent is missing."""
        missing = self.bundle.get_missing_consents(scopes)
        if missing:
            missing_names = [s.value for s in missing]
            raise PermissionError(
                f"Missing required consents: {missing_names}. "
                f"Grant consent to proceed."
            )

    def get_consent_summary(self) -> dict:
        """Get summary of all consents for display."""
        summary = {
            "session_id": self.session_id,
            "granted": [],
            "denied": [],
            "pending": [],
            "expired": [],
        }

        for scope in ConsentScope:
            record = self.bundle.consents.get(scope.value)
            if record is None:
                summary["pending"].append(scope.value)
            elif record.status == ConsentStatus.GRANTED:
                if record.is_valid():
                    summary["granted"].append(scope.value)
                else:
                    summary["expired"].append(scope.value)
            elif record.status in (ConsentStatus.DENIED, ConsentStatus.REVOKED):
                summary["denied"].append(scope.value)

        return summary

    def prompt_for_consent(
        self,
        required_scopes: list[ConsentScope],
        interactive: bool = True,
    ) -> bool:
        """
        Prompt user for required consents.

        In interactive mode, prompts via CLI.
        In non-interactive mode, returns False if any consent is missing.
        """
        missing = self.bundle.get_missing_consents(required_scopes)

        if not missing:
            return True

        if not interactive:
            self.log.warning("Missing consents in non-interactive mode", missing=[s.value for s in missing])
            return False

        # Interactive prompt
        print("\n" + "=" * 60)
        print("CONSENT REQUIRED")
        print("=" * 60)
        print("\nThis action requires your consent to share data.")
        print("Please review and approve the following:\n")

        for i, scope in enumerate(missing, 1):
            desc = self._get_scope_description(scope)
            print(f"  {i}. {scope.value}: {desc}")

        print("\nOptions:")
        print("  [A] Approve all")
        print("  [D] Deny all")
        print("  [S] Select individually")
        print("  [Q] Quit\n")

        try:
            choice = input("Your choice: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            choice = "Q"

        if choice == "A":
            self.grant_all(missing, granted_by="user_interactive")
            return True
        elif choice == "D":
            for scope in missing:
                self.deny(scope, reason="User denied interactively")
            return False
        elif choice == "S":
            all_granted = True
            for scope in missing:
                desc = self._get_scope_description(scope)
                try:
                    resp = input(f"  Allow {scope.value}? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    resp = "n"

                if resp == "y":
                    self.grant(scope, granted_by="user_interactive")
                else:
                    self.deny(scope, reason="User denied interactively")
                    all_granted = False
            return all_granted
        else:
            print("Consent not granted. Exiting.")
            return False

    def _get_scope_description(self, scope: ConsentScope) -> str:
        """Get human-readable description of a consent scope."""
        descriptions = {
            ConsentScope.SOURCE_CODE: "Read your source code files",
            ConsentScope.TEST_FILES: "Read existing test files",
            ConsentScope.CONFIG_FILES: "Read configuration files (secrets are redacted)",
            ConsentScope.SCREENSHOTS: "Capture browser screenshots during testing",
            ConsentScope.BROWSER_ACTIONS: "Automate browser interactions (click, type)",
            ConsentScope.API_RESPONSES: "Capture API response data for validation",
            ConsentScope.SEND_TO_ANTHROPIC: "Send sanitized code to Claude AI for analysis",
            ConsentScope.SEND_TO_GITHUB: "Post test results as GitHub PR comments",
            ConsentScope.SEND_TO_SLACK: "Send notifications to Slack",
            ConsentScope.STORE_LOCALLY: "Save test results and reports locally",
            ConsentScope.STORE_AUDIT_LOGS: "Maintain audit logs for compliance",
        }
        return descriptions.get(scope, scope.value)


# Required consents for different operations
CONSENT_REQUIREMENTS = {
    "analyze_code": [
        ConsentScope.SOURCE_CODE,
        ConsentScope.SEND_TO_ANTHROPIC,
    ],
    "plan_tests": [
        ConsentScope.SOURCE_CODE,
        ConsentScope.SEND_TO_ANTHROPIC,
    ],
    "execute_ui_test": [
        ConsentScope.BROWSER_ACTIONS,
        ConsentScope.SCREENSHOTS,
        ConsentScope.SEND_TO_ANTHROPIC,
    ],
    "execute_api_test": [
        ConsentScope.API_RESPONSES,
        ConsentScope.SEND_TO_ANTHROPIC,
    ],
    "generate_report": [
        ConsentScope.STORE_LOCALLY,
    ],
    "post_github_comment": [
        ConsentScope.SEND_TO_GITHUB,
    ],
    "send_slack_notification": [
        ConsentScope.SEND_TO_SLACK,
    ],
}


def get_consent_manager(
    session_id: str | None = None,
    auto_mode: str | None = None,
) -> ConsentManager:
    """Factory function for consent manager."""
    import os

    # Check for environment variable override
    env_mode = os.environ.get("CONSENT_MODE", auto_mode)

    return ConsentManager(
        session_id=session_id,
        auto_grant_mode=env_mode,
    )
