"""Secure code reader that integrates all security components.

This module provides a safe way to read code for AI analysis by:
1. Classifying files by sensitivity
2. Sanitizing content to remove secrets
3. Checking consent before processing
4. Logging all access for audit
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

from .sanitizer import CodeSanitizer, SanitizationResult
from .classifier import DataClassifier, SensitivityLevel, Classification
from .consent import ConsentManager, ConsentScope, CONSENT_REQUIREMENTS
from .audit import AuditLogger, get_audit_logger, hash_content

logger = structlog.get_logger()


@dataclass
class SecureReadResult:
    """Result of securely reading a file or codebase."""
    path: str
    content: str  # Sanitized content safe to send to AI
    classification: Classification
    was_sanitized: bool
    secrets_redacted: int
    skipped: bool
    skip_reason: Optional[str] = None


class SecureCodeReader:
    """
    Reads code securely for AI analysis.

    Enterprise security flow:
    1. Check user consent for code reading
    2. Classify file by sensitivity
    3. Skip RESTRICTED files entirely
    4. Sanitize CONFIDENTIAL files (redact secrets)
    5. Log all access for audit
    6. Return safe content for AI

    Usage:
        reader = SecureCodeReader(user_id="user-123")

        # Read single file
        result = reader.read_file("/path/to/file.py")
        if not result.skipped:
            send_to_ai(result.content)  # Safe to send

        # Read entire codebase
        results = reader.read_codebase("/path/to/app")
        safe_content = reader.get_context_for_ai(results)
    """

    def __init__(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        consent_manager: Optional[ConsentManager] = None,
        sanitizer: Optional[CodeSanitizer] = None,
        classifier: Optional[DataClassifier] = None,
        audit_logger: Optional[AuditLogger] = None,
        require_consent: bool = True,
    ):
        self.user_id = user_id
        self.session_id = session_id

        # Initialize components
        self.consent = consent_manager or ConsentManager(session_id=session_id)
        self.sanitizer = sanitizer or CodeSanitizer()
        self.classifier = classifier or DataClassifier()
        self.audit = audit_logger or get_audit_logger()

        self.require_consent = require_consent
        self.log = logger.bind(component="secure_reader", user_id=user_id)

    def check_consent(self, operation: str = "analyze_code") -> None:
        """Check if required consents are granted. Raises PermissionError if not."""
        if not self.require_consent:
            return

        required = CONSENT_REQUIREMENTS.get(operation, [ConsentScope.SOURCE_CODE])
        self.consent.require_all(required)

    def read_file(self, file_path: str | Path) -> SecureReadResult:
        """
        Securely read a single file.

        Returns sanitized content safe to send to AI.
        """
        path = Path(file_path)
        self.log.debug("Reading file", path=str(path))

        # 1. Classify the file
        classification = self.classifier.classify_file(path)

        # 2. Check if we should skip
        if classification.sensitivity == SensitivityLevel.RESTRICTED:
            self.log.warning(
                "Skipping restricted file",
                path=str(path),
                reasons=classification.reasons,
            )
            self.audit.log_file_read(
                user_id=self.user_id,
                file_path=str(path),
                classification="restricted",
                was_sanitized=False,
            )
            return SecureReadResult(
                path=str(path),
                content="",
                classification=classification,
                was_sanitized=False,
                secrets_redacted=0,
                skipped=True,
                skip_reason=f"Restricted file: {', '.join(classification.reasons)}",
            )

        # 3. Sanitize the content
        sanitized = self.sanitizer.sanitize_file(path)

        if sanitized.should_skip:
            self.log.info("File skipped by sanitizer", path=str(path), reason=sanitized.skip_reason)
            return SecureReadResult(
                path=str(path),
                content="",
                classification=classification,
                was_sanitized=False,
                secrets_redacted=0,
                skipped=True,
                skip_reason=sanitized.skip_reason,
            )

        # 4. Log for audit
        file_hash = hash_content(sanitized.sanitized_content) if sanitized.sanitized_content else None
        self.audit.log_file_read(
            user_id=self.user_id,
            file_path=str(path),
            file_hash=file_hash,
            classification=classification.sensitivity.value,
            was_sanitized=sanitized.was_modified,
            secrets_redacted=len(sanitized.secrets_found),
        )

        # 5. Log any secrets that were detected
        for secret in sanitized.secrets_found:
            self.audit.log_secret_detected(
                user_id=self.user_id,
                file_path=str(path),
                secret_type=secret.type.value,
                line_number=secret.line_number,
            )

        return SecureReadResult(
            path=str(path),
            content=sanitized.sanitized_content,
            classification=classification,
            was_sanitized=sanitized.was_modified,
            secrets_redacted=len(sanitized.secrets_found),
            skipped=False,
        )

    def read_codebase(
        self,
        root_path: str | Path,
        extensions: Optional[set[str]] = None,
        max_file_size_kb: int = 500,
        max_files: int = 200,
        max_total_size_kb: int = 2000,  # Limit total content size for AI context
    ) -> list[SecureReadResult]:
        """
        Securely read an entire codebase.

        Applies all security checks and returns sanitized content.
        """
        root = Path(root_path)
        results = []
        total_size = 0

        self.log.info("Reading codebase", root=str(root), max_files=max_files)

        # Check consent first
        self.check_consent("analyze_code")

        # Default extensions for source code
        default_extensions = {
            ".py", ".js", ".ts", ".tsx", ".jsx",
            ".java", ".go", ".rs", ".rb", ".php",
            ".html", ".css", ".scss", ".vue", ".svelte",
            ".json", ".yaml", ".yml", ".toml",
            ".md",
        }
        allowed_extensions = extensions or default_extensions

        # Classify the entire directory first
        classifications = self.classifier.classify_directory(root, max_files=max_files * 2)

        # Sort by priority (source code first, then config)
        priority_order = {
            SensitivityLevel.PUBLIC: 0,
            SensitivityLevel.INTERNAL: 1,
            SensitivityLevel.CONFIDENTIAL: 2,
            SensitivityLevel.RESTRICTED: 3,
        }
        sorted_paths = sorted(
            classifications.keys(),
            key=lambda p: priority_order.get(classifications[p].sensitivity, 2)
        )

        for path_str in sorted_paths:
            if len(results) >= max_files:
                self.log.info("Max files reached", count=max_files)
                break

            if total_size >= max_total_size_kb * 1024:
                self.log.info("Max total size reached", size_kb=max_total_size_kb)
                break

            path = Path(path_str)

            # Check extension
            if path.suffix.lower() not in allowed_extensions:
                continue

            # Check file size
            try:
                if path.stat().st_size > max_file_size_kb * 1024:
                    continue
            except Exception:
                continue

            # Read the file securely
            result = self.read_file(path)
            results.append(result)

            if not result.skipped:
                total_size += len(result.content)

        # Log summary
        total = len(results)
        skipped = sum(1 for r in results if r.skipped)
        sanitized = sum(1 for r in results if r.was_sanitized)
        secrets = sum(r.secrets_redacted for r in results)

        self.log.info(
            "Codebase reading complete",
            total_files=total,
            skipped=skipped,
            sanitized=sanitized,
            secrets_redacted=secrets,
            total_size_kb=total_size // 1024,
        )

        return results

    def get_context_for_ai(
        self,
        results: list[SecureReadResult],
        include_metadata: bool = True,
    ) -> str:
        """
        Format read results into a context string for AI consumption.

        This is what gets sent to Claude - guaranteed to be sanitized.
        """
        context_parts = []

        if include_metadata:
            context_parts.append("# CODEBASE ANALYSIS\n")
            context_parts.append(f"Files analyzed: {len(results)}\n")
            context_parts.append(f"Files skipped (restricted/binary): {sum(1 for r in results if r.skipped)}\n")
            context_parts.append(f"Secrets redacted: {sum(r.secrets_redacted for r in results)}\n")
            context_parts.append("\n---\n\n")

        for result in results:
            if result.skipped:
                continue

            if not result.content.strip():
                continue

            # Add file header
            rel_path = result.path
            context_parts.append(f"## File: {rel_path}\n")

            if result.was_sanitized:
                context_parts.append(f"[Note: {result.secrets_redacted} secrets were redacted]\n")

            context_parts.append("```\n")
            context_parts.append(result.content)
            if not result.content.endswith("\n"):
                context_parts.append("\n")
            context_parts.append("```\n\n")

        return "".join(context_parts)

    def get_file_summary(self, results: list[SecureReadResult]) -> dict:
        """Get a summary of files for quick reference."""
        summary = {
            "total": len(results),
            "readable": sum(1 for r in results if not r.skipped),
            "skipped": sum(1 for r in results if r.skipped),
            "sanitized": sum(1 for r in results if r.was_sanitized),
            "secrets_redacted": sum(r.secrets_redacted for r in results),
            "by_sensitivity": {},
            "files": [],
        }

        for result in results:
            level = result.classification.sensitivity.value
            summary["by_sensitivity"][level] = summary["by_sensitivity"].get(level, 0) + 1

            if not result.skipped:
                summary["files"].append({
                    "path": result.path,
                    "sensitivity": level,
                    "sanitized": result.was_sanitized,
                    "secrets_redacted": result.secrets_redacted,
                    "size": len(result.content),
                })

        return summary


def create_secure_reader(
    user_id: str,
    session_id: Optional[str] = None,
    auto_consent_mode: Optional[str] = None,  # "minimal", "standard", "full"
) -> SecureCodeReader:
    """Factory function for creating a secure code reader."""
    consent = ConsentManager(
        session_id=session_id,
        auto_grant_mode=auto_consent_mode,
    )

    return SecureCodeReader(
        user_id=user_id,
        session_id=session_id,
        consent_manager=consent,
    )
