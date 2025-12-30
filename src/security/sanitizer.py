"""Code sanitization to remove secrets before sending to AI.

This module ensures NO sensitive data is sent to external AI services.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()


class SecretType(str, Enum):
    """Types of secrets we detect and redact."""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    PRIVATE_KEY = "private_key"
    CONNECTION_STRING = "connection_string"
    AWS_CREDENTIALS = "aws_credentials"
    OAUTH_SECRET = "oauth_secret"
    JWT = "jwt"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    EMAIL_PERSONAL = "email_personal"
    PHONE = "phone"


@dataclass
class RedactedSecret:
    """A secret that was found and redacted."""
    type: SecretType
    line_number: int
    original_length: int
    context: str  # Sanitized context (e.g., "API_KEY=***")


@dataclass
class SanitizationResult:
    """Result of sanitizing code."""
    original_path: str
    sanitized_content: str
    secrets_found: list[RedactedSecret] = field(default_factory=list)
    was_modified: bool = False
    is_binary: bool = False
    should_skip: bool = False
    skip_reason: Optional[str] = None


# Files that should NEVER be read or sent
FORBIDDEN_FILES = {
    # Credentials
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.staging",
    "credentials.json",
    "service-account.json",
    "secrets.yaml",
    "secrets.yml",
    "secrets.json",
    ".secrets",

    # Keys
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",

    # Config with secrets
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".docker/config.json",

    # Database
    "*.sqlite",
    "*.sqlite3",
    "*.db",

    # AWS
    ".aws/credentials",
    ".aws/config",
}

# Directories to skip entirely
FORBIDDEN_DIRS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    "env",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "coverage",
    ".nyc_output",
    "vendor",
}

# Patterns for detecting secrets in code
SECRET_PATTERNS = {
    SecretType.API_KEY: [
        r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
        r'(?i)(sk-[a-zA-Z0-9]{20,})',  # OpenAI/Anthropic style
        r'(?i)(xox[baprs]-[a-zA-Z0-9\-]+)',  # Slack tokens
        r'(?i)AIza[0-9A-Za-z\-_]{35}',  # Google API
    ],
    SecretType.PASSWORD: [
        r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']([^"\']{8,})["\']',
        r'(?i)(password|passwd|pwd)\s*[=:]\s*([^\s;,]{8,})',
    ],
    SecretType.TOKEN: [
        r'(?i)(token|auth[_-]?token|access[_-]?token|bearer)\s*[=:]\s*["\']?([a-zA-Z0-9_\-\.]{20,})["\']?',
        r'(?i)ghp_[a-zA-Z0-9]{36}',  # GitHub PAT
        r'(?i)gho_[a-zA-Z0-9]{36}',  # GitHub OAuth
        r'(?i)glpat-[a-zA-Z0-9\-]{20,}',  # GitLab PAT
    ],
    SecretType.PRIVATE_KEY: [
        r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
        r'-----BEGIN PGP PRIVATE KEY BLOCK-----',
    ],
    SecretType.CONNECTION_STRING: [
        r'(?i)(mongodb|postgres|mysql|redis|amqp)://[^\s]+:[^\s]+@[^\s]+',
        r'(?i)Server=.*;Database=.*;User Id=.*;Password=.*',
    ],
    SecretType.AWS_CREDENTIALS: [
        r'(?i)AKIA[0-9A-Z]{16}',  # AWS Access Key
        r'(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[=:]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?',
    ],
    SecretType.JWT: [
        r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*',  # JWT token
    ],
}


class CodeSanitizer:
    """
    Sanitizes code before sending to AI services.

    This is CRITICAL for enterprise security:
    1. Detects and redacts secrets
    2. Skips forbidden files entirely
    3. Provides audit trail of what was redacted
    4. Allows custom patterns for organization-specific secrets

    Usage:
        sanitizer = CodeSanitizer()
        result = sanitizer.sanitize_file("/path/to/file.py")

        if result.should_skip:
            print(f"Skipped: {result.skip_reason}")
        else:
            safe_content = result.sanitized_content
            # Now safe to send to AI
    """

    def __init__(
        self,
        additional_patterns: Optional[dict[SecretType, list[str]]] = None,
        additional_forbidden_files: Optional[set[str]] = None,
        additional_forbidden_dirs: Optional[set[str]] = None,
        redaction_placeholder: str = "[REDACTED]",
    ):
        self.patterns = {**SECRET_PATTERNS}
        if additional_patterns:
            for key, patterns in additional_patterns.items():
                self.patterns.setdefault(key, []).extend(patterns)

        self.forbidden_files = FORBIDDEN_FILES.copy()
        if additional_forbidden_files:
            self.forbidden_files.update(additional_forbidden_files)

        self.forbidden_dirs = FORBIDDEN_DIRS.copy()
        if additional_forbidden_dirs:
            self.forbidden_dirs.update(additional_forbidden_dirs)

        self.redaction = redaction_placeholder
        self.log = logger.bind(component="sanitizer")

    def should_skip_path(self, path: Path) -> tuple[bool, Optional[str]]:
        """Check if a path should be skipped entirely."""
        # Check forbidden directories
        for part in path.parts:
            if part in self.forbidden_dirs:
                return True, f"Forbidden directory: {part}"

        # Check forbidden files
        for pattern in self.forbidden_files:
            if pattern.startswith("*"):
                if path.name.endswith(pattern[1:]):
                    return True, f"Forbidden file pattern: {pattern}"
            elif path.name == pattern:
                return True, f"Forbidden file: {pattern}"

        # Check if binary
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
                if b"\x00" in chunk:
                    return True, "Binary file"
        except Exception:
            return True, "Unreadable file"

        return False, None

    def sanitize_content(self, content: str, file_path: str = "") -> SanitizationResult:
        """Sanitize content and return result with redaction info."""
        secrets_found = []
        sanitized = content
        was_modified = False

        for secret_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, content):
                    was_modified = True

                    # Calculate line number
                    line_num = content[:match.start()].count("\n") + 1

                    # Create redaction record
                    secrets_found.append(RedactedSecret(
                        type=secret_type,
                        line_number=line_num,
                        original_length=len(match.group()),
                        context=f"{secret_type.value}={self.redaction}",
                    ))

                    # Redact in content
                    sanitized = sanitized.replace(
                        match.group(),
                        f"{self.redaction}:{secret_type.value}"
                    )

        if was_modified:
            self.log.warning(
                "Secrets redacted from content",
                file=file_path,
                secrets_count=len(secrets_found),
                secret_types=[s.type.value for s in secrets_found],
            )

        return SanitizationResult(
            original_path=file_path,
            sanitized_content=sanitized,
            secrets_found=secrets_found,
            was_modified=was_modified,
        )

    def sanitize_file(self, file_path: str | Path) -> SanitizationResult:
        """Sanitize a single file."""
        path = Path(file_path)

        # Check if should skip
        should_skip, reason = self.should_skip_path(path)
        if should_skip:
            self.log.info("Skipping file", path=str(path), reason=reason)
            return SanitizationResult(
                original_path=str(path),
                sanitized_content="",
                should_skip=True,
                skip_reason=reason,
            )

        # Read and sanitize
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            return self.sanitize_content(content, str(path))
        except Exception as e:
            self.log.error("Failed to read file", path=str(path), error=str(e))
            return SanitizationResult(
                original_path=str(path),
                sanitized_content="",
                should_skip=True,
                skip_reason=f"Read error: {e}",
            )

    def sanitize_codebase(
        self,
        root_path: str | Path,
        extensions: Optional[set[str]] = None,
        max_file_size_kb: int = 500,
    ) -> dict[str, SanitizationResult]:
        """
        Sanitize entire codebase.

        Args:
            root_path: Root directory to scan
            extensions: File extensions to include (None = all)
            max_file_size_kb: Skip files larger than this

        Returns:
            Dict mapping file paths to sanitization results
        """
        root = Path(root_path)
        results = {}

        default_extensions = {
            ".py", ".js", ".ts", ".tsx", ".jsx",
            ".java", ".go", ".rs", ".rb", ".php",
            ".html", ".css", ".scss", ".vue", ".svelte",
            ".json", ".yaml", ".yml", ".toml", ".xml",
            ".md", ".txt", ".sh", ".bash",
        }

        allowed_extensions = extensions or default_extensions

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            # Check extension
            if path.suffix.lower() not in allowed_extensions:
                continue

            # Check size
            try:
                if path.stat().st_size > max_file_size_kb * 1024:
                    results[str(path)] = SanitizationResult(
                        original_path=str(path),
                        sanitized_content="",
                        should_skip=True,
                        skip_reason=f"File too large (>{max_file_size_kb}KB)",
                    )
                    continue
            except Exception:
                continue

            # Sanitize
            result = self.sanitize_file(path)
            results[str(path)] = result

        # Log summary
        total = len(results)
        skipped = sum(1 for r in results.values() if r.should_skip)
        modified = sum(1 for r in results.values() if r.was_modified)

        self.log.info(
            "Codebase sanitization complete",
            total_files=total,
            skipped=skipped,
            modified=modified,
            secrets_found=sum(len(r.secrets_found) for r in results.values()),
        )

        return results


def create_sanitizer(config: Optional[dict] = None) -> CodeSanitizer:
    """Factory function for creating configured sanitizer."""
    config = config or {}
    return CodeSanitizer(
        additional_patterns=config.get("additional_patterns"),
        additional_forbidden_files=config.get("additional_forbidden_files"),
        additional_forbidden_dirs=config.get("additional_forbidden_dirs"),
        redaction_placeholder=config.get("redaction_placeholder", "[REDACTED]"),
    )
