"""Data classification for enterprise data governance.

Classifies files and content by sensitivity level to ensure
proper handling and access controls.
"""

import mimetypes
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import structlog

logger = structlog.get_logger()


class SensitivityLevel(str, Enum):
    """Data sensitivity levels following enterprise classification."""
    PUBLIC = "public"  # Can be shared externally
    INTERNAL = "internal"  # Company internal only
    CONFIDENTIAL = "confidential"  # Need-to-know basis
    RESTRICTED = "restricted"  # Highly sensitive, special handling required


class DataCategory(str, Enum):
    """Categories of data for regulatory compliance."""
    # Code
    SOURCE_CODE = "source_code"
    TEST_CODE = "test_code"
    BUILD_CONFIG = "build_config"
    INFRASTRUCTURE = "infrastructure"

    # Sensitive
    CREDENTIALS = "credentials"
    SECRETS = "secrets"
    PII = "pii"  # Personally Identifiable Information
    PHI = "phi"  # Protected Health Information
    PCI = "pci"  # Payment Card Industry data

    # Documentation
    DOCUMENTATION = "documentation"
    LEGAL = "legal"
    FINANCIAL = "financial"

    # Other
    BINARY = "binary"
    MEDIA = "media"
    UNKNOWN = "unknown"


@dataclass
class Classification:
    """Classification result for a file or content."""
    path: str
    sensitivity: SensitivityLevel
    category: DataCategory
    reasons: list[str]
    pii_detected: bool = False
    secrets_detected: bool = False
    can_send_to_ai: bool = True
    requires_consent: bool = True
    retention_days: int = 90


# File patterns for classification
CLASSIFICATION_RULES = {
    # RESTRICTED - Never send to AI
    SensitivityLevel.RESTRICTED: {
        "files": {
            ".env", ".env.local", ".env.production", ".env.staging",
            "credentials.json", "service-account.json", "secrets.yaml",
            "id_rsa", "id_ed25519", "id_ecdsa",
            ".npmrc", ".pypirc", ".netrc",
        },
        "extensions": {
            ".pem", ".key", ".p12", ".pfx", ".keystore",
        },
        "patterns": [
            r".*password.*",
            r".*secret.*\.json$",
            r".*credentials.*",
            r".*\.kdbx$",  # KeePass
        ],
        "directories": {
            ".aws", ".ssh", ".gnupg", ".docker",
        },
    },

    # CONFIDENTIAL - Can send if sanitized
    SensitivityLevel.CONFIDENTIAL: {
        "files": {
            "config.json", "settings.json", "application.yml",
            "docker-compose.yml", "kubernetes.yaml",
        },
        "extensions": {
            ".conf", ".cfg", ".ini",
        },
        "patterns": [
            r".*config.*\.(json|yaml|yml|toml)$",
            r".*settings.*\.(py|json)$",
            r"Dockerfile.*",
            r".*\.sql$",  # SQL files may contain data
        ],
        "directories": {
            "config", "configs", "settings", "deploy", "k8s",
        },
    },

    # INTERNAL - Safe to send to AI with consent
    SensitivityLevel.INTERNAL: {
        "files": {
            "package.json", "pyproject.toml", "setup.py",
            "requirements.txt", "Gemfile", "go.mod",
            "README.md", "CHANGELOG.md",
        },
        "extensions": {
            ".py", ".js", ".ts", ".tsx", ".jsx",
            ".java", ".go", ".rs", ".rb", ".php",
            ".html", ".css", ".scss", ".vue", ".svelte",
            ".md", ".txt", ".rst",
        },
        "patterns": [
            r".*test.*\.(py|js|ts)$",
            r".*spec.*\.(py|js|ts)$",
            r".*_test\.(py|go)$",
        ],
        "directories": {
            "src", "lib", "tests", "test", "docs",
        },
    },

    # PUBLIC - Open source, documentation
    SensitivityLevel.PUBLIC: {
        "files": {
            "LICENSE", "LICENSE.md", "LICENSE.txt",
            "CONTRIBUTING.md", "CODE_OF_CONDUCT.md",
            ".gitignore", ".editorconfig",
        },
        "extensions": set(),
        "patterns": [],
        "directories": set(),
    },
}

# PII detection patterns
PII_PATTERNS = [
    # Email addresses
    (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "email"),
    # Phone numbers (various formats)
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone"),
    (r"\+\d{1,3}[-.\s]?\d{1,14}", "phone_intl"),
    # SSN
    (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
    # Credit card (basic pattern)
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "credit_card"),
    # IP addresses
    (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "ip_address"),
]


class DataClassifier:
    """
    Classifies data by sensitivity level for proper handling.

    Enterprise features:
    - Multi-level classification (public, internal, confidential, restricted)
    - PII detection (emails, phones, SSN, credit cards)
    - Regulatory category tagging (PCI, PHI, PII)
    - Policy-based access control recommendations
    - Audit-friendly classification records

    Usage:
        classifier = DataClassifier()

        # Classify a file
        result = classifier.classify_file("/path/to/file.py")
        if result.sensitivity == SensitivityLevel.RESTRICTED:
            print("Cannot send to AI")
        elif result.pii_detected:
            print("PII detected - requires special handling")
        else:
            print(f"Safe to process: {result.sensitivity.value}")
    """

    def __init__(
        self,
        custom_rules: dict | None = None,
        scan_for_pii: bool = True,
        strict_mode: bool = False,  # In strict mode, unknown = CONFIDENTIAL
    ):
        self.rules = {**CLASSIFICATION_RULES}
        if custom_rules:
            for level, rules in custom_rules.items():
                self.rules.setdefault(level, {})
                for key, values in rules.items():
                    self.rules[level].setdefault(key, set()).update(values)

        self.scan_for_pii = scan_for_pii
        self.strict_mode = strict_mode
        self.log = logger.bind(component="classifier")

    def classify_path(self, path: Path | str) -> SensitivityLevel:
        """Quickly classify a path by its name/location (no content scan)."""
        path = Path(path)

        # Check each sensitivity level in order of severity
        for level in [
            SensitivityLevel.RESTRICTED,
            SensitivityLevel.CONFIDENTIAL,
            SensitivityLevel.INTERNAL,
            SensitivityLevel.PUBLIC,
        ]:
            rules = self.rules.get(level, {})

            # Check exact filename
            if path.name in rules.get("files", set()):
                return level

            # Check extension
            if path.suffix.lower() in rules.get("extensions", set()):
                return level

            # Check directory
            for part in path.parts:
                if part in rules.get("directories", set()):
                    return level

            # Check patterns
            for pattern in rules.get("patterns", []):
                if re.match(pattern, str(path), re.IGNORECASE):
                    return level

        # Default classification
        return SensitivityLevel.CONFIDENTIAL if self.strict_mode else SensitivityLevel.INTERNAL

    def classify_file(self, file_path: str | Path) -> Classification:
        """Fully classify a file including content analysis."""
        path = Path(file_path)
        reasons = []
        pii_detected = False
        secrets_detected = False

        # Start with path-based classification
        sensitivity = self.classify_path(path)
        reasons.append(f"Path-based: {sensitivity.value}")

        # Determine category
        category = self._determine_category(path)

        # Check if it's a binary file
        is_binary = self._is_binary(path)
        if is_binary:
            category = DataCategory.BINARY
            reasons.append("Binary file detected")

        # Content-based classification (if readable text file)
        if not is_binary and path.exists() and path.is_file():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")

                # Scan for PII
                if self.scan_for_pii:
                    pii_types = self._detect_pii(content)
                    if pii_types:
                        pii_detected = True
                        category = DataCategory.PII
                        sensitivity = max(sensitivity, SensitivityLevel.CONFIDENTIAL, key=lambda x: list(SensitivityLevel).index(x))
                        reasons.append(f"PII detected: {', '.join(pii_types)}")

                # Check for secrets patterns
                if self._has_secrets(content):
                    secrets_detected = True
                    sensitivity = SensitivityLevel.RESTRICTED
                    category = DataCategory.SECRETS
                    reasons.append("Secrets/credentials pattern detected")

            except Exception as e:
                reasons.append(f"Content scan failed: {e}")

        # Determine if safe to send to AI
        can_send = sensitivity not in (SensitivityLevel.RESTRICTED,)

        return Classification(
            path=str(path),
            sensitivity=sensitivity,
            category=category,
            reasons=reasons,
            pii_detected=pii_detected,
            secrets_detected=secrets_detected,
            can_send_to_ai=can_send,
            requires_consent=sensitivity != SensitivityLevel.PUBLIC,
            retention_days=self._get_retention_days(sensitivity),
        )

    def classify_content(
        self,
        content: str,
        source_hint: str | None = None,
    ) -> Classification:
        """Classify content without a file path."""
        reasons = []
        pii_detected = False
        secrets_detected = False
        sensitivity = SensitivityLevel.INTERNAL

        # Check for PII
        if self.scan_for_pii:
            pii_types = self._detect_pii(content)
            if pii_types:
                pii_detected = True
                sensitivity = SensitivityLevel.CONFIDENTIAL
                reasons.append(f"PII detected: {', '.join(pii_types)}")

        # Check for secrets
        if self._has_secrets(content):
            secrets_detected = True
            sensitivity = SensitivityLevel.RESTRICTED
            reasons.append("Secrets/credentials pattern detected")

        # Use source hint for category
        category = DataCategory.UNKNOWN
        if source_hint:
            category = self._determine_category(Path(source_hint))

        can_send = sensitivity not in (SensitivityLevel.RESTRICTED,)

        return Classification(
            path=source_hint or "<content>",
            sensitivity=sensitivity,
            category=category,
            reasons=reasons,
            pii_detected=pii_detected,
            secrets_detected=secrets_detected,
            can_send_to_ai=can_send,
            requires_consent=True,
            retention_days=self._get_retention_days(sensitivity),
        )

    def classify_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        max_files: int = 1000,
    ) -> dict[str, Classification]:
        """Classify all files in a directory."""
        root = Path(dir_path)
        results = {}
        count = 0

        iterator = root.rglob("*") if recursive else root.glob("*")

        for path in iterator:
            if not path.is_file():
                continue

            # Quick skip for obviously excluded paths
            if any(part.startswith(".") and part not in {".env"} for part in path.parts):
                continue

            results[str(path)] = self.classify_file(path)
            count += 1

            if count >= max_files:
                self.log.warning("Max file limit reached", limit=max_files)
                break

        # Log summary
        by_level = {}
        for classification in results.values():
            by_level[classification.sensitivity.value] = by_level.get(classification.sensitivity.value, 0) + 1

        self.log.info(
            "Directory classification complete",
            total_files=len(results),
            by_sensitivity=by_level,
        )

        return results

    def _determine_category(self, path: Path) -> DataCategory:
        """Determine the data category based on path."""
        name = path.name.lower()
        suffix = path.suffix.lower()

        # Credentials/secrets
        if any(x in name for x in ["secret", "credential", "password", "key", "token"]):
            return DataCategory.SECRETS
        if suffix in {".pem", ".key", ".p12", ".pfx"}:
            return DataCategory.CREDENTIALS

        # Config
        if any(x in name for x in ["config", "settings", ".env"]):
            return DataCategory.BUILD_CONFIG
        if suffix in {".yaml", ".yml", ".toml", ".ini", ".conf"}:
            return DataCategory.BUILD_CONFIG

        # Infrastructure
        if any(x in name for x in ["docker", "kubernetes", "k8s", "helm", "terraform"]):
            return DataCategory.INFRASTRUCTURE

        # Test code
        if any(x in name for x in ["test", "spec", "_test", ".test"]):
            return DataCategory.TEST_CODE

        # Source code
        if suffix in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb", ".php"}:
            return DataCategory.SOURCE_CODE

        # Documentation
        if suffix in {".md", ".rst", ".txt", ".adoc"}:
            return DataCategory.DOCUMENTATION

        # Media/binary
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico"}:
            return DataCategory.MEDIA

        return DataCategory.UNKNOWN

    def _is_binary(self, path: Path) -> bool:
        """Check if a file is binary."""
        try:
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type and not mime_type.startswith("text/"):
                return True

            # Read first chunk and check for null bytes
            with open(path, "rb") as f:
                chunk = f.read(8192)
                return b"\x00" in chunk
        except Exception:
            return True

    def _detect_pii(self, content: str) -> list[str]:
        """Detect PII patterns in content."""
        detected = []
        for pattern, pii_type in PII_PATTERNS:
            if re.search(pattern, content):
                detected.append(pii_type)
        return detected

    def _has_secrets(self, content: str) -> bool:
        """Check if content appears to contain secrets."""
        secret_patterns = [
            r"(?i)(api[_-]?key|apikey)\s*[=:]\s*[\"']?[a-zA-Z0-9_\-]{20,}",
            r"(?i)(password|passwd|pwd)\s*[=:]\s*[\"'][^\"']{8,}[\"']",
            r"(?i)(token|auth[_-]?token|access[_-]?token)\s*[=:]\s*[\"']?[a-zA-Z0-9_\-\.]{20,}",
            r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
            r"(?i)AKIA[0-9A-Z]{16}",  # AWS access key
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub PAT
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI/Anthropic
        ]

        for pattern in secret_patterns:
            if re.search(pattern, content):
                return True
        return False

    def _get_retention_days(self, sensitivity: SensitivityLevel) -> int:
        """Get data retention period based on sensitivity."""
        retention_map = {
            SensitivityLevel.PUBLIC: 365,
            SensitivityLevel.INTERNAL: 90,
            SensitivityLevel.CONFIDENTIAL: 30,
            SensitivityLevel.RESTRICTED: 7,
        }
        return retention_map.get(sensitivity, 90)

    def get_summary(self, classifications: dict[str, Classification]) -> dict:
        """Generate a summary report of classifications."""
        summary = {
            "total_files": len(classifications),
            "by_sensitivity": {},
            "by_category": {},
            "files_with_pii": [],
            "files_with_secrets": [],
            "can_send_to_ai": 0,
            "restricted": [],
        }

        for path, classification in classifications.items():
            # Count by sensitivity
            level = classification.sensitivity.value
            summary["by_sensitivity"][level] = summary["by_sensitivity"].get(level, 0) + 1

            # Count by category
            cat = classification.category.value
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1

            # Track special cases
            if classification.pii_detected:
                summary["files_with_pii"].append(path)
            if classification.secrets_detected:
                summary["files_with_secrets"].append(path)
            if classification.can_send_to_ai:
                summary["can_send_to_ai"] += 1
            if classification.sensitivity == SensitivityLevel.RESTRICTED:
                summary["restricted"].append(path)

        return summary


def create_classifier(strict: bool = False) -> DataClassifier:
    """Factory function for creating a data classifier."""
    return DataClassifier(strict_mode=strict)
