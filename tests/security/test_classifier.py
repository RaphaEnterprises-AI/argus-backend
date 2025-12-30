"""Tests for the security classifier module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestSensitivityLevel:
    """Tests for SensitivityLevel enum."""

    def test_sensitivity_levels_exist(self, mock_env_vars):
        """Test all sensitivity levels are defined."""
        from src.security.classifier import SensitivityLevel

        assert SensitivityLevel.PUBLIC == "public"
        assert SensitivityLevel.INTERNAL == "internal"
        assert SensitivityLevel.CONFIDENTIAL == "confidential"
        assert SensitivityLevel.RESTRICTED == "restricted"


class TestDataCategory:
    """Tests for DataCategory enum."""

    def test_data_categories_exist(self, mock_env_vars):
        """Test all data categories are defined."""
        from src.security.classifier import DataCategory

        # Code
        assert DataCategory.SOURCE_CODE == "source_code"
        assert DataCategory.TEST_CODE == "test_code"
        assert DataCategory.BUILD_CONFIG == "build_config"
        assert DataCategory.INFRASTRUCTURE == "infrastructure"

        # Sensitive
        assert DataCategory.CREDENTIALS == "credentials"
        assert DataCategory.SECRETS == "secrets"
        assert DataCategory.PII == "pii"
        assert DataCategory.PHI == "phi"
        assert DataCategory.PCI == "pci"

        # Documentation
        assert DataCategory.DOCUMENTATION == "documentation"
        assert DataCategory.LEGAL == "legal"
        assert DataCategory.FINANCIAL == "financial"

        # Other
        assert DataCategory.BINARY == "binary"
        assert DataCategory.MEDIA == "media"
        assert DataCategory.UNKNOWN == "unknown"


class TestClassification:
    """Tests for Classification dataclass."""

    def test_classification_creation(self, mock_env_vars):
        """Test Classification creation."""
        from src.security.classifier import Classification, SensitivityLevel, DataCategory

        classification = Classification(
            path="/path/to/file.py",
            sensitivity=SensitivityLevel.INTERNAL,
            category=DataCategory.SOURCE_CODE,
            reasons=["Path-based: internal"],
        )

        assert classification.path == "/path/to/file.py"
        assert classification.sensitivity == SensitivityLevel.INTERNAL
        assert classification.category == DataCategory.SOURCE_CODE
        assert classification.pii_detected is False
        assert classification.secrets_detected is False
        assert classification.can_send_to_ai is True
        assert classification.requires_consent is True

    def test_classification_with_secrets(self, mock_env_vars):
        """Test Classification with secrets."""
        from src.security.classifier import Classification, SensitivityLevel, DataCategory

        classification = Classification(
            path="/path/to/.env",
            sensitivity=SensitivityLevel.RESTRICTED,
            category=DataCategory.SECRETS,
            reasons=["Contains secrets"],
            secrets_detected=True,
            can_send_to_ai=False,
        )

        assert classification.secrets_detected is True
        assert classification.can_send_to_ai is False

    def test_classification_with_pii(self, mock_env_vars):
        """Test Classification with PII."""
        from src.security.classifier import Classification, SensitivityLevel, DataCategory

        classification = Classification(
            path="/path/to/users.csv",
            sensitivity=SensitivityLevel.CONFIDENTIAL,
            category=DataCategory.PII,
            reasons=["Contains PII"],
            pii_detected=True,
        )

        assert classification.pii_detected is True


class TestClassificationRules:
    """Tests for CLASSIFICATION_RULES constant."""

    def test_restricted_files(self, mock_env_vars):
        """Test CLASSIFICATION_RULES contains restricted files."""
        from src.security.classifier import CLASSIFICATION_RULES, SensitivityLevel

        restricted = CLASSIFICATION_RULES[SensitivityLevel.RESTRICTED]

        assert ".env" in restricted["files"]
        assert "credentials.json" in restricted["files"]
        assert ".pem" in restricted["extensions"]

    def test_internal_files(self, mock_env_vars):
        """Test CLASSIFICATION_RULES contains internal files."""
        from src.security.classifier import CLASSIFICATION_RULES, SensitivityLevel

        internal = CLASSIFICATION_RULES[SensitivityLevel.INTERNAL]

        assert ".py" in internal["extensions"]
        assert ".js" in internal["extensions"]
        assert "src" in internal["directories"]


class TestPIIPatterns:
    """Tests for PII_PATTERNS constant."""

    def test_pii_patterns_exist(self, mock_env_vars):
        """Test PII_PATTERNS contains expected patterns."""
        from src.security.classifier import PII_PATTERNS
        import re

        # Should have patterns for common PII types
        pii_types = [pii_type for _, pii_type in PII_PATTERNS]

        assert "email" in pii_types
        assert "phone" in pii_types
        assert "ssn" in pii_types
        assert "credit_card" in pii_types
        assert "ip_address" in pii_types


class TestDataClassifier:
    """Tests for DataClassifier class."""

    def test_classifier_creation(self, mock_env_vars):
        """Test DataClassifier creation."""
        from src.security.classifier import DataClassifier

        classifier = DataClassifier()

        assert classifier.scan_for_pii is True
        assert classifier.strict_mode is False

    def test_classifier_creation_with_options(self, mock_env_vars):
        """Test DataClassifier creation with options."""
        from src.security.classifier import DataClassifier

        classifier = DataClassifier(
            scan_for_pii=False,
            strict_mode=True,
        )

        assert classifier.scan_for_pii is False
        assert classifier.strict_mode is True

    def test_classifier_custom_rules(self, mock_env_vars):
        """Test DataClassifier with custom rules."""
        from src.security.classifier import DataClassifier, SensitivityLevel

        custom_rules = {
            SensitivityLevel.RESTRICTED: {
                "files": {"custom_secret.txt"},
            },
        }

        classifier = DataClassifier(custom_rules=custom_rules)

        # Custom file should be in restricted rules
        assert "custom_secret.txt" in classifier.rules[SensitivityLevel.RESTRICTED]["files"]

    def test_classify_path_restricted(self, mock_env_vars):
        """Test classify_path for restricted file."""
        from src.security.classifier import DataClassifier, SensitivityLevel

        classifier = DataClassifier()

        result = classifier.classify_path(Path("/project/.env"))
        assert result == SensitivityLevel.RESTRICTED

        result = classifier.classify_path(Path("/project/secrets.yaml"))
        assert result == SensitivityLevel.RESTRICTED

    def test_classify_path_confidential(self, mock_env_vars):
        """Test classify_path for confidential file."""
        from src.security.classifier import DataClassifier, SensitivityLevel

        classifier = DataClassifier()

        result = classifier.classify_path(Path("/project/config.json"))
        assert result == SensitivityLevel.CONFIDENTIAL

        result = classifier.classify_path(Path("/project/docker-compose.yml"))
        assert result == SensitivityLevel.CONFIDENTIAL

    def test_classify_path_internal(self, mock_env_vars):
        """Test classify_path for internal file."""
        from src.security.classifier import DataClassifier, SensitivityLevel

        classifier = DataClassifier()

        result = classifier.classify_path(Path("/project/src/main.py"))
        assert result == SensitivityLevel.INTERNAL

        result = classifier.classify_path(Path("/project/package.json"))
        assert result == SensitivityLevel.INTERNAL

    def test_classify_path_public(self, mock_env_vars):
        """Test classify_path for public file."""
        from src.security.classifier import DataClassifier, SensitivityLevel

        classifier = DataClassifier()

        result = classifier.classify_path(Path("/project/LICENSE"))
        assert result == SensitivityLevel.PUBLIC

    def test_classify_path_strict_mode(self, mock_env_vars):
        """Test classify_path in strict mode."""
        from src.security.classifier import DataClassifier, SensitivityLevel

        classifier = DataClassifier(strict_mode=True)

        # Unknown file should be CONFIDENTIAL in strict mode
        result = classifier.classify_path(Path("/project/unknown.xyz"))
        assert result == SensitivityLevel.CONFIDENTIAL

    def test_classify_file_basic(self, mock_env_vars):
        """Test classify_file for basic file."""
        from src.security.classifier import DataClassifier, SensitivityLevel, DataCategory

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "main.py"
            test_file.write_text("def main(): pass")

            classifier = DataClassifier()
            result = classifier.classify_file(test_file)

            assert result.sensitivity == SensitivityLevel.INTERNAL
            assert result.category == DataCategory.SOURCE_CODE
            assert result.can_send_to_ai is True

    def test_classify_file_with_pii(self, mock_env_vars):
        """Test classify_file with PII."""
        from src.security.classifier import DataClassifier, SensitivityLevel

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "users.py"
            test_file.write_text('email = "user@example.com"\nphone = "555-123-4567"')

            classifier = DataClassifier()
            result = classifier.classify_file(test_file)

            assert result.pii_detected is True
            assert result.sensitivity == SensitivityLevel.CONFIDENTIAL or result.sensitivity == SensitivityLevel.INTERNAL

    def test_classify_file_with_secrets(self, mock_env_vars):
        """Test classify_file with secrets."""
        from src.security.classifier import DataClassifier, SensitivityLevel

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.py"
            test_file.write_text('API_KEY = "sk-1234567890abcdefghijklmnop"')

            classifier = DataClassifier()
            result = classifier.classify_file(test_file)

            assert result.secrets_detected is True
            assert result.sensitivity == SensitivityLevel.RESTRICTED
            assert result.can_send_to_ai is False

    def test_classify_file_binary(self, mock_env_vars):
        """Test classify_file for binary file."""
        from src.security.classifier import DataClassifier, DataCategory

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "image.png"
            test_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")

            classifier = DataClassifier()
            result = classifier.classify_file(test_file)

            assert result.category == DataCategory.BINARY

    def test_classify_file_nonexistent(self, mock_env_vars):
        """Test classify_file for nonexistent file."""
        from src.security.classifier import DataClassifier

        classifier = DataClassifier()
        result = classifier.classify_file(Path("/nonexistent/file.py"))

        # Should still return a classification based on path
        assert result is not None
        assert result.path == "/nonexistent/file.py"

    def test_classify_content(self, mock_env_vars):
        """Test classify_content method."""
        from src.security.classifier import DataClassifier, SensitivityLevel

        classifier = DataClassifier()

        # Content without PII or secrets
        result = classifier.classify_content("def hello(): pass", source_hint="main.py")
        assert result.pii_detected is False
        assert result.secrets_detected is False

    def test_classify_content_with_pii(self, mock_env_vars):
        """Test classify_content with PII."""
        from src.security.classifier import DataClassifier, SensitivityLevel

        classifier = DataClassifier()

        result = classifier.classify_content('email = "user@example.com"')
        assert result.pii_detected is True
        assert result.sensitivity == SensitivityLevel.CONFIDENTIAL

    def test_classify_content_with_secrets(self, mock_env_vars):
        """Test classify_content with secrets."""
        from src.security.classifier import DataClassifier, SensitivityLevel

        classifier = DataClassifier()

        result = classifier.classify_content('-----BEGIN RSA PRIVATE KEY-----')
        assert result.secrets_detected is True
        assert result.sensitivity == SensitivityLevel.RESTRICTED
        assert result.can_send_to_ai is False

    def test_classify_directory(self, mock_env_vars):
        """Test classify_directory method."""
        from src.security.classifier import DataClassifier

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "main.py").write_text("def main(): pass")
            (Path(tmpdir) / "utils.py").write_text("def util(): pass")
            (Path(tmpdir) / "README.md").write_text("# README")

            classifier = DataClassifier()
            results = classifier.classify_directory(tmpdir)

            assert len(results) >= 3
            assert any("main.py" in path for path in results)

    def test_classify_directory_with_limit(self, mock_env_vars):
        """Test classify_directory with max_files limit."""
        from src.security.classifier import DataClassifier

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many test files
            for i in range(10):
                (Path(tmpdir) / f"file{i}.py").write_text(f"x = {i}")

            classifier = DataClassifier()
            results = classifier.classify_directory(tmpdir, max_files=5)

            assert len(results) == 5

    def test_determine_category_source_code(self, mock_env_vars):
        """Test _determine_category for source code."""
        from src.security.classifier import DataClassifier, DataCategory

        classifier = DataClassifier()

        assert classifier._determine_category(Path("main.py")) == DataCategory.SOURCE_CODE
        assert classifier._determine_category(Path("app.js")) == DataCategory.SOURCE_CODE
        assert classifier._determine_category(Path("index.ts")) == DataCategory.SOURCE_CODE

    def test_determine_category_test_code(self, mock_env_vars):
        """Test _determine_category for test code."""
        from src.security.classifier import DataClassifier, DataCategory

        classifier = DataClassifier()

        assert classifier._determine_category(Path("test_main.py")) == DataCategory.TEST_CODE
        assert classifier._determine_category(Path("main.spec.js")) == DataCategory.TEST_CODE

    def test_determine_category_config(self, mock_env_vars):
        """Test _determine_category for config files."""
        from src.security.classifier import DataClassifier, DataCategory

        classifier = DataClassifier()

        assert classifier._determine_category(Path("config.yaml")) == DataCategory.BUILD_CONFIG
        assert classifier._determine_category(Path("settings.toml")) == DataCategory.BUILD_CONFIG

    def test_determine_category_infrastructure(self, mock_env_vars):
        """Test _determine_category for infrastructure files."""
        from src.security.classifier import DataClassifier, DataCategory

        classifier = DataClassifier()

        assert classifier._determine_category(Path("Dockerfile")) == DataCategory.INFRASTRUCTURE
        # Docker-related files are infrastructure
        assert classifier._determine_category(Path("docker-compose")) == DataCategory.INFRASTRUCTURE
        assert classifier._determine_category(Path("terraform.tf")) == DataCategory.INFRASTRUCTURE

    def test_determine_category_documentation(self, mock_env_vars):
        """Test _determine_category for documentation."""
        from src.security.classifier import DataClassifier, DataCategory

        classifier = DataClassifier()

        assert classifier._determine_category(Path("README.md")) == DataCategory.DOCUMENTATION
        assert classifier._determine_category(Path("docs.txt")) == DataCategory.DOCUMENTATION

    def test_determine_category_media(self, mock_env_vars):
        """Test _determine_category for media files."""
        from src.security.classifier import DataClassifier, DataCategory

        classifier = DataClassifier()

        assert classifier._determine_category(Path("image.png")) == DataCategory.MEDIA
        assert classifier._determine_category(Path("logo.svg")) == DataCategory.MEDIA

    def test_is_binary(self, mock_env_vars):
        """Test _is_binary method."""
        from src.security.classifier import DataClassifier

        with tempfile.TemporaryDirectory() as tmpdir:
            # Text file
            text_file = Path(tmpdir) / "text.txt"
            text_file.write_text("Hello, world!")

            # Binary file
            binary_file = Path(tmpdir) / "binary.bin"
            binary_file.write_bytes(b"\x00\x01\x02\x03")

            classifier = DataClassifier()

            assert classifier._is_binary(text_file) is False
            assert classifier._is_binary(binary_file) is True

    def test_detect_pii(self, mock_env_vars):
        """Test _detect_pii method."""
        from src.security.classifier import DataClassifier

        classifier = DataClassifier()

        # Email
        result = classifier._detect_pii("email: user@example.com")
        assert "email" in result

        # Phone
        result = classifier._detect_pii("phone: 555-123-4567")
        assert "phone" in result

        # SSN
        result = classifier._detect_pii("ssn: 123-45-6789")
        assert "ssn" in result

        # No PII
        result = classifier._detect_pii("def hello(): pass")
        assert len(result) == 0

    def test_has_secrets(self, mock_env_vars):
        """Test _has_secrets method."""
        from src.security.classifier import DataClassifier

        classifier = DataClassifier()

        # API key
        assert classifier._has_secrets('API_KEY = "sk-abcdefghijklmnopqrstuvwxyz"') is True

        # Private key
        assert classifier._has_secrets("-----BEGIN RSA PRIVATE KEY-----") is True

        # AWS key
        assert classifier._has_secrets("AKIAIOSFODNN7EXAMPLE") is True

        # GitHub PAT
        assert classifier._has_secrets("ghp_abcdefghijklmnopqrstuvwxyz0123456789") is True

        # No secrets
        assert classifier._has_secrets("def hello(): pass") is False

    def test_get_retention_days(self, mock_env_vars):
        """Test _get_retention_days method."""
        from src.security.classifier import DataClassifier, SensitivityLevel

        classifier = DataClassifier()

        assert classifier._get_retention_days(SensitivityLevel.PUBLIC) == 365
        assert classifier._get_retention_days(SensitivityLevel.INTERNAL) == 90
        assert classifier._get_retention_days(SensitivityLevel.CONFIDENTIAL) == 30
        assert classifier._get_retention_days(SensitivityLevel.RESTRICTED) == 7

    def test_get_summary(self, mock_env_vars):
        """Test get_summary method."""
        from src.security.classifier import (
            DataClassifier,
            Classification,
            SensitivityLevel,
            DataCategory,
        )

        classifier = DataClassifier()

        classifications = {
            "/file1.py": Classification(
                path="/file1.py",
                sensitivity=SensitivityLevel.INTERNAL,
                category=DataCategory.SOURCE_CODE,
                reasons=[],
            ),
            "/file2.py": Classification(
                path="/file2.py",
                sensitivity=SensitivityLevel.CONFIDENTIAL,
                category=DataCategory.SOURCE_CODE,
                reasons=[],
                pii_detected=True,
            ),
            "/secrets.env": Classification(
                path="/secrets.env",
                sensitivity=SensitivityLevel.RESTRICTED,
                category=DataCategory.SECRETS,
                reasons=[],
                secrets_detected=True,
                can_send_to_ai=False,
            ),
        }

        summary = classifier.get_summary(classifications)

        assert summary["total_files"] == 3
        assert summary["by_sensitivity"]["internal"] == 1
        assert summary["by_sensitivity"]["confidential"] == 1
        assert summary["by_sensitivity"]["restricted"] == 1
        assert "/file2.py" in summary["files_with_pii"]
        assert "/secrets.env" in summary["files_with_secrets"]
        assert summary["can_send_to_ai"] == 2
        assert "/secrets.env" in summary["restricted"]


class TestCreateClassifier:
    """Tests for create_classifier factory function."""

    def test_create_classifier_default(self, mock_env_vars):
        """Test create_classifier with default settings."""
        from src.security.classifier import create_classifier

        classifier = create_classifier()

        assert classifier is not None
        assert classifier.strict_mode is False

    def test_create_classifier_strict(self, mock_env_vars):
        """Test create_classifier with strict mode."""
        from src.security.classifier import create_classifier

        classifier = create_classifier(strict=True)

        assert classifier.strict_mode is True
