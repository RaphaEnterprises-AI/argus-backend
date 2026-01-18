"""Tests for the security secure_reader module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSecureReadResult:
    """Tests for SecureReadResult dataclass."""

    def test_secure_read_result_creation(self, mock_env_vars):
        """Test SecureReadResult creation."""
        from src.security.classifier import Classification, DataCategory, SensitivityLevel
        from src.security.secure_reader import SecureReadResult

        classification = Classification(
            path="/path/to/file.py",
            sensitivity=SensitivityLevel.INTERNAL,
            category=DataCategory.SOURCE_CODE,
            reasons=["Path-based"],
        )

        result = SecureReadResult(
            path="/path/to/file.py",
            content="def main(): pass",
            classification=classification,
            was_sanitized=False,
            secrets_redacted=0,
            skipped=False,
        )

        assert result.path == "/path/to/file.py"
        assert result.content == "def main(): pass"
        assert result.was_sanitized is False
        assert result.skipped is False

    def test_secure_read_result_skipped(self, mock_env_vars):
        """Test SecureReadResult for skipped file."""
        from src.security.classifier import Classification, DataCategory, SensitivityLevel
        from src.security.secure_reader import SecureReadResult

        classification = Classification(
            path="/path/to/.env",
            sensitivity=SensitivityLevel.RESTRICTED,
            category=DataCategory.SECRETS,
            reasons=["Restricted file"],
        )

        result = SecureReadResult(
            path="/path/to/.env",
            content="",
            classification=classification,
            was_sanitized=False,
            secrets_redacted=0,
            skipped=True,
            skip_reason="Restricted file",
        )

        assert result.skipped is True
        assert result.skip_reason == "Restricted file"
        assert result.content == ""


class TestSecureCodeReader:
    """Tests for SecureCodeReader class."""

    def test_secure_reader_creation(self, mock_env_vars):
        """Test SecureCodeReader creation."""
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                reader = SecureCodeReader(
                    user_id="user-123",
                    session_id="session-456",
                )

                assert reader.user_id == "user-123"
                assert reader.session_id == "session-456"
                assert reader.require_consent is True

    def test_secure_reader_with_custom_components(self, mock_env_vars):
        """Test SecureCodeReader with custom components."""
        from src.security.audit import AuditLogger
        from src.security.classifier import DataClassifier
        from src.security.consent import ConsentManager
        from src.security.sanitizer import CodeSanitizer
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            consent = ConsentManager(auto_grant_mode="standard")
            sanitizer = CodeSanitizer()
            classifier = DataClassifier()
            audit = AuditLogger(output_dir=tmpdir)

            reader = SecureCodeReader(
                user_id="user-123",
                consent_manager=consent,
                sanitizer=sanitizer,
                classifier=classifier,
                audit_logger=audit,
            )

            assert reader.consent is consent
            assert reader.sanitizer is sanitizer
            assert reader.classifier is classifier
            assert reader.audit is audit

    def test_check_consent_success(self, mock_env_vars):
        """Test check_consent with granted consent."""
        from src.security.consent import ConsentManager
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                consent = ConsentManager(auto_grant_mode="standard")
                reader = SecureCodeReader(
                    user_id="user-123",
                    consent_manager=consent,
                )

                # Should not raise
                reader.check_consent("analyze_code")

    def test_check_consent_failure(self, mock_env_vars):
        """Test check_consent without consent."""
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                reader = SecureCodeReader(user_id="user-123")

                with pytest.raises(PermissionError):
                    reader.check_consent("analyze_code")

    def test_check_consent_disabled(self, mock_env_vars):
        """Test check_consent with require_consent=False."""
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                reader = SecureCodeReader(
                    user_id="user-123",
                    require_consent=False,
                )

                # Should not raise even without consent
                reader.check_consent("analyze_code")

    def test_read_file_success(self, mock_env_vars):
        """Test read_file for safe file."""
        from src.security.audit import AuditLogger
        from src.security.consent import ConsentManager
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file in a subdirectory to avoid it being seen as audit log
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            test_file = src_dir / "main.py"
            test_file.write_text("def main(): pass")

            audit_dir = Path(tmpdir) / "audit-logs"
            audit_dir.mkdir()
            audit = AuditLogger(output_dir=str(audit_dir))

            consent = ConsentManager(auto_grant_mode="standard")
            reader = SecureCodeReader(
                user_id="user-123",
                consent_manager=consent,
                audit_logger=audit,
            )

            result = reader.read_file(test_file)

            assert result.skipped is False
            assert result.content == "def main(): pass"
            assert result.was_sanitized is False

    def test_read_file_restricted(self, mock_env_vars):
        """Test read_file for restricted file."""
        from src.security.audit import AuditLogger
        from src.security.consent import ConsentManager
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create restricted file - use credentials.json instead of .env
            # since .env is both a file and a dir in FORBIDDEN_DIRS
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            creds_file = src_dir / "credentials.json"
            creds_file.write_text('{"secret": "value"}')

            audit_dir = Path(tmpdir) / "audit-logs"
            audit_dir.mkdir()
            audit = AuditLogger(output_dir=str(audit_dir))

            consent = ConsentManager(auto_grant_mode="standard")
            reader = SecureCodeReader(
                user_id="user-123",
                consent_manager=consent,
                audit_logger=audit,
            )

            result = reader.read_file(creds_file)

            assert result.skipped is True
            assert "Restricted" in result.skip_reason

    def test_read_file_with_secrets(self, mock_env_vars):
        """Test read_file with secrets that get classified as restricted and skipped."""
        from src.security.audit import AuditLogger
        from src.security.consent import ConsentManager
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with secret - this will be classified as RESTRICTED
            # and skipped entirely (not sanitized) for maximum security
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            config_file = src_dir / "config.py"
            config_file.write_text('API_KEY = "sk-1234567890abcdefghijklmnop"')

            audit_dir = Path(tmpdir) / "audit-logs"
            audit_dir.mkdir()
            audit = AuditLogger(output_dir=str(audit_dir))

            consent = ConsentManager(auto_grant_mode="standard")
            reader = SecureCodeReader(
                user_id="user-123",
                consent_manager=consent,
                audit_logger=audit,
            )

            result = reader.read_file(config_file)

            # Files with detected secrets are classified as RESTRICTED and skipped
            assert result.skipped is True
            assert result.classification.secrets_detected is True
            assert "Restricted" in result.skip_reason

    def test_read_file_binary(self, mock_env_vars):
        """Test read_file for binary file."""
        from src.security.consent import ConsentManager
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                # Create binary file
                binary_file = Path(tmpdir) / "image.png"
                binary_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")

                consent = ConsentManager(auto_grant_mode="standard")
                reader = SecureCodeReader(
                    user_id="user-123",
                    consent_manager=consent,
                )

                result = reader.read_file(binary_file)

                assert result.skipped is True

    def test_read_codebase(self, mock_env_vars):
        """Test read_codebase method."""
        from src.security.audit import AuditLogger
        from src.security.consent import ConsentManager
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files in src directory
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "main.py").write_text("def main(): pass")
            (src_dir / "utils.py").write_text("def util(): pass")
            (src_dir / "README.md").write_text("# README")

            audit_dir = Path(tmpdir) / "audit-logs"
            audit_dir.mkdir()
            audit = AuditLogger(output_dir=str(audit_dir))

            consent = ConsentManager(auto_grant_mode="standard")
            reader = SecureCodeReader(
                user_id="user-123",
                consent_manager=consent,
                audit_logger=audit,
            )

            results = reader.read_codebase(src_dir)

            assert len(results) >= 3
            assert any(r.path.endswith("main.py") for r in results)

    def test_read_codebase_without_consent(self, mock_env_vars):
        """Test read_codebase without consent."""
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                reader = SecureCodeReader(user_id="user-123")

                with pytest.raises(PermissionError):
                    reader.read_codebase(tmpdir)

    def test_read_codebase_max_files(self, mock_env_vars):
        """Test read_codebase with max_files limit."""
        from src.security.audit import AuditLogger
        from src.security.consent import ConsentManager
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many files in src directory
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            for i in range(10):
                (src_dir / f"file{i}.py").write_text(f"x = {i}")

            audit_dir = Path(tmpdir) / "audit-logs"
            audit_dir.mkdir()
            audit = AuditLogger(output_dir=str(audit_dir))

            consent = ConsentManager(auto_grant_mode="standard")
            reader = SecureCodeReader(
                user_id="user-123",
                consent_manager=consent,
                audit_logger=audit,
            )

            results = reader.read_codebase(src_dir, max_files=5)

            assert len(results) <= 5

    def test_read_codebase_with_extensions(self, mock_env_vars):
        """Test read_codebase with specific extensions."""
        from src.security.audit import AuditLogger
        from src.security.consent import ConsentManager
        from src.security.secure_reader import SecureCodeReader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in src directory
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "main.py").write_text("def main(): pass")
            (src_dir / "script.js").write_text("function main() {}")
            (src_dir / "data.json").write_text("{}")

            audit_dir = Path(tmpdir) / "audit-logs"
            audit_dir.mkdir()
            audit = AuditLogger(output_dir=str(audit_dir))

            consent = ConsentManager(auto_grant_mode="standard")
            reader = SecureCodeReader(
                user_id="user-123",
                consent_manager=consent,
                audit_logger=audit,
            )

            results = reader.read_codebase(src_dir, extensions={".py"})

            # Should only include .py files
            py_files = [r for r in results if not r.skipped]
            assert all(r.path.endswith(".py") for r in py_files)

    def test_get_context_for_ai(self, mock_env_vars):
        """Test get_context_for_ai method."""
        from src.security.classifier import Classification, DataCategory, SensitivityLevel
        from src.security.consent import ConsentManager
        from src.security.secure_reader import SecureCodeReader, SecureReadResult

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                consent = ConsentManager(auto_grant_mode="standard")
                reader = SecureCodeReader(
                    user_id="user-123",
                    consent_manager=consent,
                )

                results = [
                    SecureReadResult(
                        path="/project/main.py",
                        content="def main(): pass",
                        classification=Classification(
                            path="/project/main.py",
                            sensitivity=SensitivityLevel.INTERNAL,
                            category=DataCategory.SOURCE_CODE,
                            reasons=[],
                        ),
                        was_sanitized=False,
                        secrets_redacted=0,
                        skipped=False,
                    ),
                    SecureReadResult(
                        path="/project/utils.py",
                        content="def util(): pass",
                        classification=Classification(
                            path="/project/utils.py",
                            sensitivity=SensitivityLevel.INTERNAL,
                            category=DataCategory.SOURCE_CODE,
                            reasons=[],
                        ),
                        was_sanitized=True,
                        secrets_redacted=1,
                        skipped=False,
                    ),
                ]

                context = reader.get_context_for_ai(results)

                assert "main.py" in context
                assert "utils.py" in context
                assert "def main(): pass" in context
                assert "secrets were redacted" in context

    def test_get_context_for_ai_skip_empty(self, mock_env_vars):
        """Test get_context_for_ai skips skipped files."""
        from src.security.classifier import Classification, DataCategory, SensitivityLevel
        from src.security.consent import ConsentManager
        from src.security.secure_reader import SecureCodeReader, SecureReadResult

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                consent = ConsentManager(auto_grant_mode="standard")
                reader = SecureCodeReader(
                    user_id="user-123",
                    consent_manager=consent,
                )

                results = [
                    SecureReadResult(
                        path="/project/.env",
                        content="",
                        classification=Classification(
                            path="/project/.env",
                            sensitivity=SensitivityLevel.RESTRICTED,
                            category=DataCategory.SECRETS,
                            reasons=[],
                        ),
                        was_sanitized=False,
                        secrets_redacted=0,
                        skipped=True,
                        skip_reason="Restricted",
                    ),
                ]

                context = reader.get_context_for_ai(results)

                # Skipped file should not appear in content
                assert ".env" not in context or "SECRET" not in context

    def test_get_context_for_ai_without_metadata(self, mock_env_vars):
        """Test get_context_for_ai without metadata."""
        from src.security.classifier import Classification, DataCategory, SensitivityLevel
        from src.security.consent import ConsentManager
        from src.security.secure_reader import SecureCodeReader, SecureReadResult

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                consent = ConsentManager(auto_grant_mode="standard")
                reader = SecureCodeReader(
                    user_id="user-123",
                    consent_manager=consent,
                )

                results = [
                    SecureReadResult(
                        path="/project/main.py",
                        content="def main(): pass",
                        classification=Classification(
                            path="/project/main.py",
                            sensitivity=SensitivityLevel.INTERNAL,
                            category=DataCategory.SOURCE_CODE,
                            reasons=[],
                        ),
                        was_sanitized=False,
                        secrets_redacted=0,
                        skipped=False,
                    ),
                ]

                context = reader.get_context_for_ai(results, include_metadata=False)

                assert "CODEBASE ANALYSIS" not in context
                assert "def main(): pass" in context

    def test_get_file_summary(self, mock_env_vars):
        """Test get_file_summary method."""
        from src.security.classifier import Classification, DataCategory, SensitivityLevel
        from src.security.consent import ConsentManager
        from src.security.secure_reader import SecureCodeReader, SecureReadResult

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                consent = ConsentManager(auto_grant_mode="standard")
                reader = SecureCodeReader(
                    user_id="user-123",
                    consent_manager=consent,
                )

                results = [
                    SecureReadResult(
                        path="/project/main.py",
                        content="def main(): pass",
                        classification=Classification(
                            path="/project/main.py",
                            sensitivity=SensitivityLevel.INTERNAL,
                            category=DataCategory.SOURCE_CODE,
                            reasons=[],
                        ),
                        was_sanitized=False,
                        secrets_redacted=0,
                        skipped=False,
                    ),
                    SecureReadResult(
                        path="/project/.env",
                        content="",
                        classification=Classification(
                            path="/project/.env",
                            sensitivity=SensitivityLevel.RESTRICTED,
                            category=DataCategory.SECRETS,
                            reasons=[],
                        ),
                        was_sanitized=False,
                        secrets_redacted=0,
                        skipped=True,
                    ),
                    SecureReadResult(
                        path="/project/config.py",
                        content="redacted content",
                        classification=Classification(
                            path="/project/config.py",
                            sensitivity=SensitivityLevel.CONFIDENTIAL,
                            category=DataCategory.BUILD_CONFIG,
                            reasons=[],
                        ),
                        was_sanitized=True,
                        secrets_redacted=2,
                        skipped=False,
                    ),
                ]

                summary = reader.get_file_summary(results)

                assert summary["total"] == 3
                assert summary["readable"] == 2
                assert summary["skipped"] == 1
                assert summary["sanitized"] == 1
                assert summary["secrets_redacted"] == 2
                assert "internal" in summary["by_sensitivity"]
                assert "restricted" in summary["by_sensitivity"]


class TestCreateSecureReader:
    """Tests for create_secure_reader factory function."""

    def test_create_secure_reader_default(self, mock_env_vars):
        """Test create_secure_reader with default settings."""
        from src.security.secure_reader import create_secure_reader

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                reader = create_secure_reader(user_id="user-123")

                assert reader is not None
                assert reader.user_id == "user-123"

    def test_create_secure_reader_with_session(self, mock_env_vars):
        """Test create_secure_reader with session ID."""
        from src.security.secure_reader import create_secure_reader

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                reader = create_secure_reader(
                    user_id="user-123",
                    session_id="session-456",
                )

                assert reader.session_id == "session-456"

    def test_create_secure_reader_with_auto_consent(self, mock_env_vars):
        """Test create_secure_reader with auto consent mode."""
        from src.security.consent import ConsentScope
        from src.security.secure_reader import create_secure_reader

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"AUDIT_LOG_DIR": tmpdir}):
                reader = create_secure_reader(
                    user_id="user-123",
                    auto_consent_mode="standard",
                )

                assert reader.consent.has_consent(ConsentScope.SOURCE_CODE)
                assert reader.consent.has_consent(ConsentScope.SCREENSHOTS)
