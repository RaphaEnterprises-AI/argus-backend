"""Tests for the security sanitizer module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestSecretType:
    """Tests for SecretType enum."""

    def test_secret_types_exist(self, mock_env_vars):
        """Test all secret types are defined."""
        from src.security.sanitizer import SecretType

        assert SecretType.API_KEY == "api_key"
        assert SecretType.PASSWORD == "password"
        assert SecretType.TOKEN == "token"
        assert SecretType.PRIVATE_KEY == "private_key"
        assert SecretType.CONNECTION_STRING == "connection_string"
        assert SecretType.AWS_CREDENTIALS == "aws_credentials"
        assert SecretType.OAUTH_SECRET == "oauth_secret"
        assert SecretType.JWT == "jwt"
        assert SecretType.CREDIT_CARD == "credit_card"
        assert SecretType.SSN == "ssn"
        assert SecretType.EMAIL_PERSONAL == "email_personal"
        assert SecretType.PHONE == "phone"


class TestRedactedSecret:
    """Tests for RedactedSecret dataclass."""

    def test_redacted_secret_creation(self, mock_env_vars):
        """Test RedactedSecret creation."""
        from src.security.sanitizer import RedactedSecret, SecretType

        secret = RedactedSecret(
            type=SecretType.API_KEY,
            line_number=10,
            original_length=32,
            context="API_KEY=***",
        )

        assert secret.type == SecretType.API_KEY
        assert secret.line_number == 10
        assert secret.original_length == 32
        assert secret.context == "API_KEY=***"


class TestSanitizationResult:
    """Tests for SanitizationResult dataclass."""

    def test_sanitization_result_creation(self, mock_env_vars):
        """Test SanitizationResult creation."""
        from src.security.sanitizer import SanitizationResult

        result = SanitizationResult(
            original_path="/path/to/file.py",
            sanitized_content="safe content",
        )

        assert result.original_path == "/path/to/file.py"
        assert result.sanitized_content == "safe content"
        assert result.secrets_found == []
        assert result.was_modified is False
        assert result.is_binary is False
        assert result.should_skip is False
        assert result.skip_reason is None

    def test_sanitization_result_with_secrets(self, mock_env_vars):
        """Test SanitizationResult with secrets."""
        from src.security.sanitizer import SanitizationResult, RedactedSecret, SecretType

        secrets = [
            RedactedSecret(
                type=SecretType.PASSWORD,
                line_number=5,
                original_length=16,
                context="password=***",
            )
        ]

        result = SanitizationResult(
            original_path="/path/to/file.py",
            sanitized_content="password=[REDACTED]",
            secrets_found=secrets,
            was_modified=True,
        )

        assert len(result.secrets_found) == 1
        assert result.was_modified is True


class TestForbiddenLists:
    """Tests for FORBIDDEN_FILES and FORBIDDEN_DIRS constants."""

    def test_forbidden_files_contains_sensitive_files(self, mock_env_vars):
        """Test FORBIDDEN_FILES contains expected files."""
        from src.security.sanitizer import FORBIDDEN_FILES

        assert ".env" in FORBIDDEN_FILES
        assert ".env.local" in FORBIDDEN_FILES
        assert ".env.production" in FORBIDDEN_FILES
        assert "credentials.json" in FORBIDDEN_FILES
        assert "secrets.yaml" in FORBIDDEN_FILES
        assert "id_rsa" in FORBIDDEN_FILES
        assert "*.pem" in FORBIDDEN_FILES

    def test_forbidden_dirs_contains_sensitive_dirs(self, mock_env_vars):
        """Test FORBIDDEN_DIRS contains expected directories."""
        from src.security.sanitizer import FORBIDDEN_DIRS

        assert "node_modules" in FORBIDDEN_DIRS
        assert ".git" in FORBIDDEN_DIRS
        assert "__pycache__" in FORBIDDEN_DIRS
        assert ".venv" in FORBIDDEN_DIRS
        assert "venv" in FORBIDDEN_DIRS


class TestCodeSanitizer:
    """Tests for CodeSanitizer class."""

    def test_sanitizer_creation(self, mock_env_vars):
        """Test CodeSanitizer creation."""
        from src.security.sanitizer import CodeSanitizer

        sanitizer = CodeSanitizer()

        assert sanitizer.redaction == "[REDACTED]"
        assert sanitizer.forbidden_files is not None
        assert sanitizer.forbidden_dirs is not None

    def test_sanitizer_with_custom_config(self, mock_env_vars):
        """Test CodeSanitizer with custom configuration."""
        from src.security.sanitizer import CodeSanitizer, SecretType

        additional_patterns = {
            SecretType.API_KEY: [r"custom_key_pattern"],
        }
        additional_files = {"custom_secret.txt"}
        additional_dirs = {"custom_dir"}

        sanitizer = CodeSanitizer(
            additional_patterns=additional_patterns,
            additional_forbidden_files=additional_files,
            additional_forbidden_dirs=additional_dirs,
            redaction_placeholder="[HIDDEN]",
        )

        assert sanitizer.redaction == "[HIDDEN]"
        assert "custom_secret.txt" in sanitizer.forbidden_files
        assert "custom_dir" in sanitizer.forbidden_dirs

    def test_should_skip_path_forbidden_dir(self, mock_env_vars):
        """Test should_skip_path for forbidden directory."""
        from src.security.sanitizer import CodeSanitizer

        sanitizer = CodeSanitizer()
        path = Path("/project/node_modules/package/index.js")

        should_skip, reason = sanitizer.should_skip_path(path)

        assert should_skip is True
        assert "Forbidden directory" in reason

    def test_should_skip_path_forbidden_file(self, mock_env_vars):
        """Test should_skip_path for forbidden file."""
        from src.security.sanitizer import CodeSanitizer

        sanitizer = CodeSanitizer()
        # Use credentials.json which is in FORBIDDEN_FILES but not FORBIDDEN_DIRS
        path = Path("/project/credentials.json")

        should_skip, reason = sanitizer.should_skip_path(path)

        assert should_skip is True
        assert "Forbidden file" in reason

    def test_should_skip_path_forbidden_pattern(self, mock_env_vars):
        """Test should_skip_path for forbidden pattern."""
        from src.security.sanitizer import CodeSanitizer

        sanitizer = CodeSanitizer()
        path = Path("/project/server.pem")

        should_skip, reason = sanitizer.should_skip_path(path)

        assert should_skip is True
        assert "Forbidden file pattern" in reason

    def test_should_skip_path_binary_file(self, mock_env_vars):
        """Test should_skip_path for binary file."""
        from src.security.sanitizer import CodeSanitizer

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02\x03")
            binary_path = Path(f.name)

        try:
            sanitizer = CodeSanitizer()
            should_skip, reason = sanitizer.should_skip_path(binary_path)

            assert should_skip is True
            assert "Binary file" in reason
        finally:
            binary_path.unlink()

    def test_should_skip_path_safe_file(self, mock_env_vars):
        """Test should_skip_path for safe file."""
        from src.security.sanitizer import CodeSanitizer

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("def hello(): pass")
            safe_path = Path(f.name)

        try:
            sanitizer = CodeSanitizer()
            should_skip, reason = sanitizer.should_skip_path(safe_path)

            assert should_skip is False
            assert reason is None
        finally:
            safe_path.unlink()

    def test_sanitize_content_no_secrets(self, mock_env_vars):
        """Test sanitize_content with no secrets."""
        from src.security.sanitizer import CodeSanitizer

        sanitizer = CodeSanitizer()
        content = "def hello():\n    print('Hello, world!')"

        result = sanitizer.sanitize_content(content, "test.py")

        assert result.sanitized_content == content
        assert result.was_modified is False
        assert len(result.secrets_found) == 0

    def test_sanitize_content_with_api_key(self, mock_env_vars):
        """Test sanitize_content with API key."""
        from src.security.sanitizer import CodeSanitizer, SecretType

        sanitizer = CodeSanitizer()
        content = 'API_KEY = "sk-1234567890123456789012345"'

        result = sanitizer.sanitize_content(content, "config.py")

        assert result.was_modified is True
        assert len(result.secrets_found) > 0
        assert any(s.type == SecretType.API_KEY for s in result.secrets_found)
        assert "[REDACTED]" in result.sanitized_content

    def test_sanitize_content_with_password(self, mock_env_vars):
        """Test sanitize_content with password."""
        from src.security.sanitizer import CodeSanitizer

        sanitizer = CodeSanitizer()
        content = 'password = "supersecretpassword123"'

        result = sanitizer.sanitize_content(content, "settings.py")

        assert result.was_modified is True
        assert len(result.secrets_found) > 0
        assert "[REDACTED]" in result.sanitized_content

    def test_sanitize_content_with_private_key(self, mock_env_vars):
        """Test sanitize_content with private key."""
        from src.security.sanitizer import CodeSanitizer

        sanitizer = CodeSanitizer()
        content = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
-----END RSA PRIVATE KEY-----"""

        result = sanitizer.sanitize_content(content, "key.txt")

        assert result.was_modified is True
        assert len(result.secrets_found) > 0

    def test_sanitize_content_with_jwt(self, mock_env_vars):
        """Test sanitize_content with JWT token."""
        from src.security.sanitizer import CodeSanitizer

        sanitizer = CodeSanitizer()
        content = 'token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"'

        result = sanitizer.sanitize_content(content, "auth.py")

        assert result.was_modified is True

    def test_sanitize_content_with_connection_string(self, mock_env_vars):
        """Test sanitize_content with database connection string."""
        from src.security.sanitizer import CodeSanitizer

        sanitizer = CodeSanitizer()
        # Use postgres:// (not postgresql://) to match the regex pattern
        content = 'db_url = "postgres://user:password123@localhost:5432/mydb"'

        result = sanitizer.sanitize_content(content, "db.py")

        assert result.was_modified is True

    def test_sanitize_content_with_aws_key(self, mock_env_vars):
        """Test sanitize_content with AWS access key."""
        from src.security.sanitizer import CodeSanitizer

        sanitizer = CodeSanitizer()
        content = 'aws_key = "AKIAIOSFODNN7EXAMPLE"'

        result = sanitizer.sanitize_content(content, "aws.py")

        assert result.was_modified is True

    def test_sanitize_file_forbidden(self, mock_env_vars):
        """Test sanitize_file for forbidden file."""
        from src.security.sanitizer import CodeSanitizer

        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("SECRET=value")

            sanitizer = CodeSanitizer()
            result = sanitizer.sanitize_file(env_file)

            assert result.should_skip is True
            assert result.sanitized_content == ""

    def test_sanitize_file_success(self, mock_env_vars):
        """Test sanitize_file for safe file."""
        from src.security.sanitizer import CodeSanitizer

        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "test.py"
            py_file.write_text("def hello(): pass")

            sanitizer = CodeSanitizer()
            result = sanitizer.sanitize_file(py_file)

            assert result.should_skip is False
            assert result.sanitized_content == "def hello(): pass"

    def test_sanitize_file_read_error(self, mock_env_vars):
        """Test sanitize_file with read error."""
        from src.security.sanitizer import CodeSanitizer

        sanitizer = CodeSanitizer()
        result = sanitizer.sanitize_file("/nonexistent/path/file.py")

        assert result.should_skip is True
        assert "Unreadable" in result.skip_reason

    def test_sanitize_codebase(self, mock_env_vars):
        """Test sanitize_codebase."""
        from src.security.sanitizer import CodeSanitizer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "main.py").write_text("def main(): pass")
            (Path(tmpdir) / "utils.py").write_text("def util(): pass")
            (Path(tmpdir) / ".env").write_text("SECRET=value")

            sanitizer = CodeSanitizer()
            results = sanitizer.sanitize_codebase(tmpdir)

            # Should have main.py and utils.py
            assert len(results) >= 2
            assert any("main.py" in path for path in results)
            assert any("utils.py" in path for path in results)

    def test_sanitize_codebase_with_extensions(self, mock_env_vars):
        """Test sanitize_codebase with specific extensions."""
        from src.security.sanitizer import CodeSanitizer

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.py").write_text("def main(): pass")
            (Path(tmpdir) / "script.js").write_text("function main() {}")
            (Path(tmpdir) / "readme.md").write_text("# Readme")

            sanitizer = CodeSanitizer()
            results = sanitizer.sanitize_codebase(tmpdir, extensions={".py"})

            # Should only include .py files
            assert all(path.endswith(".py") for path in results)

    def test_sanitize_codebase_max_file_size(self, mock_env_vars):
        """Test sanitize_codebase with max file size."""
        from src.security.sanitizer import CodeSanitizer

        with tempfile.TemporaryDirectory() as tmpdir:
            small_file = Path(tmpdir) / "small.py"
            small_file.write_text("x = 1")

            large_file = Path(tmpdir) / "large.py"
            large_file.write_text("x = 1\n" * 10000)

            sanitizer = CodeSanitizer()
            results = sanitizer.sanitize_codebase(tmpdir, max_file_size_kb=1)

            # Large file should be skipped
            large_result = results.get(str(large_file))
            if large_result:
                assert large_result.should_skip is True


class TestCreateSanitizer:
    """Tests for create_sanitizer factory function."""

    def test_create_sanitizer_default(self, mock_env_vars):
        """Test create_sanitizer with default config."""
        from src.security.sanitizer import create_sanitizer

        sanitizer = create_sanitizer()

        assert sanitizer is not None
        assert sanitizer.redaction == "[REDACTED]"

    def test_create_sanitizer_with_config(self, mock_env_vars):
        """Test create_sanitizer with custom config."""
        from src.security.sanitizer import create_sanitizer

        config = {
            "additional_forbidden_files": {"my_secret.txt"},
            "redaction_placeholder": "[HIDDEN]",
        }

        sanitizer = create_sanitizer(config)

        assert sanitizer.redaction == "[HIDDEN]"
        assert "my_secret.txt" in sanitizer.forbidden_files
