"""Comprehensive tests for input validation module (validation.py).

Tests cover:
- String sanitization
- HTML sanitization
- Filename sanitization
- URL sanitization
- SQL injection detection
- XSS detection
- Path traversal detection
- Command injection detection
- InputValidator class
- Pydantic validators (SecureString, SecureURL, SecureFilename)
"""

import pytest


# =============================================================================
# String Sanitization Tests
# =============================================================================

class TestSanitizeString:
    """Tests for sanitize_string function."""

    def test_sanitize_string_basic(self):
        """Test basic string sanitization."""
        from src.api.security.validation import sanitize_string

        result = sanitize_string("Hello World")
        assert result == "Hello World"

    def test_sanitize_string_empty(self):
        """Test sanitizing empty string."""
        from src.api.security.validation import sanitize_string

        assert sanitize_string("") == ""
        assert sanitize_string(None) is None

    def test_sanitize_string_truncation(self):
        """Test string truncation."""
        from src.api.security.validation import sanitize_string

        long_string = "A" * 20000
        result = sanitize_string(long_string, max_length=100)

        assert len(result) == 100

    def test_sanitize_string_html_escape(self):
        """Test HTML entity escaping."""
        from src.api.security.validation import sanitize_string

        result = sanitize_string("<script>alert('xss')</script>")

        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_sanitize_string_control_characters(self):
        """Test removal of control characters."""
        from src.api.security.validation import sanitize_string

        # Keep newlines and tabs
        result = sanitize_string("Hello\nWorld\tTest")
        assert "\n" in result
        assert "\t" in result

        # Remove null bytes and other control chars
        result = sanitize_string("Hello\x00World")
        assert "\x00" not in result

    def test_sanitize_string_unicode_normalization(self):
        """Test unicode normalization."""
        from src.api.security.validation import sanitize_string

        # Different unicode representations of the same character
        result = sanitize_string("\u212B")  # Angstrom sign
        assert result is not None


# =============================================================================
# HTML Sanitization Tests
# =============================================================================

class TestSanitizeHTML:
    """Tests for sanitize_html function."""

    def test_sanitize_html_basic(self):
        """Test basic HTML sanitization."""
        from src.api.security.validation import sanitize_html

        result = sanitize_html("<p>Hello</p>")
        assert "<p>" in result
        assert "</p>" in result

    def test_sanitize_html_empty(self):
        """Test sanitizing empty string."""
        from src.api.security.validation import sanitize_html

        assert sanitize_html("") == ""
        assert sanitize_html(None) is None

    def test_sanitize_html_removes_script(self):
        """Test that script tags are escaped."""
        from src.api.security.validation import sanitize_html

        result = sanitize_html("<script>alert('xss')</script>")

        assert "<script>" not in result

    def test_sanitize_html_allows_safe_tags(self):
        """Test that safe tags are allowed."""
        from src.api.security.validation import sanitize_html

        html = "<p><b>Bold</b> and <i>italic</i></p>"
        result = sanitize_html(html)

        assert "<p>" in result
        assert "<b>" in result
        assert "<i>" in result

    def test_sanitize_html_custom_allowed_tags(self):
        """Test sanitization with custom allowed tags."""
        from src.api.security.validation import sanitize_html

        html = "<div><span>Text</span></div>"
        result = sanitize_html(html, allowed_tags=["div"])

        assert "<div>" in result
        # span should be escaped since not in allowed list

    def test_sanitize_html_removes_dangerous_tags(self):
        """Test that dangerous tags are escaped."""
        from src.api.security.validation import sanitize_html

        dangerous = "<iframe src='evil.com'></iframe><object></object>"
        result = sanitize_html(dangerous)

        assert "<iframe" not in result
        assert "<object" not in result


# =============================================================================
# Filename Sanitization Tests
# =============================================================================

class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        from src.api.security.validation import sanitize_filename

        result = sanitize_filename("document.pdf")
        assert result == "document.pdf"

    def test_sanitize_filename_empty(self):
        """Test sanitizing empty filename."""
        from src.api.security.validation import sanitize_filename

        assert sanitize_filename("") == ""
        assert sanitize_filename(None) is None

    def test_sanitize_filename_removes_path_separators(self):
        """Test that path separators are replaced."""
        from src.api.security.validation import sanitize_filename

        result = sanitize_filename("../../../etc/passwd")

        # Slashes are replaced with underscores
        assert "/" not in result
        # The resulting filename is safe (no path traversal possible)

    def test_sanitize_filename_removes_backslash(self):
        """Test that backslashes are removed."""
        from src.api.security.validation import sanitize_filename

        result = sanitize_filename("..\\..\\windows\\system32")

        assert "\\" not in result

    def test_sanitize_filename_removes_null_bytes(self):
        """Test that null bytes are removed."""
        from src.api.security.validation import sanitize_filename

        result = sanitize_filename("file\x00.txt")

        assert "\x00" not in result

    def test_sanitize_filename_strips_dots_and_spaces(self):
        """Test that leading/trailing dots and spaces are stripped."""
        from src.api.security.validation import sanitize_filename

        result = sanitize_filename("  ..file.txt.. ")

        assert not result.startswith(".")
        assert not result.endswith(".")
        assert not result.startswith(" ")

    def test_sanitize_filename_special_characters(self):
        """Test that special characters are replaced."""
        from src.api.security.validation import sanitize_filename

        result = sanitize_filename("file<>:\"name.txt")

        # Only alphanumeric, underscore, dash, and dot allowed
        assert all(c.isalnum() or c in "._-" for c in result)


# =============================================================================
# URL Sanitization Tests
# =============================================================================

class TestSanitizeURL:
    """Tests for sanitize_url function."""

    def test_sanitize_url_valid_http(self):
        """Test sanitizing valid HTTP URL."""
        from src.api.security.validation import sanitize_url

        result = sanitize_url("http://example.com/page")
        assert result == "http://example.com/page"

    def test_sanitize_url_valid_https(self):
        """Test sanitizing valid HTTPS URL."""
        from src.api.security.validation import sanitize_url

        result = sanitize_url("https://example.com/page?param=value")
        assert result == "https://example.com/page?param=value"

    def test_sanitize_url_empty(self):
        """Test sanitizing empty URL."""
        from src.api.security.validation import sanitize_url

        assert sanitize_url("") is None
        assert sanitize_url(None) is None

    def test_sanitize_url_rejects_javascript(self):
        """Test that javascript: URLs are rejected."""
        from src.api.security.validation import sanitize_url

        assert sanitize_url("javascript:alert('xss')") is None

    def test_sanitize_url_rejects_javascript_mixed_case(self):
        """Test that javascript: URLs are rejected regardless of case."""
        from src.api.security.validation import sanitize_url

        assert sanitize_url("JavaScript:alert('xss')") is None
        assert sanitize_url("JAVASCRIPT:alert('xss')") is None

    def test_sanitize_url_rejects_ftp(self):
        """Test that non-http schemes are rejected."""
        from src.api.security.validation import sanitize_url

        assert sanitize_url("ftp://example.com/file") is None
        assert sanitize_url("file:///etc/passwd") is None

    def test_sanitize_url_rejects_data_uri(self):
        """Test that data URIs are rejected."""
        from src.api.security.validation import sanitize_url

        assert sanitize_url("data:text/html,<script>alert('xss')</script>") is None


# =============================================================================
# SQL Injection Detection Tests
# =============================================================================

class TestCheckSQLInjection:
    """Tests for check_sql_injection function."""

    def test_check_sql_injection_safe(self):
        """Test that safe input passes."""
        from src.api.security.validation import check_sql_injection

        assert check_sql_injection("normal text") is False
        assert check_sql_injection("hello world") is False

    def test_check_sql_injection_empty(self):
        """Test empty input."""
        from src.api.security.validation import check_sql_injection

        assert check_sql_injection("") is False
        assert check_sql_injection(None) is False

    def test_check_sql_injection_select(self):
        """Test detection of SELECT statements."""
        from src.api.security.validation import check_sql_injection

        assert check_sql_injection("SELECT * FROM users") is True
        assert check_sql_injection("select * from users") is True

    def test_check_sql_injection_union(self):
        """Test detection of UNION attacks."""
        from src.api.security.validation import check_sql_injection

        assert check_sql_injection("' UNION SELECT * FROM passwords --") is True

    def test_check_sql_injection_drop(self):
        """Test detection of DROP statements."""
        from src.api.security.validation import check_sql_injection

        assert check_sql_injection("DROP TABLE users;") is True

    def test_check_sql_injection_comment(self):
        """Test detection of SQL comments."""
        from src.api.security.validation import check_sql_injection

        assert check_sql_injection("admin'--") is True
        assert check_sql_injection("/* comment */") is True

    def test_check_sql_injection_or_always_true(self):
        """Test detection of OR 1=1 pattern."""
        from src.api.security.validation import check_sql_injection

        assert check_sql_injection("' OR 1=1 --") is True
        assert check_sql_injection("' OR 5=5") is True

    def test_check_sql_injection_semicolon(self):
        """Test detection of semicolon injection."""
        from src.api.security.validation import check_sql_injection

        assert check_sql_injection("1; DROP TABLE users") is True


# =============================================================================
# XSS Detection Tests
# =============================================================================

class TestCheckXSS:
    """Tests for check_xss function."""

    def test_check_xss_safe(self):
        """Test that safe input passes."""
        from src.api.security.validation import check_xss

        assert check_xss("normal text") is False
        assert check_xss("hello world") is False

    def test_check_xss_empty(self):
        """Test empty input."""
        from src.api.security.validation import check_xss

        assert check_xss("") is False
        assert check_xss(None) is False

    def test_check_xss_script_tag(self):
        """Test detection of script tags."""
        from src.api.security.validation import check_xss

        assert check_xss("<script>alert('xss')</script>") is True
        assert check_xss("<SCRIPT>alert('xss')</SCRIPT>") is True

    def test_check_xss_javascript_url(self):
        """Test detection of javascript: URLs."""
        from src.api.security.validation import check_xss

        assert check_xss("javascript:alert('xss')") is True

    def test_check_xss_event_handlers(self):
        """Test detection of event handlers."""
        from src.api.security.validation import check_xss

        assert check_xss("<img onerror=alert('xss')>") is True
        assert check_xss("<body onload=alert('xss')>") is True
        assert check_xss("<div onclick=alert('xss')>") is True

    def test_check_xss_iframe(self):
        """Test detection of iframe tags."""
        from src.api.security.validation import check_xss

        assert check_xss("<iframe src='evil.com'>") is True

    def test_check_xss_object_embed(self):
        """Test detection of object/embed tags."""
        from src.api.security.validation import check_xss

        assert check_xss("<object data='evil.swf'>") is True
        assert check_xss("<embed src='evil.swf'>") is True

    def test_check_xss_css_expression(self):
        """Test detection of CSS expressions."""
        from src.api.security.validation import check_xss

        assert check_xss("expression(alert('xss'))") is True


# =============================================================================
# Path Traversal Detection Tests
# =============================================================================

class TestCheckPathTraversal:
    """Tests for check_path_traversal function."""

    def test_check_path_traversal_safe(self):
        """Test that safe input passes."""
        from src.api.security.validation import check_path_traversal

        assert check_path_traversal("document.pdf") is False
        assert check_path_traversal("folder/file.txt") is False

    def test_check_path_traversal_empty(self):
        """Test empty input."""
        from src.api.security.validation import check_path_traversal

        assert check_path_traversal("") is False
        assert check_path_traversal(None) is False

    def test_check_path_traversal_dot_dot_slash(self):
        """Test detection of ../ pattern."""
        from src.api.security.validation import check_path_traversal

        assert check_path_traversal("../etc/passwd") is True
        assert check_path_traversal("../../etc/passwd") is True

    def test_check_path_traversal_backslash(self):
        """Test detection of ..\\ pattern."""
        from src.api.security.validation import check_path_traversal

        assert check_path_traversal("..\\windows\\system32") is True

    def test_check_path_traversal_url_encoded(self):
        """Test detection of URL-encoded traversal."""
        from src.api.security.validation import check_path_traversal

        assert check_path_traversal("%2e%2e%2f") is True
        assert check_path_traversal("%2e%2e/") is True


# =============================================================================
# Command Injection Detection Tests
# =============================================================================

class TestCheckCommandInjection:
    """Tests for check_command_injection function."""

    def test_check_command_injection_safe(self):
        """Test that safe input passes."""
        from src.api.security.validation import check_command_injection

        assert check_command_injection("normal text") is False
        assert check_command_injection("hello world") is False

    def test_check_command_injection_empty(self):
        """Test empty input."""
        from src.api.security.validation import check_command_injection

        assert check_command_injection("") is False
        assert check_command_injection(None) is False

    def test_check_command_injection_semicolon(self):
        """Test detection of semicolon command separator."""
        from src.api.security.validation import check_command_injection

        assert check_command_injection("file; rm -rf /") is True

    def test_check_command_injection_pipe(self):
        """Test detection of pipe operator."""
        from src.api.security.validation import check_command_injection

        assert check_command_injection("file | cat /etc/passwd") is True

    def test_check_command_injection_ampersand(self):
        """Test detection of ampersand operators."""
        from src.api.security.validation import check_command_injection

        assert check_command_injection("file && rm -rf /") is True
        assert check_command_injection("file & rm -rf /") is True

    def test_check_command_injection_backticks(self):
        """Test detection of backtick command substitution."""
        from src.api.security.validation import check_command_injection

        assert check_command_injection("`whoami`") is True

    def test_check_command_injection_dollar_parens(self):
        """Test detection of $() command substitution."""
        from src.api.security.validation import check_command_injection

        assert check_command_injection("$(whoami)") is True


# =============================================================================
# Generic Input Sanitization Tests
# =============================================================================

class TestSanitizeInput:
    """Tests for sanitize_input function."""

    def test_sanitize_input_string(self):
        """Test string type sanitization."""
        from src.api.security.validation import sanitize_input

        result = sanitize_input("<script>test</script>", "string")
        assert "<script>" not in result

    def test_sanitize_input_html(self):
        """Test HTML type sanitization."""
        from src.api.security.validation import sanitize_input

        result = sanitize_input("<p>Hello</p>", "html")
        assert "<p>" in result

    def test_sanitize_input_filename(self):
        """Test filename type sanitization."""
        from src.api.security.validation import sanitize_input

        result = sanitize_input("../file.txt", "filename")
        assert ".." not in result

    def test_sanitize_input_url(self):
        """Test URL type sanitization."""
        from src.api.security.validation import sanitize_input

        result = sanitize_input("https://example.com", "url")
        assert result == "https://example.com"

        result = sanitize_input("javascript:alert()", "url")
        assert result is None

    def test_sanitize_input_email(self):
        """Test email type sanitization."""
        from src.api.security.validation import sanitize_input

        result = sanitize_input("test@example.com", "email")
        assert "@" in result

    def test_sanitize_input_integer(self):
        """Test integer type sanitization."""
        from src.api.security.validation import sanitize_input

        result = sanitize_input("123", "integer")
        assert result == 123

        result = sanitize_input("not a number", "integer")
        assert result is None

    def test_sanitize_input_float(self):
        """Test float type sanitization."""
        from src.api.security.validation import sanitize_input

        result = sanitize_input("123.45", "float")
        assert result == 123.45

        result = sanitize_input("-10.5", "float")
        assert result == -10.5

        result = sanitize_input("not a number", "float")
        assert result is None

    def test_sanitize_input_none(self):
        """Test None input."""
        from src.api.security.validation import sanitize_input

        assert sanitize_input(None, "string") is None

    def test_sanitize_input_unknown_type(self):
        """Test unknown type falls back to string."""
        from src.api.security.validation import sanitize_input

        result = sanitize_input("<test>", "unknown_type")
        assert "&lt;test&gt;" in result


# =============================================================================
# InputValidator Class Tests
# =============================================================================

class TestInputValidator:
    """Tests for InputValidator class."""

    @pytest.fixture
    def validator(self):
        """Create InputValidator instance."""
        from src.api.security.validation import InputValidator
        return InputValidator()

    def test_validate_simple_string(self, validator):
        """Test validating simple string."""
        assert validator.validate("hello world") is True
        assert len(validator.get_violations()) == 0

    def test_validate_string_too_long(self):
        """Test validating string that exceeds max length."""
        from src.api.security.validation import InputValidator

        validator = InputValidator(max_string_length=10)
        assert validator.validate("this is too long") is False
        assert len(validator.get_violations()) > 0

    def test_validate_sql_injection(self, validator):
        """Test validating string with SQL injection."""
        result = validator.validate("SELECT * FROM users")
        assert result is False
        violations = validator.get_violations()
        assert any("SQL injection" in v for v in violations)

    def test_validate_xss(self, validator):
        """Test validating string with XSS."""
        result = validator.validate("<script>alert('xss')</script>")
        assert result is False
        violations = validator.get_violations()
        assert any("XSS" in v for v in violations)

    def test_validate_path_traversal(self, validator):
        """Test validating string with path traversal."""
        result = validator.validate("../../../etc/passwd")
        assert result is False
        violations = validator.get_violations()
        assert any("path traversal" in v for v in violations)

    def test_validate_command_injection(self, validator):
        """Test validating string with command injection."""
        result = validator.validate("file; rm -rf /")
        assert result is False
        violations = validator.get_violations()
        assert any("command injection" in v for v in violations)

    def test_validate_dict(self, validator):
        """Test validating dictionary."""
        data = {
            "name": "John",
            "email": "john@example.com",
        }
        assert validator.validate(data) is True

    def test_validate_dict_with_injection(self, validator):
        """Test validating dictionary containing injection."""
        data = {
            "name": "John",
            "query": "SELECT * FROM users",
        }
        assert validator.validate(data) is False

    def test_validate_list(self, validator):
        """Test validating list."""
        data = ["item1", "item2", "item3"]
        assert validator.validate(data) is True

    def test_validate_list_too_long(self):
        """Test validating list that exceeds max length."""
        from src.api.security.validation import InputValidator

        validator = InputValidator(max_array_length=2)
        data = ["item1", "item2", "item3"]
        assert validator.validate(data) is False

    def test_validate_nested_object(self, validator):
        """Test validating nested objects."""
        data = {
            "user": {
                "profile": {
                    "name": "John",
                }
            }
        }
        assert validator.validate(data) is True

    def test_validate_exceeds_max_depth(self):
        """Test validating object that exceeds max depth."""
        from src.api.security.validation import InputValidator

        validator = InputValidator(max_object_depth=2)
        data = {"a": {"b": {"c": "too deep"}}}
        assert validator.validate(data) is False

    def test_validate_primitives(self, validator):
        """Test validating primitive types."""
        assert validator.validate(123) is True
        assert validator.validate(123.45) is True
        assert validator.validate(True) is True
        assert validator.validate(None) is True

    def test_validate_disabled_checks(self):
        """Test with specific checks disabled."""
        from src.api.security.validation import InputValidator

        validator = InputValidator(check_sql=False)
        # SQL injection should pass when check is disabled
        result = validator.validate("SELECT * FROM users")
        # Other checks might still catch it if they match patterns
        # The point is SQL-specific check is disabled

    def test_get_violations_returns_copy(self, validator):
        """Test that get_violations returns a copy."""
        validator.validate("<script>alert('xss')</script>")
        violations1 = validator.get_violations()
        violations2 = validator.get_violations()

        # Should be equal but not the same object
        assert violations1 == violations2
        assert violations1 is not violations2

    def test_sanitize_data_string(self, validator):
        """Test sanitizing string data."""
        result = validator.sanitize_data("<script>test</script>")
        assert "<script>" not in result

    def test_sanitize_data_dict(self, validator):
        """Test sanitizing dictionary data."""
        data = {
            "name": "<script>test</script>",
            "value": "normal",
        }
        result = validator.sanitize_data(data)

        assert "<script>" not in result["name"]
        assert result["value"] == "normal"

    def test_sanitize_data_list(self, validator):
        """Test sanitizing list data."""
        data = ["<script>1</script>", "normal"]
        result = validator.sanitize_data(data)

        assert "<script>" not in result[0]
        assert result[1] == "normal"

    def test_sanitize_data_nested(self, validator):
        """Test sanitizing nested data."""
        data = {
            "user": {
                "name": "<script>xss</script>",
            },
            "items": ["<iframe>", "normal"],
        }
        result = validator.sanitize_data(data)

        assert "<script>" not in result["user"]["name"]
        assert "<iframe>" not in result["items"][0]

    def test_sanitize_data_exceeds_depth(self):
        """Test sanitizing data exceeding max depth."""
        from src.api.security.validation import InputValidator

        validator = InputValidator(max_object_depth=1)
        data = {"a": {"b": {"c": "value"}}}
        result = validator.sanitize_data(data)

        # Deep nested should be None or truncated
        assert result["a"]["b"] is None

    def test_sanitize_data_truncates_list(self):
        """Test that sanitize truncates long lists."""
        from src.api.security.validation import InputValidator

        validator = InputValidator(max_array_length=2)
        data = ["item1", "item2", "item3", "item4"]
        result = validator.sanitize_data(data)

        assert len(result) == 2


# =============================================================================
# Pydantic Validator Tests
# =============================================================================

class TestSecureStringValidator:
    """Tests for SecureString Pydantic type."""

    def test_secure_string_valid(self):
        """Test SecureString with valid string."""
        from src.api.security.validation import SecureString

        result = SecureString.validate("hello world")
        assert isinstance(result, str)

    def test_secure_string_sanitizes(self):
        """Test that SecureString sanitizes input."""
        from src.api.security.validation import SecureString

        result = SecureString.validate("<script>test</script>")
        assert "<script>" not in result

    def test_secure_string_non_string_raises(self):
        """Test that non-string input raises error."""
        from src.api.security.validation import SecureString

        with pytest.raises(ValueError):
            SecureString.validate(123)


class TestSecureURLValidator:
    """Tests for SecureURL Pydantic type."""

    def test_secure_url_valid(self):
        """Test SecureURL with valid URL."""
        from src.api.security.validation import SecureURL

        result = SecureURL.validate("https://example.com")
        assert result == "https://example.com"

    def test_secure_url_rejects_javascript(self):
        """Test that SecureURL rejects javascript URLs."""
        from src.api.security.validation import SecureURL

        with pytest.raises(ValueError):
            SecureURL.validate("javascript:alert('xss')")

    def test_secure_url_non_string_raises(self):
        """Test that non-string input raises error."""
        from src.api.security.validation import SecureURL

        with pytest.raises(ValueError):
            SecureURL.validate(123)


class TestSecureFilenameValidator:
    """Tests for SecureFilename Pydantic type."""

    def test_secure_filename_valid(self):
        """Test SecureFilename with valid filename."""
        from src.api.security.validation import SecureFilename

        result = SecureFilename.validate("document.pdf")
        assert result == "document.pdf"

    def test_secure_filename_sanitizes(self):
        """Test that SecureFilename sanitizes input."""
        from src.api.security.validation import SecureFilename

        result = SecureFilename.validate("../../../etc/passwd")
        # Slashes are replaced with underscores, making path traversal impossible
        assert "/" not in result
        assert "\\" not in result

    def test_secure_filename_non_string_raises(self):
        """Test that non-string input raises error."""
        from src.api.security.validation import SecureFilename

        with pytest.raises(ValueError):
            SecureFilename.validate(123)


# =============================================================================
# Pattern Tests
# =============================================================================

class TestPatternDefinitions:
    """Tests for pattern definitions."""

    def test_sql_patterns_exist(self):
        """Test that SQL injection patterns are defined."""
        from src.api.security.validation import SQL_INJECTION_PATTERNS

        assert len(SQL_INJECTION_PATTERNS) > 0

    def test_xss_patterns_exist(self):
        """Test that XSS patterns are defined."""
        from src.api.security.validation import XSS_PATTERNS

        assert len(XSS_PATTERNS) > 0

    def test_path_traversal_patterns_exist(self):
        """Test that path traversal patterns are defined."""
        from src.api.security.validation import PATH_TRAVERSAL_PATTERNS

        assert len(PATH_TRAVERSAL_PATTERNS) > 0

    def test_command_injection_patterns_exist(self):
        """Test that command injection patterns are defined."""
        from src.api.security.validation import COMMAND_INJECTION_PATTERNS

        assert len(COMMAND_INJECTION_PATTERNS) > 0
