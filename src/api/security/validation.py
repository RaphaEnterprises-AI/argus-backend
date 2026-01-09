"""Input validation and sanitization for security.

Provides:
- XSS prevention
- SQL injection prevention
- Path traversal prevention
- Input size limits
- Content type validation
"""

import re
import html
import unicodedata
from typing import Any, Optional, List, Dict, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator
import structlog

logger = structlog.get_logger()


# =============================================================================
# Dangerous Patterns
# =============================================================================

# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER|EXEC|EXECUTE)\b)",
    r"(--)|(;)|(\/\*)|(\*\/)",
    r"(\bOR\b\s+\d+\s*=\s*\d+)",
    r"(\bAND\b\s+\d+\s*=\s*\d+)",
    r"(\'|\"|;|--)",
]

# XSS patterns
XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe[^>]*>",
    r"<object[^>]*>",
    r"<embed[^>]*>",
    r"<link[^>]*>",
    r"<meta[^>]*>",
    r"expression\s*\(",
    r"url\s*\(",
]

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e%2f",
    r"%2e%2e/",
    r"\.%2e/",
    r"%2e\./",
]

# Command injection patterns
COMMAND_INJECTION_PATTERNS = [
    r"[;&|`$]",
    r"\$\(",
    r"`.*`",
    r"\|\|",
    r"&&",
]


# =============================================================================
# Sanitization Functions
# =============================================================================

def sanitize_string(value: str, max_length: int = 10000) -> str:
    """Sanitize a string input.

    - Escapes HTML entities
    - Removes control characters
    - Truncates to max length
    - Normalizes unicode
    """
    if not value:
        return value

    # Truncate
    if len(value) > max_length:
        value = value[:max_length]

    # Normalize unicode
    value = unicodedata.normalize("NFKC", value)

    # Remove control characters (except newlines and tabs)
    value = "".join(
        char for char in value
        if unicodedata.category(char) != "Cc" or char in "\n\t\r"
    )

    # Escape HTML
    value = html.escape(value)

    return value


def sanitize_html(value: str, allowed_tags: List[str] = None) -> str:
    """Sanitize HTML content, allowing only specific tags."""
    if not value:
        return value

    if allowed_tags is None:
        allowed_tags = ["p", "br", "b", "i", "u", "strong", "em", "a", "ul", "ol", "li"]

    # Simple approach - escape everything then unescape allowed tags
    value = html.escape(value)

    for tag in allowed_tags:
        # Unescape opening tags
        value = re.sub(
            rf"&lt;({tag})(\s[^&]*)?&gt;",
            rf"<\1\2>",
            value,
            flags=re.IGNORECASE,
        )
        # Unescape closing tags
        value = re.sub(
            rf"&lt;/({tag})&gt;",
            rf"</\1>",
            value,
            flags=re.IGNORECASE,
        )

    return value


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal."""
    if not filename:
        return filename

    # Remove path separators
    filename = filename.replace("/", "_").replace("\\", "_")

    # Remove null bytes
    filename = filename.replace("\x00", "")

    # Remove leading/trailing dots and spaces
    filename = filename.strip(". ")

    # Remove unicode tricks
    filename = unicodedata.normalize("NFKC", filename)

    # Only allow safe characters
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)

    return filename


def sanitize_url(url: str) -> Optional[str]:
    """Sanitize and validate a URL."""
    if not url:
        return None

    try:
        parsed = urlparse(url)

        # Only allow http and https
        if parsed.scheme not in ("http", "https"):
            return None

        # Check for javascript: in other parts
        url_lower = url.lower()
        if "javascript:" in url_lower:
            return None

        # Reconstruct URL
        return url

    except Exception:
        return None


def sanitize_input(value: Any, field_type: str = "string") -> Any:
    """Generic input sanitization based on field type."""
    if value is None:
        return None

    sanitizers = {
        "string": lambda v: sanitize_string(str(v)),
        "html": lambda v: sanitize_html(str(v)),
        "filename": lambda v: sanitize_filename(str(v)),
        "url": lambda v: sanitize_url(str(v)),
        "email": lambda v: sanitize_string(str(v), max_length=254),
        "integer": lambda v: int(v) if str(v).isdigit() else None,
        "float": lambda v: float(v) if re.match(r"^-?\d+\.?\d*$", str(v)) else None,
    }

    sanitizer = sanitizers.get(field_type, sanitizers["string"])
    return sanitizer(value)


# =============================================================================
# Validation Functions
# =============================================================================

def check_sql_injection(value: str) -> bool:
    """Check if value contains SQL injection patterns."""
    if not value:
        return False

    value_upper = value.upper()
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, value_upper, re.IGNORECASE):
            return True
    return False


def check_xss(value: str) -> bool:
    """Check if value contains XSS patterns."""
    if not value:
        return False

    value_lower = value.lower()
    for pattern in XSS_PATTERNS:
        if re.search(pattern, value_lower, re.IGNORECASE):
            return True
    return False


def check_path_traversal(value: str) -> bool:
    """Check if value contains path traversal patterns."""
    if not value:
        return False

    value_lower = value.lower()
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if re.search(pattern, value_lower, re.IGNORECASE):
            return True
    return False


def check_command_injection(value: str) -> bool:
    """Check if value contains command injection patterns."""
    if not value:
        return False

    for pattern in COMMAND_INJECTION_PATTERNS:
        if re.search(pattern, value):
            return True
    return False


# =============================================================================
# Input Validator Class
# =============================================================================

class InputValidator:
    """Comprehensive input validator for security."""

    def __init__(
        self,
        max_string_length: int = 10000,
        max_array_length: int = 1000,
        max_object_depth: int = 10,
        check_sql: bool = True,
        check_xss: bool = True,
        check_path: bool = True,
        check_command: bool = True,
    ):
        self.max_string_length = max_string_length
        self.max_array_length = max_array_length
        self.max_object_depth = max_object_depth
        self.check_sql = check_sql
        self.check_xss = check_xss
        self.check_path = check_path
        self.check_command = check_command
        self._violations: List[str] = []

    def validate(self, data: Any, path: str = "root", depth: int = 0) -> bool:
        """Validate input data recursively."""
        self._violations = []

        if depth > self.max_object_depth:
            self._violations.append(f"{path}: Maximum object depth exceeded")
            return False

        if isinstance(data, str):
            return self._validate_string(data, path)
        elif isinstance(data, dict):
            return self._validate_dict(data, path, depth)
        elif isinstance(data, list):
            return self._validate_list(data, path, depth)
        elif isinstance(data, (int, float, bool, type(None))):
            return True
        else:
            return True

    def _validate_string(self, value: str, path: str) -> bool:
        """Validate a string value."""
        valid = True

        if len(value) > self.max_string_length:
            self._violations.append(f"{path}: String exceeds maximum length")
            valid = False

        if self.check_sql and check_sql_injection(value):
            self._violations.append(f"{path}: Potential SQL injection detected")
            valid = False

        if self.check_xss and check_xss(value):
            self._violations.append(f"{path}: Potential XSS detected")
            valid = False

        if self.check_path and check_path_traversal(value):
            self._violations.append(f"{path}: Potential path traversal detected")
            valid = False

        if self.check_command and check_command_injection(value):
            self._violations.append(f"{path}: Potential command injection detected")
            valid = False

        return valid

    def _validate_dict(self, data: dict, path: str, depth: int) -> bool:
        """Validate a dictionary."""
        valid = True

        for key, value in data.items():
            key_path = f"{path}.{key}"

            # Validate key
            if not self._validate_string(str(key), f"{key_path}[key]"):
                valid = False

            # Validate value
            if not self.validate(value, key_path, depth + 1):
                valid = False

        return valid

    def _validate_list(self, data: list, path: str, depth: int) -> bool:
        """Validate a list."""
        if len(data) > self.max_array_length:
            self._violations.append(f"{path}: Array exceeds maximum length")
            return False

        valid = True
        for i, item in enumerate(data):
            if not self.validate(item, f"{path}[{i}]", depth + 1):
                valid = False

        return valid

    def get_violations(self) -> List[str]:
        """Get list of validation violations."""
        return self._violations.copy()

    def sanitize_data(self, data: Any, depth: int = 0) -> Any:
        """Sanitize input data recursively."""
        if depth > self.max_object_depth:
            return None

        if isinstance(data, str):
            return sanitize_string(data, self.max_string_length)
        elif isinstance(data, dict):
            return {
                sanitize_string(str(k)): self.sanitize_data(v, depth + 1)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [
                self.sanitize_data(item, depth + 1)
                for item in data[:self.max_array_length]
            ]
        else:
            return data


# =============================================================================
# Pydantic Validators
# =============================================================================

class SecureString(str):
    """A string type that is automatically sanitized."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise ValueError("String required")
        return sanitize_string(v)


class SecureURL(str):
    """A URL type that is validated and sanitized."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise ValueError("String required")
        sanitized = sanitize_url(v)
        if sanitized is None:
            raise ValueError("Invalid URL")
        return sanitized


class SecureFilename(str):
    """A filename type that is sanitized."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise ValueError("String required")
        return sanitize_filename(v)
