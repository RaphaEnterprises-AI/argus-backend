"""Base template class for test export."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TestStep:
    """A test step for export."""

    action: str
    target: str | None = None
    value: str | None = None
    timeout: int | None = None
    description: str | None = None


@dataclass
class TestAssertion:
    """A test assertion for export."""

    type: str
    target: str | None = None
    expected: str | None = None


@dataclass
class TestSpec:
    """Test specification for export."""

    id: str
    name: str
    description: str = ""
    steps: list[TestStep] = None
    assertions: list[TestAssertion] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.assertions is None:
            self.assertions = []

    @classmethod
    def from_dict(cls, data: dict) -> "TestSpec":
        """Create TestSpec from dictionary."""
        steps = [
            TestStep(**s) if isinstance(s, dict) else s
            for s in data.get("steps", [])
        ]
        assertions = [
            TestAssertion(**a) if isinstance(a, dict) else a
            for a in data.get("assertions", [])
        ]
        return cls(
            id=data.get("id", "test-001"),
            name=data.get("name", "Test"),
            description=data.get("description", ""),
            steps=steps,
            assertions=assertions,
        )


class BaseTemplate(ABC):
    """Base class for export templates.

    Each language/framework combination implements this class to generate
    test code in their specific syntax.
    """

    # Override these in subclasses
    language: str = "unknown"
    framework: str = "unknown"
    file_extension: str = ".txt"
    indent: str = "    "

    def __init__(self, config: Any = None):
        """Initialize template with optional config."""
        self.config = config or {}

    @abstractmethod
    def generate_imports(self, test_spec: TestSpec) -> str:
        """Generate import statements."""
        pass

    @abstractmethod
    def generate_class_header(self, test_spec: TestSpec) -> str:
        """Generate class/test header."""
        pass

    @abstractmethod
    def generate_step_code(self, step: TestStep, index: int) -> str:
        """Generate code for a single step."""
        pass

    @abstractmethod
    def generate_assertion_code(self, assertion: TestAssertion) -> str:
        """Generate code for a single assertion."""
        pass

    @abstractmethod
    def generate_class_footer(self) -> str:
        """Generate class/test footer."""
        pass

    def generate(self, test_spec: TestSpec | dict) -> str:
        """Generate complete test code.

        Args:
            test_spec: Test specification (dict or TestSpec)

        Returns:
            Generated test code
        """
        if isinstance(test_spec, dict):
            test_spec = TestSpec.from_dict(test_spec)

        parts = []

        # Header/imports
        parts.append(self.generate_imports(test_spec))
        parts.append("")

        # Class header
        parts.append(self.generate_class_header(test_spec))

        # Steps
        for idx, step in enumerate(test_spec.steps):
            step_code = self.generate_step_code(step, idx)
            if step_code:
                parts.append(step_code)

        # Assertions
        if test_spec.assertions:
            parts.append("")
            parts.append(self._generate_assertions_header())
            for assertion in test_spec.assertions:
                assertion_code = self.generate_assertion_code(assertion)
                if assertion_code:
                    parts.append(assertion_code)

        # Footer
        parts.append(self.generate_class_footer())

        # Attribution
        parts.append(self._generate_attribution())

        return "\n".join(parts)

    def _generate_assertions_header(self) -> str:
        """Generate header for assertions section."""
        return f"{self.indent}{self.indent}# Assertions"

    def _generate_attribution(self) -> str:
        """Generate Argus attribution comment."""
        comment_prefix = self._get_comment_prefix()
        return f"\n{comment_prefix} Generated with Argus E2E Testing Agent"

    def _get_comment_prefix(self) -> str:
        """Get comment prefix for this language."""
        if self.language in ("python", "ruby"):
            return "#"
        elif self.language in ("java", "typescript", "csharp", "go"):
            return "//"
        return "#"

    def sanitize_name(self, name: str) -> str:
        """Convert name to valid identifier."""
        # Remove special characters
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        # Ensure starts with letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = "_" + sanitized
        return sanitized.lower()

    def to_camel_case(self, name: str) -> str:
        """Convert name to camelCase."""
        words = re.sub(r"[^a-zA-Z0-9]", " ", name).split()
        if not words:
            return "test"
        return words[0].lower() + "".join(w.title() for w in words[1:])

    def to_pascal_case(self, name: str) -> str:
        """Convert name to PascalCase."""
        words = re.sub(r"[^a-zA-Z0-9]", " ", name).split()
        return "".join(w.title() for w in words) if words else "Test"

    def to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        # Insert underscore before capitals
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        # Replace non-alphanumeric with underscore
        s3 = re.sub(r"[^a-zA-Z0-9]", "_", s2)
        # Clean up multiple underscores
        return re.sub(r"_+", "_", s3).lower().strip("_")

    def escape_string(self, value: str) -> str:
        """Escape string for code generation."""
        if value is None:
            return ""
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def format_comment(self, text: str, inline: bool = False) -> str:
        """Format a comment for this language."""
        prefix = self._get_comment_prefix()
        if inline:
            return f" {prefix} {text}"
        return f"{prefix} {text}"
