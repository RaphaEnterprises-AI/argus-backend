"""Code formatter for various languages."""


from ..models import SupportedLanguage


class CodeFormatter:
    """Formats generated code according to language conventions."""

    def __init__(self, language: SupportedLanguage):
        """Initialize formatter for a specific language."""
        self.language = language

    def format_code(self, code: str) -> str:
        """Format code according to language conventions.

        Args:
            code: Raw generated code

        Returns:
            Formatted code
        """
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in code.split("\n")]

        # Remove excessive blank lines (max 2 consecutive)
        formatted_lines = []
        blank_count = 0
        for line in lines:
            if line == "":
                blank_count += 1
                if blank_count <= 2:
                    formatted_lines.append(line)
            else:
                blank_count = 0
                formatted_lines.append(line)

        # Ensure file ends with newline
        result = "\n".join(formatted_lines)
        if not result.endswith("\n"):
            result += "\n"

        return result

    def get_indent(self) -> str:
        """Get indent string for this language."""
        if self.language in (SupportedLanguage.PYTHON, SupportedLanguage.RUBY):
            return "    "  # 4 spaces
        elif self.language == SupportedLanguage.GO:
            return "\t"  # Tab
        else:
            return "    "  # 4 spaces default

    def format_string_literal(self, value: str, single_quotes: bool | None = None) -> str:
        """Format a string literal for this language.

        Args:
            value: String value
            single_quotes: Force single quotes (default: language convention)

        Returns:
            Formatted string literal
        """
        # Determine quote style
        if single_quotes is None:
            single_quotes = self.language in (
                SupportedLanguage.TYPESCRIPT,
                SupportedLanguage.RUBY,
            )

        # Escape quotes and special chars
        escaped = value.replace("\\", "\\\\")
        if single_quotes:
            escaped = escaped.replace("'", "\\'")
            return f"'{escaped}'"
        else:
            escaped = escaped.replace('"', '\\"')
            return f'"{escaped}"'

    def format_multiline_string(self, value: str) -> str:
        """Format a multiline string literal.

        Args:
            value: String value with potential newlines

        Returns:
            Formatted multiline string
        """
        if self.language == SupportedLanguage.PYTHON:
            return f'"""{value}"""'
        elif self.language == SupportedLanguage.JAVA:
            # Java text blocks (Java 15+)
            return f'"""\n{value}"""'
        elif self.language == SupportedLanguage.GO:
            return f'`{value}`'
        else:
            # Fallback: escape newlines
            escaped = value.replace("\n", "\\n")
            return f'"{escaped}"'

    def format_comment(self, text: str, doc_comment: bool = False) -> str:
        """Format a comment for this language.

        Args:
            text: Comment text
            doc_comment: Whether this is a documentation comment

        Returns:
            Formatted comment
        """
        if self.language == SupportedLanguage.PYTHON:
            if doc_comment:
                return f'"""{text}"""'
            return f"# {text}"

        elif self.language == SupportedLanguage.RUBY:
            if doc_comment:
                return f"# {text}"
            return f"# {text}"

        elif self.language == SupportedLanguage.JAVA:
            if doc_comment:
                return f"/** {text} */"
            return f"// {text}"

        elif self.language == SupportedLanguage.CSHARP:
            if doc_comment:
                return f"/// <summary>{text}</summary>"
            return f"// {text}"

        elif self.language == SupportedLanguage.GO:
            if doc_comment:
                return f"// {text}"
            return f"// {text}"

        elif self.language == SupportedLanguage.TYPESCRIPT:
            if doc_comment:
                return f"/** {text} */"
            return f"// {text}"

        return f"// {text}"

    def format_block_comment(self, lines: list[str], doc_comment: bool = False) -> str:
        """Format a multi-line block comment.

        Args:
            lines: Comment lines
            doc_comment: Whether this is a documentation comment

        Returns:
            Formatted block comment
        """
        if self.language == SupportedLanguage.PYTHON:
            if doc_comment:
                content = "\n".join(lines)
                return f'"""\n{content}\n"""'
            return "\n".join(f"# {line}" for line in lines)

        elif self.language == SupportedLanguage.RUBY:
            return "\n".join(f"# {line}" for line in lines)

        elif self.language in (
            SupportedLanguage.JAVA,
            SupportedLanguage.CSHARP,
            SupportedLanguage.TYPESCRIPT,
            SupportedLanguage.GO,
        ):
            if doc_comment:
                formatted = ["/**"]
                formatted.extend(f" * {line}" for line in lines)
                formatted.append(" */")
                return "\n".join(formatted)
            formatted = ["/*"]
            formatted.extend(f" * {line}" for line in lines)
            formatted.append(" */")
            return "\n".join(formatted)

        return "\n".join(f"// {line}" for line in lines)
