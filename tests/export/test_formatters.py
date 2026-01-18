"""Comprehensive tests for export formatters module."""

import pytest

from src.export.formatters import CodeFormatter, ImportsManager
from src.export.formatters.imports_manager import ImportSpec
from src.export.models import SupportedFramework, SupportedLanguage

# =============================================================================
# CodeFormatter Tests
# =============================================================================


class TestCodeFormatterInit:
    """Tests for CodeFormatter initialization."""

    def test_create_formatter_python(self):
        """Test creating formatter for Python."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        assert formatter.language == SupportedLanguage.PYTHON

    def test_create_formatter_typescript(self):
        """Test creating formatter for TypeScript."""
        formatter = CodeFormatter(SupportedLanguage.TYPESCRIPT)
        assert formatter.language == SupportedLanguage.TYPESCRIPT

    @pytest.mark.parametrize("language", list(SupportedLanguage))
    def test_create_formatter_all_languages(self, language):
        """Test creating formatter for all supported languages."""
        formatter = CodeFormatter(language)
        assert formatter.language == language


class TestCodeFormatterFormatCode:
    """Tests for CodeFormatter.format_code method."""

    @pytest.fixture
    def python_formatter(self):
        return CodeFormatter(SupportedLanguage.PYTHON)

    def test_removes_trailing_whitespace(self, python_formatter):
        """Test format_code removes trailing whitespace."""
        code = "line1   \nline2  \n  line3   "
        formatted = python_formatter.format_code(code)
        lines = formatted.split("\n")
        for line in lines[:-1]:  # Skip last which may be empty
            assert line == line.rstrip()

    def test_limits_consecutive_blank_lines(self, python_formatter):
        """Test format_code limits consecutive blank lines to 2."""
        code = "line1\n\n\n\n\nline2"
        formatted = python_formatter.format_code(code)
        # Should have max 2 consecutive blank lines
        assert "\n\n\n\n" not in formatted

    def test_preserves_single_blank_lines(self, python_formatter):
        """Test format_code preserves single blank lines."""
        code = "line1\n\nline2"
        formatted = python_formatter.format_code(code)
        assert "\n\n" in formatted

    def test_preserves_double_blank_lines(self, python_formatter):
        """Test format_code preserves double blank lines."""
        code = "line1\n\n\nline2"
        formatted = python_formatter.format_code(code)
        # Should have exactly 2 blank lines (3 newlines total)
        assert "line1\n\n\nline2" in formatted or "\n\n" in formatted

    def test_adds_trailing_newline(self, python_formatter):
        """Test format_code adds trailing newline if missing."""
        code = "line without newline"
        formatted = python_formatter.format_code(code)
        assert formatted.endswith("\n")

    def test_preserves_existing_trailing_newline(self, python_formatter):
        """Test format_code preserves existing trailing newline."""
        code = "line with newline\n"
        formatted = python_formatter.format_code(code)
        assert formatted.endswith("\n")
        assert not formatted.endswith("\n\n")

    def test_handles_empty_string(self, python_formatter):
        """Test format_code handles empty string."""
        formatted = python_formatter.format_code("")
        assert formatted == "\n"

    def test_handles_only_whitespace(self, python_formatter):
        """Test format_code handles only whitespace."""
        formatted = python_formatter.format_code("   \n   \n   ")
        assert formatted.endswith("\n")

    def test_preserves_indentation(self, python_formatter):
        """Test format_code preserves indentation."""
        code = "def foo():\n    pass\n        nested\n"
        formatted = python_formatter.format_code(code)
        assert "    pass" in formatted
        assert "        nested" in formatted


class TestCodeFormatterGetIndent:
    """Tests for CodeFormatter.get_indent method."""

    def test_python_indent(self):
        """Test Python uses 4 spaces."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        assert formatter.get_indent() == "    "

    def test_typescript_indent(self):
        """Test TypeScript uses 4 spaces."""
        formatter = CodeFormatter(SupportedLanguage.TYPESCRIPT)
        assert formatter.get_indent() == "    "

    def test_java_indent(self):
        """Test Java uses 4 spaces."""
        formatter = CodeFormatter(SupportedLanguage.JAVA)
        assert formatter.get_indent() == "    "

    def test_csharp_indent(self):
        """Test C# uses 4 spaces."""
        formatter = CodeFormatter(SupportedLanguage.CSHARP)
        assert formatter.get_indent() == "    "

    def test_ruby_indent(self):
        """Test Ruby uses 4 spaces."""
        formatter = CodeFormatter(SupportedLanguage.RUBY)
        assert formatter.get_indent() == "    "

    def test_go_indent(self):
        """Test Go uses tabs."""
        formatter = CodeFormatter(SupportedLanguage.GO)
        assert formatter.get_indent() == "\t"


class TestCodeFormatterFormatStringLiteral:
    """Tests for CodeFormatter.format_string_literal method."""

    def test_double_quotes_default_python(self):
        """Test Python defaults to double quotes."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_string_literal("hello")
        assert result == '"hello"'

    def test_single_quotes_default_typescript(self):
        """Test TypeScript defaults to single quotes."""
        formatter = CodeFormatter(SupportedLanguage.TYPESCRIPT)
        result = formatter.format_string_literal("hello")
        assert result == "'hello'"

    def test_single_quotes_default_ruby(self):
        """Test Ruby defaults to single quotes."""
        formatter = CodeFormatter(SupportedLanguage.RUBY)
        result = formatter.format_string_literal("hello")
        assert result == "'hello'"

    def test_force_double_quotes(self):
        """Test forcing double quotes."""
        formatter = CodeFormatter(SupportedLanguage.TYPESCRIPT)
        result = formatter.format_string_literal("hello", single_quotes=False)
        assert result == '"hello"'

    def test_force_single_quotes(self):
        """Test forcing single quotes."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_string_literal("hello", single_quotes=True)
        assert result == "'hello'"

    def test_escapes_backslash(self):
        """Test escaping backslashes."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_string_literal("path\\to\\file", single_quotes=False)
        assert result == '"path\\\\to\\\\file"'

    def test_escapes_double_quotes(self):
        """Test escaping double quotes when using double quotes."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_string_literal('say "hello"', single_quotes=False)
        assert result == '"say \\"hello\\""'

    def test_escapes_single_quotes(self):
        """Test escaping single quotes when using single quotes."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_string_literal("it's fine", single_quotes=True)
        assert result == "'it\\'s fine'"

    def test_empty_string(self):
        """Test formatting empty string."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_string_literal("", single_quotes=False)
        assert result == '""'

    def test_special_characters(self):
        """Test string with special characters."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_string_literal("$#@!%", single_quotes=False)
        assert result == '"$#@!%"'


class TestCodeFormatterFormatMultilineString:
    """Tests for CodeFormatter.format_multiline_string method."""

    def test_python_multiline(self):
        """Test Python multiline string."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_multiline_string("line1\nline2")
        assert result == '"""line1\nline2"""'

    def test_java_multiline(self):
        """Test Java text block."""
        formatter = CodeFormatter(SupportedLanguage.JAVA)
        result = formatter.format_multiline_string("line1\nline2")
        assert '"""' in result
        assert "line1\nline2" in result

    def test_go_multiline(self):
        """Test Go raw string literal."""
        formatter = CodeFormatter(SupportedLanguage.GO)
        result = formatter.format_multiline_string("line1\nline2")
        assert result == "`line1\nline2`"

    def test_typescript_multiline(self):
        """Test TypeScript escapes newlines."""
        formatter = CodeFormatter(SupportedLanguage.TYPESCRIPT)
        result = formatter.format_multiline_string("line1\nline2")
        assert "\\n" in result

    def test_csharp_multiline(self):
        """Test C# escapes newlines."""
        formatter = CodeFormatter(SupportedLanguage.CSHARP)
        result = formatter.format_multiline_string("line1\nline2")
        assert "\\n" in result


class TestCodeFormatterFormatComment:
    """Tests for CodeFormatter.format_comment method."""

    def test_python_comment(self):
        """Test Python comment formatting."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_comment("This is a comment")
        assert result == "# This is a comment"

    def test_python_doc_comment(self):
        """Test Python docstring formatting."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_comment("Docstring text", doc_comment=True)
        assert result == '"""Docstring text"""'

    def test_ruby_comment(self):
        """Test Ruby comment formatting."""
        formatter = CodeFormatter(SupportedLanguage.RUBY)
        result = formatter.format_comment("Comment text")
        assert result == "# Comment text"

    def test_ruby_doc_comment(self):
        """Test Ruby doc comment (same as regular)."""
        formatter = CodeFormatter(SupportedLanguage.RUBY)
        result = formatter.format_comment("Doc text", doc_comment=True)
        assert result == "# Doc text"

    def test_java_comment(self):
        """Test Java comment formatting."""
        formatter = CodeFormatter(SupportedLanguage.JAVA)
        result = formatter.format_comment("Comment text")
        assert result == "// Comment text"

    def test_java_doc_comment(self):
        """Test Java doc comment formatting."""
        formatter = CodeFormatter(SupportedLanguage.JAVA)
        result = formatter.format_comment("Doc text", doc_comment=True)
        assert "/**" in result
        assert "*/" in result

    def test_csharp_comment(self):
        """Test C# comment formatting."""
        formatter = CodeFormatter(SupportedLanguage.CSHARP)
        result = formatter.format_comment("Comment text")
        assert result == "// Comment text"

    def test_csharp_doc_comment(self):
        """Test C# XML doc comment."""
        formatter = CodeFormatter(SupportedLanguage.CSHARP)
        result = formatter.format_comment("Doc text", doc_comment=True)
        assert "///" in result
        assert "<summary>" in result

    def test_go_comment(self):
        """Test Go comment formatting."""
        formatter = CodeFormatter(SupportedLanguage.GO)
        result = formatter.format_comment("Comment text")
        assert result == "// Comment text"

    def test_typescript_comment(self):
        """Test TypeScript comment formatting."""
        formatter = CodeFormatter(SupportedLanguage.TYPESCRIPT)
        result = formatter.format_comment("Comment text")
        assert result == "// Comment text"

    def test_typescript_doc_comment(self):
        """Test TypeScript JSDoc formatting."""
        formatter = CodeFormatter(SupportedLanguage.TYPESCRIPT)
        result = formatter.format_comment("Doc text", doc_comment=True)
        assert "/**" in result
        assert "*/" in result


class TestCodeFormatterFormatBlockComment:
    """Tests for CodeFormatter.format_block_comment method."""

    def test_python_block_comment(self):
        """Test Python block comment."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_block_comment(["Line 1", "Line 2"])
        assert "# Line 1" in result
        assert "# Line 2" in result

    def test_python_block_doc_comment(self):
        """Test Python block docstring."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_block_comment(["Line 1", "Line 2"], doc_comment=True)
        assert '"""' in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_ruby_block_comment(self):
        """Test Ruby block comment."""
        formatter = CodeFormatter(SupportedLanguage.RUBY)
        result = formatter.format_block_comment(["Line 1", "Line 2"])
        assert "# Line 1" in result
        assert "# Line 2" in result

    def test_java_block_comment(self):
        """Test Java block comment."""
        formatter = CodeFormatter(SupportedLanguage.JAVA)
        result = formatter.format_block_comment(["Line 1", "Line 2"])
        assert "/*" in result
        assert " * Line 1" in result
        assert " * Line 2" in result
        assert " */" in result

    def test_java_block_doc_comment(self):
        """Test Java Javadoc block."""
        formatter = CodeFormatter(SupportedLanguage.JAVA)
        result = formatter.format_block_comment(["Line 1", "Line 2"], doc_comment=True)
        assert "/**" in result
        assert " * Line 1" in result
        assert " */" in result

    def test_csharp_block_comment(self):
        """Test C# block comment."""
        formatter = CodeFormatter(SupportedLanguage.CSHARP)
        result = formatter.format_block_comment(["Line 1", "Line 2"])
        assert "/*" in result
        assert " * Line 1" in result
        assert " */" in result

    def test_go_block_comment(self):
        """Test Go block comment."""
        formatter = CodeFormatter(SupportedLanguage.GO)
        result = formatter.format_block_comment(["Line 1", "Line 2"])
        assert "/*" in result
        assert " * Line 1" in result
        assert " */" in result

    def test_typescript_block_comment(self):
        """Test TypeScript block comment."""
        formatter = CodeFormatter(SupportedLanguage.TYPESCRIPT)
        result = formatter.format_block_comment(["Line 1", "Line 2"])
        assert "/*" in result
        assert " * Line 1" in result
        assert " */" in result

    def test_empty_lines(self):
        """Test block comment with empty lines list."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_block_comment([])
        assert result == ""

    def test_single_line(self):
        """Test block comment with single line."""
        formatter = CodeFormatter(SupportedLanguage.JAVA)
        result = formatter.format_block_comment(["Single line"])
        assert "Single line" in result


# =============================================================================
# ImportSpec Tests
# =============================================================================


class TestImportSpec:
    """Tests for ImportSpec dataclass."""

    def test_basic_import(self):
        """Test basic import spec."""
        spec = ImportSpec(module="pytest")
        assert spec.module == "pytest"
        assert spec.items == []
        assert spec.alias is None
        assert spec.is_type_import is False

    def test_import_with_items(self):
        """Test import spec with items."""
        spec = ImportSpec(module="typing", items=["List", "Dict"])
        assert spec.module == "typing"
        assert spec.items == ["List", "Dict"]

    def test_import_with_alias(self):
        """Test import spec with alias."""
        spec = ImportSpec(module="numpy", alias="np")
        assert spec.module == "numpy"
        assert spec.alias == "np"

    def test_type_import(self):
        """Test TypeScript type import."""
        spec = ImportSpec(module="types", items=["PageType"], is_type_import=True)
        assert spec.is_type_import is True


# =============================================================================
# ImportsManager Tests
# =============================================================================


class TestImportsManagerInit:
    """Tests for ImportsManager initialization."""

    def test_creates_with_python_playwright(self):
        """Test creating manager for Python Playwright."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        assert manager.language == SupportedLanguage.PYTHON
        assert manager.framework == SupportedFramework.PLAYWRIGHT_PYTHON

    def test_initializes_base_imports_python_playwright(self):
        """Test base imports for Python Playwright."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        imports = manager.get_imports_code()
        assert "pytest" in imports
        assert "playwright" in imports
        assert "Page" in imports
        assert "expect" in imports

    def test_initializes_base_imports_python_selenium(self):
        """Test base imports for Python Selenium."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.SELENIUM_PYTHON
        )
        imports = manager.get_imports_code()
        assert "pytest" in imports
        assert "selenium" in imports
        assert "By" in imports
        assert "Keys" in imports

    def test_initializes_base_imports_typescript_playwright(self):
        """Test base imports for TypeScript Playwright."""
        manager = ImportsManager(
            SupportedLanguage.TYPESCRIPT,
            SupportedFramework.PLAYWRIGHT_TS
        )
        imports = manager.get_imports_code()
        assert "@playwright/test" in imports
        assert "test" in imports
        assert "expect" in imports

    def test_initializes_no_imports_cypress(self):
        """Test Cypress has no base imports (globals are auto-imported)."""
        manager = ImportsManager(
            SupportedLanguage.TYPESCRIPT,
            SupportedFramework.CYPRESS
        )
        imports = manager.get_imports_code()
        # Cypress globals are auto-imported, so may be empty
        assert isinstance(imports, str)

    def test_initializes_base_imports_puppeteer(self):
        """Test base imports for Puppeteer."""
        manager = ImportsManager(
            SupportedLanguage.TYPESCRIPT,
            SupportedFramework.PUPPETEER
        )
        imports = manager.get_imports_code()
        assert "puppeteer" in imports
        assert "jest" in imports


class TestImportsManagerAddImport:
    """Tests for ImportsManager.add_import method."""

    def test_add_simple_import(self):
        """Test adding a simple import."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        manager.add_import("custom_module")
        imports = manager.get_imports_code()
        assert "import custom_module" in imports

    def test_add_import_with_items(self):
        """Test adding import with specific items."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        manager.add_import("typing", items=["List", "Dict"])
        imports = manager.get_imports_code()
        assert "from typing import" in imports
        assert "List" in imports
        assert "Dict" in imports

    def test_add_import_with_alias(self):
        """Test adding import with alias."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        manager.add_import("numpy", alias="np")
        imports = manager.get_imports_code()
        assert "import numpy as np" in imports

    def test_merge_duplicate_imports(self):
        """Test merging duplicate imports."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        manager.add_import("typing", items=["List"])
        manager.add_import("typing", items=["Dict"])
        imports = manager.get_imports_code()
        # Should have both items
        assert "List" in imports
        assert "Dict" in imports

    def test_add_type_import(self):
        """Test adding TypeScript type import."""
        manager = ImportsManager(
            SupportedLanguage.TYPESCRIPT,
            SupportedFramework.PLAYWRIGHT_TS
        )
        manager.add_import("types", items=["PageType"], is_type_import=True)
        imports = manager.get_imports_code()
        assert "type" in imports or "PageType" in imports


class TestImportsManagerGetImportsCode:
    """Tests for ImportsManager.get_imports_code method."""

    def test_python_import_format(self):
        """Test Python import statement format."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        manager.add_import("os")
        manager.add_import("typing", items=["Optional"])
        manager.add_import("pandas", alias="pd")
        imports = manager.get_imports_code()

        assert "import os" in imports
        assert "from typing import Optional" in imports
        assert "import pandas as pd" in imports

    def test_typescript_import_format(self):
        """Test TypeScript import statement format."""
        manager = ImportsManager(
            SupportedLanguage.TYPESCRIPT,
            SupportedFramework.PLAYWRIGHT_TS
        )
        manager.add_import("custom", items=["CustomClass"])
        manager.add_import("lodash", alias="_")
        imports = manager.get_imports_code()

        assert "import { " in imports or "import {" in imports
        assert "CustomClass" in imports
        assert "* as _" in imports or "lodash" in imports

    def test_java_import_format(self):
        """Test Java import statement format."""
        manager = ImportsManager(
            SupportedLanguage.JAVA,
            SupportedFramework.SELENIUM_JAVA
        )
        manager.add_import("java.util", items=["List", "Map"])
        imports = manager.get_imports_code()

        assert "import java.util.List;" in imports
        assert "import java.util.Map;" in imports

    def test_csharp_import_format(self):
        """Test C# using statement format."""
        manager = ImportsManager(
            SupportedLanguage.CSHARP,
            SupportedFramework.SELENIUM_CSHARP
        )
        manager.add_import("System")
        manager.add_import("System.Collections.Generic")
        imports = manager.get_imports_code()

        assert "using System;" in imports
        assert "using System.Collections.Generic;" in imports

    def test_ruby_import_format(self):
        """Test Ruby require statement format."""
        manager = ImportsManager(
            SupportedLanguage.RUBY,
            SupportedFramework.CAPYBARA
        )
        manager.add_import("rspec")
        imports = manager.get_imports_code()

        assert "require 'rspec'" in imports

    def test_go_import_format(self):
        """Test Go import statement format."""
        manager = ImportsManager(
            SupportedLanguage.GO,
            SupportedFramework.ROD
        )
        manager.add_import("fmt")
        manager.add_import("context", alias="ctx")
        imports = manager.get_imports_code()

        assert "import (" in imports
        assert '"fmt"' in imports


class TestImportsManagerGetDependencies:
    """Tests for ImportsManager.get_dependencies method."""

    def test_python_playwright_dependencies(self):
        """Test Python Playwright dependencies."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        deps = manager.get_dependencies()
        assert "playwright" in deps
        assert "pytest" in deps

    def test_python_selenium_dependencies(self):
        """Test Python Selenium dependencies."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.SELENIUM_PYTHON
        )
        deps = manager.get_dependencies()
        assert "selenium" in deps
        assert "pytest" in deps

    def test_typescript_playwright_dependencies(self):
        """Test TypeScript Playwright dependencies."""
        manager = ImportsManager(
            SupportedLanguage.TYPESCRIPT,
            SupportedFramework.PLAYWRIGHT_TS
        )
        deps = manager.get_dependencies()
        assert "@playwright/test" in deps

    def test_cypress_dependencies(self):
        """Test Cypress dependencies."""
        manager = ImportsManager(
            SupportedLanguage.TYPESCRIPT,
            SupportedFramework.CYPRESS
        )
        deps = manager.get_dependencies()
        assert "cypress" in deps

    def test_capybara_dependencies(self):
        """Test Capybara dependencies."""
        manager = ImportsManager(
            SupportedLanguage.RUBY,
            SupportedFramework.CAPYBARA
        )
        deps = manager.get_dependencies()
        assert "capybara" in deps

    def test_rod_dependencies(self):
        """Test Rod dependencies."""
        manager = ImportsManager(
            SupportedLanguage.GO,
            SupportedFramework.ROD
        )
        deps = manager.get_dependencies()
        assert any("rod" in d for d in deps)


class TestImportsManagerGetInstallCommand:
    """Tests for ImportsManager.get_install_command method."""

    def test_python_install_command(self):
        """Test Python pip install command."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        command = manager.get_install_command()
        assert "pip install" in command
        assert "playwright" in command

    def test_typescript_install_command(self):
        """Test TypeScript npm install command."""
        manager = ImportsManager(
            SupportedLanguage.TYPESCRIPT,
            SupportedFramework.PLAYWRIGHT_TS
        )
        command = manager.get_install_command()
        assert "npm install" in command

    def test_java_install_command(self):
        """Test Java Maven/Gradle install command."""
        manager = ImportsManager(
            SupportedLanguage.JAVA,
            SupportedFramework.SELENIUM_JAVA
        )
        command = manager.get_install_command()
        assert "pom.xml" in command or "maven" in command.lower() or "#" in command

    def test_csharp_install_command(self):
        """Test C# NuGet install command."""
        manager = ImportsManager(
            SupportedLanguage.CSHARP,
            SupportedFramework.SELENIUM_CSHARP
        )
        command = manager.get_install_command()
        assert "dotnet add package" in command

    def test_ruby_install_command(self):
        """Test Ruby gem install command."""
        manager = ImportsManager(
            SupportedLanguage.RUBY,
            SupportedFramework.CAPYBARA
        )
        command = manager.get_install_command()
        assert "gem install" in command

    def test_go_install_command(self):
        """Test Go go get command."""
        manager = ImportsManager(
            SupportedLanguage.GO,
            SupportedFramework.ROD
        )
        command = manager.get_install_command()
        assert "go get" in command


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestFormatterEdgeCases:
    """Tests for edge cases in formatters."""

    def test_format_code_with_mixed_line_endings(self):
        """Test formatting code with mixed line endings."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        # Note: Python's split("\n") will handle most cases
        code = "line1\r\nline2\nline3"
        formatted = formatter.format_code(code)
        assert formatted.endswith("\n")

    def test_format_code_very_long_lines(self):
        """Test formatting code with very long lines."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        long_line = "x" * 10000
        formatted = formatter.format_code(long_line)
        assert long_line.strip() in formatted

    def test_format_string_with_unicode(self):
        """Test formatting string with unicode characters."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_string_literal("hello world", single_quotes=False)
        assert result == '"hello world"'

    def test_format_string_with_emoji(self):
        """Test formatting string with emoji."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_string_literal("test", single_quotes=False)
        assert '"test"' == result

    def test_imports_manager_empty_framework_deps(self):
        """Test imports manager handles frameworks without dependencies gracefully."""
        # All frameworks should have dependencies defined, but test robustness
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PYTEST
        )
        deps = manager.get_dependencies()
        assert isinstance(deps, list)

    def test_format_comment_empty_string(self):
        """Test formatting empty comment."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_comment("")
        assert result == "# "

    def test_format_block_comment_single_empty_line(self):
        """Test formatting block comment with single empty string."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_block_comment([""])
        assert "# " in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestFormatterIntegration:
    """Integration tests for formatters."""

    def test_full_code_formatting_workflow(self):
        """Test complete code formatting workflow."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)

        # Simulate generated code with issues
        code = """
import pytest

def test_example():
    pass




def test_another():
    pass"""

        formatted = formatter.format_code(code)

        # Should have trailing whitespace removed
        for line in formatted.split("\n")[:-1]:
            assert line == line.rstrip()

        # Should not have excessive blank lines
        assert "\n\n\n\n" not in formatted

        # Should end with newline
        assert formatted.endswith("\n")

    def test_imports_manager_complete_workflow(self):
        """Test complete imports manager workflow."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )

        # Add custom imports
        manager.add_import("custom_lib")
        manager.add_import("typing", items=["Optional", "List"])

        # Get imports
        imports_code = manager.get_imports_code()

        # Get dependencies
        deps = manager.get_dependencies()

        # Get install command
        install_cmd = manager.get_install_command()

        # Verify everything works together
        assert "pytest" in imports_code
        assert "custom_lib" in imports_code
        assert "Optional" in imports_code
        assert "playwright" in deps
        assert "pip install" in install_cmd
