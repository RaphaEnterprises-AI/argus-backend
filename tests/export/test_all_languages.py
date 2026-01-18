"""Cross-language tests for export functionality."""

import pytest

from src.export.engine import ExportEngine
from src.export.formatters import CodeFormatter, ImportsManager
from src.export.models import (
    FILE_EXTENSIONS,
    FRAMEWORK_DEPENDENCIES,
    FRAMEWORK_SUPPORT,
    ExportConfig,
    SupportedFramework,
    SupportedLanguage,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def engine():
    """Create ExportEngine instance."""
    return ExportEngine()


@pytest.fixture
def login_test_spec():
    """Standard login test spec for cross-language testing."""
    return {
        "id": "test-login-001",
        "name": "User Login Flow",
        "description": "Tests the user login functionality",
        "steps": [
            {"action": "goto", "target": "/login"},
            {"action": "fill", "target": "#username", "value": "testuser"},
            {"action": "fill", "target": "#password", "value": "password123"},
            {"action": "click", "target": "button[type=submit]"},
            {"action": "wait", "target": ".dashboard", "timeout": 5000},
        ],
        "assertions": [
            {"type": "url_contains", "expected": "/dashboard"},
            {"type": "element_visible", "target": ".welcome-message"},
            {"type": "text_contains", "target": ".user-greeting", "expected": "Welcome"},
        ],
    }


@pytest.fixture
def shopping_cart_spec():
    """E-commerce shopping cart test spec."""
    return {
        "id": "test-cart-002",
        "name": "Add to Cart Flow",
        "description": "Tests adding items to shopping cart",
        "steps": [
            {"action": "goto", "target": "https://shop.example.com/products"},
            {"action": "click", "target": ".product-item:first-child"},
            {"action": "select", "target": "#size", "value": "M"},
            {"action": "select", "target": "#color", "value": "Blue"},
            {"action": "fill", "target": "#quantity", "value": "2"},
            {"action": "click", "target": ".add-to-cart-btn"},
            {"action": "hover", "target": ".cart-icon"},
            {"action": "click", "target": ".view-cart-link"},
        ],
        "assertions": [
            {"type": "element_visible", "target": ".cart-item"},
            {"type": "text_contains", "target": ".item-quantity", "expected": "2"},
            {"type": "value_equals", "target": "#cart-total", "expected": "$99.98"},
        ],
    }


@pytest.fixture
def form_validation_spec():
    """Form with various input types test spec."""
    return {
        "id": "test-form-003",
        "name": "Form Validation Test",
        "description": "Tests form validation with various input types",
        "steps": [
            {"action": "goto", "target": "/contact"},
            {"action": "type", "target": "#name", "value": "John Doe"},
            {"action": "type", "target": "#email", "value": "john@example.com"},
            {"action": "type", "target": "#phone", "value": "+1-555-123-4567"},
            {"action": "select", "target": "#subject", "value": "Support"},
            {"action": "scroll", "value": "0,300"},
            {"action": "fill", "target": "#message", "value": "This is my message"},
            {"action": "click", "target": "#agree-terms"},
            {"action": "press_key", "value": "Tab"},
            {"action": "press_key", "value": "Enter"},
            {"action": "screenshot"},
        ],
        "assertions": [
            {"type": "element_visible", "target": ".success-message"},
            {"type": "element_hidden", "target": ".error-message"},
            {"type": "title_contains", "expected": "Thank You"},
        ],
    }


# =============================================================================
# All Languages Export Tests
# =============================================================================


class TestAllLanguagesExport:
    """Tests exporting to all supported languages."""

    ALL_COMBINATIONS = [
        (SupportedLanguage.PYTHON, SupportedFramework.PLAYWRIGHT_PYTHON),
        (SupportedLanguage.PYTHON, SupportedFramework.SELENIUM_PYTHON),
        (SupportedLanguage.TYPESCRIPT, SupportedFramework.PLAYWRIGHT_TS),
        (SupportedLanguage.JAVA, SupportedFramework.SELENIUM_JAVA),
        (SupportedLanguage.CSHARP, SupportedFramework.SELENIUM_CSHARP),
        (SupportedLanguage.RUBY, SupportedFramework.CAPYBARA),
        (SupportedLanguage.GO, SupportedFramework.ROD),
    ]

    @pytest.mark.parametrize("language,framework", ALL_COMBINATIONS)
    def test_login_flow_all_languages(self, engine, login_test_spec, language, framework):
        """Test login flow exports correctly to all languages."""
        config = ExportConfig(language=language, framework=framework)
        result = engine.export(login_test_spec, config)

        assert result.success, f"Failed for {language.value}/{framework.value}: {result.error}"
        assert result.code
        assert result.file_extension == FILE_EXTENSIONS[language]
        assert result.dependencies

        # Verify all steps are represented in code
        code_lower = result.code.lower()
        assert "/login" in result.code or "login" in code_lower
        assert "#username" in result.code or "username" in code_lower
        assert "#password" in result.code or "password" in code_lower

    @pytest.mark.parametrize("language,framework", ALL_COMBINATIONS)
    def test_shopping_cart_all_languages(self, engine, shopping_cart_spec, language, framework):
        """Test shopping cart flow exports correctly to all languages."""
        config = ExportConfig(language=language, framework=framework)
        result = engine.export(shopping_cart_spec, config)

        assert result.success, f"Failed for {language.value}/{framework.value}: {result.error}"
        assert result.code
        assert "shop.example.com" in result.code or "products" in result.code.lower()

    @pytest.mark.parametrize("language,framework", ALL_COMBINATIONS)
    def test_form_validation_all_languages(self, engine, form_validation_spec, language, framework):
        """Test form validation flow exports correctly to all languages."""
        config = ExportConfig(language=language, framework=framework)
        result = engine.export(form_validation_spec, config)

        assert result.success, f"Failed for {language.value}/{framework.value}: {result.error}"
        assert result.code

        # Should have screenshot action
        assert "screenshot" in result.code.lower()


# =============================================================================
# Language-Specific Syntax Tests
# =============================================================================


class TestLanguageSpecificSyntax:
    """Tests for language-specific syntax correctness."""

    def test_python_syntax_patterns(self, engine, login_test_spec):
        """Test Python syntax patterns."""
        config = ExportConfig(
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.PLAYWRIGHT_PYTHON,
        )
        result = engine.export(login_test_spec, config)

        assert "import " in result.code
        assert "class Test" in result.code
        assert "def test_" in result.code
        assert 'page.' in result.code
        # Should use Python string quotes
        assert '""' in result.code or '("' in result.code

    def test_typescript_syntax_patterns(self, engine, login_test_spec):
        """Test TypeScript syntax patterns."""
        config = ExportConfig(
            language=SupportedLanguage.TYPESCRIPT,
            framework=SupportedFramework.PLAYWRIGHT_TS,
        )
        result = engine.export(login_test_spec, config)

        assert "import {" in result.code or "import " in result.code
        assert "test(" in result.code or "test.describe" in result.code
        assert "await " in result.code
        # TypeScript uses semicolons
        assert ";" in result.code
        # TypeScript typically uses single quotes
        assert "'" in result.code

    def test_java_syntax_patterns(self, engine, login_test_spec):
        """Test Java syntax patterns."""
        config = ExportConfig(
            language=SupportedLanguage.JAVA,
            framework=SupportedFramework.SELENIUM_JAVA,
        )
        result = engine.export(login_test_spec, config)

        assert "import " in result.code
        assert "public class" in result.code
        assert "public void" in result.code or "@Test" in result.code
        # Java uses double quotes for strings
        assert '"' in result.code
        # Java statements end with semicolons
        assert ";" in result.code

    def test_csharp_syntax_patterns(self, engine, login_test_spec):
        """Test C# syntax patterns."""
        config = ExportConfig(
            language=SupportedLanguage.CSHARP,
            framework=SupportedFramework.SELENIUM_CSHARP,
        )
        result = engine.export(login_test_spec, config)

        assert "using " in result.code
        assert "public class" in result.code
        assert "public void" in result.code or "[Test]" in result.code
        # C# uses double quotes for strings
        assert '"' in result.code
        assert ";" in result.code

    def test_ruby_syntax_patterns(self, engine, login_test_spec):
        """Test Ruby syntax patterns."""
        config = ExportConfig(
            language=SupportedLanguage.RUBY,
            framework=SupportedFramework.CAPYBARA,
        )
        result = engine.export(login_test_spec, config)

        assert "require " in result.code
        assert "describe " in result.code or "RSpec.describe" in result.code
        # Ruby uses 'end' keyword
        assert "end" in result.code

    def test_go_syntax_patterns(self, engine, login_test_spec):
        """Test Go syntax patterns."""
        config = ExportConfig(
            language=SupportedLanguage.GO,
            framework=SupportedFramework.ROD,
        )
        result = engine.export(login_test_spec, config)

        assert "package " in result.code
        assert "import (" in result.code
        assert "func Test" in result.code
        assert "*testing.T" in result.code
        # Go uses double quotes for strings
        assert '"' in result.code


# =============================================================================
# Assertions Across Languages Tests
# =============================================================================


class TestAssertionsAcrossLanguages:
    """Tests for assertion generation across all languages."""

    ASSERTION_TYPES = [
        "element_visible",
        "element_hidden",
        "text_contains",
        "text_equals",
        "url_contains",
        "value_equals",
        "title_contains",
    ]

    @pytest.fixture
    def assertion_test_spec(self):
        """Test spec with all assertion types."""
        return {
            "id": "test-assertions",
            "name": "Assertion Test",
            "steps": [{"action": "goto", "target": "/test"}],
            "assertions": [
                {"type": "element_visible", "target": ".visible-el"},
                {"type": "element_hidden", "target": ".hidden-el"},
                {"type": "text_contains", "target": ".text-el", "expected": "contains this"},
                {"type": "text_equals", "target": ".exact-el", "expected": "exact match"},
                {"type": "url_contains", "expected": "/expected"},
                {"type": "value_equals", "target": "#input", "expected": "input value"},
                {"type": "title_contains", "expected": "Page Title"},
            ],
        }

    @pytest.mark.parametrize("language,framework", [
        (SupportedLanguage.PYTHON, SupportedFramework.PLAYWRIGHT_PYTHON),
        (SupportedLanguage.TYPESCRIPT, SupportedFramework.PLAYWRIGHT_TS),
        (SupportedLanguage.GO, SupportedFramework.ROD),
    ])
    def test_all_assertion_types(self, engine, assertion_test_spec, language, framework):
        """Test all assertion types are generated."""
        config = ExportConfig(language=language, framework=framework)
        result = engine.export(assertion_test_spec, config)

        assert result.success
        code_lower = result.code.lower()

        # Should have assertion-related keywords
        assert "assert" in code_lower or "expect" in code_lower

        # Should reference the target elements
        assert ".visible-el" in result.code
        assert ".hidden-el" in result.code


# =============================================================================
# CodeFormatter Tests
# =============================================================================


class TestCodeFormatter:
    """Tests for CodeFormatter class."""

    @pytest.mark.parametrize("language", list(SupportedLanguage))
    def test_formatter_creation(self, language):
        """Test formatter can be created for all languages."""
        formatter = CodeFormatter(language)
        assert formatter.language == language

    def test_format_removes_trailing_whitespace(self):
        """Test formatting removes trailing whitespace."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        code = "line with spaces   \n  indented line  \n"
        formatted = formatter.format_code(code)
        lines = formatted.split("\n")
        for line in lines[:-1]:  # Last may be empty
            assert line == line.rstrip()

    def test_format_limits_blank_lines(self):
        """Test formatting limits consecutive blank lines."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        code = "line1\n\n\n\n\nline2"
        formatted = formatter.format_code(code)
        # Should not have more than 2 consecutive blank lines
        assert "\n\n\n\n" not in formatted

    def test_format_ends_with_newline(self):
        """Test formatting ensures file ends with newline."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        code = "code without trailing newline"
        formatted = formatter.format_code(code)
        assert formatted.endswith("\n")

    @pytest.mark.parametrize("language,expected_indent", [
        (SupportedLanguage.PYTHON, "    "),
        (SupportedLanguage.RUBY, "    "),
        (SupportedLanguage.GO, "\t"),
        (SupportedLanguage.JAVA, "    "),
    ])
    def test_indent_by_language(self, language, expected_indent):
        """Test correct indent style by language."""
        formatter = CodeFormatter(language)
        assert formatter.get_indent() == expected_indent

    def test_format_string_literal_double_quotes(self):
        """Test string literal formatting with double quotes."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_string_literal("hello", single_quotes=False)
        assert result == '"hello"'

    def test_format_string_literal_single_quotes(self):
        """Test string literal formatting with single quotes."""
        formatter = CodeFormatter(SupportedLanguage.TYPESCRIPT)
        result = formatter.format_string_literal("hello", single_quotes=True)
        assert result == "'hello'"

    def test_format_string_escapes_quotes(self):
        """Test string literal escapes embedded quotes."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_string_literal('say "hello"', single_quotes=False)
        assert '\\"' in result

    def test_format_comment_python(self):
        """Test Python comment formatting."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_comment("This is a comment")
        assert result == "# This is a comment"

    def test_format_comment_java(self):
        """Test Java comment formatting."""
        formatter = CodeFormatter(SupportedLanguage.JAVA)
        result = formatter.format_comment("This is a comment")
        assert result == "// This is a comment"

    def test_format_doc_comment_python(self):
        """Test Python docstring formatting."""
        formatter = CodeFormatter(SupportedLanguage.PYTHON)
        result = formatter.format_comment("Docstring", doc_comment=True)
        assert '"""Docstring"""' == result

    def test_format_doc_comment_java(self):
        """Test Java doc comment formatting."""
        formatter = CodeFormatter(SupportedLanguage.JAVA)
        result = formatter.format_comment("Doc comment", doc_comment=True)
        assert "/**" in result and "*/" in result


# =============================================================================
# ImportsManager Tests
# =============================================================================


class TestImportsManager:
    """Tests for ImportsManager class."""

    def test_python_playwright_imports(self):
        """Test Python Playwright base imports."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        imports = manager.get_imports_code()
        assert "import pytest" in imports
        assert "playwright" in imports

    def test_python_selenium_imports(self):
        """Test Python Selenium base imports."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.SELENIUM_PYTHON
        )
        imports = manager.get_imports_code()
        assert "selenium" in imports
        assert "By" in imports

    def test_typescript_playwright_imports(self):
        """Test TypeScript Playwright base imports."""
        manager = ImportsManager(
            SupportedLanguage.TYPESCRIPT,
            SupportedFramework.PLAYWRIGHT_TS
        )
        imports = manager.get_imports_code()
        assert "@playwright/test" in imports

    def test_add_custom_import(self):
        """Test adding custom imports."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        manager.add_import("custom_module", items=["CustomClass"])
        imports = manager.get_imports_code()
        assert "from custom_module import CustomClass" in imports

    def test_merge_duplicate_imports(self):
        """Test merging duplicate imports."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        manager.add_import("module", items=["ItemA"])
        manager.add_import("module", items=["ItemB"])
        imports = manager.get_imports_code()
        # Should have both items in one import
        assert "module" in imports

    def test_get_dependencies(self):
        """Test getting package dependencies."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        deps = manager.get_dependencies()
        assert "playwright" in deps

    def test_get_install_command_python(self):
        """Test Python install command."""
        manager = ImportsManager(
            SupportedLanguage.PYTHON,
            SupportedFramework.PLAYWRIGHT_PYTHON
        )
        command = manager.get_install_command()
        assert "pip install" in command

    def test_get_install_command_typescript(self):
        """Test TypeScript install command."""
        manager = ImportsManager(
            SupportedLanguage.TYPESCRIPT,
            SupportedFramework.PLAYWRIGHT_TS
        )
        command = manager.get_install_command()
        assert "npm install" in command

    def test_get_install_command_java(self):
        """Test Java install command (Maven)."""
        manager = ImportsManager(
            SupportedLanguage.JAVA,
            SupportedFramework.SELENIUM_JAVA
        )
        command = manager.get_install_command()
        assert "pom.xml" in command

    def test_get_install_command_csharp(self):
        """Test C# install command (NuGet)."""
        manager = ImportsManager(
            SupportedLanguage.CSHARP,
            SupportedFramework.SELENIUM_CSHARP
        )
        command = manager.get_install_command()
        assert "dotnet add package" in command

    def test_get_install_command_ruby(self):
        """Test Ruby install command."""
        manager = ImportsManager(
            SupportedLanguage.RUBY,
            SupportedFramework.CAPYBARA
        )
        command = manager.get_install_command()
        assert "gem install" in command

    def test_get_install_command_go(self):
        """Test Go install command."""
        manager = ImportsManager(
            SupportedLanguage.GO,
            SupportedFramework.ROD
        )
        command = manager.get_install_command()
        assert "go get" in command


# =============================================================================
# Framework Dependencies Tests
# =============================================================================


class TestFrameworkDependencies:
    """Tests for framework dependency information."""

    @pytest.mark.parametrize("framework,expected_deps", [
        (SupportedFramework.PLAYWRIGHT_PYTHON, ["playwright", "pytest"]),
        (SupportedFramework.SELENIUM_PYTHON, ["selenium", "pytest"]),
        (SupportedFramework.PLAYWRIGHT_TS, ["@playwright/test"]),
        (SupportedFramework.CYPRESS, ["cypress"]),
        (SupportedFramework.SELENIUM_JAVA, ["selenium"]),
        (SupportedFramework.CAPYBARA, ["capybara"]),
        (SupportedFramework.ROD, ["rod"]),
    ])
    def test_framework_has_dependencies(self, framework, expected_deps):
        """Test frameworks have expected dependencies."""
        deps = FRAMEWORK_DEPENDENCIES.get(framework, [])
        for expected in expected_deps:
            assert any(expected in dep for dep in deps), f"Missing {expected} in {framework.value}"


# =============================================================================
# File Extension Tests
# =============================================================================


class TestFileExtensions:
    """Tests for file extension correctness."""

    @pytest.mark.parametrize("language,expected_ext", [
        (SupportedLanguage.PYTHON, ".py"),
        (SupportedLanguage.TYPESCRIPT, ".ts"),
        (SupportedLanguage.JAVA, ".java"),
        (SupportedLanguage.CSHARP, ".cs"),
        (SupportedLanguage.RUBY, ".rb"),
        (SupportedLanguage.GO, ".go"),
    ])
    def test_file_extensions(self, language, expected_ext):
        """Test correct file extensions for each language."""
        assert FILE_EXTENSIONS[language] == expected_ext


# =============================================================================
# Framework Support Matrix Tests
# =============================================================================


class TestFrameworkSupportMatrix:
    """Tests for framework support matrix."""

    def test_python_supports_playwright(self):
        """Test Python supports Playwright."""
        assert SupportedFramework.PLAYWRIGHT_PYTHON in FRAMEWORK_SUPPORT[SupportedLanguage.PYTHON]

    def test_python_supports_selenium(self):
        """Test Python supports Selenium."""
        assert SupportedFramework.SELENIUM_PYTHON in FRAMEWORK_SUPPORT[SupportedLanguage.PYTHON]

    def test_typescript_supports_playwright(self):
        """Test TypeScript supports Playwright."""
        assert SupportedFramework.PLAYWRIGHT_TS in FRAMEWORK_SUPPORT[SupportedLanguage.TYPESCRIPT]

    def test_typescript_supports_cypress(self):
        """Test TypeScript supports Cypress."""
        assert SupportedFramework.CYPRESS in FRAMEWORK_SUPPORT[SupportedLanguage.TYPESCRIPT]

    def test_java_supports_selenium(self):
        """Test Java supports Selenium."""
        assert SupportedFramework.SELENIUM_JAVA in FRAMEWORK_SUPPORT[SupportedLanguage.JAVA]

    def test_csharp_supports_selenium(self):
        """Test C# supports Selenium."""
        assert SupportedFramework.SELENIUM_CSHARP in FRAMEWORK_SUPPORT[SupportedLanguage.CSHARP]

    def test_ruby_supports_capybara(self):
        """Test Ruby supports Capybara."""
        assert SupportedFramework.CAPYBARA in FRAMEWORK_SUPPORT[SupportedLanguage.RUBY]

    def test_go_supports_rod(self):
        """Test Go supports Rod."""
        assert SupportedFramework.ROD in FRAMEWORK_SUPPORT[SupportedLanguage.GO]

    def test_all_languages_have_frameworks(self):
        """Test all languages have at least one framework."""
        for language in SupportedLanguage:
            assert language in FRAMEWORK_SUPPORT
            assert len(FRAMEWORK_SUPPORT[language]) > 0


# =============================================================================
# Batch Export Consistency Tests
# =============================================================================


class TestBatchExportConsistency:
    """Tests for batch export consistency across languages."""

    def test_batch_export_same_language(self, engine, login_test_spec, shopping_cart_spec):
        """Test batch export produces consistent results for same language."""
        config = ExportConfig(
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.PLAYWRIGHT_PYTHON,
        )
        results = engine.export_batch([login_test_spec, shopping_cart_spec], config)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.language == SupportedLanguage.PYTHON for r in results)
        assert all(r.file_extension == ".py" for r in results)

    def test_batch_export_different_test_ids(self, engine, login_test_spec, shopping_cart_spec):
        """Test batch export preserves different test IDs."""
        config = ExportConfig()
        results = engine.export_batch([login_test_spec, shopping_cart_spec], config)

        assert results[0].metadata.get("test_id") == "test-login-001"
        assert results[1].metadata.get("test_id") == "test-cart-002"
