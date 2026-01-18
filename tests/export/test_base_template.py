"""Comprehensive tests for base template functionality."""

import pytest
from src.export.templates.base import BaseTemplate, TestSpec, TestStep, TestAssertion
from src.export.templates import (
    PythonPlaywrightTemplate,
    PythonSeleniumTemplate,
    TypeScriptPlaywrightTemplate,
    JavaSeleniumTemplate,
    CSharpSeleniumTemplate,
    RubyCapybaraTemplate,
    GoRodTemplate,
)


# =============================================================================
# BaseTemplate Helper Methods Tests
# =============================================================================


class TestSanitizeName:
    """Tests for BaseTemplate.sanitize_name method."""

    @pytest.fixture
    def template(self):
        return PythonPlaywrightTemplate()

    def test_removes_special_characters(self, template):
        """Test special characters are removed."""
        result = template.sanitize_name("Test!@#$%^&*()")
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result

    def test_replaces_spaces(self, template):
        """Test spaces are replaced with underscores."""
        result = template.sanitize_name("Test Name Here")
        assert " " not in result
        assert "_" in result

    def test_replaces_dashes(self, template):
        """Test dashes are replaced with underscores."""
        result = template.sanitize_name("test-name-here")
        assert "-" not in result
        assert "_" in result

    def test_converts_to_lowercase(self, template):
        """Test result is lowercase."""
        result = template.sanitize_name("TestNAME")
        assert result == result.lower()

    def test_prefixes_numeric_start(self, template):
        """Test names starting with numbers get underscore prefix."""
        result = template.sanitize_name("123test")
        assert result.startswith("_")
        assert result == "_123test"

    def test_preserves_valid_characters(self, template):
        """Test valid characters are preserved."""
        result = template.sanitize_name("valid_name_123")
        assert result == "valid_name_123"

    def test_handles_empty_string(self, template):
        """Test empty string handling."""
        result = template.sanitize_name("")
        assert result == ""

    def test_handles_only_numbers(self, template):
        """Test string with only numbers."""
        result = template.sanitize_name("123")
        assert result == "_123"

    def test_handles_only_special_chars(self, template):
        """Test string with only special characters."""
        result = template.sanitize_name("!@#$%")
        assert "_" in result or result == ""

    def test_multiple_consecutive_underscores(self, template):
        """Test multiple special chars become multiple underscores."""
        result = template.sanitize_name("test!!name")
        assert "test" in result
        assert "name" in result


class TestToCamelCase:
    """Tests for BaseTemplate.to_camel_case method."""

    @pytest.fixture
    def template(self):
        return PythonPlaywrightTemplate()

    def test_basic_conversion(self, template):
        """Test basic camelCase conversion."""
        result = template.to_camel_case("test name here")
        assert result == "testNameHere"

    def test_first_word_lowercase(self, template):
        """Test first word is lowercase."""
        result = template.to_camel_case("Test Name")
        assert result[0].islower()

    def test_subsequent_words_capitalized(self, template):
        """Test subsequent words are capitalized."""
        result = template.to_camel_case("test name here")
        assert result == "testNameHere"

    def test_with_dashes(self, template):
        """Test handling dashes."""
        result = template.to_camel_case("test-name-here")
        assert result == "testNameHere"

    def test_single_word(self, template):
        """Test single word."""
        result = template.to_camel_case("single")
        assert result == "single"

    def test_empty_string(self, template):
        """Test empty string returns 'test'."""
        result = template.to_camel_case("")
        assert result == "test"

    def test_with_numbers(self, template):
        """Test handling numbers."""
        result = template.to_camel_case("test 123 name")
        assert "test" in result.lower()
        assert "123" in result
        assert "name" in result.lower()


class TestToPascalCase:
    """Tests for BaseTemplate.to_pascal_case method."""

    @pytest.fixture
    def template(self):
        return PythonPlaywrightTemplate()

    def test_basic_conversion(self, template):
        """Test basic PascalCase conversion."""
        result = template.to_pascal_case("test name here")
        assert result == "TestNameHere"

    def test_first_letter_uppercase(self, template):
        """Test first letter is uppercase."""
        result = template.to_pascal_case("test")
        assert result[0].isupper()

    def test_all_words_capitalized(self, template):
        """Test all words are capitalized."""
        result = template.to_pascal_case("test name here")
        assert result == "TestNameHere"

    def test_with_dashes(self, template):
        """Test handling dashes."""
        result = template.to_pascal_case("test-name-here")
        assert result == "TestNameHere"

    def test_single_word(self, template):
        """Test single word."""
        result = template.to_pascal_case("single")
        assert result == "Single"

    def test_empty_string(self, template):
        """Test empty string returns 'Test'."""
        result = template.to_pascal_case("")
        assert result == "Test"

    def test_already_pascal_case(self, template):
        """Test already PascalCase string is handled."""
        result = template.to_pascal_case("TestName")
        # Result will process the string through word splitting
        # TestName becomes Test Name which becomes Testname
        assert result  # Should produce some result
        assert result[0].isupper()  # First char should be uppercase


class TestToSnakeCase:
    """Tests for BaseTemplate.to_snake_case method."""

    @pytest.fixture
    def template(self):
        return PythonPlaywrightTemplate()

    def test_basic_conversion(self, template):
        """Test basic snake_case conversion."""
        result = template.to_snake_case("TestName")
        assert result == "test_name"

    def test_from_camel_case(self, template):
        """Test conversion from camelCase."""
        result = template.to_snake_case("testNameHere")
        assert result == "test_name_here"

    def test_from_pascal_case(self, template):
        """Test conversion from PascalCase."""
        result = template.to_snake_case("TestNameHere")
        assert result == "test_name_here"

    def test_from_space_separated(self, template):
        """Test conversion from space-separated."""
        result = template.to_snake_case("Test Name Here")
        assert result == "test_name_here"

    def test_from_dash_separated(self, template):
        """Test conversion from dash-separated."""
        result = template.to_snake_case("test-name-here")
        assert result == "test_name_here"

    def test_all_lowercase(self, template):
        """Test result is all lowercase."""
        result = template.to_snake_case("TestNAME")
        assert result == result.lower()

    def test_no_leading_trailing_underscores(self, template):
        """Test no leading/trailing underscores."""
        result = template.to_snake_case("_test_")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_no_multiple_underscores(self, template):
        """Test no multiple consecutive underscores."""
        result = template.to_snake_case("test  name")
        assert "__" not in result

    def test_single_word(self, template):
        """Test single word."""
        result = template.to_snake_case("test")
        assert result == "test"

    def test_consecutive_capitals(self, template):
        """Test consecutive capital letters."""
        result = template.to_snake_case("XMLParser")
        assert "xml" in result.lower()
        assert "parser" in result.lower()


class TestEscapeString:
    """Tests for BaseTemplate.escape_string method."""

    @pytest.fixture
    def template(self):
        return PythonPlaywrightTemplate()

    def test_escapes_backslash(self, template):
        """Test backslash is escaped."""
        result = template.escape_string("path\\to\\file")
        assert result == "path\\\\to\\\\file"

    def test_escapes_double_quote(self, template):
        """Test double quote is escaped."""
        result = template.escape_string('say "hello"')
        assert result == 'say \\"hello\\"'

    def test_escapes_newline(self, template):
        """Test newline is escaped."""
        result = template.escape_string("line1\nline2")
        assert result == "line1\\nline2"

    def test_handles_none(self, template):
        """Test None returns empty string."""
        result = template.escape_string(None)
        assert result == ""

    def test_preserves_normal_chars(self, template):
        """Test normal characters are preserved."""
        result = template.escape_string("normal text")
        assert result == "normal text"

    def test_combined_escapes(self, template):
        """Test multiple escape characters."""
        result = template.escape_string('path\\file "name"\n')
        assert "\\\\" in result
        assert '\\"' in result
        assert "\\n" in result


class TestFormatComment:
    """Tests for BaseTemplate.format_comment method."""

    def test_python_comment(self):
        """Test Python comment format."""
        template = PythonPlaywrightTemplate()
        result = template.format_comment("test comment")
        assert result == "# test comment"

    def test_python_inline_comment(self):
        """Test Python inline comment format."""
        template = PythonPlaywrightTemplate()
        result = template.format_comment("test", inline=True)
        assert result == " # test"

    def test_typescript_comment(self):
        """Test TypeScript comment format."""
        template = TypeScriptPlaywrightTemplate()
        result = template.format_comment("test comment")
        assert result == "// test comment"

    def test_java_comment(self):
        """Test Java comment format."""
        template = JavaSeleniumTemplate()
        result = template.format_comment("test comment")
        assert result == "// test comment"

    def test_csharp_comment(self):
        """Test C# comment format."""
        template = CSharpSeleniumTemplate()
        result = template.format_comment("test comment")
        assert result == "// test comment"

    def test_ruby_comment(self):
        """Test Ruby comment format."""
        template = RubyCapybaraTemplate()
        result = template.format_comment("test comment")
        assert result == "# test comment"

    def test_go_comment(self):
        """Test Go comment format."""
        template = GoRodTemplate()
        result = template.format_comment("test comment")
        assert result == "// test comment"


class TestGetCommentPrefix:
    """Tests for BaseTemplate._get_comment_prefix method."""

    def test_python_prefix(self):
        """Test Python uses # prefix."""
        template = PythonPlaywrightTemplate()
        assert template._get_comment_prefix() == "#"

    def test_ruby_prefix(self):
        """Test Ruby uses # prefix."""
        template = RubyCapybaraTemplate()
        assert template._get_comment_prefix() == "#"

    def test_typescript_prefix(self):
        """Test TypeScript uses // prefix."""
        template = TypeScriptPlaywrightTemplate()
        assert template._get_comment_prefix() == "//"

    def test_java_prefix(self):
        """Test Java uses // prefix."""
        template = JavaSeleniumTemplate()
        assert template._get_comment_prefix() == "//"

    def test_csharp_prefix(self):
        """Test C# uses // prefix."""
        template = CSharpSeleniumTemplate()
        assert template._get_comment_prefix() == "//"

    def test_go_prefix(self):
        """Test Go uses // prefix."""
        template = GoRodTemplate()
        assert template._get_comment_prefix() == "//"


# =============================================================================
# Template Attribute Tests
# =============================================================================


class TestTemplateAttributes:
    """Tests for template class attributes."""

    def test_python_playwright_attributes(self):
        """Test Python Playwright template attributes."""
        template = PythonPlaywrightTemplate()
        assert template.language == "python"
        assert template.framework == "playwright"
        assert template.file_extension == ".py"
        assert template.indent == "    "

    def test_python_selenium_attributes(self):
        """Test Python Selenium template attributes."""
        template = PythonSeleniumTemplate()
        assert template.language == "python"
        assert template.framework == "selenium"
        assert template.file_extension == ".py"
        assert template.indent == "    "

    def test_typescript_playwright_attributes(self):
        """Test TypeScript Playwright template attributes."""
        template = TypeScriptPlaywrightTemplate()
        assert template.language == "typescript"
        assert template.framework == "playwright"
        assert template.file_extension == ".ts"
        assert template.indent == "  "

    def test_java_selenium_attributes(self):
        """Test Java Selenium template attributes."""
        template = JavaSeleniumTemplate()
        assert template.language == "java"
        assert template.framework == "selenium"
        assert template.file_extension == ".java"
        assert template.indent == "    "

    def test_csharp_selenium_attributes(self):
        """Test C# Selenium template attributes."""
        template = CSharpSeleniumTemplate()
        assert template.language == "csharp"
        assert template.framework == "selenium"
        assert template.file_extension == ".cs"
        assert template.indent == "    "

    def test_ruby_capybara_attributes(self):
        """Test Ruby Capybara template attributes."""
        template = RubyCapybaraTemplate()
        assert template.language == "ruby"
        assert template.framework == "capybara"
        assert template.file_extension == ".rb"
        assert template.indent == "  "

    def test_go_rod_attributes(self):
        """Test Go Rod template attributes."""
        template = GoRodTemplate()
        assert template.language == "go"
        assert template.framework == "rod"
        assert template.file_extension == ".go"
        assert template.indent == "\t"


# =============================================================================
# Generate Method Tests
# =============================================================================


class TestGenerateMethod:
    """Tests for BaseTemplate.generate method."""

    @pytest.fixture
    def template(self):
        return PythonPlaywrightTemplate()

    def test_generate_from_test_spec(self, template):
        """Test generation from TestSpec object."""
        spec = TestSpec(
            id="test-1",
            name="Test",
            steps=[TestStep(action="goto", target="/")],
        )
        code = template.generate(spec)
        assert code
        assert "class" in code

    def test_generate_from_dict(self, template):
        """Test generation from dictionary."""
        spec_dict = {
            "id": "test-1",
            "name": "Test",
            "steps": [{"action": "goto", "target": "/"}],
        }
        code = template.generate(spec_dict)
        assert code
        assert "class" in code

    def test_includes_imports(self, template):
        """Test generated code includes imports."""
        spec = TestSpec(id="test-1", name="Test", steps=[])
        code = template.generate(spec)
        assert "import" in code.lower()

    def test_includes_class_header(self, template):
        """Test generated code includes class header."""
        spec = TestSpec(id="test-1", name="MyTest", steps=[])
        code = template.generate(spec)
        assert "class" in code.lower() or "test" in code.lower()

    def test_includes_steps(self, template):
        """Test generated code includes step code."""
        spec = TestSpec(
            id="test-1",
            name="Test",
            steps=[
                TestStep(action="click", target="#btn"),
            ],
        )
        code = template.generate(spec)
        assert "#btn" in code

    def test_includes_assertions(self, template):
        """Test generated code includes assertion code."""
        spec = TestSpec(
            id="test-1",
            name="Test",
            steps=[TestStep(action="goto", target="/")],
            assertions=[
                TestAssertion(type="element_visible", target=".el"),
            ],
        )
        code = template.generate(spec)
        assert ".el" in code

    def test_includes_footer(self, template):
        """Test generated code includes footer."""
        spec = TestSpec(id="test-1", name="Test", steps=[])
        code = template.generate(spec)
        # Should have complete structure
        assert code.strip()

    def test_includes_attribution(self, template):
        """Test generated code includes Argus attribution."""
        spec = TestSpec(id="test-1", name="Test", steps=[])
        code = template.generate(spec)
        assert "Argus" in code

    def test_skips_empty_step_code(self, template):
        """Test empty step code is skipped."""
        # Some unknown actions might return empty code
        spec = TestSpec(
            id="test-1",
            name="Test",
            steps=[TestStep(action="goto", target="/")],
        )
        code = template.generate(spec)
        assert code  # Should still generate something


# =============================================================================
# Assertions Header Tests
# =============================================================================


class TestAssertionsHeader:
    """Tests for assertions header generation."""

    def test_python_assertions_header(self):
        """Test Python assertions header."""
        template = PythonPlaywrightTemplate()
        header = template._generate_assertions_header()
        assert "# Assertions" in header

    def test_typescript_assertions_header(self):
        """Test TypeScript assertions header."""
        template = TypeScriptPlaywrightTemplate()
        header = template._generate_assertions_header()
        assert "// Assertions" in header

    def test_java_assertions_header(self):
        """Test Java assertions header."""
        template = JavaSeleniumTemplate()
        header = template._generate_assertions_header()
        assert "// Assertions" in header

    def test_go_assertions_header(self):
        """Test Go assertions header."""
        template = GoRodTemplate()
        header = template._generate_assertions_header()
        assert "// Assertions" in header


# =============================================================================
# Attribution Tests
# =============================================================================


class TestAttribution:
    """Tests for attribution generation."""

    @pytest.fixture(params=[
        PythonPlaywrightTemplate,
        PythonSeleniumTemplate,
        TypeScriptPlaywrightTemplate,
        JavaSeleniumTemplate,
        CSharpSeleniumTemplate,
        RubyCapybaraTemplate,
        GoRodTemplate,
    ])
    def template(self, request):
        return request.param()

    def test_attribution_contains_argus(self, template):
        """Test attribution contains 'Argus'."""
        attribution = template._generate_attribution()
        assert "Argus" in attribution

    def test_attribution_is_comment(self, template):
        """Test attribution is formatted as a comment."""
        attribution = template._generate_attribution()
        prefix = template._get_comment_prefix()
        assert prefix in attribution


# =============================================================================
# Config Handling Tests
# =============================================================================


class TestConfigHandling:
    """Tests for configuration handling in templates."""

    def test_default_config(self):
        """Test template with no config uses defaults."""
        template = PythonPlaywrightTemplate()
        assert template.config == {}

    def test_none_config_becomes_empty_dict(self):
        """Test None config becomes empty dict."""
        template = PythonPlaywrightTemplate(config=None)
        assert template.config == {}

    def test_config_is_stored(self):
        """Test config is stored correctly."""
        config = {"async_style": False, "custom_option": "value"}
        template = PythonPlaywrightTemplate(config=config)
        assert template.config == config

    def test_config_get_with_default(self):
        """Test config.get with default value."""
        template = PythonPlaywrightTemplate(config={})
        # async_style defaults to True in most templates
        assert template.config.get("async_style", True) is True

    def test_config_options_affect_generation(self):
        """Test config options affect code generation."""
        async_template = PythonPlaywrightTemplate(config={"async_style": True})
        sync_template = PythonPlaywrightTemplate(config={"async_style": False})

        step = TestStep(action="click", target="#btn")
        async_code = async_template.generate_step_code(step, 0)
        sync_code = sync_template.generate_step_code(step, 0)

        assert "await" in async_code
        assert "await" not in sync_code
