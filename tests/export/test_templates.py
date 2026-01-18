"""Tests for export templates."""

import pytest

from src.export.templates import (
    CSharpSeleniumTemplate,
    GoRodTemplate,
    JavaSeleniumTemplate,
    PythonPlaywrightTemplate,
    PythonSeleniumTemplate,
    RubyCapybaraTemplate,
    TypeScriptPlaywrightTemplate,
)
from src.export.templates.base import TestAssertion, TestSpec, TestStep

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_spec():
    """Standard test spec for template tests."""
    return TestSpec(
        id="test-001",
        name="Login Test",
        description="Tests user login flow",
        steps=[
            TestStep(action="goto", target="/login"),
            TestStep(action="fill", target="#email", value="test@example.com"),
            TestStep(action="fill", target="#password", value="secret123"),
            TestStep(action="click", target="button[type=submit]"),
        ],
        assertions=[
            TestAssertion(type="url_contains", expected="/dashboard"),
            TestAssertion(type="element_visible", target=".welcome-message"),
        ],
    )


@pytest.fixture
def test_spec_dict():
    """Test spec as dictionary."""
    return {
        "id": "test-002",
        "name": "Simple Flow",
        "steps": [
            {"action": "goto", "target": "/home"},
            {"action": "click", "target": "#button"},
        ],
        "assertions": [
            {"type": "element_visible", "target": ".result"},
        ],
    }


@pytest.fixture
def comprehensive_test_spec():
    """Comprehensive test spec with all action types."""
    return TestSpec(
        id="test-comprehensive",
        name="Comprehensive Test",
        description="Tests all action types",
        steps=[
            TestStep(action="goto", target="https://example.com"),
            TestStep(action="goto", target="/relative-path"),
            TestStep(action="click", target="#button"),
            TestStep(action="fill", target="#input", value="text value"),
            TestStep(action="type", target="#typed-input", value="typed text"),
            TestStep(action="select", target="#dropdown", value="option1"),
            TestStep(action="hover", target=".menu-item"),
            TestStep(action="wait", target=".async-element", timeout=5000),
            TestStep(action="scroll", value="0,500"),
            TestStep(action="scroll"),  # Default scroll
            TestStep(action="press_key", value="Enter"),
            TestStep(action="screenshot"),
            TestStep(action="double_click", target=".dbl-click-target"),
            TestStep(action="unknown_action", target="#el"),  # Unknown action
        ],
        assertions=[
            TestAssertion(type="element_visible", target=".visible-el"),
            TestAssertion(type="element_hidden", target=".hidden-el"),
            TestAssertion(type="text_contains", target=".text-el", expected="expected"),
            TestAssertion(type="text_equals", target=".exact-text", expected="exact match"),
            TestAssertion(type="url_contains", expected="/expected-path"),
            TestAssertion(type="url_matches", expected="pattern"),
            TestAssertion(type="value_equals", target="#value-input", expected="input value"),
            TestAssertion(type="title_contains", expected="Page Title"),
            TestAssertion(type="unknown_type", target="#el"),  # Unknown assertion
        ],
    )


# =============================================================================
# TestSpec Tests
# =============================================================================


class TestTestSpec:
    """Tests for TestSpec dataclass."""

    def test_create_test_spec(self):
        """Test creating TestSpec."""
        spec = TestSpec(id="test-1", name="Test")
        assert spec.id == "test-1"
        assert spec.name == "Test"
        assert spec.steps == []
        assert spec.assertions == []

    def test_test_spec_from_dict(self, test_spec_dict):
        """Test creating TestSpec from dictionary."""
        spec = TestSpec.from_dict(test_spec_dict)
        assert spec.id == "test-002"
        assert spec.name == "Simple Flow"
        assert len(spec.steps) == 2
        assert len(spec.assertions) == 1

    def test_test_spec_defaults(self):
        """Test TestSpec defaults for missing fields."""
        spec = TestSpec.from_dict({"name": "Minimal"})
        assert spec.id == "test-001"  # Default
        assert spec.name == "Minimal"
        assert spec.steps == []
        assert spec.assertions == []


# =============================================================================
# BaseTemplate Tests
# =============================================================================


class TestBaseTemplate:
    """Tests for BaseTemplate helper methods."""

    def test_sanitize_name(self):
        """Test name sanitization."""
        template = PythonPlaywrightTemplate()
        assert template.sanitize_name("Test Name!@#") == "test_name___"
        assert template.sanitize_name("123test") == "_123test"
        assert template.sanitize_name("valid_name") == "valid_name"

    def test_to_camel_case(self):
        """Test camelCase conversion."""
        template = PythonPlaywrightTemplate()
        assert template.to_camel_case("test name here") == "testNameHere"
        assert template.to_camel_case("login-flow") == "loginFlow"
        assert template.to_camel_case("single") == "single"

    def test_to_pascal_case(self):
        """Test PascalCase conversion."""
        template = PythonPlaywrightTemplate()
        assert template.to_pascal_case("test name here") == "TestNameHere"
        assert template.to_pascal_case("login-flow") == "LoginFlow"
        assert template.to_pascal_case("") == "Test"

    def test_to_snake_case(self):
        """Test snake_case conversion."""
        template = PythonPlaywrightTemplate()
        assert template.to_snake_case("TestName") == "test_name"
        assert template.to_snake_case("test name here") == "test_name_here"
        assert template.to_snake_case("testName") == "test_name"

    def test_escape_string(self):
        """Test string escaping."""
        template = PythonPlaywrightTemplate()
        assert template.escape_string('test"quote') == 'test\\"quote'
        assert template.escape_string("test\\path") == "test\\\\path"
        assert template.escape_string("test\nnewline") == "test\\nnewline"
        assert template.escape_string(None) == ""

    def test_format_comment(self):
        """Test comment formatting."""
        py_template = PythonPlaywrightTemplate()
        ts_template = TypeScriptPlaywrightTemplate()

        assert py_template.format_comment("test") == "# test"
        assert ts_template.format_comment("test") == "// test"


# =============================================================================
# Python Playwright Template Tests
# =============================================================================


class TestPythonPlaywrightTemplate:
    """Tests for Python Playwright template."""

    @pytest.fixture
    def template(self):
        return PythonPlaywrightTemplate()

    @pytest.fixture
    def async_template(self):
        return PythonPlaywrightTemplate(config={"async_style": True})

    @pytest.fixture
    def sync_template(self):
        return PythonPlaywrightTemplate(config={"async_style": False})

    def test_template_attributes(self, template):
        """Test template has correct attributes."""
        assert template.language == "python"
        assert template.framework == "playwright"
        assert template.file_extension == ".py"
        assert template.indent == "    "

    def test_generate_imports_async(self, async_template, test_spec):
        """Test async imports generation."""
        imports = async_template.generate_imports(test_spec)
        assert "import pytest" in imports
        assert "playwright.async_api" in imports
        assert "import asyncio" in imports

    def test_generate_imports_sync(self, sync_template, test_spec):
        """Test sync imports generation."""
        imports = sync_template.generate_imports(test_spec)
        assert "import pytest" in imports
        assert "playwright.sync_api" in imports
        assert "asyncio" not in imports

    def test_generate_class_header(self, template, test_spec):
        """Test class header generation."""
        header = template.generate_class_header(test_spec)
        assert "class TestLoginTest" in header
        assert "test-001" in header
        assert "test_login_test" in header

    def test_generate_step_goto(self, async_template):
        """Test goto step generation."""
        step = TestStep(action="goto", target="/login")
        code = async_template.generate_step_code(step, 0)
        assert "await page.goto" in code
        assert "/login" in code

    def test_generate_step_goto_absolute(self, async_template):
        """Test goto step with absolute URL."""
        step = TestStep(action="goto", target="https://example.com")
        code = async_template.generate_step_code(step, 0)
        assert 'await page.goto("https://example.com")' in code

    def test_generate_step_click(self, async_template):
        """Test click step generation."""
        step = TestStep(action="click", target="#button")
        code = async_template.generate_step_code(step, 0)
        assert 'await page.click("#button")' in code

    def test_generate_step_fill(self, async_template):
        """Test fill step generation."""
        step = TestStep(action="fill", target="#input", value="text")
        code = async_template.generate_step_code(step, 0)
        assert 'await page.fill("#input", "text")' in code

    def test_generate_step_type(self, async_template):
        """Test type step generation."""
        step = TestStep(action="type", target="#input", value="typed")
        code = async_template.generate_step_code(step, 0)
        assert 'await page.type("#input", "typed")' in code

    def test_generate_step_select(self, async_template):
        """Test select step generation."""
        step = TestStep(action="select", target="#dropdown", value="opt1")
        code = async_template.generate_step_code(step, 0)
        assert 'await page.select_option("#dropdown", "opt1")' in code

    def test_generate_step_hover(self, async_template):
        """Test hover step generation."""
        step = TestStep(action="hover", target=".menu")
        code = async_template.generate_step_code(step, 0)
        assert 'await page.hover(".menu")' in code

    def test_generate_step_wait(self, async_template):
        """Test wait step generation."""
        step = TestStep(action="wait", target=".async-el", timeout=5000)
        code = async_template.generate_step_code(step, 0)
        assert "await page.wait_for_selector" in code
        assert "timeout=5000" in code

    def test_generate_step_scroll(self, async_template):
        """Test scroll step generation."""
        step = TestStep(action="scroll", value="0,500")
        code = async_template.generate_step_code(step, 0)
        assert "scrollBy(0, 500)" in code

    def test_generate_step_press_key(self, async_template):
        """Test press_key step generation."""
        step = TestStep(action="press_key", value="Enter")
        code = async_template.generate_step_code(step, 0)
        assert 'await page.keyboard.press("Enter")' in code

    def test_generate_step_screenshot(self, async_template):
        """Test screenshot step generation."""
        step = TestStep(action="screenshot")
        code = async_template.generate_step_code(step, 3)
        assert "await page.screenshot" in code
        assert "screenshot_3.png" in code

    def test_generate_step_double_click(self, async_template):
        """Test double_click step generation."""
        step = TestStep(action="double_click", target=".dbl")
        code = async_template.generate_step_code(step, 0)
        assert 'await page.dblclick(".dbl")' in code

    def test_generate_assertion_element_visible(self, async_template):
        """Test element_visible assertion."""
        assertion = TestAssertion(type="element_visible", target=".el")
        code = async_template.generate_assertion_code(assertion)
        assert "expect(page.locator" in code
        assert "to_be_visible" in code

    def test_generate_assertion_element_hidden(self, async_template):
        """Test element_hidden assertion."""
        assertion = TestAssertion(type="element_hidden", target=".el")
        code = async_template.generate_assertion_code(assertion)
        assert "to_be_hidden" in code

    def test_generate_assertion_text_contains(self, async_template):
        """Test text_contains assertion."""
        assertion = TestAssertion(type="text_contains", target=".el", expected="text")
        code = async_template.generate_assertion_code(assertion)
        assert "to_contain_text" in code

    def test_generate_assertion_text_equals(self, async_template):
        """Test text_equals assertion."""
        assertion = TestAssertion(type="text_equals", target=".el", expected="exact")
        code = async_template.generate_assertion_code(assertion)
        assert "to_have_text" in code

    def test_generate_assertion_url_contains(self, async_template):
        """Test url_contains assertion."""
        assertion = TestAssertion(type="url_contains", expected="/path")
        code = async_template.generate_assertion_code(assertion)
        assert "to_have_url" in code
        assert "re.compile" in code

    def test_generate_assertion_value_equals(self, async_template):
        """Test value_equals assertion."""
        assertion = TestAssertion(type="value_equals", target="#input", expected="val")
        code = async_template.generate_assertion_code(assertion)
        assert "to_have_value" in code

    def test_generate_assertion_title_contains(self, async_template):
        """Test title_contains assertion."""
        assertion = TestAssertion(type="title_contains", expected="Title")
        code = async_template.generate_assertion_code(assertion)
        assert "to_have_title" in code

    def test_generate_complete_test(self, async_template, test_spec):
        """Test complete test generation."""
        code = async_template.generate(test_spec)
        assert "import pytest" in code
        assert "class TestLoginTest" in code
        assert "async def test_login_test" in code
        assert 'page.goto' in code
        assert 'page.fill("#email"' in code
        assert 'page.click("button[type=submit]")' in code
        assert "to_have_url" in code
        assert "to_be_visible" in code
        assert "Argus" in code

    def test_generate_from_dict(self, async_template, test_spec_dict):
        """Test generation from dictionary."""
        code = async_template.generate(test_spec_dict)
        assert "class TestSimpleFlow" in code
        assert 'page.goto' in code
        assert 'page.click("#button")' in code


# =============================================================================
# Python Selenium Template Tests
# =============================================================================


class TestPythonSeleniumTemplate:
    """Tests for Python Selenium template."""

    @pytest.fixture
    def template(self):
        return PythonSeleniumTemplate()

    def test_template_attributes(self, template):
        """Test template has correct attributes."""
        assert template.language == "python"
        assert template.framework == "selenium"
        assert template.file_extension == ".py"

    def test_generate_imports(self, template, test_spec):
        """Test imports generation."""
        imports = template.generate_imports(test_spec)
        assert "selenium" in imports
        assert "webdriver" in imports
        assert "By" in imports

    def test_generate_step_click(self, template):
        """Test click step generation."""
        step = TestStep(action="click", target="#button")
        code = template.generate_step_code(step, 0)
        # Selenium template uses explicit waits (better practice)
        assert "wait.until" in code or "find_element" in code
        assert "click()" in code

    def test_generate_step_fill(self, template):
        """Test fill step generation."""
        step = TestStep(action="fill", target="#input", value="text")
        code = template.generate_step_code(step, 0)
        assert "send_keys" in code
        assert "clear()" in code

    def test_generate_complete_test(self, template, test_spec):
        """Test complete test generation."""
        code = template.generate(test_spec)
        assert "selenium" in code
        assert "class TestLoginTest" in code


# =============================================================================
# TypeScript Playwright Template Tests
# =============================================================================


class TestTypeScriptPlaywrightTemplate:
    """Tests for TypeScript Playwright template."""

    @pytest.fixture
    def template(self):
        return TypeScriptPlaywrightTemplate()

    def test_template_attributes(self, template):
        """Test template has correct attributes."""
        assert template.language == "typescript"
        assert template.framework == "playwright"
        assert template.file_extension == ".ts"
        assert template.indent == "  "

    def test_generate_imports(self, template, test_spec):
        """Test imports generation."""
        imports = template.generate_imports(test_spec)
        assert "import { test, expect }" in imports
        assert "@playwright/test" in imports

    def test_generate_class_header(self, template, test_spec):
        """Test class header generation."""
        header = template.generate_class_header(test_spec)
        assert "test.describe('Login Test'" in header
        assert "test('Login Test'" in header
        assert "async" in header
        assert "page, baseURL" in header

    def test_generate_step_click(self, template):
        """Test click step generation."""
        step = TestStep(action="click", target="#button")
        code = template.generate_step_code(step, 0)
        assert "await page.click('#button');" in code

    def test_generate_step_fill(self, template):
        """Test fill step generation."""
        step = TestStep(action="fill", target="#input", value="text")
        code = template.generate_step_code(step, 0)
        assert "await page.fill('#input', 'text');" in code

    def test_generate_assertion_element_visible(self, template):
        """Test element_visible assertion."""
        assertion = TestAssertion(type="element_visible", target=".el")
        code = template.generate_assertion_code(assertion)
        assert "await expect(page.locator('.el')).toBeVisible();" in code

    def test_generate_footer(self, template):
        """Test footer generation."""
        footer = template.generate_class_footer()
        assert "});" in footer

    def test_generate_complete_test(self, template, test_spec):
        """Test complete test generation."""
        code = template.generate(test_spec)
        assert "import { test, expect }" in code
        assert "test.describe" in code
        assert "await page.goto" in code
        assert "toBeVisible" in code


# =============================================================================
# Java Selenium Template Tests
# =============================================================================


class TestJavaSeleniumTemplate:
    """Tests for Java Selenium template."""

    @pytest.fixture
    def template(self):
        return JavaSeleniumTemplate()

    def test_template_attributes(self, template):
        """Test template has correct attributes."""
        assert template.language == "java"
        assert template.framework == "selenium"
        assert template.file_extension == ".java"

    def test_generate_imports(self, template, test_spec):
        """Test imports generation."""
        imports = template.generate_imports(test_spec)
        assert "org.openqa.selenium" in imports
        assert "org.junit" in imports

    def test_generate_class_header(self, template, test_spec):
        """Test class header generation."""
        header = template.generate_class_header(test_spec)
        assert "public class TestLoginTest" in header
        assert "@Before" in header or "@BeforeEach" in header

    def test_generate_step_click(self, template):
        """Test click step generation."""
        step = TestStep(action="click", target="#button")
        code = template.generate_step_code(step, 0)
        # Java template uses explicit waits (better practice)
        assert "wait.until" in code or "findElement" in code
        assert ".click()" in code

    def test_generate_complete_test(self, template, test_spec):
        """Test complete test generation."""
        code = template.generate(test_spec)
        assert "import org.openqa.selenium" in code
        assert "public class" in code
        assert "WebDriver" in code


# =============================================================================
# C# Selenium Template Tests
# =============================================================================


class TestCSharpSeleniumTemplate:
    """Tests for C# Selenium template."""

    @pytest.fixture
    def template(self):
        return CSharpSeleniumTemplate()

    def test_template_attributes(self, template):
        """Test template has correct attributes."""
        assert template.language == "csharp"
        assert template.framework == "selenium"
        assert template.file_extension == ".cs"

    def test_generate_imports(self, template, test_spec):
        """Test imports generation."""
        imports = template.generate_imports(test_spec)
        assert "using OpenQA.Selenium" in imports
        assert "using NUnit" in imports

    def test_generate_class_header(self, template, test_spec):
        """Test class header generation."""
        header = template.generate_class_header(test_spec)
        assert "public class TestLoginTest" in header
        assert "[SetUp]" in header

    def test_generate_step_click(self, template):
        """Test click step generation."""
        step = TestStep(action="click", target="#button")
        code = template.generate_step_code(step, 0)
        # C# template uses explicit waits (better practice)
        assert "_wait.Until" in code or "FindElement" in code
        assert ".Click()" in code

    def test_generate_complete_test(self, template, test_spec):
        """Test complete test generation."""
        code = template.generate(test_spec)
        assert "using OpenQA.Selenium" in code
        assert "public class" in code
        assert "IWebDriver" in code


# =============================================================================
# Ruby Capybara Template Tests
# =============================================================================


class TestRubyCapybaraTemplate:
    """Tests for Ruby Capybara template."""

    @pytest.fixture
    def template(self):
        return RubyCapybaraTemplate()

    def test_template_attributes(self, template):
        """Test template has correct attributes."""
        assert template.language == "ruby"
        assert template.framework == "capybara"
        assert template.file_extension == ".rb"

    def test_generate_imports(self, template, test_spec):
        """Test imports generation."""
        imports = template.generate_imports(test_spec)
        assert "require" in imports
        assert "capybara" in imports

    def test_generate_class_header(self, template, test_spec):
        """Test class header generation."""
        header = template.generate_class_header(test_spec)
        assert "describe" in header or "RSpec.describe" in header

    def test_generate_step_click(self, template):
        """Test click step generation."""
        step = TestStep(action="click", target="#button")
        code = template.generate_step_code(step, 0)
        assert "click" in code or "find" in code

    def test_generate_complete_test(self, template, test_spec):
        """Test complete test generation."""
        code = template.generate(test_spec)
        assert "require" in code
        assert "describe" in code


# =============================================================================
# Go Rod Template Tests
# =============================================================================


class TestGoRodTemplate:
    """Tests for Go Rod template."""

    @pytest.fixture
    def template(self):
        return GoRodTemplate()

    def test_template_attributes(self, template):
        """Test template has correct attributes."""
        assert template.language == "go"
        assert template.framework == "rod"
        assert template.file_extension == ".go"
        assert template.indent == "\t"

    def test_generate_imports(self, template, test_spec):
        """Test imports generation."""
        imports = template.generate_imports(test_spec)
        assert "package tests" in imports
        assert "github.com/go-rod/rod" in imports
        assert "testing" in imports

    def test_generate_class_header(self, template, test_spec):
        """Test class header generation."""
        header = template.generate_class_header(test_spec)
        assert "func TestLoginTest" in header
        assert "*testing.T" in header
        assert "rod.New()" in header

    def test_generate_step_click(self, template):
        """Test click step generation."""
        step = TestStep(action="click", target="#button")
        code = template.generate_step_code(step, 0)
        assert "MustElement" in code
        assert "MustClick()" in code

    def test_generate_step_fill(self, template):
        """Test fill step generation."""
        step = TestStep(action="fill", target="#input", value="text")
        code = template.generate_step_code(step, 0)
        assert "MustInput" in code
        assert "MustSelectAllText" in code

    def test_generate_assertion_element_visible(self, template):
        """Test element_visible assertion."""
        assertion = TestAssertion(type="element_visible", target=".el")
        code = template.generate_assertion_code(assertion)
        assert "assert.True" in code
        assert "MustVisible()" in code

    def test_generate_footer(self, template):
        """Test footer generation."""
        footer = template.generate_class_footer()
        assert "}" in footer

    def test_generate_complete_test(self, template, test_spec):
        """Test complete test generation."""
        code = template.generate(test_spec)
        assert "package tests" in code
        assert "github.com/go-rod/rod" in code
        assert "func TestLoginTest" in code
        assert "MustNavigate" in code
        assert "assert" in code


# =============================================================================
# Comprehensive Template Tests
# =============================================================================


class TestComprehensiveTemplates:
    """Tests with comprehensive test specs covering all action types."""

    @pytest.mark.parametrize("template_class", [
        PythonPlaywrightTemplate,
        PythonSeleniumTemplate,
        TypeScriptPlaywrightTemplate,
        JavaSeleniumTemplate,
        CSharpSeleniumTemplate,
        RubyCapybaraTemplate,
        GoRodTemplate,
    ])
    def test_all_templates_generate_valid_code(self, template_class, comprehensive_test_spec):
        """Test all templates can generate code from comprehensive spec."""
        template = template_class()
        code = template.generate(comprehensive_test_spec)

        assert code is not None
        assert len(code) > 100  # Should generate substantial code
        assert "Argus" in code  # Attribution

        # Should handle all steps (even unknown)
        assert "unknown" in code.lower()  # Unknown action handled

    @pytest.mark.parametrize("template_class", [
        PythonPlaywrightTemplate,
        TypeScriptPlaywrightTemplate,
        GoRodTemplate,
    ])
    def test_step_comments(self, template_class, test_spec):
        """Test step comments are generated."""
        template = template_class()
        code = template.generate(test_spec)

        assert "Step 1" in code
        assert "Step 2" in code

    @pytest.mark.parametrize("template_class,expected_keywords", [
        (PythonPlaywrightTemplate, ["import", "class", "def", "page"]),
        (PythonSeleniumTemplate, ["import", "class", "def", "driver"]),
        (TypeScriptPlaywrightTemplate, ["import", "test", "expect", "page"]),
        (JavaSeleniumTemplate, ["import", "class", "public", "driver"]),
        (CSharpSeleniumTemplate, ["using", "class", "public", "driver"]),
        (RubyCapybaraTemplate, ["require", "describe", "visit"]),
        (GoRodTemplate, ["package", "func", "Test", "page"]),
    ])
    def test_template_keywords(self, template_class, expected_keywords, test_spec):
        """Test templates include expected language keywords."""
        template = template_class()
        code = template.generate(test_spec)

        for keyword in expected_keywords:
            assert keyword in code, f"Missing keyword '{keyword}' in {template_class.__name__}"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in templates."""

    def test_empty_test_spec(self):
        """Test handling of empty test spec."""
        template = PythonPlaywrightTemplate()
        spec = TestSpec(id="empty", name="Empty Test")
        code = template.generate(spec)
        assert "class TestEmptyTest" in code

    def test_special_characters_in_name(self):
        """Test handling of special characters in test name."""
        template = PythonPlaywrightTemplate()
        spec = TestSpec(
            id="test-special",
            name="Test with Special!@#$%^&*() Characters",
            steps=[TestStep(action="goto", target="/home")],
        )
        code = template.generate(spec)
        assert "class Test" in code  # Should have a valid class name
        assert "def test_" in code  # Should have a valid method name

    def test_quotes_in_values(self):
        """Test handling of quotes in values."""
        template = PythonPlaywrightTemplate()
        step = TestStep(action="fill", target="#input", value='value with "quotes"')
        code = template.generate_step_code(step, 0)
        assert '\\"' in code  # Quotes should be escaped

    def test_newlines_in_values(self):
        """Test handling of newlines in values."""
        template = PythonPlaywrightTemplate()
        step = TestStep(action="fill", target="#textarea", value="line1\nline2")
        code = template.generate_step_code(step, 0)
        assert "\\n" in code  # Newlines should be escaped

    def test_none_values(self):
        """Test handling of None values."""
        template = PythonPlaywrightTemplate()
        step = TestStep(action="click", target=None)
        code = template.generate_step_code(step, 0)
        assert code  # Should not raise error

    def test_long_test_name(self):
        """Test handling of very long test names."""
        template = PythonPlaywrightTemplate()
        long_name = "This is a very long test name " * 10
        spec = TestSpec(id="long", name=long_name)
        code = template.generate(spec)
        assert "class Test" in code

    def test_unicode_in_values(self):
        """Test handling of unicode characters."""
        template = PythonPlaywrightTemplate()
        step = TestStep(action="fill", target="#input", value="日本語テスト")
        code = template.generate_step_code(step, 0)
        assert code  # Should not raise error
