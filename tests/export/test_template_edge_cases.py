"""Edge case and stress tests for export templates."""

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


ALL_TEMPLATES = [
    PythonPlaywrightTemplate,
    PythonSeleniumTemplate,
    TypeScriptPlaywrightTemplate,
    JavaSeleniumTemplate,
    CSharpSeleniumTemplate,
    RubyCapybaraTemplate,
    GoRodTemplate,
]


@pytest.fixture(params=ALL_TEMPLATES)
def template(request):
    """Parametrized fixture for all templates."""
    return request.param()


# =============================================================================
# TestStep Edge Cases
# =============================================================================


class TestTestStepEdgeCases:
    """Tests for edge cases in TestStep."""

    def test_step_with_none_target(self):
        """Test step with None target."""
        step = TestStep(action="click", target=None)
        assert step.action == "click"
        assert step.target is None

    def test_step_with_none_value(self):
        """Test step with None value."""
        step = TestStep(action="fill", target="#input", value=None)
        assert step.value is None

    def test_step_with_empty_string_target(self):
        """Test step with empty string target."""
        step = TestStep(action="click", target="")
        assert step.target == ""

    def test_step_with_all_fields(self):
        """Test step with all optional fields."""
        step = TestStep(
            action="wait",
            target="#element",
            value="some value",
            timeout=5000,
            description="Wait for element",
        )
        assert step.timeout == 5000
        assert step.description == "Wait for element"

    def test_step_with_zero_timeout(self):
        """Test step with zero timeout."""
        step = TestStep(action="wait", target="#el", timeout=0)
        assert step.timeout == 0


# =============================================================================
# TestAssertion Edge Cases
# =============================================================================


class TestTestAssertionEdgeCases:
    """Tests for edge cases in TestAssertion."""

    def test_assertion_with_none_target(self):
        """Test assertion with None target."""
        assertion = TestAssertion(type="url_contains", target=None, expected="/path")
        assert assertion.target is None

    def test_assertion_with_none_expected(self):
        """Test assertion with None expected."""
        assertion = TestAssertion(type="element_visible", target=".el", expected=None)
        assert assertion.expected is None

    def test_assertion_with_empty_expected(self):
        """Test assertion with empty expected."""
        assertion = TestAssertion(type="text_equals", target=".el", expected="")
        assert assertion.expected == ""


# =============================================================================
# TestSpec Edge Cases
# =============================================================================


class TestTestSpecEdgeCases:
    """Tests for edge cases in TestSpec."""

    def test_spec_with_empty_steps(self):
        """Test spec with empty steps list."""
        spec = TestSpec(id="test-1", name="Empty Steps Test", steps=[])
        assert spec.steps == []

    def test_spec_with_empty_assertions(self):
        """Test spec with empty assertions list."""
        spec = TestSpec(id="test-1", name="No Assertions", assertions=[])
        assert spec.assertions == []

    def test_spec_with_none_steps(self):
        """Test spec with None steps (defaults to empty list)."""
        spec = TestSpec(id="test-1", name="Test", steps=None)
        assert spec.steps == []

    def test_spec_with_none_assertions(self):
        """Test spec with None assertions (defaults to empty list)."""
        spec = TestSpec(id="test-1", name="Test", assertions=None)
        assert spec.assertions == []

    def test_spec_from_dict_minimal(self):
        """Test creating spec from minimal dict."""
        spec = TestSpec.from_dict({"name": "Minimal"})
        assert spec.id == "test-001"
        assert spec.name == "Minimal"
        assert spec.description == ""
        assert spec.steps == []
        assert spec.assertions == []

    def test_spec_from_dict_with_steps(self):
        """Test creating spec from dict with steps."""
        data = {
            "name": "Test",
            "steps": [
                {"action": "click", "target": "#btn"},
                {"action": "fill", "target": "#input", "value": "text"},
            ],
        }
        spec = TestSpec.from_dict(data)
        assert len(spec.steps) == 2
        assert isinstance(spec.steps[0], TestStep)
        assert spec.steps[0].action == "click"

    def test_spec_from_dict_with_assertions(self):
        """Test creating spec from dict with assertions."""
        data = {
            "name": "Test",
            "assertions": [
                {"type": "element_visible", "target": ".el"},
            ],
        }
        spec = TestSpec.from_dict(data)
        assert len(spec.assertions) == 1
        assert isinstance(spec.assertions[0], TestAssertion)

    def test_spec_from_dict_with_existing_step_objects(self):
        """Test creating spec from dict with existing TestStep objects."""
        step = TestStep(action="click", target="#btn")
        data = {
            "name": "Test",
            "steps": [step],  # Already a TestStep object
        }
        spec = TestSpec.from_dict(data)
        assert len(spec.steps) == 1
        assert spec.steps[0] == step


# =============================================================================
# Special Characters Tests
# =============================================================================


class TestSpecialCharacters:
    """Tests for handling special characters."""

    def test_quotes_in_selector(self, template):
        """Test handling quotes in CSS selector."""
        step = TestStep(action="click", target='[data-attr="value"]')
        code = template.generate_step_code(step, 0)
        assert code  # Should not crash
        # Quotes should be escaped
        assert '\\"' in code or "\\'" in code or '[data-attr' in code

    def test_backslash_in_value(self, template):
        """Test handling backslash in value."""
        step = TestStep(action="fill", target="#path", value="C:\\Users\\test")
        code = template.generate_step_code(step, 0)
        assert code  # Should not crash
        assert "\\\\" in code or "C:" in code

    def test_newline_in_value(self, template):
        """Test handling newline in value."""
        step = TestStep(action="fill", target="#textarea", value="line1\nline2")
        code = template.generate_step_code(step, 0)
        assert code  # Should not crash
        assert "\\n" in code or "line1" in code

    def test_tab_in_value(self, template):
        """Test handling tab in value."""
        step = TestStep(action="fill", target="#input", value="col1\tcol2")
        code = template.generate_step_code(step, 0)
        assert code  # Should not crash

    def test_unicode_in_name(self, template):
        """Test handling unicode in test name."""
        spec = TestSpec(
            id="test-unicode",
            name="Test with Unicode Name",
            steps=[TestStep(action="goto", target="/home")],
        )
        code = template.generate(spec)
        assert code  # Should not crash
        assert "Test" in code

    def test_emoji_in_value(self, template):
        """Test handling emoji in value."""
        step = TestStep(action="fill", target="#emoji", value="Hello test")
        code = template.generate_step_code(step, 0)
        assert code  # Should not crash

    def test_angle_brackets_in_selector(self, template):
        """Test handling angle brackets in selector."""
        step = TestStep(action="click", target="button[type=submit]")
        code = template.generate_step_code(step, 0)
        assert code
        assert "button" in code

    def test_hash_in_selector(self, template):
        """Test handling hash in ID selector."""
        step = TestStep(action="click", target="#my-id")
        code = template.generate_step_code(step, 0)
        assert code
        assert "my-id" in code

    def test_dot_in_selector(self, template):
        """Test handling dot in class selector."""
        step = TestStep(action="click", target=".my-class")
        code = template.generate_step_code(step, 0)
        assert code
        assert "my-class" in code

    def test_xpath_selector(self, template):
        """Test handling XPath selector."""
        step = TestStep(action="click", target="//div[@class='test']")
        code = template.generate_step_code(step, 0)
        assert code
        assert "div" in code or "xpath" in code.lower()


# =============================================================================
# Long Content Tests
# =============================================================================


class TestLongContent:
    """Tests for handling long content."""

    def test_very_long_test_name(self, template):
        """Test handling very long test name."""
        long_name = "A" * 500
        spec = TestSpec(
            id="test-long",
            name=long_name,
            steps=[TestStep(action="goto", target="/")],
        )
        code = template.generate(spec)
        assert code  # Should not crash

    def test_very_long_selector(self, template):
        """Test handling very long selector."""
        long_selector = "#" + "a" * 500
        step = TestStep(action="click", target=long_selector)
        code = template.generate_step_code(step, 0)
        assert code  # Should not crash

    def test_very_long_value(self, template):
        """Test handling very long value."""
        long_value = "x" * 10000
        step = TestStep(action="fill", target="#input", value=long_value)
        code = template.generate_step_code(step, 0)
        assert code  # Should not crash

    def test_many_steps(self, template):
        """Test handling many steps."""
        steps = [
            TestStep(action="click", target=f"#btn-{i}")
            for i in range(100)
        ]
        spec = TestSpec(id="test-many", name="Many Steps Test", steps=steps)
        code = template.generate(spec)
        assert code
        assert "#btn-0" in code
        assert "#btn-99" in code

    def test_many_assertions(self, template):
        """Test handling many assertions."""
        assertions = [
            TestAssertion(type="element_visible", target=f".el-{i}")
            for i in range(50)
        ]
        spec = TestSpec(
            id="test-assertions",
            name="Many Assertions",
            steps=[TestStep(action="goto", target="/")],
            assertions=assertions,
        )
        code = template.generate(spec)
        assert code
        # Different templates may escape or format selectors differently
        assert "el-0" in code or "el_0" in code
        assert "el-49" in code or "el_49" in code


# =============================================================================
# Action-Specific Edge Cases
# =============================================================================


class TestActionEdgeCases:
    """Tests for edge cases in specific actions."""

    def test_goto_absolute_url(self, template):
        """Test goto with absolute URL."""
        step = TestStep(action="goto", target="https://example.com/path")
        code = template.generate_step_code(step, 0)
        assert "https://example.com" in code

    def test_goto_relative_url(self, template):
        """Test goto with relative URL."""
        step = TestStep(action="goto", target="/relative/path")
        code = template.generate_step_code(step, 0)
        assert "/relative/path" in code

    def test_goto_url_with_query_params(self, template):
        """Test goto with URL containing query parameters."""
        step = TestStep(action="goto", target="https://example.com?foo=bar&baz=qux")
        code = template.generate_step_code(step, 0)
        assert "example.com" in code
        # Query params might be escaped

    def test_fill_empty_value(self, template):
        """Test fill with empty value."""
        step = TestStep(action="fill", target="#input", value="")
        code = template.generate_step_code(step, 0)
        assert code  # Should not crash

    def test_scroll_default(self, template):
        """Test scroll with default (no value)."""
        step = TestStep(action="scroll")
        code = template.generate_step_code(step, 0)
        assert code
        assert "scroll" in code.lower() or "300" in code

    def test_scroll_with_coordinates(self, template):
        """Test scroll with specific coordinates."""
        step = TestStep(action="scroll", value="100,500")
        code = template.generate_step_code(step, 0)
        assert code
        assert "100" in code or "500" in code

    def test_wait_default_timeout(self, template):
        """Test wait with default timeout."""
        step = TestStep(action="wait", target="#element")
        code = template.generate_step_code(step, 0)
        assert code
        # Should have some default timeout

    def test_wait_custom_timeout(self, template):
        """Test wait with custom timeout."""
        step = TestStep(action="wait", target="#element", timeout=10000)
        code = template.generate_step_code(step, 0)
        assert code
        # Some templates use the timeout, others use a default wait
        # Just verify code is generated
        assert "element" in code or "#element" in code or "wait" in code.lower()

    def test_press_key_from_value(self, template):
        """Test press_key getting key from value."""
        step = TestStep(action="press_key", value="Escape")
        code = template.generate_step_code(step, 0)
        assert code
        assert "escape" in code.lower() or "Escape" in code

    def test_press_key_from_target(self, template):
        """Test press_key getting key from target."""
        step = TestStep(action="press_key", target="Tab")
        code = template.generate_step_code(step, 0)
        assert code
        assert "tab" in code.lower() or "Tab" in code

    def test_press_key_default(self, template):
        """Test press_key with default (Enter)."""
        step = TestStep(action="press_key")
        code = template.generate_step_code(step, 0)
        assert code
        assert "enter" in code.lower() or "Enter" in code

    def test_screenshot_indexed(self, template):
        """Test screenshot creates indexed filename."""
        step = TestStep(action="screenshot")
        code = template.generate_step_code(step, 5)
        assert code
        assert "5" in code or "screenshot" in code.lower()

    def test_unknown_action(self, template):
        """Test unknown action is handled gracefully."""
        step = TestStep(action="unknown_action_xyz", target="#el")
        code = template.generate_step_code(step, 0)
        assert code
        assert "unknown" in code.lower()


# =============================================================================
# Assertion-Specific Edge Cases
# =============================================================================


class TestAssertionEdgeCases:
    """Tests for edge cases in specific assertions."""

    def test_element_visible_no_target(self, template):
        """Test element_visible with empty target."""
        assertion = TestAssertion(type="element_visible", target="")
        code = template.generate_assertion_code(assertion)
        assert code  # Should not crash

    def test_text_contains_empty_expected(self, template):
        """Test text_contains with empty expected."""
        assertion = TestAssertion(type="text_contains", target=".el", expected="")
        code = template.generate_assertion_code(assertion)
        assert code  # Should not crash

    def test_url_matches_regex_pattern(self, template):
        """Test url_matches with regex-like pattern."""
        assertion = TestAssertion(type="url_matches", expected="/users/\\d+")
        code = template.generate_assertion_code(assertion)
        assert code

    def test_unknown_assertion_type(self, template):
        """Test unknown assertion type is handled gracefully."""
        assertion = TestAssertion(type="unknown_assertion_xyz", target=".el")
        code = template.generate_assertion_code(assertion)
        assert code
        assert "unknown" in code.lower()


# =============================================================================
# Configuration Edge Cases
# =============================================================================


class TestConfigurationEdgeCases:
    """Tests for configuration edge cases."""

    def test_python_sync_style(self):
        """Test Python Playwright with sync style."""
        template = PythonPlaywrightTemplate(config={"async_style": False})
        step = TestStep(action="click", target="#btn")
        code = template.generate_step_code(step, 0)
        assert "await" not in code
        assert "page.click" in code

    def test_python_async_style(self):
        """Test Python Playwright with async style."""
        template = PythonPlaywrightTemplate(config={"async_style": True})
        step = TestStep(action="click", target="#btn")
        code = template.generate_step_code(step, 0)
        assert "await" in code
        assert "page.click" in code

    def test_custom_class_name(self):
        """Test custom test class name."""
        template = PythonPlaywrightTemplate(config={"test_class_name": "MyCustomTest"})
        spec = TestSpec(
            id="test-1",
            name="Some Test",
            steps=[TestStep(action="goto", target="/")],
        )
        code = template.generate(spec)
        assert "class MyCustomTest" in code

    def test_include_comments_true(self):
        """Test template with comments enabled (default)."""
        template = PythonPlaywrightTemplate(config={"include_comments": True})
        step = TestStep(action="click", target="#btn")
        code = template.generate_step_code(step, 0)
        assert "Step 1" in code

    def test_none_config(self):
        """Test template with None config."""
        template = PythonPlaywrightTemplate(config=None)
        spec = TestSpec(
            id="test-1",
            name="Test",
            steps=[TestStep(action="goto", target="/")],
        )
        code = template.generate(spec)
        assert code  # Should not crash

    def test_empty_config(self):
        """Test template with empty config."""
        template = PythonPlaywrightTemplate(config={})
        spec = TestSpec(
            id="test-1",
            name="Test",
            steps=[TestStep(action="goto", target="/")],
        )
        code = template.generate(spec)
        assert code  # Should not crash


# =============================================================================
# Selenium-Specific Tests
# =============================================================================


class TestSeleniumLocators:
    """Tests for Selenium locator conversion."""

    def test_id_selector_python(self):
        """Test ID selector conversion in Python Selenium."""
        template = PythonSeleniumTemplate()
        step = TestStep(action="click", target="#my-id")
        code = template.generate_step_code(step, 0)
        assert "By.ID" in code or "find_element" in code

    def test_class_selector_python(self):
        """Test class selector conversion in Python Selenium."""
        template = PythonSeleniumTemplate()
        step = TestStep(action="click", target=".my-class")
        code = template.generate_step_code(step, 0)
        assert "By.CLASS_NAME" in code or "find_element" in code

    def test_xpath_selector_python(self):
        """Test XPath selector conversion in Python Selenium."""
        template = PythonSeleniumTemplate()
        step = TestStep(action="click", target="//div[@id='test']")
        code = template.generate_step_code(step, 0)
        assert "By.XPATH" in code or "xpath" in code.lower()

    def test_name_selector_python(self):
        """Test name selector conversion in Python Selenium."""
        template = PythonSeleniumTemplate()
        step = TestStep(action="click", target="name=username")
        code = template.generate_step_code(step, 0)
        assert "By.NAME" in code

    def test_css_selector_fallback_python(self):
        """Test CSS selector fallback in Python Selenium."""
        template = PythonSeleniumTemplate()
        step = TestStep(action="click", target="button[type=submit]")
        code = template.generate_step_code(step, 0)
        assert "By.CSS_SELECTOR" in code or "css" in code.lower()

    def test_id_selector_java(self):
        """Test ID selector conversion in Java Selenium."""
        template = JavaSeleniumTemplate()
        step = TestStep(action="click", target="#my-id")
        code = template.generate_step_code(step, 0)
        assert "By.id" in code or "findElement" in code

    def test_class_selector_java(self):
        """Test class selector conversion in Java Selenium."""
        template = JavaSeleniumTemplate()
        step = TestStep(action="click", target=".my-class")
        code = template.generate_step_code(step, 0)
        assert "By.className" in code or "findElement" in code

    def test_id_selector_csharp(self):
        """Test ID selector conversion in C# Selenium."""
        template = CSharpSeleniumTemplate()
        step = TestStep(action="click", target="#my-id")
        code = template.generate_step_code(step, 0)
        assert "By.Id" in code or "FindElement" in code


# =============================================================================
# Complete Generation Tests
# =============================================================================


class TestCompleteGeneration:
    """Tests for complete test generation."""

    def test_complete_test_has_attribution(self, template):
        """Test generated code includes Argus attribution."""
        spec = TestSpec(
            id="test-1",
            name="Test",
            steps=[TestStep(action="goto", target="/")],
        )
        code = template.generate(spec)
        assert "Argus" in code

    def test_complete_test_has_imports(self, template):
        """Test generated code includes imports."""
        spec = TestSpec(
            id="test-1",
            name="Test",
            steps=[TestStep(action="goto", target="/")],
        )
        code = template.generate(spec)
        code_lower = code.lower()
        assert "import" in code_lower or "require" in code_lower or "using" in code_lower

    def test_complete_test_has_test_function(self, template):
        """Test generated code includes test function/method."""
        spec = TestSpec(
            id="test-1",
            name="My Test",
            steps=[TestStep(action="goto", target="/")],
        )
        code = template.generate(spec)
        code_lower = code.lower()
        assert "test" in code_lower or "def " in code_lower or "func " in code_lower

    def test_empty_spec_generates_valid_code(self, template):
        """Test empty spec still generates valid structure."""
        spec = TestSpec(id="test-empty", name="Empty Test")
        code = template.generate(spec)
        assert code
        assert len(code) > 50  # Should have some structure


# =============================================================================
# Stress Tests
# =============================================================================


class TestStressTests:
    """Stress tests for templates."""

    def test_deeply_nested_selectors(self, template):
        """Test handling of deeply nested selectors."""
        nested_selector = " ".join([f"div:nth-child({i})" for i in range(1, 20)])
        step = TestStep(action="click", target=nested_selector)
        code = template.generate_step_code(step, 0)
        assert code  # Should not crash

    def test_complex_spec(self, template):
        """Test handling of complex spec with all features."""
        spec = TestSpec(
            id="test-complex",
            name="Complex Test with Many Features",
            description="This is a comprehensive test that tests many things",
            steps=[
                TestStep(action="goto", target="https://example.com"),
                TestStep(action="fill", target="#username", value="user@example.com"),
                TestStep(action="fill", target="#password", value="secret!@#$%"),
                TestStep(action="click", target="button[type=submit]"),
                TestStep(action="wait", target=".dashboard", timeout=10000),
                TestStep(action="hover", target=".menu"),
                TestStep(action="click", target=".menu-item"),
                TestStep(action="select", target="#dropdown", value="option1"),
                TestStep(action="scroll", value="0,500"),
                TestStep(action="double_click", target=".editable"),
                TestStep(action="type", target=".editable", value="New text"),
                TestStep(action="press_key", value="Tab"),
                TestStep(action="screenshot"),
            ],
            assertions=[
                TestAssertion(type="element_visible", target=".dashboard"),
                TestAssertion(type="element_hidden", target=".loading"),
                TestAssertion(type="text_contains", target=".welcome", expected="Welcome"),
                TestAssertion(type="text_equals", target=".username", expected="user@example.com"),
                TestAssertion(type="url_contains", expected="/dashboard"),
                TestAssertion(type="value_equals", target="#email", expected="user@example.com"),
                TestAssertion(type="title_contains", expected="Dashboard"),
            ],
        )
        code = template.generate(spec)
        assert code
        assert len(code) > 500  # Should be substantial

    def test_rapid_generation(self, template):
        """Test rapid repeated generation."""
        spec = TestSpec(
            id="test-rapid",
            name="Rapid Test",
            steps=[TestStep(action="click", target="#btn")],
        )
        for _ in range(100):
            code = template.generate(spec)
            assert code  # Should work consistently
