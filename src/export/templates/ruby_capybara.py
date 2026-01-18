"""Ruby Capybara export template."""

from .base import BaseTemplate, TestAssertion, TestSpec, TestStep


class RubyCapybaraTemplate(BaseTemplate):
    """Template for Ruby Capybara tests."""

    language = "ruby"
    framework = "capybara"
    file_extension = ".rb"
    indent = "  "

    def generate_imports(self, test_spec: TestSpec) -> str:
        """Generate Ruby requires."""
        imports = [
            "require 'capybara/rspec'",
            "require 'selenium-webdriver'",
            "",
            "Capybara.default_driver = :selenium_chrome",
            "Capybara.default_max_wait_time = 10",
        ]
        return "\n".join(imports)

    def generate_class_header(self, test_spec: TestSpec) -> str:
        """Generate RSpec describe block."""
        lines = [
            "",
            f"RSpec.describe '{self.escape_string(test_spec.name)}' do",
            f"  # Generated from Argus test spec: {test_spec.id}",
            "",
        ]

        if test_spec.description:
            lines.insert(3, f"  # {test_spec.description}")

        lines.extend([
            f"  it '{self.escape_string(test_spec.name)}' do",
        ])

        return "\n".join(lines)

    def generate_step_code(self, step: TestStep, index: int) -> str:
        """Generate code for a single step."""
        action = step.action.lower()
        target = step.target or ""
        value = step.value or ""

        comment = f"    # Step {index + 1}: {step.description or f'{action} {target}'.strip()}"
        code_line = ""

        if action == "goto":
            if target.startswith("http"):
                code_line = f"    visit '{self.escape_string(target)}'"
            else:
                code_line = f"    visit \"{target}\""

        elif action == "click":
            if target.startswith("#"):
                code_line = f"    find('{self.escape_string(target)}').click"
            elif target.startswith("."):
                code_line = f"    find('{self.escape_string(target)}').click"
            else:
                code_line = f"    find('{self.escape_string(target)}').click"

        elif action == "fill":
            if target.startswith("#"):
                code_line = f"    fill_in '{target[1:]}', with: '{self.escape_string(value)}'"
            else:
                code_line = f"    find('{self.escape_string(target)}').set('{self.escape_string(value)}')"

        elif action == "type":
            code_line = f"    find('{self.escape_string(target)}').send_keys('{self.escape_string(value)}')"

        elif action == "select":
            code_line = f"    select '{self.escape_string(value)}', from: '{self.escape_string(target)}'"

        elif action == "hover":
            code_line = f"    find('{self.escape_string(target)}').hover"

        elif action == "wait":
            code_line = f"    find('{self.escape_string(target)}', wait: 30)"

        elif action == "scroll":
            if value:
                parts = value.split(",")
                y = parts[1] if len(parts) > 1 else "0"
                code_line = f"    page.execute_script('window.scrollBy(0, {y})')"
            else:
                code_line = "    page.execute_script('window.scrollBy(0, 300)')"

        elif action == "press_key":
            key = value or target or "enter"
            code_line = f"    find('{self.escape_string(target)}').send_keys(:{key.lower()})"

        elif action == "screenshot":
            code_line = f"    page.save_screenshot('screenshot_{index}.png')"

        elif action == "double_click":
            code_line = f"    find('{self.escape_string(target)}').double_click"

        else:
            code_line = f"    # Unknown action: {action}"

        return f"{comment}\n{code_line}\n"

    def generate_assertion_code(self, assertion: TestAssertion) -> str:
        """Generate code for a single assertion."""
        assertion_type = assertion.type.lower()
        target = assertion.target or ""
        expected = assertion.expected or ""

        if assertion_type == "element_visible":
            return f"    expect(page).to have_selector('{self.escape_string(target)}', visible: true)"

        elif assertion_type == "element_hidden":
            return f"    expect(page).not_to have_selector('{self.escape_string(target)}', visible: true)"

        elif assertion_type == "text_contains":
            return f"    expect(find('{self.escape_string(target)}')).to have_text('{self.escape_string(expected)}')"

        elif assertion_type == "text_equals":
            return f"    expect(find('{self.escape_string(target)}').text).to eq('{self.escape_string(expected)}')"

        elif assertion_type in ("url_contains", "url_matches"):
            return f"    expect(current_url).to include('{self.escape_string(expected)}')"

        elif assertion_type == "value_equals":
            return f"    expect(find('{self.escape_string(target)}').value).to eq('{self.escape_string(expected)}')"

        elif assertion_type == "title_contains":
            return f"    expect(page).to have_title(/{self.escape_string(expected)}/)"

        else:
            return f"    # Unknown assertion type: {assertion_type}"

    def generate_class_footer(self) -> str:
        """Generate block footer."""
        return "  end\nend"

    def _generate_assertions_header(self) -> str:
        """Generate assertions section header."""
        return "    # Assertions"

    def _get_comment_prefix(self) -> str:
        """Get comment prefix for Ruby."""
        return "#"
