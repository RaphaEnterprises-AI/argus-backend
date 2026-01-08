"""Python Playwright export template."""

from .base import BaseTemplate, TestSpec, TestStep, TestAssertion


class PythonPlaywrightTemplate(BaseTemplate):
    """Template for Python Playwright tests."""

    language = "python"
    framework = "playwright"
    file_extension = ".py"
    indent = "    "

    def generate_imports(self, test_spec: TestSpec) -> str:
        """Generate Python imports."""
        imports = [
            "import pytest",
            "from playwright.sync_api import Page, expect",
            "import re",
        ]

        if self.config.get("async_style", True):
            imports[1] = "from playwright.async_api import Page, expect"
            imports.insert(1, "import asyncio")

        return "\n".join(imports)

    def generate_class_header(self, test_spec: TestSpec) -> str:
        """Generate pytest class header."""
        class_name = self.config.get("test_class_name") or f"Test{self.to_pascal_case(test_spec.name)}"
        method_name = f"test_{self.to_snake_case(test_spec.name)}"

        lines = [
            "",
            f'class {class_name}:',
            f'    """Generated from Argus test spec: {test_spec.id}"""',
            "",
        ]

        if self.config.get("async_style", True):
            lines.append(f"    async def {method_name}(self, page: Page, base_url: str):")
        else:
            lines.append(f"    def {method_name}(self, page: Page, base_url: str):")

        # Add docstring
        if test_spec.description:
            lines.append(f'        """{test_spec.description}"""')

        return "\n".join(lines)

    def generate_step_code(self, step: TestStep, index: int) -> str:
        """Generate code for a single step."""
        action = step.action.lower()
        target = step.target or ""
        value = step.value or ""
        await_prefix = "await " if self.config.get("async_style", True) else ""

        # Comment
        comment = f"        # Step {index + 1}: {step.description or f'{action} {target}'.strip()}"

        code_line = ""

        if action == "goto":
            url = target if target.startswith("http") else f'f"{{base_url}}{target}"'
            if target.startswith("http"):
                code_line = f'        {await_prefix}page.goto("{self.escape_string(target)}")'
            else:
                code_line = f'        {await_prefix}page.goto(f"{{base_url}}{target}")'

        elif action == "click":
            code_line = f'        {await_prefix}page.click("{self.escape_string(target)}")'

        elif action == "fill":
            code_line = f'        {await_prefix}page.fill("{self.escape_string(target)}", "{self.escape_string(value)}")'

        elif action == "type":
            code_line = f'        {await_prefix}page.type("{self.escape_string(target)}", "{self.escape_string(value)}")'

        elif action == "select":
            code_line = f'        {await_prefix}page.select_option("{self.escape_string(target)}", "{self.escape_string(value)}")'

        elif action == "hover":
            code_line = f'        {await_prefix}page.hover("{self.escape_string(target)}")'

        elif action == "wait":
            timeout = step.timeout or 30000
            code_line = f'        {await_prefix}page.wait_for_selector("{self.escape_string(target)}", timeout={timeout})'

        elif action == "scroll":
            if value:
                parts = value.split(",")
                x = parts[0] if len(parts) > 0 else "0"
                y = parts[1] if len(parts) > 1 else "0"
                code_line = f'        {await_prefix}page.evaluate("window.scrollBy({x}, {y})")'
            else:
                code_line = f'        {await_prefix}page.evaluate("window.scrollBy(0, 300)")'

        elif action == "press_key":
            key = value or target or "Enter"
            code_line = f'        {await_prefix}page.keyboard.press("{key}")'

        elif action == "screenshot":
            code_line = f'        {await_prefix}page.screenshot(path="screenshot_{index}.png")'

        elif action == "double_click":
            code_line = f'        {await_prefix}page.dblclick("{self.escape_string(target)}")'

        else:
            code_line = f'        # Unknown action: {action}'

        return f"{comment}\n{code_line}\n"

    def generate_assertion_code(self, assertion: TestAssertion) -> str:
        """Generate code for a single assertion."""
        await_prefix = "await " if self.config.get("async_style", True) else ""
        assertion_type = assertion.type.lower()
        target = assertion.target or ""
        expected = assertion.expected or ""

        if assertion_type == "element_visible":
            return f'        {await_prefix}expect(page.locator("{self.escape_string(target)}")).to_be_visible()'

        elif assertion_type == "element_hidden":
            return f'        {await_prefix}expect(page.locator("{self.escape_string(target)}")).to_be_hidden()'

        elif assertion_type == "text_contains":
            return f'        {await_prefix}expect(page.locator("{self.escape_string(target)}")).to_contain_text("{self.escape_string(expected)}")'

        elif assertion_type == "text_equals":
            return f'        {await_prefix}expect(page.locator("{self.escape_string(target)}")).to_have_text("{self.escape_string(expected)}")'

        elif assertion_type in ("url_contains", "url_matches"):
            return f'        {await_prefix}expect(page).to_have_url(re.compile(r".*{self.escape_string(expected)}.*"))'

        elif assertion_type == "value_equals":
            return f'        {await_prefix}expect(page.locator("{self.escape_string(target)}")).to_have_value("{self.escape_string(expected)}")'

        elif assertion_type == "title_contains":
            return f'        {await_prefix}expect(page).to_have_title(re.compile(r".*{self.escape_string(expected)}.*"))'

        else:
            return f'        # Unknown assertion type: {assertion_type}'

    def generate_class_footer(self) -> str:
        """Generate class footer."""
        return ""

    def _generate_assertions_header(self) -> str:
        """Generate assertions section header."""
        return "        # Assertions"
