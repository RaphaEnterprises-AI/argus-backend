"""TypeScript Playwright export template."""

from .base import BaseTemplate, TestAssertion, TestSpec, TestStep


class TypeScriptPlaywrightTemplate(BaseTemplate):
    """Template for TypeScript Playwright tests."""

    language = "typescript"
    framework = "playwright"
    file_extension = ".ts"
    indent = "  "

    def generate_imports(self, test_spec: TestSpec) -> str:
        """Generate TypeScript imports."""
        return "import { test, expect } from '@playwright/test';"

    def generate_class_header(self, test_spec: TestSpec) -> str:
        """Generate Playwright test header."""
        test_name = test_spec.name

        lines = [
            "",
            f"test.describe('{self.escape_string(test_name)}', () => {{",
            f"  // Generated from Argus test spec: {test_spec.id}",
            "",
            f"  test('{self.escape_string(test_name)}', async ({{ page, baseURL }}) => {{",
        ]

        if test_spec.description:
            lines.insert(3, f"  // {test_spec.description}")

        return "\n".join(lines)

    def generate_step_code(self, step: TestStep, index: int) -> str:
        """Generate code for a single step."""
        action = step.action.lower()
        target = step.target or ""
        value = step.value or ""

        comment = f"    // Step {index + 1}: {step.description or f'{action} {target}'.strip()}"
        code_line = ""

        if action == "goto":
            if target.startswith("http"):
                code_line = f"    await page.goto('{self.escape_string(target)}');"
            else:
                code_line = f"    await page.goto(`${{baseURL}}{target}`);"

        elif action == "click":
            code_line = f"    await page.click('{self.escape_string(target)}');"

        elif action == "fill":
            code_line = f"    await page.fill('{self.escape_string(target)}', '{self.escape_string(value)}');"

        elif action == "type":
            code_line = f"    await page.type('{self.escape_string(target)}', '{self.escape_string(value)}');"

        elif action == "select":
            code_line = f"    await page.selectOption('{self.escape_string(target)}', '{self.escape_string(value)}');"

        elif action == "hover":
            code_line = f"    await page.hover('{self.escape_string(target)}');"

        elif action == "wait":
            timeout = step.timeout or 30000
            code_line = f"    await page.waitForSelector('{self.escape_string(target)}', {{ timeout: {timeout} }});"

        elif action == "scroll":
            if value:
                parts = value.split(",")
                x = parts[0] if len(parts) > 0 else "0"
                y = parts[1] if len(parts) > 1 else "0"
                code_line = f"    await page.evaluate(() => window.scrollBy({x}, {y}));"
            else:
                code_line = "    await page.evaluate(() => window.scrollBy(0, 300));"

        elif action == "press_key":
            key = value or target or "Enter"
            code_line = f"    await page.keyboard.press('{key}');"

        elif action == "screenshot":
            code_line = f"    await page.screenshot({{ path: 'screenshot_{index}.png' }});"

        elif action == "double_click":
            code_line = f"    await page.dblclick('{self.escape_string(target)}');"

        else:
            code_line = f"    // Unknown action: {action}"

        return f"{comment}\n{code_line}\n"

    def generate_assertion_code(self, assertion: TestAssertion) -> str:
        """Generate code for a single assertion."""
        assertion_type = assertion.type.lower()
        target = assertion.target or ""
        expected = assertion.expected or ""

        if assertion_type == "element_visible":
            return f"    await expect(page.locator('{self.escape_string(target)}')).toBeVisible();"

        elif assertion_type == "element_hidden":
            return f"    await expect(page.locator('{self.escape_string(target)}')).toBeHidden();"

        elif assertion_type == "text_contains":
            return f"    await expect(page.locator('{self.escape_string(target)}')).toContainText('{self.escape_string(expected)}');"

        elif assertion_type == "text_equals":
            return f"    await expect(page.locator('{self.escape_string(target)}')).toHaveText('{self.escape_string(expected)}');"

        elif assertion_type in ("url_contains", "url_matches"):
            return f"    await expect(page).toHaveURL(/{self.escape_string(expected)}/);"

        elif assertion_type == "value_equals":
            return f"    await expect(page.locator('{self.escape_string(target)}')).toHaveValue('{self.escape_string(expected)}');"

        elif assertion_type == "title_contains":
            return f"    await expect(page).toHaveTitle(/{self.escape_string(expected)}/);"

        else:
            return f"    // Unknown assertion type: {assertion_type}"

    def generate_class_footer(self) -> str:
        """Generate class footer."""
        return "  });\n});"

    def _generate_assertions_header(self) -> str:
        """Generate assertions section header."""
        return "    // Assertions"

    def _get_comment_prefix(self) -> str:
        """Get comment prefix for TypeScript."""
        return "//"
