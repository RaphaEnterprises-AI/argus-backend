"""Go Rod export template."""

from .base import BaseTemplate, TestAssertion, TestSpec, TestStep


class GoRodTemplate(BaseTemplate):
    """Template for Go Rod tests."""

    language = "go"
    framework = "rod"
    file_extension = ".go"
    indent = "\t"

    def generate_imports(self, test_spec: TestSpec) -> str:
        """Generate Go imports."""
        imports = [
            "package tests",
            "",
            "import (",
            '\t"testing"',
            '\t"time"',
            "",
            '\t"github.com/go-rod/rod"',
            '\t"github.com/go-rod/rod/lib/launcher"',
            '\t"github.com/stretchr/testify/assert"',
            ")",
        ]
        return "\n".join(imports)

    def generate_class_header(self, test_spec: TestSpec) -> str:
        """Generate Go test function header."""
        func_name = f"Test{self.to_pascal_case(test_spec.name)}"

        lines = [
            "",
            f"// {func_name} - Generated from Argus test spec: {test_spec.id}",
        ]

        if test_spec.description:
            lines.append(f"// {test_spec.description}")

        lines.extend([
            f"func {func_name}(t *testing.T) {{",
            "\t// Setup",
            "\tl := launcher.New().Headless(true).MustLaunch()",
            "\tbrowser := rod.New().ControlURL(l).MustConnect()",
            "\tdefer browser.MustClose()",
            "\tpage := browser.MustPage()",
            '\tbaseURL := "http://localhost:3000"',
            "",
        ])

        return "\n".join(lines)

    def generate_step_code(self, step: TestStep, index: int) -> str:
        """Generate code for a single step."""
        action = step.action.lower()
        target = step.target or ""
        value = step.value or ""

        comment = f"\t// Step {index + 1}: {step.description or f'{action} {target}'.strip()}"
        code_line = ""

        if action == "goto":
            if target.startswith("http"):
                code_line = f'\tpage.MustNavigate("{self.escape_string(target)}")'
            else:
                code_line = f'\tpage.MustNavigate(baseURL + "{target}")'

        elif action == "click":
            code_line = f'\tpage.MustElement("{self.escape_string(target)}").MustClick()'

        elif action == "fill":
            code_line = f'\tpage.MustElement("{self.escape_string(target)}").MustSelectAllText().MustInput("{self.escape_string(value)}")'

        elif action == "type":
            code_line = f'\tpage.MustElement("{self.escape_string(target)}").MustInput("{self.escape_string(value)}")'

        elif action == "select":
            code_line = f'\tpage.MustElement("{self.escape_string(target)}").MustSelect("{self.escape_string(value)}")'

        elif action == "hover":
            code_line = f'\tpage.MustElement("{self.escape_string(target)}").MustHover()'

        elif action == "wait":
            timeout = step.timeout or 30000
            code_line = f'\tpage.Timeout({timeout} * time.Millisecond).MustElement("{self.escape_string(target)}")'

        elif action == "scroll":
            if value:
                parts = value.split(",")
                x = parts[0] if len(parts) > 0 else "0"
                y = parts[1] if len(parts) > 1 else "0"
                code_line = f'\tpage.MustEval(`window.scrollBy({x}, {y})`)'
            else:
                code_line = '\tpage.MustEval(`window.scrollBy(0, 300)`)'

        elif action == "press_key":
            key = value or target or "Enter"
            code_line = f'\tpage.Keyboard.Press(input.{key})'

        elif action == "screenshot":
            code_line = f'\tpage.MustScreenshot("screenshot_{index}.png")'

        elif action == "double_click":
            code_line = f'\tpage.MustElement("{self.escape_string(target)}").MustClick().MustClick()'

        else:
            code_line = f'\t// Unknown action: {action}'

        return f"{comment}\n{code_line}\n"

    def generate_assertion_code(self, assertion: TestAssertion) -> str:
        """Generate code for a single assertion."""
        assertion_type = assertion.type.lower()
        target = assertion.target or ""
        expected = assertion.expected or ""

        if assertion_type == "element_visible":
            return f'\tassert.True(t, page.MustElement("{self.escape_string(target)}").MustVisible())'

        elif assertion_type == "element_hidden":
            return f'\tassert.False(t, page.MustElement("{self.escape_string(target)}").MustVisible())'

        elif assertion_type == "text_contains":
            return f'\tassert.Contains(t, page.MustElement("{self.escape_string(target)}").MustText(), "{self.escape_string(expected)}")'

        elif assertion_type == "text_equals":
            return f'\tassert.Equal(t, "{self.escape_string(expected)}", page.MustElement("{self.escape_string(target)}").MustText())'

        elif assertion_type in ("url_contains", "url_matches"):
            return f'\tassert.Contains(t, page.MustInfo().URL, "{self.escape_string(expected)}")'

        elif assertion_type == "value_equals":
            return f'\tassert.Equal(t, "{self.escape_string(expected)}", page.MustElement("{self.escape_string(target)}").MustProperty("value").String())'

        elif assertion_type == "title_contains":
            return f'\tassert.Contains(t, page.MustInfo().Title, "{self.escape_string(expected)}")'

        else:
            return f'\t// Unknown assertion type: {assertion_type}'

    def generate_class_footer(self) -> str:
        """Generate function footer."""
        return "}"

    def _generate_assertions_header(self) -> str:
        """Generate assertions section header."""
        return "\t// Assertions"

    def _get_comment_prefix(self) -> str:
        """Get comment prefix for Go."""
        return "//"
