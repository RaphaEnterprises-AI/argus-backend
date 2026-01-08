"""C# Selenium export template."""

from .base import BaseTemplate, TestSpec, TestStep, TestAssertion


class CSharpSeleniumTemplate(BaseTemplate):
    """Template for C# Selenium tests."""

    language = "csharp"
    framework = "selenium"
    file_extension = ".cs"
    indent = "    "

    def generate_imports(self, test_spec: TestSpec) -> str:
        """Generate C# using statements."""
        imports = [
            "using NUnit.Framework;",
            "using OpenQA.Selenium;",
            "using OpenQA.Selenium.Chrome;",
            "using OpenQA.Selenium.Support.UI;",
            "using OpenQA.Selenium.Interactions;",
            "using System;",
        ]
        return "\n".join(imports)

    def generate_class_header(self, test_spec: TestSpec) -> str:
        """Generate C# class header."""
        class_name = self.config.get("test_class_name") or f"Test{self.to_pascal_case(test_spec.name)}"
        method_name = f"Test{self.to_pascal_case(test_spec.name)}"

        lines = [
            "",
            "namespace ArgusTests",
            "{",
            f"    /// <summary>",
            f"    /// Generated from Argus test spec: {test_spec.id}",
        ]

        if test_spec.description:
            lines.append(f"    /// {test_spec.description}")

        lines.extend([
            "    /// </summary>",
            "    [TestFixture]",
            f"    public class {class_name}",
            "    {",
            "        private IWebDriver _driver;",
            "        private WebDriverWait _wait;",
            '        private string _baseUrl = "http://localhost:3000";',
            "",
            "        [SetUp]",
            "        public void Setup()",
            "        {",
            "            _driver = new ChromeDriver();",
            "            _wait = new WebDriverWait(_driver, TimeSpan.FromSeconds(10));",
            "        }",
            "",
            "        [TearDown]",
            "        public void TearDown()",
            "        {",
            "            _driver?.Quit();",
            "        }",
            "",
            "        [Test]",
            f"        public void {method_name}()",
            "        {",
        ])

        return "\n".join(lines)

    def _get_locator(self, target: str) -> str:
        """Convert selector to C# Selenium locator."""
        if not target:
            return 'By.CssSelector("")'

        if target.startswith("#"):
            return f'By.Id("{target[1:]}")'
        elif target.startswith("."):
            return f'By.ClassName("{target[1:]}")'
        elif target.startswith("//"):
            return f'By.XPath("{self.escape_string(target)}")'
        elif target.startswith("name="):
            return f'By.Name("{target[5:]}")'
        else:
            return f'By.CssSelector("{self.escape_string(target)}")'

    def generate_step_code(self, step: TestStep, index: int) -> str:
        """Generate code for a single step."""
        action = step.action.lower()
        target = step.target or ""
        value = step.value or ""
        locator = self._get_locator(target)

        comment = f"            // Step {index + 1}: {step.description or f'{action} {target}'.strip()}"
        code_line = ""

        if action == "goto":
            if target.startswith("http"):
                code_line = f'            _driver.Navigate().GoToUrl("{self.escape_string(target)}");'
            else:
                code_line = f'            _driver.Navigate().GoToUrl(_baseUrl + "{target}");'

        elif action == "click":
            code_line = f"""            var element{index} = _wait.Until(
                SeleniumExtras.WaitHelpers.ExpectedConditions.ElementToBeClickable({locator})
            );
            element{index}.Click();"""

        elif action == "fill":
            code_line = f"""            var element{index} = _wait.Until(
                SeleniumExtras.WaitHelpers.ExpectedConditions.ElementExists({locator})
            );
            element{index}.Clear();
            element{index}.SendKeys("{self.escape_string(value)}");"""

        elif action == "type":
            code_line = f"""            var element{index} = _wait.Until(
                SeleniumExtras.WaitHelpers.ExpectedConditions.ElementExists({locator})
            );
            element{index}.SendKeys("{self.escape_string(value)}");"""

        elif action == "select":
            code_line = f"""            var element{index} = _wait.Until(
                SeleniumExtras.WaitHelpers.ExpectedConditions.ElementExists({locator})
            );
            new SelectElement(element{index}).SelectByText("{self.escape_string(value)}");"""

        elif action == "hover":
            code_line = f"""            var element{index} = _wait.Until(
                SeleniumExtras.WaitHelpers.ExpectedConditions.ElementExists({locator})
            );
            new Actions(_driver).MoveToElement(element{index}).Perform();"""

        elif action == "wait":
            code_line = f"""            _wait.Until(
                SeleniumExtras.WaitHelpers.ExpectedConditions.ElementExists({locator})
            );"""

        elif action == "scroll":
            if value:
                parts = value.split(",")
                x = parts[0] if len(parts) > 0 else "0"
                y = parts[1] if len(parts) > 1 else "0"
                code_line = f'            ((IJavaScriptExecutor)_driver).ExecuteScript("window.scrollBy({x}, {y})");'
            else:
                code_line = '            ((IJavaScriptExecutor)_driver).ExecuteScript("window.scrollBy(0, 300)");'

        elif action == "press_key":
            key = value or target or "Enter"
            code_line = f'            _driver.SwitchTo().ActiveElement().SendKeys(Keys.{key});'

        elif action == "screenshot":
            code_line = f"""            var screenshot = ((ITakesScreenshot)_driver).GetScreenshot();
            screenshot.SaveAsFile("screenshot_{index}.png");"""

        elif action == "double_click":
            code_line = f"""            var element{index} = _wait.Until(
                SeleniumExtras.WaitHelpers.ExpectedConditions.ElementToBeClickable({locator})
            );
            new Actions(_driver).DoubleClick(element{index}).Perform();"""

        else:
            code_line = f'            // Unknown action: {action}'

        return f"{comment}\n{code_line}\n"

    def generate_assertion_code(self, assertion: TestAssertion) -> str:
        """Generate code for a single assertion."""
        assertion_type = assertion.type.lower()
        target = assertion.target or ""
        expected = assertion.expected or ""
        locator = self._get_locator(target)

        if assertion_type == "element_visible":
            return f"""            var element = _wait.Until(
                SeleniumExtras.WaitHelpers.ExpectedConditions.ElementIsVisible({locator})
            );
            Assert.That(element.Displayed, Is.True);"""

        elif assertion_type == "element_hidden":
            return f"""            _wait.Until(
                SeleniumExtras.WaitHelpers.ExpectedConditions.InvisibilityOfElementLocated({locator})
            );"""

        elif assertion_type == "text_contains":
            return f"""            var element = _wait.Until(
                SeleniumExtras.WaitHelpers.ExpectedConditions.ElementExists({locator})
            );
            Assert.That(element.Text, Does.Contain("{self.escape_string(expected)}"));"""

        elif assertion_type == "text_equals":
            return f"""            var element = _wait.Until(
                SeleniumExtras.WaitHelpers.ExpectedConditions.ElementExists({locator})
            );
            Assert.That(element.Text, Is.EqualTo("{self.escape_string(expected)}"));"""

        elif assertion_type in ("url_contains", "url_matches"):
            return f'            Assert.That(_driver.Url, Does.Contain("{self.escape_string(expected)}"));'

        elif assertion_type == "value_equals":
            return f"""            var element = _wait.Until(
                SeleniumExtras.WaitHelpers.ExpectedConditions.ElementExists({locator})
            );
            Assert.That(element.GetAttribute("value"), Is.EqualTo("{self.escape_string(expected)}"));"""

        elif assertion_type == "title_contains":
            return f'            Assert.That(_driver.Title, Does.Contain("{self.escape_string(expected)}"));'

        else:
            return f'            // Unknown assertion type: {assertion_type}'

    def generate_class_footer(self) -> str:
        """Generate class footer."""
        return "        }\n    }\n}"

    def _generate_assertions_header(self) -> str:
        """Generate assertions section header."""
        return "            // Assertions"

    def _get_comment_prefix(self) -> str:
        """Get comment prefix for C#."""
        return "//"
