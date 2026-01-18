"""Java Selenium export template."""

from .base import BaseTemplate, TestAssertion, TestSpec, TestStep


class JavaSeleniumTemplate(BaseTemplate):
    """Template for Java Selenium tests."""

    language = "java"
    framework = "selenium"
    file_extension = ".java"
    indent = "    "

    def generate_imports(self, test_spec: TestSpec) -> str:
        """Generate Java imports."""
        imports = [
            "import org.junit.jupiter.api.Test;",
            "import org.junit.jupiter.api.BeforeEach;",
            "import org.junit.jupiter.api.AfterEach;",
            "import org.openqa.selenium.WebDriver;",
            "import org.openqa.selenium.WebElement;",
            "import org.openqa.selenium.By;",
            "import org.openqa.selenium.Keys;",
            "import org.openqa.selenium.chrome.ChromeDriver;",
            "import org.openqa.selenium.support.ui.WebDriverWait;",
            "import org.openqa.selenium.support.ui.ExpectedConditions;",
            "import org.openqa.selenium.support.ui.Select;",
            "import org.openqa.selenium.interactions.Actions;",
            "import org.openqa.selenium.JavascriptExecutor;",
            "import java.time.Duration;",
            "import static org.junit.jupiter.api.Assertions.*;",
        ]
        return "\n".join(imports)

    def generate_class_header(self, test_spec: TestSpec) -> str:
        """Generate Java class header."""
        class_name = self.config.get("test_class_name") or f"Test{self.to_pascal_case(test_spec.name)}"
        method_name = f"test{self.to_pascal_case(test_spec.name)}"

        lines = [
            "",
            "/**",
            f" * Generated from Argus test spec: {test_spec.id}",
        ]

        if test_spec.description:
            lines.append(f" * {test_spec.description}")

        lines.extend([
            " */",
            f"public class {class_name} {{",
            "",
            "    private WebDriver driver;",
            "    private WebDriverWait wait;",
            '    private String baseUrl = "http://localhost:3000";',
            "",
            "    @BeforeEach",
            "    public void setUp() {",
            "        driver = new ChromeDriver();",
            "        wait = new WebDriverWait(driver, Duration.ofSeconds(10));",
            "    }",
            "",
            "    @AfterEach",
            "    public void tearDown() {",
            "        if (driver != null) {",
            "            driver.quit();",
            "        }",
            "    }",
            "",
            "    @Test",
            f"    public void {method_name}() {{",
        ])

        return "\n".join(lines)

    def _get_locator(self, target: str) -> str:
        """Convert selector to Java Selenium locator."""
        if not target:
            return 'By.cssSelector("")'

        if target.startswith("#"):
            return f'By.id("{target[1:]}")'
        elif target.startswith("."):
            return f'By.className("{target[1:]}")'
        elif target.startswith("//"):
            return f'By.xpath("{self.escape_string(target)}")'
        elif target.startswith("name="):
            return f'By.name("{target[5:]}")'
        else:
            return f'By.cssSelector("{self.escape_string(target)}")'

    def generate_step_code(self, step: TestStep, index: int) -> str:
        """Generate code for a single step."""
        action = step.action.lower()
        target = step.target or ""
        value = step.value or ""
        locator = self._get_locator(target)

        comment = f"        // Step {index + 1}: {step.description or f'{action} {target}'.strip()}"
        code_line = ""

        if action == "goto":
            if target.startswith("http"):
                code_line = f'        driver.get("{self.escape_string(target)}");'
            else:
                code_line = f'        driver.get(baseUrl + "{target}");'

        elif action == "click":
            code_line = f"""        WebElement element{index} = wait.until(
            ExpectedConditions.elementToBeClickable({locator})
        );
        element{index}.click();"""

        elif action == "fill":
            code_line = f"""        WebElement element{index} = wait.until(
            ExpectedConditions.presenceOfElementLocated({locator})
        );
        element{index}.clear();
        element{index}.sendKeys("{self.escape_string(value)}");"""

        elif action == "type":
            code_line = f"""        WebElement element{index} = wait.until(
            ExpectedConditions.presenceOfElementLocated({locator})
        );
        element{index}.sendKeys("{self.escape_string(value)}");"""

        elif action == "select":
            code_line = f"""        WebElement element{index} = wait.until(
            ExpectedConditions.presenceOfElementLocated({locator})
        );
        new Select(element{index}).selectByVisibleText("{self.escape_string(value)}");"""

        elif action == "hover":
            code_line = f"""        WebElement element{index} = wait.until(
            ExpectedConditions.presenceOfElementLocated({locator})
        );
        new Actions(driver).moveToElement(element{index}).perform();"""

        elif action == "wait":
            code_line = f"""        wait.until(
            ExpectedConditions.presenceOfElementLocated({locator})
        );"""

        elif action == "scroll":
            if value:
                parts = value.split(",")
                x = parts[0] if len(parts) > 0 else "0"
                y = parts[1] if len(parts) > 1 else "0"
                code_line = f'        ((JavascriptExecutor) driver).executeScript("window.scrollBy({x}, {y})");'
            else:
                code_line = '        ((JavascriptExecutor) driver).executeScript("window.scrollBy(0, 300)");'

        elif action == "press_key":
            key = value or target or "ENTER"
            code_line = f'        driver.switchTo().activeElement().sendKeys(Keys.{key.upper()});'

        elif action == "screenshot":
            code_line = '        // Screenshot capture requires additional setup'

        elif action == "double_click":
            code_line = f"""        WebElement element{index} = wait.until(
            ExpectedConditions.elementToBeClickable({locator})
        );
        new Actions(driver).doubleClick(element{index}).perform();"""

        else:
            code_line = f'        // Unknown action: {action}'

        return f"{comment}\n{code_line}\n"

    def generate_assertion_code(self, assertion: TestAssertion) -> str:
        """Generate code for a single assertion."""
        assertion_type = assertion.type.lower()
        target = assertion.target or ""
        expected = assertion.expected or ""
        locator = self._get_locator(target)

        if assertion_type == "element_visible":
            return f"""        WebElement visible = wait.until(
            ExpectedConditions.visibilityOfElementLocated({locator})
        );
        assertTrue(visible.isDisplayed());"""

        elif assertion_type == "element_hidden":
            return f"""        wait.until(
            ExpectedConditions.invisibilityOfElementLocated({locator})
        );"""

        elif assertion_type == "text_contains":
            return f"""        WebElement element = wait.until(
            ExpectedConditions.presenceOfElementLocated({locator})
        );
        assertTrue(element.getText().contains("{self.escape_string(expected)}"));"""

        elif assertion_type == "text_equals":
            return f"""        WebElement element = wait.until(
            ExpectedConditions.presenceOfElementLocated({locator})
        );
        assertEquals("{self.escape_string(expected)}", element.getText());"""

        elif assertion_type in ("url_contains", "url_matches"):
            return f'        assertTrue(driver.getCurrentUrl().contains("{self.escape_string(expected)}"));'

        elif assertion_type == "value_equals":
            return f"""        WebElement element = wait.until(
            ExpectedConditions.presenceOfElementLocated({locator})
        );
        assertEquals("{self.escape_string(expected)}", element.getAttribute("value"));"""

        elif assertion_type == "title_contains":
            return f'        assertTrue(driver.getTitle().contains("{self.escape_string(expected)}"));'

        else:
            return f'        // Unknown assertion type: {assertion_type}'

    def generate_class_footer(self) -> str:
        """Generate class footer."""
        return "    }\n}"

    def _generate_assertions_header(self) -> str:
        """Generate assertions section header."""
        return "        // Assertions"

    def _get_comment_prefix(self) -> str:
        """Get comment prefix for Java."""
        return "//"
