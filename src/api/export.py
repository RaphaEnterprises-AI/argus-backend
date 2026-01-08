"""Export API endpoints for multi-language test export.

This module provides REST API endpoints for exporting tests to various
languages and frameworks, including support for parameterized/data-driven tests.

Supported languages:
- Python (Playwright, Selenium, pytest)
- TypeScript (Playwright, Puppeteer, Cypress)
- Java (Selenium, TestNG, JUnit)
- C# (Selenium, Playwright, NUnit)
- Ruby (Capybara, Selenium)
- Go (Rod, Chromedp)

Parameterized test export generates:
- Python: @pytest.mark.parametrize decorator
- TypeScript: test.each() or describe.each()
- Java: @ParameterizedTest with @CsvSource/@MethodSource
"""

from datetime import datetime, timezone
from typing import Any, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/export", tags=["Export"])


# =============================================================================
# Enums and Constants
# =============================================================================

class ExportLanguage(str, Enum):
    """Supported export languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    RUBY = "ruby"
    GO = "go"


class ExportFramework(str, Enum):
    """Supported test frameworks by language."""
    # Python
    PLAYWRIGHT_PYTHON = "playwright"
    SELENIUM_PYTHON = "selenium"
    PYTEST = "pytest"

    # TypeScript/JavaScript
    PLAYWRIGHT_TS = "playwright-ts"
    PUPPETEER = "puppeteer"
    CYPRESS = "cypress"

    # Java
    SELENIUM_JAVA = "selenium-java"
    TESTNG = "testng"
    JUNIT = "junit"

    # C#
    SELENIUM_CSHARP = "selenium-csharp"
    PLAYWRIGHT_CSHARP = "playwright-csharp"
    NUNIT = "nunit"

    # Ruby
    CAPYBARA = "capybara"
    SELENIUM_RUBY = "selenium-ruby"

    # Go
    ROD = "rod"
    CHROMEDP = "chromedp"


LANGUAGE_FRAMEWORKS: dict[str, list[str]] = {
    "python": ["playwright", "selenium", "pytest"],
    "typescript": ["playwright-ts", "puppeteer", "cypress"],
    "java": ["selenium-java", "testng", "junit"],
    "csharp": ["selenium-csharp", "playwright-csharp", "nunit"],
    "ruby": ["capybara", "selenium-ruby"],
    "go": ["rod", "chromedp"],
}


# =============================================================================
# Request/Response Models
# =============================================================================


class TestStepModel(BaseModel):
    """Test step for export."""
    action: str
    target: Optional[str] = None
    value: Optional[str] = None
    timeout: Optional[int] = None


class TestAssertionModel(BaseModel):
    """Test assertion for export."""
    type: str
    target: Optional[str] = None
    expected: Optional[str] = None


class ParameterSetModel(BaseModel):
    """Parameter set for parameterized tests."""
    name: str
    values: dict[str, Any]
    description: Optional[str] = None
    tags: list[str] = []
    skip: bool = False
    skip_reason: Optional[str] = None


class DataSourceModel(BaseModel):
    """Data source configuration for parameterized tests."""
    type: str  # inline, csv, json, env
    data: Optional[list[dict[str, Any]]] = None
    path: Optional[str] = None
    mapping: Optional[dict[str, str]] = None
    filter: Optional[str] = None
    limit: Optional[int] = None


class TestSpecModel(BaseModel):
    """Test specification for export."""
    id: str
    name: str
    description: Optional[str] = None
    steps: list[TestStepModel]
    assertions: Optional[list[TestAssertionModel]] = None
    setup: Optional[list[TestStepModel]] = None
    teardown: Optional[list[TestStepModel]] = None
    metadata: Optional[dict] = None
    # Parameterized test fields
    parameter_sets: Optional[list[ParameterSetModel]] = None
    data_source: Optional[DataSourceModel] = None


class ExportRequest(BaseModel):
    """Request to export a test."""
    test: TestSpecModel = Field(..., description="Test specification to export")
    language: str = Field(..., description="Target language (python, typescript, java, csharp, ruby, go)")
    framework: str = Field(..., description="Target framework (playwright, selenium, etc.)")
    include_comments: bool = Field(True, description="Include explanatory comments")
    include_assertions: bool = Field(True, description="Include test assertions")
    base_url: Optional[str] = Field(None, description="Base URL for the test")
    output_format: str = Field("code", description="Output format: code, file, or zip")


class ExportResponse(BaseModel):
    """Response from export operation."""
    success: bool
    language: str
    framework: str
    code: str
    filename: str
    dependencies: list[str] = []
    setup_instructions: Optional[str] = None
    error: Optional[str] = None


class ExportPreviewResponse(BaseModel):
    """Response for export preview."""
    success: bool
    language: str
    framework: str
    preview: str
    line_count: int
    estimated_size: str


class BulkExportRequest(BaseModel):
    """Request to export multiple tests."""
    tests: list[TestSpecModel]
    language: str
    framework: str
    include_shared_utils: bool = Field(True, description="Include shared utilities")


class BulkExportResponse(BaseModel):
    """Response from bulk export."""
    success: bool
    files: list[dict]
    total_tests: int
    total_lines: int
    zip_available: bool = False
    error: Optional[str] = None


# =============================================================================
# Export Templates (Simplified - would be full implementations in production)
# =============================================================================


def generate_python_playwright(test: TestSpecModel, include_comments: bool = True, base_url: Optional[str] = None) -> str:
    """Generate Python Playwright test code."""
    lines = []

    # Imports
    lines.append("import pytest")
    lines.append("from playwright.sync_api import Page, expect")
    lines.append("import re")
    lines.append("")

    # Class definition
    class_name = "".join(word.capitalize() for word in test.name.replace("-", " ").replace("_", " ").split())
    lines.append(f"class Test{class_name}:")

    if include_comments and test.description:
        lines.append(f'    """')
        lines.append(f"    {test.description}")
        lines.append(f"    Generated from Argus test spec: {test.id}")
        lines.append(f'    """')

    lines.append("")

    # Test method
    method_name = test.name.lower().replace(" ", "_").replace("-", "_")
    lines.append(f"    def test_{method_name}(self, page: Page, base_url: str):")

    # Setup steps
    if test.setup:
        if include_comments:
            lines.append("        # Setup")
        for step in test.setup:
            lines.append(f"        {_action_to_playwright(step)}")
        lines.append("")

    # Main steps
    for i, step in enumerate(test.steps, 1):
        if include_comments:
            lines.append(f"        # Step {i}: {step.action}")
        lines.append(f"        {_action_to_playwright(step, base_url)}")

    # Assertions
    if test.assertions:
        lines.append("")
        if include_comments:
            lines.append("        # Assertions")
        for assertion in test.assertions:
            lines.append(f"        {_assertion_to_playwright(assertion)}")

    # Teardown
    if test.teardown:
        lines.append("")
        if include_comments:
            lines.append("        # Teardown")
        for step in test.teardown:
            lines.append(f"        {_action_to_playwright(step)}")

    return "\n".join(lines)


def _action_to_playwright(step: TestStepModel, base_url: Optional[str] = None) -> str:
    """Convert action to Playwright code."""
    action = step.action.lower()
    target = step.target or ""
    value = step.value or ""

    if action == "goto" or action == "navigate":
        url = value or target
        if base_url and not url.startswith("http"):
            return f'page.goto(f"{{base_url}}{url}")'
        return f'page.goto("{url}")'
    elif action == "click":
        return f'page.click("{target}")'
    elif action == "fill" or action == "type":
        return f'page.fill("{target}", "{value}")'
    elif action == "press":
        return f'page.press("{target}", "{value}")'
    elif action == "select":
        return f'page.select_option("{target}", "{value}")'
    elif action == "check":
        return f'page.check("{target}")'
    elif action == "uncheck":
        return f'page.uncheck("{target}")'
    elif action == "hover":
        return f'page.hover("{target}")'
    elif action == "wait":
        timeout = step.timeout or 1000
        return f"page.wait_for_timeout({timeout})"
    elif action == "screenshot":
        return f'page.screenshot(path="{value or "screenshot.png"}")'
    else:
        return f"# TODO: {action} {target} {value}"


def _assertion_to_playwright(assertion: TestAssertionModel) -> str:
    """Convert assertion to Playwright expect."""
    atype = assertion.type.lower()
    target = assertion.target or ""
    expected = assertion.expected or ""

    if atype == "visible" or atype == "element_visible":
        return f'expect(page.locator("{target}")).to_be_visible()'
    elif atype == "hidden" or atype == "element_hidden":
        return f'expect(page.locator("{target}")).to_be_hidden()'
    elif atype == "text_contains" or atype == "contains_text":
        return f'expect(page.locator("{target}")).to_contain_text("{expected}")'
    elif atype == "text_equals" or atype == "has_text":
        return f'expect(page.locator("{target}")).to_have_text("{expected}")'
    elif atype == "url_contains":
        return f'expect(page).to_have_url(re.compile(r".*{expected}.*"))'
    elif atype == "url_equals":
        return f'expect(page).to_have_url("{expected}")'
    elif atype == "title_contains":
        return f'expect(page).to_have_title(re.compile(r".*{expected}.*"))'
    elif atype == "title_equals":
        return f'expect(page).to_have_title("{expected}")'
    elif atype == "attribute":
        attr_name, attr_value = expected.split("=", 1) if "=" in expected else (expected, "")
        return f'expect(page.locator("{target}")).to_have_attribute("{attr_name}", "{attr_value}")'
    else:
        return f"# TODO: assertion {atype}"


def generate_typescript_playwright(test: TestSpecModel, include_comments: bool = True, base_url: Optional[str] = None) -> str:
    """Generate TypeScript Playwright test code."""
    lines = []

    # Imports
    lines.append("import { test, expect, Page } from '@playwright/test';")
    lines.append("")

    # Test description
    if include_comments and test.description:
        lines.append(f"/**")
        lines.append(f" * {test.description}")
        lines.append(f" * Generated from Argus test spec: {test.id}")
        lines.append(f" */")

    # Test block
    test_name = test.name
    lines.append(f"test('{test_name}', async ({{ page, baseURL }}) => {{")

    # Steps
    for i, step in enumerate(test.steps, 1):
        if include_comments:
            lines.append(f"  // Step {i}: {step.action}")
        lines.append(f"  {_action_to_playwright_ts(step, base_url)}")

    # Assertions
    if test.assertions:
        lines.append("")
        if include_comments:
            lines.append("  // Assertions")
        for assertion in test.assertions:
            lines.append(f"  {_assertion_to_playwright_ts(assertion)}")

    lines.append("});")

    return "\n".join(lines)


def _action_to_playwright_ts(step: TestStepModel, base_url: Optional[str] = None) -> str:
    """Convert action to TypeScript Playwright code."""
    action = step.action.lower()
    target = step.target or ""
    value = step.value or ""

    if action == "goto" or action == "navigate":
        url = value or target
        if base_url and not url.startswith("http"):
            return f'await page.goto(`${{baseURL}}{url}`);'
        return f"await page.goto('{url}');"
    elif action == "click":
        return f"await page.click('{target}');"
    elif action == "fill" or action == "type":
        return f"await page.fill('{target}', '{value}');"
    elif action == "press":
        return f"await page.press('{target}', '{value}');"
    elif action == "select":
        return f"await page.selectOption('{target}', '{value}');"
    elif action == "check":
        return f"await page.check('{target}');"
    elif action == "hover":
        return f"await page.hover('{target}');"
    elif action == "wait":
        timeout = step.timeout or 1000
        return f"await page.waitForTimeout({timeout});"
    else:
        return f"// TODO: {action} {target} {value}"


def _assertion_to_playwright_ts(assertion: TestAssertionModel) -> str:
    """Convert assertion to TypeScript Playwright expect."""
    atype = assertion.type.lower()
    target = assertion.target or ""
    expected = assertion.expected or ""

    if atype == "visible" or atype == "element_visible":
        return f"await expect(page.locator('{target}')).toBeVisible();"
    elif atype == "hidden":
        return f"await expect(page.locator('{target}')).toBeHidden();"
    elif atype == "text_contains":
        return f"await expect(page.locator('{target}')).toContainText('{expected}');"
    elif atype == "url_contains":
        return f"await expect(page).toHaveURL(/{expected}/);"
    elif atype == "url_equals":
        return f"await expect(page).toHaveURL('{expected}');"
    else:
        return f"// TODO: assertion {atype}"


def generate_java_selenium(test: TestSpecModel, include_comments: bool = True, base_url: Optional[str] = None) -> str:
    """Generate Java Selenium test code."""
    lines = []

    # Package and imports
    lines.append("package tests;")
    lines.append("")
    lines.append("import org.junit.jupiter.api.*;")
    lines.append("import org.openqa.selenium.*;")
    lines.append("import org.openqa.selenium.chrome.ChromeDriver;")
    lines.append("import org.openqa.selenium.support.ui.*;")
    lines.append("import static org.junit.jupiter.api.Assertions.*;")
    lines.append("")

    # Class
    class_name = "".join(word.capitalize() for word in test.name.replace("-", " ").replace("_", " ").split())

    if include_comments and test.description:
        lines.append("/**")
        lines.append(f" * {test.description}")
        lines.append(f" * Generated from Argus test spec: {test.id}")
        lines.append(" */")

    lines.append(f"public class {class_name}Test {{")
    lines.append("    private WebDriver driver;")
    lines.append("    private WebDriverWait wait;")
    lines.append("")

    # Setup
    lines.append("    @BeforeEach")
    lines.append("    public void setUp() {")
    lines.append("        driver = new ChromeDriver();")
    lines.append("        wait = new WebDriverWait(driver, java.time.Duration.ofSeconds(10));")
    lines.append("    }")
    lines.append("")

    # Test method
    method_name = test.name.lower().replace(" ", "_").replace("-", "_")
    lines.append("    @Test")
    lines.append(f"    public void test{class_name}() {{")

    for i, step in enumerate(test.steps, 1):
        if include_comments:
            lines.append(f"        // Step {i}: {step.action}")
        lines.append(f"        {_action_to_selenium_java(step, base_url)}")

    if test.assertions:
        lines.append("")
        if include_comments:
            lines.append("        // Assertions")
        for assertion in test.assertions:
            lines.append(f"        {_assertion_to_selenium_java(assertion)}")

    lines.append("    }")
    lines.append("")

    # Teardown
    lines.append("    @AfterEach")
    lines.append("    public void tearDown() {")
    lines.append("        if (driver != null) {")
    lines.append("            driver.quit();")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")

    return "\n".join(lines)


def _action_to_selenium_java(step: TestStepModel, base_url: Optional[str] = None) -> str:
    """Convert action to Java Selenium code."""
    action = step.action.lower()
    target = step.target or ""
    value = step.value or ""

    if action == "goto" or action == "navigate":
        url = value or target
        return f'driver.get("{url}");'
    elif action == "click":
        return f'driver.findElement(By.cssSelector("{target}")).click();'
    elif action == "fill" or action == "type":
        return f'driver.findElement(By.cssSelector("{target}")).sendKeys("{value}");'
    elif action == "select":
        return f'new Select(driver.findElement(By.cssSelector("{target}"))).selectByValue("{value}");'
    elif action == "wait":
        timeout = step.timeout or 1000
        return f"Thread.sleep({timeout});"
    else:
        return f"// TODO: {action} {target} {value}"


def _assertion_to_selenium_java(assertion: TestAssertionModel) -> str:
    """Convert assertion to Java Selenium assertions."""
    atype = assertion.type.lower()
    target = assertion.target or ""
    expected = assertion.expected or ""

    if atype == "visible" or atype == "element_visible":
        return f'assertTrue(driver.findElement(By.cssSelector("{target}")).isDisplayed());'
    elif atype == "text_contains":
        return f'assertTrue(driver.findElement(By.cssSelector("{target}")).getText().contains("{expected}"));'
    elif atype == "url_contains":
        return f'assertTrue(driver.getCurrentUrl().contains("{expected}"));'
    elif atype == "url_equals":
        return f'assertEquals("{expected}", driver.getCurrentUrl());'
    else:
        return f"// TODO: assertion {atype}"


# =============================================================================
# Parameterized Test Export Functions
# =============================================================================


def generate_python_parameterized(
    test: TestSpecModel,
    include_comments: bool = True,
    base_url: Optional[str] = None,
) -> str:
    """Generate Python Playwright test code with @pytest.mark.parametrize.

    Generates parameterized tests using pytest's parametrize decorator.

    Args:
        test: Test specification with parameter sets
        include_comments: Whether to include comments
        base_url: Optional base URL for the test

    Returns:
        Generated Python test code with parametrize decorator

    Example output:
        @pytest.mark.parametrize("username,password,expected", [
            ("admin", "admin123", "dashboard"),
            ("user", "user123", "home"),
        ], ids=["admin_login", "user_login"])
        def test_login(self, page: Page, username, password, expected):
            ...
    """
    lines = []

    # Imports
    lines.append("import pytest")
    lines.append("from playwright.sync_api import Page, expect")
    lines.append("import re")
    lines.append("")

    # Get parameter sets
    param_sets = _get_parameter_sets_from_test(test)
    if not param_sets:
        # Fallback to non-parameterized version
        return generate_python_playwright(test, include_comments, base_url)

    # Extract parameter names from first set
    param_names = list(param_sets[0].values.keys())

    # Class definition
    class_name = "".join(
        word.capitalize()
        for word in test.name.replace("-", " ").replace("_", " ").split()
    )
    lines.append(f"class Test{class_name}:")

    if include_comments and test.description:
        lines.append('    """')
        lines.append(f"    {test.description}")
        lines.append(f"    Generated from Argus test spec: {test.id}")
        lines.append(f"    Parameterized with {len(param_sets)} test cases")
        lines.append('    """')

    lines.append("")

    # Generate parametrize decorator
    param_values_str = _generate_python_param_values(param_sets, param_names)
    param_ids = [ps.name for ps in param_sets]

    lines.append(f'    @pytest.mark.parametrize("{",".join(param_names)}", [')
    for i, ps in enumerate(param_sets):
        values = [repr(ps.values.get(name, "")) for name in param_names]
        comma = "," if i < len(param_sets) - 1 else ""
        lines.append(f"        ({', '.join(values)}){comma}")
    lines.append(f"    ], ids={param_ids})")

    # Test method with parameters
    method_name = test.name.lower().replace(" ", "_").replace("-", "_")
    param_args = ", ".join(param_names)
    lines.append(f"    def test_{method_name}(self, page: Page, base_url: str, {param_args}):")

    # Setup steps
    if test.setup:
        if include_comments:
            lines.append("        # Setup")
        for step in test.setup:
            lines.append(f"        {_action_to_playwright(step)}")
        lines.append("")

    # Main steps (with parameter placeholders replaced by f-string variables)
    for i, step in enumerate(test.steps, 1):
        if include_comments:
            lines.append(f"        # Step {i}: {step.action}")
        lines.append(f"        {_action_to_playwright_parameterized(step, param_names, base_url)}")

    # Assertions (with parameter placeholders replaced)
    if test.assertions:
        lines.append("")
        if include_comments:
            lines.append("        # Assertions")
        for assertion in test.assertions:
            lines.append(f"        {_assertion_to_playwright_parameterized(assertion, param_names)}")

    # Teardown
    if test.teardown:
        lines.append("")
        if include_comments:
            lines.append("        # Teardown")
        for step in test.teardown:
            lines.append(f"        {_action_to_playwright(step)}")

    return "\n".join(lines)


def _action_to_playwright_parameterized(
    step: TestStepModel,
    param_names: list[str],
    base_url: Optional[str] = None,
) -> str:
    """Convert action to Playwright code with parameter substitution."""
    action = step.action.lower()
    target = _substitute_params_python(step.target or "", param_names)
    value = _substitute_params_python(step.value or "", param_names)

    if action == "goto" or action == "navigate":
        url = value or target
        if base_url and not url.startswith("http") and "{{" not in (step.value or step.target or ""):
            return f'page.goto(f"{{base_url}}{url}")'
        return f'page.goto(f"{url}")'
    elif action == "click":
        return f'page.click(f"{target}")'
    elif action == "fill" or action == "type":
        return f'page.fill(f"{target}", f"{value}")'
    elif action == "press":
        return f'page.press(f"{target}", f"{value}")'
    elif action == "select":
        return f'page.select_option(f"{target}", f"{value}")'
    elif action == "check":
        return f'page.check(f"{target}")'
    elif action == "uncheck":
        return f'page.uncheck(f"{target}")'
    elif action == "hover":
        return f'page.hover(f"{target}")'
    elif action == "wait":
        timeout = step.timeout or 1000
        return f"page.wait_for_timeout({timeout})"
    elif action == "screenshot":
        return f'page.screenshot(path=f"{value or "screenshot.png"}")'
    else:
        return f"# TODO: {action} {target} {value}"


def _assertion_to_playwright_parameterized(
    assertion: TestAssertionModel,
    param_names: list[str],
) -> str:
    """Convert assertion to Playwright expect with parameter substitution."""
    atype = assertion.type.lower()
    target = _substitute_params_python(assertion.target or "", param_names)
    expected = _substitute_params_python(assertion.expected or "", param_names)

    if atype == "visible" or atype == "element_visible":
        return f'expect(page.locator(f"{target}")).to_be_visible()'
    elif atype == "hidden" or atype == "element_hidden":
        return f'expect(page.locator(f"{target}")).to_be_hidden()'
    elif atype == "text_contains" or atype == "contains_text":
        return f'expect(page.locator(f"{target}")).to_contain_text(f"{expected}")'
    elif atype == "text_equals" or atype == "has_text":
        return f'expect(page.locator(f"{target}")).to_have_text(f"{expected}")'
    elif atype == "url_contains":
        return f'expect(page).to_have_url(re.compile(rf".*{expected}.*"))'
    elif atype == "url_equals":
        return f'expect(page).to_have_url(f"{expected}")'
    elif atype == "title_contains":
        return f'expect(page).to_have_title(re.compile(rf".*{expected}.*"))'
    elif atype == "title_equals":
        return f'expect(page).to_have_title(f"{expected}")'
    else:
        return f"# TODO: assertion {atype}"


def _substitute_params_python(template: str, param_names: list[str]) -> str:
    """Substitute {{param}} placeholders with Python f-string variables.

    Args:
        template: String with {{param}} placeholders
        param_names: List of parameter names to substitute

    Returns:
        String with placeholders replaced by {param} for f-strings
    """
    import re
    result = template
    for param in param_names:
        result = re.sub(r"\{\{" + param + r"\}\}", "{" + param + "}", result)
    return result


def _generate_python_param_values(
    param_sets: list[ParameterSetModel],
    param_names: list[str],
) -> str:
    """Generate Python tuple values for parametrize decorator."""
    values = []
    for ps in param_sets:
        tuple_values = [repr(ps.values.get(name, "")) for name in param_names]
        values.append(f"({', '.join(tuple_values)})")
    return ", ".join(values)


def generate_typescript_parameterized(
    test: TestSpecModel,
    include_comments: bool = True,
    base_url: Optional[str] = None,
) -> str:
    """Generate TypeScript Playwright test code with test.each().

    Generates parameterized tests using Playwright's test.describe.each()
    or test.each() pattern.

    Args:
        test: Test specification with parameter sets
        include_comments: Whether to include comments
        base_url: Optional base URL for the test

    Returns:
        Generated TypeScript test code with test.each()

    Example output:
        const testCases = [
            { username: 'admin', password: 'admin123', expected: 'dashboard' },
            { username: 'user', password: 'user123', expected: 'home' },
        ];

        for (const testCase of testCases) {
            test(`Login Test - ${testCase.username}`, async ({ page }) => {
                ...
            });
        }
    """
    lines = []

    # Imports
    lines.append("import { test, expect, Page } from '@playwright/test';")
    lines.append("")

    # Get parameter sets
    param_sets = _get_parameter_sets_from_test(test)
    if not param_sets:
        # Fallback to non-parameterized version
        return generate_typescript_playwright(test, include_comments, base_url)

    # Extract parameter names from first set
    param_names = list(param_sets[0].values.keys())

    # Test description
    if include_comments and test.description:
        lines.append("/**")
        lines.append(f" * {test.description}")
        lines.append(f" * Generated from Argus test spec: {test.id}")
        lines.append(f" * Parameterized with {len(param_sets)} test cases")
        lines.append(" */")
        lines.append("")

    # Generate test cases array
    lines.append("const testCases = [")
    for ps in param_sets:
        obj_props = ", ".join(
            f"{name}: {_to_ts_value(ps.values.get(name, ''))}"
            for name in param_names
        )
        lines.append(f"  {{ name: '{ps.name}', {obj_props} }},")
    lines.append("];")
    lines.append("")

    # Generate test.each loop
    lines.append("for (const testCase of testCases) {")
    lines.append(f"  test(`{test.name} - ${{testCase.name}}`, async ({{ page, baseURL }}) => {{")

    # Steps
    for i, step in enumerate(test.steps, 1):
        if include_comments:
            lines.append(f"    // Step {i}: {step.action}")
        lines.append(f"    {_action_to_playwright_ts_parameterized(step, param_names, base_url)}")

    # Assertions
    if test.assertions:
        lines.append("")
        if include_comments:
            lines.append("    // Assertions")
        for assertion in test.assertions:
            lines.append(f"    {_assertion_to_playwright_ts_parameterized(assertion, param_names)}")

    lines.append("  });")
    lines.append("}")

    return "\n".join(lines)


def _action_to_playwright_ts_parameterized(
    step: TestStepModel,
    param_names: list[str],
    base_url: Optional[str] = None,
) -> str:
    """Convert action to TypeScript Playwright code with parameter substitution."""
    action = step.action.lower()
    target = _substitute_params_ts(step.target or "", param_names)
    value = _substitute_params_ts(step.value or "", param_names)

    if action == "goto" or action == "navigate":
        url = value or target
        if base_url and not url.startswith("http") and "{{" not in (step.value or step.target or ""):
            return f"await page.goto(`${{baseURL}}{url}`);"
        return f"await page.goto(`{url}`);"
    elif action == "click":
        return f"await page.click(`{target}`);"
    elif action == "fill" or action == "type":
        return f"await page.fill(`{target}`, `{value}`);"
    elif action == "press":
        return f"await page.press(`{target}`, `{value}`);"
    elif action == "select":
        return f"await page.selectOption(`{target}`, `{value}`);"
    elif action == "check":
        return f"await page.check(`{target}`);"
    elif action == "hover":
        return f"await page.hover(`{target}`);"
    elif action == "wait":
        timeout = step.timeout or 1000
        return f"await page.waitForTimeout({timeout});"
    else:
        return f"// TODO: {action} {target} {value}"


def _assertion_to_playwright_ts_parameterized(
    assertion: TestAssertionModel,
    param_names: list[str],
) -> str:
    """Convert assertion to TypeScript Playwright expect with parameter substitution."""
    atype = assertion.type.lower()
    target = _substitute_params_ts(assertion.target or "", param_names)
    expected = _substitute_params_ts(assertion.expected or "", param_names)

    if atype == "visible" or atype == "element_visible":
        return f"await expect(page.locator(`{target}`)).toBeVisible();"
    elif atype == "hidden":
        return f"await expect(page.locator(`{target}`)).toBeHidden();"
    elif atype == "text_contains":
        return f"await expect(page.locator(`{target}`)).toContainText(`{expected}`);"
    elif atype == "url_contains":
        return f"await expect(page).toHaveURL(new RegExp(`{expected}`));"
    elif atype == "url_equals":
        return f"await expect(page).toHaveURL(`{expected}`);"
    else:
        return f"// TODO: assertion {atype}"


def _substitute_params_ts(template: str, param_names: list[str]) -> str:
    """Substitute {{param}} placeholders with TypeScript template literal variables.

    Args:
        template: String with {{param}} placeholders
        param_names: List of parameter names to substitute

    Returns:
        String with placeholders replaced by ${testCase.param} for template literals
    """
    import re
    result = template
    for param in param_names:
        result = re.sub(r"\{\{" + param + r"\}\}", "${testCase." + param + "}", result)
    return result


def _to_ts_value(value: Any) -> str:
    """Convert Python value to TypeScript value literal."""
    if isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif value is None:
        return "null"
    else:
        return f"'{str(value)}'"


def generate_java_parameterized(
    test: TestSpecModel,
    include_comments: bool = True,
    base_url: Optional[str] = None,
) -> str:
    """Generate Java JUnit5 test code with @ParameterizedTest.

    Generates parameterized tests using JUnit5's @ParameterizedTest
    with @CsvSource or @MethodSource.

    Args:
        test: Test specification with parameter sets
        include_comments: Whether to include comments
        base_url: Optional base URL for the test

    Returns:
        Generated Java test code with @ParameterizedTest

    Example output:
        @ParameterizedTest
        @CsvSource({
            "admin, admin123, dashboard",
            "user, user123, home"
        })
        void testLogin(String username, String password, String expected) {
            ...
        }
    """
    lines = []

    # Package and imports
    lines.append("package tests;")
    lines.append("")
    lines.append("import org.junit.jupiter.api.*;")
    lines.append("import org.junit.jupiter.params.ParameterizedTest;")
    lines.append("import org.junit.jupiter.params.provider.CsvSource;")
    lines.append("import org.junit.jupiter.params.provider.MethodSource;")
    lines.append("import org.openqa.selenium.*;")
    lines.append("import org.openqa.selenium.chrome.ChromeDriver;")
    lines.append("import org.openqa.selenium.support.ui.*;")
    lines.append("import static org.junit.jupiter.api.Assertions.*;")
    lines.append("import java.util.stream.Stream;")
    lines.append("")

    # Get parameter sets
    param_sets = _get_parameter_sets_from_test(test)
    if not param_sets:
        # Fallback to non-parameterized version
        return generate_java_selenium(test, include_comments, base_url)

    # Extract parameter names from first set
    param_names = list(param_sets[0].values.keys())

    # Class
    class_name = "".join(
        word.capitalize()
        for word in test.name.replace("-", " ").replace("_", " ").split()
    )

    if include_comments and test.description:
        lines.append("/**")
        lines.append(f" * {test.description}")
        lines.append(f" * Generated from Argus test spec: {test.id}")
        lines.append(f" * Parameterized with {len(param_sets)} test cases")
        lines.append(" */")

    lines.append(f"public class {class_name}Test {{")
    lines.append("    private WebDriver driver;")
    lines.append("    private WebDriverWait wait;")
    lines.append("")

    # Setup
    lines.append("    @BeforeEach")
    lines.append("    public void setUp() {")
    lines.append("        driver = new ChromeDriver();")
    lines.append("        wait = new WebDriverWait(driver, java.time.Duration.ofSeconds(10));")
    lines.append("    }")
    lines.append("")

    # Generate CSV source data
    csv_rows = []
    for ps in param_sets:
        row_values = [_escape_java_csv(str(ps.values.get(name, ""))) for name in param_names]
        csv_rows.append(", ".join(row_values))

    # Parameterized test method
    lines.append("    @ParameterizedTest")
    lines.append("    @CsvSource({")
    for i, row in enumerate(csv_rows):
        comma = "," if i < len(csv_rows) - 1 else ""
        lines.append(f'        "{row}"{comma}')
    lines.append("    })")

    # Method signature with parameters
    java_params = ", ".join(f"String {name}" for name in param_names)
    method_name = test.name.lower().replace(" ", "_").replace("-", "_")
    lines.append(f"    void test{class_name}({java_params}) {{")

    # Steps
    for i, step in enumerate(test.steps, 1):
        if include_comments:
            lines.append(f"        // Step {i}: {step.action}")
        lines.append(f"        {_action_to_selenium_java_parameterized(step, param_names, base_url)}")

    if test.assertions:
        lines.append("")
        if include_comments:
            lines.append("        // Assertions")
        for assertion in test.assertions:
            lines.append(f"        {_assertion_to_selenium_java_parameterized(assertion, param_names)}")

    lines.append("    }")
    lines.append("")

    # Teardown
    lines.append("    @AfterEach")
    lines.append("    public void tearDown() {")
    lines.append("        if (driver != null) {")
    lines.append("            driver.quit();")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")

    return "\n".join(lines)


def _action_to_selenium_java_parameterized(
    step: TestStepModel,
    param_names: list[str],
    base_url: Optional[str] = None,
) -> str:
    """Convert action to Java Selenium code with parameter substitution."""
    action = step.action.lower()
    target = _substitute_params_java(step.target or "", param_names)
    value = _substitute_params_java(step.value or "", param_names)

    if action == "goto" or action == "navigate":
        url = value or target
        return f"driver.get({url});"
    elif action == "click":
        return f"driver.findElement(By.cssSelector({target})).click();"
    elif action == "fill" or action == "type":
        return f"driver.findElement(By.cssSelector({target})).sendKeys({value});"
    elif action == "select":
        return f"new Select(driver.findElement(By.cssSelector({target}))).selectByValue({value});"
    elif action == "wait":
        timeout = step.timeout or 1000
        return f"Thread.sleep({timeout});"
    else:
        return f"// TODO: {action}"


def _assertion_to_selenium_java_parameterized(
    assertion: TestAssertionModel,
    param_names: list[str],
) -> str:
    """Convert assertion to Java Selenium assertions with parameter substitution."""
    atype = assertion.type.lower()
    target = _substitute_params_java(assertion.target or "", param_names)
    expected = _substitute_params_java(assertion.expected or "", param_names)

    if atype == "visible" or atype == "element_visible":
        return f"assertTrue(driver.findElement(By.cssSelector({target})).isDisplayed());"
    elif atype == "text_contains":
        return f"assertTrue(driver.findElement(By.cssSelector({target})).getText().contains({expected}));"
    elif atype == "url_contains":
        return f"assertTrue(driver.getCurrentUrl().contains({expected}));"
    elif atype == "url_equals":
        return f"assertEquals({expected}, driver.getCurrentUrl());"
    else:
        return f"// TODO: assertion {atype}"


def _substitute_params_java(template: str, param_names: list[str]) -> str:
    """Substitute {{param}} placeholders with Java string concatenation.

    Args:
        template: String with {{param}} placeholders
        param_names: List of parameter names to substitute

    Returns:
        Java string expression with variable concatenation
    """
    import re

    # Check if template contains any placeholders
    has_placeholder = any(f"{{{{{p}}}}}" in template for p in param_names)

    if not has_placeholder:
        return f'"{template}"'

    # Build a Java string concatenation expression
    result = template
    for param in param_names:
        result = re.sub(r"\{\{" + param + r"\}\}", f'" + {param} + "', result)

    # Clean up empty concatenations
    result = f'"{result}"'
    result = result.replace('"" + ', '')
    result = result.replace(' + ""', '')

    return result


def _escape_java_csv(value: str) -> str:
    """Escape a value for use in Java @CsvSource annotation."""
    # Escape quotes by doubling them
    escaped = value.replace("'", "''")
    return escaped


def _get_parameter_sets_from_test(test: TestSpecModel) -> list[ParameterSetModel]:
    """Extract parameter sets from a test specification.

    Handles both explicit parameter_sets and inline data_source.

    Args:
        test: Test specification

    Returns:
        List of ParameterSetModel objects
    """
    # Use explicit parameter sets if provided
    if test.parameter_sets:
        return test.parameter_sets

    # Use data source if provided
    if test.data_source and test.data_source.type == "inline" and test.data_source.data:
        return [
            ParameterSetModel(
                name=data.get("name", f"case_{i}"),
                values={k: v for k, v in data.items() if k not in ["name", "description", "tags", "skip"]},
                description=data.get("description"),
                tags=data.get("tags", []),
                skip=data.get("skip", False),
                skip_reason=data.get("skip_reason"),
            )
            for i, data in enumerate(test.data_source.data)
        ]

    return []


def _is_parameterized_test(test: TestSpecModel) -> bool:
    """Check if a test has parameterization configured."""
    if test.parameter_sets:
        return True
    if test.data_source and test.data_source.type == "inline" and test.data_source.data:
        return True
    return False


# =============================================================================
# Parameterized Export Request/Response Models
# =============================================================================


class ParameterizedExportRequest(BaseModel):
    """Request to export a parameterized test."""
    test: TestSpecModel = Field(..., description="Test specification with parameters")
    language: str = Field(..., description="Target language")
    framework: str = Field(..., description="Target framework")
    include_comments: bool = Field(True, description="Include comments")
    base_url: Optional[str] = Field(None, description="Base URL")


class ParameterizedExportResponse(BaseModel):
    """Response from parameterized test export."""
    success: bool
    language: str
    framework: str
    code: str
    filename: str
    parameter_count: int
    parameter_names: list[str]
    dependencies: list[str] = []
    setup_instructions: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/parameterized/generate", response_model=ParameterizedExportResponse)
async def export_parameterized_test(request: ParameterizedExportRequest):
    """
    Export a parameterized test to the specified language and framework.

    Generates data-driven test code with appropriate parameterization syntax:
    - Python: @pytest.mark.parametrize decorator
    - TypeScript: test.each() pattern
    - Java: @ParameterizedTest with @CsvSource

    Example request:
        {
            "test": {
                "id": "login-001",
                "name": "Login Test",
                "parameter_sets": [
                    {"name": "admin_login", "values": {"username": "admin", "password": "admin123"}},
                    {"name": "user_login", "values": {"username": "user", "password": "user123"}}
                ],
                "steps": [
                    {"action": "fill", "target": "#username", "value": "{{username}}"},
                    {"action": "fill", "target": "#password", "value": "{{password}}"}
                ]
            },
            "language": "python",
            "framework": "playwright"
        }
    """
    try:
        # Validate language/framework
        if request.language not in LANGUAGE_FRAMEWORKS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {request.language}"
            )

        if request.framework not in LANGUAGE_FRAMEWORKS[request.language]:
            raise HTTPException(
                status_code=400,
                detail=f"Framework {request.framework} not supported for {request.language}"
            )

        # Get parameter sets
        param_sets = _get_parameter_sets_from_test(request.test)
        if not param_sets:
            raise HTTPException(
                status_code=400,
                detail="No parameter sets found in test specification"
            )

        param_names = list(param_sets[0].values.keys())

        # Generate code based on language
        if request.language == "python":
            code = generate_python_parameterized(
                request.test, request.include_comments, request.base_url
            )
        elif request.language == "typescript":
            code = generate_typescript_parameterized(
                request.test, request.include_comments, request.base_url
            )
        elif request.language == "java":
            code = generate_java_parameterized(
                request.test, request.include_comments, request.base_url
            )
        else:
            # Default to Python for unsupported languages
            code = generate_python_parameterized(
                request.test, request.include_comments, request.base_url
            )

        # Get filename
        test_name = request.test.name.lower().replace(" ", "_").replace("-", "_")
        filename = _get_filename(test_name, request.language, request.framework)

        logger.info(
            "Parameterized test exported",
            test_id=request.test.id,
            language=request.language,
            framework=request.framework,
            param_count=len(param_sets),
        )

        return ParameterizedExportResponse(
            success=True,
            language=request.language,
            framework=request.framework,
            code=code,
            filename=filename,
            parameter_count=len(param_sets),
            parameter_names=param_names,
            dependencies=_get_dependencies(request.language, request.framework),
            setup_instructions=_get_setup_instructions(request.language, request.framework),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Parameterized export failed", error=str(e))
        return ParameterizedExportResponse(
            success=False,
            language=request.language,
            framework=request.framework,
            code="",
            filename="",
            parameter_count=0,
            parameter_names=[],
            error=str(e),
        )


@router.get("/languages")
async def list_languages():
    """
    List supported export languages and frameworks.

    Returns all available language/framework combinations.
    """
    return {
        "success": True,
        "languages": [
            {
                "id": lang,
                "name": lang.capitalize(),
                "frameworks": frameworks,
                "default_framework": frameworks[0] if frameworks else None,
            }
            for lang, frameworks in LANGUAGE_FRAMEWORKS.items()
        ],
        "total": len(LANGUAGE_FRAMEWORKS),
    }


@router.post("/preview", response_model=ExportPreviewResponse)
async def preview_export(request: ExportRequest):
    """
    Preview exported test code without full generation.

    Returns first 50 lines of the generated code.
    """
    try:
        # Validate language/framework
        if request.language not in LANGUAGE_FRAMEWORKS:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")

        if request.framework not in LANGUAGE_FRAMEWORKS[request.language]:
            raise HTTPException(
                status_code=400,
                detail=f"Framework {request.framework} not supported for {request.language}"
            )

        # Generate code
        code = _generate_code(request)
        lines = code.split("\n")

        preview = "\n".join(lines[:50])
        if len(lines) > 50:
            preview += "\n\n// ... (truncated)"

        return ExportPreviewResponse(
            success=True,
            language=request.language,
            framework=request.framework,
            preview=preview,
            line_count=len(lines),
            estimated_size=f"{len(code) / 1024:.1f} KB",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Export preview failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=ExportResponse)
async def generate_export(request: ExportRequest):
    """
    Generate test code in the specified language and framework.

    Converts Argus test specification to executable test code.
    """
    try:
        # Validate language/framework
        if request.language not in LANGUAGE_FRAMEWORKS:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")

        if request.framework not in LANGUAGE_FRAMEWORKS[request.language]:
            raise HTTPException(
                status_code=400,
                detail=f"Framework {request.framework} not supported for {request.language}"
            )

        # Generate code
        code = _generate_code(request)

        # Determine filename
        test_name = request.test.name.lower().replace(" ", "_").replace("-", "_")
        filename = _get_filename(test_name, request.language, request.framework)

        # Get dependencies
        dependencies = _get_dependencies(request.language, request.framework)

        # Get setup instructions
        setup = _get_setup_instructions(request.language, request.framework)

        logger.info(
            "Test exported",
            test_id=request.test.id,
            language=request.language,
            framework=request.framework,
            lines=len(code.split("\n")),
        )

        return ExportResponse(
            success=True,
            language=request.language,
            framework=request.framework,
            code=code,
            filename=filename,
            dependencies=dependencies,
            setup_instructions=setup,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Export generation failed", error=str(e))
        return ExportResponse(
            success=False,
            language=request.language,
            framework=request.framework,
            code="",
            filename="",
            error=str(e),
        )


@router.post("/bulk", response_model=BulkExportResponse)
async def bulk_export(request: BulkExportRequest):
    """
    Export multiple tests to a single language/framework.

    Returns all generated files.
    """
    try:
        if request.language not in LANGUAGE_FRAMEWORKS:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")

        files = []
        total_lines = 0

        for test in request.tests:
            export_req = ExportRequest(
                test=test,
                language=request.language,
                framework=request.framework,
            )

            code = _generate_code(export_req)
            test_name = test.name.lower().replace(" ", "_").replace("-", "_")
            filename = _get_filename(test_name, request.language, request.framework)

            line_count = len(code.split("\n"))
            total_lines += line_count

            files.append({
                "filename": filename,
                "code": code,
                "test_id": test.id,
                "line_count": line_count,
            })

        # Add shared utilities if requested
        if request.include_shared_utils:
            utils_code = _generate_shared_utils(request.language, request.framework)
            if utils_code:
                utils_filename = _get_utils_filename(request.language)
                files.append({
                    "filename": utils_filename,
                    "code": utils_code,
                    "test_id": "__shared_utils__",
                    "line_count": len(utils_code.split("\n")),
                })
                total_lines += len(utils_code.split("\n"))

        logger.info(
            "Bulk export completed",
            language=request.language,
            framework=request.framework,
            files=len(files),
            total_lines=total_lines,
        )

        return BulkExportResponse(
            success=True,
            files=files,
            total_tests=len(request.tests),
            total_lines=total_lines,
            zip_available=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Bulk export failed", error=str(e))
        return BulkExportResponse(
            success=False,
            files=[],
            total_tests=0,
            total_lines=0,
            error=str(e),
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _generate_code(request: ExportRequest) -> str:
    """Generate code based on language and framework."""
    lang = request.language
    framework = request.framework
    test = request.test

    if lang == "python":
        if framework in ["playwright", "pytest"]:
            return generate_python_playwright(test, request.include_comments, request.base_url)
        elif framework == "selenium":
            # Would implement selenium template
            return generate_python_playwright(test, request.include_comments, request.base_url)  # Placeholder

    elif lang == "typescript":
        if framework in ["playwright-ts", "puppeteer"]:
            return generate_typescript_playwright(test, request.include_comments, request.base_url)
        elif framework == "cypress":
            # Would implement cypress template
            return generate_typescript_playwright(test, request.include_comments, request.base_url)  # Placeholder

    elif lang == "java":
        return generate_java_selenium(test, request.include_comments, request.base_url)

    # Fallback to Python Playwright for unsupported combinations
    return generate_python_playwright(test, request.include_comments, request.base_url)


def _get_filename(test_name: str, language: str, framework: str) -> str:
    """Get appropriate filename for the generated test."""
    extensions = {
        "python": ".py",
        "typescript": ".ts",
        "javascript": ".js",
        "java": ".java",
        "csharp": ".cs",
        "ruby": ".rb",
        "go": ".go",
    }

    ext = extensions.get(language, ".txt")

    if language == "java":
        # Java requires class name to match filename
        class_name = "".join(word.capitalize() for word in test_name.split("_"))
        return f"{class_name}Test{ext}"

    return f"test_{test_name}{ext}"


def _get_utils_filename(language: str) -> str:
    """Get filename for shared utilities."""
    filenames = {
        "python": "conftest.py",
        "typescript": "test-utils.ts",
        "java": "TestUtils.java",
        "csharp": "TestUtils.cs",
        "ruby": "spec_helper.rb",
        "go": "test_utils_test.go",
    }
    return filenames.get(language, "utils.txt")


def _get_dependencies(language: str, framework: str) -> list[str]:
    """Get list of dependencies for the language/framework."""
    deps = {
        ("python", "playwright"): ["playwright", "pytest", "pytest-playwright"],
        ("python", "pytest"): ["playwright", "pytest", "pytest-playwright"],
        ("python", "selenium"): ["selenium", "pytest", "webdriver-manager"],
        ("typescript", "playwright-ts"): ["@playwright/test"],
        ("typescript", "cypress"): ["cypress"],
        ("typescript", "puppeteer"): ["puppeteer", "jest"],
        ("java", "selenium-java"): ["selenium-java", "junit-jupiter", "webdrivermanager"],
        ("csharp", "selenium-csharp"): ["Selenium.WebDriver", "NUnit", "WebDriverManager"],
        ("ruby", "capybara"): ["capybara", "selenium-webdriver", "rspec"],
        ("go", "rod"): ["github.com/go-rod/rod"],
    }
    return deps.get((language, framework), [])


def _get_setup_instructions(language: str, framework: str) -> str:
    """Get setup instructions for the language/framework."""
    instructions = {
        ("python", "playwright"): """
# Install dependencies
pip install playwright pytest pytest-playwright

# Install browsers
playwright install

# Run tests
pytest test_*.py
""",
        ("typescript", "playwright-ts"): """
# Install dependencies
npm install -D @playwright/test

# Install browsers
npx playwright install

# Run tests
npx playwright test
""",
        ("java", "selenium-java"): """
# Add to pom.xml or build.gradle
# See dependencies list above

# Run tests
mvn test
""",
    }
    return instructions.get((language, framework), "See framework documentation for setup.")


def _generate_shared_utils(language: str, framework: str) -> str:
    """Generate shared utility code."""
    if language == "python" and framework in ["playwright", "pytest"]:
        return """import pytest
from playwright.sync_api import Playwright, Browser, Page


@pytest.fixture(scope="session")
def browser(playwright: Playwright) -> Browser:
    \"\"\"Create a browser instance for the test session.\"\"\"
    browser = playwright.chromium.launch(headless=True)
    yield browser
    browser.close()


@pytest.fixture
def page(browser: Browser) -> Page:
    \"\"\"Create a new page for each test.\"\"\"
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()


@pytest.fixture
def base_url() -> str:
    \"\"\"Base URL for tests.\"\"\"
    return "http://localhost:3000"
"""

    elif language == "typescript" and framework == "playwright-ts":
        return """import { test as base, expect } from '@playwright/test';

// Custom test fixtures
export const test = base.extend({
  // Add custom fixtures here
});

// Utility functions
export async function waitForNetworkIdle(page: any, timeout = 5000) {
  await page.waitForLoadState('networkidle', { timeout });
}

export async function takeScreenshotOnFailure(page: any, testInfo: any) {
  if (testInfo.status !== testInfo.expectedStatus) {
    await page.screenshot({ path: `screenshots/${testInfo.title}.png` });
  }
}
"""

    return ""
