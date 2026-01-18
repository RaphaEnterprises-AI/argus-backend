"""Data models for test export."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SupportedLanguage(str, Enum):
    """Supported programming languages for export."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    RUBY = "ruby"
    GO = "go"


class SupportedFramework(str, Enum):
    """Supported test frameworks for export."""

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

    # C#
    SELENIUM_CSHARP = "selenium-csharp"
    PLAYWRIGHT_CSHARP = "playwright-csharp"
    NUNIT = "nunit"

    # Ruby
    CAPYBARA = "capybara"
    SELENIUM_RUBY = "selenium-ruby"

    # Go
    ROD = "rod"


# Valid language-framework combinations
FRAMEWORK_SUPPORT = {
    SupportedLanguage.PYTHON: [
        SupportedFramework.PLAYWRIGHT_PYTHON,
        SupportedFramework.SELENIUM_PYTHON,
    ],
    SupportedLanguage.TYPESCRIPT: [
        SupportedFramework.PLAYWRIGHT_TS,
        SupportedFramework.PUPPETEER,
        SupportedFramework.CYPRESS,
    ],
    SupportedLanguage.JAVA: [
        SupportedFramework.SELENIUM_JAVA,
    ],
    SupportedLanguage.CSHARP: [
        SupportedFramework.SELENIUM_CSHARP,
        SupportedFramework.PLAYWRIGHT_CSHARP,
    ],
    SupportedLanguage.RUBY: [
        SupportedFramework.CAPYBARA,
        SupportedFramework.SELENIUM_RUBY,
    ],
    SupportedLanguage.GO: [
        SupportedFramework.ROD,
    ],
}


@dataclass
class ExportConfig:
    """Configuration for test export.

    Attributes:
        language: Target programming language
        framework: Target test framework
        include_comments: Whether to include code comments
        include_assertions: Whether to generate assertion code
        base_url_variable: Variable name for base URL
        test_class_name: Name for generated test class
        async_style: Use async/await style where supported
        page_object_pattern: Generate Page Object pattern code
        data_driven: Generate data-driven test structure
    """

    language: SupportedLanguage | str = SupportedLanguage.PYTHON
    framework: SupportedFramework | str = SupportedFramework.PLAYWRIGHT_PYTHON
    include_comments: bool = True
    include_assertions: bool = True
    base_url_variable: str = "base_url"
    test_class_name: str | None = None
    async_style: bool = True
    page_object_pattern: bool = False
    data_driven: bool = False
    custom_imports: list[str] = field(default_factory=list)
    custom_fixtures: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Convert string values to enums."""
        if isinstance(self.language, str):
            self.language = SupportedLanguage(self.language.lower())
        if isinstance(self.framework, str):
            self.framework = SupportedFramework(self.framework.lower())

    def validate(self) -> list[str]:
        """Validate the configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self.language not in FRAMEWORK_SUPPORT:
            errors.append(f"Unsupported language: {self.language}")
            return errors

        valid_frameworks = FRAMEWORK_SUPPORT[self.language]
        if self.framework not in valid_frameworks:
            valid_names = [f.value for f in valid_frameworks]
            errors.append(
                f"Framework '{self.framework.value}' is not supported for "
                f"{self.language.value}. Valid options: {valid_names}"
            )

        return errors


@dataclass
class ExportResult:
    """Result from exporting a test.

    Attributes:
        success: Whether export succeeded
        code: Generated code
        language: Language code was generated for
        framework: Framework code was generated for
        file_extension: Suggested file extension
        imports: List of required imports/dependencies
        error: Error message if failed
        metadata: Additional export metadata
    """

    success: bool
    code: str = ""
    language: SupportedLanguage | None = None
    framework: SupportedFramework | None = None
    file_extension: str = ".py"
    imports: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "success": self.success,
            "code": self.code,
            "language": self.language.value if self.language else None,
            "framework": self.framework.value if self.framework else None,
            "file_extension": self.file_extension,
            "imports": self.imports,
            "dependencies": self.dependencies,
            "error": self.error,
            "metadata": self.metadata,
        }


# File extensions for each language
FILE_EXTENSIONS = {
    SupportedLanguage.PYTHON: ".py",
    SupportedLanguage.TYPESCRIPT: ".ts",
    SupportedLanguage.JAVA: ".java",
    SupportedLanguage.CSHARP: ".cs",
    SupportedLanguage.RUBY: ".rb",
    SupportedLanguage.GO: ".go",
}


# Dependencies for each framework
FRAMEWORK_DEPENDENCIES = {
    SupportedFramework.PLAYWRIGHT_PYTHON: ["playwright", "pytest", "pytest-playwright"],
    SupportedFramework.SELENIUM_PYTHON: ["selenium", "pytest", "webdriver-manager"],
    SupportedFramework.PLAYWRIGHT_TS: ["@playwright/test"],
    SupportedFramework.PUPPETEER: ["puppeteer", "jest"],
    SupportedFramework.CYPRESS: ["cypress"],
    SupportedFramework.SELENIUM_JAVA: ["org.seleniumhq.selenium:selenium-java"],
    SupportedFramework.SELENIUM_CSHARP: ["Selenium.WebDriver", "NUnit"],
    SupportedFramework.PLAYWRIGHT_CSHARP: ["Microsoft.Playwright", "NUnit"],
    SupportedFramework.CAPYBARA: ["capybara", "selenium-webdriver"],
    SupportedFramework.SELENIUM_RUBY: ["selenium-webdriver"],
    SupportedFramework.ROD: ["github.com/go-rod/rod"],
}
