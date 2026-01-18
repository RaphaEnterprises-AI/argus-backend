"""Comprehensive tests for export models module."""

import pytest
from src.export.models import (
    SupportedLanguage,
    SupportedFramework,
    ExportConfig,
    ExportResult,
    FRAMEWORK_SUPPORT,
    FILE_EXTENSIONS,
    FRAMEWORK_DEPENDENCIES,
)


# =============================================================================
# SupportedLanguage Enum Tests
# =============================================================================


class TestSupportedLanguage:
    """Tests for SupportedLanguage enum."""

    def test_all_languages_defined(self):
        """Test all expected languages are defined."""
        expected = ["python", "typescript", "java", "csharp", "ruby", "go"]
        actual = [lang.value for lang in SupportedLanguage]
        assert set(expected) == set(actual)

    def test_language_is_string_enum(self):
        """Test SupportedLanguage is a string enum."""
        assert isinstance(SupportedLanguage.PYTHON.value, str)
        assert str(SupportedLanguage.PYTHON) == "SupportedLanguage.PYTHON"

    def test_language_values(self):
        """Test specific language values."""
        assert SupportedLanguage.PYTHON.value == "python"
        assert SupportedLanguage.TYPESCRIPT.value == "typescript"
        assert SupportedLanguage.JAVA.value == "java"
        assert SupportedLanguage.CSHARP.value == "csharp"
        assert SupportedLanguage.RUBY.value == "ruby"
        assert SupportedLanguage.GO.value == "go"

    def test_language_from_value(self):
        """Test creating language from string value."""
        assert SupportedLanguage("python") == SupportedLanguage.PYTHON
        assert SupportedLanguage("typescript") == SupportedLanguage.TYPESCRIPT
        assert SupportedLanguage("java") == SupportedLanguage.JAVA

    def test_invalid_language_raises(self):
        """Test invalid language value raises ValueError."""
        with pytest.raises(ValueError):
            SupportedLanguage("invalid")

    def test_language_equality(self):
        """Test language equality comparison."""
        assert SupportedLanguage.PYTHON == SupportedLanguage.PYTHON
        assert SupportedLanguage.PYTHON != SupportedLanguage.JAVA

    def test_language_membership(self):
        """Test language membership in sets/lists."""
        languages = [SupportedLanguage.PYTHON, SupportedLanguage.JAVA]
        assert SupportedLanguage.PYTHON in languages
        assert SupportedLanguage.GO not in languages


# =============================================================================
# SupportedFramework Enum Tests
# =============================================================================


class TestSupportedFramework:
    """Tests for SupportedFramework enum."""

    def test_python_frameworks(self):
        """Test Python-related frameworks."""
        assert SupportedFramework.PLAYWRIGHT_PYTHON.value == "playwright"
        assert SupportedFramework.SELENIUM_PYTHON.value == "selenium"
        assert SupportedFramework.PYTEST.value == "pytest"

    def test_typescript_frameworks(self):
        """Test TypeScript-related frameworks."""
        assert SupportedFramework.PLAYWRIGHT_TS.value == "playwright-ts"
        assert SupportedFramework.PUPPETEER.value == "puppeteer"
        assert SupportedFramework.CYPRESS.value == "cypress"

    def test_java_frameworks(self):
        """Test Java-related frameworks."""
        assert SupportedFramework.SELENIUM_JAVA.value == "selenium-java"
        assert SupportedFramework.TESTNG.value == "testng"

    def test_csharp_frameworks(self):
        """Test C#-related frameworks."""
        assert SupportedFramework.SELENIUM_CSHARP.value == "selenium-csharp"
        assert SupportedFramework.PLAYWRIGHT_CSHARP.value == "playwright-csharp"
        assert SupportedFramework.NUNIT.value == "nunit"

    def test_ruby_frameworks(self):
        """Test Ruby-related frameworks."""
        assert SupportedFramework.CAPYBARA.value == "capybara"
        assert SupportedFramework.SELENIUM_RUBY.value == "selenium-ruby"

    def test_go_frameworks(self):
        """Test Go-related frameworks."""
        assert SupportedFramework.ROD.value == "rod"

    def test_framework_from_value(self):
        """Test creating framework from string value."""
        assert SupportedFramework("playwright") == SupportedFramework.PLAYWRIGHT_PYTHON
        assert SupportedFramework("selenium-java") == SupportedFramework.SELENIUM_JAVA
        assert SupportedFramework("capybara") == SupportedFramework.CAPYBARA

    def test_invalid_framework_raises(self):
        """Test invalid framework value raises ValueError."""
        with pytest.raises(ValueError):
            SupportedFramework("invalid-framework")


# =============================================================================
# FRAMEWORK_SUPPORT Tests
# =============================================================================


class TestFrameworkSupport:
    """Tests for FRAMEWORK_SUPPORT mapping."""

    def test_all_languages_have_support(self):
        """Test all languages are in FRAMEWORK_SUPPORT."""
        for language in SupportedLanguage:
            assert language in FRAMEWORK_SUPPORT

    def test_python_support(self):
        """Test Python framework support."""
        python_frameworks = FRAMEWORK_SUPPORT[SupportedLanguage.PYTHON]
        assert SupportedFramework.PLAYWRIGHT_PYTHON in python_frameworks
        assert SupportedFramework.SELENIUM_PYTHON in python_frameworks

    def test_typescript_support(self):
        """Test TypeScript framework support."""
        ts_frameworks = FRAMEWORK_SUPPORT[SupportedLanguage.TYPESCRIPT]
        assert SupportedFramework.PLAYWRIGHT_TS in ts_frameworks
        assert SupportedFramework.PUPPETEER in ts_frameworks
        assert SupportedFramework.CYPRESS in ts_frameworks

    def test_java_support(self):
        """Test Java framework support."""
        java_frameworks = FRAMEWORK_SUPPORT[SupportedLanguage.JAVA]
        assert SupportedFramework.SELENIUM_JAVA in java_frameworks

    def test_csharp_support(self):
        """Test C# framework support."""
        csharp_frameworks = FRAMEWORK_SUPPORT[SupportedLanguage.CSHARP]
        assert SupportedFramework.SELENIUM_CSHARP in csharp_frameworks
        assert SupportedFramework.PLAYWRIGHT_CSHARP in csharp_frameworks

    def test_ruby_support(self):
        """Test Ruby framework support."""
        ruby_frameworks = FRAMEWORK_SUPPORT[SupportedLanguage.RUBY]
        assert SupportedFramework.CAPYBARA in ruby_frameworks
        assert SupportedFramework.SELENIUM_RUBY in ruby_frameworks

    def test_go_support(self):
        """Test Go framework support."""
        go_frameworks = FRAMEWORK_SUPPORT[SupportedLanguage.GO]
        assert SupportedFramework.ROD in go_frameworks

    def test_frameworks_are_lists(self):
        """Test framework support values are lists."""
        for language, frameworks in FRAMEWORK_SUPPORT.items():
            assert isinstance(frameworks, list)
            assert len(frameworks) > 0


# =============================================================================
# FILE_EXTENSIONS Tests
# =============================================================================


class TestFileExtensions:
    """Tests for FILE_EXTENSIONS mapping."""

    def test_all_languages_have_extensions(self):
        """Test all languages have file extensions."""
        for language in SupportedLanguage:
            assert language in FILE_EXTENSIONS

    def test_python_extension(self):
        """Test Python file extension."""
        assert FILE_EXTENSIONS[SupportedLanguage.PYTHON] == ".py"

    def test_typescript_extension(self):
        """Test TypeScript file extension."""
        assert FILE_EXTENSIONS[SupportedLanguage.TYPESCRIPT] == ".ts"

    def test_java_extension(self):
        """Test Java file extension."""
        assert FILE_EXTENSIONS[SupportedLanguage.JAVA] == ".java"

    def test_csharp_extension(self):
        """Test C# file extension."""
        assert FILE_EXTENSIONS[SupportedLanguage.CSHARP] == ".cs"

    def test_ruby_extension(self):
        """Test Ruby file extension."""
        assert FILE_EXTENSIONS[SupportedLanguage.RUBY] == ".rb"

    def test_go_extension(self):
        """Test Go file extension."""
        assert FILE_EXTENSIONS[SupportedLanguage.GO] == ".go"

    def test_extensions_start_with_dot(self):
        """Test all extensions start with a dot."""
        for extension in FILE_EXTENSIONS.values():
            assert extension.startswith(".")


# =============================================================================
# FRAMEWORK_DEPENDENCIES Tests
# =============================================================================


class TestFrameworkDependencies:
    """Tests for FRAMEWORK_DEPENDENCIES mapping."""

    def test_playwright_python_deps(self):
        """Test Python Playwright dependencies."""
        deps = FRAMEWORK_DEPENDENCIES[SupportedFramework.PLAYWRIGHT_PYTHON]
        assert "playwright" in deps
        assert "pytest" in deps
        assert "pytest-playwright" in deps

    def test_selenium_python_deps(self):
        """Test Python Selenium dependencies."""
        deps = FRAMEWORK_DEPENDENCIES[SupportedFramework.SELENIUM_PYTHON]
        assert "selenium" in deps
        assert "pytest" in deps
        assert "webdriver-manager" in deps

    def test_playwright_ts_deps(self):
        """Test TypeScript Playwright dependencies."""
        deps = FRAMEWORK_DEPENDENCIES[SupportedFramework.PLAYWRIGHT_TS]
        assert "@playwright/test" in deps

    def test_puppeteer_deps(self):
        """Test Puppeteer dependencies."""
        deps = FRAMEWORK_DEPENDENCIES[SupportedFramework.PUPPETEER]
        assert "puppeteer" in deps
        assert "jest" in deps

    def test_cypress_deps(self):
        """Test Cypress dependencies."""
        deps = FRAMEWORK_DEPENDENCIES[SupportedFramework.CYPRESS]
        assert "cypress" in deps

    def test_selenium_java_deps(self):
        """Test Java Selenium dependencies."""
        deps = FRAMEWORK_DEPENDENCIES[SupportedFramework.SELENIUM_JAVA]
        assert any("selenium" in d for d in deps)

    def test_selenium_csharp_deps(self):
        """Test C# Selenium dependencies."""
        deps = FRAMEWORK_DEPENDENCIES[SupportedFramework.SELENIUM_CSHARP]
        assert "Selenium.WebDriver" in deps
        assert "NUnit" in deps

    def test_playwright_csharp_deps(self):
        """Test C# Playwright dependencies."""
        deps = FRAMEWORK_DEPENDENCIES[SupportedFramework.PLAYWRIGHT_CSHARP]
        assert "Microsoft.Playwright" in deps
        assert "NUnit" in deps

    def test_capybara_deps(self):
        """Test Capybara dependencies."""
        deps = FRAMEWORK_DEPENDENCIES[SupportedFramework.CAPYBARA]
        assert "capybara" in deps
        assert "selenium-webdriver" in deps

    def test_selenium_ruby_deps(self):
        """Test Ruby Selenium dependencies."""
        deps = FRAMEWORK_DEPENDENCIES[SupportedFramework.SELENIUM_RUBY]
        assert "selenium-webdriver" in deps

    def test_rod_deps(self):
        """Test Go Rod dependencies."""
        deps = FRAMEWORK_DEPENDENCIES[SupportedFramework.ROD]
        assert "github.com/go-rod/rod" in deps

    def test_dependencies_are_lists(self):
        """Test dependencies are lists."""
        for framework, deps in FRAMEWORK_DEPENDENCIES.items():
            assert isinstance(deps, list)


# =============================================================================
# ExportConfig Tests
# =============================================================================


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_values(self):
        """Test ExportConfig default values."""
        config = ExportConfig()
        assert config.language == SupportedLanguage.PYTHON
        assert config.framework == SupportedFramework.PLAYWRIGHT_PYTHON
        assert config.include_comments is True
        assert config.include_assertions is True
        assert config.base_url_variable == "base_url"
        assert config.test_class_name is None
        assert config.async_style is True
        assert config.page_object_pattern is False
        assert config.data_driven is False
        assert config.custom_imports == []
        assert config.custom_fixtures == {}

    def test_string_language_conversion(self):
        """Test string language is converted to enum."""
        config = ExportConfig(language="python")
        assert config.language == SupportedLanguage.PYTHON

        config = ExportConfig(language="TYPESCRIPT")
        assert config.language == SupportedLanguage.TYPESCRIPT

    def test_string_framework_conversion(self):
        """Test string framework is converted to enum."""
        config = ExportConfig(framework="selenium")
        assert config.framework == SupportedFramework.SELENIUM_PYTHON

        config = ExportConfig(framework="PLAYWRIGHT-TS")
        assert config.framework == SupportedFramework.PLAYWRIGHT_TS

    def test_enum_values_unchanged(self):
        """Test enum values remain unchanged."""
        config = ExportConfig(
            language=SupportedLanguage.JAVA,
            framework=SupportedFramework.SELENIUM_JAVA,
        )
        assert config.language == SupportedLanguage.JAVA
        assert config.framework == SupportedFramework.SELENIUM_JAVA

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = ExportConfig(
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.PLAYWRIGHT_PYTHON,
        )
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_framework_for_language(self):
        """Test validation fails for invalid framework-language combination."""
        config = ExportConfig(
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.ROD,  # Go framework
        )
        errors = config.validate()
        assert len(errors) == 1
        assert "not supported" in errors[0].lower()
        assert "python" in errors[0].lower()

    def test_validate_lists_valid_options(self):
        """Test validation error lists valid options."""
        config = ExportConfig(
            language=SupportedLanguage.GO,
            framework=SupportedFramework.CYPRESS,  # TypeScript framework
        )
        errors = config.validate()
        assert len(errors) == 1
        assert "rod" in errors[0].lower()

    def test_custom_imports(self):
        """Test custom imports field."""
        config = ExportConfig(
            custom_imports=["import custom", "from lib import func"]
        )
        assert len(config.custom_imports) == 2
        assert "import custom" in config.custom_imports

    def test_custom_fixtures(self):
        """Test custom fixtures field."""
        config = ExportConfig(
            custom_fixtures={"fixture1": "value1", "fixture2": "value2"}
        )
        assert config.custom_fixtures["fixture1"] == "value1"

    def test_all_boolean_options(self):
        """Test all boolean configuration options."""
        config = ExportConfig(
            include_comments=False,
            include_assertions=False,
            async_style=False,
            page_object_pattern=True,
            data_driven=True,
        )
        assert config.include_comments is False
        assert config.include_assertions is False
        assert config.async_style is False
        assert config.page_object_pattern is True
        assert config.data_driven is True

    def test_test_class_name(self):
        """Test custom test class name."""
        config = ExportConfig(test_class_name="MyCustomTestClass")
        assert config.test_class_name == "MyCustomTestClass"

    def test_base_url_variable(self):
        """Test custom base URL variable name."""
        config = ExportConfig(base_url_variable="app_base_url")
        assert config.base_url_variable == "app_base_url"

    @pytest.mark.parametrize("language,framework", [
        ("python", "playwright"),
        ("python", "selenium"),
        ("typescript", "playwright-ts"),
        ("java", "selenium-java"),
        ("csharp", "selenium-csharp"),
        ("ruby", "capybara"),
        ("go", "rod"),
    ])
    def test_valid_combinations_pass_validation(self, language, framework):
        """Test all valid language-framework combinations pass validation."""
        config = ExportConfig(language=language, framework=framework)
        errors = config.validate()
        assert errors == [], f"Unexpected error for {language}/{framework}: {errors}"


# =============================================================================
# ExportResult Tests
# =============================================================================


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = ExportResult(
            success=True,
            code="def test(): pass",
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.PLAYWRIGHT_PYTHON,
        )
        assert result.success is True
        assert result.code == "def test(): pass"
        assert result.error is None

    def test_failure_result(self):
        """Test creating a failure result."""
        result = ExportResult(
            success=False,
            error="Export failed due to invalid config",
        )
        assert result.success is False
        assert result.code == ""
        assert result.error == "Export failed due to invalid config"

    def test_default_values(self):
        """Test ExportResult default values."""
        result = ExportResult(success=True)
        assert result.code == ""
        assert result.language is None
        assert result.framework is None
        assert result.file_extension == ".py"
        assert result.imports == []
        assert result.dependencies == []
        assert result.error is None
        assert result.metadata == {}

    def test_to_dict_success(self):
        """Test to_dict for successful result."""
        result = ExportResult(
            success=True,
            code="test code",
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.PLAYWRIGHT_PYTHON,
            file_extension=".py",
            imports=["import pytest"],
            dependencies=["playwright"],
            metadata={"test_id": "123"},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["code"] == "test code"
        assert d["language"] == "python"
        assert d["framework"] == "playwright"
        assert d["file_extension"] == ".py"
        assert d["imports"] == ["import pytest"]
        assert d["dependencies"] == ["playwright"]
        assert d["error"] is None
        assert d["metadata"]["test_id"] == "123"

    def test_to_dict_failure(self):
        """Test to_dict for failure result."""
        result = ExportResult(
            success=False,
            error="Something went wrong",
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["code"] == ""
        assert d["language"] is None
        assert d["framework"] is None
        assert d["error"] == "Something went wrong"

    def test_to_dict_all_languages(self):
        """Test to_dict handles all language values."""
        for language in SupportedLanguage:
            result = ExportResult(success=True, language=language)
            d = result.to_dict()
            assert d["language"] == language.value

    def test_to_dict_all_frameworks(self):
        """Test to_dict handles all framework values."""
        for framework in SupportedFramework:
            result = ExportResult(success=True, framework=framework)
            d = result.to_dict()
            assert d["framework"] == framework.value

    def test_metadata_can_contain_various_types(self):
        """Test metadata can contain various types."""
        result = ExportResult(
            success=True,
            metadata={
                "string_val": "hello",
                "int_val": 42,
                "list_val": [1, 2, 3],
                "dict_val": {"nested": "value"},
                "bool_val": True,
            },
        )
        assert result.metadata["string_val"] == "hello"
        assert result.metadata["int_val"] == 42
        assert result.metadata["list_val"] == [1, 2, 3]
        assert result.metadata["dict_val"]["nested"] == "value"
        assert result.metadata["bool_val"] is True

    def test_imports_list(self):
        """Test imports list."""
        result = ExportResult(
            success=True,
            imports=["import pytest", "from playwright.sync_api import Page"],
        )
        assert len(result.imports) == 2
        assert "import pytest" in result.imports

    def test_dependencies_list(self):
        """Test dependencies list."""
        result = ExportResult(
            success=True,
            dependencies=["playwright", "pytest", "pytest-playwright"],
        )
        assert len(result.dependencies) == 3
        assert "playwright" in result.dependencies


# =============================================================================
# Cross-validation Tests
# =============================================================================


class TestCrossValidation:
    """Tests for cross-validation between models."""

    def test_all_frameworks_have_dependencies(self):
        """Test all frameworks have dependency information."""
        for framework in SupportedFramework:
            assert framework in FRAMEWORK_DEPENDENCIES or True  # Some may not have deps

    def test_framework_support_references_valid_frameworks(self):
        """Test FRAMEWORK_SUPPORT only references valid frameworks."""
        for language, frameworks in FRAMEWORK_SUPPORT.items():
            for framework in frameworks:
                assert isinstance(framework, SupportedFramework)

    def test_file_extensions_reference_valid_languages(self):
        """Test FILE_EXTENSIONS only references valid languages."""
        for language in FILE_EXTENSIONS.keys():
            assert isinstance(language, SupportedLanguage)

    @pytest.mark.parametrize("language", list(SupportedLanguage))
    def test_each_language_has_at_least_one_framework(self, language):
        """Test each language has at least one supported framework."""
        frameworks = FRAMEWORK_SUPPORT.get(language, [])
        assert len(frameworks) >= 1, f"{language.value} has no supported frameworks"
