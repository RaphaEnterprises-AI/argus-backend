"""Tests for ExportEngine class."""

import pytest
from src.export.engine import ExportEngine, export_test
from src.export.models import (
    ExportConfig,
    ExportResult,
    SupportedLanguage,
    SupportedFramework,
    FRAMEWORK_SUPPORT,
    FILE_EXTENSIONS,
    FRAMEWORK_DEPENDENCIES,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def engine():
    """Create ExportEngine instance."""
    return ExportEngine()


@pytest.fixture
def simple_test_spec():
    """Simple test spec for basic tests."""
    return {
        "id": "test-001",
        "name": "Simple Test",
        "description": "A simple test",
        "steps": [
            {"action": "goto", "target": "/login"},
            {"action": "fill", "target": "#email", "value": "test@example.com"},
            {"action": "click", "target": "button[type=submit]"},
        ],
        "assertions": [
            {"type": "url_contains", "expected": "/dashboard"},
        ],
    }


@pytest.fixture
def complex_test_spec():
    """Complex test spec with many steps and assertions."""
    return {
        "id": "test-complex-002",
        "name": "Complex E2E Test Flow",
        "description": "Tests complete user registration and checkout flow",
        "steps": [
            {"action": "goto", "target": "https://example.com/register"},
            {"action": "fill", "target": "#first-name", "value": "John"},
            {"action": "fill", "target": "#last-name", "value": "Doe"},
            {"action": "fill", "target": "#email", "value": "john@example.com"},
            {"action": "fill", "target": "#password", "value": "SecurePass123"},
            {"action": "select", "target": "#country", "value": "US"},
            {"action": "click", "target": "#terms-checkbox"},
            {"action": "click", "target": "button#register"},
            {"action": "wait", "target": ".welcome-message", "timeout": 5000},
            {"action": "hover", "target": ".user-menu"},
            {"action": "click", "target": ".profile-link"},
            {"action": "scroll", "value": "0,500"},
            {"action": "double_click", "target": ".edit-button"},
            {"action": "press_key", "value": "Escape"},
            {"action": "screenshot"},
        ],
        "assertions": [
            {"type": "element_visible", "target": ".welcome-message"},
            {"type": "text_contains", "target": ".user-name", "expected": "John"},
            {"type": "text_equals", "target": ".email-display", "expected": "john@example.com"},
            {"type": "url_contains", "expected": "/profile"},
            {"type": "value_equals", "target": "#email-input", "expected": "john@example.com"},
            {"type": "title_contains", "expected": "Profile"},
            {"type": "element_hidden", "target": ".loading-spinner"},
        ],
    }


@pytest.fixture
def minimal_test_spec():
    """Minimal test spec with just basics."""
    return {
        "name": "Minimal Test",
        "steps": [{"action": "goto", "target": "/home"}],
    }


# =============================================================================
# ExportEngine Initialization Tests
# =============================================================================


class TestExportEngineInit:
    """Tests for ExportEngine initialization."""

    def test_create_engine(self):
        """Test basic engine creation."""
        engine = ExportEngine()
        assert engine is not None
        assert hasattr(engine, "log")

    def test_engine_has_export_method(self, engine):
        """Test engine has export method."""
        assert hasattr(engine, "export")
        assert callable(engine.export)

    def test_engine_has_batch_export(self, engine):
        """Test engine has batch export method."""
        assert hasattr(engine, "export_batch")
        assert callable(engine.export_batch)


# =============================================================================
# Basic Export Tests
# =============================================================================


class TestBasicExport:
    """Tests for basic export functionality."""

    def test_export_default_config(self, engine, simple_test_spec):
        """Test export with default configuration."""
        result = engine.export(simple_test_spec)
        assert result.success
        assert result.code
        assert result.language == SupportedLanguage.PYTHON
        assert result.framework == SupportedFramework.PLAYWRIGHT_PYTHON
        assert result.file_extension == ".py"

    def test_export_returns_export_result(self, engine, simple_test_spec):
        """Test export returns ExportResult dataclass."""
        result = engine.export(simple_test_spec)
        assert isinstance(result, ExportResult)

    def test_export_with_custom_config(self, engine, simple_test_spec):
        """Test export with custom configuration."""
        config = ExportConfig(
            language=SupportedLanguage.TYPESCRIPT,
            framework=SupportedFramework.PLAYWRIGHT_TS,
        )
        result = engine.export(simple_test_spec, config)
        assert result.success
        assert result.language == SupportedLanguage.TYPESCRIPT
        assert result.framework == SupportedFramework.PLAYWRIGHT_TS

    def test_export_with_string_language(self, engine, simple_test_spec):
        """Test export with string language value."""
        config = ExportConfig(
            language="python",
            framework="playwright",
        )
        result = engine.export(simple_test_spec, config)
        assert result.success
        assert result.language == SupportedLanguage.PYTHON

    def test_export_includes_dependencies(self, engine, simple_test_spec):
        """Test export includes dependency information."""
        result = engine.export(simple_test_spec)
        assert result.success
        assert result.dependencies
        assert "playwright" in result.dependencies

    def test_export_includes_metadata(self, engine, simple_test_spec):
        """Test export includes metadata."""
        result = engine.export(simple_test_spec)
        assert result.success
        assert result.metadata
        assert result.metadata.get("test_id") == "test-001"
        assert result.metadata.get("test_name") == "Simple Test"
        assert result.metadata.get("steps_count") == 3
        assert result.metadata.get("assertions_count") == 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestExportErrorHandling:
    """Tests for export error handling."""

    def test_invalid_language_framework_combination(self, engine, simple_test_spec):
        """Test export fails for invalid language-framework combination."""
        config = ExportConfig(
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.CYPRESS,  # Invalid for Python
        )
        result = engine.export(simple_test_spec, config)
        assert not result.success
        assert result.error
        assert "not supported" in result.error.lower()

    def test_export_empty_steps(self, engine):
        """Test export handles empty steps."""
        test_spec = {
            "name": "Empty Test",
            "steps": [],
            "assertions": [],
        }
        result = engine.export(test_spec)
        assert result.success
        assert result.code

    def test_export_handles_missing_fields(self, engine, minimal_test_spec):
        """Test export handles missing optional fields."""
        result = engine.export(minimal_test_spec)
        assert result.success
        assert result.code

    def test_export_unknown_action(self, engine):
        """Test export handles unknown actions gracefully."""
        test_spec = {
            "name": "Unknown Action Test",
            "steps": [{"action": "unknown_action", "target": "#element"}],
        }
        result = engine.export(test_spec)
        assert result.success
        assert "unknown" in result.code.lower()

    def test_export_unknown_assertion(self, engine):
        """Test export handles unknown assertions gracefully."""
        test_spec = {
            "name": "Unknown Assertion Test",
            "steps": [{"action": "goto", "target": "/home"}],
            "assertions": [{"type": "unknown_type", "target": "#el"}],
        }
        result = engine.export(test_spec)
        assert result.success
        assert "unknown" in result.code.lower()


# =============================================================================
# Supported Combinations Tests
# =============================================================================


class TestSupportedCombinations:
    """Tests for supported language-framework combinations."""

    def test_get_supported_combinations(self, engine):
        """Test getting supported combinations."""
        combinations = engine.get_supported_combinations()
        assert isinstance(combinations, dict)
        assert "python" in combinations
        assert "typescript" in combinations
        assert "java" in combinations

    def test_python_frameworks(self, engine):
        """Test Python has expected frameworks."""
        combinations = engine.get_supported_combinations()
        assert "playwright" in combinations["python"]
        assert "selenium" in combinations["python"]

    def test_typescript_frameworks(self, engine):
        """Test TypeScript has expected frameworks."""
        combinations = engine.get_supported_combinations()
        assert "playwright-ts" in combinations["typescript"]
        assert "cypress" in combinations["typescript"]

    def test_all_languages_present(self, engine):
        """Test all supported languages are present."""
        combinations = engine.get_supported_combinations()
        expected_languages = ["python", "typescript", "java", "csharp", "ruby", "go"]
        for lang in expected_languages:
            assert lang in combinations


# =============================================================================
# Batch Export Tests
# =============================================================================


class TestBatchExport:
    """Tests for batch export functionality."""

    def test_batch_export_multiple_specs(self, engine, simple_test_spec, complex_test_spec):
        """Test batch export with multiple specs."""
        results = engine.export_batch([simple_test_spec, complex_test_spec])
        assert len(results) == 2
        assert all(isinstance(r, ExportResult) for r in results)
        assert all(r.success for r in results)

    def test_batch_export_empty_list(self, engine):
        """Test batch export with empty list."""
        results = engine.export_batch([])
        assert results == []

    def test_batch_export_with_config(self, engine, simple_test_spec, minimal_test_spec):
        """Test batch export applies same config to all."""
        config = ExportConfig(
            language=SupportedLanguage.JAVA,
            framework=SupportedFramework.SELENIUM_JAVA,
        )
        results = engine.export_batch([simple_test_spec, minimal_test_spec], config)
        assert len(results) == 2
        assert all(r.language == SupportedLanguage.JAVA for r in results)

    def test_batch_export_partial_failure(self, engine, simple_test_spec):
        """Test batch export handles partial failures."""
        bad_spec = {
            "name": "Bad Test",
            "steps": [{"action": "goto", "target": "/home"}],
        }
        # Both should succeed
        results = engine.export_batch([simple_test_spec, bad_spec])
        assert len(results) == 2
        assert results[0].success


# =============================================================================
# Preview Tests
# =============================================================================


class TestPreview:
    """Tests for preview functionality."""

    def test_preview_basic(self, engine, simple_test_spec):
        """Test basic preview."""
        preview = engine.preview(simple_test_spec)
        assert preview
        assert isinstance(preview, str)

    def test_preview_max_lines(self, engine, complex_test_spec):
        """Test preview respects max_lines."""
        preview = engine.preview(complex_test_spec, max_lines=5)
        lines = preview.split("\n")
        # Should have truncation indicator
        assert len(lines) <= 7  # 5 lines + "..." + count

    def test_preview_error_handling(self, engine, simple_test_spec):
        """Test preview handles errors."""
        config = ExportConfig(
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.CYPRESS,  # Invalid
        )
        preview = engine.preview(simple_test_spec, config)
        assert "error" in preview.lower()


# =============================================================================
# File Info Tests
# =============================================================================


class TestFileInfo:
    """Tests for file info functionality."""

    def test_get_file_info_python(self, engine):
        """Test file info for Python Playwright."""
        config = ExportConfig(
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.PLAYWRIGHT_PYTHON,
        )
        info = engine.get_file_info(config)
        assert info["extension"] == ".py"
        assert info["dependencies"]
        assert "pip install" in info["install_command"]

    def test_get_file_info_typescript(self, engine):
        """Test file info for TypeScript."""
        config = ExportConfig(
            language=SupportedLanguage.TYPESCRIPT,
            framework=SupportedFramework.PLAYWRIGHT_TS,
        )
        info = engine.get_file_info(config)
        assert info["extension"] == ".ts"
        assert "npm install" in info["install_command"]

    def test_get_file_info_java(self, engine):
        """Test file info for Java."""
        config = ExportConfig(
            language=SupportedLanguage.JAVA,
            framework=SupportedFramework.SELENIUM_JAVA,
        )
        info = engine.get_file_info(config)
        assert info["extension"] == ".java"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestExportTestFunction:
    """Tests for the convenience export_test function."""

    def test_export_test_default(self, simple_test_spec):
        """Test export_test with defaults."""
        result = export_test(simple_test_spec)
        assert result.success
        assert result.language == SupportedLanguage.PYTHON
        assert result.framework == SupportedFramework.PLAYWRIGHT_PYTHON

    def test_export_test_custom_language(self, simple_test_spec):
        """Test export_test with custom language."""
        result = export_test(
            simple_test_spec,
            language="typescript",
            framework="playwright-ts",
        )
        assert result.success
        assert result.language == SupportedLanguage.TYPESCRIPT

    def test_export_test_with_options(self, simple_test_spec):
        """Test export_test with additional options."""
        result = export_test(
            simple_test_spec,
            language="python",
            framework="playwright",
            include_comments=False,
            async_style=False,
        )
        assert result.success


# =============================================================================
# Code Generation Tests
# =============================================================================


class TestCodeGeneration:
    """Tests for code generation quality."""

    def test_generated_code_has_imports(self, engine, simple_test_spec):
        """Test generated code includes imports."""
        result = engine.export(simple_test_spec)
        assert "import" in result.code.lower()

    def test_generated_code_has_class_or_function(self, engine, simple_test_spec):
        """Test generated code has class or function."""
        result = engine.export(simple_test_spec)
        assert "class" in result.code.lower() or "def" in result.code.lower() or "test" in result.code.lower()

    def test_generated_code_has_attribution(self, engine, simple_test_spec):
        """Test generated code has Argus attribution."""
        result = engine.export(simple_test_spec)
        assert "argus" in result.code.lower()

    def test_generated_code_has_test_id(self, engine, simple_test_spec):
        """Test generated code references test ID."""
        result = engine.export(simple_test_spec)
        assert "test-001" in result.code

    def test_generated_code_ends_with_newline(self, engine, simple_test_spec):
        """Test generated code ends with newline."""
        result = engine.export(simple_test_spec)
        assert result.code.endswith("\n")


# =============================================================================
# All Language Exports Test
# =============================================================================


class TestAllLanguageExports:
    """Tests to verify all language-framework combinations work."""

    @pytest.mark.parametrize("language,framework", [
        (SupportedLanguage.PYTHON, SupportedFramework.PLAYWRIGHT_PYTHON),
        (SupportedLanguage.PYTHON, SupportedFramework.SELENIUM_PYTHON),
        (SupportedLanguage.TYPESCRIPT, SupportedFramework.PLAYWRIGHT_TS),
        (SupportedLanguage.JAVA, SupportedFramework.SELENIUM_JAVA),
        (SupportedLanguage.CSHARP, SupportedFramework.SELENIUM_CSHARP),
        (SupportedLanguage.RUBY, SupportedFramework.CAPYBARA),
        (SupportedLanguage.GO, SupportedFramework.ROD),
    ])
    def test_export_all_languages(self, engine, simple_test_spec, language, framework):
        """Test export works for all supported language-framework combinations."""
        config = ExportConfig(language=language, framework=framework)
        result = engine.export(simple_test_spec, config)
        assert result.success, f"Export failed for {language.value}/{framework.value}: {result.error}"
        assert result.code, f"No code generated for {language.value}/{framework.value}"
        assert result.file_extension == FILE_EXTENSIONS[language]

    @pytest.mark.parametrize("language,framework", [
        (SupportedLanguage.PYTHON, SupportedFramework.PLAYWRIGHT_PYTHON),
        (SupportedLanguage.TYPESCRIPT, SupportedFramework.PLAYWRIGHT_TS),
        (SupportedLanguage.JAVA, SupportedFramework.SELENIUM_JAVA),
        (SupportedLanguage.GO, SupportedFramework.ROD),
    ])
    def test_complex_export_all_languages(self, engine, complex_test_spec, language, framework):
        """Test complex export works for main language-framework combinations."""
        config = ExportConfig(language=language, framework=framework)
        result = engine.export(complex_test_spec, config)
        assert result.success, f"Complex export failed for {language.value}/{framework.value}: {result.error}"
        # Verify all steps are represented
        assert result.metadata.get("steps_count") == 15
        assert result.metadata.get("assertions_count") == 7


# =============================================================================
# ExportResult Tests
# =============================================================================


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_export_result_to_dict(self, engine, simple_test_spec):
        """Test ExportResult.to_dict() method."""
        result = engine.export(simple_test_spec)
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["code"]
        assert result_dict["language"] == "python"
        assert result_dict["framework"] == "playwright"

    def test_export_result_error_to_dict(self, engine, simple_test_spec):
        """Test ExportResult.to_dict() for error case."""
        config = ExportConfig(
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.CYPRESS,
        )
        result = engine.export(simple_test_spec, config)
        result_dict = result.to_dict()
        assert result_dict["success"] is False
        assert result_dict["error"] is not None


# =============================================================================
# ExportConfig Tests
# =============================================================================


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_config_default_values(self):
        """Test ExportConfig default values."""
        config = ExportConfig()
        assert config.language == SupportedLanguage.PYTHON
        assert config.framework == SupportedFramework.PLAYWRIGHT_PYTHON
        assert config.include_comments is True
        assert config.include_assertions is True
        assert config.async_style is True

    def test_config_string_to_enum_conversion(self):
        """Test ExportConfig converts strings to enums."""
        config = ExportConfig(language="java", framework="selenium-java")
        assert config.language == SupportedLanguage.JAVA
        assert config.framework == SupportedFramework.SELENIUM_JAVA

    def test_config_validation_valid(self):
        """Test ExportConfig validation passes for valid config."""
        config = ExportConfig(
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.PLAYWRIGHT_PYTHON,
        )
        errors = config.validate()
        assert errors == []

    def test_config_validation_invalid(self):
        """Test ExportConfig validation fails for invalid combination."""
        config = ExportConfig(
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.ROD,  # Go framework, not Python
        )
        errors = config.validate()
        assert len(errors) > 0
        assert "not supported" in errors[0].lower()

    def test_config_custom_fields(self):
        """Test ExportConfig with custom fields."""
        config = ExportConfig(
            language=SupportedLanguage.PYTHON,
            framework=SupportedFramework.PLAYWRIGHT_PYTHON,
            test_class_name="MyCustomTest",
            base_url_variable="app_url",
            custom_imports=["import custom_lib"],
        )
        assert config.test_class_name == "MyCustomTest"
        assert config.base_url_variable == "app_url"
        assert "import custom_lib" in config.custom_imports
