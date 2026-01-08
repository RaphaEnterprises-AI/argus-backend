"""Export Engine - Main class for test export."""

from typing import Any, Optional

import structlog

from .models import (
    ExportConfig,
    ExportResult,
    SupportedLanguage,
    SupportedFramework,
    FILE_EXTENSIONS,
    FRAMEWORK_DEPENDENCIES,
)
from .templates import (
    BaseTemplate,
    PythonPlaywrightTemplate,
    PythonSeleniumTemplate,
    TypeScriptPlaywrightTemplate,
    JavaSeleniumTemplate,
    CSharpSeleniumTemplate,
    RubyCapybaraTemplate,
    GoRodTemplate,
)
from .formatters import CodeFormatter

logger = structlog.get_logger()


# Template registry mapping framework to template class
TEMPLATE_REGISTRY: dict[SupportedFramework, type[BaseTemplate]] = {
    SupportedFramework.PLAYWRIGHT_PYTHON: PythonPlaywrightTemplate,
    SupportedFramework.SELENIUM_PYTHON: PythonSeleniumTemplate,
    SupportedFramework.PLAYWRIGHT_TS: TypeScriptPlaywrightTemplate,
    SupportedFramework.SELENIUM_JAVA: JavaSeleniumTemplate,
    SupportedFramework.SELENIUM_CSHARP: CSharpSeleniumTemplate,
    SupportedFramework.CAPYBARA: RubyCapybaraTemplate,
    SupportedFramework.ROD: GoRodTemplate,
}


class ExportEngine:
    """Main engine for exporting tests to various languages.

    This class orchestrates the export process:
    1. Validates the export configuration
    2. Selects the appropriate template
    3. Generates code using the template
    4. Formats the output

    Example:
        engine = ExportEngine()

        result = engine.export(
            test_spec={"name": "Login Test", "steps": [...]},
            config=ExportConfig(
                language="python",
                framework="playwright",
            )
        )

        if result.success:
            print(result.code)
            print(f"Install: pip install {' '.join(result.dependencies)}")
    """

    def __init__(self):
        """Initialize the export engine."""
        self.log = logger.bind(component="export_engine")

    def export(
        self,
        test_spec: dict,
        config: Optional[ExportConfig] = None,
    ) -> ExportResult:
        """Export a test specification to code.

        Args:
            test_spec: Test specification dictionary
            config: Export configuration (uses defaults if not provided)

        Returns:
            ExportResult with generated code or error
        """
        config = config or ExportConfig()

        # Validate config
        errors = config.validate()
        if errors:
            return ExportResult(
                success=False,
                error="; ".join(errors),
            )

        # Get template
        template_class = TEMPLATE_REGISTRY.get(config.framework)
        if not template_class:
            return ExportResult(
                success=False,
                error=f"No template available for {config.framework.value}",
            )

        try:
            # Create template with config
            template = template_class(config={
                "async_style": config.async_style,
                "test_class_name": config.test_class_name,
                "include_comments": config.include_comments,
            })

            # Generate code
            code = template.generate(test_spec)

            # Format code
            formatter = CodeFormatter(config.language)
            code = formatter.format_code(code)

            self.log.info(
                "Export successful",
                language=config.language.value,
                framework=config.framework.value,
                test_name=test_spec.get("name", "Unknown"),
            )

            return ExportResult(
                success=True,
                code=code,
                language=config.language,
                framework=config.framework,
                file_extension=FILE_EXTENSIONS.get(config.language, ".txt"),
                dependencies=FRAMEWORK_DEPENDENCIES.get(config.framework, []),
                metadata={
                    "test_id": test_spec.get("id", ""),
                    "test_name": test_spec.get("name", ""),
                    "steps_count": len(test_spec.get("steps", [])),
                    "assertions_count": len(test_spec.get("assertions", [])),
                },
            )

        except Exception as e:
            self.log.error("Export failed", error=str(e))
            return ExportResult(
                success=False,
                error=str(e),
            )

    def export_batch(
        self,
        test_specs: list[dict],
        config: Optional[ExportConfig] = None,
    ) -> list[ExportResult]:
        """Export multiple test specifications.

        Args:
            test_specs: List of test specification dictionaries
            config: Export configuration (same for all)

        Returns:
            List of ExportResults
        """
        return [self.export(spec, config) for spec in test_specs]

    def get_supported_combinations(self) -> dict[str, list[str]]:
        """Get supported language-framework combinations.

        Returns:
            Dict mapping language names to list of framework names
        """
        from .models import FRAMEWORK_SUPPORT

        return {
            lang.value: [fw.value for fw in frameworks]
            for lang, frameworks in FRAMEWORK_SUPPORT.items()
        }

    def preview(
        self,
        test_spec: dict,
        config: Optional[ExportConfig] = None,
        max_lines: int = 50,
    ) -> str:
        """Generate a preview of exported code.

        Args:
            test_spec: Test specification
            config: Export configuration
            max_lines: Maximum lines to return

        Returns:
            Code preview string
        """
        result = self.export(test_spec, config)
        if not result.success:
            return f"# Error: {result.error}"

        lines = result.code.split("\n")
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n\n... ({len(lines) - max_lines} more lines)"
        return result.code

    def get_file_info(self, config: ExportConfig) -> dict:
        """Get file information for a config.

        Args:
            config: Export configuration

        Returns:
            Dict with file info (extension, dependencies, install command)
        """
        from .formatters import ImportsManager

        manager = ImportsManager(config.language, config.framework)

        return {
            "extension": FILE_EXTENSIONS.get(config.language, ".txt"),
            "dependencies": manager.get_dependencies(),
            "install_command": manager.get_install_command(),
        }


# Convenience function for quick exports
def export_test(
    test_spec: dict,
    language: str = "python",
    framework: str = "playwright",
    **config_kwargs,
) -> ExportResult:
    """Quick export function.

    Args:
        test_spec: Test specification
        language: Target language
        framework: Target framework
        **config_kwargs: Additional ExportConfig options

    Returns:
        ExportResult
    """
    engine = ExportEngine()
    config = ExportConfig(
        language=language,
        framework=framework,
        **config_kwargs,
    )
    return engine.export(test_spec, config)
