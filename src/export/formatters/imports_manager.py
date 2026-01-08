"""Import/dependency manager for exports."""

from dataclasses import dataclass, field
from typing import Optional
from ..models import SupportedLanguage, SupportedFramework, FRAMEWORK_DEPENDENCIES


@dataclass
class ImportSpec:
    """Specification for an import statement."""

    module: str
    items: list[str] = field(default_factory=list)
    alias: Optional[str] = None
    is_type_import: bool = False  # For TypeScript type imports


class ImportsManager:
    """Manages imports and dependencies for generated code."""

    def __init__(self, language: SupportedLanguage, framework: SupportedFramework):
        """Initialize imports manager.

        Args:
            language: Target language
            framework: Target framework
        """
        self.language = language
        self.framework = framework
        self._imports: list[ImportSpec] = []
        self._initialize_base_imports()

    def _initialize_base_imports(self):
        """Initialize base imports for the framework."""
        if self.framework == SupportedFramework.PLAYWRIGHT_PYTHON:
            self.add_import("pytest")
            self.add_import("playwright.sync_api", items=["Page", "expect"])
            self.add_import("re")

        elif self.framework == SupportedFramework.SELENIUM_PYTHON:
            self.add_import("pytest")
            self.add_import("selenium", items=["webdriver"])
            self.add_import("selenium.webdriver.common.by", items=["By"])
            self.add_import("selenium.webdriver.common.keys", items=["Keys"])
            self.add_import("selenium.webdriver.support.ui", items=["WebDriverWait"])
            self.add_import(
                "selenium.webdriver.support", items=["expected_conditions"], alias="EC"
            )

        elif self.framework == SupportedFramework.PLAYWRIGHT_TS:
            self.add_import("@playwright/test", items=["test", "expect"])

        elif self.framework == SupportedFramework.CYPRESS:
            pass  # Cypress globals are auto-imported

        elif self.framework == SupportedFramework.PUPPETEER:
            self.add_import("puppeteer")
            self.add_import("jest")

    def add_import(
        self,
        module: str,
        items: Optional[list[str]] = None,
        alias: Optional[str] = None,
        is_type_import: bool = False,
    ):
        """Add an import.

        Args:
            module: Module/package name
            items: Specific items to import
            alias: Alias for the import
            is_type_import: Whether this is a TypeScript type import
        """
        spec = ImportSpec(
            module=module,
            items=items or [],
            alias=alias,
            is_type_import=is_type_import,
        )
        # Avoid duplicates
        existing = next((i for i in self._imports if i.module == module), None)
        if existing:
            # Merge items
            existing.items = list(set(existing.items + spec.items))
        else:
            self._imports.append(spec)

    def get_imports_code(self) -> str:
        """Generate import statements.

        Returns:
            Import statements as code
        """
        if self.language == SupportedLanguage.PYTHON:
            return self._generate_python_imports()
        elif self.language == SupportedLanguage.TYPESCRIPT:
            return self._generate_typescript_imports()
        elif self.language == SupportedLanguage.JAVA:
            return self._generate_java_imports()
        elif self.language == SupportedLanguage.CSHARP:
            return self._generate_csharp_imports()
        elif self.language == SupportedLanguage.RUBY:
            return self._generate_ruby_imports()
        elif self.language == SupportedLanguage.GO:
            return self._generate_go_imports()
        return ""

    def _generate_python_imports(self) -> str:
        """Generate Python import statements."""
        lines = []
        for spec in self._imports:
            if spec.items:
                items_str = ", ".join(spec.items)
                if spec.alias:
                    lines.append(f"from {spec.module} import {items_str} as {spec.alias}")
                else:
                    lines.append(f"from {spec.module} import {items_str}")
            else:
                if spec.alias:
                    lines.append(f"import {spec.module} as {spec.alias}")
                else:
                    lines.append(f"import {spec.module}")
        return "\n".join(lines)

    def _generate_typescript_imports(self) -> str:
        """Generate TypeScript import statements."""
        lines = []
        for spec in self._imports:
            type_prefix = "type " if spec.is_type_import else ""
            if spec.items:
                items_str = ", ".join(spec.items)
                lines.append(f"import {{ {type_prefix}{items_str} }} from '{spec.module}';")
            else:
                if spec.alias:
                    lines.append(f"import * as {spec.alias} from '{spec.module}';")
                else:
                    lines.append(f"import '{spec.module}';")
        return "\n".join(lines)

    def _generate_java_imports(self) -> str:
        """Generate Java import statements."""
        lines = []
        for spec in self._imports:
            if spec.items:
                for item in spec.items:
                    lines.append(f"import {spec.module}.{item};")
            else:
                lines.append(f"import {spec.module}.*;")
        return "\n".join(sorted(set(lines)))  # Sort and dedupe

    def _generate_csharp_imports(self) -> str:
        """Generate C# using statements."""
        lines = []
        for spec in self._imports:
            if spec.alias:
                lines.append(f"using {spec.alias} = {spec.module};")
            else:
                lines.append(f"using {spec.module};")
        return "\n".join(sorted(set(lines)))

    def _generate_ruby_imports(self) -> str:
        """Generate Ruby require statements."""
        lines = []
        for spec in self._imports:
            lines.append(f"require '{spec.module}'")
        return "\n".join(lines)

    def _generate_go_imports(self) -> str:
        """Generate Go import statements."""
        lines = ["import ("]
        for spec in self._imports:
            if spec.alias:
                lines.append(f'\t{spec.alias} "{spec.module}"')
            else:
                lines.append(f'\t"{spec.module}"')
        lines.append(")")
        return "\n".join(lines)

    def get_dependencies(self) -> list[str]:
        """Get list of package dependencies.

        Returns:
            List of package/dependency names
        """
        return FRAMEWORK_DEPENDENCIES.get(self.framework, [])

    def get_install_command(self) -> str:
        """Get command to install dependencies.

        Returns:
            Install command string
        """
        deps = self.get_dependencies()
        if not deps:
            return ""

        if self.language == SupportedLanguage.PYTHON:
            return f"pip install {' '.join(deps)}"
        elif self.language == SupportedLanguage.TYPESCRIPT:
            return f"npm install {' '.join(deps)}"
        elif self.language == SupportedLanguage.JAVA:
            # Maven coordinates
            return f"# Add to pom.xml: {', '.join(deps)}"
        elif self.language == SupportedLanguage.CSHARP:
            return f"dotnet add package {' '.join(deps)}"
        elif self.language == SupportedLanguage.RUBY:
            return f"gem install {' '.join(deps)}"
        elif self.language == SupportedLanguage.GO:
            return f"go get {' '.join(deps)}"

        return ""
