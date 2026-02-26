"""Framework-specific test code generation from BDD scenarios.

Generates idiomatic test code for multiple frameworks (Playwright, Cypress,
Selenium, pytest, Robot Framework) from structured BDD scenarios.  Supports
both AI-powered generation (via an injected ``ai_caller``) and deterministic
template-based fallback when no AI is available.

Like ``RequirementsParser``, this module does NOT extend ``BaseAgent``.
It accepts an optional *ai_caller* async function so the owning agent
(e.g. ``TestGenAgent``) can inject its own method and retain unified
cost/token tracking.

Example usage::

    from src.agents.testgen.code_emitters import CodeEmitter, EmitterConfig

    emitter = CodeEmitter(ai_caller=self._call_ai)
    tests = await emitter.emit(scenarios, EmitterConfig(framework="cypress"))
"""

from __future__ import annotations

import json
import re
import textwrap
from enum import Enum
from typing import Any, Callable, Coroutine

import structlog
from pydantic import BaseModel, Field

from src.core.model_router import TaskType

logger = structlog.get_logger(__name__)

# Type alias matching BaseAgent._call_ai signature (kept loose to avoid
# circular imports -- we only rely on ``response.content: str``).
AICaller = Callable[..., Coroutine[Any, Any, Any]]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestFramework(str, Enum):
    """Supported test framework targets."""

    PLAYWRIGHT = "playwright"
    CYPRESS = "cypress"
    SELENIUM = "selenium"
    PYTEST = "pytest"
    ROBOT = "robot"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class BDDScenario(BaseModel):
    """A single BDD scenario derived from a requirement."""

    name: str
    feature: str
    given: list[str]
    when: list[str]
    then: list[str]
    tags: list[str] = Field(default_factory=list)
    requirement_id: str | None = None
    examples: list[dict] | None = None  # For Scenario Outline


class GeneratedTestCase(BaseModel):
    """A generated test case ready to be written to disk."""

    name: str
    code: str
    framework: str
    language: str  # python, typescript, javascript, robotframework
    bdd_scenario: BDDScenario | None = None
    requirement_ids: list[str] = Field(default_factory=list)
    quality_score: float = 0.0
    file_path: str = ""  # suggested file path


class EmitterConfig(BaseModel):
    """Configuration controlling code generation behaviour."""

    framework: str = "playwright"
    language: str = "python"  # auto-detected from framework if not specified
    base_url: str = ""
    include_comments: bool = True
    include_page_objects: bool = False
    max_tests_per_file: int = 10


# ---------------------------------------------------------------------------
# Framework  ->  default language mapping
# ---------------------------------------------------------------------------

_FRAMEWORK_DEFAULT_LANGUAGE: dict[str, str] = {
    TestFramework.PLAYWRIGHT.value: "python",
    TestFramework.CYPRESS.value: "javascript",
    TestFramework.SELENIUM.value: "python",
    TestFramework.PYTEST.value: "python",
    TestFramework.ROBOT.value: "robotframework",
}

_FRAMEWORK_LANGUAGE_OPTIONS: dict[str, set[str]] = {
    TestFramework.PLAYWRIGHT.value: {"python", "typescript"},
    TestFramework.CYPRESS.value: {"javascript", "typescript"},
    TestFramework.SELENIUM.value: {"python", "javascript", "typescript"},
    TestFramework.PYTEST.value: {"python"},
    TestFramework.ROBOT.value: {"robotframework"},
}


# ---------------------------------------------------------------------------
# Prompts for AI-powered generation
# ---------------------------------------------------------------------------

_AI_CODE_GENERATION_PROMPT = """\
You are an expert test automation engineer.  Generate idiomatic, production-\
ready test code for the **{framework}** framework in **{language}**.

REQUIREMENTS:
- Each test MUST be self-contained and runnable
- Use modern async/await patterns where the framework supports them
- Include descriptive assertion messages
- Follow the BDD steps (Given / When / Then) as inline comments
- Use page-object pattern if ``include_page_objects`` is true
{base_url_instruction}

OUTPUT FORMAT — return a JSON array where each element has:
- "name": the test function/method name (snake_case for Python, camelCase for JS/TS)
- "code": the complete test code as a string
- "file_path": suggested file path relative to project root
- "quality_score": self-assessed quality 0.0-1.0

Only output valid JSON, no markdown fences.

---
BDD SCENARIOS:
{scenarios_gherkin}
"""


# ---------------------------------------------------------------------------
# Sanitisation helpers
# ---------------------------------------------------------------------------


def _sanitize_name(name: str, style: str = "snake") -> str:
    """Convert a human-readable name to a valid identifier.

    Args:
        name: The raw name to sanitize.
        style: ``"snake"`` for Python/Robot, ``"camel"`` for JS/TS.

    Returns:
        A sanitized identifier string.
    """
    # Strip non-alphanumeric characters (keep spaces/underscores temporarily)
    cleaned = re.sub(r"[^a-zA-Z0-9\s_]", "", name)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if style == "camel":
        parts = cleaned.split()
        if not parts:
            return "unnamedTest"
        return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])

    # snake_case
    return re.sub(r"\s+", "_", cleaned).lower() or "unnamed_test"


def _sanitize_class_name(name: str) -> str:
    """Convert a feature name to a valid PascalCase class name."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", name)
    parts = cleaned.split()
    if not parts:
        return "UnnamedFeature"
    return "".join(p.capitalize() for p in parts)


# ---------------------------------------------------------------------------
# Template-based emitters (no AI required)
# ---------------------------------------------------------------------------


def _indent(text: str, spaces: int = 4) -> str:
    """Indent every line of *text* by *spaces* spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def _steps_as_comments(steps: list[str], prefix: str = "", indent_level: int = 8) -> str:
    """Render BDD steps as inline comments."""
    if not steps:
        return ""
    pad = " " * indent_level
    lines = [f"{pad}# {prefix}: {step}" for step in steps]
    return "\n".join(lines)


def _steps_as_js_comments(steps: list[str], prefix: str = "", indent_level: int = 4) -> str:
    """Render BDD steps as JavaScript-style inline comments."""
    if not steps:
        return ""
    pad = " " * indent_level
    lines = [f"{pad}// {prefix}: {step}" for step in steps]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CodeEmitter
# ---------------------------------------------------------------------------


class CodeEmitter:
    """Generates framework-specific test code from BDD scenarios.

    Args:
        ai_caller: Optional async function matching ``BaseAgent._call_ai``
            signature.  When provided, generation uses AI for idiomatic code.
            When ``None``, falls back to deterministic templates.
    """

    def __init__(self, ai_caller: AICaller | None = None) -> None:
        self._call_ai = ai_caller
        self._log = logger.bind(component="CodeEmitter")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def emit(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Generate test code for *scenarios* using *config*.

        This is the main dispatcher.  It normalises the config, picks the
        right emitter method, and returns generated test cases.

        Args:
            scenarios: BDD scenarios to convert to test code.
            config: Emission configuration (framework, language, etc.).

        Returns:
            A list of ``GeneratedTestCase`` objects.
        """
        if not scenarios:
            self._log.warning("No scenarios provided to emit")
            return []

        framework = config.framework.lower()
        language = config.language.lower()

        # Auto-detect language from framework if using the default
        valid_languages = _FRAMEWORK_LANGUAGE_OPTIONS.get(framework, set())
        if language not in valid_languages:
            language = _FRAMEWORK_DEFAULT_LANGUAGE.get(framework, "python")
            self._log.info(
                "Auto-detected language from framework",
                framework=framework,
                language=language,
            )

        self._log.info(
            "Emitting test code",
            framework=framework,
            language=language,
            scenario_count=len(scenarios),
            ai_enabled=self._call_ai is not None,
        )

        # Attempt AI-powered generation first
        if self._call_ai is not None:
            try:
                results = await self._ai_generate_code(
                    scenarios, framework, language, config
                )
                if results:
                    self._log.info(
                        "AI code generation succeeded",
                        test_count=len(results),
                    )
                    return results
                self._log.warning(
                    "AI code generation returned empty results, falling back to templates"
                )
            except Exception:
                self._log.exception(
                    "AI code generation failed, falling back to templates"
                )

        # Template-based fallback
        dispatch: dict[tuple[str, str], Callable[..., list[GeneratedTestCase]]] = {
            (TestFramework.PLAYWRIGHT.value, "python"): self._emit_playwright_python,
            (TestFramework.PLAYWRIGHT.value, "typescript"): self._emit_playwright_typescript,
            (TestFramework.CYPRESS.value, "javascript"): self._emit_cypress,
            (TestFramework.CYPRESS.value, "typescript"): self._emit_cypress,
            (TestFramework.SELENIUM.value, "python"): self._emit_selenium_python,
            (TestFramework.PYTEST.value, "python"): self._emit_pytest,
            (TestFramework.ROBOT.value, "robotframework"): self._emit_robot,
        }

        emitter_fn = dispatch.get((framework, language))
        if emitter_fn is None:
            self._log.warning(
                "No template emitter for framework/language combination, "
                "defaulting to playwright/python",
                framework=framework,
                language=language,
            )
            emitter_fn = self._emit_playwright_python

        return emitter_fn(scenarios, config)

    # ------------------------------------------------------------------
    # Framework-specific emitters (template-based)
    # ------------------------------------------------------------------

    def emit_playwright_python(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Public alias for Playwright + Python template emitter."""
        return self._emit_playwright_python(scenarios, config)

    def emit_playwright_typescript(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Public alias for Playwright + TypeScript template emitter."""
        return self._emit_playwright_typescript(scenarios, config)

    def emit_cypress(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Public alias for Cypress template emitter."""
        return self._emit_cypress(scenarios, config)

    def emit_selenium_python(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Public alias for Selenium + Python template emitter."""
        return self._emit_selenium_python(scenarios, config)

    def emit_pytest(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Public alias for pytest template emitter."""
        return self._emit_pytest(scenarios, config)

    def emit_robot(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Public alias for Robot Framework template emitter."""
        return self._emit_robot(scenarios, config)

    # ------------------------------------------------------------------
    # Playwright + Python
    # ------------------------------------------------------------------

    def _emit_playwright_python(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Generate Playwright Python test code from BDD scenarios."""
        results: list[GeneratedTestCase] = []

        # Group scenarios by feature for class-based organisation
        feature_groups: dict[str, list[BDDScenario]] = {}
        for scenario in scenarios:
            feature_groups.setdefault(scenario.feature, []).append(scenario)

        for feature_name, feature_scenarios in feature_groups.items():
            class_name = f"Test{_sanitize_class_name(feature_name)}"
            feature_slug = _sanitize_name(feature_name)

            methods: list[str] = []
            for scenario in feature_scenarios:
                method_name = f"test_{_sanitize_name(scenario.name)}"
                tags_decorator = ""
                if scenario.tags:
                    marks = ", ".join(
                        f'pytest.mark.{_sanitize_name(t)}' for t in scenario.tags
                    )
                    tags_decorator = f"    @{marks}\n"

                # Build step comments
                comment_lines: list[str] = []
                if config.include_comments:
                    for step in scenario.given:
                        comment_lines.append(f"        # Given: {step}")
                    for step in scenario.when:
                        comment_lines.append(f"        # When: {step}")
                    for step in scenario.then:
                        comment_lines.append(f"        # Then: {step}")

                comment_block = "\n".join(comment_lines) + "\n" if comment_lines else ""

                # Base URL navigation
                base_nav = ""
                if config.base_url:
                    base_nav = f'        page.goto("{config.base_url}")\n'

                method_code = (
                    f"{tags_decorator}"
                    f"    def {method_name}(self, page: Page):\n"
                    f'        """Scenario: {scenario.name}"""\n'
                    f"{comment_block}"
                    f"{base_nav}"
                    f"        # TODO: Implement test steps\n"
                    f"        pass\n"
                )
                methods.append(method_code)

            methods_block = "\n\n".join(methods)
            file_code = textwrap.dedent(f"""\
                import pytest
                from playwright.sync_api import Page, expect


                class {class_name}:
                    \"\"\"Tests for: {feature_name}\"\"\"

                {methods_block}
            """).rstrip() + "\n"

            file_path = f"tests/e2e/test_{feature_slug}.py"

            for i, scenario in enumerate(feature_scenarios):
                req_ids = [scenario.requirement_id] if scenario.requirement_id else []
                results.append(
                    GeneratedTestCase(
                        name=f"test_{_sanitize_name(scenario.name)}",
                        code=file_code if i == 0 else "",  # Full file only on first
                        framework=TestFramework.PLAYWRIGHT.value,
                        language="python",
                        bdd_scenario=scenario,
                        requirement_ids=req_ids,
                        quality_score=0.4,  # Template quality baseline
                        file_path=file_path,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Playwright + TypeScript
    # ------------------------------------------------------------------

    def _emit_playwright_typescript(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Generate Playwright TypeScript test code from BDD scenarios."""
        results: list[GeneratedTestCase] = []

        feature_groups: dict[str, list[BDDScenario]] = {}
        for scenario in scenarios:
            feature_groups.setdefault(scenario.feature, []).append(scenario)

        for feature_name, feature_scenarios in feature_groups.items():
            feature_slug = _sanitize_name(feature_name)
            tests_block_parts: list[str] = []

            for scenario in feature_scenarios:
                test_name = scenario.name
                comment_lines: list[str] = []
                if config.include_comments:
                    for step in scenario.given:
                        comment_lines.append(f"    // Given: {step}")
                    for step in scenario.when:
                        comment_lines.append(f"    // When: {step}")
                    for step in scenario.then:
                        comment_lines.append(f"    // Then: {step}")

                comment_block = "\n".join(comment_lines) + "\n" if comment_lines else ""

                base_nav = ""
                if config.base_url:
                    base_nav = f"    await page.goto('{config.base_url}');\n"

                test_block = (
                    f"  test('{test_name}', async ({{ page }}) => {{\n"
                    f"{comment_block}"
                    f"{base_nav}"
                    f"    // TODO: Implement test steps\n"
                    f"  }});\n"
                )
                tests_block_parts.append(test_block)

            tests_body = "\n".join(tests_block_parts)
            file_code = textwrap.dedent(f"""\
                import {{ test, expect }} from '@playwright/test';

                test.describe('{feature_name}', () => {{
                {tests_body}
                }});
            """).rstrip() + "\n"

            file_path = f"tests/e2e/{feature_slug}.spec.ts"

            for i, scenario in enumerate(feature_scenarios):
                req_ids = [scenario.requirement_id] if scenario.requirement_id else []
                results.append(
                    GeneratedTestCase(
                        name=_sanitize_name(scenario.name, style="camel"),
                        code=file_code if i == 0 else "",
                        framework=TestFramework.PLAYWRIGHT.value,
                        language="typescript",
                        bdd_scenario=scenario,
                        requirement_ids=req_ids,
                        quality_score=0.4,
                        file_path=file_path,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Cypress (JavaScript / TypeScript)
    # ------------------------------------------------------------------

    def _emit_cypress(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Generate Cypress test code from BDD scenarios."""
        results: list[GeneratedTestCase] = []

        feature_groups: dict[str, list[BDDScenario]] = {}
        for scenario in scenarios:
            feature_groups.setdefault(scenario.feature, []).append(scenario)

        for feature_name, feature_scenarios in feature_groups.items():
            feature_slug = _sanitize_name(feature_name)
            it_blocks: list[str] = []

            for scenario in feature_scenarios:
                comment_lines: list[str] = []
                if config.include_comments:
                    for step in scenario.given:
                        comment_lines.append(f"    // Given: {step}")
                    for step in scenario.when:
                        comment_lines.append(f"    // When: {step}")
                    for step in scenario.then:
                        comment_lines.append(f"    // Then: {step}")

                comment_block = "\n".join(comment_lines) + "\n" if comment_lines else ""

                base_nav = ""
                if config.base_url:
                    base_nav = f"    cy.visit('{config.base_url}');\n"

                it_block = (
                    f"  it('{scenario.name}', () => {{\n"
                    f"{comment_block}"
                    f"{base_nav}"
                    f"    // TODO: Implement test steps\n"
                    f"  }});\n"
                )
                it_blocks.append(it_block)

            before_each = ""
            if config.base_url:
                before_each = (
                    f"  beforeEach(() => {{\n"
                    f"    cy.visit('{config.base_url}');\n"
                    f"  }});\n\n"
                )

            its_body = "\n".join(it_blocks)
            file_code = textwrap.dedent(f"""\
                describe('{feature_name}', () => {{
                {before_each}{its_body}
                }});
            """).rstrip() + "\n"

            ext = "ts" if config.language == "typescript" else "js"
            file_path = f"cypress/e2e/{feature_slug}.cy.{ext}"

            for i, scenario in enumerate(feature_scenarios):
                req_ids = [scenario.requirement_id] if scenario.requirement_id else []
                results.append(
                    GeneratedTestCase(
                        name=_sanitize_name(scenario.name, style="camel"),
                        code=file_code if i == 0 else "",
                        framework=TestFramework.CYPRESS.value,
                        language=config.language or "javascript",
                        bdd_scenario=scenario,
                        requirement_ids=req_ids,
                        quality_score=0.4,
                        file_path=file_path,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Selenium + Python
    # ------------------------------------------------------------------

    def _emit_selenium_python(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Generate Selenium Python (unittest-style) test code."""
        results: list[GeneratedTestCase] = []

        feature_groups: dict[str, list[BDDScenario]] = {}
        for scenario in scenarios:
            feature_groups.setdefault(scenario.feature, []).append(scenario)

        for feature_name, feature_scenarios in feature_groups.items():
            class_name = f"Test{_sanitize_class_name(feature_name)}"
            feature_slug = _sanitize_name(feature_name)

            methods: list[str] = []
            for scenario in feature_scenarios:
                method_name = f"test_{_sanitize_name(scenario.name)}"

                comment_lines: list[str] = []
                if config.include_comments:
                    for step in scenario.given:
                        comment_lines.append(f"        # Given: {step}")
                    for step in scenario.when:
                        comment_lines.append(f"        # When: {step}")
                    for step in scenario.then:
                        comment_lines.append(f"        # Then: {step}")

                comment_block = "\n".join(comment_lines) + "\n" if comment_lines else ""

                base_nav = ""
                if config.base_url:
                    base_nav = f'        self.driver.get("{config.base_url}")\n'

                method_code = (
                    f"    def {method_name}(self):\n"
                    f'        """Scenario: {scenario.name}"""\n'
                    f"{comment_block}"
                    f"{base_nav}"
                    f"        # TODO: Implement test steps\n"
                    f"        pass\n"
                )
                methods.append(method_code)

            methods_block = "\n\n".join(methods)

            setup_url = ""
            if config.base_url:
                setup_url = f'        self.base_url = "{config.base_url}"\n'

            file_code = textwrap.dedent(f"""\
                import unittest

                from selenium import webdriver
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support import expected_conditions as EC
                from selenium.webdriver.support.ui import WebDriverWait


                class {class_name}(unittest.TestCase):
                    \"\"\"Tests for: {feature_name}\"\"\"

                    def setUp(self):
                        self.driver = webdriver.Chrome()
                        self.driver.implicitly_wait(10)
                {setup_url}
                    def tearDown(self):
                        self.driver.quit()

                {methods_block}


                if __name__ == "__main__":
                    unittest.main()
            """).rstrip() + "\n"

            file_path = f"tests/selenium/test_{feature_slug}.py"

            for i, scenario in enumerate(feature_scenarios):
                req_ids = [scenario.requirement_id] if scenario.requirement_id else []
                results.append(
                    GeneratedTestCase(
                        name=f"test_{_sanitize_name(scenario.name)}",
                        code=file_code if i == 0 else "",
                        framework=TestFramework.SELENIUM.value,
                        language="python",
                        bdd_scenario=scenario,
                        requirement_ids=req_ids,
                        quality_score=0.35,
                        file_path=file_path,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # pytest (unit / integration)
    # ------------------------------------------------------------------

    def _emit_pytest(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Generate pytest test code (non-browser, unit/integration style)."""
        results: list[GeneratedTestCase] = []

        feature_groups: dict[str, list[BDDScenario]] = {}
        for scenario in scenarios:
            feature_groups.setdefault(scenario.feature, []).append(scenario)

        for feature_name, feature_scenarios in feature_groups.items():
            feature_slug = _sanitize_name(feature_name)
            functions: list[str] = []

            for scenario in feature_scenarios:
                func_name = f"test_{_sanitize_name(scenario.name)}"

                # Build markers from tags
                markers: list[str] = []
                for tag in scenario.tags:
                    markers.append(f"@pytest.mark.{_sanitize_name(tag)}")

                markers_block = "\n".join(markers) + "\n" if markers else ""

                comment_lines: list[str] = []
                if config.include_comments:
                    for step in scenario.given:
                        comment_lines.append(f"    # Given: {step}")
                    for step in scenario.when:
                        comment_lines.append(f"    # When: {step}")
                    for step in scenario.then:
                        comment_lines.append(f"    # Then: {step}")

                comment_block = "\n".join(comment_lines) + "\n" if comment_lines else ""

                # Scenario Outline -> parametrize
                parametrize_block = ""
                if scenario.examples:
                    if scenario.examples:
                        param_keys = list(scenario.examples[0].keys())
                        param_str = ", ".join(f'"{k}"' for k in param_keys)
                        values_lines: list[str] = []
                        for example in scenario.examples:
                            vals = ", ".join(repr(example.get(k, "")) for k in param_keys)
                            values_lines.append(f"        ({vals}),")
                        values_block = "\n".join(values_lines)
                        parametrize_block = (
                            f'@pytest.mark.parametrize(\n'
                            f'    {param_str},\n'
                            f'    [\n'
                            f'{values_block}\n'
                            f'    ],\n'
                            f')\n'
                        )

                func_args = ""
                if scenario.examples and scenario.examples:
                    param_keys = list(scenario.examples[0].keys())
                    func_args = ", ".join(param_keys)

                func_code = (
                    f"{markers_block}"
                    f"{parametrize_block}"
                    f"def {func_name}({func_args}):\n"
                    f'    """Scenario: {scenario.name}"""\n'
                    f"{comment_block}"
                    f"    # TODO: Implement test steps\n"
                    f"    pass\n"
                )
                functions.append(func_code)

            functions_block = "\n\n".join(functions)
            file_code = textwrap.dedent(f"""\
                import pytest


                # Feature: {feature_name}

                {functions_block}
            """).rstrip() + "\n"

            file_path = f"tests/test_{feature_slug}.py"

            for i, scenario in enumerate(feature_scenarios):
                req_ids = [scenario.requirement_id] if scenario.requirement_id else []
                results.append(
                    GeneratedTestCase(
                        name=f"test_{_sanitize_name(scenario.name)}",
                        code=file_code if i == 0 else "",
                        framework=TestFramework.PYTEST.value,
                        language="python",
                        bdd_scenario=scenario,
                        requirement_ids=req_ids,
                        quality_score=0.4,
                        file_path=file_path,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Robot Framework
    # ------------------------------------------------------------------

    def _emit_robot(
        self,
        scenarios: list[BDDScenario],
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Generate Robot Framework test code from BDD scenarios."""
        results: list[GeneratedTestCase] = []

        feature_groups: dict[str, list[BDDScenario]] = {}
        for scenario in scenarios:
            feature_groups.setdefault(scenario.feature, []).append(scenario)

        for feature_name, feature_scenarios in feature_groups.items():
            feature_slug = _sanitize_name(feature_name)

            # Settings section
            settings_lines = [
                "*** Settings ***",
                "Library    Browser",
                "Library    Collections",
            ]
            if config.base_url:
                settings_lines.append(f"Suite Setup    New Browser    chromium    headless=true")
                settings_lines.append(f"Test Setup     New Page    {config.base_url}")
            settings_block = "\n".join(settings_lines)

            # Test cases section
            test_case_blocks: list[str] = []
            for scenario in feature_scenarios:
                test_name = scenario.name
                tags_line = ""
                if scenario.tags:
                    tags_line = f"    [Tags]    {'    '.join(scenario.tags)}\n"

                step_lines: list[str] = []
                if config.include_comments:
                    for step in scenario.given:
                        step_lines.append(f"    Log    Given: {step}")
                    for step in scenario.when:
                        step_lines.append(f"    Log    When: {step}")
                    for step in scenario.then:
                        step_lines.append(f"    Log    Then: {step}")

                steps_block = "\n".join(step_lines) if step_lines else "    Log    TODO: Implement test steps"

                test_block = (
                    f"{test_name}\n"
                    f"    [Documentation]    Scenario: {scenario.name}\n"
                    f"{tags_line}"
                    f"{steps_block}\n"
                )
                test_case_blocks.append(test_block)

            test_cases_body = "\n".join(test_case_blocks)
            file_code = (
                f"{settings_block}\n\n"
                f"*** Test Cases ***\n"
                f"{test_cases_body}"
            ).rstrip() + "\n"

            file_path = f"tests/robot/{feature_slug}.robot"

            for i, scenario in enumerate(feature_scenarios):
                req_ids = [scenario.requirement_id] if scenario.requirement_id else []
                results.append(
                    GeneratedTestCase(
                        name=scenario.name,
                        code=file_code if i == 0 else "",
                        framework=TestFramework.ROBOT.value,
                        language="robotframework",
                        bdd_scenario=scenario,
                        requirement_ids=req_ids,
                        quality_score=0.35,
                        file_path=file_path,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # BDD -> Gherkin conversion
    # ------------------------------------------------------------------

    def _scenario_to_gherkin(self, scenario: BDDScenario) -> str:
        """Convert a ``BDDScenario`` to Gherkin feature-file text.

        Args:
            scenario: The BDD scenario to convert.

        Returns:
            A Gherkin-formatted string for this scenario.
        """
        lines: list[str] = []

        # Tags
        if scenario.tags:
            tags_str = " ".join(f"@{t}" for t in scenario.tags)
            lines.append(f"  {tags_str}")

        # Scenario keyword (Outline if examples exist)
        keyword = "Scenario Outline" if scenario.examples else "Scenario"
        lines.append(f"  {keyword}: {scenario.name}")

        # Steps
        for i, step in enumerate(scenario.given):
            prefix = "Given" if i == 0 else "And"
            lines.append(f"    {prefix} {step}")

        for i, step in enumerate(scenario.when):
            prefix = "When" if i == 0 else "And"
            lines.append(f"    {prefix} {step}")

        for i, step in enumerate(scenario.then):
            prefix = "Then" if i == 0 else "And"
            lines.append(f"    {prefix} {step}")

        # Examples table for Scenario Outline
        if scenario.examples:
            lines.append("")
            lines.append("    Examples:")
            headers = list(scenario.examples[0].keys())
            header_line = "      | " + " | ".join(headers) + " |"
            lines.append(header_line)
            for example in scenario.examples:
                row = "      | " + " | ".join(str(example.get(h, "")) for h in headers) + " |"
                lines.append(row)

        return "\n".join(lines)

    def _scenarios_to_gherkin(self, scenarios: list[BDDScenario]) -> str:
        """Convert multiple scenarios into a single Gherkin document.

        Groups scenarios by feature and produces a well-formed feature file.

        Args:
            scenarios: BDD scenarios to convert.

        Returns:
            A complete Gherkin feature file string.
        """
        feature_groups: dict[str, list[BDDScenario]] = {}
        for scenario in scenarios:
            feature_groups.setdefault(scenario.feature, []).append(scenario)

        parts: list[str] = []
        for feature_name, feature_scenarios in feature_groups.items():
            feature_block = f"Feature: {feature_name}\n"
            for scenario in feature_scenarios:
                feature_block += "\n" + self._scenario_to_gherkin(scenario) + "\n"
            parts.append(feature_block)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # AI-powered code generation
    # ------------------------------------------------------------------

    async def _ai_generate_code(
        self,
        scenarios: list[BDDScenario],
        framework: str,
        language: str,
        config: EmitterConfig,
    ) -> list[GeneratedTestCase]:
        """Use AI to generate idiomatic framework-specific test code.

        Falls back to an empty list (triggering template fallback) if the
        AI call fails or returns unparseable output.

        Args:
            scenarios: BDD scenarios to generate code for.
            framework: Target framework name.
            language: Target programming language.
            config: Emitter configuration.

        Returns:
            List of ``GeneratedTestCase`` or empty list on failure.
        """
        assert self._call_ai is not None, "_ai_generate_code requires ai_caller"

        gherkin_text = self._scenarios_to_gherkin(scenarios)

        base_url_instruction = ""
        if config.base_url:
            base_url_instruction = f"- Navigate to base URL: {config.base_url}"

        prompt = _AI_CODE_GENERATION_PROMPT.format(
            framework=framework,
            language=language,
            base_url_instruction=base_url_instruction,
            scenarios_gherkin=gherkin_text,
        )

        self._log.debug(
            "Sending AI code generation request",
            framework=framework,
            language=language,
            scenario_count=len(scenarios),
        )

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.TEST_GENERATION,
            max_cost=0.08,
            max_tokens=8192,
            temperature=0.1,
            json_mode=True,
        )

        raw = self._safe_json_loads(response.content)

        # Handle wrapped responses
        if isinstance(raw, dict):
            for key in ("tests", "test_cases", "results", "data", "items"):
                if isinstance(raw.get(key), list):
                    raw = raw[key]
                    break
            else:
                self._log.warning(
                    "AI returned dict without recognised list key",
                    keys=list(raw.keys()),
                )
                return []

        if not isinstance(raw, list):
            self._log.warning(
                "AI returned non-list response",
                type=type(raw).__name__,
            )
            return []

        # Build a lookup from scenario name -> scenario for linking
        scenario_lookup: dict[str, BDDScenario] = {}
        for s in scenarios:
            scenario_lookup[_sanitize_name(s.name)] = s
            scenario_lookup[s.name.lower()] = s

        results: list[GeneratedTestCase] = []
        for item in raw:
            if not isinstance(item, dict):
                continue

            name = item.get("name", "unnamed_test")
            code = item.get("code", "")
            file_path = item.get("file_path", "")
            quality = float(item.get("quality_score", 0.6))

            if not code:
                self._log.debug("AI returned test with empty code, skipping", name=name)
                continue

            # Try to link back to a scenario
            matched_scenario: BDDScenario | None = None
            sanitized = _sanitize_name(name)
            # Strip common prefixes like test_
            stripped = re.sub(r"^test_?", "", sanitized)
            matched_scenario = (
                scenario_lookup.get(sanitized)
                or scenario_lookup.get(stripped)
                or scenario_lookup.get(name.lower())
            )

            req_ids: list[str] = []
            if matched_scenario and matched_scenario.requirement_id:
                req_ids = [matched_scenario.requirement_id]

            results.append(
                GeneratedTestCase(
                    name=name,
                    code=code,
                    framework=framework,
                    language=language,
                    bdd_scenario=matched_scenario,
                    requirement_ids=req_ids,
                    quality_score=min(1.0, max(0.0, quality)),
                    file_path=file_path,
                )
            )

        self._log.info(
            "AI code generation parsed",
            generated_count=len(results),
            input_scenarios=len(scenarios),
        )
        return results

    # ------------------------------------------------------------------
    # JSON parsing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_json_loads(text: str) -> Any:
        """Robustly parse JSON from AI output.

        Strips markdown fences, trailing commas, and other common AI
        output artefacts before parsing.

        Args:
            text: Raw text that should contain JSON.

        Returns:
            Parsed JSON value, or ``None`` on parse failure.
        """
        if not text:
            return None

        cleaned = text.strip()

        # Remove markdown code fences
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag)
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            # Remove closing fence
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            cleaned = cleaned.strip()

        # Remove trailing commas before closing brackets/braces
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: try to find the first JSON array or object
            for match in re.finditer(r"[\[{]", cleaned):
                try:
                    return json.loads(cleaned[match.start():])
                except json.JSONDecodeError:
                    continue
            return None
