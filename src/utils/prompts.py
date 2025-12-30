"""Reusable prompt templates for E2E testing agents.

Provides:
- Standardized prompts for different agent tasks
- Variable substitution
- Prompt composition
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PromptTemplate:
    """A reusable prompt template with variable substitution.

    Usage:
        template = PromptTemplate(
            name="analyze_code",
            template="Analyze {codebase_path} at {app_url}",
            required_vars=["codebase_path", "app_url"],
        )
        prompt = template.render(codebase_path="/app", app_url="http://localhost:3000")
    """

    name: str
    template: str
    required_vars: list[str]
    optional_vars: list[str] = None
    description: str = ""

    def __post_init__(self):
        if self.optional_vars is None:
            self.optional_vars = []

    def render(self, **kwargs) -> str:
        """Render the template with variables.

        Args:
            **kwargs: Variable values

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If required variables are missing
        """
        # Check required variables
        missing = [var for var in self.required_vars if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Set defaults for optional variables
        for var in self.optional_vars:
            if var not in kwargs:
                kwargs[var] = ""

        return self.template.format(**kwargs)

    def validate(self, **kwargs) -> list[str]:
        """Validate that all required variables are provided.

        Args:
            **kwargs: Variable values

        Returns:
            List of missing variable names
        """
        return [var for var in self.required_vars if var not in kwargs]


# Pre-defined prompt templates
PROMPTS = {
    "analyze_codebase": PromptTemplate(
        name="analyze_codebase",
        description="Analyze a codebase to identify testable surfaces",
        template="""Analyze this codebase and identify all testable surfaces.

CODEBASE PATH: {codebase_path}
APP URL: {app_url}
{changed_files_section}
{file_contents_section}

Identify:
1. User-facing pages/routes with their URLs
2. API endpoints with methods and expected responses
3. Authentication/authorization flows
4. Critical user journeys (signup, login, checkout, etc.)
5. Database operations that need validation

Respond with JSON:
{{
    "summary": "Brief description of the application",
    "framework": "detected framework",
    "testable_surfaces": [
        {{
            "type": "ui|api|db",
            "name": "descriptive name",
            "path": "URL or endpoint path",
            "priority": "critical|high|medium|low",
            "description": "what this does",
            "test_scenarios": ["scenario 1", "scenario 2"]
        }}
    ]
}}""",
        required_vars=["codebase_path", "app_url"],
        optional_vars=["changed_files_section", "file_contents_section"],
    ),
    "plan_tests": PromptTemplate(
        name="plan_tests",
        description="Create test specifications from testable surfaces",
        template="""Create comprehensive E2E test specifications for these testable surfaces.

APP URL: {app_url}
{codebase_summary_section}

TESTABLE SURFACES:
{surfaces_json}

{changed_files_section}

Generate up to {max_tests} tests per surface.
Focus on critical paths and high-impact scenarios.

Respond with JSON:
{{
    "tests": [
        {{
            "id": "unique-test-id",
            "name": "Descriptive Test Name",
            "type": "ui|api|db",
            "priority": "critical|high|medium|low",
            "description": "What this test validates",
            "preconditions": ["precondition 1"],
            "steps": [
                {{"action": "goto", "target": "/path"}},
                {{"action": "fill", "target": "#selector", "value": "text"}},
                {{"action": "click", "target": "button"}}
            ],
            "assertions": [
                {{"type": "element_visible", "target": ".selector"}},
                {{"type": "url_matches", "expected": "/expected-path"}}
            ],
            "cleanup": [],
            "tags": ["tag1", "tag2"],
            "estimated_duration_ms": 15000
        }}
    ]
}}""",
        required_vars=["app_url", "surfaces_json"],
        optional_vars=["codebase_summary_section", "changed_files_section", "max_tests"],
    ),
    "verify_screenshot": PromptTemplate(
        name="verify_screenshot",
        description="Visually verify a screenshot matches expectations",
        template="""Verify this screenshot matches the expected state.

VERIFICATION TASK:
{verification_task}

EXPECTED ELEMENTS:
{expected_elements}

Look for:
1. Presence of expected elements
2. Correct text content
3. Proper layout and positioning
4. Any error messages or unexpected states

Respond with JSON:
{{
    "passed": true|false,
    "observations": ["what you see"],
    "issues": ["any problems found"],
    "confidence": 0.0-1.0
}}""",
        required_vars=["verification_task"],
        optional_vars=["expected_elements"],
    ),
    "diagnose_failure": PromptTemplate(
        name="diagnose_failure",
        description="Diagnose a test failure and suggest fixes",
        template="""Analyze this test failure and suggest how to fix it.

TEST SPECIFICATION:
{test_spec_json}

FAILURE DETAILS:
{failure_json}

{error_logs_section}

Determine the root cause:
1. SELECTOR_CHANGED - Element moved or was renamed
2. TIMING_ISSUE - Element not ready in time
3. UI_CHANGED - Intentional UI change
4. DATA_CHANGED - Test data no longer valid
5. REAL_BUG - Actual application bug

Respond with JSON:
{{
    "diagnosis": {{
        "failure_type": "selector_changed|timing_issue|ui_changed|data_changed|real_bug|unknown",
        "confidence": 0.0-1.0,
        "explanation": "Why you think this is the cause",
        "affected_step": 0,
        "evidence": ["evidence 1", "evidence 2"]
    }},
    "fixes": [
        {{
            "fix_type": "update_selector|add_wait|increase_timeout|update_assertion|update_test_data|none",
            "old_value": "current value",
            "new_value": "suggested new value",
            "confidence": 0.0-1.0,
            "explanation": "Why this fix should work"
        }}
    ]
}}""",
        required_vars=["test_spec_json", "failure_json"],
        optional_vars=["error_logs_section"],
    ),
    "generate_report": PromptTemplate(
        name="generate_report",
        description="Generate test execution recommendations",
        template="""Analyze these test results and provide actionable recommendations.

SUMMARY:
- Total: {total_tests}, Passed: {passed}, Failed: {failed}
- Pass Rate: {pass_rate}%

FAILURES:
{failures_json}

Provide 3-5 specific, actionable recommendations to improve test reliability.
Focus on:
1. Patterns in failures
2. Test stability improvements
3. Coverage gaps
4. Performance optimizations

Respond with JSON array of strings: ["recommendation 1", "recommendation 2", ...]""",
        required_vars=["total_tests", "passed", "failed", "pass_rate", "failures_json"],
    ),
    "execute_test": PromptTemplate(
        name="execute_test",
        description="Instructions for executing a test with Computer Use",
        template="""Execute this test autonomously using browser automation.

TEST: {test_name}
APP URL: {app_url}

STEPS:
{steps_text}

ASSERTIONS TO VERIFY:
{assertions_text}

CRITICAL INSTRUCTIONS:
1. After each action, verify the result before proceeding
2. If an element is not found, wait up to 10 seconds before failing
3. If something unexpected happens, document it and continue if possible
4. Take screenshots at key checkpoints
5. Report exactly what you observed vs what was expected

After execution, report your findings in JSON:
{{
    "status": "passed|failed",
    "steps_executed": [{{"step": 1, "action": "...", "result": "success|failed"}}],
    "assertions_checked": [{{"assertion": "...", "passed": true|false}}],
    "error_message": null|"description of failure",
    "observations": ["notable observations"]
}}""",
        required_vars=["test_name", "app_url", "steps_text", "assertions_text"],
    ),
}


def get_prompt(name: str, **kwargs) -> str:
    """Get a rendered prompt by name.

    Args:
        name: Prompt template name
        **kwargs: Variables to substitute

    Returns:
        Rendered prompt string

    Raises:
        KeyError: If prompt name not found
        ValueError: If required variables missing
    """
    if name not in PROMPTS:
        raise KeyError(f"Unknown prompt template: {name}")

    return PROMPTS[name].render(**kwargs)


def list_prompts() -> list[dict]:
    """List all available prompt templates.

    Returns:
        List of prompt info dicts
    """
    return [
        {
            "name": p.name,
            "description": p.description,
            "required_vars": p.required_vars,
            "optional_vars": p.optional_vars,
        }
        for p in PROMPTS.values()
    ]


class PromptBuilder:
    """Builder for constructing complex prompts.

    Usage:
        builder = PromptBuilder()
        prompt = (builder
            .add_context("You are a testing agent")
            .add_section("TASK", "Execute this test")
            .add_json_block(test_spec)
            .add_instructions(["Step 1", "Step 2"])
            .build())
    """

    def __init__(self):
        self._parts: list[str] = []

    def add_context(self, context: str) -> "PromptBuilder":
        """Add context/system instruction."""
        self._parts.append(context)
        return self

    def add_section(self, title: str, content: str) -> "PromptBuilder":
        """Add a titled section."""
        self._parts.append(f"\n{title}:\n{content}")
        return self

    def add_json_block(self, data: Any, title: Optional[str] = None) -> "PromptBuilder":
        """Add a JSON code block."""
        import json

        json_str = json.dumps(data, indent=2)
        if title:
            self._parts.append(f"\n{title}:")
        self._parts.append(f"```json\n{json_str}\n```")
        return self

    def add_instructions(self, instructions: list[str]) -> "PromptBuilder":
        """Add numbered instructions."""
        numbered = "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(instructions))
        self._parts.append(f"\nINSTRUCTIONS:\n{numbered}")
        return self

    def add_list(self, title: str, items: list[str]) -> "PromptBuilder":
        """Add a bulleted list."""
        bullets = "\n".join(f"- {item}" for item in items)
        self._parts.append(f"\n{title}:\n{bullets}")
        return self

    def add_response_format(self, format_example: str) -> "PromptBuilder":
        """Add expected response format."""
        self._parts.append(f"\nRespond with JSON:\n{format_example}")
        return self

    def build(self) -> str:
        """Build the final prompt."""
        return "\n".join(self._parts)

    def clear(self) -> "PromptBuilder":
        """Clear all parts."""
        self._parts.clear()
        return self
