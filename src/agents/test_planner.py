"""Test Planner Agent - Creates prioritized test plans from testable surfaces.

This agent generates comprehensive test specifications including:
- Step-by-step test actions
- Assertions and validations
- Test data and preconditions
- Priority ordering
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from .base import BaseAgent, AgentResult
from .code_analyzer import TestableSurface


@dataclass
class TestStep:
    """A single step in a test."""

    action: str  # goto, click, fill, assert, wait, screenshot, etc.
    target: Optional[str] = None  # selector, URL, or element identifier
    value: Optional[str] = None  # value to input or expected value
    timeout: int = 5000
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "target": self.target,
            "value": self.value,
            "timeout": self.timeout,
            "description": self.description,
        }


@dataclass
class TestAssertion:
    """An assertion to validate test results."""

    type: str  # element_visible, text_contains, url_matches, value_equals, etc.
    target: Optional[str] = None
    expected: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "target": self.target,
            "expected": self.expected,
            "description": self.description,
        }


@dataclass
class TestSpec:
    """Complete test specification."""

    id: str
    name: str
    type: str  # ui, api, db
    priority: str
    description: str
    preconditions: list[str] = field(default_factory=list)
    steps: list[TestStep] = field(default_factory=list)
    assertions: list[TestAssertion] = field(default_factory=list)
    cleanup: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    estimated_duration_ms: int = 30000
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "priority": self.priority,
            "description": self.description,
            "preconditions": self.preconditions,
            "steps": [s.to_dict() for s in self.steps],
            "assertions": [a.to_dict() for a in self.assertions],
            "cleanup": self.cleanup,
            "tags": self.tags,
            "estimated_duration_ms": self.estimated_duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class TestPlan:
    """A complete test plan with ordered test specs."""

    tests: list[TestSpec]
    total_estimated_duration_ms: int = 0
    coverage_summary: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.total_estimated_duration_ms:
            self.total_estimated_duration_ms = sum(
                t.estimated_duration_ms for t in self.tests
            )


class TestPlannerAgent(BaseAgent):
    """Agent that creates detailed test plans from testable surfaces.

    Generates:
    - Complete test specifications with steps and assertions
    - Priority ordering based on criticality
    - Test data and preconditions
    - Estimated durations
    """

    def _get_system_prompt(self) -> str:
        return """You are an expert test architect specializing in E2E test planning.

Your task is to create comprehensive, executable test specifications that cover:
1. Happy path scenarios
2. Error handling and edge cases
3. Input validation
4. Authentication/authorization checks
5. Data integrity validations

For each test, provide:
- Clear, unique ID and descriptive name
- Step-by-step actions using these verbs: goto, click, fill, select, hover, wait, screenshot, scroll, press_key
- Specific assertions with element selectors and expected values
- Preconditions and cleanup steps
- Realistic time estimates

Use CSS selectors that are:
- Specific enough to uniquely identify elements
- Robust against minor UI changes (prefer data-testid, id, role over classes)
- Readable and maintainable

Output must be valid JSON."""

    async def execute(
        self,
        testable_surfaces: list[TestableSurface | dict],
        app_url: str,
        codebase_summary: str = "",
        changed_files: Optional[list[str]] = None,
        max_tests_per_surface: int = 3,
    ) -> AgentResult[TestPlan]:
        """Generate a test plan from testable surfaces.

        Args:
            testable_surfaces: List of surfaces to generate tests for
            app_url: Application base URL
            codebase_summary: Summary of the codebase
            changed_files: Files that changed (prioritize related tests)
            max_tests_per_surface: Maximum tests to generate per surface

        Returns:
            AgentResult containing TestPlan
        """
        self.log.info(
            "Creating test plan",
            surfaces_count=len(testable_surfaces),
            app_url=app_url,
        )

        if not self._check_cost_limit():
            return AgentResult(
                success=False,
                error="Cost limit exceeded before planning",
            )

        # Convert to dicts if needed
        surfaces_data = []
        for surface in testable_surfaces:
            if isinstance(surface, TestableSurface):
                surfaces_data.append(surface.to_dict())
            else:
                surfaces_data.append(surface)

        prompt = self._build_planning_prompt(
            surfaces_data,
            app_url,
            codebase_summary,
            changed_files,
            max_tests_per_surface,
        )

        try:
            response = self._call_claude(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
            )

            content = self._extract_text_response(response)
            result_data = self._parse_json_response(content)

            if not result_data:
                return AgentResult(
                    success=False,
                    error="Failed to parse test plan response",
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

            # Parse test specs
            tests = []
            # Handle both list and dict responses
            if isinstance(result_data, list):
                test_items = result_data
                coverage_summary = {}
            else:
                test_items = result_data.get("tests", [])
                coverage_summary = result_data.get("coverage_summary", {})

            for test_data in test_items:
                tests.append(self._parse_test_spec(test_data))

            # Sort by priority
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            tests.sort(key=lambda t: priority_order.get(t.priority, 4))

            test_plan = TestPlan(
                tests=tests,
                coverage_summary=coverage_summary,
            )

            self.log.info(
                "Test plan created",
                test_count=len(tests),
                estimated_duration_ms=test_plan.total_estimated_duration_ms,
            )

            return AgentResult(
                success=True,
                data=test_plan,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost=self._usage.total_cost,
            )

        except Exception as e:
            self.log.error("Test planning failed", error=str(e))
            return AgentResult(
                success=False,
                error=f"Planning failed: {str(e)}",
            )

    def _build_planning_prompt(
        self,
        surfaces: list[dict],
        app_url: str,
        codebase_summary: str,
        changed_files: Optional[list[str]],
        max_tests_per_surface: int,
    ) -> str:
        """Build the test planning prompt."""
        prompt_parts = [
            "Create comprehensive E2E test specifications for these testable surfaces.",
            "",
            f"APP URL: {app_url}",
        ]

        if codebase_summary:
            prompt_parts.extend([
                "",
                "CODEBASE SUMMARY:",
                codebase_summary,
            ])

        prompt_parts.extend([
            "",
            "TESTABLE SURFACES:",
            json.dumps(surfaces, indent=2),
        ])

        if changed_files:
            prompt_parts.extend([
                "",
                "CHANGED FILES (prioritize related tests):",
                json.dumps(changed_files[:10]),
            ])

        prompt_parts.extend([
            "",
            f"Generate up to {max_tests_per_surface} tests per surface.",
            "Focus on critical paths and high-impact scenarios.",
            "",
            "Respond with JSON:",
            """{
    "tests": [
        {
            "id": "unique-test-id",
            "name": "Descriptive Test Name",
            "type": "ui|api|db",
            "priority": "critical|high|medium|low",
            "description": "What this test validates",
            "preconditions": ["User must be logged out"],
            "steps": [
                {"action": "goto", "target": "/login"},
                {"action": "fill", "target": "#email", "value": "test@example.com"},
                {"action": "click", "target": "button[type=submit]"}
            ],
            "assertions": [
                {"type": "url_matches", "expected": "/dashboard"},
                {"type": "element_visible", "target": ".welcome-message"}
            ],
            "cleanup": ["Logout user"],
            "tags": ["auth", "critical-path"],
            "estimated_duration_ms": 15000
        }
    ],
    "coverage_summary": {
        "total_surfaces": 5,
        "covered_surfaces": 5,
        "ui_tests": 8,
        "api_tests": 3,
        "db_tests": 1
    }
}""",
        ])

        return "\n".join(prompt_parts)

    def _parse_test_spec(self, data: dict) -> TestSpec:
        """Parse a test spec from JSON data."""
        steps = []
        for step_data in data.get("steps", []):
            steps.append(
                TestStep(
                    action=step_data.get("action", ""),
                    target=step_data.get("target"),
                    value=step_data.get("value"),
                    timeout=step_data.get("timeout", 5000),
                    description=step_data.get("description"),
                )
            )

        assertions = []
        for assertion_data in data.get("assertions", []):
            assertions.append(
                TestAssertion(
                    type=assertion_data.get("type", ""),
                    target=assertion_data.get("target"),
                    expected=assertion_data.get("expected"),
                    description=assertion_data.get("description"),
                )
            )

        return TestSpec(
            id=data.get("id", f"test-{len(steps)}"),
            name=data.get("name", "Unnamed Test"),
            type=data.get("type", "ui"),
            priority=data.get("priority", "medium"),
            description=data.get("description", ""),
            preconditions=data.get("preconditions", []),
            steps=steps,
            assertions=assertions,
            cleanup=data.get("cleanup", []),
            tags=data.get("tags", []),
            estimated_duration_ms=data.get("estimated_duration_ms", 30000),
            metadata=data.get("metadata", {}),
        )

    async def generate_api_tests(
        self,
        endpoints: list[dict],
        app_url: str,
    ) -> AgentResult[TestPlan]:
        """Generate API-specific tests.

        Args:
            endpoints: List of API endpoint definitions
            app_url: Base API URL

        Returns:
            AgentResult containing TestPlan for API tests
        """
        # Convert endpoints to testable surfaces
        surfaces = []
        for endpoint in endpoints:
            surfaces.append({
                "type": "api",
                "name": f"API: {endpoint.get('method', 'GET')} {endpoint.get('path', '/')}",
                "path": endpoint.get("path", "/"),
                "priority": endpoint.get("priority", "high"),
                "description": endpoint.get("description", ""),
                "test_scenarios": [
                    "Valid request with correct data",
                    "Invalid request with missing fields",
                    "Unauthorized request without token",
                    "Request with invalid data types",
                ],
                "metadata": endpoint,
            })

        return await self.execute(
            testable_surfaces=surfaces,
            app_url=app_url,
            max_tests_per_surface=4,
        )
