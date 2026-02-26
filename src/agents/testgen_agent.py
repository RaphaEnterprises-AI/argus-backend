"""
TestGen Orchestrator Agent - End-to-end test generation pipeline.

Composes existing agents (NLPTestCreator, TestImpactAnalyzer, AgentAsJudge)
and adds net-new logic for:
- Requirements analysis with ambiguity detection
- BDD/Gherkin scenario generation
- Framework-specific code emission (Playwright, Cypress, Selenium, Pytest, Robot)
- Quality audit via Agent-as-Judge with refinement loop
- Requirement-to-test traceability (coverage matrix)
- Cognee knowledge graph storage for cross-session learning
- Kafka event emission for downstream processing

RAP-350: TestGen Orchestrator Agent

Architecture:
    Requirements Input
         |
         v
    [Cognee: query past patterns] --> fragile selectors, similar requirements
         |
         v
    [analyze_requirements] --> ParsedRequirements with ambiguity scores
         |
         v
    [_generate_bdd_scenarios] --> Gherkin BDD scenarios
         |
         v
    [_generate_test_code] --> delegates to code_emitters module
         |
         v
    [_quality_audit] --> AgentAsJudge + refinement loop
         |
         v
    [_build_coverage_matrix] --> requirement -> test traceability
         |
         v
    [_store_in_cognee] --> knowledge graph for future learning
         |
         v
    [_emit_events] --> Kafka: TEST_GENERATED, REQUIREMENTS_ANALYZED
         |
         v
    TestGenResult
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from .agent_judge import AgentAsJudge, AgentEvaluation
from .base import (
    AgentCapability,
    AgentResult,
    AICapability,
    BaseAgent,
)
from ..core.model_router import TaskType

logger = structlog.get_logger()


# =============================================================================
# Data Classes
# =============================================================================


class SourceType(str, Enum):
    """Source types for test generation input."""

    REQUIREMENTS_TEXT = "requirements_text"
    JIRA = "jira"
    GITHUB_PR = "github_pr"
    CODEBASE_ANALYSIS = "codebase_analysis"
    USER_STORY = "user_story"
    OPENAPI_SPEC = "openapi_spec"


class TestFramework(str, Enum):
    """Supported test frameworks."""

    PLAYWRIGHT = "playwright"
    CYPRESS = "cypress"
    SELENIUM = "selenium"
    PYTEST = "pytest"
    ROBOT = "robot"


class TestLanguage(str, Enum):
    """Supported programming languages for test output."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"


class CoverageType(str, Enum):
    """Types of test coverage."""

    FUNCTIONAL = "functional"
    NEGATIVE = "negative"
    EDGE_CASE = "edge_case"
    BOUNDARY = "boundary"
    INTEGRATION = "integration"
    SECURITY = "security"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"


class CodeEntityType(str, Enum):
    """Types of code entities that tests cover."""

    ENDPOINT = "endpoint"
    FUNCTION = "function"
    CLASS = "class"
    UI_COMPONENT = "ui_component"
    USER_FLOW = "user_flow"
    DATABASE_TABLE = "database_table"


@dataclass
class TestGenConfig:
    """Configuration for test generation."""

    framework: TestFramework = TestFramework.PLAYWRIGHT
    language: TestLanguage = TestLanguage.PYTHON
    include_negative_tests: bool = True
    include_edge_cases: bool = True
    include_accessibility: bool = False
    max_tests: int = 50
    quality_threshold: float = 0.75
    max_refinement_rounds: int = 2
    bdd_style: bool = True
    app_url: str | None = None
    selector_strategy: str = "semantic"  # semantic, data-testid, css, xpath

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework.value,
            "language": self.language.value,
            "include_negative_tests": self.include_negative_tests,
            "include_edge_cases": self.include_edge_cases,
            "include_accessibility": self.include_accessibility,
            "max_tests": self.max_tests,
            "quality_threshold": self.quality_threshold,
            "max_refinement_rounds": self.max_refinement_rounds,
            "bdd_style": self.bdd_style,
            "app_url": self.app_url,
            "selector_strategy": self.selector_strategy,
        }


@dataclass
class ParsedRequirement:
    """A single requirement parsed from source input."""

    id: str
    title: str
    description: str
    acceptance_criteria: list[str] = field(default_factory=list)
    ambiguity_score: float = 0.0  # 0.0 = crystal clear, 1.0 = completely ambiguous
    ambiguity_details: list[str] = field(default_factory=list)
    priority: str = "medium"  # critical, high, medium, low
    tags: list[str] = field(default_factory=list)
    source_ref: str | None = None  # JIRA ticket, PR number, etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "acceptance_criteria": self.acceptance_criteria,
            "ambiguity_score": self.ambiguity_score,
            "ambiguity_details": self.ambiguity_details,
            "priority": self.priority,
            "tags": self.tags,
            "source_ref": self.source_ref,
        }


@dataclass
class BDDScenario:
    """A BDD/Gherkin scenario derived from a requirement."""

    name: str
    feature: str
    given: list[str] = field(default_factory=list)
    when: list[str] = field(default_factory=list)
    then: list[str] = field(default_factory=list)
    and_steps: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    requirement_id: str = ""
    is_negative: bool = False
    is_edge_case: bool = False

    def to_gherkin(self) -> str:
        """Render this scenario as Gherkin text."""
        lines: list[str] = []
        if self.tags:
            lines.append("  " + " ".join(f"@{t}" for t in self.tags))
        lines.append(f"  Scenario: {self.name}")
        for step in self.given:
            lines.append(f"    Given {step}")
        for step in self.when:
            lines.append(f"    When {step}")
        for step in self.then:
            lines.append(f"    Then {step}")
        for step in self.and_steps:
            lines.append(f"    And {step}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature": self.feature,
            "given": self.given,
            "when": self.when,
            "then": self.then,
            "and_steps": self.and_steps,
            "tags": self.tags,
            "requirement_id": self.requirement_id,
            "is_negative": self.is_negative,
            "is_edge_case": self.is_edge_case,
            "gherkin": self.to_gherkin(),
        }


@dataclass
class GeneratedTestCase:
    """A generated test case with executable code."""

    name: str
    code: str
    framework: str
    language: str
    bdd_scenario: BDDScenario | None = None
    requirement_ids: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    quality_issues: list[str] = field(default_factory=list)
    estimated_duration_seconds: int = 30
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "code": self.code,
            "framework": self.framework,
            "language": self.language,
            "bdd_scenario": self.bdd_scenario.to_dict() if self.bdd_scenario else None,
            "requirement_ids": self.requirement_ids,
            "quality_score": self.quality_score,
            "quality_issues": self.quality_issues,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "tags": self.tags,
        }


@dataclass
class CoverageMatrixEntry:
    """A single entry in the requirement-to-test traceability matrix."""

    requirement_id: str
    requirement_title: str
    code_entity: str
    code_entity_type: CodeEntityType
    test_name: str
    test_type: str  # ui, api, unit, integration
    coverage_type: CoverageType
    confidence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "requirement_title": self.requirement_title,
            "code_entity": self.code_entity,
            "code_entity_type": self.code_entity_type.value,
            "test_name": self.test_name,
            "test_type": self.test_type,
            "coverage_type": self.coverage_type.value,
            "confidence": self.confidence,
        }


@dataclass
class TestGenResult:
    """Complete result of a test generation run."""

    job_id: str
    tests: list[GeneratedTestCase] = field(default_factory=list)
    requirements: list[ParsedRequirement] = field(default_factory=list)
    scenarios: list[BDDScenario] = field(default_factory=list)
    coverage_matrix: list[CoverageMatrixEntry] = field(default_factory=list)
    quality_score: float = 0.0
    ai_cost: float = 0.0
    duration_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    cognee_namespaces: list[str] = field(default_factory=list)
    events_emitted: list[str] = field(default_factory=list)

    @property
    def tests_count(self) -> int:
        return len(self.tests)

    @property
    def requirements_count(self) -> int:
        return len(self.requirements)

    @property
    def coverage_entries_count(self) -> int:
        return len(self.coverage_matrix)

    @property
    def avg_ambiguity(self) -> float:
        if not self.requirements:
            return 0.0
        return sum(r.ambiguity_score for r in self.requirements) / len(self.requirements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tests": [t.to_dict() for t in self.tests],
            "requirements": [r.to_dict() for r in self.requirements],
            "scenarios": [s.to_dict() for s in self.scenarios],
            "coverage_matrix": [c.to_dict() for c in self.coverage_matrix],
            "quality_score": self.quality_score,
            "ai_cost": self.ai_cost,
            "duration_ms": self.duration_ms,
            "warnings": self.warnings,
            "cognee_namespaces": self.cognee_namespaces,
            "events_emitted": self.events_emitted,
            "summary": {
                "tests_count": self.tests_count,
                "requirements_count": self.requirements_count,
                "coverage_entries_count": self.coverage_entries_count,
                "avg_ambiguity": round(self.avg_ambiguity, 3),
            },
        }


# =============================================================================
# TestGenAgent
# =============================================================================


class TestGenAgent(BaseAgent):
    """
    Test Generation Orchestrator Agent.

    Composes NLPTestCreator, TestImpactAnalyzer, and AgentAsJudge to provide
    a complete test generation pipeline:

    1. Parse requirements (detect ambiguity, extract acceptance criteria)
    2. Query Cognee for fragile selector patterns and similar past requirements
    3. Generate BDD/Gherkin scenarios from parsed requirements
    4. Emit framework-specific test code (Playwright, Cypress, Selenium, etc.)
    5. Run quality audit via AgentAsJudge with refinement loop
    6. Build requirement-to-test traceability matrix
    7. Store all artifacts in Cognee for cross-session learning
    8. Emit Kafka events for downstream processing

    Usage:
        ```python
        agent = TestGenAgent()

        result = await agent.generate(
            org_id="org_acme",
            project_id="proj_webapp",
            source_type=SourceType.REQUIREMENTS_TEXT,
            source_data="As a user, I want to login with my email...",
            config=TestGenConfig(framework=TestFramework.PLAYWRIGHT),
        )

        print(f"Generated {result.tests_count} tests")
        print(f"Quality: {result.quality_score:.2f}")
        print(f"AI cost: ${result.ai_cost:.4f}")
        ```
    """

    CAPABILITIES = [
        AgentCapability.TEST_GENERATION,
        AgentCapability.TEST_PLANNING,
        AgentCapability.CODE_ANALYSIS,
    ]

    DEFAULT_TASK_TYPE = TaskType.CODE_ANALYSIS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log = logger.bind(agent="TestGenAgent")
        self._judge: AgentAsJudge | None = None

    def _get_system_prompt(self) -> str:
        return """You are an expert QA automation architect specializing in end-to-end test generation.

Your responsibilities:
1. REQUIREMENTS ANALYSIS: Parse natural language, JIRA tickets, PRs, or OpenAPI specs
   into structured, testable requirements. Detect ambiguity and flag unclear acceptance criteria.

2. BDD SCENARIO DESIGN: Create comprehensive Gherkin/BDD scenarios that cover:
   - Happy paths (primary user flows)
   - Negative paths (error handling, invalid inputs, unauthorized access)
   - Edge cases (boundary values, empty states, concurrent operations)
   - Data variations (different user roles, locales, data volumes)

3. TEST CODE GENERATION: Produce clean, maintainable, framework-specific test code.
   Follow best practices:
   - Use semantic selectors (text, labels, roles) over brittle CSS/XPath
   - Include proper waits and assertions
   - Parameterize test data
   - Add clear comments explaining test intent
   - Follow the Page Object Model when appropriate

4. QUALITY ASSURANCE: Ensure generated tests are:
   - Deterministic (no flaky patterns)
   - Independent (no test ordering dependencies)
   - Maintainable (DRY, well-named, documented)
   - Complete (cover all acceptance criteria)

5. TRACEABILITY: Every test must trace back to at least one requirement.
   Every requirement must be covered by at least one test.

When generating tests, AVOID these known fragile patterns:
- Absolute XPath expressions
- Index-based selectors (nth-child without semantic context)
- Time-sensitive assertions without proper waits
- Hard-coded test data without parameterization
- Selectors based on auto-generated class names

ALWAYS prefer:
- data-testid attributes
- ARIA roles and labels
- Visible text content
- Semantic HTML structure"""

    @property
    def judge(self) -> AgentAsJudge:
        """Lazy-initialize the AgentAsJudge for quality audits."""
        if self._judge is None:
            self._judge = AgentAsJudge(
                enable_execution=False,
                enable_ground_truth=True,
            )
        return self._judge

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the test generation pipeline.

        Expected kwargs:
            org_id: Organization ID
            project_id: Project ID
            source_type: SourceType enum or string
            source_data: Raw input (requirements text, JIRA payload, etc.)
            config: Optional TestGenConfig
            job_id: Optional job ID for tracking

        Returns:
            AgentResult with TestGenResult data
        """
        org_id = kwargs.get("org_id", "")
        project_id = kwargs.get("project_id", "")
        source_type_raw = kwargs.get("source_type", SourceType.REQUIREMENTS_TEXT)
        source_data = kwargs.get("source_data", "")
        config = kwargs.get("config") or TestGenConfig()
        job_id = kwargs.get("job_id") or f"testgen_{uuid4().hex[:12]}"

        if isinstance(source_type_raw, str):
            try:
                source_type = SourceType(source_type_raw)
            except ValueError:
                source_type = SourceType.REQUIREMENTS_TEXT
        else:
            source_type = source_type_raw

        try:
            result = await self.generate(
                org_id=org_id,
                project_id=project_id,
                source_type=source_type,
                source_data=source_data,
                config=config,
                job_id=job_id,
            )
            return AgentResult(
                success=True,
                data=result.to_dict(),
                input_tokens=self._usage.total_input_tokens,
                output_tokens=self._usage.total_output_tokens,
                cost=self._usage.total_cost,
            )
        except Exception as e:
            self.log.error("TestGen pipeline failed", error=str(e), job_id=job_id)
            return AgentResult(
                success=False,
                error=str(e),
                input_tokens=self._usage.total_input_tokens,
                output_tokens=self._usage.total_output_tokens,
                cost=self._usage.total_cost,
            )

    async def generate(
        self,
        org_id: str,
        project_id: str,
        source_type: SourceType,
        source_data: str | dict,
        config: TestGenConfig | None = None,
        job_id: str | None = None,
    ) -> TestGenResult:
        """Main test generation pipeline.

        Orchestrates the full flow: parse requirements, query Cognee for patterns,
        generate BDD scenarios, emit code, quality audit, build coverage matrix,
        store in Cognee, emit Kafka events.

        Args:
            org_id: Organization ID for multi-tenant isolation
            project_id: Project ID for scoping
            source_type: Type of input source
            source_data: Raw input data (text, dict, etc.)
            config: Generation configuration
            job_id: Optional job tracking ID

        Returns:
            TestGenResult with all generated artifacts
        """
        start_time = time.time()
        config = config or TestGenConfig()
        job_id = job_id or f"testgen_{uuid4().hex[:12]}"

        self.log.info(
            "Starting test generation pipeline",
            job_id=job_id,
            org_id=org_id,
            project_id=project_id,
            source_type=source_type.value,
            framework=config.framework.value,
        )

        result = TestGenResult(job_id=job_id)

        # =====================================================================
        # Step 1: Query Cognee for historical patterns BEFORE generating
        # =====================================================================
        cognee_context = await self._query_cognee_context(org_id, project_id)

        # =====================================================================
        # Step 2: Parse and analyze requirements
        # =====================================================================
        requirements_text = self._extract_requirements_text(source_type, source_data)
        requirements = await self.analyze_requirements(
            requirements_text=requirements_text,
            source_type=source_type,
            cognee_context=cognee_context,
        )
        result.requirements = requirements

        if not requirements:
            result.warnings.append("No requirements could be parsed from the input.")
            result.duration_ms = int((time.time() - start_time) * 1000)
            return result

        self.log.info(
            "Requirements parsed",
            count=len(requirements),
            avg_ambiguity=round(
                sum(r.ambiguity_score for r in requirements) / len(requirements), 3
            ),
        )

        # Warn about highly ambiguous requirements
        ambiguous = [r for r in requirements if r.ambiguity_score > 0.7]
        if ambiguous:
            for req in ambiguous:
                result.warnings.append(
                    f"Requirement '{req.title}' has high ambiguity "
                    f"(score={req.ambiguity_score:.2f}): {'; '.join(req.ambiguity_details)}"
                )

        # =====================================================================
        # Step 3: Generate BDD scenarios from requirements
        # =====================================================================
        scenarios = await self._generate_bdd_scenarios(
            requirements=requirements,
            config=config,
            cognee_context=cognee_context,
        )
        result.scenarios = scenarios

        if not scenarios:
            result.warnings.append("No BDD scenarios could be generated.")
            result.duration_ms = int((time.time() - start_time) * 1000)
            result.ai_cost = self._usage.total_cost
            return result

        self.log.info("BDD scenarios generated", count=len(scenarios))

        # =====================================================================
        # Step 4: Generate framework-specific test code
        # =====================================================================
        tests = await self._generate_test_code(
            scenarios=scenarios,
            framework=config.framework,
            language=config.language,
            config=config,
            cognee_context=cognee_context,
        )
        result.tests = tests

        if not tests:
            result.warnings.append("No test code could be generated from scenarios.")
            result.duration_ms = int((time.time() - start_time) * 1000)
            result.ai_cost = self._usage.total_cost
            return result

        self.log.info(
            "Test code generated",
            count=len(tests),
            framework=config.framework.value,
        )

        # =====================================================================
        # Step 5: Quality audit with AgentAsJudge + refinement loop
        # =====================================================================
        tests, quality_score = await self._quality_audit(
            generated_tests=tests,
            requirements=requirements,
            config=config,
        )
        result.tests = tests
        result.quality_score = quality_score

        self.log.info(
            "Quality audit complete",
            quality_score=round(quality_score, 3),
            tests_after_audit=len(tests),
        )

        # =====================================================================
        # Step 6: Build coverage matrix
        # =====================================================================
        coverage_matrix = self._build_coverage_matrix(
            requirements=requirements,
            generated_tests=tests,
        )
        result.coverage_matrix = coverage_matrix

        # Check for uncovered requirements
        covered_req_ids = {e.requirement_id for e in coverage_matrix}
        uncovered = [r for r in requirements if r.id not in covered_req_ids]
        if uncovered:
            for req in uncovered:
                result.warnings.append(
                    f"Requirement '{req.title}' (id={req.id}) has no test coverage."
                )

        # =====================================================================
        # Step 7: Store in Cognee for learning
        # =====================================================================
        cognee_namespaces = await self._store_in_cognee(
            org_id=org_id,
            project_id=project_id,
            requirements=requirements,
            tests=tests,
            coverage_matrix=coverage_matrix,
            job_id=job_id,
        )
        result.cognee_namespaces = cognee_namespaces

        # =====================================================================
        # Step 8: Emit Kafka events
        # =====================================================================
        emitted = await self._emit_events(
            org_id=org_id,
            project_id=project_id,
            job_id=job_id,
            results=result,
            source_type=source_type,
            config=config,
        )
        result.events_emitted = emitted

        # Finalize
        result.ai_cost = self._usage.total_cost
        result.duration_ms = int((time.time() - start_time) * 1000)

        self.log.info(
            "Test generation pipeline complete",
            job_id=job_id,
            tests=result.tests_count,
            requirements=result.requirements_count,
            coverage_entries=result.coverage_entries_count,
            quality=round(result.quality_score, 3),
            cost=f"${result.ai_cost:.4f}",
            duration_ms=result.duration_ms,
        )

        return result

    # =========================================================================
    # Step 1: Cognee Context Query
    # =========================================================================

    async def _query_cognee_context(
        self,
        org_id: str,
        project_id: str,
    ) -> dict[str, Any]:
        """Query Cognee for historical patterns before generating tests.

        Searches three namespaces:
        - selector_fragility: Known fragile selector patterns to avoid
        - requirements: Similar past requirements for context
        - healing_patterns: Which test patterns tend to break and need healing

        Args:
            org_id: Organization ID
            project_id: Project ID

        Returns:
            Dict with keys: fragile_selectors, similar_requirements, healing_patterns
        """
        context: dict[str, Any] = {
            "fragile_selectors": [],
            "similar_requirements": [],
            "healing_patterns": [],
        }

        try:
            from ..knowledge.cognee_client import get_cognee_client

            cognee = get_cognee_client(org_id=org_id, project_id=project_id)

            # Search for fragile selector patterns to avoid
            try:
                fragile = await cognee.search(
                    namespace=("patterns", "selector_fragility"),
                    query="fragile selector patterns that caused test failures",
                    limit=10,
                    threshold=0.5,
                )
                context["fragile_selectors"] = fragile or []
            except Exception as e:
                self.log.debug(
                    "Cognee selector_fragility search failed (non-fatal)",
                    error=str(e),
                )

            # Search for similar past requirements
            try:
                similar = await cognee.search(
                    namespace=("testgen", "requirements"),
                    query="test generation requirements and acceptance criteria",
                    limit=5,
                    threshold=0.6,
                )
                context["similar_requirements"] = similar or []
            except Exception as e:
                self.log.debug(
                    "Cognee requirements search failed (non-fatal)",
                    error=str(e),
                )

            # Search for healing patterns to understand which test patterns survive
            try:
                healing = await cognee.search(
                    namespace=("patterns", "healing"),
                    query="test patterns that required healing or self-repair",
                    limit=10,
                    threshold=0.5,
                )
                context["healing_patterns"] = healing or []
            except Exception as e:
                self.log.debug(
                    "Cognee healing_patterns search failed (non-fatal)",
                    error=str(e),
                )

            self.log.info(
                "Cognee context loaded",
                fragile_selectors=len(context["fragile_selectors"]),
                similar_requirements=len(context["similar_requirements"]),
                healing_patterns=len(context["healing_patterns"]),
            )

        except Exception as e:
            self.log.warning(
                "Cognee context query failed entirely (non-fatal, proceeding without)",
                error=str(e),
            )

        return context

    # =========================================================================
    # Step 2: Requirements Analysis
    # =========================================================================

    def _extract_requirements_text(
        self,
        source_type: SourceType,
        source_data: str | dict,
    ) -> str:
        """Extract plain-text requirements from various source formats.

        Args:
            source_type: Type of input
            source_data: Raw input data

        Returns:
            Normalized requirements text
        """
        if isinstance(source_data, str):
            return source_data

        if isinstance(source_data, dict):
            if source_type == SourceType.JIRA:
                # Extract from JIRA issue payload
                parts = []
                if source_data.get("summary"):
                    parts.append(f"Title: {source_data['summary']}")
                if source_data.get("description"):
                    parts.append(f"Description: {source_data['description']}")
                if source_data.get("acceptance_criteria"):
                    parts.append(
                        f"Acceptance Criteria:\n{source_data['acceptance_criteria']}"
                    )
                return "\n\n".join(parts) if parts else str(source_data)

            elif source_type == SourceType.GITHUB_PR:
                parts = []
                if source_data.get("title"):
                    parts.append(f"PR Title: {source_data['title']}")
                if source_data.get("body"):
                    parts.append(f"PR Description: {source_data['body']}")
                if source_data.get("changed_files"):
                    files = source_data["changed_files"]
                    if isinstance(files, list):
                        parts.append(
                            f"Changed Files: {', '.join(str(f) for f in files[:20])}"
                        )
                return "\n\n".join(parts) if parts else str(source_data)

            elif source_type == SourceType.OPENAPI_SPEC:
                # Summarize relevant endpoints
                return json.dumps(source_data, indent=2)[:8000]

            else:
                return json.dumps(source_data, indent=2)[:8000]

        return str(source_data)

    async def analyze_requirements(
        self,
        requirements_text: str,
        source_type: SourceType,
        cognee_context: dict[str, Any] | None = None,
    ) -> list[ParsedRequirement]:
        """Parse and analyze requirements for ambiguity.

        Uses AI to:
        - Break input into individual requirements
        - Extract acceptance criteria for each
        - Score ambiguity (vague terms, missing specifics, unclear scope)
        - Assign priority based on content analysis
        - Tag with relevant categories

        Args:
            requirements_text: Raw requirements text
            source_type: Source type for parsing hints
            cognee_context: Optional Cognee context with similar past requirements

        Returns:
            List of ParsedRequirement objects
        """
        if not requirements_text or not requirements_text.strip():
            return []

        similar_context = ""
        if cognee_context and cognee_context.get("similar_requirements"):
            past_reqs = cognee_context["similar_requirements"][:3]
            if past_reqs:
                similar_context = (
                    "\n\nSIMILAR PAST REQUIREMENTS (for context, do not copy):\n"
                    + json.dumps(past_reqs, indent=2, default=str)[:2000]
                )

        prompt = f"""Analyze the following requirements input and break it down into individual,
testable requirements. For each requirement, extract acceptance criteria, assess ambiguity,
and assign priority.

SOURCE TYPE: {source_type.value}

INPUT:
{requirements_text[:6000]}
{similar_context}

AMBIGUITY SCORING GUIDE:
- 0.0-0.2: Crystal clear, specific, measurable
- 0.2-0.4: Mostly clear with minor vagueness
- 0.4-0.6: Some important details missing or unclear
- 0.6-0.8: Significantly ambiguous, multiple interpretations possible
- 0.8-1.0: Extremely vague, needs clarification before testing

Flag these as ambiguity sources:
- Vague terms: "fast", "user-friendly", "responsive", "appropriate", "relevant"
- Missing specifics: No concrete values, thresholds, or examples
- Unclear scope: Doesn't specify which users, pages, or conditions
- Implicit assumptions: Assumes reader knows unstated context
- Conflicting criteria: Acceptance criteria that contradict each other

Return JSON:
{{
    "requirements": [
        {{
            "id": "REQ-001",
            "title": "Short requirement title",
            "description": "Full requirement description",
            "acceptance_criteria": [
                "Criterion 1: specific, measurable condition",
                "Criterion 2: ..."
            ],
            "ambiguity_score": 0.0-1.0,
            "ambiguity_details": [
                "Specific ambiguity issue found"
            ],
            "priority": "critical|high|medium|low",
            "tags": ["authentication", "ui", "api", "data", "security"],
            "source_ref": "Optional reference to source (JIRA-123, PR #45, etc.)"
        }}
    ]
}}"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            required_capabilities=[AICapability.REASONING],
            max_tokens=4096,
            temperature=0.1,
            json_mode=True,
            max_cost=0.10,
        )

        parsed = self._parse_json_response(response.content, {"requirements": []})
        requirements: list[ParsedRequirement] = []

        for raw_req in parsed.get("requirements", []):
            req = ParsedRequirement(
                id=str(raw_req.get("id", f"REQ-{uuid4().hex[:6]}")),
                title=str(raw_req.get("title", "Untitled Requirement")),
                description=str(raw_req.get("description", "")),
                acceptance_criteria=[
                    str(c) for c in raw_req.get("acceptance_criteria", [])
                ],
                ambiguity_score=float(raw_req.get("ambiguity_score", 0.5)),
                ambiguity_details=[
                    str(d) for d in raw_req.get("ambiguity_details", [])
                ],
                priority=str(raw_req.get("priority", "medium")),
                tags=[str(t) for t in raw_req.get("tags", [])],
                source_ref=raw_req.get("source_ref"),
            )
            requirements.append(req)

        return requirements

    # =========================================================================
    # Step 3: BDD Scenario Generation
    # =========================================================================

    async def _generate_bdd_scenarios(
        self,
        requirements: list[ParsedRequirement],
        config: TestGenConfig,
        cognee_context: dict[str, Any] | None = None,
    ) -> list[BDDScenario]:
        """Generate Gherkin BDD scenarios from parsed requirements.

        Creates:
        - Happy path scenarios for each requirement
        - Negative test scenarios (if config.include_negative_tests)
        - Edge case scenarios (if config.include_edge_cases)

        Args:
            requirements: Parsed requirements
            config: Generation configuration
            cognee_context: Cognee context for pattern avoidance

        Returns:
            List of BDDScenario objects
        """
        # Build avoidance hints from Cognee context
        avoidance_hints = ""
        if cognee_context:
            fragile = cognee_context.get("fragile_selectors", [])
            if fragile:
                patterns = []
                for item in fragile[:5]:
                    if isinstance(item, dict):
                        patterns.append(
                            str(item.get("pattern", item.get("value", str(item))))
                        )
                    else:
                        patterns.append(str(item))
                avoidance_hints = (
                    "\n\nAVOID these known fragile patterns from past failures:\n"
                    + "\n".join(f"- {p}" for p in patterns)
                )

            healing = cognee_context.get("healing_patterns", [])
            if healing:
                patterns = []
                for item in healing[:5]:
                    if isinstance(item, dict):
                        patterns.append(
                            str(item.get("summary", item.get("value", str(item))))
                        )
                    else:
                        patterns.append(str(item))
                avoidance_hints += (
                    "\n\nTest patterns that frequently need healing (use with caution):\n"
                    + "\n".join(f"- {p}" for p in patterns)
                )

        # Serialize requirements for prompt
        req_summaries = []
        for req in requirements:
            req_summaries.append({
                "id": req.id,
                "title": req.title,
                "description": req.description[:500],
                "acceptance_criteria": req.acceptance_criteria[:10],
                "priority": req.priority,
                "tags": req.tags,
            })

        negative_instruction = ""
        if config.include_negative_tests:
            negative_instruction = """
- NEGATIVE SCENARIOS: For each requirement, generate 1-2 negative test scenarios:
  - Invalid input (wrong format, empty, too long, special characters)
  - Unauthorized access (wrong role, expired session, missing auth)
  - Resource not found or unavailable
  Tag negative scenarios with "negative"."""

        edge_case_instruction = ""
        if config.include_edge_cases:
            edge_case_instruction = """
- EDGE CASE SCENARIOS: For each requirement, consider:
  - Boundary values (min, max, zero, empty string, max length)
  - Concurrent operations (race conditions, double submissions)
  - State transitions (first use, expired data, cache misses)
  Tag edge case scenarios with "edge_case"."""

        max_per_req = max(2, config.max_tests // max(len(requirements), 1))

        prompt = f"""Generate BDD/Gherkin scenarios for these requirements.

REQUIREMENTS:
{json.dumps(req_summaries, indent=2)}
{avoidance_hints}

INSTRUCTIONS:
- Generate {max_per_req} scenarios per requirement (at most {config.max_tests} total).
- Each scenario MUST reference its source requirement ID.
- Use Given/When/Then format strictly.
- Be specific: include concrete values, user roles, expected messages.
{negative_instruction}
{edge_case_instruction}

SELECTOR STRATEGY: {config.selector_strategy}
- If "semantic": use text content, labels, ARIA roles
- If "data-testid": use data-testid attributes
- If "css": use CSS selectors
- If "xpath": use XPath (least preferred)

Return JSON:
{{
    "scenarios": [
        {{
            "name": "Descriptive scenario name",
            "feature": "Feature/module name",
            "given": ["precondition step 1", "precondition step 2"],
            "when": ["action step 1", "action step 2"],
            "then": ["expected result 1", "expected result 2"],
            "and_steps": ["additional verification"],
            "tags": ["happy_path", "authentication"],
            "requirement_id": "REQ-001",
            "is_negative": false,
            "is_edge_case": false
        }}
    ]
}}"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            required_capabilities=[AICapability.REASONING],
            max_tokens=8192,
            temperature=0.2,
            json_mode=True,
            max_cost=0.15,
        )

        parsed = self._parse_json_response(response.content, {"scenarios": []})
        scenarios: list[BDDScenario] = []

        for raw in parsed.get("scenarios", []):
            scenario = BDDScenario(
                name=str(raw.get("name", "Unnamed Scenario")),
                feature=str(raw.get("feature", "General")),
                given=[str(s) for s in raw.get("given", [])],
                when=[str(s) for s in raw.get("when", [])],
                then=[str(s) for s in raw.get("then", [])],
                and_steps=[str(s) for s in raw.get("and_steps", [])],
                tags=[str(t) for t in raw.get("tags", [])],
                requirement_id=str(raw.get("requirement_id", "")),
                is_negative=bool(raw.get("is_negative", False)),
                is_edge_case=bool(raw.get("is_edge_case", False)),
            )
            scenarios.append(scenario)

        # Enforce max_tests limit
        return scenarios[: config.max_tests]

    # =========================================================================
    # Step 4: Test Code Generation
    # =========================================================================

    async def _generate_test_code(
        self,
        scenarios: list[BDDScenario],
        framework: TestFramework,
        language: TestLanguage,
        config: TestGenConfig,
        cognee_context: dict[str, Any] | None = None,
    ) -> list[GeneratedTestCase]:
        """Generate framework-specific test code from BDD scenarios.

        Delegates to framework-specific code emitters when available,
        falling back to AI-generated code.

        Args:
            scenarios: BDD scenarios to implement
            framework: Target test framework
            language: Target programming language
            config: Generation configuration
            cognee_context: Cognee context for pattern avoidance

        Returns:
            List of GeneratedTestCase objects
        """
        # Try to use the dedicated code_emitters module
        try:
            from .testgen.code_emitters import emit_test_code

            emitted = await emit_test_code(
                scenarios=scenarios,
                framework=framework,
                language=language,
                config=config,
                cognee_context=cognee_context,
            )
            if emitted:
                return emitted
        except ImportError:
            self.log.debug(
                "code_emitters module not available, using AI generation fallback"
            )
        except Exception as e:
            self.log.warning(
                "code_emitters failed, falling back to AI generation",
                error=str(e),
            )

        # Fallback: generate code via AI
        return await self._generate_test_code_via_ai(
            scenarios=scenarios,
            framework=framework,
            language=language,
            config=config,
            cognee_context=cognee_context,
        )

    async def _generate_test_code_via_ai(
        self,
        scenarios: list[BDDScenario],
        framework: TestFramework,
        language: TestLanguage,
        config: TestGenConfig,
        cognee_context: dict[str, Any] | None = None,
    ) -> list[GeneratedTestCase]:
        """Generate test code using AI when code_emitters is unavailable.

        Processes scenarios in batches to stay within token limits.

        Args:
            scenarios: BDD scenarios
            framework: Target framework
            language: Target language
            config: Generation config
            cognee_context: Cognee patterns

        Returns:
            List of GeneratedTestCase
        """
        tests: list[GeneratedTestCase] = []
        batch_size = 5  # Process 5 scenarios at a time

        # Build framework-specific instructions
        framework_instructions = self._get_framework_instructions(framework, language)

        # Build avoidance hints from Cognee
        avoidance_hints = ""
        if cognee_context and cognee_context.get("fragile_selectors"):
            fragile = cognee_context["fragile_selectors"][:5]
            patterns = []
            for item in fragile:
                if isinstance(item, dict):
                    patterns.append(str(item.get("pattern", str(item))))
                else:
                    patterns.append(str(item))
            avoidance_hints = (
                "\nAVOID these selector patterns (known to be fragile):\n"
                + "\n".join(f"- {p}" for p in patterns)
            )

        for i in range(0, len(scenarios), batch_size):
            batch = scenarios[i : i + batch_size]
            scenario_data = [s.to_dict() for s in batch]

            app_url_hint = ""
            if config.app_url:
                app_url_hint = f"\nAPP URL: {config.app_url}"

            prompt = f"""Generate executable {framework.value} test code in {language.value} for these BDD scenarios.

SCENARIOS:
{json.dumps(scenario_data, indent=2)}

{framework_instructions}
{app_url_hint}
{avoidance_hints}

REQUIREMENTS:
1. Each scenario becomes one test function/method.
2. Use descriptive test names based on the scenario name.
3. Include all Given/When/Then steps as executable code.
4. Add assertions for all Then steps.
5. Include proper imports, setup, and teardown.
6. Use {config.selector_strategy} selector strategy.
7. Add comments explaining the test intent.
8. Handle async/await properly for the target framework.

Return JSON:
{{
    "tests": [
        {{
            "name": "test_descriptive_name",
            "code": "full executable test code as a string",
            "requirement_ids": ["REQ-001"],
            "tags": ["happy_path"],
            "estimated_duration_seconds": 30
        }}
    ]
}}"""

            response = await self._call_ai(
                messages=[{"role": "user", "content": prompt}],
                task_type=TaskType.TEST_GENERATION,
                required_capabilities=[AICapability.REASONING],
                max_tokens=8192,
                temperature=0.1,
                json_mode=True,
                max_cost=0.15,
            )

            parsed = self._parse_json_response(response.content, {"tests": []})

            for raw in parsed.get("tests", []):
                test = GeneratedTestCase(
                    name=str(raw.get("name", f"test_{uuid4().hex[:8]}")),
                    code=str(raw.get("code", "")),
                    framework=framework.value,
                    language=language.value,
                    bdd_scenario=batch[0] if len(batch) == 1 else None,
                    requirement_ids=[
                        str(rid) for rid in raw.get("requirement_ids", [])
                    ],
                    quality_score=0.0,  # Will be set by quality audit
                    tags=[str(t) for t in raw.get("tags", [])],
                    estimated_duration_seconds=int(
                        raw.get("estimated_duration_seconds", 30)
                    ),
                )
                tests.append(test)

        return tests

    def _get_framework_instructions(
        self,
        framework: TestFramework,
        language: TestLanguage,
    ) -> str:
        """Get framework-specific code generation instructions.

        Args:
            framework: Target framework
            language: Target language

        Returns:
            Instruction text for the AI
        """
        instructions = {
            TestFramework.PLAYWRIGHT: f"""FRAMEWORK: Playwright ({language.value})
{"- Use async/await with pytest-playwright" if language == TestLanguage.PYTHON else "- Use @playwright/test"}
- Use page.goto(), page.click(), page.fill(), page.locator()
- Prefer locators: page.get_by_role(), page.get_by_text(), page.get_by_label()
- Use expect() for assertions: expect(locator).to_be_visible()
- Include proper page.wait_for_load_state() after navigation
{"- Use @pytest.mark.asyncio decorator" if language == TestLanguage.PYTHON else "- Use test() function from @playwright/test"}""",
            TestFramework.CYPRESS: """FRAMEWORK: Cypress (JavaScript/TypeScript)
- Use cy.visit(), cy.get(), cy.contains(), cy.click(), cy.type()
- Use cy.should() for assertions
- Use cy.intercept() for API mocking
- Use describe/it blocks for test organization
- Include cy.wait() or cy.intercept() for async operations""",
            TestFramework.SELENIUM: f"""FRAMEWORK: Selenium WebDriver ({language.value})
- Use driver.get(), driver.find_element(), By.CSS_SELECTOR, By.XPATH
- Use WebDriverWait for explicit waits
- Use expected_conditions for element state checks
- Include proper setup (driver creation) and teardown (driver.quit())
- Use Page Object Model for maintainability""",
            TestFramework.PYTEST: """FRAMEWORK: pytest (Python)
- Use standard pytest conventions: test_ prefix, assert statements
- Use pytest fixtures for setup/teardown
- Use pytest.mark.parametrize for data-driven tests
- Use pytest.raises for exception testing
- Include proper docstrings for each test""",
            TestFramework.ROBOT: """FRAMEWORK: Robot Framework
- Use *** Settings ***, *** Variables ***, *** Test Cases ***, *** Keywords ***
- Use SeleniumLibrary or Browser library keywords
- Use [Tags] for test categorization
- Use [Setup] and [Teardown] for test lifecycle
- Use Built-in keywords for assertions""",
        }

        return instructions.get(framework, f"FRAMEWORK: {framework.value}")

    # =========================================================================
    # Step 5: Quality Audit
    # =========================================================================

    async def _quality_audit(
        self,
        generated_tests: list[GeneratedTestCase],
        requirements: list[ParsedRequirement],
        config: TestGenConfig,
    ) -> tuple[list[GeneratedTestCase], float]:
        """Run quality audit on generated tests using AgentAsJudge.

        Evaluates each test for:
        - Correctness: Does the test match the requirement?
        - Completeness: Are all acceptance criteria covered?
        - Reliability: Is the test likely to be flaky?
        - Maintainability: Is the code clean and well-structured?

        If quality is below threshold, triggers a refinement loop.

        Args:
            generated_tests: Tests to audit
            requirements: Requirements for context
            config: Config with quality_threshold

        Returns:
            Tuple of (refined tests, overall quality score)
        """
        if not generated_tests:
            return [], 0.0

        req_map = {r.id: r for r in requirements}
        total_score = 0.0
        refined_tests: list[GeneratedTestCase] = []

        for test in generated_tests:
            # Build context for the judge
            related_reqs = [
                req_map[rid].to_dict()
                for rid in test.requirement_ids
                if rid in req_map
            ]

            input_context = {
                "test_name": test.name,
                "framework": test.framework,
                "language": test.language,
                "requirements": related_reqs,
                "bdd_scenario": (
                    test.bdd_scenario.to_dict() if test.bdd_scenario else None
                ),
            }

            # Evaluate with AgentAsJudge
            try:
                evaluation: AgentEvaluation = (
                    await self.judge.evaluate_agent_output(
                        agent_type="TestGenAgent",
                        input_context=input_context,
                        output={"name": test.name, "code": test.code[:3000]},
                        criteria=[
                            "correctness",
                            "completeness",
                            "reliability",
                            "maintainability",
                        ],
                    )
                )

                test.quality_score = evaluation.overall_score
                test.quality_issues = [
                    f"[{issue.severity.value}] {issue.description}"
                    for issue in evaluation.issues
                ]

                # Refinement loop if quality is below threshold
                if (
                    evaluation.overall_score < config.quality_threshold
                    and config.max_refinement_rounds > 0
                ):
                    test = await self._refine_test(
                        test=test,
                        evaluation=evaluation,
                        requirements=related_reqs,
                        config=config,
                    )

                total_score += test.quality_score
                refined_tests.append(test)

            except Exception as e:
                self.log.warning(
                    "Quality audit failed for test (using as-is)",
                    test_name=test.name,
                    error=str(e),
                )
                test.quality_score = 0.5  # Assume moderate quality
                total_score += 0.5
                refined_tests.append(test)

        avg_score = total_score / len(refined_tests) if refined_tests else 0.0
        return refined_tests, avg_score

    async def _refine_test(
        self,
        test: GeneratedTestCase,
        evaluation: AgentEvaluation,
        requirements: list[dict],
        config: TestGenConfig,
    ) -> GeneratedTestCase:
        """Refine a test based on quality audit feedback.

        Args:
            test: Test to refine
            evaluation: Quality evaluation with issues and suggestions
            requirements: Related requirements
            config: Generation configuration

        Returns:
            Refined GeneratedTestCase
        """
        issues_text = "\n".join(
            f"- [{issue.severity.value}] {issue.description}"
            + (f" | Fix: {issue.suggested_fix}" if issue.suggested_fix else "")
            for issue in evaluation.issues
        )
        suggestions_text = "\n".join(f"- {s}" for s in evaluation.suggestions)

        for round_num in range(config.max_refinement_rounds):
            prompt = f"""Refine this test code based on quality audit feedback.

CURRENT TEST:
Name: {test.name}
Framework: {test.framework}
Language: {test.language}

Code:
```
{test.code[:4000]}
```

REQUIREMENTS CONTEXT:
{json.dumps(requirements, indent=2, default=str)[:2000]}

QUALITY ISSUES:
{issues_text}

SUGGESTIONS:
{suggestions_text}

CURRENT SCORE: {evaluation.overall_score:.2f} (threshold: {config.quality_threshold:.2f})

Fix all issues and apply suggestions. Return the improved test code.

Return JSON:
{{
    "name": "{test.name}",
    "code": "improved complete test code",
    "quality_improvements": ["what was fixed"]
}}"""

            response = await self._call_ai(
                messages=[{"role": "user", "content": prompt}],
                task_type=TaskType.TEST_GENERATION,
                required_capabilities=[AICapability.REASONING],
                max_tokens=4096,
                temperature=0.1,
                json_mode=True,
                max_cost=0.10,
            )

            parsed = self._parse_json_response(response.content, {})
            if parsed and parsed.get("code"):
                test.code = str(parsed["code"])
                # Bump quality score slightly for each refinement
                test.quality_score = min(
                    1.0,
                    evaluation.overall_score + 0.1 * (round_num + 1),
                )

                self.log.debug(
                    "Test refined",
                    test_name=test.name,
                    round=round_num + 1,
                    new_score=test.quality_score,
                )

                # Re-evaluate if we still have rounds left
                if (
                    round_num < config.max_refinement_rounds - 1
                    and test.quality_score < config.quality_threshold
                ):
                    continue
                break
            else:
                self.log.warning(
                    "Refinement produced no code, keeping original",
                    test_name=test.name,
                    round=round_num + 1,
                )
                break

        return test

    # =========================================================================
    # Step 6: Coverage Matrix
    # =========================================================================

    def _build_coverage_matrix(
        self,
        requirements: list[ParsedRequirement],
        generated_tests: list[GeneratedTestCase],
    ) -> list[CoverageMatrixEntry]:
        """Build requirement-to-test traceability matrix.

        Maps each requirement to the tests that cover it, determining
        coverage type based on test tags and scenario properties.

        Args:
            requirements: Parsed requirements
            generated_tests: Generated tests with requirement_ids

        Returns:
            List of CoverageMatrixEntry objects
        """
        req_map = {r.id: r for r in requirements}
        matrix: list[CoverageMatrixEntry] = []

        for test in generated_tests:
            for req_id in test.requirement_ids:
                req = req_map.get(req_id)
                if not req:
                    continue

                # Determine coverage type from tags and scenario properties
                coverage_type = CoverageType.FUNCTIONAL
                if test.bdd_scenario:
                    if test.bdd_scenario.is_negative:
                        coverage_type = CoverageType.NEGATIVE
                    elif test.bdd_scenario.is_edge_case:
                        coverage_type = CoverageType.EDGE_CASE
                elif "negative" in test.tags:
                    coverage_type = CoverageType.NEGATIVE
                elif "edge_case" in test.tags:
                    coverage_type = CoverageType.EDGE_CASE
                elif "security" in test.tags:
                    coverage_type = CoverageType.SECURITY
                elif "accessibility" in test.tags:
                    coverage_type = CoverageType.ACCESSIBILITY
                elif "performance" in test.tags:
                    coverage_type = CoverageType.PERFORMANCE

                # Determine test type
                test_type = "e2e"
                if test.framework in ("pytest",):
                    test_type = "unit"
                elif "api" in test.tags:
                    test_type = "api"
                elif "integration" in test.tags:
                    test_type = "integration"

                # Determine code entity type from requirement tags
                entity_type = CodeEntityType.USER_FLOW
                if "api" in req.tags:
                    entity_type = CodeEntityType.ENDPOINT
                elif "ui" in req.tags:
                    entity_type = CodeEntityType.UI_COMPONENT
                elif "data" in req.tags or "database" in req.tags:
                    entity_type = CodeEntityType.DATABASE_TABLE

                # Confidence based on test quality score
                confidence = min(1.0, test.quality_score + 0.1)

                entry = CoverageMatrixEntry(
                    requirement_id=req_id,
                    requirement_title=req.title,
                    code_entity=req.title,  # Best we can do without code analysis
                    code_entity_type=entity_type,
                    test_name=test.name,
                    test_type=test_type,
                    coverage_type=coverage_type,
                    confidence=round(confidence, 2),
                )
                matrix.append(entry)

        return matrix

    # =========================================================================
    # Step 7: Cognee Storage
    # =========================================================================

    async def _store_in_cognee(
        self,
        org_id: str,
        project_id: str,
        requirements: list[ParsedRequirement],
        tests: list[GeneratedTestCase],
        coverage_matrix: list[CoverageMatrixEntry],
        job_id: str,
    ) -> list[str]:
        """Store all generated artifacts in Cognee for cross-session learning.

        Stores in three namespaces:
        - testgen/requirements: Parsed requirements with ambiguity scores
        - testgen/generated_tests: Generated test code and metadata
        - testgen/coverage_matrix: Requirement-to-test traceability

        Per MEMORY.md #4: sanitize None values before Cognee storage.

        Args:
            org_id: Organization ID
            project_id: Project ID
            requirements: Parsed requirements
            tests: Generated tests
            coverage_matrix: Traceability matrix
            job_id: Job ID for tracking

        Returns:
            List of Cognee namespace strings used
        """
        namespaces_used: list[str] = []

        try:
            from ..knowledge.cognee_client import get_cognee_client

            cognee = get_cognee_client(org_id=org_id, project_id=project_id)

            # Store requirements
            ns_reqs = ("testgen", "requirements")
            for req in requirements:
                try:
                    # Sanitize None values per MEMORY.md #4
                    value = _sanitize_for_cognee({
                        "id": req.id,
                        "title": req.title,
                        "description": req.description,
                        "acceptance_criteria": req.acceptance_criteria,
                        "ambiguity_score": req.ambiguity_score,
                        "priority": req.priority,
                        "tags": req.tags,
                        "job_id": job_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    await cognee.put(
                        namespace=ns_reqs,
                        key=f"req_{req.id}_{job_id}",
                        value=value,
                        embed_text=f"{req.title}: {req.description}",
                    )
                except Exception as e:
                    self.log.debug(
                        "Failed to store requirement in Cognee (non-fatal)",
                        req_id=req.id,
                        error=str(e),
                    )
            namespaces_used.append("testgen/requirements")

            # Store generated tests
            ns_tests = ("testgen", "generated_tests")
            for test in tests:
                try:
                    value = _sanitize_for_cognee({
                        "name": test.name,
                        "framework": test.framework,
                        "language": test.language,
                        "requirement_ids": test.requirement_ids,
                        "quality_score": test.quality_score,
                        "tags": test.tags,
                        "code_preview": test.code[:500],
                        "job_id": job_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    await cognee.put(
                        namespace=ns_tests,
                        key=f"test_{test.name}_{job_id}",
                        value=value,
                        embed_text=f"Test: {test.name}, Framework: {test.framework}",
                    )
                except Exception as e:
                    self.log.debug(
                        "Failed to store test in Cognee (non-fatal)",
                        test_name=test.name,
                        error=str(e),
                    )
            namespaces_used.append("testgen/generated_tests")

            # Store coverage matrix summary
            ns_matrix = ("testgen", "coverage_matrix")
            try:
                matrix_summary = _sanitize_for_cognee({
                    "job_id": job_id,
                    "entries_count": len(coverage_matrix),
                    "requirements_covered": len(
                        {e.requirement_id for e in coverage_matrix}
                    ),
                    "total_requirements": len(requirements),
                    "tests_count": len(tests),
                    "coverage_types": list(
                        {e.coverage_type.value for e in coverage_matrix}
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                await cognee.put(
                    namespace=ns_matrix,
                    key=f"matrix_{job_id}",
                    value=matrix_summary,
                    embed_text=(
                        f"Coverage matrix: {len(coverage_matrix)} entries "
                        f"covering {len(requirements)} requirements"
                    ),
                )
                namespaces_used.append("testgen/coverage_matrix")
            except Exception as e:
                self.log.debug(
                    "Failed to store coverage matrix in Cognee (non-fatal)",
                    error=str(e),
                )

            self.log.info(
                "Artifacts stored in Cognee",
                namespaces=namespaces_used,
                requirements=len(requirements),
                tests=len(tests),
            )

        except Exception as e:
            self.log.warning(
                "Cognee storage failed entirely (non-fatal)",
                error=str(e),
            )

        return namespaces_used

    # =========================================================================
    # Step 8: Event Emission
    # =========================================================================

    async def _emit_events(
        self,
        org_id: str,
        project_id: str,
        job_id: str,
        results: TestGenResult,
        source_type: SourceType,
        config: TestGenConfig,
    ) -> list[str]:
        """Emit Kafka events for downstream processing.

        Emits:
        - TEST_GENERATED: Summary of generated tests
        - REQUIREMENTS_ANALYZED: Summary of requirements analysis

        Args:
            org_id: Organization ID
            project_id: Project ID
            job_id: Job tracking ID
            results: TestGenResult with all artifacts
            source_type: Source type for requirements
            config: Generation configuration

        Returns:
            List of event types emitted
        """
        emitted: list[str] = []

        try:
            from ..config import get_settings
            from ..events.producer import EventProducer
            from ..events.schemas import (
                EventMetadata,
                RequirementsAnalyzedEvent,
                TenantInfo,
                TestGeneratedEvent,
            )
            from ..events.topics import get_topic_for_event

            settings = get_settings()

            if not settings.redpanda_brokers:
                self.log.debug(
                    "Redpanda not configured, skipping event emission"
                )
                return emitted

            tenant = TenantInfo(org_id=org_id, project_id=project_id)
            metadata = EventMetadata(
                source="testgen_agent",
                triggered_by="agent",
            )

            # Build the events
            test_gen_event = TestGeneratedEvent(
                tenant=tenant,
                metadata=metadata,
                job_id=job_id,
                tests_generated=results.tests_count,
                framework=config.framework.value,
                source_type=source_type.value,
                quality_score=results.quality_score,
                requirements_count=results.requirements_count,
                coverage_matrix_entries=results.coverage_entries_count,
                ai_cost=results.ai_cost,
            )

            req_event = RequirementsAnalyzedEvent(
                tenant=tenant,
                metadata=metadata,
                job_id=job_id,
                requirements_count=results.requirements_count,
                ambiguous_count=len(
                    [r for r in results.requirements if r.ambiguity_score > 0.5]
                ),
                avg_ambiguity_score=results.avg_ambiguity,
                source_type=source_type.value,
                cognee_namespace=(
                    results.cognee_namespaces[0] if results.cognee_namespaces else ""
                ),
            )

            # Send events
            producer = EventProducer(
                bootstrap_servers=settings.redpanda_brokers,
            )
            await producer.start()
            try:
                test_gen_topic = get_topic_for_event(test_gen_event.event_type)
                sent = await producer.send(test_gen_event, topic=test_gen_topic)
                if sent:
                    emitted.append("TEST_GENERATED")

                req_topic = get_topic_for_event(req_event.event_type)
                sent = await producer.send(req_event, topic=req_topic)
                if sent:
                    emitted.append("REQUIREMENTS_ANALYZED")

            finally:
                await producer.stop()

            self.log.info("Kafka events emitted", events=emitted)

        except Exception as e:
            self.log.warning(
                "Event emission failed (non-fatal)",
                error=str(e),
            )

        return emitted


# =============================================================================
# Helpers
# =============================================================================


def _sanitize_for_cognee(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a dict for Cognee storage by replacing None with empty strings.

    Per MEMORY.md #4: Cognee's internal processing fails with
    'int() argument must be a string...NoneType' when None values are present.

    Args:
        data: Dictionary to sanitize

    Returns:
        Sanitized dictionary with no None values
    """
    sanitized: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_for_cognee(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_for_cognee(item) if isinstance(item, dict)
                else ("" if item is None else item)
                for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized


# =============================================================================
# Factory Functions
# =============================================================================


def create_testgen_agent(**kwargs) -> TestGenAgent:
    """Factory function for TestGenAgent.

    Args:
        **kwargs: Passed to BaseAgent constructor

    Returns:
        TestGenAgent instance
    """
    return TestGenAgent(**kwargs)
