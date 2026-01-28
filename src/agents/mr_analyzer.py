"""
MR/PR Analyzer Agent - GitLab Duo-style Analysis.

RAP-234: Analyzes merge request and pull request changes to generate
intelligent test suggestions, coverage gap analysis, and actionable comments.

Features:
- Code change analysis with semantic understanding
- Test suggestion generation based on change impact
- Coverage gap detection
- Automated comment generation for GitHub/GitLab
- Integration with existing test knowledge base
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

from src.agents.base import AgentCapability, AgentResult, BaseAgent
from src.core.model_router import TaskType

logger = structlog.get_logger(__name__)


class ChangeType(str, Enum):
    """Types of code changes."""
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    TEST = "test"
    DOCS = "docs"
    CONFIG = "config"
    DEPENDENCY = "dependency"
    SECURITY = "security"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"


class TestPriority(str, Enum):
    """Priority levels for suggested tests."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TestType(str, Enum):
    """Types of tests that can be suggested."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    API = "api"
    VISUAL = "visual"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class FileChange:
    """Represents a file change from a PR/MR."""
    path: str
    status: str  # added, modified, removed, renamed
    additions: int = 0
    deletions: int = 0
    patch: str | None = None
    previous_path: str | None = None


@dataclass
class ChangeAnalysis:
    """Analysis of a single file change."""
    file_path: str
    change_type: ChangeType
    risk_level: str  # critical, high, medium, low
    affected_areas: list[str]
    semantic_summary: str
    functions_changed: list[str] = field(default_factory=list)
    imports_changed: list[str] = field(default_factory=list)
    potential_impacts: list[str] = field(default_factory=list)


@dataclass
class MRAnalysis:
    """Complete analysis of an MR/PR."""
    file_analyses: list[ChangeAnalysis]
    overall_change_type: ChangeType
    overall_risk_score: float  # 0.0 to 1.0
    affected_components: list[str]
    affected_routes: list[str]
    coverage_gaps: list[str]
    summary: str
    recommendations: list[str]
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_analyses": [
                {
                    "file_path": fa.file_path,
                    "change_type": fa.change_type.value,
                    "risk_level": fa.risk_level,
                    "affected_areas": fa.affected_areas,
                    "semantic_summary": fa.semantic_summary,
                    "functions_changed": fa.functions_changed,
                    "imports_changed": fa.imports_changed,
                    "potential_impacts": fa.potential_impacts,
                }
                for fa in self.file_analyses
            ],
            "overall_change_type": self.overall_change_type.value,
            "overall_risk_score": self.overall_risk_score,
            "affected_components": self.affected_components,
            "affected_routes": self.affected_routes,
            "coverage_gaps": self.coverage_gaps,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "analyzed_at": self.analyzed_at.isoformat(),
        }


@dataclass
class TestSuggestion:
    """A suggested test for the MR/PR."""
    name: str
    description: str
    test_type: TestType
    priority: TestPriority
    target_files: list[str]
    coverage_target: str  # What gap this fills
    estimated_effort: str  # "trivial", "small", "medium", "large"
    example_code: str | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "priority": self.priority.value,
            "target_files": self.target_files,
            "coverage_target": self.coverage_target,
            "estimated_effort": self.estimated_effort,
            "rationale": self.rationale,
        }
        if self.example_code:
            result["example_code"] = self.example_code
        return result


class MRAnalyzerAgent(BaseAgent):
    """
    Analyzes MR/PR changes and generates test suggestions.

    This agent provides GitLab Duo-style analysis capabilities:
    - Semantic understanding of code changes
    - Test suggestion generation
    - Coverage gap detection
    - Automated comment creation

    Capabilities:
    - mr_analysis: Analyze changes in a merge request
    - test_suggestion: Generate test suggestions
    - coverage_gap: Detect coverage gaps
    """

    # RAP-231: Agent capabilities for A2A discovery
    CAPABILITIES = [
        AgentCapability.MR_ANALYSIS,
        AgentCapability.CODE_ANALYSIS,
        AgentCapability.TEST_GENERATION,
    ]
    DEFAULT_TASK_TYPE = TaskType.CODE_ANALYSIS

    def __init__(self, **kwargs):
        """Initialize the MR Analyzer Agent."""
        super().__init__(**kwargs)
        self.log = logger.bind(agent="MRAnalyzerAgent")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for analysis."""
        return """You are an expert code reviewer and test engineer with deep knowledge of:
- Software testing best practices (unit, integration, E2E, API testing)
- Test coverage analysis and gap detection
- Risk assessment of code changes
- Various programming languages and frameworks

Your role is to analyze code changes and provide:
1. Clear understanding of what changed and why
2. Assessment of risk and potential impacts
3. Specific, actionable test suggestions
4. Coverage gap identification

Be precise, thorough, and focus on providing actionable insights.
Prioritize suggestions that provide the most value for code quality and reliability.
"""

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the agent's main task."""
        changes = kwargs.get("changes", [])
        project_id = kwargs.get("project_id", "")

        if not changes:
            return AgentResult(
                success=False,
                error="No changes provided for analysis",
            )

        try:
            analysis = await self.analyze(changes, project_id)
            suggestions = await self.suggest_tests(analysis)

            return AgentResult(
                success=True,
                data={
                    "analysis": analysis.to_dict(),
                    "suggestions": [s.to_dict() for s in suggestions],
                },
            )
        except Exception as e:
            self.log.exception("Analysis failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
            )

    async def analyze(
        self,
        changes: list[FileChange] | list[dict],
        project_id: str,
    ) -> MRAnalysis:
        """
        Analyze MR/PR changes and generate a comprehensive analysis.

        Args:
            changes: List of FileChange objects or dicts with file change info
            project_id: Argus project ID for context

        Returns:
            MRAnalysis with detailed change analysis
        """
        # Convert dicts to FileChange if needed
        if changes and isinstance(changes[0], dict):
            changes = [
                FileChange(
                    path=c.get("path", ""),
                    status=c.get("status", "modified"),
                    additions=c.get("additions", 0),
                    deletions=c.get("deletions", 0),
                    patch=c.get("patch"),
                    previous_path=c.get("previous_path"),
                )
                for c in changes
            ]

        # Prepare change summaries for AI analysis
        changes_summary = self._prepare_changes_summary(changes)

        # Use AI to analyze changes
        prompt = f"""Analyze the following code changes from a merge request:

{changes_summary}

Provide a JSON analysis with the following structure:
{{
    "file_analyses": [
        {{
            "file_path": "path/to/file",
            "change_type": "feature|bugfix|refactor|test|docs|config|dependency|security|performance|unknown",
            "risk_level": "critical|high|medium|low",
            "affected_areas": ["area1", "area2"],
            "semantic_summary": "Brief description of what changed",
            "functions_changed": ["function1", "function2"],
            "imports_changed": ["module1", "module2"],
            "potential_impacts": ["impact1", "impact2"]
        }}
    ],
    "overall_change_type": "feature|bugfix|refactor|...",
    "overall_risk_score": 0.0-1.0,
    "affected_components": ["component1", "component2"],
    "affected_routes": ["/route1", "/route2"],
    "coverage_gaps": ["gap1", "gap2"],
    "summary": "Brief overall summary",
    "recommendations": ["recommendation1", "recommendation2"]
}}

Focus on:
1. Understanding the semantic meaning of changes
2. Identifying high-risk areas (auth, payments, data handling)
3. Detecting potential coverage gaps
4. Providing actionable recommendations
"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            max_tokens=4096,
            temperature=0.2,
        )

        # Parse response
        try:
            analysis_data = self._parse_json_response(response.content, {})
            return self._build_analysis(analysis_data, changes)
        except Exception as e:
            self.log.warning("Failed to parse analysis", error=str(e))
            return self._build_default_analysis(changes)

    async def suggest_tests(self, analysis: MRAnalysis) -> list[TestSuggestion]:
        """
        Generate test suggestions based on MR analysis.

        Args:
            analysis: MRAnalysis from the analyze() method

        Returns:
            List of TestSuggestion objects
        """
        # Prepare context for AI
        context = {
            "summary": analysis.summary,
            "overall_risk": analysis.overall_risk_score,
            "change_type": analysis.overall_change_type.value,
            "coverage_gaps": analysis.coverage_gaps,
            "affected_components": analysis.affected_components,
            "file_analyses": [
                {
                    "file": fa.file_path,
                    "change_type": fa.change_type.value,
                    "risk": fa.risk_level,
                    "functions": fa.functions_changed,
                    "impacts": fa.potential_impacts,
                }
                for fa in analysis.file_analyses
            ],
        }

        prompt = f"""Based on this MR analysis, suggest specific tests that should be added or run:

Analysis:
{json.dumps(context, indent=2)}

Provide test suggestions as JSON:
{{
    "suggestions": [
        {{
            "name": "Test name (e.g., 'test_user_authentication_flow')",
            "description": "What this test verifies",
            "test_type": "unit|integration|e2e|api|visual|performance|security",
            "priority": "critical|high|medium|low",
            "target_files": ["file1.py", "file2.py"],
            "coverage_target": "What coverage gap this fills",
            "estimated_effort": "trivial|small|medium|large",
            "rationale": "Why this test is important",
            "example_code": "Optional: Example test code snippet"
        }}
    ]
}}

Guidelines:
1. Prioritize tests for high-risk changes (auth, payments, data)
2. Include both unit tests for specific functions and integration tests for flows
3. Consider edge cases and error handling
4. For UI changes, suggest visual regression tests
5. For API changes, suggest contract and integration tests
6. Be specific - suggest actual test names and what they verify
"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.TEST_GENERATION,
            max_tokens=4096,
            temperature=0.3,
        )

        # Parse response
        try:
            data = self._parse_json_response(response.content, {"suggestions": []})
            return self._build_suggestions(data.get("suggestions", []))
        except Exception as e:
            self.log.warning("Failed to parse suggestions", error=str(e))
            return self._build_default_suggestions(analysis)

    async def generate_comment(
        self,
        suggestions: list[TestSuggestion],
        include_code_snippets: bool = True,
        max_suggestions: int = 5,
    ) -> str:
        """
        Generate a formatted comment for GitHub/GitLab.

        Args:
            suggestions: List of test suggestions
            include_code_snippets: Whether to include example code
            max_suggestions: Maximum number of suggestions to include

        Returns:
            Formatted markdown comment string
        """
        if not suggestions:
            return ""

        # Sort by priority
        priority_order = {
            TestPriority.CRITICAL: 0,
            TestPriority.HIGH: 1,
            TestPriority.MEDIUM: 2,
            TestPriority.LOW: 3,
        }
        sorted_suggestions = sorted(
            suggestions,
            key=lambda s: priority_order.get(s.priority, 4)
        )[:max_suggestions]

        # Build comment
        lines = [
            "## Test Suggestions",
            "",
            "Based on the changes in this MR/PR, here are recommended tests to add or verify:",
            "",
        ]

        # Group by priority
        critical = [s for s in sorted_suggestions if s.priority == TestPriority.CRITICAL]
        high = [s for s in sorted_suggestions if s.priority == TestPriority.HIGH]
        medium = [s for s in sorted_suggestions if s.priority == TestPriority.MEDIUM]
        low = [s for s in sorted_suggestions if s.priority == TestPriority.LOW]

        if critical:
            lines.append("### Critical Priority")
            lines.append("")
            for s in critical:
                lines.extend(self._format_suggestion(s, include_code_snippets))
            lines.append("")

        if high:
            lines.append("### High Priority")
            lines.append("")
            for s in high:
                lines.extend(self._format_suggestion(s, include_code_snippets))
            lines.append("")

        if medium:
            lines.append("### Medium Priority")
            lines.append("")
            for s in medium:
                lines.extend(self._format_suggestion(s, include_code_snippets))
            lines.append("")

        if low:
            lines.append("<details>")
            lines.append("<summary>Low Priority Suggestions</summary>")
            lines.append("")
            for s in low:
                lines.extend(self._format_suggestion(s, include_code_snippets))
            lines.append("</details>")
            lines.append("")

        # Add footer
        lines.extend([
            "---",
            "*Generated by [Argus E2E Testing Agent](https://github.com/argus-testing/e2e-testing-agent)*",
        ])

        return "\n".join(lines)

    def _prepare_changes_summary(self, changes: list[FileChange]) -> str:
        """Prepare a text summary of changes for AI analysis."""
        lines = []

        for change in changes:
            lines.append(f"### File: {change.path}")
            lines.append(f"Status: {change.status}")
            lines.append(f"Additions: {change.additions}, Deletions: {change.deletions}")

            if change.previous_path:
                lines.append(f"Renamed from: {change.previous_path}")

            if change.patch:
                # Truncate long patches
                patch = change.patch
                if len(patch) > 3000:
                    patch = patch[:3000] + "\n... (truncated)"
                lines.append("```diff")
                lines.append(patch)
                lines.append("```")

            lines.append("")

        return "\n".join(lines)

    def _build_analysis(
        self,
        data: dict[str, Any],
        changes: list[FileChange],
    ) -> MRAnalysis:
        """Build MRAnalysis from parsed AI response."""
        file_analyses = []

        for fa_data in data.get("file_analyses", []):
            try:
                change_type = ChangeType(fa_data.get("change_type", "unknown"))
            except ValueError:
                change_type = ChangeType.UNKNOWN

            file_analyses.append(ChangeAnalysis(
                file_path=fa_data.get("file_path", ""),
                change_type=change_type,
                risk_level=fa_data.get("risk_level", "medium"),
                affected_areas=fa_data.get("affected_areas", []),
                semantic_summary=fa_data.get("semantic_summary", ""),
                functions_changed=fa_data.get("functions_changed", []),
                imports_changed=fa_data.get("imports_changed", []),
                potential_impacts=fa_data.get("potential_impacts", []),
            ))

        try:
            overall_change_type = ChangeType(data.get("overall_change_type", "unknown"))
        except ValueError:
            overall_change_type = ChangeType.UNKNOWN

        return MRAnalysis(
            file_analyses=file_analyses,
            overall_change_type=overall_change_type,
            overall_risk_score=float(data.get("overall_risk_score", 0.5)),
            affected_components=data.get("affected_components", []),
            affected_routes=data.get("affected_routes", []),
            coverage_gaps=data.get("coverage_gaps", []),
            summary=data.get("summary", ""),
            recommendations=data.get("recommendations", []),
        )

    def _build_default_analysis(self, changes: list[FileChange]) -> MRAnalysis:
        """Build a default analysis when AI parsing fails."""
        file_analyses = []

        for change in changes:
            # Infer change type from file path
            change_type = self._infer_change_type(change.path)
            risk_level = self._infer_risk_level(change.path)

            file_analyses.append(ChangeAnalysis(
                file_path=change.path,
                change_type=change_type,
                risk_level=risk_level,
                affected_areas=[],
                semantic_summary=f"File {change.status}",
            ))

        return MRAnalysis(
            file_analyses=file_analyses,
            overall_change_type=ChangeType.UNKNOWN,
            overall_risk_score=0.5,
            affected_components=[],
            affected_routes=[],
            coverage_gaps=["Unable to detect coverage gaps - manual review recommended"],
            summary="Automated analysis completed with limited details",
            recommendations=["Review changes manually for test coverage"],
        )

    def _build_suggestions(
        self,
        suggestions_data: list[dict],
    ) -> list[TestSuggestion]:
        """Build TestSuggestion list from parsed AI response."""
        suggestions = []

        for s_data in suggestions_data:
            try:
                test_type = TestType(s_data.get("test_type", "unit"))
            except ValueError:
                test_type = TestType.UNIT

            try:
                priority = TestPriority(s_data.get("priority", "medium"))
            except ValueError:
                priority = TestPriority.MEDIUM

            suggestions.append(TestSuggestion(
                name=s_data.get("name", "Unnamed test"),
                description=s_data.get("description", ""),
                test_type=test_type,
                priority=priority,
                target_files=s_data.get("target_files", []),
                coverage_target=s_data.get("coverage_target", ""),
                estimated_effort=s_data.get("estimated_effort", "medium"),
                example_code=s_data.get("example_code"),
                rationale=s_data.get("rationale", ""),
            ))

        return suggestions

    def _build_default_suggestions(
        self,
        analysis: MRAnalysis,
    ) -> list[TestSuggestion]:
        """Build default suggestions when AI parsing fails."""
        suggestions = []

        # Generate basic suggestions based on file analysis
        for fa in analysis.file_analyses:
            if fa.risk_level in ("critical", "high"):
                suggestions.append(TestSuggestion(
                    name=f"test_{fa.file_path.replace('/', '_').replace('.', '_')}",
                    description=f"Test changes in {fa.file_path}",
                    test_type=TestType.UNIT,
                    priority=TestPriority.HIGH if fa.risk_level == "critical" else TestPriority.MEDIUM,
                    target_files=[fa.file_path],
                    coverage_target="Changed code coverage",
                    estimated_effort="medium",
                    rationale=f"High-risk changes detected in {fa.file_path}",
                ))

        return suggestions

    def _format_suggestion(
        self,
        suggestion: TestSuggestion,
        include_code: bool,
    ) -> list[str]:
        """Format a single suggestion for markdown output."""
        lines = [
            f"**{suggestion.name}** ({suggestion.test_type.value})",
            f"- {suggestion.description}",
        ]

        if suggestion.rationale:
            lines.append(f"- *Rationale:* {suggestion.rationale}")

        if suggestion.target_files:
            lines.append(f"- *Target files:* `{', '.join(suggestion.target_files)}`")

        lines.append(f"- *Effort:* {suggestion.estimated_effort}")

        if include_code and suggestion.example_code:
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>Example code</summary>")
            lines.append("")
            lines.append("```python")
            lines.append(suggestion.example_code)
            lines.append("```")
            lines.append("</details>")

        lines.append("")
        return lines

    @staticmethod
    def _infer_change_type(file_path: str) -> ChangeType:
        """Infer change type from file path."""
        path_lower = file_path.lower()

        if "test" in path_lower or "spec" in path_lower:
            return ChangeType.TEST
        elif path_lower.endswith((".md", ".rst", ".txt")):
            return ChangeType.DOCS
        elif any(p in path_lower for p in ["config", ".json", ".yaml", ".yml", ".toml", ".env"]):
            return ChangeType.CONFIG
        elif "requirements" in path_lower or "package.json" in path_lower or "pyproject" in path_lower:
            return ChangeType.DEPENDENCY
        elif any(p in path_lower for p in ["security", "auth", "password", "token"]):
            return ChangeType.SECURITY
        else:
            return ChangeType.UNKNOWN

    @staticmethod
    def _infer_risk_level(file_path: str) -> str:
        """Infer risk level from file path."""
        path_lower = file_path.lower()

        high_risk = ["auth", "payment", "billing", "security", "password", "token", "secret"]
        medium_risk = ["api", "database", "migration", "config", "user", "account"]

        if any(p in path_lower for p in high_risk):
            return "high"
        elif any(p in path_lower for p in medium_risk):
            return "medium"
        else:
            return "low"


def create_mr_analyzer() -> MRAnalyzerAgent:
    """Factory function to create MRAnalyzerAgent."""
    return MRAnalyzerAgent()
