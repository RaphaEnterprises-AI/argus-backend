"""
QA Engineer Meta-Orchestrator Agent.

RAP-400: Autonomous QA intelligence that COMPOSES the full agent fleet to deliver
end-to-end quality analysis, coverage assessment, gap identification, and
test generation.

This is NOT another leaf agent -- it is a meta-orchestrator that instantiates
and delegates to existing agents (CodeAnalyzerAgent, AutoDiscovery, TestGenAgent,
MRAnalyzerAgent, FlakyTestDetector, TestQualityIntelligenceAgent, etc.) while
adding net-new logic for:

1.  Risk-scored coverage gap identification (CRITICAL/HIGH/MEDIUM/LOW)
2.  Quality score computation (weighted multi-signal formula)
3.  Cross-agent Cognee enrichment (selector fragility, healing patterns,
    previously generated tests, past coverage snapshots)
4.  Incremental / delta analysis against previous snapshots
5.  AI-powered recommendations with effort + impact sizing
6.  Kafka event emission (COVERAGE_ANALYZED, COVERAGE_GAPS_IDENTIFIED)

Architecture:
    trigger (API / scheduler / webhook)
         |
         v
    QAEngineerAgent.analyze(org_id, project_id, analysis_type, ...)
         |
         +-- _analyze_codebase  -->  CodeAnalyzerAgent
         +-- _discover_user_flows --> AutoDiscovery / QuickDiscover
         +-- _assess_coverage  -->  AI-powered coverage mapping
         +-- _identify_gaps    -->  Risk-scored gap detection
         +-- _generate_tests_for_gaps --> TestGenAgent (for auto-fill)
         +-- _compute_quality_score   --> Multi-signal weighted score
         +-- _generate_recommendations --> AI-powered recommendations
         |
         +-- _store_in_cognee  --> Knowledge graph persistence
         +-- _emit_events      --> Kafka event pipeline
         +-- _update_job       --> Supabase job tracking
         |
         v
    QAAnalysisResult
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

from src.agents.base import (
    AgentCapability,
    AgentResult,
    AICapability,
    BaseAgent,
)
from src.core.model_router import TaskType

logger = structlog.get_logger()


# =============================================================================
# Enums
# =============================================================================


class AnalysisType(str, Enum):
    """Types of QA analysis the orchestrator can perform."""

    FULL = "full"
    PR_REVIEW = "pr_review"
    COVERAGE_ONLY = "coverage_only"
    INCREMENTAL = "incremental"


class RiskLevel(str, Enum):
    """Risk levels for coverage gaps."""

    CRITICAL = "critical"  # Auth, payment, data-integrity gaps
    HIGH = "high"          # Core user-flow gaps, frequently-changed areas
    MEDIUM = "medium"      # Standard feature gaps
    LOW = "low"            # Edge cases, cosmetic gaps


class RecommendationType(str, Enum):
    """Types of QA recommendations."""

    ADD_TESTS = "add_tests"
    FIX_FLAKY = "fix_flaky"
    REDUCE_FRAGILITY = "reduce_fragility"
    ADD_CI = "add_ci"
    REFACTOR_TESTS = "refactor_tests"


class EffortSize(str, Enum):
    """Effort sizing for recommendations."""

    SMALL = "small"    # < 1 day
    MEDIUM = "medium"  # 1-3 days
    LARGE = "large"    # > 3 days


# =============================================================================
# Pydantic-free Data Classes (matches codebase convention via @dataclass)
# =============================================================================


@dataclass
class CoverageGap:
    """A single coverage gap with risk scoring."""

    entity: str                           # Function / endpoint / component name
    entity_type: str                      # function, endpoint, component, page
    risk_level: str                       # RiskLevel value
    risk_score: float                     # 0.0 - 1.0
    description: str                      # What is missing
    suggested_tests: list[str]            # Short test descriptions
    fragility_context: str | None = None  # From TestHealer data
    change_frequency: int = 0             # How often this code changes

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "entity_type": self.entity_type,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "description": self.description,
            "suggested_tests": self.suggested_tests,
            "fragility_context": self.fragility_context or "",
            "change_frequency": self.change_frequency,
        }


@dataclass
class CoverageAssessment:
    """Aggregated coverage assessment produced by _assess_coverage."""

    test_coverage_pct: float = 0.0
    requirement_coverage_pct: float = 0.0
    total_testable_surfaces: int = 0
    tested_surfaces: int = 0
    untested_surfaces: int = 0
    total_requirements: int = 0
    covered_requirements: int = 0
    critical_paths_covered: int = 0
    critical_paths_total: int = 0
    risk_areas: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_coverage_pct": self.test_coverage_pct,
            "requirement_coverage_pct": self.requirement_coverage_pct,
            "total_testable_surfaces": self.total_testable_surfaces,
            "tested_surfaces": self.tested_surfaces,
            "untested_surfaces": self.untested_surfaces,
            "total_requirements": self.total_requirements,
            "covered_requirements": self.covered_requirements,
            "critical_paths_covered": self.critical_paths_covered,
            "critical_paths_total": self.critical_paths_total,
            "risk_areas": self.risk_areas,
        }


@dataclass
class UserFlow:
    """A discovered user flow from AutoDiscovery or cached Cognee data."""

    name: str
    description: str
    steps: list[str]
    priority: str = "medium"
    entry_url: str = ""
    related_pages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "priority": self.priority,
            "entry_url": self.entry_url,
            "related_pages": self.related_pages,
        }


@dataclass
class QualityScoreBreakdown:
    """Breakdown of the 0-100 quality score computation."""

    test_coverage_pct: float = 0.0        # 30% weight
    requirement_coverage_pct: float = 0.0  # 20% weight
    test_health_score: float = 0.0         # 25% weight
    code_quality_score: float = 0.0        # 25% weight
    critical_gap_penalty: float = 0.0
    high_gap_penalty: float = 0.0
    raw_score: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_coverage_pct": self.test_coverage_pct,
            "requirement_coverage_pct": self.requirement_coverage_pct,
            "test_health_score": self.test_health_score,
            "code_quality_score": self.code_quality_score,
            "critical_gap_penalty": self.critical_gap_penalty,
            "high_gap_penalty": self.high_gap_penalty,
            "raw_score": self.raw_score,
            "final_score": self.final_score,
        }


@dataclass
class Recommendation:
    """An actionable recommendation from the QA engineer."""

    type: str         # RecommendationType value
    priority: str     # critical / high / medium / low
    description: str
    effort: str       # EffortSize value
    impact: str       # Description of expected impact

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "priority": self.priority,
            "description": self.description,
            "effort": self.effort,
            "impact": self.impact,
        }


@dataclass
class QAAnalysisResult:
    """Complete result from a QA Engineer analysis run."""

    job_id: str
    analysis_type: str
    org_id: str
    project_id: str
    started_at: str = ""
    completed_at: str = ""
    duration_ms: int = 0

    # Sub-results
    codebase_summary: str = ""
    testable_surfaces: list[dict[str, Any]] = field(default_factory=list)
    user_flows: list[UserFlow] = field(default_factory=list)
    coverage: CoverageAssessment | None = None
    gaps: list[CoverageGap] = field(default_factory=list)
    quality_score: QualityScoreBreakdown | None = None
    recommendations: list[Recommendation] = field(default_factory=list)

    # Generated tests (if any gap-filling was triggered)
    generated_tests_count: int = 0
    generated_tests_job_id: str | None = None

    # Cognee tracking
    cognee_namespaces: list[str] = field(default_factory=list)
    kafka_events_emitted: list[str] = field(default_factory=list)

    # Cost tracking
    ai_cost: float = 0.0
    ai_calls: int = 0

    # Comparison (for incremental / compare)
    delta_from_previous: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "analysis_type": self.analysis_type,
            "org_id": self.org_id,
            "project_id": self.project_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "codebase_summary": self.codebase_summary,
            "testable_surfaces_count": len(self.testable_surfaces),
            "user_flows_count": len(self.user_flows),
            "user_flows": [f.to_dict() for f in self.user_flows],
            "coverage": self.coverage.to_dict() if self.coverage else {},
            "gaps_count": len(self.gaps),
            "gaps": [g.to_dict() for g in self.gaps],
            "quality_score": self.quality_score.to_dict() if self.quality_score else {},
            "recommendations": [r.to_dict() for r in self.recommendations],
            "generated_tests_count": self.generated_tests_count,
            "generated_tests_job_id": self.generated_tests_job_id or "",
            "cognee_namespaces": self.cognee_namespaces,
            "kafka_events_emitted": self.kafka_events_emitted,
            "ai_cost": self.ai_cost,
            "ai_calls": self.ai_calls,
            "delta_from_previous": self.delta_from_previous,
        }


# =============================================================================
# Constants
# =============================================================================

# Risk keywords used to classify gaps
_CRITICAL_KEYWORDS = frozenset({
    "auth", "authentication", "login", "password", "payment", "billing",
    "charge", "checkout", "encryption", "token", "session", "permission",
    "role", "admin", "delete", "destroy", "drop", "migration", "transaction",
    "transfer", "refund", "pii", "gdpr", "hipaa", "compliance", "secret",
})

_HIGH_KEYWORDS = frozenset({
    "signup", "register", "onboarding", "dashboard", "profile", "settings",
    "notification", "email", "webhook", "api", "endpoint", "search",
    "filter", "upload", "download", "export", "import", "integration",
    "sync", "cache", "queue", "worker",
})


# =============================================================================
# QAEngineerAgent
# =============================================================================


class QAEngineerAgent(BaseAgent):
    """
    Autonomous QA Engineer meta-orchestrator.

    Composes the full Argus agent fleet to deliver end-to-end quality
    analysis including codebase understanding, coverage assessment,
    risk-scored gap identification, automatic test generation for
    critical gaps, and AI-powered recommendations.

    Analysis types:
        full         -- Complete repo analysis (codebase + flows + coverage +
                        gaps + recommendations + optional test generation)
        pr_review    -- Focused on PR changes (composes MRAnalyzerAgent +
                        coverage delta from last snapshot)
        coverage_only -- Just coverage assessment (uses cached codebase
                        analysis from Cognee)
        incremental  -- Delta from last analysis (queries previous Cognee
                        state and computes diff)

    Usage:
        ```python
        agent = QAEngineerAgent()

        result = await agent.analyze(
            org_id="org_acme",
            project_id="proj_webapp",
            analysis_type=AnalysisType.FULL,
            trigger_context={"commit_sha": "abc123", "branch": "main"},
        )

        print(f"Quality: {result.quality_score.final_score:.1f}/100")
        print(f"Gaps: {len(result.gaps)} ({sum(1 for g in result.gaps if g.risk_level == 'critical')} critical)")
        print(f"AI cost: ${result.ai_cost:.4f}")
        ```
    """

    # Meta-orchestrators do moderate-complexity routing
    DEFAULT_TASK_TYPE = TaskType.CODE_ANALYSIS

    # RAP-231: Agent capabilities for A2A discovery
    CAPABILITIES = [
        AgentCapability.CODE_ANALYSIS,
        AgentCapability.TEST_GENERATION,
    ]

    # =========================================================================
    # BaseAgent abstract method implementations
    # =========================================================================

    def _get_system_prompt(self) -> str:
        """System prompt for the autonomous QA engineer."""
        return (
            "You are an expert autonomous QA engineer responsible for ensuring "
            "comprehensive test coverage and code quality for a software project. "
            "You analyze codebases, identify testable surfaces, assess coverage, "
            "detect gaps, and generate actionable recommendations.\n\n"
            "Your analysis should be thorough, risk-aware, and prioritized. "
            "When identifying gaps, consider:\n"
            "- Authentication/authorization flows (CRITICAL risk)\n"
            "- Payment/billing/data-integrity paths (CRITICAL risk)\n"
            "- Core user journeys (HIGH risk)\n"
            "- API endpoint coverage (HIGH risk)\n"
            "- Edge cases and error handling (MEDIUM risk)\n"
            "- Cosmetic/UI-only gaps (LOW risk)\n\n"
            "Always output valid JSON when asked. Be concise but comprehensive."
        )

    async def execute(self, **kwargs) -> AgentResult:
        """Standard BaseAgent execute entry point.

        Delegates to :meth:`analyze` with kwargs forwarded.  This allows the
        agent to be used both via the supervisor graph (which calls execute)
        and directly via ``analyze()``.
        """
        org_id: str = kwargs.get("org_id", "")
        project_id: str = kwargs.get("project_id", "")
        analysis_type_raw: str = kwargs.get("analysis_type", AnalysisType.FULL.value)
        trigger_context: dict = kwargs.get("trigger_context", {})
        job_id: str = kwargs.get("job_id", str(uuid4()))

        try:
            analysis_type = AnalysisType(analysis_type_raw)
        except ValueError:
            analysis_type = AnalysisType.FULL

        start = time.time()
        try:
            result = await self.analyze(
                org_id=org_id,
                project_id=project_id,
                analysis_type=analysis_type,
                trigger_context=trigger_context,
                job_id=job_id,
            )
            duration_ms = int((time.time() - start) * 1000)
            return AgentResult(
                success=True,
                data=result.to_dict(),
                cost=result.ai_cost,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((time.time() - start) * 1000)
            self.log.error("QA analysis failed", error=str(exc))
            return AgentResult(
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
            )

    # =========================================================================
    # Main entry point
    # =========================================================================

    async def analyze(
        self,
        org_id: str,
        project_id: str,
        analysis_type: AnalysisType = AnalysisType.FULL,
        trigger_context: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> QAAnalysisResult:
        """Main entry point for a full QA analysis run.

        Orchestrates sub-agents, computes quality scores, identifies gaps,
        stores artifacts in Cognee, emits Kafka events, and updates the
        ``qa_analysis_jobs`` table in Supabase.

        Args:
            org_id: Organization ID for multi-tenant isolation.
            project_id: Project ID being analyzed.
            analysis_type: Depth / scope of the analysis.
            trigger_context: Contextual data from the trigger source.
                May contain ``commit_sha``, ``branch``, ``pr_number``,
                ``changed_files``, ``app_url``, etc.
            job_id: Optional pre-assigned job ID (generated if None).

        Returns:
            :class:`QAAnalysisResult` with all sub-results aggregated.
        """
        trigger_context = trigger_context or {}
        job_id = job_id or str(uuid4())
        started_at = datetime.now(timezone.utc).isoformat()

        self.log.info(
            "QA analysis started",
            org_id=org_id,
            project_id=project_id,
            analysis_type=analysis_type.value,
            job_id=job_id,
        )

        result = QAAnalysisResult(
            job_id=job_id,
            analysis_type=analysis_type.value,
            org_id=org_id,
            project_id=project_id,
            started_at=started_at,
        )

        # Mark job as running
        await self._update_job(job_id, {
            "status": "running",
            "analysis_type": analysis_type.value,
            "org_id": org_id,
            "project_id": project_id,
            "started_at": started_at,
        })

        try:
            # -----------------------------------------------------------------
            # 1. Query Cognee for cross-agent context
            # -----------------------------------------------------------------
            cognee_context = await self._query_cognee_context(
                org_id, project_id
            )

            # -----------------------------------------------------------------
            # 2. Codebase analysis (skipped for coverage_only / incremental
            #    if we have a cached snapshot)
            # -----------------------------------------------------------------
            codebase_analysis: dict[str, Any] = {}
            if analysis_type == AnalysisType.PR_REVIEW:
                codebase_analysis = await self._analyze_pr(
                    org_id, project_id, trigger_context, cognee_context
                )
            elif analysis_type in (
                AnalysisType.COVERAGE_ONLY,
                AnalysisType.INCREMENTAL,
            ):
                # Try cached codebase analysis from Cognee
                cached = cognee_context.get("previous_coverage_analysis")
                if cached and isinstance(cached, dict):
                    codebase_analysis = cached.get("codebase_analysis", {})
                    self.log.info("Using cached codebase analysis from Cognee")
                if not codebase_analysis:
                    codebase_analysis = await self._analyze_codebase(
                        org_id, project_id, trigger_context
                    )
            else:
                # AnalysisType.FULL
                codebase_analysis = await self._analyze_codebase(
                    org_id, project_id, trigger_context
                )

            result.codebase_summary = codebase_analysis.get("summary", "")
            result.testable_surfaces = codebase_analysis.get(
                "testable_surfaces", []
            )

            # -----------------------------------------------------------------
            # 3. Discover user flows (skip for PR review)
            # -----------------------------------------------------------------
            user_flows: list[UserFlow] = []
            app_url = trigger_context.get("app_url", "")
            if analysis_type != AnalysisType.PR_REVIEW:
                user_flows = await self._discover_user_flows(
                    org_id, project_id, app_url, cognee_context
                )
            result.user_flows = user_flows

            # -----------------------------------------------------------------
            # 4. Assess coverage
            # -----------------------------------------------------------------
            coverage = await self._assess_coverage(
                org_id,
                project_id,
                codebase_analysis,
                user_flows,
                cognee_context,
            )
            result.coverage = coverage

            # -----------------------------------------------------------------
            # 5. Identify gaps with risk scoring
            # -----------------------------------------------------------------
            gaps = await self._identify_gaps(
                org_id,
                project_id,
                coverage,
                codebase_analysis,
                cognee_context,
            )
            result.gaps = gaps

            # -----------------------------------------------------------------
            # 6. Optionally generate tests for critical/high gaps
            # -----------------------------------------------------------------
            config = trigger_context.get("testgen_config", {})
            auto_generate = trigger_context.get("auto_generate_tests", False)
            critical_high_gaps = [
                g for g in gaps
                if g.risk_level in (RiskLevel.CRITICAL.value, RiskLevel.HIGH.value)
            ]
            if auto_generate and critical_high_gaps:
                gen_count, gen_job_id = await self._generate_tests_for_gaps(
                    org_id, project_id, critical_high_gaps, config
                )
                result.generated_tests_count = gen_count
                result.generated_tests_job_id = gen_job_id

            # -----------------------------------------------------------------
            # 7. Compute quality score
            # -----------------------------------------------------------------
            health_data = cognee_context.get("test_health", {})
            quality = self._compute_quality_score(
                coverage, gaps, health_data, codebase_analysis
            )
            result.quality_score = quality

            # -----------------------------------------------------------------
            # 8. Generate recommendations
            # -----------------------------------------------------------------
            recommendations = await self._generate_recommendations(
                codebase_analysis=codebase_analysis,
                coverage=coverage,
                gaps=gaps,
                quality=quality,
                cognee_context=cognee_context,
            )
            result.recommendations = recommendations

            # -----------------------------------------------------------------
            # 9. Delta comparison for incremental analysis
            # -----------------------------------------------------------------
            if analysis_type == AnalysisType.INCREMENTAL:
                previous = cognee_context.get("previous_coverage_analysis")
                if previous and isinstance(previous, dict):
                    result.delta_from_previous = self._compute_delta(
                        previous, result
                    )

            # -----------------------------------------------------------------
            # 10. Persist to Cognee + emit events + update job
            # -----------------------------------------------------------------
            result.completed_at = datetime.now(timezone.utc).isoformat()
            result.duration_ms = int(
                (time.time() - _iso_to_epoch(started_at)) * 1000
            )
            result.ai_cost = self._usage.total_cost
            result.ai_calls = self._usage.total_calls

            namespaces = await self._store_in_cognee(org_id, project_id, result)
            result.cognee_namespaces = namespaces

            events = await self._emit_events(org_id, project_id, job_id, result)
            result.kafka_events_emitted = events

            await self._update_job(job_id, {
                "status": "completed",
                "completed_at": result.completed_at,
                "duration_ms": result.duration_ms,
                "quality_score": quality.final_score,
                "gap_count": len(gaps),
                "critical_gaps": sum(
                    1 for g in gaps if g.risk_level == RiskLevel.CRITICAL.value
                ),
                "result_json": json.dumps(result.to_dict()),
            })

            self.log.info(
                "QA analysis completed",
                job_id=job_id,
                quality_score=quality.final_score,
                gaps_total=len(gaps),
                duration_ms=result.duration_ms,
                ai_cost=f"${result.ai_cost:.4f}",
            )

            return result

        except Exception as exc:
            await self._update_job(job_id, {
                "status": "failed",
                "error": str(exc)[:500],
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            raise

    # =========================================================================
    # Sub-orchestration: Codebase analysis
    # =========================================================================

    async def _analyze_codebase(
        self,
        org_id: str,
        project_id: str,
        trigger_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Compose CodeAnalyzerAgent to understand the codebase.

        Returns a dict with keys: summary, testable_surfaces, framework,
        language, test_files, etc.
        """
        self.log.info("Composing CodeAnalyzerAgent for codebase analysis")

        try:
            from ..code_analyzer import CodeAnalyzerAgent

            analyzer = CodeAnalyzerAgent()
            codebase_path = trigger_context.get("codebase_path", "")
            repo_url = trigger_context.get("repo_url", "")

            agent_result = await analyzer.execute(
                codebase_path=codebase_path,
                repo_url=repo_url,
            )

            if agent_result.success and agent_result.data:
                data = agent_result.data
                # Normalize to dict regardless of return type
                if hasattr(data, "to_dict"):
                    return data.to_dict()
                elif isinstance(data, dict):
                    return data
                else:
                    return {"summary": str(data), "testable_surfaces": []}

            self.log.warning(
                "CodeAnalyzerAgent returned no data, using AI fallback",
                error=agent_result.error,
            )

        except Exception as exc:
            self.log.warning(
                "CodeAnalyzerAgent composition failed, using AI fallback",
                error=str(exc),
            )

        # Fallback: use _call_ai directly
        return await self._ai_analyze_codebase(trigger_context)

    async def _ai_analyze_codebase(
        self,
        trigger_context: dict[str, Any],
    ) -> dict[str, Any]:
        """AI fallback for codebase analysis when CodeAnalyzerAgent fails."""
        context_str = json.dumps(trigger_context, default=str)[:4000]

        response = await self._call_ai(
            messages=[{
                "role": "user",
                "content": (
                    "Analyze the following project context and identify testable "
                    "surfaces. Return valid JSON with keys: summary (string), "
                    "testable_surfaces (list of {type, name, path, priority, "
                    "description, test_scenarios}), framework (string or null), "
                    "language (string or null), test_files_count (int).\n\n"
                    f"Context:\n{context_str}"
                ),
            }],
            task_type=TaskType.CODE_ANALYSIS,
            required_capabilities=[AICapability.REASONING],
            max_cost=0.10,
            max_tokens=4096,
            json_mode=True,
        )

        return _safe_json_parse(response.content, default={
            "summary": "Unable to analyze codebase",
            "testable_surfaces": [],
        })

    # =========================================================================
    # Sub-orchestration: PR-focused analysis
    # =========================================================================

    async def _analyze_pr(
        self,
        org_id: str,
        project_id: str,
        trigger_context: dict[str, Any],
        cognee_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Compose MRAnalyzerAgent for PR-scoped analysis.

        Returns codebase_analysis dict focused on changed files.
        """
        self.log.info("Composing MRAnalyzerAgent for PR review")

        try:
            from ..mr_analyzer import MRAnalyzerAgent

            analyzer = MRAnalyzerAgent()

            changed_files = trigger_context.get("changed_files", [])
            pr_number = trigger_context.get("pr_number")
            repo_url = trigger_context.get("repo_url", "")
            branch = trigger_context.get("branch", "main")

            agent_result = await analyzer.execute(
                changed_files=changed_files,
                pr_number=pr_number,
                repo_url=repo_url,
                branch=branch,
            )

            if agent_result.success and agent_result.data:
                data = agent_result.data
                if hasattr(data, "to_dict"):
                    raw = data.to_dict()
                elif isinstance(data, dict):
                    raw = data
                else:
                    raw = {"summary": str(data)}

                # Reshape MR analysis into codebase_analysis structure
                return {
                    "summary": raw.get("summary", "PR analysis"),
                    "testable_surfaces": raw.get("affected_surfaces", []),
                    "changed_files": changed_files,
                    "pr_number": pr_number,
                    "test_suggestions": raw.get("test_suggestions", []),
                }

        except Exception as exc:
            self.log.warning(
                "MRAnalyzerAgent composition failed",
                error=str(exc),
            )

        # Minimal fallback
        return {
            "summary": f"PR analysis for {trigger_context.get('pr_number', 'unknown')}",
            "testable_surfaces": [],
            "changed_files": trigger_context.get("changed_files", []),
        }

    # =========================================================================
    # Sub-orchestration: User flow discovery
    # =========================================================================

    async def _discover_user_flows(
        self,
        org_id: str,
        project_id: str,
        app_url: str,
        cognee_context: dict[str, Any],
    ) -> list[UserFlow]:
        """Compose AutoDiscovery / QuickDiscover to find user flows.

        Falls back to cached Cognee flows if no app_url is available or
        if discovery fails.
        """
        # Check Cognee cache first
        cached_flows = cognee_context.get("user_flows", [])

        if not app_url:
            self.log.info(
                "No app_url provided, using cached user flows",
                cached_count=len(cached_flows),
            )
            return [
                UserFlow(**f) if isinstance(f, dict) else f
                for f in cached_flows
            ]

        self.log.info(
            "Composing AutoDiscovery for user flow detection",
            app_url=app_url,
        )

        try:
            from ..auto_discovery import QuickDiscover

            discoverer = QuickDiscover(app_url=app_url)
            critical_flows = await discoverer.discover_critical_flows()

            flows: list[UserFlow] = []
            for raw in critical_flows:
                flows.append(UserFlow(
                    name=raw.get("name", "Unknown flow"),
                    description=raw.get("description", ""),
                    steps=raw.get("steps", []),
                    priority=raw.get("priority", "medium"),
                    entry_url=raw.get("url", app_url),
                    related_pages=raw.get("related_pages", []),
                ))

            if flows:
                self.log.info(
                    "User flows discovered",
                    count=len(flows),
                )
                return flows

        except Exception as exc:
            self.log.warning(
                "AutoDiscovery composition failed",
                error=str(exc),
            )

        # Fallback to cached
        if cached_flows:
            self.log.info(
                "Using cached user flows from Cognee",
                count=len(cached_flows),
            )
            return [
                UserFlow(**f) if isinstance(f, dict) else f
                for f in cached_flows
            ]

        # Final fallback: AI-generated flows from codebase context
        return await self._ai_infer_user_flows(app_url)

    async def _ai_infer_user_flows(self, app_url: str) -> list[UserFlow]:
        """Use AI to infer likely user flows when discovery is unavailable."""
        response = await self._call_ai(
            messages=[{
                "role": "user",
                "content": (
                    f"Given a web application at {app_url}, infer the most "
                    "likely critical user flows. Return a JSON array of objects "
                    "with keys: name, description, steps (array of strings), "
                    "priority (critical/high/medium/low). "
                    "Include flows like: login, signup, core feature usage, "
                    "settings, data management."
                ),
            }],
            task_type=TaskType.CODE_ANALYSIS,
            max_cost=0.05,
            max_tokens=2048,
            json_mode=True,
        )

        raw = _safe_json_parse(response.content, default=[])
        items = raw if isinstance(raw, list) else raw.get("flows", [])

        return [
            UserFlow(
                name=f.get("name", "Unknown"),
                description=f.get("description", ""),
                steps=f.get("steps", []),
                priority=f.get("priority", "medium"),
                entry_url=app_url,
            )
            for f in items
            if isinstance(f, dict)
        ]

    # =========================================================================
    # Coverage assessment
    # =========================================================================

    async def _assess_coverage(
        self,
        org_id: str,
        project_id: str,
        codebase_analysis: dict[str, Any],
        user_flows: list[UserFlow],
        cognee_context: dict[str, Any],
    ) -> CoverageAssessment:
        """Produce a risk-scored coverage assessment using AI.

        Combines codebase analysis, discovered flows, and Cognee context
        (healing patterns, selector fragility, generated tests) to compute
        accurate coverage percentages.
        """
        surfaces = codebase_analysis.get("testable_surfaces", [])
        test_files_count = codebase_analysis.get("test_files_count", 0)
        existing_tests = cognee_context.get("generated_tests", [])
        healing_patterns = cognee_context.get("healing_patterns", [])

        # Build concise context for AI
        surfaces_summary = json.dumps(surfaces[:50], default=str)[:3000]
        flows_summary = json.dumps(
            [f.to_dict() for f in user_flows[:20]], default=str
        )[:2000]
        existing_tests_summary = json.dumps(
            existing_tests[:30], default=str
        )[:1500]

        prompt = (
            "Assess test coverage for this project. Analyze the testable surfaces, "
            "user flows, and existing tests to compute coverage metrics.\n\n"
            "Return valid JSON with keys:\n"
            "- test_coverage_pct (float 0-100): percentage of testable surfaces with tests\n"
            "- requirement_coverage_pct (float 0-100): percentage of requirements covered\n"
            "- total_testable_surfaces (int)\n"
            "- tested_surfaces (int)\n"
            "- untested_surfaces (int)\n"
            "- total_requirements (int): inferred from flows + surfaces\n"
            "- covered_requirements (int)\n"
            "- critical_paths_covered (int)\n"
            "- critical_paths_total (int)\n"
            "- risk_areas (array of {area: string, risk: string, reason: string})\n\n"
            f"Testable surfaces ({len(surfaces)} total):\n{surfaces_summary}\n\n"
            f"User flows ({len(user_flows)} total):\n{flows_summary}\n\n"
            f"Existing tests ({len(existing_tests)} found, "
            f"{test_files_count} test files):\n{existing_tests_summary}\n\n"
            f"Healing patterns (areas that keep breaking): "
            f"{json.dumps(healing_patterns[:10], default=str)[:500]}"
        )

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            required_capabilities=[AICapability.REASONING],
            max_cost=0.10,
            max_tokens=4096,
            json_mode=True,
        )

        raw = _safe_json_parse(response.content, default={})

        return CoverageAssessment(
            test_coverage_pct=float(raw.get("test_coverage_pct", 0)),
            requirement_coverage_pct=float(raw.get("requirement_coverage_pct", 0)),
            total_testable_surfaces=int(raw.get("total_testable_surfaces", len(surfaces))),
            tested_surfaces=int(raw.get("tested_surfaces", 0)),
            untested_surfaces=int(raw.get("untested_surfaces", len(surfaces))),
            total_requirements=int(raw.get("total_requirements", 0)),
            covered_requirements=int(raw.get("covered_requirements", 0)),
            critical_paths_covered=int(raw.get("critical_paths_covered", 0)),
            critical_paths_total=int(raw.get("critical_paths_total", 0)),
            risk_areas=raw.get("risk_areas", []),
        )

    # =========================================================================
    # Gap identification with risk scoring
    # =========================================================================

    async def _identify_gaps(
        self,
        org_id: str,
        project_id: str,
        coverage: CoverageAssessment,
        codebase_analysis: dict[str, Any],
        cognee_context: dict[str, Any],
    ) -> list[CoverageGap]:
        """Identify coverage gaps and assign risk scores.

        Enriches gaps with Cognee context:
        - ``selector_fragility``: areas with fragile selectors from TestHealer
        - ``healing_patterns``: areas that keep breaking
        - ``change_frequency``: how often code changes (from code analyzer)
        """
        fragility_data = cognee_context.get("selector_fragility", [])
        healing_patterns = cognee_context.get("healing_patterns", [])
        surfaces = codebase_analysis.get("testable_surfaces", [])

        # Build fragility lookup
        fragility_lookup: dict[str, str] = {}
        for item in fragility_data:
            if isinstance(item, dict):
                entity = item.get("entity", item.get("selector", ""))
                fragility_lookup[entity] = item.get("reason", "Fragile selector detected")

        # Build healing frequency lookup
        healing_lookup: dict[str, int] = {}
        for pattern in healing_patterns:
            if isinstance(pattern, dict):
                entity = pattern.get("entity", pattern.get("test_id", ""))
                healing_lookup[entity] = healing_lookup.get(entity, 0) + 1

        # Ask AI to identify gaps
        surfaces_str = json.dumps(surfaces[:60], default=str)[:4000]
        coverage_str = json.dumps(coverage.to_dict(), default=str)
        risk_areas_str = json.dumps(coverage.risk_areas[:20], default=str)[:1500]

        prompt = (
            "Identify specific coverage gaps in this project. For each gap, "
            "determine the risk level based on what the untested code does.\n\n"
            "Risk levels:\n"
            "- critical: Auth, payment, data integrity, security, PII/compliance\n"
            "- high: Core user flows, frequently-changed areas, API endpoints\n"
            "- medium: Standard features, moderate-use paths\n"
            "- low: Edge cases, cosmetic, rarely-used features\n\n"
            "Return a JSON array of objects with keys:\n"
            "- entity (string): function/endpoint/component name\n"
            "- entity_type (string): function, endpoint, component, page\n"
            "- risk_level (string): critical, high, medium, low\n"
            "- risk_score (float 0.0-1.0): numeric risk\n"
            "- description (string): what is missing\n"
            "- suggested_tests (array of strings): 1-3 test descriptions\n"
            "- change_frequency (int): estimated changes per month\n\n"
            f"Testable surfaces:\n{surfaces_str}\n\n"
            f"Coverage assessment:\n{coverage_str}\n\n"
            f"Risk areas identified:\n{risk_areas_str}"
        )

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            required_capabilities=[AICapability.REASONING],
            max_cost=0.10,
            max_tokens=8192,
            json_mode=True,
        )

        raw = _safe_json_parse(response.content, default=[])
        items = raw if isinstance(raw, list) else raw.get("gaps", [])

        gaps: list[CoverageGap] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            entity = item.get("entity", "unknown")
            entity_lower = entity.lower()

            # Override risk_level using keyword heuristics when AI under-rates
            ai_risk = item.get("risk_level", "medium").lower()
            if ai_risk not in ("critical",) and any(
                kw in entity_lower for kw in _CRITICAL_KEYWORDS
            ):
                ai_risk = "critical"
            elif ai_risk not in ("critical", "high") and any(
                kw in entity_lower for kw in _HIGH_KEYWORDS
            ):
                ai_risk = "high"

            # Compute risk_score: base from AI + boost for fragility/healing
            base_score = float(item.get("risk_score", _risk_level_to_score(ai_risk)))
            fragility_note = fragility_lookup.get(entity)
            healing_freq = healing_lookup.get(entity, 0)

            if fragility_note:
                base_score = min(1.0, base_score + 0.1)
            if healing_freq > 0:
                base_score = min(1.0, base_score + 0.05 * healing_freq)

            gap = CoverageGap(
                entity=entity,
                entity_type=item.get("entity_type", "function"),
                risk_level=ai_risk,
                risk_score=round(base_score, 3),
                description=item.get("description", "No description"),
                suggested_tests=item.get("suggested_tests", []),
                fragility_context=fragility_note,
                change_frequency=item.get("change_frequency", healing_freq),
            )
            gaps.append(gap)

        # Sort by risk_score descending
        gaps.sort(key=lambda g: g.risk_score, reverse=True)

        self.log.info(
            "Coverage gaps identified",
            total=len(gaps),
            critical=sum(1 for g in gaps if g.risk_level == RiskLevel.CRITICAL.value),
            high=sum(1 for g in gaps if g.risk_level == RiskLevel.HIGH.value),
            medium=sum(1 for g in gaps if g.risk_level == RiskLevel.MEDIUM.value),
            low=sum(1 for g in gaps if g.risk_level == RiskLevel.LOW.value),
        )

        return gaps

    # =========================================================================
    # Test generation for gaps
    # =========================================================================

    async def _generate_tests_for_gaps(
        self,
        org_id: str,
        project_id: str,
        gaps: list[CoverageGap],
        config: dict[str, Any],
    ) -> tuple[int, str | None]:
        """Compose TestGenAgent to auto-generate tests for critical/high gaps.

        Args:
            org_id: Organization ID.
            project_id: Project ID.
            gaps: Gaps to generate tests for (already filtered to critical/high).
            config: TestGen configuration (framework, etc.).

        Returns:
            Tuple of (tests_generated_count, testgen_job_id).
        """
        self.log.info(
            "Composing TestGenAgent for gap filling",
            gap_count=len(gaps),
        )

        try:
            from ..testgen_agent import TestGenAgent, SourceType, TestGenConfig
            from ..testgen import TestFramework

            # Build requirements text from gaps
            requirements_parts: list[str] = []
            for gap in gaps[:20]:  # Limit to top 20 gaps
                tests_str = "; ".join(gap.suggested_tests[:3])
                requirements_parts.append(
                    f"- [{gap.risk_level.upper()}] {gap.entity} "
                    f"({gap.entity_type}): {gap.description}. "
                    f"Suggested tests: {tests_str}"
                )

            requirements_text = (
                "Generate tests for the following coverage gaps identified "
                "by QA analysis:\n\n" + "\n".join(requirements_parts)
            )

            # Determine framework from config
            framework_str = config.get("framework", "playwright")
            try:
                framework = TestFramework(framework_str)
            except ValueError:
                framework = TestFramework.PLAYWRIGHT

            testgen = TestGenAgent()
            testgen_job_id = str(uuid4())

            gen_result = await testgen.generate(
                org_id=org_id,
                project_id=project_id,
                source_type=SourceType.REQUIREMENTS_TEXT,
                source_data=requirements_text,
                config=TestGenConfig(framework=framework),
                job_id=testgen_job_id,
            )

            self.log.info(
                "TestGenAgent completed gap filling",
                tests_generated=gen_result.tests_count,
                quality=gen_result.quality_score,
            )

            return gen_result.tests_count, testgen_job_id

        except Exception as exc:
            self.log.warning(
                "TestGenAgent composition failed (non-fatal)",
                error=str(exc),
            )
            return 0, None

    # =========================================================================
    # Quality score computation
    # =========================================================================

    def _compute_quality_score(
        self,
        coverage: CoverageAssessment,
        gaps: list[CoverageGap],
        health_data: dict[str, Any],
        codebase_analysis: dict[str, Any],
    ) -> QualityScoreBreakdown:
        """Compute the 0-100 quality score with weighted multi-signal formula.

        Weights:
            test_coverage_pct    : 30%
            requirement_coverage : 20%
            test_health_score    : 25%
            code_quality_score   : 25%

        Penalties:
            -5 per CRITICAL gap
            -2 per HIGH gap
        """
        # Normalize coverage percentages to 0-100 scale
        test_cov = min(100.0, max(0.0, coverage.test_coverage_pct))
        req_cov = min(100.0, max(0.0, coverage.requirement_coverage_pct))

        # Test health from TestHealer/FlakyDetector data
        # Default to 70 if no data available
        test_health = float(health_data.get("health_score", 70.0))
        test_health = min(100.0, max(0.0, test_health))

        # Code quality from codebase analysis
        # Look for maintainability / complexity metrics
        code_quality = float(
            codebase_analysis.get("code_quality_score", 0)
            or codebase_analysis.get("maintainability_score", 0)
            or 65.0  # Default
        )
        code_quality = min(100.0, max(0.0, code_quality))

        # Weighted sum
        raw_score = (
            test_cov * 0.30
            + req_cov * 0.20
            + test_health * 0.25
            + code_quality * 0.25
        )

        # Penalties
        critical_count = sum(
            1 for g in gaps if g.risk_level == RiskLevel.CRITICAL.value
        )
        high_count = sum(
            1 for g in gaps if g.risk_level == RiskLevel.HIGH.value
        )

        critical_penalty = critical_count * 5.0
        high_penalty = high_count * 2.0
        total_penalty = critical_penalty + high_penalty

        final_score = max(0.0, min(100.0, raw_score - total_penalty))

        return QualityScoreBreakdown(
            test_coverage_pct=round(test_cov, 2),
            requirement_coverage_pct=round(req_cov, 2),
            test_health_score=round(test_health, 2),
            code_quality_score=round(code_quality, 2),
            critical_gap_penalty=round(critical_penalty, 2),
            high_gap_penalty=round(high_penalty, 2),
            raw_score=round(raw_score, 2),
            final_score=round(final_score, 2),
        )

    # =========================================================================
    # AI-powered recommendations
    # =========================================================================

    async def _generate_recommendations(
        self,
        codebase_analysis: dict[str, Any],
        coverage: CoverageAssessment,
        gaps: list[CoverageGap],
        quality: QualityScoreBreakdown,
        cognee_context: dict[str, Any],
    ) -> list[Recommendation]:
        """Generate prioritized, actionable recommendations using AI.

        Enriches recommendations with Cognee context (healing history,
        flaky tests, selector fragility).
        """
        healing_patterns = cognee_context.get("healing_patterns", [])
        selector_fragility = cognee_context.get("selector_fragility", [])

        # Summarize inputs for AI
        gap_summary = json.dumps(
            [g.to_dict() for g in gaps[:15]], default=str
        )[:3000]
        quality_summary = json.dumps(quality.to_dict(), default=str)
        healing_summary = json.dumps(healing_patterns[:10], default=str)[:1000]
        fragility_summary = json.dumps(selector_fragility[:10], default=str)[:1000]

        prompt = (
            "Based on this QA analysis, generate prioritized recommendations. "
            "Each recommendation should be actionable and include effort sizing.\n\n"
            "Recommendation types:\n"
            "- add_tests: New tests needed for uncovered areas\n"
            "- fix_flaky: Fix flaky/unreliable tests\n"
            "- reduce_fragility: Make selectors/assertions more robust\n"
            "- add_ci: Add CI/CD pipeline improvements\n"
            "- refactor_tests: Restructure/improve existing tests\n\n"
            "Effort sizes:\n"
            "- small: < 1 day\n"
            "- medium: 1-3 days\n"
            "- large: > 3 days\n\n"
            "Return a JSON array of objects with keys:\n"
            "- type (string): recommendation type from above\n"
            "- priority (string): critical, high, medium, low\n"
            "- description (string): specific actionable description\n"
            "- effort (string): small, medium, large\n"
            "- impact (string): expected impact description\n\n"
            f"Quality score breakdown:\n{quality_summary}\n\n"
            f"Top coverage gaps:\n{gap_summary}\n\n"
            f"Healing patterns (things that keep breaking):\n{healing_summary}\n\n"
            f"Selector fragility data:\n{fragility_summary}"
        )

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            required_capabilities=[AICapability.REASONING],
            max_cost=0.08,
            max_tokens=4096,
            json_mode=True,
        )

        raw = _safe_json_parse(response.content, default=[])
        items = raw if isinstance(raw, list) else raw.get("recommendations", [])

        recommendations: list[Recommendation] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            rec = Recommendation(
                type=item.get("type", RecommendationType.ADD_TESTS.value),
                priority=item.get("priority", "medium"),
                description=item.get("description", ""),
                effort=item.get("effort", EffortSize.MEDIUM.value),
                impact=item.get("impact", ""),
            )
            recommendations.append(rec)

        # Sort by priority (critical first)
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(
            key=lambda r: priority_order.get(r.priority, 4)
        )

        self.log.info(
            "Recommendations generated",
            count=len(recommendations),
        )

        return recommendations

    # =========================================================================
    # Analysis comparison (for incremental / compare)
    # =========================================================================

    async def compare_analyses(
        self,
        org_id: str,
        project_id: str,
        job_id_a: str,
        job_id_b: str,
    ) -> dict[str, Any]:
        """Compare two QA analysis snapshots and compute delta.

        Useful for tracking quality trends over time or comparing
        before/after a refactor.

        Args:
            org_id: Organization ID.
            project_id: Project ID.
            job_id_a: Earlier analysis job ID.
            job_id_b: Later analysis job ID.

        Returns:
            Delta dict with changes in quality score, gap counts,
            coverage, and recommendations.
        """
        self.log.info(
            "Comparing analyses",
            job_a=job_id_a,
            job_b=job_id_b,
        )

        try:
            from ...services.supabase_client import get_supabase_client

            db = get_supabase_client()

            result_a = await db.select(
                "qa_analysis_jobs",
                columns="result_json",
                filters={"id": f"eq.{job_id_a}", "organization_id": f"eq.{org_id}"},
            )
            result_b = await db.select(
                "qa_analysis_jobs",
                columns="result_json",
                filters={"id": f"eq.{job_id_b}", "organization_id": f"eq.{org_id}"},
            )

            data_a = _extract_job_result(result_a)
            data_b = _extract_job_result(result_b)

            if not data_a or not data_b:
                return {"error": "One or both analyses not found"}

            return self._compute_delta(data_a, data_b)

        except Exception as exc:
            self.log.error("Analysis comparison failed", error=str(exc))
            return {"error": str(exc)}

    def _compute_delta(
        self,
        previous: dict[str, Any] | QAAnalysisResult,
        current: dict[str, Any] | QAAnalysisResult,
    ) -> dict[str, Any]:
        """Compute the delta between two analysis results."""
        prev = previous.to_dict() if hasattr(previous, "to_dict") else previous
        curr = current.to_dict() if hasattr(current, "to_dict") else current

        prev_quality = prev.get("quality_score", {})
        curr_quality = curr.get("quality_score", {})

        prev_coverage = prev.get("coverage", {})
        curr_coverage = curr.get("coverage", {})

        return {
            "quality_score_delta": (
                curr_quality.get("final_score", 0)
                - prev_quality.get("final_score", 0)
            ),
            "test_coverage_delta": (
                curr_coverage.get("test_coverage_pct", 0)
                - prev_coverage.get("test_coverage_pct", 0)
            ),
            "requirement_coverage_delta": (
                curr_coverage.get("requirement_coverage_pct", 0)
                - prev_coverage.get("requirement_coverage_pct", 0)
            ),
            "gaps_delta": (
                curr.get("gaps_count", 0) - prev.get("gaps_count", 0)
            ),
            "critical_gaps_previous": sum(
                1 for g in prev.get("gaps", [])
                if isinstance(g, dict) and g.get("risk_level") == "critical"
            ),
            "critical_gaps_current": sum(
                1 for g in curr.get("gaps", [])
                if isinstance(g, dict) and g.get("risk_level") == "critical"
            ),
            "new_gaps": _diff_entities(
                prev.get("gaps", []), curr.get("gaps", [])
            ),
            "resolved_gaps": _diff_entities(
                curr.get("gaps", []), prev.get("gaps", [])
            ),
            "recommendations_count_delta": (
                len(curr.get("recommendations", []))
                - len(prev.get("recommendations", []))
            ),
        }

    # =========================================================================
    # Cognee: cross-agent context retrieval
    # =========================================================================

    async def _query_cognee_context(
        self,
        org_id: str,
        project_id: str,
    ) -> dict[str, Any]:
        """Query Cognee for cross-agent enrichment data.

        Retrieves:
        - selector_fragility: from TestHealer, fragile selectors
        - healing_patterns: areas that keep breaking
        - generated_tests: tests already auto-generated by TestGen
        - previous_coverage_analysis: last coverage snapshot
        - user_flows: cached user flows for incremental analysis
        - test_health: overall test suite health metrics
        """
        context: dict[str, Any] = {
            "selector_fragility": [],
            "healing_patterns": [],
            "generated_tests": [],
            "previous_coverage_analysis": {},
            "user_flows": [],
            "test_health": {},
        }

        try:
            from ...knowledge.cognee_client import get_cognee_client

            cognee = get_cognee_client(org_id=org_id, project_id=project_id)

            # Query selector fragility
            try:
                results = await cognee.search(
                    namespace=("testhealer", "selector_fragility"),
                    query="fragile selectors that frequently break",
                    limit=20,
                )
                if results:
                    context["selector_fragility"] = [
                        r.get("value", r) if isinstance(r, dict) else r
                        for r in results
                    ]
            except Exception:
                self.log.debug("No selector fragility data in Cognee")

            # Query healing patterns
            try:
                results = await cognee.search(
                    namespace=("healing", "patterns"),
                    query="test healing patterns recurring failures",
                    limit=20,
                )
                if results:
                    context["healing_patterns"] = [
                        r.get("value", r) if isinstance(r, dict) else r
                        for r in results
                    ]
            except Exception:
                self.log.debug("No healing patterns in Cognee")

            # Query generated tests
            try:
                results = await cognee.search(
                    namespace=("testgen", "generated_tests"),
                    query="auto-generated test cases",
                    limit=30,
                )
                if results:
                    context["generated_tests"] = [
                        r.get("value", r) if isinstance(r, dict) else r
                        for r in results
                    ]
            except Exception:
                self.log.debug("No generated tests in Cognee")

            # Query previous coverage analysis
            try:
                results = await cognee.search(
                    namespace=("qa_engineer", "coverage_analysis"),
                    query="latest coverage analysis snapshot",
                    limit=1,
                )
                if results:
                    latest = results[0]
                    context["previous_coverage_analysis"] = (
                        latest.get("value", latest)
                        if isinstance(latest, dict)
                        else latest
                    )
            except Exception:
                self.log.debug("No previous coverage analysis in Cognee")

            # Query user flows
            try:
                results = await cognee.search(
                    query="discovered user flows critical paths",
                    namespace=("qa_engineer", "user_flows"),
                    top_k=20,
                )
                if results:
                    context["user_flows"] = [
                        r.get("value", r) if isinstance(r, dict) else r
                        for r in results
                    ]
            except Exception:
                self.log.debug("No cached user flows in Cognee")

            # Query test health
            try:
                results = await cognee.search(
                    query="test suite health score metrics",
                    namespace=("testhealer", "health"),
                    top_k=1,
                )
                if results:
                    latest = results[0]
                    context["test_health"] = (
                        latest.get("value", latest)
                        if isinstance(latest, dict)
                        else latest
                    )
            except Exception:
                self.log.debug("No test health data in Cognee")

            self.log.info(
                "Cognee context loaded",
                selector_fragility=len(context["selector_fragility"]),
                healing_patterns=len(context["healing_patterns"]),
                generated_tests=len(context["generated_tests"]),
                has_previous_analysis=bool(context["previous_coverage_analysis"]),
                cached_flows=len(context["user_flows"]),
            )

        except Exception as exc:
            self.log.warning(
                "Cognee context query failed (proceeding without enrichment)",
                error=str(exc),
            )

        return context

    # =========================================================================
    # Cognee: persistence
    # =========================================================================

    async def _store_in_cognee(
        self,
        org_id: str,
        project_id: str,
        result: QAAnalysisResult,
    ) -> list[str]:
        """Store analysis artifacts in Cognee for cross-session learning.

        Stores in namespaces:
        - qa_engineer/coverage_analysis: full analysis snapshot
        - qa_engineer/user_flows: discovered user flows

        Per MEMORY.md #4: sanitize None values before Cognee storage.

        Args:
            org_id: Organization ID.
            project_id: Project ID.
            result: Complete analysis result.

        Returns:
            List of Cognee namespace strings used.
        """
        namespaces_used: list[str] = []

        try:
            from ...knowledge.cognee_client import get_cognee_client

            cognee = get_cognee_client(org_id=org_id, project_id=project_id)

            # Store coverage analysis snapshot
            ns_coverage = ("qa_engineer", "coverage_analysis")
            try:
                snapshot = _sanitize_for_cognee({
                    "job_id": result.job_id,
                    "analysis_type": result.analysis_type,
                    "timestamp": result.completed_at,
                    "quality_score": (
                        result.quality_score.final_score
                        if result.quality_score else 0.0
                    ),
                    "coverage": result.coverage.to_dict() if result.coverage else {},
                    "gaps_count": len(result.gaps),
                    "critical_gaps": sum(
                        1 for g in result.gaps
                        if g.risk_level == RiskLevel.CRITICAL.value
                    ),
                    "codebase_summary": result.codebase_summary[:500],
                    "codebase_analysis": {
                        "summary": result.codebase_summary[:500],
                        "testable_surfaces": result.testable_surfaces[:30],
                    },
                })

                await cognee.put(
                    namespace=ns_coverage,
                    key=f"coverage_snapshot_{result.job_id}",
                    value=snapshot,
                    embed_text=(
                        f"QA coverage analysis: quality={snapshot.get('quality_score', 0)}, "
                        f"gaps={snapshot.get('gaps_count', 0)}, "
                        f"coverage={json.dumps(snapshot.get('coverage', {}), default=str)[:200]}"
                    ),
                )
                namespaces_used.append("qa_engineer/coverage_analysis")
                self.log.debug("Stored coverage analysis in Cognee")

            except Exception as exc:
                self.log.warning(
                    "Failed to store coverage analysis in Cognee",
                    error=str(exc),
                )

            # Store user flows
            if result.user_flows:
                ns_flows = ("qa_engineer", "user_flows")
                for idx, flow in enumerate(result.user_flows[:30]):
                    try:
                        flow_data = _sanitize_for_cognee(flow.to_dict())
                        await cognee.put(
                            namespace=ns_flows,
                            key=f"flow_{result.job_id}_{idx}",
                            value=flow_data,
                            embed_text=(
                                f"User flow: {flow.name} - {flow.description}. "
                                f"Steps: {', '.join(flow.steps[:5])}"
                            ),
                        )
                    except Exception as exc:
                        self.log.debug(
                            "Failed to store user flow in Cognee",
                            flow_name=flow.name,
                            error=str(exc),
                        )

                namespaces_used.append("qa_engineer/user_flows")
                self.log.debug(
                    "Stored user flows in Cognee",
                    count=len(result.user_flows),
                )

            # Store individual gaps for semantic search
            if result.gaps:
                ns_gaps = ("qa_engineer", "coverage_gaps")
                for idx, gap in enumerate(result.gaps[:50]):
                    try:
                        gap_data = _sanitize_for_cognee(gap.to_dict())
                        await cognee.put(
                            namespace=ns_gaps,
                            key=f"gap_{result.job_id}_{idx}",
                            value=gap_data,
                            embed_text=(
                                f"Coverage gap [{gap.risk_level}]: {gap.entity} "
                                f"({gap.entity_type}) - {gap.description}"
                            ),
                        )
                    except Exception:
                        pass  # Individual gap failures are non-fatal

                namespaces_used.append("qa_engineer/coverage_gaps")

            self.log.info(
                "Cognee storage completed",
                namespaces=namespaces_used,
            )

        except Exception as exc:
            self.log.warning(
                "Cognee storage failed (non-fatal)",
                error=str(exc),
            )

        return namespaces_used

    # =========================================================================
    # Kafka event emission
    # =========================================================================

    async def _emit_events(
        self,
        org_id: str,
        project_id: str,
        job_id: str,
        result: QAAnalysisResult,
    ) -> list[str]:
        """Emit Kafka events for downstream processing.

        Emits:
        - COVERAGE_ANALYZED: summary of coverage analysis
        - COVERAGE_GAPS_IDENTIFIED: gap breakdown with risk scores

        Args:
            org_id: Organization ID.
            project_id: Project ID.
            job_id: Job tracking ID.
            result: Complete analysis result.

        Returns:
            List of event type names emitted.
        """
        emitted: list[str] = []

        try:
            from ...config import get_settings
            from ...events.producer import EventProducer
            from ...events.schemas import (
                CoverageAnalyzedEvent,
                CoverageGapsIdentifiedEvent,
                EventMetadata,
                TenantInfo,
            )
            from ...events.topics import get_topic_for_event

            settings = get_settings()

            if not settings.redpanda_brokers:
                self.log.debug(
                    "Redpanda not configured, skipping event emission"
                )
                return emitted

            tenant = TenantInfo(org_id=org_id, project_id=project_id)
            metadata = EventMetadata(
                source="qa_engineer_agent",
                triggered_by="agent",
            )

            # COVERAGE_ANALYZED event
            coverage_event = CoverageAnalyzedEvent(
                tenant=tenant,
                metadata=metadata,
                job_id=job_id,
                analysis_type=result.analysis_type,
                quality_score=(
                    result.quality_score.final_score
                    if result.quality_score else 0.0
                ),
                test_coverage_pct=(
                    result.coverage.test_coverage_pct
                    if result.coverage else 0.0
                ),
                components_analyzed=len(result.testable_surfaces),
                user_flows_discovered=len(result.user_flows),
                gap_count=len(result.gaps),
                critical_gaps=sum(
                    1 for g in result.gaps
                    if g.risk_level == RiskLevel.CRITICAL.value
                ),
                commit_sha=str(
                    result.to_dict().get("delta_from_previous", {})
                    or {}
                ).get("commit_sha", ""),
                cognee_namespace=(
                    result.cognee_namespaces[0]
                    if result.cognee_namespaces else ""
                ),
            )

            # COVERAGE_GAPS_IDENTIFIED event
            top_gaps = [g.to_dict() for g in result.gaps[:10]]
            gaps_event = CoverageGapsIdentifiedEvent(
                tenant=tenant,
                metadata=metadata,
                job_id=job_id,
                total_gaps=len(result.gaps),
                critical_gaps=sum(
                    1 for g in result.gaps
                    if g.risk_level == RiskLevel.CRITICAL.value
                ),
                high_gaps=sum(
                    1 for g in result.gaps
                    if g.risk_level == RiskLevel.HIGH.value
                ),
                medium_gaps=sum(
                    1 for g in result.gaps
                    if g.risk_level == RiskLevel.MEDIUM.value
                ),
                low_gaps=sum(
                    1 for g in result.gaps
                    if g.risk_level == RiskLevel.LOW.value
                ),
                top_gaps=top_gaps,
                auto_generation_triggered=result.generated_tests_count > 0,
            )

            # Send events
            producer = EventProducer(
                bootstrap_servers=settings.redpanda_brokers,
            )
            await producer.start()
            try:
                coverage_topic = get_topic_for_event(coverage_event.event_type)
                sent = await producer.send(coverage_event, topic=coverage_topic)
                if sent:
                    emitted.append("COVERAGE_ANALYZED")

                gaps_topic = get_topic_for_event(gaps_event.event_type)
                sent = await producer.send(gaps_event, topic=gaps_topic)
                if sent:
                    emitted.append("COVERAGE_GAPS_IDENTIFIED")

            finally:
                await producer.stop()

            self.log.info("Kafka events emitted", events=emitted)

        except Exception as exc:
            self.log.warning(
                "Event emission failed (non-fatal)",
                error=str(exc),
            )

        return emitted

    # =========================================================================
    # Job tracking (Supabase)
    # =========================================================================

    async def _update_job(
        self,
        job_id: str,
        updates: dict[str, Any],
    ) -> None:
        """Update the ``qa_analysis_jobs`` table in Supabase.

        Creates the row if it doesn't exist, updates otherwise.
        Non-fatal on failure (we don't want job tracking to break analysis).

        Args:
            job_id: Job ID.
            updates: Column values to set.
        """
        try:
            from ...services.supabase_client import get_supabase_client

            db = get_supabase_client()
            if not db.is_configured:
                self.log.debug("Supabase not configured, skipping job update")
                return

            # Upsert pattern: try update first, insert if no rows matched
            update_data = {k: v for k, v in updates.items() if v is not None}
            update_data["updated_at"] = (
                datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

            try:
                result = await db.update(
                    "qa_analysis_jobs",
                    {"id": f"eq.{job_id}"},
                    update_data,
                )
                # If update returned empty (no matching row), insert
                data = result.get("data")
                if data is not None and isinstance(data, list) and len(data) == 0:
                    insert_data = {"id": job_id, **update_data}
                    await db.insert("qa_analysis_jobs", insert_data)
            except Exception:
                # Try insert as fallback
                insert_data = {"id": job_id, **update_data}
                try:
                    await db.insert("qa_analysis_jobs", insert_data)
                except Exception as insert_exc:
                    self.log.debug(
                        "Job insert fallback also failed",
                        error=str(insert_exc),
                    )

        except Exception as exc:
            self.log.debug(
                "Job update failed (non-fatal)",
                job_id=job_id,
                error=str(exc),
            )


# =============================================================================
# Module-level helpers
# =============================================================================


def _safe_json_parse(content: str, default: Any = None) -> Any:
    """Parse JSON from LLM output with fallback.

    Handles markdown code blocks and common LLM JSON issues.
    """
    if default is None:
        default = {}

    if not content or not content.strip():
        return default

    text = content.strip()

    # Extract from code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Remove trailing commas
    import re
    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON object/array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                pass

    logger.warning(
        "Failed to parse JSON from AI response",
        content_preview=content[:200],
    )
    return default


def _sanitize_for_cognee(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a dict for Cognee storage by replacing None with empty strings.

    Per MEMORY.md #4: Cognee's internal processing fails with
    'int() argument must be a string...NoneType' when None values are present.

    Args:
        data: Dictionary to sanitize.

    Returns:
        Sanitized dictionary with no None values.
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


def _risk_level_to_score(risk_level: str) -> float:
    """Convert a risk level string to a numeric score."""
    mapping = {
        "critical": 0.95,
        "high": 0.75,
        "medium": 0.50,
        "low": 0.25,
    }
    return mapping.get(risk_level.lower(), 0.50)


def _iso_to_epoch(iso_str: str) -> float:
    """Convert an ISO 8601 string to epoch seconds."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, AttributeError):
        return time.time()


def _diff_entities(
    base_gaps: list[dict[str, Any]],
    compare_gaps: list[dict[str, Any]],
) -> list[str]:
    """Find entity names in compare_gaps that are NOT in base_gaps."""
    base_entities = {
        g.get("entity", "") for g in base_gaps if isinstance(g, dict)
    }
    return [
        g.get("entity", "")
        for g in compare_gaps
        if isinstance(g, dict) and g.get("entity", "") not in base_entities
    ]


def _extract_job_result(db_result: dict[str, Any]) -> dict[str, Any] | None:
    """Extract result_json from a Supabase select response."""
    data = db_result.get("data")
    if not data:
        return None
    rows = data if isinstance(data, list) else [data]
    if not rows:
        return None
    raw = rows[0].get("result_json", "{}")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    return raw if isinstance(raw, dict) else None
