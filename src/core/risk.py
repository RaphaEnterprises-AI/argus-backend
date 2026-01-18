"""
Risk Scorer - Calculate Risk per File/Function

Hybrid approach combining:
1. ALGORITHMIC: Formula-based risk scoring using multiple signals
2. LLM-POWERED: Contextual risk explanations and recommendations

Risk signals include:
- Error frequency and severity
- Code coverage gaps
- Code complexity
- Change frequency (churn)
- Time since last test
- Dependency count
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import structlog
from anthropic import AsyncAnthropic

from src.config import get_settings
from src.core.coverage import CoverageLevel, CoverageSummary, FileCoverage
from src.core.model_registry import get_model_id
from src.core.normalizer import NormalizedEvent, Severity

logger = structlog.get_logger()


class RiskLevel(str, Enum):
    """Overall risk levels."""
    CRITICAL = "critical"   # 80-100: Immediate attention needed
    HIGH = "high"           # 60-80: Address soon
    MEDIUM = "medium"       # 40-60: Monitor closely
    LOW = "low"             # 20-40: Acceptable risk
    MINIMAL = "minimal"     # 0-20: Well maintained


@dataclass
class RiskFactor:
    """A single risk factor contributing to overall score."""
    name: str
    weight: float  # 0-1, how much this factor contributes
    score: float   # 0-100, raw score for this factor
    weighted_score: float  # weight * score
    reason: str    # Human-readable explanation

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "weight": self.weight,
            "score": self.score,
            "weighted_score": self.weighted_score,
            "reason": self.reason,
        }


@dataclass
class FileRisk:
    """Risk assessment for a single file."""
    file_path: str
    overall_score: float  # 0-100
    risk_level: RiskLevel

    # Contributing factors
    factors: list[RiskFactor] = field(default_factory=list)

    # Raw metrics
    error_count: int = 0
    error_count_24h: int = 0
    coverage_percent: float = 0.0
    complexity_score: float = 0.0
    churn_count: int = 0  # Changes in last 30 days
    days_since_test: int | None = None
    dependency_count: int = 0

    # LLM insights
    risk_explanation: str | None = None
    recommendations: list[str] = field(default_factory=list)

    # Metadata
    last_error_at: datetime | None = None
    last_modified_at: datetime | None = None
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "overall_score": self.overall_score,
            "risk_level": self.risk_level.value,
            "factors": [f.to_dict() for f in self.factors],
            "metrics": {
                "error_count": self.error_count,
                "error_count_24h": self.error_count_24h,
                "coverage_percent": self.coverage_percent,
                "complexity_score": self.complexity_score,
                "churn_count": self.churn_count,
                "days_since_test": self.days_since_test,
                "dependency_count": self.dependency_count,
            },
            "risk_explanation": self.risk_explanation,
            "recommendations": self.recommendations,
            "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None,
            "last_modified_at": self.last_modified_at.isoformat() if self.last_modified_at else None,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class ProjectRisk:
    """Overall risk assessment for a project."""
    project_id: str
    overall_score: float  # 0-100
    risk_level: RiskLevel

    # Component scores
    error_score: float = 50.0
    coverage_score: float = 50.0
    stability_score: float = 50.0
    maintenance_score: float = 50.0

    # High-risk files
    high_risk_files: list[FileRisk] = field(default_factory=list)

    # Metrics
    total_files: int = 0
    files_at_risk: int = 0
    total_errors_24h: int = 0
    average_coverage: float = 0.0
    ci_success_rate: float = 0.0

    # Trend
    previous_score: float | None = None
    trend: str = "stable"  # "improving", "stable", "declining"

    # LLM insights
    summary: str | None = None
    top_recommendations: list[str] = field(default_factory=list)

    # Metadata
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "overall_score": self.overall_score,
            "risk_level": self.risk_level.value,
            "component_scores": {
                "error_score": self.error_score,
                "coverage_score": self.coverage_score,
                "stability_score": self.stability_score,
                "maintenance_score": self.maintenance_score,
            },
            "high_risk_files": [f.to_dict() for f in self.high_risk_files],
            "metrics": {
                "total_files": self.total_files,
                "files_at_risk": self.files_at_risk,
                "total_errors_24h": self.total_errors_24h,
                "average_coverage": self.average_coverage,
                "ci_success_rate": self.ci_success_rate,
            },
            "trend": self.trend,
            "summary": self.summary,
            "top_recommendations": self.top_recommendations,
            "calculated_at": self.calculated_at.isoformat(),
        }


class RiskScorer:
    """
    Calculates risk scores using hybrid algo + LLM approach.

    ALGORITHMIC SCORING:
    - Weighted formula combining multiple risk signals
    - Configurable weights for different factors
    - Deterministic and fast

    LLM ENHANCEMENT:
    - Natural language risk explanations
    - Context-aware recommendations
    - Pattern recognition across codebase
    """

    @property
    def SEMANTIC_MODEL(self) -> str:
        return get_model_id("claude-sonnet-4-5")

    @property
    def FAST_MODEL(self) -> str:
        return get_model_id("claude-haiku-4-5")

    # Default weights for risk factors
    DEFAULT_WEIGHTS = {
        "error_frequency": 0.30,      # Recent errors
        "error_severity": 0.15,       # Severity of errors
        "coverage_gap": 0.25,         # Test coverage
        "code_complexity": 0.10,      # Cyclomatic complexity
        "churn_rate": 0.10,           # Change frequency
        "dependency_risk": 0.10,      # External dependencies
    }

    def __init__(
        self,
        codebase_path: str | None = None,
        use_llm: bool = True,
        weights: dict[str, float] | None = None,
    ):
        self.settings = get_settings()
        self.codebase_path = Path(codebase_path) if codebase_path else None
        self.use_llm = use_llm
        self.weights = weights or self.DEFAULT_WEIGHTS

        if use_llm:
            api_key = self.settings.anthropic_api_key
            if hasattr(api_key, 'get_secret_value'):
                api_key = api_key.get_secret_value()
            self.client = AsyncAnthropic(api_key=api_key)
        else:
            self.client = None

        self.log = logger.bind(component="risk_scorer")

    # =========================================================================
    # FILE-LEVEL RISK SCORING (ALGORITHMIC)
    # =========================================================================

    async def calculate_file_risk(
        self,
        file_path: str,
        errors: list[NormalizedEvent],
        coverage: FileCoverage | None = None,
        churn_count: int = 0,
        complexity_score: float = 0.0,
        dependency_count: int = 0,
        include_llm: bool = True,
    ) -> FileRisk:
        """
        Calculate risk score for a single file.

        Args:
            file_path: Path to the file
            errors: Errors that occurred in this file
            coverage: Coverage data for this file
            churn_count: Number of changes in last 30 days
            complexity_score: Cyclomatic complexity (0-100)
            dependency_count: Number of external dependencies
            include_llm: Whether to add LLM explanations

        Returns:
            FileRisk with overall score and breakdown
        """
        factors = []

        # Factor 1: Error Frequency (0-100)
        error_count = len(errors)
        recent_errors = [e for e in errors if e.created_at > datetime.utcnow() - timedelta(hours=24)]
        error_count_24h = len(recent_errors)

        # Score: More errors = higher risk
        # 0 errors = 0, 1 error = 30, 5+ errors = 100
        if error_count == 0:
            error_freq_score = 0
            error_freq_reason = "No errors recorded"
        elif error_count_24h > 5:
            error_freq_score = 100
            error_freq_reason = f"{error_count_24h} errors in last 24h (critical)"
        elif error_count_24h > 0:
            error_freq_score = min(100, 30 + (error_count_24h * 14))
            error_freq_reason = f"{error_count_24h} errors in last 24h"
        else:
            error_freq_score = min(60, 10 + (error_count * 5))
            error_freq_reason = f"{error_count} total errors"

        factors.append(RiskFactor(
            name="error_frequency",
            weight=self.weights["error_frequency"],
            score=error_freq_score,
            weighted_score=error_freq_score * self.weights["error_frequency"],
            reason=error_freq_reason,
        ))

        # Factor 2: Error Severity (0-100)
        if errors:
            severity_scores = {
                Severity.FATAL: 100,
                Severity.ERROR: 70,
                Severity.WARNING: 40,
                Severity.INFO: 10,
            }
            max_severity = max(errors, key=lambda e: severity_scores.get(e.severity, 50))
            severity_score = severity_scores.get(max_severity.severity, 50)
            severity_reason = f"Highest severity: {max_severity.severity.value}"
        else:
            severity_score = 0
            severity_reason = "No errors to assess"

        factors.append(RiskFactor(
            name="error_severity",
            weight=self.weights["error_severity"],
            score=severity_score,
            weighted_score=severity_score * self.weights["error_severity"],
            reason=severity_reason,
        ))

        # Factor 3: Coverage Gap (0-100)
        if coverage:
            # Lower coverage = higher risk
            coverage_gap_score = max(0, 100 - coverage.lines_percent)
            if coverage.level == CoverageLevel.CRITICAL:
                coverage_reason = f"Critical coverage gap ({coverage.lines_percent}%)"
            elif coverage.level == CoverageLevel.POOR:
                coverage_reason = f"Poor coverage ({coverage.lines_percent}%)"
            else:
                coverage_reason = f"Coverage: {coverage.lines_percent}%"
            coverage_percent = coverage.lines_percent
        else:
            coverage_gap_score = 50  # Unknown = moderate risk
            coverage_reason = "Coverage data unavailable"
            coverage_percent = 0.0

        factors.append(RiskFactor(
            name="coverage_gap",
            weight=self.weights["coverage_gap"],
            score=coverage_gap_score,
            weighted_score=coverage_gap_score * self.weights["coverage_gap"],
            reason=coverage_reason,
        ))

        # Factor 4: Code Complexity (0-100)
        complexity_reason = "Low complexity" if complexity_score < 30 else (
            "High complexity" if complexity_score > 70 else "Moderate complexity"
        )
        factors.append(RiskFactor(
            name="code_complexity",
            weight=self.weights["code_complexity"],
            score=complexity_score,
            weighted_score=complexity_score * self.weights["code_complexity"],
            reason=f"{complexity_reason} (score: {complexity_score})",
        ))

        # Factor 5: Churn Rate (0-100)
        # High churn = higher risk (more changes = more bugs)
        if churn_count == 0:
            churn_score = 0
            churn_reason = "No recent changes"
        elif churn_count > 20:
            churn_score = 100
            churn_reason = f"Very high churn ({churn_count} changes in 30 days)"
        else:
            churn_score = min(100, churn_count * 5)
            churn_reason = f"{churn_count} changes in last 30 days"

        factors.append(RiskFactor(
            name="churn_rate",
            weight=self.weights["churn_rate"],
            score=churn_score,
            weighted_score=churn_score * self.weights["churn_rate"],
            reason=churn_reason,
        ))

        # Factor 6: Dependency Risk (0-100)
        if dependency_count == 0:
            dep_score = 0
            dep_reason = "No external dependencies"
        elif dependency_count > 10:
            dep_score = min(100, 50 + (dependency_count - 10) * 5)
            dep_reason = f"Many dependencies ({dependency_count})"
        else:
            dep_score = dependency_count * 5
            dep_reason = f"{dependency_count} dependencies"

        factors.append(RiskFactor(
            name="dependency_risk",
            weight=self.weights["dependency_risk"],
            score=dep_score,
            weighted_score=dep_score * self.weights["dependency_risk"],
            reason=dep_reason,
        ))

        # Calculate overall score
        overall_score = sum(f.weighted_score for f in factors)
        overall_score = min(100, max(0, overall_score))

        # Determine risk level
        risk_level = self._score_to_level(overall_score)

        # Get last error time
        last_error_at = max((e.created_at for e in errors), default=None) if errors else None

        file_risk = FileRisk(
            file_path=file_path,
            overall_score=round(overall_score, 1),
            risk_level=risk_level,
            factors=factors,
            error_count=error_count,
            error_count_24h=error_count_24h,
            coverage_percent=coverage_percent,
            complexity_score=complexity_score,
            churn_count=churn_count,
            dependency_count=dependency_count,
            last_error_at=last_error_at,
        )

        # Add LLM insights if enabled
        if include_llm and self.use_llm and self.client and overall_score > 40:
            file_risk = await self._enhance_file_risk_with_llm(file_risk, errors)

        return file_risk

    # =========================================================================
    # PROJECT-LEVEL RISK SCORING (ALGORITHMIC)
    # =========================================================================

    async def calculate_project_risk(
        self,
        project_id: str,
        file_risks: list[FileRisk],
        coverage_summary: CoverageSummary | None = None,
        ci_success_rate: float = 100.0,
        include_llm: bool = True,
    ) -> ProjectRisk:
        """
        Calculate overall risk score for a project.

        Args:
            project_id: Project identifier
            file_risks: Risk assessments for individual files
            coverage_summary: Overall coverage data
            ci_success_rate: CI pipeline success rate (0-100)
            include_llm: Whether to add LLM summary

        Returns:
            ProjectRisk with overall assessment
        """
        if not file_risks:
            return ProjectRisk(
                project_id=project_id,
                overall_score=50,
                risk_level=RiskLevel.MEDIUM,
            )

        # Component 1: Error Score (average of file error risks)
        error_scores = [f.factors[0].score for f in file_risks if f.factors]
        error_score = sum(error_scores) / len(error_scores) if error_scores else 50

        # Component 2: Coverage Score (inverse of coverage)
        if coverage_summary:
            coverage_score = max(0, 100 - coverage_summary.lines_percent)
            average_coverage = coverage_summary.lines_percent
        else:
            avg_coverage = sum(f.coverage_percent for f in file_risks) / len(file_risks)
            coverage_score = max(0, 100 - avg_coverage)
            average_coverage = avg_coverage

        # Component 3: Stability Score (based on CI and error trends)
        stability_score = max(0, 100 - ci_success_rate)

        # Component 4: Maintenance Score (based on churn and complexity)
        maintenance_factors = []
        for f in file_risks:
            churn_factor = next((fac for fac in f.factors if fac.name == "churn_rate"), None)
            complexity_factor = next((fac for fac in f.factors if fac.name == "code_complexity"), None)
            if churn_factor:
                maintenance_factors.append(churn_factor.score)
            if complexity_factor:
                maintenance_factors.append(complexity_factor.score)

        maintenance_score = sum(maintenance_factors) / len(maintenance_factors) if maintenance_factors else 50

        # Calculate overall score (weighted average of components)
        overall_score = (
            error_score * 0.35 +
            coverage_score * 0.30 +
            stability_score * 0.20 +
            maintenance_score * 0.15
        )
        overall_score = min(100, max(0, overall_score))

        # Get high-risk files
        high_risk_files = sorted(
            [f for f in file_risks if f.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)],
            key=lambda f: f.overall_score,
            reverse=True
        )[:10]

        # Count files at risk
        files_at_risk = len([f for f in file_risks if f.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)])

        # Calculate total errors in 24h
        total_errors_24h = sum(f.error_count_24h for f in file_risks)

        project_risk = ProjectRisk(
            project_id=project_id,
            overall_score=round(overall_score, 1),
            risk_level=self._score_to_level(overall_score),
            error_score=round(100 - error_score, 1),  # Invert for "health" perspective
            coverage_score=round(100 - coverage_score, 1),
            stability_score=round(100 - stability_score, 1),
            maintenance_score=round(100 - maintenance_score, 1),
            high_risk_files=high_risk_files,
            total_files=len(file_risks),
            files_at_risk=files_at_risk,
            total_errors_24h=total_errors_24h,
            average_coverage=round(average_coverage, 1),
            ci_success_rate=ci_success_rate,
        )

        # Add LLM summary if enabled
        if include_llm and self.use_llm and self.client:
            project_risk = await self._enhance_project_risk_with_llm(project_risk)

        return project_risk

    # =========================================================================
    # LLM-POWERED RISK EXPLANATIONS
    # =========================================================================

    async def _enhance_file_risk_with_llm(
        self,
        file_risk: FileRisk,
        errors: list[NormalizedEvent],
    ) -> FileRisk:
        """Add LLM-generated explanations and recommendations."""
        if not self.client:
            return file_risk

        # Build context
        factor_summary = "\n".join(
            f"- {f.name}: {f.score}/100 ({f.reason})"
            for f in file_risk.factors
        )

        error_summary = ""
        if errors:
            error_types = defaultdict(int)
            for e in errors[:10]:
                error_types[e.error_type or e.title[:50]] += 1
            error_summary = "\n".join(f"- {k}: {v} occurrences" for k, v in error_types.items())

        prompt = f"""Analyze this file's risk assessment and provide actionable recommendations:

FILE: {file_risk.file_path}
OVERALL RISK: {file_risk.overall_score}/100 ({file_risk.risk_level.value})

RISK FACTORS:
{factor_summary}

{"RECENT ERRORS:" + chr(10) + error_summary if error_summary else "No recent errors"}

Provide:
1. A 1-2 sentence explanation of why this file is at risk
2. 2-3 specific, actionable recommendations to reduce risk

Output as JSON:
{{
    "explanation": "Why this file is at risk",
    "recommendations": ["Recommendation 1", "Recommendation 2"]
}}"""

        try:
            response = await self.client.messages.create(
                model=self.FAST_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text
            json_start = text.find("{")
            json_end = text.rfind("}") + 1

            if json_start >= 0:
                analysis = json.loads(text[json_start:json_end])
                file_risk.risk_explanation = analysis.get("explanation")
                file_risk.recommendations = analysis.get("recommendations", [])

        except Exception as e:
            self.log.warning("LLM enhancement failed", error=str(e))

        return file_risk

    async def _enhance_project_risk_with_llm(
        self,
        project_risk: ProjectRisk,
    ) -> ProjectRisk:
        """Add LLM-generated project summary and recommendations."""
        if not self.client:
            return project_risk

        high_risk_summary = "\n".join(
            f"- {f.file_path}: {f.overall_score}/100"
            for f in project_risk.high_risk_files[:5]
        )

        prompt = f"""Analyze this project's overall risk and provide executive summary:

PROJECT RISK: {project_risk.overall_score}/100 ({project_risk.risk_level.value})

COMPONENT SCORES:
- Error Health: {project_risk.error_score}/100
- Coverage Health: {project_risk.coverage_score}/100
- Stability: {project_risk.stability_score}/100
- Maintenance: {project_risk.maintenance_score}/100

METRICS:
- Files at risk: {project_risk.files_at_risk}/{project_risk.total_files}
- Errors (24h): {project_risk.total_errors_24h}
- Coverage: {project_risk.average_coverage}%
- CI Success: {project_risk.ci_success_rate}%

{"HIGH RISK FILES:" + chr(10) + high_risk_summary if high_risk_summary else ""}

Provide:
1. A brief executive summary (2-3 sentences)
2. Top 3 recommendations to improve project health

Output as JSON:
{{
    "summary": "Executive summary",
    "recommendations": ["Top recommendation 1", "Top recommendation 2", "Top recommendation 3"]
}}"""

        try:
            response = await self.client.messages.create(
                model=self.FAST_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text
            json_start = text.find("{")
            json_end = text.rfind("}") + 1

            if json_start >= 0:
                analysis = json.loads(text[json_start:json_end])
                project_risk.summary = analysis.get("summary")
                project_risk.top_recommendations = analysis.get("recommendations", [])

        except Exception as e:
            self.log.warning("LLM enhancement failed", error=str(e))

        return project_risk

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score >= 80:
            return RiskLevel.CRITICAL
        elif score >= 60:
            return RiskLevel.HIGH
        elif score >= 40:
            return RiskLevel.MEDIUM
        elif score >= 20:
            return RiskLevel.LOW
        return RiskLevel.MINIMAL

    async def get_risk_trends(
        self,
        current_risk: ProjectRisk,
        previous_risk: ProjectRisk | None,
    ) -> dict:
        """Compare risk over time to identify trends."""
        if not previous_risk:
            return {
                "trend": "stable",
                "score_change": 0,
                "improving_areas": [],
                "declining_areas": [],
            }

        score_change = current_risk.overall_score - previous_risk.overall_score

        if score_change < -5:
            trend = "improving"
        elif score_change > 5:
            trend = "declining"
        else:
            trend = "stable"

        # Find specific areas of change
        improving = []
        declining = []

        if current_risk.error_score > previous_risk.error_score + 5:
            improving.append("Error rate")
        elif current_risk.error_score < previous_risk.error_score - 5:
            declining.append("Error rate")

        if current_risk.coverage_score > previous_risk.coverage_score + 5:
            improving.append("Test coverage")
        elif current_risk.coverage_score < previous_risk.coverage_score - 5:
            declining.append("Test coverage")

        if current_risk.stability_score > previous_risk.stability_score + 5:
            improving.append("CI stability")
        elif current_risk.stability_score < previous_risk.stability_score - 5:
            declining.append("CI stability")

        return {
            "trend": trend,
            "score_change": round(score_change, 1),
            "improving_areas": improving,
            "declining_areas": declining,
            "current_score": current_risk.overall_score,
            "previous_score": previous_risk.overall_score,
        }

    def get_priority_files(
        self,
        file_risks: list[FileRisk],
        max_results: int = 10,
    ) -> list[FileRisk]:
        """Get files that should be prioritized for attention."""
        # Sort by overall score (highest risk first)
        sorted_risks = sorted(file_risks, key=lambda f: f.overall_score, reverse=True)

        # Filter to only high/critical risk
        priority_files = [
            f for f in sorted_risks
            if f.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
        ]

        return priority_files[:max_results]
