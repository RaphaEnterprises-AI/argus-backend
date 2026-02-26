"""Risk-scored coverage assessment for QA Engineer agent.

Computes coverage metrics and identifies gaps with risk scoring.
"""

import structlog
from typing import Any
from pydantic import BaseModel, Field
from enum import Enum

logger = structlog.get_logger()


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CoverageGap(BaseModel):
    entity: str
    entity_type: str  # function, endpoint, component, page, user_flow
    risk_level: str
    risk_score: float  # 0.0-1.0
    description: str
    suggested_tests: list[str] = Field(default_factory=list)
    fragility_context: str | None = None
    change_frequency: int = 0
    file_path: str | None = None


class CoverageMetrics(BaseModel):
    total_entities: int = 0
    covered_entities: int = 0
    coverage_pct: float = 0.0
    by_type: dict[str, dict] = Field(default_factory=dict)  # {type: {total, covered, pct}}


class CoverageAssessment(BaseModel):
    metrics: CoverageMetrics
    gaps: list[CoverageGap] = Field(default_factory=list)
    risk_summary: dict[str, int] = Field(default_factory=dict)  # {critical: N, high: N, ...}
    top_risks: list[CoverageGap] = Field(default_factory=list)  # Top 10 by risk


class CoverageAssessor:
    """Computes risk-scored coverage assessments."""

    # Keywords that indicate high-risk areas
    CRITICAL_KEYWORDS = {
        "auth", "login", "payment", "checkout", "password", "token",
        "session", "admin", "delete", "billing", "stripe", "credit",
    }
    HIGH_KEYWORDS = {
        "user", "account", "profile", "settings", "permission", "role",
        "signup", "register", "upload", "download",
    }

    def assess(
        self,
        testable_surfaces: list[dict],
        existing_tests: list[dict],
        fragility_data: list[dict] | None = None,
        change_frequencies: dict[str, int] | None = None,
    ) -> CoverageAssessment:
        """Assess coverage and identify gaps with risk scoring.

        Args:
            testable_surfaces: From CodeAnalyzerAgent [{name, type, file_path, ...}]
            existing_tests: Known tests [{name, type, target, ...}]
            fragility_data: From TestHealer [{selector, fragility_score, failure_count}]
            change_frequencies: {file_path: change_count} from git history
        """
        fragility_data = fragility_data or []
        change_frequencies = change_frequencies or {}

        # Build test coverage map
        covered = self._build_coverage_map(existing_tests)

        # Identify gaps
        gaps: list[CoverageGap] = []
        total = len(testable_surfaces)
        covered_count = 0

        by_type: dict[str, dict] = {}

        for surface in testable_surfaces:
            name = surface.get("name", "")
            stype = surface.get("type", "unknown")
            file_path = surface.get("file_path", "")

            # Track by type
            if stype not in by_type:
                by_type[stype] = {"total": 0, "covered": 0}
            by_type[stype]["total"] += 1

            is_covered = self._is_covered(name, stype, file_path, covered)
            if is_covered:
                covered_count += 1
                by_type[stype]["covered"] += 1
                continue

            # Not covered -- compute risk
            risk_score = self._compute_risk_score(
                name, stype, file_path, fragility_data, change_frequencies
            )
            risk_level = self._score_to_level(risk_score)

            gap = CoverageGap(
                entity=name,
                entity_type=stype,
                risk_level=risk_level,
                risk_score=risk_score,
                description=self._describe_gap(name, stype, risk_level),
                suggested_tests=self._suggest_tests(name, stype),
                fragility_context=self._get_fragility_context(name, fragility_data),
                change_frequency=change_frequencies.get(file_path, 0),
                file_path=file_path,
            )
            gaps.append(gap)

        # Sort gaps by risk score descending
        gaps.sort(key=lambda g: g.risk_score, reverse=True)

        # Compute per-type percentages
        for t in by_type.values():
            t["pct"] = (t["covered"] / t["total"] * 100) if t["total"] > 0 else 0.0

        risk_summary = {
            "critical": sum(1 for g in gaps if g.risk_level == RiskLevel.CRITICAL),
            "high": sum(1 for g in gaps if g.risk_level == RiskLevel.HIGH),
            "medium": sum(1 for g in gaps if g.risk_level == RiskLevel.MEDIUM),
            "low": sum(1 for g in gaps if g.risk_level == RiskLevel.LOW),
        }

        return CoverageAssessment(
            metrics=CoverageMetrics(
                total_entities=total,
                covered_entities=covered_count,
                coverage_pct=(covered_count / total * 100) if total > 0 else 0.0,
                by_type=by_type,
            ),
            gaps=gaps,
            risk_summary=risk_summary,
            top_risks=gaps[:10],
        )

    def _build_coverage_map(self, tests: list[dict]) -> dict[str, set]:
        """Build a map of covered entities from test list."""
        coverage: dict[str, set] = {"names": set(), "files": set(), "targets": set()}
        for test in tests:
            if test.get("name"):
                # Normalize test name to extract target
                name = test["name"].lower()
                coverage["names"].add(name)
            if test.get("target"):
                coverage["targets"].add(test["target"].lower())
            if test.get("file_path"):
                coverage["files"].add(test["file_path"])
        return coverage

    def _is_covered(self, name: str, stype: str, file_path: str, covered: dict) -> bool:
        """Check if a surface is covered by existing tests."""
        name_lower = name.lower()
        # Check if any test name references this entity
        for test_name in covered["names"]:
            if name_lower in test_name or test_name in name_lower:
                return True
        # Check if target matches
        if name_lower in covered["targets"]:
            return True
        # Check file-based coverage
        if file_path:
            test_file = file_path.replace("src/", "tests/test_").replace(".py", "_test.py")
            if test_file in covered["files"] or file_path in covered["files"]:
                return True
        return False

    def _compute_risk_score(
        self,
        name: str,
        stype: str,
        file_path: str,
        fragility_data: list[dict],
        change_frequencies: dict[str, int],
    ) -> float:
        """Compute risk score 0.0-1.0 for an uncovered entity."""
        score = 0.3  # Base score for any gap

        name_lower = name.lower()

        # Keyword-based risk
        if any(kw in name_lower for kw in self.CRITICAL_KEYWORDS):
            score += 0.4
        elif any(kw in name_lower for kw in self.HIGH_KEYWORDS):
            score += 0.25

        # Type-based risk
        type_weights = {
            "endpoint": 0.15,
            "page": 0.12,
            "component": 0.1,
            "function": 0.05,
        }
        score += type_weights.get(stype, 0.05)

        # Change frequency risk
        freq = change_frequencies.get(file_path, 0)
        if freq > 20:
            score += 0.15
        elif freq > 10:
            score += 0.1
        elif freq > 5:
            score += 0.05

        # Fragility risk
        for frag in fragility_data:
            if name_lower in str(frag.get("selector", "")).lower():
                score += frag.get("fragility_score", 0) * 0.1

        return min(score, 1.0)

    def _score_to_level(self, score: float) -> str:
        if score >= 0.7:
            return RiskLevel.CRITICAL
        elif score >= 0.5:
            return RiskLevel.HIGH
        elif score >= 0.3:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _describe_gap(self, name: str, stype: str, risk_level: str) -> str:
        return f"No test coverage for {stype} '{name}' (risk: {risk_level})"

    def _suggest_tests(self, name: str, stype: str) -> list[str]:
        suggestions: list[str] = []
        name_lower = name.lower()

        if stype == "endpoint":
            suggestions.extend([
                f"Test {name} returns 200 for valid request",
                f"Test {name} returns 401 for unauthenticated request",
                f"Test {name} validates request body schema",
            ])
        elif stype == "page":
            suggestions.extend([
                f"Test {name} renders correctly",
                f"Test {name} is accessible (WCAG 2.1)",
                f"Test {name} navigation works",
            ])
        elif stype == "component":
            suggestions.extend([
                f"Test {name} renders with default props",
                f"Test {name} handles user interaction",
            ])
        elif stype == "function":
            suggestions.extend([
                f"Test {name} with valid inputs",
                f"Test {name} with edge cases",
            ])

        if any(kw in name_lower for kw in self.CRITICAL_KEYWORDS):
            suggestions.append(f"Security test for {name}")

        return suggestions

    def _get_fragility_context(self, name: str, fragility_data: list[dict]) -> str | None:
        for frag in fragility_data:
            if name.lower() in str(frag.get("selector", "")).lower():
                return (
                    f"Related selector has fragility score "
                    f"{frag.get('fragility_score', 0):.2f} "
                    f"({frag.get('failure_count', 0)} failures)"
                )
        return None
