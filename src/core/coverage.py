"""
Coverage Analyzer - Detect Untested Code Paths

Primarily ALGORITHMIC analysis of code coverage data to:
1. Identify untested files, functions, and lines
2. Correlate coverage gaps with error hotspots
3. Prioritize what needs testing based on risk
4. Track coverage trends over time

Optional LLM integration for:
- Generating test recommendations for uncovered code
- Explaining why certain code is risky to leave untested
"""

import re
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import structlog
from anthropic import AsyncAnthropic

from src.config import get_settings

logger = structlog.get_logger()


class CoverageLevel(str, Enum):
    """Coverage health levels."""
    EXCELLENT = "excellent"   # 90%+
    GOOD = "good"             # 75-90%
    FAIR = "fair"             # 50-75%
    POOR = "poor"             # 25-50%
    CRITICAL = "critical"     # <25%


class GapPriority(str, Enum):
    """Priority for coverage gaps."""
    CRITICAL = "critical"     # Must fix immediately
    HIGH = "high"             # Should fix soon
    MEDIUM = "medium"         # Address when possible
    LOW = "low"               # Nice to have


@dataclass
class FileCoverage:
    """Coverage data for a single file."""
    path: str
    lines_total: int = 0
    lines_covered: int = 0
    lines_percent: float = 0.0
    branches_total: int = 0
    branches_covered: int = 0
    branches_percent: float = 0.0
    functions_total: int = 0
    functions_covered: int = 0
    functions_percent: float = 0.0

    # Detailed line info
    uncovered_lines: list[int] = field(default_factory=list)
    partially_covered_lines: list[int] = field(default_factory=list)

    # Uncovered functions
    uncovered_functions: list[str] = field(default_factory=list)

    @property
    def level(self) -> CoverageLevel:
        if self.lines_percent >= 90:
            return CoverageLevel.EXCELLENT
        elif self.lines_percent >= 75:
            return CoverageLevel.GOOD
        elif self.lines_percent >= 50:
            return CoverageLevel.FAIR
        elif self.lines_percent >= 25:
            return CoverageLevel.POOR
        return CoverageLevel.CRITICAL

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "lines_total": self.lines_total,
            "lines_covered": self.lines_covered,
            "lines_percent": self.lines_percent,
            "branches_total": self.branches_total,
            "branches_covered": self.branches_covered,
            "branches_percent": self.branches_percent,
            "functions_total": self.functions_total,
            "functions_covered": self.functions_covered,
            "functions_percent": self.functions_percent,
            "level": self.level.value,
            "uncovered_lines": self.uncovered_lines,
            "partially_covered_lines": self.partially_covered_lines,
            "uncovered_functions": self.uncovered_functions,
        }


@dataclass
class CoverageGap:
    """A significant gap in coverage that needs attention."""
    id: str
    file_path: str
    priority: GapPriority
    gap_type: str  # "file", "function", "branch", "lines"

    # What's missing
    description: str
    uncovered_lines: list[int] = field(default_factory=list)
    uncovered_functions: list[str] = field(default_factory=list)

    # Why it matters
    risk_reason: str = ""
    error_correlation: Optional[str] = None  # If errors have occurred here

    # Recommendation
    test_recommendation: Optional[str] = None

    # Metrics
    lines_uncovered: int = 0
    impact_score: float = 0.0  # 0-1, how important is this gap

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "priority": self.priority.value,
            "gap_type": self.gap_type,
            "description": self.description,
            "uncovered_lines": self.uncovered_lines,
            "uncovered_functions": self.uncovered_functions,
            "risk_reason": self.risk_reason,
            "error_correlation": self.error_correlation,
            "test_recommendation": self.test_recommendation,
            "lines_uncovered": self.lines_uncovered,
            "impact_score": self.impact_score,
        }


@dataclass
class CoverageSummary:
    """Overall coverage summary for a project."""
    # Totals
    lines_total: int = 0
    lines_covered: int = 0
    lines_percent: float = 0.0
    branches_total: int = 0
    branches_covered: int = 0
    branches_percent: float = 0.0
    functions_total: int = 0
    functions_covered: int = 0
    functions_percent: float = 0.0

    # File breakdown
    files: list[FileCoverage] = field(default_factory=list)
    files_with_coverage: int = 0
    files_without_coverage: int = 0

    # Level distribution
    excellent_count: int = 0
    good_count: int = 0
    fair_count: int = 0
    poor_count: int = 0
    critical_count: int = 0

    # Gaps
    gaps: list[CoverageGap] = field(default_factory=list)

    # Trend
    previous_percent: Optional[float] = None
    trend: str = "stable"  # "improving", "stable", "declining"

    # Metadata
    commit_sha: Optional[str] = None
    branch: str = "main"
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def overall_level(self) -> CoverageLevel:
        if self.lines_percent >= 90:
            return CoverageLevel.EXCELLENT
        elif self.lines_percent >= 75:
            return CoverageLevel.GOOD
        elif self.lines_percent >= 50:
            return CoverageLevel.FAIR
        elif self.lines_percent >= 25:
            return CoverageLevel.POOR
        return CoverageLevel.CRITICAL

    def to_dict(self) -> dict:
        return {
            "lines_total": self.lines_total,
            "lines_covered": self.lines_covered,
            "lines_percent": self.lines_percent,
            "branches_total": self.branches_total,
            "branches_covered": self.branches_covered,
            "branches_percent": self.branches_percent,
            "functions_total": self.functions_total,
            "functions_covered": self.functions_covered,
            "functions_percent": self.functions_percent,
            "overall_level": self.overall_level.value,
            "files": [f.to_dict() for f in self.files],
            "files_with_coverage": self.files_with_coverage,
            "files_without_coverage": self.files_without_coverage,
            "level_distribution": {
                "excellent": self.excellent_count,
                "good": self.good_count,
                "fair": self.fair_count,
                "poor": self.poor_count,
                "critical": self.critical_count,
            },
            "gaps": [g.to_dict() for g in self.gaps],
            "trend": self.trend,
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "created_at": self.created_at.isoformat(),
        }


class CoverageAnalyzer:
    """
    Analyzes code coverage to identify testing gaps.

    ALGORITHMIC:
    - Parse coverage reports (LCOV, Istanbul, Cobertura)
    - Calculate file/function/line coverage
    - Identify gaps and prioritize them
    - Track trends over time

    LLM-POWERED (optional):
    - Generate test recommendations for uncovered code
    - Explain risk of leaving code untested
    """

    SEMANTIC_MODEL = "claude-sonnet-4-5-20250514"
    FAST_MODEL = "claude-haiku-4-5-20250514"

    def __init__(
        self,
        codebase_path: Optional[str] = None,
        use_llm: bool = True,
    ):
        self.settings = get_settings()
        self.codebase_path = Path(codebase_path) if codebase_path else None
        self.use_llm = use_llm

        if use_llm:
            self.client = AsyncAnthropic(
                api_key=self.settings.anthropic_api_key.get_secret_value()
            )
        else:
            self.client = None

        self.log = logger.bind(component="coverage_analyzer")

    # =========================================================================
    # COVERAGE PARSING (ALGORITHMIC)
    # =========================================================================

    def parse_lcov(self, report_data: str) -> CoverageSummary:
        """Parse LCOV coverage report format."""
        files = []
        current_file = None
        file_data = {}

        lines_total = 0
        lines_covered = 0
        branches_total = 0
        branches_covered = 0
        functions_total = 0
        functions_covered = 0

        uncovered_lines = []
        line_hits = {}  # line_number -> hit_count

        for line in report_data.split("\n"):
            line = line.strip()

            if line.startswith("SF:"):
                # Start new file
                if current_file and file_data:
                    files.append(self._create_file_coverage(current_file, file_data, uncovered_lines))

                current_file = line[3:]
                file_data = {
                    "lines_total": 0,
                    "lines_covered": 0,
                    "branches_total": 0,
                    "branches_covered": 0,
                    "functions_total": 0,
                    "functions_covered": 0,
                    "uncovered_functions": [],
                }
                uncovered_lines = []
                line_hits = {}

            elif line.startswith("DA:"):
                # Line data: DA:line_number,hit_count
                parts = line[3:].split(",")
                if len(parts) >= 2:
                    line_num = int(parts[0])
                    hits = int(parts[1])
                    line_hits[line_num] = hits
                    if hits == 0:
                        uncovered_lines.append(line_num)

            elif line.startswith("LF:"):
                file_data["lines_total"] = int(line[3:])
                lines_total += file_data["lines_total"]

            elif line.startswith("LH:"):
                file_data["lines_covered"] = int(line[3:])
                lines_covered += file_data["lines_covered"]

            elif line.startswith("BRF:"):
                file_data["branches_total"] = int(line[4:])
                branches_total += file_data["branches_total"]

            elif line.startswith("BRH:"):
                file_data["branches_covered"] = int(line[4:])
                branches_covered += file_data["branches_covered"]

            elif line.startswith("FNF:"):
                file_data["functions_total"] = int(line[4:])
                functions_total += file_data["functions_total"]

            elif line.startswith("FNH:"):
                file_data["functions_covered"] = int(line[4:])
                functions_covered += file_data["functions_covered"]

            elif line.startswith("FN:"):
                # Function definition: FN:line,name
                pass  # Track function names if needed

            elif line.startswith("FNDA:"):
                # Function hit data: FNDA:hit_count,name
                parts = line[5:].split(",", 1)
                if len(parts) >= 2 and int(parts[0]) == 0:
                    file_data["uncovered_functions"].append(parts[1])

            elif line == "end_of_record":
                if current_file and file_data:
                    files.append(self._create_file_coverage(current_file, file_data, uncovered_lines))
                    current_file = None
                    file_data = {}
                    uncovered_lines = []

        # Create summary
        summary = CoverageSummary(
            lines_total=lines_total,
            lines_covered=lines_covered,
            lines_percent=round(lines_covered / lines_total * 100, 2) if lines_total > 0 else 0,
            branches_total=branches_total,
            branches_covered=branches_covered,
            branches_percent=round(branches_covered / branches_total * 100, 2) if branches_total > 0 else 0,
            functions_total=functions_total,
            functions_covered=functions_covered,
            functions_percent=round(functions_covered / functions_total * 100, 2) if functions_total > 0 else 0,
            files=files,
        )

        # Calculate level distribution
        for f in files:
            if f.level == CoverageLevel.EXCELLENT:
                summary.excellent_count += 1
            elif f.level == CoverageLevel.GOOD:
                summary.good_count += 1
            elif f.level == CoverageLevel.FAIR:
                summary.fair_count += 1
            elif f.level == CoverageLevel.POOR:
                summary.poor_count += 1
            else:
                summary.critical_count += 1

        summary.files_with_coverage = len([f for f in files if f.lines_covered > 0])
        summary.files_without_coverage = len([f for f in files if f.lines_covered == 0])

        return summary

    def parse_istanbul_json(self, report_data: str) -> CoverageSummary:
        """Parse Istanbul JSON coverage report format."""
        try:
            data = json.loads(report_data)
        except json.JSONDecodeError:
            return CoverageSummary()

        files = []
        lines_total = 0
        lines_covered = 0
        branches_total = 0
        branches_covered = 0
        functions_total = 0
        functions_covered = 0

        for file_path, file_data in data.items():
            if not isinstance(file_data, dict):
                continue

            s = file_data.get("s", {})  # statements
            b = file_data.get("b", {})  # branches
            f = file_data.get("f", {})  # functions
            fn_map = file_data.get("fnMap", {})  # function names

            file_lines_total = len(s)
            file_lines_covered = sum(1 for v in s.values() if v > 0)
            file_branches_total = sum(len(br) for br in b.values())
            file_branches_covered = sum(sum(1 for v in br if v > 0) for br in b.values())
            file_functions_total = len(f)
            file_functions_covered = sum(1 for v in f.values() if v > 0)

            # Find uncovered lines
            statement_map = file_data.get("statementMap", {})
            uncovered_lines = []
            for stmt_id, hits in s.items():
                if hits == 0:
                    stmt_info = statement_map.get(stmt_id, {})
                    start_line = stmt_info.get("start", {}).get("line")
                    if start_line:
                        uncovered_lines.append(start_line)

            # Find uncovered functions
            uncovered_functions = []
            for fn_id, hits in f.items():
                if hits == 0:
                    fn_info = fn_map.get(fn_id, {})
                    fn_name = fn_info.get("name", f"anonymous_{fn_id}")
                    uncovered_functions.append(fn_name)

            file_coverage = FileCoverage(
                path=file_path,
                lines_total=file_lines_total,
                lines_covered=file_lines_covered,
                lines_percent=round(file_lines_covered / file_lines_total * 100, 2) if file_lines_total > 0 else 0,
                branches_total=file_branches_total,
                branches_covered=file_branches_covered,
                branches_percent=round(file_branches_covered / file_branches_total * 100, 2) if file_branches_total > 0 else 0,
                functions_total=file_functions_total,
                functions_covered=file_functions_covered,
                functions_percent=round(file_functions_covered / file_functions_total * 100, 2) if file_functions_total > 0 else 0,
                uncovered_lines=sorted(set(uncovered_lines)),
                uncovered_functions=uncovered_functions,
            )

            files.append(file_coverage)

            lines_total += file_lines_total
            lines_covered += file_lines_covered
            branches_total += file_branches_total
            branches_covered += file_branches_covered
            functions_total += file_functions_total
            functions_covered += file_functions_covered

        summary = CoverageSummary(
            lines_total=lines_total,
            lines_covered=lines_covered,
            lines_percent=round(lines_covered / lines_total * 100, 2) if lines_total > 0 else 0,
            branches_total=branches_total,
            branches_covered=branches_covered,
            branches_percent=round(branches_covered / branches_total * 100, 2) if branches_total > 0 else 0,
            functions_total=functions_total,
            functions_covered=functions_covered,
            functions_percent=round(functions_covered / functions_total * 100, 2) if functions_total > 0 else 0,
            files=files,
        )

        # Calculate level distribution
        for f in files:
            if f.level == CoverageLevel.EXCELLENT:
                summary.excellent_count += 1
            elif f.level == CoverageLevel.GOOD:
                summary.good_count += 1
            elif f.level == CoverageLevel.FAIR:
                summary.fair_count += 1
            elif f.level == CoverageLevel.POOR:
                summary.poor_count += 1
            else:
                summary.critical_count += 1

        summary.files_with_coverage = len([f for f in files if f.lines_covered > 0])
        summary.files_without_coverage = len([f for f in files if f.lines_covered == 0])

        return summary

    def _create_file_coverage(
        self,
        path: str,
        data: dict,
        uncovered_lines: list[int],
    ) -> FileCoverage:
        """Create FileCoverage from parsed data."""
        lines_total = data.get("lines_total", 0)
        lines_covered = data.get("lines_covered", 0)

        return FileCoverage(
            path=path,
            lines_total=lines_total,
            lines_covered=lines_covered,
            lines_percent=round(lines_covered / lines_total * 100, 2) if lines_total > 0 else 0,
            branches_total=data.get("branches_total", 0),
            branches_covered=data.get("branches_covered", 0),
            branches_percent=round(
                data.get("branches_covered", 0) / data.get("branches_total", 1) * 100, 2
            ) if data.get("branches_total", 0) > 0 else 0,
            functions_total=data.get("functions_total", 0),
            functions_covered=data.get("functions_covered", 0),
            functions_percent=round(
                data.get("functions_covered", 0) / data.get("functions_total", 1) * 100, 2
            ) if data.get("functions_total", 0) > 0 else 0,
            uncovered_lines=sorted(set(uncovered_lines)),
            uncovered_functions=data.get("uncovered_functions", []),
        )

    # =========================================================================
    # GAP ANALYSIS (ALGORITHMIC)
    # =========================================================================

    async def analyze_gaps(
        self,
        summary: CoverageSummary,
        error_files: Optional[list[str]] = None,
        critical_paths: Optional[list[str]] = None,
    ) -> list[CoverageGap]:
        """
        Identify and prioritize coverage gaps.

        Args:
            summary: Coverage summary to analyze
            error_files: Files that have had production errors
            critical_paths: File patterns that are critical (e.g., auth, payments)

        Returns:
            List of coverage gaps, sorted by priority
        """
        import uuid

        gaps = []
        error_files = error_files or []
        critical_paths = critical_paths or [
            "auth", "login", "payment", "checkout", "security",
            "api", "middleware", "config",
        ]

        for file_coverage in summary.files:
            # Skip files with excellent coverage
            if file_coverage.level == CoverageLevel.EXCELLENT:
                continue

            # Determine priority based on multiple factors
            priority_score = 0.0

            # Factor 1: Coverage level
            if file_coverage.level == CoverageLevel.CRITICAL:
                priority_score += 0.4
            elif file_coverage.level == CoverageLevel.POOR:
                priority_score += 0.3
            elif file_coverage.level == CoverageLevel.FAIR:
                priority_score += 0.2

            # Factor 2: Error correlation
            is_error_file = any(
                error_file in file_coverage.path or file_coverage.path in error_file
                for error_file in error_files
            )
            if is_error_file:
                priority_score += 0.3

            # Factor 3: Critical path
            is_critical = any(
                critical in file_coverage.path.lower()
                for critical in critical_paths
            )
            if is_critical:
                priority_score += 0.2

            # Factor 4: Number of uncovered lines
            if file_coverage.lines_total - file_coverage.lines_covered > 50:
                priority_score += 0.1

            # Determine priority level
            if priority_score >= 0.7:
                priority = GapPriority.CRITICAL
            elif priority_score >= 0.5:
                priority = GapPriority.HIGH
            elif priority_score >= 0.3:
                priority = GapPriority.MEDIUM
            else:
                priority = GapPriority.LOW

            # Create gap description
            gap_type = "file"
            description = f"{file_coverage.path}: {file_coverage.lines_percent}% coverage"

            if file_coverage.uncovered_functions:
                gap_type = "function"
                description = f"{len(file_coverage.uncovered_functions)} untested functions in {file_coverage.path}"

            risk_reasons = []
            if is_error_file:
                risk_reasons.append("Production errors have occurred in this file")
            if is_critical:
                risk_reasons.append("This file is in a critical path")
            if file_coverage.level == CoverageLevel.CRITICAL:
                risk_reasons.append("Extremely low coverage increases bug risk")

            gap = CoverageGap(
                id=str(uuid.uuid4()),
                file_path=file_coverage.path,
                priority=priority,
                gap_type=gap_type,
                description=description,
                uncovered_lines=file_coverage.uncovered_lines[:20],  # Limit to first 20
                uncovered_functions=file_coverage.uncovered_functions,
                risk_reason=" | ".join(risk_reasons) if risk_reasons else "Low coverage",
                error_correlation="Errors detected" if is_error_file else None,
                lines_uncovered=file_coverage.lines_total - file_coverage.lines_covered,
                impact_score=priority_score,
            )

            gaps.append(gap)

        # Sort by impact score
        gaps.sort(key=lambda g: g.impact_score, reverse=True)

        # Use LLM to generate recommendations for top gaps
        if self.use_llm and self.client:
            gaps = await self._add_llm_recommendations(gaps[:10])  # Top 10 only

        summary.gaps = gaps
        return gaps

    # =========================================================================
    # LLM-POWERED RECOMMENDATIONS
    # =========================================================================

    async def _add_llm_recommendations(self, gaps: list[CoverageGap]) -> list[CoverageGap]:
        """Add LLM-generated test recommendations to gaps."""
        if not self.client or not gaps:
            return gaps

        # Batch gaps for efficiency
        gap_descriptions = []
        for gap in gaps:
            desc = f"- {gap.file_path}: {gap.description}"
            if gap.uncovered_functions:
                desc += f" (untested: {', '.join(gap.uncovered_functions[:5])})"
            gap_descriptions.append(desc)

        prompt = f"""Analyze these coverage gaps and suggest what tests to write:

COVERAGE GAPS:
{chr(10).join(gap_descriptions)}

For each file, suggest:
1. What type of test (unit, integration, e2e)
2. Key scenarios to cover
3. Priority reasoning

Output as JSON array:
[
    {{
        "file": "path/to/file",
        "test_type": "unit|integration|e2e",
        "scenarios": ["scenario 1", "scenario 2"],
        "recommendation": "Brief recommendation"
    }}
]"""

        try:
            response = await self.client.messages.create(
                model=self.FAST_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text
            json_start = text.find("[")
            json_end = text.rfind("]") + 1

            if json_start >= 0:
                recommendations = json.loads(text[json_start:json_end])

                # Map recommendations to gaps
                rec_map = {r.get("file", ""): r for r in recommendations}

                for gap in gaps:
                    rec = rec_map.get(gap.file_path)
                    if rec:
                        scenarios = rec.get("scenarios", [])
                        test_type = rec.get("test_type", "unit")
                        gap.test_recommendation = (
                            f"{rec.get('recommendation', '')} "
                            f"[{test_type} test] "
                            f"Scenarios: {', '.join(scenarios[:3])}"
                        )

        except Exception as e:
            self.log.warning("LLM recommendation failed", error=str(e))

        return gaps

    # =========================================================================
    # TREND ANALYSIS (ALGORITHMIC)
    # =========================================================================

    def compare_coverage(
        self,
        current: CoverageSummary,
        previous: CoverageSummary,
    ) -> dict:
        """
        Compare two coverage reports to identify trends.

        Returns dict with:
        - overall_change: percentage point change
        - trend: "improving", "stable", "declining"
        - new_gaps: files that lost coverage
        - closed_gaps: files that gained coverage
        """
        overall_change = current.lines_percent - previous.lines_percent

        if overall_change > 2:
            trend = "improving"
        elif overall_change < -2:
            trend = "declining"
        else:
            trend = "stable"

        # Find files that changed significantly
        current_files = {f.path: f for f in current.files}
        previous_files = {f.path: f for f in previous.files}

        new_gaps = []
        closed_gaps = []
        regressions = []

        for path, curr_file in current_files.items():
            prev_file = previous_files.get(path)

            if prev_file:
                change = curr_file.lines_percent - prev_file.lines_percent

                if change < -10:  # Lost more than 10%
                    regressions.append({
                        "file": path,
                        "previous": prev_file.lines_percent,
                        "current": curr_file.lines_percent,
                        "change": change,
                    })
                elif change > 10:  # Gained more than 10%
                    closed_gaps.append({
                        "file": path,
                        "previous": prev_file.lines_percent,
                        "current": curr_file.lines_percent,
                        "change": change,
                    })
            else:
                # New file
                if curr_file.lines_percent < 50:
                    new_gaps.append({
                        "file": path,
                        "coverage": curr_file.lines_percent,
                    })

        return {
            "overall_change": round(overall_change, 2),
            "trend": trend,
            "current_percent": current.lines_percent,
            "previous_percent": previous.lines_percent,
            "new_gaps": new_gaps,
            "closed_gaps": closed_gaps,
            "regressions": regressions,
        }

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_critical_uncovered_files(
        self,
        summary: CoverageSummary,
        max_results: int = 10,
    ) -> list[FileCoverage]:
        """Get files with critical/poor coverage, sorted by importance."""
        critical_files = [
            f for f in summary.files
            if f.level in (CoverageLevel.CRITICAL, CoverageLevel.POOR)
        ]

        # Sort by total lines (bigger files = more important)
        critical_files.sort(key=lambda f: f.lines_total, reverse=True)

        return critical_files[:max_results]

    def get_uncovered_functions(
        self,
        summary: CoverageSummary,
    ) -> list[dict]:
        """Get all uncovered functions across the codebase."""
        uncovered = []

        for file_coverage in summary.files:
            for func_name in file_coverage.uncovered_functions:
                uncovered.append({
                    "file": file_coverage.path,
                    "function": func_name,
                    "file_coverage": file_coverage.lines_percent,
                })

        return uncovered

    def calculate_coverage_score(self, summary: CoverageSummary) -> int:
        """
        Calculate a 0-100 coverage score.

        Weighted:
        - Line coverage: 50%
        - Branch coverage: 30%
        - Function coverage: 20%
        """
        line_score = summary.lines_percent * 0.5
        branch_score = summary.branches_percent * 0.3 if summary.branches_total > 0 else summary.lines_percent * 0.3
        func_score = summary.functions_percent * 0.2 if summary.functions_total > 0 else summary.lines_percent * 0.2

        return min(100, max(0, int(line_score + branch_score + func_score)))
