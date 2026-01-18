"""Tests for the coverage analyzer module."""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestCoverageLevel:
    """Tests for CoverageLevel enum."""

    def test_coverage_levels_exist(self, mock_env_vars):
        """Test that all coverage levels are defined."""
        from src.core.coverage import CoverageLevel

        assert CoverageLevel.EXCELLENT == "excellent"
        assert CoverageLevel.GOOD == "good"
        assert CoverageLevel.FAIR == "fair"
        assert CoverageLevel.POOR == "poor"
        assert CoverageLevel.CRITICAL == "critical"


class TestGapPriority:
    """Tests for GapPriority enum."""

    def test_gap_priorities_exist(self, mock_env_vars):
        """Test that all gap priorities are defined."""
        from src.core.coverage import GapPriority

        assert GapPriority.CRITICAL == "critical"
        assert GapPriority.HIGH == "high"
        assert GapPriority.MEDIUM == "medium"
        assert GapPriority.LOW == "low"


class TestFileCoverage:
    """Tests for FileCoverage dataclass."""

    def test_file_coverage_creation(self, mock_env_vars):
        """Test creating a FileCoverage instance."""
        from src.core.coverage import FileCoverage, CoverageLevel

        fc = FileCoverage(
            path="src/utils/helpers.py",
            lines_total=100,
            lines_covered=90,
            lines_percent=90.0,
            branches_total=20,
            branches_covered=18,
            branches_percent=90.0,
            functions_total=10,
            functions_covered=9,
            functions_percent=90.0,
        )

        assert fc.path == "src/utils/helpers.py"
        assert fc.lines_total == 100
        assert fc.level == CoverageLevel.EXCELLENT

    def test_file_coverage_level_excellent(self, mock_env_vars):
        """Test coverage level determination - excellent."""
        from src.core.coverage import FileCoverage, CoverageLevel

        fc = FileCoverage(path="test.py", lines_percent=95.0)
        assert fc.level == CoverageLevel.EXCELLENT

    def test_file_coverage_level_good(self, mock_env_vars):
        """Test coverage level determination - good."""
        from src.core.coverage import FileCoverage, CoverageLevel

        fc = FileCoverage(path="test.py", lines_percent=80.0)
        assert fc.level == CoverageLevel.GOOD

    def test_file_coverage_level_fair(self, mock_env_vars):
        """Test coverage level determination - fair."""
        from src.core.coverage import FileCoverage, CoverageLevel

        fc = FileCoverage(path="test.py", lines_percent=60.0)
        assert fc.level == CoverageLevel.FAIR

    def test_file_coverage_level_poor(self, mock_env_vars):
        """Test coverage level determination - poor."""
        from src.core.coverage import FileCoverage, CoverageLevel

        fc = FileCoverage(path="test.py", lines_percent=30.0)
        assert fc.level == CoverageLevel.POOR

    def test_file_coverage_level_critical(self, mock_env_vars):
        """Test coverage level determination - critical."""
        from src.core.coverage import FileCoverage, CoverageLevel

        fc = FileCoverage(path="test.py", lines_percent=10.0)
        assert fc.level == CoverageLevel.CRITICAL

    def test_file_coverage_to_dict(self, mock_env_vars):
        """Test FileCoverage to_dict method."""
        from src.core.coverage import FileCoverage

        fc = FileCoverage(
            path="src/test.py",
            lines_total=100,
            lines_covered=75,
            lines_percent=75.0,
            uncovered_lines=[10, 20, 30],
            uncovered_functions=["func_a", "func_b"],
        )

        result = fc.to_dict()

        assert result["path"] == "src/test.py"
        assert result["lines_total"] == 100
        assert result["lines_percent"] == 75.0
        assert result["level"] == "good"
        assert result["uncovered_lines"] == [10, 20, 30]
        assert result["uncovered_functions"] == ["func_a", "func_b"]


class TestCoverageGap:
    """Tests for CoverageGap dataclass."""

    def test_coverage_gap_creation(self, mock_env_vars):
        """Test creating a CoverageGap instance."""
        from src.core.coverage import CoverageGap, GapPriority

        gap = CoverageGap(
            id="gap-001",
            file_path="src/auth/login.py",
            priority=GapPriority.CRITICAL,
            gap_type="function",
            description="Critical authentication code untested",
            uncovered_lines=[10, 11, 12, 13],
            uncovered_functions=["validate_token"],
            risk_reason="Authentication code with zero coverage",
            lines_uncovered=4,
            impact_score=0.9,
        )

        assert gap.id == "gap-001"
        assert gap.priority == GapPriority.CRITICAL
        assert len(gap.uncovered_lines) == 4

    def test_coverage_gap_to_dict(self, mock_env_vars):
        """Test CoverageGap to_dict method."""
        from src.core.coverage import CoverageGap, GapPriority

        gap = CoverageGap(
            id="gap-002",
            file_path="src/payments/checkout.py",
            priority=GapPriority.HIGH,
            gap_type="file",
            description="Payment processing untested",
            error_correlation="Errors detected",
            test_recommendation="Add integration tests for checkout flow",
        )

        result = gap.to_dict()

        assert result["id"] == "gap-002"
        assert result["priority"] == "high"
        assert result["gap_type"] == "file"
        assert result["error_correlation"] == "Errors detected"
        assert result["test_recommendation"] is not None


class TestCoverageSummary:
    """Tests for CoverageSummary dataclass."""

    def test_coverage_summary_creation(self, mock_env_vars):
        """Test creating a CoverageSummary instance."""
        from src.core.coverage import CoverageSummary, CoverageLevel

        summary = CoverageSummary(
            lines_total=1000,
            lines_covered=800,
            lines_percent=80.0,
            branches_total=200,
            branches_covered=160,
            branches_percent=80.0,
            functions_total=100,
            functions_covered=90,
            functions_percent=90.0,
        )

        assert summary.overall_level == CoverageLevel.GOOD

    def test_coverage_summary_level_excellent(self, mock_env_vars):
        """Test overall level - excellent."""
        from src.core.coverage import CoverageSummary, CoverageLevel

        summary = CoverageSummary(lines_percent=95.0)
        assert summary.overall_level == CoverageLevel.EXCELLENT

    def test_coverage_summary_level_critical(self, mock_env_vars):
        """Test overall level - critical."""
        from src.core.coverage import CoverageSummary, CoverageLevel

        summary = CoverageSummary(lines_percent=20.0)
        assert summary.overall_level == CoverageLevel.CRITICAL

    def test_coverage_summary_to_dict(self, mock_env_vars):
        """Test CoverageSummary to_dict method."""
        from src.core.coverage import CoverageSummary, FileCoverage, CoverageGap, GapPriority

        file_cov = FileCoverage(path="test.py", lines_percent=80.0)
        gap = CoverageGap(
            id="gap-1",
            file_path="test.py",
            priority=GapPriority.MEDIUM,
            gap_type="file",
            description="Test gap",
        )

        summary = CoverageSummary(
            lines_total=100,
            lines_covered=80,
            lines_percent=80.0,
            files=[file_cov],
            gaps=[gap],
            excellent_count=0,
            good_count=1,
            commit_sha="abc123",
            branch="main",
        )

        result = summary.to_dict()

        assert result["lines_percent"] == 80.0
        assert result["overall_level"] == "good"
        assert len(result["files"]) == 1
        assert len(result["gaps"]) == 1
        assert result["commit_sha"] == "abc123"
        assert "level_distribution" in result


class TestCoverageAnalyzer:
    """Tests for CoverageAnalyzer class."""

    def test_analyzer_initialization_with_llm(self, mock_env_vars):
        """Test CoverageAnalyzer initialization with LLM enabled."""
        from src.core.coverage import CoverageAnalyzer

        with patch("anthropic.AsyncAnthropic"):
            analyzer = CoverageAnalyzer(codebase_path="/test/path", use_llm=True)

            assert analyzer.codebase_path is not None
            assert analyzer.use_llm is True
            assert analyzer.client is not None

    def test_analyzer_initialization_without_llm(self, mock_env_vars):
        """Test CoverageAnalyzer initialization with LLM disabled."""
        from src.core.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer(codebase_path="/test/path", use_llm=False)

        assert analyzer.use_llm is False
        assert analyzer.client is None

    def test_parse_lcov_simple(self, mock_env_vars):
        """Test parsing a simple LCOV report."""
        from src.core.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer(use_llm=False)

        lcov_report = """TN:
SF:src/utils.py
FN:1,helper_function
FN:10,main
FNDA:5,helper_function
FNDA:0,main
FNF:2
FNH:1
DA:1,5
DA:2,5
DA:10,0
DA:11,0
LF:4
LH:2
BRF:2
BRH:1
end_of_record
"""

        summary = analyzer.parse_lcov(lcov_report)

        assert summary.lines_total == 4
        assert summary.lines_covered == 2
        assert summary.lines_percent == 50.0
        assert summary.functions_total == 2
        assert summary.functions_covered == 1
        assert len(summary.files) == 1
        assert summary.files[0].path == "src/utils.py"

    def test_parse_lcov_multiple_files(self, mock_env_vars):
        """Test parsing LCOV with multiple files."""
        from src.core.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer(use_llm=False)

        lcov_report = """TN:
SF:src/file1.py
LF:10
LH:8
end_of_record
SF:src/file2.py
LF:20
LH:15
end_of_record
"""

        summary = analyzer.parse_lcov(lcov_report)

        assert len(summary.files) == 2
        assert summary.lines_total == 30
        assert summary.lines_covered == 23

    def test_parse_lcov_with_uncovered_functions(self, mock_env_vars):
        """Test LCOV parsing captures uncovered functions."""
        from src.core.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer(use_llm=False)

        lcov_report = """TN:
SF:src/auth.py
FNDA:0,validate_token
FNDA:5,hash_password
FNDA:0,check_permissions
FNF:3
FNH:1
LF:50
LH:30
end_of_record
"""

        summary = analyzer.parse_lcov(lcov_report)

        assert len(summary.files) == 1
        assert "validate_token" in summary.files[0].uncovered_functions
        assert "check_permissions" in summary.files[0].uncovered_functions
        assert "hash_password" not in summary.files[0].uncovered_functions

    def test_parse_lcov_empty_report(self, mock_env_vars):
        """Test parsing empty LCOV report."""
        from src.core.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer(use_llm=False)

        summary = analyzer.parse_lcov("")

        assert summary.lines_total == 0
        assert summary.lines_percent == 0

    def test_parse_istanbul_json_simple(self, mock_env_vars):
        """Test parsing Istanbul JSON coverage report."""
        from src.core.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer(use_llm=False)

        istanbul_report = json.dumps({
            "src/utils.js": {
                "s": {"0": 5, "1": 0, "2": 5},
                "b": {"0": [5, 2]},
                "f": {"0": 5, "1": 0},
                "fnMap": {"0": {"name": "helper"}, "1": {"name": "unused"}},
                "statementMap": {
                    "0": {"start": {"line": 1}},
                    "1": {"start": {"line": 5}},
                    "2": {"start": {"line": 10}},
                },
            }
        })

        summary = analyzer.parse_istanbul_json(istanbul_report)

        assert len(summary.files) == 1
        assert summary.files[0].path == "src/utils.js"
        assert summary.files[0].lines_total == 3
        assert summary.files[0].lines_covered == 2
        assert "unused" in summary.files[0].uncovered_functions

    def test_parse_istanbul_json_invalid(self, mock_env_vars):
        """Test parsing invalid JSON returns empty summary."""
        from src.core.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer(use_llm=False)

        summary = analyzer.parse_istanbul_json("not valid json")

        assert summary.lines_total == 0

    def test_parse_istanbul_json_multiple_files(self, mock_env_vars):
        """Test parsing Istanbul JSON with multiple files."""
        from src.core.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer(use_llm=False)

        istanbul_report = json.dumps({
            "src/file1.js": {
                "s": {"0": 5, "1": 5},
                "b": {},
                "f": {"0": 5},
                "fnMap": {"0": {"name": "func1"}},
                "statementMap": {},
            },
            "src/file2.js": {
                "s": {"0": 0, "1": 0},
                "b": {},
                "f": {},
                "fnMap": {},
                "statementMap": {},
            }
        })

        summary = analyzer.parse_istanbul_json(istanbul_report)

        assert len(summary.files) == 2
        assert summary.lines_total == 4
        assert summary.lines_covered == 2

    @pytest.mark.asyncio
    async def test_analyze_gaps_basic(self, mock_env_vars):
        """Test basic gap analysis."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary, FileCoverage

        analyzer = CoverageAnalyzer(use_llm=False)

        file_cov = FileCoverage(
            path="src/payments/checkout.py",
            lines_total=100,
            lines_covered=20,
            lines_percent=20.0,
            uncovered_lines=list(range(20, 100)),
        )

        summary = CoverageSummary(
            lines_total=100,
            lines_covered=20,
            lines_percent=20.0,
            files=[file_cov],
        )

        gaps = await analyzer.analyze_gaps(summary)

        assert len(gaps) > 0
        # Payment file should be high priority due to critical path
        assert any(g.file_path == "src/payments/checkout.py" for g in gaps)

    @pytest.mark.asyncio
    async def test_analyze_gaps_with_error_files(self, mock_env_vars):
        """Test gap analysis with error correlation."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary, FileCoverage, GapPriority

        analyzer = CoverageAnalyzer(use_llm=False)

        file_cov = FileCoverage(
            path="src/api/users.py",
            lines_total=100,
            lines_covered=50,
            lines_percent=50.0,
        )

        summary = CoverageSummary(
            lines_total=100,
            lines_covered=50,
            lines_percent=50.0,
            files=[file_cov],
        )

        gaps = await analyzer.analyze_gaps(
            summary,
            error_files=["src/api/users.py"],
        )

        # Gap should have error correlation
        user_gap = next((g for g in gaps if "users.py" in g.file_path), None)
        assert user_gap is not None
        assert user_gap.error_correlation is not None

    @pytest.mark.asyncio
    async def test_analyze_gaps_critical_paths(self, mock_env_vars):
        """Test gap analysis with custom critical paths."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary, FileCoverage

        analyzer = CoverageAnalyzer(use_llm=False)

        file_cov = FileCoverage(
            path="src/custom_critical/handler.py",
            lines_total=100,
            lines_covered=40,
            lines_percent=40.0,
        )

        summary = CoverageSummary(files=[file_cov])

        gaps = await analyzer.analyze_gaps(
            summary,
            critical_paths=["custom_critical"],
        )

        assert len(gaps) > 0
        # Should flag as higher priority due to critical path
        assert any("critical" in g.risk_reason.lower() for g in gaps)

    @pytest.mark.asyncio
    async def test_analyze_gaps_skips_excellent(self, mock_env_vars):
        """Test that excellent coverage files are skipped in gap analysis."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary, FileCoverage

        analyzer = CoverageAnalyzer(use_llm=False)

        file_cov = FileCoverage(
            path="src/well_tested.py",
            lines_total=100,
            lines_covered=95,
            lines_percent=95.0,
        )

        summary = CoverageSummary(files=[file_cov])

        gaps = await analyzer.analyze_gaps(summary)

        assert len(gaps) == 0

    def test_compare_coverage_improving(self, mock_env_vars):
        """Test coverage comparison showing improvement."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary, FileCoverage

        analyzer = CoverageAnalyzer(use_llm=False)

        previous = CoverageSummary(
            lines_total=100,
            lines_covered=70,
            lines_percent=70.0,
            files=[FileCoverage(path="test.py", lines_percent=70.0)],
        )

        current = CoverageSummary(
            lines_total=100,
            lines_covered=85,
            lines_percent=85.0,
            files=[FileCoverage(path="test.py", lines_percent=85.0)],
        )

        comparison = analyzer.compare_coverage(current, previous)

        assert comparison["trend"] == "improving"
        assert comparison["overall_change"] == 15.0

    def test_compare_coverage_declining(self, mock_env_vars):
        """Test coverage comparison showing decline."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary, FileCoverage

        analyzer = CoverageAnalyzer(use_llm=False)

        previous = CoverageSummary(
            lines_percent=80.0,
            files=[FileCoverage(path="test.py", lines_percent=80.0)],
        )

        current = CoverageSummary(
            lines_percent=65.0,
            files=[FileCoverage(path="test.py", lines_percent=65.0)],
        )

        comparison = analyzer.compare_coverage(current, previous)

        assert comparison["trend"] == "declining"
        assert comparison["overall_change"] == -15.0

    def test_compare_coverage_stable(self, mock_env_vars):
        """Test coverage comparison showing stability."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary

        analyzer = CoverageAnalyzer(use_llm=False)

        previous = CoverageSummary(lines_percent=80.0, files=[])
        current = CoverageSummary(lines_percent=81.0, files=[])

        comparison = analyzer.compare_coverage(current, previous)

        assert comparison["trend"] == "stable"

    def test_compare_coverage_regressions(self, mock_env_vars):
        """Test coverage comparison detects regressions."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary, FileCoverage

        analyzer = CoverageAnalyzer(use_llm=False)

        previous = CoverageSummary(
            lines_percent=80.0,
            files=[
                FileCoverage(path="src/good.py", lines_percent=90.0),
                FileCoverage(path="src/regressed.py", lines_percent=85.0),
            ],
        )

        current = CoverageSummary(
            lines_percent=70.0,
            files=[
                FileCoverage(path="src/good.py", lines_percent=90.0),
                FileCoverage(path="src/regressed.py", lines_percent=60.0),  # Dropped 25%
            ],
        )

        comparison = analyzer.compare_coverage(current, previous)

        assert len(comparison["regressions"]) == 1
        assert comparison["regressions"][0]["file"] == "src/regressed.py"

    def test_compare_coverage_closed_gaps(self, mock_env_vars):
        """Test coverage comparison detects closed gaps."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary, FileCoverage

        analyzer = CoverageAnalyzer(use_llm=False)

        previous = CoverageSummary(
            lines_percent=70.0,
            files=[FileCoverage(path="src/improved.py", lines_percent=40.0)],
        )

        current = CoverageSummary(
            lines_percent=80.0,
            files=[FileCoverage(path="src/improved.py", lines_percent=80.0)],  # +40%
        )

        comparison = analyzer.compare_coverage(current, previous)

        assert len(comparison["closed_gaps"]) == 1

    def test_compare_coverage_new_low_coverage_files(self, mock_env_vars):
        """Test coverage comparison detects new low coverage files."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary, FileCoverage

        analyzer = CoverageAnalyzer(use_llm=False)

        previous = CoverageSummary(
            lines_percent=80.0,
            files=[FileCoverage(path="src/existing.py", lines_percent=80.0)],
        )

        current = CoverageSummary(
            lines_percent=70.0,
            files=[
                FileCoverage(path="src/existing.py", lines_percent=80.0),
                FileCoverage(path="src/new_untested.py", lines_percent=30.0),  # New low cov
            ],
        )

        comparison = analyzer.compare_coverage(current, previous)

        assert len(comparison["new_gaps"]) == 1
        assert comparison["new_gaps"][0]["file"] == "src/new_untested.py"

    def test_get_critical_uncovered_files(self, mock_env_vars):
        """Test getting critical uncovered files."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary, FileCoverage, CoverageLevel

        analyzer = CoverageAnalyzer(use_llm=False)

        files = [
            FileCoverage(path="big_critical.py", lines_total=500, lines_percent=20.0),
            FileCoverage(path="small_critical.py", lines_total=50, lines_percent=15.0),
            FileCoverage(path="good_coverage.py", lines_total=100, lines_percent=90.0),
            FileCoverage(path="poor_big.py", lines_total=300, lines_percent=30.0),
        ]

        summary = CoverageSummary(files=files)

        critical = analyzer.get_critical_uncovered_files(summary, max_results=2)

        assert len(critical) == 2
        # Should be sorted by lines_total (bigger files first)
        assert critical[0].path == "big_critical.py"
        assert critical[1].path == "poor_big.py"

    def test_get_uncovered_functions(self, mock_env_vars):
        """Test getting uncovered functions across codebase."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary, FileCoverage

        analyzer = CoverageAnalyzer(use_llm=False)

        files = [
            FileCoverage(
                path="src/auth.py",
                lines_percent=70.0,
                uncovered_functions=["validate_token", "check_permissions"],
            ),
            FileCoverage(
                path="src/api.py",
                lines_percent=80.0,
                uncovered_functions=["handle_error"],
            ),
        ]

        summary = CoverageSummary(files=files)

        uncovered = analyzer.get_uncovered_functions(summary)

        assert len(uncovered) == 3
        assert any(f["function"] == "validate_token" for f in uncovered)
        assert any(f["function"] == "handle_error" for f in uncovered)

    def test_calculate_coverage_score_full(self, mock_env_vars):
        """Test coverage score calculation with all metrics."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary

        analyzer = CoverageAnalyzer(use_llm=False)

        summary = CoverageSummary(
            lines_total=100,
            lines_covered=80,
            lines_percent=80.0,
            branches_total=50,
            branches_covered=40,
            branches_percent=80.0,
            functions_total=20,
            functions_covered=16,
            functions_percent=80.0,
        )

        score = analyzer.calculate_coverage_score(summary)

        # 80 * 0.5 + 80 * 0.3 + 80 * 0.2 = 40 + 24 + 16 = 80
        assert score == 80

    def test_calculate_coverage_score_no_branches(self, mock_env_vars):
        """Test coverage score when no branch data available."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary

        analyzer = CoverageAnalyzer(use_llm=False)

        summary = CoverageSummary(
            lines_percent=80.0,
            branches_total=0,  # No branch data
            functions_total=20,
            functions_covered=16,
            functions_percent=80.0,
        )

        score = analyzer.calculate_coverage_score(summary)

        # Should use line percent for missing branch data
        assert 70 <= score <= 90

    def test_calculate_coverage_score_max(self, mock_env_vars):
        """Test coverage score caps at 100."""
        from src.core.coverage import CoverageAnalyzer, CoverageSummary

        analyzer = CoverageAnalyzer(use_llm=False)

        summary = CoverageSummary(
            lines_percent=100.0,
            branches_percent=100.0,
            functions_percent=100.0,
            branches_total=1,
            functions_total=1,
        )

        score = analyzer.calculate_coverage_score(summary)

        assert score == 100

    def test_create_file_coverage_zero_totals(self, mock_env_vars):
        """Test _create_file_coverage handles zero totals."""
        from src.core.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer(use_llm=False)

        data = {
            "lines_total": 0,
            "lines_covered": 0,
            "branches_total": 0,
            "branches_covered": 0,
            "functions_total": 0,
            "functions_covered": 0,
        }

        fc = analyzer._create_file_coverage("test.py", data, [])

        assert fc.lines_percent == 0
        assert fc.branches_percent == 0
        assert fc.functions_percent == 0

    @pytest.mark.asyncio
    async def test_add_llm_recommendations(self, mock_env_vars):
        """Test LLM recommendations are added to gaps."""
        from src.core.coverage import CoverageAnalyzer, CoverageGap, GapPriority

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='''[
            {
                "file": "src/auth.py",
                "test_type": "unit",
                "scenarios": ["Test token validation", "Test expiry"],
                "recommendation": "Add unit tests for auth"
            }
        ]''')]

        with patch("anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client

            analyzer = CoverageAnalyzer(use_llm=True)
            analyzer.client = mock_client

            gaps = [
                CoverageGap(
                    id="1",
                    file_path="src/auth.py",
                    priority=GapPriority.HIGH,
                    gap_type="function",
                    description="Auth code untested",
                    uncovered_functions=["validate_token"],
                ),
            ]

            result = await analyzer._add_llm_recommendations(gaps)

            assert len(result) == 1
            assert result[0].test_recommendation is not None


class TestCoverageAnalyzerLevelDistribution:
    """Tests for coverage level distribution calculation."""

    def test_level_distribution_all_levels(self, mock_env_vars):
        """Test level distribution across all coverage levels."""
        from src.core.coverage import CoverageAnalyzer, FileCoverage

        analyzer = CoverageAnalyzer(use_llm=False)

        lcov_report = """TN:
SF:excellent.py
LF:100
LH:95
end_of_record
SF:good.py
LF:100
LH:80
end_of_record
SF:fair.py
LF:100
LH:60
end_of_record
SF:poor.py
LF:100
LH:30
end_of_record
SF:critical.py
LF:100
LH:10
end_of_record
"""

        summary = analyzer.parse_lcov(lcov_report)

        assert summary.excellent_count == 1
        assert summary.good_count == 1
        assert summary.fair_count == 1
        assert summary.poor_count == 1
        assert summary.critical_count == 1

    def test_files_with_without_coverage(self, mock_env_vars):
        """Test counting files with and without coverage."""
        from src.core.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer(use_llm=False)

        lcov_report = """TN:
SF:covered.py
LF:100
LH:50
end_of_record
SF:uncovered.py
LF:100
LH:0
end_of_record
SF:partial.py
LF:100
LH:1
end_of_record
"""

        summary = analyzer.parse_lcov(lcov_report)

        assert summary.files_with_coverage == 2  # covered.py and partial.py
        assert summary.files_without_coverage == 1  # uncovered.py
