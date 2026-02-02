"""Tests for Performance Analyzer Agent - comprehensive Core Web Vitals testing.

Following industry best practices:
- Parametrization for edge cases
- Proper mocking for external dependencies
- Async testing with pytest-asyncio
- Clear test naming with docstrings
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.performance_analyzer import (
    PerformanceGrade,
    CoreWebVitals,
    PerformanceMetrics,
    PerformanceIssue,
    PerformanceAnalysisResult,
    PerformanceAnalyzerAgent,
)


class TestPerformanceGradeEnum:
    """Tests for PerformanceGrade enum."""

    @pytest.mark.parametrize("grade,expected_value", [
        (PerformanceGrade.EXCELLENT, "excellent"),
        (PerformanceGrade.GOOD, "good"),
        (PerformanceGrade.NEEDS_WORK, "needs_work"),
        (PerformanceGrade.POOR, "poor"),
    ])
    def test_performance_grade_values(self, mock_env_vars, grade, expected_value):
        """Test PerformanceGrade enum values."""
        assert grade.value == expected_value


class TestCoreWebVitals:
    """Tests for CoreWebVitals dataclass."""

    def test_core_web_vitals_creation(self, mock_env_vars):
        """Test CoreWebVitals creation."""
        vitals = CoreWebVitals(
            lcp_ms=2000.0,
            fid_ms=50.0,
            cls=0.05,
            inp_ms=100.0,
        )

        assert vitals.lcp_ms == 2000.0
        assert vitals.fid_ms == 50.0
        assert vitals.cls == 0.05
        assert vitals.inp_ms == 100.0

    @pytest.mark.parametrize("lcp_ms,expected_grade", [
        (2000.0, PerformanceGrade.EXCELLENT),  # Under 2500ms
        (2500.0, PerformanceGrade.EXCELLENT),  # At threshold
        (3000.0, PerformanceGrade.NEEDS_WORK),  # Between 2500-4000
        (4000.0, PerformanceGrade.NEEDS_WORK),  # At upper threshold
        (5000.0, PerformanceGrade.POOR),  # Over 4000ms
    ])
    def test_get_lcp_grade(self, mock_env_vars, lcp_ms, expected_grade):
        """Test LCP grading based on thresholds."""
        vitals = CoreWebVitals(lcp_ms=lcp_ms, fid_ms=0, cls=0, inp_ms=0)
        assert vitals.get_lcp_grade() == expected_grade

    @pytest.mark.parametrize("fid_ms,expected_grade", [
        (50.0, PerformanceGrade.EXCELLENT),   # Under 100ms
        (100.0, PerformanceGrade.EXCELLENT),  # At threshold
        (200.0, PerformanceGrade.NEEDS_WORK), # Between 100-300
        (300.0, PerformanceGrade.NEEDS_WORK), # At upper threshold
        (400.0, PerformanceGrade.POOR),       # Over 300ms
    ])
    def test_get_fid_grade(self, mock_env_vars, fid_ms, expected_grade):
        """Test FID grading based on thresholds."""
        vitals = CoreWebVitals(lcp_ms=0, fid_ms=fid_ms, cls=0, inp_ms=0)
        assert vitals.get_fid_grade() == expected_grade

    @pytest.mark.parametrize("cls,expected_grade", [
        (0.05, PerformanceGrade.EXCELLENT),   # Under 0.1
        (0.1, PerformanceGrade.EXCELLENT),    # At threshold
        (0.15, PerformanceGrade.NEEDS_WORK),  # Between 0.1-0.25
        (0.25, PerformanceGrade.NEEDS_WORK),  # At upper threshold
        (0.3, PerformanceGrade.POOR),         # Over 0.25
    ])
    def test_get_cls_grade(self, mock_env_vars, cls, expected_grade):
        """Test CLS grading based on thresholds."""
        vitals = CoreWebVitals(lcp_ms=0, fid_ms=0, cls=cls, inp_ms=0)
        assert vitals.get_cls_grade() == expected_grade


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self, mock_env_vars):
        """Test PerformanceMetrics creation with all fields."""
        vitals = CoreWebVitals(lcp_ms=2000, fid_ms=50, cls=0.05, inp_ms=100)
        metrics = PerformanceMetrics(
            core_vitals=vitals,
            ttfb_ms=200.0,
            fcp_ms=1000.0,
            speed_index=1500.0,
            tti_ms=3000.0,
            tbt_ms=150.0,
            total_requests=50,
            total_transfer_size_kb=500.0,
            js_execution_time_ms=200.0,
            dom_content_loaded_ms=1500.0,
            load_time_ms=3000.0,
            performance_score=85,
            accessibility_score=90,
            best_practices_score=95,
            seo_score=88,
        )

        assert metrics.ttfb_ms == 200.0
        assert metrics.total_requests == 50
        assert metrics.performance_score == 85

    def test_performance_metrics_defaults(self, mock_env_vars):
        """Test PerformanceMetrics default values."""
        vitals = CoreWebVitals(lcp_ms=0, fid_ms=0, cls=0, inp_ms=0)
        metrics = PerformanceMetrics(core_vitals=vitals)

        assert metrics.ttfb_ms == 0.0
        assert metrics.total_requests == 0
        assert metrics.performance_score == 0


class TestPerformanceIssue:
    """Tests for PerformanceIssue dataclass."""

    def test_performance_issue_creation(self, mock_env_vars):
        """Test PerformanceIssue creation."""
        issue = PerformanceIssue(
            category="render-blocking",
            severity="critical",
            title="Render-blocking JavaScript",
            description="Scripts blocking initial render",
            savings_ms=500.0,
            savings_kb=100.0,
            affected_resources=["main.js", "vendor.js"],
            fix_suggestion="Add async or defer to script tags",
        )

        assert issue.category == "render-blocking"
        assert issue.severity == "critical"
        assert issue.savings_ms == 500.0
        assert len(issue.affected_resources) == 2

    @pytest.mark.parametrize("category", [
        "render-blocking",
        "large-resources",
        "unused-code",
        "unoptimized-images",
        "no-caching",
        "excessive-dom",
    ])
    def test_performance_issue_categories(self, mock_env_vars, category):
        """Test various performance issue categories."""
        issue = PerformanceIssue(
            category=category,
            severity="medium",
            title=f"Issue in {category}",
            description="Test issue",
        )
        assert issue.category == category


class TestPerformanceAnalysisResult:
    """Tests for PerformanceAnalysisResult dataclass."""

    def test_performance_analysis_result_creation(self, mock_env_vars):
        """Test PerformanceAnalysisResult creation."""
        vitals = CoreWebVitals(lcp_ms=2000, fid_ms=50, cls=0.05, inp_ms=100)
        metrics = PerformanceMetrics(core_vitals=vitals)

        result = PerformanceAnalysisResult(
            url="https://example.com",
            metrics=metrics,
            issues=[
                PerformanceIssue(
                    category="render-blocking",
                    severity="high",
                    title="JS blocking",
                    description="",
                )
            ],
            recommendations=["Enable compression", "Use CDN"],
            overall_grade=PerformanceGrade.GOOD,
            summary="Page performs well with some areas for improvement",
        )

        assert result.url == "https://example.com"
        assert result.overall_grade == PerformanceGrade.GOOD
        assert len(result.issues) == 1
        assert len(result.recommendations) == 2


class TestPerformanceAnalyzerAgent:
    """Tests for PerformanceAnalyzerAgent class."""

    def test_agent_initialization(self, mock_env_vars):
        """Test agent initialization."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = PerformanceAnalyzerAgent()
            assert agent is not None

    def test_get_system_prompt(self, mock_env_vars):
        """Test system prompt generation."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = PerformanceAnalyzerAgent()
            prompt = agent._get_system_prompt()

            assert "performance" in prompt.lower()
            assert "Core Web Vitals" in prompt or "Lighthouse" in prompt

    @pytest.mark.asyncio
    async def test_collect_metrics(self, mock_env_vars):
        """Test metrics collection structure."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = PerformanceAnalyzerAgent()
            metrics = await agent._collect_metrics("https://example.com", "mobile")

            assert "navigation_timing" in metrics
            assert "resource_timing" in metrics
            assert "layout_shifts" in metrics
            assert "long_tasks" in metrics

    @pytest.mark.asyncio
    async def test_analyze_metrics(self, mock_env_vars):
        """Test AI-powered metrics analysis."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = PerformanceAnalyzerAgent()

            # Mock _call_ai directly on the agent instance
            mock_response = MagicMock()
            mock_response.content = '''{
                "core_vitals": {
                    "lcp_ms": 2500,
                    "lcp_grade": "excellent",
                    "fid_ms": 100,
                    "fid_grade": "excellent",
                    "cls": 0.1,
                    "cls_grade": "excellent",
                    "inp_ms": 150
                },
                "issues": [
                    {
                        "category": "render-blocking",
                        "severity": "high",
                        "title": "Render-blocking scripts",
                        "description": "Multiple scripts blocking render",
                        "savings_ms": 300,
                        "fix_suggestion": "Defer non-critical scripts"
                    }
                ],
                "recommendations": ["Enable text compression", "Optimize images"],
                "overall_grade": "good",
                "summary": "Good performance with room for improvement"
            }'''

            with patch.object(agent, '_call_model', new_callable=AsyncMock) as mock_ai:
                mock_ai.return_value = {"content": mock_response.content}

                result = await agent._analyze_metrics(
                    "https://example.com",
                    {"navigation_timing": {}, "resource_timing": []},
                    "mobile",
                )

                assert result.overall_grade == PerformanceGrade.GOOD
                assert len(result.issues) == 1
                assert result.metrics.core_vitals.lcp_ms == 2500

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_env_vars):
        """Test successful performance analysis execution."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = PerformanceAnalyzerAgent()

            mock_response = MagicMock()
            mock_response.content = '''{
                "core_vitals": {"lcp_ms": 2000, "fid_ms": 50, "cls": 0.05, "inp_ms": 100},
                "issues": [],
                "recommendations": [],
                "overall_grade": "excellent",
                "summary": "Excellent performance"
            }'''

            with patch.object(agent, '_call_model', new_callable=AsyncMock) as mock_ai:
                mock_ai.return_value = {"content": mock_response.content}

                result = await agent.execute(
                    url="https://example.com",
                    device="mobile",
                )

                assert result.success is True
                assert result.data is not None
                assert result.data.overall_grade == PerformanceGrade.EXCELLENT

    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_env_vars):
        """Test performance analysis execution failure."""
        with patch("src.agents.base.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = Exception("API error")
            mock_anthropic.return_value = mock_client

            agent = PerformanceAnalyzerAgent()
            result = await agent.execute(url="https://example.com")

            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_execute_with_different_devices(self, mock_env_vars):
        """Test performance analysis on different devices."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = PerformanceAnalyzerAgent()

            mock_response = MagicMock()
            mock_response.content = '''{
                "core_vitals": {"lcp_ms": 2500, "fid_ms": 80, "cls": 0.08, "inp_ms": 120},
                "issues": [],
                "recommendations": [],
                "overall_grade": "good",
                "summary": "Good"
            }'''

            with patch.object(agent, '_call_model', new_callable=AsyncMock) as mock_ai:
                mock_ai.return_value = {"content": mock_response.content}

                for device in ["mobile", "desktop"]:
                    result = await agent.execute(url="https://example.com", device=device)
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_compare_before_after(self, mock_env_vars):
        """Test before/after performance comparison."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = PerformanceAnalyzerAgent()

            mock_response = MagicMock()
            mock_response.content = '''{
                "improvements": ["LCP improved by 20%", "CLS reduced"],
                "regressions": ["FID slightly increased"],
                "net_impact": "positive",
                "recommendation": "ship"
            }'''

            with patch.object(agent, '_call_model', new_callable=AsyncMock) as mock_ai:
                mock_ai.return_value = {"content": mock_response.content}

                before = PerformanceMetrics(
                    core_vitals=CoreWebVitals(lcp_ms=3000, fid_ms=80, cls=0.15, inp_ms=150),
                    load_time_ms=4000,
                )
                after = PerformanceMetrics(
                    core_vitals=CoreWebVitals(lcp_ms=2400, fid_ms=90, cls=0.08, inp_ms=140),
                    load_time_ms=3200,
                )

                result = await agent.compare_before_after(before, after)

                assert result is not None
                assert "improvements" in result or "recommendation" in result

    @pytest.mark.parametrize("grade_str,expected_grade", [
        ("excellent", PerformanceGrade.EXCELLENT),
        ("good", PerformanceGrade.GOOD),
        ("needs_work", PerformanceGrade.NEEDS_WORK),
        ("poor", PerformanceGrade.POOR),
    ])
    @pytest.mark.asyncio
    async def test_analyze_metrics_grades(self, mock_env_vars, grade_str, expected_grade):
        """Test grade parsing in analysis results."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = PerformanceAnalyzerAgent()

            mock_response = MagicMock()
            mock_response.content = f'''{{
                "core_vitals": {{"lcp_ms": 2000, "fid_ms": 50, "cls": 0.05, "inp_ms": 100}},
                "issues": [],
                "recommendations": [],
                "overall_grade": "{grade_str}",
                "summary": "Test"
            }}'''

            with patch.object(agent, '_call_model', new_callable=AsyncMock) as mock_ai:
                mock_ai.return_value = {"content": mock_response.content}

                result = await agent._analyze_metrics("https://example.com", {}, "mobile")

                assert result.overall_grade == expected_grade
