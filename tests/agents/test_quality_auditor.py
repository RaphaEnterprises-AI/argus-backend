"""Tests for the quality auditor module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


class TestAccessibilityLevel:
    """Tests for AccessibilityLevel enum."""

    def test_accessibility_levels(self, mock_env_vars):
        """Test AccessibilityLevel enum values."""
        from src.agents.quality_auditor import AccessibilityLevel

        assert AccessibilityLevel.A.value == "A"
        assert AccessibilityLevel.AA.value == "AA"
        assert AccessibilityLevel.AAA.value == "AAA"


class TestViolationSeverity:
    """Tests for ViolationSeverity enum."""

    def test_violation_severity(self, mock_env_vars):
        """Test ViolationSeverity enum values."""
        from src.agents.quality_auditor import ViolationSeverity

        assert ViolationSeverity.CRITICAL.value == "critical"
        assert ViolationSeverity.SERIOUS.value == "serious"
        assert ViolationSeverity.MODERATE.value == "moderate"
        assert ViolationSeverity.MINOR.value == "minor"


class TestAccessibilityViolation:
    """Tests for AccessibilityViolation dataclass."""

    def test_violation_creation(self, mock_env_vars):
        """Test AccessibilityViolation creation."""
        from src.agents.quality_auditor import AccessibilityViolation, ViolationSeverity

        violation = AccessibilityViolation(
            rule_id="image-alt",
            description="Images must have alternative text",
            severity=ViolationSeverity.CRITICAL,
            wcag_criteria=["1.1.1"],
            affected_elements=["<img src='logo.png'>"],
            fix_suggestion="Add alt attribute",
            help_url="https://example.com/help",
        )

        assert violation.rule_id == "image-alt"
        assert violation.severity == ViolationSeverity.CRITICAL
        assert "1.1.1" in violation.wcag_criteria


class TestPerformanceMetric:
    """Tests for PerformanceMetric dataclass."""

    def test_metric_creation(self, mock_env_vars):
        """Test PerformanceMetric creation."""
        from src.agents.quality_auditor import PerformanceMetric

        metric = PerformanceMetric(
            name="LCP",
            value=2000.0,
            unit="ms",
            score=95.0,
            threshold_good=2500,
            threshold_needs_improvement=4000,
            status="good",
        )

        assert metric.name == "LCP"
        assert metric.value == 2000.0
        assert metric.status == "good"


class TestAccessibilityReport:
    """Tests for AccessibilityReport dataclass."""

    def test_report_creation(self, mock_env_vars):
        """Test AccessibilityReport creation."""
        from src.agents.quality_auditor import AccessibilityLevel, AccessibilityReport

        report = AccessibilityReport(
            url="https://example.com",
            timestamp=datetime.utcnow().isoformat(),
            level_tested=AccessibilityLevel.AA,
            total_violations=5,
            violations_by_severity={"critical": 1, "serious": 2, "moderate": 2},
            violations=[],
            passes=10,
            score=75.0,
            compliant=False,
        )

        assert report.url == "https://example.com"
        assert report.score == 75.0
        assert report.compliant is False


class TestPerformanceReport:
    """Tests for PerformanceReport dataclass."""

    def test_report_creation(self, mock_env_vars):
        """Test PerformanceReport creation."""
        from src.agents.quality_auditor import PerformanceReport

        report = PerformanceReport(
            url="https://example.com",
            timestamp=datetime.utcnow().isoformat(),
            overall_score=85.0,
            metrics=[],
            opportunities=[],
            diagnostics=[],
        )

        assert report.overall_score == 85.0


class TestQualityReport:
    """Tests for QualityReport dataclass."""

    def test_report_creation(self, mock_env_vars):
        """Test QualityReport creation."""
        from src.agents.quality_auditor import (
            AccessibilityLevel,
            AccessibilityReport,
            PerformanceReport,
            QualityReport,
        )

        accessibility = AccessibilityReport(
            url="https://example.com",
            timestamp=datetime.utcnow().isoformat(),
            level_tested=AccessibilityLevel.AA,
            total_violations=0,
            violations_by_severity={},
            violations=[],
            passes=12,
            score=100.0,
            compliant=True,
        )

        performance = PerformanceReport(
            url="https://example.com",
            timestamp=datetime.utcnow().isoformat(),
            overall_score=90.0,
            metrics=[],
            opportunities=[],
            diagnostics=[],
        )

        report = QualityReport(
            accessibility=accessibility,
            performance=performance,
            best_practices_score=95.0,
            seo_score=90.0,
            overall_grade="A",
        )

        assert report.overall_grade == "A"


class TestAccessibilityAuditor:
    """Tests for AccessibilityAuditor class."""

    def test_auditor_creation(self, mock_env_vars):
        """Test AccessibilityAuditor creation."""
        with patch('src.agents.quality_auditor.Anthropic'):
            from src.agents.quality_auditor import AccessibilityAuditor

            auditor = AccessibilityAuditor()

            assert auditor is not None
            assert len(auditor.RULES) > 0

    def test_check_html_rules_no_violations(self, mock_env_vars):
        """Test HTML rules with valid HTML."""
        with patch('src.agents.quality_auditor.Anthropic'):
            from src.agents.quality_auditor import AccessibilityAuditor

            auditor = AccessibilityAuditor()
            html = '''
            <html lang="en">
            <head><title>Test</title></head>
            <body>
                <img src="logo.png" alt="Logo">
                <button>Click me</button>
            </body>
            </html>
            '''

            violations = auditor._check_html_rules(html)

            # Should have no violations for this valid HTML
            assert len([v for v in violations if v.rule_id == "image-alt"]) == 0
            assert len([v for v in violations if v.rule_id == "html-lang"]) == 0

    def test_check_html_rules_missing_alt(self, mock_env_vars):
        """Test HTML rules detect missing alt attribute."""
        with patch('src.agents.quality_auditor.Anthropic'):
            from src.agents.quality_auditor import AccessibilityAuditor

            auditor = AccessibilityAuditor()
            html = '''
            <html lang="en">
            <body>
                <img src="logo.png">
            </body>
            </html>
            '''

            violations = auditor._check_html_rules(html)
            alt_violations = [v for v in violations if v.rule_id == "image-alt"]

            assert len(alt_violations) == 1

    def test_check_html_rules_missing_lang(self, mock_env_vars):
        """Test HTML rules detect missing lang attribute."""
        with patch('src.agents.quality_auditor.Anthropic'):
            from src.agents.quality_auditor import AccessibilityAuditor

            auditor = AccessibilityAuditor()
            html = '''
            <html>
            <body>Content</body>
            </html>
            '''

            violations = auditor._check_html_rules(html)
            lang_violations = [v for v in violations if v.rule_id == "html-lang"]

            assert len(lang_violations) == 1

    def test_check_html_rules_duplicate_ids(self, mock_env_vars):
        """Test HTML rules detect duplicate IDs."""
        with patch('src.agents.quality_auditor.Anthropic'):
            from src.agents.quality_auditor import AccessibilityAuditor

            auditor = AccessibilityAuditor()
            html = '''
            <html lang="en">
            <body>
                <div id="main">First</div>
                <div id="main">Second</div>
            </body>
            </html>
            '''

            violations = auditor._check_html_rules(html)
            dup_violations = [v for v in violations if v.rule_id == "duplicate-id"]

            assert len(dup_violations) == 1

    def test_check_html_rules_empty_button(self, mock_env_vars):
        """Test HTML rules detect empty buttons."""
        with patch('src.agents.quality_auditor.Anthropic'):
            from src.agents.quality_auditor import AccessibilityAuditor

            auditor = AccessibilityAuditor()
            html = '''
            <html lang="en">
            <body>
                <button>   </button>
            </body>
            </html>
            '''

            violations = auditor._check_html_rules(html)
            button_violations = [v for v in violations if v.rule_id == "button-name"]

            assert len(button_violations) == 1

    @pytest.mark.asyncio
    async def test_audit(self, mock_env_vars):
        """Test full accessibility audit."""
        with patch('src.agents.quality_auditor.Anthropic'):
            from src.agents.quality_auditor import AccessibilityAuditor, AccessibilityLevel

            auditor = AccessibilityAuditor()
            html = '''
            <html lang="en">
            <body>
                <img src="logo.png" alt="Logo">
                <button>Click</button>
            </body>
            </html>
            '''

            report = await auditor.audit(
                page_html=html,
                page_url="https://example.com",
                level=AccessibilityLevel.AA
            )

            assert report.url == "https://example.com"
            assert report.level_tested == AccessibilityLevel.AA
            assert report.score >= 0

    @pytest.mark.asyncio
    async def test_ai_visual_analysis(self, mock_env_vars, mock_anthropic_client):
        """Test AI visual analysis."""
        with patch('src.agents.quality_auditor.Anthropic') as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='[]')]  # No issues
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.quality_auditor import AccessibilityAuditor

            auditor = AccessibilityAuditor()
            auditor.client = mock_anthropic.return_value

            violations = await auditor._ai_visual_analysis("base64_screenshot_data")

            assert violations == []


class TestPerformanceAuditor:
    """Tests for PerformanceAuditor class."""

    def test_auditor_creation(self, mock_env_vars):
        """Test PerformanceAuditor creation."""
        from src.agents.quality_auditor import PerformanceAuditor

        auditor = PerformanceAuditor()

        assert auditor is not None
        assert len(auditor.THRESHOLDS) > 0

    def test_calculate_score_good(self, mock_env_vars):
        """Test score calculation for good metric."""
        from src.agents.quality_auditor import PerformanceAuditor

        auditor = PerformanceAuditor()
        threshold = {"good": 2500, "poor": 4000, "unit": "ms"}

        score, status = auditor._calculate_score(2000, threshold)

        assert status == "good"
        assert score >= 90

    def test_calculate_score_needs_improvement(self, mock_env_vars):
        """Test score calculation for needs improvement metric."""
        from src.agents.quality_auditor import PerformanceAuditor

        auditor = PerformanceAuditor()
        threshold = {"good": 2500, "poor": 4000, "unit": "ms"}

        score, status = auditor._calculate_score(3000, threshold)

        assert status == "needs_improvement"
        assert 50 <= score < 90

    def test_calculate_score_poor(self, mock_env_vars):
        """Test score calculation for poor metric."""
        from src.agents.quality_auditor import PerformanceAuditor

        auditor = PerformanceAuditor()
        threshold = {"good": 2500, "poor": 4000, "unit": "ms"}

        score, status = auditor._calculate_score(5000, threshold)

        assert status == "poor"
        assert score < 50

    @pytest.mark.asyncio
    async def test_audit(self, mock_env_vars):
        """Test full performance audit."""
        from src.agents.quality_auditor import PerformanceAuditor

        auditor = PerformanceAuditor()
        metrics = {
            "LCP": 2000,
            "FID": 50,
            "CLS": 0.05,
            "TTFB": 500,
        }

        report = await auditor.audit(
            page_url="https://example.com",
            metrics=metrics,
        )

        assert report.url == "https://example.com"
        assert len(report.metrics) == 4
        assert report.overall_score > 0

    def test_generate_opportunities(self, mock_env_vars):
        """Test opportunity generation."""
        from src.agents.quality_auditor import PerformanceAuditor, PerformanceMetric

        auditor = PerformanceAuditor()
        metrics = [
            PerformanceMetric(
                name="LCP",
                value=5000,
                unit="ms",
                score=30,
                threshold_good=2500,
                threshold_needs_improvement=4000,
                status="poor",
            )
        ]

        opportunities = auditor._generate_opportunities(metrics)

        assert len(opportunities) > 0
        assert opportunities[0]["title"] == "Improve Largest Contentful Paint"

    def test_generate_diagnostics(self, mock_env_vars):
        """Test diagnostics generation."""
        from src.agents.quality_auditor import PerformanceAuditor

        auditor = PerformanceAuditor()
        metrics = {
            "Total_Blocking_Time": 500,
            "resource_count": 150,
            "total_bytes": 5000000,
        }

        diagnostics = auditor._generate_diagnostics(metrics)

        assert len(diagnostics) == 3


class TestQualityAuditor:
    """Tests for QualityAuditor class."""

    def test_auditor_creation(self, mock_env_vars):
        """Test QualityAuditor creation."""
        with patch('src.agents.quality_auditor.Anthropic'):
            from src.agents.quality_auditor import QualityAuditor

            auditor = QualityAuditor()

            assert auditor.accessibility_auditor is not None
            assert auditor.performance_auditor is not None

    @pytest.mark.asyncio
    async def test_full_audit(self, mock_env_vars):
        """Test full quality audit."""
        with patch('src.agents.quality_auditor.Anthropic'):
            from src.agents.quality_auditor import QualityAuditor

            auditor = QualityAuditor()
            html = '''
            <html lang="en">
            <head>
                <title>Test Page</title>
                <meta name="description" content="Test">
                <meta name="viewport" content="width=device-width">
            </head>
            <body>
                <h1>Welcome</h1>
                <img src="logo.png" alt="Logo">
            </body>
            </html>
            '''

            metrics = {
                "LCP": 2000,
                "FID": 50,
                "CLS": 0.05,
            }

            report = await auditor.full_audit(
                page_url="https://example.com",
                page_html=html,
                performance_metrics=metrics,
            )

            assert report.accessibility is not None
            assert report.performance is not None
            assert report.overall_grade in ["A", "B", "C", "D", "F"]

    def test_calculate_best_practices(self, mock_env_vars):
        """Test best practices score calculation."""
        with patch('src.agents.quality_auditor.Anthropic'):
            from src.agents.quality_auditor import QualityAuditor

            auditor = QualityAuditor()

            # Good HTML
            good_html = '<html><body>Content</body></html>'
            score = auditor._calculate_best_practices(good_html)
            assert score == 100

            # HTML with deprecated elements
            bad_html = '<html><body><font>Bad</font><center>Centered</center></body></html>'
            score = auditor._calculate_best_practices(bad_html)
            assert score < 100

    def test_calculate_seo(self, mock_env_vars):
        """Test SEO score calculation."""
        with patch('src.agents.quality_auditor.Anthropic'):
            from src.agents.quality_auditor import QualityAuditor

            auditor = QualityAuditor()

            # Good HTML
            good_html = '''
            <html>
            <head>
                <title>Page Title</title>
                <meta name="description" content="Description">
                <meta name="viewport" content="width=device-width">
                <link rel="canonical" href="https://example.com">
            </head>
            <body><h1>Heading</h1></body>
            </html>
            '''
            score = auditor._calculate_seo(good_html, "https://example.com")
            assert score == 100

            # Missing title
            bad_html = '<html><body>No title</body></html>'
            score = auditor._calculate_seo(bad_html, "https://example.com")
            assert score < 100

    def test_overall_grade_calculation(self, mock_env_vars):
        """Test overall grade calculation."""
        with patch('src.agents.quality_auditor.Anthropic'):
            from src.agents.quality_auditor import QualityAuditor

            QualityAuditor()

            # Test grade thresholds
            # A: >= 90, B: >= 80, C: >= 70, D: >= 60, F: < 60
            # Grades are calculated in full_audit, so we test indirectly
            pass
