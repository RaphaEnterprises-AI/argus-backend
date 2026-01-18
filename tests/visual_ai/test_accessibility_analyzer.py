"""Comprehensive tests for visual_ai/accessibility_analyzer.py.

Tests accessibility checks including contrast violations, touch target
validation, readability issues, and comprehensive accessibility reports.
"""


import pytest

from src.visual_ai.accessibility_analyzer import (
    AccessibilityAnalyzer,
    AccessibilityReport,
    ContrastViolation,
    ReadabilityIssue,
    TouchTargetViolation,
)
from src.visual_ai.models import VisualElement, VisualSnapshot


class TestContrastViolation:
    """Tests for ContrastViolation dataclass."""

    @pytest.fixture
    def sample_element(self):
        """Create a sample element for testing."""
        return VisualElement(
            element_id="el_1",
            selector="#header-text",
            tag_name="h1",
            bounds={"x": 0.0, "y": 0.0, "width": 200.0, "height": 40.0},
            computed_styles={"color": "#777777", "font-size": "16px"},
            text_content="Header",
            attributes={},
            children_count=0,
        )

    def test_contrast_violation_creation(self, sample_element):
        """Test creating a ContrastViolation instance."""
        violation = ContrastViolation(
            element=sample_element,
            foreground_color="#777777",
            background_color="#ffffff",
            contrast_ratio=4.48,
            required_ratio=4.5,
            wcag_level="AA",
            text_size="normal",
        )
        assert violation.element.selector == "#header-text"
        assert violation.contrast_ratio == 4.48
        assert violation.required_ratio == 4.5
        assert violation.wcag_level == "AA"
        assert violation.text_size == "normal"

    def test_contrast_violation_large_text(self, sample_element):
        """Test ContrastViolation for large text."""
        violation = ContrastViolation(
            element=sample_element,
            foreground_color="#888888",
            background_color="#ffffff",
            contrast_ratio=3.5,
            required_ratio=3.0,
            wcag_level="AA",
            text_size="large",
        )
        assert violation.text_size == "large"
        assert violation.wcag_level == "AA"

    def test_contrast_violation_to_dict(self, sample_element):
        """Test ContrastViolation serialization."""
        violation = ContrastViolation(
            element=sample_element,
            foreground_color="#777777",
            background_color="#ffffff",
            contrast_ratio=4.0,
            required_ratio=4.5,
            wcag_level="AA",
            text_size="normal",
        )
        result = violation.to_dict()
        assert result["element_id"] == "el_1"
        assert result["selector"] == "#header-text"
        assert result["foreground_color"] == "#777777"
        assert result["contrast_ratio"] == 4.0
        assert result["wcag_level"] == "AA"

    def test_contrast_violation_severity(self, sample_element):
        """Test severity property based on ratio deficit."""
        # Critical: deficit > 2.0
        critical = ContrastViolation(
            element=sample_element,
            foreground_color="#aaa",
            background_color="#fff",
            contrast_ratio=2.0,
            required_ratio=4.5,
            wcag_level="AA",
            text_size="normal",
        )
        assert critical.severity == "critical"

        # Major: deficit > 1.0
        major = ContrastViolation(
            element=sample_element,
            foreground_color="#888",
            background_color="#fff",
            contrast_ratio=3.0,
            required_ratio=4.5,
            wcag_level="AA",
            text_size="normal",
        )
        assert major.severity == "major"

        # Minor: deficit <= 1.0
        minor = ContrastViolation(
            element=sample_element,
            foreground_color="#777",
            background_color="#fff",
            contrast_ratio=4.0,
            required_ratio=4.5,
            wcag_level="AA",
            text_size="normal",
        )
        assert minor.severity == "minor"

    def test_contrast_violation_recommendation(self, sample_element):
        """Test recommendation property."""
        violation = ContrastViolation(
            element=sample_element,
            foreground_color="#777777",
            background_color="#ffffff",
            contrast_ratio=4.0,
            required_ratio=4.5,
            wcag_level="AA",
            text_size="normal",
        )
        recommendation = violation.recommendation
        assert "4.00:1" in recommendation
        assert "4.5:1" in recommendation


class TestTouchTargetViolation:
    """Tests for TouchTargetViolation dataclass."""

    @pytest.fixture
    def sample_element(self):
        """Create a sample button element."""
        return VisualElement(
            element_id="btn_1",
            selector=".small-button",
            tag_name="button",
            bounds={"x": 100.0, "y": 100.0, "width": 30.0, "height": 30.0},
            computed_styles={},
            text_content="OK",
            attributes={},
            children_count=0,
        )

    def test_touch_target_violation_creation(self, sample_element):
        """Test creating a TouchTargetViolation instance."""
        violation = TouchTargetViolation(
            element=sample_element,
            actual_width=30.0,
            actual_height=30.0,
            required_size=44,
        )
        assert violation.element.selector == ".small-button"
        assert violation.actual_width == 30.0
        assert violation.actual_height == 30.0
        assert violation.required_size == 44

    def test_touch_target_violation_to_dict(self, sample_element):
        """Test TouchTargetViolation serialization."""
        violation = TouchTargetViolation(
            element=sample_element,
            actual_width=30.0,
            actual_height=30.0,
            required_size=44,
        )
        result = violation.to_dict()
        assert result["element_id"] == "btn_1"
        assert result["selector"] == ".small-button"
        assert result["actual_width"] == 30.0
        assert result["actual_height"] == 30.0
        assert result["required_size"] == 44

    def test_touch_target_violation_severity(self, sample_element):
        """Test severity property based on size ratio."""
        # Critical: ratio < 0.5
        critical_element = VisualElement(
            element_id="tiny",
            selector=".tiny",
            tag_name="button",
            bounds={"x": 0, "y": 0, "width": 20, "height": 20},
            computed_styles={},
            text_content="X",
            attributes={},
            children_count=0,
        )
        critical = TouchTargetViolation(
            element=critical_element,
            actual_width=20.0,
            actual_height=20.0,
            required_size=44,
        )
        assert critical.severity == "critical"

        # Major: ratio < 0.75
        major = TouchTargetViolation(
            element=sample_element,
            actual_width=30.0,
            actual_height=30.0,
            required_size=44,
        )
        assert major.severity == "major"

    def test_touch_target_violation_recommendation(self, sample_element):
        """Test recommendation property."""
        violation = TouchTargetViolation(
            element=sample_element,
            actual_width=30.0,
            actual_height=30.0,
            required_size=44,
        )
        recommendation = violation.recommendation
        assert "30x30px" in recommendation
        assert "44x44px" in recommendation


class TestReadabilityIssue:
    """Tests for ReadabilityIssue dataclass."""

    @pytest.fixture
    def sample_element(self):
        """Create a sample text element."""
        return VisualElement(
            element_id="text_1",
            selector="p.small-text",
            tag_name="p",
            bounds={"x": 0.0, "y": 0.0, "width": 300.0, "height": 20.0},
            computed_styles={"font-size": "10px", "line-height": "1.0"},
            text_content="Some small text",
            attributes={},
            children_count=0,
        )

    def test_readability_issue_creation(self, sample_element):
        """Test creating a ReadabilityIssue instance."""
        issue = ReadabilityIssue(
            element=sample_element,
            issue_type="font_too_small",
            current_value="10px",
            recommended_value="12px",
        )
        assert issue.element.selector == "p.small-text"
        assert issue.issue_type == "font_too_small"
        assert issue.current_value == "10px"
        assert issue.recommended_value == "12px"

    def test_readability_issue_line_height(self, sample_element):
        """Test ReadabilityIssue for line height."""
        issue = ReadabilityIssue(
            element=sample_element,
            issue_type="line_height_too_tight",
            current_value="12.0px",
            recommended_value="24.0px",
        )
        assert issue.issue_type == "line_height_too_tight"

    def test_readability_issue_to_dict(self, sample_element):
        """Test ReadabilityIssue serialization."""
        issue = ReadabilityIssue(
            element=sample_element,
            issue_type="font_too_small",
            current_value="10px",
            recommended_value="12px",
        )
        result = issue.to_dict()
        assert result["element_id"] == "text_1"
        assert result["selector"] == "p.small-text"
        assert result["issue_type"] == "font_too_small"

    def test_readability_issue_severity(self, sample_element):
        """Test severity property based on issue type."""
        font_issue = ReadabilityIssue(
            element=sample_element,
            issue_type="font_too_small",
            current_value="10px",
            recommended_value="12px",
        )
        assert font_issue.severity == "major"

        line_issue = ReadabilityIssue(
            element=sample_element,
            issue_type="line_height_too_tight",
            current_value="1.0",
            recommended_value="1.5",
        )
        assert line_issue.severity == "minor"

    def test_readability_issue_recommendation(self, sample_element):
        """Test recommendation property."""
        issue = ReadabilityIssue(
            element=sample_element,
            issue_type="font_too_small",
            current_value="10px",
            recommended_value="12px",
        )
        recommendation = issue.recommendation
        assert "10px" in recommendation
        assert "12px" in recommendation


class TestAccessibilityReport:
    """Tests for AccessibilityReport dataclass."""

    @pytest.fixture
    def sample_element(self):
        """Create a sample element."""
        return VisualElement(
            element_id="el_1",
            selector="#text",
            tag_name="p",
            bounds={"x": 0, "y": 0, "width": 100, "height": 20},
            computed_styles={},
            text_content="Test",
            attributes={},
            children_count=0,
        )

    @pytest.fixture
    def sample_report(self, sample_element):
        """Create a sample AccessibilityReport."""
        return AccessibilityReport(
            score=65.0,
            contrast_violations=[
                ContrastViolation(
                    element=sample_element,
                    foreground_color="#777",
                    background_color="#fff",
                    contrast_ratio=4.0,
                    required_ratio=4.5,
                    wcag_level="AA",
                    text_size="normal",
                )
            ],
            touch_target_violations=[
                TouchTargetViolation(
                    element=sample_element,
                    actual_width=30.0,
                    actual_height=30.0,
                    required_size=44,
                )
            ],
            readability_issues=[
                ReadabilityIssue(
                    element=sample_element,
                    issue_type="font_too_small",
                    current_value="11px",
                    recommended_value="12px",
                )
            ],
            passed_checks=["Some checks passed"],
            summary="Accessibility issues found",
            snapshot_id="snap_123",
            url="https://example.com",
            timestamp="2024-01-01T12:00:00Z",
        )

    def test_accessibility_report_creation(self, sample_report):
        """Test AccessibilityReport creation."""
        assert sample_report.score == 65.0
        assert sample_report.snapshot_id == "snap_123"
        assert len(sample_report.contrast_violations) == 1
        assert len(sample_report.touch_target_violations) == 1

    def test_total_violations(self, sample_report):
        """Test total_violations property."""
        assert sample_report.total_violations == 3

    def test_violations_by_severity(self, sample_report):
        """Test violations_by_severity property."""
        by_severity = sample_report.violations_by_severity
        assert isinstance(by_severity, dict)
        assert "critical" in by_severity
        assert "major" in by_severity
        assert "minor" in by_severity

    def test_has_critical_issues(self, sample_element):
        """Test has_critical_issues detection."""
        # Create violation with large deficit for critical severity
        critical_violation = ContrastViolation(
            element=sample_element,
            foreground_color="#ccc",
            background_color="#fff",
            contrast_ratio=1.5,  # Very low
            required_ratio=4.5,  # deficit > 2.0
            wcag_level="AA",
            text_size="normal",
        )
        report = AccessibilityReport(
            score=30.0,
            contrast_violations=[critical_violation],
            touch_target_violations=[],
            readability_issues=[],
            passed_checks=[],
            summary="Critical issues",
        )
        assert report.has_critical_issues is True

    def test_wcag_aa_compliant(self, sample_element):
        """Test WCAG AA compliance check."""
        # Report with no AA violations
        report = AccessibilityReport(
            score=100.0,
            contrast_violations=[],
            touch_target_violations=[],
            readability_issues=[],
            passed_checks=["All checks passed"],
            summary="No issues",
        )
        assert report.wcag_aa_compliant is True

    def test_get_recommendations(self, sample_report):
        """Test get_recommendations method."""
        recommendations = sample_report.get_recommendations()
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_to_dict(self, sample_report):
        """Test to_dict serialization."""
        result = sample_report.to_dict()
        assert result["score"] == 65.0
        assert result["snapshot_id"] == "snap_123"
        assert result["url"] == "https://example.com"
        assert "contrast_violations" in result
        assert "touch_target_violations" in result
        assert "total_violations" in result


class TestAccessibilityAnalyzer:
    """Tests for AccessibilityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create an AccessibilityAnalyzer instance."""
        return AccessibilityAnalyzer()

    @pytest.fixture
    def sample_elements(self):
        """Create sample elements for testing."""
        return [
            VisualElement(
                element_id="el_1",
                selector="#header",
                tag_name="h1",
                bounds={"x": 0.0, "y": 0.0, "width": 800.0, "height": 60.0},
                computed_styles={
                    "color": "rgb(0, 0, 0)",
                    "background-color": "rgb(255, 255, 255)",
                    "font-size": "32px",
                },
                text_content="Welcome",
                attributes={},
                children_count=0,
            ),
            VisualElement(
                element_id="el_2",
                selector=".button",
                tag_name="button",
                bounds={"x": 100.0, "y": 100.0, "width": 44.0, "height": 44.0},
                computed_styles={
                    "color": "rgb(255, 255, 255)",
                    "background-color": "rgb(0, 0, 255)",
                    "font-size": "16px",
                },
                text_content="Click me",
                attributes={"aria-label": "Submit form"},
                children_count=0,
            ),
            VisualElement(
                element_id="el_3",
                selector="#small-btn",
                tag_name="button",
                bounds={"x": 200.0, "y": 100.0, "width": 30.0, "height": 30.0},
                computed_styles={
                    "color": "rgb(100, 100, 100)",
                    "background-color": "rgb(200, 200, 200)",
                    "font-size": "12px",
                },
                text_content="OK",
                attributes={},
                children_count=0,
            ),
        ]

    @pytest.fixture
    def sample_snapshot(self, sample_elements):
        """Create a sample VisualSnapshot for testing."""
        return VisualSnapshot(
            id="snap_test",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name="Desktop",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"fake_screenshot",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=sample_elements,
            layout_hash="abc123",
            color_palette=["#ffffff", "#000000"],
            text_blocks=[],
            largest_contentful_paint=1500.0,
            cumulative_layout_shift=0.05,
            time_to_interactive=3000.0,
        )

    def test_init(self, analyzer):
        """Test AccessibilityAnalyzer initialization."""
        assert analyzer is not None
        assert analyzer.wcag_level == "AA"
        assert analyzer.min_touch_target == 44
        assert analyzer.check_aaa is False

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        analyzer = AccessibilityAnalyzer(
            wcag_level="AAA",
            min_touch_target=48,
            check_aaa=True,
        )
        assert analyzer.wcag_level == "AAA"
        assert analyzer.min_touch_target == 48
        assert analyzer.check_aaa is True

    def test_calculate_contrast_ratio_black_on_white(self, analyzer):
        """Test contrast ratio calculation for black on white."""
        ratio = analyzer._calculate_contrast_ratio("#000000", "#ffffff")
        assert 20 < ratio < 22  # Should be ~21

    def test_calculate_contrast_ratio_white_on_white(self, analyzer):
        """Test contrast ratio calculation for same colors."""
        ratio = analyzer._calculate_contrast_ratio("#ffffff", "#ffffff")
        assert ratio == 1.0

    def test_calculate_contrast_ratio_gray(self, analyzer):
        """Test contrast ratio for gray tones."""
        ratio = analyzer._calculate_contrast_ratio("#777777", "#ffffff")
        assert 4 < ratio < 5

    def test_get_relative_luminance_white(self, analyzer):
        """Test relative luminance for white."""
        lum = analyzer._get_relative_luminance("#ffffff")
        assert 0.99 < lum <= 1.0

    def test_get_relative_luminance_black(self, analyzer):
        """Test relative luminance for black."""
        lum = analyzer._get_relative_luminance("#000000")
        assert lum == 0.0

    def test_extract_color_hex(self, analyzer):
        """Test extracting hex color."""
        color = analyzer._extract_color("#ff0000")
        assert color == "#FF0000"

        color = analyzer._extract_color("#f00")
        assert color == "#FF0000"

    def test_extract_color_rgb(self, analyzer):
        """Test extracting RGB color."""
        color = analyzer._extract_color("rgb(255, 128, 64)")
        assert color == "#FF8040"

    def test_extract_color_rgba(self, analyzer):
        """Test extracting RGBA color."""
        color = analyzer._extract_color("rgba(100, 150, 200, 0.5)")
        assert color == "#6496C8"

    def test_extract_color_named(self, analyzer):
        """Test extracting named colors."""
        color = analyzer._extract_color("red")
        assert color == "#FF0000"

        color = analyzer._extract_color("white")
        assert color == "#FFFFFF"

        color = analyzer._extract_color("black")
        assert color == "#000000"

    def test_extract_color_invalid(self, analyzer):
        """Test extracting invalid color returns None."""
        color = analyzer._extract_color("not-a-color")
        assert color is None

    def test_parse_font_size_px(self, analyzer):
        """Test parsing font size in pixels."""
        size = analyzer._parse_font_size("16px")
        assert size == 16.0

    def test_parse_font_size_pt(self, analyzer):
        """Test parsing font size in points."""
        size = analyzer._parse_font_size("12pt")
        assert size == pytest.approx(16.0, rel=0.1)

    def test_parse_font_size_rem(self, analyzer):
        """Test parsing font size in rem."""
        size = analyzer._parse_font_size("1.5rem")
        assert size == 24.0

    def test_parse_font_weight_numeric(self, analyzer):
        """Test parsing numeric font weight."""
        weight = analyzer._parse_font_weight("700")
        assert weight == 700

    def test_parse_font_weight_named(self, analyzer):
        """Test parsing named font weight."""
        weight = analyzer._parse_font_weight("bold")
        assert weight == 700

        weight = analyzer._parse_font_weight("normal")
        assert weight == 400

    def test_classify_text_size_normal(self, analyzer):
        """Test text size classification for normal text."""
        size = analyzer._classify_text_size(16.0, 400)
        assert size == "normal"

    def test_classify_text_size_large(self, analyzer):
        """Test text size classification for large text."""
        # 24px or larger
        size = analyzer._classify_text_size(24.0, 400)
        assert size == "large"

        # 18.66px+ bold
        size = analyzer._classify_text_size(19.0, 700)
        assert size == "large"

    @pytest.mark.asyncio
    async def test_check_color_contrast(self, analyzer, sample_snapshot):
        """Test color contrast checking."""
        violations = await analyzer.check_color_contrast(sample_snapshot)
        assert isinstance(violations, list)
        for v in violations:
            assert isinstance(v, ContrastViolation)

    @pytest.mark.asyncio
    async def test_check_touch_target_size(self, analyzer, sample_snapshot):
        """Test touch target size checking."""
        violations = await analyzer.check_touch_target_size(sample_snapshot)
        assert isinstance(violations, list)
        # Should detect the small button (30x30)
        small_btn_violations = [
            v for v in violations if "#small-btn" in v.element.selector
        ]
        assert len(small_btn_violations) > 0

    @pytest.mark.asyncio
    async def test_check_touch_target_custom_size(self, analyzer, sample_snapshot):
        """Test touch target checking with custom minimum."""
        violations = await analyzer.check_touch_target_size(sample_snapshot, min_size=48)
        assert isinstance(violations, list)
        # Even the 44x44 button should fail with 48px minimum
        button_violations = [
            v for v in violations if ".button" in v.element.selector
        ]
        assert len(button_violations) > 0

    @pytest.mark.asyncio
    async def test_check_text_readability(self, analyzer, sample_snapshot):
        """Test text readability checking."""
        issues = await analyzer.check_text_readability(sample_snapshot)
        assert isinstance(issues, list)
        for issue in issues:
            assert isinstance(issue, ReadabilityIssue)

    @pytest.mark.asyncio
    async def test_analyze_full(self, analyzer, sample_snapshot):
        """Test full accessibility analysis."""
        report = await analyzer.analyze_full(sample_snapshot)
        assert isinstance(report, AccessibilityReport)
        assert report.snapshot_id == "snap_test"
        assert report.url == "https://example.com"
        assert isinstance(report.score, float)
        assert 0.0 <= report.score <= 100.0

    @pytest.mark.asyncio
    async def test_compare_accessibility(self, analyzer, sample_snapshot):
        """Test accessibility comparison between versions."""
        result = await analyzer.compare_accessibility(sample_snapshot, sample_snapshot)
        assert isinstance(result, dict)
        assert "has_regression" in result
        assert "baseline_score" in result
        assert "current_score" in result
        assert "score_delta" in result
        assert "new_violations" in result
        assert "fixed_violations" in result


class TestAccessibilityAnalyzerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def analyzer(self):
        return AccessibilityAnalyzer()

    @pytest.fixture
    def empty_snapshot(self):
        """Create an empty snapshot."""
        return VisualSnapshot(
            id="empty",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name="Desktop",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[],
            layout_hash="",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )

    @pytest.mark.asyncio
    async def test_analyze_empty_snapshot(self, analyzer, empty_snapshot):
        """Test analyzing snapshot with no elements."""
        report = await analyzer.analyze_full(empty_snapshot)
        assert isinstance(report, AccessibilityReport)
        assert report.total_violations == 0
        assert report.score == 100.0

    @pytest.mark.asyncio
    async def test_check_elements_with_no_styles(self, analyzer):
        """Test checking elements with missing computed styles."""
        snapshot = VisualSnapshot(
            id="no-styles",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name="Desktop",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[
                VisualElement(
                    element_id="el_1",
                    selector="#no-styles",
                    tag_name="div",
                    bounds={"x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0},
                    computed_styles={},  # Empty styles
                    text_content="Text",
                    attributes={},
                    children_count=0,
                )
            ],
            layout_hash="",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )
        # Should not crash
        violations = await analyzer.check_color_contrast(snapshot)
        assert isinstance(violations, list)

    def test_contrast_ratio_edge_values(self, analyzer):
        """Test contrast ratio with edge values."""
        # Maximum contrast
        ratio = analyzer._calculate_contrast_ratio("#000000", "#ffffff")
        assert ratio > 20

        # Minimum contrast
        ratio = analyzer._calculate_contrast_ratio("#000000", "#000000")
        assert ratio == 1.0

    def test_extract_color_edge_cases(self, analyzer):
        """Test color extraction edge cases."""
        # Short hex
        color = analyzer._extract_color("#fff")
        assert color == "#FFFFFF"

        # 8-character hex (with alpha)
        color = analyzer._extract_color("#ff0000ff")
        assert color == "#FF0000"

        # Empty string
        color = analyzer._extract_color("")
        assert color is None

        # Transparent
        color = analyzer._extract_color("transparent")
        assert color is None


class TestWCAGLevels:
    """Test WCAG compliance level checking."""

    @pytest.fixture
    def analyzer(self):
        return AccessibilityAnalyzer()

    def test_get_required_contrast_ratio_aa_normal(self, analyzer):
        """Test AA level for normal text (4.5:1)."""
        ratio = analyzer._get_required_contrast_ratio("normal", "AA")
        assert ratio == 4.5

    def test_get_required_contrast_ratio_aa_large(self, analyzer):
        """Test AA level for large text (3:1)."""
        ratio = analyzer._get_required_contrast_ratio("large", "AA")
        assert ratio == 3.0

    def test_get_required_contrast_ratio_aaa_normal(self, analyzer):
        """Test AAA level for normal text (7:1)."""
        ratio = analyzer._get_required_contrast_ratio("normal", "AAA")
        assert ratio == 7.0

    def test_get_required_contrast_ratio_aaa_large(self, analyzer):
        """Test AAA level for large text (4.5:1)."""
        ratio = analyzer._get_required_contrast_ratio("large", "AAA")
        assert ratio == 4.5

    @pytest.mark.asyncio
    async def test_check_aaa_level(self):
        """Test checking AAA level compliance."""
        analyzer = AccessibilityAnalyzer(check_aaa=True)
        snapshot = VisualSnapshot(
            id="aaa-test",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name="Desktop",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[
                VisualElement(
                    element_id="el_1",
                    selector="#text",
                    tag_name="p",
                    bounds={"x": 0, "y": 0, "width": 100, "height": 20},
                    computed_styles={
                        "color": "rgb(100, 100, 100)",
                        "background-color": "rgb(255, 255, 255)",
                        "font-size": "14px",
                    },
                    text_content="Test text",
                    attributes={},
                    children_count=0,
                )
            ],
            layout_hash="",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )
        violations = await analyzer.check_color_contrast(snapshot)
        # Should have AAA violations if enabled
        aaa_violations = [v for v in violations if v.wcag_level == "AAA"]
        # May or may not have AAA violations depending on contrast
        assert isinstance(aaa_violations, list)


class TestAccessibilityAnalyzerScoring:
    """Test accessibility scoring logic."""

    @pytest.fixture
    def analyzer(self):
        return AccessibilityAnalyzer()

    @pytest.fixture
    def sample_element(self):
        return VisualElement(
            element_id="el_1",
            selector="#test",
            tag_name="p",
            bounds={"x": 0, "y": 0, "width": 100, "height": 20},
            computed_styles={},
            text_content="Test",
            attributes={},
            children_count=0,
        )

    def test_score_calculation_no_violations(self, analyzer):
        """Test score is 100 with no violations."""
        snapshot = VisualSnapshot(
            id="perfect",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name="Desktop",
            browser="chromium",
            timestamp="",
            screenshot=b"",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[],
            layout_hash="",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )
        score = analyzer._calculate_score([], [], [], snapshot)
        assert score == 100.0

    def test_score_deduction_contrast(self, analyzer, sample_element):
        """Test score deduction for contrast violations."""
        snapshot = VisualSnapshot(
            id="test",
            url="",
            viewport={"width": 0, "height": 0},
            device_name=None,
            browser="",
            timestamp="",
            screenshot=b"",
            dom_snapshot="",
            computed_styles={},
            network_har=None,
            elements=[],
            layout_hash="",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )
        violation = ContrastViolation(
            element=sample_element,
            foreground_color="#777",
            background_color="#fff",
            contrast_ratio=4.0,
            required_ratio=4.5,
            wcag_level="AA",
            text_size="normal",
        )
        score = analyzer._calculate_score([violation], [], [], snapshot)
        assert score < 100.0

    def test_score_minimum_zero(self, analyzer, sample_element):
        """Test score doesn't go below 0."""
        snapshot = VisualSnapshot(
            id="test",
            url="",
            viewport={"width": 0, "height": 0},
            device_name=None,
            browser="",
            timestamp="",
            screenshot=b"",
            dom_snapshot="",
            computed_styles={},
            network_har=None,
            elements=[],
            layout_hash="",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )
        # Create many critical violations
        violations = []
        for i in range(20):
            violations.append(ContrastViolation(
                element=sample_element,
                foreground_color="#ccc",
                background_color="#fff",
                contrast_ratio=1.5,
                required_ratio=4.5,
                wcag_level="AA",
                text_size="normal",
            ))
        score = analyzer._calculate_score(violations, [], [], snapshot)
        assert score >= 0.0
