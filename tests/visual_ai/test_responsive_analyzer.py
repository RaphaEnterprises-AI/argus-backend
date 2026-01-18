"""Comprehensive tests for visual_ai/responsive_analyzer.py.

Tests responsive design analysis including viewport configurations,
breakpoint detection, layout comparisons, and responsive issues.
"""

from unittest.mock import patch

import pytest

from src.visual_ai.models import VisualElement, VisualSnapshot
from src.visual_ai.responsive_analyzer import (
    BreakpointIssue,
    ResponsiveAnalyzer,
    ResponsiveDiff,
    ViewportConfig,
)


class TestViewportConfig:
    """Tests for ViewportConfig dataclass."""

    def test_viewport_config_creation(self):
        """Test creating a ViewportConfig instance."""
        config = ViewportConfig(
            name="Desktop",
            width=1920,
            height=1080,
            device_scale_factor=1.0,
            is_mobile=False,
            has_touch=False,
        )
        assert config.name == "Desktop"
        assert config.width == 1920
        assert config.height == 1080
        assert config.is_mobile is False

    def test_viewport_config_mobile(self):
        """Test ViewportConfig for mobile device."""
        config = ViewportConfig(
            name="iPhone 12",
            width=390,
            height=844,
            device_scale_factor=3.0,
            is_mobile=True,
            has_touch=True,
        )
        assert config.is_mobile is True
        assert config.has_touch is True
        assert config.device_scale_factor == 3.0

    def test_viewport_config_tablet(self):
        """Test ViewportConfig for tablet device."""
        config = ViewportConfig(
            name="iPad",
            width=768,
            height=1024,
            device_scale_factor=2.0,
            is_mobile=False,
            has_touch=True,
        )
        assert config.width == 768
        assert config.has_touch is True

    def test_viewport_config_default_values(self):
        """Test ViewportConfig with default values."""
        config = ViewportConfig(name="Test", width=800, height=600)
        assert config.device_scale_factor == 1.0
        assert config.is_mobile is False
        assert config.user_agent is None
        assert config.has_touch is False

    def test_to_playwright_config(self):
        """Test conversion to Playwright configuration."""
        config = ViewportConfig(
            name="Mobile",
            width=375,
            height=667,
            device_scale_factor=2.0,
            is_mobile=True,
            has_touch=True,
            user_agent="Mozilla/5.0 (iPhone...)",
        )
        pw_config = config.to_playwright_config()
        assert pw_config["viewport"]["width"] == 375
        assert pw_config["viewport"]["height"] == 667
        assert pw_config["device_scale_factor"] == 2.0
        assert pw_config["is_mobile"] is True
        assert pw_config["has_touch"] is True
        assert "user_agent" in pw_config

    def test_to_playwright_config_auto_touch(self):
        """Test that has_touch is auto-enabled for mobile."""
        config = ViewportConfig(
            name="Mobile",
            width=375,
            height=667,
            is_mobile=True,
            has_touch=False,  # Not explicitly set
        )
        pw_config = config.to_playwright_config()
        # Should be True because is_mobile is True
        assert pw_config["has_touch"] is True

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = ViewportConfig(
            name="Desktop",
            width=1920,
            height=1080,
            device_scale_factor=1.0,
            is_mobile=False,
        )
        result = config.to_dict()
        assert result["name"] == "Desktop"
        assert result["width"] == 1920
        assert result["height"] == 1080


class TestBreakpointIssue:
    """Tests for BreakpointIssue dataclass."""

    def test_breakpoint_issue_creation(self):
        """Test creating a BreakpointIssue instance."""
        issue = BreakpointIssue(
            viewport="mobile",
            element_selector=".sidebar",
            issue_type="overflow",
            description="Sidebar overflows viewport by 50px",
            severity=3,
        )
        assert issue.viewport == "mobile"
        assert issue.element_selector == ".sidebar"
        assert issue.issue_type == "overflow"
        assert issue.severity == 3

    def test_breakpoint_issue_overflow(self):
        """Test BreakpointIssue for horizontal overflow."""
        issue = BreakpointIssue(
            viewport="mobile",
            element_selector=".wide-table",
            issue_type="overflow",
            description="Table causes horizontal scroll",
            severity=4,
            overflow_amount=200.0,
        )
        assert issue.issue_type == "overflow"
        assert issue.severity == 4
        assert issue.overflow_amount == 200.0

    def test_breakpoint_issue_truncation(self):
        """Test BreakpointIssue for text truncation."""
        issue = BreakpointIssue(
            viewport="mobile",
            element_selector=".long-title",
            issue_type="truncation",
            description="Title text is truncated unexpectedly",
            severity=2,
        )
        assert issue.issue_type == "truncation"
        assert issue.severity == 2

    def test_breakpoint_issue_overlap(self):
        """Test BreakpointIssue for element overlap."""
        issue = BreakpointIssue(
            viewport="tablet",
            element_selector=".button-a",
            issue_type="overlap",
            description="Button overlaps with adjacent element",
            severity=4,
            related_element=".button-b",
            overlap_area=100.0,
        )
        assert issue.issue_type == "overlap"
        assert issue.related_element == ".button-b"
        assert issue.overlap_area == 100.0

    def test_breakpoint_issue_touch_target(self):
        """Test BreakpointIssue for touch target issues."""
        issue = BreakpointIssue(
            viewport="mobile",
            element_selector=".small-link",
            issue_type="touch_target",
            description="Touch target is too small (30x30px, minimum: 44px)",
            severity=3,
            element_bounds={"x": 10, "y": 20, "width": 30, "height": 30},
        )
        assert issue.issue_type == "touch_target"
        assert issue.element_bounds["width"] == 30

    def test_breakpoint_issue_to_dict(self):
        """Test BreakpointIssue serialization."""
        issue = BreakpointIssue(
            viewport="mobile",
            element_selector=".element",
            issue_type="overflow",
            description="Element overflows",
            severity=3,
            recommendation="Fix the overflow",
        )
        result = issue.to_dict()
        assert result["viewport"] == "mobile"
        assert result["element_selector"] == ".element"
        assert result["issue_type"] == "overflow"
        assert result["severity"] == 3
        assert result["recommendation"] == "Fix the overflow"


class TestResponsiveDiff:
    """Tests for ResponsiveDiff dataclass."""

    @pytest.fixture
    def sample_snapshots(self):
        """Create sample baseline and current snapshots."""
        baseline = VisualSnapshot(
            id="baseline_snap",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name="Desktop",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"baseline_screenshot",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[],
            layout_hash="baseline_hash",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=1500.0,
            cumulative_layout_shift=0.05,
            time_to_interactive=3000.0,
        )
        current = VisualSnapshot(
            id="current_snap",
            url="https://example.com",
            viewport={"width": 375, "height": 812},
            device_name="Mobile",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"current_screenshot",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[],
            layout_hash="current_hash",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=2000.0,
            cumulative_layout_shift=0.1,
            time_to_interactive=4000.0,
        )
        return baseline, current

    @pytest.fixture
    def sample_diff(self, sample_snapshots):
        """Create a sample ResponsiveDiff."""
        baseline, current = sample_snapshots
        return ResponsiveDiff(
            viewport="mobile",
            baseline_snapshot=baseline,
            current_snapshot=current,
            issues=[
                BreakpointIssue(
                    viewport="mobile",
                    element_selector=".wide-table",
                    issue_type="overflow",
                    description="Table overflows viewport",
                    severity=4,
                )
            ],
            match_percentage=75.0,
            layout_changed=True,
            elements_added=["new-element"],
            elements_removed=["old-element"],
            elements_moved=[{"selector": ".header", "delta": {"x": 0, "y": 10}}],
        )

    def test_responsive_diff_creation(self, sample_snapshots):
        """Test creating a ResponsiveDiff instance."""
        baseline, current = sample_snapshots
        diff = ResponsiveDiff(
            viewport="tablet",
            baseline_snapshot=baseline,
            current_snapshot=current,
            issues=[],
            match_percentage=98.5,
        )
        assert diff.viewport == "tablet"
        assert diff.match_percentage == 98.5
        assert diff.layout_changed is False  # default

    def test_has_blocking_issues_true(self, sample_diff):
        """Test has_blocking_issues returns True for severe issues."""
        assert sample_diff.has_blocking_issues(min_severity=3) is True

    def test_has_blocking_issues_false(self, sample_snapshots):
        """Test has_blocking_issues returns False when no severe issues."""
        baseline, current = sample_snapshots
        diff = ResponsiveDiff(
            viewport="tablet",
            baseline_snapshot=baseline,
            current_snapshot=current,
            issues=[
                BreakpointIssue(
                    viewport="tablet",
                    element_selector=".element",
                    issue_type="truncation",
                    description="Minor truncation",
                    severity=2,
                )
            ],
            match_percentage=95.0,
        )
        assert diff.has_blocking_issues(min_severity=3) is False

    def test_to_dict(self, sample_diff):
        """Test to_dict serialization."""
        result = sample_diff.to_dict()
        assert result["viewport"] == "mobile"
        assert result["baseline_snapshot_id"] == "baseline_snap"
        assert result["current_snapshot_id"] == "current_snap"
        assert len(result["issues"]) == 1
        assert result["match_percentage"] == 75.0
        assert result["layout_changed"] is True
        assert "new-element" in result["elements_added"]
        assert "old-element" in result["elements_removed"]

    def test_responsive_diff_default_values(self, sample_snapshots):
        """Test ResponsiveDiff default values."""
        baseline, current = sample_snapshots
        diff = ResponsiveDiff(
            viewport="desktop",
            baseline_snapshot=baseline,
            current_snapshot=current,
            issues=[],
            match_percentage=100.0,
        )
        assert diff.layout_changed is False
        assert diff.elements_added == []
        assert diff.elements_removed == []
        assert diff.elements_moved == []


class TestResponsiveAnalyzer:
    """Tests for ResponsiveAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a ResponsiveAnalyzer instance with mocked capture."""
        with patch("src.visual_ai.responsive_analyzer.EnhancedCapture"):
            return ResponsiveAnalyzer()

    @pytest.fixture
    def desktop_elements(self):
        """Create elements for desktop viewport."""
        return [
            VisualElement(
                element_id="header",
                selector=".header",
                tag_name="header",
                bounds={"x": 0.0, "y": 0.0, "width": 1920.0, "height": 80.0},
                computed_styles={"display": "flex"},
                text_content="Header",
                attributes={},
                children_count=5,
            ),
            VisualElement(
                element_id="sidebar",
                selector=".sidebar",
                tag_name="aside",
                bounds={"x": 0.0, "y": 80.0, "width": 300.0, "height": 920.0},
                computed_styles={"display": "block"},
                text_content="Sidebar",
                attributes={},
                children_count=10,
            ),
            VisualElement(
                element_id="button1",
                selector=".button",
                tag_name="button",
                bounds={"x": 100.0, "y": 100.0, "width": 100.0, "height": 40.0},
                computed_styles={"display": "inline-block"},
                text_content="Click Me",
                attributes={},
                children_count=0,
            ),
        ]

    @pytest.fixture
    def mobile_elements(self):
        """Create elements for mobile viewport with overflow."""
        return [
            VisualElement(
                element_id="header",
                selector=".header",
                tag_name="header",
                bounds={"x": 0.0, "y": 0.0, "width": 375.0, "height": 60.0},
                computed_styles={"display": "flex"},
                text_content="Header",
                attributes={},
                children_count=3,
            ),
            VisualElement(
                element_id="wide_element",
                selector=".wide",
                tag_name="div",
                bounds={"x": 0.0, "y": 60.0, "width": 500.0, "height": 100.0},
                computed_styles={"display": "block"},
                text_content="Wide content",
                attributes={},
                children_count=1,
            ),
            VisualElement(
                element_id="small_button",
                selector=".small-btn",
                tag_name="button",
                bounds={"x": 10.0, "y": 200.0, "width": 30.0, "height": 30.0},
                computed_styles={"display": "inline-block"},
                text_content="X",
                attributes={},
                children_count=0,
            ),
        ]

    @pytest.fixture
    def desktop_snapshot(self, desktop_elements):
        """Create a desktop viewport snapshot."""
        return VisualSnapshot(
            id="desktop_snap",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name="Desktop",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"desktop_screenshot",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=desktop_elements,
            layout_hash="desktop_hash",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=1500.0,
            cumulative_layout_shift=0.05,
            time_to_interactive=3000.0,
        )

    @pytest.fixture
    def mobile_snapshot(self, mobile_elements):
        """Create a mobile viewport snapshot."""
        return VisualSnapshot(
            id="mobile_snap",
            url="https://example.com",
            viewport={"width": 375, "height": 812},
            device_name="mobile",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"mobile_screenshot",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=mobile_elements,
            layout_hash="mobile_hash",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=2000.0,
            cumulative_layout_shift=0.1,
            time_to_interactive=4000.0,
        )

    def test_init(self, analyzer):
        """Test ResponsiveAnalyzer initialization."""
        assert analyzer is not None
        assert analyzer.min_touch_target_size == 44
        assert analyzer.overflow_threshold_px == 5
        assert analyzer.overlap_threshold_px == 2

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        with patch("src.visual_ai.responsive_analyzer.EnhancedCapture"):
            analyzer = ResponsiveAnalyzer(
                min_touch_target_size=48,
                overflow_threshold_px=10,
                overlap_threshold_px=5,
            )
        assert analyzer.min_touch_target_size == 48
        assert analyzer.overflow_threshold_px == 10
        assert analyzer.overlap_threshold_px == 5

    def test_standard_viewports(self, analyzer):
        """Test that standard viewports are defined."""
        assert len(analyzer.STANDARD_VIEWPORTS) > 0
        names = [v.name for v in analyzer.STANDARD_VIEWPORTS]
        assert "mobile" in names
        assert "tablet" in names
        assert "desktop" in names

    def test_extended_viewports(self, analyzer):
        """Test that extended viewports are defined."""
        assert len(analyzer.EXTENDED_VIEWPORTS) > 0
        names = [v.name for v in analyzer.EXTENDED_VIEWPORTS]
        assert "mobile_small" in names or "mobile_large" in names

    def test_get_viewport_config(self, analyzer):
        """Test getting viewport config by name."""
        config = analyzer._get_viewport_config("mobile")
        assert config is not None
        assert config.name == "mobile"
        assert config.is_mobile is True

    def test_get_viewport_config_not_found(self, analyzer):
        """Test getting non-existent viewport config."""
        config = analyzer._get_viewport_config("nonexistent")
        assert config is None

    def test_calculate_overlap_area(self, analyzer):
        """Test overlap area calculation."""
        bounds1 = {"x": 0, "y": 0, "width": 100, "height": 100}
        bounds2 = {"x": 50, "y": 50, "width": 100, "height": 100}
        overlap = analyzer._calculate_overlap_area(bounds1, bounds2)
        assert overlap == 2500  # 50x50 overlap

    def test_calculate_overlap_area_no_overlap(self, analyzer):
        """Test overlap calculation with no overlap."""
        bounds1 = {"x": 0, "y": 0, "width": 100, "height": 100}
        bounds2 = {"x": 200, "y": 200, "width": 100, "height": 100}
        overlap = analyzer._calculate_overlap_area(bounds1, bounds2)
        assert overlap == 0

    @pytest.mark.asyncio
    async def test_detect_overflow_issues(self, analyzer, mobile_snapshot):
        """Test detection of overflow issues."""
        # Wide element (500px) > viewport (375px)
        viewport_config = ViewportConfig(
            name="mobile", width=375, height=812, is_mobile=True
        )
        issues = await analyzer._detect_overflow_issues(mobile_snapshot, viewport_config)
        assert len(issues) > 0
        overflow_issues = [i for i in issues if i.issue_type == "overflow"]
        assert len(overflow_issues) > 0

    @pytest.mark.asyncio
    async def test_detect_touch_target_issues(self, analyzer, mobile_snapshot):
        """Test detection of touch target issues."""
        issues = await analyzer._detect_touch_target_issues(mobile_snapshot)
        assert isinstance(issues, list)
        # Small button (30x30) < minimum (44)
        touch_issues = [i for i in issues if i.issue_type == "touch_target"]
        assert len(touch_issues) > 0

    @pytest.mark.asyncio
    async def test_detect_overlap_issues(self, analyzer, desktop_snapshot):
        """Test detection of overlap issues."""
        issues = await analyzer._detect_overlap_issues(desktop_snapshot)
        assert isinstance(issues, list)

    @pytest.mark.asyncio
    async def test_detect_truncation_issues(self, analyzer):
        """Test detection of truncation issues."""
        snapshot = VisualSnapshot(
            id="test",
            url="https://example.com",
            viewport={"width": 375, "height": 812},
            device_name="mobile",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"screenshot",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[],
            layout_hash="hash",
            color_palette=[],
            text_blocks=[
                {"text": "A" * 100, "bounds": {"x": 0, "y": 0, "width": 50, "height": 20}}
            ],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )
        issues = await analyzer._detect_truncation_issues(snapshot)
        assert isinstance(issues, list)

    @pytest.mark.asyncio
    async def test_detect_breakpoint_issues(self, analyzer, mobile_snapshot):
        """Test breakpoint issue detection on a snapshot."""
        snapshots = {"mobile": mobile_snapshot}
        issues = await analyzer.detect_breakpoint_issues(snapshots)
        assert isinstance(issues, list)
        for issue in issues:
            assert isinstance(issue, BreakpointIssue)

    @pytest.mark.asyncio
    async def test_compare_responsive_regression(
        self, analyzer, desktop_snapshot, mobile_snapshot
    ):
        """Test responsive regression comparison."""
        baseline_snapshots = {"mobile": desktop_snapshot}
        current_snapshots = {"mobile": mobile_snapshot}

        diffs = await analyzer.compare_responsive_regression(
            baseline_snapshots, current_snapshots
        )
        assert isinstance(diffs, list)
        assert len(diffs) == 1
        assert isinstance(diffs[0], ResponsiveDiff)

    @pytest.mark.asyncio
    async def test_compare_layouts(self, analyzer, desktop_snapshot, mobile_snapshot):
        """Test layout comparison between snapshots."""
        layout_changed, added, removed, moved = analyzer._compare_layouts(
            desktop_snapshot, mobile_snapshot
        )
        assert isinstance(layout_changed, bool)
        assert isinstance(added, list)
        assert isinstance(removed, list)
        assert isinstance(moved, list)

    @pytest.mark.asyncio
    async def test_check_element_visibility(self, analyzer, desktop_snapshot):
        """Test element visibility checking."""
        snapshots = {"desktop": desktop_snapshot}
        visibility = await analyzer.check_element_visibility(snapshots, ".header")
        assert isinstance(visibility, dict)
        assert "desktop" in visibility

    @pytest.mark.asyncio
    async def test_detect_text_overflow(self, analyzer, mobile_snapshot):
        """Test text overflow detection."""
        overflows = await analyzer.detect_text_overflow(mobile_snapshot)
        assert isinstance(overflows, list)

    @pytest.mark.asyncio
    async def test_detect_element_overlap(self, analyzer, desktop_snapshot):
        """Test element overlap detection."""
        overlaps = await analyzer.detect_element_overlap(desktop_snapshot)
        assert isinstance(overlaps, list)

    @pytest.mark.asyncio
    async def test_generate_responsive_report(self, analyzer, mobile_snapshot):
        """Test responsive report generation."""
        snapshots = {"mobile": mobile_snapshot}
        issues = [
            BreakpointIssue(
                viewport="mobile",
                element_selector=".wide",
                issue_type="overflow",
                description="Element overflows",
                severity=4,
            )
        ]
        report = await analyzer.generate_responsive_report(snapshots, issues)
        assert "id" in report
        assert "overall_status" in report
        assert "summary" in report
        assert "viewports" in report
        assert "recommendations" in report


class TestResponsiveAnalyzerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def analyzer(self):
        with patch("src.visual_ai.responsive_analyzer.EnhancedCapture"):
            return ResponsiveAnalyzer()

    @pytest.mark.asyncio
    async def test_detect_breakpoint_issues_empty_snapshots(self, analyzer):
        """Test detecting issues with empty snapshots dict."""
        issues = await analyzer.detect_breakpoint_issues({})
        assert issues == []

    @pytest.mark.asyncio
    async def test_compare_regression_no_common_viewports(self, analyzer):
        """Test regression comparison with no common viewports."""
        snap1 = VisualSnapshot(
            id="snap1",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name="desktop",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"screenshot",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[],
            layout_hash="hash1",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )
        snap2 = VisualSnapshot(
            id="snap2",
            url="https://example.com",
            viewport={"width": 375, "height": 812},
            device_name="mobile",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"screenshot",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[],
            layout_hash="hash2",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )
        baseline = {"desktop": snap1}
        current = {"mobile": snap2}
        diffs = await analyzer.compare_responsive_regression(baseline, current)
        assert diffs == []  # No common viewports

    @pytest.mark.asyncio
    async def test_detect_overflow_issues_no_overflow(self, analyzer):
        """Test overflow detection with no overflows."""
        snapshot = VisualSnapshot(
            id="test",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name="desktop",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"screenshot",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[
                VisualElement(
                    element_id="el1",
                    selector=".box",
                    tag_name="div",
                    bounds={"x": 0, "y": 0, "width": 100, "height": 100},
                    computed_styles={},
                    text_content="Box",
                    attributes={},
                    children_count=0,
                )
            ],
            layout_hash="hash",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )
        viewport_config = ViewportConfig(
            name="desktop", width=1920, height=1080, is_mobile=False
        )
        issues = await analyzer._detect_overflow_issues(snapshot, viewport_config)
        assert len(issues) == 0

    @pytest.mark.asyncio
    async def test_generate_report_no_issues(self, analyzer):
        """Test report generation with no issues."""
        snapshot = VisualSnapshot(
            id="test",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name="desktop",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"screenshot",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[],
            layout_hash="hash",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )
        report = await analyzer.generate_responsive_report({"desktop": snapshot}, [])
        assert report["overall_status"] == "PASSED"
        assert report["summary"]["total_issues"] == 0


class TestResponsiveAnalyzerRecommendations:
    """Test recommendation generation."""

    @pytest.fixture
    def analyzer(self):
        with patch("src.visual_ai.responsive_analyzer.EnhancedCapture"):
            return ResponsiveAnalyzer()

    def test_generate_recommendations_overflow(self, analyzer):
        """Test recommendations for overflow issues."""
        issues = [
            BreakpointIssue(
                viewport="mobile",
                element_selector=".table",
                issue_type="overflow",
                description="Table overflows",
                severity=4,
                recommendation="Add overflow handling",
            )
        ]
        recommendations = analyzer._generate_recommendations(issues)
        assert len(recommendations) > 0
        assert any("overflow" in r.lower() for r in recommendations)

    def test_generate_recommendations_touch_target(self, analyzer):
        """Test recommendations for touch target issues."""
        issues = [
            BreakpointIssue(
                viewport="mobile",
                element_selector=".button",
                issue_type="touch_target",
                description="Button too small",
                severity=3,
                recommendation="Increase touch target size",
            )
        ]
        recommendations = analyzer._generate_recommendations(issues)
        assert len(recommendations) > 0
        assert any("touch" in r.lower() or "44" in r for r in recommendations)

    def test_generate_recommendations_overlap(self, analyzer):
        """Test recommendations for overlap issues."""
        issues = [
            BreakpointIssue(
                viewport="tablet",
                element_selector=".el1",
                issue_type="overlap",
                description="Elements overlap",
                severity=4,
                recommendation="Review positioning",
            )
        ]
        recommendations = analyzer._generate_recommendations(issues)
        assert len(recommendations) > 0
        assert any("overlap" in r.lower() or "position" in r.lower() for r in recommendations)

    def test_generate_recommendations_empty(self, analyzer):
        """Test recommendations with no issues."""
        recommendations = analyzer._generate_recommendations([])
        assert isinstance(recommendations, list)
