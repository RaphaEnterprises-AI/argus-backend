"""Comprehensive tests for visual_ai/cross_browser_analyzer.py.

Tests cross-browser visual comparison including multi-browser capture,
difference detection, and compatibility report generation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any
from datetime import datetime

from src.visual_ai.cross_browser_analyzer import (
    CrossBrowserAnalyzer,
    BrowserConfig,
    BrowserDifference,
    BrowserCompatibilityReport,
)
from src.visual_ai.models import VisualSnapshot, VisualElement


class TestBrowserConfig:
    """Tests for BrowserConfig dataclass."""

    def test_browser_config_creation(self):
        """Test creating a BrowserConfig instance."""
        config = BrowserConfig(
            browser="chromium",
            name="Chrome",
            version="120.0",
            channel="chrome",
        )
        assert config.browser == "chromium"
        assert config.name == "Chrome"
        assert config.version == "120.0"
        assert config.channel == "chrome"

    def test_browser_config_without_version(self):
        """Test BrowserConfig without version."""
        config = BrowserConfig(
            browser="firefox",
            name="Firefox",
        )
        assert config.browser == "firefox"
        assert config.version is None
        assert config.channel is None

    def test_browser_config_str(self):
        """Test string representation."""
        config = BrowserConfig(
            browser="webkit",
            name="Safari",
            version="17.0",
        )
        assert str(config) == "Safari 17.0"

    def test_browser_config_str_without_version(self):
        """Test string representation without version."""
        config = BrowserConfig(
            browser="webkit",
            name="Safari",
        )
        assert str(config) == "Safari"


class TestBrowserDifference:
    """Tests for BrowserDifference dataclass."""

    def test_browser_difference_creation(self):
        """Test creating a BrowserDifference instance."""
        diff = BrowserDifference(
            baseline_browser="Chrome",
            comparison_browser="Firefox",
            element_selector=".header",
            difference_type="layout",
            description="Header position shifted by 5px",
            severity=5,
            screenshot_region=None,
            details={"x_shift": 5, "y_shift": 0},
        )
        assert diff.baseline_browser == "Chrome"
        assert diff.comparison_browser == "Firefox"
        assert diff.difference_type == "layout"
        assert diff.severity == 5

    def test_is_critical_true(self):
        """Test is_critical for high severity."""
        diff = BrowserDifference(
            baseline_browser="Chrome",
            comparison_browser="Firefox",
            element_selector=".nav",
            difference_type="rendering",
            description="Navigation completely broken",
            severity=8,
        )
        assert diff.is_critical() is True

    def test_is_critical_false(self):
        """Test is_critical for low severity."""
        diff = BrowserDifference(
            baseline_browser="Chrome",
            comparison_browser="Safari",
            element_selector=".button",
            difference_type="font",
            description="Minor font rendering difference",
            severity=3,
        )
        assert diff.is_critical() is False

    def test_is_minor_true(self):
        """Test is_minor for low severity."""
        diff = BrowserDifference(
            baseline_browser="Chrome",
            comparison_browser="Firefox",
            element_selector=".text",
            difference_type="font",
            description="Antialiasing difference",
            severity=2,
        )
        assert diff.is_minor() is True

    def test_is_minor_false(self):
        """Test is_minor for high severity."""
        diff = BrowserDifference(
            baseline_browser="Chrome",
            comparison_browser="Safari",
            element_selector=".layout",
            difference_type="layout",
            description="Major layout shift",
            severity=7,
        )
        assert diff.is_minor() is False


class TestBrowserCompatibilityReport:
    """Tests for BrowserCompatibilityReport dataclass."""

    @pytest.fixture
    def sample_differences(self):
        """Create sample differences."""
        return [
            BrowserDifference(
                baseline_browser="Chrome",
                comparison_browser="Firefox",
                element_selector=".header",
                difference_type="layout",
                description="Position shift",
                severity=5,
            ),
            BrowserDifference(
                baseline_browser="Chrome",
                comparison_browser="Safari",
                element_selector=".nav",
                difference_type="rendering",
                description="Critical rendering issue",
                severity=8,
            ),
            BrowserDifference(
                baseline_browser="Chrome",
                comparison_browser="Firefox",
                element_selector=".text",
                difference_type="font",
                description="Minor font difference",
                severity=2,
            ),
        ]

    @pytest.fixture
    def sample_report(self, sample_differences):
        """Create a sample compatibility report."""
        return BrowserCompatibilityReport(
            url="https://example.com",
            browsers_tested=["Chrome", "Firefox", "Safari"],
            baseline_browser="Chrome",
            overall_compatibility=75.5,
            differences=sample_differences,
            summary="Cross-browser compatibility: 75.5%",
            snapshots={},
            timestamp="2024-01-01T12:00:00Z",
            duration_ms=5000,
        )

    def test_get_differences_by_type(self, sample_report):
        """Test filtering differences by type."""
        layout_diffs = sample_report.get_differences_by_type("layout")
        assert len(layout_diffs) == 1
        assert layout_diffs[0].difference_type == "layout"

        font_diffs = sample_report.get_differences_by_type("font")
        assert len(font_diffs) == 1

    def test_get_differences_by_browser(self, sample_report):
        """Test filtering differences by browser."""
        firefox_diffs = sample_report.get_differences_by_browser("Firefox")
        assert len(firefox_diffs) == 2

        safari_diffs = sample_report.get_differences_by_browser("Safari")
        assert len(safari_diffs) == 1

    def test_get_critical_differences(self, sample_report):
        """Test getting critical differences."""
        critical = sample_report.get_critical_differences()
        assert len(critical) == 1
        assert critical[0].severity >= 7

    def test_has_critical_issues(self, sample_report):
        """Test has_critical_issues detection."""
        assert sample_report.has_critical_issues() is True

    def test_has_critical_issues_false(self, sample_differences):
        """Test has_critical_issues when no critical issues."""
        # Filter out critical differences
        non_critical = [d for d in sample_differences if d.severity < 7]
        report = BrowserCompatibilityReport(
            url="https://example.com",
            browsers_tested=["Chrome", "Firefox"],
            baseline_browser="Chrome",
            overall_compatibility=90.0,
            differences=non_critical,
            summary="Good compatibility",
            snapshots={},
        )
        assert report.has_critical_issues() is False

    def test_to_dict(self, sample_report):
        """Test to_dict serialization."""
        result = sample_report.to_dict()
        assert result["url"] == "https://example.com"
        assert result["overall_compatibility"] == 75.5
        assert len(result["differences"]) == 3
        assert "timestamp" in result
        assert result["duration_ms"] == 5000


class TestCrossBrowserAnalyzer:
    """Tests for CrossBrowserAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a CrossBrowserAnalyzer with mocked dependencies."""
        with patch("src.visual_ai.cross_browser_analyzer.EnhancedCapture"):
            with patch("src.visual_ai.cross_browser_analyzer.PerceptualAnalyzer"):
                return CrossBrowserAnalyzer(headless=True)

    @pytest.fixture
    def sample_snapshot(self):
        """Create a sample VisualSnapshot."""
        element = VisualElement(
            element_id="el_1",
            selector=".header",
            tag_name="header",
            bounds={"x": 0.0, "y": 0.0, "width": 1920.0, "height": 80.0},
            computed_styles={"display": "flex"},
            text_content="Header",
            attributes={},
            children_count=5,
        )
        return VisualSnapshot(
            id="snap_chrome",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name="Chrome",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"screenshot_data",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[element],
            layout_hash="hash123",
            color_palette=["#fff", "#000"],
            text_blocks=[],
            largest_contentful_paint=1500.0,
            cumulative_layout_shift=0.05,
            time_to_interactive=3000.0,
        )

    def test_init(self, analyzer):
        """Test CrossBrowserAnalyzer initialization."""
        assert analyzer is not None
        assert analyzer.headless is True
        assert analyzer.default_viewport == {"width": 1920, "height": 1080}

    def test_init_custom_viewport(self):
        """Test initialization with custom viewport."""
        with patch("src.visual_ai.cross_browser_analyzer.EnhancedCapture"):
            with patch("src.visual_ai.cross_browser_analyzer.PerceptualAnalyzer"):
                analyzer = CrossBrowserAnalyzer(
                    default_viewport={"width": 1440, "height": 900}
                )
                assert analyzer.default_viewport == {"width": 1440, "height": 900}

    def test_browser_matrix(self, analyzer):
        """Test default browser matrix."""
        assert len(analyzer.BROWSER_MATRIX) == 3
        browsers = [b.browser for b in analyzer.BROWSER_MATRIX]
        assert "chromium" in browsers
        assert "firefox" in browsers
        assert "webkit" in browsers

    def test_known_quirks(self, analyzer):
        """Test known quirks dictionary."""
        assert "font_smoothing" in analyzer.KNOWN_QUIRKS
        assert "scrollbar" in analyzer.KNOWN_QUIRKS

    def test_calculate_severity_high_similarity(self, analyzer):
        """Test severity calculation for high similarity."""
        severity = analyzer._calculate_severity(99.0)
        assert severity == 1

    def test_calculate_severity_medium_similarity(self, analyzer):
        """Test severity calculation for medium similarity."""
        severity = analyzer._calculate_severity(92.0)
        assert severity == 5

    def test_calculate_severity_low_similarity(self, analyzer):
        """Test severity calculation for low similarity."""
        severity = analyzer._calculate_severity(75.0)
        assert severity == 9

    def test_position_severity(self, analyzer):
        """Test position severity calculation."""
        assert analyzer._position_severity(1.0, 1.0) == 1
        assert analyzer._position_severity(3.0, 3.0) == 2
        assert analyzer._position_severity(8.0, 8.0) == 4
        assert analyzer._position_severity(15.0, 15.0) == 6
        assert analyzer._position_severity(30.0, 30.0) == 8

    def test_size_severity(self, analyzer):
        """Test size severity calculation."""
        assert analyzer._size_severity(1.0, 1.0) == 1
        assert analyzer._size_severity(4.0, 4.0) == 3
        assert analyzer._size_severity(8.0, 8.0) == 5
        assert analyzer._size_severity(15.0, 15.0) == 7
        assert analyzer._size_severity(25.0, 25.0) == 9

    def test_compare_layouts(self, analyzer):
        """Test layout comparison between snapshots."""
        element1 = VisualElement(
            element_id="el_1",
            selector=".box",
            tag_name="div",
            bounds={"x": 100.0, "y": 100.0, "width": 200.0, "height": 150.0},
            computed_styles={},
            text_content="Box",
            attributes={},
            children_count=0,
        )
        element2 = VisualElement(
            element_id="el_2",
            selector=".box",
            tag_name="div",
            bounds={"x": 110.0, "y": 105.0, "width": 220.0, "height": 150.0},  # Shifted and resized
            computed_styles={},
            text_content="Box",
            attributes={},
            children_count=0,
        )

        snap1 = VisualSnapshot(
            id="snap1",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name=None,
            browser="chromium",
            timestamp="",
            screenshot=b"",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[element1],
            layout_hash="",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )
        snap2 = VisualSnapshot(
            id="snap2",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name=None,
            browser="firefox",
            timestamp="",
            screenshot=b"",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[element2],
            layout_hash="",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )

        diffs = analyzer._compare_layouts(snap1, snap2)
        assert isinstance(diffs, list)
        # Should detect both position and size differences
        assert len(diffs) >= 1

    def test_calculate_overall_compatibility_no_differences(self, analyzer):
        """Test compatibility calculation with no differences."""
        compatibility = analyzer._calculate_overall_compatibility([], 2)
        assert compatibility == 100.0

    def test_calculate_overall_compatibility_with_differences(self, analyzer):
        """Test compatibility calculation with differences."""
        differences = [
            BrowserDifference(
                baseline_browser="Chrome",
                comparison_browser="Firefox",
                element_selector=".box",
                difference_type="layout",
                description="Test",
                severity=5,
            ),
            BrowserDifference(
                baseline_browser="Chrome",
                comparison_browser="Safari",
                element_selector=".text",
                difference_type="font",
                description="Test",
                severity=3,
            ),
        ]
        compatibility = analyzer._calculate_overall_compatibility(differences, 2)
        assert 0.0 <= compatibility <= 100.0
        assert compatibility < 100.0  # Should be less than perfect

    def test_calculate_overall_compatibility_zero_comparisons(self, analyzer):
        """Test compatibility calculation with zero comparisons."""
        compatibility = analyzer._calculate_overall_compatibility([], 0)
        assert compatibility == 100.0

    def test_generate_summary_no_differences(self, analyzer):
        """Test summary generation with no differences."""
        summary = analyzer._generate_summary([], 100.0, ["Chrome", "Firefox"])
        assert "100%" in summary or "Excellent" in summary

    def test_generate_summary_with_differences(self, analyzer):
        """Test summary generation with differences."""
        differences = [
            BrowserDifference(
                baseline_browser="Chrome",
                comparison_browser="Firefox",
                element_selector=".header",
                difference_type="layout",
                description="Position shift",
                severity=5,
            ),
            BrowserDifference(
                baseline_browser="Chrome",
                comparison_browser="Safari",
                element_selector=".nav",
                difference_type="rendering",
                description="Critical issue",
                severity=8,
            ),
        ]
        summary = analyzer._generate_summary(differences, 75.0, ["Chrome", "Firefox", "Safari"])
        assert "75%" in summary
        assert "critical" in summary.lower()


class TestCrossBrowserAnalyzerAsync:
    """Async tests for CrossBrowserAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        with patch("src.visual_ai.cross_browser_analyzer.EnhancedCapture"):
            with patch("src.visual_ai.cross_browser_analyzer.PerceptualAnalyzer"):
                return CrossBrowserAnalyzer(headless=True)

    @pytest.fixture
    def mock_snapshots(self):
        """Create mock snapshots for multiple browsers."""
        def create_snapshot(browser_name, browser_type):
            return VisualSnapshot(
                id=f"snap_{browser_name.lower()}",
                url="https://example.com",
                viewport={"width": 1920, "height": 1080},
                device_name=browser_name,
                browser=browser_type,
                timestamp="2024-01-01T12:00:00Z",
                screenshot=b"screenshot",
                dom_snapshot="{}",
                computed_styles={},
                network_har=None,
                elements=[],
                layout_hash="hash",
                color_palette=[],
                text_blocks=[],
                largest_contentful_paint=1500.0,
                cumulative_layout_shift=0.05,
                time_to_interactive=3000.0,
            )

        return {
            "Chrome": create_snapshot("Chrome", "chromium"),
            "Firefox": create_snapshot("Firefox", "firefox"),
            "Safari": create_snapshot("Safari", "webkit"),
        }

    @pytest.mark.asyncio
    async def test_detect_browser_differences(self, analyzer, mock_snapshots):
        """Test browser difference detection."""
        # Mock perceptual analyzer methods
        analyzer.perceptual.compute_perceptual_hash = AsyncMock(return_value="hash:hash")
        analyzer.perceptual.compare_hashes = AsyncMock(return_value=0.98)
        analyzer.perceptual.detect_color_changes = AsyncMock(return_value=[])
        analyzer.perceptual.analyze_text_rendering = AsyncMock(
            return_value=Mock(font_changed=False, antialiasing_different=False, affected_regions=[])
        )

        differences = await analyzer.detect_browser_differences(
            mock_snapshots, baseline_browser="chromium"
        )
        assert isinstance(differences, list)

    @pytest.mark.asyncio
    async def test_detect_browser_differences_no_baseline(self, analyzer):
        """Test difference detection when baseline not found."""
        snapshots = {}  # Empty snapshots
        differences = await analyzer.detect_browser_differences(
            snapshots, baseline_browser="chromium"
        )
        assert differences == []

    @pytest.mark.asyncio
    async def test_generate_compatibility_report(self, analyzer, mock_snapshots):
        """Test compatibility report generation."""
        # Mock the capture and detection methods
        analyzer.capture_browser_matrix = AsyncMock(return_value=mock_snapshots)
        analyzer.detect_browser_differences = AsyncMock(return_value=[])

        report = await analyzer.generate_compatibility_report(
            url="https://example.com",
            browsers=[
                BrowserConfig("chromium", "Chrome"),
                BrowserConfig("firefox", "Firefox"),
            ],
        )
        assert isinstance(report, BrowserCompatibilityReport)
        assert report.url == "https://example.com"

    @pytest.mark.asyncio
    async def test_generate_compatibility_report_insufficient_browsers(self, analyzer):
        """Test report generation with insufficient browsers captured."""
        analyzer.capture_browser_matrix = AsyncMock(return_value={"Chrome": Mock()})

        report = await analyzer.generate_compatibility_report(
            url="https://example.com"
        )
        # Should handle gracefully with single browser
        assert isinstance(report, BrowserCompatibilityReport)
        assert report.overall_compatibility == 0.0  # Failed comparison


class TestCrossBrowserAnalyzerWithPlaywright:
    """Tests that interact with mocked Playwright."""

    @pytest.fixture
    def analyzer(self):
        with patch("src.visual_ai.cross_browser_analyzer.EnhancedCapture"):
            with patch("src.visual_ai.cross_browser_analyzer.PerceptualAnalyzer"):
                return CrossBrowserAnalyzer(headless=True)

    @pytest.mark.asyncio
    async def test_capture_browser_matrix(self, analyzer):
        """Test browser matrix capture with mocked playwright."""
        with patch("src.visual_ai.cross_browser_analyzer.async_playwright") as mock_pw:
            # Set up the mock chain
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()

            mock_page.goto = AsyncMock()
            mock_context.new_page = AsyncMock(return_value=mock_page)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_browser.close = AsyncMock()

            # Mock the browser launchers
            mock_chromium = MagicMock()
            mock_chromium.launch = AsyncMock(return_value=mock_browser)

            mock_firefox = MagicMock()
            mock_firefox.launch = AsyncMock(return_value=mock_browser)

            mock_playwright_instance = AsyncMock()
            mock_playwright_instance.chromium = mock_chromium
            mock_playwright_instance.firefox = mock_firefox
            mock_playwright_instance.__aenter__ = AsyncMock(return_value=mock_playwright_instance)
            mock_playwright_instance.__aexit__ = AsyncMock(return_value=None)

            mock_pw.return_value = mock_playwright_instance

            # Mock the capture method
            analyzer.capture.capture_snapshot = AsyncMock(
                return_value=VisualSnapshot(
                    id="test",
                    url="https://example.com",
                    viewport={"width": 1920, "height": 1080},
                    device_name="Test",
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
            )

            browsers = [
                BrowserConfig("chromium", "Chrome"),
                BrowserConfig("firefox", "Firefox"),
            ]

            snapshots = await analyzer.capture_browser_matrix(
                url="https://example.com",
                browsers=browsers,
            )
            assert isinstance(snapshots, dict)

    @pytest.mark.asyncio
    async def test_get_browser_specific_styles(self, analyzer):
        """Test getting computed styles across browsers."""
        with patch("src.visual_ai.cross_browser_analyzer.async_playwright") as mock_pw:
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()

            mock_page.goto = AsyncMock()
            mock_page.evaluate = AsyncMock(return_value={
                "font-family": "Arial",
                "font-size": "16px",
                "color": "rgb(0, 0, 0)",
            })

            mock_context.new_page = AsyncMock(return_value=mock_page)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_browser.close = AsyncMock()

            mock_chromium = MagicMock()
            mock_chromium.launch = AsyncMock(return_value=mock_browser)

            mock_playwright_instance = AsyncMock()
            mock_playwright_instance.chromium = mock_chromium
            mock_playwright_instance.__aenter__ = AsyncMock(return_value=mock_playwright_instance)
            mock_playwright_instance.__aexit__ = AsyncMock(return_value=None)

            mock_pw.return_value = mock_playwright_instance

            styles = await analyzer.get_browser_specific_styles(
                url="https://example.com",
                selector=".header",
                browsers=[BrowserConfig("chromium", "Chrome")],
            )
            assert isinstance(styles, dict)
