"""Cross-browser visual comparison analyzer.

Compares rendering across different browsers (Chromium, Firefox, WebKit/Safari)
to detect browser-specific visual differences including font rendering,
layout discrepancies, and color variations.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import structlog
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from .models import VisualSnapshot, VisualElement
from .capture import EnhancedCapture
from .perceptual_analyzer import PerceptualAnalyzer, ColorChange, TextRenderingDiff

logger = structlog.get_logger()


@dataclass
class BrowserConfig:
    """Configuration for a browser instance.

    Attributes:
        browser: Playwright browser type (chromium, firefox, webkit)
        name: Human-readable browser name (Chrome, Firefox, Safari)
        version: Optional specific browser version
        channel: Optional browser channel (chrome, msedge, etc.)
    """
    browser: str  # chromium, firefox, webkit
    name: str  # Chrome, Firefox, Safari
    version: Optional[str] = None
    channel: Optional[str] = None  # chrome, msedge, chrome-beta, etc.

    def __str__(self) -> str:
        version_str = f" {self.version}" if self.version else ""
        return f"{self.name}{version_str}"


@dataclass
class BrowserDifference:
    """Represents a visual difference detected between browsers.

    Attributes:
        baseline_browser: The browser used as baseline
        comparison_browser: The browser being compared
        element_selector: CSS selector of the affected element (if applicable)
        difference_type: Type of difference (rendering, layout, font, color)
        description: Human-readable description of the difference
        severity: Severity level (1-10, where 10 is most severe)
        screenshot_region: Optional screenshot bytes of the affected region
        details: Additional details about the difference
    """
    baseline_browser: str
    comparison_browser: str
    element_selector: Optional[str]
    difference_type: str  # "rendering", "layout", "font", "color"
    description: str
    severity: int  # 1-10
    screenshot_region: Optional[bytes] = None
    details: Dict = field(default_factory=dict)

    def is_critical(self) -> bool:
        """Check if this is a critical difference (severity >= 7)."""
        return self.severity >= 7

    def is_minor(self) -> bool:
        """Check if this is a minor difference (severity <= 3)."""
        return self.severity <= 3


@dataclass
class BrowserCompatibilityReport:
    """Full cross-browser compatibility report.

    Attributes:
        url: The URL that was tested
        browsers_tested: List of browsers that were tested
        baseline_browser: The browser used as the baseline
        overall_compatibility: Overall compatibility score (0-100%)
        differences: List of all detected differences
        summary: Human-readable summary
        snapshots: Dictionary mapping browser names to their snapshots
        timestamp: When the report was generated
        duration_ms: How long the analysis took
    """
    url: str
    browsers_tested: List[str]
    baseline_browser: str
    overall_compatibility: float  # 0-100%
    differences: List[BrowserDifference]
    summary: str
    snapshots: Dict[str, VisualSnapshot]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    duration_ms: int = 0

    def get_differences_by_type(self, diff_type: str) -> List[BrowserDifference]:
        """Get all differences of a specific type."""
        return [d for d in self.differences if d.difference_type == diff_type]

    def get_differences_by_browser(self, browser: str) -> List[BrowserDifference]:
        """Get all differences for a specific browser comparison."""
        return [d for d in self.differences if d.comparison_browser == browser]

    def get_critical_differences(self) -> List[BrowserDifference]:
        """Get all critical differences."""
        return [d for d in self.differences if d.is_critical()]

    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return len(self.get_critical_differences()) > 0

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            "url": self.url,
            "browsers_tested": self.browsers_tested,
            "baseline_browser": self.baseline_browser,
            "overall_compatibility": self.overall_compatibility,
            "differences": [
                {
                    "baseline_browser": d.baseline_browser,
                    "comparison_browser": d.comparison_browser,
                    "element_selector": d.element_selector,
                    "difference_type": d.difference_type,
                    "description": d.description,
                    "severity": d.severity,
                    "details": d.details,
                }
                for d in self.differences
            ],
            "summary": self.summary,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


class CrossBrowserAnalyzer:
    """Cross-browser visual parity testing.

    Captures screenshots across multiple browsers and analyzes them for
    visual differences including layout discrepancies, font rendering
    differences, and color variations.

    Features:
    - Multi-browser capture (Chromium, Firefox, WebKit)
    - Perceptual hash comparison for similarity
    - Font rendering difference detection
    - Color variation analysis
    - Layout shift detection
    - Element-level comparison

    Usage:
        analyzer = CrossBrowserAnalyzer()

        # Generate full compatibility report
        report = await analyzer.generate_compatibility_report(
            "https://example.com",
            browsers=[
                BrowserConfig("chromium", "Chrome"),
                BrowserConfig("firefox", "Firefox"),
                BrowserConfig("webkit", "Safari"),
            ]
        )

        print(f"Compatibility: {report.overall_compatibility}%")
        for diff in report.differences:
            print(f"- {diff.description} (severity: {diff.severity})")
    """

    BROWSER_MATRIX = [
        BrowserConfig("chromium", "Chrome"),
        BrowserConfig("firefox", "Firefox"),
        BrowserConfig("webkit", "Safari"),
    ]

    # Known browser-specific quirks to filter out
    KNOWN_QUIRKS = {
        "font_smoothing": {
            "webkit": "Safari uses different font smoothing",
            "firefox": "Firefox has different subpixel rendering",
        },
        "scrollbar": {
            "webkit": "Safari has overlay scrollbars by default",
            "firefox": "Firefox scrollbar width may differ",
        },
    }

    def __init__(
        self,
        capture: Optional[EnhancedCapture] = None,
        perceptual: Optional[PerceptualAnalyzer] = None,
        default_viewport: Optional[Dict[str, int]] = None,
        headless: bool = True,
    ):
        """Initialize CrossBrowserAnalyzer.

        Args:
            capture: EnhancedCapture instance (created if not provided)
            perceptual: PerceptualAnalyzer instance (created if not provided)
            default_viewport: Default viewport size
            headless: Whether to run browsers in headless mode
        """
        self.capture = capture or EnhancedCapture()
        self.perceptual = perceptual or PerceptualAnalyzer()
        self.default_viewport = default_viewport or {"width": 1920, "height": 1080}
        self.headless = headless
        self.log = logger.bind(component="cross_browser_analyzer")

    async def capture_browser_matrix(
        self,
        url: str,
        browsers: Optional[List[BrowserConfig]] = None,
        viewport: Optional[Dict[str, int]] = None,
        wait_for_idle: bool = True,
        timeout: int = 30000,
    ) -> Dict[str, VisualSnapshot]:
        """Capture the same URL in multiple browsers.

        Args:
            url: URL to capture
            browsers: List of browser configurations (defaults to BROWSER_MATRIX)
            viewport: Viewport size (defaults to default_viewport)
            wait_for_idle: Whether to wait for network idle
            timeout: Page load timeout in milliseconds

        Returns:
            Dictionary mapping browser names to their snapshots
        """
        browsers = browsers or self.BROWSER_MATRIX
        viewport = viewport or self.default_viewport
        snapshots: Dict[str, VisualSnapshot] = {}

        self.log.info(
            "Capturing browser matrix",
            url=url,
            browsers=[b.name for b in browsers],
        )

        async with async_playwright() as p:
            for browser_config in browsers:
                try:
                    snapshot = await self._capture_single_browser(
                        p, browser_config, url, viewport, wait_for_idle, timeout
                    )
                    snapshots[browser_config.name] = snapshot
                    self.log.info(
                        "Captured browser snapshot",
                        browser=browser_config.name,
                        snapshot_id=snapshot.id,
                    )
                except Exception as e:
                    self.log.error(
                        "Failed to capture browser",
                        browser=browser_config.name,
                        error=str(e),
                    )

        return snapshots

    async def _capture_single_browser(
        self,
        playwright,
        browser_config: BrowserConfig,
        url: str,
        viewport: Dict[str, int],
        wait_for_idle: bool,
        timeout: int,
    ) -> VisualSnapshot:
        """Capture screenshot from a single browser."""
        browser_launcher = getattr(playwright, browser_config.browser)

        launch_options = {"headless": self.headless}
        if browser_config.channel:
            launch_options["channel"] = browser_config.channel

        browser = await browser_launcher.launch(**launch_options)

        try:
            context = await browser.new_context(
                viewport=viewport,
                device_scale_factor=1,
            )

            page = await context.new_page()

            wait_until = "networkidle" if wait_for_idle else "domcontentloaded"
            await page.goto(url, wait_until=wait_until, timeout=timeout)

            # Small delay for any final rendering
            await asyncio.sleep(0.5)

            snapshot = await self.capture.capture_snapshot(
                page=page,
                url=url,
                viewport=viewport,
                browser=browser_config.browser,
                device_name=browser_config.name,
            )

            return snapshot

        finally:
            await browser.close()

    async def detect_browser_differences(
        self,
        snapshots: Dict[str, VisualSnapshot],
        baseline_browser: str = "chromium",
        similarity_threshold: float = 95.0,
    ) -> List[BrowserDifference]:
        """Find visual differences between browsers.

        Args:
            snapshots: Dictionary mapping browser names to snapshots
            baseline_browser: Browser to use as baseline
            similarity_threshold: Threshold below which to report differences

        Returns:
            List of detected differences
        """
        differences: List[BrowserDifference] = []

        # Find baseline snapshot
        baseline_snapshot = None
        baseline_name = None
        for name, snapshot in snapshots.items():
            if baseline_browser.lower() in name.lower() or baseline_browser.lower() in snapshot.browser.lower():
                baseline_snapshot = snapshot
                baseline_name = name
                break

        if not baseline_snapshot:
            self.log.error("Baseline browser not found", baseline=baseline_browser)
            return differences

        self.log.info(
            "Detecting browser differences",
            baseline=baseline_name,
            comparing=[name for name in snapshots.keys() if name != baseline_name],
        )

        # Compare each browser against baseline
        for name, snapshot in snapshots.items():
            if name == baseline_name:
                continue

            browser_diffs = await self._compare_browser_pair(
                baseline_snapshot,
                snapshot,
                baseline_name,
                name,
                similarity_threshold,
            )
            differences.extend(browser_diffs)

        return differences

    async def _compare_browser_pair(
        self,
        baseline: VisualSnapshot,
        comparison: VisualSnapshot,
        baseline_name: str,
        comparison_name: str,
        similarity_threshold: float,
    ) -> List[BrowserDifference]:
        """Compare two browser snapshots and return differences."""
        differences: List[BrowserDifference] = []

        # 1. Perceptual hash comparison for overall similarity
        hash_similarity = await self.perceptual.compare_hashes(
            await self.perceptual.compute_perceptual_hash(baseline.screenshot),
            await self.perceptual.compute_perceptual_hash(comparison.screenshot),
        )

        if hash_similarity < similarity_threshold / 100:
            differences.append(BrowserDifference(
                baseline_browser=baseline_name,
                comparison_browser=comparison_name,
                element_selector=None,
                difference_type="rendering",
                description=f"Overall rendering differs significantly ({hash_similarity*100:.1f}% similarity)",
                severity=self._calculate_severity(hash_similarity * 100),
                details={"similarity": hash_similarity * 100},
            ))

        # 2. Color change detection
        color_changes = await self.perceptual.detect_color_changes(
            baseline.screenshot,
            comparison.screenshot,
        )

        for color_change in color_changes:
            if color_change.delta_e > 5:  # Noticeable color difference
                severity = min(10, int(color_change.delta_e / 5))
                differences.append(BrowserDifference(
                    baseline_browser=baseline_name,
                    comparison_browser=comparison_name,
                    element_selector=None,
                    difference_type="color",
                    description=f"Color difference: {color_change.old_color} -> {color_change.new_color} (Delta E: {color_change.delta_e:.1f})",
                    severity=severity,
                    details={
                        "old_color": color_change.old_color,
                        "new_color": color_change.new_color,
                        "delta_e": color_change.delta_e,
                        "affected_area": color_change.affected_area_percent,
                    },
                ))

        # 3. Text rendering analysis
        text_diff = await self.perceptual.analyze_text_rendering(
            baseline.screenshot,
            comparison.screenshot,
        )

        if text_diff.font_changed:
            differences.append(BrowserDifference(
                baseline_browser=baseline_name,
                comparison_browser=comparison_name,
                element_selector=None,
                difference_type="font",
                description="Font rendering differs between browsers",
                severity=5,
                details={"affected_regions": len(text_diff.affected_regions)},
            ))

        if text_diff.antialiasing_different:
            differences.append(BrowserDifference(
                baseline_browser=baseline_name,
                comparison_browser=comparison_name,
                element_selector=None,
                difference_type="font",
                description="Font antialiasing differs (expected browser variation)",
                severity=2,  # Low severity - this is common
                details={"is_known_quirk": True},
            ))

        # 4. Layout comparison
        layout_diffs = self._compare_layouts(baseline, comparison)
        for layout_diff in layout_diffs:
            differences.append(BrowserDifference(
                baseline_browser=baseline_name,
                comparison_browser=comparison_name,
                element_selector=layout_diff.get("selector"),
                difference_type="layout",
                description=layout_diff.get("description", "Layout difference detected"),
                severity=layout_diff.get("severity", 5),
                details=layout_diff,
            ))

        return differences

    def _compare_layouts(
        self,
        baseline: VisualSnapshot,
        comparison: VisualSnapshot,
    ) -> List[Dict]:
        """Compare layouts between two snapshots."""
        diffs = []

        # Create lookup for baseline elements by selector
        baseline_elements = {el.selector: el for el in baseline.elements}

        for comp_el in comparison.elements:
            if comp_el.selector in baseline_elements:
                base_el = baseline_elements[comp_el.selector]

                # Check for position shifts
                x_diff = abs(comp_el.bounds["x"] - base_el.bounds["x"])
                y_diff = abs(comp_el.bounds["y"] - base_el.bounds["y"])

                if x_diff > 5 or y_diff > 5:
                    diffs.append({
                        "selector": comp_el.selector,
                        "description": f"Element position shifted by ({x_diff:.0f}px, {y_diff:.0f}px)",
                        "severity": self._position_severity(x_diff, y_diff),
                        "baseline_position": {"x": base_el.bounds["x"], "y": base_el.bounds["y"]},
                        "comparison_position": {"x": comp_el.bounds["x"], "y": comp_el.bounds["y"]},
                    })

                # Check for size differences
                w_diff_pct = abs(comp_el.bounds["width"] - base_el.bounds["width"]) / max(base_el.bounds["width"], 1) * 100
                h_diff_pct = abs(comp_el.bounds["height"] - base_el.bounds["height"]) / max(base_el.bounds["height"], 1) * 100

                if w_diff_pct > 5 or h_diff_pct > 5:
                    diffs.append({
                        "selector": comp_el.selector,
                        "description": f"Element size differs by {max(w_diff_pct, h_diff_pct):.1f}%",
                        "severity": self._size_severity(w_diff_pct, h_diff_pct),
                        "baseline_size": {"width": base_el.bounds["width"], "height": base_el.bounds["height"]},
                        "comparison_size": {"width": comp_el.bounds["width"], "height": comp_el.bounds["height"]},
                    })

        return diffs[:20]  # Limit to top 20 layout differences

    def _calculate_severity(self, similarity: float) -> int:
        """Calculate severity based on similarity percentage."""
        if similarity >= 98:
            return 1
        elif similarity >= 95:
            return 3
        elif similarity >= 90:
            return 5
        elif similarity >= 80:
            return 7
        else:
            return 9

    def _position_severity(self, x_diff: float, y_diff: float) -> int:
        """Calculate severity for position differences."""
        max_diff = max(x_diff, y_diff)
        if max_diff <= 2:
            return 1
        elif max_diff <= 5:
            return 2
        elif max_diff <= 10:
            return 4
        elif max_diff <= 20:
            return 6
        else:
            return 8

    def _size_severity(self, w_pct: float, h_pct: float) -> int:
        """Calculate severity for size differences."""
        max_pct = max(w_pct, h_pct)
        if max_pct <= 2:
            return 1
        elif max_pct <= 5:
            return 3
        elif max_pct <= 10:
            return 5
        elif max_pct <= 20:
            return 7
        else:
            return 9

    async def generate_compatibility_report(
        self,
        url: str,
        browsers: Optional[List[BrowserConfig]] = None,
        viewport: Optional[Dict[str, int]] = None,
        baseline_browser: str = "chromium",
    ) -> BrowserCompatibilityReport:
        """Generate a full cross-browser compatibility report.

        Args:
            url: URL to test
            browsers: List of browser configurations
            viewport: Viewport size
            baseline_browser: Browser to use as baseline

        Returns:
            Complete BrowserCompatibilityReport
        """
        start_time = datetime.utcnow()
        browsers = browsers or self.BROWSER_MATRIX

        self.log.info(
            "Generating compatibility report",
            url=url,
            browsers=[b.name for b in browsers],
            baseline=baseline_browser,
        )

        # Capture all browsers
        snapshots = await self.capture_browser_matrix(
            url=url,
            browsers=browsers,
            viewport=viewport,
        )

        if len(snapshots) < 2:
            return BrowserCompatibilityReport(
                url=url,
                browsers_tested=list(snapshots.keys()),
                baseline_browser=baseline_browser,
                overall_compatibility=0.0,
                differences=[],
                summary="Failed to capture multiple browsers for comparison",
                snapshots=snapshots,
            )

        # Detect differences
        differences = await self.detect_browser_differences(
            snapshots=snapshots,
            baseline_browser=baseline_browser,
        )

        # Calculate overall compatibility
        overall_compatibility = self._calculate_overall_compatibility(differences, len(browsers) - 1)

        # Generate summary
        summary = self._generate_summary(differences, overall_compatibility, list(snapshots.keys()))

        # Calculate duration
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        report = BrowserCompatibilityReport(
            url=url,
            browsers_tested=list(snapshots.keys()),
            baseline_browser=baseline_browser,
            overall_compatibility=overall_compatibility,
            differences=differences,
            summary=summary,
            snapshots=snapshots,
            duration_ms=duration_ms,
        )

        self.log.info(
            "Compatibility report generated",
            url=url,
            compatibility=f"{overall_compatibility:.1f}%",
            differences=len(differences),
            critical=len(report.get_critical_differences()),
        )

        return report

    def _calculate_overall_compatibility(
        self,
        differences: List[BrowserDifference],
        num_comparisons: int,
    ) -> float:
        """Calculate overall compatibility score."""
        if not differences:
            return 100.0

        if num_comparisons == 0:
            return 100.0

        # Weight differences by severity
        total_penalty = 0
        for diff in differences:
            # Critical issues (7-10) have higher penalty
            if diff.severity >= 7:
                total_penalty += diff.severity * 2
            elif diff.severity >= 4:
                total_penalty += diff.severity
            else:
                total_penalty += diff.severity * 0.5

        # Normalize by number of comparisons
        avg_penalty = total_penalty / num_comparisons

        # Convert to percentage (cap penalty at 100)
        compatibility = max(0, 100 - min(avg_penalty * 5, 100))

        return round(compatibility, 1)

    def _generate_summary(
        self,
        differences: List[BrowserDifference],
        compatibility: float,
        browsers: List[str],
    ) -> str:
        """Generate human-readable summary."""
        if not differences:
            return f"Excellent cross-browser compatibility ({compatibility:.0f}%). No significant differences detected across {', '.join(browsers)}."

        critical = [d for d in differences if d.is_critical()]
        layout = [d for d in differences if d.difference_type == "layout"]
        font = [d for d in differences if d.difference_type == "font"]
        color = [d for d in differences if d.difference_type == "color"]

        parts = [f"Cross-browser compatibility: {compatibility:.0f}%."]

        if critical:
            parts.append(f"{len(critical)} critical issue(s) detected.")

        if layout:
            parts.append(f"{len(layout)} layout difference(s).")

        if font:
            parts.append(f"{len(font)} font rendering difference(s).")

        if color:
            parts.append(f"{len(color)} color variation(s).")

        parts.append(f"Tested browsers: {', '.join(browsers)}.")

        return " ".join(parts)

    async def compare_specific_element(
        self,
        snapshots: Dict[str, VisualSnapshot],
        selector: str,
    ) -> Dict[str, bytes]:
        """Compare a specific element's rendering across browsers.

        Args:
            snapshots: Dictionary mapping browser names to snapshots
            selector: CSS selector for the element to compare

        Returns:
            Dictionary mapping browser names to element screenshot bytes
        """
        element_screenshots: Dict[str, bytes] = {}

        self.log.info("Comparing specific element", selector=selector)

        async with async_playwright() as p:
            for browser_name, snapshot in snapshots.items():
                try:
                    # Determine browser type from name
                    browser_type = "chromium"
                    if "firefox" in browser_name.lower():
                        browser_type = "firefox"
                    elif "safari" in browser_name.lower() or "webkit" in browser_name.lower():
                        browser_type = "webkit"

                    browser_launcher = getattr(p, browser_type)
                    browser = await browser_launcher.launch(headless=self.headless)

                    try:
                        context = await browser.new_context(
                            viewport=snapshot.viewport,
                        )
                        page = await context.new_page()
                        await page.goto(snapshot.url, wait_until="networkidle")

                        element = await page.query_selector(selector)
                        if element:
                            screenshot = await element.screenshot()
                            element_screenshots[browser_name] = screenshot

                    finally:
                        await browser.close()

                except Exception as e:
                    self.log.warning(
                        "Failed to capture element",
                        browser=browser_name,
                        selector=selector,
                        error=str(e),
                    )

        return element_screenshots

    async def get_browser_specific_styles(
        self,
        url: str,
        selector: str,
        browsers: Optional[List[BrowserConfig]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Get computed styles for an element across browsers.

        Useful for debugging browser-specific CSS issues.

        Args:
            url: URL to load
            selector: CSS selector for the element
            browsers: List of browser configurations

        Returns:
            Dictionary mapping browser names to computed styles
        """
        browsers = browsers or self.BROWSER_MATRIX
        styles: Dict[str, Dict[str, str]] = {}

        async with async_playwright() as p:
            for browser_config in browsers:
                try:
                    browser_launcher = getattr(p, browser_config.browser)
                    browser = await browser_launcher.launch(headless=self.headless)

                    try:
                        context = await browser.new_context()
                        page = await context.new_page()
                        await page.goto(url, wait_until="networkidle")

                        element_styles = await page.evaluate(
                            """(selector) => {
                                const el = document.querySelector(selector);
                                if (!el) return null;

                                const style = window.getComputedStyle(el);
                                const properties = [
                                    'font-family', 'font-size', 'font-weight',
                                    'line-height', 'letter-spacing', 'color',
                                    'background-color', 'padding', 'margin',
                                    'border', 'border-radius', 'box-shadow',
                                    'width', 'height', 'display', 'position',
                                    '-webkit-font-smoothing', '-moz-osx-font-smoothing',
                                    'text-rendering'
                                ];

                                const result = {};
                                for (const prop of properties) {
                                    result[prop] = style.getPropertyValue(prop);
                                }
                                return result;
                            }""",
                            selector,
                        )

                        if element_styles:
                            styles[browser_config.name] = element_styles

                    finally:
                        await browser.close()

                except Exception as e:
                    self.log.warning(
                        "Failed to get styles",
                        browser=browser_config.name,
                        error=str(e),
                    )

        return styles
