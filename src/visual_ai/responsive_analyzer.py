"""Responsive design analyzer for cross-viewport visual testing.

This module provides tools to test responsive behavior across multiple
viewport sizes, detecting layout breaks, element visibility issues,
and responsive regressions.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from playwright.async_api import Page, Browser, BrowserContext

from .models import VisualSnapshot, VisualElement
from .capture import EnhancedCapture

logger = structlog.get_logger()


@dataclass
class ViewportConfig:
    """Configuration for a viewport size to test.

    Represents a specific device or viewport configuration including
    dimensions, scale factor, and mobile-specific settings.

    Example:
        mobile = ViewportConfig(
            name="mobile",
            width=375,
            height=667,
            device_scale_factor=2.0,
            is_mobile=True,
            user_agent="Mozilla/5.0 (iPhone; ..."
        )
    """

    name: str
    width: int
    height: int
    device_scale_factor: float = 1.0
    is_mobile: bool = False
    user_agent: Optional[str] = None
    has_touch: bool = False

    def to_playwright_config(self) -> Dict[str, Any]:
        """Convert to Playwright viewport/context configuration."""
        config: Dict[str, Any] = {
            "viewport": {"width": self.width, "height": self.height},
            "device_scale_factor": self.device_scale_factor,
            "is_mobile": self.is_mobile,
            "has_touch": self.has_touch or self.is_mobile,
        }
        if self.user_agent:
            config["user_agent"] = self.user_agent
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "device_scale_factor": self.device_scale_factor,
            "is_mobile": self.is_mobile,
            "user_agent": self.user_agent,
            "has_touch": self.has_touch,
        }


@dataclass
class BreakpointIssue:
    """Detected issue at a specific viewport size.

    Represents a responsive design problem such as element overflow,
    overlap, truncation, or visibility issues at a particular viewport.

    Example:
        issue = BreakpointIssue(
            viewport="mobile",
            element_selector=".sidebar",
            issue_type="overflow",
            description="Sidebar overflows viewport by 50px",
            severity=3
        )
    """

    viewport: str
    element_selector: str
    issue_type: str  # "overflow", "overlap", "truncation", "hidden", "wrap_error", "touch_target"
    description: str
    severity: int  # 1-5, where 5 is most severe

    # Additional context
    element_bounds: Optional[Dict[str, float]] = None
    related_element: Optional[str] = None
    overflow_amount: Optional[float] = None
    overlap_area: Optional[float] = None
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "viewport": self.viewport,
            "element_selector": self.element_selector,
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity,
            "element_bounds": self.element_bounds,
            "related_element": self.related_element,
            "overflow_amount": self.overflow_amount,
            "overlap_area": self.overlap_area,
            "recommendation": self.recommendation,
        }


@dataclass
class ResponsiveDiff:
    """Comparison result between baseline and current responsive snapshots.

    Tracks visual differences at a specific viewport, including match
    percentage and detected issues.

    Example:
        diff = ResponsiveDiff(
            viewport="tablet",
            baseline_snapshot=baseline,
            current_snapshot=current,
            issues=[...],
            match_percentage=95.5
        )
    """

    viewport: str
    baseline_snapshot: VisualSnapshot
    current_snapshot: VisualSnapshot
    issues: List[BreakpointIssue]
    match_percentage: float

    # Layout comparison
    layout_changed: bool = False
    elements_added: List[str] = field(default_factory=list)
    elements_removed: List[str] = field(default_factory=list)
    elements_moved: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "viewport": self.viewport,
            "baseline_snapshot_id": self.baseline_snapshot.id,
            "current_snapshot_id": self.current_snapshot.id,
            "issues": [issue.to_dict() for issue in self.issues],
            "match_percentage": self.match_percentage,
            "layout_changed": self.layout_changed,
            "elements_added": self.elements_added,
            "elements_removed": self.elements_removed,
            "elements_moved": self.elements_moved,
        }

    def has_blocking_issues(self, min_severity: int = 3) -> bool:
        """Check if there are issues at or above the given severity."""
        return any(issue.severity >= min_severity for issue in self.issues)


class ResponsiveAnalyzer:
    """Cross-viewport visual analysis for responsive design testing.

    Captures screenshots at multiple viewport sizes and analyzes for
    responsive design issues including overflow, overlap, visibility,
    and layout breaks.

    Usage:
        analyzer = ResponsiveAnalyzer()

        # Capture all viewports
        snapshots = await analyzer.capture_all_viewports(browser, url)

        # Detect issues
        issues = await analyzer.detect_breakpoint_issues(snapshots)

        # Compare with baseline
        diffs = await analyzer.compare_responsive_regression(
            baseline_snapshots, current_snapshots
        )
    """

    # Standard viewport configurations for common device sizes
    STANDARD_VIEWPORTS = [
        ViewportConfig(
            name="mobile",
            width=375,
            height=667,
            device_scale_factor=2.0,
            is_mobile=True,
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
            ),
        ),
        ViewportConfig(
            name="tablet",
            width=768,
            height=1024,
            device_scale_factor=2.0,
            is_mobile=True,
            user_agent=(
                "Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
            ),
        ),
        ViewportConfig(
            name="desktop",
            width=1440,
            height=900,
            device_scale_factor=1.0,
            is_mobile=False,
        ),
        ViewportConfig(
            name="wide",
            width=1920,
            height=1080,
            device_scale_factor=1.0,
            is_mobile=False,
        ),
    ]

    # Additional breakpoints for comprehensive testing
    EXTENDED_VIEWPORTS = [
        ViewportConfig(name="mobile_small", width=320, height=568, is_mobile=True),
        ViewportConfig(name="mobile_large", width=414, height=896, is_mobile=True),
        ViewportConfig(name="tablet_landscape", width=1024, height=768, is_mobile=True),
        ViewportConfig(name="laptop", width=1366, height=768, is_mobile=False),
        ViewportConfig(name="desktop_large", width=2560, height=1440, is_mobile=False),
    ]

    def __init__(
        self,
        capture: Optional[EnhancedCapture] = None,
        min_touch_target_size: int = 44,
        overflow_threshold_px: int = 5,
        overlap_threshold_px: int = 2,
    ):
        """Initialize the responsive analyzer.

        Args:
            capture: EnhancedCapture instance (created if not provided)
            min_touch_target_size: Minimum touch target size in pixels (iOS HIG: 44px)
            overflow_threshold_px: Minimum overflow to flag as issue
            overlap_threshold_px: Minimum overlap to flag as issue
        """
        self.capture = capture or EnhancedCapture()
        self.min_touch_target_size = min_touch_target_size
        self.overflow_threshold_px = overflow_threshold_px
        self.overlap_threshold_px = overlap_threshold_px
        self.log = logger.bind(component="responsive_analyzer")

    async def capture_all_viewports(
        self,
        browser: Browser,
        url: str,
        viewports: Optional[List[ViewportConfig]] = None,
        full_page: bool = False,
        wait_for_selector: Optional[str] = None,
        wait_timeout_ms: int = 30000,
    ) -> Dict[str, VisualSnapshot]:
        """Capture screenshots at all viewport sizes.

        Creates a new browser context for each viewport to ensure clean
        state and proper viewport simulation.

        Args:
            browser: Playwright browser instance
            url: URL to capture
            viewports: List of ViewportConfig (uses STANDARD_VIEWPORTS if None)
            full_page: Whether to capture full scrollable page
            wait_for_selector: Optional selector to wait for before capture
            wait_timeout_ms: Timeout for wait_for_selector

        Returns:
            Dict mapping viewport name to VisualSnapshot
        """
        if viewports is None:
            viewports = self.STANDARD_VIEWPORTS

        snapshots: Dict[str, VisualSnapshot] = {}

        self.log.info(
            "Capturing viewports",
            url=url,
            viewport_count=len(viewports),
            viewports=[v.name for v in viewports],
        )

        for viewport in viewports:
            try:
                snapshot = await self._capture_viewport(
                    browser=browser,
                    url=url,
                    viewport=viewport,
                    full_page=full_page,
                    wait_for_selector=wait_for_selector,
                    wait_timeout_ms=wait_timeout_ms,
                )
                snapshots[viewport.name] = snapshot

                self.log.debug(
                    "Viewport captured",
                    viewport=viewport.name,
                    elements=len(snapshot.elements),
                )

            except Exception as e:
                self.log.error(
                    "Viewport capture failed",
                    viewport=viewport.name,
                    error=str(e),
                )
                # Continue with other viewports

        return snapshots

    async def _capture_viewport(
        self,
        browser: Browser,
        url: str,
        viewport: ViewportConfig,
        full_page: bool,
        wait_for_selector: Optional[str],
        wait_timeout_ms: int,
    ) -> VisualSnapshot:
        """Capture a single viewport with a fresh context."""
        context: Optional[BrowserContext] = None
        try:
            # Create new context with viewport settings
            context = await browser.new_context(**viewport.to_playwright_config())
            page = await context.new_page()

            # Navigate to URL
            await page.goto(url, wait_until="networkidle", timeout=wait_timeout_ms)

            # Wait for specific selector if provided
            if wait_for_selector:
                await page.wait_for_selector(
                    wait_for_selector,
                    timeout=wait_timeout_ms,
                )

            # Small delay for any animations to settle
            await asyncio.sleep(0.5)

            # Capture snapshot
            snapshot = await self.capture.capture_snapshot(
                page=page,
                full_page=full_page,
                device_name=viewport.name,
            )

            return snapshot

        finally:
            if context:
                await context.close()

    async def detect_breakpoint_issues(
        self,
        snapshots: Dict[str, VisualSnapshot],
    ) -> List[BreakpointIssue]:
        """Find layout breaks between viewport sizes.

        Analyzes snapshots from different viewports to detect:
        - Horizontal overflow (elements wider than viewport)
        - Element overlaps (unintended overlapping)
        - Touch target size issues (mobile)
        - Text truncation
        - Hidden elements

        Args:
            snapshots: Dict mapping viewport name to VisualSnapshot

        Returns:
            List of detected BreakpointIssue objects
        """
        issues: List[BreakpointIssue] = []

        for viewport_name, snapshot in snapshots.items():
            viewport_config = self._get_viewport_config(viewport_name)

            # Check for overflow issues
            overflow_issues = await self._detect_overflow_issues(
                snapshot, viewport_config
            )
            issues.extend(overflow_issues)

            # Check for element overlap
            overlap_issues = await self._detect_overlap_issues(snapshot)
            issues.extend(overlap_issues)

            # Check touch targets on mobile viewports
            if viewport_config and viewport_config.is_mobile:
                touch_issues = await self._detect_touch_target_issues(snapshot)
                issues.extend(touch_issues)

            # Check for text truncation
            truncation_issues = await self._detect_truncation_issues(snapshot)
            issues.extend(truncation_issues)

        self.log.info(
            "Breakpoint analysis complete",
            total_issues=len(issues),
            by_viewport={
                vp: len([i for i in issues if i.viewport == vp])
                for vp in snapshots.keys()
            },
        )

        return issues

    async def _detect_overflow_issues(
        self,
        snapshot: VisualSnapshot,
        viewport_config: Optional[ViewportConfig],
    ) -> List[BreakpointIssue]:
        """Detect horizontal overflow issues."""
        issues = []
        viewport_width = snapshot.viewport.get("width", 1920)
        viewport_name = viewport_config.name if viewport_config else "unknown"

        for element in snapshot.elements:
            bounds = element.bounds
            right_edge = bounds["x"] + bounds["width"]

            # Check if element extends beyond viewport
            overflow = right_edge - viewport_width
            if overflow > self.overflow_threshold_px:
                issues.append(
                    BreakpointIssue(
                        viewport=viewport_name,
                        element_selector=element.selector,
                        issue_type="overflow",
                        description=(
                            f"Element '{element.selector}' overflows viewport by {overflow:.0f}px "
                            f"(element width: {bounds['width']:.0f}px, ends at {right_edge:.0f}px, "
                            f"viewport: {viewport_width}px)"
                        ),
                        severity=4 if overflow > 100 else 3,
                        element_bounds=bounds,
                        overflow_amount=overflow,
                        recommendation=(
                            f"Add overflow handling (overflow: hidden/scroll) or adjust width "
                            f"for {element.selector} at {viewport_width}px viewport"
                        ),
                    )
                )

        return issues

    async def _detect_overlap_issues(
        self,
        snapshot: VisualSnapshot,
    ) -> List[BreakpointIssue]:
        """Detect unintended element overlaps."""
        issues = []
        viewport_name = snapshot.device_name or "unknown"

        # Get interactive elements that shouldn't overlap
        interactive_tags = {"button", "a", "input", "select", "textarea"}
        interactive_elements = [
            el for el in snapshot.elements
            if el.tag_name in interactive_tags
        ]

        # Check for overlapping pairs
        checked_pairs: set = set()
        for i, el1 in enumerate(interactive_elements):
            for el2 in interactive_elements[i + 1:]:
                pair_key = tuple(sorted([el1.element_id, el2.element_id]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                overlap_area = self._calculate_overlap_area(el1.bounds, el2.bounds)
                if overlap_area > self.overlap_threshold_px ** 2:
                    issues.append(
                        BreakpointIssue(
                            viewport=viewport_name,
                            element_selector=el1.selector,
                            issue_type="overlap",
                            description=(
                                f"Elements '{el1.selector}' and '{el2.selector}' overlap "
                                f"by {overlap_area:.0f}px^2"
                            ),
                            severity=4,
                            element_bounds=el1.bounds,
                            related_element=el2.selector,
                            overlap_area=overlap_area,
                            recommendation=(
                                f"Adjust positioning or spacing to prevent overlap between "
                                f"'{el1.selector}' and '{el2.selector}'"
                            ),
                        )
                    )

        return issues

    async def _detect_touch_target_issues(
        self,
        snapshot: VisualSnapshot,
    ) -> List[BreakpointIssue]:
        """Detect touch targets that are too small for mobile."""
        issues = []
        viewport_name = snapshot.device_name or "unknown"

        # Interactive elements that need adequate touch targets
        touch_target_tags = {"button", "a", "input", "select"}

        for element in snapshot.elements:
            if element.tag_name not in touch_target_tags:
                continue

            width = element.bounds["width"]
            height = element.bounds["height"]
            min_dimension = min(width, height)

            if min_dimension < self.min_touch_target_size:
                issues.append(
                    BreakpointIssue(
                        viewport=viewport_name,
                        element_selector=element.selector,
                        issue_type="touch_target",
                        description=(
                            f"Touch target '{element.selector}' is too small "
                            f"({width:.0f}x{height:.0f}px, minimum: {self.min_touch_target_size}px)"
                        ),
                        severity=3,
                        element_bounds=element.bounds,
                        recommendation=(
                            f"Increase touch target size to at least {self.min_touch_target_size}x"
                            f"{self.min_touch_target_size}px using padding or min-width/min-height"
                        ),
                    )
                )

        return issues

    async def _detect_truncation_issues(
        self,
        snapshot: VisualSnapshot,
    ) -> List[BreakpointIssue]:
        """Detect potential text truncation issues."""
        issues = []
        viewport_name = snapshot.device_name or "unknown"

        for text_block in snapshot.text_blocks:
            bounds = text_block.get("bounds", {})
            text = text_block.get("text", "")

            # Heuristic: very narrow containers with long text might truncate
            if bounds.get("width", 0) < 100 and len(text) > 50:
                issues.append(
                    BreakpointIssue(
                        viewport=viewport_name,
                        element_selector="text_block",
                        issue_type="truncation",
                        description=(
                            f"Potential text truncation: container width {bounds.get('width', 0):.0f}px "
                            f"with {len(text)} characters"
                        ),
                        severity=2,
                        element_bounds=bounds,
                        recommendation=(
                            "Ensure text has adequate space or proper text-overflow handling"
                        ),
                    )
                )

        return issues

    def _calculate_overlap_area(
        self,
        bounds1: Dict[str, float],
        bounds2: Dict[str, float],
    ) -> float:
        """Calculate the overlapping area between two elements."""
        x1_left, y1_top = bounds1["x"], bounds1["y"]
        x1_right = x1_left + bounds1["width"]
        y1_bottom = y1_top + bounds1["height"]

        x2_left, y2_top = bounds2["x"], bounds2["y"]
        x2_right = x2_left + bounds2["width"]
        y2_bottom = y2_top + bounds2["height"]

        # Calculate intersection
        x_overlap = max(0, min(x1_right, x2_right) - max(x1_left, x2_left))
        y_overlap = max(0, min(y1_bottom, y2_bottom) - max(y1_top, y2_top))

        return x_overlap * y_overlap

    def _get_viewport_config(self, name: str) -> Optional[ViewportConfig]:
        """Get ViewportConfig by name from standard viewports."""
        for vp in self.STANDARD_VIEWPORTS + self.EXTENDED_VIEWPORTS:
            if vp.name == name:
                return vp
        return None

    async def compare_responsive_regression(
        self,
        baseline_snapshots: Dict[str, VisualSnapshot],
        current_snapshots: Dict[str, VisualSnapshot],
        pixel_threshold: float = 0.05,
    ) -> List[ResponsiveDiff]:
        """Compare responsive behavior changes between baseline and current.

        Analyzes visual differences at each viewport and detects:
        - Layout structure changes
        - Element position/size changes
        - Added/removed elements
        - Visual regressions

        Args:
            baseline_snapshots: Baseline snapshots by viewport name
            current_snapshots: Current snapshots by viewport name
            pixel_threshold: Maximum pixel difference ratio to consider a match

        Returns:
            List of ResponsiveDiff objects for each viewport
        """
        diffs: List[ResponsiveDiff] = []

        # Compare each viewport present in both sets
        common_viewports = set(baseline_snapshots.keys()) & set(current_snapshots.keys())

        for viewport_name in common_viewports:
            baseline = baseline_snapshots[viewport_name]
            current = current_snapshots[viewport_name]

            # Calculate visual match percentage
            match_percentage = await self._calculate_match_percentage(baseline, current)

            # Detect layout changes
            layout_changed, added, removed, moved = self._compare_layouts(baseline, current)

            # Detect issues at this viewport
            issues = await self._detect_regression_issues(
                viewport_name, baseline, current, match_percentage
            )

            diff = ResponsiveDiff(
                viewport=viewport_name,
                baseline_snapshot=baseline,
                current_snapshot=current,
                issues=issues,
                match_percentage=match_percentage,
                layout_changed=layout_changed,
                elements_added=added,
                elements_removed=removed,
                elements_moved=moved,
            )
            diffs.append(diff)

        self.log.info(
            "Responsive regression comparison complete",
            viewports_compared=len(diffs),
            viewports_with_issues=[d.viewport for d in diffs if d.issues],
        )

        return diffs

    async def _calculate_match_percentage(
        self,
        baseline: VisualSnapshot,
        current: VisualSnapshot,
    ) -> float:
        """Calculate visual similarity percentage between snapshots."""
        try:
            import numpy as np
            from PIL import Image
            import io

            # Load images
            img1 = Image.open(io.BytesIO(baseline.screenshot)).convert("RGB")
            img2 = Image.open(io.BytesIO(current.screenshot)).convert("RGB")

            # Resize to same dimensions if needed
            if img1.size != img2.size:
                img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

            # Calculate pixel difference
            arr1 = np.array(img1, dtype=np.float32)
            arr2 = np.array(img2, dtype=np.float32)

            diff = np.abs(arr1 - arr2)
            max_diff = 255.0 * 3  # RGB channels
            similarity = 1.0 - (np.mean(diff) / max_diff)

            return round(similarity * 100, 2)

        except Exception as e:
            self.log.warning("Match calculation failed", error=str(e))
            # Fall back to layout hash comparison
            if baseline.layout_hash == current.layout_hash:
                return 100.0
            return 0.0

    def _compare_layouts(
        self,
        baseline: VisualSnapshot,
        current: VisualSnapshot,
    ) -> tuple[bool, List[str], List[str], List[Dict[str, Any]]]:
        """Compare layout structures between snapshots."""
        baseline_selectors = {el.selector for el in baseline.elements}
        current_selectors = {el.selector for el in current.elements}

        added = list(current_selectors - baseline_selectors)
        removed = list(baseline_selectors - current_selectors)

        # Check for moved elements (same selector, different position)
        moved = []
        common_selectors = baseline_selectors & current_selectors

        baseline_map = {el.selector: el for el in baseline.elements}
        current_map = {el.selector: el for el in current.elements}

        for selector in common_selectors:
            base_el = baseline_map[selector]
            curr_el = current_map[selector]

            x_diff = abs(base_el.bounds["x"] - curr_el.bounds["x"])
            y_diff = abs(base_el.bounds["y"] - curr_el.bounds["y"])

            if x_diff > 10 or y_diff > 10:  # Movement threshold
                moved.append({
                    "selector": selector,
                    "from": {"x": base_el.bounds["x"], "y": base_el.bounds["y"]},
                    "to": {"x": curr_el.bounds["x"], "y": curr_el.bounds["y"]},
                    "delta": {"x": x_diff, "y": y_diff},
                })

        layout_changed = len(added) > 0 or len(removed) > 0 or len(moved) > 0

        return layout_changed, added, removed, moved

    async def _detect_regression_issues(
        self,
        viewport_name: str,
        baseline: VisualSnapshot,
        current: VisualSnapshot,
        match_percentage: float,
    ) -> List[BreakpointIssue]:
        """Detect issues specific to regression comparison."""
        issues = []

        # Check for significant visual change
        if match_percentage < 95:
            issues.append(
                BreakpointIssue(
                    viewport=viewport_name,
                    element_selector="*",
                    issue_type="visual_regression",
                    description=(
                        f"Visual match is {match_percentage:.1f}% "
                        f"(threshold: 95%)"
                    ),
                    severity=4 if match_percentage < 80 else 3,
                    recommendation="Review visual changes and update baseline if intentional",
                )
            )

        # Check for layout hash change
        if baseline.layout_hash != current.layout_hash:
            issues.append(
                BreakpointIssue(
                    viewport=viewport_name,
                    element_selector="*",
                    issue_type="layout_change",
                    description="Layout structure has changed from baseline",
                    severity=2,
                    recommendation="Verify layout changes are intentional",
                )
            )

        return issues

    async def check_element_visibility(
        self,
        snapshots: Dict[str, VisualSnapshot],
        element_selector: str,
    ) -> Dict[str, bool]:
        """Check if element is visible at each viewport.

        Useful for verifying responsive show/hide behavior works correctly.

        Args:
            snapshots: Dict mapping viewport name to VisualSnapshot
            element_selector: CSS selector for the element to check

        Returns:
            Dict mapping viewport name to visibility boolean
        """
        visibility: Dict[str, bool] = {}

        for viewport_name, snapshot in snapshots.items():
            # Check if element exists in snapshot
            element = snapshot.get_element_by_selector(element_selector)

            if element is None:
                visibility[viewport_name] = False
            else:
                # Check if element has visible dimensions
                bounds = element.bounds
                has_size = bounds["width"] > 0 and bounds["height"] > 0

                # Check computed styles for visibility
                styles = snapshot.computed_styles.get(element.element_id, {})
                display_visible = styles.get("display") != "none"
                visibility_visible = styles.get("visibility") != "hidden"
                opacity_visible = styles.get("opacity", "1") != "0"

                visibility[viewport_name] = (
                    has_size and display_visible and visibility_visible and opacity_visible
                )

        return visibility

    async def detect_text_overflow(
        self,
        snapshot: VisualSnapshot,
    ) -> List[Dict[str, Any]]:
        """Detect text that overflows its container.

        Analyzes text blocks in the snapshot to identify overflow issues.

        Args:
            snapshot: VisualSnapshot to analyze

        Returns:
            List of overflow info dicts
        """
        overflows = []
        viewport_width = snapshot.viewport.get("width", 1920)

        for text_block in snapshot.text_blocks:
            bounds = text_block.get("bounds", {})
            text = text_block.get("text", "")

            # Check if text extends beyond viewport
            right_edge = bounds.get("x", 0) + bounds.get("width", 0)
            if right_edge > viewport_width:
                overflows.append({
                    "type": "viewport_overflow",
                    "text": text[:100],
                    "bounds": bounds,
                    "overflow_amount": right_edge - viewport_width,
                })

            # Check for suspiciously narrow containers with long text
            width = bounds.get("width", 0)
            if width > 0 and len(text) / width > 0.5:  # High text density
                overflows.append({
                    "type": "potential_truncation",
                    "text": text[:100],
                    "bounds": bounds,
                    "density": len(text) / width,
                })

        return overflows

    async def detect_element_overlap(
        self,
        snapshot: VisualSnapshot,
    ) -> List[Dict[str, Any]]:
        """Detect elements that overlap incorrectly.

        Checks all element pairs for unintended overlapping.

        Args:
            snapshot: VisualSnapshot to analyze

        Returns:
            List of overlap info dicts
        """
        overlaps = []

        elements = snapshot.elements
        for i, el1 in enumerate(elements):
            for el2 in elements[i + 1:]:
                overlap_area = self._calculate_overlap_area(el1.bounds, el2.bounds)

                if overlap_area > 0:
                    overlaps.append({
                        "element1": el1.selector,
                        "element2": el2.selector,
                        "overlap_area": overlap_area,
                        "element1_bounds": el1.bounds,
                        "element2_bounds": el2.bounds,
                    })

        return overlaps

    async def generate_responsive_report(
        self,
        snapshots: Dict[str, VisualSnapshot],
        issues: List[BreakpointIssue],
    ) -> Dict[str, Any]:
        """Generate a comprehensive responsive testing report.

        Args:
            snapshots: Captured viewport snapshots
            issues: Detected issues

        Returns:
            Report dictionary with summary and details
        """
        # Group issues by viewport
        issues_by_viewport: Dict[str, List[BreakpointIssue]] = {}
        for issue in issues:
            if issue.viewport not in issues_by_viewport:
                issues_by_viewport[issue.viewport] = []
            issues_by_viewport[issue.viewport].append(issue)

        # Group issues by type
        issues_by_type: Dict[str, int] = {}
        for issue in issues:
            issues_by_type[issue.issue_type] = issues_by_type.get(issue.issue_type, 0) + 1

        # Calculate severity distribution
        severity_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

        # Determine overall status
        has_critical = severity_counts.get(5, 0) > 0
        has_major = severity_counts.get(4, 0) > 0

        if has_critical:
            overall_status = "CRITICAL"
        elif has_major:
            overall_status = "FAILED"
        elif len(issues) > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASSED"

        report = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "viewports_tested": len(snapshots),
                "total_issues": len(issues),
                "issues_by_type": issues_by_type,
                "severity_distribution": severity_counts,
            },
            "viewports": {
                name: {
                    "snapshot_id": snapshot.id,
                    "dimensions": snapshot.viewport,
                    "element_count": len(snapshot.elements),
                    "issues": [i.to_dict() for i in issues_by_viewport.get(name, [])],
                }
                for name, snapshot in snapshots.items()
            },
            "recommendations": self._generate_recommendations(issues),
        }

        return report

    def _generate_recommendations(
        self,
        issues: List[BreakpointIssue],
    ) -> List[str]:
        """Generate actionable recommendations from issues."""
        recommendations = set()

        for issue in issues:
            if issue.recommendation:
                recommendations.add(issue.recommendation)

        # Add general recommendations based on issue patterns
        issue_types = [i.issue_type for i in issues]

        if "overflow" in issue_types:
            recommendations.add(
                "Consider adding 'overflow-x: hidden' to the body or using "
                "'max-width: 100%' on problematic elements"
            )

        if "touch_target" in issue_types:
            recommendations.add(
                "Increase touch target sizes to at least 44x44px for better mobile UX "
                "(Apple HIG recommendation)"
            )

        if "overlap" in issue_types:
            recommendations.add(
                "Review CSS positioning and z-index values to resolve overlapping elements"
            )

        return list(recommendations)
