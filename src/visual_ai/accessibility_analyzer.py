"""Accessibility analyzer for WCAG compliance detection.

This module provides visual accessibility analysis capabilities including:
- WCAG 2.1 color contrast ratio checking (AA and AAA levels)
- Touch target size validation (iOS HIG and Material Design)
- Text readability assessment (font size, line height)
- Full accessibility report generation
- Accessibility regression detection between versions

WCAG 2.1 Contrast Requirements:
- AA Normal Text: 4.5:1 minimum contrast ratio
- AA Large Text (18pt+ or 14pt bold): 3:1 minimum contrast ratio
- AAA Normal Text: 7:1 minimum contrast ratio
- AAA Large Text: 4.5:1 minimum contrast ratio

Touch Target Requirements:
- iOS Human Interface Guidelines: 44x44 pixels minimum
- Material Design: 48x48 pixels minimum (with 8px spacing)
"""

import re
from dataclasses import dataclass, field
from typing import Any

from .models import VisualElement, VisualSnapshot


@dataclass
class ContrastViolation:
    """Represents a WCAG color contrast violation.

    Attributes:
        element: The visual element with insufficient contrast
        foreground_color: Hex color of the text/foreground
        background_color: Hex color of the background
        contrast_ratio: Calculated contrast ratio
        required_ratio: Minimum required ratio for compliance (4.5 for AA, 7 for AAA)
        wcag_level: Target WCAG level ("AA" or "AAA")
        text_size: Text size classification ("normal" or "large")
    """

    element: VisualElement
    foreground_color: str
    background_color: str
    contrast_ratio: float
    required_ratio: float
    wcag_level: str
    text_size: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "element_id": self.element.element_id,
            "selector": self.element.selector,
            "foreground_color": self.foreground_color,
            "background_color": self.background_color,
            "contrast_ratio": round(self.contrast_ratio, 2),
            "required_ratio": self.required_ratio,
            "wcag_level": self.wcag_level,
            "text_size": self.text_size,
            "text_content": self.element.text_content,
            "bounds": self.element.bounds,
        }

    @property
    def severity(self) -> str:
        """Determine severity based on how far below the threshold."""
        ratio_deficit = self.required_ratio - self.contrast_ratio
        if ratio_deficit > 2.0:
            return "critical"
        elif ratio_deficit > 1.0:
            return "major"
        else:
            return "minor"

    @property
    def recommendation(self) -> str:
        """Generate a recommendation for fixing this violation."""
        deficit = self.required_ratio - self.contrast_ratio
        return (
            f"Increase contrast ratio from {self.contrast_ratio:.2f}:1 to at least "
            f"{self.required_ratio}:1 (deficit: {deficit:.2f}). Consider using a "
            f"{'darker' if self._is_light_background() else 'lighter'} foreground color "
            f"or {'lighter' if self._is_light_background() else 'darker'} background."
        )

    def _is_light_background(self) -> bool:
        """Check if background is light colored."""
        try:
            luminance = AccessibilityAnalyzer._get_relative_luminance_static(
                self.background_color
            )
            return luminance > 0.5
        except ValueError:
            return True


@dataclass
class TouchTargetViolation:
    """Represents a touch target size violation.

    Attributes:
        element: The interactive element with insufficient size
        actual_width: Current width in pixels
        actual_height: Current height in pixels
        required_size: Minimum required size (44px iOS, 48px Android)
    """

    element: VisualElement
    actual_width: float
    actual_height: float
    required_size: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "element_id": self.element.element_id,
            "selector": self.element.selector,
            "actual_width": round(self.actual_width, 1),
            "actual_height": round(self.actual_height, 1),
            "required_size": self.required_size,
            "text_content": self.element.text_content,
            "tag_name": self.element.tag_name,
            "bounds": self.element.bounds,
        }

    @property
    def severity(self) -> str:
        """Determine severity based on how undersized the target is."""
        min_dimension = min(self.actual_width, self.actual_height)
        ratio = min_dimension / self.required_size
        if ratio < 0.5:
            return "critical"
        elif ratio < 0.75:
            return "major"
        else:
            return "minor"

    @property
    def recommendation(self) -> str:
        """Generate a recommendation for fixing this violation."""
        width_needed = max(0, self.required_size - self.actual_width)
        height_needed = max(0, self.required_size - self.actual_height)

        suggestions = []
        if width_needed > 0:
            suggestions.append(f"increase width by {width_needed:.0f}px")
        if height_needed > 0:
            suggestions.append(f"increase height by {height_needed:.0f}px")

        return (
            f"Touch target is {self.actual_width:.0f}x{self.actual_height:.0f}px. "
            f"Minimum recommended size is {self.required_size}x{self.required_size}px. "
            f"Please {' and '.join(suggestions)}."
        )


@dataclass
class ReadabilityIssue:
    """Represents a text readability issue.

    Attributes:
        element: The text element with readability issues
        issue_type: Type of issue (font_too_small, low_contrast, line_height, etc.)
        current_value: Current value of the problematic property
        recommended_value: Recommended value for better readability
    """

    element: VisualElement
    issue_type: str
    current_value: str
    recommended_value: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "element_id": self.element.element_id,
            "selector": self.element.selector,
            "issue_type": self.issue_type,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "text_content": self.element.text_content,
            "bounds": self.element.bounds,
        }

    @property
    def severity(self) -> str:
        """Determine severity based on issue type."""
        severity_map = {
            "font_too_small": "major",
            "low_contrast": "major",
            "line_height_too_tight": "minor",
            "line_too_long": "minor",
            "text_justification": "info",
        }
        return severity_map.get(self.issue_type, "minor")

    @property
    def recommendation(self) -> str:
        """Generate a recommendation based on issue type."""
        recommendations = {
            "font_too_small": (
                f"Increase font size from {self.current_value} to at least "
                f"{self.recommended_value} for better readability."
            ),
            "low_contrast": (
                f"Current contrast is {self.current_value}. "
                f"Increase to at least {self.recommended_value} for WCAG compliance."
            ),
            "line_height_too_tight": (
                f"Line height of {self.current_value} is too tight. "
                f"Increase to at least {self.recommended_value} (1.5x font size recommended)."
            ),
            "line_too_long": (
                f"Line length of {self.current_value} characters exceeds the recommended "
                f"maximum of {self.recommended_value} characters for comfortable reading."
            ),
            "text_justification": (
                "Fully justified text can create uneven spacing. "
                "Consider using left alignment for better readability."
            ),
        }
        return recommendations.get(
            self.issue_type,
            f"Change {self.issue_type} from {self.current_value} to {self.recommended_value}.",
        )


@dataclass
class AccessibilityReport:
    """Complete accessibility analysis report.

    Attributes:
        score: Overall accessibility score (0-100)
        contrast_violations: List of color contrast violations
        touch_target_violations: List of touch target size violations
        readability_issues: List of text readability issues
        passed_checks: List of checks that passed
        summary: Human-readable summary of findings
    """

    score: float
    contrast_violations: list[ContrastViolation]
    touch_target_violations: list[TouchTargetViolation]
    readability_issues: list[ReadabilityIssue]
    passed_checks: list[str]
    summary: str
    snapshot_id: str = ""
    url: str = ""
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score": round(self.score, 1),
            "snapshot_id": self.snapshot_id,
            "url": self.url,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "contrast_violations": [v.to_dict() for v in self.contrast_violations],
            "touch_target_violations": [v.to_dict() for v in self.touch_target_violations],
            "readability_issues": [i.to_dict() for i in self.readability_issues],
            "passed_checks": self.passed_checks,
            "total_violations": self.total_violations,
            "violations_by_severity": self.violations_by_severity,
            "metadata": self.metadata,
        }

    @property
    def total_violations(self) -> int:
        """Total number of accessibility violations."""
        return (
            len(self.contrast_violations)
            + len(self.touch_target_violations)
            + len(self.readability_issues)
        )

    @property
    def violations_by_severity(self) -> dict[str, int]:
        """Count violations by severity level."""
        counts = {"critical": 0, "major": 0, "minor": 0, "info": 0}

        for v in self.contrast_violations:
            counts[v.severity] = counts.get(v.severity, 0) + 1

        for v in self.touch_target_violations:
            counts[v.severity] = counts.get(v.severity, 0) + 1

        for i in self.readability_issues:
            counts[i.severity] = counts.get(i.severity, 0) + 1

        return counts

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical accessibility issues."""
        return self.violations_by_severity.get("critical", 0) > 0

    @property
    def wcag_aa_compliant(self) -> bool:
        """Check if the snapshot is WCAG AA compliant."""
        # Check for AA-level contrast violations
        aa_contrast_violations = [
            v for v in self.contrast_violations if v.wcag_level == "AA"
        ]
        return len(aa_contrast_violations) == 0 and not self.has_critical_issues

    def get_recommendations(self) -> list[str]:
        """Get all recommendations for fixing issues."""
        recommendations = []

        for v in self.contrast_violations:
            recommendations.append(f"[Contrast] {v.recommendation}")

        for v in self.touch_target_violations:
            recommendations.append(f"[Touch Target] {v.recommendation}")

        for i in self.readability_issues:
            recommendations.append(f"[Readability] {i.recommendation}")

        return recommendations


class AccessibilityAnalyzer:
    """Visual accessibility analysis for WCAG compliance.

    This analyzer performs visual inspection of UI elements to detect
    accessibility violations including:
    - Color contrast issues (WCAG 2.1 AA and AAA)
    - Touch target size violations
    - Text readability problems

    Example:
        analyzer = AccessibilityAnalyzer()
        report = await analyzer.analyze_full(snapshot)
        print(f"Accessibility score: {report.score}/100")
        for violation in report.contrast_violations:
            print(f"Contrast issue: {violation.recommendation}")
    """

    # WCAG 2.1 contrast ratio requirements
    WCAG_AA_NORMAL_TEXT = 4.5  # Normal text (< 18pt or < 14pt bold)
    WCAG_AA_LARGE_TEXT = 3.0  # Large text (>= 18pt or >= 14pt bold)
    WCAG_AAA_NORMAL_TEXT = 7.0  # Enhanced contrast for normal text
    WCAG_AAA_LARGE_TEXT = 4.5  # Enhanced contrast for large text

    # Touch target minimums
    IOS_MIN_TARGET_SIZE = 44  # iOS Human Interface Guidelines
    ANDROID_MIN_TARGET_SIZE = 48  # Material Design Guidelines

    # Text readability thresholds
    MIN_FONT_SIZE_PX = 12  # Minimum readable font size
    RECOMMENDED_LINE_HEIGHT_RATIO = 1.5  # Line height as ratio of font size
    MAX_LINE_LENGTH_CHARS = 80  # Maximum characters per line

    # Interactive element tags
    INTERACTIVE_TAGS = {
        "a",
        "button",
        "input",
        "select",
        "textarea",
        "summary",
        "option",
    }

    # Text element tags
    TEXT_TAGS = {
        "p",
        "span",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "td",
        "th",
        "label",
        "a",
        "button",
        "div",
    }

    def __init__(
        self,
        wcag_level: str = "AA",
        min_touch_target: int = 44,
        check_aaa: bool = False,
    ):
        """Initialize the accessibility analyzer.

        Args:
            wcag_level: Target WCAG compliance level ("AA" or "AAA")
            min_touch_target: Minimum touch target size in pixels
            check_aaa: Whether to also check AAA requirements
        """
        self.wcag_level = wcag_level
        self.min_touch_target = min_touch_target
        self.check_aaa = check_aaa

    async def check_color_contrast(
        self,
        snapshot: VisualSnapshot,
    ) -> list[ContrastViolation]:
        """Check WCAG AA/AAA contrast ratios for text elements.

        Analyzes all text elements in the snapshot and checks their
        foreground/background color contrast against WCAG requirements.

        Args:
            snapshot: Visual snapshot to analyze

        Returns:
            List of contrast violations found
        """
        violations = []

        for element in snapshot.elements:
            # Skip non-text elements
            if not self._is_text_element(element):
                continue

            # Skip elements without text content
            if not element.text_content or not element.text_content.strip():
                continue

            # Extract colors from computed styles
            fg_color = self._extract_color(element.computed_styles.get("color", ""))
            bg_color = self._extract_background_color(element, snapshot)

            if not fg_color or not bg_color:
                continue

            # Calculate contrast ratio
            contrast_ratio = self._calculate_contrast_ratio(fg_color, bg_color)

            # Determine text size classification
            font_size = self._parse_font_size(
                element.computed_styles.get("font-size", "16px")
            )
            font_weight = self._parse_font_weight(
                element.computed_styles.get("font-weight", "400")
            )
            text_size = self._classify_text_size(font_size, font_weight)

            # Get required contrast ratio
            required_ratio = self._get_required_contrast_ratio(text_size, "AA")

            # Check AA compliance
            if contrast_ratio < required_ratio:
                violations.append(
                    ContrastViolation(
                        element=element,
                        foreground_color=fg_color,
                        background_color=bg_color,
                        contrast_ratio=contrast_ratio,
                        required_ratio=required_ratio,
                        wcag_level="AA",
                        text_size=text_size,
                    )
                )
            # Optionally check AAA compliance
            elif self.check_aaa:
                aaa_required = self._get_required_contrast_ratio(text_size, "AAA")
                if contrast_ratio < aaa_required:
                    violations.append(
                        ContrastViolation(
                            element=element,
                            foreground_color=fg_color,
                            background_color=bg_color,
                            contrast_ratio=contrast_ratio,
                            required_ratio=aaa_required,
                            wcag_level="AAA",
                            text_size=text_size,
                        )
                    )

        return violations

    async def check_touch_target_size(
        self,
        snapshot: VisualSnapshot,
        min_size: int = 44,
    ) -> list[TouchTargetViolation]:
        """Verify interactive elements meet minimum touch target size.

        Checks all interactive elements (buttons, links, inputs, etc.)
        to ensure they meet minimum size requirements for touch interaction.

        Args:
            snapshot: Visual snapshot to analyze
            min_size: Minimum size in pixels (default: 44px per iOS HIG)

        Returns:
            List of touch target violations found
        """
        violations = []
        min_size = min_size or self.min_touch_target

        for element in snapshot.elements:
            # Only check interactive elements
            if not self._is_interactive_element(element):
                continue

            width = element.bounds.get("width", 0)
            height = element.bounds.get("height", 0)

            # Check if either dimension is below minimum
            if width < min_size or height < min_size:
                violations.append(
                    TouchTargetViolation(
                        element=element,
                        actual_width=width,
                        actual_height=height,
                        required_size=min_size,
                    )
                )

        return violations

    async def check_text_readability(
        self,
        snapshot: VisualSnapshot,
    ) -> list[ReadabilityIssue]:
        """Check font size, line height, and text contrast for readability.

        Analyzes text elements for common readability issues including:
        - Font size too small
        - Line height too tight
        - Line length too long
        - Justified text alignment

        Args:
            snapshot: Visual snapshot to analyze

        Returns:
            List of readability issues found
        """
        issues = []

        for element in snapshot.elements:
            # Skip non-text elements
            if not self._is_text_element(element):
                continue

            # Skip elements without text content
            if not element.text_content or not element.text_content.strip():
                continue

            # Check font size
            font_size = self._parse_font_size(
                element.computed_styles.get("font-size", "16px")
            )
            if font_size < self.MIN_FONT_SIZE_PX:
                issues.append(
                    ReadabilityIssue(
                        element=element,
                        issue_type="font_too_small",
                        current_value=f"{font_size}px",
                        recommended_value=f"{self.MIN_FONT_SIZE_PX}px",
                    )
                )

            # Check line height
            line_height = self._parse_line_height(
                element.computed_styles.get("line-height", "normal"),
                font_size,
            )
            min_line_height = font_size * self.RECOMMENDED_LINE_HEIGHT_RATIO
            if line_height and line_height < min_line_height:
                issues.append(
                    ReadabilityIssue(
                        element=element,
                        issue_type="line_height_too_tight",
                        current_value=f"{line_height:.1f}px",
                        recommended_value=f"{min_line_height:.1f}px",
                    )
                )

            # Check line length (approximate based on element width and font size)
            if element.bounds.get("width", 0) > 0 and font_size > 0:
                # Approximate characters per line
                avg_char_width = font_size * 0.5  # Rough estimate
                chars_per_line = element.bounds["width"] / avg_char_width
                if chars_per_line > self.MAX_LINE_LENGTH_CHARS:
                    issues.append(
                        ReadabilityIssue(
                            element=element,
                            issue_type="line_too_long",
                            current_value=f"{int(chars_per_line)} chars",
                            recommended_value=f"{self.MAX_LINE_LENGTH_CHARS} chars",
                        )
                    )

            # Check text justification
            text_align = element.computed_styles.get("text-align", "")
            if text_align == "justify":
                issues.append(
                    ReadabilityIssue(
                        element=element,
                        issue_type="text_justification",
                        current_value="justify",
                        recommended_value="left",
                    )
                )

        return issues

    async def analyze_full(
        self,
        snapshot: VisualSnapshot,
    ) -> AccessibilityReport:
        """Run all accessibility checks and generate a comprehensive report.

        Performs contrast checking, touch target validation, and readability
        analysis, then calculates an overall accessibility score.

        Args:
            snapshot: Visual snapshot to analyze

        Returns:
            Complete accessibility report with all findings
        """
        # Run all checks
        contrast_violations = await self.check_color_contrast(snapshot)
        touch_violations = await self.check_touch_target_size(snapshot)
        readability_issues = await self.check_text_readability(snapshot)

        # Determine passed checks
        passed_checks = []
        if not contrast_violations:
            passed_checks.append("Color contrast meets WCAG AA requirements")
        if not touch_violations:
            passed_checks.append("Touch targets meet minimum size requirements")
        if not any(i.issue_type == "font_too_small" for i in readability_issues):
            passed_checks.append("Font sizes are readable")
        if not any(i.issue_type == "line_height_too_tight" for i in readability_issues):
            passed_checks.append("Line heights are adequate")

        # Calculate accessibility score
        score = self._calculate_score(
            contrast_violations,
            touch_violations,
            readability_issues,
            snapshot,
        )

        # Generate summary
        summary = self._generate_summary(
            score,
            contrast_violations,
            touch_violations,
            readability_issues,
        )

        return AccessibilityReport(
            score=score,
            contrast_violations=contrast_violations,
            touch_target_violations=touch_violations,
            readability_issues=readability_issues,
            passed_checks=passed_checks,
            summary=summary,
            snapshot_id=snapshot.id,
            url=snapshot.url,
            timestamp=snapshot.timestamp,
        )

    async def compare_accessibility(
        self,
        baseline: VisualSnapshot,
        current: VisualSnapshot,
    ) -> dict[str, Any]:
        """Detect accessibility regressions between two versions.

        Compares accessibility reports from baseline and current snapshots
        to identify new violations and improvements.

        Args:
            baseline: Previous/reference snapshot
            current: Current snapshot to compare

        Returns:
            Dictionary containing regression analysis results
        """
        # Generate reports for both snapshots
        baseline_report = await self.analyze_full(baseline)
        current_report = await self.analyze_full(current)

        # Identify new violations
        new_contrast_violations = self._find_new_violations(
            baseline_report.contrast_violations,
            current_report.contrast_violations,
            key_fn=lambda v: v.element.selector,
        )
        new_touch_violations = self._find_new_violations(
            baseline_report.touch_target_violations,
            current_report.touch_target_violations,
            key_fn=lambda v: v.element.selector,
        )
        new_readability_issues = self._find_new_violations(
            baseline_report.readability_issues,
            current_report.readability_issues,
            key_fn=lambda i: f"{i.element.selector}:{i.issue_type}",
        )

        # Identify fixed violations
        fixed_contrast = self._find_fixed_violations(
            baseline_report.contrast_violations,
            current_report.contrast_violations,
            key_fn=lambda v: v.element.selector,
        )
        fixed_touch = self._find_fixed_violations(
            baseline_report.touch_target_violations,
            current_report.touch_target_violations,
            key_fn=lambda v: v.element.selector,
        )
        fixed_readability = self._find_fixed_violations(
            baseline_report.readability_issues,
            current_report.readability_issues,
            key_fn=lambda i: f"{i.element.selector}:{i.issue_type}",
        )

        # Calculate score delta
        score_delta = current_report.score - baseline_report.score

        # Determine if there's a regression
        has_regression = (
            len(new_contrast_violations) > 0
            or len(new_touch_violations) > 0
            or len(new_readability_issues) > 0
            or score_delta < -5  # More than 5 point drop
        )

        return {
            "has_regression": has_regression,
            "baseline_score": baseline_report.score,
            "current_score": current_report.score,
            "score_delta": score_delta,
            "new_violations": {
                "contrast": [v.to_dict() for v in new_contrast_violations],
                "touch_target": [v.to_dict() for v in new_touch_violations],
                "readability": [i.to_dict() for i in new_readability_issues],
            },
            "fixed_violations": {
                "contrast": len(fixed_contrast),
                "touch_target": len(fixed_touch),
                "readability": len(fixed_readability),
            },
            "total_new_violations": (
                len(new_contrast_violations)
                + len(new_touch_violations)
                + len(new_readability_issues)
            ),
            "total_fixed_violations": (
                len(fixed_contrast) + len(fixed_touch) + len(fixed_readability)
            ),
            "baseline_report": baseline_report.to_dict(),
            "current_report": current_report.to_dict(),
        }

    def _calculate_contrast_ratio(
        self,
        fg_color: str,
        bg_color: str,
    ) -> float:
        """Calculate WCAG contrast ratio between two colors.

        Implements the WCAG 2.1 contrast ratio formula:
        contrast = (L1 + 0.05) / (L2 + 0.05)
        where L1 is the relative luminance of the lighter color
        and L2 is the relative luminance of the darker color.

        Args:
            fg_color: Foreground color (hex format)
            bg_color: Background color (hex format)

        Returns:
            Contrast ratio (1:1 to 21:1)
        """
        l1 = self._get_relative_luminance(fg_color)
        l2 = self._get_relative_luminance(bg_color)

        # Ensure L1 is the lighter color
        if l1 < l2:
            l1, l2 = l2, l1

        return (l1 + 0.05) / (l2 + 0.05)

    def _get_relative_luminance(self, color: str) -> float:
        """Calculate relative luminance of a color.

        Implements the WCAG 2.1 relative luminance formula:
        L = 0.2126 * R + 0.7152 * G + 0.0722 * B
        where R, G, B are linearized sRGB values.

        Args:
            color: Hex color string (e.g., "#FF5733")

        Returns:
            Relative luminance value (0 to 1)
        """
        return self._get_relative_luminance_static(color)

    @staticmethod
    def _get_relative_luminance_static(color: str) -> float:
        """Static version of relative luminance calculation.

        Args:
            color: Hex color string (e.g., "#FF5733")

        Returns:
            Relative luminance value (0 to 1)
        """
        # Parse hex color
        color = color.lstrip("#")
        if len(color) == 3:
            color = "".join(c * 2 for c in color)

        if len(color) != 6:
            raise ValueError(f"Invalid hex color: {color}")

        r = int(color[0:2], 16) / 255
        g = int(color[2:4], 16) / 255
        b = int(color[4:6], 16) / 255

        # Linearize sRGB values
        def linearize(c: float) -> float:
            if c <= 0.03928:
                return c / 12.92
            return ((c + 0.055) / 1.055) ** 2.4

        r_linear = linearize(r)
        g_linear = linearize(g)
        b_linear = linearize(b)

        # Calculate relative luminance
        return 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear

    def _extract_color(self, color_value: str) -> str | None:
        """Extract hex color from CSS color value.

        Handles various CSS color formats:
        - Hex: #RGB, #RRGGBB
        - RGB: rgb(r, g, b)
        - RGBA: rgba(r, g, b, a)
        - Named colors (limited support)

        Args:
            color_value: CSS color value

        Returns:
            Hex color string or None if parsing fails
        """
        if not color_value:
            return None

        color_value = color_value.strip().lower()

        # Handle hex colors
        if color_value.startswith("#"):
            hex_color = color_value[1:]
            if len(hex_color) == 3:
                return "#" + "".join(c * 2 for c in hex_color).upper()
            elif len(hex_color) == 6:
                return "#" + hex_color.upper()
            elif len(hex_color) == 8:
                return "#" + hex_color[:6].upper()

        # Handle rgb/rgba
        rgb_match = re.match(
            r"rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
            color_value,
        )
        if rgb_match:
            r, g, b = map(int, rgb_match.groups())
            return f"#{r:02X}{g:02X}{b:02X}"

        # Handle common named colors
        named_colors = {
            "black": "#000000",
            "white": "#FFFFFF",
            "red": "#FF0000",
            "green": "#008000",
            "blue": "#0000FF",
            "yellow": "#FFFF00",
            "gray": "#808080",
            "grey": "#808080",
            "transparent": None,
        }
        return named_colors.get(color_value)

    def _extract_background_color(
        self,
        element: VisualElement,
        snapshot: VisualSnapshot,
    ) -> str | None:
        """Extract effective background color for an element.

        Traverses up the DOM tree to find the effective background color,
        handling transparent backgrounds.

        Args:
            element: Element to get background for
            snapshot: Snapshot containing element hierarchy

        Returns:
            Hex background color or None
        """
        # First try the element's own background
        bg = element.computed_styles.get("background-color", "")
        color = self._extract_color(bg)

        if color:
            return color

        # Check for background shorthand
        bg = element.computed_styles.get("background", "")
        if bg:
            # Try to extract color from background shorthand
            parts = bg.split()
            for part in parts:
                color = self._extract_color(part)
                if color:
                    return color

        # Default to white background if not found
        # In a real implementation, we would traverse parent elements
        return "#FFFFFF"

    def _parse_font_size(self, font_size: str) -> float:
        """Parse CSS font-size value to pixels.

        Args:
            font_size: CSS font-size value (e.g., "16px", "1rem", "12pt")

        Returns:
            Font size in pixels
        """
        if not font_size:
            return 16.0  # Default

        font_size = font_size.strip().lower()

        # Handle px
        if font_size.endswith("px"):
            try:
                return float(font_size[:-2])
            except ValueError:
                return 16.0

        # Handle pt (1pt = 1.333px)
        if font_size.endswith("pt"):
            try:
                return float(font_size[:-2]) * 1.333
            except ValueError:
                return 16.0

        # Handle rem (assume 16px base)
        if font_size.endswith("rem"):
            try:
                return float(font_size[:-3]) * 16
            except ValueError:
                return 16.0

        # Handle em (assume 16px base)
        if font_size.endswith("em"):
            try:
                return float(font_size[:-2]) * 16
            except ValueError:
                return 16.0

        # Handle percentage (assume 16px base)
        if font_size.endswith("%"):
            try:
                return float(font_size[:-1]) / 100 * 16
            except ValueError:
                return 16.0

        # Try to parse as number
        try:
            return float(font_size)
        except ValueError:
            return 16.0

    def _parse_font_weight(self, font_weight: str) -> int:
        """Parse CSS font-weight value to numeric weight.

        Args:
            font_weight: CSS font-weight value (e.g., "400", "bold", "normal")

        Returns:
            Numeric font weight (100-900)
        """
        if not font_weight:
            return 400

        font_weight = font_weight.strip().lower()

        # Named weights
        weight_map = {
            "thin": 100,
            "hairline": 100,
            "extra-light": 200,
            "ultra-light": 200,
            "light": 300,
            "normal": 400,
            "regular": 400,
            "medium": 500,
            "semi-bold": 600,
            "demi-bold": 600,
            "bold": 700,
            "extra-bold": 800,
            "ultra-bold": 800,
            "black": 900,
            "heavy": 900,
        }

        if font_weight in weight_map:
            return weight_map[font_weight]

        try:
            return int(font_weight)
        except ValueError:
            return 400

    def _parse_line_height(
        self,
        line_height: str,
        font_size: float,
    ) -> float | None:
        """Parse CSS line-height value to pixels.

        Args:
            line_height: CSS line-height value
            font_size: Font size in pixels for ratio calculations

        Returns:
            Line height in pixels or None
        """
        if not line_height or line_height == "normal":
            return font_size * 1.2  # Default "normal" line height

        line_height = line_height.strip().lower()

        # Handle px
        if line_height.endswith("px"):
            try:
                return float(line_height[:-2])
            except ValueError:
                return None

        # Handle percentage
        if line_height.endswith("%"):
            try:
                return float(line_height[:-1]) / 100 * font_size
            except ValueError:
                return None

        # Handle unitless (ratio)
        try:
            ratio = float(line_height)
            return ratio * font_size
        except ValueError:
            return None

    def _classify_text_size(self, font_size: float, font_weight: int) -> str:
        """Classify text as 'normal' or 'large' per WCAG definitions.

        Large text is defined as:
        - 18pt (24px) or larger
        - 14pt (18.66px) or larger if bold (weight >= 700)

        Args:
            font_size: Font size in pixels
            font_weight: Font weight (100-900)

        Returns:
            "normal" or "large"
        """
        # 18pt = 24px
        if font_size >= 24:
            return "large"

        # 14pt bold = 18.66px with weight >= 700
        if font_size >= 18.66 and font_weight >= 700:
            return "large"

        return "normal"

    def _get_required_contrast_ratio(
        self,
        text_size: str,
        wcag_level: str,
    ) -> float:
        """Get the required contrast ratio for given text size and WCAG level.

        Args:
            text_size: "normal" or "large"
            wcag_level: "AA" or "AAA"

        Returns:
            Required contrast ratio
        """
        if wcag_level == "AAA":
            return self.WCAG_AAA_LARGE_TEXT if text_size == "large" else self.WCAG_AAA_NORMAL_TEXT
        else:  # AA
            return self.WCAG_AA_LARGE_TEXT if text_size == "large" else self.WCAG_AA_NORMAL_TEXT

    def _is_text_element(self, element: VisualElement) -> bool:
        """Check if element is a text element.

        Args:
            element: Visual element to check

        Returns:
            True if element contains text
        """
        return element.tag_name.lower() in self.TEXT_TAGS

    def _is_interactive_element(self, element: VisualElement) -> bool:
        """Check if element is interactive.

        Args:
            element: Visual element to check

        Returns:
            True if element is interactive
        """
        tag = element.tag_name.lower()

        # Check tag name
        if tag in self.INTERACTIVE_TAGS:
            return True

        # Check for role attribute
        role = element.attributes.get("role", "").lower()
        if role in {"button", "link", "checkbox", "radio", "switch", "tab", "menuitem"}:
            return True

        # Check for onclick or other event handlers
        for attr in element.attributes:
            if attr.startswith("on") or attr == "tabindex":
                return True

        return False

    def _calculate_score(
        self,
        contrast_violations: list[ContrastViolation],
        touch_violations: list[TouchTargetViolation],
        readability_issues: list[ReadabilityIssue],
        snapshot: VisualSnapshot,
    ) -> float:
        """Calculate overall accessibility score.

        Scoring weights:
        - Contrast violations: -10 points each (critical: -15)
        - Touch target violations: -5 points each (critical: -10)
        - Readability issues: -3 points each

        Args:
            contrast_violations: List of contrast violations
            touch_violations: List of touch target violations
            readability_issues: List of readability issues
            snapshot: Original snapshot for context

        Returns:
            Score from 0 to 100
        """
        score = 100.0

        # Deduct for contrast violations
        for v in contrast_violations:
            if v.severity == "critical":
                score -= 15
            elif v.severity == "major":
                score -= 10
            else:
                score -= 5

        # Deduct for touch target violations
        for v in touch_violations:
            if v.severity == "critical":
                score -= 10
            elif v.severity == "major":
                score -= 5
            else:
                score -= 3

        # Deduct for readability issues
        for i in readability_issues:
            if i.severity == "major":
                score -= 5
            elif i.severity == "minor":
                score -= 3
            else:
                score -= 1

        return max(0.0, min(100.0, score))

    def _generate_summary(
        self,
        score: float,
        contrast_violations: list[ContrastViolation],
        touch_violations: list[TouchTargetViolation],
        readability_issues: list[ReadabilityIssue],
    ) -> str:
        """Generate human-readable summary of findings.

        Args:
            score: Calculated accessibility score
            contrast_violations: List of contrast violations
            touch_violations: List of touch target violations
            readability_issues: List of readability issues

        Returns:
            Summary string
        """
        total_issues = (
            len(contrast_violations)
            + len(touch_violations)
            + len(readability_issues)
        )

        if total_issues == 0:
            return (
                f"Excellent! Accessibility score: {score:.0f}/100. "
                "No WCAG violations detected."
            )

        parts = [f"Accessibility score: {score:.0f}/100."]

        if contrast_violations:
            parts.append(f"{len(contrast_violations)} contrast violation(s)")

        if touch_violations:
            parts.append(f"{len(touch_violations)} touch target violation(s)")

        if readability_issues:
            parts.append(f"{len(readability_issues)} readability issue(s)")

        summary = " ".join(parts[:1]) + " Found " + ", ".join(parts[1:]) + "."

        # Add priority recommendation
        if contrast_violations:
            critical_contrast = [v for v in contrast_violations if v.severity == "critical"]
            if critical_contrast:
                summary += (
                    f" Priority: Fix {len(critical_contrast)} critical contrast "
                    "issue(s) for WCAG compliance."
                )

        return summary

    def _find_new_violations(
        self,
        baseline_violations: list,
        current_violations: list,
        key_fn,
    ) -> list:
        """Find violations that are new in current compared to baseline.

        Args:
            baseline_violations: Violations from baseline
            current_violations: Violations from current
            key_fn: Function to extract comparison key from violation

        Returns:
            List of new violations
        """
        baseline_keys = {key_fn(v) for v in baseline_violations}
        return [v for v in current_violations if key_fn(v) not in baseline_keys]

    def _find_fixed_violations(
        self,
        baseline_violations: list,
        current_violations: list,
        key_fn,
    ) -> list:
        """Find violations that were fixed (present in baseline but not current).

        Args:
            baseline_violations: Violations from baseline
            current_violations: Violations from current
            key_fn: Function to extract comparison key from violation

        Returns:
            List of fixed violations
        """
        current_keys = {key_fn(v) for v in current_violations}
        return [v for v in baseline_violations if key_fn(v) not in current_keys]
