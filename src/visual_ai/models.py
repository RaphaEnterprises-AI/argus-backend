"""Visual AI data models for visual regression testing.

This module contains dataclasses and enums used throughout the Visual AI
testing system for representing snapshots, changes, comparisons, and results.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ChangeCategory(Enum):
    """Categories of visual changes detected during comparison."""

    LAYOUT = "layout"
    CONTENT = "content"
    STYLE = "style"
    STRUCTURE = "structure"
    RESPONSIVE = "responsive"
    ANIMATION = "animation"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"


class ChangeIntent(Enum):
    """Classification of whether a change was intentional or a regression."""

    INTENTIONAL = "intentional"
    REGRESSION = "regression"
    DYNAMIC = "dynamic"
    ENVIRONMENTAL = "environmental"
    UNKNOWN = "unknown"


class Severity(Enum):
    """Severity levels for visual changes."""

    CRITICAL = 4
    MAJOR = 3
    MINOR = 2
    INFO = 1
    SAFE = 0

    def __lt__(self, other: "Severity") -> bool:
        if isinstance(other, Severity):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other: "Severity") -> bool:
        if isinstance(other, Severity):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other: "Severity") -> bool:
        if isinstance(other, Severity):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other: "Severity") -> bool:
        if isinstance(other, Severity):
            return self.value >= other.value
        return NotImplemented


@dataclass
class VisualElement:
    """Represents a visual element extracted from the DOM."""

    element_id: str
    selector: str
    tag_name: str
    bounds: dict[str, float]  # x, y, width, height
    computed_styles: dict[str, str]
    text_content: str | None
    attributes: dict[str, str]
    children_count: int
    screenshot_region: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        # Convert bytes to base64 string if present
        if self.screenshot_region is not None:
            import base64
            result["screenshot_region"] = base64.b64encode(self.screenshot_region).decode("utf-8")
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualElement":
        """Create instance from dictionary."""
        if data.get("screenshot_region") is not None:
            import base64
            data["screenshot_region"] = base64.b64decode(data["screenshot_region"])
        return cls(**data)

    def get_center(self) -> tuple[float, float]:
        """Get the center point of the element bounds."""
        return (
            self.bounds["x"] + self.bounds["width"] / 2,
            self.bounds["y"] + self.bounds["height"] / 2
        )

    def get_area(self) -> float:
        """Get the area of the element in pixels squared."""
        return self.bounds["width"] * self.bounds["height"]

    def overlaps(self, other: "VisualElement") -> bool:
        """Check if this element overlaps with another."""
        return not (
            self.bounds["x"] + self.bounds["width"] < other.bounds["x"] or
            other.bounds["x"] + other.bounds["width"] < self.bounds["x"] or
            self.bounds["y"] + self.bounds["height"] < other.bounds["y"] or
            other.bounds["y"] + other.bounds["height"] < self.bounds["y"]
        )


@dataclass
class VisualChange:
    """Represents a detected visual change between baseline and current snapshot."""

    id: str
    category: ChangeCategory
    intent: ChangeIntent
    severity: Severity
    element: VisualElement | None
    bounds_baseline: dict[str, float] | None
    bounds_current: dict[str, float] | None
    property_name: str | None
    baseline_value: Any
    current_value: Any
    description: str
    root_cause: str | None
    impact_assessment: str
    recommendation: str
    confidence: float
    related_commit: str | None
    related_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "category": self.category.value,
            "intent": self.intent.value,
            "severity": self.severity.value,
            "element": self.element.to_dict() if self.element else None,
            "bounds_baseline": self.bounds_baseline,
            "bounds_current": self.bounds_current,
            "property_name": self.property_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "description": self.description,
            "root_cause": self.root_cause,
            "impact_assessment": self.impact_assessment,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "related_commit": self.related_commit,
            "related_files": self.related_files,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualChange":
        """Create instance from dictionary."""
        return cls(
            id=data["id"],
            category=ChangeCategory(data["category"]),
            intent=ChangeIntent(data["intent"]),
            severity=Severity(data["severity"]),
            element=VisualElement.from_dict(data["element"]) if data.get("element") else None,
            bounds_baseline=data.get("bounds_baseline"),
            bounds_current=data.get("bounds_current"),
            property_name=data.get("property_name"),
            baseline_value=data.get("baseline_value"),
            current_value=data.get("current_value"),
            description=data["description"],
            root_cause=data.get("root_cause"),
            impact_assessment=data["impact_assessment"],
            recommendation=data["recommendation"],
            confidence=data["confidence"],
            related_commit=data.get("related_commit"),
            related_files=data.get("related_files", []),
        )

    def is_blocking(self, threshold: Severity = Severity.MAJOR) -> bool:
        """Check if this change should block deployment based on severity."""
        return self.severity >= threshold and self.intent != ChangeIntent.INTENTIONAL

    def is_regression(self) -> bool:
        """Check if this change is classified as a regression."""
        return self.intent == ChangeIntent.REGRESSION

    def get_bounds_delta(self) -> dict[str, float] | None:
        """Calculate the difference in bounds between baseline and current."""
        if not self.bounds_baseline or not self.bounds_current:
            return None
        return {
            "x": self.bounds_current["x"] - self.bounds_baseline["x"],
            "y": self.bounds_current["y"] - self.bounds_baseline["y"],
            "width": self.bounds_current["width"] - self.bounds_baseline["width"],
            "height": self.bounds_current["height"] - self.bounds_baseline["height"],
        }


@dataclass
class VisualSnapshot:
    """Represents a captured visual snapshot of a page."""

    id: str
    url: str
    viewport: dict[str, int]
    device_name: str | None
    browser: str
    timestamp: str
    screenshot: bytes
    dom_snapshot: str
    computed_styles: dict[str, dict]
    network_har: dict | None
    elements: list[VisualElement]
    layout_hash: str
    color_palette: list[str]
    text_blocks: list[dict]
    largest_contentful_paint: float | None
    cumulative_layout_shift: float | None
    time_to_interactive: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        import base64
        return {
            "id": self.id,
            "url": self.url,
            "viewport": self.viewport,
            "device_name": self.device_name,
            "browser": self.browser,
            "timestamp": self.timestamp,
            "screenshot": base64.b64encode(self.screenshot).decode("utf-8"),
            "dom_snapshot": self.dom_snapshot,
            "computed_styles": self.computed_styles,
            "network_har": self.network_har,
            "elements": [el.to_dict() for el in self.elements],
            "layout_hash": self.layout_hash,
            "color_palette": self.color_palette,
            "text_blocks": self.text_blocks,
            "largest_contentful_paint": self.largest_contentful_paint,
            "cumulative_layout_shift": self.cumulative_layout_shift,
            "time_to_interactive": self.time_to_interactive,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualSnapshot":
        """Create instance from dictionary."""
        import base64
        return cls(
            id=data["id"],
            url=data["url"],
            viewport=data["viewport"],
            device_name=data.get("device_name"),
            browser=data["browser"],
            timestamp=data["timestamp"],
            screenshot=base64.b64decode(data["screenshot"]),
            dom_snapshot=data["dom_snapshot"],
            computed_styles=data["computed_styles"],
            network_har=data.get("network_har"),
            elements=[VisualElement.from_dict(el) for el in data.get("elements", [])],
            layout_hash=data["layout_hash"],
            color_palette=data.get("color_palette", []),
            text_blocks=data.get("text_blocks", []),
            largest_contentful_paint=data.get("largest_contentful_paint"),
            cumulative_layout_shift=data.get("cumulative_layout_shift"),
            time_to_interactive=data.get("time_to_interactive"),
        )

    def get_element_by_selector(self, selector: str) -> VisualElement | None:
        """Find an element by its CSS selector."""
        for element in self.elements:
            if element.selector == selector:
                return element
        return None

    def get_elements_by_tag(self, tag_name: str) -> list[VisualElement]:
        """Find all elements with a specific tag name."""
        return [el for el in self.elements if el.tag_name.lower() == tag_name.lower()]

    def get_performance_score(self) -> float | None:
        """Calculate a simple performance score based on Core Web Vitals."""
        scores = []

        if self.largest_contentful_paint is not None:
            # LCP: Good < 2.5s, Needs improvement < 4s, Poor >= 4s
            if self.largest_contentful_paint < 2500:
                scores.append(1.0)
            elif self.largest_contentful_paint < 4000:
                scores.append(0.5)
            else:
                scores.append(0.0)

        if self.cumulative_layout_shift is not None:
            # CLS: Good < 0.1, Needs improvement < 0.25, Poor >= 0.25
            if self.cumulative_layout_shift < 0.1:
                scores.append(1.0)
            elif self.cumulative_layout_shift < 0.25:
                scores.append(0.5)
            else:
                scores.append(0.0)

        if self.time_to_interactive is not None:
            # TTI: Good < 3.8s, Needs improvement < 7.3s, Poor >= 7.3s
            if self.time_to_interactive < 3800:
                scores.append(1.0)
            elif self.time_to_interactive < 7300:
                scores.append(0.5)
            else:
                scores.append(0.0)

        if not scores:
            return None
        return sum(scores) / len(scores)

    def to_json(self) -> str:
        """Serialize snapshot to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "VisualSnapshot":
        """Deserialize snapshot from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class VisualComparisonResult:
    """Represents the result of comparing two visual snapshots."""

    id: str
    baseline_snapshot: str
    current_snapshot: str
    match: bool
    match_percentage: float
    changes: list[VisualChange]
    changes_by_category: dict[str, int]
    changes_by_severity: dict[str, int]
    auto_approval_recommendation: bool
    approval_confidence: float
    blocking_changes: list[str]
    diff_image_url: str
    side_by_side_url: str
    animated_gif_url: str | None
    lcp_delta: float | None
    cls_delta: float | None
    analysis_cost_usd: float
    analysis_duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "baseline_snapshot": self.baseline_snapshot,
            "current_snapshot": self.current_snapshot,
            "match": self.match,
            "match_percentage": self.match_percentage,
            "changes": [c.to_dict() for c in self.changes],
            "changes_by_category": self.changes_by_category,
            "changes_by_severity": self.changes_by_severity,
            "auto_approval_recommendation": self.auto_approval_recommendation,
            "approval_confidence": self.approval_confidence,
            "blocking_changes": self.blocking_changes,
            "diff_image_url": self.diff_image_url,
            "side_by_side_url": self.side_by_side_url,
            "animated_gif_url": self.animated_gif_url,
            "lcp_delta": self.lcp_delta,
            "cls_delta": self.cls_delta,
            "analysis_cost_usd": self.analysis_cost_usd,
            "analysis_duration_ms": self.analysis_duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualComparisonResult":
        """Create instance from dictionary."""
        return cls(
            id=data["id"],
            baseline_snapshot=data["baseline_snapshot"],
            current_snapshot=data["current_snapshot"],
            match=data["match"],
            match_percentage=data["match_percentage"],
            changes=[VisualChange.from_dict(c) for c in data.get("changes", [])],
            changes_by_category=data.get("changes_by_category", {}),
            changes_by_severity=data.get("changes_by_severity", {}),
            auto_approval_recommendation=data.get("auto_approval_recommendation", False),
            approval_confidence=data.get("approval_confidence", 0.0),
            blocking_changes=data.get("blocking_changes", []),
            diff_image_url=data["diff_image_url"],
            side_by_side_url=data["side_by_side_url"],
            animated_gif_url=data.get("animated_gif_url"),
            lcp_delta=data.get("lcp_delta"),
            cls_delta=data.get("cls_delta"),
            analysis_cost_usd=data.get("analysis_cost_usd", 0.0),
            analysis_duration_ms=data.get("analysis_duration_ms", 0),
        )

    def has_blocking_changes(self) -> bool:
        """Check if there are any changes that should block deployment."""
        return len(self.blocking_changes) > 0

    def get_blocking_change_objects(self) -> list[VisualChange]:
        """Get the actual VisualChange objects for blocking changes."""
        blocking_ids = set(self.blocking_changes)
        return [c for c in self.changes if c.id in blocking_ids]

    def get_changes_by_category(self, category: ChangeCategory) -> list[VisualChange]:
        """Get all changes in a specific category."""
        return [c for c in self.changes if c.category == category]

    def get_changes_by_severity(self, severity: Severity) -> list[VisualChange]:
        """Get all changes with a specific severity."""
        return [c for c in self.changes if c.severity == severity]

    def get_regressions(self) -> list[VisualChange]:
        """Get all changes classified as regressions."""
        return [c for c in self.changes if c.is_regression()]

    def get_highest_severity(self) -> Severity | None:
        """Get the highest severity level among all changes."""
        if not self.changes:
            return None
        return max(c.severity for c in self.changes)

    def get_summary(self) -> str:
        """Generate a human-readable summary of the comparison result."""
        if self.match:
            return f"Visual comparison passed with {self.match_percentage:.1f}% match."

        summary_parts = [
            f"Visual comparison found {len(self.changes)} change(s) "
            f"({self.match_percentage:.1f}% match)."
        ]

        if self.blocking_changes:
            summary_parts.append(
                f"{len(self.blocking_changes)} blocking change(s) detected."
            )

        if self.changes_by_severity:
            severity_summary = ", ".join(
                f"{count} {severity}"
                for severity, count in sorted(
                    self.changes_by_severity.items(),
                    key=lambda x: Severity[x[0].upper()].value if isinstance(x[0], str) else x[0],
                    reverse=True
                )
                if count > 0
            )
            if severity_summary:
                summary_parts.append(f"By severity: {severity_summary}")

        return " ".join(summary_parts)

    def should_auto_approve(self, confidence_threshold: float = 0.9) -> bool:
        """Determine if this result should be auto-approved."""
        return (
            self.auto_approval_recommendation and
            self.approval_confidence >= confidence_threshold and
            not self.has_blocking_changes()
        )

    def to_json(self) -> str:
        """Serialize comparison result to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "VisualComparisonResult":
        """Deserialize comparison result from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_performance_regression(self) -> str | None:
        """Check for performance regressions based on Core Web Vitals deltas."""
        regressions = []

        if self.lcp_delta is not None and self.lcp_delta > 500:  # 500ms threshold
            regressions.append(f"LCP increased by {self.lcp_delta:.0f}ms")

        if self.cls_delta is not None and self.cls_delta > 0.05:  # 0.05 threshold
            regressions.append(f"CLS increased by {self.cls_delta:.3f}")

        if regressions:
            return "; ".join(regressions)
        return None
