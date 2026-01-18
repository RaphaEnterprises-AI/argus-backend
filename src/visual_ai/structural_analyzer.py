"""Structural analysis for visual comparison.

This module provides structural diff analysis between screenshots,
extracting element-level changes before AI semantic analysis.

Key Features:
- DOM-level diffing to understand WHAT changed structurally
- Element matching by ID, selector, and position proximity
- Layout shift detection using Euclidean distance
- Component change tracking across versions
- Text comparison using difflib for similarity scoring
"""

import difflib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from html.parser import HTMLParser
from typing import Any

from .models import VisualElement, VisualSnapshot

logger = logging.getLogger(__name__)


class StructuralChangeType(str, Enum):
    """Types of structural changes detected."""

    ADDED = "added"  # Element appeared
    REMOVED = "removed"  # Element disappeared
    MOVED = "moved"  # Element position changed
    RESIZED = "resized"  # Element dimensions changed
    MODIFIED = "modified"  # Element content/style changed
    UNCHANGED = "unchanged"  # No change detected


@dataclass
class ElementBounds:
    """Bounding box for an element."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        """Get center point."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Get area in pixels."""
        return self.width * self.height

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "ElementBounds":
        return cls(
            x=data.get("x", 0),
            y=data.get("y", 0),
            width=data.get("width", 0),
            height=data.get("height", 0),
        )

    def distance_to(self, other: "ElementBounds") -> float:
        """Calculate distance between centers of two bounds."""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5

    def overlap_ratio(self, other: "ElementBounds") -> float:
        """Calculate overlap ratio between two bounds."""
        x_overlap = max(
            0,
            min(self.x + self.width, other.x + other.width)
            - max(self.x, other.x),
        )
        y_overlap = max(
            0,
            min(self.y + self.height, other.y + other.height)
            - max(self.y, other.y),
        )
        overlap_area = x_overlap * y_overlap
        total_area = self.area + other.area - overlap_area
        if total_area == 0:
            return 0.0
        return overlap_area / total_area


@dataclass
class StructuralElement:
    """An element detected in structural analysis."""

    id: str
    tag: str
    bounds: ElementBounds
    selector: str | None = None
    text_content: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    computed_styles: dict[str, str] = field(default_factory=dict)
    children_count: int = 0
    z_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tag": self.tag,
            "bounds": self.bounds.to_dict(),
            "selector": self.selector,
            "text_content": self.text_content,
            "attributes": self.attributes,
            "computed_styles": self.computed_styles,
            "children_count": self.children_count,
            "z_index": self.z_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuralElement":
        return cls(
            id=data.get("id", ""),
            tag=data.get("tag", ""),
            bounds=ElementBounds.from_dict(data.get("bounds", {})),
            selector=data.get("selector"),
            text_content=data.get("text_content"),
            attributes=data.get("attributes", {}),
            computed_styles=data.get("computed_styles", {}),
            children_count=data.get("children_count", 0),
            z_index=data.get("z_index", 0),
        )


@dataclass
class StructuralChange:
    """A detected structural change between snapshots."""

    change_type: StructuralChangeType
    element_id: str
    baseline_element: StructuralElement | None = None
    current_element: StructuralElement | None = None
    property_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    position_delta: tuple[int, int] | None = None
    size_delta: tuple[int, int] | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_type": self.change_type.value,
            "element_id": self.element_id,
            "baseline_element": (
                self.baseline_element.to_dict() if self.baseline_element else None
            ),
            "current_element": (
                self.current_element.to_dict() if self.current_element else None
            ),
            "property_changes": {
                k: {"baseline": v[0], "current": v[1]}
                for k, v in self.property_changes.items()
            },
            "position_delta": self.position_delta,
            "size_delta": self.size_delta,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuralChange":
        property_changes = {}
        for k, v in data.get("property_changes", {}).items():
            if isinstance(v, dict):
                property_changes[k] = (v.get("baseline"), v.get("current"))
            else:
                property_changes[k] = v

        return cls(
            change_type=StructuralChangeType(data.get("change_type", "modified")),
            element_id=data.get("element_id", ""),
            baseline_element=(
                StructuralElement.from_dict(data["baseline_element"])
                if data.get("baseline_element")
                else None
            ),
            current_element=(
                StructuralElement.from_dict(data["current_element"])
                if data.get("current_element")
                else None
            ),
            property_changes=property_changes,
            position_delta=tuple(data["position_delta"]) if data.get("position_delta") else None,
            size_delta=tuple(data["size_delta"]) if data.get("size_delta") else None,
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class LayoutRegion:
    """A region of the page for layout analysis."""

    name: str  # e.g., "header", "navigation", "main-content", "footer"
    bounds: ElementBounds
    elements: list[StructuralElement] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bounds": self.bounds.to_dict(),
            "elements": [e.to_dict() for e in self.elements],
        }


@dataclass
class StructuralDiff:
    """Complete structural diff between two snapshots.

    This is the output of structural analysis that gets passed to the
    SemanticAnalyzer for AI-powered interpretation.
    """

    # Metadata
    baseline_id: str = ""
    current_id: str = ""
    timestamp: str = ""

    # Change summary
    total_elements_baseline: int = 0
    total_elements_current: int = 0
    elements_added: int = 0
    elements_removed: int = 0
    elements_modified: int = 0
    elements_unchanged: int = 0

    # Detailed changes
    changes: list[StructuralChange] = field(default_factory=list)

    # Layout analysis
    baseline_layout_regions: list[LayoutRegion] = field(default_factory=list)
    current_layout_regions: list[LayoutRegion] = field(default_factory=list)
    layout_shift_score: float = 0.0  # CLS-like metric

    # Pixel-level diff
    pixel_diff_percentage: float = 0.0  # 0-100
    pixel_diff_regions: list[ElementBounds] = field(default_factory=list)

    # Hashes for quick comparison
    baseline_layout_hash: str = ""
    current_layout_hash: str = ""
    baseline_content_hash: str = ""
    current_content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "current_id": self.current_id,
            "timestamp": self.timestamp,
            "total_elements_baseline": self.total_elements_baseline,
            "total_elements_current": self.total_elements_current,
            "elements_added": self.elements_added,
            "elements_removed": self.elements_removed,
            "elements_modified": self.elements_modified,
            "elements_unchanged": self.elements_unchanged,
            "changes": [c.to_dict() for c in self.changes],
            "baseline_layout_regions": [r.to_dict() for r in self.baseline_layout_regions],
            "current_layout_regions": [r.to_dict() for r in self.current_layout_regions],
            "layout_shift_score": self.layout_shift_score,
            "pixel_diff_percentage": self.pixel_diff_percentage,
            "pixel_diff_regions": [r.to_dict() for r in self.pixel_diff_regions],
            "baseline_layout_hash": self.baseline_layout_hash,
            "current_layout_hash": self.current_layout_hash,
            "baseline_content_hash": self.baseline_content_hash,
            "current_content_hash": self.current_content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuralDiff":
        return cls(
            baseline_id=data.get("baseline_id", ""),
            current_id=data.get("current_id", ""),
            timestamp=data.get("timestamp", ""),
            total_elements_baseline=data.get("total_elements_baseline", 0),
            total_elements_current=data.get("total_elements_current", 0),
            elements_added=data.get("elements_added", 0),
            elements_removed=data.get("elements_removed", 0),
            elements_modified=data.get("elements_modified", 0),
            elements_unchanged=data.get("elements_unchanged", 0),
            changes=[
                StructuralChange.from_dict(c) for c in data.get("changes", [])
            ],
            baseline_layout_regions=[
                LayoutRegion(
                    name=r.get("name", ""),
                    bounds=ElementBounds.from_dict(r.get("bounds", {})),
                    elements=[
                        StructuralElement.from_dict(e) for e in r.get("elements", [])
                    ],
                )
                for r in data.get("baseline_layout_regions", [])
            ],
            current_layout_regions=[
                LayoutRegion(
                    name=r.get("name", ""),
                    bounds=ElementBounds.from_dict(r.get("bounds", {})),
                    elements=[
                        StructuralElement.from_dict(e) for e in r.get("elements", [])
                    ],
                )
                for r in data.get("current_layout_regions", [])
            ],
            layout_shift_score=data.get("layout_shift_score", 0.0),
            pixel_diff_percentage=data.get("pixel_diff_percentage", 0.0),
            pixel_diff_regions=[
                ElementBounds.from_dict(r) for r in data.get("pixel_diff_regions", [])
            ],
            baseline_layout_hash=data.get("baseline_layout_hash", ""),
            current_layout_hash=data.get("current_layout_hash", ""),
            baseline_content_hash=data.get("baseline_content_hash", ""),
            current_content_hash=data.get("current_content_hash", ""),
        )

    def has_layout_changes(self) -> bool:
        """Check if there are any layout changes."""
        return self.baseline_layout_hash != self.current_layout_hash

    def has_content_changes(self) -> bool:
        """Check if there are any content changes."""
        return self.baseline_content_hash != self.current_content_hash

    def get_significant_changes(
        self, min_confidence: float = 0.7
    ) -> list[StructuralChange]:
        """Get changes with confidence above threshold."""
        return [c for c in self.changes if c.confidence >= min_confidence]

    def get_changes_by_type(
        self, change_type: StructuralChangeType
    ) -> list[StructuralChange]:
        """Get all changes of a specific type."""
        return [c for c in self.changes if c.change_type == change_type]

    def get_summary(self) -> str:
        """Get a human-readable summary of the diff."""
        parts = []
        if self.elements_added > 0:
            parts.append(f"{self.elements_added} added")
        if self.elements_removed > 0:
            parts.append(f"{self.elements_removed} removed")
        if self.elements_modified > 0:
            parts.append(f"{self.elements_modified} modified")

        if not parts:
            return "No structural changes detected"

        summary = ", ".join(parts)
        if self.pixel_diff_percentage > 0:
            summary += f" ({self.pixel_diff_percentage:.1f}% pixel difference)"

        return summary


# =============================================================================
# New dataclasses using VisualElement from models.py
# =============================================================================


@dataclass
class VisualStructuralDiff:
    """Result of comparing DOM structures using VisualElement from models.

    This dataclass works with VisualElement objects from the visual_ai.models
    module, providing a bridge between the structural analysis and the visual
    snapshot system.
    """

    added_elements: list[VisualElement] = field(default_factory=list)
    removed_elements: list[VisualElement] = field(default_factory=list)
    moved_elements: list[tuple[VisualElement, VisualElement]] = field(
        default_factory=list
    )  # (baseline, current)
    modified_elements: list[tuple[VisualElement, VisualElement, list[str]]] = field(
        default_factory=list
    )  # (baseline, current, changed_props)
    text_changes: list[dict[str, Any]] = field(
        default_factory=list
    )  # {element, old_text, new_text}

    def has_changes(self) -> bool:
        """Check if there are any structural changes."""
        return bool(
            self.added_elements
            or self.removed_elements
            or self.moved_elements
            or self.modified_elements
            or self.text_changes
        )

    def total_changes(self) -> int:
        """Get total number of changes across all categories."""
        return (
            len(self.added_elements)
            + len(self.removed_elements)
            + len(self.moved_elements)
            + len(self.modified_elements)
            + len(self.text_changes)
        )

    def get_summary(self) -> str:
        """Generate a human-readable summary of the structural diff."""
        parts = []
        if self.added_elements:
            parts.append(f"{len(self.added_elements)} element(s) added")
        if self.removed_elements:
            parts.append(f"{len(self.removed_elements)} element(s) removed")
        if self.moved_elements:
            parts.append(f"{len(self.moved_elements)} element(s) moved")
        if self.modified_elements:
            parts.append(f"{len(self.modified_elements)} element(s) modified")
        if self.text_changes:
            parts.append(f"{len(self.text_changes)} text change(s)")

        if not parts:
            return "No structural changes detected"
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "added_elements": [el.to_dict() for el in self.added_elements],
            "removed_elements": [el.to_dict() for el in self.removed_elements],
            "moved_elements": [
                {"baseline": b.to_dict(), "current": c.to_dict()}
                for b, c in self.moved_elements
            ],
            "modified_elements": [
                {
                    "baseline": b.to_dict(),
                    "current": c.to_dict(),
                    "changed_properties": props,
                }
                for b, c, props in self.modified_elements
            ],
            "text_changes": self.text_changes,
            "summary": self.get_summary(),
            "total_changes": self.total_changes(),
        }


@dataclass
class LayoutShift:
    """Represents a layout shift for a visual element.

    Layout shifts are detected when an element's position changes between
    the baseline and current snapshot. The shift distance is calculated
    using Euclidean distance between element centers.
    """

    element: VisualElement
    delta_x: float
    delta_y: float
    delta_width: float
    delta_height: float
    shift_distance: float

    def is_significant(self, threshold: float = 5.0) -> bool:
        """Check if the layout shift is significant (exceeds threshold in pixels)."""
        return self.shift_distance >= threshold

    def get_shift_direction(self) -> str:
        """Determine the primary direction of the shift."""
        directions = []

        if abs(self.delta_x) > abs(self.delta_y):
            if self.delta_x > 0:
                directions.append("right")
            elif self.delta_x < 0:
                directions.append("left")
        else:
            if self.delta_y > 0:
                directions.append("down")
            elif self.delta_y < 0:
                directions.append("up")

        if self.delta_width > 0:
            directions.append("expanded horizontally")
        elif self.delta_width < 0:
            directions.append("contracted horizontally")

        if self.delta_height > 0:
            directions.append("expanded vertically")
        elif self.delta_height < 0:
            directions.append("contracted vertically")

        return ", ".join(directions) if directions else "no movement"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "element": self.element.to_dict(),
            "delta_x": self.delta_x,
            "delta_y": self.delta_y,
            "delta_width": self.delta_width,
            "delta_height": self.delta_height,
            "shift_distance": self.shift_distance,
            "is_significant": self.is_significant(),
            "direction": self.get_shift_direction(),
        }


class DOMTreeParser(HTMLParser):
    """Simple HTML parser to extract DOM structure for analysis."""

    def __init__(self):
        super().__init__()
        self.elements: list[dict[str, Any]] = []
        self._stack: list[dict[str, Any]] = []
        self._current_text: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        element = {
            "tag": tag,
            "attributes": dict(attrs),
            "children": [],
            "text": "",
            "depth": len(self._stack),
        }
        if self._stack:
            self._stack[-1]["children"].append(element)
        else:
            self.elements.append(element)
        self._stack.append(element)

    def handle_endtag(self, tag: str) -> None:
        if self._stack and self._stack[-1]["tag"] == tag:
            # Combine collected text
            self._stack[-1]["text"] = " ".join(self._current_text).strip()
            self._current_text = []
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._current_text.append(text)


class StructuralAnalyzer:
    """Analyzes DOM structure changes between snapshots.

    This analyzer performs DOM-level diffing to understand WHAT changed
    structurally between a baseline and current snapshot. It uses multiple
    matching strategies:
    1. Match by element_id (exact match)
    2. Match by CSS selector (exact match)
    3. Match by position proximity (fuzzy match)

    The analyzer works with VisualElement objects from the visual_ai.models
    module.
    """

    def __init__(
        self,
        position_threshold: float = 50.0,
        size_threshold: float = 20.0,
        text_similarity_threshold: float = 0.8,
    ):
        """Initialize the structural analyzer.

        Args:
            position_threshold: Maximum distance in pixels to consider
                               elements as potentially matched by position.
            size_threshold: Maximum size difference in pixels for dimension matching.
            text_similarity_threshold: Minimum ratio for text content matching (0-1).
        """
        self.position_threshold = position_threshold
        self.size_threshold = size_threshold
        self.text_similarity_threshold = text_similarity_threshold

    async def compare_structure(
        self,
        baseline_dom: str,
        current_dom: str,
    ) -> VisualStructuralDiff:
        """Compare DOM trees and return structural differences.

        Args:
            baseline_dom: HTML string of the baseline DOM.
            current_dom: HTML string of the current DOM.

        Returns:
            VisualStructuralDiff containing all detected structural changes.
        """
        # Parse both DOMs
        baseline_elements = self._parse_dom_to_elements(baseline_dom)
        current_elements = self._parse_dom_to_elements(current_dom)

        # Match elements between snapshots
        matches, unmatched_baseline, unmatched_current = self._match_elements(
            baseline_elements, current_elements
        )

        diff = VisualStructuralDiff()

        # Elements only in current are "added"
        diff.added_elements = unmatched_current

        # Elements only in baseline are "removed"
        diff.removed_elements = unmatched_baseline

        # Analyze matched elements for modifications and movements
        for baseline_el, current_el in matches:
            # Check for position changes (moves)
            position_delta = self._calculate_position_delta(baseline_el, current_el)
            if position_delta > self.position_threshold / 2:
                diff.moved_elements.append((baseline_el, current_el))

            # Check for property modifications
            changed_props = self._get_changed_properties(baseline_el, current_el)
            if changed_props:
                diff.modified_elements.append((baseline_el, current_el, changed_props))

            # Check for text content changes
            if (
                baseline_el.text_content is not None
                and current_el.text_content is not None
            ):
                if baseline_el.text_content != current_el.text_content:
                    diff.text_changes.append(
                        {
                            "element_selector": current_el.selector,
                            "element_id": current_el.element_id,
                            "old_text": baseline_el.text_content,
                            "new_text": current_el.text_content,
                            "similarity": self._text_similarity(
                                baseline_el.text_content, current_el.text_content
                            ),
                        }
                    )

        return diff

    async def detect_layout_shifts(
        self,
        baseline_elements: list[VisualElement],
        current_elements: list[VisualElement],
    ) -> list[LayoutShift]:
        """Detect elements that moved position between snapshots.

        Uses Euclidean distance to calculate the shift distance between
        element centers.

        Args:
            baseline_elements: List of elements from baseline snapshot.
            current_elements: List of elements from current snapshot.

        Returns:
            List of LayoutShift objects for elements that moved.
        """
        shifts: list[LayoutShift] = []

        # Match elements
        matches, _, _ = self._match_elements(baseline_elements, current_elements)

        for baseline_el, current_el in matches:
            delta_x = current_el.bounds["x"] - baseline_el.bounds["x"]
            delta_y = current_el.bounds["y"] - baseline_el.bounds["y"]
            delta_width = current_el.bounds["width"] - baseline_el.bounds["width"]
            delta_height = current_el.bounds["height"] - baseline_el.bounds["height"]

            # Calculate Euclidean distance for position shift
            shift_distance = math.sqrt(delta_x**2 + delta_y**2)

            # Only include if there's any change
            if shift_distance > 0 or delta_width != 0 or delta_height != 0:
                shift = LayoutShift(
                    element=current_el,
                    delta_x=delta_x,
                    delta_y=delta_y,
                    delta_width=delta_width,
                    delta_height=delta_height,
                    shift_distance=shift_distance,
                )
                shifts.append(shift)

        # Sort by shift distance (most significant first)
        shifts.sort(key=lambda s: s.shift_distance, reverse=True)

        return shifts

    async def track_component_changes(
        self,
        baseline: VisualSnapshot,
        current: VisualSnapshot,
        component_selectors: list[str],
    ) -> dict[str, Any]:
        """Track specific components across versions.

        Args:
            baseline: Baseline visual snapshot.
            current: Current visual snapshot.
            component_selectors: List of CSS selectors for components to track.

        Returns:
            Dictionary mapping selectors to their change information.
        """
        results: dict[str, Any] = {}

        for selector in component_selectors:
            baseline_el = baseline.get_element_by_selector(selector)
            current_el = current.get_element_by_selector(selector)

            component_result: dict[str, Any] = {
                "selector": selector,
                "found_in_baseline": baseline_el is not None,
                "found_in_current": current_el is not None,
                "status": "unchanged",
                "changes": [],
            }

            if baseline_el is None and current_el is None:
                component_result["status"] = "not_found"
            elif baseline_el is None and current_el is not None:
                component_result["status"] = "added"
                component_result["current_bounds"] = current_el.bounds
            elif baseline_el is not None and current_el is None:
                component_result["status"] = "removed"
                component_result["baseline_bounds"] = baseline_el.bounds
            else:
                # Both exist - compare them
                changes = self._get_changed_properties(baseline_el, current_el)
                if changes:
                    component_result["status"] = "modified"
                    component_result["changes"] = changes

                # Check for position changes
                delta = self._calculate_position_delta(baseline_el, current_el)
                if delta > 0:
                    component_result["position_delta"] = delta
                    if "position" not in changes:
                        component_result["changes"].append("position")
                    if component_result["status"] == "unchanged":
                        component_result["status"] = "moved"

                # Check text changes
                if baseline_el.text_content != current_el.text_content:
                    component_result["text_changed"] = True
                    component_result["old_text"] = baseline_el.text_content
                    component_result["new_text"] = current_el.text_content
                    if "text_content" not in changes:
                        component_result["changes"].append("text_content")
                    if component_result["status"] == "unchanged":
                        component_result["status"] = "modified"

                component_result["baseline_bounds"] = baseline_el.bounds
                component_result["current_bounds"] = current_el.bounds

            results[selector] = component_result

        return results

    def _match_elements(
        self,
        baseline_elements: list[VisualElement],
        current_elements: list[VisualElement],
    ) -> tuple[
        list[tuple[VisualElement, VisualElement]],
        list[VisualElement],
        list[VisualElement],
    ]:
        """Match elements between snapshots using ID, selector, position.

        Uses a three-stage matching strategy:
        1. Match by element_id (exact)
        2. Match by selector (exact)
        3. Match by position proximity (fuzzy)

        Args:
            baseline_elements: Elements from baseline snapshot.
            current_elements: Elements from current snapshot.

        Returns:
            Tuple of (matched_pairs, unmatched_baseline, unmatched_current)
        """
        matches: list[tuple[VisualElement, VisualElement]] = []
        unmatched_baseline = list(baseline_elements)
        unmatched_current = list(current_elements)

        # Stage 1: Match by element_id
        baseline_by_id = {
            el.element_id: el for el in unmatched_baseline if el.element_id
        }
        for current_el in list(unmatched_current):
            if current_el.element_id and current_el.element_id in baseline_by_id:
                baseline_el = baseline_by_id[current_el.element_id]
                matches.append((baseline_el, current_el))
                unmatched_baseline.remove(baseline_el)
                unmatched_current.remove(current_el)

        # Stage 2: Match by selector
        baseline_by_selector = {
            el.selector: el for el in unmatched_baseline if el.selector
        }
        for current_el in list(unmatched_current):
            if current_el.selector and current_el.selector in baseline_by_selector:
                baseline_el = baseline_by_selector[current_el.selector]
                matches.append((baseline_el, current_el))
                unmatched_baseline.remove(baseline_el)
                unmatched_current.remove(current_el)

        # Stage 3: Match by position proximity (for remaining elements)
        for current_el in list(unmatched_current):
            best_match: VisualElement | None = None
            best_distance = float("inf")

            for baseline_el in unmatched_baseline:
                # Must be same tag type
                if baseline_el.tag_name != current_el.tag_name:
                    continue

                # Calculate distance between center points
                distance = self._calculate_position_delta(baseline_el, current_el)

                # Check if within threshold and better than previous best
                if distance < self.position_threshold and distance < best_distance:
                    # Additional check: text content similarity (if both have text)
                    if baseline_el.text_content and current_el.text_content:
                        similarity = self._text_similarity(
                            baseline_el.text_content, current_el.text_content
                        )
                        if similarity < self.text_similarity_threshold:
                            continue

                    best_match = baseline_el
                    best_distance = distance

            if best_match is not None:
                matches.append((best_match, current_el))
                unmatched_baseline.remove(best_match)
                unmatched_current.remove(current_el)

        return matches, unmatched_baseline, unmatched_current

    def _calculate_position_delta(
        self, baseline: VisualElement, current: VisualElement
    ) -> float:
        """Calculate the Euclidean distance between element center points."""
        baseline_center = baseline.get_center()
        current_center = current.get_center()

        delta_x = current_center[0] - baseline_center[0]
        delta_y = current_center[1] - baseline_center[1]

        return math.sqrt(delta_x**2 + delta_y**2)

    def _get_changed_properties(
        self, baseline: VisualElement, current: VisualElement
    ) -> list[str]:
        """Get list of properties that changed between two elements."""
        changed: list[str] = []

        # Check bounds
        if baseline.bounds != current.bounds:
            if baseline.bounds["x"] != current.bounds["x"]:
                changed.append("x")
            if baseline.bounds["y"] != current.bounds["y"]:
                changed.append("y")
            if baseline.bounds["width"] != current.bounds["width"]:
                changed.append("width")
            if baseline.bounds["height"] != current.bounds["height"]:
                changed.append("height")

        # Check tag name
        if baseline.tag_name != current.tag_name:
            changed.append("tag_name")

        # Check attributes
        baseline_attrs = set(baseline.attributes.items())
        current_attrs = set(current.attributes.items())
        if baseline_attrs != current_attrs:
            # Determine which attributes changed
            added_attrs = current_attrs - baseline_attrs
            removed_attrs = baseline_attrs - current_attrs
            for attr_name, _ in added_attrs:
                changed.append(f"attr:{attr_name}")
            for attr_name, _ in removed_attrs:
                if not any(attr_name == a[0] for a in added_attrs):
                    changed.append(f"attr:{attr_name}")

        # Check computed styles (compare key styles)
        key_styles = [
            "display",
            "visibility",
            "opacity",
            "position",
            "color",
            "background-color",
            "font-size",
            "font-weight",
            "margin",
            "padding",
            "border",
        ]
        for style in key_styles:
            baseline_value = baseline.computed_styles.get(style)
            current_value = current.computed_styles.get(style)
            if baseline_value != current_value:
                changed.append(f"style:{style}")

        # Check children count
        if baseline.children_count != current.children_count:
            changed.append("children_count")

        return changed

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio using difflib."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def _parse_dom_to_elements(self, dom_html: str) -> list[VisualElement]:
        """Parse HTML DOM string into a list of VisualElement objects.

        Note: This is a simplified parser that creates elements from the DOM.
        In production, you would use the elements from VisualSnapshot which
        have accurate bounds from the browser.
        """
        elements: list[VisualElement] = []

        try:
            parser = DOMTreeParser()
            parser.feed(dom_html)

            # Convert parsed elements to VisualElements
            element_counter = 0
            for parsed in self._flatten_parsed_elements(parser.elements):
                element_counter += 1

                # Generate selector from attributes
                selector = self._generate_selector(parsed)

                element = VisualElement(
                    element_id=parsed["attributes"].get(
                        "id", f"auto_{element_counter}"
                    ),
                    selector=selector,
                    tag_name=parsed["tag"],
                    bounds={
                        "x": 0.0,
                        "y": 0.0,
                        "width": 0.0,
                        "height": 0.0,
                    },  # Bounds not available from HTML parsing
                    computed_styles={},
                    text_content=parsed.get("text", ""),
                    attributes=parsed["attributes"],
                    children_count=len(parsed.get("children", [])),
                )
                elements.append(element)
        except Exception as e:
            logger.warning(f"Error parsing DOM: {e}")

        return elements

    def _flatten_parsed_elements(
        self, elements: list[dict[str, Any]], parent_selector: str = ""
    ) -> list[dict[str, Any]]:
        """Flatten nested element structure into a flat list."""
        result: list[dict[str, Any]] = []

        for i, el in enumerate(elements):
            # Build path-based selector
            tag = el["tag"]
            el_id = el["attributes"].get("id", "")
            el_class = el["attributes"].get("class", "")

            if el_id:
                current_selector = f"#{el_id}"
            elif el_class:
                classes = el_class.split()
                current_selector = f"{tag}.{'.'.join(classes)}"
            else:
                current_selector = f"{parent_selector} > {tag}:nth-child({i + 1})"

            el["_selector"] = current_selector.strip()
            result.append(el)

            # Recursively process children
            if el.get("children"):
                result.extend(
                    self._flatten_parsed_elements(el["children"], current_selector)
                )

        return result

    def _generate_selector(self, parsed_element: dict[str, Any]) -> str:
        """Generate a CSS selector for a parsed element."""
        if "_selector" in parsed_element:
            return parsed_element["_selector"]

        tag = parsed_element["tag"]
        attrs = parsed_element["attributes"]

        # Prefer id
        if "id" in attrs:
            return f"#{attrs['id']}"

        # Then data-testid
        if "data-testid" in attrs:
            return f"[data-testid='{attrs['data-testid']}']"

        # Then class
        if "class" in attrs:
            classes = attrs["class"].split()
            if classes:
                return f"{tag}.{'.'.join(classes)}"

        return tag

    def generate_diff_report(
        self,
        diff: VisualStructuralDiff,
        baseline_dom: str,
        current_dom: str,
    ) -> dict[str, Any]:
        """Generate a comprehensive diff report with line-by-line changes.

        Args:
            diff: The structural diff result.
            baseline_dom: Original DOM HTML.
            current_dom: Current DOM HTML.

        Returns:
            Dictionary containing detailed diff report.
        """
        # Generate unified diff of the DOM strings
        baseline_lines = baseline_dom.splitlines(keepends=True)
        current_lines = current_dom.splitlines(keepends=True)

        unified_diff = list(
            difflib.unified_diff(
                baseline_lines,
                current_lines,
                fromfile="baseline",
                tofile="current",
                lineterm="",
            )
        )

        # Generate HTML diff for visualization
        html_diff = difflib.HtmlDiff().make_table(
            baseline_lines,
            current_lines,
            fromdesc="Baseline",
            todesc="Current",
            context=True,
            numlines=3,
        )

        return {
            "structural_diff": diff.to_dict(),
            "unified_diff": "".join(unified_diff),
            "html_diff": html_diff,
            "baseline_line_count": len(baseline_lines),
            "current_line_count": len(current_lines),
            "line_delta": len(current_lines) - len(baseline_lines),
        }

    def calculate_cumulative_layout_shift(
        self, shifts: list[LayoutShift], viewport_height: float = 800.0
    ) -> float:
        """Calculate the Cumulative Layout Shift (CLS) score.

        CLS is a Core Web Vital that measures visual stability.
        It's calculated as the sum of impact fractions * distance fractions.

        Args:
            shifts: List of layout shifts detected.
            viewport_height: Height of the viewport in pixels.

        Returns:
            CLS score (lower is better, < 0.1 is good, < 0.25 needs improvement).
        """
        if not shifts:
            return 0.0

        cls_score = 0.0

        for shift in shifts:
            if not shift.is_significant():
                continue

            element = shift.element
            area = element.get_area()

            # Skip elements with no area
            if area <= 0:
                continue

            # Impact fraction: portion of viewport affected by unstable element
            viewport_area = viewport_height * (viewport_height * 16 / 9)  # Assume 16:9
            impact_fraction = min(area / viewport_area, 1.0)

            # Distance fraction: how far the element moved
            max_dimension = max(viewport_height, viewport_height * 16 / 9)
            distance_fraction = min(shift.shift_distance / max_dimension, 1.0)

            # Layout shift score for this element
            element_score = impact_fraction * distance_fraction
            cls_score += element_score

        return cls_score

    def convert_to_structural_diff(
        self, visual_diff: VisualStructuralDiff
    ) -> StructuralDiff:
        """Convert a VisualStructuralDiff to the existing StructuralDiff format.

        This method provides interoperability between the new VisualElement-based
        diff and the existing StructuralDiff format used elsewhere in the system.

        Args:
            visual_diff: The VisualStructuralDiff to convert.

        Returns:
            A StructuralDiff with equivalent data.
        """
        changes: list[StructuralChange] = []

        # Convert added elements
        for el in visual_diff.added_elements:
            changes.append(
                StructuralChange(
                    change_type=StructuralChangeType.ADDED,
                    element_id=el.element_id,
                    current_element=StructuralElement(
                        id=el.element_id,
                        tag=el.tag_name,
                        bounds=ElementBounds(
                            x=int(el.bounds["x"]),
                            y=int(el.bounds["y"]),
                            width=int(el.bounds["width"]),
                            height=int(el.bounds["height"]),
                        ),
                        selector=el.selector,
                        text_content=el.text_content,
                        attributes=el.attributes,
                        computed_styles=el.computed_styles,
                        children_count=el.children_count,
                    ),
                )
            )

        # Convert removed elements
        for el in visual_diff.removed_elements:
            changes.append(
                StructuralChange(
                    change_type=StructuralChangeType.REMOVED,
                    element_id=el.element_id,
                    baseline_element=StructuralElement(
                        id=el.element_id,
                        tag=el.tag_name,
                        bounds=ElementBounds(
                            x=int(el.bounds["x"]),
                            y=int(el.bounds["y"]),
                            width=int(el.bounds["width"]),
                            height=int(el.bounds["height"]),
                        ),
                        selector=el.selector,
                        text_content=el.text_content,
                        attributes=el.attributes,
                        computed_styles=el.computed_styles,
                        children_count=el.children_count,
                    ),
                )
            )

        # Convert moved elements
        for baseline_el, current_el in visual_diff.moved_elements:
            delta_x = int(current_el.bounds["x"] - baseline_el.bounds["x"])
            delta_y = int(current_el.bounds["y"] - baseline_el.bounds["y"])
            changes.append(
                StructuralChange(
                    change_type=StructuralChangeType.MOVED,
                    element_id=current_el.element_id,
                    baseline_element=StructuralElement(
                        id=baseline_el.element_id,
                        tag=baseline_el.tag_name,
                        bounds=ElementBounds(
                            x=int(baseline_el.bounds["x"]),
                            y=int(baseline_el.bounds["y"]),
                            width=int(baseline_el.bounds["width"]),
                            height=int(baseline_el.bounds["height"]),
                        ),
                        selector=baseline_el.selector,
                        text_content=baseline_el.text_content,
                        attributes=baseline_el.attributes,
                        computed_styles=baseline_el.computed_styles,
                        children_count=baseline_el.children_count,
                    ),
                    current_element=StructuralElement(
                        id=current_el.element_id,
                        tag=current_el.tag_name,
                        bounds=ElementBounds(
                            x=int(current_el.bounds["x"]),
                            y=int(current_el.bounds["y"]),
                            width=int(current_el.bounds["width"]),
                            height=int(current_el.bounds["height"]),
                        ),
                        selector=current_el.selector,
                        text_content=current_el.text_content,
                        attributes=current_el.attributes,
                        computed_styles=current_el.computed_styles,
                        children_count=current_el.children_count,
                    ),
                    position_delta=(delta_x, delta_y),
                )
            )

        # Convert modified elements
        for baseline_el, current_el, changed_props in visual_diff.modified_elements:
            property_changes = {}
            for prop in changed_props:
                if prop.startswith("style:"):
                    style_name = prop[6:]
                    property_changes[prop] = (
                        baseline_el.computed_styles.get(style_name),
                        current_el.computed_styles.get(style_name),
                    )
                elif prop.startswith("attr:"):
                    attr_name = prop[5:]
                    property_changes[prop] = (
                        baseline_el.attributes.get(attr_name),
                        current_el.attributes.get(attr_name),
                    )
                elif prop in ("x", "y", "width", "height"):
                    property_changes[prop] = (
                        baseline_el.bounds.get(prop),
                        current_el.bounds.get(prop),
                    )

            changes.append(
                StructuralChange(
                    change_type=StructuralChangeType.MODIFIED,
                    element_id=current_el.element_id,
                    baseline_element=StructuralElement(
                        id=baseline_el.element_id,
                        tag=baseline_el.tag_name,
                        bounds=ElementBounds(
                            x=int(baseline_el.bounds["x"]),
                            y=int(baseline_el.bounds["y"]),
                            width=int(baseline_el.bounds["width"]),
                            height=int(baseline_el.bounds["height"]),
                        ),
                        selector=baseline_el.selector,
                        text_content=baseline_el.text_content,
                        attributes=baseline_el.attributes,
                        computed_styles=baseline_el.computed_styles,
                        children_count=baseline_el.children_count,
                    ),
                    current_element=StructuralElement(
                        id=current_el.element_id,
                        tag=current_el.tag_name,
                        bounds=ElementBounds(
                            x=int(current_el.bounds["x"]),
                            y=int(current_el.bounds["y"]),
                            width=int(current_el.bounds["width"]),
                            height=int(current_el.bounds["height"]),
                        ),
                        selector=current_el.selector,
                        text_content=current_el.text_content,
                        attributes=current_el.attributes,
                        computed_styles=current_el.computed_styles,
                        children_count=current_el.children_count,
                    ),
                    property_changes=property_changes,
                )
            )

        return StructuralDiff(
            elements_added=len(visual_diff.added_elements),
            elements_removed=len(visual_diff.removed_elements),
            elements_modified=len(visual_diff.modified_elements)
            + len(visual_diff.moved_elements),
            changes=changes,
        )
