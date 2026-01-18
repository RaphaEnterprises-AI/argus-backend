"""Comprehensive tests for visual_ai/structural_analyzer.py.

Tests structural analysis including DOM diffing, element matching,
layout shift detection, and structural change tracking.
"""

import pytest
from typing import Dict, List, Any
import math

from src.visual_ai.structural_analyzer import (
    StructuralChangeType,
    ElementBounds,
    StructuralElement,
    StructuralChange,
    LayoutRegion,
    StructuralDiff,
    VisualStructuralDiff,
    LayoutShift,
    DOMTreeParser,
    StructuralAnalyzer,
)
from src.visual_ai.models import VisualElement, VisualSnapshot


class TestStructuralChangeType:
    """Tests for StructuralChangeType enum."""

    def test_all_change_types_exist(self):
        """Test that all expected change types exist."""
        expected = ["added", "removed", "moved", "resized", "modified", "unchanged"]
        for change_type in expected:
            assert hasattr(StructuralChangeType, change_type.upper())

    def test_change_type_values(self):
        """Test enum values are strings."""
        assert StructuralChangeType.ADDED.value == "added"
        assert StructuralChangeType.REMOVED.value == "removed"
        assert StructuralChangeType.MOVED.value == "moved"
        assert StructuralChangeType.RESIZED.value == "resized"
        assert StructuralChangeType.MODIFIED.value == "modified"
        assert StructuralChangeType.UNCHANGED.value == "unchanged"


class TestElementBounds:
    """Tests for ElementBounds dataclass."""

    def test_element_bounds_creation(self):
        """Test creating ElementBounds instance."""
        bounds = ElementBounds(x=100, y=200, width=300, height=150)
        assert bounds.x == 100
        assert bounds.y == 200
        assert bounds.width == 300
        assert bounds.height == 150

    def test_center_property(self):
        """Test center calculation."""
        bounds = ElementBounds(x=100, y=100, width=200, height=100)
        center = bounds.center
        assert center == (200, 150)  # (100+200/2, 100+100/2)

    def test_area_property(self):
        """Test area calculation."""
        bounds = ElementBounds(x=0, y=0, width=100, height=50)
        assert bounds.area == 5000

    def test_to_dict(self):
        """Test to_dict serialization."""
        bounds = ElementBounds(x=10, y=20, width=30, height=40)
        result = bounds.to_dict()
        assert result == {"x": 10, "y": 20, "width": 30, "height": 40}

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {"x": 50, "y": 60, "width": 70, "height": 80}
        bounds = ElementBounds.from_dict(data)
        assert bounds.x == 50
        assert bounds.y == 60
        assert bounds.width == 70
        assert bounds.height == 80

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing keys uses defaults."""
        bounds = ElementBounds.from_dict({})
        assert bounds.x == 0
        assert bounds.y == 0
        assert bounds.width == 0
        assert bounds.height == 0

    def test_distance_to(self):
        """Test distance calculation between bounds."""
        bounds1 = ElementBounds(x=0, y=0, width=100, height=100)
        bounds2 = ElementBounds(x=100, y=100, width=100, height=100)
        distance = bounds1.distance_to(bounds2)
        # Centers: (50,50) and (150,150)
        expected = math.sqrt(100**2 + 100**2)
        assert abs(distance - expected) < 0.01

    def test_overlap_ratio_no_overlap(self):
        """Test overlap ratio for non-overlapping bounds."""
        bounds1 = ElementBounds(x=0, y=0, width=50, height=50)
        bounds2 = ElementBounds(x=100, y=100, width=50, height=50)
        ratio = bounds1.overlap_ratio(bounds2)
        assert ratio == 0.0

    def test_overlap_ratio_full_overlap(self):
        """Test overlap ratio for identical bounds."""
        bounds1 = ElementBounds(x=0, y=0, width=100, height=100)
        bounds2 = ElementBounds(x=0, y=0, width=100, height=100)
        ratio = bounds1.overlap_ratio(bounds2)
        assert ratio == 1.0

    def test_overlap_ratio_partial(self):
        """Test overlap ratio for partially overlapping bounds."""
        bounds1 = ElementBounds(x=0, y=0, width=100, height=100)
        bounds2 = ElementBounds(x=50, y=50, width=100, height=100)
        ratio = bounds1.overlap_ratio(bounds2)
        # Overlap area: 50x50 = 2500
        # Total area: 10000 + 10000 - 2500 = 17500
        # Ratio: 2500/17500 ~ 0.143
        assert 0.1 < ratio < 0.2

    def test_overlap_ratio_zero_area(self):
        """Test overlap ratio when area is zero."""
        bounds1 = ElementBounds(x=0, y=0, width=0, height=0)
        bounds2 = ElementBounds(x=0, y=0, width=0, height=0)
        ratio = bounds1.overlap_ratio(bounds2)
        assert ratio == 0.0


class TestStructuralElement:
    """Tests for StructuralElement dataclass."""

    @pytest.fixture
    def sample_element(self):
        """Create a sample StructuralElement."""
        return StructuralElement(
            id="el_001",
            tag="button",
            bounds=ElementBounds(x=100, y=100, width=200, height=50),
            selector="#submit-btn",
            text_content="Submit",
            attributes={"class": "btn primary", "type": "submit"},
            computed_styles={"color": "white", "background": "blue"},
            children_count=0,
            z_index=1,
        )

    def test_to_dict(self, sample_element):
        """Test to_dict serialization."""
        result = sample_element.to_dict()
        assert result["id"] == "el_001"
        assert result["tag"] == "button"
        assert result["bounds"]["x"] == 100
        assert result["selector"] == "#submit-btn"
        assert result["text_content"] == "Submit"
        assert "class" in result["attributes"]

    def test_from_dict(self, sample_element):
        """Test from_dict deserialization."""
        data = sample_element.to_dict()
        recreated = StructuralElement.from_dict(data)
        assert recreated.id == sample_element.id
        assert recreated.tag == sample_element.tag
        assert recreated.bounds.x == sample_element.bounds.x


class TestStructuralChange:
    """Tests for StructuralChange dataclass."""

    @pytest.fixture
    def sample_change(self):
        """Create a sample StructuralChange."""
        baseline = StructuralElement(
            id="el_1",
            tag="div",
            bounds=ElementBounds(x=0, y=0, width=100, height=100),
            selector=".box",
            text_content="Hello",
            attributes={},
            computed_styles={},
            children_count=0,
            z_index=0,
        )
        current = StructuralElement(
            id="el_1",
            tag="div",
            bounds=ElementBounds(x=50, y=50, width=100, height=100),
            selector=".box",
            text_content="Hello World",
            attributes={},
            computed_styles={},
            children_count=0,
            z_index=0,
        )
        return StructuralChange(
            change_type=StructuralChangeType.MOVED,
            element_id="el_1",
            baseline_element=baseline,
            current_element=current,
            property_changes={"text": ("Hello", "Hello World")},
            position_delta=(50, 50),
            size_delta=None,
            confidence=0.95,
        )

    def test_to_dict(self, sample_change):
        """Test to_dict serialization."""
        result = sample_change.to_dict()
        assert result["change_type"] == "moved"
        assert result["element_id"] == "el_1"
        assert result["position_delta"] == (50, 50)
        assert result["confidence"] == 0.95
        assert result["property_changes"]["text"]["baseline"] == "Hello"
        assert result["property_changes"]["text"]["current"] == "Hello World"

    def test_from_dict(self, sample_change):
        """Test from_dict deserialization."""
        data = sample_change.to_dict()
        recreated = StructuralChange.from_dict(data)
        assert recreated.change_type == StructuralChangeType.MOVED
        assert recreated.element_id == "el_1"
        assert recreated.position_delta == (50, 50)
        assert recreated.property_changes["text"] == ("Hello", "Hello World")


class TestLayoutRegion:
    """Tests for LayoutRegion dataclass."""

    def test_layout_region_creation(self):
        """Test creating a LayoutRegion."""
        region = LayoutRegion(
            name="header",
            bounds=ElementBounds(x=0, y=0, width=1920, height=80),
            elements=[],
        )
        assert region.name == "header"
        assert region.bounds.height == 80

    def test_to_dict(self):
        """Test to_dict serialization."""
        element = StructuralElement(
            id="logo",
            tag="img",
            bounds=ElementBounds(x=20, y=10, width=100, height=60),
            selector="#logo",
            text_content=None,
            attributes={"alt": "Logo"},
            computed_styles={},
            children_count=0,
            z_index=1,
        )
        region = LayoutRegion(
            name="navigation",
            bounds=ElementBounds(x=0, y=0, width=1920, height=80),
            elements=[element],
        )
        result = region.to_dict()
        assert result["name"] == "navigation"
        assert len(result["elements"]) == 1
        assert result["elements"][0]["id"] == "logo"


class TestStructuralDiff:
    """Tests for StructuralDiff dataclass."""

    @pytest.fixture
    def sample_diff(self):
        """Create a sample StructuralDiff."""
        change = StructuralChange(
            change_type=StructuralChangeType.MODIFIED,
            element_id="el_1",
            baseline_element=None,
            current_element=None,
            property_changes={},
            position_delta=None,
            size_delta=None,
            confidence=0.9,
        )
        return StructuralDiff(
            baseline_id="base_1",
            current_id="current_1",
            timestamp="2024-01-01T12:00:00Z",
            total_elements_baseline=100,
            total_elements_current=102,
            elements_added=5,
            elements_removed=3,
            elements_modified=10,
            elements_unchanged=85,
            changes=[change],
            baseline_layout_regions=[],
            current_layout_regions=[],
            layout_shift_score=0.05,
            pixel_diff_percentage=2.5,
            pixel_diff_regions=[],
            baseline_layout_hash="abc123",
            current_layout_hash="def456",
            baseline_content_hash="hash1",
            current_content_hash="hash2",
        )

    def test_to_dict(self, sample_diff):
        """Test to_dict serialization."""
        result = sample_diff.to_dict()
        assert result["baseline_id"] == "base_1"
        assert result["elements_added"] == 5
        assert result["layout_shift_score"] == 0.05
        assert len(result["changes"]) == 1

    def test_from_dict(self, sample_diff):
        """Test from_dict deserialization."""
        data = sample_diff.to_dict()
        recreated = StructuralDiff.from_dict(data)
        assert recreated.baseline_id == sample_diff.baseline_id
        assert recreated.elements_added == sample_diff.elements_added
        assert len(recreated.changes) == 1

    def test_has_layout_changes(self, sample_diff):
        """Test has_layout_changes detection."""
        assert sample_diff.has_layout_changes() is True

    def test_has_layout_changes_false(self, sample_diff):
        """Test has_layout_changes when hashes match."""
        sample_diff.baseline_layout_hash = "same_hash"
        sample_diff.current_layout_hash = "same_hash"
        assert sample_diff.has_layout_changes() is False

    def test_has_content_changes(self, sample_diff):
        """Test has_content_changes detection."""
        assert sample_diff.has_content_changes() is True

    def test_get_significant_changes(self, sample_diff):
        """Test get_significant_changes filtering."""
        significant = sample_diff.get_significant_changes(min_confidence=0.8)
        assert len(significant) == 1

        significant = sample_diff.get_significant_changes(min_confidence=0.95)
        assert len(significant) == 0

    def test_get_changes_by_type(self, sample_diff):
        """Test get_changes_by_type filtering."""
        modified = sample_diff.get_changes_by_type(StructuralChangeType.MODIFIED)
        assert len(modified) == 1

        added = sample_diff.get_changes_by_type(StructuralChangeType.ADDED)
        assert len(added) == 0

    def test_get_summary(self, sample_diff):
        """Test get_summary generation."""
        summary = sample_diff.get_summary()
        assert "5 added" in summary
        assert "3 removed" in summary
        assert "10 modified" in summary
        assert "2.5% pixel difference" in summary

    def test_get_summary_no_changes(self):
        """Test get_summary when no changes."""
        diff = StructuralDiff()
        summary = diff.get_summary()
        assert "No structural changes" in summary


class TestVisualStructuralDiff:
    """Tests for VisualStructuralDiff dataclass."""

    @pytest.fixture
    def sample_elements(self):
        """Create sample VisualElements."""
        return [
            VisualElement(
                element_id="el_1",
                selector="#header",
                tag_name="header",
                bounds={"x": 0.0, "y": 0.0, "width": 800.0, "height": 60.0},
                computed_styles={},
                text_content="Header",
                attributes={},
                children_count=2,
            ),
            VisualElement(
                element_id="el_2",
                selector="#footer",
                tag_name="footer",
                bounds={"x": 0.0, "y": 700.0, "width": 800.0, "height": 60.0},
                computed_styles={},
                text_content="Footer",
                attributes={},
                children_count=3,
            ),
        ]

    def test_has_changes_true(self, sample_elements):
        """Test has_changes when there are changes."""
        diff = VisualStructuralDiff(
            added_elements=sample_elements,
            removed_elements=[],
            moved_elements=[],
            modified_elements=[],
            text_changes=[],
        )
        assert diff.has_changes() is True

    def test_has_changes_false(self):
        """Test has_changes when empty."""
        diff = VisualStructuralDiff()
        assert diff.has_changes() is False

    def test_total_changes(self, sample_elements):
        """Test total_changes calculation."""
        diff = VisualStructuralDiff(
            added_elements=sample_elements,
            removed_elements=[sample_elements[0]],
            moved_elements=[],
            modified_elements=[],
            text_changes=[{"element_id": "x", "old": "a", "new": "b"}],
        )
        assert diff.total_changes() == 4

    def test_get_summary(self, sample_elements):
        """Test get_summary generation."""
        diff = VisualStructuralDiff(
            added_elements=sample_elements,
            removed_elements=[],
            moved_elements=[(sample_elements[0], sample_elements[1])],
            modified_elements=[],
            text_changes=[],
        )
        summary = diff.get_summary()
        assert "2 element(s) added" in summary
        assert "1 element(s) moved" in summary

    def test_get_summary_no_changes(self):
        """Test get_summary when no changes."""
        diff = VisualStructuralDiff()
        summary = diff.get_summary()
        assert "No structural changes" in summary

    def test_to_dict(self, sample_elements):
        """Test to_dict serialization."""
        diff = VisualStructuralDiff(
            added_elements=[sample_elements[0]],
            removed_elements=[],
            moved_elements=[],
            modified_elements=[],
            text_changes=[],
        )
        result = diff.to_dict()
        assert len(result["added_elements"]) == 1
        assert "summary" in result
        assert result["total_changes"] == 1


class TestLayoutShift:
    """Tests for LayoutShift dataclass."""

    @pytest.fixture
    def sample_element(self):
        """Create a sample VisualElement."""
        return VisualElement(
            element_id="el_1",
            selector=".box",
            tag_name="div",
            bounds={"x": 100.0, "y": 100.0, "width": 200.0, "height": 100.0},
            computed_styles={},
            text_content="Box",
            attributes={},
            children_count=0,
        )

    def test_is_significant_true(self, sample_element):
        """Test is_significant for significant shift."""
        shift = LayoutShift(
            element=sample_element,
            delta_x=50.0,
            delta_y=30.0,
            delta_width=0.0,
            delta_height=0.0,
            shift_distance=58.3,  # sqrt(50^2 + 30^2)
        )
        assert shift.is_significant(threshold=5.0) is True

    def test_is_significant_false(self, sample_element):
        """Test is_significant for minor shift."""
        shift = LayoutShift(
            element=sample_element,
            delta_x=2.0,
            delta_y=1.0,
            delta_width=0.0,
            delta_height=0.0,
            shift_distance=2.24,
        )
        assert shift.is_significant(threshold=5.0) is False

    def test_get_shift_direction_right_down(self, sample_element):
        """Test shift direction detection."""
        shift = LayoutShift(
            element=sample_element,
            delta_x=100.0,
            delta_y=50.0,
            delta_width=0.0,
            delta_height=0.0,
            shift_distance=111.8,
        )
        direction = shift.get_shift_direction()
        assert "right" in direction

    def test_get_shift_direction_left_up(self, sample_element):
        """Test shift direction for left/up movement."""
        shift = LayoutShift(
            element=sample_element,
            delta_x=-100.0,
            delta_y=-50.0,
            delta_width=0.0,
            delta_height=0.0,
            shift_distance=111.8,
        )
        direction = shift.get_shift_direction()
        assert "left" in direction or "up" in direction

    def test_get_shift_direction_expanded(self, sample_element):
        """Test shift direction for expansion."""
        shift = LayoutShift(
            element=sample_element,
            delta_x=0.0,
            delta_y=0.0,
            delta_width=50.0,
            delta_height=20.0,
            shift_distance=0.0,
        )
        direction = shift.get_shift_direction()
        assert "expanded" in direction

    def test_get_shift_direction_no_movement(self, sample_element):
        """Test shift direction when no movement."""
        shift = LayoutShift(
            element=sample_element,
            delta_x=0.0,
            delta_y=0.0,
            delta_width=0.0,
            delta_height=0.0,
            shift_distance=0.0,
        )
        direction = shift.get_shift_direction()
        assert "no movement" in direction

    def test_to_dict(self, sample_element):
        """Test to_dict serialization."""
        shift = LayoutShift(
            element=sample_element,
            delta_x=10.0,
            delta_y=20.0,
            delta_width=5.0,
            delta_height=3.0,
            shift_distance=22.4,
        )
        result = shift.to_dict()
        assert result["delta_x"] == 10.0
        assert result["delta_y"] == 20.0
        assert result["is_significant"] is True
        assert "direction" in result


class TestDOMTreeParser:
    """Tests for DOMTreeParser class."""

    def test_parse_simple_html(self):
        """Test parsing simple HTML."""
        parser = DOMTreeParser()
        parser.feed("<html><body><div>Hello</div></body></html>")
        assert len(parser.elements) > 0
        assert parser.elements[0]["tag"] == "html"

    def test_parse_nested_elements(self):
        """Test parsing nested elements."""
        parser = DOMTreeParser()
        parser.feed("<div><p>Paragraph</p><span>Span</span></div>")
        assert len(parser.elements) == 1
        div = parser.elements[0]
        assert div["tag"] == "div"
        assert len(div["children"]) == 2

    def test_parse_with_attributes(self):
        """Test parsing elements with attributes."""
        parser = DOMTreeParser()
        parser.feed('<button id="submit" class="btn primary">Submit</button>')
        assert len(parser.elements) == 1
        button = parser.elements[0]
        assert button["tag"] == "button"
        assert button["attributes"]["id"] == "submit"
        assert "btn primary" in button["attributes"]["class"]

    def test_parse_text_content(self):
        """Test extracting text content."""
        parser = DOMTreeParser()
        parser.feed("<p>Hello World</p>")
        assert parser.elements[0]["text"] == "Hello World"

    def test_parse_empty_html(self):
        """Test parsing empty HTML."""
        parser = DOMTreeParser()
        parser.feed("")
        assert len(parser.elements) == 0


class TestStructuralAnalyzer:
    """Tests for StructuralAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a StructuralAnalyzer instance."""
        return StructuralAnalyzer()

    @pytest.fixture
    def baseline_elements(self):
        """Create baseline VisualElements."""
        return [
            VisualElement(
                element_id="header",
                selector="#header",
                tag_name="header",
                bounds={"x": 0.0, "y": 0.0, "width": 800.0, "height": 60.0},
                computed_styles={"display": "flex"},
                text_content="Header",
                attributes={},
                children_count=3,
            ),
            VisualElement(
                element_id="content",
                selector="#content",
                tag_name="main",
                bounds={"x": 0.0, "y": 60.0, "width": 800.0, "height": 600.0},
                computed_styles={"display": "block"},
                text_content="Content",
                attributes={},
                children_count=10,
            ),
        ]

    @pytest.fixture
    def current_elements(self):
        """Create current VisualElements with changes."""
        return [
            VisualElement(
                element_id="header",
                selector="#header",
                tag_name="header",
                bounds={"x": 0.0, "y": 0.0, "width": 800.0, "height": 80.0},  # Height changed
                computed_styles={"display": "flex"},
                text_content="Updated Header",  # Text changed
                attributes={},
                children_count=3,
            ),
            VisualElement(
                element_id="content",
                selector="#content",
                tag_name="main",
                bounds={"x": 0.0, "y": 80.0, "width": 800.0, "height": 580.0},  # Position/size changed
                computed_styles={"display": "block"},
                text_content="Content",
                attributes={},
                children_count=10,
            ),
            VisualElement(
                element_id="footer",  # New element
                selector="#footer",
                tag_name="footer",
                bounds={"x": 0.0, "y": 660.0, "width": 800.0, "height": 60.0},
                computed_styles={"display": "block"},
                text_content="Footer",
                attributes={},
                children_count=2,
            ),
        ]

    def test_init(self, analyzer):
        """Test StructuralAnalyzer initialization."""
        assert analyzer.position_threshold == 50.0
        assert analyzer.size_threshold == 20.0
        assert analyzer.text_similarity_threshold == 0.8

    def test_init_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        analyzer = StructuralAnalyzer(
            position_threshold=100.0,
            size_threshold=30.0,
            text_similarity_threshold=0.9,
        )
        assert analyzer.position_threshold == 100.0
        assert analyzer.size_threshold == 30.0
        assert analyzer.text_similarity_threshold == 0.9

    @pytest.mark.asyncio
    async def test_compare_structure(self, analyzer):
        """Test DOM structure comparison."""
        baseline_dom = "<html><body><div id='main'>Hello</div></body></html>"
        current_dom = "<html><body><div id='main'>World</div><p>New</p></body></html>"

        diff = await analyzer.compare_structure(baseline_dom, current_dom)
        assert isinstance(diff, VisualStructuralDiff)

    @pytest.mark.asyncio
    async def test_compare_structure_identical(self, analyzer):
        """Test comparing identical DOM."""
        dom = "<html><body><div>Same</div></body></html>"
        diff = await analyzer.compare_structure(dom, dom)
        # Should have minimal/no changes
        assert diff.total_changes() <= 1  # May detect minor differences

    @pytest.mark.asyncio
    async def test_detect_layout_shifts(
        self, analyzer, baseline_elements, current_elements
    ):
        """Test layout shift detection."""
        shifts = await analyzer.detect_layout_shifts(baseline_elements, current_elements)
        assert isinstance(shifts, list)
        for shift in shifts:
            assert isinstance(shift, LayoutShift)

    @pytest.mark.asyncio
    async def test_detect_layout_shifts_header_moved(
        self, analyzer, baseline_elements, current_elements
    ):
        """Test that header size change is detected."""
        shifts = await analyzer.detect_layout_shifts(baseline_elements, current_elements)
        # Header height changed from 60 to 80
        header_shifts = [s for s in shifts if s.element.element_id == "header"]
        assert len(header_shifts) == 1
        assert header_shifts[0].delta_height == 20.0

    @pytest.mark.asyncio
    async def test_track_component_changes(self, analyzer):
        """Test component tracking."""
        baseline = VisualSnapshot(
            id="baseline",
            url="https://example.com",
            viewport={"width": 800, "height": 600},
            device_name=None,
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"screenshot",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[
                VisualElement(
                    element_id="nav",
                    selector="#navigation",
                    tag_name="nav",
                    bounds={"x": 0.0, "y": 0.0, "width": 800.0, "height": 50.0},
                    computed_styles={},
                    text_content="Nav",
                    attributes={},
                    children_count=5,
                ),
            ],
            layout_hash="hash1",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )
        current = VisualSnapshot(
            id="current",
            url="https://example.com",
            viewport={"width": 800, "height": 600},
            device_name=None,
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"screenshot",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[
                VisualElement(
                    element_id="nav",
                    selector="#navigation",
                    tag_name="nav",
                    bounds={"x": 0.0, "y": 0.0, "width": 800.0, "height": 60.0},  # Changed
                    computed_styles={},
                    text_content="Navigation",  # Changed
                    attributes={},
                    children_count=6,  # Changed
                ),
            ],
            layout_hash="hash2",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=None,
            cumulative_layout_shift=None,
            time_to_interactive=None,
        )

        results = await analyzer.track_component_changes(
            baseline, current, ["#navigation"]
        )
        assert "#navigation" in results
        nav_result = results["#navigation"]
        assert nav_result["status"] in ["modified", "moved"]

    def test_match_elements_by_id(self, analyzer, baseline_elements, current_elements):
        """Test element matching by ID."""
        matches, unmatched_baseline, unmatched_current = analyzer._match_elements(
            baseline_elements, current_elements
        )
        # Header and content should match by ID
        assert len(matches) == 2
        # Footer is new
        assert len(unmatched_current) == 1
        assert unmatched_current[0].element_id == "footer"

    def test_calculate_position_delta(self, analyzer):
        """Test position delta calculation."""
        el1 = VisualElement(
            element_id="el_1",
            selector=".box",
            tag_name="div",
            bounds={"x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0},
            computed_styles={},
            text_content="",
            attributes={},
            children_count=0,
        )
        el2 = VisualElement(
            element_id="el_2",
            selector=".box",
            tag_name="div",
            bounds={"x": 100.0, "y": 100.0, "width": 100.0, "height": 100.0},
            computed_styles={},
            text_content="",
            attributes={},
            children_count=0,
        )
        delta = analyzer._calculate_position_delta(el1, el2)
        # Centers: (50,50) and (150,150), distance = sqrt(10000+10000) ~ 141.4
        assert 141 < delta < 142

    def test_text_similarity(self, analyzer):
        """Test text similarity calculation."""
        assert analyzer._text_similarity("hello", "hello") == 1.0
        assert analyzer._text_similarity("", "") == 1.0
        assert analyzer._text_similarity("hello", "") == 0.0
        assert analyzer._text_similarity("", "world") == 0.0
        sim = analyzer._text_similarity("hello", "hallo")
        assert 0.7 < sim < 1.0

    def test_get_changed_properties(self, analyzer):
        """Test property change detection."""
        baseline = VisualElement(
            element_id="el_1",
            selector=".box",
            tag_name="div",
            bounds={"x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0},
            computed_styles={"color": "red", "display": "block"},
            text_content="Hello",
            attributes={"class": "old"},
            children_count=2,
        )
        current = VisualElement(
            element_id="el_1",
            selector=".box",
            tag_name="div",
            bounds={"x": 10.0, "y": 0.0, "width": 110.0, "height": 100.0},  # x, width changed
            computed_styles={"color": "blue", "display": "block"},  # color changed
            text_content="Hello",
            attributes={"class": "new"},  # class changed
            children_count=3,  # children changed
        )
        changes = analyzer._get_changed_properties(baseline, current)
        assert "x" in changes
        assert "width" in changes
        assert "style:color" in changes
        assert "attr:class" in changes
        assert "children_count" in changes

    def test_generate_diff_report(self, analyzer):
        """Test diff report generation."""
        diff = VisualStructuralDiff()
        baseline_dom = "<html><body><p>Hello</p></body></html>"
        current_dom = "<html><body><p>World</p></body></html>"

        report = analyzer.generate_diff_report(diff, baseline_dom, current_dom)
        assert "structural_diff" in report
        assert "unified_diff" in report
        assert "html_diff" in report
        assert "line_delta" in report

    def test_calculate_cumulative_layout_shift(self, analyzer):
        """Test CLS calculation."""
        element = VisualElement(
            element_id="el_1",
            selector=".box",
            tag_name="div",
            bounds={"x": 0.0, "y": 0.0, "width": 200.0, "height": 100.0},
            computed_styles={},
            text_content="",
            attributes={},
            children_count=0,
        )
        shifts = [
            LayoutShift(
                element=element,
                delta_x=50.0,
                delta_y=50.0,
                delta_width=0.0,
                delta_height=0.0,
                shift_distance=70.7,
            )
        ]
        cls_score = analyzer.calculate_cumulative_layout_shift(shifts)
        assert isinstance(cls_score, float)
        assert cls_score >= 0.0

    def test_calculate_cls_empty_shifts(self, analyzer):
        """Test CLS calculation with no shifts."""
        cls_score = analyzer.calculate_cumulative_layout_shift([])
        assert cls_score == 0.0

    def test_convert_to_structural_diff(self, analyzer):
        """Test conversion from VisualStructuralDiff to StructuralDiff."""
        element = VisualElement(
            element_id="new_el",
            selector=".new-box",
            tag_name="div",
            bounds={"x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0},
            computed_styles={},
            text_content="New element",
            attributes={},
            children_count=0,
        )
        visual_diff = VisualStructuralDiff(
            added_elements=[element],
            removed_elements=[],
            moved_elements=[],
            modified_elements=[],
            text_changes=[],
        )
        structural_diff = analyzer.convert_to_structural_diff(visual_diff)
        assert isinstance(structural_diff, StructuralDiff)
        assert structural_diff.elements_added == 1
        assert len(structural_diff.changes) == 1
        assert structural_diff.changes[0].change_type == StructuralChangeType.ADDED
