"""Comprehensive tests for visual_ai/models.py.

Tests all data models including enums, dataclasses, serialization/deserialization,
and helper methods.
"""

import base64
import json

import pytest

from src.visual_ai.models import (
    ChangeCategory,
    ChangeIntent,
    Severity,
    VisualChange,
    VisualComparisonResult,
    VisualElement,
    VisualSnapshot,
)


class TestChangeCategory:
    """Tests for ChangeCategory enum."""

    def test_all_categories_exist(self):
        """Test that all expected categories exist."""
        expected = [
            "layout", "content", "style", "structure",
            "responsive", "animation", "accessibility", "performance"
        ]
        for cat in expected:
            assert hasattr(ChangeCategory, cat.upper())

    def test_category_values(self):
        """Test enum values are strings."""
        assert ChangeCategory.LAYOUT.value == "layout"
        assert ChangeCategory.CONTENT.value == "content"
        assert ChangeCategory.STYLE.value == "style"
        assert ChangeCategory.STRUCTURE.value == "structure"
        assert ChangeCategory.RESPONSIVE.value == "responsive"
        assert ChangeCategory.ANIMATION.value == "animation"
        assert ChangeCategory.ACCESSIBILITY.value == "accessibility"
        assert ChangeCategory.PERFORMANCE.value == "performance"


class TestChangeIntent:
    """Tests for ChangeIntent enum."""

    def test_all_intents_exist(self):
        """Test that all expected intents exist."""
        expected = ["intentional", "regression", "dynamic", "environmental", "unknown"]
        for intent in expected:
            assert hasattr(ChangeIntent, intent.upper())

    def test_intent_values(self):
        """Test enum values are strings."""
        assert ChangeIntent.INTENTIONAL.value == "intentional"
        assert ChangeIntent.REGRESSION.value == "regression"
        assert ChangeIntent.DYNAMIC.value == "dynamic"
        assert ChangeIntent.ENVIRONMENTAL.value == "environmental"
        assert ChangeIntent.UNKNOWN.value == "unknown"


class TestSeverity:
    """Tests for Severity enum."""

    def test_all_severities_exist(self):
        """Test that all expected severities exist."""
        expected = ["critical", "major", "minor", "info", "safe"]
        for sev in expected:
            assert hasattr(Severity, sev.upper())

    def test_severity_values_are_ordered(self):
        """Test severity values are ordered numerically."""
        assert Severity.SAFE.value == 0
        assert Severity.INFO.value == 1
        assert Severity.MINOR.value == 2
        assert Severity.MAJOR.value == 3
        assert Severity.CRITICAL.value == 4

    def test_severity_less_than(self):
        """Test severity __lt__ comparison."""
        assert Severity.SAFE < Severity.INFO
        assert Severity.INFO < Severity.MINOR
        assert Severity.MINOR < Severity.MAJOR
        assert Severity.MAJOR < Severity.CRITICAL

    def test_severity_less_than_equal(self):
        """Test severity __le__ comparison."""
        assert Severity.SAFE <= Severity.SAFE
        assert Severity.INFO <= Severity.MINOR
        assert Severity.MINOR <= Severity.MAJOR

    def test_severity_greater_than(self):
        """Test severity __gt__ comparison."""
        assert Severity.CRITICAL > Severity.MAJOR
        assert Severity.MAJOR > Severity.MINOR
        assert Severity.MINOR > Severity.INFO

    def test_severity_greater_than_equal(self):
        """Test severity __ge__ comparison."""
        assert Severity.CRITICAL >= Severity.CRITICAL
        assert Severity.MAJOR >= Severity.MINOR
        assert Severity.INFO >= Severity.SAFE

    def test_severity_comparison_with_non_severity(self):
        """Test comparison with non-Severity returns NotImplemented."""
        assert Severity.MAJOR.__lt__(5) is NotImplemented
        assert Severity.MAJOR.__le__(5) is NotImplemented
        assert Severity.MAJOR.__gt__(5) is NotImplemented
        assert Severity.MAJOR.__ge__(5) is NotImplemented


class TestVisualElement:
    """Tests for VisualElement dataclass."""

    @pytest.fixture
    def sample_element(self):
        """Create a sample VisualElement for testing."""
        return VisualElement(
            element_id="el_123",
            selector="#main-button",
            tag_name="button",
            bounds={"x": 100.0, "y": 200.0, "width": 150.0, "height": 50.0},
            computed_styles={"color": "red", "background": "blue"},
            text_content="Click me",
            attributes={"class": "btn", "data-testid": "submit-btn"},
            children_count=2,
            screenshot_region=None,
        )

    @pytest.fixture
    def element_with_screenshot(self):
        """Create a VisualElement with screenshot region."""
        return VisualElement(
            element_id="el_456",
            selector=".image-box",
            tag_name="div",
            bounds={"x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0},
            computed_styles={},
            text_content=None,
            attributes={},
            children_count=0,
            screenshot_region=b"\x89PNG\r\n\x1a\ntest",
        )

    def test_to_dict_without_screenshot(self, sample_element):
        """Test to_dict without screenshot region."""
        result = sample_element.to_dict()
        assert result["element_id"] == "el_123"
        assert result["selector"] == "#main-button"
        assert result["tag_name"] == "button"
        assert result["bounds"]["x"] == 100.0
        assert result["text_content"] == "Click me"
        assert result["screenshot_region"] is None

    def test_to_dict_with_screenshot(self, element_with_screenshot):
        """Test to_dict with screenshot region encodes to base64."""
        result = element_with_screenshot.to_dict()
        assert result["screenshot_region"] is not None
        # Decode and verify
        decoded = base64.b64decode(result["screenshot_region"])
        assert decoded == b"\x89PNG\r\n\x1a\ntest"

    def test_from_dict_without_screenshot(self, sample_element):
        """Test from_dict recreates element correctly."""
        dict_data = sample_element.to_dict()
        recreated = VisualElement.from_dict(dict_data)
        assert recreated.element_id == sample_element.element_id
        assert recreated.selector == sample_element.selector
        assert recreated.bounds == sample_element.bounds

    def test_from_dict_with_screenshot(self, element_with_screenshot):
        """Test from_dict decodes base64 screenshot."""
        dict_data = element_with_screenshot.to_dict()
        recreated = VisualElement.from_dict(dict_data)
        assert recreated.screenshot_region == b"\x89PNG\r\n\x1a\ntest"

    def test_get_center(self, sample_element):
        """Test get_center calculates center point correctly."""
        center = sample_element.get_center()
        assert center == (175.0, 225.0)  # 100+150/2, 200+50/2

    def test_get_area(self, sample_element):
        """Test get_area calculates area correctly."""
        area = sample_element.get_area()
        assert area == 7500.0  # 150 * 50

    def test_overlaps_true(self):
        """Test overlaps returns True for overlapping elements."""
        el1 = VisualElement(
            element_id="1",
            selector="a",
            tag_name="div",
            bounds={"x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0},
            computed_styles={},
            text_content=None,
            attributes={},
            children_count=0,
        )
        el2 = VisualElement(
            element_id="2",
            selector="b",
            tag_name="div",
            bounds={"x": 50.0, "y": 50.0, "width": 100.0, "height": 100.0},
            computed_styles={},
            text_content=None,
            attributes={},
            children_count=0,
        )
        assert el1.overlaps(el2) is True

    def test_overlaps_false(self):
        """Test overlaps returns False for non-overlapping elements."""
        el1 = VisualElement(
            element_id="1",
            selector="a",
            tag_name="div",
            bounds={"x": 0.0, "y": 0.0, "width": 50.0, "height": 50.0},
            computed_styles={},
            text_content=None,
            attributes={},
            children_count=0,
        )
        el2 = VisualElement(
            element_id="2",
            selector="b",
            tag_name="div",
            bounds={"x": 100.0, "y": 100.0, "width": 50.0, "height": 50.0},
            computed_styles={},
            text_content=None,
            attributes={},
            children_count=0,
        )
        assert el1.overlaps(el2) is False


class TestVisualChange:
    """Tests for VisualChange dataclass."""

    @pytest.fixture
    def sample_element(self):
        """Create a sample element for change."""
        return VisualElement(
            element_id="el_789",
            selector=".header",
            tag_name="header",
            bounds={"x": 0.0, "y": 0.0, "width": 1920.0, "height": 80.0},
            computed_styles={},
            text_content="Header",
            attributes={},
            children_count=3,
        )

    @pytest.fixture
    def sample_change(self, sample_element):
        """Create a sample VisualChange for testing."""
        return VisualChange(
            id="change_001",
            category=ChangeCategory.LAYOUT,
            intent=ChangeIntent.REGRESSION,
            severity=Severity.MAJOR,
            element=sample_element,
            bounds_baseline={"x": 0.0, "y": 0.0, "width": 1920.0, "height": 80.0},
            bounds_current={"x": 0.0, "y": 10.0, "width": 1920.0, "height": 90.0},
            property_name="height",
            baseline_value="80px",
            current_value="90px",
            description="Header height increased",
            root_cause="CSS change in header.css",
            impact_assessment="May affect above-the-fold content",
            recommendation="Review header styling",
            confidence=0.95,
            related_commit="abc123",
            related_files=["src/styles/header.css"],
        )

    def test_to_dict(self, sample_change):
        """Test to_dict serialization."""
        result = sample_change.to_dict()
        assert result["id"] == "change_001"
        assert result["category"] == "layout"
        assert result["intent"] == "regression"
        assert result["severity"] == 3  # MAJOR value
        assert result["element"]["element_id"] == "el_789"
        assert result["description"] == "Header height increased"
        assert "src/styles/header.css" in result["related_files"]

    def test_from_dict(self, sample_change):
        """Test from_dict deserialization."""
        dict_data = sample_change.to_dict()
        recreated = VisualChange.from_dict(dict_data)
        assert recreated.id == sample_change.id
        assert recreated.category == ChangeCategory.LAYOUT
        assert recreated.intent == ChangeIntent.REGRESSION
        assert recreated.severity == Severity.MAJOR

    def test_from_dict_without_element(self):
        """Test from_dict handles missing element."""
        data = {
            "id": "test_id",
            "category": "content",
            "intent": "intentional",
            "severity": 1,
            "element": None,
            "description": "Test change",
            "impact_assessment": "None",
            "recommendation": "None",
            "confidence": 0.8,
        }
        change = VisualChange.from_dict(data)
        assert change.element is None

    def test_is_blocking_true(self, sample_change):
        """Test is_blocking returns True for major regression."""
        assert sample_change.is_blocking() is True

    def test_is_blocking_false_when_intentional(self, sample_change):
        """Test is_blocking returns False when intentional."""
        sample_change.intent = ChangeIntent.INTENTIONAL
        assert sample_change.is_blocking() is False

    def test_is_blocking_false_when_minor(self, sample_change):
        """Test is_blocking returns False for minor severity."""
        sample_change.severity = Severity.MINOR
        assert sample_change.is_blocking() is False

    def test_is_regression(self, sample_change):
        """Test is_regression correctly identifies regressions."""
        assert sample_change.is_regression() is True
        sample_change.intent = ChangeIntent.INTENTIONAL
        assert sample_change.is_regression() is False

    def test_get_bounds_delta(self, sample_change):
        """Test get_bounds_delta calculates differences correctly."""
        delta = sample_change.get_bounds_delta()
        assert delta["x"] == 0.0
        assert delta["y"] == 10.0
        assert delta["width"] == 0.0
        assert delta["height"] == 10.0

    def test_get_bounds_delta_returns_none_when_missing(self, sample_change):
        """Test get_bounds_delta returns None when bounds missing."""
        sample_change.bounds_baseline = None
        assert sample_change.get_bounds_delta() is None


class TestVisualSnapshot:
    """Tests for VisualSnapshot dataclass."""

    @pytest.fixture
    def sample_snapshot(self):
        """Create a sample VisualSnapshot for testing."""
        element = VisualElement(
            element_id="el_1",
            selector="#content",
            tag_name="div",
            bounds={"x": 0.0, "y": 0.0, "width": 800.0, "height": 600.0},
            computed_styles={"display": "block"},
            text_content="Content",
            attributes={},
            children_count=5,
        )
        return VisualSnapshot(
            id="snapshot_001",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name="Desktop",
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"\x89PNG\r\n\x1a\ntest_screenshot",
            dom_snapshot='{"nodeId": 1, "tagName": "html"}',
            computed_styles={"el_1": {"display": "block"}},
            network_har=None,
            elements=[element],
            layout_hash="abc123def456",
            color_palette=["#ffffff", "#000000", "#ff0000"],
            text_blocks=[{"text": "Hello", "bounds": {"x": 0, "y": 0}}],
            largest_contentful_paint=1500.0,
            cumulative_layout_shift=0.05,
            time_to_interactive=3000.0,
        )

    def test_to_dict(self, sample_snapshot):
        """Test to_dict serialization."""
        result = sample_snapshot.to_dict()
        assert result["id"] == "snapshot_001"
        assert result["url"] == "https://example.com"
        assert result["browser"] == "chromium"
        # Screenshot should be base64 encoded
        decoded = base64.b64decode(result["screenshot"])
        assert decoded == b"\x89PNG\r\n\x1a\ntest_screenshot"
        assert len(result["elements"]) == 1

    def test_from_dict(self, sample_snapshot):
        """Test from_dict deserialization."""
        dict_data = sample_snapshot.to_dict()
        recreated = VisualSnapshot.from_dict(dict_data)
        assert recreated.id == sample_snapshot.id
        assert recreated.url == sample_snapshot.url
        assert recreated.screenshot == sample_snapshot.screenshot
        assert len(recreated.elements) == 1

    def test_to_json(self, sample_snapshot):
        """Test to_json produces valid JSON."""
        json_str = sample_snapshot.to_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "snapshot_001"

    def test_from_json(self, sample_snapshot):
        """Test from_json reconstructs snapshot."""
        json_str = sample_snapshot.to_json()
        recreated = VisualSnapshot.from_json(json_str)
        assert recreated.id == sample_snapshot.id

    def test_get_element_by_selector(self, sample_snapshot):
        """Test get_element_by_selector finds element."""
        element = sample_snapshot.get_element_by_selector("#content")
        assert element is not None
        assert element.element_id == "el_1"

    def test_get_element_by_selector_not_found(self, sample_snapshot):
        """Test get_element_by_selector returns None when not found."""
        element = sample_snapshot.get_element_by_selector("#nonexistent")
        assert element is None

    def test_get_elements_by_tag(self, sample_snapshot):
        """Test get_elements_by_tag finds elements."""
        elements = sample_snapshot.get_elements_by_tag("div")
        assert len(elements) == 1
        assert elements[0].tag_name == "div"

    def test_get_elements_by_tag_case_insensitive(self, sample_snapshot):
        """Test get_elements_by_tag is case insensitive."""
        elements = sample_snapshot.get_elements_by_tag("DIV")
        assert len(elements) == 1

    def test_get_performance_score_good(self, sample_snapshot):
        """Test get_performance_score with good metrics."""
        score = sample_snapshot.get_performance_score()
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_get_performance_score_poor(self):
        """Test get_performance_score with poor metrics."""
        snapshot = VisualSnapshot(
            id="poor_perf",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name=None,
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"test",
            dom_snapshot="{}",
            computed_styles={},
            network_har=None,
            elements=[],
            layout_hash="",
            color_palette=[],
            text_blocks=[],
            largest_contentful_paint=5000.0,  # Poor LCP
            cumulative_layout_shift=0.5,  # Poor CLS
            time_to_interactive=10000.0,  # Poor TTI
        )
        score = snapshot.get_performance_score()
        assert score == 0.0

    def test_get_performance_score_no_metrics(self):
        """Test get_performance_score returns None when no metrics."""
        snapshot = VisualSnapshot(
            id="no_perf",
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            device_name=None,
            browser="chromium",
            timestamp="2024-01-01T12:00:00Z",
            screenshot=b"test",
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
        assert snapshot.get_performance_score() is None


class TestVisualComparisonResult:
    """Tests for VisualComparisonResult dataclass."""

    @pytest.fixture
    def sample_change(self):
        """Create a sample change for comparison result."""
        return VisualChange(
            id="change_001",
            category=ChangeCategory.STYLE,
            intent=ChangeIntent.REGRESSION,
            severity=Severity.MAJOR,
            element=None,
            bounds_baseline=None,
            bounds_current=None,
            property_name="color",
            baseline_value="#000",
            current_value="#333",
            description="Text color changed",
            root_cause=None,
            impact_assessment="Minor visibility impact",
            recommendation="Verify color change",
            confidence=0.9,
            related_commit=None,
        )

    @pytest.fixture
    def sample_comparison(self, sample_change):
        """Create a sample comparison result."""
        return VisualComparisonResult(
            id="comparison_001",
            baseline_snapshot="snapshot_baseline",
            current_snapshot="snapshot_current",
            match=False,
            match_percentage=85.5,
            changes=[sample_change],
            changes_by_category={"style": 1},
            changes_by_severity={"MAJOR": 1},
            auto_approval_recommendation=False,
            approval_confidence=0.7,
            blocking_changes=["change_001"],
            diff_image_url="https://cdn/diff.png",
            side_by_side_url="https://cdn/sbs.png",
            animated_gif_url="https://cdn/anim.gif",
            lcp_delta=500.0,
            cls_delta=0.02,
            analysis_cost_usd=0.05,
            analysis_duration_ms=2500,
        )

    def test_to_dict(self, sample_comparison):
        """Test to_dict serialization."""
        result = sample_comparison.to_dict()
        assert result["id"] == "comparison_001"
        assert result["match_percentage"] == 85.5
        assert len(result["changes"]) == 1
        assert result["blocking_changes"] == ["change_001"]

    def test_from_dict(self, sample_comparison):
        """Test from_dict deserialization."""
        dict_data = sample_comparison.to_dict()
        recreated = VisualComparisonResult.from_dict(dict_data)
        assert recreated.id == sample_comparison.id
        assert len(recreated.changes) == 1
        assert recreated.match_percentage == 85.5

    def test_has_blocking_changes(self, sample_comparison):
        """Test has_blocking_changes detection."""
        assert sample_comparison.has_blocking_changes() is True

    def test_has_blocking_changes_false(self, sample_comparison):
        """Test has_blocking_changes when none blocking."""
        sample_comparison.blocking_changes = []
        assert sample_comparison.has_blocking_changes() is False

    def test_get_blocking_change_objects(self, sample_comparison):
        """Test get_blocking_change_objects returns correct changes."""
        blocking = sample_comparison.get_blocking_change_objects()
        assert len(blocking) == 1
        assert blocking[0].id == "change_001"

    def test_get_changes_by_category(self, sample_comparison):
        """Test get_changes_by_category filtering."""
        style_changes = sample_comparison.get_changes_by_category(ChangeCategory.STYLE)
        assert len(style_changes) == 1
        layout_changes = sample_comparison.get_changes_by_category(ChangeCategory.LAYOUT)
        assert len(layout_changes) == 0

    def test_get_changes_by_severity(self, sample_comparison):
        """Test get_changes_by_severity filtering."""
        major_changes = sample_comparison.get_changes_by_severity(Severity.MAJOR)
        assert len(major_changes) == 1
        minor_changes = sample_comparison.get_changes_by_severity(Severity.MINOR)
        assert len(minor_changes) == 0

    def test_get_regressions(self, sample_comparison):
        """Test get_regressions returns regression changes."""
        regressions = sample_comparison.get_regressions()
        assert len(regressions) == 1

    def test_get_highest_severity(self, sample_comparison):
        """Test get_highest_severity returns highest."""
        assert sample_comparison.get_highest_severity() == Severity.MAJOR

    def test_get_highest_severity_empty(self):
        """Test get_highest_severity returns None when no changes."""
        comparison = VisualComparisonResult(
            id="empty",
            baseline_snapshot="a",
            current_snapshot="b",
            match=True,
            match_percentage=100.0,
            changes=[],
            changes_by_category={},
            changes_by_severity={},
            auto_approval_recommendation=True,
            approval_confidence=1.0,
            blocking_changes=[],
            diff_image_url="",
            side_by_side_url="",
            animated_gif_url=None,
            lcp_delta=None,
            cls_delta=None,
            analysis_cost_usd=0.0,
            analysis_duration_ms=0,
        )
        assert comparison.get_highest_severity() is None

    def test_get_summary_match(self):
        """Test get_summary when matching."""
        comparison = VisualComparisonResult(
            id="match",
            baseline_snapshot="a",
            current_snapshot="b",
            match=True,
            match_percentage=99.5,
            changes=[],
            changes_by_category={},
            changes_by_severity={},
            auto_approval_recommendation=True,
            approval_confidence=1.0,
            blocking_changes=[],
            diff_image_url="",
            side_by_side_url="",
            animated_gif_url=None,
            lcp_delta=None,
            cls_delta=None,
            analysis_cost_usd=0.0,
            analysis_duration_ms=0,
        )
        summary = comparison.get_summary()
        assert "99.5%" in summary

    def test_get_summary_with_changes(self, sample_comparison):
        """Test get_summary with changes."""
        summary = sample_comparison.get_summary()
        assert "1 change(s)" in summary
        assert "1 blocking" in summary

    def test_should_auto_approve_true(self):
        """Test should_auto_approve when conditions met."""
        comparison = VisualComparisonResult(
            id="approve",
            baseline_snapshot="a",
            current_snapshot="b",
            match=True,
            match_percentage=99.0,
            changes=[],
            changes_by_category={},
            changes_by_severity={},
            auto_approval_recommendation=True,
            approval_confidence=0.95,
            blocking_changes=[],
            diff_image_url="",
            side_by_side_url="",
            animated_gif_url=None,
            lcp_delta=None,
            cls_delta=None,
            analysis_cost_usd=0.0,
            analysis_duration_ms=0,
        )
        assert comparison.should_auto_approve() is True

    def test_should_auto_approve_false_low_confidence(self, sample_comparison):
        """Test should_auto_approve when low confidence."""
        sample_comparison.auto_approval_recommendation = True
        sample_comparison.approval_confidence = 0.5
        assert sample_comparison.should_auto_approve() is False

    def test_to_json_and_from_json(self, sample_comparison):
        """Test JSON round-trip."""
        json_str = sample_comparison.to_json()
        recreated = VisualComparisonResult.from_json(json_str)
        assert recreated.id == sample_comparison.id
        assert recreated.match_percentage == sample_comparison.match_percentage

    def test_get_performance_regression(self, sample_comparison):
        """Test get_performance_regression detects regression."""
        # lcp_delta=500.0 in fixture is at threshold, set above to trigger
        sample_comparison.lcp_delta = 600.0  # Above 500ms threshold
        result = sample_comparison.get_performance_regression()
        assert result is not None
        assert "LCP" in result

    def test_get_performance_regression_none(self):
        """Test get_performance_regression when no regression."""
        comparison = VisualComparisonResult(
            id="no_perf_reg",
            baseline_snapshot="a",
            current_snapshot="b",
            match=True,
            match_percentage=100.0,
            changes=[],
            changes_by_category={},
            changes_by_severity={},
            auto_approval_recommendation=True,
            approval_confidence=1.0,
            blocking_changes=[],
            diff_image_url="",
            side_by_side_url="",
            animated_gif_url=None,
            lcp_delta=100.0,  # Below threshold
            cls_delta=0.01,  # Below threshold
            analysis_cost_usd=0.0,
            analysis_duration_ms=0,
        )
        assert comparison.get_performance_regression() is None
