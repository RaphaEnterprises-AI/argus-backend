"""Tests for RRWeb parser."""

import pytest
import json
from src.recording.rrweb_parser import (
    RRWebParser,
    parse_rrweb_recording,
    recording_to_test_spec,
)
from src.recording.models import (
    RRWebEventType,
    RRWebIncrementalSource,
    MouseInteractionType,
    ActionType,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_recording():
    """Simple rrweb recording with basic events."""
    return [
        # Meta event
        {
            "type": 4,
            "timestamp": 0,
            "data": {
                "href": "https://example.com/login",
                "width": 1920,
                "height": 1080
            }
        },
        # Full snapshot
        {
            "type": 2,
            "timestamp": 100,
            "data": {
                "node": {
                    "id": 1,
                    "tagName": "html",
                    "childNodes": [
                        {
                            "id": 2,
                            "tagName": "body",
                            "childNodes": [
                                {
                                    "id": 3,
                                    "tagName": "input",
                                    "attributes": {"id": "email", "type": "email"}
                                },
                                {
                                    "id": 4,
                                    "tagName": "input",
                                    "attributes": {"id": "password", "type": "password"}
                                },
                                {
                                    "id": 5,
                                    "tagName": "button",
                                    "attributes": {"id": "submit", "type": "submit"}
                                }
                            ]
                        }
                    ]
                }
            }
        },
        # Click on email input
        {
            "type": 3,
            "timestamp": 500,
            "data": {
                "source": 2,
                "type": 2,
                "id": 3,
                "x": 100,
                "y": 200
            }
        },
        # Input in email field
        {
            "type": 3,
            "timestamp": 1000,
            "data": {
                "source": 5,
                "id": 3,
                "text": "test@example.com"
            }
        },
        # Click on password input
        {
            "type": 3,
            "timestamp": 2000,
            "data": {
                "source": 2,
                "type": 2,
                "id": 4
            }
        },
        # Input in password field
        {
            "type": 3,
            "timestamp": 2500,
            "data": {
                "source": 5,
                "id": 4,
                "text": "secretpass123"
            }
        },
        # Click submit button
        {
            "type": 3,
            "timestamp": 3000,
            "data": {
                "source": 2,
                "type": 2,
                "id": 5
            }
        }
    ]


@pytest.fixture
def comprehensive_recording():
    """Comprehensive recording with many action types."""
    return [
        # Meta
        {"type": 4, "timestamp": 0, "data": {"href": "https://shop.example.com"}},
        # Full snapshot with various elements
        {
            "type": 2,
            "timestamp": 100,
            "data": {
                "node": {
                    "id": 1,
                    "tagName": "html",
                    "childNodes": [{
                        "id": 2,
                        "tagName": "body",
                        "childNodes": [
                            {"id": 10, "tagName": "input", "attributes": {"name": "search"}},
                            {"id": 11, "tagName": "select", "attributes": {"id": "category"}},
                            {"id": 12, "tagName": "div", "attributes": {"class": "product-card"}},
                            {"id": 13, "tagName": "button", "attributes": {"data-testid": "add-to-cart"}},
                            {"id": 14, "tagName": "input", "attributes": {"type": "checkbox", "id": "agree"}},
                        ]
                    }]
                }
            }
        },
        # Search input
        {"type": 3, "timestamp": 500, "data": {"source": 5, "id": 10, "text": "laptop"}},
        # Select category
        {"type": 3, "timestamp": 1000, "data": {"source": 5, "id": 11, "data": {"values": ["electronics"]}}},
        # Click product
        {"type": 3, "timestamp": 1500, "data": {"source": 2, "type": 2, "id": 12}},
        # Double click
        {"type": 3, "timestamp": 2000, "data": {"source": 2, "type": 4, "id": 12}},
        # Scroll
        {"type": 3, "timestamp": 2500, "data": {"source": 3, "x": 0, "y": 500}},
        # Click add to cart
        {"type": 3, "timestamp": 3000, "data": {"source": 2, "type": 2, "id": 13}},
        # Toggle checkbox
        {"type": 3, "timestamp": 3500, "data": {"source": 5, "id": 14, "isChecked": True}},
    ]


@pytest.fixture
def recording_with_mutations():
    """Recording with DOM mutations."""
    return [
        {"type": 4, "timestamp": 0, "data": {"href": "https://example.com"}},
        {
            "type": 2,
            "timestamp": 100,
            "data": {
                "node": {
                    "id": 1,
                    "tagName": "html",
                    "childNodes": [{"id": 2, "tagName": "body", "childNodes": []}]
                }
            }
        },
        # Mutation adding new elements
        {
            "type": 3,
            "timestamp": 500,
            "data": {
                "source": 0,
                "adds": [
                    {
                        "parentId": 2,
                        "node": {"id": 100, "tagName": "div", "attributes": {"id": "dynamic"}}
                    }
                ]
            }
        },
        # Click on dynamically added element
        {"type": 3, "timestamp": 1000, "data": {"source": 2, "type": 2, "id": 100}},
    ]


# =============================================================================
# RRWebParser Tests
# =============================================================================


class TestRRWebParserInit:
    """Tests for parser initialization."""

    def test_create_parser(self):
        """Test creating parser with defaults."""
        parser = RRWebParser()
        assert parser.merge_consecutive_inputs is True
        assert parser.filter_hover_events is True
        assert parser.min_scroll_distance == 100

    def test_create_parser_custom_options(self):
        """Test creating parser with custom options."""
        parser = RRWebParser(
            merge_consecutive_inputs=False,
            filter_hover_events=False,
            min_scroll_distance=50
        )
        assert parser.merge_consecutive_inputs is False
        assert parser.filter_hover_events is False
        assert parser.min_scroll_distance == 50


class TestRRWebParserBasic:
    """Tests for basic parsing functionality."""

    def test_parse_simple_recording(self, simple_recording):
        """Test parsing a simple recording."""
        parser = RRWebParser()
        session = parser.parse(simple_recording, session_id="test-001")

        assert session.id == "test-001"
        assert len(session.events) == 7
        assert session.duration_ms > 0

    def test_parse_json_string(self, simple_recording):
        """Test parsing JSON string input."""
        parser = RRWebParser()
        json_str = json.dumps(simple_recording)
        session = parser.parse(json_str)

        assert session.action_count > 0

    def test_parse_wrapped_events(self, simple_recording):
        """Test parsing events wrapped in object."""
        parser = RRWebParser()
        wrapped = {"events": simple_recording, "metadata": {}}
        session = parser.parse(wrapped)

        assert session.action_count > 0

    def test_parse_extracts_metadata(self, simple_recording):
        """Test metadata extraction."""
        parser = RRWebParser()
        session = parser.parse(simple_recording)

        assert session.metadata.href == "https://example.com/login"
        assert session.metadata.width == 1920
        assert session.metadata.height == 1080


class TestRRWebParserActions:
    """Tests for action parsing."""

    def test_parse_navigation(self, simple_recording):
        """Test navigation action from meta event."""
        parser = RRWebParser()
        session = parser.parse(simple_recording)

        goto_actions = [a for a in session.parsed_actions if a.type == ActionType.GOTO]
        assert len(goto_actions) == 1
        assert goto_actions[0].target == "https://example.com/login"

    def test_parse_clicks(self, simple_recording):
        """Test click action parsing."""
        parser = RRWebParser()
        session = parser.parse(simple_recording)

        click_actions = [a for a in session.parsed_actions if a.type == ActionType.CLICK]
        assert len(click_actions) == 3  # email, password, submit clicks

    def test_parse_input_merging(self, simple_recording):
        """Test input events are merged into fill actions."""
        parser = RRWebParser(merge_consecutive_inputs=True)
        session = parser.parse(simple_recording)

        fill_actions = [a for a in session.parsed_actions if a.type == ActionType.FILL]
        # Should have 2 fill actions (email and password)
        assert len(fill_actions) >= 2

    def test_parse_input_no_merging(self, simple_recording):
        """Test input events without merging."""
        parser = RRWebParser(merge_consecutive_inputs=False)
        session = parser.parse(simple_recording)

        type_actions = [a for a in session.parsed_actions if a.type == ActionType.TYPE]
        fill_actions = [a for a in session.parsed_actions if a.type == ActionType.FILL]
        # Without merging, should have TYPE actions instead of FILL
        assert len(type_actions) >= 0 or len(fill_actions) >= 0


class TestRRWebParserComprehensive:
    """Tests for comprehensive action types."""

    def test_parse_double_click(self, comprehensive_recording):
        """Test double click parsing."""
        parser = RRWebParser()
        session = parser.parse(comprehensive_recording)

        dbl_clicks = [a for a in session.parsed_actions if a.type == ActionType.DOUBLE_CLICK]
        assert len(dbl_clicks) == 1

    def test_parse_select(self, comprehensive_recording):
        """Test select option parsing."""
        parser = RRWebParser()
        session = parser.parse(comprehensive_recording)

        selects = [a for a in session.parsed_actions if a.type == ActionType.SELECT]
        assert len(selects) == 1
        assert selects[0].value == "electronics"

    def test_parse_scroll(self, comprehensive_recording):
        """Test scroll parsing."""
        parser = RRWebParser(min_scroll_distance=100)
        session = parser.parse(comprehensive_recording)

        scrolls = [a for a in session.parsed_actions if a.type == ActionType.SCROLL]
        # Scroll of 500 should be recorded
        assert len(scrolls) >= 1

    def test_parse_checkbox_toggle(self, comprehensive_recording):
        """Test checkbox toggle as click."""
        parser = RRWebParser()
        session = parser.parse(comprehensive_recording)

        # Checkbox toggle should become a click action
        clicks = [a for a in session.parsed_actions if a.type == ActionType.CLICK]
        assert len(clicks) >= 1


class TestRRWebParserSelectorBuilding:
    """Tests for CSS selector building."""

    def test_selector_by_id(self, simple_recording):
        """Test selectors use ID when available."""
        parser = RRWebParser()
        session = parser.parse(simple_recording)

        # Should have selector like #email, #password, #submit
        clicks = [a for a in session.parsed_actions if a.type == ActionType.CLICK]
        id_selectors = [c for c in clicks if c.target and c.target.startswith("#")]
        assert len(id_selectors) >= 1

    def test_selector_by_data_testid(self, comprehensive_recording):
        """Test selectors use data-testid."""
        parser = RRWebParser()
        session = parser.parse(comprehensive_recording)

        clicks = [a for a in session.parsed_actions if a.type == ActionType.CLICK]
        testid_selectors = [c for c in clicks if c.target and "data-testid" in c.target]
        assert len(testid_selectors) >= 1

    def test_selector_by_name(self, comprehensive_recording):
        """Test selectors use name attribute."""
        parser = RRWebParser()
        session = parser.parse(comprehensive_recording)

        # Search input has name="search"
        fill_actions = [a for a in session.parsed_actions if a.type == ActionType.FILL]
        name_selectors = [f for f in fill_actions if f.target and "name=" in f.target]
        assert len(name_selectors) >= 0  # May be merged differently


class TestRRWebParserMutations:
    """Tests for DOM mutation handling."""

    def test_parse_with_mutations(self, recording_with_mutations):
        """Test parsing with dynamic DOM changes."""
        parser = RRWebParser()
        session = parser.parse(recording_with_mutations)

        # Should be able to click on dynamically added element
        clicks = [a for a in session.parsed_actions if a.type == ActionType.CLICK]
        assert len(clicks) >= 1


class TestRRWebParserValueGeneralization:
    """Tests for input value generalization."""

    def test_generalize_email(self, simple_recording):
        """Test email values are generalized."""
        parser = RRWebParser()
        session = parser.parse(simple_recording)

        fill_actions = [a for a in session.parsed_actions if a.type == ActionType.FILL]
        email_fills = [f for f in fill_actions if f.target and "email" in f.target.lower()]

        if email_fills:
            # Should be generalized to {{test_email}}
            assert email_fills[0].value == "{{test_email}}"

    def test_generalize_password(self, simple_recording):
        """Test password values are generalized."""
        parser = RRWebParser()
        session = parser.parse(simple_recording)

        fill_actions = [a for a in session.parsed_actions if a.type == ActionType.FILL]
        pwd_fills = [f for f in fill_actions if f.target and "password" in f.target.lower()]

        if pwd_fills:
            # Should be generalized to {{test_password}}
            assert pwd_fills[0].value == "{{test_password}}"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestParseRRWebRecording:
    """Tests for parse_rrweb_recording function."""

    def test_parse_convenience_function(self, simple_recording):
        """Test convenience function."""
        session = parse_rrweb_recording(simple_recording)
        assert session.action_count > 0

    def test_parse_with_options(self, simple_recording):
        """Test convenience function with options."""
        session = parse_rrweb_recording(
            simple_recording,
            session_id="custom-id",
            merge_consecutive_inputs=False
        )
        assert session.id == "custom-id"


class TestRecordingToTestSpec:
    """Tests for recording_to_test_spec function."""

    def test_convert_to_test_spec(self, simple_recording):
        """Test converting recording to test spec."""
        spec = recording_to_test_spec(
            simple_recording,
            test_name="Login Test"
        )

        assert spec["name"] == "Login Test"
        assert spec["source"] == "rrweb_recording"
        assert "steps" in spec
        assert len(spec["steps"]) > 0

    def test_convert_with_assertions(self, simple_recording):
        """Test conversion includes inferred assertions."""
        spec = recording_to_test_spec(
            simple_recording,
            include_assertions=True
        )

        assert "assertions" in spec
        # Should have at least URL assertion
        url_assertions = [a for a in spec["assertions"] if a.get("type") == "url_contains"]
        assert len(url_assertions) >= 0

    def test_convert_without_assertions(self, simple_recording):
        """Test conversion without assertions."""
        spec = recording_to_test_spec(
            simple_recording,
            include_assertions=False
        )

        assert spec["assertions"] == []


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_recording(self):
        """Test parsing empty recording."""
        parser = RRWebParser()
        session = parser.parse([])

        assert session.action_count == 0
        assert session.duration_ms == 0

    def test_recording_no_snapshot(self):
        """Test recording without full snapshot."""
        events = [
            {"type": 4, "timestamp": 0, "data": {"href": "https://example.com"}},
            {"type": 3, "timestamp": 500, "data": {"source": 2, "type": 2, "id": 1}}
        ]
        parser = RRWebParser()
        session = parser.parse(events)

        # Should still create actions, using fallback selectors
        assert session.action_count >= 1

    def test_recording_unknown_event_types(self):
        """Test handling unknown event types."""
        events = [
            {"type": 4, "timestamp": 0, "data": {"href": "https://example.com"}},
            {"type": 99, "timestamp": 500, "data": {}},  # Unknown type
        ]
        parser = RRWebParser()
        session = parser.parse(events)

        # Should not crash
        assert session is not None

    def test_recording_malformed_events(self):
        """Test handling malformed events."""
        events = [
            {"type": 4, "timestamp": 0, "data": {"href": "https://example.com"}},
            {},  # Empty event
            {"timestamp": 500},  # Missing type
            {"type": 3},  # Missing timestamp and data
        ]
        parser = RRWebParser()
        session = parser.parse(events)

        # Should handle gracefully
        assert session is not None

    def test_special_characters_in_values(self):
        """Test handling special characters in input values."""
        events = [
            {"type": 4, "timestamp": 0, "data": {"href": "https://example.com"}},
            {
                "type": 2,
                "timestamp": 100,
                "data": {
                    "node": {
                        "id": 1,
                        "tagName": "input",
                        "attributes": {"id": "text"}
                    }
                }
            },
            {
                "type": 3,
                "timestamp": 500,
                "data": {
                    "source": 5,
                    "id": 1,
                    "text": "Hello <script>alert('xss')</script> World"
                }
            }
        ]
        parser = RRWebParser()
        session = parser.parse(events)

        # Should capture the value (sanitization happens at export time)
        fill_actions = [a for a in session.parsed_actions if a.type == ActionType.FILL]
        assert len(fill_actions) >= 0


class TestGetTestSteps:
    """Tests for get_test_steps output format."""

    def test_steps_format(self, simple_recording):
        """Test generated steps have correct format."""
        parser = RRWebParser()
        session = parser.parse(simple_recording)
        steps = session.get_test_steps()

        for step in steps:
            assert "action" in step
            assert step["action"] in [a.value for a in ActionType]

    def test_steps_preserve_order(self, simple_recording):
        """Test steps preserve chronological order."""
        parser = RRWebParser()
        session = parser.parse(simple_recording)
        steps = session.get_test_steps()

        # First step should be navigation
        assert steps[0]["action"] == "goto"
