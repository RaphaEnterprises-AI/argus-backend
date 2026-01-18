"""Tests for recording models."""

from src.recording.models import (
    ActionType,
    MouseInteractionType,
    NodeLookup,
    ParsedAction,
    RecordingMetadata,
    RecordingSession,
    RRWebEvent,
    RRWebEventType,
    RRWebIncrementalSource,
    RRWebMutation,
    RRWebSnapshot,
)

# =============================================================================
# Enum Tests
# =============================================================================


class TestRRWebEventType:
    """Tests for RRWebEventType enum."""

    def test_event_type_values(self):
        """Test event type values match rrweb spec."""
        assert RRWebEventType.DOM_CONTENT_LOADED == 0
        assert RRWebEventType.LOAD == 1
        assert RRWebEventType.FULL_SNAPSHOT == 2
        assert RRWebEventType.INCREMENTAL_SNAPSHOT == 3
        assert RRWebEventType.META == 4
        assert RRWebEventType.CUSTOM == 5
        assert RRWebEventType.PLUGIN == 6


class TestRRWebIncrementalSource:
    """Tests for RRWebIncrementalSource enum."""

    def test_incremental_source_values(self):
        """Test incremental source values."""
        assert RRWebIncrementalSource.MUTATION == 0
        assert RRWebIncrementalSource.MOUSE_MOVE == 1
        assert RRWebIncrementalSource.MOUSE_INTERACTION == 2
        assert RRWebIncrementalSource.SCROLL == 3
        assert RRWebIncrementalSource.INPUT == 5


class TestMouseInteractionType:
    """Tests for MouseInteractionType enum."""

    def test_mouse_interaction_values(self):
        """Test mouse interaction type values."""
        assert MouseInteractionType.CLICK == 2
        assert MouseInteractionType.DBL_CLICK == 4
        assert MouseInteractionType.FOCUS == 5
        assert MouseInteractionType.BLUR == 6


class TestActionType:
    """Tests for ActionType enum."""

    def test_action_type_values(self):
        """Test action type string values."""
        assert ActionType.GOTO.value == "goto"
        assert ActionType.CLICK.value == "click"
        assert ActionType.FILL.value == "fill"
        assert ActionType.TYPE.value == "type"
        assert ActionType.SELECT.value == "select"
        assert ActionType.SCROLL.value == "scroll"


# =============================================================================
# RRWebEvent Tests
# =============================================================================


class TestRRWebEvent:
    """Tests for RRWebEvent dataclass."""

    def test_create_event(self):
        """Test creating an event."""
        event = RRWebEvent(
            type=RRWebEventType.META,
            timestamp=1000,
            data={"href": "https://example.com"}
        )
        assert event.type == RRWebEventType.META
        assert event.timestamp == 1000
        assert event.data["href"] == "https://example.com"

    def test_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            "type": 4,
            "timestamp": 2000,
            "data": {"width": 1920, "height": 1080}
        }
        event = RRWebEvent.from_dict(data)
        assert event.type == RRWebEventType.META
        assert event.timestamp == 2000
        assert event.data["width"] == 1920

    def test_from_dict_defaults(self):
        """Test from_dict with missing fields."""
        event = RRWebEvent.from_dict({})
        assert event.type == RRWebEventType.DOM_CONTENT_LOADED
        assert event.timestamp == 0
        assert event.data == {}


# =============================================================================
# ParsedAction Tests
# =============================================================================


class TestParsedAction:
    """Tests for ParsedAction dataclass."""

    def test_create_action(self):
        """Test creating a parsed action."""
        action = ParsedAction(
            type=ActionType.CLICK,
            target="#button",
            timestamp=1000,
            description="Click button"
        )
        assert action.type == ActionType.CLICK
        assert action.target == "#button"
        assert action.timestamp == 1000

    def test_to_step_dict(self):
        """Test converting action to step dictionary."""
        action = ParsedAction(
            type=ActionType.FILL,
            target="#email",
            value="test@example.com",
            description="Enter email"
        )
        step = action.to_step_dict()

        assert step["action"] == "fill"
        assert step["target"] == "#email"
        assert step["value"] == "test@example.com"
        assert step["description"] == "Enter email"

    def test_to_step_dict_minimal(self):
        """Test step dict with minimal fields."""
        action = ParsedAction(type=ActionType.SCREENSHOT)
        step = action.to_step_dict()

        assert step == {"action": "screenshot"}

    def test_to_step_dict_no_none_values(self):
        """Test step dict excludes None values."""
        action = ParsedAction(
            type=ActionType.CLICK,
            target="#btn",
            value=None,
            description=None
        )
        step = action.to_step_dict()

        assert "value" not in step
        assert "description" not in step


# =============================================================================
# RecordingMetadata Tests
# =============================================================================


class TestRecordingMetadata:
    """Tests for RecordingMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating metadata."""
        meta = RecordingMetadata(
            href="https://example.com/page",
            width=1920,
            height=1080,
            title="Test Page"
        )
        assert meta.href == "https://example.com/page"
        assert meta.width == 1920

    def test_from_meta_event(self):
        """Test creating from rrweb meta event data."""
        data = {
            "href": "https://example.com",
            "width": 1440,
            "height": 900
        }
        meta = RecordingMetadata.from_meta_event(data)

        assert meta.href == "https://example.com"
        assert meta.width == 1440
        assert meta.height == 900

    def test_from_meta_event_defaults(self):
        """Test from_meta_event with missing fields."""
        meta = RecordingMetadata.from_meta_event({})
        assert meta.href == ""
        assert meta.width == 0


# =============================================================================
# RecordingSession Tests
# =============================================================================


class TestRecordingSession:
    """Tests for RecordingSession dataclass."""

    def test_create_session(self):
        """Test creating a recording session."""
        session = RecordingSession(id="test-session")
        assert session.id == "test-session"
        assert session.events == []
        assert session.parsed_actions == []

    def test_action_count(self):
        """Test action count property."""
        session = RecordingSession(id="test")
        session.parsed_actions = [
            ParsedAction(type=ActionType.CLICK, target="#a"),
            ParsedAction(type=ActionType.FILL, target="#b", value="x"),
        ]
        assert session.action_count == 2

    def test_get_test_steps(self):
        """Test converting actions to test steps."""
        session = RecordingSession(id="test")
        session.parsed_actions = [
            ParsedAction(type=ActionType.GOTO, target="https://example.com"),
            ParsedAction(type=ActionType.CLICK, target="#login"),
        ]
        steps = session.get_test_steps()

        assert len(steps) == 2
        assert steps[0]["action"] == "goto"
        assert steps[1]["action"] == "click"


# =============================================================================
# NodeLookup Tests
# =============================================================================


class TestNodeLookup:
    """Tests for NodeLookup class."""

    def test_set_and_get_node(self):
        """Test setting and getting node info."""
        lookup = NodeLookup()
        lookup.set_node(1, "#button", "button", {"class": "btn"})

        assert lookup.get_selector(1) == "#button"
        assert lookup.id_to_tag[1] == "button"
        assert lookup.id_to_attributes[1]["class"] == "btn"

    def test_get_selector_missing(self):
        """Test getting selector for missing node."""
        lookup = NodeLookup()
        assert lookup.get_selector(999) is None

    def test_set_node_minimal(self):
        """Test setting node with minimal info."""
        lookup = NodeLookup()
        lookup.set_node(1, ".element")

        assert lookup.get_selector(1) == ".element"
        assert 1 not in lookup.id_to_tag
        assert 1 not in lookup.id_to_attributes


# =============================================================================
# RRWebSnapshot Tests
# =============================================================================


class TestRRWebSnapshot:
    """Tests for RRWebSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a snapshot."""
        snapshot = RRWebSnapshot(
            node={"tagName": "html", "childNodes": []},
            initial_offset={"top": 0, "left": 0}
        )
        assert snapshot.node["tagName"] == "html"
        assert snapshot.initial_offset["top"] == 0


# =============================================================================
# RRWebMutation Tests
# =============================================================================


class TestRRWebMutation:
    """Tests for RRWebMutation dataclass."""

    def test_create_mutation(self):
        """Test creating a mutation."""
        mutation = RRWebMutation(
            adds=[{"parentId": 1, "node": {"tagName": "div"}}],
            removes=[{"parentId": 2, "id": 3}],
        )
        assert len(mutation.adds) == 1
        assert len(mutation.removes) == 1

    def test_mutation_defaults(self):
        """Test mutation defaults."""
        mutation = RRWebMutation()
        assert mutation.adds == []
        assert mutation.removes == []
        assert mutation.texts == []
        assert mutation.attributes == []
