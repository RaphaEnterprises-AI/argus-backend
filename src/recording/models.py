"""Data models for rrweb recording parsing."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, Enum
from typing import Any, Optional


class RRWebEventType(IntEnum):
    """rrweb event types."""

    DOM_CONTENT_LOADED = 0
    LOAD = 1
    FULL_SNAPSHOT = 2
    INCREMENTAL_SNAPSHOT = 3
    META = 4
    CUSTOM = 5
    PLUGIN = 6


class RRWebIncrementalSource(IntEnum):
    """rrweb incremental snapshot sources."""

    MUTATION = 0
    MOUSE_MOVE = 1
    MOUSE_INTERACTION = 2
    SCROLL = 3
    VIEWPORT_RESIZE = 4
    INPUT = 5
    TOUCH_MOVE = 6
    MEDIA_INTERACTION = 7
    STYLE_SHEET_RULE = 8
    CANVAS_MUTATION = 9
    FONT = 10
    LOG = 11
    DRAG = 12
    STYLE_DECLARATION = 13
    SELECTION = 14
    ADOPTED_STYLE_SHEET = 15


class MouseInteractionType(IntEnum):
    """Mouse interaction types in rrweb."""

    MOUSE_UP = 0
    MOUSE_DOWN = 1
    CLICK = 2
    CONTEXT_MENU = 3
    DBL_CLICK = 4
    FOCUS = 5
    BLUR = 6
    TOUCH_START = 7
    TOUCH_MOVE_DEPARTED = 8
    TOUCH_END = 9
    TOUCH_CANCEL = 10


class ActionType(str, Enum):
    """Parsed action types for test generation."""

    GOTO = "goto"
    CLICK = "click"
    FILL = "fill"
    TYPE = "type"
    SELECT = "select"
    SCROLL = "scroll"
    HOVER = "hover"
    WAIT = "wait"
    DOUBLE_CLICK = "double_click"
    FOCUS = "focus"
    BLUR = "blur"
    PRESS_KEY = "press_key"
    DRAG = "drag"
    SCREENSHOT = "screenshot"


@dataclass
class RRWebEvent:
    """A single rrweb event."""

    type: RRWebEventType
    timestamp: int  # Milliseconds since recording start
    data: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RRWebEvent":
        """Create RRWebEvent from dictionary."""
        event_type = data.get("type", 0)
        # Handle unknown event types gracefully
        try:
            event_type = RRWebEventType(event_type)
        except ValueError:
            # Map unknown types to CUSTOM
            event_type = RRWebEventType.CUSTOM

        return cls(
            type=event_type,
            timestamp=data.get("timestamp", 0),
            data=data.get("data", {}),
        )


@dataclass
class RRWebSnapshot:
    """Full DOM snapshot from rrweb."""

    node: dict  # DOM tree structure
    initial_offset: dict = field(default_factory=dict)


@dataclass
class RRWebMutation:
    """DOM mutation event."""

    adds: list[dict] = field(default_factory=list)
    removes: list[dict] = field(default_factory=list)
    texts: list[dict] = field(default_factory=list)
    attributes: list[dict] = field(default_factory=list)


@dataclass
class ParsedAction:
    """A parsed user action ready for test generation."""

    type: ActionType
    target: Optional[str] = None  # CSS selector
    value: Optional[str] = None
    timestamp: int = 0  # ms since recording start
    description: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_step_dict(self) -> dict:
        """Convert to test step dictionary format."""
        step = {"action": self.type.value}
        if self.target:
            step["target"] = self.target
        if self.value:
            step["value"] = self.value
        if self.description:
            step["description"] = self.description
        return step


@dataclass
class RecordingMetadata:
    """Metadata about a recording session."""

    href: str = ""
    width: int = 0
    height: int = 0
    title: str = ""
    user_agent: str = ""
    recorded_at: Optional[datetime] = None

    @classmethod
    def from_meta_event(cls, data: dict) -> "RecordingMetadata":
        """Create metadata from rrweb meta event."""
        return cls(
            href=data.get("href", ""),
            width=data.get("width", 0),
            height=data.get("height", 0),
        )


@dataclass
class RecordingSession:
    """A complete recording session with parsed events."""

    id: str
    events: list[RRWebEvent] = field(default_factory=list)
    metadata: RecordingMetadata = field(default_factory=RecordingMetadata)
    parsed_actions: list[ParsedAction] = field(default_factory=list)
    duration_ms: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def action_count(self) -> int:
        """Get count of parsed actions."""
        return len(self.parsed_actions)

    def get_test_steps(self) -> list[dict]:
        """Convert parsed actions to test steps."""
        return [action.to_step_dict() for action in self.parsed_actions]


@dataclass
class NodeLookup:
    """Lookup table for rrweb node IDs to selectors."""

    id_to_selector: dict[int, str] = field(default_factory=dict)
    id_to_tag: dict[int, str] = field(default_factory=dict)
    id_to_attributes: dict[int, dict] = field(default_factory=dict)

    def get_selector(self, node_id: int) -> Optional[str]:
        """Get CSS selector for a node ID."""
        return self.id_to_selector.get(node_id)

    def set_node(
        self,
        node_id: int,
        selector: str,
        tag: str = "",
        attributes: dict = None,
    ):
        """Set node information."""
        self.id_to_selector[node_id] = selector
        if tag:
            self.id_to_tag[node_id] = tag
        if attributes:
            self.id_to_attributes[node_id] = attributes
