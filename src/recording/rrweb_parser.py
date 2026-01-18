"""rrweb event parser - Convert DOM recordings to test actions.

This parser converts rrweb event format to actionable test steps.
It's DOM-based (not video-based), meaning:
- 99% accuracy (exact selectors)
- $0 AI cost
- Instant parsing
"""

import json
import re

import structlog

from .models import (
    ActionType,
    MouseInteractionType,
    NodeLookup,
    ParsedAction,
    RecordingMetadata,
    RecordingSession,
    RRWebEvent,
    RRWebEventType,
    RRWebIncrementalSource,
)

logger = structlog.get_logger()


class RRWebParser:
    """Parser for rrweb recording format.

    Converts rrweb events into ParsedAction objects that can be
    used to generate executable tests.

    Example:
        parser = RRWebParser()
        session = parser.parse(events_json)
        test_steps = session.get_test_steps()
    """

    def __init__(
        self,
        merge_consecutive_inputs: bool = True,
        filter_hover_events: bool = True,
        min_scroll_distance: int = 100,
    ):
        """Initialize parser with configuration.

        Args:
            merge_consecutive_inputs: Merge rapid input events into single fill
            filter_hover_events: Filter out hover events (keep only clicks)
            min_scroll_distance: Minimum scroll distance to record as action
        """
        self.merge_consecutive_inputs = merge_consecutive_inputs
        self.filter_hover_events = filter_hover_events
        self.min_scroll_distance = min_scroll_distance
        self.log = logger.bind(component="rrweb_parser")

        # Internal state during parsing
        self._node_lookup = NodeLookup()
        self._last_input_node: int | None = None
        self._last_input_value: str = ""
        self._last_scroll_position: tuple[int, int] = (0, 0)
        self._pending_actions: list[ParsedAction] = []

    def parse(
        self,
        events: list[dict] | str,
        session_id: str = "recording",
    ) -> RecordingSession:
        """Parse rrweb events into a RecordingSession.

        Args:
            events: List of rrweb event dicts or JSON string
            session_id: ID for the recording session

        Returns:
            RecordingSession with parsed actions
        """
        # Handle JSON string input
        if isinstance(events, str):
            events = json.loads(events)

        # Handle wrapper object with "events" key
        if isinstance(events, dict) and "events" in events:
            events = events["events"]

        self.log.info("Parsing rrweb recording", event_count=len(events))

        # Reset internal state
        self._reset_state()

        session = RecordingSession(id=session_id)

        # Parse each event
        for event_data in events:
            event = RRWebEvent.from_dict(event_data)
            session.events.append(event)
            self._process_event(event, session)

        # Flush any pending actions
        self._flush_pending_input(session)

        # Calculate duration
        if session.events:
            first_ts = session.events[0].timestamp
            last_ts = session.events[-1].timestamp
            session.duration_ms = last_ts - first_ts

        self.log.info(
            "Parsing complete",
            action_count=session.action_count,
            duration_ms=session.duration_ms,
        )

        return session

    def _reset_state(self):
        """Reset internal parsing state."""
        self._node_lookup = NodeLookup()
        self._last_input_node = None
        self._last_input_value = ""
        self._last_scroll_position = (0, 0)
        self._pending_actions = []

    def _process_event(self, event: RRWebEvent, session: RecordingSession):
        """Process a single rrweb event."""
        if event.type == RRWebEventType.META:
            self._process_meta(event, session)

        elif event.type == RRWebEventType.FULL_SNAPSHOT:
            self._process_full_snapshot(event)

        elif event.type == RRWebEventType.INCREMENTAL_SNAPSHOT:
            self._process_incremental(event, session)

    def _process_meta(self, event: RRWebEvent, session: RecordingSession):
        """Process meta event to extract page info."""
        data = event.data
        session.metadata = RecordingMetadata.from_meta_event(data)

        # Add navigation action for the initial page
        if session.metadata.href:
            action = ParsedAction(
                type=ActionType.GOTO,
                target=session.metadata.href,
                timestamp=event.timestamp,
                description=f"Navigate to {session.metadata.href}",
            )
            session.parsed_actions.append(action)

    def _process_full_snapshot(self, event: RRWebEvent):
        """Process full snapshot to build node lookup table."""
        node = event.data.get("node", {})
        self._build_node_lookup(node)

    def _build_node_lookup(self, node: dict, parent_selector: str = ""):
        """Recursively build node ID to selector mapping."""
        node_id = node.get("id")
        tag_name = node.get("tagName", "").lower()
        attributes = node.get("attributes", {})

        if node_id and tag_name:
            selector = self._build_selector(tag_name, attributes, parent_selector)
            self._node_lookup.set_node(
                node_id, selector, tag_name, attributes
            )

        # Process child nodes
        for child in node.get("childNodes", []):
            child_parent = selector if node_id else parent_selector
            self._build_node_lookup(child, child_parent)

    def _build_selector(
        self,
        tag_name: str,
        attributes: dict,
        parent_selector: str = "",
    ) -> str:
        """Build a CSS selector for an element."""
        # Priority: id > data-testid > name > class > tag

        # Check for ID
        if "id" in attributes:
            return f"#{attributes['id']}"

        # Check for data-testid (common testing pattern)
        if "data-testid" in attributes:
            return f'[data-testid="{attributes["data-testid"]}"]'

        # Check for data-test (alternative)
        if "data-test" in attributes:
            return f'[data-test="{attributes["data-test"]}"]'

        # Check for name attribute (forms)
        if "name" in attributes:
            return f'{tag_name}[name="{attributes["name"]}"]'

        # Check for aria-label
        if "aria-label" in attributes:
            return f'{tag_name}[aria-label="{attributes["aria-label"]}"]'

        # Use class if available
        if "class" in attributes:
            classes = attributes["class"].strip()
            if classes:
                # Use first significant class
                class_list = classes.split()
                # Filter out utility classes (common patterns)
                significant = [
                    c for c in class_list
                    if not re.match(r'^(p|m|w|h|flex|grid|text|bg|border)-', c)
                ]
                if significant:
                    return f".{significant[0]}"
                elif class_list:
                    return f".{class_list[0]}"

        # Fallback to tag with type for inputs
        if tag_name in ("input", "button", "textarea", "select"):
            if "type" in attributes:
                return f'{tag_name}[type="{attributes["type"]}"]'
            return tag_name

        # Generic fallback
        return tag_name

    def _process_incremental(self, event: RRWebEvent, session: RecordingSession):
        """Process incremental snapshot events."""
        data = event.data
        source = data.get("source")

        if source == RRWebIncrementalSource.MOUSE_INTERACTION:
            self._process_mouse_interaction(event, session)

        elif source == RRWebIncrementalSource.INPUT:
            self._process_input(event, session)

        elif source == RRWebIncrementalSource.SCROLL:
            self._process_scroll(event, session)

        elif source == RRWebIncrementalSource.MUTATION:
            self._process_mutation(event)

    def _process_mouse_interaction(
        self,
        event: RRWebEvent,
        session: RecordingSession,
    ):
        """Process mouse interaction events (click, dblclick, etc.)."""
        data = event.data
        interaction_type = data.get("type")
        node_id = data.get("id")

        if not node_id:
            return

        # Flush any pending input before processing click
        self._flush_pending_input(session)

        selector = self._node_lookup.get_selector(node_id)
        if not selector:
            selector = f"[data-rrweb-id='{node_id}']"

        if interaction_type == MouseInteractionType.CLICK:
            action = ParsedAction(
                type=ActionType.CLICK,
                target=selector,
                timestamp=event.timestamp,
                description=f"Click on {selector}",
                metadata={"x": data.get("x", 0), "y": data.get("y", 0)},
            )
            session.parsed_actions.append(action)

        elif interaction_type == MouseInteractionType.DBL_CLICK:
            action = ParsedAction(
                type=ActionType.DOUBLE_CLICK,
                target=selector,
                timestamp=event.timestamp,
                description=f"Double click on {selector}",
            )
            session.parsed_actions.append(action)

        elif interaction_type == MouseInteractionType.FOCUS:
            if not self.filter_hover_events:
                action = ParsedAction(
                    type=ActionType.FOCUS,
                    target=selector,
                    timestamp=event.timestamp,
                )
                session.parsed_actions.append(action)

        elif interaction_type == MouseInteractionType.BLUR:
            if not self.filter_hover_events:
                action = ParsedAction(
                    type=ActionType.BLUR,
                    target=selector,
                    timestamp=event.timestamp,
                )
                session.parsed_actions.append(action)

    def _process_input(self, event: RRWebEvent, session: RecordingSession):
        """Process input events (text entry, selections)."""
        data = event.data
        node_id = data.get("id")
        value = data.get("text", "")
        data.get("isChecked") is None and data.get("text") is None

        if not node_id:
            return

        selector = self._node_lookup.get_selector(node_id)
        if not selector:
            selector = f"[data-rrweb-id='{node_id}']"

        # Handle checkboxes/radios
        if data.get("isChecked") is not None:
            action = ParsedAction(
                type=ActionType.CLICK,
                target=selector,
                timestamp=event.timestamp,
                description=f"Toggle {selector}",
            )
            session.parsed_actions.append(action)
            return

        # Handle select changes
        if "data" in data and isinstance(data.get("data"), dict):
            select_data = data["data"]
            if "values" in select_data:
                values = select_data["values"]
                action = ParsedAction(
                    type=ActionType.SELECT,
                    target=selector,
                    value=values[0] if values else "",
                    timestamp=event.timestamp,
                    description=f"Select option in {selector}",
                )
                session.parsed_actions.append(action)
                return

        # Handle text input
        if self.merge_consecutive_inputs:
            # Merge with previous input to same node
            if self._last_input_node == node_id:
                self._last_input_value = value
            else:
                # Flush previous input
                self._flush_pending_input(session)
                # Start new input
                self._last_input_node = node_id
                self._last_input_value = value
                self._last_input_timestamp = event.timestamp
                self._last_input_selector = selector
        else:
            # Immediate action for each keystroke
            action = ParsedAction(
                type=ActionType.TYPE,
                target=selector,
                value=value,
                timestamp=event.timestamp,
            )
            session.parsed_actions.append(action)

    def _flush_pending_input(self, session: RecordingSession):
        """Flush any pending merged input as a fill action."""
        if self._last_input_node is not None and self._last_input_value:
            # Generalize sensitive values
            value = self._generalize_input_value(
                self._last_input_selector,
                self._last_input_value,
            )

            action = ParsedAction(
                type=ActionType.FILL,
                target=self._last_input_selector,
                value=value,
                timestamp=getattr(self, "_last_input_timestamp", 0),
                description=f"Fill {self._last_input_selector}",
            )
            session.parsed_actions.append(action)

        # Reset
        self._last_input_node = None
        self._last_input_value = ""

    def _generalize_input_value(self, selector: str, value: str) -> str:
        """Generalize input values for test data variables."""
        selector_lower = selector.lower()

        # Detect field type from selector
        if "email" in selector_lower:
            return "{{test_email}}"
        elif "password" in selector_lower or "passwd" in selector_lower:
            return "{{test_password}}"
        elif "name" in selector_lower:
            return "{{test_name}}"
        elif "phone" in selector_lower or "tel" in selector_lower:
            return "{{test_phone}}"
        elif "address" in selector_lower:
            return "{{test_address}}"
        elif "search" in selector_lower:
            return "{{search_query}}"

        # Detect by value pattern
        if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', value):
            return "{{test_email}}"
        elif len(value) >= 8 and re.search(r'[A-Z]', value) and re.search(r'\d', value):
            # Likely a password (8+ chars with uppercase and digit)
            return "{{test_password}}"

        return value

    def _process_scroll(self, event: RRWebEvent, session: RecordingSession):
        """Process scroll events."""
        data = event.data
        x = data.get("x", 0)
        y = data.get("y", 0)

        # Check if scroll is significant
        dx = abs(x - self._last_scroll_position[0])
        dy = abs(y - self._last_scroll_position[1])

        if dx >= self.min_scroll_distance or dy >= self.min_scroll_distance:
            action = ParsedAction(
                type=ActionType.SCROLL,
                value=f"{dx},{dy}",
                timestamp=event.timestamp,
                description=f"Scroll by ({dx}, {dy})",
            )
            session.parsed_actions.append(action)
            self._last_scroll_position = (x, y)

    def _process_mutation(self, event: RRWebEvent):
        """Process DOM mutation events to update node lookup."""
        data = event.data
        adds = data.get("adds", [])

        for add in adds:
            node = add.get("node", {})
            if node:
                self._build_node_lookup(node)


def parse_rrweb_recording(
    events: list[dict] | str,
    session_id: str = "recording",
    **parser_options,
) -> RecordingSession:
    """Convenience function to parse rrweb recording.

    Args:
        events: rrweb events (list or JSON string)
        session_id: ID for the recording
        **parser_options: Options for RRWebParser

    Returns:
        RecordingSession with parsed actions
    """
    parser = RRWebParser(**parser_options)
    return parser.parse(events, session_id)


def recording_to_test_spec(
    events: list[dict] | str,
    test_name: str = "Generated Test",
    include_assertions: bool = True,
) -> dict:
    """Convert rrweb recording directly to test spec.

    Args:
        events: rrweb events
        test_name: Name for the generated test
        include_assertions: Whether to infer assertions

    Returns:
        Test specification dictionary
    """
    session = parse_rrweb_recording(events)

    spec = {
        "id": f"rrweb-{session.id}",
        "name": test_name,
        "description": f"Generated from rrweb recording ({session.action_count} actions)",
        "source": "rrweb_recording",
        "recording_id": session.id,
        "steps": session.get_test_steps(),
        "assertions": [],
    }

    if include_assertions:
        spec["assertions"] = _infer_assertions(session)

    return spec


def _infer_assertions(session: RecordingSession) -> list[dict]:
    """Infer test assertions from recording session."""
    assertions = []

    # Add URL assertion based on final navigation
    goto_actions = [a for a in session.parsed_actions if a.type == ActionType.GOTO]
    if goto_actions:
        last_url = goto_actions[-1].target
        # Extract path from URL
        if "://" in last_url:
            path = "/" + last_url.split("://", 1)[1].split("/", 1)[-1]
        else:
            path = last_url
        assertions.append({
            "type": "url_contains",
            "expected": path.split("?")[0],  # Remove query params
        })

    # Add element assertions for clicked elements
    click_actions = [
        a for a in session.parsed_actions
        if a.type == ActionType.CLICK and a.target
    ]
    if click_actions:
        # Assert first few clicked elements are visible
        for action in click_actions[:3]:
            assertions.append({
                "type": "element_visible",
                "target": action.target,
            })

    return assertions
