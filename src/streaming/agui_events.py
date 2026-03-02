"""AG-UI protocol event type definitions for real-time agent activity updates.

Implements the AG-UI (Agent-User Interaction) protocol events for streaming
agent swarm activity to the dashboard via Server-Sent Events (SSE).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AGUIEventType(str, Enum):
    """AG-UI protocol event types for agent activity streaming."""

    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_ERROR = "run_error"
    STEP_STARTED = "step_started"
    STEP_FINISHED = "step_finished"
    TEXT_MESSAGE_START = "text_message_start"
    TEXT_MESSAGE_CONTENT = "text_message_content"
    TEXT_MESSAGE_END = "text_message_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_ARGS = "tool_call_args"
    TOOL_CALL_END = "tool_call_end"
    STATE_SNAPSHOT = "state_snapshot"
    STATE_DELTA = "state_delta"
    CUSTOM = "custom"


@dataclass
class AGUIEvent:
    """Base AG-UI event with type, timestamp, and SSE serialization."""

    type: AGUIEventType
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_sse_data(self) -> str:
        """Serialize the event to an SSE-compatible JSON string."""
        data = {}
        for k, v in asdict(self).items():
            if isinstance(v, AGUIEventType):
                data[k] = v.value
            else:
                data[k] = v
        return json.dumps(data, default=str)


@dataclass
class RunStartedEvent(AGUIEvent):
    """Emitted when an agent swarm run begins."""

    type: AGUIEventType = field(default=AGUIEventType.RUN_STARTED, init=False)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    swarm_id: str = ""
    thread_id: str = ""
    mode: str = ""
    worker_count: int = 0
    agent_types: list[str] = field(default_factory=list)


@dataclass
class RunFinishedEvent(AGUIEvent):
    """Emitted when an agent swarm run completes successfully."""

    type: AGUIEventType = field(default=AGUIEventType.RUN_FINISHED, init=False)
    run_id: str = ""
    swarm_id: str = ""
    thread_id: str = ""
    success: bool = True
    total_duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    workers_completed: int = 0
    workers_failed: int = 0
    consensus_score: float = 0.0
    summary: str = ""


@dataclass
class RunErrorEvent(AGUIEvent):
    """Emitted when an agent swarm run fails with an error."""

    type: AGUIEventType = field(default=AGUIEventType.RUN_ERROR, init=False)
    run_id: str = ""
    error: str = ""
    code: str = ""


@dataclass
class StepStartedEvent(AGUIEvent):
    """Emitted when an individual agent step begins within a run."""

    type: AGUIEventType = field(default=AGUIEventType.STEP_STARTED, init=False)
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    swarm_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    agent_type: str = ""
    step_name: str = ""
    task: str = ""


@dataclass
class StepFinishedEvent(AGUIEvent):
    """Emitted when an individual agent step completes."""

    type: AGUIEventType = field(default=AGUIEventType.STEP_FINISHED, init=False)
    step_id: str = ""
    run_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    agent_type: str = ""
    success: bool = True
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    findings_count: int = 0
    result_summary: str = ""
    output: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextMessageContentEvent(AGUIEvent):
    """Emitted for streaming text content chunks from an agent."""

    type: AGUIEventType = field(default=AGUIEventType.TEXT_MESSAGE_CONTENT, init=False)
    message_id: str = ""
    delta: str = ""


@dataclass
class ToolCallStartEvent(AGUIEvent):
    """Emitted when an agent begins a tool call."""

    type: AGUIEventType = field(default=AGUIEventType.TOOL_CALL_START, init=False)
    tool_call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    agent_name: str = ""
    run_id: str = ""


@dataclass
class ToolCallArgsEvent(AGUIEvent):
    """Emitted with the arguments of a tool call (may stream incrementally)."""

    type: AGUIEventType = field(default=AGUIEventType.TOOL_CALL_ARGS, init=False)
    tool_call_id: str = ""
    delta: str = ""


@dataclass
class ToolCallEndEvent(AGUIEvent):
    """Emitted when a tool call completes with its result."""

    type: AGUIEventType = field(default=AGUIEventType.TOOL_CALL_END, init=False)
    tool_call_id: str = ""
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateDeltaEvent(AGUIEvent):
    """Emitted with incremental state changes for the swarm."""

    type: AGUIEventType = field(default=AGUIEventType.STATE_DELTA, init=False)
    agent_id: str = ""
    progress: int = 0
    phase: str = ""
    message: str = ""
    delta: list[dict[str, Any]] = field(default_factory=list)
