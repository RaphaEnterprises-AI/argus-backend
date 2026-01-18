"""Browser recording module - Convert rrweb recordings to test specs.

This module provides DOM-based recording conversion that is:
- 99% accurate (exact selectors from DOM)
- $0 AI cost (no vision model needed)
- Instant parsing (no video processing)
- Privacy-first (sensitive fields masked at recording time)
"""

from .models import (
    ActionType,
    ParsedAction,
    RecordingMetadata,
    RecordingSession,
    RRWebEvent,
    RRWebEventType,
    RRWebIncrementalSource,
    RRWebMutation,
    RRWebSnapshot,
)
from .recorder_snippet import RecorderSnippetGenerator
from .rrweb_parser import RRWebParser

__all__ = [
    # Models
    "RRWebEventType",
    "RRWebIncrementalSource",
    "RRWebEvent",
    "RRWebSnapshot",
    "RRWebMutation",
    "RecordingSession",
    "RecordingMetadata",
    "ParsedAction",
    "ActionType",
    # Parser
    "RRWebParser",
    # Snippet Generator
    "RecorderSnippetGenerator",
]
