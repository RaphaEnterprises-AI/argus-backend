"""Recording API endpoints for browser recording to test conversion."""

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.api.teams import get_current_user

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/recording", tags=["Recording"])


# =============================================================================
# In-memory Storage (would be database in production)
# =============================================================================

_recordings: dict[str, dict] = {}


# =============================================================================
# Enums
# =============================================================================


class RRWebEventType(int, Enum):
    """rrweb event types."""

    DOM_CONTENT_LOADED = 0
    LOAD = 1
    FULL_SNAPSHOT = 2
    INCREMENTAL_SNAPSHOT = 3
    META = 4
    CUSTOM = 5
    PLUGIN = 6


class IncrementalSource(int, Enum):
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
    ADOPT_STYLE_SHEET = 15


class MouseInteraction(int, Enum):
    """rrweb mouse interaction types."""

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


# =============================================================================
# Request/Response Models
# =============================================================================


class RRWebEvent(BaseModel):
    """Single rrweb event."""

    type: int
    data: dict
    timestamp: int


class RecordingMetadata(BaseModel):
    """Metadata about the recording."""

    duration: int = Field(..., description="Recording duration in milliseconds")
    start_time: str = Field(..., description="ISO timestamp when recording started")
    url: str | None = Field(None, description="URL where recording was made")
    user_agent: str | None = Field(None, description="Browser user agent")
    viewport: dict | None = Field(None, description="Viewport dimensions")


class RecordingUploadRequest(BaseModel):
    """Request to upload a browser recording.

    Size limits are enforced to prevent DoS via memory exhaustion:
    - Maximum 50,000 events per recording
    - Maximum 50MB estimated payload size
    """

    events: list[RRWebEvent] = Field(
        ...,
        description="List of rrweb events",
        max_length=50000  # Max 50K events to prevent memory exhaustion
    )
    metadata: RecordingMetadata = Field(..., description="Recording metadata")
    project_id: str | None = Field(None, description="Project to associate recording with")
    name: str | None = Field(None, description="Name for the recording")

    @field_validator("events")
    @classmethod
    def validate_events_payload_size(cls, v: list[RRWebEvent]) -> list[RRWebEvent]:
        """Validate that the events payload is not too large.

        Estimates payload size by sampling events to avoid O(n) serialization.
        """
        if not v:
            return v

        # Sample-based size estimation for performance
        # Check first, middle, and last 10 events to estimate average size
        sample_size = min(30, len(v))
        if len(v) > 30:
            sample_indices = list(range(10)) + list(range(len(v)//2 - 5, len(v)//2 + 5)) + list(range(len(v) - 10, len(v)))
            sample = [v[i] for i in sample_indices]
        else:
            sample = v

        # Estimate average event size
        total_sample_size = sum(len(str(e.data)) for e in sample)
        avg_event_size = total_sample_size / sample_size

        # Estimate total payload size
        estimated_total = avg_event_size * len(v)

        # 50MB limit
        max_payload_bytes = 50 * 1024 * 1024
        if estimated_total > max_payload_bytes:
            raise ValueError(
                f"Recording payload too large. Estimated size: {estimated_total / 1024 / 1024:.1f}MB, "
                f"maximum allowed: 50MB. Try recording a shorter session."
            )

        return v


class RecordingUploadResponse(BaseModel):
    """Response from recording upload."""

    success: bool
    recording_id: str
    events_count: int
    duration_ms: int
    estimated_steps: int
    message: str
    error: str | None = None


class TestStepModel(BaseModel):
    """Generated test step."""

    action: str
    target: str | None = None
    value: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: int | None = None


class TestAssertionModel(BaseModel):
    """Generated test assertion."""

    type: str
    target: str | None = None
    expected: str | None = None


class ConvertRequest(BaseModel):
    """Request to convert recording to test."""

    recording_id: str = Field(..., description="ID of uploaded recording")
    test_name: str | None = Field(None, description="Name for generated test")
    include_waits: bool = Field(True, description="Include wait steps for timing")
    include_scrolls: bool = Field(False, description="Include scroll actions")
    min_wait_threshold: int = Field(500, description="Minimum pause to include as wait (ms)")
    generalize_data: bool = Field(True, description="Replace specific data with variables")


class ConvertResponse(BaseModel):
    """Response from recording conversion."""

    success: bool
    test: dict | None = None
    recording_id: str
    duration_ms: int
    steps_generated: int
    assertions_generated: int
    warnings: list[str] = []
    error: str | None = None


class RecordingReplayResponse(BaseModel):
    """Response for replay data request."""

    success: bool
    recording_id: str
    events: list[dict] = []
    metadata: dict | None = None
    error: str | None = None


class RecorderSnippetRequest(BaseModel):
    """Request to generate recorder JavaScript snippet."""

    project_id: str | None = Field(None, description="Project ID to associate recordings")
    upload_url: str | None = Field(None, description="URL to upload recordings to")
    options: dict | None = Field(None, description="rrweb recorder options")


class RecorderSnippetResponse(BaseModel):
    """Response with recorder snippet."""

    success: bool
    snippet: str
    cdn_script: str
    instructions: str


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/upload", response_model=RecordingUploadResponse)
async def upload_recording(request: Request, body: RecordingUploadRequest):
    """
    Upload a browser recording (rrweb format).

    Stores the recording for later conversion to test specification.
    Requires authentication.
    """
    # Authenticate the request
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        recording_id = str(uuid4())

        # Analyze events to estimate test steps
        interaction_events = _count_interaction_events(body.events)

        # Store recording with user context
        _recordings[recording_id] = {
            "id": recording_id,
            "events": [e.model_dump() for e in body.events],
            "metadata": body.metadata.model_dump(),
            "project_id": body.project_id,
            "name": body.name or f"Recording {recording_id[:8]}",
            "created_at": datetime.now(UTC).isoformat(),
            "events_count": len(body.events),
            "interaction_count": interaction_events,
            "user_id": user.get("user_id"),
            "organization_id": user.get("organization_id"),
        }

        logger.info(
            "Recording uploaded",
            recording_id=recording_id,
            events=len(body.events),
            duration=body.metadata.duration,
            interactions=interaction_events,
            user_id=user.get("user_id"),
        )

        return RecordingUploadResponse(
            success=True,
            recording_id=recording_id,
            events_count=len(body.events),
            duration_ms=body.metadata.duration,
            estimated_steps=interaction_events,
            message=f"Recording uploaded successfully. {interaction_events} user interactions detected.",
        )

    except Exception as e:
        logger.exception("Recording upload failed", error=str(e))
        return RecordingUploadResponse(
            success=False,
            recording_id="",
            events_count=0,
            duration_ms=0,
            estimated_steps=0,
            message="",
            error=str(e),
        )


@router.post("/convert", response_model=ConvertResponse)
async def convert_recording(request: Request, body: ConvertRequest):
    """
    Convert a browser recording to a test specification.

    Analyzes rrweb events and generates executable test steps.
    Requires authentication.
    """
    # Authenticate the request
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Get recording
        recording = _recordings.get(body.recording_id)
        if not recording:
            raise HTTPException(status_code=404, detail="Recording not found")

        events = recording["events"]
        metadata = recording["metadata"]

        # Parse events into test steps
        steps, assertions, warnings = _parse_rrweb_events(
            events,
            include_waits=body.include_waits,
            include_scrolls=body.include_scrolls,
            min_wait_threshold=body.min_wait_threshold,
            generalize_data=body.generalize_data,
        )

        # Generate test name
        test_name = body.test_name or recording.get(
            "name", f"Test from recording {body.recording_id[:8]}"
        )

        # Build test specification
        test_spec = {
            "id": f"test-{uuid4().hex[:12]}",
            "name": test_name,
            "description": "Auto-generated from browser recording",
            "source": "rrweb_recording",
            "recording_id": body.recording_id,
            "steps": [s.model_dump() for s in steps],
            "assertions": [a.model_dump() for a in assertions],
            "metadata": {
                "generated_at": datetime.now(UTC).isoformat(),
                "recording_duration_ms": metadata.get("duration", 0),
                "recording_url": metadata.get("url"),
                "original_events": len(events),
            },
        }

        logger.info(
            "Recording converted to test",
            recording_id=body.recording_id,
            steps=len(steps),
            assertions=len(assertions),
            warnings=len(warnings),
            user_id=user.get("user_id"),
        )

        return ConvertResponse(
            success=True,
            test=test_spec,
            recording_id=body.recording_id,
            duration_ms=metadata.get("duration", 0),
            steps_generated=len(steps),
            assertions_generated=len(assertions),
            warnings=warnings,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Recording conversion failed", error=str(e))
        return ConvertResponse(
            success=False,
            recording_id=body.recording_id,
            duration_ms=0,
            steps_generated=0,
            assertions_generated=0,
            error=str(e),
        )


@router.get("/replay/{recording_id}", response_model=RecordingReplayResponse)
async def get_replay_data(request: Request, recording_id: str):
    """
    Get recording data for replay.

    Returns the rrweb events for client-side replay.
    Requires authentication.
    """
    # Authenticate the request
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        recording = _recordings.get(recording_id)
        if not recording:
            raise HTTPException(status_code=404, detail="Recording not found")

        # Verify user has access to this recording (same org or owner)
        recording_org = recording.get("organization_id")
        user_org = user.get("organization_id")
        recording_user = recording.get("user_id")
        current_user = user.get("user_id")

        if recording_org and user_org and recording_org != user_org:
            if recording_user != current_user:
                raise HTTPException(status_code=403, detail="Access denied to this recording")

        return RecordingReplayResponse(
            success=True,
            recording_id=recording_id,
            events=recording["events"],
            metadata=recording["metadata"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get replay data", error=str(e))
        return RecordingReplayResponse(
            success=False,
            recording_id=recording_id,
            error=str(e),
        )


@router.post("/snippet", response_model=RecorderSnippetResponse)
async def generate_recorder_snippet(request: Request, body: RecorderSnippetRequest):
    """
    Generate JavaScript snippet for browser recording.

    Returns code to embed in websites for recording user sessions.
    Requires authentication.
    """
    # Authenticate the request
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        upload_url = body.upload_url or "/api/v1/recording/upload"

        # Default rrweb options
        options = body.options or {}
        options.setdefault("checkoutEveryNms", 10000)  # Checkpoint every 10s
        options.setdefault("blockClass", "rr-block")
        options.setdefault("maskAllInputs", False)  # Don't mask for test generation
        options.setdefault("maskInputOptions", {"password": True})

        # Generate the snippet
        snippet = f"""
(function() {{
  // Load rrweb from CDN
  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/rrweb@2.0.0-alpha.11/dist/rrweb.min.js';
  script.onload = function() {{
    let events = [];
    let startTime = new Date().toISOString();

    // Start recording
    const stopFn = rrweb.record({{
      emit(event) {{
        events.push(event);
      }},
      checkoutEveryNms: {options.get("checkoutEveryNms", 10000)},
      blockClass: '{options.get("blockClass", "rr-block")}',
      maskAllInputs: {str(options.get("maskAllInputs", False)).lower()},
      maskInputOptions: {{ password: true }},
    }});

    // Stop and upload function
    window.argusStopRecording = function() {{
      stopFn();
      const duration = events.length > 0
        ? events[events.length - 1].timestamp - events[0].timestamp
        : 0;

      return fetch('{upload_url}', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          events: events,
          metadata: {{
            duration: duration,
            start_time: startTime,
            url: window.location.href,
            user_agent: navigator.userAgent,
            viewport: {{
              width: window.innerWidth,
              height: window.innerHeight,
            }},
          }},
          project_id: '{body.project_id or ""}' || undefined,
        }}),
      }})
      .then(r => r.json())
      .then(data => {{
        console.log('Recording uploaded:', data);
        return data;
      }});
    }};

    console.log('Argus recording started. Call argusStopRecording() to stop and upload.');
  }};
  document.head.appendChild(script);
}})();
"""

        cdn_script = "https://cdn.jsdelivr.net/npm/rrweb@2.0.0-alpha.11/dist/rrweb.min.js"

        instructions = """
## How to Use the Recording Snippet

1. **Add to your page**: Copy the snippet and add it to your webpage, either:
   - In the `<head>` section
   - Via browser console for quick testing
   - Through a bookmarklet

2. **Perform your test flow**: Navigate through your application normally.
   All user interactions will be recorded.

3. **Stop and upload**: When done, call `argusStopRecording()` in the browser console.
   This will upload the recording and return the recording ID.

4. **Convert to test**: Use the /convert endpoint with the recording ID to
   generate a test specification.

## Privacy Notes
- Passwords are masked by default
- Add `rr-block` class to sensitive elements to exclude them
- Recording happens client-side; events are sent to your specified endpoint

## Example Usage
```javascript
// Start recording (automatic when snippet loads)

// ... perform user actions ...

// Stop and upload
argusStopRecording().then(result => {
  console.log('Recording ID:', result.recording_id);
});
```
"""

        return RecorderSnippetResponse(
            success=True,
            snippet=snippet.strip(),
            cdn_script=cdn_script,
            instructions=instructions,
        )

    except Exception as e:
        logger.exception("Failed to generate recorder snippet", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recordings")
async def get_recordings(
    request: Request,
    project_id: str | None = None,
    limit: int = 20,
):
    """
    List uploaded recordings for the authenticated user/organization.

    Returns a list of recordings with metadata. Supports filtering by project_id.
    Requires authentication.
    """
    # Authenticate the request
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    user_org = user.get("organization_id")
    current_user = user.get("user_id")

    recordings = list(_recordings.values())

    # Filter by organization - only show recordings from same org or owned by user
    recordings = [
        r
        for r in recordings
        if r.get("organization_id") == user_org
        or r.get("user_id") == current_user
        or not r.get("organization_id")  # Legacy recordings without org
    ]

    if project_id:
        recordings = [r for r in recordings if r.get("project_id") == project_id]

    # Sort by created_at descending
    recordings.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return {
        "success": True,
        "recordings": [
            {
                "id": r["id"],
                "name": r.get("name"),
                "project_id": r.get("project_id"),
                "events_count": r.get("events_count", 0),
                "interaction_count": r.get("interaction_count", 0),
                "duration_ms": r.get("metadata", {}).get("duration", 0),
                "url": r.get("metadata", {}).get("url"),
                "created_at": r.get("created_at"),
            }
            for r in recordings[:limit]
        ],
        "total": len(recordings),
    }


@router.get("/list")
async def list_recordings(
    request: Request,
    project_id: str | None = None,
    limit: int = 20,
):
    """
    List uploaded recordings.
    Requires authentication. Only shows recordings from user's organization.

    Note: This endpoint is an alias for /recordings and is maintained for backward compatibility.
    """
    # Authenticate the request
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    user_org = user.get("organization_id")
    current_user = user.get("user_id")

    recordings = list(_recordings.values())

    # Filter by organization - only show recordings from same org or owned by user
    recordings = [
        r
        for r in recordings
        if r.get("organization_id") == user_org
        or r.get("user_id") == current_user
        or not r.get("organization_id")  # Legacy recordings without org
    ]

    if project_id:
        recordings = [r for r in recordings if r.get("project_id") == project_id]

    # Sort by created_at descending
    recordings.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return {
        "success": True,
        "recordings": [
            {
                "id": r["id"],
                "name": r.get("name"),
                "project_id": r.get("project_id"),
                "events_count": r.get("events_count", 0),
                "interaction_count": r.get("interaction_count", 0),
                "duration_ms": r.get("metadata", {}).get("duration", 0),
                "url": r.get("metadata", {}).get("url"),
                "created_at": r.get("created_at"),
            }
            for r in recordings[:limit]
        ],
        "total": len(recordings),
    }


@router.delete("/{recording_id}")
async def delete_recording(request: Request, recording_id: str):
    """
    Delete a recording.
    Requires authentication. User must have access to the recording.
    """
    # Authenticate the request
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    if recording_id not in _recordings:
        raise HTTPException(status_code=404, detail="Recording not found")

    recording = _recordings[recording_id]

    # Verify user has access to delete this recording
    recording_org = recording.get("organization_id")
    user_org = user.get("organization_id")
    recording_user = recording.get("user_id")
    current_user = user.get("user_id")

    if recording_org and user_org and recording_org != user_org:
        if recording_user != current_user:
            raise HTTPException(status_code=403, detail="Access denied to this recording")

    del _recordings[recording_id]

    logger.info(
        "Recording deleted",
        recording_id=recording_id,
        user_id=current_user,
    )

    return {
        "success": True,
        "message": f"Recording {recording_id} deleted",
    }


# =============================================================================
# Helper Functions
# =============================================================================


def _count_interaction_events(events: list[RRWebEvent]) -> int:
    """Count user interaction events in recording."""
    count = 0
    for event in events:
        if event.type == RRWebEventType.INCREMENTAL_SNAPSHOT.value:
            data = event.data
            source = data.get("source")

            # Mouse interactions (clicks, etc.)
            if source == IncrementalSource.MOUSE_INTERACTION.value:
                interaction_type = data.get("type")
                if interaction_type in [
                    MouseInteraction.CLICK.value,
                    MouseInteraction.DBL_CLICK.value,
                ]:
                    count += 1

            # Input events (typing)
            elif source == IncrementalSource.INPUT.value:
                count += 1

    return count


def _parse_rrweb_events(
    events: list[dict],
    include_waits: bool = True,
    include_scrolls: bool = False,
    min_wait_threshold: int = 500,
    generalize_data: bool = True,
) -> tuple[list[TestStepModel], list[TestAssertionModel], list[str]]:
    """Parse rrweb events into test steps and assertions."""
    steps: list[TestStepModel] = []
    assertions: list[TestAssertionModel] = []
    warnings: list[str] = []

    # Build node ID to selector map from full snapshot
    node_map: dict[int, str] = {}
    current_url: str | None = None
    last_timestamp: int | None = None

    for event in events:
        event_type = event.get("type")
        data = event.get("data", {})
        timestamp = event.get("timestamp", 0)

        # Handle full snapshot - build node map
        if event_type == RRWebEventType.FULL_SNAPSHOT.value:
            _build_node_map(data.get("node", {}), node_map)

        # Handle meta event - get URL
        elif event_type == RRWebEventType.META.value:
            new_url = data.get("href")
            if new_url and new_url != current_url:
                if current_url is not None:
                    # Navigation detected
                    steps.append(
                        TestStepModel(
                            action="goto",
                            target=new_url,
                            timestamp=timestamp,
                        )
                    )
                current_url = new_url

        # Handle incremental snapshot - user actions
        elif event_type == RRWebEventType.INCREMENTAL_SNAPSHOT.value:
            source = data.get("source")

            # Add wait step if significant pause
            if include_waits and last_timestamp:
                pause = timestamp - last_timestamp
                if pause >= min_wait_threshold:
                    steps.append(
                        TestStepModel(
                            action="wait",
                            value=str(pause),
                            timestamp=timestamp,
                        )
                    )

            # Mouse interactions
            if source == IncrementalSource.MOUSE_INTERACTION.value:
                interaction_type = data.get("type")
                node_id = data.get("id")
                selector = node_map.get(node_id, f"[data-rrweb-id='{node_id}']")

                if interaction_type == MouseInteraction.CLICK.value:
                    steps.append(
                        TestStepModel(
                            action="click",
                            target=selector,
                            timestamp=timestamp,
                        )
                    )
                elif interaction_type == MouseInteraction.DBL_CLICK.value:
                    steps.append(
                        TestStepModel(
                            action="dblclick",
                            target=selector,
                            timestamp=timestamp,
                        )
                    )

            # Input events (typing)
            elif source == IncrementalSource.INPUT.value:
                node_id = data.get("id")
                text = data.get("text", "")
                selector = node_map.get(node_id, f"[data-rrweb-id='{node_id}']")

                # Generalize data if requested
                if generalize_data and text:
                    if "@" in text:
                        text = "{{test_email}}"
                    elif len(text) > 20:
                        text = "{{test_text}}"

                steps.append(
                    TestStepModel(
                        action="fill",
                        target=selector,
                        value=text,
                        timestamp=timestamp,
                    )
                )

            # Scroll events
            elif source == IncrementalSource.SCROLL.value and include_scrolls:
                node_id = data.get("id")
                x = data.get("x", 0)
                y = data.get("y", 0)
                selector = node_map.get(node_id, "window")

                steps.append(
                    TestStepModel(
                        action="scroll",
                        target=selector,
                        value=f"{x},{y}",
                        timestamp=timestamp,
                    )
                )

            last_timestamp = timestamp

    # Generate basic assertions
    if current_url:
        # Assert we ended up at expected URL
        assertions.append(
            TestAssertionModel(
                type="url_contains",
                expected=current_url.split("/")[-1] if "/" in current_url else current_url,
            )
        )

    # Add warning if few interactions detected
    if len([s for s in steps if s.action in ["click", "fill"]]) < 2:
        warnings.append("Few user interactions detected. Recording may be incomplete.")

    # Add warning if selectors are low quality
    low_quality_selectors = sum(1 for s in steps if s.target and "data-rrweb-id" in s.target)
    if low_quality_selectors > len(steps) * 0.3:
        warnings.append(
            f"{low_quality_selectors} steps use auto-generated selectors. "
            "Consider adding better IDs/classes to your HTML."
        )

    return steps, assertions, warnings


def _build_node_map(node: dict, node_map: dict[int, str], parent_path: str = "") -> None:
    """Recursively build a map from node IDs to CSS selectors."""
    node_id = node.get("id")
    if not node_id:
        return

    # Determine selector for this node
    tag_name = node.get("tagName", "").lower()
    attributes = node.get("attributes", {})

    selector = ""

    # Prefer ID
    if "id" in attributes and attributes["id"]:
        selector = f"#{attributes['id']}"

    # Then data-testid
    elif "data-testid" in attributes:
        selector = f"[data-testid='{attributes['data-testid']}']"

    # Then class with tag
    elif "class" in attributes and attributes["class"]:
        classes = attributes["class"].split()
        if classes:
            # Use first meaningful class
            meaningful_class = next(
                (c for c in classes if not c.startswith("css-") and len(c) > 2), classes[0]
            )
            selector = f"{tag_name}.{meaningful_class}"

    # Then name attribute for inputs
    elif tag_name in ["input", "select", "textarea"] and "name" in attributes:
        selector = f"{tag_name}[name='{attributes['name']}']"

    # Fallback to tag with index (less reliable)
    else:
        selector = f"{tag_name}" if tag_name else f"[data-rrweb-id='{node_id}']"

    node_map[node_id] = selector

    # Process children
    child_nodes = node.get("childNodes", [])
    for child in child_nodes:
        if isinstance(child, dict):
            _build_node_map(child, node_map, selector)
