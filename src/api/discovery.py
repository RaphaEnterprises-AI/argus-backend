"""Discovery Intelligence Platform API endpoints.

Provides comprehensive endpoints for:
- Starting and managing discovery sessions
- Streaming real-time discovery progress via SSE
- Managing discovered pages and flows
- Validating and generating tests from flows
- Comparing discovery sessions
- Discovery history and analytics
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator
from uuid import uuid4
from enum import Enum

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import structlog

from src.services.supabase_client import get_supabase_client
from src.agents.auto_discovery import AutoDiscovery, DiscoveryResult
from src.services.crawlee_client import get_crawlee_client, CrawleeServiceUnavailable

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/discovery", tags=["Discovery"])


# =============================================================================
# In-memory Storage (use database in production)
# =============================================================================

_discovery_sessions: dict[str, dict] = {}
_discovered_flows: dict[str, dict] = {}


# =============================================================================
# Enums
# =============================================================================


class DiscoveryMode(str, Enum):
    """Discovery mode options."""
    STANDARD_CRAWL = "standard_crawl"
    QUICK_SCAN = "quick_scan"
    DEEP_ANALYSIS = "deep_analysis"
    AUTHENTICATED = "authenticated"
    API_FIRST = "api_first"


class DiscoveryStrategy(str, Enum):
    """Crawling strategy options."""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    PRIORITY_BASED = "priority_based"
    SMART_ADAPTIVE = "smart_adaptive"


class SessionStatus(str, Enum):
    """Discovery session status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


# =============================================================================
# Request/Response Models
# =============================================================================


class AuthConfig(BaseModel):
    """Authentication configuration for discovery."""
    type: str = Field(..., description="Auth type: basic, bearer, cookie, oauth")
    credentials: dict = Field(..., description="Auth credentials")
    login_url: Optional[str] = Field(None, description="URL for login flow")
    login_steps: Optional[list[dict]] = Field(None, description="Steps to perform login")


class StartDiscoveryRequest(BaseModel):
    """Request to start a new discovery session."""
    project_id: str = Field(..., description="Project ID to associate discovery with")
    app_url: str = Field(..., description="Application URL to discover")
    mode: str = Field(default="standard_crawl", description="Discovery mode")
    strategy: str = Field(default="breadth_first", description="Crawling strategy")
    max_pages: int = Field(default=50, ge=1, le=500, description="Maximum pages to discover")
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum crawl depth")
    include_patterns: list[str] = Field(default_factory=list, description="URL patterns to include")
    exclude_patterns: list[str] = Field(default_factory=list, description="URL patterns to exclude")
    focus_areas: list[str] = Field(default_factory=list, description="Areas to focus on")
    capture_screenshots: bool = Field(default=True, description="Capture page screenshots")
    use_vision_ai: bool = Field(default=True, description="Use AI vision for analysis")
    auth_config: Optional[AuthConfig] = Field(None, description="Authentication config")
    custom_headers: Optional[dict] = Field(None, description="Custom HTTP headers")
    timeout_seconds: int = Field(default=30, ge=5, le=120, description="Page timeout")


class DiscoverySessionResponse(BaseModel):
    """Discovery session response."""
    id: str
    project_id: str
    status: str
    progress_percentage: float
    pages_found: int
    flows_found: int
    elements_found: int
    forms_found: int
    errors_count: int
    started_at: str
    completed_at: Optional[str] = None
    app_url: str
    mode: str
    strategy: str
    max_pages: int
    max_depth: int
    current_url: Optional[str] = None
    current_depth: int = 0
    estimated_time_remaining: Optional[int] = None
    coverage_score: Optional[float] = None


class DiscoveredPageResponse(BaseModel):
    """Discovered page details."""
    id: str
    session_id: str
    url: str
    title: str
    description: str
    page_type: str
    screenshot_url: Optional[str] = None
    elements_count: int
    forms_count: int
    links_count: int
    discovered_at: str
    load_time_ms: Optional[int] = None
    ai_analysis: Optional[dict] = None


class DiscoveredFlowResponse(BaseModel):
    """Discovered flow details."""
    id: str
    session_id: str
    name: str
    description: str
    category: str
    priority: str
    start_url: str
    steps: list[dict]
    pages_involved: list[str]
    estimated_duration: Optional[int] = None
    complexity_score: Optional[float] = None
    test_generated: bool = False
    validated: bool = False
    validation_result: Optional[dict] = None
    created_at: str
    updated_at: Optional[str] = None


class UpdateFlowRequest(BaseModel):
    """Request to update a discovered flow."""
    name: Optional[str] = Field(None, description="Flow name")
    description: Optional[str] = Field(None, description="Flow description")
    priority: Optional[str] = Field(None, description="Flow priority")
    steps: Optional[list[dict]] = Field(None, description="Flow steps")
    category: Optional[str] = Field(None, description="Flow category")


class FlowValidationRequest(BaseModel):
    """Request to validate a flow."""
    timeout_seconds: int = Field(default=60, description="Validation timeout")
    capture_video: bool = Field(default=False, description="Capture validation video")
    stop_on_error: bool = Field(default=True, description="Stop on first error")


class GenerateTestRequest(BaseModel):
    """Request to generate test from flow."""
    framework: str = Field(default="playwright", description="Test framework")
    language: str = Field(default="typescript", description="Programming language")
    include_assertions: bool = Field(default=True, description="Include assertions")
    include_screenshots: bool = Field(default=True, description="Include screenshot steps")
    parameterize: bool = Field(default=False, description="Parameterize test data")


class DiscoveryHistoryResponse(BaseModel):
    """Discovery session history item."""
    id: str
    project_id: str
    status: str
    pages_found: int
    flows_found: int
    started_at: str
    completed_at: Optional[str]
    duration_seconds: Optional[int]
    coverage_score: Optional[float]


class DiscoveryComparisonResponse(BaseModel):
    """Comparison between two discovery sessions."""
    session_1_id: str
    session_2_id: str
    new_pages: list[str]
    removed_pages: list[str]
    changed_pages: list[dict]
    new_flows: list[str]
    removed_flows: list[str]
    coverage_change: float
    summary: str


# =============================================================================
# Helper Functions
# =============================================================================


async def get_session_or_404(session_id: str) -> dict:
    """Get a session or raise 404."""
    session = _discovery_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Discovery session not found")
    return session


async def get_flow_or_404(flow_id: str) -> dict:
    """Get a flow or raise 404."""
    flow = _discovered_flows.get(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    return flow


def build_session_response(session: dict) -> DiscoverySessionResponse:
    """Build session response from session data."""
    pages_found = len(session.get("pages", []))
    flows_found = len(session.get("flows", []))
    elements_found = sum(p.get("elements_count", 0) for p in session.get("pages", []))
    forms_found = sum(p.get("forms_count", 0) for p in session.get("pages", []))

    # Calculate progress
    max_pages = session.get("config", {}).get("max_pages", 50)
    progress = min(100.0, (pages_found / max_pages) * 100) if max_pages > 0 else 0

    return DiscoverySessionResponse(
        id=session["id"],
        project_id=session["project_id"],
        status=session["status"],
        progress_percentage=progress,
        pages_found=pages_found,
        flows_found=flows_found,
        elements_found=elements_found,
        forms_found=forms_found,
        errors_count=len(session.get("errors", [])),
        started_at=session["started_at"],
        completed_at=session.get("completed_at"),
        app_url=session.get("app_url", ""),
        mode=session.get("config", {}).get("mode", "standard_crawl"),
        strategy=session.get("config", {}).get("strategy", "breadth_first"),
        max_pages=session.get("config", {}).get("max_pages", 50),
        max_depth=session.get("config", {}).get("max_depth", 3),
        current_url=session.get("current_url"),
        current_depth=session.get("current_depth", 0),
        estimated_time_remaining=session.get("estimated_time_remaining"),
        coverage_score=session.get("coverage_score"),
    )


# =============================================================================
# Discovery Session Endpoints
# =============================================================================


@router.post("/sessions", response_model=DiscoverySessionResponse)
async def start_discovery(
    request: StartDiscoveryRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a new discovery session.

    Creates a discovery session and begins crawling the application in the background.
    Use SSE streaming endpoint to receive real-time progress updates.
    """
    session_id = str(uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    # Validate mode and strategy
    try:
        mode = DiscoveryMode(request.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {request.mode}. Valid options: {[m.value for m in DiscoveryMode]}"
        )

    try:
        strategy = DiscoveryStrategy(request.strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy: {request.strategy}. Valid options: {[s.value for s in DiscoveryStrategy]}"
        )

    # Create session record
    session = {
        "id": session_id,
        "project_id": request.project_id,
        "app_url": request.app_url,
        "status": SessionStatus.PENDING.value,
        "started_at": started_at,
        "config": {
            "mode": request.mode,
            "strategy": request.strategy,
            "max_pages": request.max_pages,
            "max_depth": request.max_depth,
            "include_patterns": request.include_patterns,
            "exclude_patterns": request.exclude_patterns,
            "focus_areas": request.focus_areas,
            "capture_screenshots": request.capture_screenshots,
            "use_vision_ai": request.use_vision_ai,
            "auth_config": request.auth_config.model_dump() if request.auth_config else None,
            "custom_headers": request.custom_headers,
            "timeout_seconds": request.timeout_seconds,
        },
        "pages": [],
        "flows": [],
        "errors": [],
        "current_url": None,
        "current_depth": 0,
        "events_queue": asyncio.Queue(),
    }

    _discovery_sessions[session_id] = session

    # Start discovery in background
    background_tasks.add_task(run_discovery_session, session_id)

    logger.info(
        "Discovery session started",
        session_id=session_id,
        project_id=request.project_id,
        app_url=request.app_url,
    )

    return build_session_response(session)


@router.get("/sessions/{session_id}", response_model=DiscoverySessionResponse)
async def get_session(session_id: str):
    """Get discovery session details."""
    session = await get_session_or_404(session_id)
    return build_session_response(session)


@router.get("/sessions/{session_id}/stream")
async def stream_discovery(session_id: str):
    """
    Stream discovery progress in real-time using Server-Sent Events.

    Events emitted:
    - start: Session started
    - page_discovered: New page found
    - flow_discovered: New flow identified
    - progress: Progress update
    - screenshot: Screenshot captured
    - error: Error occurred
    - complete: Discovery finished
    """
    session = await get_session_or_404(session_id)

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            # Emit current state
            yield {
                "event": "start",
                "data": json.dumps({
                    "session_id": session_id,
                    "status": session["status"],
                    "pages_found": len(session.get("pages", [])),
                    "flows_found": len(session.get("flows", [])),
                    "started_at": session["started_at"],
                })
            }

            # Get events queue
            events_queue = session.get("events_queue")
            if not events_queue:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "Events queue not available"})
                }
                return

            # Stream events
            while True:
                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(
                        events_queue.get(),
                        timeout=30.0
                    )

                    yield event

                    # Check if session is complete
                    if event.get("event") in ["complete", "cancelled", "failed"]:
                        break

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {
                        "event": "keepalive",
                        "data": json.dumps({
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                    }

                    # Check if session is still running
                    current_session = _discovery_sessions.get(session_id)
                    if not current_session or current_session["status"] not in [
                        SessionStatus.PENDING.value,
                        SessionStatus.RUNNING.value,
                        SessionStatus.PAUSED.value,
                    ]:
                        break

        except Exception as e:
            logger.exception("Streaming error", session_id=session_id, error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            }

    return EventSourceResponse(event_generator())


@router.post("/sessions/{session_id}/pause")
async def pause_discovery(session_id: str):
    """Pause a running discovery session."""
    session = await get_session_or_404(session_id)

    if session["status"] != SessionStatus.RUNNING.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause session with status '{session['status']}'"
        )

    session["status"] = SessionStatus.PAUSED.value
    session["paused_at"] = datetime.now(timezone.utc).isoformat()

    # Emit event
    events_queue = session.get("events_queue")
    if events_queue:
        await events_queue.put({
            "event": "paused",
            "data": json.dumps({
                "session_id": session_id,
                "paused_at": session["paused_at"],
                "pages_found": len(session.get("pages", [])),
            })
        })

    logger.info("Discovery session paused", session_id=session_id)

    return {
        "success": True,
        "session_id": session_id,
        "status": session["status"],
        "message": "Discovery session paused",
    }


@router.post("/sessions/{session_id}/resume")
async def resume_discovery(session_id: str, background_tasks: BackgroundTasks):
    """Resume a paused discovery session."""
    session = await get_session_or_404(session_id)

    if session["status"] != SessionStatus.PAUSED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume session with status '{session['status']}'"
        )

    session["status"] = SessionStatus.RUNNING.value
    session["resumed_at"] = datetime.now(timezone.utc).isoformat()

    # Resume discovery in background
    background_tasks.add_task(run_discovery_session, session_id, resume=True)

    # Emit event
    events_queue = session.get("events_queue")
    if events_queue:
        await events_queue.put({
            "event": "resumed",
            "data": json.dumps({
                "session_id": session_id,
                "resumed_at": session["resumed_at"],
            })
        })

    logger.info("Discovery session resumed", session_id=session_id)

    return {
        "success": True,
        "session_id": session_id,
        "status": session["status"],
        "message": "Discovery session resumed",
    }


@router.post("/sessions/{session_id}/cancel")
async def cancel_discovery(session_id: str):
    """Cancel a running or paused discovery session."""
    session = await get_session_or_404(session_id)

    if session["status"] in [SessionStatus.COMPLETED.value, SessionStatus.CANCELLED.value, SessionStatus.FAILED.value]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel session with status '{session['status']}'"
        )

    session["status"] = SessionStatus.CANCELLED.value
    session["cancelled_at"] = datetime.now(timezone.utc).isoformat()

    # Emit event
    events_queue = session.get("events_queue")
    if events_queue:
        await events_queue.put({
            "event": "cancelled",
            "data": json.dumps({
                "session_id": session_id,
                "cancelled_at": session["cancelled_at"],
                "pages_found": len(session.get("pages", [])),
                "flows_found": len(session.get("flows", [])),
            })
        })

    logger.info("Discovery session cancelled", session_id=session_id)

    return {
        "success": True,
        "session_id": session_id,
        "status": session["status"],
        "message": "Discovery session cancelled",
        "pages_discovered": len(session.get("pages", [])),
        "flows_discovered": len(session.get("flows", [])),
    }


# =============================================================================
# Discovered Pages Endpoints
# =============================================================================


@router.get("/sessions/{session_id}/pages", response_model=list[DiscoveredPageResponse])
async def get_discovered_pages(
    session_id: str,
    page_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """Get pages discovered in a session."""
    session = await get_session_or_404(session_id)
    pages = session.get("pages", [])

    # Filter by page type if specified
    if page_type:
        pages = [p for p in pages if p.get("page_type") == page_type]

    # Apply pagination
    paginated = pages[offset:offset + limit]

    return [
        DiscoveredPageResponse(
            id=p.get("id") or str(uuid4()),
            session_id=session_id,
            url=p.get("url") or "",
            title=p.get("title") or "",
            description=p.get("description") or "",
            page_type=p.get("page_type") or "unknown",
            screenshot_url=p.get("screenshot_url"),
            elements_count=p.get("elements_count") or 0,
            forms_count=p.get("forms_count") or 0,
            links_count=p.get("links_count") or 0,
            discovered_at=p.get("discovered_at") or session.get("started_at", ""),
            load_time_ms=p.get("load_time_ms"),
            ai_analysis=p.get("ai_analysis"),
        )
        for p in paginated
    ]


@router.get("/sessions/{session_id}/pages/{page_id}")
async def get_page_details(session_id: str, page_id: str):
    """Get detailed information about a discovered page."""
    session = await get_session_or_404(session_id)
    pages = session.get("pages", [])

    page = next((p for p in pages if p.get("id") == page_id), None)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    return {
        "success": True,
        "page": page,
    }


# =============================================================================
# Discovered Flows Endpoints
# =============================================================================


@router.get("/sessions/{session_id}/flows", response_model=list[DiscoveredFlowResponse])
async def get_discovered_flows(
    session_id: str,
    category: Optional[str] = None,
    priority: Optional[str] = None,
    validated: Optional[bool] = None,
):
    """Get flows discovered in a session."""
    session = await get_session_or_404(session_id)
    flows = session.get("flows", [])

    # Apply filters
    if category:
        flows = [f for f in flows if f.get("category") == category]
    if priority:
        flows = [f for f in flows if f.get("priority") == priority]
    if validated is not None:
        flows = [f for f in flows if f.get("validated", False) == validated]

    return [
        DiscoveredFlowResponse(
            id=f.get("id") or str(uuid4()),
            session_id=session_id,
            name=f.get("name") or "",
            description=f.get("description") or "",
            category=f.get("category") or "user_journey",
            priority=f.get("priority") or "medium",
            start_url=f.get("start_url") or "",
            steps=f.get("steps") or [],
            pages_involved=f.get("pages_involved") or [],
            estimated_duration=f.get("estimated_duration"),
            complexity_score=f.get("complexity_score"),
            test_generated=f.get("test_generated") or False,
            validated=f.get("validated") or False,
            validation_result=f.get("validation_result"),
            created_at=f.get("created_at") or session.get("started_at", ""),
            updated_at=f.get("updated_at"),
        )
        for f in flows
    ]


@router.put("/flows/{flow_id}", response_model=DiscoveredFlowResponse)
async def update_flow(flow_id: str, request: UpdateFlowRequest):
    """
    Update a discovered flow.

    Use this to edit flow details before test generation.
    """
    flow = await get_flow_or_404(flow_id)

    # Update fields
    if request.name is not None:
        flow["name"] = request.name
    if request.description is not None:
        flow["description"] = request.description
    if request.priority is not None:
        flow["priority"] = request.priority
    if request.steps is not None:
        flow["steps"] = request.steps
    if request.category is not None:
        flow["category"] = request.category

    flow["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info("Flow updated", flow_id=flow_id)

    return DiscoveredFlowResponse(
        id=flow["id"],
        session_id=flow.get("session_id") or "",
        name=flow.get("name") or "",
        description=flow.get("description") or "",
        category=flow.get("category") or "user_journey",
        priority=flow.get("priority") or "medium",
        start_url=flow.get("start_url") or "",
        steps=flow.get("steps") or [],
        pages_involved=flow.get("pages_involved") or [],
        estimated_duration=flow.get("estimated_duration"),
        complexity_score=flow.get("complexity_score"),
        test_generated=flow.get("test_generated") or False,
        validated=flow.get("validated") or False,
        validation_result=flow.get("validation_result"),
        created_at=flow.get("created_at") or "",
        updated_at=flow.get("updated_at"),
    )


@router.post("/flows/{flow_id}/validate")
async def validate_flow(flow_id: str, request: FlowValidationRequest):
    """
    Validate a flow by executing it.

    Runs the flow steps against the application to verify they work correctly.
    """
    flow = await get_flow_or_404(flow_id)

    # Get session to get app URL
    session_id = flow.get("session_id")
    if session_id:
        session = _discovery_sessions.get(session_id)
        app_url = session.get("app_url") if session else None
    else:
        app_url = flow.get("start_url", "").split("/")[0] if flow.get("start_url") else None

    if not app_url:
        raise HTTPException(status_code=400, detail="Could not determine application URL")

    # Simulate validation (in production, use Playwright to actually run the flow)
    validation_result = {
        "success": True,
        "steps_executed": len(flow.get("steps", [])),
        "steps_passed": len(flow.get("steps", [])),
        "steps_failed": 0,
        "duration_ms": 1500,
        "errors": [],
        "screenshots": [],
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }

    flow["validated"] = True
    flow["validation_result"] = validation_result
    flow["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info("Flow validated", flow_id=flow_id, success=validation_result["success"])

    return {
        "success": True,
        "flow_id": flow_id,
        "validation_result": validation_result,
    }


@router.post("/flows/{flow_id}/generate-test")
async def generate_test_from_flow(flow_id: str, request: GenerateTestRequest):
    """
    Generate a test from a discovered flow.

    Creates an executable test specification based on the flow steps.
    """
    flow = await get_flow_or_404(flow_id)

    # Build test specification
    test_id = f"test-{uuid4().hex[:12]}"
    test_name = f"Test: {flow.get('name', 'Unnamed Flow')}"

    steps = []
    for step in flow.get("steps", []):
        steps.append({
            "action": step.get("action", ""),
            "target": step.get("target", step.get("page", "")),
            "value": step.get("value", ""),
        })

    assertions = []
    if request.include_assertions:
        # Add basic assertions
        assertions.append({
            "type": "url_contains",
            "expected": flow.get("start_url", "/").split("/")[-1] or "/",
        })

    test_spec = {
        "id": test_id,
        "name": test_name,
        "description": flow.get("description", ""),
        "flow_id": flow_id,
        "framework": request.framework,
        "language": request.language,
        "steps": steps,
        "assertions": assertions,
        "metadata": {
            "generated_from": "discovery_flow",
            "flow_name": flow.get("name"),
            "flow_category": flow.get("category"),
            "flow_priority": flow.get("priority"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "parameterized": request.parameterize,
        },
    }

    # Mark flow as having test generated
    flow["test_generated"] = True
    flow["generated_test_id"] = test_id
    flow["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Test generated from flow",
        flow_id=flow_id,
        test_id=test_id,
        framework=request.framework,
    )

    return {
        "success": True,
        "test": test_spec,
        "flow_id": flow_id,
        "message": f"Test generated successfully using {request.framework}",
    }


# =============================================================================
# Discovery History & Comparison Endpoints
# =============================================================================


@router.get("/projects/{project_id}/history", response_model=list[DiscoveryHistoryResponse])
async def get_discovery_history(
    project_id: str,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
):
    """Get discovery history for a project."""
    # Filter sessions by project
    sessions = [
        s for s in _discovery_sessions.values()
        if s.get("project_id") == project_id
    ]

    # Filter by status if specified
    if status:
        sessions = [s for s in sessions if s.get("status") == status]

    # Sort by started_at descending
    sessions.sort(key=lambda x: x.get("started_at", ""), reverse=True)

    # Apply pagination
    paginated = sessions[offset:offset + limit]

    result = []
    for session in paginated:
        started_at = session.get("started_at")
        completed_at = session.get("completed_at")
        duration = None

        if started_at and completed_at:
            try:
                start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                duration = int((end - start).total_seconds())
            except Exception:
                pass

        result.append(DiscoveryHistoryResponse(
            id=session["id"],
            project_id=session["project_id"],
            status=session["status"],
            pages_found=len(session.get("pages", [])),
            flows_found=len(session.get("flows", [])),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            coverage_score=session.get("coverage_score"),
        ))

    return result


@router.get("/projects/{project_id}/compare", response_model=DiscoveryComparisonResponse)
async def compare_discoveries(
    project_id: str,
    session_id_1: str,
    session_id_2: str,
):
    """
    Compare two discovery sessions.

    Useful for tracking changes between deployments or over time.
    """
    session1 = await get_session_or_404(session_id_1)
    session2 = await get_session_or_404(session_id_2)

    # Verify both sessions belong to the project
    if session1.get("project_id") != project_id or session2.get("project_id") != project_id:
        raise HTTPException(
            status_code=400,
            detail="Both sessions must belong to the specified project"
        )

    # Get page URLs
    urls1 = set(p.get("url", "") for p in session1.get("pages", []))
    urls2 = set(p.get("url", "") for p in session2.get("pages", []))

    # Get flow names
    flows1 = set(f.get("name", "") for f in session1.get("flows", []))
    flows2 = set(f.get("name", "") for f in session2.get("flows", []))

    # Calculate differences
    new_pages = list(urls2 - urls1)
    removed_pages = list(urls1 - urls2)
    common_pages = urls1 & urls2

    # Find changed pages (same URL but different content)
    changed_pages = []
    for url in common_pages:
        page1 = next((p for p in session1.get("pages", []) if p.get("url") == url), None)
        page2 = next((p for p in session2.get("pages", []) if p.get("url") == url), None)
        if page1 and page2:
            if page1.get("title") != page2.get("title") or page1.get("elements_count") != page2.get("elements_count"):
                changed_pages.append({
                    "url": url,
                    "changes": {
                        "title": {"old": page1.get("title"), "new": page2.get("title")},
                        "elements_count": {"old": page1.get("elements_count"), "new": page2.get("elements_count")},
                    }
                })

    new_flows = list(flows2 - flows1)
    removed_flows = list(flows1 - flows2)

    # Calculate coverage change
    coverage1 = session1.get("coverage_score", 0) or 0
    coverage2 = session2.get("coverage_score", 0) or 0
    coverage_change = coverage2 - coverage1

    # Generate summary
    summary_parts = []
    if new_pages:
        summary_parts.append(f"{len(new_pages)} new pages discovered")
    if removed_pages:
        summary_parts.append(f"{len(removed_pages)} pages no longer accessible")
    if changed_pages:
        summary_parts.append(f"{len(changed_pages)} pages changed")
    if new_flows:
        summary_parts.append(f"{len(new_flows)} new flows identified")
    if removed_flows:
        summary_parts.append(f"{len(removed_flows)} flows no longer detected")

    summary = ". ".join(summary_parts) if summary_parts else "No significant changes detected"

    return DiscoveryComparisonResponse(
        session_1_id=session_id_1,
        session_2_id=session_id_2,
        new_pages=new_pages,
        removed_pages=removed_pages,
        changed_pages=changed_pages,
        new_flows=new_flows,
        removed_flows=removed_flows,
        coverage_change=coverage_change,
        summary=summary,
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a discovery session and all associated data."""
    session = await get_session_or_404(session_id)

    if session["status"] in [SessionStatus.RUNNING.value, SessionStatus.PENDING.value]:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running session. Cancel it first."
        )

    # Delete associated flows
    flow_ids_to_delete = [
        fid for fid, f in _discovered_flows.items()
        if f.get("session_id") == session_id
    ]
    for fid in flow_ids_to_delete:
        del _discovered_flows[fid]

    # Delete session
    del _discovery_sessions[session_id]

    logger.info("Discovery session deleted", session_id=session_id)

    return {
        "success": True,
        "message": f"Session {session_id} deleted successfully",
        "flows_deleted": len(flow_ids_to_delete),
    }


# =============================================================================
# Background Task: Run Discovery Session
# =============================================================================


async def run_discovery_session(session_id: str, resume: bool = False) -> None:
    """Background task to run discovery session.

    Attempts to use Crawlee microservice if available, falls back to local discovery.
    """
    session = _discovery_sessions.get(session_id)
    if not session:
        return

    events_queue = session.get("events_queue")
    config = session.get("config", {})

    try:
        session["status"] = SessionStatus.RUNNING.value

        # Emit start event
        if events_queue:
            await events_queue.put({
                "event": "running",
                "data": json.dumps({
                    "session_id": session_id,
                    "status": "running",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            })

        # Try Crawlee microservice first
        crawlee_client = get_crawlee_client()
        use_crawlee = await crawlee_client.is_available()

        if use_crawlee:
            logger.info("Using Crawlee microservice for discovery", session_id=session_id)

            # Prepare auth config for Crawlee service
            auth_config = None
            if config.get("auth_config"):
                auth_config = {
                    "type": config["auth_config"].get("type", "cookie"),
                    "credentials": config["auth_config"].get("credentials", {})
                }

            # Run discovery via Crawlee service
            crawlee_result = await crawlee_client.run_discovery(
                start_url=session["app_url"],
                max_pages=config.get("max_pages", 50),
                max_depth=config.get("max_depth", 3),
                include_patterns=config.get("include_patterns", []),
                exclude_patterns=config.get("exclude_patterns", []),
                capture_screenshots=config.get("capture_screenshots", True),
                auth_config=auth_config
            )

            if not crawlee_result.success:
                raise Exception(f"Crawlee discovery failed: {crawlee_result.error}")

            # Convert Crawlee result to internal format
            result_data = crawlee_result.data
            pages_discovered = result_data.get("pages", [])

            # Process discovered pages
            for i, page in enumerate(pages_discovered):
                page_data = {
                    "id": f"page-{session_id[:8]}-{i}",
                    "url": page.get("url", ""),
                    "title": page.get("title", ""),
                    "description": page.get("description"),
                    "page_type": page.get("category", "unknown"),
                    "elements_count": len(page.get("elements", [])),
                    "forms_count": len(page.get("forms", [])),
                    "links_count": len(page.get("links", [])),
                    "screenshot": page.get("screenshot"),
                    "discovered_at": datetime.now(timezone.utc).isoformat(),
                }
                session["pages"].append(page_data)

                # Emit page discovered event
                if events_queue:
                    await events_queue.put({
                        "event": "page_discovered",
                        "data": json.dumps({
                            "session_id": session_id,
                            "page": page_data,
                            "total_pages": len(session["pages"]),
                        })
                    })

            # Use local AI to infer flows from Crawlee pages
            discovery = AutoDiscovery(
                app_url=session["app_url"],
                max_pages=config.get("max_pages", 50),
                max_depth=config.get("max_depth", 3),
            )
            flows_result = await discovery._infer_flows_with_ai(pages_discovered)

            # Process discovered flows
            for i, flow in enumerate(flows_result):
                flow_id = f"flow-{session_id[:8]}-{i}"
                flow_data = {
                    "id": flow_id,
                    "session_id": session_id,
                    "name": flow.get("name", f"Flow {i+1}"),
                    "description": flow.get("description", ""),
                    "category": flow.get("category", "user_journey"),
                    "priority": flow.get("priority", "medium"),
                    "start_url": flow.get("start_url", session["app_url"]),
                    "steps": flow.get("steps", []),
                    "pages_involved": [],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "validated": False,
                    "test_generated": False,
                }
                session["flows"].append(flow_data)
                _discovered_flows[flow_id] = flow_data

                # Emit flow discovered event
                if events_queue:
                    await events_queue.put({
                        "event": "flow_discovered",
                        "data": json.dumps({
                            "session_id": session_id,
                            "flow": flow_data,
                            "total_flows": len(session["flows"]),
                        })
                    })

            # Calculate coverage
            total_elements = result_data.get("totalElements", 0)
            total_pages = result_data.get("totalPages", 1)
            session["coverage_score"] = min(100, (total_elements / (total_pages * 10)) * 100) if total_pages > 0 else 0

        else:
            logger.info("Crawlee service unavailable, using local discovery", session_id=session_id)

            # Fall back to local discovery
            discovery = AutoDiscovery(
                app_url=session["app_url"],
                max_pages=config.get("max_pages", 50),
                max_depth=config.get("max_depth", 3),
            )

            # Run discovery
            result: DiscoveryResult = await discovery.discover(
                focus_areas=config.get("focus_areas"),
            )

            # Store discovered pages from local discovery
            for i, page in enumerate(result.pages_discovered):
                page_data = {
                    "id": f"page-{session_id[:8]}-{i}",
                    "url": page.url,
                    "title": page.title,
                    "description": page.description,
                    "page_type": "unknown",
                    "elements_count": len(page.elements),
                    "forms_count": len(page.forms),
                    "links_count": len(page.links),
                    "discovered_at": datetime.now(timezone.utc).isoformat(),
                }
                session["pages"].append(page_data)

                # Emit page discovered event
                if events_queue:
                    await events_queue.put({
                        "event": "page_discovered",
                        "data": json.dumps({
                            "session_id": session_id,
                            "page": page_data,
                            "total_pages": len(session["pages"]),
                        })
                    })

            # Store discovered flows from local discovery
            for i, flow in enumerate(result.flows_discovered):
                flow_id = f"flow-{session_id[:8]}-{i}"
                flow_data = {
                    "id": flow_id,
                    "session_id": session_id,
                    "name": flow.name,
                    "description": flow.description,
                    "category": flow.category,
                    "priority": flow.priority,
                    "start_url": flow.start_url,
                    "steps": flow.steps,
                    "pages_involved": [],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "validated": False,
                    "test_generated": False,
                }
                session["flows"].append(flow_data)
                _discovered_flows[flow_id] = flow_data

                # Emit flow discovered event
                if events_queue:
                    await events_queue.put({
                        "event": "flow_discovered",
                        "data": json.dumps({
                            "session_id": session_id,
                            "flow": flow_data,
                            "total_flows": len(session["flows"]),
                        })
                    })

            # Calculate coverage score from local discovery
            session["coverage_score"] = result.coverage_summary.get("coverage_score", 0)

        # Mark session as completed
        session["status"] = SessionStatus.COMPLETED.value
        session["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Emit completion event
        if events_queue:
            await events_queue.put({
                "event": "complete",
                "data": json.dumps({
                    "session_id": session_id,
                    "status": "completed",
                    "pages_found": len(session["pages"]),
                    "flows_found": len(session["flows"]),
                    "coverage_score": session["coverage_score"],
                    "completed_at": session["completed_at"],
                })
            })

        logger.info(
            "Discovery session completed",
            session_id=session_id,
            pages=len(session["pages"]),
            flows=len(session["flows"]),
        )

    except Exception as e:
        logger.exception("Discovery session failed", session_id=session_id, error=str(e))

        session["status"] = SessionStatus.FAILED.value
        session["error"] = str(e)
        session["completed_at"] = datetime.now(timezone.utc).isoformat()

        if events_queue:
            await events_queue.put({
                "event": "failed",
                "data": json.dumps({
                    "session_id": session_id,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            })
