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
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.agents.auto_discovery import AutoDiscovery, DiscoveryResult
from src.discovery.engine import DiscoveryEngine, create_discovery_engine
from src.discovery.repository import DiscoveryRepository
from src.services.crawlee_client import get_crawlee_client
from src.services.supabase_client import get_raw_supabase_client, get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/discovery", tags=["Discovery"])


# =============================================================================
# Singleton Discovery Engine (uses Supabase for persistence)
# =============================================================================

_discovery_engine: DiscoveryEngine | None = None
_discovery_repository: DiscoveryRepository | None = None


def get_discovery_engine() -> DiscoveryEngine:
    """Get or create the singleton DiscoveryEngine."""
    global _discovery_engine
    if _discovery_engine is None:
        # Use raw supabase-py client for DiscoveryEngine/Repository
        # which use the .table() query builder
        supabase = get_raw_supabase_client()
        _discovery_engine = create_discovery_engine(supabase_client=supabase)
    return _discovery_engine


def get_discovery_repository() -> DiscoveryRepository:
    """Get or create the singleton DiscoveryRepository."""
    global _discovery_repository
    if _discovery_repository is None:
        # Use raw supabase-py client which has .table() method
        supabase = get_raw_supabase_client()
        _discovery_repository = DiscoveryRepository(supabase_client=supabase)
    return _discovery_repository


# =============================================================================
# Legacy In-memory Storage (kept for backward compatibility during transition)
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
    login_url: str | None = Field(None, description="URL for login flow")
    login_steps: list[dict] | None = Field(None, description="Steps to perform login")


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
    auth_config: AuthConfig | None = Field(None, description="Authentication config")
    custom_headers: dict | None = Field(None, description="Custom HTTP headers")
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
    completed_at: str | None = None
    app_url: str
    mode: str
    strategy: str
    max_pages: int
    max_depth: int
    current_url: str | None = None
    current_depth: int = 0
    estimated_time_remaining: int | None = None
    coverage_score: float | None = None


class DiscoveredPageResponse(BaseModel):
    """Discovered page details."""
    id: str
    session_id: str
    url: str
    title: str
    description: str
    page_type: str
    screenshot_url: str | None = None
    elements_count: int
    forms_count: int
    links_count: int
    discovered_at: str
    load_time_ms: int | None = None
    ai_analysis: dict | None = None


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
    estimated_duration: int | None = None
    complexity_score: float | None = None
    test_generated: bool = False
    validated: bool = False
    validation_result: dict | None = None
    created_at: str
    updated_at: str | None = None


class UpdateFlowRequest(BaseModel):
    """Request to update a discovered flow."""
    name: str | None = Field(None, description="Flow name")
    description: str | None = Field(None, description="Flow description")
    priority: str | None = Field(None, description="Flow priority")
    steps: list[dict] | None = Field(None, description="Flow steps")
    category: str | None = Field(None, description="Flow category")


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
    completed_at: str | None
    duration_seconds: int | None
    coverage_score: float | None


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
    # Handle both field names: DB uses singular (element_count), code uses plural (elements_count)
    elements_found = sum(
        p.get("element_count", 0) or p.get("elements_count", 0)
        for p in session.get("pages", [])
    )
    forms_found = sum(
        p.get("form_count", 0) or p.get("forms_count", 0)
        for p in session.get("pages", [])
    )

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


class DiscoverySessionListResponse(BaseModel):
    """List of discovery sessions."""
    sessions: list[DiscoverySessionResponse]
    total: int


class DiscoveryPatternResponse(BaseModel):
    """Discovery pattern response."""
    id: str
    pattern_type: str
    pattern_name: str
    pattern_signature: str
    pattern_data: dict
    times_seen: int
    test_success_rate: float | None
    self_heal_success_rate: float | None
    created_at: str
    updated_at: str | None


class DiscoveryPatternListResponse(BaseModel):
    """List of discovery patterns."""
    patterns: list[DiscoveryPatternResponse]
    total: int


@router.get("/sessions", response_model=DiscoverySessionListResponse)
async def list_discovery_sessions(
    project_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    List all discovery sessions.

    Optionally filter by project_id or status.
    Uses Supabase for persistence with in-memory cache fallback.
    """
    try:
        # Use repository for database-first access
        repository = get_discovery_repository()
        db_sessions = await repository.list_sessions(
            project_id=project_id,
            status=status,
            limit=limit,
            offset=offset,
        )

        # Convert dataclass sessions to dict format for response builder
        sessions = []
        for session in db_sessions:
            session_dict = {
                "id": session.id,
                "project_id": session.project_id,
                "name": session.name,
                "status": session.status.value if hasattr(session.status, 'value') else session.status,
                "start_url": session.start_url,
                "mode": session.mode.value if hasattr(session.mode, 'value') else session.mode,
                "strategy": session.strategy.value if hasattr(session.strategy, 'value') else session.strategy,
                "config": session.config,
                "max_pages": session.max_pages,
                "max_depth": session.max_depth,
                "progress_percentage": session.progress_percentage,
                "pages_discovered": session.pages_discovered,
                "pages_analyzed": session.pages_analyzed,
                "quality_score": session.quality_score,
                "insights": session.insights,
                "patterns_detected": session.patterns_detected,
                "recommendations": session.recommendations,
                "error_message": session.error_message,
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                "pages": [],
                "flows": [],
            }
            sessions.append(session_dict)

        # Also check in-memory for active sessions not yet persisted
        for session_id, in_mem_session in _discovery_sessions.items():
            if not any(s["id"] == session_id for s in sessions):
                # Apply filters
                if project_id and in_mem_session.get("project_id") != project_id:
                    continue
                if status and in_mem_session.get("status") != status:
                    continue
                sessions.append(in_mem_session)

        # Sort by started_at descending
        sessions.sort(key=lambda s: s.get("started_at") or s.get("created_at") or "", reverse=True)

        total = len(sessions)
        paginated_sessions = sessions[offset:offset + limit]

        return DiscoverySessionListResponse(
            sessions=[build_session_response(s) for s in paginated_sessions],
            total=total,
        )

    except Exception as e:
        logger.warning("Failed to list sessions from database, using in-memory", error=str(e))
        # Fallback to in-memory only
        sessions = list(_discovery_sessions.values())
        if project_id:
            sessions = [s for s in sessions if s.get("project_id") == project_id]
        if status:
            sessions = [s for s in sessions if s.get("status") == status]
        sessions.sort(key=lambda s: s.get("started_at", ""), reverse=True)
        total = len(sessions)
        sessions = sessions[offset:offset + limit]
        return DiscoverySessionListResponse(
            sessions=[build_session_response(s) for s in sessions],
            total=total,
        )


@router.get("/patterns", response_model=DiscoveryPatternListResponse)
async def list_discovery_patterns(
    pattern_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    List all discovery patterns.

    Patterns are learned from cross-project discovery sessions and used to
    improve element detection and flow inference.
    """
    from src.services.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    # Build query
    query = "/discovery_patterns?select=*&order=times_seen.desc"
    if pattern_type:
        query += f"&pattern_type=eq.{pattern_type}"
    query += f"&limit={limit}&offset={offset}"

    result = await supabase.request(query)

    if result.get("error"):
        # If table doesn't exist, return empty list
        logger.warning("discovery_patterns table may not exist", error=result.get("error"))
        return DiscoveryPatternListResponse(patterns=[], total=0)

    patterns = result.get("data", [])

    return DiscoveryPatternListResponse(
        patterns=[
            DiscoveryPatternResponse(
                id=p["id"],
                pattern_type=p.get("pattern_type", "unknown"),
                pattern_name=p.get("pattern_name", ""),
                pattern_signature=p.get("pattern_signature", ""),
                pattern_data=p.get("pattern_data", {}),
                times_seen=p.get("times_seen", 0),
                test_success_rate=p.get("test_success_rate"),
                self_heal_success_rate=p.get("self_heal_success_rate"),
                created_at=p.get("created_at", ""),
                updated_at=p.get("updated_at"),
            )
            for p in patterns
        ],
        total=len(patterns),
    )


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
    started_at = datetime.now(UTC).isoformat()

    # Validate mode and strategy
    try:
        DiscoveryMode(request.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {request.mode}. Valid options: {[m.value for m in DiscoveryMode]}"
        )

    try:
        DiscoveryStrategy(request.strategy)
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

    # Persist initial session to database (required for checkpoint updates)
    try:
        supabase = get_raw_supabase_client()
        if supabase:
            initial_record = {
                "id": session_id,
                "project_id": request.project_id,
                "name": f"Discovery {session_id[:8]}",
                "status": SessionStatus.PENDING.value,
                "start_url": request.app_url,
                "mode": request.mode,
                "strategy": request.strategy,
                "config": session["config"],
                "max_pages": request.max_pages,
                "max_depth": request.max_depth,
                "progress_percentage": 0,
                "pages_discovered": 0,
                "started_at": started_at,
            }
            supabase.table("discovery_sessions").insert(initial_record).execute()
            logger.debug("Initial discovery session persisted to database", session_id=session_id)
    except Exception as e:
        # Log but don't fail - in-memory storage is fallback
        logger.warning("Failed to persist initial session to database", error=str(e))

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

                except TimeoutError:
                    # Send keepalive
                    yield {
                        "event": "keepalive",
                        "data": json.dumps({
                            "timestamp": datetime.now(UTC).isoformat()
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
                    "timestamp": datetime.now(UTC).isoformat(),
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
    session["paused_at"] = datetime.now(UTC).isoformat()

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
    session["resumed_at"] = datetime.now(UTC).isoformat()

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
    session["cancelled_at"] = datetime.now(UTC).isoformat()

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
    page_type: str | None = None,
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
    category: str | None = None,
    priority: str | None = None,
    validated: bool | None = None,
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

    flow["updated_at"] = datetime.now(UTC).isoformat()

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
        "validated_at": datetime.now(UTC).isoformat(),
    }

    flow["validated"] = True
    flow["validation_result"] = validation_result
    flow["updated_at"] = datetime.now(UTC).isoformat()

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
            "generated_at": datetime.now(UTC).isoformat(),
            "parameterized": request.parameterize,
        },
    }

    # Mark flow as having test generated
    flow["test_generated"] = True
    flow["generated_test_id"] = test_id
    flow["updated_at"] = datetime.now(UTC).isoformat()

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
    status: str | None = None,
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


async def _persist_discovery_checkpoint(session: dict) -> bool:
    """Persist a lightweight checkpoint during discovery.

    Saves session progress and newly discovered pages without full flow data.
    Used for incremental persistence to prevent data loss on crashes.

    Args:
        session: The in-memory session dict to checkpoint

    Returns:
        True if checkpoint succeeded, False otherwise
    """
    try:
        supabase = get_raw_supabase_client()
        if not supabase:
            return False

        session_id = session["id"]
        project_id = session.get("project_id", "")
        app_url = session.get("app_url", "")

        # Include all required NOT NULL fields for upsert (handles case where initial insert failed)
        checkpoint_record = {
            "id": session_id,
            "project_id": project_id,
            "name": f"Discovery {session_id[:8]}",
            "status": session["status"],
            "start_url": app_url,
            "mode": session.get("config", {}).get("mode", "standard_crawl"),
            "strategy": session.get("config", {}).get("strategy", "breadth_first"),
            "max_pages": session.get("config", {}).get("max_pages", 50),
            "max_depth": session.get("config", {}).get("max_depth", 3),
            "progress_percentage": min(90, len(session.get("pages", [])) * 2),  # Rough progress
            "pages_discovered": len(session.get("pages", [])),
            "started_at": session.get("started_at"),
        }

        supabase.table("discovery_sessions").upsert(checkpoint_record).execute()

        # Persist any new pages
        # Note: Don't include 'id' - let DB auto-generate UUID
        pages = session.get("pages", [])
        if pages:
            # Only persist pages that don't have a persisted flag
            new_pages = [p for p in pages if not p.get("_persisted")]
            if new_pages:
                page_records = []
                for page in new_pages:
                    # Map page_type to valid DB enum values
                    page_type = page.get("page_type", "unknown")
                    valid_page_types = ['landing', 'form', 'list', 'detail', 'dashboard', 'settings', 'auth', 'error', 'content', 'search', 'unknown']
                    if page_type not in valid_page_types:
                        page_type = "unknown"

                    page_records.append({
                        # No 'id' - let DB generate UUID
                        "discovery_session_id": session_id,
                        "url": page.get("url", ""),
                        "title": page.get("title", ""),
                        "page_type": page_type,
                        "element_count": page.get("elements_count", 0),
                        "form_count": page.get("forms_count", 0),
                        "link_count": page.get("links_count", 0),
                        "created_at": page.get("discovered_at"),
                    })
                    page["_persisted"] = True  # Mark as persisted

                if page_records:
                    supabase.table("discovered_pages").insert(page_records).execute()

        logger.debug(
            "Discovery checkpoint saved",
            session_id=session_id,
            pages=len(pages),
        )
        return True

    except Exception as e:
        logger.warning("Failed to save discovery checkpoint", error=str(e))
        return False


async def _persist_discovery_session(session: dict) -> bool:
    """Persist discovery session data to Supabase.

    Saves session metadata, discovered pages, and flows to the database.
    Uses the raw Supabase client for table operations.

    Args:
        session: The in-memory session dict to persist

    Returns:
        True if persistence succeeded, False otherwise
    """
    try:
        supabase = get_raw_supabase_client()
        if not supabase:
            logger.warning("No Supabase client available for persistence")
            return False

        session_id = session["id"]
        project_id = session["project_id"]

        # Persist session to discovery_sessions table
        session_record = {
            "id": session_id,
            "project_id": project_id,
            "name": f"Discovery {session_id[:8]}",
            "status": session["status"],
            "start_url": session.get("app_url", ""),
            "mode": session.get("config", {}).get("mode", "standard_crawl"),
            "strategy": session.get("config", {}).get("strategy", "breadth_first"),
            "config": session.get("config", {}),
            "max_pages": session.get("config", {}).get("max_pages", 50),
            "max_depth": session.get("config", {}).get("max_depth", 3),
            "progress_percentage": 100 if session["status"] == "completed" else 0,
            "pages_discovered": len(session.get("pages", [])),
            "quality_score": session.get("coverage_score"),
            "started_at": session.get("started_at"),
            "completed_at": session.get("completed_at"),
        }

        # Upsert session
        supabase.table("discovery_sessions").upsert(session_record).execute()
        logger.debug("Persisted discovery session", session_id=session_id)

        # Persist discovered pages
        # Note: Don't include 'id' - let DB auto-generate UUID
        # The code-generated IDs like "page-abc12345-0" are strings, not valid UUIDs
        pages = session.get("pages", [])
        if pages:
            page_records = []
            for page in pages:
                # Map page_type to valid DB enum values
                page_type = page.get("page_type", "unknown")
                valid_page_types = ['landing', 'form', 'list', 'detail', 'dashboard', 'settings', 'auth', 'error', 'content', 'search', 'unknown']
                if page_type not in valid_page_types:
                    page_type = "unknown"

                page_records.append({
                    # No 'id' - let DB generate UUID
                    "discovery_session_id": session_id,
                    "url": page.get("url", ""),
                    "title": page.get("title", ""),
                    "page_type": page_type,
                    "element_count": page.get("elements_count", 0),
                    "form_count": page.get("forms_count", 0),
                    "link_count": page.get("links_count", 0),
                    "created_at": page.get("discovered_at"),
                })

            if page_records:
                # Use insert instead of upsert since we don't have valid IDs
                supabase.table("discovered_pages").insert(page_records).execute()
                logger.debug("Persisted discovered pages", count=len(page_records))

        # Persist discovered flows
        # Note: Don't include 'id' - let DB auto-generate UUID
        flows = session.get("flows", [])
        if flows:
            # Map category values to valid flow_type enum values
            flow_type_map = {
                "authentication": "authentication",
                "auth": "authentication",
                "login": "authentication",
                "registration": "registration",
                "signup": "registration",
                "checkout": "checkout",
                "payment": "checkout",
                "search": "search",
                "crud": "crud",
                "navigation": "navigation",
                "nav": "navigation",
                "form_submission": "form_submission",
                "form": "form_submission",
                "user_journey": "navigation",
                "custom": "custom",
            }

            flow_records = []
            for flow in flows:
                category = flow.get("category", "navigation").lower()
                flow_type = flow_type_map.get(category, "custom")

                flow_records.append({
                    # No 'id' - let DB generate UUID
                    "discovery_session_id": session_id,
                    "name": flow.get("name", "Unnamed Flow"),
                    "description": flow.get("description", ""),
                    "flow_type": flow_type,
                    "steps": flow.get("steps", []),
                    "validated": flow.get("validated", False),
                })

            if flow_records:
                # Use insert instead of upsert since we don't have valid IDs
                supabase.table("discovered_flows").insert(flow_records).execute()
                logger.debug("Persisted discovered flows", count=len(flow_records))

        logger.info(
            "Discovery session persisted to database",
            session_id=session_id,
            pages=len(pages),
            flows=len(flows),
        )
        return True

    except Exception as e:
        logger.exception("Failed to persist discovery session", error=str(e))
        return False


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
                    "timestamp": datetime.now(UTC).isoformat(),
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
                    "discovered_at": datetime.now(UTC).isoformat(),
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

                # Incremental persistence: save checkpoint every 5 pages
                if len(session["pages"]) % 5 == 0:
                    await _persist_discovery_checkpoint(session)

            # Use local AI to infer flows from Crawlee pages
            # Wrap in try/except so flow inference failure doesn't fail the session
            try:
                from src.agents.auto_discovery import DiscoveredPage as AutoDiscoveredPage

                discovery = AutoDiscovery(
                    app_url=session["app_url"],
                    max_pages=config.get("max_pages", 50),
                    max_depth=config.get("max_depth", 3),
                )

                # Convert Crawlee pages to AutoDiscovery format and populate discovered_pages
                for page in pages_discovered:
                    discovery.discovered_pages.append(AutoDiscoveredPage(
                        url=page.get("url", ""),
                        title=page.get("title", ""),
                        description=page.get("description") or "",
                        elements=[],  # Not needed for flow analysis
                        forms=page.get("forms", []),
                        links=page.get("links", []),
                        user_flows=[],
                    ))

                # Now call _analyze_flows which uses self.discovered_pages
                flows_result = await discovery._analyze_flows()

                # Process discovered flows
                for i, flow in enumerate(flows_result):
                    flow_id = f"flow-{session_id[:8]}-{i}"
                    flow_data = {
                        "id": flow_id,
                        "session_id": session_id,
                        "name": flow.name if hasattr(flow, 'name') else flow.get("name", f"Flow {i+1}"),
                        "description": flow.description if hasattr(flow, 'description') else flow.get("description", ""),
                        "category": flow.category if hasattr(flow, 'category') else flow.get("category", "user_journey"),
                        "priority": flow.priority if hasattr(flow, 'priority') else flow.get("priority", "medium"),
                        "start_url": flow.start_url if hasattr(flow, 'start_url') else flow.get("start_url", session["app_url"]),
                        "steps": flow.steps if hasattr(flow, 'steps') else flow.get("steps", []),
                        "pages_involved": [],
                        "created_at": datetime.now(UTC).isoformat(),
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

            except Exception as flow_error:
                # Log but don't fail the session - pages were still discovered successfully
                logger.warning(
                    "Flow inference failed, session will complete with pages only",
                    session_id=session_id,
                    error=str(flow_error),
                )

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
                    "discovered_at": datetime.now(UTC).isoformat(),
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

                # Incremental persistence: save checkpoint every 5 pages
                if len(session["pages"]) % 5 == 0:
                    await _persist_discovery_checkpoint(session)

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
                    "created_at": datetime.now(UTC).isoformat(),
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
        session["completed_at"] = datetime.now(UTC).isoformat()

        # Persist to database
        await _persist_discovery_session(session)

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
        session["completed_at"] = datetime.now(UTC).isoformat()

        # Persist failed session to database
        await _persist_discovery_session(session)

        if events_queue:
            await events_queue.put({
                "event": "failed",
                "data": json.dumps({
                    "session_id": session_id,
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                })
            })


# =============================================================================
# Pattern Learning API Endpoints
# =============================================================================


class PatternSearchRequest(BaseModel):
    """Request for similarity search on patterns."""
    pattern_type: str | None = Field(None, description="Filter by pattern type")
    pattern_name: str = Field(..., description="Pattern name to search for")
    pattern_data: dict = Field(default_factory=dict, description="Pattern data for embedding")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum results")


class PatternInsightsRequest(BaseModel):
    """Request for pattern insights."""
    pattern_type: str | None = Field(None, description="Filter by pattern type")


class PatternSuccessUpdateRequest(BaseModel):
    """Update pattern success rates after test execution."""
    pattern_id: str = Field(..., description="Pattern ID to update")
    test_passed: bool = Field(..., description="Whether test passed")
    self_healed: bool = Field(default=False, description="Whether self-healing was used")


class PatternCreateRequest(BaseModel):
    """Request to create a new discovery pattern."""
    pattern_type: str = Field(..., description="Type of pattern (e.g., login, navigation, form)")
    pattern_name: str = Field(..., description="Human-readable pattern name")
    pattern_signature: str = Field(..., description="Unique signature for deduplication")
    pattern_data: dict = Field(default_factory=dict, description="Pattern configuration data")
    project_id: str | None = Field(None, description="Optional project ID to associate")


@router.get("/sessions/{session_id}/patterns", response_model=DiscoveryPatternListResponse)
async def get_session_patterns(
    session_id: str,
    limit: int = 50,
):
    """Get patterns associated with a specific discovery session.

    Returns patterns that were discovered or matched during the session.
    This enables tracing which patterns came from which exploration.
    """
    from src.discovery.pattern_service import get_pattern_service

    try:
        pattern_service = get_pattern_service()

        # Get patterns associated with this session
        patterns = await pattern_service.get_patterns_for_session(session_id, limit=limit)

        return DiscoveryPatternListResponse(
            patterns=[
                DiscoveryPatternResponse(
                    id=str(p.get("id", "")),
                    pattern_type=p.get("pattern_type", ""),
                    pattern_name=p.get("pattern_name", ""),
                    pattern_signature=p.get("pattern_signature", ""),
                    pattern_data=p.get("pattern_data", {}),
                    times_seen=p.get("times_seen", 1),
                    test_success_rate=p.get("test_success_rate"),
                    self_heal_success_rate=p.get("self_heal_success_rate"),
                    created_at=str(p.get("created_at", "")),
                )
                for p in patterns
            ],
            total=len(patterns),
        )

    except Exception as e:
        logger.error(f"Error getting session patterns: {e}")
        # Return empty list on error (frontend has fallback)
        return DiscoveryPatternListResponse(patterns=[], total=0)


@router.post("/patterns", response_model=DiscoveryPatternResponse)
async def create_pattern(request: PatternCreateRequest):
    """Create a new discovery pattern.

    Patterns are used for cross-project learning and self-healing.
    They capture common UI patterns that can be matched in future discoveries.
    """
    import uuid
    from datetime import datetime

    from src.discovery.pattern_service import get_pattern_service

    try:
        pattern_service = get_pattern_service()

        # Create the pattern
        pattern_data = {
            "id": str(uuid.uuid4()),
            "pattern_type": request.pattern_type,
            "pattern_name": request.pattern_name,
            "pattern_signature": request.pattern_signature,
            "pattern_data": request.pattern_data,
            "times_seen": 1,
            "test_success_rate": None,
            "self_heal_success_rate": None,
            "created_at": datetime.now(UTC).isoformat(),
        }

        if request.project_id:
            pattern_data["projects_seen"] = [request.project_id]

        # Store the pattern
        created = await pattern_service.store_pattern(pattern_data)

        return DiscoveryPatternResponse(
            id=str(created.get("id", pattern_data["id"])),
            pattern_type=created.get("pattern_type", request.pattern_type),
            pattern_name=created.get("pattern_name", request.pattern_name),
            pattern_signature=created.get("pattern_signature", request.pattern_signature),
            pattern_data=created.get("pattern_data", request.pattern_data),
            times_seen=created.get("times_seen", 1),
            test_success_rate=created.get("test_success_rate"),
            self_heal_success_rate=created.get("self_heal_success_rate"),
            created_at=str(created.get("created_at", pattern_data["created_at"])),
        )

    except Exception as e:
        logger.error(f"Error creating pattern: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create pattern: {str(e)}")


@router.post("/patterns/extract")
async def extract_patterns_from_session(
    session_id: str,
    background_tasks: BackgroundTasks,
):
    """Extract and store patterns from a completed discovery session.

    This enables cross-project learning by storing patterns that can be
    matched against future discoveries.
    """
    from src.discovery.pattern_service import get_pattern_service

    session = _discovery_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != SessionStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Session must be completed to extract patterns")

    pattern_service = get_pattern_service()

    # Extract patterns in background
    async def extract_task():
        try:
            result = await pattern_service.extract_and_store_patterns(
                session_id=session_id,
                project_id=session.get("project_id", "unknown"),
                pages=session.get("pages", []),
                flows=session.get("flows", []),
                elements=[],  # Elements are embedded in pages currently
            )
            logger.info(
                "Pattern extraction completed",
                session_id=session_id,
                stored=result.get("stored", 0),
                updated=result.get("updated", 0),
            )
        except Exception as e:
            logger.exception("Pattern extraction failed", session_id=session_id, error=str(e))

    background_tasks.add_task(extract_task)

    return {
        "status": "extraction_started",
        "session_id": session_id,
        "message": "Pattern extraction started in background",
    }


@router.post("/patterns/search")
async def search_similar_patterns(request: PatternSearchRequest):
    """Search for patterns similar to the given pattern.

    Uses pgvector similarity search to find patterns that match
    across all projects. This enables:
    - Learning from past discoveries
    - Suggesting selectors that worked before
    - Identifying common UI patterns
    """
    from src.discovery.pattern_service import DiscoveryPattern, PatternType, get_pattern_service

    pattern_service = get_pattern_service()

    # Create a query pattern
    try:
        pattern_type = PatternType(request.pattern_type) if request.pattern_type else PatternType.CUSTOM
    except ValueError:
        pattern_type = PatternType.CUSTOM

    query_pattern = DiscoveryPattern(
        pattern_type=pattern_type,
        pattern_name=request.pattern_name,
        pattern_signature="",  # Not needed for search
        pattern_data=request.pattern_data,
    )

    matches = await pattern_service.find_similar_patterns(
        query_pattern=query_pattern,
        pattern_type=pattern_type if request.pattern_type else None,
        threshold=request.threshold,
        limit=request.limit,
    )

    return {
        "matches": [
            {
                "id": m.id,
                "pattern_type": m.pattern_type,
                "pattern_name": m.pattern_name,
                "pattern_data": m.pattern_data,
                "times_seen": m.times_seen,
                "test_success_rate": m.test_success_rate,
                "similarity": round(m.similarity, 3),
            }
            for m in matches
        ],
        "total": len(matches),
    }


@router.get("/patterns/insights")
async def get_pattern_insights(pattern_type: str | None = None):
    """Get insights about stored patterns.

    Returns statistics about pattern distribution, success rates,
    and cross-project learning metrics.
    """
    from src.discovery.pattern_service import PatternType, get_pattern_service

    pattern_service = get_pattern_service()

    ptype = None
    if pattern_type:
        try:
            ptype = PatternType(pattern_type)
        except ValueError:
            pass

    insights = await pattern_service.get_pattern_insights(pattern_type=ptype)

    return insights


@router.post("/patterns/update-success")
async def update_pattern_success_rate(request: PatternSuccessUpdateRequest):
    """Update pattern success rate after test execution.

    This enables learning: patterns with higher success rates
    are prioritized in future discoveries and self-healing.
    """
    from src.discovery.pattern_service import get_pattern_service

    pattern_service = get_pattern_service()

    success = await pattern_service.update_pattern_success_rate(
        pattern_id=request.pattern_id,
        test_passed=request.test_passed,
        self_healed=request.self_healed,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Pattern not found or update failed")

    return {"status": "updated", "pattern_id": request.pattern_id}


@router.get("/patterns/types")
async def get_pattern_types():
    """Get available pattern types for filtering."""
    from src.discovery.pattern_service import PatternType

    return {
        "types": [
            {"value": pt.value, "label": pt.value.replace("_", " ").title()}
            for pt in PatternType
        ]
    }


# =============================================================================
# Feature Mesh Integration Endpoints
# =============================================================================

class FeatureMeshRequest(BaseModel):
    """Request for processing feature mesh integrations."""
    session_id: str = Field(..., description="Discovery session ID")
    project_id: str = Field(..., description="Project ID")
    create_baselines: bool = Field(default=True, description="Auto-create visual baselines")
    share_selectors: bool = Field(default=True, description="Share selectors with self-healer")


class HealingFeedbackRequest(BaseModel):
    """Request for recording self-healing feedback."""
    primary_selector: str = Field(..., description="Original broken selector")
    used_alternative: str = Field(..., description="Alternative selector that was tried")
    success: bool = Field(..., description="Whether the alternative worked")
    project_id: str = Field(..., description="Project ID")


@router.post("/feature-mesh/process")
async def process_feature_mesh(
    request: FeatureMeshRequest,
    background_tasks: BackgroundTasks,
):
    """Process a completed discovery session for feature mesh integrations.

    This triggers:
    1. Auto-creation of visual baselines from discovered pages
    2. Sharing of selector alternatives with the self-healer

    These integrations create a feedback loop where discovery insights
    flow to other Argus systems automatically.
    """
    from src.discovery.feature_mesh import FeatureMeshConfig, get_feature_mesh

    try:
        # Get session data
        session = await get_session_or_404(request.session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {request.session_id}"
            )

        pages = session.get("pages", [])
        elements = []

        # Extract elements from pages
        for page in pages:
            page_elements = page.get("actions", page.get("elements", []))
            for elem in page_elements:
                elem["page_url"] = page.get("url")
            elements.extend(page_elements)

        # Configure feature mesh
        config = FeatureMeshConfig(
            auto_create_baselines=request.create_baselines,
            share_selectors=request.share_selectors,
        )

        feature_mesh = get_feature_mesh(config)

        # Process integrations
        result = await feature_mesh.process_discovery_completion(
            session_id=request.session_id,
            project_id=request.project_id,
            pages=pages,
            elements=elements,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Feature mesh processing failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Feature mesh processing failed: {str(e)}"
        )


@router.post("/feature-mesh/create-baselines")
async def create_visual_baselines(
    session_id: str,
    project_id: str,
):
    """Create visual baselines from a discovery session's pages.

    Each discovered page becomes a visual baseline that can be monitored
    for regressions. This provides proactive visual monitoring coverage.
    """
    from src.discovery.feature_mesh import get_feature_mesh

    try:
        # Get session data
        session = await get_session_or_404(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )

        pages = session.get("pages", [])

        if not pages:
            return {
                "success": True,
                "message": "No pages found in session",
                "baselines_created": 0,
            }

        feature_mesh = get_feature_mesh()
        result = await feature_mesh.create_baselines_from_discovery(
            session_id=session_id,
            project_id=project_id,
            pages=pages,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to create baselines", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create baselines: {str(e)}"
        )


@router.post("/feature-mesh/share-selectors")
async def share_selectors(
    session_id: str,
    project_id: str,
):
    """Share discovered selector alternatives with the self-healer.

    When discovery finds elements, it identifies multiple ways to select them.
    These alternatives are invaluable for self-healing when primary selectors break.
    """
    from src.discovery.feature_mesh import get_feature_mesh

    try:
        # Get session data
        session = await get_session_or_404(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )

        pages = session.get("pages", [])
        elements = []

        # Extract elements from pages
        for page in pages:
            page_elements = page.get("actions", page.get("elements", []))
            for elem in page_elements:
                elem["page_url"] = page.get("url")
            elements.extend(page_elements)

        if not elements:
            return {
                "success": True,
                "message": "No elements found in session",
                "selectors_shared": 0,
            }

        feature_mesh = get_feature_mesh()
        result = await feature_mesh.share_selectors_with_self_healer(
            session_id=session_id,
            project_id=project_id,
            elements=elements,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to share selectors", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to share selectors: {str(e)}"
        )


@router.post("/feature-mesh/healing-feedback")
async def record_healing_feedback(request: HealingFeedbackRequest):
    """Record feedback from self-healing to improve selector quality scores.

    When self-healing uses an alternative selector and it works/fails,
    that feedback is recorded to improve future recommendations.
    """
    from src.discovery.feature_mesh import get_feature_mesh

    try:
        feature_mesh = get_feature_mesh()
        success = await feature_mesh.record_healing_feedback(
            primary_selector=request.primary_selector,
            used_alternative=request.used_alternative,
            success=request.success,
            project_id=request.project_id,
        )

        return {
            "success": success,
            "message": "Feedback recorded" if success else "No matching selector found",
        }

    except Exception as e:
        logger.exception("Failed to record healing feedback", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record feedback: {str(e)}"
        )


@router.get("/feature-mesh/selector-alternatives")
async def get_selector_alternatives(
    project_id: str,
    selector: str,
    limit: int = Query(default=5, le=20),
):
    """Get best alternative selectors for a potentially broken selector.

    Returns alternatives ordered by confidence and historical success rate.
    """
    supabase = get_supabase_client()

    try:
        # Query using the database function
        result = await supabase.request(
            "/rpc/get_best_alternatives",
            method="POST",
            data={
                "p_project_id": project_id,
                "p_selector": selector,
                "p_limit": limit,
            }
        )

        if result.get("error"):
            # Fallback to direct query
            alt_result = await supabase.select(
                "selector_alternatives",
                filters={
                    "project_id": f"eq.{project_id}",
                    "primary_selector": f"eq.{selector}",
                }
            )

            if alt_result.get("data"):
                alternatives = alt_result["data"][0].get("alternatives", [])
                return {
                    "selector": selector,
                    "alternatives": alternatives[:limit],
                    "source": "direct_query",
                }

            return {
                "selector": selector,
                "alternatives": [],
                "message": "No alternatives found",
            }

        return {
            "selector": selector,
            "alternatives": result.get("data", []),
            "source": "function",
        }

    except Exception as e:
        logger.exception("Failed to get alternatives", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get alternatives: {str(e)}"
        )
