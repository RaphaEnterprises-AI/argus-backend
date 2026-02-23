# src/api/swarms.py
"""Swarm API endpoints.

Provides REST + SSE endpoints for launching, streaming, and managing agent swarms.

Endpoints:
- POST /api/v1/swarms/launch — Start a new swarm
- GET  /api/v1/swarms/{swarm_id}/stream — SSE stream of AG-UI events
- DELETE /api/v1/swarms/{swarm_id} — Cancel a running swarm
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.api.context import require_organization_id
from src.api.security.auth import get_current_user
from src.orchestrator.swarm_orchestrator import (
    SwarmConfig,
    SwarmMode,
    SwarmQuotaExceeded,
    get_swarm_orchestrator,
)
from src.streaming.agui_emitter import get_emitter

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/swarms", tags=["Swarms"])


class LaunchSwarmRequest(BaseModel):
    """Request to launch a new agent swarm."""

    mode: str = Field(
        ...,
        description="Swarm mode: full_crawl, targeted_blitz, or pr_analysis",
    )
    project_id: str = Field(..., description="Project ID")
    target_url: str | None = Field(None, description="URL to test (full_crawl/blitz)")
    target_flow: str | None = Field(None, description="Specific flow to test (blitz)")
    pr_number: int | None = Field(None, description="PR number (pr_analysis)")
    changed_files: list[str] | None = Field(None, description="Changed files (pr_analysis)")
    agent_types: list[str] | None = Field(
        None, description="Override default agent types for the mode"
    )


class LaunchSwarmResponse(BaseModel):
    """Response from launching a swarm."""

    swarm_id: str
    mode: str
    worker_count: int
    stream_url: str


@router.post("/launch")
async def launch_swarm(request: Request, body: LaunchSwarmRequest):
    """Launch a new agent swarm.

    Returns the swarm_id and stream_url for SSE consumption.
    """
    user = await get_current_user(request)
    org_id = await require_organization_id(request)

    # Validate mode
    try:
        mode = SwarmMode(body.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {body.mode}. Must be one of: {[m.value for m in SwarmMode]}",
        )

    config = SwarmConfig(
        mode=mode,
        org_id=org_id,
        project_id=body.project_id,
        user_id=user["user_id"],
        target_url=body.target_url,
        target_flow=body.target_flow,
        pr_number=body.pr_number,
        changed_files=body.changed_files,
        agent_types=body.agent_types,
    )

    orchestrator = get_swarm_orchestrator()

    try:
        swarm_id, _emitter = await orchestrator.launch(config)
    except SwarmQuotaExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))

    # Determine worker count
    from src.orchestrator.swarm_orchestrator import SWARM_MODE_AGENTS

    agent_types = body.agent_types or SWARM_MODE_AGENTS.get(
        mode, SWARM_MODE_AGENTS[SwarmMode.TARGETED_BLITZ]
    )

    return LaunchSwarmResponse(
        swarm_id=swarm_id,
        mode=mode.value,
        worker_count=len(agent_types),
        stream_url=f"/api/v1/swarms/{swarm_id}/stream",
    )


@router.get("/{swarm_id}/stream")
async def stream_swarm(swarm_id: str, request: Request):
    """Stream AG-UI events for a swarm via Server-Sent Events."""
    emitter = get_emitter(swarm_id)
    if not emitter:
        raise HTTPException(status_code=404, detail="Swarm not found or already completed")

    return EventSourceResponse(emitter.stream())


@router.delete("/{swarm_id}")
async def cancel_swarm(swarm_id: str, request: Request):
    """Cancel a running swarm."""
    orchestrator = get_swarm_orchestrator()
    cancelled = await orchestrator.cancel(swarm_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Swarm not found or already completed")
    return {"status": "cancelled", "swarm_id": swarm_id}
