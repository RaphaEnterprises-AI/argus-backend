"""MCP Screenshots API.

Handles screenshot registration and retrieval for MCP sessions.
"""

from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import structlog

from src.services.supabase_client import get_supabase_client
from src.api.teams import get_current_user

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/mcp/screenshots", tags=["MCP Screenshots"])


class RegisterScreenshotRequest(BaseModel):
    connection_id: str
    activity_id: Optional[str] = None
    r2_key: str
    step_index: Optional[int] = None
    screenshot_type: str = "step"
    tool_name: Optional[str] = None
    url_tested: Optional[str] = None
    file_size_bytes: Optional[int] = None


class RegisterScreenshotResponse(BaseModel):
    screenshot_id: str
    r2_key: str
    created_at: str


@router.post("/register", response_model=RegisterScreenshotResponse)
async def register_screenshot(request: Request, body: RegisterScreenshotRequest):
    """Register a screenshot in the database (called by MCP server)."""
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Get connection to find org_id
    conn = await supabase.request(
        f"/mcp_connections?id=eq.{body.connection_id}&select=organization_id,user_id"
    )

    if not conn.get("data"):
        raise HTTPException(status_code=404, detail="Connection not found")

    org_id = conn["data"][0].get("organization_id")

    # Insert screenshot record
    result = await supabase.insert("mcp_screenshots", {
        "connection_id": body.connection_id,
        "activity_id": body.activity_id,
        "organization_id": org_id,
        "r2_key": body.r2_key,
        "step_index": body.step_index,
        "screenshot_type": body.screenshot_type,
        "tool_name": body.tool_name,
        "url_tested": body.url_tested,
        "file_size_bytes": body.file_size_bytes,
    })

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to register screenshot")

    data = result["data"][0]
    return RegisterScreenshotResponse(
        screenshot_id=str(data["id"]),
        r2_key=body.r2_key,
        created_at=data["created_at"]
    )


@router.get("/{screenshot_id}")
async def get_screenshot(request: Request, screenshot_id: str):
    """Get screenshot metadata."""
    user = await get_current_user(request)
    supabase = get_supabase_client()

    result = await supabase.request(
        f"/mcp_screenshots?id=eq.{screenshot_id}&select=*"
    )

    if not result.get("data"):
        raise HTTPException(status_code=404, detail="Screenshot not found")

    return result["data"][0]
