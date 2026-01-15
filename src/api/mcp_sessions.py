"""MCP Sessions Management API.

Provides endpoints for:
- Listing active MCP connections
- Viewing connection details and activity
- Revoking MCP sessions
- Dashboard statistics

These endpoints allow users to see and manage MCP clients (Claude Code, Cursor, etc.)
that are connected to their Argus account.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List
from enum import Enum

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field
import structlog

from src.services.supabase_client import get_supabase_client
from src.api.teams import get_current_user, verify_org_access, log_audit, translate_clerk_org_id

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/mcp", tags=["MCP Sessions"])


# ============================================================================
# Enums and Constants
# ============================================================================


class ConnectionStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"


class ActivityType(str, Enum):
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    TOOL_CALL = "tool_call"
    AUTH_REFRESH = "auth_refresh"
    ERROR = "error"


# ============================================================================
# Request/Response Models
# ============================================================================


class MCPConnectionResponse(BaseModel):
    """MCP connection details for API response."""

    id: str
    user_id: str
    organization_id: Optional[str]
    client_id: str
    client_name: Optional[str]
    client_type: str
    session_id: Optional[str]
    device_name: Optional[str]
    ip_address: Optional[str]
    scopes: List[str]
    status: str
    last_activity_at: str
    request_count: int
    tools_used: List[str]
    connected_at: str
    disconnected_at: Optional[str]
    revoked_at: Optional[str]
    # Computed fields
    is_active: bool
    seconds_since_activity: Optional[float]
    connection_duration_hours: Optional[float]


class MCPConnectionListResponse(BaseModel):
    """List of MCP connections."""

    connections: List[MCPConnectionResponse]
    total: int
    active_count: int


class MCPActivityResponse(BaseModel):
    """Activity log entry."""

    id: str
    connection_id: str
    activity_type: str
    tool_name: Optional[str]
    request_id: Optional[str]
    duration_ms: Optional[int]
    success: bool
    error_message: Optional[str]
    created_at: str


class MCPActivityListResponse(BaseModel):
    """List of activity entries."""

    activities: List[MCPActivityResponse]
    total: int


class MCPStatsResponse(BaseModel):
    """MCP connection statistics for dashboard."""

    active_connections: int
    total_connections: int
    total_requests: int
    unique_users: int
    client_types: List[str]
    last_activity: Optional[str]
    # Time-based stats
    connections_today: int
    connections_this_week: int
    requests_today: int
    # Top tools
    top_tools: List[dict]


class RevokeConnectionRequest(BaseModel):
    """Request to revoke an MCP connection."""

    reason: Optional[str] = Field(default="User revoked", max_length=500)


class RegisterConnectionRequest(BaseModel):
    """Request to register a new MCP connection (called by MCP server)."""

    session_id: str
    client_id: str = "argus-mcp"
    client_name: Optional[str] = None
    client_type: str = "mcp"
    device_name: Optional[str] = None
    scopes: List[str] = Field(default=["read", "write"])
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class RecordActivityRequest(BaseModel):
    """Request to record tool usage activity."""

    connection_id: str
    tool_name: str
    request_id: Optional[str] = None
    duration_ms: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================


def format_connection(row: dict) -> MCPConnectionResponse:
    """Format a database row into a connection response."""
    now = datetime.now(timezone.utc)
    last_activity = None
    if row.get("last_activity_at"):
        try:
            last_activity = datetime.fromisoformat(row["last_activity_at"].replace("Z", "+00:00"))
        except:
            pass

    connected_at = None
    if row.get("connected_at"):
        try:
            connected_at = datetime.fromisoformat(row["connected_at"].replace("Z", "+00:00"))
        except:
            pass

    seconds_since = None
    if last_activity:
        seconds_since = (now - last_activity).total_seconds()

    duration_hours = None
    if connected_at:
        duration_hours = (now - connected_at).total_seconds() / 3600

    return MCPConnectionResponse(
        id=str(row["id"]),
        user_id=row["user_id"],
        organization_id=str(row["organization_id"]) if row.get("organization_id") else None,
        client_id=row.get("client_id", "argus-mcp"),
        client_name=row.get("client_name"),
        client_type=row.get("client_type", "mcp"),
        session_id=row.get("session_id"),
        device_name=row.get("device_name"),
        ip_address=str(row["ip_address"]) if row.get("ip_address") else None,
        scopes=row.get("scopes", ["read", "write"]),
        status=row.get("status", "active"),
        last_activity_at=row.get("last_activity_at", ""),
        request_count=row.get("request_count", 0),
        tools_used=row.get("tools_used", []) or [],
        connected_at=row.get("connected_at", ""),
        disconnected_at=row.get("disconnected_at"),
        revoked_at=row.get("revoked_at"),
        is_active=row.get("status") == "active",
        seconds_since_activity=seconds_since,
        connection_duration_hours=duration_hours,
    )


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/connections", response_model=MCPConnectionListResponse)
async def list_mcp_connections(
    request: Request,
    org_id: Optional[str] = Query(None, description="Organization ID"),
    status: Optional[ConnectionStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List MCP connections for the current user or organization.

    Returns active and historical MCP client connections that can be
    viewed in the dashboard.
    """
    user = await get_current_user(request)
    supabase = await get_supabase_client()

    # Build query
    query = "/mcp_connections?select=*&order=last_activity_at.desc"

    if org_id:
        # Get org connections (requires org access)
        supabase_org_id = await translate_clerk_org_id(org_id)
        await verify_org_access(org_id, user["user_id"])
        query += f"&organization_id=eq.{supabase_org_id}"
    else:
        # Get user's own connections
        query += f"&user_id=eq.{user["user_id"]}"

    if status:
        query += f"&status=eq.{status.value}"

    query += f"&limit={limit}&offset={offset}"

    result = await supabase.select(query)

    if not result.get("success"):
        raise HTTPException(status_code=500, detail="Failed to fetch connections")

    connections = [format_connection(row) for row in result.get("data", [])]
    active_count = sum(1 for c in connections if c.is_active)

    return MCPConnectionListResponse(
        connections=connections,
        total=len(connections),
        active_count=active_count,
    )


@router.get("/connections/{connection_id}", response_model=MCPConnectionResponse)
async def get_mcp_connection(
    request: Request,
    connection_id: str,
):
    """Get details for a specific MCP connection."""
    user = await get_current_user(request)
    supabase = await get_supabase_client()

    result = await supabase.select(f"/mcp_connections?id=eq.{connection_id}&select=*")

    if not result.get("success") or not result.get("data"):
        raise HTTPException(status_code=404, detail="Connection not found")

    connection = result["data"][0]

    # Verify access
    if connection["user_id"] != user["user_id"]:
        if connection.get("organization_id"):
            await verify_org_access(str(connection["organization_id"]), user["user_id"])
        else:
            raise HTTPException(status_code=403, detail="Access denied")

    return format_connection(connection)


@router.delete("/connections/{connection_id}")
async def revoke_mcp_connection(
    request: Request,
    connection_id: str,
    body: RevokeConnectionRequest = None,
):
    """Revoke an MCP connection.

    This will invalidate the session and prevent further API calls
    from this connection. The client will need to re-authenticate.
    """
    user = await get_current_user(request)
    supabase = await get_supabase_client()

    # Get the connection first
    result = await supabase.select(f"/mcp_connections?id=eq.{connection_id}&select=*")

    if not result.get("success") or not result.get("data"):
        raise HTTPException(status_code=404, detail="Connection not found")

    connection = result["data"][0]

    # Verify access
    if connection["user_id"] != user["user_id"]:
        if connection.get("organization_id"):
            await verify_org_access(str(connection["organization_id"]), user["user_id"])
        else:
            raise HTTPException(status_code=403, detail="Access denied")

    if connection["status"] != "active":
        raise HTTPException(status_code=400, detail="Connection is not active")

    # Revoke the connection
    reason = body.reason if body else "User revoked"
    update_result = await supabase.update(
        "mcp_connections",
        {
            "status": "revoked",
            "revoked_at": datetime.now(timezone.utc).isoformat(),
            "revoked_by": user["user_id"],
            "revoke_reason": reason,
        },
        f"id=eq.{connection_id}",
    )

    if not update_result.get("success"):
        raise HTTPException(status_code=500, detail="Failed to revoke connection")

    # Log activity
    await supabase.insert(
        "mcp_connection_activity",
        {
            "connection_id": connection_id,
            "activity_type": "disconnect",
            "metadata": {"reason": reason, "revoked_by": user["user_id"]},
        },
    )

    # Audit log
    await log_audit(
        supabase=supabase,
        user_id=user["user_id"],
        action="mcp.connection.revoked",
        resource_type="mcp_connection",
        resource_id=connection_id,
        details={"reason": reason, "client_name": connection.get("client_name")},
    )

    logger.info(
        "MCP connection revoked",
        connection_id=connection_id,
        revoked_by=user["user_id"],
        reason=reason,
    )

    return {"status": "revoked", "connection_id": connection_id}


@router.get("/connections/{connection_id}/activity", response_model=MCPActivityListResponse)
async def get_connection_activity(
    request: Request,
    connection_id: str,
    activity_type: Optional[ActivityType] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Get activity log for a specific MCP connection."""
    user = await get_current_user(request)
    supabase = await get_supabase_client()

    # Verify access to the connection
    conn_result = await supabase.select(
        f"/mcp_connections?id=eq.{connection_id}&select=user_id,organization_id"
    )

    if not conn_result.get("success") or not conn_result.get("data"):
        raise HTTPException(status_code=404, detail="Connection not found")

    connection = conn_result["data"][0]
    if connection["user_id"] != user["user_id"]:
        if connection.get("organization_id"):
            await verify_org_access(str(connection["organization_id"]), user["user_id"])
        else:
            raise HTTPException(status_code=403, detail="Access denied")

    # Get activity
    query = (
        f"/mcp_connection_activity?connection_id=eq.{connection_id}&select=*&order=created_at.desc"
    )

    if activity_type:
        query += f"&activity_type=eq.{activity_type.value}"

    query += f"&limit={limit}&offset={offset}"

    result = await supabase.select(query)

    activities = [
        MCPActivityResponse(
            id=str(row["id"]),
            connection_id=str(row["connection_id"]),
            activity_type=row["activity_type"],
            tool_name=row.get("tool_name"),
            request_id=row.get("request_id"),
            duration_ms=row.get("duration_ms"),
            success=row.get("success", True),
            error_message=row.get("error_message"),
            created_at=row.get("created_at", ""),
        )
        for row in result.get("data", [])
    ]

    return MCPActivityListResponse(
        activities=activities,
        total=len(activities),
    )


@router.get("/stats", response_model=MCPStatsResponse)
async def get_mcp_stats(
    request: Request,
    org_id: Optional[str] = Query(None, description="Organization ID"),
):
    """Get MCP connection statistics for the dashboard.

    Returns aggregated stats about MCP connections including
    active connections, tool usage, and trends.
    """
    user = await get_current_user(request)
    supabase = await get_supabase_client()

    # Build base query
    if org_id:
        supabase_org_id = await translate_clerk_org_id(org_id)
        await verify_org_access(org_id, user["user_id"])
        filter_clause = f"organization_id=eq.{supabase_org_id}"
    else:
        filter_clause = f"user_id=eq.{user["user_id"]}"

    # Get all connections for this user/org
    result = await supabase.select(f"/mcp_connections?{filter_clause}&select=*")

    connections = result.get("data", [])

    # Calculate stats
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)

    active_connections = sum(1 for c in connections if c.get("status") == "active")
    total_requests = sum(c.get("request_count", 0) for c in connections)
    unique_users = len(set(c.get("user_id") for c in connections))

    # Get client types
    client_types = list(
        set(
            c.get("client_name") or c.get("client_type", "mcp")
            for c in connections
            if c.get("client_name") or c.get("client_type")
        )
    )

    # Last activity
    last_activity = None
    for c in connections:
        if c.get("last_activity_at"):
            last_activity = c["last_activity_at"]
            break

    # Connections today/this week
    connections_today = 0
    connections_this_week = 0
    requests_today = 0

    for c in connections:
        connected_at = c.get("connected_at")
        if connected_at:
            try:
                conn_time = datetime.fromisoformat(connected_at.replace("Z", "+00:00"))
                if conn_time >= today_start:
                    connections_today += 1
                if conn_time >= week_start:
                    connections_this_week += 1
            except:
                pass

        # Estimate requests today (would need activity table for accurate count)
        last_act = c.get("last_activity_at")
        if last_act:
            try:
                act_time = datetime.fromisoformat(last_act.replace("Z", "+00:00"))
                if act_time >= today_start and c.get("status") == "active":
                    # Rough estimate
                    requests_today += min(c.get("request_count", 0), 100)
            except:
                pass

    # Aggregate tool usage
    tool_counts = {}
    for c in connections:
        for tool in c.get("tools_used", []) or []:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    top_tools = [
        {"tool": tool, "count": count}
        for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:10]
    ]

    return MCPStatsResponse(
        active_connections=active_connections,
        total_connections=len(connections),
        total_requests=total_requests,
        unique_users=unique_users,
        client_types=client_types,
        last_activity=last_activity,
        connections_today=connections_today,
        connections_this_week=connections_this_week,
        requests_today=requests_today,
        top_tools=top_tools,
    )


# ============================================================================
# Internal Endpoints (Called by MCP Server)
# ============================================================================


@router.post("/connections/register", include_in_schema=False)
async def register_mcp_connection(
    request: Request,
    body: RegisterConnectionRequest,
):
    """Register a new MCP connection (internal endpoint).

    Called by the MCP server when a client authenticates successfully.
    This creates or updates the connection record.
    """
    user = await get_current_user(request)
    supabase = await get_supabase_client()

    # Check for existing connection with same session
    existing = await supabase.select(
        f"/mcp_connections?session_id=eq.{body.session_id}&status=eq.active&select=id"
    )

    if existing.get("data"):
        # Update existing connection
        connection_id = existing["data"][0]["id"]
        await supabase.update(
            "mcp_connections",
            {
                "last_activity_at": datetime.now(timezone.utc).isoformat(),
                "ip_address": body.ip_address,
                "user_agent": body.user_agent,
            },
            f"id=eq.{connection_id}",
        )
    else:
        # Get organization ID for user
        org_id = None
        if user.get("organization_id"):
            org_id = user.get("organization_id")

        # Create new connection
        result = await supabase.insert(
            "mcp_connections",
            {
                "user_id": user["user_id"],
                "organization_id": org_id,
                "session_id": body.session_id,
                "client_id": body.client_id,
                "client_name": body.client_name,
                "client_type": body.client_type,
                "device_name": body.device_name,
                "ip_address": body.ip_address,
                "user_agent": body.user_agent,
                "scopes": body.scopes,
            },
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail="Failed to register connection")

        connection_id = result["data"][0]["id"]

        # Log connect activity
        await supabase.insert(
            "mcp_connection_activity",
            {
                "connection_id": connection_id,
                "activity_type": "connect",
                "metadata": {
                    "client_name": body.client_name,
                    "client_type": body.client_type,
                },
            },
        )

    logger.info(
        "MCP connection registered",
        session_id=body.session_id,
        user_id=user["user_id"],
        client_name=body.client_name,
    )

    return {"status": "registered", "connection_id": connection_id}


@router.post("/connections/activity", include_in_schema=False)
async def record_mcp_activity(
    request: Request,
    body: RecordActivityRequest,
):
    """Record tool usage activity (internal endpoint).

    Called by the MCP server after each tool invocation.
    """
    user = await get_current_user(request)
    supabase = await get_supabase_client()

    # Verify connection belongs to user
    conn_result = await supabase.select(
        f"/mcp_connections?id=eq.{body.connection_id}&user_id=eq.{user["user_id"]}&select=id"
    )

    if not conn_result.get("data"):
        raise HTTPException(status_code=404, detail="Connection not found")

    # Update connection stats
    await supabase.rpc(
        "record_mcp_tool_usage",
        {
            "p_connection_id": body.connection_id,
            "p_tool_name": body.tool_name,
            "p_request_id": body.request_id,
            "p_duration_ms": body.duration_ms,
            "p_success": body.success,
            "p_error_message": body.error_message,
        },
    )

    return {"status": "recorded"}


# ============================================================================
# Device Auth Session Endpoints
# ============================================================================


@router.get("/auth/pending")
async def get_pending_auth_sessions(
    request: Request,
):
    """Get pending device authorization sessions for current user.

    Shows sessions that are waiting for approval.
    """
    user = await get_current_user(request)
    supabase = await get_supabase_client()

    result = await supabase.select(
        f"/device_auth_sessions?user_id=eq.{user["user_id"]}&status=eq.pending&select=*&order=created_at.desc"
    )

    return {
        "sessions": result.get("data", []),
        "count": len(result.get("data", [])),
    }
