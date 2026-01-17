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
from uuid import UUID
import re

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field
import structlog

from src.services.supabase_client import get_supabase_client
from src.api.teams import get_current_user, verify_org_access, log_audit, translate_clerk_org_id

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/mcp", tags=["MCP Sessions"])


# ============================================================================
# Input Validation Utilities
# ============================================================================

def validate_uuid(value: str, field_name: str = "id") -> str:
    """Validate that a string is a valid UUID.

    Raises HTTPException 400 if invalid, preventing SQL injection
    and ensuring data integrity.
    """
    if not value:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")

    try:
        # Attempt to parse as UUID
        UUID(value)
        return value
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name} format. Expected UUID."
        )


def validate_org_id(org_id: str) -> str:
    """Validate organization ID - accepts both UUID and Clerk formats.

    UUID format: 550e8400-e29b-41d4-a716-446655440000
    Clerk format: org_xxxxx
    """
    if not org_id:
        raise HTTPException(status_code=400, detail="organization_id is required")

    # Clerk org IDs start with "org_"
    if org_id.startswith("org_"):
        # Basic validation for Clerk format
        if not re.match(r'^org_[a-zA-Z0-9]+$', org_id):
            raise HTTPException(status_code=400, detail="Invalid Clerk organization ID format")
        return org_id

    # Otherwise, validate as UUID
    return validate_uuid(org_id, "organization_id")


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
    # Extended fields for richer activity tracking
    screenshot_key: Optional[str] = None  # R2 key for screenshot
    input_tokens: Optional[int] = None    # AI cost tracking
    output_tokens: Optional[int] = None   # AI cost tracking
    metadata: Optional[dict] = None       # Tool-specific data


# ============================================================================
# Helper Functions
# ============================================================================


def format_connection(row: dict) -> MCPConnectionResponse:
    """Format a database row into a connection response.

    Handles null values safely to prevent KeyError exceptions.
    """
    if not row:
        raise ValueError("Cannot format empty row")

    now = datetime.now(timezone.utc)
    last_activity = None
    last_activity_str = row.get("last_activity_at")
    if last_activity_str:
        try:
            last_activity = datetime.fromisoformat(str(last_activity_str).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass

    connected_at = None
    connected_at_str = row.get("connected_at")
    if connected_at_str:
        try:
            connected_at = datetime.fromisoformat(str(connected_at_str).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass

    seconds_since = None
    if last_activity:
        seconds_since = (now - last_activity).total_seconds()

    duration_hours = None
    if connected_at:
        duration_hours = (now - connected_at).total_seconds() / 3600

    # Handle required fields with safe defaults
    row_id = row.get("id")
    if row_id is None:
        raise ValueError("Connection row missing required 'id' field")

    user_id = row.get("user_id")
    if user_id is None:
        raise ValueError("Connection row missing required 'user_id' field")

    return MCPConnectionResponse(
        id=str(row_id),
        user_id=str(user_id),
        organization_id=str(row["organization_id"]) if row.get("organization_id") else None,
        client_id=row.get("client_id") or "argus-mcp",
        client_name=row.get("client_name"),
        client_type=row.get("client_type") or "mcp",
        session_id=row.get("session_id"),
        device_name=row.get("device_name"),
        ip_address=str(row["ip_address"]) if row.get("ip_address") else None,
        scopes=row.get("scopes") or ["read", "write"],
        status=row.get("status") or "active",
        last_activity_at=str(last_activity_str) if last_activity_str else "",
        request_count=row.get("request_count") or 0,
        tools_used=row.get("tools_used") or [],
        connected_at=str(connected_at_str) if connected_at_str else "",
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
    user_id = user["user_id"]
    supabase = get_supabase_client()

    # Build query
    query = "/mcp_connections?select=*&order=last_activity_at.desc"

    if org_id:
        # INPUT VALIDATION: Validate org_id format
        org_id = validate_org_id(org_id)
        # Get org connections (requires org access) - verify_org_access handles translation
        _, supabase_org_id = await verify_org_access(org_id, user_id, user_email=user.get("email"), request=request)
        query += f"&organization_id=eq.{supabase_org_id}"
    else:
        # Get user's own connections
        query += f"&user_id=eq.{user_id}"

    if status:
        query += f"&status=eq.{status.value}"

    query += f"&limit={limit}&offset={offset}"

    try:
        result = await supabase.request(query)

        if result.get("error"):
            error_msg = result.get("error", "")
            # Handle missing table gracefully
            if "mcp_connections" in str(error_msg) and ("does not exist" in str(error_msg) or "relation" in str(error_msg)):
                logger.warning("mcp_connections table not found, returning empty list")
                return MCPConnectionListResponse(connections=[], total=0, active_count=0)
            raise HTTPException(status_code=500, detail="Failed to fetch connections")

        connections = [format_connection(row) for row in result.get("data", []) if row]
        active_count = sum(1 for c in connections if c.is_active)

        return MCPConnectionListResponse(
            connections=connections,
            total=len(connections),
            active_count=active_count,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error listing MCP connections", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch connections")


@router.get("/connections/{connection_id}", response_model=MCPConnectionResponse)
async def get_mcp_connection(
    request: Request,
    connection_id: str,
):
    """Get details for a specific MCP connection."""
    # INPUT VALIDATION: Prevent injection attacks
    connection_id = validate_uuid(connection_id, "connection_id")

    user = await get_current_user(request)
    supabase = get_supabase_client()

    try:
        result = await supabase.request(f"/mcp_connections?id=eq.{connection_id}&select=*")

        if result.get("error"):
            error_msg = result.get("error", "")
            if "mcp_connections" in str(error_msg) and ("does not exist" in str(error_msg) or "relation" in str(error_msg)):
                raise HTTPException(status_code=404, detail="Connection not found")
            raise HTTPException(status_code=500, detail="Failed to fetch connection")

        if not result.get("data"):
            raise HTTPException(status_code=404, detail="Connection not found")

        connection = result["data"][0]

        # Verify access - check user_id safely
        connection_user_id = connection.get("user_id")
        if connection_user_id != user["user_id"]:
            if connection.get("organization_id"):
                _, _ = await verify_org_access(str(connection["organization_id"]), user["user_id"], user_email=user.get("email"), request=request)
            else:
                raise HTTPException(status_code=403, detail="Access denied")

        return format_connection(connection)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching MCP connection", connection_id=connection_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch connection")


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
    # INPUT VALIDATION: Prevent injection attacks
    connection_id = validate_uuid(connection_id, "connection_id")

    user = await get_current_user(request)
    supabase = get_supabase_client()

    try:
        # Get the connection first
        result = await supabase.request(f"/mcp_connections?id=eq.{connection_id}&select=*")

        if result.get("error"):
            error_msg = result.get("error", "")
            if "mcp_connections" in str(error_msg) and ("does not exist" in str(error_msg) or "relation" in str(error_msg)):
                raise HTTPException(status_code=404, detail="Connection not found")
            raise HTTPException(status_code=500, detail="Failed to fetch connection")

        if not result.get("data"):
            raise HTTPException(status_code=404, detail="Connection not found")

        connection = result["data"][0]

        # Verify access - check user_id safely
        connection_user_id = connection.get("user_id")
        if connection_user_id != user["user_id"]:
            if connection.get("organization_id"):
                _, _ = await verify_org_access(str(connection["organization_id"]), user["user_id"], user_email=user.get("email"), request=request)
            else:
                raise HTTPException(status_code=403, detail="Access denied")

        # Check status safely
        if connection.get("status") != "active":
            raise HTTPException(status_code=400, detail="Connection is not active")

        # Revoke the connection
        reason = body.reason if body else "User revoked"
        update_result = await supabase.update(
            "mcp_connections",
            {"id": f"eq.{connection_id}"},
            {
                "status": "revoked",
                "revoked_at": datetime.now(timezone.utc).isoformat(),
                "revoked_by": user["user_id"],
                "revoke_reason": reason,
            },
        )

        if update_result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to revoke connection")

        # Log activity (non-critical, don't fail if this errors)
        try:
            await supabase.insert(
                "mcp_connection_activity",
                {
                    "connection_id": connection_id,
                    "activity_type": "disconnect",
                    "metadata": {"reason": reason, "revoked_by": user["user_id"]},
                },
            )
        except Exception as e:
            logger.warning("Failed to log revoke activity", connection_id=connection_id, error=str(e))

        # Audit log (skip if no organization - audit logs are org-scoped)
        if connection.get("organization_id"):
            try:
                await log_audit(
                    organization_id=str(connection["organization_id"]),
                    user_id=user["user_id"],
                    user_email=user.get("email", ""),
                    action="mcp.connection.revoked",
                    resource_type="mcp_connection",
                    resource_id=connection_id,
                    description=f"Revoked MCP connection: {connection.get('client_name', 'Unknown')}",
                    metadata={"reason": reason},
                )
            except Exception as e:
                logger.warning("Failed to create audit log", connection_id=connection_id, error=str(e))

        logger.info(
            "MCP connection revoked",
            connection_id=connection_id,
            revoked_by=user["user_id"],
            reason=reason,
        )

        return {"status": "revoked", "connection_id": connection_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error revoking MCP connection", connection_id=connection_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to revoke connection")


@router.get("/connections/{connection_id}/activity", response_model=MCPActivityListResponse)
async def get_connection_activity(
    request: Request,
    connection_id: str,
    activity_type: Optional[ActivityType] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Get activity log for a specific MCP connection."""
    # INPUT VALIDATION: Prevent injection attacks
    connection_id = validate_uuid(connection_id, "connection_id")

    user = await get_current_user(request)
    supabase = get_supabase_client()

    try:
        # Verify access to the connection
        conn_result = await supabase.request(
            f"/mcp_connections?id=eq.{connection_id}&select=user_id,organization_id"
        )

        if conn_result.get("error"):
            error_msg = conn_result.get("error", "")
            if "mcp_connections" in str(error_msg) and ("does not exist" in str(error_msg) or "relation" in str(error_msg)):
                raise HTTPException(status_code=404, detail="Connection not found")
            raise HTTPException(status_code=500, detail="Failed to fetch connection")

        if not conn_result.get("data"):
            raise HTTPException(status_code=404, detail="Connection not found")

        connection = conn_result["data"][0]
        connection_user_id = connection.get("user_id")
        if connection_user_id != user["user_id"]:
            if connection.get("organization_id"):
                _, _ = await verify_org_access(str(connection["organization_id"]), user["user_id"], user_email=user.get("email"), request=request)
            else:
                raise HTTPException(status_code=403, detail="Access denied")

        # Get activity
        query = (
            f"/mcp_connection_activity?connection_id=eq.{connection_id}&select=*&order=created_at.desc"
        )

        if activity_type:
            query += f"&activity_type=eq.{activity_type.value}"

        query += f"&limit={limit}&offset={offset}"

        result = await supabase.request(query)

        # Handle missing activity table gracefully
        if result.get("error"):
            error_msg = result.get("error", "")
            if "mcp_connection_activity" in str(error_msg) and ("does not exist" in str(error_msg) or "relation" in str(error_msg)):
                logger.warning("mcp_connection_activity table not found, returning empty list")
                return MCPActivityListResponse(activities=[], total=0)
            # Log but don't fail for other errors - just return empty list
            logger.warning("Error fetching MCP activity", error=error_msg)
            return MCPActivityListResponse(activities=[], total=0)

        activities = [
            MCPActivityResponse(
                id=str(row["id"]) if row.get("id") else "",
                connection_id=str(row["connection_id"]) if row.get("connection_id") else connection_id,
                activity_type=row.get("activity_type", "unknown"),
                tool_name=row.get("tool_name"),
                request_id=row.get("request_id"),
                duration_ms=row.get("duration_ms"),
                success=row.get("success", True),
                error_message=row.get("error_message"),
                created_at=row.get("created_at", ""),
            )
            for row in result.get("data", [])
            if row
        ]

        return MCPActivityListResponse(
            activities=activities,
            total=len(activities),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching MCP activity", connection_id=connection_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch activity")


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
    user_id = user["user_id"]
    supabase = get_supabase_client()

    # Build base query
    if org_id:
        # INPUT VALIDATION: Validate org_id format
        org_id = validate_org_id(org_id)
        # verify_org_access handles translation internally
        _, supabase_org_id = await verify_org_access(org_id, user_id, user_email=user.get("email"), request=request)
        filter_clause = f"organization_id=eq.{supabase_org_id}"
    else:
        filter_clause = f"user_id=eq.{user_id}"

    try:
        # Get all connections for this user/org
        result = await supabase.request(f"/mcp_connections?{filter_clause}&select=*")

        # Handle missing table gracefully
        if result.get("error"):
            error_msg = result.get("error", "")
            if "mcp_connections" in str(error_msg) and ("does not exist" in str(error_msg) or "relation" in str(error_msg)):
                logger.warning("mcp_connections table not found, returning empty stats")
                return MCPStatsResponse(
                    active_connections=0,
                    total_connections=0,
                    total_requests=0,
                    unique_users=0,
                    client_types=[],
                    last_activity=None,
                    connections_today=0,
                    connections_this_week=0,
                    requests_today=0,
                    top_tools=[],
                )
            logger.warning("Error fetching MCP connections for stats", error=error_msg)

        connections = result.get("data", []) or []

        # Calculate stats
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=7)

        active_connections = sum(1 for c in connections if c and c.get("status") == "active")
        total_requests = sum(c.get("request_count", 0) for c in connections if c)
        unique_users = len(set(c.get("user_id") for c in connections if c and c.get("user_id")))

        # Get client types
        client_types = list(
            set(
                c.get("client_name") or c.get("client_type", "mcp")
                for c in connections
                if c and (c.get("client_name") or c.get("client_type"))
            )
        )

        # Last activity
        last_activity = None
        for c in connections:
            if c and c.get("last_activity_at"):
                last_activity = c["last_activity_at"]
                break

        # Connections today/this week
        connections_today = 0
        connections_this_week = 0
        requests_today = 0

        for c in connections:
            if not c:
                continue
            connected_at = c.get("connected_at")
            if connected_at:
                try:
                    conn_time = datetime.fromisoformat(connected_at.replace("Z", "+00:00"))
                    if conn_time >= today_start:
                        connections_today += 1
                    if conn_time >= week_start:
                        connections_this_week += 1
                except Exception:
                    pass

            # Estimate requests today (would need activity table for accurate count)
            last_act = c.get("last_activity_at")
            if last_act:
                try:
                    act_time = datetime.fromisoformat(last_act.replace("Z", "+00:00"))
                    if act_time >= today_start and c.get("status") == "active":
                        # Rough estimate
                        requests_today += min(c.get("request_count", 0), 100)
                except Exception:
                    pass

        # Aggregate tool usage
        tool_counts = {}
        for c in connections:
            if not c:
                continue
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
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching MCP stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")


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
    supabase = get_supabase_client()
    connection_id = None

    try:
        # Check for existing connection with same session
        existing = await supabase.request(
            f"/mcp_connections?session_id=eq.{body.session_id}&status=eq.active&select=id"
        )

        if existing.get("data") and len(existing["data"]) > 0:
            # Update existing connection
            connection_id = existing["data"][0].get("id")
            if connection_id:
                await supabase.update(
                    "mcp_connections",
                    {"id": f"eq.{connection_id}"},
                    {
                        "last_activity_at": datetime.now(timezone.utc).isoformat(),
                        "ip_address": body.ip_address,
                        "user_agent": body.user_agent,
                    },
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

            if result.get("error"):
                error_msg = result.get("error", "")
                # Handle missing table gracefully
                if "mcp_connections" in str(error_msg) and ("does not exist" in str(error_msg) or "relation" in str(error_msg)):
                    logger.warning("mcp_connections table not found, skipping registration")
                    return {"status": "skipped", "connection_id": None, "reason": "table_not_found"}
                raise HTTPException(status_code=500, detail="Failed to register connection")

            if not result.get("data") or len(result["data"]) == 0:
                raise HTTPException(status_code=500, detail="Failed to register connection - no data returned")

            connection_id = result["data"][0].get("id")

            # Log connect activity (non-critical, don't fail if this errors)
            try:
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
            except Exception as e:
                logger.warning("Failed to log connect activity", connection_id=connection_id, error=str(e))

        logger.info(
            "MCP connection registered",
            session_id=body.session_id,
            user_id=user["user_id"],
            client_name=body.client_name,
        )

        return {"status": "registered", "connection_id": connection_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error registering MCP connection", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to register connection")


@router.post("/connections/activity", include_in_schema=False)
async def record_mcp_activity(
    request: Request,
    body: RecordActivityRequest,
):
    """Record tool usage activity (internal endpoint).

    Called by the MCP server after each tool invocation.
    """
    # INPUT VALIDATION: Validate connection_id format
    connection_id = validate_uuid(body.connection_id, "connection_id")

    user = await get_current_user(request)
    user_id = user["user_id"]
    supabase = get_supabase_client()

    try:
        # Verify connection belongs to user
        conn_result = await supabase.request(
            f"/mcp_connections?id=eq.{connection_id}&user_id=eq.{user_id}&select=id"
        )

        if conn_result.get("error"):
            error_msg = conn_result.get("error", "")
            # Handle missing table gracefully
            if "mcp_connections" in str(error_msg) and ("does not exist" in str(error_msg) or "relation" in str(error_msg)):
                logger.warning("mcp_connections table not found, skipping activity recording")
                return {"status": "skipped", "reason": "table_not_found"}
            raise HTTPException(status_code=500, detail="Failed to verify connection")

        if not conn_result.get("data"):
            raise HTTPException(status_code=404, detail="Connection not found")

        # Build activity metadata with new fields
        activity_metadata = body.metadata or {}
        if body.screenshot_key:
            activity_metadata["screenshot_key"] = body.screenshot_key
        if body.input_tokens is not None:
            activity_metadata["input_tokens"] = body.input_tokens
        if body.output_tokens is not None:
            activity_metadata["output_tokens"] = body.output_tokens

        # Update connection stats via RPC
        rpc_result = await supabase.rpc(
            "record_mcp_tool_usage",
            {
                "p_connection_id": connection_id,
                "p_tool_name": body.tool_name,
                "p_request_id": body.request_id,
                "p_duration_ms": body.duration_ms,
                "p_success": body.success,
                "p_error_message": body.error_message,
                "p_metadata": activity_metadata if activity_metadata else None,
            },
        )

        # Handle RPC errors gracefully (function might not exist)
        if rpc_result.get("error"):
            error_msg = rpc_result.get("error", "")
            if "record_mcp_tool_usage" in str(error_msg) and ("does not exist" in str(error_msg) or "function" in str(error_msg).lower()):
                logger.warning("record_mcp_tool_usage function not found, skipping activity recording")
                return {"status": "skipped", "reason": "function_not_found"}
            logger.warning("Error calling record_mcp_tool_usage RPC", error=error_msg)
            # Don't fail the request, just log the warning

        return {"status": "recorded"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error recording MCP activity", connection_id=connection_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to record activity")


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
    user_id = user["user_id"]
    supabase = get_supabase_client()

    try:
        result = await supabase.request(
            f"/device_auth_sessions?user_id=eq.{user_id}&status=eq.pending&select=*&order=created_at.desc"
        )

        # Handle missing table gracefully
        if result.get("error"):
            error_msg = result.get("error", "")
            if "device_auth_sessions" in str(error_msg) and ("does not exist" in str(error_msg) or "relation" in str(error_msg)):
                logger.warning("device_auth_sessions table not found, returning empty list")
                return {"sessions": [], "count": 0}
            logger.warning("Error fetching pending auth sessions", error=error_msg)

        sessions = result.get("data", []) or []
        return {
            "sessions": sessions,
            "count": len(sessions),
        }
    except Exception as e:
        logger.exception("Error fetching pending auth sessions", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch pending sessions")
