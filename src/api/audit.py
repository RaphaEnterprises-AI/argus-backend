"""Audit Log API endpoints.

Provides endpoints for:
- Viewing audit logs
- Filtering and searching logs
- Exporting logs for compliance
"""

from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from src.api.teams import get_current_user, verify_org_access
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/audit", tags=["Audit Logs"])


# ============================================================================
# Request/Response Models
# ============================================================================

class AuditLogEntry(BaseModel):
    """Single audit log entry."""
    id: str
    user_id: str
    user_email: str | None
    user_role: str | None
    action: str
    resource_type: str
    resource_id: str | None
    description: str
    metadata: dict
    ip_address: str | None
    user_agent: str | None
    status: str
    error_message: str | None
    created_at: str


class AuditLogResponse(BaseModel):
    """Paginated audit log response."""
    logs: list[AuditLogEntry]
    total: int
    page: int
    page_size: int
    has_more: bool


class AuditSummary(BaseModel):
    """Summary statistics for audit logs."""
    total_events: int
    events_today: int
    events_this_week: int
    events_this_month: int
    by_action: dict
    by_resource_type: dict
    by_user: dict
    by_status: dict
    recent_activity: list[AuditLogEntry]


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/organizations/{org_id}/logs", response_model=AuditLogResponse)
async def get_audit_logs(
    org_id: str,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    action: str | None = None,
    resource_type: str | None = None,
    user_id: str | None = None,
    status: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    search: str | None = None,
):
    """Get audit logs for an organization with filtering."""
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    try:
        # Build query
        query_parts = [f"organization_id=eq.{supabase_org_id}"]

        if action:
            query_parts.append(f"action=eq.{action}")
        if resource_type:
            query_parts.append(f"resource_type=eq.{resource_type}")
        if user_id:
            query_parts.append(f"user_id=eq.{user_id}")
        if status:
            query_parts.append(f"status=eq.{status}")
        if start_date:
            query_parts.append(f"created_at=gte.{start_date}")
        if end_date:
            query_parts.append(f"created_at=lte.{end_date}")
        if search:
            query_parts.append(f"description=ilike.*{search}*")

        query = "&".join(query_parts)

        # Get total count
        count_result = await supabase.request(
            f"/audit_logs?{query}&select=id",
            headers={"Prefer": "count=exact"}
        )

        # Handle missing table gracefully
        if count_result.get("error"):
            error_msg = str(count_result.get("error", ""))
            if "does not exist" in error_msg or "42703" in error_msg or "42P01" in error_msg:
                logger.warning("audit_logs table not found - returning empty list")
                return AuditLogResponse(logs=[], total=0, page=page, page_size=page_size, has_more=False)
            raise HTTPException(status_code=500, detail="Failed to fetch audit logs")

        total = len(count_result.get("data") or [])

        # Get paginated results
        offset = (page - 1) * page_size
        logs_result = await supabase.request(
            f"/audit_logs?{query}&select=*&order=created_at.desc&limit={page_size}&offset={offset}"
        )

        if logs_result.get("error"):
            error_msg = str(logs_result.get("error", ""))
            if "does not exist" in error_msg or "42703" in error_msg or "42P01" in error_msg:
                logger.warning("audit_logs table not found - returning empty list")
                return AuditLogResponse(logs=[], total=0, page=page, page_size=page_size, has_more=False)
            raise HTTPException(status_code=500, detail="Failed to fetch audit logs")

        logs_data = logs_result.get("data") or []
        logs = [
            AuditLogEntry(
                id=log["id"],
                user_id=log["user_id"],
                user_email=log.get("user_email"),
                user_role=log.get("user_role"),
                action=log["action"],
                resource_type=log["resource_type"],
                resource_id=log.get("resource_id"),
                description=log.get("description") or "",
                metadata=log.get("metadata") or {},
                ip_address=log.get("ip_address"),
                user_agent=log.get("user_agent"),
                status=log["status"],
                error_message=log.get("error_message"),
                created_at=log["created_at"],
            )
            for log in logs_data
        ]

        return AuditLogResponse(
            logs=logs,
            total=total,
            page=page,
            page_size=page_size,
            has_more=offset + len(logs) < total,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch audit logs", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch audit logs: {str(e)}")


@router.get("/organizations/{org_id}/summary", response_model=AuditSummary)
async def get_audit_summary(org_id: str, request: Request):
    """Get audit log summary statistics for the organization."""
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    try:
        now = datetime.now(UTC)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=now.weekday())
        month_start = today_start.replace(day=1)

        # Get all logs for this month (for aggregation)
        # Use 'Z' suffix instead of '+00:00' to avoid URL encoding issues with '+'
        month_start_str = month_start.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        logs_result = await supabase.request(
            f"/audit_logs?organization_id=eq.{supabase_org_id}&created_at=gte.{month_start_str}&select=*&order=created_at.desc"
        )

        # Handle missing table gracefully
        if logs_result.get("error"):
            error_msg = str(logs_result.get("error", ""))
            if "does not exist" in error_msg or "42703" in error_msg or "42P01" in error_msg:
                logger.warning("audit_logs table not found - returning empty summary")
                return AuditSummary(
                    total_events=0,
                    events_today=0,
                    events_this_week=0,
                    events_this_month=0,
                    by_action={},
                    by_resource_type={},
                    by_user={},
                    by_status={},
                    recent_activity=[],
                )
            raise HTTPException(status_code=500, detail="Failed to fetch audit logs")

        logs = logs_result.get("data") or []

        # Calculate statistics
        events_today = 0
        events_this_week = 0
        events_this_month = len(logs)

        by_action: dict[str, int] = {}
        by_resource_type: dict[str, int] = {}
        by_user: dict[str, int] = {}
        by_status: dict[str, int] = {}

        for log in logs:
            try:
                created_at = datetime.fromisoformat(log["created_at"].replace("Z", "+00:00"))

                if created_at >= today_start:
                    events_today += 1
                if created_at >= week_start:
                    events_this_week += 1
            except (ValueError, KeyError, TypeError):
                # Handle malformed date gracefully
                pass

            # Aggregate by action
            action = log.get("action") or "unknown"
            by_action[action] = by_action.get(action, 0) + 1

            # Aggregate by resource type
            resource = log.get("resource_type") or "unknown"
            by_resource_type[resource] = by_resource_type.get(resource, 0) + 1

            # Aggregate by user
            user_email = log.get("user_email") or log.get("user_id") or "unknown"
            by_user[user_email] = by_user.get(user_email, 0) + 1

            # Aggregate by status
            log_status = log.get("status") or "unknown"
            by_status[log_status] = by_status.get(log_status, 0) + 1

        # Get total count (all time) - use supabase_org_id, not org_id
        total_result = await supabase.request(
            f"/audit_logs?organization_id=eq.{supabase_org_id}&select=id"
        )

        # Handle potential error/None in total count query
        if total_result.get("error"):
            total = events_this_month  # Fall back to month count
        else:
            total = len(total_result.get("data") or [])

        # Recent activity (last 10)
        recent = [
            AuditLogEntry(
                id=log["id"],
                user_id=log["user_id"],
                user_email=log.get("user_email"),
                user_role=log.get("user_role"),
                action=log.get("action") or "unknown",
                resource_type=log.get("resource_type") or "unknown",
                resource_id=log.get("resource_id"),
                description=log.get("description") or "",
                metadata=log.get("metadata") or {},
                ip_address=log.get("ip_address"),
                user_agent=log.get("user_agent"),
                status=log.get("status") or "unknown",
                error_message=log.get("error_message"),
                created_at=log["created_at"],
            )
            for log in logs[:10]
        ]

        return AuditSummary(
            total_events=total,
            events_today=events_today,
            events_this_week=events_this_week,
            events_this_month=events_this_month,
            by_action=by_action,
            by_resource_type=by_resource_type,
            by_user=by_user,
            by_status=by_status,
            recent_activity=recent,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get audit summary", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get audit summary: {str(e)}")


@router.get("/organizations/{org_id}/export")
async def export_audit_logs(
    org_id: str,
    request: Request,
    format: str = Query("json", pattern="^(json|csv)$"),
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Export audit logs for compliance (JSON or CSV)."""
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], ["owner"], user.get("email"), request=request)

    supabase = get_supabase_client()

    try:
        # Build query
        query_parts = [f"organization_id=eq.{supabase_org_id}"]

        if start_date:
            query_parts.append(f"created_at=gte.{start_date}")
        if end_date:
            query_parts.append(f"created_at=lte.{end_date}")

        query = "&".join(query_parts)

        # Get all matching logs
        logs_result = await supabase.request(
            f"/audit_logs?{query}&select=*&order=created_at.desc"
        )

        # Handle missing table gracefully
        if logs_result.get("error"):
            error_msg = str(logs_result.get("error", ""))
            if "does not exist" in error_msg or "42703" in error_msg or "42P01" in error_msg:
                logger.warning("audit_logs table not found - returning empty export")
                return {
                    "organization_id": supabase_org_id,
                    "exported_at": datetime.now(UTC).isoformat(),
                    "log_count": 0,
                    "date_range": {"start": start_date, "end": end_date},
                    "logs": [],
                    "message": "Run migrations to enable audit logging",
                }
            raise HTTPException(status_code=500, detail="Failed to fetch audit logs")

        logs = logs_result.get("data") or []

        if format == "csv":
            # Generate CSV
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow([
                "id", "created_at", "user_id", "user_email", "user_role",
                "action", "resource_type", "resource_id", "description",
                "status", "ip_address"
            ])

            # Data rows
            for log in logs:
                writer.writerow([
                    log.get("id", ""),
                    log.get("created_at", ""),
                    log.get("user_id", ""),
                    log.get("user_email", ""),
                    log.get("user_role", ""),
                    log.get("action", ""),
                    log.get("resource_type", ""),
                    log.get("resource_id", ""),
                    log.get("description", ""),
                    log.get("status", ""),
                    log.get("ip_address", ""),
                ])

            from fastapi.responses import Response
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=audit_logs_{supabase_org_id}_{datetime.now().strftime('%Y%m%d')}.csv"
                }
            )

        # Default: JSON format
        return {
            "organization_id": supabase_org_id,
            "exported_at": datetime.now(UTC).isoformat(),
            "log_count": len(logs),
            "date_range": {
                "start": start_date,
                "end": end_date,
            },
            "logs": logs,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to export audit logs", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to export audit logs: {str(e)}")


@router.get("/organizations/{org_id}/actions")
async def get_available_actions(org_id: str, request: Request):
    """Get list of all available audit actions for filtering."""
    user = await get_current_user(request)
    _, _ = await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    # Return all possible actions categorized
    return {
        "categories": {
            "member": {
                "label": "Team Members",
                "actions": [
                    {"value": "member.invite", "label": "Invite Member"},
                    {"value": "member.accept", "label": "Accept Invitation"},
                    {"value": "member.remove", "label": "Remove Member"},
                    {"value": "member.role_change", "label": "Change Role"},
                ]
            },
            "api_key": {
                "label": "API Keys",
                "actions": [
                    {"value": "api_key.create", "label": "Create Key"},
                    {"value": "api_key.rotate", "label": "Rotate Key"},
                    {"value": "api_key.revoke", "label": "Revoke Key"},
                    {"value": "api_key.use", "label": "Use Key"},
                ]
            },
            "project": {
                "label": "Projects",
                "actions": [
                    {"value": "project.create", "label": "Create Project"},
                    {"value": "project.update", "label": "Update Project"},
                    {"value": "project.delete", "label": "Delete Project"},
                    {"value": "project.settings_change", "label": "Change Settings"},
                ]
            },
            "test": {
                "label": "Tests",
                "actions": [
                    {"value": "test.generate", "label": "Generate Test"},
                    {"value": "test.approve", "label": "Approve Test"},
                    {"value": "test.reject", "label": "Reject Test"},
                    {"value": "test.run", "label": "Run Test"},
                ]
            },
            "healing": {
                "label": "Self-Healing",
                "actions": [
                    {"value": "healing.apply", "label": "Apply Healing"},
                    {"value": "healing.learn", "label": "Learn Pattern"},
                    {"value": "healing.reject", "label": "Reject Healing"},
                ]
            },
            "auth": {
                "label": "Authentication",
                "actions": [
                    {"value": "auth.login", "label": "Login"},
                    {"value": "auth.logout", "label": "Logout"},
                    {"value": "auth.mfa_enable", "label": "Enable MFA"},
                    {"value": "auth.mfa_disable", "label": "Disable MFA"},
                ]
            },
            "organization": {
                "label": "Organization",
                "actions": [
                    {"value": "org.create", "label": "Create Organization"},
                    {"value": "org.update", "label": "Update Organization"},
                    {"value": "org.plan_change", "label": "Change Plan"},
                    {"value": "org.settings_change", "label": "Change Settings"},
                ]
            },
        },
        "resource_types": [
            {"value": "organization", "label": "Organization"},
            {"value": "member", "label": "Member"},
            {"value": "project", "label": "Project"},
            {"value": "api_key", "label": "API Key"},
            {"value": "test", "label": "Test"},
            {"value": "event", "label": "Event"},
            {"value": "webhook", "label": "Webhook"},
            {"value": "healing_pattern", "label": "Healing Pattern"},
            {"value": "settings", "label": "Settings"},
        ],
        "statuses": [
            {"value": "success", "label": "Success"},
            {"value": "failure", "label": "Failure"},
            {"value": "pending", "label": "Pending"},
        ],
    }
