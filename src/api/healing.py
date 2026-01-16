"""Self-Healing Configuration API endpoints.

Provides endpoints for:
- Configuring self-healing behavior
- Managing healing patterns
- Viewing healing statistics
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field
import structlog

from src.services.supabase_client import get_supabase_client
from src.api.teams import get_current_user, verify_org_access, log_audit

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/healing", tags=["Self-Healing"])


# ============================================================================
# Request/Response Models
# ============================================================================


class HealingConfigRequest(BaseModel):
    """Request to update healing configuration."""

    enabled: Optional[bool] = None
    auto_apply: Optional[bool] = None
    min_confidence_auto: Optional[float] = Field(None, ge=0.5, le=1.0)
    min_confidence_suggest: Optional[float] = Field(None, ge=0.3, le=1.0)
    heal_selectors: Optional[bool] = None
    max_selector_variations: Optional[int] = Field(None, ge=1, le=20)
    preferred_selector_strategies: Optional[list[str]] = None
    heal_timeouts: Optional[bool] = None
    max_wait_time_ms: Optional[int] = Field(None, ge=1000, le=120000)
    heal_text_content: Optional[bool] = None
    text_similarity_threshold: Optional[float] = Field(None, ge=0.5, le=1.0)
    learn_from_success: Optional[bool] = None
    learn_from_manual_fixes: Optional[bool] = None
    share_patterns_across_projects: Optional[bool] = None
    notify_on_heal: Optional[bool] = None
    notify_on_suggestion: Optional[bool] = None
    notification_channels: Optional[dict] = None
    require_approval: Optional[bool] = None
    auto_approve_after_hours: Optional[int] = Field(None, ge=1, le=168)
    approvers: Optional[list[str]] = None
    max_heals_per_hour: Optional[int] = Field(None, ge=1, le=1000)
    max_heals_per_test: Optional[int] = Field(None, ge=1, le=50)


class HealingConfigResponse(BaseModel):
    """Healing configuration response."""

    id: str
    organization_id: str
    project_id: Optional[str]
    enabled: bool
    auto_apply: bool
    min_confidence_auto: float
    min_confidence_suggest: float
    heal_selectors: bool
    max_selector_variations: int
    preferred_selector_strategies: list[str]
    heal_timeouts: bool
    max_wait_time_ms: int
    heal_text_content: bool
    text_similarity_threshold: float
    learn_from_success: bool
    learn_from_manual_fixes: bool
    share_patterns_across_projects: bool
    notify_on_heal: bool
    notify_on_suggestion: bool
    notification_channels: dict
    require_approval: bool
    auto_approve_after_hours: Optional[int]
    approvers: Optional[list[str]]
    max_heals_per_hour: int
    max_heals_per_test: int
    created_at: str
    updated_at: str


class HealingPatternResponse(BaseModel):
    """Healing pattern response."""

    id: str
    fingerprint: str
    original_selector: str
    healed_selector: str
    error_type: str
    success_count: int
    failure_count: int
    confidence: float
    project_id: Optional[str]
    created_at: str


class HealingStatsResponse(BaseModel):
    """Healing statistics response."""

    total_patterns: int
    total_heals_applied: int
    total_heals_suggested: int
    success_rate: float
    top_error_types: dict
    patterns_by_project: dict
    heals_last_24h: int
    heals_last_7d: int
    heals_last_30d: int
    avg_confidence: float
    recent_heals: list[dict]


# ============================================================================
# Configuration Endpoints
# ============================================================================


@router.get("/organizations/{org_id}/config", response_model=HealingConfigResponse)
async def get_healing_config(org_id: str, request: Request, project_id: Optional[str] = None):
    """Get healing configuration for organization (optionally project-specific)."""
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

    supabase = get_supabase_client()

    # Try to get project-specific config first
    if project_id:
        result = await supabase.request(
            f"/self_healing_config?organization_id=eq.{org_id}&project_id=eq.{project_id}&select=*"
        )
        if result.get("data"):
            return _config_to_response(result["data"][0])

    # Fall back to org-level config
    result = await supabase.request(
        f"/self_healing_config?organization_id=eq.{org_id}&project_id=is.null&select=*"
    )

    if not result.get("data"):
        # Create default config
        default = await supabase.insert(
            "self_healing_config",
            {
                "organization_id": org_id,
            },
        )
        if default.get("data"):
            return _config_to_response(default["data"][0])
        raise HTTPException(status_code=500, detail="Failed to create default config")

    return _config_to_response(result["data"][0])


@router.put("/organizations/{org_id}/config", response_model=HealingConfigResponse)
async def update_healing_config(
    org_id: str, body: HealingConfigRequest, request: Request, project_id: Optional[str] = None
):
    """Update healing configuration."""
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Build update data
    update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}

    for field, value in body.model_dump(exclude_unset=True).items():
        if value is not None:
            update_data[field] = value

    # Find existing config
    if project_id:
        existing = await supabase.request(
            f"/self_healing_config?organization_id=eq.{org_id}&project_id=eq.{project_id}&select=id"
        )
    else:
        existing = await supabase.request(
            f"/self_healing_config?organization_id=eq.{org_id}&project_id=is.null&select=id"
        )

    if existing.get("data"):
        # Update existing
        config_id = existing["data"][0]["id"]
        await supabase.update("self_healing_config", {"id": f"eq.{config_id}"}, update_data)
    else:
        # Create new
        update_data["organization_id"] = org_id
        if project_id:
            update_data["project_id"] = project_id
        result = await supabase.insert("self_healing_config", update_data)
        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to create config")
        config_id = result["data"][0]["id"]

    # Audit log
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user["email"],
        action="org.settings_change",
        resource_type="settings",
        resource_id=config_id,
        description="Updated self-healing configuration",
        metadata={"changes": update_data, "project_id": project_id},
        request=request,
    )

    return await get_healing_config(org_id, request, project_id)


def _config_to_response(config: dict) -> HealingConfigResponse:
    """Convert database config to response model."""
    return HealingConfigResponse(
        id=config["id"],
        organization_id=config["organization_id"],
        project_id=config.get("project_id"),
        enabled=config.get("enabled", True),
        auto_apply=config.get("auto_apply", False),
        min_confidence_auto=float(config.get("min_confidence_auto", 0.95)),
        min_confidence_suggest=float(config.get("min_confidence_suggest", 0.70)),
        heal_selectors=config.get("heal_selectors", True),
        max_selector_variations=config.get("max_selector_variations", 9),
        preferred_selector_strategies=config.get(
            "preferred_selector_strategies", ["id", "data-testid", "role", "text", "css"]
        ),
        heal_timeouts=config.get("heal_timeouts", True),
        max_wait_time_ms=config.get("max_wait_time_ms", 30000),
        heal_text_content=config.get("heal_text_content", True),
        text_similarity_threshold=float(config.get("text_similarity_threshold", 0.85)),
        learn_from_success=config.get("learn_from_success", True),
        learn_from_manual_fixes=config.get("learn_from_manual_fixes", True),
        share_patterns_across_projects=config.get("share_patterns_across_projects", False),
        notify_on_heal=config.get("notify_on_heal", True),
        notify_on_suggestion=config.get("notify_on_suggestion", True),
        notification_channels=config.get("notification_channels", {"email": True, "slack": False}),
        require_approval=config.get("require_approval", True),
        auto_approve_after_hours=config.get("auto_approve_after_hours"),
        approvers=config.get("approvers"),
        max_heals_per_hour=config.get("max_heals_per_hour", 50),
        max_heals_per_test=config.get("max_heals_per_test", 5),
        created_at=config["created_at"],
        updated_at=config.get("updated_at", config["created_at"]),
    )


# ============================================================================
# Patterns Endpoints
# ============================================================================


@router.get("/organizations/{org_id}/patterns", response_model=list[HealingPatternResponse])
async def list_healing_patterns(
    org_id: str,
    request: Request,
    project_id: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = Query(50, ge=1, le=500),
    offset: int = 0,
):
    """List learned healing patterns."""
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

    supabase = get_supabase_client()

    # Get projects for this org
    projects = await supabase.request(f"/projects?organization_id=eq.{org_id}&select=id")
    project_ids = [p["id"] for p in projects.get("data", [])]

    if not project_ids:
        return []

    # Build query
    query = f"/healing_patterns?project_id=in.({','.join(project_ids)})"

    if project_id:
        query = f"/healing_patterns?project_id=eq.{project_id}"

    query += f"&confidence=gte.{min_confidence}&select=*&order=confidence.desc,success_count.desc&limit={limit}&offset={offset}"

    result = await supabase.request(query)

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch patterns")

    return [
        HealingPatternResponse(
            id=p["id"],
            fingerprint=p["fingerprint"],
            original_selector=p["original_selector"],
            healed_selector=p["healed_selector"],
            error_type=p["error_type"],
            success_count=p.get("success_count", 0),
            failure_count=p.get("failure_count", 0),
            confidence=float(p.get("confidence", 0)),
            project_id=p.get("project_id"),
            created_at=p["created_at"],
        )
        for p in result.get("data", [])
    ]


@router.delete("/organizations/{org_id}/patterns/{pattern_id}")
async def delete_healing_pattern(org_id: str, pattern_id: str, request: Request):
    """Delete a healing pattern."""
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Verify pattern belongs to org's projects
    pattern = await supabase.request(
        f"/healing_patterns?id=eq.{pattern_id}&select=*,projects!inner(organization_id)"
    )

    if not pattern.get("data"):
        raise HTTPException(status_code=404, detail="Pattern not found")

    # Delete pattern
    await supabase.request(f"/healing_patterns?id=eq.{pattern_id}", method="DELETE")

    # Audit log
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user["email"],
        action="healing.reject",
        resource_type="healing_pattern",
        resource_id=pattern_id,
        description="Deleted healing pattern",
        request=request,
    )

    return {"success": True, "message": "Pattern deleted"}


# ============================================================================
# Statistics Endpoints
# ============================================================================


@router.get("/organizations/{org_id}/stats", response_model=HealingStatsResponse)
async def get_healing_stats(
    org_id: str,
    request: Request,
    project_id: Optional[str] = None,
):
    """Get healing statistics for the organization."""
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

    supabase = get_supabase_client()

    now = datetime.now(timezone.utc)
    day_ago = (now - timedelta(days=1)).isoformat()
    week_ago = (now - timedelta(days=7)).isoformat()
    month_ago = (now - timedelta(days=30)).isoformat()

    # Get projects for this org
    projects = await supabase.request(f"/projects?organization_id=eq.{org_id}&select=id,name")
    project_data = {p["id"]: p["name"] for p in projects.get("data", [])}
    project_ids = list(project_data.keys())

    if not project_ids:
        return HealingStatsResponse(
            total_patterns=0,
            total_heals_applied=0,
            total_heals_suggested=0,
            success_rate=0,
            top_error_types={},
            patterns_by_project={},
            heals_last_24h=0,
            heals_last_7d=0,
            heals_last_30d=0,
            avg_confidence=0,
            recent_heals=[],
        )

    # Build project filter
    if project_id:
        project_filter = f"project_id=eq.{project_id}"
    else:
        project_filter = f"project_id=in.({','.join(project_ids)})"

    # Get all patterns
    patterns = await supabase.request(f"/healing_patterns?{project_filter}&select=*")
    patterns_data = patterns.get("data", [])

    # Calculate stats
    total_patterns = len(patterns_data)
    total_success = sum(p.get("success_count", 0) for p in patterns_data)
    total_failure = sum(p.get("failure_count", 0) for p in patterns_data)
    total_heals = total_success + total_failure
    success_rate = (total_success / total_heals * 100) if total_heals > 0 else 0

    # Error type breakdown
    error_types = {}
    for p in patterns_data:
        error_type = p.get("error_type", "unknown")
        error_types[error_type] = error_types.get(error_type, 0) + 1

    # Patterns by project
    patterns_by_project = {}
    for p in patterns_data:
        pid = p.get("project_id", "unknown")
        pname = project_data.get(pid, "Unknown Project")
        patterns_by_project[pname] = patterns_by_project.get(pname, 0) + 1

    # Average confidence
    confidences = [float(p.get("confidence", 0)) for p in patterns_data if p.get("confidence")]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Recent heals (approximated from patterns with recent updates)
    recent_heals = [
        {
            "id": p["id"],
            "original": p["original_selector"][:50],
            "healed": p["healed_selector"][:50],
            "error_type": p["error_type"],
            "confidence": float(p.get("confidence", 0)),
            "created_at": p["created_at"],
        }
        for p in sorted(patterns_data, key=lambda x: x.get("created_at", ""), reverse=True)[:10]
    ]

    return HealingStatsResponse(
        total_patterns=total_patterns,
        total_heals_applied=total_success,
        total_heals_suggested=total_heals,
        success_rate=round(success_rate, 2),
        top_error_types=dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]),
        patterns_by_project=patterns_by_project,
        heals_last_24h=sum(1 for p in patterns_data if p.get("created_at", "") > day_ago),
        heals_last_7d=sum(1 for p in patterns_data if p.get("created_at", "") > week_ago),
        heals_last_30d=sum(1 for p in patterns_data if p.get("created_at", "") > month_ago),
        avg_confidence=round(avg_confidence, 3),
        recent_heals=recent_heals,
    )


# ============================================================================
# Approval Endpoints
# ============================================================================


@router.get("/organizations/{org_id}/pending-approvals")
async def get_pending_approvals(
    org_id: str,
    request: Request,
    project_id: Optional[str] = None,
):
    """Get healing suggestions pending approval."""
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

    # For now, return empty list - would be populated by actual healing suggestions
    # This would query a healing_suggestions table in production
    return {
        "pending": [],
        "total": 0,
        "message": "No pending healing suggestions",
    }


@router.post("/organizations/{org_id}/approve/{pattern_id}")
async def approve_healing(org_id: str, pattern_id: str, request: Request):
    """Approve a healing suggestion."""
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Use atomic RPC function to increment success count (prevents race conditions)
    result = await supabase.request(
        "/rpc/increment_healing_success", method="POST", json={"pattern_id": pattern_id}
    )

    if result.get("error"):
        raise HTTPException(
            status_code=500, detail=f"Failed to update healing pattern: {result.get('error')}"
        )

    # Audit log
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user["email"],
        action="healing.apply",
        resource_type="healing_pattern",
        resource_id=pattern_id,
        description="Approved healing pattern",
        request=request,
    )

    return {"success": True, "message": "Healing approved"}


@router.post("/organizations/{org_id}/reject/{pattern_id}")
async def reject_healing(org_id: str, pattern_id: str, request: Request):
    """Reject a healing suggestion."""
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Use atomic RPC function to increment failure count (prevents race conditions)
    result = await supabase.request(
        "/rpc/increment_healing_failure", method="POST", json={"pattern_id": pattern_id}
    )

    if result.get("error"):
        raise HTTPException(
            status_code=500, detail=f"Failed to update healing pattern: {result.get('error')}"
        )

    # Audit log
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user["email"],
        action="healing.reject",
        resource_type="healing_pattern",
        resource_id=pattern_id,
        description="Rejected healing pattern",
        request=request,
    )

    return {"success": True, "message": "Healing rejected"}
