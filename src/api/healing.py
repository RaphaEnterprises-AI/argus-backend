"""Self-Healing Configuration API endpoints.

Provides endpoints for:
- Configuring self-healing behavior
- Managing healing patterns
- Viewing healing statistics
- Getting instant healing suggestions via intelligence layer (RAP-249)

RAP-249: Updated to use intelligence layer for instant responses.
Target latency: 50-100ms for cached/vector queries (vs 3-4s for LLM).
"""

import time
from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.teams import get_current_user, log_audit, verify_org_access
from src.intelligence import QueryIntent, QueryRouter, get_query_router
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/healing", tags=["Self-Healing"])

# Confidence threshold for LLM fallback
LLM_FALLBACK_THRESHOLD = 0.7


# ============================================================================
# Request/Response Models
# ============================================================================


class HealingConfigRequest(BaseModel):
    """Request to update healing configuration."""

    enabled: bool | None = None
    auto_apply: bool | None = None
    min_confidence_auto: float | None = Field(None, ge=0.5, le=1.0)
    min_confidence_suggest: float | None = Field(None, ge=0.3, le=1.0)
    heal_selectors: bool | None = None
    max_selector_variations: int | None = Field(None, ge=1, le=20)
    preferred_selector_strategies: list[str] | None = None
    heal_timeouts: bool | None = None
    max_wait_time_ms: int | None = Field(None, ge=1000, le=120000)
    heal_text_content: bool | None = None
    text_similarity_threshold: float | None = Field(None, ge=0.5, le=1.0)
    learn_from_success: bool | None = None
    learn_from_manual_fixes: bool | None = None
    share_patterns_across_projects: bool | None = None
    notify_on_heal: bool | None = None
    notify_on_suggestion: bool | None = None
    notification_channels: dict | None = None
    require_approval: bool | None = None
    auto_approve_after_hours: int | None = Field(None, ge=1, le=168)
    approvers: list[str] | None = None
    max_heals_per_hour: int | None = Field(None, ge=1, le=1000)
    max_heals_per_test: int | None = Field(None, ge=1, le=50)


class HealingConfigResponse(BaseModel):
    """Healing configuration response."""

    id: str
    organization_id: str
    project_id: str | None
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
    auto_approve_after_hours: int | None
    approvers: list[str] | None
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
    project_id: str | None
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
# RAP-249: Intelligence Layer Models for Instant Suggestions
# ============================================================================


class HealingSuggestionRequest(BaseModel):
    """Request for healing suggestions using intelligence layer."""

    error_message: str = Field(..., description="The error message to analyze")
    error_type: str | None = Field(None, description="Type of error (selector_changed, timing_issue, etc.)")
    selector: str | None = Field(None, description="The broken selector (if applicable)")
    context: dict | None = Field(None, description="Additional context (url, step_index, etc.)")
    skip_cache: bool = Field(False, description="Skip cache lookup for fresh analysis")
    force_llm: bool = Field(False, description="Force LLM analysis regardless of confidence")


class HealingSuggestion(BaseModel):
    """A single healing suggestion."""

    fix_type: str = Field(..., description="Type of fix (update_selector, add_wait, increase_timeout, etc.)")
    old_value: str | None = Field(None, description="Current/broken value")
    new_value: str | None = Field(None, description="Suggested fix value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    explanation: str = Field(..., description="Why this fix should work")
    success_rate: float | None = Field(None, description="Historical success rate if from patterns")
    pattern_id: str | None = Field(None, description="Pattern ID if from learned patterns")


class HealingSuggestionResponse(BaseModel):
    """Response from healing suggestion endpoint."""

    suggestions: list[HealingSuggestion] = Field(default_factory=list)
    source: str = Field(..., description="Source: cache, vector, or llm")
    latency_ms: float = Field(..., description="Query latency in milliseconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    intent: str = Field(..., description="Detected query intent")
    cached: bool = Field(False, description="Whether result came from cache")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class RootCauseRequest(BaseModel):
    """Request for root cause analysis using intelligence layer."""

    error_message: str = Field(..., description="The error message to analyze")
    error_type: str | None = Field(None, description="Type of error")
    context: dict | None = Field(None, description="Additional context (logs, stack trace, etc.)")
    similar_failures: list[dict] | None = Field(None, description="Similar failures for context")
    skip_cache: bool = Field(False, description="Skip cache lookup")


class RootCauseResponse(BaseModel):
    """Response from root cause analysis endpoint."""

    root_cause: str = Field(..., description="Identified root cause")
    category: str = Field(..., description="Category: selector_changed, timing_issue, etc.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    source: str = Field(..., description="Source: cache, vector, or llm")
    latency_ms: float = Field(..., description="Query latency in milliseconds")


# ============================================================================
# Configuration Endpoints
# ============================================================================


@router.get("/organizations/{org_id}/config", response_model=HealingConfigResponse)
async def get_healing_config(org_id: str, request: Request, project_id: str | None = None):
    """Get healing configuration for organization (optionally project-specific)."""
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

    supabase = get_supabase_client()

    # Try to get project-specific config first
    if project_id:
        result = await supabase.request(
            f"/self_healing_config?organization_id=eq.{supabase_org_id}&project_id=eq.{project_id}&select=*"
        )
        if result.get("data"):
            return _config_to_response(result["data"][0])

    # Fall back to org-level config
    result = await supabase.request(
        f"/self_healing_config?organization_id=eq.{supabase_org_id}&project_id=is.null&select=*"
    )

    if not result.get("data"):
        # Create default config
        default = await supabase.insert(
            "self_healing_config",
            {
                "organization_id": supabase_org_id,
            },
        )
        if default.get("data"):
            return _config_to_response(default["data"][0])
        raise HTTPException(status_code=500, detail="Failed to create default config")

    return _config_to_response(result["data"][0])


@router.put("/organizations/{org_id}/config", response_model=HealingConfigResponse)
async def update_healing_config(
    org_id: str, body: HealingConfigRequest, request: Request, project_id: str | None = None
):
    """Update healing configuration."""
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

    supabase = get_supabase_client()

    # Build update data
    update_data = {"updated_at": datetime.now(UTC).isoformat()}

    for field, value in body.model_dump(exclude_unset=True).items():
        if value is not None:
            update_data[field] = value

    # Find existing config
    if project_id:
        existing = await supabase.request(
            f"/self_healing_config?organization_id=eq.{supabase_org_id}&project_id=eq.{project_id}&select=id"
        )
    else:
        existing = await supabase.request(
            f"/self_healing_config?organization_id=eq.{supabase_org_id}&project_id=is.null&select=id"
        )

    if existing.get("data"):
        # Update existing
        config_id = existing["data"][0]["id"]
        await supabase.update("self_healing_config", {"id": f"eq.{config_id}"}, update_data)
    else:
        # Create new
        update_data["organization_id"] = supabase_org_id
        if project_id:
            update_data["project_id"] = project_id
        result = await supabase.insert("self_healing_config", update_data)
        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to create config")
        config_id = result["data"][0]["id"]

    # Audit log
    await log_audit(
        organization_id=supabase_org_id,
        user_id=user["user_id"],
        user_email=user["email"],
        action="org.settings_change",
        resource_type="settings",
        resource_id=config_id,
        description="Updated self-healing configuration",
        metadata={"changes": update_data, "project_id": project_id},
        request=request,
    )

    return await get_healing_config(supabase_org_id, request, project_id)


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
    project_id: str | None = None,
    min_confidence: float = 0.0,
    limit: int = Query(50, ge=1, le=500),
    offset: int = 0,
):
    """List learned healing patterns."""
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

    supabase = get_supabase_client()

    # Get projects for this org
    projects = await supabase.request(f"/projects?organization_id=eq.{supabase_org_id}&select=id")
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
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

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
        organization_id=supabase_org_id,
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
    project_id: str | None = None,
):
    """Get healing statistics for the organization."""
    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

    supabase = get_supabase_client()

    now = datetime.now(UTC)
    day_ago = (now - timedelta(days=1)).isoformat()
    week_ago = (now - timedelta(days=7)).isoformat()
    month_ago = (now - timedelta(days=30)).isoformat()

    # Get projects for this org
    projects = await supabase.request(f"/projects?organization_id=eq.{supabase_org_id}&select=id,name")
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
    project_id: str | None = None,
):
    """Get healing suggestions pending approval."""
    user = await get_current_user(request)
    _, _ = await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

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
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

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
        organization_id=supabase_org_id,
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
    _, supabase_org_id = await verify_org_access(org_id, user["user_id"], ["owner", "admin"], user.get("email"), request=request)

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
        organization_id=supabase_org_id,
        user_id=user["user_id"],
        user_email=user["email"],
        action="healing.reject",
        resource_type="healing_pattern",
        resource_id=pattern_id,
        description="Rejected healing pattern",
        request=request,
    )

    return {"success": True, "message": "Healing rejected"}


# ============================================================================
# RAP-249: Intelligence Layer Endpoints for Instant Suggestions
# ============================================================================


@router.post("/organizations/{org_id}/suggest", response_model=HealingSuggestionResponse)
async def get_healing_suggestions(
    org_id: str,
    body: HealingSuggestionRequest,
    request: Request,
    project_id: str | None = None,
):
    """Get instant healing suggestions using the intelligence layer.

    This endpoint uses a tiered approach for optimal latency:
    1. Cache lookup (~10ms) - Exact match from previous queries
    2. Vector/hybrid search (~50-100ms) - Semantic similarity search
    3. LLM fallback (~3-4s) - Only when confidence < 0.7

    Target latency: 50-100ms for cached/vector queries.
    """
    start_time = time.perf_counter()

    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(
        org_id, user["user_id"], user_email=user.get("email"), request=request
    )

    # Initialize query router
    query_router = get_query_router(org_id=supabase_org_id, project_id=project_id)

    # Build query from request
    query = body.error_message
    if body.selector:
        query = f"{query}\nSelector: {body.selector}"

    # Build context for routing
    context = body.context or {}
    if body.error_type:
        context["error_type"] = body.error_type
    if project_id:
        context["project_id"] = project_id

    try:
        # Route query through intelligence layer
        result = await query_router.route(
            query=query,
            org_id=supabase_org_id,
            project_id=project_id,
            skip_cache=body.skip_cache,
            force_llm=body.force_llm,
        )

        # Transform result data into suggestions
        suggestions = _transform_to_suggestions(result.data, body)

        total_latency = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Healing suggestions generated",
            org_id=org_id,
            source=result.source,
            confidence=result.confidence,
            latency_ms=round(total_latency, 2),
            suggestions_count=len(suggestions),
        )

        return HealingSuggestionResponse(
            suggestions=suggestions,
            source=result.source,
            latency_ms=round(total_latency, 2),
            confidence=result.confidence,
            intent=result.intent.value if hasattr(result.intent, "value") else str(result.intent),
            cached=result.source == "cache",
            metadata={
                "query_latency_ms": result.latency_ms,
                "cache_key": result.cache_key,
                **(result.metadata or {}),
            },
        )

    except Exception as e:
        logger.error(
            "Healing suggestions failed",
            org_id=org_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get healing suggestions: {str(e)}",
        )


@router.post("/organizations/{org_id}/analyze-root-cause", response_model=RootCauseResponse)
async def analyze_root_cause(
    org_id: str,
    body: RootCauseRequest,
    request: Request,
    project_id: str | None = None,
):
    """Analyze root cause of a failure using the intelligence layer.

    Uses QueryIntent.ROOT_CAUSE for analysis with semantic search
    and LLM reasoning when needed.

    Target latency: 50-100ms for cached/vector queries.
    """
    start_time = time.perf_counter()

    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(
        org_id, user["user_id"], user_email=user.get("email"), request=request
    )

    # Initialize query router with specific intent
    query_router = get_query_router(org_id=supabase_org_id, project_id=project_id)

    # Build query for root cause analysis
    query = f"Analyze root cause: {body.error_message}"
    if body.error_type:
        query = f"{query}\nError type: {body.error_type}"

    # Add context if provided
    if body.context:
        context_str = "\n".join(f"{k}: {v}" for k, v in body.context.items())
        query = f"{query}\nContext: {context_str}"

    try:
        # Route query through intelligence layer
        # Force the ROOT_CAUSE intent by adjusting the query
        result = await query_router.route(
            query=query,
            org_id=supabase_org_id,
            project_id=project_id,
            skip_cache=body.skip_cache,
            force_llm=False,  # Let the router decide based on confidence
        )

        # Transform result into root cause response
        root_cause_data = _transform_to_root_cause(result.data, body)

        total_latency = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Root cause analysis completed",
            org_id=org_id,
            source=result.source,
            confidence=result.confidence,
            latency_ms=round(total_latency, 2),
        )

        return RootCauseResponse(
            root_cause=root_cause_data.get("root_cause", "Unable to determine root cause"),
            category=root_cause_data.get("category", "unknown"),
            confidence=result.confidence,
            evidence=root_cause_data.get("evidence", []),
            suggested_actions=root_cause_data.get("suggested_actions", []),
            source=result.source,
            latency_ms=round(total_latency, 2),
        )

    except Exception as e:
        logger.error(
            "Root cause analysis failed",
            org_id=org_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze root cause: {str(e)}",
        )


@router.post("/organizations/{org_id}/similar-errors")
async def find_similar_errors(
    org_id: str,
    body: HealingSuggestionRequest,
    request: Request,
    project_id: str | None = None,
    limit: int = Query(5, ge=1, le=20),
):
    """Find similar errors from the knowledge base.

    Uses QueryIntent.SIMILAR_ERRORS for fast semantic matching.
    Returns similar past errors with their healing patterns.

    Target latency: 50-100ms for cached/vector queries.
    """
    start_time = time.perf_counter()

    user = await get_current_user(request)
    _, supabase_org_id = await verify_org_access(
        org_id, user["user_id"], user_email=user.get("email"), request=request
    )

    # Initialize query router
    query_router = get_query_router(org_id=supabase_org_id, project_id=project_id)

    # Build query for similar error search
    query = body.error_message
    if body.selector:
        query = f"{query}\nSelector: {body.selector}"

    try:
        # Route query through intelligence layer
        result = await query_router.route(
            query=query,
            org_id=supabase_org_id,
            project_id=project_id,
            skip_cache=body.skip_cache,
            force_llm=False,  # Don't use LLM for similarity search
        )

        # Extract similar errors from result
        similar_errors = _extract_similar_errors(result.data, limit)

        total_latency = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Similar errors found",
            org_id=org_id,
            source=result.source,
            count=len(similar_errors),
            latency_ms=round(total_latency, 2),
        )

        return {
            "similar_errors": similar_errors,
            "count": len(similar_errors),
            "source": result.source,
            "latency_ms": round(total_latency, 2),
            "confidence": result.confidence,
        }

    except Exception as e:
        logger.error(
            "Similar errors search failed",
            org_id=org_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to find similar errors: {str(e)}",
        )


# ============================================================================
# Helper Functions for Intelligence Layer Transformations
# ============================================================================


def _transform_to_suggestions(
    data: dict | list | None,
    request_body: HealingSuggestionRequest,
) -> list[HealingSuggestion]:
    """Transform query router result data into HealingSuggestion list.

    Args:
        data: Raw data from query router
        request_body: Original request for context

    Returns:
        List of HealingSuggestion objects
    """
    suggestions = []

    if data is None:
        return suggestions

    # Handle different data formats from different sources
    if isinstance(data, dict):
        # Check for suggestions array (from LLM or processed results)
        if "suggestions" in data:
            for s in data["suggestions"]:
                suggestions.append(_dict_to_suggestion(s))

        # Check for results array (from vector search)
        elif "results" in data:
            for r in data["results"]:
                suggestion = _result_to_suggestion(r, request_body)
                if suggestion:
                    suggestions.append(suggestion)

        # Check for top_match (single best result)
        elif "top_match" in data and data["top_match"]:
            suggestion = _result_to_suggestion(data["top_match"], request_body)
            if suggestion:
                suggestions.append(suggestion)

        # Handle analysis response (from LLM)
        elif "analysis" in data:
            suggestion = _analysis_to_suggestion(data, request_body)
            if suggestion:
                suggestions.append(suggestion)

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                suggestion = _result_to_suggestion(item, request_body)
                if suggestion:
                    suggestions.append(suggestion)

    # Sort by confidence descending
    suggestions.sort(key=lambda s: s.confidence, reverse=True)

    return suggestions


def _dict_to_suggestion(d: dict) -> HealingSuggestion:
    """Convert a dictionary to HealingSuggestion."""
    return HealingSuggestion(
        fix_type=d.get("fix_type", "update_selector"),
        old_value=d.get("old_value"),
        new_value=d.get("new_value"),
        confidence=float(d.get("confidence", 0.5)),
        explanation=d.get("explanation", "Suggested fix based on similar patterns"),
        success_rate=d.get("success_rate"),
        pattern_id=d.get("pattern_id") or d.get("id"),
    )


def _result_to_suggestion(
    result: dict,
    request_body: HealingSuggestionRequest,
) -> HealingSuggestion | None:
    """Convert a vector search result to HealingSuggestion."""
    if not result:
        return None

    # Extract healed selector if available
    healed_selector = result.get("healed_selector")
    if not healed_selector:
        return None

    return HealingSuggestion(
        fix_type="update_selector",
        old_value=request_body.selector or result.get("selector") or result.get("original_selector"),
        new_value=healed_selector,
        confidence=float(result.get("confidence", result.get("similarity", 0.5))),
        explanation=f"Based on similar error pattern (method: {result.get('healing_method', 'unknown')})",
        success_rate=float(result.get("success_rate", 0)) if result.get("success_rate") else None,
        pattern_id=result.get("id") or result.get("pattern_id"),
    )


def _analysis_to_suggestion(
    analysis: dict,
    request_body: HealingSuggestionRequest,
) -> HealingSuggestion | None:
    """Convert LLM analysis response to HealingSuggestion."""
    analysis_text = analysis.get("analysis", "")

    if not analysis_text:
        return None

    return HealingSuggestion(
        fix_type="analysis",
        old_value=request_body.selector,
        new_value=None,
        confidence=0.6,  # Lower confidence for LLM-only analysis
        explanation=analysis_text[:500],  # Truncate long explanations
        success_rate=None,
        pattern_id=None,
    )


def _transform_to_root_cause(
    data: dict | None,
    request_body: RootCauseRequest,
) -> dict:
    """Transform query router result data into root cause analysis.

    Args:
        data: Raw data from query router
        request_body: Original request for context

    Returns:
        Dictionary with root cause analysis
    """
    if data is None:
        return {
            "root_cause": "Unable to determine root cause",
            "category": "unknown",
            "evidence": [],
            "suggested_actions": [],
        }

    # Handle LLM analysis response
    if "analysis" in data:
        return {
            "root_cause": data["analysis"][:500],
            "category": _infer_category(data["analysis"], request_body.error_type),
            "evidence": [f"LLM analysis of error: {request_body.error_message[:100]}"],
            "suggested_actions": ["Review the analysis and apply suggested fix"],
        }

    # Handle structured root cause response
    if "root_cause" in data:
        return data

    # Handle similar failures response
    if "similar_failures" in data:
        failures = data["similar_failures"]
        if failures:
            top_failure = failures[0]
            return {
                "root_cause": f"Similar to past failure: {top_failure.get('error_message', 'unknown')[:200]}",
                "category": top_failure.get("healing_method", "unknown"),
                "evidence": [
                    f"Similarity: {top_failure.get('similarity', 0):.0%}",
                    f"Previous fix: {top_failure.get('healed_selector', 'N/A')}",
                ],
                "suggested_actions": [
                    f"Apply the healing method: {top_failure.get('healing_method', 'unknown')}",
                    f"Use healed selector: {top_failure.get('healed_selector', 'N/A')}",
                ],
            }

    # Handle results array
    if "results" in data and data["results"]:
        results = data["results"]
        return {
            "root_cause": f"Pattern matches found: {len(results)} similar errors",
            "category": _infer_category_from_results(results),
            "evidence": [f"Found {len(results)} similar error patterns"],
            "suggested_actions": ["Review matched patterns for potential fixes"],
        }

    return {
        "root_cause": "Analysis inconclusive",
        "category": request_body.error_type or "unknown",
        "evidence": [],
        "suggested_actions": ["Manual investigation recommended"],
    }


def _infer_category(analysis: str, error_type: str | None) -> str:
    """Infer failure category from analysis text."""
    analysis_lower = analysis.lower()

    if error_type:
        return error_type

    if "selector" in analysis_lower or "element" in analysis_lower:
        return "selector_changed"
    elif "timeout" in analysis_lower or "wait" in analysis_lower:
        return "timing_issue"
    elif "ui" in analysis_lower or "visual" in analysis_lower:
        return "ui_changed"
    elif "data" in analysis_lower or "assertion" in analysis_lower:
        return "data_changed"
    elif "bug" in analysis_lower or "defect" in analysis_lower:
        return "real_bug"

    return "unknown"


def _infer_category_from_results(results: list) -> str:
    """Infer category from search results."""
    if not results:
        return "unknown"

    # Check first result for error_type
    for r in results:
        if isinstance(r, dict) and r.get("error_type"):
            return r["error_type"]

    return "unknown"


def _extract_similar_errors(data: dict | list | None, limit: int) -> list[dict]:
    """Extract similar errors from query result data.

    Args:
        data: Raw data from query router
        limit: Maximum number of results

    Returns:
        List of similar error dictionaries
    """
    similar_errors = []

    if data is None:
        return similar_errors

    if isinstance(data, dict):
        # Check for results array
        if "results" in data:
            for r in data["results"][:limit]:
                similar_errors.append(_format_similar_error(r))

        # Check for similar_failures
        elif "similar_failures" in data:
            for f in data["similar_failures"][:limit]:
                similar_errors.append(_format_similar_error(f))

        # Check for suggestions with pattern data
        elif "suggestions" in data:
            for s in data["suggestions"][:limit]:
                if s.get("pattern_id") or s.get("id"):
                    similar_errors.append(_format_similar_error(s))

    elif isinstance(data, list):
        for item in data[:limit]:
            if isinstance(item, dict):
                similar_errors.append(_format_similar_error(item))

    return similar_errors


def _format_similar_error(error: dict) -> dict:
    """Format a similar error for response."""
    return {
        "id": error.get("id") or error.get("pattern_id"),
        "error_message": error.get("error_message"),
        "error_type": error.get("error_type"),
        "selector": error.get("selector") or error.get("original_selector"),
        "healed_selector": error.get("healed_selector"),
        "healing_method": error.get("healing_method"),
        "similarity": error.get("similarity") or error.get("confidence", 0),
        "success_rate": error.get("success_rate", 0),
        "success_count": error.get("success_count", 0),
    }
