"""Correlation API endpoints for the SDLC cross-correlation engine.

Query the unified SDLC timeline and get AI-powered insights. This is the key
differentiator for Argus - enabling correlation queries across the entire SDLC.

Provides endpoints for:
- Querying the unified event timeline
- Commit impact analysis
- Root cause tracing
- AI-generated correlation insights
- Natural language queries for correlation data
"""

from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.security.auth import UserContext, get_current_user
from src.config import get_settings
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/correlations", tags=["Correlations"])


# =============================================================================
# Response Models
# =============================================================================


class SDLCEvent(BaseModel):
    """A single SDLC event from the unified timeline."""

    id: str
    event_type: str
    source_platform: str
    external_id: str
    external_url: str | None = None
    title: str | None = None
    occurred_at: datetime
    commit_sha: str | None = None
    pr_number: int | None = None
    jira_key: str | None = None
    deploy_id: str | None = None
    data: dict = Field(default_factory=dict)


class CorrelationInsight(BaseModel):
    """An AI-generated insight from correlation analysis."""

    id: str
    insight_type: str
    severity: str
    title: str
    description: str
    recommendations: list[dict] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    status: str = "active"
    created_at: datetime


class TimelineResponse(BaseModel):
    """Response for timeline queries."""

    events: list[SDLCEvent]
    total_count: int


class ImpactAnalysisResponse(BaseModel):
    """Response for commit impact analysis."""

    commit_sha: str
    related_events: list[SDLCEvent]
    risk_score: float
    potential_impacts: list[str]


class RootCauseChain(BaseModel):
    """A chain of events leading to a root cause."""

    event: SDLCEvent
    correlation_type: str | None = None
    confidence: float = 1.0


class RootCauseResponse(BaseModel):
    """Response for root cause analysis."""

    target_event: SDLCEvent
    root_cause_chain: list[RootCauseChain]
    likely_root_cause: SDLCEvent | None = None
    confidence: float
    analysis_summary: str


class NLQueryResponse(BaseModel):
    """Response for natural language queries."""

    query: str
    interpreted_as: str
    events: list[SDLCEvent]
    insights: list[str]
    suggested_actions: list[str]


# =============================================================================
# Helper Functions
# =============================================================================


def _row_to_sdlc_event(row: dict) -> SDLCEvent:
    """Convert a database row to an SDLCEvent model."""
    return SDLCEvent(
        id=row["id"],
        event_type=row["event_type"],
        source_platform=row["source_platform"],
        external_id=row["external_id"],
        external_url=row.get("external_url"),
        title=row.get("title"),
        occurred_at=datetime.fromisoformat(
            row["occurred_at"].replace("Z", "+00:00")
        ) if isinstance(row["occurred_at"], str) else row["occurred_at"],
        commit_sha=row.get("commit_sha"),
        pr_number=row.get("pr_number"),
        jira_key=row.get("jira_key"),
        deploy_id=row.get("deploy_id"),
        data=row.get("data") or {},
    )


def _row_to_insight(row: dict) -> CorrelationInsight:
    """Convert a database row to a CorrelationInsight model."""
    return CorrelationInsight(
        id=row["id"],
        insight_type=row["insight_type"],
        severity=row["severity"],
        title=row["title"],
        description=row["description"],
        recommendations=row.get("recommendations") or [],
        event_ids=[str(eid) for eid in (row.get("event_ids") or [])],
        status=row.get("status", "active"),
        created_at=datetime.fromisoformat(
            row["created_at"].replace("Z", "+00:00")
        ) if isinstance(row["created_at"], str) else row["created_at"],
    )


async def _calculate_impact_risk_score(events: list[dict]) -> float:
    """Calculate a risk score based on related events."""
    if not events:
        return 0.0

    risk = 0.0
    weights = {
        "error": 0.3,
        "incident": 0.4,
        "test_run": 0.1,  # Failed test runs increase risk
        "deploy": 0.15,
        "commit": 0.05,
    }

    for event in events:
        event_type = event.get("event_type", "")
        base_weight = weights.get(event_type, 0.05)

        # Increase weight for errors and incidents
        if event_type in ("error", "incident"):
            # Check severity in data
            data = event.get("data") or {}
            severity = data.get("severity", "").lower()
            if severity in ("fatal", "critical"):
                base_weight *= 2.0
            elif severity == "error":
                base_weight *= 1.5

        risk += base_weight

    # Normalize to 0-1 range with a cap
    return min(1.0, risk / 2.0)


async def _get_potential_impacts(events: list[dict]) -> list[str]:
    """Generate a list of potential impact descriptions based on events."""
    impacts = []
    event_types_found = set(e.get("event_type") for e in events)

    if "error" in event_types_found:
        error_count = sum(1 for e in events if e.get("event_type") == "error")
        impacts.append(f"{error_count} production error(s) may be related to this commit")

    if "incident" in event_types_found:
        impacts.append("This commit may be linked to production incidents")

    if "test_run" in event_types_found:
        # Check for failed test runs
        failed_tests = [
            e for e in events
            if e.get("event_type") == "test_run"
            and (e.get("data") or {}).get("status") in ("failed", "error")
        ]
        if failed_tests:
            impacts.append(f"{len(failed_tests)} test run(s) failed after this commit")

    if "deploy" in event_types_found:
        impacts.append("Deployments were triggered that include this commit")

    if not impacts:
        impacts.append("No significant impacts detected")

    return impacts


# =============================================================================
# Timeline Endpoints
# =============================================================================


@router.get("/timeline", response_model=TimelineResponse)
async def get_timeline(
    request: Request,
    project_id: str | None = Query(None, description="Filter by project ID"),
    event_types: list[str] | None = Query(None, description="Filter by event types"),
    days: int = Query(7, ge=1, le=90, description="Number of days to look back"),
    limit: int = Query(100, ge=1, le=500, description="Maximum events to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    user: UserContext = Depends(get_current_user),
):
    """Get unified timeline of SDLC events.

    Returns events from all integrations in chronological order.
    This enables viewing the full picture of what happened across
    Jira, GitHub, Sentry, and other connected platforms.
    """
    supabase = get_supabase_client()

    try:
        # Build query
        since_date = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        query_path = f"/sdlc_events?occurred_at=gte.{since_date}"

        if project_id:
            query_path += f"&project_id=eq.{project_id}"

        if event_types:
            types_filter = ",".join(event_types)
            query_path += f"&event_type=in.({types_filter})"

        query_path += f"&order=occurred_at.desc&limit={limit}&offset={offset}"

        result = await supabase.request(query_path)

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            # Handle missing table gracefully
            if "does not exist" in error_msg or "42P01" in error_msg:
                logger.warning("sdlc_events table not found - run migrations")
                return TimelineResponse(events=[], total_count=0)
            raise HTTPException(status_code=500, detail="Failed to fetch timeline")

        events_data = result.get("data") or []
        events = [_row_to_sdlc_event(row) for row in events_data]

        # Get total count for pagination
        count_path = f"/sdlc_events?occurred_at=gte.{since_date}"
        if project_id:
            count_path += f"&project_id=eq.{project_id}"
        if event_types:
            count_path += f"&event_type=in.({','.join(event_types)})"
        count_path += "&select=count"

        count_result = await supabase.request(
            count_path,
            headers={"Prefer": "count=exact"},
        )
        total_count = len(events_data)
        if count_result.get("data"):
            # Try to get count from response
            if isinstance(count_result["data"], list) and count_result["data"]:
                total_count = count_result["data"][0].get("count", len(events_data))

        logger.info(
            "Timeline fetched",
            project_id=project_id,
            event_count=len(events),
            days=days,
            user_id=user.user_id,
        )

        return TimelineResponse(events=events, total_count=total_count)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch timeline", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch timeline: {str(e)}")


@router.get("/event/{event_id}")
async def get_event(
    event_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Get a single SDLC event by ID with full details."""
    supabase = get_supabase_client()

    try:
        result = await supabase.request(f"/sdlc_events?id=eq.{event_id}")

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                raise HTTPException(status_code=404, detail="Event not found")
            raise HTTPException(status_code=500, detail="Failed to fetch event")

        events_data = result.get("data") or []
        if not events_data:
            raise HTTPException(status_code=404, detail="Event not found")

        event = _row_to_sdlc_event(events_data[0])

        # Also fetch correlated events using the RPC function
        correlated_result = await supabase.rpc(
            "get_correlated_events",
            {"event_id": event_id},
        )

        correlated = []
        if correlated_result.get("data"):
            correlated = correlated_result["data"]

        return {
            "event": event,
            "correlations": correlated,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch event", error=str(e), event_id=event_id)
        raise HTTPException(status_code=500, detail=f"Failed to fetch event: {str(e)}")


# =============================================================================
# Impact Analysis Endpoints
# =============================================================================


@router.get("/impact/{commit_sha}", response_model=ImpactAnalysisResponse)
async def get_commit_impact(
    commit_sha: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Get impact analysis for a commit.

    Finds all downstream effects: builds, tests, deploys, errors.
    This helps answer "What happened after this commit was merged?"
    """
    supabase = get_supabase_client()

    try:
        # Find all events related to this commit
        result = await supabase.request(
            f"/sdlc_events?commit_sha=eq.{commit_sha}&order=occurred_at.asc"
        )

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                logger.warning("sdlc_events table not found")
                return ImpactAnalysisResponse(
                    commit_sha=commit_sha,
                    related_events=[],
                    risk_score=0.0,
                    potential_impacts=["Unable to analyze - SDLC events table not configured"],
                )
            raise HTTPException(status_code=500, detail="Failed to analyze impact")

        events_data = result.get("data") or []
        events = [_row_to_sdlc_event(row) for row in events_data]

        # Calculate risk score and impacts
        risk_score = await _calculate_impact_risk_score(events_data)
        potential_impacts = await _get_potential_impacts(events_data)

        logger.info(
            "Commit impact analyzed",
            commit_sha=commit_sha,
            related_events=len(events),
            risk_score=risk_score,
            user_id=user.user_id,
        )

        return ImpactAnalysisResponse(
            commit_sha=commit_sha,
            related_events=events,
            risk_score=risk_score,
            potential_impacts=potential_impacts,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to analyze commit impact", error=str(e), commit_sha=commit_sha)
        raise HTTPException(status_code=500, detail=f"Failed to analyze impact: {str(e)}")


# =============================================================================
# Root Cause Analysis Endpoints
# =============================================================================


@router.get("/root-cause/{event_id}", response_model=RootCauseResponse)
async def get_root_cause(
    event_id: str,
    request: Request,
    hours_before: int = Query(48, ge=1, le=168, description="Hours to look back for causes"),
    user: UserContext = Depends(get_current_user),
):
    """Trace back to find root cause of an error or incident.

    Returns the chain of events that led to this outcome, helping
    answer "Why did this error/incident happen?"
    """
    supabase = get_supabase_client()

    try:
        # Get the target event
        target_result = await supabase.request(f"/sdlc_events?id=eq.{event_id}")

        if target_result.get("error"):
            error_msg = str(target_result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                raise HTTPException(status_code=404, detail="Event not found")
            raise HTTPException(status_code=500, detail="Failed to fetch event")

        target_data = target_result.get("data") or []
        if not target_data:
            raise HTTPException(status_code=404, detail="Event not found")

        target_event = _row_to_sdlc_event(target_data[0])
        target_row = target_data[0]

        # Get timeline around this event using the RPC function
        timeline_result = await supabase.rpc(
            "get_event_timeline",
            {
                "target_event_id": event_id,
                "hours_before": hours_before,
                "hours_after": 0,  # Only look back, not forward
            },
        )

        root_cause_chain: list[RootCauseChain] = []
        likely_root_cause = None
        confidence = 0.0

        # Also check explicit correlations
        correlations_result = await supabase.request(
            f"/event_correlations?target_event_id=eq.{event_id}"
            "&correlation_type=in.(caused_by,introduced_by)"
            "&order=confidence.desc"
        )

        # Build the root cause chain
        if correlations_result.get("data"):
            for corr in correlations_result["data"]:
                # Fetch the source event
                source_result = await supabase.request(
                    f"/sdlc_events?id=eq.{corr['source_event_id']}"
                )
                if source_result.get("data"):
                    source_event = _row_to_sdlc_event(source_result["data"][0])
                    root_cause_chain.append(RootCauseChain(
                        event=source_event,
                        correlation_type=corr["correlation_type"],
                        confidence=float(corr.get("confidence", 1.0)),
                    ))

                    # The first caused_by with high confidence is likely the root cause
                    if not likely_root_cause and corr["correlation_type"] == "caused_by":
                        likely_root_cause = source_event
                        confidence = float(corr.get("confidence", 0.8))

        # If no explicit correlations, use heuristics based on timeline
        if not root_cause_chain and timeline_result.get("data"):
            for row in timeline_result["data"]:
                if row["event_id"] == event_id:
                    continue  # Skip the target event itself

                # Look for commits and deploys that happened before
                if row["event_type"] in ("commit", "deploy", "pr"):
                    event_result = await supabase.request(
                        f"/sdlc_events?id=eq.{row['event_id']}"
                    )
                    if event_result.get("data"):
                        chain_event = _row_to_sdlc_event(event_result["data"][0])
                        # Check if it shares correlation keys with target
                        confidence_score = 0.5
                        if (target_row.get("commit_sha") and
                                event_result["data"][0].get("commit_sha") == target_row.get("commit_sha")):
                            confidence_score = 0.8
                        elif (target_row.get("deploy_id") and
                              event_result["data"][0].get("deploy_id") == target_row.get("deploy_id")):
                            confidence_score = 0.7

                        root_cause_chain.append(RootCauseChain(
                            event=chain_event,
                            correlation_type="related_to",
                            confidence=confidence_score,
                        ))

                        if not likely_root_cause and row["event_type"] in ("commit", "deploy"):
                            likely_root_cause = chain_event
                            confidence = confidence_score

        # Sort chain by occurred_at (oldest first)
        root_cause_chain.sort(key=lambda x: x.event.occurred_at)

        # Generate analysis summary
        if likely_root_cause:
            analysis_summary = (
                f"The most likely root cause is a {likely_root_cause.event_type} event "
                f"({likely_root_cause.title or likely_root_cause.external_id}) "
                f"that occurred at {likely_root_cause.occurred_at.isoformat()}. "
                f"Confidence: {confidence * 100:.0f}%"
            )
        elif root_cause_chain:
            analysis_summary = (
                f"Found {len(root_cause_chain)} potentially related events in the "
                f"{hours_before} hours before this event, but no definitive root cause "
                "could be determined."
            )
        else:
            analysis_summary = (
                "No related events found in the specified time window. "
                "This may be an isolated incident or the root cause is not tracked in the system."
            )

        logger.info(
            "Root cause analyzed",
            event_id=event_id,
            chain_length=len(root_cause_chain),
            has_root_cause=likely_root_cause is not None,
            user_id=user.user_id,
        )

        return RootCauseResponse(
            target_event=target_event,
            root_cause_chain=root_cause_chain,
            likely_root_cause=likely_root_cause,
            confidence=confidence,
            analysis_summary=analysis_summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to analyze root cause", error=str(e), event_id=event_id)
        raise HTTPException(status_code=500, detail=f"Failed to analyze root cause: {str(e)}")


# =============================================================================
# Insights Endpoints
# =============================================================================


@router.get("/insights", response_model=list[CorrelationInsight])
async def get_insights(
    request: Request,
    project_id: str | None = Query(None, description="Filter by project ID"),
    insight_types: list[str] | None = Query(None, description="Filter by insight types"),
    status: str = Query("active", description="Filter by status (active, acknowledged, resolved, dismissed)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum insights to return"),
    user: UserContext = Depends(get_current_user),
):
    """Get AI-generated insights from correlation analysis.

    Returns insights about patterns, risks, and opportunities discovered
    by analyzing the unified SDLC timeline.
    """
    supabase = get_supabase_client()

    try:
        query_path = f"/correlation_insights?status=eq.{status}"

        if project_id:
            query_path += f"&project_id=eq.{project_id}"

        if insight_types:
            types_filter = ",".join(insight_types)
            query_path += f"&insight_type=in.({types_filter})"

        query_path += f"&order=severity.desc,created_at.desc&limit={limit}"

        result = await supabase.request(query_path)

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                logger.warning("correlation_insights table not found")
                return []
            raise HTTPException(status_code=500, detail="Failed to fetch insights")

        insights_data = result.get("data") or []
        insights = [_row_to_insight(row) for row in insights_data]

        logger.info(
            "Insights fetched",
            project_id=project_id,
            insight_count=len(insights),
            status=status,
            user_id=user.user_id,
        )

        return insights

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch insights", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch insights: {str(e)}")


@router.post("/insights/{insight_id}/acknowledge")
async def acknowledge_insight(
    insight_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Mark an insight as acknowledged.

    Acknowledging indicates the user has seen and is aware of the insight.
    """
    supabase = get_supabase_client()

    try:
        result = await supabase.update(
            "correlation_insights",
            {"id": f"eq.{insight_id}"},
            {
                "status": "acknowledged",
                "acknowledged_at": datetime.now(UTC).isoformat(),
                "acknowledged_by": user.user_id,
            },
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to acknowledge insight")

        logger.info(
            "Insight acknowledged",
            insight_id=insight_id,
            user_id=user.user_id,
        )

        return {"success": True, "message": "Insight acknowledged"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to acknowledge insight", error=str(e), insight_id=insight_id)
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge: {str(e)}")


@router.post("/insights/{insight_id}/resolve")
async def resolve_insight(
    insight_id: str,
    request: Request,
    resolution_notes: str | None = Query(None, description="Optional notes about resolution"),
    user: UserContext = Depends(get_current_user),
):
    """Mark an insight as resolved.

    Resolving indicates the issue identified by the insight has been addressed.
    """
    supabase = get_supabase_client()

    try:
        update_data = {
            "status": "resolved",
            "resolved_at": datetime.now(UTC).isoformat(),
            "resolved_by": user.user_id,
        }

        result = await supabase.update(
            "correlation_insights",
            {"id": f"eq.{insight_id}"},
            update_data,
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to resolve insight")

        logger.info(
            "Insight resolved",
            insight_id=insight_id,
            user_id=user.user_id,
        )

        return {"success": True, "message": "Insight resolved"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to resolve insight", error=str(e), insight_id=insight_id)
        raise HTTPException(status_code=500, detail=f"Failed to resolve: {str(e)}")


@router.post("/insights/{insight_id}/dismiss")
async def dismiss_insight(
    insight_id: str,
    request: Request,
    reason: str = Query(..., description="Reason for dismissing the insight"),
    user: UserContext = Depends(get_current_user),
):
    """Dismiss an insight as not relevant.

    Dismissing indicates the insight is not actionable or relevant to the team.
    """
    supabase = get_supabase_client()

    try:
        result = await supabase.update(
            "correlation_insights",
            {"id": f"eq.{insight_id}"},
            {
                "status": "dismissed",
                "dismissed_at": datetime.now(UTC).isoformat(),
                "dismissed_by": user.user_id,
                "dismiss_reason": reason,
            },
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to dismiss insight")

        logger.info(
            "Insight dismissed",
            insight_id=insight_id,
            reason=reason,
            user_id=user.user_id,
        )

        return {"success": True, "message": "Insight dismissed"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to dismiss insight", error=str(e), insight_id=insight_id)
        raise HTTPException(status_code=500, detail=f"Failed to dismiss: {str(e)}")


# =============================================================================
# AI Insights Generation Endpoint
# =============================================================================


class GenerateInsightsRequest(BaseModel):
    """Request body for generating AI insights."""

    days: int = Field(7, ge=1, le=90, description="Number of days to analyze")
    max_insights: int = Field(5, ge=1, le=20, description="Maximum insights to generate")
    save_to_database: bool = Field(True, description="Save generated insights to database")


class GenerateInsightsResponse(BaseModel):
    """Response for AI insights generation."""

    insights: list[CorrelationInsight]
    analysis_summary: dict
    generated_at: datetime


@router.post("/insights/generate", response_model=GenerateInsightsResponse)
async def generate_insights(
    request_body: GenerateInsightsRequest,
    request: Request,
    project_id: str = Query(..., description="Project ID to analyze"),
    user: UserContext = Depends(get_current_user),
):
    """Generate AI-powered insights from correlation analysis.

    Analyzes the SDLC timeline using the correlation engine to detect:
    - Failure clusters (related errors with common causes)
    - Deployment risks (deployments that led to errors)
    - Performance trends
    - Coverage gaps
    - General recommendations

    The AI analysis uses Claude to identify patterns that may not be
    obvious from simple rule-based detection.
    """
    from src.services.correlation_engine import get_correlation_engine

    settings = get_settings()
    engine = get_correlation_engine()

    try:
        # Generate insights using the correlation engine
        logger.info(
            "Generating AI insights",
            project_id=project_id,
            days=request_body.days,
            max_insights=request_body.max_insights,
            user_id=user.user_id,
        )

        generated = await engine.generate_insights(
            project_id=project_id,
            days=request_body.days,
            max_insights=request_body.max_insights,
        )

        # Convert to response format and optionally save
        insights: list[CorrelationInsight] = []
        saved_count = 0

        for gen_insight in generated:
            # Save to database if requested
            insight_id = None
            if request_body.save_to_database:
                insight_id = await engine.save_insight(project_id, gen_insight)
                if insight_id:
                    saved_count += 1

            # Create response model
            insights.append(CorrelationInsight(
                id=insight_id or f"temp-{len(insights)}",
                insight_type=gen_insight.insight_type.value,
                severity=gen_insight.severity.value,
                title=gen_insight.title,
                description=gen_insight.description,
                recommendations=gen_insight.recommendations,
                event_ids=gen_insight.event_ids,
                status="active",
                created_at=datetime.now(UTC),
            ))

        # Build analysis summary
        analysis_summary = {
            "project_id": project_id,
            "days_analyzed": request_body.days,
            "insights_generated": len(insights),
            "insights_saved": saved_count,
            "insight_types": list(set(i.insight_type for i in insights)),
            "severities": {
                "critical": sum(1 for i in insights if i.severity == "critical"),
                "high": sum(1 for i in insights if i.severity == "high"),
                "medium": sum(1 for i in insights if i.severity == "medium"),
                "low": sum(1 for i in insights if i.severity == "low"),
                "info": sum(1 for i in insights if i.severity == "info"),
            },
        }

        logger.info(
            "AI insights generated successfully",
            project_id=project_id,
            insights_count=len(insights),
            saved_count=saved_count,
            user_id=user.user_id,
        )

        return GenerateInsightsResponse(
            insights=insights,
            analysis_summary=analysis_summary,
            generated_at=datetime.now(UTC),
        )

    except Exception as e:
        logger.exception("Failed to generate insights", error=str(e), project_id=project_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate insights: {str(e)}"
        )


# =============================================================================
# Commit Impact Analysis Endpoint (Alternative Path)
# =============================================================================


class CommitImpactResponse(BaseModel):
    """Response for commit impact analysis using the correlation engine."""

    commit_sha: str
    found: bool
    total_events: int = 0
    events_by_type: dict = Field(default_factory=dict)
    event_ids: list[str] = Field(default_factory=list)
    risk_score: float = 0.0
    risk_factors: list[dict] = Field(default_factory=list)
    time_span_hours: float = 0.0
    first_event: str | None = None
    last_event: str | None = None
    message: str | None = None


@router.get("/impact", response_model=CommitImpactResponse)
async def get_commit_impact_analysis(
    request: Request,
    commit_sha: str = Query(..., description="Commit SHA to analyze"),
    project_id: str = Query(..., description="Project ID"),
    user: UserContext = Depends(get_current_user),
):
    """Analyze the downstream impact of a specific commit.

    Uses the correlation engine to find all events related to this commit
    and calculate a risk score based on:
    - Production errors caused
    - Incidents triggered
    - Test failures
    - Time span of impact

    This endpoint provides a more comprehensive analysis than the simple
    /impact/{commit_sha} endpoint by using the correlation engine's
    algorithms.
    """
    from src.services.correlation_engine import get_correlation_engine

    engine = get_correlation_engine()

    try:
        result = await engine.analyze_commit_impact(project_id, commit_sha)

        logger.info(
            "Commit impact analyzed",
            commit_sha=commit_sha,
            project_id=project_id,
            found=result.get("found", False),
            risk_score=result.get("risk_score", 0),
            user_id=user.user_id,
        )

        return CommitImpactResponse(**result)

    except Exception as e:
        logger.exception(
            "Failed to analyze commit impact",
            error=str(e),
            commit_sha=commit_sha,
            project_id=project_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze commit impact: {str(e)}"
        )


# =============================================================================
# Natural Language Query Endpoint
# =============================================================================


@router.post("/query", response_model=NLQueryResponse)
async def natural_language_query(
    request: Request,
    query: str = Query(..., min_length=5, description="Natural language query"),
    project_id: str | None = Query(None, description="Filter results to specific project"),
    user: UserContext = Depends(get_current_user),
):
    """Natural language query for correlation data.

    Uses AI to interpret natural language questions and translate them
    into database queries. Supports questions like:
    - "Show me PRs without tests that caused errors"
    - "What deployments happened in the last week?"
    - "Find errors related to the payment component"
    """
    settings = get_settings()
    supabase = get_supabase_client()

    try:
        # Use Claude to interpret the query
        if not settings.anthropic_api_key:
            raise HTTPException(
                status_code=500,
                detail="Anthropic API key not configured for natural language queries"
            )

        import anthropic

        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )

        # Prepare the prompt for query interpretation
        interpretation_prompt = f"""You are an expert at understanding software development lifecycle (SDLC) queries.
Given a natural language question about SDLC events, determine the appropriate database filters.

The SDLC events database has these event types:
- requirement (Jira tickets)
- pr (Pull requests)
- commit (Git commits)
- build (CI/CD builds)
- test_run (Test executions)
- deploy (Deployments)
- error (Production errors)
- incident (PagerDuty/Opsgenie incidents)
- feature_flag (Feature flag changes)
- session (User session recordings)

Source platforms include: jira, github, gitlab, sentry, pagerduty, launchdarkly, argus

Correlation keys: commit_sha, pr_number, jira_key, deploy_id

USER QUERY: {query}

Respond with a JSON object containing:
{{
  "interpreted_as": "A clear description of what the query is asking",
  "event_types": ["list", "of", "relevant", "event_types"] or null for all,
  "source_platforms": ["list", "of", "platforms"] or null for all,
  "time_range_days": number of days to look back (default 7),
  "correlation_key": "commit_sha" or "pr_number" or "jira_key" or null,
  "correlation_value": "specific value if mentioned" or null,
  "insights": ["helpful insight 1", "helpful insight 2"],
  "suggested_actions": ["action 1", "action 2"]
}}

Return ONLY valid JSON, no markdown or explanations."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            temperature=0.2,
            messages=[{"role": "user", "content": interpretation_prompt}],
        )

        # Parse the AI response
        import json
        try:
            interpretation = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            interpretation = {
                "interpreted_as": query,
                "event_types": None,
                "time_range_days": 7,
                "insights": ["Unable to fully interpret query, showing recent events"],
                "suggested_actions": ["Try rephrasing your question"],
            }

        # Build the query based on interpretation
        days = interpretation.get("time_range_days", 7)
        since_date = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        query_path = f"/sdlc_events?occurred_at=gte.{since_date}"

        if project_id:
            query_path += f"&project_id=eq.{project_id}"

        if interpretation.get("event_types"):
            types_filter = ",".join(interpretation["event_types"])
            query_path += f"&event_type=in.({types_filter})"

        if interpretation.get("source_platforms"):
            platforms_filter = ",".join(interpretation["source_platforms"])
            query_path += f"&source_platform=in.({platforms_filter})"

        # Handle correlation key filtering
        if interpretation.get("correlation_key") and interpretation.get("correlation_value"):
            key = interpretation["correlation_key"]
            value = interpretation["correlation_value"]
            query_path += f"&{key}=eq.{value}"

        query_path += "&order=occurred_at.desc&limit=50"

        result = await supabase.request(query_path)

        events = []
        if not result.get("error") and result.get("data"):
            events = [_row_to_sdlc_event(row) for row in result["data"]]

        logger.info(
            "Natural language query executed",
            query=query[:100],
            interpreted_as=interpretation.get("interpreted_as", "")[:100],
            result_count=len(events),
            user_id=user.user_id,
        )

        return NLQueryResponse(
            query=query,
            interpreted_as=interpretation.get("interpreted_as", query),
            events=events,
            insights=interpretation.get("insights", []),
            suggested_actions=interpretation.get("suggested_actions", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to execute natural language query", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute query: {str(e)}"
        )


# =============================================================================
# Event Correlation Endpoints
# =============================================================================


@router.get("/by-commit/{commit_sha}")
async def get_events_by_commit(
    commit_sha: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Get all events related to a specific commit SHA."""
    supabase = get_supabase_client()

    try:
        result = await supabase.request(
            f"/sdlc_events?commit_sha=eq.{commit_sha}&order=occurred_at.asc"
        )

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return {"commit_sha": commit_sha, "events": []}
            raise HTTPException(status_code=500, detail="Failed to fetch events")

        events_data = result.get("data") or []
        events = [_row_to_sdlc_event(row) for row in events_data]

        return {"commit_sha": commit_sha, "events": events}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch events by commit", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch events: {str(e)}")


@router.get("/by-pr/{pr_number}")
async def get_events_by_pr(
    pr_number: int,
    request: Request,
    project_id: str | None = Query(None, description="Filter by project ID"),
    user: UserContext = Depends(get_current_user),
):
    """Get all events related to a specific pull request number."""
    supabase = get_supabase_client()

    try:
        query_path = f"/sdlc_events?pr_number=eq.{pr_number}&order=occurred_at.asc"
        if project_id:
            query_path += f"&project_id=eq.{project_id}"

        result = await supabase.request(query_path)

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return {"pr_number": pr_number, "events": []}
            raise HTTPException(status_code=500, detail="Failed to fetch events")

        events_data = result.get("data") or []
        events = [_row_to_sdlc_event(row) for row in events_data]

        return {"pr_number": pr_number, "events": events}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch events by PR", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch events: {str(e)}")


@router.get("/by-jira/{jira_key}")
async def get_events_by_jira(
    jira_key: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Get all events related to a specific Jira issue key (e.g., PROJ-123)."""
    supabase = get_supabase_client()

    try:
        result = await supabase.request(
            f"/sdlc_events?jira_key=eq.{jira_key}&order=occurred_at.asc"
        )

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return {"jira_key": jira_key, "events": []}
            raise HTTPException(status_code=500, detail="Failed to fetch events")

        events_data = result.get("data") or []
        events = [_row_to_sdlc_event(row) for row in events_data]

        return {"jira_key": jira_key, "events": events}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch events by Jira key", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch events: {str(e)}")


@router.get("/by-deploy/{deploy_id}")
async def get_events_by_deploy(
    deploy_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Get all events related to a specific deployment."""
    supabase = get_supabase_client()

    try:
        result = await supabase.request(
            f"/sdlc_events?deploy_id=eq.{deploy_id}&order=occurred_at.asc"
        )

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return {"deploy_id": deploy_id, "events": []}
            raise HTTPException(status_code=500, detail="Failed to fetch events")

        events_data = result.get("data") or []
        events = [_row_to_sdlc_event(row) for row in events_data]

        return {"deploy_id": deploy_id, "events": events}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch events by deploy", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch events: {str(e)}")


# =============================================================================
# Statistics Endpoints
# =============================================================================


@router.get("/stats")
async def get_correlation_stats(
    request: Request,
    project_id: str | None = Query(None, description="Filter by project ID"),
    days: int = Query(30, ge=1, le=90, description="Days to analyze"),
    user: UserContext = Depends(get_current_user),
):
    """Get correlation statistics and metrics.

    Provides an overview of SDLC events, correlations, and insights.
    """
    supabase = get_supabase_client()

    try:
        since_date = (datetime.now(UTC) - timedelta(days=days)).isoformat()

        # Get event counts by type
        events_query = f"/sdlc_events?occurred_at=gte.{since_date}&select=event_type"
        if project_id:
            events_query += f"&project_id=eq.{project_id}"

        events_result = await supabase.request(events_query)
        events_data = events_result.get("data") or []

        # Count by type
        event_counts: dict[str, int] = {}
        for event in events_data:
            event_type = event.get("event_type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Get insight counts by status
        insights_query = f"/correlation_insights?created_at=gte.{since_date}&select=status,severity"
        if project_id:
            insights_query += f"&project_id=eq.{project_id}"

        insights_result = await supabase.request(insights_query)
        insights_data = insights_result.get("data") or []

        insight_counts = {"total": len(insights_data)}
        for insight in insights_data:
            status = insight.get("status", "unknown")
            severity = insight.get("severity", "info")
            insight_counts[f"status_{status}"] = insight_counts.get(f"status_{status}", 0) + 1
            insight_counts[f"severity_{severity}"] = insight_counts.get(f"severity_{severity}", 0) + 1

        # Get correlation counts
        correlations_query = "/event_correlations?select=correlation_type"
        correlations_result = await supabase.request(correlations_query)
        correlations_data = correlations_result.get("data") or []

        correlation_counts: dict[str, int] = {"total": len(correlations_data)}
        for corr in correlations_data:
            corr_type = corr.get("correlation_type", "unknown")
            correlation_counts[corr_type] = correlation_counts.get(corr_type, 0) + 1

        return {
            "time_range_days": days,
            "project_id": project_id,
            "events": {
                "total": len(events_data),
                "by_type": event_counts,
            },
            "insights": insight_counts,
            "correlations": correlation_counts,
            "generated_at": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception("Failed to get correlation stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
