"""Activity Feed API endpoints.

Provides endpoints for:
- Fetching aggregated activity from multiple sources
- Activity statistics for dashboard widgets
"""

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from src.api.middleware.tenant import validate_uuid, validate_uuid_optional
from src.api.security.auth import UserContext, get_current_user
from src.services.supabase_client import get_supabase_client
from src.utils import safe_datetime

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/activity", tags=["Activity Feed"])


# ============================================================================
# Request/Response Models
# ============================================================================


class ActivityUser(BaseModel):
    """User info for activity event."""
    name: str
    avatar: str | None = None


class ActivityMetadata(BaseModel):
    """Metadata for activity event."""
    project_id: str | None = None
    project_name: str | None = None
    test_id: str | None = None
    test_name: str | None = None
    duration: int | None = None
    link: str | None = None


class ActivityEvent(BaseModel):
    """Single activity event."""
    id: str
    type: str
    title: str
    description: str
    timestamp: str
    user: ActivityUser | None = None
    metadata: ActivityMetadata | None = None


class ActivityFeedResponse(BaseModel):
    """Paginated activity feed response."""
    activities: list[ActivityEvent]
    total: int


class ActivityStatsResponse(BaseModel):
    """Activity statistics for dashboard."""
    last_hour: int
    test_runs: int
    heals_applied: int
    failures: int


# ============================================================================
# Helper Functions
# ============================================================================


def _get_activity_link(activity_type: str) -> str:
    """Get the link for an activity type."""
    links = {
        "test_run": "/tests",
        "discovery": "/discovery",
        "visual_test": "/visual",
        "quality_audit": "/quality",
    }
    return links.get(activity_type, "/")


def _map_activity_log_type(activity_type: str, event_type: str, metadata: dict | None) -> str:
    """Map activity_type and event_type to frontend event type."""
    if activity_type == "test_run":
        if event_type == "started":
            return "test_started"
        elif event_type == "completed":
            success = (metadata or {}).get("success", False)
            return "test_passed" if success else "test_failed"
    elif activity_type == "discovery":
        return "discovery_completed" if event_type == "completed" else "discovery_started"
    elif activity_type == "visual_test":
        return "visual_test_completed" if event_type == "completed" else "visual_test_started"
    elif activity_type == "quality_audit":
        return "quality_audit_completed" if event_type == "completed" else "quality_audit_started"
    return "test_started"


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/feed", response_model=ActivityFeedResponse)
async def get_activity_feed(
    request: Request,
    project_ids: str = Query(..., description="Comma-separated list of project IDs"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    user: UserContext = Depends(get_current_user),
):
    """Get aggregated activity feed from multiple sources.

    Aggregates activity from:
    - activity_logs table
    - test_runs table
    - discovery_sessions table
    - healing_patterns table
    - schedule_runs table

    Results are sorted by timestamp descending.
    """
    # Parse and validate project IDs
    project_id_list = [pid.strip() for pid in project_ids.split(",") if pid.strip()]
    for pid in project_id_list:
        validate_uuid(pid, "project_id")

    if not project_id_list:
        return ActivityFeedResponse(activities=[], total=0)

    supabase = get_supabase_client()
    activities: list[ActivityEvent] = []

    # Build project ID filter
    project_ids_filter = ",".join(project_id_list)

    # 1. Get activity_logs
    try:
        activity_logs_result = await supabase.request(
            f"/activity_logs?project_id=in.({project_ids_filter})"
            f"&order=created_at.desc&limit={limit}"
        )
        if not activity_logs_result.get("error"):
            for log in activity_logs_result.get("data") or []:
                event_type = _map_activity_log_type(
                    log.get("activity_type", ""),
                    log.get("event_type", ""),
                    log.get("metadata")
                )
                activities.append(ActivityEvent(
                    id=log["id"],
                    type=event_type,
                    title=log.get("title", ""),
                    description=log.get("description") or "",
                    timestamp=safe_datetime(log.get("created_at")),
                    user=ActivityUser(name="System"),
                    metadata=ActivityMetadata(
                        project_id=log.get("project_id"),
                        duration=log.get("duration_ms"),
                        link=_get_activity_link(log.get("activity_type", "")),
                    ),
                ))
    except Exception as e:
        logger.warning("Failed to fetch activity_logs", error=str(e))

    # 2. Get recent test runs
    try:
        test_runs_result = await supabase.request(
            f"/test_runs?project_id=in.({project_ids_filter})"
            "&select=id,project_id,name,status,trigger,total_tests,passed_tests,failed_tests,duration_ms,created_at,started_at,completed_at"
            f"&order=created_at.desc&limit=20"
        )
        if not test_runs_result.get("error"):
            for run in test_runs_result.get("data") or []:
                status = run.get("status", "")
                name = run.get("name") or "Test run"
                total_tests = run.get("total_tests", 0)
                failed_tests = run.get("failed_tests", 0)

                if status == "passed":
                    event_type = "test_passed"
                    title = "Test run completed successfully"
                    description = f"{name} passed all {total_tests} tests"
                elif status == "failed":
                    event_type = "test_failed"
                    title = "Test run failed"
                    description = f"{name} failed: {failed_tests}/{total_tests} tests failed"
                elif status == "running":
                    event_type = "test_started"
                    title = "Test run in progress"
                    description = f"{name} is currently running"
                else:
                    event_type = "test_started"
                    title = "Test run started"
                    description = f"{name} started"

                trigger = run.get("trigger", "")
                user_name = "Scheduler" if trigger == "scheduled" else "System"

                activities.append(ActivityEvent(
                    id=f"run-{run['id']}",
                    type=event_type,
                    title=title,
                    description=description,
                    timestamp=safe_datetime(run.get("completed_at") or run.get("started_at") or run.get("created_at")),
                    user=ActivityUser(name=user_name),
                    metadata=ActivityMetadata(
                        project_id=run.get("project_id"),
                        duration=run.get("duration_ms"),
                        link="/tests",
                    ),
                ))
    except Exception as e:
        logger.warning("Failed to fetch test_runs", error=str(e))

    # 3. Get recent discoveries
    try:
        discoveries_result = await supabase.request(
            f"/discovery_sessions?project_id=in.({project_ids_filter})"
            "&select=id,project_id,status,pages_found,flows_found,created_at,completed_at"
            f"&order=created_at.desc&limit=10"
        )
        if not discoveries_result.get("error"):
            for disc in discoveries_result.get("data") or []:
                is_complete = disc.get("status") == "completed"
                pages_found = disc.get("pages_found", 0)
                flows_found = disc.get("flows_found", 0)

                activities.append(ActivityEvent(
                    id=f"disc-{disc['id']}",
                    type="discovery_completed" if is_complete else "discovery_started",
                    title="Discovery completed" if is_complete else "Discovery started",
                    description=f"Found {pages_found} pages and {flows_found} flows" if is_complete else "AI discovery session started",
                    timestamp=safe_datetime(disc.get("completed_at") or disc.get("created_at")),
                    user=ActivityUser(name="Argus AI"),
                    metadata=ActivityMetadata(
                        project_id=disc.get("project_id"),
                        link="/discovery",
                    ),
                ))
    except Exception as e:
        logger.warning("Failed to fetch discovery_sessions", error=str(e))

    # 4. Get recent healing events
    try:
        healings_result = await supabase.request(
            f"/healing_patterns?project_id=in.({project_ids_filter})"
            "&select=id,project_id,error_type,original_selector,healed_selector,confidence,success_count,created_at"
            f"&order=created_at.desc&limit=10"
        )
        if not healings_result.get("error"):
            for heal in healings_result.get("data") or []:
                is_applied = (heal.get("success_count") or 0) > 0
                original = heal.get("original_selector") or ""
                healed = heal.get("healed_selector") or ""

                if original and healed:
                    description = f"Updated selector: {original[:30]}... -> {healed[:30]}..."
                else:
                    description = "Fix identified for test issue"

                activities.append(ActivityEvent(
                    id=f"heal-{heal['id']}",
                    type="healing_applied" if is_applied else "healing_suggested",
                    title="Self-healing fix applied" if is_applied else "Healing suggestion available",
                    description=description,
                    timestamp=safe_datetime(heal.get("created_at")),
                    user=ActivityUser(name="Argus AI"),
                    metadata=ActivityMetadata(
                        project_id=heal.get("project_id"),
                        link="/healing",
                    ),
                ))
    except Exception as e:
        logger.warning("Failed to fetch healing_patterns", error=str(e))

    # 5. Get recent schedule runs
    try:
        schedule_runs_result = await supabase.request(
            "/schedule_runs?"
            "select=id,schedule_id,status,trigger_type,tests_total,tests_failed,triggered_at,"
            "test_schedules(name,project_id)"
            f"&order=triggered_at.desc&limit=10"
        )
        if not schedule_runs_result.get("error"):
            for run in schedule_runs_result.get("data") or []:
                schedule = run.get("test_schedules")
                if not schedule:
                    continue

                # Filter by project_id
                schedule_project_id = schedule.get("project_id")
                if schedule_project_id not in project_id_list:
                    continue

                trigger_type = run.get("trigger_type", "")
                tests_total = run.get("tests_total", 0)
                schedule_name = schedule.get("name", "Schedule")

                description = f"{schedule_name} {'manually triggered' if trigger_type == 'manual' else 'started'} ({tests_total} tests)"
                user_name = "User" if trigger_type == "manual" else "Scheduler"

                activities.append(ActivityEvent(
                    id=f"sched-{run['id']}",
                    type="schedule_triggered",
                    title="Scheduled run triggered",
                    description=description,
                    timestamp=run.get("triggered_at") or datetime.now(UTC).isoformat(),
                    user=ActivityUser(name=user_name),
                    metadata=ActivityMetadata(
                        project_id=schedule_project_id,
                        link="/schedules",
                    ),
                ))
    except Exception as e:
        logger.warning("Failed to fetch schedule_runs", error=str(e))

    # Sort all activities by timestamp and return top limit
    activities.sort(key=lambda a: a.timestamp, reverse=True)
    activities = activities[:limit]

    return ActivityFeedResponse(
        activities=activities,
        total=len(activities),
    )


@router.get("/stats", response_model=ActivityStatsResponse)
async def get_activity_stats(
    request: Request,
    project_ids: str = Query(..., description="Comma-separated list of project IDs"),
    user: UserContext = Depends(get_current_user),
):
    """Get activity statistics for dashboard widgets.

    Calculates:
    - Events in the last hour
    - Total test runs (passed + failed)
    - Healing fixes applied
    - Test failures
    """
    # Parse and validate project IDs
    project_id_list = [pid.strip() for pid in project_ids.split(",") if pid.strip()]
    for pid in project_id_list:
        validate_uuid(pid, "project_id")

    if not project_id_list:
        return ActivityStatsResponse(
            last_hour=0,
            test_runs=0,
            heals_applied=0,
            failures=0,
        )

    # Get activity feed with 100 items for stats calculation
    feed_response = await get_activity_feed(
        request=request,
        project_ids=project_ids,
        limit=100,
        user=user,
    )

    activities = feed_response.activities
    now = datetime.now(UTC)
    one_hour_ago_ts = (now.timestamp() - 3600) * 1000  # milliseconds

    last_hour = 0
    test_runs = 0
    heals_applied = 0
    failures = 0

    for activity in activities:
        # Parse timestamp and check if within last hour
        try:
            activity_ts = datetime.fromisoformat(activity.timestamp.replace("Z", "+00:00"))
            if activity_ts.timestamp() > (now.timestamp() - 3600):
                last_hour += 1
        except (ValueError, AttributeError):
            pass

        # Count test runs
        if activity.type in ("test_passed", "test_failed"):
            test_runs += 1

        # Count heals applied
        if activity.type == "healing_applied":
            heals_applied += 1

        # Count failures
        if activity.type == "test_failed":
            failures += 1

    return ActivityStatsResponse(
        last_hour=last_hour,
        test_runs=test_runs,
        heals_applied=heals_applied,
        failures=failures,
    )
