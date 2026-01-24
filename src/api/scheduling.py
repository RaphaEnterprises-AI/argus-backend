"""Test Scheduling API endpoints.

Provides endpoints for:
- Creating and managing scheduled test runs
- Cron-style scheduling with croniter pattern
- Manual trigger of scheduled runs
- Run history and statistics

Now with Supabase persistence for production use.
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Literal

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.integrations.supabase import get_supabase, is_supabase_configured

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/schedules", tags=["Scheduling"])

# Active run queues for SSE streaming
_active_run_queues: dict[str, asyncio.Queue] = {}


# ============================================================================
# In-Memory Storage (fallback when Supabase not configured)
# ============================================================================

schedules: dict[str, dict] = {}
schedule_runs: dict[str, list[dict]] = {}  # schedule_id -> list of runs


# ============================================================================
# Supabase Helper Functions
# ============================================================================

async def _get_schedule_from_db(schedule_id: str) -> dict | None:
    """Get a schedule from Supabase by ID."""
    supabase = await get_supabase()
    if not supabase:
        return schedules.get(schedule_id)

    result = await supabase.select(
        "test_schedules",
        columns="*",
        filters={"id": schedule_id},
        limit=1
    )
    return result[0] if result else None


async def _get_schedule_runs_from_db(schedule_id: str, status: str | None = None, limit: int = 20, offset: int = 0) -> list[dict]:
    """Get schedule runs from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        runs = schedule_runs.get(schedule_id, [])
        if status:
            runs = [r for r in runs if r.get("status") == status]
        return runs[offset:offset + limit]

    filters = {"schedule_id": schedule_id}
    if status:
        filters["status"] = status

    result = await supabase.select(
        "schedule_runs",
        columns="*",
        filters=filters,
        order_by="triggered_at",
        ascending=False,
        limit=limit
    )
    return result


async def _save_schedule_to_db(schedule: dict) -> bool:
    """Save a schedule to Supabase."""
    supabase = await get_supabase()
    if not supabase:
        schedules[schedule["id"]] = schedule
        return True

    try:
        success = await supabase.insert("test_schedules", [schedule])
        if not success:
            logger.error(
                "Failed to save schedule to database",
                schedule_id=schedule.get("id"),
                schedule_name=schedule.get("name"),
            )
        return success
    except Exception as e:
        logger.exception(
            "Exception while saving schedule to database",
            schedule_id=schedule.get("id"),
            schedule_name=schedule.get("name"),
            error=str(e),
        )
        return False


async def _update_schedule_in_db(schedule_id: str, updates: dict) -> bool:
    """Update a schedule in Supabase."""
    supabase = await get_supabase()
    if not supabase:
        if schedule_id in schedules:
            schedules[schedule_id].update(updates)
            return True
        return False

    try:
        success = await supabase.update("test_schedules", updates, {"id": schedule_id})
        if not success:
            logger.error(
                "Failed to update schedule in database",
                schedule_id=schedule_id,
                update_keys=list(updates.keys()),
            )
        return success
    except Exception as e:
        logger.exception(
            "Exception while updating schedule in database",
            schedule_id=schedule_id,
            update_keys=list(updates.keys()),
            error=str(e),
        )
        return False


async def _delete_schedule_from_db(schedule_id: str) -> bool:
    """Delete a schedule from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        if schedule_id in schedules:
            del schedules[schedule_id]
            return True
        return False

    try:
        success = await supabase.delete("test_schedules", {"id": schedule_id})
        if not success:
            logger.error(
                "Failed to delete schedule from database",
                schedule_id=schedule_id,
            )
        return success
    except Exception as e:
        logger.exception(
            "Exception while deleting schedule from database",
            schedule_id=schedule_id,
            error=str(e),
        )
        return False


async def _save_schedule_run_to_db(run: dict) -> bool:
    """Save a schedule run to Supabase."""
    supabase = await get_supabase()
    if not supabase:
        runs = schedule_runs.setdefault(run["schedule_id"], [])
        runs.append(run)
        return True

    try:
        success = await supabase.insert("schedule_runs", [run])
        if not success:
            logger.error(
                "Failed to save schedule run to database",
                run_id=run.get("id"),
                schedule_id=run.get("schedule_id"),
            )
        return success
    except Exception as e:
        logger.exception(
            "Exception while saving schedule run to database",
            run_id=run.get("id"),
            schedule_id=run.get("schedule_id"),
            error=str(e),
        )
        return False


async def _update_schedule_run_in_db(run_id: str, updates: dict) -> bool:
    """Update a schedule run in Supabase."""
    supabase = await get_supabase()
    if not supabase:
        for runs in schedule_runs.values():
            for run in runs:
                if run["id"] == run_id:
                    run.update(updates)
                    return True
        return False

    try:
        success = await supabase.update("schedule_runs", updates, {"id": run_id})
        if not success:
            logger.error(
                "Failed to update schedule run in database",
                run_id=run_id,
                update_keys=list(updates.keys()),
            )
        return success
    except Exception as e:
        logger.exception(
            "Exception while updating schedule run in database",
            run_id=run_id,
            update_keys=list(updates.keys()),
            error=str(e),
        )
        return False


# ============================================================================
# Cron Validation & Calculation Helpers
# ============================================================================

# Standard cron field patterns
CRON_FIELD_PATTERNS = {
    "minute": r"^(\*|[0-5]?\d)(\/\d+)?$|^(\*|[0-5]?\d)(-[0-5]?\d)?(,(\*|[0-5]?\d)(-[0-5]?\d)?)*$",
    "hour": r"^(\*|1?\d|2[0-3])(\/\d+)?$|^(\*|1?\d|2[0-3])(-(?:1?\d|2[0-3]))?(,(\*|1?\d|2[0-3])(-(?:1?\d|2[0-3]))?)*$",
    "day_of_month": r"^(\*|[1-9]|[12]\d|3[01])(\/\d+)?$|^(\*|[1-9]|[12]\d|3[01])(-(?:[1-9]|[12]\d|3[01]))?(,(\*|[1-9]|[12]\d|3[01])(-(?:[1-9]|[12]\d|3[01]))?)*$",
    "month": r"^(\*|[1-9]|1[0-2])(\/\d+)?$|^(\*|[1-9]|1[0-2])(-(?:[1-9]|1[0-2]))?(,(\*|[1-9]|1[0-2])(-(?:[1-9]|1[0-2]))?)*$",
    "day_of_week": r"^(\*|[0-6])(\/\d+)?$|^(\*|[0-6])(-[0-6])?(,(\*|[0-6])(-[0-6])?)*$",
}


def validate_cron_expression(cron_expression: str) -> tuple[bool, str | None]:
    """
    Validate a cron expression syntax.

    Supports standard 5-field cron format:
    - minute (0-59)
    - hour (0-23)
    - day of month (1-31)
    - month (1-12)
    - day of week (0-6, where 0 is Sunday)

    Special characters supported:
    - * (any value)
    - , (value list separator)
    - - (range of values)
    - / (step values)

    Returns:
        tuple of (is_valid, error_message)
    """
    if not cron_expression or not isinstance(cron_expression, str):
        return False, "Cron expression cannot be empty"

    parts = cron_expression.strip().split()

    if len(parts) != 5:
        return False, f"Invalid cron expression: expected 5 fields, got {len(parts)}"

    field_ranges = [
        (0, 59, "minute"),
        (0, 23, "hour"),
        (1, 31, "day of month"),
        (1, 12, "month"),
        (0, 6, "day of week"),
    ]

    for i, (part, (min_val, max_val, name)) in enumerate(zip(parts, field_ranges)):
        # Check for wildcard
        if part == "*":
            continue

        # Check for step values (e.g., */5)
        if "/" in part:
            base, step = part.split("/", 1)
            if not step.isdigit() or int(step) < 1:
                return False, f"Invalid step value in {name} field: {step}"
            if base != "*" and not base.isdigit():
                return False, f"Invalid base value in {name} field: {base}"
            continue

        # Check for range (e.g., 1-5)
        if "-" in part:
            range_parts = part.split("-")
            if len(range_parts) != 2:
                return False, f"Invalid range in {name} field: {part}"
            try:
                start, end = int(range_parts[0]), int(range_parts[1])
                if start < min_val or end > max_val or start > end:
                    return False, f"Range out of bounds in {name} field: {part} (valid: {min_val}-{max_val})"
            except ValueError:
                return False, f"Invalid range values in {name} field: {part}"
            continue

        # Check for list (e.g., 1,3,5)
        if "," in part:
            for item in part.split(","):
                try:
                    val = int(item)
                    if val < min_val or val > max_val:
                        return False, f"Value out of range in {name} field: {item} (valid: {min_val}-{max_val})"
                except ValueError:
                    return False, f"Invalid value in {name} field: {item}"
            continue

        # Check single value
        try:
            val = int(part)
            if val < min_val or val > max_val:
                return False, f"Value out of range in {name} field: {part} (valid: {min_val}-{max_val})"
        except ValueError:
            return False, f"Invalid value in {name} field: {part}"

    return True, None


def calculate_next_run(cron_expression: str, from_time: datetime | None = None) -> datetime | None:
    """
    Calculate the next run time based on cron expression.

    This is a simplified implementation. For production, use croniter library:
        from croniter import croniter
        cron = croniter(cron_expression, from_time)
        return cron.get_next(datetime)

    Args:
        cron_expression: Standard 5-field cron expression
        from_time: Starting time to calculate from (defaults to now)

    Returns:
        Next scheduled run time as datetime, or None if invalid
    """
    is_valid, error = validate_cron_expression(cron_expression)
    if not is_valid:
        return None

    if from_time is None:
        from_time = datetime.now(UTC)

    parts = cron_expression.strip().split()
    minute, hour, day_of_month, month, day_of_week = parts

    # Simple implementation for common patterns
    # For production, use croniter library

    def parse_field(field: str, min_val: int, max_val: int) -> list[int]:
        """Parse a cron field into list of valid values."""
        if field == "*":
            return list(range(min_val, max_val + 1))

        if "/" in field:
            base, step = field.split("/")
            step = int(step)
            if base == "*":
                return list(range(min_val, max_val + 1, step))
            else:
                start = int(base)
                return list(range(start, max_val + 1, step))

        if "-" in field:
            start, end = map(int, field.split("-"))
            return list(range(start, end + 1))

        if "," in field:
            return [int(x) for x in field.split(",")]

        return [int(field)]

    valid_minutes = parse_field(minute, 0, 59)
    valid_hours = parse_field(hour, 0, 23)
    valid_days = parse_field(day_of_month, 1, 31)
    valid_months = parse_field(month, 1, 12)
    valid_dow = parse_field(day_of_week, 0, 6)

    # Start searching from the next minute
    candidate = from_time.replace(second=0, microsecond=0) + timedelta(minutes=1)

    # Search up to 1 year ahead
    max_search = candidate + timedelta(days=366)

    while candidate < max_search:
        if (
            candidate.month in valid_months
            and candidate.day in valid_days
            and candidate.weekday() in [(d - 1) % 7 for d in valid_dow]  # Convert Sunday=0 to Monday=0
            and candidate.hour in valid_hours
            and candidate.minute in valid_minutes
        ):
            return candidate

        candidate += timedelta(minutes=1)

    return None


def calculate_previous_runs(
    cron_expression: str,
    from_time: datetime | None = None,
    count: int = 5
) -> list[datetime]:
    """
    Calculate previous run times based on cron expression.

    For production, use croniter:
        cron = croniter(cron_expression, from_time)
        return [cron.get_prev(datetime) for _ in range(count)]
    """
    # Simplified: return empty list, production would use croniter
    return []


# ============================================================================
# Request/Response Models
# ============================================================================

class ScheduleCreateRequest(BaseModel):
    """Request to create a new schedule."""
    project_id: str = Field(..., description="Project ID to associate the schedule with")
    name: str = Field(..., min_length=1, max_length=100, description="Schedule name")
    cron_expression: str = Field(..., description="Cron expression (5 fields: min hour day month dow)")
    test_ids: list[str] | None = Field(None, description="Specific test IDs to run (None = all tests)")
    app_url: str = Field(..., description="Application URL to test")
    enabled: bool = Field(True, description="Whether the schedule is enabled")
    notify_on_failure: bool = Field(True, description="Send notifications on test failures")
    notification_channels: dict | None = Field(
        default_factory=lambda: {"email": True, "slack": False},
        description="Notification channel settings"
    )
    description: str | None = Field(None, max_length=500, description="Schedule description")
    timeout_minutes: int = Field(60, ge=5, le=480, description="Maximum run duration in minutes")
    retry_count: int = Field(0, ge=0, le=3, description="Number of retries on failure")
    environment_variables: dict | None = Field(None, description="Environment variables for test runs")
    tags: list[str] | None = Field(None, description="Tags for categorization")

    # AI Configuration
    auto_heal_enabled: bool = Field(False, description="Enable AI-powered auto-healing for failed tests")
    auto_heal_confidence_threshold: float = Field(
        0.9, ge=0.0, le=1.0,
        description="Minimum confidence threshold for applying auto-healing fixes"
    )
    quarantine_flaky_tests: bool = Field(False, description="Automatically quarantine flaky tests")
    flaky_threshold: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Failure rate threshold to mark a test as flaky"
    )

    @field_validator("cron_expression")
    @classmethod
    def validate_cron(cls, v: str) -> str:
        is_valid, error = validate_cron_expression(v)
        if not is_valid:
            raise ValueError(error)
        return v


class ScheduleUpdateRequest(BaseModel):
    """Request to update an existing schedule."""
    name: str | None = Field(None, min_length=1, max_length=100)
    cron_expression: str | None = None
    test_ids: list[str] | None = None
    app_url: str | None = None
    enabled: bool | None = None
    notify_on_failure: bool | None = None
    notification_channels: dict | None = None
    description: str | None = Field(None, max_length=500)
    timeout_minutes: int | None = Field(None, ge=5, le=480)
    retry_count: int | None = Field(None, ge=0, le=3)
    environment_variables: dict | None = None
    tags: list[str] | None = None

    # AI Configuration
    auto_heal_enabled: bool | None = Field(None, description="Enable AI-powered auto-healing for failed tests")
    auto_heal_confidence_threshold: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Minimum confidence threshold for applying auto-healing fixes"
    )
    quarantine_flaky_tests: bool | None = Field(None, description="Automatically quarantine flaky tests")
    flaky_threshold: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Failure rate threshold to mark a test as flaky"
    )

    @field_validator("cron_expression")
    @classmethod
    def validate_cron(cls, v: str | None) -> str | None:
        if v is not None:
            is_valid, error = validate_cron_expression(v)
            if not is_valid:
                raise ValueError(error)
        return v


class ScheduleResponse(BaseModel):
    """Schedule details response."""
    id: str
    project_id: str
    name: str
    cron_expression: str
    cron_readable: str  # Human-readable description
    test_ids: list[str] | None
    app_url: str
    enabled: bool
    status: Literal["active", "paused", "running", "error"]
    notify_on_failure: bool
    notification_channels: dict
    description: str | None
    timeout_minutes: int
    retry_count: int
    environment_variables: dict | None
    tags: list[str] | None
    next_run_at: str | None
    last_run_at: str | None
    last_run_status: Literal["pending", "queued", "running", "passed", "failed", "cancelled", "timeout", "success", "failure"] | None
    run_count: int
    success_count: int
    failure_count: int
    avg_duration_seconds: float | None
    created_at: str
    updated_at: str
    created_by: str | None

    # AI Configuration
    auto_heal_enabled: bool = False
    auto_heal_confidence_threshold: float = 0.9
    quarantine_flaky_tests: bool = False
    flaky_threshold: float = 0.3


class ScheduleRunResponse(BaseModel):
    """Schedule run history response."""
    id: str
    schedule_id: str
    status: Literal["pending", "queued", "running", "passed", "failed", "success", "failure", "cancelled", "timeout"]
    started_at: str
    completed_at: str | None
    duration_seconds: int | None
    trigger_type: Literal["scheduled", "manual", "webhook", "api"]
    triggered_by: str | None
    test_results: dict | None
    error_message: str | None
    retry_attempt: int
    logs_url: str | None

    # AI Analysis
    ai_analysis: dict | None = None
    is_flaky: bool = False
    flaky_score: float = 0.0
    failure_category: str | None = None
    failure_confidence: float | None = None

    # Auto-healing
    auto_healed: bool = False
    healing_details: dict | None = None


class ScheduleListResponse(BaseModel):
    """Response for listing schedules."""
    schedules: list[ScheduleResponse]
    total: int
    page: int
    per_page: int


class TriggerResponse(BaseModel):
    """Response for manual trigger."""
    success: bool
    message: str
    run_id: str
    schedule_id: str
    started_at: str


# ============================================================================
# Helper Functions
# ============================================================================

def cron_to_readable(cron_expression: str) -> str:
    """
    Convert cron expression to human-readable format.

    Examples:
        "0 0 * * *" -> "Daily at midnight"
        "0 9 * * 1-5" -> "Weekdays at 9:00 AM"
        "*/15 * * * *" -> "Every 15 minutes"
    """
    parts = cron_expression.strip().split()
    if len(parts) != 5:
        return cron_expression

    minute, hour, day, month, dow = parts

    # Common patterns
    if minute == "0" and hour == "0" and day == "*" and month == "*" and dow == "*":
        return "Daily at midnight"

    if minute == "0" and hour.isdigit() and day == "*" and month == "*" and dow == "*":
        h = int(hour)
        period = "AM" if h < 12 else "PM"
        h = h if h <= 12 else h - 12
        h = 12 if h == 0 else h
        return f"Daily at {h}:00 {period}"

    if minute == "0" and hour.isdigit() and day == "*" and month == "*" and dow == "1-5":
        h = int(hour)
        period = "AM" if h < 12 else "PM"
        h = h if h <= 12 else h - 12
        h = 12 if h == 0 else h
        return f"Weekdays at {h}:00 {period}"

    if minute.startswith("*/") and hour == "*" and day == "*" and month == "*" and dow == "*":
        interval = minute[2:]
        return f"Every {interval} minutes"

    if minute == "0" and hour.startswith("*/") and day == "*" and month == "*" and dow == "*":
        interval = hour[2:]
        return f"Every {interval} hours"

    if minute == "0" and hour == "0" and day == "1" and month == "*" and dow == "*":
        return "Monthly on the 1st at midnight"

    if minute == "0" and hour == "0" and day == "*" and month == "*" and dow == "0":
        return "Weekly on Sunday at midnight"

    if minute == "0" and hour == "0" and day == "*" and month == "*" and dow == "1":
        return "Weekly on Monday at midnight"

    # Default: return the raw expression
    return f"Cron: {cron_expression}"


def schedule_to_response(schedule: dict) -> ScheduleResponse:
    """Convert internal schedule dict to response model (sync version for in-memory)."""
    runs = schedule_runs.get(schedule["id"], [])
    return _build_schedule_response(schedule, runs)


async def schedule_to_response_async(schedule: dict) -> ScheduleResponse:
    """Convert internal schedule dict to response model (async version with DB lookup)."""
    runs = await _get_schedule_runs_from_db(schedule["id"], limit=100)
    return _build_schedule_response(schedule, runs)


def _build_schedule_response(schedule: dict, runs: list[dict]) -> ScheduleResponse:
    """Build ScheduleResponse from schedule dict and runs list."""
    # Calculate statistics
    success_count = sum(1 for r in runs if r.get("status") in ("success", "passed"))
    failure_count = sum(1 for r in runs if r.get("status") in ("failure", "failed"))

    durations = [r.get("duration_seconds") or r.get("duration_ms", 0) // 1000 for r in runs if r.get("duration_seconds") or r.get("duration_ms")]
    avg_duration = sum(durations) / len(durations) if durations else None

    last_run = runs[0] if runs else None  # Runs are ordered by triggered_at DESC

    # Calculate next run time
    next_run = schedule.get("next_run_at")
    if not next_run and schedule.get("enabled", True):
        next_run_dt = calculate_next_run(schedule["cron_expression"])
        next_run = next_run_dt.isoformat() if next_run_dt else None

    # Determine status
    status: Literal["active", "paused", "running", "error"] = "active"
    if not schedule.get("enabled", True):
        status = "paused"
    elif last_run and last_run.get("status") == "running":
        status = "running"
    elif last_run and last_run.get("status") in ("failure", "failed") and failure_count > success_count:
        status = "error"

    # Handle notification config (DB format vs API format)
    notification_config = schedule.get("notification_config", {})
    notify_on_failure = notification_config.get("on_failure", True) if isinstance(notification_config, dict) else True
    notification_channels = schedule.get("notification_channels", {"email": True, "slack": False})
    if isinstance(notification_config, dict) and "channels" in notification_config:
        notification_channels = {ch: True for ch in notification_config.get("channels", [])}

    # Handle timeout (DB stores ms, API uses minutes)
    timeout_ms = schedule.get("timeout_ms", 3600000)
    timeout_minutes = schedule.get("timeout_minutes", timeout_ms // 60000)

    # Handle app_url (DB uses app_url_override)
    app_url = schedule.get("app_url") or schedule.get("app_url_override", "")

    return ScheduleResponse(
        id=schedule["id"],
        project_id=schedule["project_id"],
        name=schedule["name"],
        cron_expression=schedule["cron_expression"],
        cron_readable=cron_to_readable(schedule["cron_expression"]),
        test_ids=schedule.get("test_ids"),
        app_url=app_url,
        enabled=schedule.get("enabled", True),
        status=status,
        notify_on_failure=notify_on_failure,
        notification_channels=notification_channels,
        description=schedule.get("description"),
        timeout_minutes=timeout_minutes,
        retry_count=schedule.get("retry_count", 0),
        environment_variables=schedule.get("environment_variables"),
        tags=schedule.get("tags"),
        next_run_at=next_run,
        last_run_at=last_run.get("started_at") or last_run.get("triggered_at") if last_run else None,
        last_run_status=last_run.get("status") if last_run else None,
        run_count=schedule.get("run_count", len(runs)),
        success_count=success_count,
        failure_count=schedule.get("failure_count", failure_count),
        avg_duration_seconds=round(avg_duration, 2) if avg_duration else None,
        created_at=schedule["created_at"],
        updated_at=schedule.get("updated_at", schedule["created_at"]),
        created_by=schedule.get("created_by"),
        # AI Configuration
        auto_heal_enabled=schedule.get("auto_heal_enabled", False),
        auto_heal_confidence_threshold=schedule.get("auto_heal_confidence_threshold", 0.9),
        quarantine_flaky_tests=schedule.get("quarantine_flaky_tests", False),
        flaky_threshold=schedule.get("flaky_threshold", 0.3),
    )


async def run_scheduled_tests(schedule_id: str, run_id: str, triggered_by: str | None = None):
    """
    Background task to run scheduled tests.

    Executes tests using Selenium Grid and:
    1. Streams progress events via SSE queue
    2. Updates run status in real-time
    3. Sends Slack notifications on failure
    """
    from src.api.schedule_executor import execute_scheduled_run

    schedule = await _get_schedule_from_db(schedule_id)
    if not schedule:
        logger.error("Schedule not found", schedule_id=schedule_id)
        return

    # Create events queue for SSE streaming
    events_queue: asyncio.Queue = asyncio.Queue()
    queue_key = f"{schedule_id}:{run_id}"
    _active_run_queues[queue_key] = events_queue

    try:
        # Update run status to running
        await _update_schedule_run_in_db(run_id, {"status": "running"})
        logger.info("Starting scheduled test run", schedule_id=schedule_id, run_id=run_id)

        # Execute tests using the schedule executor
        results = await execute_scheduled_run(
            schedule_id=schedule_id,
            run_id=run_id,
            schedule=schedule,
            events_queue=events_queue,
        )

        # Determine final status
        completed_at = datetime.now(UTC)
        final_status = "passed" if results["tests_failed"] == 0 else "failed"

        # =====================================================================
        # AGGREGATE AI ANALYSIS FROM TEST RESULTS (with error handling)
        # =====================================================================
        ai_update_fields = {}
        try:
            test_results = results.get("test_results", [])
            if test_results and isinstance(test_results, list):
                for tr in test_results:
                    if isinstance(tr, dict) and tr.get("ai_analysis"):
                        ai_analysis = tr["ai_analysis"]
                        ai_update_fields["ai_analysis"] = ai_analysis
                        ai_update_fields["failure_category"] = ai_analysis.get("category")
                        ai_update_fields["failure_confidence"] = ai_analysis.get("confidence")
                        ai_update_fields["is_flaky"] = ai_analysis.get("is_flaky", False)
                        break
                    if isinstance(tr, dict) and tr.get("auto_healed"):
                        ai_update_fields["auto_healed"] = True
                        ai_update_fields["healing_details"] = tr.get("healing_details")

                flaky_tests = results.get("flaky_tests", [])
                if flaky_tests and isinstance(flaky_tests, list):
                    ai_update_fields["is_flaky"] = True
                    ai_update_fields["flaky_score"] = max(
                        (ft.get("flaky_score", 0.0) for ft in flaky_tests if isinstance(ft, dict)),
                        default=0.0
                    )
        except Exception as e:
            logger.warning("Failed to aggregate AI analysis", error=str(e))

        # Update run with results
        await _update_schedule_run_in_db(run_id, {
            "status": final_status,
            "completed_at": completed_at.isoformat(),
            "duration_ms": results["duration_ms"],
            "tests_total": results["tests_total"],
            "tests_passed": results["tests_passed"],
            "tests_failed": results["tests_failed"],
            "tests_skipped": results["tests_skipped"],
            **ai_update_fields,  # Include AI fields if aggregation succeeded
        })

        # Update schedule statistics
        await _update_schedule_in_db(schedule_id, {
            "last_run_at": completed_at.isoformat(),
            "run_count": schedule.get("run_count", 0) + 1,
            "failure_count": schedule.get("failure_count", 0) + (1 if results["tests_failed"] > 0 else 0),
        })

        logger.info(
            "Scheduled test run completed",
            schedule_id=schedule_id,
            run_id=run_id,
            status=final_status,
            tests_passed=results["tests_passed"],
            tests_failed=results["tests_failed"],
            duration_ms=results["duration_ms"],
        )

        # Send Slack notification on failure
        if results["tests_failed"] > 0:
            await _send_failure_notification(schedule, run_id, results)

    except Exception as e:
        logger.exception("Scheduled test run failed", schedule_id=schedule_id, run_id=run_id, error=str(e))

        await _update_schedule_run_in_db(run_id, {
            "status": "failed",
            "completed_at": datetime.now(UTC).isoformat(),
            "error_message": str(e),
        })

        # Send notification for exception
        await _send_failure_notification(schedule, run_id, {
            "tests_total": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "duration_ms": 0,
            "failures": [{"error": str(e)}],
        })

    finally:
        # Clean up the events queue
        if queue_key in _active_run_queues:
            del _active_run_queues[queue_key]


async def _send_failure_notification(schedule: dict, run_id: str, results: dict) -> None:
    """Send AI-enhanced Slack notification for failed test run.

    If `ai_analysis` is present in results, creates a rich notification with:
    - Root cause category and confidence
    - Human-readable summary
    - Suggested fix
    - Flakiness indicator

    Otherwise falls back to basic failure notification.
    """
    try:
        notification_config = schedule.get("notification_config", {})
        notify_on_failure = notification_config.get("on_failure", True) if isinstance(notification_config, dict) else True

        if not notify_on_failure:
            return

        # Check if Slack is configured
        import os
        if not os.environ.get("SLACK_WEBHOOK_URL"):
            logger.debug("Slack webhook not configured, skipping notification")
            return

        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration()
        schedule_name = schedule.get("name", "Unknown")
        tests_failed = results.get("tests_failed", 0)
        tests_passed = results.get("tests_passed", 0)

        # Check for AI analysis in results
        ai_analysis = results.get("ai_analysis")

        if ai_analysis:
            # Build rich AI-enhanced notification blocks
            blocks = _build_ai_notification_blocks(
                schedule_name=schedule_name,
                schedule_id=schedule["id"],
                run_id=run_id,
                tests_failed=tests_failed,
                tests_passed=tests_passed,
                ai_analysis=ai_analysis,
                failures=results.get("failures", []),
            )

            # Send using blocks directly
            await slack._send_message(
                text=f"Schedule Failed: {schedule_name} - {tests_failed} failure(s)",
                blocks=blocks,
            )
        else:
            # Fall back to basic notification
            from src.integrations.slack_integration import TestSummary

            summary = TestSummary(
                total=results.get("tests_total", 0),
                passed=tests_passed,
                failed=tests_failed,
                skipped=results.get("tests_skipped", 0),
                duration_seconds=results.get("duration_ms", 0) / 1000,
                cost_usd=0,
                failures=[
                    {"name": f.get("test_name", "Unknown"), "error": f.get("error", "Unknown error")}
                    for f in results.get("failures", [])
                ],
                report_url=f"/schedules/{schedule['id']}/runs/{run_id}",
            )

            await slack.send_test_results(
                summary,
                title=f"Schedule Failed: {schedule_name}"
            )

        logger.info("Sent Slack failure notification", schedule_id=schedule.get("id"), run_id=run_id)

    except Exception as e:
        # Don't fail the run if notification fails
        logger.warning("Failed to send Slack notification", error=str(e))


def _build_ai_notification_blocks(
    schedule_name: str,
    schedule_id: str,
    run_id: str,
    tests_failed: int,
    tests_passed: int,
    ai_analysis: dict,
    failures: list[dict],
) -> list[dict]:
    """Build rich Slack blocks with AI analysis.

    Args:
        schedule_name: Name of the schedule
        schedule_id: ID of the schedule
        run_id: ID of the run
        tests_failed: Number of failed tests
        tests_passed: Number of passed tests
        ai_analysis: Dict with category, confidence, summary, suggested_fix, is_flaky
        failures: List of failure dicts with test_name and error

    Returns:
        List of Slack block objects
    """
    # Extract AI analysis fields with defaults
    category = ai_analysis.get("category", "UNKNOWN")
    confidence = ai_analysis.get("confidence", 0)
    summary = ai_analysis.get("summary", "No analysis available")
    suggested_fix = ai_analysis.get("suggested_fix")
    is_flaky = ai_analysis.get("is_flaky", False)
    recent_failure_count = ai_analysis.get("recent_failure_count", 0)

    # Build failure count text
    failure_text = f"{tests_failed} Failure" if tests_failed == 1 else f"{tests_failed} Failures"

    blocks = [
        # Header
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f":x: {schedule_name} - {failure_text}",
                "emoji": True,
            }
        },
        # Root cause section
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":mag: *Root Cause:* {category} ({confidence}% confidence)\n{summary}"
            }
        },
    ]

    # Add suggested fix if available
    if suggested_fix:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":bulb: *Suggested Fix:* {suggested_fix}"
            }
        })

    # Add flakiness warning if applicable
    if is_flaky or recent_failure_count >= 3:
        flaky_text = f":bar_chart: *Pattern:* This test has failed {recent_failure_count} times recently"
        if is_flaky:
            flaky_text += ", may be flaky."
        else:
            flaky_text += "."

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": flaky_text
            }
        })

    # Add divider before failure details
    if failures:
        blocks.append({"type": "divider"})

        # Show up to 3 failures with details
        for failure in failures[:3]:
            test_name = failure.get("test_name", "Unknown test")
            error = failure.get("error", "Unknown error")[:150]  # Truncate long errors

            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":small_red_triangle: *{test_name}*\n```{error}```"
                }
            })

        # Note if there are more failures
        if len(failures) > 3:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"_...and {len(failures) - 3} more failure(s)_"
                    }
                ]
            })

    # Add action button to view full report
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "View Full Report",
                    "emoji": True,
                },
                "url": f"/schedules/{schedule_id}/runs/{run_id}",
            }
        ]
    })

    # Footer with timestamp
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f":robot_face: AI-analyzed by E2E Testing Agent | {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"
            }
        ]
    })

    return blocks


# ============================================================================
# Schedule CRUD Endpoints
# ============================================================================

@router.post("", response_model=ScheduleResponse, status_code=201)
async def create_schedule(body: ScheduleCreateRequest, request: Request):
    """
    Create a new test schedule.

    The schedule will automatically run tests based on the cron expression.
    Use standard 5-field cron format: minute hour day month day-of-week

    Examples:
    - "0 9 * * *" - Daily at 9:00 AM
    - "0 */6 * * *" - Every 6 hours
    - "0 9 * * 1-5" - Weekdays at 9:00 AM
    - "*/30 * * * *" - Every 30 minutes
    """
    schedule_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    now_iso = now.isoformat()

    # Get user from request headers (set by auth middleware)
    user_id = request.headers.get("x-user-id")

    # Calculate next run time
    next_run = calculate_next_run(body.cron_expression, now)

    schedule = {
        "id": schedule_id,
        "project_id": body.project_id,
        "name": body.name,
        "cron_expression": body.cron_expression,
        "test_ids": body.test_ids or [],
        "test_filter": {},
        "timezone": "UTC",
        "enabled": body.enabled,
        "is_recurring": True,
        "next_run_at": next_run.isoformat() if next_run else None,
        "last_run_at": None,
        "run_count": 0,
        "failure_count": 0,
        "success_rate": 0,
        "notification_config": {
            "on_failure": body.notify_on_failure,
            "on_success": False,
            "channels": list(k for k, v in (body.notification_channels or {}).items() if v)
        },
        "max_parallel_tests": 5,
        "timeout_ms": body.timeout_minutes * 60 * 1000,
        "retry_failed_tests": body.retry_count > 0,
        "retry_count": body.retry_count,
        "environment": "staging",
        "browser": "chromium",
        "app_url_override": body.app_url,
        "created_by": user_id,
        "created_at": now_iso,
        "updated_at": now_iso,
        # Fields for backward compatibility with response model
        "description": body.description,
        "tags": body.tags,
        "environment_variables": body.environment_variables,
        # AI Configuration
        "auto_heal_enabled": body.auto_heal_enabled,
        "auto_heal_confidence_threshold": body.auto_heal_confidence_threshold,
        "quarantine_flaky_tests": body.quarantine_flaky_tests,
        "flaky_threshold": body.flaky_threshold,
    }

    # Save to database
    success = await _save_schedule_to_db(schedule)
    if not success:
        logger.error(
            "Failed to persist schedule to database",
            schedule_id=schedule_id,
            name=body.name,
            project_id=body.project_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to save schedule to database. Please try again or contact support."
        )

    logger.info(
        "Schedule created",
        schedule_id=schedule_id,
        name=body.name,
        cron=body.cron_expression,
        project_id=body.project_id,
        persistent=is_supabase_configured(),
    )

    return await schedule_to_response_async(schedule)


@router.get("", response_model=ScheduleListResponse)
async def list_schedules(
    project_id: str | None = Query(None, description="Filter by project ID"),
    enabled: bool | None = Query(None, description="Filter by enabled status"),
    tags: str | None = Query(None, description="Filter by tags (comma-separated)"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """
    List all schedules with optional filtering.

    Supports pagination and filtering by project, status, and tags.
    """
    supabase = await get_supabase()

    if supabase:
        # Build filters for Supabase query
        filters = {}
        if project_id:
            filters["project_id"] = project_id
        if enabled is not None:
            filters["enabled"] = enabled

        # Get schedules from Supabase
        all_schedules = await supabase.select(
            "test_schedules",
            columns="*",
            filters=filters if filters else None,
            order_by="created_at",
            ascending=False,
            limit=per_page * page  # Get enough for pagination
        )

        # Apply tag filter in Python (JSONB array filtering is complex in Supabase)
        if tags:
            tag_list = [t.strip() for t in tags.split(",")]
            all_schedules = [
                s for s in all_schedules
                if s.get("tags") and any(t in s["tags"] for t in tag_list)
            ]

        filtered = all_schedules
    else:
        # Fallback to in-memory
        filtered = list(schedules.values())

        if project_id:
            filtered = [s for s in filtered if s.get("project_id") == project_id]

        if enabled is not None:
            filtered = [s for s in filtered if s.get("enabled", True) == enabled]

        if tags:
            tag_list = [t.strip() for t in tags.split(",")]
            filtered = [
                s for s in filtered
                if s.get("tags") and any(t in s["tags"] for t in tag_list)
            ]

        filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Paginate
    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    paginated = filtered[start:end]

    # Build responses
    responses = []
    for s in paginated:
        responses.append(await schedule_to_response_async(s))

    return ScheduleListResponse(
        schedules=responses,
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{schedule_id}", response_model=ScheduleResponse)
async def get_schedule(schedule_id: str):
    """
    Get schedule details by ID.

    Returns full schedule configuration and statistics.
    """
    schedule = await _get_schedule_from_db(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    return await schedule_to_response_async(schedule)


@router.patch("/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(schedule_id: str, body: ScheduleUpdateRequest):
    """
    Update an existing schedule.

    Only provided fields will be updated. Use this to:
    - Enable/disable the schedule
    - Change the cron expression
    - Update notification settings
    """
    schedule = await _get_schedule_from_db(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # Update only provided fields
    update_data = body.model_dump(exclude_unset=True)
    update_data["updated_at"] = datetime.now(UTC).isoformat()

    # Recalculate next_run if cron changed
    if "cron_expression" in update_data:
        next_run = calculate_next_run(update_data["cron_expression"])
        update_data["next_run_at"] = next_run.isoformat() if next_run else None

    # Update in database
    success = await _update_schedule_in_db(schedule_id, update_data)
    if not success:
        logger.error(
            "Failed to update schedule in database",
            schedule_id=schedule_id,
            updates=list(update_data.keys()),
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to update schedule in database. Please try again or contact support."
        )

    # Merge updates for response
    schedule.update(update_data)

    logger.info(
        "Schedule updated",
        schedule_id=schedule_id,
        updates=list(update_data.keys()),
    )

    return await schedule_to_response_async(schedule)


@router.delete("/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """
    Delete a schedule.

    This will stop all future runs. Run history is preserved.
    """
    schedule = await _get_schedule_from_db(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    schedule_name = schedule.get("name")
    success = await _delete_schedule_from_db(schedule_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete schedule")

    logger.info(
        "Schedule deleted",
        schedule_id=schedule_id,
        name=schedule_name,
    )

    return {
        "success": True,
        "message": "Schedule deleted successfully",
        "schedule_id": schedule_id,
    }


# ============================================================================
# Schedule Run Endpoints
# ============================================================================

@router.post("/{schedule_id}/trigger", response_model=TriggerResponse)
async def trigger_schedule(
    schedule_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Manually trigger a scheduled run.

    This immediately starts a test run regardless of the schedule.
    Useful for testing schedule configuration or running on-demand.
    """
    schedule = await _get_schedule_from_db(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # Get user from request headers
    user_id = request.headers.get("x-user-id")
    user_email = request.headers.get("x-user-email")
    triggered_by = user_email or user_id or "api"

    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    run = {
        "id": run_id,
        "schedule_id": schedule_id,
        "status": "pending",
        "triggered_at": now.isoformat(),
        "started_at": now.isoformat(),
        "completed_at": None,
        "duration_ms": None,
        "trigger_type": "manual",
        "triggered_by": triggered_by,
        "tests_total": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
        "error_message": None,
        "error_details": None,
        "logs": [],
        "metadata": {},
        "created_at": now.isoformat(),
    }

    # Save run to database
    await _save_schedule_run_to_db(run)

    # Run tests in background
    background_tasks.add_task(run_scheduled_tests, schedule_id, run_id, triggered_by)

    logger.info(
        "Schedule manually triggered",
        schedule_id=schedule_id,
        run_id=run_id,
        triggered_by=triggered_by,
    )

    return TriggerResponse(
        success=True,
        message="Test run started",
        run_id=run_id,
        schedule_id=schedule_id,
        started_at=now.isoformat(),
    )


@router.get("/{schedule_id}/runs", response_model=list[ScheduleRunResponse])
async def get_schedule_runs(
    schedule_id: str,
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum runs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    Get run history for a schedule.

    Returns recent runs with their status, duration, and results.
    """
    schedule = await _get_schedule_from_db(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    runs = await _get_schedule_runs_from_db(schedule_id, status=status, limit=limit, offset=offset)

    return [
        ScheduleRunResponse(
            id=r["id"],
            schedule_id=r["schedule_id"],
            status=r.get("status", "pending"),
            started_at=r.get("started_at") or r.get("triggered_at", ""),
            completed_at=r.get("completed_at"),
            duration_seconds=r.get("duration_seconds") or (r.get("duration_ms", 0) // 1000 if r.get("duration_ms") else None),
            trigger_type=r.get("trigger_type", "scheduled"),
            triggered_by=r.get("triggered_by"),
            test_results={
                "total": r.get("tests_total", 0),
                "passed": r.get("tests_passed", 0),
                "failed": r.get("tests_failed", 0),
                "skipped": r.get("tests_skipped", 0),
            } if r.get("tests_total") else r.get("test_results"),
            error_message=r.get("error_message"),
            retry_attempt=r.get("retry_attempt", 0),
            logs_url=r.get("logs_url"),
            # AI Analysis
            ai_analysis=r.get("ai_analysis"),
            is_flaky=r.get("is_flaky", False),
            flaky_score=r.get("flaky_score", 0.0),
            failure_category=r.get("failure_category"),
            failure_confidence=r.get("failure_confidence"),
            # Auto-healing
            auto_healed=r.get("auto_healed", False),
            healing_details=r.get("healing_details"),
        )
        for r in runs
    ]


@router.get("/{schedule_id}/runs/{run_id}", response_model=ScheduleRunResponse)
async def get_schedule_run(schedule_id: str, run_id: str):
    """
    Get details for a specific run.

    Returns full run details including test results and logs.
    """
    if schedule_id not in schedules:
        raise HTTPException(status_code=404, detail="Schedule not found")

    runs = schedule_runs.get(schedule_id, [])
    run = next((r for r in runs if r["id"] == run_id), None)

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return ScheduleRunResponse(
        id=run["id"],
        schedule_id=run["schedule_id"],
        status=run.get("status", "pending"),
        started_at=run["started_at"],
        completed_at=run.get("completed_at"),
        duration_seconds=run.get("duration_seconds"),
        trigger_type=run.get("trigger_type", "scheduled"),
        triggered_by=run.get("triggered_by"),
        test_results=run.get("test_results"),
        error_message=run.get("error_message"),
        retry_attempt=run.get("retry_attempt", 0),
        logs_url=run.get("logs_url"),
        # AI Analysis
        ai_analysis=run.get("ai_analysis"),
        is_flaky=run.get("is_flaky", False),
        flaky_score=run.get("flaky_score", 0.0),
        failure_category=run.get("failure_category"),
        failure_confidence=run.get("failure_confidence"),
        # Auto-healing
        auto_healed=run.get("auto_healed", False),
        healing_details=run.get("healing_details"),
    )


@router.post("/{schedule_id}/runs/{run_id}/cancel")
async def cancel_run(schedule_id: str, run_id: str):
    """
    Cancel a running test run.

    Only runs in 'pending' or 'running' status can be cancelled.
    """
    if schedule_id not in schedules:
        raise HTTPException(status_code=404, detail="Schedule not found")

    runs = schedule_runs.get(schedule_id, [])
    run = next((r for r in runs if r["id"] == run_id), None)

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.get("status") not in ("pending", "running"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel run with status '{run.get('status')}'"
        )

    run["status"] = "cancelled"
    run["completed_at"] = datetime.now(UTC).isoformat()

    logger.info("Run cancelled", schedule_id=schedule_id, run_id=run_id)

    return {
        "success": True,
        "message": "Run cancelled",
        "run_id": run_id,
    }


@router.get("/{schedule_id}/runs/{run_id}/stream")
async def stream_schedule_run(schedule_id: str, run_id: str):
    """
    Stream run progress via Server-Sent Events.

    Events emitted:
    - run_started: When the run begins
    - tests_fetched: When tests are loaded (includes test count)
    - test_started: When a test begins execution
    - step_started: When a step begins
    - step_completed: When a step completes (includes success/failure)
    - test_completed: When a test completes (includes results)
    - progress: Periodic progress updates (includes percent complete)
    - run_completed: When the entire run finishes
    - error: If an error occurs
    - heartbeat: Periodic keepalive

    Connection will close when:
    - Run completes (run_completed event)
    - Run is cancelled
    - Client disconnects
    - Timeout (5 minutes)
    """
    import json

    schedule = await _get_schedule_from_db(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    queue_key = f"{schedule_id}:{run_id}"

    async def event_generator():
        """Generate SSE events from the run queue."""
        heartbeat_interval = 15  # seconds
        last_heartbeat = asyncio.get_event_loop().time()
        timeout = 300  # 5 minutes max connection

        start_time = asyncio.get_event_loop().time()

        while True:
            current_time = asyncio.get_event_loop().time()

            # Check timeout
            if current_time - start_time > timeout:
                yield f"data: {json.dumps({'type': 'timeout', 'message': 'Stream timeout'})}\n\n"
                break

            # Send heartbeat if needed
            if current_time - last_heartbeat > heartbeat_interval:
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now(UTC).isoformat()})}\n\n"
                last_heartbeat = current_time

            # Check if queue exists
            if queue_key not in _active_run_queues:
                # Run might have completed before we connected, or not started yet
                # Check run status from DB
                runs = await _get_schedule_runs_from_db(schedule_id, limit=1)
                run = next((r for r in runs if r["id"] == run_id), None)

                if run and run.get("status") in ("passed", "failed", "cancelled"):
                    # Run already completed
                    yield f"data: {json.dumps({'type': 'run_already_completed', 'status': run.get('status'), 'tests_passed': run.get('tests_passed', 0), 'tests_failed': run.get('tests_failed', 0)})}\n\n"
                    break

                # Wait a bit for the queue to be created
                await asyncio.sleep(1)
                continue

            # Try to get an event from the queue
            try:
                queue = _active_run_queues[queue_key]
                event = await asyncio.wait_for(queue.get(), timeout=1.0)

                yield f"data: {json.dumps(event)}\n\n"

                # Check for terminal events
                if event.get("type") in ("run_completed", "error"):
                    break

            except asyncio.TimeoutError:
                # No event available, continue loop
                continue
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# ============================================================================
# AI Analysis Endpoints
# ============================================================================


class FlakyTestResponse(BaseModel):
    """Response for a flaky test."""
    test_id: str
    test_name: str
    flaky_score: float
    total_runs: int
    passed_runs: int
    failed_runs: int
    failure_rate: float
    last_failure_at: str | None
    last_success_at: str | None
    is_quarantined: bool = False
    failure_categories: list[str] = []
    recommended_action: str | None = None


@router.get("/{schedule_id}/flaky-tests", response_model=list[FlakyTestResponse])
async def get_flaky_tests(
    schedule_id: str,
    min_runs: int = Query(5, ge=1, description="Minimum number of runs to consider a test"),
    min_flaky_score: float = Query(0.1, ge=0.0, le=1.0, description="Minimum flaky score threshold"),
    include_quarantined: bool = Query(True, description="Include quarantined tests"),
):
    """
    Get list of flaky tests for this schedule.

    Analyzes test run history to identify tests with inconsistent results.
    A test is considered flaky if it alternates between passing and failing
    across multiple runs.

    The flaky score is calculated based on:
    - Failure rate (tests that always pass or always fail are not flaky)
    - Alternation frequency (how often the result changes between runs)
    - Recent trends (more weight given to recent runs)

    Returns tests sorted by flaky score (highest first).
    """
    schedule = await _get_schedule_from_db(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # Get all runs for analysis
    runs = await _get_schedule_runs_from_db(schedule_id, limit=100)

    if not runs:
        return []

    # Aggregate test results across runs
    test_stats: dict[str, dict] = {}

    for run in runs:
        # Get test results from run (may be in different formats)
        test_results = run.get("test_results") or {}
        individual_results = test_results.get("results", [])

        # Also check for flaky annotations on the run itself
        run_is_flaky = run.get("is_flaky", False)
        run_flaky_score = run.get("flaky_score", 0.0)

        for result in individual_results:
            test_id = result.get("test_id", result.get("id", "unknown"))
            test_name = result.get("test_name", result.get("name", test_id))
            status = result.get("status", "unknown")

            if test_id not in test_stats:
                test_stats[test_id] = {
                    "test_id": test_id,
                    "test_name": test_name,
                    "runs": [],
                    "passed": 0,
                    "failed": 0,
                    "last_failure_at": None,
                    "last_success_at": None,
                    "failure_categories": set(),
                    "is_quarantined": result.get("is_quarantined", False),
                }

            stats = test_stats[test_id]
            run_time = run.get("completed_at") or run.get("started_at")

            if status in ("passed", "success"):
                stats["passed"] += 1
                stats["runs"].append(1)  # 1 = pass
                if run_time:
                    stats["last_success_at"] = run_time
            elif status in ("failed", "failure"):
                stats["failed"] += 1
                stats["runs"].append(0)  # 0 = fail
                if run_time:
                    stats["last_failure_at"] = run_time
                # Track failure categories
                if result.get("failure_category"):
                    stats["failure_categories"].add(result["failure_category"])

    # Calculate flaky scores
    flaky_tests = []

    for test_id, stats in test_stats.items():
        total_runs = stats["passed"] + stats["failed"]

        if total_runs < min_runs:
            continue

        # Calculate failure rate
        failure_rate = stats["failed"] / total_runs if total_runs > 0 else 0

        # Calculate flaky score
        # A test is flaky if it sometimes passes and sometimes fails
        # Pure pass (0% failure) or pure fail (100% failure) = not flaky
        # 50% failure rate = maximum flakiness potential

        # Base flakiness from failure rate (peaks at 50%)
        base_flakiness = 1 - abs(2 * failure_rate - 1)

        # Calculate alternation score (how often does result change?)
        runs_sequence = stats["runs"]
        alternations = 0
        for i in range(1, len(runs_sequence)):
            if runs_sequence[i] != runs_sequence[i - 1]:
                alternations += 1

        max_alternations = len(runs_sequence) - 1 if len(runs_sequence) > 1 else 1
        alternation_score = alternations / max_alternations if max_alternations > 0 else 0

        # Combined flaky score (weighted average)
        flaky_score = (base_flakiness * 0.4) + (alternation_score * 0.6)

        if flaky_score < min_flaky_score:
            continue

        if not include_quarantined and stats["is_quarantined"]:
            continue

        # Determine recommended action
        recommended_action = None
        if flaky_score >= 0.7:
            recommended_action = "quarantine"
        elif flaky_score >= 0.4:
            recommended_action = "investigate"
        elif flaky_score >= 0.2:
            recommended_action = "monitor"

        flaky_tests.append(FlakyTestResponse(
            test_id=test_id,
            test_name=stats["test_name"],
            flaky_score=round(flaky_score, 3),
            total_runs=total_runs,
            passed_runs=stats["passed"],
            failed_runs=stats["failed"],
            failure_rate=round(failure_rate, 3),
            last_failure_at=stats["last_failure_at"],
            last_success_at=stats["last_success_at"],
            is_quarantined=stats["is_quarantined"],
            failure_categories=list(stats["failure_categories"]),
            recommended_action=recommended_action,
        ))

    # Sort by flaky score (highest first)
    flaky_tests.sort(key=lambda x: x.flaky_score, reverse=True)

    return flaky_tests


@router.get("/{schedule_id}/ai-stats")
async def get_ai_stats(schedule_id: str):
    """
    Get AI analysis statistics for a schedule.

    Returns aggregated AI analysis data including:
    - Total auto-healed tests
    - Flaky test count and trends
    - Failure category distribution
    - Healing success rate
    """
    schedule = await _get_schedule_from_db(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # Get recent runs for analysis
    runs = await _get_schedule_runs_from_db(schedule_id, limit=100)

    if not runs:
        return {
            "total_runs": 0,
            "auto_healed_count": 0,
            "flaky_tests_detected": 0,
            "failure_categories": {},
            "healing_success_rate": None,
            "average_flaky_score": None,
            "ai_analysis_enabled": schedule.get("auto_heal_enabled", False),
        }

    # Aggregate stats
    auto_healed_count = 0
    healing_successes = 0
    healing_attempts = 0
    flaky_runs = 0
    total_flaky_score = 0.0
    failure_categories: dict[str, int] = {}

    for run in runs:
        if run.get("auto_healed"):
            auto_healed_count += 1
            healing_attempts += 1
            # Check if the healed run passed
            if run.get("status") in ("passed", "success"):
                healing_successes += 1

        if run.get("healing_details"):
            healing_attempts += 1

        if run.get("is_flaky"):
            flaky_runs += 1
            total_flaky_score += run.get("flaky_score", 0.0)

        if run.get("failure_category"):
            category = run["failure_category"]
            failure_categories[category] = failure_categories.get(category, 0) + 1

    return {
        "total_runs": len(runs),
        "auto_healed_count": auto_healed_count,
        "flaky_tests_detected": flaky_runs,
        "failure_categories": failure_categories,
        "healing_success_rate": round(healing_successes / healing_attempts, 3) if healing_attempts > 0 else None,
        "average_flaky_score": round(total_flaky_score / flaky_runs, 3) if flaky_runs > 0 else None,
        "ai_analysis_enabled": schedule.get("auto_heal_enabled", False),
        "quarantine_enabled": schedule.get("quarantine_flaky_tests", False),
        "flaky_threshold": schedule.get("flaky_threshold", 0.3),
        "auto_heal_confidence_threshold": schedule.get("auto_heal_confidence_threshold", 0.9),
    }


# ============================================================================
# Utility Endpoints
# ============================================================================

@router.post("/validate-cron")
async def validate_cron_endpoint(cron_expression: str = Query(..., description="Cron expression to validate")):
    """
    Validate a cron expression and get next run times.

    Returns validation result and next 5 scheduled run times.
    """
    is_valid, error = validate_cron_expression(cron_expression)

    if not is_valid:
        return {
            "valid": False,
            "error": error,
            "readable": None,
            "next_runs": [],
        }

    # Calculate next 5 run times
    next_runs = []
    current = datetime.now(UTC)
    for _ in range(5):
        next_run = calculate_next_run(cron_expression, current)
        if next_run:
            next_runs.append(next_run.isoformat())
            current = next_run
        else:
            break

    return {
        "valid": True,
        "error": None,
        "readable": cron_to_readable(cron_expression),
        "next_runs": next_runs,
    }


@router.get("/presets")
async def get_schedule_presets():
    """
    Get common schedule presets.

    Returns a list of commonly used cron expressions with descriptions.
    """
    return {
        "presets": [
            {
                "name": "Every 15 minutes",
                "cron": "*/15 * * * *",
                "description": "Run tests every 15 minutes",
            },
            {
                "name": "Every hour",
                "cron": "0 * * * *",
                "description": "Run tests at the start of every hour",
            },
            {
                "name": "Every 6 hours",
                "cron": "0 */6 * * *",
                "description": "Run tests every 6 hours (4 times daily)",
            },
            {
                "name": "Daily at midnight",
                "cron": "0 0 * * *",
                "description": "Run tests once daily at midnight UTC",
            },
            {
                "name": "Daily at 9 AM",
                "cron": "0 9 * * *",
                "description": "Run tests once daily at 9:00 AM UTC",
            },
            {
                "name": "Weekdays at 9 AM",
                "cron": "0 9 * * 1-5",
                "description": "Run tests Monday-Friday at 9:00 AM UTC",
            },
            {
                "name": "Weekly on Monday",
                "cron": "0 0 * * 1",
                "description": "Run tests every Monday at midnight UTC",
            },
            {
                "name": "Monthly on the 1st",
                "cron": "0 0 1 * *",
                "description": "Run tests on the first day of each month at midnight UTC",
            },
        ]
    }
