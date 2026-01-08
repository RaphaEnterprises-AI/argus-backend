"""Test Scheduling API endpoints.

Provides endpoints for:
- Creating and managing scheduled test runs
- Cron-style scheduling with croniter pattern
- Manual trigger of scheduled runs
- Run history and statistics
"""

import asyncio
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal

from fastapi import APIRouter, HTTPException, Request, Query, BackgroundTasks
from pydantic import BaseModel, Field, field_validator
import structlog

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/schedules", tags=["Scheduling"])


# ============================================================================
# In-Memory Storage (use database for production)
# ============================================================================

schedules: dict[str, dict] = {}
schedule_runs: dict[str, list[dict]] = {}  # schedule_id -> list of runs


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


def validate_cron_expression(cron_expression: str) -> tuple[bool, Optional[str]]:
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

    field_names = ["minute", "hour", "day_of_month", "month", "day_of_week"]
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


def calculate_next_run(cron_expression: str, from_time: Optional[datetime] = None) -> Optional[datetime]:
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
        from_time = datetime.now(timezone.utc)

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
    from_time: Optional[datetime] = None,
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
    test_ids: Optional[list[str]] = Field(None, description="Specific test IDs to run (None = all tests)")
    app_url: str = Field(..., description="Application URL to test")
    enabled: bool = Field(True, description="Whether the schedule is enabled")
    notify_on_failure: bool = Field(True, description="Send notifications on test failures")
    notification_channels: Optional[dict] = Field(
        default_factory=lambda: {"email": True, "slack": False},
        description="Notification channel settings"
    )
    description: Optional[str] = Field(None, max_length=500, description="Schedule description")
    timeout_minutes: int = Field(60, ge=5, le=480, description="Maximum run duration in minutes")
    retry_count: int = Field(0, ge=0, le=3, description="Number of retries on failure")
    environment_variables: Optional[dict] = Field(None, description="Environment variables for test runs")
    tags: Optional[list[str]] = Field(None, description="Tags for categorization")

    @field_validator("cron_expression")
    @classmethod
    def validate_cron(cls, v: str) -> str:
        is_valid, error = validate_cron_expression(v)
        if not is_valid:
            raise ValueError(error)
        return v


class ScheduleUpdateRequest(BaseModel):
    """Request to update an existing schedule."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    cron_expression: Optional[str] = None
    test_ids: Optional[list[str]] = None
    app_url: Optional[str] = None
    enabled: Optional[bool] = None
    notify_on_failure: Optional[bool] = None
    notification_channels: Optional[dict] = None
    description: Optional[str] = Field(None, max_length=500)
    timeout_minutes: Optional[int] = Field(None, ge=5, le=480)
    retry_count: Optional[int] = Field(None, ge=0, le=3)
    environment_variables: Optional[dict] = None
    tags: Optional[list[str]] = None

    @field_validator("cron_expression")
    @classmethod
    def validate_cron(cls, v: Optional[str]) -> Optional[str]:
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
    test_ids: Optional[list[str]]
    app_url: str
    enabled: bool
    status: Literal["active", "paused", "running", "error"]
    notify_on_failure: bool
    notification_channels: dict
    description: Optional[str]
    timeout_minutes: int
    retry_count: int
    environment_variables: Optional[dict]
    tags: Optional[list[str]]
    next_run_at: Optional[str]
    last_run_at: Optional[str]
    last_run_status: Optional[Literal["success", "failure", "running", "cancelled"]]
    run_count: int
    success_count: int
    failure_count: int
    avg_duration_seconds: Optional[float]
    created_at: str
    updated_at: str
    created_by: Optional[str]


class ScheduleRunResponse(BaseModel):
    """Schedule run history response."""
    id: str
    schedule_id: str
    status: Literal["pending", "running", "success", "failure", "cancelled", "timeout"]
    started_at: str
    completed_at: Optional[str]
    duration_seconds: Optional[int]
    trigger_type: Literal["scheduled", "manual"]
    triggered_by: Optional[str]
    test_results: Optional[dict]
    error_message: Optional[str]
    retry_attempt: int
    logs_url: Optional[str]


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
    """Convert internal schedule dict to response model."""
    runs = schedule_runs.get(schedule["id"], [])

    # Calculate statistics
    success_count = sum(1 for r in runs if r.get("status") == "success")
    failure_count = sum(1 for r in runs if r.get("status") == "failure")

    durations = [r.get("duration_seconds") for r in runs if r.get("duration_seconds")]
    avg_duration = sum(durations) / len(durations) if durations else None

    last_run = runs[-1] if runs else None

    # Calculate next run time
    next_run = None
    if schedule.get("enabled", True):
        next_run_dt = calculate_next_run(schedule["cron_expression"])
        next_run = next_run_dt.isoformat() if next_run_dt else None

    # Determine status
    status: Literal["active", "paused", "running", "error"] = "active"
    if not schedule.get("enabled", True):
        status = "paused"
    elif last_run and last_run.get("status") == "running":
        status = "running"
    elif last_run and last_run.get("status") == "failure" and failure_count > success_count:
        status = "error"

    return ScheduleResponse(
        id=schedule["id"],
        project_id=schedule["project_id"],
        name=schedule["name"],
        cron_expression=schedule["cron_expression"],
        cron_readable=cron_to_readable(schedule["cron_expression"]),
        test_ids=schedule.get("test_ids"),
        app_url=schedule["app_url"],
        enabled=schedule.get("enabled", True),
        status=status,
        notify_on_failure=schedule.get("notify_on_failure", True),
        notification_channels=schedule.get("notification_channels", {"email": True, "slack": False}),
        description=schedule.get("description"),
        timeout_minutes=schedule.get("timeout_minutes", 60),
        retry_count=schedule.get("retry_count", 0),
        environment_variables=schedule.get("environment_variables"),
        tags=schedule.get("tags"),
        next_run_at=next_run,
        last_run_at=last_run["started_at"] if last_run else None,
        last_run_status=last_run.get("status") if last_run else None,
        run_count=len(runs),
        success_count=success_count,
        failure_count=failure_count,
        avg_duration_seconds=round(avg_duration, 2) if avg_duration else None,
        created_at=schedule["created_at"],
        updated_at=schedule.get("updated_at", schedule["created_at"]),
        created_by=schedule.get("created_by"),
    )


async def run_scheduled_tests(schedule_id: str, run_id: str, triggered_by: Optional[str] = None):
    """
    Background task to run scheduled tests.

    In production, this would:
    1. Create a test run using the orchestrator
    2. Track progress and update run status
    3. Send notifications on completion
    """
    schedule = schedules.get(schedule_id)
    if not schedule:
        return

    runs = schedule_runs.setdefault(schedule_id, [])
    run = next((r for r in runs if r["id"] == run_id), None)
    if not run:
        return

    try:
        # Update run status to running
        run["status"] = "running"
        logger.info("Starting scheduled test run", schedule_id=schedule_id, run_id=run_id)

        # Simulate test execution (replace with actual orchestrator call in production)
        # In production:
        # from src.orchestrator.graph import TestingOrchestrator
        # orchestrator = TestingOrchestrator(
        #     codebase_path="...",
        #     app_url=schedule["app_url"],
        # )
        # result = await orchestrator.run(...)

        await asyncio.sleep(2)  # Simulate work

        # Update run with results
        completed_at = datetime.now(timezone.utc)
        run["status"] = "success"
        run["completed_at"] = completed_at.isoformat()
        run["duration_seconds"] = int((completed_at - datetime.fromisoformat(run["started_at"])).total_seconds())
        run["test_results"] = {
            "total": 10,
            "passed": 10,
            "failed": 0,
            "skipped": 0,
        }

        logger.info(
            "Scheduled test run completed",
            schedule_id=schedule_id,
            run_id=run_id,
            status="success",
            duration_seconds=run["duration_seconds"],
        )

    except Exception as e:
        logger.exception("Scheduled test run failed", schedule_id=schedule_id, run_id=run_id, error=str(e))
        run["status"] = "failure"
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        run["error_message"] = str(e)

        # Handle notifications on failure
        if schedule.get("notify_on_failure", True):
            # In production: send notifications via configured channels
            logger.info("Would send failure notification", schedule_id=schedule_id, run_id=run_id)


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
    now = datetime.now(timezone.utc).isoformat()

    # Get user from request headers (set by auth middleware)
    user_id = request.headers.get("x-user-id")

    schedule = {
        "id": schedule_id,
        "project_id": body.project_id,
        "name": body.name,
        "cron_expression": body.cron_expression,
        "test_ids": body.test_ids,
        "app_url": body.app_url,
        "enabled": body.enabled,
        "notify_on_failure": body.notify_on_failure,
        "notification_channels": body.notification_channels or {"email": True, "slack": False},
        "description": body.description,
        "timeout_minutes": body.timeout_minutes,
        "retry_count": body.retry_count,
        "environment_variables": body.environment_variables,
        "tags": body.tags,
        "created_at": now,
        "updated_at": now,
        "created_by": user_id,
    }

    schedules[schedule_id] = schedule
    schedule_runs[schedule_id] = []

    logger.info(
        "Schedule created",
        schedule_id=schedule_id,
        name=body.name,
        cron=body.cron_expression,
        project_id=body.project_id,
    )

    return schedule_to_response(schedule)


@router.get("", response_model=ScheduleListResponse)
async def list_schedules(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    enabled: Optional[bool] = Query(None, description="Filter by enabled status"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """
    List all schedules with optional filtering.

    Supports pagination and filtering by project, status, and tags.
    """
    filtered = list(schedules.values())

    # Apply filters
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

    # Sort by created_at descending
    filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Paginate
    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    paginated = filtered[start:end]

    return ScheduleListResponse(
        schedules=[schedule_to_response(s) for s in paginated],
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
    schedule = schedules.get(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    return schedule_to_response(schedule)


@router.patch("/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(schedule_id: str, body: ScheduleUpdateRequest):
    """
    Update an existing schedule.

    Only provided fields will be updated. Use this to:
    - Enable/disable the schedule
    - Change the cron expression
    - Update notification settings
    """
    schedule = schedules.get(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # Update only provided fields
    update_data = body.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        schedule[key] = value

    schedule["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Schedule updated",
        schedule_id=schedule_id,
        updates=list(update_data.keys()),
    )

    return schedule_to_response(schedule)


@router.delete("/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """
    Delete a schedule.

    This will stop all future runs. Run history is preserved.
    """
    if schedule_id not in schedules:
        raise HTTPException(status_code=404, detail="Schedule not found")

    schedule = schedules.pop(schedule_id)

    logger.info(
        "Schedule deleted",
        schedule_id=schedule_id,
        name=schedule.get("name"),
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
    schedule = schedules.get(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # Get user from request headers
    user_id = request.headers.get("x-user-id")
    user_email = request.headers.get("x-user-email")
    triggered_by = user_email or user_id or "api"

    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    run = {
        "id": run_id,
        "schedule_id": schedule_id,
        "status": "pending",
        "started_at": now.isoformat(),
        "completed_at": None,
        "duration_seconds": None,
        "trigger_type": "manual",
        "triggered_by": triggered_by,
        "test_results": None,
        "error_message": None,
        "retry_attempt": 0,
        "logs_url": None,
    }

    runs = schedule_runs.setdefault(schedule_id, [])
    runs.append(run)

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
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum runs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    Get run history for a schedule.

    Returns recent runs with their status, duration, and results.
    """
    if schedule_id not in schedules:
        raise HTTPException(status_code=404, detail="Schedule not found")

    runs = schedule_runs.get(schedule_id, [])

    # Filter by status if provided
    if status:
        runs = [r for r in runs if r.get("status") == status]

    # Sort by started_at descending (most recent first)
    runs = sorted(runs, key=lambda x: x.get("started_at", ""), reverse=True)

    # Apply pagination
    paginated = runs[offset:offset + limit]

    return [
        ScheduleRunResponse(
            id=r["id"],
            schedule_id=r["schedule_id"],
            status=r.get("status", "pending"),
            started_at=r["started_at"],
            completed_at=r.get("completed_at"),
            duration_seconds=r.get("duration_seconds"),
            trigger_type=r.get("trigger_type", "scheduled"),
            triggered_by=r.get("triggered_by"),
            test_results=r.get("test_results"),
            error_message=r.get("error_message"),
            retry_attempt=r.get("retry_attempt", 0),
            logs_url=r.get("logs_url"),
        )
        for r in paginated
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
    run["completed_at"] = datetime.now(timezone.utc).isoformat()

    logger.info("Run cancelled", schedule_id=schedule_id, run_id=run_id)

    return {
        "success": True,
        "message": "Run cancelled",
        "run_id": run_id,
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
    current = datetime.now(timezone.utc)
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
