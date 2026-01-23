"""Background Scheduler Daemon.

Automatically triggers scheduled test runs based on cron expressions.
Runs as a background asyncio task on server startup.
"""

import asyncio
import uuid
from datetime import UTC, datetime

import structlog

from src.integrations.supabase import get_supabase, is_supabase_configured

logger = structlog.get_logger()

# Global flag to control the scheduler
_scheduler_running = False
_scheduler_task: asyncio.Task | None = None


async def get_due_schedules() -> list[dict]:
    """
    Query schedules that are due to run.

    Returns schedules where:
    - enabled = true
    - next_run_at <= now()
    """
    supabase = await get_supabase()
    if not supabase:
        logger.debug("Supabase not configured, scheduler cannot run")
        return []

    try:
        now = datetime.now(UTC).isoformat()

        # Query for due schedules
        result = await supabase.select(
            "test_schedules",
            columns="*",
            filters={"enabled": True},
            order_by="next_run_at",
            ascending=True,
            limit=10,  # Process max 10 schedules per cycle
        )

        # Filter for schedules where next_run_at <= now
        due_schedules = []
        for schedule in result:
            next_run_at = schedule.get("next_run_at")
            if next_run_at and next_run_at <= now:
                due_schedules.append(schedule)

        return due_schedules

    except Exception as e:
        logger.exception("Failed to query due schedules", error=str(e))
        return []


async def calculate_next_run_time(cron_expression: str) -> str | None:
    """
    Calculate the next run time using croniter.

    Args:
        cron_expression: Standard 5-field cron expression

    Returns:
        ISO format datetime string of next run, or None if invalid
    """
    try:
        from croniter import croniter

        now = datetime.now(UTC)
        cron = croniter(cron_expression, now)
        next_run = cron.get_next(datetime)

        return next_run.isoformat()

    except ImportError:
        # Fallback to simple calculation if croniter not available
        logger.warning("croniter not installed, using fallback calculation")
        from src.api.scheduling import calculate_next_run
        next_run = calculate_next_run(cron_expression)
        return next_run.isoformat() if next_run else None

    except Exception as e:
        logger.error("Failed to calculate next run time", cron=cron_expression, error=str(e))
        return None


async def update_schedule_next_run(schedule_id: str, cron_expression: str) -> bool:
    """
    Update the next_run_at field for a schedule.

    Args:
        schedule_id: ID of the schedule to update
        cron_expression: Cron expression to calculate next run from

    Returns:
        True if update successful
    """
    supabase = await get_supabase()
    if not supabase:
        return False

    try:
        next_run_at = await calculate_next_run_time(cron_expression)
        if not next_run_at:
            logger.error("Could not calculate next run time", schedule_id=schedule_id)
            return False

        success = await supabase.update(
            "test_schedules",
            {"next_run_at": next_run_at, "updated_at": datetime.now(UTC).isoformat()},
            {"id": schedule_id}
        )

        if success:
            logger.info("Updated schedule next_run_at", schedule_id=schedule_id, next_run_at=next_run_at)

        return success

    except Exception as e:
        logger.exception("Failed to update schedule next_run", schedule_id=schedule_id, error=str(e))
        return False


async def create_scheduled_run(schedule_id: str) -> str | None:
    """
    Create a new schedule run record.

    Args:
        schedule_id: ID of the schedule

    Returns:
        Run ID if created successfully, None otherwise
    """
    supabase = await get_supabase()
    if not supabase:
        return None

    try:
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
            "trigger_type": "scheduled",
            "triggered_by": "scheduler_daemon",
            "tests_total": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "error_message": None,
            "error_details": None,
            "logs": [],
            "metadata": {"triggered_by_daemon": True},
            "created_at": now.isoformat(),
        }

        success = await supabase.insert("schedule_runs", [run])
        if success:
            logger.info("Created scheduled run", run_id=run_id, schedule_id=schedule_id)
            return run_id

        logger.error("Failed to create scheduled run", schedule_id=schedule_id)
        return None

    except Exception as e:
        logger.exception("Exception creating scheduled run", schedule_id=schedule_id, error=str(e))
        return None


async def trigger_scheduled_run(schedule: dict) -> None:
    """
    Trigger a scheduled run for a due schedule.

    Args:
        schedule: Schedule dict with configuration
    """
    schedule_id = schedule["id"]
    schedule_name = schedule.get("name", "Unknown")

    logger.info("Triggering scheduled run", schedule_id=schedule_id, name=schedule_name)

    try:
        # Create run record
        run_id = await create_scheduled_run(schedule_id)
        if not run_id:
            logger.error("Failed to create run record", schedule_id=schedule_id)
            return

        # Import and run the test execution
        from src.api.scheduling import run_scheduled_tests

        # Run in background task (don't await - let it run asynchronously)
        asyncio.create_task(run_scheduled_tests(schedule_id, run_id, "scheduler_daemon"))

        # Update next_run_at immediately so we don't trigger again
        await update_schedule_next_run(schedule_id, schedule["cron_expression"])

        logger.info(
            "Scheduled run triggered successfully",
            schedule_id=schedule_id,
            run_id=run_id,
            name=schedule_name,
        )

    except Exception as e:
        logger.exception("Failed to trigger scheduled run", schedule_id=schedule_id, error=str(e))


async def scheduler_loop() -> None:
    """
    Main scheduler loop that runs continuously.

    Checks for due schedules every 60 seconds and triggers them.
    """
    global _scheduler_running

    logger.info("Scheduler daemon started")
    _scheduler_running = True

    while _scheduler_running:
        try:
            # Get due schedules
            due_schedules = await get_due_schedules()

            if due_schedules:
                logger.info("Found due schedules", count=len(due_schedules))

                # Trigger each due schedule
                for schedule in due_schedules:
                    await trigger_scheduled_run(schedule)

        except Exception as e:
            logger.exception("Scheduler loop error", error=str(e))

        # Wait before next check
        await asyncio.sleep(60)

    logger.info("Scheduler daemon stopped")


async def start_scheduler() -> None:
    """
    Start the background scheduler daemon.

    Should be called from server startup event handler.
    """
    global _scheduler_task

    if not is_supabase_configured():
        logger.warning("Supabase not configured, scheduler daemon not starting")
        return

    if _scheduler_task is not None and not _scheduler_task.done():
        logger.warning("Scheduler daemon already running")
        return

    _scheduler_task = asyncio.create_task(scheduler_loop())
    logger.info("Background scheduler daemon started")


async def stop_scheduler() -> None:
    """
    Stop the background scheduler daemon gracefully.

    Should be called from server shutdown event handler.
    """
    global _scheduler_running, _scheduler_task

    _scheduler_running = False

    if _scheduler_task is not None:
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
        _scheduler_task = None

    logger.info("Background scheduler daemon stopped")
