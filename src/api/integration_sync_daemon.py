"""Integration Sync Daemon.

Background daemon that periodically syncs data from connected integrations
(Sentry, Datadog, GitHub, PagerDuty, Amplitude, etc.) for AI analysis.

Runs as a background asyncio task on server startup, following the same
pattern as schedule_daemon.py.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import structlog

from src.integrations.observability_hub import (
    DatadogProvider,
    FullStoryProvider,
    NewRelicProvider,
    ObservabilityProvider,
    PostHogProvider,
    SentryProvider,
)
from src.integrations.supabase import get_supabase, is_supabase_configured

logger = structlog.get_logger()

# Global control flags
_sync_daemon_running = False
_sync_daemon_task: asyncio.Task | None = None

# Sync intervals per platform
# None = webhook-only (no polling needed)
SYNC_INTERVALS: dict[str, timedelta | None] = {
    "sentry": timedelta(minutes=5),
    "datadog": timedelta(minutes=15),
    "github": timedelta(minutes=10),
    "pagerduty": timedelta(minutes=5),
    "amplitude": timedelta(hours=1),
    "new_relic": timedelta(minutes=15),
    "fullstory": timedelta(minutes=30),
    "posthog": timedelta(minutes=15),
    "slack": None,  # Webhook-only
    "mixpanel": timedelta(hours=1),
    "segment": None,  # Webhook-only
    "honeycomb": timedelta(minutes=15),
    "grafana": timedelta(minutes=15),
    "elastic_apm": timedelta(minutes=15),
    "dynatrace": timedelta(minutes=15),
    "logrocket": timedelta(minutes=30),
}

# Main loop check interval (how often we check for due integrations)
DAEMON_CHECK_INTERVAL_SECONDS = 60


def create_provider(platform: str, config: dict) -> ObservabilityProvider | None:
    """Create the appropriate provider instance from config.

    Args:
        platform: Integration platform type (e.g., "sentry", "datadog")
        config: Configuration dict with credentials and settings

    Returns:
        ObservabilityProvider instance or None if platform not supported
    """
    credentials = config.get("credentials", {})

    try:
        if platform == "sentry":
            return SentryProvider(
                auth_token=credentials.get("auth_token", ""),
                organization=credentials.get("organization", ""),
                project=credentials.get("project", ""),
            )

        elif platform == "datadog":
            return DatadogProvider(
                api_key=credentials.get("api_key", ""),
                app_key=credentials.get("app_key", ""),
                site=credentials.get("site", "datadoghq.com"),
            )

        elif platform == "new_relic":
            return NewRelicProvider(
                api_key=credentials.get("api_key", ""),
                account_id=credentials.get("account_id", ""),
            )

        elif platform == "fullstory":
            return FullStoryProvider(
                api_key=credentials.get("api_key", ""),
            )

        elif platform == "posthog":
            return PostHogProvider(
                api_key=credentials.get("api_key", ""),
                host=credentials.get("host", "https://app.posthog.com"),
            )

        else:
            logger.debug("Unsupported platform for sync", platform=platform)
            return None

    except Exception as e:
        logger.error("Failed to create provider", platform=platform, error=str(e))
        return None


async def get_integrations_due_for_sync() -> list[dict]:
    """Query integrations where last_sync_at + interval < now().

    Returns:
        List of integration records that are due for syncing
    """
    supabase = await get_supabase()
    if not supabase:
        logger.debug("Supabase not configured, integration sync cannot run")
        return []

    try:
        now = datetime.now(UTC)

        # Query all active/connected integrations
        result = await supabase.select(
            "integrations",
            columns="*",
            filters={"status": "connected"},
            order_by="last_sync_at",
            ascending=True,
            limit=20,  # Process max 20 integrations per cycle
        )

        due_integrations = []
        for integration in result:
            platform = integration.get("type", "").lower()
            interval = SYNC_INTERVALS.get(platform)

            # Skip webhook-only integrations
            if interval is None:
                continue

            last_sync_at = integration.get("last_sync_at")
            if last_sync_at:
                # Parse ISO datetime
                if isinstance(last_sync_at, str):
                    last_sync = datetime.fromisoformat(last_sync_at.replace("Z", "+00:00"))
                else:
                    last_sync = last_sync_at

                # Check if due for sync
                next_sync_at = last_sync + interval
                if now >= next_sync_at:
                    due_integrations.append(integration)
            else:
                # Never synced, add to queue
                due_integrations.append(integration)

        return due_integrations

    except Exception as e:
        logger.exception("Failed to query integrations due for sync", error=str(e))
        return []


async def update_sync_status(
    integration_id: str,
    status: str,
    error_message: str | None = None,
    items_synced: int | None = None,
) -> bool:
    """Update sync_status and last_sync_at in database.

    Args:
        integration_id: ID of the integration
        status: New sync status ("syncing", "success", "failed")
        error_message: Error message if sync failed
        items_synced: Number of items synced (for success)

    Returns:
        True if update successful
    """
    supabase = await get_supabase()
    if not supabase:
        return False

    try:
        update_data: dict = {
            "updated_at": datetime.now(UTC).isoformat(),
        }

        if status == "syncing":
            update_data["sync_status"] = "syncing"
        elif status == "success":
            update_data["sync_status"] = "success"
            update_data["last_sync_at"] = datetime.now(UTC).isoformat()
            update_data["error_message"] = None  # Clear any previous error
            if items_synced is not None:
                # Store in metadata
                update_data["metadata"] = {"last_items_synced": items_synced}
        elif status == "failed":
            update_data["sync_status"] = "failed"
            update_data["error_message"] = error_message

        success = await supabase.update(
            "integrations",
            update_data,
            {"id": integration_id},
        )

        if success:
            logger.debug(
                "Updated integration sync status",
                integration_id=integration_id,
                status=status,
            )

        return success

    except Exception as e:
        logger.exception(
            "Failed to update sync status",
            integration_id=integration_id,
            error=str(e),
        )
        return False


async def sync_integration(integration: dict) -> None:
    """Sync data from a single integration using its provider.

    Args:
        integration: Integration dict with configuration and credentials
    """
    integration_id = integration.get("id", "unknown")
    platform = integration.get("type", "").lower()
    project_id = integration.get("project_id")

    logger.info(
        "Starting integration sync",
        integration_id=integration_id,
        platform=platform,
        project_id=project_id,
    )

    # Mark as syncing
    await update_sync_status(integration_id, "syncing")

    try:
        # Create provider instance
        provider = create_provider(platform, integration)
        if not provider:
            await update_sync_status(
                integration_id,
                "failed",
                error_message=f"Unsupported platform: {platform}",
            )
            return

        # Calculate time range for sync
        since = datetime.now(UTC) - timedelta(hours=24)

        # Sync errors from the integration
        items_synced = 0

        try:
            errors = await provider.get_errors(limit=100, since=since)
            items_synced += len(errors)

            if errors:
                logger.info(
                    "Synced errors from integration",
                    integration_id=integration_id,
                    platform=platform,
                    error_count=len(errors),
                )

                # Store synced errors as production events
                await store_synced_errors(project_id, platform, errors)
        except Exception as e:
            logger.warning(
                "Failed to sync errors",
                integration_id=integration_id,
                platform=platform,
                error=str(e),
            )

        # Sync sessions if supported
        try:
            sessions = await provider.get_recent_sessions(limit=50, since=since)
            items_synced += len(sessions)

            if sessions:
                logger.info(
                    "Synced sessions from integration",
                    integration_id=integration_id,
                    platform=platform,
                    session_count=len(sessions),
                )
        except Exception as e:
            logger.debug(
                "Failed to sync sessions (may not be supported)",
                integration_id=integration_id,
                platform=platform,
                error=str(e),
            )

        # Sync performance anomalies if supported
        try:
            anomalies = await provider.get_performance_anomalies(since=since)
            items_synced += len(anomalies)

            if anomalies:
                logger.info(
                    "Synced performance anomalies from integration",
                    integration_id=integration_id,
                    platform=platform,
                    anomaly_count=len(anomalies),
                )
        except Exception as e:
            logger.debug(
                "Failed to sync anomalies (may not be supported)",
                integration_id=integration_id,
                platform=platform,
                error=str(e),
            )

        # Close provider connection
        await provider.close()

        # Mark sync as successful
        await update_sync_status(integration_id, "success", items_synced=items_synced)

        logger.info(
            "Integration sync completed",
            integration_id=integration_id,
            platform=platform,
            items_synced=items_synced,
        )

    except Exception as e:
        logger.exception(
            "Integration sync failed",
            integration_id=integration_id,
            platform=platform,
            error=str(e),
        )
        await update_sync_status(
            integration_id,
            "failed",
            error_message=str(e)[:500],
        )


async def store_synced_errors(
    project_id: str | None,
    platform: str,
    errors: list,
) -> None:
    """Store synced errors as production events in the database.

    Args:
        project_id: Project ID to associate errors with
        platform: Source platform name
        errors: List of ProductionError objects
    """
    if not project_id or not errors:
        return

    supabase = await get_supabase()
    if not supabase:
        return

    try:
        records = []
        for error in errors:
            records.append({
                "project_id": project_id,
                "title": error.message[:500] if error.message else "Unknown Error",
                "message": error.stack_trace[:2000] if error.stack_trace else None,
                "severity": error.severity,
                "status": "pending",
                "source": f"integration:{platform}",
                "occurrence_count": error.occurrence_count,
                "first_seen_at": error.first_seen.isoformat() if error.first_seen else None,
                "last_seen_at": error.last_seen.isoformat() if error.last_seen else None,
                "environment": error.environment,
                "external_id": error.error_id,
                "external_url": error.issue_url,
                "metadata": {
                    "platform": platform,
                    "tags": error.tags,
                    "context": error.context,
                    "release": error.release,
                    "affected_users": error.affected_users,
                },
            })

        if records:
            await supabase.insert("production_events", records)
            logger.debug(
                "Stored synced errors as production events",
                project_id=project_id,
                count=len(records),
            )

    except Exception as e:
        logger.error(
            "Failed to store synced errors",
            project_id=project_id,
            error=str(e),
        )


async def sync_daemon_loop() -> None:
    """Main daemon loop that runs continuously.

    Checks for due integrations every 60 seconds and syncs them.
    Uses asyncio.create_task for concurrent syncs.
    """
    global _sync_daemon_running

    logger.info("Integration sync daemon started")
    _sync_daemon_running = True

    while _sync_daemon_running:
        try:
            # Get due integrations
            due_integrations = await get_integrations_due_for_sync()

            if due_integrations:
                logger.info(
                    "Found integrations due for sync",
                    count=len(due_integrations),
                )

                # Create sync tasks for each integration (concurrent)
                sync_tasks = []
                for integration in due_integrations:
                    task = asyncio.create_task(
                        sync_integration(integration),
                        name=f"sync-{integration.get('id', 'unknown')[:8]}",
                    )
                    sync_tasks.append(task)

                # Wait for all sync tasks to complete (with timeout)
                if sync_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*sync_tasks, return_exceptions=True),
                            timeout=300,  # 5 minute timeout for all syncs
                        )
                    except TimeoutError:
                        logger.warning("Some integration syncs timed out")

        except Exception as e:
            logger.exception("Integration sync daemon loop error", error=str(e))

        # Wait before next check
        await asyncio.sleep(DAEMON_CHECK_INTERVAL_SECONDS)

    logger.info("Integration sync daemon stopped")


async def start_integration_sync_daemon() -> None:
    """Start the background integration sync daemon.

    Should be called from server startup event handler.
    """
    global _sync_daemon_task

    if not is_supabase_configured():
        logger.warning("Supabase not configured, integration sync daemon not starting")
        return

    if _sync_daemon_task is not None and not _sync_daemon_task.done():
        logger.warning("Integration sync daemon already running")
        return

    _sync_daemon_task = asyncio.create_task(sync_daemon_loop())
    logger.info("Background integration sync daemon started")


async def stop_integration_sync_daemon() -> None:
    """Stop the background integration sync daemon gracefully.

    Should be called from server shutdown event handler.
    """
    global _sync_daemon_running, _sync_daemon_task

    _sync_daemon_running = False

    if _sync_daemon_task is not None:
        _sync_daemon_task.cancel()
        try:
            await _sync_daemon_task
        except asyncio.CancelledError:
            pass
        _sync_daemon_task = None

    logger.info("Background integration sync daemon stopped")


async def trigger_manual_sync(integration_id: str) -> dict:
    """Trigger a manual sync for a specific integration.

    Args:
        integration_id: ID of the integration to sync

    Returns:
        Result dict with success status and message
    """
    supabase = await get_supabase()
    if not supabase:
        return {"success": False, "error": "Supabase not configured"}

    try:
        # Fetch the integration
        result = await supabase.select(
            "integrations",
            columns="*",
            filters={"id": integration_id},
            limit=1,
        )

        if not result:
            return {"success": False, "error": "Integration not found"}

        integration = result[0]

        # Sync in background
        asyncio.create_task(sync_integration(integration))

        return {
            "success": True,
            "message": f"Sync started for integration {integration_id}",
            "platform": integration.get("type"),
        }

    except Exception as e:
        logger.exception("Failed to trigger manual sync", error=str(e))
        return {"success": False, "error": str(e)}
