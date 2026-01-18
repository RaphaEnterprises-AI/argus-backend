"""Notifications API for managing Slack and other notification integrations.

Provides endpoints for:
- Sending test notifications
- Configuring notification settings
- Managing notification channels and rules
- Checking notification service status

Now with Supabase persistence for production use.
"""

import uuid
from datetime import UTC, datetime
from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from src.api.teams import get_current_user
from src.integrations.slack import (
    FailureDetails,
    QualityReport,
    ScheduleInfo,
    SlackConfig,
    SlackNotifier,
    TestResult,
    create_slack_notifier,
)
from src.integrations.supabase import get_supabase, is_supabase_configured

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/notifications", tags=["Notifications"])


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_url(url: str | None, field_name: str) -> str | None:
    """Validate that a URL starts with http:// or https://."""
    if url is None:
        return None
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"{field_name} must start with http:// or https://")
    return url


# =============================================================================
# Request/Response Models
# =============================================================================

class SlackConfigureRequest(BaseModel):
    """Request to configure Slack settings."""
    webhook_url: str | None = Field(None, description="Slack webhook URL")
    bot_token: str | None = Field(None, description="Slack bot token")
    default_channel: str = Field("#testing", description="Default notification channel")

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook_url(cls, v: str | None) -> str | None:
        return validate_url(v, "webhook_url")


class SlackTestRequest(BaseModel):
    """Request to send a test notification."""
    channel: str | None = Field(None, description="Target channel (optional)")
    message_type: Literal["test_result", "failure", "schedule", "quality", "simple"] = Field(
        "simple",
        description="Type of test message to send"
    )
    custom_message: str | None = Field(None, description="Custom message for simple type")


class TestResultNotificationRequest(BaseModel):
    """Request to send a test result notification."""
    channel: str | None = Field(None, description="Target channel")
    title: str = Field("E2E Test Results", description="Notification title")
    total: int = Field(..., ge=0, description="Total tests")
    passed: int = Field(..., ge=0, description="Passed tests")
    failed: int = Field(..., ge=0, description="Failed tests")
    skipped: int = Field(0, ge=0, description="Skipped tests")
    duration_seconds: float = Field(..., ge=0, description="Test duration in seconds")
    cost_usd: float = Field(0.0, ge=0, description="AI cost in USD")
    failures: list[dict] = Field(default_factory=list, description="Failure details")
    report_url: str | None = Field(None, description="URL to full report")
    pr_url: str | None = Field(None, description="PR URL")
    pr_number: int | None = Field(None, description="PR number")
    branch: str | None = Field(None, description="Git branch")
    commit_sha: str | None = Field(None, description="Git commit SHA")
    job_id: str | None = Field(None, description="Job ID for rerun")


class FailureAlertRequest(BaseModel):
    """Request to send a failure alert."""
    channel: str | None = Field(None, description="Target channel")
    test_id: str = Field(..., description="Test identifier")
    test_name: str = Field(..., description="Human-readable test name")
    error_message: str = Field(..., description="Error message")
    stack_trace: str | None = Field(None, description="Stack trace")
    screenshot_url: str | None = Field(None, description="Screenshot URL")
    root_cause: str | None = Field(None, description="AI-analyzed root cause")
    component: str | None = Field(None, description="Affected component")
    url: str | None = Field(None, description="Page URL where error occurred")
    duration_ms: int | None = Field(None, description="Test duration in milliseconds")
    retry_count: int = Field(0, ge=0, description="Number of retries attempted")


class ScheduleReminderRequest(BaseModel):
    """Request to send a schedule reminder."""
    channel: str | None = Field(None, description="Target channel")
    schedule_id: str = Field(..., description="Schedule identifier")
    schedule_name: str = Field(..., description="Schedule name")
    next_run_at: datetime = Field(..., description="Next run time (UTC)")
    test_suite: str = Field(..., description="Test suite name")
    estimated_duration_minutes: int = Field(30, ge=1, description="Estimated duration")
    environment: str = Field("staging", description="Target environment")
    notify_channel: str | None = Field(None, description="Override notification channel")


class QualityReportRequest(BaseModel):
    """Request to send a quality report."""
    channel: str | None = Field(None, description="Target channel")
    project_id: str = Field(..., description="Project identifier")
    project_name: str = Field(..., description="Project name")
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score")
    grade: str = Field(..., description="Letter grade (A-F)")
    test_coverage: float = Field(..., ge=0, le=100, description="Test coverage percentage")
    error_count: int = Field(0, ge=0, description="Active error count")
    resolved_count: int = Field(0, ge=0, description="Resolved error count")
    risk_level: str = Field("medium", description="Risk level (low/medium/high)")
    trends: dict = Field(default_factory=dict, description="Trend data")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")
    report_url: str | None = Field(None, description="Full report URL")


class NotificationResponse(BaseModel):
    """Response for notification operations."""
    success: bool
    message: str
    details: dict | None = None


class SlackStatusResponse(BaseModel):
    """Response for Slack status check."""
    configured: bool
    webhook_configured: bool
    bot_configured: bool
    default_channel: str
    webhook_status: str
    api_status: str
    bot_info: dict | None = None


# =============================================================================
# Additional Models for Channel/Rule Management
# =============================================================================

class ChannelCreateRequest(BaseModel):
    """Request to create a notification channel."""
    organization_id: str = Field(..., description="Organization ID")
    project_id: str | None = Field(None, description="Project ID (null = org-wide)")
    name: str = Field(..., min_length=1, max_length=100, description="Channel name")
    channel_type: Literal["slack", "email", "webhook", "discord", "teams", "pagerduty", "opsgenie"] = Field(
        ..., description="Channel type"
    )
    config: dict = Field(..., description="Channel-specific configuration")
    enabled: bool = Field(True, description="Whether the channel is enabled")
    rate_limit_per_hour: int = Field(100, ge=1, le=1000, description="Rate limit per hour")

    @model_validator(mode="after")
    def validate_config_urls(self) -> "ChannelCreateRequest":
        """Validate webhook_url in config for channel types that use it."""
        if self.config and "webhook_url" in self.config:
            webhook_url = self.config["webhook_url"]
            if webhook_url is not None and not webhook_url.startswith(("http://", "https://")):
                raise ValueError("webhook_url in config must start with http:// or https://")
        return self


class ChannelUpdateRequest(BaseModel):
    """Request to update a notification channel."""
    name: str | None = Field(None, min_length=1, max_length=100)
    config: dict | None = None
    enabled: bool | None = None
    rate_limit_per_hour: int | None = Field(None, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_config_urls(self) -> "ChannelUpdateRequest":
        """Validate webhook_url in config if provided."""
        if self.config and "webhook_url" in self.config:
            webhook_url = self.config["webhook_url"]
            if webhook_url is not None and not webhook_url.startswith(("http://", "https://")):
                raise ValueError("webhook_url in config must start with http:// or https://")
        return self


class ChannelResponse(BaseModel):
    """Notification channel response."""
    id: str
    organization_id: str
    project_id: str | None
    name: str
    channel_type: str
    config: dict
    enabled: bool
    verified: bool
    rate_limit_per_hour: int
    last_sent_at: str | None
    sent_today: int
    created_at: str
    updated_at: str


class RuleCreateRequest(BaseModel):
    """Request to create a notification rule."""
    channel_id: str = Field(..., description="Channel ID to send notifications to")
    name: str | None = Field(None, description="Rule name")
    event_type: str = Field(..., description="Event type that triggers this rule")
    conditions: dict | None = Field(default_factory=dict, description="Conditions for triggering")
    message_template: str | None = Field(None, description="Custom message template")
    priority: Literal["low", "normal", "high", "urgent"] = Field("normal", description="Notification priority")
    cooldown_minutes: int = Field(0, ge=0, le=1440, description="Cooldown between notifications")
    enabled: bool = Field(True, description="Whether the rule is enabled")


class RuleResponse(BaseModel):
    """Notification rule response."""
    id: str
    channel_id: str
    name: str | None
    event_type: str
    conditions: dict
    message_template: str | None
    priority: str
    cooldown_minutes: int
    enabled: bool
    last_triggered_at: str | None
    created_at: str
    updated_at: str


class NotificationLogResponse(BaseModel):
    """Notification log entry response."""
    id: str
    channel_id: str
    rule_id: str | None
    event_type: str
    status: str
    response_code: int | None
    error_message: str | None
    sent_at: str | None
    created_at: str


# =============================================================================
# User Notification Models
# =============================================================================

class UserNotification(BaseModel):
    """User notification model."""
    id: str
    user_id: str
    organization_id: str | None = None
    type: str = Field(..., description="Notification type: test_result, failure, schedule, quality, system, mention")
    title: str
    message: str
    read: bool = False
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    action_url: str | None = Field(None, description="URL to navigate to when clicked")
    metadata: dict = Field(default_factory=dict, description="Additional notification data")
    created_at: str
    read_at: str | None = None


class UserNotificationListResponse(BaseModel):
    """Response for listing user notifications."""
    notifications: list[UserNotification]
    total: int
    unread_count: int
    has_more: bool


class UnreadCountResponse(BaseModel):
    """Response for unread notification count."""
    unread_count: int
    by_priority: dict = Field(default_factory=dict, description="Counts by priority level")


class NotificationPreferences(BaseModel):
    """User notification preferences."""
    user_id: str
    email_enabled: bool = True
    email_frequency: Literal["instant", "hourly", "daily", "weekly"] = "instant"
    email_types: list[str] = Field(
        default_factory=lambda: ["test_result", "failure", "quality", "system", "mention"],
        description="Notification types to receive via email"
    )
    in_app_enabled: bool = True
    in_app_types: list[str] = Field(
        default_factory=lambda: ["test_result", "failure", "schedule", "quality", "system", "mention"],
        description="Notification types to show in-app"
    )
    slack_enabled: bool = False
    slack_channel: str | None = None
    slack_types: list[str] = Field(
        default_factory=lambda: ["failure", "quality"],
        description="Notification types to send to Slack"
    )
    quiet_hours_enabled: bool = False
    quiet_hours_start: str | None = Field(None, description="Start time in HH:MM format (UTC)")
    quiet_hours_end: str | None = Field(None, description="End time in HH:MM format (UTC)")
    created_at: str | None = None
    updated_at: str | None = None


class NotificationPreferencesUpdate(BaseModel):
    """Request to update notification preferences."""
    email_enabled: bool | None = None
    email_frequency: Literal["instant", "hourly", "daily", "weekly"] | None = None
    email_types: list[str] | None = None
    in_app_enabled: bool | None = None
    in_app_types: list[str] | None = None
    slack_enabled: bool | None = None
    slack_channel: str | None = None
    slack_types: list[str] | None = None
    quiet_hours_enabled: bool | None = None
    quiet_hours_start: str | None = None
    quiet_hours_end: str | None = None


# =============================================================================
# In-Memory Configuration Store (fallback when Supabase not configured)
# =============================================================================

_slack_config: SlackConfig | None = None
_channels: dict[str, dict] = {}  # In-memory fallback
_rules: dict[str, dict] = {}  # In-memory fallback
_logs: list[dict] = []  # In-memory fallback
_user_notifications: dict[str, dict] = {}  # In-memory fallback for user notifications
_user_preferences: dict[str, dict] = {}  # In-memory fallback for user preferences


# =============================================================================
# Supabase Helper Functions
# =============================================================================

async def _get_channel_from_db(channel_id: str) -> dict | None:
    """Get a notification channel from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        return _channels.get(channel_id)

    result = await supabase.select(
        "notification_channels",
        columns="*",
        filters={"id": channel_id},
        limit=1
    )
    return result[0] if result else None


async def _list_channels_from_db(
    organization_id: str | None = None,
    project_id: str | None = None,
    channel_type: str | None = None,
    enabled: bool | None = None,
    limit: int = 50
) -> list[dict]:
    """List notification channels from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        channels = list(_channels.values())
        if organization_id:
            channels = [c for c in channels if c.get("organization_id") == organization_id]
        if project_id:
            channels = [c for c in channels if c.get("project_id") == project_id]
        if channel_type:
            channels = [c for c in channels if c.get("channel_type") == channel_type]
        if enabled is not None:
            channels = [c for c in channels if c.get("enabled") == enabled]
        return channels[:limit]

    filters = {}
    if organization_id:
        filters["organization_id"] = organization_id
    if project_id:
        filters["project_id"] = project_id
    if channel_type:
        filters["channel_type"] = channel_type
    if enabled is not None:
        filters["enabled"] = enabled

    return await supabase.select(
        "notification_channels",
        columns="*",
        filters=filters if filters else None,
        order_by="created_at",
        ascending=False,
        limit=limit
    )


async def _save_channel_to_db(channel: dict) -> bool:
    """Save a notification channel to Supabase."""
    supabase = await get_supabase()
    if not supabase:
        _channels[channel["id"]] = channel
        return True

    return await supabase.insert("notification_channels", [channel])


async def _update_channel_in_db(channel_id: str, updates: dict) -> bool:
    """Update a notification channel in Supabase."""
    supabase = await get_supabase()
    if not supabase:
        if channel_id in _channels:
            _channels[channel_id].update(updates)
            return True
        return False

    return await supabase.update("notification_channels", updates, {"id": channel_id})


async def _delete_channel_from_db(channel_id: str) -> bool:
    """Delete a notification channel from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        if channel_id in _channels:
            del _channels[channel_id]
            return True
        return False

    return await supabase.delete("notification_channels", {"id": channel_id})


async def _get_rule_from_db(rule_id: str) -> dict | None:
    """Get a notification rule from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        return _rules.get(rule_id)

    result = await supabase.select(
        "notification_rules",
        columns="*",
        filters={"id": rule_id},
        limit=1
    )
    return result[0] if result else None


async def _list_rules_from_db(channel_id: str | None = None, event_type: str | None = None, limit: int = 50) -> list[dict]:
    """List notification rules from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        rules = list(_rules.values())
        if channel_id:
            rules = [r for r in rules if r.get("channel_id") == channel_id]
        if event_type:
            rules = [r for r in rules if r.get("event_type") == event_type]
        return rules[:limit]

    filters = {}
    if channel_id:
        filters["channel_id"] = channel_id
    if event_type:
        filters["event_type"] = event_type

    return await supabase.select(
        "notification_rules",
        columns="*",
        filters=filters if filters else None,
        order_by="created_at",
        ascending=False,
        limit=limit
    )


async def _save_rule_to_db(rule: dict) -> bool:
    """Save a notification rule to Supabase."""
    supabase = await get_supabase()
    if not supabase:
        _rules[rule["id"]] = rule
        return True

    return await supabase.insert("notification_rules", [rule])


async def _update_rule_in_db(rule_id: str, updates: dict) -> bool:
    """Update a notification rule in Supabase."""
    supabase = await get_supabase()
    if not supabase:
        if rule_id in _rules:
            _rules[rule_id].update(updates)
            return True
        return False

    return await supabase.update("notification_rules", updates, {"id": rule_id})


async def _delete_rule_from_db(rule_id: str) -> bool:
    """Delete a notification rule from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        if rule_id in _rules:
            del _rules[rule_id]
            return True
        return False

    return await supabase.delete("notification_rules", {"id": rule_id})


async def _save_notification_log(log: dict) -> bool:
    """Save a notification log entry to Supabase."""
    supabase = await get_supabase()
    if not supabase:
        _logs.append(log)
        if len(_logs) > 1000:  # Keep last 1000 in memory
            _logs.pop(0)
        return True

    return await supabase.insert("notification_logs", [log])


async def _list_notification_logs(channel_id: str | None = None, status: str | None = None, limit: int = 50) -> list[dict]:
    """List notification logs from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        logs = _logs.copy()
        if channel_id:
            logs = [l for l in logs if l.get("channel_id") == channel_id]
        if status:
            logs = [l for l in logs if l.get("status") == status]
        return logs[-limit:]

    filters = {}
    if channel_id:
        filters["channel_id"] = channel_id
    if status:
        filters["status"] = status

    return await supabase.select(
        "notification_logs",
        columns="*",
        filters=filters if filters else None,
        order_by="created_at",
        ascending=False,
        limit=limit
    )


def get_slack_notifier() -> SlackNotifier:
    """Get configured Slack notifier instance."""
    global _slack_config

    if _slack_config:
        return SlackNotifier(config=_slack_config)

    # Fall back to environment variables
    return create_slack_notifier()


def set_slack_config(config: SlackConfig) -> None:
    """Set Slack configuration."""
    global _slack_config
    _slack_config = config


# =============================================================================
# User Notification Helper Functions
# =============================================================================

async def _get_user_notification_from_db(notification_id: str) -> dict | None:
    """Get a user notification from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        return _user_notifications.get(notification_id)

    result = await supabase.select(
        "user_notifications",
        columns="*",
        filters={"id": notification_id},
        limit=1
    )
    return result[0] if result else None


async def _list_user_notifications_from_db(
    user_id: str,
    read: bool | None = None,
    notification_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """List user notifications from Supabase. Returns (notifications, total_count)."""
    supabase = await get_supabase()
    if not supabase:
        notifications = [n for n in _user_notifications.values() if n.get("user_id") == user_id]
        if read is not None:
            notifications = [n for n in notifications if n.get("read") == read]
        if notification_type:
            notifications = [n for n in notifications if n.get("type") == notification_type]
        # Sort by created_at descending
        notifications.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        total = len(notifications)
        return notifications[offset:offset + limit], total

    filters = {"user_id": user_id}
    if read is not None:
        filters["read"] = read
    if notification_type:
        filters["type"] = notification_type

    # Get total count
    count_result = await supabase.select(
        "user_notifications",
        columns="id",
        filters=filters,
    )
    total = len(count_result) if count_result else 0

    # Get paginated results
    result = await supabase.select(
        "user_notifications",
        columns="*",
        filters=filters,
        order_by="created_at",
        ascending=False,
        limit=limit,
        offset=offset
    )
    return result or [], total


async def _get_unread_count_from_db(user_id: str) -> dict:
    """Get unread notification count for a user."""
    supabase = await get_supabase()
    if not supabase:
        notifications = [n for n in _user_notifications.values() if n.get("user_id") == user_id and not n.get("read")]
        by_priority = {}
        for n in notifications:
            priority = n.get("priority", "normal")
            by_priority[priority] = by_priority.get(priority, 0) + 1
        return {"unread_count": len(notifications), "by_priority": by_priority}

    result = await supabase.select(
        "user_notifications",
        columns="id,priority",
        filters={"user_id": user_id, "read": False},
    )
    notifications = result or []
    by_priority = {}
    for n in notifications:
        priority = n.get("priority", "normal")
        by_priority[priority] = by_priority.get(priority, 0) + 1
    return {"unread_count": len(notifications), "by_priority": by_priority}


async def _save_user_notification_to_db(notification: dict) -> bool:
    """Save a user notification to Supabase."""
    supabase = await get_supabase()
    if not supabase:
        _user_notifications[notification["id"]] = notification
        return True

    return await supabase.insert("user_notifications", [notification])


async def _update_user_notification_in_db(notification_id: str, updates: dict) -> bool:
    """Update a user notification in Supabase."""
    supabase = await get_supabase()
    if not supabase:
        if notification_id in _user_notifications:
            _user_notifications[notification_id].update(updates)
            return True
        return False

    return await supabase.update("user_notifications", updates, {"id": notification_id})


async def _mark_all_notifications_read(user_id: str) -> int:
    """Mark all notifications as read for a user. Returns count updated."""
    supabase = await get_supabase()
    now = datetime.now(UTC).isoformat()

    if not supabase:
        count = 0
        for notification in _user_notifications.values():
            if notification.get("user_id") == user_id and not notification.get("read"):
                notification["read"] = True
                notification["read_at"] = now
                count += 1
        return count

    # Get unread notifications for this user
    unread = await supabase.select(
        "user_notifications",
        columns="id",
        filters={"user_id": user_id, "read": False},
    )

    if not unread:
        return 0

    # Update each notification
    for notification in unread:
        await supabase.update(
            "user_notifications",
            {"read": True, "read_at": now},
            {"id": notification["id"]}
        )

    return len(unread)


async def _get_user_preferences_from_db(user_id: str) -> dict | None:
    """Get user notification preferences from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        return _user_preferences.get(user_id)

    result = await supabase.select(
        "notification_preferences",
        columns="*",
        filters={"user_id": user_id},
        limit=1
    )
    return result[0] if result else None


async def _save_user_preferences_to_db(preferences: dict) -> bool:
    """Save user notification preferences to Supabase."""
    supabase = await get_supabase()
    if not supabase:
        _user_preferences[preferences["user_id"]] = preferences
        return True

    # Use upsert pattern - try insert, update on conflict
    existing = await _get_user_preferences_from_db(preferences["user_id"])
    if existing:
        return await supabase.update(
            "notification_preferences",
            preferences,
            {"user_id": preferences["user_id"]}
        )
    return await supabase.insert("notification_preferences", [preferences])


async def _update_user_preferences_in_db(user_id: str, updates: dict) -> bool:
    """Update user notification preferences in Supabase."""
    supabase = await get_supabase()
    if not supabase:
        if user_id in _user_preferences:
            _user_preferences[user_id].update(updates)
            return True
        return False

    return await supabase.update("notification_preferences", updates, {"user_id": user_id})


# =============================================================================
# Slack Notification Endpoints
# =============================================================================

@router.post("/slack/test", response_model=NotificationResponse)
async def send_test_notification(request: SlackTestRequest):
    """
    Send a test notification to verify Slack integration.

    Use this endpoint to test that your Slack configuration is working correctly.
    """
    notifier = get_slack_notifier()

    if not notifier.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Slack is not configured. Set SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN environment variables, or use POST /api/v1/notifications/slack/configure"
        )

    try:
        success = False

        if request.message_type == "simple":
            message = request.custom_message or "This is a test notification from Argus E2E Testing Agent!"
            success = await notifier.send_message(
                channel=request.channel,
                message=message,
            )

        elif request.message_type == "test_result":
            # Send sample test result
            result = TestResult(
                total=100,
                passed=95,
                failed=3,
                skipped=2,
                duration_seconds=245.5,
                cost_usd=0.0125,
                failures=[
                    {"test_id": "login-test", "error_message": "Element not found: #login-button"},
                    {"test_id": "checkout-test", "error_message": "Timeout waiting for cart to load"},
                    {"test_id": "search-test", "error_message": "Expected 5 results, got 3"},
                ],
                branch="feature/new-ui",
                commit_sha="abc123def",
            )
            success = await notifier.send_test_result(result, channel=request.channel)

        elif request.message_type == "failure":
            # Send sample failure alert
            failure = FailureDetails(
                test_id="login-test",
                test_name="User Login Flow",
                error_message="Element not found: #login-button. The login button selector may have changed.",
                stack_trace="at LoginPage.clickLogin (login.spec.ts:45:12)\nat async runTest (runner.ts:123:8)",
                root_cause="The login button selector changed from #login-button to .btn-login in the latest UI update.",
                component="LoginPage",
                url="https://app.example.com/login",
                duration_ms=5432,
                retry_count=2,
            )
            success = await notifier.send_failure_alert(failure, channel=request.channel)

        elif request.message_type == "schedule":
            # Send sample schedule reminder
            schedule = ScheduleInfo(
                schedule_id="sched-123",
                schedule_name="Nightly Regression Suite",
                next_run_at=datetime.now(UTC),
                test_suite="Full E2E Suite",
                estimated_duration_minutes=45,
                environment="staging",
            )
            success = await notifier.send_schedule_reminder(schedule, channel=request.channel)

        elif request.message_type == "quality":
            # Send sample quality report
            report = QualityReport(
                project_id="proj-123",
                project_name="My Application",
                overall_score=78.5,
                grade="C+",
                test_coverage=65.2,
                error_count=12,
                resolved_count=45,
                risk_level="medium",
                trends={"score": +3.5, "coverage": +2.1, "errors": -5},
                recommendations=[
                    "Add tests for the checkout flow - high error frequency",
                    "Update selectors for login page - recent UI changes",
                    "Increase coverage for payment processing module",
                ],
            )
            success = await notifier.send_quality_report(report, channel=request.channel)

        if success:
            return NotificationResponse(
                success=True,
                message=f"Test notification ({request.message_type}) sent successfully",
                details={"channel": request.channel or notifier.config.default_channel},
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to send notification. Check Slack configuration and logs."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to send test notification", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/slack/configure", response_model=NotificationResponse)
async def configure_slack(request: SlackConfigureRequest):
    """
    Configure Slack notification settings.

    This configuration is stored in memory and overrides environment variables.
    For production, store configuration in a database.
    """
    config = SlackConfig(
        webhook_url=request.webhook_url,
        bot_token=request.bot_token,
        default_channel=request.default_channel,
    )

    if not config.webhook_url and not config.bot_token:
        raise HTTPException(
            status_code=400,
            detail="At least one of webhook_url or bot_token must be provided"
        )

    set_slack_config(config)

    # Verify the configuration works
    notifier = SlackNotifier(config=config)
    status = await notifier.check_connection()

    return NotificationResponse(
        success=True,
        message="Slack configuration updated successfully",
        details=status,
    )


@router.get("/slack/status", response_model=SlackStatusResponse)
async def get_slack_status():
    """
    Check Slack connection status and configuration.

    Returns information about the current Slack configuration
    and whether connections are working.
    """
    notifier = get_slack_notifier()
    status = await notifier.check_connection()

    return SlackStatusResponse(
        configured=status["configured"],
        webhook_configured=status["webhook_configured"],
        bot_configured=status["bot_configured"],
        default_channel=status["default_channel"],
        webhook_status=status["webhook_status"],
        api_status=status["api_status"],
        bot_info=status.get("bot_info"),
    )


# =============================================================================
# Notification Sending Endpoints
# =============================================================================

@router.post("/slack/test-result", response_model=NotificationResponse)
async def send_test_result_notification(request: TestResultNotificationRequest):
    """
    Send a test result notification to Slack.

    Use this endpoint to notify your team about test run results.
    """
    notifier = get_slack_notifier()

    if not notifier.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Slack is not configured"
        )

    try:
        result = TestResult(
            total=request.total,
            passed=request.passed,
            failed=request.failed,
            skipped=request.skipped,
            duration_seconds=request.duration_seconds,
            cost_usd=request.cost_usd,
            failures=request.failures,
            report_url=request.report_url,
            pr_url=request.pr_url,
            pr_number=request.pr_number,
            branch=request.branch,
            commit_sha=request.commit_sha,
            job_id=request.job_id,
        )

        success = await notifier.send_test_result(
            result,
            channel=request.channel,
            title=request.title,
        )

        if success:
            return NotificationResponse(
                success=True,
                message="Test result notification sent",
                details={
                    "passed": request.passed,
                    "failed": request.failed,
                    "total": request.total,
                },
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send notification")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to send test result notification", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/slack/failure-alert", response_model=NotificationResponse)
async def send_failure_alert_notification(request: FailureAlertRequest):
    """
    Send a failure alert notification to Slack.

    Use this endpoint for immediate failure alerts during test execution.
    """
    notifier = get_slack_notifier()

    if not notifier.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Slack is not configured"
        )

    try:
        failure = FailureDetails(
            test_id=request.test_id,
            test_name=request.test_name,
            error_message=request.error_message,
            stack_trace=request.stack_trace,
            screenshot_url=request.screenshot_url,
            root_cause=request.root_cause,
            component=request.component,
            url=request.url,
            duration_ms=request.duration_ms,
            retry_count=request.retry_count,
        )

        success = await notifier.send_failure_alert(failure, channel=request.channel)

        if success:
            return NotificationResponse(
                success=True,
                message="Failure alert sent",
                details={"test_id": request.test_id, "test_name": request.test_name},
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send notification")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to send failure alert", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/slack/schedule-reminder", response_model=NotificationResponse)
async def send_schedule_reminder_notification(request: ScheduleReminderRequest):
    """
    Send a schedule reminder notification to Slack.

    Use this endpoint to notify about upcoming scheduled test runs.
    """
    notifier = get_slack_notifier()

    if not notifier.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Slack is not configured"
        )

    try:
        schedule = ScheduleInfo(
            schedule_id=request.schedule_id,
            schedule_name=request.schedule_name,
            next_run_at=request.next_run_at,
            test_suite=request.test_suite,
            estimated_duration_minutes=request.estimated_duration_minutes,
            environment=request.environment,
            notify_channel=request.notify_channel,
        )

        success = await notifier.send_schedule_reminder(schedule, channel=request.channel)

        if success:
            return NotificationResponse(
                success=True,
                message="Schedule reminder sent",
                details={
                    "schedule_id": request.schedule_id,
                    "next_run_at": request.next_run_at.isoformat(),
                },
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send notification")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to send schedule reminder", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/slack/quality-report", response_model=NotificationResponse)
async def send_quality_report_notification(request: QualityReportRequest):
    """
    Send a quality report notification to Slack.

    Use this endpoint to share quality scores and insights with your team.
    """
    notifier = get_slack_notifier()

    if not notifier.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Slack is not configured"
        )

    try:
        report = QualityReport(
            project_id=request.project_id,
            project_name=request.project_name,
            overall_score=request.overall_score,
            grade=request.grade,
            test_coverage=request.test_coverage,
            error_count=request.error_count,
            resolved_count=request.resolved_count,
            risk_level=request.risk_level,
            trends=request.trends,
            recommendations=request.recommendations,
            report_url=request.report_url,
        )

        success = await notifier.send_quality_report(report, channel=request.channel)

        if success:
            return NotificationResponse(
                success=True,
                message="Quality report sent",
                details={
                    "project_id": request.project_id,
                    "score": request.overall_score,
                    "grade": request.grade,
                },
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send notification")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to send quality report", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/slack/message", response_model=NotificationResponse)
async def send_custom_message(
    message: str,
    channel: str | None = None,
):
    """
    Send a custom text message to Slack.

    Use this endpoint for simple notifications that don't fit other categories.
    """
    notifier = get_slack_notifier()

    if not notifier.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Slack is not configured"
        )

    try:
        success = await notifier.send_message(
            channel=channel,
            message=message,
        )

        if success:
            return NotificationResponse(
                success=True,
                message="Message sent",
                details={"channel": channel or notifier.config.default_channel},
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send message")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to send message", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Channel Management Endpoints
# =============================================================================

@router.post("/channels", response_model=ChannelResponse, status_code=201)
async def create_channel(body: ChannelCreateRequest, request: Request):
    """
    Create a new notification channel.

    Channels can be Slack, email, webhook, Discord, Teams, PagerDuty, or OpsGenie.
    """
    user = await get_current_user(request)
    channel_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    user_id = user["user_id"]

    channel = {
        "id": channel_id,
        "organization_id": body.organization_id,
        "project_id": body.project_id,
        "name": body.name,
        "channel_type": body.channel_type,
        "config": body.config,
        "enabled": body.enabled,
        "verified": False,
        "rate_limit_per_hour": body.rate_limit_per_hour,
        "last_sent_at": None,
        "sent_today": 0,
        "created_by": user_id,
        "created_at": now,
        "updated_at": now,
    }

    await _save_channel_to_db(channel)

    logger.info(
        "Notification channel created",
        channel_id=channel_id,
        name=body.name,
        type=body.channel_type,
        persistent=is_supabase_configured(),
    )

    return ChannelResponse(**channel)


@router.get("/channels", response_model=list[ChannelResponse])
async def list_channels(
    organization_id: str | None = Query(None, description="Filter by organization"),
    project_id: str | None = Query(None, description="Filter by project"),
    channel_type: str | None = Query(None, description="Filter by channel type"),
    enabled: bool | None = Query(None, description="Filter by enabled status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum channels to return"),
):
    """
    List notification channels with optional filtering.
    """
    channels = await _list_channels_from_db(
        organization_id=organization_id,
        project_id=project_id,
        channel_type=channel_type,
        enabled=enabled,
        limit=limit
    )

    return [ChannelResponse(**c) for c in channels]


@router.get("/channels/{channel_id}", response_model=ChannelResponse)
async def get_channel(channel_id: str):
    """
    Get a notification channel by ID.
    """
    channel = await _get_channel_from_db(channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    return ChannelResponse(**channel)


@router.patch("/channels/{channel_id}", response_model=ChannelResponse)
async def update_channel(channel_id: str, body: ChannelUpdateRequest):
    """
    Update a notification channel.
    """
    channel = await _get_channel_from_db(channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    update_data = body.model_dump(exclude_unset=True)
    update_data["updated_at"] = datetime.now(UTC).isoformat()

    await _update_channel_in_db(channel_id, update_data)
    channel.update(update_data)

    logger.info("Notification channel updated", channel_id=channel_id, updates=list(update_data.keys()))

    return ChannelResponse(**channel)


@router.delete("/channels/{channel_id}")
async def delete_channel(channel_id: str):
    """
    Delete a notification channel.
    """
    channel = await _get_channel_from_db(channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    await _delete_channel_from_db(channel_id)

    logger.info("Notification channel deleted", channel_id=channel_id, name=channel.get("name"))

    return {"success": True, "message": "Channel deleted", "channel_id": channel_id}


@router.post("/channels/{channel_id}/test", response_model=NotificationResponse)
async def test_channel(channel_id: str):
    """
    Send a test notification to verify a channel is working.
    """
    channel = await _get_channel_from_db(channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    # For now, only support Slack testing through this endpoint
    if channel["channel_type"] != "slack":
        raise HTTPException(
            status_code=400,
            detail=f"Testing for {channel['channel_type']} channels not yet implemented"
        )

    config = channel.get("config", {})
    slack_config = SlackConfig(
        webhook_url=config.get("webhook_url"),
        bot_token=config.get("bot_token"),
        default_channel=config.get("channel", "#testing"),
    )

    notifier = SlackNotifier(config=slack_config)
    success = await notifier.send_message(
        message=f"Test notification from Argus - Channel: {channel['name']}",
    )

    if success:
        # Mark channel as verified
        await _update_channel_in_db(channel_id, {
            "verified": True,
            "updated_at": datetime.now(UTC).isoformat()
        })

        return NotificationResponse(
            success=True,
            message="Test notification sent successfully",
            details={"channel_id": channel_id, "verified": True},
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to send test notification")


# =============================================================================
# Rule Management Endpoints
# =============================================================================

@router.post("/rules", response_model=RuleResponse, status_code=201)
async def create_rule(body: RuleCreateRequest, request: Request):
    """
    Create a new notification rule.

    Rules define when notifications are sent based on event types and conditions.
    """
    user = await get_current_user(request)

    # Verify channel exists
    channel = await _get_channel_from_db(body.channel_id)
    if not channel:
        raise HTTPException(status_code=400, detail="Channel not found")

    rule_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    user_id = user["user_id"]

    rule = {
        "id": rule_id,
        "channel_id": body.channel_id,
        "name": body.name or f"Rule for {body.event_type}",
        "event_type": body.event_type,
        "conditions": body.conditions or {},
        "message_template": body.message_template,
        "priority": body.priority,
        "cooldown_minutes": body.cooldown_minutes,
        "enabled": body.enabled,
        "last_triggered_at": None,
        "trigger_count": 0,
        "created_by": user_id,
        "created_at": now,
        "updated_at": now,
    }

    await _save_rule_to_db(rule)

    logger.info(
        "Notification rule created",
        rule_id=rule_id,
        event_type=body.event_type,
        channel_id=body.channel_id,
    )

    return RuleResponse(**rule)


@router.get("/rules", response_model=list[RuleResponse])
async def list_rules(
    channel_id: str | None = Query(None, description="Filter by channel"),
    event_type: str | None = Query(None, description="Filter by event type"),
    limit: int = Query(50, ge=1, le=100, description="Maximum rules to return"),
):
    """
    List notification rules with optional filtering.
    """
    rules = await _list_rules_from_db(channel_id=channel_id, event_type=event_type, limit=limit)
    return [RuleResponse(**r) for r in rules]


@router.get("/rules/{rule_id}", response_model=RuleResponse)
async def get_rule(rule_id: str):
    """
    Get a notification rule by ID.
    """
    rule = await _get_rule_from_db(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    return RuleResponse(**rule)


@router.patch("/rules/{rule_id}", response_model=RuleResponse)
async def update_rule(rule_id: str, body: RuleCreateRequest):
    """
    Update a notification rule.
    """
    rule = await _get_rule_from_db(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    update_data = body.model_dump(exclude_unset=True)
    update_data["updated_at"] = datetime.now(UTC).isoformat()

    await _update_rule_in_db(rule_id, update_data)
    rule.update(update_data)

    logger.info("Notification rule updated", rule_id=rule_id, updates=list(update_data.keys()))

    return RuleResponse(**rule)


@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """
    Delete a notification rule.
    """
    rule = await _get_rule_from_db(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    await _delete_rule_from_db(rule_id)

    logger.info("Notification rule deleted", rule_id=rule_id, event_type=rule.get("event_type"))

    return {"success": True, "message": "Rule deleted", "rule_id": rule_id}


# =============================================================================
# Notification Logs Endpoints
# =============================================================================

@router.get("/logs", response_model=list[NotificationLogResponse])
async def list_notification_logs(
    channel_id: str | None = Query(None, description="Filter by channel"),
    status: str | None = Query(None, description="Filter by status (sent, failed, pending)"),
    limit: int = Query(50, ge=1, le=100, description="Maximum logs to return"),
):
    """
    List notification delivery logs.

    Use this to monitor notification delivery status and troubleshoot failures.
    """
    logs = await _list_notification_logs(channel_id=channel_id, status=status, limit=limit)
    return [NotificationLogResponse(**log) for log in logs]


# =============================================================================
# Event Types Reference
# =============================================================================

@router.get("/event-types")
async def list_event_types():
    """
    List all available event types for notification rules.
    """
    return {
        "event_types": [
            {"type": "test.started", "description": "Test execution started"},
            {"type": "test.passed", "description": "Test execution passed"},
            {"type": "test.failed", "description": "Test execution failed"},
            {"type": "test.flaky", "description": "Flaky test detected"},
            {"type": "schedule.started", "description": "Scheduled run started"},
            {"type": "schedule.completed", "description": "Scheduled run completed"},
            {"type": "schedule.failed", "description": "Scheduled run failed"},
            {"type": "healing.applied", "description": "Self-healing fix applied"},
            {"type": "healing.suggested", "description": "Self-healing fix suggested"},
            {"type": "quality.threshold", "description": "Quality score crossed threshold"},
            {"type": "quality.report", "description": "Weekly quality report"},
            {"type": "alert.security", "description": "Security alert"},
            {"type": "alert.performance", "description": "Performance degradation"},
        ]
    }


# =============================================================================
# User Notification Endpoints
# =============================================================================

@router.get("", response_model=UserNotificationListResponse)
async def list_user_notifications(
    request: Request,
    read: bool | None = Query(None, description="Filter by read status (true/false)"),
    notification_type: str | None = Query(None, alias="type", description="Filter by notification type"),
    limit: int = Query(50, ge=1, le=100, description="Maximum notifications to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    List notifications for the authenticated user.

    Supports filtering by read status and notification type.
    Returns paginated results with total count and unread count.
    """
    user = await get_current_user(request)
    user_id = user["user_id"]

    notifications, total = await _list_user_notifications_from_db(
        user_id=user_id,
        read=read,
        notification_type=notification_type,
        limit=limit,
        offset=offset,
    )

    # Get unread count
    unread_data = await _get_unread_count_from_db(user_id)

    logger.info(
        "Listed user notifications",
        user_id=user_id,
        total=total,
        returned=len(notifications),
        unread=unread_data["unread_count"],
    )

    return UserNotificationListResponse(
        notifications=[UserNotification(**n) for n in notifications],
        total=total,
        unread_count=unread_data["unread_count"],
        has_more=(offset + limit) < total,
    )


@router.get("/unread-count", response_model=UnreadCountResponse)
async def get_unread_notification_count(request: Request):
    """
    Get the count of unread notifications for the authenticated user.

    Returns total unread count and breakdown by priority level.
    """
    user = await get_current_user(request)
    user_id = user["user_id"]

    unread_data = await _get_unread_count_from_db(user_id)

    return UnreadCountResponse(
        unread_count=unread_data["unread_count"],
        by_priority=unread_data["by_priority"],
    )


@router.put("/{notification_id}/read", response_model=NotificationResponse)
async def mark_notification_as_read(notification_id: str, request: Request):
    """
    Mark a specific notification as read.

    The notification must belong to the authenticated user.
    """
    user = await get_current_user(request)
    user_id = user["user_id"]

    notification = await _get_user_notification_from_db(notification_id)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    if notification.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this notification")

    if notification.get("read"):
        return NotificationResponse(
            success=True,
            message="Notification already marked as read",
            details={"notification_id": notification_id},
        )

    now = datetime.now(UTC).isoformat()
    await _update_user_notification_in_db(notification_id, {
        "read": True,
        "read_at": now,
    })

    logger.info("Notification marked as read", notification_id=notification_id, user_id=user_id)

    return NotificationResponse(
        success=True,
        message="Notification marked as read",
        details={"notification_id": notification_id, "read_at": now},
    )


@router.put("/mark-all-read", response_model=NotificationResponse)
async def mark_all_notifications_as_read(request: Request):
    """
    Mark all notifications as read for the authenticated user.

    Returns the count of notifications that were marked as read.
    """
    user = await get_current_user(request)
    user_id = user["user_id"]

    count = await _mark_all_notifications_read(user_id)

    logger.info("All notifications marked as read", user_id=user_id, count=count)

    return NotificationResponse(
        success=True,
        message=f"Marked {count} notifications as read",
        details={"count": count},
    )


# =============================================================================
# Notification Preferences Endpoints
# =============================================================================

@router.get("/preferences", response_model=NotificationPreferences)
async def get_notification_preferences(request: Request):
    """
    Get notification preferences for the authenticated user.

    Returns preferences for email, in-app, and Slack notifications.
    If no preferences exist, returns default preferences.
    """
    user = await get_current_user(request)
    user_id = user["user_id"]

    preferences = await _get_user_preferences_from_db(user_id)

    if preferences:
        return NotificationPreferences(**preferences)

    # Return default preferences for new users
    now = datetime.now(UTC).isoformat()
    default_preferences = NotificationPreferences(
        user_id=user_id,
        email_enabled=True,
        email_frequency="instant",
        email_types=["test_result", "failure", "quality", "system", "mention"],
        in_app_enabled=True,
        in_app_types=["test_result", "failure", "schedule", "quality", "system", "mention"],
        slack_enabled=False,
        slack_channel=None,
        slack_types=["failure", "quality"],
        quiet_hours_enabled=False,
        quiet_hours_start=None,
        quiet_hours_end=None,
        created_at=now,
        updated_at=now,
    )

    return default_preferences


@router.put("/preferences", response_model=NotificationPreferences)
async def update_notification_preferences(
    body: NotificationPreferencesUpdate,
    request: Request,
):
    """
    Update notification preferences for the authenticated user.

    Supports partial updates - only provided fields will be updated.
    Preferences control notifications for email, in-app, and Slack channels.
    """
    user = await get_current_user(request)
    user_id = user["user_id"]

    now = datetime.now(UTC).isoformat()

    # Get existing preferences or create defaults
    existing = await _get_user_preferences_from_db(user_id)

    if existing:
        # Update existing preferences
        update_data = body.model_dump(exclude_unset=True)
        update_data["updated_at"] = now

        await _update_user_preferences_in_db(user_id, update_data)
        existing.update(update_data)
        preferences = existing
    else:
        # Create new preferences with provided values
        preferences = {
            "user_id": user_id,
            "email_enabled": body.email_enabled if body.email_enabled is not None else True,
            "email_frequency": body.email_frequency or "instant",
            "email_types": body.email_types or ["test_result", "failure", "quality", "system", "mention"],
            "in_app_enabled": body.in_app_enabled if body.in_app_enabled is not None else True,
            "in_app_types": body.in_app_types or ["test_result", "failure", "schedule", "quality", "system", "mention"],
            "slack_enabled": body.slack_enabled if body.slack_enabled is not None else False,
            "slack_channel": body.slack_channel,
            "slack_types": body.slack_types or ["failure", "quality"],
            "quiet_hours_enabled": body.quiet_hours_enabled if body.quiet_hours_enabled is not None else False,
            "quiet_hours_start": body.quiet_hours_start,
            "quiet_hours_end": body.quiet_hours_end,
            "created_at": now,
            "updated_at": now,
        }
        await _save_user_preferences_to_db(preferences)

    logger.info(
        "Notification preferences updated",
        user_id=user_id,
        updates=list(body.model_dump(exclude_unset=True).keys()),
    )

    return NotificationPreferences(**preferences)
