"""Notifications API for managing Slack and other notification integrations.

Provides endpoints for:
- Sending test notifications
- Configuring notification settings
- Checking notification service status
"""

import os
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import structlog

from src.integrations.slack import (
    SlackNotifier,
    SlackConfig,
    TestResult,
    FailureDetails,
    ScheduleInfo,
    QualityReport,
    create_slack_notifier,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/notifications", tags=["Notifications"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SlackConfigureRequest(BaseModel):
    """Request to configure Slack settings."""
    webhook_url: Optional[str] = Field(None, description="Slack webhook URL")
    bot_token: Optional[str] = Field(None, description="Slack bot token")
    default_channel: str = Field("#testing", description="Default notification channel")


class SlackTestRequest(BaseModel):
    """Request to send a test notification."""
    channel: Optional[str] = Field(None, description="Target channel (optional)")
    message_type: Literal["test_result", "failure", "schedule", "quality", "simple"] = Field(
        "simple",
        description="Type of test message to send"
    )
    custom_message: Optional[str] = Field(None, description="Custom message for simple type")


class TestResultNotificationRequest(BaseModel):
    """Request to send a test result notification."""
    channel: Optional[str] = Field(None, description="Target channel")
    title: str = Field("E2E Test Results", description="Notification title")
    total: int = Field(..., ge=0, description="Total tests")
    passed: int = Field(..., ge=0, description="Passed tests")
    failed: int = Field(..., ge=0, description="Failed tests")
    skipped: int = Field(0, ge=0, description="Skipped tests")
    duration_seconds: float = Field(..., ge=0, description="Test duration in seconds")
    cost_usd: float = Field(0.0, ge=0, description="AI cost in USD")
    failures: list[dict] = Field(default_factory=list, description="Failure details")
    report_url: Optional[str] = Field(None, description="URL to full report")
    pr_url: Optional[str] = Field(None, description="PR URL")
    pr_number: Optional[int] = Field(None, description="PR number")
    branch: Optional[str] = Field(None, description="Git branch")
    commit_sha: Optional[str] = Field(None, description="Git commit SHA")
    job_id: Optional[str] = Field(None, description="Job ID for rerun")


class FailureAlertRequest(BaseModel):
    """Request to send a failure alert."""
    channel: Optional[str] = Field(None, description="Target channel")
    test_id: str = Field(..., description="Test identifier")
    test_name: str = Field(..., description="Human-readable test name")
    error_message: str = Field(..., description="Error message")
    stack_trace: Optional[str] = Field(None, description="Stack trace")
    screenshot_url: Optional[str] = Field(None, description="Screenshot URL")
    root_cause: Optional[str] = Field(None, description="AI-analyzed root cause")
    component: Optional[str] = Field(None, description="Affected component")
    url: Optional[str] = Field(None, description="Page URL where error occurred")
    duration_ms: Optional[int] = Field(None, description="Test duration in milliseconds")
    retry_count: int = Field(0, ge=0, description="Number of retries attempted")


class ScheduleReminderRequest(BaseModel):
    """Request to send a schedule reminder."""
    channel: Optional[str] = Field(None, description="Target channel")
    schedule_id: str = Field(..., description="Schedule identifier")
    schedule_name: str = Field(..., description="Schedule name")
    next_run_at: datetime = Field(..., description="Next run time (UTC)")
    test_suite: str = Field(..., description="Test suite name")
    estimated_duration_minutes: int = Field(30, ge=1, description="Estimated duration")
    environment: str = Field("staging", description="Target environment")
    notify_channel: Optional[str] = Field(None, description="Override notification channel")


class QualityReportRequest(BaseModel):
    """Request to send a quality report."""
    channel: Optional[str] = Field(None, description="Target channel")
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
    report_url: Optional[str] = Field(None, description="Full report URL")


class NotificationResponse(BaseModel):
    """Response for notification operations."""
    success: bool
    message: str
    details: Optional[dict] = None


class SlackStatusResponse(BaseModel):
    """Response for Slack status check."""
    configured: bool
    webhook_configured: bool
    bot_configured: bool
    default_channel: str
    webhook_status: str
    api_status: str
    bot_info: Optional[dict] = None


# =============================================================================
# In-Memory Configuration Store (use database in production)
# =============================================================================

_slack_config: Optional[SlackConfig] = None


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
                next_run_at=datetime.now(timezone.utc),
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
    channel: Optional[str] = None,
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
