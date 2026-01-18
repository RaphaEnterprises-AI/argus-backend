"""Enhanced Slack notifications for E2E Testing Agent.

Provides rich Slack notifications using Block Kit for:
- Test run results with color-coded status
- Failure alerts with stack trace previews
- Scheduled run reminders
- Quality score reports
- Action buttons for interactive workflows

Supports both Webhook URL (simple) and Bot Token (full features).
"""

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class NotificationStatus(str, Enum):
    """Status types for notifications."""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    INFO = "info"


@dataclass
class SlackConfig:
    """Configuration for Slack integration."""
    webhook_url: str | None = None
    bot_token: str | None = None
    default_channel: str = "#testing"
    timeout_seconds: float = 30.0
    retry_attempts: int = 3

    @classmethod
    def from_env(cls) -> "SlackConfig":
        """Create configuration from environment variables."""
        return cls(
            webhook_url=os.environ.get("SLACK_WEBHOOK_URL"),
            bot_token=os.environ.get("SLACK_BOT_TOKEN"),
            default_channel=os.environ.get("SLACK_DEFAULT_CHANNEL", "#testing"),
        )


@dataclass
class TestResult:
    """Test run result data for notifications."""
    total: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    cost_usd: float = 0.0
    failures: list[dict] = field(default_factory=list)
    report_url: str | None = None
    pr_url: str | None = None
    pr_number: int | None = None
    branch: str | None = None
    commit_sha: str | None = None
    job_id: str | None = None

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage."""
        return (self.passed / self.total * 100) if self.total > 0 else 0.0

    @property
    def status(self) -> NotificationStatus:
        """Determine overall status."""
        if self.failed == 0:
            return NotificationStatus.SUCCESS
        elif self.failed > self.total / 2:
            return NotificationStatus.FAILURE
        else:
            return NotificationStatus.WARNING


@dataclass
class FailureDetails:
    """Detailed failure information for alerts."""
    test_id: str
    test_name: str
    error_message: str
    stack_trace: str | None = None
    screenshot_url: str | None = None
    root_cause: str | None = None
    component: str | None = None
    url: str | None = None
    duration_ms: int | None = None
    retry_count: int = 0


@dataclass
class ScheduleInfo:
    """Scheduled run information."""
    schedule_id: str
    schedule_name: str
    next_run_at: datetime
    test_suite: str
    estimated_duration_minutes: int = 30
    environment: str = "staging"
    notify_channel: str | None = None


@dataclass
class QualityReport:
    """Quality score report data."""
    project_id: str
    project_name: str
    overall_score: float
    grade: str
    test_coverage: float
    error_count: int
    resolved_count: int
    risk_level: str
    trends: dict = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    report_url: str | None = None


class SlackNotifier:
    """
    Slack notifier with Block Kit formatting and async support.

    Features:
    - Rich message formatting with Slack Block Kit
    - Color-coded status indicators
    - Interactive action buttons
    - Support for both Webhook and Bot Token
    - Async HTTP requests with httpx
    - Thread support for organizing messages
    - Reaction support (with Bot Token)

    Usage:
        notifier = SlackNotifier()

        # Send test results
        await notifier.send_test_result(TestResult(
            total=100,
            passed=95,
            failed=5,
            skipped=0,
            duration_seconds=120.5,
        ))

        # Send failure alert
        await notifier.send_failure_alert(FailureDetails(
            test_id="login-test",
            test_name="User Login Flow",
            error_message="Element not found: #login-button",
        ))
    """

    # Color constants for attachment colors
    COLOR_SUCCESS = "#36a64f"  # Green
    COLOR_FAILURE = "#dc3545"  # Red
    COLOR_WARNING = "#ffc107"  # Yellow
    COLOR_INFO = "#17a2b8"     # Blue

    def __init__(
        self,
        config: SlackConfig | None = None,
        webhook_url: str | None = None,
        bot_token: str | None = None,
        default_channel: str | None = None,
    ):
        """
        Initialize SlackNotifier.

        Args:
            config: SlackConfig instance (takes precedence)
            webhook_url: Slack webhook URL (fallback)
            bot_token: Slack bot token for API access (fallback)
            default_channel: Default channel for messages (fallback)
        """
        if config:
            self.config = config
        else:
            self.config = SlackConfig(
                webhook_url=webhook_url or os.environ.get("SLACK_WEBHOOK_URL"),
                bot_token=bot_token or os.environ.get("SLACK_BOT_TOKEN"),
                default_channel=default_channel or os.environ.get("SLACK_DEFAULT_CHANNEL", "#testing"),
            )

        self.log = logger.bind(component="slack_notifier")

        if not self.config.webhook_url and not self.config.bot_token:
            self.log.warning(
                "No Slack credentials configured. "
                "Set SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN environment variables."
            )

    @property
    def is_configured(self) -> bool:
        """Check if Slack is properly configured."""
        return bool(self.config.webhook_url or self.config.bot_token)

    def _get_status_color(self, status: NotificationStatus) -> str:
        """Get color for status."""
        colors = {
            NotificationStatus.SUCCESS: self.COLOR_SUCCESS,
            NotificationStatus.FAILURE: self.COLOR_FAILURE,
            NotificationStatus.WARNING: self.COLOR_WARNING,
            NotificationStatus.INFO: self.COLOR_INFO,
        }
        return colors.get(status, self.COLOR_INFO)

    def _get_status_emoji(self, status: NotificationStatus) -> str:
        """Get emoji for status."""
        emojis = {
            NotificationStatus.SUCCESS: ":white_check_mark:",
            NotificationStatus.FAILURE: ":x:",
            NotificationStatus.WARNING: ":warning:",
            NotificationStatus.INFO: ":information_source:",
        }
        return emojis.get(status, ":grey_question:")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def _truncate(self, text: str, max_length: int = 500) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

    def _build_test_result_blocks(self, result: TestResult, title: str = "E2E Test Results") -> list[dict]:
        """Build Slack blocks for test results."""
        status = result.status
        emoji = self._get_status_emoji(status)

        blocks = [
            # Header
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title}",
                    "emoji": True,
                }
            },
            # Summary section
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Total Tests:*\n{result.total}"},
                    {"type": "mrkdwn", "text": f"*Pass Rate:*\n{result.pass_rate:.1f}%"},
                    {"type": "mrkdwn", "text": f"*Passed:* {result.passed} :white_check_mark:"},
                    {"type": "mrkdwn", "text": f"*Failed:* {result.failed} :x:"},
                    {"type": "mrkdwn", "text": f"*Skipped:* {result.skipped} :fast_forward:"},
                    {"type": "mrkdwn", "text": f"*Duration:* {self._format_duration(result.duration_seconds)}"},
                ]
            },
        ]

        # Add cost if available
        if result.cost_usd > 0:
            blocks.append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*AI Cost:* ${result.cost_usd:.4f}"},
                ]
            })

        # Add branch/commit info if available
        context_parts = []
        if result.branch:
            context_parts.append(f":git: Branch: `{result.branch}`")
        if result.commit_sha:
            context_parts.append(f"Commit: `{result.commit_sha[:8]}`")
        if result.pr_number:
            pr_text = f"PR #{result.pr_number}"
            if result.pr_url:
                pr_text = f"<{result.pr_url}|PR #{result.pr_number}>"
            context_parts.append(pr_text)

        if context_parts:
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": " | ".join(context_parts)}]
            })

        # Add failures section if there are failures
        if result.failed > 0 and result.failures:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*:rotating_light: Failed Tests:*"}
            })

            # Show up to 5 failures
            for failure in result.failures[:5]:
                test_id = failure.get("test_id", "unknown")
                error = self._truncate(failure.get("error_message", "Unknown error"), 100)
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*`{test_id}`*\n_{error}_"}
                })

            if len(result.failures) > 5:
                blocks.append({
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"_...and {len(result.failures) - 5} more failures_"}
                    ]
                })

        # Add action buttons
        actions = []
        if result.report_url:
            actions.append({
                "type": "button",
                "text": {"type": "plain_text", "text": "View Report", "emoji": True},
                "url": result.report_url,
                "style": "primary",
            })
        if result.pr_url:
            actions.append({
                "type": "button",
                "text": {"type": "plain_text", "text": "View PR", "emoji": True},
                "url": result.pr_url,
            })
        if result.job_id:
            actions.append({
                "type": "button",
                "text": {"type": "plain_text", "text": "Rerun Tests", "emoji": True},
                "action_id": f"rerun_tests_{result.job_id}",
            })

        if actions:
            blocks.append({"type": "divider"})
            blocks.append({"type": "actions", "elements": actions})

        # Footer
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":robot_face: Argus E2E Testing Agent | {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"
                }
            ]
        })

        return blocks

    def _build_failure_alert_blocks(self, failure: FailureDetails) -> list[dict]:
        """Build Slack blocks for failure alert."""
        blocks = [
            # Header
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":rotating_light: Test Failure Alert",
                    "emoji": True,
                }
            },
            # Test info
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Test:*\n`{failure.test_id}`"},
                    {"type": "mrkdwn", "text": f"*Name:*\n{failure.test_name}"},
                ]
            },
        ]

        # Add component/URL if available
        if failure.component or failure.url:
            fields = []
            if failure.component:
                fields.append({"type": "mrkdwn", "text": f"*Component:*\n{failure.component}"})
            if failure.url:
                fields.append({"type": "mrkdwn", "text": f"*URL:*\n{failure.url}"})
            blocks.append({"type": "section", "fields": fields})

        # Error message
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Error:*\n```{self._truncate(failure.error_message, 500)}```"
            }
        })

        # Stack trace preview (collapsed by default via truncation)
        if failure.stack_trace:
            stack_preview = self._truncate(failure.stack_trace, 300)
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Stack Trace Preview:*\n```{stack_preview}```"
                }
            })

        # Root cause analysis if available
        if failure.root_cause:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*:brain: AI Root Cause Analysis:*\n_{failure.root_cause}_"
                }
            })

        # Screenshot if available
        if failure.screenshot_url:
            blocks.append({
                "type": "image",
                "image_url": failure.screenshot_url,
                "alt_text": "Failure screenshot",
            })

        # Metadata
        meta_parts = []
        if failure.duration_ms:
            meta_parts.append(f"Duration: {failure.duration_ms}ms")
        if failure.retry_count > 0:
            meta_parts.append(f"Retries: {failure.retry_count}")

        if meta_parts:
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": " | ".join(meta_parts)}]
            })

        # Actions
        actions = []
        actions.append({
            "type": "button",
            "text": {"type": "plain_text", "text": "View Details", "emoji": True},
            "action_id": f"view_failure_{failure.test_id}",
            "style": "primary",
        })
        actions.append({
            "type": "button",
            "text": {"type": "plain_text", "text": "Ignore", "emoji": True},
            "action_id": f"ignore_failure_{failure.test_id}",
        })

        blocks.append({"type": "actions", "elements": actions})

        # Footer
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":robot_face: Argus E2E Testing Agent | {datetime.now(UTC).strftime('%H:%M UTC')}"
                }
            ]
        })

        return blocks

    def _build_schedule_reminder_blocks(self, schedule: ScheduleInfo) -> list[dict]:
        """Build Slack blocks for schedule reminder."""
        time_until = schedule.next_run_at - datetime.now(UTC)
        minutes_until = int(time_until.total_seconds() / 60)

        blocks = [
            # Header
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":alarm_clock: Scheduled Test Run Reminder",
                    "emoji": True,
                }
            },
            # Schedule info
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Schedule:*\n{schedule.schedule_name}"},
                    {"type": "mrkdwn", "text": f"*Test Suite:*\n{schedule.test_suite}"},
                    {"type": "mrkdwn", "text": f"*Environment:*\n{schedule.environment}"},
                    {"type": "mrkdwn", "text": f"*Starts In:*\n{minutes_until} minutes"},
                ]
            },
            # Time info
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Scheduled Time:* {schedule.next_run_at.strftime('%Y-%m-%d %H:%M UTC')}\n*Estimated Duration:* ~{schedule.estimated_duration_minutes} minutes"
                }
            },
            # Actions
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Run Now", "emoji": True},
                        "action_id": f"run_now_{schedule.schedule_id}",
                        "style": "primary",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Skip This Run", "emoji": True},
                        "action_id": f"skip_run_{schedule.schedule_id}",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Schedule", "emoji": True},
                        "action_id": f"view_schedule_{schedule.schedule_id}",
                    },
                ]
            },
            # Footer
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": ":robot_face: Argus E2E Testing Agent"}
                ]
            },
        ]

        return blocks

    def _build_quality_report_blocks(self, report: QualityReport) -> list[dict]:
        """Build Slack blocks for quality report."""
        # Determine status based on score
        if report.overall_score >= 80:
            grade_color = ":large_green_circle:"
        elif report.overall_score >= 60:
            grade_color = ":large_yellow_circle:"
        else:
            grade_color = ":red_circle:"

        blocks = [
            # Header
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f":bar_chart: Quality Report - {report.project_name}",
                    "emoji": True,
                }
            },
            # Score section
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{grade_color} *Overall Score: {report.overall_score:.1f}/100* (Grade: *{report.grade}*)"
                }
            },
            # Metrics
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Test Coverage:*\n{report.test_coverage:.1f}%"},
                    {"type": "mrkdwn", "text": f"*Risk Level:*\n{report.risk_level.capitalize()}"},
                    {"type": "mrkdwn", "text": f"*Active Errors:*\n{report.error_count}"},
                    {"type": "mrkdwn", "text": f"*Resolved:*\n{report.resolved_count}"},
                ]
            },
        ]

        # Add trends if available
        if report.trends:
            trend_text = []
            for metric, change in report.trends.items():
                arrow = ":arrow_up:" if change > 0 else ":arrow_down:" if change < 0 else ":arrow_right:"
                trend_text.append(f"{metric}: {arrow} {abs(change):.1f}%")

            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Trends (vs last week):*\n{' | '.join(trend_text)}"}
            })

        # Recommendations
        if report.recommendations:
            rec_text = "\n".join([f"- {rec}" for rec in report.recommendations[:3]])
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*:bulb: Recommendations:*\n{rec_text}"}
            })

        # Actions
        actions = []
        if report.report_url:
            actions.append({
                "type": "button",
                "text": {"type": "plain_text", "text": "View Full Report", "emoji": True},
                "url": report.report_url,
                "style": "primary",
            })
        actions.append({
            "type": "button",
            "text": {"type": "plain_text", "text": "Generate Tests", "emoji": True},
            "action_id": f"generate_tests_{report.project_id}",
        })

        if actions:
            blocks.append({"type": "actions", "elements": actions})

        # Footer
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":robot_face: Argus Quality Intelligence | {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"
                }
            ]
        })

        return blocks

    async def send_message(
        self,
        channel: str | None = None,
        message: str = "",
        blocks: list[dict] | None = None,
        attachments: list[dict] | None = None,
        thread_ts: str | None = None,
        reply_broadcast: bool = False,
    ) -> bool:
        """
        Send a generic message to Slack.

        Args:
            channel: Target channel (uses default if not specified)
            message: Fallback text message
            blocks: Slack Block Kit blocks
            attachments: Legacy attachments
            thread_ts: Thread timestamp for replies
            reply_broadcast: Also post to channel when replying to thread

        Returns:
            True if sent successfully
        """
        if not self.is_configured:
            self.log.warning("Slack not configured, skipping message")
            return False

        target_channel = channel or self.config.default_channel

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            # Try webhook first (simpler, doesn't need channel for default)
            if self.config.webhook_url:
                return await self._send_via_webhook(client, message, blocks, attachments)

            # Use Bot Token API
            if self.config.bot_token:
                return await self._send_via_api(
                    client,
                    target_channel,
                    message,
                    blocks,
                    attachments,
                    thread_ts,
                    reply_broadcast,
                )

        return False

    async def _send_via_webhook(
        self,
        client: httpx.AsyncClient,
        text: str,
        blocks: list[dict] | None,
        attachments: list[dict] | None,
    ) -> bool:
        """Send message via webhook URL."""
        payload: dict[str, Any] = {"text": text}

        if blocks:
            payload["blocks"] = blocks
        if attachments:
            payload["attachments"] = attachments

        for attempt in range(self.config.retry_attempts):
            try:
                response = await client.post(
                    self.config.webhook_url,
                    json=payload,
                )

                if response.status_code == 200:
                    self.log.info("Sent Slack notification via webhook")
                    return True

                self.log.warning(
                    "Slack webhook failed",
                    status=response.status_code,
                    response=response.text[:200],
                    attempt=attempt + 1,
                )

            except httpx.RequestError as e:
                self.log.error(
                    "Slack webhook request error",
                    error=str(e),
                    attempt=attempt + 1,
                )

        return False

    async def _send_via_api(
        self,
        client: httpx.AsyncClient,
        channel: str,
        text: str,
        blocks: list[dict] | None,
        attachments: list[dict] | None,
        thread_ts: str | None,
        reply_broadcast: bool,
    ) -> bool:
        """Send message via Slack API with Bot Token."""
        payload: dict[str, Any] = {
            "channel": channel,
            "text": text,
        }

        if blocks:
            payload["blocks"] = blocks
        if attachments:
            payload["attachments"] = attachments
        if thread_ts:
            payload["thread_ts"] = thread_ts
            if reply_broadcast:
                payload["reply_broadcast"] = True

        headers = {
            "Authorization": f"Bearer {self.config.bot_token}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.config.retry_attempts):
            try:
                response = await client.post(
                    "https://slack.com/api/chat.postMessage",
                    headers=headers,
                    json=payload,
                )

                data = response.json()

                if data.get("ok"):
                    self.log.info(
                        "Sent Slack notification via API",
                        channel=channel,
                        ts=data.get("ts"),
                    )
                    return True

                self.log.warning(
                    "Slack API failed",
                    error=data.get("error"),
                    attempt=attempt + 1,
                )

                # Don't retry on auth errors
                if data.get("error") in ("invalid_auth", "not_authed", "token_revoked"):
                    break

            except httpx.RequestError as e:
                self.log.error(
                    "Slack API request error",
                    error=str(e),
                    attempt=attempt + 1,
                )

        return False

    async def add_reaction(
        self,
        channel: str,
        timestamp: str,
        emoji: str,
    ) -> bool:
        """
        Add a reaction to a message (requires Bot Token).

        Args:
            channel: Channel ID where the message is
            timestamp: Message timestamp
            emoji: Emoji name without colons (e.g., "thumbsup")

        Returns:
            True if successful
        """
        if not self.config.bot_token:
            self.log.warning("Bot token required for reactions")
            return False

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                "https://slack.com/api/reactions.add",
                headers={
                    "Authorization": f"Bearer {self.config.bot_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "channel": channel,
                    "timestamp": timestamp,
                    "name": emoji,
                },
            )

            data = response.json()

            if data.get("ok"):
                return True

            # already_reacted is not an error
            if data.get("error") == "already_reacted":
                return True

            self.log.warning("Failed to add reaction", error=data.get("error"))
            return False

    async def send_test_result(
        self,
        result: TestResult,
        channel: str | None = None,
        title: str = "E2E Test Results",
    ) -> bool:
        """
        Send test run results notification.

        Args:
            result: Test result data
            channel: Target channel (optional)
            title: Custom title for the notification

        Returns:
            True if sent successfully
        """
        blocks = self._build_test_result_blocks(result, title)

        # Add attachment for color indicator
        color = self._get_status_color(result.status)
        attachments = [{"color": color, "blocks": []}]

        fallback_text = f"{title}: {result.passed}/{result.total} passed ({result.failed} failed)"

        return await self.send_message(
            channel=channel,
            message=fallback_text,
            blocks=blocks,
            attachments=attachments,
        )

    async def send_failure_alert(
        self,
        failure: FailureDetails,
        channel: str | None = None,
    ) -> bool:
        """
        Send immediate failure alert notification.

        Args:
            failure: Failure details
            channel: Target channel (optional)

        Returns:
            True if sent successfully
        """
        blocks = self._build_failure_alert_blocks(failure)

        # Red attachment for failure
        attachments = [{"color": self.COLOR_FAILURE, "blocks": []}]

        fallback_text = f"Test failure: {failure.test_name} - {failure.error_message[:100]}"

        return await self.send_message(
            channel=channel,
            message=fallback_text,
            blocks=blocks,
            attachments=attachments,
        )

    async def send_schedule_reminder(
        self,
        schedule: ScheduleInfo,
        channel: str | None = None,
    ) -> bool:
        """
        Send scheduled run reminder notification.

        Args:
            schedule: Schedule information
            channel: Target channel (optional, uses schedule's notify_channel if set)

        Returns:
            True if sent successfully
        """
        target_channel = channel or schedule.notify_channel
        blocks = self._build_schedule_reminder_blocks(schedule)

        # Blue attachment for info
        attachments = [{"color": self.COLOR_INFO, "blocks": []}]

        fallback_text = f"Scheduled test run '{schedule.schedule_name}' starting in {int((schedule.next_run_at - datetime.now(UTC)).total_seconds() / 60)} minutes"

        return await self.send_message(
            channel=target_channel,
            message=fallback_text,
            blocks=blocks,
            attachments=attachments,
        )

    async def send_quality_report(
        self,
        report: QualityReport,
        channel: str | None = None,
    ) -> bool:
        """
        Send quality score summary notification.

        Args:
            report: Quality report data
            channel: Target channel (optional)

        Returns:
            True if sent successfully
        """
        blocks = self._build_quality_report_blocks(report)

        # Color based on score
        if report.overall_score >= 80:
            color = self.COLOR_SUCCESS
        elif report.overall_score >= 60:
            color = self.COLOR_WARNING
        else:
            color = self.COLOR_FAILURE

        attachments = [{"color": color, "blocks": []}]

        fallback_text = f"Quality Report for {report.project_name}: Score {report.overall_score:.1f}/100 (Grade: {report.grade})"

        return await self.send_message(
            channel=channel,
            message=fallback_text,
            blocks=blocks,
            attachments=attachments,
        )

    async def check_connection(self) -> dict:
        """
        Check Slack connection status.

        Returns:
            Dict with connection status information
        """
        result = {
            "configured": self.is_configured,
            "webhook_configured": bool(self.config.webhook_url),
            "bot_configured": bool(self.config.bot_token),
            "default_channel": self.config.default_channel,
            "webhook_status": "not_configured",
            "api_status": "not_configured",
            "bot_info": None,
        }

        if not self.is_configured:
            return result

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            # Test webhook
            if self.config.webhook_url:
                try:
                    # Slack webhooks respond with "ok" to properly formed requests
                    # We can't really test without sending, so just verify URL format
                    result["webhook_status"] = "configured"
                except Exception as e:
                    result["webhook_status"] = f"error: {str(e)}"

            # Test API with auth.test
            if self.config.bot_token:
                try:
                    response = await client.post(
                        "https://slack.com/api/auth.test",
                        headers={
                            "Authorization": f"Bearer {self.config.bot_token}",
                            "Content-Type": "application/json",
                        },
                    )
                    data = response.json()

                    if data.get("ok"):
                        result["api_status"] = "connected"
                        result["bot_info"] = {
                            "team": data.get("team"),
                            "user": data.get("user"),
                            "team_id": data.get("team_id"),
                            "user_id": data.get("user_id"),
                        }
                    else:
                        result["api_status"] = f"error: {data.get('error')}"

                except Exception as e:
                    result["api_status"] = f"error: {str(e)}"

        return result


# Factory function for easy instantiation
def create_slack_notifier(
    webhook_url: str | None = None,
    bot_token: str | None = None,
    default_channel: str | None = None,
) -> SlackNotifier:
    """
    Create a SlackNotifier instance.

    Args:
        webhook_url: Slack webhook URL (or use SLACK_WEBHOOK_URL env var)
        bot_token: Slack bot token (or use SLACK_BOT_TOKEN env var)
        default_channel: Default channel (or use SLACK_DEFAULT_CHANNEL env var)

    Returns:
        Configured SlackNotifier instance
    """
    return SlackNotifier(
        webhook_url=webhook_url,
        bot_token=bot_token,
        default_channel=default_channel,
    )
