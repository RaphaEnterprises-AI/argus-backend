"""Slack integration for test notifications.

Sends test results to Slack channels via webhooks or API.
"""

import os
from dataclasses import dataclass
from datetime import datetime

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class TestSummary:
    """Summary of test results for Slack."""
    total: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    cost_usd: float
    failures: list[dict]
    report_url: str | None = None
    pr_url: str | None = None


class SlackIntegration:
    """
    Slack integration for test notifications.

    Features:
    - Send test results via webhook
    - Rich message formatting with blocks
    - Failure summaries with expandable details
    - Links to reports and PRs

    Usage:
        slack = SlackIntegration(webhook_url="https://hooks.slack.com/...")

        # Send notification
        await slack.send_test_results(
            summary=test_summary,
            channel="#qa-alerts",
        )

        # Send failure alert
        await slack.send_failure_alert(
            test_id="login-test",
            error="Element not found",
            screenshot_url="https://...",
        )
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        bot_token: str | None = None,
        default_channel: str = "#testing",
    ):
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self.bot_token = bot_token or os.environ.get("SLACK_BOT_TOKEN")
        self.default_channel = default_channel
        self.log = logger.bind(component="slack")

        if not self.webhook_url and not self.bot_token:
            self.log.warning("No Slack credentials provided - integration will be disabled")

    def _format_results_blocks(
        self,
        summary: TestSummary,
        title: str = "E2E Test Results",
    ) -> list[dict]:
        """Format test results as Slack blocks."""
        pass_rate = (summary.passed / summary.total * 100) if summary.total > 0 else 0
        status_emoji = ":white_check_mark:" if summary.failed == 0 else ":x:"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} {title}",
                    "emoji": True,
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Total:* {summary.total}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Pass Rate:* {pass_rate:.1f}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Passed:* {summary.passed} :white_check_mark:"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Failed:* {summary.failed} :x:"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Duration:* {summary.duration_seconds:.1f}s"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Cost:* ${summary.cost_usd:.4f}"
                    },
                ]
            },
        ]

        # Add failures section
        if summary.failed > 0:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*:rotating_light: Failed Tests:*"
                }
            })

            for failure in summary.failures[:5]:  # Limit to 5
                test_id = failure.get("test_id", "unknown")
                error = failure.get("error_message", "Unknown error")[:100]

                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"• `{test_id}`\n  _{error}_"
                    }
                })

            if len(summary.failures) > 5:
                blocks.append({
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"_...and {len(summary.failures) - 5} more failures_"
                        }
                    ]
                })

        # Add action buttons
        actions = []
        if summary.report_url:
            actions.append({
                "type": "button",
                "text": {"type": "plain_text", "text": "View Report", "emoji": True},
                "url": summary.report_url,
            })
        if summary.pr_url:
            actions.append({
                "type": "button",
                "text": {"type": "plain_text", "text": "View PR", "emoji": True},
                "url": summary.pr_url,
            })

        if actions:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "actions",
                "elements": actions,
            })

        # Add footer
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":robot_face: Generated by E2E Testing Agent • {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
                }
            ]
        })

        return blocks

    def _format_failure_blocks(
        self,
        test_id: str,
        error: str,
        root_cause: str | None = None,
        screenshot_url: str | None = None,
    ) -> list[dict]:
        """Format a failure alert as Slack blocks."""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":rotating_light: Test Failure Alert",
                    "emoji": True,
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Test:*\n`{test_id}`"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.utcnow().strftime('%H:%M UTC')}"
                    },
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Error:*\n```{error[:500]}```"
                }
            },
        ]

        if root_cause:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Root Cause Analysis:*\n_{root_cause}_"
                }
            })

        if screenshot_url:
            blocks.append({
                "type": "image",
                "image_url": screenshot_url,
                "alt_text": "Failure screenshot",
            })

        return blocks

    async def send_test_results(
        self,
        summary: TestSummary,
        channel: str | None = None,
        title: str = "E2E Test Results",
    ) -> bool:
        """
        Send test results to Slack.

        Args:
            summary: Test results summary
            channel: Target channel (uses default if not specified)
            title: Message title

        Returns:
            True if sent successfully
        """
        if not self.webhook_url and not self.bot_token:
            self.log.warning("Slack not configured, skipping notification")
            return False

        blocks = self._format_results_blocks(summary, title)

        return await self._send_message(
            blocks=blocks,
            channel=channel or self.default_channel,
            text=f"{title}: {summary.passed}/{summary.total} passed",
        )

    async def send_failure_alert(
        self,
        test_id: str,
        error: str,
        root_cause: str | None = None,
        screenshot_url: str | None = None,
        channel: str | None = None,
    ) -> bool:
        """
        Send immediate failure alert.

        Args:
            test_id: Failed test ID
            error: Error message
            root_cause: AI-analyzed root cause
            screenshot_url: URL to failure screenshot
            channel: Target channel

        Returns:
            True if sent successfully
        """
        if not self.webhook_url and not self.bot_token:
            self.log.warning("Slack not configured, skipping alert")
            return False

        blocks = self._format_failure_blocks(test_id, error, root_cause, screenshot_url)

        return await self._send_message(
            blocks=blocks,
            channel=channel or self.default_channel,
            text=f"Test failure: {test_id}",
        )

    async def send_simple_message(
        self,
        message: str,
        channel: str | None = None,
    ) -> bool:
        """Send a simple text message."""
        if not self.webhook_url and not self.bot_token:
            return False

        return await self._send_message(
            text=message,
            channel=channel or self.default_channel,
        )

    async def _send_message(
        self,
        text: str,
        channel: str | None = None,
        blocks: list | None = None,
    ) -> bool:
        """Send message to Slack."""
        async with httpx.AsyncClient() as client:
            # Try webhook first
            if self.webhook_url:
                try:
                    payload = {"text": text}
                    if blocks:
                        payload["blocks"] = blocks

                    response = await client.post(
                        self.webhook_url,
                        json=payload,
                    )
                    if response.status_code == 200:
                        self.log.info("Sent Slack notification via webhook")
                        return True
                    else:
                        self.log.warning("Slack webhook failed", status=response.status_code)
                except Exception as e:
                    self.log.error("Slack webhook error", error=str(e))

            # Try bot API
            if self.bot_token:
                try:
                    payload = {
                        "channel": channel,
                        "text": text,
                    }
                    if blocks:
                        payload["blocks"] = blocks

                    response = await client.post(
                        "https://slack.com/api/chat.postMessage",
                        headers={"Authorization": f"Bearer {self.bot_token}"},
                        json=payload,
                    )
                    data = response.json()
                    if data.get("ok"):
                        self.log.info("Sent Slack notification via API", channel=channel)
                        return True
                    else:
                        self.log.warning("Slack API failed", error=data.get("error"))
                except Exception as e:
                    self.log.error("Slack API error", error=str(e))

            return False

    async def send_test_started(
        self,
        test_count: int,
        pr_number: int | None = None,
        channel: str | None = None,
    ) -> bool:
        """Send notification that tests are starting."""
        pr_text = f" for PR #{pr_number}" if pr_number else ""
        message = f":runner: Starting {test_count} E2E tests{pr_text}..."

        return await self.send_simple_message(message, channel)

    async def send_test_completed(
        self,
        summary: TestSummary,
        channel: str | None = None,
    ) -> bool:
        """Send notification that tests completed."""
        return await self.send_test_results(summary, channel)


class SlackNotifier:
    """
    High-level Slack notifier for common scenarios.

    Wraps SlackIntegration with convenient methods.
    """

    def __init__(self, webhook_url: str | None = None):
        self.slack = SlackIntegration(webhook_url=webhook_url)

    async def notify_start(self, test_count: int, pr: int | None = None) -> bool:
        """Notify that tests are starting."""
        return await self.slack.send_test_started(test_count, pr)

    async def notify_complete(
        self,
        passed: int,
        failed: int,
        skipped: int,
        duration: float,
        cost: float,
        failures: list[dict],
    ) -> bool:
        """Notify that tests completed."""
        summary = TestSummary(
            total=passed + failed + skipped,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration_seconds=duration,
            cost_usd=cost,
            failures=failures,
        )
        return await self.slack.send_test_completed(summary)

    async def notify_failure(
        self,
        test_id: str,
        error: str,
        root_cause: str | None = None,
    ) -> bool:
        """Notify of immediate failure (for critical tests)."""
        return await self.slack.send_failure_alert(test_id, error, root_cause)


def create_slack_integration(
    webhook_url: str | None = None,
    bot_token: str | None = None,
) -> SlackIntegration:
    """Factory function for SlackIntegration."""
    return SlackIntegration(webhook_url=webhook_url, bot_token=bot_token)
