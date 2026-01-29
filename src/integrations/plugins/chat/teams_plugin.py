"""Microsoft Teams integration plugin.

Provides integration with Microsoft Teams for sending notifications via
Incoming Webhooks. Supports Adaptive Cards for rich message formatting.
"""

import re
import time
from datetime import datetime
from typing import Any, Optional

from src.integrations.base import (
    AuthType,
    ConnectionTestResult,
    FieldDefinition,
    IntegrationCategory,
    IntegrationCredentials,
    IntegrationMetadata,
    IntegrationPlugin,
    SyncResult,
)


class TeamsPlugin(IntegrationPlugin):
    """Microsoft Teams integration plugin.

    Supports:
    - Incoming Webhook URL authentication
    - Adaptive Cards for rich formatting
    - Action buttons (OpenUrl, etc.)
    - Facts and columns layouts
    - Images and media
    """

    # Teams webhook URLs can be from multiple domains
    TEAMS_WEBHOOK_PATTERNS = [
        re.compile(r"^https://[a-z0-9-]+\.webhook\.office\.com/"),
        re.compile(r"^https://outlook\.office\.com/webhook/"),
        re.compile(r"^https://[a-z0-9-]+\.logic\.azure\.com/"),  # Power Automate
    ]

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Teams integration metadata."""
        return IntegrationMetadata(
            slug="microsoft_teams",
            name="Microsoft Teams",
            description="Send notifications to Microsoft Teams channels via webhooks",
            category=IntegrationCategory.CHAT,
            auth_type=AuthType.WEBHOOK,
            icon_url="https://upload.wikimedia.org/wikipedia/commons/c/c9/Microsoft_Office_Teams_%282018%E2%80%93present%29.svg",
            docs_url="https://learn.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/add-incoming-webhook",
            website_url="https://www.microsoft.com/en-us/microsoft-teams/",
            features=[
                "webhooks",
                "adaptive_cards",
                "action_buttons",
                "rich_formatting",
                "facts_tables",
                "images",
            ],
            required_fields=[
                FieldDefinition(
                    name="webhook_url",
                    label="Incoming Webhook URL",
                    type="url",
                    required=True,
                    placeholder="https://xxx.webhook.office.com/webhookb2/...",
                    help_text="Teams Incoming Webhook URL from channel connector settings",
                ),
            ],
            optional_fields=[],
            supports_webhooks=True,
            supports_sync=False,  # Teams webhooks don't support data sync
            sync_resources=[],
        )

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Teams webhook connection.

        Note: Teams webhooks don't have a dedicated test endpoint,
        so we validate the URL format only.

        Args:
            credentials: Teams credentials with webhook_url.

        Returns:
            ConnectionTestResult with validation result.
        """
        start_time = time.time()

        if not credentials.webhook_url:
            return ConnectionTestResult(
                success=False,
                message="No webhook URL provided",
                error_code="missing_webhook_url",
            )

        # Validate URL format
        if not self._validate_webhook_url(credentials.webhook_url):
            return ConnectionTestResult(
                success=False,
                message="Invalid Microsoft Teams webhook URL format",
                error_code="invalid_webhook_url",
                error_details={
                    "expected_formats": [
                        "https://*.webhook.office.com/webhookb2/...",
                        "https://outlook.office.com/webhook/...",
                    ]
                },
            )

        latency_ms = (time.time() - start_time) * 1000

        # Teams doesn't provide a way to validate webhooks without sending a message
        # We can only validate the URL format
        return ConnectionTestResult(
            success=True,
            message="Webhook URL format is valid",
            latency_ms=latency_ms,
            account_info={
                "webhook_configured": True,
                "note": "Teams webhooks cannot be validated without sending a message",
            },
        )

    def _validate_webhook_url(self, url: str) -> bool:
        """Validate Teams webhook URL format.

        Args:
            url: URL to validate.

        Returns:
            True if URL matches Teams webhook pattern.
        """
        return any(pattern.match(url) for pattern in self.TEAMS_WEBHOOK_PATTERNS)

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Teams (not supported for webhooks).

        Teams webhooks are outbound-only and don't support data sync.

        Args:
            credentials: Teams credentials.
            resources: Not used.
            since: Not used.
            limit: Not used.

        Returns:
            SyncResult indicating sync is not supported.
        """
        return SyncResult(
            success=False,
            message="Microsoft Teams webhook integration does not support data sync",
            warnings=["Sync is not available for Teams webhook integrations"],
        )

    async def send_webhook(
        self,
        credentials: IntegrationCredentials,
        card: dict[str, Any],
    ) -> dict[str, Any]:
        """Send an Adaptive Card to Teams.

        Args:
            credentials: Teams credentials with webhook_url.
            card: Adaptive Card JSON object.

        Returns:
            Dict with success status.
        """
        if not credentials.webhook_url:
            return {
                "success": False,
                "error": "No webhook URL provided",
            }

        try:
            # Teams expects the card wrapped in an attachment
            payload = {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "contentUrl": None,
                        "content": card,
                    }
                ],
            }

            response = await self.http_client.post(
                credentials.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            # Teams returns 200 with "1" on success
            if response.status_code == 200:
                return {"success": True}

            # Handle common errors
            if response.status_code == 400:
                return {
                    "success": False,
                    "error": "Invalid card format",
                    "details": response.text,
                }

            if response.status_code == 404:
                return {
                    "success": False,
                    "error": "Webhook not found - connector may have been removed",
                }

            if response.status_code == 429:
                return {
                    "success": False,
                    "error": "Rate limited",
                }

            return {
                "success": False,
                "error": f"Teams API error: HTTP {response.status_code}",
                "details": response.text,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def send_simple_message(
        self,
        credentials: IntegrationCredentials,
        text: str,
    ) -> dict[str, Any]:
        """Send a simple text message to Teams.

        Args:
            credentials: Teams credentials.
            text: Message text (supports basic Markdown).

        Returns:
            Dict with send result.
        """
        card = self._create_text_card(text)
        return await self.send_webhook(credentials, card)

    def _create_text_card(self, text: str) -> dict[str, Any]:
        """Create a simple Adaptive Card with text.

        Args:
            text: Text content.

        Returns:
            Adaptive Card JSON object.
        """
        return {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {
                    "type": "TextBlock",
                    "text": text,
                    "wrap": True,
                }
            ],
        }

    async def send_adaptive_card(
        self,
        credentials: IntegrationCredentials,
        title: Optional[str] = None,
        body_text: Optional[str] = None,
        facts: Optional[list[dict[str, str]]] = None,
        actions: Optional[list[dict[str, Any]]] = None,
        image_url: Optional[str] = None,
        theme_color: str = "0078D7",
    ) -> dict[str, Any]:
        """Send a formatted Adaptive Card.

        Args:
            credentials: Teams credentials.
            title: Card title.
            body_text: Main text content.
            facts: List of fact dicts with "title" and "value".
            actions: List of action button dicts.
            image_url: URL for hero image.
            theme_color: Accent color hex (without #).

        Returns:
            Dict with send result.
        """
        body: list[dict[str, Any]] = []

        # Add image if provided
        if image_url:
            body.append({
                "type": "Image",
                "url": image_url,
                "size": "auto",
            })

        # Add title
        if title:
            body.append({
                "type": "TextBlock",
                "text": title,
                "weight": "bolder",
                "size": "large",
                "wrap": True,
            })

        # Add body text
        if body_text:
            body.append({
                "type": "TextBlock",
                "text": body_text,
                "wrap": True,
            })

        # Add facts as FactSet
        if facts:
            body.append({
                "type": "FactSet",
                "facts": [
                    {"title": f.get("title", ""), "value": f.get("value", "")}
                    for f in facts
                ],
            })

        card: dict[str, Any] = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": body,
        }

        # Add actions if provided
        if actions:
            card["actions"] = actions

        # Note: Theme color is not directly supported in Adaptive Cards
        # but can be set via the webhook payload for older message cards

        return await self.send_webhook(credentials, card)

    async def send_test_result_notification(
        self,
        credentials: IntegrationCredentials,
        test_run_id: str,
        project_name: str,
        status: str,
        passed: int,
        failed: int,
        skipped: int,
        duration_seconds: float,
        dashboard_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a formatted test result notification.

        Args:
            credentials: Teams credentials.
            test_run_id: Test run identifier.
            project_name: Name of the project.
            status: Overall status (passed, failed, etc.).
            passed: Number of passed tests.
            failed: Number of failed tests.
            skipped: Number of skipped tests.
            duration_seconds: Total duration.
            dashboard_url: Optional link to dashboard.

        Returns:
            Dict with send result.
        """
        # Determine icon and color based on status
        if status.lower() == "passed":
            icon = "CheckmarkCircle"
            color = "good"
            emoji = "tick"
        elif status.lower() == "failed":
            icon = "XCircle"
            color = "attention"
            emoji = "x"
        else:
            icon = "Warning"
            color = "warning"
            emoji = "warning"

        total = passed + failed + skipped
        duration_str = f"{duration_seconds:.1f}s" if duration_seconds < 60 else f"{duration_seconds / 60:.1f}m"

        # Build card body
        body: list[dict[str, Any]] = [
            {
                "type": "TextBlock",
                "text": f"Test Run {status.title()}",
                "weight": "bolder",
                "size": "large",
                "color": color,
            },
            {
                "type": "ColumnSet",
                "columns": [
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "Project",
                                "weight": "bolder",
                                "isSubtle": True,
                            },
                            {
                                "type": "TextBlock",
                                "text": project_name,
                            },
                        ],
                    },
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "Run ID",
                                "weight": "bolder",
                                "isSubtle": True,
                            },
                            {
                                "type": "TextBlock",
                                "text": test_run_id,
                            },
                        ],
                    },
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "Duration",
                                "weight": "bolder",
                                "isSubtle": True,
                            },
                            {
                                "type": "TextBlock",
                                "text": duration_str,
                            },
                        ],
                    },
                ],
            },
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Passed", "value": str(passed)},
                    {"title": "Failed", "value": str(failed)},
                    {"title": "Skipped", "value": str(skipped)},
                    {"title": "Total", "value": str(total)},
                ],
            },
        ]

        # Build actions
        actions: list[dict[str, Any]] = []
        if dashboard_url:
            actions.append({
                "type": "Action.OpenUrl",
                "title": "View in Dashboard",
                "url": dashboard_url,
            })

        card: dict[str, Any] = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": body,
        }

        if actions:
            card["actions"] = actions

        return await self.send_webhook(credentials, card)

    async def send_alert(
        self,
        credentials: IntegrationCredentials,
        title: str,
        message: str,
        severity: str = "info",
        action_url: Optional[str] = None,
        action_title: str = "View Details",
    ) -> dict[str, Any]:
        """Send an alert notification.

        Args:
            credentials: Teams credentials.
            title: Alert title.
            message: Alert message.
            severity: Alert severity (info, warning, error, critical).
            action_url: Optional URL for action button.
            action_title: Action button text.

        Returns:
            Dict with send result.
        """
        # Map severity to color
        severity_colors = {
            "info": "accent",
            "warning": "warning",
            "error": "attention",
            "critical": "attention",
        }
        color = severity_colors.get(severity.lower(), "default")

        body: list[dict[str, Any]] = [
            {
                "type": "TextBlock",
                "text": f"Alert: {title}",
                "weight": "bolder",
                "size": "medium",
                "color": color,
            },
            {
                "type": "TextBlock",
                "text": message,
                "wrap": True,
            },
            {
                "type": "TextBlock",
                "text": f"Severity: {severity.upper()}",
                "isSubtle": True,
                "size": "small",
            },
        ]

        card: dict[str, Any] = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": body,
        }

        if action_url:
            card["actions"] = [
                {
                    "type": "Action.OpenUrl",
                    "title": action_title,
                    "url": action_url,
                }
            ]

        return await self.send_webhook(credentials, card)
