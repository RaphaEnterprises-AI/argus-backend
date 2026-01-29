"""Google Chat integration plugin.

Provides integration with Google Chat for sending notifications via webhooks.
Supports Cards v2 API for rich message formatting.
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


class GoogleChatPlugin(IntegrationPlugin):
    """Google Chat integration plugin.

    Supports:
    - Webhook URL based authentication
    - Cards v2 API for rich formatting
    - Buttons and action widgets
    - Images and icons
    - Formatted text sections
    """

    GOOGLE_CHAT_WEBHOOK_PATTERN = re.compile(
        r"^https://chat\.googleapis\.com/v1/spaces/[A-Za-z0-9_-]+/messages\?key=[A-Za-z0-9_-]+&token=[A-Za-z0-9%_-]+$"
    )

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Google Chat integration metadata."""
        return IntegrationMetadata(
            slug="google_chat",
            name="Google Chat",
            description="Send notifications to Google Chat spaces via webhooks",
            category=IntegrationCategory.CHAT,
            auth_type=AuthType.WEBHOOK,
            icon_url="https://fonts.gstatic.com/s/i/productlogos/chat_2020q4/v8/192px.svg",
            docs_url="https://developers.google.com/chat/how-tos/webhooks",
            website_url="https://chat.google.com/",
            features=[
                "webhooks",
                "cards_v2",
                "action_buttons",
                "rich_formatting",
                "images",
                "icons",
            ],
            required_fields=[
                FieldDefinition(
                    name="webhook_url",
                    label="Webhook URL",
                    type="url",
                    required=True,
                    placeholder="https://chat.googleapis.com/v1/spaces/.../messages?key=...&token=...",
                    help_text="Google Chat webhook URL from space settings",
                ),
            ],
            optional_fields=[],
            supports_webhooks=True,
            supports_sync=False,  # Google Chat webhooks don't support data sync
            sync_resources=[],
        )

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Google Chat webhook connection.

        Note: Google Chat webhooks don't have a dedicated test endpoint,
        so we validate the URL format only.

        Args:
            credentials: Google Chat credentials with webhook_url.

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
                message="Invalid Google Chat webhook URL format",
                error_code="invalid_webhook_url",
                error_details={
                    "expected_format": "https://chat.googleapis.com/v1/spaces/{space}/messages?key={key}&token={token}"
                },
            )

        latency_ms = (time.time() - start_time) * 1000

        # Google Chat doesn't provide a way to validate webhooks without sending a message
        return ConnectionTestResult(
            success=True,
            message="Webhook URL format is valid",
            latency_ms=latency_ms,
            account_info={
                "webhook_configured": True,
                "note": "Google Chat webhooks cannot be validated without sending a message",
            },
        )

    def _validate_webhook_url(self, url: str) -> bool:
        """Validate Google Chat webhook URL format.

        Args:
            url: URL to validate.

        Returns:
            True if URL matches Google Chat webhook pattern.
        """
        # Basic validation - check if it's a Google Chat URL
        if not url.startswith("https://chat.googleapis.com/"):
            return False

        # Check for required parameters
        if "key=" not in url or "token=" not in url:
            return False

        return True

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Google Chat (not supported for webhooks).

        Google Chat webhooks are outbound-only and don't support data sync.

        Args:
            credentials: Google Chat credentials.
            resources: Not used.
            since: Not used.
            limit: Not used.

        Returns:
            SyncResult indicating sync is not supported.
        """
        return SyncResult(
            success=False,
            message="Google Chat webhook integration does not support data sync",
            warnings=["Sync is not available for Google Chat webhook integrations"],
        )

    async def send_webhook(
        self,
        credentials: IntegrationCredentials,
        text: Optional[str] = None,
        cards: Optional[list[dict[str, Any]]] = None,
        thread_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a message to Google Chat.

        Args:
            credentials: Google Chat credentials with webhook_url.
            text: Simple text message.
            cards: List of Cards v2 card objects.
            thread_key: Optional thread key to reply in a thread.

        Returns:
            Dict with success status.
        """
        if not credentials.webhook_url:
            return {
                "success": False,
                "error": "No webhook URL provided",
            }

        if not text and not cards:
            return {
                "success": False,
                "error": "Either text or cards must be provided",
            }

        try:
            payload: dict[str, Any] = {}

            if text:
                payload["text"] = text

            if cards:
                payload["cardsV2"] = cards

            # Add thread key if specified
            url = credentials.webhook_url
            if thread_key:
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}threadKey={thread_key}"

            response = await self.http_client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message_name": data.get("name"),
                    "space": data.get("space", {}).get("name"),
                    "thread": data.get("thread", {}).get("name"),
                }

            # Handle common errors
            if response.status_code == 400:
                error_data = response.json() if response.content else {}
                return {
                    "success": False,
                    "error": "Invalid message format",
                    "details": error_data.get("error", {}).get("message", response.text),
                }

            if response.status_code == 401:
                return {
                    "success": False,
                    "error": "Invalid webhook credentials",
                }

            if response.status_code == 403:
                return {
                    "success": False,
                    "error": "Webhook access denied - check space permissions",
                }

            if response.status_code == 404:
                return {
                    "success": False,
                    "error": "Webhook not found - space may have been deleted",
                }

            return {
                "success": False,
                "error": f"Google Chat API error: HTTP {response.status_code}",
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
        thread_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a simple text message.

        Args:
            credentials: Google Chat credentials.
            text: Message text (supports limited formatting).
            thread_key: Optional thread key.

        Returns:
            Dict with send result.
        """
        return await self.send_webhook(
            credentials=credentials,
            text=text,
            thread_key=thread_key,
        )

    async def send_card(
        self,
        credentials: IntegrationCredentials,
        card_id: str,
        header_title: Optional[str] = None,
        header_subtitle: Optional[str] = None,
        header_image_url: Optional[str] = None,
        sections: Optional[list[dict[str, Any]]] = None,
        thread_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a Cards v2 card message.

        Args:
            credentials: Google Chat credentials.
            card_id: Unique card identifier.
            header_title: Card header title.
            header_subtitle: Card header subtitle.
            header_image_url: URL for header image.
            sections: List of section objects.
            thread_key: Optional thread key.

        Returns:
            Dict with send result.
        """
        card: dict[str, Any] = {
            "cardId": card_id,
            "card": {},
        }

        # Build header
        if header_title or header_subtitle or header_image_url:
            header: dict[str, Any] = {}
            if header_title:
                header["title"] = header_title
            if header_subtitle:
                header["subtitle"] = header_subtitle
            if header_image_url:
                header["imageUrl"] = header_image_url
                header["imageType"] = "CIRCLE"
            card["card"]["header"] = header

        # Add sections
        if sections:
            card["card"]["sections"] = sections

        return await self.send_webhook(
            credentials=credentials,
            cards=[card],
            thread_key=thread_key,
        )

    def _create_text_paragraph(self, text: str) -> dict[str, Any]:
        """Create a text paragraph widget.

        Args:
            text: Text content (supports HTML subset).

        Returns:
            Text paragraph widget.
        """
        return {
            "textParagraph": {
                "text": text,
            }
        }

    def _create_decorated_text(
        self,
        text: str,
        top_label: Optional[str] = None,
        bottom_label: Optional[str] = None,
        icon_name: Optional[str] = None,
        button_text: Optional[str] = None,
        button_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a decorated text widget.

        Args:
            text: Main text content.
            top_label: Label above text.
            bottom_label: Label below text.
            icon_name: Material icon name.
            button_text: Optional button text.
            button_url: Optional button URL.

        Returns:
            Decorated text widget.
        """
        widget: dict[str, Any] = {
            "decoratedText": {
                "text": text,
            }
        }

        if top_label:
            widget["decoratedText"]["topLabel"] = top_label
        if bottom_label:
            widget["decoratedText"]["bottomLabel"] = bottom_label
        if icon_name:
            widget["decoratedText"]["startIcon"] = {
                "knownIcon": icon_name,
            }
        if button_text and button_url:
            widget["decoratedText"]["button"] = {
                "text": button_text,
                "onClick": {
                    "openLink": {
                        "url": button_url,
                    }
                },
            }

        return widget

    def _create_button_list(
        self,
        buttons: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Create a button list widget.

        Args:
            buttons: List of button dicts with "text" and "url".

        Returns:
            Button list widget.
        """
        return {
            "buttonList": {
                "buttons": [
                    {
                        "text": btn.get("text", ""),
                        "onClick": {
                            "openLink": {
                                "url": btn.get("url", ""),
                            }
                        },
                    }
                    for btn in buttons
                ]
            }
        }

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
            credentials: Google Chat credentials.
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
        # Determine icon based on status
        if status.lower() == "passed":
            icon = "CONFIRMATION_NUMBER_ICON"
            status_text = "<font color=\"#36a64f\">PASSED</font>"
        elif status.lower() == "failed":
            icon = "TICKET"
            status_text = "<font color=\"#dc3545\">FAILED</font>"
        else:
            icon = "BOOKMARK"
            status_text = "<font color=\"#ffc107\">WARNING</font>"

        total = passed + failed + skipped
        duration_str = f"{duration_seconds:.1f}s" if duration_seconds < 60 else f"{duration_seconds / 60:.1f}m"

        # Build sections
        sections: list[dict[str, Any]] = [
            {
                "header": "Test Results",
                "collapsible": False,
                "widgets": [
                    self._create_decorated_text(
                        text=project_name,
                        top_label="Project",
                        icon_name="DESCRIPTION",
                    ),
                    self._create_decorated_text(
                        text=test_run_id,
                        top_label="Run ID",
                        icon_name="BOOKMARK",
                    ),
                    self._create_decorated_text(
                        text=duration_str,
                        top_label="Duration",
                        icon_name="CLOCK",
                    ),
                ],
            },
            {
                "header": "Statistics",
                "collapsible": False,
                "widgets": [
                    {
                        "columns": {
                            "columnItems": [
                                {
                                    "horizontalSizeStyle": "FILL_AVAILABLE_SPACE",
                                    "horizontalAlignment": "CENTER",
                                    "verticalAlignment": "CENTER",
                                    "widgets": [
                                        self._create_text_paragraph(f"<b>Passed</b><br>{passed}"),
                                    ],
                                },
                                {
                                    "horizontalSizeStyle": "FILL_AVAILABLE_SPACE",
                                    "horizontalAlignment": "CENTER",
                                    "verticalAlignment": "CENTER",
                                    "widgets": [
                                        self._create_text_paragraph(f"<b>Failed</b><br>{failed}"),
                                    ],
                                },
                                {
                                    "horizontalSizeStyle": "FILL_AVAILABLE_SPACE",
                                    "horizontalAlignment": "CENTER",
                                    "verticalAlignment": "CENTER",
                                    "widgets": [
                                        self._create_text_paragraph(f"<b>Skipped</b><br>{skipped}"),
                                    ],
                                },
                            ],
                        },
                    },
                ],
            },
        ]

        # Add dashboard button if URL provided
        if dashboard_url:
            sections.append({
                "widgets": [
                    self._create_button_list([
                        {"text": "View in Dashboard", "url": dashboard_url},
                    ]),
                ],
            })

        return await self.send_card(
            credentials=credentials,
            card_id=f"test-result-{test_run_id}",
            header_title=f"Test Run {status.title()}",
            header_subtitle=f"{passed}/{total} tests passed",
            sections=sections,
        )

    async def send_alert(
        self,
        credentials: IntegrationCredentials,
        title: str,
        message: str,
        severity: str = "info",
        action_url: Optional[str] = None,
        action_title: str = "View Details",
        thread_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send an alert notification.

        Args:
            credentials: Google Chat credentials.
            title: Alert title.
            message: Alert message.
            severity: Alert severity (info, warning, error, critical).
            action_url: Optional URL for action button.
            action_title: Action button text.
            thread_key: Optional thread key.

        Returns:
            Dict with send result.
        """
        # Map severity to icon
        severity_icons = {
            "info": "DESCRIPTION",
            "warning": "STAR",
            "error": "TICKET",
            "critical": "TICKET",
        }
        icon = severity_icons.get(severity.lower(), "DESCRIPTION")

        sections: list[dict[str, Any]] = [
            {
                "widgets": [
                    self._create_text_paragraph(message),
                    self._create_decorated_text(
                        text=severity.upper(),
                        top_label="Severity",
                        icon_name=icon,
                    ),
                ],
            },
        ]

        if action_url:
            sections.append({
                "widgets": [
                    self._create_button_list([
                        {"text": action_title, "url": action_url},
                    ]),
                ],
            })

        return await self.send_card(
            credentials=credentials,
            card_id=f"alert-{int(time.time())}",
            header_title=f"Alert: {title}",
            sections=sections,
            thread_key=thread_key,
        )

    async def send_deployment_notification(
        self,
        credentials: IntegrationCredentials,
        environment: str,
        version: str,
        deployer: str,
        status: str,
        commit_sha: Optional[str] = None,
        commit_message: Optional[str] = None,
        deployment_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a deployment notification.

        Args:
            credentials: Google Chat credentials.
            environment: Deployment environment (prod, staging, etc.).
            version: Version being deployed.
            deployer: Who triggered the deployment.
            status: Deployment status.
            commit_sha: Optional commit SHA.
            commit_message: Optional commit message.
            deployment_url: Optional URL to deployment.

        Returns:
            Dict with send result.
        """
        widgets: list[dict[str, Any]] = [
            self._create_decorated_text(
                text=environment,
                top_label="Environment",
                icon_name="HOTEL_ROOM_TYPE",
            ),
            self._create_decorated_text(
                text=version,
                top_label="Version",
                icon_name="BOOKMARK",
            ),
            self._create_decorated_text(
                text=deployer,
                top_label="Deployed by",
                icon_name="PERSON",
            ),
        ]

        if commit_sha:
            text = commit_sha[:8]
            if commit_message:
                text = f"{text}: {commit_message[:50]}"
            widgets.append(
                self._create_decorated_text(
                    text=text,
                    top_label="Commit",
                    icon_name="DESCRIPTION",
                )
            )

        sections: list[dict[str, Any]] = [
            {
                "header": "Deployment Details",
                "widgets": widgets,
            },
        ]

        if deployment_url:
            sections.append({
                "widgets": [
                    self._create_button_list([
                        {"text": "View Deployment", "url": deployment_url},
                    ]),
                ],
            })

        return await self.send_card(
            credentials=credentials,
            card_id=f"deployment-{int(time.time())}",
            header_title=f"Deployment {status.title()}",
            header_subtitle=f"{version} to {environment}",
            sections=sections,
        )
