"""Discord integration plugin.

Provides integration with Discord for sending notifications via webhooks.
Supports rich embeds, mentions, and message formatting.
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


class DiscordPlugin(IntegrationPlugin):
    """Discord integration plugin.

    Supports:
    - Webhook URL based authentication
    - Rich embed messages
    - Mentions (users, roles, everyone)
    - File attachments via URLs
    - Custom username and avatar per message
    """

    DISCORD_WEBHOOK_PATTERN = re.compile(
        r"^https://discord(?:app)?\.com/api/webhooks/(\d+)/([a-zA-Z0-9_-]+)$"
    )

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Discord integration metadata."""
        return IntegrationMetadata(
            slug="discord",
            name="Discord",
            description="Send notifications to Discord channels via webhooks",
            category=IntegrationCategory.CHAT,
            auth_type=AuthType.WEBHOOK,
            icon_url="https://assets-global.website-files.com/6257adef93867e50d84d30e2/636e0a6a49cf127bf92de1e2_icon_clyde_blurple_RGB.png",
            docs_url="https://discord.com/developers/docs/resources/webhook",
            website_url="https://discord.com/",
            features=[
                "webhooks",
                "rich_embeds",
                "mentions",
                "custom_username",
                "custom_avatar",
                "thread_support",
            ],
            required_fields=[
                FieldDefinition(
                    name="webhook_url",
                    label="Webhook URL",
                    type="url",
                    required=True,
                    placeholder="https://discord.com/api/webhooks/...",
                    help_text="Discord webhook URL from channel settings",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra",
                    label="Default Username",
                    type="text",
                    required=False,
                    placeholder="Argus Bot",
                    help_text="Default username for webhook messages",
                ),
            ],
            supports_webhooks=True,
            supports_sync=False,  # Discord webhooks don't support data sync
            sync_resources=[],
        )

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Discord webhook connection.

        Args:
            credentials: Discord credentials with webhook_url.

        Returns:
            ConnectionTestResult with webhook info.
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
                message="Invalid Discord webhook URL format",
                error_code="invalid_webhook_url",
                error_details={
                    "expected_format": "https://discord.com/api/webhooks/{id}/{token}"
                },
            )

        try:
            # GET request to webhook URL returns webhook info
            response = await self.http_client.get(credentials.webhook_url)
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid webhook token",
                    latency_ms=latency_ms,
                    error_code="invalid_token",
                )

            if response.status_code == 404:
                return ConnectionTestResult(
                    success=False,
                    message="Webhook not found - it may have been deleted",
                    latency_ms=latency_ms,
                    error_code="webhook_not_found",
                )

            if response.status_code != 200:
                return ConnectionTestResult(
                    success=False,
                    message=f"Discord API error: HTTP {response.status_code}",
                    latency_ms=latency_ms,
                    error_code=f"http_{response.status_code}",
                )

            data = response.json()

            return ConnectionTestResult(
                success=True,
                message=f"Connected to webhook: {data.get('name', 'Unknown')}",
                latency_ms=latency_ms,
                api_version="10",  # Discord API v10
                account_info={
                    "webhook_id": data.get("id"),
                    "name": data.get("name"),
                    "channel_id": data.get("channel_id"),
                    "guild_id": data.get("guild_id"),
                    "type": data.get("type"),
                    "avatar": data.get("avatar"),
                },
            )

        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="connection_error",
                error_details={"exception": str(e)},
            )

    def _validate_webhook_url(self, url: str) -> bool:
        """Validate Discord webhook URL format.

        Args:
            url: URL to validate.

        Returns:
            True if URL matches Discord webhook pattern.
        """
        return bool(self.DISCORD_WEBHOOK_PATTERN.match(url))

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Discord (not supported for webhooks).

        Discord webhooks are outbound-only and don't support data sync.

        Args:
            credentials: Discord credentials.
            resources: Not used.
            since: Not used.
            limit: Not used.

        Returns:
            SyncResult indicating sync is not supported.
        """
        return SyncResult(
            success=False,
            message="Discord webhook integration does not support data sync",
            warnings=["Sync is not available for Discord webhook integrations"],
        )

    async def send_webhook(
        self,
        credentials: IntegrationCredentials,
        content: Optional[str] = None,
        embeds: Optional[list[dict[str, Any]]] = None,
        username: Optional[str] = None,
        avatar_url: Optional[str] = None,
        tts: bool = False,
        allowed_mentions: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a message via Discord webhook.

        Args:
            credentials: Discord credentials with webhook_url.
            content: Message text content (max 2000 characters).
            embeds: List of embed objects (max 10).
            username: Override webhook's default username.
            avatar_url: Override webhook's default avatar.
            tts: Enable text-to-speech.
            allowed_mentions: Control which mentions ping.
            thread_id: Post to a specific thread.

        Returns:
            Dict with success status and message details.
        """
        if not credentials.webhook_url:
            return {
                "success": False,
                "error": "No webhook URL provided",
            }

        if not content and not embeds:
            return {
                "success": False,
                "error": "Either content or embeds must be provided",
            }

        try:
            payload: dict[str, Any] = {}

            if content:
                payload["content"] = content[:2000]  # Discord limit

            if embeds:
                payload["embeds"] = embeds[:10]  # Discord limit

            if username:
                payload["username"] = username

            if avatar_url:
                payload["avatar_url"] = avatar_url

            if tts:
                payload["tts"] = True

            if allowed_mentions:
                payload["allowed_mentions"] = allowed_mentions

            # Build URL with thread_id if specified
            url = credentials.webhook_url
            if thread_id:
                url = f"{url}?thread_id={thread_id}"

            response = await self.http_client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            # 204 No Content is success for webhooks without wait=true
            if response.status_code in (200, 204):
                result: dict[str, Any] = {"success": True}
                if response.status_code == 200:
                    result["message"] = response.json()
                return result

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.json().get("retry_after", 1)
                return {
                    "success": False,
                    "error": "Rate limited",
                    "retry_after": retry_after,
                }

            return {
                "success": False,
                "error": f"Discord API error: HTTP {response.status_code}",
                "details": response.json() if response.content else None,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def send_embed(
        self,
        credentials: IntegrationCredentials,
        title: Optional[str] = None,
        description: Optional[str] = None,
        color: Optional[int] = None,
        url: Optional[str] = None,
        fields: Optional[list[dict[str, Any]]] = None,
        thumbnail_url: Optional[str] = None,
        image_url: Optional[str] = None,
        footer_text: Optional[str] = None,
        footer_icon_url: Optional[str] = None,
        author_name: Optional[str] = None,
        author_url: Optional[str] = None,
        author_icon_url: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a rich embed message to Discord.

        Args:
            credentials: Discord credentials.
            title: Embed title (max 256 characters).
            description: Embed description (max 4096 characters).
            color: Embed color as integer (e.g., 0x00ff00 for green).
            url: URL for the title hyperlink.
            fields: List of field dicts with name, value, inline.
            thumbnail_url: Small image in top right.
            image_url: Large image at bottom.
            footer_text: Footer text.
            footer_icon_url: Footer icon URL.
            author_name: Author name.
            author_url: Author URL.
            author_icon_url: Author icon URL.
            timestamp: ISO8601 timestamp string.

        Returns:
            Dict with send result.
        """
        embed: dict[str, Any] = {}

        if title:
            embed["title"] = title[:256]
        if description:
            embed["description"] = description[:4096]
        if color is not None:
            embed["color"] = color
        if url:
            embed["url"] = url
        if timestamp:
            embed["timestamp"] = timestamp

        if fields:
            embed["fields"] = [
                {
                    "name": f.get("name", "")[:256],
                    "value": f.get("value", "")[:1024],
                    "inline": f.get("inline", False),
                }
                for f in fields[:25]  # Max 25 fields
            ]

        if thumbnail_url:
            embed["thumbnail"] = {"url": thumbnail_url}

        if image_url:
            embed["image"] = {"url": image_url}

        if footer_text or footer_icon_url:
            embed["footer"] = {}
            if footer_text:
                embed["footer"]["text"] = footer_text[:2048]
            if footer_icon_url:
                embed["footer"]["icon_url"] = footer_icon_url

        if author_name:
            embed["author"] = {"name": author_name[:256]}
            if author_url:
                embed["author"]["url"] = author_url
            if author_icon_url:
                embed["author"]["icon_url"] = author_icon_url

        return await self.send_webhook(credentials, embeds=[embed])

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
            credentials: Discord credentials.
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
        # Determine color based on status
        if status.lower() == "passed":
            color = 0x36A64F  # Green
            emoji = "white_check_mark"
        elif status.lower() == "failed":
            color = 0xDC3545  # Red
            emoji = "x"
        else:
            color = 0xFFC107  # Yellow
            emoji = "warning"

        total = passed + failed + skipped
        duration_str = f"{duration_seconds:.1f}s" if duration_seconds < 60 else f"{duration_seconds / 60:.1f}m"

        fields = [
            {"name": "Project", "value": project_name, "inline": True},
            {"name": "Run ID", "value": test_run_id, "inline": True},
            {"name": "Duration", "value": duration_str, "inline": True},
            {"name": "Passed", "value": str(passed), "inline": True},
            {"name": "Failed", "value": str(failed), "inline": True},
            {"name": "Skipped", "value": str(skipped), "inline": True},
        ]

        description = f"**{passed}/{total}** tests passed"
        if dashboard_url:
            description += f"\n\n[View in Dashboard]({dashboard_url})"

        return await self.send_embed(
            credentials=credentials,
            title=f":{emoji}: Test Run {status.title()}",
            description=description,
            color=color,
            fields=fields,
            footer_text="Argus E2E Testing",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    async def send_mention(
        self,
        credentials: IntegrationCredentials,
        content: str,
        user_ids: Optional[list[str]] = None,
        role_ids: Optional[list[str]] = None,
        mention_everyone: bool = False,
    ) -> dict[str, Any]:
        """Send a message with mentions.

        Args:
            credentials: Discord credentials.
            content: Message content.
            user_ids: List of user IDs to mention.
            role_ids: List of role IDs to mention.
            mention_everyone: Whether to mention @everyone.

        Returns:
            Dict with send result.
        """
        mentions = []

        if mention_everyone:
            mentions.append("@everyone")

        if user_ids:
            mentions.extend([f"<@{uid}>" for uid in user_ids])

        if role_ids:
            mentions.extend([f"<@&{rid}>" for rid in role_ids])

        # Prepend mentions to content
        full_content = " ".join(mentions) + " " + content if mentions else content

        # Set allowed_mentions to control who gets pinged
        allowed_mentions: dict[str, Any] = {}
        if user_ids:
            allowed_mentions["users"] = user_ids
        if role_ids:
            allowed_mentions["roles"] = role_ids
        if mention_everyone:
            allowed_mentions["parse"] = ["everyone"]

        return await self.send_webhook(
            credentials=credentials,
            content=full_content,
            allowed_mentions=allowed_mentions if allowed_mentions else None,
        )
