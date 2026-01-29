"""Slack integration plugin.

Provides integration with Slack for team notifications, channel management,
and message posting. Supports both OAuth2 Bot Token and Incoming Webhooks.
"""

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
    OAuthConfig,
    SyncResult,
)


class SlackPlugin(IntegrationPlugin):
    """Slack integration plugin.

    Supports:
    - OAuth2 Bot Token authentication
    - Incoming Webhooks for simple notifications
    - Web API for full channel/user management
    - Rich message formatting with Blocks
    """

    SLACK_API_BASE = "https://slack.com/api"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Slack integration metadata."""
        return IntegrationMetadata(
            slug="slack",
            name="Slack",
            description="Team communication and notifications via Slack",
            category=IntegrationCategory.CHAT,
            auth_type=AuthType.OAUTH2,
            icon_url="https://a.slack-edge.com/80588/marketing/img/icons/icon_slack_hash_colored.png",
            docs_url="https://api.slack.com/",
            website_url="https://slack.com/",
            features=[
                "notifications",
                "channels",
                "direct_messages",
                "webhooks",
                "rich_formatting",
                "file_uploads",
                "reactions",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Bot Token",
                    type="password",
                    required=True,
                    placeholder="xoxb-...",
                    help_text="Slack Bot User OAuth Token (starts with xoxb-)",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="webhook_url",
                    label="Incoming Webhook URL",
                    type="url",
                    required=False,
                    placeholder="https://hooks.slack.com/services/...",
                    help_text="Incoming Webhook URL for simple notifications",
                ),
                FieldDefinition(
                    name="workspace_id",
                    label="Workspace ID",
                    type="text",
                    required=False,
                    placeholder="T0123456789",
                    help_text="Slack Workspace ID (Team ID)",
                ),
            ],
            oauth_config=OAuthConfig(
                authorize_url="https://slack.com/oauth/v2/authorize",
                token_url="https://slack.com/api/oauth.v2.access",
                scopes=[
                    "channels:read",
                    "channels:join",
                    "chat:write",
                    "chat:write.public",
                    "users:read",
                    "team:read",
                    "incoming-webhook",
                ],
                client_id_env_var="SLACK_CLIENT_ID",
                client_secret_env_var="SLACK_CLIENT_SECRET",
            ),
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["channels", "users"],
        )

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Slack connection using auth.test API.

        Args:
            credentials: Slack credentials with access_token.

        Returns:
            ConnectionTestResult with workspace info and permissions.
        """
        start_time = time.time()

        # If only webhook URL is provided, validate the webhook
        if not credentials.access_token and credentials.webhook_url:
            return await self._test_webhook_connection(credentials)

        if not credentials.access_token:
            return ConnectionTestResult(
                success=False,
                message="No access token provided",
                error_code="missing_token",
            )

        try:
            response = await self.http_client.post(
                f"{self.SLACK_API_BASE}/auth.test",
                headers={
                    "Authorization": f"Bearer {credentials.access_token}",
                    "Content-Type": "application/json",
                },
            )

            latency_ms = (time.time() - start_time) * 1000
            data = response.json()

            if not data.get("ok"):
                return ConnectionTestResult(
                    success=False,
                    message=f"Slack API error: {data.get('error', 'Unknown error')}",
                    latency_ms=latency_ms,
                    error_code=data.get("error"),
                    error_details=data,
                )

            # Get bot permissions
            permissions = await self._get_permissions(credentials.access_token)

            return ConnectionTestResult(
                success=True,
                message=f"Connected to workspace: {data.get('team', 'Unknown')}",
                latency_ms=latency_ms,
                permissions=permissions,
                account_info={
                    "team_id": data.get("team_id"),
                    "team": data.get("team"),
                    "user_id": data.get("user_id"),
                    "user": data.get("user"),
                    "bot_id": data.get("bot_id"),
                    "is_enterprise_install": data.get("is_enterprise_install", False),
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

    async def _test_webhook_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Slack webhook URL validity.

        Args:
            credentials: Credentials with webhook_url.

        Returns:
            ConnectionTestResult for webhook validation.
        """
        if not credentials.webhook_url:
            return ConnectionTestResult(
                success=False,
                message="No webhook URL provided",
                error_code="missing_webhook_url",
            )

        # Validate URL format
        if not credentials.webhook_url.startswith("https://hooks.slack.com/"):
            return ConnectionTestResult(
                success=False,
                message="Invalid Slack webhook URL format",
                error_code="invalid_webhook_url",
            )

        return ConnectionTestResult(
            success=True,
            message="Webhook URL format is valid",
            account_info={"webhook_configured": True},
        )

    async def _get_permissions(self, access_token: str) -> list[str]:
        """Get bot permissions/scopes.

        Args:
            access_token: Slack bot token.

        Returns:
            List of OAuth scopes.
        """
        try:
            response = await self.http_client.get(
                f"{self.SLACK_API_BASE}/auth.test",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            # Note: auth.test doesn't return scopes directly
            # In a real implementation, you'd track scopes from OAuth flow
            return self.metadata.oauth_config.scopes if self.metadata.oauth_config else []
        except Exception:
            return []

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync channels and users from Slack.

        Args:
            credentials: Slack credentials.
            resources: Optional list of resources to sync ("channels", "users").
            since: Not used for Slack (no incremental sync support).
            limit: Maximum items per resource.

        Returns:
            SyncResult with channels and users data.
        """
        started_at = datetime.utcnow()
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        if not credentials.access_token:
            return SyncResult(
                success=False,
                message="No access token provided",
                errors=["Missing access_token in credentials"],
            )

        resources_to_sync = resources or ["channels", "users"]

        # Sync channels
        if "channels" in resources_to_sync:
            channels_result = await self._sync_channels(credentials.access_token, limit)
            if channels_result["success"]:
                data["channels"] = channels_result["data"]
                total_synced += len(channels_result["data"])
            else:
                errors.append(f"Failed to sync channels: {channels_result['error']}")

        # Sync users
        if "users" in resources_to_sync:
            users_result = await self._sync_users(credentials.access_token, limit)
            if users_result["success"]:
                data["users"] = users_result["data"]
                total_synced += len(users_result["data"])
            else:
                errors.append(f"Failed to sync users: {users_result['error']}")

        completed_at = datetime.utcnow()

        return SyncResult(
            success=len(errors) == 0,
            message=f"Synced {total_synced} items from Slack",
            data_points_synced=total_synced,
            started_at=started_at,
            completed_at=completed_at,
            errors=errors,
            warnings=warnings,
            data=data,
        )

    async def _sync_channels(
        self,
        access_token: str,
        limit: int,
    ) -> dict[str, Any]:
        """Sync public channels from Slack.

        Args:
            access_token: Slack bot token.
            limit: Maximum channels to fetch.

        Returns:
            Dict with success status and channel data.
        """
        try:
            response = await self.http_client.get(
                f"{self.SLACK_API_BASE}/conversations.list",
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "types": "public_channel,private_channel",
                    "limit": min(limit, 1000),
                    "exclude_archived": "true",
                },
            )

            data = response.json()
            if not data.get("ok"):
                return {"success": False, "error": data.get("error", "Unknown error")}

            channels = [
                {
                    "id": ch["id"],
                    "name": ch["name"],
                    "is_private": ch.get("is_private", False),
                    "is_member": ch.get("is_member", False),
                    "num_members": ch.get("num_members", 0),
                    "topic": ch.get("topic", {}).get("value", ""),
                    "purpose": ch.get("purpose", {}).get("value", ""),
                    "created": ch.get("created"),
                }
                for ch in data.get("channels", [])
            ]

            return {"success": True, "data": channels}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _sync_users(
        self,
        access_token: str,
        limit: int,
    ) -> dict[str, Any]:
        """Sync users from Slack workspace.

        Args:
            access_token: Slack bot token.
            limit: Maximum users to fetch.

        Returns:
            Dict with success status and user data.
        """
        try:
            response = await self.http_client.get(
                f"{self.SLACK_API_BASE}/users.list",
                headers={"Authorization": f"Bearer {access_token}"},
                params={"limit": min(limit, 1000)},
            )

            data = response.json()
            if not data.get("ok"):
                return {"success": False, "error": data.get("error", "Unknown error")}

            users = [
                {
                    "id": user["id"],
                    "name": user.get("name"),
                    "real_name": user.get("real_name"),
                    "display_name": user.get("profile", {}).get("display_name"),
                    "email": user.get("profile", {}).get("email"),
                    "is_admin": user.get("is_admin", False),
                    "is_bot": user.get("is_bot", False),
                    "deleted": user.get("deleted", False),
                    "avatar": user.get("profile", {}).get("image_72"),
                    "timezone": user.get("tz"),
                }
                for user in data.get("members", [])
                if not user.get("is_bot") and not user.get("deleted")
            ]

            return {"success": True, "data": users}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def send_notification(
        self,
        credentials: IntegrationCredentials,
        channel: str,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        attachments: Optional[list[dict[str, Any]]] = None,
        thread_ts: Optional[str] = None,
        unfurl_links: bool = False,
        unfurl_media: bool = True,
    ) -> dict[str, Any]:
        """Send a message to a Slack channel.

        Args:
            credentials: Slack credentials.
            channel: Channel ID or name (e.g., "#general" or "C0123456789").
            text: Message text (used as fallback for blocks).
            blocks: Optional Block Kit blocks for rich formatting.
            attachments: Optional legacy attachments.
            thread_ts: Optional thread timestamp to reply to.
            unfurl_links: Whether to unfurl links.
            unfurl_media: Whether to unfurl media.

        Returns:
            Dict with success status and message details.
        """
        # Prefer webhook if available and no blocks/advanced features needed
        if (
            credentials.webhook_url
            and not blocks
            and not attachments
            and not thread_ts
        ):
            return await self._send_via_webhook(credentials.webhook_url, text)

        if not credentials.access_token:
            return {
                "success": False,
                "error": "No access token provided",
            }

        try:
            payload: dict[str, Any] = {
                "channel": channel,
                "text": text,
                "unfurl_links": unfurl_links,
                "unfurl_media": unfurl_media,
            }

            if blocks:
                payload["blocks"] = blocks
            if attachments:
                payload["attachments"] = attachments
            if thread_ts:
                payload["thread_ts"] = thread_ts

            response = await self.http_client.post(
                f"{self.SLACK_API_BASE}/chat.postMessage",
                headers={
                    "Authorization": f"Bearer {credentials.access_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            data = response.json()

            if not data.get("ok"):
                return {
                    "success": False,
                    "error": data.get("error", "Unknown error"),
                    "response_metadata": data.get("response_metadata"),
                }

            return {
                "success": True,
                "channel": data.get("channel"),
                "ts": data.get("ts"),
                "message": data.get("message"),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _send_via_webhook(
        self,
        webhook_url: str,
        text: str,
    ) -> dict[str, Any]:
        """Send a simple message via incoming webhook.

        Args:
            webhook_url: Slack incoming webhook URL.
            text: Message text.

        Returns:
            Dict with success status.
        """
        try:
            response = await self.http_client.post(
                webhook_url,
                json={"text": text},
            )

            # Slack webhooks return "ok" on success
            if response.status_code == 200 and response.text == "ok":
                return {"success": True}

            return {
                "success": False,
                "error": f"Webhook returned: {response.text}",
                "status_code": response.status_code,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def send_test_result_notification(
        self,
        credentials: IntegrationCredentials,
        channel: str,
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
            credentials: Slack credentials.
            channel: Target channel.
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
        # Determine emoji and color based on status
        if status.lower() == "passed":
            emoji = ":white_check_mark:"
            color = "#36a64f"
        elif status.lower() == "failed":
            emoji = ":x:"
            color = "#dc3545"
        else:
            emoji = ":warning:"
            color = "#ffc107"

        total = passed + failed + skipped
        duration_str = f"{duration_seconds:.1f}s" if duration_seconds < 60 else f"{duration_seconds / 60:.1f}m"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Test Run {status.title()}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Project:*\n{project_name}"},
                    {"type": "mrkdwn", "text": f"*Run ID:*\n{test_run_id}"},
                    {"type": "mrkdwn", "text": f"*Total Tests:*\n{total}"},
                    {"type": "mrkdwn", "text": f"*Duration:*\n{duration_str}"},
                ],
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*:white_check_mark: Passed:*\n{passed}"},
                    {"type": "mrkdwn", "text": f"*:x: Failed:*\n{failed}"},
                    {"type": "mrkdwn", "text": f"*:fast_forward: Skipped:*\n{skipped}"},
                ],
            },
        ]

        if dashboard_url:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View in Dashboard",
                            "emoji": True,
                        },
                        "url": dashboard_url,
                        "action_id": "view_dashboard",
                    }
                ],
            })

        text = f"Test Run {status.title()}: {passed}/{total} passed in {duration_str}"

        return await self.send_notification(
            credentials=credentials,
            channel=channel,
            text=text,
            blocks=blocks,
        )
