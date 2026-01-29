"""Chat integration plugins.

Provides integrations for team communication and notification platforms:
- Slack: Full API + OAuth2 + Webhooks
- Discord: Webhook-based with rich embeds
- Microsoft Teams: Webhook-based with Adaptive Cards
- Google Chat: Webhook-based with Cards v2
"""

from src.integrations.plugins.chat.discord_plugin import DiscordPlugin
from src.integrations.plugins.chat.google_chat_plugin import GoogleChatPlugin
from src.integrations.plugins.chat.slack_plugin import SlackPlugin
from src.integrations.plugins.chat.teams_plugin import TeamsPlugin

__all__ = [
    "SlackPlugin",
    "DiscordPlugin",
    "TeamsPlugin",
    "GoogleChatPlugin",
]
