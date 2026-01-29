"""Issue tracking integration plugins.

This module provides plugins for integrating with issue tracking
and project management platforms.

Available plugins:
- JiraPlugin: Jira Cloud and Server/Data Center
- LinearPlugin: Linear modern issue tracking
- AsanaPlugin: Asana task management
- TrelloPlugin: Trello visual project management
- ClickUpPlugin: ClickUp comprehensive project management
- HeightPlugin: Height.app issue tracking
- MondayPlugin: Monday.com work OS
- NotionPlugin: Notion databases and pages
- ShortcutPlugin: Shortcut (formerly Clubhouse) project management
"""

from src.integrations.plugins.issue_tracking.asana_plugin import AsanaPlugin
from src.integrations.plugins.issue_tracking.clickup_plugin import ClickUpPlugin
from src.integrations.plugins.issue_tracking.height_plugin import HeightPlugin
from src.integrations.plugins.issue_tracking.jira_plugin import JiraPlugin
from src.integrations.plugins.issue_tracking.linear_plugin import LinearPlugin
from src.integrations.plugins.issue_tracking.monday_plugin import MondayPlugin
from src.integrations.plugins.issue_tracking.notion_plugin import NotionPlugin
from src.integrations.plugins.issue_tracking.shortcut_plugin import ShortcutPlugin
from src.integrations.plugins.issue_tracking.trello_plugin import TrelloPlugin

__all__ = [
    "AsanaPlugin",
    "ClickUpPlugin",
    "HeightPlugin",
    "JiraPlugin",
    "LinearPlugin",
    "MondayPlugin",
    "NotionPlugin",
    "ShortcutPlugin",
    "TrelloPlugin",
]
