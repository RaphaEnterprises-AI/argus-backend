"""Source code integration plugins.

This module provides plugins for source code management platforms:
- GitHub: OAuth2 + REST API v3
- GitLab: OAuth2/PAT + REST API v4 (cloud and self-hosted)
- Bitbucket: App Password + REST API 2.0
- Azure DevOps: Personal Access Token + REST API
"""

from src.integrations.plugins.source_code.azure_devops_plugin import AzureDevOpsPlugin
from src.integrations.plugins.source_code.bitbucket_plugin import BitbucketPlugin
from src.integrations.plugins.source_code.github_plugin import GitHubPlugin
from src.integrations.plugins.source_code.gitlab_plugin import GitLabPlugin

__all__ = [
    "GitHubPlugin",
    "GitLabPlugin",
    "BitbucketPlugin",
    "AzureDevOpsPlugin",
]

# Plugin registry for dynamic loading
PLUGINS = {
    "github": GitHubPlugin,
    "gitlab": GitLabPlugin,
    "bitbucket": BitbucketPlugin,
    "azure_devops": AzureDevOpsPlugin,
}


def get_plugin(slug: str):
    """Get a plugin class by its slug.

    Args:
        slug: The plugin identifier (e.g., "github", "gitlab")

    Returns:
        The plugin class, or None if not found
    """
    return PLUGINS.get(slug)


def list_plugins() -> list[str]:
    """List all available source code plugin slugs."""
    return list(PLUGINS.keys())
