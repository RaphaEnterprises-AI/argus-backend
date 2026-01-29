"""Feature flags integration plugins.

This module provides integrations with popular feature flag management platforms:

- **LaunchDarkly**: Enterprise feature management platform with advanced targeting
- **Split**: Feature delivery platform for controlled rollouts and experimentation
- **Flagsmith**: Open-source feature flag and remote config service (cloud or self-hosted)
- **Unleash**: Open-source feature management solution (self-hosted)

All plugins implement the IntegrationPlugin interface and support:
- Connection testing
- Data synchronization (flags, environments, segments, etc.)
- Targeting rules and variations
- Environment-specific configuration

Example:
    from src.integrations.plugins.feature_flags import LaunchDarklyPlugin

    async with LaunchDarklyPlugin() as plugin:
        credentials = IntegrationCredentials(
            access_token="api-xxx",
            extra={"project_key": "my-project"}
        )
        result = await plugin.test_connection(credentials)
        if result.success:
            sync_result = await plugin.sync_data(credentials)
"""

from src.integrations.plugins.feature_flags.flagsmith_plugin import FlagsmithPlugin
from src.integrations.plugins.feature_flags.launchdarkly_plugin import LaunchDarklyPlugin
from src.integrations.plugins.feature_flags.split_plugin import SplitPlugin
from src.integrations.plugins.feature_flags.unleash_plugin import UnleashPlugin

__all__ = [
    "LaunchDarklyPlugin",
    "SplitPlugin",
    "FlagsmithPlugin",
    "UnleashPlugin",
]
