"""Plugin registry for integration discovery and management.

This module provides a centralized registry for all integration plugins,
supporting auto-discovery, lazy loading, and plugin lifecycle management.
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Optional, Type

from .base import (
    AuthType,
    IntegrationCategory,
    IntegrationCredentials,
    IntegrationMetadata,
    IntegrationPlugin,
)

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for discovering and managing integration plugins.

    The registry supports:
    - Auto-discovery of plugins in the plugins/ directory
    - Manual plugin registration
    - Lazy loading of plugin classes
    - Category and auth type filtering

    Example:
        # Get the global registry instance
        registry = PluginRegistry.get_instance()

        # Get a specific plugin
        slack_plugin = registry.get_plugin("slack")
        if slack_plugin:
            result = await slack_plugin.test_connection(credentials)

        # List all plugins in a category
        chat_plugins = registry.get_plugins_by_category(IntegrationCategory.CHAT)
    """

    _instance: Optional["PluginRegistry"] = None
    _initialized: bool = False

    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: dict[str, Type[IntegrationPlugin]] = {}
        self._metadata_cache: dict[str, IntegrationMetadata] = {}
        self._plugin_instances: dict[str, IntegrationPlugin] = {}

    @classmethod
    def get_instance(cls) -> "PluginRegistry":
        """Get the singleton registry instance.

        Returns:
            The global PluginRegistry instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        if not cls._initialized:
            cls._instance.discover_plugins()
            cls._initialized = True
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (mainly for testing)."""
        cls._instance = None
        cls._initialized = False

    def register(
        self,
        plugin_class: Type[IntegrationPlugin],
        slug: Optional[str] = None,
    ) -> None:
        """Register a plugin class.

        Args:
            plugin_class: The plugin class to register.
            slug: Optional slug override. If not provided, uses metadata.slug.
        """
        # Instantiate temporarily to get metadata
        try:
            temp_instance = plugin_class()
            metadata = temp_instance.metadata
            plugin_slug = slug or metadata.slug

            self._plugins[plugin_slug] = plugin_class
            self._metadata_cache[plugin_slug] = metadata

            logger.debug(f"Registered plugin: {plugin_slug} ({metadata.name})")
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_class}: {e}")

    def unregister(self, slug: str) -> bool:
        """Unregister a plugin.

        Args:
            slug: The plugin slug to unregister.

        Returns:
            True if the plugin was unregistered, False if it wasn't registered.
        """
        if slug in self._plugins:
            del self._plugins[slug]
            self._metadata_cache.pop(slug, None)
            self._plugin_instances.pop(slug, None)
            return True
        return False

    def discover_plugins(self) -> int:
        """Discover and register all plugins in the plugins/ directory.

        Returns:
            Number of plugins discovered.
        """
        plugins_dir = Path(__file__).parent / "plugins"
        if not plugins_dir.exists():
            logger.warning(f"Plugins directory not found: {plugins_dir}")
            return 0

        discovered = 0

        # Walk through all subdirectories (by category)
        for category_dir in plugins_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith("_"):
                continue

            # Import all Python modules in the category
            for module_info in pkgutil.iter_modules([str(category_dir)]):
                if module_info.name.startswith("_"):
                    continue

                module_name = f"src.integrations.plugins.{category_dir.name}.{module_info.name}"
                try:
                    module = importlib.import_module(module_name)

                    # Find all IntegrationPlugin subclasses in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, IntegrationPlugin)
                            and attr is not IntegrationPlugin
                        ):
                            self.register(attr)
                            discovered += 1

                except Exception as e:
                    logger.error(f"Failed to load plugin module {module_name}: {e}")

        logger.info(f"Discovered {discovered} integration plugins")
        return discovered

    def get_plugin(self, slug: str) -> Optional[IntegrationPlugin]:
        """Get a plugin instance by slug.

        Args:
            slug: The plugin slug (e.g., "slack", "github").

        Returns:
            The plugin instance, or None if not found.
        """
        if slug not in self._plugins:
            return None

        # Return cached instance or create new one
        if slug not in self._plugin_instances:
            self._plugin_instances[slug] = self._plugins[slug]()

        return self._plugin_instances[slug]

    def get_plugin_class(self, slug: str) -> Optional[Type[IntegrationPlugin]]:
        """Get a plugin class by slug.

        Args:
            slug: The plugin slug.

        Returns:
            The plugin class, or None if not found.
        """
        return self._plugins.get(slug)

    def get_metadata(self, slug: str) -> Optional[IntegrationMetadata]:
        """Get plugin metadata by slug.

        Args:
            slug: The plugin slug.

        Returns:
            The plugin metadata, or None if not found.
        """
        return self._metadata_cache.get(slug)

    def list_plugins(self) -> list[IntegrationMetadata]:
        """List all registered plugins.

        Returns:
            List of IntegrationMetadata for all plugins.
        """
        return list(self._metadata_cache.values())

    def list_slugs(self) -> list[str]:
        """List all registered plugin slugs.

        Returns:
            List of plugin slugs.
        """
        return list(self._plugins.keys())

    def get_plugins_by_category(
        self,
        category: IntegrationCategory,
    ) -> list[IntegrationMetadata]:
        """Get all plugins in a specific category.

        Args:
            category: The category to filter by.

        Returns:
            List of IntegrationMetadata for plugins in the category.
        """
        return [
            metadata
            for metadata in self._metadata_cache.values()
            if metadata.category == category
        ]

    def get_plugins_by_auth_type(
        self,
        auth_type: AuthType,
    ) -> list[IntegrationMetadata]:
        """Get all plugins with a specific auth type.

        Args:
            auth_type: The auth type to filter by.

        Returns:
            List of IntegrationMetadata for plugins with the auth type.
        """
        return [
            metadata
            for metadata in self._metadata_cache.values()
            if metadata.auth_type == auth_type
        ]

    def has_plugin(self, slug: str) -> bool:
        """Check if a plugin is registered.

        Args:
            slug: The plugin slug.

        Returns:
            True if the plugin is registered.
        """
        return slug in self._plugins

    async def test_connection(
        self,
        slug: str,
        credentials: IntegrationCredentials,
    ):
        """Test a connection using a plugin.

        Args:
            slug: The plugin slug.
            credentials: The credentials to test.

        Returns:
            ConnectionTestResult from the plugin.

        Raises:
            ValueError: If the plugin is not found.
        """
        from .base import ConnectionTestResult

        plugin = self.get_plugin(slug)
        if not plugin:
            raise ValueError(f"Plugin not found: {slug}")

        # Validate credentials first
        is_valid, missing = plugin.validate_credentials(credentials)
        if not is_valid:
            return ConnectionTestResult(
                success=False,
                message=f"Missing required credentials: {', '.join(missing)}",
                error_code="MISSING_CREDENTIALS",
            )

        return await plugin.test_connection(credentials)

    async def sync_data(
        self,
        slug: str,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since=None,
        limit: int = 100,
    ):
        """Sync data using a plugin.

        Args:
            slug: The plugin slug.
            credentials: The credentials to use.
            resources: Optional list of resources to sync.
            since: Optional datetime to sync since.
            limit: Maximum items per resource.

        Returns:
            SyncResult from the plugin.

        Raises:
            ValueError: If the plugin is not found.
        """
        from .base import SyncResult

        plugin = self.get_plugin(slug)
        if not plugin:
            raise ValueError(f"Plugin not found: {slug}")

        # Validate credentials first
        is_valid, missing = plugin.validate_credentials(credentials)
        if not is_valid:
            return SyncResult(
                success=False,
                message=f"Missing required credentials: {', '.join(missing)}",
            )

        return await plugin.sync_data(
            credentials,
            resources=resources,
            since=since,
            limit=limit,
        )


# Convenience function to get the global registry
def get_registry() -> PluginRegistry:
    """Get the global plugin registry.

    Returns:
        The global PluginRegistry instance.
    """
    return PluginRegistry.get_instance()
