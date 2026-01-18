"""
Discovery Intelligence Platform for autonomous E2E testing.

This module provides intelligent application discovery capabilities including:
- Autonomous crawling and page exploration
- Element detection and classification
- User flow identification
- Cross-project learning
"""

from src.discovery.crawlers.crawlee_bridge import (
    CrawleeBridge,
    CrawlProgress,
    check_crawlee_available,
    discover_application,
)
from src.discovery.engine import (
    DiscoveryEngine,
    DiscoveryError,
    create_discovery_engine,
)
from src.discovery.models import (
    # Configuration
    AuthConfig,
    # Discovery Results
    CrawlError,
    CrawlResult,
    DiscoveredElement,
    DiscoveredFlow,
    DiscoveredPage,
    DiscoveryConfig,
    # Enums
    DiscoveryMode,
    DiscoverySession,
    DiscoveryStatus,
    ElementBounds,
    ElementCategory,
    ExplorationStrategy,
    FlowCategory,
    FlowStep,
    PageCategory,
    PageGraph,
    PageGraphEdge,
)

__all__ = [
    # Enums
    "DiscoveryMode",
    "DiscoveryStatus",
    "ElementCategory",
    "ExplorationStrategy",
    "FlowCategory",
    "PageCategory",
    # Configuration
    "AuthConfig",
    "DiscoveryConfig",
    # Discovery Results
    "CrawlError",
    "CrawlResult",
    "DiscoveredElement",
    "DiscoveredFlow",
    "DiscoveredPage",
    "DiscoverySession",
    "ElementBounds",
    "FlowStep",
    "PageGraph",
    "PageGraphEdge",
    # Engine
    "DiscoveryEngine",
    "DiscoveryError",
    "create_discovery_engine",
    # Crawler
    "CrawleeBridge",
    "CrawlProgress",
    "check_crawlee_available",
    "discover_application",
]
