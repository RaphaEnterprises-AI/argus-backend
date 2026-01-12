"""Crawler implementations for the Discovery Intelligence Platform.

This module provides crawling capabilities for autonomous web application discovery:
- CrawleeBridge: Bridge to Crawlee (Node.js) or Playwright-based crawler
- discover_application: Convenience function for quick discovery
- check_crawlee_available: Check if Crawlee is available

Example:
    from src.discovery.crawlers import discover_application, CrawleeBridge

    # Quick discovery
    result = await discover_application("https://example.com", max_pages=50)
    print(f"Found {result.total_pages} pages")

    # Full control
    bridge = CrawleeBridge(use_crawlee=False)
    result = await bridge.run_crawl(start_url, config)
"""

from src.discovery.crawlers.crawlee_bridge import (
    CrawleeBridge,
    CrawlProgress,
    check_crawlee_available,
    discover_application,
)

__all__ = [
    "CrawleeBridge",
    "CrawlProgress",
    "check_crawlee_available",
    "discover_application",
]
