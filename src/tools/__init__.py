"""Tools for E2E testing - Browser automation, API, Database.

This module provides a multi-framework browser automation layer:
- PlaywrightAutomation: Fast, reliable programmatic automation
- SeleniumAutomation: WebDriver-based automation for legacy support
- ComputerUseAutomation: Visual/pixel-based automation via Claude Computer Use
- HybridAutomation: Programmatic + Computer Use fallback
- ExtensionBridge: Chrome extension for real browser automation (like Claude in Chrome)

Usage:
    from src.tools import create_browser, AutomationFramework

    # Use Playwright (default)
    browser = await create_browser(framework=AutomationFramework.PLAYWRIGHT)

    # Use Chrome Extension (for real browser session)
    browser = await create_browser(framework=AutomationFramework.EXTENSION)

    # Use hybrid mode (programmatic + Computer Use fallback)
    browser = await create_browser(framework=AutomationFramework.HYBRID)
"""

from .playwright_tools import (
    BrowserManager,
    PlaywrightTools,
    create_browser_context,
)

from .browser_abstraction import (
    BrowserAutomation,
    BrowserConfig,
    ActionResult,
    AutomationFramework,
    PlaywrightAutomation,
    SeleniumAutomation,
    ComputerUseAutomation,
    HybridAutomation,
    create_browser,
)

from .extension_bridge import (
    ExtensionBridge,
    ExtensionMessage,
    ExtensionResponse,
    create_extension_bridge,
)

from .browser_worker_client import (
    BrowserWorkerClient,
    ActionResult as WorkerActionResult,
    TestResult as WorkerTestResult,
    DiscoveryResult,
    ExtractionResult,
    AgentResult,
    get_browser_client,
    cleanup_browser_client,
)

__all__ = [
    # Legacy exports
    "BrowserManager",
    "PlaywrightTools",
    "create_browser_context",
    # Browser abstraction layer
    "BrowserAutomation",
    "BrowserConfig",
    "ActionResult",
    "AutomationFramework",
    "PlaywrightAutomation",
    "SeleniumAutomation",
    "ComputerUseAutomation",
    "HybridAutomation",
    "create_browser",
    # Chrome Extension bridge
    "ExtensionBridge",
    "ExtensionMessage",
    "ExtensionResponse",
    "create_extension_bridge",
    # Browser Worker client (Cloudflare)
    "BrowserWorkerClient",
    "WorkerActionResult",
    "WorkerTestResult",
    "DiscoveryResult",
    "ExtractionResult",
    "AgentResult",
    "get_browser_client",
    "cleanup_browser_client",
]
