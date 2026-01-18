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

from .browser_abstraction import (
    ActionResult,
    AutomationFramework,
    BrowserAutomation,
    BrowserConfig,
    ComputerUseAutomation,
    HybridAutomation,
    PlaywrightAutomation,
    SeleniumAutomation,
    create_browser,
)
from .browser_worker_client import (
    ActionResult as WorkerActionResult,
)
from .browser_worker_client import (
    AgentResult,
    BrowserWorkerClient,
    DiscoveryResult,
    ExtractionResult,
    cleanup_browser_client,
    get_browser_client,
)
from .browser_worker_client import (
    TestResult as WorkerTestResult,
)
from .extension_bridge import (
    ExtensionBridge,
    ExtensionMessage,
    ExtensionResponse,
    create_extension_bridge,
)
from .playwright_tools import (
    BrowserManager,
    PlaywrightTools,
    create_browser_context,
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
