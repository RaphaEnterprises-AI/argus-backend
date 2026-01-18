"""
Browser automation module - Unified Hetzner Browser Pool.

This module provides scalable browser automation via a Kubernetes-based
browser pool on Hetzner Cloud, with automatic fallback to Cloudflare.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                     BrowserPoolClient (Primary)                      │
│           Hetzner K8s cluster with HPA (5-500 pods)                 │
└─────────────────────────────────────────────────────────────────────┘
                              │ fallback
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                E2EBrowserClient (Legacy/Fallback)                    │
│         Cloudflare Browser Rendering + TestingBot                   │
└─────────────────────────────────────────────────────────────────────┘

Key Features:
- Scalable: 5-500 browser pods via Kubernetes HPA
- Self-healing: Vision fallback via Claude Computer Use
- MCP-compatible: /observe, /act, /test endpoints
- Cost-effective: ~70% cheaper than cloud providers
- Natural language: client.act("Click the login button")

Usage (New - Recommended):
    from src.browser import BrowserPoolClient

    async with BrowserPoolClient() as client:
        # Discover elements
        result = await client.observe("https://example.com")

        # Execute action with self-healing
        result = await client.act("https://example.com", "Click Sign In")

        # Run multi-step test
        result = await client.test(
            url="https://example.com",
            steps=["Click Login", "Type email", "Submit"],
        )

Usage (Legacy - Still Supported):
    from src.browser import E2EBrowserClient, run_test_with_e2e_client

    async with E2EBrowserClient() as client:
        page = await client.new_page("https://example.com")
        await page.act("Click Sign In")
"""

# New unified browser pool client (Primary)
# Legacy client (Fallback)
from .e2e_client import (
    ActionResult,
    BrowserAction,
    BrowserPage,
    E2EBrowserClient,
    ExtractionSchema,
    PageState,
    run_test_with_e2e_client,
)
from .pool_client import BrowserPoolClient
from .pool_models import (
    ActionResult as PoolActionResult,
)
from .pool_models import (
    # Enums
    ActionType,
    ActResult,
    BrowserPoolConfig,
    BrowserType,
    # Data classes
    ElementInfo,
    ExecutionMode,
    ExtractResult,
    ObserveResult,
    PoolHealth,
    SessionInfo,
    TestResult,
)
from .pool_models import (
    StepResult as PoolStepResult,
)

__all__ = [
    # Primary (New)
    "BrowserPoolClient",
    "ActionType",
    "BrowserType",
    "ExecutionMode",
    "ElementInfo",
    "PoolActionResult",
    "ObserveResult",
    "ActResult",
    "PoolStepResult",
    "TestResult",
    "ExtractResult",
    "SessionInfo",
    "PoolHealth",
    "BrowserPoolConfig",
    # Legacy (Fallback)
    "E2EBrowserClient",
    "BrowserPage",
    "BrowserAction",
    "ActionResult",
    "ExtractionSchema",
    "PageState",
    "run_test_with_e2e_client",
]
