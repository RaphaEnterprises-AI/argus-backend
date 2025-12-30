"""
Browser automation module using Cloudflare Worker + TestingBot.

This module provides AI-powered browser automation via our custom
Cloudflare Worker that supports multiple browser backends.

Key Features:
- Natural language actions: page.act("Click the login button")
- Built-in self-healing: Auto-fixes broken selectors
- Multi-backend: Cloudflare Browser (free) + TestingBot (cross-browser)
- Cross-browser: Chrome, Firefox, Safari, Edge
- Real devices: iOS and Android testing
- AI-powered: Workers AI for intelligent automation

Usage:
    from src.browser import E2EBrowserClient, run_test_with_e2e_client

    # Full control
    async with E2EBrowserClient() as client:
        page = await client.new_page("https://example.com")
        await page.act("Click Sign In")
        data = await page.extract({"username": "string"})

    # Quick test
    result = await run_test_with_e2e_client(
        url="https://example.com",
        steps=["Click Login", "Type email", "Submit"],
    )
"""

from .e2e_client import (
    E2EBrowserClient,
    BrowserPage,
    BrowserAction,
    ActionResult,
    ExtractionSchema,
    PageState,
    run_test_with_e2e_client,
)

__all__ = [
    "E2EBrowserClient",
    "BrowserPage",
    "BrowserAction",
    "ActionResult",
    "ExtractionSchema",
    "PageState",
    "run_test_with_e2e_client",
]
