"""Client for interacting with the Browser Automation Cloudflare Worker.

This module provides a Python interface to the Argus Browser Worker,
which handles all browser automation via Cloudflare's @cloudflare/playwright.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Any
import httpx
import structlog

from src.config import get_settings

logger = structlog.get_logger()


@dataclass
class ActionResult:
    """Result of a browser action."""
    success: bool
    message: Optional[str] = None
    actions: Optional[list[dict]] = None
    screenshot: Optional[str] = None  # Base64 encoded
    healed: bool = False
    healing_method: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TestResult:
    """Result of a multi-step test."""
    success: bool
    steps: Optional[list[dict]] = None
    screenshots: Optional[list[str]] = None
    final_screenshot: Optional[str] = None
    healing_report: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class DiscoveryResult:
    """Result of element discovery."""
    success: bool
    actions: Optional[list[dict]] = None
    error: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result of data extraction."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class AgentResult:
    """Result of autonomous agent task."""
    success: bool
    completed: bool = False
    message: Optional[str] = None
    actions: Optional[list[dict]] = None
    screenshots: Optional[list[str]] = None
    usage: Optional[dict] = None
    error: Optional[str] = None


class BrowserWorkerClient:
    """Client for the Argus Browser Automation Worker (Cloudflare)."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0):
        """Initialize the browser worker client.

        Args:
            base_url: URL of the browser worker. Defaults to config value.
            timeout: Default timeout for requests in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.browser_worker_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if the browser worker is healthy."""
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning("Browser worker health check failed", error=str(e))
            return False

    async def execute_action(
        self,
        url: str,
        instruction: str,
        self_heal: bool = True,
        screenshot: bool = True,
    ) -> ActionResult:
        """Execute a single browser action.

        Args:
            url: URL of the page to test
            instruction: Natural language instruction (e.g., "Click the login button")
            self_heal: Enable self-healing for selectors
            screenshot: Capture screenshot after action

        Returns:
            ActionResult with success status and details
        """
        try:
            client = await self._get_client()
            response = await client.post(
                "/act",
                json={
                    "url": url,
                    "instruction": instruction,
                    "selfHeal": self_heal,
                    "screenshot": screenshot,
                },
                timeout=60.0,
            )

            if not response.is_success:
                return ActionResult(
                    success=False,
                    error=f"Action failed: {response.text}",
                )

            data = response.json()
            return ActionResult(
                success=data.get("success", False),
                message=data.get("message"),
                actions=data.get("actions"),
                screenshot=data.get("screenshot"),
                healed=data.get("healed", False),
                healing_method=data.get("healingMethod"),
            )

        except httpx.TimeoutException:
            return ActionResult(success=False, error="Action timed out")
        except Exception as e:
            logger.exception("Action execution failed", error=str(e))
            return ActionResult(success=False, error=str(e))

    async def run_test(
        self,
        url: str,
        steps: list[str],
        browser: str = "chrome",
        capture_screenshots: bool = True,
    ) -> TestResult:
        """Run a multi-step E2E test.

        Args:
            url: Application URL to test
            steps: Array of test step instructions
            browser: Browser to use (chrome, firefox, safari)
            capture_screenshots: Capture screenshots during test

        Returns:
            TestResult with step results and screenshots
        """
        try:
            # Calculate timeout based on number of steps
            timeout = max(120.0, len(steps) * 30.0 + 30.0)

            client = await self._get_client()
            response = await client.post(
                "/test",
                json={
                    "url": url,
                    "steps": steps,
                    "browser": browser,
                    "screenshot": True,
                    "captureScreenshots": capture_screenshots,
                },
                timeout=timeout,
            )

            if not response.is_success:
                return TestResult(
                    success=False,
                    error=f"Test failed: {response.text}",
                )

            data = response.json()
            return TestResult(
                success=data.get("success", False),
                steps=data.get("steps"),
                screenshots=data.get("screenshots"),
                final_screenshot=data.get("finalScreenshot"),
                healing_report=data.get("healingReport"),
            )

        except httpx.TimeoutException:
            return TestResult(success=False, error="Test timed out")
        except Exception as e:
            logger.exception("Test execution failed", error=str(e))
            return TestResult(success=False, error=str(e))

    async def discover_elements(
        self,
        url: str,
        instruction: Optional[str] = None,
    ) -> DiscoveryResult:
        """Discover interactive elements on a page.

        Args:
            url: URL to analyze
            instruction: What to look for (optional)

        Returns:
            DiscoveryResult with list of discovered actions
        """
        try:
            client = await self._get_client()
            response = await client.post(
                "/observe",
                json={
                    "url": url,
                    "instruction": instruction or "What actions can I take on this page?",
                },
                timeout=60.0,
            )

            if not response.is_success:
                return DiscoveryResult(
                    success=False,
                    error=f"Discovery failed: {response.text}",
                )

            data = response.json()
            return DiscoveryResult(
                success=True,
                actions=data.get("actions"),
            )

        except httpx.TimeoutException:
            return DiscoveryResult(success=False, error="Discovery timed out")
        except Exception as e:
            logger.exception("Discovery failed", error=str(e))
            return DiscoveryResult(success=False, error=str(e))

    async def extract_data(
        self,
        url: str,
        instruction: str,
        schema: Optional[dict] = None,
    ) -> ExtractionResult:
        """Extract structured data from a web page.

        Args:
            url: URL to extract data from
            instruction: What data to extract
            schema: Expected data schema (optional)

        Returns:
            ExtractionResult with extracted data
        """
        try:
            client = await self._get_client()
            response = await client.post(
                "/extract",
                json={
                    "url": url,
                    "instruction": instruction,
                    "schema": schema or {},
                },
                timeout=60.0,
            )

            if not response.is_success:
                return ExtractionResult(
                    success=False,
                    error=f"Extraction failed: {response.text}",
                )

            data = response.json()
            return ExtractionResult(
                success=True,
                data=data,
            )

        except httpx.TimeoutException:
            return ExtractionResult(success=False, error="Extraction timed out")
        except Exception as e:
            logger.exception("Extraction failed", error=str(e))
            return ExtractionResult(success=False, error=str(e))

    async def run_agent(
        self,
        url: str,
        instruction: str,
        max_steps: int = 10,
        capture_screenshots: bool = True,
    ) -> AgentResult:
        """Run an autonomous agent to complete a complex task.

        Args:
            url: Starting URL
            instruction: Task to complete
            max_steps: Maximum steps to take
            capture_screenshots: Capture screenshots during execution

        Returns:
            AgentResult with execution details
        """
        try:
            # Calculate timeout based on max steps
            timeout = max(120.0, max_steps * 30.0 + 60.0)

            client = await self._get_client()
            response = await client.post(
                "/agent",
                json={
                    "url": url,
                    "instruction": instruction,
                    "maxSteps": max_steps,
                    "captureScreenshots": capture_screenshots,
                },
                timeout=timeout,
            )

            if not response.is_success:
                return AgentResult(
                    success=False,
                    completed=False,
                    error=f"Agent failed: {response.text}",
                )

            data = response.json()
            return AgentResult(
                success=data.get("success", False),
                completed=data.get("completed", False),
                message=data.get("message"),
                actions=data.get("actions"),
                screenshots=data.get("screenshots"),
                usage=data.get("usage"),
            )

        except httpx.TimeoutException:
            return AgentResult(success=False, completed=False, error="Agent timed out")
        except Exception as e:
            logger.exception("Agent execution failed", error=str(e))
            return AgentResult(success=False, completed=False, error=str(e))

    # Convenience aliases
    async def discover(
        self,
        url: str,
        task: Optional[str] = None,
    ) -> DiscoveryResult:
        """Alias for discover_elements with task parameter."""
        return await self.discover_elements(url=url, instruction=task)

    async def extract(
        self,
        url: str,
        instruction: str,
        schema: Optional[dict] = None,
    ) -> ExtractionResult:
        """Alias for extract_data."""
        return await self.extract_data(url=url, instruction=instruction, schema=schema)


# Singleton instance
_client: Optional[BrowserWorkerClient] = None


def get_browser_client() -> BrowserWorkerClient:
    """Get the singleton browser worker client."""
    global _client
    if _client is None:
        _client = BrowserWorkerClient()
    return _client


async def cleanup_browser_client() -> None:
    """Cleanup the browser worker client."""
    global _client
    if _client:
        await _client.close()
        _client = None
