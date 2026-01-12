"""
Crawlee Service Client

HTTP client for communicating with the Crawlee microservice.
Handles discovery crawling, visual capture, and test execution.
"""

import os
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Configuration
CRAWLEE_SERVICE_URL = os.getenv("CRAWLEE_SERVICE_URL", "http://localhost:3000")
CRAWLEE_TIMEOUT = int(os.getenv("CRAWLEE_TIMEOUT", "120"))


@dataclass
class CrawleeResponse:
    """Response from Crawlee service"""
    success: bool
    request_id: str
    duration: int
    data: Dict[str, Any]
    error: Optional[str] = None


class CrawleeServiceUnavailable(Exception):
    """Raised when Crawlee service is not available"""
    pass


class CrawleeClient:
    """
    Client for the Crawlee microservice.

    Provides methods for:
    - Discovery crawling (graph-based web crawling)
    - Visual capture (screenshots with DOM extraction)
    - Test execution (running test steps)
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or CRAWLEE_SERVICE_URL
        self.timeout = httpx.Timeout(CRAWLEE_TIMEOUT, connect=10.0)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> Dict[str, Any]:
        """Check if Crawlee service is healthy"""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            raise CrawleeServiceUnavailable(
                f"Cannot connect to Crawlee service at {self.base_url}"
            )
        except Exception as e:
            raise CrawleeServiceUnavailable(f"Crawlee service error: {str(e)}")

    async def is_available(self) -> bool:
        """Check if service is available"""
        try:
            await self.health_check()
            return True
        except CrawleeServiceUnavailable:
            return False

    # ==========================================
    # Discovery Crawling
    # ==========================================

    async def run_discovery(
        self,
        start_url: str,
        max_pages: int = 50,
        max_depth: int = 3,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        capture_screenshots: bool = True,
        viewport: Optional[Dict[str, int]] = None,
        auth_config: Optional[Dict[str, Any]] = None
    ) -> CrawleeResponse:
        """
        Run a discovery crawl on a website.

        Args:
            start_url: URL to start crawling from
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth from start URL
            include_patterns: Glob patterns for URLs to include
            exclude_patterns: Glob patterns for URLs to exclude
            capture_screenshots: Whether to capture screenshots
            viewport: Browser viewport size
            auth_config: Authentication configuration

        Returns:
            CrawleeResponse with discovered pages, elements, and graph
        """
        try:
            client = await self._get_client()

            payload = {
                "startUrl": start_url,
                "maxPages": max_pages,
                "maxDepth": max_depth,
                "includePatterns": include_patterns or [],
                "excludePatterns": exclude_patterns or [],
                "captureScreenshots": capture_screenshots,
                "viewport": viewport or {"width": 1920, "height": 1080}
            }

            if auth_config:
                payload["authConfig"] = auth_config

            logger.info(f"Starting discovery crawl: {start_url}")
            response = await client.post("/api/v1/crawl/discovery", json=payload)
            response.raise_for_status()

            data = response.json()
            return CrawleeResponse(
                success=data.get("success", True),
                request_id=data.get("requestId", ""),
                duration=data.get("duration", 0),
                data=data.get("result", {}),
                error=data.get("error")
            )

        except httpx.ConnectError:
            raise CrawleeServiceUnavailable(
                f"Cannot connect to Crawlee service at {self.base_url}"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Discovery crawl failed: {e.response.text}")
            return CrawleeResponse(
                success=False,
                request_id="",
                duration=0,
                data={},
                error=str(e)
            )

    # ==========================================
    # Visual Capture
    # ==========================================

    async def capture_screenshot(
        self,
        url: str,
        viewport: Optional[Dict[str, int]] = None,
        full_page: bool = False,
        selector: Optional[str] = None,
        wait_for_selector: Optional[str] = None,
        wait_for_timeout: int = 5000,
        capture_dom: bool = True
    ) -> CrawleeResponse:
        """
        Capture a screenshot of a URL.

        Args:
            url: URL to capture
            viewport: Browser viewport size
            full_page: Whether to capture full page
            selector: Specific element to capture
            wait_for_selector: Wait for this selector before capture
            wait_for_timeout: Timeout for waiting
            capture_dom: Whether to capture DOM snapshot

        Returns:
            CrawleeResponse with screenshot and metadata
        """
        try:
            client = await self._get_client()

            payload = {
                "url": url,
                "viewport": viewport or {"width": 1920, "height": 1080},
                "fullPage": full_page,
                "waitForTimeout": wait_for_timeout,
                "captureDom": capture_dom
            }

            if selector:
                payload["selector"] = selector
            if wait_for_selector:
                payload["waitForSelector"] = wait_for_selector

            logger.info(f"Capturing screenshot: {url}")
            response = await client.post("/api/v1/capture/screenshot", json=payload)
            response.raise_for_status()

            data = response.json()
            return CrawleeResponse(
                success=data.get("success", True),
                request_id=data.get("requestId", ""),
                duration=data.get("duration", 0),
                data=data.get("result", {}),
                error=data.get("error")
            )

        except httpx.ConnectError:
            raise CrawleeServiceUnavailable(
                f"Cannot connect to Crawlee service at {self.base_url}"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Screenshot capture failed: {e.response.text}")
            return CrawleeResponse(
                success=False,
                request_id="",
                duration=0,
                data={},
                error=str(e)
            )

    async def capture_responsive(
        self,
        url: str,
        viewports: Optional[List[Dict[str, Any]]] = None
    ) -> CrawleeResponse:
        """
        Capture screenshots at multiple viewport sizes.

        Args:
            url: URL to capture
            viewports: List of viewport configurations

        Returns:
            CrawleeResponse with screenshots for each viewport
        """
        try:
            client = await self._get_client()

            payload = {
                "url": url,
                "viewports": viewports
            }

            logger.info(f"Capturing responsive screenshots: {url}")
            response = await client.post("/api/v1/capture/responsive", json=payload)
            response.raise_for_status()

            data = response.json()
            return CrawleeResponse(
                success=data.get("success", True),
                request_id=data.get("requestId", ""),
                duration=data.get("duration", 0),
                data=data.get("results", {}),
                error=data.get("error")
            )

        except httpx.ConnectError:
            raise CrawleeServiceUnavailable(
                f"Cannot connect to Crawlee service at {self.base_url}"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Responsive capture failed: {e.response.text}")
            return CrawleeResponse(
                success=False,
                request_id="",
                duration=0,
                data={},
                error=str(e)
            )

    # ==========================================
    # Test Execution
    # ==========================================

    async def execute_test(
        self,
        test_id: str,
        steps: List[Dict[str, Any]],
        base_url: str,
        viewport: Optional[Dict[str, int]] = None,
        timeout: int = 30000,
        capture_screenshots: bool = True,
        capture_video: bool = False
    ) -> CrawleeResponse:
        """
        Execute a test with given steps.

        Args:
            test_id: Test identifier
            steps: List of test steps to execute
            base_url: Base URL for relative URLs in steps
            viewport: Browser viewport size
            timeout: Timeout per step in ms
            capture_screenshots: Capture screenshots after each step
            capture_video: Record video of execution

        Returns:
            CrawleeResponse with test results
        """
        try:
            client = await self._get_client()

            payload = {
                "testId": test_id,
                "steps": steps,
                "baseUrl": base_url,
                "viewport": viewport or {"width": 1920, "height": 1080},
                "timeout": timeout,
                "captureScreenshots": capture_screenshots,
                "captureVideo": capture_video
            }

            logger.info(f"Executing test: {test_id}")
            response = await client.post("/api/v1/execute/test", json=payload)
            response.raise_for_status()

            data = response.json()
            return CrawleeResponse(
                success=data.get("success", True),
                request_id=data.get("requestId", ""),
                duration=data.get("duration", 0),
                data=data.get("result", {}),
                error=data.get("error")
            )

        except httpx.ConnectError:
            raise CrawleeServiceUnavailable(
                f"Cannot connect to Crawlee service at {self.base_url}"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Test execution failed: {e.response.text}")
            return CrawleeResponse(
                success=False,
                request_id="",
                duration=0,
                data={},
                error=str(e)
            )

    # ==========================================
    # Element Extraction
    # ==========================================

    async def extract_elements(
        self,
        url: str,
        selectors: Optional[List[str]] = None
    ) -> CrawleeResponse:
        """
        Extract interactive elements from a page.

        Args:
            url: URL to extract elements from
            selectors: Optional specific selectors to look for

        Returns:
            CrawleeResponse with extracted elements
        """
        try:
            client = await self._get_client()

            payload = {
                "url": url,
                "selectors": selectors
            }

            logger.info(f"Extracting elements: {url}")
            response = await client.post("/api/v1/extract/elements", json=payload)
            response.raise_for_status()

            data = response.json()
            return CrawleeResponse(
                success=data.get("success", True),
                request_id=data.get("requestId", ""),
                duration=data.get("duration", 0),
                data={
                    "elements": data.get("elements", []),
                    "forms": data.get("forms", []),
                    "links": data.get("links", [])
                },
                error=data.get("error")
            )

        except httpx.ConnectError:
            raise CrawleeServiceUnavailable(
                f"Cannot connect to Crawlee service at {self.base_url}"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Element extraction failed: {e.response.text}")
            return CrawleeResponse(
                success=False,
                request_id="",
                duration=0,
                data={},
                error=str(e)
            )


# Singleton instance
_client: Optional[CrawleeClient] = None


def get_crawlee_client() -> CrawleeClient:
    """Get the singleton Crawlee client"""
    global _client
    if _client is None:
        _client = CrawleeClient()
    return _client


async def check_crawlee_service() -> bool:
    """Check if Crawlee service is available"""
    client = get_crawlee_client()
    return await client.is_available()
