"""
Selenium Grid Client for Video Recording

This client is specifically for video-enabled browser sessions.
It connects to Selenium Grid (separate from browser-pool/Playwright)
which has ffmpeg video sidecars that auto-upload to R2.

Architecture:
- browser-pool (Playwright): Fast, no video recording
- selenium-grid (WebDriver): Video recording via sidecars, uploads to R2

Video flow:
1. Create session → video sidecar starts recording
2. Navigate, interact with pages
3. End session → video sidecar stops, uploads to r2://argus-artifacts/videos/{session-id}.mp4
"""

import asyncio
import base64
import re
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class SeleniumElement:
    """Element discovered on a page."""
    tag_name: str
    text: str
    selector: str
    element_type: str
    attributes: dict = field(default_factory=dict)


@dataclass
class SeleniumPage:
    """Page discovered during crawl."""
    url: str
    title: str
    elements: list[SeleniumElement]
    links: list[str]
    screenshot: str | None = None


@dataclass
class SeleniumDiscoveryResult:
    """Result from Selenium Grid discovery."""
    success: bool
    pages: list[SeleniumPage] = field(default_factory=list)
    session_id: str | None = None
    video_path: str | None = None
    video_artifact_id: str | None = None
    recording_url: str | None = None
    error: str | None = None
    duration_ms: int = 0


class SeleniumGridError(Exception):
    """Error from Selenium Grid."""
    pass


class SeleniumGridClient:
    """
    Client for Selenium Grid with video recording support.

    Uses WebDriver protocol (W3C standard) to control browsers.
    Video recording is handled automatically by ffmpeg sidecars in Kubernetes.
    """

    def __init__(self, grid_url: str | None = None):
        """
        Initialize the Selenium Grid client.

        Args:
            grid_url: Selenium Grid hub URL (e.g., http://65.20.71.218:4444)
        """
        settings = get_settings()
        self.grid_url = grid_url or settings.selenium_grid_url

        if not self.grid_url:
            raise SeleniumGridError(
                "Selenium Grid URL not configured. Set SELENIUM_GRID_URL env var."
            )

        # Normalize URL
        if not self.grid_url.endswith("/"):
            self.grid_url = self.grid_url + "/"

        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None

        logger.info("SeleniumGridClient initialized", grid_url=self.grid_url)

    async def __aenter__(self) -> "SeleniumGridClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.grid_url,
                headers={"Content-Type": "application/json"},
                timeout=httpx.Timeout(120.0),  # Long timeout for grid operations
            )

    async def close(self) -> None:
        """Close the client."""
        # End session if active
        if self._session_id:
            try:
                await self.end_session()
            except Exception as e:
                logger.warning("Failed to end session on close", error=str(e))

        if self._client:
            await self._client.aclose()
            self._client = None

    async def health(self) -> dict:
        """Check Selenium Grid health."""
        await self._ensure_client()
        try:
            response = await self._client.get("status")
            response.raise_for_status()
            data = response.json()
            return {
                "healthy": data.get("value", {}).get("ready", False),
                "nodes": len(data.get("value", {}).get("nodes", [])),
            }
        except Exception as e:
            logger.error("Selenium Grid health check failed", error=str(e))
            return {"healthy": False, "error": str(e)}

    async def start_session(
        self,
        browser: str = "chrome",
        capabilities: dict | None = None,
    ) -> str:
        """
        Start a new browser session.

        Video recording starts automatically via the sidecar container.

        Args:
            browser: Browser type (chrome, firefox, edge)
            capabilities: Additional WebDriver capabilities

        Returns:
            Session ID
        """
        await self._ensure_client()

        # WebDriver capabilities
        caps = {
            "browserName": browser,
            "platformName": "linux",
            # Enable video recording via SE_VIDEO env var in sidecar
            "se:recordVideo": True,
            "se:screenResolution": "1920x1080",
        }
        if capabilities:
            caps.update(capabilities)

        # W3C WebDriver session request
        payload = {
            "capabilities": {
                "alwaysMatch": caps,
            }
        }

        try:
            response = await self._client.post("session", json=payload)
            response.raise_for_status()
            data = response.json()

            self._session_id = data.get("value", {}).get("sessionId")
            if not self._session_id:
                raise SeleniumGridError("No session ID in response")

            logger.info(
                "Started Selenium session with video recording",
                session_id=self._session_id,
                browser=browser,
            )
            return self._session_id

        except httpx.HTTPStatusError as e:
            raise SeleniumGridError(f"Failed to start session: {e.response.text}")
        except Exception as e:
            raise SeleniumGridError(f"Failed to start session: {e}")

    async def end_session(self, wait_for_upload: bool = True) -> dict:
        """
        End the current browser session.

        This triggers the video sidecar to stop recording and upload to R2.
        Video upload happens asynchronously after session close.

        Architecture (from Selenium Grid docs):
        1. Session delete triggers video recorder to stop
        2. Video file is finalized as {session_id}.mp4
        3. Uploader copies to R2://argus-artifacts/videos/{session_id}.mp4
        4. This takes 5-30 seconds depending on video length

        Args:
            wait_for_upload: Wait a bit for upload to complete (default True)

        Returns:
            Session end result with video info
        """
        if not self._session_id:
            return {"success": False, "error": "No active session"}

        await self._ensure_client()
        session_id = self._session_id

        try:
            response = await self._client.delete(f"session/{session_id}")
            response.raise_for_status()

            # Video is uploaded to R2 by the Selenium Grid sidecar
            # Path convention: videos/{session_id}.mp4
            # SE_VIDEO_FILE_NAME=auto ensures filename = session_id
            video_path = f"videos/{session_id}.mp4"

            logger.info(
                "Ended Selenium session - video upload starting",
                session_id=session_id,
                video_path=video_path,
            )

            self._session_id = None

            # Give the uploader sidecar time to finish
            # Video upload is async and may take 5-30 seconds
            if wait_for_upload:
                logger.info("Waiting for video upload to complete...", session_id=session_id)
                await asyncio.sleep(10)  # Wait for upload to start/complete

            return {
                "success": True,
                "session_id": session_id,
                "video_path": video_path,
                "video_artifact_id": session_id,  # Session ID = video filename
            }

        except Exception as e:
            logger.error("Failed to end session", session_id=session_id, error=str(e))
            self._session_id = None
            return {"success": False, "error": str(e)}

    async def navigate(self, url: str) -> bool:
        """Navigate to a URL."""
        if not self._session_id:
            raise SeleniumGridError("No active session")

        await self._ensure_client()

        try:
            response = await self._client.post(
                f"session/{self._session_id}/url",
                json={"url": url}
            )
            response.raise_for_status()
            logger.debug("Navigated to URL", url=url)
            return True
        except Exception as e:
            logger.error("Navigation failed", url=url, error=str(e))
            return False

    async def get_page_info(self) -> dict:
        """Get current page info (URL, title)."""
        if not self._session_id:
            raise SeleniumGridError("No active session")

        await self._ensure_client()

        try:
            # Get URL
            url_response = await self._client.get(
                f"session/{self._session_id}/url"
            )
            url_response.raise_for_status()
            url = url_response.json().get("value", "")

            # Get title
            title_response = await self._client.get(
                f"session/{self._session_id}/title"
            )
            title_response.raise_for_status()
            title = title_response.json().get("value", "")

            return {"url": url, "title": title}
        except Exception as e:
            logger.error("Failed to get page info", error=str(e))
            return {"url": "", "title": "", "error": str(e)}

    async def screenshot(self) -> str | None:
        """Capture screenshot of current page."""
        if not self._session_id:
            return None

        await self._ensure_client()

        try:
            response = await self._client.get(
                f"session/{self._session_id}/screenshot"
            )
            response.raise_for_status()
            return response.json().get("value")  # Base64 encoded
        except Exception as e:
            logger.error("Screenshot failed", error=str(e))
            return None

    async def find_elements(self, css_selector: str = "*") -> list[dict]:
        """
        Find elements on the page using CSS selector.

        Args:
            css_selector: CSS selector (default: all elements)

        Returns:
            List of element info dicts
        """
        if not self._session_id:
            return []

        await self._ensure_client()

        try:
            response = await self._client.post(
                f"session/{self._session_id}/elements",
                json={"using": "css selector", "value": css_selector}
            )
            response.raise_for_status()
            elements = response.json().get("value", [])

            # Get details for each element
            result = []
            for el in elements[:100]:  # Limit to 100 elements
                element_id = el.get("ELEMENT") or el.get("element-6066-11e4-a52e-4f735466cecf")
                if element_id:
                    info = await self._get_element_info(element_id)
                    if info:
                        result.append(info)

            return result
        except Exception as e:
            logger.error("Find elements failed", selector=css_selector, error=str(e))
            return []

    async def _get_element_info(self, element_id: str) -> dict | None:
        """Get info about a specific element."""
        try:
            # Get tag name
            tag_response = await self._client.get(
                f"session/{self._session_id}/element/{element_id}/name"
            )
            tag_name = tag_response.json().get("value", "")

            # Get text
            text_response = await self._client.get(
                f"session/{self._session_id}/element/{element_id}/text"
            )
            text = text_response.json().get("value", "")

            # Get common attributes
            attrs = {}
            for attr in ["href", "type", "name", "id", "class", "placeholder", "value"]:
                try:
                    attr_response = await self._client.get(
                        f"session/{self._session_id}/element/{element_id}/attribute/{attr}"
                    )
                    value = attr_response.json().get("value")
                    if value:
                        attrs[attr] = value
                except Exception:
                    pass

            return {
                "tag_name": tag_name,
                "text": text[:200] if text else "",
                "attributes": attrs,
                "element_id": element_id,
            }
        except Exception:
            return None

    async def get_links(self) -> list[str]:
        """Get all links on the current page."""
        if not self._session_id:
            return []

        await self._ensure_client()

        try:
            response = await self._client.post(
                f"session/{self._session_id}/elements",
                json={"using": "css selector", "value": "a[href]"}
            )
            response.raise_for_status()
            elements = response.json().get("value", [])

            links = []
            for el in elements:
                element_id = el.get("ELEMENT") or el.get("element-6066-11e4-a52e-4f735466cecf")
                if element_id:
                    try:
                        href_response = await self._client.get(
                            f"session/{self._session_id}/element/{element_id}/attribute/href"
                        )
                        href = href_response.json().get("value")
                        if href and not href.startswith("#") and not href.startswith("javascript:"):
                            links.append(href)
                    except Exception:
                        pass

            return links
        except Exception as e:
            logger.error("Get links failed", error=str(e))
            return []

    async def discover(
        self,
        start_url: str,
        max_pages: int = 50,
        max_depth: int = 3,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        capture_screenshots: bool = True,
    ) -> SeleniumDiscoveryResult:
        """
        Discover pages by crawling from a starting URL.

        Video recording is automatically enabled for the entire session.

        Args:
            start_url: URL to start crawling from
            max_pages: Maximum pages to discover
            max_depth: Maximum crawl depth
            include_patterns: URL patterns to include
            exclude_patterns: URL patterns to exclude
            capture_screenshots: Capture screenshot of each page

        Returns:
            SeleniumDiscoveryResult with pages and video info
        """
        start_time = time.time()

        logger.info(
            "Starting Selenium Grid discovery with video",
            start_url=start_url,
            max_pages=max_pages,
        )

        # Compile patterns
        include_regex = [re.compile(p) for p in (include_patterns or [])]
        exclude_regex = [re.compile(p) for p in (exclude_patterns or [])]
        base_domain = urlparse(start_url).netloc

        def should_visit(url: str) -> bool:
            if urlparse(url).netloc != base_domain:
                return False
            for pattern in exclude_regex:
                if pattern.search(url):
                    return False
            if include_regex:
                return any(p.search(url) for p in include_regex)
            return True

        visited_urls: set[str] = set()
        discovered_pages: list[SeleniumPage] = []
        queue: list[tuple[str, int]] = [(start_url, 0)]

        try:
            # Start session (video recording begins)
            session_id = await self.start_session()

            while queue and len(discovered_pages) < max_pages:
                url, depth = queue.pop(0)

                if url in visited_urls:
                    continue
                if depth > max_depth:
                    continue
                if not should_visit(url):
                    continue

                visited_urls.add(url)

                try:
                    # Navigate to page
                    if not await self.navigate(url):
                        continue

                    # Wait for page to load
                    await asyncio.sleep(1)

                    # Get page info
                    page_info = await self.get_page_info()

                    # Find interactive elements
                    elements = await self.find_elements(
                        "a, button, input, select, textarea, [onclick], [role='button']"
                    )

                    # Get links for crawling
                    links = await self.get_links()
                    for link in links:
                        full_url = urljoin(url, link)
                        if depth < max_depth and full_url not in visited_urls:
                            queue.append((full_url, depth + 1))

                    # Capture screenshot
                    screenshot = None
                    if capture_screenshots:
                        screenshot = await self.screenshot()

                    # Create page object
                    page = SeleniumPage(
                        url=page_info.get("url", url),
                        title=page_info.get("title", ""),
                        elements=[
                            SeleniumElement(
                                tag_name=e.get("tag_name", ""),
                                text=e.get("text", ""),
                                selector=f"[id='{e['attributes'].get('id')}']" if e.get("attributes", {}).get("id") else e.get("tag_name", ""),
                                element_type=e.get("tag_name", ""),
                                attributes=e.get("attributes", {}),
                            )
                            for e in elements
                        ],
                        links=links,
                        screenshot=screenshot,
                    )
                    discovered_pages.append(page)

                    logger.debug(
                        "Discovered page",
                        url=url,
                        depth=depth,
                        elements=len(elements),
                        links=len(links),
                    )

                except Exception as e:
                    logger.warning("Error discovering page", url=url, error=str(e))
                    continue

            # End session (video stops, uploads to R2)
            end_result = await self.end_session()

            duration_ms = int((time.time() - start_time) * 1000)

            # Get video info
            video_artifact_id = end_result.get("video_artifact_id")
            video_path = end_result.get("video_path")

            # Generate signed URL for video (will be available after upload completes)
            recording_url = None
            if video_artifact_id:
                settings = get_settings()
                # Video is at: videos/{session_id}.mp4 in R2
                recording_url = f"{settings.cloudflare_worker_url}/artifacts/video/{video_artifact_id}"

            logger.info(
                "Selenium discovery completed",
                pages=len(discovered_pages),
                session_id=session_id,
                video_artifact_id=video_artifact_id,
                duration_ms=duration_ms,
            )

            return SeleniumDiscoveryResult(
                success=True,
                pages=discovered_pages,
                session_id=session_id,
                video_path=video_path,
                video_artifact_id=video_artifact_id,
                recording_url=recording_url,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error("Selenium discovery failed", error=str(e))

            # Try to end session on error
            try:
                await self.end_session()
            except Exception:
                pass

            return SeleniumDiscoveryResult(
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )


# Singleton instance
_default_client: SeleniumGridClient | None = None


def get_selenium_grid_client() -> SeleniumGridClient:
    """Get the default Selenium Grid client instance."""
    global _default_client
    if _default_client is None:
        _default_client = SeleniumGridClient()
    return _default_client


async def is_selenium_grid_available() -> bool:
    """Check if Selenium Grid is available."""
    settings = get_settings()
    if not settings.selenium_grid_url:
        logger.warning("Selenium Grid URL not configured")
        return False

    logger.info("Checking Selenium Grid availability", url=settings.selenium_grid_url)
    try:
        async with SeleniumGridClient() as client:
            health = await client.health()
            is_healthy = health.get("healthy", False)
            logger.info("Selenium Grid health check result", healthy=is_healthy, health=health)
            return is_healthy
    except Exception as e:
        logger.error("Selenium Grid availability check failed", error=str(e))
        return False
