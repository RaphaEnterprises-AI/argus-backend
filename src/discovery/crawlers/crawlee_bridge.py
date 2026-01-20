"""Bridge to Crawlee for autonomous web crawling.

This module provides a Python bridge to Crawlee (Node.js crawling framework) or
a fallback Playwright-based crawler when Crawlee is not available.

Crawlee Features:
- Automatic browser management
- Smart request queue handling
- Built-in rate limiting and politeness
- Session rotation and proxy support
- Automatic data extraction
"""

import asyncio
import base64
import json
import re
import tempfile
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import structlog
from functools import wraps

from src.discovery.models import (
    CrawlError,
    CrawlResult,
    DiscoveredElement,
    DiscoveredPage,
    DiscoveryConfig,
    ElementBounds,
    ElementCategory,
    PageCategory,
    PageGraph,
)

logger = structlog.get_logger()


@dataclass
class CrawlProgress:
    """Progress update from the crawler.

    Attributes:
        pages_crawled: Number of pages successfully crawled
        pages_queued: Number of pages in queue
        current_url: URL currently being processed
        errors_count: Number of errors encountered
        progress_percent: Overall progress percentage
    """
    pages_crawled: int = 0
    pages_queued: int = 0
    current_url: str = ""
    errors_count: int = 0
    progress_percent: float = 0.0


async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    recoverable_errors: tuple = (TimeoutError, ConnectionError, asyncio.TimeoutError),
    logger=None,
):
    """Execute an async function with exponential backoff retry.

    Args:
        func: Async callable to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Multiplier for exponential backoff
        recoverable_errors: Exception types that trigger retry
        logger: Logger instance for retry messages

    Returns:
        Result from successful function execution

    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except recoverable_errors as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                if logger:
                    logger.warning(
                        "Retry attempt",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=delay,
                        error=str(e),
                    )
                await asyncio.sleep(delay)
            else:
                raise
    raise last_exception


class CrawleeBridge:
    """Bridge to Crawlee or fallback Playwright-based crawler.

    This class provides a unified interface for web crawling, using Crawlee
    (via subprocess) when available, or falling back to a pure Playwright
    implementation for simplicity.

    Example:
        bridge = CrawleeBridge()
        result = await bridge.run_crawl(config)
        for url, page in result.pages.items():
            print(f"Found: {page.title}")
    """

    def __init__(
        self,
        use_crawlee: bool = False,
        crawlee_script_path: str | None = None,
    ):
        """Initialize the crawler bridge.

        Args:
            use_crawlee: Whether to attempt using Crawlee (requires Node.js)
            crawlee_script_path: Path to custom Crawlee crawler script
        """
        self.use_crawlee = use_crawlee
        self.crawlee_script_path = crawlee_script_path
        self.log = logger.bind(component="crawlee_bridge")

        # Crawl state
        self._visited_urls: set[str] = set()
        self._queued_urls: set[str] = set()
        self._current_depth: dict[str, int] = {}
        self._graph = PageGraph()
        self._errors: list[CrawlError] = []
        self._progress_callback: Callable[[CrawlProgress], None] | None = None

    def set_progress_callback(
        self,
        callback: Callable[[CrawlProgress], None],
    ) -> None:
        """Set callback for progress updates.

        Args:
            callback: Function to call with progress updates
        """
        self._progress_callback = callback

    async def run_crawl(
        self,
        start_url: str,
        config: DiscoveryConfig,
    ) -> CrawlResult:
        """Run the crawl operation.

        Args:
            start_url: URL to start crawling from
            config: Discovery configuration

        Returns:
            CrawlResult with discovered pages and metadata
        """
        self.log.info(
            "Starting crawl",
            start_url=start_url,
            max_pages=config.max_pages,
            max_depth=config.max_depth,
        )

        # Reset state
        self._visited_urls.clear()
        self._queued_urls.clear()
        self._current_depth.clear()
        self._graph = PageGraph()
        self._errors.clear()

        start_time = datetime.utcnow()

        try:
            if self.use_crawlee:
                result = await self._run_crawlee(start_url, config)
            else:
                result = await self._run_playwright_crawler(start_url, config)
        except Exception as e:
            self.log.exception("Crawl failed", error=str(e))
            result = CrawlResult(
                pages={},
                errors=[
                    CrawlError(
                        url=start_url,
                        error_type="crawl_failed",
                        message=str(e),
                        recoverable=False,
                    )
                ],
            )

        end_time = datetime.utcnow()
        result.duration_ms = int((end_time - start_time).total_seconds() * 1000)
        result.graph = self._graph
        result.errors.extend(self._errors)

        self.log.info(
            "Crawl completed",
            pages_found=len(result.pages),
            duration_ms=result.duration_ms,
            errors=len(result.errors),
        )

        return result

    async def _run_crawlee(
        self,
        start_url: str,
        config: DiscoveryConfig,
    ) -> CrawlResult:
        """Run crawl using Crawlee (Node.js).

        Args:
            start_url: Starting URL
            config: Discovery configuration

        Returns:
            CrawlResult with discovered pages
        """
        # Create config file for Crawlee
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            crawl_config = {
                "startUrl": start_url,
                "maxPages": config.max_pages,
                "maxDepth": config.max_depth,
                "includePatterns": config.include_patterns,
                "excludePatterns": config.exclude_patterns,
                "captureScreenshots": config.capture_screenshots,
                "captureDom": config.capture_dom,
            }
            json.dump(crawl_config, f)
            config_path = f.name

        # Create output directory
        with tempfile.TemporaryDirectory() as output_dir:
            # Run Crawlee script
            script_path = self.crawlee_script_path or self._get_default_crawlee_script()

            try:
                process = await asyncio.create_subprocess_exec(
                    "npx",
                    "tsx",
                    script_path,
                    "--config",
                    config_path,
                    "--output",
                    output_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.max_duration_seconds,
                )

                if process.returncode != 0:
                    raise RuntimeError(f"Crawlee failed: {stderr.decode()}")

                # Parse results
                result_path = Path(output_dir) / "results.json"
                if result_path.exists():
                    with open(result_path) as f:
                        data = json.load(f)
                        return self._parse_crawlee_results(data)
                else:
                    raise RuntimeError("No results file generated")

            except TimeoutError:
                process.kill()
                raise RuntimeError("Crawl timed out")
            except FileNotFoundError:
                self.log.warning("Crawlee not available, falling back to Playwright")
                return await self._run_playwright_crawler(start_url, config)

    def _get_default_crawlee_script(self) -> str:
        """Get path to default Crawlee script."""
        # Look for script in the discovery/crawlers directory
        script_path = Path(__file__).parent / "crawlee_crawler.ts"
        if script_path.exists():
            return str(script_path)
        raise FileNotFoundError("Default Crawlee script not found")

    def _parse_crawlee_results(self, data: dict[str, Any]) -> CrawlResult:
        """Parse results from Crawlee output.

        Args:
            data: JSON data from Crawlee

        Returns:
            CrawlResult with parsed pages
        """
        pages = {}

        for page_data in data.get("pages", []):
            page_id = str(uuid.uuid4())
            url = page_data.get("url", "")

            elements = []
            for el_data in page_data.get("elements", []):
                elements.append(
                    DiscoveredElement(
                        id=str(uuid.uuid4()),
                        page_url=url,
                        selector=el_data.get("selector", ""),
                        xpath=el_data.get("xpath"),
                        category=ElementCategory(el_data.get("category", "content")),
                        purpose=el_data.get("purpose"),
                        label=el_data.get("label"),
                        tag_name=el_data.get("tagName", "div"),
                        html_attributes=el_data.get("attributes", {}),
                    )
                )

            page = DiscoveredPage(
                id=page_id,
                url=url,
                title=page_data.get("title"),
                description=page_data.get("description"),
                category=PageCategory(page_data.get("category", "other")),
                elements=elements,
                outgoing_links=set(page_data.get("outgoingLinks", [])),
                depth=page_data.get("depth", 0),
                screenshot_base64=page_data.get("screenshot"),
                load_time_ms=page_data.get("loadTimeMs"),
            )
            pages[url] = page

        return CrawlResult(pages=pages)

    async def _run_playwright_crawler(
        self,
        start_url: str,
        config: DiscoveryConfig,
    ) -> CrawlResult:
        """Run crawl using Playwright directly.

        This is the fallback when Crawlee is not available.

        Args:
            start_url: Starting URL
            config: Discovery configuration

        Returns:
            CrawlResult with discovered pages
        """
        pages: dict[str, DiscoveredPage] = {}
        base_domain = urlparse(start_url).netloc

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            self.log.error("Playwright not installed")
            return CrawlResult(
                pages={},
                errors=[
                    CrawlError(
                        url=start_url,
                        error_type="dependency_missing",
                        message="Playwright not installed. Run: pip install playwright && playwright install",
                        recoverable=False,
                    )
                ],
            )

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=config.auth_config.headers.get("User-Agent")
                if config.auth_config
                else None,
            )

            # Set auth cookies if provided
            if config.auth_config and config.auth_config.cookies:
                cookies = [
                    {
                        "name": name,
                        "value": value,
                        "domain": base_domain,
                        "path": "/",
                    }
                    for name, value in config.auth_config.cookies.items()
                ]
                await context.add_cookies(cookies)

            # Queue the start URL
            self._queued_urls.add(start_url)
            self._current_depth[start_url] = 0

            try:
                while self._queued_urls and len(pages) < config.max_pages:
                    # Get next URL to crawl
                    url = self._queued_urls.pop()

                    # Skip if already visited
                    if url in self._visited_urls:
                        continue

                    # Check depth limit
                    depth = self._current_depth.get(url, 0)
                    if depth > config.max_depth:
                        continue

                    # Check URL patterns
                    if not self._should_crawl_url(url, base_domain, config):
                        continue

                    self._visited_urls.add(url)

                    # Emit progress
                    self._emit_progress(
                        pages_crawled=len(pages),
                        pages_queued=len(self._queued_urls),
                        current_url=url,
                    )

                    # Crawl the page
                    page_result = await self._crawl_single_page(
                        context,
                        url,
                        depth,
                        config,
                        base_domain,
                    )

                    if page_result:
                        pages[url] = page_result

                        # Queue new links
                        for link in page_result.outgoing_links:
                            full_url = urljoin(url, link)
                            if full_url not in self._visited_urls:
                                self._queued_urls.add(full_url)
                                self._current_depth[full_url] = depth + 1

                                # Add to graph
                                self._graph.add_edge(url, full_url)

            finally:
                await context.close()
                await browser.close()

        return CrawlResult(pages=pages)

    async def _crawl_single_page(
        self,
        context,
        url: str,
        depth: int,
        config: DiscoveryConfig,
        base_domain: str,
    ) -> DiscoveredPage | None:
        """Crawl a single page and extract information with self-healing.

        Enhanced with:
        - Retry logic for transient failures
        - Intelligent waiting for dynamic content (mutation observer)
        - Shadow DOM and iframe support
        - Robust error recovery

        Args:
            context: Playwright browser context
            url: URL to crawl
            depth: Current crawl depth
            config: Discovery configuration
            base_domain: Base domain for link filtering

        Returns:
            DiscoveredPage or None if crawl failed
        """
        page = await context.new_page()

        try:
            # Navigate with retry logic
            start_time = datetime.utcnow()
            response = await self._navigate_with_retry(page, url, max_retries=3)
            load_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            if not response or response.status >= 400:
                # Check if this is a recoverable error
                is_recoverable = response.status in [429, 502, 503, 504] if response else True
                self._errors.append(
                    CrawlError(
                        url=url,
                        error_type="http_error",
                        message=f"HTTP {response.status if response else 'no response'}",
                        recoverable=is_recoverable,
                    )
                )
                return None

            # Intelligent waiting: Wait for DOM mutations to settle
            await self._wait_for_dynamic_content(page)

            # Get page info
            title = await page.title()

            # Get meta description
            description = await page.evaluate("""
                () => {
                    const meta = document.querySelector('meta[name="description"]');
                    return meta ? meta.getAttribute('content') : null;
                }
            """)

            # Extract elements with self-healing (includes shadow DOM and iframes)
            elements = await self._extract_elements_robust(page, url)

            # Extract links
            links = await self._extract_links(page, base_domain)

            # Take screenshot if enabled
            screenshot_base64 = None
            if config.capture_screenshots:
                screenshot_bytes = await page.screenshot()
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode()

            # Classify page category
            category = self._classify_page(url, title, elements)

            self.log.debug(
                "Page crawled successfully",
                url=url,
                elements_count=len(elements),
                links_count=len(links),
                load_time_ms=load_time,
            )

            return DiscoveredPage(
                id=str(uuid.uuid4()),
                url=url,
                title=title,
                description=description,
                category=category,
                elements=elements,
                outgoing_links=set(links),
                depth=depth,
                screenshot_base64=screenshot_base64,
                load_time_ms=load_time,
            )

        except Exception as e:
            error_type = "crawl_error"
            is_recoverable = False

            # Categorize errors for better handling
            error_str = str(e).lower()
            if "timeout" in error_str:
                error_type = "timeout"
                is_recoverable = True
            elif "navigation" in error_str:
                error_type = "navigation_error"
                is_recoverable = True
            elif "net::" in error_str:
                error_type = "network_error"
                is_recoverable = True

            self.log.warning(
                "Failed to crawl page",
                url=url,
                error_type=error_type,
                recoverable=is_recoverable,
                error=str(e),
            )
            self._errors.append(
                CrawlError(
                    url=url,
                    error_type=error_type,
                    message=str(e),
                    recoverable=is_recoverable,
                )
            )
            return None

        finally:
            await page.close()

    async def _navigate_with_retry(
        self,
        page,
        url: str,
        max_retries: int = 3,
    ):
        """Navigate to URL with retry logic for transient failures.

        Args:
            page: Playwright page
            url: URL to navigate to
            max_retries: Maximum retry attempts

        Returns:
            Response object from successful navigation
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await page.goto(
                    url,
                    wait_until="domcontentloaded",  # Faster than networkidle
                    timeout=30000,
                )
                # Try to wait for network idle, but don't fail if it times out
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass  # Page may have ongoing XHR, that's ok
                return response
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Only retry on transient errors
                if any(err in error_str for err in ["timeout", "net::", "connection"]):
                    if attempt < max_retries - 1:
                        delay = (attempt + 1) * 2  # 2s, 4s, 6s
                        self.log.warning(
                            "Navigation retry",
                            url=url,
                            attempt=attempt + 1,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
                        continue
                raise

        raise last_error

    async def _wait_for_dynamic_content(
        self,
        page,
        timeout_ms: int = 5000,
        stability_ms: int = 500,
    ) -> None:
        """Wait for dynamic content to settle using mutation observer.

        This is smarter than a fixed wait - it observes DOM mutations and
        waits until the page stabilizes (no mutations for stability_ms).

        Args:
            page: Playwright page
            timeout_ms: Maximum time to wait
            stability_ms: Time with no mutations to consider stable
        """
        try:
            await page.evaluate(f"""
                () => new Promise((resolve) => {{
                    let lastMutation = Date.now();
                    const observer = new MutationObserver(() => {{
                        lastMutation = Date.now();
                    }});

                    observer.observe(document.body, {{
                        childList: true,
                        subtree: true,
                        attributes: true,
                        characterData: true,
                    }});

                    const checkStable = () => {{
                        const elapsed = Date.now() - lastMutation;
                        if (elapsed >= {stability_ms}) {{
                            observer.disconnect();
                            resolve();
                        }} else {{
                            setTimeout(checkStable, 100);
                        }}
                    }};

                    // Start checking after initial delay
                    setTimeout(checkStable, 200);

                    // Timeout fallback
                    setTimeout(() => {{
                        observer.disconnect();
                        resolve();
                    }}, {timeout_ms});
                }})
            """)
        except Exception as e:
            # If mutation observer fails, fall back to simple wait
            self.log.debug("Mutation observer failed, using fallback", error=str(e))
            await page.wait_for_timeout(1000)

    async def _extract_elements_robust(
        self,
        page,
        url: str,
    ) -> list[DiscoveredElement]:
        """Extract elements with robust self-healing approach.

        Enhanced extraction that:
        - Pierces shadow DOM
        - Traverses iframes
        - Generates multiple selector strategies
        - Handles JavaScript framework components

        Args:
            page: Playwright page
            url: Current page URL

        Returns:
            List of discovered elements
        """
        elements = []

        # Extract from main document
        main_elements = await self._extract_elements(page, url)
        elements.extend(main_elements)

        # Extract from shadow DOMs
        try:
            shadow_elements = await self._extract_shadow_dom_elements(page, url)
            elements.extend(shadow_elements)
        except Exception as e:
            self.log.debug("Shadow DOM extraction failed", error=str(e))

        # Extract from iframes
        try:
            iframe_elements = await self._extract_iframe_elements(page, url)
            elements.extend(iframe_elements)
        except Exception as e:
            self.log.debug("Iframe extraction failed", error=str(e))

        return elements

    async def _extract_shadow_dom_elements(
        self,
        page,
        url: str,
    ) -> list[DiscoveredElement]:
        """Extract elements from shadow DOMs.

        Args:
            page: Playwright page
            url: Current page URL

        Returns:
            List of elements found in shadow DOMs
        """
        shadow_data = await page.evaluate("""
            () => {
                const elements = [];

                function traverseShadowRoots(root, path = '') {
                    const shadowHosts = root.querySelectorAll('*');
                    shadowHosts.forEach((host, hostIdx) => {
                        if (host.shadowRoot) {
                            const shadowPath = path + (host.id ? `#${host.id}` : `[${hostIdx}]`) + ' >>> ';

                            // Extract interactive elements from shadow root
                            const interactives = host.shadowRoot.querySelectorAll(
                                'button, a, input, textarea, select, [role="button"], [onclick]'
                            );

                            interactives.forEach((el, idx) => {
                                const rect = el.getBoundingClientRect();
                                elements.push({
                                    type: el.tagName.toLowerCase(),
                                    shadowPath: shadowPath,
                                    selector: el.id ? `#${el.id}` :
                                             el.className ? `.${el.className.split(' ')[0]}` :
                                             `${el.tagName.toLowerCase()}:nth-of-type(${idx + 1})`,
                                    label: el.textContent?.trim()?.slice(0, 100) ||
                                           el.getAttribute('aria-label') || '',
                                    bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                                    visible: rect.width > 0 && rect.height > 0,
                                    inShadowDom: true,
                                });
                            });

                            // Recurse into nested shadow roots
                            traverseShadowRoots(host.shadowRoot, shadowPath);
                        }
                    });
                }

                traverseShadowRoots(document);
                return elements;
            }
        """)

        elements = []
        for el_data in shadow_data:
            bounds_data = el_data.get("bounds", {})
            # Create a pierce selector for shadow DOM
            shadow_selector = el_data.get("shadowPath", "") + el_data.get("selector", "")

            elements.append(
                DiscoveredElement(
                    id=str(uuid.uuid4()),
                    page_url=url,
                    selector=shadow_selector,
                    category=self._categorize_element(el_data),
                    label=el_data.get("label"),
                    bounds=ElementBounds(
                        x=bounds_data.get("x", 0),
                        y=bounds_data.get("y", 0),
                        width=bounds_data.get("width", 0),
                        height=bounds_data.get("height", 0),
                    ),
                    tag_name=el_data.get("type", "div"),
                    html_attributes={"_shadow_dom": True},
                    is_visible=el_data.get("visible", True),
                )
            )

        return elements

    async def _extract_iframe_elements(
        self,
        page,
        url: str,
    ) -> list[DiscoveredElement]:
        """Extract elements from iframes.

        Args:
            page: Playwright page
            url: Current page URL

        Returns:
            List of elements found in iframes
        """
        elements = []

        # Get all frames
        frames = page.frames
        for frame in frames:
            # Skip the main frame
            if frame == page.main_frame:
                continue

            # Skip cross-origin frames (can't access)
            try:
                frame_url = frame.url
                if not frame_url or frame_url == "about:blank":
                    continue

                # Extract elements from this frame
                frame_elements = await frame.evaluate("""
                    () => {
                        const elements = [];
                        document.querySelectorAll(
                            'button, a[href], input, textarea, select, [role="button"]'
                        ).forEach((el, idx) => {
                            const rect = el.getBoundingClientRect();
                            elements.push({
                                type: el.tagName.toLowerCase(),
                                selector: el.id ? `#${el.id}` :
                                         el.name ? `[name="${el.name}"]` :
                                         `${el.tagName.toLowerCase()}:nth-of-type(${idx + 1})`,
                                label: el.textContent?.trim()?.slice(0, 100) ||
                                       el.getAttribute('aria-label') || '',
                                bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                                visible: rect.width > 0 && rect.height > 0,
                            });
                        });
                        return elements;
                    }
                """)

                for el_data in frame_elements:
                    bounds_data = el_data.get("bounds", {})
                    # Create iframe-aware selector
                    iframe_selector = f"iframe >> {el_data.get('selector', '')}"

                    elements.append(
                        DiscoveredElement(
                            id=str(uuid.uuid4()),
                            page_url=url,
                            selector=iframe_selector,
                            category=self._categorize_element(el_data),
                            label=el_data.get("label"),
                            bounds=ElementBounds(
                                x=bounds_data.get("x", 0),
                                y=bounds_data.get("y", 0),
                                width=bounds_data.get("width", 0),
                                height=bounds_data.get("height", 0),
                            ),
                            tag_name=el_data.get("type", "div"),
                            html_attributes={"_iframe": True, "_frame_url": frame_url},
                            is_visible=el_data.get("visible", True),
                        )
                    )

            except Exception as e:
                # Cross-origin or other access error
                self.log.debug("Cannot access iframe", error=str(e))
                continue

        return elements

    async def _extract_elements(
        self,
        page,
        url: str,
    ) -> list[DiscoveredElement]:
        """Extract interactive elements from a page.

        Args:
            page: Playwright page
            url: Current page URL

        Returns:
            List of discovered elements
        """
        elements_data = await page.evaluate("""
            () => {
                const elements = [];

                // Buttons
                document.querySelectorAll('button, [role="button"], input[type="submit"], input[type="button"]').forEach((el, idx) => {
                    const rect = el.getBoundingClientRect();
                    elements.push({
                        type: 'button',
                        tagName: el.tagName.toLowerCase(),
                        selector: el.id ? `#${el.id}` :
                                  el.className ? `.${el.className.split(' ')[0]}` :
                                  `button:nth-of-type(${idx + 1})`,
                        label: el.textContent?.trim() || el.value || el.getAttribute('aria-label') || '',
                        bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                        visible: rect.width > 0 && rect.height > 0,
                        attributes: {
                            id: el.id,
                            class: el.className,
                            type: el.type,
                            'aria-label': el.getAttribute('aria-label'),
                        }
                    });
                });

                // Links
                document.querySelectorAll('a[href]').forEach((el, idx) => {
                    const rect = el.getBoundingClientRect();
                    const href = el.getAttribute('href');
                    if (!href?.startsWith('#') && !href?.startsWith('javascript:')) {
                        elements.push({
                            type: 'link',
                            tagName: 'a',
                            selector: el.id ? `#${el.id}` :
                                      el.className ? `a.${el.className.split(' ')[0]}` :
                                      `a:nth-of-type(${idx + 1})`,
                            label: el.textContent?.trim() || el.getAttribute('aria-label') || '',
                            bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                            visible: rect.width > 0 && rect.height > 0,
                            attributes: {
                                id: el.id,
                                href: href,
                                class: el.className,
                            }
                        });
                    }
                });

                // Form inputs
                document.querySelectorAll('input, textarea, select').forEach((el, idx) => {
                    const rect = el.getBoundingClientRect();
                    const type = el.type || el.tagName.toLowerCase();
                    if (!['hidden', 'submit', 'button'].includes(type)) {
                        elements.push({
                            type: 'input',
                            tagName: el.tagName.toLowerCase(),
                            inputType: type,
                            selector: el.id ? `#${el.id}` :
                                      el.name ? `[name="${el.name}"]` :
                                      `${el.tagName.toLowerCase()}:nth-of-type(${idx + 1})`,
                            label: el.placeholder || el.getAttribute('aria-label') ||
                                   document.querySelector(`label[for="${el.id}"]`)?.textContent?.trim() || '',
                            bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                            visible: rect.width > 0 && rect.height > 0,
                            required: el.required,
                            attributes: {
                                id: el.id,
                                name: el.name,
                                type: type,
                                placeholder: el.placeholder,
                                required: el.required,
                            }
                        });
                    }
                });

                return elements;
            }
        """)

        elements = []
        for el_data in elements_data:
            category = self._categorize_element(el_data)
            bounds_data = el_data.get("bounds", {})

            elements.append(
                DiscoveredElement(
                    id=str(uuid.uuid4()),
                    page_url=url,
                    selector=el_data.get("selector", ""),
                    category=category,
                    label=el_data.get("label"),
                    bounds=ElementBounds(
                        x=bounds_data.get("x", 0),
                        y=bounds_data.get("y", 0),
                        width=bounds_data.get("width", 0),
                        height=bounds_data.get("height", 0),
                    ),
                    tag_name=el_data.get("tagName", "div"),
                    html_attributes=el_data.get("attributes", {}),
                    is_visible=el_data.get("visible", True),
                    is_required=el_data.get("required", False),
                )
            )

        return elements

    async def _extract_links(
        self,
        page,
        base_domain: str,
    ) -> list[str]:
        """Extract links from a page.

        Args:
            page: Playwright page
            base_domain: Base domain for filtering

        Returns:
            List of link URLs
        """
        links = await page.evaluate("""
            () => {
                const links = [];
                document.querySelectorAll('a[href]').forEach(el => {
                    const href = el.getAttribute('href');
                    if (href && !href.startsWith('#') && !href.startsWith('javascript:') && !href.startsWith('mailto:')) {
                        links.push(href);
                    }
                });
                return [...new Set(links)];
            }
        """)

        # Filter to same domain
        page_url = page.url
        filtered_links = []
        for link in links:
            full_url = urljoin(page_url, link)
            parsed = urlparse(full_url)
            if parsed.netloc == base_domain or parsed.netloc == "":
                filtered_links.append(full_url)

        return filtered_links

    def _should_crawl_url(
        self,
        url: str,
        base_domain: str,
        config: DiscoveryConfig,
    ) -> bool:
        """Check if URL should be crawled based on config.

        Args:
            url: URL to check
            base_domain: Base domain
            config: Discovery configuration

        Returns:
            True if URL should be crawled
        """
        parsed = urlparse(url)

        # Check domain
        if parsed.netloc != base_domain:
            return False

        # Check include patterns
        if config.include_patterns:
            matched = False
            for pattern in config.include_patterns:
                if re.search(pattern, url):
                    matched = True
                    break
            if not matched:
                return False

        # Check exclude patterns
        for pattern in config.exclude_patterns:
            if re.search(pattern, url):
                return False

        return True

    def _categorize_element(self, el_data: dict[str, Any]) -> ElementCategory:
        """Categorize an element based on its properties.

        Args:
            el_data: Element data dictionary

        Returns:
            ElementCategory for the element
        """
        el_type = el_data.get("type", "")
        label = (el_data.get("label") or "").lower()
        el_data.get("attributes", {})

        # Check for authentication elements
        if any(
            kw in label
            for kw in ["login", "sign in", "log in", "logout", "log out", "sign out"]
        ):
            return ElementCategory.authentication

        if el_data.get("inputType") in ["password", "email"]:
            return ElementCategory.authentication

        # Check for commerce elements (by label, before generic type checks)
        if any(
            kw in label
            for kw in ["buy", "cart", "checkout", "purchase", "add to"]
        ):
            return ElementCategory.commerce

        # Check for form elements
        if el_type == "input":
            return ElementCategory.form

        # Check for action elements
        if el_type == "button":
            return ElementCategory.action

        # Check for navigation elements
        if el_type == "link":
            return ElementCategory.navigation

        return ElementCategory.content

    def _classify_page(
        self,
        url: str,
        title: str,
        elements: list[DiscoveredElement],
    ) -> PageCategory:
        """Classify a page based on URL, title, and elements.

        Args:
            url: Page URL
            title: Page title
            elements: Discovered elements

        Returns:
            PageCategory for the page
        """
        url_lower = url.lower()
        title_lower = (title or "").lower()

        # Check URL patterns
        if any(kw in url_lower for kw in ["/login", "/signin", "/auth"]):
            return PageCategory.auth_login

        if any(kw in url_lower for kw in ["/signup", "/register", "/join"]):
            return PageCategory.auth_signup

        if any(kw in url_lower for kw in ["/reset", "/forgot"]):
            return PageCategory.auth_reset

        if any(kw in url_lower for kw in ["/dashboard", "/home", "/overview"]):
            return PageCategory.dashboard

        if any(kw in url_lower for kw in ["/settings", "/preferences", "/config"]):
            return PageCategory.settings

        if any(kw in url_lower for kw in ["/profile", "/account", "/me"]):
            return PageCategory.profile

        if any(kw in url_lower for kw in ["/checkout", "/payment", "/cart"]):
            return PageCategory.checkout

        if any(kw in url_lower for kw in ["/404", "/error", "/not-found"]):
            return PageCategory.error

        # Check for list pages
        if any(kw in url_lower for kw in ["/list", "/all", "/browse", "?page="]):
            return PageCategory.list

        # Check element composition
        auth_elements = [
            e
            for e in elements
            if e.category == ElementCategory.authentication
        ]
        form_elements = [e for e in elements if e.category == ElementCategory.form]

        if len(auth_elements) >= 2:
            return PageCategory.auth_login

        if len(form_elements) >= 3:
            return PageCategory.form

        # Check title
        if "login" in title_lower or "sign in" in title_lower:
            return PageCategory.auth_login

        if "register" in title_lower or "sign up" in title_lower:
            return PageCategory.auth_signup

        if "dashboard" in title_lower:
            return PageCategory.dashboard

        return PageCategory.other

    def _emit_progress(
        self,
        pages_crawled: int,
        pages_queued: int,
        current_url: str,
    ) -> None:
        """Emit progress update.

        Args:
            pages_crawled: Number of pages crawled
            pages_queued: Number of pages in queue
            current_url: Current URL being processed
        """
        if self._progress_callback:
            total = pages_crawled + pages_queued
            progress = CrawlProgress(
                pages_crawled=pages_crawled,
                pages_queued=pages_queued,
                current_url=current_url,
                errors_count=len(self._errors),
                progress_percent=(pages_crawled / total * 100) if total > 0 else 0,
            )
            self._progress_callback(progress)

    async def stream_crawl(
        self,
        start_url: str,
        config: DiscoveryConfig,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream crawl progress as events.

        Args:
            start_url: URL to start crawling from
            config: Discovery configuration

        Yields:
            Dictionary events with crawl progress
        """
        # Set up progress streaming
        progress_queue: asyncio.Queue = asyncio.Queue()

        def progress_handler(progress: CrawlProgress):
            progress_queue.put_nowait(
                {
                    "type": "progress",
                    "data": {
                        "pages_crawled": progress.pages_crawled,
                        "pages_queued": progress.pages_queued,
                        "current_url": progress.current_url,
                        "errors_count": progress.errors_count,
                        "progress_percent": progress.progress_percent,
                    },
                }
            )

        self.set_progress_callback(progress_handler)

        # Start crawl in background
        crawl_task = asyncio.create_task(self.run_crawl(start_url, config))

        # Yield start event
        yield {
            "type": "start",
            "data": {
                "start_url": start_url,
                "max_pages": config.max_pages,
            },
        }

        # Stream progress events
        while not crawl_task.done():
            try:
                event = await asyncio.wait_for(
                    progress_queue.get(),
                    timeout=1.0,
                )
                yield event
            except TimeoutError:
                continue

        # Get final result
        result = await crawl_task

        # Yield completion event
        yield {
            "type": "complete",
            "data": {
                "total_pages": len(result.pages),
                "total_errors": len(result.errors),
                "duration_ms": result.duration_ms,
            },
        }

        # Clear callback
        self._progress_callback = None


# =============================================================================
# Convenience Functions
# =============================================================================


async def discover_application(
    start_url: str,
    max_pages: int = 50,
    max_depth: int = 3,
    timeout_seconds: int = 600,
    capture_screenshots: bool = False,
    auth_config: dict[str, Any] | None = None,
    on_page: Callable[[DiscoveredPage], None] | None = None,
    use_crawlee: bool = False,
    **kwargs,
) -> CrawlResult:
    """Convenience function for quick application discovery.

    This is the simplest way to run a discovery crawl from Python.

    Args:
        start_url: URL to start crawling from
        max_pages: Maximum number of pages to discover (default: 50)
        max_depth: Maximum link depth (default: 3)
        timeout_seconds: Maximum crawl duration (default: 600)
        capture_screenshots: Whether to capture screenshots (default: False)
        auth_config: Optional authentication configuration dict
        on_page: Optional callback for streaming page updates (streaming mode)
        use_crawlee: Whether to use Crawlee (Node.js) instead of Playwright
        **kwargs: Additional DiscoveryConfig options

    Returns:
        CrawlResult with discovered pages and structure

    Example:
        # Quick scan
        result = await discover_application("https://example.com")
        print(f"Found {result.total_pages} pages")

        # With options
        result = await discover_application(
            "https://example.com",
            max_pages=100,
            capture_screenshots=True,
        )

        # With streaming updates
        async for event in bridge.stream_crawl(start_url, config):
            print(f"Event: {event['type']}")
    """
    from src.discovery.models import AuthConfig as AuthConfigModel

    # Build auth config if provided
    auth_model = None
    if auth_config:
        auth_model = AuthConfigModel.from_dict(auth_config)

    # Build discovery config
    config = DiscoveryConfig(
        max_pages=max_pages,
        max_depth=max_depth,
        max_duration_seconds=timeout_seconds,
        capture_screenshots=capture_screenshots,
        auth_required=auth_model is not None,
        auth_config=auth_model,
        **kwargs,
    )

    # Create bridge and run
    bridge = CrawleeBridge(use_crawlee=use_crawlee)

    if on_page:
        # Use streaming mode with callback
        pages: dict[str, DiscoveredPage] = {}

        async for event in bridge.stream_crawl(start_url, config):
            if event["type"] == "progress":
                # Create a minimal page object for callback
                current_url = event["data"].get("current_url", "")
                if current_url and current_url not in pages:
                    # We don't have full page data in progress events
                    # The callback will get full pages from the final result
                    pass
            elif event["type"] == "complete":
                # Final result is in the crawl_task
                pass

        # Get the actual result from run_crawl
        result = await bridge.run_crawl(start_url, config)

        # Call on_page for each discovered page
        for page in result.pages.values():
            try:
                on_page(page)
            except Exception as e:
                logger.warning("on_page callback error", error=str(e))

        return result
    else:
        return await bridge.run_crawl(start_url, config)


async def check_crawlee_available() -> bool:
    """Check if Crawlee and dependencies are available.

    Returns:
        True if Crawlee can be executed, False otherwise
    """
    try:
        # Use create_subprocess_exec (safe, no shell injection)
        process = await asyncio.create_subprocess_exec(
            "npx",
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        return process.returncode == 0
    except (FileNotFoundError, PermissionError):
        return False


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "CrawleeBridge",
    "CrawlProgress",
    "discover_application",
    "check_crawlee_available",
]
