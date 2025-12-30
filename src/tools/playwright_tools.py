"""Playwright browser automation tools for E2E testing."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class BrowserConfig:
    """Configuration for browser instances."""
    headless: bool = True
    slow_mo: int = 0  # Milliseconds between actions
    viewport_width: int = 1920
    viewport_height: int = 1080
    device_scale_factor: float = 1.0
    timeout_ms: int = 30000
    ignore_https_errors: bool = True
    locale: str = "en-US"
    timezone_id: str = "America/Los_Angeles"
    user_agent: Optional[str] = None
    extra_http_headers: dict = field(default_factory=dict)


@dataclass
class PageInfo:
    """Information about a page."""
    url: str
    title: str
    viewport: dict
    cookies: list[dict] = field(default_factory=list)


class BrowserManager:
    """
    Manages Playwright browser instances.

    Handles browser lifecycle, context creation, and page management.
    """

    def __init__(self, config: Optional[BrowserConfig] = None):
        self.config = config or BrowserConfig()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self.log = logger.bind(component="browser")

    async def start(self) -> None:
        """Start the browser."""
        from playwright.async_api import async_playwright

        self.log.info("Starting browser", headless=self.config.headless)

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo,
        )

        await self._create_context()
        self.log.info("Browser started")

    async def _create_context(self) -> None:
        """Create a new browser context."""
        self._context = await self._browser.new_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
            device_scale_factor=self.config.device_scale_factor,
            ignore_https_errors=self.config.ignore_https_errors,
            locale=self.config.locale,
            timezone_id=self.config.timezone_id,
            user_agent=self.config.user_agent,
            extra_http_headers=self.config.extra_http_headers or {},
        )

        # Set default timeout
        self._context.set_default_timeout(self.config.timeout_ms)

        # Create initial page
        self._page = await self._context.new_page()

    async def new_page(self):
        """Create a new page in the context."""
        if not self._context:
            raise RuntimeError("Browser not started")
        return await self._context.new_page()

    @property
    def page(self):
        """Get the current page."""
        return self._page

    async def reset_context(self) -> None:
        """Reset the browser context (clear cookies, storage, etc.)."""
        if self._context:
            await self._context.close()
        await self._create_context()
        self.log.debug("Browser context reset")

    async def stop(self) -> None:
        """Stop the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self.log.info("Browser stopped")

    async def __aenter__(self) -> "BrowserManager":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


@asynccontextmanager
async def create_browser_context(
    config: Optional[BrowserConfig] = None,
):
    """
    Context manager for browser sessions.

    Usage:
        async with create_browser_context() as browser:
            page = browser.page
            await page.goto("https://example.com")
    """
    manager = BrowserManager(config)
    try:
        await manager.start()
        yield manager
    finally:
        await manager.stop()


class PlaywrightTools:
    """
    High-level Playwright tools for E2E testing.

    Provides a clean interface for common testing operations.
    """

    def __init__(self, page):
        """
        Initialize with a Playwright page.

        Args:
            page: Playwright page object
        """
        self.page = page
        self.log = logger.bind(component="playwright_tools")

    # ==========================================================================
    # Navigation
    # ==========================================================================

    async def goto(
        self,
        url: str,
        wait_until: str = "load",
        timeout_ms: int = 30000,
    ) -> str:
        """
        Navigate to URL.

        Args:
            url: URL to navigate to
            wait_until: "load", "domcontentloaded", "networkidle"
            timeout_ms: Timeout in milliseconds

        Returns:
            Final URL after navigation
        """
        self.log.info("Navigating", url=url)
        await self.page.goto(url, wait_until=wait_until, timeout=timeout_ms)
        return self.page.url

    async def reload(self, wait_until: str = "load") -> None:
        """Reload the current page."""
        await self.page.reload(wait_until=wait_until)

    async def go_back(self) -> Optional[str]:
        """Go back in history."""
        response = await self.page.go_back()
        return self.page.url if response else None

    async def go_forward(self) -> Optional[str]:
        """Go forward in history."""
        response = await self.page.go_forward()
        return self.page.url if response else None

    # ==========================================================================
    # Element Interaction
    # ==========================================================================

    async def click(
        self,
        selector: str,
        timeout_ms: int = 5000,
        force: bool = False,
    ) -> None:
        """
        Click an element.

        Args:
            selector: CSS selector or text selector
            timeout_ms: Timeout waiting for element
            force: Force click even if element is not visible
        """
        self.log.debug("Clicking", selector=selector)
        await self.page.click(selector, timeout=timeout_ms, force=force)

    async def double_click(self, selector: str, timeout_ms: int = 5000) -> None:
        """Double-click an element."""
        await self.page.dblclick(selector, timeout=timeout_ms)

    async def right_click(self, selector: str, timeout_ms: int = 5000) -> None:
        """Right-click an element."""
        await self.page.click(selector, button="right", timeout=timeout_ms)

    async def hover(self, selector: str, timeout_ms: int = 5000) -> None:
        """Hover over an element."""
        await self.page.hover(selector, timeout=timeout_ms)

    async def fill(
        self,
        selector: str,
        value: str,
        timeout_ms: int = 5000,
    ) -> None:
        """
        Fill a text input.

        Args:
            selector: Input selector
            value: Value to fill
            timeout_ms: Timeout waiting for element
        """
        self.log.debug("Filling", selector=selector, value_length=len(value))
        await self.page.fill(selector, value, timeout=timeout_ms)

    async def type_text(
        self,
        selector: str,
        text: str,
        delay_ms: int = 50,
    ) -> None:
        """
        Type text character by character.

        Args:
            selector: Input selector
            text: Text to type
            delay_ms: Delay between keystrokes
        """
        await self.page.type(selector, text, delay=delay_ms)

    async def clear(self, selector: str) -> None:
        """Clear an input field."""
        await self.page.fill(selector, "")

    async def press_key(self, key: str, selector: Optional[str] = None) -> None:
        """
        Press a keyboard key.

        Args:
            key: Key to press (e.g., "Enter", "Tab", "Escape")
            selector: Optional element to focus first
        """
        if selector:
            await self.page.press(selector, key)
        else:
            await self.page.keyboard.press(key)

    async def select_option(
        self,
        selector: str,
        value: Optional[str] = None,
        label: Optional[str] = None,
        index: Optional[int] = None,
    ) -> list[str]:
        """
        Select option from dropdown.

        Args:
            selector: Select element selector
            value: Option value
            label: Option label
            index: Option index

        Returns:
            Selected option values
        """
        if value:
            return await self.page.select_option(selector, value=value)
        elif label:
            return await self.page.select_option(selector, label=label)
        elif index is not None:
            return await self.page.select_option(selector, index=index)
        else:
            raise ValueError("Must provide value, label, or index")

    async def check(self, selector: str) -> None:
        """Check a checkbox."""
        await self.page.check(selector)

    async def uncheck(self, selector: str) -> None:
        """Uncheck a checkbox."""
        await self.page.uncheck(selector)

    # ==========================================================================
    # Waiting
    # ==========================================================================

    async def wait_for_selector(
        self,
        selector: str,
        state: str = "visible",
        timeout_ms: int = 30000,
    ) -> Any:
        """
        Wait for element to be in specified state.

        Args:
            selector: Element selector
            state: "attached", "detached", "visible", "hidden"
            timeout_ms: Timeout

        Returns:
            Element handle
        """
        return await self.page.wait_for_selector(
            selector,
            state=state,
            timeout=timeout_ms,
        )

    async def wait_for_navigation(
        self,
        url: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> None:
        """
        Wait for navigation to complete.

        Args:
            url: Optional URL pattern to wait for
            timeout_ms: Timeout
        """
        if url:
            await self.page.wait_for_url(url, timeout=timeout_ms)
        else:
            await self.page.wait_for_load_state("networkidle", timeout=timeout_ms)

    async def wait_for_load_state(
        self,
        state: str = "load",
        timeout_ms: int = 30000,
    ) -> None:
        """
        Wait for page load state.

        Args:
            state: "load", "domcontentloaded", "networkidle"
            timeout_ms: Timeout
        """
        await self.page.wait_for_load_state(state, timeout=timeout_ms)

    async def wait(self, ms: int) -> None:
        """Wait for specified milliseconds."""
        await asyncio.sleep(ms / 1000)

    # ==========================================================================
    # Element Queries
    # ==========================================================================

    async def query_selector(self, selector: str) -> Any:
        """Find first element matching selector."""
        return await self.page.query_selector(selector)

    async def query_selector_all(self, selector: str) -> list[Any]:
        """Find all elements matching selector."""
        return await self.page.query_selector_all(selector)

    async def get_text(self, selector: str) -> Optional[str]:
        """Get text content of element."""
        element = await self.page.query_selector(selector)
        if element:
            return await element.text_content()
        return None

    async def get_value(self, selector: str) -> str:
        """Get value of input element."""
        return await self.page.input_value(selector)

    async def get_attribute(self, selector: str, name: str) -> Optional[str]:
        """Get attribute value of element."""
        return await self.page.get_attribute(selector, name)

    async def is_visible(self, selector: str) -> bool:
        """Check if element is visible."""
        element = await self.page.query_selector(selector)
        if element:
            return await element.is_visible()
        return False

    async def is_enabled(self, selector: str) -> bool:
        """Check if element is enabled."""
        element = await self.page.query_selector(selector)
        if element:
            return await element.is_enabled()
        return False

    async def is_checked(self, selector: str) -> bool:
        """Check if checkbox/radio is checked."""
        return await self.page.is_checked(selector)

    async def count_elements(self, selector: str) -> int:
        """Count elements matching selector."""
        elements = await self.page.query_selector_all(selector)
        return len(elements)

    # ==========================================================================
    # Screenshots
    # ==========================================================================

    async def screenshot(
        self,
        full_page: bool = False,
        path: Optional[str] = None,
    ) -> bytes:
        """
        Take a screenshot.

        Args:
            full_page: Capture full scrollable page
            path: Optional path to save screenshot

        Returns:
            Screenshot bytes
        """
        kwargs = {"type": "png", "full_page": full_page}
        if path:
            kwargs["path"] = path

        return await self.page.screenshot(**kwargs)

    async def screenshot_element(
        self,
        selector: str,
        path: Optional[str] = None,
    ) -> bytes:
        """
        Screenshot a specific element.

        Args:
            selector: Element selector
            path: Optional path to save

        Returns:
            Screenshot bytes
        """
        element = await self.page.query_selector(selector)
        if not element:
            raise ValueError(f"Element not found: {selector}")

        kwargs = {"type": "png"}
        if path:
            kwargs["path"] = path

        return await element.screenshot(**kwargs)

    # ==========================================================================
    # Page Info
    # ==========================================================================

    async def get_page_info(self) -> PageInfo:
        """Get current page information."""
        return PageInfo(
            url=self.page.url,
            title=await self.page.title(),
            viewport=self.page.viewport_size,
            cookies=await self.page.context.cookies(),
        )

    async def get_url(self) -> str:
        """Get current URL."""
        return self.page.url

    async def get_title(self) -> str:
        """Get page title."""
        return await self.page.title()

    # ==========================================================================
    # JavaScript Execution
    # ==========================================================================

    async def evaluate(self, expression: str) -> Any:
        """
        Execute JavaScript in the page.

        Args:
            expression: JavaScript expression

        Returns:
            Result of expression
        """
        return await self.page.evaluate(expression)

    async def evaluate_handle(self, expression: str) -> Any:
        """
        Execute JavaScript and return handle.

        Args:
            expression: JavaScript expression

        Returns:
            JSHandle to result
        """
        return await self.page.evaluate_handle(expression)

    # ==========================================================================
    # Frames
    # ==========================================================================

    async def frame(self, name: Optional[str] = None, url: Optional[str] = None):
        """
        Get frame by name or URL.

        Args:
            name: Frame name
            url: Frame URL pattern

        Returns:
            Frame object
        """
        if name:
            return self.page.frame(name=name)
        elif url:
            return self.page.frame(url=url)
        return None

    # ==========================================================================
    # Downloads & Uploads
    # ==========================================================================

    async def upload_file(self, selector: str, file_path: str) -> None:
        """
        Upload a file.

        Args:
            selector: File input selector
            file_path: Path to file to upload
        """
        await self.page.set_input_files(selector, file_path)

    @asynccontextmanager
    async def expect_download(self):
        """
        Context manager for handling downloads.

        Usage:
            async with tools.expect_download() as download_info:
                await tools.click("button.download")
            download = await download_info.value
            await download.save_as("/path/to/file")
        """
        async with self.page.expect_download() as download_info:
            yield download_info

    # ==========================================================================
    # Dialogs
    # ==========================================================================

    async def accept_dialog(self, prompt_text: Optional[str] = None) -> None:
        """Set up handler to accept next dialog."""
        async def handler(dialog):
            await dialog.accept(prompt_text)
        self.page.once("dialog", handler)

    async def dismiss_dialog(self) -> None:
        """Set up handler to dismiss next dialog."""
        async def handler(dialog):
            await dialog.dismiss()
        self.page.once("dialog", handler)

    # ==========================================================================
    # Network
    # ==========================================================================

    async def set_extra_headers(self, headers: dict[str, str]) -> None:
        """Set extra HTTP headers for all requests."""
        await self.page.set_extra_http_headers(headers)

    async def route_intercept(
        self,
        url_pattern: str,
        handler,
    ) -> None:
        """
        Intercept network requests.

        Args:
            url_pattern: URL pattern to match
            handler: Async handler function
        """
        await self.page.route(url_pattern, handler)

    # ==========================================================================
    # Assertions
    # ==========================================================================

    async def assert_url(self, expected: str) -> None:
        """Assert current URL contains expected string."""
        current = self.page.url
        if expected not in current:
            raise AssertionError(f"URL mismatch: expected '{expected}' in '{current}'")

    async def assert_title(self, expected: str) -> None:
        """Assert page title contains expected string."""
        title = await self.page.title()
        if expected not in title:
            raise AssertionError(f"Title mismatch: expected '{expected}' in '{title}'")

    async def assert_visible(self, selector: str) -> None:
        """Assert element is visible."""
        is_visible = await self.is_visible(selector)
        if not is_visible:
            raise AssertionError(f"Element not visible: {selector}")

    async def assert_hidden(self, selector: str) -> None:
        """Assert element is hidden."""
        is_visible = await self.is_visible(selector)
        if is_visible:
            raise AssertionError(f"Element should be hidden: {selector}")

    async def assert_text(self, selector: str, expected: str) -> None:
        """Assert element contains expected text."""
        text = await self.get_text(selector)
        if expected not in (text or ""):
            raise AssertionError(
                f"Text mismatch: expected '{expected}' in '{text}'"
            )

    async def assert_value(self, selector: str, expected: str) -> None:
        """Assert input has expected value."""
        value = await self.get_value(selector)
        if value != expected:
            raise AssertionError(
                f"Value mismatch: expected '{expected}', got '{value}'"
            )

    async def assert_checked(self, selector: str) -> None:
        """Assert checkbox is checked."""
        is_checked = await self.is_checked(selector)
        if not is_checked:
            raise AssertionError(f"Checkbox not checked: {selector}")

    async def assert_not_checked(self, selector: str) -> None:
        """Assert checkbox is not checked."""
        is_checked = await self.is_checked(selector)
        if is_checked:
            raise AssertionError(f"Checkbox should not be checked: {selector}")

    async def assert_element_count(self, selector: str, expected: int) -> None:
        """Assert number of elements matching selector."""
        count = await self.count_elements(selector)
        if count != expected:
            raise AssertionError(
                f"Element count mismatch: expected {expected}, got {count}"
            )
