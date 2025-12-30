"""Browser Automation Abstraction Layer.

This module provides a unified interface for browser automation that works with:
- Playwright (default)
- Selenium WebDriver
- Puppeteer (via Pyppeteer)
- Pure Computer Use (visual/pixel-based, no framework needed)
- Custom user frameworks via plugin interface

The abstraction allows the testing agent to work with ANY browser automation
tool while maintaining consistent behavior.

Architecture:
                      ┌─────────────────────────────┐
                      │     BrowserAutomation       │
                      │     (Abstract Interface)    │
                      └─────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  Playwright   │         │    Selenium     │         │  Computer Use   │
│   Executor    │         │    Executor     │         │   Executor      │
└───────────────┘         └─────────────────┘         └─────────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
   Playwright              WebDriver/Grid              Screenshot + Claude
   (programmatic)          (programmatic)              (visual/pixel-based)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, TypeVar

import structlog

logger = structlog.get_logger()


class AutomationFramework(Enum):
    """Supported automation frameworks."""

    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    PUPPETEER = "puppeteer"
    COMPUTER_USE = "computer_use"  # Pure visual, no framework
    EXTENSION = "extension"  # Chrome extension bridge (like Claude in Chrome)
    HYBRID = "hybrid"  # Programmatic + Computer Use fallback
    CUSTOM = "custom"


@dataclass
class BrowserConfig:
    """Configuration for browser automation."""

    framework: AutomationFramework = AutomationFramework.PLAYWRIGHT
    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit
    viewport_width: int = 1920
    viewport_height: int = 1080
    timeout_ms: int = 30000
    slow_mo_ms: int = 0  # Slow down actions for debugging
    extra_options: dict = field(default_factory=dict)


@dataclass
class ActionResult:
    """Result from a browser action."""

    success: bool
    action: str
    duration_ms: int
    error: Optional[str] = None
    screenshot: Optional[bytes] = None
    data: Any = None  # Action-specific return data


class BrowserAutomation(ABC):
    """Abstract base class for browser automation.

    Implement this interface to add support for new automation frameworks.
    The testing agent uses this interface exclusively, making it framework-agnostic.
    """

    def __init__(self, config: Optional[BrowserConfig] = None):
        self.config = config or BrowserConfig()
        self.log = logger.bind(framework=self.config.framework.value)

    @abstractmethod
    async def start(self) -> None:
        """Start the browser/automation session."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the browser/automation session."""
        pass

    @abstractmethod
    async def goto(self, url: str, wait_until: str = "load") -> ActionResult:
        """Navigate to a URL."""
        pass

    @abstractmethod
    async def click(self, selector: str, timeout_ms: Optional[int] = None) -> ActionResult:
        """Click an element."""
        pass

    @abstractmethod
    async def fill(self, selector: str, value: str) -> ActionResult:
        """Fill a form field."""
        pass

    @abstractmethod
    async def type_text(self, selector: str, text: str, delay_ms: int = 0) -> ActionResult:
        """Type text character by character."""
        pass

    @abstractmethod
    async def screenshot(self, full_page: bool = False) -> bytes:
        """Take a screenshot."""
        pass

    @abstractmethod
    async def get_text(self, selector: str) -> str:
        """Get text content of an element."""
        pass

    @abstractmethod
    async def is_visible(self, selector: str) -> bool:
        """Check if element is visible."""
        pass

    @abstractmethod
    async def wait_for_selector(self, selector: str, timeout_ms: Optional[int] = None) -> ActionResult:
        """Wait for element to appear."""
        pass

    @abstractmethod
    async def get_current_url(self) -> str:
        """Get the current page URL."""
        pass

    # Optional methods with default implementations

    async def hover(self, selector: str) -> ActionResult:
        """Hover over an element."""
        raise NotImplementedError("Hover not implemented for this framework")

    async def select_option(self, selector: str, value: str) -> ActionResult:
        """Select from dropdown."""
        raise NotImplementedError("Select not implemented for this framework")

    async def press_key(self, key: str) -> ActionResult:
        """Press a keyboard key."""
        raise NotImplementedError("Key press not implemented for this framework")

    async def scroll(self, x: int = 0, y: int = 0) -> ActionResult:
        """Scroll the page."""
        raise NotImplementedError("Scroll not implemented for this framework")

    async def evaluate(self, script: str) -> Any:
        """Execute JavaScript in the browser."""
        raise NotImplementedError("Evaluate not implemented for this framework")

    # Context manager support

    async def __aenter__(self) -> "BrowserAutomation":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


class PlaywrightAutomation(BrowserAutomation):
    """Playwright-based browser automation."""

    def __init__(self, config: Optional[BrowserConfig] = None):
        super().__init__(config or BrowserConfig(framework=AutomationFramework.PLAYWRIGHT))
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def start(self) -> None:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()

        browser_type = getattr(self._playwright, self.config.browser_type)
        self._browser = await browser_type.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo_ms,
        )

        self._context = await self._browser.new_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            }
        )
        self._page = await self._context.new_page()
        self.log.info("Playwright browser started")

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self.log.info("Playwright browser stopped")

    async def goto(self, url: str, wait_until: str = "load") -> ActionResult:
        import time

        start = time.time()
        try:
            await self._page.goto(url, wait_until=wait_until)
            return ActionResult(
                success=True,
                action="goto",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="goto",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def click(self, selector: str, timeout_ms: Optional[int] = None) -> ActionResult:
        import time

        start = time.time()
        try:
            await self._page.click(selector, timeout=timeout_ms or self.config.timeout_ms)
            return ActionResult(
                success=True,
                action="click",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="click",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def fill(self, selector: str, value: str) -> ActionResult:
        import time

        start = time.time()
        try:
            await self._page.fill(selector, value)
            return ActionResult(
                success=True,
                action="fill",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="fill",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def type_text(self, selector: str, text: str, delay_ms: int = 0) -> ActionResult:
        import time

        start = time.time()
        try:
            await self._page.type(selector, text, delay=delay_ms)
            return ActionResult(
                success=True,
                action="type",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="type",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def screenshot(self, full_page: bool = False) -> bytes:
        return await self._page.screenshot(full_page=full_page)

    async def get_text(self, selector: str) -> str:
        return await self._page.inner_text(selector)

    async def is_visible(self, selector: str) -> bool:
        return await self._page.is_visible(selector)

    async def wait_for_selector(self, selector: str, timeout_ms: Optional[int] = None) -> ActionResult:
        import time

        start = time.time()
        try:
            await self._page.wait_for_selector(selector, timeout=timeout_ms or self.config.timeout_ms)
            return ActionResult(
                success=True,
                action="wait_for_selector",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="wait_for_selector",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def get_current_url(self) -> str:
        return self._page.url

    async def hover(self, selector: str) -> ActionResult:
        import time

        start = time.time()
        try:
            await self._page.hover(selector)
            return ActionResult(success=True, action="hover", duration_ms=int((time.time() - start) * 1000))
        except Exception as e:
            return ActionResult(success=False, action="hover", duration_ms=int((time.time() - start) * 1000), error=str(e))

    async def select_option(self, selector: str, value: str) -> ActionResult:
        import time

        start = time.time()
        try:
            await self._page.select_option(selector, value)
            return ActionResult(success=True, action="select", duration_ms=int((time.time() - start) * 1000))
        except Exception as e:
            return ActionResult(success=False, action="select", duration_ms=int((time.time() - start) * 1000), error=str(e))

    async def press_key(self, key: str) -> ActionResult:
        import time

        start = time.time()
        try:
            await self._page.keyboard.press(key)
            return ActionResult(success=True, action="press_key", duration_ms=int((time.time() - start) * 1000))
        except Exception as e:
            return ActionResult(success=False, action="press_key", duration_ms=int((time.time() - start) * 1000), error=str(e))

    async def scroll(self, x: int = 0, y: int = 0) -> ActionResult:
        import time

        start = time.time()
        try:
            await self._page.evaluate(f"window.scrollBy({x}, {y})")
            return ActionResult(success=True, action="scroll", duration_ms=int((time.time() - start) * 1000))
        except Exception as e:
            return ActionResult(success=False, action="scroll", duration_ms=int((time.time() - start) * 1000), error=str(e))

    async def evaluate(self, script: str) -> Any:
        return await self._page.evaluate(script)


class SeleniumAutomation(BrowserAutomation):
    """Selenium WebDriver-based browser automation.

    Requires: pip install selenium webdriver-manager
    """

    def __init__(self, config: Optional[BrowserConfig] = None):
        super().__init__(config or BrowserConfig(framework=AutomationFramework.SELENIUM))
        self._driver = None

    async def start(self) -> None:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        try:
            from webdriver_manager.chrome import ChromeDriverManager

            options = Options()
            if self.config.headless:
                options.add_argument("--headless=new")
            options.add_argument(f"--window-size={self.config.viewport_width},{self.config.viewport_height}")

            self._driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options,
            )
            self._driver.implicitly_wait(self.config.timeout_ms / 1000)
            self.log.info("Selenium browser started")

        except ImportError:
            raise ImportError(
                "Selenium not installed. Run: pip install selenium webdriver-manager"
            )

    async def stop(self) -> None:
        if self._driver:
            self._driver.quit()
            self.log.info("Selenium browser stopped")

    async def goto(self, url: str, wait_until: str = "load") -> ActionResult:
        import time

        start = time.time()
        try:
            self._driver.get(url)
            return ActionResult(
                success=True,
                action="goto",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="goto",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def click(self, selector: str, timeout_ms: Optional[int] = None) -> ActionResult:
        import time

        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        start = time.time()
        try:
            wait = WebDriverWait(self._driver, (timeout_ms or self.config.timeout_ms) / 1000)
            element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            element.click()
            return ActionResult(
                success=True,
                action="click",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="click",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def fill(self, selector: str, value: str) -> ActionResult:
        import time

        from selenium.webdriver.common.by import By

        start = time.time()
        try:
            element = self._driver.find_element(By.CSS_SELECTOR, selector)
            element.clear()
            element.send_keys(value)
            return ActionResult(
                success=True,
                action="fill",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="fill",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def type_text(self, selector: str, text: str, delay_ms: int = 0) -> ActionResult:
        # Selenium doesn't have built-in character delay, use fill
        return await self.fill(selector, text)

    async def screenshot(self, full_page: bool = False) -> bytes:
        return self._driver.get_screenshot_as_png()

    async def get_text(self, selector: str) -> str:
        from selenium.webdriver.common.by import By

        element = self._driver.find_element(By.CSS_SELECTOR, selector)
        return element.text

    async def is_visible(self, selector: str) -> bool:
        from selenium.webdriver.common.by import By

        try:
            element = self._driver.find_element(By.CSS_SELECTOR, selector)
            return element.is_displayed()
        except Exception:
            return False

    async def wait_for_selector(self, selector: str, timeout_ms: Optional[int] = None) -> ActionResult:
        import time

        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        start = time.time()
        try:
            wait = WebDriverWait(self._driver, (timeout_ms or self.config.timeout_ms) / 1000)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            return ActionResult(
                success=True,
                action="wait_for_selector",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="wait_for_selector",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def get_current_url(self) -> str:
        return self._driver.current_url

    async def evaluate(self, script: str) -> Any:
        return self._driver.execute_script(script)


class ComputerUseAutomation(BrowserAutomation):
    """Pure Computer Use (visual/pixel-based) automation.

    This works on ANY web app without selectors - just like Claude Desktop!
    Uses Claude's vision to understand the screen and perform actions.

    Benefits:
    - No selectors needed
    - Works with any UI
    - Framework-agnostic
    - Handles dynamic content naturally

    Tradeoffs:
    - Slower (2-5s per action vs 10ms)
    - Higher API costs (screenshots)
    - Requires visual stability
    """

    def __init__(
        self,
        config: Optional[BrowserConfig] = None,
        anthropic_client=None,
        screenshot_fn: Optional[Callable[[], bytes]] = None,
    ):
        """Initialize Computer Use automation.

        Args:
            config: Browser configuration
            anthropic_client: Anthropic client for API calls
            screenshot_fn: Function to capture screenshots (from any source)
        """
        super().__init__(config or BrowserConfig(framework=AutomationFramework.COMPUTER_USE))
        self._client = anthropic_client
        self._screenshot_fn = screenshot_fn
        self._display_width = config.viewport_width if config else 1920
        self._display_height = config.viewport_height if config else 1080

    async def start(self) -> None:
        """Initialize Computer Use automation."""
        import anthropic

        if not self._client:
            self._client = anthropic.Anthropic()

        self.log.info(
            "Computer Use automation started",
            display=f"{self._display_width}x{self._display_height}",
        )

    async def stop(self) -> None:
        """Stop Computer Use automation."""
        self.log.info("Computer Use automation stopped")

    async def _execute_computer_use(self, task: str) -> dict:
        """Execute a task using Computer Use API.

        This is the core method that sends screenshots to Claude and
        receives coordinate-based actions.
        """
        import base64

        # Get current screenshot
        screenshot = await self.screenshot()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(screenshot).decode(),
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Looking at this screen, {task}. Return the action needed.",
                    },
                ],
            }
        ]

        response = self._client.beta.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            betas=["computer-use-2025-01-24"],
            tools=[
                {
                    "type": "computer_20250124",
                    "name": "computer",
                    "display_width_px": self._display_width,
                    "display_height_px": self._display_height,
                    "display_number": 1,
                }
            ],
            messages=messages,
        )

        # Extract tool use from response
        for block in response.content:
            if hasattr(block, "type") and block.type == "tool_use":
                return block.input

        return {"action": "none"}

    async def goto(self, url: str, wait_until: str = "load") -> ActionResult:
        """Navigate using Computer Use (type URL in address bar)."""
        import time

        start = time.time()
        try:
            # Use Computer Use to navigate: Ctrl+L to focus address bar, then type URL
            result = await self._execute_computer_use(
                f"navigate to {url} by pressing Ctrl+L to focus the address bar, then typing the URL and pressing Enter"
            )
            return ActionResult(
                success=True,
                action="goto",
                duration_ms=int((time.time() - start) * 1000),
                data=result,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="goto",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def click(self, selector: str, timeout_ms: Optional[int] = None) -> ActionResult:
        """Click using Computer Use (visual identification).

        Instead of using CSS selectors, we describe what to click:
        - "Click the Login button"
        - "Click the blue Submit button"
        - "Click the search icon"
        """
        import time

        start = time.time()
        try:
            # Selector becomes a visual description
            result = await self._execute_computer_use(f"click on {selector}")
            return ActionResult(
                success=True,
                action="click",
                duration_ms=int((time.time() - start) * 1000),
                data=result,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="click",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def fill(self, selector: str, value: str) -> ActionResult:
        """Fill using Computer Use (click field, then type)."""
        import time

        start = time.time()
        try:
            result = await self._execute_computer_use(
                f"click on {selector}, then type '{value}'"
            )
            return ActionResult(
                success=True,
                action="fill",
                duration_ms=int((time.time() - start) * 1000),
                data=result,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="fill",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def type_text(self, selector: str, text: str, delay_ms: int = 0) -> ActionResult:
        return await self.fill(selector, text)

    async def screenshot(self, full_page: bool = False) -> bytes:
        """Take screenshot from the configured source."""
        if self._screenshot_fn:
            return self._screenshot_fn()

        # Fallback: try to capture from display
        raise NotImplementedError(
            "No screenshot function configured. Provide screenshot_fn to __init__"
        )

    async def get_text(self, selector: str) -> str:
        """Get text using Computer Use (ask Claude to read it)."""
        import base64

        screenshot = await self.screenshot()

        response = self._client.messages.create(
            model="claude-haiku-4-5",  # Use Haiku for speed
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(screenshot).decode(),
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Read and return ONLY the text content of: {selector}",
                        },
                    ],
                }
            ],
        )

        return response.content[0].text

    async def is_visible(self, selector: str) -> bool:
        """Check visibility using Computer Use (ask Claude if visible)."""
        import base64

        screenshot = await self.screenshot()

        response = self._client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(screenshot).decode(),
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Is this visible on the screen: {selector}? Answer only 'yes' or 'no'.",
                        },
                    ],
                }
            ],
        )

        return "yes" in response.content[0].text.lower()

    async def wait_for_selector(self, selector: str, timeout_ms: Optional[int] = None) -> ActionResult:
        """Wait for element using polling with Computer Use."""
        import asyncio
        import time

        start = time.time()
        timeout = (timeout_ms or self.config.timeout_ms) / 1000
        poll_interval = 0.5

        while (time.time() - start) < timeout:
            if await self.is_visible(selector):
                return ActionResult(
                    success=True,
                    action="wait_for_selector",
                    duration_ms=int((time.time() - start) * 1000),
                )
            await asyncio.sleep(poll_interval)

        return ActionResult(
            success=False,
            action="wait_for_selector",
            duration_ms=int((time.time() - start) * 1000),
            error=f"Timeout waiting for: {selector}",
        )

    async def get_current_url(self) -> str:
        """Get URL using Computer Use (read from address bar)."""
        return await self.get_text("the URL in the address bar")


class HybridAutomation(BrowserAutomation):
    """Hybrid automation: Programmatic first, Computer Use fallback.

    This provides the best of both worlds:
    - Fast programmatic execution for most actions
    - Computer Use fallback when selectors fail
    - Visual verification using Computer Use
    """

    def __init__(
        self,
        primary: BrowserAutomation,
        fallback: Optional[ComputerUseAutomation] = None,
        config: Optional[BrowserConfig] = None,
    ):
        super().__init__(config)
        self.primary = primary
        self.fallback = fallback
        self.use_fallback_on_failure = True

    async def start(self) -> None:
        await self.primary.start()
        if self.fallback:
            await self.fallback.start()

    async def stop(self) -> None:
        await self.primary.stop()
        if self.fallback:
            await self.fallback.stop()

    async def _with_fallback(self, action_name: str, primary_fn, fallback_fn) -> ActionResult:
        """Execute with fallback on failure."""
        result = await primary_fn()

        if not result.success and self.use_fallback_on_failure and self.fallback:
            self.log.info(f"Primary {action_name} failed, trying Computer Use fallback")
            result = await fallback_fn()
            result.data = {"fallback_used": True}

        return result

    async def goto(self, url: str, wait_until: str = "load") -> ActionResult:
        return await self._with_fallback(
            "goto",
            lambda: self.primary.goto(url, wait_until),
            lambda: self.fallback.goto(url, wait_until),
        )

    async def click(self, selector: str, timeout_ms: Optional[int] = None) -> ActionResult:
        return await self._with_fallback(
            "click",
            lambda: self.primary.click(selector, timeout_ms),
            lambda: self.fallback.click(selector, timeout_ms),
        )

    async def fill(self, selector: str, value: str) -> ActionResult:
        return await self._with_fallback(
            "fill",
            lambda: self.primary.fill(selector, value),
            lambda: self.fallback.fill(selector, value),
        )

    async def type_text(self, selector: str, text: str, delay_ms: int = 0) -> ActionResult:
        return await self._with_fallback(
            "type",
            lambda: self.primary.type_text(selector, text, delay_ms),
            lambda: self.fallback.type_text(selector, text, delay_ms),
        )

    async def screenshot(self, full_page: bool = False) -> bytes:
        return await self.primary.screenshot(full_page)

    async def get_text(self, selector: str) -> str:
        try:
            return await self.primary.get_text(selector)
        except Exception:
            if self.fallback:
                return await self.fallback.get_text(selector)
            raise

    async def is_visible(self, selector: str) -> bool:
        try:
            return await self.primary.is_visible(selector)
        except Exception:
            if self.fallback:
                return await self.fallback.is_visible(selector)
            return False

    async def wait_for_selector(self, selector: str, timeout_ms: Optional[int] = None) -> ActionResult:
        return await self._with_fallback(
            "wait",
            lambda: self.primary.wait_for_selector(selector, timeout_ms),
            lambda: self.fallback.wait_for_selector(selector, timeout_ms),
        )

    async def get_current_url(self) -> str:
        return await self.primary.get_current_url()


# Factory functions


async def create_browser(
    framework: AutomationFramework | str = AutomationFramework.PLAYWRIGHT,
    config: Optional[BrowserConfig] = None,
    **kwargs,
) -> BrowserAutomation:
    """Factory function to create and start a browser automation instance.

    This is the recommended way to create browser automation. It creates
    the instance and starts it, ready for use.

    Args:
        framework: Framework to use
        config: Browser configuration
        **kwargs: Framework-specific options

    Returns:
        Started BrowserAutomation instance

    Example:
        # Use Playwright (default)
        browser = await create_browser()
        await browser.goto("https://example.com")
        await browser.stop()

        # Use Chrome Extension (for real browser session)
        browser = await create_browser("extension")

        # Use hybrid mode (programmatic + Computer Use fallback)
        browser = await create_browser("hybrid")

        # Use context manager
        async with await create_browser("playwright") as browser:
            await browser.goto("https://example.com")
    """
    automation = create_automation(framework, config, **kwargs)
    await automation.start()
    return automation


def create_automation(
    framework: AutomationFramework | str = AutomationFramework.PLAYWRIGHT,
    config: Optional[BrowserConfig] = None,
    **kwargs,
) -> BrowserAutomation:
    """Factory function to create browser automation instance (not started).

    Use create_browser() for a started instance.

    Args:
        framework: Framework to use
        config: Browser configuration
        **kwargs: Framework-specific options

    Returns:
        BrowserAutomation instance (not started - call start() first)

    Example:
        # Use Playwright
        async with create_automation("playwright") as browser:
            await browser.goto("https://example.com")

        # Use Selenium
        async with create_automation("selenium") as browser:
            await browser.goto("https://example.com")

        # Use pure Computer Use (no framework)
        async with create_automation("computer_use", anthropic_client=client) as browser:
            await browser.click("the Login button")  # Visual, no selectors!

        # Use Chrome Extension bridge (like Claude in Chrome / Antigravity)
        async with create_automation("extension", port=8765) as browser:
            await browser.goto("https://example.com")
            logs = await browser.get_console_logs()  # Extension-specific!

        # Use hybrid mode (fast + visual fallback)
        async with create_automation("hybrid") as browser:
            await browser.click("#button")  # Fast Playwright first
            # Falls back to Computer Use if selector fails
    """
    if isinstance(framework, str):
        framework = AutomationFramework(framework)

    if config is None:
        config = BrowserConfig(framework=framework)

    if framework == AutomationFramework.PLAYWRIGHT:
        return PlaywrightAutomation(config)

    elif framework == AutomationFramework.SELENIUM:
        return SeleniumAutomation(config)

    elif framework == AutomationFramework.COMPUTER_USE:
        return ComputerUseAutomation(config, **kwargs)

    elif framework == AutomationFramework.EXTENSION:
        # Import here to avoid circular imports
        from .extension_bridge import ExtensionBridge

        port = kwargs.get("port", 8765)
        host = kwargs.get("host", "localhost")
        return ExtensionBridge(config=config, port=port, host=host)

    elif framework == AutomationFramework.HYBRID:
        # Create hybrid with Playwright primary and Computer Use fallback
        primary = PlaywrightAutomation(config)
        fallback = ComputerUseAutomation(config, **kwargs) if kwargs.get("with_fallback", True) else None
        return HybridAutomation(primary, fallback, config)

    elif framework == AutomationFramework.CUSTOM:
        if "automation_class" not in kwargs:
            raise ValueError("Custom framework requires 'automation_class' argument")
        return kwargs["automation_class"](config, **kwargs)

    else:
        raise ValueError(f"Unknown framework: {framework}")
