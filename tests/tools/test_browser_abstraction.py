"""Tests for browser abstraction layer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.browser_abstraction import (
    ActionResult,
    AutomationFramework,
    BrowserAutomation,
    BrowserConfig,
    ComputerUseAutomation,
    HybridAutomation,
    PlaywrightAutomation,
    SeleniumAutomation,
    create_automation,
    create_browser,
)


class TestAutomationFramework:
    """Tests for AutomationFramework enum."""

    def test_framework_values(self):
        """Test all framework values."""
        assert AutomationFramework.PLAYWRIGHT.value == "playwright"
        assert AutomationFramework.SELENIUM.value == "selenium"
        assert AutomationFramework.PUPPETEER.value == "puppeteer"
        assert AutomationFramework.COMPUTER_USE.value == "computer_use"
        assert AutomationFramework.EXTENSION.value == "extension"
        assert AutomationFramework.HYBRID.value == "hybrid"
        assert AutomationFramework.CUSTOM.value == "custom"

    def test_all_frameworks_exist(self):
        """Test all expected frameworks are defined."""
        frameworks = list(AutomationFramework)
        assert len(frameworks) == 7


class TestBrowserConfig:
    """Tests for BrowserConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BrowserConfig()
        assert config.framework == AutomationFramework.PLAYWRIGHT
        assert config.headless is True
        assert config.browser_type == "chromium"
        assert config.viewport_width == 1920
        assert config.viewport_height == 1080
        assert config.timeout_ms == 30000
        assert config.slow_mo_ms == 0
        assert config.extra_options == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BrowserConfig(
            framework=AutomationFramework.SELENIUM,
            headless=False,
            browser_type="firefox",
            viewport_width=1280,
            viewport_height=720,
            timeout_ms=10000,
            slow_mo_ms=100,
            extra_options={"proxy": "http://proxy:8080"},
        )
        assert config.framework == AutomationFramework.SELENIUM
        assert config.headless is False
        assert config.browser_type == "firefox"
        assert config.viewport_width == 1280
        assert config.viewport_height == 720
        assert config.timeout_ms == 10000
        assert config.slow_mo_ms == 100
        assert config.extra_options == {"proxy": "http://proxy:8080"}


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_success_result(self):
        """Test successful action result."""
        result = ActionResult(
            success=True,
            action="click",
            duration_ms=150,
        )
        assert result.success is True
        assert result.action == "click"
        assert result.duration_ms == 150
        assert result.error is None
        assert result.screenshot is None
        assert result.data is None

    def test_failure_result(self):
        """Test failed action result."""
        result = ActionResult(
            success=False,
            action="goto",
            duration_ms=5000,
            error="Navigation timeout",
        )
        assert result.success is False
        assert result.action == "goto"
        assert result.error == "Navigation timeout"

    def test_result_with_screenshot(self):
        """Test result with screenshot."""
        screenshot = b"\x89PNG\r\n\x1a\n"
        result = ActionResult(
            success=True,
            action="screenshot",
            duration_ms=100,
            screenshot=screenshot,
        )
        assert result.screenshot == screenshot

    def test_result_with_data(self):
        """Test result with custom data."""
        result = ActionResult(
            success=True,
            action="evaluate",
            duration_ms=50,
            data={"title": "Test Page"},
        )
        assert result.data == {"title": "Test Page"}


class TestBrowserAutomationBase:
    """Tests for BrowserAutomation base class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        # Create a concrete subclass for testing
        class ConcreteBrowser(BrowserAutomation):
            async def start(self): pass
            async def stop(self): pass
            async def goto(self, url, wait_until="load"): return ActionResult(True, "goto", 0)
            async def click(self, selector, timeout_ms=None): return ActionResult(True, "click", 0)
            async def fill(self, selector, value): return ActionResult(True, "fill", 0)
            async def type_text(self, selector, text, delay_ms=0): return ActionResult(True, "type", 0)
            async def screenshot(self, full_page=False): return b""
            async def get_text(self, selector): return ""
            async def is_visible(self, selector): return True
            async def wait_for_selector(self, selector, timeout_ms=None): return ActionResult(True, "wait", 0)
            async def get_current_url(self): return ""

        browser = ConcreteBrowser()
        assert browser.config is not None
        assert browser.config.framework == AutomationFramework.PLAYWRIGHT

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        class ConcreteBrowser(BrowserAutomation):
            async def start(self): pass
            async def stop(self): pass
            async def goto(self, url, wait_until="load"): return ActionResult(True, "goto", 0)
            async def click(self, selector, timeout_ms=None): return ActionResult(True, "click", 0)
            async def fill(self, selector, value): return ActionResult(True, "fill", 0)
            async def type_text(self, selector, text, delay_ms=0): return ActionResult(True, "type", 0)
            async def screenshot(self, full_page=False): return b""
            async def get_text(self, selector): return ""
            async def is_visible(self, selector): return True
            async def wait_for_selector(self, selector, timeout_ms=None): return ActionResult(True, "wait", 0)
            async def get_current_url(self): return ""

        config = BrowserConfig(framework=AutomationFramework.SELENIUM, headless=False)
        browser = ConcreteBrowser(config)
        assert browser.config.framework == AutomationFramework.SELENIUM
        assert browser.config.headless is False

    @pytest.mark.asyncio
    async def test_optional_methods_raise_not_implemented(self):
        """Test that optional methods raise NotImplementedError."""
        class ConcreteBrowser(BrowserAutomation):
            async def start(self): pass
            async def stop(self): pass
            async def goto(self, url, wait_until="load"): return ActionResult(True, "goto", 0)
            async def click(self, selector, timeout_ms=None): return ActionResult(True, "click", 0)
            async def fill(self, selector, value): return ActionResult(True, "fill", 0)
            async def type_text(self, selector, text, delay_ms=0): return ActionResult(True, "type", 0)
            async def screenshot(self, full_page=False): return b""
            async def get_text(self, selector): return ""
            async def is_visible(self, selector): return True
            async def wait_for_selector(self, selector, timeout_ms=None): return ActionResult(True, "wait", 0)
            async def get_current_url(self): return ""

        browser = ConcreteBrowser()

        with pytest.raises(NotImplementedError):
            await browser.hover("#button")

        with pytest.raises(NotImplementedError):
            await browser.select_option("#select", "value")

        with pytest.raises(NotImplementedError):
            await browser.press_key("Enter")

        with pytest.raises(NotImplementedError):
            await browser.scroll(0, 100)

        with pytest.raises(NotImplementedError):
            await browser.evaluate("document.title")

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager support."""
        started = False
        stopped = False

        class ConcreteBrowser(BrowserAutomation):
            async def start(self):
                nonlocal started
                started = True

            async def stop(self):
                nonlocal stopped
                stopped = True

            async def goto(self, url, wait_until="load"): return ActionResult(True, "goto", 0)
            async def click(self, selector, timeout_ms=None): return ActionResult(True, "click", 0)
            async def fill(self, selector, value): return ActionResult(True, "fill", 0)
            async def type_text(self, selector, text, delay_ms=0): return ActionResult(True, "type", 0)
            async def screenshot(self, full_page=False): return b""
            async def get_text(self, selector): return ""
            async def is_visible(self, selector): return True
            async def wait_for_selector(self, selector, timeout_ms=None): return ActionResult(True, "wait", 0)
            async def get_current_url(self): return ""

        async with ConcreteBrowser():
            assert started is True
            assert stopped is False

        assert stopped is True


class TestPlaywrightAutomation:
    """Tests for PlaywrightAutomation class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        automation = PlaywrightAutomation()
        assert automation.config.framework == AutomationFramework.PLAYWRIGHT
        assert automation._playwright is None
        assert automation._browser is None
        assert automation._context is None
        assert automation._page is None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = BrowserConfig(
            framework=AutomationFramework.PLAYWRIGHT,
            headless=False,
            browser_type="firefox",
        )
        automation = PlaywrightAutomation(config)
        assert automation.config.headless is False
        assert automation.config.browser_type == "firefox"

    @pytest.mark.asyncio
    async def test_start(self):
        """Test starting Playwright browser."""
        automation = PlaywrightAutomation()

        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_pw_fn = MagicMock()
        mock_pw_fn.return_value.start = AsyncMock(return_value=mock_playwright)

        with patch("playwright.async_api.async_playwright", mock_pw_fn):
            await automation.start()

        assert automation._playwright == mock_playwright
        assert automation._browser == mock_browser
        assert automation._context == mock_context
        assert automation._page == mock_page

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stopping Playwright browser."""
        automation = PlaywrightAutomation()
        automation._browser = MagicMock()
        automation._browser.close = AsyncMock()
        automation._playwright = MagicMock()
        automation._playwright.stop = AsyncMock()

        await automation.stop()

        automation._browser.close.assert_called_once()
        automation._playwright.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_no_browser(self):
        """Test stopping when no browser is running."""
        automation = PlaywrightAutomation()
        await automation.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_goto_success(self):
        """Test successful navigation."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.goto = AsyncMock()

        result = await automation.goto("https://example.com")

        assert result.success is True
        assert result.action == "goto"
        automation._page.goto.assert_called_once_with(
            "https://example.com", wait_until="load"
        )

    @pytest.mark.asyncio
    async def test_goto_failure(self):
        """Test navigation failure."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.goto = AsyncMock(side_effect=Exception("Navigation failed"))

        result = await automation.goto("https://example.com")

        assert result.success is False
        assert result.action == "goto"
        assert "Navigation failed" in result.error

    @pytest.mark.asyncio
    async def test_click_success(self):
        """Test successful click."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.click = AsyncMock()

        result = await automation.click("#button")

        assert result.success is True
        assert result.action == "click"
        automation._page.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_with_timeout(self):
        """Test click with custom timeout."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.click = AsyncMock()

        await automation.click("#button", timeout_ms=5000)

        automation._page.click.assert_called_once_with("#button", timeout=5000)

    @pytest.mark.asyncio
    async def test_click_failure(self):
        """Test click failure."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.click = AsyncMock(side_effect=Exception("Element not found"))

        result = await automation.click("#button")

        assert result.success is False
        assert "Element not found" in result.error

    @pytest.mark.asyncio
    async def test_fill_success(self):
        """Test successful fill."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.fill = AsyncMock()

        result = await automation.fill("#input", "test value")

        assert result.success is True
        assert result.action == "fill"
        automation._page.fill.assert_called_once_with("#input", "test value")

    @pytest.mark.asyncio
    async def test_fill_failure(self):
        """Test fill failure."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.fill = AsyncMock(side_effect=Exception("Fill failed"))

        result = await automation.fill("#input", "value")

        assert result.success is False
        assert "Fill failed" in result.error

    @pytest.mark.asyncio
    async def test_type_text_success(self):
        """Test successful type_text."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.type = AsyncMock()

        result = await automation.type_text("#input", "hello", delay_ms=50)

        assert result.success is True
        assert result.action == "type"
        automation._page.type.assert_called_once_with("#input", "hello", delay=50)

    @pytest.mark.asyncio
    async def test_type_text_failure(self):
        """Test type_text failure."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.type = AsyncMock(side_effect=Exception("Type failed"))

        result = await automation.type_text("#input", "hello")

        assert result.success is False
        assert "Type failed" in result.error

    @pytest.mark.asyncio
    async def test_screenshot(self):
        """Test screenshot capture."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.screenshot = AsyncMock(return_value=b"screenshot_data")

        result = await automation.screenshot(full_page=True)

        assert result == b"screenshot_data"
        automation._page.screenshot.assert_called_once_with(full_page=True)

    @pytest.mark.asyncio
    async def test_get_text(self):
        """Test getting text content."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.inner_text = AsyncMock(return_value="Element text")

        result = await automation.get_text("#element")

        assert result == "Element text"
        automation._page.inner_text.assert_called_once_with("#element")

    @pytest.mark.asyncio
    async def test_is_visible(self):
        """Test visibility check."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.is_visible = AsyncMock(return_value=True)

        result = await automation.is_visible("#element")

        assert result is True
        automation._page.is_visible.assert_called_once_with("#element")

    @pytest.mark.asyncio
    async def test_wait_for_selector_success(self):
        """Test successful wait for selector."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.wait_for_selector = AsyncMock()

        result = await automation.wait_for_selector("#element")

        assert result.success is True
        assert result.action == "wait_for_selector"

    @pytest.mark.asyncio
    async def test_wait_for_selector_failure(self):
        """Test wait for selector timeout."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.wait_for_selector = AsyncMock(
            side_effect=Exception("Timeout")
        )

        result = await automation.wait_for_selector("#element", timeout_ms=1000)

        assert result.success is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_get_current_url(self):
        """Test getting current URL."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.url = "https://example.com/page"

        result = await automation.get_current_url()

        assert result == "https://example.com/page"

    @pytest.mark.asyncio
    async def test_hover_success(self):
        """Test successful hover."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.hover = AsyncMock()

        result = await automation.hover("#element")

        assert result.success is True
        assert result.action == "hover"

    @pytest.mark.asyncio
    async def test_hover_failure(self):
        """Test hover failure."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.hover = AsyncMock(side_effect=Exception("Hover failed"))

        result = await automation.hover("#element")

        assert result.success is False
        assert "Hover failed" in result.error

    @pytest.mark.asyncio
    async def test_select_option_success(self):
        """Test successful select option."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.select_option = AsyncMock()

        result = await automation.select_option("#select", "value1")

        assert result.success is True
        assert result.action == "select"

    @pytest.mark.asyncio
    async def test_select_option_failure(self):
        """Test select option failure."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.select_option = AsyncMock(
            side_effect=Exception("Select failed")
        )

        result = await automation.select_option("#select", "value1")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_press_key_success(self):
        """Test successful key press."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.keyboard = MagicMock()
        automation._page.keyboard.press = AsyncMock()

        result = await automation.press_key("Enter")

        assert result.success is True
        assert result.action == "press_key"
        automation._page.keyboard.press.assert_called_once_with("Enter")

    @pytest.mark.asyncio
    async def test_press_key_failure(self):
        """Test key press failure."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.keyboard = MagicMock()
        automation._page.keyboard.press = AsyncMock(
            side_effect=Exception("Key press failed")
        )

        result = await automation.press_key("Enter")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_scroll_success(self):
        """Test successful scroll."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.evaluate = AsyncMock()

        result = await automation.scroll(0, 500)

        assert result.success is True
        assert result.action == "scroll"
        automation._page.evaluate.assert_called_once_with("window.scrollBy(0, 500)")

    @pytest.mark.asyncio
    async def test_scroll_failure(self):
        """Test scroll failure."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.evaluate = AsyncMock(side_effect=Exception("Scroll failed"))

        result = await automation.scroll(0, 500)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_evaluate(self):
        """Test JavaScript evaluation."""
        automation = PlaywrightAutomation()
        automation._page = MagicMock()
        automation._page.evaluate = AsyncMock(return_value="Test Title")

        result = await automation.evaluate("document.title")

        assert result == "Test Title"


class TestSeleniumAutomation:
    """Tests for SeleniumAutomation class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        automation = SeleniumAutomation()
        assert automation.config.framework == AutomationFramework.SELENIUM
        assert automation._driver is None

    @pytest.mark.asyncio
    async def test_stop_with_driver(self):
        """Test stopping with active driver."""
        automation = SeleniumAutomation()
        automation._driver = MagicMock()
        automation._driver.quit = MagicMock()

        await automation.stop()

        automation._driver.quit.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_no_driver(self):
        """Test stopping without driver."""
        automation = SeleniumAutomation()
        await automation.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_goto_success(self):
        """Test successful navigation."""
        automation = SeleniumAutomation()
        automation._driver = MagicMock()
        automation._driver.get = MagicMock()

        result = await automation.goto("https://example.com")

        assert result.success is True
        assert result.action == "goto"
        automation._driver.get.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_goto_failure(self):
        """Test navigation failure."""
        automation = SeleniumAutomation()
        automation._driver = MagicMock()
        automation._driver.get = MagicMock(side_effect=Exception("Navigation failed"))

        result = await automation.goto("https://example.com")

        assert result.success is False
        assert "Navigation failed" in result.error

    @pytest.mark.asyncio
    async def test_type_text_uses_fill(self):
        """Test that type_text delegates to fill."""
        automation = SeleniumAutomation()
        automation._driver = MagicMock()
        mock_element = MagicMock()
        automation._driver.find_element = MagicMock(return_value=mock_element)

        result = await automation.type_text("#input", "hello")

        assert result.success is True
        mock_element.clear.assert_called_once()
        mock_element.send_keys.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_screenshot(self):
        """Test screenshot capture."""
        automation = SeleniumAutomation()
        automation._driver = MagicMock()
        automation._driver.get_screenshot_as_png = MagicMock(
            return_value=b"screenshot_data"
        )

        result = await automation.screenshot()

        assert result == b"screenshot_data"

    @pytest.mark.asyncio
    async def test_get_current_url(self):
        """Test getting current URL."""
        automation = SeleniumAutomation()
        automation._driver = MagicMock()
        automation._driver.current_url = "https://example.com/page"

        result = await automation.get_current_url()

        assert result == "https://example.com/page"

    @pytest.mark.asyncio
    async def test_evaluate(self):
        """Test JavaScript execution."""
        automation = SeleniumAutomation()
        automation._driver = MagicMock()
        automation._driver.execute_script = MagicMock(return_value="result")

        result = await automation.evaluate("return document.title")

        assert result == "result"
        automation._driver.execute_script.assert_called_once_with(
            "return document.title"
        )

    @pytest.mark.asyncio
    async def test_is_visible_true(self):
        """Test element is visible."""
        automation = SeleniumAutomation()
        automation._driver = MagicMock()
        mock_element = MagicMock()
        mock_element.is_displayed = MagicMock(return_value=True)
        automation._driver.find_element = MagicMock(return_value=mock_element)

        result = await automation.is_visible("#element")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_visible_false_exception(self):
        """Test element visibility when element not found."""
        automation = SeleniumAutomation()
        automation._driver = MagicMock()
        automation._driver.find_element = MagicMock(
            side_effect=Exception("Element not found")
        )

        result = await automation.is_visible("#element")

        assert result is False


class TestComputerUseAutomation:
    """Tests for ComputerUseAutomation class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        automation = ComputerUseAutomation()
        assert automation.config.framework == AutomationFramework.COMPUTER_USE
        assert automation._client is None
        assert automation._screenshot_fn is None
        assert automation._display_width == 1920
        assert automation._display_height == 1080

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = BrowserConfig(
            framework=AutomationFramework.COMPUTER_USE,
            viewport_width=1280,
            viewport_height=720,
        )
        screenshot_fn = MagicMock(return_value=b"screenshot")
        mock_client = MagicMock()

        automation = ComputerUseAutomation(
            config=config,
            anthropic_client=mock_client,
            screenshot_fn=screenshot_fn,
        )

        assert automation._display_width == 1280
        assert automation._display_height == 720
        assert automation._client == mock_client
        assert automation._screenshot_fn == screenshot_fn

    @pytest.mark.asyncio
    async def test_start_creates_client(self):
        """Test that start creates Anthropic client if not provided."""
        automation = ComputerUseAutomation()

        mock_client = MagicMock()
        with patch("anthropic.Anthropic", return_value=mock_client):
            await automation.start()

        assert automation._client == mock_client

    @pytest.mark.asyncio
    async def test_start_uses_existing_client(self):
        """Test that start uses existing client."""
        mock_client = MagicMock()
        automation = ComputerUseAutomation(anthropic_client=mock_client)

        await automation.start()

        assert automation._client == mock_client

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stopping Computer Use automation."""
        automation = ComputerUseAutomation()
        await automation.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_screenshot_with_fn(self):
        """Test screenshot with provided function."""
        screenshot_fn = MagicMock(return_value=b"screenshot_data")
        automation = ComputerUseAutomation(screenshot_fn=screenshot_fn)

        result = await automation.screenshot()

        assert result == b"screenshot_data"
        screenshot_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_screenshot_without_fn_raises(self):
        """Test screenshot without function raises error."""
        automation = ComputerUseAutomation()

        with pytest.raises(NotImplementedError):
            await automation.screenshot()

    @pytest.mark.asyncio
    async def test_type_text_delegates_to_fill(self):
        """Test that type_text delegates to fill."""
        automation = ComputerUseAutomation()
        automation.fill = AsyncMock(
            return_value=ActionResult(success=True, action="fill", duration_ms=100)
        )

        result = await automation.type_text("#input", "hello")

        assert result.success is True
        automation.fill.assert_called_once_with("#input", "hello")

    @pytest.mark.asyncio
    async def test_get_current_url(self):
        """Test getting URL via Computer Use."""
        automation = ComputerUseAutomation()
        automation.get_text = AsyncMock(return_value="https://example.com")

        result = await automation.get_current_url()

        assert result == "https://example.com"
        automation.get_text.assert_called_once_with("the URL in the address bar")


class TestHybridAutomation:
    """Tests for HybridAutomation class."""

    def test_init(self):
        """Test initialization."""
        primary = MagicMock(spec=PlaywrightAutomation)
        fallback = MagicMock(spec=ComputerUseAutomation)
        config = BrowserConfig(framework=AutomationFramework.HYBRID)

        automation = HybridAutomation(primary, fallback, config)

        assert automation.primary == primary
        assert automation.fallback == fallback
        assert automation.use_fallback_on_failure is True

    @pytest.mark.asyncio
    async def test_start(self):
        """Test starting both primary and fallback."""
        primary = MagicMock()
        primary.start = AsyncMock()
        fallback = MagicMock()
        fallback.start = AsyncMock()

        automation = HybridAutomation(primary, fallback)
        await automation.start()

        primary.start.assert_called_once()
        fallback.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_no_fallback(self):
        """Test starting without fallback."""
        primary = MagicMock()
        primary.start = AsyncMock()

        automation = HybridAutomation(primary, None)
        await automation.start()

        primary.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stopping both primary and fallback."""
        primary = MagicMock()
        primary.stop = AsyncMock()
        fallback = MagicMock()
        fallback.stop = AsyncMock()

        automation = HybridAutomation(primary, fallback)
        await automation.stop()

        primary.stop.assert_called_once()
        fallback.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_goto_primary_success(self):
        """Test goto uses primary when successful."""
        primary = MagicMock()
        primary.goto = AsyncMock(
            return_value=ActionResult(success=True, action="goto", duration_ms=100)
        )
        fallback = MagicMock()
        fallback.goto = AsyncMock()

        automation = HybridAutomation(primary, fallback)
        result = await automation.goto("https://example.com")

        assert result.success is True
        primary.goto.assert_called_once()
        fallback.goto.assert_not_called()

    @pytest.mark.asyncio
    async def test_goto_fallback_on_failure(self):
        """Test goto falls back when primary fails."""
        primary = MagicMock()
        primary.goto = AsyncMock(
            return_value=ActionResult(
                success=False, action="goto", duration_ms=100, error="Failed"
            )
        )
        fallback = MagicMock()
        fallback.goto = AsyncMock(
            return_value=ActionResult(success=True, action="goto", duration_ms=2000)
        )

        automation = HybridAutomation(primary, fallback)
        result = await automation.goto("https://example.com")

        assert result.success is True
        assert result.data == {"fallback_used": True}
        primary.goto.assert_called_once()
        fallback.goto.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_primary_success(self):
        """Test click uses primary when successful."""
        primary = MagicMock()
        primary.click = AsyncMock(
            return_value=ActionResult(success=True, action="click", duration_ms=50)
        )
        fallback = MagicMock()

        automation = HybridAutomation(primary, fallback)
        result = await automation.click("#button")

        assert result.success is True
        primary.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_fill_fallback(self):
        """Test fill falls back on failure."""
        primary = MagicMock()
        primary.fill = AsyncMock(
            return_value=ActionResult(
                success=False, action="fill", duration_ms=50, error="Failed"
            )
        )
        fallback = MagicMock()
        fallback.fill = AsyncMock(
            return_value=ActionResult(success=True, action="fill", duration_ms=1000)
        )

        automation = HybridAutomation(primary, fallback)
        result = await automation.fill("#input", "value")

        assert result.success is True
        primary.fill.assert_called_once()
        fallback.fill.assert_called_once()

    @pytest.mark.asyncio
    async def test_screenshot(self):
        """Test screenshot uses primary."""
        primary = MagicMock()
        primary.screenshot = AsyncMock(return_value=b"screenshot")
        fallback = MagicMock()

        automation = HybridAutomation(primary, fallback)
        result = await automation.screenshot(full_page=True)

        assert result == b"screenshot"
        primary.screenshot.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_get_text_primary_success(self):
        """Test get_text uses primary when successful."""
        primary = MagicMock()
        primary.get_text = AsyncMock(return_value="Text content")
        fallback = MagicMock()

        automation = HybridAutomation(primary, fallback)
        result = await automation.get_text("#element")

        assert result == "Text content"
        primary.get_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_text_fallback(self):
        """Test get_text falls back on exception."""
        primary = MagicMock()
        primary.get_text = AsyncMock(side_effect=Exception("Failed"))
        fallback = MagicMock()
        fallback.get_text = AsyncMock(return_value="Fallback text")

        automation = HybridAutomation(primary, fallback)
        result = await automation.get_text("#element")

        assert result == "Fallback text"
        fallback.get_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_text_no_fallback_raises(self):
        """Test get_text raises when no fallback and primary fails."""
        primary = MagicMock()
        primary.get_text = AsyncMock(side_effect=Exception("Failed"))

        automation = HybridAutomation(primary, None)

        with pytest.raises(Exception, match="Failed"):
            await automation.get_text("#element")

    @pytest.mark.asyncio
    async def test_is_visible_primary_success(self):
        """Test is_visible uses primary when successful."""
        primary = MagicMock()
        primary.is_visible = AsyncMock(return_value=True)
        fallback = MagicMock()

        automation = HybridAutomation(primary, fallback)
        result = await automation.is_visible("#element")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_visible_fallback(self):
        """Test is_visible falls back on exception."""
        primary = MagicMock()
        primary.is_visible = AsyncMock(side_effect=Exception("Failed"))
        fallback = MagicMock()
        fallback.is_visible = AsyncMock(return_value=True)

        automation = HybridAutomation(primary, fallback)
        result = await automation.is_visible("#element")

        assert result is True
        fallback.is_visible.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_visible_no_fallback_returns_false(self):
        """Test is_visible returns False when no fallback and primary fails."""
        primary = MagicMock()
        primary.is_visible = AsyncMock(side_effect=Exception("Failed"))

        automation = HybridAutomation(primary, None)
        result = await automation.is_visible("#element")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_current_url(self):
        """Test get_current_url uses primary."""
        primary = MagicMock()
        primary.get_current_url = AsyncMock(return_value="https://example.com")
        fallback = MagicMock()

        automation = HybridAutomation(primary, fallback)
        result = await automation.get_current_url()

        assert result == "https://example.com"

    @pytest.mark.asyncio
    async def test_type_text_fallback(self):
        """Test type_text fallback."""
        primary = MagicMock()
        primary.type_text = AsyncMock(
            return_value=ActionResult(
                success=False, action="type", duration_ms=50, error="Failed"
            )
        )
        fallback = MagicMock()
        fallback.type_text = AsyncMock(
            return_value=ActionResult(success=True, action="type", duration_ms=500)
        )

        automation = HybridAutomation(primary, fallback)
        result = await automation.type_text("#input", "hello", delay_ms=50)

        assert result.success is True
        fallback.type_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_selector(self):
        """Test wait_for_selector with fallback."""
        primary = MagicMock()
        primary.wait_for_selector = AsyncMock(
            return_value=ActionResult(
                success=False, action="wait", duration_ms=5000, error="Timeout"
            )
        )
        fallback = MagicMock()
        fallback.wait_for_selector = AsyncMock(
            return_value=ActionResult(success=True, action="wait", duration_ms=2000)
        )

        automation = HybridAutomation(primary, fallback)
        result = await automation.wait_for_selector("#element", timeout_ms=10000)

        assert result.success is True


class TestCreateAutomation:
    """Tests for create_automation factory function."""

    def test_create_playwright(self):
        """Test creating Playwright automation."""
        automation = create_automation(AutomationFramework.PLAYWRIGHT)
        assert isinstance(automation, PlaywrightAutomation)

    def test_create_playwright_from_string(self):
        """Test creating Playwright from string."""
        automation = create_automation("playwright")
        assert isinstance(automation, PlaywrightAutomation)

    def test_create_selenium(self):
        """Test creating Selenium automation."""
        automation = create_automation(AutomationFramework.SELENIUM)
        assert isinstance(automation, SeleniumAutomation)

    def test_create_computer_use(self):
        """Test creating Computer Use automation."""
        mock_client = MagicMock()
        automation = create_automation(
            AutomationFramework.COMPUTER_USE, anthropic_client=mock_client
        )
        assert isinstance(automation, ComputerUseAutomation)
        assert automation._client == mock_client

    def test_create_extension(self):
        """Test creating Extension bridge automation."""
        from src.tools.extension_bridge import ExtensionBridge

        automation = create_automation(AutomationFramework.EXTENSION, port=9000)
        assert isinstance(automation, ExtensionBridge)

    def test_create_hybrid(self):
        """Test creating Hybrid automation."""
        automation = create_automation(AutomationFramework.HYBRID)
        assert isinstance(automation, HybridAutomation)
        assert isinstance(automation.primary, PlaywrightAutomation)

    def test_create_hybrid_without_fallback(self):
        """Test creating Hybrid automation without fallback."""
        automation = create_automation(
            AutomationFramework.HYBRID, with_fallback=False
        )
        assert isinstance(automation, HybridAutomation)
        assert automation.fallback is None

    def test_create_custom_requires_class(self):
        """Test that custom framework requires automation_class."""
        with pytest.raises(ValueError, match="automation_class"):
            create_automation(AutomationFramework.CUSTOM)

    def test_create_custom_with_class(self):
        """Test creating custom automation with class."""
        class CustomAutomation(BrowserAutomation):
            def __init__(self, config, **kwargs):
                super().__init__(config)
                self.custom_arg = kwargs.get("custom_arg")

            async def start(self): pass
            async def stop(self): pass
            async def goto(self, url, wait_until="load"): return ActionResult(True, "goto", 0)
            async def click(self, selector, timeout_ms=None): return ActionResult(True, "click", 0)
            async def fill(self, selector, value): return ActionResult(True, "fill", 0)
            async def type_text(self, selector, text, delay_ms=0): return ActionResult(True, "type", 0)
            async def screenshot(self, full_page=False): return b""
            async def get_text(self, selector): return ""
            async def is_visible(self, selector): return True
            async def wait_for_selector(self, selector, timeout_ms=None): return ActionResult(True, "wait", 0)
            async def get_current_url(self): return ""

        automation = create_automation(
            AutomationFramework.CUSTOM,
            automation_class=CustomAutomation,
            custom_arg="test_value",
        )

        assert isinstance(automation, CustomAutomation)
        assert automation.custom_arg == "test_value"

    def test_create_with_config(self):
        """Test creating automation with custom config."""
        config = BrowserConfig(headless=False, viewport_width=1280)
        automation = create_automation("playwright", config=config)

        assert automation.config.headless is False
        assert automation.config.viewport_width == 1280


class TestCreateBrowser:
    """Tests for create_browser factory function."""

    @pytest.mark.asyncio
    async def test_create_browser_starts_automation(self):
        """Test that create_browser starts the automation."""
        with patch.object(
            PlaywrightAutomation, "start", new_callable=AsyncMock
        ) as mock_start:
            browser = await create_browser(AutomationFramework.PLAYWRIGHT)

            mock_start.assert_called_once()
            assert isinstance(browser, PlaywrightAutomation)

    @pytest.mark.asyncio
    async def test_create_browser_with_config(self):
        """Test create_browser with custom config."""
        config = BrowserConfig(headless=False)

        with patch.object(
            PlaywrightAutomation, "start", new_callable=AsyncMock
        ):
            browser = await create_browser("playwright", config=config)

            assert browser.config.headless is False

    @pytest.mark.asyncio
    async def test_create_browser_from_string(self):
        """Test create_browser from string framework name."""
        with patch.object(
            SeleniumAutomation, "start", new_callable=AsyncMock
        ):
            browser = await create_browser("selenium")

            assert isinstance(browser, SeleniumAutomation)
