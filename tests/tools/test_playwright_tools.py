"""Tests for playwright tools module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from src.tools.playwright_tools import (
    BrowserConfig,
    PageInfo,
    BrowserManager,
    PlaywrightTools,
    create_browser_context,
)


class TestBrowserConfig:
    """Tests for BrowserConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BrowserConfig()
        assert config.headless is True
        assert config.slow_mo == 0
        assert config.viewport_width == 1920
        assert config.viewport_height == 1080
        assert config.device_scale_factor == 1.0
        assert config.timeout_ms == 30000
        assert config.ignore_https_errors is True
        assert config.locale == "en-US"
        assert config.timezone_id == "America/Los_Angeles"
        assert config.user_agent is None
        assert config.extra_http_headers == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BrowserConfig(
            headless=False,
            slow_mo=100,
            viewport_width=1280,
            viewport_height=720,
            device_scale_factor=2.0,
            timeout_ms=60000,
            ignore_https_errors=False,
            locale="en-GB",
            timezone_id="Europe/London",
            user_agent="Custom Agent",
            extra_http_headers={"X-Custom": "Header"},
        )
        assert config.headless is False
        assert config.slow_mo == 100
        assert config.viewport_width == 1280
        assert config.viewport_height == 720
        assert config.device_scale_factor == 2.0
        assert config.timeout_ms == 60000
        assert config.ignore_https_errors is False
        assert config.locale == "en-GB"
        assert config.timezone_id == "Europe/London"
        assert config.user_agent == "Custom Agent"
        assert config.extra_http_headers["X-Custom"] == "Header"


class TestPageInfo:
    """Tests for PageInfo dataclass."""

    def test_create_page_info(self):
        """Test creating page info."""
        info = PageInfo(
            url="https://example.com",
            title="Example",
            viewport={"width": 1920, "height": 1080},
            cookies=[{"name": "session", "value": "abc"}],
        )
        assert info.url == "https://example.com"
        assert info.title == "Example"
        assert info.viewport["width"] == 1920
        assert len(info.cookies) == 1

    def test_default_cookies(self):
        """Test default empty cookies list."""
        info = PageInfo(
            url="https://example.com",
            title="Test",
            viewport={},
        )
        assert info.cookies == []


class TestBrowserManager:
    """Tests for BrowserManager class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        manager = BrowserManager()
        assert manager.config is not None
        assert manager.config.headless is True
        assert manager._playwright is None
        assert manager._browser is None
        assert manager._context is None
        assert manager._page is None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = BrowserConfig(headless=False, timeout_ms=60000)
        manager = BrowserManager(config=config)
        assert manager.config.headless is False
        assert manager.config.timeout_ms == 60000

    @pytest.mark.asyncio
    async def test_start_creates_browser(self):
        """Test that start creates browser and context."""
        manager = BrowserManager()

        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.set_default_timeout = MagicMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        # Create a mock for async_playwright that returns a context manager
        mock_pw_fn = MagicMock()
        mock_pw_fn.return_value.start = AsyncMock(return_value=mock_playwright)

        with patch("playwright.async_api.async_playwright", mock_pw_fn):
            await manager.start()

            assert manager._playwright == mock_playwright
            assert manager._browser == mock_browser

    @pytest.mark.asyncio
    async def test_stop_closes_browser(self):
        """Test that stop closes browser and playwright."""
        manager = BrowserManager()
        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()
        mock_playwright = MagicMock()
        mock_playwright.stop = AsyncMock()

        manager._browser = mock_browser
        manager._playwright = mock_playwright

        await manager.stop()

        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()
        assert manager._browser is None
        assert manager._playwright is None

    @pytest.mark.asyncio
    async def test_stop_no_browser(self):
        """Test stop when no browser is running."""
        manager = BrowserManager()
        await manager.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_new_page_without_context_raises(self):
        """Test new_page raises error without context."""
        manager = BrowserManager()
        with pytest.raises(RuntimeError, match="Browser not started"):
            await manager.new_page()

    @pytest.mark.asyncio
    async def test_new_page_with_context(self):
        """Test new_page creates page from context."""
        manager = BrowserManager()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        manager._context = mock_context

        page = await manager.new_page()

        assert page == mock_page
        mock_context.new_page.assert_called_once()

    def test_page_property(self):
        """Test page property returns current page."""
        manager = BrowserManager()
        mock_page = MagicMock()
        manager._page = mock_page

        assert manager.page == mock_page

    @pytest.mark.asyncio
    async def test_reset_context(self):
        """Test reset_context closes and recreates context."""
        manager = BrowserManager()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_browser = MagicMock()
        mock_new_context = MagicMock()
        mock_page = MagicMock()
        mock_new_context.set_default_timeout = MagicMock()
        mock_new_context.new_page = AsyncMock(return_value=mock_page)
        mock_browser.new_context = AsyncMock(return_value=mock_new_context)

        manager._context = mock_context
        manager._browser = mock_browser

        await manager.reset_context()

        mock_context.close.assert_called_once()
        mock_browser.new_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test BrowserManager as context manager."""
        manager = BrowserManager()
        manager.start = AsyncMock()
        manager.stop = AsyncMock()

        async with manager as m:
            assert m == manager
            manager.start.assert_called_once()

        manager.stop.assert_called_once()


class TestPlaywrightTools:
    """Tests for PlaywrightTools class."""

    @pytest.fixture
    def mock_page(self):
        """Create a mock Playwright page."""
        page = MagicMock()
        page.goto = AsyncMock()
        page.url = "https://example.com"
        page.reload = AsyncMock()
        page.go_back = AsyncMock(return_value=True)
        page.go_forward = AsyncMock(return_value=True)
        page.click = AsyncMock()
        page.dblclick = AsyncMock()
        page.hover = AsyncMock()
        page.fill = AsyncMock()
        page.type = AsyncMock()
        page.press = AsyncMock()
        page.select_option = AsyncMock(return_value=["option1"])
        page.check = AsyncMock()
        page.uncheck = AsyncMock()
        page.wait_for_selector = AsyncMock(return_value=MagicMock())
        page.wait_for_url = AsyncMock()
        page.wait_for_load_state = AsyncMock()
        page.query_selector = AsyncMock()
        page.query_selector_all = AsyncMock(return_value=[])
        page.keyboard = MagicMock()
        page.keyboard.press = AsyncMock()
        return page

    @pytest.fixture
    def tools(self, mock_page):
        """Create PlaywrightTools instance."""
        return PlaywrightTools(mock_page)

    def test_init(self, mock_page):
        """Test PlaywrightTools initialization."""
        tools = PlaywrightTools(mock_page)
        assert tools.page == mock_page

    @pytest.mark.asyncio
    async def test_goto(self, tools, mock_page):
        """Test goto navigation."""
        result = await tools.goto("https://example.com")

        mock_page.goto.assert_called_once_with(
            "https://example.com",
            wait_until="load",
            timeout=30000,
        )
        assert result == "https://example.com"

    @pytest.mark.asyncio
    async def test_goto_with_options(self, tools, mock_page):
        """Test goto with custom options."""
        await tools.goto(
            "https://example.com",
            wait_until="networkidle",
            timeout_ms=60000,
        )

        mock_page.goto.assert_called_once_with(
            "https://example.com",
            wait_until="networkidle",
            timeout=60000,
        )

    @pytest.mark.asyncio
    async def test_reload(self, tools, mock_page):
        """Test page reload."""
        await tools.reload()
        mock_page.reload.assert_called_once_with(wait_until="load")

    @pytest.mark.asyncio
    async def test_go_back(self, tools, mock_page):
        """Test go back in history."""
        result = await tools.go_back()
        mock_page.go_back.assert_called_once()
        assert result == "https://example.com"

    @pytest.mark.asyncio
    async def test_go_back_no_history(self, tools, mock_page):
        """Test go back with no history."""
        mock_page.go_back = AsyncMock(return_value=None)
        result = await tools.go_back()
        assert result is None

    @pytest.mark.asyncio
    async def test_go_forward(self, tools, mock_page):
        """Test go forward in history."""
        result = await tools.go_forward()
        mock_page.go_forward.assert_called_once()
        assert result == "https://example.com"

    @pytest.mark.asyncio
    async def test_go_forward_no_history(self, tools, mock_page):
        """Test go forward with no history."""
        mock_page.go_forward = AsyncMock(return_value=None)
        result = await tools.go_forward()
        assert result is None

    @pytest.mark.asyncio
    async def test_click(self, tools, mock_page):
        """Test click element."""
        await tools.click("#button")
        mock_page.click.assert_called_once_with("#button", timeout=5000, force=False)

    @pytest.mark.asyncio
    async def test_click_with_options(self, tools, mock_page):
        """Test click with options."""
        await tools.click("#button", timeout_ms=10000, force=True)
        mock_page.click.assert_called_once_with("#button", timeout=10000, force=True)

    @pytest.mark.asyncio
    async def test_double_click(self, tools, mock_page):
        """Test double click."""
        await tools.double_click("#element")
        mock_page.dblclick.assert_called_once_with("#element", timeout=5000)

    @pytest.mark.asyncio
    async def test_right_click(self, tools, mock_page):
        """Test right click."""
        await tools.right_click("#element")
        mock_page.click.assert_called_once_with("#element", button="right", timeout=5000)

    @pytest.mark.asyncio
    async def test_hover(self, tools, mock_page):
        """Test hover over element."""
        await tools.hover("#element")
        mock_page.hover.assert_called_once_with("#element", timeout=5000)

    @pytest.mark.asyncio
    async def test_fill(self, tools, mock_page):
        """Test fill input."""
        await tools.fill("#input", "test value")
        mock_page.fill.assert_called_once_with("#input", "test value", timeout=5000)

    @pytest.mark.asyncio
    async def test_type_text(self, tools, mock_page):
        """Test type text character by character."""
        await tools.type_text("#input", "hello", delay_ms=100)
        mock_page.type.assert_called_once_with("#input", "hello", delay=100)

    @pytest.mark.asyncio
    async def test_clear(self, tools, mock_page):
        """Test clear input."""
        await tools.clear("#input")
        mock_page.fill.assert_called_once_with("#input", "")

    @pytest.mark.asyncio
    async def test_press_key_with_selector(self, tools, mock_page):
        """Test press key with selector."""
        await tools.press_key("Enter", selector="#input")
        mock_page.press.assert_called_once_with("#input", "Enter")

    @pytest.mark.asyncio
    async def test_press_key_without_selector(self, tools, mock_page):
        """Test press key without selector."""
        await tools.press_key("Escape")
        mock_page.keyboard.press.assert_called_once_with("Escape")

    @pytest.mark.asyncio
    async def test_select_option_by_value(self, tools, mock_page):
        """Test select option by value."""
        result = await tools.select_option("#select", value="option1")
        mock_page.select_option.assert_called_once_with("#select", value="option1")
        assert result == ["option1"]

    @pytest.mark.asyncio
    async def test_select_option_by_label(self, tools, mock_page):
        """Test select option by label."""
        await tools.select_option("#select", label="Option 1")
        mock_page.select_option.assert_called_once_with("#select", label="Option 1")

    @pytest.mark.asyncio
    async def test_select_option_by_index(self, tools, mock_page):
        """Test select option by index."""
        await tools.select_option("#select", index=0)
        mock_page.select_option.assert_called_once_with("#select", index=0)

    @pytest.mark.asyncio
    async def test_select_option_no_selector(self, tools, mock_page):
        """Test select option without selection criteria raises error."""
        with pytest.raises(ValueError, match="Must provide value, label, or index"):
            await tools.select_option("#select")

    @pytest.mark.asyncio
    async def test_check(self, tools, mock_page):
        """Test check checkbox."""
        await tools.check("#checkbox")
        mock_page.check.assert_called_once_with("#checkbox")

    @pytest.mark.asyncio
    async def test_uncheck(self, tools, mock_page):
        """Test uncheck checkbox."""
        await tools.uncheck("#checkbox")
        mock_page.uncheck.assert_called_once_with("#checkbox")

    @pytest.mark.asyncio
    async def test_wait_for_selector(self, tools, mock_page):
        """Test wait for selector."""
        result = await tools.wait_for_selector("#element")
        mock_page.wait_for_selector.assert_called_once_with(
            "#element",
            state="visible",
            timeout=30000,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_wait_for_selector_with_state(self, tools, mock_page):
        """Test wait for selector with state."""
        await tools.wait_for_selector("#element", state="hidden", timeout_ms=5000)
        mock_page.wait_for_selector.assert_called_once_with(
            "#element",
            state="hidden",
            timeout=5000,
        )

    @pytest.mark.asyncio
    async def test_wait_for_navigation_with_url(self, tools, mock_page):
        """Test wait for navigation with URL."""
        await tools.wait_for_navigation(url="**/dashboard")
        mock_page.wait_for_url.assert_called_once_with("**/dashboard", timeout=30000)

    @pytest.mark.asyncio
    async def test_wait_for_navigation_without_url(self, tools, mock_page):
        """Test wait for navigation without URL."""
        await tools.wait_for_navigation()
        mock_page.wait_for_load_state.assert_called_once_with(
            "networkidle",
            timeout=30000,
        )

    @pytest.mark.asyncio
    async def test_wait_for_load_state(self, tools, mock_page):
        """Test wait for load state."""
        await tools.wait_for_load_state(state="domcontentloaded", timeout_ms=10000)
        mock_page.wait_for_load_state.assert_called_once_with(
            "domcontentloaded",
            timeout=10000,
        )

    @pytest.mark.asyncio
    async def test_wait(self, tools):
        """Test wait for milliseconds."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await tools.wait(1000)
            mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_query_selector(self, tools, mock_page):
        """Test query selector."""
        mock_element = MagicMock()
        mock_page.query_selector = AsyncMock(return_value=mock_element)

        result = await tools.query_selector("#element")

        mock_page.query_selector.assert_called_once_with("#element")
        assert result == mock_element

    @pytest.mark.asyncio
    async def test_query_selector_all(self, tools, mock_page):
        """Test query selector all."""
        mock_elements = [MagicMock(), MagicMock()]
        mock_page.query_selector_all = AsyncMock(return_value=mock_elements)

        result = await tools.query_selector_all(".items")

        mock_page.query_selector_all.assert_called_once_with(".items")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_text(self, tools, mock_page):
        """Test get text content."""
        mock_element = MagicMock()
        mock_element.text_content = AsyncMock(return_value="Hello World")
        mock_page.query_selector = AsyncMock(return_value=mock_element)

        result = await tools.get_text("#element")

        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_get_text_not_found(self, tools, mock_page):
        """Test get text when element not found."""
        mock_page.query_selector = AsyncMock(return_value=None)

        result = await tools.get_text("#nonexistent")

        assert result is None


class TestCreateBrowserContext:
    """Tests for create_browser_context context manager."""

    @pytest.mark.asyncio
    async def test_creates_and_stops_manager(self):
        """Test that context manager starts and stops browser."""
        mock_manager = MagicMock()
        mock_manager.start = AsyncMock()
        mock_manager.stop = AsyncMock()

        with patch("src.tools.playwright_tools.BrowserManager", return_value=mock_manager):
            async with create_browser_context() as browser:
                assert browser == mock_manager
                mock_manager.start.assert_called_once()

            mock_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stops_on_exception(self):
        """Test that context manager stops browser even on exception."""
        mock_manager = MagicMock()
        mock_manager.start = AsyncMock()
        mock_manager.stop = AsyncMock()

        with patch("src.tools.playwright_tools.BrowserManager", return_value=mock_manager):
            with pytest.raises(ValueError):
                async with create_browser_context():
                    raise ValueError("Test error")

            mock_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_config(self):
        """Test that context manager passes config to manager."""
        config = BrowserConfig(headless=False)

        with patch("src.tools.playwright_tools.BrowserManager") as MockManager:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock()
            mock_instance.stop = AsyncMock()
            MockManager.return_value = mock_instance

            async with create_browser_context(config=config):
                pass

            MockManager.assert_called_once_with(config)
