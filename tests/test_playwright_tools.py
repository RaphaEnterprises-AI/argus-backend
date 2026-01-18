"""Tests for Playwright tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_page():
    """Create a mock Playwright page."""
    page = AsyncMock()
    page.url = "https://example.com"
    page.goto = AsyncMock()
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.type = AsyncMock()
    page.screenshot = AsyncMock(return_value=b"fake_screenshot_data")
    page.query_selector = AsyncMock()
    page.wait_for_selector = AsyncMock()
    page.title = AsyncMock(return_value="Test Page")
    page.input_value = AsyncMock(return_value="test_value")
    page.is_checked = AsyncMock(return_value=False)
    page.viewport_size = {"width": 1920, "height": 1080}
    page.context = MagicMock()
    page.context.cookies = AsyncMock(return_value=[])
    return page


class TestPlaywrightTools:
    """Tests for PlaywrightTools class."""

    @pytest.mark.asyncio
    async def test_goto(self, mock_page):
        """Test navigation."""
        from src.tools.playwright_tools import PlaywrightTools

        tools = PlaywrightTools(mock_page)
        result = await tools.goto("https://example.com/page")

        mock_page.goto.assert_called_once()
        assert result == mock_page.url

    @pytest.mark.asyncio
    async def test_click(self, mock_page):
        """Test click action."""
        from src.tools.playwright_tools import PlaywrightTools

        tools = PlaywrightTools(mock_page)
        await tools.click("#button")

        mock_page.click.assert_called_once_with("#button", timeout=5000, force=False)

    @pytest.mark.asyncio
    async def test_fill(self, mock_page):
        """Test fill action."""
        from src.tools.playwright_tools import PlaywrightTools

        tools = PlaywrightTools(mock_page)
        await tools.fill("#input", "test value")

        mock_page.fill.assert_called_once_with("#input", "test value", timeout=5000)

    @pytest.mark.asyncio
    async def test_screenshot(self, mock_page):
        """Test screenshot capture."""
        from src.tools.playwright_tools import PlaywrightTools

        tools = PlaywrightTools(mock_page)
        result = await tools.screenshot()

        mock_page.screenshot.assert_called_once()
        assert result == b"fake_screenshot_data"

    @pytest.mark.asyncio
    async def test_get_page_info(self, mock_page):
        """Test getting page info."""
        from src.tools.playwright_tools import PlaywrightTools

        tools = PlaywrightTools(mock_page)
        info = await tools.get_page_info()

        assert info.url == "https://example.com"
        assert info.title == "Test Page"
        assert info.viewport == {"width": 1920, "height": 1080}

    @pytest.mark.asyncio
    async def test_assert_visible_success(self, mock_page):
        """Test assert_visible when element is visible."""
        from src.tools.playwright_tools import PlaywrightTools

        element = AsyncMock()
        element.is_visible = AsyncMock(return_value=True)
        mock_page.query_selector.return_value = element

        tools = PlaywrightTools(mock_page)
        await tools.assert_visible("#element")  # Should not raise

    @pytest.mark.asyncio
    async def test_assert_visible_failure(self, mock_page):
        """Test assert_visible when element is not visible."""
        from src.tools.playwright_tools import PlaywrightTools

        element = AsyncMock()
        element.is_visible = AsyncMock(return_value=False)
        mock_page.query_selector.return_value = element

        tools = PlaywrightTools(mock_page)

        with pytest.raises(AssertionError, match="not visible"):
            await tools.assert_visible("#element")

    @pytest.mark.asyncio
    async def test_assert_url_success(self, mock_page):
        """Test assert_url when URL matches."""
        from src.tools.playwright_tools import PlaywrightTools

        mock_page.url = "https://example.com/dashboard"

        tools = PlaywrightTools(mock_page)
        await tools.assert_url("dashboard")  # Should not raise

    @pytest.mark.asyncio
    async def test_assert_url_failure(self, mock_page):
        """Test assert_url when URL doesn't match."""
        from src.tools.playwright_tools import PlaywrightTools

        mock_page.url = "https://example.com/login"

        tools = PlaywrightTools(mock_page)

        with pytest.raises(AssertionError, match="URL mismatch"):
            await tools.assert_url("dashboard")


class TestBrowserManager:
    """Tests for BrowserManager class."""

    @pytest.mark.asyncio
    async def test_browser_config_defaults(self):
        """Test BrowserConfig has correct defaults."""
        from src.tools.playwright_tools import BrowserConfig

        config = BrowserConfig()

        assert config.headless is True
        assert config.viewport_width == 1920
        assert config.viewport_height == 1080
        assert config.timeout_ms == 30000

    @pytest.mark.asyncio
    async def test_browser_config_custom(self):
        """Test BrowserConfig with custom values."""
        from src.tools.playwright_tools import BrowserConfig

        config = BrowserConfig(
            headless=False,
            viewport_width=1280,
            viewport_height=720,
            timeout_ms=10000,
        )

        assert config.headless is False
        assert config.viewport_width == 1280
        assert config.viewport_height == 720
        assert config.timeout_ms == 10000
