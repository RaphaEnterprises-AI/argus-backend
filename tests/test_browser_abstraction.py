"""Tests for browser automation abstraction layer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAutomationFramework:
    """Tests for AutomationFramework enum."""

    def test_framework_values(self):
        """Test all framework values exist."""
        from src.tools.browser_abstraction import AutomationFramework

        assert AutomationFramework.PLAYWRIGHT.value == "playwright"
        assert AutomationFramework.SELENIUM.value == "selenium"
        assert AutomationFramework.COMPUTER_USE.value == "computer_use"
        assert AutomationFramework.EXTENSION.value == "extension"
        assert AutomationFramework.HYBRID.value == "hybrid"
        assert AutomationFramework.CUSTOM.value == "custom"


class TestBrowserConfig:
    """Tests for BrowserConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.tools.browser_abstraction import AutomationFramework, BrowserConfig

        config = BrowserConfig()

        assert config.framework == AutomationFramework.PLAYWRIGHT
        assert config.headless is True
        assert config.browser_type == "chromium"
        assert config.viewport_width == 1920
        assert config.viewport_height == 1080
        assert config.timeout_ms == 30000
        assert config.slow_mo_ms == 0

    def test_custom_config(self):
        """Test custom configuration values."""
        from src.tools.browser_abstraction import AutomationFramework, BrowserConfig

        config = BrowserConfig(
            framework=AutomationFramework.SELENIUM,
            headless=False,
            browser_type="firefox",
            viewport_width=1280,
            viewport_height=720,
            timeout_ms=60000,
        )

        assert config.framework == AutomationFramework.SELENIUM
        assert config.headless is False
        assert config.browser_type == "firefox"
        assert config.viewport_width == 1280


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_success_result(self):
        """Test successful action result."""
        from src.tools.browser_abstraction import ActionResult

        result = ActionResult(
            success=True,
            action="click",
            duration_ms=150,
        )

        assert result.success is True
        assert result.action == "click"
        assert result.duration_ms == 150
        assert result.error is None

    def test_failed_result(self):
        """Test failed action result."""
        from src.tools.browser_abstraction import ActionResult

        result = ActionResult(
            success=False,
            action="fill",
            duration_ms=5000,
            error="Element not found: #username",
        )

        assert result.success is False
        assert result.error == "Element not found: #username"


class TestCreateAutomation:
    """Tests for create_automation factory function."""

    def test_create_playwright(self):
        """Test creating Playwright automation."""
        from src.tools.browser_abstraction import (
            AutomationFramework,
            PlaywrightAutomation,
            create_automation,
        )

        automation = create_automation("playwright")

        assert isinstance(automation, PlaywrightAutomation)
        assert automation.config.framework == AutomationFramework.PLAYWRIGHT

    def test_create_selenium(self):
        """Test creating Selenium automation."""
        from src.tools.browser_abstraction import (
            SeleniumAutomation,
            create_automation,
        )

        automation = create_automation("selenium")

        assert isinstance(automation, SeleniumAutomation)

    def test_create_computer_use(self):
        """Test creating Computer Use automation."""
        from src.tools.browser_abstraction import (
            ComputerUseAutomation,
            create_automation,
        )

        automation = create_automation("computer_use")

        assert isinstance(automation, ComputerUseAutomation)

    def test_create_extension(self):
        """Test creating Extension bridge automation."""
        from src.tools.browser_abstraction import create_automation
        from src.tools.extension_bridge import ExtensionBridge

        automation = create_automation("extension", port=9999)

        assert isinstance(automation, ExtensionBridge)
        assert automation.port == 9999

    def test_create_hybrid(self):
        """Test creating Hybrid automation."""
        from src.tools.browser_abstraction import (
            HybridAutomation,
            PlaywrightAutomation,
            create_automation,
        )

        automation = create_automation("hybrid", with_fallback=False)

        assert isinstance(automation, HybridAutomation)
        assert isinstance(automation.primary, PlaywrightAutomation)

    def test_create_from_enum(self):
        """Test creating automation from enum value."""
        from src.tools.browser_abstraction import (
            AutomationFramework,
            PlaywrightAutomation,
            create_automation,
        )

        automation = create_automation(AutomationFramework.PLAYWRIGHT)

        assert isinstance(automation, PlaywrightAutomation)

    def test_create_custom_requires_class(self):
        """Test custom framework requires automation_class."""
        from src.tools.browser_abstraction import create_automation

        with pytest.raises(ValueError, match="automation_class"):
            create_automation("custom")

    def test_invalid_framework(self):
        """Test invalid framework raises error."""
        from src.tools.browser_abstraction import create_automation

        with pytest.raises(ValueError):
            # Enum constructor raises ValueError for invalid value
            create_automation("invalid_framework")


class TestPlaywrightAutomation:
    """Tests for PlaywrightAutomation class."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping Playwright."""
        with patch("playwright.async_api.async_playwright") as mock_pw:
            mock_playwright = MagicMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()

            # async_playwright() returns object with async start() method
            mock_pw_instance = MagicMock()
            mock_pw_instance.start = AsyncMock(return_value=mock_playwright)
            mock_pw.return_value = mock_pw_instance

            mock_playwright.chromium = MagicMock()
            mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)
            # playwright.stop() is also async
            mock_playwright.stop = AsyncMock()

            from src.tools.browser_abstraction import PlaywrightAutomation

            automation = PlaywrightAutomation()

            await automation.start()

            assert automation._page is not None

            await automation.stop()

            mock_browser.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_goto(self):
        """Test navigation."""
        from src.tools.browser_abstraction import PlaywrightAutomation

        automation = PlaywrightAutomation()
        automation._page = AsyncMock()
        automation._page.goto = AsyncMock()

        result = await automation.goto("https://example.com")

        assert result.success is True
        assert result.action == "goto"
        automation._page.goto.assert_called_once_with(
            "https://example.com", wait_until="load"
        )

    @pytest.mark.asyncio
    async def test_click(self):
        """Test clicking element."""
        from src.tools.browser_abstraction import PlaywrightAutomation

        automation = PlaywrightAutomation()
        automation._page = AsyncMock()

        result = await automation.click("#button")

        assert result.success is True
        assert result.action == "click"

    @pytest.mark.asyncio
    async def test_fill(self):
        """Test filling input."""
        from src.tools.browser_abstraction import PlaywrightAutomation

        automation = PlaywrightAutomation()
        automation._page = AsyncMock()

        result = await automation.fill("#input", "test value")

        assert result.success is True
        assert result.action == "fill"
        automation._page.fill.assert_called_once_with("#input", "test value")

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        with patch("playwright.async_api.async_playwright") as mock_pw:
            mock_playwright = MagicMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()

            # async_playwright() returns object with async start() method
            mock_pw_instance = MagicMock()
            mock_pw_instance.start = AsyncMock(return_value=mock_playwright)
            mock_pw.return_value = mock_pw_instance

            mock_playwright.chromium = MagicMock()
            mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)
            # playwright.stop() is also async
            mock_playwright.stop = AsyncMock()

            from src.tools.browser_abstraction import PlaywrightAutomation

            async with PlaywrightAutomation() as automation:
                assert automation._page is not None

            mock_browser.close.assert_called()


class TestHybridAutomation:
    """Tests for HybridAutomation class."""

    @pytest.mark.asyncio
    async def test_uses_primary_first(self):
        """Test that primary automation is tried first."""
        from src.tools.browser_abstraction import ActionResult, HybridAutomation

        mock_primary = AsyncMock()
        mock_primary.click = AsyncMock(
            return_value=ActionResult(success=True, action="click", duration_ms=50)
        )

        mock_fallback = AsyncMock()

        hybrid = HybridAutomation(mock_primary, mock_fallback)

        result = await hybrid.click("#button")

        assert result.success is True
        mock_primary.click.assert_called_once()
        mock_fallback.click.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_on_failure(self):
        """Test fallback is used when primary fails."""
        from src.tools.browser_abstraction import ActionResult, HybridAutomation

        mock_primary = AsyncMock()
        mock_primary.click = AsyncMock(
            return_value=ActionResult(
                success=False, action="click", duration_ms=50, error="Not found"
            )
        )

        mock_fallback = AsyncMock()
        mock_fallback.click = AsyncMock(
            return_value=ActionResult(success=True, action="click", duration_ms=500)
        )

        hybrid = HybridAutomation(mock_primary, mock_fallback)

        result = await hybrid.click("#button")

        assert result.success is True
        assert result.data == {"fallback_used": True}
        mock_primary.click.assert_called_once()
        mock_fallback.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self):
        """Test fallback is not used when disabled."""
        from src.tools.browser_abstraction import ActionResult, HybridAutomation

        mock_primary = AsyncMock()
        mock_primary.click = AsyncMock(
            return_value=ActionResult(
                success=False, action="click", duration_ms=50, error="Not found"
            )
        )

        mock_fallback = AsyncMock()

        hybrid = HybridAutomation(mock_primary, mock_fallback)
        hybrid.use_fallback_on_failure = False

        result = await hybrid.click("#button")

        assert result.success is False
        mock_fallback.click.assert_not_called()


class TestExtensionBridge:
    """Tests for ExtensionBridge class."""

    def test_init(self):
        """Test ExtensionBridge initialization."""
        from src.tools.extension_bridge import ExtensionBridge

        bridge = ExtensionBridge(port=9999, host="127.0.0.1")

        assert bridge.port == 9999
        assert bridge.host == "127.0.0.1"
        assert bridge._connections == set()
        assert bridge._pending_requests == {}

    @pytest.mark.asyncio
    async def test_start_requires_websockets(self):
        """Test that start requires websockets package."""
        from src.tools.extension_bridge import ExtensionBridge

        with patch.dict("sys.modules", {"websockets": None}):
            with patch("src.tools.extension_bridge.websockets", None):
                bridge = ExtensionBridge()

                with pytest.raises(ImportError, match="websockets"):
                    await bridge.start()

    def test_extension_message(self):
        """Test ExtensionMessage creation."""
        from src.tools.extension_bridge import ExtensionMessage

        msg = ExtensionMessage(
            action="click",
            tabId=123,
            params={"selector": "#button"},
        )

        result = msg.to_dict()

        assert result["action"] == "click"
        assert result["tabId"] == 123
        assert result["params"]["selector"] == "#button"
        assert "requestId" in result

    def test_extension_response(self):
        """Test ExtensionResponse creation."""
        from src.tools.extension_bridge import ExtensionResponse

        response = ExtensionResponse(
            requestId="test-123",
            success=True,
            data={"value": "test"},
        )

        assert response.success is True
        assert response.data["value"] == "test"


class TestComputerUseAutomation:
    """Tests for ComputerUseAutomation class."""

    def test_init(self):
        """Test ComputerUseAutomation initialization."""
        from src.tools.browser_abstraction import BrowserConfig, ComputerUseAutomation

        config = BrowserConfig(viewport_width=1280, viewport_height=720)
        automation = ComputerUseAutomation(config=config)

        assert automation._display_width == 1280
        assert automation._display_height == 720

    @pytest.mark.asyncio
    async def test_start(self):
        """Test starting Computer Use automation."""
        # anthropic is imported inside start(), so we patch at the module level
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            import anthropic
            mock_client = MagicMock()
            anthropic.Anthropic = MagicMock(return_value=mock_client)

            from src.tools.browser_abstraction import ComputerUseAutomation

            automation = ComputerUseAutomation()

            await automation.start()

            assert automation._client is not None

    @pytest.mark.asyncio
    async def test_screenshot_requires_fn(self):
        """Test screenshot requires screenshot_fn."""
        from src.tools.browser_abstraction import ComputerUseAutomation

        automation = ComputerUseAutomation()

        with pytest.raises(NotImplementedError, match="screenshot function"):
            await automation.screenshot()

    @pytest.mark.asyncio
    async def test_screenshot_with_fn(self):
        """Test screenshot with provided function."""
        from src.tools.browser_abstraction import ComputerUseAutomation

        mock_screenshot = MagicMock(return_value=b"fake_png_data")

        automation = ComputerUseAutomation(screenshot_fn=mock_screenshot)

        result = await automation.screenshot()

        assert result == b"fake_png_data"
        mock_screenshot.assert_called_once()


class TestCreateBrowser:
    """Tests for create_browser async factory function."""

    @pytest.mark.asyncio
    async def test_create_browser_starts_automation(self):
        """Test that create_browser starts the automation."""
        with patch("playwright.async_api.async_playwright") as mock_pw:
            mock_playwright = MagicMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()

            # async_playwright() returns object with async start() method
            mock_pw_instance = MagicMock()
            mock_pw_instance.start = AsyncMock(return_value=mock_playwright)
            mock_pw.return_value = mock_pw_instance

            mock_playwright.chromium = MagicMock()
            mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)
            # playwright.stop() is also async
            mock_playwright.stop = AsyncMock()

            from src.tools.browser_abstraction import create_browser

            browser = await create_browser("playwright")

            assert browser._page is not None

            await browser.stop()


class TestModuleExports:
    """Tests for module exports in __init__.py."""

    def test_browser_abstraction_exports(self):
        """Test browser abstraction exports are available."""
        from src.tools import (
            BrowserAutomation,
            BrowserConfig,
            create_browser,
        )

        assert BrowserAutomation is not None
        assert BrowserConfig is not None
        assert create_browser is not None

    def test_extension_bridge_exports(self):
        """Test extension bridge exports are available."""
        from src.tools import (
            ExtensionBridge,
            create_extension_bridge,
        )

        assert ExtensionBridge is not None
        assert create_extension_bridge is not None
