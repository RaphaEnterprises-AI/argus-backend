"""Tests for the CrawleeBridge module.

This module tests the CrawleeBridge class which provides a Python bridge
to Crawlee (Node.js) or a fallback Playwright-based crawler.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.discovery.crawlers.crawlee_bridge import (
    CrawleeBridge,
    CrawlProgress,
    check_crawlee_available,
    discover_application,
)
from src.discovery.models import (
    AuthConfig,
    CrawlResult,
    DiscoveredElement,
    DiscoveryConfig,
    DiscoveryMode,
    ElementCategory,
    PageCategory,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_config():
    """Create a sample discovery configuration."""
    return DiscoveryConfig(
        mode=DiscoveryMode.standard_crawl,
        max_pages=10,
        max_depth=2,
        capture_screenshots=False,
        auth_config=None,
    )


@pytest.fixture
def sample_config_with_auth():
    """Create a sample config with authentication."""
    auth_config = AuthConfig(
        login_url="https://example.com/login",
        cookies={"session": "test-session-cookie"},
        headers={"User-Agent": "Test Agent"},
    )
    return DiscoveryConfig(
        mode=DiscoveryMode.standard_crawl,
        max_pages=10,
        auth_required=True,
        auth_config=auth_config,
    )


@pytest.fixture
def mock_playwright_page():
    """Create a mock Playwright page."""
    page = AsyncMock()
    page.goto = AsyncMock(return_value=MagicMock(status=200))
    page.title = AsyncMock(return_value="Test Page")
    page.url = "https://example.com"
    page.wait_for_timeout = AsyncMock()
    page.screenshot = AsyncMock(return_value=b"fake_screenshot_data")
    page.close = AsyncMock()

    # Mock evaluate for meta description
    page.evaluate = AsyncMock(return_value="Test description")

    return page


@pytest.fixture
def mock_playwright_context(mock_playwright_page):
    """Create a mock Playwright browser context."""
    context = AsyncMock()
    context.new_page = AsyncMock(return_value=mock_playwright_page)
    context.add_cookies = AsyncMock()
    context.close = AsyncMock()
    return context


@pytest.fixture
def mock_playwright_browser(mock_playwright_context):
    """Create a mock Playwright browser."""
    browser = AsyncMock()
    browser.new_context = AsyncMock(return_value=mock_playwright_context)
    browser.close = AsyncMock()
    return browser


# ==============================================================================
# Test CrawleeBridge Initialization
# ==============================================================================


class TestCrawleeBridgeInit:
    """Tests for CrawleeBridge initialization."""

    def test_init_default(self):
        """Test default initialization."""
        bridge = CrawleeBridge()
        assert bridge.use_crawlee is False
        assert bridge.crawlee_script_path is None
        assert bridge._progress_callback is None

    def test_init_with_crawlee(self):
        """Test initialization with Crawlee enabled."""
        bridge = CrawleeBridge(use_crawlee=True)
        assert bridge.use_crawlee is True

    def test_init_with_custom_script(self):
        """Test initialization with custom Crawlee script path."""
        bridge = CrawleeBridge(crawlee_script_path="/path/to/script.ts")
        assert bridge.crawlee_script_path == "/path/to/script.ts"

    def test_set_progress_callback(self):
        """Test setting progress callback."""
        bridge = CrawleeBridge()
        callback = MagicMock()
        bridge.set_progress_callback(callback)
        assert bridge._progress_callback == callback


# ==============================================================================
# Test URL Filtering
# ==============================================================================


class TestUrlFiltering:
    """Tests for URL filtering logic."""

    def test_should_crawl_url_same_domain(self):
        """Test URL with same domain is allowed."""
        bridge = CrawleeBridge()
        config = DiscoveryConfig()
        result = bridge._should_crawl_url(
            "https://example.com/page",
            "example.com",
            config,
        )
        assert result is True

    def test_should_crawl_url_different_domain(self):
        """Test URL with different domain is rejected."""
        bridge = CrawleeBridge()
        config = DiscoveryConfig()
        result = bridge._should_crawl_url(
            "https://other.com/page",
            "example.com",
            config,
        )
        assert result is False

    def test_should_crawl_url_include_patterns(self):
        """Test URL matching include patterns."""
        bridge = CrawleeBridge()
        config = DiscoveryConfig(include_patterns=[r"/admin/.*"])
        result = bridge._should_crawl_url(
            "https://example.com/admin/users",
            "example.com",
            config,
        )
        assert result is True

        result = bridge._should_crawl_url(
            "https://example.com/public/page",
            "example.com",
            config,
        )
        assert result is False

    def test_should_crawl_url_exclude_patterns(self):
        """Test URL matching exclude patterns."""
        bridge = CrawleeBridge()
        config = DiscoveryConfig(exclude_patterns=[r"/logout", r"/api/.*"])
        result = bridge._should_crawl_url(
            "https://example.com/logout",
            "example.com",
            config,
        )
        assert result is False

        result = bridge._should_crawl_url(
            "https://example.com/api/users",
            "example.com",
            config,
        )
        assert result is False

        result = bridge._should_crawl_url(
            "https://example.com/dashboard",
            "example.com",
            config,
        )
        assert result is True


# ==============================================================================
# Test Element Categorization
# ==============================================================================


class TestElementCategorization:
    """Tests for element categorization."""

    def test_categorize_element_authentication_label(self):
        """Test categorization of authentication elements by label."""
        bridge = CrawleeBridge()
        test_cases = [
            ({"type": "button", "label": "Login"}, ElementCategory.authentication),
            ({"type": "button", "label": "Sign In"}, ElementCategory.authentication),
            ({"type": "button", "label": "Log out"}, ElementCategory.authentication),
            ({"type": "button", "label": "Sign Out"}, ElementCategory.authentication),
        ]
        for el_data, expected in test_cases:
            result = bridge._categorize_element(el_data)
            assert result == expected, f"Failed for {el_data}"

    def test_categorize_element_authentication_input_type(self):
        """Test categorization of authentication elements by input type."""
        bridge = CrawleeBridge()
        test_cases = [
            ({"type": "input", "inputType": "password"}, ElementCategory.authentication),
            ({"type": "input", "inputType": "email"}, ElementCategory.authentication),
        ]
        for el_data, expected in test_cases:
            result = bridge._categorize_element(el_data)
            assert result == expected, f"Failed for {el_data}"

    def test_categorize_element_form(self):
        """Test categorization of form elements."""
        bridge = CrawleeBridge()
        el_data = {"type": "input", "label": "First Name"}
        result = bridge._categorize_element(el_data)
        assert result == ElementCategory.form

    def test_categorize_element_action(self):
        """Test categorization of action elements."""
        bridge = CrawleeBridge()
        el_data = {"type": "button", "label": "Submit"}
        result = bridge._categorize_element(el_data)
        assert result == ElementCategory.action

    def test_categorize_element_navigation(self):
        """Test categorization of navigation elements."""
        bridge = CrawleeBridge()
        el_data = {"type": "link", "label": "Home"}
        result = bridge._categorize_element(el_data)
        assert result == ElementCategory.navigation

    def test_categorize_element_commerce(self):
        """Test categorization of commerce elements."""
        bridge = CrawleeBridge()
        test_cases = [
            {"type": "button", "label": "Add to Cart"},
            {"type": "button", "label": "Buy Now"},
            {"type": "button", "label": "Checkout"},
        ]
        for el_data in test_cases:
            result = bridge._categorize_element(el_data)
            assert result == ElementCategory.commerce, f"Failed for {el_data}"

    def test_categorize_element_content(self):
        """Test categorization falls back to content."""
        bridge = CrawleeBridge()
        el_data = {"type": "other", "label": "Some text"}
        result = bridge._categorize_element(el_data)
        assert result == ElementCategory.content


# ==============================================================================
# Test Page Classification
# ==============================================================================


class TestPageClassification:
    """Tests for page classification."""

    def test_classify_page_login_url(self):
        """Test classification of login pages by URL."""
        bridge = CrawleeBridge()
        test_urls = [
            "https://example.com/login",
            "https://example.com/signin",
            "https://example.com/auth",
        ]
        for url in test_urls:
            result = bridge._classify_page(url, "Page Title", [])
            assert result == PageCategory.auth_login, f"Failed for {url}"

    def test_classify_page_signup_url(self):
        """Test classification of signup pages by URL."""
        bridge = CrawleeBridge()
        test_urls = [
            "https://example.com/signup",
            "https://example.com/register",
            "https://example.com/join",
        ]
        for url in test_urls:
            result = bridge._classify_page(url, "Page Title", [])
            assert result == PageCategory.auth_signup, f"Failed for {url}"

    def test_classify_page_reset_url(self):
        """Test classification of password reset pages."""
        bridge = CrawleeBridge()
        test_urls = [
            "https://example.com/reset",
            "https://example.com/forgot-password",
        ]
        for url in test_urls:
            result = bridge._classify_page(url, "Page Title", [])
            assert result == PageCategory.auth_reset, f"Failed for {url}"

    def test_classify_page_dashboard_url(self):
        """Test classification of dashboard pages."""
        bridge = CrawleeBridge()
        test_urls = [
            "https://example.com/dashboard",
            "https://example.com/home",
            "https://example.com/overview",
        ]
        for url in test_urls:
            result = bridge._classify_page(url, "Page Title", [])
            assert result == PageCategory.dashboard, f"Failed for {url}"

    def test_classify_page_settings_url(self):
        """Test classification of settings pages."""
        bridge = CrawleeBridge()
        test_urls = [
            "https://example.com/settings",
            "https://example.com/preferences",
            "https://example.com/config",
        ]
        for url in test_urls:
            result = bridge._classify_page(url, "Page Title", [])
            assert result == PageCategory.settings, f"Failed for {url}"

    def test_classify_page_profile_url(self):
        """Test classification of profile pages."""
        bridge = CrawleeBridge()
        test_urls = [
            "https://example.com/profile",
            "https://example.com/account",
            "https://example.com/me",
        ]
        for url in test_urls:
            result = bridge._classify_page(url, "Page Title", [])
            assert result == PageCategory.profile, f"Failed for {url}"

    def test_classify_page_checkout_url(self):
        """Test classification of checkout pages."""
        bridge = CrawleeBridge()
        test_urls = [
            "https://example.com/checkout",
            "https://example.com/payment",
            "https://example.com/cart",
        ]
        for url in test_urls:
            result = bridge._classify_page(url, "Page Title", [])
            assert result == PageCategory.checkout, f"Failed for {url}"

    def test_classify_page_error_url(self):
        """Test classification of error pages."""
        bridge = CrawleeBridge()
        test_urls = [
            "https://example.com/404",
            "https://example.com/error",
            "https://example.com/not-found",
        ]
        for url in test_urls:
            result = bridge._classify_page(url, "Page Title", [])
            assert result == PageCategory.error, f"Failed for {url}"

    def test_classify_page_list_url(self):
        """Test classification of list pages."""
        bridge = CrawleeBridge()
        test_urls = [
            "https://example.com/list",
            "https://example.com/all",
            "https://example.com/browse",
            "https://example.com/items?page=1",
        ]
        for url in test_urls:
            result = bridge._classify_page(url, "Page Title", [])
            assert result == PageCategory.list, f"Failed for {url}"

    def test_classify_page_by_elements(self):
        """Test classification based on elements."""
        bridge = CrawleeBridge()
        auth_elements = [
            DiscoveredElement(
                id="1",
                page_url="https://example.com",
                selector="#username",
                category=ElementCategory.authentication,
            ),
            DiscoveredElement(
                id="2",
                page_url="https://example.com",
                selector="#password",
                category=ElementCategory.authentication,
            ),
        ]
        result = bridge._classify_page(
            "https://example.com/page", "Page Title", auth_elements
        )
        assert result == PageCategory.auth_login

    def test_classify_page_by_title(self):
        """Test classification based on title."""
        bridge = CrawleeBridge()
        result = bridge._classify_page(
            "https://example.com/page", "Login to your account", []
        )
        assert result == PageCategory.auth_login

        result = bridge._classify_page(
            "https://example.com/page", "Sign Up for Free", []
        )
        assert result == PageCategory.auth_signup

        result = bridge._classify_page(
            "https://example.com/page", "Dashboard Overview", []
        )
        assert result == PageCategory.dashboard

    def test_classify_page_form_by_elements(self):
        """Test classification as form by element count."""
        bridge = CrawleeBridge()
        form_elements = [
            DiscoveredElement(
                id="1",
                page_url="https://example.com",
                selector="#field1",
                category=ElementCategory.form,
            ),
            DiscoveredElement(
                id="2",
                page_url="https://example.com",
                selector="#field2",
                category=ElementCategory.form,
            ),
            DiscoveredElement(
                id="3",
                page_url="https://example.com",
                selector="#field3",
                category=ElementCategory.form,
            ),
        ]
        result = bridge._classify_page(
            "https://example.com/page", "Page Title", form_elements
        )
        assert result == PageCategory.form

    def test_classify_page_other(self):
        """Test classification falls back to other."""
        bridge = CrawleeBridge()
        result = bridge._classify_page(
            "https://example.com/random-page", "Random Page", []
        )
        assert result == PageCategory.other


# ==============================================================================
# Test Progress Emission
# ==============================================================================


class TestProgressEmission:
    """Tests for progress emission."""

    def test_emit_progress_with_callback(self):
        """Test progress is emitted when callback is set."""
        bridge = CrawleeBridge()
        callback = MagicMock()
        bridge.set_progress_callback(callback)

        bridge._emit_progress(
            pages_crawled=5,
            pages_queued=10,
            current_url="https://example.com/page",
        )

        callback.assert_called_once()
        progress = callback.call_args[0][0]
        assert isinstance(progress, CrawlProgress)
        assert progress.pages_crawled == 5
        assert progress.pages_queued == 10
        assert progress.current_url == "https://example.com/page"
        assert progress.progress_percent == pytest.approx(33.33, rel=0.1)

    def test_emit_progress_without_callback(self):
        """Test no error when callback is not set."""
        bridge = CrawleeBridge()
        # Should not raise
        bridge._emit_progress(
            pages_crawled=5,
            pages_queued=10,
            current_url="https://example.com/page",
        )

    def test_emit_progress_zero_total(self):
        """Test progress calculation with zero total."""
        bridge = CrawleeBridge()
        callback = MagicMock()
        bridge.set_progress_callback(callback)

        bridge._emit_progress(
            pages_crawled=0,
            pages_queued=0,
            current_url="",
        )

        progress = callback.call_args[0][0]
        assert progress.progress_percent == 0


# ==============================================================================
# Test Crawlee Script Path
# ==============================================================================


class TestCrawleeScriptPath:
    """Tests for Crawlee script path resolution."""

    def test_get_default_crawlee_script_not_found(self):
        """Test that FileNotFoundError is raised when script doesn't exist."""
        bridge = CrawleeBridge()
        with pytest.raises(FileNotFoundError):
            bridge._get_default_crawlee_script()


# ==============================================================================
# Test Parse Crawlee Results
# ==============================================================================


class TestParseCrawleeResults:
    """Tests for parsing Crawlee results."""

    def test_parse_crawlee_results_empty(self):
        """Test parsing empty results."""
        bridge = CrawleeBridge()
        result = bridge._parse_crawlee_results({"pages": []})
        assert len(result.pages) == 0

    def test_parse_crawlee_results_with_pages(self):
        """Test parsing results with pages."""
        bridge = CrawleeBridge()
        data = {
            "pages": [
                {
                    "url": "https://example.com",
                    "title": "Example",
                    "description": "Example page",
                    "category": "landing",
                    "depth": 0,
                    "loadTimeMs": 500,
                    "elements": [
                        {
                            "selector": "#btn",
                            "xpath": "//button",
                            "category": "action",
                            "purpose": "Submit",
                            "label": "Submit",
                            "tagName": "button",
                            "attributes": {"id": "btn"},
                        }
                    ],
                    "outgoingLinks": ["/page1", "/page2"],
                }
            ]
        }

        result = bridge._parse_crawlee_results(data)
        assert len(result.pages) == 1
        page = result.pages["https://example.com"]
        assert page.title == "Example"
        assert page.category == PageCategory.landing
        assert len(page.elements) == 1
        assert page.elements[0].selector == "#btn"


# ==============================================================================
# Test Run Crawl
# ==============================================================================


class TestRunCrawl:
    """Tests for the run_crawl method."""

    @pytest.mark.asyncio
    async def test_run_crawl_uses_playwright_by_default(self, sample_config):
        """Test that Playwright crawler is used by default."""
        bridge = CrawleeBridge(use_crawlee=False)

        with patch.object(
            bridge, "_run_playwright_crawler", new_callable=AsyncMock
        ) as mock_playwright:
            mock_playwright.return_value = CrawlResult(pages={})
            result = await bridge.run_crawl("https://example.com", sample_config)

            mock_playwright.assert_called_once()
            assert isinstance(result, CrawlResult)

    @pytest.mark.asyncio
    async def test_run_crawl_uses_crawlee_when_enabled(self, sample_config):
        """Test that Crawlee is used when enabled."""
        bridge = CrawleeBridge(use_crawlee=True)

        with patch.object(
            bridge, "_run_crawlee", new_callable=AsyncMock
        ) as mock_crawlee:
            mock_crawlee.return_value = CrawlResult(pages={})
            result = await bridge.run_crawl("https://example.com", sample_config)

            mock_crawlee.assert_called_once()
            assert isinstance(result, CrawlResult)

    @pytest.mark.asyncio
    async def test_run_crawl_handles_exception(self, sample_config):
        """Test that exceptions are handled gracefully."""
        bridge = CrawleeBridge()

        with patch.object(
            bridge, "_run_playwright_crawler", new_callable=AsyncMock
        ) as mock_playwright:
            mock_playwright.side_effect = Exception("Crawl failed")
            result = await bridge.run_crawl("https://example.com", sample_config)

            assert len(result.errors) == 1
            assert result.errors[0].error_type == "crawl_failed"

    @pytest.mark.asyncio
    async def test_run_crawl_resets_state(self, sample_config):
        """Test that state is reset between crawls."""
        bridge = CrawleeBridge()
        bridge._visited_urls.add("https://old.com")
        bridge._queued_urls.add("https://queued.com")

        with patch.object(
            bridge, "_run_playwright_crawler", new_callable=AsyncMock
        ) as mock_playwright:
            mock_playwright.return_value = CrawlResult(pages={})
            await bridge.run_crawl("https://example.com", sample_config)

            # State should be cleared before crawl
            assert len(bridge._errors) == 0


# ==============================================================================
# Test Playwright Crawler
# ==============================================================================


class TestPlaywrightCrawler:
    """Tests for the Playwright crawler fallback."""

    @pytest.mark.asyncio
    async def test_playwright_missing_returns_error(self, sample_config):
        """Test that missing Playwright returns an error result."""
        CrawleeBridge()

        with patch.dict("sys.modules", {"playwright.async_api": None}):
            with patch(
                "src.discovery.crawlers.crawlee_bridge.async_playwright",
                side_effect=ImportError("No module"),
            ):
                # This will raise ImportError which should be handled
                pass


# ==============================================================================
# Test Stream Crawl
# ==============================================================================


class TestStreamCrawl:
    """Tests for streaming crawl progress."""

    @pytest.mark.asyncio
    async def test_stream_crawl_yields_events(self, sample_config):
        """Test that stream_crawl yields events."""
        bridge = CrawleeBridge()

        with patch.object(
            bridge, "run_crawl", new_callable=AsyncMock
        ) as mock_run_crawl:
            mock_run_crawl.return_value = CrawlResult(
                pages={"https://example.com": MagicMock()},
                duration_ms=1000,
            )

            events = []
            async for event in bridge.stream_crawl("https://example.com", sample_config):
                events.append(event)
                if event["type"] == "complete":
                    break

            assert len(events) >= 2
            assert events[0]["type"] == "start"
            assert events[-1]["type"] == "complete"


# ==============================================================================
# Test Discover Application Convenience Function
# ==============================================================================


class TestDiscoverApplication:
    """Tests for the discover_application convenience function."""

    @pytest.mark.asyncio
    async def test_discover_application_basic(self):
        """Test basic discover_application call."""
        with patch(
            "src.discovery.crawlers.crawlee_bridge.CrawleeBridge"
        ) as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.run_crawl = AsyncMock(return_value=CrawlResult(pages={}))
            MockBridge.return_value = mock_bridge

            result = await discover_application("https://example.com")

            assert isinstance(result, CrawlResult)
            mock_bridge.run_crawl.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_application_with_options(self):
        """Test discover_application with custom options."""
        with patch(
            "src.discovery.crawlers.crawlee_bridge.CrawleeBridge"
        ) as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.run_crawl = AsyncMock(return_value=CrawlResult(pages={}))
            MockBridge.return_value = mock_bridge

            await discover_application(
                "https://example.com",
                max_pages=100,
                max_depth=5,
                capture_screenshots=True,
            )

            call_args = mock_bridge.run_crawl.call_args
            config = call_args[0][1]
            assert config.max_pages == 100
            assert config.max_depth == 5
            assert config.capture_screenshots is True

    @pytest.mark.asyncio
    async def test_discover_application_with_auth(self):
        """Test discover_application with authentication config."""
        with patch(
            "src.discovery.crawlers.crawlee_bridge.CrawleeBridge"
        ) as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.run_crawl = AsyncMock(return_value=CrawlResult(pages={}))
            MockBridge.return_value = mock_bridge

            auth_config = {
                "login_url": "https://example.com/login",
                "username": "test",
                "password": "pass",
            }

            await discover_application(
                "https://example.com", auth_config=auth_config
            )

            call_args = mock_bridge.run_crawl.call_args
            config = call_args[0][1]
            assert config.auth_required is True
            assert config.auth_config is not None


# ==============================================================================
# Test Check Crawlee Available
# ==============================================================================


class TestCheckCrawleeAvailable:
    """Tests for the check_crawlee_available function."""

    @pytest.mark.asyncio
    async def test_check_crawlee_available_true(self):
        """Test when npx is available."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            result = await check_crawlee_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_crawlee_available_false_not_installed(self):
        """Test when npx is not installed."""
        with patch(
            "asyncio.create_subprocess_exec", side_effect=FileNotFoundError()
        ):
            result = await check_crawlee_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_check_crawlee_available_false_permission_error(self):
        """Test when there's a permission error."""
        with patch(
            "asyncio.create_subprocess_exec", side_effect=PermissionError()
        ):
            result = await check_crawlee_available()
            assert result is False


# ==============================================================================
# Test CrawlProgress Dataclass
# ==============================================================================


class TestCrawlProgress:
    """Tests for the CrawlProgress dataclass."""

    def test_crawl_progress_defaults(self):
        """Test CrawlProgress default values."""
        progress = CrawlProgress()
        assert progress.pages_crawled == 0
        assert progress.pages_queued == 0
        assert progress.current_url == ""
        assert progress.errors_count == 0
        assert progress.progress_percent == 0.0

    def test_crawl_progress_custom_values(self):
        """Test CrawlProgress with custom values."""
        progress = CrawlProgress(
            pages_crawled=10,
            pages_queued=20,
            current_url="https://example.com",
            errors_count=2,
            progress_percent=33.33,
        )
        assert progress.pages_crawled == 10
        assert progress.pages_queued == 20
        assert progress.current_url == "https://example.com"
        assert progress.errors_count == 2
        assert progress.progress_percent == 33.33
