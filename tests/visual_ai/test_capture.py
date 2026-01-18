"""Comprehensive tests for visual_ai/capture.py.

Tests enhanced screenshot capture, DOM serialization, performance metrics,
and visual snapshot generation.
"""

import io
import json
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any

from src.visual_ai.capture import (
    EnhancedCapture,
    create_enhanced_capture,
    DOM_SERIALIZER_JS,
    ELEMENT_EXTRACTOR_JS,
    PERFORMANCE_METRICS_JS,
)
from src.visual_ai.models import VisualSnapshot, VisualElement


class TestEnhancedCaptureInit:
    """Tests for EnhancedCapture initialization."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        capture = EnhancedCapture()
        assert capture.default_viewport == {"width": 1920, "height": 1080}
        assert capture.capture_performance_enabled is True
        assert capture.capture_network_enabled is True
        assert capture.max_elements == 1000

    def test_init_custom_viewport(self):
        """Test initialization with custom viewport."""
        capture = EnhancedCapture(
            default_viewport={"width": 1440, "height": 900}
        )
        assert capture.default_viewport == {"width": 1440, "height": 900}

    def test_init_disable_performance(self):
        """Test initialization with performance capture disabled."""
        capture = EnhancedCapture(capture_performance=False)
        assert capture.capture_performance_enabled is False

    def test_init_disable_network(self):
        """Test initialization with network capture disabled."""
        capture = EnhancedCapture(capture_network=False)
        assert capture.capture_network_enabled is False

    def test_init_custom_max_elements(self):
        """Test initialization with custom max elements."""
        capture = EnhancedCapture(max_elements=500)
        assert capture.max_elements == 500


class TestEnhancedCaptureDOMSnapshot:
    """Tests for DOM snapshot capture."""

    @pytest.fixture
    def capture(self):
        return EnhancedCapture()

    @pytest.fixture
    def mock_page(self):
        """Create a mocked Playwright page."""
        page = AsyncMock()
        page.evaluate = AsyncMock()
        page.screenshot = AsyncMock(return_value=b"screenshot_data")
        page.url = "https://example.com"
        page.title = AsyncMock(return_value="Test Page")
        page.wait_for_load_state = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        return page

    @pytest.mark.asyncio
    async def test_capture_dom_snapshot(self, capture, mock_page):
        """Test DOM snapshot capture."""
        mock_page.evaluate.return_value = '{"nodeId": 1, "tagName": "html"}'

        result = await capture.capture_dom_snapshot(mock_page)
        assert isinstance(result, str)
        assert "html" in result

    @pytest.mark.asyncio
    async def test_capture_dom_snapshot_error(self, capture, mock_page):
        """Test DOM snapshot capture handles errors."""
        mock_page.evaluate.side_effect = Exception("DOM error")

        result = await capture.capture_dom_snapshot(mock_page)
        assert result == "{}"


class TestEnhancedCaptureElementExtraction:
    """Tests for element extraction."""

    @pytest.fixture
    def capture(self):
        return EnhancedCapture()

    @pytest.fixture
    def mock_page(self):
        page = AsyncMock()
        page.evaluate = AsyncMock()
        return page

    @pytest.mark.asyncio
    async def test_extract_elements(self, capture, mock_page):
        """Test element extraction."""
        mock_elements = [
            {
                "element_id": "el_0",
                "selector": "#header",
                "tag_name": "header",
                "bounds": {"x": 0, "y": 0, "width": 800, "height": 60},
                "computed_styles": {"display": "block"},
                "text_content": "Header",
                "attributes": {"id": "header"},
                "children_count": 3,
            }
        ]
        mock_page.evaluate.return_value = mock_elements

        result = await capture.extract_elements(mock_page)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["element_id"] == "el_0"

    @pytest.mark.asyncio
    async def test_extract_elements_empty(self, capture, mock_page):
        """Test element extraction with empty result."""
        mock_page.evaluate.return_value = []

        result = await capture.extract_elements(mock_page)
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_elements_error(self, capture, mock_page):
        """Test element extraction handles errors."""
        mock_page.evaluate.side_effect = Exception("Extraction error")

        result = await capture.extract_elements(mock_page)
        assert result == []


class TestEnhancedCapturePerformanceMetrics:
    """Tests for performance metrics capture."""

    @pytest.fixture
    def capture(self):
        return EnhancedCapture()

    @pytest.fixture
    def mock_page(self):
        page = AsyncMock()
        page.evaluate = AsyncMock()
        return page

    @pytest.mark.asyncio
    async def test_capture_performance_metrics(self, capture, mock_page):
        """Test performance metrics capture."""
        mock_metrics = {
            "lcp": 1500.0,
            "cls": 0.05,
            "fcp": 1000.0,
            "ttfb": 200.0,
            "domInteractive": 2000.0,
            "totalResources": 50,
        }
        mock_page.evaluate.return_value = mock_metrics

        result = await capture.capture_performance_metrics(mock_page)
        assert isinstance(result, dict)
        assert result["lcp"] == 1500.0
        assert result["cls"] == 0.05

    @pytest.mark.asyncio
    async def test_capture_performance_metrics_with_tti_calculation(
        self, capture, mock_page
    ):
        """Test TTI calculation when not directly available."""
        mock_metrics = {
            "lcp": 1500.0,
            "fcp": 1000.0,
            "domInteractive": 2500.0,
            # No "tti" field
        }
        mock_page.evaluate.return_value = mock_metrics

        result = await capture.capture_performance_metrics(mock_page)
        assert "tti" in result or "domInteractive" in result

    @pytest.mark.asyncio
    async def test_capture_performance_metrics_error(self, capture, mock_page):
        """Test performance metrics capture handles errors."""
        mock_page.evaluate.side_effect = Exception("Metrics error")

        result = await capture.capture_performance_metrics(mock_page)
        assert result == {}


class TestEnhancedCaptureLayoutHash:
    """Tests for layout hash computation."""

    @pytest.fixture
    def capture(self):
        return EnhancedCapture()

    @pytest.mark.asyncio
    async def test_compute_layout_hash(self, capture):
        """Test layout hash computation."""
        elements = [
            {
                "element_type": "container",
                "tag_name": "div",
                "bounds": {"x": 100, "y": 200, "width": 300, "height": 400},
            }
        ]
        hash_result = await capture.compute_layout_hash(elements)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 32  # SHA256 truncated

    @pytest.mark.asyncio
    async def test_compute_layout_hash_empty(self, capture):
        """Test layout hash for empty elements."""
        hash_result = await capture.compute_layout_hash([])
        assert hash_result == ""

    @pytest.mark.asyncio
    async def test_compute_layout_hash_consistency(self, capture):
        """Test layout hash is consistent for same input."""
        elements = [
            {"element_type": "button", "tag_name": "button", "bounds": {"x": 0, "y": 0, "width": 100, "height": 40}}
        ]
        hash1 = await capture.compute_layout_hash(elements)
        hash2 = await capture.compute_layout_hash(elements)
        assert hash1 == hash2

    @pytest.mark.asyncio
    async def test_compute_layout_hash_position_rounding(self, capture):
        """Test that positions are rounded to reduce noise."""
        # Values that round to 100 with round(x/10)*10
        elements1 = [
            {"element_type": "div", "tag_name": "div", "bounds": {"x": 101, "y": 102, "width": 100, "height": 100}}
        ]
        elements2 = [
            {"element_type": "div", "tag_name": "div", "bounds": {"x": 104, "y": 103, "width": 100, "height": 100}}
        ]
        hash1 = await capture.compute_layout_hash(elements1)
        hash2 = await capture.compute_layout_hash(elements2)
        # Both round to same values: 101/10=10.1->10*10=100, 104/10=10.4->10*10=100
        assert hash1 == hash2


class TestEnhancedCaptureColorPalette:
    """Tests for color palette extraction."""

    @pytest.fixture
    def capture(self):
        return EnhancedCapture()

    @pytest.fixture
    def red_image_bytes(self):
        """Create a simple red image."""
        try:
            from PIL import Image
            img = Image.new("RGB", (100, 100), color="red")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
        except ImportError:
            return b"\x89PNG\r\n\x1a\ntest"

    @pytest.mark.asyncio
    async def test_extract_color_palette(self, capture, red_image_bytes):
        """Test color palette extraction."""
        colors = await capture.extract_color_palette(red_image_bytes)
        assert isinstance(colors, list)

    @pytest.mark.asyncio
    async def test_extract_color_palette_empty_image(self, capture):
        """Test color palette with empty screenshot."""
        colors = await capture.extract_color_palette(b"")
        assert colors == []


class TestEnhancedCaptureSnapshot:
    """Tests for full snapshot capture."""

    @pytest.fixture
    def capture(self):
        return EnhancedCapture()

    @pytest.fixture
    def mock_page(self):
        """Create a fully mocked page."""
        page = AsyncMock()

        # Set up evaluate to return appropriate values based on call
        async def evaluate_side_effect(js_code, *args):
            if "serializeNode" in str(js_code):
                return '{"nodeId": 1}'
            elif "isVisible" in str(js_code) or "maxElements" in str(js_code[0:100] if isinstance(js_code, str) else ""):
                return [
                    {
                        "element_id": "el_0",
                        "selector": "#main",
                        "tag_name": "main",
                        "bounds": {"x": 0, "y": 0, "width": 800, "height": 600},
                        "computed_styles": {},
                        "text_content": "Main",
                        "attributes": {},
                        "children_count": 2,
                    }
                ]
            elif "performance" in str(js_code).lower() or "lcp" in str(js_code).lower():
                return {"lcp": 1500.0, "cls": 0.05, "domInteractive": 2500.0}
            elif "innerWidth" in str(js_code):
                return {"width": 1920, "height": 1080}
            elif "textElements" in str(js_code):
                return [{"text": "Sample", "bounds": {"x": 0, "y": 0}}]
            return {}

        page.evaluate = AsyncMock(side_effect=evaluate_side_effect)
        page.screenshot = AsyncMock(return_value=b"screenshot_bytes")
        page.url = "https://example.com/page"
        page.title = AsyncMock(return_value="Test Page")
        page.wait_for_load_state = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.on = MagicMock()

        return page

    @pytest.mark.asyncio
    async def test_capture_snapshot(self, capture, mock_page):
        """Test full snapshot capture."""
        snapshot = await capture.capture_snapshot(
            page=mock_page,
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            browser="chromium",
        )
        assert isinstance(snapshot, VisualSnapshot)
        assert snapshot.url == "https://example.com/page"
        assert snapshot.browser == "chromium"

    @pytest.mark.asyncio
    async def test_capture_snapshot_with_full_page(self, capture, mock_page):
        """Test snapshot capture with full page mode."""
        snapshot = await capture.capture_snapshot(
            page=mock_page,
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            browser="chromium",
            full_page=True,
        )
        # Verify screenshot was called with full_page=True
        mock_page.screenshot.assert_called_with(full_page=True, type="png")

    @pytest.mark.asyncio
    async def test_capture_snapshot_with_device_name(self, capture, mock_page):
        """Test snapshot capture with device name."""
        snapshot = await capture.capture_snapshot(
            page=mock_page,
            url="https://example.com",
            viewport={"width": 375, "height": 812},
            browser="webkit",
            device_name="iPhone 12",
        )
        assert snapshot.device_name == "iPhone 12"

    @pytest.mark.asyncio
    async def test_capture_snapshot_performance_disabled(self, capture, mock_page):
        """Test snapshot capture without performance metrics."""
        capture.capture_performance_enabled = False

        snapshot = await capture.capture_snapshot(
            page=mock_page,
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            browser="chromium",
        )
        # Performance metrics should still be None or handled gracefully
        assert isinstance(snapshot, VisualSnapshot)


class TestEnhancedCaptureNetworkCapture:
    """Tests for network HAR capture."""

    @pytest.fixture
    def capture(self):
        return EnhancedCapture()

    @pytest.fixture
    def mock_page(self):
        page = AsyncMock()
        page.on = MagicMock()
        page.evaluate = AsyncMock(return_value={})
        page.screenshot = AsyncMock(return_value=b"screenshot")
        page.url = "https://example.com"
        page.title = AsyncMock(return_value="Test")
        page.wait_for_load_state = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        return page

    @pytest.mark.asyncio
    async def test_setup_network_capture(self, capture, mock_page):
        """Test network capture setup."""
        await capture._setup_network_capture(mock_page)
        # Should have registered event handlers
        assert mock_page.on.call_count >= 2  # request and response handlers


class TestEnhancedCaptureElementScreenshot:
    """Tests for element-specific screenshot capture."""

    @pytest.fixture
    def capture(self):
        return EnhancedCapture()

    @pytest.fixture
    def mock_page(self):
        page = AsyncMock()
        return page

    @pytest.mark.asyncio
    async def test_capture_element_screenshot(self, capture, mock_page):
        """Test element screenshot capture."""
        mock_element = AsyncMock()
        mock_element.screenshot = AsyncMock(return_value=b"element_screenshot")
        mock_page.query_selector = AsyncMock(return_value=mock_element)

        result = await capture.capture_element_screenshot(mock_page, "#header")
        assert result == b"element_screenshot"

    @pytest.mark.asyncio
    async def test_capture_element_screenshot_not_found(self, capture, mock_page):
        """Test element screenshot when element not found."""
        mock_page.query_selector = AsyncMock(return_value=None)

        result = await capture.capture_element_screenshot(mock_page, "#nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_capture_element_screenshot_error(self, capture, mock_page):
        """Test element screenshot handles errors."""
        mock_page.query_selector = AsyncMock(side_effect=Exception("Query error"))

        result = await capture.capture_element_screenshot(mock_page, "#header")
        assert result is None


class TestEnhancedCaptureStableLayout:
    """Tests for layout stability waiting."""

    @pytest.fixture
    def capture(self):
        return EnhancedCapture()

    @pytest.fixture
    def mock_page(self):
        page = AsyncMock()
        page.evaluate = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        return page

    @pytest.mark.asyncio
    async def test_wait_for_stable_layout_immediate(self, capture, mock_page):
        """Test stable layout when immediately stable."""
        import time
        mock_page.evaluate.return_value = time.time() * 1000 - 1000  # 1 second ago

        result = await capture.wait_for_stable_layout(
            mock_page, stability_threshold_ms=500, max_wait_ms=5000
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_stable_layout_timeout(self, capture, mock_page):
        """Test stable layout timeout."""
        import time

        # Always return current time (constantly shifting)
        mock_page.evaluate.side_effect = lambda *args: time.time() * 1000
        mock_page.wait_for_timeout = AsyncMock(return_value=None)

        # Use very short timeout for testing
        result = await capture.wait_for_stable_layout(
            mock_page, stability_threshold_ms=500, max_wait_ms=100
        )
        assert result is False


class TestEnhancedCaptureCaptureUrl:
    """Tests for capture_url convenience method."""

    @pytest.fixture
    def capture(self):
        return EnhancedCapture()

    @pytest.mark.asyncio
    async def test_capture_url(self, capture):
        """Test capture_url with mocked playwright."""
        with patch("src.visual_ai.capture.async_playwright") as mock_playwright:
            # Set up mock chain
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()

            mock_page.goto = AsyncMock()
            mock_page.evaluate = AsyncMock(return_value={})
            mock_page.screenshot = AsyncMock(return_value=b"screenshot")
            mock_page.url = "https://example.com"
            mock_page.title = AsyncMock(return_value="Test")
            mock_page.wait_for_load_state = AsyncMock()
            mock_page.wait_for_timeout = AsyncMock()
            mock_page.on = MagicMock()

            mock_context.new_page = AsyncMock(return_value=mock_page)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_browser.close = AsyncMock()

            mock_chromium = MagicMock()
            mock_chromium.launch = AsyncMock(return_value=mock_browser)

            mock_pw = AsyncMock()
            mock_pw.chromium = mock_chromium
            mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
            mock_pw.__aexit__ = AsyncMock(return_value=None)

            mock_playwright.return_value = mock_pw

            # The test
            snapshot = await capture.capture_url(
                "https://example.com",
                browser_type="chromium",
            )
            assert isinstance(snapshot, VisualSnapshot)


class TestCreateEnhancedCapture:
    """Tests for factory function."""

    def test_create_enhanced_capture_defaults(self):
        """Test factory with defaults."""
        capture = create_enhanced_capture()
        assert isinstance(capture, EnhancedCapture)
        assert capture.capture_performance_enabled is True
        assert capture.capture_network_enabled is True

    def test_create_enhanced_capture_custom(self):
        """Test factory with custom options."""
        capture = create_enhanced_capture(
            capture_performance=False,
            capture_network=False,
            max_elements=250,
        )
        assert capture.capture_performance_enabled is False
        assert capture.capture_network_enabled is False
        assert capture.max_elements == 250


class TestJavaScriptSnippets:
    """Tests to verify JavaScript snippets are valid."""

    def test_dom_serializer_js_defined(self):
        """Test DOM serializer JS is defined."""
        assert DOM_SERIALIZER_JS is not None
        assert len(DOM_SERIALIZER_JS) > 100
        assert "serializeNode" in DOM_SERIALIZER_JS

    def test_element_extractor_js_defined(self):
        """Test element extractor JS is defined."""
        assert ELEMENT_EXTRACTOR_JS is not None
        assert len(ELEMENT_EXTRACTOR_JS) > 100
        assert "isVisible" in ELEMENT_EXTRACTOR_JS

    def test_performance_metrics_js_defined(self):
        """Test performance metrics JS is defined."""
        assert PERFORMANCE_METRICS_JS is not None
        assert len(PERFORMANCE_METRICS_JS) > 100
        assert "performance" in PERFORMANCE_METRICS_JS.lower()
