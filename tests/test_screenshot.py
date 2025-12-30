"""Tests for screenshot capture utilities."""

import io
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from PIL import Image


def create_test_image(width: int = 100, height: int = 100, color: str = "red") -> bytes:
    """Create a test image for testing."""
    img = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestScreenshot:
    """Tests for Screenshot dataclass."""

    def test_screenshot_creation(self):
        """Test creating a screenshot."""
        from src.computer_use.screenshot import Screenshot

        data = create_test_image()
        screenshot = Screenshot(
            data=data,
            width=100,
            height=100,
        )

        assert screenshot.width == 100
        assert screenshot.height == 100
        assert screenshot.format == "png"
        assert len(screenshot.data) > 0

    def test_screenshot_base64(self):
        """Test base64 encoding."""
        from src.computer_use.screenshot import Screenshot
        import base64

        data = create_test_image()
        screenshot = Screenshot(data=data, width=100, height=100)

        encoded = screenshot.base64
        decoded = base64.b64decode(encoded)

        assert decoded == data

    def test_screenshot_data_uri(self):
        """Test data URI generation."""
        from src.computer_use.screenshot import Screenshot

        data = create_test_image()
        screenshot = Screenshot(data=data, width=100, height=100)

        uri = screenshot.data_uri

        assert uri.startswith("data:image/png;base64,")

    def test_screenshot_resize(self):
        """Test screenshot resizing."""
        from src.computer_use.screenshot import Screenshot

        data = create_test_image(200, 200)
        screenshot = Screenshot(data=data, width=200, height=200)

        resized = screenshot.resize(100, 100)

        assert resized.width == 100
        assert resized.height == 100

        # Verify the image is actually resized
        img = Image.open(io.BytesIO(resized.data))
        assert img.size == (100, 100)

    def test_screenshot_save(self, tmp_path):
        """Test saving screenshot to file."""
        from src.computer_use.screenshot import Screenshot

        data = create_test_image()
        screenshot = Screenshot(data=data, width=100, height=100)

        file_path = tmp_path / "test.png"
        screenshot.save(file_path)

        assert file_path.exists()
        assert file_path.read_bytes() == data


class TestScreenshotCapture:
    """Tests for ScreenshotCapture class."""

    def test_capture_init_defaults(self):
        """Test ScreenshotCapture initialization with defaults."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture()

        assert capture.default_width == 1920
        assert capture.default_height == 1080
        assert capture.optimize is True

    def test_capture_init_custom(self):
        """Test ScreenshotCapture initialization with custom values."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture(
            default_width=1280,
            default_height=720,
            optimize=False,
        )

        assert capture.default_width == 1280
        assert capture.default_height == 720
        assert capture.optimize is False

    @pytest.mark.asyncio
    async def test_capture_playwright(self):
        """Test capturing from Playwright page."""
        from src.computer_use.screenshot import ScreenshotCapture

        # Create mock page with consistent 1920x1080 test image
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=create_test_image(1920, 1080))

        capture = ScreenshotCapture()
        screenshot = await capture.capture_playwright(mock_page)

        # The returned dimensions should match the actual image
        assert screenshot.width == 1920
        assert screenshot.height == 1080
        mock_page.screenshot.assert_called_once()

    def test_compare_screenshots_identical(self):
        """Test comparing identical screenshots."""
        from src.computer_use.screenshot import ScreenshotCapture, Screenshot

        data = create_test_image(100, 100, "blue")
        screenshot1 = Screenshot(data=data, width=100, height=100)
        screenshot2 = Screenshot(data=data, width=100, height=100)

        capture = ScreenshotCapture()
        is_same, diff = capture.compare_screenshots(screenshot1, screenshot2)

        # Use == instead of is for numpy bool comparison
        assert bool(is_same) == True
        assert float(diff) == 0.0

    def test_compare_screenshots_different(self):
        """Test comparing different screenshots."""
        from src.computer_use.screenshot import ScreenshotCapture, Screenshot

        data1 = create_test_image(100, 100, "red")
        data2 = create_test_image(100, 100, "blue")
        screenshot1 = Screenshot(data=data1, width=100, height=100)
        screenshot2 = Screenshot(data=data2, width=100, height=100)

        capture = ScreenshotCapture()
        is_same, diff = capture.compare_screenshots(screenshot1, screenshot2)

        # Use == instead of is for numpy bool comparison
        assert bool(is_same) == False
        assert float(diff) > 0.0

    def test_highlight_differences(self):
        """Test highlighting differences between screenshots."""
        from src.computer_use.screenshot import ScreenshotCapture, Screenshot

        data1 = create_test_image(100, 100, "red")
        data2 = create_test_image(100, 100, "blue")
        screenshot1 = Screenshot(data=data1, width=100, height=100)
        screenshot2 = Screenshot(data=data2, width=100, height=100)

        capture = ScreenshotCapture()
        diff_screenshot = capture.highlight_differences(screenshot1, screenshot2)

        assert diff_screenshot.width == 100
        assert diff_screenshot.height == 100
        # Diff image should exist and be a valid PNG
        assert len(diff_screenshot.data) > 0
        assert diff_screenshot.data[:4] == b'\x89PNG'


class TestCaptureMethod:
    """Tests for CaptureMethod enum."""

    def test_capture_methods(self):
        """Test all capture methods are defined."""
        from src.computer_use.screenshot import CaptureMethod

        assert CaptureMethod.PLAYWRIGHT.value == "playwright"
        assert CaptureMethod.XVFB.value == "xvfb"
        assert CaptureMethod.MACOS.value == "macos"
        assert CaptureMethod.DOCKER.value == "docker"
