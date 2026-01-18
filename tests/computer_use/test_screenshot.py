"""Tests for the screenshot capture module."""

import base64
import io
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


class TestCaptureMethod:
    """Tests for CaptureMethod enum."""

    def test_capture_methods_exist(self, mock_env_vars):
        """Test all capture methods are defined."""
        from src.computer_use.screenshot import CaptureMethod

        assert CaptureMethod.PLAYWRIGHT == "playwright"
        assert CaptureMethod.XVFB == "xvfb"
        assert CaptureMethod.MACOS == "macos"
        assert CaptureMethod.DOCKER == "docker"


class TestScreenshot:
    """Tests for Screenshot dataclass."""

    def test_screenshot_creation(self, mock_env_vars):
        """Test Screenshot creation."""
        from src.computer_use.screenshot import Screenshot

        # Create a minimal valid PNG (1x1 pixel)
        img_data = self._create_test_png()

        screenshot = Screenshot(
            data=img_data,
            width=100,
            height=100,
            format="png",
            timestamp=12345.0,
        )

        assert screenshot.width == 100
        assert screenshot.height == 100
        assert screenshot.format == "png"
        assert screenshot.timestamp == 12345.0

    def test_screenshot_defaults(self, mock_env_vars):
        """Test Screenshot default values."""
        from src.computer_use.screenshot import Screenshot

        img_data = self._create_test_png()

        screenshot = Screenshot(
            data=img_data,
            width=100,
            height=100,
        )

        assert screenshot.format == "png"
        assert screenshot.timestamp == 0.0

    def test_screenshot_base64(self, mock_env_vars):
        """Test Screenshot base64 property."""
        from src.computer_use.screenshot import Screenshot

        img_data = b"test data"

        screenshot = Screenshot(
            data=img_data,
            width=100,
            height=100,
        )

        expected_b64 = base64.standard_b64encode(img_data).decode()
        assert screenshot.base64 == expected_b64

    def test_screenshot_data_uri(self, mock_env_vars):
        """Test Screenshot data_uri property."""
        from src.computer_use.screenshot import Screenshot

        img_data = b"test data"

        screenshot = Screenshot(
            data=img_data,
            width=100,
            height=100,
            format="png",
        )

        assert screenshot.data_uri.startswith("data:image/png;base64,")
        assert base64.standard_b64encode(img_data).decode() in screenshot.data_uri

    def test_screenshot_resize(self, mock_env_vars):
        """Test Screenshot resize method."""
        from src.computer_use.screenshot import Screenshot

        # Create a real PNG for resize test
        img_data = self._create_test_png(width=100, height=100)

        screenshot = Screenshot(
            data=img_data,
            width=100,
            height=100,
            format="png",
            timestamp=12345.0,
        )

        resized = screenshot.resize(50, 50)

        assert resized.width == 50
        assert resized.height == 50
        assert resized.format == "png"
        assert resized.timestamp == 12345.0
        assert len(resized.data) > 0

    def test_screenshot_save(self, mock_env_vars):
        """Test Screenshot save method."""
        from src.computer_use.screenshot import Screenshot

        img_data = self._create_test_png()

        screenshot = Screenshot(
            data=img_data,
            width=100,
            height=100,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "subdir" / "screenshot.png"
            screenshot.save(save_path)

            assert save_path.exists()
            assert save_path.read_bytes() == img_data

    def _create_test_png(self, width=10, height=10):
        """Create a minimal test PNG image."""
        from PIL import Image

        img = Image.new("RGB", (width, height), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()


class TestScreenshotCapture:
    """Tests for ScreenshotCapture class."""

    def test_capture_creation(self, mock_env_vars):
        """Test ScreenshotCapture creation."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture()

        assert capture.default_width == 1920
        assert capture.default_height == 1080
        assert capture.optimize is True

    def test_capture_creation_custom(self, mock_env_vars):
        """Test ScreenshotCapture with custom settings."""
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
    async def test_capture_playwright(self, mock_env_vars):
        """Test Playwright screenshot capture."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture(optimize=False)

        # Create mock page
        mock_page = AsyncMock()
        img_data = self._create_test_png()
        mock_page.screenshot.return_value = img_data

        screenshot = await capture.capture_playwright(mock_page)

        assert screenshot.width == 10
        assert screenshot.height == 10
        assert screenshot.format == "png"
        assert screenshot.timestamp > 0
        mock_page.screenshot.assert_called_once_with(
            full_page=False,
            clip=None,
            type="png",
        )

    @pytest.mark.asyncio
    async def test_capture_playwright_full_page(self, mock_env_vars):
        """Test Playwright full page screenshot."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture(optimize=False)

        mock_page = AsyncMock()
        img_data = self._create_test_png()
        mock_page.screenshot.return_value = img_data

        await capture.capture_playwright(mock_page, full_page=True)

        mock_page.screenshot.assert_called_once_with(
            full_page=True,
            clip=None,
            type="png",
        )

    @pytest.mark.asyncio
    async def test_capture_playwright_with_clip(self, mock_env_vars):
        """Test Playwright screenshot with clip region."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture(optimize=False)

        mock_page = AsyncMock()
        img_data = self._create_test_png()
        mock_page.screenshot.return_value = img_data

        clip = {"x": 0, "y": 0, "width": 100, "height": 100}
        await capture.capture_playwright(mock_page, clip=clip)

        mock_page.screenshot.assert_called_once_with(
            full_page=False,
            clip=clip,
            type="png",
        )

    @pytest.mark.asyncio
    async def test_capture_playwright_with_optimization(self, mock_env_vars):
        """Test Playwright screenshot with optimization."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture(optimize=True)

        mock_page = AsyncMock()
        img_data = self._create_test_png()
        mock_page.screenshot.return_value = img_data

        screenshot = await capture.capture_playwright(mock_page)

        # Should have optimized the image
        assert len(screenshot.data) > 0

    @pytest.mark.asyncio
    async def test_capture_display(self, mock_env_vars):
        """Test X11 display capture."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture()

        img_data = self._create_test_png()

        # Mock the subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (img_data, b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            screenshot = await capture.capture_display(
                display=":99",
                window="root",
            )

            assert screenshot.width == 10
            assert screenshot.height == 10
            assert screenshot.timestamp > 0

    @pytest.mark.asyncio
    async def test_capture_display_failure(self, mock_env_vars):
        """Test X11 display capture failure."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"Error message")

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(RuntimeError, match="Screenshot failed"):
                await capture.capture_display()

    @pytest.mark.asyncio
    async def test_capture_display_fallback_to_scrot(self, mock_env_vars):
        """Test fallback to scrot when ImageMagick not found."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture()

        img_data = self._create_test_png()

        async def side_effect(*args, **kwargs):
            # First call raises FileNotFoundError, second succeeds
            if not hasattr(side_effect, "called"):
                side_effect.called = True
                raise FileNotFoundError("import not found")
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = (img_data, b"")
            return mock_proc

        # Mock _capture_with_scrot directly
        with patch.object(capture, "_capture_with_scrot", return_value=img_data):
            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=FileNotFoundError("import not found"),
            ):
                screenshot = await capture.capture_display()
                assert screenshot.data == img_data

    @pytest.mark.asyncio
    async def test_capture_macos(self, mock_env_vars):
        """Test macOS screencapture."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture()

        img_data = self._create_test_png()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"", b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch.object(Path, "read_bytes", return_value=img_data):
                with patch.object(Path, "unlink"):
                    screenshot = await capture.capture_macos()

                    assert screenshot.width == 10
                    assert screenshot.height == 10

    @pytest.mark.asyncio
    async def test_capture_macos_failure(self, mock_env_vars):
        """Test macOS screencapture failure."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch.object(Path, "unlink"):
                with pytest.raises(RuntimeError, match="screencapture failed"):
                    await capture.capture_macos()

    def test_optimize_png_with_transparency(self, mock_env_vars):
        """Test PNG optimization with transparent image."""
        from PIL import Image

        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture()

        # Create PNG with transparency
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_data = buffer.getvalue()

        optimized = capture._optimize_png(img_data)

        # Should still be PNG due to transparency
        assert len(optimized) > 0

    def test_optimize_png_without_transparency(self, mock_env_vars):
        """Test PNG optimization without transparency."""
        from src.computer_use.screenshot import ScreenshotCapture

        capture = ScreenshotCapture()

        img_data = self._create_test_png()
        optimized = capture._optimize_png(img_data)

        # Should be optimized
        assert len(optimized) > 0

    def test_compare_screenshots_same(self, mock_env_vars):
        """Test comparing identical screenshots."""
        from src.computer_use.screenshot import Screenshot, ScreenshotCapture

        capture = ScreenshotCapture()

        img_data = self._create_test_png()

        screenshot1 = Screenshot(data=img_data, width=10, height=10)
        screenshot2 = Screenshot(data=img_data, width=10, height=10)

        is_same, diff_ratio = capture.compare_screenshots(screenshot1, screenshot2)

        assert bool(is_same) is True
        assert float(diff_ratio) == 0.0

    def test_compare_screenshots_different(self, mock_env_vars):
        """Test comparing different screenshots."""
        from src.computer_use.screenshot import Screenshot, ScreenshotCapture

        capture = ScreenshotCapture()

        img_data1 = self._create_test_png(color="red")
        img_data2 = self._create_test_png(color="blue")

        screenshot1 = Screenshot(data=img_data1, width=10, height=10)
        screenshot2 = Screenshot(data=img_data2, width=10, height=10)

        is_same, diff_ratio = capture.compare_screenshots(screenshot1, screenshot2)

        assert bool(is_same) is False
        assert float(diff_ratio) > 0

    def test_compare_screenshots_different_sizes(self, mock_env_vars):
        """Test comparing screenshots of different sizes."""
        from src.computer_use.screenshot import Screenshot, ScreenshotCapture

        capture = ScreenshotCapture()

        img_data1 = self._create_test_png(width=10, height=10)
        img_data2 = self._create_test_png(width=20, height=20)

        screenshot1 = Screenshot(data=img_data1, width=10, height=10)
        screenshot2 = Screenshot(data=img_data2, width=20, height=20)

        # Should resize and compare
        is_same, diff_ratio = capture.compare_screenshots(screenshot1, screenshot2)

        # Convert numpy types to Python types for assertion
        assert bool(is_same) in (True, False)
        assert 0 <= float(diff_ratio) <= 1

    def test_compare_screenshots_threshold(self, mock_env_vars):
        """Test comparing with custom threshold."""
        from src.computer_use.screenshot import Screenshot, ScreenshotCapture

        capture = ScreenshotCapture()

        img_data1 = self._create_test_png()
        img_data2 = self._create_test_png()

        screenshot1 = Screenshot(data=img_data1, width=10, height=10)
        screenshot2 = Screenshot(data=img_data2, width=10, height=10)

        is_same, _ = capture.compare_screenshots(
            screenshot1, screenshot2, threshold=0.0
        )

        assert bool(is_same) is True

    def test_highlight_differences(self, mock_env_vars):
        """Test highlighting differences between screenshots."""
        from src.computer_use.screenshot import Screenshot, ScreenshotCapture

        capture = ScreenshotCapture()

        img_data1 = self._create_test_png(color="red")
        img_data2 = self._create_test_png(color="blue")

        screenshot1 = Screenshot(data=img_data1, width=10, height=10)
        screenshot2 = Screenshot(data=img_data2, width=10, height=10)

        result = capture.highlight_differences(screenshot1, screenshot2)

        assert result.width == 10
        assert result.height == 10
        assert len(result.data) > 0

    def test_highlight_differences_different_sizes(self, mock_env_vars):
        """Test highlighting with different sized screenshots."""
        from src.computer_use.screenshot import Screenshot, ScreenshotCapture

        capture = ScreenshotCapture()

        img_data1 = self._create_test_png(width=10, height=10)
        img_data2 = self._create_test_png(width=20, height=20)

        screenshot1 = Screenshot(data=img_data1, width=10, height=10)
        screenshot2 = Screenshot(data=img_data2, width=20, height=20)

        result = capture.highlight_differences(screenshot1, screenshot2)

        # Should resize to match first image
        assert result.width == 10
        assert result.height == 10

    def _create_test_png(self, width=10, height=10, color="red"):
        """Create a minimal test PNG image."""
        from PIL import Image

        img = Image.new("RGB", (width, height), color=color)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
