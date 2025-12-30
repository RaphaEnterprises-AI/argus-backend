"""Screenshot capture utilities for E2E testing."""

import asyncio
import base64
import io
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import structlog
from PIL import Image

logger = structlog.get_logger()


class CaptureMethod(str, Enum):
    """Screenshot capture methods."""
    PLAYWRIGHT = "playwright"
    XVFB = "xvfb"
    MACOS = "macos"
    DOCKER = "docker"


@dataclass
class Screenshot:
    """Captured screenshot with metadata."""
    data: bytes
    width: int
    height: int
    format: str = "png"
    timestamp: float = 0.0

    @property
    def base64(self) -> str:
        """Return base64 encoded screenshot."""
        return base64.standard_b64encode(self.data).decode()

    @property
    def data_uri(self) -> str:
        """Return data URI for embedding."""
        return f"data:image/{self.format};base64,{self.base64}"

    def resize(self, width: int, height: int) -> "Screenshot":
        """Resize screenshot to given dimensions."""
        img = Image.open(io.BytesIO(self.data))
        img = img.resize((width, height), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format=self.format.upper())

        return Screenshot(
            data=buffer.getvalue(),
            width=width,
            height=height,
            format=self.format,
            timestamp=self.timestamp,
        )

    def save(self, path: Path | str) -> None:
        """Save screenshot to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)


class ScreenshotCapture:
    """
    Capture screenshots from various sources.

    Supports:
    - Playwright page screenshots
    - X11 display capture (for Docker/Xvfb)
    - macOS screencapture
    """

    def __init__(
        self,
        default_width: int = 1920,
        default_height: int = 1080,
        optimize: bool = True,
    ):
        self.default_width = default_width
        self.default_height = default_height
        self.optimize = optimize
        self.log = logger.bind(component="screenshot")

    async def capture_playwright(
        self,
        page,
        full_page: bool = False,
        clip: Optional[dict] = None,
    ) -> Screenshot:
        """
        Capture screenshot from Playwright page.

        Args:
            page: Playwright page object
            full_page: Capture entire scrollable page
            clip: Region to capture {"x": 0, "y": 0, "width": 100, "height": 100}

        Returns:
            Screenshot object
        """
        import time

        start = time.time()

        screenshot_bytes = await page.screenshot(
            full_page=full_page,
            clip=clip,
            type="png",
        )

        # Get dimensions
        img = Image.open(io.BytesIO(screenshot_bytes))
        width, height = img.size

        # Optimize if needed
        if self.optimize:
            screenshot_bytes = self._optimize_png(screenshot_bytes)

        self.log.debug(
            "Playwright screenshot captured",
            width=width,
            height=height,
            size_kb=len(screenshot_bytes) / 1024,
            duration_ms=(time.time() - start) * 1000,
        )

        return Screenshot(
            data=screenshot_bytes,
            width=width,
            height=height,
            timestamp=time.time(),
        )

    async def capture_display(
        self,
        display: str = ":99",
        window: str = "root",
    ) -> Screenshot:
        """
        Capture screenshot from X11 display (for Docker/Xvfb).

        Args:
            display: X11 display (e.g., ":99")
            window: Window to capture ("root" for full screen)

        Returns:
            Screenshot object
        """
        import time

        start = time.time()

        # Use ImageMagick import command
        cmd = [
            "import",
            "-window", window,
            "-display", display,
            "png:-"
        ]

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                raise RuntimeError(f"Screenshot failed: {stderr.decode()}")

            screenshot_bytes = stdout

        except FileNotFoundError:
            # Fall back to scrot if available
            self.log.warning("ImageMagick not found, trying scrot")
            screenshot_bytes = await self._capture_with_scrot(display)

        # Get dimensions
        img = Image.open(io.BytesIO(screenshot_bytes))
        width, height = img.size

        self.log.debug(
            "Display screenshot captured",
            display=display,
            width=width,
            height=height,
            size_kb=len(screenshot_bytes) / 1024,
            duration_ms=(time.time() - start) * 1000,
        )

        return Screenshot(
            data=screenshot_bytes,
            width=width,
            height=height,
            timestamp=time.time(),
        )

    async def capture_macos(self) -> Screenshot:
        """
        Capture screenshot on macOS using screencapture.

        Returns:
            Screenshot object
        """
        import time
        import tempfile

        start = time.time()

        # Create temp file for screenshot
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            # Capture screen
            result = await asyncio.create_subprocess_exec(
                "screencapture",
                "-x",  # No sound
                "-C",  # Capture cursor
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()

            if result.returncode != 0:
                raise RuntimeError("screencapture failed")

            # Read the file
            screenshot_bytes = Path(temp_path).read_bytes()

        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)

        # Get dimensions
        img = Image.open(io.BytesIO(screenshot_bytes))
        width, height = img.size

        self.log.debug(
            "macOS screenshot captured",
            width=width,
            height=height,
            size_kb=len(screenshot_bytes) / 1024,
            duration_ms=(time.time() - start) * 1000,
        )

        return Screenshot(
            data=screenshot_bytes,
            width=width,
            height=height,
            timestamp=time.time(),
        )

    async def _capture_with_scrot(self, display: str) -> bytes:
        """Fallback capture using scrot."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            env = {"DISPLAY": display}
            result = await asyncio.create_subprocess_exec(
                "scrot",
                temp_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()

            return Path(temp_path).read_bytes()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _optimize_png(self, data: bytes, quality: int = 85) -> bytes:
        """
        Optimize PNG by reducing color depth if needed.

        For screenshots with gradients, keep as PNG.
        For UI screenshots, can convert to JPEG for size reduction.
        """
        img = Image.open(io.BytesIO(data))

        # Check if image has transparency
        has_alpha = img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        )

        if has_alpha:
            # Keep as PNG for transparency
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()

        # For non-transparent, check if JPEG would be smaller
        png_buffer = io.BytesIO()
        img.save(png_buffer, format="PNG", optimize=True)

        jpg_buffer = io.BytesIO()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(jpg_buffer, format="JPEG", quality=quality, optimize=True)

        # Use smaller format (but keep PNG for now for consistency)
        # return jpg_buffer.getvalue() if len(jpg_buffer.getvalue()) < len(png_buffer.getvalue()) else png_buffer.getvalue()
        return png_buffer.getvalue()

    def compare_screenshots(
        self,
        screenshot1: Screenshot,
        screenshot2: Screenshot,
        threshold: float = 0.05,
    ) -> tuple[bool, float]:
        """
        Compare two screenshots for visual differences.

        Args:
            screenshot1: First screenshot
            screenshot2: Second screenshot
            threshold: Maximum difference ratio to consider "same"

        Returns:
            Tuple of (is_same, difference_ratio)
        """
        img1 = Image.open(io.BytesIO(screenshot1.data))
        img2 = Image.open(io.BytesIO(screenshot2.data))

        # Resize to same dimensions if needed
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

        # Convert to same mode
        if img1.mode != img2.mode:
            img1 = img1.convert("RGB")
            img2 = img2.convert("RGB")

        # Calculate pixel difference
        import numpy as np

        arr1 = np.array(img1)
        arr2 = np.array(img2)

        diff = np.abs(arr1.astype(float) - arr2.astype(float))
        diff_ratio = np.mean(diff) / 255.0

        is_same = diff_ratio <= threshold

        self.log.debug(
            "Screenshot comparison",
            is_same=is_same,
            difference_ratio=diff_ratio,
            threshold=threshold,
        )

        return is_same, diff_ratio

    def highlight_differences(
        self,
        screenshot1: Screenshot,
        screenshot2: Screenshot,
    ) -> Screenshot:
        """
        Create a diff image highlighting differences.

        Args:
            screenshot1: First screenshot (base)
            screenshot2: Second screenshot (compare)

        Returns:
            Screenshot with differences highlighted in red
        """
        import numpy as np

        img1 = Image.open(io.BytesIO(screenshot1.data)).convert("RGB")
        img2 = Image.open(io.BytesIO(screenshot2.data)).convert("RGB")

        # Resize to same dimensions if needed
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Find differences
        diff = np.any(np.abs(arr1.astype(int) - arr2.astype(int)) > 10, axis=2)

        # Create output image
        result = arr2.copy()
        result[diff] = [255, 0, 0]  # Red for differences

        # Convert back to image
        result_img = Image.fromarray(result)

        buffer = io.BytesIO()
        result_img.save(buffer, format="PNG")

        return Screenshot(
            data=buffer.getvalue(),
            width=result_img.width,
            height=result_img.height,
        )
