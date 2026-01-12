"""Scientific image comparison algorithms for visual regression testing.

Provides perceptual image analysis using SSIM, perceptual hashing, color detection,
and text rendering comparison. All methods work with images as bytes (PNG format).
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import io
import math
from collections import Counter

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import imagehash
import numpy as np


@dataclass
class ColorChange:
    """Represents a detected color change between images."""
    old_color: str  # hex format, e.g., "#ff0000"
    new_color: str  # hex format
    delta_e: float  # perceptual color difference (0-100+)
    affected_area_percent: float  # percentage of image affected


@dataclass
class TextRenderingDiff:
    """Results of text rendering comparison."""
    font_changed: bool
    size_changed: bool
    antialiasing_different: bool
    affected_regions: List[Dict]


def _bytes_to_image(image_bytes: bytes) -> Image.Image:
    """Convert bytes to PIL Image."""
    return Image.open(io.BytesIO(image_bytes))


def _image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def _ensure_same_size(img1: Image.Image, img2: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Ensure two images have the same dimensions by resizing the second to match the first."""
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    return img1, img2


def _ensure_same_mode(img1: Image.Image, img2: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Ensure both images have the same color mode."""
    target_mode = "RGBA" if img1.mode == "RGBA" or img2.mode == "RGBA" else "RGB"
    if img1.mode != target_mode:
        img1 = img1.convert(target_mode)
    if img2.mode != target_mode:
        img2 = img2.convert(target_mode)
    return img1, img2


def _rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to CIELAB color space for perceptual comparison."""
    # Normalize RGB values
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    # Apply gamma correction
    def gamma_correct(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    r_lin = gamma_correct(r_norm)
    g_lin = gamma_correct(g_norm)
    b_lin = gamma_correct(b_norm)

    # Convert to XYZ
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # Reference white point (D65)
    x_ref, y_ref, z_ref = 0.95047, 1.0, 1.08883

    # Convert to Lab
    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t) + (16/116)

    L = (116 * f(y / y_ref)) - 16
    a = 500 * (f(x / x_ref) - f(y / y_ref))
    b_val = 200 * (f(y / y_ref) - f(z / z_ref))

    return L, a, b_val


def _calculate_delta_e(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    """Calculate Delta E (CIE76) between two LAB colors."""
    return math.sqrt(
        (lab1[0] - lab2[0]) ** 2 +
        (lab1[1] - lab2[1]) ** 2 +
        (lab1[2] - lab2[2]) ** 2
    )


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex string."""
    return f"#{r:02x}{g:02x}{b:02x}"


class PerceptualAnalyzer:
    """Scientific image comparison algorithms.

    Provides methods for structural similarity (SSIM), perceptual hashing,
    color change detection, text rendering analysis, and visual diff generation.

    All methods accept and return images as bytes in PNG format.

    Example:
        analyzer = PerceptualAnalyzer()

        # Compute structural similarity
        ssim_score, heatmap = await analyzer.compute_ssim(baseline, current)
        print(f"SSIM: {ssim_score:.4f}")

        # Compare perceptual hashes
        hash1 = await analyzer.compute_perceptual_hash(baseline)
        hash2 = await analyzer.compute_perceptual_hash(current)
        similarity = await analyzer.compare_hashes(hash1, hash2)
    """

    def __init__(self):
        """Initialize the perceptual analyzer."""
        self._ssim_available = self._check_ssim_availability()

    def _check_ssim_availability(self) -> bool:
        """Check if scikit-image is available for SSIM."""
        try:
            from skimage.metrics import structural_similarity
            return True
        except ImportError:
            return False

    async def compute_ssim(
        self,
        baseline: bytes,
        current: bytes,
        window_size: int = 11
    ) -> Tuple[float, bytes]:
        """
        Compute Structural Similarity Index (SSIM) between two images.

        SSIM measures perceptual similarity based on luminance, contrast, and
        structure. Returns a score from 0 to 1, where 1 means identical images.

        Args:
            baseline: Baseline image as PNG bytes
            current: Current image as PNG bytes
            window_size: Size of the sliding window for comparison (must be odd)

        Returns:
            Tuple of (ssim_score, diff_heatmap_bytes)
            - ssim_score: Float from 0.0 to 1.0 (1.0 = identical)
            - diff_heatmap_bytes: PNG image showing difference heatmap
        """
        img1 = _bytes_to_image(baseline).convert("L")  # Convert to grayscale
        img2 = _bytes_to_image(current).convert("L")

        # Ensure same size
        img1, img2 = _ensure_same_size(img1, img2)

        arr1 = np.array(img1, dtype=np.float64)
        arr2 = np.array(img2, dtype=np.float64)

        if self._ssim_available:
            # Use scikit-image for accurate SSIM
            from skimage.metrics import structural_similarity
            ssim_score, diff_image = structural_similarity(
                arr1, arr2,
                win_size=window_size,
                full=True,
                data_range=255.0
            )
            # Convert diff to heatmap
            diff_normalized = ((1 - diff_image) * 255).astype(np.uint8)
        else:
            # Fallback: simplified SSIM implementation
            ssim_score, diff_normalized = self._compute_ssim_basic(arr1, arr2, window_size)

        # Create colored heatmap
        heatmap = self._create_heatmap_from_diff(diff_normalized)

        return float(ssim_score), _image_to_bytes(heatmap)

    def _compute_ssim_basic(
        self,
        arr1: np.ndarray,
        arr2: np.ndarray,
        window_size: int
    ) -> Tuple[float, np.ndarray]:
        """Basic SSIM implementation when scikit-image is not available."""
        # Constants for numerical stability
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Compute means using box filter (simplified)
        kernel_size = min(window_size, min(arr1.shape[0], arr1.shape[1]) - 1)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel_size = max(3, kernel_size)

        # Create uniform kernel
        from PIL import ImageFilter

        # Convert to PIL for filtering
        img1_pil = Image.fromarray(arr1.astype(np.uint8))
        img2_pil = Image.fromarray(arr2.astype(np.uint8))

        # Apply box blur for local means
        mu1 = np.array(img1_pil.filter(ImageFilter.BoxBlur(kernel_size // 2)), dtype=np.float64)
        mu2 = np.array(img2_pil.filter(ImageFilter.BoxBlur(kernel_size // 2)), dtype=np.float64)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = np.array(
            Image.fromarray((arr1 ** 2).astype(np.uint8)).filter(
                ImageFilter.BoxBlur(kernel_size // 2)
            ),
            dtype=np.float64
        ) - mu1_sq
        sigma1_sq = np.maximum(sigma1_sq, 0)

        sigma2_sq = np.array(
            Image.fromarray((arr2 ** 2).astype(np.uint8)).filter(
                ImageFilter.BoxBlur(kernel_size // 2)
            ),
            dtype=np.float64
        ) - mu2_sq
        sigma2_sq = np.maximum(sigma2_sq, 0)

        sigma12 = np.array(
            Image.fromarray((arr1 * arr2).astype(np.uint8)).filter(
                ImageFilter.BoxBlur(kernel_size // 2)
            ),
            dtype=np.float64
        ) - mu1_mu2

        # SSIM formula
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / (denominator + 1e-10)
        ssim_score = float(np.mean(ssim_map))

        # Create diff image
        diff = ((1 - np.clip(ssim_map, 0, 1)) * 255).astype(np.uint8)

        return ssim_score, diff

    def _create_heatmap_from_diff(self, diff: np.ndarray) -> Image.Image:
        """Create a colored heatmap from a grayscale difference array."""
        # Colormap: blue (low diff) -> green -> yellow -> red (high diff)
        height, width = diff.shape
        heatmap = Image.new("RGB", (width, height))
        pixels = heatmap.load()

        for y in range(height):
            for x in range(width):
                value = diff[y, x] / 255.0
                # Simple blue-to-red gradient
                if value < 0.5:
                    r = int(value * 2 * 255)
                    g = int(value * 2 * 255)
                    b = int((1 - value * 2) * 255)
                else:
                    r = 255
                    g = int((1 - (value - 0.5) * 2) * 255)
                    b = 0
                pixels[x, y] = (r, g, b)

        return heatmap

    async def compute_perceptual_hash(self, image: bytes) -> str:
        """
        Compute perceptual hash for fuzzy image matching.

        Perceptual hashing creates a fingerprint that is robust to minor
        variations like compression artifacts, small resizes, or color shifts.

        Args:
            image: Image as PNG bytes

        Returns:
            Hex string of the perceptual hash
        """
        img = _bytes_to_image(image)
        # Use average hash combined with difference hash for better accuracy
        ahash = imagehash.average_hash(img, hash_size=16)
        dhash = imagehash.dhash(img, hash_size=16)
        phash = imagehash.phash(img, hash_size=16)

        # Combine hashes for more robust fingerprint
        combined = f"{ahash}:{dhash}:{phash}"
        return combined

    async def compare_hashes(self, hash1: str, hash2: str) -> float:
        """
        Compare two perceptual hashes and return similarity score.

        Args:
            hash1: First perceptual hash (from compute_perceptual_hash)
            hash2: Second perceptual hash

        Returns:
            Similarity score from 0.0 to 1.0 (1.0 = identical)
        """
        parts1 = hash1.split(":")
        parts2 = hash2.split(":")

        if len(parts1) != len(parts2):
            # Fall back to simple comparison if formats don't match
            h1 = imagehash.hex_to_hash(parts1[0])
            h2 = imagehash.hex_to_hash(parts2[0])
            max_diff = len(h1.hash) ** 2
            diff = h1 - h2
            return 1.0 - (diff / max_diff)

        total_similarity = 0.0
        for p1, p2 in zip(parts1, parts2):
            h1 = imagehash.hex_to_hash(p1)
            h2 = imagehash.hex_to_hash(p2)
            max_diff = len(h1.hash) ** 2
            diff = h1 - h2
            total_similarity += 1.0 - (diff / max_diff)

        return total_similarity / len(parts1)

    async def detect_color_changes(
        self,
        baseline: bytes,
        current: bytes,
        threshold: float = 0.05
    ) -> List[ColorChange]:
        """
        Detect significant color palette changes between images.

        Analyzes the dominant colors in both images and identifies changes
        using Delta E (perceptual color difference).

        Args:
            baseline: Baseline image as PNG bytes
            current: Current image as PNG bytes
            threshold: Minimum area percentage to consider (0.0-1.0)

        Returns:
            List of ColorChange objects for significant changes
        """
        img1 = _bytes_to_image(baseline).convert("RGB")
        img2 = _bytes_to_image(current).convert("RGB")

        img1, img2 = _ensure_same_size(img1, img2)

        # Quantize colors for analysis (reduce to palette)
        img1_quantized = img1.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
        img2_quantized = img2.quantize(colors=64, method=Image.Quantize.MEDIANCUT)

        # Get color histograms
        def get_color_distribution(img_quantized: Image.Image, original: Image.Image) -> Dict[str, float]:
            """Get color distribution as hex:percentage dict."""
            palette = img_quantized.getpalette()
            if not palette:
                return {}

            # Convert back to RGB to count pixels
            rgb_img = img_quantized.convert("RGB")
            pixels = list(rgb_img.getdata())
            total_pixels = len(pixels)

            # Count colors
            color_counts = Counter(pixels)

            # Convert to hex and percentage
            distribution = {}
            for color, count in color_counts.most_common(20):  # Top 20 colors
                hex_color = _rgb_to_hex(*color)
                percentage = count / total_pixels
                if percentage >= threshold:
                    distribution[hex_color] = percentage

            return distribution

        colors1 = get_color_distribution(img1_quantized, img1)
        colors2 = get_color_distribution(img2_quantized, img2)

        changes = []

        # Find color changes
        all_colors = set(colors1.keys()) | set(colors2.keys())

        for color in all_colors:
            area1 = colors1.get(color, 0)
            area2 = colors2.get(color, 0)

            if color in colors1 and color not in colors2:
                # Color disappeared - find closest replacement
                closest_new = self._find_closest_color(color, list(colors2.keys()))
                if closest_new:
                    rgb1 = _hex_to_rgb(color)
                    rgb2 = _hex_to_rgb(closest_new)
                    lab1 = _rgb_to_lab(*rgb1)
                    lab2 = _rgb_to_lab(*rgb2)
                    delta_e = _calculate_delta_e(lab1, lab2)

                    if delta_e > 2.0:  # Noticeable difference
                        changes.append(ColorChange(
                            old_color=color,
                            new_color=closest_new,
                            delta_e=delta_e,
                            affected_area_percent=area1 * 100
                        ))

            elif color not in colors1 and color in colors2:
                # New color appeared - find what it replaced
                closest_old = self._find_closest_color(color, list(colors1.keys()))
                if closest_old:
                    rgb1 = _hex_to_rgb(closest_old)
                    rgb2 = _hex_to_rgb(color)
                    lab1 = _rgb_to_lab(*rgb1)
                    lab2 = _rgb_to_lab(*rgb2)
                    delta_e = _calculate_delta_e(lab1, lab2)

                    if delta_e > 2.0:
                        changes.append(ColorChange(
                            old_color=closest_old,
                            new_color=color,
                            delta_e=delta_e,
                            affected_area_percent=area2 * 100
                        ))

        # Deduplicate and sort by affected area
        seen = set()
        unique_changes = []
        for change in sorted(changes, key=lambda c: c.affected_area_percent, reverse=True):
            key = (change.old_color, change.new_color)
            if key not in seen:
                seen.add(key)
                unique_changes.append(change)

        return unique_changes

    def _find_closest_color(self, target_hex: str, candidates: List[str]) -> Optional[str]:
        """Find the closest color to target from candidates."""
        if not candidates:
            return None

        target_rgb = _hex_to_rgb(target_hex)
        target_lab = _rgb_to_lab(*target_rgb)

        best_match = None
        best_distance = float("inf")

        for candidate in candidates:
            candidate_rgb = _hex_to_rgb(candidate)
            candidate_lab = _rgb_to_lab(*candidate_rgb)
            distance = _calculate_delta_e(target_lab, candidate_lab)

            if distance < best_distance:
                best_distance = distance
                best_match = candidate

        return best_match

    async def analyze_text_rendering(
        self,
        baseline: bytes,
        current: bytes
    ) -> TextRenderingDiff:
        """
        Compare text rendering between images (font, size, antialiasing).

        Uses edge detection and frequency analysis to detect changes in
        text rendering characteristics.

        Args:
            baseline: Baseline image as PNG bytes
            current: Current image as PNG bytes

        Returns:
            TextRenderingDiff with detected differences
        """
        img1 = _bytes_to_image(baseline).convert("L")
        img2 = _bytes_to_image(current).convert("L")

        img1, img2 = _ensure_same_size(img1, img2)

        # Apply edge detection to highlight text
        edges1 = img1.filter(ImageFilter.FIND_EDGES)
        edges2 = img2.filter(ImageFilter.FIND_EDGES)

        arr1 = np.array(edges1)
        arr2 = np.array(edges2)

        # Analyze edge patterns for font changes
        # Different fonts have different edge distributions
        hist1, _ = np.histogram(arr1.flatten(), bins=256, range=(0, 256))
        hist2, _ = np.histogram(arr2.flatten(), bins=256, range=(0, 256))

        # Normalize histograms
        hist1 = hist1.astype(np.float64) / (hist1.sum() + 1e-10)
        hist2 = hist2.astype(np.float64) / (hist2.sum() + 1e-10)

        # Chi-square distance for font change detection
        chi_sq = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))
        font_changed = chi_sq > 0.1

        # Analyze edge density for size changes
        density1 = np.mean(arr1 > 30)
        density2 = np.mean(arr2 > 30)
        size_changed = abs(density1 - density2) > 0.05

        # Analyze antialiasing by looking at intermediate values
        def count_antialiased(arr):
            # Antialiased text has many intermediate values (not just 0 and 255)
            intermediate = ((arr > 20) & (arr < 235)).sum()
            total = (arr > 20).sum()
            return intermediate / (total + 1e-10)

        aa1 = count_antialiased(arr1)
        aa2 = count_antialiased(arr2)
        antialiasing_different = abs(aa1 - aa2) > 0.1

        # Find affected regions using block comparison
        affected_regions = []
        block_size = 64

        for y in range(0, arr1.shape[0] - block_size, block_size):
            for x in range(0, arr1.shape[1] - block_size, block_size):
                block1 = arr1[y:y+block_size, x:x+block_size]
                block2 = arr2[y:y+block_size, x:x+block_size]

                diff = np.mean(np.abs(block1.astype(float) - block2.astype(float)))
                if diff > 20:  # Significant difference
                    affected_regions.append({
                        "x": int(x),
                        "y": int(y),
                        "width": block_size,
                        "height": block_size,
                        "difference_score": float(diff)
                    })

        return TextRenderingDiff(
            font_changed=font_changed,
            size_changed=size_changed,
            antialiasing_different=antialiasing_different,
            affected_regions=affected_regions
        )

    async def generate_diff_heatmap(
        self,
        baseline: bytes,
        current: bytes
    ) -> bytes:
        """
        Generate a visual diff heatmap highlighting differences.

        Args:
            baseline: Baseline image as PNG bytes
            current: Current image as PNG bytes

        Returns:
            PNG bytes of the heatmap image
        """
        img1 = _bytes_to_image(baseline).convert("RGB")
        img2 = _bytes_to_image(current).convert("RGB")

        img1, img2 = _ensure_same_size(img1, img2)

        arr1 = np.array(img1, dtype=np.float64)
        arr2 = np.array(img2, dtype=np.float64)

        # Compute per-pixel difference
        diff = np.sqrt(np.sum((arr1 - arr2) ** 2, axis=2))

        # Normalize to 0-255
        max_diff = np.sqrt(3 * 255 ** 2)  # Maximum possible difference
        diff_normalized = (diff / max_diff * 255).astype(np.uint8)

        # Create colored heatmap
        heatmap = self._create_heatmap_from_diff(diff_normalized)

        return _image_to_bytes(heatmap)

    async def create_side_by_side(
        self,
        baseline: bytes,
        current: bytes,
        highlight_regions: Optional[List[Dict]] = None
    ) -> bytes:
        """
        Create a side-by-side comparison image.

        Args:
            baseline: Baseline image as PNG bytes
            current: Current image as PNG bytes
            highlight_regions: Optional list of regions to highlight
                Each dict should have: x, y, width, height, color (optional)

        Returns:
            PNG bytes of the side-by-side comparison
        """
        img1 = _bytes_to_image(baseline).convert("RGBA")
        img2 = _bytes_to_image(current).convert("RGBA")

        # Make images same size
        img1, img2 = _ensure_same_size(img1, img2)

        width1, height1 = img1.size
        width2, height2 = img2.size

        # Create combined image with label space
        label_height = 30
        gap = 10
        combined = Image.new(
            "RGBA",
            (width1 + gap + width2, max(height1, height2) + label_height),
            (255, 255, 255, 255)
        )

        # Paste images
        combined.paste(img1, (0, label_height))
        combined.paste(img2, (width1 + gap, label_height))

        # Add labels
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except (OSError, IOError):
                font = ImageFont.load_default()

        draw.text((10, 5), "Baseline", fill=(0, 0, 0), font=font)
        draw.text((width1 + gap + 10, 5), "Current", fill=(0, 0, 0), font=font)

        # Draw divider
        draw.line(
            [(width1 + gap // 2, 0), (width1 + gap // 2, combined.height)],
            fill=(128, 128, 128),
            width=2
        )

        # Highlight regions if provided
        if highlight_regions:
            for region in highlight_regions:
                x = region.get("x", 0)
                y = region.get("y", 0) + label_height
                w = region.get("width", 50)
                h = region.get("height", 50)
                color = region.get("color", (255, 0, 0, 128))

                if isinstance(color, str):
                    rgb = _hex_to_rgb(color)
                    color = (*rgb, 128)

                # Draw on both images
                for offset in [0, width1 + gap]:
                    # Create semi-transparent rectangle
                    overlay = Image.new("RGBA", (w, h), color)
                    combined.paste(overlay, (x + offset, y), overlay)

                    # Draw border
                    draw.rectangle(
                        [x + offset, y, x + offset + w, y + h],
                        outline=(255, 0, 0),
                        width=2
                    )

        return _image_to_bytes(combined.convert("RGB"))

    async def create_animated_diff(
        self,
        baseline: bytes,
        current: bytes,
        duration: int = 500
    ) -> bytes:
        """
        Create an animated GIF toggling between baseline and current images.

        Args:
            baseline: Baseline image as PNG bytes
            current: Current image as PNG bytes
            duration: Duration of each frame in milliseconds

        Returns:
            Bytes of the animated GIF
        """
        img1 = _bytes_to_image(baseline).convert("RGBA")
        img2 = _bytes_to_image(current).convert("RGBA")

        img1, img2 = _ensure_same_size(img1, img2)

        # Add labels
        label_height = 30

        def add_label(img: Image.Image, label: str) -> Image.Image:
            new_img = Image.new("RGBA", (img.width, img.height + label_height), (255, 255, 255, 255))
            new_img.paste(img, (0, label_height))

            draw = ImageDraw.Draw(new_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                except (OSError, IOError):
                    font = ImageFont.load_default()

            draw.text((10, 5), label, fill=(0, 0, 0), font=font)
            return new_img

        frame1 = add_label(img1, "Baseline")
        frame2 = add_label(img2, "Current")

        # Convert to P mode for GIF (required for animated GIF)
        frame1_p = frame1.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        frame2_p = frame2.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)

        # Create animated GIF
        buffer = io.BytesIO()
        frame1_p.save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=[frame2_p],
            duration=duration,
            loop=0  # Loop forever
        )

        return buffer.getvalue()
