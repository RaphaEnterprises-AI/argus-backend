"""Shared fixtures for visual_ai tests.

This module provides common fixtures and utilities used across all
visual_ai test modules.
"""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.visual_ai.models import (
    ChangeCategory,
    ChangeIntent,
    Severity,
    VisualChange,
    VisualElement,
    VisualSnapshot,
)


@pytest.fixture
def sample_element():
    """Create a standard VisualElement for testing."""
    return VisualElement(
        element_id="el_test_001",
        selector="#test-element",
        tag_name="div",
        bounds={"x": 100.0, "y": 200.0, "width": 300.0, "height": 150.0},
        computed_styles={
            "display": "block",
            "position": "relative",
            "color": "rgb(0, 0, 0)",
            "background-color": "rgb(255, 255, 255)",
            "font-size": "16px",
            "font-family": "Arial, sans-serif",
        },
        text_content="Test Element Content",
        attributes={"class": "test-class", "data-testid": "test-element"},
        children_count=3,
        screenshot_region=None,
    )


@pytest.fixture
def sample_elements():
    """Create a list of sample elements simulating a page structure."""
    return [
        VisualElement(
            element_id="header",
            selector="header",
            tag_name="header",
            bounds={"x": 0.0, "y": 0.0, "width": 1920.0, "height": 80.0},
            computed_styles={"display": "flex", "background-color": "#ffffff"},
            text_content="Site Header",
            attributes={"role": "banner"},
            children_count=5,
        ),
        VisualElement(
            element_id="nav",
            selector="nav.main-nav",
            tag_name="nav",
            bounds={"x": 0.0, "y": 80.0, "width": 1920.0, "height": 50.0},
            computed_styles={"display": "flex", "background-color": "#f5f5f5"},
            text_content="Navigation",
            attributes={"role": "navigation"},
            children_count=8,
        ),
        VisualElement(
            element_id="main",
            selector="main",
            tag_name="main",
            bounds={"x": 0.0, "y": 130.0, "width": 1920.0, "height": 800.0},
            computed_styles={"display": "block"},
            text_content="Main Content",
            attributes={"role": "main"},
            children_count=20,
        ),
        VisualElement(
            element_id="footer",
            selector="footer",
            tag_name="footer",
            bounds={"x": 0.0, "y": 930.0, "width": 1920.0, "height": 150.0},
            computed_styles={"display": "block", "background-color": "#333333"},
            text_content="Footer Content",
            attributes={"role": "contentinfo"},
            children_count=10,
        ),
    ]


@pytest.fixture
def sample_snapshot(sample_elements):
    """Create a standard VisualSnapshot for testing."""
    return VisualSnapshot(
        id="snapshot_test_001",
        url="https://example.com/test-page",
        viewport={"width": 1920, "height": 1080},
        device_name="Desktop",
        browser="chromium",
        timestamp="2024-01-15T12:00:00Z",
        screenshot=b"\x89PNG\r\n\x1a\ntest_screenshot_data",
        dom_snapshot='{"nodeId": 1, "nodeType": 1, "tagName": "html"}',
        computed_styles={"header": {"display": "flex"}},
        network_har=None,
        elements=sample_elements,
        layout_hash="abc123def456789",
        color_palette=["#ffffff", "#000000", "#f5f5f5", "#333333"],
        text_blocks=[
            {"text": "Site Header", "bounds": {"x": 100, "y": 20}},
            {"text": "Main Content", "bounds": {"x": 100, "y": 200}},
        ],
        largest_contentful_paint=1500.0,
        cumulative_layout_shift=0.05,
        time_to_interactive=3000.0,
    )


@pytest.fixture
def sample_change(sample_element):
    """Create a standard VisualChange for testing."""
    return VisualChange(
        id="change_test_001",
        category=ChangeCategory.LAYOUT,
        intent=ChangeIntent.REGRESSION,
        severity=Severity.MAJOR,
        element=sample_element,
        bounds_baseline={"x": 100.0, "y": 200.0, "width": 300.0, "height": 150.0},
        bounds_current={"x": 110.0, "y": 210.0, "width": 320.0, "height": 160.0},
        property_name="position",
        baseline_value="100, 200",
        current_value="110, 210",
        description="Element position shifted by 10px in both directions",
        root_cause="CSS change in layout.css",
        impact_assessment="May affect page layout and user experience",
        recommendation="Review the CSS changes and ensure intentional",
        confidence=0.95,
        related_commit="abc123",
        related_files=["src/styles/layout.css"],
    )


@pytest.fixture
def mock_playwright_page():
    """Create a mocked Playwright page object."""
    page = AsyncMock()

    # Basic page properties
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Test Page")

    # Navigation and waiting
    page.goto = AsyncMock()
    page.wait_for_load_state = AsyncMock()
    page.wait_for_timeout = AsyncMock()

    # Screenshot
    page.screenshot = AsyncMock(return_value=b"\x89PNG\r\n\x1a\ntest")

    # JavaScript evaluation
    async def evaluate_handler(js_code, *args):
        if "innerWidth" in str(js_code):
            return {"width": 1920, "height": 1080}
        elif "serializeNode" in str(js_code):
            return '{"nodeId": 1}'
        elif "isVisible" in str(js_code) or "maxElements" in str(js_code[:100] if isinstance(js_code, str) else ""):
            return []
        elif "performance" in str(js_code).lower():
            return {"lcp": 1500.0, "cls": 0.05}
        return {}

    page.evaluate = AsyncMock(side_effect=evaluate_handler)

    # Event handlers
    page.on = MagicMock()

    # Element queries
    page.query_selector = AsyncMock(return_value=None)
    page.query_selector_all = AsyncMock(return_value=[])

    return page


@pytest.fixture
def mock_browser():
    """Create a mocked Playwright browser object."""
    browser = AsyncMock()
    browser.close = AsyncMock()
    return browser


@pytest.fixture
def mock_browser_context(mock_playwright_page):
    """Create a mocked Playwright browser context."""
    context = AsyncMock()
    context.new_page = AsyncMock(return_value=mock_playwright_page)
    context.close = AsyncMock()
    return context


@pytest.fixture
def red_image_bytes():
    """Create a red test image as bytes."""
    try:
        from PIL import Image
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError:
        # Fallback: minimal PNG data
        return b"\x89PNG\r\n\x1a\nred_test_image"


@pytest.fixture
def blue_image_bytes():
    """Create a blue test image as bytes."""
    try:
        from PIL import Image
        img = Image.new("RGB", (100, 100), color=(0, 0, 255))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError:
        return b"\x89PNG\r\n\x1a\nblue_test_image"


@pytest.fixture
def gradient_image_bytes():
    """Create a gradient test image as bytes."""
    try:
        from PIL import Image
        img = Image.new("RGB", (100, 100))
        for x in range(100):
            for y in range(100):
                img.putpixel((x, y), (int(x * 2.55), int(y * 2.55), 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError:
        return b"\x89PNG\r\n\x1a\ngradient_test_image"


@pytest.fixture
def viewport_configs():
    """Create common viewport configurations."""
    from src.visual_ai.responsive_analyzer import ViewportConfig

    return [
        ViewportConfig(
            name="Mobile Small",
            width=320,
            height=568,
            device_scale_factor=2.0,
            is_mobile=True,
            has_touch=True,
        ),
        ViewportConfig(
            name="Mobile",
            width=375,
            height=812,
            device_scale_factor=3.0,
            is_mobile=True,
            has_touch=True,
        ),
        ViewportConfig(
            name="Tablet",
            width=768,
            height=1024,
            device_scale_factor=2.0,
            is_mobile=False,
            has_touch=True,
        ),
        ViewportConfig(
            name="Desktop",
            width=1920,
            height=1080,
            device_scale_factor=1.0,
            is_mobile=False,
            has_touch=False,
        ),
    ]


@pytest.fixture
def browser_configs():
    """Create common browser configurations."""
    from src.visual_ai.cross_browser_analyzer import BrowserConfig

    return [
        BrowserConfig(browser="chromium", name="Chrome", version="120.0"),
        BrowserConfig(browser="firefox", name="Firefox", version="121.0"),
        BrowserConfig(browser="webkit", name="Safari", version="17.0"),
    ]


# Utility fixtures for common test scenarios

@pytest.fixture
def mock_anthropic_api():
    """Create a mocked Anthropic API client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text='{"summary": "Test", "changes": [], "overall_intent": "intentional", "risk_level": "minor", "confidence": 0.9, "recommendations": [], "affected_user_flows": [], "breaking_changes": [], "context_notes": ""}')
    ]
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def empty_snapshot():
    """Create a minimal empty snapshot for edge case testing."""
    return VisualSnapshot(
        id="empty_snapshot",
        url="",
        viewport={"width": 0, "height": 0},
        device_name=None,
        browser="",
        timestamp="",
        screenshot=b"",
        dom_snapshot="{}",
        computed_styles={},
        network_har=None,
        elements=[],
        layout_hash="",
        color_palette=[],
        text_blocks=[],
        largest_contentful_paint=None,
        cumulative_layout_shift=None,
        time_to_interactive=None,
    )


# Markers for conditional tests

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "requires_pillow: mark test as requiring PIL/Pillow"
    )
    config.addinivalue_line(
        "markers", "requires_skimage: mark test as requiring scikit-image"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
