"""Enhanced visual state capture module.

Captures complete visual snapshots with DOM state, computed styles, layout information,
network HAR, and performance metrics for comprehensive visual analysis.

This module provides:
- Screenshot capture with configurable viewport and format
- Full DOM tree serialization with element IDs
- Computed CSS extraction for all visible elements
- Network HAR capture for request/response analysis
- Performance metrics (LCP, CLS, TTI) from Performance API
- Color palette extraction from screenshots
- Layout fingerprinting for structural comparison
"""

import base64
import hashlib
import json
import time
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog

try:
    from playwright.async_api import Page, BrowserContext, Request, Response, async_playwright
except ImportError:
    # Type hints only - actual import handled at runtime
    Page = Any
    BrowserContext = Any
    Request = Any
    Response = Any

from .models import VisualSnapshot, VisualElement

logger = structlog.get_logger(__name__)


# JavaScript code for full DOM tree serialization with element IDs
DOM_SERIALIZER_JS = """
() => {
    const serializeNode = (node, nodeIdCounter) => {
        const result = {
            nodeId: nodeIdCounter.id++,
            nodeType: node.nodeType
        };

        if (node.nodeType === Node.ELEMENT_NODE) {
            result.tagName = node.tagName.toLowerCase();

            // Get attributes
            const attrs = {};
            for (const attr of node.attributes || []) {
                attrs[attr.name] = attr.value;
            }
            if (Object.keys(attrs).length > 0) {
                result.attributes = attrs;
            }

            // Get children (skip script/style contents)
            const children = [];
            for (const child of node.childNodes) {
                if (node.tagName === 'SCRIPT' || node.tagName === 'STYLE') {
                    continue;
                }
                const serialized = serializeNode(child, nodeIdCounter);
                if (serialized) {
                    children.push(serialized);
                }
            }
            if (children.length > 0) {
                result.children = children;
            }

            // Handle shadow DOM
            if (node.shadowRoot) {
                result.isShadowRoot = true;
                const shadowChildren = [];
                for (const child of node.shadowRoot.childNodes) {
                    const serialized = serializeNode(child, nodeIdCounter);
                    if (serialized) {
                        shadowChildren.push(serialized);
                    }
                }
                if (shadowChildren.length > 0) {
                    result.shadowChildren = shadowChildren;
                }
            }
        } else if (node.nodeType === Node.TEXT_NODE) {
            const text = node.textContent.trim();
            if (text) {
                result.textContent = text.substring(0, 1000);
            } else {
                return null;  // Skip empty text nodes
            }
        } else if (node.nodeType === Node.COMMENT_NODE) {
            return null;  // Skip comments
        }

        return result;
    };

    const nodeIdCounter = { id: 0 };
    return JSON.stringify(serializeNode(document.documentElement, nodeIdCounter));
}
"""

# JavaScript code for extracting all visible elements with full computed styles
ELEMENT_EXTRACTOR_JS = """
(maxElements) => {
    const elements = [];
    const interactiveTags = new Set(['a', 'button', 'input', 'select', 'textarea', 'details', 'summary']);
    const focusableTags = new Set(['a', 'button', 'input', 'select', 'textarea', 'iframe', 'area']);

    const isVisible = (el) => {
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
            return false;
        }
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    };

    const isInViewport = (rect, viewport) => {
        return (
            rect.top < viewport.height &&
            rect.bottom > 0 &&
            rect.left < viewport.width &&
            rect.right > 0
        );
    };

    const getElementType = (el) => {
        const tag = el.tagName.toLowerCase();
        const role = el.getAttribute('role');

        if (tag === 'button' || role === 'button') return 'button';
        if (tag === 'a' || role === 'link') return 'link';
        if (tag === 'input' || tag === 'textarea' || tag === 'select') return 'input';
        if (tag === 'img' || role === 'img') return 'image';
        if (tag === 'nav' || role === 'navigation') return 'navigation';
        if (['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tag) || role === 'heading') return 'heading';
        if (tag === 'ul' || tag === 'ol' || role === 'list') return 'list';
        if (tag === 'table' || role === 'table') return 'table';
        if (tag === 'form' || role === 'form') return 'form';
        if (tag === 'iframe') return 'iframe';
        if (tag === 'video') return 'video';
        if (tag === 'canvas') return 'canvas';
        if (tag === 'svg') return 'svg';
        if (tag === 'div' || tag === 'section' || tag === 'article' || tag === 'main') return 'container';
        if (tag === 'p' || tag === 'span' || tag === 'label') return 'text';
        return 'other';
    };

    const getCssSelector = (el) => {
        if (el.id) {
            return `#${CSS.escape(el.id)}`;
        }

        const parts = [];
        let current = el;

        while (current && current.tagName) {
            let selector = current.tagName.toLowerCase();

            if (current.id) {
                parts.unshift(`#${CSS.escape(current.id)}`);
                break;
            }

            const classes = Array.from(current.classList || [])
                .filter(c => c && !c.match(/^(ng-|v-|react-|css-)/))
                .slice(0, 2);

            if (classes.length > 0) {
                selector += '.' + classes.map(c => CSS.escape(c)).join('.');
            }

            const parent = current.parentElement;
            if (parent) {
                const siblings = Array.from(parent.children).filter(
                    child => child.tagName === current.tagName
                );
                if (siblings.length > 1) {
                    const index = siblings.indexOf(current) + 1;
                    selector += `:nth-child(${index})`;
                }
            }

            parts.unshift(selector);
            current = parent;

            if (parts.length >= 5) break;
        }

        return parts.join(' > ');
    };

    const getXPath = (el) => {
        if (el.id) {
            return `//*[@id="${el.id}"]`;
        }

        const parts = [];
        let current = el;

        while (current && current.nodeType === Node.ELEMENT_NODE) {
            let tag = current.tagName.toLowerCase();

            if (current.id) {
                parts.unshift(`*[@id="${current.id}"]`);
                break;
            }

            const parent = current.parentElement;
            if (parent) {
                const siblings = Array.from(parent.children).filter(
                    child => child.tagName === current.tagName
                );
                if (siblings.length > 1) {
                    const index = siblings.indexOf(current) + 1;
                    tag += `[${index}]`;
                }
            }

            parts.unshift(tag);
            current = parent;
        }

        return '/' + parts.join('/');
    };

    const getComputedStylesSubset = (el) => {
        const style = window.getComputedStyle(el);
        return {
            display: style.display,
            position: style.position,
            float: style.cssFloat,
            visibility: style.visibility,
            opacity: style.opacity,
            overflow: style.overflow,
            width: style.width,
            height: style.height,
            margin: style.margin,
            padding: style.padding,
            border: style.border,
            'box-sizing': style.boxSizing,
            'background-color': style.backgroundColor,
            'background-image': style.backgroundImage,
            'background-position': style.backgroundPosition,
            'background-size': style.backgroundSize,
            color: style.color,
            'font-family': style.fontFamily,
            'font-size': style.fontSize,
            'font-weight': style.fontWeight,
            'line-height': style.lineHeight,
            'text-align': style.textAlign,
            'text-decoration': style.textDecoration,
            transform: style.transform,
            'z-index': style.zIndex,
            'border-radius': style.borderRadius,
            'box-shadow': style.boxShadow,
            cursor: style.cursor
        };
    };

    const viewport = {
        width: window.innerWidth,
        height: window.innerHeight
    };

    // Walk all elements using TreeWalker for better performance
    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_ELEMENT,
        null,
        false
    );

    let elementCounter = 0;
    let node;

    while ((node = walker.nextNode()) && elementCounter < maxElements) {
        if (!isVisible(node)) continue;

        const tag = node.tagName.toLowerCase();
        if (['script', 'style', 'noscript', 'head', 'meta', 'link'].includes(tag)) {
            continue;
        }

        const rect = node.getBoundingClientRect();

        // Get attributes (skip event handlers)
        const attrs = {};
        for (const attr of node.attributes || []) {
            if (!attr.name.startsWith('on')) {
                attrs[attr.name] = attr.value.substring(0, 500);
            }
        }

        elements.push({
            element_id: `el_${elementCounter++}`,
            selector: getCssSelector(node),
            tag_name: tag,
            bounds: {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            },
            computed_styles: getComputedStylesSubset(node),
            text_content: (node.textContent || '').trim().substring(0, 500),
            attributes: attrs,
            children_count: node.children.length,
            is_visible: true,
            is_in_viewport: isInViewport(rect, viewport),
            is_interactive: interactiveTags.has(tag) ||
                           attrs.role === 'button' ||
                           attrs.onclick !== undefined ||
                           attrs.tabindex !== undefined,
            is_focusable: focusableTags.has(tag) || attrs.tabindex !== undefined,
            is_editable: tag === 'input' || tag === 'textarea' || attrs.contenteditable === 'true',
            element_type: getElementType(node),
            xpath: getXPath(node),
            aria_label: attrs['aria-label'] || null,
            test_id: attrs['data-testid'] || attrs['data-test-id'] || attrs['data-cy'] || null
        });
    }

    return elements;
}
"""

# JavaScript to set up performance observers before page load
SETUP_PERFORMANCE_OBSERVERS_JS = """
() => {
    // LCP Observer
    if (window.PerformanceObserver) {
        try {
            const lcpObserver = new PerformanceObserver((entryList) => {
                const entries = entryList.getEntries();
                const lastEntry = entries[entries.length - 1];
                window.__lcp_value = lastEntry.startTime;
                if (lastEntry.element) {
                    window.__lcp_element = lastEntry.element.tagName;
                }
                if (lastEntry.url) {
                    window.__lcp_url = lastEntry.url;
                }
            });
            lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true });
        } catch (e) {}

        // CLS Observer
        try {
            window.__cls_value = 0;
            const clsObserver = new PerformanceObserver((entryList) => {
                for (const entry of entryList.getEntries()) {
                    if (!entry.hadRecentInput) {
                        window.__cls_value += entry.value;
                    }
                }
            });
            clsObserver.observe({ type: 'layout-shift', buffered: true });
        } catch (e) {}

        // FID/INP Observer
        try {
            const fidObserver = new PerformanceObserver((entryList) => {
                const entries = entryList.getEntries();
                if (entries.length > 0) {
                    window.__fid_value = entries[0].processingStart - entries[0].startTime;
                }
            });
            fidObserver.observe({ type: 'first-input', buffered: true });
        } catch (e) {}
    }
    return true;
}
"""

# JavaScript code for extracting performance metrics
PERFORMANCE_METRICS_JS = """
() => {
    const metrics = {};

    // Navigation timing
    const navEntries = performance.getEntriesByType('navigation');
    if (navEntries.length > 0) {
        const nav = navEntries[0];
        metrics.ttfb = nav.responseStart - nav.requestStart;
        metrics.domContentLoaded = nav.domContentLoadedEventEnd - nav.startTime;
        metrics.load = nav.loadEventEnd - nav.startTime;
        metrics.domInteractive = nav.domInteractive - nav.startTime;
    }

    // Paint timing
    const paintEntries = performance.getEntriesByType('paint');
    for (const entry of paintEntries) {
        if (entry.name === 'first-paint') {
            metrics.fp = entry.startTime;
        }
        if (entry.name === 'first-contentful-paint') {
            metrics.fcp = entry.startTime;
        }
    }

    // LCP from PerformanceObserver
    if (window.__lcp_value !== undefined) {
        metrics.lcp = window.__lcp_value;
        metrics.lcp_element = window.__lcp_element || null;
        metrics.lcp_url = window.__lcp_url || null;
    }

    // CLS from stored value
    if (window.__cls_value !== undefined) {
        metrics.cls = window.__cls_value;
    }

    // FID from stored value
    if (window.__fid_value !== undefined) {
        metrics.fid = window.__fid_value;
    }

    // Resource timing summary
    const resources = performance.getEntriesByType('resource');
    let totalTransferSize = 0;
    let totalResourceTime = 0;
    const resourcesByType = {};

    for (const resource of resources) {
        totalTransferSize += resource.transferSize || 0;
        totalResourceTime += resource.duration || 0;

        const type = resource.initiatorType || 'other';
        resourcesByType[type] = (resourcesByType[type] || 0) + 1;
    }

    metrics.totalResources = resources.length;
    metrics.totalTransferSize = totalTransferSize;
    metrics.totalResourceTime = totalResourceTime;
    metrics.resourcesByType = resourcesByType;

    // Memory (Chrome only)
    if (performance.memory) {
        metrics.jsHeapSize = performance.memory.totalJSHeapSize;
        metrics.jsHeapUsed = performance.memory.usedJSHeapSize;
        metrics.jsHeapLimit = performance.memory.jsHeapSizeLimit;
    }

    // Long tasks (if available)
    try {
        const longTasks = performance.getEntriesByType('longtask');
        if (longTasks.length > 0) {
            metrics.longTaskCount = longTasks.length;
            metrics.totalLongTaskTime = longTasks.reduce((sum, task) => sum + task.duration, 0);
        }
    } catch (e) {}

    return metrics;
}
"""


class EnhancedCapture:
    """Captures complete visual state of a page.

    This class provides comprehensive visual state capture using Playwright,
    including screenshots, DOM snapshots, computed styles, network activity,
    and performance metrics.

    Features:
    - Full page or viewport screenshots
    - Complete DOM tree serialization with element IDs
    - Computed CSS extraction for all visible elements
    - Network HAR capture for request/response analysis
    - Performance metrics (LCP, CLS, TTI, FCP, TTFB) from Performance API
    - Color palette extraction from screenshots
    - Layout fingerprinting for structural comparison
    - Text block identification with typography info

    Usage:
        capture = EnhancedCapture()

        # Use with existing page
        snapshot = await capture.capture_snapshot(
            page=page,
            url="https://example.com",
            viewport={"width": 1920, "height": 1080},
            browser="chromium"
        )

        # Use with fresh browser context
        snapshot = await capture.capture_url(
            "https://example.com",
            browser_type="chromium",
            viewport={"width": 1920, "height": 1080}
        )

    Attributes:
        default_viewport: Default viewport dimensions
        capture_performance_enabled: Whether to capture performance metrics
        capture_network_enabled: Whether to capture network HAR
        max_elements: Maximum number of elements to extract
        log: Structured logger instance
    """

    def __init__(
        self,
        default_viewport: Optional[Dict[str, int]] = None,
        capture_performance: bool = True,
        capture_network: bool = True,
        max_elements: int = 1000,
    ):
        """Initialize EnhancedCapture.

        Args:
            default_viewport: Default viewport size {"width": int, "height": int}
            capture_performance: Whether to capture Core Web Vitals metrics
            capture_network: Whether to capture network HAR data
            max_elements: Maximum elements to extract (prevents memory issues)
        """
        self.default_viewport = default_viewport or {"width": 1920, "height": 1080}
        self.capture_performance_enabled = capture_performance
        self.capture_network_enabled = capture_network
        self.max_elements = max_elements
        self.log = logger.bind(component="enhanced_capture")

        # Network capture state
        self._network_entries: List[Dict[str, Any]] = []
        self._har_entries: List[Dict[str, Any]] = []
        self._request_start_times: Dict[str, float] = {}

    async def capture_snapshot(
        self,
        page: Page,
        url: str,
        viewport: Dict[str, int],
        browser: str,
        full_page: bool = False,
        device_name: Optional[str] = None,
        capture_network: Optional[bool] = None,
        capture_performance: Optional[bool] = None,
        extract_colors: bool = True,
        timeout_ms: int = 30000,
    ) -> VisualSnapshot:
        """Capture full visual snapshot with all metadata.

        Args:
            page: Playwright page instance
            url: The URL being captured
            viewport: Viewport dimensions {"width": int, "height": int}
            browser: Browser name (chromium, firefox, webkit)
            full_page: Whether to capture full scrollable page
            device_name: Device name for metadata
            capture_network: Whether to capture network HAR (overrides default)
            capture_performance: Whether to capture performance metrics (overrides default)
            extract_colors: Whether to extract color palette from screenshot
            timeout_ms: Maximum time to wait for capture operations

        Returns:
            VisualSnapshot containing all captured data

        Raises:
            Exception: If critical capture operations fail
        """
        start_time = time.time()
        snapshot_id = str(uuid.uuid4())

        # Use instance defaults if not overridden
        should_capture_network = capture_network if capture_network is not None else self.capture_network_enabled
        should_capture_performance = capture_performance if capture_performance is not None else self.capture_performance_enabled

        self.log.info(
            "Starting snapshot capture",
            snapshot_id=snapshot_id,
            url=url,
            viewport=viewport,
            browser=browser,
            capture_network=should_capture_network,
            capture_performance=should_capture_performance,
        )

        # Reset network capture state
        self._network_entries = []
        self._har_entries = []
        self._request_start_times = {}

        # Set up network capture if requested
        if should_capture_network:
            await self._setup_network_capture(page)

        # Set up performance observers
        if should_capture_performance:
            try:
                await page.evaluate(SETUP_PERFORMANCE_OBSERVERS_JS)
            except Exception as e:
                self.log.warning("Performance observer setup failed", error=str(e))

        # Wait for network idle and metrics to collect
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception as e:
            self.log.warning("Network idle timeout", error=str(e))

        # Small delay for metrics to stabilize
        await page.wait_for_timeout(300)

        # Capture screenshot
        screenshot = await page.screenshot(full_page=full_page, type="png")

        # Get current URL (may have redirected)
        current_url = page.url

        # Get page title
        try:
            title = await page.title()
        except Exception:
            title = ""

        # Get actual viewport size
        actual_viewport = await page.evaluate(
            """() => ({
                width: window.innerWidth,
                height: window.innerHeight
            })"""
        )

        # Capture full DOM snapshot with element IDs
        dom_snapshot = await self.capture_dom_snapshot(page)

        # Extract all visible elements with bounds and computed styles
        elements_data = await self.extract_elements(page)

        # Compute layout hash for structural comparison
        layout_hash = await self.compute_layout_hash(elements_data)

        # Extract color palette from screenshot
        color_palette: List[str] = []
        if extract_colors and screenshot:
            color_palette = await self.extract_color_palette(screenshot)

        # Capture performance metrics
        lcp, cls, tti = None, None, None
        if should_capture_performance:
            perf_metrics = await self.capture_performance_metrics(page)
            lcp = perf_metrics.get("lcp")
            cls = perf_metrics.get("cls")
            tti = perf_metrics.get("tti") or perf_metrics.get("domInteractive")

        # Build network HAR
        network_har: Optional[Dict] = None
        if should_capture_network and self._har_entries:
            network_har = {
                "log": {
                    "version": "1.2",
                    "creator": {
                        "name": "EnhancedCapture",
                        "version": "1.0"
                    },
                    "entries": self._har_entries,
                    "pages": [{
                        "startedDateTime": datetime.utcnow().isoformat(),
                        "id": snapshot_id,
                        "title": title,
                        "pageTimings": {}
                    }]
                }
            }

        # Extract text blocks for content analysis
        text_blocks = await self._extract_text_blocks(page)

        # Build computed styles map from elements
        computed_styles = self._build_computed_styles_map(elements_data)

        # Convert elements to VisualElement format
        visual_elements = self._convert_to_visual_elements(elements_data)

        # Calculate capture duration
        capture_duration_ms = (time.time() - start_time) * 1000

        # Create snapshot
        snapshot = VisualSnapshot(
            id=snapshot_id,
            url=current_url,
            viewport=actual_viewport,
            device_name=device_name,
            browser=browser,
            timestamp=datetime.utcnow().isoformat(),
            screenshot=screenshot,
            dom_snapshot=dom_snapshot,
            computed_styles=computed_styles,
            network_har=network_har,
            elements=visual_elements,
            layout_hash=layout_hash,
            color_palette=color_palette,
            text_blocks=text_blocks,
            largest_contentful_paint=lcp,
            cumulative_layout_shift=cls,
            time_to_interactive=tti,
        )

        self.log.info(
            "Snapshot capture complete",
            snapshot_id=snapshot_id,
            element_count=len(visual_elements),
            capture_duration_ms=capture_duration_ms,
            has_performance=lcp is not None,
            has_network=network_har is not None,
            layout_hash=layout_hash[:16] if layout_hash else None,
        )

        return snapshot

    async def _setup_network_capture(self, page: Page) -> None:
        """Set up network request/response capture for HAR.

        Args:
            page: Playwright page instance
        """
        async def on_request(request: Request) -> None:
            request_id = request.url + str(time.time())
            self._request_start_times[request.url] = time.time()

            entry = {
                "url": request.url,
                "method": request.method,
                "headers": dict(request.headers),
                "resource_type": request.resource_type,
                "started_at": datetime.utcnow().isoformat(),
                "start_time": time.time(),
            }
            self._network_entries.append(entry)

        async def on_response(response: Response) -> None:
            end_time = time.time()

            # Find matching request entry
            for entry in reversed(self._network_entries):
                if entry["url"] == response.url and "status" not in entry:
                    start_time = entry.get("start_time", end_time)
                    duration_ms = (end_time - start_time) * 1000

                    entry["status"] = response.status
                    entry["status_text"] = response.status_text
                    entry["response_headers"] = dict(response.headers)
                    entry["finished_at"] = datetime.utcnow().isoformat()
                    entry["duration_ms"] = duration_ms

                    # Try to get timing info
                    try:
                        timing = await response.request.timing()
                        entry["timing"] = timing
                    except Exception:
                        pass

                    # Build HAR entry
                    har_entry = {
                        "startedDateTime": entry["started_at"],
                        "time": duration_ms,
                        "request": {
                            "method": entry["method"],
                            "url": entry["url"],
                            "headers": [{"name": k, "value": v} for k, v in entry["headers"].items()],
                            "queryString": [],
                            "headersSize": -1,
                            "bodySize": -1,
                        },
                        "response": {
                            "status": response.status,
                            "statusText": response.status_text,
                            "headers": [{"name": k, "value": v} for k, v in entry["response_headers"].items()],
                            "content": {
                                "size": -1,
                                "mimeType": response.headers.get("content-type", ""),
                            },
                            "headersSize": -1,
                            "bodySize": -1,
                        },
                        "cache": {},
                        "timings": {
                            "blocked": -1,
                            "dns": -1,
                            "connect": -1,
                            "ssl": -1,
                            "send": -1,
                            "wait": duration_ms,
                            "receive": -1,
                        },
                    }
                    self._har_entries.append(har_entry)
                    break

        async def on_request_failed(request: Request) -> None:
            for entry in reversed(self._network_entries):
                if entry["url"] == request.url and "status" not in entry:
                    entry["failed"] = True
                    entry["failure_reason"] = request.failure
                    break

        page.on("request", on_request)
        page.on("response", on_response)
        page.on("requestfailed", on_request_failed)

    async def capture_url(
        self,
        url: str,
        browser_type: str = "chromium",
        viewport: Optional[Dict[str, int]] = None,
        full_page: bool = False,
        device_name: Optional[str] = None,
        headless: bool = True,
    ) -> VisualSnapshot:
        """Capture a snapshot by launching a fresh browser.

        Args:
            url: URL to capture
            browser_type: Browser to use (chromium, firefox, webkit)
            viewport: Viewport size
            full_page: Whether to capture full page
            device_name: Device name for metadata
            headless: Whether to run headless

        Returns:
            VisualSnapshot with captured data
        """
        viewport = viewport or self.default_viewport

        async with async_playwright() as p:
            browser_launcher = getattr(p, browser_type)
            browser = await browser_launcher.launch(headless=headless)

            context = await browser.new_context(
                viewport=viewport,
                device_scale_factor=1,
            )

            page = await context.new_page()

            try:
                await page.goto(url, wait_until="domcontentloaded")
                snapshot = await self.capture_snapshot(
                    page=page,
                    url=url,
                    viewport=viewport,
                    browser=browser_type,
                    full_page=full_page,
                    device_name=device_name,
                )
                return snapshot
            finally:
                await browser.close()

    async def capture_dom_snapshot(self, page: Page) -> str:
        """Serialize full DOM tree to JSON with element IDs.

        Captures the complete DOM structure including:
        - All element nodes with tag names and attributes
        - Text node content
        - Shadow DOM subtrees
        - Unique node IDs for reference

        Args:
            page: Playwright page instance

        Returns:
            JSON string representation of the DOM tree
        """
        try:
            dom_json = await page.evaluate(DOM_SERIALIZER_JS)
            self.log.debug("DOM snapshot captured", size_bytes=len(dom_json))
            return dom_json
        except Exception as e:
            self.log.error("DOM snapshot capture failed", error=str(e))
            return "{}"

    async def extract_elements(self, page: Page) -> List[Dict[str, Any]]:
        """Extract all visible elements with bounds and styles.

        Extracts comprehensive information about each visible element:
        - Unique element ID and CSS selector
        - Bounding box coordinates
        - Full computed styles
        - Text content and attributes
        - Interactivity indicators

        Args:
            page: Playwright page instance

        Returns:
            List of element dictionaries with full metadata
        """
        try:
            elements = await page.evaluate(ELEMENT_EXTRACTOR_JS, self.max_elements)
            self.log.debug("Elements extracted", count=len(elements))
            return elements
        except Exception as e:
            self.log.error("Element extraction failed", error=str(e))
            return []

    async def capture_performance_metrics(self, page: Page) -> Dict[str, Any]:
        """Get LCP, CLS, TTI and other metrics from Performance API.

        Captures Core Web Vitals and additional performance metrics:
        - LCP (Largest Contentful Paint) with element info
        - CLS (Cumulative Layout Shift)
        - TTI (Time to Interactive) approximation
        - FCP (First Contentful Paint)
        - TTFB (Time to First Byte)
        - Resource timing summary
        - Memory usage (Chrome only)

        Args:
            page: Playwright page instance

        Returns:
            Dictionary with all captured performance metrics
        """
        try:
            metrics = await page.evaluate(PERFORMANCE_METRICS_JS)

            # Calculate approximate TTI if not already present
            if "tti" not in metrics or metrics.get("tti") is None:
                dom_interactive = metrics.get("domInteractive")
                fcp = metrics.get("fcp")
                if dom_interactive and fcp:
                    metrics["tti"] = max(dom_interactive, fcp + 100)

            self.log.debug(
                "Performance metrics captured",
                lcp=metrics.get("lcp"),
                cls=metrics.get("cls"),
                tti=metrics.get("tti"),
                fcp=metrics.get("fcp"),
            )

            return metrics
        except Exception as e:
            self.log.error("Performance metrics capture failed", error=str(e))
            return {}

    async def compute_layout_hash(self, elements: List[Dict[str, Any]]) -> str:
        """Generate structural fingerprint of layout.

        Creates a hash based on element positions, sizes, and types
        that can be used for quick structural comparison. Positions
        are rounded to reduce noise from minor shifts.

        Args:
            elements: List of extracted element dictionaries

        Returns:
            SHA256 hash string (first 32 chars) representing layout structure
        """
        if not elements:
            return ""

        # Build structural representation with rounded positions
        structural_data = []
        for el in elements:
            bounds = el.get("bounds", {})
            structural_data.append({
                "type": el.get("element_type", "other"),
                "tag": el.get("tag_name", ""),
                # Round positions to nearest 10px to reduce noise
                "x": round(bounds.get("x", 0) / 10) * 10,
                "y": round(bounds.get("y", 0) / 10) * 10,
                "w": round(bounds.get("width", 0) / 10) * 10,
                "h": round(bounds.get("height", 0) / 10) * 10,
            })

        # Sort by position for consistent ordering
        structural_data.sort(key=lambda e: (e["y"], e["x"]))

        # Generate hash
        data_str = json.dumps(structural_data, sort_keys=True)
        layout_hash = hashlib.sha256(data_str.encode()).hexdigest()[:32]

        self.log.debug("Layout hash computed", hash=layout_hash, element_count=len(elements))
        return layout_hash

    async def extract_color_palette(self, screenshot: bytes) -> List[str]:
        """Extract dominant colors from screenshot.

        Uses image processing to find the most prominent colors
        in the screenshot, excluding very dark/light colors.

        Args:
            screenshot: PNG screenshot bytes

        Returns:
            List of hex color strings sorted by frequency
        """
        try:
            # Try to use Pillow if available
            from PIL import Image
            import io

            # Load image
            img = Image.open(io.BytesIO(screenshot))

            # Resize for faster processing
            img.thumbnail((150, 150))

            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Get pixels and quantize colors
            pixels = list(img.getdata())

            # Quantize to reduce color space (round to nearest 16)
            quantized = []
            for r, g, b in pixels:
                qr = (r // 16) * 16
                qg = (g // 16) * 16
                qb = (b // 16) * 16
                quantized.append((qr, qg, qb))

            # Count colors
            color_counts = Counter(quantized)

            # Get top colors (excluding very dark and very light)
            filtered_colors = []
            for color, count in color_counts.most_common(50):
                r, g, b = color
                # Skip near-black and near-white
                if (r + g + b) < 30 or (r + g + b) > 735:
                    continue
                filtered_colors.append((color, count))

            # Convert to hex strings
            palette = []
            for (r, g, b), _ in filtered_colors[:10]:
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                palette.append(hex_color)

            self.log.debug("Color palette extracted", colors=palette[:5])
            return palette

        except ImportError:
            # Fallback: extract colors from page CSS
            self.log.warning("Pillow not installed, using fallback color extraction")
            return []
        except Exception as e:
            self.log.error("Color extraction failed", error=str(e))
            return []

    async def _extract_text_blocks(self, page: Page) -> List[Dict]:
        """Extract text blocks with their positions and typography.

        Args:
            page: Playwright page instance

        Returns:
            List of text block dictionaries
        """
        try:
            text_blocks = await page.evaluate(
                """() => {
                    const blocks = [];
                    const textElements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, span, li, td, th, label, a');

                    for (const el of textElements) {
                        const text = el.textContent?.trim();
                        if (!text || text.length < 5) continue;

                        const rect = el.getBoundingClientRect();
                        if (rect.width === 0 || rect.height === 0) continue;

                        const style = window.getComputedStyle(el);

                        blocks.push({
                            text: text.slice(0, 300),
                            tag: el.tagName.toLowerCase(),
                            bounds: {
                                x: rect.x,
                                y: rect.y,
                                width: rect.width,
                                height: rect.height
                            },
                            font_family: style.fontFamily,
                            font_size: style.fontSize,
                            font_weight: style.fontWeight,
                            line_height: style.lineHeight,
                            color: style.color,
                            text_align: style.textAlign,
                        });

                        if (blocks.length >= 100) break;
                    }

                    return blocks;
                }"""
            )
            return text_blocks
        except Exception as e:
            self.log.warning("Failed to extract text blocks", error=str(e))
            return []

    def _build_computed_styles_map(self, elements: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Build a map of element IDs to computed styles.

        Args:
            elements: List of extracted element dictionaries

        Returns:
            Dictionary mapping element_id to computed_styles
        """
        styles_map = {}
        for el in elements:
            element_id = el.get("element_id", "")
            if element_id and "computed_styles" in el:
                styles_map[element_id] = el["computed_styles"]
        return styles_map

    def _convert_to_visual_elements(self, elements: List[Dict[str, Any]]) -> List[VisualElement]:
        """Convert raw element dictionaries to VisualElement instances.

        Args:
            elements: List of extracted element dictionaries

        Returns:
            List of VisualElement instances
        """
        visual_elements = []

        for el in elements:
            try:
                # Filter out undefined attributes
                attrs = {k: v for k, v in el.get("attributes", {}).items() if v}
                visual_element = VisualElement(
                    element_id=el.get("element_id", ""),
                    selector=el.get("selector", ""),
                    tag_name=el.get("tag_name", ""),
                    bounds=el.get("bounds", {"x": 0, "y": 0, "width": 0, "height": 0}),
                    computed_styles=el.get("computed_styles", {}),
                    text_content=el.get("text_content"),
                    attributes=attrs,
                    children_count=el.get("children_count", 0),
                )
                visual_elements.append(visual_element)
            except Exception as e:
                self.log.warning(
                    "Failed to convert element",
                    element_id=el.get("element_id"),
                    error=str(e),
                )

        return visual_elements

    async def capture_element_screenshot(
        self,
        page: Page,
        selector: str,
    ) -> Optional[bytes]:
        """Capture screenshot of a specific element.

        Args:
            page: Playwright page instance
            selector: CSS selector for the element

        Returns:
            PNG bytes of the element, or None if element not found
        """
        try:
            element = await page.query_selector(selector)
            if element:
                screenshot = await element.screenshot(type="png")
                return screenshot
            return None
        except Exception as e:
            self.log.warning(
                "Element screenshot failed",
                selector=selector,
                error=str(e),
            )
            return None

    async def wait_for_stable_layout(
        self,
        page: Page,
        stability_threshold_ms: int = 500,
        max_wait_ms: int = 5000,
    ) -> bool:
        """Wait for layout to stabilize before capture.

        Monitors layout shifts and waits until no significant
        changes occur for the specified threshold.

        Args:
            page: Playwright page instance
            stability_threshold_ms: Time without layout shifts to consider stable
            max_wait_ms: Maximum time to wait

        Returns:
            True if layout stabilized, False if timeout
        """
        start_time = time.time()

        # Set up CLS monitoring
        await page.evaluate("""
        () => {
            window.__last_layout_shift_time = Date.now();
            if (window.PerformanceObserver) {
                const observer = new PerformanceObserver((list) => {
                    for (const entry of list.getEntries()) {
                        if (entry.value > 0.001) {
                            window.__last_layout_shift_time = Date.now();
                        }
                    }
                });
                try {
                    observer.observe({ type: 'layout-shift', buffered: false });
                } catch (e) {}
            }
        }
        """)

        while (time.time() - start_time) * 1000 < max_wait_ms:
            try:
                last_shift = await page.evaluate("() => window.__last_layout_shift_time")
                if last_shift:
                    time_since_shift = time.time() * 1000 - last_shift
                    if time_since_shift >= stability_threshold_ms:
                        self.log.debug(
                            "Layout stabilized",
                            time_since_shift_ms=time_since_shift,
                        )
                        return True
            except Exception:
                pass

            await page.wait_for_timeout(100)

        self.log.warning("Layout did not stabilize within timeout", max_wait_ms=max_wait_ms)
        return False


# Convenience factory function
def create_enhanced_capture(
    capture_performance: bool = True,
    capture_network: bool = True,
    max_elements: int = 1000,
) -> EnhancedCapture:
    """Factory function for creating EnhancedCapture instance.

    Args:
        capture_performance: Whether to capture performance metrics
        capture_network: Whether to capture network HAR
        max_elements: Maximum elements to extract

    Returns:
        Configured EnhancedCapture instance
    """
    return EnhancedCapture(
        capture_performance=capture_performance,
        capture_network=capture_network,
        max_elements=max_elements,
    )
