"""
Browser Pool Models

Unified types and models for the Vultr Browser Pool.
These replace the fragmented models across multiple clients.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ActionType(str, Enum):
    """Types of browser actions."""
    CLICK = "click"
    FILL = "fill"
    TYPE = "type"
    PRESS = "press"
    NAVIGATE = "navigate"
    HOVER = "hover"
    SELECT = "select"
    WAIT = "wait"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"


class BrowserType(str, Enum):
    """Supported browser types."""
    CHROMIUM = "chromium"
    CHROME = "chrome"
    FIREFOX = "firefox"
    WEBKIT = "webkit"
    EDGE = "edge"


class ExecutionMode(str, Enum):
    """How the action was executed."""
    DOM = "dom"           # Fast DOM-based execution
    VISION = "vision"     # Claude Computer Use fallback
    HYBRID = "hybrid"     # DOM with vision fallback
    CACHED = "cached"     # Used cached selector


@dataclass
class ElementInfo:
    """Information about a discovered element."""
    selector: str
    type: str
    tag_name: str
    text: str = ""
    value: str = ""
    placeholder: str = ""
    description: str = ""
    confidence: float = 0.5
    bounds: dict | None = None
    attributes: dict | None = None

    def to_dict(self) -> dict:
        return {
            "selector": self.selector,
            "type": self.type,
            "tagName": self.tag_name,
            "text": self.text,
            "value": self.value,
            "placeholder": self.placeholder,
            "description": self.description,
            "confidence": self.confidence,
            "bounds": self.bounds,
            "attributes": self.attributes,
        }


@dataclass
class ActionResult:
    """Result of a single browser action."""
    action: str
    success: bool
    selector: str | None = None
    value: str | None = None
    url: str | None = None
    error: str | None = None
    duration_ms: int | None = None

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "success": self.success,
            "selector": self.selector,
            "value": self.value,
            "url": self.url,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ObserveResult:
    """Result of observing/discovering page elements."""
    success: bool
    url: str
    title: str = ""
    elements: list[ElementInfo] = field(default_factory=list)
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def actions(self) -> list[ElementInfo]:
        """Alias for elements (MCP compatibility)."""
        return self.elements

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "url": self.url,
            "title": self.title,
            "actions": [e.to_dict() for e in self.elements],
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class ActResult:
    """Result of executing a browser action."""
    success: bool
    message: str = ""
    actions: list[ActionResult] = field(default_factory=list)
    url: str = ""
    screenshot: str | None = None  # Base64 encoded
    error: str | None = None
    execution_mode: ExecutionMode = ExecutionMode.DOM
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "actions": [a.to_dict() for a in self.actions],
            "url": self.url,
            "screenshot": self.screenshot,
            "error": self.error,
            "execution_mode": self.execution_mode.value,
            "timestamp": self.timestamp,
        }


@dataclass
class StepResult:
    """Result of a single test step."""
    step_index: int
    instruction: str
    success: bool
    actions: list[ActionResult] = field(default_factory=list)
    duration_ms: int = 0
    screenshot: str | None = None
    error: str | None = None
    execution_mode: ExecutionMode = ExecutionMode.DOM

    def to_dict(self) -> dict:
        return {
            "stepIndex": self.step_index,
            "instruction": self.instruction,
            "success": self.success,
            "actions": [a.to_dict() for a in self.actions],
            "duration_ms": self.duration_ms,
            "screenshot": self.screenshot,
            "error": self.error,
            "execution_mode": self.execution_mode.value,
        }


@dataclass
class TestResult:
    """Result of a multi-step test execution."""
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0
    final_screenshot: str | None = None
    video_artifact_id: str | None = None  # Video recording of test execution
    total_duration_ms: int = 0
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self):
        if self.steps:
            self.total_steps = len(self.steps)
            self.passed_steps = sum(1 for s in self.steps if s.success)
            self.failed_steps = self.total_steps - self.passed_steps
            self.total_duration_ms = sum(s.duration_ms for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "steps": [s.to_dict() for s in self.steps],
            "summary": {
                "total": self.total_steps,
                "passed": self.passed_steps,
                "failed": self.failed_steps,
            },
            "finalScreenshot": self.final_screenshot,
            "videoArtifactId": self.video_artifact_id,
            "totalDuration_ms": self.total_duration_ms,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class ExtractResult:
    """Result of data extraction from a page."""
    success: bool
    data: dict = field(default_factory=dict)
    url: str = ""
    screenshot: str | None = None
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "url": self.url,
            "screenshot": self.screenshot,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class SessionInfo:
    """Information about a browser session."""
    session_id: str
    pod_name: str
    pod_ip: str
    browser_type: BrowserType = BrowserType.CHROMIUM
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_active: bool = True


@dataclass
class PoolHealth:
    """Health status of the browser pool."""
    healthy: bool
    total_pods: int = 0
    available_pods: int = 0
    active_sessions: int = 0
    pool_url: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "totalPods": self.total_pods,
            "availablePods": self.available_pods,
            "activeSessions": self.active_sessions,
            "poolUrl": self.pool_url,
            "timestamp": self.timestamp,
        }


@dataclass
class BrowserPoolConfig:
    """Configuration for the browser pool client."""
    pool_url: str
    timeout_ms: int = 60000
    retry_count: int = 3
    retry_delay_ms: int = 1000
    default_viewport: tuple[int, int] = (1920, 1080)
    default_browser: BrowserType = BrowserType.CHROMIUM
    capture_screenshots: bool = True
    enable_caching: bool = True
    enable_self_healing: bool = True
    vision_fallback_enabled: bool = True
    vision_fallback_timeout_ms: int = 30000

    def to_dict(self) -> dict:
        return {
            "poolUrl": self.pool_url,
            "timeoutMs": self.timeout_ms,
            "retryCount": self.retry_count,
            "retryDelayMs": self.retry_delay_ms,
            "defaultViewport": {
                "width": self.default_viewport[0],
                "height": self.default_viewport[1],
            },
            "defaultBrowser": self.default_browser.value,
            "captureScreenshots": self.capture_screenshots,
            "enableCaching": self.enable_caching,
            "enableSelfHealing": self.enable_self_healing,
            "visionFallbackEnabled": self.vision_fallback_enabled,
            "visionFallbackTimeoutMs": self.vision_fallback_timeout_ms,
        }


@dataclass
class DiscoveredPage:
    """A page discovered during crawling."""
    url: str
    title: str
    description: str = ""
    page_type: str = "unknown"
    elements: list[ElementInfo] = field(default_factory=list)
    forms: list[dict] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    screenshot: str | None = None
    discovered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "pageType": self.page_type,
            "elements": [e.to_dict() for e in self.elements],
            "forms": self.forms,
            "links": self.links,
            "screenshot": self.screenshot,
            "discoveredAt": self.discovered_at,
        }


@dataclass
class DiscoveryResult:
    """Result of a discovery crawl operation."""
    success: bool
    pages: list[DiscoveredPage] = field(default_factory=list)
    total_elements: int = 0
    total_forms: int = 0
    total_links: int = 0
    video_artifact_id: str | None = None
    recording_url: str | None = None
    error: str | None = None
    duration_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "pages": [p.to_dict() for p in self.pages],
            "totalElements": self.total_elements,
            "totalForms": self.total_forms,
            "totalLinks": self.total_links,
            "videoArtifactId": self.video_artifact_id,
            "recordingUrl": self.recording_url,
            "error": self.error,
            "durationMs": self.duration_ms,
            "timestamp": self.timestamp,
        }
