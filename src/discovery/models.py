"""
Data models for the Discovery Intelligence Platform.

This module defines all the data structures used for autonomous crawling,
element discovery, page classification, and flow detection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# =============================================================================
# Enums
# =============================================================================

class DiscoveryMode(str, Enum):
    """
    Discovery mode determines the depth and thoroughness of exploration.

    Attributes:
        quick_scan: Fast surface-level scan, minimal depth
        standard_crawl: Balanced exploration with moderate depth
        deep: Thorough exploration including edge cases
        focused: Target specific areas or patterns
        diff: Compare changes between sessions
        autonomous: AI-guided adaptive exploration
    """
    quick_scan = "quick_scan"
    standard_crawl = "standard_crawl"
    deep = "deep"
    focused = "focused"
    diff = "diff"
    autonomous = "autonomous"


class ExplorationStrategy(str, Enum):
    """
    Strategy for traversing and exploring discovered pages.

    Attributes:
        breadth_first: Explore all links at current depth before going deeper
        depth_first: Follow links deeply before backtracking
        priority: Visit high-priority pages first based on scoring
        random: Randomized exploration for fuzzing
        ai_guided: AI determines next action based on context
    """
    breadth_first = "breadth_first"
    depth_first = "depth_first"
    priority = "priority"
    random = "random"
    ai_guided = "ai_guided"


class ElementCategory(str, Enum):
    """
    Category of discovered UI elements.

    Attributes:
        navigation: Links, menus, breadcrumbs
        form: Input fields, textareas, selects
        action: Buttons, submit controls
        content: Static text, images, media
        interactive: Modals, dropdowns, accordions
        authentication: Login/logout, session controls
        commerce: Cart, checkout, payment elements
        social: Share buttons, comments, reactions
    """
    navigation = "navigation"
    form = "form"
    action = "action"
    content = "content"
    interactive = "interactive"
    authentication = "authentication"
    commerce = "commerce"
    social = "social"


class PageCategory(str, Enum):
    """
    Classification of discovered pages by their purpose.

    Attributes:
        landing: Homepage or marketing pages
        auth_login: Login page
        auth_signup: Registration page
        auth_reset: Password reset page
        dashboard: Main user dashboard
        list: List/index views (tables, grids)
        detail: Single item detail views
        form: Forms for data entry/editing
        settings: User or app settings
        profile: User profile pages
        checkout: E-commerce checkout flow
        error: Error pages (404, 500, etc)
        other: Uncategorized pages
    """
    landing = "landing"
    auth_login = "auth_login"
    auth_signup = "auth_signup"
    auth_reset = "auth_reset"
    dashboard = "dashboard"
    list = "list"
    detail = "detail"
    form = "form"
    settings = "settings"
    profile = "profile"
    checkout = "checkout"
    error = "error"
    other = "other"


class FlowCategory(str, Enum):
    """
    Classification of discovered user flows.

    Attributes:
        authentication: Login, logout, session management
        registration: User signup and onboarding
        navigation: Site navigation patterns
        search: Search and filtering
        crud: Create, read, update, delete operations
        checkout: E-commerce purchase flows
        profile: Profile viewing and editing
        settings: Configuration changes
        social: Social interactions (share, comment, like)
        admin: Administrative operations
    """
    authentication = "authentication"
    registration = "registration"
    navigation = "navigation"
    search = "search"
    crud = "crud"
    checkout = "checkout"
    profile = "profile"
    settings = "settings"
    social = "social"
    admin = "admin"


class DiscoveryStatus(str, Enum):
    """
    Status of a discovery session.

    Attributes:
        pending: Session created but not started
        running: Discovery in progress
        paused: Discovery paused by user
        completed: Discovery finished successfully
        failed: Discovery failed with errors
        cancelled: Discovery cancelled by user
    """
    pending = "pending"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class AuthConfig:
    """
    Configuration for authentication during discovery.

    Attributes:
        login_url: URL of the login page
        username_selector: CSS selector for username field
        password_selector: CSS selector for password field
        submit_selector: CSS selector for submit button
        username: Username credential
        password: Password credential
        success_indicator: Selector or text indicating successful login
        cookies: Pre-authenticated cookies to use
        headers: Authentication headers (e.g., Bearer tokens)
    """
    login_url: str | None = None
    username_selector: str | None = None
    password_selector: str | None = None
    submit_selector: str | None = None
    username: str | None = None
    password: str | None = None
    success_indicator: str | None = None
    cookies: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "login_url": self.login_url,
            "username_selector": self.username_selector,
            "password_selector": self.password_selector,
            "submit_selector": self.submit_selector,
            "username": self.username,
            "password": "***" if self.password else None,  # Mask password
            "success_indicator": self.success_indicator,
            "cookies": {k: "***" for k in self.cookies.keys()},  # Mask cookies
            "headers": {k: "***" for k in self.headers.keys()},  # Mask headers
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthConfig":
        """Create from dictionary representation."""
        return cls(
            login_url=data.get("login_url"),
            username_selector=data.get("username_selector"),
            password_selector=data.get("password_selector"),
            submit_selector=data.get("submit_selector"),
            username=data.get("username"),
            password=data.get("password"),
            success_indicator=data.get("success_indicator"),
            cookies=data.get("cookies", {}),
            headers=data.get("headers", {}),
        )


@dataclass
class DiscoveryConfig:
    """
    Configuration for a discovery session.

    Attributes:
        mode: Discovery mode (quick_scan, standard_crawl, deep, etc.)
        strategy: Exploration strategy (breadth_first, depth_first, etc.)
        max_pages: Maximum number of pages to discover
        max_depth: Maximum link depth from start URL
        max_duration_seconds: Maximum discovery duration in seconds
        include_patterns: URL patterns to include (regex)
        exclude_patterns: URL patterns to exclude (regex)
        focus_areas: Specific areas to focus on (e.g., ["authentication", "checkout"])
        auth_required: Whether authentication is needed
        auth_config: Authentication configuration
        capture_screenshots: Whether to capture page screenshots
        capture_dom: Whether to capture DOM snapshots
        capture_network: Whether to capture network traffic
        use_vision_ai: Whether to use vision AI for element detection
        use_computer_use: Whether to use Claude Computer Use for exploration
        use_cross_project_learning: Learn from other projects
        learn_from_session: Allow this session to contribute to learning
    """
    mode: DiscoveryMode = DiscoveryMode.standard_crawl
    strategy: ExplorationStrategy = ExplorationStrategy.breadth_first
    max_pages: int = 100
    max_depth: int = 5
    max_duration_seconds: int = 3600
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=list)
    auth_required: bool = False
    auth_config: AuthConfig | None = None
    capture_screenshots: bool = True
    capture_dom: bool = True
    capture_network: bool = False
    use_vision_ai: bool = True
    use_computer_use: bool = False
    use_cross_project_learning: bool = True
    learn_from_session: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "mode": self.mode.value,
            "strategy": self.strategy.value,
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "max_duration_seconds": self.max_duration_seconds,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
            "focus_areas": self.focus_areas,
            "auth_required": self.auth_required,
            "auth_config": self.auth_config.to_dict() if self.auth_config else None,
            "capture_screenshots": self.capture_screenshots,
            "capture_dom": self.capture_dom,
            "capture_network": self.capture_network,
            "use_vision_ai": self.use_vision_ai,
            "use_computer_use": self.use_computer_use,
            "use_cross_project_learning": self.use_cross_project_learning,
            "learn_from_session": self.learn_from_session,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveryConfig":
        """Create from dictionary representation."""
        auth_config_data = data.get("auth_config")
        return cls(
            mode=DiscoveryMode(data.get("mode", "standard_crawl")),
            strategy=ExplorationStrategy(data.get("strategy", "breadth_first")),
            max_pages=data.get("max_pages", 100),
            max_depth=data.get("max_depth", 5),
            max_duration_seconds=data.get("max_duration_seconds", 3600),
            include_patterns=data.get("include_patterns", []),
            exclude_patterns=data.get("exclude_patterns", []),
            focus_areas=data.get("focus_areas", []),
            auth_required=data.get("auth_required", False),
            auth_config=AuthConfig.from_dict(auth_config_data) if auth_config_data else None,
            capture_screenshots=data.get("capture_screenshots", True),
            capture_dom=data.get("capture_dom", True),
            capture_network=data.get("capture_network", False),
            use_vision_ai=data.get("use_vision_ai", True),
            use_computer_use=data.get("use_computer_use", False),
            use_cross_project_learning=data.get("use_cross_project_learning", True),
            learn_from_session=data.get("learn_from_session", True),
        )


# =============================================================================
# Discovery Result Dataclasses
# =============================================================================

@dataclass
class ElementBounds:
    """
    Bounding box for an element.

    Attributes:
        x: X coordinate (left)
        y: Y coordinate (top)
        width: Element width
        height: Element height
    """
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "ElementBounds":
        """Create from dictionary representation."""
        return cls(
            x=data.get("x", 0),
            y=data.get("y", 0),
            width=data.get("width", 0),
            height=data.get("height", 0),
        )


@dataclass
class DiscoveredElement:
    """
    A discovered UI element on a page.

    Attributes:
        id: Unique identifier for the element
        page_url: URL of the page containing this element
        selector: Primary CSS selector
        xpath: XPath selector
        category: Element category (navigation, form, action, etc.)
        purpose: Inferred purpose of the element
        label: Human-readable label
        bounds: Bounding box coordinates
        importance_score: How important this element is (0-1)
        stability_score: How stable the selector is (0-1)
        alternative_selectors: List of fallback selectors
        tag_name: HTML tag name
        html_attributes: Relevant HTML attributes
        is_visible: Whether element is visible
        is_enabled: Whether element is enabled/interactive
        is_required: Whether element is required (for forms)
        aria_label: ARIA label if present
        role: ARIA role if present
    """
    id: str
    page_url: str
    selector: str
    xpath: str | None = None
    category: ElementCategory = ElementCategory.content
    purpose: str | None = None
    label: str | None = None
    bounds: ElementBounds | None = None
    importance_score: float = 0.5
    stability_score: float = 0.5
    alternative_selectors: list[str] = field(default_factory=list)
    tag_name: str = "div"
    html_attributes: dict[str, str] = field(default_factory=dict)
    is_visible: bool = True
    is_enabled: bool = True
    is_required: bool = False
    aria_label: str | None = None
    role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "page_url": self.page_url,
            "selector": self.selector,
            "xpath": self.xpath,
            "category": self.category.value,
            "purpose": self.purpose,
            "label": self.label,
            "bounds": self.bounds.to_dict() if self.bounds else None,
            "importance_score": self.importance_score,
            "stability_score": self.stability_score,
            "alternative_selectors": self.alternative_selectors,
            "tag_name": self.tag_name,
            "html_attributes": self.html_attributes,
            "is_visible": self.is_visible,
            "is_enabled": self.is_enabled,
            "is_required": self.is_required,
            "aria_label": self.aria_label,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveredElement":
        """Create from dictionary representation."""
        bounds_data = data.get("bounds")
        return cls(
            id=data["id"],
            page_url=data["page_url"],
            selector=data["selector"],
            xpath=data.get("xpath"),
            category=ElementCategory(data.get("category", "content")),
            purpose=data.get("purpose"),
            label=data.get("label"),
            bounds=ElementBounds.from_dict(bounds_data) if bounds_data else None,
            importance_score=data.get("importance_score", 0.5),
            stability_score=data.get("stability_score", 0.5),
            alternative_selectors=data.get("alternative_selectors", []),
            tag_name=data.get("tag_name", "div"),
            html_attributes=data.get("html_attributes", {}),
            is_visible=data.get("is_visible", True),
            is_enabled=data.get("is_enabled", True),
            is_required=data.get("is_required", False),
            aria_label=data.get("aria_label"),
            role=data.get("role"),
        )


@dataclass
class DiscoveredPage:
    """
    A discovered page in the application.

    Attributes:
        id: Unique identifier for the page
        url: Full URL of the page
        title: Page title
        description: Page description or meta description
        category: Page category (landing, dashboard, form, etc.)
        elements: List of discovered elements on this page
        outgoing_links: URLs this page links to
        incoming_links: URLs that link to this page
        importance_score: Business importance of this page (0-1)
        coverage_score: Test coverage score (0-1)
        risk_score: Risk/complexity score (0-1)
        depth: Link depth from start URL
        screenshot_base64: Base64 encoded screenshot
        dom_snapshot_url: URL to stored DOM snapshot
        load_time_ms: Page load time in milliseconds
        requires_auth: Whether page requires authentication
    """
    id: str
    url: str
    title: str | None = None
    description: str | None = None
    category: PageCategory = PageCategory.other
    elements: list[DiscoveredElement] = field(default_factory=list)
    outgoing_links: set[str] = field(default_factory=set)
    incoming_links: set[str] = field(default_factory=set)
    importance_score: float = 0.5
    coverage_score: float = 0.0
    risk_score: float = 0.5
    depth: int = 0
    screenshot_base64: str | None = None
    dom_snapshot_url: str | None = None
    load_time_ms: int | None = None
    requires_auth: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "elements": [e.to_dict() for e in self.elements],
            "outgoing_links": list(self.outgoing_links),
            "incoming_links": list(self.incoming_links),
            "importance_score": self.importance_score,
            "coverage_score": self.coverage_score,
            "risk_score": self.risk_score,
            "depth": self.depth,
            "screenshot_base64": self.screenshot_base64,
            "dom_snapshot_url": self.dom_snapshot_url,
            "load_time_ms": self.load_time_ms,
            "requires_auth": self.requires_auth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveredPage":
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            url=data["url"],
            title=data.get("title"),
            description=data.get("description"),
            category=PageCategory(data.get("category", "other")),
            elements=[DiscoveredElement.from_dict(e) for e in data.get("elements", [])],
            outgoing_links=set(data.get("outgoing_links", [])),
            incoming_links=set(data.get("incoming_links", [])),
            importance_score=data.get("importance_score", 0.5),
            coverage_score=data.get("coverage_score", 0.0),
            risk_score=data.get("risk_score", 0.5),
            depth=data.get("depth", 0),
            screenshot_base64=data.get("screenshot_base64"),
            dom_snapshot_url=data.get("dom_snapshot_url"),
            load_time_ms=data.get("load_time_ms"),
            requires_auth=data.get("requires_auth", False),
        )


@dataclass
class FlowStep:
    """
    A single step in a discovered flow.

    Attributes:
        order: Step order in the flow
        page_url: URL of the page for this step
        action: Action to perform (click, type, navigate, etc.)
        element_selector: Selector for the target element
        input_value: Value to input (for type actions)
        expected_result: Expected outcome of this step
        wait_condition: Condition to wait for after action
    """
    order: int
    page_url: str
    action: str
    element_selector: str | None = None
    input_value: str | None = None
    expected_result: str | None = None
    wait_condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "order": self.order,
            "page_url": self.page_url,
            "action": self.action,
            "element_selector": self.element_selector,
            "input_value": self.input_value,
            "expected_result": self.expected_result,
            "wait_condition": self.wait_condition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowStep":
        """Create from dictionary representation."""
        return cls(
            order=data["order"],
            page_url=data["page_url"],
            action=data["action"],
            element_selector=data.get("element_selector"),
            input_value=data.get("input_value"),
            expected_result=data.get("expected_result"),
            wait_condition=data.get("wait_condition"),
        )


@dataclass
class DiscoveredFlow:
    """
    A discovered user flow through the application.

    Attributes:
        id: Unique identifier for the flow
        name: Human-readable flow name
        description: Description of what this flow accomplishes
        category: Flow category (authentication, checkout, etc.)
        priority: Priority ranking (1=highest)
        start_url: Starting URL for the flow
        pages: List of page IDs involved in this flow
        steps: Ordered list of flow steps
        success_criteria: How to verify flow success
        failure_indicators: Signs of flow failure
        complexity_score: How complex the flow is (0-1)
        business_value_score: Business importance (0-1)
        confidence_score: Confidence in flow accuracy (0-1)
        validated: Whether flow has been validated
        step_count: Number of steps in the flow
    """
    id: str
    name: str
    description: str | None = None
    category: FlowCategory = FlowCategory.navigation
    priority: int = 5
    start_url: str | None = None
    pages: list[str] = field(default_factory=list)
    steps: list[FlowStep] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    failure_indicators: list[str] = field(default_factory=list)
    complexity_score: float = 0.5
    business_value_score: float = 0.5
    confidence_score: float = 0.5
    validated: bool = False
    step_count: int = 0

    def __post_init__(self):
        """Update step_count after initialization."""
        self.step_count = len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority,
            "start_url": self.start_url,
            "pages": self.pages,
            "steps": [s.to_dict() for s in self.steps],
            "success_criteria": self.success_criteria,
            "failure_indicators": self.failure_indicators,
            "complexity_score": self.complexity_score,
            "business_value_score": self.business_value_score,
            "confidence_score": self.confidence_score,
            "validated": self.validated,
            "step_count": self.step_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveredFlow":
        """Create from dictionary representation."""
        flow = cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            category=FlowCategory(data.get("category", "navigation")),
            priority=data.get("priority", 5),
            start_url=data.get("start_url"),
            pages=data.get("pages", []),
            steps=[FlowStep.from_dict(s) for s in data.get("steps", [])],
            success_criteria=data.get("success_criteria", []),
            failure_indicators=data.get("failure_indicators", []),
            complexity_score=data.get("complexity_score", 0.5),
            business_value_score=data.get("business_value_score", 0.5),
            confidence_score=data.get("confidence_score", 0.5),
            validated=data.get("validated", False),
        )
        return flow


@dataclass
class PageGraphEdge:
    """
    An edge in the page graph representing a link between pages.

    Attributes:
        source: Source page URL
        target: Target page URL
        link_text: Text of the link
        link_selector: CSS selector for the link
        weight: Edge weight (for graph algorithms)
    """
    source: str
    target: str
    link_text: str | None = None
    link_selector: str | None = None
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source": self.source,
            "target": self.target,
            "link_text": self.link_text,
            "link_selector": self.link_selector,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PageGraphEdge":
        """Create from dictionary representation."""
        return cls(
            source=data["source"],
            target=data["target"],
            link_text=data.get("link_text"),
            link_selector=data.get("link_selector"),
            weight=data.get("weight", 1.0),
        )


@dataclass
class PageGraph:
    """
    Graph representation of discovered pages and their connections.

    Attributes:
        nodes: List of page URLs (nodes)
        edges: List of graph edges
        adjacency_list: Adjacency list representation
    """
    nodes: list[str] = field(default_factory=list)
    edges: list[PageGraphEdge] = field(default_factory=list)
    adjacency_list: dict[str, list[str]] = field(default_factory=dict)

    def add_edge(self, source: str, target: str, **kwargs) -> None:
        """Add an edge to the graph."""
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)

        self.edges.append(PageGraphEdge(source=source, target=target, **kwargs))

        if source not in self.adjacency_list:
            self.adjacency_list[source] = []
        if target not in self.adjacency_list[source]:
            self.adjacency_list[source].append(target)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "adjacency_list": self.adjacency_list,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PageGraph":
        """Create from dictionary representation."""
        return cls(
            nodes=data.get("nodes", []),
            edges=[PageGraphEdge.from_dict(e) for e in data.get("edges", [])],
            adjacency_list=data.get("adjacency_list", {}),
        )


@dataclass
class CrawlError:
    """
    An error that occurred during crawling.

    Attributes:
        url: URL where error occurred
        error_type: Type of error
        message: Error message
        timestamp: When the error occurred
        recoverable: Whether the error is recoverable
    """
    url: str
    error_type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    recoverable: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "url": self.url,
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "recoverable": self.recoverable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrawlError":
        """Create from dictionary representation."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()

        return cls(
            url=data["url"],
            error_type=data["error_type"],
            message=data["message"],
            timestamp=timestamp,
            recoverable=data.get("recoverable", True),
        )


@dataclass
class CrawlResult:
    """
    Result of a crawl operation.

    Attributes:
        pages: Dictionary of discovered pages by URL
        total_pages: Total number of pages discovered
        graph: Page graph structure
        duration_ms: Total crawl duration in milliseconds
        errors: List of errors encountered
    """
    pages: dict[str, DiscoveredPage] = field(default_factory=dict)
    total_pages: int = 0
    graph: PageGraph | None = None
    duration_ms: int = 0
    errors: list[CrawlError] = field(default_factory=list)

    def __post_init__(self):
        """Update total_pages after initialization."""
        self.total_pages = len(self.pages)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pages": {url: page.to_dict() for url, page in self.pages.items()},
            "total_pages": self.total_pages,
            "graph": self.graph.to_dict() if self.graph else None,
            "duration_ms": self.duration_ms,
            "errors": [e.to_dict() for e in self.errors],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrawlResult":
        """Create from dictionary representation."""
        graph_data = data.get("graph")
        return cls(
            pages={url: DiscoveredPage.from_dict(p) for url, p in data.get("pages", {}).items()},
            graph=PageGraph.from_dict(graph_data) if graph_data else None,
            duration_ms=data.get("duration_ms", 0),
            errors=[CrawlError.from_dict(e) for e in data.get("errors", [])],
        )


@dataclass
class DiscoverySession:
    """
    A discovery session tracking the state of an exploration.

    Attributes:
        id: Unique session identifier
        project_id: Associated project ID
        status: Current session status
        mode: Discovery mode
        strategy: Exploration strategy
        config: Full discovery configuration
        progress_percentage: Progress (0-100)
        current_page: URL currently being explored
        started_at: Session start timestamp
        completed_at: Session completion timestamp
        pages_found: Number of pages discovered
        flows_found: Number of flows discovered
        elements_found: Number of elements discovered
    """
    id: str
    project_id: str
    status: DiscoveryStatus = DiscoveryStatus.pending
    mode: DiscoveryMode = DiscoveryMode.standard_crawl
    strategy: ExplorationStrategy = ExplorationStrategy.breadth_first
    config: DiscoveryConfig | None = None
    progress_percentage: float = 0.0
    current_page: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    pages_found: int = 0
    flows_found: int = 0
    elements_found: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "status": self.status.value,
            "mode": self.mode.value,
            "strategy": self.strategy.value,
            "config": self.config.to_dict() if self.config else None,
            "progress_percentage": self.progress_percentage,
            "current_page": self.current_page,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "pages_found": self.pages_found,
            "flows_found": self.flows_found,
            "elements_found": self.elements_found,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoverySession":
        """Create from dictionary representation."""
        config_data = data.get("config")
        started_at = data.get("started_at")
        completed_at = data.get("completed_at")

        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)

        return cls(
            id=data["id"],
            project_id=data["project_id"],
            status=DiscoveryStatus(data.get("status", "pending")),
            mode=DiscoveryMode(data.get("mode", "standard_crawl")),
            strategy=ExplorationStrategy(data.get("strategy", "breadth_first")),
            config=DiscoveryConfig.from_dict(config_data) if config_data else None,
            progress_percentage=data.get("progress_percentage", 0.0),
            current_page=data.get("current_page"),
            started_at=started_at,
            completed_at=completed_at,
            pages_found=data.get("pages_found", 0),
            flows_found=data.get("flows_found", 0),
            elements_found=data.get("elements_found", 0),
        )
