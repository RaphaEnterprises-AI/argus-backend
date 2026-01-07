"""Autonomous app discovery and test generation.

Like Octomind - automatically explores your app and discovers test scenarios.
Uses AI to understand user flows and generate comprehensive test coverage.
"""

import asyncio
import base64
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

import anthropic
import structlog

from ..config import get_settings

logger = structlog.get_logger()


@dataclass
class DiscoveredElement:
    """An interactive element discovered on a page."""
    type: str  # button, link, input, form, etc.
    text: str  # Visible text
    selector: str  # CSS selector
    action: str  # click, fill, submit, etc.
    purpose: str  # What this element does (AI-inferred)


@dataclass
class DiscoveredPage:
    """A page discovered during crawling."""
    url: str
    title: str
    description: str
    elements: list[DiscoveredElement] = field(default_factory=list)
    screenshot: Optional[str] = None  # Base64
    forms: list[dict] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    user_flows: list[str] = field(default_factory=list)


@dataclass
class DiscoveredFlow:
    """A user flow discovered through exploration."""
    id: str
    name: str
    description: str
    start_url: str
    steps: list[dict] = field(default_factory=list)
    priority: str = "medium"
    category: str = "user_journey"  # user_journey, form_submission, navigation, etc.


@dataclass
class DiscoveryResult:
    """Complete discovery result for an application."""
    app_url: str
    pages_discovered: list[DiscoveredPage] = field(default_factory=list)
    flows_discovered: list[DiscoveredFlow] = field(default_factory=list)
    suggested_tests: list[dict] = field(default_factory=list)
    coverage_summary: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "app_url": self.app_url,
            "pages_discovered": [
                {
                    "url": p.url,
                    "title": p.title,
                    "description": p.description,
                    "elements": [{"type": e.type, "selector": e.selector, "text": e.text, "purpose": e.purpose} for e in p.elements],
                    "forms": p.forms,
                    "links": p.links,
                    "user_flows": p.user_flows,
                }
                for p in self.pages_discovered
            ],
            "flows_discovered": [
                {
                    "id": f.id,
                    "name": f.name,
                    "description": f.description,
                    "start_url": f.start_url,
                    "steps": f.steps,
                    "priority": f.priority,
                    "category": f.category,
                }
                for f in self.flows_discovered
            ],
            "suggested_tests": self.suggested_tests,
            "coverage_summary": self.coverage_summary,
            "timestamp": self.timestamp,
        }


class AutoDiscovery:
    """
    Autonomous app discovery using AI-powered exploration.

    Features:
    - Crawls app starting from URL
    - Identifies interactive elements
    - Discovers user flows and journeys
    - Generates test suggestions
    - Uses Claude Vision to understand UI

    Usage:
        discovery = AutoDiscovery("http://localhost:3000")

        # Full discovery
        result = await discovery.discover()

        # Get suggested tests
        for test in result.suggested_tests:
            print(f"Suggested: {test['name']}")

        # Convert to executable test specs
        specs = discovery.to_test_specs(result)
    """

    def __init__(
        self,
        app_url: str,
        max_pages: int = 20,
        max_depth: int = 3,
        model: Optional[str] = None,
    ):
        settings = get_settings()
        api_key = settings.anthropic_api_key
        if hasattr(api_key, 'get_secret_value'):
            api_key = api_key.get_secret_value()
        self.client = anthropic.Anthropic(api_key=api_key)
        from src.core.model_registry import get_model_id
        model = model or get_model_id("claude-sonnet-4-5")
        self.app_url = app_url
        self.base_domain = urlparse(app_url).netloc
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.model = model
        self.log = logger.bind(component="auto_discovery")

        self.visited_urls = set()
        self.discovered_pages = []
        self.discovered_flows = []

    async def discover(
        self,
        start_paths: Optional[list[str]] = None,
        focus_areas: Optional[list[str]] = None,
    ) -> DiscoveryResult:
        """
        Discover app structure and generate test suggestions.

        Args:
            start_paths: Specific paths to start from (default: ["/"])
            focus_areas: Areas to focus on (e.g., ["authentication", "checkout"])

        Returns:
            DiscoveryResult with pages, flows, and test suggestions
        """
        self.log.info("Starting app discovery", url=self.app_url)

        start_paths = start_paths or ["/"]

        try:
            from ..tools.playwright_tools import BrowserManager, BrowserConfig

            config = BrowserConfig(headless=True, viewport_width=1920, viewport_height=1080)

            async with BrowserManager(config) as browser:
                page = browser.page

                # Crawl from each start path
                for path in start_paths:
                    url = urljoin(self.app_url, path)
                    await self._crawl_page(page, url, depth=0)

                    if len(self.discovered_pages) >= self.max_pages:
                        break

        except ImportError:
            self.log.warning("Playwright not available, using simulation mode")
            await self._simulate_discovery(start_paths)

        # Analyze discovered pages and generate flows
        self.discovered_flows = await self._analyze_flows(focus_areas)

        # Generate test suggestions
        suggested_tests = await self._generate_test_suggestions(focus_areas)

        result = DiscoveryResult(
            app_url=self.app_url,
            pages_discovered=self.discovered_pages,
            flows_discovered=self.discovered_flows,
            suggested_tests=suggested_tests,
            coverage_summary=self._calculate_coverage(),
        )

        self.log.info(
            "Discovery complete",
            pages=len(result.pages_discovered),
            flows=len(result.flows_discovered),
            tests=len(result.suggested_tests),
        )

        return result

    async def _crawl_page(self, page, url: str, depth: int) -> None:
        """Crawl a single page and discover its elements."""
        if url in self.visited_urls:
            return
        if depth > self.max_depth:
            return
        if len(self.discovered_pages) >= self.max_pages:
            return

        # Check if URL is within our domain
        if urlparse(url).netloc != self.base_domain:
            return

        self.visited_urls.add(url)
        self.log.debug("Crawling page", url=url, depth=depth)

        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(1000)  # Let JS settle

            # Get page info
            title = await page.title()

            # Take screenshot
            screenshot_bytes = await page.screenshot()
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

            # Extract elements using JavaScript
            elements_data = await page.evaluate("""
                () => {
                    const elements = [];

                    // Buttons
                    document.querySelectorAll('button, [role="button"], input[type="submit"]').forEach(el => {
                        elements.push({
                            type: 'button',
                            text: el.textContent?.trim() || el.value || '',
                            selector: el.id ? `#${el.id}` : null,
                            tag: el.tagName.toLowerCase()
                        });
                    });

                    // Links
                    document.querySelectorAll('a[href]').forEach(el => {
                        const href = el.getAttribute('href');
                        if (href && !href.startsWith('#') && !href.startsWith('javascript:')) {
                            elements.push({
                                type: 'link',
                                text: el.textContent?.trim() || '',
                                href: href,
                                selector: el.id ? `#${el.id}` : null
                            });
                        }
                    });

                    // Forms
                    document.querySelectorAll('form').forEach(form => {
                        const inputs = [];
                        form.querySelectorAll('input, textarea, select').forEach(input => {
                            inputs.push({
                                type: input.type || input.tagName.toLowerCase(),
                                name: input.name || input.id,
                                placeholder: input.placeholder || '',
                                required: input.required
                            });
                        });
                        elements.push({
                            type: 'form',
                            action: form.action,
                            method: form.method,
                            inputs: inputs
                        });
                    });

                    // Inputs outside forms
                    document.querySelectorAll('input:not(form input), textarea:not(form textarea)').forEach(el => {
                        elements.push({
                            type: 'input',
                            inputType: el.type,
                            placeholder: el.placeholder || '',
                            name: el.name || el.id,
                            selector: el.id ? `#${el.id}` : `[name="${el.name}"]`
                        });
                    });

                    return elements;
                }
            """)

            # Extract links for further crawling
            links = await page.evaluate("""
                () => {
                    const links = [];
                    document.querySelectorAll('a[href]').forEach(el => {
                        const href = el.getAttribute('href');
                        if (href && !href.startsWith('#') && !href.startsWith('javascript:') && !href.startsWith('mailto:')) {
                            links.push(href);
                        }
                    });
                    return [...new Set(links)];
                }
            """)

            # Analyze page with Claude Vision
            page_analysis = await self._analyze_page_with_vision(
                screenshot_b64, title, url, elements_data
            )

            # Build discovered elements
            discovered_elements = []
            for el in elements_data:
                if el.get("type") in ("button", "link", "input"):
                    discovered_elements.append(DiscoveredElement(
                        type=el["type"],
                        text=el.get("text", el.get("placeholder", "")),
                        selector=el.get("selector", ""),
                        action="click" if el["type"] in ("button", "link") else "fill",
                        purpose=page_analysis.get("element_purposes", {}).get(el.get("text", ""), ""),
                    ))

            # Store discovered page
            discovered_page = DiscoveredPage(
                url=url,
                title=title,
                description=page_analysis.get("description", ""),
                elements=discovered_elements,
                screenshot=screenshot_b64,
                forms=[el for el in elements_data if el.get("type") == "form"],
                links=links,
                user_flows=page_analysis.get("possible_flows", []),
            )
            self.discovered_pages.append(discovered_page)

            # Crawl linked pages
            for link in links[:10]:  # Limit links per page
                full_url = urljoin(url, link)
                await self._crawl_page(page, full_url, depth + 1)

        except Exception as e:
            self.log.warning("Failed to crawl page", url=url, error=str(e))

    async def _analyze_page_with_vision(
        self,
        screenshot_b64: str,
        title: str,
        url: str,
        elements: list,
    ) -> dict:
        """Use Claude Vision to understand the page."""
        prompt = f"""Analyze this web page screenshot.

URL: {url}
Title: {title}

Elements found: {len(elements)} (buttons, links, forms, inputs)

Provide a JSON analysis:
{{
    "description": "What is this page for?",
    "page_type": "login|signup|dashboard|settings|list|detail|form|landing|other",
    "key_actions": ["main actions user can take"],
    "possible_flows": ["user journeys that could start/continue here"],
    "element_purposes": {{"element text": "what it does"}},
    "test_priority": "critical|high|medium|low"
}}

Focus on identifying:
1. What the user is trying to accomplish
2. Critical user paths
3. Potential error scenarios
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_b64,
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )

            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())

        except Exception as e:
            self.log.warning("Vision analysis failed", error=str(e))
            return {"description": title, "possible_flows": []}

    async def _analyze_flows(self, focus_areas: Optional[list[str]] = None) -> list[DiscoveredFlow]:
        """Analyze discovered pages and identify user flows."""
        if not self.discovered_pages:
            return []

        pages_summary = []
        for page in self.discovered_pages:
            pages_summary.append({
                "url": page.url,
                "title": page.title,
                "description": page.description,
                "forms": len(page.forms),
                "links": len(page.links),
                "flows_mentioned": page.user_flows,
            })

        focus_text = ""
        if focus_areas:
            focus_text = f"\n\nFOCUS AREAS: {', '.join(focus_areas)}"

        prompt = f"""Analyze these discovered pages and identify complete user flows.

DISCOVERED PAGES:
{json.dumps(pages_summary, indent=2)}
{focus_text}

Identify user flows (journeys across pages). For each flow:
1. Give it a clear name
2. Describe what the user accomplishes
3. List the steps (which pages, what actions)

Respond with JSON:
{{
    "flows": [
        {{
            "id": "flow-1",
            "name": "User Login Flow",
            "description": "User logs into the application",
            "category": "authentication|checkout|profile|navigation|form_submission",
            "priority": "critical|high|medium|low",
            "start_url": "/login",
            "steps": [
                {{"page": "/login", "action": "fill email and password"}},
                {{"page": "/login", "action": "click login button"}},
                {{"page": "/dashboard", "action": "verify logged in"}}
            ]
        }}
    ]
}}

Prioritize:
- Authentication flows (critical)
- Core business flows (high)
- Settings/profile (medium)
- Edge cases (low)
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            flows = []
            for f in data.get("flows", []):
                flows.append(DiscoveredFlow(
                    id=f.get("id", f"flow-{len(flows)}"),
                    name=f.get("name", ""),
                    description=f.get("description", ""),
                    start_url=f.get("start_url", "/"),
                    steps=f.get("steps", []),
                    priority=f.get("priority", "medium"),
                    category=f.get("category", "user_journey"),
                ))

            return flows

        except Exception as e:
            self.log.error("Flow analysis failed", error=str(e))
            return []

    async def _generate_test_suggestions(
        self,
        focus_areas: Optional[list[str]] = None,
    ) -> list[dict]:
        """Generate test suggestions from discovered flows."""
        if not self.discovered_flows:
            return []

        flows_summary = [
            {
                "name": f.name,
                "description": f.description,
                "steps": f.steps,
                "priority": f.priority,
            }
            for f in self.discovered_flows
        ]

        prompt = f"""Generate executable test specifications for these user flows.

DISCOVERED FLOWS:
{json.dumps(flows_summary, indent=2)}

APP URL: {self.app_url}

For each flow, generate a complete test spec with:
1. Specific steps (goto, click, fill, etc.)
2. Assertions to verify success
3. Edge case variations

Respond with JSON:
{{
    "tests": [
        {{
            "id": "test-id",
            "name": "Test name",
            "description": "What this tests",
            "flow_id": "which flow this tests",
            "priority": "critical|high|medium|low",
            "type": "happy_path|edge_case|error_handling",
            "steps": [
                {{"action": "goto", "target": "/login"}},
                {{"action": "fill", "target": "placeholder=Email", "value": "test@example.com"}},
                {{"action": "click", "target": "text=Login"}}
            ],
            "assertions": [
                {{"type": "url_matches", "expected": "/dashboard"}},
                {{"type": "element_visible", "target": "text=Welcome"}}
            ]
        }}
    ]
}}
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())
            return data.get("tests", [])

        except Exception as e:
            self.log.error("Test generation failed", error=str(e))
            return []

    async def _simulate_discovery(self, start_paths: list[str]) -> None:
        """Simulate discovery when Playwright isn't available."""
        self.log.info("Running simulated discovery")

        prompt = f"""Simulate discovering a web application.

APP URL: {self.app_url}
START PATHS: {start_paths}

Imagine this is a typical web application. What pages and flows would you expect?

Respond with JSON:
{{
    "pages": [
        {{
            "url": "/path",
            "title": "Page Title",
            "description": "What this page does",
            "elements": ["button: Login", "input: Email", "form: Login Form"],
            "links": ["/other-page"]
        }}
    ]
}}

Include typical pages like login, signup, dashboard, settings, etc.
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]

        data = json.loads(content.strip())

        for page_data in data.get("pages", []):
            self.discovered_pages.append(DiscoveredPage(
                url=urljoin(self.app_url, page_data.get("url", "/")),
                title=page_data.get("title", ""),
                description=page_data.get("description", ""),
                links=page_data.get("links", []),
            ))

    def _calculate_coverage(self) -> dict:
        """Calculate test coverage summary."""
        return {
            "pages_discovered": len(self.discovered_pages),
            "flows_identified": len(self.discovered_flows),
            "forms_found": sum(len(p.forms) for p in self.discovered_pages),
            "interactive_elements": sum(len(p.elements) for p in self.discovered_pages),
            "critical_flows": sum(1 for f in self.discovered_flows if f.priority == "critical"),
            "coverage_score": min(100, len(self.discovered_flows) * 10),  # Simple score
        }

    def to_test_specs(self, result: DiscoveryResult) -> list[dict]:
        """Convert discovery result to executable test specifications."""
        return result.suggested_tests


class QuickDiscover:
    """
    Quick discovery mode - just the essentials.

    For when you want fast results without full crawling.
    """

    def __init__(self, app_url: str):
        self.app_url = app_url
        self.discovery = AutoDiscovery(app_url, max_pages=5, max_depth=1)

    async def discover_login_flow(self) -> list[dict]:
        """Quickly discover and generate login tests."""
        result = await self.discovery.discover(
            start_paths=["/login", "/signin", "/auth"],
            focus_areas=["authentication"],
        )
        return [t for t in result.suggested_tests if "login" in t.get("name", "").lower()]

    async def discover_signup_flow(self) -> list[dict]:
        """Quickly discover and generate signup tests."""
        result = await self.discovery.discover(
            start_paths=["/signup", "/register", "/join"],
            focus_areas=["registration"],
        )
        return [t for t in result.suggested_tests if "signup" in t.get("name", "").lower() or "register" in t.get("name", "").lower()]

    async def discover_critical_flows(self) -> list[dict]:
        """Discover all critical priority flows."""
        result = await self.discovery.discover()
        return [t for t in result.suggested_tests if t.get("priority") == "critical"]


# Convenience function
def create_auto_discovery(app_url: str, max_pages: int = 20) -> AutoDiscovery:
    """Factory function for AutoDiscovery."""
    return AutoDiscovery(app_url=app_url, max_pages=max_pages)
