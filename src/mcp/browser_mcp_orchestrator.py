"""MCP Browser Orchestrator — unified browser control with self-healing.

Combines Playwright MCP (execution) + Chrome DevTools MCP (debugging) into a
single interface with:
- Self-healing via backend healing API
- Kafka event emission for the AI learning loop
- Automatic diagnostic enrichment on failure

NOTE: For local development, prefer the `argus-mcp` package (argus-mcp/) which
bundles Playwright directly in-process with built-in CDP. This orchestrator is
used by the backend for server-side browser control via subprocess MCP servers.

Architecture:
    Agent Code → MCPBrowserOrchestrator
                    ├── PlaywrightMCPClient  (test execution)
                    ├── DevToolsMCPClient    (debugging/diagnostics)
                    ├── Healing API          (self-healing on failure)
                    └── Kafka Producer       (event emission)

Usage:
    async with MCPBrowserOrchestrator(org_id="...", project_id="...") as browser:
        await browser.navigate("https://example.com")
        await browser.click("#login-button")  # auto-heals if selector fails
        await browser.fill("#email", "test@example.com")

        # DevTools-only capabilities
        requests = await browser.get_network_requests()
        console = await browser.get_console_messages()
        perf = await browser.capture_performance()
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from .devtools_mcp import (
    ConsoleMessage,
    DevToolsMCPClient,
    DevToolsMCPConfig,
    NetworkRequest,
    PerformanceMetrics,
)
from .playwright_mcp import MCPServerConfig, MCPToolResult, PlaywrightMCPClient

logger = structlog.get_logger()


@dataclass
class HealingResult:
    """Result of a self-healing attempt."""

    success: bool
    original_selector: str
    healed_selector: str | None = None
    confidence: float = 0.0
    strategy: str | None = None
    error: str | None = None


@dataclass
class ActionEvent:
    """Event emitted for each browser action (for Kafka)."""

    action: str
    selector: str | None = None
    value: str | None = None
    url: str | None = None
    success: bool = True
    duration_ms: float = 0.0
    healed: bool = False
    healed_selector: str | None = None
    error: str | None = None
    diagnostics: dict | None = None


@dataclass
class OrchestratorConfig:
    """Configuration for the MCP Browser Orchestrator."""

    # Backend API for self-healing
    backend_url: str = field(
        default_factory=lambda: os.environ.get(
            "ARGUS_BRAIN_URL",
            "https://argus-brain-production.up.railway.app",
        )
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("ARGUS_API_KEY", "")
    )

    # Organization/project context
    org_id: str = ""
    project_id: str = ""

    # Which MCP servers to use
    use_playwright: bool = True
    use_devtools: bool = True

    # Self-healing settings
    enable_healing: bool = True
    healing_max_alternatives: int = 5

    # Event emission
    emit_events: bool = True

    # Diagnostic capture on failure
    capture_diagnostics_on_failure: bool = True


class MCPBrowserOrchestrator:
    """Unified browser control combining Playwright MCP + DevTools MCP.

    This orchestrator:
    1. Uses Playwright MCP for primary browser actions (click, fill, navigate)
    2. Uses DevTools MCP for diagnostics (network, console, performance)
    3. Self-heals failed selector actions via the backend healing API
    4. Emits events to Kafka for the AI learning loop
    """

    def __init__(self, config: OrchestratorConfig | None = None, **kwargs):
        self.config = config or OrchestratorConfig(**kwargs)
        self._playwright: PlaywrightMCPClient | None = None
        self._devtools: DevToolsMCPClient | None = None
        self._current_url: str = ""
        self._action_log: list[ActionEvent] = []
        self.log = logger.bind(component="mcp_browser_orchestrator")

    async def start(self) -> None:
        """Start both MCP server processes."""
        if self.config.use_playwright:
            self._playwright = PlaywrightMCPClient()
            await self._playwright.start()
            self.log.info("Playwright MCP started")

        if self.config.use_devtools:
            self._devtools = DevToolsMCPClient()
            await self._devtools.start()
            self.log.info("Chrome DevTools MCP started")

    async def stop(self) -> None:
        """Stop both MCP server processes."""
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        if self._devtools:
            await self._devtools.stop()
            self._devtools = None

        self.log.info("MCP browser orchestrator stopped")

    # ── Navigation ──────────────────────────────────────────────

    async def navigate(self, url: str) -> MCPToolResult:
        """Navigate to a URL (uses Playwright MCP, syncs DevTools)."""
        start = time.monotonic()
        result = await self._pw().navigate(url)
        elapsed = (time.monotonic() - start) * 1000

        self._current_url = url

        # Also navigate DevTools to same URL for diagnostic capture
        if self._devtools:
            await self._devtools.navigate(url)

        self._emit(ActionEvent(
            action="navigate", url=url, success=result.success,
            duration_ms=elapsed, error=result.error,
        ))

        return result

    async def go_back(self) -> MCPToolResult:
        """Navigate back in history."""
        return await self._pw().call_tool("browser_navigate_back", {})

    async def screenshot(self) -> MCPToolResult:
        """Take a screenshot (uses Playwright MCP)."""
        return await self._pw().screenshot()

    async def get_current_url(self) -> str:
        """Get the current page URL."""
        if self._devtools:
            info = await self._devtools.get_page_info()
            if info.success:
                content = self._devtools._extract_first_content(info)
                if isinstance(content, dict):
                    return content.get("url", self._current_url)
        return self._current_url

    # ── Actions with Self-Healing ───────────────────────────────

    async def click(self, selector: str) -> MCPToolResult:
        """Click an element, with self-healing if the selector fails."""
        return await self._action_with_healing("click", selector)

    async def fill(self, selector: str, value: str) -> MCPToolResult:
        """Fill a form field, with self-healing if the selector fails."""
        return await self._action_with_healing("fill", selector, value=value)

    async def hover(self, selector: str) -> MCPToolResult:
        """Hover over an element, with self-healing if the selector fails."""
        return await self._action_with_healing("hover", selector)

    async def select_option(self, selector: str, value: str) -> MCPToolResult:
        """Select from dropdown, with self-healing if the selector fails."""
        return await self._action_with_healing("select", selector, value=value)

    async def type_text(self, selector: str, text: str) -> MCPToolResult:
        """Type text into an element, with self-healing if the selector fails."""
        # First click the element, then type
        click_result = await self._action_with_healing("click", selector)
        if not click_result.success:
            return click_result
        return await self._pw().call_tool("browser_type", {"text": text})

    async def press_key(self, key: str) -> MCPToolResult:
        """Press a keyboard key."""
        return await self._pw().call_tool("browser_press_key", {"key": key})

    async def wait_for_selector(
        self, selector: str, timeout_ms: int | None = None
    ) -> MCPToolResult:
        """Wait for an element to appear."""
        if self._devtools:
            result = await self._devtools.wait(selector=selector, timeout_ms=timeout_ms)
            return MCPToolResult(
                success=result.success, content=result.content, error=result.error,
            )
        # Fallback: use Playwright's snapshot + evaluate
        return await self._pw().call_tool("browser_evaluate", {
            "script": f"document.querySelector('{selector}') !== null",
        })

    # ── JavaScript Execution ────────────────────────────────────

    async def evaluate(self, script: str) -> Any:
        """Execute JavaScript in the browser."""
        result = await self._pw().evaluate(script)
        return result.content if result.success else None

    # ── DevTools-Only: Network Inspection ───────────────────────

    async def get_network_requests(self) -> list[NetworkRequest]:
        """Get all captured network requests (DevTools MCP only)."""
        if not self._devtools:
            self.log.warning("DevTools MCP not available for network inspection")
            return []
        return await self._devtools.get_network_requests()

    async def get_network_request_detail(self, request_id: str) -> NetworkRequest | None:
        """Get full request/response details including body."""
        if not self._devtools:
            return None
        return await self._devtools.get_network_request_detail(request_id)

    # ── DevTools-Only: Console & Debugging ──────────────────────

    async def get_console_messages(self) -> list[ConsoleMessage]:
        """Get console messages with source-mapped stack traces."""
        if not self._devtools:
            return []
        return await self._devtools.get_console_messages()

    async def get_console_errors(self) -> list[ConsoleMessage]:
        """Get only console error messages."""
        messages = await self.get_console_messages()
        return [m for m in messages if m.level == "error"]

    # ── DevTools-Only: Performance ──────────────────────────────

    async def capture_performance(self, url: str | None = None) -> PerformanceMetrics:
        """Capture Core Web Vitals for a page.

        If url is provided, navigates there first. Otherwise measures current page.
        """
        if not self._devtools:
            return PerformanceMetrics()

        await self._devtools.start_performance_trace()

        if url:
            await self.navigate(url)
        else:
            # Reload to capture fresh metrics
            await self._devtools.evaluate("location.reload()")

        # Wait for page to settle
        await self._devtools.wait(timeout_ms=3000)

        return await self._devtools.stop_performance_trace()

    # ── DevTools-Only: Emulation ────────────────────────────────

    async def emulate_device(self, device: str) -> bool:
        """Emulate a device (e.g., 'iPhone 14', 'Pixel 7')."""
        if not self._devtools:
            return False
        result = await self._devtools.set_device(device)
        return result.success

    async def set_geolocation(self, latitude: float, longitude: float) -> bool:
        """Set device geolocation for location-based testing."""
        if not self._devtools:
            return False
        result = await self._devtools.set_geolocation(latitude, longitude)
        return result.success

    # ── Diagnostic Capture ──────────────────────────────────────

    async def capture_diagnostics(self) -> dict:
        """Capture a full diagnostic snapshot (network, console, performance).

        Useful for debugging test failures.
        """
        diagnostics: dict[str, Any] = {"timestamp": time.time(), "url": self._current_url}

        if self._devtools:
            # Console errors
            errors = await self.get_console_errors()
            diagnostics["console_errors"] = [
                {"level": e.level, "text": e.text, "stack": e.stack_trace}
                for e in errors
            ]

            # Failed network requests (4xx/5xx)
            requests = await self.get_network_requests()
            failed_requests = [
                {"url": r.url, "status": r.status, "method": r.method}
                for r in requests
                if r.status and r.status >= 400
            ]
            diagnostics["failed_requests"] = failed_requests

            # Page info
            page_info = await self._devtools.get_page_info()
            if page_info.success:
                diagnostics["page_info"] = self._devtools._extract_first_content(page_info)

        return diagnostics

    # ── Action Log ──────────────────────────────────────────────

    @property
    def action_log(self) -> list[ActionEvent]:
        """Get the log of all actions taken in this session."""
        return list(self._action_log)

    def clear_action_log(self) -> None:
        """Clear the action log."""
        self._action_log.clear()

    # ── Self-Healing Core ───────────────────────────────────────

    async def _action_with_healing(
        self,
        action: str,
        selector: str,
        value: str | None = None,
    ) -> MCPToolResult:
        """Execute a browser action with self-healing on failure.

        1. Try the original selector via Playwright MCP
        2. If it fails and healing is enabled, query the backend healing API
        3. Try each suggested alternative selector
        4. Emit an event with the result (for AI learning loop)
        """
        start = time.monotonic()

        # Step 1: Try original selector
        result = await self._execute_action(action, selector, value)
        elapsed = (time.monotonic() - start) * 1000

        if result.success:
            self._emit(ActionEvent(
                action=action, selector=selector, value=value,
                url=self._current_url, success=True, duration_ms=elapsed,
            ))
            return result

        # Step 2: Self-healing
        if not self.config.enable_healing:
            self._emit(ActionEvent(
                action=action, selector=selector, value=value,
                url=self._current_url, success=False, duration_ms=elapsed,
                error=result.error,
            ))
            return result

        self.log.info(
            "Action failed, attempting self-healing",
            action=action, selector=selector, error=result.error,
        )

        healing = await self._request_healing(selector, result.error or "Element not found")

        if not healing.success:
            # Capture diagnostics on healing failure
            diagnostics = None
            if self.config.capture_diagnostics_on_failure:
                diagnostics = await self.capture_diagnostics()

            elapsed = (time.monotonic() - start) * 1000
            self._emit(ActionEvent(
                action=action, selector=selector, value=value,
                url=self._current_url, success=False, duration_ms=elapsed,
                error=result.error, diagnostics=diagnostics,
            ))
            return result

        # Step 3: Try healed selector
        healed_result = await self._execute_action(action, healing.healed_selector, value)
        elapsed = (time.monotonic() - start) * 1000

        if healed_result.success:
            self.log.info(
                "Self-healing succeeded",
                original=selector,
                healed=healing.healed_selector,
                confidence=healing.confidence,
                strategy=healing.strategy,
            )

        self._emit(ActionEvent(
            action=action, selector=selector, value=value,
            url=self._current_url,
            success=healed_result.success,
            duration_ms=elapsed,
            healed=healed_result.success,
            healed_selector=healing.healed_selector,
            error=healed_result.error if not healed_result.success else None,
        ))

        return healed_result

    async def _execute_action(
        self, action: str, selector: str, value: str | None = None
    ) -> MCPToolResult:
        """Execute a raw browser action via Playwright MCP."""
        pw = self._pw()

        if action == "click":
            return await pw.click(selector)
        elif action == "fill":
            return await pw.fill(selector, value or "")
        elif action == "hover":
            return await pw.hover(selector)
        elif action == "select":
            return await pw.select(selector, value or "")
        else:
            return MCPToolResult(success=False, content=None, error=f"Unknown action: {action}")

    async def _request_healing(
        self, failed_selector: str, error_message: str
    ) -> HealingResult:
        """Request self-healing from the backend healing API."""
        if not self.config.api_key or not self.config.org_id:
            return HealingResult(
                success=False, original_selector=failed_selector,
                error="No API key or org_id configured for healing",
            )

        try:
            import httpx

            url = (
                f"{self.config.backend_url}/api/v1/healing"
                f"/organizations/{self.config.org_id}/suggest"
            )
            payload = {
                "project_id": self.config.project_id,
                "failed_selector": failed_selector,
                "error_message": error_message,
                "page_url": self._current_url,
                "max_alternatives": self.config.healing_max_alternatives,
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "X-API-Key": self.config.api_key,
                        "Content-Type": "application/json",
                    },
                )

            if response.status_code != 200:
                return HealingResult(
                    success=False, original_selector=failed_selector,
                    error=f"Healing API returned {response.status_code}",
                )

            data = response.json()
            alternatives = data.get("alternatives", [])

            if not alternatives:
                return HealingResult(
                    success=False, original_selector=failed_selector,
                    error="No alternative selectors suggested",
                )

            # Return the highest-confidence alternative
            best = max(alternatives, key=lambda a: a.get("confidence", 0))
            return HealingResult(
                success=True,
                original_selector=failed_selector,
                healed_selector=best.get("selector"),
                confidence=best.get("confidence", 0.0),
                strategy=best.get("strategy"),
            )

        except Exception as e:
            self.log.warning("Healing request failed", error=str(e))
            return HealingResult(
                success=False, original_selector=failed_selector,
                error=str(e),
            )

    # ── Event Emission ──────────────────────────────────────────

    def _emit(self, event: ActionEvent) -> None:
        """Record action event and optionally emit to Kafka."""
        self._action_log.append(event)

        if not self.config.emit_events:
            return

        # Emit to Kafka asynchronously (fire-and-forget)
        try:
            import asyncio
            asyncio.get_event_loop().create_task(self._emit_to_kafka(event))
        except RuntimeError:
            pass  # No event loop — skip emission

    async def _emit_to_kafka(self, event: ActionEvent) -> None:
        """Emit action event to Kafka for the AI learning loop."""
        try:
            from src.events.producer import get_event_producer

            producer = get_event_producer()
            topic = "argus.test.executed" if event.success else "argus.test.failed"

            await producer.send(
                topic=topic,
                event={
                    "type": "browser_action",
                    "action": event.action,
                    "selector": event.selector,
                    "value": event.value,
                    "url": event.url,
                    "success": event.success,
                    "duration_ms": event.duration_ms,
                    "healed": event.healed,
                    "healed_selector": event.healed_selector,
                    "error": event.error,
                    "org_id": self.config.org_id,
                    "project_id": self.config.project_id,
                },
            )

            # If healing occurred, also emit healing event for learning
            if event.healed:
                await producer.send(
                    topic="argus.healing.completed",
                    event={
                        "type": "selector_healed",
                        "original_selector": event.selector,
                        "healed_selector": event.healed_selector,
                        "page_url": event.url,
                        "org_id": self.config.org_id,
                        "project_id": self.config.project_id,
                    },
                )

        except ImportError:
            pass  # Kafka producer not available
        except Exception as e:
            self.log.debug("Kafka emission failed (non-critical)", error=str(e))

    # ── Helpers ──────────────────────────────────────────────────

    def _pw(self) -> PlaywrightMCPClient:
        """Get the Playwright MCP client, raising if not started."""
        if not self._playwright:
            raise RuntimeError(
                "Playwright MCP not started. Use 'async with MCPBrowserOrchestrator()' "
                "or call start() first."
            )
        return self._playwright

    async def __aenter__(self) -> "MCPBrowserOrchestrator":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
