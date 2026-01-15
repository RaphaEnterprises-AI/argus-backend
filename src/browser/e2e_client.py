"""
E2E Browser Client - AI-Powered Browser Automation via Cloudflare Worker

.. deprecated::
    This client is maintained for backward compatibility.
    Use `BrowserPoolClient` instead for better scalability and reliability.

    Migration guide: docs/BROWSER_POOL_MIGRATION.md

    Before:
        from src.browser import E2EBrowserClient
        async with E2EBrowserClient() as client:
            page = await client.new_page(url)
            await page.act("Click Sign In")

    After:
        from src.browser import BrowserPoolClient
        async with BrowserPoolClient() as client:
            await client.act(url, "Click Sign In")

Connects to our custom Cloudflare Worker that provides:
- Multi-backend support: Cloudflare Browser (free) + TestingBot (cross-browser)
- Cross-browser testing: Chrome, Firefox, Safari, Edge
- Real device testing: iOS and Android devices
- AI-powered actions: Natural language browser automation
- Self-healing selectors: Auto-fixes broken selectors

USAGE:
    async with E2EBrowserClient() as client:
        page = await client.new_page("https://example.com")

        # Natural language actions
        await page.act("Click the Sign In button")
        await page.act("Type 'user@example.com' in the email field")
        await page.act("Click Submit")

        # Structured data extraction
        user_data = await page.extract({
            "username": "string",
            "email": "string"
        })

        # AI observation
        forms = await page.observe("What forms are visible on this page?")
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
import httpx
import structlog

from ..config import get_settings, InferenceGateway
from ..core.model_router import ModelRouter, TaskType


logger = structlog.get_logger()


class BrowserAction(str, Enum):
    """Browser automation action types."""
    ACT = "act"          # Perform an action (click, type, navigate)
    EXTRACT = "extract"  # Extract structured data from page
    OBSERVE = "observe"  # AI observation of page state


@dataclass
class ActionResult:
    """Result from a E2E Browser action."""
    success: bool
    action: BrowserAction
    instruction: str
    result: Optional[Any] = None
    error: Optional[str] = None
    cached: bool = False  # Whether selector was cached
    healed: bool = False  # Whether self-healing was triggered
    duration_ms: int = 0
    tokens_used: int = 0
    model_used: Optional[str] = None


@dataclass
class ExtractionSchema:
    """Schema for structured data extraction."""
    fields: dict[str, str]  # field_name -> type ("string", "number", "boolean", "array")

    def to_zod_schema(self) -> dict:
        """Convert to Zod-compatible schema for E2E Browser."""
        type_mapping = {
            "string": {"type": "string"},
            "number": {"type": "number"},
            "boolean": {"type": "boolean"},
            "array": {"type": "array", "items": {"type": "string"}},
        }
        return {
            "type": "object",
            "properties": {
                name: type_mapping.get(t, {"type": "string"})
                for name, t in self.fields.items()
            }
        }


@dataclass
class PageState:
    """Current state of a browser page."""
    url: str
    title: str
    visible_text: Optional[str] = None
    forms: list[dict] = field(default_factory=list)
    buttons: list[str] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    screenshot_base64: Optional[str] = None


class BrowserPage:
    """
    AI-powered browser page with natural language actions.

    Wraps E2E Browser's primitives for our testing use case.
    """

    def __init__(
        self,
        client: "E2EBrowserClient",
        page_id: str,
        url: str,
    ):
        self.client = client
        self.page_id = page_id
        self.url = url
        self._action_history: list[ActionResult] = []
        self.log = logger.bind(page_id=page_id, url=url)

    async def act(
        self,
        instruction: str,
        timeout_ms: int = 10000,
    ) -> ActionResult:
        """
        Perform a natural language action on the page.

        Examples:
            await page.act("Click the login button")
            await page.act("Type 'hello@example.com' in the email field")
            await page.act("Select 'United States' from the country dropdown")
            await page.act("Scroll down to the pricing section")

        Args:
            instruction: Natural language description of action
            timeout_ms: Maximum time to wait for action

        Returns:
            ActionResult with success status and details
        """
        start_time = datetime.now()

        try:
            result = await self.client._execute_action(
                page_id=self.page_id,
                action=BrowserAction.ACT,
                instruction=instruction,
                timeout_ms=timeout_ms,
            )

            duration = int((datetime.now() - start_time).total_seconds() * 1000)

            action_result = ActionResult(
                success=result.get("success", False),
                action=BrowserAction.ACT,
                instruction=instruction,
                result=result.get("action_taken"),
                cached=result.get("cached", False),
                healed=result.get("healed", False),
                duration_ms=duration,
                tokens_used=result.get("tokens_used", 0),
                model_used=result.get("model_used"),
            )

            self._action_history.append(action_result)

            if action_result.healed:
                self.log.info(
                    "Self-healing triggered",
                    instruction=instruction,
                    original_selector=result.get("original_selector"),
                    healed_selector=result.get("healed_selector"),
                )

            return action_result

        except Exception as e:
            self.log.error("Action failed", instruction=instruction, error=str(e))
            return ActionResult(
                success=False,
                action=BrowserAction.ACT,
                instruction=instruction,
                error=str(e),
            )

    async def extract(
        self,
        schema: dict[str, str] | ExtractionSchema,
        instruction: Optional[str] = None,
    ) -> ActionResult:
        """
        Extract structured data from the page.

        Examples:
            # Simple extraction
            data = await page.extract({
                "title": "string",
                "price": "number",
                "in_stock": "boolean"
            })

            # With instruction
            data = await page.extract(
                {"items": "array"},
                instruction="Extract all product names from the grid"
            )

        Args:
            schema: Field names and types to extract
            instruction: Optional hint for extraction

        Returns:
            ActionResult with extracted data
        """
        if isinstance(schema, dict):
            schema = ExtractionSchema(fields=schema)

        start_time = datetime.now()

        try:
            result = await self.client._execute_action(
                page_id=self.page_id,
                action=BrowserAction.EXTRACT,
                instruction=instruction or "Extract the requested data",
                schema=schema.to_zod_schema(),
            )

            duration = int((datetime.now() - start_time).total_seconds() * 1000)

            return ActionResult(
                success=result.get("success", False),
                action=BrowserAction.EXTRACT,
                instruction=instruction or "extract",
                result=result.get("data"),
                duration_ms=duration,
                tokens_used=result.get("tokens_used", 0),
                model_used=result.get("model_used"),
            )

        except Exception as e:
            self.log.error("Extraction failed", error=str(e))
            return ActionResult(
                success=False,
                action=BrowserAction.EXTRACT,
                instruction=instruction or "extract",
                error=str(e),
            )

    async def observe(
        self,
        instruction: str = "What is visible on this page?",
    ) -> ActionResult:
        """
        AI observation of the current page state.

        Examples:
            # General observation
            state = await page.observe("What forms are visible?")

            # Specific checks
            errors = await page.observe("Are there any error messages?")
            nav = await page.observe("What navigation options are available?")

        Args:
            instruction: What to observe/analyze

        Returns:
            ActionResult with AI observation
        """
        start_time = datetime.now()

        try:
            result = await self.client._execute_action(
                page_id=self.page_id,
                action=BrowserAction.OBSERVE,
                instruction=instruction,
            )

            duration = int((datetime.now() - start_time).total_seconds() * 1000)

            return ActionResult(
                success=result.get("success", False),
                action=BrowserAction.OBSERVE,
                instruction=instruction,
                result=result.get("observation"),
                duration_ms=duration,
                tokens_used=result.get("tokens_used", 0),
                model_used=result.get("model_used"),
            )

        except Exception as e:
            self.log.error("Observation failed", error=str(e))
            return ActionResult(
                success=False,
                action=BrowserAction.OBSERVE,
                instruction=instruction,
                error=str(e),
            )

    async def get_state(self) -> PageState:
        """Get current page state including URL, title, visible elements."""
        result = await self.client._get_page_state(self.page_id)
        return PageState(**result)

    async def screenshot(self) -> bytes:
        """Capture current page screenshot."""
        return await self.client._capture_screenshot(self.page_id)

    async def goto(self, url: str, wait_until: str = "networkidle") -> ActionResult:
        """Navigate to a URL."""
        return await self.act(f"Navigate to {url}")

    @property
    def action_history(self) -> list[ActionResult]:
        """Get history of all actions performed on this page."""
        return self._action_history.copy()

    def get_stats(self) -> dict:
        """Get statistics about actions on this page."""
        total_actions = len(self._action_history)
        cached_actions = sum(1 for a in self._action_history if a.cached)
        healed_actions = sum(1 for a in self._action_history if a.healed)
        total_tokens = sum(a.tokens_used for a in self._action_history)

        return {
            "total_actions": total_actions,
            "cached_actions": cached_actions,
            "cache_hit_rate": cached_actions / total_actions if total_actions > 0 else 0,
            "healed_actions": healed_actions,
            "heal_rate": healed_actions / total_actions if total_actions > 0 else 0,
            "total_tokens": total_tokens,
        }


class E2EBrowserClient:
    """
    Client for E2E Browser browser automation via Cloudflare Worker.

    This client connects to our deployed Cloudflare Worker that runs E2E Browser.
    The Worker handles browser sessions and AI-powered automation.

    Deployment options:
    1. Deploy the cloudflare-worker/ directory to get your Worker URL
    2. Set E2E_WORKER_URL in environment

    Usage:
        async with E2EBrowserClient() as client:
            page = await client.new_page("https://example.com")
            await page.act("Click the Sign In button")
            data = await page.extract({"username": "string"})

    Environment variables:
        E2E_WORKER_URL: URL of deployed Cloudflare Worker
        E2E_API_TOKEN: Optional API token for Worker authentication
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,  # Worker URL (e.g., https://e2e-testing-agent.you.workers.dev)
        api_token: Optional[str] = None,  # API token for Worker auth
        model_provider: Optional[str] = None,  # "openai", "anthropic", "workers-ai"
        cache_enabled: bool = True,
        self_healing_enabled: bool = True,
        timeout_seconds: int = 60,
    ):
        self.settings = get_settings()

        # Worker endpoint - from param, env var, or constructed from account
        self.endpoint = endpoint or self._resolve_endpoint()
        self.api_token = api_token or self._get_env("E2E_API_TOKEN")

        self.model_provider = model_provider
        self.cache_enabled = cache_enabled
        self.self_healing_enabled = self_healing_enabled
        self.timeout_seconds = timeout_seconds

        self._http_client: Optional[httpx.AsyncClient] = None
        self._pages: dict[str, BrowserPage] = {}
        self._connected = False

        self.log = logger.bind(component="e2e_browser_client", endpoint=self.endpoint)

    def _resolve_endpoint(self) -> str:
        """Resolve the E2E Browser Worker endpoint."""
        # Check environment variable first
        env_url = self._get_env("E2E_WORKER_URL")
        if env_url:
            return env_url.rstrip("/")

        # Fallback: construct from Cloudflare account (if deploying standard Worker)
        account_id = self.settings.cloudflare_account_id
        if account_id:
            # Default subdomain pattern - user should override via env var
            return f"https://e2e-testing-agent.{account_id}.workers.dev"

        raise ValueError(
            "E2E Browser Worker endpoint not configured. "
            "Set E2E_WORKER_URL environment variable to your deployed Worker URL. "
            "See cloudflare-worker/README.md for deployment instructions."
        )

    def _get_env(self, key: str) -> Optional[str]:
        """Get environment variable safely."""
        import os
        return os.environ.get(key)

    def _get_headers(self) -> dict:
        """Get request headers for Worker API."""
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    async def __aenter__(self) -> "E2EBrowserClient":
        """Async context manager entry."""
        await self._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._disconnect()

    async def _connect(self):
        """Connect to E2E Browser Worker."""
        self._http_client = httpx.AsyncClient(timeout=self.timeout_seconds)

        # Verify Worker is reachable
        try:
            response = await self._http_client.get(
                f"{self.endpoint}/health",
                headers=self._get_headers(),
            )
            if response.status_code == 200:
                self._connected = True
                self.log.info("Connected to E2E Browser Worker")
            else:
                raise ConnectionError(f"Worker health check failed: {response.text}")
        except httpx.RequestError as e:
            raise ConnectionError(
                f"Failed to connect to E2E Browser Worker at {self.endpoint}: {e}. "
                "Ensure the Worker is deployed and E2E_WORKER_URL is correct."
            )

    async def _disconnect(self):
        """Disconnect and cleanup."""
        if self._http_client:
            await self._http_client.aclose()
            self._connected = False

    async def new_page(self, url: str) -> BrowserPage:
        """
        Create a new browser page and navigate to URL.

        Note: The E2E Browser Worker uses a stateless per-request model.
        Each page is actually a new test execution on the Worker.

        Args:
            url: URL to navigate to

        Returns:
            BrowserPage for interaction
        """
        if not self._connected:
            raise RuntimeError("Client not connected. Use 'async with' context manager.")

        # Generate a page ID for tracking
        import uuid
        page_id = str(uuid.uuid4())

        page = BrowserPage(client=self, page_id=page_id, url=url)
        self._pages[page_id] = page

        self.log.info("Created page", page_id=page_id, url=url)
        return page

    async def _execute_action(
        self,
        page_id: str,
        action: BrowserAction,
        instruction: str,
        timeout_ms: int = 10000,
        schema: Optional[dict] = None,
        url: Optional[str] = None,
    ) -> dict:
        """Execute a E2E Browser action via the Worker API.

        The Worker exposes different endpoints for each action type:
        - /test: For act actions (steps)
        - /extract: For extraction
        - /observe: For AI observation
        """
        page = self._pages.get(page_id)
        page_url = url or (page.url if page else None)

        if not page_url:
            return {"success": False, "error": "No URL available for page"}

        try:
            if action == BrowserAction.ACT:
                # Use /test endpoint with single step
                response = await self._http_client.post(
                    f"{self.endpoint}/test",
                    json={
                        "url": page_url,
                        "steps": [instruction],
                        "timeout": timeout_ms,
                    },
                    headers=self._get_headers(),
                    timeout=timeout_ms / 1000 + 10,
                )

                if response.status_code == 200:
                    data = response.json()
                    step_result = data.get("steps", [{}])[0] if data.get("steps") else {}
                    return {
                        "success": step_result.get("success", False),
                        "action_taken": instruction,
                        "cached": data.get("stats", {}).get("cachedActions", 0) > 0,
                        "healed": data.get("stats", {}).get("healedActions", 0) > 0,
                        "error": step_result.get("error"),
                    }
                else:
                    return {"success": False, "error": response.text}

            elif action == BrowserAction.EXTRACT:
                # Use /extract endpoint
                payload = {
                    "url": page_url,
                    "schema": schema or {},
                }
                if instruction:
                    payload["instruction"] = instruction

                response = await self._http_client.post(
                    f"{self.endpoint}/extract",
                    json=payload,
                    headers=self._get_headers(),
                    timeout=timeout_ms / 1000 + 10,
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": data.get("success", False),
                        "data": data.get("data"),
                    }
                else:
                    return {"success": False, "error": response.text}

            elif action == BrowserAction.OBSERVE:
                # Use /observe endpoint
                response = await self._http_client.post(
                    f"{self.endpoint}/observe",
                    json={
                        "url": page_url,
                        "instruction": instruction,
                    },
                    headers=self._get_headers(),
                    timeout=timeout_ms / 1000 + 10,
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": data.get("success", False),
                        "observation": data.get("observation"),
                    }
                else:
                    return {"success": False, "error": response.text}

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

        except httpx.TimeoutException:
            return {"success": False, "error": f"Action timed out after {timeout_ms}ms"}
        except httpx.RequestError as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}

    async def _get_page_state(self, page_id: str) -> dict:
        """Get current state of a page via observation."""
        page = self._pages.get(page_id)
        if not page:
            return {}

        result = await self._execute_action(
            page_id=page_id,
            action=BrowserAction.OBSERVE,
            instruction="Describe the current page state: URL, title, visible forms, buttons, and any error messages.",
            url=page.url,
        )

        return {
            "url": page.url,
            "title": "",  # Would need screenshot analysis for this
            "visible_text": result.get("observation", ""),
        }

    async def _capture_screenshot(self, page_id: str) -> bytes:
        """Capture screenshot of a page.

        Note: This requires a test with screenshot=true.
        """
        page = self._pages.get(page_id)
        if not page:
            return b""

        try:
            response = await self._http_client.post(
                f"{self.endpoint}/test",
                json={
                    "url": page.url,
                    "steps": [],  # No steps, just capture current state
                    "screenshot": True,
                },
                headers=self._get_headers(),
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                screenshot_b64 = data.get("screenshot", "")
                if screenshot_b64:
                    import base64
                    return base64.b64decode(screenshot_b64)
            return b""
        except Exception as e:
            self.log.error("Screenshot capture failed", error=str(e))
            return b""

    async def run_test(
        self,
        url: str,
        steps: list[str],
        extract_schema: Optional[dict] = None,
        screenshot: bool = False,
        timeout_ms: int = 30000,
    ) -> dict:
        """Run a complete test via the Worker.

        This is the most efficient way to run multi-step tests, as it
        executes all steps in a single Worker invocation.

        Args:
            url: Starting URL
            steps: List of natural language instructions
            extract_schema: Optional schema for data extraction after steps
            screenshot: Whether to capture final screenshot
            timeout_ms: Overall timeout for the test

        Returns:
            Dict with test results
        """
        payload = {
            "url": url,
            "steps": steps,
            "screenshot": screenshot,
            "timeout": timeout_ms,
        }

        if extract_schema:
            payload["extract"] = extract_schema

        try:
            response = await self._http_client.post(
                f"{self.endpoint}/test",
                json=payload,
                headers=self._get_headers(),
                timeout=timeout_ms / 1000 + 10,
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text, "steps": []}

        except httpx.TimeoutException:
            return {"success": False, "error": "Test timed out", "steps": []}
        except httpx.RequestError as e:
            return {"success": False, "error": str(e), "steps": []}


# Convenience functions for quick usage

async def run_test_with_e2e_client(
    url: str,
    steps: list[str],
    extract_schema: Optional[dict] = None,
) -> dict:
    """
    Quick helper to run a test with E2E Browser.

    Args:
        url: Starting URL
        steps: List of natural language instructions
        extract_schema: Optional schema for final data extraction

    Returns:
        Dict with results and stats

    Example:
        result = await run_test_with_e2e_client(
            url="https://example.com/login",
            steps=[
                "Click Sign In button",
                "Type 'test@example.com' in email field",
                "Type 'password' in password field",
                "Click Submit",
            ],
            extract_schema={"welcome_message": "string"}
        )
    """
    async with E2EBrowserClient() as client:
        page = await client.new_page(url)

        results = []
        for step in steps:
            result = await page.act(step)
            results.append(result)
            if not result.success:
                break

        extracted = None
        if extract_schema and all(r.success for r in results):
            extraction = await page.extract(extract_schema)
            extracted = extraction.result

        return {
            "success": all(r.success for r in results),
            "steps": [
                {
                    "instruction": r.instruction,
                    "success": r.success,
                    "cached": r.cached,
                    "healed": r.healed,
                    "error": r.error,
                }
                for r in results
            ],
            "extracted_data": extracted,
            "stats": page.get_stats(),
        }
