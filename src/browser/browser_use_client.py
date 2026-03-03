"""AI-native browser client using Browser Use + Steel.dev.

Architecture:
- Steel.dev provides cloud browser sessions (CDP-based, scales horizontally)
  Self-hosted on Fly.io — same infra as the rest of Argus, no extra cost.
- Browser Use wraps Playwright with an LLM agent loop for NL automation
- Auth login: Agent-driven (handles any form — Clerk, Auth0, custom, SSO)
- Discovery: Playwright-direct BFS using accessibility tree (no CSS selectors)

This replaces SeleniumGridClient for:
  1. Pre-flight auth login (150 lines of brittle CSS → 20 lines of NL instruction)
  2. Page discovery (WebDriver BFS → Playwright CDP BFS with a11y tree)

Selenium Grid is kept for VIDEO RECORDING only (it has the ffmpeg sidecars).

Configuration:
  STEEL_API_KEY=...       # Steel API key (or "local" for self-hosted without auth)
  STEEL_API_URL=...       # Steel API URL (default: https://api.steel.dev)
                          # Point to your self-hosted Steel instance on Fly.io

If neither is configured, falls back to local Playwright (dev/testing only).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BrowserUseDiscoveryResult:
    """Result from AI-native browser discovery."""

    success: bool
    pages: list[dict] = field(default_factory=list)
    session_id: str | None = None
    recording_url: str | None = None
    error: str | None = None
    duration_ms: int = 0


class BrowserUseClient:
    """AI-native browser automation using Browser Use + Steel.dev.

    Replaces SeleniumGridClient for auth preflight and page discovery.
    Steel.dev provides the browser session pool — CDP-native, purpose-built
    for AI agents, much lighter than Selenium Grid.

    Usage:
        async with BrowserUseClient() as client:
            # Auth preflight — returns storageState dict
            state = await client.login_and_capture_session(
                "https://app.example.com",
                {"username": "user@example.com", "password": "secret"},
            )

            # Discovery
            result = await client.discover("https://app.example.com", max_pages=10)
    """

    def __init__(
        self,
        steel_api_key: str | None = None,
        steel_api_url: str | None = None,
    ) -> None:
        from src.config import get_settings

        settings = get_settings()
        self.steel_api_key = steel_api_key or getattr(settings, "steel_api_key", None)
        self.steel_api_url = (
            steel_api_url
            or getattr(settings, "steel_api_url", None)
            or "https://api.steel.dev"
        )
        self._steel_client = None
        self._active_session_id: str | None = None

    async def __aenter__(self) -> "BrowserUseClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        """Release any active Steel.dev session."""
        if self._active_session_id and self._steel_client:
            try:
                self._steel_client.sessions.release(self._active_session_id)
                logger.debug("Steel session released", session_id=self._active_session_id)
            except Exception as e:
                logger.debug("Failed to release Steel session", error=str(e))
            self._active_session_id = None

    # ── Steel.dev session management ─────────────────────────────────────

    def _get_steel_client(self):
        """Lazy-initialize Steel SDK client (optional dependency)."""
        if self._steel_client is not None:
            return self._steel_client
        if not self.steel_api_key:
            return None
        try:
            from steel import Steel

            self._steel_client = Steel(
                steel_api_key=self.steel_api_key,
                base_url=self.steel_api_url,
            )
        except ImportError:
            logger.debug("steel-sdk not installed, running without Steel.dev")
        return self._steel_client

    async def _create_steel_session(self) -> tuple[str | None, str | None]:
        """Create a Steel.dev browser session.

        Returns (session_id, cdp_url) or (None, None) if Steel unavailable.
        The CDP URL is passed to Playwright's connect_over_cdp() or to
        Browser Use's BrowserConfig(cdp_url=...).
        """
        client = self._get_steel_client()
        if not client:
            return None, None
        try:
            session = await asyncio.get_event_loop().run_in_executor(
                None, client.sessions.create
            )
            self._active_session_id = session.id
            # Steel exposes both a Selenium URL and a CDP (Playwright) URL
            cdp_url = (
                getattr(session, "cdp_url", None)
                or getattr(session, "connect_url", None)
            )
            logger.info(
                "Steel.dev session created",
                session_id=session.id,
                cdp_url=cdp_url,
            )
            return session.id, cdp_url
        except Exception as e:
            logger.warning(
                "Steel.dev session creation failed, falling back to local Playwright",
                error=str(e),
            )
            return None, None

    # ── Auth preflight login ──────────────────────────────────────────────

    async def login_and_capture_session(
        self,
        target_url: str,
        credentials: dict,
    ) -> dict | None:
        """Log in to target app and return Playwright storageState dict.

        Uses Browser Use Agent (LLM-driven) to handle any login form layout
        automatically — Clerk multi-step, Auth0, Okta, custom forms, SSO.

        The agent is given a natural-language task and uses Claude Haiku to
        navigate the form. After completion, the full browser context state
        (cookies + localStorage) is captured and returned in Playwright
        storageState format for injection into downstream agent sessions.

        Returns:
            storageState dict on success, None on failure.
        """
        username = credentials.get("username", "")
        password = credentials.get("password", "")
        auth_url = credentials.get("auth_url") or target_url

        if not username or not password:
            logger.warning("No credentials provided for login")
            return None

        session_id, cdp_url = await self._create_steel_session()

        try:
            from browser_use import Agent, Browser, BrowserConfig
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            logger.warning(
                "browser-use not installed, cannot perform AI-driven login",
                missing=str(e),
                hint="pip install browser-use",
            )
            return None

        try:
            if cdp_url:
                browser_config = BrowserConfig(cdp_url=cdp_url, headless=True)
            else:
                # Dev/local fallback — no Steel
                browser_config = BrowserConfig(headless=True)

            browser = Browser(config=browser_config)

            # Haiku is cheap and sufficient for form-filling
            llm = ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                timeout=30,
                max_tokens=1024,
            )

            task = (
                f"Log in to {auth_url} using the email/username '{username}' "
                f"and password '{password}'. "
                "Complete the full login flow including any multi-step forms "
                "(e.g. email first then Continue, then password then Sign In). "
                "After successfully logging in and the dashboard/app loads, STOP. "
                "Do NOT browse around or take any further actions."
            )

            agent = Agent(
                task=task,
                llm=llm,
                browser=browser,
                max_actions_per_step=6,
            )

            await agent.run(max_steps=20)

            # Extract storageState from the live browser context
            storage_state: dict | None = None
            try:
                # browser-use exposes the Playwright context via agent.browser
                ctx = (
                    getattr(agent.browser, "context", None)
                    or getattr(agent.browser, "_context", None)
                    or getattr(agent.browser, "playwright_context", None)
                )
                if ctx is not None:
                    storage_state = await ctx.storage_state()
                    logger.info(
                        "Auth session captured via Browser Use",
                        session_id=session_id,
                        cookies=len(storage_state.get("cookies", [])),
                        origins=len(storage_state.get("origins", [])),
                    )
                else:
                    logger.warning("Could not access browser context from Browser Use agent")
            finally:
                try:
                    await agent.browser.close()
                except Exception:
                    pass

            return storage_state

        except Exception as e:
            logger.warning("Browser Use auth login failed", error=str(e))
            return None
        finally:
            await self.close()

    # ── Page discovery ────────────────────────────────────────────────────

    async def discover(
        self,
        start_url: str,
        max_pages: int = 10,
        max_depth: int = 2,
        storage_state: dict | None = None,
        storage_state_path: str | None = None,
    ) -> BrowserUseDiscoveryResult:
        """Discover pages via BFS crawl using Playwright connected to Steel.dev.

        Uses the accessibility tree (ARIA tree snapshot) for element extraction
        instead of brittle CSS selectors — survives DOM restructuring, works on
        React/Vue/Angular SPAs, and provides semantic understanding of page content.

        Auth session injection:
        - If storage_state dict provided: cookies injected directly into context
        - If storage_state_path provided: loaded from file first
        - Either way, localStorage is injected after navigating to the domain

        Args:
            start_url: Starting URL for the BFS crawl
            max_pages: Maximum number of pages to visit
            max_depth: Maximum link-following depth (0 = start URL only)
            storage_state: Auth session dict (from login_and_capture_session)
            storage_state_path: Path to storageState JSON file (legacy compat)

        Returns:
            BrowserUseDiscoveryResult with discovered pages and accessibility data
        """
        start_time = time.time()

        # Load storageState from file if path given (legacy compatibility)
        if storage_state_path and storage_state is None:
            try:
                with open(storage_state_path) as f:
                    storage_state = json.load(f)
            except Exception as e:
                logger.warning(
                    "Failed to load storage state file",
                    path=storage_state_path,
                    error=str(e),
                )

        session_id, cdp_url = await self._create_steel_session()

        try:
            from playwright.async_api import async_playwright

            base_domain = urlparse(start_url).netloc
            visited: set[str] = set()
            queue: list[tuple[str, int]] = [(start_url, 0)]
            pages: list[dict] = []

            async with async_playwright() as pw:
                # Connect to Steel.dev via CDP or launch local browser
                if cdp_url:
                    browser = await pw.chromium.connect_over_cdp(cdp_url)
                    context = (
                        browser.contexts[0]
                        if browser.contexts
                        else await browser.new_context()
                    )
                else:
                    browser = await pw.chromium.launch(headless=True)
                    context = await browser.new_context()

                # Inject auth cookies into the context before any navigation
                if storage_state:
                    cookies = storage_state.get("cookies", [])
                    if cookies:
                        try:
                            await context.add_cookies(cookies)
                            logger.info(
                                "Auth cookies injected into browser context",
                                count=len(cookies),
                            )
                        except Exception as e:
                            logger.warning("Cookie injection failed", error=str(e))

                page = await context.new_page()

                # Inject localStorage: must navigate to the target origin first
                if storage_state and storage_state.get("origins"):
                    await page.goto(
                        start_url,
                        wait_until="domcontentloaded",
                        timeout=20000,
                    )
                    for origin_data in storage_state.get("origins", []):
                        for item in origin_data.get("localStorage", []):
                            try:
                                await page.evaluate(
                                    "([k, v]) => window.localStorage.setItem(k, v)",
                                    [item["name"], item["value"]],
                                )
                            except Exception:
                                pass
                    await page.reload(wait_until="domcontentloaded", timeout=20000)
                    # Mark start_url as already loaded
                    visited.add(start_url)
                    queue.pop(0)
                    await self._extract_and_enqueue(
                        page, start_url, 0, max_depth,
                        visited, base_domain, queue, pages,
                    )

                # BFS crawl
                while queue and len(pages) < max_pages:
                    url, depth = queue.pop(0)

                    if url in visited:
                        continue
                    if urlparse(url).netloc != base_domain:
                        continue
                    if depth > max_depth:
                        continue

                    visited.add(url)
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                        await asyncio.sleep(0.3)  # let JS settle
                        await self._extract_and_enqueue(
                            page, url, depth, max_depth,
                            visited, base_domain, queue, pages,
                        )
                    except Exception as e:
                        logger.debug("Page visit failed", url=url, error=str(e))

                await context.close()
                if not cdp_url:
                    await browser.close()

        except Exception as e:
            logger.warning("BrowserUseClient.discover failed", error=str(e))
            return BrowserUseDiscoveryResult(
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000),
            )
        finally:
            await self.close()

        logger.info(
            "Browser discovery complete",
            pages=len(pages),
            session_id=session_id,
            duration_ms=int((time.time() - start_time) * 1000),
        )
        return BrowserUseDiscoveryResult(
            success=True,
            pages=pages,
            session_id=session_id,
            duration_ms=int((time.time() - start_time) * 1000),
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _extract_and_enqueue(
        self,
        page,
        url: str,
        depth: int,
        max_depth: int,
        visited: set[str],
        base_domain: str,
        queue: list[tuple[str, int]],
        pages: list[dict],
    ) -> None:
        """Extract page metadata and enqueue discovered links.

        Uses the accessibility tree for element extraction — semantically
        richer than CSS selectors and works on any modern web framework.
        """
        try:
            title = await page.title()

            # Accessibility tree gives us semantic element info (role, name, etc.)
            # Much richer than raw HTML element enumeration via CSS
            accessibility_tree = await page.accessibility.snapshot(
                interesting_only=True
            )

            # Extract links for BFS
            links: list[str] = await page.eval_on_selector_all(
                "a[href]",
                "els => els.map(e => e.href).filter(h => "
                "h && !h.startsWith('#') && !h.startsWith('javascript:') "
                "&& !h.startsWith('mailto:'))",
            )
            # Cap link list to avoid explosion on link-heavy pages
            links = links[:100]

            pages.append({
                "url": url,
                "title": title,
                "links": links[:20],  # keep payload small
                "accessibility_tree": accessibility_tree,
            })

            # Enqueue new links within same domain
            if depth < max_depth:
                for link in links:
                    if urlparse(link).netloc == base_domain and link not in visited:
                        queue.append((link, depth + 1))

        except Exception as e:
            logger.debug("Page extraction failed", url=url, error=str(e))
            # Still record the URL even if extraction fails
            pages.append({"url": url, "title": "", "links": [], "accessibility_tree": None})


# ── Module-level helpers ──────────────────────────────────────────────────────


def get_browser_use_client() -> BrowserUseClient:
    """Return a BrowserUseClient configured from environment settings."""
    return BrowserUseClient()


async def is_steel_available() -> bool:
    """Return True if Steel.dev is configured and reachable."""
    client = BrowserUseClient()
    steel = client._get_steel_client()
    if not steel:
        return False
    try:
        # Attempt a lightweight health check by creating + immediately releasing a session
        _sid, _url = await client._create_steel_session()
        if _sid:
            await client.close()
            return True
    except Exception:
        pass
    return False
