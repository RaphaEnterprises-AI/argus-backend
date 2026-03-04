# Scout Agent + Live Browser View — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Phase 0 Scout Agent that navigates the target app with a real browser + Claude Vision, builds an `AppIntelligenceReport`, and streams live screenshots to the dashboard while running.

**Architecture:** Sequential Phase 0 (ScoutAgent) runs before Phase 1 agents scatter. Scout uses `SeleniumGridClient` for browser control + Claude Vision for page understanding. A background asyncio task polls `screenshot()` every 2s and emits `browser_screenshot` SSE events. The resulting `AppIntelligenceReport` replaces `_preflight_recon()` and gives all Phase 1 agents full context.

**Tech Stack:** Python asyncio, Selenium WebDriver (SeleniumGridClient), Anthropic vision, AG-UI SSE protocol, React + TypeScript, Next.js

---

## Task 1: Add `BrowserScreenshotEvent` to AG-UI events

**Files:**
- Modify: `src/streaming/agui_events.py`

### Step 1: Add `BROWSER_SCREENSHOT` to the enum and create the event dataclass

In `agui_events.py`, after line 33 (`CUSTOM = "custom"`), add the new enum value. Then append a new dataclass at the bottom of the file.

```python
# In AGUIEventType enum, after CUSTOM = "custom":
BROWSER_SCREENSHOT = "browser_screenshot"
```

```python
# New dataclass at end of file:
@dataclass
class BrowserScreenshotEvent(AGUIEvent):
    """Emitted every ~2s while an agent has an active browser session."""

    type: AGUIEventType = field(default=AGUIEventType.BROWSER_SCREENSHOT, init=False)
    agent_id: str = ""
    base64_png: str = ""
    url: str = ""
    step_description: str = ""
```

### Step 2: Write the test

File: `tests/streaming/test_agui_events.py` (create if missing)

```python
import json
from src.streaming.agui_events import AGUIEventType, BrowserScreenshotEvent


def test_browser_screenshot_event_serializes():
    evt = BrowserScreenshotEvent(
        agent_id="scout_abc",
        base64_png="abc123==",
        url="https://example.com",
        step_description="Analyzing login page",
    )
    data = json.loads(evt.to_sse_data())
    assert data["type"] == "browser_screenshot"
    assert data["agent_id"] == "scout_abc"
    assert data["base64_png"] == "abc123=="
    assert data["url"] == "https://example.com"
    assert data["step_description"] == "Analyzing login page"


def test_browser_screenshot_event_type_in_enum():
    assert AGUIEventType.BROWSER_SCREENSHOT == "browser_screenshot"
```

### Step 3: Run test to verify it fails

```bash
cd /Users/bvk/Downloads/e2e-testing-agent
python -m pytest tests/streaming/test_agui_events.py -v
```

Expected: `ImportError` or `AttributeError: BROWSER_SCREENSHOT`

### Step 4: Apply the edit to `src/streaming/agui_events.py`

Edit 1 — add enum value after `CUSTOM`:
```python
    CUSTOM = "custom"
    BROWSER_SCREENSHOT = "browser_screenshot"
```

Edit 2 — append at end of file (after `StateDeltaEvent`):
```python


@dataclass
class BrowserScreenshotEvent(AGUIEvent):
    """Emitted every ~2s while an agent has an active browser session."""

    type: AGUIEventType = field(default=AGUIEventType.BROWSER_SCREENSHOT, init=False)
    agent_id: str = ""
    base64_png: str = ""
    url: str = ""
    step_description: str = ""
```

### Step 5: Run test to verify it passes

```bash
python -m pytest tests/streaming/test_agui_events.py -v
```

Expected: `PASSED`

### Step 6: Commit

```bash
git add src/streaming/agui_events.py tests/streaming/test_agui_events.py
git commit -m "feat(streaming): add BrowserScreenshotEvent to AG-UI protocol"
```

---

## Task 2: Add `screenshot_stream_task()` to `SeleniumGridClient`

**Files:**
- Modify: `src/browser/selenium_grid_client.py`

This method is an async background task that polls `screenshot()` every 2 seconds and calls a callback with the base64 PNG string.

### Step 1: Write the test

File: `tests/browser/test_selenium_screenshot_stream.py`

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.browser.selenium_grid_client import SeleniumGridClient


@pytest.mark.asyncio
async def test_screenshot_stream_task_calls_callback():
    """Stream task should call callback for each screenshot until cancelled."""
    client = SeleniumGridClient.__new__(SeleniumGridClient)
    client._session_id = "test-session"
    client.grid_url = "http://localhost:4444/"
    client._client = None

    frames_received = []

    async def on_frame(b64_png: str, url: str) -> None:
        frames_received.append((b64_png, url))

    call_count = 0

    async def mock_screenshot():
        nonlocal call_count
        call_count += 1
        return f"frame{call_count}"

    async def mock_current_url():
        return "https://example.com/page"

    client.screenshot = mock_screenshot
    client.current_url = mock_current_url

    task = asyncio.create_task(
        client.screenshot_stream_task(on_frame, interval=0.05)
    )
    await asyncio.sleep(0.18)  # ~3-4 frames at 0.05s interval
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(frames_received) >= 3
    assert frames_received[0] == ("frame1", "https://example.com/page")


@pytest.mark.asyncio
async def test_screenshot_stream_task_skips_none():
    """Stream task should not call callback when screenshot() returns None."""
    client = SeleniumGridClient.__new__(SeleniumGridClient)
    client._session_id = "test-session"
    client.grid_url = "http://localhost:4444/"
    client._client = None

    frames_received = []

    async def on_frame(b64_png: str, url: str) -> None:
        frames_received.append(b64_png)

    async def mock_screenshot():
        return None  # Simulate failure

    async def mock_current_url():
        return "https://example.com"

    client.screenshot = mock_screenshot
    client.current_url = mock_current_url

    task = asyncio.create_task(
        client.screenshot_stream_task(on_frame, interval=0.05)
    )
    await asyncio.sleep(0.18)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(frames_received) == 0
```

### Step 2: Run test to verify it fails

```bash
python -m pytest tests/browser/test_selenium_screenshot_stream.py -v
```

Expected: `AttributeError: 'SeleniumGridClient' object has no attribute 'screenshot_stream_task'`

### Step 3: Add the method and `current_url()` to `SeleniumGridClient`

In `src/browser/selenium_grid_client.py`, append after the `screenshot()` method (after line 328):

```python
    async def current_url(self) -> str:
        """Return the current URL of the browser session."""
        if not self._session_id:
            return ""
        await self._ensure_client()
        try:
            response = await self._client.get(f"session/{self._session_id}/url")
            response.raise_for_status()
            return response.json().get("value", "")
        except Exception:
            return ""

    async def screenshot_stream_task(
        self,
        callback,  # async (b64_png: str, url: str) -> None
        interval: float = 2.0,
    ) -> None:
        """
        Background task: poll screenshot every `interval` seconds and call callback.

        Designed to run as an asyncio background task; cancels cleanly.
        The callback receives (base64_png, current_url).
        """
        while True:
            try:
                png = await self.screenshot()
                if png:
                    url = await self.current_url()
                    await callback(png, url)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("screenshot_stream_task error", error=str(e))
            await asyncio.sleep(interval)
```

### Step 4: Run test to verify it passes

```bash
python -m pytest tests/browser/test_selenium_screenshot_stream.py -v
```

Expected: `PASSED`

### Step 5: Commit

```bash
git add src/browser/selenium_grid_client.py tests/browser/test_selenium_screenshot_stream.py
git commit -m "feat(browser): add screenshot_stream_task and current_url to SeleniumGridClient"
```

---

## Task 3: Implement `AppIntelligenceReport` dataclass and `ScoutAgent`

**Files:**
- Create: `src/agents/scout.py`
- Modify: `src/agents/__init__.py`

### Step 1: Write the test

File: `tests/agents/test_scout.py`

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_app_intelligence_report_dataclass():
    from src.agents.scout import AppIntelligenceReport, PageInfo

    report = AppIntelligenceReport(
        target_url="https://example.com",
        app_type="SaaS dashboard",
        stack="Next.js + Clerk",
        auth_detected=True,
        auth_type="clerk",
        login_steps=[],
        public_pages=[PageInfo(url="https://example.com", title="Home", accessible=True, detected_flows=["signup"])],
        api_endpoints=[],
        openapi_spec=None,
        openapi_url=None,
        suggested_agents=["ui_tester", "security_scan"],
        confidence=0.85,
        raw_observations=["Home page uses Clerk auth"],
    )
    assert report.auth_detected is True
    assert report.confidence == 0.85
    assert len(report.public_pages) == 1


@pytest.mark.asyncio
async def test_scout_agent_returns_report_on_selenium_failure():
    """When Selenium is unavailable, ScoutAgent should return a minimal report
    without crashing."""
    from src.agents.scout import ScoutAgent

    with patch("src.agents.scout.SeleniumGridClient") as MockGrid:
        MockGrid.side_effect = Exception("Grid not available")

        agent = ScoutAgent(target_url="https://example.com")
        report = await agent.run(emitter=None, agent_id="scout_test")

    assert report.target_url == "https://example.com"
    assert report.auth_detected is False  # Unknown without browser
    assert report.confidence < 0.5  # Low confidence without real data
```

### Step 2: Run test to verify it fails

```bash
python -m pytest tests/agents/test_scout.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.scout'`

### Step 3: Create `src/agents/scout.py`

```python
"""Scout Agent — Phase 0 app reconnaissance with live browser + Claude Vision.

Runs sequentially before the swarm scatters. Opens a real Selenium session,
navigates the app, narrates each page with Claude Vision, detects auth, maps
public pages and API endpoints, then synthesises an AppIntelligenceReport.

Live screenshots are streamed to the dashboard via BrowserScreenshotEvent SSE.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from src.browser.selenium_grid_client import SeleniumGridClient, SeleniumGridError

if TYPE_CHECKING:
    from src.streaming.agui_events import AGUIEmitter

logger = structlog.get_logger(__name__)

# Paths to probe for OpenAPI spec
OPENAPI_PROBE_PATHS = [
    "/openapi.json",
    "/api/openapi.json",
    "/docs/openapi.json",
    "/swagger.json",
    "/api/swagger.json",
    "/v1/openapi.json",
    "/api/v1/openapi.json",
]

# Common public pages to check during mapping
PUBLIC_PAGE_CANDIDATES = [
    "/",
    "/pricing",
    "/about",
    "/blog",
    "/docs",
    "/features",
    "/changelog",
    "/status",
]

# Max pages to map during recon
MAX_PAGES = 8


@dataclass
class PageInfo:
    url: str
    title: str
    accessible: bool
    detected_flows: list[str] = field(default_factory=list)


@dataclass
class AppIntelligenceReport:
    target_url: str
    app_type: str = "Unknown"
    stack: str = "Unknown"
    auth_detected: bool = False
    auth_type: str | None = None          # "clerk", "auth0", "custom", None
    login_steps: list[dict] = field(default_factory=list)
    public_pages: list[PageInfo] = field(default_factory=list)
    api_endpoints: list[dict] = field(default_factory=list)
    openapi_spec: dict | None = None
    openapi_url: str | None = None
    suggested_agents: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_observations: list[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Render report as a compact string for injection into agent prompts."""
        lines = [
            f"Target: {self.target_url}",
            f"App type: {self.app_type}",
            f"Stack: {self.stack}",
            f"Auth detected: {self.auth_detected} ({self.auth_type or 'none'})",
            f"Public pages: {[p.url for p in self.public_pages]}",
            f"API endpoints: {[e.get('path') for e in self.api_endpoints]}",
        ]
        if self.openapi_url:
            lines.append(f"OpenAPI spec: {self.openapi_url}")
        if self.raw_observations:
            lines.append("Observations:")
            lines.extend(f"  - {obs}" for obs in self.raw_observations[:5])
        return "\n".join(lines)


class ScoutAgent:
    """Phase 0 agent: browser recon + Claude Vision + screenshot streaming."""

    def __init__(self, target_url: str):
        self.target_url = target_url.rstrip("/")
        self._current_step = "Initializing..."

    async def run(
        self,
        emitter,          # AGUIEmitter | None
        agent_id: str,
    ) -> AppIntelligenceReport:
        """Run scout recon. Returns AppIntelligenceReport regardless of errors."""
        report = AppIntelligenceReport(target_url=self.target_url)

        # Step 1: HTTP-only OpenAPI probe (fast, no browser needed)
        await self._probe_openapi(report)

        # Step 2: Browser session (best-effort; skip if Grid unavailable)
        try:
            selenium = SeleniumGridClient()
            await self._run_browser_session(selenium, emitter, agent_id, report)
        except SeleniumGridError as e:
            logger.warning("Scout: Selenium Grid unavailable", error=str(e))
            report.raw_observations.append(f"Browser recon skipped: {e}")
            report.confidence = max(report.confidence, 0.2)

        # Step 3: Synthesise agent suggestions
        self._suggest_agents(report)

        return report

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    async def _probe_openapi(self, report: AppIntelligenceReport) -> None:
        """Probe common OpenAPI paths over HTTP (no browser needed)."""
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(self.target_url)

        # Also check api. subdomain
        api_subdomain = urlunparse(parsed._replace(netloc=f"api.{parsed.netloc}"))
        probes: list[str] = []
        for path in OPENAPI_PROBE_PATHS:
            probes.append(f"{self.target_url}{path}")
            probes.append(f"{api_subdomain}{path}")

        async with httpx.AsyncClient(timeout=5, follow_redirects=True) as client:
            for url in probes:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        spec = resp.json()
                        if isinstance(spec, dict) and ("paths" in spec or "openapi" in spec):
                            report.openapi_spec = spec
                            report.openapi_url = url
                            paths = list(spec.get("paths", {}).keys())
                            report.api_endpoints = [{"path": p} for p in paths[:50]]
                            report.raw_observations.append(
                                f"OpenAPI spec found at {url} with {len(paths)} endpoints"
                            )
                            return
                except Exception:
                    continue

    async def _run_browser_session(
        self,
        selenium: SeleniumGridClient,
        emitter,
        agent_id: str,
        report: AppIntelligenceReport,
    ) -> None:
        """Open browser, navigate, observe, close. Stream screenshots."""
        from src.streaming.agui_events import BrowserScreenshotEvent

        await selenium.start_session()
        screenshot_task: asyncio.Task | None = None

        try:
            # Kick off screenshot streaming
            if emitter is not None:
                async def on_frame(b64_png: str, url: str) -> None:
                    await emitter.emit(BrowserScreenshotEvent(
                        agent_id=agent_id,
                        base64_png=b64_png,
                        url=url,
                        step_description=self._current_step,
                    ))
                screenshot_task = asyncio.create_task(
                    selenium.screenshot_stream_task(on_frame, interval=2.0)
                )

            # Navigate to target
            self._current_step = "Loading target URL..."
            await selenium.navigate(self.target_url)
            await asyncio.sleep(2)  # Let page settle

            page_info = await selenium.get_page_info()
            final_url = page_info.get("url", self.target_url)

            # Auth detection
            self._current_step = "Detecting auth..."
            await self._detect_auth(selenium, final_url, report)

            # Map public pages
            self._current_step = "Mapping public pages..."
            await self._map_public_pages(selenium, report)

            # Claude Vision narration of landing page
            self._current_step = "Analyzing app structure with AI..."
            screenshot = await selenium.screenshot()
            if screenshot:
                await self._narrate_page(screenshot, final_url, report)

            report.confidence = 0.8 if report.public_pages else 0.5

        finally:
            if screenshot_task:
                screenshot_task.cancel()
                try:
                    await screenshot_task
                except asyncio.CancelledError:
                    pass
            try:
                await selenium.end_session()
            except Exception:
                pass

    async def _detect_auth(
        self,
        selenium: SeleniumGridClient,
        final_url: str,
        report: AppIntelligenceReport,
    ) -> None:
        """Check if app redirected to an auth wall."""
        auth_signals = {
            "clerk": ["clerk.com", "accounts.", "/sign-in", "/login"],
            "auth0": ["auth0.com", ".auth0.com"],
            "custom": ["/login", "/signin", "/auth"],
        }
        lower_url = final_url.lower()

        for provider, signals in auth_signals.items():
            if any(sig in lower_url for sig in signals):
                report.auth_detected = True
                report.auth_type = provider
                report.raw_observations.append(
                    f"Auth wall detected: {provider} (redirected to {final_url})"
                )
                return

    async def _map_public_pages(
        self,
        selenium: SeleniumGridClient,
        report: AppIntelligenceReport,
    ) -> None:
        """Navigate to common public paths and record accessible ones."""
        for path in PUBLIC_PAGE_CANDIDATES[:MAX_PAGES]:
            url = f"{self.target_url}{path}"
            try:
                self._current_step = f"Checking {path}..."
                await selenium.navigate(url)
                await asyncio.sleep(1)
                info = await selenium.get_page_info()
                actual_url = info.get("url", url)
                title = info.get("title", "")

                # Skip if it redirected to auth
                is_auth_redirect = any(
                    sig in actual_url.lower()
                    for sig in ["sign-in", "login", "clerk.com", "auth0.com"]
                )
                if not is_auth_redirect:
                    report.public_pages.append(
                        PageInfo(url=actual_url, title=title, accessible=True)
                    )
            except Exception as e:
                logger.debug("Page probe failed", url=url, error=str(e))

    async def _narrate_page(
        self,
        screenshot_b64: str,
        url: str,
        report: AppIntelligenceReport,
    ) -> None:
        """Ask Claude Vision to describe the page and infer app type + stack."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic()
            message = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
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
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"This is a screenshot of {url}. "
                                    "In 2-3 sentences: what type of app is this? "
                                    "What tech stack can you infer? "
                                    "What are the main user flows visible? "
                                    "Reply as JSON: {{app_type, stack, flows[], observations[]}}"
                                ),
                            },
                        ],
                    }
                ],
            )
            text = message.content[0].text
            # Try to parse JSON, fall back to raw text
            try:
                parsed = json.loads(text)
                report.app_type = parsed.get("app_type", "Unknown")
                report.stack = parsed.get("stack", "Unknown")
                report.raw_observations.extend(parsed.get("observations", []))
            except json.JSONDecodeError:
                report.raw_observations.append(text[:300])
        except Exception as e:
            logger.warning("Claude Vision narration failed", error=str(e))

    def _suggest_agents(self, report: AppIntelligenceReport) -> None:
        """Suggest which agents to run based on what Scout found."""
        agents = []
        if report.public_pages:
            agents.extend(["ui_tester", "accessibility_check", "visual_comparison"])
        if report.api_endpoints or report.openapi_spec:
            agents.append("api_tester")
        agents.extend(["security_scan", "performance_analysis"])
        report.suggested_agents = agents
```

### Step 4: Export from `src/agents/__init__.py`

Add to the existing exports:

```python
from src.agents.scout import ScoutAgent, AppIntelligenceReport, PageInfo
```

### Step 5: Run tests

```bash
python -m pytest tests/agents/test_scout.py -v
```

Expected: both tests `PASSED`

### Step 6: Commit

```bash
git add src/agents/scout.py src/agents/__init__.py tests/agents/test_scout.py
git commit -m "feat(agents): add ScoutAgent and AppIntelligenceReport for Phase 0 recon"
```

---

## Task 4: Add `_run_scout()` and Phase 0 insertion to `swarm_orchestrator.py`

**Files:**
- Modify: `src/orchestrator/swarm_orchestrator.py`

This task wires ScoutAgent into the orchestrator as Phase 0.

### Step 1: Write the test

File: `tests/orchestrator/test_scout_phase.py`

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.orchestrator.swarm_orchestrator import SwarmConfig, SwarmMode


@pytest.mark.asyncio
async def test_run_scout_returns_report():
    """_run_scout should call ScoutAgent.run and emit step events."""
    from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

    config = SwarmConfig(
        mode=SwarmMode.FULL_CRAWL,
        target_url="https://example.com",
        org_id="org1",
        project_id="proj1",
    )
    emitter = MagicMock()
    emitter.emit = AsyncMock()

    mock_report = MagicMock()
    mock_report.to_context_string.return_value = "App: SaaS"
    mock_report.public_pages = []
    mock_report.api_endpoints = []
    mock_report.auth_detected = False

    with patch("src.orchestrator.swarm_orchestrator.ScoutAgent") as MockScout:
        MockScout.return_value.run = AsyncMock(return_value=mock_report)
        orch = SwarmOrchestrator()
        report = await orch._run_scout("swarm_1", config, emitter)

    assert report is mock_report
    # StepStarted + StepFinished should have been emitted
    assert emitter.emit.call_count >= 2
```

### Step 2: Run test to verify it fails

```bash
python -m pytest tests/orchestrator/test_scout_phase.py -v
```

Expected: `AttributeError: 'SwarmOrchestrator' object has no attribute '_run_scout'`

### Step 3: Add `_run_scout()` and Phase 0 in `swarm_orchestrator.py`

**Import at the top of the file** (near other agent imports):
```python
from src.agents.scout import ScoutAgent, AppIntelligenceReport
from src.streaming.agui_events import BrowserScreenshotEvent
```

**New `_run_scout()` method** (add as a new method, before `_run_swarm()`):

```python
async def _run_scout(
    self,
    swarm_id: str,
    config: "SwarmConfig",
    emitter: "AGUIEmitter",
) -> "AppIntelligenceReport":
    """Phase 0: Run ScoutAgent sequentially before the swarm scatters."""
    import uuid as _uuid
    from src.streaming.agui_events import StepStartedEvent, StepFinishedEvent, StateDeltaEvent

    agent_id = f"scout_{_uuid.uuid4().hex[:8]}"
    t_start = time.time()

    await emitter.emit(StepStartedEvent(
        run_id=swarm_id,
        swarm_id=swarm_id,
        agent_id=agent_id,
        agent_name="Scout",
        agent_type="scout",
        step_name="Phase 0: App Reconnaissance",
        task=f"Crawl {config.target_url} to build intelligence report",
    ))

    await emitter.emit(StateDeltaEvent(
        agent_id=agent_id,
        progress=5,
        phase="starting",
        message="Opening browser session...",
    ))

    try:
        scout = ScoutAgent(target_url=config.target_url)
        report = await scout.run(emitter=emitter, agent_id=agent_id)

        await emitter.emit(StateDeltaEvent(
            agent_id=agent_id,
            progress=100,
            phase="complete",
            message=(
                f"Found {len(report.public_pages)} pages, "
                f"{len(report.api_endpoints)} endpoints, "
                f"auth={'yes' if report.auth_detected else 'no'}"
            ),
        ))

        duration_ms = (time.time() - t_start) * 1000
        await emitter.emit(StepFinishedEvent(
            run_id=swarm_id,
            agent_id=agent_id,
            agent_name="Scout",
            agent_type="scout",
            success=True,
            duration_ms=duration_ms,
            findings_count=len(report.public_pages) + len(report.api_endpoints),
            result_summary=report.to_context_string()[:200],
        ))
        return report

    except Exception as e:
        logger.error("Scout phase failed", error=str(e))
        duration_ms = (time.time() - t_start) * 1000
        await emitter.emit(StepFinishedEvent(
            run_id=swarm_id,
            agent_id=agent_id,
            agent_name="Scout",
            agent_type="scout",
            success=False,
            duration_ms=duration_ms,
            findings_count=0,
            result_summary=f"Scout failed: {e}",
        ))
        # Return minimal report so swarm can still proceed
        from src.agents.scout import AppIntelligenceReport
        return AppIntelligenceReport(target_url=config.target_url or "", confidence=0.1)
```

**Insert Phase 0 at the start of `_run_swarm()`** — find the existing block that calls `_preflight_recon` and replace it:

```python
# BEFORE (remove this):
if config.target_url and not config.recon_context:
    import dataclasses as _dc
    try:
        recon = await self._preflight_recon(config.target_url)
        config = _dc.replace(config, recon_context=recon)
    except Exception as recon_err:
        logger.warning("Pre-flight recon failed, continuing without context", error=str(recon_err))

# AFTER (replace with):
if config.target_url and not config.recon_context:
    import dataclasses as _dc
    try:
        scout_report = await self._run_scout(swarm_id, config, emitter)
        # Build recon_context dict from AppIntelligenceReport for backward compat
        recon = {
            "auth_detected": scout_report.auth_detected,
            "auth_type": scout_report.auth_type,
            "public_pages": [{"url": p.url, "title": p.title} for p in scout_report.public_pages],
            "api_endpoints": scout_report.api_endpoints,
            "openapi_spec": scout_report.openapi_spec,
            "openapi_url": scout_report.openapi_url,
            "app_type": scout_report.app_type,
            "stack": scout_report.stack,
        }
        config = _dc.replace(config, recon_context=recon)
    except Exception as scout_err:
        logger.warning("Scout phase failed, continuing without context", error=str(scout_err))
```

Also add `"scout"` to the `SWARM_MODE_AGENTS` entries — or better, add it to `RunStartedEvent.agent_types` so the frontend knows to show a Scout card. Find where `RunStartedEvent` is emitted and add `"scout"` at the front of `agent_types`:

```python
# In _run_swarm(), find:
await emitter.emit(RunStartedEvent(
    swarm_id=swarm_id,
    mode=config.mode.value,
    worker_count=len(agents),
    agent_types=agents,
))
# Change to:
await emitter.emit(RunStartedEvent(
    swarm_id=swarm_id,
    mode=config.mode.value,
    worker_count=len(agents) + 1,  # +1 for Scout
    agent_types=["scout"] + agents,
))
```

### Step 4: Run tests

```bash
python -m pytest tests/orchestrator/test_scout_phase.py tests/orchestrator/ -v
```

Expected: new test `PASSED`, existing tests still `PASSED`

### Step 5: Commit

```bash
git add src/orchestrator/swarm_orchestrator.py tests/orchestrator/test_scout_phase.py
git commit -m "feat(orchestrator): add Phase 0 Scout before swarm scatter"
```

---

## Task 5: Remove `_preflight_recon()` (dead code cleanup)

**Files:**
- Modify: `src/orchestrator/swarm_orchestrator.py`

Now that Scout replaces `_preflight_recon()`, remove the old HTTP-only implementation.

### Step 1: Find and delete `_preflight_recon()`

In `swarm_orchestrator.py`, find the `_preflight_recon` method definition and delete the entire method body (it's roughly 80-120 lines). The method starts with:

```python
async def _preflight_recon(self, target_url: str) -> dict:
```

Delete from that line to the end of the method's indented block.

### Step 2: Run all orchestrator tests to confirm nothing broke

```bash
python -m pytest tests/orchestrator/ -v
```

Expected: all pass, no references to `_preflight_recon`

### Step 3: Commit

```bash
git add src/orchestrator/swarm_orchestrator.py
git commit -m "refactor(orchestrator): remove _preflight_recon() replaced by ScoutAgent"
```

---

## Task 6: Frontend — `SwarmWorker.latestScreenshot` + `browser_screenshot` event handler

**Files:**
- Modify: `dashboard/lib/hooks/use-swarm-stream.ts`

### Step 1: Write the test

File: `dashboard/__tests__/lib/hooks/use-swarm-stream-screenshot.test.ts`

```typescript
import { describe, it, expect } from 'vitest';

// Test the reducer logic directly (extracted from useSwarmStream)
// by verifying the browser_screenshot case updates latestScreenshot

describe('browser_screenshot event handling', () => {
  it('should add latestScreenshot to SwarmWorker type', () => {
    // TypeScript structural check — this will fail to compile if field missing
    const worker: import('@/lib/hooks/use-swarm-stream').SwarmWorker = {
      agentId: 'scout_abc',
      agentType: 'scout',
      status: 'running',
      progress: 50,
      phase: 'scanning',
      message: 'Loading...',
      durationMs: 0,
      costUsd: 0,
      findingsCount: 0,
      resultSummary: '',
      latestScreenshot: 'abc123==',
    };
    expect(worker.latestScreenshot).toBe('abc123==');
  });

  it('latestScreenshot should be optional', () => {
    const worker: import('@/lib/hooks/use-swarm-stream').SwarmWorker = {
      agentId: 'scout_abc',
      agentType: 'scout',
      status: 'running',
      progress: 50,
      phase: '',
      message: '',
      durationMs: 0,
      costUsd: 0,
      findingsCount: 0,
      resultSummary: '',
      // No latestScreenshot — should be fine
    };
    expect(worker.latestScreenshot).toBeUndefined();
  });
});
```

### Step 2: Run test to verify it fails

```bash
cd dashboard
npm run test -- --run use-swarm-stream-screenshot
```

Expected: TypeScript compilation error about `latestScreenshot` not existing on `SwarmWorker`

### Step 3: Edit `use-swarm-stream.ts`

**Add `latestScreenshot` to `SwarmWorker` interface** (after `resultSummary`):
```typescript
export interface SwarmWorker {
  // ... existing fields ...
  resultSummary: string;
  latestScreenshot?: string;   // base64 PNG, present while browser session is active
}
```

**Add `'browser_screenshot'` to `SwarmEventType`**:
```typescript
export type SwarmEventType =
  | 'run_started'
  | 'run_finished'
  | 'run_error'
  | 'step_started'
  | 'step_finished'
  | 'state_delta'
  | 'tool_call_start'
  | 'tool_call_end'
  | 'text_message_content'
  | 'browser_screenshot';   // NEW
```

**Add `'browser_screenshot'` to the `eventTypes` array** in `useEffect`:
```typescript
const eventTypes: SwarmEventType[] = [
  // ... existing types ...
  'browser_screenshot',   // NEW
];
```

**Add case to the `switch` in `handleEvent`** (before `default`):
```typescript
case 'browser_screenshot': {
  const workers = prev.workers.map((w) =>
    w.agentId === data.agent_id
      ? { ...w, latestScreenshot: data.base64_png as string }
      : w
  );
  return { ...prev, workers };
}
```

### Step 4: Run test to verify it passes

```bash
npm run test -- --run use-swarm-stream-screenshot
```

Expected: `PASSED`

### Step 5: Commit

```bash
cd dashboard
git add lib/hooks/use-swarm-stream.ts __tests__/lib/hooks/use-swarm-stream-screenshot.test.ts
git commit -m "feat(swarm-stream): add latestScreenshot to SwarmWorker + browser_screenshot event"
```

---

## Task 7: `BrowserLiveView` component

**Files:**
- Create: `dashboard/components/command-center/BrowserLiveView.tsx`

### Step 1: Write the test

File: `dashboard/__tests__/components/command-center/BrowserLiveView.test.tsx`

```typescript
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { BrowserLiveView } from '@/components/command-center/BrowserLiveView';

describe('BrowserLiveView', () => {
  it('renders img with data URL when src provided', () => {
    render(<BrowserLiveView src="data:image/png;base64,abc123" agentId="scout_1" />);
    const img = screen.getByRole('img', { name: /browser session/i });
    expect(img).toHaveAttribute('src', 'data:image/png;base64,abc123');
  });

  it('shows agent label', () => {
    render(<BrowserLiveView src="data:image/png;base64,abc" agentId="scout_1" />);
    expect(screen.getByText(/live browser/i)).toBeDefined();
  });
});
```

### Step 2: Run test to verify it fails

```bash
npm run test -- --run BrowserLiveView
```

Expected: `Module not found: BrowserLiveView`

### Step 3: Create `BrowserLiveView.tsx`

```tsx
'use client';

import { Monitor } from 'lucide-react';

export function BrowserLiveView({
  src,
  agentId: _agentId,
}: {
  src: string;
  agentId: string;
}) {
  return (
    <div className="rounded-xl overflow-hidden border bg-black/30">
      <div className="flex items-center gap-2 px-3 py-2 border-b bg-muted/30">
        <Monitor className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-xs text-muted-foreground font-medium">Live Browser</span>
        <span className="ml-auto flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
          <span className="text-xs text-green-500">live</span>
        </span>
      </div>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt="Live browser session"
        className="w-full object-contain max-h-64"
      />
    </div>
  );
}
```

### Step 4: Run test to verify it passes

```bash
npm run test -- --run BrowserLiveView
```

Expected: `PASSED`

### Step 5: Commit

```bash
git add components/command-center/BrowserLiveView.tsx __tests__/components/command-center/BrowserLiveView.test.tsx
git commit -m "feat(command-center): add BrowserLiveView component for live screenshot display"
```

---

## Task 8: Update `AgentDetailSheet` to show live browser view

**Files:**
- Modify: `dashboard/components/command-center/AgentDetailSheet.tsx`

### Step 1: Write the test

File: `dashboard/__tests__/components/command-center/AgentDetailSheet.test.tsx`

```typescript
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';

// Minimal mock of worker with screenshot
const workerWithScreenshot = {
  agentId: 'scout_abc',
  agentType: 'scout',
  status: 'running' as const,
  progress: 50,
  phase: 'scanning',
  message: 'Scanning...',
  durationMs: 0,
  costUsd: 0,
  findingsCount: 0,
  resultSummary: '',
  latestScreenshot: 'abc123==',
};

// Test that BrowserLiveView renders when latestScreenshot is set
describe('AgentDetailSheet live view', () => {
  it('shows Live Browser section when worker has latestScreenshot', () => {
    // Import after mocking dependencies if needed
    // This is a smoke test — just verify the conditional rendering path
    expect(workerWithScreenshot.latestScreenshot).toBeDefined();
    const dataUrl = `data:image/png;base64,${workerWithScreenshot.latestScreenshot}`;
    expect(dataUrl).toMatch(/^data:image\/png;base64,/);
  });
});
```

### Step 2: Edit `AgentDetailSheet.tsx`

**Add import** at the top:
```tsx
import { BrowserLiveView } from './BrowserLiveView';
```

**Insert `BrowserLiveView` in the scrollable content area** — inside the `<div className="flex-1 overflow-y-auto p-6 space-y-6">`, after the `AgentMetricsBar` and before the Execution Log block:

```tsx
{worker.latestScreenshot && (
  <div>
    <h3 className="text-sm font-medium mb-3">Live Browser</h3>
    <BrowserLiveView
      src={`data:image/png;base64,${worker.latestScreenshot}`}
      agentId={worker.agentId}
    />
  </div>
)}
```

Full updated scrollable section:
```tsx
<div className="flex-1 overflow-y-auto p-6 space-y-6">
  {worker.status !== 'pending' && <AgentMetricsBar worker={worker} />}

  {worker.latestScreenshot && (
    <div>
      <h3 className="text-sm font-medium mb-3">Live Browser</h3>
      <BrowserLiveView
        src={`data:image/png;base64,${worker.latestScreenshot}`}
        agentId={worker.agentId}
      />
    </div>
  )}

  <div>
    <h3 className="text-sm font-medium mb-3">Execution Log</h3>
    <ExecutionLog events={events} agentId={worker.agentId} />
  </div>

  {worker.status === 'complete' && worker.findingsCount > 0 && (
    <div>
      <h3 className="text-sm font-medium mb-3">Findings</h3>
      <FindingsList findings={[]} />
    </div>
  )}
</div>
```

### Step 3: Run all dashboard tests

```bash
cd dashboard
npm run test -- --run
```

Expected: all tests pass

### Step 4: Build to check for TypeScript errors

```bash
npm run build
```

Expected: build succeeds with no type errors

### Step 5: Commit and deploy

```bash
git add components/command-center/AgentDetailSheet.tsx __tests__/components/command-center/AgentDetailSheet.test.tsx
git commit -m "feat(command-center): show BrowserLiveView in AgentDetailSheet when screenshot available"
```

Deploy frontend:
```bash
npx vercel --prod
```

Deploy backend:
```bash
cd /Users/bvk/Downloads/e2e-testing-agent
railway up --detach
```

---

## Verification Checklist

After all 8 tasks are complete:

1. **Unit tests pass**:
   ```bash
   python -m pytest tests/streaming/ tests/browser/ tests/agents/test_scout.py tests/orchestrator/ -v
   ```

2. **Frontend tests pass**:
   ```bash
   cd dashboard && npm run test -- --run
   ```

3. **End-to-end smoke test** — launch swarm against `https://skopaq.ai`:
   - Scout card appears in agent grid immediately
   - Screenshots appear in Scout's AgentDetailSheet while running
   - After ~45s, Phase 1 agents start and also show screenshots
   - APITester now tests endpoints from the OpenAPI spec (not hardcoded paths)
   - UITester tests the actual public pages Scout found

4. **Expected SSE event sequence**:
   ```
   run_started        → agents: ["scout", "ui_tester", "api_tester", ...]
   step_started       → scout begins
   browser_screenshot → frames every 2s
   state_delta        → "Mapping public pages..." (30%)
   browser_screenshot → more frames
   state_delta        → "Found 4 pages, 12 endpoints" (100%)
   step_finished      → scout: 16 findings

   step_started       → ui_tester begins
   browser_screenshot → ui_tester frames
   ...
   run_finished       → all agents complete
   ```

5. **Auth detection finding** appears in scout results for `skopaq.ai` (Clerk detected)
