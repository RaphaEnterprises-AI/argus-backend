# Scout Agent + Live Browser View — Design

**Date**: 2026-03-03
**Status**: Approved, ready for implementation

---

## Problem

The current swarm launches all 8 agents simultaneously with only `target_url`. This causes:
- Agents hit auth walls (Clerk redirect) and return 0 findings
- APITester probes 4 hardcoded paths — doesn't know actual endpoints
- UITester runs 1 generic "goto + assert body" test
- CodeAnalyzer has no codebase and can't infer anything from a URL alone
- No transparency — users can't see what the AI is doing

---

## Solution: Scout Agent as Phase 0

A dedicated **Scout Agent** runs sequentially before the scatter phase. It opens a real browser session, navigates the app with Claude Vision narration, builds a structured `AppIntelligenceReport`, and streams live screenshots to the dashboard. When complete (~45s), its report is injected into `SwarmConfig.recon_context` and all Phase 1 agents scatter with full context.

---

## Architecture

```
swarm launch (target_url)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 0: SCOUT  (~45s, sequential)                     │
│                                                         │
│  ScoutAgent                                             │
│   ├─ Selenium session starts                            │
│   ├─ background task: screenshot every 2s               │
│   │    └─ emits: browser_screenshot {agent_id, b64}     │
│   ├─ navigates + Claude Vision narrates each page       │
│   ├─ detects auth (Clerk redirect, form, OAuth)         │
│   ├─ maps public pages + user flows                     │
│   ├─ fetches OpenAPI spec at common paths               │
│   └─ returns AppIntelligenceReport                      │
└──────────────────────┬──────────────────────────────────┘
                       │  injects into SwarmConfig.recon_context
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 1: SCATTER  (parallel, with full context)        │
│                                                         │
│  UITester │ APITester │ Security │ Accessibility │ ...  │
│  (each also streams browser_screenshot during session)  │
└─────────────────────────────────────────────────────────┘
```

---

## Components

### Backend (Python)

| Component | Location | Purpose |
|-----------|----------|---------|
| `ScoutAgent` | `src/agents/scout.py` | New agent. Selenium session + Claude Vision loop. Returns `AppIntelligenceReport`. |
| `AppIntelligenceReport` | `src/agents/scout.py` | Typed dataclass: auth, pages, api_endpoints, stack, suggested_agents, confidence |
| `BrowserScreenshotEvent` | `src/streaming/agui_events.py` | New SSE event: `{ agent_id, base64_png, url, step_description }` |
| `screenshot_stream_task()` | `src/browser/selenium_grid_client.py` | Async background loop: poll screenshot every 2s, call callback |
| `_run_scout()` | `src/orchestrator/swarm_orchestrator.py` | New runner. Calls ScoutAgent, streams screenshots, returns AppIntelligenceReport |
| `_run_swarm()` update | `src/orchestrator/swarm_orchestrator.py` | Insert Phase 0 (Scout) before Phase 1 scatter, for all URL-based modes |
| Remove `_preflight_recon()` | `src/orchestrator/swarm_orchestrator.py` | Replaced entirely by ScoutAgent |

### Frontend (TypeScript)

| Component | Location | Purpose |
|-----------|----------|---------|
| `BrowserLiveView` | `components/command-center/BrowserLiveView.tsx` | `<img>` that updates src on `browser_screenshot` SSE events for an agent |
| `AgentDetailSheet` update | existing | Show `BrowserLiveView` when agent has screenshots |
| `use-swarm-stream.ts` update | existing | Handle `browser_screenshot` event; update `worker.latestScreenshot` |
| `SwarmWorker` type update | `use-swarm-stream.ts` | Add `latestScreenshot?: string` field |

### New SSE Event Type

```
Event name: browser_screenshot
Payload:
{
  "agent_id": "scout_abc123",
  "base64_png": "<base64 encoded PNG>",
  "url": "https://skopaq.ai/pricing",
  "step_description": "Analyzing pricing page flows..."
}
```

---

## Data Flow

### ScoutAgent execution loop

```
1. Start Selenium session (SeleniumGridClient)
2. Kick off _screenshot_loop() as background asyncio task
   └─ every 2s: selenium.screenshot() → base64 PNG
   └─ emitter.emit(BrowserScreenshotEvent(...))
3. Navigate to target_url
   └─ Claude Vision: "What is this page? What app type?"
4. Detect auth:
   - Redirect to /sign-in → auth_detected=True, auth_type="clerk"
   - Login form visible → extract form selectors for login_steps
5. For each public page (up to 8 pages):
   - navigate, screenshot, Claude Vision("What user flows exist here?")
   - record: title, accessible=True, detected_flows[]
6. Probe OpenAPI spec at 7 common paths
7. Cancel _screenshot_loop()
8. Close Selenium session (triggers R2 video upload)
9. Ask Claude to synthesize AppIntelligenceReport from all observations
10. Return AppIntelligenceReport
```

### AppIntelligenceReport shape

```python
@dataclass
class AppIntelligenceReport:
    target_url: str
    app_type: str           # "SaaS dashboard", "marketing site", "API", etc.
    stack: str              # "Next.js + Clerk + FastAPI"
    auth_detected: bool
    auth_type: str | None   # "clerk", "auth0", "custom", None
    login_steps: list[dict] # [{ action, selector, value }] for future auth replay
    public_pages: list[PageInfo]
    api_endpoints: list[dict]
    openapi_spec: dict | None
    openapi_url: str | None
    suggested_agents: list[str]  # orchestrator can prune unnecessary agents
    confidence: float
    raw_observations: list[str]  # Claude's per-page narrations
```

### Screenshot streaming

```
ScoutAgent._screenshot_loop(emitter, agent_id):
  while running:
    png = await selenium.screenshot()   # base64 PNG
    await emitter.emit(BrowserScreenshotEvent(
      agent_id=agent_id,
      base64_png=png,
      url=await selenium.current_url(),
      step_description=self._current_step,
    ))
    await asyncio.sleep(2.0)

Frontend hook (use-swarm-stream.ts):
  eventSource.addEventListener("browser_screenshot", (e) => {
    const { agent_id, base64_png } = JSON.parse(e.data)
    setState(prev => ({
      ...prev,
      workers: prev.workers.map(w =>
        w.agentId === agent_id
          ? { ...w, latestScreenshot: base64_png }
          : w
      )
    }))
  })

AgentDetailSheet:
  {worker.latestScreenshot && (
    <BrowserLiveView
      src={`data:image/png;base64,${worker.latestScreenshot}`}
      agentId={worker.agentId}
    />
  )}
```

---

## What the Scout replaces

The HTTP-only `_preflight_recon()` added in the previous session is replaced entirely by ScoutAgent. Scout does everything recon does, plus:
- Real browser rendering (not just HTTP headers)
- Claude Vision understanding of what the UI actually shows
- Auth form selector extraction (enables future auth replay)
- User flow identification from page structure
- AI synthesis into structured report

---

## Implementation Sequence

1. `BrowserScreenshotEvent` in `agui_events.py` + `AGUIEventType` enum
2. `screenshot_stream_task()` on `SeleniumGridClient`
3. `AppIntelligenceReport` dataclass + `ScoutAgent` in `src/agents/scout.py`
4. `_run_scout()` in swarm_orchestrator + Phase 0 insertion in `_run_swarm()`
5. Remove `_preflight_recon()`, update all agent runners to use richer recon_context fields from Scout
6. Frontend: `SwarmWorker.latestScreenshot` + `browser_screenshot` event handler
7. `BrowserLiveView` component
8. `AgentDetailSheet` update to show live view

---

## Success Criteria

- Scout runs and produces `AppIntelligenceReport` for `skopaq.ai`
- Live screenshots appear in the Agent Detail Sheet while Scout is running
- APITester now tests real endpoints from the discovered OpenAPI spec
- UITester tests the actual public pages Scout found (not just root URL)
- Auth detection appears as a finding with provider name
- All subsequent agents show screenshots while their browser session is active
