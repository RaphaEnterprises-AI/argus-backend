# Browser Execution Workflow

**Created**: January 20, 2026
**Status**: Production Ready

---

## Overview

Argus supports multiple browser execution paths depending on the client (MCP/CLI vs Dashboard) and configuration. This document explains how browser automation flows through the system.

---

## Execution Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                     │
├─────────────────────────────┬─────────────────────────────────────────────────┤
│  MCP Server (Claude Code)   │              Dashboard (Next.js)                │
│  argus_discover/act/test    │         Test Management UI                      │
└─────────────────────────────┴─────────────────────────────────────────────────┘
                │                                    │
                ▼                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           API GATEWAY LAYER                                   │
├───────────────────────────────────────────────────────────────────────────────┤
│  Cloudflare Worker (argus-mcp.samuelvinay-kumar.workers.dev)                  │
│  Routes: /act, /extract, /observe, /test, /agent                              │
│  Handles: Request validation, backend routing, failover                       │
└───────────────────────────────────────────────────────────────────────────────┘
                │
    ┌───────────┴───────────────────────────┐
    ▼                                       ▼
┌─────────────────────────┐     ┌─────────────────────────────────────────────┐
│  Vultr Browser Pool     │     │  Cloudflare Browser (fallback)              │
│  (Primary - Kubernetes) │     │  (puppeteer in Worker)                      │
│  URL: env.BROWSER_POOL_URL    │                                             │
└─────────────────────────┘     └─────────────────────────────────────────────┘
```

---

## Execution Paths

### 1. MCP Server → Cloudflare Worker → Browser Pool

**Used by**: Claude Code, Cursor, Windsurf (via MCP protocol)

```
Claude Code
    │ MCP Protocol (SSE)
    ▼
Argus MCP Server (Cloudflare Worker)
    │ REST API
    ▼
Cloudflare Worker Gateway
    │ HTTP POST
    ▼
Vultr Browser Pool (Kubernetes)
    │ Browser instance
    ▼
Target Website
```

**Tools Available**:
- `argus_health` - API health check
- `argus_discover` - Find interactive elements
- `argus_act` - Execute single action (click, type)
- `argus_test` - Run multi-step tests
- `argus_extract` - Extract structured data
- `argus_agent` - Autonomous task completion
- `argus_generate_test` - AI test generation

### 2. Dashboard → FastAPI → LangGraph → Agents → Browser Pool

**Used by**: Web dashboard for test management

```
Dashboard (Next.js)
    │ HTTP/WebSocket
    ▼
FastAPI Backend (src/api/)
    │
    ▼
LangGraph Orchestrator (src/orchestrator/)
    │ Checkpointed state machine
    ▼
Test Executor Agents
    │ UITesterAgent, etc.
    ▼
HybridExecutor (src/browser/hybrid_executor.py)
    │ DOM-first strategy
    ▼
BrowserPoolClient (src/browser/pool_client.py)
    │ HTTP to pool
    ▼
Vultr Browser Pool
```

### 3. Local Playwright (Development/Testing)

**Used by**: Unit tests, local development

```
pytest / local script
    │
    ▼
HybridExecutor
    │ DOM-first path
    ▼
Local Playwright
    │ Direct browser control
    ▼
Target Website
```

---

## Configuration

### Environment Variables

```bash
# Browser Pool Configuration
BROWSER_POOL_URL=https://your-vultr-pool.example.com
BROWSER_POOL_JWT_SECRET=your-jwt-secret

# Cloudflare Worker (MCP Gateway)
ARGUS_API_URL=https://argus-mcp.samuelvinay-kumar.workers.dev
ARGUS_BRAIN_URL=https://argus-brain.samuelvinay-kumar.workers.dev

# API Keys
ANTHROPIC_API_KEY=sk-ant-...
```

### Backend Selection

The Cloudflare Worker chooses backends in this order:

1. **Vultr Pool** (if `BROWSER_POOL_URL` configured) - Primary
2. **Cloudflare Browser** - Fallback for simple operations
3. **TestingBot** - Alternative cloud provider (if configured)

```typescript
// cloudflare-worker/src/index.ts
function shouldUseVultrPool(backend: Backend, env: Env): boolean {
  return env.BROWSER_POOL_URL && (backend === 'vultr' || backend === 'auto');
}
```

---

## Data Flow: Test Execution

### From MCP (Claude Code)

```
1. User: "Test the login form on example.com"
2. Claude Code → mcp__argus__argus_test
3. MCP Server → POST /test to Cloudflare Worker
4. Worker → POST /test to Vultr Pool
5. Pool → Playwright browser executes steps
6. Results flow back with screenshots
7. MCP Server formats markdown response
```

### From Dashboard

```
1. User clicks "Run Test" on test_id=abc123
2. Dashboard → POST /api/v1/tests/abc123/run
3. FastAPI → Creates LangGraph execution
4. LangGraph → Routes to UITesterAgent
5. Agent → HybridExecutor.execute()
6. HybridExecutor → BrowserPoolClient.test()
7. Pool → Playwright executes steps
8. Results stored in PostgreSQL
9. Dashboard updates via SSE stream
```

---

## Key Components

### BrowserPoolClient (`src/browser/pool_client.py`)

Handles all communication with the Vultr browser pool:

```python
class BrowserPoolClient:
    async def observe(self, url: str, instruction: str) -> ObserveResult
    async def act(self, url: str, action: ActionSpec) -> ActResult
    async def test(self, url: str, steps: list[str]) -> TestResult
    async def extract(self, url: str, instruction: str) -> ExtractResult
```

### HybridExecutor (`src/browser/hybrid_executor.py`)

Combines DOM-first speed with vision-based fallback:

```python
class HybridExecutor:
    async def execute(self, action: ActionSpec) -> ExecutionResult:
        # 1. Try DOM-based execution (fast)
        # 2. If fails, try vision-based (accurate)
        # 3. Apply self-healing if needed
```

### MCP Server (`argus-mcp-server/src/index.ts`)

Exposes browser automation to AI IDEs:

```typescript
this.server.tool("argus_test", ..., async ({ url, steps }) => {
    const result = await callArgusAPI<ArgusTestResponse>("/test", { url, steps });
    // Format results as markdown
});
```

---

## Self-Healing

Both execution paths support self-healing:

1. **Selector Healing**: When a selector fails, the system:
   - Searches for cached healing patterns
   - Uses AI to find alternative selectors
   - Stores successful patterns for future use

2. **Timing Healing**: When an element isn't found:
   - Implements smart waits
   - Retries with exponential backoff
   - Reports timing issues to self-healer agent

---

## Monitoring

### Execution Logs

```python
# All browser actions are logged with structlog
logger.info("browser_action_executed",
    action=action.type,
    selector=action.selector,
    success=result.success,
    duration_ms=duration,
    backend="vultr")
```

### MCP Activity Tracking

```typescript
// MCP server tracks all tool invocations
this.recordActivity("argus_act", {
    durationMs: Date.now() - startTime,
    success: true,
    metadata: { url, instruction }
});
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Unknown error" from act | No matching action found | Improve instruction clarity |
| "undefined" from extract | AI failed to extract | Check page content, refine schema |
| Timeout errors | Slow page load | Increase timeout, check network |
| Pool connection failed | JWT expired | Rotate BROWSER_POOL_JWT_SECRET |

### Debug Commands

```bash
# Test browser pool directly
curl -X POST https://your-pool-url/observe \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Check MCP health
mcp__argus__argus_health
```

---

## Deployment

### MCP Server

```bash
cd argus-mcp-server
npm run deploy  # Deploys to Cloudflare Workers
```

### Cloudflare Worker (Gateway)

```bash
cd cloudflare-worker
npm run deploy
```

### Backend (FastAPI)

```bash
# Docker
docker-compose up -d

# Direct
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## Summary

| Client | Entry Point | Browser Backend |
|--------|-------------|-----------------|
| MCP (Claude Code) | argus_* tools | Vultr Pool (via Worker) |
| Dashboard | FastAPI REST | Vultr Pool (via HybridExecutor) |
| Local Dev | pytest | Local Playwright |

All paths support:
- Self-healing selectors
- Screenshot capture
- Structured logging
- Retry with failover
