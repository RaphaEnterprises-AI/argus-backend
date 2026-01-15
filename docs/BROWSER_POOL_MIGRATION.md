# Browser Pool Migration Guide

This guide covers migrating from the fragmented browser automation clients to the unified **BrowserPoolClient** backed by Hetzner Kubernetes.

## What Changed

### Before (Fragmented)
```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ E2EBrowserClient│  │BrowserWorkerCli │  │  PlaywrightTools│
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Cloudflare    │  │   TestingBot    │  │   Local PW      │
│ Browser Render  │  │     Cloud       │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### After (Unified)
```
┌─────────────────────────────────────────────────────────────┐
│                    BrowserPoolClient                         │
│          (Unified client with vision fallback)              │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│               Hetzner K8s Browser Pool                       │
│        (5-500 pods, auto-scaling, MCP-compatible)           │
└─────────────────────────────────────────────────────────────┘
                             │ fallback
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Claude Computer Use (Vision)                    │
│             (Automatic fallback for failures)               │
└─────────────────────────────────────────────────────────────┘
```

## Deployment

### 1. Deploy Hetzner Browser Pool

```bash
# Set Hetzner Cloud token
export HCLOUD_TOKEN=your-token-here

# Deploy (choose scale preset)
cd browser-pool
./deploy.sh dev      # ~€25/mo, 1-10 pods
./deploy.sh small    # ~€55/mo, 5-20 pods
./deploy.sh medium   # ~€280/mo, 10-50 pods
./deploy.sh large    # ~€1,600/mo, 20-200 pods
./deploy.sh enterprise  # ~€6,000/mo, 50-500 pods

# Get the load balancer IP
cd terraform && terraform output load_balancer_ip
```

### 2. Configure Environment

Add to your `.env`:
```bash
BROWSER_POOL_URL=http://YOUR_LB_IP
```

For MCP server, update `argus-mcp-server/wrangler.toml`:
```toml
[vars]
BROWSER_POOL_URL = "http://YOUR_LB_IP"
```

## Migration Examples

### UITesterAgent (Automatic)

**No code changes required!** The import path remains the same, but now uses V2:

```python
# This automatically uses UITesterAgentV2 (BrowserPoolClient)
from src.agents import UITesterAgent

agent = UITesterAgent()
result = await agent.execute(test_spec=spec, app_url="https://example.com")
```

If you need the legacy agent during migration:
```python
from src.agents import UITesterAgentLegacy

agent = UITesterAgentLegacy()  # Uses E2EBrowserClient
```

### Direct Browser Client Usage

**Before:**
```python
from src.browser import E2EBrowserClient

async with E2EBrowserClient() as client:
    page = await client.new_page("https://example.com")
    await page.act("Click Sign In")
    data = await page.extract({"username": "string"})
```

**After:**
```python
from src.browser import BrowserPoolClient

async with BrowserPoolClient() as client:
    # Discover elements (like MCP /observe)
    result = await client.observe("https://example.com")

    # Execute action with self-healing
    result = await client.act("https://example.com", "Click Sign In")

    # Extract data
    result = await client.extract(
        "https://example.com",
        {"username": "string"}
    )
```

### Multi-Step Tests

**Before:**
```python
result = await run_test_with_e2e_client(
    url="https://example.com",
    steps=["Click Login", "Type email", "Submit"],
)
```

**After:**
```python
from src.browser import BrowserPoolClient

async with BrowserPoolClient() as client:
    result = await client.test(
        url="https://example.com",
        steps=["Click Login", "Type email", "Submit"],
        capture_screenshots=True,
    )

    # Access step-by-step results
    for step in result.steps:
        print(f"Step {step.step_index}: {step.instruction}")
        print(f"  Success: {step.success}")
        print(f"  Mode: {step.execution_mode.value}")  # dom/vision
```

### MCP Server Tools

MCP tools (`argus_discover`, `argus_act`, `argus_test`) automatically route through:
1. **Hetzner Browser Pool** (if `BROWSER_POOL_URL` is set)
2. **Cloudflare Browser Rendering** (fallback)

No changes needed in MCP tool implementations.

## API Reference

### BrowserPoolClient

```python
class BrowserPoolClient:
    async def observe(url: str, instruction: str = None) -> ObserveResult
    async def act(url: str, instruction: str, ...) -> ActResult
    async def test(url: str, steps: list[str], ...) -> TestResult
    async def extract(url: str, schema: dict, ...) -> ExtractResult
    async def screenshot(url: str) -> bytes
    async def health() -> PoolHealth
```

### Result Types

```python
@dataclass
class ObserveResult:
    success: bool
    url: str
    title: str
    elements: list[ElementInfo]  # Discovered interactive elements
    error: Optional[str]

@dataclass
class ActResult:
    success: bool
    message: str
    actions: list[ActionResult]
    url: str
    screenshot: Optional[str]  # Base64
    execution_mode: ExecutionMode  # dom/vision/hybrid

@dataclass
class TestResult:
    success: bool
    steps: list[StepResult]
    total_steps: int
    passed_steps: int
    failed_steps: int
    total_duration_ms: int
```

## Deprecation Timeline

| Component | Status | Action |
|-----------|--------|--------|
| `BrowserPoolClient` | **New (Primary)** | Use this for all new code |
| `UITesterAgentV2` | **New (Primary)** | Automatic via `UITesterAgent` |
| `E2EBrowserClient` | Maintained | Use `BrowserPoolClient` instead |
| `UITesterAgentLegacy` | Maintained | Use `UITesterAgent` (V2) instead |
| `BrowserWorkerClient` | Deprecated | Migrate to `BrowserPoolClient` |
| `PlaywrightTools` | Deprecated | Migrate to `BrowserPoolClient` |

### Deprecation Warnings

The following will show deprecation warnings in future versions:
- `from src.browser import E2EBrowserClient`
- Direct usage of `BrowserWorkerClient`
- Direct usage of `PlaywrightTools` for remote automation

## Troubleshooting

### Pool Connection Failed
```
Browser Pool error: Connection refused, falling back to Cloudflare
```
**Solution:** Verify `BROWSER_POOL_URL` is correct and pool is deployed.

### Vision Fallback Triggered
```
Execution mode: vision (fallback triggered)
```
**This is normal!** DOM automation failed, Claude Computer Use handled it.

### High Latency
```
Step duration: 5000ms+
```
**Check:**
1. Pool location (should be near your users)
2. HPA scaling (may need more replicas)
3. Pod resources (may need larger worker nodes)

## Performance Comparison

| Metric | Cloudflare | Hetzner Pool |
|--------|------------|--------------|
| Cold start | 2-5s | <500ms |
| Timeout limit | 30s | Unlimited |
| Max concurrent | Limited | 500+ pods |
| Cost (100 tests/day) | ~$0 | ~€25/mo |
| Cost (10k tests/day) | Limited | ~€280/mo |
| Safari/IE support | No | Via TestingBot fallback |
