# Argus Test Bench — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy 5 sample apps on Railway that simulate real Argus customers, with 30 toggleable bugs and 13 automated scenarios that exercise every Argus feature end-to-end.

**Architecture:** Hybrid approach — 3 forked open-source apps (Conduit, Plane, JSON API) for realistic test surfaces + a custom Chaos App with toggleable bugs + a Chaos Controller (FastAPI) that orchestrates scenarios by calling both the Chaos App and Argus API. Nightly GitHub Actions CI pipeline runs all scenarios and reports to Slack.

**Tech Stack:** React (Vite), Express, FastAPI (Python), PostgreSQL, Railway, GitHub Actions

**Design Doc:** `docs/plans/2026-02-27-testbench-design.md`

---

## Phase 1: Scaffold + JSON API (simplest app first)

### Task 1: Create testbench directory structure

**Files:**
- Create: `testbench/README.md`
- Create: `testbench/.gitignore`

**Step 1: Create scaffold**

```bash
mkdir -p testbench/{json-api,chaos-app/{frontend,backend},chaos-controller,conduit,plane}
mkdir -p testbench/.github/workflows
```

**Step 2: Create testbench README**

```markdown
# Argus Test Bench

Sample apps deployed on Railway to dogfood Argus end-to-end.

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Conduit | conduit-testbench.up.railway.app | Simple blog (RealWorld) |
| Plane | plane-testbench.up.railway.app | Project management SaaS |
| JSON API | jsonapi-testbench.up.railway.app | Pure REST with OpenAPI spec |
| Chaos App | chaos-testbench.up.railway.app | E-commerce with 30 toggleable bugs |
| Chaos Controller | chaos-ctrl-testbench.up.railway.app | Scenario orchestrator |

## Quick Start

```bash
# Deploy everything to Railway
./testbench/deploy.sh

# Run a scenario
curl -X POST https://chaos-ctrl-testbench.up.railway.app/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{"scenario": "golden-path"}'
```
```

**Step 3: Create .gitignore**

```
node_modules/
__pycache__/
*.pyc
.env
dist/
build/
.venv/
```

**Step 4: Commit**

```bash
git add testbench/
git commit -m "feat(testbench): scaffold directory structure"
```

---

### Task 2: Build JSON API server

The simplest service — a standalone Express REST API with auto-generated OpenAPI spec. No frontend, no database, just a clean API for Argus to discover and test.

**Files:**
- Create: `testbench/json-api/package.json`
- Create: `testbench/json-api/src/server.js`
- Create: `testbench/json-api/src/routes/users.js`
- Create: `testbench/json-api/src/routes/posts.js`
- Create: `testbench/json-api/src/routes/comments.js`
- Create: `testbench/json-api/src/routes/todos.js`
- Create: `testbench/json-api/src/openapi.yaml`
- Create: `testbench/json-api/Dockerfile`
- Create: `testbench/json-api/railway.json`

**Step 1: Create package.json**

```json
{
  "name": "argus-testbench-json-api",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "start": "node src/server.js",
    "dev": "node --watch src/server.js"
  },
  "dependencies": {
    "express": "^4.21.0",
    "swagger-ui-express": "^5.0.0",
    "yamljs": "^0.3.0",
    "cors": "^2.8.5",
    "uuid": "^10.0.0"
  }
}
```

**Step 2: Create server.js**

Express server with in-memory storage, Swagger UI at `/api/v1/docs`, OpenAPI spec at `/api/v1/openapi.json`. Mounts 4 route modules. Health check at `/health`. Listens on `PORT` env var (Railway injects this).

Key implementation details:
- In-memory arrays for users, posts, comments, todos (seeded with 5 items each on startup)
- Full CRUD on all resources
- Proper HTTP status codes (201 for create, 204 for delete, 404 for not found)
- Pagination via `?page=1&limit=10` query params
- Filtering: `GET /posts?userId=1`, `GET /comments?postId=1`
- Request validation (return 422 for missing required fields)

**Step 3: Create OpenAPI spec (openapi.yaml)**

Full OpenAPI 3.0 spec describing all endpoints, request/response schemas, error responses. This is what Argus's API discovery will consume.

Endpoints to document:
```
GET/POST       /api/v1/users
GET/PUT/DELETE  /api/v1/users/:id
GET/POST       /api/v1/posts
GET/PUT/DELETE  /api/v1/posts/:id
GET            /api/v1/posts/:id/comments
GET/POST       /api/v1/comments
GET/PUT/DELETE  /api/v1/comments/:id
GET/POST       /api/v1/todos
PATCH          /api/v1/todos/:id
DELETE         /api/v1/todos/:id
```

**Step 4: Create Dockerfile**

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --production
COPY src/ ./src/
EXPOSE ${PORT:-3000}
CMD ["node", "src/server.js"]
```

**Step 5: Create railway.json**

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "testbench/json-api/Dockerfile"
  },
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 30,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

**Step 6: Test locally**

```bash
cd testbench/json-api
npm install
npm run dev
# Verify: curl http://localhost:3000/health
# Verify: curl http://localhost:3000/api/v1/users
# Verify: curl http://localhost:3000/api/v1/docs (Swagger UI)
# Verify: curl http://localhost:3000/api/v1/openapi.json
```

**Step 7: Commit**

```bash
git add testbench/json-api/
git commit -m "feat(testbench): add JSON API server with OpenAPI spec"
```

---

## Phase 2: Chaos App (custom React + Express)

### Task 3: Build Chaos App backend (Express + bug registry)

The core of the test bench. An Express server with an e-commerce API and a bug registry that toggles 30 bugs via middleware.

**Files:**
- Create: `testbench/chaos-app/backend/package.json`
- Create: `testbench/chaos-app/backend/src/server.js`
- Create: `testbench/chaos-app/backend/src/bugs/registry.js`
- Create: `testbench/chaos-app/backend/src/bugs/selectors.js`
- Create: `testbench/chaos-app/backend/src/bugs/performance.js`
- Create: `testbench/chaos-app/backend/src/bugs/a11y.js`
- Create: `testbench/chaos-app/backend/src/bugs/security.js`
- Create: `testbench/chaos-app/backend/src/bugs/api.js`
- Create: `testbench/chaos-app/backend/src/bugs/flaky.js`
- Create: `testbench/chaos-app/backend/src/middleware/chaos.js`
- Create: `testbench/chaos-app/backend/src/routes/products.js`
- Create: `testbench/chaos-app/backend/src/routes/cart.js`
- Create: `testbench/chaos-app/backend/src/routes/auth.js`
- Create: `testbench/chaos-app/backend/src/routes/users.js`
- Create: `testbench/chaos-app/backend/src/routes/orders.js`
- Create: `testbench/chaos-app/backend/src/routes/chaos.js` (bug toggle API)
- Create: `testbench/chaos-app/backend/src/db.js` (PostgreSQL connection)

**Step 1: Create bug registry (registry.js)**

Central store for all 30 bugs. Each bug has:
- `id` (string, e.g., `selector-login-id`)
- `category` (enum: `selectors`, `performance`, `a11y`, `security`, `api`, `flaky`)
- `description` (human-readable)
- `enabled` (boolean, default false)
- `metadata` (object, bug-specific config)

Exports: `getBug(id)`, `enableBug(id)`, `disableBug(id)`, `resetAll()`, `listBugs()`, `enableScenario(name)`.

Scenario presets:
- `selector-drift`: enables all 5 selector bugs
- `perf-degradation`: enables all 4 performance bugs
- `a11y-violations`: enables all 5 accessibility bugs
- `security-vulns`: enables all 6 security bugs
- `api-chaos`: enables all 5 API bugs
- `flaky-everything`: enables all 4 flaky bugs
- `full-chaos`: enables all 30 bugs

**Step 2: Create bug implementation modules**

Each module exports middleware functions that check the registry and apply bugs:

- `selectors.js` — Modifies HTML responses to rename IDs, classes, restructure DOM, randomize IDs, or remove elements. Uses cheerio for server-side HTML manipulation.
- `performance.js` — Adds artificial delays (`setTimeout`), injects render-blocking scripts, inflates response payloads, leaks memory references.
- `a11y.js` — Strips `alt` attrs, injects low-contrast CSS overrides, removes `<label>` elements, disables `tabindex`, strips ARIA attrs.
- `security.js` — Reflects unsanitized query params (XSS), concatenates SQL in login query (SQLi), skips JWT expiry check, removes ownership check on user routes (IDOR), follows user-provided URLs (SSRF), embeds fake API key in frontend bundle.
- `api.js` — Changes field types in responses, omits fields, returns wrong status codes, renames query params, randomly returns 500.
- `flaky.js` — Adds random delays, defers button interactivity, returns stale cached data, drops connections randomly.

**Step 3: Create chaos middleware (chaos.js)**

Express middleware that runs on every request. Checks which bugs are enabled and applies them:

```javascript
// Simplified structure
module.exports = function chaosMiddleware(req, res, next) {
  // Performance bugs (applied before route handlers)
  applyPerformanceBugs(req, res);

  // Security bugs (modify auth behavior)
  applySecurityBugs(req, res);

  // API bugs (modify response)
  applyApiBugs(req, res);

  // Flaky bugs (random failures)
  applyFlakyBugs(req, res);

  // Selector/a11y bugs applied in response interceptor
  const originalSend = res.send;
  res.send = function(body) {
    if (typeof body === 'string' && body.includes('<!DOCTYPE')) {
      body = applySelectorBugs(body);
      body = applyA11yBugs(body);
    }
    originalSend.call(this, body);
  };

  next();
};
```

**Step 4: Create e-commerce routes**

Standard CRUD routes for a minimal e-commerce app:
- `products.js` — GET /api/products, GET /api/products/:id, POST/PUT/DELETE
- `cart.js` — GET /api/cart, POST /api/cart/add, DELETE /api/cart/:itemId, POST /api/cart/checkout
- `auth.js` — POST /api/auth/login, POST /api/auth/register, POST /api/auth/refresh, GET /api/auth/me
- `users.js` — GET /api/users/:id, PUT /api/users/:id, GET /api/users/:id/orders
- `orders.js` — GET /api/orders, GET /api/orders/:id, POST /api/orders

**Step 5: Create chaos toggle routes (chaos.js route)**

```
GET  /chaos/bugs              → listBugs()
POST /chaos/bugs/:id/enable   → enableBug(id)
POST /chaos/bugs/:id/disable  → disableBug(id)
POST /chaos/bugs/reset        → resetAll()
POST /chaos/bugs/scenario/:name → enableScenario(name)
GET  /chaos/health            → { status, activeBugs: count, uptime }
```

**Step 6: Test locally with PostgreSQL**

```bash
cd testbench/chaos-app/backend
npm install
# Start with local Postgres
DATABASE_URL=postgresql://localhost:5432/chaos_app npm run dev
# Verify: curl http://localhost:3001/health
# Verify: curl http://localhost:3001/chaos/bugs | jq '.bugs | length'  → 30
# Verify: curl -X POST http://localhost:3001/chaos/bugs/selector-login-id/enable
# Verify: curl http://localhost:3001/chaos/bugs | jq '.bugs[] | select(.id=="selector-login-id") | .enabled'  → true
# Verify: curl -X POST http://localhost:3001/chaos/bugs/reset
```

**Step 7: Commit**

```bash
git add testbench/chaos-app/backend/
git commit -m "feat(testbench): add chaos app backend with 30 toggleable bugs"
```

---

### Task 4: Build Chaos App frontend (React + Vite)

A simple e-commerce UI that renders pages the backend serves. The frontend itself is straightforward — the bugs are injected server-side via middleware.

**Files:**
- Create: `testbench/chaos-app/frontend/package.json`
- Create: `testbench/chaos-app/frontend/vite.config.js`
- Create: `testbench/chaos-app/frontend/index.html`
- Create: `testbench/chaos-app/frontend/src/main.jsx`
- Create: `testbench/chaos-app/frontend/src/App.jsx`
- Create: `testbench/chaos-app/frontend/src/pages/Home.jsx` (product listing)
- Create: `testbench/chaos-app/frontend/src/pages/Product.jsx` (product detail)
- Create: `testbench/chaos-app/frontend/src/pages/Cart.jsx`
- Create: `testbench/chaos-app/frontend/src/pages/Checkout.jsx` (multi-step)
- Create: `testbench/chaos-app/frontend/src/pages/Login.jsx`
- Create: `testbench/chaos-app/frontend/src/pages/Register.jsx`
- Create: `testbench/chaos-app/frontend/src/pages/Profile.jsx`
- Create: `testbench/chaos-app/frontend/src/pages/Search.jsx`
- Create: `testbench/chaos-app/frontend/src/pages/Admin.jsx`
- Create: `testbench/chaos-app/frontend/src/components/Navbar.jsx`
- Create: `testbench/chaos-app/frontend/src/components/ProductCard.jsx`
- Create: `testbench/chaos-app/frontend/src/components/CartItem.jsx`

**Step 1: Create package.json and vite config**

Dependencies: `react`, `react-dom`, `react-router-dom`. Vite proxy to backend at `/api` and `/chaos`.

**Step 2: Create pages**

Each page is a simple functional component with:
- Semantic HTML (headings, nav, forms with labels, buttons with IDs)
- Specific element IDs and class names that the selector bugs will target:
  - `#login-btn` on Login page (target of `selector-login-id` bug)
  - `.add-to-cart` on ProductCard (target of `selector-class-rename` bug)
  - Nested `div > button` structure in Checkout (target of `selector-nesting`)
  - `#item-{id}` on Cart items (target of `selector-dynamic-id`)
  - Checkout submit button (target of `selector-removed`)
- Images with alt text (target of `a11y-no-alt`)
- Form labels (target of `a11y-no-labels`)
- Keyboard-navigable links (target of `a11y-no-keyboard`)
- ARIA landmarks (target of `a11y-missing-aria`)
- Search that displays query (target of `sec-xss-reflected`)

Keep it minimal but functional — this is a test target, not a production app. Use basic CSS (inline or a single stylesheet), no component library.

**Step 3: Create unified Dockerfile**

```dockerfile
# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Production server
FROM node:20-alpine
WORKDIR /app
COPY backend/package.json backend/package-lock.json* ./
RUN npm ci --production
COPY backend/src/ ./src/
COPY --from=frontend-builder /app/frontend/dist ./public/
EXPOSE ${PORT:-3001}
CMD ["node", "src/server.js"]
```

The Express server serves the React build from `./public/` as static files, with SPA fallback for client-side routing.

**Step 4: Create railway.json**

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "testbench/chaos-app/Dockerfile"
  },
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 30,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

**Step 5: Test locally**

```bash
cd testbench/chaos-app
docker compose up  # or run frontend + backend separately
# Verify: http://localhost:3001 shows product listing
# Verify: Navigate to /login, /cart, /checkout, /search
# Verify: Enable a bug, reload, see effect
```

**Step 6: Commit**

```bash
git add testbench/chaos-app/
git commit -m "feat(testbench): add chaos app frontend and Dockerfile"
```

---

## Phase 3: Chaos Controller (FastAPI)

### Task 5: Build Chaos Controller — core framework

**Files:**
- Create: `testbench/chaos-controller/pyproject.toml`
- Create: `testbench/chaos-controller/src/__init__.py`
- Create: `testbench/chaos-controller/src/main.py`
- Create: `testbench/chaos-controller/src/config.py`
- Create: `testbench/chaos-controller/src/clients/__init__.py`
- Create: `testbench/chaos-controller/src/clients/argus.py`
- Create: `testbench/chaos-controller/src/clients/chaos.py`
- Create: `testbench/chaos-controller/Dockerfile`
- Create: `testbench/chaos-controller/railway.json`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "chaos-controller"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "pytest-httpx"]
```

**Step 2: Create config.py**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App URLs
    conduit_url: str = "https://conduit-testbench.up.railway.app"
    plane_url: str = "https://plane-testbench.up.railway.app"
    json_api_url: str = "https://jsonapi-testbench.up.railway.app"
    chaos_app_url: str = "https://chaos-testbench.up.railway.app"

    # Argus API
    argus_url: str = "https://argus-brain-production.up.railway.app"
    argus_api_key: str = ""
    argus_org_id: str = ""

    # Controller
    port: int = 8000

    class Config:
        env_prefix = "TESTBENCH_"
```

**Step 3: Create Argus client (argus.py)**

Async httpx client that wraps the Argus API. Methods map to the scenario steps:

```python
class ArgusClient:
    """Client for Argus API — simulates a real customer."""

    async def create_project(self, name: str, url: str) -> dict: ...
    async def run_discovery(self, project_id: str, url: str) -> dict: ...
    async def generate_tests(self, project_id: str) -> dict: ...
    async def run_tests(self, project_id: str) -> dict: ...
    async def get_test_results(self, run_id: str) -> dict: ...
    async def run_visual_capture(self, project_id: str, url: str) -> dict: ...
    async def run_visual_compare(self, baseline_id: str, url: str) -> dict: ...
    async def run_performance_test(self, project_id: str, url: str) -> dict: ...
    async def run_accessibility_audit(self, project_id: str, url: str) -> dict: ...
    async def run_pentest(self, project_id: str, url: str) -> dict: ...
    async def run_sast(self, project_id: str, repo_url: str) -> dict: ...
    async def get_healing_patterns(self, org_id: str) -> dict: ...
    async def get_flaky_tests(self, project_id: str) -> dict: ...
    async def generate_report(self, run_id: str) -> dict: ...
    async def run_api_discovery(self, project_id: str, spec_url: str) -> dict: ...
    async def run_api_tests(self, project_id: str) -> dict: ...
```

Each method makes the actual HTTP request to Argus with proper auth headers (`X-API-Key`, `X-Organization-Id`).

**Step 4: Create Chaos client (chaos.py)**

```python
class ChaosClient:
    """Client for Chaos App bug toggle API."""

    async def enable_bug(self, bug_id: str) -> dict: ...
    async def disable_bug(self, bug_id: str) -> dict: ...
    async def reset_all(self) -> dict: ...
    async def enable_scenario(self, scenario_name: str) -> dict: ...
    async def list_bugs(self) -> list[dict]: ...
    async def health(self) -> dict: ...
```

**Step 5: Create main.py (FastAPI app)**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI(title="Argus Testbench Controller", version="1.0.0")

@app.get("/health")
async def health(): ...

@app.get("/scenarios")
async def list_scenarios(): ...

@app.post("/scenarios/run")
async def run_scenario(request: ScenarioRunRequest): ...

@app.get("/scenarios/{run_id}/status")
async def scenario_status(run_id: str): ...

@app.get("/scenarios/{run_id}/stream")
async def scenario_stream(run_id: str): ...

@app.post("/scenarios/{run_id}/cancel")
async def cancel_scenario(run_id: str): ...

@app.post("/chaos/{app}/enable-bug/{bug_id}")
async def enable_bug(app: str, bug_id: str): ...

@app.post("/chaos/{app}/disable-bug/{bug_id}")
async def disable_bug(app: str, bug_id: str): ...

@app.post("/chaos/{app}/reset")
async def reset_bugs(app: str): ...

@app.get("/results")
async def list_results(): ...

@app.get("/results/summary")
async def results_summary(): ...

@app.get("/results/{run_id}")
async def get_result(run_id: str): ...
```

**Step 6: Create Dockerfile**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir .
COPY src/ ./src/
EXPOSE ${PORT:-8000}
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
```

Note: Use shell form CMD so `$PORT` is expanded by the shell.

```dockerfile
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

**Step 7: Commit**

```bash
git add testbench/chaos-controller/
git commit -m "feat(testbench): add chaos controller framework with Argus + Chaos clients"
```

---

### Task 6: Build Chaos Controller — scenario engine

**Files:**
- Create: `testbench/chaos-controller/src/scenarios/__init__.py`
- Create: `testbench/chaos-controller/src/scenarios/base.py`
- Create: `testbench/chaos-controller/src/scenarios/onboarding.py`
- Create: `testbench/chaos-controller/src/scenarios/healing.py`
- Create: `testbench/chaos-controller/src/scenarios/regression.py`
- Create: `testbench/chaos-controller/src/scenarios/security.py`
- Create: `testbench/chaos-controller/src/scenarios/flaky.py`
- Create: `testbench/chaos-controller/src/scenarios/golden_path.py`
- Create: `testbench/chaos-controller/src/scenarios/registry.py`

**Step 1: Create base scenario class (base.py)**

```python
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ScenarioStep:
    name: str
    description: str
    status: StepStatus = StepStatus.PENDING
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict | None = None

@dataclass
class ScenarioRun:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scenario: str = ""
    status: str = "pending"  # pending, running, passed, failed, cancelled
    steps: list[ScenarioStep] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

class BaseScenario:
    """Base class for all test bench scenarios."""

    name: str = ""
    description: str = ""
    category: str = ""

    def __init__(self, argus: ArgusClient, chaos: ChaosClient):
        self.argus = argus
        self.chaos = chaos
        self.run = ScenarioRun(scenario=self.name)

    async def execute(self) -> ScenarioRun:
        """Execute all steps in order. Override in subclasses."""
        raise NotImplementedError

    async def step(self, name: str, description: str, fn):
        """Execute a single step with status tracking."""
        step = ScenarioStep(name=name, description=description)
        self.run.steps.append(step)
        step.status = StepStatus.RUNNING
        step.started_at = datetime.utcnow()
        try:
            result = await fn()
            step.status = StepStatus.PASSED
            step.result = result
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            raise
        finally:
            step.completed_at = datetime.utcnow()
```

**Step 2: Create onboarding scenarios (onboarding.py)**

Three scenarios: `onboard-conduit`, `onboard-plane`, `onboard-api`.

Each follows the pattern:
1. Create an Argus project pointing at the app URL
2. Run discovery (auto-crawl for UI apps, OpenAPI spec for API)
3. Generate tests from discovered flows/endpoints
4. Execute tests
5. Verify all pass (no bugs enabled)
6. Generate a report

`onboard-api` uses `run_api_discovery(spec_url)` instead of `run_discovery(url)`.

**Step 3: Create healing scenarios (healing.py)**

Three scenarios: `heal-selector-drift`, `heal-cascade`, `heal-proactive`.

`heal-selector-drift`:
1. Run tests against Chaos App (verify pass)
2. `chaos.enable_bug("selector-login-id")`
3. Run tests again (verify login test fails)
4. Wait up to 60s for Argus healing to produce a fix
5. Verify healing pattern stored (`argus.get_healing_patterns()`)
6. Run healed tests (verify pass)
7. `chaos.reset_all()`

`heal-cascade`:
1. Enable 3 bugs: `selector-login-id`, `selector-class-rename`, `selector-nesting`
2. Run tests → expect 3 failures
3. Wait for healing
4. Verify all 3 healed with confidence scores > 0.5
5. Reset

`heal-proactive`:
1. Trigger proactive DOM diff scan
2. Enable `selector-nesting`
3. Trigger another scan
4. Verify Argus detects drift BEFORE running tests
5. Check health score decreased
6. Reset

**Step 4: Create regression scenarios (regression.py)**

Three scenarios: `regress-visual`, `regress-perf`, `regress-api`.

`regress-visual`:
1. Capture visual baseline of Chaos App home page
2. `chaos.enable_bug("a11y-low-contrast")`
3. Run visual comparison against baseline
4. Verify diff detected with AI explanation containing "contrast" or "color"
5. Reset

`regress-perf`:
1. Run performance baseline
2. `chaos.enable_bug("perf-slow-api")`
3. Run performance test
4. Verify LCP or TTFB regression flagged (value increased significantly)
5. Reset

`regress-api`:
1. Run API tests against JSON API (verify pass)
2. Enable `api-schema-drift` on Chaos App (but we test against chaos app API)
3. Run API tests again
4. Verify schema violation caught
5. Reset

**Step 5: Create security scenarios (security.py)**

Two scenarios: `security-scan`, `security-sast`.

`security-scan`:
1. `chaos.enable_scenario("security-vulns")` (enables all 6 security bugs)
2. Run pentest against Chaos App
3. Verify XSS, SQLi, and IDOR findings in results
4. Verify CVSS scores present
5. Reset

`security-sast`:
1. `chaos.enable_bug("sec-secrets-in-code")`
2. Run SAST analysis (if repo URL available, otherwise skip)
3. Verify hardcoded secret detected
4. Reset

**Step 6: Create flaky scenarios (flaky.py)**

Two scenarios: `flaky-detect`, `flaky-vs-real`.

`flaky-detect`:
1. `chaos.enable_bug("flaky-timing")`
2. Run same test suite 10 times (sequentially)
3. Verify flaky detector flags at least 1 test as flaky
4. Verify flakiness score > 0
5. Quarantine the flaky test
6. Verify quarantine applied
7. Reset

`flaky-vs-real`:
1. Enable `flaky-timing` + `selector-removed`
2. Run 10 times
3. Verify Argus classifies timing issue as FLAKY and removed element as REAL_BUG
4. Reset

**Step 7: Create golden path scenario (golden_path.py)**

The full 15-step journey (see design doc). Composes steps from the other scenarios:
1. Create project → 2. Discovery → 3. Generate tests → 4. Execute (all pass) → 5. Capture visual baseline → 6. Enable 5 bugs (one per category) → 7. Execute (failures) → 8. Verify healing → 9. Verify visual regression detected → 10. Verify perf regression → 11. Security scan → 12. Generate report → 13. Verify report completeness → 14. Reset all bugs → 15. Execute (all pass again)

**Step 8: Create scenario registry (registry.py)**

Maps scenario names to classes:

```python
SCENARIOS = {
    "onboard-conduit": OnboardConduit,
    "onboard-plane": OnboardPlane,
    "onboard-api": OnboardAPI,
    "heal-selector-drift": HealSelectorDrift,
    "heal-cascade": HealCascade,
    "heal-proactive": HealProactive,
    "regress-visual": RegressVisual,
    "regress-perf": RegressPerf,
    "regress-api": RegressAPI,
    "security-scan": SecurityScan,
    "security-sast": SecuritySAST,
    "flaky-detect": FlakyDetect,
    "flaky-vs-real": FlakyVsReal,
    "golden-path": GoldenPath,
}
```

**Step 9: Commit**

```bash
git add testbench/chaos-controller/src/scenarios/
git commit -m "feat(testbench): add 13 scenario definitions for chaos controller"
```

---

## Phase 4: Fork and Configure Real Apps

### Task 7: Set up Conduit (RealWorld)

**Files:**
- Create: `testbench/conduit/README.md`
- Create: `testbench/conduit/Dockerfile`
- Create: `testbench/conduit/railway.json`

**Step 1: Fork and vendor the RealWorld app**

We use the [node-express-realworld](https://github.com/gothinkster/node-express-realworld-example-app) backend + [react-redux-realworld](https://github.com/gothinkster/react-redux-realworld-example-app) frontend.

Option A (recommended): Use an existing full-stack fork like [danjac/realworld-react-express](https://github.com) or the official realworld starters.

Option B: Use [Conduit by Biscuit](https://github.com/TanStack/router/tree/main/examples/react/with-realworld) or similar maintained fork.

The exact fork depends on what's available and maintained at implementation time. Key requirements:
- React frontend (not Angular/Vue)
- Express or Node backend
- PostgreSQL (not MongoDB)
- Standard RealWorld API spec

**Step 2: Create unified Dockerfile**

```dockerfile
# Similar to Chaos App — multi-stage build
# Stage 1: Build React frontend
# Stage 2: Express server serving built frontend + API
```

**Step 3: Create railway.json**

Same pattern as JSON API — DOCKERFILE builder, `/health` healthcheck.

**Step 4: Verify locally**

```bash
cd testbench/conduit
docker build -t conduit-test .
docker run -p 3002:3002 -e DATABASE_URL=... conduit-test
# Verify: http://localhost:3002 shows Conduit home page
# Verify: Can register, login, post article, comment
```

**Step 5: Commit**

```bash
git add testbench/conduit/
git commit -m "feat(testbench): add Conduit (RealWorld) app"
```

---

### Task 8: Set up Plane

**Files:**
- Create: `testbench/plane/README.md`
- Create: `testbench/plane/docker-compose.railway.yml` (Railway template)
- Create: `testbench/plane/railway.json`

**Step 1: Fork Plane**

Plane is a large app (Next.js + Django + Celery + Redis + PostgreSQL). Railway deployment options:

Option A (recommended): Use Plane's [official Docker images](https://github.com/makeplane/plane/tree/master/deploy/selfhost) — they provide `plane-frontend`, `plane-backend`, `plane-worker` images.

Option B: Use Plane's one-click deploy template if Railway has one.

**Step 2: Configure for Railway**

Plane needs 3 services + 2 databases. Create separate Railway services:
- `plane-web` — Next.js frontend
- `plane-api` — Django backend
- `plane-worker` — Celery worker
- Postgres and Redis as Railway plugins

Set env vars:
- `DATABASE_URL`, `REDIS_URL` (Railway provides these)
- `SECRET_KEY`, `NEXT_PUBLIC_API_BASE_URL`
- `WEB_URL`, `CORS_ALLOWED_ORIGINS`

**Step 3: Verify locally**

```bash
cd testbench/plane
docker compose -f docker-compose.railway.yml up
# Verify: Workspace creation, project creation, issue CRUD
```

**Step 4: Commit**

```bash
git add testbench/plane/
git commit -m "feat(testbench): add Plane project management app"
```

---

## Phase 5: Deploy Script + CI Pipeline

### Task 9: Create deploy script

**Files:**
- Create: `testbench/deploy.sh`

**Step 1: Create deploy.sh**

Follow the same pattern as `fly-data/deploy.sh` and `fly-selenium/deploy.sh` but for Railway CLI:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Parse args: --only [json-api|chaos-app|chaos-controller|conduit|plane]
# Default: deploy all

RAILWAY_PROJECT="argus-testbench"

deploy_service() {
    local name=$1
    local path=$2
    echo "Deploying $name..."
    railway link -p "$RAILWAY_PROJECT" -s "$name"
    railway up --detach -d "$path"
    echo "Waiting for $name to be healthy..."
    # Poll health endpoint
}

# Order matters: databases first, then apps, then controller
deploy_service "jsonapi-testbench" "testbench/json-api"
deploy_service "chaos-testbench" "testbench/chaos-app"
deploy_service "conduit-testbench" "testbench/conduit"
deploy_service "plane-testbench-web" "testbench/plane"
# Controller last (depends on all apps being up)
deploy_service "chaos-ctrl-testbench" "testbench/chaos-controller"

echo ""
echo "=== Test Bench Deployed ==="
echo "JSON API:          https://jsonapi-testbench.up.railway.app"
echo "Chaos App:         https://chaos-testbench.up.railway.app"
echo "Conduit:           https://conduit-testbench.up.railway.app"
echo "Plane:             https://plane-testbench.up.railway.app"
echo "Chaos Controller:  https://chaos-ctrl-testbench.up.railway.app"
```

**Step 2: Make executable and commit**

```bash
chmod +x testbench/deploy.sh
git add testbench/deploy.sh
git commit -m "feat(testbench): add Railway deploy script"
```

---

### Task 10: Create GitHub Actions nightly pipeline

**Files:**
- Create: `testbench/.github/workflows/testbench-nightly.yml`

**Step 1: Create workflow**

```yaml
name: Argus Testbench Nightly
on:
  schedule:
    - cron: '0 3 * * *'  # 3 AM UTC daily
  workflow_dispatch:
    inputs:
      scenario:
        description: 'Specific scenario to run (default: all)'
        required: false
        default: 'all'

env:
  CONTROLLER_URL: https://chaos-ctrl-testbench.up.railway.app

jobs:
  testbench:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - name: Health check all services
        run: |
          for url in \
            "https://jsonapi-testbench.up.railway.app/health" \
            "https://chaos-testbench.up.railway.app/health" \
            "https://conduit-testbench.up.railway.app/health" \
            "https://chaos-ctrl-testbench.up.railway.app/health"; do
            echo "Checking $url..."
            curl -sf "$url" || echo "WARN: $url unhealthy"
          done

      - name: Reset all bugs
        run: curl -sf -X POST "$CONTROLLER_URL/chaos/chaos-app/reset"

      - name: Run scenarios
        id: scenarios
        run: |
          if [ "${{ github.event.inputs.scenario }}" = "all" ] || [ -z "${{ github.event.inputs.scenario }}" ]; then
            SCENARIOS="onboard-conduit onboard-api heal-selector-drift heal-cascade regress-visual regress-perf regress-api security-scan flaky-detect golden-path"
          else
            SCENARIOS="${{ github.event.inputs.scenario }}"
          fi

          PASSED=0
          FAILED=0
          RESULTS=""

          for scenario in $SCENARIOS; do
            echo "Running scenario: $scenario"
            RESPONSE=$(curl -sf -X POST "$CONTROLLER_URL/scenarios/run" \
              -H "Content-Type: application/json" \
              -d "{\"scenario\": \"$scenario\"}" || echo '{"status":"error"}')

            RUN_ID=$(echo "$RESPONSE" | jq -r '.run_id // "unknown"')

            # Poll for completion (max 10 min per scenario)
            for i in $(seq 1 60); do
              STATUS=$(curl -sf "$CONTROLLER_URL/scenarios/$RUN_ID/status" | jq -r '.status')
              if [ "$STATUS" = "passed" ] || [ "$STATUS" = "failed" ]; then
                break
              fi
              sleep 10
            done

            if [ "$STATUS" = "passed" ]; then
              PASSED=$((PASSED + 1))
              RESULTS="$RESULTS\n:white_check_mark: $scenario"
            else
              FAILED=$((FAILED + 1))
              RESULTS="$RESULTS\n:x: $scenario"
            fi
          done

          echo "passed=$PASSED" >> $GITHUB_OUTPUT
          echo "failed=$FAILED" >> $GITHUB_OUTPUT
          echo "results<<EOF" >> $GITHUB_OUTPUT
          echo -e "$RESULTS" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT

      - name: Post to Slack
        if: always()
        uses: slackapi/slack-github-action@v2.0.0
        with:
          webhook: ${{ secrets.TESTBENCH_SLACK_WEBHOOK }}
          webhook-type: incoming-webhook
          payload: |
            {
              "text": "*Argus Testbench Nightly* — ${{ steps.scenarios.outputs.passed }} passed, ${{ steps.scenarios.outputs.failed }} failed\n${{ steps.scenarios.outputs.results }}"
            }

      - name: Fail if any scenario failed
        if: steps.scenarios.outputs.failed != '0'
        run: exit 1
```

**Step 2: Commit**

```bash
git add testbench/.github/
git commit -m "feat(testbench): add nightly GitHub Actions pipeline"
```

---

## Phase 6: Deploy and Verify

### Task 11: Create Railway project and deploy all services

**Step 1: Create Railway project**

```bash
railway login
railway init --name argus-testbench
```

**Step 2: Create services**

Create each service in the Railway dashboard or via CLI:
- `jsonapi-testbench`
- `chaos-testbench` + Postgres plugin
- `conduit-testbench` + Postgres plugin
- `plane-testbench-web`, `plane-testbench-api`, `plane-testbench-worker` + Postgres + Redis plugins
- `chaos-ctrl-testbench`

**Step 3: Set environment variables**

For Chaos Controller:
```bash
railway variables --set "TESTBENCH_ARGUS_URL=https://argus-brain-production.up.railway.app"
railway variables --set "TESTBENCH_ARGUS_API_KEY=argus_sk_..."
railway variables --set "TESTBENCH_ARGUS_ORG_ID=229904aa-..."
railway variables --set "TESTBENCH_CHAOS_APP_URL=https://chaos-testbench.up.railway.app"
railway variables --set "TESTBENCH_CONDUIT_URL=https://conduit-testbench.up.railway.app"
railway variables --set "TESTBENCH_PLANE_URL=https://plane-testbench.up.railway.app"
railway variables --set "TESTBENCH_JSON_API_URL=https://jsonapi-testbench.up.railway.app"
```

**Step 4: Deploy**

```bash
./testbench/deploy.sh
```

**Step 5: Verify all services healthy**

```bash
for url in \
  "https://jsonapi-testbench.up.railway.app/health" \
  "https://chaos-testbench.up.railway.app/health" \
  "https://conduit-testbench.up.railway.app/health" \
  "https://chaos-ctrl-testbench.up.railway.app/health"; do
  echo "$url: $(curl -sf $url | jq -r '.status')"
done
```

**Step 6: Run smoke scenario**

```bash
# Run the simplest scenario first
curl -X POST https://chaos-ctrl-testbench.up.railway.app/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{"scenario": "onboard-api"}'
```

**Step 7: Run golden path**

```bash
curl -X POST https://chaos-ctrl-testbench.up.railway.app/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{"scenario": "golden-path"}'
```

**Step 8: Commit any deployment fixes**

```bash
git add .
git commit -m "fix(testbench): deployment configuration adjustments"
```

---

## Implementation Order Summary

| Phase | Task | Description | Est. Time |
|-------|------|-------------|-----------|
| 1 | 1 | Scaffold directories | 5 min |
| 1 | 2 | JSON API server | 30 min |
| 2 | 3 | Chaos App backend (bugs + API) | 60 min |
| 2 | 4 | Chaos App frontend (React) | 45 min |
| 3 | 5 | Chaos Controller framework | 30 min |
| 3 | 6 | Scenario engine (13 scenarios) | 60 min |
| 4 | 7 | Conduit (RealWorld) fork | 30 min |
| 4 | 8 | Plane fork | 30 min |
| 5 | 9 | Deploy script | 15 min |
| 5 | 10 | GitHub Actions pipeline | 15 min |
| 6 | 11 | Deploy + verify | 30 min |

**Total estimated: ~6 hours of implementation work.**

Build order rationale: JSON API first (simplest, validates Railway deploy pattern), then Chaos App (core value, most custom code), then Controller (needs Chaos App running), then real apps (forks, less custom), then deploy script + CI (ties everything together).
