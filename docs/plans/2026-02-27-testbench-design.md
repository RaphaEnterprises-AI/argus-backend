# Argus Test Bench — Design Document

**Date:** 2026-02-27
**Status:** Approved
**Goal:** Deploy a portfolio of sample apps on Railway that simulate real customers using Argus, exercising every feature end-to-end with programmable bug injection and automated nightly validation.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARGUS TEST BENCH                                  │
│                     Railway Project: argus-testbench                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │  App 1: Conduit   │  │  App 2: Plane    │  │  App 3: JSON API │      │
│  │  (Simple Blog)    │  │  (Project Mgmt)  │  │  (Pure REST)     │      │
│  │  React + Express  │  │  Next.js + Node   │  │  Express + OpenAPI│      │
│  │  PostgreSQL       │  │  PostgreSQL       │  │  SQLite/Memory    │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────────────────────────────┐     │
│  │  App 4: Chaos     │  │  App 5: Chaos Controller                │     │
│  │  (Bug Injection)  │  │  (Scenario Orchestrator)                │     │
│  │  React + Express  │  │  FastAPI + Scenario Engine              │     │
│  │  Toggleable bugs  │  │  Calls Argus API as customer            │     │
│  └──────────────────┘  └──────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────┐     ┌───────────────────────────┐
│  GitHub Actions   │     │  Argus Production API      │
│  Nightly Pipeline │────▶│  argus-brain-production    │
│                   │     │  .up.railway.app           │
│  • Run scenarios  │     └───────────────────────────┘
│  • Cycle bugs     │                  │
│  • Verify healing │                  ▼
│  • Post to Slack  │     ┌───────────────────────────┐
└──────────────────┘     │  Slack #argus-dogfood      │
                          └───────────────────────────┘
```

**5 services, ~$20/mo on Railway.**

---

## App Details

### App 1: Conduit (RealWorld) — Simple Blog

- **Source:** Fork of [gothinkster/realworld](https://github.com/gothinkster/realworld)
- **Stack:** React frontend + Node/Express API + PostgreSQL
- **Pages:** Home feed, login, register, article editor, article view + comments, user profile, settings
- **API:** Full RealWorld REST spec (users, articles, comments, tags, profiles, favorites)
- **Railway:** shared-cpu-1x 256MB + Postgres (~$2/mo)
- **Argus features exercised:** Discovery, UI testing, API testing, visual regression, accessibility, test generation, parameterized testing, reporting, export

### App 2: Plane — Complex Project Management

- **Source:** Fork of [makeplane/plane](https://github.com/makeplane/plane)
- **Stack:** Next.js frontend + Django API + PostgreSQL + Redis
- **Pages:** Workspace onboarding, project creation, issue CRUD, Kanban board, list/spreadsheet, cycles, modules, pages/wiki, settings + members
- **Railway:** shared-cpu-2x 512MB (web) + shared-cpu-1x 512MB (api) + shared-cpu-1x 256MB (worker) + Postgres + Redis (~$12/mo)
- **Argus features exercised:** Complex flow discovery, multi-step UI testing, visual regression (board layouts), performance, CI/CD webhooks, test impact analysis

### App 3: JSON API Server — Pure REST

- **Source:** Custom Express server with auto-generated OpenAPI 3.0 spec
- **Stack:** Express + Swagger UI + SQLite (in-memory)
- **Endpoints:** CRUD for users, posts, comments, todos + `/api/v1/openapi.json`
- **Railway:** shared-cpu-1x 256MB (~$1/mo)
- **Argus features exercised:** API discovery from OpenAPI spec, schema validation, API test generation, contract testing

### App 4: Chaos App — Bug Injection Target

- **Source:** Custom-built (React Vite + Express + PostgreSQL)
- **Pages:** Product listing, product detail, cart, multi-step checkout, login/register, profile, search, admin
- **30 toggleable bugs** across 6 categories (see Bug Catalog below)
- **Railway:** shared-cpu-1x 256MB + Postgres (~$2/mo)
- **Argus features exercised:** Self-healing, proactive healing, flaky detection, failure patterns, security/pentest, SAST, performance regression, accessibility violations

### App 5: Chaos Controller — Scenario Orchestrator

- **Source:** Custom-built (Python FastAPI)
- **API:** Scenario management, chaos control, results/history, health
- **13 scenarios** covering every Argus feature (see Scenario Engine below)
- **Railway:** shared-cpu-1x 256MB (~$1/mo)

---

## Chaos App — Bug Catalog (30 Bugs)

### Category 1: UI / Selector Drift (Self-Healing)

| Bug ID | Description | Argus Feature |
|--------|-------------|---------------|
| `selector-login-id` | Login button `#login-btn` → `#sign-in-button` | Self-healing, healing patterns |
| `selector-class-rename` | `.add-to-cart` → `.btn-add-cart` | Selector healing |
| `selector-nesting` | Button moves deeper in DOM tree | Code-aware healing |
| `selector-dynamic-id` | Element ID becomes random | Healing suggest |
| `selector-removed` | Checkout button deleted entirely | Failure patterns, root cause |

### Category 2: Performance Degradation

| Bug ID | Description | Argus Feature |
|--------|-------------|---------------|
| `perf-slow-api` | 5s delay on `/api/products` | Performance analyzer, SLO |
| `perf-memory-leak` | Frontend leaks DOM nodes | Core Web Vitals |
| `perf-large-payload` | 10MB JSON response | Performance testing |
| `perf-blocking-render` | 3s render-blocking script | Visual regression, performance |

### Category 3: Accessibility Violations

| Bug ID | Description | Argus Feature |
|--------|-------------|---------------|
| `a11y-no-alt` | Remove all image alt attributes | WCAG 1.1.1 audit |
| `a11y-low-contrast` | Text color `#ccc` on `#fff` | WCAG 1.4.3 audit |
| `a11y-no-labels` | Remove form input labels | WCAG 1.3.1 audit |
| `a11y-no-keyboard` | Disable keyboard focus on nav | WCAG 2.1.1 audit |
| `a11y-missing-aria` | Remove ARIA landmarks/roles | WCAG 4.1.2 audit |

### Category 4: Security Vulnerabilities

| Bug ID | Description | Argus Feature |
|--------|-------------|---------------|
| `sec-xss-reflected` | Search reflects query unescaped | Pentest, security scanner |
| `sec-sqli` | Login concatenates SQL | Pentest, SAST |
| `sec-broken-auth` | API accepts expired JWT | Security scanner |
| `sec-idor` | User endpoint lacks ownership check | Pentest, API testing |
| `sec-ssrf` | Image fetcher follows internal URLs | Security scanner |
| `sec-secrets-in-code` | API key hardcoded in frontend JS | SAST analysis |

### Category 5: API Contract Violations

| Bug ID | Description | Argus Feature |
|--------|-------------|---------------|
| `api-schema-drift` | `price` field changes number→string | API testing, schema validation |
| `api-missing-field` | Remove `email` from response | API testing |
| `api-wrong-status` | POST returns 200 instead of 201 | API testing |
| `api-breaking-pagination` | `?page=` → `?offset=` | API testing, self-healing |
| `api-500-random` | 10% of requests return 500 | Flaky detection |

### Category 6: Flaky Behavior

| Bug ID | Description | Argus Feature |
|--------|-------------|---------------|
| `flaky-timing` | Random 0-3s page load delay | Flaky detector |
| `flaky-animation` | Button clickable after random animation | Flaky detection |
| `flaky-race-condition` | Cart count updates async, sometimes stale | Flaky trend |
| `flaky-network` | 20% of API calls drop connection | Failure patterns |

### Toggle API

```
GET  /chaos/bugs              → list all bugs + state
POST /chaos/bugs/:id/enable   → activate bug
POST /chaos/bugs/:id/disable  → deactivate bug
POST /chaos/bugs/reset        → disable all
POST /chaos/bugs/scenario/:name → enable preset group
GET  /chaos/health            → health + active bug count
```

---

## Scenario Engine (Chaos Controller)

### 13 Scenarios

**Category A: Customer Onboarding**

| Scenario | Steps | Features Tested |
|----------|-------|-----------------|
| `onboard-conduit` | Create project → Discovery → Generate tests → Execute → Report | Discovery, test gen, execution, reporting |
| `onboard-plane` | Same against Plane (complex SaaS) | Complex flow discovery |
| `onboard-api` | Create project → API discovery (OpenAPI) → Generate → Execute → Validate | API discovery, schema validation |

**Category B: Self-Healing**

| Scenario | Steps | Features Tested |
|----------|-------|-----------------|
| `heal-selector-drift` | Pass → Enable bug → Fail → Heal → Verify → Check patterns | Self-healing, Cognee learning |
| `heal-cascade` | Enable 3 selector bugs → Run → Verify all healed | Multi-failure healing |
| `heal-proactive` | Trigger DOM diff scan → Enable bug → Verify pre-emptive detection | Proactive healing CI/CD |

**Category C: Regression Detection**

| Scenario | Steps | Features Tested |
|----------|-------|-----------------|
| `regress-visual` | Baseline → Enable a11y bug → Compare → Verify diff | Visual AI, baselines |
| `regress-perf` | Baseline → Enable perf bug → Test → Verify regression | Performance, SLO |
| `regress-api` | Pass → Enable schema drift → Test → Verify caught | API contract testing |

**Category D: Security**

| Scenario | Steps | Features Tested |
|----------|-------|-----------------|
| `security-scan` | Enable XSS+SQLi+IDOR → Pentest → Verify all found | Pentest, OWASP |
| `security-sast` | Enable secrets bug → SAST scan → Verify detected | SAST, code scanning |

**Category E: Flaky Detection**

| Scenario | Steps | Features Tested |
|----------|-------|-----------------|
| `flaky-detect` | Enable timing bug → Run 10x → Verify flagged + quarantine | Flaky detection, quarantine |
| `flaky-vs-real` | Enable timing + removed element → Run 10x → Verify distinction | Root cause analysis |

**Category F: Full Lifecycle**

| Scenario | Steps | Features Tested |
|----------|-------|-----------------|
| `golden-path` | Full 15-step journey: create → discover → test → baseline → inject 5 bugs → heal → detect regressions → security scan → report → reset → verify clean | **Every major feature** |

### Controller API

```
POST /scenarios/run           → start async scenario run
GET  /scenarios               → list available scenarios
GET  /scenarios/{id}/status   → running/passed/failed + steps
GET  /scenarios/{id}/stream   → SSE real-time progress
POST /scenarios/{id}/cancel   → abort

POST /chaos/{app}/enable-bug/{bug_id}
POST /chaos/{app}/disable-bug/{bug_id}
POST /chaos/{app}/reset

GET  /results                 → all runs with pass/fail
GET  /results/summary         → aggregate pass rate
GET  /results/{run_id}        → step-by-step detail

GET  /health                  → controller + all apps
```

---

## CI Pipeline — Nightly

```yaml
name: Argus Testbench Nightly
on:
  schedule:
    - cron: '0 3 * * *'
  workflow_dispatch: {}

jobs:
  testbench:
    runs-on: ubuntu-latest
    steps:
      - name: Run Golden Path
        run: curl -X POST $CONTROLLER_URL/scenarios/run -d '{"scenario":"golden-path"}'

      - name: Run Healing Scenarios
        run: |
          for s in heal-selector-drift heal-cascade heal-proactive; do
            curl -X POST $CONTROLLER_URL/scenarios/run -d "{\"scenario\":\"$s\"}"
          done

      - name: Run Security Scenarios
        run: curl -X POST $CONTROLLER_URL/scenarios/run -d '{"scenario":"security-scan"}'

      - name: Collect Results
        run: curl $CONTROLLER_URL/results/summary > results.json

      - name: Post to Slack
        if: always()
        uses: slackapi/slack-github-action@v2
        with:
          payload: '{"text":"Argus Testbench Nightly: see results"}'
```

---

## Feature Coverage Matrix

| Argus Feature | Conduit | Plane | JSON API | Chaos App | Scenario |
|---|---|---|---|---|---|
| Discovery (auto-crawl) | x | x | | x | `onboard-*` |
| UI Test Execution | x | x | | x | `onboard-*`, `golden-path` |
| API Test Discovery | x | | x | x | `onboard-api` |
| API Test Execution | x | | x | x | `onboard-api`, `regress-api` |
| Visual Regression | x | x | | x | `regress-visual` |
| Accessibility Audit | x | x | | x | `regress-visual` |
| Performance Testing | x | | | x | `regress-perf` |
| Security/Pentest | | | | x | `security-scan` |
| SAST | | | | x | `security-sast` |
| Self-Healing | | | | x | `heal-*` |
| Proactive Healing | | | | x | `heal-proactive` |
| Flaky Detection | | | | x | `flaky-detect` |
| Failure Patterns | | | | x | `flaky-vs-real` |
| Root Cause Analysis | | | | x | `heal-cascade` |
| Test Generation | x | x | x | | `onboard-*` |
| Parameterized Tests | x | | x | | `onboard-conduit` |
| Reporting | x | x | x | x | `golden-path` |
| CI/CD Webhooks | x | x | | | Nightly pipeline |
| Quality Scoring | x | x | x | x | `golden-path` |
| SLO Monitoring | | | | x | `regress-perf` |
| Chat Interface | x | x | x | x | Future |
| Swarm Execution | x | x | | | Future |
| Export (code gen) | x | | x | | `onboard-*` |
| Test Impact Analysis | x | x | | | CI pipeline |
| NLP Test Creation | x | x | | x | Future |
| Incident Correlation | | | | x | Future |

**26/26 features covered.**

---

## Repository Structure

```
testbench/
├── README.md
├── docker-compose.yml
├── deploy.sh
├── conduit/
│   ├── Dockerfile
│   ├── railway.toml
│   └── ...
├── plane/
│   ├── Dockerfile
│   ├── railway.toml
│   └── ...
├── json-api/
│   ├── Dockerfile
│   ├── railway.toml
│   ├── src/
│   └── package.json
├── chaos-app/
│   ├── Dockerfile
│   ├── railway.toml
│   ├── frontend/
│   ├── backend/
│   │   └── src/
│   │       ├── server.js
│   │       ├── routes/
│   │       ├── bugs/
│   │       │   ├── registry.js
│   │       │   ├── selectors.js
│   │       │   ├── performance.js
│   │       │   ├── a11y.js
│   │       │   ├── security.js
│   │       │   ├── api.js
│   │       │   └── flaky.js
│   │       └── middleware/
│   │           └── chaos.js
│   └── docker-compose.yml
├── chaos-controller/
│   ├── Dockerfile
│   ├── railway.toml
│   ├── requirements.txt
│   └── src/
│       ├── main.py
│       ├── scenarios/
│       │   ├── base.py
│       │   ├── onboarding.py
│       │   ├── healing.py
│       │   ├── regression.py
│       │   ├── security.py
│       │   ├── flaky.py
│       │   └── golden_path.py
│       ├── clients/
│       │   ├── argus.py
│       │   └── chaos.py
│       └── config.py
└── .github/
    └── workflows/
        └── testbench-nightly.yml
```

---

## Railway Cost Estimate

| Service | VM | Memory | Cost |
|---------|-----|--------|------|
| Conduit (monolith) | shared-cpu-1x | 256MB | $2/mo |
| Conduit Postgres | plugin | — | $1/mo |
| Plane Web | shared-cpu-2x | 512MB | $4/mo |
| Plane API | shared-cpu-1x | 512MB | $4/mo |
| Plane Worker | shared-cpu-1x | 256MB | $2/mo |
| Plane Postgres | plugin | — | $1/mo |
| Plane Redis | plugin | — | $1/mo |
| JSON API | shared-cpu-1x | 256MB | $1/mo |
| Chaos App | shared-cpu-1x | 256MB | $2/mo |
| Chaos App Postgres | plugin | — | $1/mo |
| Chaos Controller | shared-cpu-1x | 256MB | $1/mo |
| **Total** | | | **~$20/mo** |

---

## Rollback / Risk

- All apps are isolated in their own Railway project — no impact on production Argus
- Chaos bugs are opt-in (all disabled by default) — apps are healthy unless explicitly broken
- Nightly pipeline has no destructive operations — read-only against Argus (creates test projects, runs tests)
- Conduit and Plane are forks — we control the code, no upstream surprises
