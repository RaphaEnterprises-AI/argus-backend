# Argus Platform - Complete Technical Architecture

**Version:** 1.0.0
**Last Updated:** January 2026
**Status:** Production Ready

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [Layer-by-Layer Breakdown](#4-layer-by-layer-breakdown)
5. [Data Flow Diagrams](#5-data-flow-diagrams)
6. [Database Schema](#6-database-schema)
7. [API Reference](#7-api-reference)
8. [Agent System](#8-agent-system)
9. [External Integrations](#9-external-integrations)
10. [Security Architecture](#10-security-architecture)
11. [Cost Management](#11-cost-management)
12. [Deployment Architecture](#12-deployment-architecture)

---

## 1. Executive Summary

**Argus** is an autonomous end-to-end testing platform that combines:
- **AI-Powered Test Generation** from production errors
- **Self-Healing Test Execution** with automatic selector repair
- **Multi-Browser Cross-Device Testing** via edge computing
- **Quality Intelligence** correlating errors with test coverage gaps

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Autonomous Testing** | AI generates, executes, and maintains tests |
| **Production Error → Test** | Converts Sentry/Datadog errors to regression tests |
| **Self-Healing** | Automatically fixes broken selectors |
| **Multi-Model AI** | Routes tasks to optimal model (60-80% cost savings) |
| **Edge Browser Automation** | Global low-latency browser execution |
| **Quality Intelligence** | Risk scoring, coverage gaps, predictive insights |

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL SOURCES                                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Sentry  │ │ Datadog │ │FullStory│ │LogRocket│ │ GitHub  │ │  Slack  │       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
└───────┼──────────┼──────────┼──────────┼──────────┼──────────┼─────────────────┘
        │          │          │          │          │          │
        └──────────┴──────────┴──────────┼──────────┴──────────┘
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CLOUDFLARE EDGE LAYER                                    │
│                         (argus-api Worker)                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Browser Automation Endpoints                                            │    │
│  │  POST /act     - Execute single action (click, type, scroll)            │    │
│  │  POST /extract - Extract structured data from page                       │    │
│  │  POST /observe - Discover available actions/elements                     │    │
│  │  POST /agent   - Run autonomous multi-step workflows                     │    │
│  │  POST /test    - Cross-browser multi-device test execution              │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │   Browser    │ │  KV Cache    │ │ R2 Storage   │ │  Vectorize   │           │
│  │  Rendering   │ │  (sessions,  │ │ (screenshots │ │  (semantic   │           │
│  │  (Chromium)  │ │   dedup)     │ │  artifacts)  │ │   search)    │           │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘           │
│                                                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │   Queues     │ │  Hyperdrive  │ │   Workers    │ │   Durable    │           │
│  │  (async      │ │  (DB pool)   │ │     AI       │ │   Objects    │           │
│  │   events)    │ │              │ │   (Llama)    │ │  (WebSocket) │           │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘           │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RAILWAY BRAIN SERVICE                                    │
│                         (Python FastAPI + LangGraph)                             │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  API Layer (src/api/)                                                    │    │
│  │  ├── server.py      - Main FastAPI application                          │    │
│  │  ├── webhooks.py    - Platform webhook handlers                         │    │
│  │  └── quality.py     - Quality Intelligence API                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Orchestration Layer (src/orchestrator/)                                 │    │
│  │  ├── graph.py       - LangGraph state machine                           │    │
│  │  ├── state.py       - Shared state definitions                          │    │
│  │  └── nodes.py       - Graph node implementations                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Agent System (src/agents/) - 20+ Specialized Agents                     │    │
│  │  ├── code_analyzer.py    - Static analysis, test surface discovery      │    │
│  │  ├── test_planner.py     - Prioritized test plan generation             │    │
│  │  ├── ui_tester.py        - Browser-based UI test execution              │    │
│  │  ├── api_tester.py       - REST API testing with schema validation      │    │
│  │  ├── self_healer.py      - Auto-fix broken tests and selectors          │    │
│  │  ├── quality_auditor.py  - Quality metrics and coverage analysis        │    │
│  │  └── ... (15+ more agents)                                              │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Core Intelligence (src/core/)                                           │    │
│  │  ├── model_router.py     - Multi-model AI routing (cost optimization)   │    │
│  │  ├── cognitive_engine.py - Multi-step reasoning                         │    │
│  │  ├── correlator.py       - Error pattern correlation                    │    │
│  │  ├── normalizer.py       - Error message normalization                  │    │
│  │  ├── coverage.py         - Test coverage calculation                    │    │
│  │  └── risk.py             - Risk assessment scoring                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Services Layer (src/services/)                                          │    │
│  │  ├── supabase_client.py  - Database operations                          │    │
│  │  ├── cache.py            - Cloudflare KV caching                        │    │
│  │  ├── vectorize.py        - Semantic search (Cloudflare Vectorize)       │    │
│  │  └── ai_cost_tracker.py  - Token/cost tracking                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SUPABASE (PostgreSQL)                                    │
│                         Source of Truth                                          │
│                                                                                  │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐   │
│  │ organizations  │ │production_events│ │ generated_tests│ │  risk_scores   │   │
│  │ (multi-tenant) │ │ (error data)   │ │ (AI-generated) │ │ (assessments)  │   │
│  └────────────────┘ └────────────────┘ └────────────────┘ └────────────────┘   │
│                                                                                  │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐   │
│  │   ci_events    │ │coverage_reports│ │  ai_usage      │ │healing_patterns│   │
│  │ (CI/CD data)   │ │ (test coverage)│ │ (cost tracking)│ │ (self-healing) │   │
│  └────────────────┘ └────────────────┘ └────────────────┘ └────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

### 3.1 Complete Stack Overview

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Edge** | Cloudflare Workers (TypeScript) | Browser automation, low-latency API |
| **Backend** | Python 3.11 + FastAPI | AI orchestration, business logic |
| **Orchestration** | LangGraph | Multi-agent state machine |
| **Database** | Supabase (PostgreSQL) | Persistent storage, RLS |
| **Cache** | Cloudflare KV | Session, dedup, API cache |
| **Vector DB** | Cloudflare Vectorize | Semantic error matching |
| **Object Storage** | Cloudflare R2 | Screenshots, artifacts |
| **Queue** | Cloudflare Queues | Async event processing |
| **AI Primary** | Claude (Anthropic) | Sonnet 4.5, Haiku 4.5, Opus 4.5 |
| **AI Fallback** | OpenAI, Groq, Together | GPT-4o, Llama 3.1 |
| **Browser** | Cloudflare Browser Rendering | Chromium automation |
| **Browser (Paid)** | TestingBot | Cross-browser + real devices |

### 3.2 Python Dependencies

```toml
# Core AI
anthropic>=0.75.0        # Claude API client
langgraph>=1.0.5         # Multi-agent orchestration
langchain-anthropic>=1.3.0
langchain-core>=1.2.5

# Web Automation
playwright>=1.48.0       # Browser automation
selenium>=4.25.0         # Fallback browser driver
httpx>=0.27.0            # Async HTTP client

# Data Validation
pydantic>=2.9.0          # Schema validation
pydantic-settings>=2.5.0 # Config management

# Database
sqlalchemy>=2.0.0        # ORM
asyncpg>=0.29.0          # Async PostgreSQL

# API Server
fastapi>=0.115.0         # REST API framework
uvicorn>=0.32.0          # ASGI server

# Utilities
structlog>=24.4.0        # Structured logging
tiktoken>=0.8.0          # Token counting
pillow>=10.4.0           # Image processing
```

### 3.3 Cloudflare Services Used

| Service | Binding | Purpose |
|---------|---------|---------|
| **Browser Rendering** | `BROWSER` | Chromium browser instances |
| **Workers AI** | `AI` | Llama fallback, embeddings |
| **KV** | `CACHE` | Key-value caching |
| **R2** | `ARTIFACTS` | Screenshot/artifact storage |
| **Vectorize** | `VECTOR_INDEX` | Vector similarity search |
| **Queues** | `EVENT_QUEUE`, `DLQ` | Async processing |
| **Hyperdrive** | `DB` | PostgreSQL connection pooling |
| **Durable Objects** | `REALTIME` | WebSocket state |

---

## 4. Layer-by-Layer Breakdown

### 4.1 Edge Layer (Cloudflare Worker)

**Location:** `/cloudflare-worker/src/index.ts`

```
cloudflare-worker/
├── src/
│   ├── index.ts          # Main worker entry point
│   ├── utils.ts          # Cache/storage helpers
│   └── realtime.ts       # WebSocket Durable Object
├── wrangler.toml         # Cloudflare configuration
└── package.json
```

**Capabilities:**
- Browser automation (Playwright on Cloudflare)
- Self-healing selectors with fallback generation
- AI-powered element discovery
- Multi-device/browser testing via TestingBot
- Real-time WebSocket updates

**Supported Device Presets:**
```typescript
DEVICE_PRESETS = {
  // Desktop
  "desktop": { width: 1920, height: 1080 },
  "desktop-hd": { width: 2560, height: 1440 },
  "desktop-mac": { width: 1920, height: 1080 },

  // Tablets
  "tablet": { width: 768, height: 1024 },
  "tablet-landscape": { width: 1024, height: 768 },

  // Mobile
  "mobile": { width: 375, height: 812 },
  "mobile-android": { width: 412, height: 915 },

  // Real Devices (TestingBot)
  "iphone-15", "iphone-14", "pixel-8", "samsung-s24"
}
```

### 4.2 Backend Layer (Python/Railway)

**Location:** `/src/`

```
src/
├── main.py                 # CLI entry point
├── config.py               # Configuration management (150+ settings)
│
├── api/                    # FastAPI REST API
│   ├── server.py           # Main application, all endpoints
│   ├── webhooks.py         # Sentry/Datadog/etc webhook handlers
│   └── quality.py          # Quality Intelligence API
│
├── orchestrator/           # LangGraph State Machine
│   ├── graph.py            # Graph definition and routing
│   ├── state.py            # TestingState TypedDict
│   └── nodes.py            # Node implementations
│
├── agents/                 # Specialized AI Agents (20+)
│   ├── base.py             # BaseAgent abstract class
│   ├── code_analyzer.py    # Codebase analysis
│   ├── test_planner.py     # Test plan generation
│   ├── ui_tester.py        # Browser test execution
│   ├── api_tester.py       # API testing
│   ├── self_healer.py      # Auto-fix broken tests
│   ├── quality_auditor.py  # Quality metrics
│   ├── root_cause_analyzer.py
│   ├── flaky_detector.py
│   ├── nlp_test_creator.py
│   ├── visual_ai.py
│   ├── accessibility_checker.py
│   ├── security_scanner.py
│   ├── performance_analyzer.py
│   └── ...
│
├── core/                   # Intelligence Modules
│   ├── model_router.py     # Multi-model AI routing
│   ├── cognitive_engine.py # Multi-step reasoning
│   ├── correlator.py       # Error correlation
│   ├── normalizer.py       # Error normalization
│   ├── coverage.py         # Coverage calculation
│   └── risk.py             # Risk scoring
│
├── services/               # External Service Clients
│   ├── supabase_client.py  # Database operations
│   ├── cache.py            # Cloudflare KV client
│   ├── vectorize.py        # Semantic search
│   └── ai_cost_tracker.py  # Cost management
│
├── browser/                # Browser Automation
│   ├── e2e_client.py       # High-level browser client
│   └── stagehand_client.py # Stagehand integration
│
├── security/               # Security & Compliance
│   ├── sanitizer.py        # Secret redaction
│   ├── audit.py            # Audit logging
│   └── classifier.py       # Data classification
│
├── integrations/           # Third-Party Integrations
│   ├── github_integration.py
│   ├── slack_integration.py
│   └── observability_hub.py
│
└── mcp/                    # MCP Servers
    ├── langgraph_mcp.py
    ├── playwright_mcp.py
    └── quality_mcp.py
```

### 4.3 Database Layer (Supabase)

**Location:** `/supabase/migrations/`

**Core Tables:**

| Table | Purpose |
|-------|---------|
| `organizations` | Multi-tenant organization management |
| `organization_members` | User-organization relationships |
| `projects` | Test projects per organization |
| `production_events` | Errors from Sentry/Datadog/etc |
| `ci_events` | CI/CD pipeline events |
| `coverage_reports` | Test coverage data |
| `generated_tests` | AI-generated test code |
| `risk_scores` | Component risk assessments |
| `healing_patterns` | Successful selector fixes |
| `ai_usage` | Per-request AI cost tracking |
| `ai_usage_daily` | Aggregated daily costs |
| `api_keys` | Programmatic API access |
| `webhook_logs` | Incoming webhook audit trail |

---

## 5. Data Flow Diagrams

### 5.1 Test Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TEST EXECUTION WORKFLOW                                │
└─────────────────────────────────────────────────────────────────────────────────┘

    User Request                    LangGraph Orchestrator
         │                                   │
         ▼                                   │
    ┌─────────┐                              │
    │  /run   │──────────────────────────────┤
    │  tests  │                              │
    └─────────┘                              │
                                             ▼
                                    ┌─────────────────┐
                                    │  Create Initial │
                                    │     State       │
                                    └────────┬────────┘
                                             │
                        ┌────────────────────┼────────────────────┐
                        │                    │                    │
                        ▼                    ▼                    ▼
               ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
               │  Code Analyzer  │  │  Test Planner   │  │    Executor     │
               │                 │  │                 │  │  (UI/API/DB)    │
               │  • Parse code   │  │  • Prioritize   │  │                 │
               │  • Find surfaces│  │  • Create plan  │  │  • Run tests    │
               │  • Extract deps │  │  • Risk-based   │  │  • Screenshots  │
               └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                        │                    │                    │
                        └────────────────────┼────────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │  Test Failed?   │
                                    └────────┬────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              │ YES                    NO   │
                              ▼                             ▼
                     ┌─────────────────┐           ┌─────────────────┐
                     │   Self-Healer   │           │    Reporter     │
                     │                 │           │                 │
                     │  • Analyze fail │           │  • HTML report  │
                     │  • Fix selector │           │  • GitHub PR    │
                     │  • Retry test   │           │  • Slack notify │
                     └────────┬────────┘           └────────┬────────┘
                              │                             │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │  Return Results │
                                    └─────────────────┘
```

### 5.2 Quality Intelligence Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        QUALITY INTELLIGENCE WORKFLOW                             │
└─────────────────────────────────────────────────────────────────────────────────┘

    Production Error                 Argus Brain                    Actions
         │                               │                             │
         ▼                               │                             │
    ┌─────────┐                          │                             │
    │ Sentry  │──────┐                   │                             │
    │ Webhook │      │                   │                             │
    └─────────┘      │                   │                             │
    ┌─────────┐      │                   │                             │
    │ Datadog │──────┤                   │                             │
    │ Webhook │      │                   │                             │
    └─────────┘      │                   │                             │
    ┌─────────┐      │                   │                             │
    │FullStory│──────┼───────────────────▶  ┌─────────────────┐        │
    │ Webhook │      │                      │   Normalize &   │        │
    └─────────┘      │                      │     Dedupe      │        │
    ┌─────────┐      │                      └────────┬────────┘        │
    │ Other   │──────┘                               │                 │
    │ Sources │                                      ▼                 │
    └─────────┘                             ┌─────────────────┐        │
                                            │  Store Event +  │        │
                                            │  Index Vector   │        │
                                            └────────┬────────┘        │
                                                     │                 │
                        ┌────────────────────────────┼────────────────────────┐
                        │                            │                        │
                        ▼                            ▼                        ▼
               ┌─────────────────┐          ┌─────────────────┐      ┌─────────────────┐
               │ Semantic Search │          │  Risk Scoring   │      │ Test Generation │
               │                 │          │                 │      │                 │
               │ Find similar    │          │ • Error freq    │      │ • Claude AI     │
               │ past errors     │          │ • Severity      │      │ • Playwright    │
               │ (Vectorize)     │          │ • User impact   │      │ • Auto PR       │
               └────────┬────────┘          └────────┬────────┘      └────────┬────────┘
                        │                            │                        │
                        └────────────────────────────┼────────────────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │   Dashboard     │
                                            │   + Alerts      │
                                            └─────────────────┘
```

### 5.3 Multi-Model AI Routing

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AI MODEL ROUTING STRATEGY                              │
└─────────────────────────────────────────────────────────────────────────────────┘

    Incoming Task
         │
         ▼
    ┌─────────────────┐
    │ Classify Task   │
    │ Type/Complexity │
    └────────┬────────┘
             │
    ┌────────┴────────────────────────────────────────────────────┐
    │                                                              │
    │  Task Complexity                                             │
    │                                                              │
    │  TRIVIAL ─────────▶  Llama 3.1 8B (Groq)      ~$0.0001/1K   │
    │  (element classify)                                          │
    │                                                              │
    │  SIMPLE ──────────▶  GPT-4o-mini             ~$0.0015/1K   │
    │  (action extract)    OR Claude Haiku                         │
    │                                                              │
    │  MODERATE ────────▶  Claude Haiku 4.5        ~$0.004/1K    │
    │  (test generation)                                           │
    │                                                              │
    │  COMPLEX ─────────▶  Claude Sonnet 4.5       ~$0.018/1K    │
    │  (root cause)                                                │
    │                                                              │
    │  EXPERT ──────────▶  Claude Opus 4.5         ~$0.05/1K     │
    │  (architecture)                                              │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  Budget Check   │───▶ Over budget? Queue for later
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Execute + Track│
    │  ai_usage table │
    └─────────────────┘


    COST SAVINGS: 60-80% vs using Claude Sonnet for everything
```

---

## 6. Database Schema

### 6.1 Core Tables

```sql
-- Organizations (Multi-Tenancy)
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    plan TEXT DEFAULT 'free',           -- free, pro, enterprise
    ai_budget_daily_usd NUMERIC DEFAULT 10.00,
    ai_budget_monthly_usd NUMERIC DEFAULT 100.00,
    ai_spend_today_usd NUMERIC DEFAULT 0,
    ai_spend_this_month_usd NUMERIC DEFAULT 0,
    features JSONB DEFAULT '{}',        -- max_projects, self_healing, etc.
    stripe_customer_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Production Events (from Sentry/Datadog/etc)
CREATE TABLE production_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    source TEXT NOT NULL,               -- sentry, datadog, fullstory, etc.
    external_id TEXT,
    event_type TEXT,                    -- error, exception, performance
    severity TEXT,                      -- fatal, error, warning, info
    title TEXT NOT NULL,
    message TEXT,
    stack_trace TEXT,
    fingerprint TEXT,                   -- for deduplication
    url TEXT,
    component TEXT,
    occurrence_count INTEGER DEFAULT 1,
    affected_users INTEGER DEFAULT 1,
    status TEXT DEFAULT 'new',
    ai_analysis JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Generated Tests (AI-created)
CREATE TABLE generated_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    production_event_id UUID REFERENCES production_events(id),
    name TEXT NOT NULL,
    test_type TEXT DEFAULT 'e2e',
    framework TEXT DEFAULT 'playwright',
    test_code TEXT NOT NULL,
    test_file_path TEXT,
    confidence_score NUMERIC,
    status TEXT DEFAULT 'pending',      -- pending, approved, rejected
    review_notes TEXT,
    github_pr_url TEXT,
    github_pr_number INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Risk Scores (Component Risk Assessment)
CREATE TABLE risk_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    entity_type TEXT NOT NULL,          -- page, component, flow, endpoint
    entity_identifier TEXT NOT NULL,
    overall_risk_score INTEGER,         -- 0-100
    factors JSONB,                       -- breakdown of risk factors
    error_count INTEGER,
    affected_users INTEGER,
    trend TEXT DEFAULT 'stable',        -- improving, stable, degrading
    calculated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(project_id, entity_type, entity_identifier)
);

-- AI Usage Tracking
CREATE TABLE ai_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id),
    request_id TEXT UNIQUE NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    task_type TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd NUMERIC NOT NULL,
    latency_ms INTEGER,
    cache_hit BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Healing Patterns (Self-Healing Knowledge Base)
CREATE TABLE healing_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fingerprint TEXT UNIQUE NOT NULL,
    original_selector TEXT NOT NULL,
    healed_selector TEXT NOT NULL,
    error_type TEXT NOT NULL,
    success_count INTEGER DEFAULT 1,
    failure_count INTEGER DEFAULT 0,
    confidence NUMERIC GENERATED ALWAYS AS (
        success_count::numeric / GREATEST(success_count + failure_count, 1)
    ) STORED,
    project_id UUID REFERENCES projects(id),
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### 6.2 Row-Level Security (RLS)

```sql
-- Users can only see their organization's data
ALTER TABLE production_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY "org_isolation" ON production_events
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organization_members om ON om.organization_id = p.organization_id
            WHERE om.user_id = auth.uid()
        )
    );
```

---

## 7. API Reference

### 7.1 Brain Service Endpoints (Railway)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/tests/run` | POST | Start test execution job |
| `/api/v1/jobs/{id}` | GET | Get job status/results |
| `/api/v1/tests/create` | POST | Create test from natural language |
| `/api/v1/visual/compare` | POST | Visual regression comparison |
| `/api/v1/discover` | POST | Auto-discover test scenarios |
| `/api/v1/webhooks/sentry` | POST | Sentry error webhook |
| `/api/v1/webhooks/datadog` | POST | Datadog event webhook |
| `/api/v1/webhooks/github-actions` | POST | GitHub Actions webhook |
| `/api/v1/quality/generate-test` | POST | Generate test from error |
| `/api/v1/quality/calculate-risk` | POST | Calculate risk scores |
| `/api/v1/quality/similar-errors` | GET | Semantic error search |
| `/api/v1/quality/backfill-index` | POST | Index historical errors |
| `/api/semantic-search` | POST | Find similar error patterns |

### 7.2 Edge Worker Endpoints (Cloudflare)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Worker health check |
| `/act` | POST | Execute browser action |
| `/extract` | POST | Extract data from page |
| `/observe` | POST | Discover page elements |
| `/agent` | POST | Multi-step autonomous workflow |
| `/test` | POST | Cross-browser test execution |

---

## 8. Agent System

### 8.1 Agent Hierarchy

```
BaseAgent (Abstract)
    │
    ├── CodeAnalyzerAgent       # Static code analysis
    ├── TestPlannerAgent        # Test prioritization
    │
    ├── ExecutionAgents
    │   ├── UITesterAgent       # Browser testing
    │   ├── APITesterAgent      # REST API testing
    │   └── DBTesterAgent       # Database validation
    │
    ├── IntelligenceAgents
    │   ├── SelfHealerAgent     # Auto-fix tests
    │   ├── RootCauseAnalyzer   # Failure analysis
    │   ├── QualityAuditorAgent # Quality metrics
    │   └── FlakyDetectorAgent  # Flakiness detection
    │
    ├── GenerationAgents
    │   ├── NLPTestCreator      # Natural language → test
    │   ├── SessionToTestAgent  # Recording → test
    │   └── AutoDiscoveryAgent  # Crawl → tests
    │
    └── SpecializedAgents
        ├── VisualAIAgent       # Visual regression
        ├── AccessibilityAgent  # WCAG compliance
        ├── SecurityScanner     # Vulnerability scan
        └── PerformanceAnalyzer # Performance metrics
```

### 8.2 LangGraph State Machine

```python
# State Definition
class TestingState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    codebase_path: str
    app_url: str
    testable_surfaces: list[dict]
    test_plan: list[dict]
    test_priorities: dict[str, int]
    current_test_index: int
    test_results: list[dict]
    failures: list[dict]
    healing_queue: list[dict]
    healing_attempts: int
    healed_tests: list[dict]
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    iteration: int
    security_summary: dict
    next_agent: str
    should_continue: bool

# Graph Flow
analyze → plan → execute → (heal if failed) → report
```

---

## 9. External Integrations

### 9.1 Observability Platforms

| Platform | Webhook Endpoint | Events Captured |
|----------|------------------|-----------------|
| **Sentry** | `/api/v1/webhooks/sentry` | Errors, exceptions, issues |
| **Datadog** | `/api/v1/webhooks/datadog` | Alerts, errors, metrics |
| **FullStory** | `/api/v1/webhooks/fullstory` | Rage clicks, dead clicks |
| **LogRocket** | `/api/v1/webhooks/logrocket` | Frontend errors |
| **NewRelic** | `/api/v1/webhooks/newrelic` | APM alerts |
| **Bugsnag** | `/api/v1/webhooks/bugsnag` | Error tracking |
| **Rollbar** | `/api/v1/webhooks/rollbar` | Error tracking |

### 9.2 CI/CD Integration

| Platform | Integration Type |
|----------|------------------|
| **GitHub Actions** | Webhook + PR comments + check runs |
| **GitLab CI** | Webhook support |
| **CircleCI** | Webhook support |

### 9.3 AI Providers

| Provider | Models | Use Case |
|----------|--------|----------|
| **Anthropic** | Claude Opus/Sonnet/Haiku 4.5 | Primary AI |
| **OpenAI** | GPT-4o, GPT-4o-mini | Fallback |
| **Groq** | Llama 3.1 8B/70B | Fast inference |
| **Together** | Llama, Mixtral | Cost optimization |
| **Google** | Gemini 1.5 Flash/Pro | Alternative |
| **Workers AI** | Llama, BGE embeddings | Edge AI |

---

## 10. Security Architecture

### 10.1 Security Features

| Feature | Implementation |
|---------|----------------|
| **Secret Redaction** | Automatic in `sanitizer.py` |
| **RLS** | Supabase row-level security |
| **API Keys** | Hashed storage, scoped permissions |
| **Audit Trail** | All actions logged with context |
| **JWT Auth** | Supabase Auth integration |
| **Webhook Signatures** | HMAC verification per platform |
| **Budget Controls** | Daily/monthly AI spend limits |

### 10.2 Data Classification

```python
# Patterns automatically redacted:
- API keys (sk-*, ghp_*, etc.)
- Passwords in URLs
- Bearer tokens
- AWS credentials
- Database connection strings
- Private keys (PEM format)
```

---

## 11. Cost Management

### 11.1 AI Cost Tracking

```
Per Request:
    ai_usage table → input_tokens, output_tokens, cost_usd

Daily Aggregation:
    ai_usage_daily → total cost per org per day

Budget Enforcement:
    organizations.ai_budget_daily_usd
    organizations.ai_spend_today_usd

    → Requests rejected when budget exceeded
```

### 11.2 Estimated Monthly Costs (at scale)

| Component | Provider | Cost/Month |
|-----------|----------|------------|
| Brain Service | Railway | $150-300 |
| Database | Supabase Scale | $150 |
| Edge Workers | Cloudflare Pro | $25 |
| KV + R2 + Vectorize | Cloudflare | $50 |
| AI (Claude) | Anthropic | $1,500-3,000 |
| **Total** | | **~$2,000-3,500** |

---

## 12. Deployment Architecture

### 12.1 Production Setup

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PRODUCTION DEPLOYMENT                               │
└─────────────────────────────────────────────────────────────────────────────────┘

    Internet Traffic
         │
         ▼
    ┌─────────────────┐
    │  Cloudflare     │ ◄─── Global CDN + DDoS protection
    │  Edge Network   │
    └────────┬────────┘
             │
    ┌────────┴────────────────────────────────────┐
    │                                              │
    ▼                                              ▼
┌─────────────────┐                    ┌─────────────────┐
│ argus-api       │                    │ argus-brain     │
│ (CF Worker)     │                    │ (Railway)       │
│                 │                    │                 │
│ Browser auto    │◄───────────────────│ LangGraph       │
│ at edge         │    /agent calls    │ Orchestration   │
│                 │                    │                 │
│ Global: ~50ms   │                    │ Region: US-East │
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         │                                      │
         └──────────────┬───────────────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │    Supabase     │
               │   (PostgreSQL)  │
               │                 │
               │ AWS us-east-1   │
               │ Pooler enabled  │
               └─────────────────┘
```

### 12.2 Environment Variables

```bash
# Brain Service (Railway)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...

# Cloudflare
CLOUDFLARE_API_TOKEN=xxx
CLOUDFLARE_ACCOUNT_ID=xxx
CLOUDFLARE_KV_NAMESPACE_ID=xxx
CLOUDFLARE_VECTORIZE_INDEX=argus-patterns

# Optional
GITHUB_TOKEN=ghp_...
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
TESTINGBOT_KEY=xxx
TESTINGBOT_SECRET=xxx
```

---

## Summary

Argus is a **production-ready autonomous testing platform** featuring:

- **3-Tier Architecture**: Edge (Cloudflare) → Brain (Railway) → Database (Supabase)
- **20+ AI Agents**: Specialized for different testing and analysis tasks
- **Multi-Model AI**: 60-80% cost savings through intelligent routing
- **Self-Healing**: Automatic test maintenance and selector repair
- **Quality Intelligence**: Production error → test coverage gap analysis
- **Enterprise Features**: Multi-tenancy, RBAC, audit trails, budget controls

**Current Status**: Production-ready with semantic search, auto-indexing, and multi-model support deployed.
