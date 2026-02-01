# Argus E2E Testing Platform - Enterprise Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                        ARGUS E2E TESTING PLATFORM                                                            │
│                                                     Enterprise AI Agent Architecture                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐  ┌────────────────────────────────────────────────────────────────────────┐  ┌──────────────────────────────────┐
│      OBSERVABILITY PLANE         │  │                        USER EXPERIENCE PLANE                          │  │        SECURITY PLANE            │
├──────────────────────────────────┤  ├────────────────────────────────────────────────────────────────────────┤  ├──────────────────────────────────┤
│                                  │  │                                                                        │  │                                  │
│  ┌────────────────────────────┐  │  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌───────────────┐  │  │  ┌────────────────────────────┐  │
│  │      Observability        │  │  │  │  End-user Experiences│  │    Agent Builder    │  │    DevEx      │  │  │  │  Identity & Tenant Security│  │
│  ├────────────────────────────┤  │  │  ├─────────────────────┤  ├─────────────────────┤  ├───────────────┤  │  │  ├────────────────────────────┤  │
│  │ • LangFuse Traces/Spans   │  │  │  │ • Dashboard (Next.js)│  │ • LangGraph Studio  │  │ • MCP Server  │  │  │  │ • Clerk SSO/OIDC           │  │
│  │ • Prometheus Metrics      │  │  │  │ • SSE Live Updates   │  │ • Agent Catalog     │  │ • Python SDK  │  │  │  │ • Multi-tenant RLS         │  │
│  │ • structlog Logging       │  │  │  │ • Time Travel Debug  │  │   (29 Agents)       │  │ • REST API    │  │  │  │ • 9-Role RBAC              │  │
│  │ • 36+ Prometheus Alerts   │  │  │  │ • HITL Approval UI   │  │ • Tool/MCP Registry │  │ • TypeScript  │  │  │  │ • Per-org Isolation        │  │
│  │ • Grafana Dashboards      │  │  │  │ • Test Reports       │  │   (41 Capabilities) │  │   SDK         │  │  │  │ • JWT + API Key Auth       │  │
│  │ • Debug & Replay          │  │  │  │ • Cost Analytics     │  │ • Workflow Builder  │  │ • OpenAPI 3.1 │  │  │  │ • Service Accounts         │  │
│  └────────────────────────────┘  │  │  └─────────────────────┘  └─────────────────────┘  └───────────────┘  │  │  └────────────────────────────┘  │
│                                  │  │                                                                        │  │                                  │
│  ┌────────────────────────────┐  │  └────────────────────────────────────────────────────────────────────────┘  │  ┌────────────────────────────┐  │
│  │         Audit              │  │                                                                              │  │    Policy Enforcement      │  │
│  ├────────────────────────────┤  │  ┌────────────────────────────────────────────────────────────────────────┐  │  ├────────────────────────────┤  │
│  │ • Immutable Audit Trail   │  │  │                         TRUST BOUNDARY                                  │  │  │ • 5-Layer Guardrail Stack  │  │
│  │   (365-day retention)     │  │  │  ┌──────────────────────────────────────────────────────────────────┐  │  │  │ • Input Sanitization       │  │
│  │ • Security Event Logs     │  │  │  │  JWT Auth │ API Key Auth │ Rate Limits │ Guardrails │ RBAC Check │  │  │  │ • Schema Validation        │  │
│  │ • AI Request/Response Log │  │  │  │           │              │ (30-2000/m) │ (5 layers) │ (30+ perms)│  │  │  │ • Docker Sandboxing        │  │
│  │ • Change Audit            │  │  │  └──────────────────────────────────────────────────────────────────┘  │  │  │ • Rate Limits (Tier-based) │  │
│  │ • Compliance Export API   │  │  │                                                                        │  │  │ • Action Constraints       │  │
│  └────────────────────────────┘  │  └────────────────────────────────────────────────────────────────────────┘  │  │ • HITL Gates               │  │
│                                  │                                                                              │  └────────────────────────────┘  │
│  ┌────────────────────────────┐  │  ┌────────────────────────────────────────────────────────────────────────┐  │                                  │
│  │         FinOps            │  │  │                          CONTROL PLANE                                  │  │  ┌────────────────────────────┐  │
│  ├────────────────────────────┤  │  ├────────────────────────────────────────────────────────────────────────┤  │  │  Secrets & Data Protection │  │
│  │ • Token Cost Attribution  │  │  │                                                                        │  │  ├────────────────────────────┤  │
│  │   (per-task, per-model)   │  │  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌───────────────┐  │  │  │ • Cloudflare Key Vault     │  │
│  │ • Per-org Budgets         │  │  │  │ Orchestration Engine│  │   Agent Lifecycle   │  │ Tool/MCP      │  │  │  │   (Zero-knowledge BYOK)    │  │
│  │ • Quota Enforcement       │  │  │  ├─────────────────────┤  ├─────────────────────┤  │   Lifecycle   │  │  │  │ • AES-256 Encryption       │  │
│  │ • Model Cost Optimization │  │  │  │ • LangGraph 1.0     │  │ • Agent Registry    │  ├───────────────┤  │  │  │ • DEK Key Versioning       │  │
│  │   (TaskType routing)      │  │  │  │ • Supervisor Pattern│  │   (41 capabilities) │  │ • Tool        │  │  │  │ • KMS Integration          │  │
│  │ • AI Spend Forecasting    │  │  │  │ • Complexity Router │  │ • A2A Protocol      │  │   Discovery   │  │  │  │ • Data Classification      │  │
│  │ • Infra Optimization      │  │  │  │ • Planning Middle-  │  │   (Kafka mesh)      │  │   Agent       │  │  │  │ • PII/PHI Detection        │  │
│  └────────────────────────────┘  │  │  │   ware (TodoList)   │  │ • MARP Consensus    │  │ • Tool        │  │  │  │ • Retention Policies       │  │
│                                  │  │  │ • Task Decomposer   │  │ • Health Monitoring │  │   Registry    │  │  │  └────────────────────────────┘  │
│                                  │  │  └─────────────────────┘  └─────────────────────┘  │ • DFSDT       │  │  │                                  │
│                                  │  │                                                    │   Selection   │  │  │                                  │
│                                  │  │  ┌─────────────────────┐  ┌─────────────────────┐  └───────────────┘  │  │                                  │
│                                  │  │  │ Evals & Scorecards  │  │   Semantic Routing  │                     │  │                                  │
│                                  │  │  ├─────────────────────┤  ├─────────────────────┤                     │  │                                  │
│                                  │  │  │ • Agent-as-Judge    │  │ • Model Router      │                     │  │                                  │
│                                  │  │  │ • MetaJudge (Debate)│  │   (300+ models)     │                     │  │                                  │
│                                  │  │  │ • Test Validation   │  │ • TaskType-based    │                     │  │                                  │
│                                  │  │  │ • Healing Validation│  │ • Cost Optimization │                     │  │                                  │
│                                  │  │  │ • DSPy Optimization │  │ • Auto-failover     │                     │  │                                  │
│                                  │  │  └─────────────────────┘  └─────────────────────┘                     │  │                                  │
│                                  │  │                                                                        │  │                                  │
│                                  │  └────────────────────────────────────────────────────────────────────────┘  │                                  │
│                                  │                                                                              │                                  │
│                                  │  ┌────────────────────────────────────────────────────────────────────────┐  │                                  │
│                                  │  │                          RUNTIME PLANE                                  │  │                                  │
│                                  │  ├────────────────────────────────────────────────────────────────────────┤  │                                  │
│                                  │  │                                                                        │  │                                  │
│                                  │  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌───────────────┐  │  │                                  │
│                                  │  │  │  Workflow Execution │  │ Execution Sandboxes │  │   Gateways    │  │  │                                  │
│                                  │  │  ├─────────────────────┤  ├─────────────────────┤  ├───────────────┤  │  │                                  │
│                                  │  │  │ • Sync/Async Flows  │  │ • Docker Containers │  │ • Event       │  │  │                                  │
│                                  │  │  │ • Parallel Executor │  │ • Browser Pool      │  │   Gateway     │  │  │                                  │
│                                  │  │  │ • Batch Processing  │  │   (Selenium Grid)   │  │ • MCP Server  │  │  │                                  │
│                                  │  │  │ • Cron Scheduling   │  │ • Subgraph Isolation│  │ • Model       │  │  │                                  │
│                                  │  │  │ • Retries/Timeouts  │  │ • Resource Quotas   │  │   Gateway     │  │  │                                  │
│                                  │  │  │ • HITL Breakpoints  │  │ • Concurrency Limits│  │   (OpenRouter)│  │  │                                  │
│                                  │  │  │ • Compensation      │  │ • TTL Enforcement   │  │ • Allowlists  │  │  │                                  │
│                                  │  │  └─────────────────────┘  └─────────────────────┘  └───────────────┘  │  │                                  │
│                                  │  │                                                                        │  │                                  │
│                                  │  └────────────────────────────────────────────────────────────────────────┘  │                                  │
│                                  │                                                                              │                                  │
│                                  │  ┌────────────────────────────────────────────────────────────────────────┐  │                                  │
│                                  │  │                     CONTEXT & MEMORY PLANE                             │  │                                  │
│                                  │  ├────────────────────────────────────────────────────────────────────────┤  │                                  │
│                                  │  │                                                                        │  │                                  │
│                                  │  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌───────────────┐  │  │                                  │
│                                  │  │  │   Context Service   │  │      Run State      │  │ Index & Search│  │  │                                  │
│                                  │  │  ├─────────────────────┤  ├─────────────────────┤  ├───────────────┤  │  │                                  │
│                                  │  │  │ • Valkey Cache      │  │ • PostgreSQL        │  │ • Cognee      │  │  │                                  │
│                                  │  │  │   (3-tier: AI GW →  │  │   Checkpoints       │  │   Vector      │  │  │                                  │
│                                  │  │  │   Valkey → Cognee)  │  │ • Event Journaling  │  │   Search      │  │  │                                  │
│                                  │  │  │ • Token Budgeting   │  │   (Kafka 12 topics) │  │ • pgvector    │  │  │                                  │
│                                  │  │  │ • Episodic Memory   │  │ • State Retention   │  │ • Semantic    │  │  │                                  │
│                                  │  │  │ • Procedural Memory │  │ • Point-in-time     │  │   Ranking     │  │  │                                  │
│                                  │  │  │ • Zettelkasten Links│  │   Replay            │  │ • Precomputed │  │  │                                  │
│                                  │  │  └─────────────────────┘  └─────────────────────┘  │   Intelligence│  │  │                                  │
│                                  │  │                                                    └───────────────┘  │  │                                  │
│                                  │  │                                                                        │  │                                  │
│                                  │  └────────────────────────────────────────────────────────────────────────┘  │                                  │
│                                  │                                                                              │                                  │
└──────────────────────────────────┘                                                                              └──────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                           DATA PLANE                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                                              │
│  ┌────────────────────────────────────┐  ┌────────────────────────────────────┐  ┌────────────────────────────────────┐                     │
│  │      Sources & Connectors         │  │      Storage & Processing          │  │      Governance & Metadata         │                     │
│  ├────────────────────────────────────┤  ├────────────────────────────────────┤  ├────────────────────────────────────┤                     │
│  │ • GitHub/GitLab Webhooks          │  │ • Supabase PostgreSQL              │  │ • Integration Catalog              │                     │
│  │ • Redpanda/Kafka (12 topics)      │  │ • Cloudflare R2 (Screenshots)      │  │ • PII/PHI Classification           │                     │
│  │ • 60+ Integration Plugins         │  │ • Apache Flink Streaming           │  │ • AES-256 Encryption               │                     │
│  │   (Jira, Slack, Sentry, etc.)     │  │ • Cognee ECL Pipeline              │  │ • Test Impact Lineage Graph        │                     │
│  │ • VCS Webhook Events              │  │ • Semantic Chunking                │  │ • Row-Level Security (RLS)         │                     │
│  │ • CI/CD Pipeline Events           │  │ • Cohere Embeddings (1024-dim)     │  │ • Column-Level Security            │                     │
│  │ • n8n Workflow Triggers           │  │ • Neo4j Aura (Knowledge Graph)     │  │ • 90+ Supabase Migrations          │                     │
│  │ • MCP Protocol Connections        │  │ • FalkorDB (Graph Queries)         │  │ • Schema Evolution                 │                     │
│  └────────────────────────────────────┘  └────────────────────────────────────┘  └────────────────────────────────────┘                     │
│                                                                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                    29 SPECIALIZED AI AGENTS                                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │    Code     │ │    Test     │ │     UI      │ │    API      │ │    Self     │ │    SRE      │ │  Corrective │ │    Tool     │            │
│  │  Analyzer   │ │   Planner   │ │   Tester    │ │   Tester    │ │   Healer    │ │   Agent     │ │     RAG     │ │  Discovery  │            │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   Visual    │ │    NLP      │ │    Auto     │ │   Flaky     │ │ Performance │ │  Security   │ │Accessibility│ │   Agent     │            │
│  │     AI      │ │   Creator   │ │  Discovery  │ │  Detector   │ │  Analyzer   │ │  Scanner    │ │  Checker    │ │   Judge     │            │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   Quality   │ │  Root Cause │ │     MR      │ │   Router    │ │    Test     │ │  Reporter   │ │    DB       │ │  Session    │            │
│  │   Auditor   │ │  Analyzer   │ │  Analyzer   │ │   Agent     │ │   Impact    │ │   Agent     │ │   Tester    │ │  to Test    │            │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │
│                                                                                                                                              │
│  Advanced Patterns: Reflexion (self-improvement) │ MetaJudge (debate) │ MARP (consensus) │ DSPy Optimization │ Zettelkasten Memory         │
│                                                                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                      INFRASTRUCTURE                                                                          │
├───────────────────────────────────┬───────────────────────────────────┬───────────────────────────────────┬──────────────────────────────────┤
│         Railway (Backend)         │       Vercel (Dashboard)          │       Vultr VKE (Data Layer)      │      Cloudflare (Edge)           │
├───────────────────────────────────┼───────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────┤
│ • FastAPI Backend                 │ • Next.js 14 App Router           │ • Redpanda (Kafka)                │ • R2 Object Storage              │
│ • LangGraph Orchestrator          │ • React Components                │ • Cognee Worker                   │ • Key Vault (Secrets)            │
│ • 29 AI Agents                    │ • TailwindCSS                     │ • FalkorDB (Graph)                │ • AI Gateway (Cache)             │
│ • PostgreSQL (Checkpoints)        │ • SSE Streaming                   │ • Valkey (Cache)                  │ • Tunnel (Monitoring)            │
│                                   │                                   │ • Selenium Grid (3 nodes)         │ • Workers (Webhooks)             │
│                                   │                                   │ • Prometheus + Grafana            │                                  │
└───────────────────────────────────┴───────────────────────────────────┴───────────────────────────────────┴──────────────────────────────────┘
```

## Architecture Compliance: 79%

| Plane | Status | Key Components |
|-------|--------|----------------|
| Observability | ✅ | LangFuse, Prometheus, 36 Alerts, Grafana |
| User Experience | ✅ | Dashboard, Agent Builder, MCP Server |
| Security | ✅ | Clerk SSO, RBAC (9 roles), KMS, RLS |
| Trust Boundary | ✅ | JWT, Rate Limits, 5-layer Guardrails |
| Control Plane | ✅ | LangGraph, A2A Protocol, 41 Capabilities |
| Runtime | ✅ | Docker Sandbox, Browser Pool, Gateways |
| Context & Memory | ✅ | Cognee, Valkey, Episodic/Procedural Memory |
| Data Plane | ✅ | Kafka (12 topics), R2, Flink, pgvector |

## Remaining Gaps (3 items)
- Agent Promotion Pipeline (Dev→Staging→Prod)
- Explicit Regression Gates
- Formalized DLP Policy Engine
