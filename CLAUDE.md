# Autonomous E2E Full-Stack Testing Agent

## Project Overview

This project builds a **fully autonomous end-to-end testing agent** that can:
1. Understand any codebase and generate comprehensive test plans
2. Execute UI tests using Claude's Computer Use API
3. Execute API tests with schema validation
4. Validate database state and data integrity
5. Self-heal broken tests by analyzing failures and fixing selectors/assertions
6. Generate human-readable reports and integrate with CI/CD pipelines

## ⚠️ Monorepo Structure

**This is a monorepo containing multiple independent packages/services.** When working on this codebase, always be aware of which package you're modifying.

### Packages Overview

| Package | Type | Location | Description |
|---------|------|----------|-------------|
| **Backend API** | Python (FastAPI) | `src/` | Main orchestrator, LangGraph agents, API endpoints |
| **Dashboard** | Next.js (TypeScript) | `dashboard/` | Web UI for managing test runs, viewing results |
| **MCP Server** | TypeScript | `argus-mcp-server/` | Model Context Protocol server (extracted to separate repo) |
| **Cloudflare Worker** | TypeScript | `cloudflare-worker/` | Edge functions for webhook handling |
| **Crawlee Service** | TypeScript | `crawlee-service/` | Web scraping/crawling service |
| **Docs Site** | MkDocs | `docs-site/` + `docs/` | Documentation website |
| **Browser Extension** | JavaScript | `extension/` | Browser extension for capturing tests |
| **Status Page** | Static | `status-page/` | Service status page |
| **Supabase** | SQL | `supabase/` | Database migrations and configuration |

### Working in the Monorepo

1. **Identify the package first** - Before making changes, confirm which package(s) are affected
2. **Package-specific commands**:
   - Python backend: `cd src && python -m pytest` or use root `pyproject.toml`
   - Dashboard: `cd dashboard && npm run dev`
   - MCP Server: `cd argus-mcp-server && npm run build`
   - Cloudflare Worker: `cd cloudflare-worker && npm run dev`
3. **Shared configuration**:
   - Root `.env` contains shared environment variables
   - Root `package.json` contains workspace scripts (if applicable)
   - Supabase migrations apply to all services using the database
4. **Cross-package changes** - When changes span multiple packages, update each package's relevant files and ensure compatibility

### Git Considerations

- All packages share the same git history in this repo
- Some packages (like `argus-mcp-server`) have been extracted to separate repositories
- Commits should clearly indicate which package(s) are affected in the commit message

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR (LangGraph)                     │
│                    Manages state, routes to agents                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│ CODE ANALYZER │           │ TEST EXECUTOR │           │  SELF-HEALER  │
│    AGENT      │           │    AGENTS     │           │     AGENT     │
├───────────────┤           ├───────────────┤           ├───────────────┤
│ • Parse code  │           │ • UI Tester   │           │ • Analyze fail│
│ • Find tests  │           │ • API Tester  │           │ • Fix selector│
│ • Gen specs   │           │ • DB Tester   │           │ • Update tests│
└───────────────┘           └───────────────┘           └───────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
                         ┌───────────────────┐
                         │  REPORTER AGENT   │
                         │ • Generate report │
                         │ • Create tickets  │
                         │ • Notify team     │
                         └───────────────────┘
```

## LangGraph 1.0 Features

This project leverages the full power of LangGraph 1.0 for production-ready orchestration:

### Durable Execution (PostgresSaver)
- All graph state persists to PostgreSQL via Supabase
- Test runs survive server restarts
- Automatic checkpoint creation at each node
- File: `src/orchestrator/checkpointer.py`

### Long-term Memory (PostgresStore + pgvector)
- Semantic search on failure patterns using pgvector
- Cross-session learning for self-healing
- Memory namespaces for different contexts
- Files:
  - `src/orchestrator/memory_store.py`
  - `supabase/migrations/20260109000001_langgraph_memory_store.sql`

### Streaming (SSE)
- Real-time execution updates via Server-Sent Events
- Multiple stream modes: values, updates, messages, custom
- Live log streaming to dashboard
- File: `src/api/streaming.py`

### Human-in-the-Loop
- Breakpoints before/after critical nodes
- Approval workflow for destructive operations
- Resume from interruption with modified state
- File: `src/api/approvals.py`

### Time Travel Debugging
- Browse historical state checkpoints
- Replay from any checkpoint
- Fork test runs for A/B testing
- Compare divergent executions
- File: `src/api/time_travel.py`

### Multi-agent Supervisor
- Supervisor pattern for dynamic agent routing
- Specialized agents: CodeAnalyzer, UITester, APITester, SelfHealer, Reporter
- Advanced agents: SREAgent, CorrectiveRAG, ToolDiscovery
- Automatic task delegation based on current state
- File: `src/orchestrator/supervisor.py`

### Advanced 2026 Agentic AI Patterns

Argus implements state-of-the-art agentic AI patterns from 2026 research:

| Pattern | Agent/Module | Description |
|---------|--------------|-------------|
| **Reflexion** | `BaseAgent._call_ai_with_reflexion()` | Execute → Critique → Refine loop for self-improvement |
| **Agent-as-Judge** | `AgentAsJudge`, `MetaJudge` | Multi-agent evaluation with debate for quality assurance |
| **Corrective RAG** | `CorrectiveRAGAgent` | Self-correcting retrieval with relevance assessment |
| **Adaptive RAG** | `AdaptiveRAGRouter` | Dynamic strategy selection based on query complexity |
| **ToolLLM/CREATOR** | `ToolDiscoveryAgent`, `ToolCreatorAgent` | Dynamic tool discovery and creation |
| **SRE Agent** | `SREAgent` | Unified incident correlation, runbook execution, MTTR |
| **Episodic Memory** | `MemoryManager` | Session-based memory with importance weighting |
| **Procedural Memory** | `MemoryManager` | Learned procedures with Zettelkasten linking |
| **Deep Agent** | `PlanningMiddleware` | TodoList-based planning and task decomposition |
| **DSPy Optimization** | `src/optimization/` | Automatic prompt optimization (COPRO, MIPRO, SIMBA, GEPA) |

Key files:
- `src/agents/agent_judge.py` - Agent-as-Judge + MetaJudge
- `src/agents/corrective_rag_agent.py` - CRAG + Adaptive RAG
- `src/agents/sre_agent.py` - Unified SRE operations
- `src/agents/tool_discovery_agent.py` - ToolLLM + CREATOR patterns
- `src/orchestrator/memory_manager.py` - Episodic + Procedural memory
- `src/orchestrator/planning_middleware.py` - Deep Agent TodoList
- `src/optimization/prompt_optimizer.py` - DSPy-style prompt optimization

### Chat-through-Orchestrator
- All chat routed through LangGraph
- Full tool execution with checkpointing
- Conversation memory with semantic search
- File: `src/orchestrator/chat_graph.py`

## 🤖 Agent-to-Agent (A2A) Architecture

Argus uses a **mesh + supervisor hybrid** pattern where agents can communicate peer-to-peer while still being orchestrated by a supervisor for high-level coordination.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AGENT MESH LAYER                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      Agent Registry                                     │ │
│  │  - 41 capabilities registered (code_analysis, self_healing, etc.)      │ │
│  │  - Health monitoring with 60-second heartbeat timeout                  │ │
│  │  - Service discovery for capability-based routing                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│      ┌─────────────────────────────┼─────────────────────────────┐          │
│      │                             │                             │          │
│      ▼                             ▼                             ▼          │
│  ┌──────────┐              ┌──────────┐              ┌──────────┐          │
│  │ Agent A  │◀────────────▶│ Agent B  │◀────────────▶│ Agent C  │          │
│  │          │  A2A Protocol│          │  A2A Protocol│          │          │
│  └────┬─────┘              └────┬─────┘              └────┬─────┘          │
│       │                         │                         │                 │
│       └─────────────────────────┴─────────────────────────┘                 │
│                                 │                                            │
│  ┌──────────────────────────────▼───────────────────────────────────────┐   │
│  │                         MESSAGE BUS (Kafka)                           │   │
│  │  argus.agent.request   │ argus.agent.response │ argus.agent.broadcast │   │
│  │  argus.agent.heartbeat │ (6 partitions each)                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                          ┌─────────▼───────────┐
                          │     SUPERVISOR      │
                          │  (High-level only)  │
                          │  - Task decomposition│
                          │  - Final aggregation │
                          └─────────────────────┘
```

### A2A Protocol (src/orchestrator/a2a_protocol.py)

Agents communicate via Kafka topics using a request-response pattern:

```python
from src.orchestrator.a2a_protocol import A2AProtocol

# Initialize protocol
a2a = A2AProtocol()

# Query another agent
response = await a2a.request(
    to_agent="CodeAnalyzer",
    capability="git_blame",
    payload={"file": "src/auth.py", "line": 42},
    timeout=30.0
)

# Broadcast to all agents
await a2a.broadcast(
    topic="failure_detected",
    message={"test_id": "123", "error": "Selector changed"}
)

# Subscribe to broadcasts
await a2a.subscribe(
    topic="failure_detected",
    handler=self.handle_failure
)
```

### Agent Registry (src/orchestrator/agent_registry.py)

Central registry for agent capabilities and health monitoring:

```python
from src.orchestrator.agent_registry import AgentRegistry, AgentCapability

# 41 capability constants available
AgentCapability.CODE_ANALYSIS      # Analyze source code
AgentCapability.SELECTOR_ANALYSIS  # Analyze UI selectors
AgentCapability.GIT_BLAME          # Git history analysis
AgentCapability.SELF_HEALING       # Fix broken tests
AgentCapability.TEST_GENERATION    # Generate test cases
AgentCapability.VISUAL_COMPARISON  # Compare screenshots
# ... and 35 more

# Discover agents by capability
registry = AgentRegistry.get_instance()
agents = registry.discover(AgentCapability.SELF_HEALING)
# Returns: [AgentInfo(type=SelfHealerAgent, healthy=True, ...)]

# Query specific agent
response = await registry.query(
    agent_type=CodeAnalyzerAgent,
    request=AgentRequest(capability="git_blame", payload={...})
)
```

### BaseAgent A2A Integration (src/agents/base.py)

All agents inherit A2A capabilities:

```python
class MyAgent(BaseAgent):
    # Declare agent capabilities
    CAPABILITIES = [
        AgentCapability.CODE_ANALYSIS,
        AgentCapability.TEST_GENERATION,
    ]

    async def execute(self, state: TestingState) -> TestingState:
        # Query another agent via A2A
        blame_info = await self.query_agent(
            agent_type=CodeAnalyzerAgent,
            capability="git_blame",
            payload={"file": "src/auth.py", "line": 42}
        )

        # Use the response
        if blame_info.success:
            author = blame_info.payload["author"]
            commit = blame_info.payload["commit"]
```

### Dynamic Workflow Composition (src/orchestrator/workflow_composer.py)

Create workflows at runtime based on task requirements:

```python
from src.orchestrator.workflow_composer import WorkflowComposer
from src.orchestrator.task_decomposer import TaskDecomposer

# Decompose complex task
decomposer = TaskDecomposer()
subtasks = await decomposer.decompose(
    "Analyze auth.py failure and suggest fix"
)
# Returns: [
#   TaskDefinition(type="code_analysis", target="auth.py"),
#   TaskDefinition(type="git_blame", target="auth.py:42"),
#   TaskDefinition(type="self_healing", depends_on=[0, 1]),
# ]

# Compose workflow from subtasks
composer = WorkflowComposer()
workflow = composer.compose(
    tasks=subtasks,
    constraints=WorkflowConstraints(max_parallel=3, timeout=120)
)

# Execute compiled workflow
result = await workflow.run()
```

### Multi-Agent Reasoning Protocol (MARP)

For complex decisions requiring multiple agent perspectives:

```python
from src.orchestrator.marp import MARP
from src.orchestrator.consensus import ConsensusStrategy

marp = MARP()

# Agents propose solutions
await marp.propose(
    topic="root_cause_analysis",
    agent="SelfHealer",
    solution={"cause": "selector_changed", "fix": "update_xpath"},
    confidence=0.85,
    reasoning="DOM structure changed after refactor"
)

await marp.propose(
    topic="root_cause_analysis",
    agent="CodeAnalyzer",
    solution={"cause": "api_changed", "fix": "update_endpoint"},
    confidence=0.72,
    reasoning="Backend endpoint renamed"
)

# Resolve with consensus
resolution = await marp.resolve(
    topic="root_cause_analysis",
    strategy=ConsensusStrategy.CONFIDENCE_WEIGHTED
)
# Returns highest confidence solution with supporting evidence
```

### Consensus Strategies (src/orchestrator/consensus.py)

Six strategies for multi-agent voting:

| Strategy | Use Case |
|----------|----------|
| `MajorityVoting` | Simple decisions, equal agent expertise |
| `ConfidenceWeighted` | Weight votes by confidence scores |
| `ExpertiseWeighted` | Weight by agent's domain expertise |
| `SuperMajority` | Critical decisions (67%+ agreement) |
| `BordaCount` | Ranked preferences with multiple options |
| `QuadraticVoting` | Prevent single agent domination |

### Incremental Codebase Indexing (src/indexer/)

Delta-aware analysis for large codebases:

```python
from src.indexer.incremental_indexer import IncrementalIndexer

indexer = IncrementalIndexer()

# Analyze only changed files (10x faster)
changes = await indexer.analyze_changes(
    repo_url="https://github.com/org/repo",
    from_commit="abc123",
    to_commit="def456"
)

# Smart indexing (auto-chooses full vs incremental)
update = await indexer.smart_index(
    project_id="proj_123",
    force_full=False  # Only if needed
)
```

### Proactive CI/CD Monitoring (src/services/cicd_monitor.py)

GitLab Duo-style automatic analysis:

```python
from src.services.cicd_monitor import CICDMonitor

monitor = CICDMonitor()

# Poll for new MRs/PRs
github_prs = await monitor.poll_github(project_id, repo="org/repo")
gitlab_mrs = await monitor.poll_gitlab(project_id, repo="org/repo")

# Auto-trigger analysis
for pr in github_prs:
    if pr.is_new or pr.has_new_commits:
        analysis = await monitor.trigger_analysis(pr)
        # Posts test suggestions as PR comment
```

### A2A Kafka Topics

```
argus.agent.request     → Agent-to-agent requests (6 partitions)
argus.agent.response    → Responses to requests (6 partitions)
argus.agent.broadcast   → Pub/sub broadcasts (6 partitions)
argus.agent.heartbeat   → Health monitoring (6 partitions)
```

### Key A2A Files

| File | Purpose |
|------|---------|
| `src/orchestrator/agent_registry.py` | Agent capability registry + discovery |
| `src/orchestrator/a2a_protocol.py` | Kafka-based peer-to-peer messaging |
| `src/orchestrator/workflow_composer.py` | Runtime workflow composition |
| `src/orchestrator/task_decomposer.py` | Break complex tasks into subtasks |
| `src/orchestrator/parallel_executor.py` | Concurrent agent execution |
| `src/orchestrator/marp.py` | Multi-agent reasoning protocol |
| `src/orchestrator/consensus.py` | 6 voting/consensus strategies |
| `src/orchestrator/resolver.py` | Conflict resolution patterns |
| `src/indexer/incremental_indexer.py` | Delta-aware codebase analysis |
| `src/indexer/change_manifest.py` | Track file changes |
| `src/indexer/git_integration.py` | Git diff analysis |
| `src/services/cicd_monitor.py` | GitHub + GitLab polling |
| `src/agents/mr_analyzer.py` | MR/PR analysis agent |
| `src/integrations/comment_poster.py` | Auto-post to PRs/MRs |

### Circuit Breaker Pattern

A2A protocol includes fault tolerance:

```python
# Built into A2AProtocol
class CircuitBreaker:
    failure_threshold: int = 5      # Open after 5 failures
    reset_timeout: float = 60.0     # Try again after 60s
    half_open_max_calls: int = 3    # Test calls in half-open

# Automatic protection
response = await a2a.request(...)  # Fails fast if circuit open
```

## Unified Instant Intelligence Layer (UIIL)

**Goal**: Transform LLM-dependent operations (2-5s) to instant responses (<100ms) via 3-tier resolution.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED INSTANT INTELLIGENCE LAYER                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Query → [AI Gateway Cache] → [Valkey Cache] → [Cognee] → [LLM Fallback]   │
│            (Cloudflare)        (exact match)   (vector)    (last resort)    │
│              ~5ms                 ~2ms          ~30ms         ~2000ms        │
│                                                                              │
│  Expected Hit Rates:                                                         │
│  - AI Gateway cache: 30-40% (semantic similarity)                           │
│  - Valkey cache: 20-30% (exact match)                                       │
│  - Cognee vector: 25-35% (confidence > 0.7)                                 │
│  - LLM fallback: 5-15% (novel queries only)                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Intelligence Cache** | `src/intelligence/cache.py` | Valkey wrapper for Cognee search caching |
| **Query Router** | `src/intelligence/query_router.py` | Intent detection + tier routing |
| **Precomputed Results** | `src/intelligence/precomputed.py` | Read from intelligence_precomputed table |
| **Cognee Worker** | `data-layer/cognee-worker/src/worker.py` | ECL pipeline + integration handlers |
| **AI Gateway** | `src/services/cloudflare_storage.py` | LLM caching via Cloudflare |

### Query Intents (7 Types)

| Intent | Resolution Strategy | Target Latency |
|--------|---------------------|----------------|
| `SIMILAR_ERRORS` | Cache → Vector → Graph | <80ms |
| `TEST_IMPACT` | Cache → Precomputed Matrix | <50ms |
| `ROOT_CAUSE` | Cache → Vector → LLM Fallback | <100ms |
| `DOCUMENTATION` | Cache → Vector Search | <50ms |
| `CODE_CONTEXT` | Cache → Vector → Graph | <80ms |
| `COVERAGE_GAPS` | Precomputed (nightly) | <30ms |
| `SECURITY_IMPACT` | Cache → Graph → LLM | <100ms |

### Integration Indexing (via Cognee Worker)

| Integration | Kafka Topic | Status |
|-------------|-------------|--------|
| Codebase | `argus.codebase.ingested` | ✅ Working |
| Test Events | `argus.test.*` | ✅ Working |
| Healing | `argus.healing.requested` | ✅ Working |
| Sentry | `argus.integration.sentry` | 🚧 In Progress |
| Jira | `argus.integration.jira` | 📋 Planned |
| GitHub PRs | `argus.integration.github.pr` | 📋 Planned |
| Confluence | `argus.integration.confluence` | 📋 Planned |

### Pre-Computation Jobs

| Job | Schedule | Output |
|-----|----------|--------|
| Test Impact Matrix | Daily 2 AM UTC | `intelligence_precomputed` table |
| Failure Clustering | Real-time (Flink) | `flink_failure_patterns` table |
| Flaky Test Ranking | Daily 3 AM UTC | `intelligence_precomputed` table |
| Coverage Gaps | Weekly Sunday 4 AM | `intelligence_precomputed` table |

### Industry Patterns Used

- **Netflix Pattern**: Pre-compute heavy signals in batch, contextualize in real-time <100ms
- **Anthropic Prompt Caching**: 85% latency reduction via cache_control API
- **GPTCache Semantic Caching**: 61-68% hit rate with similarity matching
- **Uber/Netflix Flink**: Stateless streaming with idempotent Supabase writes

## Tech Stack

### AI & ML Layer
- **AI Router**: OpenRouter (300+ models, unified API, auto-failover)
- **Primary Models**: Claude Sonnet 4.5 (default), Claude Haiku 4.5 (fast), Claude Opus 4.5 (complex)
- **Providers**: Anthropic, OpenAI, Google Vertex, DeepSeek, Mistral, Cerebras, Fireworks, Cohere, xAI
- **Knowledge Graph**: Cognee (ECL pipeline) + Neo4j Aura (graph DB) + pgvector (embeddings)
- **Embeddings**: Cohere embed-multilingual-v3.0 (1024-dim)

### Event & Streaming Layer
- **Message Broker**: Redpanda/Kafka (6 partitions, ZSTD compression)
- **Stream Processing**: Apache Flink (exactly-once semantics)
- **Real-time**: Server-Sent Events (SSE) for dashboard

### Orchestration & Agents
- **Orchestration**: LangGraph (Python) with PostgresSaver
- **UI Testing**: Claude Computer Use API + Playwright (hybrid)
- **API Testing**: httpx + pydantic for validation

### Infrastructure
- **Database**: Supabase (PostgreSQL + Auth + Realtime + pgvector)
- **Storage**: Cloudflare R2 (screenshots, artifacts)
- **Secrets**: Cloudflare Key Vault (AES-256 encryption)
- **Cache**: Valkey/Redis
- **Kubernetes**: Vultr VKE (data layer)

### Integrations
- **MCP Servers**: filesystem, github, postgres (configurable)
- **Workflow Trigger**: n8n webhooks (optional)
- **Monitoring**: Prometheus + Grafana + Alertmanager

## Directory Structure

```
e2e-testing-agent/                    # MONOREPO ROOT
├── CLAUDE.md                         # This file - project instructions
├── README.md                         # User documentation
├── pyproject.toml                    # Python dependencies (backend)
├── package.json                      # Root package.json (workspace scripts)
├── .env.example                      # Environment variables template
│
├── src/                              # 🐍 PYTHON BACKEND (FastAPI + LangGraph)
│   ├── __init__.py
│   ├── main.py                       # Entry point
│   ├── config.py                     # Configuration management
│   ├── api/                          # FastAPI routes
│   │   ├── cicd.py                   # CI/CD, builds, pipelines, test impact
│   │   └── ...
│   ├── orchestrator/                 # LangGraph state machine & nodes
│   │   ├── supervisor.py             # Multi-agent supervisor
│   │   ├── agent_registry.py         # 🆕 A2A capability registry
│   │   ├── a2a_protocol.py           # 🆕 Peer-to-peer messaging
│   │   ├── workflow_composer.py      # 🆕 Dynamic workflow creation
│   │   ├── task_decomposer.py        # 🆕 Task breakdown
│   │   ├── parallel_executor.py      # 🆕 Concurrent execution
│   │   ├── marp.py                   # 🆕 Multi-agent reasoning
│   │   ├── consensus.py              # 🆕 Voting strategies
│   │   └── resolver.py               # 🆕 Conflict resolution
│   ├── agents/                       # AI agents (24+ agents)
│   │   ├── base.py                   # BaseAgent with A2A support
│   │   ├── code_analyzer.py          # Code analysis
│   │   ├── self_healer.py            # Self-healing
│   │   ├── mr_analyzer.py            # 🆕 MR/PR analysis
│   │   └── ...                       # 20+ more agents
│   ├── indexer/                      # 🆕 Codebase indexing
│   │   ├── incremental_indexer.py    # Delta-aware analysis
│   │   ├── change_manifest.py        # Change tracking
│   │   └── git_integration.py        # Git diff analysis
│   ├── services/                     # Background services
│   │   ├── cicd_monitor.py           # 🆕 GitHub/GitLab polling
│   │   └── ...
│   ├── integrations/                 # External integrations
│   │   ├── comment_poster.py         # 🆕 PR/MR comments
│   │   └── ...
│   ├── events/                       # Kafka event system
│   │   ├── topics.py                 # Topic definitions (12 topics)
│   │   ├── schemas.py                # Event schemas
│   │   └── producer.py               # Event producer
│   ├── computer_use/                 # Claude Computer Use API wrapper
│   ├── tools/                        # Playwright, API, DB tools
│   └── utils/                        # Logging, tokens, prompts
│
├── dashboard/                        # ⚛️ NEXT.JS DASHBOARD
│   ├── package.json
│   ├── app/                          # Next.js App Router pages
│   ├── components/                   # React components
│   └── lib/                          # Utilities and API client
│
├── argus-mcp-server/                 # 🔌 MCP SERVER (TypeScript)
│   ├── package.json                  # Note: Extracted to separate repo
│   ├── src/                          # MCP server implementation
│   └── README.md
│
├── cloudflare-worker/                # ☁️ CLOUDFLARE WORKER
│   ├── package.json
│   ├── src/                          # Worker source
│   └── wrangler.toml
│
├── crawlee-service/                  # 🕷️ CRAWLEE WEB SCRAPING
│   ├── package.json
│   └── src/
│
├── extension/                        # 🧩 BROWSER EXTENSION
│   └── manifest.json
│
├── docs/                             # 📚 DOCUMENTATION (MkDocs source)
│   └── *.md
│
├── docs-site/                        # 📚 DOCS SITE CONFIG
│   └── mkdocs.yml
│
├── status-page/                      # 📊 STATUS PAGE
│
├── supabase/                         # 🗄️ DATABASE
│   └── migrations/                   # SQL migrations
│
├── tests/                            # 🧪 PYTHON TESTS
│
└── scripts/                          # 🔧 UTILITY SCRIPTS
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Set up project structure with pyproject.toml
- [ ] Implement config management with pydantic-settings
- [ ] Create base agent class with Claude API integration
- [ ] Set up LangGraph orchestrator skeleton
- [ ] Implement logging and token cost tracking

### Phase 2: Computer Use Integration (Week 1-2)
- [ ] Implement Computer Use API client wrapper
- [ ] Create Docker sandbox management
- [ ] Build action execution layer (click, type, screenshot)
- [ ] Integrate with Playwright for hybrid testing
- [ ] Add screenshot analysis utilities

### Phase 3: Testing Agents (Week 2-3)
- [ ] Code Analyzer Agent - parses codebase, identifies test surfaces
- [ ] Test Planner Agent - creates prioritized test plans
- [ ] UI Tester Agent - executes browser-based tests
- [ ] API Tester Agent - tests endpoints with schema validation
- [ ] DB Tester Agent - validates data integrity

### Phase 4: Intelligence Layer (Week 3-4)
- [ ] Self-Healing Agent - analyzes failures, fixes tests
- [ ] Reporter Agent - generates reports, creates tickets
- [ ] Implement test result persistence
- [ ] Add regression detection

### Phase 5: Integration & Deployment (Week 4)
- [ ] MCP server for external integrations
- [ ] n8n webhook triggers
- [ ] CI/CD pipeline integration (GitHub Actions)
- [ ] Docker deployment configuration
- [ ] Documentation

## Key Implementation Details

### LangGraph State Schema

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages

class TestingState(TypedDict):
    # Conversation messages
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Codebase context
    codebase_path: str
    codebase_summary: str
    testable_surfaces: list[dict]
    
    # Test planning
    test_plan: list[dict]
    current_test_index: int
    
    # Execution state
    test_results: list[dict]
    failures: list[dict]
    screenshots: list[str]  # Base64 encoded
    
    # Self-healing
    healing_attempts: int
    healed_tests: list[dict]
    
    # Metadata
    iteration: int
    total_tokens: int
    total_cost: float
    
    # Control flow
    next_agent: str
    should_continue: bool
```

### Computer Use Client Pattern

```python
class ComputerUseClient:
    """Wrapper for Claude Computer Use API with cost tracking."""
    
    def __init__(self, model: str = "claude-sonnet-4-5"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    async def execute_task(
        self,
        task: str,
        screenshot_fn: Callable,
        action_fn: Callable,
        max_iterations: int = 30
    ) -> dict:
        """Execute a computer use task with full agent loop."""
        # Implementation in src/computer_use/client.py
```

### Agent Base Class

```python
class BaseAgent(ABC):
    """Base class for all testing agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = anthropic.Anthropic()
        self.model = config.model
        self.tools = self._get_tools()
    
    @abstractmethod
    def _get_tools(self) -> list[dict]:
        """Return tools available to this agent."""
        pass
    
    @abstractmethod
    async def execute(self, state: TestingState) -> TestingState:
        """Execute agent logic and return updated state."""
        pass
    
    def _call_claude(self, messages: list, **kwargs) -> Message:
        """Make a Claude API call with automatic retry and logging."""
        # Implementation with retry logic, token tracking
```

## ⚠️ CRITICAL: Unified AI Architecture

**NEVER add ad-hoc AI/Claude calls directly in API endpoints or services.** This codebase has a sophisticated unified AI architecture that ALL AI features MUST follow.

### The Golden Rule

```python
# ❌ WRONG - Never do this
def some_api_endpoint():
    client = anthropic.Anthropic()
    response = client.messages.create(model="claude-sonnet-4-5", ...)

# ✅ CORRECT - Always use the unified abstraction
class SomeAgent(BaseAgent):
    async def execute(self):
        response = await self._call_ai(
            messages=[...],
            task_type=TaskType.CODE_ANALYSIS,
            required_capabilities=[AICapability.REASONING],
            max_cost=0.05
        )
```

### Unified `_call_ai()` Method

All agents inherit from `BaseAgent` and use `_call_ai()` which provides:

```python
async def _call_ai(
    messages: list[dict],                          # OpenAI format
    task_type: TaskType | None = None,             # Routing hint
    required_capabilities: list[AICapability] = [], # VISION, TOOLS, JSON_MODE
    preferred_provider: str | None = None,         # "openrouter", "anthropic"
    max_cost: float | None = None,                 # Budget limit per call
    max_tokens: int = 4096,
    temperature: float = 0.0,
    tools: list[dict] | None = None,
    images: list[bytes] | None = None,
    json_mode: bool = False,
) -> AIResponse:
    """
    Unified AI abstraction that:
    1. Routes to best model based on TaskType + capabilities
    2. Manages costs (cheapest model for simple tasks)
    3. Provides automatic failover across 300+ models via OpenRouter
    4. Tracks all usage for billing
    5. Enforces budgets
    """
```

### TaskType-Based Model Routing

```
TaskType                    → Model Tier      → Cost
─────────────────────────────────────────────────────
TRIVIAL (classification)    → Flash           → $0.10-0.30/1M
SIMPLE (extraction)         → Value           → $0.14-0.50/1M
MODERATE (code analysis)    → Standard        → $0.50-2.00/1M
COMPLEX (visual comparison) → Premium         → $3-15/1M
EXPERT (self-healing)       → Expert          → $15+/1M
```

### Event-Driven AI Architecture

AI operations MUST emit events for downstream processing:

```
┌─────────────────────────────────────────────────────────────────────┐
│  API Endpoint → Agent._call_ai() → Model Router → Provider          │
│       │                                                │             │
│       ▼                                                ▼             │
│  EventProducer.send()                           AIResponse          │
│       │                                                              │
│       ▼                                                              │
│  Kafka Topic (argus.*)                                              │
│       │                                                              │
│       ▼                                                              │
│  Cognee Worker → ECL Pipeline → Knowledge Graph (Neo4j)            │
└─────────────────────────────────────────────────────────────────────┘
```

### Cognee Knowledge Graph Integration

Store patterns for semantic learning:

```python
# Store failure patterns for future reference
cognee_client = get_cognee_client()

# Add to knowledge base with embeddings for semantic search
await cognee_client.put(
    key=f"failure_pattern_{pattern_hash}",
    value={"failure_type": "selector_changed", "fix": "...", "confidence": 0.95},
    namespace=f"org_{org_id}_failure_patterns",
    embeddings=True,
)

# Later: Find similar past failures via semantic search
similar = await cognee_client.semantic_search(
    query="Button selector changed after refactor",
    namespace=f"org_{org_id}_failure_patterns",
    top_k=5,
)
```

### Multi-Tenant Dataset Naming

```
Format: org_{org_id}_project_{project_id}_{type}

Examples:
  org_acme_project_webapp_codebases
  org_acme_project_webapp_failure_patterns
  org_acme_project_webapp_healing_history
```

### Kafka Topics (12 Total)

```
# AI Learning Loop Topics
argus.codebase.ingested     → Code added to system
argus.codebase.analyzed     → Code analysis complete
argus.test.created          → New test created
argus.test.executed         → Test run complete
argus.test.failed           → Test failure (triggers healing)
argus.healing.requested     → Healing needed
argus.healing.completed     → Healing done (learning opportunity)
argus.dlq                   → Dead letter queue

# A2A Communication Topics (NEW)
argus.agent.request         → Agent-to-agent requests
argus.agent.response        → Responses to requests
argus.agent.broadcast       → Pub/sub broadcasts
argus.agent.heartbeat       → Health monitoring
```

### 🔄 Complete AI Learning Loop (CI/CD → Cognee → Self-Healing)

This is the **critical integration** that enables autonomous learning and self-healing. Understanding this flow is essential for maintaining and extending Argus.

#### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CI/CD API (src/api/cicd.py)                          │
│                                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐ │
│  │ Test Impact Analysis│    │ Deployment Risk    │    │ Test Execution  │ │
│  │ POST /test-impact   │    │ GET /deploy-risk   │    │ POST /tests/run │ │
│  └─────────┬───────────┘    └─────────┬──────────┘    └────────┬────────┘ │
│            │                          │                        │          │
│            ▼                          ▼                        ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  _emit_test_event()  |  _emit_healing_request()  |  _store_in_cognee│  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Redpanda/Kafka (Vultr K8s)                             │
│                                                                             │
│  argus.test.executed → argus.test.failed → argus.healing.requested         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Cognee Worker (data-layer/cognee-worker/)                  │
│                                                                             │
│  1. Consumes events from Kafka                                              │
│  2. Runs ECL Pipeline (Extract → Cognify → Load)                           │
│  3. Stores in Neo4j knowledge graph with embeddings                        │
│  4. Enables semantic search for pattern matching                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Self-Healing Agent (src/agents/self_healer.py)          │
│                                                                             │
│  1. Queries Cognee for similar past failures                               │
│  2. Finds successful fixes for similar patterns                            │
│  3. Applies fix with confidence score                                      │
│  4. Stores result back in Cognee (learning loop!)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Helper Functions (src/api/cicd.py)

These functions enable the AI learning loop in CI/CD endpoints:

```python
# 1. Emit test events to Kafka for downstream processing
async def _emit_test_event(
    event_type: EventType,
    test_data: dict,
    user_id: str,
    org_id: str,
    project_id: str,
) -> None:
    """Emit test execution events to Kafka for Cognee processing."""
    from src.events.producer import get_event_producer

    producer = get_event_producer()
    await producer.send(
        topic=f"argus.test.{event_type.value}",
        event=TestExecutedEvent(
            metadata=EventMetadata(user_id=user_id, org_id=org_id),
            project_id=project_id,
            **test_data,
        )
    )

# 2. Request self-healing for failed tests
async def _emit_healing_request(
    failure_data: dict,
    user_id: str,
    org_id: str,
    project_id: str,
) -> None:
    """Emit healing request when test fails - triggers SelfHealerAgent."""
    from src.events.producer import get_event_producer

    producer = get_event_producer()
    await producer.send(
        topic="argus.healing.requested",
        event=HealingRequestedEvent(
            metadata=EventMetadata(user_id=user_id, org_id=org_id),
            project_id=project_id,
            **failure_data,
        )
    )

# 3. Store analysis results in Cognee for learning
async def _store_analysis_in_cognee(
    analysis_type: str,  # "test_impact", "deployment_risk", etc.
    data: dict,
    org_id: str,
    project_id: str,
) -> None:
    """Store analysis results in Cognee knowledge graph."""
    from src.knowledge.cognee_client import get_cognee_client

    cognee = get_cognee_client()
    namespace = f"org_{org_id}_project_{project_id}_{analysis_type}"

    await cognee.put(
        key=f"{analysis_type}_{data.get('commit_sha', 'unknown')}",
        value=data,
        namespace=namespace,
        embeddings=True,  # Enable semantic search
    )
```

#### Wiring Up New Features

When adding CI/CD or testing features, follow this pattern:

```python
@router.post("/new-feature")
async def new_feature_endpoint(request: Request):
    # 1. Authenticate and get context
    user_id = request.state.user_id
    org_id = request.state.organization_id

    # 2. Query Cognee for historical context (optional but recommended)
    cognee = get_cognee_client()
    similar_patterns = await cognee.semantic_search(
        query="relevant search query",
        namespace=f"org_{org_id}_project_{project_id}_patterns",
        top_k=5,
    )

    # 3. Run AI analysis (use existing agents!)
    result = await some_agent.analyze(...)

    # 4. CRITICAL: Emit event to Kafka
    await _emit_test_event(
        event_type=EventType.TEST_EXECUTED,
        test_data=result.to_dict(),
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
    )

    # 5. CRITICAL: Store in Cognee for learning
    await _store_analysis_in_cognee(
        analysis_type="feature_analysis",
        data=result.to_dict(),
        org_id=org_id,
        project_id=project_id,
    )

    return result
```

#### Current Integration Status

| Endpoint | Kafka Events | Cognee Storage | Notes |
|----------|--------------|----------------|-------|
| `/cicd/test-impact/analyze` | ✅ | ✅ | Stores impact analysis for learning |
| `/cicd/deployment-risk` | ✅ | ✅ | AI-powered risk assessment |
| `/tests/run` | ⚠️ | ⚠️ | Needs `_emit_test_event` on failure |
| `/cicd/pipelines/*` | ⚠️ | ⚠️ | Pipeline events need wiring |
| `/cicd/builds/*` | ⚠️ | ⚠️ | Build events need wiring |
| Schedule executor | ⚠️ | ⚠️ | Scheduled tests need event emission |

**⚠️ = Partially implemented, needs completion**

#### Cognee Dataset Namespaces

```
org_{org_id}_project_{project_id}_test_impact       → Test impact analyses
org_{org_id}_project_{project_id}_deployment_risk   → Deployment risk assessments
org_{org_id}_project_{project_id}_failure_patterns  → Test failure patterns
org_{org_id}_project_{project_id}_healing_history   → Self-healing attempts
org_{org_id}_project_{project_id}_code_changes      → Code change patterns
```

### Creating New AI Features (Checklist)

When adding ANY new AI-powered feature:

1. [ ] **Create an Agent class** extending `BaseAgent`
2. [ ] **Use `_call_ai()`** with appropriate `TaskType`
3. [ ] **Set `max_cost`** to prevent runaway spending
4. [ ] **Query Cognee first** for historical patterns
5. [ ] **Emit events** to appropriate Kafka topic
6. [ ] **Store results in Cognee** for future learning
7. [ ] **Track costs** via `ai_cost_tracker`

### Example: Proper AI Feature Implementation

```python
# src/agents/test_impact_agent.py
class TestImpactAgent(BaseAgent):
    """AI-powered test impact analysis."""

    DEFAULT_TASK_TYPE = TaskType.CODE_ANALYSIS

    async def analyze_impact(
        self,
        changed_files: list[str],
        tests: list[dict],
        org_id: str,
        project_id: str,
    ) -> TestImpactResult:
        # 1. Query Cognee for historical patterns
        cognee = get_cognee_client()
        similar_changes = await cognee.semantic_search(
            query=f"code changes affecting {changed_files}",
            namespace=f"org_{org_id}_code_impact_patterns",
            top_k=5,
        )

        # 2. Build context-aware prompt
        prompt = self._build_prompt(changed_files, tests, similar_changes)

        # 3. Use unified _call_ai() - routes to best model automatically
        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            required_capabilities=[AICapability.REASONING],
            max_cost=0.05,  # Budget limit
            json_mode=True,
        )

        # 4. Parse and validate response
        result = self._parse_response(response.content)

        # 5. Emit event for downstream processing
        await self.event_producer.send(TestImpactAnalyzedEvent(
            org_id=org_id,
            project_id=project_id,
            changed_files=changed_files,
            impacted_tests=result.impacted_tests,
            confidence=result.confidence,
        ))

        # 6. Store in Cognee for future learning
        await cognee.put(
            key=f"impact_analysis_{hash(tuple(changed_files))}",
            value=result.to_dict(),
            namespace=f"org_{org_id}_code_impact_patterns",
            embeddings=True,
        )

        return result
```

### Key Files for AI Architecture

| File | Purpose |
|------|---------|
| `src/agents/base.py` | BaseAgent with `_call_ai()` abstraction |
| `src/agents/__init__.py` | Agent registry - all agents exported here |
| `src/agents/test_impact_analyzer.py` | AI-powered test impact analysis |
| `src/agents/router_agent.py` | Intelligent model selection |
| `src/core/providers/` | Provider implementations (11 total) |
| `src/core/model_router.py` | TaskType → Model routing logic |
| `src/knowledge/cognee_client.py` | Cognee knowledge graph client |
| `src/services/correlation_engine.py` | AI-powered SDLC event correlation |
| `src/events/producer.py` | Kafka event producer |
| `src/events/schemas.py` | Event type definitions |
| `src/services/ai_cost_tracker.py` | Cost tracking & budgets |
| `src/orchestrator/supervisor.py` | LangGraph supervisor - dynamic agent routing |
| `data-layer/cognee-worker/` | Kafka consumer → Cognee pipeline |

### ⚠️ CRITICAL: Agent Reuse Pattern

**NEVER duplicate AI logic in API endpoints.** Always reuse existing agents.

The agents in `src/agents/` already implement sophisticated AI logic. API endpoints should
IMPORT and USE these agents, not implement their own heuristics.

**Existing Agents (30+):**
```python
from src.agents import (
    # Core Testing Agents
    TestImpactAnalyzer,     # AI-powered test impact analysis
    SmartTestSelector,      # Risk-based test selection
    SelfHealerAgent,        # Self-healing with Cognee learning
    CodeAnalyzerAgent,      # Codebase understanding
    UITesterAgent,          # Browser testing
    APITesterAgent,         # API testing
    VisualAI,               # Screenshot comparison
    NLPTestCreator,         # Natural language → tests
    AutoDiscovery,          # Autonomous app crawling
    FlakyTestDetector,      # Flaky test detection
    PerformanceAnalyzerAgent,
    SecurityScannerAgent,
    AccessibilityCheckerAgent,

    # Advanced 2026 Agentic AI Patterns
    SREAgent,               # Unified incident correlation, runbooks, MTTR
    AgentAsJudge,           # Multi-agent evaluation with debate
    MetaJudge,              # Arbitrates judge disagreements
    CorrectiveRAGAgent,     # Self-correcting retrieval (CRAG)
    AdaptiveRAGRouter,      # Dynamic retrieval strategy selection
    ToolDiscoveryAgent,     # Discover tools from API docs (ToolLLM)
    ToolCreatorAgent,       # Create tools for capability gaps (CREATOR)
)
```

**WRONG - Duplicating logic in API endpoint:**
```python
# ❌ src/api/cicd.py
def _calculate_file_impact_score(file_path: str) -> float:
    # Heuristic implementation - BAD!
    if "auth" in file_path: return 0.8
    ...
```

**CORRECT - Reuse existing agent:**
```python
# ✅ src/api/cicd.py
from src.agents import TestImpactAnalyzer

async def analyze_test_impact(...):
    analyzer = TestImpactAnalyzer()
    analysis = await analyzer.analyze_impact(change, all_tests)
    return analysis
```

### Dynamic Agent Routing (LangGraph Supervisor)

The supervisor (`src/orchestrator/supervisor.py`) routes to agents DYNAMICALLY using an LLM.
To add a new agent:

1. Create agent in `src/agents/new_agent.py` extending `BaseAgent`
2. Export from `src/agents/__init__.py`
3. Add to `AGENTS` list and `AGENT_DESCRIPTIONS` in supervisor.py
4. Create wrapper function `supervisor_new_agent_node()`
5. Register in `create_supervisor_graph()`

The supervisor LLM will automatically route to the new agent when appropriate.

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional - for specific integrations
GITHUB_TOKEN=ghp_...
DATABASE_URL=postgresql://...
SLACK_WEBHOOK_URL=https://hooks.slack.com/...

# Configuration
DEFAULT_MODEL=claude-sonnet-4-5
MAX_ITERATIONS=50
SCREENSHOT_RESOLUTION=1920x1080
COST_LIMIT_PER_RUN=10.00  # USD
```

## Usage Examples

### Basic Test Run

```python
from e2e_testing_agent import TestingOrchestrator

orchestrator = TestingOrchestrator(
    codebase_path="/path/to/your/app",
    app_url="http://localhost:3000"
)

results = await orchestrator.run_full_test_suite()
print(results.summary)
```

### With n8n Webhook

```python
# Triggered by n8n workflow on PR creation
@app.post("/webhook/test")
async def run_tests(payload: PRPayload):
    orchestrator = TestingOrchestrator(
        codebase_path=payload.repo_path,
        app_url=payload.preview_url,
        pr_number=payload.pr_number
    )
    results = await orchestrator.run_changed_file_tests()
    return results.to_github_check()
```

## Important Constraints

1. **Cost Control**: Always implement max iteration limits and cost tracking
2. **Security**: Never execute Computer Use on production systems - always use sandboxes
3. **Idempotency**: Tests should be runnable multiple times with same results
4. **Observability**: Log all Claude API calls, screenshots, and decisions
5. **Graceful Degradation**: If Computer Use fails, fall back to Playwright-only mode

## Claude Code Instructions

When implementing this project:

1. **Start with Phase 1** - get the infrastructure solid before adding agents
2. **Test Computer Use locally first** using the Anthropic reference Docker container
3. **Use Sonnet 4.5 as default** - it has the best cost/capability balance for testing
4. **Implement token tracking early** - costs can spiral quickly with screenshots
5. **Build the hybrid Playwright+ComputerUse approach** - Playwright for speed, Computer Use for verification

## 🚀 Deployment URLs & Infrastructure

### Production Endpoints

| Service | URL | Description |
|---------|-----|-------------|
| **Backend API** | `https://argus-brain-production.up.railway.app` | Main FastAPI backend (Railway) |
| **Dashboard** | `https://app.skopaq.ai` | Next.js dashboard (Vercel) |
| **Docs** | `https://docs.skopaq.ai` | MkDocs documentation |
| **Status Page** | `https://status.skopaq.ai` | Service status |

### API Health Check

```bash
# Check backend health
curl https://argus-brain-production.up.railway.app/health

# Check API version
curl https://argus-brain-production.up.railway.app/api/v1/health
```

### Railway CLI Commands

```bash
# Check current service
railway status

# Get domain
railway domain

# View logs
railway logs

# Deploy
railway up --detach

# Check variables
railway variables
```

### Kubernetes (Vultr VKE) - Data Layer

```bash
# Namespaces
kubectl get ns argus-data    # Data layer (Redpanda, FalkorDB, Valkey, Cognee)
kubectl get ns monitoring    # Prometheus, Grafana, Alertmanager

# Check pods
kubectl get pods -n argus-data
kubectl get pods -n monitoring

# Redpanda cluster health
kubectl exec -n argus-data redpanda-0 -- rpk cluster health

# List Kafka topics
kubectl exec -n argus-data redpanda-0 -- rpk topic list

# Check consumer group (Cognee worker)
kubectl exec -n argus-data redpanda-0 -- rpk group describe argus-cognee-workers

# View Grafana dashboards (via Cloudflare Tunnel)
# Access: https://monitoring.skopaq.ai/grafana
```

### Data Layer Health Check (from API)

```bash
# Full data layer health (9 components)
curl -s "https://argus-brain-production.up.railway.app/api/v1/health/data-layer" \
  -H "X-API-Key: YOUR_API_KEY" | jq .

# Individual component
curl -s "https://argus-brain-production.up.railway.app/api/v1/health/data-layer/redpanda" \
  -H "X-API-Key: YOUR_API_KEY"
```

### Verified Components (Jan 2026)

| Component | Status | Access | Notes |
|-----------|--------|--------|-------|
| Redpanda | ✅ Healthy | K8s Internal + NodePort | 65.20.67.126:31092 |
| Cognee | ✅ v0.5.1 | Supabase PostgreSQL | pgvector enabled |
| FalkorDB | ✅ Healthy | K8s Internal | Graph DB for knowledge |
| Valkey | ✅ Healthy | K8s Internal | Redis-compatible cache |
| Flink | ✅ Healthy | K8s Internal | Stream processing |
| Selenium Grid | ✅ 3 nodes | K8s + NodePort | Browser automation |
| Prometheus | ✅ Healthy | Cloudflare Tunnel | Metrics collection |
| Grafana | ✅ v12.3.1 | Cloudflare Tunnel | Dashboards |
| Supabase | ✅ Healthy | Public API | Database + Auth |

### A2A Architecture Verification

```bash
# Verify A2A Kafka topics exist
kubectl exec -n argus-data redpanda-0 -- rpk topic list | grep agent
# Expected: argus.agent.{request,response,broadcast,heartbeat}

# Check topic partitions
kubectl exec -n argus-data redpanda-0 -- rpk topic describe argus.agent.request

# Test AI-powered test impact analysis
curl -X POST "https://argus-brain-production.up.railway.app/api/v1/cicd/test-impact/analyze" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "X-Organization-Id: YOUR_ORG_ID" \
  -H "Content-Type: application/json" \
  -d '{"project_id":"PROJECT_ID","commit_sha":"test","branch":"main","changed_files":[{"path":"src/auth.py","change_type":"modified","additions":10,"deletions":5}]}'

# Test CI/CD endpoints
curl "https://argus-brain-production.up.railway.app/api/v1/cicd/builds?project_id=PROJECT_ID" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "X-Organization-Id: YOUR_ORG_ID"
```

### Environment Variable Groups

| Group | Source | Services |
|-------|--------|----------|
| Supabase | Railway secrets | Backend, Dashboard |
| Cloudflare | Railway secrets | Backend, Workers |
| Anthropic/OpenRouter | Railway secrets | Backend |
| Kubernetes | ConfigMaps/Secrets | Data layer |

## References

- [Claude Computer Use Docs](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Anthropic Quickstarts - Computer Use Demo](https://github.com/anthropics/anthropic-quickstarts)
- [Playwright Python Docs](https://playwright.dev/python/)
