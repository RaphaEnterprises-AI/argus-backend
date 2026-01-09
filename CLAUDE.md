# Autonomous E2E Full-Stack Testing Agent

## Project Overview

This project builds a **fully autonomous end-to-end testing agent** that can:
1. Understand any codebase and generate comprehensive test plans
2. Execute UI tests using Claude's Computer Use API
3. Execute API tests with schema validation
4. Validate database state and data integrity
5. Self-heal broken tests by analyzing failures and fixing selectors/assertions
6. Generate human-readable reports and integrate with CI/CD pipelines

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
- Automatic task delegation based on current state
- File: `src/orchestrator/supervisor.py`

### Chat-through-Orchestrator
- All chat routed through LangGraph
- Full tool execution with checkpointing
- Conversation memory with semantic search
- File: `src/orchestrator/chat_graph.py`

## Tech Stack

- **Orchestration**: LangGraph (Python)
- **AI Models**: Claude Sonnet 4.5 (primary), Claude Haiku 4.5 (fast checks), Claude Opus 4.5 (complex debugging)
- **UI Testing**: Claude Computer Use API + Playwright (hybrid)
- **API Testing**: httpx + pydantic for validation
- **Database**: sqlalchemy for DB introspection
- **MCP Servers**: filesystem, github, postgres (configurable)
- **Workflow Trigger**: n8n webhooks (optional)

## Directory Structure

```
e2e-testing-agent/
├── CLAUDE.md                    # This file - project instructions
├── README.md                    # User documentation
├── pyproject.toml               # Python dependencies
├── .env.example                 # Environment variables template
│
├── src/
│   ├── __init__.py
│   ├── main.py                  # Entry point
│   ├── config.py                # Configuration management
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── graph.py             # LangGraph state machine
│   │   ├── state.py             # Shared state definitions
│   │   └── nodes.py             # Graph node implementations
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # Base agent class
│   │   ├── code_analyzer.py     # Analyzes codebase, generates test specs
│   │   ├── test_planner.py      # Creates prioritized test plans
│   │   ├── ui_tester.py         # Computer Use + Playwright hybrid
│   │   ├── api_tester.py        # API endpoint testing
│   │   ├── db_tester.py         # Database validation
│   │   ├── self_healer.py       # Auto-fixes broken tests
│   │   └── reporter.py          # Generates reports
│   │
│   ├── computer_use/
│   │   ├── __init__.py
│   │   ├── client.py            # Claude Computer Use API wrapper
│   │   ├── sandbox.py           # Docker sandbox management
│   │   ├── actions.py           # Action execution (click, type, etc)
│   │   └── screenshot.py        # Screenshot capture utilities
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── playwright_tools.py  # Playwright automation tools
│   │   ├── api_tools.py         # HTTP request tools
│   │   ├── db_tools.py          # Database query tools
│   │   ├── git_tools.py         # Git operations
│   │   └── file_tools.py        # File system operations
│   │
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py            # MCP server implementation
│   │   └── tools.py             # MCP tool definitions
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # Structured logging
│       ├── tokens.py            # Token counting & cost tracking
│       └── prompts.py           # Prompt templates
│
├── tests/
│   ├── __init__.py
│   ├── test_orchestrator.py
│   ├── test_agents.py
│   └── test_computer_use.py
│
├── skills/                      # Claude Code skills for this project
│   └── e2e-testing/
│       └── SKILL.md
│
└── docs/
    ├── ARCHITECTURE.md
    ├── COMPUTER_USE_GUIDE.md
    ├── API_REFERENCE.md
    └── DEPLOYMENT.md
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

## References

- [Claude Computer Use Docs](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Anthropic Quickstarts - Computer Use Demo](https://github.com/anthropics/anthropic-quickstarts)
- [Playwright Python Docs](https://playwright.dev/python/)
