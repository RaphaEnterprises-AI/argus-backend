# Argus E2E Testing Platform - Complete Architecture Diagrams

> **Comprehensive technical diagrams showing every feature, functionality, component, and data flow in the Argus platform.**

This document contains the following diagram types:
- **C4 Model Diagrams** (Context, Container, Component levels)
- **Sequence Diagrams** (UI → API → DB flows)
- **Entity-Relationship Diagrams** (Database schema)
- **Module Dependency Graphs**
- **Data Flow Diagrams**
- **Agent Capability Matrix**
- **Authentication & Authorization Flows**

---

## Table of Contents

1. [C4 Level 1 - System Context Diagram](#1-c4-model---system-context-level-1)
2. [C4 Level 2 - Container Diagram](#2-c4-model---container-diagram-level-2)
3. [C4 Level 3 - Component Diagram (Backend)](#3-c4-model---component-diagram-level-3---backend)
4. [Sequence Diagram - Test Execution Flow](#4-sequence-diagram---ui--api--db-flow-test-execution)
5. [Entity-Relationship Diagram (ERD)](#5-entity-relationship-diagram-erd---database-schema)
6. [Module Dependency Graph](#6-module-dependency-graph)
7. [Data Flow Diagram - Complete User Journey](#7-data-flow-diagram---complete-user-journey)
8. [API Endpoints → Database Table Mapping](#8-api-endpoints--database-table-mapping)
9. [Agent Capability Matrix](#9-agent-capability-matrix)
10. [Frontend Component Hierarchy](#10-frontend-component-hierarchy)
11. [Authentication & Authorization Flow](#11-authentication--authorization-flow)
12. [Self-Healing Intelligence Flow](#12-self-healing-intelligence-flow)
13. [Diagram Types Reference](#diagram-types--terminology)

---

## 1. C4 MODEL - SYSTEM CONTEXT (Level 1)

> Shows the Argus platform in context with external systems and users.

```mermaid
graph TB
    subgraph External["External Systems"]
        User["👤 QA Engineer/Developer"]
        GitHub["GitHub<br/>Code Repository"]
        Slack["Slack<br/>Notifications"]
        CICD["CI/CD Pipeline<br/>GitHub Actions"]
        Claude["Claude AI<br/>Anthropic API"]
        Cloudflare["Cloudflare<br/>Browser Rendering"]
        Crawlee["Crawlee Service<br/>Web Crawling"]
    end

    subgraph Argus["ARGUS PLATFORM"]
        Dashboard["Next.js Dashboard<br/>Web UI"]
        Backend["FastAPI Backend<br/>Python"]
        Supabase["Supabase<br/>PostgreSQL + Auth"]
    end

    User --> Dashboard
    Dashboard --> Backend
    Backend --> Supabase
    Backend --> Claude
    Backend --> Cloudflare
    Backend --> Crawlee
    Backend --> GitHub
    Backend --> Slack
    CICD --> Backend

    style Argus fill:#e1f5fe,stroke:#01579b
    style External fill:#f3e5f5,stroke:#7b1fa2
```

---

## 2. C4 MODEL - CONTAINER DIAGRAM (Level 2)

> Shows the major containers (applications/services) that make up the Argus platform.

```mermaid
graph TB
    subgraph Frontend["📱 FRONTEND LAYER"]
        Dashboard["Next.js Dashboard<br/>React + TypeScript<br/>:3000"]
        Extension["Browser Extension<br/>Chrome/Firefox"]
    end

    subgraph Backend["⚙️ BACKEND LAYER"]
        API["FastAPI Server<br/>Python 3.14<br/>:8000"]
        Worker["Cloudflare Worker<br/>Browser Automation"]
        CrawleeService["Crawlee Service<br/>Node.js<br/>:3002"]
        MCPServer["MCP Server<br/>Tool Interface"]
    end

    subgraph Orchestration["🔄 ORCHESTRATION LAYER"]
        LangGraph["LangGraph<br/>State Machine"]
        Supervisor["Supervisor Agent<br/>Task Router"]
        Agents["AI Agents Pool<br/>Specialized Workers"]
    end

    subgraph Storage["💾 DATA LAYER"]
        Supabase["Supabase PostgreSQL<br/>+ pgvector"]
        R2["Cloudflare R2<br/>Screenshots/Artifacts"]
        Redis["Redis Cache<br/>Session State"]
    end

    subgraph External["🌐 EXTERNAL SERVICES"]
        Claude["Claude API<br/>Anthropic"]
        Clerk["Clerk Auth<br/>Authentication"]
        Resend["Resend<br/>Email Service"]
        Stripe["Stripe<br/>Billing"]
    end

    Dashboard --> API
    Extension --> API
    Dashboard --> Clerk

    API --> LangGraph
    LangGraph --> Supervisor
    Supervisor --> Agents
    API --> CrawleeService
    API --> Worker

    Agents --> Claude
    API --> Supabase
    API --> R2
    API --> Redis

    API --> Resend
    API --> Stripe

    style Frontend fill:#c8e6c9,stroke:#2e7d32
    style Backend fill:#bbdefb,stroke:#1565c0
    style Orchestration fill:#fff3e0,stroke:#ef6c00
    style Storage fill:#f3e5f5,stroke:#7b1fa2
    style External fill:#fce4ec,stroke:#c2185b
```

---

## 3. C4 MODEL - COMPONENT DIAGRAM (Level 3) - BACKEND

> Shows the internal components of the FastAPI backend.

```mermaid
graph TB
    subgraph API["📡 API Layer (src/api/)"]
        Server["server.py<br/>FastAPI App"]
        Auth["security/<br/>auth.py, rbac.py"]
        Projects["projects.py"]
        Tests["tests.py"]
        Discovery["discovery.py"]
        Visual["visual_ai.py"]
        Healing["healing.py"]
        Chat["chat.py"]
        Streaming["streaming.py"]
        Webhooks["webhooks.py"]
        TimeTrav["time_travel.py"]
        Approvals["approvals.py"]
    end

    subgraph Agents["🤖 AI Agents (src/agents/)"]
        BaseAgent["base.py<br/>BaseAgent"]
        CodeAnalyzer["code_analyzer.py"]
        TestPlanner["test_planner.py"]
        UITester["ui_tester.py"]
        APITester["api_tester.py"]
        DBTester["db_tester.py"]
        SelfHealer["self_healer.py"]
        Reporter["reporter.py"]
        VisualAI["visual_ai.py"]
        SecurityScanner["security_scanner.py"]
        AccessibilityChecker["accessibility_checker.py"]
        FlakyDetector["flaky_detector.py"]
        NLPTestCreator["nlp_test_creator.py"]
        AutoDiscovery["auto_discovery.py"]
    end

    subgraph Orchestrator["🎯 Orchestration (src/orchestrator/)"]
        Graph["graph.py<br/>LangGraph State Machine"]
        GraphV2["graph_v2.py"]
        SupervisorMod["supervisor.py<br/>Multi-Agent Router"]
        State["state.py<br/>TestingState"]
        Nodes["nodes.py"]
        ChatGraph["chat_graph.py"]
        Checkpointer["checkpointer.py<br/>PostgresSaver"]
        MemoryStore["memory_store.py<br/>pgvector"]
    end

    subgraph Services["🔧 Services (src/services/)"]
        SupabaseClient["supabase_client.py"]
        EmailService["email_service.py"]
        CrawleeClient["crawlee_client.py"]
        GitAnalyzer["git_analyzer.py"]
        AITracker["ai_cost_tracker.py"]
        Cache["cache.py"]
        Vectorize["vectorize.py"]
    end

    subgraph ComputerUse["🖥️ Computer Use (src/computer_use/)"]
        Client["client.py"]
        Actions["actions.py"]
        Sandbox["sandbox.py"]
        Screenshot["screenshot.py"]
    end

    subgraph VisualAICore["👁️ Visual AI (src/visual_ai/)"]
        SemanticAnalyzer["semantic_analyzer.py"]
        StructuralAnalyzer["structural_analyzer.py"]
        PerceptualAnalyzer["perceptual_analyzer.py"]
        CrossBrowser["cross_browser_analyzer.py"]
        Responsive["responsive_analyzer.py"]
        Accessibility["accessibility_analyzer.py"]
    end

    Server --> Auth
    Server --> Projects
    Server --> Tests
    Server --> Discovery
    Server --> Visual
    Server --> Healing
    Server --> Chat
    Server --> Streaming
    Server --> Webhooks

    Projects --> Graph
    Tests --> Graph
    Discovery --> Graph

    Graph --> SupervisorMod
    Graph --> State
    Graph --> Nodes
    SupervisorMod --> Agents

    Agents --> BaseAgent
    BaseAgent --> CodeAnalyzer
    BaseAgent --> TestPlanner
    BaseAgent --> UITester
    BaseAgent --> APITester
    BaseAgent --> SelfHealer

    UITester --> ComputerUse
    VisualAI --> VisualAICore

    Graph --> Checkpointer
    Graph --> MemoryStore

    Checkpointer --> SupabaseClient
    MemoryStore --> SupabaseClient

    style API fill:#e3f2fd
    style Agents fill:#fff8e1
    style Orchestrator fill:#e8f5e9
    style Services fill:#fce4ec
    style ComputerUse fill:#f3e5f5
    style VisualAICore fill:#e0f7fa
```

---

## 4. SEQUENCE DIAGRAM - UI → API → DB FLOW (Test Execution)

> Shows the complete flow of a test execution from user click to results display.

```mermaid
sequenceDiagram
    participant U as User
    participant D as Dashboard (Next.js)
    participant A as FastAPI Backend
    participant L as LangGraph
    participant S as Supervisor
    participant AG as AI Agents
    participant C as Claude API
    participant DB as Supabase

    U->>D: Click "Run Tests"
    D->>A: POST /api/v1/tests/run
    A->>A: Validate JWT + RBAC
    A->>DB: INSERT test_runs (status=pending)
    DB-->>A: test_run_id

    A->>L: start_test_run(config)
    L->>L: Create checkpoint
    L->>DB: INSERT langgraph_checkpoints

    L->>S: Route to Code Analyzer
    S->>AG: CodeAnalyzer.execute()
    AG->>C: Analyze codebase
    C-->>AG: Analysis result
    AG->>DB: UPDATE test_runs (specs)
    AG-->>S: Analysis complete

    S->>AG: TestPlanner.execute()
    AG->>C: Generate test plan
    C-->>AG: Test plan
    AG->>DB: INSERT generated_tests
    AG-->>S: Plan complete

    S->>AG: UITester.execute()
    AG->>C: Computer Use API
    C-->>AG: Actions
    AG->>DB: INSERT test_results
    AG-->>S: Tests complete

    loop Self-Healing (if failures)
        S->>AG: SelfHealer.execute()
        AG->>DB: SELECT test_failure_patterns
        AG->>C: Analyze + fix
        C-->>AG: Healed selectors
        AG->>DB: UPDATE healing_patterns
    end

    S->>AG: Reporter.execute()
    AG->>DB: SELECT test_results
    AG->>C: Generate report
    AG->>DB: UPDATE test_runs (completed)

    L-->>A: Final state
    A-->>D: SSE: test_completed
    D-->>U: Show results
```

---

## 5. ENTITY-RELATIONSHIP DIAGRAM (ERD) - DATABASE SCHEMA

> Shows all database tables and their relationships.

```mermaid
erDiagram
    ORGANIZATIONS ||--o{ ORGANIZATION_MEMBERS : has
    ORGANIZATIONS ||--o{ PROJECTS : contains
    ORGANIZATIONS ||--o{ AI_USAGE : tracks
    ORGANIZATIONS ||--o{ AI_USAGE_DAILY : aggregates
    ORGANIZATIONS ||--o{ API_KEYS : owns

    ORGANIZATION_MEMBERS }o--|| USER_PROFILES : represents

    PROJECTS ||--o{ TEST_RUNS : executes
    PROJECTS ||--o{ GENERATED_TESTS : contains
    PROJECTS ||--o{ RISK_SCORES : has
    PROJECTS ||--o{ DISCOVERY_SESSIONS : runs
    PROJECTS ||--o{ DISCOVERY_HISTORY : snapshots

    TEST_RUNS ||--o{ TEST_RESULTS : produces
    TEST_RUNS ||--o{ LANGGRAPH_CHECKPOINTS : saves

    DISCOVERY_SESSIONS ||--o{ DISCOVERED_PAGES : finds
    DISCOVERY_SESSIONS ||--o{ DISCOVERED_ELEMENTS : extracts
    DISCOVERY_SESSIONS ||--o{ DISCOVERED_FLOWS : identifies

    DISCOVERED_PAGES ||--o{ DISCOVERED_ELEMENTS : contains

    TEST_FAILURE_PATTERNS ||--o{ HEALING_PATTERNS : informs

    LANGGRAPH_MEMORY_STORE ||--o{ TEST_FAILURE_PATTERNS : learns

    ORGANIZATIONS {
        uuid id PK
        text name
        text slug UK
        text plan
        numeric ai_budget_daily
        numeric ai_budget_monthly
        jsonb features
        text stripe_customer_id
    }

    ORGANIZATION_MEMBERS {
        uuid id PK
        uuid organization_id FK
        text user_id
        text email
        text role
        text status
    }

    USER_PROFILES {
        uuid id PK
        text user_id UK
        text email
        text display_name
        uuid default_organization_id FK
        boolean onboarding_completed
    }

    PROJECTS {
        uuid id PK
        uuid organization_id FK
        text name
        text url
        jsonb config
    }

    TEST_RUNS {
        uuid id PK
        uuid project_id FK
        text status
        jsonb test_plan
        jsonb results
        timestamptz started_at
        timestamptz completed_at
    }

    GENERATED_TESTS {
        uuid id PK
        uuid project_id FK
        text name
        text code
        numeric confidence_score
        text github_pr_url
    }

    DISCOVERY_SESSIONS {
        uuid id PK
        uuid project_id FK
        text name
        text status
        text start_url
        text mode
        text strategy
        integer max_pages
        jsonb page_graph
        numeric quality_score
        jsonb insights
    }

    DISCOVERED_PAGES {
        uuid id PK
        uuid discovery_session_id FK
        text url
        text page_type
        text screenshot_url
        integer element_count
        numeric importance_score
        numeric risk_score
    }

    DISCOVERED_ELEMENTS {
        uuid id PK
        uuid page_id FK
        text selector
        text category
        text label
        numeric importance_score
        numeric stability_score
    }

    DISCOVERED_FLOWS {
        uuid id PK
        uuid discovery_session_id FK
        text name
        text flow_type
        jsonb steps
        numeric complexity_score
        boolean validated
    }

    AI_USAGE {
        uuid id PK
        uuid organization_id FK
        text model
        integer input_tokens
        integer output_tokens
        numeric cost_usd
        text task_type
    }

    LANGGRAPH_CHECKPOINTS {
        uuid id PK
        text thread_id
        text checkpoint_id
        bytea state
    }

    LANGGRAPH_MEMORY_STORE {
        uuid id PK
        text_array namespace
        text key
        jsonb value
        vector embedding
    }

    TEST_FAILURE_PATTERNS {
        uuid id PK
        text error_message
        text healed_selector
        text healing_method
        vector embedding
    }

    HEALING_PATTERNS {
        uuid id PK
        text pattern_type
        text original_selector
        text healed_selector
        jsonb git_context
    }
```

---

## 6. MODULE DEPENDENCY GRAPH

> Shows how Python modules depend on each other.

```mermaid
graph LR
    subgraph Entry["Entry Points"]
        Main["main.py"]
        Server["api/server.py"]
    end

    subgraph Core["Core Modules"]
        Config["config.py"]
        State["orchestrator/state.py"]
    end

    subgraph Orchestration
        Graph["orchestrator/graph.py"]
        Supervisor["orchestrator/supervisor.py"]
        Nodes["orchestrator/nodes.py"]
        ChatGraph["orchestrator/chat_graph.py"]
        Checkpointer["orchestrator/checkpointer.py"]
        Memory["orchestrator/memory_store.py"]
    end

    subgraph Agents
        Base["agents/base.py"]
        CodeAn["agents/code_analyzer.py"]
        TestPlan["agents/test_planner.py"]
        UITest["agents/ui_tester.py"]
        APITest["agents/api_tester.py"]
        Healer["agents/self_healer.py"]
        Report["agents/reporter.py"]
        VisAI["agents/visual_ai.py"]
    end

    subgraph Services
        Supa["services/supabase_client.py"]
        Crawlee["services/crawlee_client.py"]
        Email["services/email_service.py"]
        Cost["services/ai_cost_tracker.py"]
    end

    subgraph ComputerUse
        CUClient["computer_use/client.py"]
        CUActions["computer_use/actions.py"]
        CUSandbox["computer_use/sandbox.py"]
    end

    subgraph Integrations
        GitHub["integrations/github_integration.py"]
        Slack["integrations/slack.py"]
        Observ["integrations/observability_hub.py"]
    end

    Main --> Server
    Server --> Config
    Server --> Graph
    Server --> Supervisor

    Graph --> State
    Graph --> Nodes
    Graph --> Checkpointer
    Graph --> Memory

    Supervisor --> Base
    Base --> CodeAn
    Base --> TestPlan
    Base --> UITest
    Base --> APITest
    Base --> Healer
    Base --> Report
    Base --> VisAI

    Checkpointer --> Supa
    Memory --> Supa
    Base --> Cost

    UITest --> CUClient
    CUClient --> CUActions
    CUClient --> CUSandbox

    Report --> GitHub
    Report --> Slack

    style Entry fill:#ffcdd2
    style Core fill:#fff9c4
    style Orchestration fill:#c8e6c9
    style Agents fill:#bbdefb
    style Services fill:#e1bee7
    style ComputerUse fill:#b2dfdb
    style Integrations fill:#ffe0b2
```

---

## 7. DATA FLOW DIAGRAM - COMPLETE USER JOURNEY

> Shows how data flows through the entire system from user action to database.

```mermaid
flowchart TB
    subgraph User["👤 USER ACTIONS"]
        Login["Login via Clerk"]
        SelectProject["Select Project"]
        StartDiscovery["Start Discovery"]
        RunTests["Run Tests"]
        ViewResults["View Results"]
    end

    subgraph Dashboard["📱 NEXT.JS DASHBOARD"]
        AuthProvider["Clerk Auth Provider"]
        OrgContext["Organization Context"]
        ProjectsPage["Projects Page"]
        DiscoveryPage["Discovery Page"]
        TestsPage["Tests Page"]
        ResultsPage["Results Page"]
        RealtimeSSE["SSE Streaming"]
    end

    subgraph API["⚙️ FASTAPI BACKEND"]
        AuthMiddleware["JWT Validation"]
        RBACCheck["RBAC Authorization"]
        ProjectsAPI["GET /projects"]
        DiscoveryAPI["POST /discovery/start"]
        TestsAPI["POST /tests/run"]
        StreamAPI["GET /stream/{session_id}"]
    end

    subgraph Orchestration["🎯 LANGGRAPH"]
        TestGraph["Testing Graph"]
        DiscoveryGraph["Discovery Graph"]
        StateManager["State Manager"]
        Checkpoint["PostgresSaver"]
    end

    subgraph Agents["🤖 AI AGENTS"]
        direction TB
        CodeAnalyzer["Code Analyzer"]
        TestPlanner["Test Planner"]
        UITester["UI Tester"]
        SelfHealer["Self Healer"]
        Reporter["Reporter"]
        AutoDiscovery["Auto Discovery"]
    end

    subgraph External["🌐 EXTERNAL"]
        Claude["Claude API"]
        Cloudflare["Cloudflare Browser"]
        CrawleeExt["Crawlee Service"]
    end

    subgraph Database["💾 SUPABASE"]
        direction TB
        OrgTable["organizations"]
        ProjectTable["projects"]
        TestRunTable["test_runs"]
        DiscoveryTable["discovery_sessions"]
        PagesTable["discovered_pages"]
        CheckpointTable["langgraph_checkpoints"]
        MemoryTable["langgraph_memory_store"]
    end

    Login --> AuthProvider
    AuthProvider --> OrgContext
    OrgContext --> ProjectsPage

    SelectProject --> ProjectsPage
    ProjectsPage --> ProjectsAPI
    ProjectsAPI --> AuthMiddleware
    AuthMiddleware --> RBACCheck
    RBACCheck --> ProjectTable

    StartDiscovery --> DiscoveryPage
    DiscoveryPage --> DiscoveryAPI
    DiscoveryAPI --> DiscoveryGraph
    DiscoveryGraph --> AutoDiscovery
    AutoDiscovery --> CrawleeExt
    AutoDiscovery --> Claude
    AutoDiscovery --> DiscoveryTable
    AutoDiscovery --> PagesTable

    RunTests --> TestsPage
    TestsPage --> TestsAPI
    TestsAPI --> TestGraph
    TestGraph --> StateManager
    StateManager --> Checkpoint
    Checkpoint --> CheckpointTable

    TestGraph --> CodeAnalyzer
    CodeAnalyzer --> Claude
    CodeAnalyzer --> TestPlanner
    TestPlanner --> Claude
    TestPlanner --> UITester
    UITester --> Claude
    UITester --> Cloudflare
    UITester --> SelfHealer
    SelfHealer --> MemoryTable
    SelfHealer --> Reporter
    Reporter --> TestRunTable

    StreamAPI --> RealtimeSSE
    RealtimeSSE --> ResultsPage
    ViewResults --> ResultsPage

    style User fill:#e8f5e9
    style Dashboard fill:#e3f2fd
    style API fill:#fff3e0
    style Orchestration fill:#f3e5f5
    style Agents fill:#fffde7
    style External fill:#fce4ec
    style Database fill:#e0f2f1
```

---

## 8. API ENDPOINTS → DATABASE TABLE MAPPING

> Shows which API endpoints interact with which database tables.

```mermaid
flowchart LR
    subgraph Endpoints["API ENDPOINTS"]
        E1["POST /api/v1/projects"]
        E2["GET /api/v1/projects/{id}"]
        E3["POST /api/v1/tests/run"]
        E4["GET /api/v1/tests/{id}/results"]
        E5["POST /api/v1/discovery/start"]
        E6["GET /api/v1/discovery/{id}/pages"]
        E7["GET /api/v1/healing/patterns"]
        E8["POST /api/v1/visual/compare"]
        E9["GET /api/v1/organizations"]
        E10["POST /api/v1/chat"]
        E11["GET /api/v1/users/me"]
    end

    subgraph Tables["DATABASE TABLES"]
        T1["projects"]
        T2["test_runs"]
        T3["test_results"]
        T4["discovery_sessions"]
        T5["discovered_pages"]
        T6["discovered_elements"]
        T7["discovered_flows"]
        T8["healing_patterns"]
        T9["test_failure_patterns"]
        T10["visual_comparisons"]
        T11["organizations"]
        T12["organization_members"]
        T13["user_profiles"]
        T14["langgraph_checkpoints"]
        T15["langgraph_memory_store"]
        T16["ai_usage"]
    end

    E1 --> T1
    E2 --> T1
    E3 --> T2
    E3 --> T14
    E3 --> T16
    E4 --> T2
    E4 --> T3
    E5 --> T4
    E5 --> T5
    E5 --> T6
    E5 --> T7
    E6 --> T5
    E7 --> T8
    E7 --> T9
    E8 --> T10
    E9 --> T11
    E9 --> T12
    E10 --> T15
    E10 --> T14
    E11 --> T13

    style Endpoints fill:#e3f2fd
    style Tables fill:#e8f5e9
```

---

## 9. AGENT CAPABILITY MATRIX

> Shows which agents have which capabilities and use which tools.

```mermaid
graph TB
    subgraph Agents["🤖 AI AGENTS"]
        CA["CodeAnalyzer"]
        TP["TestPlanner"]
        UT["UITester"]
        AT["APITester"]
        DT["DBTester"]
        SH["SelfHealer"]
        RP["Reporter"]
        VA["VisualAI"]
        AD["AutoDiscovery"]
        SS["SecurityScanner"]
        AC["AccessibilityChecker"]
        FD["FlakyDetector"]
        NT["NLPTestCreator"]
        PA["PerformanceAnalyzer"]
        RA["RootCauseAnalyzer"]
    end

    subgraph Capabilities["CAPABILITIES"]
        C1["Parse Codebase"]
        C2["Generate Tests"]
        C3["Execute UI Tests"]
        C4["Execute API Tests"]
        C5["Validate Database"]
        C6["Fix Broken Tests"]
        C7["Generate Reports"]
        C8["Visual Comparison"]
        C9["Discover App Structure"]
        C10["Security Scanning"]
        C11["Accessibility Check"]
        C12["Detect Flaky Tests"]
        C13["NLP to Tests"]
        C14["Performance Analysis"]
        C15["Root Cause Analysis"]
    end

    subgraph Tools["TOOLS USED"]
        T1["Claude API"]
        T2["Computer Use"]
        T3["Playwright"]
        T4["httpx"]
        T5["SQLAlchemy"]
        T6["pgvector"]
        T7["Tree-sitter"]
        T8["Crawlee"]
        T9["Cloudflare Browser"]
    end

    CA --> C1
    CA --> T7

    TP --> C2
    TP --> T1

    UT --> C3
    UT --> T2
    UT --> T3
    UT --> T9

    AT --> C4
    AT --> T4

    DT --> C5
    DT --> T5

    SH --> C6
    SH --> T6
    SH --> T1

    RP --> C7
    RP --> T1

    VA --> C8
    VA --> T1
    VA --> T9

    AD --> C9
    AD --> T8
    AD --> T9

    SS --> C10
    SS --> T1

    AC --> C11
    AC --> T1

    FD --> C12
    FD --> T6

    NT --> C13
    NT --> T1

    PA --> C14
    PA --> T1

    RA --> C15
    RA --> T1
    RA --> T6

    style Agents fill:#fff8e1
    style Capabilities fill:#e8f5e9
    style Tools fill:#e3f2fd
```

---

## 10. FRONTEND COMPONENT HIERARCHY

> Shows the Next.js application structure and component organization.

```mermaid
graph TB
    subgraph App["📱 NEXT.JS APP (dashboard/app/)"]
        Layout["layout.tsx<br/>Root Layout + Providers"]

        subgraph Pages["Pages"]
            Home["page.tsx<br/>Landing"]
            Dashboard["dashboard/page.tsx"]
            Projects["projects/page.tsx"]
            ProjectDetail["projects/[id]/page.tsx"]
            Tests["tests/page.tsx"]
            Discovery["discovery/page.tsx"]
            Visual["visual/page.tsx"]
            VisualDetail["visual/[comparisonId]/page.tsx"]
            Healing["healing/page.tsx"]
            Quality["quality/page.tsx"]
            Insights["insights/page.tsx"]
            Schedules["schedules/page.tsx"]
            Settings["settings/page.tsx"]
            Team["team/page.tsx"]
            Organizations["organizations/page.tsx"]
            Audit["audit/page.tsx"]
            Chat["chat/page.tsx"]
            Orchestrator["orchestrator/[sessionId]/page.tsx"]
            Profile["profile/page.tsx"]
            Notifications["notifications/page.tsx"]
            Onboarding["onboarding/page.tsx"]
            APIKeys["api-keys/page.tsx"]
        end
    end

    subgraph Components["🧩 COMPONENTS (dashboard/components/)"]
        UI["ui/<br/>shadcn/ui"]
        Layout2["layout/<br/>Sidebar, Header"]
        ProjectsComp["projects/<br/>ProjectCard, List"]
        TestsComp["tests/<br/>TestRunner, Results"]
        DiscoveryComp["discovery/<br/>FlowEditor, PageGraph"]
        VisualComp["visual/<br/>ComparisonViewer, Matrix"]
        ChatComp["chat/<br/>ChatInterface"]
        DashboardComp["dashboard/<br/>Metrics, Charts"]
        Shared["shared/<br/>LoadingStates, Errors"]
    end

    subgraph Lib["📚 LIB (dashboard/lib/)"]
        API["api.ts<br/>Org-scoped Fetch"]
        AuthAPI["auth-api.ts<br/>Authenticated Fetch"]
        Utils["utils.ts"]
        Providers["providers.tsx<br/>Clerk, Themes"]
    end

    subgraph Stores["🗄️ STORES (dashboard/stores/)"]
        OrgStore["Organization Store"]
        ProjectStore["Project Store"]
        NotifStore["Notification Store"]
    end

    Layout --> Pages
    Pages --> Components
    Components --> UI
    Pages --> Lib
    Pages --> Stores

    style App fill:#e8f5e9
    style Components fill:#e3f2fd
    style Lib fill:#fff8e1
    style Stores fill:#fce4ec
```

---

## 11. AUTHENTICATION & AUTHORIZATION FLOW

> Shows the complete auth flow from login to authorized API access.

```mermaid
sequenceDiagram
    participant U as User
    participant D as Dashboard
    participant CL as Clerk Auth
    participant API as FastAPI
    participant RB as RBAC Middleware
    participant DB as Supabase

    U->>D: Access /dashboard
    D->>CL: Check session
    CL-->>D: No session
    D->>U: Redirect to /sign-in

    U->>CL: Sign in with email/OAuth
    CL->>CL: Verify credentials
    CL-->>U: JWT Token + Session

    U->>D: Access /projects
    D->>D: getToken() from Clerk
    D->>API: GET /projects<br/>Authorization: Bearer {JWT}<br/>X-Organization-ID: {org_id}

    API->>API: Verify JWT signature
    API->>RB: Check permissions
    RB->>DB: SELECT role FROM organization_members<br/>WHERE user_id = ?
    DB-->>RB: role = 'admin'
    RB->>RB: Check role >= required
    RB-->>API: Authorized

    API->>DB: SELECT * FROM projects<br/>WHERE organization_id = ?
    DB-->>API: Projects list
    API-->>D: 200 OK + projects
    D-->>U: Display projects

    Note over U,DB: Organization-scoped multi-tenancy
```

---

## 12. SELF-HEALING INTELLIGENCE FLOW

> Shows how the self-healing system learns from failures and fixes tests automatically.

```mermaid
flowchart TB
    subgraph Failure["❌ Test Failure Detected"]
        F1["UI Test Fails"]
        F2["Selector Not Found"]
        F3["Element Changed"]
    end

    subgraph Analysis["🔍 Failure Analysis"]
        A1["Extract Error Message"]
        A2["Get Screenshot"]
        A3["Capture DOM State"]
        A4["Get Git Context"]
    end

    subgraph Memory["🧠 Memory Search (pgvector)"]
        M1["Generate Error Embedding"]
        M2["search_similar_failures()"]
        M3["Retrieve Past Solutions"]
        M4["Calculate Success Rates"]
    end

    subgraph Healing["💊 Healing Strategy"]
        H1["Apply Best Past Solution"]
        H2["Generate New Selector"]
        H3["Use Claude for Analysis"]
        H4["Try Alternative Strategies"]
    end

    subgraph Validation["✅ Validation"]
        V1["Re-run Failed Test"]
        V2["Verify Fix Works"]
        V3["Update Success Count"]
    end

    subgraph Learning["📚 Learning Update"]
        L1["Store New Pattern"]
        L2["Update Embeddings"]
        L3["Increment Stats"]
        L4["Cross-project Learning"]
    end

    F1 --> A1
    F2 --> A1
    F3 --> A1

    A1 --> A2
    A2 --> A3
    A3 --> A4

    A4 --> M1
    M1 --> M2
    M2 --> M3
    M3 --> M4

    M4 -->|"High Success Rate"| H1
    M4 -->|"Low/No Match"| H2
    H2 --> H3
    H3 --> H4

    H1 --> V1
    H4 --> V1
    V1 --> V2

    V2 -->|"Success"| V3
    V3 --> L1
    L1 --> L2
    L2 --> L3
    L3 --> L4

    V2 -->|"Failure"| H4

    style Failure fill:#ffcdd2
    style Analysis fill:#fff9c4
    style Memory fill:#e1bee7
    style Healing fill:#c8e6c9
    style Validation fill:#bbdefb
    style Learning fill:#b2dfdb
```

---

## Diagram Types & Terminology

| Diagram Type | Purpose | What It Shows | When to Use |
|-------------|---------|---------------|-------------|
| **C4 Model** | Multi-level architecture | System → Container → Component → Code | Stakeholder presentations, documentation |
| **Sequence Diagram** | Runtime behavior | UI → API → DB call flow | Understanding request/response flows |
| **Entity-Relationship Diagram (ERD)** | Data modeling | Tables, columns, relationships | Database design, data governance |
| **Component Diagram** | Static structure | Modules and dependencies | Understanding code organization |
| **Data Flow Diagram (DFD)** | Data movement | How data transforms through system | Security analysis, GDPR compliance |
| **Dependency Graph** | Module coupling | Import/dependency relationships | Refactoring, build optimization |
| **State Machine Diagram** | Process flows | States and transitions | Workflow engines like LangGraph |
| **Deployment Diagram** | Infrastructure | Servers, containers, networks | DevOps, cloud architecture |

---

## Quick Reference: Platform Summary

| Layer | Technologies | Key Files |
|-------|-------------|-----------|
| **Frontend** | Next.js 15, React, TypeScript, Tailwind, shadcn/ui | `dashboard/app/`, `dashboard/components/` |
| **Backend API** | FastAPI, Python 3.14, Pydantic | `src/api/server.py`, `src/api/*.py` |
| **Orchestration** | LangGraph, PostgresSaver, pgvector | `src/orchestrator/*.py` |
| **AI Agents** | Claude Sonnet/Opus/Haiku, 15+ specialized agents | `src/agents/*.py` |
| **Computer Use** | Claude Computer Use API, Playwright hybrid | `src/computer_use/*.py` |
| **Visual AI** | Semantic, Structural, Perceptual analyzers | `src/visual_ai/*.py` |
| **Discovery** | Crawlee, Page Graph, Flow Detection | `src/discovery/*.py` |
| **Database** | Supabase PostgreSQL, pgvector, 25+ tables | `supabase/migrations/*.sql` |
| **Auth** | Clerk, JWT, RBAC | `src/api/security/*.py` |
| **Integrations** | GitHub, Slack, Stripe, Resend | `src/integrations/*.py` |

---

*Generated on: 2026-01-12*
*Last updated: Auto-generated from codebase analysis*
