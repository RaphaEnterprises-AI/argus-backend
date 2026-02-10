# Skopaq Architecture - Mermaid Diagrams

## High-Level Platform Architecture

```mermaid
flowchart TB
    subgraph UX["USER EXPERIENCE PLANE"]
        Dashboard["Dashboard<br/>(Next.js)"]
        AgentBuilder["Agent Builder<br/>(29 Agents)"]
        DevEx["DevEx<br/>(MCP, SDK, API)"]
    end

    subgraph Trust["TRUST BOUNDARY"]
        Auth["JWT + API Key"]
        RateLimit["Rate Limits<br/>(30-2000/min)"]
        Guardrails["5-Layer<br/>Guardrails"]
        RBAC["RBAC<br/>(9 roles)"]
    end

    subgraph Control["CONTROL PLANE"]
        Orchestrator["LangGraph<br/>Orchestrator"]
        AgentRegistry["Agent Registry<br/>(41 capabilities)"]
        A2A["A2A Protocol<br/>(Kafka mesh)"]
        Router["Model Router<br/>(300+ models)"]
        Evals["Agent-as-Judge<br/>+ MetaJudge"]
    end

    subgraph Runtime["RUNTIME PLANE"]
        Executor["Parallel<br/>Executor"]
        Sandbox["Docker<br/>Sandbox"]
        BrowserPool["Browser Pool<br/>(Selenium)"]
        Gateway["Event<br/>Gateway"]
    end

    subgraph Memory["CONTEXT & MEMORY PLANE"]
        Cache["Valkey<br/>Cache"]
        Checkpoints["PostgreSQL<br/>Checkpoints"]
        Cognee["Cognee<br/>Vector Search"]
        EpisodicMem["Episodic +<br/>Procedural Memory"]
    end

    subgraph Data["DATA PLANE"]
        Kafka["Redpanda/Kafka<br/>(12 topics)"]
        Supabase["Supabase<br/>PostgreSQL"]
        R2["Cloudflare R2<br/>(Screenshots)"]
        Flink["Apache Flink<br/>(Streaming)"]
        Neo4j["Neo4j Aura<br/>(Knowledge Graph)"]
    end

    subgraph Observe["OBSERVABILITY PLANE"]
        LangFuse["LangFuse<br/>Traces"]
        Prometheus["Prometheus<br/>Metrics"]
        Alerts["36+ Alerts<br/>(Alertmanager)"]
        Audit["Audit Logs<br/>(365 days)"]
    end

    subgraph Security["SECURITY PLANE"]
        SSO["Clerk SSO<br/>/OIDC"]
        KMS["Cloudflare<br/>Key Vault"]
        RLS["Row-Level<br/>Security"]
        Encryption["AES-256<br/>Encryption"]
    end

    UX --> Trust
    Trust --> Control
    Control --> Runtime
    Runtime --> Memory
    Memory --> Data

    Control -.-> Observe
    Runtime -.-> Observe
    Data -.-> Observe

    Trust -.-> Security
    Data -.-> Security
```

## Agent Mesh Architecture (A2A)

```mermaid
flowchart LR
    subgraph Supervisor["SUPERVISOR"]
        LangGraph["LangGraph<br/>State Machine"]
    end

    subgraph Agents["29 AI AGENTS"]
        CodeAnalyzer["Code<br/>Analyzer"]
        UITester["UI<br/>Tester"]
        APITester["API<br/>Tester"]
        SelfHealer["Self<br/>Healer"]
        SREAgent["SRE<br/>Agent"]
        CRAG["Corrective<br/>RAG"]
        ToolDiscovery["Tool<br/>Discovery"]
        AgentJudge["Agent<br/>Judge"]
    end

    subgraph Registry["AGENT REGISTRY"]
        Capabilities["41 Capabilities"]
        Health["Health Monitor<br/>(60s heartbeat)"]
    end

    subgraph Kafka["KAFKA MESH"]
        Request["argus.agent.request"]
        Response["argus.agent.response"]
        Broadcast["argus.agent.broadcast"]
        Heartbeat["argus.agent.heartbeat"]
    end

    subgraph Consensus["MARP CONSENSUS"]
        Propose["Propose"]
        Debate["Debate"]
        Resolve["Resolve"]
    end

    LangGraph --> Agents
    Agents <--> Kafka
    Kafka <--> Registry
    Agents --> Consensus
    Consensus --> LangGraph
```

## Unified Instant Intelligence Layer (UIIL)

```mermaid
flowchart LR
    Query["Query"] --> T1

    subgraph UIIL["3-TIER RESOLUTION"]
        T1["Tier 1: AI Gateway<br/>~5ms, 30-40% hit"]
        T2["Tier 2: Valkey Cache<br/>~2ms, 20-30% hit"]
        T3["Tier 3: Cognee Vector<br/>~30ms, 25-35% hit"]
        T4["Tier 4: LLM Fallback<br/>~2000ms, 5-15%"]
    end

    T1 -->|miss| T2
    T2 -->|miss| T3
    T3 -->|low confidence| T4

    T1 -->|hit| Response
    T2 -->|hit| Response
    T3 -->|hit| Response
    T4 --> Response

    Response["Response<br/><100ms target"]
```

## Data Flow Architecture

```mermaid
flowchart TB
    subgraph Ingest["INGESTION"]
        GitHub["GitHub<br/>Webhooks"]
        GitLab["GitLab<br/>Webhooks"]
        API["REST API"]
        MCP["MCP<br/>Connections"]
    end

    subgraph Process["PROCESSING"]
        Kafka["Redpanda<br/>(12 topics)"]
        Flink["Apache Flink<br/>(Streaming)"]
        Cognee["Cognee ECL<br/>Pipeline"]
    end

    subgraph Store["STORAGE"]
        Supabase["Supabase<br/>(PostgreSQL)"]
        Neo4j["Neo4j Aura<br/>(Graph)"]
        R2["Cloudflare R2<br/>(Blobs)"]
        Valkey["Valkey<br/>(Cache)"]
    end

    subgraph Serve["SERVING"]
        Backend["FastAPI<br/>Backend"]
        Dashboard["Next.js<br/>Dashboard"]
        SSE["SSE<br/>Streaming"]
    end

    Ingest --> Kafka
    Kafka --> Flink
    Kafka --> Cognee
    Flink --> Supabase
    Cognee --> Neo4j
    Cognee --> Supabase

    Store --> Backend
    Backend --> Dashboard
    Backend --> SSE
```

## Security Architecture

```mermaid
flowchart TB
    subgraph External["EXTERNAL"]
        User["User"]
        Service["Service"]
    end

    subgraph Auth["AUTHENTICATION"]
        Clerk["Clerk SSO"]
        JWT["JWT Token"]
        APIKey["API Key"]
    end

    subgraph Authz["AUTHORIZATION"]
        RBAC["9-Role RBAC"]
        RLS["Row-Level Security"]
        Permissions["30+ Permissions"]
    end

    subgraph Protection["PROTECTION"]
        RateLimit["Rate Limits"]
        Guardrails["5-Layer Guardrails"]
        Sandbox["Docker Sandbox"]
        Validation["Schema Validation"]
    end

    subgraph Secrets["SECRETS"]
        KMS["Cloudflare KMS"]
        BYOK["Zero-Knowledge BYOK"]
        DEK["DEK Versioning"]
    end

    User --> Clerk
    Service --> APIKey
    Clerk --> JWT
    JWT --> RBAC
    APIKey --> RBAC
    RBAC --> RLS
    RLS --> Permissions
    Permissions --> Protection
    Protection --> Secrets
```

## Observability Stack

```mermaid
flowchart LR
    subgraph Sources["SOURCES"]
        App["FastAPI App"]
        Agents["AI Agents"]
        K8s["Kubernetes"]
        Data["Data Layer"]
    end

    subgraph Collection["COLLECTION"]
        LangFuse["LangFuse<br/>(Traces)"]
        Prometheus["Prometheus<br/>(Metrics)"]
        Logs["structlog<br/>(Logs)"]
    end

    subgraph Alerting["ALERTING"]
        Rules["36+ Alert Rules"]
        AM["Alertmanager"]
        Routes["Severity Routing"]
    end

    subgraph Viz["VISUALIZATION"]
        Grafana["Grafana<br/>Dashboards"]
        Dashboard["Skopaq<br/>Dashboard"]
    end

    App --> LangFuse
    App --> Prometheus
    App --> Logs
    Agents --> LangFuse
    K8s --> Prometheus
    Data --> Prometheus

    Prometheus --> Rules
    Rules --> AM
    AM --> Routes

    LangFuse --> Dashboard
    Prometheus --> Grafana
    Grafana --> Dashboard
```

## Deployment Architecture

```mermaid
flowchart TB
    subgraph Cloud["CLOUD PROVIDERS"]
        subgraph Railway["Railway"]
            Backend["FastAPI Backend"]
            LangGraph["LangGraph Agents"]
        end

        subgraph Vercel["Vercel"]
            NextJS["Next.js Dashboard"]
        end

        subgraph Vultr["Vultr VKE"]
            Redpanda["Redpanda"]
            CogneeW["Cognee Worker"]
            FalkorDB["FalkorDB"]
            Valkey["Valkey"]
            Selenium["Selenium Grid"]
            Monitoring["Prometheus + Grafana"]
        end

        subgraph Cloudflare["Cloudflare"]
            R2["R2 Storage"]
            KV["Key Vault"]
            Gateway["AI Gateway"]
            Tunnel["Tunnel"]
            Workers["Workers"]
        end

        subgraph Supabase["Supabase"]
            PG["PostgreSQL"]
            Auth["Auth"]
            Realtime["Realtime"]
        end
    end

    Backend <--> PG
    Backend <--> Redpanda
    Backend <--> R2
    NextJS <--> Backend
    CogneeW <--> Redpanda
    Monitoring --> Tunnel
```
