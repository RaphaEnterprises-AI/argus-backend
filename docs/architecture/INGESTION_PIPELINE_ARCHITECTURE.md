# Argus Scalable Data Ingestion Pipeline Architecture

## Executive Summary

This document defines the architecture for Argus's real-time, multi-tenant data ingestion pipeline, synthesizing best practices from industry leaders (Netflix, Uber) and modern AI frameworks (Cognee, LlamaIndex) to create a production-grade system that powers our AI-driven quality intelligence platform.

## Research Foundation

### Industry Benchmarks
- **Netflix Keystone**: Processes 700B+ messages/day across 36 Kafka clusters
- **Uber Streaming**: Apache Flink for real-time feature engineering
- **Cognee**: 1GB processed in ~40 minutes with 100+ containers, graph+vector hybrid

### Key Principles Adopted
1. **Event-Driven First**: All data flows as events through Redpanda (Kafka-compatible)
2. **Shift-Left Processing**: Enrich and transform data early in the pipeline
3. **Graph + Vector Hybrid**: Semantic understanding + symbolic reasoning
4. **Multi-Tenant Isolation**: Clean data boundaries per organization
5. **Incremental Processing**: Update knowledge without full rebuilds

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           ARGUS INGESTION PIPELINE                              │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                    LAYER 1: DATA SOURCES (Producers)                      │  │
│  ├──────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                            │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │  │
│  │   │   GitHub    │  │   Sentry    │  │  Dashboard  │  │   Webhooks  │     │  │
│  │   │  Webhooks   │  │   Events    │  │   Actions   │  │    (n8n)    │     │  │
│  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │  │
│  │          │                │                │                │             │  │
│  └──────────┼────────────────┼────────────────┼────────────────┼─────────────┘  │
│             │                │                │                │                 │
│             ▼                ▼                ▼                ▼                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                    LAYER 2: EVENT GATEWAY (Ingestion)                     │  │
│  ├──────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                            │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐    │  │
│  │   │                    Event Gateway (FastAPI)                       │    │  │
│  │   │  • Schema validation (Pydantic)                                  │    │  │
│  │   │  • Tenant extraction (org_id, project_id)                        │    │  │
│  │   │  • Event enrichment (timestamps, request_id, user_id)            │    │  │
│  │   │  • Deduplication (idempotency keys)                              │    │  │
│  │   │  • Rate limiting per tenant                                      │    │  │
│  │   └──────────────────────────────────┬──────────────────────────────┘    │  │
│  │                                       │                                    │  │
│  └───────────────────────────────────────┼────────────────────────────────────┘  │
│                                          │                                       │
│                                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                    LAYER 3: MESSAGE BUS (Transport)                       │  │
│  ├──────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                            │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐    │  │
│  │   │                     Redpanda Cluster                             │    │  │
│  │   │                                                                   │    │  │
│  │   │  Topics:                                                          │    │  │
│  │   │  ├── argus.codebase.ingested    (PR opened, repo connected)      │    │  │
│  │   │  ├── argus.codebase.analyzed    (Analysis complete)              │    │  │
│  │   │  ├── argus.test.created         (Test generated)                 │    │  │
│  │   │  ├── argus.test.executed        (Test run complete)              │    │  │
│  │   │  ├── argus.test.failed          (Test failure detected)          │    │  │
│  │   │  ├── argus.error.reported       (Production error from Sentry)   │    │  │
│  │   │  ├── argus.healing.requested    (Self-heal trigger)              │    │  │
│  │   │  ├── argus.healing.completed    (Heal successful)                │    │  │
│  │   │  ├── argus.insight.generated    (AI insight ready)               │    │  │
│  │   │  └── argus.dlq                  (Dead letter queue)              │    │  │
│  │   │                                                                   │    │  │
│  │   │  Partitioning: By org_id (tenant isolation)                       │    │  │
│  │   │  Retention: 7 days (configurable per topic)                       │    │  │
│  │   │  Replication: 3x for durability                                   │    │  │
│  │   └─────────────────────────────────────────────────────────────────┘    │  │
│  │                                                                            │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                       │
│         ┌────────────────────────────────┼────────────────────────────────┐     │
│         │                                │                                │     │
│         ▼                                ▼                                ▼     │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  ┌─────────────────┐ │
│  │ LAYER 4A: COGNEE│  │ LAYER 4B: REAL-TIME PROCESSORS  │  │ LAYER 4C: STORE │ │
│  │ (Knowledge Eng) │  │    (Flink/Python Workers)       │  │  (Persistence)  │ │
│  ├─────────────────┤  ├─────────────────────────────────┤  ├─────────────────┤ │
│  │                 │  │                                 │  │                 │ │
│  │ ECL Pipeline:   │  │ ┌─────────────────────────────┐ │  │ ┌─────────────┐ │ │
│  │                 │  │ │ Stream Processors           │ │  │ │  Supabase   │ │ │
│  │ ┌─────────────┐ │  │ │ • Aggregation (counts,     │ │  │ │  (OLTP)     │ │ │
│  │ │   EXTRACT   │ │  │ │   rates, trends)           │ │  │ │             │ │ │
│  │ │ • Parse code│ │  │ │ • Windowed analytics       │ │  │ │ • Tests     │ │ │
│  │ │ • Chunk docs│ │  │ │ • Pattern detection        │ │  │ │ • Runs      │ │ │
│  │ │ • Extract   │ │  │ │ • Anomaly flagging         │ │  │ │ • Projects  │ │ │
│  │ │   entities  │ │  │ └─────────────────────────────┘ │  │ │ • Orgs      │ │ │
│  │ └──────┬──────┘ │  │                                 │  │ └──────┬──────┘ │ │
│  │        │        │  │ ┌─────────────────────────────┐ │  │        │        │ │
│  │        ▼        │  │ │ AI Enrichment Workers      │ │  │        │        │ │
│  │ ┌─────────────┐ │  │ │ • LLM summarization        │ │  │ ┌─────────────┐ │ │
│  │ │   COGNIFY   │ │  │ │ • Classification           │ │  │ │  pgvector   │ │ │
│  │ │ • Generate  │ │  │ │ • Embedding generation     │ │  │ │ (Semantic)  │ │ │
│  │ │   embeddings│ │  │ │ • Insight extraction       │ │  │ │             │ │ │
│  │ │ • Build     │ │  │ └─────────────────────────────┘ │  │ │ • Failure   │ │ │
│  │ │   relations │ │  │                                 │  │ │   patterns  │ │ │
│  │ │ • Memify    │ │  │ ┌─────────────────────────────┐ │  │ │ • Code      │ │ │
│  │ │   (refresh) │ │  │ │ Routing & Orchestration    │ │  │ │   semantics │ │ │
│  │ └──────┬──────┘ │  │ │ • Event routing rules      │ │  │ └─────────────┘ │ │
│  │        │        │  │ │ • Fan-out to consumers     │ │  │                 │ │
│  │        ▼        │  │ │ • DLQ handling             │ │  │ ┌─────────────┐ │ │
│  │ ┌─────────────┐ │  │ │ • Retry policies           │ │  │ │Neo4j / Aura │ │ │
│  │ │    LOAD     │ │  │ └─────────────────────────────┘ │  │ │(Knowledge   │ │ │
│  │ │ • Store to  │ │  │                                 │  │ │ Graph)      │ │ │
│  │ │   Neo4j     │ │  └─────────────────────────────────┘  │ │             │ │ │
│  │ │ • Index to  │ │                                       │ │ • Codebase  │ │ │
│  │ │   pgvector  │ │                                       │ │   structure │ │ │
│  │ │ • Sync to   │ │                                       │ │ • Test deps │ │ │
│  │ │   Supabase  │ │                                       │ │ • Failure   │ │ │
│  │ └─────────────┘ │                                       │ │   causality │ │ │
│  │                 │                                       │ └─────────────┘ │ │
│  └─────────────────┘                                       └─────────────────┘ │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                    LAYER 5: SERVING (Retrieval)                           │  │
│  ├──────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                            │  │
│  │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │  │
│  │   │   Graph Query   │  │  Vector Search  │  │  Hybrid Search  │          │  │
│  │   │   (Cypher)      │  │  (pgvector)     │  │  (Combined)     │          │  │
│  │   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │  │
│  │            │                    │                    │                    │  │
│  │            └────────────────────┼────────────────────┘                    │  │
│  │                                 │                                          │  │
│  │                                 ▼                                          │  │
│  │            ┌─────────────────────────────────────────┐                    │  │
│  │            │          Argus Brain (LangGraph)        │                    │  │
│  │            │  • Context assembly                     │                    │  │
│  │            │  • Fact grounding                       │                    │  │
│  │            │  • Response generation                  │                    │  │
│  │            └─────────────────────────────────────────┘                    │  │
│  │                                                                            │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Details

### Layer 1: Data Sources

| Source | Event Type | Trigger | Data |
|--------|------------|---------|------|
| **GitHub Webhooks** | `codebase.ingested` | PR opened/updated | Diff, files, metadata |
| **Sentry Integration** | `error.reported` | New error/exception | Stack trace, context |
| **Dashboard Actions** | `test.created` | User creates test | Test spec, project |
| **n8n Workflows** | Various | Custom automations | Configurable payload |
| **Browser Extension** | `recording.captured` | User records flow | DOM events, screenshots |

### Layer 2: Event Gateway

The Event Gateway is the single entry point for all events, ensuring:

```python
# src/events/gateway.py (simplified)
class EventGateway:
    async def ingest(self, event: RawEvent) -> ProcessedEvent:
        # 1. Validate schema
        validated = self.schema_registry.validate(event)

        # 2. Extract tenant context
        tenant = TenantContext(
            org_id=event.org_id,
            project_id=event.project_id,
            user_id=event.user_id
        )

        # 3. Enrich with metadata
        enriched = EnrichedEvent(
            **validated,
            event_id=uuid4(),
            timestamp=datetime.utcnow(),
            request_id=correlation_id(),
            source_ip=request.client.host,
            tenant=tenant
        )

        # 4. Deduplication check
        if await self.dedupe_cache.exists(enriched.idempotency_key):
            return DuplicateEvent(original_id=cached_id)

        # 5. Rate limiting
        if not await self.rate_limiter.allow(tenant.org_id):
            raise RateLimitExceeded(tenant.org_id)

        # 6. Publish to Redpanda
        await self.producer.send(
            topic=self.route_event(enriched),
            key=tenant.org_id,  # Partition by org
            value=enriched.serialize()
        )

        return enriched
```

### Layer 3: Message Bus (Redpanda)

**Topic Design:**

```yaml
# Topic naming convention: argus.<domain>.<action>
topics:
  # Codebase events
  argus.codebase.ingested:
    partitions: 12
    retention: 7d
    compaction: false

  argus.codebase.analyzed:
    partitions: 12
    retention: 30d
    compaction: true  # Keep latest analysis per file

  # Test events
  argus.test.created:
    partitions: 6
    retention: 7d

  argus.test.executed:
    partitions: 12
    retention: 30d

  argus.test.failed:
    partitions: 6
    retention: 90d  # Keep failures longer for learning

  # Error events
  argus.error.reported:
    partitions: 6
    retention: 30d

  # Healing events
  argus.healing.requested:
    partitions: 3
    retention: 7d

  argus.healing.completed:
    partitions: 3
    retention: 30d

  # Dead letter queue
  argus.dlq:
    partitions: 3
    retention: 30d

  # AI insights
  argus.insight.generated:
    partitions: 6
    retention: 90d
```

**Partitioning Strategy:**

```python
# Partition by org_id ensures:
# 1. Tenant isolation (events from same org go to same partition)
# 2. Ordering guarantee (events processed in order per tenant)
# 3. Parallel processing (different tenants processed concurrently)

def partition_key(event: Event) -> str:
    return f"{event.org_id}"

# For high-volume tenants, sub-partition by project:
def partition_key_granular(event: Event) -> str:
    return f"{event.org_id}:{event.project_id}"
```

### Layer 4A: Cognee Knowledge Engineering

The ECL (Extract, Cognify, Load) pipeline transforms raw events into knowledge:

```python
# data-layer/cognee-worker/src/worker.py (conceptual)
class CogneeWorker:
    async def process_codebase_event(self, event: CodebaseEvent):
        # EXTRACT: Parse and chunk code
        chunks = await cognee.extract(
            data=event.code_content,
            content_type="code",
            metadata={
                "org_id": event.org_id,
                "project_id": event.project_id,
                "file_path": event.file_path,
                "language": event.language
            }
        )

        # COGNIFY: Generate embeddings and build relations
        await cognee.cognify(
            chunks,
            pipeline=[
                EntityExtraction(),      # Extract functions, classes, imports
                EmbeddingGeneration(),   # Generate semantic embeddings
                RelationshipBuilder(),   # Build call graph, dependencies
                SummaryGeneration(),     # LLM-generated summaries
            ],
            namespace=f"org:{event.org_id}:project:{event.project_id}"
        )

        # LOAD: Persist to knowledge stores
        await cognee.load(
            target_stores=[
                Neo4jStore(graph_name=f"argus_{event.org_id}"),
                PgVectorStore(namespace=event.org_id),
            ]
        )
```

**Multi-Tenant Isolation:**

```python
# Each organization gets isolated:
# 1. Neo4j graph namespace (separate subgraph)
# 2. pgvector namespace (filtered by org_id)
# 3. Supabase RLS (row-level security)

class TenantIsolation:
    def get_graph_namespace(self, org_id: str) -> str:
        return f"argus_{org_id}"

    def get_vector_filter(self, org_id: str) -> dict:
        return {"org_id": {"$eq": org_id}}

    def get_rls_policy(self, org_id: str) -> str:
        return f"organization_id = '{org_id}'"
```

### Layer 4B: Real-Time Processors

Stream processors for aggregation and analytics:

```python
# Windowed aggregation example
class TestMetricsAggregator:
    async def process(self, events: List[TestExecutedEvent]):
        # 5-minute tumbling window
        window = TumblingWindow(duration=timedelta(minutes=5))

        for event in events:
            window.add(event)

        if window.is_complete():
            metrics = TestMetrics(
                org_id=window.key,
                window_start=window.start,
                window_end=window.end,
                total_tests=len(window.events),
                passed=sum(1 for e in window.events if e.status == "passed"),
                failed=sum(1 for e in window.events if e.status == "failed"),
                flaky=sum(1 for e in window.events if e.is_flaky),
                avg_duration_ms=mean(e.duration_ms for e in window.events),
            )

            await self.metrics_store.upsert(metrics)
            await self.publisher.send("argus.insight.generated", metrics)
```

### Layer 4C: Persistence Stores

| Store | Purpose | Data Type | Query Pattern |
|-------|---------|-----------|---------------|
| **Supabase** | Operational data | Tests, runs, projects | CRUD, real-time subscriptions |
| **pgvector** | Semantic search | Embeddings, failure patterns | Similarity search |
| **Neo4j Aura** | Knowledge graph | Code structure, dependencies | Graph traversal, Cypher |
| **Valkey** | Caching | Hot data, sessions | Key-value, TTL |
| **Cloudflare R2** | Artifacts | Screenshots, logs, diffs | Object storage |

### Layer 5: Serving (Retrieval)

Hybrid retrieval combining graph and vector:

```python
class HybridRetriever:
    async def retrieve(
        self,
        query: str,
        org_id: str,
        project_id: Optional[str] = None
    ) -> RetrievalResult:
        # 1. Generate query embedding
        query_embedding = await self.embedder.embed(query)

        # 2. Vector similarity search
        vector_results = await self.pgvector.search(
            embedding=query_embedding,
            filter={"org_id": org_id, "project_id": project_id},
            limit=20
        )

        # 3. Graph traversal for related entities
        graph_results = await self.neo4j.query(f"""
            MATCH (n:Entity {{org_id: $org_id}})
            WHERE n.name CONTAINS $search_term
            OPTIONAL MATCH (n)-[r]-(related)
            RETURN n, r, related
            LIMIT 50
        """, org_id=org_id, search_term=extract_key_terms(query))

        # 4. Combine and re-rank
        combined = self.reranker.rerank(
            query=query,
            candidates=[
                *vector_results,
                *self.graph_to_candidates(graph_results)
            ]
        )

        return RetrievalResult(
            documents=combined[:10],
            graph_context=graph_results,
            confidence_scores=self.compute_confidences(combined)
        )
```

---

## Event Schema Design

### Base Event Schema

```python
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

class BaseEvent(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0"

    # Tenant context
    org_id: UUID
    project_id: Optional[UUID] = None
    user_id: Optional[UUID] = None

    # Correlation
    correlation_id: Optional[UUID] = None
    causation_id: Optional[UUID] = None

    # Metadata
    source: str  # e.g., "github-webhook", "dashboard", "sentry"
    idempotency_key: Optional[str] = None

class CodebaseIngestedEvent(BaseEvent):
    event_type: str = "codebase.ingested"

    repo_url: str
    branch: str
    commit_sha: str
    pr_number: Optional[int] = None

    files_changed: List[FileChange]
    diff_summary: str

class TestExecutedEvent(BaseEvent):
    event_type: str = "test.executed"

    test_id: UUID
    run_id: UUID
    status: Literal["passed", "failed", "skipped", "error"]
    duration_ms: int

    # Failure details
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    screenshot_url: Optional[str] = None

    # Flakiness detection
    is_flaky: bool = False
    flaky_confidence: float = 0.0

class ErrorReportedEvent(BaseEvent):
    event_type: str = "error.reported"

    error_id: str  # Sentry issue ID
    title: str
    culprit: str

    exception: ExceptionInfo
    contexts: dict
    tags: dict

    # Correlation
    related_test_ids: List[UUID] = []
    affected_files: List[str] = []
```

---

## Scaling Considerations

### Horizontal Scaling

```yaml
# Kubernetes HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cognee-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cognee-worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: External
      external:
        metric:
          name: kafka_consumer_lag
        target:
          type: AverageValue
          averageValue: "1000"
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Backpressure Handling

```python
class BackpressureHandler:
    def __init__(self, max_queue_size: int = 10000):
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )

    async def handle_event(self, event: Event):
        if self.queue.full():
            # Shed load to DLQ
            await self.dlq_producer.send(event)
            metrics.increment("events_shed")
            return

        if self.circuit_breaker.is_open:
            await self.dlq_producer.send(event)
            return

        try:
            await self.queue.put(event)
        except Exception as e:
            self.circuit_breaker.record_failure()
            await self.dlq_producer.send(event)
```

### Cost Optimization

| Component | Strategy | Expected Savings |
|-----------|----------|------------------|
| **LLM Calls** | Batch similar requests, cache embeddings | 40-60% |
| **Vector DB** | Quantization (int8), dimension reduction | 30-50% storage |
| **Graph DB** | Prune old relationships, archive cold data | 20-30% |
| **Compute** | Spot instances for batch, reserved for real-time | 50-70% |

---

## Monitoring & Observability

### Key Metrics

```python
# Prometheus metrics
EVENT_INGESTED = Counter(
    "argus_events_ingested_total",
    "Total events ingested",
    ["event_type", "org_id", "source"]
)

EVENT_PROCESSING_TIME = Histogram(
    "argus_event_processing_seconds",
    "Time to process event",
    ["event_type", "processor"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)

CONSUMER_LAG = Gauge(
    "argus_kafka_consumer_lag",
    "Kafka consumer lag",
    ["topic", "partition", "consumer_group"]
)

KNOWLEDGE_GRAPH_SIZE = Gauge(
    "argus_knowledge_graph_nodes",
    "Number of nodes in knowledge graph",
    ["org_id", "node_type"]
)
```

### Health Checks

```python
@app.get("/health/ingestion")
async def ingestion_health():
    checks = {
        "redpanda": await check_redpanda_connection(),
        "neo4j": await check_neo4j_connection(),
        "pgvector": await check_pgvector_connection(),
        "cognee": await check_cognee_worker_health(),
    }

    all_healthy = all(c["status"] == "healthy" for c in checks.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
```

---

## Implementation Phases

### Phase 1: Foundation (RAP-120)
- [ ] Event Gateway with schema validation
- [ ] Redpanda topic creation and configuration
- [ ] Basic producer/consumer infrastructure
- [ ] Multi-tenant partitioning

### Phase 2: Knowledge Engineering
- [ ] Cognee worker deployment
- [ ] ECL pipeline for codebase events
- [ ] Neo4j graph population
- [ ] pgvector embedding storage

### Phase 3: Real-Time Processing
- [ ] Stream aggregators (metrics, windows)
- [ ] Anomaly detection
- [ ] Pattern recognition

### Phase 4: Advanced Features
- [ ] Hybrid retrieval (graph + vector)
- [ ] Cross-tenant learning (anonymized)
- [ ] Predictive analytics

---

## References

- [Building Scalable Production-Grade Agentic RAG](https://levelup.gitconnected.com/building-a-scalable-production-grade-agentic-rag-pipeline-1168dcd36260)
- [Redpanda Real-Time Data Ingestion Guide](https://www.redpanda.com/guides/fundamentals-of-data-engineering-real-time-data-ingestion)
- [Neo4j RAG Tutorial](https://neo4j.com/blog/developer/rag-tutorial/)
- [Cognee Knowledge Graph Architecture](https://github.com/topoteretes/cognee)
- [Netflix Data Streaming Architecture](https://netflixtechblog.com/building-and-scaling-data-lineage-at-netflix-to-improve-data-infrastructure-reliability-and-1a52526a7977)
- [Uber Streaming Pipelines](https://www.uber.com/blog/building-scalable-streaming-pipelines/)
- [Data Streaming Landscape 2026](https://www.kai-waehner.de/blog/2025/12/05/the-data-streaming-landscape-2026/)
- [AWS RAG Data Ingestion at Scale](https://aws.amazon.com/blogs/big-data/build-a-rag-data-ingestion-pipeline-for-large-scale-ml-workloads/)
