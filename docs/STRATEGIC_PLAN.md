# Argus Strategic Plan 2025

## Executive Summary

Based on the architecture assessment, Argus requires a phased transformation to achieve enterprise-scale reliability. This plan outlines a 12-week roadmap divided into four phases, prioritizing stability, scalability, and intelligence.

**Current State**: MVP with 6/10 readiness
**Target State**: Production-ready for 1M+ users
**Investment Required**: ~8 engineering weeks (can parallelize)

---

## Phase 1: Foundation Hardening (Weeks 1-2)

### Goal
Eliminate technical debt and establish a stable foundation for scale.

### 1.1 Model Registry Completion

**Status**: 90% Complete

| Task | Files Affected | Effort |
|------|----------------|--------|
| Verify all agents use `get_model_id()` | `src/agents/*.py` | 2 hours |
| Add model health checks | `src/core/model_registry.py` | 4 hours |
| Implement fallback chain testing | `tests/unit/test_model_registry.py` | 4 hours |

**Success Criteria**: No hardcoded model IDs in codebase

### 1.2 SecretStr Standardization

**Status**: Complete

All API keys now use `hasattr(key, 'get_secret_value')` pattern.

### 1.3 Deprecation Cleanup

| Warning | Location | Fix |
|---------|----------|-----|
| `datetime.utcnow()` | Multiple files | Use `datetime.now(UTC)` |
| `websockets.server.serve` | `extension_bridge.py` | Update to new API |

**Estimated Effort**: 4 hours

### 1.4 Test Coverage Gap Analysis

| Module | Current | Target |
|--------|---------|--------|
| `src/core/` | 45% | 80% |
| `src/agents/` | 30% | 70% |
| `src/api/` | 25% | 60% |

**Action**: Add unit tests for critical paths

---

## Phase 2: Caching & Performance (Weeks 3-4)

### Goal
Reduce latency by 10x through strategic caching.

### 2.1 Redis Integration

```
┌─────────────────────────────────────────────────────────────┐
│                     CACHING ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Request   │───▶│ Redis Cache │───▶│  Database   │     │
│  │   Handler   │    │  (L1 Cache) │    │  (Source)   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                           │                                 │
│                     Cache Strategy:                         │
│                     • TTL: 5min for scores                  │
│                     • TTL: 1hr for reports                  │
│                     • TTL: 24hr for healing patterns        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Tasks**:

| Task | File | Priority |
|------|------|----------|
| Add Redis connection pool | `src/services/cache.py` | P0 |
| Cache quality scores | `src/api/quality.py` | P0 |
| Cache healing patterns | `src/agents/self_healer.py` | P1 |
| Cache model responses | `src/core/cognitive_engine.py` | P1 |

**Redis Key Schema**:
```
argus:score:{project_id}           → Quality score (TTL: 5min)
argus:report:{report_id}           → Generated report (TTL: 1hr)
argus:heal:{fingerprint}           → Healing pattern (TTL: 24hr)
argus:rate:{project}:{minute}      → Rate limit counter (TTL: 2min)
argus:llm:{prompt_hash}            → LLM response cache (TTL: 7d)
```

**Estimated Effort**: 3 days

### 2.2 Cloudflare KV for Edge Caching

Already configured in `cloudflare-worker/wrangler.toml`.

**Additional Keys**:
```
fingerprint:{hash}     → Deduplication (TTL: 24h)
session:{id}           → User session (TTL: 24h)
```

### 2.3 Database Query Optimization

| Query | Current Time | Target | Optimization |
|-------|--------------|--------|--------------|
| Get quality scores | 200ms | 20ms | Add composite index |
| List events by project | 500ms | 50ms | Pagination + index |
| Correlation lookup | 300ms | 30ms | Materialized view |

**Migration File**: `supabase/migrations/20260107_query_optimization.sql`

---

## Phase 3: Async Processing Pipeline (Weeks 5-8)

### Goal
Handle 10,000+ requests/second through event-driven architecture.

### 3.1 Message Queue Integration

**Option Analysis**:

| Queue | Pros | Cons | Recommendation |
|-------|------|------|----------------|
| Kafka | High throughput, replay | Complex setup | Production at scale |
| Redis Streams | Simple, fast | Limited retention | Good for MVP |
| Cloudflare Queues | Edge-native | Limited features | Already integrated |
| SQS | Managed, reliable | AWS lock-in | Alternative |

**Recommendation**: Start with Redis Streams, migrate to Kafka at scale.

### 3.2 Worker Pool Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ASYNC PROCESSING                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INGESTION                QUEUE                 WORKERS     │
│  ─────────                ─────                 ───────     │
│  ┌─────────┐    ┌───────────────────────┐    ┌─────────┐   │
│  │ Webhook │───▶│   Redis Stream:       │───▶│Normalizer│   │
│  │ Handler │    │   events:incoming     │    │ Worker  │   │
│  └─────────┘    └───────────────────────┘    └────┬────┘   │
│                                                    │        │
│                 ┌───────────────────────┐    ┌────▼────┐   │
│                 │   Redis Stream:       │◀───│Correlator│   │
│                 │   events:normalized   │    │ Worker  │   │
│                 └───────────────────────┘    └────┬────┘   │
│                                                    │        │
│                 ┌───────────────────────┐    ┌────▼────┐   │
│                 │   Redis Stream:       │◀───│ Analyzer │   │
│                 │   events:correlated   │    │ Worker  │   │
│                 └───────────────────────┘    └─────────┘   │
│                                                             │
│  Worker Config:                                             │
│  • Normalizer: 4 instances, CPU 0.5, Mem 256MB              │
│  • Correlator: 2 instances, CPU 1.0, Mem 512MB              │
│  • Analyzer:   2 instances, CPU 2.0, Mem 2GB                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Tasks**:

| Task | New Files | Effort |
|------|-----------|--------|
| Redis Streams wrapper | `src/services/queue.py` | 1 day |
| Normalizer worker | `src/workers/normalizer.py` | 2 days |
| Correlator worker | `src/workers/correlator.py` | 2 days |
| Analyzer worker | `src/workers/analyzer.py` | 3 days |
| Worker orchestration | `src/workers/manager.py` | 1 day |

### 3.3 Decoupled AI Processing

Current flow (blocking):
```
Request → Normalize → AI Analysis → Response (500ms-2s)
```

New flow (async):
```
Request → Normalize → Queue → Response (50ms)
                        ↓
              AI Analysis Worker (background)
                        ↓
              WebSocket/SSE push to client
```

**Benefits**:
- Response time: 50ms vs 500ms+
- Throughput: 10,000+ req/sec
- Cost: Batch AI calls for efficiency

---

## Phase 4: Intelligence Enhancement (Weeks 9-12)

### Goal
Transform from reactive to proactive testing intelligence.

### 4.1 Healing Knowledge Base

**Database Schema**:
```sql
CREATE TABLE healing_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fingerprint TEXT UNIQUE NOT NULL,
    original_selector TEXT NOT NULL,
    healed_selector TEXT NOT NULL,
    error_type TEXT NOT NULL,
    success_count INT DEFAULT 1,
    failure_count INT DEFAULT 0,
    last_used_at TIMESTAMPTZ DEFAULT now(),
    confidence NUMERIC GENERATED ALWAYS AS (
        success_count::numeric / GREATEST(success_count + failure_count, 1)
    ) STORED,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_healing_fingerprint ON healing_patterns(fingerprint);
CREATE INDEX idx_healing_confidence ON healing_patterns(confidence DESC);
```

**Healing Flow**:
```
┌─────────────────────────────────────────────────────────────┐
│                   HEALING KNOWLEDGE BASE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Test Fails with selector error                          │
│        ↓                                                    │
│  2. Check knowledge base for fingerprint                    │
│        ↓                                                    │
│  ┌────────────────────┬───────────────────────┐            │
│  │ FOUND (confidence) │ NOT FOUND             │            │
│  ├────────────────────┼───────────────────────┤            │
│  │ > 0.9: Apply fix   │ Call self-healer AI   │            │
│  │ 0.5-0.9: Try fix,  │ Store new pattern     │            │
│  │   validate         │                       │            │
│  │ < 0.5: AI healing  │                       │            │
│  └────────────────────┴───────────────────────┘            │
│        ↓                                                    │
│  3. Update pattern stats (success/failure)                  │
│        ↓                                                    │
│  4. Share pattern across similar tests                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Estimated Effort**: 1 week

### 4.2 Cross-Test Learning

**Feature**: When one test heals, apply to similar tests proactively.

**Implementation**:
```python
async def propagate_healing(pattern: HealingPattern):
    """Apply successful healing to similar tests."""
    similar_selectors = await find_similar_selectors(
        pattern.original_selector,
        similarity_threshold=0.8
    )

    for selector in similar_selectors:
        await suggest_healing(
            test_id=selector.test_id,
            suggested_fix=pattern.healed_selector,
            confidence=pattern.confidence * 0.9,  # Reduce for derived
            source="cross_test_learning"
        )
```

### 4.3 Predictive Healing

**Goal**: Fix tests BEFORE they break.

**Signals**:
| Signal | Weight | Action |
|--------|--------|--------|
| DOM structure changed | 0.3 | Pre-analyze new selectors |
| CSS class renamed | 0.5 | Update all references |
| Component moved | 0.7 | Rebuild locator strategies |
| API response schema changed | 0.8 | Update assertions |

**Implementation**: GitHub webhook → Diff analysis → Proactive updates

### 4.4 Accessibility Auto-Healing

| Issue | Auto-Fix |
|-------|----------|
| Missing alt text | AI generates description |
| Low color contrast | Suggest compliant colors |
| Missing form labels | Add aria-label |
| Keyboard trap | Suggest escape route |

---

## Phase 5: Scale Infrastructure (Ongoing)

### 5.1 Kubernetes Migration

**Current**: Single Railway instance
**Target**: Kubernetes with HPA

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: argus-brain
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: argus-brain
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 5.2 Multi-Region Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTI-REGION SETUP                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   US-EAST              EU-WEST              ASIA-PACIFIC    │
│  ┌───────┐            ┌───────┐            ┌───────┐       │
│  │Cluster│            │Cluster│            │Cluster│       │
│  │   +   │◀──────────▶│   +   │◀──────────▶│   +   │       │
│  │ Redis │            │ Redis │            │ Redis │       │
│  └───┬───┘            └───┬───┘            └───┬───┘       │
│      │                    │                    │            │
│      └────────────────────┼────────────────────┘            │
│                           │                                 │
│                    ┌──────▼──────┐                         │
│                    │  Supabase   │                         │
│                    │ (Primary)   │                         │
│                    └─────────────┘                         │
│                                                             │
│  Cloudflare handles routing to nearest healthy region       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Cost Projections

### Current Monthly Costs (MVP)

| Service | Cost | Notes |
|---------|------|-------|
| Railway | $20 | Single instance |
| Supabase | $25 | Pro plan |
| Cloudflare | $5 | Workers |
| Anthropic | $200 | ~65K requests |
| **Total** | **$250** | |

### Projected Costs at 1M Users

| Service | Cost | Notes |
|---------|------|-------|
| Kubernetes | $500 | GKE/EKS |
| Redis | $100 | Managed cluster |
| Kafka | $200 | Confluent Cloud |
| Supabase | $100 | Scale plan |
| Cloudflare | $50 | Pro plan |
| Anthropic | $5,000 | ~1.6M requests |
| **Total** | **$5,950** | ~$0.006/user/month |

### Cost Optimization Strategies

1. **LLM Caching**: 30% reduction via response caching
2. **Model Tiering**: Use Haiku for classification (10x cheaper)
3. **Batch Processing**: Combine related requests
4. **Edge Filtering**: Reject duplicates at Cloudflare

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| AI API downtime | Medium | High | Multi-provider fallback |
| Database overload | High | Critical | Caching + read replicas |
| Cost overrun | Medium | High | Per-project limits |
| Security breach | Low | Critical | Auth at edge, RLS |
| Model degradation | Medium | Medium | Quality monitoring |

---

## Success Metrics

### Phase 1 Success Criteria
- [ ] Zero hardcoded model IDs
- [ ] All deprecation warnings resolved
- [ ] Test coverage > 60%

### Phase 2 Success Criteria
- [ ] 90% cache hit rate for scores
- [ ] API latency < 100ms (p95)
- [ ] Database queries < 50ms (p95)

### Phase 3 Success Criteria
- [ ] Handle 1,000 req/sec sustained
- [ ] AI processing fully async
- [ ] Zero dropped events

### Phase 4 Success Criteria
- [ ] 70% of healings from knowledge base
- [ ] Cross-test learning operational
- [ ] Accessibility auto-fix coverage > 50%

---

## Immediate Next Steps (This Week)

| Priority | Task | Owner | Due |
|----------|------|-------|-----|
| P0 | Add Redis connection | Backend | Day 1-2 |
| P0 | Cache quality scores | Backend | Day 2-3 |
| P1 | Create healing_patterns table | Backend | Day 3 |
| P1 | Fix deprecation warnings | Backend | Day 4 |
| P2 | Add worker pool skeleton | Backend | Day 5 |

---

## Appendix: Migration Scripts

### A. Redis Setup
```bash
# Local development
docker run -d --name argus-redis -p 6379:6379 redis:7-alpine

# Production (Upstash)
# Add to .env:
REDIS_URL=redis://default:xxx@xxx.upstash.io:6379
```

### B. Healing Patterns Migration
```sql
-- File: supabase/migrations/20260107_healing_patterns.sql
CREATE TABLE IF NOT EXISTS healing_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fingerprint TEXT UNIQUE NOT NULL,
    original_selector TEXT NOT NULL,
    healed_selector TEXT NOT NULL,
    error_type TEXT NOT NULL,
    success_count INT DEFAULT 1,
    failure_count INT DEFAULT 0,
    confidence NUMERIC GENERATED ALWAYS AS (
        success_count::numeric / GREATEST(success_count + failure_count, 1)
    ) STORED,
    project_id UUID REFERENCES projects(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    last_used_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX idx_healing_fingerprint ON healing_patterns(fingerprint);
CREATE INDEX idx_healing_project ON healing_patterns(project_id);
CREATE INDEX idx_healing_confidence ON healing_patterns(confidence DESC);

-- RLS
ALTER TABLE healing_patterns ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own project patterns"
    ON healing_patterns FOR SELECT
    USING (project_id IN (
        SELECT id FROM projects WHERE user_id = auth.uid()
    ));
```

### C. Query Optimization Migration
```sql
-- File: supabase/migrations/20260107_query_optimization.sql

-- Composite index for quality score lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_quality_scores_project_type
    ON quality_scores(project_id, score_type, created_at DESC);

-- Partial index for recent events only
CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_events_recent
    ON production_events(project_id, created_at DESC)
    WHERE created_at > now() - interval '7 days';

-- Materialized view for correlation dashboard
CREATE MATERIALIZED VIEW IF NOT EXISTS correlation_summary AS
SELECT
    project_id,
    date_trunc('hour', created_at) as hour,
    count(*) as event_count,
    count(DISTINCT fingerprint) as unique_errors
FROM production_events
WHERE created_at > now() - interval '30 days'
GROUP BY project_id, date_trunc('hour', created_at);

CREATE UNIQUE INDEX idx_correlation_summary
    ON correlation_summary(project_id, hour);
```

---

*Last Updated: January 2025*
*Version: 1.0*
