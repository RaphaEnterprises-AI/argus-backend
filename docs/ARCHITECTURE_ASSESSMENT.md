# Argus Architecture Assessment

## Executive Summary

This document provides a comprehensive assessment of the Argus platform architecture based on deep analysis of the codebase. It covers self-healing capabilities, data pipeline scalability, computer use features, and model management.

**Overall Readiness Score: 6/10**

---

## 1. Self-Healing Capabilities

### Current State: 6/10 Maturity

#### Implemented Features

| Fix Type | Description | Implementation |
|----------|-------------|----------------|
| `SELECTOR_CHANGED` | Auto-update CSS/XPath selectors | ✅ Full implementation |
| `TIMING_ISSUE` | Add waits, retry logic | ✅ Smart wait injection |
| `ASSERTION_UPDATED` | Update expected values | ✅ Value normalization |
| `DATA_REFRESHED` | Reset test data state | ✅ State restoration |
| `NO_FIX_NEEDED` | Transient failures | ✅ Retry mechanism |

#### Key Components

- **`src/agents/self_healer.py`**: Main healing orchestration
- **`src/agents/root_cause_analyzer.py`**: Failure categorization (7 categories)
- **`src/computer_use/actions.py`**: Smart locator strategies

#### Gaps Identified

| Gap | Impact | Priority |
|-----|--------|----------|
| No cross-test learning | Repeated fixes for similar failures | HIGH |
| No ML-based pattern learning | Missing predictive healing | MEDIUM |
| No accessibility auto-healing | Manual WCAG fixes required | MEDIUM |
| No performance test auto-healing | Manual perf optimization | LOW |
| Single-session memory only | Lost healing knowledge on restart | HIGH |

#### Recommendations

1. **Implement Healing Knowledge Base**
   ```python
   # Store successful healing patterns in Supabase
   healing_patterns = {
       "fingerprint": hash(selector + error_type),
       "original_selector": "button.old-class",
       "healed_selector": "button[data-testid='submit']",
       "success_rate": 0.95,
       "applied_count": 47
   }
   ```

2. **Add Cross-Test Learning**
   - Share selector mappings across test suites
   - Build cumulative locator confidence scores
   - Implement semantic similarity for related failures

3. **Accessibility Auto-Healing**
   - Auto-add aria-labels when missing
   - Suggest color contrast fixes
   - Generate keyboard navigation alternatives

---

## 2. Data Pipeline Architecture

### Current State: NOT Scalable for High Volume

#### Current Architecture

```
Webhook → FastAPI → Normalizer → Correlator → Supabase
         (sync)     (sync)       (sync)       (3 queries)
```

**Maximum Throughput: ~50-100 requests/second**

#### Bottlenecks Identified

| Bottleneck | Current | Required for 1M+ Users |
|------------|---------|------------------------|
| Webhook Processing | Synchronous | Async with message queue |
| Database Writes | 3 round trips/event | Batched bulk inserts |
| AI Analysis | Inline during request | Decoupled async workers |
| Memory | In-process dicts | Distributed cache (Redis) |
| Scaling | Single Railway instance | Horizontal pod autoscaling |

#### Recommended Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           CLOUDFLARE EDGE               │
                    │  • Rate limiting (KV)                   │
                    │  • Deduplication (Vectorize)            │
                    │  • Fast rejection of invalid events     │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │         KAFKA / REDPANDA                │
                    │  • Event ingestion topic                │
                    │  • Partitioned by project_id            │
                    │  • 7-day retention                      │
                    └─────────────────┬───────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────┐
         ▼                            ▼                        ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ NORMALIZER POOL │      │ CORRELATOR POOL │      │ ANALYZER POOL   │
│ (K8s replicas)  │      │ (K8s replicas)  │      │ (K8s replicas)  │
│ CPU: 0.5        │      │ CPU: 1.0        │      │ CPU: 2.0        │
│ Mem: 512MB      │      │ Mem: 1GB        │      │ Mem: 4GB        │
└────────┬────────┘      └────────┬────────┘      └────────┬────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ▼
                    ┌─────────────────────────────────────────┐
                    │           TIMESCALEDB / CLICKHOUSE      │
                    │  • Time-series optimized                │
                    │  • Automatic partitioning               │
                    │  • Compression for old data             │
                    └─────────────────────────────────────────┘
```

#### Migration Path

| Phase | Change | Effort |
|-------|--------|--------|
| 1 | Add Redis for caching | 2 days |
| 2 | Implement Kafka ingestion | 1 week |
| 3 | Move AI analysis to worker pool | 1 week |
| 4 | Switch to TimescaleDB | 2 weeks |
| 5 | Kubernetes with HPA | 2 weeks |

---

## 3. Computer Use Capabilities

### Current State: Solid Claude Implementation

#### Provider Support

| Provider | Computer Use | Vision | Status |
|----------|--------------|--------|--------|
| Claude (Anthropic) | ✅ Full | ✅ Full | Production ready |
| Claude (Vertex AI) | ✅ Full | ✅ Full | Production ready |
| Gemini 2.0 | ❌ No API | ✅ Full | Vision-only fallback |
| GPT-4o | ❌ No API | ✅ Full | Vision-only fallback |

**Note**: Google Gemini does NOT have a Computer Use API as of January 2025. Claims of "Gemini computer use" refer to Project Mariner, which is an internal Google product, not an API.

#### Implementation Quality

```
src/computer_use/
├── client.py          # ✅ Full agent loop with retry
├── actions.py         # ✅ 6 action types (click, type, scroll, etc.)
├── sandbox.py         # ✅ Docker isolation
└── screenshot.py      # ✅ Optimized capture
```

#### Browser Abstraction

| Framework | Integration | Use Case |
|-----------|-------------|----------|
| Playwright | ✅ Primary | Fast, reliable automation |
| Puppeteer | ✅ Supported | Node.js environments |
| Selenium | ✅ Supported | Legacy compatibility |
| CDP | ✅ Direct | Low-level control |
| Cloudflare Browser | ✅ Worker | Serverless execution |
| Claude Computer Use | ✅ Vision | Intelligent automation |

#### Recommendations

1. **Add Vision-Based Fallback for Gemini**
   ```python
   async def execute_with_vision_fallback(task: str, screenshot: bytes):
       """Use vision analysis when computer use unavailable."""
       if provider == "google":
           # Extract element coordinates from vision
           elements = await gemini.analyze_screenshot(screenshot)
           # Execute via Playwright based on coordinates
           return await playwright.click_at(elements[0].center)
   ```

2. **Implement Hybrid Mode**
   - Use Playwright for speed
   - Use Computer Use for verification
   - Fall back to Computer Use when Playwright fails

---

## 4. Model Management

### Current State: Centralized (Just Implemented)

#### Model Registry Location

**`src/core/model_registry.py`** - Single source of truth

#### Registered Models

| Key | Model ID | Provider | Best For |
|-----|----------|----------|----------|
| `claude-opus-4-5` | claude-opus-4-5 | Anthropic | Complex debugging |
| `claude-sonnet-4-5` | claude-sonnet-4-5 | Anthropic | Default tasks |
| `claude-haiku-4-5` | claude-haiku-4-5 | Anthropic | Fast classification |
| `gemini-2.0-flash` | gemini-2.0-flash-exp | Google | Fast + cheap |
| `gemini-2.0-pro` | gemini-2.0-pro-exp | Google | Complex reasoning |
| `gpt-4o` | gpt-4o | OpenAI | Cross-validation |
| `gpt-4o-mini` | gpt-4o-mini | OpenAI | Fast tasks |
| `llama-3.3-70b` | llama-3.3-70b-versatile | Groq | Fast open source |
| `deepseek-v3` | deepseek-ai/DeepSeek-V3 | Together | Code generation |

#### Task-to-Model Mapping

```python
TASK_MODEL_PRIORITY = {
    TaskType.CLASSIFICATION: ["claude-haiku-4-5", "gemini-2.0-flash"],
    TaskType.TEST_GENERATION: ["claude-sonnet-4-5", "gemini-2.0-pro"],
    TaskType.DEBUGGING: ["claude-opus-4-5", "claude-sonnet-4-5"],
    TaskType.COMPUTER_USE: ["claude-sonnet-4-5", "claude-opus-4-5"],
    TaskType.SELF_HEALING: ["claude-sonnet-4-5", "gemini-2.0-pro"],
}
```

#### Migration Required

The following files still have hardcoded model IDs:
- `src/agents/*.py` - Multiple agents
- `src/core/cognitive_engine.py`
- `src/integrations/*.py`

**Action**: Import from `model_registry.py` instead of hardcoding.

---

## 5. Scalability Assessment

### Can We Handle 1M+ Users?

**Current Answer: NO**

| Metric | Current | Required | Gap |
|--------|---------|----------|-----|
| Requests/sec | 50-100 | 10,000+ | 100x |
| Concurrent users | ~100 | 100,000+ | 1000x |
| Event storage/day | 1M | 1B+ | 1000x |
| AI calls/day | 10K | 10M+ | 1000x |

### Path to Scale

1. **Short-term (Current Sprint)**
   - Add Redis caching
   - Implement request batching
   - Use Cloudflare KV for deduplication

2. **Medium-term (Next Month)**
   - Deploy Kafka for event streaming
   - Move AI analysis to async workers
   - Implement proper rate limiting

3. **Long-term (Quarter)**
   - Kubernetes with horizontal autoscaling
   - TimescaleDB for time-series data
   - Multi-region deployment

---

## 6. Immediate Action Items

| Priority | Task | Owner | Status |
|----------|------|-------|--------|
| P0 | Fix remaining hardcoded model IDs | Dev | In Progress |
| P0 | Add SecretStr handling to all agents | Dev | Done |
| P1 | Implement Redis caching layer | Dev | Not Started |
| P1 | Add Kafka for event ingestion | Infra | Not Started |
| P2 | Build healing knowledge base | Dev | Not Started |
| P2 | Add accessibility auto-healing | Dev | Not Started |
| P3 | Kubernetes migration | Infra | Not Started |

---

## Appendix: File Reference

### Core Intelligence
- `src/core/model_registry.py` - Model configuration (NEW)
- `src/core/model_router.py` - Multi-model routing
- `src/core/normalizer.py` - Event normalization
- `src/core/correlator.py` - Error correlation
- `src/core/risk.py` - Risk scoring

### Agents
- `src/agents/self_healer.py` - Test healing
- `src/agents/root_cause_analyzer.py` - Failure analysis
- `src/agents/quality_auditor.py` - Quality scoring

### Computer Use
- `src/computer_use/client.py` - Claude Computer Use
- `src/tools/browser_worker_client.py` - Browser automation

### Data Pipeline
- `src/api/webhooks.py` - Event ingestion
- `src/api/quality.py` - Quality API
