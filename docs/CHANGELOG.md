# Argus Changelog

All notable changes to the Argus platform are documented here.

---

## [2026.01.29] - January 29, 2026

### Major Changes

#### AI Systems Layer - Bug Fixes & Hardening

**Critical Bug Fixes:**
- **Supervisor Orchestration Fixed** (`src/orchestrator/supervisor.py`)
  - Changed `config: dict` to `config: RunnableConfig` for LangGraph auto-injection
  - Supervisor multi-agent orchestration now works correctly
  - Commit: `de3aa00`

- **Chat API Input Validation** (`src/api/chat.py`)
  - Added `Literal["user", "assistant", "system"]` type for role validation
  - Added content length validation (max 100KB)
  - Added empty messages array validation
  - All invalid inputs now return proper 422 errors instead of 500/502 crashes
  - Commit: `58cb22a`

**Validation Results:**
| Test Category | Pass Rate |
|---------------|-----------|
| Input Validation | 100% (8/8) |
| Security | 100% (4/4) |
| Concurrency | 100% (2/2) |
| Error Handling | 100% (4/4) |
| Cognee Integration | 100% (3/3) |
| LLM Safety | 100% (2/2) |
| Healing Pipeline | 100% (3/3) |
| Supervisor Orchestration | 100% (3/3) |
| **Overall** | **96.7%** |

---

#### Observability & Monitoring Stack

**Langfuse LLM Tracing** (New)
- Centralized LLM observability across all AI calls
- Files: `src/orchestrator/langfuse_integration.py`, `src/core/ai_client.py`
- Features:
  - Token usage tracking per request
  - Cost calculation per model
  - Trace correlation across agents
  - User/session attribution

**Kubernetes Monitoring** (Enhanced)
- Added kube-prometheus-stack with custom values
- Cloudflare Tunnel for secure external access
- New Grafana dashboards:
  - AI Intelligence Dashboard
  - Cognee Pipeline Metrics
  - Browser Pool Metrics
- ServiceMonitors for argus-data and browser-pool
- Alerting rules for critical metrics

**Files Changed:**
```
data-layer/kubernetes/monitoring/
├── alerting-rules.yaml (NEW)
├── cloudflare-tunnel-config.yaml
├── cloudflare-tunnel.yaml
├── grafana-dashboards-configmap.yaml (NEW)
├── intelligence-dashboard-configmap.yaml (NEW)
├── kube-prometheus-stack-values.yaml (NEW)
├── langfuse-secrets.yaml (NEW)
├── langfuse-values.yaml (NEW)
├── servicemonitors-argus-data.yaml (NEW)
└── servicemonitors-browser-pool.yaml (NEW)
```

---

#### Cognee Knowledge Layer

**Cognee Worker Enhancements** (`data-layer/cognee-worker/src/worker.py`)
- Added Langfuse SDK instrumentation for LLM tracing
- Updated to Cognee 0.5+ API (positional args)
- Improved ArgusEvent 'data' field handling
- Network policy updated to allow Langfuse connectivity

---

#### Infrastructure Health Status

| Component | Status | Notes |
|-----------|--------|-------|
| Cognee | Healthy | v0.5.1, postgres + pgvector |
| FalkorDB | Healthy | Graph DB for knowledge |
| Supabase | Healthy | Database REST API |
| Selenium Grid | Healthy | 3 nodes ready |
| Prometheus | Healthy | via Cloudflare Tunnel |
| Grafana | Healthy | v12.3.1 |
| Valkey | Healthy | Cache (K8s internal) |
| Flink | Healthy | Stream processing |
| Redpanda | Warning | SSL config needs attention |

---

### Technical Debt Addressed

1. **Input Validation Gap** - Chat API was crashing on malformed input
   - Root cause: Missing Pydantic validators
   - Fix: Added field_validator decorators

2. **LangGraph Config Injection** - Supervisor node wasn't receiving config
   - Root cause: Wrong type hint (`dict` vs `RunnableConfig`)
   - Fix: Proper type annotation for auto-injection

3. **Observability Gap** - No LLM tracing across agents
   - Root cause: Each agent had separate logging
   - Fix: Centralized Langfuse integration

---

### API Changes

**Chat Endpoint** (`POST /api/v1/chat/message`)

Request validation now enforces:
```python
class ChatMessage:
    role: Literal["user", "assistant", "system"]  # Strict enum
    content: str  # Required, max 100KB

class ChatRequest:
    messages: list[ChatMessage]  # Cannot be empty
```

Error responses are now proper 422 with details:
```json
{
  "detail": [
    {
      "type": "literal_error",
      "loc": ["body", "messages", 0, "role"],
      "msg": "Input should be 'user', 'assistant' or 'system'"
    }
  ]
}
```

---

### Deployment Notes

**Railway Deployment:**
```bash
# Deploy latest changes
railway up --detach

# Verify deployment
railway deployment list --json | jq '.[0].status'
```

**Kubernetes Monitoring:**
```bash
# Apply monitoring stack
kubectl apply -f data-layer/kubernetes/monitoring/

# Access Grafana via Cloudflare Tunnel
# URL: Configured in cloudflare-tunnel-config.yaml
```

---

## [2026.01.28] - January 28, 2026

### Changes

- Added Cloudflare Tunnel for secure monitoring access
- Updated backend proxy configuration for kube-prometheus-stack
- Added Upstash Redis credentials to cognee-worker
- Enhanced Cognee metrics in Grafana dashboards

---

## [2026.01.27] - January 27, 2026

### Changes

- Added extended model info fields for UI
- Added SSL context for SASL_SSL connections in event-gateway
- Comprehensive tests for UpstashRedisClient

---

## Previous Releases

See git history for changes prior to January 27, 2026.
