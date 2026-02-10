# Skopaq Observability Architecture

**Version:** 1.0
**Last Updated:** January 29, 2026

---

## Overview

Skopaq implements a comprehensive observability stack covering:
1. **LLM Tracing** - Langfuse for AI model observability
2. **Infrastructure Metrics** - Prometheus + Grafana
3. **Application Logging** - Structured logging with correlation
4. **Distributed Tracing** - Request tracking across services

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ARGUS OBSERVABILITY STACK                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         USER REQUESTS                                    │    │
│  │                                                                          │    │
│  │    Dashboard ──────▶ API Gateway ──────▶ Backend Services               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                    │                                             │
│                 ┌──────────────────┼──────────────────┐                         │
│                 ▼                  ▼                  ▼                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │    LANGFUSE      │  │   PROMETHEUS     │  │    LOGGING       │              │
│  │                  │  │                  │  │                  │              │
│  │  • LLM Traces    │  │  • Metrics       │  │  • Structlog     │              │
│  │  • Token Usage   │  │  • Alerts        │  │  • JSON Format   │              │
│  │  • Cost Tracking │  │  • SLOs          │  │  • Correlation   │              │
│  │  • Model Perf    │  │  • Dashboards    │  │  • Request IDs   │              │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘              │
│           │                     │                     │                         │
│           ▼                     ▼                     ▼                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         GRAFANA DASHBOARDS                               │    │
│  │                                                                          │    │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │    │
│  │   │ AI Intel    │  │   Cognee    │  │  Browser    │  │   System    │    │    │
│  │   │ Dashboard   │  │  Pipeline   │  │    Pool     │  │  Overview   │    │    │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. LLM Observability (Langfuse)

### Overview

Langfuse provides comprehensive LLM observability including:
- Token usage and cost tracking per request
- Model performance metrics (latency, errors)
- Trace visualization for multi-step AI workflows
- User/session attribution

### Integration Points

| Component | Integration | File |
|-----------|-------------|------|
| Chat API | CallbackHandler | `src/api/chat.py` |
| Orchestrator | Trace spans | `src/orchestrator/langfuse_integration.py` |
| Agents | Per-agent traces | `src/agents/base.py` |
| Cognee Worker | LLM calls | `data-layer/cognee-worker/src/worker.py` |

### Configuration

```python
# src/orchestrator/langfuse_integration.py

def get_langfuse_handler(
    user_id: str,
    session_id: str,
    trace_name: str,
    tags: list[str],
    metadata: dict
) -> CallbackHandler:
    """Create Langfuse callback handler for LangChain/LangGraph."""
    return CallbackHandler(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        user_id=user_id,
        session_id=session_id,
        trace_name=trace_name,
        tags=tags,
        metadata=metadata,
    )
```

### Environment Variables

```bash
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Kubernetes Secrets

```yaml
# data-layer/kubernetes/monitoring/langfuse-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: langfuse-credentials
type: Opaque
data:
  LANGFUSE_PUBLIC_KEY: <base64>
  LANGFUSE_SECRET_KEY: <base64>
```

---

## 2. Infrastructure Metrics (Prometheus)

### Stack Components

| Component | Purpose | Configuration |
|-----------|---------|---------------|
| Prometheus | Metrics collection | `kube-prometheus-stack-values.yaml` |
| Grafana | Visualization | `grafana-dashboards-configmap.yaml` |
| AlertManager | Alert routing | `alerting-rules.yaml` |
| ServiceMonitors | Endpoint scraping | `servicemonitors-*.yaml` |

### ServiceMonitors

**Skopaq Data Layer:**
```yaml
# data-layer/kubernetes/monitoring/servicemonitors-argus-data.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: argus-data-metrics
spec:
  selector:
    matchLabels:
      app: cognee-worker
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics
```

**Browser Pool:**
```yaml
# data-layer/kubernetes/monitoring/servicemonitors-browser-pool.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: browser-pool-metrics
spec:
  selector:
    matchLabels:
      app: selenium-grid
  endpoints:
    - port: web
      interval: 30s
```

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `cognee_ecl_operations_total` | Counter | Cognee ECL pipeline operations |
| `cognee_search_latency_seconds` | Histogram | Knowledge search latency |
| `healing_suggestions_total` | Counter | Self-healing suggestions generated |
| `llm_tokens_total` | Counter | LLM token usage by model |
| `browser_pool_active_sessions` | Gauge | Active browser sessions |

---

## 3. Grafana Dashboards

### Dashboard Inventory

| Dashboard | Purpose | Panels |
|-----------|---------|--------|
| AI Intelligence | LLM performance, costs | Token usage, latency, errors |
| Cognee Pipeline | Knowledge layer health | ECL metrics, search latency |
| Browser Pool | Browser automation | Session count, execution time |
| System Overview | Infrastructure health | CPU, memory, network |

### AI Intelligence Dashboard

```json
{
  "title": "AI Intelligence Dashboard",
  "panels": [
    {
      "title": "LLM Token Usage",
      "type": "timeseries",
      "targets": [
        {
          "expr": "sum(rate(llm_tokens_total[5m])) by (model)"
        }
      ]
    },
    {
      "title": "LLM Latency P95",
      "type": "stat",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))"
        }
      ]
    },
    {
      "title": "Healing Success Rate",
      "type": "gauge",
      "targets": [
        {
          "expr": "sum(healing_suggestions_accepted) / sum(healing_suggestions_total) * 100"
        }
      ]
    }
  ]
}
```

---

## 4. Alerting Rules

### Critical Alerts

```yaml
# data-layer/kubernetes/monitoring/alerting-rules.yaml

groups:
  - name: argus-critical
    rules:
      - alert: HighLLMErrorRate
        expr: rate(llm_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High LLM error rate detected"

      - alert: CogneeUnhealthy
        expr: up{job="cognee-worker"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Cognee worker is down"

      - alert: BrowserPoolExhausted
        expr: browser_pool_available_sessions == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "No available browser sessions"
```

---

## 5. Secure Access (Cloudflare Tunnel)

### Architecture

```
┌──────────────┐      ┌──────────────────┐      ┌──────────────┐
│   External   │      │   Cloudflare     │      │   K8s        │
│   User       │─────▶│   Tunnel         │─────▶│   Grafana    │
│              │ HTTPS│   (cloudflared)  │      │   Service    │
└──────────────┘      └──────────────────┘      └──────────────┘
```

### Configuration

```yaml
# data-layer/kubernetes/monitoring/cloudflare-tunnel-config.yaml
tunnel: argus-monitoring
credentials-file: /etc/cloudflared/credentials.json
ingress:
  - hostname: grafana.argus.example.com
    service: http://kube-prometheus-stack-grafana:80
  - hostname: prometheus.argus.example.com
    service: http://kube-prometheus-stack-prometheus:9090
  - service: http_status:404
```

---

## 6. Logging Standards

### Structured Logging Format

```python
import structlog

logger = structlog.get_logger()

# Standard log format
logger.info(
    "Chat message processed",
    thread_id=thread_id,
    user_id=user_id,
    model=model_id,
    tokens_used=token_count,
    latency_ms=latency,
)
```

### Log Fields

| Field | Type | Description |
|-------|------|-------------|
| `thread_id` | string | Conversation thread identifier |
| `user_id` | string | Authenticated user ID |
| `org_id` | string | Organization ID (multi-tenant) |
| `model` | string | LLM model used |
| `tokens_used` | int | Token count for request |
| `latency_ms` | int | Request latency |
| `trace_id` | string | Langfuse trace ID |

---

## 7. Health Endpoints

### API Health Checks

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `/health` | Basic liveness | `{"status": "ok"}` |
| `/health/ready` | Readiness probe | `{"ready": true}` |
| `/api/v1/health/data-layer` | Infrastructure health | Component statuses |

### Example Response

```json
{
  "status": "healthy",
  "components": {
    "cognee": {"status": "healthy", "version": "0.5.1"},
    "falkordb": {"status": "healthy"},
    "supabase": {"status": "healthy"},
    "selenium_grid": {"status": "healthy", "nodes": 3},
    "prometheus": {"status": "healthy"},
    "grafana": {"status": "healthy", "version": "12.3.1"}
  }
}
```

---

## 8. Deployment

### Enable Monitoring Stack

```bash
# Apply Prometheus stack
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  -f data-layer/kubernetes/monitoring/kube-prometheus-stack-values.yaml

# Apply custom configs
kubectl apply -f data-layer/kubernetes/monitoring/

# Verify deployment
kubectl get servicemonitors
kubectl get prometheusrules
```

### Environment Variables (Railway)

```bash
# Set in Railway dashboard or CLI
railway variables set LANGFUSE_ENABLED=true
railway variables set LANGFUSE_PUBLIC_KEY=pk-lf-xxx
railway variables set LANGFUSE_SECRET_KEY=sk-lf-xxx
```

---

## 9. Troubleshooting

### Common Issues

| Issue | Symptom | Resolution |
|-------|---------|------------|
| Missing Langfuse traces | No traces in dashboard | Check `LANGFUSE_ENABLED=true` |
| Metrics not appearing | Empty Grafana panels | Verify ServiceMonitor labels match |
| Alerts not firing | No notifications | Check AlertManager config |

### Diagnostic Commands

```bash
# Check Prometheus targets
kubectl port-forward svc/kube-prometheus-stack-prometheus 9090
# Visit http://localhost:9090/targets

# Check Langfuse connectivity
curl -X POST https://cloud.langfuse.com/api/public/health

# View Cognee worker logs
kubectl logs -l app=cognee-worker -f
```

---

## 10. Future Enhancements

1. **OpenTelemetry Integration** - Unified tracing standard
2. **Custom Langfuse Dashboards** - LLM-specific analytics
3. **Anomaly Detection** - ML-based alert generation
4. **Cost Allocation** - Per-tenant AI cost tracking
5. **SLO Monitoring** - Service level objective tracking
