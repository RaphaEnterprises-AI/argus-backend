# Monitoring and Observability

Configure monitoring, logging, and alerting for Skopaq Enterprise.

## Metrics Overview

Skopaq exposes Prometheus metrics at `/metrics` on port 8000.

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `argus_tests_total` | Counter | Total tests executed |
| `argus_tests_passed` | Counter | Passed tests |
| `argus_tests_failed` | Counter | Failed tests |
| `argus_healing_attempts` | Counter | Self-healing attempts |
| `argus_healing_success` | Counter | Successful healings |
| `argus_llm_requests_total` | Counter | LLM API calls |
| `argus_llm_tokens_total` | Counter | Tokens consumed |
| `argus_llm_cost_usd` | Counter | LLM costs in USD |
| `argus_request_duration_seconds` | Histogram | API request latency |

## Prometheus Integration

### Enable ServiceMonitor

```yaml
# values.yaml
metrics:
  enabled: true
  serviceMonitor:
    enabled: true
    namespace: monitoring  # Where Prometheus Operator is installed
    interval: 30s
    scrapeTimeout: 10s
    labels:
      release: prometheus  # Match your Prometheus Operator labels
```

### Manual Prometheus Config

If not using Prometheus Operator:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'argus-brain'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: ['argus']
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_component]
        action: keep
        regex: brain
      - source_labels: [__meta_kubernetes_pod_container_port_number]
        action: keep
        regex: "8000"
```

## Grafana Dashboards

### Import Pre-built Dashboard

1. Go to Grafana → Dashboards → Import
2. Enter dashboard ID or paste JSON
3. Select Prometheus data source

### Dashboard Panels

**Test Execution Overview**
```
rate(argus_tests_total[5m])
sum(argus_tests_passed) / sum(argus_tests_total) * 100
```

**Self-Healing Rate**
```
sum(argus_healing_success) / sum(argus_healing_attempts) * 100
```

**LLM Cost Tracking**
```
sum(increase(argus_llm_cost_usd[24h]))
```

**API Latency**
```
histogram_quantile(0.95, rate(argus_request_duration_seconds_bucket[5m]))
```

### Sample Dashboard JSON

```json
{
  "title": "Skopaq Enterprise Overview",
  "panels": [
    {
      "title": "Tests per Hour",
      "type": "stat",
      "targets": [
        {
          "expr": "sum(increase(argus_tests_total[1h]))"
        }
      ]
    },
    {
      "title": "Pass Rate",
      "type": "gauge",
      "targets": [
        {
          "expr": "sum(argus_tests_passed) / sum(argus_tests_total) * 100"
        }
      ]
    },
    {
      "title": "Self-Healing Success Rate",
      "type": "gauge",
      "targets": [
        {
          "expr": "sum(argus_healing_success) / sum(argus_healing_attempts) * 100"
        }
      ]
    },
    {
      "title": "Daily LLM Cost",
      "type": "stat",
      "targets": [
        {
          "expr": "sum(increase(argus_llm_cost_usd[24h]))"
        }
      ]
    }
  ]
}
```

## Alerting

### Prometheus AlertManager Rules

```yaml
# alerting-rules.yaml
groups:
  - name: argus
    rules:
      # High test failure rate
      - alert: SkopaqHighFailureRate
        expr: |
          sum(rate(argus_tests_failed[5m])) /
          sum(rate(argus_tests_total[5m])) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High test failure rate (> 20%)"
          description: "Test failure rate is {{ $value | humanizePercentage }}"

      # Self-healing not working
      - alert: SkopaqSelfHealingFailing
        expr: |
          sum(rate(argus_healing_success[1h])) /
          sum(rate(argus_healing_attempts[1h])) < 0.5
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Self-healing success rate below 50%"

      # High LLM costs
      - alert: SkopaqHighLLMCost
        expr: sum(increase(argus_llm_cost_usd[1h])) > 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM costs (> $50/hour)"
          description: "LLM costs in the last hour: ${{ $value | humanize }}"

      # API latency
      - alert: SkopaqHighLatency
        expr: |
          histogram_quantile(0.95, rate(argus_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency (p95 > 5s)"

      # Pod not ready
      - alert: SkopaqPodNotReady
        expr: |
          kube_pod_status_ready{namespace="argus", condition="true"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Skopaq pod {{ $labels.pod }} not ready"
```

### Configure AlertManager

```yaml
# alertmanager.yml
route:
  receiver: 'argus-team'
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - match:
        severity: critical
      receiver: 'argus-pager'

receivers:
  - name: 'argus-team'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/xxx'
        channel: '#argus-alerts'

  - name: 'argus-pager'
    pagerduty_configs:
      - service_key: 'xxx'
```

## Logging

### Log Configuration

```yaml
brain:
  env:
    LOG_LEVEL: "INFO"        # DEBUG, INFO, WARNING, ERROR
    LOG_FORMAT: "json"       # json or text
    LOG_OUTPUT: "stdout"     # stdout or file
```

### Log Aggregation

#### Loki + Promtail

```yaml
# promtail config
scrape_configs:
  - job_name: argus
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: ['argus']
    pipeline_stages:
      - json:
          expressions:
            level: level
            message: message
            timestamp: timestamp
      - labels:
          level:
      - timestamp:
          source: timestamp
          format: RFC3339
```

#### Elasticsearch + Fluentd

```yaml
# fluentd config
<match argus.**>
  @type elasticsearch
  host elasticsearch.logging.svc
  port 9200
  logstash_format true
  logstash_prefix argus
</match>
```

### Structured Log Fields

```json
{
  "timestamp": "2026-01-30T08:30:00Z",
  "level": "INFO",
  "service": "argus-brain",
  "component": "self_healer",
  "message": "Test healed successfully",
  "test_id": "test_123",
  "healing_type": "selector_update",
  "duration_ms": 450,
  "trace_id": "abc123"
}
```

## Distributed Tracing

### OpenTelemetry Configuration

```yaml
brain:
  env:
    TRACING_ENABLED: "true"
    OTEL_EXPORTER_OTLP_ENDPOINT: "http://jaeger-collector:4317"
    OTEL_SERVICE_NAME: "argus-brain"
    OTEL_RESOURCE_ATTRIBUTES: "deployment.environment=production"
```

### Jaeger Integration

```bash
# Deploy Jaeger
kubectl apply -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.50.0/jaeger-operator.yaml

# Create Jaeger instance
kubectl apply -f - <<EOF
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: argus-jaeger
  namespace: argus
spec:
  strategy: production
  storage:
    type: elasticsearch
    elasticsearch:
      nodeCount: 3
EOF
```

## Health Checks

### Liveness and Readiness Probes

```yaml
brain:
  livenessProbe:
    httpGet:
      path: /health
      port: 8000
    initialDelaySeconds: 30
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 3

  readinessProbe:
    httpGet:
      path: /health
      port: 8000
    initialDelaySeconds: 5
    periodSeconds: 5
    timeoutSeconds: 3
    failureThreshold: 3
```

### Health Check Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Basic health check |
| `/health/ready` | Readiness (all dependencies) |
| `/health/live` | Liveness (process running) |
| `/health/data-layer` | Data layer components |

```bash
# Check all health endpoints
curl http://argus-brain:8000/health
curl http://argus-brain:8000/health/data-layer
```

## Dashboard Access

### Grafana via Ingress

```yaml
# grafana-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: grafana
  namespace: monitoring
spec:
  ingressClassName: nginx
  rules:
    - host: grafana.company.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: grafana
                port:
                  number: 3000
```

### Port-Forward for Development

```bash
# Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring

# Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n monitoring

# Jaeger
kubectl port-forward svc/argus-jaeger-query 16686:16686 -n argus
```
