# KEDA Setup for Selenium Grid 4 on Kubernetes

## Overview

KEDA (Kubernetes Event-Driven Autoscaling) provides advanced autoscaling capabilities beyond CPU/memory metrics. This guide covers setting up KEDA to scale Selenium Grid 4 nodes based on:

1. **Session queue length** - Primary scaler for browser nodes
2. **Prometheus metrics** - Custom metrics from Selenium Grid
3. **HTTP endpoints** - Direct metrics from Grid API
4. **Request backlog** - Queue buildup detection

## Architecture

```
┌────────────────────────────────────────────────────┐
│            KEDA Scaler Controller                   │
│         (Runs in keda namespace)                    │
└────────────────────────────────────────────────────┘
                        ▲
                        │ Monitors metrics from:
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Selenium    │ │  Prometheus  │ │   Custom     │
│  Grid API    │ │  (via HTTP)  │ │  Scaler gRPC │
│  /CapacityAPI│ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
        ▲               ▲               ▲
        │               │               │
┌──────────────────────────────────────────────────┐
│    Selenium Grid Hub + Chrome/Firefox Nodes      │
│    (Deployments scaled by KEDA ScaledObjects)    │
└──────────────────────────────────────────────────┘
```

## Prerequisites

- Kubernetes 1.20+
- KEDA 2.13+ installed (`helm repo add kedacore https://kedacore.github.io/charts`)
- Selenium Grid 4.27+ deployed
- Prometheus (optional, for advanced metrics)
- kubectl configured

## Step 1: Install KEDA

```bash
# Add KEDA Helm repository
helm repo add kedacore https://kedacore.github.io/charts
helm repo update

# Create namespace
kubectl create namespace keda

# Install KEDA
helm upgrade --install keda kedacore/keda \
  --namespace keda \
  --set serviceAccount.annotations."azure\.workload\.identity/client-id"='' \
  --values - <<EOF
podIdentities: []
logging:
  level: info
  format: json
EOF

# Verify installation
kubectl get pods -n keda
```

## Step 2: Install Selenium Grid with Prometheus Metrics

Deploy Selenium Grid with metrics enabled:

```bash
helm repo add selenium https://www.selenium.dev/docker-selenium
helm upgrade --install selenium-grid selenium/selenium-grid \
  --namespace selenium-grid \
  --create-namespace \
  -f /Users/bvk/Downloads/e2e-testing-agent/browser-pool/selenium-grid/values.yaml
```

## Available Scalers for Selenium Grid

### Option 1: HTTP Endpoint Scaler (Recommended for Selenium Grid)

**Best for:** Session queue monitoring via Selenium Grid REST API

**Pros:**
- No additional infrastructure needed
- Direct access to Grid metrics
- Simplest setup

**Cons:**
- Requires Grid to expose metrics endpoint
- TLS validation needed for secure endpoints

### Option 2: Prometheus Scaler

**Best for:** Advanced metrics aggregation and complex queries

**Pros:**
- Full PromQL capabilities
- Flexible metric combinations
- Integrates with existing monitoring

**Cons:**
- Requires Prometheus installation
- Extra infrastructure to maintain
- Metric collection overhead

### Option 3: External Scaler (Custom)

**Best for:** Complex custom metrics or legacy systems

**Pros:**
- Full control over scaling logic
- Support for any data source
- Custom business logic

**Cons:**
- Must maintain scaler service
- Additional gRPC service
- Higher complexity

## Configuration Details

### Selenium Grid Metrics

Selenium Grid 4 exposes the following important metrics:

**Session Queue Metrics:**
```
# Via /status endpoint (HTTP)
GET /status

# Via Prometheus metrics (if enabled)
selenium_sessions_queued      # Number of sessions waiting to be created
selenium_sessions_active      # Number of active sessions
selenium_nodes_up             # Number of available nodes
selenium_node_slots_available # Available slots across all nodes
selenium_node_slots_used      # Used slots
```

**Health Check:**
```
GET /readyz    # Readiness probe
GET /livez     # Liveness probe
```

## Best Practices

### 1. Scaling Strategy

```
Min Replicas: 1-2 (always keep 1-2 nodes running)
Max Replicas: 100+ (depends on budget)
Target Metric: Queue length / Available slots
Scale Up: Aggressive (add nodes every 30s if queue > 0)
Scale Down: Conservative (wait 5min, remove 1 node at a time)
```

### 2. Metric Selection Priority

1. **Queue Length** (Primary) - Most reliable indicator
2. **Active Sessions** (Secondary) - Detect stuck sessions
3. **Node Availability** (Tertiary) - Prevent starvation

### 3. Scale-to-Zero

Selenium Grid typically shouldn't scale to zero (you always want 1-2 idle nodes), but KEDA supports it with:
- Multiple scalers with OR logic
- Fallback to CPU-based scaling
- Minimum replica guarantees

### 4. Multi-Browser Scaling

Scale different browser types independently:
- Chrome nodes (fast, lightweight) - aggressive scaling
- Firefox nodes (slower, heavier) - conservative scaling
- Edge nodes (rare) - manual scaling

### 5. Cost Optimization

```
During Off-Hours:
  - Chrome:  2 replicas
  - Firefox: 0 replicas
  - Scale-down wait: 10 minutes

During Peak Hours:
  - Chrome:  auto scale to 50
  - Firefox: auto scale to 20
  - Scale-down wait: 5 minutes
```

### 6. Monitoring & Alerts

Key metrics to monitor:
- `keda_scaler_active` - Number of active scalers
- `keda_scaler_error` - Scaler errors
- `keda_trigger_errors_total` - Total trigger errors
- Custom: Queue length / Available slots ratio

Alert thresholds:
- Queue length > 100 for 5 min
- Active sessions = Max replicas for 5 min
- Scaler error rate > 10%

## Common Pitfalls

### Pitfall 1: Too Aggressive Scale-Up
**Problem:** Creates many pods too quickly, wasting resources
**Solution:** Set `stabilizationWindowSeconds: 60` and use Percent-based policies

### Pitfall 2: Forgetting Min Replicas
**Problem:** Selenium Grid doesn't respond (scales to 0)
**Solution:** Always set `minReplicas: 1` or higher

### Pitfall 3: Incorrect Queue Threshold
**Problem:** Scales too early or too late
**Solution:** Monitor baseline queue size and set threshold 20-30% above average

### Pitfall 4: Mixing Scalers Incorrectly
**Problem:** Conflicting scale decisions (HPA vs KEDA)
**Solution:** Use KEDA alone, remove HPA for the same deployment

### Pitfall 5: Not Accounting for Startup Time
**Problem:** New nodes take 30+ seconds to register, queue keeps growing
**Solution:** Add padding to threshold or use predictive scaling

## Testing Your KEDA Setup

```bash
# 1. Check KEDA operator is running
kubectl get pods -n keda

# 2. Apply ScaledObject
kubectl apply -f scaledobject-selenium-grid.yaml

# 3. Verify ScaledObject
kubectl get scaledobject -n selenium-grid

# 4. Check scaler status
kubectl describe scaledobject selenium-chrome-scaler -n selenium-grid

# 5. Monitor scaling events
kubectl get events -n selenium-grid --sort-by='.lastTimestamp'

# 6. Generate load to test
# Use a stress test to queue up sessions
curl -X POST http://hub:4444/session -d '{"desiredCapabilities":{"browserName":"chrome"}}'

# 7. Watch pods scaling
watch kubectl get pods -n selenium-grid
```

## Troubleshooting

### ScaledObject Not Triggering

```bash
# Check scaler status
kubectl describe scaledobject selenium-chrome-scaler -n selenium-grid

# Check KEDA logs
kubectl logs -n keda -l app=keda-operator --tail=100

# Verify endpoint is accessible
kubectl run -it curl --image=curlimages/curl -- sh
curl http://selenium-hub:4444/status
```

### Pods Not Scaling Down

```bash
# Check stabilization window
# Default: 300s (5 min)

# Check minimum replicas
kubectl get deployment selenium-chrome-nodes -n selenium-grid -o yaml | grep minReplicas

# Check if pods are still handling sessions
kubectl top pods -n selenium-grid
kubectl exec -it <pod> -- ps aux | grep chrome
```

### High Error Rate in KEDA

```bash
# Check metric endpoint availability
kubectl port-forward -n selenium-grid svc/selenium-hub 4444:4444
curl http://localhost:4444/status

# Check authentication if using secrets
kubectl get secrets -n selenium-grid

# Review KEDA operator logs for API errors
kubectl logs -n keda deployment/keda-operator -f
```

## References

- [KEDA Official Documentation](https://keda.sh/docs/)
- [Selenium Grid 4 Documentation](https://www.selenium.dev/documentation/grid/)
- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Prometheus Queries](https://prometheus.io/docs/prometheus/latest/querying/basics/)
