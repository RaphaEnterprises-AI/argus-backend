# KEDA for Selenium Grid 4 - Complete Reference

This directory contains a complete KEDA (Kubernetes Event-Driven Autoscaling) setup for Selenium Grid 4 on Kubernetes.

## Files Overview

### Core Configuration Files

| File | Purpose |
|------|---------|
| `keda-setup.md` | Complete setup guide with architecture and best practices |
| `keda-values.yaml` | Helm values for installing KEDA operator |
| `keda-deployment-guide.md` | Step-by-step deployment instructions |
| `scaledobject-selenium-grid.yaml` | KEDA ScaledObjects for Chrome/Firefox/Edge nodes |
| `selenium-keda-scaler.yaml` | External scaler service (gRPC-based metrics) |
| `keda-auth-config.yaml` | Authentication and TLS configuration |
| `keda-prometheus-advanced.yaml` | Prometheus-based advanced scaling strategies |
| `Dockerfile.keda-scaler` | Docker image for external scaler |

## Quick Start (3 Steps)

### Step 1: Install KEDA
```bash
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda -n keda --create-namespace -f keda-values.yaml
```

### Step 2: Deploy Selenium Grid
```bash
helm repo add selenium https://www.selenium.dev/docker-selenium
helm install selenium-grid selenium/selenium-grid -n selenium-grid --create-namespace
```

### Step 3: Apply Scaling Configuration
```bash
kubectl apply -f scaledobject-selenium-grid.yaml
kubectl apply -f selenium-keda-scaler.yaml  # Optional: external scaler
```

## Scaling Strategies

### Strategy 1: External Scaler (Recommended)

**Best for:** Direct access to Selenium Grid metrics via gRPC

```bash
# Deploy external scaler
kubectl apply -f selenium-keda-scaler.yaml

# This enables:
# - session_queue_length: Number of queued sessions
# - firefox_queue_length: Firefox-specific queue
# - sessions_active: Active sessions count
# - slots_available: Available browser slots
```

**Advantages:**
- No additional infrastructure needed
- Direct Selenium Grid integration
- Simple configuration
- Reliable metric collection

**Disadvantages:**
- Custom scaler to maintain
- gRPC protocol overhead
- Single point of failure if not HA

### Strategy 2: Prometheus Scaler

**Best for:** Complex metrics aggregation and advanced queries

```bash
# Install Prometheus (optional)
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# Apply Prometheus-based scaling
kubectl apply -f keda-prometheus-advanced.yaml
```

**Advantages:**
- Full PromQL capabilities
- Flexible metric combinations
- Integrates with existing monitoring
- Multiple scalers can cooperate
- Better for multi-cluster setups

**Disadvantages:**
- Requires Prometheus installation
- Extra infrastructure to maintain
- Metric collection latency
- More complex configuration

### Strategy 3: CPU/Memory Scaler

**Best for:** Fallback or simple workloads

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: selenium-cpu-fallback
  namespace: selenium-grid
spec:
  scaleTargetRef:
    name: selenium-chrome-node
    kind: Deployment
  minReplicaCount: 2
  maxReplicaCount: 50
  triggers:
    - type: cpu
      metadata:
        type: Utilization
        value: "70"
```

**Advantages:**
- No additional setup needed
- Uses native Kubernetes metrics
- Good fallback option

**Disadvantages:**
- Reactive (scales after resource usage)
- Not ideal for browser workloads
- May miss queue buildup

## Configuration Reference

### ScaledObject Parameters

```yaml
spec:
  scaleTargetRef:              # What to scale
    apiVersion: apps/v1
    kind: Deployment
    name: deployment-name

  minReplicaCount: 2           # Minimum pods (0 = scale to zero)
  maxReplicaCount: 50          # Maximum pods

  pollingInterval: 15          # Check metrics every N seconds
  cooldownPeriod: 60           # Wait N seconds before scaling down

  fallback:                    # If scaler fails
    failureThreshold: 3        # Failures before fallback
    replicas: 2                # Replicas when fallback active

  advanced:                    # Advanced scaling behavior
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 30
          policies:
          - type: Pods
            value: 5           # Add 5 pods at a time
            periodSeconds: 30  # Every 30 seconds
          - type: Percent
            value: 50          # Or 50% increase
            periodSeconds: 30
          selectPolicy: Max    # Use most aggressive

        scaleDown:
          stabilizationWindowSeconds: 300  # Wait 5 minutes
          policies:
          - type: Pods
            value: 2           # Remove 2 pods at a time
            periodSeconds: 60  # Every 60 seconds
          selectPolicy: Min    # Use most conservative
```

### Trigger Types

#### External Scaler

```yaml
triggers:
  - type: external
    metadata:
      scalerAddress: "scaler-service:6000"
      metric: session_queue_length
      threshold: "10"          # Scale when metric > 10
      unsafeSsl: "false"
```

#### Prometheus

```yaml
triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      query: "ceil(max(selenium_sessions_queued))"
      threshold: "10"
      activationThreshold: "1"  # Min before active
```

#### CPU

```yaml
triggers:
  - type: cpu
    metadata:
      type: Utilization        # Or "AverageValue"
      value: "70"              # Target utilization %
```

## Best Practices

### 1. Set Reasonable Min/Max Replicas

```yaml
# Too low - insufficient capacity, users wait
minReplicaCount: 1
maxReplicaCount: 10

# Good - starts small, scales as needed
minReplicaCount: 2
maxReplicaCount: 100

# Too high - wasted resources
minReplicaCount: 50
maxReplicaCount: 500
```

### 2. Browser-Specific Configuration

Chrome (lightweight, fast startup):
```yaml
minReplicaCount: 2
maxReplicaCount: 100
pollingInterval: 15
cooldownPeriod: 30
```

Firefox (heavier, slower startup):
```yaml
minReplicaCount: 1
maxReplicaCount: 20
pollingInterval: 30
cooldownPeriod: 300
```

### 3. Multi-Trigger Scaling

Combine multiple metrics for better decisions:

```yaml
triggers:
  # Primary: queue length
  - type: external
    metadata:
      scalerAddress: "selenium-keda-scaler:6000"
      metric: session_queue_length
      threshold: "5"

  # Secondary: available slots ratio
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      query: "(available / (available + used))"
      threshold: "0.2"

  # Fallback: CPU utilization
  - type: cpu
    metadata:
      type: Utilization
      value: "80"
```

### 4. Graceful Shutdown Configuration

In your Deployment:

```yaml
terminationGracePeriodSeconds: 120  # Time for active sessions
lifecycle:
  preStop:
    exec:
      command: ["/bin/sh", "-c", "sleep 15"]  # Wait for load balancer update
```

### 5. Pod Disruption Budgets

Prevent KEDA from being disrupted:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: keda-operator-pdb
  namespace: keda
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: keda-operator
```

## Monitoring

### Key Metrics to Watch

**KEDA Metrics:**
```
keda_trigger_active           # Whether trigger is active
keda_scaler_error             # Scaler errors
keda_scaler_active_total      # Number of active scalers
```

**Kubernetes HPA Metrics:**
```
keda_hpa_last_scale_time      # When last scaling occurred
keda_hpa_target_metrics       # Target metric values
```

**Custom Metrics:**
```
selenium_sessions_queued      # Queued sessions
selenium_sessions_active      # Active sessions
selenium_node_slots_available # Available slots
selenium_node_slots_used      # Used slots
```

### Set Up Alerts

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: selenium-grid-alerts
spec:
  groups:
  - name: selenium
    rules:
    # Queue backing up
    - alert: SeleniumQueueBackup
      expr: selenium_sessions_queued > 50
      for: 5m
      annotations:
        summary: "{{ $value }} sessions queued"

    # At capacity
    - alert: SeleniumAtCapacity
      expr: (used / (used + available)) > 0.9
      for: 5m
      annotations:
        summary: "Grid at {{ $value | humanizePercentage }} capacity"

    # Scaler error
    - alert: KEDAScalerError
      expr: increase(keda_scaler_error[5m]) > 0
      for: 2m
      annotations:
        summary: "KEDA scaler errors detected"
```

## Troubleshooting

### Verify Installation

```bash
# Check KEDA operator
kubectl get pods -n keda
kubectl logs -n keda -l app=keda-operator

# Check KEDA webhooks
kubectl get validatingwebhookconfigurations | grep keda

# Check custom resources
kubectl get crd | grep keda
```

### Test External Scaler

```bash
# Port forward to scaler
kubectl port-forward -n selenium-grid svc/selenium-keda-scaler 6000:6000

# Test gRPC connection (requires grpcurl)
grpcurl -plaintext -d '{"scalerMetadata":{"metric":"session_queue_length"}}' \
  localhost:6000 externalscaler.ExternalScaler/IsActive
```

### Check Selenium Grid Status

```bash
# Port forward to hub
kubectl port-forward -n selenium-grid svc/selenium-hub 4444:4444

# Get status
curl http://localhost:4444/status | jq '.value'

# Check specific metrics
curl http://localhost:4444/status | jq '.value | {ready, nodeCount, maxSession, sessionCount}'
```

### Verify ScaledObject

```bash
# Get scaler status
kubectl describe scaledobject selenium-chrome-nodes-scaler -n selenium-grid

# Check associated HPA
kubectl describe hpa -n selenium-grid

# Watch scaling events
kubectl get events -n selenium-grid --sort-by='.lastTimestamp'
```

## Performance Tuning

### For High Throughput (1000+ sessions/day)

```yaml
minReplicaCount: 10
maxReplicaCount: 500
pollingInterval: 10      # Check very frequently
cooldownPeriod: 15       # Fast scale-down

triggers:
  - type: external
    metadata:
      scalerAddress: "selenium-keda-scaler:6000"
      metric: session_queue_length
      threshold: "2"     # Aggressive threshold
```

### For Cost Optimization

```yaml
minReplicaCount: 1
maxReplicaCount: 20
pollingInterval: 60      # Check less frequently
cooldownPeriod: 600      # Wait 10 minutes

triggers:
  - type: external
    metadata:
      scalerAddress: "selenium-keda-scaler:6000"
      metric: session_queue_length
      threshold: "50"    # Conservative threshold
```

### For Development/Testing

```yaml
minReplicaCount: 1
maxReplicaCount: 5
pollingInterval: 30

triggers:
  - type: cpu
    metadata:
      type: Utilization
      value: "50"
```

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| ScaledObject in "Unknown" state | External scaler not running | `kubectl apply -f selenium-keda-scaler.yaml` |
| Pods not scaling | Metrics endpoint unreachable | `kubectl port-forward svc/selenium-hub 4444:4444 && curl http://localhost:4444/status` |
| Scaling too aggressive | Threshold too low | Increase `threshold` value in trigger |
| Scaling too slow | Polling too infrequent | Decrease `pollingInterval` |
| Pods not scaling down | `cooldownPeriod` too high | Reduce `cooldownPeriod` or adjust HPA behavior |
| High error rate | Selenium Hub unstable | Check `kubectl logs -n selenium-grid selenium-hub-*` |

## Next Steps

1. **Read detailed guides:**
   - `keda-setup.md` - Architecture and best practices
   - `KEDA-DEPLOYMENT-GUIDE.md` - Step-by-step deployment

2. **Explore configuration:**
   - `scaledobject-selenium-grid.yaml` - ScaledObject examples
   - `keda-prometheus-advanced.yaml` - Prometheus integration
   - `keda-auth-config.yaml` - Authentication setup

3. **Test your setup:**
   - Run load tests to verify scaling
   - Monitor metrics during peak usage
   - Adjust thresholds based on real-world usage

4. **Production considerations:**
   - Set up monitoring and alerts
   - Configure Pod Disruption Budgets
   - Implement graceful shutdown
   - Use external scaler in HA configuration
   - Regular backups of ScaledObject configs

## References

- [KEDA Official Docs](https://keda.sh/docs/)
- [Selenium Grid Docs](https://www.selenium.dev/documentation/grid/)
- [Kubernetes HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Prometheus Queries](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [gRPC Protocol](https://grpc.io/docs/)
