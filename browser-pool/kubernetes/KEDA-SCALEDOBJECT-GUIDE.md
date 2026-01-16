# KEDA ScaledObject Deployment Guide for Selenium Grid 4

## Overview

This guide provides comprehensive instructions for deploying KEDA autoscaling manifests for Selenium Grid 4 browser nodes. The configuration uses KEDA's ScaledObject to dynamically scale Chrome, Firefox, and Edge browser nodes based on session queue length and resource utilization metrics from Prometheus.

## Prerequisites

1. **Kubernetes Cluster** (1.19+)
2. **KEDA Operator** (2.14+) installed in the cluster
3. **Prometheus** deployed for metrics collection
4. **Selenium Grid 4** deployed with metrics exposition
5. **kubectl** configured to access the cluster

### Pre-deployment Verification

```bash
# Verify KEDA is installed
kubectl get namespace keda
kubectl get deployment -n keda keda-operator keda-operator-metrics-apiserver

# Verify Prometheus is running
kubectl get pods -n prometheus -l app=prometheus

# Verify Selenium Grid is running
kubectl get pods -n selenium-grid
kubectl get svc -n selenium-grid selenium-grid-selenium-hub
```

## File Structure

```
/Users/bvk/Downloads/e2e-testing-agent/browser-pool/kubernetes/
├── keda-scaledobject.yaml          # Main autoscaling manifests (THIS FILE)
├── KEDA-SCALEDOBJECT-GUIDE.md      # This deployment guide
├── keda-setup.md                    # KEDA installation guide
├── keda-prometheus-advanced.yaml    # Advanced Prometheus configuration
├── keda-auth-config.yaml            # Authentication configuration
└── scaledobject-selenium-grid.yaml  # Legacy alternative configuration
```

## Deployment Instructions

### Step 1: Validate YAML Syntax

```bash
# Validate the YAML before deployment
kubectl apply -f keda-scaledobject.yaml --dry-run=client

# Check for any syntax errors
kubectl apply -f keda-scaledobject.yaml --dry-run=server
```

### Step 2: Deploy the ScaledObjects

```bash
# Deploy the KEDA autoscaling configuration
kubectl apply -f keda-scaledobject.yaml

# Verify deployment
kubectl get scaledobjects -n selenium-grid
kubectl describe scaledobject selenium-chrome-nodes-scaler -n selenium-grid
```

### Step 3: Verify Deployment Status

```bash
# Check ScaledObject status
kubectl get scaledobjects -n selenium-grid -o wide

# View detailed status
kubectl describe scaledobject selenium-chrome-nodes-scaler -n selenium-grid

# Monitor HPA created by KEDA
kubectl get hpa -n selenium-grid
kubectl describe hpa selenium-chrome-nodes-scaler -n selenium-grid

# Check KEDA scaler logs
kubectl logs -n keda deployment/keda-operator -f | grep -i "selenium"
```

## Configuration Details

### Chrome Nodes Scaler (`selenium-chrome-nodes-scaler`)

**Purpose**: Autoscales Chrome browser nodes for high-throughput testing

**Configuration**:
- **Min Replicas**: 2 (maintains baseline availability)
- **Max Replicas**: 20 (cost control)
- **Target Deployment**: `selenium-grid-selenium-node-chrome`

**Scaling Triggers**:

1. **Primary: Session Queue Length**
   - Prometheus Query: `ceil(selenium_sessions_queued{node="chrome"}) > 0 or on() vector(0)`
   - Threshold: >= 1 (scale up immediately if sessions queued)
   - Response: Adds 5 pods every 30 seconds or 50% increase

2. **Secondary: Available Slots Ratio**
   - Prometheus Query: Calculates available slots percentage
   - Threshold: 0.2 (scale up when only 20% slots available)
   - Response: Ensures responsive scaling before queue builds

3. **Tertiary: CPU Utilization Fallback**
   - Threshold: 70% average CPU
   - Response: Fallback if Prometheus unavailable

**Scale Behavior**:
- **Scale Up**: Aggressive (5 pods/30s, 50% increase)
- **Scale Down**: Conservative (2 pods/60s, 10% decrease, 5-minute stabilization)

### Firefox Nodes Scaler (`selenium-firefox-nodes-scaler`)

**Purpose**: Autoscales Firefox nodes with conservative approach (heavier resource consumption)

**Configuration**:
- **Min Replicas**: 1 (Firefox consumes more resources)
- **Max Replicas**: 10 (cost control)
- **Target Deployment**: `selenium-grid-selenium-node-firefox`

**Scaling Triggers**:

1. **Session Queue Length** (threshold: >= 1)
2. **Available Slots Ratio** (threshold: 0.15 = 85% utilization)
3. **CPU Utilization** (threshold: 80%)

**Scale Behavior**:
- **Scale Up**: Moderate (2 pods/45s, 30% increase)
- **Scale Down**: Very conservative (1 pod/120s, 5% decrease, 10-minute stabilization)

### Edge Nodes Scaler (`selenium-edge-nodes-scaler`)

**Purpose**: Autoscales optional Edge nodes with aggressive cost-saving approach

**Configuration**:
- **Min Replicas**: 0 (scales to zero when not needed)
- **Max Replicas**: 15 (cost savings potential)
- **Target Deployment**: `selenium-grid-selenium-node-edge`

**Scaling Triggers**:

1. **Session Queue Length** (threshold: >= 1)
2. **Available Slots Ratio** (threshold: 0.10 = 90% utilization)
3. **CPU Utilization** (threshold: no explicit fallback)

**Scale Behavior**:
- **Scale Up**: Very aggressive (10 pods/15s, 100% increase)
- **Scale Down**: Aggressive (3 pods/60s, 20% decrease, 3-minute stabilization)

## Prometheus Integration

### Required Metrics

The Selenium Grid must expose the following metrics:

```
# Session queue metrics
selenium_sessions_queued{node="chrome"}
selenium_sessions_queued{node="firefox"}
selenium_sessions_queued{node="edge"}

# Node slot metrics
selenium_node_slots_available{node="chrome"}
selenium_node_slots_used{node="chrome"}
selenium_node_slots_available{node="firefox"}
selenium_node_slots_used{node="firefox"}
selenium_node_slots_available{node="edge"}
selenium_node_slots_used{node="edge"}
```

### Verify Metrics in Prometheus

```bash
# Port-forward to Prometheus
kubectl port-forward -n prometheus svc/prometheus 9090:9090

# Navigate to: http://localhost:9090
# Run queries:
# - selenium_sessions_queued
# - selenium_node_slots_available
# - selenium_node_slots_used
```

### Enable ServiceMonitor (if using Prometheus Operator)

The YAML includes a ServiceMonitor resource that automatically adds Selenium Grid to Prometheus scrape targets.

## Monitoring and Troubleshooting

### Monitor Scaling Events

```bash
# Watch scaling events in real-time
kubectl get events -n selenium-grid --sort-by='.lastTimestamp' -w

# Check HPA metrics
kubectl get hpa -n selenium-grid -o wide
kubectl top pods -n selenium-grid -l app=selenium-grid-selenium-node-chrome

# View KEDA scaler metrics
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq .
```

### Common Issues and Solutions

#### Issue 1: ScaledObjects show "unknown" status

```bash
# Check KEDA operator logs
kubectl logs -n keda deployment/keda-operator -f

# Verify metrics API is working
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1/namespaces/selenium-grid/pods/*/session_queue_length
```

**Solution**: Ensure KEDA metrics API server is running:
```bash
kubectl get deployment -n keda keda-operator-metrics-apiserver
kubectl logs -n keda deployment/keda-operator-metrics-apiserver
```

#### Issue 2: Prometheus metrics not found

```bash
# Verify Prometheus is scraping Selenium Grid
kubectl port-forward -n prometheus svc/prometheus 9090:9090
# Navigate to: http://localhost:9090/targets
# Look for: selenium-grid-selenium-hub
```

**Solution**: Ensure ServiceMonitor is properly configured:
```bash
kubectl get servicemonitor -n selenium-grid
kubectl describe servicemonitor selenium-grid-metrics -n selenium-grid
```

#### Issue 3: Scaling not triggering

```bash
# Check KEDA scaler status and events
kubectl describe scaledobject selenium-chrome-nodes-scaler -n selenium-grid
kubectl get events -n selenium-grid | grep scaledobject

# Check if metrics are returning values
kubectl logs -n keda deployment/keda-operator -f | grep "session_queue"
```

**Solution**:
1. Verify Prometheus metrics are being collected
2. Check metric query syntax in ScaledObject
3. Verify thresholds are appropriate for your load

#### Issue 4: Scaling too aggressive or too conservative

**If scaling too aggressively**:
- Increase `stabilizationWindowSeconds` in scale up behavior
- Decrease `value` in scale up policies
- Increase thresholds for triggers

**If scaling too conservatively**:
- Decrease `stabilizationWindowSeconds` in scale up behavior
- Increase `value` in scale up policies
- Decrease thresholds for triggers

Example modification:
```yaml
scaleUp:
  stabilizationWindowSeconds: 60  # Increase from 30
  policies:
  - type: Pods
    value: 3  # Decrease from 5
    periodSeconds: 30
```

## Performance Tuning

### Polling Interval

Current settings: 15s (Chrome), 20s (Firefox), 10s (Edge)

- **Faster polling** (lower value): More responsive but higher API load
- **Slower polling** (higher value): Lower API load but less responsive

```bash
# Recommended values based on load patterns:
# High frequency, low latency: 5-10 seconds
# Medium frequency: 15-30 seconds
# Low frequency, cost-optimized: 30-60 seconds
```

### Cooldown Period

Current settings: 60s (Chrome), 120s (Firefox), 90s (Edge)

- **Shorter cooldown**: Allow faster recovery from metric changes
- **Longer cooldown**: Reduce thrashing and API calls

### Stabilization Window

Prevents rapid scaling up/down when metrics fluctuate:

```yaml
scaleUp:
  stabilizationWindowSeconds: 30  # Wait 30s before committing scale decision
scaleDown:
  stabilizationWindowSeconds: 300 # Wait 5 min before scaling down
```

## Best Practices

### 1. Resource Requests and Limits

Ensure each browser node pod has appropriate resource requests:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

KEDA uses these requests for node selection and scaling calculations.

### 2. Pod Disruption Budgets

The YAML includes PodDisruptionBudgets to maintain availability during maintenance:

```bash
kubectl get pdb -n selenium-grid
kubectl describe pdb selenium-chrome-nodes-pdb -n selenium-grid
```

### 3. Node Affinity

Consider adding node affinity to spread browser pods across nodes:

```yaml
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - selenium-grid-selenium-node-chrome
        topologyKey: kubernetes.io/hostname
```

### 4. Monitoring and Alerting

Set up Prometheus alerts for scaling anomalies:

```yaml
groups:
- name: selenium-grid-scaling
  rules:
  - alert: SeleniumGridHighQueueLength
    expr: selenium_sessions_queued{node="chrome"} > 50
    for: 5m
    annotations:
      summary: "High session queue for Chrome nodes"

  - alert: SeleniumGridMaxReplicasReached
    expr: keda_scaler_replicas{scaled_object="selenium-chrome-nodes-scaler"} >= 20
    for: 5m
    annotations:
      summary: "Chrome nodes at maximum replicas"
```

### 5. Cost Optimization

Monitor scaling metrics and adjust:

```bash
# Query average replicas over time
avg(rate(keda_scaler_replicas[1h]))

# Calculate cost based on replica hours
# Cost = (avg_replicas * hourly_rate) * hours_per_month
```

## Updating the Configuration

### Rolling Update

To update scaling parameters without downtime:

```bash
# Edit the ScaledObject
kubectl edit scaledobject selenium-chrome-nodes-scaler -n selenium-grid

# Or apply updated YAML
kubectl apply -f keda-scaledobject.yaml

# Verify changes took effect
kubectl describe scaledobject selenium-chrome-nodes-scaler -n selenium-grid
```

### Pausing Autoscaling

To temporarily disable autoscaling:

```bash
# Pause via annotation
kubectl annotate scaledobject selenium-chrome-nodes-scaler \
  -n selenium-grid \
  keda.sh/paused=true \
  --overwrite

# Resume autoscaling
kubectl annotate scaledobject selenium-chrome-nodes-scaler \
  -n selenium-grid \
  keda.sh/paused=false \
  --overwrite
```

## Cleanup

To remove KEDA scaling configuration:

```bash
# Delete ScaledObjects
kubectl delete scaledobject -n selenium-grid -l app=selenium-grid

# Verify removal
kubectl get scaledobjects -n selenium-grid

# HPA objects created by KEDA will be automatically removed
kubectl get hpa -n selenium-grid
```

## Advanced Configuration

### Custom Metrics

For custom metrics beyond queue length and available slots:

```yaml
triggers:
- type: prometheus
  metadata:
    serverAddress: http://prometheus-operated:9090
    query: |
      # Custom: Scale based on response time
      (selenium_response_time_ms{node="chrome"} / 1000)
    threshold: "5"  # 5 second threshold
    activationThreshold: "1"
```

### Multiple Scalers

Combine multiple KEDA scalers for different metrics:

```bash
# Each ScaledObject can have multiple triggers
# KEDA scales when ANY trigger is active (OR logic)
triggers:
- type: prometheus  # Trigger 1
- type: cpu         # Trigger 2
- type: memory      # Trigger 3
```

### Integration with Other Tools

KEDA can be integrated with:
- **Slack**: Alert notifications
- **PagerDuty**: Incident escalation
- **Datadog**: Advanced monitoring
- **n8n**: Workflow automation

## Support and References

- **KEDA Documentation**: https://keda.sh/
- **Selenium Grid Metrics**: https://www.selenium.dev/documentation/grid/
- **Prometheus Queries**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Kubernetes HPA**: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/

## Appendix: Quick Reference

### Verify Deployment

```bash
# All in one
kubectl get scaledobjects,hpa,pdb -n selenium-grid

# Detailed status
for obj in $(kubectl get scaledobjects -n selenium-grid -o name); do
  echo "=== $obj ==="
  kubectl describe $obj -n selenium-grid | grep -A 10 "Status:"
done
```

### View Current Metrics

```bash
# Chrome queue length
kubectl exec -it -n selenium-grid deployment/selenium-grid-selenium-hub -- \
  curl -s http://localhost:4444/status | jq '.value.nodes[0].stereotypes[0].slots'

# Available slots
kubectl exec -it -n selenium-grid deployment/selenium-grid-selenium-hub -- \
  curl -s http://localhost:4444/status | jq '.value.nodes[].stereotypes[] | select(.stereotypeName=="chrome") | .slots'
```

### Force Scaling

```bash
# Manually set replica count
kubectl scale deployment selenium-grid-selenium-node-chrome \
  -n selenium-grid \
  --replicas=5

# Resume automatic scaling (KEDA will take over)
kubectl delete hpa -n selenium-grid -l app=selenium-grid
kubectl apply -f keda-scaledobject.yaml
```
