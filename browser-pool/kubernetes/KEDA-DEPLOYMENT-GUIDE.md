# KEDA for Selenium Grid 4 - Deployment Guide

Complete guide to deploy and configure KEDA for autoscaling Selenium Grid on Kubernetes.

## Quick Start (5 minutes)

```bash
# 1. Install KEDA
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
kubectl create namespace keda
helm install keda kedacore/keda -n keda -f keda-values.yaml

# 2. Deploy Selenium Grid (if not already installed)
helm repo add selenium https://www.selenium.dev/docker-selenium
helm install selenium-grid selenium/selenium-grid \
  -n selenium-grid \
  --create-namespace

# 3. Apply KEDA ScaledObjects
kubectl apply -f scaledobject-selenium-grid.yaml

# 4. Deploy KEDA External Scaler (optional, for gRPC-based scaling)
kubectl apply -f selenium-keda-scaler.yaml

# 5. Verify
kubectl get scaledobject -n selenium-grid
kubectl get pods -n selenium-grid
watch kubectl get hpa -n selenium-grid
```

## Full Installation Steps

### Step 1: Prerequisites

```bash
# Check Kubernetes version
kubectl version --short

# Check metrics server installed (required for CPU-based fallback)
kubectl get deployment metrics-server -n kube-system

# If metrics server not installed:
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Step 2: Install KEDA Operator

```bash
# Add KEDA Helm repo
helm repo add kedacore https://kedacore.github.io/charts
helm repo update

# Create namespace
kubectl create namespace keda

# Install KEDA with custom values
helm upgrade --install keda kedacore/keda \
  --namespace keda \
  --values keda-values.yaml \
  --wait

# Verify installation
kubectl get pods -n keda
kubectl get crds | grep keda
```

### Step 3: Deploy Selenium Grid

Option A: Using Helm (Recommended)

```bash
helm repo add selenium https://www.selenium.dev/docker-selenium
helm repo update

helm upgrade --install selenium-grid selenium/selenium-grid \
  --namespace selenium-grid \
  --create-namespace \
  --values /path/to/your/selenium-values.yaml
```

Option B: Using provided Helm values

```bash
helm upgrade --install selenium-grid selenium/selenium-grid \
  --namespace selenium-grid \
  --create-namespace \
  --set global.seleniumGrid.imageTag="4.27.0" \
  --set hub.replicas=1 \
  --set chromeNode.replicas=2 \
  --set chromeNode.maxSessions=2 \
  --set firefoxNode.enabled=true \
  --set firefoxNode.replicas=0
```

### Step 4: Deploy External Scaler (Optional)

The external scaler provides direct access to Selenium Grid metrics via gRPC:

```bash
# Apply external scaler deployment
kubectl apply -f selenium-keda-scaler.yaml

# Verify it's running
kubectl get pods -n selenium-grid -l app=selenium-keda-scaler
kubectl logs -n selenium-grid -l app=selenium-keda-scaler
```

### Step 5: Apply KEDA ScaledObjects

```bash
# Apply ScaledObjects (configure Chrome/Firefox/Edge scaling)
kubectl apply -f scaledobject-selenium-grid.yaml

# Verify ScaledObjects created
kubectl get scaledobject -n selenium-grid
kubectl describe scaledobject selenium-chrome-nodes-scaler -n selenium-grid
```

### Step 6: Apply Authentication (Optional)

If your Selenium Grid requires authentication:

```bash
# Create secrets with credentials
kubectl apply -f keda-auth-config.yaml

# Update ScaledObject to reference auth
# Edit scaledobject-selenium-grid.yaml and add:
# authenticationRef:
#   name: selenium-grid-auth
```

### Step 7: Setup Prometheus (Optional)

For advanced metrics-based scaling:

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Apply Prometheus-based KEDA config
kubectl apply -f keda-prometheus-advanced.yaml

# Access Prometheus
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
# Open http://localhost:9090
```

## Configuration Examples

### Example 1: Basic Queue-Based Scaling

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: selenium-basic
  namespace: selenium-grid
spec:
  scaleTargetRef:
    name: selenium-chrome-node
    apiVersion: apps/v1
    kind: Deployment

  minReplicaCount: 2
  maxReplicaCount: 50
  pollingInterval: 15
  cooldownPeriod: 60

  triggers:
    # Scale based on queue length
    - type: external
      metadata:
        scalerAddress: "selenium-keda-scaler:6000"
        metric: session_queue_length
        threshold: "10"
```

### Example 2: Multi-Metric Scaling with Prometheus

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: selenium-multi-metric
  namespace: selenium-grid
spec:
  scaleTargetRef:
    name: selenium-chrome-node
    apiVersion: apps/v1
    kind: Deployment

  minReplicaCount: 2
  maxReplicaCount: 100

  triggers:
    # Trigger 1: Queue length (primary)
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        query: "ceil(max(selenium_sessions_queued))"
        threshold: "5"

    # Trigger 2: Available slots ratio
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        query: |
          (
            selenium_node_slots_available /
            (selenium_node_slots_available + selenium_node_slots_used)
          )
        threshold: "0.2"
```

### Example 3: Browser-Specific Scaling

```yaml
---
# Chrome: Aggressive scaling
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: selenium-chrome-aggressive
  namespace: selenium-grid
spec:
  scaleTargetRef:
    name: selenium-chrome-node
    kind: Deployment
  minReplicaCount: 2
  maxReplicaCount: 100
  pollingInterval: 15
  cooldownPeriod: 60
  triggers:
    - type: external
      metadata:
        scalerAddress: "selenium-keda-scaler:6000"
        metric: session_queue_length
        threshold: "5"

---
# Firefox: Conservative scaling
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: selenium-firefox-conservative
  namespace: selenium-grid
spec:
  scaleTargetRef:
    name: selenium-firefox-node
    kind: Deployment
  minReplicaCount: 1
  maxReplicaCount: 20
  pollingInterval: 30      # Check less frequently
  cooldownPeriod: 300      # Wait longer before scaling down
  triggers:
    - type: external
      metadata:
        scalerAddress: "selenium-keda-scaler:6000"
        metric: firefox_queue_length
        threshold: "10"     # Higher threshold

---
# Edge: Manual scaling (via HPA)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: selenium-edge-hpa
  namespace: selenium-grid
spec:
  scaleTargetRef:
    name: selenium-edge-node
    kind: Deployment
  minReplicas: 0
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Example 4: Time-Based Scaling

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: selenium-time-based
  namespace: selenium-grid
spec:
  scaleTargetRef:
    name: selenium-chrome-node
    kind: Deployment

  triggers:
    # Scale up during business hours
    - type: cron
      metadata:
        timezone: UTC
        start: "0 9 * * 1-5"        # 9 AM weekdays
        end: "0 18 * * 1-5"         # 6 PM weekdays
        desiredReplicas: "20"
        desiredReplicasInactive: "2"

    # Fallback: queue-based scaling
    - type: external
      metadata:
        scalerAddress: "selenium-keda-scaler:6000"
        metric: session_queue_length
        threshold: "5"
```

## Monitoring & Troubleshooting

### Check KEDA Status

```bash
# Get KEDA operator status
kubectl get deployment -n keda
kubectl logs -n keda -l app=keda-operator --tail=100

# Get ScaledObject status
kubectl get scaledobject -n selenium-grid
kubectl describe scaledobject selenium-chrome-nodes-scaler -n selenium-grid

# Get HPA status (created by KEDA)
kubectl get hpa -n selenium-grid
kubectl describe hpa keda-hpa-selenium-chrome-node -n selenium-grid
```

### Monitor Scaling Events

```bash
# Watch scaling events in real-time
kubectl get events -n selenium-grid --sort-by='.lastTimestamp' --watch

# Get pod scaling history
kubectl describe deployment selenium-chrome-node -n selenium-grid

# Check KEDA metrics
kubectl port-forward -n keda svc/keda-operator 8080:8080
# View metrics at http://localhost:8080/metrics
```

### Common Issues & Solutions

**Issue 1: ScaledObject stuck in "Unknown" state**

```bash
# Check for errors
kubectl describe scaledobject selenium-chrome-nodes-scaler -n selenium-grid

# Common causes:
# - External scaler not reachable
# - Prometheus endpoint not accessible
# - Selenium Hub not responding

# Solution:
kubectl port-forward -n selenium-grid svc/selenium-hub 4444:4444
curl http://localhost:4444/status
```

**Issue 2: Pods not scaling despite high queue**

```bash
# Check HPA details
kubectl get hpa -n selenium-grid
kubectl describe hpa <hpa-name> -n selenium-grid

# Check metrics flow
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1/namespaces/selenium-grid/pods/*/session_queue_length

# Check resource requests (required by HPA)
kubectl get pods -n selenium-grid -o json | grep -A5 "resources:"
```

**Issue 3: Scaling down too slow**

```bash
# Reduce cooldownPeriod in ScaledObject
# Change from 60 to 30 seconds:
kubectl patch scaledobject selenium-chrome-nodes-scaler -n selenium-grid \
  --type merge -p '{"spec":{"cooldownPeriod":30}}'

# Adjust HPA behavior
kubectl patch hpa keda-hpa-selenium-chrome-node -n selenium-grid \
  --type merge -p '{"spec":{"behavior":{"scaleDown":{"stabilizationWindowSeconds":60}}}}'
```

### Performance Tuning

**For High Throughput (1000+ sessions/day):**

```yaml
spec:
  minReplicaCount: 10              # Start with more
  maxReplicaCount: 200
  pollingInterval: 10              # Check more frequently
  cooldownPeriod: 30

  triggers:
    - type: external
      metadata:
        scalerAddress: "selenium-keda-scaler:6000"
        metric: session_queue_length
        threshold: "2"              # Lower threshold
```

**For Cost Optimization (minimal resources):**

```yaml
spec:
  minReplicaCount: 1               # Minimal baseline
  maxReplicaCount: 20
  pollingInterval: 30              # Check less frequently
  cooldownPeriod: 300              # Wait longer before scaling down

  triggers:
    - type: external
      metadata:
        scalerAddress: "selenium-keda-scaler:6000"
        metric: session_queue_length
        threshold: "20"             # Higher threshold
```

## Complete YAML Deployment

Deploy everything at once:

```bash
# Create all resources
kubectl apply -f keda-values.yaml          # Install KEDA
kubectl apply -f scaledobject-selenium-grid.yaml   # ScaledObjects
kubectl apply -f selenium-keda-scaler.yaml         # External scaler
kubectl apply -f keda-auth-config.yaml             # Auth (optional)
kubectl apply -f keda-prometheus-advanced.yaml     # Prometheus (optional)

# Verify
kubectl get all -n selenium-grid
kubectl get all -n keda
```

## Testing Your Setup

### Load Test

```bash
# Create a simple load test script
cat > /tmp/load-test.sh <<'EOF'
#!/bin/bash
for i in {1..100}; do
  curl -X POST http://selenium-hub:4444/session \
    -H "Content-Type: application/json" \
    -d '{"capabilities":{"alwaysMatch":{"browserName":"chrome"}}}' &
  sleep 0.1
done
wait
echo "100 sessions requested"
EOF

chmod +x /tmp/load-test.sh

# Run load test
kubectl run load-test --image=curlimages/curl -n selenium-grid -- /tmp/load-test.sh

# Watch pods scaling
watch kubectl get pods -n selenium-grid
```

### Monitor Queue

```bash
# In one terminal, watch pods
watch kubectl get pods -n selenium-grid

# In another terminal, check queue via API
kubectl port-forward -n selenium-grid svc/selenium-hub 4444:4444

# In third terminal
while true; do
  curl -s http://localhost:4444/status | jq '.value.sessionQueue | length'
  sleep 5
done
```

## Uninstall

```bash
# Remove ScaledObjects
kubectl delete scaledobject --all -n selenium-grid

# Remove external scaler
kubectl delete deployment selenium-keda-scaler -n selenium-grid
kubectl delete service selenium-keda-scaler -n selenium-grid

# Remove KEDA
helm uninstall keda -n keda
kubectl delete namespace keda

# Remove Selenium Grid
helm uninstall selenium-grid -n selenium-grid
kubectl delete namespace selenium-grid
```

## References

- KEDA Documentation: https://keda.sh/docs/
- Selenium Grid Documentation: https://www.selenium.dev/documentation/grid/
- Kubernetes HPA: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- Prometheus Queries: https://prometheus.io/docs/prometheus/latest/querying/basics/
