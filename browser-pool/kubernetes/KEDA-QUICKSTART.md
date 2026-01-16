# KEDA Autoscaling for Selenium Grid 4 - Quick Start Guide

## What Was Created

This directory now contains complete KEDA autoscaling manifests for Selenium Grid 4 browser nodes. Three new primary files have been created:

### Core Files

1. **keda-scaledobject.yaml** (543 lines)
   - Main autoscaling configuration for Chrome, Firefox, and Edge nodes
   - Uses Prometheus metrics for intelligent scaling decisions
   - Includes PodDisruptionBudgets and ServiceMonitor
   - Production-ready with fallback mechanisms

2. **KEDA-SCALEDOBJECT-GUIDE.md**
   - Comprehensive deployment and configuration guide
   - Step-by-step instructions for all scenarios
   - Troubleshooting section with common issues and solutions
   - Performance tuning recommendations

3. **keda-prometheus-queries.yaml**
   - Ready-to-use Prometheus queries for monitoring
   - Prometheus alert rules
   - Grafana dashboard queries
   - Recording rules for performance optimization

4. **keda-validation-script.sh**
   - Automated validation script to verify deployment
   - Health checks for all components
   - Troubleshooting diagnostics
   - Load testing capabilities

## Quick Start (5 Minutes)

### 1. Verify Prerequisites

```bash
# Check Kubernetes and namespaces
kubectl cluster-info
kubectl get namespace selenium-grid
kubectl get namespace keda

# Verify KEDA is installed
kubectl get pods -n keda | grep keda-operator
```

### 2. Deploy KEDA ScaledObjects

```bash
# Navigate to the kubernetes directory
cd /Users/bvk/Downloads/e2e-testing-agent/browser-pool/kubernetes

# Deploy the manifests
kubectl apply -f keda-scaledobject.yaml

# Verify deployment
kubectl get scaledobjects -n selenium-grid
```

### 3. Validate Deployment

```bash
# Run validation script
bash keda-validation-script.sh all

# Or run specific checks
bash keda-validation-script.sh kubernetes
bash keda-validation-script.sh keda
bash keda-validation-script.sh scaledobjects
```

### 4. Monitor Scaling

```bash
# Watch HPA status
kubectl get hpa -n selenium-grid -w

# View current replicas
kubectl get deployment -n selenium-grid -l app=selenium-grid-selenium-node-chrome

# Check scaling events
kubectl get events -n selenium-grid --sort-by='.lastTimestamp' | tail -20
```

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│         KEDA ScaledObject                       │
├─────────────────────────────────────────────────┤
│  Monitors:                                      │
│  • Prometheus metrics (primary)                 │
│  • CPU utilization (fallback)                   │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│    Kubernetes HPA (created by KEDA)             │
├─────────────────────────────────────────────────┤
│  Scales based on metric thresholds              │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│   Browser Node Deployments                      │
├─────────────────────────────────────────────────┤
│  • selenium-grid-selenium-node-chrome   (2-20)  │
│  • selenium-grid-selenium-node-firefox  (1-10)  │
│  • selenium-grid-selenium-node-edge     (0-15)  │
└─────────────────────────────────────────────────┘
```

## Scaling Configuration Summary

| Metric | Chrome | Firefox | Edge |
|--------|--------|---------|------|
| Min Replicas | 2 | 1 | 0 |
| Max Replicas | 20 | 10 | 15 |
| Primary Trigger | Session Queue | Session Queue | Session Queue |
| Secondary Trigger | Slot Utilization (20%) | Slot Utilization (15%) | Slot Utilization (10%) |
| Scale Up Speed | Fast (5 pods/30s) | Moderate (2 pods/45s) | Very Fast (10 pods/15s) |
| Scale Down Speed | Conservative (2 pods/60s) | Very Conservative (1 pod/120s) | Aggressive (3 pods/60s) |

## Key Features

### 1. Multi-Metric Scaling
- **Session Queue Length**: Primary metric for responsive scaling
- **Slot Utilization Ratio**: Secondary metric for predictive scaling
- **CPU Utilization**: Fallback metric if Prometheus unavailable

### 2. Intelligent Thresholds
- Chrome: Scale up at 80% utilization
- Firefox: Scale up at 85% utilization (more conservative)
- Edge: Scale up at 90% utilization (cost-optimized)

### 3. Graceful Degradation
- Fallback to CPU metrics if Prometheus fails
- Falls back to 2 replicas if all metrics fail
- Automatically recovers when metrics available

### 4. Cost Optimization
- Edge nodes scale to 0 when not needed
- Conservative scale-down prevents resource thrashing
- Maximum replica limits prevent runaway costs

### 5. High Availability
- PodDisruptionBudgets maintain minimum replicas during maintenance
- Multiple scaling policies (pods/percentage) for flexibility
- Stabilization windows prevent rapid fluctuations

## Prometheus Metrics Required

The Selenium Grid hub must expose these metrics:

```
# Queue metrics (per browser type)
selenium_sessions_queued{node="chrome"}
selenium_sessions_queued{node="firefox"}
selenium_sessions_queued{node="edge"}

# Slot metrics (per browser type)
selenium_node_slots_available{node="chrome"}
selenium_node_slots_used{node="chrome"}
# ... repeat for firefox and edge
```

## Deployment Checklist

- [ ] Kubernetes cluster running and accessible
- [ ] KEDA operator installed (`helm install keda kedacore/keda --namespace keda --create-namespace`)
- [ ] Prometheus deployed and scraping Selenium Grid metrics
- [ ] Selenium Grid 4 deployed with metrics exposition enabled
- [ ] `keda-scaledobject.yaml` applied
- [ ] Validation script passes all checks
- [ ] HPA objects created: `kubectl get hpa -n selenium-grid`
- [ ] Browser nodes at minimum replicas: `kubectl get deployments -n selenium-grid`

## Common Commands

### Deploy
```bash
# Deploy KEDA autoscaling
kubectl apply -f keda-scaledobject.yaml

# Deploy with Prometheus alert rules
kubectl apply -f keda-prometheus-queries.yaml
```

### Monitor
```bash
# Watch autoscaler activity
kubectl get hpa -n selenium-grid -w

# View scaler status
kubectl describe scaledobject selenium-chrome-nodes-scaler -n selenium-grid

# Check metrics
kubectl logs -n keda deployment/keda-operator -f | grep selenium
```

### Troubleshoot
```bash
# Run validation
bash keda-validation-script.sh all

# Check specific component
bash keda-validation-script.sh scaledobjects
bash keda-validation-script.sh prometheus

# View events
kubectl get events -n selenium-grid --sort-by='.lastTimestamp'
```

### Pause/Resume
```bash
# Pause autoscaling (keep current replicas)
kubectl annotate scaledobject selenium-chrome-nodes-scaler \
  -n selenium-grid keda.sh/paused=true --overwrite

# Resume autoscaling
kubectl annotate scaledobject selenium-chrome-nodes-scaler \
  -n selenium-grid keda.sh/paused=false --overwrite
```

### Update Configuration
```bash
# Edit configuration
kubectl edit scaledobject selenium-chrome-nodes-scaler -n selenium-grid

# Apply new manifest
kubectl apply -f keda-scaledobject.yaml
```

## Monitoring Dashboard

### Grafana Dashboard Queries

Create a dashboard with these panels:

**Panel 1: Current Replicas**
```
keda_scaler_active{scaled_object="selenium-chrome-nodes-scaler"}
```

**Panel 2: Session Queue**
```
ceil(selenium_sessions_queued{node="chrome"})
```

**Panel 3: Slot Utilization**
```
(selenium_node_slots_used{node="chrome"} / (selenium_node_slots_available{node="chrome"} + selenium_node_slots_used{node="chrome"})) * 100
```

**Panel 4: CPU Usage**
```
avg(rate(container_cpu_usage_seconds_total{pod=~"selenium-grid-selenium-node-chrome.*"}[5m]))
```

## Troubleshooting Guide

### Scaling Not Triggering
1. Check KEDA operator: `kubectl get pods -n keda`
2. Verify Prometheus: `kubectl get pods -n prometheus | grep prometheus`
3. Verify metrics: Check Prometheus UI for `selenium_sessions_queued`
4. Check scaler status: `kubectl describe scaledobject -n selenium-grid`

### Scaling Too Aggressive
1. Increase stabilization window: Edit `scaleDown.stabilizationWindowSeconds`
2. Reduce scale-up pod count: Edit `policies[0].value` in `scaleUp`
3. Increase threshold: Edit `triggers[0].threshold`

### High Costs
1. Lower max replicas: Edit `maxReplicaCount`
2. Increase scale-down aggressiveness: Reduce `stabilizationWindowSeconds`
3. Set stricter thresholds for scale-up: Increase `threshold` values

### Pods Getting Evicted (OOM)
1. Check memory limits: `kubectl top pods -n selenium-grid`
2. Increase pod limits: Edit deployment resource limits
3. Reduce pods per node: Adjust node affinity

## File Reference

| File | Purpose | Size |
|------|---------|------|
| `keda-scaledobject.yaml` | Main autoscaling manifests | 543 lines |
| `KEDA-SCALEDOBJECT-GUIDE.md` | Comprehensive deployment guide | ~500 lines |
| `keda-prometheus-queries.yaml` | Prometheus config and queries | ~600 lines |
| `keda-validation-script.sh` | Validation and troubleshooting | ~350 lines |
| `KEDA-QUICKSTART.md` | This file | Quick reference |

## Next Steps

1. **Deploy**: `kubectl apply -f keda-scaledobject.yaml`
2. **Validate**: `bash keda-validation-script.sh all`
3. **Monitor**: Set up Prometheus dashboard with queries from `keda-prometheus-queries.yaml`
4. **Test**: Generate load and verify scaling works
5. **Tune**: Adjust thresholds based on your test load patterns

## Support Resources

- KEDA Documentation: https://keda.sh/
- Prometheus PromQL: https://prometheus.io/docs/prometheus/latest/querying/
- Selenium Grid Metrics: https://www.selenium.dev/documentation/grid/
- Kubernetes HPA: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/

## Additional Documentation

For detailed information, see:
- **KEDA-SCALEDOBJECT-GUIDE.md** - Complete deployment and troubleshooting guide
- **keda-prometheus-queries.yaml** - All monitoring queries and alert rules
- **keda-validation-script.sh** - Automated validation and health checks

## Support

If you encounter issues:

1. Run validation: `bash keda-validation-script.sh all`
2. Check logs: `kubectl logs -n keda deployment/keda-operator -f`
3. Review guide: See KEDA-SCALEDOBJECT-GUIDE.md troubleshooting section
4. Test Prometheus: Verify metrics in Prometheus UI at http://prometheus:9090

---

**Created**: 2026-01-16
**Version**: 1.0
**Status**: Production Ready
