# KEDA for Selenium Grid 4 - Complete Package Index

This directory contains a complete, production-ready KEDA setup for autoscaling Selenium Grid on Kubernetes.

## Files by Purpose

### 📖 Documentation Files

1. **KEDA-README.md** (START HERE)
   - Overview of all KEDA files
   - Quick start guide
   - Scaling strategies comparison
   - Configuration reference
   - Best practices
   - Troubleshooting guide

2. **keda-setup.md**
   - Detailed architecture explanation
   - Step-by-step KEDA installation
   - Available scalers overview
   - Best practices for scaling
   - Common pitfalls and solutions
   - Testing procedures

3. **KEDA-DEPLOYMENT-GUIDE.md**
   - Complete deployment walkthrough
   - Installation steps for each component
   - Configuration examples by use case
   - Monitoring setup instructions
   - Load testing procedures
   - Performance tuning

### 🔧 Configuration Files (YAML)

#### Tier 1: Core Installation

4. **keda-values.yaml**
   - Helm values for KEDA operator installation
   - Operator configuration
   - Webhook settings
   - Service account setup
   - Resource limits
   - Pod disruption budget

#### Tier 2: Selenium Grid Scaling

5. **scaledobject-selenium-grid.yaml**
   - KEDA ScaledObjects for Chrome nodes (primary)
   - KEDA ScaledObjects for Firefox nodes (conservative)
   - Kubernetes HPA for Edge nodes (fallback)
   - Scale-up/scale-down policies
   - External scaler triggers
   - 3 different scaling strategies in one file

#### Tier 3: Advanced Features

6. **selenium-keda-scaler.yaml**
   - External scaler deployment (gRPC service)
   - Metrics provider configuration
   - Health checks and monitoring
   - Service definition
   - Pod Disruption Budget for scaler
   - **Includes embedded Python code**

7. **keda-auth-config.yaml**
   - TriggerAuthentication for Selenium Grid
   - Basic auth credentials
   - TLS certificate configuration
   - Prometheus authentication
   - Nginx reverse proxy configuration
   - htpasswd for basic auth

8. **keda-prometheus-advanced.yaml**
   - Prometheus ServiceMonitor
   - Prometheus alerting rules
   - Multi-metric ScaledObjects
   - Time-based (cron) scaling examples
   - Browser-specific scaling (Firefox conservative)
   - Prometheus query reference
   - Grafana dashboard ConfigMap

#### Tier 4: Container Image

9. **Dockerfile.keda-scaler**
   - Docker image for external scaler
   - Python dependencies
   - Health check configuration
   - Production-ready setup

## Setup Workflow

### Minimal Setup (5 minutes)

```bash
# 1. Install KEDA
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda -n keda --create-namespace -f keda-values.yaml

# 2. Install Selenium Grid
helm repo add selenium https://www.selenium.dev/docker-selenium
helm install selenium-grid selenium/selenium-grid -n selenium-grid --create-namespace

# 3. Apply scaling
kubectl apply -f scaledobject-selenium-grid.yaml
```

### Full Setup (30 minutes, recommended)

```bash
# All steps above plus:

# 4. Deploy external scaler for better metrics
kubectl apply -f selenium-keda-scaler.yaml

# 5. Optional: Setup Prometheus for advanced metrics
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace

# 6. Apply Prometheus-based scaling
kubectl apply -f keda-prometheus-advanced.yaml

# 7. Optional: Configure authentication
kubectl apply -f keda-auth-config.yaml
```

## Scaling Strategies Included

### Strategy 1: External Scaler (Recommended)
- **File:** `scaledobject-selenium-grid.yaml` (primary trigger)
- **Implementation:** `selenium-keda-scaler.yaml`
- **Use case:** Direct Selenium Grid metrics
- **Pros:** Simple, reliable, no external infrastructure
- **Cons:** Requires scaler service

### Strategy 2: Prometheus Multi-Metric
- **File:** `keda-prometheus-advanced.yaml`
- **Use case:** Advanced metrics aggregation
- **Includes:** Queue length, available slots, error rate, session duration
- **Pros:** Flexible, integrates with monitoring
- **Cons:** Requires Prometheus, more complex

### Strategy 3: Time-Based (Cron)
- **File:** `keda-prometheus-advanced.yaml` (second ScaledObject)
- **Use case:** Scale by business hours
- **Example:** 9-17 UTC scale-up, nights scale-down
- **Pros:** Predictable, cost-effective
- **Cons:** Requires fixed schedule

### Strategy 4: Browser-Specific
- **Chrome:** Aggressive scaling (scaledobject-selenium-grid.yaml)
- **Firefox:** Conservative scaling (scaledobject-selenium-grid.yaml)
- **Edge:** Manual via HPA
- **Tailored:** Each browser type has optimized settings

## Configuration Examples

### Quick Reference Table

| File | Purpose | Required | Difficulty | Setup Time |
|------|---------|----------|-----------|-----------|
| keda-values.yaml | KEDA installation | Yes | Easy | 5 min |
| scaledobject-selenium-grid.yaml | Basic scaling | Yes | Easy | 2 min |
| selenium-keda-scaler.yaml | External scaler | Optional | Medium | 10 min |
| keda-prometheus-advanced.yaml | Advanced metrics | Optional | Hard | 15 min |
| keda-auth-config.yaml | Authentication | Optional | Medium | 10 min |
| Dockerfile.keda-scaler | Custom scaler image | Optional | Medium | 5 min |

## Key Features

### 1. Multi-Browser Scaling
- Chrome: 2-50 replicas (aggressive)
- Firefox: 1-20 replicas (conservative)
- Edge: Manual scaling via HPA

### 2. Failover & Resilience
- Fallback replicas if scaler fails
- Minimum replica guarantees
- Pod Disruption Budgets

### 3. Advanced Metrics
- Session queue length (primary)
- Available slots ratio
- Error rate monitoring
- Custom HTTP/Prometheus queries

### 4. Time-Based Scaling
- Business hours scaling
- Night-time cost optimization
- Cron-based configuration

### 5. Multi-Trigger Cooperation
- Multiple metrics evaluated together
- OR logic (any trigger scales)
- Configurable policies for scale-up/down

## Deployment Decision Tree

```
START
  ↓
Is Selenium Grid already running?
  → NO: Install via Helm first
  → YES: Continue
  ↓
Want simplest setup?
  → YES: Use scaledobject-selenium-grid.yaml only
  → NO: Continue
  ↓
Want better metrics?
  → YES: Deploy selenium-keda-scaler.yaml (external scaler)
  → NO: Use CPU/memory triggers
  ↓
Want advanced monitoring?
  → YES: Install Prometheus + use keda-prometheus-advanced.yaml
  → NO: Done
  ↓
Need authentication?
  → YES: Apply keda-auth-config.yaml
  → NO: Done
  ↓
Ready to test!
```

## File Sizes & Dependencies

```
KEDA-README.md
├── keda-setup.md (reference)
├── KEDA-DEPLOYMENT-GUIDE.md (reference)
└── Configuration files:
    ├── keda-values.yaml (Helm)
    │   └── Deploys KEDA operator
    │
    ├── scaledobject-selenium-grid.yaml (KEDA resources)
    │   ├── Requires: KEDA, Selenium Grid
    │   └── Optional: selenium-keda-scaler.yaml
    │
    ├── selenium-keda-scaler.yaml (Deployment)
    │   ├── Python service
    │   ├── Dockerfile.keda-scaler (for custom image)
    │   └── ConfigMap with scaler code
    │
    ├── keda-prometheus-advanced.yaml (Monitoring)
    │   ├── Requires: Prometheus
    │   └── Advanced ScaledObjects
    │
    └── keda-auth-config.yaml (Security)
        ├── TriggerAuthentication
        ├── Secrets for credentials
        └── Nginx proxy config
```

## Performance Profiles

### Development/Testing
```yaml
minReplicas: 1, maxReplicas: 5
pollingInterval: 30s, cooldownPeriod: 60s
Threshold: 20 (conservative)
Cost: ~$5-10/day
```

### Production Small
```yaml
minReplicas: 2, maxReplicas: 50
pollingInterval: 15s, cooldownPeriod: 60s
Threshold: 10 (balanced)
Cost: ~$50-100/day
```

### Production Large
```yaml
minReplicas: 10, maxReplicas: 500
pollingInterval: 10s, cooldownPeriod: 30s
Threshold: 2 (aggressive)
Cost: ~$500-1000+/day
```

## Validation Checklist

Before going to production:

- [ ] KEDA operator installed: `kubectl get pods -n keda`
- [ ] Selenium Grid running: `kubectl get pods -n selenium-grid`
- [ ] ScaledObject created: `kubectl get scaledobject -n selenium-grid`
- [ ] External scaler healthy: `kubectl logs -n selenium-grid -l app=selenium-keda-scaler`
- [ ] Selenium Hub responding: `curl http://selenium-hub:4444/status`
- [ ] HPA created: `kubectl get hpa -n selenium-grid`
- [ ] Pod Disruption Budgets set: `kubectl get pdb -n selenium-grid`
- [ ] Monitoring configured: `kubectl get servicemonitor -n selenium-grid`
- [ ] Alerts created: `kubectl get prometheusrule -n selenium-grid`
- [ ] Load test passed: Scale test with 100+ concurrent sessions

## Troubleshooting Quick Links

- ScaledObject not active: See `keda-setup.md` → Troubleshooting
- Pods not scaling: See `KEDA-DEPLOYMENT-GUIDE.md` → Common Issues
- Metrics not flowing: See `KEDA-README.md` → Monitoring section
- Authentication issues: See `keda-auth-config.yaml` → Comments

## Support & References

- **KEDA Docs:** https://keda.sh/docs/
- **Selenium Grid:** https://www.selenium.dev/documentation/grid/
- **Kubernetes HPA:** https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- **Prometheus:** https://prometheus.io/docs/prometheus/latest/querying/basics/

## Next Steps

1. **Read:** Start with `KEDA-README.md`
2. **Plan:** Choose your scaling strategy
3. **Deploy:** Follow `KEDA-DEPLOYMENT-GUIDE.md`
4. **Configure:** Customize `scaledobject-selenium-grid.yaml` for your needs
5. **Monitor:** Set up alerts using `keda-prometheus-advanced.yaml`
6. **Test:** Run load tests (see deployment guide)
7. **Optimize:** Tune thresholds based on real usage

---

**Created:** January 2026
**Version:** 1.0
**Tested with:** KEDA 2.15, Selenium Grid 4.27, Kubernetes 1.28+
