# Argus Browser Pool - Hetzner Kubernetes

Scalable browser automation infrastructure on Hetzner Cloud using Kubernetes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HETZNER KUBERNETES CLUSTER                    │
│                        (K3s via Terraform)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Browser Pool Manager                    │    │
│  │           (Session allocation, health checks)            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │
│  │ Browser Pod │      │ Browser Pod │      │ Browser Pod │ ... │
│  │ (Playwright)│      │ (Playwright)│      │ (Playwright)│     │
│  └─────────────┘      └─────────────┘      └─────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Traefik Ingress │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  MCP Server/API   │
                    └───────────────────┘
```

## Quick Start

### Prerequisites

- [Terraform](https://terraform.io) >= 1.0
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Hetzner Cloud Account](https://www.hetzner.com/cloud)
- Hetzner API Token

### 1. Deploy Kubernetes Cluster

```bash
cd terraform

# Copy and configure variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your Hetzner API token

# Initialize and deploy
terraform init
terraform plan
terraform apply
```

### 2. Deploy Browser Pool

```bash
# Get kubeconfig
export KUBECONFIG=$(terraform output -raw kubeconfig_path)

# Deploy browser pool
kubectl apply -f ../kubernetes/
```

### 3. Verify Deployment

```bash
# Check pods
kubectl get pods -n browser-pool

# Check service
kubectl get svc -n browser-pool

# Test health endpoint
curl http://<LOAD_BALANCER_IP>/health
```

## Configuration

### Scaling

Edit `kubernetes/hpa.yaml` to adjust auto-scaling:

```yaml
minReplicas: 10      # Minimum browser pods
maxReplicas: 1000    # Maximum browser pods (adjust based on budget)
targetCPUUtilization: 70
```

### Browser Resources

Edit `kubernetes/browser-deployment.yaml`:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

## Cost Estimation

| Scale | Pods | Node Type | Nodes | Monthly Cost |
|-------|------|-----------|-------|--------------|
| Dev | 10 | CX21 | 3 | ~€30 |
| Small | 50 | CX31 | 5 | ~€100 |
| Medium | 200 | CX41 | 15 | ~€450 |
| Large | 1000 | CX51 | 50 | ~€2,000 |
| Enterprise | 10000 | CCX* | 200+ | ~€15,000 |

*CCX = Dedicated CPU instances for production

## API Endpoints

The browser pool manager exposes these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/observe` | POST | Discover page elements (MCP compatible) |
| `/act` | POST | Execute browser action (MCP compatible) |
| `/test` | POST | Run multi-step test |
| `/session` | POST | Create browser session |
| `/session/:id` | DELETE | Close browser session |
| `/metrics` | GET | Prometheus metrics |

## Monitoring

Prometheus metrics available at `/metrics`:

- `browser_pool_sessions_total` - Total sessions created
- `browser_pool_sessions_active` - Currently active sessions
- `browser_pool_action_duration_seconds` - Action execution time
- `browser_pool_errors_total` - Error count by type
