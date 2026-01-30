# Helm Installation Guide

Complete guide for deploying Argus Enterprise using Helm.

## Prerequisites

```bash
# Kubernetes 1.25+
kubectl version --client

# Helm 3.10+
helm version

# Cluster access
kubectl cluster-info
```

## Installation Methods

### Method 1: From Helm Repository (Recommended)

```bash
# Add Argus repository
helm repo add argus https://charts.heyargus.ai
helm repo update

# View available versions
helm search repo argus/argus-enterprise --versions

# Install latest version
helm install argus argus/argus-enterprise \
  --namespace argus \
  --create-namespace \
  -f values.yaml
```

### Method 2: From Local Directory

```bash
# Clone repository
git clone https://github.com/raphaenterprises-ai/argus-e2e-testing-agent.git
cd argus-e2e-testing-agent

# Update dependencies
helm dependency update ./helm/argus-enterprise

# Install from local chart
helm install argus ./helm/argus-enterprise \
  --namespace argus \
  --create-namespace \
  -f values.yaml
```

## Configuration Files

### Minimal Production Setup

```yaml
# values-production.yaml
global:
  domain: "argus.company.com"
  storageClass: "standard"  # Your storage class

brain:
  replicas: 2
  env:
    LLM_PROVIDER: "anthropic"
  secrets:
    anthropicApiKey: "sk-ant-xxx"
  resources:
    requests:
      cpu: "1000m"
      memory: "2Gi"
    limits:
      cpu: "2000m"
      memory: "4Gi"

postgresql:
  auth:
    password: "secure-password-here"
  primary:
    persistence:
      size: 100Gi

redis:
  auth:
    password: "secure-password-here"

minio:
  auth:
    rootPassword: "secure-password-here"

ingress:
  enabled: true
  className: "nginx"
  hosts:
    brain: "argus-api.company.com"
    mcp: "argus-mcp.company.com"
  tls:
    enabled: true
    secretName: "argus-tls"
```

### Air-Gap Deployment

```yaml
# values-airgap.yaml
global:
  domain: "argus.internal"

brain:
  replicas: 2
  env:
    LLM_PROVIDER: "ollama"

# Enable local LLM
ollama:
  enabled: true
  models:
    - "llama3.1:70b"
    - "codellama:34b"
  persistence:
    enabled: true
    size: 200Gi
  resources:
    limits:
      nvidia.com/gpu: 1

# Disable external dependencies
postgresql:
  enabled: true

redis:
  enabled: true

minio:
  enabled: true
```

### External Databases

```yaml
# values-external-db.yaml

# Use external PostgreSQL
postgresql:
  enabled: false
  external:
    host: "postgres.company.com"
    port: 5432
    database: "argus"
    username: "argus"
    existingSecret: "postgres-credentials"
    # Secret should contain key: postgresql-password

# Use external Redis
redis:
  enabled: false
  external:
    host: "redis.company.com"
    port: 6379
    existingSecret: "redis-credentials"
    # Secret should contain key: redis-password

# Use external S3/MinIO
minio:
  enabled: false
  external:
    endpoint: "s3.company.com"
    accessKey: "access-key"
    secretKey: "secret-key"
    bucket: "argus-artifacts"
    region: "us-east-1"
    useSSL: true
```

### High Availability

```yaml
# values-ha.yaml
brain:
  replicas: 3
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80
  podDisruptionBudget:
    minAvailable: 2

postgresql:
  architecture: replication
  primary:
    persistence:
      size: 200Gi
  readReplicas:
    replicaCount: 2
    persistence:
      size: 200Gi

redis:
  architecture: replication
  replica:
    replicaCount: 2
```

## Installation Commands

### Basic Install

```bash
helm install argus argus/argus-enterprise \
  --namespace argus \
  --create-namespace
```

### With Custom Values

```bash
helm install argus argus/argus-enterprise \
  --namespace argus \
  --create-namespace \
  -f values-production.yaml
```

### Multiple Value Files

```bash
helm install argus argus/argus-enterprise \
  --namespace argus \
  --create-namespace \
  -f values-production.yaml \
  -f values-ha.yaml \
  -f values-secrets.yaml
```

### Set Individual Values

```bash
helm install argus argus/argus-enterprise \
  --namespace argus \
  --create-namespace \
  --set brain.replicas=3 \
  --set brain.secrets.anthropicApiKey=sk-ant-xxx \
  --set ingress.enabled=true
```

## Verification

### Check Pod Status

```bash
# All pods should be Running
kubectl get pods -n argus

# Expected output:
# NAME                                READY   STATUS    RESTARTS   AGE
# argus-brain-xxx-xxx                 1/1     Running   0          2m
# argus-mcp-xxx-xxx                   1/1     Running   0          2m
# argus-postgresql-0                  1/1     Running   0          2m
# argus-redis-master-0                1/1     Running   0          2m
# argus-minio-xxx-xxx                 1/1     Running   0          2m
# argus-selenium-hub-xxx-xxx          1/1     Running   0          2m
# argus-selenium-chrome-node-xxx-xxx  1/1     Running   0          2m
```

### Health Checks

```bash
# Brain API health
kubectl exec deploy/argus-brain -n argus -- \
  curl -s localhost:8000/health | jq .

# MCP Server health
kubectl exec deploy/argus-mcp -n argus -- \
  curl -s localhost:3000/health

# Database connectivity
kubectl exec deploy/argus-brain -n argus -- \
  psql "$DATABASE_URL" -c "SELECT 1"
```

### View Installation Notes

```bash
helm get notes argus -n argus
```

## Upgrade

### View Current Values

```bash
helm get values argus -n argus -o yaml > current-values.yaml
```

### Upgrade Chart

```bash
# Update repository
helm repo update

# Dry-run to preview changes
helm upgrade argus argus/argus-enterprise \
  --namespace argus \
  -f values-production.yaml \
  --dry-run

# Apply upgrade
helm upgrade argus argus/argus-enterprise \
  --namespace argus \
  -f values-production.yaml
```

### Rollback

```bash
# View revision history
helm history argus -n argus

# Rollback to previous revision
helm rollback argus 1 -n argus
```

## Uninstall

### Remove Helm Release

```bash
helm uninstall argus -n argus
```

### Clean Up PVCs (Data Loss!)

```bash
# List PVCs
kubectl get pvc -n argus

# Delete all PVCs
kubectl delete pvc -l app.kubernetes.io/instance=argus -n argus
```

### Delete Namespace

```bash
kubectl delete namespace argus
```

## Chart Values Reference

See [configuration.md](configuration.md) for complete values reference.

### Key Configuration Sections

| Section | Description |
|---------|-------------|
| `global` | Domain, storage class, image settings |
| `brain` | Main API backend configuration |
| `mcp` | MCP Server configuration |
| `postgresql` | PostgreSQL database settings |
| `redis` | Redis cache settings |
| `minio` | Object storage settings |
| `seleniumGrid` | Browser automation settings |
| `ollama` | Local LLM settings (optional) |
| `ingress` | Ingress controller settings |
| `networkPolicies` | Network security policies |
| `metrics` | Prometheus metrics settings |
| `serviceAccount` | Kubernetes service account |
