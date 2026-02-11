# Argus Enterprise Helm Chart

Deploy the complete Argus E2E Testing stack on Kubernetes for self-hosted and air-gap environments.

## Overview

This Helm chart deploys:
- **Argus Brain** - FastAPI backend with LangGraph orchestration
- **Argus MCP Server** - Model Context Protocol for AI IDEs
- **PostgreSQL + pgvector** - Vector embeddings storage
- **Redis/Valkey** - Caching and sessions
- **MinIO** - S3-compatible object storage
- **Selenium Grid** - Browser automation
- **Ollama** (optional) - Local LLM inference

## Prerequisites

- Kubernetes 1.25+
- Helm 3.10+
- PV provisioner (for persistent storage)
- (Optional) NVIDIA GPU operator for local LLM

## Quick Start

```bash
# Add the Argus Helm repository
helm repo add argus https://charts.skopaq.ai
helm repo update

# Install with default values
helm install argus argus/argus-enterprise \
  --namespace argus \
  --create-namespace

# Or from local directory
helm install argus ./helm/argus-enterprise \
  --namespace argus \
  --create-namespace
```

## Configuration

### Minimal Production Setup

```yaml
# values-production.yaml
global:
  domain: "argus.company.com"

brain:
  replicas: 3
  env:
    LLM_PROVIDER: "anthropic"
  secrets:
    anthropicApiKey: "sk-ant-xxx"

postgresql:
  auth:
    password: "secure-password"

redis:
  auth:
    password: "secure-password"

minio:
  auth:
    rootPassword: "secure-password"

ingress:
  enabled: true
  className: "nginx"
  tls:
    enabled: true
```

Install with custom values:
```bash
helm install argus ./helm/argus-enterprise \
  -f values-production.yaml \
  --namespace argus
```

### Air-Gap Deployment (Local LLM)

```yaml
# values-airgap.yaml
brain:
  env:
    LLM_PROVIDER: "ollama"

ollama:
  enabled: true
  models:
    - "llama3.1:8b"
  persistence:
    size: 100Gi
  resources:
    limits:
      nvidia.com/gpu: 1  # If GPU available
```

### Using External Databases

```yaml
# values-external-db.yaml
postgresql:
  enabled: false
  external:
    host: "postgres.company.com"
    database: "argus"
    username: "argus"
    existingSecret: "postgres-credentials"

redis:
  enabled: false
  external:
    host: "redis.company.com"
    existingSecret: "redis-credentials"

minio:
  enabled: false
  external:
    endpoint: "s3.company.com"
    accessKey: "xxx"
    secretKey: "xxx"
    bucket: "argus-artifacts"
```

## Components

### Argus Brain

The main API backend. Configure resources based on load:

| Size | Replicas | CPU | Memory |
|------|----------|-----|--------|
| Small | 1 | 500m | 1Gi |
| Medium | 2 | 1000m | 2Gi |
| Large | 3+ | 2000m | 4Gi |

### MCP Server

Exposes Argus capabilities to AI coding assistants (Claude Code, Cursor, Windsurf).

SSE endpoint: `http://<mcp-service>:3000/sse`

### Selenium Grid

Browser automation for E2E testing. Scale Chrome nodes based on test parallelism:

```yaml
seleniumGrid:
  chrome:
    replicas: 5  # Run 5 tests in parallel
```

### Ollama (Local LLM)

For air-gap deployments without external API access:

```yaml
ollama:
  enabled: true
  models:
    - "llama3.1:8b"    # 4.7GB, general purpose
    - "llama3.1:70b"   # 40GB, higher quality
  resources:
    limits:
      nvidia.com/gpu: 1
```

## Network Policies

Network policies are enabled by default for security. To allow traffic from specific namespaces:

```yaml
networkPolicies:
  enabled: true
  allowedNamespaces:
    - ingress-nginx
    - monitoring
```

## Monitoring

Enable Prometheus metrics:

```yaml
metrics:
  enabled: true
  serviceMonitor:
    enabled: true  # Requires Prometheus Operator
```

## Upgrading

```bash
helm upgrade argus ./helm/argus-enterprise \
  -f values.yaml \
  --namespace argus
```

## Uninstalling

```bash
helm uninstall argus --namespace argus

# Clean up PVCs (data will be lost!)
kubectl delete pvc -l app.kubernetes.io/instance=argus -n argus
```

## Troubleshooting

### Check pod status
```bash
kubectl get pods -n argus
```

### View logs
```bash
kubectl logs -f deploy/argus-brain -n argus
kubectl logs -f deploy/argus-mcp -n argus
```

### Health check
```bash
kubectl exec deploy/argus-brain -n argus -- curl localhost:8000/health
```

### Database connection
```bash
kubectl exec deploy/argus-brain -n argus -- \
  psql "$DATABASE_URL" -c "SELECT 1"
```

## Support

- Documentation: https://docs.skopaq.ai/self-hosted
- Issues: https://github.com/raphaenterprises-ai/argus-e2e-testing-agent/issues
- Email: support@skopaq.ai
