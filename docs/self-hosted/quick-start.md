# Quick Start Guide

Get Argus Enterprise running in 15 minutes.

## Option 1: Helm (Recommended for Production)

### Prerequisites

```bash
# Verify Kubernetes cluster
kubectl cluster-info

# Verify Helm
helm version
```

### Install Argus

```bash
# Add the Argus Helm repository
helm repo add argus https://charts.heyargus.ai
helm repo update

# Create namespace
kubectl create namespace argus

# Install with default values
helm install argus argus/argus-enterprise \
  --namespace argus \
  --set brain.secrets.anthropicApiKey=sk-ant-xxx
```

### Verify Installation

```bash
# Wait for pods to be ready (2-3 minutes)
kubectl get pods -n argus -w

# Check health
kubectl exec deploy/argus-brain -n argus -- curl -s localhost:8000/health
```

### Access Services

```bash
# Port-forward to access locally
kubectl port-forward svc/argus-brain 8000:8000 -n argus &
kubectl port-forward svc/argus-mcp 3000:3000 -n argus &

# Test the API
curl http://localhost:8000/health
```

## Option 2: Docker Compose (Development/Single Node)

### Prerequisites

```bash
# Verify Docker
docker --version

# Verify Docker Compose
docker compose version
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/raphaenterprises-ai/argus-e2e-testing-agent.git
cd argus-e2e-testing-agent

# Start MCP Server with all dependencies
cd argus-mcp-server/standalone
docker compose up -d
```

### Verify Services

```bash
# Check running containers
docker compose ps

# View logs
docker compose logs -f argus-mcp

# Test health endpoint
curl http://localhost:3000/health
```

## Connect to AI Assistant

### Claude Code

Add to your `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "argus": {
      "command": "curl",
      "args": ["-N", "http://localhost:3000/sse"]
    }
  }
}
```

### Cursor / Windsurf

Add to your MCP settings:

```json
{
  "servers": {
    "argus": {
      "url": "http://localhost:3000/sse"
    }
  }
}
```

### Verify Connection

In your AI assistant, ask:

```
What Argus tools are available?
```

You should see a list of 73+ MCP tools for test management.

## Run Your First Test

### 1. Analyze a Codebase

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my-project",
    "repo_url": "https://github.com/your-org/your-app",
    "branch": "main"
  }'
```

### 2. Generate Test Plan

```bash
curl -X POST http://localhost:8000/api/v1/plan \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my-project",
    "target_url": "http://your-app:3000",
    "test_types": ["ui", "api"]
  }'
```

### 3. Execute Tests

```bash
curl -X POST http://localhost:8000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my-project",
    "plan_id": "plan_xxx",
    "parallel": 3
  }'
```

## Common First Steps

### Set Up External Database

For production, use an external PostgreSQL:

```yaml
# values-production.yaml
postgresql:
  enabled: false
  external:
    host: "postgres.your-company.com"
    database: "argus"
    username: "argus"
    existingSecret: "postgres-credentials"
```

### Enable TLS/Ingress

```yaml
# values-production.yaml
ingress:
  enabled: true
  className: "nginx"
  hosts:
    brain: "argus-api.your-company.com"
    mcp: "argus-mcp.your-company.com"
  tls:
    enabled: true
    secretName: "argus-tls"
```

### Configure API Key

```bash
# Using Kubernetes secret
kubectl create secret generic argus-secrets \
  --from-literal=anthropic-api-key=sk-ant-xxx \
  -n argus

# Or in values.yaml
brain:
  secrets:
    anthropicApiKey: "sk-ant-xxx"
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod -l app.kubernetes.io/name=argus -n argus

# Check logs
kubectl logs -l app.kubernetes.io/name=argus-brain -n argus
```

### Database Connection Failed

```bash
# Verify PostgreSQL is ready
kubectl exec deploy/argus-brain -n argus -- \
  psql "$DATABASE_URL" -c "SELECT 1"
```

### MCP Connection Issues

```bash
# Test SSE endpoint directly
curl -N http://localhost:3000/sse

# Check MCP logs
kubectl logs deploy/argus-mcp -n argus
```

## Next Steps

- [Full Configuration Guide](configuration.md)
- [Helm Installation Details](helm-installation.md)
- [Network Configuration](networking.md)
- [Air-Gap Deployment](llm-configuration.md#air-gap)
