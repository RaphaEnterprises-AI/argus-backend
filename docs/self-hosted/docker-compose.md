# Docker Compose Deployment

Deploy Skopaq Enterprise on a single node using Docker Compose.

## When to Use Docker Compose

- Development and testing environments
- Small-scale deployments (< 10 concurrent tests)
- Quick evaluation of Skopaq features
- CI/CD pipelines

For production deployments, see [Helm Installation](helm-installation.md).

## Prerequisites

```bash
# Docker 24+
docker --version

# Docker Compose 2.20+
docker compose version

# Minimum resources
# 8 CPU cores, 32GB RAM, 200GB storage
```

## Quick Start

### Clone and Configure

```bash
# Clone repository
git clone https://github.com/raphaenterprises-ai/skopaq-e2e-testing-agent.git
cd skopaq-e2e-testing-agent/skopaq-mcp-server/standalone

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### Configure Environment

```bash
# .env file
# Required
ANTHROPIC_API_KEY=sk-ant-xxx

# MinIO storage
MINIO_ENDPOINT=http://minio:9000
MINIO_ACCESS_KEY=argus
MINIO_SECRET_KEY=argus-secret-key
MINIO_BUCKET=argus-artifacts

# Redis
REDIS_URL=redis://:argus-redis-password@redis:6379

# PostgreSQL (if running full stack)
DATABASE_URL=postgresql://argus:argus-password@postgres:5432/argus

# Server settings
PORT=3000
NODE_ENV=production
```

### Start Services

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps
```

## Docker Compose File

### MCP Server Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  argus-mcp:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - PORT=3000
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - MINIO_ENDPOINT=http://minio:9000
      - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
      - MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
      - MINIO_BUCKET=argus-artifacts
    depends_on:
      - redis
      - minio
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=${MINIO_ACCESS_KEY}
      - MINIO_ROOT_PASSWORD=${MINIO_SECRET_KEY}
    volumes:
      - minio-data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

volumes:
  redis-data:
  minio-data:
```

### Full Stack (Brain + MCP)

```yaml
# docker-compose-full.yml
version: '3.8'

services:
  argus-brain:
    image: ghcr.io/raphaenterprises-ai/argus-brain:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://argus:${POSTGRES_PASSWORD}@postgres:5432/argus
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - MINIO_ENDPOINT=http://minio:9000
      - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
      - MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - LLM_PROVIDER=anthropic
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  argus-mcp:
    image: ghcr.io/raphaenterprises-ai/argus-mcp:latest
    ports:
      - "3000:3000"
    environment:
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - MINIO_ENDPOINT=http://minio:9000
      - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
      - MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
    depends_on:
      - redis
      - minio

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_USER=argus
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=argus
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U argus"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=${MINIO_ACCESS_KEY}
      - MINIO_ROOT_PASSWORD=${MINIO_SECRET_KEY}
    volumes:
      - minio-data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  selenium-hub:
    image: selenium/hub:4.15.0
    ports:
      - "4444:4444"
    environment:
      - GRID_MAX_SESSION=10

  selenium-chrome:
    image: selenium/node-chrome:4.15.0
    depends_on:
      - selenium-hub
    environment:
      - SE_EVENT_BUS_HOST=selenium-hub
      - SE_EVENT_BUS_PUBLISH_PORT=4442
      - SE_EVENT_BUS_SUBSCRIBE_PORT=4443
      - SE_NODE_MAX_SESSIONS=2
    shm_size: 2gb
    deploy:
      replicas: 3

volumes:
  postgres-data:
  redis-data:
  minio-data:
```

## Operations

### Start/Stop

```bash
# Start
docker compose up -d

# Stop (keeps data)
docker compose stop

# Stop and remove containers
docker compose down

# Stop and remove ALL data
docker compose down -v
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f argus-brain

# Last 100 lines
docker compose logs --tail 100 argus-brain
```

### Scale Services

```bash
# Scale Chrome nodes
docker compose up -d --scale selenium-chrome=5
```

### Update Images

```bash
# Pull latest images
docker compose pull

# Recreate containers
docker compose up -d --force-recreate
```

## Health Checks

```bash
# Check all services
docker compose ps

# Test Brain API
curl http://localhost:8000/health

# Test MCP Server
curl http://localhost:3000/health

# Test Selenium Grid
curl http://localhost:4444/status

# Test MinIO
curl http://localhost:9000/minio/health/live
```

## Resource Limits

Add resource constraints for production:

```yaml
services:
  argus-brain:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs argus-brain

# Check resource usage
docker stats

# Recreate container
docker compose up -d --force-recreate argus-brain
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker compose exec postgres pg_isready -U argus

# Connect to database
docker compose exec postgres psql -U argus -d argus

# Check migrations
docker compose exec argus-brain alembic current
```

### Storage Issues

```bash
# Check MinIO buckets
docker compose exec minio mc ls local/

# Create bucket manually
docker compose exec minio mc mb local/argus-artifacts
```

## Migration to Kubernetes

When ready to move to production Kubernetes:

1. Export your configuration:
   ```bash
   docker compose config > docker-compose-export.yaml
   ```

2. Convert to Kubernetes manifests (optional):
   ```bash
   kompose convert -f docker-compose.yml
   ```

3. Use Helm chart with equivalent values:
   ```yaml
   # values.yaml for Helm
   brain:
     env:
       LLM_PROVIDER: "anthropic"
     secrets:
       anthropicApiKey: "sk-ant-xxx"
   ```

See [Helm Installation](helm-installation.md) for full Kubernetes deployment.
