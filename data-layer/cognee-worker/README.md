# Cognee Kafka Worker

Event-driven knowledge graph builder for Argus. Consumes events from Redpanda/Kafka and builds knowledge graphs using Cognee's ECL pipeline with FalkorDB storage.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Redpanda/Kafka                                │
│                                                                       │
│  ┌─────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │
│  │ codebase.ingested│  │ test.created   │  │ healing.requested      │ │
│  └────────┬────────┘  └───────┬────────┘  └───────────┬────────────┘ │
└───────────┼───────────────────┼───────────────────────┼──────────────┘
            │                   │                       │
            └───────────────────┼───────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Cognee Kafka Worker │
                    │                       │
                    │  ┌─────────────────┐  │
                    │  │ Event Router    │  │
                    │  └────────┬────────┘  │
                    │           │           │
                    │  ┌────────▼────────┐  │
                    │  │ Cognee ECL      │  │
                    │  │ Pipeline        │  │
                    │  │ • Extract       │  │
                    │  │ • Cognify       │  │
                    │  │ • Load          │  │
                    │  └────────┬────────┘  │
                    └───────────┼───────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │   FalkorDB    │   │    Valkey     │   │   Redpanda    │
    │ (Graph Store) │   │   (Cache)     │   │   (Output)    │
    └───────────────┘   └───────────────┘   └───────────────┘
```

## Features

- **Event-Driven**: Consumes from multiple Kafka topics
- **Knowledge Graphs**: Builds graphs using Cognee's ECL pipeline
- **FalkorDB Storage**: Persists knowledge graphs to FalkorDB
- **SASL Authentication**: Secure Kafka/Redpanda connections
- **Health Checks**: HTTP endpoints for K8s probes
- **DLQ Support**: Failed messages go to dead letter queue
- **Graceful Shutdown**: Handles SIGTERM/SIGINT signals

## Topics

### Input Topics
- `argus.codebase.ingested` - New codebase analysis requests
- `argus.test.created` - Test created events
- `argus.test.executed` - Test execution results
- `argus.test.failed` - Test failure events
- `argus.healing.requested` - Self-healing requests

### Output Topics
- `argus.codebase.analyzed` - Analysis completion events
- `argus.healing.completed` - Healing analysis results
- `argus.dlq` - Dead letter queue for failed messages

## Configuration

All configuration is via environment variables:

### Kafka/Redpanda
| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BOOTSTRAP_SERVERS` | `redpanda.argus-data.svc.cluster.local:9092` | Kafka broker addresses |
| `KAFKA_CONSUMER_GROUP` | `cognee-worker` | Consumer group ID |
| `KAFKA_SECURITY_PROTOCOL` | `SASL_PLAINTEXT` | Security protocol |
| `KAFKA_SASL_MECHANISM` | `SCRAM-SHA-512` | SASL mechanism |
| `KAFKA_SASL_USERNAME` | `argus-service` | SASL username |
| `KAFKA_SASL_PASSWORD` | - | SASL password (required) |

### FalkorDB
| Variable | Default | Description |
|----------|---------|-------------|
| `FALKORDB_HOST` | `falkordb-headless.argus-data.svc.cluster.local` | FalkorDB host |
| `FALKORDB_PORT` | `6379` | FalkorDB port |
| `FALKORDB_PASSWORD` | - | FalkorDB password (required) |
| `FALKORDB_GRAPH_NAME` | `argus_knowledge` | Graph name |

### Valkey
| Variable | Default | Description |
|----------|---------|-------------|
| `VALKEY_HOST` | `valkey-headless.argus-data.svc.cluster.local` | Valkey host |
| `VALKEY_PORT` | `6379` | Valkey port |
| `VALKEY_PASSWORD` | - | Valkey password (required) |

### Cognee/LLM
| Variable | Default | Description |
|----------|---------|-------------|
| `COGNEE_LLM_PROVIDER` | `anthropic` | LLM provider |
| `COGNEE_LLM_MODEL` | `claude-sonnet-4-5-20250514` | LLM model |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `COGNEE_EMBEDDING_PROVIDER` | `openai` | Embedding provider |
| `COGNEE_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_API_KEY` | - | OpenAI API key |

### Worker
| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_RETRIES` | `3` | Max connection retries |
| `HEALTH_CHECK_PORT` | `8080` | Health check HTTP port |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires Kafka/Redpanda running)
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export FALKORDB_HOST=localhost
export VALKEY_HOST=localhost
python src/worker.py
```

## Docker Build

```bash
# Build image
docker build -t ghcr.io/heyargus/cognee-worker:latest .

# Run container
docker run -e KAFKA_BOOTSTRAP_SERVERS=host.docker.internal:9092 \
           -e KAFKA_SASL_PASSWORD=secret \
           -e FALKORDB_PASSWORD=secret \
           -e VALKEY_PASSWORD=secret \
           ghcr.io/heyargus/cognee-worker:latest
```

## Kubernetes Deployment

The worker is deployed via `data-layer/kubernetes/cognee-worker.yaml`:

```bash
# Deploy to Kubernetes
kubectl apply -f ../kubernetes/cognee-worker.yaml

# Check status
kubectl get pods -n argus-data -l app.kubernetes.io/name=cognee-worker

# View logs
kubectl logs -n argus-data -l app.kubernetes.io/name=cognee-worker -f
```

## Health Endpoints

- `GET /health` - Liveness probe (always returns 200 if process is running)
- `GET /ready` - Readiness probe (returns 200 only when connected to Kafka)

## Message Format

### codebase.ingested
```json
{
  "project_id": "proj_123",
  "repo_url": "https://github.com/org/repo",
  "content": {
    "files": [
      {"path": "src/main.py", "content": "..."},
      {"path": "README.md", "content": "..."}
    ]
  }
}
```

### codebase.analyzed
```json
{
  "project_id": "proj_123",
  "repo_url": "https://github.com/org/repo",
  "status": "analyzed",
  "graph_name": "argus_knowledge",
  "file_count": 42,
  "timestamp": "2025-01-25T12:00:00Z"
}
```

### test.failed
```json
{
  "test_id": "test_456",
  "project_id": "proj_123",
  "test_name": "test_login_flow",
  "status": "failed",
  "error_message": "Element not found: #login-button",
  "stack_trace": "...",
  "duration_ms": 5000
}
```

### healing.requested
```json
{
  "test_id": "test_456",
  "project_id": "proj_123",
  "failure_reason": "Element not found: #login-button"
}
```

## License

Proprietary - Argus
