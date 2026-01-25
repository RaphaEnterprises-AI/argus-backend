# Argus Data Layer

Event-driven data infrastructure for the Argus E2E Testing Agent, built on enterprise-grade open-source components:

- **Redpanda** - Kafka-compatible streaming (BSL → Apache 2.0)
- **FalkorDB** - Graph database for knowledge graphs (SSPLv1 - internal use)
- **Valkey** - Redis-compatible cache (BSD-3-Clause)
- **Cognee** - AI Memory orchestration (Apache 2.0)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRODUCERS                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Backend   │  │  Dashboard  │  │   GitHub    │  │    Jira     │        │
│  │    API      │  │   (Next.js) │  │  Webhooks   │  │  Webhooks   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
└─────────┼────────────────┼────────────────┼────────────────┼────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REDPANDA (Kafka-compatible)                          │
│  Topics: argus.codebase.*, argus.test.*, argus.healing.*, argus.integration.*│
└─────────────────────────────────────────────────────────────────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONSUMERS                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Cognee    │  │   Healing   │  │ Notification│  │ Integration │        │
│  │   Worker    │  │   Worker    │  │   Worker    │  │   Worker    │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
└─────────┼────────────────┼────────────────┼────────────────┼────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA STORES                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │    FalkorDB      │  │     Valkey       │  │    Supabase      │          │
│  │  (Graph Store)   │  │     (Cache)      │  │   (PostgreSQL)   │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
data-layer/
├── kubernetes/           # K8s manifests for production
│   ├── namespace.yaml    # argus-data namespace with quotas
│   ├── secrets.yaml      # Secret templates (edit before applying)
│   ├── redpanda-values.yaml  # Helm values for Redpanda cluster
│   ├── falkordb.yaml     # FalkorDB StatefulSet
│   ├── valkey.yaml       # Valkey StatefulSet
│   ├── network-policies.yaml  # Network isolation
│   └── services.yaml     # ClusterIP services
├── docker/               # Local development
│   ├── docker-compose.data-layer.yml  # Full local stack
│   └── Dockerfile.cognee-worker       # Cognee worker image
├── scripts/              # Utility scripts
│   └── create-topics.sh  # Create Redpanda topics
└── README.md             # This file
```

## Quick Start (Local Development)

1. **Start the data layer:**
   ```bash
   cd data-layer/docker
   docker compose -f docker-compose.data-layer.yml up -d
   ```

2. **Create topics:**
   ```bash
   cd ../scripts
   ./create-topics.sh
   ```

3. **Access services:**
   - Redpanda Console: http://localhost:8080
   - FalkorDB: `redis-cli -p 6379 -a argus_dev_password`
   - Valkey: `redis-cli -p 6380 -a argus_cache_password`

## Production Deployment (Vultr K8s)

1. **Add Helm repo:**
   ```bash
   helm repo add redpanda https://charts.redpanda.com/
   helm repo update
   ```

2. **Configure secrets:**
   ```bash
   # Generate passwords
   export FALKORDB_PASSWORD=$(openssl rand -base64 24)
   export REDPANDA_PASSWORD=$(openssl rand -base64 24)

   # Edit secrets.yaml with actual values
   vim kubernetes/secrets.yaml
   ```

3. **Deploy:**
   ```bash
   kubectl apply -f kubernetes/namespace.yaml
   kubectl apply -f kubernetes/secrets.yaml
   kubectl apply -f kubernetes/network-policies.yaml
   helm install redpanda redpanda/redpanda -n argus-data -f kubernetes/redpanda-values.yaml --wait
   kubectl apply -f kubernetes/falkordb.yaml
   kubectl apply -f kubernetes/valkey.yaml
   kubectl apply -f kubernetes/services.yaml
   ```

## Event Topics

| Topic | Partitions | Description |
|-------|------------|-------------|
| `argus.codebase.ingested` | 6 | New codebase ready for analysis |
| `argus.codebase.analyzed` | 6 | Analysis complete with test surfaces |
| `argus.test.created` | 12 | New test generated |
| `argus.test.executed` | 12 | Test execution complete |
| `argus.test.failed` | 6 | Test failure detected |
| `argus.healing.requested` | 6 | Healing requested |
| `argus.healing.completed` | 6 | Healing complete |
| `argus.integration.github` | 6 | GitHub webhook events |
| `argus.integration.jira` | 3 | Jira integration events |
| `argus.integration.slack` | 3 | Slack notifications |
| `argus.notification.send` | 6 | Generic notifications |
| `argus.dlq` | 3 | Dead letter queue |

## Connection Strings

### From within K8s cluster (argus-data namespace):

| Component | Connection String |
|-----------|------------------|
| FalkorDB | `redis://:${FALKORDB_PASSWORD}@falkordb.argus-data.svc.cluster.local:6379` |
| Valkey | `redis://valkey.argus-data.svc.cluster.local:6379` |
| Redpanda Kafka | `redpanda-kafka.argus-data.svc.cluster.local:9092` |
| Schema Registry | `http://redpanda-schema-registry.argus-data.svc.cluster.local:8081` |

### From local development:

| Component | Connection String |
|-----------|------------------|
| FalkorDB | `redis://:argus_dev_password@localhost:6379` |
| Valkey | `redis://:argus_cache_password@localhost:6380` |
| Redpanda Kafka | `localhost:19092` |
| Schema Registry | `http://localhost:18081` |

## License Compliance

| Component | License | Usage |
|-----------|---------|-------|
| Redpanda | BSL 1.1 → Apache 2.0 | ✅ Production ready |
| FalkorDB | SSPLv1 | ✅ Internal use only |
| Valkey | BSD-3-Clause | ✅ Fully open |
| Cognee | Apache 2.0 | ✅ Fully open |

## Cost Estimates

### Development (~$150/mo):
- Single-node Redpanda: $40/mo
- FalkorDB (2GB): $30/mo
- Valkey (256MB): $15/mo
- Supabase (Free tier): $0

### Production (~$629/mo):
- 3-node Redpanda cluster: $300/mo
- FalkorDB HA (2 replicas): $150/mo
- Valkey HA: $80/mo
- Supabase Pro: $25/mo
- Network/Storage: $74/mo

### Enterprise HA (~$2,655/mo):
- 5-node Redpanda: $700/mo
- FalkorDB cluster (3 replicas): $400/mo
- Valkey cluster (6 nodes): $300/mo
- Supabase Team: $599/mo
- Dedicated Cognee workers: $400/mo
- Monitoring/Logging: $256/mo
