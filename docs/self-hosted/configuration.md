# Configuration Reference

Complete reference for all Skopaq Enterprise configuration options.

## Environment Variables

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, `openrouter`, `ollama`, `azure`, `bedrock` |
| `DEFAULT_MODEL` | `claude-sonnet-4-5` | Default model for AI operations |
| `MAX_ITERATIONS` | `50` | Maximum iterations per test run |
| `COST_LIMIT_PER_RUN` | `10.00` | USD cost limit per test execution |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string |
| `DB_POOL_SIZE` | `10` | Connection pool size |
| `DB_MAX_OVERFLOW` | `20` | Max overflow connections |

### Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | - | Redis connection string |
| `CACHE_TTL_SECONDS` | `3600` | Default cache TTL |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_PROVIDER` | `minio` | Storage backend: `minio`, `cloudflare`, `s3` |
| `MINIO_ENDPOINT` | - | MinIO server endpoint |
| `MINIO_ACCESS_KEY` | - | MinIO access key |
| `MINIO_SECRET_KEY` | - | MinIO secret key |
| `MINIO_BUCKET` | `argus-artifacts` | Bucket for artifacts |
| `MINIO_SECURE` | `true` | Use HTTPS |

### LLM Providers

#### Anthropic

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Model to use |

#### OpenRouter

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | - | OpenRouter API key |
| `OPENROUTER_MODEL` | `anthropic/claude-3.5-sonnet` | Model identifier |

#### Ollama (Local)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:70b` | Model to use |

#### Azure OpenAI

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | - | Azure endpoint URL |
| `AZURE_OPENAI_API_KEY` | - | Azure API key |
| `AZURE_OPENAI_DEPLOYMENT` | - | Deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-02-01` | API version |

#### AWS Bedrock

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_ACCESS_KEY_ID` | - | AWS access key (or use IAM role) |
| `AWS_SECRET_ACCESS_KEY` | - | AWS secret key |
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-sonnet` | Bedrock model ID |

### Selenium Grid

| Variable | Default | Description |
|----------|---------|-------------|
| `SELENIUM_HUB_URL` | `http://selenium-hub:4444` | Selenium Hub URL |
| `BROWSER_TIMEOUT` | `30` | Browser timeout in seconds |
| `SCREENSHOT_RESOLUTION` | `1920x1080` | Screenshot resolution |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET` | - | JWT signing secret |
| `API_KEY_SALT` | - | Salt for API key hashing |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `RATE_LIMIT_PER_MINUTE` | `60` | API rate limit |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | Log format: `json`, `text` |
| `METRICS_ENABLED` | `true` | Enable Prometheus metrics |
| `TRACING_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `OTEL_EXPORTER_ENDPOINT` | - | OpenTelemetry collector URL |

## Helm Values Reference

### Global Settings

```yaml
global:
  # Domain for ingress hosts
  domain: "argus.company.com"

  # Kubernetes storage class
  storageClass: "standard"

  # Image pull policy
  imagePullPolicy: IfNotPresent

  # Image pull secrets for private registries
  imagePullSecrets: []
    # - name: registry-secret
```

### Brain Configuration

```yaml
brain:
  # Enable Brain deployment
  enabled: true

  # Number of replicas
  replicas: 2

  # Container image
  image:
    repository: ghcr.io/raphaenterprises-ai/argus-brain
    tag: latest
    pullPolicy: IfNotPresent

  # Environment variables
  env:
    LLM_PROVIDER: "anthropic"
    DEFAULT_MODEL: "claude-sonnet-4-5"
    LOG_LEVEL: "INFO"

  # Secrets (create Kubernetes Secret)
  secrets:
    anthropicApiKey: ""
    openrouterApiKey: ""
    jwtSecret: ""

  # Resource limits
  resources:
    requests:
      cpu: "500m"
      memory: "1Gi"
    limits:
      cpu: "2000m"
      memory: "4Gi"

  # Service configuration
  service:
    type: ClusterIP
    port: 8000

  # Autoscaling
  autoscaling:
    enabled: false
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80

  # Pod disruption budget
  podDisruptionBudget:
    enabled: false
    minAvailable: 1

  # Node selector
  nodeSelector: {}

  # Tolerations
  tolerations: []

  # Affinity rules
  affinity: {}
```

### MCP Server Configuration

```yaml
mcp:
  # Enable MCP Server deployment
  enabled: true

  # Number of replicas
  replicas: 1

  # Container image
  image:
    repository: ghcr.io/raphaenterprises-ai/argus-mcp
    tag: latest

  # Resource limits
  resources:
    requests:
      cpu: "200m"
      memory: "256Mi"
    limits:
      cpu: "500m"
      memory: "512Mi"

  # Service configuration
  service:
    type: ClusterIP
    port: 3000
```

### PostgreSQL Configuration

```yaml
postgresql:
  # Enable embedded PostgreSQL
  enabled: true

  # External PostgreSQL (when enabled: false)
  external:
    host: ""
    port: 5432
    database: "argus"
    username: "argus"
    existingSecret: ""  # Secret containing postgresql-password

  # Authentication
  auth:
    username: argus
    password: ""  # Generated if empty
    database: argus

  # Primary instance
  primary:
    persistence:
      enabled: true
      size: 50Gi
    resources:
      requests:
        cpu: "250m"
        memory: "256Mi"

  # Enable pgvector extension
  image:
    repository: pgvector/pgvector
    tag: pg16
```

### Redis Configuration

```yaml
redis:
  # Enable embedded Redis
  enabled: true

  # External Redis
  external:
    host: ""
    port: 6379
    existingSecret: ""  # Secret containing redis-password

  # Authentication
  auth:
    enabled: true
    password: ""  # Generated if empty

  # Architecture
  architecture: standalone  # or "replication"

  # Master resources
  master:
    persistence:
      enabled: true
      size: 8Gi
```

### MinIO Configuration

```yaml
minio:
  # Enable embedded MinIO
  enabled: true

  # External S3/MinIO
  external:
    endpoint: ""
    accessKey: ""
    secretKey: ""
    bucket: "argus-artifacts"
    region: "us-east-1"
    useSSL: true

  # Authentication
  auth:
    rootUser: argus
    rootPassword: ""  # Generated if empty

  # Persistence
  persistence:
    enabled: true
    size: 50Gi

  # Default buckets
  defaultBuckets: "argus-artifacts,argus-screenshots"
```

### Selenium Grid Configuration

```yaml
seleniumGrid:
  # Enable Selenium Grid
  enabled: true

  # Hub configuration
  hub:
    replicas: 1
    resources:
      requests:
        cpu: "500m"
        memory: "512Mi"

  # Chrome node configuration
  chrome:
    enabled: true
    replicas: 3
    maxSessions: 2
    resources:
      requests:
        cpu: "500m"
        memory: "1Gi"
      limits:
        cpu: "1"
        memory: "2Gi"

  # Firefox node (optional)
  firefox:
    enabled: false
    replicas: 0
```

### Ollama Configuration (Air-Gap)

```yaml
ollama:
  # Enable Ollama for local LLM
  enabled: false

  # Container image
  image:
    repository: ollama/ollama
    tag: latest

  # Models to pull on startup
  models:
    - "llama3.1:70b"
    - "codellama:34b"

  # GPU resources
  resources:
    limits:
      nvidia.com/gpu: 1
    requests:
      cpu: "4"
      memory: "16Gi"

  # Model storage
  persistence:
    enabled: true
    size: 200Gi
```

### Ingress Configuration

```yaml
ingress:
  # Enable ingress
  enabled: false

  # Ingress class
  className: "nginx"

  # Annotations
  annotations: {}
    # kubernetes.io/tls-acme: "true"
    # nginx.ingress.kubernetes.io/proxy-body-size: "100m"

  # Hosts
  hosts:
    brain: "argus-api.company.com"
    mcp: "argus-mcp.company.com"

  # TLS
  tls:
    enabled: false
    secretName: "argus-tls"
```

### Network Policies

```yaml
networkPolicies:
  # Enable network policies
  enabled: true

  # Allow traffic from these namespaces
  allowedNamespaces:
    - ingress-nginx
    - monitoring
```

### Metrics

```yaml
metrics:
  # Enable Prometheus metrics
  enabled: true

  # ServiceMonitor for Prometheus Operator
  serviceMonitor:
    enabled: false
    interval: 30s
    scrapeTimeout: 10s
```

## Configuration Examples

### Development Environment

```yaml
# values-dev.yaml
brain:
  replicas: 1
  resources:
    requests:
      cpu: "250m"
      memory: "512Mi"

postgresql:
  primary:
    persistence:
      size: 10Gi

seleniumGrid:
  chrome:
    replicas: 1

ingress:
  enabled: false
```

### Production Environment

```yaml
# values-prod.yaml
brain:
  replicas: 3
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
  resources:
    requests:
      cpu: "1000m"
      memory: "2Gi"
    limits:
      cpu: "2000m"
      memory: "4Gi"

postgresql:
  architecture: replication
  primary:
    persistence:
      size: 100Gi

redis:
  architecture: replication

seleniumGrid:
  chrome:
    replicas: 5

ingress:
  enabled: true
  tls:
    enabled: true

networkPolicies:
  enabled: true

metrics:
  enabled: true
  serviceMonitor:
    enabled: true
```
