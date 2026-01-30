# LLM Configuration Guide

Configure Large Language Model providers for Argus Enterprise.

## Provider Options

| Provider | Internet Required | Use Case |
|----------|-------------------|----------|
| Anthropic | Yes | Production (recommended) |
| OpenRouter | Yes | Multi-model access |
| Azure OpenAI | Yes* | Azure-first enterprises |
| AWS Bedrock | Yes* | AWS-first enterprises |
| Ollama | No | Air-gap deployments |

*Traffic stays within cloud provider network

## Quick Configuration

### Anthropic (Default)

```yaml
# values.yaml
brain:
  env:
    LLM_PROVIDER: "anthropic"
    DEFAULT_MODEL: "claude-sonnet-4-5"
  secrets:
    anthropicApiKey: "sk-ant-xxx"
```

### OpenRouter (300+ Models)

```yaml
brain:
  env:
    LLM_PROVIDER: "openrouter"
    OPENROUTER_MODEL: "anthropic/claude-3.5-sonnet"
  secrets:
    openrouterApiKey: "sk-or-xxx"
```

### Ollama (Air-Gap)

```yaml
brain:
  env:
    LLM_PROVIDER: "ollama"
    OLLAMA_HOST: "http://argus-ollama:11434"
    OLLAMA_MODEL: "llama3.1:70b"

ollama:
  enabled: true
  models:
    - "llama3.1:70b"
    - "codellama:34b"
```

## Anthropic Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | Set to `anthropic` |
| `ANTHROPIC_API_KEY` | - | Your Anthropic API key |
| `DEFAULT_MODEL` | `claude-sonnet-4-5` | Default model |

### Available Models

| Model | Use Case | Cost |
|-------|----------|------|
| `claude-opus-4-5` | Complex reasoning | $$$$ |
| `claude-sonnet-4-5` | Balanced (recommended) | $$ |
| `claude-haiku-4-5` | Fast, simple tasks | $ |

### Helm Configuration

```yaml
brain:
  env:
    LLM_PROVIDER: "anthropic"
    DEFAULT_MODEL: "claude-sonnet-4-5"
  secrets:
    anthropicApiKey: "sk-ant-xxx"
```

## OpenRouter Configuration

OpenRouter provides unified access to 300+ models from multiple providers.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | - | Set to `openrouter` |
| `OPENROUTER_API_KEY` | - | Your OpenRouter API key |
| `OPENROUTER_MODEL` | `anthropic/claude-3.5-sonnet` | Model identifier |

### Popular Models

| Model ID | Provider | Use Case |
|----------|----------|----------|
| `anthropic/claude-3.5-sonnet` | Anthropic | General purpose |
| `openai/gpt-4o` | OpenAI | Alternative to Claude |
| `google/gemini-pro-1.5` | Google | Large context |
| `meta-llama/llama-3.1-70b-instruct` | Meta | Open source |

### Helm Configuration

```yaml
brain:
  env:
    LLM_PROVIDER: "openrouter"
    OPENROUTER_MODEL: "anthropic/claude-3.5-sonnet"
  secrets:
    openrouterApiKey: "sk-or-xxx"
```

## Azure OpenAI Configuration

Keep LLM traffic within your Azure network.

### Prerequisites

1. Azure OpenAI resource created
2. Model deployed (e.g., GPT-4o)
3. API key or Azure AD authentication

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | - | Set to `azure` |
| `AZURE_OPENAI_ENDPOINT` | - | Your Azure endpoint |
| `AZURE_OPENAI_API_KEY` | - | API key |
| `AZURE_OPENAI_DEPLOYMENT` | - | Deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-02-01` | API version |

### Helm Configuration

```yaml
brain:
  env:
    LLM_PROVIDER: "azure"
    AZURE_OPENAI_ENDPOINT: "https://your-resource.openai.azure.com"
    AZURE_OPENAI_DEPLOYMENT: "gpt-4o"
    AZURE_OPENAI_API_VERSION: "2024-02-01"
  secrets:
    azureOpenaiApiKey: "xxx"
```

### Azure AD Authentication

For managed identity authentication:

```yaml
brain:
  env:
    LLM_PROVIDER: "azure"
    AZURE_OPENAI_ENDPOINT: "https://your-resource.openai.azure.com"
    AZURE_OPENAI_DEPLOYMENT: "gpt-4o"
    AZURE_USE_MANAGED_IDENTITY: "true"
  serviceAccount:
    annotations:
      azure.workload.identity/client-id: "xxx"
```

## AWS Bedrock Configuration

Keep LLM traffic within your AWS network.

### Prerequisites

1. Bedrock model access enabled
2. IAM role with bedrock:InvokeModel permission

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | - | Set to `bedrock` |
| `AWS_REGION` | `us-east-1` | AWS region |
| `BEDROCK_MODEL_ID` | - | Model ID |
| `AWS_ACCESS_KEY_ID` | - | (Optional) AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | - | (Optional) AWS credentials |

### Available Models

| Model ID | Provider |
|----------|----------|
| `anthropic.claude-3-5-sonnet-20241022-v2:0` | Anthropic |
| `anthropic.claude-3-5-haiku-20241022-v1:0` | Anthropic |
| `meta.llama3-1-70b-instruct-v1:0` | Meta |

### Helm Configuration

```yaml
brain:
  env:
    LLM_PROVIDER: "bedrock"
    AWS_REGION: "us-east-1"
    BEDROCK_MODEL_ID: "anthropic.claude-3-5-sonnet-20241022-v2:0"
  serviceAccount:
    annotations:
      eks.amazonaws.com/role-arn: "arn:aws:iam::123456789:role/argus-bedrock-access"
```

### IAM Policy

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-*"
            ]
        }
    ]
}
```

## Ollama (Air-Gap) {#air-gap}

Deploy Argus with zero external connectivity using Ollama for local LLM inference.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Air-Gapped Environment                               │
│                                                                              │
│  ┌───────────────┐         ┌───────────────┐         ┌───────────────┐     │
│  │  Argus Brain  │────────►│    Ollama     │◄───────►│  GPU Server   │     │
│  │               │   HTTP  │  :11434       │         │  NVIDIA A100  │     │
│  └───────────────┘         └───────────────┘         └───────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│                            ┌───────────────┐                                │
│                            │  Model Store  │                                │
│                            │  (200GB PVC)  │                                │
│                            └───────────────┘                                │
│                                                                              │
│  No outbound internet connectivity required                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Helm Configuration

```yaml
# values-airgap.yaml
brain:
  env:
    LLM_PROVIDER: "ollama"
    OLLAMA_HOST: "http://argus-ollama:11434"
    OLLAMA_MODEL: "llama3.1:70b"

ollama:
  enabled: true
  image:
    repository: ollama/ollama
    tag: latest
  models:
    - "llama3.1:70b"     # General purpose
    - "codellama:34b"    # Code analysis
  persistence:
    enabled: true
    size: 200Gi
  resources:
    limits:
      nvidia.com/gpu: 1
    requests:
      cpu: "4"
      memory: "32Gi"
```

### GPU Requirements

| Model | VRAM | Recommended GPU |
|-------|------|-----------------|
| llama3.1:8b | 8GB | RTX 4070 / A4000 |
| llama3.1:70b | 48GB | A100 80GB / 2x A10 |
| codellama:34b | 24GB | A10 / A6000 |
| mixtral:8x7b | 48GB | A100 80GB |

### Pre-loading Models (Air-Gap)

For fully air-gapped deployments, pre-load models:

```bash
# On internet-connected machine
ollama pull llama3.1:70b
ollama pull codellama:34b

# Export models
ollama export llama3.1:70b > llama3.1-70b.tar
ollama export codellama:34b > codellama-34b.tar

# Transfer to air-gap environment
# ...

# Import on air-gap Ollama
ollama import llama3.1:70b < llama3.1-70b.tar
ollama import codellama:34b < codellama-34b.tar
```

### Multi-GPU Configuration

For larger models or higher throughput:

```yaml
ollama:
  resources:
    limits:
      nvidia.com/gpu: 2  # Use 2 GPUs
  env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: "0,1"
```

## Model Selection by Task

Argus uses different models based on task complexity:

| Task Type | Recommended Model | Fallback |
|-----------|-------------------|----------|
| Code Analysis | claude-sonnet-4-5 | llama3.1:70b |
| Test Generation | claude-sonnet-4-5 | codellama:34b |
| Self-Healing | claude-sonnet-4-5 | llama3.1:70b |
| Simple Classification | claude-haiku-4-5 | llama3.1:8b |
| Complex Reasoning | claude-opus-4-5 | llama3.1:70b |

### Configure Task-Specific Models

```yaml
brain:
  env:
    # Default model
    DEFAULT_MODEL: "claude-sonnet-4-5"

    # Task-specific overrides
    MODEL_CODE_ANALYSIS: "claude-sonnet-4-5"
    MODEL_TEST_GENERATION: "claude-sonnet-4-5"
    MODEL_CLASSIFICATION: "claude-haiku-4-5"
    MODEL_COMPLEX: "claude-opus-4-5"
```

## Cost Management

### Set Budget Limits

```yaml
brain:
  env:
    COST_LIMIT_PER_RUN: "10.00"      # USD per test run
    COST_LIMIT_MONTHLY: "1000.00"    # USD monthly
```

### Model Cost Comparison

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| claude-opus-4-5 | $15.00 | $75.00 |
| claude-sonnet-4-5 | $3.00 | $15.00 |
| claude-haiku-4-5 | $0.25 | $1.25 |
| gpt-4o | $2.50 | $10.00 |
| llama3.1:70b (local) | $0 | $0 |

## Troubleshooting

### Test LLM Connection

```bash
# Test Anthropic
curl -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"Hi"}]}'

# Test Ollama
kubectl exec deploy/argus-ollama -n argus -- \
  curl -s http://localhost:11434/api/tags

# Test from Brain pod
kubectl exec deploy/argus-brain -n argus -- \
  curl -s http://argus-ollama:11434/api/tags
```

### Common Issues

**Rate Limiting**
```yaml
brain:
  env:
    LLM_RETRY_MAX_ATTEMPTS: "5"
    LLM_RETRY_DELAY_MS: "1000"
```

**Timeout Issues**
```yaml
brain:
  env:
    LLM_TIMEOUT_SECONDS: "120"
```

**GPU Not Detected**
```bash
# Verify NVIDIA runtime
kubectl exec deploy/argus-ollama -n argus -- nvidia-smi
```

## External Dependencies Reference {#external-dependencies}

This section documents all external network dependencies for each deployment mode.
Use this to plan firewall rules and verify air-gap compliance.

### Cloud Deployment Dependencies

When using cloud LLM providers, the following external endpoints are required:

| Component | Endpoint | Port | Purpose |
|-----------|----------|------|---------|
| **Anthropic** | api.anthropic.com | 443 | Claude API |
| **OpenRouter** | api.openrouter.ai | 443 | Multi-model API |
| **Azure OpenAI** | *.openai.azure.com | 443 | Azure OpenAI API |
| **AWS Bedrock** | bedrock.*.amazonaws.com | 443 | Bedrock API |
| **OpenAI** | api.openai.com | 443 | OpenAI API (via OpenRouter) |
| **Google** | generativelanguage.googleapis.com | 443 | Gemini API (via OpenRouter) |
| **Cohere** | api.cohere.ai | 443 | Embeddings (optional) |

**Storage (if using cloud storage):**

| Component | Endpoint | Port | Purpose |
|-----------|----------|------|---------|
| **Cloudflare R2** | *.r2.cloudflarestorage.com | 443 | Artifact storage |
| **AWS S3** | s3.*.amazonaws.com | 443 | S3 storage |
| **Supabase Storage** | *.supabase.co | 443 | Supabase storage |

### Air-Gap Mode Dependencies

When configured for air-gap operation with Ollama and MinIO, **no external network access is required**.

| Component | Endpoint | Port | Notes |
|-----------|----------|------|-------|
| **Ollama** | localhost:11434 | 11434 | Local only |
| **MinIO** | localhost:9000 | 9000 | Local only |
| **PostgreSQL** | localhost:5432 | 5432 | Local only |
| **Valkey/Redis** | localhost:6379 | 6379 | Local only |

### Components That Require Pre-Installation for Air-Gap

The following must be pre-installed/downloaded before disconnecting from the internet:

1. **Ollama Models** (40-100GB depending on models)
   ```bash
   # Pre-download required models
   ollama pull llama3.1:70b        # 40GB
   ollama pull codellama:34b       # 19GB
   ollama pull nomic-embed-text    # 300MB (for embeddings)
   ```

2. **Container Images** (~5GB total)
   ```bash
   # Pull and save all required images
   docker pull ollama/ollama:latest
   docker pull minio/minio:latest
   docker pull postgres:15
   docker pull redis:7-alpine
   docker pull ghcr.io/argus/argus-brain:latest

   # Save to archive for air-gap transfer
   docker save ollama/ollama minio/minio postgres:15 redis:7-alpine \
     ghcr.io/argus/argus-brain:latest | gzip > argus-images.tar.gz
   ```

3. **Helm Charts**
   ```bash
   # Download charts for offline installation
   helm pull argus/argus-enterprise --version 1.0.0
   ```

4. **Python Dependencies** (for development only)
   ```bash
   pip download -r requirements.txt -d ./offline-packages/
   ```

### Verifying Air-Gap Compliance

Run the air-gap integration tests to verify no external calls are made:

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run air-gap validation tests
pytest tests/integration/test_airgap.py -v -m airgap

# Generate validation report
pytest tests/integration/test_airgap.py::test_airgap_validation_report -v
```

The test suite verifies:
- Ollama responds without external network calls
- MinIO handles artifact storage locally
- Full test generation workflow completes offline
- No DNS queries or TCP connections to external hosts

### Network Isolation Testing

For strict air-gap validation, run tests with network isolation:

```bash
# Linux: Use network namespace
sudo unshare --net pytest tests/integration/test_airgap.py -v

# Docker: Use --network=none
docker run --network=none -v $(pwd):/app argus-test \
  pytest /app/tests/integration/test_airgap.py -v

# Kubernetes: Use NetworkPolicy to block egress
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-external-egress
spec:
  podSelector:
    matchLabels:
      app: argus
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector: {}  # Allow within cluster only
EOF
```

### Remaining Dependencies to Address

The following features have optional external dependencies that can be disabled:

| Feature | External Dependency | Air-Gap Alternative | Configuration |
|---------|---------------------|---------------------|---------------|
| Telemetry | Sentry | Disable | `SENTRY_ENABLED=false` |
| Webhooks | External URLs | Internal only | Configure internal endpoints |
| GitHub/GitLab Integration | API | Disable or use self-hosted | Configure self-hosted URLs |
| Documentation links | docs.heyargus.ai | Bundle locally | Use offline docs |

To fully disable all external features:

```yaml
# values-airgap.yaml
brain:
  env:
    # Disable telemetry
    SENTRY_ENABLED: "false"
    TELEMETRY_ENABLED: "false"

    # Use local providers only
    LLM_PROVIDER: "ollama"
    STORAGE_PROVIDER: "minio"

    # Disable external integrations
    GITHUB_INTEGRATION_ENABLED: "false"
    GITLAB_INTEGRATION_ENABLED: "false"

    # Use local embedding model
    EMBEDDING_PROVIDER: "ollama"
    EMBEDDING_MODEL: "nomic-embed-text"
```
