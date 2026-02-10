# System Requirements

## Hardware Requirements

### Minimum Configuration (Evaluation/Development)

| Component | Specification |
|-----------|--------------|
| CPU | 8 cores (x86_64) |
| RAM | 32 GB |
| Storage | 200 GB SSD |
| Network | 100 Mbps |
| GPU | None required |

### Recommended Configuration (Production)

| Component | Specification |
|-----------|--------------|
| CPU | 16+ cores (x86_64) |
| RAM | 64 GB |
| Storage | 500 GB NVMe SSD |
| Network | 1 Gbps |
| GPU | NVIDIA A10/A100 (for local LLM) |

### Storage Sizing Guide

| Data Type | Size per 1000 Tests | Retention |
|-----------|---------------------|-----------|
| Screenshots | 2 GB | Configurable |
| Test Reports | 100 MB | Configurable |
| Logs | 500 MB | 30 days default |
| Database | 1 GB | Permanent |
| LLM Models | 8-80 GB | Permanent |

## Software Requirements

### Kubernetes Deployment

| Software | Version | Notes |
|----------|---------|-------|
| Kubernetes | 1.25+ | EKS, GKE, AKS, OpenShift, RKE2 |
| Helm | 3.10+ | For chart installation |
| kubectl | 1.25+ | Cluster management |
| PV Provisioner | Any | For persistent storage |

### Docker Compose Deployment

| Software | Version | Notes |
|----------|---------|-------|
| Docker | 24+ | Docker Desktop or Docker CE |
| Docker Compose | 2.20+ | Usually bundled with Docker |

### Local LLM (Optional)

| Software | Version | Notes |
|----------|---------|-------|
| NVIDIA Driver | 535+ | For GPU acceleration |
| NVIDIA Container Toolkit | Latest | For GPU in containers |
| CUDA | 12.0+ | Included in Ollama images |

## Network Requirements

### Outbound Connectivity (Cloud LLM Mode)

| Destination | Port | Purpose |
|-------------|------|---------|
| api.anthropic.com | 443 | Claude API |
| api.openai.com | 443 | OpenAI API (if used) |
| *.openrouter.ai | 443 | OpenRouter (if used) |

### Air-Gap Mode (No Outbound Required)

In air-gap mode with local Ollama LLM, no outbound connectivity is required.

### Internal Ports

| Service | Port | Protocol |
|---------|------|----------|
| Skopaq Brain API | 8000 | HTTP |
| MCP Server | 3000 | HTTP/SSE |
| PostgreSQL | 5432 | TCP |
| Redis | 6379 | TCP |
| MinIO API | 9000 | HTTP |
| MinIO Console | 9001 | HTTP |
| Selenium Hub | 4444 | HTTP |
| Ollama | 11434 | HTTP |

## Browser Requirements

### Selenium Grid Nodes

| Browser | Version | Notes |
|---------|---------|-------|
| Chrome | Latest stable | Primary supported browser |
| Firefox | Latest stable | Secondary support |
| Edge | Latest stable | Windows environments |

### Recommended Chrome Node Configuration

```yaml
seleniumGrid:
  chrome:
    replicas: 3        # Parallel test capacity
    resources:
      limits:
        cpu: "1"
        memory: "2Gi"
      requests:
        cpu: "500m"
        memory: "1Gi"
```

## Supported Platforms

### Kubernetes Distributions

| Distribution | Version | Status |
|--------------|---------|--------|
| Amazon EKS | 1.25+ | Fully Supported |
| Google GKE | 1.25+ | Fully Supported |
| Azure AKS | 1.25+ | Fully Supported |
| Red Hat OpenShift | 4.12+ | Fully Supported |
| Rancher RKE2 | 1.25+ | Fully Supported |
| VMware Tanzu | 1.25+ | Tested |
| K3s | 1.25+ | Tested |
| Kind | 0.20+ | Development only |
| Minikube | 1.30+ | Development only |

### Operating Systems (Docker Compose)

| OS | Version | Status |
|----|---------|--------|
| Ubuntu | 22.04 LTS, 24.04 LTS | Fully Supported |
| RHEL | 8, 9 | Fully Supported |
| Debian | 11, 12 | Fully Supported |
| macOS | 13+ | Development only |
| Windows | Server 2022, 11 | Via Docker Desktop |

## GPU Requirements (Local LLM)

### NVIDIA GPU Sizing

| Model | VRAM Required | Recommended GPU |
|-------|---------------|-----------------|
| llama3.1:8b | 8 GB | RTX 4070, A4000 |
| llama3.1:70b | 48 GB | A100 80GB, 2x A10 |
| mistral:7b | 8 GB | RTX 4070, A4000 |
| codellama:34b | 24 GB | A10, A6000 |

### Multi-GPU Configuration

```yaml
ollama:
  resources:
    limits:
      nvidia.com/gpu: 2   # Use 2 GPUs
```

## Compliance Considerations

### Data Residency

All data remains within your infrastructure:
- Test data and results
- Screenshots and artifacts
- AI conversation logs
- Vector embeddings

### Encryption Requirements

| Data State | Encryption |
|------------|------------|
| At Rest | AES-256 (MinIO, PostgreSQL) |
| In Transit | TLS 1.3 |
| Secrets | Kubernetes Secrets or Vault |

### Audit Logging

Enable comprehensive audit logging:
```yaml
brain:
  env:
    AUDIT_LOG_ENABLED: "true"
    AUDIT_LOG_LEVEL: "INFO"
```

## Validation Checklist

Before installation, verify:

- [ ] Kubernetes cluster meets version requirements
- [ ] Sufficient CPU, memory, and storage available
- [ ] PersistentVolume provisioner configured
- [ ] Network policies allow required ports
- [ ] (If using GPU) NVIDIA drivers and toolkit installed
- [ ] (If using external LLM) Outbound connectivity available
- [ ] Container registry accessible (or images pre-loaded)
