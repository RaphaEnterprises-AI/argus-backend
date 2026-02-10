# Skopaq Enterprise Self-Hosted Deployment

## Overview

Skopaq Enterprise enables organizations to deploy the complete Skopaq E2E Testing platform within their own infrastructure, ensuring:

- **Data Sovereignty**: All test data, screenshots, and AI interactions stay within your network
- **Air-Gap Support**: Optional fully offline operation with local LLM inference
- **Compliance**: Meet SOC2, HIPAA, GDPR, and industry-specific requirements
- **Data Residency**: Flexible deployment options for regional data requirements ([details](data-residency.md))
- **Integration**: Connect to your existing CI/CD, observability, and security tools

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Customer Kubernetes Cluster                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                           Skopaq Namespace                                ││
│  │                                                                          ││
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────────┐││
│  │  │  Skopaq Brain  │    │   MCP Server  │    │      Selenium Grid        │││
│  │  │   (FastAPI)   │◄──►│  (Node.js)    │    │  Hub + Chrome Nodes       │││
│  │  │   Replicas: 2 │    │   Replicas: 1 │    │  Replicas: configurable   │││
│  │  └───────┬───────┘    └───────────────┘    └───────────────────────────┘││
│  │          │                                                               ││
│  │  ┌───────┴───────────────────────────────────────────────────────────┐  ││
│  │  │                         Data Layer                                 │  ││
│  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────────┐ │  ││
│  │  │  │PostgreSQL │  │   Redis   │  │   MinIO   │  │ Ollama (opt)    │ │  ││
│  │  │  │ + pgvector│  │ (Valkey)  │  │S3 Storage │  │ Local LLM       │ │  ││
│  │  │  └───────────┘  └───────────┘  └───────────┘  └─────────────────┘ │  ││
│  │  └───────────────────────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Ingress Controller                               ││
│  │                    (nginx, traefik, or cloud LB)                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### Skopaq Brain (FastAPI Backend)

The main orchestrator handling:
- LangGraph-based agent coordination (25+ specialized agents)
- Test planning and execution
- Self-healing with pattern learning
- API endpoints for dashboard and integrations

### MCP Server (Model Context Protocol)

Enables AI coding assistants to interact with Skopaq:
- SSE-based communication
- 73+ tools for test management
- Compatible with Claude Code, Cursor, Windsurf, Continue

### Selenium Grid

Browser automation infrastructure:
- Hub for test distribution
- Chrome nodes for parallel execution
- Configurable replica count for throughput

### Data Layer

- **PostgreSQL + pgvector**: Relational data with vector embeddings
- **Redis/Valkey**: Session management and caching
- **MinIO**: S3-compatible artifact storage (screenshots, reports)
- **Ollama (optional)**: Local LLM inference for air-gap deployments

## Deployment Options

| Option | Use Case | Complexity |
|--------|----------|------------|
| [Quick Start](quick-start.md) | Evaluation, development | Low |
| [Docker Compose](docker-compose.md) | Single-node deployment | Low |
| [Helm Chart](helm-installation.md) | Production Kubernetes | Medium |
| [Air-Gap](llm-configuration.md#air-gap) | No internet connectivity | High |

## Prerequisites

- Kubernetes 1.25+ (for Helm deployment)
- Docker 24+ (for Docker Compose)
- Helm 3.10+
- 8+ CPU cores, 32GB RAM minimum
- 200GB storage (500GB+ recommended)
- (Optional) NVIDIA GPU for local LLM

## Quick Links

- [Requirements](requirements.md) - Hardware and software requirements
- [Quick Start](quick-start.md) - Get running in 15 minutes
- [Configuration](configuration.md) - All configuration options
- [Networking](networking.md) - Network architecture and policies
- [Data Residency](data-residency.md) - Data storage locations and compliance
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## Compliance & Data Residency

For enterprises with strict compliance requirements, see:

- [Data Residency Guide](data-residency.md) - Where your data is stored
- [Data Processing Agreement](/docs/compliance/DATA_PROCESSING_AGREEMENT.md) - DPA template
- [SOC2 Evidence](/docs/SOC2_EVIDENCE.md) - Security controls documentation
- [Air-Gap Configuration](llm-configuration.md#air-gap) - Zero external connectivity

## Support

- Documentation: https://docs.skopaq.ai/self-hosted
- Issues: https://github.com/raphaenterprises-ai/skopaq-e2e-testing-agent/issues
- Email: support@skopaq.ai
- Enterprise Support: enterprise@skopaq.ai
