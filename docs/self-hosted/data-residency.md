# Data Residency Guide

**Document Version:** 1.0
**Last Updated:** 2026-01-30
**Classification:** PUBLIC

This document provides comprehensive information about data storage locations, data flows, and residency guarantees for Skopaq Enterprise deployments to support compliance requirements (GDPR, HIPAA, SOC2, FedRAMP, etc.).

---

## Executive Summary

Skopaq Enterprise provides three deployment modes with different data residency characteristics:

| Deployment Mode | Data Location | External Dependencies | Compliance Level |
|-----------------|---------------|----------------------|------------------|
| **Self-Hosted** | Customer infrastructure | Cloud LLM (configurable) | High |
| **Self-Hosted (Air-Gap)** | Customer infrastructure | None | Maximum |
| **Cloud (SaaS)** | Skopaq-managed infrastructure | Yes | Standard |

---

## Data Storage Locations by Component

### Self-Hosted Deployment

| Data Type | Component | Storage Location | Encryption |
|-----------|-----------|------------------|------------|
| Test definitions | PostgreSQL | Customer Kubernetes cluster | AES-256 at rest |
| Test results | PostgreSQL | Customer Kubernetes cluster | AES-256 at rest |
| User accounts | PostgreSQL | Customer Kubernetes cluster | AES-256 at rest |
| Session data | Redis/Valkey | Customer Kubernetes cluster | In-memory (optional persistence) |
| Screenshots | MinIO/S3 | Customer object storage | AES-256 at rest |
| Test recordings | MinIO/S3 | Customer object storage | AES-256 at rest |
| Reports (PDF/HTML) | MinIO/S3 | Customer object storage | AES-256 at rest |
| Audit logs | PostgreSQL | Customer Kubernetes cluster | AES-256 at rest |
| LLM conversations | Local/External | See LLM Configuration below | Varies |
| Vector embeddings | PostgreSQL (pgvector) | Customer Kubernetes cluster | AES-256 at rest |
| Healing patterns | PostgreSQL | Customer Kubernetes cluster | AES-256 at rest |
| Integration credentials | Encrypted vault | Customer infrastructure | AES-256-GCM |

### Cloud (SaaS) Deployment

| Data Type | Component | Storage Location | Region |
|-----------|-----------|------------------|--------|
| All relational data | Supabase PostgreSQL | US (default) or EU | Configurable |
| Object storage | Cloudflare R2 | Global edge (US primary) | Configurable |
| LLM processing | Anthropic API | US | Fixed |
| Session cache | Cloudflare Workers KV | Global edge | Automatic |
| Audit logs | Supabase PostgreSQL | Same as primary | Automatic |

---

## Data Flow Diagrams

### Self-Hosted Data Flow (Standard Mode)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CUSTOMER INFRASTRUCTURE                                │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                              Kubernetes Cluster                            │  │
│  │                                                                            │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────┐ │  │
│  │  │   Browser   │───►│   Ingress   │───►│ Skopaq Brain │───►│ PostgreSQL │ │  │
│  │  │   (Users)   │    │ Controller  │    │  (FastAPI)  │    │ + pgvector │ │  │
│  │  └─────────────┘    └─────────────┘    └──────┬──────┘    └────────────┘ │  │
│  │                                               │                           │  │
│  │                          ┌───────────────────┼───────────────────┐       │  │
│  │                          │                   │                   │       │  │
│  │                          ▼                   ▼                   ▼       │  │
│  │                    ┌───────────┐       ┌───────────┐       ┌───────────┐ │  │
│  │                    │   MinIO   │       │   Redis   │       │ Selenium  │ │  │
│  │                    │ (Objects) │       │  (Cache)  │       │   Grid    │ │  │
│  │                    └───────────┘       └───────────┘       └───────────┘ │  │
│  │                                                                            │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                       │
│                                          │ LLM Requests (HTTPS)                 │
│                                          │ (configurable provider)              │
│                                          ▼                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
           ┌───────────────────────────────┼───────────────────────────────┐
           │                               │                               │
           ▼                               ▼                               ▼
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│   Anthropic API     │      │   Azure OpenAI      │      │   AWS Bedrock       │
│   (US region)       │      │ (Customer region)   │      │ (Customer region)   │
│                     │      │                     │      │                     │
│ Data retained: None │      │ Data retained: None │      │ Data retained: None │
│ (stateless API)     │      │ (customer managed)  │      │ (customer managed)  │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘

EXTERNAL DATA FLOW:
- LLM API calls only (no customer test data sent)
- Only prompts and model responses (transient)
- No persistent storage at LLM provider
```

### Air-Gap Data Flow (Zero External Connectivity)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       AIR-GAPPED CUSTOMER INFRASTRUCTURE                         │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                              Kubernetes Cluster                            │  │
│  │                                                                            │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────┐ │  │
│  │  │   Browser   │───►│   Ingress   │───►│ Skopaq Brain │───►│ PostgreSQL │ │  │
│  │  │   (Users)   │    │ Controller  │    │  (FastAPI)  │    │ + pgvector │ │  │
│  │  └─────────────┘    └─────────────┘    └──────┬──────┘    └────────────┘ │  │
│  │                                               │                           │  │
│  │                          ┌───────────────────┼───────────────────────────┐│  │
│  │                          │                   │                           ││  │
│  │                          ▼                   ▼                           ▼│  │
│  │                    ┌───────────┐       ┌───────────┐              ┌───────┴─┐│
│  │                    │   MinIO   │       │   Redis   │              │ Ollama  ││
│  │                    │ (Objects) │       │  (Cache)  │              │(Local   ││
│  │                    └───────────┘       └───────────┘              │ LLM)    ││
│  │                                                                   └─────────┘│
│  │                                                                            │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║  ZERO EXTERNAL NETWORK CALLS                                               ║  │
│  ║  - All LLM inference runs locally via Ollama                              ║  │
│  ║  - No telemetry or analytics transmitted                                  ║  │
│  ║  - No license validation calls required                                   ║  │
│  ║  - All data remains within customer boundary                              ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

NETWORK FIREWALL: Block all egress traffic (optional)
```

### Cloud (SaaS) Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CUSTOMER ENVIRONMENT                                │
│                                                                                  │
│  ┌─────────────────┐                                                            │
│  │  Browser/CLI    │                                                            │
│  │  (Dashboard)    │                                                            │
│  └────────┬────────┘                                                            │
│           │ HTTPS (TLS 1.3)                                                     │
└───────────┼─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ARGUS CLOUD INFRASTRUCTURE                          │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                         Cloudflare (Edge Layer)                           │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │  │
│  │  │     WAF     │───►│     CDN     │───►│   Workers   │                   │  │
│  │  │  (DDoS/Bot) │    │  (Static)   │    │ (Edge Func) │                   │  │
│  │  └─────────────┘    └─────────────┘    └──────┬──────┘                   │  │
│  └───────────────────────────────────────────────┼───────────────────────────┘  │
│                                                   │                              │
│  ┌───────────────────────────────────────────────┼───────────────────────────┐  │
│  │                         Railway (Compute)      │                           │  │
│  │  ┌────────────────────────────────────────────▼────────────────────────┐  │  │
│  │  │                        Skopaq Brain (FastAPI)                        │  │  │
│  │  └────────────────────────────────┬────────────────────────────────────┘  │  │
│  └───────────────────────────────────┼───────────────────────────────────────┘  │
│                                      │                                           │
│  ┌───────────────────────────────────┼───────────────────────────────────────┐  │
│  │                    Supabase (US or EU Region)                             │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │  │
│  │  │   PostgreSQL    │  │      Auth       │  │   Realtime      │           │  │
│  │  │   + pgvector    │  │   (Sessions)    │  │   (WebSocket)   │           │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                           │
│  ┌───────────────────────────────────┼───────────────────────────────────────┐  │
│  │                    Cloudflare R2 (Object Storage)                         │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │  │
│  │  │   Screenshots   │  │   Recordings    │  │     Reports     │           │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                           │
└──────────────────────────────────────┼──────────────────────────────────────────┘
                                       │ LLM API Calls
                                       ▼
                              ┌─────────────────┐
                              │  Anthropic API  │
                              │   (US Region)   │
                              └─────────────────┘
```

---

## Air-Gap Data Isolation Guarantees

### Contractual Guarantees

When deployed in air-gap mode, Skopaq Enterprise provides the following guarantees:

1. **Zero External Network Calls**
   - All LLM inference runs locally via Ollama
   - No license validation or telemetry calls
   - No external API dependencies
   - Deployment can operate in a fully disconnected network

2. **Complete Data Isolation**
   - All test data stored in customer-owned PostgreSQL
   - All artifacts stored in customer-owned MinIO/S3
   - All session data stored in customer-owned Redis
   - All LLM conversations processed locally (no external transmission)

3. **No Data Exfiltration Vectors**
   - No analytics or usage tracking
   - No error reporting to external services
   - No automatic update checks
   - All logging is local only

4. **Verifiable Isolation**
   - Network policy templates provided to block all egress
   - Audit mode available to log any external connection attempts
   - Container images scanned and signed for supply chain security

### Technical Implementation

```yaml
# Network Policy to enforce air-gap
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: argus-airgap-egress-deny
  namespace: argus
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    # Allow only intra-namespace communication
    - to:
        - namespaceSelector:
            matchLabels:
              name: argus
    # Allow DNS resolution
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: UDP
          port: 53
```

### Air-Gap Validation Checklist

| Check | Command | Expected Result |
|-------|---------|-----------------|
| No external DNS | `kubectl exec -n argus deploy/argus-brain -- nslookup anthropic.com` | Should fail |
| No external HTTPS | `kubectl exec -n argus deploy/argus-brain -- curl -I https://api.anthropic.com` | Should timeout |
| Ollama accessible | `kubectl exec -n argus deploy/argus-brain -- curl http://argus-ollama:11434/api/tags` | Should return model list |
| PostgreSQL accessible | `kubectl exec -n argus deploy/argus-brain -- pg_isready -h argus-postgresql` | Should succeed |

---

## EU Data Residency Configuration

For organizations requiring EU data residency, Skopaq supports the following configurations:

### Self-Hosted EU Deployment

```yaml
# values-eu.yaml
global:
  region: "eu-west-1"

postgresql:
  # Deploy PostgreSQL in EU
  nodeSelector:
    topology.kubernetes.io/region: eu-west-1

minio:
  # Object storage in EU
  nodeSelector:
    topology.kubernetes.io/region: eu-west-1

brain:
  env:
    # Use Azure OpenAI in EU
    LLM_PROVIDER: "azure"
    AZURE_OPENAI_ENDPOINT: "https://your-eu-resource.openai.azure.com"
    # Or use AWS Bedrock EU
    # LLM_PROVIDER: "bedrock"
    # AWS_REGION: "eu-west-1"
```

### Cloud (SaaS) EU Residency

For Skopaq Cloud customers requiring EU data residency:

| Component | EU Location | Notes |
|-----------|-------------|-------|
| Supabase PostgreSQL | Frankfurt (eu-central-1) | Primary data storage |
| Cloudflare R2 | EU region | Object storage |
| LLM Processing | Azure OpenAI (EU) | Optional, additional cost |

Contact enterprise@skopaq.ai to enable EU data residency for cloud deployments.

---

## Sub-Processors List (Cloud Mode Only)

The following sub-processors are used in Skopaq Cloud (SaaS) mode:

| Sub-Processor | Purpose | Location | Data Processed | DPA Available |
|---------------|---------|----------|----------------|---------------|
| **Supabase** | Database, Auth | US (default), EU (optional) | All relational data, user accounts | Yes |
| **Cloudflare** | CDN, WAF, Workers, R2 Storage | Global (US primary) | Request routing, static assets, object storage | Yes |
| **Railway** | Application hosting | US | Application runtime, logs | Yes |
| **Anthropic** | LLM inference | US | Prompts and responses (transient) | Yes |
| **Clerk** | Authentication | US | User identity, sessions | Yes |
| **Sentry** | Error tracking | US | Error logs, stack traces (no PII) | Yes |
| **Langfuse** | LLM observability | EU (Frankfurt) | LLM traces (anonymized) | Yes |

### Sub-Processor Changes

We maintain a changelog of sub-processor changes:

| Date | Change | Sub-Processor | Impact |
|------|--------|---------------|--------|
| 2026-01-01 | Added | Langfuse | LLM observability |
| 2025-11-15 | Added | Sentry | Error tracking |
| 2025-09-01 | Initial | Supabase, Cloudflare, Railway, Anthropic, Clerk | Platform launch |

Customers will be notified 30 days in advance of any sub-processor changes.

---

## Compliance Certifications

### Current Certifications

| Certification | Status | Scope | Evidence |
|---------------|--------|-------|----------|
| SOC 2 Type II | In Progress (Q2 2026) | All cloud services | [SOC2_EVIDENCE.md](/docs/SOC2_EVIDENCE.md) |
| GDPR | Compliant | All deployments | DPA available |
| CCPA | Compliant | All deployments | Privacy policy |
| ISO 27001 | Planned (Q4 2026) | All services | - |
| HIPAA | Ready (with BAA) | Self-hosted only | BAA template available |
| FedRAMP | Planned (2027) | Self-hosted | - |

### HIPAA Compliance (Self-Hosted)

For healthcare organizations, Skopaq self-hosted can be deployed in HIPAA-compliant configuration:

1. **Business Associate Agreement (BAA)**: Available upon request
2. **PHI Handling**: No PHI processed by default; if testing PHI systems, all data stays in customer infrastructure
3. **Audit Controls**: Comprehensive audit logging enabled
4. **Encryption**: AES-256 at rest, TLS 1.3 in transit

### FedRAMP Readiness (Self-Hosted)

For US government agencies:

1. **Air-Gap Deployment**: Meets disconnected operation requirements
2. **FIPS 140-2**: Compatible with FIPS-validated encryption modules
3. **Supply Chain**: Container images from trusted registries, SBOM available
4. **Audit Logging**: Meets NIST 800-53 logging requirements

---

## Data Retention

### Default Retention Periods

| Data Type | Cloud Retention | Self-Hosted Retention | Configurable |
|-----------|-----------------|----------------------|--------------|
| Test definitions | Indefinite | Customer controlled | Yes |
| Test results | 90 days | Customer controlled | Yes |
| Screenshots | 30 days | Customer controlled | Yes |
| Recordings | 30 days | Customer controlled | Yes |
| Audit logs | 1 year | Customer controlled | Yes |
| Healing patterns | Indefinite | Customer controlled | Yes |

### Data Deletion

- **Cloud**: Data deleted within 30 days of account termination
- **Self-Hosted**: Customer controls all data deletion
- **Right to Erasure**: GDPR Article 17 requests honored within 30 days

---

## Security Controls Summary

| Control | Cloud Mode | Self-Hosted | Air-Gap |
|---------|------------|-------------|---------|
| Encryption at Rest | AES-256 (provider managed) | AES-256 (customer managed) | AES-256 (customer managed) |
| Encryption in Transit | TLS 1.3 | TLS 1.2+ (configurable) | TLS 1.2+ (configurable) |
| Multi-Tenancy | Logical (RLS) | Dedicated instance | Dedicated instance |
| Data Location | US/EU | Customer choice | Customer choice |
| Backup | Automatic | Customer managed | Customer managed |
| Audit Logging | Enabled | Configurable | Configurable |
| External Dependencies | Yes | Minimal | None |

---

## Contact Information

For data residency questions or to request compliance documentation:

- **Enterprise Sales**: enterprise@skopaq.ai
- **Security Team**: security@skopaq.ai
- **DPA Requests**: legal@skopaq.ai
- **Documentation**: https://docs.skopaq.ai/compliance

---

## Related Documents

- [Self-Hosted Overview](overview.md)
- [LLM Configuration](llm-configuration.md)
- [Storage Configuration](storage.md)
- [Networking](networking.md)
- [SOC2 Evidence](/docs/SOC2_EVIDENCE.md)
- [Data Processing Agreement](/docs/compliance/DATA_PROCESSING_AGREEMENT.md)
- [Information Security Policy](/docs/security/policies/information-security-policy.md)
