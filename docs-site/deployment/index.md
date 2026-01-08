# Deployment Options

**ARGUS works with or without code access. Choose the mode that fits your needs.**

---

## Overview

| Mode | Code Access | Healing Accuracy | Best For |
|------|-------------|------------------|----------|
| **DOM-Only** | None required | 90-95% | Privacy-first, quick POCs, agencies |
| **Git-Aware** | Read-only | 99.9% | Teams wanting maximum reliability |
| **Self-Hosted** | On-premise | 99.9% | Enterprise, regulated industries |

!!! info "Key Point"
    Git-aware healing is a **PREMIUM feature**, not a requirement. ARGUS works great with **zero code access**.

---

## DOM-Only Mode (Default)

**No code access required.** This is the default mode.

### What We Analyze

```
✅ Live DOM structure of running app
✅ Visual screenshots using AI vision
✅ Element attributes, text content, positions
✅ Historical test runs from our database
✅ CSS selectors and element relationships
```

### What You Still Get

| Feature | Available |
|---------|:---------:|
| AI test generation from plain English | ✅ |
| Visual regression testing | ✅ |
| Self-healing at 90-95% accuracy | ✅ |
| Multi-model cost savings (60-80%) | ✅ |
| Full UI/API/Database testing | ✅ |
| All 23 AI agents | ✅ |
| Production monitoring integration | ✅ |

### Healing Process

1. Test fails → Selector not found
2. Analyze current DOM structure
3. Use visual AI to identify element
4. Calculate semantic similarity
5. Suggest replacement based on DOM patterns
6. Cache solution for future runs

**Accuracy: 90-95%** (matches Mabl, testRigor, Katalon)

---

## Git-Aware Mode (Premium)

**Read-only repository access for maximum healing accuracy.**

### Security Guarantees

```
✅ READ-ONLY access (never writes to repository)
✅ Code analyzed in memory, never stored permanently
✅ SOC2 compliant, encrypted in transit
✅ Revoke access anytime via GitHub/GitLab settings
✅ Same permissions as CircleCI, GitHub Actions
```

### Additional Features

| Feature | DOM-Only | Git-Aware |
|---------|:--------:|:---------:|
| 99.9% healing accuracy | ❌ | ✅ |
| Commit attribution | ❌ | ✅ |
| Component rename handling | ❌ | ✅ |
| Developer blame info | ❌ | ✅ |

### Healing Process

1. Test fails → Selector not found
2. Query git blame → Find who changed the file
3. Parse source code → Extract current selectors
4. Compare old vs new → Understand the rename
5. Apply fix with **99.9% confidence**
6. Show: "Changed by @sarah in commit abc123"

---

## Self-Hosted Mode (Enterprise)

**Full deployment in your infrastructure.**

### Deployment Options

- AWS (EKS, ECS, EC2)
- Google Cloud (GKE, Cloud Run)
- Azure (AKS, Container Instances)
- On-premise Kubernetes
- Docker Compose
- Air-gapped networks

### Security Benefits

```
✅ Code NEVER leaves your network
✅ Full git-aware features with complete privacy
✅ Customer controls all data retention
✅ Meets HIPAA, PCI-DSS, FedRAMP requirements
✅ Custom audit logging to your SIEM
```

---

## Feature Availability by Mode

| Feature | DOM-Only | Git-Aware | Self-Hosted |
|---------|:--------:|:---------:|:-----------:|
| AI test generation (NLP) | ✅ | ✅ | ✅ |
| Visual regression testing | ✅ | ✅ | ✅ |
| Multi-model cost savings | ✅ | ✅ | ✅ |
| UI/API/Database testing | ✅ | ✅ | ✅ |
| Production learning | ✅ | ✅ | ✅ |
| 23 specialized AI agents | ✅ | ✅ | ✅ |
| DOM-based healing (90-95%) | ✅ | ✅ | ✅ |
| **Git-aware healing (99.9%)** | ❌ | ✅ | ✅ |
| **Commit attribution** | ❌ | ✅ | ✅ |
| **Refactor handling** | ❌ | ✅ | ✅ |
| **Air-gapped deployment** | ❌ | ❌ | ✅ |
| **Custom data retention** | ❌ | ❌ | ✅ |

---

## Pricing Tier Mapping

| Tier | Price | Deployment Modes |
|------|-------|------------------|
| Free | $0/mo | DOM-Only |
| Starter | $49/mo | DOM-Only |
| Pro | $99/mo | DOM-Only + Git-Aware |
| Team | $299/mo | DOM-Only + Git-Aware |
| Enterprise | Custom | All (including Self-Hosted) |

---

## FAQ

For detailed questions about deployment options, see the [Deployment FAQ](faq.md).

---

*Last Updated: January 2026*
