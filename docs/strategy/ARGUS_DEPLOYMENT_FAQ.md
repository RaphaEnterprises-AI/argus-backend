# ARGUS Deployment FAQ

**Frequently Asked Questions About Deployment Options & Data Security**

---

## Table of Contents

1. [General Questions](#general-questions)
2. [DOM-Only Mode](#dom-only-mode)
3. [Git-Aware Mode](#git-aware-mode)
4. [Self-Hosted / On-Premise](#self-hosted--on-premise)
5. [Security & Compliance](#security--compliance)
6. [Pricing & Features](#pricing--features)
7. [Technical Questions](#technical-questions)
8. [Migration & Upgrades](#migration--upgrades)

---

## General Questions

### Q: Do I need to share my source code to use Argus?

**No.** Argus works perfectly without any access to your source code. Our DOM-Only mode analyzes your running application and delivers 90-95% self-healing accuracy—matching or exceeding industry leaders like Mabl, testRigor, and Katalon.

Git-aware healing (99.9% accuracy) is an optional premium feature for teams who want maximum reliability.

---

### Q: What are the different deployment options?

Argus offers three deployment modes:

| Mode | Code Access | Healing Accuracy | Best For |
|------|-------------|------------------|----------|
| **DOM-Only** | None required | 90-95% | Most customers, quick setup |
| **Git-Aware** | Read-only repo access | 99.9% | Teams wanting maximum accuracy |
| **Self-Hosted** | On-premise deployment | 99.9% | Enterprise, regulated industries |

---

### Q: Which mode should I choose?

**Choose DOM-Only if:**
- You want to get started quickly without any setup
- You prefer not to grant repository access
- You're testing third-party applications
- You're an agency testing client applications
- You want to evaluate Argus before deeper integration

**Choose Git-Aware if:**
- You want the highest possible healing accuracy (99.9%)
- Your team frequently refactors code
- You want to know exactly which commit broke a test
- You're comfortable with read-only repository access

**Choose Self-Hosted if:**
- You have strict data residency requirements
- You're in a regulated industry (healthcare, finance, government)
- Your security policy prohibits external code access
- You need air-gapped deployment

---

### Q: Can I switch between modes later?

**Yes.** You can upgrade from DOM-Only to Git-Aware at any time by connecting your repository. You can also downgrade or disconnect repository access whenever you want.

For Self-Hosted, contact our sales team for migration assistance.

---

## DOM-Only Mode

### Q: What does Argus analyze in DOM-Only mode?

In DOM-Only mode, Argus analyzes:

- **Live DOM structure** of your running application
- **Visual screenshots** using AI vision models
- **Element attributes** (IDs, classes, data attributes, text content)
- **Element positions and relationships** in the page
- **Historical test runs** stored in our database
- **CSS selectors and XPath patterns**

We do NOT access:
- Your source code repository
- Git history
- Build artifacts
- CI/CD pipelines

---

### Q: What features are available in DOM-Only mode?

**Full feature set available:**

| Feature | Available in DOM-Only |
|---------|:--------------------:|
| AI test generation from plain English | ✅ |
| Visual regression testing | ✅ |
| Self-healing (90-95% accuracy) | ✅ |
| Multi-model cost savings (60-80%) | ✅ |
| UI testing | ✅ |
| API testing | ✅ |
| Database testing | ✅ |
| All 23 AI agents | ✅ |
| Production monitoring integration | ✅ |
| MCP IDE integration | ✅ |
| Test reports & analytics | ✅ |

**Not available in DOM-Only:**

| Feature | Requires Git-Aware |
|---------|:------------------:|
| 99.9% healing accuracy | ✅ |
| Commit attribution ("Changed by @sarah in abc123") | ✅ |
| Component rename handling | ✅ |
| Git blame integration | ✅ |

---

### Q: How does self-healing work without code access?

In DOM-Only mode, our self-healing uses:

1. **DOM Analysis** - When a selector fails, we analyze the current DOM structure to find similar elements
2. **Visual AI** - We use AI vision to identify elements by their visual appearance
3. **Semantic Matching** - We match elements based on text content, position, and surrounding context
4. **Historical Patterns** - We learn from your previous test runs to predict selector changes
5. **Attribute Scoring** - We rank potential matches based on stability scores

This achieves 90-95% accuracy, which matches the industry standard.

---

### Q: Is 90-95% healing accuracy good enough?

**Yes, for most use cases.** Here's how we compare:

| Tool | Self-Healing Accuracy |
|------|----------------------|
| **Argus DOM-Only** | 90-95% |
| Mabl | ~95% |
| testRigor | ~90% |
| Katalon | ~90% |
| Applitools | ~85% |

Our DOM-Only mode is competitive with industry leaders. Git-Aware mode (99.9%) is for teams who need maximum reliability and want to eliminate nearly all maintenance.

---

## Git-Aware Mode

### Q: What repository access does Argus need?

Argus requires **read-only access** to your repository. We:

- ✅ **Read** source code files
- ✅ **Read** git history (commits, blame, diffs)
- ❌ **Never write** to your repository
- ❌ **Never store** your code permanently
- ❌ **Never share** your code with third parties

---

### Q: How do I grant repository access?

**GitHub:**
1. Go to Settings > Integrations in Argus dashboard
2. Click "Connect GitHub"
3. Authorize the Argus GitHub App
4. Select which repositories to grant access to

**GitLab:**
1. Go to Settings > Integrations in Argus dashboard
2. Click "Connect GitLab"
3. Create a read-only access token
4. Paste the token in Argus

**Bitbucket:**
1. Go to Settings > Integrations in Argus dashboard
2. Click "Connect Bitbucket"
3. Authorize the Argus app
4. Select repositories

---

### Q: How is my code protected?

**Security measures:**

| Protection | Description |
|------------|-------------|
| **Read-only access** | We never write to your repository |
| **In-memory analysis** | Code is analyzed in memory, not stored on disk |
| **No permanent storage** | Code is discarded after analysis |
| **Encrypted transit** | All data encrypted with TLS 1.3 |
| **SOC2 compliant** | Full audit logging of all access |
| **Revocable access** | Disconnect anytime via your Git provider |

---

### Q: What does Argus do with my code?

When you grant repository access, Argus:

1. **Reads changed files** when a test fails
2. **Queries git blame** to find who changed a selector
3. **Parses source code** to extract current selectors
4. **Compares old vs new** to understand renames
5. **Discards the code** after analysis

We do NOT:
- Store your code in our database
- Train AI models on your code
- Share your code with anyone
- Keep copies of your repository

---

### Q: Can I limit which repositories Argus can access?

**Yes.** When connecting GitHub/GitLab/Bitbucket, you can:

- Grant access to specific repositories only
- Exclude sensitive repositories
- Revoke access to individual repos at any time

---

### Q: What's the difference between 95% and 99.9% accuracy?

| Scenario | DOM-Only (95%) | Git-Aware (99.9%) |
|----------|----------------|-------------------|
| 100 tests, 10 selector changes | ~9-10 auto-healed | ~10 auto-healed |
| 500 tests, 50 selector changes | ~45-48 auto-healed | ~50 auto-healed |
| 1000 tests, 100 selector changes | ~90-95 auto-healed | ~99-100 auto-healed |

**The real difference:** Git-Aware handles **component renames during refactoring**, which DOM-Only cannot. If your team frequently refactors, Git-Aware eliminates nearly all false failures.

---

## Self-Hosted / On-Premise

### Q: What is Self-Hosted deployment?

Self-Hosted means Argus runs entirely within your own infrastructure. Your code and data never leave your network.

---

### Q: What infrastructure do I need?

**Minimum requirements:**

| Component | Requirement |
|-----------|-------------|
| **Compute** | 4 vCPUs, 16GB RAM (Kubernetes or Docker) |
| **Database** | PostgreSQL 14+ |
| **Storage** | 100GB+ for screenshots and reports |
| **Network** | Outbound HTTPS for LLM API calls |

**Supported platforms:**
- AWS (EKS, ECS, EC2)
- Google Cloud (GKE, Cloud Run)
- Azure (AKS, Container Instances)
- On-premise Kubernetes
- Docker Compose (small deployments)

---

### Q: Can Argus run in an air-gapped environment?

**Yes, with limitations.** In air-gapped environments:

- ✅ Core test execution works
- ✅ DOM-based healing works
- ✅ Visual comparison works (local models)
- ⚠️ AI features require local LLM deployment (Llama, etc.)
- ⚠️ Multi-model routing limited to on-premise models

Contact our enterprise team for air-gapped deployment guidance.

---

### Q: What about LLM API calls in Self-Hosted?

By default, Self-Hosted Argus makes outbound API calls to LLM providers (Anthropic, OpenAI, Google). For fully air-gapped deployments, you can:

1. **Use local LLMs** - Deploy Llama 3, Mistral, or other open models
2. **Use Azure OpenAI** - Keep traffic within Azure network
3. **Use AWS Bedrock** - Keep traffic within AWS network
4. **Use Google Vertex AI** - Keep traffic within GCP network

---

### Q: How do updates work for Self-Hosted?

We provide:
- **Docker images** via private registry
- **Helm charts** for Kubernetes
- **Release notes** with upgrade instructions
- **Migration scripts** for database changes

You control when to update your deployment.

---

## Security & Compliance

### Q: Is Argus SOC2 compliant?

**Yes.** Argus is SOC2 Type II compliant. We can provide our SOC2 report upon request under NDA.

---

### Q: What compliance certifications does Argus have?

| Standard | Status |
|----------|--------|
| SOC2 Type II | ✅ Certified |
| ISO 27001 | ✅ Certified |
| GDPR | ✅ Compliant |
| HIPAA | ✅ Available (requires BAA) |
| PCI-DSS | ⚠️ Self-Hosted only |
| FedRAMP | ⚠️ Self-Hosted only |

---

### Q: Where is my data stored?

**Cloud deployment:**
- Data stored in Supabase (PostgreSQL) in US-East
- Screenshots stored in Cloudflare R2 (global CDN)
- EU data residency available on Enterprise plans

**Self-Hosted:**
- All data stored in your infrastructure
- You control data residency completely

---

### Q: Do you train AI models on my data?

**No.** We do not use your test data, code, or screenshots to train AI models. Your data is used only to provide the Argus service to you.

---

### Q: How long do you retain my data?

| Data Type | Retention (Cloud) | Retention (Self-Hosted) |
|-----------|-------------------|------------------------|
| Test results | 90 days (configurable) | You control |
| Screenshots | 30 days (configurable) | You control |
| Audit logs | 1 year | You control |
| Source code | Never stored | Never stored |

---

### Q: Can I get a DPA (Data Processing Agreement)?

**Yes.** We provide DPAs for all Enterprise customers. Contact sales@argus.dev for details.

---

## Pricing & Features

### Q: Which deployment mode is included in each plan?

| Plan | Price | DOM-Only | Git-Aware | Self-Hosted |
|------|-------|:--------:|:---------:|:-----------:|
| Free | $0/mo | ✅ | ❌ | ❌ |
| Starter | $49/mo | ✅ | ❌ | ❌ |
| Pro | $99/mo | ✅ | ✅ | ❌ |
| Team | $299/mo | ✅ | ✅ | ❌ |
| Enterprise | Custom | ✅ | ✅ | ✅ |

---

### Q: Is there a free trial of Git-Aware mode?

**Yes.** Contact us for a 14-day trial of Git-Aware features on any plan.

---

### Q: What's the cost difference between modes?

The deployment mode is included in your plan—there's no additional cost for Git-Aware on Pro+ plans. Self-Hosted requires an Enterprise agreement with custom pricing based on your infrastructure needs.

---

## Technical Questions

### Q: Which Git providers do you support?

| Provider | Support |
|----------|---------|
| GitHub (Cloud & Enterprise) | ✅ Full |
| GitLab (Cloud & Self-Managed) | ✅ Full |
| Bitbucket (Cloud & Server) | ✅ Full |
| Azure DevOps | ✅ Full |
| AWS CodeCommit | ⚠️ Coming Q2 2026 |

---

### Q: Which languages/frameworks does Git-Aware healing support?

| Framework | Support |
|-----------|---------|
| React / Next.js | ✅ Full |
| Vue / Nuxt | ✅ Full |
| Angular | ✅ Full |
| Svelte / SvelteKit | ✅ Full |
| Plain HTML/CSS/JS | ✅ Full |
| TypeScript | ✅ Full |
| Python (Django, Flask) | ✅ Backend analysis |
| Node.js (Express, Fastify) | ✅ Backend analysis |

---

### Q: How quickly does Git-Aware healing respond?

| Operation | Typical Time |
|-----------|-------------|
| Git blame query | 50-200ms |
| Source code parsing | 100-500ms |
| Selector extraction | 50-100ms |
| Total healing time | 200-800ms |

Git-Aware healing adds minimal latency compared to DOM-Only.

---

### Q: Can I use Git-Aware for some projects and DOM-Only for others?

**Yes.** You can configure deployment mode per project:

```
Project A: Git-Aware (connected to repo)
Project B: DOM-Only (no repo connection)
Project C: DOM-Only (third-party app)
```

---

## Migration & Upgrades

### Q: How do I upgrade from DOM-Only to Git-Aware?

1. Go to **Settings > Integrations** in your project
2. Click **Connect Repository**
3. Authorize Argus to read your repository
4. Existing tests will automatically use Git-Aware healing

No test changes required—it's seamless.

---

### Q: Will I lose my test history if I switch modes?

**No.** All your test history, results, and healing patterns are preserved when switching modes. Git-Aware simply adds additional context for future healing.

---

### Q: How do I migrate to Self-Hosted?

1. Contact our Enterprise team
2. We'll provision your Self-Hosted license
3. Deploy using our Docker images or Helm charts
4. Export your data from cloud using our migration tool
5. Import into your Self-Hosted instance

Typical migration takes 1-2 weeks with our support.

---

### Q: Can I migrate from Self-Hosted back to Cloud?

**Yes.** We provide export tools to migrate data from Self-Hosted to Cloud. Contact support for assistance.

---

## Still Have Questions?

- **Documentation:** docs.argus.dev
- **Email:** support@argus.dev
- **Sales:** sales@argus.dev
- **Enterprise:** enterprise@argus.dev

---

*Last Updated: January 2026*
