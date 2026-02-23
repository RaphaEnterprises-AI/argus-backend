# IndiaAI Compute Portal - Project Proposal

## Applicant Information

| Field | Details |
|-------|---------|
| **Organization** | Rapha Enterprises AI (trading as Skopaq) |
| **Entity Type** | Sole Proprietorship |
| **Udyam Registration** | [To be obtained - MSME Micro Enterprise] |
| **GSTIN** | [Your GST number] |
| **PAN** | [Your PAN] |
| **Website** | https://skopaq.ai |
| **Contact Email** | team@skopaq.ai |
| **Sector** | AI/ML - Developer Tools & Quality Assurance |
| **Application Category** | MSME (via Udyam Registration) |
| **Stage** | Growth (Product in production, paying customers) |

---

## 1. Project Title

**Skopaq: GPU-Accelerated Autonomous Software Quality Intelligence Platform**

---

## 2. Executive Summary

Skopaq is an autonomous end-to-end software testing platform that uses agentic AI to understand codebases, generate test plans, execute tests, self-heal broken tests, and provide quality intelligence — all without human intervention. Our platform currently serves enterprises through a SaaS model at https://app.skopaq.ai.

We seek access to IndiaAI compute infrastructure (NVIDIA H100/H200 GPUs) to:

1. **Self-host embedding models** — Replace third-party embedding API calls (Cohere) with self-hosted models, reducing per-query latency from ~200ms to ~5ms and eliminating recurring API costs
2. **Fine-tune a domain-specific healing model** — Train a specialized model on 50,000+ failure-fix pairs from our production knowledge graph to improve self-healing accuracy from ~72% to >90%
3. **GPU-accelerated visual regression** — Run perceptual image comparison (SSIM, perceptual hashing) on GPU for 10x faster screenshot-based test validation
4. **Batch intelligence pre-computation** — Nightly GPU-powered clustering, impact analysis, and flaky test detection across customer codebases

This compute access will enable Skopaq to reduce operational costs by ~60%, improve core AI accuracy, and maintain competitive pricing for Indian startups and enterprises adopting AI-powered quality assurance.

---

## 3. Problem Statement

### The Software Quality Crisis

- **68% of software releases** contain bugs that reach production (Stripe Developer Report, 2025)
- **Manual E2E testing** takes 40-60% of development cycle time
- **Test maintenance** is the #1 reason teams abandon automation — selectors break, APIs change, UIs evolve
- **Indian enterprises** spend $2.4B annually on QA (NASSCOM, 2025), with 70% still relying on manual testing

### Current Limitations Without GPU Access

| Capability | Current (CPU/API) | With GPU | Improvement |
|-----------|-------------------|----------|-------------|
| Embedding generation | 200ms/query (Cohere API) | 5ms/query (self-hosted) | 40x faster |
| Self-healing accuracy | 72% (generic LLM) | >90% (fine-tuned) | +25% accuracy |
| Visual regression | 2.5s/comparison (CPU) | 250ms/comparison (GPU) | 10x faster |
| Nightly batch jobs | 4.5 hours (CPU) | 25 minutes (GPU) | 10x faster |
| Monthly embedding costs | ~$1,200 (API fees) | ~$200 (GPU compute) | 83% reduction |

---

## 4. Technical Approach

### 4.1 Architecture Overview

Skopaq's architecture consists of 30+ AI agents orchestrated by LangGraph, communicating via Kafka (Redpanda), with a knowledge graph (FalkorDB + pgvector) for semantic memory. GPU compute slots into four specific workloads:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SKOPAQ GPU WORKLOADS                               │
├──────────────────┬──────────────────┬──────────────────┬────────────┤
│   EMBEDDINGS     │   FINE-TUNING    │  VISUAL AI       │  BATCH     │
│   (Real-time)    │   (Periodic)     │  (On-demand)     │  (Nightly) │
├──────────────────┼──────────────────┼──────────────────┼────────────┤
│ • Cohere embed   │ • LoRA/QLoRA on  │ • SSIM on GPU    │ • Failure  │
│   multilingual   │   Mistral 7B or  │ • Perceptual     │   cluster  │
│   v3.0 → self-   │   Llama 3.1 8B   │   hashing        │ • Impact   │
│   hosted e5-     │ • 50K+ failure-  │ • DOM diffing    │   matrix   │
│   large-v2       │   fix pairs from │   with visual    │ • Flaky    │
│ • 1024-dim       │   production     │   context        │   ranking  │
│   vectors        │ • Healing agent  │                  │ • Coverage │
│ • ~500K queries/ │   specialization │                  │   gaps     │
│   month          │                  │                  │            │
└──────────────────┴──────────────────┴──────────────────┴────────────┘
          │                  │                  │                │
          ▼                  ▼                  ▼                ▼
    H100 (1 GPU)       H100 (2 GPUs)     H100 (1 GPU)    H100 (1 GPU)
    Always-on          Training runs      On-demand        Nightly batch
    ~720 hrs/mo        ~120 hrs/mo        ~150 hrs/mo      ~80 hrs/mo
```

### 4.2 Workload 1: Self-Hosted Embeddings (Real-Time)

**Current state**: Every semantic search query (failure pattern matching, code context lookup, documentation search) calls Cohere's `embed-multilingual-v3.0` API at ~200ms latency and $0.0001/query.

**Proposed change**: Self-host `intfloat/multilingual-e5-large-v2` (560M params) on a single H100 GPU using vLLM or Text Embeddings Inference (TEI) server.

**Technical details**:
- Model: `multilingual-e5-large-v2` (1024-dim, matches current Cohere output)
- Framework: Hugging Face TEI (optimized CUDA kernels, dynamic batching)
- Throughput: ~3,000 embeddings/sec on H100 (vs. ~50/sec API rate limit)
- Memory: ~2.5 GB VRAM (leaves headroom for batching)
- Integration: Replace `CohereEmbeddings` client in `src/knowledge/cognee_client.py` with local HTTP endpoint

**Expected impact**:
- Latency: 200ms → 5ms per query (40x improvement)
- Cost: $1,200/month → $200/month (GPU compute share)
- Availability: No rate limits, no external dependency, works in air-gapped deployments

### 4.3 Workload 2: Domain-Specific Healing Model (Fine-Tuning)

**Current state**: The Self-Healer Agent (`src/agents/self_healer.py`) uses generic Claude/GPT models for generating test fixes. While powerful, these models lack domain-specific knowledge about common UI framework patterns, selector evolution, and API migration patterns.

**Proposed change**: Fine-tune a 7-8B parameter open-source model (Mistral 7B v0.3 or Llama 3.1 8B) on our proprietary dataset of 50,000+ failure-fix pairs extracted from the production knowledge graph.

**Technical details**:
- Base model: Mistral 7B v0.3 (Apache 2.0 license) or Llama 3.1 8B
- Method: QLoRA (4-bit quantization + LoRA adapters, rank=64)
- Dataset: 50K+ structured pairs from `failure_patterns` and `healing_history` tables
  ```json
  {
    "input": "Error: selector '#login-btn' not found. Page: /auth/login. Framework: React. Previous selector: '#login-button'. DOM context: ...",
    "output": "Fix: Update selector to 'button[data-testid=\"login\"]'. Reason: Component refactored to use data-testid attributes. Confidence: 0.94"
  }
  ```
- Training: 3 epochs, batch size 8, learning rate 2e-5, gradient accumulation 4
- VRAM: ~40 GB (QLoRA on 7B model) — fits on 1x H100 (80 GB)
- Training time: ~8 hours per run on 2x H100
- Inference: Deploy quantized (GPTQ 4-bit) on single H100, ~50 tokens/sec

**Expected impact**:
- Healing accuracy: 72% → >90% on common failure patterns
- Healing latency: 3-5 seconds (API call) → 500ms (local inference)
- Domain coverage: Specialized knowledge of React, Vue, Angular, Playwright, Cypress selector patterns
- IP asset: Proprietary model becomes core competitive advantage

### 4.4 Workload 3: GPU-Accelerated Visual Regression (On-Demand)

**Current state**: Visual AI (`src/agents/visual_ai.py`) uses CPU-based SSIM and perceptual hashing via `scikit-image` and `imagehash`. Processing a single 1920x1080 screenshot comparison takes ~2.5 seconds.

**Proposed change**: Offload image comparison to GPU using CUDA-accelerated libraries (cuCIM, RAPIDS, or custom CUDA kernels via PyTorch).

**Technical details**:
- Libraries: PyTorch + torchvision (GPU SSIM), cuCIM (structural comparison)
- Batch processing: Compare up to 64 screenshot pairs simultaneously
- Resolution: 1920x1080 (standard) and 2560x1440 (retina)
- Integration: Add GPU backend option to `src/agents/visual_ai.py`

**Expected impact**:
- Per-comparison: 2.5s → 250ms (10x faster)
- Batch (100 screenshots): 4 minutes → 25 seconds
- Enables real-time visual regression during CI/CD pipelines

### 4.5 Workload 4: Batch Intelligence Pre-Computation (Nightly)

**Current state**: Nightly batch jobs (failure clustering, test impact matrix, flaky test ranking) run on CPU and take 4.5 hours for a medium-sized customer.

**Proposed change**: Use GPU-accelerated clustering (RAPIDS cuML) and matrix operations (cuPy) for batch intelligence jobs.

**Technical details**:
- Framework: RAPIDS cuML (GPU-accelerated scikit-learn compatible)
- Operations: DBSCAN clustering, cosine similarity matrix, PCA dimensionality reduction
- Data volume: ~100K test results, ~50K failure patterns per customer per month
- Schedule: Daily 2-4 AM UTC
- Integration: Replace CPU implementations in `src/intelligence/precomputed.py`

**Expected impact**:
- Batch job duration: 4.5 hours → 25 minutes
- Enables more frequent pre-computation (hourly instead of daily)
- Better freshness of quality intelligence metrics

---

## 5. Bill of Materials (GPU Compute)

### Monthly GPU Requirements

| Workload | GPU Type | GPUs | Hours/Month | Total GPU-Hours |
|----------|----------|------|-------------|-----------------|
| Embeddings (always-on) | H100 80GB | 1 | 720 | 720 |
| Fine-tuning (periodic) | H100 80GB | 2 | 60 | 120 |
| Visual regression (on-demand) | H100 80GB | 1 | 150 | 150 |
| Batch pre-computation (nightly) | H100 80GB | 1 | 80 | 80 |
| **Total** | | **5 peak / 2 avg** | | **1,070** |

### Cost Estimate

| Item | Rate (IndiaAI) | Monthly Cost | With 40% Subsidy |
|------|-----------------|--------------|-------------------|
| 1,070 GPU-hours (H100) | ~₹65/hr | ₹69,550 | ₹41,730 |
| Storage (model weights, datasets) | ₹5/GB/mo | ₹2,500 | ₹1,500 |
| Network egress | ₹2/GB | ₹1,000 | ₹600 |
| **Total monthly** | | **₹73,050** | **₹43,830** |
| **Total annual** | | **₹8,76,600** | **₹5,25,960** |

### Comparison with Commercial Cloud

| Provider | Monthly Cost (1,070 H100-hrs) | vs IndiaAI (subsidized) |
|----------|-------------------------------|--------------------------|
| AWS (p5.48xlarge) | ₹5,35,000 | 12x more expensive |
| GCP (a3-highgpu-8g) | ₹4,82,000 | 11x more expensive |
| Azure (ND H100 v5) | ₹5,10,000 | 12x more expensive |
| Lambda Labs | ₹2,14,000 | 5x more expensive |
| **IndiaAI (with 40% subsidy)** | **₹43,830** | **Baseline** |

---

## 6. Project Timeline

### Phase 1: Setup & Embedding Migration (Month 1-2)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-2 | Environment setup | GPU cluster access, CUDA drivers, container runtime |
| 3-4 | Embedding model deployment | TEI server running `multilingual-e5-large-v2` |
| 5-6 | Integration & testing | Replace Cohere API calls, A/B test quality |
| 7-8 | Production migration | Full cutover to self-hosted embeddings |

**Success criteria**: Embedding latency < 10ms, quality parity with Cohere (cosine similarity > 0.98 on test set)

### Phase 2: Healing Model Fine-Tuning (Month 2-4)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 9-10 | Dataset preparation | 50K+ failure-fix pairs curated and formatted |
| 11-12 | Training pipeline | QLoRA training scripts, evaluation harness |
| 13-14 | First training run | Baseline model, accuracy metrics |
| 15-16 | Iteration & optimization | Hyperparameter tuning, data augmentation |
| 17-18 | Deployment | GPTQ-quantized model serving via vLLM |

**Success criteria**: Healing accuracy > 90% on held-out test set, inference latency < 1 second

### Phase 3: Visual AI & Batch Acceleration (Month 4-6)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 19-20 | GPU visual regression pipeline | CUDA-accelerated SSIM + perceptual hash |
| 21-22 | Batch job migration | RAPIDS cuML clustering and matrix ops |
| 23-24 | End-to-end benchmarking | Full pipeline benchmarks, cost analysis |

**Success criteria**: 10x speedup on visual regression, batch jobs complete in < 30 minutes

### Phase 4: Optimization & Scaling (Month 6-12)

| Activity | Timeline | Description |
|----------|----------|-------------|
| Model iteration | Ongoing | Retrain healing model monthly with new data |
| Multi-tenant optimization | Month 7-8 | Efficient GPU sharing across customer workloads |
| H200 migration | Month 9-10 | Evaluate H200 for improved performance/watt |
| Open-source contribution | Month 10-12 | Publish benchmarks and learnings |

---

## 7. Impact & Outcomes

### Direct Impact

| Metric | Current | After GPU Access | Improvement |
|--------|---------|------------------|-------------|
| Embedding cost | $1,200/mo | $200/mo | 83% reduction |
| Self-healing accuracy | 72% | >90% | +25% |
| Visual regression speed | 2.5s/pair | 250ms/pair | 10x |
| Batch intelligence freshness | Daily | Hourly | 24x |
| Total AI infra cost | $4,500/mo | $1,800/mo | 60% reduction |

### Market Impact

- **Indian startup enablement**: Reduced infrastructure costs allow us to offer competitive pricing to Indian startups (current: $199/mo plan)
- **Make in India**: Self-hosted models reduce dependency on foreign AI API providers
- **Data sovereignty**: Customer code and test data never leave Indian infrastructure
- **Job creation**: Plan to grow engineering team from 5 to 15 within 12 months, all India-based

### Open-Source Commitments

1. **Publish benchmarks**: Self-hosted embedding performance vs. API providers on Indian infrastructure
2. **Release training pipeline**: QLoRA fine-tuning scripts for test-healing domain (without proprietary data)
3. **Contribute to RAPIDS**: Any optimizations made to visual regression pipelines
4. **IndiaAI case study**: Document cost savings and performance gains for other startups

### Alignment with National AI Mission Goals

| IndiaAI Pillar | Skopaq Contribution |
|----------------|---------------------|
| **Compute** | Demonstrating efficient GPU utilization for AI-powered developer tools |
| **Datasets** | Building India's largest test-failure knowledge graph (500K+ patterns) |
| **Innovation** | Novel agentic AI architecture with 30+ specialized agents |
| **Skilling** | Creating AI-native QA engineering roles (new job category) |
| **Startups** | Enabling Indian startups to ship higher-quality software faster |

---

## 8. Team

| Name | Role | Relevant Experience |
|------|------|---------------------|
| [Founder Name] | CEO & Lead Architect | [X] years in AI/ML, previously at [Company]. Built Skopaq's agentic AI architecture from ground up. |
| [CTO Name] | CTO | [X] years in infrastructure, Kubernetes, GPU computing. Manages Vultr VKE cluster and data layer. |
| [ML Lead Name] | ML Engineer | [X] years in NLP/embeddings, fine-tuning LLMs. Will lead healing model training. |
| [Platform Lead] | Platform Engineer | [X] years in distributed systems, Kafka, real-time streaming. |
| [QA Lead] | QA Architect | [X] years in test automation, Playwright, Selenium. Domain expert for training data curation. |

*[Fill in actual names and details]*

---

## 9. Existing Infrastructure

| Component | Provider | Details |
|-----------|----------|---------|
| **Application Backend** | Railway (India region) | FastAPI + LangGraph, auto-scaling |
| **Dashboard** | Vercel | Next.js 15, edge functions |
| **Database** | Supabase | PostgreSQL + pgvector + Auth |
| **Data Layer** | Vultr VKE (Mumbai) | Kubernetes: Redpanda, FalkorDB, Valkey, Flink |
| **Object Storage** | Cloudflare R2 | Screenshots, artifacts, model weights |
| **Monitoring** | Prometheus + Grafana | Full observability stack |
| **AI Providers** | Anthropic, OpenRouter | 300+ models via unified routing |

GPU access from IndiaAI will complement this existing infrastructure — the embedding server and model inference endpoints will be added as services in our Kubernetes cluster, accessed via internal networking.

---

## 10. Budget & Sustainability

### Year 1 (IndiaAI Subsidized)

| Category | Monthly | Annual |
|----------|---------|--------|
| GPU Compute (subsidized) | ₹43,830 | ₹5,25,960 |
| Storage & networking | ₹4,100 | ₹49,200 |
| Engineering time (ML ops) | ₹2,00,000 | ₹24,00,000 |
| **Total Year 1** | | **₹29,75,160** |

### Post-Subsidy Sustainability Plan

After IndiaAI subsidy period:
1. **Revenue growth**: Current MRR trajectory projects sufficient revenue to cover GPU costs at commercial rates by Month 12
2. **Efficiency gains**: Fine-tuned model replaces 70% of expensive API calls, net-positive ROI
3. **Reserved instances**: Negotiate long-term GPU contracts with Indian cloud providers (E2E Networks, Jio Cloud)
4. **Hybrid approach**: Keep embeddings self-hosted (highest ROI), use API for infrequent fine-tuning

---

## 11. Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| GPU allocation delays | Medium | Start with embedding workload (smallest GPU need), scale up |
| Fine-tuning doesn't improve accuracy | Low | Conservative target (90%), extensive evaluation harness, fallback to API |
| Model drift over time | Medium | Monthly retraining pipeline, A/B testing against API baseline |
| Data privacy concerns | Low | All customer data stays in Indian infrastructure, SOC 2 compliance planned |
| Team scaling challenges | Medium | Partner with IIT/IIIT internship programs for ML roles |

---

## 12. Appendix

### A. Product Screenshots

- Dashboard: https://app.skopaq.ai (live demo available)
- Documentation: https://docs.skopaq.ai

### B. Technical Publications

- Architecture documentation: https://docs.skopaq.ai/architecture
- API reference: https://docs.skopaq.ai/api

### C. Key Open-Source Dependencies

| Library | Purpose | License |
|---------|---------|---------|
| LangGraph | Agent orchestration | MIT |
| Playwright | Browser automation | Apache 2.0 |
| FastAPI | API framework | MIT |
| Cognee | Knowledge graph pipeline | Apache 2.0 |
| Hugging Face TEI | Embedding inference | Apache 2.0 |
| RAPIDS cuML | GPU-accelerated ML | Apache 2.0 |
| vLLM | LLM inference server | Apache 2.0 |

### D. Contact

- **Technical queries**: team@skopaq.ai
- **Product demo**: https://app.skopaq.ai
- **GitHub**: https://github.com/RaphaEnterprises-AI
