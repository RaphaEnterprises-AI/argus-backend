# Multi-Model Orchestration: Opus Orchestrator + Together AI Primary

**Date:** 2026-02-28
**Status:** Approved (v2 — expanded to Together AI as primary provider)
**Author:** Claude + bvk

## Problem

1. **Orchestrator uses mid-tier model**: The supervisor (`supervisor.py:311`) uses `get_default_api_model_id()` which resolves to Claude Sonnet 4.5. The orchestrator's routing decisions cascade through everything — a smarter model here reduces downstream waste.

2. **All models route through OpenRouter with 5.5% markup**: Every entry in the `MODELS` dict uses `ModelProvider.OPENROUTER`. Together AI offers the same (and better) models at direct pricing with no markup.

3. **Missing top agentic models**: Together AI hosts GLM-5 (#2 global agentic rank), Kimi K2.5 (#6), and MiniMax M2.5 (#10) — all with function calling support. These outperform our current models at lower cost.

## Solution

### 1. Upgrade Orchestrator to Claude Opus 4.6

- Add `ORCHESTRATOR_MODEL` to model registry (separate from `DEFAULT_MODEL`)
- Supervisor uses Opus 4.6 via Anthropic Direct for all routing decisions
- Increase `max_tokens` from 1024 to 4096
- Env var override: `ARGUS_ORCHESTRATOR_MODEL`

### 2. Together AI as Primary Provider (15+ models)

New `TogetherProvider` class covering all tiers:

| Tier | Models | Price Range |
|------|--------|-------------|
| FLASH | Gemma 3n, gpt-oss-20B, Llama 3.2 3B, Mistral Small 3 | $0.02-0.10 |
| VALUE | Qwen3-Next-80B, Qwen3 235B, Llama 4 Maverick, GLM-4.5-Air, MiniMax M2.5 | $0.15-0.30 |
| STANDARD | GLM-4.7, Kimi K2.5, Qwen3-Coder-Next, DeepSeek V3.1 | $0.45-0.60 |
| REASONING | Qwen3 235B Thinking, GLM-5, Kimi K2 Thinking | $0.65-1.20 |
| EXPERT | Qwen3-Coder 480B, DeepSeek-R1-0528 | $2.00-7.00 |

### 3. OpenRouter as Fallback Only

- If `TOGETHER_API_KEY` is not set, fall back to OpenRouter for Together-tier models
- Computer Use tasks still use Claude/Gemini via OpenRouter (Together doesn't host CU models)
- OpenRouter remains available as last-resort failover

### 4. Three-Provider Architecture

| Provider | Role | Models |
|----------|------|--------|
| **Anthropic Direct** | Orchestrator + Computer Use | Opus 4.6, Sonnet 4.5 |
| **Together AI** | Primary for all other tasks | 15+ models across all tiers |
| **OpenRouter** | Fallback if Together is down | Same models, 5.5% markup |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR: Claude Opus 4.6 (Anthropic Direct)                    │
│  Only model via Anthropic Direct. Best multi-tool orchestration.     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────────┐
│                    TOGETHER AI (PRIMARY)                              │
│                    24 models with function calling                    │
│                                                                      │
│  FLASH ($0.02-0.10):    gemma-3n, gpt-oss-20b, mistral-small-3     │
│  VALUE ($0.15-0.30):    minimax-m2.5, qwen3-235b, llama-4-maverick │
│  STANDARD ($0.45-0.60): kimi-k2.5, qwen3-coder-next, glm-4.7      │
│  REASONING ($0.65-1.20): glm-5, qwen3-235b-thinking                │
│  EXPERT ($2.00-7.00):   deepseek-r1-0528, qwen3-coder-480b         │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ FALLBACK ONLY (if Together down)
┌──────────────────────────┴──────────────────────────────────────────┐
│  OPENROUTER: Same models, auto-failover, 5.5% markup                │
└─────────────────────────────────────────────────────────────────────┘
                           │ COMPUTER USE ONLY
┌──────────────────────────┴──────────────────────────────────────────┐
│  ANTHROPIC/OPENROUTER: Claude Sonnet 4.5, Gemini 2.5 Pro            │
└─────────────────────────────────────────────────────────────────────┘
```

## TASK_MODEL_MAPPING (Revised)

```python
# TRIVIAL — cheapest Together models
ELEMENT_CLASSIFICATION: ["gemma-3n", "gpt-oss-20b", "mistral-small-3"]
ACTION_EXTRACTION:      ["gpt-oss-20b", "gemma-3n", "mistral-small-3"]
SELECTOR_VALIDATION:    ["gpt-oss-20b", "mistral-small-3", "gemma-3n"]
TEXT_EXTRACTION:        ["gemma-3n", "gpt-oss-20b", "mistral-small-3"]
JSON_PARSING:           ["gemma-3n", "gpt-oss-20b", "mistral-small-3"]

# MODERATE — MiniMax M2.5 (SWE-bench 80.2%) + Qwen3-Coder
CODE_ANALYSIS:          ["minimax-m2.5", "qwen3-coder-next", "qwen3-235b", "glm-4.5-air"]
TEST_GENERATION:        ["minimax-m2.5", "qwen3-coder-next", "llama-4-maverick", "qwen3-235b"]
ASSERTION_GENERATION:   ["qwen3-coder-next", "minimax-m2.5", "qwen3-235b"]
ERROR_CLASSIFICATION:   ["qwen3-next-80b", "mistral-small-3", "glm-4.5-air"]

# COMPLEX — GLM-5 (#2 global), Kimi K2.5 (#6 global)
VISUAL_COMPARISON:      ["llama-4-maverick", "kimi-k2.5", "sonnet"]
SEMANTIC_UNDERSTANDING: ["kimi-k2.5", "glm-5", "qwen3-235b"]
FLOW_DISCOVERY:         ["kimi-k2.5", "glm-5", "minimax-m2.5"]
ROOT_CAUSE_ANALYSIS:    ["glm-5", "kimi-k2.5", "deepseek-r1-0528"]

# EXPERT — GLM-5 for agentic, DeepSeek R1 for deep reasoning
SELF_HEALING:           ["glm-5", "kimi-k2.5", "deepseek-r1-0528", "opus"]
FAILURE_PREDICTION:     ["glm-5", "qwen3-235b-thinking", "kimi-k2-thinking"]
COGNITIVE_MODELING:      ["glm-5", "deepseek-r1-0528", "kimi-k2-thinking"]
COMPLEX_DEBUGGING:      ["glm-5", "minimax-m2.5", "deepseek-r1-0528"]

# COMPUTER USE — Claude only (Together doesn't host CU models)
COMPUTER_USE_SIMPLE:    ["claude-computer-use", "gemini-computer-use"]
COMPUTER_USE_COMPLEX:   ["claude-computer-use", "sonnet", "gemini-computer-use"]
COMPUTER_USE_MOBILE:    ["claude-computer-use", "gemini-computer-use"]

# GENERAL
GENERAL:                ["qwen3-235b", "minimax-m2.5", "llama-4-maverick", "glm-4.5-air"]
```

## Files Changed

| File | Change |
|------|--------|
| `src/core/providers/together_provider.py` | **NEW**: Together AI provider (OpenAI-compatible) |
| `src/core/providers/__init__.py` | Add Together exports |
| `src/core/model_router.py` | Replace MODELS dict entries with Together models, update TASK_MODEL_MAPPING, add `_get_client()` Together branch, add Opus 4.6 |
| `src/core/model_registry.py` | Add `ORCHESTRATOR_MODEL` + `get_orchestrator_model()` |
| `src/orchestrator/supervisor.py` | Use orchestrator model, increase max_tokens to 4096 |
| `.env.example` | Add `TOGETHER_API_KEY` |

## Key Model Justifications

| Model | Why | Benchmark |
|-------|-----|-----------|
| **GLM-5** | #2 global agentic rank, complex agent workflows | Quality Index 49.64 |
| **MiniMax M2.5** | SWE-bench 80.2%, code generation at 1/10 Claude cost | #10 global agentic |
| **Kimi K2.5** | Agent Swarm, 256K context, native multimodal | #6 global, Quality 46.73 |
| **Qwen3-Coder-Next** | SWE-bench 74.2%, code agent specialist | 80B params (3B active) |
| **Gemma 3n** | $0.02/1M — cheapest inference available | N/A (trivial tasks) |

## Cost Impact (Revised)

| Category | Before (all OpenRouter) | After (Together Primary) | Change |
|----------|------------------------|--------------------------|--------|
| Trivial (60%) | $0.10-0.12/1M | $0.02-0.06/1M | -70% |
| Moderate (25%) | $0.14-0.30/1M (DeepSeek) | $0.20-0.30/1M (MiniMax, Qwen) | ~same, better quality |
| Complex (10%) | $0.55-2.00/1M (DeepSeek R1) | $0.50-1.00/1M (GLM-5, Kimi) | -30%, higher agentic rank |
| Expert (5%) | $3-15/1M (Sonnet/Opus) | $2-7/1M (DeepSeek R1, Qwen3-Coder 480B) | -50% |
| Orchestrator | $15/1M (Sonnet via OR) | $25/1M (Opus Direct) | +67% (tiny volume, huge quality gain) |
| **Monthly est.** | **~$4,225** | **~$1,800** | **~$2,400/mo savings (57%)** |
