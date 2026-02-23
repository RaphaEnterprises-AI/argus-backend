# Agent Swarms — Design Document

**Date**: 2026-02-23
**Status**: Approved
**Author**: Claude + BVK

## Problem

Skopaq has 30+ specialized AI agents and full orchestration infrastructure (A2A protocol, agent registry, parallel executor, MARP consensus, workflow composer) — but they only run one-at-a-time through a sequential supervisor. There's no way to spawn multiple agents simultaneously on a task, which limits both throughput and the "wow factor" of seeing an AI swarm analyze your app in real-time.

Additionally, the dashboard has a full AG-UI-style event consumer (`streaming-protocol.ts`, `useAgentActivity`, 30 agent type configs) that the backend never wires into — the backend emits Vercel AI SDK protocol for chat and custom SSE for tests, neither matching the dashboard's consumer.

## Goals

1. **Dynamic scaling** — spawn 3-8 agents concurrently based on task complexity
2. **Emergent collaboration** — agents share findings mid-execution via A2A protocol
3. **User-facing feature** — real-time swarm visualization in the dashboard
4. **Marketing differentiator** — "Agent Swarms" as a headline capability

## Architecture

### Swarm Orchestrator (Scatter-Gather Pattern)

```
┌─────────────────────────────────────────────────────────┐
│                    SwarmOrchestrator                       │
│  - Spawns N workers (capped at 8 concurrent LLM calls)   │
│  - AG-UI event emitter for real-time dashboard updates    │
│  - 3-layer throttling (semaphore + quota + token bucket)  │
│                                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Worker 1 │ │ Worker 2 │ │ Worker 3 │ │ Worker N │   │
│  │CodeAnalyz│ │ UITester │ │APITester │ │SelfHealer│   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
│       │             │            │             │         │
│       └─────────────┴────────────┴─────────────┘         │
│                         │                                 │
│                  Consensus Layer                          │
│            (ConfidenceWeighted voting)                    │
└─────────────────────────────────────────────────────────┘
                          │
                    AG-UI SSE Events
                          │
                    ┌──────────────┐
                    │   Dashboard  │
                    │  Swarm View  │
                    └──────────────┘
```

### Three Swarm Modes

| Mode | Trigger | Workers | Duration |
|------|---------|---------|----------|
| **Full Crawl** | "Test my entire app" | 5-8 agents (CodeAnalyzer, AutoDiscovery, UITester, APITester, SecurityScanner, AccessibilityChecker, PerformanceAnalyzer, VisualAI) | Minutes |
| **Targeted Blitz** | "Test the checkout flow" | 3-5 agents (CodeAnalyzer, UITester, APITester, SelfHealer) | Seconds |
| **PR Analysis** | GitHub webhook / "Analyze PR #123" | 4-6 agents (CodeAnalyzer, TestImpactAnalyzer, SmartTestSelector, SecurityScanner, MRAnalyzer, FlakyDetector) | 30-60s |

## AG-UI Event Protocol

New event emitter maps internal events to AG-UI protocol:

| Internal Event | AG-UI Event | Data |
|---------------|-------------|------|
| Worker spawned | `RUN_STARTED` | `{runId, agentType, swarmId}` |
| Agent begins | `STEP_STARTED` | `{stepId, agentType, task}` |
| LLM thinking | `TEXT_MESSAGE_CONTENT` | `{delta}` |
| Tool invoked | `TOOL_CALL_START` + `TOOL_CALL_ARGS` | `{toolName, args}` |
| Tool result | `TOOL_CALL_END` | `{result}` |
| Agent done | `STEP_FINISHED` | `{result, duration, cost}` |
| Swarm complete | `RUN_FINISHED` | `{summary, totalCost, consensus}` |
| Progress | `STATE_DELTA` | `{agentId, progress, phase}` |
| Error | `RUN_ERROR` | `{error, agentId}` |

## Resource Management (3-Layer Throttling)

```python
class SwarmThrottler:
    # Layer 1: Semaphore — cap concurrent LLM calls
    semaphore = asyncio.Semaphore(8)

    # Layer 2: Workflow quota — reserve before starting
    max_workers_per_swarm = 8
    max_concurrent_swarms = 3  # per org

    # Layer 3: Token bucket — global cost control
    max_cost_per_swarm = 2.00       # USD
    max_cost_per_org_per_hour = 10.00  # USD
```

Production patterns applied:
- **CrewAI pattern**: Scatter-gather with 3-8 concurrent calls (not 100)
- **Netflix pattern**: Pre-compute heavy signals, contextualize in real-time
- **OpenAI Swarm pattern**: Lightweight handoffs between agents, no heavy framework

## Consensus Integration

After all workers complete, SwarmOrchestrator runs consensus:
- Uses existing `src/orchestrator/consensus.py` (ConfidenceWeighted strategy)
- Each worker's findings weighted by confidence score
- Produces unified swarm report with agreement scores

## Files

### New Files (Backend)

| File | Purpose |
|------|---------|
| `src/orchestrator/swarm_orchestrator.py` | Core swarm coordinator — scatter-gather, worker lifecycle, consensus |
| `src/orchestrator/swarm_throttler.py` | 3-layer resource management (semaphore + quota + token bucket) |
| `src/streaming/agui_emitter.py` | AG-UI protocol event emitter over SSE |
| `src/streaming/agui_events.py` | AG-UI event type definitions (dataclasses) |
| `src/api/swarms.py` | REST endpoints: `POST /swarms/launch`, `GET /swarms/{id}/stream`, `GET /swarms/{id}/status`, `DELETE /swarms/{id}` |

### Modified Files (Backend)

| File | Change |
|------|--------|
| `src/api/server.py` | Register swarms router |
| `src/agents/base.py` | Add `emit_agui_event()` helper to BaseAgent |
| `src/orchestrator/parallel_executor.py` | Wire into swarm throttler |
| `src/events/schemas.py` | Add `SWARM_STARTED`, `SWARM_COMPLETED`, `SWARM_WORKER_*` event types |
| `src/events/topics.py` | Add `argus.swarm.started`, `argus.swarm.completed` Kafka topics |

### New Files (Dashboard)

| File | Purpose |
|------|---------|
| `dashboard/components/swarm/SwarmView.tsx` | Real-time swarm visualization (agent cards, progress, connections) |
| `dashboard/components/swarm/SwarmWorkerCard.tsx` | Individual agent card with status, progress bar, cost |
| `dashboard/components/swarm/SwarmSummary.tsx` | Post-swarm results with consensus scores |
| `dashboard/components/swarm/SwarmLauncher.tsx` | Launch form — select mode, target, options |
| `dashboard/lib/hooks/use-swarm-stream.ts` | SSE consumer hook for swarm AG-UI events |
| `dashboard/app/swarms/page.tsx` | Swarm launcher + history page |
| `dashboard/app/swarms/[id]/page.tsx` | Individual swarm run view |

### Modified Files (Dashboard)

| File | Change |
|------|--------|
| `dashboard/lib/chat/streaming-protocol.ts` | Add swarm-specific event types (`swarm_started`, `worker_spawned`, etc.) |
| `dashboard/components/layout/sidebar.tsx` | Add Swarms nav item |
| `dashboard/middleware.ts` | Add `/swarms` route |

## Implementation Order

1. AG-UI event layer (`agui_events.py`, `agui_emitter.py`)
2. Swarm throttler (`swarm_throttler.py`)
3. Swarm orchestrator (`swarm_orchestrator.py`)
4. API endpoints (`swarms.py`) + server registration
5. Backend integration (BaseAgent, events, topics)
6. Dashboard SSE consumer (`use-swarm-stream.ts`)
7. Dashboard UI (SwarmView, WorkerCard, Summary, Launcher)
8. Dashboard pages + routing
9. Landing page swarm section

## Non-Goals (Deferred)

- Stripe billing per swarm execution (deferred to monetization phase)
- Custom agent training/fine-tuning within swarms
- Cross-org swarm sharing
- Swarm templates marketplace
