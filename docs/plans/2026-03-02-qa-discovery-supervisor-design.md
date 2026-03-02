# QA Discovery — Progressive Streaming via Supervisor

**Date**: 2026-03-02
**Status**: Approved

## Problem

The current `qa_discovery` swarm mode chains AutoDiscovery → TestGenAgent sequentially. This has three problems:

1. **Latency wall** — Crawling 50 pages takes 2-5 min. No test output until discovery finishes.
2. **Fragile** — Auth walls, CAPTCHAs, rate limits, SPA hydration failures kill the whole pipeline.
3. **No shared understanding** — TestGen receives a text blob, not the rich shared state that other supervisor agents can read and build upon.

## Solution: Progressive Streaming via Supervisor

Wire `auto_discovery` as a new LangGraph supervisor node that yields pages in **batches**. The supervisor interleaves discovery and test generation so users see output within seconds.

### Execution Flow

```
Supervisor loop:
  Iter 1: → auto_discovery (crawls batch of 3-5 pages)
          state.discovery_queue = [pg1, pg2, pg3]
          state.discovery_complete = False
  Iter 2: → testgen_agent (processes queue, generates BDD + code)
          state.generated_tests += 5 scenarios
          state.discovery_queue = []
  Iter 3: → auto_discovery (crawls next batch)
          state.discovery_queue = [pg4, pg5, pg6]
  Iter 4: → testgen_agent (processes new batch)
          state.generated_tests += 4 scenarios
  Iter 5: → auto_discovery (finished or timed out)
          state.discovery_complete = True
  Iter 6: → testgen_agent (final dedup + coherence pass)
  Iter 7: → reporter
  FINISH
```

User sees first test scenarios ~10 seconds in, not 3 minutes in.

## Architecture

### 1. New SupervisorState Fields

```python
# Discovery streaming state
discovery_queue: list[dict] | None     # Pages/flows waiting for testgen
discovery_complete: bool               # True when crawling is done
discovery_pages_found: int             # Total pages discovered so far
discovery_batch_index: int             # Which batch we're on (for resume)

# Test generation accumulator
generated_tests: list[dict] | None     # Accumulated BDD scenarios + test cases
generated_requirements: list[dict] | None  # Accumulated parsed requirements

# Multi-tenant context (needed by testgen, missing today)
org_id: str | None
project_id: str | None
```

### 2. New `auto_discovery` Supervisor Node

- Registered in `AGENTS` list + `AGENT_DESCRIPTIONS` + `create_supervisor_graph()`
- Description tells the LLM when to use it:
  > "Crawls a web application to discover pages, forms, flows, and API endpoints.
  > Use when app_url is set but discovery is not complete. Returns batches of
  > 3-5 pages per invocation. Call repeatedly until discovery_complete is True."
- Phase: `"discovery"`
- Behavior:
  - Reads `app_url`, `discovery_batch_index` from state
  - Creates `AutoDiscovery(app_url, max_pages=5)` with offset for batch
  - Writes discovered pages/flows to `discovery_queue`
  - Enriches `testable_surfaces` incrementally
  - Sets `discovery_complete = True` when no more pages or max iterations reached
  - Selenium Grid fallback on Playwright failure (same as swarm)

### 3. Enhanced `testgen_agent` Node

Add a third source type branch:

```python
# Existing branches:
if state.get("changed_files"):   → source_type = "github_pr"
elif state.get("testable_surfaces"): → source_type = "codebase_analysis"

# NEW branch (highest priority when discovery_queue exists):
elif state.get("discovery_queue"):
    source_type = SourceType.USER_STORY
    source_data = _discovery_to_testgen_input(discovery_queue)
    # After generation, clear queue and accumulate results
    # On final pass (discovery_complete + empty queue): dedup
```

Testgen also gets `app_url` forwarded into `TestGenConfig` for Playwright code generation.

### 4. Supervisor Prompt Updates

Add decision rules:
```
- If app_url is set and discovery_complete is False, use auto_discovery
- If discovery_queue has items, use testgen_agent to generate tests for them
- If discovery_complete is True and discovery_queue is empty and generated_tests
  exist, use reporter (or testgen for final dedup if not done)
```

### 5. Swarm Integration

`_run_qa_discovery_swarm()` creates a `SupervisorOrchestrator` and runs the graph:

```python
async def _run_qa_discovery_swarm(self, swarm_id, config, emitter):
    orchestrator = SupervisorOrchestrator(
        codebase_path=config.codebase_path or ".",
        app_url=config.target_url,
    )
    # Inject QA discovery context into initial state
    initial_state = create_initial_supervisor_state(
        codebase_path=config.codebase_path or ".",
        app_url=config.target_url,
        initial_message=f"Discover the application at {config.target_url} and generate comprehensive BDD test scenarios with executable Playwright code.",
    )
    initial_state["org_id"] = config.org_id
    initial_state["project_id"] = config.project_id
    initial_state["discovery_complete"] = False
    initial_state["discovery_queue"] = []
    initial_state["generated_tests"] = []

    # Run supervisor graph, emit SSE events from state transitions
    final_state = await orchestrator.run_with_state(initial_state)

    # Convert final state → WorkerResults for swarm SSE
    ...
```

The swarm emits SSE events by observing supervisor state changes (each iteration that modifies `discovery_queue` or `generated_tests` emits a `StateDeltaEvent`).

### 6. Batch Sizing and Timeouts

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Pages per discovery batch | 3-5 | Fast enough to return in <10s |
| Max discovery batches | 10 | Cap at ~50 pages total |
| Discovery timeout per batch | 30s | Skip slow/stuck pages |
| Overall discovery timeout | 120s | Hard cap, partial results OK |
| Max supervisor iterations | 25 | ~10 discovery + 10 testgen + 5 other |

### 7. Error Handling

- **Auth wall on a page**: Skip that page, continue crawling others. Log it as a finding.
- **Playwright unavailable**: Fall back to Selenium Grid (existing pattern).
- **Crawl timeout**: Set `discovery_complete = True` with what we have. Testgen works on partial data.
- **Testgen fails on a batch**: Log warning, continue. Other batches still produce tests.
- **All discovery fails**: Supervisor routes to reporter with error summary.

## Files to Modify

| File | Change |
|------|--------|
| `src/orchestrator/supervisor.py` | Add `auto_discovery` to AGENTS, AGENT_DESCRIPTIONS, PHASE_DESCRIPTIONS. Add `supervisor_auto_discovery_node()`. Register in `create_supervisor_graph()`. Add new state fields. Update supervisor prompt with discovery rules. Enhance `supervisor_testgen_agent_node()` for discovery_queue. |
| `src/orchestrator/swarm_orchestrator.py` | Update `_run_qa_discovery_swarm()` to use SupervisorOrchestrator instead of direct agent calls. |
| `src/agents/auto_discovery.py` | Add `discover_batch()` method that takes offset/limit and returns a subset of pages (or reuse existing `discover()` with `max_pages` set low). |
| `tests/orchestrator/test_swarm.py` | Update QA discovery tests to verify supervisor integration. |
| `tests/orchestrator/test_supervisor_qa.py` | New test file for supervisor QA discovery flow. |

## What We're NOT Doing

- No changes to the dashboard (QA Discovery card/pill already exists from the swarm work)
- No changes to `src/api/swarms.py` (mode string flows through unchanged)
- No new Kafka topics or A2A protocol changes
- No changes to AutoDiscovery's core crawling logic (just batching via max_pages)
