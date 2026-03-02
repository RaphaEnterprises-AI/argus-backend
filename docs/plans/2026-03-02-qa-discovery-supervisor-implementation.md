# QA Discovery — Supervisor-Wired Progressive Streaming Implementation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire `auto_discovery` as a LangGraph supervisor node that yields page batches, interleaving with `testgen_agent` so users see test output within seconds instead of waiting for full crawl.

**Architecture:** The supervisor LLM alternates between `auto_discovery` (crawl 3-5 pages per batch) and `testgen_agent` (generate BDD+code for the batch). New `SupervisorState` fields (`discovery_queue`, `discovery_complete`, `generated_tests`) coordinate the handoff. The swarm's `_run_qa_discovery_swarm()` delegates to the supervisor graph instead of calling agents directly.

**Tech Stack:** LangGraph StateGraph, AutoDiscovery agent, TestGenAgent, Playwright, SSE streaming

**Design doc:** `docs/plans/2026-03-02-qa-discovery-supervisor-design.md`

---

### Task 1: Add new SupervisorState fields

**Files:**
- Modify: `src/orchestrator/supervisor.py:57-109` (SupervisorState TypedDict)

**Step 1: Add discovery streaming + multi-tenant fields to SupervisorState**

In `SupervisorState` (line 57), add these fields after the existing `error: str | None` field (line 109):

```python
    # Discovery streaming (QA Discovery mode)
    discovery_queue: list[dict] | None       # Pages/flows awaiting testgen processing
    discovery_complete: bool                 # True when crawling finished or timed out
    discovery_pages_found: int               # Total pages discovered across all batches
    discovery_batch_index: int               # Current batch number (for resume)

    # Test generation accumulator (QA Discovery mode)
    generated_tests: list[dict] | None       # Accumulated BDD scenarios + test cases
    generated_requirements: list[dict] | None  # Accumulated parsed requirements

    # Multi-tenant context
    org_id: str | None
    project_id: str | None
```

**Step 2: Update `create_initial_supervisor_state()` with new field defaults**

In `create_initial_supervisor_state()` (line 214), add defaults to the returned dict:

```python
        # Discovery streaming defaults
        discovery_queue=[],
        discovery_complete=False,
        discovery_pages_found=0,
        discovery_batch_index=0,
        generated_tests=[],
        generated_requirements=[],
        org_id=None,
        project_id=None,
```

**Step 3: Run existing supervisor tests to verify no regressions**

Run: `python -m pytest tests/orchestrator/ -v -k "supervisor" --timeout=60 2>&1 | tail -20`
Expected: All existing tests pass (new fields have defaults, so backwards-compatible).

**Step 4: Commit**

```bash
git add src/orchestrator/supervisor.py
git commit -m "feat(supervisor): add discovery streaming + multi-tenant state fields"
```

---

### Task 2: Add `auto_discovery` to supervisor agent registry

**Files:**
- Modify: `src/orchestrator/supervisor.py:112-164` (AGENTS, AGENT_DESCRIPTIONS, PHASE_DESCRIPTIONS)

**Step 1: Add to AGENTS list**

After `"qa_engineer"` (line 129), add:

```python
    # Discovery
    "auto_discovery",
```

**Step 2: Add to AGENT_DESCRIPTIONS**

After the `qa_engineer` entry (line 147), add:

```python
    "auto_discovery": "Crawls a web application to discover pages, forms, user flows, and API endpoints. Use when app_url is set and discovery_complete is False. Returns batches of 3-5 pages per invocation. Call repeatedly until discovery_complete is True.",
```

**Step 3: Add to PHASE_DESCRIPTIONS**

After `"quality_audit"` (line 163), add:

```python
    "discovery": "Crawling the target application to discover pages, forms, flows, and testable surfaces",
```

**Step 4: Add discovery routing rules to supervisor prompt**

In `create_supervisor_prompt()` (line 167), add to the Decision Rules section (after line 205):

```
- If app_url is set and discovery_complete is False and discovery_queue is empty, use auto_discovery
- If discovery_queue has items waiting to be processed, use testgen_agent
- If discovery_complete is True and discovery_queue is empty and generated_tests exist, use reporter
```

Also add to the Workflow Guidelines (after line 194):

```
- For QA discovery: alternate between auto_discovery (crawl batch) and testgen_agent (generate tests for batch)
```

**Step 5: Add phase routing in supervisor_node**

In `supervisor_node()` (around line 418-430), add:

```python
        elif next_agent == "auto_discovery":
            new_phase = "discovery"
```

**Step 6: Commit**

```bash
git add src/orchestrator/supervisor.py
git commit -m "feat(supervisor): register auto_discovery agent with decision rules"
```

---

### Task 3: Implement `supervisor_auto_discovery_node()`

**Files:**
- Modify: `src/orchestrator/supervisor.py` (add new function before `create_supervisor_graph()`)

**Step 1: Write the node function**

Add before `create_supervisor_graph()` (~line 1207). Follows the same pattern as other supervisor nodes:

```python
async def supervisor_auto_discovery_node(state: SupervisorState) -> dict:
    """Wrapper for AutoDiscovery that yields page batches into supervisor state.

    Each invocation crawls up to BATCH_SIZE pages and writes them to
    discovery_queue for testgen to process. Sets discovery_complete=True
    when no more pages to crawl or max batches reached.
    """
    from src.agents import AutoDiscovery

    log = logger.bind(node="supervisor_auto_discovery")

    BATCH_SIZE = 5
    MAX_BATCHES = 10
    BATCH_TIMEOUT = 30  # seconds

    app_url = state.get("app_url")
    if not app_url:
        return {
            "messages": [AIMessage(content="Auto-discovery: No app_url provided.")],
            "discovery_complete": True,
            "current_phase": "reporting",
        }

    batch_index = state.get("discovery_batch_index", 0)
    pages_found_so_far = state.get("discovery_pages_found", 0)

    if batch_index >= MAX_BATCHES:
        log.info("Max discovery batches reached", batches=batch_index)
        return {
            "messages": [AIMessage(content=f"Discovery complete: reached max {MAX_BATCHES} batches ({pages_found_so_far} pages total).")],
            "discovery_complete": True,
            "current_phase": "test_generation" if pages_found_so_far > 0 else "reporting",
        }

    log.info("Running discovery batch", batch=batch_index, pages_so_far=pages_found_so_far)

    try:
        import asyncio

        discovery = AutoDiscovery(app_url=app_url, max_pages=BATCH_SIZE)

        # Build start_paths for this batch (skip already-discovered pages)
        existing_surfaces = state.get("testable_surfaces") or []
        already_discovered_urls = {
            s.get("url", "") for s in existing_surfaces if s.get("url")
        }

        try:
            result = await asyncio.wait_for(
                discovery.discover(),
                timeout=BATCH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("Discovery batch timed out", batch=batch_index)
            return {
                "messages": [AIMessage(content=f"Discovery batch {batch_index + 1} timed out after {BATCH_TIMEOUT}s. Continuing with what we have.")],
                "discovery_complete": True,
                "current_phase": "test_generation" if pages_found_so_far > 0 else "reporting",
            }
        except Exception as playwright_err:
            # Selenium Grid fallback
            log.warning("Playwright unavailable, trying Selenium Grid", error=str(playwright_err))
            try:
                from src.browser.selenium_grid_client import SeleniumGridClient
                async with SeleniumGridClient() as grid:
                    grid_result = await asyncio.wait_for(
                        grid.discover(start_url=app_url, max_pages=BATCH_SIZE, max_depth=2),
                        timeout=BATCH_TIMEOUT,
                    )
                    new_pages = [
                        {"url": p.url, "title": getattr(p, "title", ""), "type": "page"}
                        for p in grid_result.pages
                        if p.url not in already_discovered_urls
                    ]
                    # Treat Selenium Grid result as final (no multi-batch)
                    return {
                        "messages": [AIMessage(content=f"Discovery (Selenium Grid): found {len(new_pages)} pages.")],
                        "discovery_queue": new_pages,
                        "discovery_complete": True,
                        "discovery_pages_found": pages_found_so_far + len(new_pages),
                        "discovery_batch_index": batch_index + 1,
                        "testable_surfaces": [
                            *(state.get("testable_surfaces") or []),
                            *new_pages,
                        ],
                        "current_phase": "test_generation" if new_pages else "reporting",
                    }
            except Exception as grid_err:
                log.error("Both Playwright and Selenium Grid failed", error=str(grid_err))
                return {
                    "messages": [AIMessage(content=f"Discovery failed: Playwright ({playwright_err}), Selenium Grid ({grid_err})")],
                    "discovery_complete": True,
                    "error": f"Discovery failed: {grid_err}",
                    "current_phase": "reporting",
                }

        # Convert DiscoveryResult pages to queue items (filter already-seen)
        new_pages = []
        for page in result.pages_discovered:
            url = getattr(page, "url", "")
            if url and url not in already_discovered_urls:
                new_pages.append({
                    "url": url,
                    "title": getattr(page, "title", ""),
                    "description": getattr(page, "description", ""),
                    "forms": [
                        f if isinstance(f, dict) else str(f)
                        for f in getattr(page, "forms", [])[:5]
                    ],
                    "type": "page",
                })

        new_flows = []
        for flow in result.flows_discovered:
            new_flows.append({
                "name": getattr(flow, "name", ""),
                "description": getattr(flow, "description", ""),
                "category": getattr(flow, "category", ""),
                "priority": getattr(flow, "priority", "medium"),
                "steps": getattr(flow, "steps", [])[:10],
                "type": "flow",
            })

        queue_items = new_pages + new_flows
        new_surfaces = [{"url": p["url"], "title": p["title"], "type": "page"} for p in new_pages]

        # Determine if discovery is done
        # Done if: this batch found fewer pages than BATCH_SIZE (exhausted)
        is_done = len(result.pages_discovered) < BATCH_SIZE

        total_pages = pages_found_so_far + len(new_pages)
        summary_parts = [f"Batch {batch_index + 1}: {len(new_pages)} pages"]
        if new_flows:
            summary_parts.append(f"{len(new_flows)} flows")
        summary_parts.append(f"({total_pages} total)")
        if is_done:
            summary_parts.append("- crawling complete")

        summary = "Discovery " + ", ".join(summary_parts)

        return {
            "messages": [AIMessage(content=summary)],
            "discovery_queue": queue_items,
            "discovery_complete": is_done,
            "discovery_pages_found": total_pages,
            "discovery_batch_index": batch_index + 1,
            "testable_surfaces": [
                *(state.get("testable_surfaces") or []),
                *new_surfaces,
            ],
            "current_phase": "test_generation",
            "results": {
                **state.get("results", {}),
                f"discovery_batch_{batch_index}": {
                    "pages": len(new_pages),
                    "flows": len(new_flows),
                    "total_pages": total_pages,
                },
            },
        }

    except Exception as e:
        log.error("Auto-discovery node failed", error=str(e))
        return {
            "messages": [AIMessage(content=f"Discovery error: {str(e)}")],
            "discovery_complete": True,
            "current_phase": "reporting" if pages_found_so_far == 0 else "test_generation",
        }
```

**Step 2: Commit**

```bash
git add src/orchestrator/supervisor.py
git commit -m "feat(supervisor): implement auto_discovery node with batch crawling"
```

---

### Task 4: Register `auto_discovery` in `create_supervisor_graph()`

**Files:**
- Modify: `src/orchestrator/supervisor.py` (~line 1247-1305 in `create_supervisor_graph()`)

**Step 1: Add the node**

After the QA Intelligence agent nodes (~line 1271), add:

```python
    # Discovery
    graph.add_node("auto_discovery", supervisor_auto_discovery_node)
```

**Step 2: Add to conditional edges**

In the `graph.add_conditional_edges(...)` dict (~line 1277-1298), add:

```python
            # Discovery
            "auto_discovery": "auto_discovery",
```

**Step 3: Commit**

```bash
git add src/orchestrator/supervisor.py
git commit -m "feat(supervisor): register auto_discovery in graph routing"
```

---

### Task 5: Enhance `supervisor_testgen_agent_node()` for discovery queue

**Files:**
- Modify: `src/orchestrator/supervisor.py:1042-1098` (`supervisor_testgen_agent_node()`)

**Step 1: Add discovery_queue branch**

Replace the existing `supervisor_testgen_agent_node()` function. The key change: add a `discovery_queue` branch as the highest-priority source type, using `_discovery_to_testgen_input()` from the swarm orchestrator:

```python
async def supervisor_testgen_agent_node(state: SupervisorState) -> dict:
    """Wrapper for TestGen Agent that works with supervisor state.

    Handles three source types in priority order:
    1. discovery_queue (from auto_discovery batches)
    2. changed_files (from PR analysis)
    3. testable_surfaces (from code analysis)
    """
    from src.agents.testgen_agent import TestGenAgent, TestGenConfig, SourceType

    log = logger.bind(node="supervisor_testgen_agent")
    log.info("Running TestGen agent")

    try:
        agent = TestGenAgent()
        source_data = {}
        source_type_str = "codebase_analysis"
        gen_config = None

        # Priority 1: Discovery queue (from progressive QA discovery)
        discovery_queue = state.get("discovery_queue") or []
        if discovery_queue:
            source_type_str = "user_story"
            # Format queue items as user stories
            source_data = _format_discovery_queue(discovery_queue, state.get("app_url", ""))
            gen_config = TestGenConfig(
                app_url=state.get("app_url"),
                bdd_style=True,
            )

        # Priority 2: Changed files (PR analysis)
        elif state.get("changed_files"):
            source_type_str = "github_pr"
            source_data = {
                "title": f"PR #{state.get('pr_number', 'unknown')}",
                "changed_files": state.get("changed_files", []),
                "body": "",
            }

        # Priority 3: Testable surfaces (code analysis)
        elif state.get("testable_surfaces"):
            source_type_str = "codebase_analysis"
            source_data = {
                "functions": [s for s in state.get("testable_surfaces", []) if s.get("type") == "function"],
                "endpoints": [s for s in state.get("testable_surfaces", []) if s.get("type") == "endpoint"],
                "components": [s for s in state.get("testable_surfaces", []) if s.get("type") == "component"],
            }

        result = await agent.generate(
            org_id=state.get("org_id", "default"),
            project_id=state.get("project_id", "default"),
            source_type=source_type_str,
            source_data=source_data,
            config=gen_config or {},
        )

        # Extract results
        if hasattr(result, "tests"):
            new_tests = [t.to_dict() if hasattr(t, "to_dict") else {"name": t.name} for t in result.tests]
            new_reqs = [r.to_dict() if hasattr(r, "to_dict") else {"title": r.title} for r in getattr(result, "requirements", [])]
            tests_count = len(new_tests)
            quality = getattr(result, "quality_score", 0)
            cost = getattr(result, "ai_cost", 0)
        elif isinstance(result, dict):
            tests_count = result.get("tests_generated", 0)
            new_tests = result.get("tests", [])
            new_reqs = result.get("requirements", [])
            quality = result.get("quality_score", 0)
            cost = result.get("cost", 0)
        else:
            tests_count = 0
            new_tests = []
            new_reqs = []
            quality = 0
            cost = 0

        # Accumulate tests across batches
        accumulated_tests = list(state.get("generated_tests") or []) + new_tests
        accumulated_reqs = list(state.get("generated_requirements") or []) + new_reqs

        # Determine next phase
        discovery_complete = state.get("discovery_complete", True)
        if not discovery_complete:
            next_phase = "discovery"  # Go back for more pages
        else:
            next_phase = "reporting"  # All done

        summary = f"Generated {tests_count} tests from {source_type_str} (total: {len(accumulated_tests)})"
        if quality:
            summary += f", quality: {quality:.0%}"

        return {
            "messages": [AIMessage(content=f"TestGen: {summary}")],
            "discovery_queue": [],  # Clear processed queue
            "generated_tests": accumulated_tests,
            "generated_requirements": accumulated_reqs,
            "current_phase": next_phase,
            "total_cost": state.get("total_cost", 0) + cost,
            "results": {
                **state.get("results", {}),
                "test_generation": {
                    "tests_generated": len(accumulated_tests),
                    "source_type": source_type_str,
                    "quality_score": quality,
                    "batch_count": len([k for k in state.get("results", {}) if k.startswith("discovery_batch")]),
                }
            },
        }
    except Exception as e:
        log.error("TestGen agent failed", error=str(e))
        return {
            "messages": [AIMessage(content=f"TestGen error: {str(e)}")],
            "current_phase": state.get("current_phase", "analysis"),
        }
```

**Step 2: Add the `_format_discovery_queue()` helper**

Add right above `supervisor_testgen_agent_node()`:

```python
def _format_discovery_queue(queue: list[dict], app_url: str) -> str:
    """Format discovery queue items as user story text for TestGenAgent."""
    parts = [f"# Discovered Pages and Flows: {app_url}\n"]

    pages = [item for item in queue if item.get("type") == "page"]
    flows = [item for item in queue if item.get("type") == "flow"]

    if pages:
        parts.append("## Pages\n")
        for i, page in enumerate(pages, 1):
            title = page.get("title") or page.get("url", f"Page {i}")
            parts.append(f"### {title}")
            parts.append(f"URL: {page.get('url', '')}")
            if page.get("description"):
                parts.append(f"Description: {page['description']}")
            forms = page.get("forms", [])
            if forms:
                parts.append("Forms: " + ", ".join(
                    f.get("action", str(f)) if isinstance(f, dict) else str(f)
                    for f in forms[:5]
                ))
            parts.append("")

    if flows:
        parts.append("## User Flows\n")
        for flow in flows:
            parts.append(f"### {flow.get('name', 'Flow')}")
            if flow.get("category"):
                parts.append(f"Category: {flow['category']}")
            if flow.get("description"):
                parts.append(f"Description: {flow['description']}")
            steps = flow.get("steps", [])
            if steps:
                parts.append("Steps:")
                for j, step in enumerate(steps[:10], 1):
                    if isinstance(step, dict):
                        parts.append(f"  {j}. {step.get('action', 'navigate')}: {step.get('target', step.get('url', ''))}")
                    else:
                        parts.append(f"  {j}. {step}")
            parts.append("")

    return "\n".join(parts)
```

**Step 3: Commit**

```bash
git add src/orchestrator/supervisor.py
git commit -m "feat(supervisor): enhance testgen node for discovery queue + batch accumulation"
```

---

### Task 6: Update `_run_qa_discovery_swarm()` to use supervisor

**Files:**
- Modify: `src/orchestrator/swarm_orchestrator.py` (replace `_run_qa_discovery_swarm()`)

**Step 1: Replace the method**

Replace the existing `_run_qa_discovery_swarm()` and `_discovery_to_testgen_input()` with a new version that delegates to the supervisor graph:

```python
    async def _run_qa_discovery_swarm(
        self,
        swarm_id: str,
        config: SwarmConfig,
        emitter: AGUIEmitter,
    ) -> SwarmResult:
        """QA discovery via supervisor: progressive crawl → interleaved test generation.

        Delegates to the LangGraph supervisor which alternates between
        auto_discovery (batch crawl) and testgen_agent (generate BDD + code).
        The supervisor LLM decides agent ordering based on shared state.
        """
        start_time = time.time()

        try:
            from src.orchestrator.supervisor import (
                SupervisorState,
                create_initial_supervisor_state,
                create_supervisor_graph,
            )

            # Build initial state with QA discovery context
            initial_state = create_initial_supervisor_state(
                codebase_path=config.codebase_path or ".",
                app_url=config.target_url,
                initial_message=(
                    f"Discover the application at {config.target_url} and generate "
                    f"comprehensive BDD test scenarios with executable Playwright code. "
                    f"Use auto_discovery to crawl pages in batches, then testgen_agent "
                    f"to generate tests for each batch. Alternate between discovery and "
                    f"test generation until all pages are covered."
                ),
            )
            initial_state["org_id"] = config.org_id
            initial_state["project_id"] = config.project_id
            initial_state["discovery_complete"] = False

            # Compile and run the supervisor graph
            graph = create_supervisor_graph()

            from src.orchestrator.checkpointer import get_checkpointer
            checkpointer = get_checkpointer()
            app = graph.compile(checkpointer=checkpointer)

            thread_id = f"qa_discovery_{swarm_id}"
            run_config = {"configurable": {"thread_id": thread_id}}

            # Stream state updates for SSE
            prev_phase = None
            prev_pages = 0
            prev_tests_count = 0
            final_state = None

            async for event in app.astream(initial_state, run_config, stream_mode="values"):
                final_state = event

                # Emit SSE progress based on state changes
                cur_phase = event.get("current_phase", "")
                cur_pages = event.get("discovery_pages_found", 0)
                cur_tests = len(event.get("generated_tests") or [])
                iteration = event.get("iteration", 0)

                if cur_phase != prev_phase:
                    agent_id = f"{cur_phase}_{iteration}"
                    if cur_phase == "discovery":
                        await emitter.emit(StateDeltaEvent(
                            agent_id="auto_discovery",
                            progress=min(90, cur_pages * 10),
                            phase="crawling",
                            message=f"Discovering pages... ({cur_pages} found)",
                        ))
                    elif cur_phase == "test_generation":
                        await emitter.emit(StateDeltaEvent(
                            agent_id="testgen",
                            progress=min(90, cur_tests * 5),
                            phase="generating",
                            message=f"Generating tests... ({cur_tests} so far)",
                        ))
                    elif cur_phase == "reporting":
                        await emitter.emit(StateDeltaEvent(
                            agent_id="reporter",
                            progress=90,
                            phase="reporting",
                            message="Compiling final report...",
                        ))

                if cur_pages > prev_pages:
                    await emitter.emit(StateDeltaEvent(
                        agent_id="auto_discovery",
                        progress=min(90, cur_pages * 10),
                        phase="crawling",
                        message=f"Found {cur_pages} pages",
                    ))

                if cur_tests > prev_tests_count:
                    await emitter.emit(StateDeltaEvent(
                        agent_id="testgen",
                        progress=min(90, cur_tests * 5),
                        phase="generating",
                        message=f"Generated {cur_tests} tests",
                    ))

                prev_phase = cur_phase
                prev_pages = cur_pages
                prev_tests_count = cur_tests

            if final_state is None:
                raise RuntimeError("Supervisor graph produced no output")

            # Convert supervisor state → WorkerResults
            total_duration = (time.time() - start_time) * 1000
            total_cost = final_state.get("total_cost", 0.0)
            generated_tests = final_state.get("generated_tests") or []
            pages_found = final_state.get("discovery_pages_found", 0)

            worker_results = []

            # Discovery worker result
            worker_results.append(WorkerResult(
                agent_id="auto_discovery",
                agent_type="auto_discovery",
                success=pages_found > 0,
                duration_ms=total_duration * 0.4,  # Approximate split
                cost_usd=0.0,
                findings=[
                    {"type": "page", "url": s.get("url", "")}
                    for s in (final_state.get("testable_surfaces") or [])
                ],
                summary=f"Discovered {pages_found} pages in {final_state.get('discovery_batch_index', 0)} batches",
                confidence=0.75 if pages_found > 0 else 0.0,
            ))

            # Testgen worker result
            worker_results.append(WorkerResult(
                agent_id="testgen",
                agent_type="testgen",
                success=len(generated_tests) > 0,
                duration_ms=total_duration * 0.6,
                cost_usd=total_cost,
                findings=[
                    {"type": "test_case", "name": t.get("name", "test")}
                    for t in generated_tests
                ],
                summary=f"Generated {len(generated_tests)} tests, {len(final_state.get('generated_requirements') or [])} requirements",
                confidence=0.85 if generated_tests else 0.0,
            ))

            consensus_score = self._compute_consensus(worker_results)
            workers_completed = sum(1 for r in worker_results if r.success)
            workers_failed = len(worker_results) - workers_completed

            summary = (
                f"QA discovery: {pages_found} pages, {len(generated_tests)} tests, "
                f"{workers_completed}/{len(worker_results)} phases completed, "
                f"{consensus_score:.0%} consensus"
            )

            verdict = await self._verify_intent(config, worker_results, total_cost)
            if verdict:
                summary += f" | Intent: {verdict.verdict} ({verdict.completion_score:.0%})"

            await emitter.emit(
                RunFinishedEvent(
                    run_id=swarm_id,
                    swarm_id=swarm_id,
                    success=workers_failed == 0,
                    total_duration_ms=total_duration,
                    total_cost_usd=total_cost,
                    workers_completed=workers_completed,
                    workers_failed=workers_failed,
                    consensus_score=consensus_score,
                    summary=summary,
                )
            )

            return SwarmResult(
                swarm_id=swarm_id,
                mode=config.mode,
                success=workers_failed == 0,
                total_duration_ms=total_duration,
                total_cost_usd=total_cost,
                worker_results=worker_results,
                consensus_score=consensus_score,
                summary=summary,
                intent_verdict=verdict,
            )

        except Exception as e:
            logger.exception("QA discovery supervisor run failed", swarm_id=swarm_id)
            total_duration = (time.time() - start_time) * 1000

            await emitter.emit(
                RunFinishedEvent(
                    run_id=swarm_id,
                    swarm_id=swarm_id,
                    success=False,
                    total_duration_ms=total_duration,
                    total_cost_usd=0.0,
                    workers_completed=0,
                    workers_failed=1,
                    consensus_score=0.0,
                    summary=f"QA discovery failed: {e}",
                )
            )

            return SwarmResult(
                swarm_id=swarm_id,
                mode=config.mode,
                success=False,
                total_duration_ms=total_duration,
                total_cost_usd=0.0,
                summary=f"QA discovery failed: {e}",
            )
```

**Step 2: Remove `_discovery_to_testgen_input()` static method**

This is now replaced by `_format_discovery_queue()` in supervisor.py. Remove the old method from swarm_orchestrator.py.

**Step 3: Commit**

```bash
git add src/orchestrator/swarm_orchestrator.py
git commit -m "feat(swarm): delegate qa_discovery to supervisor graph for progressive streaming"
```

---

### Task 7: Write tests for supervisor QA discovery

**Files:**
- Modify: `tests/orchestrator/test_swarm.py` (update existing QA discovery tests)

**Step 1: Update test_qa_discovery_chains_both_phases**

The test needs to mock the supervisor graph instead of direct agent calls. Replace the existing test:

```python
    @pytest.mark.asyncio
    async def test_qa_discovery_uses_supervisor(self, mock_emitter):
        """QA discovery should delegate to the supervisor graph."""
        from src.orchestrator.swarm_orchestrator import SwarmConfig, SwarmMode, SwarmOrchestrator

        orch = SwarmOrchestrator()
        config = SwarmConfig(
            mode=SwarmMode.QA_DISCOVERY,
            org_id="test-org",
            project_id="test-project",
            user_id="test-user",
            target_url="https://example.com",
        )

        # Mock the supervisor graph to return a final state
        mock_final_state = {
            "task_complete": True,
            "current_phase": "complete",
            "iteration": 5,
            "total_cost": 0.05,
            "discovery_pages_found": 3,
            "discovery_batch_index": 1,
            "discovery_complete": True,
            "testable_surfaces": [
                {"url": "https://example.com/", "title": "Home", "type": "page"},
                {"url": "https://example.com/login", "title": "Login", "type": "page"},
            ],
            "generated_tests": [
                {"name": "test_home_loads", "framework": "playwright"},
                {"name": "test_login_flow", "framework": "playwright"},
            ],
            "generated_requirements": [{"title": "Home page loads"}],
            "results": {},
        }

        # Mock astream to yield the final state
        async def mock_astream(initial_state, config, stream_mode="values"):
            yield mock_final_state

        mock_graph = MagicMock()
        mock_compiled = MagicMock()
        mock_compiled.astream = mock_astream
        mock_graph.compile = MagicMock(return_value=mock_compiled)

        with (
            patch("src.orchestrator.supervisor.create_supervisor_graph", return_value=mock_graph),
            patch("src.orchestrator.supervisor.create_initial_supervisor_state", return_value={
                "messages": [], "discovery_complete": False, "discovery_queue": [],
                "generated_tests": [], "org_id": "test-org", "project_id": "test-project",
                "discovery_pages_found": 0, "discovery_batch_index": 0,
                "testable_surfaces": [], "results": {}, "total_cost": 0,
            }),
            patch("src.orchestrator.checkpointer.get_checkpointer", return_value=MagicMock()),
            patch(
                "src.orchestrator.swarm_orchestrator.SwarmOrchestrator._verify_intent",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await orch._run_qa_discovery_swarm(
                swarm_id="swarm_test",
                config=config,
                emitter=mock_emitter,
            )

        assert result.success is True
        assert len(result.worker_results) == 2
        assert result.worker_results[0].agent_type == "auto_discovery"
        assert result.worker_results[1].agent_type == "testgen"
        assert result.total_cost_usd == 0.05
```

**Step 2: Keep the simpler tests that still work**

Keep: `test_qa_discovery_mode_exists`, `test_qa_discovery_in_swarm_mode_agents`, `test_launch_swarm_request_accepts_qa_discovery_mode`

Remove/replace: `test_discovery_to_testgen_conversion_*` (function moved to supervisor), `test_qa_discovery_handles_phase1_failure` (error handling is now in supervisor)

Add a new test for `_format_discovery_queue`:

```python
    def test_format_discovery_queue(self):
        """_format_discovery_queue should format pages and flows as text."""
        from src.orchestrator.supervisor import _format_discovery_queue

        queue = [
            {"url": "https://example.com/", "title": "Home", "description": "Landing page", "forms": [], "type": "page"},
            {"name": "Login Flow", "description": "User logs in", "category": "auth", "steps": [{"action": "click", "target": "Login"}], "type": "flow"},
        ]
        text = _format_discovery_queue(queue, "https://example.com")
        assert "Home" in text
        assert "Landing page" in text
        assert "Login Flow" in text
        assert "auth" in text
```

**Step 3: Run all tests**

Run: `python -m pytest tests/orchestrator/test_swarm.py -v --timeout=300 2>&1 | tail -30`
Expected: All tests pass (41 existing + updated QA discovery tests)

**Step 4: Commit**

```bash
git add tests/orchestrator/test_swarm.py
git commit -m "test: update QA discovery tests for supervisor integration"
```

---

### Task 8: Test the supervisor auto_discovery node in isolation

**Files:**
- Create: `tests/orchestrator/test_supervisor_qa_discovery.py`

**Step 1: Write focused tests for the new supervisor node**

```python
"""Tests for supervisor auto_discovery node and discovery-testgen interleaving."""

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSupervisorAutoDiscoveryNode:
    """Test the supervisor_auto_discovery_node wrapper."""

    @pytest.mark.asyncio
    async def test_discovery_node_returns_batch(self):
        """Should discover pages and write them to discovery_queue."""
        from src.orchestrator.supervisor import supervisor_auto_discovery_node

        @dataclass
        class FakePage:
            url: str
            title: str = ""
            description: str = ""
            forms: list = field(default_factory=list)

        @dataclass
        class FakeDiscoveryResult:
            pages_discovered: list = field(default_factory=list)
            flows_discovered: list = field(default_factory=list)
            suggested_tests: list = field(default_factory=list)

        fake_result = FakeDiscoveryResult(
            pages_discovered=[
                FakePage(url="https://example.com/", title="Home"),
                FakePage(url="https://example.com/about", title="About"),
            ],
        )

        mock_discovery = AsyncMock()
        mock_discovery.discover = AsyncMock(return_value=fake_result)

        state = {
            "app_url": "https://example.com",
            "discovery_batch_index": 0,
            "discovery_pages_found": 0,
            "testable_surfaces": [],
            "results": {},
        }

        with patch("src.agents.AutoDiscovery", return_value=mock_discovery):
            result = await supervisor_auto_discovery_node(state)

        assert len(result["discovery_queue"]) == 2
        assert result["discovery_pages_found"] == 2
        assert result["discovery_batch_index"] == 1
        # Found fewer than BATCH_SIZE(5) pages → discovery_complete
        assert result["discovery_complete"] is True

    @pytest.mark.asyncio
    async def test_discovery_node_filters_already_seen(self):
        """Should not add pages that are already in testable_surfaces."""
        from src.orchestrator.supervisor import supervisor_auto_discovery_node

        @dataclass
        class FakePage:
            url: str
            title: str = ""
            description: str = ""
            forms: list = field(default_factory=list)

        @dataclass
        class FakeDiscoveryResult:
            pages_discovered: list = field(default_factory=list)
            flows_discovered: list = field(default_factory=list)
            suggested_tests: list = field(default_factory=list)

        fake_result = FakeDiscoveryResult(
            pages_discovered=[
                FakePage(url="https://example.com/", title="Home"),
                FakePage(url="https://example.com/new", title="New Page"),
            ],
        )

        mock_discovery = AsyncMock()
        mock_discovery.discover = AsyncMock(return_value=fake_result)

        state = {
            "app_url": "https://example.com",
            "discovery_batch_index": 1,
            "discovery_pages_found": 1,
            "testable_surfaces": [
                {"url": "https://example.com/", "title": "Home", "type": "page"},
            ],
            "results": {},
        }

        with patch("src.agents.AutoDiscovery", return_value=mock_discovery):
            result = await supervisor_auto_discovery_node(state)

        # Only the new page should be in queue
        assert len(result["discovery_queue"]) == 1
        assert result["discovery_queue"][0]["url"] == "https://example.com/new"

    @pytest.mark.asyncio
    async def test_discovery_node_max_batches(self):
        """Should set discovery_complete when max batches reached."""
        from src.orchestrator.supervisor import supervisor_auto_discovery_node

        state = {
            "app_url": "https://example.com",
            "discovery_batch_index": 10,  # Already at max
            "discovery_pages_found": 50,
            "testable_surfaces": [],
            "results": {},
        }

        result = await supervisor_auto_discovery_node(state)
        assert result["discovery_complete"] is True

    @pytest.mark.asyncio
    async def test_discovery_node_no_url(self):
        """Should handle missing app_url gracefully."""
        from src.orchestrator.supervisor import supervisor_auto_discovery_node

        state = {"app_url": None, "results": {}}
        result = await supervisor_auto_discovery_node(state)
        assert result["discovery_complete"] is True


class TestFormatDiscoveryQueue:
    """Test the _format_discovery_queue helper."""

    def test_formats_pages_and_flows(self):
        from src.orchestrator.supervisor import _format_discovery_queue

        queue = [
            {"url": "https://a.com/", "title": "Home", "description": "Main page", "forms": [], "type": "page"},
            {"name": "Checkout", "description": "Buy stuff", "category": "ecommerce", "steps": [], "type": "flow"},
        ]
        text = _format_discovery_queue(queue, "https://a.com")
        assert "Home" in text
        assert "Checkout" in text
        assert "ecommerce" in text

    def test_empty_queue(self):
        from src.orchestrator.supervisor import _format_discovery_queue

        text = _format_discovery_queue([], "https://a.com")
        assert "https://a.com" in text


class TestSupervisorDiscoveryDecisionRules:
    """Test that the supervisor prompt includes discovery rules."""

    def test_prompt_has_auto_discovery(self):
        from src.orchestrator.supervisor import create_supervisor_prompt

        prompt = create_supervisor_prompt()
        assert "auto_discovery" in prompt
        assert "discovery_complete" in prompt

    def test_auto_discovery_in_agents_list(self):
        from src.orchestrator.supervisor import AGENTS, AGENT_DESCRIPTIONS

        assert "auto_discovery" in AGENTS
        assert "auto_discovery" in AGENT_DESCRIPTIONS
```

**Step 2: Run the new tests**

Run: `python -m pytest tests/orchestrator/test_supervisor_qa_discovery.py -v --timeout=120 2>&1 | tail -20`
Expected: All pass

**Step 3: Run full test suite regression check**

Run: `python -m pytest tests/orchestrator/test_swarm.py tests/orchestrator/test_supervisor_qa_discovery.py -v --timeout=300 2>&1 | tail -20`
Expected: All pass

**Step 4: Commit**

```bash
git add tests/orchestrator/test_supervisor_qa_discovery.py
git commit -m "test: add supervisor auto_discovery node + discovery queue tests"
```
