"""Enhanced LangGraph orchestrator with advanced features.

Features:
- Parallel test execution using Send API (map-reduce pattern)
- Human-in-the-loop approval for auto-healing
- Streaming support for real-time updates
- Subgraphs for modular test workflows
- Persistent checkpointing with time-travel
- Short-term and long-term memory
"""

from typing import Literal, Annotated, Sequence, Any
from dataclasses import dataclass
import asyncio
import structlog

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send, Command, interrupt
from langgraph.prebuilt import ToolNode

from .state import TestingState, TestStatus, TestType
from ..config import Settings, get_settings

logger = structlog.get_logger()


# ============================================================================
# ENHANCED STATE WITH PARALLEL EXECUTION SUPPORT
# ============================================================================

@dataclass
class ParallelTestBatch:
    """A batch of tests to execute in parallel."""
    batch_id: str
    test_ids: list[str]
    test_type: TestType
    max_concurrent: int = 5


@dataclass
class HumanApprovalRequest:
    """Request for human approval."""
    request_id: str
    request_type: Literal["heal_approval", "deploy_approval", "skip_approval"]
    context: dict
    options: list[str]
    timeout_seconds: int = 300


# ============================================================================
# PARALLEL EXECUTION - MAP-REDUCE PATTERN
# ============================================================================

def create_parallel_test_batches(state: TestingState) -> list[Send]:
    """
    Create parallel test batches using LangGraph's Send API.

    Groups tests by type and sends them to parallel execution nodes.
    This implements the map-reduce pattern for concurrent test execution.
    """
    test_plan = state.get("test_plan", [])
    if not test_plan:
        return []

    # Group tests by type for parallel execution
    ui_tests = [t for t in test_plan if t.get("type") == "ui"]
    api_tests = [t for t in test_plan if t.get("type") == "api"]
    db_tests = [t for t in test_plan if t.get("type") == "db"]

    sends = []

    # Send UI tests to UI executor (max 3 parallel browsers)
    if ui_tests:
        sends.append(Send("execute_ui_tests_parallel", {
            "tests": ui_tests,
            "max_concurrent": 3,
            "parent_state": state,
        }))

    # Send API tests to API executor (max 10 parallel)
    if api_tests:
        sends.append(Send("execute_api_tests_parallel", {
            "tests": api_tests,
            "max_concurrent": 10,
            "parent_state": state,
        }))

    # Send DB tests to DB executor (max 5 parallel)
    if db_tests:
        sends.append(Send("execute_db_tests_parallel", {
            "tests": db_tests,
            "max_concurrent": 5,
            "parent_state": state,
        }))

    logger.info(
        "Created parallel test batches",
        ui_count=len(ui_tests),
        api_count=len(api_tests),
        db_count=len(db_tests),
    )

    return sends


async def execute_ui_tests_parallel(batch_state: dict) -> dict:
    """Execute UI tests in parallel with controlled concurrency."""
    from ..agents.ui_tester import UITesterAgent

    tests = batch_state["tests"]
    max_concurrent = batch_state.get("max_concurrent", 3)
    parent_state = batch_state["parent_state"]

    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def run_single_test(test: dict) -> dict:
        async with semaphore:
            agent = UITesterAgent()
            result = await agent.execute(
                test_spec=test,
                app_url=parent_state["app_url"],
            )
            return {
                "test_id": test["id"],
                "result": result.data if result.success else None,
                "error": result.error,
                "success": result.success,
            }

    # Run all tests concurrently with semaphore limiting
    tasks = [run_single_test(test) for test in tests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "test_id": tests[i]["id"],
                "success": False,
                "error": str(result),
            })
        else:
            processed_results.append(result)

    return {"ui_results": processed_results}


async def execute_api_tests_parallel(batch_state: dict) -> dict:
    """Execute API tests in parallel with high concurrency."""
    from ..agents.api_tester import APITesterAgent

    tests = batch_state["tests"]
    max_concurrent = batch_state.get("max_concurrent", 10)
    parent_state = batch_state["parent_state"]

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_single_test(test: dict) -> dict:
        async with semaphore:
            agent = APITesterAgent()
            result = await agent.execute(
                test_spec=test,
                base_url=parent_state["app_url"],
            )
            return {
                "test_id": test["id"],
                "result": result.data if result.success else None,
                "error": result.error,
                "success": result.success,
            }

    tasks = [run_single_test(test) for test in tests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "test_id": tests[i]["id"],
                "success": False,
                "error": str(result),
            })
        else:
            processed_results.append(result)

    return {"api_results": processed_results}


async def execute_db_tests_parallel(batch_state: dict) -> dict:
    """Execute DB tests in parallel."""
    from ..agents.db_tester import DBTesterAgent

    tests = batch_state["tests"]
    max_concurrent = batch_state.get("max_concurrent", 5)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_single_test(test: dict) -> dict:
        async with semaphore:
            agent = DBTesterAgent()
            result = await agent.execute(test_spec=test)
            return {
                "test_id": test["id"],
                "result": result.data if result.success else None,
                "error": result.error,
                "success": result.success,
            }

    tasks = [run_single_test(test) for test in tests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "test_id": tests[i]["id"],
                "success": False,
                "error": str(result),
            })
        else:
            processed_results.append(result)

    return {"db_results": processed_results}


def aggregate_parallel_results(state: TestingState) -> TestingState:
    """
    Reduce function: Aggregate results from parallel test execution.

    This is the 'reduce' part of the map-reduce pattern.
    """
    ui_results = state.get("ui_results", [])
    api_results = state.get("api_results", [])
    db_results = state.get("db_results", [])

    all_results = ui_results + api_results + db_results

    passed = sum(1 for r in all_results if r.get("success"))
    failed = sum(1 for r in all_results if not r.get("success"))

    # Collect failures for healing
    failures = [r for r in all_results if not r.get("success")]
    healing_queue = [f["test_id"] for f in failures]

    return {
        **state,
        "test_results": all_results,
        "passed_count": passed,
        "failed_count": failed,
        "healing_queue": healing_queue,
        "failures": failures,
    }


# ============================================================================
# HUMAN-IN-THE-LOOP APPROVAL
# ============================================================================

def request_human_approval(state: TestingState) -> TestingState:
    """
    Request human approval before auto-healing.

    Uses LangGraph's interrupt() to pause execution and wait for human input.
    The execution can be resumed from any machine after approval.
    """
    failures = state.get("failures", [])
    healing_queue = state.get("healing_queue", [])

    if not healing_queue:
        return {**state, "human_approved": True}

    # Create approval request
    approval_request = {
        "request_type": "heal_approval",
        "failures_count": len(failures),
        "tests_to_heal": healing_queue,
        "failure_summaries": [
            {
                "test_id": f.get("test_id"),
                "error": f.get("error", "Unknown error")[:200],
            }
            for f in failures[:5]  # Show first 5
        ],
        "options": [
            "approve_all",      # Heal all failures
            "approve_critical", # Only heal critical tests
            "skip_healing",     # Skip healing, proceed to report
            "abort_run",        # Stop the entire run
        ],
    }

    logger.info("Requesting human approval for healing", request=approval_request)

    # Interrupt execution and wait for human input
    # This saves state and can be resumed later
    human_response = interrupt(approval_request)

    # Process human response
    if human_response == "abort_run":
        return {**state, "should_continue": False, "error": "Run aborted by user"}
    elif human_response == "skip_healing":
        return {**state, "healing_queue": [], "human_approved": False}
    elif human_response == "approve_critical":
        # Filter to only critical tests
        critical_tests = [
            t for t in healing_queue
            if state.get("test_priorities", {}).get(t) == "critical"
        ]
        return {**state, "healing_queue": critical_tests, "human_approved": True}
    else:  # approve_all
        return {**state, "human_approved": True}


def should_request_approval(state: TestingState) -> bool:
    """Check if human approval is required."""
    settings = get_settings()

    # Always request approval if enabled and there are failures
    if settings.require_human_approval_for_healing:
        if state.get("healing_queue"):
            return True

    # Request approval if failure rate is high
    total = state.get("passed_count", 0) + state.get("failed_count", 0)
    if total > 0:
        failure_rate = state.get("failed_count", 0) / total
        if failure_rate > 0.5:  # More than 50% failing
            return True

    return False


# ============================================================================
# STREAMING SUPPORT
# ============================================================================

async def stream_test_progress(state: TestingState) -> dict:
    """
    Yield streaming updates during test execution.

    LangGraph automatically streams:
    - Node transitions
    - State updates
    - LLM tokens
    - Tool calls

    This function adds custom progress events.
    """
    current_test = state.get("current_test")
    if current_test:
        yield {
            "type": "test_started",
            "test_id": current_test.get("id"),
            "test_name": current_test.get("name"),
            "timestamp": state.get("started_at"),
        }

    # Progress update
    total_tests = len(state.get("test_plan", []))
    completed = state.get("passed_count", 0) + state.get("failed_count", 0)

    yield {
        "type": "progress",
        "completed": completed,
        "total": total_tests,
        "passed": state.get("passed_count", 0),
        "failed": state.get("failed_count", 0),
        "percentage": (completed / total_tests * 100) if total_tests > 0 else 0,
    }


# ============================================================================
# ENHANCED ROUTING FUNCTIONS
# ============================================================================

def route_after_analysis(state: TestingState) -> Literal["plan_tests", "report", "__end__"]:
    """Route after code analysis with enhanced logic."""
    if state.get("error"):
        logger.error("Analysis failed", error=state["error"])
        return "report"

    if not state.get("testable_surfaces"):
        logger.warning("No testable surfaces found")
        return "report"

    return "plan_tests"


def route_after_planning(state: TestingState) -> Literal["execute_parallel", "execute_sequential", "report"]:
    """Route to parallel or sequential execution based on test count."""
    if state.get("error"):
        return "report"

    test_plan = state.get("test_plan", [])
    if not test_plan:
        return "report"

    settings = get_settings()

    # Use parallel execution for larger test suites
    if len(test_plan) >= settings.parallel_threshold:
        return "execute_parallel"

    return "execute_sequential"


def route_after_execution(state: TestingState) -> Literal["request_approval", "self_heal", "report"]:
    """Route after test execution with human-in-the-loop."""
    if state.get("error"):
        return "report"

    # Check if we need healing
    if state.get("healing_queue"):
        settings = get_settings()

        if settings.self_heal_enabled:
            # Check if approval is needed
            if should_request_approval(state):
                return "request_approval"
            return "self_heal"

    return "report"


def route_after_approval(state: TestingState) -> Literal["self_heal", "report"]:
    """Route after human approval."""
    if state.get("human_approved") and state.get("healing_queue"):
        return "self_heal"
    return "report"


def route_after_healing(state: TestingState) -> Literal["execute_healed", "report"]:
    """Route after self-healing."""
    if state.get("healed_tests"):
        return "execute_healed"
    return "report"


# ============================================================================
# SUBGRAPHS FOR MODULAR WORKFLOWS
# ============================================================================

def create_quality_subgraph() -> StateGraph:
    """
    Create a subgraph for quality analysis.

    This runs performance, security, and accessibility checks.
    Can be embedded in the main graph or run standalone.
    """
    from .nodes import (
        performance_analysis_node,
        security_scan_node,
        accessibility_check_node,
        quality_report_node,
    )

    graph = StateGraph(TestingState)

    graph.add_node("performance", performance_analysis_node)
    graph.add_node("security", security_scan_node)
    graph.add_node("accessibility", accessibility_check_node)
    graph.add_node("quality_report", quality_report_node)

    # All quality checks can run in parallel
    graph.set_entry_point("performance")
    graph.add_edge("performance", "security")
    graph.add_edge("security", "accessibility")
    graph.add_edge("accessibility", "quality_report")
    graph.add_edge("quality_report", END)

    return graph


# ============================================================================
# MAIN ENHANCED GRAPH
# ============================================================================

def create_enhanced_testing_graph(settings: Settings) -> StateGraph:
    """
    Create the enhanced LangGraph state machine with all advanced features.

    Graph structure:

        [START]
           │
           ▼
      ┌─────────────┐
      │analyze_code │
      └─────────────┘
           │
           ▼
      ┌─────────────┐
      │ plan_tests  │
      └─────────────┘
           │
           ├── (small suite) ──────────────────┐
           │                                    │
           ▼                                    ▼
      ┌──────────────────┐            ┌────────────────┐
      │execute_parallel  │            │execute_sequence│
      │  ┌─────┐         │            └────────────────┘
      │  │ UI  │ ←──┐    │                    │
      │  └─────┘    │    │                    │
      │  ┌─────┐    │    │                    │
      │  │ API │ ←──┤    │                    │
      │  └─────┘    │    │                    │
      │  ┌─────┐    │    │                    │
      │  │ DB  │ ←──┘    │                    │
      │  └─────┘         │                    │
      │       │          │                    │
      │       ▼          │                    │
      │  ┌──────────┐    │                    │
      │  │aggregate │    │                    │
      │  └──────────┘    │                    │
      └──────────────────┘                    │
                │                             │
                └─────────────┬───────────────┘
                              │
                              ▼
                   (failures detected?)
                        │         │
                   (yes)│         │(no)
                        ▼         │
                ┌───────────────┐ │
                │request_approve│ │
                └───────────────┘ │
                        │         │
                   (approved?)    │
                        │         │
                        ▼         │
                  ┌──────────┐    │
                  │self_heal │    │
                  └──────────┘    │
                        │         │
                        ▼         │
                  ┌──────────┐    │
                  │exec_healed│   │
                  └──────────┘    │
                        │         │
                        └────┬────┘
                             │
                             ▼
                    ┌─────────────┐
                    │quality_check│ (subgraph)
                    └─────────────┘
                             │
                             ▼
                      ┌──────────┐
                      │  report  │
                      └──────────┘
                             │
                             ▼
                          [END]
    """
    from .nodes import (
        analyze_code_node,
        plan_tests_node,
        execute_test_node,
        self_heal_node,
        report_node,
    )

    graph = StateGraph(TestingState)

    # Add main nodes
    graph.add_node("analyze_code", analyze_code_node)
    graph.add_node("plan_tests", plan_tests_node)
    graph.add_node("execute_sequential", execute_test_node)
    graph.add_node("execute_parallel", create_parallel_test_batches)
    graph.add_node("execute_ui_tests_parallel", execute_ui_tests_parallel)
    graph.add_node("execute_api_tests_parallel", execute_api_tests_parallel)
    graph.add_node("execute_db_tests_parallel", execute_db_tests_parallel)
    graph.add_node("aggregate_results", aggregate_parallel_results)
    graph.add_node("request_approval", request_human_approval)
    graph.add_node("self_heal", self_heal_node)
    graph.add_node("execute_healed", execute_test_node)
    graph.add_node("quality_check", create_quality_subgraph().compile())
    graph.add_node("report", report_node)

    # Entry point
    graph.set_entry_point("analyze_code")

    # Analysis → Planning
    graph.add_conditional_edges(
        "analyze_code",
        route_after_analysis,
        {
            "plan_tests": "plan_tests",
            "report": "report",
            "__end__": END,
        }
    )

    # Planning → Execution (parallel or sequential)
    graph.add_conditional_edges(
        "plan_tests",
        route_after_planning,
        {
            "execute_parallel": "execute_parallel",
            "execute_sequential": "execute_sequential",
            "report": "report",
        }
    )

    # Parallel execution edges
    graph.add_edge("execute_ui_tests_parallel", "aggregate_results")
    graph.add_edge("execute_api_tests_parallel", "aggregate_results")
    graph.add_edge("execute_db_tests_parallel", "aggregate_results")

    # Aggregation → Approval/Healing/Report
    graph.add_conditional_edges(
        "aggregate_results",
        route_after_execution,
        {
            "request_approval": "request_approval",
            "self_heal": "self_heal",
            "report": "quality_check",
        }
    )

    # Sequential execution → same routing
    graph.add_conditional_edges(
        "execute_sequential",
        route_after_execution,
        {
            "request_approval": "request_approval",
            "self_heal": "self_heal",
            "report": "quality_check",
        }
    )

    # Approval → Heal or Report
    graph.add_conditional_edges(
        "request_approval",
        route_after_approval,
        {
            "self_heal": "self_heal",
            "report": "quality_check",
        }
    )

    # Healing → Re-execute or Report
    graph.add_conditional_edges(
        "self_heal",
        route_after_healing,
        {
            "execute_healed": "execute_healed",
            "report": "quality_check",
        }
    )

    # Healed execution → Report
    graph.add_edge("execute_healed", "quality_check")

    # Quality check → Report
    graph.add_edge("quality_check", "report")

    # Report → End
    graph.add_edge("report", END)

    return graph


# ============================================================================
# ENHANCED ORCHESTRATOR
# ============================================================================

class EnhancedTestingOrchestrator:
    """
    Enhanced orchestrator with parallel execution, streaming, and human-in-the-loop.

    Features:
    - Parallel test execution for large suites
    - Human approval for auto-healing
    - Real-time streaming updates
    - Persistent checkpointing
    - Time-travel debugging

    Usage:
        orchestrator = EnhancedTestingOrchestrator(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000"
        )

        # Streaming execution
        async for event in orchestrator.stream():
            print(event)

        # Resume from checkpoint
        results = await orchestrator.resume(thread_id="previous-run-id")
    """

    def __init__(
        self,
        codebase_path: str,
        app_url: str,
        settings: Settings | None = None,
        pr_number: int | None = None,
        changed_files: list[str] | None = None,
    ):
        self.codebase_path = codebase_path
        self.app_url = app_url
        self.settings = settings or get_settings()
        self.pr_number = pr_number
        self.changed_files = changed_files or []

        # Create enhanced graph
        graph = create_enhanced_testing_graph(self.settings)

        # Compile with persistent checkpointing
        self.memory = MemorySaver()
        self.app = graph.compile(
            checkpointer=self.memory,
            interrupt_before=["request_approval"] if self.settings.require_human_approval_for_healing else [],
        )

        self.log = logger.bind(
            codebase=codebase_path,
            app_url=app_url,
            enhanced=True,
        )

    async def run(self, thread_id: str | None = None) -> dict:
        """Run the full test suite."""
        from .state import create_initial_state

        initial_state = create_initial_state(
            codebase_path=self.codebase_path,
            app_url=self.app_url,
            pr_number=self.pr_number,
            changed_files=self.changed_files,
        )

        config = {"configurable": {"thread_id": thread_id or initial_state["run_id"]}}

        self.log.info("Starting enhanced test run", run_id=initial_state["run_id"])

        final_state = await self.app.ainvoke(initial_state, config)

        return final_state

    async def stream(self, thread_id: str | None = None):
        """Stream test execution with real-time updates."""
        from .state import create_initial_state

        initial_state = create_initial_state(
            codebase_path=self.codebase_path,
            app_url=self.app_url,
            pr_number=self.pr_number,
            changed_files=self.changed_files,
        )

        config = {"configurable": {"thread_id": thread_id or initial_state["run_id"]}}

        # Stream with multiple modes
        async for event in self.app.astream(
            initial_state,
            config,
            stream_mode=["values", "updates", "messages"],
        ):
            yield event

    async def resume(self, thread_id: str, human_response: str | None = None) -> dict:
        """Resume execution from a checkpoint, optionally with human input."""
        config = {"configurable": {"thread_id": thread_id}}

        if human_response:
            # Resume with human input
            final_state = await self.app.ainvoke(
                Command(resume=human_response),
                config,
            )
        else:
            # Resume from last checkpoint
            final_state = await self.app.ainvoke(None, config)

        return final_state

    def get_checkpoint_history(self, thread_id: str) -> list[dict]:
        """Get checkpoint history for time-travel debugging."""
        checkpoints = []
        for checkpoint in self.memory.list({"configurable": {"thread_id": thread_id}}):
            checkpoints.append({
                "checkpoint_id": checkpoint.checkpoint_id,
                "created_at": checkpoint.metadata.get("created_at"),
                "node": checkpoint.metadata.get("source"),
            })
        return checkpoints

    async def replay_from_checkpoint(self, thread_id: str, checkpoint_id: str) -> dict:
        """Replay execution from a specific checkpoint."""
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        return await self.app.ainvoke(None, config)
