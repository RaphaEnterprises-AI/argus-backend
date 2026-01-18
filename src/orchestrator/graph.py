"""LangGraph orchestrator for the testing agent.

Supports human-in-the-loop breakpoints for approval workflows:
- interrupt_before self_heal node to require healing approval
- interrupt_before plan_tests node for test plan approval
- interrupt() function for cleaner human-in-the-loop (enhanced mode)

Enhanced features (v2):
- Parallel test execution using Send API (map-reduce pattern)
- interrupt() function for human approval (cleaner than interrupt_before)
- Command(resume=...) for passing human input when resuming
- Subgraphs for modular quality analysis
- Multi-mode streaming support

When breakpoints are enabled, execution will pause before the specified nodes,
allowing external systems to approve/reject/modify state before resuming.
"""

import asyncio
from dataclasses import dataclass
from typing import Literal

import structlog
from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send, interrupt

from ..config import Settings, get_settings
from .checkpointer import get_checkpointer
from .state import TestingState, TestType

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


def get_interrupt_nodes(settings: Settings) -> list[str]:
    """
    Determine which nodes should have interrupt_before breakpoints.

    Based on configuration settings, returns a list of node names
    that should pause for human approval.

    Args:
        settings: Application settings

    Returns:
        List of node names to interrupt before
    """
    interrupt_nodes = []

    # Add self_heal to interrupt list if healing approval is required
    if settings.require_healing_approval or settings.require_human_approval_for_healing:
        interrupt_nodes.append("self_heal")
        logger.info("Breakpoint enabled: self_heal node requires approval")

    # Add plan_tests to interrupt list if test plan approval is required
    if settings.require_test_plan_approval:
        interrupt_nodes.append("execute_test")  # Interrupt before first test execution
        logger.info("Breakpoint enabled: test execution requires plan approval")

    return interrupt_nodes


def route_after_analysis(state: TestingState) -> Literal["plan_tests", "report", "__end__"]:
    """Route after code analysis."""
    if state.get("error"):
        return "report"
    if not state.get("testable_surfaces"):
        logger.warning("No testable surfaces found")
        return "report"
    return "plan_tests"


def route_after_planning(state: TestingState) -> Literal["execute_test", "report", "__end__"]:
    """Route after test planning."""
    if state.get("error"):
        return "report"
    if not state.get("test_plan"):
        logger.warning("No tests in plan")
        return "report"
    return "execute_test"


def route_after_execution(
    state: TestingState
) -> Literal["execute_test", "self_heal", "report", "__end__"]:
    """Route after test execution."""
    # Check for errors
    if state.get("error"):
        return "report"

    # Check if we need to heal any failures
    if state.get("healing_queue") and state["should_continue"]:
        settings = get_settings()
        if settings.self_heal_enabled:
            return "self_heal"

    # Check if more tests to run
    current_idx = state.get("current_test_index", 0)
    test_plan = state.get("test_plan", [])

    if current_idx < len(test_plan):
        return "execute_test"

    return "report"


def route_after_healing(
    state: TestingState
) -> Literal["execute_test", "report", "__end__"]:
    """Route after self-healing attempt."""
    # If healing was successful, retry the test
    if state.get("healing_queue"):
        return "execute_test"

    # Check if more tests to run
    current_idx = state.get("current_test_index", 0)
    test_plan = state.get("test_plan", [])

    if current_idx < len(test_plan):
        return "execute_test"

    return "report"


def should_continue(state: TestingState) -> bool:
    """Check if we should continue the test run."""
    # Check iteration limit
    if state["iteration"] >= state["max_iterations"]:
        logger.warning("Max iterations reached")
        return False

    # Check cost limit
    settings = get_settings()
    if state["total_cost"] >= settings.cost_limit_per_run:
        logger.warning("Cost limit reached", cost=state["total_cost"])
        return False

    # Check error state
    if state.get("error"):
        return False

    return state.get("should_continue", True)


# ============================================================================
# PARALLEL EXECUTION - MAP-REDUCE PATTERN (Send API)
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
# HUMAN-IN-THE-LOOP APPROVAL (interrupt() function)
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
# ENHANCED ROUTING FUNCTIONS (for parallel execution)
# ============================================================================

def route_after_planning_enhanced(
    state: TestingState
) -> Literal["execute_parallel", "execute_sequential", "report"]:
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


def route_after_execution_enhanced(
    state: TestingState
) -> Literal["request_approval", "self_heal", "report"]:
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


def route_after_healing_enhanced(
    state: TestingState
) -> Literal["execute_healed", "report"]:
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
    # Import node implementations (may not exist yet, graceful fallback)
    try:
        from .nodes import (
            accessibility_check_node,
            performance_analysis_node,
            quality_report_node,
            security_scan_node,
        )
    except ImportError:
        # Placeholder nodes if not implemented yet
        def performance_analysis_node(state): return state
        def security_scan_node(state): return state
        def accessibility_check_node(state): return state
        def quality_report_node(state): return state

    graph = StateGraph(TestingState)

    graph.add_node("performance", performance_analysis_node)
    graph.add_node("security", security_scan_node)
    graph.add_node("accessibility", accessibility_check_node)
    graph.add_node("quality_report", quality_report_node)

    # All quality checks run sequentially (can be made parallel with Send)
    graph.set_entry_point("performance")
    graph.add_edge("performance", "security")
    graph.add_edge("security", "accessibility")
    graph.add_edge("accessibility", "quality_report")
    graph.add_edge("quality_report", END)

    return graph


def create_testing_graph(settings: Settings) -> StateGraph:
    """
    Create the LangGraph state machine for testing orchestration.

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
           ▼
      ┌─────────────┐◄─────────────┐
      │execute_test │              │
      └─────────────┘              │
           │                       │
           ├─── (more tests) ──────┤
           │                       │
           ├─── (failure) ────┐    │
           │                  ▼    │
           │            ┌──────────┴──┐
           │            │  self_heal  │
           │            └─────────────┘
           │
           ▼
      ┌─────────────┐
      │   report    │
      └─────────────┘
           │
           ▼
         [END]
    """
    # Import node implementations
    from .nodes import (
        analyze_code_node,
        execute_test_node,
        plan_tests_node,
        report_node,
        self_heal_node,
    )

    # Create graph
    graph = StateGraph(TestingState)

    # Add nodes
    graph.add_node("analyze_code", analyze_code_node)
    graph.add_node("plan_tests", plan_tests_node)
    graph.add_node("execute_test", execute_test_node)
    graph.add_node("self_heal", self_heal_node)
    graph.add_node("report", report_node)

    # Define edges
    graph.set_entry_point("analyze_code")

    graph.add_conditional_edges(
        "analyze_code",
        route_after_analysis,
        {
            "plan_tests": "plan_tests",
            "report": "report",
            "__end__": END,
        }
    )

    graph.add_conditional_edges(
        "plan_tests",
        route_after_planning,
        {
            "execute_test": "execute_test",
            "report": "report",
            "__end__": END,
        }
    )

    graph.add_conditional_edges(
        "execute_test",
        route_after_execution,
        {
            "execute_test": "execute_test",
            "self_heal": "self_heal",
            "report": "report",
            "__end__": END,
        }
    )

    graph.add_conditional_edges(
        "self_heal",
        route_after_healing,
        {
            "execute_test": "execute_test",
            "report": "report",
            "__end__": END,
        }
    )

    graph.add_edge("report", END)

    return graph


def create_testing_graph_with_interrupts(settings: Settings) -> StateGraph:
    """
    Create graph with human-in-the-loop breakpoints.

    This is a convenience wrapper that creates the graph but doesn't compile it,
    allowing the caller to specify interrupt_before nodes at compile time.

    Args:
        settings: Application settings

    Returns:
        Uncompiled StateGraph
    """
    return create_testing_graph(settings)


class TestingOrchestrator:
    """
    Main orchestrator for E2E testing.

    Supports human-in-the-loop breakpoints for approval workflows.
    When require_healing_approval is True, execution pauses before self_heal
    and waits for approval via the approvals API.

    Usage:
        orchestrator = TestingOrchestrator(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000"
        )
        results = await orchestrator.run()

    With breakpoints:
        # Enable approval requirement in settings or env
        # REQUIRE_HEALING_APPROVAL=true

        # Start test run - will pause at self_heal
        result = await orchestrator.run(thread_id="my-run-123")

        # Check if paused (result will have next node info)
        # Use approvals API to approve/reject

        # Resume execution
        result = await orchestrator.resume(thread_id="my-run-123")
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

        # Create graph
        graph = create_testing_graph(self.settings)

        # Compile with checkpointer for state persistence
        # Uses PostgresSaver if DATABASE_URL is set, otherwise MemorySaver
        self.checkpointer = get_checkpointer()

        # Determine which nodes need approval breakpoints
        interrupt_nodes = get_interrupt_nodes(self.settings)

        # Compile with interrupt_before for human-in-the-loop
        self.app = graph.compile(
            checkpointer=self.checkpointer,
            interrupt_before=interrupt_nodes if interrupt_nodes else None,
        )

        self.log = logger.bind(
            codebase=codebase_path,
            app_url=app_url,
            breakpoints=interrupt_nodes,
        )

    async def run(self, thread_id: str | None = None) -> dict:
        """
        Run the full test suite.

        Args:
            thread_id: Optional thread ID for checkpointing

        Returns:
            Final state with all test results
        """
        from .state import create_initial_state

        initial_state = create_initial_state(
            codebase_path=self.codebase_path,
            app_url=self.app_url,
            pr_number=self.pr_number,
            changed_files=self.changed_files,
        )

        config = {"configurable": {"thread_id": thread_id or initial_state["run_id"]}}

        self.log.info("Starting test run", run_id=initial_state["run_id"])

        try:
            # Run the graph
            final_state = await self.app.ainvoke(initial_state, config)

            self.log.info(
                "Test run completed",
                passed=final_state["passed_count"],
                failed=final_state["failed_count"],
                skipped=final_state["skipped_count"],
                cost=final_state["total_cost"],
            )

            return final_state

        except Exception as e:
            self.log.error("Test run failed", error=str(e))
            raise

    async def run_single_test(self, test_spec: dict, thread_id: str | None = None) -> dict:
        """Run a single test by ID or spec."""
        from .state import create_initial_state

        initial_state = create_initial_state(
            codebase_path=self.codebase_path,
            app_url=self.app_url,
        )

        # Skip analysis and planning, go directly to execution
        initial_state["test_plan"] = [test_spec]
        initial_state["next_agent"] = "execute_test"

        config = {"configurable": {"thread_id": thread_id or initial_state["run_id"]}}

        # Create a modified graph that starts at execute_test
        # For now, use the full graph but with pre-populated state
        final_state = await self.app.ainvoke(initial_state, config)

        return final_state

    def get_run_summary(self, state: dict) -> dict:
        """Get a summary of the test run."""
        total_tests = state["passed_count"] + state["failed_count"] + state["skipped_count"]

        return {
            "run_id": state["run_id"],
            "started_at": state["started_at"],
            "total_tests": total_tests,
            "passed": state["passed_count"],
            "failed": state["failed_count"],
            "skipped": state["skipped_count"],
            "pass_rate": state["passed_count"] / total_tests if total_tests > 0 else 0,
            "total_cost": state["total_cost"],
            "iterations": state["iteration"],
            "error": state.get("error"),
        }

    async def resume(self, thread_id: str, state_updates: dict | None = None) -> dict:
        """
        Resume a paused execution.

        Used after approving a breakpoint to continue execution.

        Args:
            thread_id: The thread ID of the paused execution
            state_updates: Optional state modifications to apply before resuming

        Returns:
            Final state after execution completes (or hits another breakpoint)
        """
        config = {"configurable": {"thread_id": thread_id}}

        self.log.info("Resuming execution", thread_id=thread_id)

        try:
            # Apply state updates if provided
            if state_updates:
                await self.app.aupdate_state(config, state_updates)
                self.log.info("Applied state updates before resume", updates=list(state_updates.keys()))

            # Resume execution from the paused state
            final_state = await self.app.ainvoke(None, config)

            self.log.info(
                "Resumed execution completed",
                thread_id=thread_id,
                passed=final_state.get("passed_count", 0),
                failed=final_state.get("failed_count", 0),
            )

            return final_state

        except Exception as e:
            self.log.error("Resume failed", thread_id=thread_id, error=str(e))
            raise

    async def get_state(self, thread_id: str) -> dict | None:
        """
        Get the current state of a paused or completed execution.

        Args:
            thread_id: The thread ID to query

        Returns:
            Current state dict or None if not found
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state_snapshot = await self.app.aget_state(config)

            if state_snapshot:
                return {
                    "values": state_snapshot.values,
                    "next": state_snapshot.next,
                    "config": state_snapshot.config,
                    "metadata": state_snapshot.metadata,
                    "created_at": state_snapshot.created_at if hasattr(state_snapshot, 'created_at') else None,
                    "is_paused": bool(state_snapshot.next),
                    "paused_before": state_snapshot.next[0] if state_snapshot.next else None,
                }

            return None

        except Exception as e:
            self.log.error("Failed to get state", thread_id=thread_id, error=str(e))
            return None

    async def update_state(self, thread_id: str, updates: dict) -> bool:
        """
        Update the state of a paused execution.

        Useful for modifying state before approving a breakpoint.

        Args:
            thread_id: The thread ID to update
            updates: State updates to apply

        Returns:
            True if update succeeded, False otherwise
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            await self.app.aupdate_state(config, updates)
            self.log.info("State updated", thread_id=thread_id, updates=list(updates.keys()))
            return True

        except Exception as e:
            self.log.error("Failed to update state", thread_id=thread_id, error=str(e))
            return False

    async def abort(self, thread_id: str, reason: str = "Aborted by user") -> dict:
        """
        Abort a paused execution.

        Sets error state and skips to reporting.

        Args:
            thread_id: The thread ID to abort
            reason: Reason for aborting

        Returns:
            Final state after abort
        """
        config = {"configurable": {"thread_id": thread_id}}

        self.log.info("Aborting execution", thread_id=thread_id, reason=reason)

        try:
            # Update state to mark as aborted
            await self.app.aupdate_state(
                config,
                {
                    "error": reason,
                    "should_continue": False,
                    "healing_queue": [],  # Clear healing queue
                },
            )

            # Resume to let it complete the report
            final_state = await self.app.ainvoke(None, config)

            return final_state

        except Exception as e:
            self.log.error("Abort failed", thread_id=thread_id, error=str(e))
            raise


# ============================================================================
# ENHANCED GRAPH WITH PARALLEL EXECUTION
# ============================================================================

def create_enhanced_testing_graph(settings: Settings) -> StateGraph:
    """
    Create the enhanced LangGraph state machine with all advanced features.

    Enhanced features:
    - Parallel test execution for large suites (via Send API)
    - Human-in-the-loop with interrupt() function
    - Quality analysis subgraph
    - Multi-mode streaming support

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
        execute_test_node,
        plan_tests_node,
        report_node,
        self_heal_node,
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
        route_after_planning_enhanced,
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
        route_after_execution_enhanced,
        {
            "request_approval": "request_approval",
            "self_heal": "self_heal",
            "report": "quality_check",
        }
    )

    # Sequential execution → same routing
    graph.add_conditional_edges(
        "execute_sequential",
        route_after_execution_enhanced,
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
        route_after_healing_enhanced,
        {
            "execute_healed": "execute_healed",
            "report": "quality_check",
        }
    )

    # Healed execution → Quality Check
    graph.add_edge("execute_healed", "quality_check")

    # Quality check → Report
    graph.add_edge("quality_check", "report")

    # Report → End
    graph.add_edge("report", END)

    return graph


# ============================================================================
# ENHANCED ORCHESTRATOR WITH STREAMING & PARALLEL EXECUTION
# ============================================================================

class EnhancedTestingOrchestrator:
    """
    Enhanced orchestrator with parallel execution, streaming, and human-in-the-loop.

    Features:
    - Parallel test execution for large suites (triggered when tests >= parallel_threshold)
    - Human approval using interrupt() function (cleaner than interrupt_before)
    - Real-time streaming updates (values, updates, messages modes)
    - Persistent checkpointing with time-travel debugging
    - Resume with human input via Command(resume=...)

    Usage:
        orchestrator = EnhancedTestingOrchestrator(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000"
        )

        # Standard execution
        results = await orchestrator.run()

        # Streaming execution
        async for event in orchestrator.stream():
            print(event)

        # Resume with human input after interrupt
        results = await orchestrator.resume(
            thread_id="previous-run-id",
            human_response="approve_all"
        )
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

        # Use persistent checkpointer if available, otherwise memory
        self.checkpointer = get_checkpointer()

        # Compile - request_approval node uses interrupt() internally
        self.app = graph.compile(checkpointer=self.checkpointer)

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

        try:
            final_state = await self.app.ainvoke(initial_state, config)

            self.log.info(
                "Enhanced test run completed",
                passed=final_state.get("passed_count", 0),
                failed=final_state.get("failed_count", 0),
                cost=final_state.get("total_cost", 0),
            )

            return final_state

        except Exception as e:
            self.log.error("Enhanced test run failed", error=str(e))
            raise

    async def stream(self, thread_id: str | None = None):
        """
        Stream test execution with real-time updates.

        Yields events in multiple stream modes:
        - values: Full state after each node
        - updates: Delta updates per node
        - messages: LLM messages and tool calls

        Usage:
            async for event in orchestrator.stream():
                if "values" in event:
                    print(f"State: {event['values']}")
                elif "updates" in event:
                    print(f"Update: {event['updates']}")
        """
        from .state import create_initial_state

        initial_state = create_initial_state(
            codebase_path=self.codebase_path,
            app_url=self.app_url,
            pr_number=self.pr_number,
            changed_files=self.changed_files,
        )

        config = {"configurable": {"thread_id": thread_id or initial_state["run_id"]}}

        self.log.info("Starting streaming test run", run_id=initial_state["run_id"])

        # Stream with multiple modes
        async for event in self.app.astream(
            initial_state,
            config,
            stream_mode=["values", "updates", "messages"],
        ):
            yield event

    async def resume(self, thread_id: str, human_response: str | None = None) -> dict:
        """
        Resume execution from a checkpoint, optionally with human input.

        When the graph hits an interrupt() call (e.g., in request_human_approval),
        use this method to resume with the human's decision.

        Args:
            thread_id: The thread ID of the paused execution
            human_response: Human's response to the approval request
                          Options: "approve_all", "approve_critical", "skip_healing", "abort_run"

        Returns:
            Final state after execution completes (or hits another interrupt)
        """
        config = {"configurable": {"thread_id": thread_id}}

        self.log.info(
            "Resuming enhanced execution",
            thread_id=thread_id,
            human_response=human_response,
        )

        try:
            if human_response:
                # Resume with human input using Command
                final_state = await self.app.ainvoke(
                    Command(resume=human_response),
                    config,
                )
            else:
                # Resume from last checkpoint
                final_state = await self.app.ainvoke(None, config)

            self.log.info(
                "Enhanced execution resumed",
                thread_id=thread_id,
                passed=final_state.get("passed_count", 0),
                failed=final_state.get("failed_count", 0),
            )

            return final_state

        except Exception as e:
            self.log.error("Enhanced resume failed", thread_id=thread_id, error=str(e))
            raise

    async def get_state(self, thread_id: str) -> dict | None:
        """Get the current state of a paused or completed execution."""
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state_snapshot = await self.app.aget_state(config)

            if state_snapshot:
                return {
                    "values": state_snapshot.values,
                    "next": state_snapshot.next,
                    "config": state_snapshot.config,
                    "metadata": state_snapshot.metadata,
                    "is_paused": bool(state_snapshot.next),
                    "paused_at_node": state_snapshot.next[0] if state_snapshot.next else None,
                }

            return None

        except Exception as e:
            self.log.error("Failed to get state", thread_id=thread_id, error=str(e))
            return None

    def get_checkpoint_history(self, thread_id: str) -> list[dict]:
        """
        Get checkpoint history for time-travel debugging.

        Returns list of checkpoints that can be used to replay execution
        from any previous state.
        """
        try:
            checkpoints = []
            for checkpoint in self.checkpointer.list({"configurable": {"thread_id": thread_id}}):
                checkpoints.append({
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "created_at": checkpoint.metadata.get("created_at") if checkpoint.metadata else None,
                    "node": checkpoint.metadata.get("source") if checkpoint.metadata else None,
                })
            return checkpoints
        except Exception as e:
            self.log.error("Failed to get checkpoint history", thread_id=thread_id, error=str(e))
            return []

    async def replay_from_checkpoint(self, thread_id: str, checkpoint_id: str) -> dict:
        """
        Replay execution from a specific checkpoint (time-travel).

        Useful for debugging or retrying from a known good state.
        """
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

        self.log.info(
            "Replaying from checkpoint",
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
        )

        return await self.app.ainvoke(None, config)
