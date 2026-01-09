"""LangGraph orchestrator for the testing agent.

Supports human-in-the-loop breakpoints for approval workflows:
- interrupt_before self_heal node to require healing approval
- interrupt_before plan_tests node for test plan approval

When breakpoints are enabled, execution will pause before the specified nodes,
allowing external systems to approve/reject/modify state before resuming.
"""

from typing import Literal, Optional
import structlog

from langgraph.graph import StateGraph, END

from .state import TestingState, TestStatus
from .checkpointer import get_checkpointer
from ..config import Settings, get_settings

logger = structlog.get_logger()


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
        plan_tests_node,
        execute_test_node,
        self_heal_node,
        report_node,
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

    async def resume(self, thread_id: str, state_updates: Optional[dict] = None) -> dict:
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

    async def get_state(self, thread_id: str) -> Optional[dict]:
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
