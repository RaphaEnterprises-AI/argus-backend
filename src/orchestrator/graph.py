"""LangGraph orchestrator for the testing agent."""

from typing import Literal
import structlog

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import TestingState, TestStatus
from ..config import Settings, get_settings

logger = structlog.get_logger()


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


class TestingOrchestrator:
    """
    Main orchestrator for E2E testing.
    
    Usage:
        orchestrator = TestingOrchestrator(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000"
        )
        results = await orchestrator.run()
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
        
        # Compile with memory for checkpointing
        self.memory = MemorySaver()
        self.app = graph.compile(checkpointer=self.memory)
        
        self.log = logger.bind(
            codebase=codebase_path,
            app_url=app_url,
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
