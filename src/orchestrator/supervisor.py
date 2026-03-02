"""Multi-agent Supervisor pattern for test orchestration.

Uses LangGraph's Supervisor pattern where a supervisor agent routes tasks
to specialized worker agents based on the current need.

The supervisor coordinates:
- CodeAnalyzer: Analyzes codebase structure and identifies test surfaces
- TestPlanner: Creates comprehensive test plans from analysis
- UITester: Executes browser-based UI tests using Computer Use + Playwright
- APITester: Tests REST/GraphQL APIs with schema validation
- SelfHealer: Analyzes test failures and fixes selectors/assertions
- Reporter: Generates human-readable reports and notifications

Advanced agents (2026 patterns):
- SREAgent: Incident correlation, runbook execution, MTTR tracking
- CorrectiveRAG: Self-correcting knowledge retrieval for context
- ToolDiscovery: Dynamic tool creation from API documentation

Example usage:
    from src.orchestrator.supervisor import create_supervisor_graph, SupervisorState

    # Create and compile the graph
    graph = create_supervisor_graph()
    app = graph.compile(checkpointer=get_checkpointer())

    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content="Run E2E tests for the app at http://localhost:3000")],
        "next_agent": None,
        "task_complete": False,
        "results": {},
        "current_phase": "analysis",
        "iteration": 0,
        "codebase_path": "/path/to/app",
        "app_url": "http://localhost:3000",
    }

    # Run the supervisor
    final_state = await app.ainvoke(initial_state, config)
"""

from typing import Annotated, TypedDict

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.config import get_settings
from src.core.model_registry import get_orchestrator_api_model_id
from src.orchestrator.langfuse_integration import get_langfuse_handler, flush_langfuse, score_trace

logger = structlog.get_logger()


class SupervisorState(TypedDict):
    """State for the multi-agent supervisor.

    Attributes:
        messages: Conversation messages between supervisor and agents
        next_agent: The agent to route to next (None if finished)
        task_complete: Whether the overall task is complete
        results: Accumulated results from all agents
        current_phase: Current testing phase (analysis, planning, execution, healing, reporting)
        iteration: Number of supervisor iterations
        codebase_path: Path to the codebase being tested
        app_url: URL of the application being tested
        pr_number: Optional PR number for GitHub integration
        changed_files: Optional list of changed files to focus on
        test_plan: Generated test plan
        test_results: Results from test execution
        failures: List of test failures
        passed_count: Number of passed tests
        failed_count: Number of failed tests
        total_cost: Accumulated API cost
        error: Error message if something went wrong
    """
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str | None
    task_complete: bool
    results: dict
    current_phase: str  # analysis, planning, execution, healing, reporting
    iteration: int

    # Testing context
    codebase_path: str | None
    app_url: str | None
    pr_number: int | None
    changed_files: list[str] | None

    # Test state (shared with worker nodes)
    codebase_summary: str | None
    testable_surfaces: list[dict] | None
    test_plan: list[dict] | None
    test_results: list[dict] | None
    failures: list[dict] | None
    healing_queue: list[str] | None

    # Metrics
    passed_count: int
    failed_count: int
    skipped_count: int
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int

    # Error handling
    error: str | None

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


# Available agents and their capabilities
AGENTS = [
    "code_analyzer",
    "test_planner",
    "ui_tester",
    "api_tester",
    "self_healer",
    "reporter",
    # Advanced agents (2026 patterns)
    "sre_agent",
    "corrective_rag",
    "tool_discovery",
    # Penetration testing
    "pentest_coordinator",
    # QA Intelligence agents
    "testgen_agent",
    "testhealer_cicd",
    "qa_engineer",
    # Discovery
    "auto_discovery",
]

AGENT_DESCRIPTIONS = {
    "code_analyzer": "Analyzes codebase structure, finds test surfaces, understands architecture. Use at the start to understand what to test.",
    "test_planner": "Creates comprehensive test plans from analysis, prioritizes tests by risk and importance. Use after code analysis.",
    "ui_tester": "Executes browser-based UI tests using Computer Use + Playwright. Handles user interactions, form submissions, navigation.",
    "api_tester": "Tests REST/GraphQL APIs with schema validation. Handles HTTP requests, response validation, authentication flows.",
    "self_healer": "Analyzes test failures and fixes selectors/assertions. Use when tests fail due to UI changes, not real bugs.",
    "reporter": "Generates human-readable reports, creates GitHub PR comments, sends Slack notifications. Use at the end.",
    # Advanced agents (2026 patterns)
    "sre_agent": "Handles incident correlation, alert triage, runbook execution, infrastructure healing, and MTTR tracking. Use for production issues, infrastructure problems, or when correlating failures across systems.",
    "corrective_rag": "Self-correcting retrieval agent that finds relevant documentation, code context, and historical patterns. Use when you need accurate context from knowledge bases, when initial retrieval seems incomplete, or for complex queries requiring multiple sources.",
    "tool_discovery": "Discovers and creates tools dynamically from API documentation (OpenAPI, GraphQL, MCP). Use when existing tools are insufficient, when integrating new APIs, or when capability gaps are identified.",
    "pentest_coordinator": "AI-driven penetration testing coordinator. Runs recon, vulnerability scanning (Nuclei + SQLMap + AI), and controlled exploitation in Docker sandboxes. Use when the user requests security testing, penetration testing, or vulnerability assessment of a web application.",
    # QA Intelligence agents
    "testgen_agent": "Generates test suites from requirements (Jira/Figma/text), codebase analysis, PR diffs, or natural language. Use when new tests need to be created, when coverage gaps are identified, or when a PR needs test suggestions.",
    "testhealer_cicd": "CI/CD-aware healing orchestration. Processes CI failures, runs proactive scans, computes health scores, posts fix suggestions to PRs. Use for automated test maintenance, CI failure analysis, or proactive fragility detection.",
    "qa_engineer": "Autonomous QA engineer. Analyzes repo coverage, discovers user flows, identifies gaps with risk scoring, generates missing tests, computes quality scores. Use for quality audits, PR reviews requiring coverage analysis, or when comprehensive testing assessment is needed.",
    # Discovery
    "auto_discovery": "Crawls a web application to discover pages, forms, user flows, and API endpoints. Use when app_url is set and discovery_complete is False. Returns batches of 3-5 pages per invocation. Call repeatedly until discovery_complete is True.",
}

PHASE_DESCRIPTIONS = {
    "analysis": "Understanding the codebase and identifying what to test",
    "planning": "Creating a comprehensive test plan based on analysis",
    "execution": "Running UI and API tests against the application",
    "healing": "Analyzing failures and attempting to fix broken tests",
    "reporting": "Generating final reports and notifications",
    # Advanced phases
    "incident_response": "Correlating failures, triaging alerts, executing runbooks for infrastructure issues",
    "knowledge_retrieval": "Finding relevant documentation and historical patterns for context",
    "tool_creation": "Discovering or creating new tools to fill capability gaps",
    # QA Intelligence phases
    "test_generation": "Generating new test cases from requirements or coverage gaps",
    "healing_scan": "Proactively scanning and healing fragile tests",
    "quality_audit": "Analyzing test coverage and computing quality scores",
    "discovery": "Crawling the target application to discover pages, forms, flows, and testable surfaces",
}


def create_supervisor_prompt() -> str:
    """Create the supervisor system prompt."""
    agent_list = "\n".join([f"- {name}: {desc}" for name, desc in AGENT_DESCRIPTIONS.items()])
    phase_list = "\n".join([f"- {phase}: {desc}" for phase, desc in PHASE_DESCRIPTIONS.items()])

    return f'''You are the Argus Test Orchestrator Supervisor. You coordinate a team of specialized agents to perform comprehensive E2E testing.

Available Agents:
{agent_list}

Testing Phases:
{phase_list}

Your job is to:
1. Analyze the current state and determine which agent should work next
2. Route tasks to the most appropriate agent based on the current phase
3. Monitor progress and handle failures gracefully
4. Ensure tests are executed efficiently without redundant work
5. Decide when testing is complete

Workflow Guidelines:
- Start with code_analyzer to understand the codebase
- Then use test_planner to create a test plan
- Execute tests with ui_tester and/or api_tester based on test types
- If tests fail, consider using self_healer before giving up
- For infrastructure issues or incident correlation, use sre_agent
- When context is unclear or retrieval seems incomplete, use corrective_rag
- If tools are missing for a task, use tool_discovery to create them
- For QA discovery: alternate between auto_discovery (crawl batch) and testgen_agent (generate tests for batch)
- Always end with reporter to generate final reports

Decision Rules:
- If current_phase is "analysis" and no testable_surfaces exist, use code_analyzer
- If current_phase is "planning" and no test_plan exists, use test_planner
- If current_phase is "execution" and test_plan has items, use ui_tester or api_tester
- If failures exist and self-healing hasn't been tried, use self_healer
- If infrastructure issues, alerts, or incident correlation is needed, use sre_agent
- If you need documentation, historical patterns, or context retrieval, use corrective_rag
- If existing tools are insufficient or API integration is needed, use tool_discovery
- If app_url is set and discovery_complete is False and discovery_queue is empty, use auto_discovery
- If discovery_queue has items waiting to be processed, use testgen_agent
- If discovery_complete is True and discovery_queue is empty and generated_tests exist, use reporter
- If all tests are complete or an error occurred, use reporter

IMPORTANT: Respond with EXACTLY ONE of these options:
- One of the agent names: {', '.join(AGENTS)}
- FINISH (if all testing is complete)

Always explain your routing decision briefly (1-2 sentences) before stating your choice.'''


def create_initial_supervisor_state(
    codebase_path: str,
    app_url: str,
    pr_number: int | None = None,
    changed_files: list[str] | None = None,
    initial_message: str | None = None,
) -> SupervisorState:
    """Create initial state for the supervisor graph.

    Args:
        codebase_path: Path to the codebase to analyze
        app_url: URL of the application to test
        pr_number: Optional PR number for GitHub integration
        changed_files: Optional list of files to focus on
        initial_message: Optional initial message to the supervisor

    Returns:
        Initial SupervisorState
    """
    if initial_message is None:
        initial_message = f"Run comprehensive E2E tests for the application at {app_url}"
        if changed_files:
            initial_message += f" focusing on changes in: {', '.join(changed_files[:5])}"

    return SupervisorState(
        messages=[HumanMessage(content=initial_message)],
        next_agent=None,
        task_complete=False,
        results={},
        current_phase="analysis",
        iteration=0,
        codebase_path=codebase_path,
        app_url=app_url,
        pr_number=pr_number,
        changed_files=changed_files or [],
        codebase_summary=None,
        testable_surfaces=None,
        test_plan=None,
        test_results=[],
        failures=[],
        healing_queue=[],
        passed_count=0,
        failed_count=0,
        skipped_count=0,
        total_cost=0.0,
        total_input_tokens=0,
        total_output_tokens=0,
        error=None,
        # Discovery streaming defaults
        discovery_queue=[],
        discovery_complete=False,
        discovery_pages_found=0,
        discovery_batch_index=0,
        generated_tests=[],
        generated_requirements=[],
        org_id=None,
        project_id=None,
    )


async def supervisor_node(state: SupervisorState, config: RunnableConfig) -> dict:
    """Supervisor node that routes to appropriate agents.

    Analyzes the current state and decides which specialized agent
    should work next based on the testing phase and results.

    Args:
        state: Current supervisor state
        config: Configuration including thread_id

    Returns:
        State updates including next_agent selection
    """
    settings = get_settings()
    log = logger.bind(node="supervisor", iteration=state.get("iteration", 0))

    # Check for termination conditions
    max_iterations = 50  # Safety limit
    if state.get("iteration", 0) >= max_iterations:
        log.warning("Max iterations reached, forcing completion")
        return {
            "messages": [AIMessage(content="Max iterations reached. Moving to reporting phase.")],
            "next_agent": "reporter",
            "current_phase": "reporting",
            "iteration": state.get("iteration", 0) + 1,
        }

    # Check if we hit an error
    if state.get("error"):
        log.warning("Error detected, moving to reporting", error=state.get("error"))
        return {
            "messages": [AIMessage(content=f"Error detected: {state.get('error')}. Moving to reporting.")],
            "next_agent": "reporter",
            "current_phase": "reporting",
            "iteration": state.get("iteration", 0) + 1,
        }

    try:
        from langchain_anthropic import ChatAnthropic

        # Get API key
        api_key = settings.anthropic_api_key
        if api_key and hasattr(api_key, 'get_secret_value'):
            api_key = api_key.get_secret_value()

        llm = ChatAnthropic(
            model=get_orchestrator_api_model_id(),
            api_key=api_key,
            max_tokens=4096,
        )

        system_prompt = create_supervisor_prompt()

        # Build context about current state
        context_parts = [
            f"Current Phase: {state.get('current_phase', 'analysis')}",
            f"Iteration: {state.get('iteration', 0)}",
            f"Codebase Path: {state.get('codebase_path', 'Not set')}",
            f"App URL: {state.get('app_url', 'Not set')}",
        ]

        # Add analysis status
        if state.get("codebase_summary"):
            context_parts.append(f"Codebase Summary: {state['codebase_summary'][:200]}...")
            context_parts.append(f"Testable Surfaces Found: {len(state.get('testable_surfaces', []))}")
        else:
            context_parts.append("Codebase: Not yet analyzed")

        # Add planning status
        if state.get("test_plan"):
            context_parts.append(f"Test Plan: {len(state['test_plan'])} tests planned")
        else:
            context_parts.append("Test Plan: Not yet created")

        # Add execution status
        if state.get("test_results"):
            context_parts.append(f"Tests Executed: {len(state['test_results'])}")
            context_parts.append(f"Passed: {state.get('passed_count', 0)}, Failed: {state.get('failed_count', 0)}")

        # Add failure status
        if state.get("failures"):
            context_parts.append(f"Unhealed Failures: {len(state['failures'])}")
            context_parts.append(f"Healing Queue: {len(state.get('healing_queue', []))}")

        # Add cost tracking
        context_parts.append(f"Total Cost: ${state.get('total_cost', 0):.4f}")

        context_msg = "Current State:\n" + "\n".join(f"- {part}" for part in context_parts)
        context_msg += f"\n\nBased on this state, which agent should work next? Choose one of: {', '.join(AGENTS)} or FINISH"

        messages = [
            SystemMessage(content=system_prompt),
            *list(state.get("messages", [])),
            HumanMessage(content=context_msg),
        ]

        log.info("Invoking supervisor LLM", phase=state.get("current_phase"))
        response = await llm.ainvoke(messages)
        content = response.content.upper().strip()

        log.debug("Supervisor response", content=content[:200])

        # Parse the agent selection
        next_agent = None
        for agent in AGENTS:
            if agent.upper() in content:
                next_agent = agent
                break

        # Check for finish
        if "FINISH" in content:
            log.info("Supervisor decided to finish")
            return {
                "messages": [response],
                "next_agent": None,
                "task_complete": True,
                "current_phase": "complete",
                "iteration": state.get("iteration", 0) + 1,
            }

        if next_agent is None:
            # Fallback logic based on phase
            log.warning("Could not parse agent selection, using fallback logic")
            phase = state.get("current_phase", "analysis")

            if phase == "analysis" and not state.get("testable_surfaces"):
                next_agent = "code_analyzer"
            elif phase == "planning" and not state.get("test_plan"):
                next_agent = "test_planner"
            elif phase == "execution" and state.get("test_plan"):
                # Check if we have tests of different types
                test_plan = state.get("test_plan", [])
                if any(t.get("type") == "api" for t in test_plan):
                    next_agent = "api_tester"
                else:
                    next_agent = "ui_tester"
            elif state.get("failures") and state.get("healing_queue"):
                next_agent = "self_healer"
            else:
                next_agent = "reporter"

        # Update phase based on agent
        new_phase = state.get("current_phase", "analysis")
        if next_agent == "code_analyzer":
            new_phase = "analysis"
        elif next_agent == "test_planner":
            new_phase = "planning"
        elif next_agent in ["ui_tester", "api_tester"]:
            new_phase = "execution"
        elif next_agent == "self_healer":
            new_phase = "healing"
        elif next_agent == "reporter":
            new_phase = "reporting"
        # Advanced agent phases
        elif next_agent == "sre_agent":
            new_phase = "incident_response"
        elif next_agent == "corrective_rag":
            new_phase = "knowledge_retrieval"
        elif next_agent == "tool_discovery":
            new_phase = "tool_creation"
        elif next_agent == "testgen_agent":
            new_phase = "test_generation"
        elif next_agent == "testhealer_cicd":
            new_phase = "healing_scan"
        elif next_agent == "qa_engineer":
            new_phase = "quality_audit"
        elif next_agent == "auto_discovery":
            new_phase = "discovery"

        log.info("Supervisor routing", next_agent=next_agent, new_phase=new_phase)

        return {
            "messages": [response],
            "next_agent": next_agent,
            "current_phase": new_phase,
            "iteration": state.get("iteration", 0) + 1,
        }

    except Exception as e:
        log.error("Supervisor failed", error=str(e))
        return {
            "messages": [AIMessage(content=f"Supervisor error: {str(e)}. Moving to reporting.")],
            "next_agent": "reporter",
            "current_phase": "reporting",
            "error": str(e),
            "iteration": state.get("iteration", 0) + 1,
        }


def route_to_agent(state: SupervisorState) -> str:
    """Route to the selected agent or end.

    Args:
        state: Current supervisor state

    Returns:
        Name of the next node or "end"
    """
    if state.get("task_complete") or state.get("next_agent") is None:
        return "end"
    return state["next_agent"]


# Wrapper functions to adapt existing nodes for supervisor state
async def supervisor_code_analyzer_node(state: SupervisorState) -> dict:
    """Wrapper for code analyzer that works with supervisor state."""
    from src.orchestrator.nodes import analyze_code_node
    from src.orchestrator.state import create_initial_state

    log = logger.bind(node="supervisor_code_analyzer")
    log.info("Running code analyzer")

    # Create a TestingState from supervisor state
    testing_state = create_initial_state(
        codebase_path=state.get("codebase_path", ""),
        app_url=state.get("app_url", ""),
        pr_number=state.get("pr_number"),
        changed_files=state.get("changed_files", []),
    )

    # Run the actual node
    result = await analyze_code_node(testing_state)

    # Extract relevant results back to supervisor state
    summary = f"Analyzed codebase. Found {len(result.get('testable_surfaces', []))} testable surfaces."
    if result.get("codebase_summary"):
        summary = result["codebase_summary"][:500]

    return {
        "messages": [AIMessage(content=f"Code analysis complete: {summary}")],
        "codebase_summary": result.get("codebase_summary"),
        "testable_surfaces": result.get("testable_surfaces", []),
        "total_cost": state.get("total_cost", 0) + result.get("total_cost", 0),
        "total_input_tokens": state.get("total_input_tokens", 0) + result.get("total_input_tokens", 0),
        "total_output_tokens": state.get("total_output_tokens", 0) + result.get("total_output_tokens", 0),
        "current_phase": "planning",  # Move to next phase
        "results": {
            **state.get("results", {}),
            "code_analysis": {
                "summary": result.get("codebase_summary"),
                "surfaces_count": len(result.get("testable_surfaces", [])),
            }
        },
    }


async def supervisor_test_planner_node(state: SupervisorState) -> dict:
    """Wrapper for test planner that works with supervisor state."""
    from src.orchestrator.nodes import plan_tests_node
    from src.orchestrator.state import create_initial_state

    log = logger.bind(node="supervisor_test_planner")
    log.info("Running test planner")

    # Create a TestingState from supervisor state
    testing_state = create_initial_state(
        codebase_path=state.get("codebase_path", ""),
        app_url=state.get("app_url", ""),
    )
    testing_state["codebase_summary"] = state.get("codebase_summary", "")
    testing_state["testable_surfaces"] = state.get("testable_surfaces", [])
    testing_state["changed_files"] = state.get("changed_files", [])

    # Run the actual node
    result = await plan_tests_node(testing_state)

    test_plan = result.get("test_plan", [])
    summary = f"Created test plan with {len(test_plan)} tests."

    return {
        "messages": [AIMessage(content=f"Test planning complete: {summary}")],
        "test_plan": test_plan,
        "total_cost": state.get("total_cost", 0) + result.get("total_cost", 0),
        "total_input_tokens": state.get("total_input_tokens", 0) + result.get("total_input_tokens", 0),
        "total_output_tokens": state.get("total_output_tokens", 0) + result.get("total_output_tokens", 0),
        "current_phase": "execution",  # Move to next phase
        "results": {
            **state.get("results", {}),
            "test_planning": {
                "tests_count": len(test_plan),
                "test_types": list(set(t.get("type", "unknown") for t in test_plan)),
            }
        },
    }


async def supervisor_execute_test_node(state: SupervisorState) -> dict:
    """Wrapper for test executor that works with supervisor state."""
    from src.orchestrator.nodes import execute_test_node
    from src.orchestrator.state import create_initial_state

    log = logger.bind(node="supervisor_execute_test")

    # Create a TestingState from supervisor state
    testing_state = create_initial_state(
        codebase_path=state.get("codebase_path", ""),
        app_url=state.get("app_url", ""),
    )
    testing_state["test_plan"] = state.get("test_plan", [])
    testing_state["test_results"] = state.get("test_results", [])
    testing_state["failures"] = state.get("failures", [])
    testing_state["healing_queue"] = state.get("healing_queue", [])
    testing_state["passed_count"] = state.get("passed_count", 0)
    testing_state["failed_count"] = state.get("failed_count", 0)
    testing_state["current_test_index"] = len(state.get("test_results", []))

    # Run a batch of tests (up to 5 at a time)
    batch_size = 5
    test_results = list(state.get("test_results", []))
    failures = list(state.get("failures", []))
    healing_queue = list(state.get("healing_queue", []))
    passed_count = state.get("passed_count", 0)
    failed_count = state.get("failed_count", 0)

    test_plan = state.get("test_plan", [])
    start_idx = len(test_results)

    log.info("Executing test batch", start=start_idx, total=len(test_plan))

    for i in range(min(batch_size, len(test_plan) - start_idx)):
        testing_state["current_test_index"] = start_idx + i
        result = await execute_test_node(testing_state)

        # Extract results
        if result.get("test_results"):
            new_result = result["test_results"][-1]
            test_results.append(new_result)

            if new_result.get("status") == "passed":
                passed_count += 1
            else:
                failed_count += 1
                if result.get("failures"):
                    failures.extend(result["failures"][-1:])
                if result.get("healing_queue"):
                    healing_queue.extend(result["healing_queue"][-1:])

        # Update for next iteration
        testing_state["test_results"] = test_results
        testing_state["failures"] = failures
        testing_state["healing_queue"] = healing_queue
        testing_state["passed_count"] = passed_count
        testing_state["failed_count"] = failed_count

    # Check if all tests are complete
    all_complete = len(test_results) >= len(test_plan)
    next_phase = "reporting" if all_complete else "execution"
    if healing_queue and not all_complete:
        next_phase = "healing"

    summary = f"Executed {len(test_results)}/{len(test_plan)} tests. Passed: {passed_count}, Failed: {failed_count}"

    return {
        "messages": [AIMessage(content=f"Test execution update: {summary}")],
        "test_results": test_results,
        "failures": failures,
        "healing_queue": healing_queue,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "current_phase": next_phase,
        "total_cost": state.get("total_cost", 0) + result.get("total_cost", 0),
        "results": {
            **state.get("results", {}),
            "test_execution": {
                "total": len(test_plan),
                "executed": len(test_results),
                "passed": passed_count,
                "failed": failed_count,
            }
        },
    }


async def supervisor_self_healer_node(state: SupervisorState) -> dict:
    """Wrapper for self-healer that works with supervisor state."""
    from src.orchestrator.nodes import self_heal_node
    from src.orchestrator.state import create_initial_state

    log = logger.bind(node="supervisor_self_healer")
    log.info("Running self-healer", healing_queue=len(state.get("healing_queue", [])))

    # Create a TestingState from supervisor state
    testing_state = create_initial_state(
        codebase_path=state.get("codebase_path", ""),
        app_url=state.get("app_url", ""),
    )
    testing_state["test_plan"] = state.get("test_plan", [])
    testing_state["test_results"] = state.get("test_results", [])
    testing_state["failures"] = state.get("failures", [])
    testing_state["healing_queue"] = state.get("healing_queue", [])

    # Run the actual node
    result = await self_heal_node(testing_state)

    healed_count = len(state.get("healing_queue", [])) - len(result.get("healing_queue", []))
    summary = f"Healing attempt complete. Healed {healed_count} tests."

    # Determine next phase
    remaining_tests = len(state.get("test_plan", [])) - len(state.get("test_results", []))
    next_phase = "execution" if remaining_tests > 0 else "reporting"

    return {
        "messages": [AIMessage(content=summary)],
        "test_plan": result.get("test_plan", state.get("test_plan", [])),
        "failures": result.get("failures", []),
        "healing_queue": result.get("healing_queue", []),
        "total_cost": state.get("total_cost", 0) + result.get("total_cost", 0),
        "current_phase": next_phase,
        "results": {
            **state.get("results", {}),
            "self_healing": {
                "attempted": len(state.get("healing_queue", [])),
                "healed": healed_count,
            }
        },
    }


async def supervisor_reporter_node(state: SupervisorState) -> dict:
    """Wrapper for reporter that works with supervisor state."""
    from src.orchestrator.nodes import report_node
    from src.orchestrator.state import create_initial_state

    log = logger.bind(node="supervisor_reporter")
    log.info("Running reporter")

    # Create a TestingState from supervisor state
    testing_state = create_initial_state(
        codebase_path=state.get("codebase_path", ""),
        app_url=state.get("app_url", ""),
        pr_number=state.get("pr_number"),
    )
    testing_state["test_results"] = state.get("test_results", [])
    testing_state["failures"] = state.get("failures", [])
    testing_state["passed_count"] = state.get("passed_count", 0)
    testing_state["failed_count"] = state.get("failed_count", 0)
    testing_state["skipped_count"] = state.get("skipped_count", 0)
    testing_state["total_cost"] = state.get("total_cost", 0)
    testing_state["iteration"] = state.get("iteration", 0)

    # Run the actual node
    result = await report_node(testing_state)

    summary = result.get("executive_summary", "Report generation complete.")

    return {
        "messages": [AIMessage(content=f"Final Report: {summary}")],
        "current_phase": "complete",
        "task_complete": True,
        "total_cost": state.get("total_cost", 0) + result.get("total_cost", 0),
        "results": {
            **state.get("results", {}),
            "report": {
                "summary": summary,
                "report_paths": result.get("report_paths", {}),
            }
        },
    }


# =============================================================================
# Advanced Agent Nodes (2026 Patterns)
# =============================================================================


async def supervisor_sre_agent_node(state: SupervisorState) -> dict:
    """Wrapper for SRE Agent that works with supervisor state.

    Handles incident correlation, alert triage, runbook execution,
    infrastructure healing, and MTTR tracking.
    """
    from src.agents.sre_agent import SREAgent

    log = logger.bind(node="supervisor_sre_agent")
    log.info("Running SRE agent")

    try:
        agent = SREAgent()

        # Determine what SRE operation to perform based on state
        failures = state.get("failures", [])
        error = state.get("error")

        if failures:
            # Correlate test failures with infrastructure signals
            correlation_result = await agent.correlate_signals(
                signals=[
                    {"type": "test_failure", "data": f} for f in failures[:5]
                ],
                time_window_minutes=30,
            )

            if correlation_result.get("incidents"):
                incidents = correlation_result["incidents"]
                summary = f"Correlated {len(incidents)} potential incidents from {len(failures)} test failures."

                # Check if runbook execution is needed
                runbook_results = []
                for incident in incidents[:2]:  # Limit to 2 runbooks
                    if incident.get("suggested_runbook"):
                        result = await agent.execute_runbook(
                            runbook_id=incident["suggested_runbook"],
                            dry_run=True,  # Safety: dry run first
                        )
                        runbook_results.append(result)

                return {
                    "messages": [AIMessage(content=f"SRE Analysis: {summary}")],
                    "current_phase": "healing",  # Move back to healing with new context
                    "results": {
                        **state.get("results", {}),
                        "sre_analysis": {
                            "incidents": len(incidents),
                            "correlation": correlation_result,
                            "runbooks_checked": len(runbook_results),
                        }
                    },
                }
            else:
                summary = "No infrastructure incidents detected. Failures likely application-level."
        elif error:
            # Analyze error for potential infrastructure issues
            summary = f"Analyzed error: {error[:100]}. Checking for infrastructure correlation."
        else:
            summary = "No failures or errors to analyze. SRE agent check complete."

        return {
            "messages": [AIMessage(content=f"SRE Analysis: {summary}")],
            "current_phase": state.get("current_phase", "execution"),
            "results": {
                **state.get("results", {}),
                "sre_analysis": {"summary": summary}
            },
        }

    except Exception as e:
        log.error("SRE agent failed", error=str(e))
        return {
            "messages": [AIMessage(content=f"SRE agent error: {str(e)}")],
            "current_phase": state.get("current_phase", "execution"),
        }


async def supervisor_corrective_rag_node(state: SupervisorState) -> dict:
    """Wrapper for Corrective RAG Agent that works with supervisor state.

    Provides self-correcting knowledge retrieval for documentation,
    code context, and historical patterns.
    """
    from src.agents.corrective_rag_agent import CorrectiveRAGAgent

    log = logger.bind(node="supervisor_corrective_rag")
    log.info("Running Corrective RAG agent")

    try:
        agent = CorrectiveRAGAgent()

        # Build query from current context
        query_parts = []

        if state.get("failures"):
            failure_types = list(set(f.get("type", "unknown") for f in state["failures"][:5]))
            query_parts.append(f"test failures: {', '.join(failure_types)}")

        if state.get("error"):
            query_parts.append(f"error: {state['error'][:200]}")

        if state.get("testable_surfaces"):
            surfaces = [s.get("name", "unknown") for s in state["testable_surfaces"][:3]]
            query_parts.append(f"testing surfaces: {', '.join(surfaces)}")

        if not query_parts:
            query_parts.append("E2E testing best practices and patterns")

        query = "Find relevant documentation and patterns for: " + "; ".join(query_parts)

        # Execute CRAG query
        result = await agent.query(
            query=query,
            org_id=state.get("org_id", "default"),
            project_id=state.get("project_id", "default"),
        )

        if result.success and result.data:
            documents = result.data.get("documents", [])
            confidence = result.data.get("confidence", 0)

            summary = f"Retrieved {len(documents)} relevant documents (confidence: {confidence:.2f})"

            # Extract key insights
            insights = []
            for doc in documents[:3]:
                if doc.get("summary"):
                    insights.append(doc["summary"][:100])

            return {
                "messages": [AIMessage(content=f"Knowledge Retrieval: {summary}\nKey insights: {'; '.join(insights)}")],
                "current_phase": state.get("current_phase", "analysis"),
                "results": {
                    **state.get("results", {}),
                    "knowledge_retrieval": {
                        "documents_found": len(documents),
                        "confidence": confidence,
                        "query": query,
                    }
                },
            }
        else:
            summary = "No highly relevant documents found. Continuing with available context."

        return {
            "messages": [AIMessage(content=f"Knowledge Retrieval: {summary}")],
            "current_phase": state.get("current_phase", "analysis"),
        }

    except Exception as e:
        log.error("Corrective RAG agent failed", error=str(e))
        return {
            "messages": [AIMessage(content=f"Knowledge retrieval error: {str(e)}")],
            "current_phase": state.get("current_phase", "analysis"),
        }


async def supervisor_tool_discovery_node(state: SupervisorState) -> dict:
    """Wrapper for Tool Discovery Agent that works with supervisor state.

    Discovers and creates tools dynamically from API documentation
    when existing tools are insufficient.
    """
    from src.agents.tool_discovery_agent import ToolDiscoveryAgent, ToolCreatorAgent

    log = logger.bind(node="supervisor_tool_discovery")
    log.info("Running Tool Discovery agent")

    try:
        discovery_agent = ToolDiscoveryAgent()

        # Check if we have API documentation to process
        app_url = state.get("app_url", "")
        tools_discovered = []

        # Try to discover tools from common API documentation paths
        doc_paths = [
            f"{app_url}/openapi.json",
            f"{app_url}/api/openapi.json",
            f"{app_url}/swagger.json",
            f"{app_url}/api/docs",
        ]

        for doc_url in doc_paths[:2]:  # Limit attempts
            try:
                result = await discovery_agent.discover_tools_from_docs(
                    doc_url=doc_url,
                    org_id=state.get("org_id", "default"),
                    project_id=state.get("project_id", "default"),
                )
                if result.success and result.data:
                    tools_discovered.extend(result.data)
                    break  # Found docs, stop searching
            except Exception:
                continue  # Try next path

        if tools_discovered:
            summary = f"Discovered {len(tools_discovered)} tools from API documentation."

            # Verify and integrate top tools
            verified_count = 0
            for tool in tools_discovered[:5]:
                try:
                    verification = await discovery_agent.verify_tool(tool)
                    if verification.get("is_valid"):
                        await discovery_agent.integrate_tool(
                            tool,
                            org_id=state.get("org_id", "default"),
                        )
                        verified_count += 1
                except Exception:
                    continue

            summary += f" Verified and integrated {verified_count} tools."

            return {
                "messages": [AIMessage(content=f"Tool Discovery: {summary}")],
                "current_phase": "planning",  # Move to planning with new tools
                "results": {
                    **state.get("results", {}),
                    "tool_discovery": {
                        "tools_discovered": len(tools_discovered),
                        "tools_integrated": verified_count,
                    }
                },
            }
        else:
            # No tools discovered, try creating if there's a capability gap
            creator = ToolCreatorAgent()

            # Check for capability gaps based on test plan
            test_plan = state.get("test_plan", [])
            if test_plan:
                # Identify potential gaps
                test_types = set(t.get("type", "unknown") for t in test_plan)
                summary = f"No API documentation found. Analyzed {len(test_types)} test types for capability gaps."
            else:
                summary = "No API documentation found and no test plan to analyze for gaps."

        return {
            "messages": [AIMessage(content=f"Tool Discovery: {summary}")],
            "current_phase": state.get("current_phase", "analysis"),
        }

    except Exception as e:
        log.error("Tool Discovery agent failed", error=str(e))
        return {
            "messages": [AIMessage(content=f"Tool discovery error: {str(e)}")],
            "current_phase": state.get("current_phase", "analysis"),
        }


async def supervisor_pentest_coordinator_node(state: SupervisorState) -> dict:
    """Wrapper for pentest coordinator that works with supervisor state.

    Runs the full pentest pipeline: recon → vuln scan → exploitation → report.
    """
    from src.orchestrator.pentest_graph import (
        create_initial_pentest_state,
        create_pentest_graph,
    )

    log = logger.bind(node="supervisor_pentest_coordinator")
    log.info("Running Pentest Coordinator")

    try:
        app_url = state.get("app_url", "")
        if not app_url:
            return {
                "messages": [AIMessage(content="Pentest Coordinator: No app_url provided, cannot run pentest.")],
                "current_phase": state.get("current_phase", "analysis"),
            }

        pentest_state = create_initial_pentest_state(
            target_url=app_url,
            scope=[app_url],
            scan_profile="standard",
            organization_id=state.get("org_id", "default"),
            project_id=state.get("project_id", "default"),
            enable_exploitation=False,  # Disabled by default in supervisor
        )

        graph = create_pentest_graph()
        compiled = graph.compile()
        final_state = await compiled.ainvoke(pentest_state)

        report = final_state.get("pentest_report", {})
        total = report.get("total_findings", 0)
        critical = report.get("critical_count", 0)
        summary = report.get("executive_summary", "Scan completed.")

        return {
            "messages": [AIMessage(content=f"Pentest Coordinator: {summary}")],
            "current_phase": "reporting",
            "results": {
                **state.get("results", {}),
                "pentest": {
                    "total_findings": total,
                    "critical_count": critical,
                    "risk_score": report.get("risk_score", 0),
                    "status": final_state.get("status", "completed"),
                },
            },
        }

    except Exception as e:
        log.error("Pentest Coordinator failed", error=str(e))
        return {
            "messages": [AIMessage(content=f"Pentest error: {str(e)}")],
            "current_phase": state.get("current_phase", "analysis"),
        }


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


async def supervisor_testgen_agent_node(state: SupervisorState) -> dict:
    """Wrapper for TestGen Agent that works with supervisor state.

    Handles three source types in priority order:
    1. discovery_queue (from auto_discovery batches)
    2. changed_files (from PR analysis)
    3. testable_surfaces (from code analysis)
    """
    from src.agents.testgen_agent import TestGenAgent, TestGenConfig

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

        # Extract results from either object or dict response
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


async def supervisor_testhealer_cicd_node(state: SupervisorState) -> dict:
    """Wrapper for TestHealer CI/CD Agent that works with supervisor state."""
    from src.agents.testhealer_cicd_agent import TestHealerCICDAgent

    log = logger.bind(node="supervisor_testhealer_cicd")
    log.info("Running TestHealer CI/CD agent")

    try:
        agent = TestHealerCICDAgent()

        # Determine scan type from context
        scan_type = "manual"
        trigger_context = {}

        if state.get("failures"):
            scan_type = "ci_failure"
            trigger_context = {
                "failures": [f.get("error_message", "")[:200] for f in state.get("failures", [])[:5]],
                "failed_count": state.get("failed_count", 0),
            }
        else:
            scan_type = "proactive"

        result = await agent.scan(
            org_id=state.get("org_id", "default"),
            project_id=state.get("project_id", "default"),
            scan_type=scan_type,
            trigger_context=trigger_context,
        )

        fixes = result.get("fixes_proposed", 0) if isinstance(result, dict) else 0
        health = result.get("health_score_after", 0) if isinstance(result, dict) else 0
        summary = f"Healing scan ({scan_type}): {fixes} fixes proposed, health score: {health:.0f}"

        return {
            "messages": [AIMessage(content=f"TestHealer CI/CD: {summary}")],
            "current_phase": "healing" if fixes > 0 else state.get("current_phase", "execution"),
            "results": {
                **state.get("results", {}),
                "healing_scan": {
                    "scan_type": scan_type,
                    "fixes_proposed": fixes,
                    "health_score": health,
                }
            },
        }
    except Exception as e:
        log.error("TestHealer CI/CD agent failed", error=str(e))
        return {
            "messages": [AIMessage(content=f"TestHealer CI/CD error: {str(e)}")],
            "current_phase": state.get("current_phase", "execution"),
        }


async def supervisor_qa_engineer_node(state: SupervisorState) -> dict:
    """Wrapper for QA Engineer Agent that works with supervisor state."""
    from src.agents.qa_engineer_agent import QAEngineerAgent

    log = logger.bind(node="supervisor_qa_engineer")
    log.info("Running QA Engineer agent")

    try:
        agent = QAEngineerAgent()

        # Determine analysis type
        analysis_type = "full"
        trigger_context = {}

        if state.get("pr_number"):
            analysis_type = "pr_review"
            trigger_context = {
                "pr_number": state.get("pr_number"),
                "changed_files": state.get("changed_files", []),
            }

        result = await agent.analyze(
            org_id=state.get("org_id", "default"),
            project_id=state.get("project_id", "default"),
            analysis_type=analysis_type,
            trigger_context=trigger_context,
        )

        quality = result.get("quality_score", 0) if isinstance(result, dict) else 0
        gaps = result.get("gap_count", 0) if isinstance(result, dict) else 0
        summary = f"QA analysis ({analysis_type}): quality score {quality:.0f}/100, {gaps} coverage gaps"

        return {
            "messages": [AIMessage(content=f"QA Engineer: {summary}")],
            "current_phase": "test_generation" if gaps > 5 else "reporting",
            "results": {
                **state.get("results", {}),
                "qa_analysis": {
                    "quality_score": quality,
                    "gap_count": gaps,
                    "analysis_type": analysis_type,
                }
            },
        }
    except Exception as e:
        log.error("QA Engineer agent failed", error=str(e))
        return {
            "messages": [AIMessage(content=f"QA Engineer error: {str(e)}")],
            "current_phase": state.get("current_phase", "analysis"),
        }


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

        # Track already-discovered URLs to avoid duplicates
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

        # Done if this batch found fewer pages than BATCH_SIZE (exhausted)
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


def create_supervisor_graph() -> StateGraph:
    """Create the multi-agent supervisor graph.

    The graph follows the Supervisor pattern where a central supervisor
    routes tasks to specialized worker agents based on the current state.

    Graph structure:

        [START]
           |
           v
      +-----------+
      | supervisor|<---------+
      +-----------+          |
           |                 |
           v                 |
      (route_to_agent)       |
           |                 |
     +-----+-----+-----+     |
     |     |     |     |     |
     v     v     v     v     |
    [CA]  [TP]  [UT]  [SH]   |
     |     |     |     |     |
     +-----+-----+-----+     |
           |                 |
           +-----------------+
           |
           v (when done)
         [END]

    Where:
    - CA = code_analyzer
    - TP = test_planner
    - UT = ui_tester / api_tester
    - SH = self_healer
    - RP = reporter

    Returns:
        Compiled StateGraph
    """
    graph = StateGraph(SupervisorState)

    # Add supervisor node
    graph.add_node("supervisor", supervisor_node)

    # Add worker agent nodes (wrapped for supervisor state)
    graph.add_node("code_analyzer", supervisor_code_analyzer_node)
    graph.add_node("test_planner", supervisor_test_planner_node)
    graph.add_node("ui_tester", supervisor_execute_test_node)
    graph.add_node("api_tester", supervisor_execute_test_node)  # Same handler for now
    graph.add_node("self_healer", supervisor_self_healer_node)
    graph.add_node("reporter", supervisor_reporter_node)

    # Advanced agent nodes (2026 patterns)
    graph.add_node("sre_agent", supervisor_sre_agent_node)
    graph.add_node("corrective_rag", supervisor_corrective_rag_node)
    graph.add_node("tool_discovery", supervisor_tool_discovery_node)

    # Penetration testing
    graph.add_node("pentest_coordinator", supervisor_pentest_coordinator_node)

    # QA Intelligence agent nodes
    graph.add_node("testgen_agent", supervisor_testgen_agent_node)
    graph.add_node("testhealer_cicd", supervisor_testhealer_cicd_node)
    graph.add_node("qa_engineer", supervisor_qa_engineer_node)

    # Discovery
    graph.add_node("auto_discovery", supervisor_auto_discovery_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Conditional routing from supervisor
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "code_analyzer": "code_analyzer",
            "test_planner": "test_planner",
            "ui_tester": "ui_tester",
            "api_tester": "api_tester",
            "self_healer": "self_healer",
            "reporter": "reporter",
            # Advanced agents
            "sre_agent": "sre_agent",
            "corrective_rag": "corrective_rag",
            "tool_discovery": "tool_discovery",
            # Penetration testing
            "pentest_coordinator": "pentest_coordinator",
            # QA Intelligence agents
            "testgen_agent": "testgen_agent",
            "testhealer_cicd": "testhealer_cicd",
            "qa_engineer": "qa_engineer",
            # Discovery
            "auto_discovery": "auto_discovery",
            "end": END,
        }
    )

    # All agents return to supervisor
    for agent in AGENTS:
        graph.add_edge(agent, "supervisor")

    return graph


class SupervisorOrchestrator:
    """
    Multi-agent supervisor orchestrator for E2E testing.

    Uses the LangGraph Supervisor pattern to coordinate specialized agents:
    - CodeAnalyzer: Understands codebase structure
    - TestPlanner: Creates comprehensive test plans
    - UITester: Executes browser-based tests
    - APITester: Tests API endpoints
    - SelfHealer: Fixes broken tests
    - Reporter: Generates reports

    Example usage:
        orchestrator = SupervisorOrchestrator(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000"
        )
        results = await orchestrator.run()
    """

    def __init__(
        self,
        codebase_path: str,
        app_url: str,
        pr_number: int | None = None,
        changed_files: list[str] | None = None,
        # Langfuse tracing options
        langfuse_user_id: str | None = None,
        langfuse_tags: list[str] | None = None,
        langfuse_metadata: dict | None = None,
    ):
        self.codebase_path = codebase_path
        self.app_url = app_url
        self.pr_number = pr_number
        self.changed_files = changed_files or []

        # Langfuse tracing configuration
        self.langfuse_user_id = langfuse_user_id
        self.langfuse_tags = langfuse_tags or []
        self.langfuse_metadata = langfuse_metadata or {}

        # Create and compile graph
        from src.orchestrator.checkpointer import get_checkpointer

        graph = create_supervisor_graph()
        self.checkpointer = get_checkpointer()
        self.app = graph.compile(checkpointer=self.checkpointer)

        self.log = logger.bind(
            orchestrator="supervisor",
            codebase=codebase_path,
            app_url=app_url,
        )

    async def run(self, thread_id: str | None = None) -> SupervisorState:
        """
        Run the full supervised test suite.

        Args:
            thread_id: Optional thread ID for checkpointing

        Returns:
            Final supervisor state with all results
        """
        import uuid

        thread_id = thread_id or str(uuid.uuid4())

        initial_state = create_initial_supervisor_state(
            codebase_path=self.codebase_path,
            app_url=self.app_url,
            pr_number=self.pr_number,
            changed_files=self.changed_files,
        )

        # Create Langfuse callback handler for tracing
        langfuse_handler = get_langfuse_handler(
            user_id=self.langfuse_user_id,
            session_id=thread_id,
            trace_name="supervisor_orchestrator",
            tags=["supervisor", "multi-agent", *self.langfuse_tags],
            metadata={
                "codebase_path": self.codebase_path,
                "app_url": self.app_url,
                "pr_number": self.pr_number,
                **self.langfuse_metadata,
            },
        )

        # Build config with Langfuse callbacks
        config = {"configurable": {"thread_id": thread_id}}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        self.log.info("Starting supervised test run", thread_id=thread_id, langfuse_enabled=langfuse_handler is not None)

        try:
            final_state = await self.app.ainvoke(initial_state, config)

            # Add scores to trace for test results
            if langfuse_handler:
                passed = final_state.get("passed_count", 0)
                failed = final_state.get("failed_count", 0)
                total_tests = passed + failed
                if total_tests > 0:
                    pass_rate = passed / total_tests
                    score_trace(
                        trace_id=thread_id,
                        name="test_pass_rate",
                        value=pass_rate,
                        comment=f"{passed}/{total_tests} tests passed",
                    )
                score_trace(
                    trace_id=thread_id,
                    name="total_cost",
                    value=final_state.get("total_cost", 0),
                    comment=f"Total LLM cost: ${final_state.get('total_cost', 0):.4f}",
                )
                score_trace(
                    trace_id=thread_id,
                    name="iterations",
                    value=final_state.get("iteration", 0),
                    comment=f"Supervisor iterations: {final_state.get('iteration', 0)}",
                )

            self.log.info(
                "Supervised test run completed",
                passed=final_state.get("passed_count", 0),
                failed=final_state.get("failed_count", 0),
                iterations=final_state.get("iteration", 0),
                cost=final_state.get("total_cost", 0),
            )

            return final_state

        except Exception as e:
            self.log.error("Supervised test run failed", error=str(e))
            raise
        finally:
            # Ensure Langfuse events are flushed
            flush_langfuse()

    async def get_state(self, thread_id: str) -> dict | None:
        """Get current state of a supervised run."""
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state_snapshot = await self.app.aget_state(config)
            if state_snapshot:
                return {
                    "values": state_snapshot.values,
                    "next": state_snapshot.next,
                    "is_complete": state_snapshot.values.get("task_complete", False),
                    "current_phase": state_snapshot.values.get("current_phase"),
                    "iteration": state_snapshot.values.get("iteration", 0),
                }
            return None
        except Exception as e:
            self.log.error("Failed to get state", thread_id=thread_id, error=str(e))
            return None

    async def resume(self, thread_id: str) -> SupervisorState:
        """Resume a paused supervised run."""
        # Create Langfuse handler for resumed execution
        langfuse_handler = get_langfuse_handler(
            user_id=self.langfuse_user_id,
            session_id=thread_id,
            trace_name="supervisor_orchestrator_resume",
            tags=["supervisor", "resume", *self.langfuse_tags],
            metadata=self.langfuse_metadata,
        )

        config = {"configurable": {"thread_id": thread_id}}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        self.log.info("Resuming supervised run", thread_id=thread_id)

        try:
            final_state = await self.app.ainvoke(None, config)
            return final_state
        finally:
            flush_langfuse()

    def get_summary(self, state: SupervisorState) -> dict:
        """Get a summary of the supervised test run."""
        return {
            "thread_id": state.get("thread_id"),
            "current_phase": state.get("current_phase"),
            "task_complete": state.get("task_complete", False),
            "iterations": state.get("iteration", 0),
            "tests": {
                "total": len(state.get("test_plan", [])),
                "executed": len(state.get("test_results", [])),
                "passed": state.get("passed_count", 0),
                "failed": state.get("failed_count", 0),
            },
            "cost": state.get("total_cost", 0),
            "results": state.get("results", {}),
            "error": state.get("error"),
        }
