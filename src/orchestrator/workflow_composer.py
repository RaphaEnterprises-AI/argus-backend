"""Dynamic Workflow Composer for runtime workflow generation.

RAP-232: Creates executable workflows at runtime by:
1. Analyzing task definitions to identify required capabilities
2. Finding agents with matching capabilities from the registry
3. Building execution graphs that respect dependencies
4. Compiling workflows into executable LangGraph state machines

This enables dynamic composition of agent pipelines based on the task at hand,
rather than relying on static, pre-defined workflows.

Example usage:
    from src.orchestrator.workflow_composer import WorkflowComposer, TaskDefinition
    from src.orchestrator.agent_registry import AgentRegistry

    registry = AgentRegistry()
    composer = WorkflowComposer(registry)

    # Define a complex task
    task = TaskDefinition(
        task_id="full-e2e-test",
        description="Run comprehensive E2E tests for the checkout flow",
        required_capabilities=["code_analysis", "ui_testing", "api_testing"],
        input_schema={"app_url": str, "codebase_path": str},
        output_schema={"test_results": list, "report": dict},
    )

    # Compose and execute
    workflow = composer.compose(task, WorkflowConstraints(max_agents=5))
    result = await workflow.execute(initial_state)
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict

import networkx as nx
import structlog
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from langchain_core.messages import BaseMessage

from ..config import get_settings
from .agent_registry import AgentInfo, AgentRegistry, Capability, get_agent_registry

logger = structlog.get_logger()


# =============================================================================
# Data Classes for Task & Workflow Definition
# =============================================================================


class TaskPriority(int, Enum):
    """Task priority levels for scheduling."""
    CRITICAL = 0  # Execute first, blocking
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4  # Execute when resources available


class ExecutionMode(str, Enum):
    """How tasks should be executed."""
    SEQUENTIAL = "sequential"  # One at a time
    PARALLEL = "parallel"  # All at once (respecting max_parallel)
    PIPELINE = "pipeline"  # Streaming between stages
    CONSENSUS = "consensus"  # Multiple agents must agree


@dataclass
class TaskDefinition:
    """Definition of a task to be composed into a workflow.

    A task definition captures what needs to be done without specifying
    how it should be done. The WorkflowComposer uses this to find
    appropriate agents and build an execution plan.

    Attributes:
        task_id: Unique identifier for this task
        description: Human-readable description of what the task does
        required_capabilities: List of capability names needed (e.g., ["ui_testing", "vision"])
        input_schema: JSON schema defining expected inputs
        output_schema: JSON schema defining expected outputs
        priority: Execution priority (lower = higher priority)
        timeout_seconds: Maximum execution time for this task
        retry_count: Number of retries on failure
        depends_on: List of task_ids that must complete before this task
        metadata: Additional task-specific data
    """
    task_id: str
    description: str
    required_capabilities: list[str]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout_seconds: int = 300
    retry_count: int = 1
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate task definition."""
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if not self.required_capabilities:
            raise ValueError("Task must specify at least one required capability")


@dataclass
class WorkflowConstraints:
    """Constraints for workflow composition.

    These constraints guide how the WorkflowComposer builds and executes
    the workflow, ensuring resource limits and quality requirements are met.

    Attributes:
        max_agents: Maximum number of different agents to use
        timeout_seconds: Total workflow timeout
        max_parallel: Maximum concurrent tasks
        require_consensus: Whether multiple agents must agree on results
        cost_limit: Maximum USD cost for the workflow
        quality_threshold: Minimum confidence score (0.0-1.0)
        fallback_enabled: Allow fallback agents if primary fails
        checkpoint_enabled: Save checkpoints for resumability
    """
    max_agents: int = 5
    timeout_seconds: int = 300
    max_parallel: int = 3
    require_consensus: bool = False
    cost_limit: float | None = None
    quality_threshold: float = 0.8
    fallback_enabled: bool = True
    checkpoint_enabled: bool = True
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL


@dataclass
class WorkflowStage:
    """A stage in the compiled workflow.

    Each stage represents a unit of work that can be executed independently
    once its dependencies are satisfied.
    """
    stage_id: str
    task: TaskDefinition
    assigned_agent: str  # Agent ID from registry
    fallback_agents: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # Stage IDs
    parallel_group: str | None = None  # For parallel execution grouping


@dataclass
class WorkflowPlan:
    """Complete execution plan for a workflow.

    Contains all stages, their dependencies, and execution metadata.
    """
    plan_id: str
    stages: list[WorkflowStage]
    dependency_graph: nx.DiGraph
    execution_order: list[list[str]]  # Groups of stage_ids that can run in parallel
    estimated_duration_seconds: int
    estimated_cost: float
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_stages_by_order(self) -> list[list[WorkflowStage]]:
        """Get stages grouped by execution order."""
        stage_map = {s.stage_id: s for s in self.stages}
        return [
            [stage_map[sid] for sid in group]
            for group in self.execution_order
        ]


class WorkflowState(TypedDict):
    """State passed through the compiled workflow."""
    messages: list[BaseMessage]
    task_inputs: dict[str, Any]  # Input data for tasks
    stage_results: dict[str, Any]  # Results keyed by stage_id
    current_stage: str | None
    completed_stages: list[str]
    failed_stages: list[str]
    error: str | None
    total_cost: float
    started_at: str
    workflow_id: str


@dataclass
class CompiledWorkflow:
    """A compiled, executable workflow.

    This is the result of WorkflowComposer.compose() and contains
    everything needed to execute the workflow.
    """
    workflow_id: str
    plan: WorkflowPlan
    graph: StateGraph
    constraints: WorkflowConstraints

    async def execute(
        self,
        initial_inputs: dict[str, Any],
        config: dict | None = None,
    ) -> WorkflowResult:
        """Execute the compiled workflow.

        Args:
            initial_inputs: Input data for the workflow
            config: Optional LangGraph config (thread_id, etc.)

        Returns:
            WorkflowResult with outputs and execution metadata
        """
        from .checkpointer import get_checkpointer

        # Compile graph
        checkpointer = get_checkpointer() if self.constraints.checkpoint_enabled else None
        app = self.graph.compile(checkpointer=checkpointer)

        # Create initial state
        initial_state = WorkflowState(
            messages=[],
            task_inputs=initial_inputs,
            stage_results={},
            current_stage=None,
            completed_stages=[],
            failed_stages=[],
            error=None,
            total_cost=0.0,
            started_at=datetime.now(UTC).isoformat(),
            workflow_id=self.workflow_id,
        )

        # Configure execution
        run_config = config or {}
        if "configurable" not in run_config:
            run_config["configurable"] = {}
        if "thread_id" not in run_config["configurable"]:
            run_config["configurable"]["thread_id"] = self.workflow_id

        # Execute
        start_time = datetime.now(UTC)
        try:
            final_state = await app.ainvoke(initial_state, run_config)

            return WorkflowResult(
                workflow_id=self.workflow_id,
                success=not final_state.get("error"),
                outputs=final_state.get("stage_results", {}),
                error=final_state.get("error"),
                total_cost=final_state.get("total_cost", 0.0),
                duration_seconds=(datetime.now(UTC) - start_time).total_seconds(),
                completed_stages=final_state.get("completed_stages", []),
                failed_stages=final_state.get("failed_stages", []),
            )
        except Exception as e:
            return WorkflowResult(
                workflow_id=self.workflow_id,
                success=False,
                outputs={},
                error=str(e),
                total_cost=0.0,
                duration_seconds=(datetime.now(UTC) - start_time).total_seconds(),
                completed_stages=[],
                failed_stages=[],
            )


@dataclass
class WorkflowResult:
    """Result of executing a workflow."""
    workflow_id: str
    success: bool
    outputs: dict[str, Any]
    error: str | None
    total_cost: float
    duration_seconds: float
    completed_stages: list[str]
    failed_stages: list[str]


# =============================================================================
# Workflow Composer
# =============================================================================


class WorkflowComposer:
    """Composes dynamic workflows at runtime.

    The WorkflowComposer takes task definitions and constraints, then:
    1. Identifies required capabilities for each task
    2. Finds agents with those capabilities from the registry
    3. Creates an execution graph respecting dependencies
    4. Returns a compiled workflow ready for execution

    Example:
        composer = WorkflowComposer(registry)

        # Simple task
        task = TaskDefinition(
            task_id="analyze-code",
            description="Analyze codebase structure",
            required_capabilities=["code_analysis"],
            input_schema={"path": str},
            output_schema={"summary": str},
        )
        workflow = composer.compose(task, WorkflowConstraints())
        result = await workflow.execute({"path": "/app"})

        # Complex task (auto-decomposed)
        complex_task = "Test the user registration flow end-to-end"
        subtasks = await composer.decompose_task(complex_task)
        workflow = composer.compose_multi(subtasks, WorkflowConstraints())
    """

    def __init__(
        self,
        registry: AgentRegistry,
        task_decomposer: TaskDecomposer | None = None,
    ):
        """Initialize the workflow composer.

        Args:
            registry: Agent registry for capability lookup
            task_decomposer: Optional task decomposer for complex tasks
        """
        self.registry = registry
        self._decomposer = task_decomposer
        self.settings = get_settings()
        self.log = structlog.get_logger().bind(component="workflow_composer")

    @property
    def decomposer(self) -> TaskDecomposer:
        """Lazy-initialize task decomposer."""
        if self._decomposer is None:
            self._decomposer = TaskDecomposer()
        return self._decomposer

    def compose(
        self,
        task: TaskDefinition,
        constraints: WorkflowConstraints,
    ) -> CompiledWorkflow:
        """Build a workflow from a single task definition.

        Args:
            task: The task to build a workflow for
            constraints: Execution constraints

        Returns:
            Compiled workflow ready for execution
        """
        return self.compose_multi([task], constraints)

    def compose_multi(
        self,
        tasks: list[TaskDefinition],
        constraints: WorkflowConstraints,
    ) -> CompiledWorkflow:
        """Build a workflow from multiple task definitions.

        Args:
            tasks: List of tasks to compose
            constraints: Execution constraints

        Returns:
            Compiled workflow

        Raises:
            ValueError: If no agents can satisfy required capabilities
        """
        workflow_id = str(uuid.uuid4())
        self.log.info(
            "Composing workflow",
            workflow_id=workflow_id,
            task_count=len(tasks),
            constraints=constraints,
        )

        # Step 1: Create workflow stages
        stages = self._create_stages(tasks, constraints)

        # Step 2: Build dependency graph
        dep_graph = self._build_dependency_graph(stages)

        # Step 3: Determine execution order
        execution_order = self._compute_execution_order(dep_graph, constraints)

        # Step 4: Estimate cost and duration
        estimated_cost = self._estimate_cost(stages)
        estimated_duration = self._estimate_duration(stages, execution_order, constraints)

        # Step 5: Create workflow plan
        plan = WorkflowPlan(
            plan_id=workflow_id,
            stages=stages,
            dependency_graph=dep_graph,
            execution_order=execution_order,
            estimated_duration_seconds=estimated_duration,
            estimated_cost=estimated_cost,
        )

        # Step 6: Build LangGraph state machine
        graph = self._build_graph(plan, constraints)

        self.log.info(
            "Workflow composed",
            workflow_id=workflow_id,
            stages=len(stages),
            parallel_groups=len(execution_order),
            estimated_cost=estimated_cost,
            estimated_duration=estimated_duration,
        )

        return CompiledWorkflow(
            workflow_id=workflow_id,
            plan=plan,
            graph=graph,
            constraints=constraints,
        )

    async def decompose_task(self, complex_task: str) -> list[TaskDefinition]:
        """Use LLM to break complex task into subtasks.

        Args:
            complex_task: Natural language description of a complex task

        Returns:
            List of TaskDefinition subtasks
        """
        return await self.decomposer.decompose(complex_task)

    def _create_stages(
        self,
        tasks: list[TaskDefinition],
        constraints: WorkflowConstraints,
    ) -> list[WorkflowStage]:
        """Create workflow stages by assigning agents to tasks."""
        stages = []

        for task in tasks:
            # Find agents with required capabilities
            matching_agents = self._find_capable_agents(task.required_capabilities)

            if not matching_agents:
                raise ValueError(
                    f"No agents found with capabilities: {task.required_capabilities}"
                )

            # Select primary agent (highest capability match)
            primary_agent = self._select_best_agent(matching_agents, task)

            # Select fallback agents if enabled
            fallback_agents = []
            if constraints.fallback_enabled:
                fallback_agents = [
                    a.agent_id for a in matching_agents[1:constraints.max_agents]
                ]

            stage = WorkflowStage(
                stage_id=f"stage-{task.task_id}",
                task=task,
                assigned_agent=primary_agent.agent_id,
                fallback_agents=fallback_agents,
                dependencies=[f"stage-{dep}" for dep in task.depends_on],
            )
            stages.append(stage)

        return stages

    def _find_capable_agents(
        self,
        required_capabilities: list[str],
    ) -> list[AgentInfo]:
        """Find agents that have all required capabilities.

        Uses the AgentRegistry.discover_all() method which finds agents
        with ALL specified capabilities.
        """
        # Try to use discover_all with Capability enum values
        try:
            # Convert string capabilities to Capability enum
            enum_caps = []
            for cap in required_capabilities:
                try:
                    enum_caps.append(Capability(cap))
                except ValueError:
                    # Unknown capability, skip
                    self.log.warning(
                        "Unknown capability requested",
                        capability=cap,
                    )
                    continue

            if enum_caps:
                matching = self.registry.discover_all(enum_caps)
            else:
                matching = []

        except Exception as e:
            self.log.warning(
                "Error discovering agents by capability",
                error=str(e),
                capabilities=required_capabilities,
            )
            matching = []

        # Sort by number of capabilities (more capabilities = better match)
        def score_agent(agent: AgentInfo) -> int:
            return len(agent.capabilities)

        matching.sort(key=score_agent, reverse=True)

        # Filter to only healthy agents
        matching = [a for a in matching if a.status == "healthy"]

        return matching

    def _select_best_agent(
        self,
        agents: list[AgentInfo],
        task: TaskDefinition,
    ) -> AgentInfo:
        """Select the best agent for a task."""
        # For now, use the first (highest scored) agent
        # In future, consider:
        # - Agent load/availability
        # - Cost optimization
        # - Historical performance
        return agents[0]

    def _build_dependency_graph(
        self,
        stages: list[WorkflowStage],
    ) -> nx.DiGraph:
        """Build a directed graph of stage dependencies."""
        graph = nx.DiGraph()

        for stage in stages:
            graph.add_node(stage.stage_id, stage=stage)
            for dep in stage.dependencies:
                graph.add_edge(dep, stage.stage_id)

        # Verify no cycles
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Workflow has circular dependencies")

        return graph

    def _compute_execution_order(
        self,
        graph: nx.DiGraph,
        constraints: WorkflowConstraints,
    ) -> list[list[str]]:
        """Compute execution order respecting dependencies and parallelism.

        Returns groups of stage_ids that can run in parallel.
        """
        if constraints.execution_mode == ExecutionMode.SEQUENTIAL:
            # Strictly sequential: one stage per group
            return [[node] for node in nx.topological_sort(graph)]

        # Group stages by their "depth" in the dependency graph
        # Stages at the same depth can run in parallel
        depths: dict[str, int] = {}

        for node in nx.topological_sort(graph):
            predecessors = list(graph.predecessors(node))
            if not predecessors:
                depths[node] = 0
            else:
                depths[node] = max(depths[p] for p in predecessors) + 1

        # Group by depth
        max_depth = max(depths.values()) if depths else 0
        groups: list[list[str]] = [[] for _ in range(max_depth + 1)]

        for node, depth in depths.items():
            groups[depth].append(node)

        # Respect max_parallel constraint
        constrained_groups = []
        for group in groups:
            # Split large groups
            for i in range(0, len(group), constraints.max_parallel):
                constrained_groups.append(group[i:i + constraints.max_parallel])

        return constrained_groups

    def _estimate_cost(self, stages: list[WorkflowStage]) -> float:
        """Estimate total workflow cost."""
        # Simple estimation: sum of estimated costs per stage
        # In practice, would use historical data or model pricing
        total = 0.0
        for stage in stages:
            # Estimate based on task timeout and typical token usage
            # Assume ~0.01 USD per second of agent time as rough estimate
            total += stage.task.timeout_seconds * 0.01
        return total

    def _estimate_duration(
        self,
        stages: list[WorkflowStage],
        execution_order: list[list[str]],
        constraints: WorkflowConstraints,
    ) -> int:
        """Estimate total workflow duration in seconds."""
        stage_map = {s.stage_id: s for s in stages}
        total = 0

        for group in execution_order:
            # Duration of a group is the max of its members
            group_duration = max(
                stage_map[sid].task.timeout_seconds
                for sid in group
            ) if group else 0
            total += group_duration

        return min(total, constraints.timeout_seconds)

    def _build_graph(
        self,
        plan: WorkflowPlan,
        constraints: WorkflowConstraints,
    ) -> StateGraph:
        """Build a LangGraph state machine from the workflow plan."""
        graph = StateGraph(WorkflowState)

        stage_map = {s.stage_id: s for s in plan.stages}

        # Create node functions for each stage
        def make_stage_node(stage: WorkflowStage) -> Callable:
            async def stage_node(state: WorkflowState) -> dict:
                return await self._execute_stage(stage, state)
            return stage_node

        # Add nodes
        for stage in plan.stages:
            graph.add_node(stage.stage_id, make_stage_node(stage))

        # Add routing logic
        if plan.execution_order:
            # Set entry to first group
            first_group = plan.execution_order[0]
            if len(first_group) == 1:
                graph.set_entry_point(first_group[0])
            else:
                # Use a dispatcher for parallel start
                def dispatch_start(state: WorkflowState):
                    return [Send(sid, state) for sid in first_group]
                graph.add_node("__start_dispatch__", dispatch_start)
                graph.set_entry_point("__start_dispatch__")

        # Add edges between groups
        for i, group in enumerate(plan.execution_order[:-1]):
            next_group = plan.execution_order[i + 1]

            # Connect all nodes in current group to next group
            for current_sid in group:
                if len(next_group) == 1:
                    graph.add_edge(current_sid, next_group[0])
                else:
                    # Create aggregator node before parallel dispatch
                    agg_name = f"__agg_{i}__"
                    if agg_name not in [n for n in graph.nodes]:
                        def make_aggregator(next_sids: list[str]) -> Callable:
                            def aggregator(state: WorkflowState):
                                # Check if we should continue
                                if state.get("error"):
                                    return END
                                return [Send(sid, state) for sid in next_sids]
                            return aggregator
                        graph.add_node(agg_name, make_aggregator(next_group))
                    graph.add_edge(current_sid, agg_name)

        # Add edge from last group to END
        if plan.execution_order:
            last_group = plan.execution_order[-1]
            for sid in last_group:
                graph.add_edge(sid, END)

        return graph

    async def _execute_stage(
        self,
        stage: WorkflowStage,
        state: WorkflowState,
    ) -> dict:
        """Execute a single workflow stage.

        Uses the agent type to instantiate the appropriate agent class
        and execute it with the stage inputs.
        """
        self.log.info(
            "Executing stage",
            stage_id=stage.stage_id,
            agent=stage.assigned_agent,
        )

        try:
            # Prepare inputs
            task_inputs = state.get("task_inputs", {})
            prior_results = state.get("stage_results", {})

            # Execute the agent
            result = await self._invoke_agent(
                agent_id=stage.assigned_agent,
                inputs={
                    **task_inputs,
                    "prior_results": prior_results,
                    "task_description": stage.task.description,
                },
                timeout=stage.task.timeout_seconds,
            )

            return {
                "stage_results": {
                    **prior_results,
                    stage.stage_id: result,
                },
                "completed_stages": [*state.get("completed_stages", []), stage.stage_id],
                "current_stage": stage.stage_id,
                "total_cost": state.get("total_cost", 0.0) + result.get("cost", 0.0),
            }

        except Exception as e:
            self.log.error(
                "Stage execution failed",
                stage_id=stage.stage_id,
                error=str(e),
            )

            # Try fallback agents
            for fallback_id in stage.fallback_agents:
                try:
                    result = await self._invoke_agent(
                        agent_id=fallback_id,
                        inputs={
                            **state.get("task_inputs", {}),
                            "prior_results": state.get("stage_results", {}),
                            "task_description": stage.task.description,
                        },
                        timeout=stage.task.timeout_seconds,
                    )

                    self.log.info(
                        "Fallback agent succeeded",
                        stage_id=stage.stage_id,
                        fallback_agent=fallback_id,
                    )

                    return {
                        "stage_results": {
                            **state.get("stage_results", {}),
                            stage.stage_id: result,
                        },
                        "completed_stages": [*state.get("completed_stages", []), stage.stage_id],
                        "current_stage": stage.stage_id,
                        "total_cost": state.get("total_cost", 0.0) + result.get("cost", 0.0),
                    }
                except Exception:
                    continue

            # All agents failed
            return {
                "failed_stages": [*state.get("failed_stages", []), stage.stage_id],
                "error": f"Stage {stage.stage_id} failed: {str(e)}",
            }

    async def _invoke_agent(
        self,
        agent_id: str,
        inputs: dict[str, Any],
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Invoke an agent by its ID.

        Looks up the agent type in the registry and instantiates the
        appropriate agent class to execute the task.

        Args:
            agent_id: Agent ID from the registry
            inputs: Input data for the agent
            timeout: Maximum execution time in seconds

        Returns:
            Agent result as a dictionary
        """
        from datetime import datetime, timezone

        start_time = datetime.now(timezone.utc)

        # Get agent info from registry
        agent_info = self.registry.get(agent_id)
        if not agent_info:
            raise ValueError(f"Agent not found in registry: {agent_id}")

        # Import agent class based on type
        agent_instance = self._get_agent_instance(agent_info.agent_type)

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                agent_instance.execute(**inputs),
                timeout=timeout,
            )

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "cost": result.cost,
                "duration_seconds": duration,
            }

        except asyncio.TimeoutError:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {
                "success": False,
                "data": None,
                "error": f"Execution timed out after {timeout}s",
                "cost": 0.0,
                "duration_seconds": duration,
            }

    def _get_agent_instance(self, agent_type: str) -> Any:
        """Get an agent instance by type.

        Maps agent type strings to actual agent classes and instantiates them.

        Args:
            agent_type: The type of agent (e.g., "code_analyzer", "ui_tester")

        Returns:
            An instantiated agent object

        Raises:
            ValueError: If agent type is unknown
        """
        # Import agent classes
        from ..agents import (
            APITesterAgent,
            CodeAnalyzerAgent,
            DBTesterAgent,
            ReporterAgent,
            SelfHealerAgent,
            TestPlannerAgent,
            UITesterAgent,
        )

        agent_classes = {
            "code_analyzer": CodeAnalyzerAgent,
            "test_planner": TestPlannerAgent,
            "ui_tester": UITesterAgent,
            "api_tester": APITesterAgent,
            "db_tester": DBTesterAgent,
            "self_healer": SelfHealerAgent,
            "reporter": ReporterAgent,
        }

        # Also try to load additional agents
        try:
            from ..agents import (
                AccessibilityCheckerAgent,
                FlakyTestDetector,
                NLPTestCreator,
                PerformanceAnalyzerAgent,
                SecurityScannerAgent,
                VisualAI,
            )
            agent_classes.update({
                "visual_ai": VisualAI,
                "nlp_test_creator": NLPTestCreator,
                "performance_analyzer": PerformanceAnalyzerAgent,
                "security_scanner": SecurityScannerAgent,
                "accessibility_checker": AccessibilityCheckerAgent,
                "flaky_detector": FlakyTestDetector,
            })
        except ImportError:
            pass

        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent_classes[agent_type]()


# =============================================================================
# Task Decomposer (imported here for convenience)
# =============================================================================


class TaskDecomposer:
    """Breaks complex tasks into executable subtasks.

    Uses Claude to analyze natural language task descriptions and
    decompose them into structured TaskDefinitions with proper
    dependencies and capability requirements.

    See task_decomposer.py for full implementation.
    """

    async def decompose(self, task_description: str) -> list[TaskDefinition]:
        """Use Claude to analyze task and identify subtasks.

        This is a stub - see task_decomposer.py for full implementation.
        """
        # Import actual implementation
        from .task_decomposer import TaskDecomposer as FullDecomposer
        decomposer = FullDecomposer()
        return await decomposer.decompose(task_description)

    def identify_capabilities(self, subtask: str) -> list[str]:
        """Map subtask to required capabilities."""
        from .task_decomposer import TaskDecomposer as FullDecomposer
        decomposer = FullDecomposer()
        return decomposer.identify_capabilities(subtask)

    def create_dependency_graph(self, subtasks: list[TaskDefinition]) -> nx.DiGraph:
        """Build dependency graph between subtasks."""
        from .task_decomposer import TaskDecomposer as FullDecomposer
        decomposer = FullDecomposer()
        return decomposer.create_dependency_graph(subtasks)
