"""Parallel Executor for running workflow tasks concurrently.

RAP-232: Executes workflow tasks in parallel where dependencies allow,
providing:
- Dependency-aware scheduling
- Concurrent execution with configurable limits
- Result aggregation
- Error handling and partial failure recovery
- Resource management (semaphores, rate limiting)

This module works with the WorkflowComposer to execute compiled workflows
efficiently, maximizing parallelism while respecting constraints.

Example usage:
    from src.orchestrator.parallel_executor import ParallelExecutor
    from src.orchestrator.workflow_composer import CompiledWorkflow

    executor = ParallelExecutor(max_workers=5)

    # Execute workflow
    result = await executor.execute(workflow)

    # Or execute with streaming updates
    async for update in executor.stream_execute(workflow):
        print(f"Stage {update.stage_id}: {update.status}")
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

import structlog

from ..config import get_settings

if TYPE_CHECKING:
    from .workflow_composer import CompiledWorkflow, WorkflowPlan, WorkflowStage

logger = structlog.get_logger()


# =============================================================================
# Execution Status & Events
# =============================================================================


class StageStatus(str, Enum):
    """Status of a workflow stage."""
    PENDING = "pending"
    WAITING = "waiting"  # Waiting for dependencies
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class StageExecutionUpdate:
    """Update event from stage execution."""
    stage_id: str
    status: StageStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    result: Any = None
    error: str | None = None
    duration_seconds: float = 0.0
    cost: float = 0.0


@dataclass
class ExecutionProgress:
    """Overall execution progress."""
    total_stages: int
    completed: int
    failed: int
    running: int
    pending: int
    elapsed_seconds: float
    estimated_remaining_seconds: float | None


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
    stage_results: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Parallel Executor
# =============================================================================


class ParallelExecutor:
    """Executes workflow tasks in parallel where possible.

    The ParallelExecutor takes a compiled workflow and executes its stages
    concurrently, respecting dependency constraints and resource limits.

    Features:
    - Dependency-aware scheduling using topological ordering
    - Configurable concurrency limits
    - Streaming execution updates
    - Graceful error handling and cancellation
    - Cost tracking across stages

    Example:
        executor = ParallelExecutor(max_workers=5)
        result = await executor.execute(workflow)

        # With streaming
        async for update in executor.stream_execute(workflow):
            if update.status == StageStatus.COMPLETED:
                print(f"Stage {update.stage_id} completed in {update.duration_seconds}s")
    """

    def __init__(
        self,
        max_workers: int = 5,
        stage_timeout: float = 300.0,
        fail_fast: bool = False,
        retry_count: int = 1,
    ):
        """Initialize the parallel executor.

        Args:
            max_workers: Maximum concurrent stage executions
            stage_timeout: Default timeout per stage in seconds
            fail_fast: Stop all execution on first failure
            retry_count: Number of retries for failed stages
        """
        self.max_workers = max_workers
        self.stage_timeout = stage_timeout
        self.fail_fast = fail_fast
        self.retry_count = retry_count
        self.settings = get_settings()
        self.log = structlog.get_logger().bind(component="parallel_executor")

        # Execution state
        self._semaphore: asyncio.Semaphore | None = None
        self._stage_status: dict[str, StageStatus] = {}
        self._stage_results: dict[str, Any] = {}
        self._stage_errors: dict[str, str] = {}
        self._cancel_event: asyncio.Event | None = None
        self._updates_queue: asyncio.Queue | None = None

    async def execute(
        self,
        workflow: CompiledWorkflow,
        initial_inputs: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Execute the compiled workflow.

        Args:
            workflow: Compiled workflow from WorkflowComposer
            initial_inputs: Initial input data for the workflow

        Returns:
            WorkflowResult with outputs and execution metadata
        """
        start_time = datetime.now(UTC)
        self.log.info(
            "Starting workflow execution",
            workflow_id=workflow.workflow_id,
            stages=len(workflow.plan.stages),
        )

        # Initialize execution state
        self._reset_state(workflow.plan)
        initial_inputs = initial_inputs or {}

        try:
            # Execute stages by execution order groups
            for group in workflow.plan.execution_order:
                if self._is_cancelled():
                    break

                # Execute group stages in parallel
                await self._execute_group(
                    workflow=workflow,
                    stage_ids=group,
                    inputs=initial_inputs,
                )

                # Check for failures if fail_fast
                if self.fail_fast and self._has_failures():
                    self.log.warning(
                        "Stopping execution due to failure (fail_fast=True)"
                    )
                    break

            # Build result
            duration = (datetime.now(UTC) - start_time).total_seconds()
            completed = [
                sid for sid, status in self._stage_status.items()
                if status == StageStatus.COMPLETED
            ]
            failed = [
                sid for sid, status in self._stage_status.items()
                if status == StageStatus.FAILED
            ]

            total_cost = sum(
                r.get("cost", 0.0) if isinstance(r, dict) else 0.0
                for r in self._stage_results.values()
            )

            success = len(failed) == 0

            self.log.info(
                "Workflow execution completed",
                workflow_id=workflow.workflow_id,
                success=success,
                completed=len(completed),
                failed=len(failed),
                duration=duration,
                total_cost=total_cost,
            )

            return WorkflowResult(
                workflow_id=workflow.workflow_id,
                success=success,
                outputs=self._stage_results,
                error=self._stage_errors.get(failed[0]) if failed else None,
                total_cost=total_cost,
                duration_seconds=duration,
                completed_stages=completed,
                failed_stages=failed,
                stage_results=self._stage_results.copy(),
            )

        except asyncio.CancelledError:
            self.log.warning("Workflow execution cancelled")
            raise
        except Exception as e:
            self.log.error("Workflow execution failed", error=str(e))
            duration = (datetime.now(UTC) - start_time).total_seconds()
            return WorkflowResult(
                workflow_id=workflow.workflow_id,
                success=False,
                outputs={},
                error=str(e),
                total_cost=0.0,
                duration_seconds=duration,
                completed_stages=[],
                failed_stages=list(self._stage_status.keys()),
            )

    async def stream_execute(
        self,
        workflow: CompiledWorkflow,
        initial_inputs: dict[str, Any] | None = None,
    ) -> AsyncIterator[StageExecutionUpdate]:
        """Execute workflow with streaming updates.

        Args:
            workflow: Compiled workflow from WorkflowComposer
            initial_inputs: Initial input data for the workflow

        Yields:
            StageExecutionUpdate for each stage state change
        """
        # Initialize updates queue
        self._updates_queue = asyncio.Queue()
        initial_inputs = initial_inputs or {}

        # Start execution in background task
        execution_task = asyncio.create_task(
            self._execute_with_updates(workflow, initial_inputs)
        )

        try:
            # Yield updates as they come in
            while True:
                try:
                    update = await asyncio.wait_for(
                        self._updates_queue.get(),
                        timeout=1.0,
                    )

                    if update is None:  # Sentinel for completion
                        break

                    yield update

                except asyncio.TimeoutError:
                    # Check if execution is done
                    if execution_task.done():
                        break

        finally:
            # Ensure execution task completes
            if not execution_task.done():
                execution_task.cancel()
                try:
                    await execution_task
                except asyncio.CancelledError:
                    pass

            self._updates_queue = None

    async def _execute_with_updates(
        self,
        workflow: CompiledWorkflow,
        inputs: dict[str, Any],
    ) -> None:
        """Execute workflow and post updates to queue."""
        try:
            await self.execute(workflow, inputs)
        finally:
            # Signal completion
            if self._updates_queue:
                await self._updates_queue.put(None)

    def cancel(self) -> None:
        """Cancel the current execution."""
        if self._cancel_event:
            self._cancel_event.set()

    def get_progress(self) -> ExecutionProgress:
        """Get current execution progress."""
        total = len(self._stage_status)
        completed = sum(1 for s in self._stage_status.values() if s == StageStatus.COMPLETED)
        failed = sum(1 for s in self._stage_status.values() if s == StageStatus.FAILED)
        running = sum(1 for s in self._stage_status.values() if s == StageStatus.RUNNING)
        pending = sum(1 for s in self._stage_status.values() if s in (StageStatus.PENDING, StageStatus.WAITING))

        return ExecutionProgress(
            total_stages=total,
            completed=completed,
            failed=failed,
            running=running,
            pending=pending,
            elapsed_seconds=0.0,  # Would need to track start time
            estimated_remaining_seconds=None,
        )

    def _reset_state(self, plan: WorkflowPlan) -> None:
        """Reset execution state for a new run."""
        self._semaphore = asyncio.Semaphore(self.max_workers)
        self._stage_status = {s.stage_id: StageStatus.PENDING for s in plan.stages}
        self._stage_results = {}
        self._stage_errors = {}
        self._cancel_event = asyncio.Event()

    def _is_cancelled(self) -> bool:
        """Check if execution has been cancelled."""
        return self._cancel_event is not None and self._cancel_event.is_set()

    def _has_failures(self) -> bool:
        """Check if any stage has failed."""
        return any(s == StageStatus.FAILED for s in self._stage_status.values())

    async def _execute_group(
        self,
        workflow: CompiledWorkflow,
        stage_ids: list[str],
        inputs: dict[str, Any],
    ) -> None:
        """Execute a group of stages in parallel."""
        stage_map = {s.stage_id: s for s in workflow.plan.stages}

        # Create tasks for each stage
        tasks = []
        for stage_id in stage_ids:
            if self._is_cancelled():
                break

            stage = stage_map[stage_id]

            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(stage):
                self._stage_status[stage_id] = StageStatus.SKIPPED
                self.log.info(
                    "Skipping stage due to failed dependencies",
                    stage_id=stage_id,
                )
                continue

            # Create task
            task = asyncio.create_task(
                self._execute_stage(workflow, stage, inputs)
            )
            tasks.append(task)

        # Wait for all tasks in group
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _dependencies_satisfied(self, stage: WorkflowStage) -> bool:
        """Check if all dependencies of a stage are satisfied."""
        for dep_id in stage.dependencies:
            status = self._stage_status.get(dep_id)
            if status != StageStatus.COMPLETED:
                return False
        return True

    async def _execute_stage(
        self,
        workflow: CompiledWorkflow,
        stage: WorkflowStage,
        inputs: dict[str, Any],
    ) -> None:
        """Execute a single stage with retry logic."""
        start_time = datetime.now(UTC)

        async with self._semaphore:
            self._stage_status[stage.stage_id] = StageStatus.RUNNING
            await self._post_update(StageExecutionUpdate(
                stage_id=stage.stage_id,
                status=StageStatus.RUNNING,
            ))

            self.log.info(
                "Executing stage",
                stage_id=stage.stage_id,
                agent=stage.assigned_agent,
            )

            last_error: str | None = None

            # Retry loop
            for attempt in range(self.retry_count + 1):
                if self._is_cancelled():
                    self._stage_status[stage.stage_id] = StageStatus.CANCELLED
                    return

                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        self._run_stage(workflow, stage, inputs),
                        timeout=stage.task.timeout_seconds or self.stage_timeout,
                    )

                    # Success
                    duration = (datetime.now(UTC) - start_time).total_seconds()
                    self._stage_status[stage.stage_id] = StageStatus.COMPLETED
                    self._stage_results[stage.stage_id] = result

                    await self._post_update(StageExecutionUpdate(
                        stage_id=stage.stage_id,
                        status=StageStatus.COMPLETED,
                        result=result,
                        duration_seconds=duration,
                        cost=result.get("cost", 0.0) if isinstance(result, dict) else 0.0,
                    ))

                    self.log.info(
                        "Stage completed",
                        stage_id=stage.stage_id,
                        duration=duration,
                        attempt=attempt + 1,
                    )
                    return

                except asyncio.TimeoutError:
                    last_error = f"Stage timed out after {self.stage_timeout}s"
                    self.log.warning(
                        "Stage timeout",
                        stage_id=stage.stage_id,
                        attempt=attempt + 1,
                    )

                except Exception as e:
                    last_error = str(e)
                    self.log.warning(
                        "Stage execution error",
                        stage_id=stage.stage_id,
                        error=str(e),
                        attempt=attempt + 1,
                    )

                # Brief delay before retry
                if attempt < self.retry_count:
                    await asyncio.sleep(1.0 * (attempt + 1))

            # All retries exhausted
            duration = (datetime.now(UTC) - start_time).total_seconds()
            self._stage_status[stage.stage_id] = StageStatus.FAILED
            self._stage_errors[stage.stage_id] = last_error or "Unknown error"

            await self._post_update(StageExecutionUpdate(
                stage_id=stage.stage_id,
                status=StageStatus.FAILED,
                error=last_error,
                duration_seconds=duration,
            ))

            self.log.error(
                "Stage failed after retries",
                stage_id=stage.stage_id,
                error=last_error,
                attempts=self.retry_count + 1,
            )

    async def _run_stage(
        self,
        workflow: CompiledWorkflow,
        stage: WorkflowStage,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run stage execution via workflow's agent invocation.

        Uses the WorkflowComposer's agent invocation method to execute
        the stage with the appropriate agent.
        """
        # Get prior results for this stage's dependencies
        prior_results = {
            dep_id: self._stage_results.get(dep_id)
            for dep_id in stage.dependencies
            if dep_id in self._stage_results
        }

        # Import the WorkflowComposer to use its agent invocation
        from .workflow_composer import WorkflowComposer
        from .agent_registry import get_agent_registry

        # Create a temporary composer to invoke the agent
        registry = get_agent_registry()
        composer = WorkflowComposer(registry)

        result = await composer._invoke_agent(
            agent_id=stage.assigned_agent,
            inputs={
                **inputs,
                "prior_results": prior_results,
                "task_description": stage.task.description,
            },
            timeout=stage.task.timeout_seconds,
        )

        return result

    async def _post_update(self, update: StageExecutionUpdate) -> None:
        """Post an update to the streaming queue if available."""
        if self._updates_queue:
            await self._updates_queue.put(update)


# =============================================================================
# Execution Strategies
# =============================================================================


class ExecutionStrategy:
    """Base class for execution strategies."""

    async def execute(
        self,
        stages: list[WorkflowStage],
        executor: ParallelExecutor,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute stages according to the strategy."""
        raise NotImplementedError


class SequentialStrategy(ExecutionStrategy):
    """Execute stages one at a time in order."""

    async def execute(
        self,
        stages: list[WorkflowStage],
        executor: ParallelExecutor,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        results = {}
        for stage in stages:
            # Would execute stage here
            results[stage.stage_id] = {}
        return results


class ParallelStrategy(ExecutionStrategy):
    """Execute independent stages in parallel."""

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent

    async def execute(
        self,
        stages: list[WorkflowStage],
        executor: ParallelExecutor,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = {}

        async def run_with_semaphore(stage: WorkflowStage):
            async with semaphore:
                # Would execute stage here
                return stage.stage_id, {}

        tasks = [run_with_semaphore(s) for s in stages]
        stage_results = await asyncio.gather(*tasks)

        for stage_id, result in stage_results:
            results[stage_id] = result

        return results


class PipelineStrategy(ExecutionStrategy):
    """Execute stages in a streaming pipeline fashion."""

    async def execute(
        self,
        stages: list[WorkflowStage],
        executor: ParallelExecutor,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        # Pipeline execution allows data to flow between stages
        # as soon as partial results are available
        results = {}
        current_input = inputs

        for stage in stages:
            # Execute stage with current input
            # Would pass current_input to stage
            result = {}
            results[stage.stage_id] = result

            # Update input for next stage
            current_input = {**current_input, **result}

        return results


# =============================================================================
# Resource Manager
# =============================================================================


class ResourceManager:
    """Manages shared resources for parallel execution.

    Handles:
    - Rate limiting per agent/provider
    - Resource pooling (browser instances, DB connections)
    - Cost tracking and limits
    """

    def __init__(
        self,
        max_cost: float | None = None,
        rate_limits: dict[str, int] | None = None,
    ):
        """Initialize resource manager.

        Args:
            max_cost: Maximum cost before stopping execution
            rate_limits: Rate limits per agent/provider (requests per minute)
        """
        self.max_cost = max_cost
        self.rate_limits = rate_limits or {}
        self._current_cost = 0.0
        self._request_counts: dict[str, list[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self.log = structlog.get_logger().bind(component="resource_manager")

    async def acquire(self, resource_type: str) -> bool:
        """Acquire a resource slot.

        Args:
            resource_type: Type of resource (e.g., agent ID, provider name)

        Returns:
            True if resource was acquired, False if rate limited
        """
        async with self._lock:
            # Check rate limit
            if resource_type in self.rate_limits:
                limit = self.rate_limits[resource_type]
                now = datetime.now(UTC)
                minute_ago = now.replace(second=now.second - 60)

                # Clean old entries
                self._request_counts[resource_type] = [
                    t for t in self._request_counts[resource_type]
                    if t > minute_ago
                ]

                # Check limit
                if len(self._request_counts[resource_type]) >= limit:
                    self.log.warning(
                        "Rate limit reached",
                        resource=resource_type,
                        limit=limit,
                    )
                    return False

                # Record request
                self._request_counts[resource_type].append(now)

            return True

    async def track_cost(self, cost: float) -> bool:
        """Track cost and check against limit.

        Args:
            cost: Cost to add

        Returns:
            True if within budget, False if exceeded
        """
        async with self._lock:
            self._current_cost += cost

            if self.max_cost and self._current_cost > self.max_cost:
                self.log.warning(
                    "Cost limit exceeded",
                    current=self._current_cost,
                    limit=self.max_cost,
                )
                return False

            return True

    @property
    def current_cost(self) -> float:
        """Get current accumulated cost."""
        return self._current_cost

    def reset(self) -> None:
        """Reset tracking state."""
        self._current_cost = 0.0
        self._request_counts.clear()
