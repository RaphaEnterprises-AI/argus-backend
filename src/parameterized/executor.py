"""Execution engine for parameterized tests.

This module provides the ParameterizedTestExecutor class which handles:
- Executing expanded tests via the browser pool API
- Tracking results per iteration
- Supporting sequential, parallel, and random iteration modes
- Handling retries and stop-on-failure logic
- Real-time progress updates via callbacks

The executor integrates with:
- ParameterizationEngine for test expansion and parameter substitution
- Browser Pool API for actual test execution
- Supabase for persisting results
"""

import asyncio
import random
import re
import time
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Callable

import httpx
import structlog

from src.parameterized.engine import ParameterizationEngine
from src.parameterized.models import (
    DataSource,
    ExpandedTest,
    IterationMode,
    ParameterizedResult,
    ParameterizedTest,
    ParameterSet,
    ParameterSetResult,
)

logger = structlog.get_logger()


class ExecutionStatus(str, Enum):
    """Status of a parameterized test execution."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ERROR = "error"


class IterationStatus(str, Enum):
    """Status of a single iteration."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class IterationResult:
    """Result from executing a single parameter set iteration.

    Attributes:
        iteration_index: Index of this iteration
        parameter_set: The parameter set used
        status: Execution status
        duration_ms: Execution duration in milliseconds
        step_results: Results for each step
        assertions_passed: Number of passed assertions
        assertions_failed: Number of failed assertions
        error_message: Error message if failed
        error_screenshot: Screenshot captured at error
        retry_count: Number of retries attempted
    """

    def __init__(
        self,
        iteration_index: int,
        parameter_set: ParameterSet,
        parameter_set_id: str | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.iteration_index = iteration_index
        self.parameter_set = parameter_set
        self.parameter_set_id = parameter_set_id
        self.status = IterationStatus.PENDING
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.duration_ms: int = 0
        self.step_results: list[dict[str, Any]] = []
        self.assertions_passed: int = 0
        self.assertions_failed: int = 0
        self.error_message: str | None = None
        self.error_stack: str | None = None
        self.error_screenshot_url: str | None = None
        self.retry_count: int = 0
        self.is_retry: bool = False
        self.original_iteration_id: str | None = None
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/API response."""
        return {
            "id": self.id,
            "iteration_index": self.iteration_index,
            "parameter_set_id": self.parameter_set_id,
            "parameter_values": self.parameter_set.values,
            "parameter_set_name": self.parameter_set.name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "step_results": self.step_results,
            "assertions_passed": self.assertions_passed,
            "assertions_failed": self.assertions_failed,
            "error_message": self.error_message,
            "error_stack": self.error_stack,
            "error_screenshot_url": self.error_screenshot_url,
            "retry_count": self.retry_count,
            "is_retry": self.is_retry,
            "original_iteration_id": self.original_iteration_id,
            "metadata": self.metadata,
        }


class ExecutionResult:
    """Aggregated result from executing all iterations.

    Attributes:
        parameterized_test_id: ID of the parameterized test
        total_iterations: Total number of iterations
        passed: Number of passed iterations
        failed: Number of failed iterations
        skipped: Number of skipped iterations
        error: Number of error iterations
        duration_ms: Total execution duration
        iteration_results: Individual iteration results
        status: Overall execution status
    """

    def __init__(self, parameterized_test_id: str):
        self.id = str(uuid.uuid4())
        self.parameterized_test_id = parameterized_test_id
        self.test_run_id: str | None = None
        self.schedule_run_id: str | None = None
        self.total_iterations: int = 0
        self.passed: int = 0
        self.failed: int = 0
        self.skipped: int = 0
        self.error: int = 0
        self.duration_ms: int = 0
        self.avg_iteration_ms: float | None = None
        self.min_iteration_ms: int | None = None
        self.max_iteration_ms: int | None = None
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.iteration_mode: str = "sequential"
        self.parallel_workers: int = 1
        self.status = ExecutionStatus.PENDING
        self.iteration_results: list[IterationResult] = []
        self.failure_summary: dict[str, Any] = {}
        self.environment: str | None = None
        self.browser: str | None = None
        self.app_url: str | None = None
        self.triggered_by: str | None = None
        self.trigger_type: str | None = None
        self.metadata: dict[str, Any] = {}

    def calculate_stats(self):
        """Calculate aggregate statistics from iteration results."""
        durations = [r.duration_ms for r in self.iteration_results if r.duration_ms > 0]
        if durations:
            self.avg_iteration_ms = sum(durations) / len(durations)
            self.min_iteration_ms = min(durations)
            self.max_iteration_ms = max(durations)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/API response."""
        return {
            "id": self.id,
            "parameterized_test_id": self.parameterized_test_id,
            "test_run_id": self.test_run_id,
            "schedule_run_id": self.schedule_run_id,
            "total_iterations": self.total_iterations,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "avg_iteration_ms": self.avg_iteration_ms,
            "min_iteration_ms": self.min_iteration_ms,
            "max_iteration_ms": self.max_iteration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "iteration_mode": self.iteration_mode,
            "parallel_workers": self.parallel_workers,
            "status": self.status.value,
            "iteration_results": [r.to_dict() for r in self.iteration_results],
            "failure_summary": self.failure_summary,
            "environment": self.environment,
            "browser": self.browser,
            "app_url": self.app_url,
            "triggered_by": self.triggered_by,
            "trigger_type": self.trigger_type,
            "metadata": self.metadata,
        }


class ParameterizedTestExecutor:
    """Executor for parameterized tests.

    This class handles the actual execution of parameterized tests by:
    1. Expanding tests using the ParameterizationEngine
    2. Iterating through parameter sets according to the iteration mode
    3. Executing each expanded test via the browser pool API
    4. Collecting and aggregating results
    5. Supporting retries, timeouts, and stop-on-failure

    Example:
        executor = ParameterizedTestExecutor(
            browser_api_url="http://localhost:8000/api/v1/browser/test",
            timeout_per_iteration_ms=60000,
        )

        result = await executor.execute(
            test=parameterized_test,
            parameter_sets=parameter_sets,
            app_url="http://localhost:3000",
        )

        print(f"Passed: {result.passed}/{result.total_iterations}")
    """

    def __init__(
        self,
        browser_api_url: str | None = None,
        timeout_per_iteration_ms: int = 60000,
        max_retries: int = 0,
        stop_on_failure: bool = False,
        max_parallel: int = 5,
        on_iteration_start: Callable[[int, ParameterSet], None] | None = None,
        on_iteration_complete: Callable[[IterationResult], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ):
        """Initialize the executor.

        Args:
            browser_api_url: URL for the browser pool API (default: uses internal)
            timeout_per_iteration_ms: Timeout per iteration in milliseconds
            max_retries: Maximum number of retries for failed iterations
            stop_on_failure: Stop execution on first failure
            max_parallel: Maximum parallel workers for parallel mode
            on_iteration_start: Callback when an iteration starts
            on_iteration_complete: Callback when an iteration completes
            on_progress: Callback for progress updates (current, total)
        """
        self.browser_api_url = browser_api_url or "http://localhost:8000/api/v1/browser/test"
        self.timeout_per_iteration_ms = timeout_per_iteration_ms
        self.max_retries = max_retries
        self.stop_on_failure = stop_on_failure
        self.max_parallel = max_parallel
        self.on_iteration_start = on_iteration_start
        self.on_iteration_complete = on_iteration_complete
        self.on_progress = on_progress
        self.engine = ParameterizationEngine(strict_validation=False)
        self._cancelled = False

    def cancel(self):
        """Cancel the execution."""
        self._cancelled = True

    async def execute(
        self,
        test: ParameterizedTest | dict[str, Any],
        parameter_sets: list[ParameterSet] | list[dict[str, Any]],
        app_url: str,
        browser: str = "chromium",
        environment: str = "staging",
        test_id: str | None = None,
        triggered_by: str | None = None,
        trigger_type: str = "manual",
        iteration_mode: IterationMode | str = IterationMode.SEQUENTIAL,
        selected_set_ids: list[str] | None = None,
    ) -> ExecutionResult:
        """Execute a parameterized test with the given parameter sets.

        Args:
            test: Parameterized test specification (model or dict)
            parameter_sets: List of parameter sets to iterate over
            app_url: Base URL of the application to test
            browser: Browser type (chromium, firefox, webkit)
            environment: Environment name
            test_id: ID of the parameterized test
            triggered_by: User or system that triggered the execution
            trigger_type: Type of trigger (manual, scheduled, webhook)
            iteration_mode: How to iterate (sequential, parallel, random)
            selected_set_ids: Optional list of specific parameter set IDs to run

        Returns:
            ExecutionResult with aggregated results from all iterations
        """
        self._cancelled = False
        start_time = time.time()

        # Convert to model if dict
        if isinstance(test, dict):
            test = ParameterizedTest(**test)

        # Convert parameter sets to models
        param_set_models = []
        for ps in parameter_sets:
            if isinstance(ps, dict):
                param_set_models.append(ParameterSet(
                    name=ps.get("name", f"set_{len(param_set_models)}"),
                    values=ps.get("values", {k: v for k, v in ps.items() if k not in ["name", "id", "skip", "skip_reason"]}),
                    description=ps.get("description"),
                    tags=ps.get("tags", []),
                    skip=ps.get("skip", False),
                    skip_reason=ps.get("skip_reason"),
                ))
            else:
                param_set_models.append(ps)

        # Filter by selected IDs if provided
        if selected_set_ids:
            param_set_models = [
                ps for i, ps in enumerate(param_set_models)
                if str(i) in selected_set_ids or (hasattr(ps, 'id') and ps.id in selected_set_ids)
            ]

        # Filter out skipped parameter sets
        active_sets = [ps for ps in param_set_models if not ps.skip]

        # Normalize iteration mode
        if isinstance(iteration_mode, str):
            iteration_mode = IterationMode(iteration_mode)

        # Apply iteration mode (reorder if needed)
        if iteration_mode == IterationMode.RANDOM:
            random.shuffle(active_sets)

        # Initialize result
        result = ExecutionResult(parameterized_test_id=test_id or test.id or str(uuid.uuid4()))
        result.total_iterations = len(active_sets)
        result.iteration_mode = iteration_mode.value
        result.started_at = datetime.now(UTC)
        result.status = ExecutionStatus.RUNNING
        result.environment = environment
        result.browser = browser
        result.app_url = app_url
        result.triggered_by = triggered_by
        result.trigger_type = trigger_type

        logger.info(
            "Starting parameterized test execution",
            test_name=test.name,
            total_iterations=result.total_iterations,
            iteration_mode=iteration_mode.value,
            browser=browser,
            app_url=app_url,
        )

        try:
            if iteration_mode == IterationMode.PARALLEL:
                result.parallel_workers = min(self.max_parallel, len(active_sets))
                await self._execute_parallel(test, active_sets, app_url, browser, result)
            else:
                await self._execute_sequential(test, active_sets, app_url, browser, result, iteration_mode)

            # Calculate final status
            if self._cancelled:
                result.status = ExecutionStatus.CANCELLED
            elif result.error > 0:
                result.status = ExecutionStatus.ERROR
            elif result.failed > 0:
                result.status = ExecutionStatus.FAILED
            else:
                result.status = ExecutionStatus.PASSED

        except Exception as e:
            logger.exception("Parameterized test execution failed", error=str(e))
            result.status = ExecutionStatus.ERROR
            result.metadata["execution_error"] = str(e)

        result.completed_at = datetime.now(UTC)
        result.duration_ms = int((time.time() - start_time) * 1000)
        result.calculate_stats()

        # Build failure summary
        if result.failed > 0 or result.error > 0:
            result.failure_summary = self._build_failure_summary(result.iteration_results)

        logger.info(
            "Parameterized test execution completed",
            test_name=test.name,
            status=result.status.value,
            passed=result.passed,
            failed=result.failed,
            error=result.error,
            duration_ms=result.duration_ms,
        )

        return result

    async def _execute_sequential(
        self,
        test: ParameterizedTest,
        parameter_sets: list[ParameterSet],
        app_url: str,
        browser: str,
        result: ExecutionResult,
        iteration_mode: IterationMode,
    ):
        """Execute iterations sequentially."""
        for i, param_set in enumerate(parameter_sets):
            if self._cancelled:
                logger.info("Execution cancelled", iteration=i)
                break

            iteration_result = await self._execute_iteration(
                test=test,
                param_set=param_set,
                iteration_index=i,
                app_url=app_url,
                browser=browser,
            )

            result.iteration_results.append(iteration_result)
            self._update_counts(result, iteration_result)

            if self.on_progress:
                self.on_progress(i + 1, len(parameter_sets))

            # Check stop on failure
            if (iteration_mode == IterationMode.FIRST_FAILURE or self.stop_on_failure):
                if iteration_result.status in (IterationStatus.FAILED, IterationStatus.ERROR):
                    logger.info(
                        "Stopping execution on failure",
                        iteration=i,
                        status=iteration_result.status.value,
                    )
                    break

    async def _execute_parallel(
        self,
        test: ParameterizedTest,
        parameter_sets: list[ParameterSet],
        app_url: str,
        browser: str,
        result: ExecutionResult,
    ):
        """Execute iterations in parallel with limited concurrency."""
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def run_with_semaphore(i: int, param_set: ParameterSet):
            async with semaphore:
                if self._cancelled:
                    return None
                return await self._execute_iteration(
                    test=test,
                    param_set=param_set,
                    iteration_index=i,
                    app_url=app_url,
                    browser=browser,
                )

        tasks = [
            run_with_semaphore(i, ps)
            for i, ps in enumerate(parameter_sets)
        ]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            iteration_result = await coro
            if iteration_result:
                result.iteration_results.append(iteration_result)
                self._update_counts(result, iteration_result)
                completed += 1
                if self.on_progress:
                    self.on_progress(completed, len(parameter_sets))

        # Sort results by iteration index for consistent ordering
        result.iteration_results.sort(key=lambda r: r.iteration_index)

    async def _execute_iteration(
        self,
        test: ParameterizedTest,
        param_set: ParameterSet,
        iteration_index: int,
        app_url: str,
        browser: str,
        retry_count: int = 0,
        original_iteration_id: str | None = None,
    ) -> IterationResult:
        """Execute a single iteration with a specific parameter set."""
        iteration = IterationResult(
            iteration_index=iteration_index,
            parameter_set=param_set,
        )
        iteration.retry_count = retry_count
        iteration.is_retry = retry_count > 0
        iteration.original_iteration_id = original_iteration_id
        iteration.started_at = datetime.now(UTC)
        iteration.status = IterationStatus.RUNNING

        if self.on_iteration_start:
            self.on_iteration_start(iteration_index, param_set)

        logger.debug(
            "Executing iteration",
            index=iteration_index,
            param_set=param_set.name,
            retry=retry_count,
        )

        start_time = time.time()

        try:
            # Expand the test with parameters
            expanded = self.engine.expand_test(test, param_set)

            # Convert expanded steps to instruction strings
            steps = self._expanded_steps_to_instructions(expanded, app_url)

            # Execute via browser API
            async with httpx.AsyncClient(timeout=self.timeout_per_iteration_ms / 1000) as client:
                response = await client.post(
                    self.browser_api_url,
                    json={
                        "url": app_url,
                        "steps": steps,
                        "browser": browser,
                        "screenshot": True,
                        "timeout": self.timeout_per_iteration_ms,
                    },
                )
                response.raise_for_status()
                api_result = response.json()

            # Process API response
            iteration.step_results = api_result.get("steps", [])
            success = api_result.get("success", False)

            # Count assertions from step results
            for step_result in iteration.step_results:
                if step_result.get("success"):
                    iteration.assertions_passed += 1
                else:
                    iteration.assertions_failed += 1

            if success:
                iteration.status = IterationStatus.PASSED
            else:
                iteration.status = IterationStatus.FAILED
                iteration.error_message = api_result.get("error") or "One or more steps failed"

                # Find first failed step for screenshot
                for step in iteration.step_results:
                    if not step.get("success"):
                        iteration.error_screenshot_url = step.get("screenshot")
                        iteration.error_message = step.get("error") or iteration.error_message
                        break

        except httpx.TimeoutException as e:
            logger.warning("Iteration timed out", index=iteration_index, error=str(e))
            iteration.status = IterationStatus.ERROR
            iteration.error_message = f"Timeout after {self.timeout_per_iteration_ms}ms"
        except httpx.HTTPStatusError as e:
            logger.warning("Browser API error", index=iteration_index, status=e.response.status_code)
            iteration.status = IterationStatus.ERROR
            iteration.error_message = f"Browser API error: {e.response.status_code}"
            iteration.error_stack = str(e)
        except Exception as e:
            logger.exception("Iteration execution failed", index=iteration_index, error=str(e))
            iteration.status = IterationStatus.ERROR
            iteration.error_message = str(e)
            iteration.error_stack = str(e.__traceback__) if e.__traceback__ else None

        iteration.duration_ms = int((time.time() - start_time) * 1000)
        iteration.completed_at = datetime.now(UTC)

        # Handle retry logic
        if iteration.status in (IterationStatus.FAILED, IterationStatus.ERROR):
            if retry_count < self.max_retries:
                logger.info(
                    "Retrying failed iteration",
                    index=iteration_index,
                    retry=retry_count + 1,
                    max_retries=self.max_retries,
                )
                return await self._execute_iteration(
                    test=test,
                    param_set=param_set,
                    iteration_index=iteration_index,
                    app_url=app_url,
                    browser=browser,
                    retry_count=retry_count + 1,
                    original_iteration_id=original_iteration_id or iteration.id,
                )

        if self.on_iteration_complete:
            self.on_iteration_complete(iteration)

        return iteration

    def _expanded_steps_to_instructions(
        self,
        expanded: ExpandedTest,
        app_url: str
    ) -> list[str]:
        """Convert expanded test steps to natural language instructions.

        Args:
            expanded: Expanded test with substituted values
            app_url: Base application URL

        Returns:
            List of instruction strings for the browser API
        """
        instructions = []

        # Add setup steps
        for step in expanded.setup:
            instruction = self._step_to_instruction(step)
            if instruction:
                instructions.append(instruction)

        # Add main steps
        for step in expanded.steps:
            instruction = self._step_to_instruction(step)
            if instruction:
                instructions.append(instruction)

        # Add assertion steps (convert to verification instructions)
        for assertion in expanded.assertions:
            instruction = self._assertion_to_instruction(assertion)
            if instruction:
                instructions.append(instruction)

        # Add teardown steps
        for step in expanded.teardown:
            instruction = self._step_to_instruction(step)
            if instruction:
                instructions.append(instruction)

        return instructions

    def _step_to_instruction(self, step: dict[str, Any]) -> str | None:
        """Convert a step dict to a natural language instruction."""
        action = step.get("action", "").lower()
        target = step.get("target", "")
        value = step.get("value", "")
        description = step.get("description", "")

        # Use description if provided
        if description:
            return description

        # Generate instruction based on action type
        if action == "fill" or action == "type":
            if target and value:
                return f"Type '{value}' into {target}"
            elif value:
                return f"Type '{value}'"
        elif action == "click":
            if target:
                return f"Click {target}"
        elif action == "goto" or action == "navigate":
            url = value or target
            if url:
                return f"Navigate to {url}"
        elif action == "wait":
            duration = value or "2 seconds"
            return f"Wait {duration}"
        elif action == "screenshot":
            return "Take a screenshot"
        elif action == "press":
            key = value or target
            if key:
                return f"Press {key}"
        elif action == "hover":
            if target:
                return f"Hover over {target}"
        elif action == "scroll":
            direction = value or "down"
            return f"Scroll {direction}"
        elif action == "select":
            if target and value:
                return f"Select '{value}' from {target}"
        elif action:
            # Generic action
            parts = [action]
            if target:
                parts.append(target)
            if value:
                parts.append(f"with '{value}'")
            return " ".join(parts)

        return None

    def _assertion_to_instruction(self, assertion: dict[str, Any]) -> str | None:
        """Convert an assertion dict to a verification instruction."""
        assertion_type = assertion.get("type", "").lower()
        target = assertion.get("target", "")
        expected = assertion.get("expected", "")
        description = assertion.get("description", "")

        # Use description if provided
        if description:
            return f"Verify {description}"

        # Generate instruction based on assertion type
        if assertion_type == "visible":
            if target:
                return f"Verify {target} is visible"
        elif assertion_type == "not_visible" or assertion_type == "hidden":
            if target:
                return f"Verify {target} is not visible"
        elif assertion_type == "text_contains" or assertion_type == "contains":
            if expected:
                if target:
                    return f"Verify {target} contains '{expected}'"
                return f"Verify page contains '{expected}'"
        elif assertion_type == "text_equals" or assertion_type == "equals":
            if target and expected:
                return f"Verify {target} equals '{expected}'"
        elif assertion_type == "url_contains":
            if expected:
                return f"Verify URL contains '{expected}'"
        elif assertion_type == "url_equals":
            if expected:
                return f"Verify URL equals '{expected}'"
        elif assertion_type == "title_contains":
            if expected:
                return f"Verify page title contains '{expected}'"
        elif assertion_type == "enabled":
            if target:
                return f"Verify {target} is enabled"
        elif assertion_type == "disabled":
            if target:
                return f"Verify {target} is disabled"
        elif assertion_type == "checked":
            if target:
                return f"Verify {target} is checked"
        elif assertion_type:
            # Generic assertion
            parts = [f"Verify {assertion_type}"]
            if target:
                parts.append(f"on {target}")
            if expected:
                parts.append(f"is '{expected}'")
            return " ".join(parts)

        return None

    def _update_counts(self, result: ExecutionResult, iteration: IterationResult):
        """Update result counts based on iteration status."""
        if iteration.status == IterationStatus.PASSED:
            result.passed += 1
        elif iteration.status == IterationStatus.FAILED:
            result.failed += 1
        elif iteration.status == IterationStatus.SKIPPED:
            result.skipped += 1
        elif iteration.status == IterationStatus.ERROR:
            result.error += 1

    def _build_failure_summary(
        self,
        iterations: list[IterationResult]
    ) -> dict[str, Any]:
        """Build a summary of failures for reporting."""
        failures = [
            it for it in iterations
            if it.status in (IterationStatus.FAILED, IterationStatus.ERROR)
        ]

        if not failures:
            return {}

        # Group by error message
        error_groups: dict[str, list[str]] = {}
        for failure in failures:
            error = failure.error_message or "Unknown error"
            # Normalize error message for grouping
            normalized = error[:100]  # Truncate for grouping
            if normalized not in error_groups:
                error_groups[normalized] = []
            error_groups[normalized].append(failure.parameter_set.name)

        return {
            "total_failures": len(failures),
            "error_groups": [
                {
                    "error": error,
                    "count": len(param_sets),
                    "parameter_sets": param_sets[:5],  # Limit to 5 examples
                }
                for error, param_sets in error_groups.items()
            ],
            "first_failure": {
                "iteration_index": failures[0].iteration_index,
                "parameter_set": failures[0].parameter_set.name,
                "error": failures[0].error_message,
            } if failures else None,
        }


def resolve_parameters(
    template: str | dict | list,
    values: dict[str, Any],
    placeholder_pattern: str = r"\{\{(\w+)\}\}",
) -> str | dict | list:
    """Resolve {{param}} placeholders in a template with actual values.

    This is a convenience function for parameter substitution that handles
    strings, dicts, and lists recursively.

    Args:
        template: Template containing {{param}} placeholders
        values: Dictionary mapping parameter names to values
        placeholder_pattern: Regex pattern for placeholders (default: {{param}})

    Returns:
        Template with all placeholders replaced by their values

    Example:
        >>> resolve_parameters("Hello {{name}}!", {"name": "World"})
        "Hello World!"

        >>> resolve_parameters(
        ...     {"url": "/users/{{user_id}}", "name": "{{username}}"},
        ...     {"user_id": "123", "username": "admin"}
        ... )
        {"url": "/users/123", "name": "admin"}
    """
    pattern = re.compile(placeholder_pattern)

    def replace_in_string(s: str) -> str:
        def replacer(match: re.Match) -> str:
            param = match.group(1)
            if param in values:
                return str(values[param])
            return match.group(0)  # Keep original if not found
        return pattern.sub(replacer, s)

    if isinstance(template, str):
        return replace_in_string(template)
    elif isinstance(template, dict):
        return {
            key: resolve_parameters(val, values, placeholder_pattern)
            for key, val in template.items()
        }
    elif isinstance(template, list):
        return [
            resolve_parameters(item, values, placeholder_pattern)
            for item in template
        ]
    else:
        return template
