"""Tests for the parameterized test executor.

Tests the ParameterizedTestExecutor class which handles:
- Parameter resolution ({{param}} substitution)
- Iteration over parameter sets
- Results tracking per iteration
- Sequential, parallel, and random iteration modes
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.parameterized.executor import (
    ExecutionResult,
    ExecutionStatus,
    IterationResult,
    IterationStatus,
    ParameterizedTestExecutor,
    resolve_parameters,
)
from src.parameterized.models import (
    IterationMode,
    ParameterizedTest,
    ParameterSet,
    TestAssertion,
    TestStep,
)


class TestResolveParameters:
    """Tests for the resolve_parameters function."""

    def test_resolve_string_placeholders(self):
        """Test resolving placeholders in a string."""
        template = "Hello {{name}}, your email is {{email}}"
        values = {"name": "Alice", "email": "alice@example.com"}
        result = resolve_parameters(template, values)
        assert result == "Hello Alice, your email is alice@example.com"

    def test_resolve_dict_placeholders(self):
        """Test resolving placeholders in a dictionary."""
        template = {
            "url": "/users/{{user_id}}",
            "name": "{{username}}",
            "nested": {"value": "{{nested_value}}"},
        }
        values = {"user_id": "123", "username": "admin", "nested_value": "deep"}
        result = resolve_parameters(template, values)
        assert result == {
            "url": "/users/123",
            "name": "admin",
            "nested": {"value": "deep"},
        }

    def test_resolve_list_placeholders(self):
        """Test resolving placeholders in a list."""
        template = ["{{item1}}", "{{item2}}", "static"]
        values = {"item1": "first", "item2": "second"}
        result = resolve_parameters(template, values)
        assert result == ["first", "second", "static"]

    def test_missing_placeholder_kept(self):
        """Test that missing placeholders are kept unchanged."""
        template = "Hello {{name}}, {{unknown}} placeholder"
        values = {"name": "World"}
        result = resolve_parameters(template, values)
        assert result == "Hello World, {{unknown}} placeholder"

    def test_numeric_values(self):
        """Test resolving numeric values."""
        template = "User ID: {{id}}, Amount: {{amount}}"
        values = {"id": 42, "amount": 99.99}
        result = resolve_parameters(template, values)
        assert result == "User ID: 42, Amount: 99.99"


class TestIterationResult:
    """Tests for the IterationResult class."""

    def test_to_dict(self):
        """Test converting iteration result to dictionary."""
        param_set = ParameterSet(name="test_set", values={"key": "value"})
        iteration = IterationResult(
            iteration_index=0,
            parameter_set=param_set,
        )
        iteration.status = IterationStatus.PASSED
        iteration.duration_ms = 1500
        iteration.assertions_passed = 3
        iteration.assertions_failed = 0

        result = iteration.to_dict()

        assert result["iteration_index"] == 0
        assert result["parameter_set_name"] == "test_set"
        assert result["parameter_values"] == {"key": "value"}
        assert result["status"] == "passed"
        assert result["duration_ms"] == 1500
        assert result["assertions_passed"] == 3
        assert result["assertions_failed"] == 0


class TestExecutionResult:
    """Tests for the ExecutionResult class."""

    def test_calculate_stats(self):
        """Test calculating statistics from iteration results."""
        result = ExecutionResult(parameterized_test_id="test-123")

        # Add some iteration results
        for i, duration in enumerate([1000, 2000, 3000]):
            iteration = IterationResult(
                iteration_index=i,
                parameter_set=ParameterSet(name=f"set_{i}", values={"x": i}),
            )
            iteration.duration_ms = duration
            result.iteration_results.append(iteration)

        result.calculate_stats()

        assert result.avg_iteration_ms == 2000.0
        assert result.min_iteration_ms == 1000
        assert result.max_iteration_ms == 3000

    def test_to_dict(self):
        """Test converting execution result to dictionary."""
        result = ExecutionResult(parameterized_test_id="test-456")
        result.total_iterations = 5
        result.passed = 4
        result.failed = 1
        result.status = ExecutionStatus.FAILED

        data = result.to_dict()

        assert data["parameterized_test_id"] == "test-456"
        assert data["total_iterations"] == 5
        assert data["passed"] == 4
        assert data["failed"] == 1
        assert data["status"] == "failed"


class TestParameterizedTestExecutor:
    """Tests for the ParameterizedTestExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create an executor instance."""
        return ParameterizedTestExecutor(
            browser_api_url="http://localhost:8000/api/v1/browser/test",
            timeout_per_iteration_ms=30000,
            max_retries=0,
            stop_on_failure=False,
        )

    @pytest.fixture
    def simple_test(self):
        """Create a simple parameterized test."""
        return ParameterizedTest(
            id="test-001",
            name="Login Test",
            steps=[
                TestStep(action="fill", target="#username", value="{{username}}"),
                TestStep(action="fill", target="#password", value="{{password}}"),
                TestStep(action="click", target="#submit"),
            ],
            assertions=[
                TestAssertion(type="url_contains", expected="{{expected_url}}"),
            ],
        )

    @pytest.fixture
    def parameter_sets(self):
        """Create sample parameter sets."""
        return [
            ParameterSet(
                name="admin_user",
                values={"username": "admin", "password": "admin123", "expected_url": "dashboard"},
            ),
            ParameterSet(
                name="regular_user",
                values={"username": "user", "password": "user123", "expected_url": "home"},
            ),
        ]

    def test_step_to_instruction(self, executor):
        """Test converting a step to a natural language instruction."""
        step = {"action": "fill", "target": "#username", "value": "admin"}
        instruction = executor._step_to_instruction(step)
        assert instruction == "Type 'admin' into #username"

        step = {"action": "click", "target": "#submit"}
        instruction = executor._step_to_instruction(step)
        assert instruction == "Click #submit"

        step = {"action": "goto", "value": "http://example.com"}
        instruction = executor._step_to_instruction(step)
        assert instruction == "Navigate to http://example.com"

    def test_assertion_to_instruction(self, executor):
        """Test converting an assertion to a verification instruction."""
        assertion = {"type": "visible", "target": "#success-message"}
        instruction = executor._assertion_to_instruction(assertion)
        assert instruction == "Verify #success-message is visible"

        assertion = {"type": "url_contains", "expected": "dashboard"}
        instruction = executor._assertion_to_instruction(assertion)
        assert instruction == "Verify URL contains 'dashboard'"

        assertion = {"type": "text_contains", "target": "#header", "expected": "Welcome"}
        instruction = executor._assertion_to_instruction(assertion)
        assert instruction == "Verify #header contains 'Welcome'"

    @pytest.mark.asyncio
    async def test_execute_with_mock_browser_api(
        self, executor, simple_test, parameter_sets
    ):
        """Test executing a parameterized test with mocked browser API."""
        # Mock the httpx client
        mock_response = {
            "success": True,
            "steps": [
                {"step_index": 0, "instruction": "fill", "success": True, "duration_ms": 100},
                {"step_index": 1, "instruction": "fill", "success": True, "duration_ms": 100},
                {"step_index": 2, "instruction": "click", "success": True, "duration_ms": 150},
                {"step_index": 3, "instruction": "verify", "success": True, "duration_ms": 50},
            ],
            "duration": 400,
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value.status_code = 200
            mock_instance.post.return_value.raise_for_status = lambda: None
            mock_instance.post.return_value.json.return_value = mock_response

            result = await executor.execute(
                test=simple_test,
                parameter_sets=parameter_sets,
                app_url="http://localhost:3000",
                browser="chromium",
            )

        assert result.total_iterations == 2
        assert result.passed == 2
        assert result.failed == 0
        assert result.status == ExecutionStatus.PASSED
        assert len(result.iteration_results) == 2

    @pytest.mark.asyncio
    async def test_execute_with_failure(self, executor, simple_test, parameter_sets):
        """Test executing with a failed iteration."""
        # First call succeeds, second fails
        success_response = {"success": True, "steps": [{"success": True}]}
        failure_response = {
            "success": False,
            "error": "Element not found",
            "steps": [{"success": False, "error": "Element not found"}],
        }

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            response = AsyncMock()
            response.status_code = 200
            response.raise_for_status = lambda: None
            if call_count == 0:
                response.json.return_value = success_response
            else:
                response.json.return_value = failure_response
            call_count += 1
            return response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post = mock_post

            result = await executor.execute(
                test=simple_test,
                parameter_sets=parameter_sets,
                app_url="http://localhost:3000",
            )

        assert result.total_iterations == 2
        assert result.passed == 1
        assert result.failed == 1
        assert result.status == ExecutionStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_stop_on_failure(
        self, simple_test, parameter_sets
    ):
        """Test that stop_on_failure stops execution after first failure."""
        executor = ParameterizedTestExecutor(
            browser_api_url="http://localhost:8000/api/v1/browser/test",
            stop_on_failure=True,
        )

        failure_response = {
            "success": False,
            "error": "Element not found",
            "steps": [{"success": False, "error": "Element not found"}],
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value.status_code = 200
            mock_instance.post.return_value.raise_for_status = lambda: None
            mock_instance.post.return_value.json.return_value = failure_response

            result = await executor.execute(
                test=simple_test,
                parameter_sets=parameter_sets,
                app_url="http://localhost:3000",
            )

        # Should stop after first failure
        assert len(result.iteration_results) == 1
        assert result.failed == 1

    def test_expanded_steps_to_instructions(self, executor, simple_test):
        """Test converting expanded test to instructions."""
        from src.parameterized.engine import ParameterizationEngine

        engine = ParameterizationEngine()
        param_set = ParameterSet(
            name="test",
            values={"username": "admin", "password": "secret", "expected_url": "dashboard"},
        )
        expanded = engine.expand_test(simple_test, param_set)

        instructions = executor._expanded_steps_to_instructions(
            expanded, "http://localhost:3000"
        )

        assert len(instructions) == 4  # 3 steps + 1 assertion
        assert "Type 'admin'" in instructions[0]
        assert "Type 'secret'" in instructions[1]
        assert "Click #submit" in instructions[2]
        assert "dashboard" in instructions[3]

    def test_build_failure_summary(self, executor):
        """Test building failure summary from iterations."""
        iterations = []
        for i in range(3):
            iteration = IterationResult(
                iteration_index=i,
                parameter_set=ParameterSet(name=f"set_{i}", values={"x": i}),
            )
            if i < 2:
                iteration.status = IterationStatus.FAILED
                iteration.error_message = "Element not found"
            else:
                iteration.status = IterationStatus.PASSED
            iterations.append(iteration)

        summary = executor._build_failure_summary(iterations)

        assert summary["total_failures"] == 2
        assert len(summary["error_groups"]) == 1
        assert summary["error_groups"][0]["count"] == 2
        assert summary["first_failure"]["iteration_index"] == 0

    def test_cancel_execution(self, executor):
        """Test cancelling execution."""
        assert executor._cancelled is False
        executor.cancel()
        assert executor._cancelled is True
