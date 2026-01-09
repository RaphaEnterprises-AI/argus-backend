"""Integration tests for the full orchestration flow."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from typing import TypedDict


class TestFullOrchestrationFlow:
    """Integration tests for complete test runs."""

    @pytest.mark.asyncio
    async def test_orchestrator_initializes_correctly(self):
        """Should initialize orchestrator with correct settings."""
        from src.orchestrator.graph import TestingOrchestrator

        with patch("src.orchestrator.graph.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            orchestrator = TestingOrchestrator(
                codebase_path="/test/app",
                app_url="http://localhost:3000"
            )

            assert orchestrator.codebase_path == "/test/app"
            assert orchestrator.app_url == "http://localhost:3000"
            assert orchestrator.app is not None
            assert orchestrator.checkpointer is not None

    @pytest.mark.asyncio
    async def test_orchestrator_with_pr_context(self):
        """Should initialize with PR context."""
        from src.orchestrator.graph import TestingOrchestrator

        with patch("src.orchestrator.graph.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            orchestrator = TestingOrchestrator(
                codebase_path="/test/app",
                app_url="http://localhost:3000",
                pr_number=123,
                changed_files=["src/main.py", "tests/test_main.py"],
            )

            assert orchestrator.pr_number == 123
            assert orchestrator.changed_files == ["src/main.py", "tests/test_main.py"]

    @pytest.mark.asyncio
    async def test_create_initial_state(self):
        """Should create valid initial state."""
        from src.orchestrator.state import create_initial_state

        state = create_initial_state(
            codebase_path="/test/app",
            app_url="http://localhost:3000",
            pr_number=42,
            changed_files=["file.py"],
        )

        assert state["codebase_path"] == "/test/app"
        assert state["app_url"] == "http://localhost:3000"
        assert state["pr_number"] == 42
        assert state["changed_files"] == ["file.py"]
        assert state["iteration"] == 0
        assert state["passed_count"] == 0
        assert state["failed_count"] == 0
        assert state["should_continue"] is True
        assert state["run_id"] is not None

    @pytest.mark.asyncio
    async def test_graph_structure(self):
        """Should create graph with expected structure."""
        from src.orchestrator.graph import create_testing_graph
        from src.config import get_settings

        settings = get_settings()
        graph = create_testing_graph(settings)

        # Check nodes exist
        assert "analyze_code" in graph.nodes
        assert "plan_tests" in graph.nodes
        assert "execute_test" in graph.nodes
        assert "self_heal" in graph.nodes
        assert "report" in graph.nodes

    @pytest.mark.asyncio
    async def test_routing_after_analysis_success(self):
        """Should route to plan_tests after successful analysis."""
        from src.orchestrator.graph import route_after_analysis

        state = {
            "error": None,
            "testable_surfaces": [{"type": "ui", "path": "/login"}],
        }

        next_node = route_after_analysis(state)
        assert next_node == "plan_tests"

    @pytest.mark.asyncio
    async def test_routing_after_analysis_no_surfaces(self):
        """Should route to report if no testable surfaces found."""
        from src.orchestrator.graph import route_after_analysis

        state = {
            "error": None,
            "testable_surfaces": [],
        }

        next_node = route_after_analysis(state)
        assert next_node == "report"

    @pytest.mark.asyncio
    async def test_routing_after_analysis_error(self):
        """Should route to report on error."""
        from src.orchestrator.graph import route_after_analysis

        state = {
            "error": "Analysis failed",
            "testable_surfaces": [],
        }

        next_node = route_after_analysis(state)
        assert next_node == "report"

    @pytest.mark.asyncio
    async def test_routing_after_planning_success(self):
        """Should route to execute_test after successful planning."""
        from src.orchestrator.graph import route_after_planning

        state = {
            "error": None,
            "test_plan": [{"id": "test-1", "name": "Login Test"}],
        }

        next_node = route_after_planning(state)
        assert next_node == "execute_test"

    @pytest.mark.asyncio
    async def test_routing_after_planning_empty_plan(self):
        """Should route to report if no tests planned."""
        from src.orchestrator.graph import route_after_planning

        state = {
            "error": None,
            "test_plan": [],
        }

        next_node = route_after_planning(state)
        assert next_node == "report"

    @pytest.mark.asyncio
    async def test_routing_after_execution_more_tests(self):
        """Should route to execute_test if more tests to run."""
        from src.orchestrator.graph import route_after_execution

        state = {
            "error": None,
            "healing_queue": [],
            "should_continue": True,
            "current_test_index": 1,
            "test_plan": [{"id": "test-1"}, {"id": "test-2"}, {"id": "test-3"}],
        }

        next_node = route_after_execution(state)
        assert next_node == "execute_test"

    @pytest.mark.asyncio
    async def test_routing_after_execution_complete(self):
        """Should route to report when all tests complete."""
        from src.orchestrator.graph import route_after_execution

        state = {
            "error": None,
            "healing_queue": [],
            "should_continue": True,
            "current_test_index": 3,
            "test_plan": [{"id": "test-1"}, {"id": "test-2"}, {"id": "test-3"}],
        }

        next_node = route_after_execution(state)
        assert next_node == "report"

    @pytest.mark.asyncio
    async def test_routing_after_execution_needs_healing(self):
        """Should route to self_heal when failures need healing."""
        from src.orchestrator.graph import route_after_execution

        with patch("src.orchestrator.graph.get_settings") as mock_settings:
            settings = MagicMock()
            settings.self_heal_enabled = True
            mock_settings.return_value = settings

            state = {
                "error": None,
                "healing_queue": ["test-1"],
                "should_continue": True,
                "current_test_index": 1,
                "test_plan": [{"id": "test-1"}],
            }

            next_node = route_after_execution(state)
            assert next_node == "self_heal"

    @pytest.mark.asyncio
    async def test_should_continue_max_iterations(self):
        """Should stop when max iterations reached."""
        from src.orchestrator.graph import should_continue

        with patch("src.orchestrator.graph.get_settings") as mock_settings:
            settings = MagicMock()
            settings.cost_limit_per_run = 100.0
            mock_settings.return_value = settings

            state = {
                "iteration": 100,
                "max_iterations": 100,
                "total_cost": 0.0,
                "error": None,
                "should_continue": True,
            }

            result = should_continue(state)
            assert result is False

    @pytest.mark.asyncio
    async def test_should_continue_cost_limit(self):
        """Should stop when cost limit reached."""
        from src.orchestrator.graph import should_continue

        with patch("src.orchestrator.graph.get_settings") as mock_settings:
            settings = MagicMock()
            settings.cost_limit_per_run = 10.0
            mock_settings.return_value = settings

            state = {
                "iteration": 5,
                "max_iterations": 100,
                "total_cost": 15.0,
                "error": None,
                "should_continue": True,
            }

            result = should_continue(state)
            assert result is False

    @pytest.mark.asyncio
    async def test_should_continue_on_error(self):
        """Should stop on error."""
        from src.orchestrator.graph import should_continue

        with patch("src.orchestrator.graph.get_settings") as mock_settings:
            settings = MagicMock()
            settings.cost_limit_per_run = 100.0
            mock_settings.return_value = settings

            state = {
                "iteration": 5,
                "max_iterations": 100,
                "total_cost": 0.0,
                "error": "Something went wrong",
                "should_continue": True,
            }

            result = should_continue(state)
            assert result is False


class TestChatGraphFlow:
    """Tests for chat graph functionality."""

    @pytest.mark.asyncio
    async def test_chat_graph_creation(self):
        """Should create chat graph with expected structure."""
        from src.orchestrator.chat_graph import create_chat_graph

        graph = create_chat_graph()
        assert graph is not None

        # Verify nodes exist
        assert "chat" in graph.nodes
        assert "tools" in graph.nodes

    @pytest.mark.asyncio
    async def test_chat_should_continue_with_tools(self):
        """Should continue to tools when tool calls present."""
        from src.orchestrator.chat_graph import should_continue

        mock_message = MagicMock()
        mock_message.tool_calls = [{"id": "call-1", "name": "runTest"}]

        state = {"messages": [mock_message]}

        result = should_continue(state)
        assert result == "tools"

    @pytest.mark.asyncio
    async def test_chat_should_end_without_tools(self):
        """Should end when no tool calls."""
        from src.orchestrator.chat_graph import should_continue

        mock_message = MagicMock()
        mock_message.tool_calls = None

        state = {"messages": [mock_message]}

        result = should_continue(state)
        assert result == "end"

    @pytest.mark.asyncio
    async def test_system_prompt_generation(self):
        """Should generate system prompt with app URL."""
        from src.orchestrator.chat_graph import create_system_prompt

        prompt = create_system_prompt("http://example.com")

        assert "http://example.com" in prompt
        assert "Argus" in prompt
        assert "E2E Testing" in prompt


class TestOrchestratorMethods:
    """Tests for TestingOrchestrator methods."""

    @pytest.mark.asyncio
    async def test_get_run_summary(self):
        """Should generate correct run summary."""
        from src.orchestrator.graph import TestingOrchestrator

        with patch("src.orchestrator.graph.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            orchestrator = TestingOrchestrator(
                codebase_path="/test/app",
                app_url="http://localhost:3000"
            )

            state = {
                "run_id": "run-123",
                "started_at": "2024-01-01T00:00:00Z",
                "passed_count": 5,
                "failed_count": 2,
                "skipped_count": 1,
                "total_cost": 1.50,
                "iteration": 10,
                "error": None,
            }

            summary = orchestrator.get_run_summary(state)

            assert summary["run_id"] == "run-123"
            assert summary["total_tests"] == 8
            assert summary["passed"] == 5
            assert summary["failed"] == 2
            assert summary["skipped"] == 1
            assert summary["pass_rate"] == 5 / 8
            assert summary["total_cost"] == 1.50

    @pytest.mark.asyncio
    async def test_get_state(self):
        """Should get state for thread."""
        from src.orchestrator.graph import TestingOrchestrator

        with patch("src.orchestrator.graph.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            orchestrator = TestingOrchestrator(
                codebase_path="/test/app",
                app_url="http://localhost:3000"
            )

            # Mock the app.aget_state
            mock_state = MagicMock()
            mock_state.values = {"iteration": 5}
            mock_state.next = ["execute_test"]
            mock_state.config = {"configurable": {"thread_id": "test-thread"}}
            mock_state.metadata = {}

            orchestrator.app.aget_state = AsyncMock(return_value=mock_state)

            state = await orchestrator.get_state("test-thread")

            assert state is not None
            assert state["values"]["iteration"] == 5
            assert state["is_paused"] is True
            assert state["paused_before"] == "execute_test"

    @pytest.mark.asyncio
    async def test_update_state(self):
        """Should update state for thread."""
        from src.orchestrator.graph import TestingOrchestrator

        with patch("src.orchestrator.graph.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            orchestrator = TestingOrchestrator(
                codebase_path="/test/app",
                app_url="http://localhost:3000"
            )

            orchestrator.app.aupdate_state = AsyncMock()

            result = await orchestrator.update_state("test-thread", {"should_continue": False})

            assert result is True
            orchestrator.app.aupdate_state.assert_called_once()


class TestInterruptNodes:
    """Tests for interrupt_before configuration."""

    def test_no_interrupts_by_default(self):
        """Should have no interrupts by default."""
        from src.orchestrator.graph import get_interrupt_nodes

        settings = MagicMock()
        settings.require_healing_approval = False
        settings.require_human_approval_for_healing = False
        settings.require_test_plan_approval = False

        nodes = get_interrupt_nodes(settings)
        assert nodes == []

    def test_healing_approval_interrupt(self):
        """Should add self_heal interrupt when healing approval required."""
        from src.orchestrator.graph import get_interrupt_nodes

        settings = MagicMock()
        settings.require_healing_approval = True
        settings.require_human_approval_for_healing = False
        settings.require_test_plan_approval = False

        nodes = get_interrupt_nodes(settings)
        assert "self_heal" in nodes

    def test_test_plan_approval_interrupt(self):
        """Should add execute_test interrupt when plan approval required."""
        from src.orchestrator.graph import get_interrupt_nodes

        settings = MagicMock()
        settings.require_healing_approval = False
        settings.require_human_approval_for_healing = False
        settings.require_test_plan_approval = True

        nodes = get_interrupt_nodes(settings)
        assert "execute_test" in nodes

    def test_multiple_interrupts(self):
        """Should support multiple interrupts."""
        from src.orchestrator.graph import get_interrupt_nodes

        settings = MagicMock()
        settings.require_healing_approval = True
        settings.require_human_approval_for_healing = False
        settings.require_test_plan_approval = True

        nodes = get_interrupt_nodes(settings)
        assert "self_heal" in nodes
        assert "execute_test" in nodes


class TestStateModels:
    """Tests for state dataclasses and models."""

    def test_test_spec_to_dict(self):
        """Should convert TestSpec to dict."""
        from src.orchestrator.state import TestSpec, TestType, Priority

        spec = TestSpec(
            id="test-123",
            name="Login Test",
            type=TestType.UI,
            priority=Priority.HIGH,
            steps=[{"action": "click", "target": "#button"}],
            assertions=[{"type": "visible", "target": "#result"}],
        )

        result = spec.to_dict()

        assert result["id"] == "test-123"
        assert result["name"] == "Login Test"
        assert result["type"] == "ui"
        assert result["priority"] == "high"

    def test_test_result_to_dict(self):
        """Should convert TestResult to dict."""
        from src.orchestrator.state import TestResult, TestStatus

        result = TestResult(
            test_id="test-123",
            status=TestStatus.PASSED,
            duration_seconds=5.5,
            assertions_passed=3,
            assertions_failed=0,
        )

        result_dict = result.to_dict()

        assert result_dict["test_id"] == "test-123"
        assert result_dict["status"] == "passed"
        assert result_dict["duration_seconds"] == 5.5

    def test_failure_analysis_to_dict(self):
        """Should convert FailureAnalysis to dict."""
        from src.orchestrator.state import FailureAnalysis

        analysis = FailureAnalysis(
            test_id="test-123",
            failure_type="selector_changed",
            root_cause="Button ID changed from #old to #new",
            confidence=0.95,
        )

        result = analysis.to_dict()

        assert result["test_id"] == "test-123"
        assert result["failure_type"] == "selector_changed"
        assert result["confidence"] == 0.95
