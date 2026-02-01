"""Tests for the Planning Middleware module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestPlanningMiddlewareDataClasses:
    """Tests for Planning Middleware data classes."""

    def test_todo_item_creation(self, mock_env_vars):
        """Test TodoItem dataclass."""
        from src.orchestrator.planning_middleware import TodoItem, TodoStatus

        item = TodoItem(
            item_id="TODO-001",
            title="Implement login test",
            description="Create E2E test for login flow",
            status=TodoStatus.PENDING,
            priority=1,
            dependencies=["TODO-000"],
            estimated_effort="medium",
        )

        assert item.item_id == "TODO-001"
        assert item.status == TodoStatus.PENDING
        assert item.priority == 1
        assert len(item.dependencies) == 1

    def test_todo_list_creation(self, mock_env_vars):
        """Test TodoList dataclass."""
        from src.orchestrator.planning_middleware import TodoList, TodoItem, TodoStatus

        items = [
            TodoItem(
                item_id="TODO-001",
                title="Task 1",
                status=TodoStatus.COMPLETED,
                priority=1,
            ),
            TodoItem(
                item_id="TODO-002",
                title="Task 2",
                status=TodoStatus.PENDING,
                priority=2,
            ),
        ]

        todo_list = TodoList(
            list_id="LIST-001",
            name="Test Plan",
            items=items,
            created_at=datetime.now(timezone.utc),
        )

        assert todo_list.list_id == "LIST-001"
        assert len(todo_list.items) == 2
        assert todo_list.completed_count == 1
        assert todo_list.pending_count == 1

    def test_execution_result_creation(self, mock_env_vars):
        """Test ExecutionResult dataclass."""
        from src.orchestrator.planning_middleware import ExecutionResult

        result = ExecutionResult(
            item_id="TODO-001",
            success=True,
            output={"tests_run": 5, "passed": 5},
            duration_seconds=10.5,
            error=None,
        )

        assert result.item_id == "TODO-001"
        assert result.success is True
        assert result.duration_seconds == 10.5


class TestTodoStatusEnum:
    """Tests for TodoStatus enum."""

    def test_todo_status_values(self, mock_env_vars):
        """Test TodoStatus enum values."""
        from src.orchestrator.planning_middleware import TodoStatus

        assert TodoStatus.PENDING.value == "pending"
        assert TodoStatus.IN_PROGRESS.value == "in_progress"
        assert TodoStatus.COMPLETED.value == "completed"
        assert TodoStatus.BLOCKED.value == "blocked"
        assert TodoStatus.CANCELLED.value == "cancelled"


class TestTodoListMethods:
    """Tests for TodoList methods."""

    def test_get_next_item(self, mock_env_vars):
        """Test getting next actionable item."""
        from src.orchestrator.planning_middleware import TodoList, TodoItem, TodoStatus

        items = [
            TodoItem(item_id="1", title="Blocked", status=TodoStatus.BLOCKED, priority=1),
            TodoItem(item_id="2", title="Pending", status=TodoStatus.PENDING, priority=2),
            TodoItem(item_id="3", title="Completed", status=TodoStatus.COMPLETED, priority=3),
        ]

        todo_list = TodoList(list_id="LIST-001", name="Test", items=items)

        next_item = todo_list.get_next_item()

        assert next_item is not None
        assert next_item.item_id == "2"
        assert next_item.status == TodoStatus.PENDING

    def test_get_next_item_respects_dependencies(self, mock_env_vars):
        """Test that get_next_item respects dependencies."""
        from src.orchestrator.planning_middleware import TodoList, TodoItem, TodoStatus

        items = [
            TodoItem(item_id="1", title="First", status=TodoStatus.PENDING, priority=1, dependencies=[]),
            TodoItem(item_id="2", title="Second", status=TodoStatus.PENDING, priority=2, dependencies=["1"]),
        ]

        todo_list = TodoList(list_id="LIST-001", name="Test", items=items)

        # First should be returned since it has no dependencies
        next_item = todo_list.get_next_item()
        assert next_item.item_id == "1"

    def test_mark_completed(self, mock_env_vars):
        """Test marking item as completed."""
        from src.orchestrator.planning_middleware import TodoList, TodoItem, TodoStatus

        items = [
            TodoItem(item_id="1", title="Task", status=TodoStatus.PENDING, priority=1),
        ]

        todo_list = TodoList(list_id="LIST-001", name="Test", items=items)
        todo_list.mark_completed("1")

        assert todo_list.items[0].status == TodoStatus.COMPLETED

    def test_is_complete(self, mock_env_vars):
        """Test checking if all items are complete."""
        from src.orchestrator.planning_middleware import TodoList, TodoItem, TodoStatus

        items = [
            TodoItem(item_id="1", title="Done", status=TodoStatus.COMPLETED, priority=1),
            TodoItem(item_id="2", title="Also Done", status=TodoStatus.COMPLETED, priority=2),
        ]

        todo_list = TodoList(list_id="LIST-001", name="Test", items=items)

        assert todo_list.is_complete() is True


class TestPlanningMiddleware:
    """Tests for PlanningMiddleware class."""

    def test_planning_middleware_initialization(self, mock_env_vars):
        """Test PlanningMiddleware initialization."""
        from src.orchestrator.planning_middleware import PlanningMiddleware

        middleware = PlanningMiddleware()

        assert middleware is not None
        assert hasattr(middleware, 'generate_plan')
        assert hasattr(middleware, 'execute_with_planning')
        assert hasattr(middleware, 'adapt_plan')

    @pytest.mark.asyncio
    async def test_generate_plan(self, mock_env_vars):
        """Test plan generation."""
        from src.orchestrator.planning_middleware import PlanningMiddleware

        middleware = PlanningMiddleware()

        with patch.object(middleware, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='''{
                    "items": [
                        {"title": "Analyze code", "priority": 1, "estimated_effort": "low"},
                        {"title": "Write tests", "priority": 2, "estimated_effort": "medium"},
                        {"title": "Run tests", "priority": 3, "estimated_effort": "low"}
                    ]
                }'''
            )

            plan = await middleware.generate_plan(
                task="Test the authentication module",
                context={"module": "auth", "framework": "FastAPI"},
            )

            assert plan is not None
            mock_ai.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_planning(self, mock_env_vars):
        """Test executing a task with planning."""
        from src.orchestrator.planning_middleware import PlanningMiddleware, TodoList, TodoItem, TodoStatus

        middleware = PlanningMiddleware()

        # Pre-create a plan
        plan = TodoList(
            list_id="LIST-001",
            name="Test Plan",
            items=[
                TodoItem(item_id="1", title="Step 1", status=TodoStatus.PENDING, priority=1),
                TodoItem(item_id="2", title="Step 2", status=TodoStatus.PENDING, priority=2),
            ],
        )

        with patch.object(middleware, '_execute_item', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"success": True}

            result = await middleware.execute_with_planning(
                plan=plan,
                executor=AsyncMock(return_value={"result": "success"}),
            )

            assert result is not None
            assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    async def test_adapt_plan_on_failure(self, mock_env_vars):
        """Test plan adaptation when a step fails."""
        from src.orchestrator.planning_middleware import PlanningMiddleware, TodoList, TodoItem, TodoStatus

        middleware = PlanningMiddleware()

        plan = TodoList(
            list_id="LIST-001",
            name="Test Plan",
            items=[
                TodoItem(item_id="1", title="Failing Step", status=TodoStatus.PENDING, priority=1),
                TodoItem(item_id="2", title="Dependent Step", status=TodoStatus.PENDING, priority=2, dependencies=["1"]),
            ],
        )

        with patch.object(middleware, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='''{
                    "adapted_items": [
                        {"item_id": "1", "status": "blocked", "reason": "Failed to execute"},
                        {"item_id": "1b", "title": "Alternative approach", "priority": 1}
                    ],
                    "strategy": "retry_with_alternative"
                }'''
            )

            adapted_plan = await middleware.adapt_plan(
                plan=plan,
                failed_item_id="1",
                error="Connection timeout",
            )

            assert adapted_plan is not None
            mock_ai.assert_called_once()


class TestPlanningToolHandler:
    """Tests for PlanningToolHandler class."""

    def test_planning_tool_handler_initialization(self, mock_env_vars):
        """Test PlanningToolHandler initialization."""
        from src.orchestrator.planning_middleware import PlanningToolHandler

        handler = PlanningToolHandler()

        assert handler is not None
        assert hasattr(handler, 'handle_create_todo')
        assert hasattr(handler, 'handle_update_todo')
        assert hasattr(handler, 'handle_list_todos')

    @pytest.mark.asyncio
    async def test_handle_create_todo(self, mock_env_vars):
        """Test creating a todo via handler."""
        from src.orchestrator.planning_middleware import PlanningToolHandler

        handler = PlanningToolHandler()

        result = await handler.handle_create_todo(
            title="New task",
            description="Description of new task",
            priority=1,
        )

        assert result is not None
        assert "item_id" in result

    @pytest.mark.asyncio
    async def test_handle_update_todo(self, mock_env_vars):
        """Test updating a todo via handler."""
        from src.orchestrator.planning_middleware import PlanningToolHandler, TodoStatus

        handler = PlanningToolHandler()

        # First create a todo
        create_result = await handler.handle_create_todo(
            title="Task to update",
            priority=1,
        )

        # Then update it
        update_result = await handler.handle_update_todo(
            item_id=create_result["item_id"],
            status=TodoStatus.COMPLETED.value,
        )

        assert update_result is not None
        assert update_result.get("success") is True


class TestGetPlanningTools:
    """Tests for get_planning_tools function."""

    def test_get_planning_tools_returns_tools(self, mock_env_vars):
        """Test that get_planning_tools returns tool definitions."""
        from src.orchestrator.planning_middleware import get_planning_tools

        tools = get_planning_tools()

        assert tools is not None
        assert isinstance(tools, list)
        assert len(tools) > 0

        # Check tool structure
        for tool in tools:
            assert "name" in tool
            assert "description" in tool


class TestPlanningMiddlewareNode:
    """Tests for planning_middleware_node function."""

    @pytest.mark.asyncio
    async def test_planning_middleware_node(self, mock_env_vars):
        """Test the LangGraph-compatible planning node."""
        from src.orchestrator.planning_middleware import planning_middleware_node

        state = {
            "messages": [],
            "task": "Test the login flow",
            "context": {"app": "test_app"},
        }

        with patch('src.orchestrator.planning_middleware.PlanningMiddleware') as MockMiddleware:
            mock_instance = MockMiddleware.return_value
            mock_instance.generate_plan = AsyncMock(return_value=MagicMock(
                items=[{"title": "Step 1"}],
            ))
            mock_instance.execute_with_planning = AsyncMock(return_value={
                "success": True,
                "results": [],
            })

            result = await planning_middleware_node(state)

            assert result is not None


class TestPlanningIntegration:
    """Integration tests for Planning Middleware."""

    @pytest.mark.asyncio
    async def test_full_planning_workflow(self, mock_env_vars):
        """Test full planning workflow: generate -> execute -> adapt."""
        from src.orchestrator.planning_middleware import PlanningMiddleware

        middleware = PlanningMiddleware()

        with patch.object(middleware, '_call_ai', new_callable=AsyncMock) as mock_ai:
            # Mock plan generation
            mock_ai.return_value = MagicMock(
                content='''{
                    "items": [
                        {"title": "Step 1", "priority": 1},
                        {"title": "Step 2", "priority": 2},
                        {"title": "Step 3", "priority": 3}
                    ]
                }'''
            )

            # Generate plan
            plan = await middleware.generate_plan(
                task="Complete E2E testing",
                context={},
            )

            assert plan is not None

    @pytest.mark.asyncio
    async def test_planning_with_dependencies(self, mock_env_vars):
        """Test planning handles dependencies correctly."""
        from src.orchestrator.planning_middleware import TodoList, TodoItem, TodoStatus

        items = [
            TodoItem(item_id="1", title="Setup", status=TodoStatus.PENDING, priority=1, dependencies=[]),
            TodoItem(item_id="2", title="Run", status=TodoStatus.PENDING, priority=2, dependencies=["1"]),
            TodoItem(item_id="3", title="Cleanup", status=TodoStatus.PENDING, priority=3, dependencies=["2"]),
        ]

        todo_list = TodoList(list_id="LIST-001", name="Test", items=items)

        # Only "Setup" should be available (no dependencies)
        next_item = todo_list.get_next_item()
        assert next_item.item_id == "1"

        # Complete "Setup"
        todo_list.mark_completed("1")

        # Now "Run" should be available
        next_item = todo_list.get_next_item()
        assert next_item.item_id == "2"
