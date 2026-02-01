"""
Planning Middleware for LangChain Deep Agents Pattern.

Implements the TodoListMiddleware pattern for autonomous task decomposition
and iterative execution with state persistence.

RAP-245: Deep Agent Planning Pattern Implementation

Key Features:
- Automatic task decomposition based on complexity
- Todo list with dependencies and priorities
- State persistence to Supabase for long-running tasks
- Adaptive replanning based on execution results
- Integration with LangGraph supervisor

References:
- LangChain Deep Agents: TodoList pattern for multi-step tasks
- ReAct Pattern: Reason + Act iteratively
- Plan-and-Execute: Decompose then execute
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from ..agents.base import BaseAgent
from ..core.model_router import TaskType

logger = structlog.get_logger()


class TodoStatus(str, Enum):
    """Status of a todo item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class TodoItem:
    """A single todo item in the plan.

    Represents one step in a multi-step task with dependencies,
    priority, and execution tracking.
    """
    id: str
    description: str
    status: TodoStatus = TodoStatus.PENDING
    priority: int = 3  # 1 (highest) to 5 (lowest)
    dependencies: list[str] = field(default_factory=list)  # IDs of blocking items
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    started_at: datetime | None = None
    notes: str | None = None
    subtasks: list["TodoItem"] = field(default_factory=list)
    agent_type: str | None = None  # Preferred agent for this task
    estimated_duration_ms: int | None = None
    actual_duration_ms: int | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        description: str,
        priority: int = 3,
        dependencies: list[str] | None = None,
        agent_type: str | None = None,
        **kwargs,
    ) -> "TodoItem":
        """Factory method to create a new TodoItem with auto-generated ID."""
        return cls(
            id=f"todo_{uuid4().hex[:12]}",
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            agent_type=agent_type,
            **kwargs,
        )

    def start(self) -> None:
        """Mark item as in progress."""
        self.status = TodoStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()

    def complete(self, result: dict[str, Any] | None = None) -> None:
        """Mark item as completed."""
        self.status = TodoStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
        if self.started_at:
            self.actual_duration_ms = int(
                (self.completed_at - self.started_at).total_seconds() * 1000
            )

    def fail(self, error: str) -> None:
        """Mark item as failed."""
        self.status = TodoStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()

    def block(self, reason: str | None = None) -> None:
        """Mark item as blocked."""
        self.status = TodoStatus.BLOCKED
        if reason:
            self.notes = reason

    def skip(self, reason: str | None = None) -> None:
        """Mark item as skipped."""
        self.status = TodoStatus.SKIPPED
        if reason:
            self.notes = reason

    def is_actionable(self, completed_ids: set[str]) -> bool:
        """Check if this item can be executed (all dependencies met)."""
        if self.status != TodoStatus.PENDING:
            return False
        return all(dep_id in completed_ids for dep_id in self.dependencies)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "notes": self.notes,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "agent_type": self.agent_type,
            "estimated_duration_ms": self.estimated_duration_ms,
            "actual_duration_ms": self.actual_duration_ms,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TodoItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            status=TodoStatus(data.get("status", "pending")),
            priority=data.get("priority", 3),
            dependencies=data.get("dependencies", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            notes=data.get("notes"),
            subtasks=[cls.from_dict(s) for s in data.get("subtasks", [])],
            agent_type=data.get("agent_type"),
            estimated_duration_ms=data.get("estimated_duration_ms"),
            actual_duration_ms=data.get("actual_duration_ms"),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TodoList:
    """A complete todo list (plan) for a complex task.

    Contains multiple TodoItems with dependencies, forming a DAG
    of tasks to execute.
    """
    id: str
    name: str
    items: list[TodoItem] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    context: dict[str, Any] = field(default_factory=dict)  # Original task context
    org_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        context: dict[str, Any] | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
    ) -> "TodoList":
        """Factory method to create a new TodoList."""
        return cls(
            id=f"plan_{uuid4().hex[:12]}",
            name=name,
            context=context or {},
            org_id=org_id,
            project_id=project_id,
        )

    def add_item(self, item: TodoItem) -> None:
        """Add a todo item to the list."""
        self.items.append(item)
        self.last_updated = datetime.utcnow()

    def add_subtask(self, parent_id: str, subtask: TodoItem) -> bool:
        """Add a subtask to an existing item."""
        parent = self.get_item(parent_id)
        if parent:
            parent.subtasks.append(subtask)
            self.last_updated = datetime.utcnow()
            return True
        return False

    def get_item(self, item_id: str) -> TodoItem | None:
        """Find an item by ID (searches recursively in subtasks)."""
        def search(items: list[TodoItem]) -> TodoItem | None:
            for item in items:
                if item.id == item_id:
                    return item
                found = search(item.subtasks)
                if found:
                    return found
            return None
        return search(self.items)

    def update_status(self, item_id: str, status: TodoStatus) -> bool:
        """Update the status of an item."""
        item = self.get_item(item_id)
        if item:
            item.status = status
            self.last_updated = datetime.utcnow()
            return True
        return False

    def complete_item(self, item_id: str, result: dict[str, Any] | None = None) -> bool:
        """Mark an item as completed."""
        item = self.get_item(item_id)
        if item:
            item.complete(result)
            self.last_updated = datetime.utcnow()
            return True
        return False

    def fail_item(self, item_id: str, error: str) -> bool:
        """Mark an item as failed."""
        item = self.get_item(item_id)
        if item:
            item.fail(error)
            self.last_updated = datetime.utcnow()
            return True
        return False

    def get_next_actionable(self) -> TodoItem | None:
        """Get the highest-priority actionable item."""
        completed_ids = self._get_completed_ids()
        actionable = [
            item for item in self.items
            if item.is_actionable(completed_ids)
        ]
        if not actionable:
            return None
        # Sort by priority (lower = higher priority)
        actionable.sort(key=lambda x: x.priority)
        return actionable[0]

    def get_all_actionable(self) -> list[TodoItem]:
        """Get all actionable items (for parallel execution)."""
        completed_ids = self._get_completed_ids()
        return [
            item for item in self.items
            if item.is_actionable(completed_ids)
        ]

    def get_blocked_items(self) -> list[TodoItem]:
        """Get all blocked items."""
        return [item for item in self.items if item.status == TodoStatus.BLOCKED]

    def get_in_progress_items(self) -> list[TodoItem]:
        """Get all in-progress items."""
        return [item for item in self.items if item.status == TodoStatus.IN_PROGRESS]

    def get_pending_items(self) -> list[TodoItem]:
        """Get all pending items."""
        return [item for item in self.items if item.status == TodoStatus.PENDING]

    def _get_completed_ids(self) -> set[str]:
        """Get IDs of all completed items."""
        def collect(items: list[TodoItem]) -> set[str]:
            ids = set()
            for item in items:
                if item.status == TodoStatus.COMPLETED:
                    ids.add(item.id)
                ids.update(collect(item.subtasks))
            return ids
        return collect(self.items)

    @property
    def progress(self) -> dict[str, int]:
        """Get progress statistics."""
        def count_statuses(items: list[TodoItem]) -> dict[str, int]:
            counts = {s.value: 0 for s in TodoStatus}
            for item in items:
                counts[item.status.value] += 1
                for subtask in item.subtasks:
                    subtask_counts = count_statuses([subtask])
                    for k, v in subtask_counts.items():
                        counts[k] += v
            return counts

        counts = count_statuses(self.items)
        total = sum(counts.values())
        return {
            "total": total,
            "completed": counts["completed"],
            "failed": counts["failed"],
            "pending": counts["pending"],
            "in_progress": counts["in_progress"],
            "blocked": counts["blocked"],
            "skipped": counts["skipped"],
            "completion_pct": round(counts["completed"] / total * 100, 1) if total > 0 else 0,
        }

    @property
    def is_complete(self) -> bool:
        """Check if all items are done (completed, failed, or skipped)."""
        def all_done(items: list[TodoItem]) -> bool:
            for item in items:
                if item.status in [TodoStatus.PENDING, TodoStatus.IN_PROGRESS, TodoStatus.BLOCKED]:
                    return False
                if not all_done(item.subtasks):
                    return False
            return True
        return all_done(self.items)

    def to_markdown(self) -> str:
        """Generate a markdown representation of the plan."""
        lines = [f"# {self.name}", "", f"**Plan ID:** {self.id}", ""]

        progress = self.progress
        lines.append(f"**Progress:** {progress['completed']}/{progress['total']} ({progress['completion_pct']}%)")
        lines.append("")

        status_icons = {
            TodoStatus.PENDING: "⬜",
            TodoStatus.IN_PROGRESS: "🔄",
            TodoStatus.COMPLETED: "✅",
            TodoStatus.BLOCKED: "🚫",
            TodoStatus.SKIPPED: "⏭️",
            TodoStatus.FAILED: "❌",
        }

        def format_item(item: TodoItem, indent: int = 0) -> list[str]:
            prefix = "  " * indent
            icon = status_icons.get(item.status, "⬜")
            priority_str = f"P{item.priority}" if item.priority != 3 else ""
            deps_str = f" (deps: {', '.join(item.dependencies)})" if item.dependencies else ""

            item_lines = [f"{prefix}- {icon} **{item.description}** {priority_str}{deps_str}"]

            if item.notes:
                item_lines.append(f"{prefix}  _Note: {item.notes}_")
            if item.error:
                item_lines.append(f"{prefix}  _Error: {item.error}_")

            for subtask in item.subtasks:
                item_lines.extend(format_item(subtask, indent + 1))

            return item_lines

        lines.append("## Tasks")
        for item in self.items:
            lines.extend(format_item(item))

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "items": [item.to_dict() for item in self.items],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "context": self.context,
            "org_id": self.org_id,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TodoList":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            items=[TodoItem.from_dict(i) for i in data.get("items", [])],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.utcnow(),
            context=data.get("context", {}),
            org_id=data.get("org_id"),
            project_id=data.get("project_id"),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionResult:
    """Result of executing a plan."""
    success: bool
    todo_list: TodoList
    results: dict[str, Any] = field(default_factory=dict)  # Accumulated results
    error: str | None = None
    duration_ms: int = 0
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0


def get_planning_tools() -> list[dict[str, Any]]:
    """Get planning tool definitions for Claude.

    Returns tool definitions that can be passed to the Claude API
    for agents that support planning.
    """
    return [
        {
            "name": "write_todos",
            "description": "Create a todo list to plan and track a multi-step task. Use this when facing a complex task that requires multiple steps.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "plan_name": {
                        "type": "string",
                        "description": "A descriptive name for this plan",
                    },
                    "items": {
                        "type": "array",
                        "description": "List of todo items to add",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {
                                    "type": "string",
                                    "description": "What needs to be done",
                                },
                                "priority": {
                                    "type": "integer",
                                    "description": "Priority 1-5 (1 = highest)",
                                    "default": 3,
                                },
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "Indices of items this depends on",
                                    "default": [],
                                },
                                "agent_type": {
                                    "type": "string",
                                    "description": "Preferred agent (code_analyzer, ui_tester, etc.)",
                                },
                            },
                            "required": ["description"],
                        },
                    },
                },
                "required": ["plan_name", "items"],
            },
        },
        {
            "name": "update_todo_status",
            "description": "Update the status of a todo item",
            "input_schema": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "ID of the todo item",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["completed", "failed", "blocked", "skipped"],
                        "description": "New status for the item",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional notes about the status change",
                    },
                    "result": {
                        "type": "object",
                        "description": "Result data if completed",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if failed",
                    },
                },
                "required": ["item_id", "status"],
            },
        },
        {
            "name": "get_next_todo",
            "description": "Get the next actionable todo item",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "get_plan_progress",
            "description": "Get the current progress of the plan",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "add_todo_item",
            "description": "Add a new todo item to the plan",
            "input_schema": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What needs to be done",
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority 1-5 (1 = highest)",
                        "default": 3,
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of items this depends on",
                        "default": [],
                    },
                    "parent_id": {
                        "type": "string",
                        "description": "ID of parent item (for subtasks)",
                    },
                },
                "required": ["description"],
            },
        },
    ]


class PlanningToolHandler:
    """Handles planning tool calls from agents."""

    def __init__(self, todo_list: TodoList | None = None):
        self.todo_list = todo_list
        self.log = logger.bind(component="PlanningToolHandler")

    def handle_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Process a planning tool call and return the result."""
        if tool_name == "write_todos":
            return self._handle_write_todos(tool_input)
        elif tool_name == "update_todo_status":
            return self._handle_update_status(tool_input)
        elif tool_name == "get_next_todo":
            return self._handle_get_next()
        elif tool_name == "get_plan_progress":
            return self._handle_get_progress()
        elif tool_name == "add_todo_item":
            return self._handle_add_item(tool_input)
        else:
            return {"error": f"Unknown planning tool: {tool_name}"}

    def _handle_write_todos(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new todo list."""
        plan_name = input_data.get("plan_name", "Unnamed Plan")
        items_data = input_data.get("items", [])

        self.todo_list = TodoList.create(name=plan_name)

        # Create items and track for dependency resolution
        item_id_map: dict[int, str] = {}

        for i, item_data in enumerate(items_data):
            item = TodoItem.create(
                description=item_data["description"],
                priority=item_data.get("priority", 3),
                agent_type=item_data.get("agent_type"),
            )
            item_id_map[i] = item.id
            self.todo_list.add_item(item)

        # Resolve dependencies (convert indices to IDs)
        for i, item_data in enumerate(items_data):
            dep_indices = item_data.get("dependencies", [])
            item = self.todo_list.items[i]
            item.dependencies = [item_id_map[idx] for idx in dep_indices if idx in item_id_map]

        self.log.info(
            "Created todo list",
            plan_id=self.todo_list.id,
            item_count=len(self.todo_list.items),
        )

        return {
            "success": True,
            "plan_id": self.todo_list.id,
            "plan_name": plan_name,
            "items_created": len(self.todo_list.items),
            "items": [{"id": item.id, "description": item.description} for item in self.todo_list.items],
        }

    def _handle_update_status(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Update a todo item's status."""
        if not self.todo_list:
            return {"error": "No active plan"}

        item_id = input_data.get("item_id")
        status = input_data.get("status")
        notes = input_data.get("notes")
        result = input_data.get("result")
        error = input_data.get("error")

        item = self.todo_list.get_item(item_id)
        if not item:
            return {"error": f"Item not found: {item_id}"}

        if status == "completed":
            item.complete(result)
        elif status == "failed":
            item.fail(error or "Unknown error")
        elif status == "blocked":
            item.block(notes)
        elif status == "skipped":
            item.skip(notes)
        else:
            return {"error": f"Invalid status: {status}"}

        return {
            "success": True,
            "item_id": item_id,
            "new_status": status,
        }

    def _handle_get_next(self) -> dict[str, Any]:
        """Get the next actionable item."""
        if not self.todo_list:
            return {"error": "No active plan"}

        next_item = self.todo_list.get_next_actionable()
        if not next_item:
            if self.todo_list.is_complete:
                return {"message": "Plan complete", "complete": True}
            else:
                return {"message": "No actionable items (dependencies not met)", "blocked": True}

        return {
            "item_id": next_item.id,
            "description": next_item.description,
            "priority": next_item.priority,
            "agent_type": next_item.agent_type,
            "dependencies": next_item.dependencies,
        }

    def _handle_get_progress(self) -> dict[str, Any]:
        """Get plan progress."""
        if not self.todo_list:
            return {"error": "No active plan"}

        progress = self.todo_list.progress
        return {
            "plan_id": self.todo_list.id,
            "plan_name": self.todo_list.name,
            "progress": progress,
            "is_complete": self.todo_list.is_complete,
        }

    def _handle_add_item(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Add a new item to the plan."""
        if not self.todo_list:
            return {"error": "No active plan"}

        item = TodoItem.create(
            description=input_data["description"],
            priority=input_data.get("priority", 3),
            dependencies=input_data.get("dependencies", []),
        )

        parent_id = input_data.get("parent_id")
        if parent_id:
            success = self.todo_list.add_subtask(parent_id, item)
            if not success:
                return {"error": f"Parent not found: {parent_id}"}
        else:
            self.todo_list.add_item(item)

        return {
            "success": True,
            "item_id": item.id,
            "description": item.description,
        }


class PlanningMiddleware:
    """
    Middleware that adds planning capabilities to any BaseAgent.

    Wraps an agent to automatically:
    1. Analyze task complexity
    2. Generate a plan for complex tasks
    3. Execute items in dependency order
    4. Adapt the plan based on results
    5. Persist state for recovery

    Usage:
        ```python
        from src.agents.code_analyzer import CodeAnalyzerAgent
        from src.orchestrator.planning_middleware import PlanningMiddleware

        agent = CodeAnalyzerAgent()
        middleware = PlanningMiddleware(agent, auto_plan_threshold=3)

        result = await middleware.execute_with_planning(
            task="Analyze auth module and generate comprehensive tests",
            context={"codebase_path": "/app"},
            org_id="org_123",
        )
        ```
    """

    def __init__(
        self,
        agent: BaseAgent,
        auto_plan_threshold: int = 3,
        persist_plans: bool = True,
    ):
        """Initialize planning middleware.

        Args:
            agent: The agent to wrap with planning capabilities
            auto_plan_threshold: Number of estimated steps to trigger auto-planning
            persist_plans: Whether to persist plans to database
        """
        self.agent = agent
        self.auto_plan_threshold = auto_plan_threshold
        self.persist_plans = persist_plans
        self.log = logger.bind(
            component="PlanningMiddleware",
            agent=agent.__class__.__name__,
        )

    async def execute_with_planning(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        force_plan: bool = False,
    ) -> ExecutionResult:
        """Execute a task with automatic planning if needed.

        Args:
            task: The task description
            context: Additional context for the task
            org_id: Organization ID for multi-tenancy
            project_id: Project ID for scoping
            force_plan: Always generate a plan, even for simple tasks

        Returns:
            ExecutionResult with plan and accumulated results
        """
        import time
        start_time = time.time()

        context = context or {}

        # Analyze complexity
        complexity = await self._analyze_complexity(task, context)
        self.log.info(
            "Task complexity analyzed",
            task=task[:50],
            estimated_steps=complexity["estimated_steps"],
            complexity_score=complexity["score"],
        )

        # Decide if planning is needed
        needs_planning = force_plan or complexity["estimated_steps"] >= self.auto_plan_threshold

        if not needs_planning:
            # Execute directly without planning
            self.log.info("Executing task directly (no planning needed)")
            result = await self.agent.execute(task=task, **context)
            return ExecutionResult(
                success=result.success,
                todo_list=TodoList.create(name=task, context=context),
                results={"direct_result": result.data},
                error=result.error,
                duration_ms=int((time.time() - start_time) * 1000),
                total_items=1,
                completed_items=1 if result.success else 0,
                failed_items=0 if result.success else 1,
            )

        # Generate plan
        todo_list = await self.generate_plan(task, context, org_id, project_id)

        if self.persist_plans:
            await self._persist_plan(todo_list)

        # Execute plan items iteratively
        accumulated_results: dict[str, Any] = {}
        max_iterations = len(todo_list.items) * 2  # Safety limit

        for iteration in range(max_iterations):
            next_item = todo_list.get_next_actionable()
            if not next_item:
                if todo_list.is_complete:
                    break
                # Check for blocked items
                blocked = todo_list.get_blocked_items()
                if blocked:
                    self.log.warning("Items blocked, cannot proceed", blocked_count=len(blocked))
                    break
                # No actionable items but not complete - deadlock
                self.log.error("Execution deadlock - no actionable items")
                break

            # Execute the item
            item_result = await self._execute_item(next_item, todo_list, accumulated_results)

            if item_result["success"]:
                todo_list.complete_item(next_item.id, item_result.get("result"))
                accumulated_results[next_item.id] = item_result.get("result")
            else:
                todo_list.fail_item(next_item.id, item_result.get("error", "Unknown error"))

            # Persist after each step
            if self.persist_plans:
                await self._persist_plan(todo_list)

            # Check if we should adapt the plan
            adapted = await self._maybe_adapt_plan(todo_list, next_item, item_result)
            if adapted:
                self.log.info("Plan adapted based on execution results")

        progress = todo_list.progress
        return ExecutionResult(
            success=progress["failed"] == 0 and progress["completed"] == progress["total"],
            todo_list=todo_list,
            results=accumulated_results,
            duration_ms=int((time.time() - start_time) * 1000),
            total_items=progress["total"],
            completed_items=progress["completed"],
            failed_items=progress["failed"],
        )

    async def generate_plan(
        self,
        task: str,
        context: dict[str, Any],
        org_id: str | None = None,
        project_id: str | None = None,
    ) -> TodoList:
        """Generate a plan for a complex task using AI."""
        plan_prompt = f"""Break down this task into a step-by-step plan with dependencies.

Task: {task}

Context: {context}

Generate a JSON plan with the following structure:
{{
    "plan_name": "Descriptive name for this plan",
    "items": [
        {{
            "description": "Clear, actionable step description",
            "priority": 1-5 (1 = highest priority),
            "dependencies": [indices of items this depends on],
            "agent_type": "preferred agent type (optional)"
        }}
    ]
}}

Requirements:
1. Break the task into 3-10 concrete steps
2. Each step should be independently executable
3. Dependencies should form a valid DAG (no cycles)
4. Order steps logically with proper dependencies
5. Include analysis, implementation, and verification steps

Return ONLY the JSON, no explanation."""

        try:
            response = await self.agent._call_ai(
                messages=[{"role": "user", "content": plan_prompt}],
                task_type=TaskType.CODE_ANALYSIS,
                json_mode=True,
                max_tokens=2000,
            )

            plan_data = self.agent._parse_json_response(response.content, {
                "plan_name": task[:50],
                "items": [{"description": task, "priority": 3, "dependencies": []}],
            })

            # Create the TodoList
            todo_list = TodoList.create(
                name=plan_data.get("plan_name", task[:50]),
                context=context,
                org_id=org_id,
                project_id=project_id,
            )

            # Track IDs for dependency resolution
            item_id_map: dict[int, str] = {}

            for i, item_data in enumerate(plan_data.get("items", [])):
                item = TodoItem.create(
                    description=item_data.get("description", f"Step {i+1}"),
                    priority=item_data.get("priority", 3),
                    agent_type=item_data.get("agent_type"),
                )
                item_id_map[i] = item.id
                todo_list.add_item(item)

            # Resolve dependencies
            for i, item_data in enumerate(plan_data.get("items", [])):
                dep_indices = item_data.get("dependencies", [])
                item = todo_list.items[i]
                item.dependencies = [item_id_map[idx] for idx in dep_indices if idx in item_id_map]

            self.log.info(
                "Generated plan",
                plan_id=todo_list.id,
                item_count=len(todo_list.items),
            )

            return todo_list

        except Exception as e:
            self.log.error("Plan generation failed", error=str(e))
            # Fallback to single-item plan
            todo_list = TodoList.create(name=task[:50], context=context, org_id=org_id, project_id=project_id)
            todo_list.add_item(TodoItem.create(description=task))
            return todo_list

    async def _analyze_complexity(
        self,
        task: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze task complexity to decide if planning is needed."""
        analysis_prompt = f"""Analyze this task and estimate its complexity.

Task: {task}

Context: {context}

Provide a JSON response:
{{
    "estimated_steps": <number of steps needed, 1-10>,
    "score": <complexity score 0.0-1.0>,
    "reasoning": "<brief explanation>"
}}

Return ONLY the JSON."""

        try:
            response = await self.agent._call_ai(
                messages=[{"role": "user", "content": analysis_prompt}],
                task_type=TaskType.TEXT_EXTRACTION,
                json_mode=True,
                max_tokens=500,
            )

            return self.agent._parse_json_response(response.content, {
                "estimated_steps": 1,
                "score": 0.3,
                "reasoning": "Default estimation",
            })

        except Exception as e:
            self.log.warning("Complexity analysis failed", error=str(e))
            return {"estimated_steps": 1, "score": 0.3, "reasoning": "Analysis failed"}

    async def _execute_item(
        self,
        item: TodoItem,
        todo_list: TodoList,
        accumulated_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single todo item."""
        self.log.info("Executing item", item_id=item.id, description=item.description[:50])

        item.start()

        try:
            # Build context from accumulated results
            execution_context = {
                "task": item.description,
                "plan_context": todo_list.context,
                "previous_results": accumulated_results,
                "dependencies_met": [
                    {"id": dep_id, "result": accumulated_results.get(dep_id)}
                    for dep_id in item.dependencies
                ],
            }

            result = await self.agent.execute(**execution_context)

            return {
                "success": result.success,
                "result": result.data,
                "error": result.error,
            }

        except Exception as e:
            self.log.error("Item execution failed", item_id=item.id, error=str(e))
            return {"success": False, "error": str(e)}

    async def _maybe_adapt_plan(
        self,
        todo_list: TodoList,
        completed_item: TodoItem,
        result: dict[str, Any],
    ) -> bool:
        """Check if the plan needs adaptation based on results."""
        # Simple heuristic: don't adapt on success
        if result.get("success"):
            return False

        # On failure, check if we should skip dependent items
        for item in todo_list.items:
            if completed_item.id in item.dependencies:
                if item.status == TodoStatus.PENDING:
                    item.block(f"Dependency {completed_item.id} failed")

        return True

    async def _persist_plan(self, todo_list: TodoList) -> None:
        """Persist plan to database."""
        try:
            from ..services.supabase_client import get_supabase_client
            client = get_supabase_client()

            data = todo_list.to_dict()

            # Upsert the plan
            await client.table("agent_plans").upsert({
                "id": todo_list.id,
                "name": todo_list.name,
                "items": data["items"],
                "context": data["context"],
                "org_id": todo_list.org_id,
                "project_id": todo_list.project_id,
                "session_id": todo_list.session_id,
                "metadata": data["metadata"],
                "progress": data["progress"],
                "updated_at": datetime.utcnow().isoformat(),
            }).execute()

            self.log.debug("Plan persisted", plan_id=todo_list.id)

        except Exception as e:
            self.log.warning("Failed to persist plan", error=str(e))

    async def load_plan(self, plan_id: str) -> TodoList | None:
        """Load a plan from database."""
        try:
            from ..services.supabase_client import get_supabase_client
            client = get_supabase_client()

            result = await client.table("agent_plans").select("*").eq("id", plan_id).single().execute()

            if result.data:
                return TodoList.from_dict(result.data)
            return None

        except Exception as e:
            self.log.warning("Failed to load plan", plan_id=plan_id, error=str(e))
            return None


def planning_middleware_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node for planning middleware integration.

    Use in supervisor graphs to enable planning across agent handoffs.
    """
    # Restore plan from state if exists
    active_plan = state.get("active_plan")
    if active_plan:
        todo_list = TodoList.from_dict(active_plan)
        return {
            "active_plan": todo_list.to_dict(),
            "plan_progress": todo_list.progress,
        }

    return state
