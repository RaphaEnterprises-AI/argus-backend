"""Task Decomposer for breaking complex tasks into executable subtasks.

RAP-232: Uses Claude to analyze natural language task descriptions and
decompose them into structured TaskDefinitions with:
- Proper capability requirements
- Dependency relationships
- Input/output schemas

This enables users to describe high-level goals which are automatically
broken down into executable steps.

Example usage:
    decomposer = TaskDecomposer()

    # Decompose a complex task
    subtasks = await decomposer.decompose(
        "Test the complete user registration flow including email verification"
    )

    # Each subtask has proper structure
    for task in subtasks:
        print(f"{task.task_id}: {task.description}")
        print(f"  Capabilities: {task.required_capabilities}")
        print(f"  Depends on: {task.depends_on}")
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import structlog

from ..config import get_settings

logger = structlog.get_logger()


# =============================================================================
# Capability Mappings
# =============================================================================

# Map keywords to capabilities
CAPABILITY_KEYWORDS = {
    # Code analysis
    "analyze": ["code_analysis"],
    "parse": ["code_analysis"],
    "understand": ["code_analysis"],
    "extract": ["code_analysis"],
    "scan": ["code_analysis"],

    # UI Testing
    "click": ["ui_testing"],
    "type": ["ui_testing"],
    "navigate": ["ui_testing"],
    "fill": ["ui_testing"],
    "submit": ["ui_testing"],
    "form": ["ui_testing"],
    "button": ["ui_testing"],
    "browser": ["ui_testing"],
    "screenshot": ["ui_testing", "vision"],

    # API Testing
    "api": ["api_testing"],
    "endpoint": ["api_testing"],
    "request": ["api_testing"],
    "response": ["api_testing"],
    "rest": ["api_testing"],
    "graphql": ["api_testing"],
    "http": ["api_testing"],

    # Database
    "database": ["db_testing"],
    "query": ["db_testing"],
    "sql": ["db_testing"],
    "table": ["db_testing"],
    "record": ["db_testing"],

    # Visual/Vision
    "visual": ["vision"],
    "image": ["vision"],
    "compare": ["vision"],
    "screenshot": ["vision"],
    "look": ["vision"],

    # Performance
    "performance": ["performance"],
    "speed": ["performance"],
    "load": ["performance"],
    "metrics": ["performance"],

    # Security
    "security": ["security"],
    "vulnerability": ["security"],
    "injection": ["security"],
    "xss": ["security"],

    # Accessibility
    "accessibility": ["accessibility"],
    "a11y": ["accessibility"],
    "wcag": ["accessibility"],

    # Reporting
    "report": ["reporting"],
    "summary": ["reporting"],
    "document": ["reporting"],
}

# Ordered phases for automatic dependency inference
TASK_PHASES = [
    "analysis",
    "planning",
    "setup",
    "execution",
    "verification",
    "cleanup",
    "reporting",
]

PHASE_KEYWORDS = {
    "analysis": ["analyze", "understand", "scan", "discover", "identify"],
    "planning": ["plan", "design", "prepare", "determine"],
    "setup": ["setup", "configure", "initialize", "create", "start"],
    "execution": ["run", "execute", "test", "perform", "click", "navigate", "submit"],
    "verification": ["verify", "check", "assert", "validate", "confirm"],
    "cleanup": ["cleanup", "teardown", "reset", "restore", "delete"],
    "reporting": ["report", "summarize", "document", "output"],
}


@dataclass
class DecompositionConfig:
    """Configuration for task decomposition."""
    max_subtasks: int = 10
    min_subtasks: int = 1
    infer_dependencies: bool = True
    validate_capabilities: bool = True
    model: str = "claude-sonnet-4-5"
    temperature: float = 0.3


# Import TaskDefinition and TaskPriority from workflow_composer
# to avoid circular imports, define locally if needed
@dataclass
class TaskDefinition:
    """Definition of a task to be composed into a workflow."""
    task_id: str
    description: str
    required_capabilities: list[str]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    priority: int = 2  # MEDIUM
    timeout_seconds: int = 300
    retry_count: int = 1
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskDecomposer:
    """Breaks complex tasks into executable subtasks.

    Uses Claude to analyze natural language task descriptions and
    decompose them into structured TaskDefinitions with proper
    dependencies and capability requirements.
    """

    def __init__(self, config: DecompositionConfig | None = None):
        """Initialize task decomposer.

        Args:
            config: Optional decomposition configuration
        """
        self.config = config or DecompositionConfig()
        self.settings = get_settings()
        self.log = structlog.get_logger().bind(component="task_decomposer")

    async def decompose(self, task_description: str) -> list[TaskDefinition]:
        """Use Claude to analyze task and identify subtasks.

        Args:
            task_description: Natural language description of the task

        Returns:
            List of TaskDefinition subtasks

        Example:
            >>> subtasks = await decomposer.decompose(
            ...     "Test user registration with email verification"
            ... )
            >>> len(subtasks)
            5
            >>> subtasks[0].description
            'Navigate to registration page'
        """
        self.log.info("Decomposing task", task=task_description[:100])

        # Build prompt for Claude
        prompt = self._build_decomposition_prompt(task_description)

        # Call Claude
        try:
            subtasks_json = await self._call_llm(prompt)
            subtasks = self._parse_subtasks(subtasks_json, task_description)
        except Exception as e:
            self.log.warning(
                "LLM decomposition failed, using heuristic fallback",
                error=str(e),
            )
            subtasks = self._heuristic_decompose(task_description)

        # Infer dependencies if enabled
        if self.config.infer_dependencies:
            subtasks = self._infer_dependencies(subtasks)

        # Validate capabilities
        if self.config.validate_capabilities:
            for task in subtasks:
                if not task.required_capabilities:
                    task.required_capabilities = self.identify_capabilities(
                        task.description
                    )

        self.log.info(
            "Task decomposed",
            subtask_count=len(subtasks),
            subtasks=[t.task_id for t in subtasks],
        )

        return subtasks

    def identify_capabilities(self, subtask: str) -> list[str]:
        """Map subtask description to required capabilities.

        Args:
            subtask: Description of the subtask

        Returns:
            List of capability names required
        """
        subtask_lower = subtask.lower()
        capabilities = set()

        for keyword, caps in CAPABILITY_KEYWORDS.items():
            if keyword in subtask_lower:
                capabilities.update(caps)

        # Default to general testing if no specific capability found
        if not capabilities:
            capabilities.add("general")

        return list(capabilities)

    def create_dependency_graph(self, subtasks: list[TaskDefinition]) -> nx.DiGraph:
        """Build dependency graph between subtasks.

        Args:
            subtasks: List of task definitions

        Returns:
            Directed graph of dependencies
        """
        graph = nx.DiGraph()

        for task in subtasks:
            graph.add_node(task.task_id, task=task)
            for dep in task.depends_on:
                graph.add_edge(dep, task.task_id)

        return graph

    def _build_decomposition_prompt(self, task_description: str) -> str:
        """Build the prompt for LLM decomposition."""
        return f"""You are a test automation expert. Decompose the following task into discrete, executable subtasks.

TASK: {task_description}

Return a JSON array of subtasks. Each subtask should have:
- id: A short, unique identifier (e.g., "navigate-to-login")
- description: Clear description of what to do
- capabilities: Array of required capabilities from this list:
  - code_analysis: Analyzing source code
  - ui_testing: Browser-based UI interactions
  - api_testing: HTTP API calls
  - db_testing: Database queries
  - vision: Visual/screenshot analysis
  - performance: Performance metrics
  - security: Security scanning
  - accessibility: Accessibility checking
  - reporting: Report generation
- inputs: Object describing required inputs
- outputs: Object describing expected outputs
- phase: One of [analysis, planning, setup, execution, verification, cleanup, reporting]

Guidelines:
1. Create {self.config.min_subtasks}-{self.config.max_subtasks} subtasks
2. Each subtask should be atomic and independently executable
3. Order subtasks logically
4. Include setup and cleanup tasks where appropriate
5. Ensure proper test isolation

Example response format:
```json
[
  {{
    "id": "navigate-to-page",
    "description": "Navigate to the registration page",
    "capabilities": ["ui_testing"],
    "inputs": {{"url": "string"}},
    "outputs": {{"page_loaded": "boolean"}},
    "phase": "setup"
  }},
  {{
    "id": "fill-form",
    "description": "Fill out the registration form with test data",
    "capabilities": ["ui_testing"],
    "inputs": {{"form_data": "object"}},
    "outputs": {{"form_filled": "boolean"}},
    "phase": "execution"
  }}
]
```

Respond with ONLY the JSON array, no explanation."""

    async def _call_llm(self, prompt: str) -> str:
        """Call Claude for task decomposition."""
        import anthropic

        client = anthropic.Anthropic(
            api_key=self.settings.anthropic_api_key.get_secret_value()
            if self.settings.anthropic_api_key
            else None
        )

        response = client.messages.create(
            model=self.config.model,
            max_tokens=4096,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def _parse_subtasks(
        self,
        json_response: str,
        original_task: str,
    ) -> list[TaskDefinition]:
        """Parse LLM response into TaskDefinition objects."""
        # Extract JSON from response (handle markdown code blocks)
        json_str = json_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]

        try:
            subtasks_data = json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            self.log.error("Failed to parse LLM response", error=str(e))
            return self._heuristic_decompose(original_task)

        subtasks = []
        for i, data in enumerate(subtasks_data):
            task = TaskDefinition(
                task_id=data.get("id", f"subtask-{i}"),
                description=data.get("description", f"Subtask {i}"),
                required_capabilities=data.get("capabilities", []),
                input_schema=data.get("inputs", {}),
                output_schema=data.get("outputs", {}),
                depends_on=[],  # Will be inferred later
                metadata={
                    "phase": data.get("phase", "execution"),
                    "original_task": original_task,
                },
            )
            subtasks.append(task)

        return subtasks

    def _heuristic_decompose(self, task_description: str) -> list[TaskDefinition]:
        """Fallback heuristic decomposition when LLM fails.

        Breaks task into standard phases: setup, execute, verify, cleanup.
        """
        subtasks = []

        # Setup phase
        subtasks.append(TaskDefinition(
            task_id="setup",
            description=f"Set up environment for: {task_description}",
            required_capabilities=self.identify_capabilities(task_description),
            input_schema={"config": "object"},
            output_schema={"ready": "boolean"},
            metadata={"phase": "setup"},
        ))

        # Main execution
        subtasks.append(TaskDefinition(
            task_id="execute",
            description=f"Execute: {task_description}",
            required_capabilities=self.identify_capabilities(task_description),
            input_schema={"inputs": "object"},
            output_schema={"result": "object"},
            depends_on=["setup"],
            metadata={"phase": "execution"},
        ))

        # Verification
        subtasks.append(TaskDefinition(
            task_id="verify",
            description=f"Verify results of: {task_description}",
            required_capabilities=self.identify_capabilities(task_description),
            input_schema={"result": "object"},
            output_schema={"verified": "boolean"},
            depends_on=["execute"],
            metadata={"phase": "verification"},
        ))

        # Cleanup
        subtasks.append(TaskDefinition(
            task_id="cleanup",
            description="Clean up test environment",
            required_capabilities=["general"],
            input_schema={},
            output_schema={"cleaned": "boolean"},
            depends_on=["verify"],
            metadata={"phase": "cleanup"},
        ))

        return subtasks

    def _infer_dependencies(
        self,
        subtasks: list[TaskDefinition],
    ) -> list[TaskDefinition]:
        """Infer dependencies based on task phases.

        Tasks in earlier phases should complete before later phases.
        """
        # Group tasks by phase
        phase_tasks: dict[str, list[TaskDefinition]] = {
            phase: [] for phase in TASK_PHASES
        }

        for task in subtasks:
            phase = task.metadata.get("phase", "execution")
            if phase in phase_tasks:
                phase_tasks[phase].append(task)
            else:
                phase_tasks["execution"].append(task)

        # Add dependencies based on phase ordering
        previous_phase_tasks: list[str] = []

        for phase in TASK_PHASES:
            tasks_in_phase = phase_tasks[phase]

            for task in tasks_in_phase:
                # Add dependencies from previous phase
                for prev_task_id in previous_phase_tasks:
                    if prev_task_id not in task.depends_on:
                        task.depends_on.append(prev_task_id)

            # Update previous phase tasks
            if tasks_in_phase:
                previous_phase_tasks = [t.task_id for t in tasks_in_phase]

        return subtasks

    def _infer_phase(self, description: str) -> str:
        """Infer the phase of a task from its description."""
        description_lower = description.lower()

        for phase, keywords in PHASE_KEYWORDS.items():
            if any(kw in description_lower for kw in keywords):
                return phase

        return "execution"  # Default phase


class SmartDecomposer(TaskDecomposer):
    """Enhanced decomposer with learning and caching.

    Extends TaskDecomposer with:
    - Caching of decomposition patterns
    - Learning from execution results
    - Domain-specific decomposition strategies
    """

    def __init__(
        self,
        config: DecompositionConfig | None = None,
        cache_enabled: bool = True,
    ):
        super().__init__(config)
        self.cache_enabled = cache_enabled
        self._pattern_cache: dict[str, list[TaskDefinition]] = {}

    async def decompose(self, task_description: str) -> list[TaskDefinition]:
        """Decompose with caching support."""
        # Check cache
        cache_key = self._compute_cache_key(task_description)
        if self.cache_enabled and cache_key in self._pattern_cache:
            self.log.debug("Using cached decomposition", cache_key=cache_key)
            return self._clone_tasks(self._pattern_cache[cache_key])

        # Decompose normally
        subtasks = await super().decompose(task_description)

        # Cache result
        if self.cache_enabled:
            self._pattern_cache[cache_key] = subtasks

        return subtasks

    def _compute_cache_key(self, task_description: str) -> str:
        """Compute a cache key from task description.

        Normalizes the description to improve cache hit rate.
        """
        # Simple normalization: lowercase and sort words
        words = sorted(task_description.lower().split())
        return " ".join(words[:10])  # Use first 10 words

    def _clone_tasks(self, tasks: list[TaskDefinition]) -> list[TaskDefinition]:
        """Create new task instances with fresh IDs."""
        cloned = []
        id_map: dict[str, str] = {}

        for task in tasks:
            new_id = f"{task.task_id}-{uuid.uuid4().hex[:6]}"
            id_map[task.task_id] = new_id

            cloned.append(TaskDefinition(
                task_id=new_id,
                description=task.description,
                required_capabilities=task.required_capabilities.copy(),
                input_schema=task.input_schema.copy(),
                output_schema=task.output_schema.copy(),
                priority=task.priority,
                timeout_seconds=task.timeout_seconds,
                retry_count=task.retry_count,
                depends_on=[],  # Will be updated below
                metadata=task.metadata.copy(),
            ))

        # Update dependencies with new IDs
        for i, original in enumerate(tasks):
            cloned[i].depends_on = [
                id_map.get(dep, dep) for dep in original.depends_on
            ]

        return cloned

    def learn_from_execution(
        self,
        task_description: str,
        subtasks: list[TaskDefinition],
        success: bool,
    ) -> None:
        """Learn from execution results to improve future decompositions.

        Args:
            task_description: Original task description
            subtasks: Subtasks that were executed
            success: Whether execution was successful
        """
        if success:
            # Good decomposition - cache it
            cache_key = self._compute_cache_key(task_description)
            self._pattern_cache[cache_key] = subtasks
            self.log.info(
                "Learned successful decomposition pattern",
                cache_key=cache_key,
                subtask_count=len(subtasks),
            )
        else:
            # Failed decomposition - remove from cache
            cache_key = self._compute_cache_key(task_description)
            if cache_key in self._pattern_cache:
                del self._pattern_cache[cache_key]
                self.log.info(
                    "Removed failed decomposition pattern from cache",
                    cache_key=cache_key,
                )
