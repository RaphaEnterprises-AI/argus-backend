"""
Task Complexity Router for Enterprise Multi-Agent Architecture.

Implements Anthropic's scaling rules to prevent over-investment in simple problems:
- Simple fact check: 1 agent, 3-10 tool calls
- Direct comparison: 2-4 agents, 10-15 calls
- Complex research: 10+ agents, 25-50 calls

This ensures optimal resource allocation and cost efficiency.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class TaskComplexity(Enum):
    """Task complexity tiers based on Anthropic's production scaling rules."""

    SIMPLE = "simple"  # Single agent, quick resolution
    MODERATE = "moderate"  # Few agents, moderate coordination
    COMPLEX = "complex"  # Multi-agent, significant coordination
    RESEARCH = "research"  # Many agents, extensive exploration


@dataclass
class TaskConfig:
    """Configuration for a task complexity tier."""

    complexity: TaskComplexity
    max_agents: int
    max_tool_calls: int
    timeout_seconds: int
    model: str
    max_iterations: int
    parallel_execution: bool = False
    require_human_approval: bool = False

    # Cost controls
    max_input_tokens: int = 50000
    max_output_tokens: int = 10000
    budget_usd: float = 1.0


# Anthropic-style scaling rules
COMPLEXITY_CONFIGS: dict[TaskComplexity, TaskConfig] = {
    TaskComplexity.SIMPLE: TaskConfig(
        complexity=TaskComplexity.SIMPLE,
        max_agents=1,
        max_tool_calls=10,
        timeout_seconds=30,
        model="claude-haiku-4-5",
        max_iterations=5,
        parallel_execution=False,
        require_human_approval=False,
        max_input_tokens=20000,
        max_output_tokens=4000,
        budget_usd=0.10,
    ),
    TaskComplexity.MODERATE: TaskConfig(
        complexity=TaskComplexity.MODERATE,
        max_agents=4,
        max_tool_calls=20,
        timeout_seconds=120,
        model="claude-sonnet-4-5",
        max_iterations=15,
        parallel_execution=True,
        require_human_approval=False,
        max_input_tokens=50000,
        max_output_tokens=8000,
        budget_usd=0.50,
    ),
    TaskComplexity.COMPLEX: TaskConfig(
        complexity=TaskComplexity.COMPLEX,
        max_agents=10,
        max_tool_calls=30,
        timeout_seconds=300,
        model="claude-sonnet-4-5",
        max_iterations=25,
        parallel_execution=True,
        require_human_approval=False,
        max_input_tokens=100000,
        max_output_tokens=15000,
        budget_usd=2.00,
    ),
    TaskComplexity.RESEARCH: TaskConfig(
        complexity=TaskComplexity.RESEARCH,
        max_agents=15,
        max_tool_calls=50,
        timeout_seconds=600,
        model="claude-opus-4-5",
        max_iterations=50,
        parallel_execution=True,
        require_human_approval=True,  # High-stakes requires approval
        max_input_tokens=200000,
        max_output_tokens=30000,
        budget_usd=10.00,
    ),
}


@dataclass
class TaskAnalysis:
    """Analysis result for task complexity classification."""

    complexity: TaskComplexity
    config: TaskConfig
    confidence: float
    reasoning: str
    estimated_agents: int
    estimated_tool_calls: int
    risk_factors: list[str] = field(default_factory=list)


class TaskComplexityRouter:
    """
    Routes tasks to appropriate resources based on complexity analysis.

    Implements Anthropic's scaling rules:
    - Analyzes task scope, dependencies, and codebase size
    - Assigns appropriate agent count and resource limits
    - Prevents over-investment in simple problems
    - Ensures complex tasks get adequate resources
    """

    def __init__(self):
        self.log = logger.bind(component="complexity_router")

        # Keywords that indicate higher complexity
        self.complexity_indicators = {
            "simple": [
                "check",
                "verify",
                "validate",
                "single",
                "quick",
                "simple",
                "one file",
                "specific",
            ],
            "moderate": [
                "compare",
                "analyze",
                "multiple",
                "several",
                "few",
                "test suite",
                "module",
            ],
            "complex": [
                "refactor",
                "migrate",
                "comprehensive",
                "full",
                "entire",
                "all",
                "system-wide",
                "integration",
            ],
            "research": [
                "investigate",
                "research",
                "explore",
                "unknown",
                "debug complex",
                "architecture",
                "redesign",
            ],
        }

        # Risk factors that may increase complexity
        self.risk_indicators = [
            "production",
            "critical",
            "security",
            "authentication",
            "payment",
            "sensitive",
            "database migration",
            "breaking change",
        ]

    def classify_task(
        self,
        task_description: str,
        codebase_size: int | None = None,
        file_count: int | None = None,
        test_count: int | None = None,
        has_dependencies: bool = False,
        is_production: bool = False,
    ) -> TaskAnalysis:
        """
        Classify a task's complexity based on multiple factors.

        Args:
            task_description: Natural language description of the task
            codebase_size: Approximate lines of code (optional)
            file_count: Number of files in scope (optional)
            test_count: Number of tests to run/analyze (optional)
            has_dependencies: Whether task has external dependencies
            is_production: Whether this affects production systems

        Returns:
            TaskAnalysis with complexity classification and config
        """
        task_lower = task_description.lower()

        # Score each complexity level
        scores = {
            TaskComplexity.SIMPLE: 0.0,
            TaskComplexity.MODERATE: 0.0,
            TaskComplexity.COMPLEX: 0.0,
            TaskComplexity.RESEARCH: 0.0,
        }

        # Keyword-based scoring
        for indicator in self.complexity_indicators["simple"]:
            if indicator in task_lower:
                scores[TaskComplexity.SIMPLE] += 1.0

        for indicator in self.complexity_indicators["moderate"]:
            if indicator in task_lower:
                scores[TaskComplexity.MODERATE] += 1.0

        for indicator in self.complexity_indicators["complex"]:
            if indicator in task_lower:
                scores[TaskComplexity.COMPLEX] += 1.5

        for indicator in self.complexity_indicators["research"]:
            if indicator in task_lower:
                scores[TaskComplexity.RESEARCH] += 2.0

        # Codebase size factor
        if codebase_size:
            if codebase_size < 1000:
                scores[TaskComplexity.SIMPLE] += 1.0
            elif codebase_size < 10000:
                scores[TaskComplexity.MODERATE] += 1.0
            elif codebase_size < 100000:
                scores[TaskComplexity.COMPLEX] += 1.0
            else:
                scores[TaskComplexity.RESEARCH] += 1.0

        # File count factor
        if file_count:
            if file_count <= 3:
                scores[TaskComplexity.SIMPLE] += 1.0
            elif file_count <= 10:
                scores[TaskComplexity.MODERATE] += 1.0
            elif file_count <= 50:
                scores[TaskComplexity.COMPLEX] += 1.0
            else:
                scores[TaskComplexity.RESEARCH] += 1.0

        # Test count factor
        if test_count:
            if test_count <= 5:
                scores[TaskComplexity.SIMPLE] += 0.5
            elif test_count <= 20:
                scores[TaskComplexity.MODERATE] += 0.5
            elif test_count <= 100:
                scores[TaskComplexity.COMPLEX] += 0.5
            else:
                scores[TaskComplexity.RESEARCH] += 0.5

        # Dependency and production factors
        if has_dependencies:
            scores[TaskComplexity.MODERATE] += 0.5
            scores[TaskComplexity.COMPLEX] += 0.5

        if is_production:
            scores[TaskComplexity.COMPLEX] += 1.0
            scores[TaskComplexity.RESEARCH] += 0.5

        # Identify risk factors
        risk_factors = []
        for risk in self.risk_indicators:
            if risk in task_lower:
                risk_factors.append(risk)
                # Risk factors bump up complexity
                scores[TaskComplexity.COMPLEX] += 0.5
                scores[TaskComplexity.RESEARCH] += 0.5

        # Determine winning complexity
        max_score = max(scores.values())
        if max_score == 0:
            # Default to moderate if no clear indicators
            complexity = TaskComplexity.MODERATE
            confidence = 0.5
        else:
            complexity = max(scores, key=scores.get)
            # Confidence based on score margin
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[0] > 0:
                margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
                confidence = min(0.95, 0.6 + margin * 0.35)
            else:
                confidence = 0.7

        config = COMPLEXITY_CONFIGS[complexity]

        # Estimate agents and tool calls based on factors
        base_agents = config.max_agents // 2
        base_calls = config.max_tool_calls // 2

        if file_count:
            estimated_agents = min(config.max_agents, base_agents + file_count // 5)
            estimated_calls = min(config.max_tool_calls, base_calls + file_count * 2)
        else:
            estimated_agents = base_agents
            estimated_calls = base_calls

        # Generate reasoning
        reasoning_parts = [f"Task classified as {complexity.value}"]
        if scores[complexity] > 0:
            top_indicators = [
                ind
                for ind in self.complexity_indicators.get(complexity.value, [])
                if ind in task_lower
            ]
            if top_indicators:
                reasoning_parts.append(f"Keywords found: {', '.join(top_indicators[:3])}")
        if codebase_size:
            reasoning_parts.append(f"Codebase size: {codebase_size:,} LOC")
        if risk_factors:
            reasoning_parts.append(f"Risk factors: {', '.join(risk_factors)}")

        self.log.info(
            "Task classified",
            complexity=complexity.value,
            confidence=confidence,
            estimated_agents=estimated_agents,
            estimated_calls=estimated_calls,
            risk_factors=risk_factors,
        )

        return TaskAnalysis(
            complexity=complexity,
            config=config,
            confidence=confidence,
            reasoning=". ".join(reasoning_parts),
            estimated_agents=estimated_agents,
            estimated_tool_calls=estimated_calls,
            risk_factors=risk_factors,
        )

    def get_config_for_complexity(self, complexity: TaskComplexity) -> TaskConfig:
        """Get the configuration for a specific complexity level."""
        return COMPLEXITY_CONFIGS[complexity]

    def should_use_parallel_execution(self, analysis: TaskAnalysis) -> bool:
        """Determine if parallel agent execution should be used."""
        return (
            analysis.config.parallel_execution and analysis.estimated_agents > 1
        )

    def should_require_approval(self, analysis: TaskAnalysis) -> bool:
        """Determine if human approval is required."""
        return (
            analysis.config.require_human_approval
            or len(analysis.risk_factors) >= 2
            or "production" in analysis.risk_factors
            or "security" in analysis.risk_factors
        )

    def get_model_for_task(self, analysis: TaskAnalysis) -> str:
        """Get the appropriate model for the task."""
        # Override to Opus for security-critical tasks
        if "security" in analysis.risk_factors:
            return "claude-opus-4-5"
        return analysis.config.model


class AdaptiveResourceManager:
    """
    Manages resource allocation adaptively during task execution.

    Features:
    - Dynamic scaling based on progress
    - Early termination on success
    - Budget enforcement
    - Escalation to higher complexity if needed
    """

    def __init__(self, initial_analysis: TaskAnalysis):
        self.analysis = initial_analysis
        self.config = initial_analysis.config
        self.log = logger.bind(component="resource_manager")

        # Tracking
        self.agents_used = 0
        self.tool_calls_made = 0
        self.tokens_consumed = 0
        self.cost_incurred = 0.0
        self.iterations = 0

    def can_continue(self) -> tuple[bool, str | None]:
        """Check if execution can continue within limits."""
        if self.tool_calls_made >= self.config.max_tool_calls:
            return False, f"Tool call limit reached ({self.config.max_tool_calls})"

        if self.iterations >= self.config.max_iterations:
            return False, f"Iteration limit reached ({self.config.max_iterations})"

        if self.cost_incurred >= self.config.budget_usd:
            return False, f"Budget exhausted (${self.config.budget_usd:.2f})"

        if self.tokens_consumed >= self.config.max_input_tokens:
            return False, f"Token limit reached ({self.config.max_input_tokens:,})"

        return True, None

    def record_usage(
        self,
        tool_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ):
        """Record resource usage."""
        self.tool_calls_made += tool_calls
        self.tokens_consumed += input_tokens
        self.cost_incurred += cost_usd
        self.iterations += 1

        self.log.debug(
            "Resource usage recorded",
            tool_calls=self.tool_calls_made,
            tokens=self.tokens_consumed,
            cost=self.cost_incurred,
            iteration=self.iterations,
        )

    def should_escalate(self) -> bool:
        """Determine if task should be escalated to higher complexity."""
        # Escalate if we're hitting limits but not making progress
        usage_ratio = self.tool_calls_made / self.config.max_tool_calls
        return usage_ratio > 0.8 and self.analysis.complexity != TaskComplexity.RESEARCH

    def escalate(self) -> TaskAnalysis:
        """Escalate to the next complexity tier."""
        complexity_order = [
            TaskComplexity.SIMPLE,
            TaskComplexity.MODERATE,
            TaskComplexity.COMPLEX,
            TaskComplexity.RESEARCH,
        ]

        current_idx = complexity_order.index(self.analysis.complexity)
        if current_idx < len(complexity_order) - 1:
            new_complexity = complexity_order[current_idx + 1]
            new_config = COMPLEXITY_CONFIGS[new_complexity]

            self.log.warning(
                "Escalating task complexity",
                from_complexity=self.analysis.complexity.value,
                to_complexity=new_complexity.value,
            )

            self.analysis = TaskAnalysis(
                complexity=new_complexity,
                config=new_config,
                confidence=self.analysis.confidence,
                reasoning=f"Escalated from {self.analysis.complexity.value} due to resource limits",
                estimated_agents=new_config.max_agents,
                estimated_tool_calls=new_config.max_tool_calls,
                risk_factors=self.analysis.risk_factors,
            )
            self.config = new_config

        return self.analysis

    def get_usage_summary(self) -> dict[str, Any]:
        """Get a summary of resource usage."""
        return {
            "complexity": self.analysis.complexity.value,
            "agents_used": self.agents_used,
            "tool_calls": self.tool_calls_made,
            "tool_calls_limit": self.config.max_tool_calls,
            "tokens_consumed": self.tokens_consumed,
            "tokens_limit": self.config.max_input_tokens,
            "cost_usd": round(self.cost_incurred, 4),
            "budget_usd": self.config.budget_usd,
            "iterations": self.iterations,
            "iterations_limit": self.config.max_iterations,
        }
