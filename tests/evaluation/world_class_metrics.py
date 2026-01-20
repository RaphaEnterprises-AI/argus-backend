"""
World-Class Agent Evaluation Metrics.

Implements industry-standard evaluation metrics used by:
- Anthropic (Bloom framework, MCP evaluations)
- OpenAI (SWE-bench Verified, WebArena)
- Berkeley (BFCL function calling leaderboard)
- Sierra Research (TAU-bench)

Key metrics:
- Pass@k: Probability of getting at least one correct answer in k attempts
- Task Success Rate: End-to-end task completion (WebArena style)
- Elicitation Rate: Behavior frequency measurement (Bloom style)
- Cost Efficiency: Cost per successful task
- Multi-turn Accuracy: Stateful conversation handling
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskDifficulty(str, Enum):
    """Difficulty levels aligned with SWE-bench stratification."""
    EASY = "easy"           # Simple, single-step tasks
    MEDIUM = "medium"       # Multi-step, requires planning
    HARD = "hard"           # Complex, requires reasoning + recovery
    EXPERT = "expert"       # Real-world complexity, edge cases


class EvalDomain(str, Enum):
    """Evaluation domains aligned with industry benchmarks."""
    CODE_UNDERSTANDING = "code_understanding"      # SWE-bench style
    WEB_NAVIGATION = "web_navigation"              # WebArena style
    FUNCTION_CALLING = "function_calling"          # BFCL style
    TOOL_USE = "tool_use"                          # MCP/TAU-bench style
    MULTI_TURN_REASONING = "multi_turn_reasoning"  # TAU-bench style
    SELF_HEALING = "self_healing"                  # Unique to E2E testing
    VISUAL_UNDERSTANDING = "visual_understanding"  # Computer use


@dataclass
class PassAtKResult:
    """
    Pass@k metric - probability of success in k attempts.

    Standard metric used by SWE-bench, HumanEval, and most code benchmarks.

    Formula: pass@k = 1 - C(n-c, k) / C(n, k)
    where n = total attempts, c = correct attempts, k = number to sample
    """
    task_id: str
    total_attempts: int
    successful_attempts: int

    @property
    def pass_at_1(self) -> float:
        """Probability of success in 1 attempt."""
        return self._calculate_pass_at_k(1)

    @property
    def pass_at_3(self) -> float:
        """Probability of success in 3 attempts."""
        return self._calculate_pass_at_k(3)

    @property
    def pass_at_5(self) -> float:
        """Probability of success in 5 attempts."""
        return self._calculate_pass_at_k(5)

    def _calculate_pass_at_k(self, k: int) -> float:
        """Calculate pass@k using unbiased estimator."""
        n = self.total_attempts
        c = self.successful_attempts

        if n < k:
            # Not enough samples, return simple success rate
            return c / max(n, 1)

        if c >= n:
            return 1.0

        # Unbiased estimator: 1 - C(n-c, k) / C(n, k)
        # Using log to avoid overflow for large numbers
        try:
            numerator = math.comb(n - c, k)
            denominator = math.comb(n, k)
            return 1.0 - (numerator / denominator)
        except (ValueError, ZeroDivisionError):
            return c / max(n, 1)


@dataclass
class TaskSuccessMetrics:
    """
    WebArena-style task success metrics.

    Measures end-to-end task completion with:
    - Binary success/failure
    - Partial completion tracking
    - Step-level analysis
    """
    task_id: str
    task_description: str
    difficulty: TaskDifficulty
    domain: EvalDomain

    # Execution results
    completed: bool = False
    partial_completion_pct: float = 0.0
    steps_attempted: int = 0
    steps_succeeded: int = 0

    # Timing
    latency_ms: float = 0.0
    timeout_occurred: bool = False

    # Cost tracking
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    # Error analysis
    error_type: str | None = None
    error_message: str | None = None
    recovery_attempted: bool = False
    recovery_succeeded: bool = False

    @property
    def step_success_rate(self) -> float:
        """Percentage of steps completed successfully."""
        if self.steps_attempted == 0:
            return 0.0
        return self.steps_succeeded / self.steps_attempted

    @property
    def cost_per_step(self) -> float:
        """Cost per attempted step."""
        if self.steps_attempted == 0:
            return 0.0
        return self.cost_usd / self.steps_attempted


@dataclass
class ElicitationMetrics:
    """
    Anthropic Bloom-style elicitation metrics.

    Measures behavior frequency and severity across scenarios.
    Used for alignment evaluations and capability testing.
    """
    behavior_name: str
    total_scenarios: int = 0
    scenarios_with_behavior: int = 0
    severity_scores: list[float] = field(default_factory=list)  # 1-10 scale

    @property
    def elicitation_rate(self) -> float:
        """Primary metric: % of scenarios with behavior score >= 7."""
        if not self.severity_scores:
            return 0.0
        high_severity = sum(1 for s in self.severity_scores if s >= 7)
        return high_severity / len(self.severity_scores)

    @property
    def average_severity(self) -> float:
        """Average severity score across all scenarios."""
        if not self.severity_scores:
            return 0.0
        return sum(self.severity_scores) / len(self.severity_scores)

    @property
    def frequency_rate(self) -> float:
        """How often the behavior appears at all."""
        if self.total_scenarios == 0:
            return 0.0
        return self.scenarios_with_behavior / self.total_scenarios


@dataclass
class MultiTurnMetrics:
    """
    TAU-bench style multi-turn conversation metrics.

    Evaluates agent's ability to:
    - Maintain context across turns
    - Handle state transitions
    - Recover from errors mid-conversation
    """
    conversation_id: str
    total_turns: int = 0
    successful_turns: int = 0
    context_maintained: bool = True
    state_transitions_correct: int = 0
    state_transitions_total: int = 0

    # Long-context specific
    context_length_tokens: int = 0
    context_utilization_pct: float = 0.0  # How much context was actually used

    @property
    def turn_accuracy(self) -> float:
        """Accuracy across conversation turns."""
        if self.total_turns == 0:
            return 0.0
        return self.successful_turns / self.total_turns

    @property
    def state_transition_accuracy(self) -> float:
        """Accuracy of state transitions."""
        if self.state_transitions_total == 0:
            return 1.0  # No transitions needed
        return self.state_transitions_correct / self.state_transitions_total


@dataclass
class CostEfficiencyMetrics:
    """
    Cost efficiency metrics for production deployment evaluation.

    Critical for comparing agents at scale.
    """
    total_tasks: int = 0
    successful_tasks: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_latency_ms: float = 0.0

    @property
    def cost_per_success(self) -> float:
        """Cost per successful task completion."""
        if self.successful_tasks == 0:
            return float('inf')
        return self.total_cost_usd / self.successful_tasks

    @property
    def cost_per_attempt(self) -> float:
        """Cost per task attempt."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_cost_usd / self.total_tasks

    @property
    def tokens_per_success(self) -> float:
        """Token efficiency per success."""
        if self.successful_tasks == 0:
            return float('inf')
        return self.total_tokens / self.successful_tasks

    @property
    def latency_per_task_ms(self) -> float:
        """Average latency per task."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_latency_ms / self.total_tasks


@dataclass
class BenchmarkComparison:
    """
    Compare agent performance against industry benchmarks.
    """
    benchmark_name: str
    agent_score: float
    human_baseline: float | None = None
    sota_score: float | None = None  # State of the art
    percentile_rank: float | None = None

    @property
    def human_parity_pct(self) -> float | None:
        """Percentage of human performance achieved."""
        if self.human_baseline is None or self.human_baseline == 0:
            return None
        return (self.agent_score / self.human_baseline) * 100

    @property
    def sota_gap_pct(self) -> float | None:
        """Gap to state-of-the-art performance."""
        if self.sota_score is None:
            return None
        return self.sota_score - self.agent_score


# Human baselines from published benchmarks (2025 data)
HUMAN_BASELINES = {
    "webarena": 78.24,          # WebArena human performance
    "swe_bench_verified": 97.0,  # Estimated human developer performance
    "bfcl": 95.0,               # Human function calling accuracy
    "tau_bench": 92.0,          # Human multi-turn performance
}

# State-of-the-art scores (2025 data)
SOTA_SCORES = {
    "webarena": 61.7,           # IBM CUGA (Feb 2025)
    "swe_bench_verified": 75.2, # Bytedance (2025)
    "bfcl": 70.36,              # Claude Opus 4.1
    "tau_bench": 65.0,          # Estimated
    "mcp_mark": 52.6,           # GPT-5 pass@1
}


@dataclass
class WorldClassEvalReport:
    """
    Comprehensive evaluation report aligned with industry standards.
    """
    # Metadata
    agent_name: str
    model_version: str
    evaluation_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Pass@k metrics (SWE-bench style)
    pass_at_k_results: list[PassAtKResult] = field(default_factory=list)

    # Task success metrics (WebArena style)
    task_metrics: list[TaskSuccessMetrics] = field(default_factory=list)

    # Elicitation metrics (Bloom style)
    elicitation_metrics: list[ElicitationMetrics] = field(default_factory=list)

    # Multi-turn metrics (TAU-bench style)
    multi_turn_metrics: list[MultiTurnMetrics] = field(default_factory=list)

    # Cost efficiency
    cost_metrics: CostEfficiencyMetrics = field(default_factory=CostEfficiencyMetrics)

    # Benchmark comparisons
    comparisons: list[BenchmarkComparison] = field(default_factory=list)

    def aggregate_pass_at_k(self) -> dict[str, float]:
        """Aggregate pass@k across all tasks."""
        if not self.pass_at_k_results:
            return {"pass@1": 0.0, "pass@3": 0.0, "pass@5": 0.0}

        return {
            "pass@1": sum(r.pass_at_1 for r in self.pass_at_k_results) / len(self.pass_at_k_results),
            "pass@3": sum(r.pass_at_3 for r in self.pass_at_k_results) / len(self.pass_at_k_results),
            "pass@5": sum(r.pass_at_5 for r in self.pass_at_k_results) / len(self.pass_at_k_results),
        }

    def task_success_by_difficulty(self) -> dict[str, dict[str, float]]:
        """Break down success rate by difficulty level."""
        by_difficulty: dict[str, list[TaskSuccessMetrics]] = {}

        for metric in self.task_metrics:
            diff = metric.difficulty.value
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(metric)

        result = {}
        for diff, metrics in by_difficulty.items():
            completed = sum(1 for m in metrics if m.completed)
            result[diff] = {
                "total": len(metrics),
                "completed": completed,
                "success_rate": completed / len(metrics) if metrics else 0.0,
                "avg_latency_ms": sum(m.latency_ms for m in metrics) / len(metrics) if metrics else 0.0,
            }

        return result

    def task_success_by_domain(self) -> dict[str, dict[str, float]]:
        """Break down success rate by evaluation domain."""
        by_domain: dict[str, list[TaskSuccessMetrics]] = {}

        for metric in self.task_metrics:
            domain = metric.domain.value
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(metric)

        result = {}
        for domain, metrics in by_domain.items():
            completed = sum(1 for m in metrics if m.completed)
            result[domain] = {
                "total": len(metrics),
                "completed": completed,
                "success_rate": completed / len(metrics) if metrics else 0.0,
                "avg_cost_usd": sum(m.cost_usd for m in metrics) / len(metrics) if metrics else 0.0,
            }

        return result

    def overall_grade(self) -> str:
        """
        Calculate overall grade based on multiple dimensions.

        Weighting:
        - Pass@1: 30% (primary success metric)
        - Task completion: 25%
        - Cost efficiency: 20%
        - Multi-turn accuracy: 15%
        - SOTA comparison: 10%
        """
        scores = []

        # Pass@k component
        pass_at_k = self.aggregate_pass_at_k()
        if pass_at_k["pass@1"] > 0:
            scores.append(("pass@1", pass_at_k["pass@1"], 0.30))

        # Task completion component
        if self.task_metrics:
            task_success = sum(1 for t in self.task_metrics if t.completed) / len(self.task_metrics)
            scores.append(("task_success", task_success, 0.25))

        # Cost efficiency (normalized, lower is better - invert)
        if self.cost_metrics.cost_per_success < float('inf'):
            # Assume $0.10 per task is excellent, $1.00 is poor
            cost_score = max(0, 1 - (self.cost_metrics.cost_per_success / 1.0))
            scores.append(("cost_efficiency", cost_score, 0.20))

        # Multi-turn accuracy
        if self.multi_turn_metrics:
            mt_accuracy = sum(m.turn_accuracy for m in self.multi_turn_metrics) / len(self.multi_turn_metrics)
            scores.append(("multi_turn", mt_accuracy, 0.15))

        # SOTA comparison (average human parity across benchmarks)
        if self.comparisons:
            parities = [c.human_parity_pct for c in self.comparisons if c.human_parity_pct is not None]
            if parities:
                avg_parity = sum(parities) / len(parities) / 100  # Convert to 0-1
                scores.append(("sota_comparison", min(avg_parity, 1.0), 0.10))

        if not scores:
            return "N/A"

        # Calculate weighted average
        total_weight = sum(w for _, _, w in scores)
        weighted_score = sum(s * w for _, s, w in scores) / total_weight

        # Grade mapping
        if weighted_score >= 0.90:
            return "A+"
        elif weighted_score >= 0.85:
            return "A"
        elif weighted_score >= 0.80:
            return "A-"
        elif weighted_score >= 0.75:
            return "B+"
        elif weighted_score >= 0.70:
            return "B"
        elif weighted_score >= 0.65:
            return "B-"
        elif weighted_score >= 0.60:
            return "C+"
        elif weighted_score >= 0.55:
            return "C"
        elif weighted_score >= 0.50:
            return "C-"
        elif weighted_score >= 0.45:
            return "D"
        else:
            return "F"

    def to_dict(self) -> dict[str, Any]:
        """Export report as dictionary for JSON serialization."""
        return {
            "metadata": {
                "agent_name": self.agent_name,
                "model_version": self.model_version,
                "timestamp": self.evaluation_timestamp.isoformat(),
                "overall_grade": self.overall_grade(),
            },
            "pass_at_k": self.aggregate_pass_at_k(),
            "task_success": {
                "by_difficulty": self.task_success_by_difficulty(),
                "by_domain": self.task_success_by_domain(),
                "total_tasks": len(self.task_metrics),
                "completed": sum(1 for t in self.task_metrics if t.completed),
            },
            "cost_efficiency": {
                "cost_per_success": self.cost_metrics.cost_per_success,
                "cost_per_attempt": self.cost_metrics.cost_per_attempt,
                "tokens_per_success": self.cost_metrics.tokens_per_success,
                "total_cost_usd": self.cost_metrics.total_cost_usd,
            },
            "multi_turn": {
                "avg_turn_accuracy": (
                    sum(m.turn_accuracy for m in self.multi_turn_metrics) / len(self.multi_turn_metrics)
                    if self.multi_turn_metrics else 0.0
                ),
                "context_maintained_pct": (
                    sum(1 for m in self.multi_turn_metrics if m.context_maintained) / len(self.multi_turn_metrics) * 100
                    if self.multi_turn_metrics else 0.0
                ),
            },
            "benchmark_comparisons": [
                {
                    "benchmark": c.benchmark_name,
                    "agent_score": c.agent_score,
                    "human_baseline": c.human_baseline,
                    "sota_score": c.sota_score,
                    "human_parity_pct": c.human_parity_pct,
                }
                for c in self.comparisons
            ],
        }
