"""
Evaluation Metrics for Agent Quality Assessment.

Implements multi-dimensional scoring based on:
- Task completion accuracy
- Reasoning quality
- Decision correctness
- Cost efficiency
- Latency performance
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EvalCategory(str, Enum):
    """Evaluation categories aligned with agent capabilities."""
    CODE_ANALYSIS = "code_analysis"
    TEST_PLANNING = "test_planning"
    UI_EXECUTION = "ui_execution"
    API_TESTING = "api_testing"
    SELF_HEALING = "self_healing"
    VISUAL_AI = "visual_ai"
    NLP_UNDERSTANDING = "nlp_understanding"
    ORCHESTRATION = "orchestration"


@dataclass
class AgentScore:
    """Score for a single agent evaluation."""
    agent_name: str
    category: EvalCategory
    task_id: str

    # Core metrics (0.0 - 1.0)
    accuracy: float = 0.0  # Did it complete the task correctly?
    reasoning_quality: float = 0.0  # Quality of decision-making process
    plan_correctness: float = 0.0  # Were intermediate steps correct?

    # Performance metrics
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0

    # Metadata
    passed: bool = False
    error: str | None = None
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def overall_score(self) -> float:
        """Weighted overall score (0.0 - 1.0)."""
        return (
            self.accuracy * 0.5 +
            self.reasoning_quality * 0.3 +
            self.plan_correctness * 0.2
        )

    @property
    def grade(self) -> str:
        """Letter grade based on overall score."""
        score = self.overall_score
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        return "F"


@dataclass
class EvaluationMetrics:
    """Aggregated metrics across all evaluations."""

    scores: list[AgentScore] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    def add_score(self, score: AgentScore) -> None:
        """Add a score to the evaluation."""
        self.scores.append(score)

    @property
    def total_tests(self) -> int:
        return len(self.scores)

    @property
    def passed_tests(self) -> int:
        return sum(1 for s in self.scores if s.passed)

    @property
    def pass_rate(self) -> float:
        if not self.scores:
            return 0.0
        return self.passed_tests / self.total_tests

    @property
    def avg_accuracy(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.accuracy for s in self.scores) / len(self.scores)

    @property
    def avg_reasoning_quality(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.reasoning_quality for s in self.scores) / len(self.scores)

    @property
    def avg_overall_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.overall_score for s in self.scores) / len(self.scores)

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.scores)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.scores)

    @property
    def avg_latency_ms(self) -> float:
        if not self.scores:
            return 0.0
        return self.total_latency_ms / len(self.scores)

    def scores_by_category(self, category: EvalCategory) -> list[AgentScore]:
        """Get scores filtered by category."""
        return [s for s in self.scores if s.category == category]

    def scores_by_agent(self, agent_name: str) -> list[AgentScore]:
        """Get scores filtered by agent."""
        return [s for s in self.scores if s.agent_name == agent_name]

    def category_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary metrics per category."""
        summary = {}
        for category in EvalCategory:
            cat_scores = self.scores_by_category(category)
            if cat_scores:
                summary[category.value] = {
                    "count": len(cat_scores),
                    "passed": sum(1 for s in cat_scores if s.passed),
                    "avg_accuracy": sum(s.accuracy for s in cat_scores) / len(cat_scores),
                    "avg_score": sum(s.overall_score for s in cat_scores) / len(cat_scores),
                    "avg_latency_ms": sum(s.latency_ms for s in cat_scores) / len(cat_scores),
                    "total_cost": sum(s.cost_usd for s in cat_scores),
                }
        return summary

    def to_report(self) -> dict:
        """Generate a comprehensive evaluation report."""
        self.completed_at = datetime.utcnow()

        return {
            "summary": {
                "total_tests": self.total_tests,
                "passed": self.passed_tests,
                "failed": self.total_tests - self.passed_tests,
                "pass_rate": f"{self.pass_rate:.1%}",
                "overall_score": f"{self.avg_overall_score:.2f}",
                "grade": self._overall_grade(),
            },
            "quality_metrics": {
                "accuracy": f"{self.avg_accuracy:.2f}",
                "reasoning_quality": f"{self.avg_reasoning_quality:.2f}",
            },
            "performance": {
                "total_latency_ms": f"{self.total_latency_ms:.0f}",
                "avg_latency_ms": f"{self.avg_latency_ms:.0f}",
                "total_cost_usd": f"${self.total_cost:.4f}",
            },
            "by_category": self.category_summary(),
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at else 0
            ),
        }

    def _overall_grade(self) -> str:
        score = self.avg_overall_score
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        return "F"
