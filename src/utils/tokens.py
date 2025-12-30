"""Token counting and cost tracking utilities.

Provides:
- Token estimation for text and images
- Cost tracking per model
- Budget enforcement
- Usage reporting
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..config import MODEL_PRICING, ModelName, SCREENSHOT_TOKENS


def estimate_tokens(text: str, model: str = "claude-sonnet-4-5") -> int:
    """Estimate token count for text.

    Uses a simple heuristic: ~4 characters per token for English.
    For more accurate counts, use tiktoken with cl100k_base encoding.

    Args:
        text: Text to estimate tokens for
        model: Model name (unused, for future model-specific estimation)

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Simple heuristic: ~4 chars per token for English
    # This is approximate; actual tokenization may differ
    return len(text) // 4 + 1


def estimate_image_tokens(
    width: int,
    height: int,
    detail: str = "auto",
) -> int:
    """Estimate token count for an image.

    Based on Claude's image token calculation.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        detail: Detail level (auto, low, high)

    Returns:
        Estimated token count
    """
    # Find closest resolution in our estimates
    for resolution, tokens in sorted(SCREENSHOT_TOKENS.items()):
        if width <= resolution[0] and height <= resolution[1]:
            return tokens

    # For larger images, estimate based on area ratio
    base_resolution = (1920, 1080)
    base_tokens = SCREENSHOT_TOKENS.get(base_resolution, 2500)

    area_ratio = (width * height) / (base_resolution[0] * base_resolution[1])
    return int(base_tokens * min(area_ratio, 4))  # Cap at 4x


def estimate_screenshot_tokens(
    screenshot: bytes,
    resolution: tuple[int, int] = (1920, 1080),
) -> int:
    """Estimate tokens for a screenshot.

    Args:
        screenshot: Screenshot bytes
        resolution: Screenshot resolution

    Returns:
        Estimated token count
    """
    return estimate_image_tokens(resolution[0], resolution[1])


@dataclass
class TokenUsage:
    """Record of token usage for a single API call."""

    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    operation: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class UsageSummary:
    """Summary of token usage."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    call_count: int = 0
    by_model: dict = field(default_factory=dict)
    by_operation: dict = field(default_factory=dict)


class TokenCounter:
    """Tracks token usage across API calls.

    Usage:
        counter = TokenCounter()
        counter.add_usage("claude-sonnet-4-5", input_tokens=1000, output_tokens=500)
        print(counter.summary())
    """

    def __init__(self):
        self._usage: list[TokenUsage] = []

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "",
        **metadata,
    ) -> float:
        """Record token usage.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation: Optional operation name
            **metadata: Additional metadata

        Returns:
            Cost of this usage
        """
        # Calculate cost
        try:
            model_enum = ModelName(model)
            pricing = MODEL_PRICING[model_enum]
        except (ValueError, KeyError):
            # Fallback to Sonnet pricing for unknown models
            pricing = MODEL_PRICING[ModelName.SONNET]

        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )

        usage = TokenUsage(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            operation=operation,
            metadata=metadata,
        )
        self._usage.append(usage)

        return cost

    def get_summary(self) -> UsageSummary:
        """Get usage summary.

        Returns:
            UsageSummary with aggregated statistics
        """
        summary = UsageSummary()

        for usage in self._usage:
            summary.total_input_tokens += usage.input_tokens
            summary.total_output_tokens += usage.output_tokens
            summary.total_cost += usage.cost
            summary.call_count += 1

            # By model
            if usage.model not in summary.by_model:
                summary.by_model[usage.model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                    "calls": 0,
                }
            summary.by_model[usage.model]["input_tokens"] += usage.input_tokens
            summary.by_model[usage.model]["output_tokens"] += usage.output_tokens
            summary.by_model[usage.model]["cost"] += usage.cost
            summary.by_model[usage.model]["calls"] += 1

            # By operation
            if usage.operation:
                if usage.operation not in summary.by_operation:
                    summary.by_operation[usage.operation] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost": 0.0,
                        "calls": 0,
                    }
                summary.by_operation[usage.operation]["input_tokens"] += usage.input_tokens
                summary.by_operation[usage.operation]["output_tokens"] += usage.output_tokens
                summary.by_operation[usage.operation]["cost"] += usage.cost
                summary.by_operation[usage.operation]["calls"] += 1

        return summary

    @property
    def total_cost(self) -> float:
        """Get total cost."""
        return sum(u.cost for u in self._usage)

    @property
    def total_tokens(self) -> int:
        """Get total tokens (input + output)."""
        return sum(u.input_tokens + u.output_tokens for u in self._usage)

    def reset(self) -> None:
        """Reset all usage tracking."""
        self._usage.clear()

    def get_usage_log(self) -> list[dict]:
        """Get full usage log.

        Returns:
            List of usage records as dicts
        """
        return [
            {
                "timestamp": u.timestamp.isoformat(),
                "model": u.model,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cost": u.cost,
                "operation": u.operation,
            }
            for u in self._usage
        ]


class CostTracker:
    """Tracks costs and enforces budget limits.

    Usage:
        tracker = CostTracker(budget_limit=10.0)

        if tracker.can_afford(estimated_cost=0.50):
            # Make API call
            tracker.record_cost(0.45)

        if tracker.is_budget_exceeded():
            raise BudgetExceeded()
    """

    def __init__(
        self,
        budget_limit: float = 10.0,
        warning_threshold: float = 0.8,
    ):
        """Initialize cost tracker.

        Args:
            budget_limit: Maximum budget in USD
            warning_threshold: Fraction of budget at which to warn
        """
        self.budget_limit = budget_limit
        self.warning_threshold = warning_threshold
        self._total_cost = 0.0
        self._cost_log: list[tuple[datetime, float, str]] = []

    def record_cost(self, cost: float, operation: str = "") -> None:
        """Record a cost.

        Args:
            cost: Cost in USD
            operation: Optional operation name
        """
        self._total_cost += cost
        self._cost_log.append((datetime.now(), cost, operation))

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if an operation can be afforded.

        Args:
            estimated_cost: Estimated cost of operation

        Returns:
            True if within budget
        """
        return (self._total_cost + estimated_cost) <= self.budget_limit

    def is_budget_exceeded(self) -> bool:
        """Check if budget is exceeded.

        Returns:
            True if over budget
        """
        return self._total_cost >= self.budget_limit

    def is_warning_threshold_reached(self) -> bool:
        """Check if warning threshold is reached.

        Returns:
            True if at or above warning threshold
        """
        return self._total_cost >= (self.budget_limit * self.warning_threshold)

    @property
    def total_cost(self) -> float:
        """Get total cost so far."""
        return self._total_cost

    @property
    def remaining_budget(self) -> float:
        """Get remaining budget."""
        return max(0, self.budget_limit - self._total_cost)

    @property
    def budget_usage_percent(self) -> float:
        """Get budget usage as percentage."""
        if self.budget_limit <= 0:
            return 100.0
        return (self._total_cost / self.budget_limit) * 100

    def reset(self) -> None:
        """Reset cost tracking."""
        self._total_cost = 0.0
        self._cost_log.clear()

    def get_cost_breakdown(self) -> dict:
        """Get cost breakdown by operation.

        Returns:
            Dict of operation to total cost
        """
        breakdown = {}
        for _, cost, operation in self._cost_log:
            op = operation or "unknown"
            breakdown[op] = breakdown.get(op, 0.0) + cost
        return breakdown


class BudgetExceeded(Exception):
    """Exception raised when budget is exceeded."""

    def __init__(self, current_cost: float, budget_limit: float):
        self.current_cost = current_cost
        self.budget_limit = budget_limit
        super().__init__(
            f"Budget exceeded: ${current_cost:.4f} spent of ${budget_limit:.2f} limit"
        )
