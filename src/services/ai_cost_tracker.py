"""AI Cost Tracker Service for tracking token usage and costs.

This service tracks AI usage across the platform, including:
- Token counts per request
- Cost calculations based on model pricing
- Budget enforcement per organization
- Usage analytics and reporting
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional
import uuid
import structlog

from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()


class TaskType(str, Enum):
    """Types of AI tasks for usage tracking."""

    ERROR_ANALYSIS = "error_analysis"
    TEST_GENERATION = "test_generation"
    CODE_REVIEW = "code_review"
    SELF_HEALING = "self_healing"
    CORRELATION = "correlation"
    RISK_ASSESSMENT = "risk_assessment"
    PATTERN_MATCHING = "pattern_matching"
    OTHER = "other"


@dataclass
class ModelPricing:
    """Pricing information for an AI model (per 1M tokens)."""

    input_price: Decimal  # USD per 1M input tokens
    output_price: Decimal  # USD per 1M output tokens
    provider: str = "anthropic"


# Model pricing as of January 2026 (per 1M tokens)
MODEL_PRICING: dict[str, ModelPricing] = {
    # ===========================================
    # ANTHROPIC CLAUDE MODELS
    # ===========================================
    # Correct model IDs
    "claude-opus-4-5": ModelPricing(
        input_price=Decimal("15.00"),
        output_price=Decimal("75.00"),
        provider="anthropic",
    ),
    "claude-sonnet-4-5": ModelPricing(
        input_price=Decimal("3.00"),
        output_price=Decimal("15.00"),
        provider="anthropic",
    ),
    "claude-haiku-4-5": ModelPricing(
        input_price=Decimal("0.80"),
        output_price=Decimal("4.00"),
        provider="anthropic",
    ),
    # Legacy model IDs (for compatibility)
    "claude-3-opus-20240229": ModelPricing(
        input_price=Decimal("15.00"),
        output_price=Decimal("75.00"),
        provider="anthropic",
    ),
    "claude-3-sonnet-20240229": ModelPricing(
        input_price=Decimal("3.00"),
        output_price=Decimal("15.00"),
        provider="anthropic",
    ),
    "claude-3-haiku-20240307": ModelPricing(
        input_price=Decimal("0.25"),
        output_price=Decimal("1.25"),
        provider="anthropic",
    ),
    "claude-3-5-haiku-latest": ModelPricing(
        input_price=Decimal("0.80"),
        output_price=Decimal("4.00"),
        provider="anthropic",
    ),

    # ===========================================
    # GOOGLE GEMINI 3.0 SERIES (Preview)
    # ===========================================
    "gemini-3-pro-preview": ModelPricing(
        input_price=Decimal("2.00"),  # $4.00 for >200k context
        output_price=Decimal("12.00"),  # $18.00 for >200k context
        provider="google",
    ),
    "gemini-3-flash-preview": ModelPricing(
        input_price=Decimal("0.50"),
        output_price=Decimal("3.00"),
        provider="google",
    ),

    # ===========================================
    # GOOGLE GEMINI 2.5 SERIES (Stable)
    # ===========================================
    "gemini-2.5-pro": ModelPricing(
        input_price=Decimal("1.25"),  # $2.50 for >200k context
        output_price=Decimal("10.00"),  # $15.00 for >200k context
        provider="google",
    ),
    "gemini-2.5-flash": ModelPricing(
        input_price=Decimal("0.30"),
        output_price=Decimal("2.50"),
        provider="google",
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        input_price=Decimal("0.10"),
        output_price=Decimal("0.40"),
        provider="google",
    ),
    # Gemini Computer Use
    "gemini-2.5-computer-use-preview-10-2025": ModelPricing(
        input_price=Decimal("1.25"),
        output_price=Decimal("5.00"),
        provider="google",
    ),

    # ===========================================
    # GOOGLE GEMINI 2.0 SERIES (Legacy)
    # ===========================================
    "gemini-2.0-flash": ModelPricing(
        input_price=Decimal("0.10"),
        output_price=Decimal("0.40"),
        provider="google",
    ),
    "gemini-2.0-flash-lite": ModelPricing(
        input_price=Decimal("0.075"),
        output_price=Decimal("0.30"),
        provider="google",
    ),

    # Legacy Gemini 1.5 (for compatibility)
    "gemini-1.5-flash-latest": ModelPricing(
        input_price=Decimal("0.075"),
        output_price=Decimal("0.30"),
        provider="google",
    ),
    "gemini-1.5-pro-latest": ModelPricing(
        input_price=Decimal("1.25"),
        output_price=Decimal("5.00"),
        provider="google",
    ),

    # ===========================================
    # OPENAI MODELS
    # ===========================================
    "gpt-4o": ModelPricing(
        input_price=Decimal("2.50"),
        output_price=Decimal("10.00"),
        provider="openai",
    ),
    "gpt-4o-mini": ModelPricing(
        input_price=Decimal("0.15"),
        output_price=Decimal("0.60"),
        provider="openai",
    ),
    "o1": ModelPricing(
        input_price=Decimal("15.00"),
        output_price=Decimal("60.00"),
        provider="openai",
    ),

    # ===========================================
    # GROQ (Fast Llama inference)
    # ===========================================
    "llama-3.1-8b-instant": ModelPricing(
        input_price=Decimal("0.05"),
        output_price=Decimal("0.08"),
        provider="groq",
    ),
    "llama-3.1-70b-versatile": ModelPricing(
        input_price=Decimal("0.59"),
        output_price=Decimal("0.79"),
        provider="groq",
    ),

    # ===========================================
    # TOGETHER (DeepSeek)
    # ===========================================
    "deepseek-ai/DeepSeek-V3": ModelPricing(
        input_price=Decimal("0.27"),
        output_price=Decimal("1.10"),
        provider="together",
    ),

    # ===========================================
    # CLOUDFLARE WORKERS AI (Free tier)
    # ===========================================
    "workers-ai": ModelPricing(
        input_price=Decimal("0.00"),
        output_price=Decimal("0.00"),
        provider="cloudflare",
    ),
}

# Default pricing for unknown models (conservative estimate)
DEFAULT_PRICING = ModelPricing(
    input_price=Decimal("3.00"),
    output_price=Decimal("15.00"),
    provider="unknown",
)


@dataclass
class UsageRecord:
    """Record of a single AI API call."""

    request_id: str
    organization_id: str
    project_id: Optional[str]
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    task_type: TaskType
    latency_ms: Optional[int] = None
    cached: bool = False
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class BudgetStatus:
    """Current budget status for an organization."""

    has_daily_budget: bool
    has_monthly_budget: bool
    daily_remaining: Decimal
    monthly_remaining: Decimal
    daily_limit: Decimal
    monthly_limit: Decimal
    daily_used: Decimal
    monthly_used: Decimal


class AICostTracker:
    """Service for tracking AI costs and enforcing budgets."""

    def __init__(self):
        self._supabase = get_supabase_client()

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> Decimal:
        """Calculate cost in USD for a given usage.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-5")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD (Decimal for precision)
        """
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)

        input_cost = (Decimal(input_tokens) / Decimal("1000000")) * pricing.input_price
        output_cost = (
            Decimal(output_tokens) / Decimal("1000000")
        ) * pricing.output_price

        total_cost = input_cost + output_cost

        logger.debug(
            "Calculated AI cost",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=float(input_cost),
            output_cost=float(output_cost),
            total_cost=float(total_cost),
        )

        return total_cost

    async def check_budget(self, organization_id: str) -> BudgetStatus:
        """Check if organization has remaining AI budget.

        Args:
            organization_id: UUID of the organization

        Returns:
            BudgetStatus with remaining budget information
        """
        if not self._supabase.is_configured:
            # Return unlimited budget if Supabase not configured
            return BudgetStatus(
                has_daily_budget=True,
                has_monthly_budget=True,
                daily_remaining=Decimal("999999"),
                monthly_remaining=Decimal("999999"),
                daily_limit=Decimal("999999"),
                monthly_limit=Decimal("999999"),
                daily_used=Decimal("0"),
                monthly_used=Decimal("0"),
            )

        # Query organization budget
        result = await self._supabase.select(
            "organizations",
            columns="ai_budget_daily,ai_budget_monthly,ai_spend_today,ai_spend_this_month",
            filters={"id": f"eq.{organization_id}"},
        )

        if result.get("error") or not result.get("data"):
            logger.warning(
                "Failed to get organization budget",
                organization_id=organization_id,
                error=result.get("error"),
            )
            # Return default budget on error
            return BudgetStatus(
                has_daily_budget=True,
                has_monthly_budget=True,
                daily_remaining=Decimal("1.00"),
                monthly_remaining=Decimal("25.00"),
                daily_limit=Decimal("1.00"),
                monthly_limit=Decimal("25.00"),
                daily_used=Decimal("0"),
                monthly_used=Decimal("0"),
            )

        org = result["data"][0] if result["data"] else {}

        daily_limit = Decimal(str(org.get("ai_budget_daily", 1.0)))
        monthly_limit = Decimal(str(org.get("ai_budget_monthly", 25.0)))
        daily_used = Decimal(str(org.get("ai_spend_today", 0)))
        monthly_used = Decimal(str(org.get("ai_spend_this_month", 0)))

        daily_remaining = max(Decimal("0"), daily_limit - daily_used)
        monthly_remaining = max(Decimal("0"), monthly_limit - monthly_used)

        return BudgetStatus(
            has_daily_budget=daily_remaining > 0,
            has_monthly_budget=monthly_remaining > 0,
            daily_remaining=daily_remaining,
            monthly_remaining=monthly_remaining,
            daily_limit=daily_limit,
            monthly_limit=monthly_limit,
            daily_used=daily_used,
            monthly_used=monthly_used,
        )

    async def record_usage(
        self,
        organization_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_type: TaskType,
        project_id: Optional[str] = None,
        latency_ms: Optional[int] = None,
        cached: bool = False,
        metadata: Optional[dict] = None,
        request_id: Optional[str] = None,
    ) -> UsageRecord:
        """Record AI usage and update organization spend.

        Args:
            organization_id: UUID of the organization
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            task_type: Type of AI task
            project_id: Optional project UUID
            latency_ms: API call latency in milliseconds
            cached: Whether response was from cache
            metadata: Additional metadata to store
            request_id: Optional request ID for idempotency

        Returns:
            UsageRecord with calculated cost
        """
        # Calculate cost
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        provider = MODEL_PRICING.get(model, DEFAULT_PRICING).provider

        # Generate request ID for idempotency
        if not request_id:
            request_id = str(uuid.uuid4())

        record = UsageRecord(
            request_id=request_id,
            organization_id=organization_id,
            project_id=project_id,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            task_type=task_type,
            latency_ms=latency_ms,
            cached=cached,
            metadata=metadata or {},
        )

        # Persist to database
        if self._supabase.is_configured:
            await self._persist_usage(record)

        logger.info(
            "Recorded AI usage",
            request_id=request_id,
            organization_id=organization_id,
            model=model,
            total_tokens=record.total_tokens,
            cost_usd=float(cost),
            task_type=task_type.value,
            cached=cached,
        )

        return record

    async def _persist_usage(self, record: UsageRecord) -> None:
        """Persist usage record to Supabase."""
        # Insert into ai_usage table
        usage_data = {
            "request_id": record.request_id,
            "organization_id": record.organization_id,
            "project_id": record.project_id,
            "model": record.model,
            "provider": record.provider,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "cost_usd": float(record.cost_usd),
            "task_type": record.task_type.value,
            "latency_ms": record.latency_ms,
            "cached": record.cached,
            "metadata": record.metadata,
        }

        result = await self._supabase.insert("ai_usage", usage_data)

        if result.get("error"):
            logger.warning(
                "Failed to persist AI usage",
                request_id=record.request_id,
                error=result.get("error"),
            )
        else:
            logger.debug("Persisted AI usage", request_id=record.request_id)

    async def get_usage_summary(
        self,
        organization_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """Get usage summary for an organization.

        Args:
            organization_id: UUID of the organization
            start_date: Start of period (default: today)
            end_date: End of period (default: now)

        Returns:
            Summary with total tokens, costs, and breakdowns
        """
        if not self._supabase.is_configured:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "by_model": {},
                "by_task": {},
            }

        # Build date filter
        filters = {"organization_id": f"eq.{organization_id}"}
        if start_date:
            filters["created_at"] = f"gte.{start_date.isoformat()}"
        if end_date:
            filters["created_at"] = f"lte.{end_date.isoformat()}"

        result = await self._supabase.select(
            "ai_usage_daily",
            columns="*",
            filters=filters,
        )

        if result.get("error") or not result.get("data"):
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "by_model": {},
                "by_task": {},
            }

        # Aggregate results
        total_requests = 0
        total_input = 0
        total_output = 0
        total_cost = Decimal("0")
        by_model: dict = {}
        by_task: dict = {}

        for day in result["data"]:
            total_requests += day.get("total_requests", 0)
            total_input += day.get("total_input_tokens", 0)
            total_output += day.get("total_output_tokens", 0)
            total_cost += Decimal(str(day.get("total_cost_usd", 0)))

            # Merge model breakdowns
            for model, stats in day.get("usage_by_model", {}).items():
                if model not in by_model:
                    by_model[model] = {"requests": 0, "tokens": 0, "cost": 0.0}
                by_model[model]["requests"] += stats.get("requests", 0)
                by_model[model]["tokens"] += stats.get("tokens", 0)
                by_model[model]["cost"] += stats.get("cost", 0)

            # Merge task breakdowns
            for task, stats in day.get("usage_by_task", {}).items():
                if task not in by_task:
                    by_task[task] = {"requests": 0, "cost": 0.0}
                by_task[task]["requests"] += stats.get("requests", 0)
                by_task[task]["cost"] += stats.get("cost", 0)

        return {
            "total_requests": total_requests,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": float(total_cost),
            "by_model": by_model,
            "by_task": by_task,
        }

    def estimate_cost(
        self, model: str, estimated_input_tokens: int, estimated_output_tokens: int
    ) -> dict:
        """Estimate cost for a planned AI call.

        Args:
            model: Model identifier
            estimated_input_tokens: Expected input tokens
            estimated_output_tokens: Expected output tokens

        Returns:
            Estimate with cost breakdown
        """
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        cost = self.calculate_cost(model, estimated_input_tokens, estimated_output_tokens)

        return {
            "model": model,
            "provider": pricing.provider,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens,
            "input_price_per_million": float(pricing.input_price),
            "output_price_per_million": float(pricing.output_price),
            "estimated_cost_usd": float(cost),
        }


# Global instance
_cost_tracker: Optional[AICostTracker] = None


def get_cost_tracker() -> AICostTracker:
    """Get or create global AI cost tracker."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = AICostTracker()
    return _cost_tracker


# Convenience functions
async def record_ai_usage(
    organization_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    task_type: TaskType,
    **kwargs,
) -> UsageRecord:
    """Record AI usage (convenience function)."""
    tracker = get_cost_tracker()
    return await tracker.record_usage(
        organization_id=organization_id,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        task_type=task_type,
        **kwargs,
    )


async def check_ai_budget(organization_id: str) -> BudgetStatus:
    """Check AI budget (convenience function)."""
    tracker = get_cost_tracker()
    return await tracker.check_budget(organization_id)


def calculate_ai_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """Calculate AI cost (convenience function)."""
    tracker = get_cost_tracker()
    return tracker.calculate_cost(model, input_tokens, output_tokens)
