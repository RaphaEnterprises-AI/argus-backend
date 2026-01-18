"""Tests for the AI Cost Tracker service.

This module tests:
- Cost calculation for different AI models
- Budget checking and enforcement
- Usage recording and persistence
- Usage summary aggregation
- Cost estimation
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import uuid


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_type_values(self, mock_env_vars):
        """Test that TaskType enum has expected values."""
        from src.services.ai_cost_tracker import TaskType

        assert TaskType.ERROR_ANALYSIS == "error_analysis"
        assert TaskType.TEST_GENERATION == "test_generation"
        assert TaskType.CODE_REVIEW == "code_review"
        assert TaskType.SELF_HEALING == "self_healing"
        assert TaskType.CORRELATION == "correlation"
        assert TaskType.RISK_ASSESSMENT == "risk_assessment"
        assert TaskType.PATTERN_MATCHING == "pattern_matching"
        assert TaskType.OTHER == "other"

    def test_task_type_is_string_enum(self, mock_env_vars):
        """Test that TaskType values are strings."""
        from src.services.ai_cost_tracker import TaskType

        for task_type in TaskType:
            assert isinstance(task_type.value, str)


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    def test_model_pricing_creation(self, mock_env_vars):
        """Test creating a ModelPricing instance."""
        from src.services.ai_cost_tracker import ModelPricing

        pricing = ModelPricing(
            input_price=Decimal("3.00"),
            output_price=Decimal("15.00"),
            provider="anthropic",
        )

        assert pricing.input_price == Decimal("3.00")
        assert pricing.output_price == Decimal("15.00")
        assert pricing.provider == "anthropic"

    def test_model_pricing_default_provider(self, mock_env_vars):
        """Test ModelPricing default provider."""
        from src.services.ai_cost_tracker import ModelPricing

        pricing = ModelPricing(
            input_price=Decimal("1.00"),
            output_price=Decimal("5.00"),
        )

        assert pricing.provider == "anthropic"


class TestModelPricingDict:
    """Tests for MODEL_PRICING dictionary."""

    def test_claude_models_present(self, mock_env_vars):
        """Test that Claude models are in pricing dict."""
        from src.services.ai_cost_tracker import MODEL_PRICING

        assert "claude-opus-4-5" in MODEL_PRICING
        assert "claude-sonnet-4-5" in MODEL_PRICING
        assert "claude-haiku-4-5" in MODEL_PRICING

    def test_gemini_models_present(self, mock_env_vars):
        """Test that Gemini models are in pricing dict."""
        from src.services.ai_cost_tracker import MODEL_PRICING

        assert "gemini-2.5-pro" in MODEL_PRICING
        assert "gemini-2.5-flash" in MODEL_PRICING

    def test_openai_models_present(self, mock_env_vars):
        """Test that OpenAI models are in pricing dict."""
        from src.services.ai_cost_tracker import MODEL_PRICING

        assert "gpt-4o" in MODEL_PRICING
        assert "gpt-4o-mini" in MODEL_PRICING

    def test_free_tier_model(self, mock_env_vars):
        """Test that Cloudflare Workers AI has zero pricing."""
        from src.services.ai_cost_tracker import MODEL_PRICING

        workers_pricing = MODEL_PRICING["workers-ai"]
        assert workers_pricing.input_price == Decimal("0.00")
        assert workers_pricing.output_price == Decimal("0.00")


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""

    def test_usage_record_creation(self, mock_env_vars):
        """Test creating a UsageRecord instance."""
        from src.services.ai_cost_tracker import UsageRecord, TaskType

        record = UsageRecord(
            request_id="test-123",
            organization_id="org-456",
            project_id="proj-789",
            model="claude-sonnet-4-5",
            provider="anthropic",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=Decimal("0.01"),
            task_type=TaskType.TEST_GENERATION,
        )

        assert record.request_id == "test-123"
        assert record.organization_id == "org-456"
        assert record.project_id == "proj-789"
        assert record.model == "claude-sonnet-4-5"
        assert record.provider == "anthropic"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.cost_usd == Decimal("0.01")
        assert record.task_type == TaskType.TEST_GENERATION

    def test_usage_record_total_tokens(self, mock_env_vars):
        """Test UsageRecord total_tokens property."""
        from src.services.ai_cost_tracker import UsageRecord, TaskType

        record = UsageRecord(
            request_id="test-123",
            organization_id="org-456",
            project_id=None,
            model="claude-sonnet-4-5",
            provider="anthropic",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=Decimal("0.01"),
            task_type=TaskType.OTHER,
        )

        assert record.total_tokens == 1500

    def test_usage_record_default_values(self, mock_env_vars):
        """Test UsageRecord default values."""
        from src.services.ai_cost_tracker import UsageRecord, TaskType

        record = UsageRecord(
            request_id="test-123",
            organization_id="org-456",
            project_id=None,
            model="claude-sonnet-4-5",
            provider="anthropic",
            input_tokens=100,
            output_tokens=50,
            cost_usd=Decimal("0.001"),
            task_type=TaskType.OTHER,
        )

        assert record.latency_ms is None
        assert record.cached is False
        assert record.metadata == {}
        assert record.created_at is not None


class TestBudgetStatus:
    """Tests for BudgetStatus dataclass."""

    def test_budget_status_creation(self, mock_env_vars):
        """Test creating a BudgetStatus instance."""
        from src.services.ai_cost_tracker import BudgetStatus

        status = BudgetStatus(
            has_daily_budget=True,
            has_monthly_budget=True,
            daily_remaining=Decimal("5.00"),
            monthly_remaining=Decimal("20.00"),
            daily_limit=Decimal("10.00"),
            monthly_limit=Decimal("100.00"),
            daily_used=Decimal("5.00"),
            monthly_used=Decimal("80.00"),
        )

        assert status.has_daily_budget is True
        assert status.has_monthly_budget is True
        assert status.daily_remaining == Decimal("5.00")
        assert status.monthly_remaining == Decimal("20.00")

    def test_budget_status_exhausted(self, mock_env_vars):
        """Test BudgetStatus when budget is exhausted."""
        from src.services.ai_cost_tracker import BudgetStatus

        status = BudgetStatus(
            has_daily_budget=False,
            has_monthly_budget=True,
            daily_remaining=Decimal("0.00"),
            monthly_remaining=Decimal("5.00"),
            daily_limit=Decimal("10.00"),
            monthly_limit=Decimal("100.00"),
            daily_used=Decimal("10.00"),
            monthly_used=Decimal("95.00"),
        )

        assert status.has_daily_budget is False
        assert status.daily_remaining == Decimal("0.00")


class TestAICostTracker:
    """Tests for AICostTracker class."""

    @pytest.fixture
    def mock_supabase(self):
        """Create a mock Supabase client."""
        mock_client = MagicMock()
        mock_client.is_configured = True
        mock_client.select = AsyncMock()
        mock_client.insert = AsyncMock()
        return mock_client

    @pytest.fixture
    def cost_tracker(self, mock_supabase):
        """Create an AICostTracker with mocked Supabase."""
        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker
            tracker = AICostTracker()
            return tracker

    def test_calculate_cost_known_model(self, mock_env_vars, cost_tracker):
        """Test cost calculation for a known model."""
        cost = cost_tracker.calculate_cost(
            model="claude-sonnet-4-5",
            input_tokens=1000000,
            output_tokens=100000,
        )

        # 1M input tokens @ $3.00 = $3.00
        # 100K output tokens @ $15.00 = $1.50
        # Total: $4.50
        expected = Decimal("3.00") + Decimal("1.50")
        assert cost == expected

    def test_calculate_cost_unknown_model(self, mock_env_vars, cost_tracker):
        """Test cost calculation falls back to default pricing."""
        cost = cost_tracker.calculate_cost(
            model="unknown-model-xyz",
            input_tokens=1000000,
            output_tokens=100000,
        )

        # Should use DEFAULT_PRICING: $3.00 input, $15.00 output per 1M
        expected = Decimal("3.00") + Decimal("1.50")
        assert cost == expected

    def test_calculate_cost_zero_tokens(self, mock_env_vars, cost_tracker):
        """Test cost calculation with zero tokens."""
        cost = cost_tracker.calculate_cost(
            model="claude-sonnet-4-5",
            input_tokens=0,
            output_tokens=0,
        )

        assert cost == Decimal("0")

    def test_calculate_cost_free_model(self, mock_env_vars, cost_tracker):
        """Test cost calculation for free tier model."""
        cost = cost_tracker.calculate_cost(
            model="workers-ai",
            input_tokens=1000000,
            output_tokens=1000000,
        )

        assert cost == Decimal("0")

    def test_calculate_cost_small_tokens(self, mock_env_vars, cost_tracker):
        """Test cost calculation with small token counts."""
        cost = cost_tracker.calculate_cost(
            model="claude-sonnet-4-5",
            input_tokens=1000,
            output_tokens=500,
        )

        # 1K input tokens @ $3.00/1M = $0.003
        # 500 output tokens @ $15.00/1M = $0.0075
        # Total: $0.0105
        expected = Decimal("0.003") + Decimal("0.0075")
        assert cost == expected

    @pytest.mark.asyncio
    async def test_check_budget_supabase_not_configured(self, mock_env_vars):
        """Test check_budget when Supabase is not configured."""
        mock_client = MagicMock()
        mock_client.is_configured = False

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_client):
            from src.services.ai_cost_tracker import AICostTracker
            tracker = AICostTracker()
            status = await tracker.check_budget("org-123")

        assert status.has_daily_budget is True
        assert status.has_monthly_budget is True
        assert status.daily_remaining == Decimal("999999")
        assert status.monthly_remaining == Decimal("999999")

    @pytest.mark.asyncio
    async def test_check_budget_organization_not_found(self, mock_env_vars, mock_supabase):
        """Test check_budget when organization is not found."""
        mock_supabase.select = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker
            tracker = AICostTracker()
            status = await tracker.check_budget("org-nonexistent")

        # Should return default budget
        assert status.has_daily_budget is True
        assert status.daily_limit == Decimal("1.00")
        assert status.monthly_limit == Decimal("25.00")

    @pytest.mark.asyncio
    async def test_check_budget_success(self, mock_env_vars, mock_supabase):
        """Test check_budget with valid organization data."""
        mock_supabase.select = AsyncMock(return_value={
            "data": [{
                "ai_budget_daily": 10.0,
                "ai_budget_monthly": 100.0,
                "ai_spend_today": 3.0,
                "ai_spend_this_month": 40.0,
            }],
            "error": None,
        })

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker
            tracker = AICostTracker()
            status = await tracker.check_budget("org-123")

        assert status.has_daily_budget is True
        assert status.has_monthly_budget is True
        assert status.daily_remaining == Decimal("7.0")
        assert status.monthly_remaining == Decimal("60.0")
        assert status.daily_used == Decimal("3.0")
        assert status.monthly_used == Decimal("40.0")

    @pytest.mark.asyncio
    async def test_check_budget_daily_exhausted(self, mock_env_vars, mock_supabase):
        """Test check_budget when daily budget is exhausted."""
        mock_supabase.select = AsyncMock(return_value={
            "data": [{
                "ai_budget_daily": 5.0,
                "ai_budget_monthly": 100.0,
                "ai_spend_today": 6.0,  # Over budget
                "ai_spend_this_month": 30.0,
            }],
            "error": None,
        })

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker
            tracker = AICostTracker()
            status = await tracker.check_budget("org-123")

        assert status.has_daily_budget is False
        assert status.daily_remaining == Decimal("0")

    @pytest.mark.asyncio
    async def test_check_budget_error(self, mock_env_vars, mock_supabase):
        """Test check_budget when database query fails."""
        mock_supabase.select = AsyncMock(return_value={
            "data": None,
            "error": "Database connection failed",
        })

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker
            tracker = AICostTracker()
            status = await tracker.check_budget("org-123")

        # Should return default budget on error
        assert status.daily_limit == Decimal("1.00")
        assert status.monthly_limit == Decimal("25.00")

    @pytest.mark.asyncio
    async def test_record_usage_success(self, mock_env_vars, mock_supabase):
        """Test recording usage successfully."""
        mock_supabase.insert = AsyncMock(return_value={"data": {}, "error": None})

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker, TaskType
            tracker = AICostTracker()

            record = await tracker.record_usage(
                organization_id="org-123",
                model="claude-sonnet-4-5",
                input_tokens=1000,
                output_tokens=500,
                task_type=TaskType.TEST_GENERATION,
                project_id="proj-456",
                latency_ms=250,
            )

        assert record.organization_id == "org-123"
        assert record.model == "claude-sonnet-4-5"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.task_type == TaskType.TEST_GENERATION
        assert record.project_id == "proj-456"
        assert record.latency_ms == 250
        assert record.cost_usd > 0
        mock_supabase.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_usage_with_custom_request_id(self, mock_env_vars, mock_supabase):
        """Test recording usage with a custom request ID."""
        mock_supabase.insert = AsyncMock(return_value={"data": {}, "error": None})

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker, TaskType
            tracker = AICostTracker()

            record = await tracker.record_usage(
                organization_id="org-123",
                model="claude-sonnet-4-5",
                input_tokens=100,
                output_tokens=50,
                task_type=TaskType.OTHER,
                request_id="custom-req-id",
            )

        assert record.request_id == "custom-req-id"

    @pytest.mark.asyncio
    async def test_record_usage_generates_request_id(self, mock_env_vars, mock_supabase):
        """Test that record_usage generates a request ID if not provided."""
        mock_supabase.insert = AsyncMock(return_value={"data": {}, "error": None})

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker, TaskType
            tracker = AICostTracker()

            record = await tracker.record_usage(
                organization_id="org-123",
                model="claude-sonnet-4-5",
                input_tokens=100,
                output_tokens=50,
                task_type=TaskType.OTHER,
            )

        assert record.request_id is not None
        assert len(record.request_id) > 0

    @pytest.mark.asyncio
    async def test_record_usage_supabase_not_configured(self, mock_env_vars):
        """Test recording usage when Supabase is not configured."""
        mock_client = MagicMock()
        mock_client.is_configured = False

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_client):
            from src.services.ai_cost_tracker import AICostTracker, TaskType
            tracker = AICostTracker()

            record = await tracker.record_usage(
                organization_id="org-123",
                model="claude-sonnet-4-5",
                input_tokens=100,
                output_tokens=50,
                task_type=TaskType.OTHER,
            )

        # Should still return a record, just not persisted
        assert record.organization_id == "org-123"

    @pytest.mark.asyncio
    async def test_record_usage_persist_failure(self, mock_env_vars, mock_supabase):
        """Test recording usage when persistence fails."""
        mock_supabase.insert = AsyncMock(return_value={
            "data": None,
            "error": "Insert failed",
        })

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker, TaskType
            tracker = AICostTracker()

            # Should not raise, just log warning
            record = await tracker.record_usage(
                organization_id="org-123",
                model="claude-sonnet-4-5",
                input_tokens=100,
                output_tokens=50,
                task_type=TaskType.OTHER,
            )

        assert record is not None

    @pytest.mark.asyncio
    async def test_record_usage_with_cached_flag(self, mock_env_vars, mock_supabase):
        """Test recording usage with cached flag."""
        mock_supabase.insert = AsyncMock(return_value={"data": {}, "error": None})

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker, TaskType
            tracker = AICostTracker()

            record = await tracker.record_usage(
                organization_id="org-123",
                model="claude-sonnet-4-5",
                input_tokens=100,
                output_tokens=50,
                task_type=TaskType.OTHER,
                cached=True,
            )

        assert record.cached is True

    @pytest.mark.asyncio
    async def test_record_usage_with_metadata(self, mock_env_vars, mock_supabase):
        """Test recording usage with custom metadata."""
        mock_supabase.insert = AsyncMock(return_value={"data": {}, "error": None})

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker, TaskType
            tracker = AICostTracker()

            record = await tracker.record_usage(
                organization_id="org-123",
                model="claude-sonnet-4-5",
                input_tokens=100,
                output_tokens=50,
                task_type=TaskType.OTHER,
                metadata={"test_id": "test-001", "step": 5},
            )

        assert record.metadata == {"test_id": "test-001", "step": 5}

    @pytest.mark.asyncio
    async def test_get_usage_summary_supabase_not_configured(self, mock_env_vars):
        """Test get_usage_summary when Supabase is not configured."""
        mock_client = MagicMock()
        mock_client.is_configured = False

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_client):
            from src.services.ai_cost_tracker import AICostTracker
            tracker = AICostTracker()
            summary = await tracker.get_usage_summary("org-123")

        assert summary["total_requests"] == 0
        assert summary["total_tokens"] == 0
        assert summary["total_cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_get_usage_summary_no_data(self, mock_env_vars, mock_supabase):
        """Test get_usage_summary when no data exists."""
        mock_supabase.select = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker
            tracker = AICostTracker()
            summary = await tracker.get_usage_summary("org-123")

        assert summary["total_requests"] == 0
        assert summary["total_cost_usd"] == 0.0
        assert summary["by_model"] == {}
        assert summary["by_task"] == {}

    @pytest.mark.asyncio
    async def test_get_usage_summary_with_data(self, mock_env_vars, mock_supabase):
        """Test get_usage_summary with aggregated data."""
        mock_supabase.select = AsyncMock(return_value={
            "data": [
                {
                    "total_requests": 10,
                    "total_input_tokens": 5000,
                    "total_output_tokens": 2500,
                    "total_cost_usd": 0.05,
                    "usage_by_model": {
                        "claude-sonnet-4-5": {"requests": 8, "tokens": 6000, "cost": 0.04},
                        "claude-haiku-4-5": {"requests": 2, "tokens": 1500, "cost": 0.01},
                    },
                    "usage_by_task": {
                        "test_generation": {"requests": 5, "cost": 0.03},
                        "error_analysis": {"requests": 5, "cost": 0.02},
                    },
                },
                {
                    "total_requests": 5,
                    "total_input_tokens": 2000,
                    "total_output_tokens": 1000,
                    "total_cost_usd": 0.02,
                    "usage_by_model": {
                        "claude-sonnet-4-5": {"requests": 5, "tokens": 3000, "cost": 0.02},
                    },
                    "usage_by_task": {
                        "test_generation": {"requests": 5, "cost": 0.02},
                    },
                },
            ],
            "error": None,
        })

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker
            tracker = AICostTracker()
            summary = await tracker.get_usage_summary("org-123")

        assert summary["total_requests"] == 15
        assert summary["total_input_tokens"] == 7000
        assert summary["total_output_tokens"] == 3500
        assert summary["total_tokens"] == 10500
        assert summary["total_cost_usd"] == 0.07
        assert "claude-sonnet-4-5" in summary["by_model"]
        assert "test_generation" in summary["by_task"]

    @pytest.mark.asyncio
    async def test_get_usage_summary_with_date_filters(self, mock_env_vars, mock_supabase):
        """Test get_usage_summary with date range filters."""
        mock_supabase.select = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_supabase):
            from src.services.ai_cost_tracker import AICostTracker
            tracker = AICostTracker()

            start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
            await tracker.get_usage_summary("org-123", start_date=start_date, end_date=end_date)

        mock_supabase.select.assert_called_once()

    def test_estimate_cost(self, mock_env_vars, cost_tracker):
        """Test cost estimation."""
        estimate = cost_tracker.estimate_cost(
            model="claude-sonnet-4-5",
            estimated_input_tokens=10000,
            estimated_output_tokens=5000,
        )

        assert estimate["model"] == "claude-sonnet-4-5"
        assert estimate["provider"] == "anthropic"
        assert estimate["estimated_input_tokens"] == 10000
        assert estimate["estimated_output_tokens"] == 5000
        assert estimate["estimated_total_tokens"] == 15000
        assert estimate["input_price_per_million"] == 3.0
        assert estimate["output_price_per_million"] == 15.0
        assert estimate["estimated_cost_usd"] > 0

    def test_estimate_cost_unknown_model(self, mock_env_vars, cost_tracker):
        """Test cost estimation for unknown model uses defaults."""
        estimate = cost_tracker.estimate_cost(
            model="future-model-xyz",
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
        )

        assert estimate["provider"] == "unknown"
        assert estimate["input_price_per_million"] == 3.0  # DEFAULT_PRICING


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_cost_tracker_singleton(self, mock_env_vars):
        """Test that get_cost_tracker returns a singleton."""
        import src.services.ai_cost_tracker as module

        # Reset global instance
        module._cost_tracker = None

        with patch("src.services.ai_cost_tracker.get_supabase_client"):
            tracker1 = module.get_cost_tracker()
            tracker2 = module.get_cost_tracker()

        assert tracker1 is tracker2

    @pytest.mark.asyncio
    async def test_record_ai_usage_convenience(self, mock_env_vars):
        """Test the record_ai_usage convenience function."""
        mock_client = MagicMock()
        mock_client.is_configured = False

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_client):
            import src.services.ai_cost_tracker as module
            module._cost_tracker = None

            from src.services.ai_cost_tracker import record_ai_usage, TaskType

            record = await record_ai_usage(
                organization_id="org-123",
                model="claude-sonnet-4-5",
                input_tokens=100,
                output_tokens=50,
                task_type=TaskType.CODE_REVIEW,
            )

        assert record.organization_id == "org-123"
        assert record.task_type == TaskType.CODE_REVIEW

    @pytest.mark.asyncio
    async def test_check_ai_budget_convenience(self, mock_env_vars):
        """Test the check_ai_budget convenience function."""
        mock_client = MagicMock()
        mock_client.is_configured = False

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_client):
            import src.services.ai_cost_tracker as module
            module._cost_tracker = None

            from src.services.ai_cost_tracker import check_ai_budget

            status = await check_ai_budget("org-123")

        assert status.has_daily_budget is True
        assert status.daily_remaining == Decimal("999999")

    def test_calculate_ai_cost_convenience(self, mock_env_vars):
        """Test the calculate_ai_cost convenience function."""
        mock_client = MagicMock()
        mock_client.is_configured = False

        with patch("src.services.ai_cost_tracker.get_supabase_client", return_value=mock_client):
            import src.services.ai_cost_tracker as module
            module._cost_tracker = None

            from src.services.ai_cost_tracker import calculate_ai_cost

            cost = calculate_ai_cost(
                model="claude-sonnet-4-5",
                input_tokens=1000,
                output_tokens=500,
            )

        assert cost > 0
