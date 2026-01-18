"""Tests for the AI-driven infrastructure optimizer service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from decimal import Decimal
import json


class TestRecommendationType:
    """Tests for RecommendationType enum."""

    def test_recommendation_types_exist(self):
        """Test that all recommendation types are defined."""
        from src.services.infra_optimizer import RecommendationType

        assert RecommendationType.SCALE_DOWN == "scale_down"
        assert RecommendationType.SCALE_UP == "scale_up"
        assert RecommendationType.RIGHT_SIZE == "right_size"
        assert RecommendationType.SCHEDULE_SCALING == "schedule_scaling"
        assert RecommendationType.CLEANUP_SESSIONS == "cleanup_sessions"
        assert RecommendationType.COST_ALERT == "cost_alert"
        assert RecommendationType.ANOMALY == "anomaly"


class TestRecommendationPriority:
    """Tests for RecommendationPriority enum."""

    def test_priority_levels_exist(self):
        """Test that all priority levels are defined."""
        from src.services.infra_optimizer import RecommendationPriority

        assert RecommendationPriority.CRITICAL == "critical"
        assert RecommendationPriority.HIGH == "high"
        assert RecommendationPriority.MEDIUM == "medium"
        assert RecommendationPriority.LOW == "low"


class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""

    def test_approval_statuses_exist(self):
        """Test that all approval statuses are defined."""
        from src.services.infra_optimizer import ApprovalStatus

        assert ApprovalStatus.PENDING == "pending"
        assert ApprovalStatus.APPROVED == "approved"
        assert ApprovalStatus.REJECTED == "rejected"
        assert ApprovalStatus.AUTO_APPLIED == "auto_applied"
        assert ApprovalStatus.EXPIRED == "expired"


class TestInfraRecommendation:
    """Tests for InfraRecommendation dataclass."""

    def test_infra_recommendation_creation(self):
        """Test creating an InfraRecommendation instance."""
        from src.services.infra_optimizer import (
            InfraRecommendation,
            RecommendationType,
            RecommendationPriority,
            ApprovalStatus,
        )

        rec = InfraRecommendation(
            id="rec-123",
            type=RecommendationType.SCALE_DOWN,
            priority=RecommendationPriority.MEDIUM,
            title="Scale down Chrome nodes",
            description="Chrome nodes are underutilized",
            estimated_savings_monthly=Decimal("50.00"),
            confidence=0.85,
            action={"target": "chrome", "operation": "set_min_replicas", "params": {"min": 2}},
            reasoning="CPU utilization is below 20%",
            metrics_snapshot={"cpu": 15},
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=7),
            status=ApprovalStatus.PENDING,
            org_id="org-123",
        )

        assert rec.id == "rec-123"
        assert rec.type == RecommendationType.SCALE_DOWN
        assert rec.estimated_savings_monthly == Decimal("50.00")
        assert rec.status == ApprovalStatus.PENDING


class TestCostReport:
    """Tests for CostReport dataclass."""

    def test_cost_report_creation(self):
        """Test creating a CostReport instance."""
        from src.services.infra_optimizer import CostReport

        report = CostReport(
            period_start=datetime.now() - timedelta(days=7),
            period_end=datetime.now(),
            total_cost=Decimal("100.00"),
            breakdown={"compute": Decimal("80.00"), "network": Decimal("20.00")},
            daily_costs=[(datetime.now(), Decimal("14.28"))],
            projected_monthly=Decimal("428.40"),
            comparison_to_browserstack=Decimal("990.00"),
            savings_achieved=Decimal("561.60"),
            recommendations=[],
        )

        assert report.total_cost == Decimal("100.00")
        assert report.savings_achieved == Decimal("561.60")


class TestDemandForecast:
    """Tests for DemandForecast dataclass."""

    def test_demand_forecast_creation(self):
        """Test creating a DemandForecast instance."""
        from src.services.infra_optimizer import DemandForecast

        now = datetime.now()
        forecast = DemandForecast(
            forecast_start=now,
            forecast_end=now + timedelta(hours=24),
            hourly_predictions=[{"hour": "2024-01-01T10:00:00", "predicted_sessions": 5.0, "confidence": 0.8}],
            peak_times=[now + timedelta(hours=10)],
            recommended_min_replicas={"chrome": 3, "firefox": 1, "edge": 1},
            confidence=0.7,
        )

        assert forecast.confidence == 0.7
        assert forecast.recommended_min_replicas["chrome"] == 3


class TestAnomaly:
    """Tests for Anomaly dataclass."""

    def test_anomaly_creation(self):
        """Test creating an Anomaly instance."""
        from src.services.infra_optimizer import Anomaly, RecommendationPriority

        anomaly = Anomaly(
            id="anomaly-123",
            type="stuck_queue",
            severity=RecommendationPriority.HIGH,
            description="Session queue has 10 pending sessions",
            detected_at=datetime.now(),
            metrics={"queued": 10, "wait_time": 120},
            suggested_action="Scale up browser nodes",
        )

        assert anomaly.type == "stuck_queue"
        assert anomaly.severity == RecommendationPriority.HIGH


class TestAIInfraOptimizerInit:
    """Tests for AIInfraOptimizer initialization."""

    def test_init_with_defaults(self, mock_env_vars):
        """Test initialization with default parameters."""
        from src.services.infra_optimizer import AIInfraOptimizer

        with patch("src.services.infra_optimizer.create_prometheus_collector") as mock_prometheus:
            with patch("anthropic.Anthropic") as mock_anthropic:
                optimizer = AIInfraOptimizer()

        assert optimizer.model == "claude-sonnet-4-5-20241022"

    def test_init_with_custom_params(self, mock_env_vars):
        """Test initialization with custom parameters."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
            model="claude-opus-4-5",
        )

        assert optimizer.prometheus is mock_prometheus
        assert optimizer.client is mock_anthropic
        assert optimizer.model == "claude-opus-4-5"

    def test_supabase_lazy_loading(self, mock_env_vars):
        """Test that Supabase client is lazily loaded."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        assert optimizer._supabase is None

        with patch("src.services.infra_optimizer.get_supabase_client") as mock_get:
            mock_get.return_value = MagicMock()
            _ = optimizer.supabase

        mock_get.assert_called_once()


class TestAIInfraOptimizerAnalyzeAndRecommend:
    """Tests for analyze_and_recommend method."""

    @pytest.mark.asyncio
    async def test_analyze_and_recommend_success(self, mock_env_vars):
        """Test successful analysis and recommendation generation."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        # Create mock infrastructure snapshot
        mock_snapshot = MagicMock()
        mock_snapshot.selenium = MagicMock(
            sessions_queued=0,
            sessions_active=5,
            nodes_available=3,
            nodes_total=5,
            avg_session_duration_seconds=30.0,
            queue_wait_time_seconds=2.0,
        )
        mock_snapshot.chrome_nodes = MagicMock(
            replicas_current=3,
            replicas_min=2,
            replicas_max=10,
            cpu_utilization=MagicMock(cpu_usage_percent=40.0),
            memory_utilization=MagicMock(memory_usage_percent=50.0),
        )
        mock_snapshot.firefox_nodes = MagicMock(
            replicas_current=1,
            replicas_min=1,
            replicas_max=5,
            cpu_utilization=MagicMock(cpu_usage_percent=30.0),
            memory_utilization=MagicMock(memory_usage_percent=40.0),
        )
        mock_snapshot.edge_nodes = MagicMock(
            replicas_current=1,
            replicas_min=1,
            replicas_max=5,
            cpu_utilization=MagicMock(cpu_usage_percent=25.0),
            memory_utilization=MagicMock(memory_usage_percent=35.0),
        )
        mock_snapshot.total_pods = 10
        mock_snapshot.total_nodes = 3
        mock_snapshot.cluster_cpu_utilization = 35.0
        mock_snapshot.cluster_memory_utilization = 45.0

        mock_prometheus.get_infrastructure_snapshot = AsyncMock(return_value=mock_snapshot)
        mock_prometheus.get_usage_patterns = AsyncMock(return_value={
            "hourly_averages": [1.0] * 24,
            "daily_averages": [1.0] * 7,
            "peak_hour": 10,
            "min_hour": 3,
            "peak_day": 2,
            "min_day": 6,
        })
        mock_prometheus.get_test_execution_metrics = AsyncMock(return_value={
            "total_tests": 100,
            "successful_tests": 95,
            "failed_tests": 5,
            "success_rate": 95.0,
            "avg_duration_seconds": 10.0,
            "p95_duration_seconds": 30.0,
        })

        # Mock AI response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([{
            "type": "scale_down",
            "priority": "medium",
            "title": "Scale down during off-hours",
            "description": "Reduce replicas during low usage",
            "estimated_savings_monthly": 25.0,
            "confidence": 0.8,
            "action": {"target": "chrome", "operation": "set_min_replicas", "params": {"min": 1}},
            "reasoning": "Low usage during night hours",
        }]))]
        mock_anthropic.messages.create.return_value = mock_response

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        # Mock store_recommendation to avoid Supabase calls
        with patch.object(optimizer, "_store_recommendation", new_callable=AsyncMock):
            recommendations = await optimizer.analyze_and_recommend("org-123")

        assert len(recommendations) == 1
        assert recommendations[0].type.value == "scale_down"

    @pytest.mark.asyncio
    async def test_analyze_and_recommend_handles_snapshot_error(self, mock_env_vars):
        """Test handling of infrastructure snapshot errors."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_prometheus.get_infrastructure_snapshot = AsyncMock(
            side_effect=Exception("Prometheus unavailable")
        )

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        recommendations = await optimizer.analyze_and_recommend("org-123")

        assert recommendations == []

    @pytest.mark.asyncio
    async def test_analyze_and_recommend_handles_usage_patterns_error(self, mock_env_vars):
        """Test handling of usage patterns errors."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_snapshot = MagicMock()
        mock_snapshot.selenium = MagicMock(
            sessions_queued=0,
            sessions_active=5,
            nodes_available=3,
            nodes_total=5,
            avg_session_duration_seconds=30.0,
            queue_wait_time_seconds=2.0,
        )
        mock_snapshot.chrome_nodes = MagicMock(
            replicas_current=3,
            replicas_min=2,
            replicas_max=10,
            cpu_utilization=MagicMock(cpu_usage_percent=40.0),
            memory_utilization=MagicMock(memory_usage_percent=50.0),
        )
        mock_snapshot.firefox_nodes = MagicMock(
            replicas_current=1,
            replicas_min=1,
            replicas_max=5,
            cpu_utilization=MagicMock(cpu_usage_percent=30.0),
            memory_utilization=MagicMock(memory_usage_percent=40.0),
        )
        mock_snapshot.edge_nodes = MagicMock(
            replicas_current=1,
            replicas_min=1,
            replicas_max=5,
            cpu_utilization=MagicMock(cpu_usage_percent=25.0),
            memory_utilization=MagicMock(memory_usage_percent=35.0),
        )
        mock_snapshot.total_pods = 10
        mock_snapshot.total_nodes = 3
        mock_snapshot.cluster_cpu_utilization = 35.0
        mock_snapshot.cluster_memory_utilization = 45.0

        mock_prometheus.get_infrastructure_snapshot = AsyncMock(return_value=mock_snapshot)
        mock_prometheus.get_usage_patterns = AsyncMock(side_effect=Exception("Error"))
        mock_prometheus.get_test_execution_metrics = AsyncMock(return_value={})

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="[]")]
        mock_anthropic.messages.create.return_value = mock_response

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        with patch.object(optimizer, "_store_recommendation", new_callable=AsyncMock):
            recommendations = await optimizer.analyze_and_recommend("org-123")

        # Should still proceed with default usage patterns
        assert isinstance(recommendations, list)


class TestAIInfraOptimizerPrepareAnalysisContext:
    """Tests for _prepare_analysis_context method."""

    def test_prepare_analysis_context(self, mock_env_vars):
        """Test preparing analysis context string."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        # Create mock snapshot
        mock_snapshot = MagicMock()
        mock_snapshot.selenium = MagicMock(
            sessions_queued=2,
            sessions_active=5,
            nodes_available=3,
            nodes_total=5,
            avg_session_duration_seconds=30.0,
            queue_wait_time_seconds=5.0,
        )
        mock_snapshot.chrome_nodes = MagicMock(
            replicas_current=3,
            replicas_min=2,
            replicas_max=10,
            cpu_utilization=MagicMock(cpu_usage_percent=40.0),
            memory_utilization=MagicMock(memory_usage_percent=50.0),
        )
        mock_snapshot.firefox_nodes = MagicMock(
            replicas_current=1,
            replicas_min=1,
            replicas_max=5,
            cpu_utilization=MagicMock(cpu_usage_percent=30.0),
            memory_utilization=MagicMock(memory_usage_percent=40.0),
        )
        mock_snapshot.edge_nodes = MagicMock(
            replicas_current=1,
            replicas_min=1,
            replicas_max=5,
            cpu_utilization=MagicMock(cpu_usage_percent=25.0),
            memory_utilization=MagicMock(memory_usage_percent=35.0),
        )
        mock_snapshot.total_pods = 10
        mock_snapshot.total_nodes = 3
        mock_snapshot.cluster_cpu_utilization = 35.0
        mock_snapshot.cluster_memory_utilization = 45.0

        usage_patterns = {
            "hourly_averages": [5.0] * 24,
            "daily_averages": [100.0] * 7,
            "peak_hour": 14,
            "min_hour": 4,
            "peak_day": 2,
            "min_day": 0,
        }

        test_metrics = {
            "total_tests": 100,
            "success_rate": 95.0,
            "avg_duration_seconds": 10.0,
            "p95_duration_seconds": 25.0,
        }

        context = optimizer._prepare_analysis_context(
            mock_snapshot, usage_patterns, test_metrics
        )

        assert "Sessions Queued: 2" in context
        assert "Sessions Active: 5" in context
        assert "Chrome Nodes" in context
        assert "Peak Hour: 14" in context
        assert "Total Tests: 100" in context

    def test_prepare_analysis_context_handles_empty_patterns(self, mock_env_vars):
        """Test context preparation with empty usage patterns."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        mock_snapshot = MagicMock()
        mock_snapshot.selenium = MagicMock(
            sessions_queued=0,
            sessions_active=0,
            nodes_available=0,
            nodes_total=0,
            avg_session_duration_seconds=0.0,
            queue_wait_time_seconds=0.0,
        )
        mock_snapshot.chrome_nodes = MagicMock(
            replicas_current=0,
            replicas_min=0,
            replicas_max=0,
            cpu_utilization=MagicMock(cpu_usage_percent=0.0),
            memory_utilization=MagicMock(memory_usage_percent=0.0),
        )
        mock_snapshot.firefox_nodes = mock_snapshot.chrome_nodes
        mock_snapshot.edge_nodes = mock_snapshot.chrome_nodes
        mock_snapshot.total_pods = 0
        mock_snapshot.total_nodes = 0
        mock_snapshot.cluster_cpu_utilization = 0.0
        mock_snapshot.cluster_memory_utilization = 0.0

        # Empty patterns
        usage_patterns = {}
        test_metrics = {}

        # Should not raise
        context = optimizer._prepare_analysis_context(
            mock_snapshot, usage_patterns, test_metrics
        )

        assert "Current Infrastructure State" in context


class TestAIInfraOptimizerGetAIRecommendations:
    """Tests for _get_ai_recommendations method."""

    @pytest.mark.asyncio
    async def test_get_ai_recommendations_parses_json(self, mock_env_vars):
        """Test parsing JSON recommendations from AI."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {
                "type": "scale_down",
                "priority": "medium",
                "title": "Test recommendation",
                "description": "Test description",
                "estimated_savings_monthly": 25.0,
                "confidence": 0.8,
                "action": {"target": "chrome", "operation": "set_min_replicas", "params": {}},
                "reasoning": "Test reasoning",
            }
        ]))]
        mock_anthropic.messages.create.return_value = mock_response

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        recommendations = await optimizer._get_ai_recommendations("org-123", "context")

        assert len(recommendations) == 1
        assert recommendations[0].title == "Test recommendation"

    @pytest.mark.asyncio
    async def test_get_ai_recommendations_handles_markdown_json(self, mock_env_vars):
        """Test parsing JSON wrapped in markdown code blocks."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""```json
[{
    "type": "cost_alert",
    "priority": "high",
    "title": "Cost alert",
    "description": "Costs are high",
    "estimated_savings_monthly": 100.0,
    "confidence": 0.9,
    "action": {"target": "cluster", "operation": "alert", "params": {}},
    "reasoning": "Monthly costs exceed threshold"
}]
```""")]
        mock_anthropic.messages.create.return_value = mock_response

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        recommendations = await optimizer._get_ai_recommendations("org-123", "context")

        assert len(recommendations) == 1
        assert recommendations[0].type.value == "cost_alert"

    @pytest.mark.asyncio
    async def test_get_ai_recommendations_handles_json_error(self, mock_env_vars):
        """Test handling of JSON parse errors."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="invalid json {{{")]
        mock_anthropic.messages.create.return_value = mock_response

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        recommendations = await optimizer._get_ai_recommendations("org-123", "context")

        assert recommendations == []

    @pytest.mark.asyncio
    async def test_get_ai_recommendations_handles_api_error(self, mock_env_vars):
        """Test handling of API errors."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_anthropic.messages.create.side_effect = Exception("API error")

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        recommendations = await optimizer._get_ai_recommendations("org-123", "context")

        assert recommendations == []


class TestAIInfraOptimizerGetCostReport:
    """Tests for get_cost_report method."""

    @pytest.mark.asyncio
    async def test_get_cost_report_success(self, mock_env_vars):
        """Test successful cost report generation."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        # Mock series data
        mock_series = MagicMock()
        mock_series.values = [
            (datetime.now() - timedelta(hours=i), 3)
            for i in range(24)
        ]
        mock_prometheus.query_range = AsyncMock(return_value=[mock_series])

        mock_snapshot = MagicMock()
        mock_snapshot.chrome_nodes = MagicMock(replicas_max=10)
        mock_snapshot.firefox_nodes = MagicMock(replicas_max=5)
        mock_snapshot.edge_nodes = MagicMock(replicas_max=5)
        mock_prometheus.get_infrastructure_snapshot = AsyncMock(return_value=mock_snapshot)

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        # Mock analyze_and_recommend to avoid full analysis
        with patch.object(optimizer, "analyze_and_recommend", return_value=[]):
            report = await optimizer.get_cost_report("org-123", days=7)

        assert report.total_cost >= Decimal("0")
        assert report.comparison_to_browserstack > Decimal("0")

    @pytest.mark.asyncio
    async def test_get_cost_report_handles_empty_series(self, mock_env_vars):
        """Test cost report with empty metrics series."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_prometheus.query_range = AsyncMock(return_value=[])

        mock_snapshot = MagicMock()
        mock_snapshot.chrome_nodes = MagicMock(replicas_max=5)
        mock_snapshot.firefox_nodes = MagicMock(replicas_max=2)
        mock_snapshot.edge_nodes = MagicMock(replicas_max=2)
        mock_prometheus.get_infrastructure_snapshot = AsyncMock(return_value=mock_snapshot)

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        with patch.object(optimizer, "analyze_and_recommend", return_value=[]):
            report = await optimizer.get_cost_report("org-123")

        assert report.total_cost == Decimal("0")
        assert report.daily_costs == []


class TestAIInfraOptimizerPredictDemand:
    """Tests for predict_demand method."""

    @pytest.mark.asyncio
    async def test_predict_demand_success(self, mock_env_vars):
        """Test successful demand prediction."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_prometheus.get_usage_patterns = AsyncMock(return_value={
            "hourly_averages": [5.0, 3.0, 2.0, 1.0] + [8.0] * 8 + [12.0] * 4 + [6.0] * 8,
            "daily_averages": [100.0, 110.0, 120.0, 115.0, 105.0, 50.0, 45.0],
            "peak_hour": 14,
            "min_hour": 4,
            "peak_day": 2,
            "min_day": 6,
        })

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        forecast = await optimizer.predict_demand("org-123", horizon_hours=24)

        assert len(forecast.hourly_predictions) == 24
        assert forecast.confidence == 0.7
        assert "chrome" in forecast.recommended_min_replicas

    @pytest.mark.asyncio
    async def test_predict_demand_handles_error(self, mock_env_vars):
        """Test demand prediction with usage patterns error."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_prometheus.get_usage_patterns = AsyncMock(
            side_effect=Exception("Error")
        )

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        forecast = await optimizer.predict_demand("org-123")

        # Should still return a forecast with defaults
        assert len(forecast.hourly_predictions) == 24
        assert forecast.recommended_min_replicas["chrome"] >= 2


class TestAIInfraOptimizerDetectAnomalies:
    """Tests for detect_anomalies method."""

    @pytest.mark.asyncio
    async def test_detect_anomalies_stuck_queue(self, mock_env_vars):
        """Test detecting stuck queue anomaly."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_snapshot = MagicMock()
        mock_snapshot.selenium = MagicMock(
            sessions_queued=10,
            queue_wait_time_seconds=120.0,
        )
        mock_snapshot.chrome_nodes = MagicMock(
            cpu_utilization=MagicMock(cpu_usage_percent=50.0),
            memory_utilization=MagicMock(memory_usage_percent=50.0),
        )
        mock_snapshot.firefox_nodes = mock_snapshot.chrome_nodes
        mock_snapshot.edge_nodes = mock_snapshot.chrome_nodes
        mock_snapshot.cluster_cpu_utilization = 50.0
        mock_snapshot.total_nodes = 5

        mock_prometheus.get_infrastructure_snapshot = AsyncMock(return_value=mock_snapshot)

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        anomalies = await optimizer.detect_anomalies("org-123")

        assert any(a.type == "stuck_queue" for a in anomalies)

    @pytest.mark.asyncio
    async def test_detect_anomalies_cpu_exhaustion(self, mock_env_vars):
        """Test detecting CPU exhaustion anomaly."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_snapshot = MagicMock()
        mock_snapshot.selenium = MagicMock(
            sessions_queued=0,
            queue_wait_time_seconds=0,
        )
        mock_snapshot.chrome_nodes = MagicMock(
            cpu_utilization=MagicMock(cpu_usage_percent=95.0),
            memory_utilization=MagicMock(memory_usage_percent=50.0),
        )
        mock_snapshot.firefox_nodes = MagicMock(
            cpu_utilization=MagicMock(cpu_usage_percent=50.0),
            memory_utilization=MagicMock(memory_usage_percent=50.0),
        )
        mock_snapshot.edge_nodes = MagicMock(
            cpu_utilization=MagicMock(cpu_usage_percent=50.0),
            memory_utilization=MagicMock(memory_usage_percent=50.0),
        )
        mock_snapshot.cluster_cpu_utilization = 70.0
        mock_snapshot.total_nodes = 5

        mock_prometheus.get_infrastructure_snapshot = AsyncMock(return_value=mock_snapshot)

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        anomalies = await optimizer.detect_anomalies("org-123")

        assert any(a.type == "cpu_exhaustion" for a in anomalies)

    @pytest.mark.asyncio
    async def test_detect_anomalies_memory_exhaustion(self, mock_env_vars):
        """Test detecting memory exhaustion anomaly."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_snapshot = MagicMock()
        mock_snapshot.selenium = MagicMock(
            sessions_queued=0,
            queue_wait_time_seconds=0,
        )
        mock_snapshot.chrome_nodes = MagicMock(
            cpu_utilization=MagicMock(cpu_usage_percent=50.0),
            memory_utilization=MagicMock(memory_usage_percent=95.0),
        )
        mock_snapshot.firefox_nodes = MagicMock(
            cpu_utilization=MagicMock(cpu_usage_percent=50.0),
            memory_utilization=MagicMock(memory_usage_percent=50.0),
        )
        mock_snapshot.edge_nodes = MagicMock(
            cpu_utilization=MagicMock(cpu_usage_percent=50.0),
            memory_utilization=MagicMock(memory_usage_percent=50.0),
        )
        mock_snapshot.cluster_cpu_utilization = 50.0
        mock_snapshot.total_nodes = 5

        mock_prometheus.get_infrastructure_snapshot = AsyncMock(return_value=mock_snapshot)

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        anomalies = await optimizer.detect_anomalies("org-123")

        assert any(a.type == "memory_exhaustion" for a in anomalies)

    @pytest.mark.asyncio
    async def test_detect_anomalies_over_provisioned(self, mock_env_vars):
        """Test detecting over-provisioning anomaly."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_snapshot = MagicMock()
        mock_snapshot.selenium = MagicMock(
            sessions_queued=0,
            queue_wait_time_seconds=0,
        )
        mock_snapshot.chrome_nodes = MagicMock(
            cpu_utilization=MagicMock(cpu_usage_percent=10.0),
            memory_utilization=MagicMock(memory_usage_percent=10.0),
        )
        mock_snapshot.firefox_nodes = mock_snapshot.chrome_nodes
        mock_snapshot.edge_nodes = mock_snapshot.chrome_nodes
        mock_snapshot.cluster_cpu_utilization = 15.0
        mock_snapshot.total_nodes = 5

        mock_prometheus.get_infrastructure_snapshot = AsyncMock(return_value=mock_snapshot)

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        anomalies = await optimizer.detect_anomalies("org-123")

        assert any(a.type == "over_provisioned" for a in anomalies)

    @pytest.mark.asyncio
    async def test_detect_anomalies_handles_error(self, mock_env_vars):
        """Test anomaly detection handles errors gracefully."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        mock_prometheus.get_infrastructure_snapshot = AsyncMock(
            side_effect=Exception("Error")
        )

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        anomalies = await optimizer.detect_anomalies("org-123")

        assert anomalies == []


class TestAIInfraOptimizerApplyRecommendation:
    """Tests for apply_recommendation method."""

    @pytest.mark.asyncio
    async def test_apply_recommendation_success(self, mock_env_vars):
        """Test successful recommendation application."""
        from src.services.infra_optimizer import AIInfraOptimizer, ApprovalStatus

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={
            "data": [{
                "id": "rec-123",
                "status": "pending",
                "action": {"target": "chrome", "operation": "set_min_replicas"},
            }]
        })
        mock_supabase.update = AsyncMock(return_value={})
        optimizer._supabase = mock_supabase

        result = await optimizer.apply_recommendation("rec-123")

        assert result["success"] is True
        assert result["status"] == ApprovalStatus.APPROVED.value

    @pytest.mark.asyncio
    async def test_apply_recommendation_auto_mode(self, mock_env_vars):
        """Test auto-applying recommendation."""
        from src.services.infra_optimizer import AIInfraOptimizer, ApprovalStatus

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={
            "data": [{
                "id": "rec-123",
                "status": "pending",
                "action": {},
            }]
        })
        mock_supabase.update = AsyncMock(return_value={})
        optimizer._supabase = mock_supabase

        result = await optimizer.apply_recommendation("rec-123", auto=True)

        assert result["success"] is True
        assert result["status"] == ApprovalStatus.AUTO_APPLIED.value

    @pytest.mark.asyncio
    async def test_apply_recommendation_not_found(self, mock_env_vars):
        """Test applying nonexistent recommendation."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={"data": []})
        optimizer._supabase = mock_supabase

        result = await optimizer.apply_recommendation("nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_apply_recommendation_already_applied(self, mock_env_vars):
        """Test applying already applied recommendation."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={
            "data": [{
                "id": "rec-123",
                "status": "approved",
                "action": {},
            }]
        })
        optimizer._supabase = mock_supabase

        result = await optimizer.apply_recommendation("rec-123")

        assert result["success"] is False
        assert "already applied" in result["error"].lower()


class TestAIInfraOptimizerGetSavingsSummary:
    """Tests for get_savings_summary method."""

    @pytest.mark.asyncio
    async def test_get_savings_summary_success(self, mock_env_vars):
        """Test successful savings summary generation."""
        from src.services.infra_optimizer import AIInfraOptimizer, CostReport
        from decimal import Decimal

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(return_value={
            "data": [
                {"estimated_savings_monthly": 25.0},
                {"estimated_savings_monthly": 50.0},
            ]
        })
        optimizer._supabase = mock_supabase

        mock_cost_report = CostReport(
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            total_cost=Decimal("300.00"),
            breakdown={},
            daily_costs=[],
            projected_monthly=Decimal("300.00"),
            comparison_to_browserstack=Decimal("990.00"),
            savings_achieved=Decimal("690.00"),
            recommendations=[],
        )

        with patch.object(optimizer, "get_cost_report", return_value=mock_cost_report):
            summary = await optimizer.get_savings_summary("org-123")

        assert summary["total_monthly_savings"] == 75.0
        assert summary["recommendations_applied"] == 2
        assert summary["savings_vs_browserstack"] == 690.0

    @pytest.mark.asyncio
    async def test_get_savings_summary_handles_errors(self, mock_env_vars):
        """Test savings summary handles errors gracefully."""
        from src.services.infra_optimizer import AIInfraOptimizer

        mock_prometheus = MagicMock()
        mock_anthropic = MagicMock()

        optimizer = AIInfraOptimizer(
            prometheus_collector=mock_prometheus,
            anthropic_client=mock_anthropic,
        )

        mock_supabase = MagicMock()
        mock_supabase.request = AsyncMock(side_effect=Exception("DB error"))
        optimizer._supabase = mock_supabase

        with patch.object(optimizer, "get_cost_report", side_effect=Exception("Error")):
            summary = await optimizer.get_savings_summary("org-123")

        # Should return default values
        assert summary["total_monthly_savings"] == 0.0
        assert summary["recommendations_applied"] == 0


class TestCreateInfraOptimizer:
    """Tests for create_infra_optimizer factory function."""

    def test_create_infra_optimizer_default(self, mock_env_vars):
        """Test creating optimizer with defaults."""
        from src.services.infra_optimizer import create_infra_optimizer

        with patch("src.services.infra_optimizer.create_prometheus_collector") as mock_prometheus:
            mock_prometheus.return_value = MagicMock()
            optimizer = create_infra_optimizer()

        assert optimizer is not None

    def test_create_infra_optimizer_custom_params(self, mock_env_vars):
        """Test creating optimizer with custom parameters."""
        from src.services.infra_optimizer import create_infra_optimizer

        with patch("src.services.infra_optimizer.create_prometheus_collector") as mock_prometheus:
            mock_prometheus.return_value = MagicMock()
            optimizer = create_infra_optimizer(
                prometheus_url="http://prometheus:9090",
                model="claude-opus-4-5",
            )

        assert optimizer.model == "claude-opus-4-5"


class TestNodePricingConstants:
    """Tests for pricing constants."""

    def test_node_hourly_cost_exists(self):
        """Test that NODE_HOURLY_COST is defined."""
        from src.services.infra_optimizer import NODE_HOURLY_COST

        assert NODE_HOURLY_COST == Decimal("0.0667")

    def test_browserstack_pricing_exists(self):
        """Test that BrowserStack pricing is defined."""
        from src.services.infra_optimizer import BROWSERSTACK_PER_SESSION_MONTHLY

        assert BROWSERSTACK_PER_SESSION_MONTHLY == Decimal("99.00")
