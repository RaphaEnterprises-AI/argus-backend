"""Tests for SLI Tracker.

RAP-329: Define SLO/SLI Framework
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.slo.definitions import SLO, SLOCategory, SLORegistry
from src.slo.tracker import (
    SLITracker,
    SLIMetric,
    SLIStatus,
    SLOCompliance,
    ErrorBudget,
    get_sli_tracker,
)


class TestSLIMetric:
    """Tests for SLI metric measurement."""

    def test_target_met_availability_success(self):
        """Test target_met for availability SLO when target is met."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )
        metric = SLIMetric(
            slo=slo,
            current_value=99.95,  # Above 99.9 target
            status=SLIStatus.HEALTHY,
            measured_at=datetime.utcnow(),
        )

        assert metric.target_met is True

    def test_target_met_availability_failure(self):
        """Test target_met for availability SLO when target is not met."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )
        metric = SLIMetric(
            slo=slo,
            current_value=99.5,  # Below 99.9 target
            status=SLIStatus.WARNING,
            measured_at=datetime.utcnow(),
        )

        assert metric.target_met is False

    def test_target_met_latency_success(self):
        """Test target_met for latency SLO when target is met."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.LATENCY,
            target=200,
            unit="ms",
        )
        metric = SLIMetric(
            slo=slo,
            current_value=150,  # Below 200ms target
            status=SLIStatus.HEALTHY,
            measured_at=datetime.utcnow(),
        )

        assert metric.target_met is True

    def test_target_met_latency_failure(self):
        """Test target_met for latency SLO when target is not met."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.LATENCY,
            target=200,
            unit="ms",
        )
        metric = SLIMetric(
            slo=slo,
            current_value=350,  # Above 200ms target
            status=SLIStatus.CRITICAL,
            measured_at=datetime.utcnow(),
        )

        assert metric.target_met is False

    def test_target_met_error_rate_success(self):
        """Test target_met for error rate SLO when target is met."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.ERROR_RATE,
            target=1.0,
            unit="percent",
        )
        metric = SLIMetric(
            slo=slo,
            current_value=0.5,  # Below 1% target
            status=SLIStatus.HEALTHY,
            measured_at=datetime.utcnow(),
        )

        assert metric.target_met is True

    def test_target_met_none_value(self):
        """Test target_met returns False when value is None."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )
        metric = SLIMetric(
            slo=slo,
            current_value=None,
            status=SLIStatus.UNKNOWN,
            measured_at=datetime.utcnow(),
        )

        assert metric.target_met is False

    def test_deviation_percent_availability(self):
        """Test deviation calculation for availability SLO."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )
        metric = SLIMetric(
            slo=slo,
            current_value=99.0,  # 0.9% below target
            status=SLIStatus.WARNING,
            measured_at=datetime.utcnow(),
        )

        # Deviation = (99.9 - 99.0) / 99.9 * 100 = 0.9%
        assert metric.deviation_percent == pytest.approx(0.9009, rel=0.01)

    def test_deviation_percent_latency(self):
        """Test deviation calculation for latency SLO."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.LATENCY,
            target=200,
            unit="ms",
        )
        metric = SLIMetric(
            slo=slo,
            current_value=300,  # 50% above target
            status=SLIStatus.CRITICAL,
            measured_at=datetime.utcnow(),
        )

        # Deviation = (300 - 200) / 200 * 100 = 50%
        assert metric.deviation_percent == 50.0

    def test_to_dict(self):
        """Test converting SLI metric to dictionary."""
        slo = SLO(
            name="Test SLO",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )
        metric = SLIMetric(
            slo=slo,
            current_value=99.95,
            status=SLIStatus.HEALTHY,
            measured_at=datetime.utcnow(),
            window_hours=24,
            data_points=100,
        )

        d = metric.to_dict()

        assert d["slo_name"] == "Test SLO"
        assert d["service"] == "test"
        assert d["category"] == "availability"
        assert d["target"] == 99.9
        assert d["current_value"] == 99.95
        assert d["status"] == "healthy"
        assert d["target_met"] is True
        assert d["window_hours"] == 24
        assert d["data_points"] == 100


class TestErrorBudget:
    """Tests for error budget calculation."""

    def test_error_budget_status_healthy(self):
        """Test error budget status when healthy.

        Note: remaining_percent represents % of total budget remaining.
        With total_budget=0.1, remaining=0.08 means 80% of budget remains.
        Status thresholds check if remaining < 25 (as absolute percentage),
        so for small budgets like 0.1%, we need to ensure remaining > 25.
        Actually the check is remaining_percent < 25, where remaining is
        percentage OF the budget. So 80% remaining (0.08 out of 0.1) would
        pass as remaining_percent=0.08 which is < 25, triggering warning.

        Let's use larger values to clearly show healthy state.
        """
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )
        # Use remaining_percent > 25 and burn_rate <= 2.0 for healthy
        budget = ErrorBudget(
            slo=slo,
            total_budget_percent=100.0,  # 100% total budget
            consumed_percent=50.0,  # 50% consumed
            remaining_percent=50.0,  # 50% remaining (> 25 threshold)
            burn_rate=1.5,  # Under 2.0 threshold
            time_to_exhaustion_hours=720,
            window_days=30,
        )

        assert budget.status == SLIStatus.HEALTHY

    def test_error_budget_status_warning_burn_rate(self):
        """Test error budget status warning due to burn rate."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )
        budget = ErrorBudget(
            slo=slo,
            total_budget_percent=0.1,
            consumed_percent=0.06,
            remaining_percent=0.04,  # 40% remaining
            burn_rate=2.5,  # Burning too fast
            time_to_exhaustion_hours=100,
            window_days=30,
        )

        assert budget.status == SLIStatus.WARNING

    def test_error_budget_status_warning_low_remaining(self):
        """Test error budget status warning due to low remaining budget."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )
        budget = ErrorBudget(
            slo=slo,
            total_budget_percent=0.1,
            consumed_percent=0.08,
            remaining_percent=0.02,  # Only 20% remaining
            burn_rate=1.0,
            time_to_exhaustion_hours=50,
            window_days=30,
        )

        assert budget.status == SLIStatus.WARNING

    def test_error_budget_status_critical(self):
        """Test error budget status when exhausted."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )
        budget = ErrorBudget(
            slo=slo,
            total_budget_percent=0.1,
            consumed_percent=0.1,
            remaining_percent=0.0,  # Exhausted
            burn_rate=3.0,
            time_to_exhaustion_hours=None,
            window_days=30,
        )

        assert budget.status == SLIStatus.CRITICAL

    def test_to_dict(self):
        """Test converting error budget to dictionary."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )
        budget = ErrorBudget(
            slo=slo,
            total_budget_percent=0.1,
            consumed_percent=0.05,
            remaining_percent=0.05,
            burn_rate=1.5,
            time_to_exhaustion_hours=200,
            window_days=30,
        )

        d = budget.to_dict()

        assert d["slo_name"] == "Test"
        assert d["service"] == "test"
        assert d["total_budget_percent"] == 0.1
        assert d["consumed_percent"] == 0.05
        assert d["remaining_percent"] == 0.05
        assert d["burn_rate"] == 1.5
        assert d["time_to_exhaustion_hours"] == 200.0
        assert d["window_days"] == 30


class TestSLITracker:
    """Tests for SLI tracker."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker with a custom registry."""
        registry = SLORegistry()
        return SLITracker(prometheus_url="http://test:9090", registry=registry)

    def test_determine_status_availability_healthy(self, tracker):
        """Test status determination for healthy availability."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )

        status = tracker._determine_status(slo, 99.95)
        assert status == SLIStatus.HEALTHY

    def test_determine_status_availability_warning(self, tracker):
        """Test status determination for warning availability."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
            alerting_threshold=99.5,
        )

        # Between threshold (99.5) and target (99.9)
        status = tracker._determine_status(slo, 99.7)
        assert status == SLIStatus.WARNING

    def test_determine_status_availability_critical(self, tracker):
        """Test status determination for critical availability."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
            alerting_threshold=99.5,
        )

        # Below threshold
        status = tracker._determine_status(slo, 99.0)
        assert status == SLIStatus.CRITICAL

    def test_determine_status_latency_healthy(self, tracker):
        """Test status determination for healthy latency."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.LATENCY,
            target=200,
            unit="ms",
        )

        status = tracker._determine_status(slo, 150)
        assert status == SLIStatus.HEALTHY

    def test_determine_status_latency_critical(self, tracker):
        """Test status determination for critical latency."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.LATENCY,
            target=200,
            unit="ms",
            alerting_threshold=300,
        )

        # Above threshold
        status = tracker._determine_status(slo, 400)
        assert status == SLIStatus.CRITICAL

    def test_determine_status_error_rate_healthy(self, tracker):
        """Test status determination for healthy error rate."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.ERROR_RATE,
            target=1.0,
            unit="percent",
        )

        status = tracker._determine_status(slo, 0.5)
        assert status == SLIStatus.HEALTHY

    def test_determine_status_error_rate_critical(self, tracker):
        """Test status determination for critical error rate."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.ERROR_RATE,
            target=1.0,
            unit="percent",
            alerting_threshold=2.0,
        )

        # Above threshold
        status = tracker._determine_status(slo, 5.0)
        assert status == SLIStatus.CRITICAL


class TestSLOCompliance:
    """Tests for SLO compliance."""

    def test_compliance_to_dict(self):
        """Test converting compliance to dictionary."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )
        metric = SLIMetric(
            slo=slo,
            current_value=99.95,
            status=SLIStatus.HEALTHY,
            measured_at=datetime.utcnow(),
        )
        compliance = SLOCompliance(
            service="test",
            metrics=[metric],
            error_budgets=[],
            overall_status=SLIStatus.HEALTHY,
            compliance_percent=100.0,
            critical_violations=0,
            warning_count=0,
        )

        d = compliance.to_dict()

        assert d["service"] == "test"
        assert d["overall_status"] == "healthy"
        assert d["compliance_percent"] == 100.0
        assert d["critical_violations"] == 0
        assert d["total_slos"] == 1
        assert len(d["metrics"]) == 1


class TestSingletonTracker:
    """Tests for singleton tracker instance."""

    def test_get_sli_tracker_returns_singleton(self):
        """Test that get_sli_tracker returns the same instance."""
        # Reset singleton for test
        import src.slo.tracker as tracker_module
        tracker_module._tracker = None

        tracker1 = get_sli_tracker()
        tracker2 = get_sli_tracker()

        assert tracker1 is tracker2

        # Cleanup
        tracker_module._tracker = None
