"""SLI Tracker for measuring SLO compliance.

Provides measurement and tracking of Service Level Indicators (SLIs)
to calculate SLO compliance and error budget consumption.

RAP-329: Define SLO/SLI Framework

Features:
- Real-time SLI measurement via Prometheus queries
- Error budget calculation and burn rate monitoring
- Historical compliance tracking
- Multi-window analysis (hourly, daily, weekly, monthly)

Usage:
    tracker = SLITracker()

    # Get current compliance for a service
    compliance = await tracker.calculate_compliance("api", window_hours=24)

    # Get error budget status
    budget = await tracker.get_error_budget("api", "API Availability")

    # Get all SLO statuses
    status = await tracker.get_all_slo_status()
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from .definitions import SLO, SLOCategory, SLORegistry, get_slo_registry

logger = structlog.get_logger()


class SLIStatus(str, Enum):
    """Status of an SLI measurement."""
    HEALTHY = "healthy"       # Within SLO target
    WARNING = "warning"       # Approaching threshold
    CRITICAL = "critical"     # Breaching SLO
    UNKNOWN = "unknown"       # Unable to measure


@dataclass
class SLIMetric:
    """A single SLI measurement.

    Attributes:
        slo: The SLO this measurement is for
        current_value: Current measured value
        status: Health status based on thresholds
        measured_at: Timestamp of measurement
        window_hours: Time window for the measurement
        data_points: Number of data points in the window
    """
    slo: SLO
    current_value: float | None
    status: SLIStatus
    measured_at: datetime
    window_hours: int = 24
    data_points: int = 0
    error_message: str | None = None

    @property
    def target_met(self) -> bool:
        """Check if current value meets the SLO target."""
        if self.current_value is None:
            return False

        if self.slo.category == SLOCategory.AVAILABILITY:
            return self.current_value >= self.slo.target
        elif self.slo.category == SLOCategory.ERROR_RATE:
            return self.current_value <= self.slo.target
        elif self.slo.category in [SLOCategory.LATENCY, SLOCategory.SATURATION]:
            return self.current_value <= self.slo.target
        return False

    @property
    def deviation_percent(self) -> float | None:
        """Calculate percentage deviation from target."""
        if self.current_value is None:
            return None

        if self.slo.target == 0:
            return None

        if self.slo.category == SLOCategory.AVAILABILITY:
            # For availability, deviation is how much below target
            return ((self.slo.target - self.current_value) / self.slo.target) * 100
        else:
            # For latency/error rate, deviation is how much above target
            return ((self.current_value - self.slo.target) / self.slo.target) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "slo_name": self.slo.name,
            "service": self.slo.service,
            "category": self.slo.category.value,
            "target": self.slo.target,
            "current_value": self.current_value,
            "unit": self.slo.unit,
            "status": self.status.value,
            "target_met": self.target_met,
            "deviation_percent": self.deviation_percent,
            "measured_at": self.measured_at.isoformat(),
            "window_hours": self.window_hours,
            "data_points": self.data_points,
            "error_message": self.error_message,
        }


@dataclass
class ErrorBudget:
    """Error budget status for an SLO.

    Error budget = allowed failure percentage based on SLO target.
    For 99.9% availability SLO, error budget is 0.1% (43.8 minutes/month).

    Attributes:
        slo: The SLO this budget is for
        total_budget_percent: Total error budget (e.g., 0.1 for 99.9% SLO)
        consumed_percent: Percentage of budget consumed
        remaining_percent: Percentage of budget remaining
        burn_rate: Current burn rate (1.0 = on track, >1.0 = burning too fast)
        time_to_exhaustion_hours: Estimated hours until budget exhausted at current rate
        window_days: Window for budget calculation
    """
    slo: SLO
    total_budget_percent: float
    consumed_percent: float
    remaining_percent: float
    burn_rate: float
    time_to_exhaustion_hours: float | None
    window_days: int = 30
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def status(self) -> SLIStatus:
        """Determine status based on burn rate and remaining budget."""
        if self.remaining_percent <= 0:
            return SLIStatus.CRITICAL
        elif self.burn_rate > 2.0 or self.remaining_percent < 25:
            return SLIStatus.WARNING
        return SLIStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "slo_name": self.slo.name,
            "service": self.slo.service,
            "total_budget_percent": self.total_budget_percent,
            "consumed_percent": round(self.consumed_percent, 3),
            "remaining_percent": round(self.remaining_percent, 3),
            "burn_rate": round(self.burn_rate, 2),
            "time_to_exhaustion_hours": (
                round(self.time_to_exhaustion_hours, 1)
                if self.time_to_exhaustion_hours else None
            ),
            "status": self.status.value,
            "window_days": self.window_days,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class SLOCompliance:
    """Overall SLO compliance status for a service.

    Aggregates all SLI metrics for a service into a single compliance view.
    """
    service: str
    metrics: list[SLIMetric]
    error_budgets: list[ErrorBudget]
    overall_status: SLIStatus
    compliance_percent: float
    critical_violations: int
    warning_count: int
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "service": self.service,
            "overall_status": self.overall_status.value,
            "compliance_percent": round(self.compliance_percent, 2),
            "critical_violations": self.critical_violations,
            "warning_count": self.warning_count,
            "total_slos": len(self.metrics),
            "metrics": [m.to_dict() for m in self.metrics],
            "error_budgets": [b.to_dict() for b in self.error_budgets],
            "calculated_at": self.calculated_at.isoformat(),
        }


class SLITracker:
    """Tracks SLIs and calculates SLO compliance.

    Integrates with Prometheus to query metrics and calculate SLI values.

    Usage:
        tracker = SLITracker()
        compliance = await tracker.calculate_compliance("api")
    """

    def __init__(
        self,
        prometheus_url: str | None = None,
        registry: SLORegistry | None = None,
    ):
        """Initialize the SLI tracker.

        Args:
            prometheus_url: URL of Prometheus server. Defaults to env var or K8s internal.
            registry: SLO registry to use. Defaults to global registry.
        """
        self.prometheus_url = prometheus_url or os.getenv(
            "PROMETHEUS_URL",
            "http://prometheus-server.monitoring.svc.cluster.local:9090"
        )
        self.registry = registry or get_slo_registry()
        self.log = logger.bind(component="SLITracker")

        # Lazy-loaded Prometheus collector
        self._prometheus: Any = None

    async def _get_prometheus(self):
        """Get or create Prometheus collector."""
        if self._prometheus is None:
            try:
                from ..services.prometheus_collector import PrometheusCollector
                self._prometheus = PrometheusCollector(
                    prometheus_url=self.prometheus_url
                )
            except ImportError:
                self.log.warning("prometheus_collector_not_available")
                return None
        return self._prometheus

    async def measure_sli(
        self,
        slo: SLO,
        window_hours: int = 24,
    ) -> SLIMetric:
        """Measure the current SLI value for an SLO.

        Args:
            slo: The SLO to measure
            window_hours: Time window for measurement

        Returns:
            SLIMetric with current value and status
        """
        prometheus = await self._get_prometheus()

        if prometheus is None or not slo.prometheus_query:
            return SLIMetric(
                slo=slo,
                current_value=None,
                status=SLIStatus.UNKNOWN,
                measured_at=datetime.utcnow(),
                window_hours=window_hours,
                error_message="Prometheus not available or no query defined",
            )

        try:
            # Query Prometheus
            results = await prometheus.query(slo.prometheus_query.strip())

            if not results:
                return SLIMetric(
                    slo=slo,
                    current_value=None,
                    status=SLIStatus.UNKNOWN,
                    measured_at=datetime.utcnow(),
                    window_hours=window_hours,
                    error_message="No data returned from Prometheus",
                )

            # Get the value (first result)
            current_value = results[0].value

            # Determine status
            status = self._determine_status(slo, current_value)

            return SLIMetric(
                slo=slo,
                current_value=current_value,
                status=status,
                measured_at=datetime.utcnow(),
                window_hours=window_hours,
                data_points=len(results),
            )

        except Exception as e:
            self.log.error(
                "sli_measurement_failed",
                slo_name=slo.name,
                error=str(e),
            )
            return SLIMetric(
                slo=slo,
                current_value=None,
                status=SLIStatus.UNKNOWN,
                measured_at=datetime.utcnow(),
                window_hours=window_hours,
                error_message=str(e),
            )

    def _determine_status(self, slo: SLO, value: float) -> SLIStatus:
        """Determine SLI status based on value and thresholds."""
        if slo.category == SLOCategory.AVAILABILITY:
            # Higher is better
            if value >= slo.target:
                return SLIStatus.HEALTHY
            elif value >= slo.alerting_threshold:
                return SLIStatus.WARNING
            else:
                return SLIStatus.CRITICAL

        elif slo.category == SLOCategory.ERROR_RATE:
            # Lower is better
            if value <= slo.target:
                return SLIStatus.HEALTHY
            elif value <= slo.alerting_threshold:
                return SLIStatus.WARNING
            else:
                return SLIStatus.CRITICAL

        else:  # LATENCY, SATURATION, THROUGHPUT
            # Lower is better for latency/saturation
            if value <= slo.target:
                return SLIStatus.HEALTHY
            elif value <= slo.alerting_threshold:
                return SLIStatus.WARNING
            else:
                return SLIStatus.CRITICAL

    async def calculate_error_budget(
        self,
        slo: SLO,
        window_days: int = 30,
    ) -> ErrorBudget:
        """Calculate error budget status for an SLO.

        Args:
            slo: The SLO to calculate budget for
            window_days: Rolling window in days

        Returns:
            ErrorBudget with consumption and burn rate
        """
        # Get current SLI value
        metric = await self.measure_sli(slo, window_hours=window_days * 24)

        # Calculate total error budget
        total_budget = slo.error_budget_percent

        # Calculate consumed budget
        if metric.current_value is None:
            consumed = 0.0
        elif slo.category == SLOCategory.AVAILABILITY:
            # Error budget = 100 - target = 0.1 for 99.9%
            # Consumed = 100 - actual (if actual < target)
            actual_error = 100 - metric.current_value
            consumed = min(actual_error, total_budget) if actual_error > 0 else 0.0
        elif slo.category == SLOCategory.ERROR_RATE:
            # Error budget = target error rate
            # Consumed = actual error rate (capped at budget)
            consumed = min(metric.current_value, total_budget)
        else:
            # For latency, calculate based on percentage over target
            if metric.current_value > slo.target:
                over_target_pct = ((metric.current_value - slo.target) / slo.target) * 100
                consumed = min(over_target_pct, 100)
            else:
                consumed = 0.0

        # Calculate remaining budget
        remaining = max(0, total_budget - consumed)

        # Calculate burn rate
        # Burn rate = 1.0 means consuming budget exactly as expected
        # Burn rate = 2.0 means consuming 2x as fast
        expected_consumed = (total_budget / window_days) * (datetime.utcnow().day % window_days or 1)
        burn_rate = consumed / expected_consumed if expected_consumed > 0 else 1.0

        # Calculate time to exhaustion
        if burn_rate > 0 and remaining > 0:
            hours_in_window = window_days * 24
            consumed_per_hour = consumed / hours_in_window if hours_in_window > 0 else 0
            time_to_exhaustion = remaining / consumed_per_hour if consumed_per_hour > 0 else None
        else:
            time_to_exhaustion = None

        return ErrorBudget(
            slo=slo,
            total_budget_percent=total_budget,
            consumed_percent=consumed,
            remaining_percent=remaining,
            burn_rate=burn_rate,
            time_to_exhaustion_hours=time_to_exhaustion,
            window_days=window_days,
            calculated_at=datetime.utcnow(),
        )

    async def calculate_compliance(
        self,
        service: str,
        window_hours: int = 24,
    ) -> SLOCompliance:
        """Calculate overall SLO compliance for a service.

        Args:
            service: Service name (e.g., "api", "browser-pool")
            window_hours: Time window for measurement

        Returns:
            SLOCompliance with all metrics and overall status
        """
        slos = self.registry.get_by_service(service)

        if not slos:
            return SLOCompliance(
                service=service,
                metrics=[],
                error_budgets=[],
                overall_status=SLIStatus.UNKNOWN,
                compliance_percent=0.0,
                critical_violations=0,
                warning_count=0,
            )

        # Measure all SLIs
        metrics = []
        error_budgets = []

        for slo in slos:
            metric = await self.measure_sli(slo, window_hours)
            metrics.append(metric)

            # Calculate error budget for availability SLOs
            if slo.category == SLOCategory.AVAILABILITY:
                budget = await self.calculate_error_budget(slo)
                error_budgets.append(budget)

        # Calculate overall status
        critical_violations = sum(1 for m in metrics if m.status == SLIStatus.CRITICAL)
        warning_count = sum(1 for m in metrics if m.status == SLIStatus.WARNING)
        healthy_count = sum(1 for m in metrics if m.status == SLIStatus.HEALTHY)

        if critical_violations > 0:
            overall_status = SLIStatus.CRITICAL
        elif warning_count > 0:
            overall_status = SLIStatus.WARNING
        elif healthy_count > 0:
            overall_status = SLIStatus.HEALTHY
        else:
            overall_status = SLIStatus.UNKNOWN

        # Calculate compliance percentage
        total_with_data = sum(1 for m in metrics if m.current_value is not None)
        targets_met = sum(1 for m in metrics if m.target_met)
        compliance_percent = (targets_met / total_with_data * 100) if total_with_data > 0 else 0.0

        return SLOCompliance(
            service=service,
            metrics=metrics,
            error_budgets=error_budgets,
            overall_status=overall_status,
            compliance_percent=compliance_percent,
            critical_violations=critical_violations,
            warning_count=warning_count,
        )

    async def get_all_slo_status(
        self,
        window_hours: int = 24,
    ) -> dict[str, SLOCompliance]:
        """Get SLO compliance status for all services.

        Args:
            window_hours: Time window for measurement

        Returns:
            Dictionary mapping service name to compliance status
        """
        services = self.registry.get_services()
        results = {}

        for service in services:
            compliance = await self.calculate_compliance(service, window_hours)
            results[service] = compliance

        return results

    async def get_critical_slo_status(
        self,
        window_hours: int = 24,
    ) -> list[SLIMetric]:
        """Get status of all critical SLOs.

        Args:
            window_hours: Time window for measurement

        Returns:
            List of SLI metrics for critical SLOs
        """
        critical_slos = self.registry.get_critical()
        metrics = []

        for slo in critical_slos:
            metric = await self.measure_sli(slo, window_hours)
            metrics.append(metric)

        return metrics

    async def close(self):
        """Close any open connections."""
        if self._prometheus is not None:
            await self._prometheus.close()


# Singleton instance
_tracker: SLITracker | None = None


def get_sli_tracker() -> SLITracker:
    """Get the global SLI tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = SLITracker()
    return _tracker
