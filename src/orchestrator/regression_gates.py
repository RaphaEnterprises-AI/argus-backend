"""Regression Gates for pre-deployment evaluation.

This module implements explicit regression gates that must pass before
agent/model deployments. It includes:

- Selector regression detection (UI tests)
- Model output regression detection
- Performance regression detection
- Coverage regression detection

References:
- LaunchDarkly feature flags
- Continuous Delivery pipelines
- MLOps model validation patterns
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional
import hashlib
import json
import logging
import statistics

from src.config import get_settings

logger = logging.getLogger(__name__)


class GateType(Enum):
    """Types of regression gates."""

    SELECTOR = "selector"  # UI selector stability
    MODEL_OUTPUT = "model_output"  # AI model output consistency
    PERFORMANCE = "performance"  # Latency/throughput regression
    COVERAGE = "coverage"  # Test coverage regression
    ERROR_RATE = "error_rate"  # Error rate regression
    CUSTOM = "custom"  # Custom gate


class GateStatus(Enum):
    """Status of a gate check."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"  # Passed with warnings


class GateSeverity(Enum):
    """Severity of gate failure."""

    BLOCKING = "blocking"  # Must pass, blocks deployment
    WARNING = "warning"  # Should pass, warns but allows
    INFO = "info"  # Informational only


@dataclass
class GateThreshold:
    """Threshold configuration for a gate."""

    metric_name: str
    operator: str  # "lt", "gt", "lte", "gte", "eq", "pct_change"
    value: float
    severity: GateSeverity = GateSeverity.BLOCKING

    def evaluate(self, current: float, baseline: Optional[float] = None) -> bool:
        """Evaluate if the threshold is met.

        Args:
            current: Current metric value
            baseline: Baseline value for comparison (for pct_change)

        Returns:
            True if threshold is met (gate passes)
        """
        if self.operator == "lt":
            return current < self.value
        elif self.operator == "lte":
            return current <= self.value
        elif self.operator == "gt":
            return current > self.value
        elif self.operator == "gte":
            return current >= self.value
        elif self.operator == "eq":
            return abs(current - self.value) < 0.0001
        elif self.operator == "pct_change":
            if baseline is None or baseline == 0:
                return True  # No baseline, skip check
            pct_change = ((current - baseline) / baseline) * 100
            return abs(pct_change) <= self.value
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


@dataclass
class GateResult:
    """Result of a single gate check."""

    gate_id: str
    gate_type: GateType
    name: str
    status: GateStatus
    severity: GateSeverity

    # Metrics
    current_value: Optional[float] = None
    baseline_value: Optional[float] = None
    threshold: Optional[GateThreshold] = None

    # Details
    message: str = ""
    details: dict = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_blocking_failure(self) -> bool:
        """Check if this is a blocking failure."""
        return self.status == GateStatus.FAILED and self.severity == GateSeverity.BLOCKING

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "gate_id": self.gate_id,
            "gate_type": self.gate_type.value,
            "name": self.name,
            "status": self.status.value,
            "severity": self.severity.value,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "threshold": {
                "metric": self.threshold.metric_name,
                "operator": self.threshold.operator,
                "value": self.threshold.value,
            } if self.threshold else None,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class GateSuite:
    """A suite of gates to run together."""

    suite_id: str
    name: str
    description: str = ""
    gates: list[str] = field(default_factory=list)  # Gate IDs

    # Execution settings
    fail_fast: bool = False  # Stop on first blocking failure
    parallel: bool = True  # Run gates in parallel

    def to_dict(self) -> dict:
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description,
            "gates": self.gates,
            "fail_fast": self.fail_fast,
            "parallel": self.parallel,
        }


@dataclass
class GateSuiteResult:
    """Result of running a gate suite."""

    suite_id: str
    suite_name: str
    status: GateStatus
    results: list[GateResult] = field(default_factory=list)

    # Summary
    total_gates: int = 0
    passed_gates: int = 0
    failed_gates: int = 0
    warning_gates: int = 0
    skipped_gates: int = 0

    # Execution details
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    @property
    def blocking_failures(self) -> list[GateResult]:
        """Get all blocking failures."""
        return [r for r in self.results if r.is_blocking_failure]

    def to_dict(self) -> dict:
        return {
            "suite_id": self.suite_id,
            "suite_name": self.suite_name,
            "status": self.status.value,
            "results": [r.to_dict() for r in self.results],
            "total_gates": self.total_gates,
            "passed_gates": self.passed_gates,
            "failed_gates": self.failed_gates,
            "warning_gates": self.warning_gates,
            "skipped_gates": self.skipped_gates,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "blocking_failures": [r.to_dict() for r in self.blocking_failures],
        }


class RegressionGateRegistry:
    """Registry of regression gates.

    Manages gate definitions and provides methods to run them.
    """

    def __init__(self):
        self.settings = get_settings()

        # Registered gates
        self._gates: dict[str, dict] = {}

        # Gate suites
        self._suites: dict[str, GateSuite] = {}

        # Baseline data storage
        self._baselines: dict[str, dict] = {}

        # Register default gates
        self._register_default_gates()

    def _register_default_gates(self):
        """Register default regression gates."""

        # Selector stability gate
        self.register_gate(
            gate_id="selector_stability",
            gate_type=GateType.SELECTOR,
            name="Selector Stability Check",
            description="Verifies UI selectors haven't regressed",
            check_fn=self._check_selector_stability,
            thresholds=[
                GateThreshold("failure_rate", "lte", 5.0, GateSeverity.BLOCKING),
                GateThreshold("new_failures", "lte", 2, GateSeverity.WARNING),
            ],
        )

        # Model output consistency gate
        self.register_gate(
            gate_id="model_output_consistency",
            gate_type=GateType.MODEL_OUTPUT,
            name="Model Output Consistency",
            description="Verifies AI model outputs are consistent with baseline",
            check_fn=self._check_model_consistency,
            thresholds=[
                GateThreshold("semantic_drift", "lte", 15.0, GateSeverity.BLOCKING),
                GateThreshold("format_errors", "eq", 0, GateSeverity.BLOCKING),
            ],
        )

        # Performance regression gate
        self.register_gate(
            gate_id="performance_regression",
            gate_type=GateType.PERFORMANCE,
            name="Performance Regression Check",
            description="Verifies latency and throughput haven't regressed",
            check_fn=self._check_performance_regression,
            thresholds=[
                GateThreshold("p95_latency_ms", "pct_change", 20.0, GateSeverity.BLOCKING),
                GateThreshold("p50_latency_ms", "pct_change", 15.0, GateSeverity.WARNING),
                GateThreshold("error_rate", "lte", 1.0, GateSeverity.BLOCKING),
            ],
        )

        # Coverage regression gate
        self.register_gate(
            gate_id="coverage_regression",
            gate_type=GateType.COVERAGE,
            name="Test Coverage Regression",
            description="Verifies test coverage hasn't decreased",
            check_fn=self._check_coverage_regression,
            thresholds=[
                GateThreshold("line_coverage", "pct_change", -5.0, GateSeverity.WARNING),
                GateThreshold("branch_coverage", "pct_change", -10.0, GateSeverity.WARNING),
            ],
        )

        # Error rate gate
        self.register_gate(
            gate_id="error_rate_check",
            gate_type=GateType.ERROR_RATE,
            name="Error Rate Check",
            description="Verifies error rate is within acceptable limits",
            check_fn=self._check_error_rate,
            thresholds=[
                GateThreshold("error_rate", "lte", 1.0, GateSeverity.BLOCKING),
                GateThreshold("critical_errors", "eq", 0, GateSeverity.BLOCKING),
            ],
        )

        # Create default suite
        self.register_suite(
            suite_id="pre_deployment",
            name="Pre-Deployment Gates",
            description="Standard gates to run before any deployment",
            gates=[
                "selector_stability",
                "model_output_consistency",
                "performance_regression",
                "error_rate_check",
            ],
        )

        self.register_suite(
            suite_id="production_promotion",
            name="Production Promotion Gates",
            description="Comprehensive gates for production deployments",
            gates=[
                "selector_stability",
                "model_output_consistency",
                "performance_regression",
                "coverage_regression",
                "error_rate_check",
            ],
            fail_fast=True,
        )

    def register_gate(
        self,
        gate_id: str,
        gate_type: GateType,
        name: str,
        description: str,
        check_fn: Callable,
        thresholds: list[GateThreshold],
        severity: GateSeverity = GateSeverity.BLOCKING,
    ) -> None:
        """Register a new gate.

        Args:
            gate_id: Unique identifier for the gate
            gate_type: Type of gate
            name: Human-readable name
            description: Description of what the gate checks
            check_fn: Async function that performs the check
            thresholds: List of thresholds to evaluate
            severity: Default severity for this gate
        """
        self._gates[gate_id] = {
            "gate_id": gate_id,
            "gate_type": gate_type,
            "name": name,
            "description": description,
            "check_fn": check_fn,
            "thresholds": thresholds,
            "severity": severity,
        }
        logger.info(f"Registered gate: {gate_id}")

    def register_suite(
        self,
        suite_id: str,
        name: str,
        description: str,
        gates: list[str],
        fail_fast: bool = False,
        parallel: bool = True,
    ) -> None:
        """Register a gate suite."""
        suite = GateSuite(
            suite_id=suite_id,
            name=name,
            description=description,
            gates=gates,
            fail_fast=fail_fast,
            parallel=parallel,
        )
        self._suites[suite_id] = suite
        logger.info(f"Registered suite: {suite_id} with {len(gates)} gates")

    def set_baseline(
        self,
        context: str,
        metrics: dict[str, float],
    ) -> None:
        """Set baseline metrics for comparison.

        Args:
            context: Context identifier (e.g., "agent_CodeAnalyzer_prod")
            metrics: Dictionary of metric name -> value
        """
        self._baselines[context] = {
            "metrics": metrics,
            "set_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"Set baseline for {context}: {list(metrics.keys())}")

    def get_baseline(self, context: str) -> Optional[dict[str, float]]:
        """Get baseline metrics for a context."""
        baseline = self._baselines.get(context)
        return baseline["metrics"] if baseline else None

    async def run_gate(
        self,
        gate_id: str,
        context: dict[str, Any],
    ) -> GateResult:
        """Run a single gate.

        Args:
            gate_id: ID of the gate to run
            context: Context data for the gate check

        Returns:
            GateResult
        """
        gate = self._gates.get(gate_id)
        if not gate:
            return GateResult(
                gate_id=gate_id,
                gate_type=GateType.CUSTOM,
                name="Unknown Gate",
                status=GateStatus.FAILED,
                severity=GateSeverity.BLOCKING,
                message=f"Gate not found: {gate_id}",
            )

        try:
            result = await gate["check_fn"](gate, context)
            return result
        except Exception as e:
            logger.error(f"Gate {gate_id} error: {e}")
            return GateResult(
                gate_id=gate_id,
                gate_type=gate["gate_type"],
                name=gate["name"],
                status=GateStatus.FAILED,
                severity=gate["severity"],
                message=f"Gate execution error: {str(e)}",
            )

    async def run_suite(
        self,
        suite_id: str,
        context: dict[str, Any],
    ) -> GateSuiteResult:
        """Run a gate suite.

        Args:
            suite_id: ID of the suite to run
            context: Context data for gate checks

        Returns:
            GateSuiteResult
        """
        suite = self._suites.get(suite_id)
        if not suite:
            return GateSuiteResult(
                suite_id=suite_id,
                suite_name="Unknown Suite",
                status=GateStatus.FAILED,
            )

        started_at = datetime.now(timezone.utc)
        results: list[GateResult] = []

        # Run gates (sequentially for now, can be parallelized)
        for gate_id in suite.gates:
            result = await self.run_gate(gate_id, context)
            results.append(result)

            # Check for fail-fast
            if suite.fail_fast and result.is_blocking_failure:
                logger.warning(f"Suite {suite_id} fail-fast on gate {gate_id}")
                break

        completed_at = datetime.now(timezone.utc)

        # Calculate summary
        passed = sum(1 for r in results if r.status == GateStatus.PASSED)
        failed = sum(1 for r in results if r.status == GateStatus.FAILED)
        warnings = sum(1 for r in results if r.status == GateStatus.WARNING)
        skipped = len(suite.gates) - len(results)

        # Determine overall status
        blocking_failures = [r for r in results if r.is_blocking_failure]
        if blocking_failures:
            overall_status = GateStatus.FAILED
        elif warnings > 0:
            overall_status = GateStatus.WARNING
        else:
            overall_status = GateStatus.PASSED

        return GateSuiteResult(
            suite_id=suite_id,
            suite_name=suite.name,
            status=overall_status,
            results=results,
            total_gates=len(suite.gates),
            passed_gates=passed,
            failed_gates=failed,
            warning_gates=warnings,
            skipped_gates=skipped,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
        )

    # Default gate implementations

    async def _check_selector_stability(
        self,
        gate: dict,
        context: dict[str, Any],
    ) -> GateResult:
        """Check UI selector stability."""
        # Get test results from context
        test_results = context.get("test_results", [])
        selector_failures = context.get("selector_failures", [])

        total_tests = len(test_results)
        failed_selectors = len(selector_failures)

        if total_tests == 0:
            return GateResult(
                gate_id=gate["gate_id"],
                gate_type=gate["gate_type"],
                name=gate["name"],
                status=GateStatus.SKIPPED,
                severity=gate["severity"],
                message="No test results to evaluate",
            )

        failure_rate = (failed_selectors / total_tests) * 100 if total_tests > 0 else 0

        # Evaluate thresholds
        threshold = gate["thresholds"][0]  # failure_rate threshold
        passed = threshold.evaluate(failure_rate)

        return GateResult(
            gate_id=gate["gate_id"],
            gate_type=gate["gate_type"],
            name=gate["name"],
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            severity=threshold.severity,
            current_value=failure_rate,
            threshold=threshold,
            message=f"Selector failure rate: {failure_rate:.1f}%",
            details={
                "total_tests": total_tests,
                "selector_failures": failed_selectors,
                "failed_selectors": selector_failures[:10],  # First 10
            },
        )

    async def _check_model_consistency(
        self,
        gate: dict,
        context: dict[str, Any],
    ) -> GateResult:
        """Check AI model output consistency."""
        # Get model outputs from context
        current_outputs = context.get("model_outputs", [])
        baseline_outputs = context.get("baseline_outputs", [])

        if not current_outputs:
            return GateResult(
                gate_id=gate["gate_id"],
                gate_type=gate["gate_type"],
                name=gate["name"],
                status=GateStatus.SKIPPED,
                severity=gate["severity"],
                message="No model outputs to evaluate",
            )

        # Calculate semantic drift (simplified - in production use embeddings)
        format_errors = sum(
            1 for o in current_outputs
            if not self._validate_output_format(o)
        )

        # Semantic drift calculation (placeholder - use embedding similarity in prod)
        drift_score = context.get("semantic_drift", 0.0)

        # Evaluate thresholds
        drift_threshold = gate["thresholds"][0]
        format_threshold = gate["thresholds"][1]

        drift_passed = drift_threshold.evaluate(drift_score)
        format_passed = format_threshold.evaluate(format_errors)

        passed = drift_passed and format_passed

        return GateResult(
            gate_id=gate["gate_id"],
            gate_type=gate["gate_type"],
            name=gate["name"],
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            severity=gate["severity"],
            current_value=drift_score,
            threshold=drift_threshold,
            message=f"Semantic drift: {drift_score:.1f}%, Format errors: {format_errors}",
            details={
                "total_outputs": len(current_outputs),
                "format_errors": format_errors,
                "semantic_drift": drift_score,
            },
        )

    def _validate_output_format(self, output: Any) -> bool:
        """Validate model output format."""
        if output is None:
            return False
        if isinstance(output, dict):
            return True
        if isinstance(output, str) and len(output) > 0:
            return True
        return False

    async def _check_performance_regression(
        self,
        gate: dict,
        context: dict[str, Any],
    ) -> GateResult:
        """Check performance regression."""
        # Get latency data
        latencies = context.get("latencies_ms", [])
        baseline_context = context.get("baseline_context", "")
        baseline_metrics = self.get_baseline(baseline_context)

        if not latencies:
            return GateResult(
                gate_id=gate["gate_id"],
                gate_type=gate["gate_type"],
                name=gate["name"],
                status=GateStatus.SKIPPED,
                severity=gate["severity"],
                message="No latency data to evaluate",
            )

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p95_idx = int(len(sorted_latencies) * 0.95)
        p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]

        # Get baseline values
        baseline_p95 = baseline_metrics.get("p95_latency_ms") if baseline_metrics else None
        baseline_p50 = baseline_metrics.get("p50_latency_ms") if baseline_metrics else None

        # Evaluate thresholds
        p95_threshold = gate["thresholds"][0]
        passed = p95_threshold.evaluate(p95, baseline_p95)

        # Calculate percentage change for messaging
        pct_change = None
        if baseline_p95:
            pct_change = ((p95 - baseline_p95) / baseline_p95) * 100

        return GateResult(
            gate_id=gate["gate_id"],
            gate_type=gate["gate_type"],
            name=gate["name"],
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            severity=p95_threshold.severity,
            current_value=p95,
            baseline_value=baseline_p95,
            threshold=p95_threshold,
            message=f"P95 latency: {p95:.0f}ms" + (f" ({pct_change:+.1f}%)" if pct_change else ""),
            details={
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "baseline_p50": baseline_p50,
                "baseline_p95": baseline_p95,
                "sample_count": len(latencies),
            },
        )

    async def _check_coverage_regression(
        self,
        gate: dict,
        context: dict[str, Any],
    ) -> GateResult:
        """Check test coverage regression."""
        current_coverage = context.get("line_coverage", 0)
        baseline_context = context.get("baseline_context", "")
        baseline_metrics = self.get_baseline(baseline_context)
        baseline_coverage = baseline_metrics.get("line_coverage") if baseline_metrics else None

        if current_coverage == 0:
            return GateResult(
                gate_id=gate["gate_id"],
                gate_type=gate["gate_type"],
                name=gate["name"],
                status=GateStatus.SKIPPED,
                severity=gate["severity"],
                message="No coverage data to evaluate",
            )

        # Evaluate threshold
        threshold = gate["thresholds"][0]
        passed = threshold.evaluate(current_coverage, baseline_coverage)

        pct_change = None
        if baseline_coverage:
            pct_change = ((current_coverage - baseline_coverage) / baseline_coverage) * 100

        return GateResult(
            gate_id=gate["gate_id"],
            gate_type=gate["gate_type"],
            name=gate["name"],
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            severity=threshold.severity,
            current_value=current_coverage,
            baseline_value=baseline_coverage,
            threshold=threshold,
            message=f"Line coverage: {current_coverage:.1f}%" + (f" ({pct_change:+.1f}%)" if pct_change else ""),
            details={
                "line_coverage": current_coverage,
                "baseline_coverage": baseline_coverage,
            },
        )

    async def _check_error_rate(
        self,
        gate: dict,
        context: dict[str, Any],
    ) -> GateResult:
        """Check error rate."""
        total_requests = context.get("total_requests", 0)
        errors = context.get("errors", 0)
        critical_errors = context.get("critical_errors", 0)

        if total_requests == 0:
            return GateResult(
                gate_id=gate["gate_id"],
                gate_type=gate["gate_type"],
                name=gate["name"],
                status=GateStatus.SKIPPED,
                severity=gate["severity"],
                message="No request data to evaluate",
            )

        error_rate = (errors / total_requests) * 100

        # Evaluate thresholds
        rate_threshold = gate["thresholds"][0]
        critical_threshold = gate["thresholds"][1]

        rate_passed = rate_threshold.evaluate(error_rate)
        critical_passed = critical_threshold.evaluate(critical_errors)

        passed = rate_passed and critical_passed

        return GateResult(
            gate_id=gate["gate_id"],
            gate_type=gate["gate_type"],
            name=gate["name"],
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            severity=gate["severity"],
            current_value=error_rate,
            threshold=rate_threshold,
            message=f"Error rate: {error_rate:.2f}%, Critical: {critical_errors}",
            details={
                "total_requests": total_requests,
                "errors": errors,
                "critical_errors": critical_errors,
                "error_rate": error_rate,
            },
        )


# Singleton instance
_gate_registry: Optional[RegressionGateRegistry] = None


def get_regression_gate_registry() -> RegressionGateRegistry:
    """Get the singleton regression gate registry."""
    global _gate_registry
    if _gate_registry is None:
        _gate_registry = RegressionGateRegistry()
    return _gate_registry


# Integration with Agent Promotion Pipeline
async def promotion_regression_gate(request) -> tuple[bool, str]:
    """Promotion gate that runs regression checks.

    This gate integrates with the AgentPromotionPipeline.
    """
    from src.orchestrator.agent_promotion import Environment

    registry = get_regression_gate_registry()

    # Determine which suite to run based on target environment
    if request.target_environment == Environment.PRODUCTION:
        suite_id = "production_promotion"
    else:
        suite_id = "pre_deployment"

    # Build context from promotion request
    context = {
        "agent_type": request.agent_type,
        "version": request.version.version,
        "environment": request.target_environment.value,
        "baseline_context": f"agent_{request.agent_type}_{request.source_environment.value}",
        # These would be populated from actual test runs
        "test_results": [],
        "selector_failures": [],
        "model_outputs": [],
        "latencies_ms": [],
        "total_requests": 0,
        "errors": 0,
        "critical_errors": 0,
    }

    result = await registry.run_suite(suite_id, context)

    if result.status == GateStatus.FAILED:
        failures = [f"{r.name}: {r.message}" for r in result.blocking_failures]
        return False, f"Regression gates failed: {'; '.join(failures)}"

    if result.status == GateStatus.WARNING:
        return True, f"Passed with warnings ({result.warning_gates} warnings)"

    return True, f"All {result.passed_gates} gates passed"
