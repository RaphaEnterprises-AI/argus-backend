"""Tests for Regression Gates module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestGateThreshold:
    """Tests for GateThreshold dataclass."""

    def test_threshold_less_than(self, mock_env_vars):
        """Test less than threshold."""
        from src.orchestrator.regression_gates import GateThreshold, GateSeverity

        threshold = GateThreshold(
            metric_name="error_rate",
            operator="lt",
            value=5.0,
            severity=GateSeverity.BLOCKING,
        )

        assert threshold.evaluate(3.0) is True
        assert threshold.evaluate(5.0) is False
        assert threshold.evaluate(7.0) is False

    def test_threshold_less_than_or_equal(self, mock_env_vars):
        """Test less than or equal threshold."""
        from src.orchestrator.regression_gates import GateThreshold, GateSeverity

        threshold = GateThreshold(
            metric_name="latency",
            operator="lte",
            value=100.0,
        )

        assert threshold.evaluate(99.0) is True
        assert threshold.evaluate(100.0) is True
        assert threshold.evaluate(101.0) is False

    def test_threshold_percentage_change(self, mock_env_vars):
        """Test percentage change threshold."""
        from src.orchestrator.regression_gates import GateThreshold

        threshold = GateThreshold(
            metric_name="latency",
            operator="pct_change",
            value=20.0,  # Allow up to 20% change
        )

        # 10% increase - should pass
        assert threshold.evaluate(110.0, baseline=100.0) is True

        # 25% increase - should fail
        assert threshold.evaluate(125.0, baseline=100.0) is False

        # 15% decrease - should pass
        assert threshold.evaluate(85.0, baseline=100.0) is True


class TestGateTypeEnum:
    """Tests for GateType enum."""

    def test_gate_type_values(self, mock_env_vars):
        """Test GateType enum values."""
        from src.orchestrator.regression_gates import GateType

        assert GateType.SELECTOR.value == "selector"
        assert GateType.MODEL_OUTPUT.value == "model_output"
        assert GateType.PERFORMANCE.value == "performance"
        assert GateType.COVERAGE.value == "coverage"
        assert GateType.ERROR_RATE.value == "error_rate"


class TestGateStatusEnum:
    """Tests for GateStatus enum."""

    def test_gate_status_values(self, mock_env_vars):
        """Test GateStatus enum values."""
        from src.orchestrator.regression_gates import GateStatus

        assert GateStatus.PENDING.value == "pending"
        assert GateStatus.PASSED.value == "passed"
        assert GateStatus.FAILED.value == "failed"
        assert GateStatus.WARNING.value == "warning"
        assert GateStatus.SKIPPED.value == "skipped"


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_gate_result_creation(self, mock_env_vars):
        """Test GateResult creation."""
        from src.orchestrator.regression_gates import (
            GateResult,
            GateType,
            GateStatus,
            GateSeverity,
        )

        result = GateResult(
            gate_id="test_gate",
            gate_type=GateType.PERFORMANCE,
            name="Performance Test",
            status=GateStatus.PASSED,
            severity=GateSeverity.BLOCKING,
            current_value=95.0,
            baseline_value=100.0,
            message="Performance within limits",
        )

        assert result.gate_id == "test_gate"
        assert result.status == GateStatus.PASSED
        assert result.is_blocking_failure is False

    def test_gate_result_blocking_failure(self, mock_env_vars):
        """Test blocking failure detection."""
        from src.orchestrator.regression_gates import (
            GateResult,
            GateType,
            GateStatus,
            GateSeverity,
        )

        result = GateResult(
            gate_id="failing_gate",
            gate_type=GateType.ERROR_RATE,
            name="Error Rate Check",
            status=GateStatus.FAILED,
            severity=GateSeverity.BLOCKING,
        )

        assert result.is_blocking_failure is True

    def test_gate_result_to_dict(self, mock_env_vars):
        """Test GateResult serialization."""
        from src.orchestrator.regression_gates import (
            GateResult,
            GateType,
            GateStatus,
            GateSeverity,
        )

        result = GateResult(
            gate_id="test",
            gate_type=GateType.SELECTOR,
            name="Selector Test",
            status=GateStatus.PASSED,
            severity=GateSeverity.WARNING,
        )

        data = result.to_dict()

        assert data["gate_id"] == "test"
        assert data["status"] == "passed"
        assert data["severity"] == "warning"


class TestRegressionGateRegistry:
    """Tests for RegressionGateRegistry class."""

    def test_registry_initialization(self, mock_env_vars):
        """Test registry initialization with default gates."""
        from src.orchestrator.regression_gates import RegressionGateRegistry

        registry = RegressionGateRegistry()

        assert registry is not None
        assert "selector_stability" in registry._gates
        assert "model_output_consistency" in registry._gates
        assert "performance_regression" in registry._gates
        assert "coverage_regression" in registry._gates
        assert "error_rate_check" in registry._gates

    def test_registry_default_suites(self, mock_env_vars):
        """Test default suites are registered."""
        from src.orchestrator.regression_gates import RegressionGateRegistry

        registry = RegressionGateRegistry()

        assert "pre_deployment" in registry._suites
        assert "production_promotion" in registry._suites

    def test_register_custom_gate(self, mock_env_vars):
        """Test registering a custom gate."""
        from src.orchestrator.regression_gates import (
            RegressionGateRegistry,
            GateType,
            GateThreshold,
        )

        registry = RegressionGateRegistry()

        async def custom_check(gate, context):
            return None

        registry.register_gate(
            gate_id="custom_gate",
            gate_type=GateType.CUSTOM,
            name="Custom Gate",
            description="A custom test gate",
            check_fn=custom_check,
            thresholds=[GateThreshold("metric", "lt", 100)],
        )

        assert "custom_gate" in registry._gates

    def test_set_and_get_baseline(self, mock_env_vars):
        """Test setting and getting baseline metrics."""
        from src.orchestrator.regression_gates import RegressionGateRegistry

        registry = RegressionGateRegistry()

        registry.set_baseline(
            context="agent_TestAgent_prod",
            metrics={
                "p95_latency_ms": 150.0,
                "error_rate": 0.5,
            },
        )

        baseline = registry.get_baseline("agent_TestAgent_prod")

        assert baseline is not None
        assert baseline["p95_latency_ms"] == 150.0
        assert baseline["error_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_run_gate_selector_stability(self, mock_env_vars):
        """Test running selector stability gate."""
        from src.orchestrator.regression_gates import (
            RegressionGateRegistry,
            GateStatus,
        )

        registry = RegressionGateRegistry()

        context = {
            "test_results": [{"id": i} for i in range(100)],
            "selector_failures": [{"selector": "#btn"} for _ in range(3)],
        }

        result = await registry.run_gate("selector_stability", context)

        assert result.status == GateStatus.PASSED  # 3% failure rate < 5% threshold
        assert result.current_value == 3.0

    @pytest.mark.asyncio
    async def test_run_gate_error_rate(self, mock_env_vars):
        """Test running error rate gate."""
        from src.orchestrator.regression_gates import (
            RegressionGateRegistry,
            GateStatus,
        )

        registry = RegressionGateRegistry()

        context = {
            "total_requests": 1000,
            "errors": 5,
            "critical_errors": 0,
        }

        result = await registry.run_gate("error_rate_check", context)

        assert result.status == GateStatus.PASSED  # 0.5% error rate < 1% threshold

    @pytest.mark.asyncio
    async def test_run_gate_skipped_no_data(self, mock_env_vars):
        """Test gate is skipped when no data available."""
        from src.orchestrator.regression_gates import (
            RegressionGateRegistry,
            GateStatus,
        )

        registry = RegressionGateRegistry()

        result = await registry.run_gate("selector_stability", {})

        assert result.status == GateStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_run_suite(self, mock_env_vars):
        """Test running a gate suite."""
        from src.orchestrator.regression_gates import (
            RegressionGateRegistry,
            GateStatus,
        )

        registry = RegressionGateRegistry()

        context = {
            "test_results": [{"id": i} for i in range(100)],
            "selector_failures": [],
            "model_outputs": [{"result": "ok"}],
            "latencies_ms": [50, 60, 70, 80, 90],
            "total_requests": 100,
            "errors": 0,
            "critical_errors": 0,
        }

        result = await registry.run_suite("pre_deployment", context)

        assert result.suite_id == "pre_deployment"
        assert result.total_gates == 4
        assert result.passed_gates + result.skipped_gates >= result.total_gates

    @pytest.mark.asyncio
    async def test_run_suite_fail_fast(self, mock_env_vars):
        """Test suite fail-fast behavior."""
        from src.orchestrator.regression_gates import (
            RegressionGateRegistry,
            GateStatus,
        )

        registry = RegressionGateRegistry()

        # Context that will fail first gate
        context = {
            "test_results": [{"id": i} for i in range(100)],
            "selector_failures": [{"selector": f"#btn{i}"} for i in range(50)],  # 50% failure
        }

        result = await registry.run_suite("production_promotion", context)

        # Should fail and stop early (fail_fast=True for production_promotion)
        assert result.status == GateStatus.FAILED
        assert len(result.results) == 1  # Stopped after first failure


class TestGateSuiteResult:
    """Tests for GateSuiteResult dataclass."""

    def test_suite_result_blocking_failures(self, mock_env_vars):
        """Test getting blocking failures from suite result."""
        from src.orchestrator.regression_gates import (
            GateSuiteResult,
            GateResult,
            GateType,
            GateStatus,
            GateSeverity,
        )

        suite_result = GateSuiteResult(
            suite_id="test_suite",
            suite_name="Test Suite",
            status=GateStatus.FAILED,
            results=[
                GateResult(
                    gate_id="gate1",
                    gate_type=GateType.SELECTOR,
                    name="Gate 1",
                    status=GateStatus.PASSED,
                    severity=GateSeverity.BLOCKING,
                ),
                GateResult(
                    gate_id="gate2",
                    gate_type=GateType.PERFORMANCE,
                    name="Gate 2",
                    status=GateStatus.FAILED,
                    severity=GateSeverity.BLOCKING,
                ),
                GateResult(
                    gate_id="gate3",
                    gate_type=GateType.COVERAGE,
                    name="Gate 3",
                    status=GateStatus.FAILED,
                    severity=GateSeverity.WARNING,  # Not blocking
                ),
            ],
        )

        blocking = suite_result.blocking_failures

        assert len(blocking) == 1
        assert blocking[0].gate_id == "gate2"


class TestPromotionRegressionGate:
    """Tests for promotion_regression_gate integration."""

    @pytest.mark.asyncio
    async def test_promotion_regression_gate(self, mock_env_vars):
        """Test promotion regression gate integration."""
        from src.orchestrator.regression_gates import promotion_regression_gate
        from src.orchestrator.agent_promotion import (
            AgentVersion,
            Environment,
            PromotionRequest,
        )

        version = AgentVersion(
            agent_type="TestAgent",
            version="1.0.0",
            config_hash="hash",
        )

        request = PromotionRequest(
            request_id="test_request",
            agent_type="TestAgent",
            version=version,
            source_environment=Environment.DEVELOPMENT,
            target_environment=Environment.STAGING,
            requested_by="user",
        )

        # With empty context, gates should be skipped (passing)
        passed, message = await promotion_regression_gate(request)

        # Should pass because no data to evaluate
        assert passed is True or "passed" in message.lower() or "skipped" in message.lower()


class TestGetRegressionGateRegistrySingleton:
    """Tests for get_regression_gate_registry singleton."""

    def test_returns_singleton(self, mock_env_vars):
        """Test singleton pattern."""
        from src.orchestrator.regression_gates import get_regression_gate_registry

        # Reset singleton
        import src.orchestrator.regression_gates as module
        module._gate_registry = None

        registry1 = get_regression_gate_registry()
        registry2 = get_regression_gate_registry()

        assert registry1 is registry2
