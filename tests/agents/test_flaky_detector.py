"""Tests for the flaky test detector module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


class TestFlakinessLevel:
    """Tests for FlakinessLevel enum."""

    def test_flakiness_levels(self, mock_env_vars):
        """Test FlakinessLevel enum values."""
        from src.agents.flaky_detector import FlakinessLevel

        assert FlakinessLevel.STABLE.value == "stable"
        assert FlakinessLevel.SLIGHTLY_FLAKY.value == "slightly_flaky"
        assert FlakinessLevel.MODERATELY_FLAKY.value == "moderately_flaky"
        assert FlakinessLevel.HIGHLY_FLAKY.value == "highly_flaky"
        assert FlakinessLevel.QUARANTINED.value == "quarantined"


class TestFlakyCause:
    """Tests for FlakyCause enum."""

    def test_flaky_causes(self, mock_env_vars):
        """Test FlakyCause enum values."""
        from src.agents.flaky_detector import FlakyCause

        assert FlakyCause.TIMING.value == "timing"
        assert FlakyCause.NETWORK.value == "network"
        assert FlakyCause.DATA_DEPENDENCY.value == "data"
        assert FlakyCause.RESOURCE_CONTENTION.value == "resource"
        assert FlakyCause.UI_ANIMATION.value == "animation"
        assert FlakyCause.THIRD_PARTY.value == "third_party"
        assert FlakyCause.UNKNOWN.value == "unknown"


class TestTestRun:
    """Tests for TestRun dataclass."""

    def test_test_run_creation(self, mock_env_vars):
        """Test TestRun creation."""
        from src.agents.flaky_detector import TestRun

        run = TestRun(
            test_id="test-001",
            passed=True,
            duration_ms=150.0,
            timestamp=datetime.now(),
        )

        assert run.test_id == "test-001"
        assert run.passed is True
        assert run.duration_ms == 150.0
        assert run.error_message is None
        assert run.retry_number == 0

    def test_test_run_with_error(self, mock_env_vars):
        """Test TestRun with error message."""
        from src.agents.flaky_detector import TestRun

        run = TestRun(
            test_id="test-002",
            passed=False,
            duration_ms=5000.0,
            timestamp=datetime.now(),
            error_message="Element not found",
            environment="CI",
            retry_number=1,
            ci_run_id="run-123",
        )

        assert run.passed is False
        assert run.error_message == "Element not found"
        assert run.environment == "CI"


class TestFlakinessReport:
    """Tests for FlakinessReport dataclass."""

    def test_flakiness_report_creation(self, mock_env_vars):
        """Test FlakinessReport creation."""
        from src.agents.flaky_detector import FlakinessReport, FlakinessLevel, FlakyCause

        report = FlakinessReport(
            test_id="test-001",
            test_name="Login Test",
            flakiness_level=FlakinessLevel.SLIGHTLY_FLAKY,
            flakiness_score=0.1,
            pass_rate=0.9,
            total_runs=100,
            recent_failures=5,
            avg_duration_ms=200.0,
            duration_variance=50.0,
            likely_cause=FlakyCause.TIMING,
            cause_confidence=0.8,
            recommended_action="Add explicit waits",
            should_quarantine=False,
        )

        assert report.test_id == "test-001"
        assert report.flakiness_level == FlakinessLevel.SLIGHTLY_FLAKY
        assert report.pass_rate == 0.9


class TestQuarantineConfig:
    """Tests for QuarantineConfig dataclass."""

    def test_quarantine_config_defaults(self, mock_env_vars):
        """Test QuarantineConfig defaults."""
        from src.agents.flaky_detector import QuarantineConfig

        config = QuarantineConfig()

        assert config.auto_quarantine_threshold == 0.3
        assert config.auto_restore_threshold == 0.95
        assert config.min_runs_for_decision == 10
        assert config.quarantine_duration_days == 7
        assert config.retry_on_failure == 3

    def test_quarantine_config_custom(self, mock_env_vars):
        """Test QuarantineConfig with custom values."""
        from src.agents.flaky_detector import QuarantineConfig

        config = QuarantineConfig(
            auto_quarantine_threshold=0.5,
            retry_on_failure=5,
        )

        assert config.auto_quarantine_threshold == 0.5
        assert config.retry_on_failure == 5


class TestFlakyTestDetector:
    """Tests for FlakyTestDetector class."""

    def test_detector_creation(self, mock_env_vars):
        """Test FlakyTestDetector creation."""
        from src.agents.flaky_detector import FlakyTestDetector

        detector = FlakyTestDetector()

        assert detector.config is not None
        assert detector.test_history == {}
        assert detector.quarantined_tests == {}

    def test_detector_custom_config(self, mock_env_vars):
        """Test FlakyTestDetector with custom config."""
        from src.agents.flaky_detector import FlakyTestDetector, QuarantineConfig

        config = QuarantineConfig(auto_quarantine_threshold=0.4)
        detector = FlakyTestDetector(config=config)

        assert detector.config.auto_quarantine_threshold == 0.4

    def test_record_run(self, mock_env_vars):
        """Test recording a test run."""
        from src.agents.flaky_detector import FlakyTestDetector, TestRun

        detector = FlakyTestDetector()
        run = TestRun(
            test_id="test-001",
            passed=True,
            duration_ms=150.0,
            timestamp=datetime.now(),
        )

        detector.record_run(run)

        assert "test-001" in detector.test_history
        assert len(detector.test_history["test-001"]) == 1

    def test_record_run_limits_history(self, mock_env_vars):
        """Test that record_run limits history to 100 runs."""
        from src.agents.flaky_detector import FlakyTestDetector, TestRun

        detector = FlakyTestDetector()

        # Add 105 runs
        for i in range(105):
            run = TestRun(
                test_id="test-001",
                passed=True,
                duration_ms=150.0,
                timestamp=datetime.now(),
            )
            detector.record_run(run)

        # Should be limited to 100
        assert len(detector.test_history["test-001"]) == 100

    def test_analyze_test_no_runs(self, mock_env_vars):
        """Test analyzing a test with no runs."""
        from src.agents.flaky_detector import FlakyTestDetector, FlakinessLevel

        detector = FlakyTestDetector()
        report = detector.analyze_test("test-001", "Test Name")

        assert report.test_id == "test-001"
        assert report.flakiness_level == FlakinessLevel.STABLE
        assert report.total_runs == 0

    def test_analyze_test_stable(self, mock_env_vars):
        """Test analyzing a stable test."""
        from src.agents.flaky_detector import FlakyTestDetector, TestRun, FlakinessLevel

        detector = FlakyTestDetector()

        # Add 20 passing runs
        for i in range(20):
            run = TestRun(
                test_id="test-001",
                passed=True,
                duration_ms=150.0,
                timestamp=datetime.now(),
            )
            detector.record_run(run)

        report = detector.analyze_test("test-001", "Stable Test")

        assert report.pass_rate == 1.0
        assert report.flakiness_level == FlakinessLevel.STABLE

    def test_analyze_test_flaky(self, mock_env_vars):
        """Test analyzing a flaky test."""
        from src.agents.flaky_detector import FlakyTestDetector, TestRun, FlakinessLevel

        detector = FlakyTestDetector()

        # Add alternating pass/fail runs
        for i in range(20):
            run = TestRun(
                test_id="test-001",
                passed=(i % 2 == 0),
                duration_ms=150.0,
                timestamp=datetime.now(),
                error_message=None if i % 2 == 0 else "timeout error",
            )
            detector.record_run(run)

        report = detector.analyze_test("test-001", "Flaky Test")

        assert report.pass_rate == 0.5
        assert report.flakiness_level in [
            FlakinessLevel.MODERATELY_FLAKY,
            FlakinessLevel.HIGHLY_FLAKY,
        ]

    def test_analyze_test_caching(self, mock_env_vars):
        """Test that analysis results are cached."""
        from src.agents.flaky_detector import FlakyTestDetector, TestRun

        detector = FlakyTestDetector()

        for i in range(10):
            run = TestRun(
                test_id="test-001",
                passed=True,
                duration_ms=150.0,
                timestamp=datetime.now(),
            )
            detector.record_run(run)

        report1 = detector.analyze_test("test-001")
        report2 = detector.analyze_test("test-001")

        # Should be the same cached object
        assert report1 is report2

    def test_analyze_test_cache_invalidation(self, mock_env_vars):
        """Test that cache is invalidated on new run."""
        from src.agents.flaky_detector import FlakyTestDetector, TestRun

        detector = FlakyTestDetector()

        for i in range(10):
            run = TestRun(
                test_id="test-001",
                passed=True,
                duration_ms=150.0,
                timestamp=datetime.now(),
            )
            detector.record_run(run)

        report1 = detector.analyze_test("test-001")

        # Add new run
        detector.record_run(TestRun(
            test_id="test-001",
            passed=False,
            duration_ms=200.0,
            timestamp=datetime.now(),
        ))

        report2 = detector.analyze_test("test-001")

        # Should be different (cache invalidated)
        assert report1 is not report2

    def test_calculate_flakiness_score_few_runs(self, mock_env_vars):
        """Test flakiness score with few runs."""
        from src.agents.flaky_detector import FlakyTestDetector, TestRun

        detector = FlakyTestDetector()

        # Only 2 runs
        for i in range(2):
            run = TestRun(
                test_id="test-001",
                passed=True,
                duration_ms=150.0,
                timestamp=datetime.now(),
            )
            detector.record_run(run)

        runs = detector.test_history["test-001"]
        score = detector._calculate_flakiness_score(runs)

        assert score == 0.0  # Too few runs

    def test_determine_level(self, mock_env_vars):
        """Test flakiness level determination."""
        from src.agents.flaky_detector import FlakyTestDetector, FlakinessLevel

        detector = FlakyTestDetector()

        assert detector._determine_level(0.02, 0.98) == FlakinessLevel.STABLE
        assert detector._determine_level(0.10, 0.90) == FlakinessLevel.SLIGHTLY_FLAKY
        assert detector._determine_level(0.20, 0.80) == FlakinessLevel.MODERATELY_FLAKY
        assert detector._determine_level(0.40, 0.60) == FlakinessLevel.HIGHLY_FLAKY

    def test_analyze_patterns_timing(self, mock_env_vars):
        """Test pattern analysis for timing issues."""
        from src.agents.flaky_detector import FlakyTestDetector, TestRun, FlakyCause

        detector = FlakyTestDetector()

        # Add runs with timeout errors
        for i in range(10):
            run = TestRun(
                test_id="test-001",
                passed=i < 5,
                duration_ms=150.0,
                timestamp=datetime.now(),
                error_message=None if i < 5 else "timeout waiting for element",
            )
            detector.record_run(run)

        runs = detector.test_history["test-001"]
        patterns, cause, confidence = detector._analyze_patterns(runs)

        assert FlakyCause.TIMING == cause or len(patterns) > 0

    def test_should_quarantine(self, mock_env_vars):
        """Test quarantine decision."""
        from src.agents.flaky_detector import FlakyTestDetector

        detector = FlakyTestDetector()

        # Not enough runs
        assert detector._should_quarantine(0.5, 0.5, 5) is False

        # Enough runs, high fail rate
        assert detector._should_quarantine(0.5, 0.5, 15) is True

        # Enough runs, acceptable fail rate
        assert detector._should_quarantine(0.1, 0.9, 15) is False

    def test_generate_recommendation(self, mock_env_vars):
        """Test recommendation generation."""
        from src.agents.flaky_detector import FlakyTestDetector, FlakinessLevel, FlakyCause

        detector = FlakyTestDetector()

        rec = detector._generate_recommendation(
            FlakinessLevel.STABLE,
            FlakyCause.UNKNOWN,
            False
        )
        assert "stable" in rec.lower()

        rec = detector._generate_recommendation(
            FlakinessLevel.HIGHLY_FLAKY,
            FlakyCause.TIMING,
            True
        )
        assert "QUARANTINE" in rec

    def test_quarantine_test(self, mock_env_vars):
        """Test quarantining a test."""
        from src.agents.flaky_detector import FlakyTestDetector

        detector = FlakyTestDetector()
        detector.quarantine_test("test-001")

        assert "test-001" in detector.quarantined_tests
        assert detector.is_quarantined("test-001") is True

    def test_restore_test(self, mock_env_vars):
        """Test restoring a quarantined test."""
        from src.agents.flaky_detector import FlakyTestDetector

        detector = FlakyTestDetector()
        detector.quarantine_test("test-001")
        detector.restore_test("test-001")

        assert "test-001" not in detector.quarantined_tests
        assert detector.is_quarantined("test-001") is False

    def test_is_quarantined_expired(self, mock_env_vars):
        """Test quarantine expiration."""
        from src.agents.flaky_detector import FlakyTestDetector, QuarantineConfig

        config = QuarantineConfig(quarantine_duration_days=7)
        detector = FlakyTestDetector(config=config)

        # Set quarantine date to 8 days ago
        detector.quarantined_tests["test-001"] = datetime.utcnow() - timedelta(days=8)

        # Should be expired
        assert detector.is_quarantined("test-001") is False
        assert "test-001" not in detector.quarantined_tests

    def test_should_retry_not_flaky(self, mock_env_vars):
        """Test retry decision for stable test."""
        from src.agents.flaky_detector import FlakyTestDetector

        detector = FlakyTestDetector()

        # No history - shouldn't retry
        assert detector.should_retry("test-001", 0) is False

    def test_should_retry_max_attempts(self, mock_env_vars):
        """Test retry max attempts limit."""
        from src.agents.flaky_detector import FlakyTestDetector, QuarantineConfig

        config = QuarantineConfig(retry_on_failure=3)
        detector = FlakyTestDetector(config=config)

        assert detector.should_retry("test-001", 3) is False

    def test_get_flaky_tests_report(self, mock_env_vars):
        """Test generating flaky tests report."""
        from src.agents.flaky_detector import FlakyTestDetector, TestRun

        detector = FlakyTestDetector()

        # Add stable test
        for i in range(20):
            detector.record_run(TestRun(
                test_id="stable-test",
                passed=True,
                duration_ms=100.0,
                timestamp=datetime.now(),
            ))

        # Add flaky test
        for i in range(20):
            detector.record_run(TestRun(
                test_id="flaky-test",
                passed=(i % 2 == 0),
                duration_ms=100.0,
                timestamp=datetime.now(),
                error_message=None if i % 2 == 0 else "error",
            ))

        report = detector.get_flaky_tests_report()

        assert report["total_tests"] == 2
        assert report["flaky_tests"] >= 1
        assert "by_level" in report
        assert "by_cause" in report
        assert "tests" in report


class TestSmartRetryStrategy:
    """Tests for SmartRetryStrategy class."""

    def test_strategy_creation(self, mock_env_vars):
        """Test SmartRetryStrategy creation."""
        from src.agents.flaky_detector import SmartRetryStrategy, FlakyTestDetector

        detector = FlakyTestDetector()
        strategy = SmartRetryStrategy(detector)

        assert strategy.detector is detector

    def test_get_retry_config_stable(self, mock_env_vars):
        """Test retry config for stable test."""
        from src.agents.flaky_detector import SmartRetryStrategy, FlakyTestDetector, TestRun

        detector = FlakyTestDetector()
        strategy = SmartRetryStrategy(detector)

        # Add stable runs
        for i in range(20):
            detector.record_run(TestRun(
                test_id="test-001",
                passed=True,
                duration_ms=100.0,
                timestamp=datetime.now(),
            ))

        config = strategy.get_retry_config("test-001")

        assert config["max_retries"] == 1
        assert config["delay_ms"] == 0

    def test_get_retry_config_timing(self, mock_env_vars):
        """Test retry config for timing-related flakiness."""
        from src.agents.flaky_detector import (
            SmartRetryStrategy, FlakyTestDetector, TestRun,
            FlakinessLevel, FlakyCause, FlakinessReport
        )

        detector = FlakyTestDetector()
        strategy = SmartRetryStrategy(detector)

        # Manually set a cached report with timing cause
        detector.flakiness_cache["test-001"] = FlakinessReport(
            test_id="test-001",
            test_name="Test",
            flakiness_level=FlakinessLevel.SLIGHTLY_FLAKY,
            flakiness_score=0.1,
            pass_rate=0.9,
            total_runs=20,
            recent_failures=2,
            avg_duration_ms=100.0,
            duration_variance=10.0,
            likely_cause=FlakyCause.TIMING,
            cause_confidence=0.8,
            recommended_action="Add waits",
            should_quarantine=False,
        )

        config = strategy.get_retry_config("test-001")

        assert config["max_retries"] == 3
        assert config["delay_ms"] == 2000
        assert config["backoff_multiplier"] == 1.5

    def test_get_retry_config_network(self, mock_env_vars):
        """Test retry config for network-related flakiness."""
        from src.agents.flaky_detector import (
            SmartRetryStrategy, FlakyTestDetector,
            FlakinessLevel, FlakyCause, FlakinessReport
        )

        detector = FlakyTestDetector()
        strategy = SmartRetryStrategy(detector)

        detector.flakiness_cache["test-001"] = FlakinessReport(
            test_id="test-001",
            test_name="Test",
            flakiness_level=FlakinessLevel.SLIGHTLY_FLAKY,
            flakiness_score=0.1,
            pass_rate=0.9,
            total_runs=20,
            recent_failures=2,
            avg_duration_ms=100.0,
            duration_variance=10.0,
            likely_cause=FlakyCause.NETWORK,
            cause_confidence=0.8,
            recommended_action="Add retries",
            should_quarantine=False,
        )

        config = strategy.get_retry_config("test-001")

        assert config["max_retries"] == 3
        assert config["backoff_multiplier"] == 2

    def test_get_retry_config_resource(self, mock_env_vars):
        """Test retry config for resource contention."""
        from src.agents.flaky_detector import (
            SmartRetryStrategy, FlakyTestDetector,
            FlakinessLevel, FlakyCause, FlakinessReport
        )

        detector = FlakyTestDetector()
        strategy = SmartRetryStrategy(detector)

        detector.flakiness_cache["test-001"] = FlakinessReport(
            test_id="test-001",
            test_name="Test",
            flakiness_level=FlakinessLevel.SLIGHTLY_FLAKY,
            flakiness_score=0.1,
            pass_rate=0.9,
            total_runs=20,
            recent_failures=2,
            avg_duration_ms=100.0,
            duration_variance=10.0,
            likely_cause=FlakyCause.RESOURCE_CONTENTION,
            cause_confidence=0.8,
            recommended_action="Reduce parallelism",
            should_quarantine=False,
        )

        config = strategy.get_retry_config("test-001")

        assert config["max_retries"] == 2
        assert config["delay_ms"] == 5000
