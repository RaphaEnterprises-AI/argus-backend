"""
Flaky Test Detection & Quarantine System

This is a KEY DIFFERENTIATOR. We automatically:
1. Detect flaky tests through statistical analysis
2. Quarantine them to prevent blocking CI/CD
3. Retry intelligently to confirm true failures
4. Report on flakiness trends over time
5. Suggest fixes for common flakiness patterns
"""

import json
import statistics
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

from src.config import get_settings


class FlakinessLevel(str, Enum):
    STABLE = "stable"           # < 5% fail rate, consistent
    SLIGHTLY_FLAKY = "slightly_flaky"  # 5-15% intermittent failures
    MODERATELY_FLAKY = "moderately_flaky"  # 15-30% failures
    HIGHLY_FLAKY = "highly_flaky"  # > 30% failures
    QUARANTINED = "quarantined"  # Removed from critical path


class FlakyCause(str, Enum):
    TIMING = "timing"           # Race conditions, async issues
    NETWORK = "network"         # API latency, network variability
    DATA_DEPENDENCY = "data"    # Test data or state issues
    RESOURCE_CONTENTION = "resource"  # CPU/memory issues
    UI_ANIMATION = "animation"  # Animation timing
    THIRD_PARTY = "third_party"  # External service variability
    UNKNOWN = "unknown"


@dataclass
class TestRun:
    """Record of a single test run."""
    test_id: str
    passed: bool
    duration_ms: float
    timestamp: datetime
    error_message: Optional[str] = None
    environment: Optional[str] = None
    retry_number: int = 0
    ci_run_id: Optional[str] = None


@dataclass
class FlakinessReport:
    """Comprehensive flakiness report for a test."""
    test_id: str
    test_name: str
    flakiness_level: FlakinessLevel
    flakiness_score: float  # 0.0 (stable) to 1.0 (always flaky)
    pass_rate: float
    total_runs: int
    recent_failures: int
    avg_duration_ms: float
    duration_variance: float
    likely_cause: FlakyCause
    cause_confidence: float
    recommended_action: str
    should_quarantine: bool
    first_flaky_date: Optional[datetime] = None
    failure_patterns: list[str] = field(default_factory=list)


@dataclass
class QuarantineConfig:
    """Configuration for test quarantine."""
    auto_quarantine_threshold: float = 0.3  # 30% fail rate
    auto_restore_threshold: float = 0.95    # 95% pass rate
    min_runs_for_decision: int = 10
    quarantine_duration_days: int = 7
    retry_on_failure: int = 3


class FlakyTestDetector:
    """
    Intelligent flaky test detection and management.

    Features:
    - Statistical analysis of test history
    - Pattern recognition for failure causes
    - Automatic quarantine and restoration
    - Smart retry strategies
    - Trend reporting
    """

    def __init__(self, config: Optional[QuarantineConfig] = None):
        self.config = config or QuarantineConfig()
        self.test_history: dict[str, list[TestRun]] = defaultdict(list)
        self.quarantined_tests: dict[str, datetime] = {}
        self.flakiness_cache: dict[str, FlakinessReport] = {}

    def record_run(self, run: TestRun):
        """Record a test run for analysis."""
        self.test_history[run.test_id].append(run)

        # Keep last 100 runs per test
        if len(self.test_history[run.test_id]) > 100:
            self.test_history[run.test_id] = self.test_history[run.test_id][-100:]

        # Invalidate cache
        if run.test_id in self.flakiness_cache:
            del self.flakiness_cache[run.test_id]

    def analyze_test(
        self,
        test_id: str,
        test_name: str = ""
    ) -> FlakinessReport:
        """Analyze a test for flakiness."""
        if test_id in self.flakiness_cache:
            return self.flakiness_cache[test_id]

        runs = self.test_history.get(test_id, [])

        if not runs:
            return FlakinessReport(
                test_id=test_id,
                test_name=test_name,
                flakiness_level=FlakinessLevel.STABLE,
                flakiness_score=0.0,
                pass_rate=1.0,
                total_runs=0,
                recent_failures=0,
                avg_duration_ms=0,
                duration_variance=0,
                likely_cause=FlakyCause.UNKNOWN,
                cause_confidence=0.0,
                recommended_action="Need more test runs for analysis",
                should_quarantine=False
            )

        # Calculate basic stats
        total_runs = len(runs)
        passes = sum(1 for r in runs if r.passed)
        pass_rate = passes / total_runs

        durations = [r.duration_ms for r in runs]
        avg_duration = statistics.mean(durations)
        duration_variance = statistics.variance(durations) if len(durations) > 1 else 0

        # Recent failures (last 20 runs)
        recent_runs = runs[-20:]
        recent_failures = sum(1 for r in recent_runs if not r.passed)

        # Calculate flakiness score
        flakiness_score = self._calculate_flakiness_score(runs)

        # Determine level
        flakiness_level = self._determine_level(flakiness_score, pass_rate)

        # Analyze failure patterns
        failure_patterns, likely_cause, cause_confidence = self._analyze_patterns(runs)

        # Determine if should quarantine
        should_quarantine = self._should_quarantine(flakiness_score, pass_rate, total_runs)

        # Generate recommendation
        recommended_action = self._generate_recommendation(
            flakiness_level, likely_cause, should_quarantine
        )

        # Find first flaky date
        first_flaky_date = self._find_first_flaky_date(runs)

        report = FlakinessReport(
            test_id=test_id,
            test_name=test_name or test_id,
            flakiness_level=flakiness_level,
            flakiness_score=flakiness_score,
            pass_rate=pass_rate,
            total_runs=total_runs,
            recent_failures=recent_failures,
            avg_duration_ms=avg_duration,
            duration_variance=duration_variance,
            likely_cause=likely_cause,
            cause_confidence=cause_confidence,
            recommended_action=recommended_action,
            should_quarantine=should_quarantine,
            first_flaky_date=first_flaky_date,
            failure_patterns=failure_patterns
        )

        self.flakiness_cache[test_id] = report
        return report

    def _calculate_flakiness_score(self, runs: list[TestRun]) -> float:
        """
        Calculate a flakiness score from 0 (stable) to 1 (chaotic).

        Considers:
        - Overall pass rate
        - Run-to-run transitions (pass→fail, fail→pass)
        - Clustering of failures
        - Duration variance
        """
        if len(runs) < 3:
            return 0.0

        # Pass rate factor (inverted - lower pass rate = more flaky)
        passes = sum(1 for r in runs if r.passed)
        pass_rate = passes / len(runs)

        # But very low pass rate might be a real bug, not flakiness
        if pass_rate < 0.2:
            return 0.3  # Likely a real issue, not flakiness

        # Transition factor - how often does it flip between pass/fail?
        transitions = 0
        for i in range(1, len(runs)):
            if runs[i].passed != runs[i-1].passed:
                transitions += 1

        transition_rate = transitions / (len(runs) - 1)

        # High transition rate = flaky (oscillating)
        # Low transition rate with mixed results = clustered failures

        # Combine factors
        if pass_rate > 0.95:
            # Mostly passing - check for occasional flakes
            flakiness = transition_rate * 0.5
        elif pass_rate > 0.7:
            # Some failures - weight transitions more
            flakiness = (1 - pass_rate) * 0.5 + transition_rate * 0.5
        else:
            # Many failures - could be flaky or broken
            flakiness = (1 - pass_rate) * 0.7 + transition_rate * 0.3

        return min(flakiness, 1.0)

    def _determine_level(
        self,
        flakiness_score: float,
        pass_rate: float
    ) -> FlakinessLevel:
        """Determine flakiness level from score."""
        if flakiness_score < 0.05:
            return FlakinessLevel.STABLE
        elif flakiness_score < 0.15:
            return FlakinessLevel.SLIGHTLY_FLAKY
        elif flakiness_score < 0.30:
            return FlakinessLevel.MODERATELY_FLAKY
        else:
            return FlakinessLevel.HIGHLY_FLAKY

    def _analyze_patterns(
        self,
        runs: list[TestRun]
    ) -> tuple[list[str], FlakyCause, float]:
        """Analyze failure patterns to determine likely cause."""
        patterns = []
        cause = FlakyCause.UNKNOWN
        confidence = 0.0

        failed_runs = [r for r in runs if not r.passed]
        if not failed_runs:
            return [], FlakyCause.UNKNOWN, 0.0

        # Analyze error messages
        error_messages = [r.error_message for r in failed_runs if r.error_message]

        # Timing patterns
        timing_keywords = ["timeout", "wait", "not found", "detached", "stale"]
        timing_matches = sum(
            1 for msg in error_messages
            if any(kw in msg.lower() for kw in timing_keywords)
        )
        if timing_matches > len(failed_runs) * 0.5:
            patterns.append("Frequent timeout/wait-related failures")
            cause = FlakyCause.TIMING
            confidence = timing_matches / len(failed_runs)

        # Network patterns
        network_keywords = ["network", "fetch", "api", "500", "503", "connection"]
        network_matches = sum(
            1 for msg in error_messages
            if any(kw in msg.lower() for kw in network_keywords)
        )
        if network_matches > len(failed_runs) * 0.4:
            patterns.append("Network/API-related failures")
            if network_matches / len(failed_runs) > confidence:
                cause = FlakyCause.NETWORK
                confidence = network_matches / len(failed_runs)

        # Duration variance analysis
        durations = [r.duration_ms for r in runs]
        if len(durations) > 5:
            cv = statistics.stdev(durations) / statistics.mean(durations)
            if cv > 0.5:
                patterns.append("High duration variance (resource contention?)")
                if confidence < 0.5:
                    cause = FlakyCause.RESOURCE_CONTENTION
                    confidence = 0.5

        # Time-of-day patterns
        failure_hours = [r.timestamp.hour for r in failed_runs]
        if failure_hours:
            hour_counts = defaultdict(int)
            for h in failure_hours:
                hour_counts[h] += 1
            max_hour = max(hour_counts.values())
            if max_hour > len(failed_runs) * 0.4:
                patterns.append(f"Failures cluster around certain hours")

        # Environment patterns
        failure_envs = [r.environment for r in failed_runs if r.environment]
        if failure_envs:
            env_counts = defaultdict(int)
            for e in failure_envs:
                env_counts[e] += 1
            if len(env_counts) == 1:
                patterns.append(f"All failures in {list(env_counts.keys())[0]} environment")

        return patterns, cause, confidence

    def _should_quarantine(
        self,
        flakiness_score: float,
        pass_rate: float,
        total_runs: int
    ) -> bool:
        """Determine if test should be quarantined."""
        if total_runs < self.config.min_runs_for_decision:
            return False

        # Quarantine if fail rate exceeds threshold
        fail_rate = 1 - pass_rate
        return fail_rate >= self.config.auto_quarantine_threshold

    def _generate_recommendation(
        self,
        level: FlakinessLevel,
        cause: FlakyCause,
        should_quarantine: bool
    ) -> str:
        """Generate actionable recommendation."""
        if level == FlakinessLevel.STABLE:
            return "Test is stable. No action needed."

        recommendations = []

        if should_quarantine:
            recommendations.append("QUARANTINE: Remove from critical CI path")

        cause_recommendations = {
            FlakyCause.TIMING: "Add explicit waits or increase timeouts",
            FlakyCause.NETWORK: "Add retry logic for API calls or mock external services",
            FlakyCause.DATA_DEPENDENCY: "Isolate test data or reset state between runs",
            FlakyCause.RESOURCE_CONTENTION: "Reduce parallelism or increase CI resources",
            FlakyCause.UI_ANIMATION: "Wait for animations to complete before assertions",
            FlakyCause.THIRD_PARTY: "Mock third-party services or add retry with backoff",
        }

        if cause in cause_recommendations:
            recommendations.append(cause_recommendations[cause])

        if level in [FlakinessLevel.MODERATELY_FLAKY, FlakinessLevel.HIGHLY_FLAKY]:
            recommendations.append("Consider rewriting test for better stability")

        return " | ".join(recommendations)

    def _find_first_flaky_date(self, runs: list[TestRun]) -> Optional[datetime]:
        """Find when test first became flaky."""
        if len(runs) < 3:
            return None

        # Look for first failure after a streak of passes
        streak = 0
        for run in runs:
            if run.passed:
                streak += 1
            else:
                if streak >= 3:  # Had at least 3 passes before
                    return run.timestamp
                streak = 0

        return None

    def quarantine_test(self, test_id: str):
        """Quarantine a test."""
        self.quarantined_tests[test_id] = datetime.utcnow()

    def restore_test(self, test_id: str):
        """Restore a quarantined test."""
        if test_id in self.quarantined_tests:
            del self.quarantined_tests[test_id]

    def is_quarantined(self, test_id: str) -> bool:
        """Check if a test is quarantined."""
        if test_id not in self.quarantined_tests:
            return False

        # Check if quarantine has expired
        quarantine_date = self.quarantined_tests[test_id]
        expiry = quarantine_date + timedelta(days=self.config.quarantine_duration_days)

        if datetime.utcnow() > expiry:
            del self.quarantined_tests[test_id]
            return False

        return True

    def should_retry(self, test_id: str, current_attempt: int) -> bool:
        """Determine if a failed test should be retried."""
        if current_attempt >= self.config.retry_on_failure:
            return False

        # Check if test is known to be flaky
        if test_id in self.flakiness_cache:
            report = self.flakiness_cache[test_id]
            if report.flakiness_level != FlakinessLevel.STABLE:
                return True

        return False

    def get_flaky_tests_report(self) -> dict:
        """Generate a report of all flaky tests."""
        reports = []

        for test_id in self.test_history:
            report = self.analyze_test(test_id)
            if report.flakiness_level != FlakinessLevel.STABLE:
                reports.append(report)

        # Sort by flakiness score
        reports.sort(key=lambda r: r.flakiness_score, reverse=True)

        return {
            "total_tests": len(self.test_history),
            "flaky_tests": len(reports),
            "quarantined_tests": len(self.quarantined_tests),
            "by_level": {
                "highly_flaky": len([r for r in reports if r.flakiness_level == FlakinessLevel.HIGHLY_FLAKY]),
                "moderately_flaky": len([r for r in reports if r.flakiness_level == FlakinessLevel.MODERATELY_FLAKY]),
                "slightly_flaky": len([r for r in reports if r.flakiness_level == FlakinessLevel.SLIGHTLY_FLAKY]),
            },
            "by_cause": {
                cause.value: len([r for r in reports if r.likely_cause == cause])
                for cause in FlakyCause
            },
            "tests": [
                {
                    "test_id": r.test_id,
                    "test_name": r.test_name,
                    "level": r.flakiness_level.value,
                    "score": round(r.flakiness_score, 3),
                    "pass_rate": round(r.pass_rate, 3),
                    "cause": r.likely_cause.value,
                    "recommendation": r.recommended_action,
                    "quarantined": r.test_id in self.quarantined_tests,
                }
                for r in reports
            ]
        }


class SmartRetryStrategy:
    """Intelligent retry strategy based on flakiness patterns."""

    def __init__(self, detector: FlakyTestDetector):
        self.detector = detector

    def get_retry_config(self, test_id: str) -> dict:
        """Get retry configuration for a test."""
        report = self.detector.analyze_test(test_id)

        if report.flakiness_level == FlakinessLevel.STABLE:
            return {
                "max_retries": 1,
                "delay_ms": 0,
                "backoff_multiplier": 1
            }

        if report.likely_cause == FlakyCause.TIMING:
            return {
                "max_retries": 3,
                "delay_ms": 2000,
                "backoff_multiplier": 1.5
            }

        if report.likely_cause == FlakyCause.NETWORK:
            return {
                "max_retries": 3,
                "delay_ms": 1000,
                "backoff_multiplier": 2
            }

        if report.likely_cause == FlakyCause.RESOURCE_CONTENTION:
            return {
                "max_retries": 2,
                "delay_ms": 5000,
                "backoff_multiplier": 1
            }

        # Default for flaky tests
        return {
            "max_retries": 2,
            "delay_ms": 1000,
            "backoff_multiplier": 1.5
        }
