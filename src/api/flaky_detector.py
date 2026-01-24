"""Flaky Test Detection System.

Tracks test outcomes over time and automatically flags flaky tests.
A flaky test is one that produces inconsistent results (passes and fails)
without any code changes.

Flaky Score Calculation:
    Score = number of status changes / (total runs - 1)
    - A perfectly consistent test = 0.0 (always passes or always fails)
    - A completely random test = ~0.5 (alternates between pass/fail)
    - Score > 0.3 is typically considered flaky

Recommendations:
    - Score > 0.5: quarantine (disable from CI, investigate immediately)
    - Score 0.3-0.5: investigate (likely environmental or timing issues)
    - Score 0.1-0.3: monitor (might be genuinely flaky or have edge cases)
    - Score < 0.1: stable (normal test behavior)
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Literal

import structlog

from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()


class FlakyRecommendation(str, Enum):
    """Recommendations based on flaky score."""
    QUARANTINE = "quarantine"  # Score > 0.5: Disable and investigate immediately
    INVESTIGATE = "investigate"  # Score 0.3-0.5: Likely environmental/timing issues
    MONITOR = "monitor"  # Score 0.1-0.3: Watch for patterns
    STABLE = "stable"  # Score < 0.1: Normal test behavior


@dataclass
class TestOutcome:
    """Represents a single test outcome."""
    test_id: str
    test_run_id: str
    status: Literal["passed", "failed", "skipped", "pending", "running"]
    completed_at: str | None
    duration_ms: int | None


@dataclass
class FlakyTestResult:
    """Result of flaky test analysis."""
    test_id: str
    test_name: str | None
    flaky_score: float
    recommendation: FlakyRecommendation
    total_runs: int
    status_changes: int
    pass_count: int
    fail_count: int
    pass_rate: float
    recent_statuses: list[str]  # Most recent first
    first_detected_at: str | None
    analysis_window_start: str | None
    analysis_window_end: str | None


class FlakyDetector:
    """Detects and tracks flaky tests based on historical outcomes.

    A flaky test is one that produces inconsistent results without code changes.
    This detector analyzes test execution history and calculates a flaky score
    based on how often the test status changes between runs.
    """

    # Thresholds for recommendations
    QUARANTINE_THRESHOLD = 0.5  # Score > 0.5: Quarantine the test
    INVESTIGATE_THRESHOLD = 0.3  # Score > 0.3: Investigate the test
    MONITOR_THRESHOLD = 0.1  # Score > 0.1: Monitor the test

    # Minimum runs required for reliable flaky detection
    MIN_RUNS_FOR_DETECTION = 5

    def __init__(self):
        """Initialize the flaky detector."""
        self._supabase = None

    @property
    def supabase(self):
        """Lazy-load Supabase client."""
        if self._supabase is None:
            self._supabase = get_supabase_client()
        return self._supabase

    async def calculate_flaky_score(
        self,
        test_id: str,
        window_size: int = 20,
    ) -> float:
        """Calculate flaky score for a test based on recent runs.

        The flaky score measures how often a test's status changes between runs.
        A higher score indicates more inconsistent behavior.

        Score = number of status changes / (total runs - 1)
        - A perfectly consistent test = 0.0 (always passes or always fails)
        - A completely random test = ~0.5 (alternates between pass/fail)

        Args:
            test_id: The ID of the test to analyze
            window_size: Number of recent runs to consider (default: 20)

        Returns:
            Flaky score between 0.0 and 1.0
        """
        outcomes = await self._get_test_outcomes(test_id, limit=window_size)

        if len(outcomes) < 2:
            # Not enough data to calculate flaky score
            return 0.0

        # Count status changes
        status_changes = 0
        for i in range(1, len(outcomes)):
            current_status = self._normalize_status(outcomes[i].status)
            previous_status = self._normalize_status(outcomes[i - 1].status)

            if current_status != previous_status:
                status_changes += 1

        # Calculate flaky score
        flaky_score = status_changes / (len(outcomes) - 1)

        logger.debug(
            "Calculated flaky score",
            test_id=test_id,
            window_size=len(outcomes),
            status_changes=status_changes,
            flaky_score=round(flaky_score, 3),
        )

        return flaky_score

    async def analyze_test(
        self,
        test_id: str,
        window_size: int = 20,
    ) -> FlakyTestResult | None:
        """Analyze a test for flakiness with detailed results.

        Args:
            test_id: The ID of the test to analyze
            window_size: Number of recent runs to consider

        Returns:
            FlakyTestResult with detailed analysis, or None if insufficient data
        """
        outcomes = await self._get_test_outcomes(test_id, limit=window_size)

        if len(outcomes) < self.MIN_RUNS_FOR_DETECTION:
            logger.debug(
                "Insufficient runs for flaky detection",
                test_id=test_id,
                runs=len(outcomes),
                min_required=self.MIN_RUNS_FOR_DETECTION,
            )
            return None

        # Get test metadata
        test_info = await self._get_test_info(test_id)
        test_name = test_info.get("name") if test_info else None

        # Calculate metrics
        statuses = [self._normalize_status(o.status) for o in outcomes]
        pass_count = sum(1 for s in statuses if s == "passed")
        fail_count = sum(1 for s in statuses if s == "failed")
        total_completed = pass_count + fail_count

        # Count status changes
        status_changes = 0
        for i in range(1, len(statuses)):
            if statuses[i] != statuses[i - 1]:
                status_changes += 1

        flaky_score = status_changes / (len(statuses) - 1) if len(statuses) > 1 else 0.0
        pass_rate = pass_count / total_completed if total_completed > 0 else 0.0

        # Determine recommendation
        recommendation = self._get_recommendation(flaky_score)

        # Get time window
        analysis_window_start = outcomes[-1].completed_at if outcomes else None
        analysis_window_end = outcomes[0].completed_at if outcomes else None

        # Find when flakiness was first detected (if applicable)
        first_detected_at = None
        if flaky_score >= self.MONITOR_THRESHOLD:
            first_detected_at = await self._get_first_flaky_detection(test_id)

        return FlakyTestResult(
            test_id=test_id,
            test_name=test_name,
            flaky_score=round(flaky_score, 4),
            recommendation=recommendation,
            total_runs=len(outcomes),
            status_changes=status_changes,
            pass_count=pass_count,
            fail_count=fail_count,
            pass_rate=round(pass_rate, 4),
            recent_statuses=statuses[:10],  # Last 10 statuses
            first_detected_at=first_detected_at,
            analysis_window_start=analysis_window_start,
            analysis_window_end=analysis_window_end,
        )

    async def detect_flaky_tests(
        self,
        schedule_id: str,
        window_size: int = 20,
        min_flaky_score: float = 0.1,
    ) -> list[dict]:
        """Analyze all tests in a schedule and return flaky ones.

        Args:
            schedule_id: The schedule ID to analyze
            window_size: Number of recent runs to consider per test
            min_flaky_score: Minimum score to be considered flaky (default: 0.1)

        Returns:
            List of flaky test results:
            [{"test_id": "...", "test_name": "...", "flaky_score": 0.4, "recommendation": "investigate", ...}]
        """
        # Get all test IDs from schedule runs
        test_ids = await self._get_tests_from_schedule(schedule_id)

        if not test_ids:
            logger.info("No tests found for schedule", schedule_id=schedule_id)
            return []

        logger.info(
            "Analyzing tests for flakiness",
            schedule_id=schedule_id,
            test_count=len(test_ids),
            window_size=window_size,
        )

        flaky_tests = []

        for test_id in test_ids:
            result = await self.analyze_test(test_id, window_size)

            if result and result.flaky_score >= min_flaky_score:
                flaky_tests.append({
                    "test_id": result.test_id,
                    "test_name": result.test_name,
                    "flaky_score": result.flaky_score,
                    "recommendation": result.recommendation.value,
                    "total_runs": result.total_runs,
                    "status_changes": result.status_changes,
                    "pass_count": result.pass_count,
                    "fail_count": result.fail_count,
                    "pass_rate": result.pass_rate,
                    "recent_statuses": result.recent_statuses,
                    "first_detected_at": result.first_detected_at,
                })

        # Sort by flaky score descending
        flaky_tests.sort(key=lambda x: x["flaky_score"], reverse=True)

        logger.info(
            "Flaky test detection completed",
            schedule_id=schedule_id,
            total_tests=len(test_ids),
            flaky_count=len(flaky_tests),
        )

        return flaky_tests

    async def detect_flaky_tests_for_project(
        self,
        project_id: str,
        window_size: int = 20,
        min_flaky_score: float = 0.1,
    ) -> list[dict]:
        """Analyze all tests in a project and return flaky ones.

        Args:
            project_id: The project ID to analyze
            window_size: Number of recent runs to consider per test
            min_flaky_score: Minimum score to be considered flaky

        Returns:
            List of flaky test results
        """
        # Get all active test IDs from project
        test_result = await self.supabase.request(
            f"/tests?project_id=eq.{project_id}&is_active=eq.true&select=id"
        )

        if test_result.get("error") or not test_result.get("data"):
            logger.warning("No tests found for project", project_id=project_id)
            return []

        test_ids = [t["id"] for t in test_result["data"]]

        logger.info(
            "Analyzing project tests for flakiness",
            project_id=project_id,
            test_count=len(test_ids),
        )

        flaky_tests = []

        for test_id in test_ids:
            result = await self.analyze_test(test_id, window_size)

            if result and result.flaky_score >= min_flaky_score:
                flaky_tests.append({
                    "test_id": result.test_id,
                    "test_name": result.test_name,
                    "flaky_score": result.flaky_score,
                    "recommendation": result.recommendation.value,
                    "total_runs": result.total_runs,
                    "status_changes": result.status_changes,
                    "pass_count": result.pass_count,
                    "fail_count": result.fail_count,
                    "pass_rate": result.pass_rate,
                    "recent_statuses": result.recent_statuses,
                })

        flaky_tests.sort(key=lambda x: x["flaky_score"], reverse=True)

        return flaky_tests

    async def update_flaky_status(
        self,
        test_id: str,
        is_flaky: bool,
        score: float,
    ) -> None:
        """Update the test's flaky status in the database.

        Stores flaky detection results in the test metadata for reference.

        Args:
            test_id: The test ID to update
            is_flaky: Whether the test is considered flaky
            score: The calculated flaky score
        """
        recommendation = self._get_recommendation(score)

        # Get current test to preserve existing metadata
        test_result = await self.supabase.request(
            f"/tests?id=eq.{test_id}&select=id,settings"
        )

        if test_result.get("error") or not test_result.get("data"):
            logger.warning("Test not found for flaky status update", test_id=test_id)
            return

        test = test_result["data"][0]
        current_settings = test.get("settings") or {}

        # Update flaky detection metadata in settings
        flaky_metadata = {
            "is_flaky": is_flaky,
            "flaky_score": round(score, 4),
            "recommendation": recommendation.value,
            "last_analyzed_at": datetime.now(UTC).isoformat(),
        }

        # If newly detected as flaky, record the detection time
        if is_flaky and not current_settings.get("flaky_detection", {}).get("is_flaky"):
            flaky_metadata["first_detected_at"] = datetime.now(UTC).isoformat()
        elif not is_flaky:
            # Test is no longer flaky - record when it was resolved
            flaky_metadata["resolved_at"] = datetime.now(UTC).isoformat()

        current_settings["flaky_detection"] = flaky_metadata

        # Update test
        update_result = await self.supabase.update(
            "tests",
            {"id": f"eq.{test_id}"},
            {
                "settings": current_settings,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )

        if update_result.get("error"):
            logger.error(
                "Failed to update flaky status",
                test_id=test_id,
                error=update_result.get("error"),
            )
            return

        logger.info(
            "Updated flaky status for test",
            test_id=test_id,
            is_flaky=is_flaky,
            score=round(score, 4),
            recommendation=recommendation.value,
        )

    async def record_test_outcome(
        self,
        test_id: str,
        test_run_id: str,
        status: str,
        schedule_id: str | None = None,
    ) -> dict | None:
        """Record a test outcome and update flaky detection.

        Call this after each test execution to keep flaky detection up to date.

        Args:
            test_id: The test ID
            test_run_id: The test run ID
            status: The test status (passed/failed)
            schedule_id: Optional schedule ID for context

        Returns:
            Updated flaky analysis result if the test is flaky, None otherwise
        """
        # Analyze the test with latest data
        result = await self.analyze_test(test_id)

        if result is None:
            # Not enough data yet
            return None

        is_flaky = result.flaky_score >= self.MONITOR_THRESHOLD

        # Update flaky status in database
        await self.update_flaky_status(test_id, is_flaky, result.flaky_score)

        if is_flaky:
            logger.warning(
                "Flaky test detected",
                test_id=test_id,
                test_name=result.test_name,
                flaky_score=result.flaky_score,
                recommendation=result.recommendation.value,
                schedule_id=schedule_id,
            )

            return {
                "test_id": result.test_id,
                "test_name": result.test_name,
                "flaky_score": result.flaky_score,
                "recommendation": result.recommendation.value,
                "status_changes": result.status_changes,
                "total_runs": result.total_runs,
            }

        return None

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _normalize_status(self, status: str) -> str:
        """Normalize test status for comparison.

        Groups statuses into pass/fail/other for flaky detection.
        """
        status_lower = status.lower()
        if status_lower in ("passed", "success"):
            return "passed"
        elif status_lower in ("failed", "failure", "error"):
            return "failed"
        else:
            # Skip, pending, running, etc. - treat as previous status
            return "other"

    def _get_recommendation(self, score: float) -> FlakyRecommendation:
        """Get recommendation based on flaky score."""
        if score >= self.QUARANTINE_THRESHOLD:
            return FlakyRecommendation.QUARANTINE
        elif score >= self.INVESTIGATE_THRESHOLD:
            return FlakyRecommendation.INVESTIGATE
        elif score >= self.MONITOR_THRESHOLD:
            return FlakyRecommendation.MONITOR
        else:
            return FlakyRecommendation.STABLE

    async def _get_test_outcomes(
        self,
        test_id: str,
        limit: int = 20,
    ) -> list[TestOutcome]:
        """Get recent test outcomes from the database.

        Queries test_results table for historical outcomes.
        """
        result = await self.supabase.request(
            f"/test_results?test_id=eq.{test_id}&status=in.(passed,failed)"
            f"&select=id,test_id,test_run_id,status,completed_at,duration_ms"
            f"&order=completed_at.desc&limit={limit}"
        )

        if result.get("error"):
            logger.error(
                "Failed to fetch test outcomes",
                test_id=test_id,
                error=result.get("error"),
            )
            return []

        outcomes = []
        for row in result.get("data", []):
            outcomes.append(TestOutcome(
                test_id=row["test_id"],
                test_run_id=row["test_run_id"],
                status=row["status"],
                completed_at=row.get("completed_at"),
                duration_ms=row.get("duration_ms"),
            ))

        return outcomes

    async def _get_test_info(self, test_id: str) -> dict | None:
        """Get test metadata from the database."""
        result = await self.supabase.request(
            f"/tests?id=eq.{test_id}&select=id,name,project_id,tags,priority"
        )

        if result.get("error") or not result.get("data"):
            return None

        return result["data"][0]

    async def _get_tests_from_schedule(self, schedule_id: str) -> list[str]:
        """Get all unique test IDs that have been run in a schedule.

        Looks up test_ids from the schedule configuration and also
        scans schedule_runs for any tests that were executed.
        """
        # First, get test_ids from schedule config
        schedule_result = await self.supabase.request(
            f"/test_schedules?id=eq.{schedule_id}&select=test_ids,project_id"
        )

        if schedule_result.get("error") or not schedule_result.get("data"):
            logger.warning("Schedule not found", schedule_id=schedule_id)
            return []

        schedule = schedule_result["data"][0]
        test_ids = set(schedule.get("test_ids") or [])

        # If no specific test_ids, get all tests from the project
        if not test_ids and schedule.get("project_id"):
            project_tests = await self.supabase.request(
                f"/tests?project_id=eq.{schedule['project_id']}&is_active=eq.true&select=id"
            )
            if project_tests.get("data"):
                test_ids = {t["id"] for t in project_tests["data"]}

        return list(test_ids)

    async def _get_first_flaky_detection(self, test_id: str) -> str | None:
        """Get the timestamp when a test was first detected as flaky.

        Checks the test's settings for stored detection metadata.
        """
        result = await self.supabase.request(
            f"/tests?id=eq.{test_id}&select=settings"
        )

        if result.get("error") or not result.get("data"):
            return None

        settings = result["data"][0].get("settings") or {}
        flaky_detection = settings.get("flaky_detection", {})

        return flaky_detection.get("first_detected_at")


# ============================================================================
# Module-level convenience functions
# ============================================================================

_detector: FlakyDetector | None = None


def get_flaky_detector() -> FlakyDetector:
    """Get or create the global FlakyDetector instance."""
    global _detector
    if _detector is None:
        _detector = FlakyDetector()
    return _detector


async def calculate_flaky_score(test_id: str, window_size: int = 20) -> float:
    """Calculate flaky score for a test based on recent runs.

    Convenience function that uses the global detector instance.

    Args:
        test_id: The test ID to analyze
        window_size: Number of recent runs to consider

    Returns:
        Flaky score between 0.0 and 1.0
    """
    detector = get_flaky_detector()
    return await detector.calculate_flaky_score(test_id, window_size)


async def detect_flaky_tests(schedule_id: str) -> list[dict]:
    """Analyze all tests in a schedule and return flaky ones.

    Convenience function that uses the global detector instance.

    Args:
        schedule_id: The schedule ID to analyze

    Returns:
        List of flaky test dicts with test_id, flaky_score, and recommendation
    """
    detector = get_flaky_detector()
    return await detector.detect_flaky_tests(schedule_id)


async def update_flaky_status(test_id: str, is_flaky: bool, score: float) -> None:
    """Update the test's flaky status in the database.

    Convenience function that uses the global detector instance.

    Args:
        test_id: The test ID to update
        is_flaky: Whether the test is considered flaky
        score: The calculated flaky score
    """
    detector = get_flaky_detector()
    await detector.update_flaky_status(test_id, is_flaky, score)
