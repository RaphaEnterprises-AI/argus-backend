"""
Flink Supabase Sink - Uber/Netflix Style Stateless Processing

This module provides a simple pattern for Flink jobs that:
1. Read from Redpanda/Kafka
2. Compute aggregations (stateless, short windows only)
3. Write directly to Supabase with idempotency keys

No Flink state management needed = easy multi-region!
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from supabase import Client, create_client

# Supabase client (reuse your existing connection)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
PROCESSING_REGION = os.getenv("PROCESSING_REGION", "ap-south-1")


def get_supabase_client() -> Client:
    """Get Supabase client for Flink jobs."""
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


@dataclass
class TestMetricsWindow:
    """Aggregated test metrics for a time window."""
    org_id: str
    project_id: str
    window_start: datetime
    window_end: datetime
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    total_duration_ms: int = 0
    durations: list[int] = None  # For percentile calculation

    def __post_init__(self):
        if self.durations is None:
            self.durations = []

    @property
    def idempotency_key(self) -> str:
        """Generate unique key for this window to prevent duplicates."""
        # Format: {window_epoch}:{org}:{project}
        epoch = int(self.window_start.timestamp())
        return f"{epoch}:{self.org_id}:{self.project_id}"

    @property
    def avg_duration_ms(self) -> float | None:
        if self.total_tests > 0:
            return self.total_duration_ms / self.total_tests
        return None

    @property
    def p95_duration_ms(self) -> int | None:
        if not self.durations:
            return None
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.95)
        return sorted_durations[min(idx, len(sorted_durations) - 1)]

    def add_test(self, status: str, duration_ms: int):
        """Add a test result to the window."""
        self.total_tests += 1
        self.total_duration_ms += duration_ms
        self.durations.append(duration_ms)

        if status == "passed":
            self.passed_tests += 1
        elif status == "failed":
            self.failed_tests += 1
        elif status == "skipped":
            self.skipped_tests += 1


class SupabaseSink:
    """
    Sink that writes Flink results to Supabase with idempotency.

    This is the Uber/Netflix pattern:
    - Flink jobs are STATELESS (no RocksDB state)
    - All state lives in Supabase (globally replicated)
    - Idempotency keys prevent duplicate writes from multiple regions
    """

    def __init__(self):
        self.client = get_supabase_client()
        self.region = PROCESSING_REGION

    def write_test_metrics(self, metrics: TestMetricsWindow) -> bool:
        """
        Write test metrics to Supabase with idempotency.

        If another region already wrote this window, the insert is ignored
        (ON CONFLICT DO NOTHING behavior via idempotency_key).
        """
        data = {
            "idempotency_key": metrics.idempotency_key,
            "org_id": metrics.org_id,
            "project_id": metrics.project_id,
            "window_start": metrics.window_start.isoformat(),
            "window_end": metrics.window_end.isoformat(),
            "window_size_minutes": 5,
            "total_tests": metrics.total_tests,
            "passed_tests": metrics.passed_tests,
            "failed_tests": metrics.failed_tests,
            "skipped_tests": metrics.skipped_tests,
            "total_duration_ms": metrics.total_duration_ms,
            "avg_duration_ms": metrics.avg_duration_ms,
            "p95_duration_ms": metrics.p95_duration_ms,
            "processing_region": self.region,
        }

        try:
            # Upsert with idempotency - if key exists, do nothing
            result = self.client.table("flink_test_metrics").upsert(
                data,
                on_conflict="idempotency_key",
                ignore_duplicates=True
            ).execute()
            return True
        except Exception as e:
            print(f"Error writing metrics: {e}")
            return False

    def write_failure_pattern(
        self,
        org_id: str,
        project_id: str,
        test_id: str,
        window_start: datetime,
        window_end: datetime,
        failure_count: int,
        last_error: str,
        last_selector: str = None,
    ) -> bool:
        """Write failure pattern for self-healing."""
        idempotency_key = f"{int(window_start.timestamp())}:{org_id}:{test_id}"

        # Determine priority based on failure count
        if failure_count >= 10:
            priority = "critical"
        elif failure_count >= 5:
            priority = "high"
        elif failure_count >= 3:
            priority = "medium"
        else:
            priority = "low"

        data = {
            "idempotency_key": idempotency_key,
            "org_id": org_id,
            "project_id": project_id,
            "test_id": test_id,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "failure_count": failure_count,
            "last_error_message": last_error,
            "last_selector": last_selector,
            "error_fingerprint": hashlib.md5(last_error.encode()).hexdigest()[:16],
            "healing_requested": failure_count >= 3,
            "healing_priority": priority,
            "processing_region": self.region,
        }

        try:
            self.client.table("flink_failure_patterns").upsert(
                data,
                on_conflict="idempotency_key",
                ignore_duplicates=True
            ).execute()
            return True
        except Exception as e:
            print(f"Error writing failure pattern: {e}")
            return False

    def increment_counter(self, org_id: str, counter_type: str, increment: int = 1) -> int:
        """
        Atomically increment a real-time counter.

        Uses Supabase RPC to call the increment_counter function
        which handles the atomic upsert.
        """
        try:
            result = self.client.rpc(
                "increment_counter",
                {
                    "p_org_id": org_id,
                    "p_counter_type": counter_type,
                    "p_increment": increment,
                    "p_region": self.region
                }
            ).execute()
            return result.data
        except Exception as e:
            print(f"Error incrementing counter: {e}")
            return -1


# =============================================================================
# EXAMPLE: Simple Flink Job Using This Pattern
# =============================================================================

def process_test_events_window(events: list[dict[str, Any]]) -> None:
    """
    Example Flink window function that processes test events.

    In a real Flink job, this would be called by a tumbling window operator.
    The key insight: NO FLINK STATE NEEDED - we just compute and write to Supabase.
    """
    if not events:
        return

    # Group events by org_id + project_id
    windows: dict[str, TestMetricsWindow] = {}

    # Determine window boundaries (5-minute tumbling window)
    first_event_time = datetime.fromisoformat(events[0]["timestamp"])
    window_start = first_event_time.replace(
        minute=(first_event_time.minute // 5) * 5,
        second=0,
        microsecond=0
    )
    window_end = window_start + timedelta(minutes=5)

    # Aggregate events
    for event in events:
        key = f"{event['org_id']}:{event['project_id']}"

        if key not in windows:
            windows[key] = TestMetricsWindow(
                org_id=event["org_id"],
                project_id=event["project_id"],
                window_start=window_start,
                window_end=window_end,
            )

        windows[key].add_test(
            status=event.get("status", "unknown"),
            duration_ms=event.get("duration_ms", 0)
        )

    # Write to Supabase (idempotent!)
    sink = SupabaseSink()
    for window in windows.values():
        sink.write_test_metrics(window)

        # Also update real-time counters
        sink.increment_counter(window.org_id, "tests_today", window.total_tests)
        sink.increment_counter(window.org_id, "failures_today", window.failed_tests)


# =============================================================================
# WHY THIS IS MULTI-REGION READY
# =============================================================================
#
# 1. No Flink state to replicate - all state is in Supabase
# 2. Idempotency keys prevent duplicate processing
# 3. Both regions can process the same events - only one write succeeds
# 4. Dashboard sees results instantly via Supabase real-time
# 5. If one region fails, the other continues automatically
#
# This is EXACTLY what Uber does for surge pricing:
# - Flink jobs compute prices
# - Write to CockroachDB (their global DB)
# - Multiple regions run the same job
# - Idempotency prevents duplicates
# =============================================================================
