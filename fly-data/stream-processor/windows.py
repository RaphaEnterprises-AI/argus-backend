"""
Tumbling window implementation for stream processing.

Replaces 4 Flink SQL jobs with pure Python:
  1. test-metrics-aggregation (5 min)
  2. anomaly-latency (5 min)
  3. anomaly-failures (15 min)
  4. failure-cluster-update (15 min)
"""

import hashlib
import time
from dataclasses import dataclass, field


@dataclass
class MetricsWindowState:
    """State for Job 1: test-metrics-aggregation (5-min window)."""

    org_id: str
    project_id: str
    window_start: float  # epoch seconds
    window_end: float
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    durations: list[int] = field(default_factory=list)

    def add(self, status: str, duration_ms: int) -> None:
        self.total += 1
        self.durations.append(duration_ms)
        if status == "passed":
            self.passed += 1
        elif status == "failed":
            self.failed += 1
        elif status == "skipped":
            self.skipped += 1

    @property
    def avg_duration_ms(self) -> float:
        if not self.durations:
            return 0.0
        return sum(self.durations) / len(self.durations)

    @property
    def p95_duration_ms(self) -> int:
        if not self.durations:
            return 0
        s = sorted(self.durations)
        idx = min(int(len(s) * 0.95), len(s) - 1)
        return s[idx]

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def to_kafka_dict(self) -> dict:
        from datetime import datetime, timezone

        return {
            "window_start": datetime.fromtimestamp(self.window_start, tz=timezone.utc).isoformat(),
            "window_end": datetime.fromtimestamp(self.window_end, tz=timezone.utc).isoformat(),
            "org_id": self.org_id,
            "project_id": self.project_id,
            "total_tests": self.total,
            "passed_tests": self.passed,
            "failed_tests": self.failed,
            "skipped_tests": self.skipped,
            "pass_rate": round(self.pass_rate, 4),
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "p95_duration_ms": self.p95_duration_ms,
            "processing_time": datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class LatencyWindowState:
    """State for Job 2: anomaly-latency (5-min window per test_id)."""

    org_id: str
    project_id: str
    test_id: str
    test_name: str
    window_start: float
    window_end: float
    durations: list[int] = field(default_factory=list)

    def add(self, duration_ms: int, test_name: str) -> None:
        self.durations.append(duration_ms)
        if test_name:
            self.test_name = test_name

    @property
    def avg_duration_ms(self) -> float:
        if not self.durations:
            return 0.0
        return sum(self.durations) / len(self.durations)

    @property
    def p95_duration_ms(self) -> int:
        if not self.durations:
            return 0
        s = sorted(self.durations)
        idx = min(int(len(s) * 0.95), len(s) - 1)
        return s[idx]

    @property
    def max_duration_ms(self) -> int:
        return max(self.durations) if self.durations else 0

    @property
    def is_anomalous(self) -> bool:
        return self.avg_duration_ms > 5000 or self.p95_duration_ms > 15000

    @property
    def severity(self) -> str:
        avg = self.avg_duration_ms
        if avg > 30000:
            return "critical"
        if avg > 15000:
            return "high"
        if avg > 5000:
            return "medium"
        return "low"

    def to_kafka_dict(self) -> dict:
        from datetime import datetime, timezone

        epoch = int(self.window_start)
        return {
            "alert_id": f"lat-{self.org_id}-{self.test_id}-{epoch}",
            "org_id": self.org_id,
            "project_id": self.project_id,
            "test_id": self.test_id,
            "test_name": self.test_name,
            "window_start": datetime.fromtimestamp(self.window_start, tz=timezone.utc).isoformat(),
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "p95_duration_ms": self.p95_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "severity": self.severity,
            "detection_time": datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class FailureWindowState:
    """State for Job 3: anomaly-failures (15-min window per test_id)."""

    org_id: str
    project_id: str
    test_id: str
    test_name: str
    window_start: float
    window_end: float
    failure_count: int = 0
    last_error_message: str = ""
    last_error_type: str = ""
    last_selector: str = ""

    def add(self, error_message: str, error_type: str, selector: str, test_name: str) -> None:
        self.failure_count += 1
        if error_message:
            self.last_error_message = error_message
        if error_type:
            self.last_error_type = error_type
        if selector:
            self.last_selector = selector
        if test_name:
            self.test_name = test_name

    @property
    def should_heal(self) -> bool:
        return self.failure_count >= 3

    @property
    def priority(self) -> str:
        if self.failure_count >= 10:
            return "critical"
        if self.failure_count >= 5:
            return "high"
        if self.failure_count >= 3:
            return "medium"
        return "low"

    @property
    def healing_type(self) -> str:
        et = self.last_error_type or ""
        if "Selector" in et:
            return "selector_update"
        if "Timeout" in et:
            return "timeout_adjustment"
        return "ai_analysis"

    def to_kafka_dict(self) -> dict:
        from datetime import datetime, timezone

        epoch = int(self.window_start)
        return {
            "request_id": f"heal-{self.org_id}-{self.test_id}-{epoch}",
            "org_id": self.org_id,
            "project_id": self.project_id,
            "test_id": self.test_id,
            "test_name": self.test_name,
            "failure_count": self.failure_count,
            "window_minutes": 15,
            "last_error_message": self.last_error_message,
            "last_error_type": self.last_error_type,
            "last_selector": self.last_selector,
            "error_fingerprint": hashlib.md5(
                (self.last_error_message or "").encode()
            ).hexdigest(),
            "priority": self.priority,
            "healing_type": self.healing_type,
            "request_time": datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class FailureClusterState:
    """State for Job 4: failure-cluster-update (15-min window per test_id).

    Writes to both Supabase (flink_failure_patterns) and Kafka.
    """

    org_id: str
    project_id: str
    test_id: str
    window_start: float
    window_end: float
    failure_count: int = 0
    last_error_message: str = ""
    last_selector: str = ""

    def add(self, error_message: str, selector: str) -> None:
        self.failure_count += 1
        if error_message:
            self.last_error_message = error_message
        if selector:
            self.last_selector = selector

    @property
    def idempotency_key(self) -> str:
        epoch = int(self.window_start)
        return f"{epoch}:{self.org_id}:{self.test_id}"

    @property
    def error_fingerprint(self) -> str:
        msg = self.last_error_message or "unknown"
        return hashlib.md5(msg.encode()).hexdigest()

    @property
    def healing_priority(self) -> str:
        if self.failure_count >= 10:
            return "critical"
        if self.failure_count >= 5:
            return "high"
        if self.failure_count >= 3:
            return "medium"
        return "low"

    def to_supabase_dict(self, region: str) -> dict:
        from datetime import datetime, timezone

        return {
            "idempotency_key": self.idempotency_key,
            "org_id": self.org_id,
            "project_id": self.project_id,
            "test_id": self.test_id,
            "window_start": datetime.fromtimestamp(self.window_start, tz=timezone.utc).isoformat(),
            "window_end": datetime.fromtimestamp(self.window_end, tz=timezone.utc).isoformat(),
            "failure_count": self.failure_count,
            "last_error_message": self.last_error_message,
            "last_selector": self.last_selector,
            "error_fingerprint": self.error_fingerprint,
            "healing_priority": self.healing_priority,
            "healing_requested": self.failure_count >= 3,
            "processing_region": region,
        }

    def to_kafka_dict(self, region: str) -> dict:
        return self.to_supabase_dict(region)


class TumblingWindowManager:
    """Manages tumbling windows keyed by (window_epoch, *group_keys).

    When current time exceeds window_end, the window is flushed (returned
    for output) and removed from memory.
    """

    def __init__(self, window_seconds: int):
        self.window_seconds = window_seconds
        self._windows: dict[tuple, object] = {}

    def window_bounds(self, event_epoch: float) -> tuple[float, float]:
        """Return (start, end) for the window containing event_epoch."""
        start = (int(event_epoch) // self.window_seconds) * self.window_seconds
        return float(start), float(start + self.window_seconds)

    def get_or_create(self, key: tuple, factory) -> object:
        """Get existing window state or create with factory(window_start, window_end)."""
        if key not in self._windows:
            self._windows[key] = factory()
        return self._windows[key]

    def flush_expired(self, now: float | None = None) -> list:
        """Return and remove all windows whose window_end <= now."""
        if now is None:
            now = time.time()
        expired = []
        to_remove = []
        for key, state in self._windows.items():
            if state.window_end <= now:
                expired.append(state)
                to_remove.append(key)
        for key in to_remove:
            del self._windows[key]
        return expired

    @property
    def active_count(self) -> int:
        return len(self._windows)
