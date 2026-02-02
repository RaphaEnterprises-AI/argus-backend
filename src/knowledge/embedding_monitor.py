"""
Embedding Drift Detection and Quality Monitoring.

RAP-341: Monitors embedding quality and detects drift over time.

This module provides:
- Baseline embedding storage and comparison
- Cosine similarity drift detection between embedding batches
- Quality metrics (dimension consistency, norm statistics)
- Alerting when drift exceeds configurable thresholds

Usage:
    ```python
    from src.knowledge.embedding_monitor import (
        EmbeddingMonitor,
        get_embedding_monitor,
        DriftSeverity,
    )

    monitor = get_embedding_monitor(org_id="org123", project_id="proj456")

    # Store baseline embeddings
    await monitor.store_baseline(
        baseline_id="failure_patterns_v1",
        embeddings=embeddings_list,
        metadata={"model": "bge-large-en-v1.5", "source": "failure_patterns"},
    )

    # Check drift against baseline
    drift_result = await monitor.check_drift(
        baseline_id="failure_patterns_v1",
        current_embeddings=new_embeddings,
    )

    if drift_result.severity >= DriftSeverity.WARNING:
        print(f"Drift detected: {drift_result.mean_drift:.3f}")
    ```
"""

import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
from typing import Any

import numpy as np
import structlog
from prometheus_client import Counter, Gauge, Histogram

from src.utils import safe_datetime

logger = structlog.get_logger(__name__)


# =============================================================================
# Prometheus Metrics for Embedding Drift Monitoring
# =============================================================================

EMBEDDING_DRIFT_VALUE = Gauge(
    "embedding_drift_value",
    "Current embedding drift value (1 - cosine similarity)",
    ["org_id", "project_id", "baseline_id"],
)

EMBEDDING_DRIFT_CHECKS_TOTAL = Counter(
    "embedding_drift_checks_total",
    "Total number of drift checks performed",
    ["org_id", "project_id", "severity"],
)

EMBEDDING_DRIFT_ALERTS_TOTAL = Counter(
    "embedding_drift_alerts_total",
    "Total drift alerts triggered",
    ["org_id", "project_id", "severity"],
)

EMBEDDING_QUALITY_DIMENSION = Gauge(
    "embedding_quality_dimension",
    "Embedding dimension being used",
    ["org_id", "project_id", "baseline_id"],
)

EMBEDDING_QUALITY_NORM_MEAN = Gauge(
    "embedding_quality_norm_mean",
    "Mean L2 norm of embeddings",
    ["org_id", "project_id", "baseline_id"],
)

EMBEDDING_QUALITY_NORM_STDDEV = Gauge(
    "embedding_quality_norm_stddev",
    "Standard deviation of L2 norms",
    ["org_id", "project_id", "baseline_id"],
)

EMBEDDING_CHECK_DURATION = Histogram(
    "embedding_check_duration_seconds",
    "Time to perform drift check",
    ["org_id", "project_id"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)


class DriftSeverity(IntEnum):
    """Severity levels for embedding drift."""

    NONE = 0       # No significant drift (< 0.05)
    INFO = 1       # Minor drift (0.05 - 0.10)
    WARNING = 2    # Moderate drift (0.10 - 0.20)
    CRITICAL = 3   # Severe drift (> 0.20)


@dataclass
class EmbeddingQualityMetrics:
    """Quality metrics for a set of embeddings."""

    dimension: int
    count: int
    norm_mean: float
    norm_stddev: float
    norm_min: float
    norm_max: float
    has_nan: bool
    has_inf: bool
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension,
            "count": self.count,
            "norm_mean": self.norm_mean,
            "norm_stddev": self.norm_stddev,
            "norm_min": self.norm_min,
            "norm_max": self.norm_max,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "timestamp": self.timestamp,
        }


@dataclass
class DriftResult:
    """Result of a drift detection check."""

    baseline_id: str
    mean_drift: float         # Average drift (1 - cosine_similarity)
    max_drift: float          # Maximum drift observed
    min_drift: float          # Minimum drift observed
    stddev_drift: float       # Stddev of drift values
    severity: DriftSeverity
    baseline_count: int       # Number of baseline embeddings
    current_count: int        # Number of current embeddings
    samples_compared: int     # Actual number compared (min of both)
    check_duration_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_id": self.baseline_id,
            "mean_drift": self.mean_drift,
            "max_drift": self.max_drift,
            "min_drift": self.min_drift,
            "stddev_drift": self.stddev_drift,
            "severity": self.severity.name,
            "baseline_count": self.baseline_count,
            "current_count": self.current_count,
            "samples_compared": self.samples_compared,
            "check_duration_ms": self.check_duration_ms,
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class EmbeddingBaseline:
    """Stored baseline for drift comparison."""

    baseline_id: str
    embeddings: list[list[float]]
    quality_metrics: EmbeddingQualityMetrics
    metadata: dict[str, Any]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "baseline_id": self.baseline_id,
            "embeddings": self.embeddings,
            "quality_metrics": self.quality_metrics.to_dict(),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingBaseline":
        """Create from dictionary."""
        return cls(
            baseline_id=data["baseline_id"],
            embeddings=data["embeddings"],
            quality_metrics=EmbeddingQualityMetrics(**data["quality_metrics"]),
            metadata=data.get("metadata", {}),
            created_at=safe_datetime(data.get("created_at")),
            updated_at=safe_datetime(data.get("updated_at")),
        )


@dataclass
class DriftThresholds:
    """Configurable thresholds for drift detection."""

    info_threshold: float = 0.05      # Drift > 0.05 triggers INFO
    warning_threshold: float = 0.10   # Drift > 0.10 triggers WARNING
    critical_threshold: float = 0.20  # Drift > 0.20 triggers CRITICAL

    def get_severity(self, drift: float) -> DriftSeverity:
        """Determine severity based on drift value."""
        if drift >= self.critical_threshold:
            return DriftSeverity.CRITICAL
        elif drift >= self.warning_threshold:
            return DriftSeverity.WARNING
        elif drift >= self.info_threshold:
            return DriftSeverity.INFO
        return DriftSeverity.NONE


class EmbeddingMonitor:
    """
    Monitors embedding quality and detects drift over time.

    The monitor stores baseline embeddings and compares new embeddings
    against them to detect semantic drift. This is critical for:

    - Detecting model changes that affect embedding quality
    - Monitoring for data distribution shifts
    - Ensuring consistent semantic search performance
    - Alerting when re-indexing may be needed

    Drift is computed as (1 - cosine_similarity) between corresponding
    embeddings. A drift of 0 means identical, while 2 is maximal drift.
    """

    def __init__(
        self,
        org_id: str,
        project_id: str,
        thresholds: DriftThresholds | None = None,
        alert_callback: Any | None = None,
    ):
        """
        Initialize the embedding monitor.

        Args:
            org_id: Organization ID for multi-tenant isolation
            project_id: Project ID for multi-tenant isolation
            thresholds: Custom drift thresholds (default: DriftThresholds())
            alert_callback: Optional async callback for alerts:
                            async def callback(drift_result: DriftResult) -> None
        """
        self.org_id = org_id
        self.project_id = project_id
        self.thresholds = thresholds or DriftThresholds()
        self.alert_callback = alert_callback
        self._log = logger.bind(
            component="embedding_monitor",
            org_id=org_id,
            project_id=project_id,
        )

        # In-memory baseline storage (can be extended to use Cognee/DB)
        self._baselines: dict[str, EmbeddingBaseline] = {}

        # History of drift checks for trend analysis
        self._drift_history: list[DriftResult] = []

    def compute_quality_metrics(
        self,
        embeddings: list[list[float]],
    ) -> EmbeddingQualityMetrics:
        """
        Compute quality metrics for a set of embeddings.

        Args:
            embeddings: List of embedding vectors

        Returns:
            EmbeddingQualityMetrics with dimension, norm stats, etc.
        """
        if not embeddings:
            return EmbeddingQualityMetrics(
                dimension=0,
                count=0,
                norm_mean=0.0,
                norm_stddev=0.0,
                norm_min=0.0,
                norm_max=0.0,
                has_nan=False,
                has_inf=False,
            )

        arr = np.array(embeddings)

        # Check for invalid values
        has_nan = bool(np.isnan(arr).any())
        has_inf = bool(np.isinf(arr).any())

        # Compute L2 norms
        norms = np.linalg.norm(arr, axis=1)

        return EmbeddingQualityMetrics(
            dimension=arr.shape[1] if len(arr.shape) > 1 else 0,
            count=len(embeddings),
            norm_mean=float(np.mean(norms)),
            norm_stddev=float(np.std(norms)),
            norm_min=float(np.min(norms)),
            norm_max=float(np.max(norms)),
            has_nan=has_nan,
            has_inf=has_inf,
        )

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def compute_drift(
        self,
        baseline_embeddings: list[list[float]],
        current_embeddings: list[list[float]],
        sample_size: int | None = None,
    ) -> tuple[float, float, float, float, int]:
        """
        Compute drift between baseline and current embeddings.

        Uses centroid comparison for overall drift, plus sample comparison
        for distribution-level analysis.

        Args:
            baseline_embeddings: Baseline embedding vectors
            current_embeddings: Current embedding vectors
            sample_size: Max samples to compare (None = all)

        Returns:
            Tuple of (mean_drift, max_drift, min_drift, stddev_drift, samples_compared)
        """
        if not baseline_embeddings or not current_embeddings:
            return (0.0, 0.0, 0.0, 0.0, 0)

        baseline_arr = np.array(baseline_embeddings)
        current_arr = np.array(current_embeddings)

        # Validate dimensions match
        if baseline_arr.shape[1] != current_arr.shape[1]:
            self._log.warning(
                "Dimension mismatch in drift check",
                baseline_dim=baseline_arr.shape[1],
                current_dim=current_arr.shape[1],
            )
            return (1.0, 1.0, 1.0, 0.0, 0)

        # Compute centroid drift (overall semantic shift)
        baseline_centroid = np.mean(baseline_arr, axis=0)
        current_centroid = np.mean(current_arr, axis=0)
        centroid_similarity = self._cosine_similarity(baseline_centroid, current_centroid)
        centroid_drift = 1.0 - centroid_similarity

        # Sample-wise comparison for distribution analysis
        n_baseline = len(baseline_embeddings)
        n_current = len(current_embeddings)
        n_compare = min(n_baseline, n_current)

        if sample_size and sample_size < n_compare:
            n_compare = sample_size

        drifts = []
        for i in range(n_compare):
            similarity = self._cosine_similarity(baseline_arr[i], current_arr[i])
            drifts.append(1.0 - similarity)

        if not drifts:
            return (centroid_drift, centroid_drift, centroid_drift, 0.0, 0)

        mean_drift = statistics.mean(drifts)
        max_drift = max(drifts)
        min_drift = min(drifts)
        stddev_drift = statistics.stdev(drifts) if len(drifts) > 1 else 0.0

        return (mean_drift, max_drift, min_drift, stddev_drift, n_compare)

    async def store_baseline(
        self,
        baseline_id: str,
        embeddings: list[list[float]],
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingBaseline:
        """
        Store embeddings as a baseline for future drift comparisons.

        Args:
            baseline_id: Unique identifier for this baseline
            embeddings: List of embedding vectors
            metadata: Optional metadata (model name, source, etc.)

        Returns:
            EmbeddingBaseline object
        """
        quality_metrics = self.compute_quality_metrics(embeddings)
        now = datetime.now(UTC).isoformat()

        baseline = EmbeddingBaseline(
            baseline_id=baseline_id,
            embeddings=embeddings,
            quality_metrics=quality_metrics,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

        self._baselines[baseline_id] = baseline

        # Update Prometheus metrics
        EMBEDDING_QUALITY_DIMENSION.labels(
            org_id=self.org_id,
            project_id=self.project_id,
            baseline_id=baseline_id,
        ).set(quality_metrics.dimension)

        EMBEDDING_QUALITY_NORM_MEAN.labels(
            org_id=self.org_id,
            project_id=self.project_id,
            baseline_id=baseline_id,
        ).set(quality_metrics.norm_mean)

        EMBEDDING_QUALITY_NORM_STDDEV.labels(
            org_id=self.org_id,
            project_id=self.project_id,
            baseline_id=baseline_id,
        ).set(quality_metrics.norm_stddev)

        self._log.info(
            "Stored embedding baseline",
            baseline_id=baseline_id,
            count=quality_metrics.count,
            dimension=quality_metrics.dimension,
            norm_mean=quality_metrics.norm_mean,
        )

        return baseline

    async def get_baseline(
        self,
        baseline_id: str,
    ) -> EmbeddingBaseline | None:
        """
        Retrieve a stored baseline.

        Args:
            baseline_id: Baseline identifier

        Returns:
            EmbeddingBaseline or None if not found
        """
        return self._baselines.get(baseline_id)

    async def check_drift(
        self,
        baseline_id: str,
        current_embeddings: list[list[float]],
        sample_size: int | None = 100,
    ) -> DriftResult:
        """
        Check drift between stored baseline and current embeddings.

        Args:
            baseline_id: ID of baseline to compare against
            current_embeddings: Current embedding vectors
            sample_size: Max samples to compare (None = all)

        Returns:
            DriftResult with drift metrics and severity
        """
        start_time = time.perf_counter()

        baseline = await self.get_baseline(baseline_id)
        if baseline is None:
            self._log.warning("Baseline not found for drift check", baseline_id=baseline_id)
            return DriftResult(
                baseline_id=baseline_id,
                mean_drift=0.0,
                max_drift=0.0,
                min_drift=0.0,
                stddev_drift=0.0,
                severity=DriftSeverity.NONE,
                baseline_count=0,
                current_count=len(current_embeddings),
                samples_compared=0,
                check_duration_ms=0.0,
                details={"error": "Baseline not found"},
            )

        # Compute drift
        mean_drift, max_drift, min_drift, stddev_drift, samples = self.compute_drift(
            baseline.embeddings,
            current_embeddings,
            sample_size,
        )

        # Determine severity
        severity = self.thresholds.get_severity(mean_drift)

        duration_ms = (time.perf_counter() - start_time) * 1000

        result = DriftResult(
            baseline_id=baseline_id,
            mean_drift=mean_drift,
            max_drift=max_drift,
            min_drift=min_drift,
            stddev_drift=stddev_drift,
            severity=severity,
            baseline_count=len(baseline.embeddings),
            current_count=len(current_embeddings),
            samples_compared=samples,
            check_duration_ms=duration_ms,
            details={
                "baseline_model": baseline.metadata.get("model"),
                "baseline_created_at": baseline.created_at,
                "current_quality": self.compute_quality_metrics(current_embeddings).to_dict(),
            },
        )

        # Update Prometheus metrics
        EMBEDDING_DRIFT_VALUE.labels(
            org_id=self.org_id,
            project_id=self.project_id,
            baseline_id=baseline_id,
        ).set(mean_drift)

        EMBEDDING_DRIFT_CHECKS_TOTAL.labels(
            org_id=self.org_id,
            project_id=self.project_id,
            severity=severity.name,
        ).inc()

        EMBEDDING_CHECK_DURATION.labels(
            org_id=self.org_id,
            project_id=self.project_id,
        ).observe(duration_ms / 1000)

        # Store in history
        self._drift_history.append(result)
        if len(self._drift_history) > 100:
            self._drift_history = self._drift_history[-100:]

        # Log result
        log_method = self._log.info
        if severity >= DriftSeverity.CRITICAL:
            log_method = self._log.error
        elif severity >= DriftSeverity.WARNING:
            log_method = self._log.warning

        log_method(
            "Drift check completed",
            baseline_id=baseline_id,
            mean_drift=round(mean_drift, 4),
            max_drift=round(max_drift, 4),
            severity=severity.name,
            samples_compared=samples,
            duration_ms=round(duration_ms, 2),
        )

        # Trigger alert if severity is WARNING or higher
        if severity >= DriftSeverity.WARNING:
            EMBEDDING_DRIFT_ALERTS_TOTAL.labels(
                org_id=self.org_id,
                project_id=self.project_id,
                severity=severity.name,
            ).inc()

            if self.alert_callback:
                try:
                    await self.alert_callback(result)
                except Exception as e:
                    self._log.error("Alert callback failed", error=str(e))

        return result

    async def update_baseline(
        self,
        baseline_id: str,
        embeddings: list[list[float]],
        reason: str | None = None,
    ) -> EmbeddingBaseline | None:
        """
        Update an existing baseline with new embeddings.

        Use this when drift is expected (model upgrade, data refresh).

        Args:
            baseline_id: Baseline to update
            embeddings: New embedding vectors
            reason: Optional reason for update

        Returns:
            Updated EmbeddingBaseline or None if not found
        """
        existing = await self.get_baseline(baseline_id)
        if existing is None:
            self._log.warning("Cannot update non-existent baseline", baseline_id=baseline_id)
            return None

        quality_metrics = self.compute_quality_metrics(embeddings)
        now = datetime.now(UTC).isoformat()

        updated = EmbeddingBaseline(
            baseline_id=baseline_id,
            embeddings=embeddings,
            quality_metrics=quality_metrics,
            metadata={
                **existing.metadata,
                "previous_updated_at": existing.updated_at,
                "update_reason": reason,
            },
            created_at=existing.created_at,
            updated_at=now,
        )

        self._baselines[baseline_id] = updated

        self._log.info(
            "Updated embedding baseline",
            baseline_id=baseline_id,
            reason=reason,
            new_count=quality_metrics.count,
        )

        return updated

    async def delete_baseline(self, baseline_id: str) -> bool:
        """
        Delete a stored baseline.

        Args:
            baseline_id: Baseline to delete

        Returns:
            True if deleted, False if not found
        """
        if baseline_id in self._baselines:
            del self._baselines[baseline_id]
            self._log.info("Deleted embedding baseline", baseline_id=baseline_id)
            return True
        return False

    def get_drift_trend(
        self,
        baseline_id: str,
        last_n: int = 10,
    ) -> list[DriftResult]:
        """
        Get recent drift history for trend analysis.

        Args:
            baseline_id: Baseline to get history for
            last_n: Number of recent results to return

        Returns:
            List of recent DriftResult objects
        """
        filtered = [r for r in self._drift_history if r.baseline_id == baseline_id]
        return filtered[-last_n:]

    def list_baselines(self) -> list[dict[str, Any]]:
        """
        List all stored baselines with summary info.

        Returns:
            List of baseline summaries
        """
        summaries = []
        for baseline_id, baseline in self._baselines.items():
            summaries.append({
                "baseline_id": baseline_id,
                "count": baseline.quality_metrics.count,
                "dimension": baseline.quality_metrics.dimension,
                "created_at": baseline.created_at,
                "updated_at": baseline.updated_at,
                "metadata": baseline.metadata,
            })
        return summaries


# =============================================================================
# Global Instance Management
# =============================================================================

_embedding_monitors: dict[str, EmbeddingMonitor] = {}


def get_embedding_monitor(
    org_id: str = "default",
    project_id: str = "default",
    thresholds: DriftThresholds | None = None,
    alert_callback: Any | None = None,
) -> EmbeddingMonitor:
    """
    Get or create an EmbeddingMonitor for the given org/project.

    Args:
        org_id: Organization ID
        project_id: Project ID
        thresholds: Optional custom thresholds
        alert_callback: Optional alert callback

    Returns:
        EmbeddingMonitor instance
    """
    key = f"{org_id}:{project_id}"

    if key not in _embedding_monitors:
        _embedding_monitors[key] = EmbeddingMonitor(
            org_id=org_id,
            project_id=project_id,
            thresholds=thresholds,
            alert_callback=alert_callback,
        )

    return _embedding_monitors[key]


def reset_embedding_monitor(org_id: str, project_id: str) -> None:
    """Reset the embedding monitor for a given org/project."""
    key = f"{org_id}:{project_id}"
    if key in _embedding_monitors:
        del _embedding_monitors[key]


def reset_all_embedding_monitors() -> None:
    """Reset all embedding monitors."""
    _embedding_monitors.clear()
