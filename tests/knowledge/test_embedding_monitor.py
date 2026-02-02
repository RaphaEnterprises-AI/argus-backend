"""Tests for EmbeddingMonitor - Drift Detection and Quality Monitoring.

Tests the embedding drift detection system that monitors:
- Baseline embedding storage and comparison
- Cosine similarity drift detection
- Quality metrics (dimension consistency, norm statistics)
- Alerting when drift exceeds thresholds

RAP-341: Embedding drift detection tests
"""

import numpy as np
import pytest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    return [np.random.randn(768).tolist() for _ in range(10)]


@pytest.fixture
def similar_embeddings(sample_embeddings):
    """Generate embeddings similar to sample with small perturbation."""
    np.random.seed(42)
    return [
        (np.array(emb) + np.random.randn(768) * 0.01).tolist()
        for emb in sample_embeddings
    ]


@pytest.fixture
def different_embeddings():
    """Generate completely different embeddings."""
    np.random.seed(123)  # Different seed
    return [np.random.randn(768).tolist() for _ in range(10)]


class TestDriftSeverity:
    """Tests for DriftSeverity enum."""

    def test_drift_severity_ordering(self):
        """Test DriftSeverity values are properly ordered."""
        from src.knowledge.embedding_monitor import DriftSeverity

        assert DriftSeverity.NONE < DriftSeverity.INFO
        assert DriftSeverity.INFO < DriftSeverity.WARNING
        assert DriftSeverity.WARNING < DriftSeverity.CRITICAL

    def test_drift_severity_values(self):
        """Test DriftSeverity has expected values."""
        from src.knowledge.embedding_monitor import DriftSeverity

        assert DriftSeverity.NONE == 0
        assert DriftSeverity.INFO == 1
        assert DriftSeverity.WARNING == 2
        assert DriftSeverity.CRITICAL == 3


class TestDriftThresholds:
    """Tests for DriftThresholds configuration."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        from src.knowledge.embedding_monitor import DriftThresholds

        thresholds = DriftThresholds()

        assert thresholds.info_threshold == 0.05
        assert thresholds.warning_threshold == 0.10
        assert thresholds.critical_threshold == 0.20

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        from src.knowledge.embedding_monitor import DriftThresholds

        thresholds = DriftThresholds(
            info_threshold=0.03,
            warning_threshold=0.08,
            critical_threshold=0.15,
        )

        assert thresholds.info_threshold == 0.03
        assert thresholds.warning_threshold == 0.08
        assert thresholds.critical_threshold == 0.15

    def test_get_severity_none(self):
        """Test get_severity returns NONE for small drift."""
        from src.knowledge.embedding_monitor import DriftSeverity, DriftThresholds

        thresholds = DriftThresholds()

        assert thresholds.get_severity(0.02) == DriftSeverity.NONE
        assert thresholds.get_severity(0.04) == DriftSeverity.NONE

    def test_get_severity_info(self):
        """Test get_severity returns INFO for moderate drift."""
        from src.knowledge.embedding_monitor import DriftSeverity, DriftThresholds

        thresholds = DriftThresholds()

        assert thresholds.get_severity(0.05) == DriftSeverity.INFO
        assert thresholds.get_severity(0.07) == DriftSeverity.INFO
        assert thresholds.get_severity(0.09) == DriftSeverity.INFO

    def test_get_severity_warning(self):
        """Test get_severity returns WARNING for significant drift."""
        from src.knowledge.embedding_monitor import DriftSeverity, DriftThresholds

        thresholds = DriftThresholds()

        assert thresholds.get_severity(0.10) == DriftSeverity.WARNING
        assert thresholds.get_severity(0.15) == DriftSeverity.WARNING
        assert thresholds.get_severity(0.19) == DriftSeverity.WARNING

    def test_get_severity_critical(self):
        """Test get_severity returns CRITICAL for severe drift."""
        from src.knowledge.embedding_monitor import DriftSeverity, DriftThresholds

        thresholds = DriftThresholds()

        assert thresholds.get_severity(0.20) == DriftSeverity.CRITICAL
        assert thresholds.get_severity(0.50) == DriftSeverity.CRITICAL
        assert thresholds.get_severity(1.0) == DriftSeverity.CRITICAL


class TestEmbeddingQualityMetrics:
    """Tests for EmbeddingQualityMetrics dataclass."""

    def test_metrics_to_dict(self):
        """Test EmbeddingQualityMetrics can be converted to dict."""
        from src.knowledge.embedding_monitor import EmbeddingQualityMetrics

        metrics = EmbeddingQualityMetrics(
            dimension=768,
            count=100,
            norm_mean=1.0,
            norm_stddev=0.05,
            norm_min=0.9,
            norm_max=1.1,
            has_nan=False,
            has_inf=False,
        )

        d = metrics.to_dict()

        assert d["dimension"] == 768
        assert d["count"] == 100
        assert d["norm_mean"] == 1.0
        assert d["has_nan"] is False
        assert "timestamp" in d

    def test_metrics_detects_problems(self):
        """Test metrics can represent problematic embeddings."""
        from src.knowledge.embedding_monitor import EmbeddingQualityMetrics

        metrics = EmbeddingQualityMetrics(
            dimension=768,
            count=10,
            norm_mean=0.5,
            norm_stddev=2.0,  # High variance
            norm_min=0.01,
            norm_max=5.0,
            has_nan=True,  # Problem!
            has_inf=False,
        )

        assert metrics.has_nan is True
        d = metrics.to_dict()
        assert d["has_nan"] is True


class TestDriftResult:
    """Tests for DriftResult dataclass."""

    def test_drift_result_creation(self):
        """Test DriftResult can be created with required fields."""
        from src.knowledge.embedding_monitor import DriftResult, DriftSeverity

        result = DriftResult(
            baseline_id="baseline_v1",
            mean_drift=0.08,
            max_drift=0.15,
            min_drift=0.02,
            stddev_drift=0.03,
            severity=DriftSeverity.INFO,
            baseline_count=100,
            current_count=100,
            samples_compared=100,
            check_duration_ms=25.5,
        )

        assert result.baseline_id == "baseline_v1"
        assert result.mean_drift == 0.08
        assert result.severity == DriftSeverity.INFO

    def test_drift_result_to_dict(self):
        """Test DriftResult to_dict includes severity name."""
        from src.knowledge.embedding_monitor import DriftResult, DriftSeverity

        result = DriftResult(
            baseline_id="test",
            mean_drift=0.12,
            max_drift=0.20,
            min_drift=0.05,
            stddev_drift=0.04,
            severity=DriftSeverity.WARNING,
            baseline_count=50,
            current_count=50,
            samples_compared=50,
            check_duration_ms=15.0,
        )

        d = result.to_dict()

        assert d["severity"] == "WARNING"  # String name, not int
        assert d["baseline_id"] == "test"
        assert d["mean_drift"] == 0.12


class TestEmbeddingBaseline:
    """Tests for EmbeddingBaseline dataclass."""

    def test_baseline_to_dict(self, sample_embeddings):
        """Test EmbeddingBaseline can be serialized to dict."""
        from src.knowledge.embedding_monitor import (
            EmbeddingBaseline,
            EmbeddingQualityMetrics,
        )

        now = datetime.now(UTC).isoformat()
        metrics = EmbeddingQualityMetrics(
            dimension=768,
            count=10,
            norm_mean=1.0,
            norm_stddev=0.05,
            norm_min=0.9,
            norm_max=1.1,
            has_nan=False,
            has_inf=False,
        )

        baseline = EmbeddingBaseline(
            baseline_id="test_v1",
            embeddings=sample_embeddings,
            quality_metrics=metrics,
            metadata={"model": "bge-large"},
            created_at=now,
            updated_at=now,
        )

        d = baseline.to_dict()

        assert d["baseline_id"] == "test_v1"
        assert len(d["embeddings"]) == 10
        assert d["metadata"]["model"] == "bge-large"

    def test_baseline_from_dict(self, sample_embeddings):
        """Test EmbeddingBaseline can be deserialized from dict."""
        from src.knowledge.embedding_monitor import EmbeddingBaseline

        now = datetime.now(UTC).isoformat()
        data = {
            "baseline_id": "test_v1",
            "embeddings": sample_embeddings,
            "quality_metrics": {
                "dimension": 768,
                "count": 10,
                "norm_mean": 1.0,
                "norm_stddev": 0.05,
                "norm_min": 0.9,
                "norm_max": 1.1,
                "has_nan": False,
                "has_inf": False,
                "timestamp": now,
            },
            "metadata": {"model": "bge-large"},
            "created_at": now,
            "updated_at": now,
        }

        baseline = EmbeddingBaseline.from_dict(data)

        assert baseline.baseline_id == "test_v1"
        assert baseline.quality_metrics.dimension == 768


class TestEmbeddingMonitorQualityMetrics:
    """Tests for EmbeddingMonitor quality metrics computation."""

    def test_compute_quality_metrics_empty(self):
        """Test compute_quality_metrics with empty list."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        metrics = monitor.compute_quality_metrics([])

        assert metrics.dimension == 0
        assert metrics.count == 0
        assert metrics.norm_mean == 0.0
        assert metrics.has_nan is False

    def test_compute_quality_metrics_normal(self, sample_embeddings):
        """Test compute_quality_metrics with normal embeddings."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        metrics = monitor.compute_quality_metrics(sample_embeddings)

        assert metrics.dimension == 768
        assert metrics.count == 10
        assert metrics.norm_mean > 0
        assert metrics.norm_stddev >= 0
        assert metrics.has_nan is False
        assert metrics.has_inf is False

    def test_compute_quality_metrics_detects_nan(self):
        """Test compute_quality_metrics detects NaN values."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        # Create embeddings with NaN
        embeddings = [[1.0, float("nan"), 3.0] for _ in range(5)]

        metrics = monitor.compute_quality_metrics(embeddings)

        assert metrics.has_nan is True

    def test_compute_quality_metrics_detects_inf(self):
        """Test compute_quality_metrics detects Inf values."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        # Create embeddings with Inf
        embeddings = [[1.0, float("inf"), 3.0] for _ in range(5)]

        metrics = monitor.compute_quality_metrics(embeddings)

        assert metrics.has_inf is True


class TestEmbeddingMonitorDriftComputation:
    """Tests for EmbeddingMonitor drift computation."""

    def test_compute_drift_empty_baseline(self, sample_embeddings):
        """Test compute_drift with empty baseline."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        mean, max_, min_, std, samples = monitor.compute_drift([], sample_embeddings)

        assert mean == 0.0
        assert samples == 0

    def test_compute_drift_empty_current(self, sample_embeddings):
        """Test compute_drift with empty current embeddings."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        mean, max_, min_, std, samples = monitor.compute_drift(sample_embeddings, [])

        assert mean == 0.0
        assert samples == 0

    def test_compute_drift_identical(self, sample_embeddings):
        """Test compute_drift with identical embeddings (zero drift)."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        mean, max_, min_, std, samples = monitor.compute_drift(
            sample_embeddings, sample_embeddings
        )

        assert mean == pytest.approx(0.0, abs=1e-10)
        assert samples == 10

    def test_compute_drift_similar(self, sample_embeddings, similar_embeddings):
        """Test compute_drift with similar embeddings (small drift)."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        mean, max_, min_, std, samples = monitor.compute_drift(
            sample_embeddings, similar_embeddings
        )

        # Small perturbation should result in small drift
        # Drift can be very small (near zero) due to floating point precision
        assert mean < 0.1
        assert mean >= 0.0 or abs(mean) < 1e-10  # Allow near-zero values
        assert samples == 10

    def test_compute_drift_different(self, sample_embeddings, different_embeddings):
        """Test compute_drift with different embeddings (large drift)."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        mean, max_, min_, std, samples = monitor.compute_drift(
            sample_embeddings, different_embeddings
        )

        # Random embeddings should have significant drift
        assert mean > 0.3  # Expect non-trivial drift
        assert samples == 10

    def test_compute_drift_dimension_mismatch(self):
        """Test compute_drift with dimension mismatch."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        baseline = [[1.0, 2.0, 3.0]]
        current = [[1.0, 2.0, 3.0, 4.0]]  # Different dimension

        mean, max_, min_, std, samples = monitor.compute_drift(baseline, current)

        # Dimension mismatch should return max drift
        assert mean == 1.0
        assert samples == 0

    def test_compute_drift_sample_size(self, sample_embeddings, similar_embeddings):
        """Test compute_drift with limited sample size."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        mean, max_, min_, std, samples = monitor.compute_drift(
            sample_embeddings, similar_embeddings, sample_size=5
        )

        assert samples == 5


class TestEmbeddingMonitorBaselineOperations:
    """Tests for EmbeddingMonitor baseline storage operations."""

    @pytest.mark.asyncio
    async def test_store_baseline(self, sample_embeddings):
        """Test store_baseline creates baseline with metrics."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        baseline = await monitor.store_baseline(
            baseline_id="test_v1",
            embeddings=sample_embeddings,
            metadata={"model": "test-model"},
        )

        assert baseline.baseline_id == "test_v1"
        assert baseline.quality_metrics.count == 10
        assert baseline.metadata["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_get_baseline(self, sample_embeddings):
        """Test get_baseline retrieves stored baseline."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        await monitor.store_baseline("test_v1", sample_embeddings)

        baseline = await monitor.get_baseline("test_v1")

        assert baseline is not None
        assert baseline.baseline_id == "test_v1"

    @pytest.mark.asyncio
    async def test_get_baseline_not_found(self):
        """Test get_baseline returns None for missing baseline."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        baseline = await monitor.get_baseline("nonexistent")

        assert baseline is None

    @pytest.mark.asyncio
    async def test_delete_baseline(self, sample_embeddings):
        """Test delete_baseline removes stored baseline."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        await monitor.store_baseline("test_v1", sample_embeddings)
        deleted = await monitor.delete_baseline("test_v1")
        baseline = await monitor.get_baseline("test_v1")

        assert deleted is True
        assert baseline is None

    @pytest.mark.asyncio
    async def test_delete_baseline_not_found(self):
        """Test delete_baseline returns False for missing baseline."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        deleted = await monitor.delete_baseline("nonexistent")

        assert deleted is False

    @pytest.mark.asyncio
    async def test_update_baseline(self, sample_embeddings, different_embeddings):
        """Test update_baseline replaces embeddings."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        await monitor.store_baseline("test_v1", sample_embeddings)
        updated = await monitor.update_baseline(
            "test_v1", different_embeddings, reason="model upgrade"
        )

        assert updated is not None
        assert updated.metadata.get("update_reason") == "model upgrade"

    @pytest.mark.asyncio
    async def test_list_baselines(self, sample_embeddings):
        """Test list_baselines returns all stored baselines."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        await monitor.store_baseline("v1", sample_embeddings)
        await monitor.store_baseline("v2", sample_embeddings)

        baselines = monitor.list_baselines()

        assert len(baselines) == 2
        ids = [b["baseline_id"] for b in baselines]
        assert "v1" in ids
        assert "v2" in ids


class TestEmbeddingMonitorDriftCheck:
    """Tests for EmbeddingMonitor drift checking."""

    @pytest.mark.asyncio
    async def test_check_drift_baseline_not_found(self, sample_embeddings):
        """Test check_drift with missing baseline."""
        from src.knowledge.embedding_monitor import DriftSeverity, EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        result = await monitor.check_drift("nonexistent", sample_embeddings)

        assert result.severity == DriftSeverity.NONE
        assert result.details.get("error") == "Baseline not found"

    @pytest.mark.asyncio
    async def test_check_drift_identical(self, sample_embeddings):
        """Test check_drift with identical embeddings."""
        from src.knowledge.embedding_monitor import DriftSeverity, EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        await monitor.store_baseline("test_v1", sample_embeddings)
        result = await monitor.check_drift("test_v1", sample_embeddings)

        assert result.severity == DriftSeverity.NONE
        assert result.mean_drift == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.asyncio
    async def test_check_drift_similar(self, sample_embeddings, similar_embeddings):
        """Test check_drift with similar embeddings (small drift)."""
        from src.knowledge.embedding_monitor import DriftSeverity, EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        await monitor.store_baseline("test_v1", sample_embeddings)
        result = await monitor.check_drift("test_v1", similar_embeddings)

        # Small drift should result in NONE or INFO severity
        assert result.severity in [DriftSeverity.NONE, DriftSeverity.INFO]
        assert result.mean_drift < 0.1

    @pytest.mark.asyncio
    async def test_check_drift_different(self, sample_embeddings, different_embeddings):
        """Test check_drift with different embeddings (large drift)."""
        from src.knowledge.embedding_monitor import DriftSeverity, EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        await monitor.store_baseline("test_v1", sample_embeddings)
        result = await monitor.check_drift("test_v1", different_embeddings)

        # Large drift should trigger WARNING or CRITICAL
        assert result.severity >= DriftSeverity.WARNING
        assert result.mean_drift > 0.1

    @pytest.mark.asyncio
    async def test_check_drift_triggers_alert_callback(
        self, sample_embeddings, different_embeddings
    ):
        """Test check_drift calls alert callback on high severity."""
        from src.knowledge.embedding_monitor import DriftSeverity, EmbeddingMonitor

        alert_called = False
        alert_result = None

        async def alert_callback(result):
            nonlocal alert_called, alert_result
            alert_called = True
            alert_result = result

        monitor = EmbeddingMonitor(
            org_id="test", project_id="test", alert_callback=alert_callback
        )

        await monitor.store_baseline("test_v1", sample_embeddings)
        await monitor.check_drift("test_v1", different_embeddings)

        assert alert_called is True
        assert alert_result is not None
        assert alert_result.severity >= DriftSeverity.WARNING

    @pytest.mark.asyncio
    async def test_check_drift_stores_history(self, sample_embeddings, similar_embeddings):
        """Test check_drift stores results in history."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        await monitor.store_baseline("test_v1", sample_embeddings)
        await monitor.check_drift("test_v1", similar_embeddings)
        await monitor.check_drift("test_v1", similar_embeddings)

        trend = monitor.get_drift_trend("test_v1")

        assert len(trend) == 2


class TestEmbeddingMonitorDriftTrend:
    """Tests for EmbeddingMonitor drift trend analysis."""

    @pytest.mark.asyncio
    async def test_get_drift_trend_empty(self):
        """Test get_drift_trend with no history."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        trend = monitor.get_drift_trend("test_v1")

        assert len(trend) == 0

    @pytest.mark.asyncio
    async def test_get_drift_trend_filters_by_baseline(
        self, sample_embeddings, similar_embeddings
    ):
        """Test get_drift_trend filters by baseline ID."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        await monitor.store_baseline("v1", sample_embeddings)
        await monitor.store_baseline("v2", sample_embeddings)

        await monitor.check_drift("v1", similar_embeddings)
        await monitor.check_drift("v2", similar_embeddings)
        await monitor.check_drift("v1", similar_embeddings)

        trend_v1 = monitor.get_drift_trend("v1")
        trend_v2 = monitor.get_drift_trend("v2")

        assert len(trend_v1) == 2
        assert len(trend_v2) == 1

    @pytest.mark.asyncio
    async def test_get_drift_trend_respects_limit(
        self, sample_embeddings, similar_embeddings
    ):
        """Test get_drift_trend respects last_n limit."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        await monitor.store_baseline("v1", sample_embeddings)

        for _ in range(5):
            await monitor.check_drift("v1", similar_embeddings)

        trend = monitor.get_drift_trend("v1", last_n=3)

        assert len(trend) == 3


class TestGlobalInstanceManagement:
    """Tests for global EmbeddingMonitor instance management."""

    def test_get_embedding_monitor_creates_instance(self):
        """Test get_embedding_monitor creates new instance."""
        from src.knowledge.embedding_monitor import (
            get_embedding_monitor,
            reset_all_embedding_monitors,
        )

        reset_all_embedding_monitors()

        monitor = get_embedding_monitor(org_id="org1", project_id="proj1")

        assert monitor is not None
        assert monitor.org_id == "org1"
        assert monitor.project_id == "proj1"

    def test_get_embedding_monitor_reuses_instance(self):
        """Test get_embedding_monitor reuses existing instance."""
        from src.knowledge.embedding_monitor import (
            get_embedding_monitor,
            reset_all_embedding_monitors,
        )

        reset_all_embedding_monitors()

        m1 = get_embedding_monitor(org_id="org1", project_id="proj1")
        m2 = get_embedding_monitor(org_id="org1", project_id="proj1")

        assert m1 is m2

    def test_get_embedding_monitor_different_tenants(self):
        """Test get_embedding_monitor creates different instances for different tenants."""
        from src.knowledge.embedding_monitor import (
            get_embedding_monitor,
            reset_all_embedding_monitors,
        )

        reset_all_embedding_monitors()

        m1 = get_embedding_monitor(org_id="org1", project_id="proj1")
        m2 = get_embedding_monitor(org_id="org2", project_id="proj2")

        assert m1 is not m2

    def test_reset_embedding_monitor(self):
        """Test reset_embedding_monitor clears specific instance."""
        from src.knowledge.embedding_monitor import (
            get_embedding_monitor,
            reset_embedding_monitor,
        )

        m1 = get_embedding_monitor(org_id="org1", project_id="proj1")
        reset_embedding_monitor("org1", "proj1")
        m2 = get_embedding_monitor(org_id="org1", project_id="proj1")

        assert m1 is not m2

    def test_reset_all_embedding_monitors(self):
        """Test reset_all_embedding_monitors clears all instances."""
        from src.knowledge.embedding_monitor import (
            get_embedding_monitor,
            reset_all_embedding_monitors,
        )

        m1 = get_embedding_monitor(org_id="org1", project_id="proj1")
        m2 = get_embedding_monitor(org_id="org2", project_id="proj2")

        reset_all_embedding_monitors()

        m3 = get_embedding_monitor(org_id="org1", project_id="proj1")
        m4 = get_embedding_monitor(org_id="org2", project_id="proj2")

        assert m1 is not m3
        assert m2 is not m4


class TestCosineSimiarity:
    """Tests for internal cosine similarity computation."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors is 1."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        vec = np.array([1.0, 2.0, 3.0])
        sim = monitor._cosine_similarity(vec, vec)

        assert sim == pytest.approx(1.0, abs=1e-10)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors is 0."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        sim = monitor._cosine_similarity(vec1, vec2)

        assert sim == pytest.approx(0.0, abs=1e-10)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors is -1."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        sim = monitor._cosine_similarity(vec1, vec2)

        assert sim == pytest.approx(-1.0, abs=1e-10)

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector returns 0."""
        from src.knowledge.embedding_monitor import EmbeddingMonitor

        monitor = EmbeddingMonitor(org_id="test", project_id="test")

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        sim = monitor._cosine_similarity(vec1, vec2)

        assert sim == 0.0
