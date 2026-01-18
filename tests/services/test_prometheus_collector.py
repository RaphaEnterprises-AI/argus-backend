"""Tests for the Prometheus Metrics Collector service."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.services.prometheus_collector import (
    BrowserNodeMetrics,
    InfrastructureSnapshot,
    MetricSeries,
    MetricValue,
    PrometheusCollector,
    ResourceUtilization,
    SeleniumMetrics,
    create_prometheus_collector,
)


class TestDataClasses:
    """Tests for dataclass definitions."""

    def test_metric_value_creation(self):
        """Test MetricValue dataclass creation."""
        now = datetime.now()
        metric = MetricValue(
            value=42.5,
            timestamp=now,
            labels={"job": "test", "instance": "localhost:9090"}
        )

        assert metric.value == 42.5
        assert metric.timestamp == now
        assert metric.labels["job"] == "test"
        assert metric.labels["instance"] == "localhost:9090"

    def test_metric_series_creation(self):
        """Test MetricSeries dataclass creation."""
        now = datetime.now()
        values = [
            (now, 1.0),
            (now + timedelta(minutes=1), 2.0),
            (now + timedelta(minutes=2), 3.0),
        ]
        series = MetricSeries(
            metric_name="test_metric",
            values=values,
            labels={"job": "test"}
        )

        assert series.metric_name == "test_metric"
        assert len(series.values) == 3
        assert series.values[0][1] == 1.0
        assert series.labels["job"] == "test"

    def test_resource_utilization_creation(self):
        """Test ResourceUtilization dataclass creation."""
        now = datetime.now()
        util = ResourceUtilization(
            cpu_usage_percent=75.5,
            cpu_request_percent=80.0,
            memory_usage_bytes=1073741824,  # 1GB
            memory_usage_percent=50.0,
            timestamp=now
        )

        assert util.cpu_usage_percent == 75.5
        assert util.cpu_request_percent == 80.0
        assert util.memory_usage_bytes == 1073741824
        assert util.memory_usage_percent == 50.0

    def test_selenium_metrics_creation(self):
        """Test SeleniumMetrics dataclass creation."""
        now = datetime.now()
        metrics = SeleniumMetrics(
            sessions_queued=5,
            sessions_active=10,
            sessions_total=15,
            nodes_available=8,
            nodes_total=10,
            avg_session_duration_seconds=120.5,
            queue_wait_time_seconds=3.2,
            timestamp=now
        )

        assert metrics.sessions_queued == 5
        assert metrics.sessions_active == 10
        assert metrics.sessions_total == 15
        assert metrics.nodes_available == 8
        assert metrics.avg_session_duration_seconds == 120.5

    def test_browser_node_metrics_creation(self):
        """Test BrowserNodeMetrics dataclass creation."""
        now = datetime.now()
        cpu_util = ResourceUtilization(50.0, 60.0, 0, 0.0, now)
        mem_util = ResourceUtilization(0.0, 0.0, 512000000, 25.0, now)

        metrics = BrowserNodeMetrics(
            browser_type="chrome",
            replicas_current=5,
            replicas_desired=7,
            replicas_min=2,
            replicas_max=15,
            cpu_utilization=cpu_util,
            memory_utilization=mem_util,
            sessions_active=4,
            timestamp=now
        )

        assert metrics.browser_type == "chrome"
        assert metrics.replicas_current == 5
        assert metrics.replicas_desired == 7
        assert metrics.replicas_min == 2
        assert metrics.replicas_max == 15

    def test_infrastructure_snapshot_creation(self):
        """Test InfrastructureSnapshot dataclass creation."""
        now = datetime.now()
        selenium = SeleniumMetrics(
            sessions_queued=0, sessions_active=5, sessions_total=5,
            nodes_available=3, nodes_total=3, avg_session_duration_seconds=60.0,
            queue_wait_time_seconds=0.0, timestamp=now
        )

        cpu_util = ResourceUtilization(50.0, 50.0, 0, 0.0, now)
        mem_util = ResourceUtilization(0.0, 0.0, 0, 50.0, now)

        chrome = BrowserNodeMetrics(
            browser_type="chrome", replicas_current=3, replicas_desired=3,
            replicas_min=2, replicas_max=15, cpu_utilization=cpu_util,
            memory_utilization=mem_util, sessions_active=3, timestamp=now
        )
        firefox = BrowserNodeMetrics(
            browser_type="firefox", replicas_current=1, replicas_desired=1,
            replicas_min=1, replicas_max=8, cpu_utilization=cpu_util,
            memory_utilization=mem_util, sessions_active=1, timestamp=now
        )
        edge = BrowserNodeMetrics(
            browser_type="edge", replicas_current=1, replicas_desired=1,
            replicas_min=1, replicas_max=8, cpu_utilization=cpu_util,
            memory_utilization=mem_util, sessions_active=1, timestamp=now
        )

        snapshot = InfrastructureSnapshot(
            selenium=selenium,
            chrome_nodes=chrome,
            firefox_nodes=firefox,
            edge_nodes=edge,
            total_pods=10,
            total_nodes=3,
            cluster_cpu_utilization=45.0,
            cluster_memory_utilization=55.0,
            timestamp=now
        )

        assert snapshot.selenium.sessions_active == 5
        assert snapshot.chrome_nodes.browser_type == "chrome"
        assert snapshot.total_pods == 10
        assert snapshot.cluster_cpu_utilization == 45.0


class TestPrometheusCollectorInit:
    """Tests for PrometheusCollector initialization."""

    def test_init_with_url(self):
        """Test initialization with explicit URL."""
        collector = PrometheusCollector(
            prometheus_url="http://prometheus.example.com:9090",
            timeout=60.0
        )

        assert collector.prometheus_url == "http://prometheus.example.com:9090"
        assert collector.timeout == 60.0
        assert collector._client is None

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from URL."""
        collector = PrometheusCollector(
            prometheus_url="http://prometheus.example.com:9090/"
        )

        assert collector.prometheus_url == "http://prometheus.example.com:9090"

    def test_init_default_timeout(self):
        """Test default timeout value."""
        collector = PrometheusCollector(prometheus_url="http://localhost:9090")

        assert collector.timeout == 30.0


class TestPrometheusCollectorClient:
    """Tests for HTTP client management."""

    @pytest.mark.asyncio
    async def test_get_client_creates_new(self):
        """Test that _get_client creates a new client if none exists."""
        collector = PrometheusCollector(prometheus_url="http://localhost:9090")

        client = await collector._get_client()

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        assert collector._client is client

        await collector.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self):
        """Test that _get_client reuses existing client."""
        collector = PrometheusCollector(prometheus_url="http://localhost:9090")

        client1 = await collector._get_client()
        client2 = await collector._get_client()

        assert client1 is client2

        await collector.close()

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing the HTTP client."""
        collector = PrometheusCollector(prometheus_url="http://localhost:9090")

        await collector._get_client()
        assert collector._client is not None

        await collector.close()
        # After close, _client should be closed but not None (it's still the reference)

    @pytest.mark.asyncio
    async def test_get_client_recreates_after_close(self):
        """Test that a new client is created after closing."""
        collector = PrometheusCollector(prometheus_url="http://localhost:9090")

        client1 = await collector._get_client()
        await collector.close()

        client2 = await collector._get_client()

        assert client1 is not client2

        await collector.close()


class TestPrometheusCollectorQuery:
    """Tests for instant query method."""

    @pytest.mark.asyncio
    async def test_query_success(self, mock_httpx_client):
        """Test successful instant query."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {"__name__": "up", "job": "prometheus"},
                        "value": [1609459200.0, "1"]
                    }
                ]
            }
        }
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        results = await collector.query("up")

        assert len(results) == 1
        assert results[0].value == 1.0
        assert results[0].labels["job"] == "prometheus"

    @pytest.mark.asyncio
    async def test_query_empty_results(self, mock_httpx_client):
        """Test query returning empty results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"result": []}
        }
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        results = await collector.query("nonexistent_metric")

        assert results == []

    @pytest.mark.asyncio
    async def test_query_failure_status(self, mock_httpx_client):
        """Test query with failed status in response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "status": "error",
            "error": "invalid query"
        }
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        results = await collector.query("invalid{}")

        assert results == []

    @pytest.mark.asyncio
    async def test_query_http_error(self, mock_httpx_client):
        """Test query with HTTP error."""
        mock_httpx_client.get = AsyncMock(
            side_effect=httpx.HTTPError("Connection refused")
        )

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        results = await collector.query("up")

        assert results == []

    @pytest.mark.asyncio
    async def test_query_invalid_value(self, mock_httpx_client):
        """Test query with invalid value that cannot be converted to float."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {"__name__": "test"},
                        "value": [1609459200.0, "NaN"]
                    }
                ]
            }
        }
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        # NaN should be converted (it's a valid float representation)
        results = await collector.query("test")
        # The code tries float() on "NaN" which works in Python
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_missing_value_field(self, mock_httpx_client):
        """Test query with missing value field."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {"metric": {"__name__": "test"}, "value": []}  # Empty value
                ]
            }
        }
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        results = await collector.query("test")

        assert results == []


class TestPrometheusCollectorQueryRange:
    """Tests for range query method."""

    @pytest.mark.asyncio
    async def test_query_range_success(self, mock_httpx_client):
        """Test successful range query."""
        now = datetime.now()
        start = now - timedelta(hours=1)
        end = now

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {"__name__": "test_metric", "job": "test"},
                        "values": [
                            [start.timestamp(), "1.0"],
                            [end.timestamp(), "2.0"]
                        ]
                    }
                ]
            }
        }
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        results = await collector.query_range("test_metric", start, end, step="1m")

        assert len(results) == 1
        assert results[0].metric_name == "test_metric"
        assert len(results[0].values) == 2
        assert results[0].values[0][1] == 1.0
        assert results[0].values[1][1] == 2.0

    @pytest.mark.asyncio
    async def test_query_range_empty_results(self, mock_httpx_client):
        """Test range query with empty results."""
        now = datetime.now()
        start = now - timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"result": []}
        }
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        results = await collector.query_range("test_metric", start, now)

        assert results == []

    @pytest.mark.asyncio
    async def test_query_range_failure_status(self, mock_httpx_client):
        """Test range query with error status."""
        now = datetime.now()
        start = now - timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "status": "error",
            "error": "query execution failed"
        }
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        results = await collector.query_range("test_metric", start, now)

        assert results == []

    @pytest.mark.asyncio
    async def test_query_range_http_error(self, mock_httpx_client):
        """Test range query with HTTP error."""
        now = datetime.now()
        start = now - timedelta(hours=1)

        mock_httpx_client.get = AsyncMock(
            side_effect=httpx.HTTPError("Timeout")
        )

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        results = await collector.query_range("test_metric", start, now)

        assert results == []

    @pytest.mark.asyncio
    async def test_query_range_invalid_values(self, mock_httpx_client):
        """Test range query with some invalid values."""
        now = datetime.now()
        start = now - timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {"__name__": "test"},
                        "values": [
                            [start.timestamp(), "1.0"],
                            [now.timestamp(), "not_a_number"]  # Invalid
                        ]
                    }
                ]
            }
        }
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        results = await collector.query_range("test", start, now)

        # Should only have one valid value
        assert len(results) == 1
        assert len(results[0].values) == 1


class TestPrometheusCollectorSeleniumMetrics:
    """Tests for Selenium metrics collection."""

    @pytest.mark.asyncio
    async def test_get_selenium_metrics_success(self, mock_httpx_client):
        """Test getting Selenium metrics successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        # Different responses for different queries
        call_count = 0
        def json_side_effect():
            nonlocal call_count
            call_count += 1
            values = {
                1: [{"metric": {}, "value": [1609459200.0, "5"]}],     # queued
                2: [{"metric": {}, "value": [1609459200.0, "10"]}],    # active
                3: [{"metric": {}, "value": [1609459200.0, "15"]}],    # total
                4: [{"metric": {}, "value": [1609459200.0, "8"]}],     # nodes_available
                5: [{"metric": {}, "value": [1609459200.0, "10"]}],    # nodes_total
                6: [{"metric": {}, "value": [1609459200.0, "120.5"]}], # avg_duration
                7: [{"metric": {}, "value": [1609459200.0, "3.2"]}],   # queue_wait
            }
            return {
                "status": "success",
                "data": {"result": values.get(call_count, [])}
            }

        mock_response.json = json_side_effect
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        metrics = await collector.get_selenium_metrics()

        assert metrics is not None
        assert metrics.sessions_queued == 5
        assert metrics.sessions_active == 10
        assert metrics.sessions_total == 15

    @pytest.mark.asyncio
    async def test_get_selenium_metrics_partial_data(self, mock_httpx_client):
        """Test Selenium metrics with missing data (defaults to 0)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"result": []}  # No data
        }
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        metrics = await collector.get_selenium_metrics()

        assert metrics is not None
        assert metrics.sessions_queued == 0
        assert metrics.sessions_active == 0
        assert metrics.avg_session_duration_seconds == 0.0


class TestPrometheusCollectorBrowserNodeMetrics:
    """Tests for browser node metrics collection."""

    @pytest.mark.asyncio
    async def test_get_browser_node_metrics_chrome(self, mock_httpx_client):
        """Test getting Chrome browser node metrics."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        call_count = 0
        def json_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # replicas query
                return {"status": "success", "data": {"result": []}}
            elif call_count == 2:  # cpu query
                return {"status": "success", "data": {"result": [{"metric": {}, "value": [1609459200.0, "75.5"]}]}}
            elif call_count == 3:  # memory query
                return {"status": "success", "data": {"result": [{"metric": {}, "value": [1609459200.0, "50.0"]}]}}
            elif call_count == 4:  # current replicas
                return {"status": "success", "data": {"result": [{"metric": {}, "value": [1609459200.0, "5"]}]}}
            elif call_count == 5:  # desired replicas
                return {"status": "success", "data": {"result": [{"metric": {}, "value": [1609459200.0, "7"]}]}}
            return {"status": "success", "data": {"result": []}}

        mock_response.json = json_side_effect
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        metrics = await collector.get_browser_node_metrics("chrome")

        assert metrics is not None
        assert metrics.browser_type == "chrome"
        assert metrics.replicas_min == 2  # Chrome default
        assert metrics.replicas_max == 15  # Chrome default

    @pytest.mark.asyncio
    async def test_get_browser_node_metrics_firefox(self, mock_httpx_client):
        """Test getting Firefox browser node metrics."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "success", "data": {"result": []}}
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        metrics = await collector.get_browser_node_metrics("firefox")

        assert metrics is not None
        assert metrics.browser_type == "firefox"
        assert metrics.replicas_min == 1  # Firefox default
        assert metrics.replicas_max == 8  # Firefox default

    @pytest.mark.asyncio
    async def test_get_browser_node_metrics_edge(self, mock_httpx_client):
        """Test getting Edge browser node metrics."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "success", "data": {"result": []}}
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        metrics = await collector.get_browser_node_metrics("edge")

        assert metrics is not None
        assert metrics.browser_type == "edge"
        assert metrics.replicas_min == 1  # Edge default
        assert metrics.replicas_max == 8  # Edge default


class TestPrometheusCollectorInfrastructureSnapshot:
    """Tests for infrastructure snapshot collection."""

    @pytest.mark.asyncio
    async def test_get_infrastructure_snapshot(self, mock_httpx_client):
        """Test getting complete infrastructure snapshot."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "success", "data": {"result": []}}
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        snapshot = await collector.get_infrastructure_snapshot()

        assert snapshot is not None
        assert snapshot.selenium is not None
        assert snapshot.chrome_nodes is not None
        assert snapshot.firefox_nodes is not None
        assert snapshot.edge_nodes is not None


class TestPrometheusCollectorUsagePatterns:
    """Tests for usage pattern analysis."""

    @pytest.mark.asyncio
    async def test_get_usage_patterns_with_data(self, mock_httpx_client):
        """Test getting usage patterns with data."""
        now = datetime.now()

        # Create sample time series data
        values = []
        for i in range(168):  # 7 days of hourly data
            ts = now - timedelta(hours=i)
            value = 10 + (i % 24) * 2  # Vary by hour
            values.append([ts.timestamp(), str(value)])

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {"metric": {"__name__": "selenium_sessions_active"}, "values": values}
                ]
            }
        }
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        patterns = await collector.get_usage_patterns(hours=168)

        assert "hourly_averages" in patterns
        assert len(patterns["hourly_averages"]) == 24
        assert "daily_averages" in patterns
        assert len(patterns["daily_averages"]) == 7
        assert "peak_hour" in patterns
        assert "peak_day" in patterns

    @pytest.mark.asyncio
    async def test_get_usage_patterns_no_data(self, mock_httpx_client):
        """Test getting usage patterns with no data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "success", "data": {"result": []}}
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        patterns = await collector.get_usage_patterns()

        assert patterns["hourly_averages"] == [0.0] * 24
        assert patterns["daily_averages"] == [0.0] * 7
        assert patterns["peak_hour"] == 0
        assert patterns["min_hour"] == 0


class TestPrometheusCollectorTestExecutionMetrics:
    """Tests for test execution metrics collection."""

    @pytest.mark.asyncio
    async def test_get_test_execution_metrics_success(self, mock_httpx_client):
        """Test getting test execution metrics."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        call_count = 0
        def json_side_effect():
            nonlocal call_count
            call_count += 1
            values = {
                1: [{"metric": {}, "value": [1609459200.0, "100"]}],   # total
                2: [{"metric": {}, "value": [1609459200.0, "95"]}],    # successful
                3: [{"metric": {}, "value": [1609459200.0, "5"]}],     # failed
                4: [{"metric": {}, "value": [1609459200.0, "30.5"]}],  # avg_duration
                5: [{"metric": {}, "value": [1609459200.0, "60.2"]}],  # p95_duration
            }
            return {
                "status": "success",
                "data": {"result": values.get(call_count, [])}
            }

        mock_response.json = json_side_effect
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        metrics = await collector.get_test_execution_metrics(hours=24)

        assert metrics["total_tests"] == 100
        assert metrics["successful_tests"] == 95
        assert metrics["failed_tests"] == 5
        assert metrics["success_rate"] == 95.0

    @pytest.mark.asyncio
    async def test_get_test_execution_metrics_no_tests(self, mock_httpx_client):
        """Test execution metrics with no tests."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "success", "data": {"result": []}}
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        collector._client = mock_httpx_client

        metrics = await collector.get_test_execution_metrics()

        assert metrics["total_tests"] == 0
        assert metrics["success_rate"] == 0.0  # Division by zero handled


class TestCreatePrometheusCollector:
    """Tests for factory function."""

    def test_create_with_url(self):
        """Test creating collector with explicit URL."""
        collector = create_prometheus_collector(
            prometheus_url="http://custom-prometheus:9090"
        )

        assert collector.prometheus_url == "http://custom-prometheus:9090"

    def test_create_with_env_variable(self, monkeypatch):
        """Test creating collector from environment variable."""
        monkeypatch.setenv("PROMETHEUS_URL", "http://env-prometheus:9090")

        collector = create_prometheus_collector()

        assert collector.prometheus_url == "http://env-prometheus:9090"

    def test_create_with_default(self, monkeypatch):
        """Test creating collector with default URL."""
        monkeypatch.delenv("PROMETHEUS_URL", raising=False)

        collector = create_prometheus_collector()

        assert "prometheus-server" in collector.prometheus_url
