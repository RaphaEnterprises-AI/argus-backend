"""Prometheus Metrics Collector for Infrastructure Optimization.

This service collects metrics from Prometheus/VictoriaMetrics for:
- Selenium Grid session metrics (queue length, active sessions, duration)
- Container resource utilization (CPU, memory)
- KEDA scaler metrics (replica counts)
- Custom Argus metrics (test execution times, success rates)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class MetricValue:
    """A single metric value with timestamp."""

    value: float
    timestamp: datetime
    labels: dict[str, str]


@dataclass
class MetricSeries:
    """A time series of metric values."""

    metric_name: str
    values: list[tuple[datetime, float]]  # (timestamp, value) pairs
    labels: dict[str, str]


@dataclass
class ResourceUtilization:
    """Resource utilization snapshot."""

    cpu_usage_percent: float
    cpu_request_percent: float  # Usage vs requests
    memory_usage_bytes: int
    memory_usage_percent: float
    timestamp: datetime


@dataclass
class SeleniumMetrics:
    """Selenium Grid metrics snapshot."""

    sessions_queued: int
    sessions_active: int
    sessions_total: int
    nodes_available: int
    nodes_total: int
    avg_session_duration_seconds: float
    queue_wait_time_seconds: float
    timestamp: datetime


@dataclass
class BrowserNodeMetrics:
    """Metrics for a specific browser type."""

    browser_type: str  # chrome, firefox, edge
    replicas_current: int
    replicas_desired: int
    replicas_min: int
    replicas_max: int
    cpu_utilization: ResourceUtilization
    memory_utilization: ResourceUtilization
    sessions_active: int
    timestamp: datetime


@dataclass
class InfrastructureSnapshot:
    """Complete infrastructure metrics snapshot."""

    selenium: SeleniumMetrics
    chrome_nodes: BrowserNodeMetrics
    firefox_nodes: BrowserNodeMetrics
    edge_nodes: BrowserNodeMetrics
    total_pods: int
    total_nodes: int
    cluster_cpu_utilization: float
    cluster_memory_utilization: float
    timestamp: datetime


class PrometheusCollector:
    """Collects metrics from Prometheus for infrastructure optimization."""

    def __init__(
        self,
        prometheus_url: str,
        timeout: float = 30.0,
    ):
        """Initialize the collector.

        Args:
            prometheus_url: Base URL of Prometheus server (e.g., http://prometheus:9090)
            timeout: Request timeout in seconds
        """
        self.prometheus_url = prometheus_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def query(self, query: str) -> list[MetricValue]:
        """Execute an instant query against Prometheus.

        Args:
            query: PromQL query string

        Returns:
            List of metric values
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query}
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                logger.warning(
                    "prometheus_query_failed",
                    query=query,
                    error=data.get("error", "Unknown error")
                )
                return []

            results = []
            for result in data.get("data", {}).get("result", []):
                metric = result.get("metric", {})
                value = result.get("value", [])

                if len(value) >= 2:
                    timestamp = datetime.fromtimestamp(float(value[0]))
                    try:
                        val = float(value[1])
                    except (ValueError, TypeError):
                        continue

                    results.append(MetricValue(
                        value=val,
                        timestamp=timestamp,
                        labels=metric
                    ))

            return results

        except httpx.HTTPError as e:
            logger.error("prometheus_query_error", query=query, error=str(e))
            return []

    async def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "1m"
    ) -> list[MetricSeries]:
        """Execute a range query against Prometheus.

        Args:
            query: PromQL query string
            start: Start time
            end: End time
            step: Query resolution (e.g., "1m", "5m", "1h")

        Returns:
            List of metric time series
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start.timestamp(),
                    "end": end.timestamp(),
                    "step": step
                }
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                logger.warning(
                    "prometheus_range_query_failed",
                    query=query,
                    error=data.get("error", "Unknown error")
                )
                return []

            results = []
            for result in data.get("data", {}).get("result", []):
                metric = result.get("metric", {})
                values = result.get("values", [])

                series_values = []
                for v in values:
                    if len(v) >= 2:
                        timestamp = datetime.fromtimestamp(float(v[0]))
                        try:
                            val = float(v[1])
                            series_values.append((timestamp, val))
                        except (ValueError, TypeError):
                            continue

                if series_values:
                    results.append(MetricSeries(
                        metric_name=metric.get("__name__", "unknown"),
                        values=series_values,
                        labels=metric
                    ))

            return results

        except httpx.HTTPError as e:
            logger.error("prometheus_range_query_error", query=query, error=str(e))
            return []

    async def get_selenium_metrics(self) -> SeleniumMetrics | None:
        """Get current browser pool metrics.

        Note: Supports both Selenium Grid metrics (selenium_*) and
        Vultr Browser Pool metrics (browser_*) for compatibility.
        """
        # Try browser pool metrics first (Vultr K8s), fall back to Selenium Grid
        queries = {
            # Browser pool metrics (primary)
            "active": "browser_sessions_active",
            "total": "browser_sessions_total",
            # Selenium Grid metrics (fallback)
            "queued": "selenium_sessions_queued",
            "nodes_available": "selenium_nodes_available",
            "nodes_total": "selenium_nodes_total",
            "avg_duration": "avg(browser_action_duration_seconds_sum / browser_action_duration_seconds_count)",
            "queue_wait": "avg(selenium_session_queue_wait_time_seconds)",
        }

        results = {}
        for key, query in queries.items():
            values = await self.query(query)
            if values:
                results[key] = values[0].value
            else:
                results[key] = 0.0

        return SeleniumMetrics(
            sessions_queued=int(results.get("queued", 0)),
            sessions_active=int(results.get("active", 0)),
            sessions_total=int(results.get("total", 0)),
            nodes_available=int(results.get("nodes_available", 0)),
            nodes_total=int(results.get("nodes_total", 0)),
            avg_session_duration_seconds=results.get("avg_duration", 0.0),
            queue_wait_time_seconds=results.get("queue_wait", 0.0),
            timestamp=datetime.now()
        )

    async def get_browser_node_metrics(
        self,
        browser_type: str
    ) -> BrowserNodeMetrics | None:
        """Get metrics for a specific browser node type.

        Args:
            browser_type: One of "chrome", "firefox", "edge"

        Note: For Vultr Browser Pool, we use browser-worker metrics.
        Only Chrome is available in the custom pool (Playwright Chromium).
        Firefox and Edge return zeros (not deployed in custom pool).
        """
        # Vultr Browser Pool only has Chromium via Playwright
        # Firefox/Edge would require Selenium Grid or TestingBot
        if browser_type != "chrome":
            min_replicas = {"firefox": 1, "edge": 1}.get(browser_type, 1)
            max_replicas = {"firefox": 8, "edge": 8}.get(browser_type, 8)
            return BrowserNodeMetrics(
                browser_type=browser_type,
                replicas_current=0,
                replicas_desired=0,
                replicas_min=min_replicas,
                replicas_max=max_replicas,
                cpu_utilization=ResourceUtilization(0, 0, 0, 0, datetime.now()),
                memory_utilization=ResourceUtilization(0, 0, 0, 0, datetime.now()),
                sessions_active=0,
                timestamp=datetime.now()
            )

        # For Chrome (browser-worker), use available metrics
        # Count active browser workers
        worker_count_query = 'count(up{job="browser-worker"} == 1)'
        worker_values = await self.query(worker_count_query)
        current_replicas = int(worker_values[0].value) if worker_values else 0

        # Get active sessions from browser pool
        sessions_query = 'sum(browser_sessions_active)'
        session_values = await self.query(sessions_query)
        sessions_active = int(session_values[0].value) if session_values else 0

        # Get CPU utilization from process metrics
        cpu_query = 'avg(rate(process_cpu_seconds_total{job="browser-worker"}[5m])) * 100'
        cpu_values = await self.query(cpu_query)
        cpu_percent = cpu_values[0].value if cpu_values else 0.0

        # Get memory utilization from process metrics (as percentage of 512MB limit)
        memory_query = 'avg(process_resident_memory_bytes{job="browser-worker"}) / (512 * 1024 * 1024) * 100'
        memory_values = await self.query(memory_query)
        memory_percent = memory_values[0].value if memory_values else 0.0

        # KEDA min/max for browser workers
        min_replicas = 2
        max_replicas = 15

        return BrowserNodeMetrics(
            browser_type=browser_type,
            replicas_current=current_replicas,
            replicas_desired=desired_replicas,
            replicas_min=min_replicas,
            replicas_max=max_replicas,
            cpu_utilization=ResourceUtilization(
                cpu_usage_percent=cpu_percent,
                cpu_request_percent=cpu_percent,
                memory_usage_bytes=0,
                memory_usage_percent=0.0,
                timestamp=datetime.now()
            ),
            memory_utilization=ResourceUtilization(
                cpu_usage_percent=0.0,
                cpu_request_percent=0.0,
                memory_usage_bytes=0,
                memory_usage_percent=memory_percent,
                timestamp=datetime.now()
            ),
            sessions_active=0,  # Would need Selenium Grid API
            timestamp=datetime.now()
        )

    async def get_infrastructure_snapshot(self) -> InfrastructureSnapshot:
        """Get complete infrastructure metrics snapshot.

        Note: Uses Vultr Browser Pool metrics when available,
        falls back to Selenium Grid metrics if configured.
        """
        selenium = await self.get_selenium_metrics()
        chrome = await self.get_browser_node_metrics("chrome")
        firefox = await self.get_browser_node_metrics("firefox")
        edge = await self.get_browser_node_metrics("edge")

        # Get cluster-wide metrics from browser pool (simpler queries)
        # CPU: average across all browser workers
        cluster_cpu_query = 'avg(rate(process_cpu_seconds_total[5m])) * 100'
        cpu_values = await self.query(cluster_cpu_query)
        cluster_cpu = cpu_values[0].value if cpu_values else 0.0

        # Memory: average memory usage as percentage
        cluster_mem_query = 'avg(process_resident_memory_bytes) / (512 * 1024 * 1024) * 100'
        mem_values = await self.query(cluster_mem_query)
        cluster_mem = mem_values[0].value if mem_values else 0.0

        # Count total workers (browser-worker + browser-manager)
        worker_query = 'count(up{job=~"browser-worker|browser-manager"} == 1)'
        worker_values = await self.query(worker_query)
        total_pods = int(worker_values[0].value) if worker_values else 0

        # Nodes = unique instances (for browser pool, each pod is on its own "node")
        total_nodes = total_pods  # Simplified for browser pool

        return InfrastructureSnapshot(
            selenium=selenium or SeleniumMetrics(
                sessions_queued=0, sessions_active=0, sessions_total=0,
                nodes_available=0, nodes_total=0, avg_session_duration_seconds=0.0,
                queue_wait_time_seconds=0.0, timestamp=datetime.now()
            ),
            chrome_nodes=chrome or BrowserNodeMetrics(
                browser_type="chrome", replicas_current=0, replicas_desired=0,
                replicas_min=2, replicas_max=15,
                cpu_utilization=ResourceUtilization(0, 0, 0, 0, datetime.now()),
                memory_utilization=ResourceUtilization(0, 0, 0, 0, datetime.now()),
                sessions_active=0, timestamp=datetime.now()
            ),
            firefox_nodes=firefox or BrowserNodeMetrics(
                browser_type="firefox", replicas_current=0, replicas_desired=0,
                replicas_min=1, replicas_max=8,
                cpu_utilization=ResourceUtilization(0, 0, 0, 0, datetime.now()),
                memory_utilization=ResourceUtilization(0, 0, 0, 0, datetime.now()),
                sessions_active=0, timestamp=datetime.now()
            ),
            edge_nodes=edge or BrowserNodeMetrics(
                browser_type="edge", replicas_current=0, replicas_desired=0,
                replicas_min=1, replicas_max=8,
                cpu_utilization=ResourceUtilization(0, 0, 0, 0, datetime.now()),
                memory_utilization=ResourceUtilization(0, 0, 0, 0, datetime.now()),
                sessions_active=0, timestamp=datetime.now()
            ),
            total_pods=total_pods,
            total_nodes=total_nodes,
            cluster_cpu_utilization=cluster_cpu,
            cluster_memory_utilization=cluster_mem,
            timestamp=datetime.now()
        )

    async def get_usage_patterns(
        self,
        hours: int = 168  # 7 days
    ) -> dict:
        """Get usage patterns over time for demand prediction.

        Returns hourly averages for:
        - Sessions by hour of day
        - Sessions by day of week
        - Peak usage times
        """
        end = datetime.now()
        start = end - timedelta(hours=hours)

        # Get session activity over time
        series = await self.query_range(
            "selenium_sessions_active",
            start=start,
            end=end,
            step="1h"
        )

        if not series:
            return {
                "hourly_averages": [0.0] * 24,
                "daily_averages": [0.0] * 7,
                "peak_hour": 0,
                "peak_day": 0,
                "min_hour": 0,
                "min_day": 0,
            }

        # Aggregate by hour of day
        hourly_sums = [0.0] * 24
        hourly_counts = [0] * 24
        daily_sums = [0.0] * 7
        daily_counts = [0] * 7

        for ts, value in series[0].values:
            hour = ts.hour
            day = ts.weekday()

            hourly_sums[hour] += value
            hourly_counts[hour] += 1
            daily_sums[day] += value
            daily_counts[day] += 1

        hourly_averages = [
            hourly_sums[i] / hourly_counts[i] if hourly_counts[i] > 0 else 0.0
            for i in range(24)
        ]

        daily_averages = [
            daily_sums[i] / daily_counts[i] if daily_counts[i] > 0 else 0.0
            for i in range(7)
        ]

        return {
            "hourly_averages": hourly_averages,
            "daily_averages": daily_averages,
            "peak_hour": hourly_averages.index(max(hourly_averages)),
            "peak_day": daily_averages.index(max(daily_averages)),
            "min_hour": hourly_averages.index(min(hourly_averages)),
            "min_day": daily_averages.index(min(daily_averages)),
        }

    async def get_test_execution_metrics(
        self,
        hours: int = 24
    ) -> dict:
        """Get test execution metrics for the given time period."""
        end = datetime.now()
        end - timedelta(hours=hours)

        # Query custom Argus metrics
        queries = {
            "total_tests": "sum(increase(argus_test_executions_total[24h]))",
            "successful_tests": "sum(increase(argus_test_executions_total{status=\"success\"}[24h]))",
            "failed_tests": "sum(increase(argus_test_executions_total{status=\"failed\"}[24h]))",
            "avg_duration": "avg(argus_test_duration_seconds)",
            "p95_duration": "histogram_quantile(0.95, rate(argus_test_duration_seconds_bucket[24h]))",
        }

        results = {}
        for key, query in queries.items():
            values = await self.query(query)
            if values:
                results[key] = values[0].value
            else:
                results[key] = 0.0

        total = results.get("total_tests", 0)
        successful = results.get("successful_tests", 0)

        return {
            "total_tests": int(total),
            "successful_tests": int(successful),
            "failed_tests": int(results.get("failed_tests", 0)),
            "success_rate": (successful / total * 100) if total > 0 else 0.0,
            "avg_duration_seconds": results.get("avg_duration", 0.0),
            "p95_duration_seconds": results.get("p95_duration", 0.0),
        }


# Factory function
def create_prometheus_collector(
    prometheus_url: str | None = None
) -> PrometheusCollector:
    """Create a Prometheus collector instance.

    Args:
        prometheus_url: Prometheus URL. If not provided, uses environment variable.
    """
    import os

    url = prometheus_url or os.getenv(
        "PROMETHEUS_URL",
        "http://prometheus-server.monitoring.svc.cluster.local:9090"
    )

    return PrometheusCollector(prometheus_url=url)
