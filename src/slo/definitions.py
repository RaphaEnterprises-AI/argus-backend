"""SLO Definitions for Argus E2E Testing Platform.

Defines Service Level Objectives (SLOs) for all critical services:
- API endpoints (availability, latency, error rate)
- Browser pool (session availability, execution time)
- Data layer (Redpanda, FalkorDB, Valkey, Cognee)
- LLM services (latency, error rate, cost)

RAP-329: Define SLO/SLI Framework

SLO Target Philosophy:
- Availability: 99.9% for critical paths, 99.5% for non-critical
- Latency P95: <200ms for API, <500ms for AI-powered endpoints
- Error Rate: <1% for critical, <5% for background jobs

References:
- Google SRE Book: https://sre.google/sre-book/service-level-objectives/
- Datadog SLO Best Practices: https://www.datadoghq.com/blog/slo-monitoring-tracking/
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SLOCategory(str, Enum):
    """Categories of SLOs for organization."""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    SATURATION = "saturation"


@dataclass
class SLO:
    """Service Level Objective definition.

    Attributes:
        name: Human-readable SLO name
        service: Service this SLO applies to
        category: Type of SLO (availability, latency, etc.)
        target: Target value (e.g., 99.9 for 99.9% availability)
        unit: Unit of measurement (percent, ms, requests/sec)
        window_days: Rolling window for calculation (default 30 days)
        description: Detailed description of what this SLO measures
        prometheus_query: PromQL query to calculate the SLI
        alerting_threshold: Threshold for alerting (e.g., 99.5 when target is 99.9)
        critical: Whether this SLO is critical (affects error budget burn rate alerts)
        tags: Additional tags for filtering/grouping
    """
    name: str
    service: str
    category: SLOCategory
    target: float
    unit: str
    window_days: int = 30
    description: str = ""
    prometheus_query: str = ""
    alerting_threshold: float | None = None
    critical: bool = False
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Set default alerting threshold if not provided."""
        if self.alerting_threshold is None:
            # Default: alert when 50% of error budget is consumed
            if self.category == SLOCategory.AVAILABILITY:
                # For 99.9% target, alert at 99.5%
                self.alerting_threshold = self.target - (100 - self.target) * 0.5
            elif self.category == SLOCategory.ERROR_RATE:
                # For 1% error rate target, alert at 2%
                self.alerting_threshold = self.target * 2
            else:
                # For latency, alert at 1.5x target
                self.alerting_threshold = self.target * 1.5

    @property
    def error_budget_percent(self) -> float:
        """Calculate error budget as percentage.

        For availability SLOs: error budget = 100 - target
        For error rate SLOs: error budget = target
        """
        if self.category == SLOCategory.AVAILABILITY:
            return 100 - self.target
        elif self.category == SLOCategory.ERROR_RATE:
            return self.target
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "service": self.service,
            "category": self.category.value,
            "target": self.target,
            "unit": self.unit,
            "window_days": self.window_days,
            "description": self.description,
            "alerting_threshold": self.alerting_threshold,
            "error_budget_percent": self.error_budget_percent,
            "critical": self.critical,
            "tags": self.tags,
        }


# =============================================================================
# API SLOs - Backend REST API Endpoints
# =============================================================================

API_SLOS = [
    # Overall API Availability
    SLO(
        name="API Availability",
        service="api",
        category=SLOCategory.AVAILABILITY,
        target=99.9,
        unit="percent",
        description="Percentage of successful API requests (non-5xx responses)",
        prometheus_query="""
            (
                sum(rate(http_requests_total{job="argus-api", status!~"5.."}[5m]))
                /
                sum(rate(http_requests_total{job="argus-api"}[5m]))
            ) * 100
        """,
        critical=True,
        tags=["api", "core"],
    ),

    # API Latency P95
    SLO(
        name="API Latency P95",
        service="api",
        category=SLOCategory.LATENCY,
        target=200,
        unit="ms",
        description="95th percentile response time for API requests",
        prometheus_query="""
            histogram_quantile(0.95,
                sum(rate(http_request_duration_seconds_bucket{job="argus-api"}[5m])) by (le)
            ) * 1000
        """,
        critical=True,
        tags=["api", "latency"],
    ),

    # API Error Rate
    SLO(
        name="API Error Rate",
        service="api",
        category=SLOCategory.ERROR_RATE,
        target=1.0,
        unit="percent",
        description="Percentage of API requests returning 5xx errors",
        prometheus_query="""
            (
                sum(rate(http_requests_total{job="argus-api", status=~"5.."}[5m]))
                /
                sum(rate(http_requests_total{job="argus-api"}[5m]))
            ) * 100
        """,
        critical=True,
        tags=["api", "errors"],
    ),

    # Health Endpoint Availability (critical path)
    SLO(
        name="Health Endpoint Availability",
        service="api",
        category=SLOCategory.AVAILABILITY,
        target=99.99,
        unit="percent",
        description="Availability of /health endpoint for load balancer checks",
        prometheus_query="""
            (
                sum(rate(http_requests_total{job="argus-api", path="/health", status="200"}[5m]))
                /
                sum(rate(http_requests_total{job="argus-api", path="/health"}[5m]))
            ) * 100
        """,
        critical=True,
        tags=["api", "health", "critical-path"],
    ),

    # Chat/AI Endpoint Latency (higher threshold for AI)
    SLO(
        name="AI Endpoint Latency P95",
        service="api",
        category=SLOCategory.LATENCY,
        target=5000,
        unit="ms",
        description="95th percentile response time for AI-powered endpoints (chat, analysis)",
        prometheus_query="""
            histogram_quantile(0.95,
                sum(rate(http_request_duration_seconds_bucket{job="argus-api", path=~"/api/v1/chat.*|/api/v1/analyze.*"}[5m])) by (le)
            ) * 1000
        """,
        critical=False,
        tags=["api", "ai", "latency"],
    ),
]


# =============================================================================
# Browser Pool SLOs - Selenium Grid / Browser Workers
# =============================================================================

BROWSER_POOL_SLOS = [
    # Browser Pool Availability
    SLO(
        name="Browser Pool Availability",
        service="browser-pool",
        category=SLOCategory.AVAILABILITY,
        target=99.5,
        unit="percent",
        description="Percentage of time at least one browser worker is available",
        prometheus_query="""
            (
                sum_over_time((count(up{job="browser-worker"} == 1) > 0)[5m:1m])
                /
                count_over_time(vector(1)[5m:1m])
            ) * 100
        """,
        critical=True,
        tags=["browser", "availability"],
    ),

    # Test Execution Success Rate
    SLO(
        name="Test Execution Success Rate",
        service="browser-pool",
        category=SLOCategory.AVAILABILITY,
        target=95.0,
        unit="percent",
        description="Percentage of tests that execute without infrastructure failures",
        prometheus_query="""
            (
                sum(rate(browser_tests_total{status="success"}[5m]))
                /
                sum(rate(browser_tests_total[5m]))
            ) * 100
        """,
        critical=False,
        tags=["browser", "tests"],
    ),

    # Test Duration P95
    SLO(
        name="Test Duration P95",
        service="browser-pool",
        category=SLOCategory.LATENCY,
        target=60000,
        unit="ms",
        description="95th percentile test execution time (1 minute target)",
        prometheus_query="""
            histogram_quantile(0.95,
                sum(rate(browser_test_duration_seconds_bucket[5m])) by (le)
            ) * 1000
        """,
        critical=False,
        tags=["browser", "latency"],
    ),

    # Session Queue Wait Time
    SLO(
        name="Session Queue Wait Time P95",
        service="browser-pool",
        category=SLOCategory.LATENCY,
        target=10000,
        unit="ms",
        description="95th percentile wait time for browser session allocation",
        prometheus_query="""
            histogram_quantile(0.95,
                sum(rate(selenium_session_queue_wait_time_seconds_bucket[5m])) by (le)
            ) * 1000
        """,
        critical=False,
        tags=["browser", "queue"],
    ),
]


# =============================================================================
# Data Layer SLOs - Redpanda, FalkorDB, Valkey, Cognee
# =============================================================================

DATA_LAYER_SLOS = [
    # Redpanda Availability
    SLO(
        name="Redpanda Availability",
        service="redpanda",
        category=SLOCategory.AVAILABILITY,
        target=99.9,
        unit="percent",
        description="Redpanda cluster availability (all brokers up, no underreplicated partitions)",
        prometheus_query="""
            (
                (up{job="redpanda"} == 1)
                and
                (redpanda_cluster_partitions_underreplicated == 0)
            ) * 100
        """,
        critical=True,
        tags=["data-layer", "kafka", "streaming"],
    ),

    # FalkorDB Availability
    SLO(
        name="FalkorDB Availability",
        service="falkordb",
        category=SLOCategory.AVAILABILITY,
        target=99.9,
        unit="percent",
        description="FalkorDB graph database availability",
        prometheus_query="up{job=\"falkordb\"} * 100",
        critical=True,
        tags=["data-layer", "graph"],
    ),

    # Valkey Cache Availability
    SLO(
        name="Valkey Cache Availability",
        service="valkey",
        category=SLOCategory.AVAILABILITY,
        target=99.9,
        unit="percent",
        description="Valkey cache availability",
        prometheus_query="up{job=\"valkey\"} * 100",
        critical=True,
        tags=["data-layer", "cache"],
    ),

    # Cognee Processing Latency
    SLO(
        name="Cognee Processing Latency P95",
        service="cognee",
        category=SLOCategory.LATENCY,
        target=30000,
        unit="ms",
        description="95th percentile event processing time for Cognee ECL pipeline",
        prometheus_query="""
            histogram_quantile(0.95,
                sum(rate(cognee_event_processing_duration_seconds_bucket[5m])) by (le)
            ) * 1000
        """,
        critical=False,
        tags=["data-layer", "cognee", "ai"],
    ),

    # Cognee Error Rate
    SLO(
        name="Cognee Error Rate",
        service="cognee",
        category=SLOCategory.ERROR_RATE,
        target=5.0,
        unit="percent",
        description="Percentage of Cognee events failing processing",
        prometheus_query="""
            (
                sum(rate(cognee_events_processed_total{status="error"}[5m]))
                /
                sum(rate(cognee_events_processed_total[5m]))
            ) * 100
        """,
        critical=False,
        tags=["data-layer", "cognee"],
    ),

    # Cognee Search Latency
    SLO(
        name="Cognee Search Latency P95",
        service="cognee",
        category=SLOCategory.LATENCY,
        target=100,
        unit="ms",
        description="95th percentile knowledge graph search latency (target <100ms)",
        prometheus_query="""
            histogram_quantile(0.95,
                sum(rate(cognee_search_duration_seconds_bucket[5m])) by (le)
            ) * 1000
        """,
        critical=False,
        tags=["data-layer", "cognee", "search"],
    ),

    # Kafka Consumer Lag
    SLO(
        name="Kafka Consumer Lag",
        service="redpanda",
        category=SLOCategory.SATURATION,
        target=1000,
        unit="messages",
        description="Maximum consumer lag for Cognee workers (should stay under 1000)",
        prometheus_query="""
            max(kafka_consumer_group_lag{group="argus-cognee-workers"}) by (topic)
        """,
        critical=False,
        tags=["data-layer", "kafka", "lag"],
    ),
]


# =============================================================================
# LLM Service SLOs - AI Model Endpoints
# =============================================================================

LLM_SLOS = [
    # LLM Request Latency P95
    SLO(
        name="LLM Request Latency P95",
        service="llm",
        category=SLOCategory.LATENCY,
        target=10000,
        unit="ms",
        description="95th percentile LLM request latency (10 second target)",
        prometheus_query="""
            histogram_quantile(0.95,
                sum(rate(llm_request_duration_seconds_bucket[5m])) by (le)
            ) * 1000
        """,
        critical=False,
        tags=["llm", "ai", "latency"],
    ),

    # LLM Error Rate
    SLO(
        name="LLM Error Rate",
        service="llm",
        category=SLOCategory.ERROR_RATE,
        target=2.0,
        unit="percent",
        description="Percentage of LLM requests failing",
        prometheus_query="""
            (
                sum(rate(llm_requests_total{status="error"}[5m]))
                /
                sum(rate(llm_requests_total[5m]))
            ) * 100
        """,
        critical=False,
        tags=["llm", "ai", "errors"],
    ),

    # Intelligence Layer Cache Hit Rate
    SLO(
        name="Intelligence Cache Hit Rate",
        service="intelligence",
        category=SLOCategory.AVAILABILITY,
        target=50.0,
        unit="percent",
        description="Percentage of intelligence queries served from cache (target >50%)",
        prometheus_query="""
            (
                sum(rate(intelligence_cache_hits_total[5m]))
                /
                (sum(rate(intelligence_cache_hits_total[5m])) + sum(rate(intelligence_cache_misses_total[5m])))
            ) * 100
        """,
        critical=False,
        tags=["intelligence", "cache"],
    ),

    # Intelligence Query Latency P95
    SLO(
        name="Intelligence Query Latency P95",
        service="intelligence",
        category=SLOCategory.LATENCY,
        target=200,
        unit="ms",
        description="95th percentile intelligence query latency (target <200ms)",
        prometheus_query="""
            histogram_quantile(0.95,
                sum(rate(intelligence_query_duration_seconds_bucket[5m])) by (le)
            ) * 1000
        """,
        critical=False,
        tags=["intelligence", "latency"],
    ),

    # LLM Fallback Rate (should be low)
    SLO(
        name="Intelligence LLM Fallback Rate",
        service="intelligence",
        category=SLOCategory.ERROR_RATE,
        target=20.0,
        unit="percent",
        description="Percentage of intelligence queries falling back to LLM (target <20%)",
        prometheus_query="""
            (
                sum(rate(intelligence_llm_fallback_total[5m]))
                /
                sum(rate(intelligence_queries_total[5m]))
            ) * 100
        """,
        critical=False,
        tags=["intelligence", "llm-fallback"],
    ),
]


# =============================================================================
# SLO Registry
# =============================================================================

class SLORegistry:
    """Registry for all SLO definitions.

    Provides lookup by service, category, and tags.

    Usage:
        registry = SLORegistry()
        api_slos = registry.get_by_service("api")
        critical_slos = registry.get_critical()
        latency_slos = registry.get_by_category(SLOCategory.LATENCY)
    """

    def __init__(self):
        """Initialize registry with all SLO definitions."""
        self._slos: list[SLO] = []
        self._by_service: dict[str, list[SLO]] = {}
        self._by_name: dict[str, SLO] = {}

        # Register all SLOs
        all_slos = API_SLOS + BROWSER_POOL_SLOS + DATA_LAYER_SLOS + LLM_SLOS
        for slo in all_slos:
            self.register(slo)

    def register(self, slo: SLO) -> None:
        """Register an SLO."""
        self._slos.append(slo)
        self._by_name[slo.name] = slo

        if slo.service not in self._by_service:
            self._by_service[slo.service] = []
        self._by_service[slo.service].append(slo)

    def get_all(self) -> list[SLO]:
        """Get all registered SLOs."""
        return self._slos.copy()

    def get_by_name(self, name: str) -> SLO | None:
        """Get SLO by name."""
        return self._by_name.get(name)

    def get_by_service(self, service: str) -> list[SLO]:
        """Get all SLOs for a service."""
        return self._by_service.get(service, [])

    def get_by_category(self, category: SLOCategory) -> list[SLO]:
        """Get all SLOs of a specific category."""
        return [slo for slo in self._slos if slo.category == category]

    def get_critical(self) -> list[SLO]:
        """Get all critical SLOs."""
        return [slo for slo in self._slos if slo.critical]

    def get_by_tags(self, tags: list[str]) -> list[SLO]:
        """Get all SLOs matching any of the given tags."""
        return [
            slo for slo in self._slos
            if any(tag in slo.tags for tag in tags)
        ]

    def get_services(self) -> list[str]:
        """Get list of all services with SLOs."""
        return list(self._by_service.keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert registry to dictionary for API responses."""
        return {
            "total_slos": len(self._slos),
            "critical_slos": len(self.get_critical()),
            "services": self.get_services(),
            "slos_by_service": {
                service: [slo.to_dict() for slo in slos]
                for service, slos in self._by_service.items()
            },
        }


# Singleton instance
_registry: SLORegistry | None = None


def get_slo_registry() -> SLORegistry:
    """Get the global SLO registry instance."""
    global _registry
    if _registry is None:
        _registry = SLORegistry()
    return _registry
