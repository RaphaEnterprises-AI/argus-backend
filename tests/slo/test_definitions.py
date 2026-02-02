"""Tests for SLO definitions.

RAP-329: Define SLO/SLI Framework
"""

import pytest

from src.slo.definitions import (
    SLO,
    SLOCategory,
    SLORegistry,
    get_slo_registry,
    API_SLOS,
    BROWSER_POOL_SLOS,
    DATA_LAYER_SLOS,
    LLM_SLOS,
)


class TestSLODefinition:
    """Tests for SLO dataclass."""

    def test_availability_slo_creation(self):
        """Test creating an availability SLO."""
        slo = SLO(
            name="Test Availability",
            service="test-service",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
            description="Test availability SLO",
        )

        assert slo.name == "Test Availability"
        assert slo.service == "test-service"
        assert slo.category == SLOCategory.AVAILABILITY
        assert slo.target == 99.9
        assert slo.error_budget_percent == pytest.approx(0.1)  # 100 - 99.9

    def test_latency_slo_creation(self):
        """Test creating a latency SLO."""
        slo = SLO(
            name="Test Latency",
            service="test-service",
            category=SLOCategory.LATENCY,
            target=200,
            unit="ms",
        )

        assert slo.target == 200
        assert slo.category == SLOCategory.LATENCY
        assert slo.error_budget_percent == 0.0  # N/A for latency

    def test_error_rate_slo_creation(self):
        """Test creating an error rate SLO."""
        slo = SLO(
            name="Test Error Rate",
            service="test-service",
            category=SLOCategory.ERROR_RATE,
            target=1.0,
            unit="percent",
        )

        assert slo.target == 1.0
        assert slo.error_budget_percent == 1.0  # Equal to target for error rate

    def test_default_alerting_threshold_availability(self):
        """Test default alerting threshold for availability SLOs."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
        )

        # Default: target - (100 - target) * 0.5 = 99.9 - 0.05 = 99.85
        assert slo.alerting_threshold == pytest.approx(99.85)

    def test_default_alerting_threshold_error_rate(self):
        """Test default alerting threshold for error rate SLOs."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.ERROR_RATE,
            target=1.0,
            unit="percent",
        )

        # Default: target * 2 = 2.0
        assert slo.alerting_threshold == 2.0

    def test_default_alerting_threshold_latency(self):
        """Test default alerting threshold for latency SLOs."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.LATENCY,
            target=200,
            unit="ms",
        )

        # Default: target * 1.5 = 300
        assert slo.alerting_threshold == 300

    def test_custom_alerting_threshold(self):
        """Test custom alerting threshold."""
        slo = SLO(
            name="Test",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
            alerting_threshold=99.0,  # Custom
        )

        assert slo.alerting_threshold == 99.0

    def test_slo_to_dict(self):
        """Test converting SLO to dictionary."""
        slo = SLO(
            name="Test SLO",
            service="test",
            category=SLOCategory.AVAILABILITY,
            target=99.9,
            unit="percent",
            critical=True,
            tags=["test", "critical"],
        )

        d = slo.to_dict()

        assert d["name"] == "Test SLO"
        assert d["service"] == "test"
        assert d["category"] == "availability"
        assert d["target"] == 99.9
        assert d["unit"] == "percent"
        assert d["critical"] is True
        assert d["tags"] == ["test", "critical"]
        assert d["error_budget_percent"] == pytest.approx(0.1)


class TestPredefinedSLOs:
    """Tests for predefined SLO definitions."""

    def test_api_slos_defined(self):
        """Test that API SLOs are defined."""
        assert len(API_SLOS) > 0

        # Check for required SLOs
        names = [slo.name for slo in API_SLOS]
        assert "API Availability" in names
        assert "API Latency P95" in names
        assert "API Error Rate" in names

    def test_api_availability_slo(self):
        """Test API availability SLO details."""
        api_avail = next(s for s in API_SLOS if s.name == "API Availability")

        assert api_avail.target == 99.9
        assert api_avail.category == SLOCategory.AVAILABILITY
        assert api_avail.critical is True
        assert api_avail.prometheus_query  # Has a query defined

    def test_browser_pool_slos_defined(self):
        """Test that browser pool SLOs are defined."""
        assert len(BROWSER_POOL_SLOS) > 0

        names = [slo.name for slo in BROWSER_POOL_SLOS]
        assert "Browser Pool Availability" in names
        assert "Test Execution Success Rate" in names

    def test_data_layer_slos_defined(self):
        """Test that data layer SLOs are defined."""
        assert len(DATA_LAYER_SLOS) > 0

        names = [slo.name for slo in DATA_LAYER_SLOS]
        assert "Redpanda Availability" in names
        assert "FalkorDB Availability" in names
        assert "Valkey Cache Availability" in names
        assert "Cognee Processing Latency P95" in names

    def test_llm_slos_defined(self):
        """Test that LLM SLOs are defined."""
        assert len(LLM_SLOS) > 0

        names = [slo.name for slo in LLM_SLOS]
        assert "LLM Request Latency P95" in names
        assert "LLM Error Rate" in names

    def test_all_slos_have_prometheus_queries(self):
        """Test that all SLOs have Prometheus queries defined."""
        all_slos = API_SLOS + BROWSER_POOL_SLOS + DATA_LAYER_SLOS + LLM_SLOS

        for slo in all_slos:
            assert slo.prometheus_query, f"SLO '{slo.name}' missing Prometheus query"


class TestSLORegistry:
    """Tests for SLO registry."""

    def test_registry_initialization(self):
        """Test that registry initializes with all SLOs."""
        registry = SLORegistry()

        total_expected = len(API_SLOS) + len(BROWSER_POOL_SLOS) + len(DATA_LAYER_SLOS) + len(LLM_SLOS)
        assert len(registry.get_all()) == total_expected

    def test_get_by_service(self):
        """Test getting SLOs by service."""
        registry = SLORegistry()

        api_slos = registry.get_by_service("api")
        assert len(api_slos) == len(API_SLOS)
        assert all(slo.service == "api" for slo in api_slos)

    def test_get_by_service_nonexistent(self):
        """Test getting SLOs for nonexistent service returns empty."""
        registry = SLORegistry()

        slos = registry.get_by_service("nonexistent")
        assert slos == []

    def test_get_by_name(self):
        """Test getting SLO by name."""
        registry = SLORegistry()

        slo = registry.get_by_name("API Availability")
        assert slo is not None
        assert slo.name == "API Availability"

    def test_get_by_name_nonexistent(self):
        """Test getting nonexistent SLO by name returns None."""
        registry = SLORegistry()

        slo = registry.get_by_name("Nonexistent SLO")
        assert slo is None

    def test_get_by_category(self):
        """Test getting SLOs by category."""
        registry = SLORegistry()

        availability_slos = registry.get_by_category(SLOCategory.AVAILABILITY)
        assert len(availability_slos) > 0
        assert all(slo.category == SLOCategory.AVAILABILITY for slo in availability_slos)

        latency_slos = registry.get_by_category(SLOCategory.LATENCY)
        assert len(latency_slos) > 0
        assert all(slo.category == SLOCategory.LATENCY for slo in latency_slos)

    def test_get_critical(self):
        """Test getting critical SLOs."""
        registry = SLORegistry()

        critical_slos = registry.get_critical()
        assert len(critical_slos) > 0
        assert all(slo.critical is True for slo in critical_slos)

        # API Availability should be critical
        names = [slo.name for slo in critical_slos]
        assert "API Availability" in names

    def test_get_by_tags(self):
        """Test getting SLOs by tags."""
        registry = SLORegistry()

        api_slos = registry.get_by_tags(["api"])
        assert len(api_slos) > 0
        assert all(any("api" in tag for tag in slo.tags) for slo in api_slos)

    def test_get_services(self):
        """Test getting list of services."""
        registry = SLORegistry()

        services = registry.get_services()
        assert "api" in services
        assert "browser-pool" in services
        assert "redpanda" in services

    def test_registry_to_dict(self):
        """Test converting registry to dictionary."""
        registry = SLORegistry()

        d = registry.to_dict()
        assert "total_slos" in d
        assert "critical_slos" in d
        assert "services" in d
        assert "slos_by_service" in d

        assert d["total_slos"] > 0
        assert d["critical_slos"] > 0
        assert len(d["services"]) > 0

    def test_singleton_registry(self):
        """Test that get_slo_registry returns singleton."""
        registry1 = get_slo_registry()
        registry2 = get_slo_registry()

        assert registry1 is registry2
