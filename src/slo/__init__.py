"""SLO/SLI Framework for Argus E2E Testing Platform.

This module provides:
- SLO definitions with availability, latency, and error rate targets
- SLI measurement and tracking
- Error budget calculation and burn rate monitoring
- Integration with Prometheus metrics

RAP-329: Define SLO/SLI Framework

Usage:
    from src.slo import SLORegistry, SLITracker, get_slo_registry

    # Get the SLO registry
    registry = get_slo_registry()

    # Get SLO for a specific service
    api_slo = registry.get_slo("api")

    # Track SLIs
    tracker = SLITracker()
    compliance = await tracker.calculate_compliance("api", window_hours=24)
"""

from .definitions import (
    SLO,
    SLOCategory,
    SLORegistry,
    get_slo_registry,
    # Individual SLO definitions
    API_SLOS,
    BROWSER_POOL_SLOS,
    DATA_LAYER_SLOS,
    LLM_SLOS,
)
from .tracker import (
    SLITracker,
    SLIMetric,
    SLOCompliance,
    ErrorBudget,
    get_sli_tracker,
)

__all__ = [
    # Definitions
    "SLO",
    "SLOCategory",
    "SLORegistry",
    "get_slo_registry",
    "API_SLOS",
    "BROWSER_POOL_SLOS",
    "DATA_LAYER_SLOS",
    "LLM_SLOS",
    # Tracker
    "SLITracker",
    "SLIMetric",
    "SLOCompliance",
    "ErrorBudget",
    "get_sli_tracker",
]
