"""Incident management integration plugins.

This module provides integration plugins for incident management platforms
including alerting, on-call scheduling, and incident response tools.

Available plugins:
- PagerDutyPlugin: PagerDuty incident management and alerting
- OpsgeniePlugin: Opsgenie alert and incident management (Atlassian)
- IncidentIOPlugin: incident.io modern incident management
- SpikePlugin: Spike.sh incident management
- VictorOpsPlugin: Splunk On-Call (VictorOps) incident management
"""

from src.integrations.plugins.incident_management.incident_io_plugin import (
    IncidentIOPlugin,
)
from src.integrations.plugins.incident_management.opsgenie_plugin import OpsgeniePlugin
from src.integrations.plugins.incident_management.pagerduty_plugin import (
    PagerDutyPlugin,
)
from src.integrations.plugins.incident_management.spike_plugin import SpikePlugin
from src.integrations.plugins.incident_management.victorops_plugin import VictorOpsPlugin

__all__ = [
    "PagerDutyPlugin",
    "OpsgeniePlugin",
    "IncidentIOPlugin",
    "SpikePlugin",
    "VictorOpsPlugin",
]
