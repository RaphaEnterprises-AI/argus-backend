"""Observability integration plugins.

This module provides integrations for observability, monitoring,
and alerting platforms.

Available plugins:
- SentryPlugin: Error tracking and performance monitoring
- DatadogPlugin: Infrastructure and application monitoring
- NewRelicPlugin: Full-stack observability platform (APM, infrastructure, logs)
- GrafanaPlugin: Grafana dashboards, alerts, and annotations
- SplunkPlugin: Splunk search, alerts, and notable events
- DynatracePlugin: Dynatrace problems, metrics, and events

Usage:
    from src.integrations.plugins.observability import (
        SentryPlugin,
        DatadogPlugin,
        NewRelicPlugin,
        GrafanaPlugin,
        SplunkPlugin,
        DynatracePlugin,
    )

    # Sentry example
    async with SentryPlugin() as plugin:
        credentials = IntegrationCredentials(
            access_token="your-auth-token",
            org_slug="your-org",
        )
        result = await plugin.test_connection(credentials)

    # Datadog example
    async with DatadogPlugin() as plugin:
        credentials = IntegrationCredentials(
            api_key="your-api-key",
            api_secret="your-application-key",
            site="datadoghq.com",
        )
        result = await plugin.test_connection(credentials)

    # New Relic example
    async with NewRelicPlugin() as plugin:
        credentials = IntegrationCredentials(
            api_key="NRAK-...",
            extra={"region": "us"},
        )
        result = await plugin.test_connection(credentials)

    # Grafana example
    async with GrafanaPlugin() as plugin:
        credentials = IntegrationCredentials(
            api_key="your-api-key",
            instance_url="https://your-instance.grafana.net",
        )
        result = await plugin.test_connection(credentials)
"""

from src.integrations.plugins.observability.datadog_plugin import DatadogPlugin
from src.integrations.plugins.observability.dynatrace_plugin import DynatracePlugin
from src.integrations.plugins.observability.grafana_plugin import GrafanaPlugin
from src.integrations.plugins.observability.newrelic_plugin import NewRelicPlugin
from src.integrations.plugins.observability.sentry_plugin import SentryPlugin
from src.integrations.plugins.observability.splunk_plugin import SplunkPlugin

__all__ = [
    "SentryPlugin",
    "DatadogPlugin",
    "NewRelicPlugin",
    "GrafanaPlugin",
    "SplunkPlugin",
    "DynatracePlugin",
]
