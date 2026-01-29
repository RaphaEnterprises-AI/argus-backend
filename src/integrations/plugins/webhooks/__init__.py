"""Webhook integration plugins.

This module provides webhook-based integrations for triggering external
automation workflows when events occur in Argus.

Available plugins:
- WebhookPlugin: Generic webhook with HMAC-SHA256 signature support
- ZapierPlugin: Trigger Zaps via Webhooks by Zapier
- N8nPlugin: Trigger n8n workflows via Webhook nodes

All plugins support:
- Outbound event sending
- Robust retry logic with exponential backoff
- Custom headers and payload formatting
- Connection testing

Usage:
    from src.integrations.plugins.webhooks import WebhookPlugin, ZapierPlugin, N8nPlugin
    from src.integrations.base import IntegrationCredentials

    # Generic webhook
    webhook = WebhookPlugin()
    await webhook.send_event(
        credentials=IntegrationCredentials(
            webhook_url="https://example.com/webhook",
            webhook_secret="optional-secret",
        ),
        event_type="test.completed",
        payload={"test_id": "123", "status": "passed"},
    )

    # Zapier
    zapier = ZapierPlugin()
    await zapier.send_event(
        credentials=IntegrationCredentials(
            webhook_url="https://hooks.zapier.com/hooks/catch/123456/abcdef/",
        ),
        event_type="test.failed",
        payload={"test_name": "Login Flow", "error": "Element not found"},
    )

    # n8n
    n8n = N8nPlugin()
    await n8n.send_event(
        credentials=IntegrationCredentials(
            webhook_url="https://n8n.example.com/webhook/abc123",
            username="user",  # Optional basic auth
            password="pass",
        ),
        event_type="test.completed",
        payload={"test_name": "Checkout Flow", "duration_ms": 5000},
    )
"""

from src.integrations.plugins.webhooks.n8n_plugin import (
    N8nHttpMethod,
    N8nPlugin,
    N8nResponse,
)
from src.integrations.plugins.webhooks.webhook_plugin import (
    RetryConfig,
    SignatureHeader,
    WebhookMethod,
    WebhookPlugin,
    WebhookResponse,
)
from src.integrations.plugins.webhooks.zapier_plugin import (
    ZapierPayloadFormat,
    ZapierPlugin,
    ZapierResponse,
)

__all__ = [
    # Generic Webhook
    "WebhookPlugin",
    "WebhookResponse",
    "WebhookMethod",
    "SignatureHeader",
    "RetryConfig",
    # Zapier
    "ZapierPlugin",
    "ZapierResponse",
    "ZapierPayloadFormat",
    # n8n
    "N8nPlugin",
    "N8nResponse",
    "N8nHttpMethod",
]
