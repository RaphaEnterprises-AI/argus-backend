"""Zapier Webhook Integration Plugin.

This plugin provides integration with Zapier via Webhooks by Zapier,
allowing Argus to trigger Zaps with custom payloads.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from src.integrations.base import (
    AuthType,
    ConnectionTestResult,
    FieldDefinition,
    IntegrationCategory,
    IntegrationCredentials,
    IntegrationMetadata,
    IntegrationPlugin,
    SyncResult,
)

logger = logging.getLogger(__name__)


class ZapierPayloadFormat(str, Enum):
    """Payload formats supported by Zapier webhooks."""
    JSON = "json"
    FORM = "form"


@dataclass
class ZapierResponse:
    """Response from a Zapier webhook call."""
    success: bool
    status_code: int
    response_body: Optional[str] = None
    response_data: Optional[dict[str, Any]] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    retry_count: int = 0
    request_id: Optional[str] = None


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    retry_on_status_codes: list[int] = field(
        default_factory=lambda: [408, 429, 500, 502, 503, 504]
    )


class ZapierPlugin(IntegrationPlugin):
    """Zapier Webhook Integration Plugin.

    Integrates with Zapier via "Webhooks by Zapier" to trigger Zaps
    when events occur in Argus.

    Zapier Webhook Setup:
    1. Create a Zap in Zapier
    2. Choose "Webhooks by Zapier" as the trigger
    3. Select "Catch Hook" as the event
    4. Copy the provided webhook URL
    5. Configure this integration with that URL

    Features:
    - Trigger any Zap via webhook
    - Support for JSON and form-encoded payloads
    - Custom payload formatting
    - Robust retry logic

    Usage:
        plugin = ZapierPlugin()
        credentials = IntegrationCredentials(
            webhook_url="https://hooks.zapier.com/hooks/catch/123456/abcdef/",
        )

        # Test connection
        result = await plugin.test_connection(credentials)

        # Trigger a Zap
        response = await plugin.send_event(
            credentials=credentials,
            event_type="test.completed",
            payload={
                "test_name": "Login Flow",
                "status": "passed",
                "duration_ms": 1234,
            },
        )
    """

    ZAPIER_WEBHOOK_DOMAIN = "hooks.zapier.com"
    DEFAULT_TIMEOUT = 30.0
    DEFAULT_RETRY_CONFIG = RetryConfig()

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get metadata about this integration."""
        return IntegrationMetadata(
            slug="zapier",
            name="Zapier",
            description="Trigger Zaps when test events occur using Webhooks by Zapier",
            category=IntegrationCategory.WEBHOOKS,
            auth_type=AuthType.WEBHOOK,
            icon_url="https://cdn.heyargus.ai/integrations/zapier.svg",
            docs_url="https://docs.heyargus.ai/integrations/zapier",
            website_url="https://zapier.com",
            features=[
                "outbound_events",
                "json_payloads",
                "form_payloads",
                "custom_formatting",
                "retry_logic",
                "zap_triggering",
            ],
            required_fields=[
                FieldDefinition(
                    name="webhook_url",
                    label="Zapier Webhook URL",
                    type="url",
                    required=True,
                    placeholder="https://hooks.zapier.com/hooks/catch/123456/abcdef/",
                    help_text="The webhook URL from your Zapier 'Webhooks by Zapier' trigger",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra",
                    label="Custom Fields",
                    type="text",
                    required=False,
                    help_text="Additional fields to include with every webhook (JSON format)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=False,
            sync_resources=[],
        )

    def _validate_zapier_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate that a URL is a valid Zapier webhook URL.

        Args:
            url: The URL to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            parsed = urlparse(url)

            if not parsed.scheme:
                return False, "URL must include a scheme (https://)"

            if parsed.scheme != "https":
                return False, "Zapier webhooks must use HTTPS"

            if not parsed.netloc:
                return False, "URL must include a host"

            # Check if it's a Zapier webhook URL
            if self.ZAPIER_WEBHOOK_DOMAIN not in parsed.netloc:
                logger.warning(
                    f"URL does not appear to be a Zapier webhook: {parsed.netloc}"
                )
                # Still allow it, but log a warning
                return True, None

            # Validate Zapier webhook path format
            if not parsed.path.startswith("/hooks/catch/"):
                return False, "Invalid Zapier webhook URL format. URL should start with /hooks/catch/"

            return True, None

        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"

    def _format_payload(
        self,
        event_type: str,
        payload: dict[str, Any],
        include_metadata: bool = True,
        custom_fields: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Format the payload for Zapier.

        Zapier works best with flat payloads where each field is directly
        accessible in the Zap editor.

        Args:
            event_type: Type of event being sent.
            payload: Event payload data.
            include_metadata: Whether to include Argus metadata.
            custom_fields: Additional custom fields to include.

        Returns:
            Formatted payload dictionary.
        """
        formatted = {}

        # Add metadata if requested
        if include_metadata:
            formatted["argus_event_type"] = event_type
            formatted["argus_timestamp"] = datetime.utcnow().isoformat() + "Z"
            formatted["argus_source"] = "argus"

        # Flatten nested payload for better Zapier compatibility
        def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep).items())
                elif isinstance(v, list):
                    # Convert lists to JSON strings for Zapier
                    items.append((new_key, json.dumps(v)))
                else:
                    items.append((new_key, v))
            return dict(items)

        # Add flattened payload
        flattened = flatten_dict(payload)
        formatted.update(flattened)

        # Add custom fields
        if custom_fields:
            formatted.update(custom_fields)

        return formatted

    async def _execute_with_retry(
        self,
        request_func,
        retry_config: Optional[RetryConfig] = None,
    ) -> ZapierResponse:
        """Execute a request with retry logic.

        Args:
            request_func: Async function that performs the request.
            retry_config: Configuration for retry behavior.

        Returns:
            ZapierResponse with results.
        """
        config = retry_config or self.DEFAULT_RETRY_CONFIG
        last_error = None
        last_status_code = 0

        for attempt in range(config.max_retries + 1):
            start_time = time.time()
            request_id = f"zap-{int(time.time() * 1000)}-{attempt}"

            try:
                response = await request_func()
                latency_ms = (time.time() - start_time) * 1000

                # Parse response
                response_data = None
                try:
                    response_data = response.json()
                except (json.JSONDecodeError, ValueError):
                    pass

                # Success case (Zapier returns 200 on success)
                if response.status_code < 400:
                    return ZapierResponse(
                        success=True,
                        status_code=response.status_code,
                        response_body=response.text,
                        response_data=response_data,
                        latency_ms=latency_ms,
                        retry_count=attempt,
                        request_id=request_id,
                    )

                # Check if we should retry
                last_status_code = response.status_code
                if response.status_code not in config.retry_on_status_codes:
                    return ZapierResponse(
                        success=False,
                        status_code=response.status_code,
                        response_body=response.text,
                        response_data=response_data,
                        latency_ms=latency_ms,
                        error=f"HTTP {response.status_code}: {response.text[:200]}",
                        retry_count=attempt,
                        request_id=request_id,
                    )

                last_error = f"HTTP {response.status_code}"

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {str(e)}"
                last_status_code = 0
            except httpx.ConnectError as e:
                last_error = f"Connection error: {str(e)}"
                last_status_code = 0
            except Exception as e:
                last_error = f"Request failed: {str(e)}"
                last_status_code = 0
                logger.exception("Zapier webhook request failed")

            # Calculate delay for next retry
            if attempt < config.max_retries:
                delay = min(
                    config.initial_delay_seconds * (config.exponential_base ** attempt),
                    config.max_delay_seconds,
                )
                logger.info(
                    f"Zapier webhook attempt {attempt + 1} failed, retrying in {delay:.1f}s: {last_error}"
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        return ZapierResponse(
            success=False,
            status_code=last_status_code,
            error=last_error,
            retry_count=config.max_retries,
        )

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test if the Zapier webhook URL is valid.

        Note: Zapier webhooks don't support HEAD or OPTIONS requests for validation,
        so we validate the URL format and optionally send a test event.

        Args:
            credentials: Credentials containing the webhook_url.

        Returns:
            ConnectionTestResult with success status and details.
        """
        start_time = time.time()

        # Validate URL is provided
        if not credentials.webhook_url:
            return ConnectionTestResult(
                success=False,
                message="Zapier webhook URL is required",
                error_code="MISSING_URL",
            )

        # Validate URL format
        is_valid, error = self._validate_zapier_url(credentials.webhook_url)
        if not is_valid:
            return ConnectionTestResult(
                success=False,
                message=error or "Invalid Zapier webhook URL",
                error_code="INVALID_URL",
            )

        latency_ms = (time.time() - start_time) * 1000

        # URL format is valid
        # Note: We could send a test payload here, but Zapier webhooks always accept
        # any POST request, so format validation is sufficient
        is_zapier_domain = self.ZAPIER_WEBHOOK_DOMAIN in credentials.webhook_url

        return ConnectionTestResult(
            success=True,
            message="Zapier webhook URL format is valid"
            + (" (verified Zapier domain)" if is_zapier_domain else ""),
            latency_ms=latency_ms,
            account_info={
                "url": credentials.webhook_url,
                "is_zapier_domain": is_zapier_domain,
            },
        )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from the integration.

        Note: Zapier webhooks are outbound-only and do not support data sync.

        Args:
            credentials: The credentials to use.
            resources: Not applicable for Zapier webhooks.
            since: Not applicable for Zapier webhooks.
            limit: Not applicable for Zapier webhooks.

        Returns:
            SyncResult indicating sync is not supported.
        """
        return SyncResult(
            success=False,
            message="Zapier webhooks are outbound-only and do not support data sync. "
            "Use Zapier's native integrations to pull data from other services.",
            data_points_synced=0,
        )

    async def send_event(
        self,
        credentials: IntegrationCredentials,
        event_type: str,
        payload: dict[str, Any],
        payload_format: ZapierPayloadFormat = ZapierPayloadFormat.JSON,
        include_metadata: bool = True,
        custom_fields: Optional[dict[str, Any]] = None,
        retry_config: Optional[RetryConfig] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> ZapierResponse:
        """Send an event to trigger a Zap.

        Args:
            credentials: Credentials containing the webhook_url.
            event_type: Type of event being sent (e.g., "test.completed").
            payload: Event payload data.
            payload_format: Format for the payload (json or form).
            include_metadata: Whether to include Argus metadata fields.
            custom_fields: Additional custom fields to include.
            retry_config: Optional retry configuration.
            timeout: Request timeout in seconds.

        Returns:
            ZapierResponse with the result of the request.
        """
        if not credentials.webhook_url:
            return ZapierResponse(
                success=False,
                status_code=0,
                error="Zapier webhook URL is required",
            )

        # Parse custom fields from credentials.extra if available
        extra_custom_fields = {}
        if credentials.extra and credentials.extra.get("custom_fields"):
            try:
                if isinstance(credentials.extra["custom_fields"], str):
                    extra_custom_fields = json.loads(credentials.extra["custom_fields"])
                else:
                    extra_custom_fields = credentials.extra["custom_fields"]
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse custom fields from credentials")

        # Merge custom fields
        merged_custom_fields = {**extra_custom_fields, **(custom_fields or {})}

        # Format the payload
        formatted_payload = self._format_payload(
            event_type=event_type,
            payload=payload,
            include_metadata=include_metadata,
            custom_fields=merged_custom_fields,
        )

        # Build headers
        headers = {
            "User-Agent": "Argus-Zapier/1.0",
        }

        # Define the request function for retry logic
        async def make_request():
            if payload_format == ZapierPayloadFormat.JSON:
                headers["Content-Type"] = "application/json"
                return await self.http_client.post(
                    credentials.webhook_url,
                    json=formatted_payload,
                    headers=headers,
                    timeout=timeout,
                )
            else:
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                return await self.http_client.post(
                    credentials.webhook_url,
                    data=formatted_payload,
                    headers=headers,
                    timeout=timeout,
                )

        logger.info(
            f"Sending Zapier event: {event_type} to {credentials.webhook_url}"
        )

        response = await self._execute_with_retry(make_request, retry_config)

        if response.success:
            logger.info(
                f"Zapier event sent successfully: {event_type} "
                f"(status={response.status_code}, latency={response.latency_ms:.0f}ms)"
            )
        else:
            logger.warning(
                f"Zapier event failed: {event_type} - {response.error} "
                f"(retries={response.retry_count})"
            )

        return response

    async def send_batch_events(
        self,
        credentials: IntegrationCredentials,
        events: list[tuple[str, dict[str, Any]]],
        payload_format: ZapierPayloadFormat = ZapierPayloadFormat.JSON,
        include_metadata: bool = True,
        retry_config: Optional[RetryConfig] = None,
        concurrency_limit: int = 5,
    ) -> list[ZapierResponse]:
        """Send multiple events to trigger Zaps.

        Note: Each event triggers a separate Zap run. For high volumes,
        consider using Zapier's bulk actions instead.

        Args:
            credentials: Credentials containing the webhook_url.
            events: List of (event_type, payload) tuples.
            payload_format: Format for the payloads.
            include_metadata: Whether to include Argus metadata.
            retry_config: Optional retry configuration.
            concurrency_limit: Maximum concurrent requests.

        Returns:
            List of ZapierResponse for each event.
        """
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def send_with_limit(event_type: str, payload: dict[str, Any]):
            async with semaphore:
                return await self.send_event(
                    credentials=credentials,
                    event_type=event_type,
                    payload=payload,
                    payload_format=payload_format,
                    include_metadata=include_metadata,
                    retry_config=retry_config,
                )

        tasks = [
            send_with_limit(event_type, payload)
            for event_type, payload in events
        ]

        return await asyncio.gather(*tasks)

    async def trigger_zap(
        self,
        credentials: IntegrationCredentials,
        data: dict[str, Any],
    ) -> ZapierResponse:
        """Simplified method to trigger a Zap with raw data.

        This method sends data directly without event type formatting,
        useful for simple Zap triggers.

        Args:
            credentials: Credentials containing the webhook_url.
            data: Data to send to the Zap.

        Returns:
            ZapierResponse with the result.
        """
        if not credentials.webhook_url:
            return ZapierResponse(
                success=False,
                status_code=0,
                error="Zapier webhook URL is required",
            )

        try:
            response = await self.http_client.post(
                credentials.webhook_url,
                json=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Argus-Zapier/1.0",
                },
                timeout=self.DEFAULT_TIMEOUT,
            )

            response_data = None
            try:
                response_data = response.json()
            except (json.JSONDecodeError, ValueError):
                pass

            return ZapierResponse(
                success=response.status_code < 400,
                status_code=response.status_code,
                response_body=response.text,
                response_data=response_data,
                error=None if response.status_code < 400 else f"HTTP {response.status_code}",
            )

        except Exception as e:
            logger.exception("Failed to trigger Zap")
            return ZapierResponse(
                success=False,
                status_code=0,
                error=str(e),
            )
