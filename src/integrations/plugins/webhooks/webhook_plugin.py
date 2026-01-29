"""Generic Webhook Integration Plugin.

This plugin provides a generic webhook integration that can send events
to any HTTP endpoint with optional HMAC-SHA256 signature verification.
"""

import asyncio
import hashlib
import hmac
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


class WebhookMethod(str, Enum):
    """HTTP methods supported for webhooks."""
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"


class SignatureHeader(str, Enum):
    """Common signature header formats."""
    X_SIGNATURE_256 = "X-Signature-256"
    X_HUB_SIGNATURE_256 = "X-Hub-Signature-256"
    X_WEBHOOK_SIGNATURE = "X-Webhook-Signature"
    AUTHORIZATION = "Authorization"


@dataclass
class WebhookResponse:
    """Response from a webhook call."""
    success: bool
    status_code: int
    response_body: Optional[str] = None
    response_headers: dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None
    retry_count: int = 0


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


class WebhookPlugin(IntegrationPlugin):
    """Generic Webhook Integration Plugin.

    Provides outbound webhook functionality with:
    - HMAC-SHA256 signature support
    - Configurable HTTP methods
    - Robust retry logic with exponential backoff
    - Custom headers and payload formatting

    Usage:
        plugin = WebhookPlugin()
        credentials = IntegrationCredentials(
            webhook_url="https://example.com/webhook",
            webhook_secret="my-secret-key",  # Optional
        )

        # Test connection
        result = await plugin.test_connection(credentials)

        # Send an event
        response = await plugin.send_event(
            credentials=credentials,
            event_type="test.completed",
            payload={"test_id": "123", "status": "passed"},
        )
    """

    DEFAULT_TIMEOUT = 30.0
    DEFAULT_RETRY_CONFIG = RetryConfig()

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get metadata about this integration."""
        return IntegrationMetadata(
            slug="webhook",
            name="Generic Webhook",
            description="Send events to any HTTP endpoint with optional HMAC-SHA256 signatures",
            category=IntegrationCategory.WEBHOOKS,
            auth_type=AuthType.WEBHOOK,
            icon_url="https://cdn.heyargus.ai/integrations/webhook.svg",
            docs_url="https://docs.heyargus.ai/integrations/webhook",
            features=[
                "outbound_events",
                "hmac_signatures",
                "custom_headers",
                "retry_logic",
                "configurable_methods",
            ],
            required_fields=[
                FieldDefinition(
                    name="webhook_url",
                    label="Webhook URL",
                    type="url",
                    required=True,
                    placeholder="https://example.com/webhook",
                    help_text="The URL to send webhook events to",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="webhook_secret",
                    label="Webhook Secret",
                    type="password",
                    required=False,
                    placeholder="Enter secret for HMAC signing",
                    help_text="Secret key for HMAC-SHA256 signature verification",
                ),
            ],
            supports_webhooks=True,
            supports_sync=False,
            sync_resources=[],
        )

    def _validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate that a URL is well-formed and uses HTTPS.

        Args:
            url: The URL to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            parsed = urlparse(url)

            if not parsed.scheme:
                return False, "URL must include a scheme (http:// or https://)"

            if parsed.scheme not in ("http", "https"):
                return False, f"Invalid URL scheme: {parsed.scheme}. Use http or https."

            if not parsed.netloc:
                return False, "URL must include a host"

            return True, None

        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"

    def _compute_signature(
        self,
        payload: bytes,
        secret: str,
        algorithm: str = "sha256",
    ) -> str:
        """Compute HMAC signature for a payload.

        Args:
            payload: The payload bytes to sign.
            secret: The secret key for signing.
            algorithm: Hash algorithm (default: sha256).

        Returns:
            Hex-encoded signature with algorithm prefix.
        """
        signature = hmac.new(
            secret.encode("utf-8"),
            payload,
            getattr(hashlib, algorithm),
        ).hexdigest()

        return f"{algorithm}={signature}"

    def _build_headers(
        self,
        credentials: IntegrationCredentials,
        payload: bytes,
        event_type: Optional[str] = None,
        custom_headers: Optional[dict[str, str]] = None,
        signature_header: SignatureHeader = SignatureHeader.X_SIGNATURE_256,
    ) -> dict[str, str]:
        """Build headers for the webhook request.

        Args:
            credentials: The credentials containing optional webhook_secret.
            payload: The payload bytes for signature computation.
            event_type: Optional event type to include in headers.
            custom_headers: Optional custom headers to include.
            signature_header: Which header to use for the signature.

        Returns:
            Dictionary of headers.
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Argus-Webhook/1.0",
            "X-Request-Id": f"argus-{int(time.time() * 1000)}",
        }

        if event_type:
            headers["X-Event-Type"] = event_type

        # Add timestamp for replay protection
        timestamp = str(int(time.time()))
        headers["X-Timestamp"] = timestamp

        # Add signature if secret is provided
        if credentials.webhook_secret:
            # Include timestamp in signature payload for replay protection
            signature_payload = timestamp.encode("utf-8") + b"." + payload
            signature = self._compute_signature(
                signature_payload,
                credentials.webhook_secret,
            )
            headers[signature_header.value] = signature

        # Add custom headers (override defaults if specified)
        if custom_headers:
            headers.update(custom_headers)

        return headers

    async def _execute_with_retry(
        self,
        request_func,
        retry_config: Optional[RetryConfig] = None,
    ) -> WebhookResponse:
        """Execute a request with retry logic.

        Args:
            request_func: Async function that performs the request.
            retry_config: Configuration for retry behavior.

        Returns:
            WebhookResponse with results.
        """
        config = retry_config or self.DEFAULT_RETRY_CONFIG
        last_error = None
        last_status_code = 0

        for attempt in range(config.max_retries + 1):
            start_time = time.time()

            try:
                response = await request_func()
                latency_ms = (time.time() - start_time) * 1000

                # Success case
                if response.status_code < 400:
                    return WebhookResponse(
                        success=True,
                        status_code=response.status_code,
                        response_body=response.text,
                        response_headers=dict(response.headers),
                        latency_ms=latency_ms,
                        retry_count=attempt,
                    )

                # Check if we should retry
                last_status_code = response.status_code
                if response.status_code not in config.retry_on_status_codes:
                    return WebhookResponse(
                        success=False,
                        status_code=response.status_code,
                        response_body=response.text,
                        response_headers=dict(response.headers),
                        latency_ms=latency_ms,
                        error=f"HTTP {response.status_code}: {response.text[:200]}",
                        retry_count=attempt,
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
                logger.exception("Webhook request failed")

            # Calculate delay for next retry
            if attempt < config.max_retries:
                delay = min(
                    config.initial_delay_seconds * (config.exponential_base ** attempt),
                    config.max_delay_seconds,
                )
                logger.info(
                    f"Webhook attempt {attempt + 1} failed, retrying in {delay:.1f}s: {last_error}"
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        return WebhookResponse(
            success=False,
            status_code=last_status_code,
            error=last_error,
            retry_count=config.max_retries,
        )

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test if the webhook URL is reachable.

        Performs a HEAD request to verify the endpoint is accessible.
        Note: Some webhook endpoints may not support HEAD requests,
        so a 405 Method Not Allowed is also considered valid.

        Args:
            credentials: Credentials containing the webhook_url.

        Returns:
            ConnectionTestResult with success status and details.
        """
        start_time = time.time()

        # Validate URL format
        if not credentials.webhook_url:
            return ConnectionTestResult(
                success=False,
                message="Webhook URL is required",
                error_code="MISSING_URL",
            )

        is_valid, error = self._validate_url(credentials.webhook_url)
        if not is_valid:
            return ConnectionTestResult(
                success=False,
                message=error or "Invalid webhook URL",
                error_code="INVALID_URL",
            )

        try:
            # Try HEAD request first
            response = await self.http_client.head(
                credentials.webhook_url,
                timeout=10.0,
                follow_redirects=True,
            )

            latency_ms = (time.time() - start_time) * 1000

            # HEAD returning 2xx, 3xx, or 405 (method not allowed) is acceptable
            if response.status_code < 400 or response.status_code == 405:
                return ConnectionTestResult(
                    success=True,
                    message=f"Webhook endpoint is reachable (HTTP {response.status_code})",
                    latency_ms=latency_ms,
                    account_info={
                        "url": credentials.webhook_url,
                        "has_secret": bool(credentials.webhook_secret),
                    },
                )

            return ConnectionTestResult(
                success=False,
                message=f"Webhook endpoint returned HTTP {response.status_code}",
                latency_ms=latency_ms,
                error_code=f"HTTP_{response.status_code}",
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection to webhook endpoint timed out",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="TIMEOUT",
            )
        except httpx.ConnectError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Failed to connect to webhook endpoint: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="CONNECTION_ERROR",
            )
        except Exception as e:
            logger.exception("Webhook connection test failed")
            return ConnectionTestResult(
                success=False,
                message=f"Connection test failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="UNKNOWN_ERROR",
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from the integration.

        Note: Generic webhooks are outbound-only and do not support data sync.

        Args:
            credentials: The credentials to use.
            resources: Not applicable for webhooks.
            since: Not applicable for webhooks.
            limit: Not applicable for webhooks.

        Returns:
            SyncResult indicating sync is not supported.
        """
        return SyncResult(
            success=False,
            message="Generic webhooks are outbound-only and do not support data sync",
            data_points_synced=0,
        )

    async def send_event(
        self,
        credentials: IntegrationCredentials,
        event_type: str,
        payload: dict[str, Any],
        method: WebhookMethod = WebhookMethod.POST,
        custom_headers: Optional[dict[str, str]] = None,
        retry_config: Optional[RetryConfig] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> WebhookResponse:
        """Send an event to the webhook endpoint.

        Args:
            credentials: Credentials containing webhook_url and optional webhook_secret.
            event_type: Type of event being sent (e.g., "test.completed").
            payload: Event payload data.
            method: HTTP method to use (default: POST).
            custom_headers: Optional custom headers to include.
            retry_config: Optional retry configuration.
            timeout: Request timeout in seconds.

        Returns:
            WebhookResponse with the result of the request.
        """
        if not credentials.webhook_url:
            return WebhookResponse(
                success=False,
                status_code=0,
                error="Webhook URL is required",
            )

        # Build the full payload with event metadata
        full_payload = {
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": payload,
        }

        # Serialize payload
        payload_bytes = json.dumps(full_payload, separators=(",", ":")).encode("utf-8")

        # Build headers
        headers = self._build_headers(
            credentials=credentials,
            payload=payload_bytes,
            event_type=event_type,
            custom_headers=custom_headers,
        )

        # Define the request function for retry logic
        async def make_request():
            return await self.http_client.request(
                method=method.value,
                url=credentials.webhook_url,
                content=payload_bytes,
                headers=headers,
                timeout=timeout,
            )

        logger.info(
            f"Sending webhook event: {event_type} to {credentials.webhook_url}"
        )

        response = await self._execute_with_retry(make_request, retry_config)

        if response.success:
            logger.info(
                f"Webhook event sent successfully: {event_type} "
                f"(status={response.status_code}, latency={response.latency_ms:.0f}ms)"
            )
        else:
            logger.warning(
                f"Webhook event failed: {event_type} - {response.error} "
                f"(retries={response.retry_count})"
            )

        return response

    async def send_batch_events(
        self,
        credentials: IntegrationCredentials,
        events: list[tuple[str, dict[str, Any]]],
        method: WebhookMethod = WebhookMethod.POST,
        retry_config: Optional[RetryConfig] = None,
        concurrency_limit: int = 5,
    ) -> list[WebhookResponse]:
        """Send multiple events to the webhook endpoint.

        Args:
            credentials: Credentials containing webhook_url and optional webhook_secret.
            events: List of (event_type, payload) tuples.
            method: HTTP method to use (default: POST).
            retry_config: Optional retry configuration.
            concurrency_limit: Maximum concurrent requests.

        Returns:
            List of WebhookResponse for each event.
        """
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def send_with_limit(event_type: str, payload: dict[str, Any]):
            async with semaphore:
                return await self.send_event(
                    credentials=credentials,
                    event_type=event_type,
                    payload=payload,
                    method=method,
                    retry_config=retry_config,
                )

        tasks = [
            send_with_limit(event_type, payload)
            for event_type, payload in events
        ]

        return await asyncio.gather(*tasks)
