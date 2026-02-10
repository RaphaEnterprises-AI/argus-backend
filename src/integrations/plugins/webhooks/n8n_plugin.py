"""n8n Webhook Integration Plugin.

This plugin provides integration with n8n via the Webhook node,
allowing Argus to trigger n8n workflows with custom payloads.
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


class N8nHttpMethod(str, Enum):
    """HTTP methods supported by n8n webhook nodes."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class N8nAuthType(str, Enum):
    """Authentication types supported by n8n webhooks."""
    NONE = "none"
    BASIC_AUTH = "basic"
    HEADER_AUTH = "header"


@dataclass
class N8nResponse:
    """Response from an n8n webhook call."""
    success: bool
    status_code: int
    response_body: Optional[str] = None
    response_data: Optional[Any] = None  # n8n can return various types
    latency_ms: float = 0.0
    error: Optional[str] = None
    retry_count: int = 0
    workflow_execution_id: Optional[str] = None


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


class N8nPlugin(IntegrationPlugin):
    """n8n Webhook Integration Plugin.

    Integrates with n8n via the Webhook node to trigger workflows
    when events occur in Argus.

    n8n Webhook Setup:
    1. Create a workflow in n8n
    2. Add a "Webhook" node as the trigger
    3. Configure the HTTP method (GET or POST recommended)
    4. Copy the webhook URL (Production or Test URL)
    5. Configure this integration with that URL

    Features:
    - Support for GET and POST webhook methods
    - Optional basic auth or header auth
    - Custom headers and query parameters
    - Robust retry logic
    - Wait for workflow response (synchronous) or fire-and-forget (async)

    Usage:
        plugin = N8nPlugin()
        credentials = IntegrationCredentials(
            webhook_url="https://n8n.example.com/webhook/abc123",
            # Optional authentication
            username="user",  # For basic auth
            password="pass",
            # Or header auth via extra
            extra={"header_name": "X-API-Key", "header_value": "secret"},
        )

        # Test connection
        result = await plugin.test_connection(credentials)

        # Trigger a workflow with POST
        response = await plugin.send_event(
            credentials=credentials,
            event_type="test.completed",
            payload={
                "test_name": "Login Flow",
                "status": "passed",
            },
        )

        # Trigger with GET (for simple triggers)
        response = await plugin.trigger_workflow(
            credentials=credentials,
            method=N8nHttpMethod.GET,
            query_params={"test_id": "123"},
        )
    """

    DEFAULT_TIMEOUT = 60.0  # n8n workflows may take longer to respond
    DEFAULT_RETRY_CONFIG = RetryConfig()

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get metadata about this integration."""
        return IntegrationMetadata(
            slug="n8n",
            name="n8n",
            description="Trigger n8n workflows when test events occur via Webhook nodes",
            category=IntegrationCategory.WEBHOOKS,
            auth_type=AuthType.WEBHOOK,
            icon_url="https://cdn.skopaq.ai/integrations/n8n.svg",
            docs_url="https://docs.skopaq.ai/integrations/n8n",
            website_url="https://n8n.io",
            features=[
                "outbound_events",
                "get_webhooks",
                "post_webhooks",
                "basic_auth",
                "header_auth",
                "query_parameters",
                "custom_headers",
                "retry_logic",
                "sync_response",
            ],
            required_fields=[
                FieldDefinition(
                    name="webhook_url",
                    label="n8n Webhook URL",
                    type="url",
                    required=True,
                    placeholder="https://n8n.example.com/webhook/abc123",
                    help_text="The webhook URL from your n8n Webhook node",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="username",
                    label="Basic Auth Username",
                    type="text",
                    required=False,
                    help_text="Username for Basic Auth (if configured in n8n)",
                ),
                FieldDefinition(
                    name="password",
                    label="Basic Auth Password",
                    type="password",
                    required=False,
                    help_text="Password for Basic Auth (if configured in n8n)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=False,
            sync_resources=[],
        )

    def _validate_n8n_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate that a URL is a valid n8n webhook URL.

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

            # Check for common n8n webhook patterns
            # n8n webhooks typically have /webhook/ or /webhook-test/ in the path
            if "/webhook" not in parsed.path.lower():
                logger.warning(
                    f"URL does not appear to be an n8n webhook (missing /webhook/ in path): {url}"
                )
                # Still allow it, as users might have custom routing

            return True, None

        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"

    def _build_auth_headers(
        self,
        credentials: IntegrationCredentials,
    ) -> dict[str, str]:
        """Build authentication headers.

        Args:
            credentials: Credentials potentially containing auth info.

        Returns:
            Dictionary of authentication headers.
        """
        headers = {}

        # Basic auth
        if credentials.username and credentials.password:
            import base64
            auth_string = f"{credentials.username}:{credentials.password}"
            auth_bytes = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
            headers["Authorization"] = f"Basic {auth_bytes}"

        # Header auth (custom header from extra)
        if credentials.extra:
            header_name = credentials.extra.get("header_name")
            header_value = credentials.extra.get("header_value")
            if header_name and header_value:
                headers[header_name] = header_value

        return headers

    def _build_request_headers(
        self,
        credentials: IntegrationCredentials,
        content_type: Optional[str] = None,
        custom_headers: Optional[dict[str, str]] = None,
    ) -> dict[str, str]:
        """Build complete request headers.

        Args:
            credentials: Credentials for authentication.
            content_type: Optional content type header.
            custom_headers: Optional custom headers.

        Returns:
            Complete headers dictionary.
        """
        headers = {
            "User-Agent": "Argus-N8n/1.0",
            "Accept": "application/json",
        }

        if content_type:
            headers["Content-Type"] = content_type

        # Add auth headers
        auth_headers = self._build_auth_headers(credentials)
        headers.update(auth_headers)

        # Add custom headers
        if custom_headers:
            headers.update(custom_headers)

        return headers

    async def _execute_with_retry(
        self,
        request_func,
        retry_config: Optional[RetryConfig] = None,
    ) -> N8nResponse:
        """Execute a request with retry logic.

        Args:
            request_func: Async function that performs the request.
            retry_config: Configuration for retry behavior.

        Returns:
            N8nResponse with results.
        """
        config = retry_config or self.DEFAULT_RETRY_CONFIG
        last_error = None
        last_status_code = 0

        for attempt in range(config.max_retries + 1):
            start_time = time.time()

            try:
                response = await request_func()
                latency_ms = (time.time() - start_time) * 1000

                # Parse response
                response_data = None
                try:
                    response_data = response.json()
                except (json.JSONDecodeError, ValueError):
                    # n8n might return plain text or empty response
                    response_data = response.text if response.text else None

                # Extract workflow execution ID if present
                execution_id = None
                if isinstance(response_data, dict):
                    execution_id = response_data.get("executionId")

                # Success case
                if response.status_code < 400:
                    return N8nResponse(
                        success=True,
                        status_code=response.status_code,
                        response_body=response.text,
                        response_data=response_data,
                        latency_ms=latency_ms,
                        retry_count=attempt,
                        workflow_execution_id=execution_id,
                    )

                # Check if we should retry
                last_status_code = response.status_code
                if response.status_code not in config.retry_on_status_codes:
                    return N8nResponse(
                        success=False,
                        status_code=response.status_code,
                        response_body=response.text,
                        response_data=response_data,
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
                logger.exception("n8n webhook request failed")

            # Calculate delay for next retry
            if attempt < config.max_retries:
                delay = min(
                    config.initial_delay_seconds * (config.exponential_base ** attempt),
                    config.max_delay_seconds,
                )
                logger.info(
                    f"n8n webhook attempt {attempt + 1} failed, retrying in {delay:.1f}s: {last_error}"
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        return N8nResponse(
            success=False,
            status_code=last_status_code,
            error=last_error,
            retry_count=config.max_retries,
        )

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test if the n8n webhook URL is reachable.

        Attempts a HEAD request to verify the endpoint exists.
        Falls back to GET if HEAD is not supported.

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
                message="n8n webhook URL is required",
                error_code="MISSING_URL",
            )

        # Validate URL format
        is_valid, error = self._validate_n8n_url(credentials.webhook_url)
        if not is_valid:
            return ConnectionTestResult(
                success=False,
                message=error or "Invalid n8n webhook URL",
                error_code="INVALID_URL",
            )

        # Build headers with auth
        headers = self._build_request_headers(credentials)

        try:
            # Try HEAD request first
            response = await self.http_client.head(
                credentials.webhook_url,
                headers=headers,
                timeout=10.0,
                follow_redirects=True,
            )

            latency_ms = (time.time() - start_time) * 1000

            # n8n webhook nodes typically respond to HEAD with 200 or 405
            if response.status_code in (200, 204, 405):
                return ConnectionTestResult(
                    success=True,
                    message=f"n8n webhook endpoint is reachable (HTTP {response.status_code})",
                    latency_ms=latency_ms,
                    account_info={
                        "url": credentials.webhook_url,
                        "has_basic_auth": bool(credentials.username and credentials.password),
                        "has_header_auth": bool(
                            credentials.extra
                            and credentials.extra.get("header_name")
                        ),
                    },
                )

            # Try GET as fallback (some n8n setups might not support HEAD)
            if response.status_code == 404:
                response = await self.http_client.get(
                    credentials.webhook_url,
                    headers=headers,
                    timeout=10.0,
                    follow_redirects=True,
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code < 400:
                    return ConnectionTestResult(
                        success=True,
                        message=f"n8n webhook endpoint is reachable (HTTP GET {response.status_code})",
                        latency_ms=latency_ms,
                        account_info={
                            "url": credentials.webhook_url,
                            "method_tested": "GET",
                        },
                    )

            return ConnectionTestResult(
                success=False,
                message=f"n8n webhook endpoint returned HTTP {response.status_code}",
                latency_ms=latency_ms,
                error_code=f"HTTP_{response.status_code}",
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection to n8n webhook endpoint timed out",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="TIMEOUT",
            )
        except httpx.ConnectError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Failed to connect to n8n webhook endpoint: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="CONNECTION_ERROR",
            )
        except Exception as e:
            logger.exception("n8n webhook connection test failed")
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

        Note: n8n webhooks are outbound-only triggers and do not support data sync.
        Use n8n's native integrations to pull data from other services.

        Args:
            credentials: The credentials to use.
            resources: Not applicable for n8n webhooks.
            since: Not applicable for n8n webhooks.
            limit: Not applicable for n8n webhooks.

        Returns:
            SyncResult indicating sync is not supported.
        """
        return SyncResult(
            success=False,
            message="n8n webhooks are outbound triggers and do not support data sync. "
            "Use n8n's native nodes to pull data from other services.",
            data_points_synced=0,
        )

    async def send_event(
        self,
        credentials: IntegrationCredentials,
        event_type: str,
        payload: dict[str, Any],
        method: N8nHttpMethod = N8nHttpMethod.POST,
        query_params: Optional[dict[str, str]] = None,
        custom_headers: Optional[dict[str, str]] = None,
        retry_config: Optional[RetryConfig] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> N8nResponse:
        """Send an event to trigger an n8n workflow.

        Args:
            credentials: Credentials containing webhook_url and optional auth.
            event_type: Type of event being sent (e.g., "test.completed").
            payload: Event payload data.
            method: HTTP method to use (default: POST).
            query_params: Optional query parameters to include.
            custom_headers: Optional custom headers to include.
            retry_config: Optional retry configuration.
            timeout: Request timeout in seconds.

        Returns:
            N8nResponse with the result of the request.
        """
        if not credentials.webhook_url:
            return N8nResponse(
                success=False,
                status_code=0,
                error="n8n webhook URL is required",
            )

        # Build the full payload with event metadata
        full_payload = {
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "argus",
            "data": payload,
        }

        # Build headers
        content_type = "application/json" if method != N8nHttpMethod.GET else None
        headers = self._build_request_headers(
            credentials=credentials,
            content_type=content_type,
            custom_headers=custom_headers,
        )

        # Define the request function for retry logic
        async def make_request():
            if method == N8nHttpMethod.GET:
                # For GET requests, send data as query params
                params = query_params or {}
                params["event"] = event_type
                params["timestamp"] = datetime.utcnow().isoformat() + "Z"
                # Flatten payload for query params
                for key, value in payload.items():
                    if isinstance(value, (dict, list)):
                        params[key] = json.dumps(value)
                    else:
                        params[key] = str(value)

                return await self.http_client.get(
                    credentials.webhook_url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )
            else:
                return await self.http_client.request(
                    method=method.value,
                    url=credentials.webhook_url,
                    json=full_payload,
                    params=query_params,
                    headers=headers,
                    timeout=timeout,
                )

        logger.info(
            f"Sending n8n event: {event_type} via {method.value} to {credentials.webhook_url}"
        )

        response = await self._execute_with_retry(make_request, retry_config)

        if response.success:
            logger.info(
                f"n8n event sent successfully: {event_type} "
                f"(status={response.status_code}, latency={response.latency_ms:.0f}ms"
                f"{f', execution_id={response.workflow_execution_id}' if response.workflow_execution_id else ''})"
            )
        else:
            logger.warning(
                f"n8n event failed: {event_type} - {response.error} "
                f"(retries={response.retry_count})"
            )

        return response

    async def trigger_workflow(
        self,
        credentials: IntegrationCredentials,
        method: N8nHttpMethod = N8nHttpMethod.POST,
        payload: Optional[dict[str, Any]] = None,
        query_params: Optional[dict[str, str]] = None,
        custom_headers: Optional[dict[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> N8nResponse:
        """Simplified method to trigger an n8n workflow.

        This method sends data directly without event type formatting,
        useful for simple workflow triggers.

        Args:
            credentials: Credentials containing webhook_url and optional auth.
            method: HTTP method to use (default: POST).
            payload: Data to send (for POST/PUT/PATCH).
            query_params: Query parameters (for GET or in addition to body).
            custom_headers: Optional custom headers.
            timeout: Request timeout in seconds.

        Returns:
            N8nResponse with the result.
        """
        if not credentials.webhook_url:
            return N8nResponse(
                success=False,
                status_code=0,
                error="n8n webhook URL is required",
            )

        headers = self._build_request_headers(
            credentials=credentials,
            content_type="application/json" if payload and method != N8nHttpMethod.GET else None,
            custom_headers=custom_headers,
        )

        try:
            start_time = time.time()

            if method == N8nHttpMethod.GET:
                response = await self.http_client.get(
                    credentials.webhook_url,
                    params=query_params,
                    headers=headers,
                    timeout=timeout,
                )
            else:
                response = await self.http_client.request(
                    method=method.value,
                    url=credentials.webhook_url,
                    json=payload,
                    params=query_params,
                    headers=headers,
                    timeout=timeout,
                )

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            response_data = None
            try:
                response_data = response.json()
            except (json.JSONDecodeError, ValueError):
                response_data = response.text if response.text else None

            execution_id = None
            if isinstance(response_data, dict):
                execution_id = response_data.get("executionId")

            return N8nResponse(
                success=response.status_code < 400,
                status_code=response.status_code,
                response_body=response.text,
                response_data=response_data,
                latency_ms=latency_ms,
                workflow_execution_id=execution_id,
                error=None if response.status_code < 400 else f"HTTP {response.status_code}",
            )

        except Exception as e:
            logger.exception("Failed to trigger n8n workflow")
            return N8nResponse(
                success=False,
                status_code=0,
                error=str(e),
            )

    async def send_batch_events(
        self,
        credentials: IntegrationCredentials,
        events: list[tuple[str, dict[str, Any]]],
        method: N8nHttpMethod = N8nHttpMethod.POST,
        retry_config: Optional[RetryConfig] = None,
        concurrency_limit: int = 5,
    ) -> list[N8nResponse]:
        """Send multiple events to trigger n8n workflows.

        Note: Each event triggers a separate workflow execution.

        Args:
            credentials: Credentials containing webhook_url and optional auth.
            events: List of (event_type, payload) tuples.
            method: HTTP method to use (default: POST).
            retry_config: Optional retry configuration.
            concurrency_limit: Maximum concurrent requests.

        Returns:
            List of N8nResponse for each event.
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
