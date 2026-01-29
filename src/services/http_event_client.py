"""
HTTP Event Client for External Deployments

When deployed outside K8s (e.g., Railway), use this HTTP client
to publish events to the HTTP Event Gateway running inside K8s.

This solves the critical blocker where Railway cannot directly
connect to Redpanda/Kafka running inside the K8s cluster.

Usage:
    from src.services.http_event_client import get_http_event_client

    client = get_http_event_client()
    await client.emit_test_executed(
        org_id="org-123",
        project_id="project-456",
        test_id="test-789",
        test_name="Login Flow",
        status="passed",
        duration_ms=1500,
    )
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
import structlog

logger = structlog.get_logger(__name__)


class HttpEventClient:
    """
    HTTP client for publishing events via the HTTP Event Gateway.

    Used by Railway (external) to publish events to K8s-hosted Kafka.
    """

    def __init__(
        self,
        gateway_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the HTTP event client.

        Args:
            gateway_url: Base URL of the HTTP Event Gateway
                         (e.g., https://api.argus.dev or K8s internal URL)
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self._gateway_url = gateway_url or os.getenv(
            "EVENT_GATEWAY_URL",
            os.getenv("BACKEND_URL", "http://localhost:8000")
        )
        self._api_key = api_key or os.getenv("ARGUS_API_KEY")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            self._client = httpx.AsyncClient(
                base_url=self._gateway_url,
                headers=headers,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def publish(
        self,
        event_type: str,
        data: dict[str, Any],
        org_id: str,
        project_id: str | None = None,
        user_id: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        source: str = "http-client",
    ) -> dict[str, Any] | None:
        """
        Publish an event via HTTP.

        Args:
            event_type: Event type (e.g., "test.executed")
            data: Event payload
            org_id: Organization ID
            project_id: Optional project ID
            user_id: Optional user ID
            correlation_id: Optional correlation ID
            causation_id: Optional causation ID
            source: Event source identifier

        Returns:
            Response data if successful, None if failed
        """
        client = await self._get_client()

        payload = {
            "org_id": org_id,
            "data": data,
            "source": source,
        }
        if project_id:
            payload["project_id"] = project_id
        if user_id:
            payload["user_id"] = user_id
        if correlation_id:
            payload["correlation_id"] = correlation_id
        if causation_id:
            payload["causation_id"] = causation_id

        try:
            response = await client.post(
                f"/api/events/{event_type}",
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            logger.info(
                "Event published via HTTP",
                event_type=event_type,
                event_id=result.get("event_id"),
                org_id=org_id,
            )
            return result

        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP event publish failed",
                event_type=event_type,
                status_code=e.response.status_code,
                error=str(e),
            )
            return None
        except Exception as e:
            logger.error(
                "HTTP event publish error",
                event_type=event_type,
                error=str(e),
            )
            return None

    async def emit_test_executed(
        self,
        org_id: str,
        test_id: str,
        test_name: str,
        status: str,
        duration_ms: int,
        project_id: str | None = None,
        user_id: str | None = None,
        steps_count: int | None = None,
        assertions_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Convenience method for test.executed events."""
        data = {
            "test_id": test_id,
            "test_name": test_name,
            "status": status,
            "duration_ms": duration_ms,
        }
        if steps_count is not None:
            data["steps_count"] = steps_count
        if assertions_count is not None:
            data["assertions_count"] = assertions_count
        if metadata:
            data["metadata"] = metadata

        return await self.publish(
            event_type="test.executed",
            data=data,
            org_id=org_id,
            project_id=project_id,
            user_id=user_id,
        )

    async def emit_test_failed(
        self,
        org_id: str,
        test_id: str,
        test_name: str,
        error_message: str,
        failure_type: str,
        project_id: str | None = None,
        user_id: str | None = None,
        stack_trace: str | None = None,
        screenshot_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Convenience method for test.failed events."""
        data = {
            "test_id": test_id,
            "test_name": test_name,
            "error_message": error_message,
            "failure_type": failure_type,
        }
        if stack_trace:
            data["stack_trace"] = stack_trace
        if screenshot_url:
            data["screenshot_url"] = screenshot_url
        if metadata:
            data["metadata"] = metadata

        return await self.publish(
            event_type="test.failed",
            data=data,
            org_id=org_id,
            project_id=project_id,
            user_id=user_id,
        )

    async def emit_healing_requested(
        self,
        org_id: str,
        test_id: str,
        failure_id: str,
        failure_type: str,
        project_id: str | None = None,
        user_id: str | None = None,
        original_selector: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Convenience method for healing.requested events."""
        data = {
            "test_id": test_id,
            "failure_id": failure_id,
            "failure_type": failure_type,
        }
        if original_selector:
            data["original_selector"] = original_selector
        if context:
            data["context"] = context

        return await self.publish(
            event_type="healing.requested",
            data=data,
            org_id=org_id,
            project_id=project_id,
            user_id=user_id,
            causation_id=failure_id,
        )

    async def emit_codebase_ingested(
        self,
        org_id: str,
        codebase_id: str,
        repo_url: str,
        branch: str,
        commit_sha: str,
        files_count: int,
        project_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Convenience method for codebase.ingested events."""
        return await self.publish(
            event_type="codebase.ingested",
            data={
                "codebase_id": codebase_id,
                "repo_url": repo_url,
                "branch": branch,
                "commit_sha": commit_sha,
                "files_count": files_count,
            },
            org_id=org_id,
            project_id=project_id,
            user_id=user_id,
        )

    async def health_check(self) -> dict[str, Any]:
        """Check health of the HTTP Event Gateway."""
        client = await self._get_client()

        try:
            response = await client.get("/api/events/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Event gateway health check failed", error=str(e))
            return {
                "healthy": False,
                "error": str(e),
            }


# =============================================================================
# Global Instance
# =============================================================================

_http_client: HttpEventClient | None = None


def get_http_event_client() -> HttpEventClient:
    """Get the global HTTP event client instance."""
    global _http_client
    if _http_client is None:
        _http_client = HttpEventClient()
    return _http_client


# =============================================================================
# Auto-Detection: Use HTTP or Direct Kafka
# =============================================================================


async def get_best_event_publisher():
    """
    Automatically select the best event publisher based on environment.

    Returns:
        - EventGateway if Kafka is reachable (K8s deployment)
        - HttpEventClient if not (Railway/external deployment)
    """
    from src.services.event_gateway import EventGateway, get_event_gateway

    # Check if we should force HTTP mode
    force_http = os.getenv("FORCE_HTTP_EVENTS", "").lower() in ("true", "1", "yes")
    if force_http:
        logger.info("Forcing HTTP event mode via FORCE_HTTP_EVENTS")
        return get_http_event_client()

    # Try direct Kafka first
    try:
        gateway = get_event_gateway()
        await gateway.start()

        if gateway.is_running:
            logger.info("Using direct Kafka event publisher")
            return gateway
    except Exception as e:
        logger.warning(
            "Direct Kafka unavailable, falling back to HTTP",
            error=str(e),
        )

    # Fall back to HTTP
    logger.info("Using HTTP event publisher (Kafka unreachable)")
    return get_http_event_client()
