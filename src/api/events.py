"""
HTTP Event Gateway API

Provides HTTP endpoints for publishing events to Kafka/Redpanda.
This solves the critical blocker where Railway (external) cannot
directly connect to Redpanda running inside K8s.

Architecture:
    Railway Backend (External)
            │
            │ HTTPS POST /api/events/{event_type}
            ▼
    ┌───────────────────────┐
    │  HTTP Event Gateway   │  (This file - runs inside K8s)
    │  FastAPI Endpoint     │
    └───────────────────────┘
            │
            │ Kafka Protocol (Internal)
            ▼
    ┌───────────────────────┐
    │   Redpanda (K8s)      │
    │   Internal Service    │
    └───────────────────────┘
            │
            │ Consumer Group
            ▼
    ┌───────────────────────┐
    │   Cognee Worker       │
    │   (Pattern Learning)  │
    └───────────────────────┘

Usage from Railway:
    response = await httpx.post(
        f"{K8S_EVENT_GATEWAY_URL}/api/events/test.executed",
        json={
            "org_id": "org-123",
            "project_id": "project-456",
            "data": {"test_id": "test-789", "status": "passed", ...}
        },
        headers={"X-API-Key": api_key}
    )
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.security.auth import UserContext, get_current_user
from src.services.event_gateway import (
    ArgusEvent,
    EventGateway,
    EventType,
    get_event_gateway,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


# =============================================================================
# Request/Response Models
# =============================================================================


class EventPayload(BaseModel):
    """Request payload for publishing an event."""

    org_id: str = Field(..., description="Organization ID (required)")
    project_id: str | None = Field(None, description="Project ID")
    user_id: str | None = Field(None, description="User ID who triggered the event")
    data: dict[str, Any] = Field(default_factory=dict, description="Event payload data")
    correlation_id: str | None = Field(None, description="Correlation ID for tracing")
    causation_id: str | None = Field(None, description="ID of causing event")
    source: str = Field("http-gateway", description="Event source identifier")
    idempotency_key: str | None = Field(
        None,
        description="Idempotency key to prevent duplicate processing"
    )


class EventResponse(BaseModel):
    """Response after publishing an event."""

    success: bool
    event_id: str
    event_type: str
    topic: str
    timestamp: str
    message: str


class BatchEventPayload(BaseModel):
    """Request payload for publishing multiple events."""

    events: list[dict[str, Any]] = Field(
        ...,
        description="List of events to publish",
        min_length=1,
        max_length=100,
    )


class BatchEventResponse(BaseModel):
    """Response after publishing multiple events."""

    success: bool
    total: int
    published: int
    failed: int
    events: list[EventResponse]
    errors: list[dict[str, str]]


class GatewayHealthResponse(BaseModel):
    """Health check response for event gateway."""

    healthy: bool
    kafka_connected: bool
    bootstrap_servers: str
    events_published: int
    events_failed: int
    dlq_messages: int


# =============================================================================
# Helper Functions
# =============================================================================


def _event_type_from_string(event_type_str: str) -> EventType | None:
    """Convert string to EventType enum."""
    # Map common patterns
    type_map = {
        "codebase.ingested": EventType.CODEBASE_INGESTED,
        "codebase.analyzed": EventType.CODEBASE_ANALYZED,
        "test.created": EventType.TEST_CREATED,
        "test.executed": EventType.TEST_EXECUTED,
        "test.failed": EventType.TEST_FAILED,
        "healing.requested": EventType.HEALING_REQUESTED,
        "healing.completed": EventType.HEALING_COMPLETED,
        "integration.github": EventType.INTEGRATION_GITHUB,
        "integration.jira": EventType.INTEGRATION_JIRA,
        "integration.slack": EventType.INTEGRATION_SLACK,
        "notification.send": EventType.NOTIFICATION_SEND,
    }
    return type_map.get(event_type_str)


# =============================================================================
# Gateway State Management
# =============================================================================

_gateway_instance: EventGateway | None = None
_gateway_started: bool = False


async def get_or_start_gateway() -> EventGateway:
    """Get or lazily start the event gateway."""
    global _gateway_instance, _gateway_started

    if _gateway_instance is None:
        _gateway_instance = get_event_gateway()

    if not _gateway_started:
        try:
            await _gateway_instance.start()
            _gateway_started = True
            logger.info("HTTP Event Gateway started successfully")
        except ImportError as e:
            logger.warning(
                "Kafka client not available, events will be logged only",
                error=str(e)
            )
            # Continue without Kafka - events will just be logged
        except Exception as e:
            logger.error("Failed to start event gateway", error=str(e))
            # Don't raise - allow logging-only mode

    return _gateway_instance


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/health", response_model=GatewayHealthResponse)
async def event_gateway_health() -> GatewayHealthResponse:
    """
    Check health of the event gateway.

    Returns connection status to Kafka/Redpanda and metrics.
    """
    gateway = await get_or_start_gateway()

    return GatewayHealthResponse(
        healthy=gateway.is_running,
        kafka_connected=gateway.is_running,
        bootstrap_servers=os.getenv("REDPANDA_BROKERS", "localhost:9092"),
        events_published=gateway.metrics.get("events_published", 0),
        events_failed=gateway.metrics.get("events_failed", 0),
        dlq_messages=gateway.metrics.get("dlq_messages", 0),
    )


@router.post("/{event_type}", response_model=EventResponse)
async def publish_event(
    event_type: str,
    payload: EventPayload,
    background_tasks: BackgroundTasks,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> EventResponse:
    """
    Publish a single event to Kafka/Redpanda via HTTP.

    This endpoint acts as an HTTP-to-Kafka bridge, allowing external services
    (like Railway) to publish events without direct Kafka connectivity.

    Args:
        event_type: Event type (e.g., "test.executed", "test.failed")
        payload: Event data including org_id and payload

    Returns:
        EventResponse with event_id and confirmation

    Example:
        POST /api/events/test.executed
        {
            "org_id": "org-123",
            "project_id": "project-456",
            "data": {
                "test_id": "test-789",
                "test_name": "Login Flow Test",
                "status": "passed",
                "duration_ms": 1500
            }
        }
    """
    # Parse event type
    event_type_enum = _event_type_from_string(event_type)
    if event_type_enum is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown event type: {event_type}. "
            f"Valid types: {[e.value for e in EventType]}"
        )

    # Use authenticated user's org if not specified
    org_id = payload.org_id
    if user.organization_id and payload.org_id != user.organization_id:
        # Log potential cross-tenant access attempt
        logger.warning(
            "Cross-tenant event publish attempt",
            requested_org=payload.org_id,
            user_org=user.organization_id,
            user_id=user.user_id,
        )
        org_id = user.organization_id  # Force to user's org

    gateway = await get_or_start_gateway()
    event_id = str(uuid4())
    topic = f"argus.{event_type}"
    timestamp = datetime.now(UTC).isoformat()

    # Attempt Kafka publish
    if gateway.is_running:
        try:
            result = await gateway.publish(
                event_type=event_type_enum,
                data=payload.data,
                org_id=org_id,
                project_id=payload.project_id,
                user_id=payload.user_id or user.user_id,
                correlation_id=payload.correlation_id,
                causation_id=payload.causation_id,
                source=payload.source,
            )
            if result:
                event_id = result.event_id
                logger.info(
                    "Event published via HTTP gateway",
                    event_id=event_id,
                    event_type=event_type,
                    org_id=org_id,
                    topic=topic,
                )
                return EventResponse(
                    success=True,
                    event_id=event_id,
                    event_type=event_type,
                    topic=topic,
                    timestamp=timestamp,
                    message="Event published to Kafka successfully",
                )
        except Exception as e:
            logger.error(
                "Failed to publish event to Kafka",
                event_type=event_type,
                error=str(e),
            )
            # Fall through to logging-only mode

    # Logging-only mode (Kafka unavailable)
    logger.warning(
        "Event logged (Kafka unavailable)",
        event_id=event_id,
        event_type=event_type,
        org_id=org_id,
        data=payload.data,
    )

    return EventResponse(
        success=True,
        event_id=event_id,
        event_type=event_type,
        topic=topic,
        timestamp=timestamp,
        message="Event logged (Kafka connection pending)",
    )


@router.post("/batch", response_model=BatchEventResponse)
async def publish_batch_events(
    payload: BatchEventPayload,
    background_tasks: BackgroundTasks,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> BatchEventResponse:
    """
    Publish multiple events in a single request.

    More efficient than multiple single-event calls when publishing
    many events (e.g., after a test run with multiple test results).

    Args:
        payload: List of events to publish

    Returns:
        BatchEventResponse with success/failure counts

    Example:
        POST /api/events/batch
        {
            "events": [
                {
                    "event_type": "test.executed",
                    "org_id": "org-123",
                    "data": {"test_id": "test-1", "status": "passed"}
                },
                {
                    "event_type": "test.failed",
                    "org_id": "org-123",
                    "data": {"test_id": "test-2", "error": "Timeout"}
                }
            ]
        }
    """
    gateway = await get_or_start_gateway()

    results: list[EventResponse] = []
    errors: list[dict[str, str]] = []
    published = 0
    failed = 0

    for i, event_data in enumerate(payload.events):
        try:
            # Validate event structure
            if "event_type" not in event_data:
                raise ValueError("Missing 'event_type' field")
            if "org_id" not in event_data:
                raise ValueError("Missing 'org_id' field")

            event_type = event_data["event_type"]
            event_type_enum = _event_type_from_string(event_type)
            if event_type_enum is None:
                raise ValueError(f"Unknown event type: {event_type}")

            org_id = event_data["org_id"]
            data = event_data.get("data", {})
            project_id = event_data.get("project_id")
            event_id = str(uuid4())
            topic = f"argus.{event_type}"
            timestamp = datetime.now(UTC).isoformat()

            if gateway.is_running:
                result = await gateway.publish(
                    event_type=event_type_enum,
                    data=data,
                    org_id=org_id,
                    project_id=project_id,
                    user_id=event_data.get("user_id") or user.user_id,
                    correlation_id=event_data.get("correlation_id"),
                    causation_id=event_data.get("causation_id"),
                    source=event_data.get("source", "http-gateway-batch"),
                )
                if result:
                    event_id = result.event_id

            results.append(EventResponse(
                success=True,
                event_id=event_id,
                event_type=event_type,
                topic=topic,
                timestamp=timestamp,
                message="Published",
            ))
            published += 1

        except Exception as e:
            failed += 1
            errors.append({
                "index": str(i),
                "error": str(e),
                "event": str(event_data)[:200],  # Truncate for logging
            })
            logger.error(
                "Failed to publish batch event",
                index=i,
                error=str(e),
            )

    return BatchEventResponse(
        success=failed == 0,
        total=len(payload.events),
        published=published,
        failed=failed,
        events=results,
        errors=errors,
    )


# =============================================================================
# Convenience Endpoints for Common Event Types
# =============================================================================


class TestExecutedPayload(BaseModel):
    """Payload for test.executed events."""

    org_id: str
    project_id: str | None = None
    test_id: str
    test_name: str
    status: str = Field(..., pattern="^(passed|failed|skipped|pending)$")
    duration_ms: int = Field(..., ge=0)
    steps_count: int | None = None
    assertions_count: int | None = None
    metadata: dict[str, Any] | None = None


class TestFailedPayload(BaseModel):
    """Payload for test.failed events."""

    org_id: str
    project_id: str | None = None
    test_id: str
    test_name: str
    error_message: str
    failure_type: str = Field(..., pattern="^(assertion|timeout|element_not_found|network|unknown)$")
    stack_trace: str | None = None
    screenshot_url: str | None = None
    metadata: dict[str, Any] | None = None


@router.post("/test/executed", response_model=EventResponse)
async def publish_test_executed(
    payload: TestExecutedPayload,
    user: UserContext = Depends(get_current_user),
) -> EventResponse:
    """
    Convenience endpoint for publishing test.executed events.

    This is the primary endpoint for Cognee pattern learning.
    """
    gateway = await get_or_start_gateway()
    event_id = str(uuid4())
    topic = "argus.test.executed"
    timestamp = datetime.now(UTC).isoformat()

    data = {
        "test_id": payload.test_id,
        "test_name": payload.test_name,
        "status": payload.status,
        "duration_ms": payload.duration_ms,
    }
    if payload.steps_count is not None:
        data["steps_count"] = payload.steps_count
    if payload.assertions_count is not None:
        data["assertions_count"] = payload.assertions_count
    if payload.metadata:
        data["metadata"] = payload.metadata

    if gateway.is_running:
        try:
            result = await gateway.publish(
                event_type=EventType.TEST_EXECUTED,
                data=data,
                org_id=payload.org_id,
                project_id=payload.project_id,
                user_id=user.user_id,
            )
            if result:
                event_id = result.event_id
                logger.info(
                    "Test executed event published",
                    event_id=event_id,
                    test_id=payload.test_id,
                    status=payload.status,
                )
        except Exception as e:
            logger.error("Failed to publish test.executed", error=str(e))

    return EventResponse(
        success=True,
        event_id=event_id,
        event_type="test.executed",
        topic=topic,
        timestamp=timestamp,
        message=f"Test {payload.test_id} execution recorded",
    )


@router.post("/test/failed", response_model=EventResponse)
async def publish_test_failed(
    payload: TestFailedPayload,
    user: UserContext = Depends(get_current_user),
) -> EventResponse:
    """
    Convenience endpoint for publishing test.failed events.

    Critical for Cognee failure pattern learning and self-healing.
    """
    gateway = await get_or_start_gateway()
    event_id = str(uuid4())
    topic = "argus.test.failed"
    timestamp = datetime.now(UTC).isoformat()

    data = {
        "test_id": payload.test_id,
        "test_name": payload.test_name,
        "error_message": payload.error_message,
        "failure_type": payload.failure_type,
    }
    if payload.stack_trace:
        data["stack_trace"] = payload.stack_trace
    if payload.screenshot_url:
        data["screenshot_url"] = payload.screenshot_url
    if payload.metadata:
        data["metadata"] = payload.metadata

    if gateway.is_running:
        try:
            result = await gateway.publish(
                event_type=EventType.TEST_FAILED,
                data=data,
                org_id=payload.org_id,
                project_id=payload.project_id,
                user_id=user.user_id,
            )
            if result:
                event_id = result.event_id
                logger.info(
                    "Test failed event published",
                    event_id=event_id,
                    test_id=payload.test_id,
                    failure_type=payload.failure_type,
                )
        except Exception as e:
            logger.error("Failed to publish test.failed", error=str(e))

    return EventResponse(
        success=True,
        event_id=event_id,
        event_type="test.failed",
        topic=topic,
        timestamp=timestamp,
        message=f"Test failure {payload.test_id} recorded for pattern learning",
    )
