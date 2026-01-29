"""
Argus Event Gateway - Enterprise Event Streaming Service

Provides a high-level interface for publishing events to Redpanda/Kafka
with support for SASL authentication, compression, and dead letter queues.

Features:
- SASL_SSL authentication for enterprise security
- LZ4 compression for efficient bandwidth usage
- Idempotent producer for exactly-once semantics
- Dead letter queue for failed messages
- Multi-tenant event routing with org_id partitioning
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Event types for the Argus platform."""

    # Codebase events
    CODEBASE_INGESTED = "codebase.ingested"
    CODEBASE_ANALYZED = "codebase.analyzed"

    # Test lifecycle events
    TEST_CREATED = "test.created"
    TEST_EXECUTED = "test.executed"
    TEST_FAILED = "test.failed"

    # Self-healing events
    HEALING_REQUESTED = "healing.requested"
    HEALING_COMPLETED = "healing.completed"

    # Integration events
    INTEGRATION_GITHUB = "integration.github"
    INTEGRATION_JIRA = "integration.jira"
    INTEGRATION_SLACK = "integration.slack"

    # Notification events
    NOTIFICATION_SEND = "notification.send"


class ArgusEvent(BaseModel):
    """
    Standard event model for all Argus platform events.

    Follows the CloudEvents specification pattern with additional
    Argus-specific fields for multi-tenancy and distributed tracing.
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this event instance"
    )
    event_type: EventType = Field(
        ...,
        description="Type of event being published"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the event was created"
    )
    org_id: str = Field(
        ...,
        description="Organization ID for multi-tenancy"
    )
    project_id: str | None = Field(
        default=None,
        description="Optional project ID for scoping"
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID who triggered the event"
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload data"
    )
    correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for tracing related events across services"
    )
    causation_id: str | None = Field(
        default=None,
        description="ID of the event that caused this event"
    )
    source: str = Field(
        default="argus-backend",
        description="Source service that generated the event"
    )
    version: str = Field(
        default="1.0",
        description="Event schema version for compatibility"
    )

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class EventGateway:
    """
    Enterprise Event Gateway for Redpanda/Kafka integration.

    Features:
    - SASL_SSL authentication support
    - LZ4 compression for efficient bandwidth usage
    - Idempotent producer for exactly-once semantics
    - Automatic retries with exponential backoff
    - Dead letter queue for failed messages
    - Async batching for high throughput
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        topic_prefix: str = "argus",
        dlq_topic: str | None = None,
        sasl_username: str | None = None,
        sasl_password: str | None = None,
        sasl_mechanism: str | None = None,
        security_protocol: str | None = None,
        compression_type: str = "gzip",  # Use gzip as default (lz4 requires extra lib)
        enable_idempotence: bool = True,
        max_retries: int = 5,
        retry_backoff_ms: int = 100,
        request_timeout_ms: int = 30000,
        acks: str = "all",
    ):
        """
        Initialize the Event Gateway.

        Args:
            bootstrap_servers: Kafka/Redpanda bootstrap servers (comma-separated)
            topic_prefix: Prefix for all topics (e.g., "argus" -> "argus.test.executed")
            dlq_topic: Dead letter queue topic name
            sasl_username: SASL username for authentication
            sasl_password: SASL password for authentication
            sasl_mechanism: SASL mechanism (SCRAM-SHA-256, SCRAM-SHA-512, PLAIN)
            security_protocol: Security protocol (SASL_SSL, SASL_PLAINTEXT, SSL, PLAINTEXT)
            compression_type: Compression algorithm (lz4, gzip, snappy, zstd, none)
            enable_idempotence: Enable idempotent producer for exactly-once delivery
            max_retries: Maximum number of retries for failed sends
            retry_backoff_ms: Initial backoff time between retries
            request_timeout_ms: Timeout for producer requests
            acks: Acknowledgment level ('all', '0', '1')
        """
        # Configuration from parameters or environment
        self._bootstrap_servers = bootstrap_servers or os.getenv(
            "REDPANDA_BROKERS",
            os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        )
        self._topic_prefix = topic_prefix
        self._dlq_topic = dlq_topic or f"{topic_prefix}.dlq"

        # Authentication - read from env vars if not provided
        self._sasl_username = sasl_username or os.getenv("REDPANDA_SASL_USERNAME")
        self._sasl_password = sasl_password or os.getenv("REDPANDA_SASL_PASSWORD")
        self._sasl_mechanism = sasl_mechanism or os.getenv(
            "REDPANDA_SASL_MECHANISM", "SCRAM-SHA-512"
        )
        self._security_protocol = security_protocol or os.getenv(
            "REDPANDA_SECURITY_PROTOCOL", "SASL_PLAINTEXT"
        )

        # Producer settings
        self._compression_type = compression_type
        self._enable_idempotence = enable_idempotence
        self._max_retries = max_retries
        self._retry_backoff_ms = retry_backoff_ms
        self._request_timeout_ms = request_timeout_ms
        self._acks = acks

        # Runtime state
        self._producer = None
        self._started = False
        self._metrics = {
            "events_published": 0,
            "events_failed": 0,
            "dlq_messages": 0,
        }

    async def start(self) -> None:
        """
        Start the event gateway and connect to Redpanda/Kafka.

        This method initializes the Kafka producer with the configured
        settings and establishes connection to the cluster.
        """
        if self._started:
            logger.warning("Event gateway already started")
            return

        try:
            from aiokafka import AIOKafkaProducer
        except ImportError:
            logger.error(
                "aiokafka not installed. Install with: pip install aiokafka",
                hint="Add 'aiokafka>=0.10.0' to pyproject.toml dependencies"
            )
            raise ImportError(
                "aiokafka is required for EventGateway. "
                "Install with: pip install aiokafka"
            )

        # Build producer configuration
        producer_config = {
            "bootstrap_servers": self._bootstrap_servers,
            "compression_type": self._compression_type,
            "enable_idempotence": self._enable_idempotence,
            "max_batch_size": 16384,
            "linger_ms": 10,
            "request_timeout_ms": self._request_timeout_ms,
            "retry_backoff_ms": self._retry_backoff_ms,
            "acks": self._acks if self._acks != "all" else -1,
        }

        # Add SASL authentication if configured
        if self._sasl_username and self._sasl_password:
            producer_config.update({
                "security_protocol": self._security_protocol,
                "sasl_mechanism": self._sasl_mechanism,
                "sasl_plain_username": self._sasl_username,
                "sasl_plain_password": self._sasl_password,
            })

            # Add SSL context for SASL_SSL or SSL protocols
            if self._security_protocol in ("SASL_SSL", "SSL"):
                import ssl
                ssl_context = ssl.create_default_context()
                # For Redpanda Cloud, use default CA certificates
                ssl_context.check_hostname = True
                ssl_context.verify_mode = ssl.CERT_REQUIRED
                producer_config["ssl_context"] = ssl_context
                logger.info(
                    "SSL context created for secure connection",
                    protocol=self._security_protocol,
                )

            logger.info(
                "Event gateway configured with SASL authentication",
                mechanism=self._sasl_mechanism,
                protocol=self._security_protocol,
            )
        else:
            logger.info(
                "Event gateway configured without authentication",
                protocol="PLAINTEXT",
            )

        self._producer = AIOKafkaProducer(**producer_config)

        try:
            await self._producer.start()
            self._started = True
            logger.info(
                "Event gateway started successfully",
                bootstrap_servers=self._bootstrap_servers,
                compression=self._compression_type,
                idempotence=self._enable_idempotence,
            )
        except Exception as e:
            logger.error(
                "Failed to start event gateway",
                error=str(e),
                bootstrap_servers=self._bootstrap_servers,
            )
            self._producer = None
            raise

    async def stop(self) -> None:
        """Stop the event gateway and close connections."""
        if not self._started or not self._producer:
            logger.warning("Event gateway not running")
            return

        try:
            await self._producer.flush()
            await self._producer.stop()
            self._producer = None
            self._started = False
            logger.info(
                "Event gateway stopped",
                events_published=self._metrics["events_published"],
                events_failed=self._metrics["events_failed"],
                dlq_messages=self._metrics["dlq_messages"],
            )
        except Exception as e:
            logger.error("Error stopping event gateway", error=str(e))
            raise

    def _get_topic_name(self, event_type: EventType) -> str:
        """Generate the topic name for an event type."""
        return f"{self._topic_prefix}.{event_type.value}"

    async def publish(
        self,
        event_type: EventType,
        data: dict[str, Any],
        org_id: str,
        project_id: str | None = None,
        user_id: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        source: str = "argus-backend",
        key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> ArgusEvent | None:
        """
        Publish an event to Redpanda/Kafka.

        Args:
            event_type: Type of event to publish
            data: Event payload data
            org_id: Organization ID (required for multi-tenancy)
            project_id: Optional project ID
            user_id: Optional user ID
            correlation_id: Optional correlation ID for distributed tracing
            causation_id: Optional ID of the event that caused this event
            source: Source service name
            key: Optional message key for partitioning (defaults to org_id)
            headers: Optional additional headers

        Returns:
            The published ArgusEvent, or None if publishing failed
        """
        if not self._started or not self._producer:
            logger.error("Event gateway not started. Call start() first.")
            return None

        event = ArgusEvent(
            event_type=event_type,
            org_id=org_id,
            project_id=project_id,
            user_id=user_id,
            data=data,
            correlation_id=correlation_id,
            causation_id=causation_id,
            source=source,
        )

        event_json = event.model_dump_json()

        kafka_headers = [
            ("event_id", event.event_id.encode()),
            ("event_type", event.event_type.value.encode()),
            ("org_id", org_id.encode()),
            ("version", event.version.encode()),
        ]
        if headers:
            kafka_headers.extend(
                (k, v.encode()) for k, v in headers.items()
            )

        partition_key = (key or org_id).encode()
        topic = self._get_topic_name(event_type)

        try:
            await self._producer.send_and_wait(
                topic=topic,
                value=event_json.encode(),
                key=partition_key,
                headers=kafka_headers,
            )
            self._metrics["events_published"] += 1
            logger.debug(
                "Event published",
                event_id=event.event_id,
                event_type=event_type.value,
                topic=topic,
                org_id=org_id,
            )
            return event
        except Exception as e:
            self._metrics["events_failed"] += 1
            logger.error(
                "Failed to publish event",
                event_id=event.event_id,
                event_type=event_type.value,
                topic=topic,
                error=str(e),
            )
            await self._send_to_dlq(event, str(e))
            return None

    async def _send_to_dlq(
        self,
        event: ArgusEvent,
        error_message: str,
    ) -> bool:
        """Send a failed event to the dead letter queue."""
        if not self._producer:
            logger.error("Cannot send to DLQ: producer not initialized")
            return False

        try:
            import json
            dlq_payload = {
                "original_event": event.model_dump(),
                "error": error_message,
                "failed_at": datetime.now(UTC).isoformat(),
                "retry_count": 0,
            }
            dlq_json = json.dumps(dlq_payload, default=str)

            dlq_headers = [
                ("original_event_id", event.event_id.encode()),
                ("original_event_type", event.event_type.value.encode()),
                ("error_reason", error_message[:200].encode()),
            ]

            await self._producer.send_and_wait(
                topic=self._dlq_topic,
                value=dlq_json.encode(),
                key=event.org_id.encode(),
                headers=dlq_headers,
            )
            self._metrics["dlq_messages"] += 1
            logger.warning(
                "Event sent to DLQ",
                event_id=event.event_id,
                event_type=event.event_type.value,
                dlq_topic=self._dlq_topic,
                error=error_message[:100],
            )
            return True
        except Exception as dlq_error:
            logger.error(
                "Failed to send event to DLQ",
                event_id=event.event_id,
                dlq_topic=self._dlq_topic,
                original_error=error_message,
                dlq_error=str(dlq_error),
            )
            return False

    @property
    def is_running(self) -> bool:
        """Check if the gateway is running."""
        return self._started

    @property
    def metrics(self) -> dict[str, int]:
        """Get current metrics."""
        return self._metrics.copy()


# =============================================================================
# Global Gateway Instance
# =============================================================================

_gateway: EventGateway | None = None


def get_event_gateway() -> EventGateway:
    """Get the global EventGateway instance."""
    global _gateway
    if _gateway is None:
        _gateway = EventGateway()
    return _gateway


# =============================================================================
# Convenience Functions
# =============================================================================


async def emit_codebase_ingested(
    codebase_id: str,
    repo_url: str,
    branch: str,
    commit_sha: str,
    files_count: int,
    org_id: str,
    project_id: str | None = None,
    user_id: str | None = None,
    correlation_id: str | None = None,
) -> ArgusEvent | None:
    """Emit a CODEBASE_INGESTED event."""
    gateway = get_event_gateway()
    return await gateway.publish(
        event_type=EventType.CODEBASE_INGESTED,
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
        correlation_id=correlation_id,
    )


async def emit_test_created(
    test_id: str,
    test_name: str,
    test_type: str,
    org_id: str,
    project_id: str | None = None,
    user_id: str | None = None,
    correlation_id: str | None = None,
    source: str | None = None,
    priority: str | None = None,
    tags: list[str] | None = None,
    steps_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArgusEvent | None:
    """Emit a TEST_CREATED event."""
    gateway = get_event_gateway()
    data = {
        "test_id": test_id,
        "test_name": test_name,
        "test_type": test_type,
    }
    if source:
        data["source"] = source
    if priority:
        data["priority"] = priority
    if tags:
        data["tags"] = tags
    if steps_count is not None:
        data["steps_count"] = steps_count
    if metadata:
        data["metadata"] = metadata

    return await gateway.publish(
        event_type=EventType.TEST_CREATED,
        data=data,
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        correlation_id=correlation_id,
    )


async def emit_test_failed(
    test_id: str,
    test_name: str,
    error_message: str,
    failure_type: str,
    org_id: str,
    project_id: str | None = None,
    user_id: str | None = None,
    correlation_id: str | None = None,
    stack_trace: str | None = None,
    screenshot_url: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArgusEvent | None:
    """Emit a TEST_FAILED event."""
    gateway = get_event_gateway()
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

    return await gateway.publish(
        event_type=EventType.TEST_FAILED,
        data=data,
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        correlation_id=correlation_id,
    )


async def emit_test_executed(
    test_id: str,
    test_name: str,
    status: str,
    duration_ms: int,
    org_id: str,
    project_id: str | None = None,
    user_id: str | None = None,
    correlation_id: str | None = None,
    steps_count: int | None = None,
    assertions_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArgusEvent | None:
    """Emit a TEST_EXECUTED event."""
    gateway = get_event_gateway()
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

    return await gateway.publish(
        event_type=EventType.TEST_EXECUTED,
        data=data,
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        correlation_id=correlation_id,
    )


async def emit_healing_requested(
    test_id: str,
    failure_id: str,
    failure_type: str,
    original_selector: str | None,
    org_id: str,
    project_id: str | None = None,
    user_id: str | None = None,
    correlation_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> ArgusEvent | None:
    """Emit a HEALING_REQUESTED event."""
    gateway = get_event_gateway()
    data = {
        "test_id": test_id,
        "failure_id": failure_id,
        "failure_type": failure_type,
    }
    if original_selector:
        data["original_selector"] = original_selector
    if context:
        data["context"] = context

    return await gateway.publish(
        event_type=EventType.HEALING_REQUESTED,
        data=data,
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        correlation_id=correlation_id,
        causation_id=failure_id,
    )


async def emit_healing_completed(
    healing_id: str,
    test_id: str,
    success: bool,
    healing_type: str,
    org_id: str,
    project_id: str | None = None,
    user_id: str | None = None,
    correlation_id: str | None = None,
    original_selector: str | None = None,
    healed_selector: str | None = None,
    confidence_score: float | None = None,
    healing_strategy: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArgusEvent | None:
    """Emit a HEALING_COMPLETED event."""
    gateway = get_event_gateway()
    data = {
        "healing_id": healing_id,
        "test_id": test_id,
        "success": success,
        "healing_type": healing_type,
    }
    if original_selector:
        data["original_selector"] = original_selector
    if healed_selector:
        data["healed_selector"] = healed_selector
    if confidence_score is not None:
        data["confidence_score"] = confidence_score
    if healing_strategy:
        data["healing_strategy"] = healing_strategy
    if metadata:
        data["metadata"] = metadata

    return await gateway.publish(
        event_type=EventType.HEALING_COMPLETED,
        data=data,
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        correlation_id=correlation_id,
    )
