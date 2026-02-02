"""
Kafka Event Producer with Multi-Tenant Support

Provides an async producer that automatically:
- Routes events to correct topics based on event type
- Sets message keys for partition ordering (by org_id:project_id)
- Adds correlation headers for tracing
- Handles retries and error reporting
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from src.events.schemas import (
    BaseEvent,
    DLQEvent,
    EventMetadata,
    EventType,
    TenantInfo,
)
from src.events.topics import TOPIC_DLQ, get_topic_for_event

logger = logging.getLogger(__name__)


class EventProducer:
    """Async Kafka producer for Argus events.

    Usage:
        async with EventProducer.create(bootstrap_servers="localhost:9092") as producer:
            await producer.send(event)

    Or for long-lived usage:
        producer = EventProducer(bootstrap_servers="localhost:9092")
        await producer.start()
        try:
            await producer.send(event)
        finally:
            await producer.stop()
    """

    def __init__(
        self,
        bootstrap_servers: str,
        sasl_username: str | None = None,
        sasl_password: str | None = None,
        security_protocol: str = "SASL_PLAINTEXT",
        sasl_mechanism: str = "SCRAM-SHA-512",
        client_id: str = "argus-api",
        acks: str = "all",
        compression_type: str = "zstd",
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        request_timeout_ms: int = 30000,
    ):
        """Initialize the producer configuration.

        Args:
            bootstrap_servers: Kafka broker addresses
            sasl_username: SASL username for authentication
            sasl_password: SASL password for authentication
            security_protocol: Security protocol (SASL_PLAINTEXT, SASL_SSL)
            sasl_mechanism: SASL mechanism (SCRAM-SHA-512, PLAIN)
            client_id: Client identifier for broker logs
            acks: Acknowledgment level ('all', '1', '0')
            compression_type: Message compression (zstd, lz4, gzip, snappy)
            max_request_size: Maximum request size in bytes
            request_timeout_ms: Request timeout in milliseconds
        """
        self._bootstrap_servers = bootstrap_servers
        self._client_id = client_id

        # Build producer config
        self._config: dict[str, Any] = {
            "bootstrap_servers": bootstrap_servers,
            "client_id": client_id,
            "acks": acks,
            "compression_type": compression_type,
            "max_request_size": max_request_size,
            "request_timeout_ms": request_timeout_ms,
            "key_serializer": lambda k: k.encode("utf-8") if k else None,
            "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
        }

        # Add SASL config if credentials provided
        if sasl_username and sasl_password:
            self._config.update({
                "security_protocol": security_protocol,
                "sasl_mechanism": sasl_mechanism,
                "sasl_plain_username": sasl_username,
                "sasl_plain_password": sasl_password,
            })

        self._producer: AIOKafkaProducer | None = None
        self._started = False

    @classmethod
    @asynccontextmanager
    async def create(cls, **kwargs):
        """Create a producer as an async context manager.

        Args:
            **kwargs: Arguments passed to __init__

        Yields:
            Started EventProducer instance
        """
        producer = cls(**kwargs)
        await producer.start()
        try:
            yield producer
        finally:
            await producer.stop()

    async def start(self) -> None:
        """Start the producer and connect to Kafka."""
        if self._started:
            return

        self._producer = AIOKafkaProducer(**self._config)
        await self._producer.start()
        self._started = True
        logger.info(
            "EventProducer started",
            extra={"bootstrap_servers": self._bootstrap_servers, "client_id": self._client_id}
        )

    async def stop(self) -> None:
        """Stop the producer and flush pending messages."""
        if not self._started or self._producer is None:
            return

        await self._producer.stop()
        self._started = False
        self._producer = None
        logger.info("EventProducer stopped")

    async def send(
        self,
        event: BaseEvent,
        topic: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> bool:
        """Send an event to Kafka.

        Args:
            event: The event to send
            topic: Override topic (defaults to event type's topic)
            headers: Additional headers to include

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._started or self._producer is None:
            raise RuntimeError("Producer not started. Call start() first.")

        # Determine topic
        if topic is None:
            topic = get_topic_for_event(event.event_type)

        # Build message key for partitioning
        key = event.to_kafka_key()

        # Build headers
        msg_headers = [
            ("event_id", event.event_id.encode()),
            ("event_type", event.event_type.value.encode()),
            ("event_version", event.event_version.encode()),
            ("org_id", event.tenant.org_id.encode()),
            ("timestamp", event.metadata.timestamp.isoformat().encode()),
            ("idempotency_key", event.get_idempotency_key().encode()),
        ]

        if event.tenant.project_id:
            msg_headers.append(("project_id", event.tenant.project_id.encode()))

        if event.metadata.request_id:
            msg_headers.append(("request_id", event.metadata.request_id.encode()))

        if event.metadata.trace_id:
            msg_headers.append(("trace_id", event.metadata.trace_id.encode()))

        if headers:
            for k, v in headers.items():
                msg_headers.append((k, v.encode()))

        try:
            # Send message
            result = await self._producer.send_and_wait(
                topic=topic,
                key=key,
                value=event.to_dict(),
                headers=msg_headers,
            )

            logger.info(
                "Event sent",
                extra={
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "topic": topic,
                    "partition": result.partition,
                    "offset": result.offset,
                    "org_id": event.tenant.org_id,
                }
            )
            return True

        except KafkaError as e:
            logger.error(
                "Failed to send event",
                extra={
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "topic": topic,
                    "error": str(e),
                }
            )
            return False

    async def send_to_dlq(
        self,
        original_event: BaseEvent,
        error: Exception,
        original_topic: str,
        retry_count: int = 0,
        max_retries: int = 3,
        dlq_write_retries: int = 3,
        dlq_retry_delay: float = 1.0,
    ) -> bool:
        """Send a failed event to the dead letter queue with retry logic.

        This method ensures DLQ writes are verified and retried on failure.
        If the DLQ write ultimately fails, the error is logged for alerting.

        Args:
            original_event: The event that failed processing
            error: The exception that caused the failure
            original_topic: Topic the event came from
            retry_count: Number of retry attempts made on original event
            max_retries: Maximum retries configured for original event
            dlq_write_retries: Number of retries for DLQ write itself
            dlq_retry_delay: Base delay in seconds between DLQ write retries

        Returns:
            True if sent to DLQ successfully, False otherwise
        """
        import asyncio
        import traceback

        dlq_event = DLQEvent(
            event_id=str(uuid4()),
            event_type=EventType.DLQ,
            tenant=original_event.tenant,
            metadata=EventMetadata(
                source="dlq-handler",
                triggered_by="error",
            ),
            original_event_id=original_event.event_id,
            original_event_type=original_event.event_type.value,
            original_topic=original_topic,
            original_payload=original_event.to_dict(),
            error_message=str(error),
            error_type=type(error).__name__,
            stack_trace=traceback.format_exc(),
            retry_count=retry_count,
            max_retries=max_retries,
            first_failed_at=datetime.utcnow(),
        )

        # Retry DLQ writes to handle transient failures
        last_error: Exception | None = None
        for attempt in range(1, dlq_write_retries + 1):
            try:
                success = await self.send(dlq_event, topic=TOPIC_DLQ)
                if success:
                    logger.info(
                        "DLQ write successful",
                        extra={
                            "original_event_id": original_event.event_id,
                            "original_event_type": original_event.event_type.value,
                            "dlq_event_id": dlq_event.event_id,
                            "attempt": attempt,
                        }
                    )
                    return True
                else:
                    # send() returned False - Kafka error occurred
                    logger.warning(
                        "DLQ write returned False, retrying",
                        extra={
                            "original_event_id": original_event.event_id,
                            "attempt": attempt,
                            "max_attempts": dlq_write_retries,
                        }
                    )
                    if attempt < dlq_write_retries:
                        await asyncio.sleep(dlq_retry_delay * attempt)
                        continue

            except Exception as e:
                last_error = e
                logger.warning(
                    "DLQ write exception, retrying",
                    extra={
                        "original_event_id": original_event.event_id,
                        "error": str(e),
                        "attempt": attempt,
                        "max_attempts": dlq_write_retries,
                    }
                )
                if attempt < dlq_write_retries:
                    await asyncio.sleep(dlq_retry_delay * attempt)
                    continue

        # All retries exhausted - log critical error for alerting
        logger.error(
            "CRITICAL: DLQ write failed after all retries - message may be lost",
            extra={
                "original_event_id": original_event.event_id,
                "original_event_type": original_event.event_type.value,
                "original_topic": original_topic,
                "error_message": str(error),
                "dlq_write_error": str(last_error) if last_error else "send returned False",
                "attempts": dlq_write_retries,
            }
        )
        return False


def create_event_metadata(
    source: str,
    triggered_by: str = "api",
    request_id: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
) -> EventMetadata:
    """Helper to create EventMetadata with common defaults.

    Args:
        source: Service name generating the event
        triggered_by: What triggered the event
        request_id: Correlation ID (generated if not provided)
        trace_id: Distributed tracing ID
        span_id: Distributed tracing span

    Returns:
        EventMetadata instance
    """
    return EventMetadata(
        request_id=request_id or str(uuid4()),
        source=source,
        triggered_by=triggered_by,
        trace_id=trace_id,
        span_id=span_id,
    )


def create_tenant_info(
    org_id: str,
    project_id: str | None = None,
    user_id: str | None = None,
) -> TenantInfo:
    """Helper to create TenantInfo.

    Args:
        org_id: Organization ID (required)
        project_id: Project ID (optional)
        user_id: User ID (optional)

    Returns:
        TenantInfo instance
    """
    return TenantInfo(
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
    )
