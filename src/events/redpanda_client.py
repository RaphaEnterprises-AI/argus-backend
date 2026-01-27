"""
Redpanda Client for Argus Event Pipeline

Production-ready Kafka client using confluent-kafka library.
Designed for Redpanda Serverless with SASL_SSL authentication.

Usage:
    from src.events.redpanda_client import RedpandaClient, ArgusEvent

    client = RedpandaClient()
    client.publish(ArgusEvent(
        event_type="test.executed",
        org_id="org-123",
        payload={"test_id": "test-456", "status": "passed"}
    ))
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional, Callable, Any, List, Dict
from uuid import uuid4
from dataclasses import dataclass, field, asdict
from enum import Enum
import structlog

from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Argus event types following the architecture spec."""
    # Codebase events
    CODEBASE_INGESTED = "codebase.ingested"
    CODEBASE_ANALYZED = "codebase.analyzed"

    # Test events
    TEST_CREATED = "test.created"
    TEST_EXECUTED = "test.executed"
    TEST_FAILED = "test.failed"

    # Error events
    ERROR_REPORTED = "error.reported"

    # Healing events
    HEALING_REQUESTED = "healing.requested"
    HEALING_COMPLETED = "healing.completed"

    # Insight events
    INSIGHT_GENERATED = "insight.generated"


@dataclass
class ArgusEvent:
    """
    Base event schema for all Argus events.

    Designed for:
    - Multi-tenant isolation (org_id partitioning)
    - Event correlation (correlation_id, causation_id)
    - Idempotency (idempotency_key)
    - Exactly-once processing support
    """
    event_type: str
    org_id: str
    payload: dict

    # Auto-generated fields
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0"

    # Optional context
    project_id: Optional[str] = None
    user_id: Optional[str] = None

    # Correlation for distributed tracing
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None

    # Source metadata
    source: str = "argus-backend"
    idempotency_key: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> bytes:
        """Serialize to JSON bytes for Kafka."""
        return json.dumps(self.to_dict()).encode('utf-8')

    @classmethod
    def from_json(cls, data: bytes) -> 'ArgusEvent':
        """Deserialize from JSON bytes."""
        parsed = json.loads(data.decode('utf-8'))
        return cls(**parsed)

    @property
    def topic(self) -> str:
        """Get the Kafka topic for this event type."""
        return f"argus.{self.event_type}"


class RedpandaConfig:
    """Configuration for Redpanda connection."""

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        sasl_username: Optional[str] = None,
        sasl_password: Optional[str] = None,
        sasl_mechanism: str = "SCRAM-SHA-256",
        security_protocol: str = "SASL_SSL",
    ):
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "REDPANDA_BROKERS",
            "localhost:9092"
        )
        self.sasl_username = sasl_username or os.getenv("REDPANDA_SASL_USERNAME")
        self.sasl_password = sasl_password or os.getenv("REDPANDA_SASL_PASSWORD")
        self.sasl_mechanism = sasl_mechanism or os.getenv(
            "REDPANDA_SASL_MECHANISM",
            "SCRAM-SHA-256"
        )
        self.security_protocol = security_protocol or os.getenv(
            "REDPANDA_SECURITY_PROTOCOL",
            "SASL_SSL"
        )

    def to_confluent_config(self, extra: Optional[dict] = None) -> dict:
        """Convert to confluent-kafka configuration dict."""
        config = {
            "bootstrap.servers": self.bootstrap_servers,
        }

        # Add SASL config if credentials provided
        if self.sasl_username and self.sasl_password:
            config.update({
                "security.protocol": self.security_protocol,
                "sasl.mechanism": self.sasl_mechanism,
                "sasl.username": self.sasl_username,
                "sasl.password": self.sasl_password,
            })

        if extra:
            config.update(extra)

        return config


class RedpandaClient:
    """
    Production-ready Redpanda client for Argus using confluent-kafka.

    Features:
    - Async-friendly producer with delivery reports
    - Consumer with automatic offset management
    - Topic auto-creation with proper partitioning
    - Multi-tenant partitioning by org_id
    - Structured logging
    - Error handling and retries
    """

    # Topic configurations
    TOPIC_CONFIGS = {
        "argus.codebase.ingested": {"partitions": 6, "retention_ms": 7 * 24 * 60 * 60 * 1000},
        "argus.codebase.analyzed": {"partitions": 6, "retention_ms": 30 * 24 * 60 * 60 * 1000},
        "argus.test.created": {"partitions": 6, "retention_ms": 7 * 24 * 60 * 60 * 1000},
        "argus.test.executed": {"partitions": 12, "retention_ms": 30 * 24 * 60 * 60 * 1000},
        "argus.test.failed": {"partitions": 6, "retention_ms": 90 * 24 * 60 * 60 * 1000},
        "argus.error.reported": {"partitions": 6, "retention_ms": 30 * 24 * 60 * 60 * 1000},
        "argus.healing.requested": {"partitions": 3, "retention_ms": 7 * 24 * 60 * 60 * 1000},
        "argus.healing.completed": {"partitions": 3, "retention_ms": 30 * 24 * 60 * 60 * 1000},
        "argus.insight.generated": {"partitions": 6, "retention_ms": 90 * 24 * 60 * 60 * 1000},
        "argus.dlq": {"partitions": 3, "retention_ms": 30 * 24 * 60 * 60 * 1000},
    }

    def __init__(self, config: Optional[RedpandaConfig] = None):
        self.config = config or RedpandaConfig()
        self._producer: Optional[Producer] = None
        self._admin: Optional[AdminClient] = None
        self._consumers: Dict[str, Consumer] = {}
        self._delivery_callbacks: Dict[str, Callable] = {}

    def _get_admin(self) -> AdminClient:
        """Lazy initialization of admin client."""
        if self._admin is None:
            self._admin = AdminClient(self.config.to_confluent_config())
            logger.info("Redpanda admin client initialized", bootstrap_servers=self.config.bootstrap_servers)
        return self._admin

    def _get_producer(self) -> Producer:
        """Lazy initialization of producer."""
        if self._producer is None:
            producer_config = self.config.to_confluent_config({
                "acks": "all",
                "retries": 3,
                "retry.backoff.ms": 1000,
                "linger.ms": 5,  # Small batching for better throughput
                "batch.size": 16384,
            })
            self._producer = Producer(producer_config)
            logger.info("Redpanda producer initialized", bootstrap_servers=self.config.bootstrap_servers)
        return self._producer

    def _delivery_report(self, err, msg):
        """Callback for delivery reports."""
        if err is not None:
            logger.error(
                "Message delivery failed",
                topic=msg.topic(),
                error=str(err),
            )
        else:
            logger.debug(
                "Message delivered",
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
            )

    def list_topics(self) -> List[str]:
        """List all topics in the cluster."""
        admin = self._get_admin()
        metadata = admin.list_topics(timeout=30)
        return list(metadata.topics.keys())

    def create_topics(self, topics: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Create topics with proper configuration.

        Args:
            topics: List of topic names. If None, creates all standard Argus topics.

        Returns:
            Dict mapping topic name to success status
        """
        admin = self._get_admin()
        topics_to_create = topics or list(self.TOPIC_CONFIGS.keys())
        results = {}

        new_topics = []
        for topic_name in topics_to_create:
            config = self.TOPIC_CONFIGS.get(topic_name, {"partitions": 6})
            new_topics.append(NewTopic(
                topic_name,
                num_partitions=config.get("partitions", 6),
                replication_factor=3,  # Redpanda Serverless requires 3
            ))

        # Create topics
        futures = admin.create_topics(new_topics, operation_timeout=30)

        for topic_name, future in futures.items():
            try:
                future.result()  # Wait for completion
                logger.info("Created topic", topic=topic_name)
                results[topic_name] = True
            except KafkaException as e:
                if "TOPIC_ALREADY_EXISTS" in str(e):
                    logger.debug("Topic already exists", topic=topic_name)
                    results[topic_name] = True
                else:
                    logger.error("Failed to create topic", topic=topic_name, error=str(e))
                    results[topic_name] = False

        return results

    def publish(
        self,
        event: ArgusEvent,
        on_delivery: Optional[Callable] = None,
    ) -> None:
        """
        Publish an event to Redpanda (async with callback).

        Events are partitioned by org_id to ensure:
        - Tenant isolation
        - Ordering within a tenant
        - Parallel processing across tenants

        Args:
            event: The ArgusEvent to publish
            on_delivery: Optional callback for delivery confirmation
        """
        producer = self._get_producer()
        topic = event.topic
        partition_key = event.org_id.encode('utf-8')

        callback = on_delivery or self._delivery_report

        producer.produce(
            topic,
            key=partition_key,
            value=event.to_json(),
            callback=callback,
        )

        # Trigger delivery reports (non-blocking)
        producer.poll(0)

    def publish_sync(self, event: ArgusEvent, timeout: float = 10.0) -> bool:
        """
        Synchronously publish an event and wait for confirmation.

        Args:
            event: The ArgusEvent to publish
            timeout: Timeout in seconds

        Returns:
            True on success, raises exception on failure
        """
        producer = self._get_producer()
        topic = event.topic
        partition_key = event.org_id.encode('utf-8')

        delivered = {"success": False, "error": None}

        def delivery_callback(err, msg):
            if err:
                delivered["error"] = err
            else:
                delivered["success"] = True
                logger.info(
                    "Event published (sync)",
                    topic=msg.topic(),
                    partition=msg.partition(),
                    offset=msg.offset(),
                    event_id=event.event_id,
                )

        producer.produce(
            topic,
            key=partition_key,
            value=event.to_json(),
            callback=delivery_callback,
        )

        # Wait for delivery
        producer.flush(timeout=timeout)

        if delivered["error"]:
            raise KafkaException(delivered["error"])

        return delivered["success"]

    def subscribe(
        self,
        topics: List[str],
        group_id: str,
        auto_offset_reset: str = "earliest",
    ) -> Consumer:
        """
        Create a consumer subscribed to topics.

        Args:
            topics: List of topics to subscribe to
            group_id: Consumer group ID
            auto_offset_reset: Where to start if no offset exists

        Returns:
            Consumer instance
        """
        consumer_key = f"{group_id}:{','.join(sorted(topics))}"

        if consumer_key not in self._consumers:
            consumer_config = self.config.to_confluent_config({
                "group.id": group_id,
                "auto.offset.reset": auto_offset_reset,
                "enable.auto.commit": True,
                "auto.commit.interval.ms": 5000,
            })
            consumer = Consumer(consumer_config)
            consumer.subscribe(topics)
            self._consumers[consumer_key] = consumer
            logger.info(
                "Consumer created",
                group_id=group_id,
                topics=topics,
            )

        return self._consumers[consumer_key]

    def consume(
        self,
        topics: List[str],
        group_id: str,
        handler: Callable[[ArgusEvent], None],
        max_messages: Optional[int] = None,
        timeout_sec: float = 1.0,
    ) -> int:
        """
        Consume messages and process with handler.

        Args:
            topics: Topics to consume from
            group_id: Consumer group ID
            handler: Function to process each event
            max_messages: Max messages to process (None = infinite)
            timeout_sec: Poll timeout in seconds

        Returns:
            Number of messages processed
        """
        consumer = self.subscribe(topics, group_id)
        count = 0

        logger.info("Starting consumption", topics=topics, group_id=group_id)

        try:
            while True:
                msg = consumer.poll(timeout=timeout_sec)

                if msg is None:
                    if max_messages and count >= max_messages:
                        break
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug("Reached end of partition", partition=msg.partition())
                        if max_messages and count >= max_messages:
                            break
                        continue
                    else:
                        logger.error("Consumer error", error=msg.error())
                        continue

                try:
                    event = ArgusEvent.from_json(msg.value())
                    logger.debug(
                        "Received event",
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                        event_type=event.event_type,
                        org_id=event.org_id,
                    )
                    handler(event)
                except Exception as e:
                    logger.error(
                        "Handler error",
                        error=str(e),
                        topic=msg.topic(),
                        offset=msg.offset(),
                    )
                    # Send to DLQ
                    self._send_to_dlq_raw(msg.value(), str(e))

                count += 1
                if max_messages and count >= max_messages:
                    break

        except KeyboardInterrupt:
            logger.info("Consumption interrupted")
        finally:
            logger.info("Consumption stopped", messages_processed=count)

        return count

    def _send_to_dlq_raw(self, raw_message: bytes, error: str) -> None:
        """Send failed raw message to Dead Letter Queue."""
        try:
            dlq_payload = {
                "original_message": raw_message.decode('utf-8'),
                "error": error,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            producer = self._get_producer()
            producer.produce(
                "argus.dlq",
                key=b"error",
                value=json.dumps(dlq_payload).encode('utf-8'),
            )
            logger.warning("Message sent to DLQ", error=error[:100])
        except Exception as e:
            logger.error("Failed to send to DLQ", error=str(e))

    def flush(self, timeout: float = 10.0) -> int:
        """Flush any pending messages. Returns number of messages still in queue."""
        if self._producer:
            return self._producer.flush(timeout=timeout)
        return 0

    def close(self) -> None:
        """Close all connections."""
        if self._producer:
            self._producer.flush(timeout=10)
            self._producer = None

        for consumer in self._consumers.values():
            consumer.close()
        self._consumers.clear()

        self._admin = None

        logger.info("Redpanda client closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions for common event types
def emit_test_created(
    client: RedpandaClient,
    org_id: str,
    project_id: str,
    test_id: str,
    test_name: str,
    test_type: str,
    **kwargs
) -> None:
    """Emit a test.created event."""
    event = ArgusEvent(
        event_type=EventType.TEST_CREATED,
        org_id=org_id,
        project_id=project_id,
        payload={
            "test_id": test_id,
            "test_name": test_name,
            "test_type": test_type,
            **kwargs,
        },
    )
    client.publish(event)


def emit_test_executed(
    client: RedpandaClient,
    org_id: str,
    project_id: str,
    test_id: str,
    run_id: str,
    status: str,
    duration_ms: int,
    **kwargs
) -> None:
    """Emit a test.executed event."""
    event = ArgusEvent(
        event_type=EventType.TEST_EXECUTED,
        org_id=org_id,
        project_id=project_id,
        payload={
            "test_id": test_id,
            "run_id": run_id,
            "status": status,
            "duration_ms": duration_ms,
            **kwargs,
        },
    )
    client.publish(event)


def emit_codebase_ingested(
    client: RedpandaClient,
    org_id: str,
    project_id: str,
    repo_url: str,
    branch: str,
    commit_sha: str,
    **kwargs
) -> None:
    """Emit a codebase.ingested event."""
    event = ArgusEvent(
        event_type=EventType.CODEBASE_INGESTED,
        org_id=org_id,
        project_id=project_id,
        payload={
            "repo_url": repo_url,
            "branch": branch,
            "commit_sha": commit_sha,
            **kwargs,
        },
    )
    client.publish(event)


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Redpanda client testing")
    parser.add_argument("--list-topics", action="store_true", help="List all topics")
    parser.add_argument("--create-topics", action="store_true", help="Create all Argus topics")
    parser.add_argument("--produce", action="store_true", help="Produce test events")
    parser.add_argument("--consume", action="store_true", help="Consume events")
    parser.add_argument("--topic", default="argus.test.executed", help="Topic to use")
    parser.add_argument("--group", default="argus-test-consumer", help="Consumer group")
    parser.add_argument("--count", type=int, default=10, help="Number of events")
    args = parser.parse_args()

    with RedpandaClient() as client:
        if args.list_topics:
            print("Listing topics...")
            topics = client.list_topics()
            print(f"Found {len(topics)} topics:")
            for topic in sorted(topics):
                print(f"  - {topic}")

        if args.create_topics:
            print("Creating Argus topics...")
            results = client.create_topics()
            for topic, success in results.items():
                status = "✅" if success else "❌"
                print(f"  {status} {topic}")

        if args.produce:
            print(f"Producing {args.count} test events...")
            for i in range(args.count):
                event = ArgusEvent(
                    event_type="test.executed",
                    org_id=f"org-{i % 3}",
                    project_id="project-1",
                    payload={
                        "test_id": f"test-{i}",
                        "run_id": f"run-{int(datetime.now().timestamp())}",
                        "status": "passed" if i % 2 == 0 else "failed",
                        "duration_ms": 100 + i * 10,
                    },
                )
                success = client.publish_sync(event)
                print(f"  {'✅' if success else '❌'} Event {i+1}/{args.count}")
            print("Done!")

        if args.consume:
            print(f"Consuming from {args.topic} (group: {args.group})...")

            def handler(event: ArgusEvent):
                print(f"  📥 [{event.org_id}] {event.event_type}: {event.payload}")

            count = client.consume([args.topic], args.group, handler, max_messages=args.count)
            print(f"Processed {count} messages")
