"""
Kafka Topic Configuration for Argus Events

Topic naming convention: argus.{domain}.{event}
All topics use 6 partitions for parallelism with ordering within org+project.
"""

from dataclasses import dataclass
from typing import Optional

from src.events.schemas import EventType

# =============================================================================
# Topic Names
# =============================================================================

TOPIC_CODEBASE_INGESTED = "argus.codebase.ingested"
TOPIC_CODEBASE_ANALYZED = "argus.codebase.analyzed"
TOPIC_TEST_CREATED = "argus.test.created"
TOPIC_TEST_EXECUTED = "argus.test.executed"
TOPIC_TEST_FAILED = "argus.test.failed"
TOPIC_HEALING_REQUESTED = "argus.healing.requested"
TOPIC_HEALING_COMPLETED = "argus.healing.completed"
TOPIC_DLQ = "argus.dlq"

# A2A Communication Topics
TOPIC_AGENT_REQUEST = "argus.agent.request"
TOPIC_AGENT_RESPONSE = "argus.agent.response"
TOPIC_AGENT_BROADCAST = "argus.agent.broadcast"
TOPIC_AGENT_HEARTBEAT = "argus.agent.heartbeat"


@dataclass
class TopicConfig:
    """Configuration for a Kafka topic."""

    name: str
    partitions: int = 6
    replication_factor: int = 2
    retention_ms: int = 7 * 24 * 60 * 60 * 1000  # 7 days default
    cleanup_policy: str = "delete"
    compression_type: str = "zstd"

    # Consumer configuration hints
    consumer_group: str = "argus-workers"
    max_poll_records: int = 100

    def to_rpk_create_args(self) -> str:
        """Generate rpk topic create command arguments."""
        return (
            f"--partitions {self.partitions} "
            f"--replicas {self.replication_factor} "
            f"-c retention.ms={self.retention_ms} "
            f"-c cleanup.policy={self.cleanup_policy} "
            f"-c compression.type={self.compression_type}"
        )


# =============================================================================
# Topic Configurations
# =============================================================================

TOPIC_CONFIGS: dict[str, TopicConfig] = {
    TOPIC_CODEBASE_INGESTED: TopicConfig(
        name=TOPIC_CODEBASE_INGESTED,
        partitions=6,
        retention_ms=7 * 24 * 60 * 60 * 1000,  # 7 days
    ),
    TOPIC_CODEBASE_ANALYZED: TopicConfig(
        name=TOPIC_CODEBASE_ANALYZED,
        partitions=6,
        retention_ms=7 * 24 * 60 * 60 * 1000,  # 7 days
    ),
    TOPIC_TEST_CREATED: TopicConfig(
        name=TOPIC_TEST_CREATED,
        partitions=6,
        retention_ms=30 * 24 * 60 * 60 * 1000,  # 30 days
    ),
    TOPIC_TEST_EXECUTED: TopicConfig(
        name=TOPIC_TEST_EXECUTED,
        partitions=6,
        retention_ms=30 * 24 * 60 * 60 * 1000,  # 30 days
    ),
    TOPIC_TEST_FAILED: TopicConfig(
        name=TOPIC_TEST_FAILED,
        partitions=6,
        retention_ms=30 * 24 * 60 * 60 * 1000,  # 30 days
        consumer_group="argus-healers",  # Dedicated consumer group for healing
    ),
    TOPIC_HEALING_REQUESTED: TopicConfig(
        name=TOPIC_HEALING_REQUESTED,
        partitions=6,
        retention_ms=14 * 24 * 60 * 60 * 1000,  # 14 days
        consumer_group="argus-healers",
    ),
    TOPIC_HEALING_COMPLETED: TopicConfig(
        name=TOPIC_HEALING_COMPLETED,
        partitions=6,
        retention_ms=30 * 24 * 60 * 60 * 1000,  # 30 days
    ),
    TOPIC_DLQ: TopicConfig(
        name=TOPIC_DLQ,
        partitions=3,  # Fewer partitions for DLQ
        retention_ms=90 * 24 * 60 * 60 * 1000,  # 90 days - keep failures longer
        cleanup_policy="compact,delete",  # Compact to keep latest per key
    ),
    # A2A Communication Topics
    TOPIC_AGENT_REQUEST: TopicConfig(
        name=TOPIC_AGENT_REQUEST,
        partitions=6,
        retention_ms=24 * 60 * 60 * 1000,  # 1 day - requests are short-lived
        consumer_group="argus-agents",
    ),
    TOPIC_AGENT_RESPONSE: TopicConfig(
        name=TOPIC_AGENT_RESPONSE,
        partitions=6,
        retention_ms=24 * 60 * 60 * 1000,  # 1 day - responses are short-lived
        consumer_group="argus-agents",
    ),
    TOPIC_AGENT_BROADCAST: TopicConfig(
        name=TOPIC_AGENT_BROADCAST,
        partitions=3,  # Fewer partitions for broadcasts (all agents consume all messages)
        retention_ms=7 * 24 * 60 * 60 * 1000,  # 7 days
        consumer_group="argus-agents",
    ),
    TOPIC_AGENT_HEARTBEAT: TopicConfig(
        name=TOPIC_AGENT_HEARTBEAT,
        partitions=3,  # Fewer partitions for heartbeats
        retention_ms=1 * 60 * 60 * 1000,  # 1 hour - heartbeats are ephemeral
        cleanup_policy="compact,delete",  # Compact to keep latest per agent
        consumer_group="argus-monitor",
    ),
}


# =============================================================================
# Topic Routing
# =============================================================================

EVENT_TYPE_TO_TOPIC: dict[EventType, str] = {
    EventType.CODEBASE_INGESTED: TOPIC_CODEBASE_INGESTED,
    EventType.CODEBASE_ANALYZED: TOPIC_CODEBASE_ANALYZED,
    EventType.TEST_CREATED: TOPIC_TEST_CREATED,
    EventType.TEST_EXECUTED: TOPIC_TEST_EXECUTED,
    EventType.TEST_FAILED: TOPIC_TEST_FAILED,
    EventType.HEALING_REQUESTED: TOPIC_HEALING_REQUESTED,
    EventType.HEALING_COMPLETED: TOPIC_HEALING_COMPLETED,
    EventType.DLQ: TOPIC_DLQ,
    # A2A Communication
    EventType.AGENT_REQUEST: TOPIC_AGENT_REQUEST,
    EventType.AGENT_RESPONSE: TOPIC_AGENT_RESPONSE,
    EventType.AGENT_BROADCAST: TOPIC_AGENT_BROADCAST,
    EventType.AGENT_HEARTBEAT: TOPIC_AGENT_HEARTBEAT,
}


def get_topic_for_event(event_type: EventType) -> str:
    """Get the topic name for an event type.

    Args:
        event_type: The event type enum

    Returns:
        Topic name string

    Raises:
        ValueError: If event type has no configured topic
    """
    topic = EVENT_TYPE_TO_TOPIC.get(event_type)
    if topic is None:
        raise ValueError(f"No topic configured for event type: {event_type}")
    return topic


def get_topic_config(topic_name: str) -> TopicConfig | None:
    """Get configuration for a topic.

    Args:
        topic_name: Topic name

    Returns:
        TopicConfig if found, None otherwise
    """
    return TOPIC_CONFIGS.get(topic_name)


def get_all_topics() -> list[str]:
    """Get list of all topic names.

    Returns:
        List of topic name strings
    """
    return list(TOPIC_CONFIGS.keys())


def generate_topic_creation_script() -> str:
    """Generate rpk commands to create all topics.

    Returns:
        Shell script content for topic creation
    """
    lines = [
        "#!/bin/bash",
        "# Argus Kafka Topic Creation Script",
        "# Run inside Redpanda pod: kubectl exec -n argus-data redpanda-0 -- bash < this_script.sh",
        "",
    ]

    for topic, config in TOPIC_CONFIGS.items():
        lines.append(f"rpk topic create {topic} {config.to_rpk_create_args()}")

    lines.append("")
    lines.append("# Verify topics created")
    lines.append("rpk topic list")

    return "\n".join(lines)
