"""
Argus Event System - Multi-tenant Kafka Events

This module provides:
- Versioned event schemas with tenant isolation
- EventProducer for publishing events to Redpanda/Kafka
- Topic configuration and routing
"""

from src.events.producer import EventProducer
from src.events.schemas import (
    AgentBroadcastEvent,
    AgentHeartbeatEvent,
    # A2A Communication Events
    AgentRequestEvent,
    AgentResponseEvent,
    BaseEvent,
    CodebaseAnalyzedEvent,
    CodebaseIngestedEvent,
    DLQEvent,
    EventMetadata,
    EventType,
    HealingCompletedEvent,
    HealingRequestedEvent,
    TenantInfo,
    TestCreatedEvent,
    TestExecutedEvent,
    TestFailedEvent,
)
from src.events.topics import (
    TOPIC_AGENT_BROADCAST,
    TOPIC_AGENT_HEARTBEAT,
    # A2A Communication Topics
    TOPIC_AGENT_REQUEST,
    TOPIC_AGENT_RESPONSE,
    TOPIC_CODEBASE_ANALYZED,
    TOPIC_CODEBASE_INGESTED,
    TOPIC_DLQ,
    TOPIC_HEALING_COMPLETED,
    TOPIC_HEALING_REQUESTED,
    TOPIC_TEST_CREATED,
    TOPIC_TEST_EXECUTED,
    TOPIC_TEST_FAILED,
    TopicConfig,
    get_topic_for_event,
)

__all__ = [
    # Schemas
    "TenantInfo",
    "EventMetadata",
    "BaseEvent",
    "CodebaseIngestedEvent",
    "CodebaseAnalyzedEvent",
    "TestCreatedEvent",
    "TestExecutedEvent",
    "TestFailedEvent",
    "HealingRequestedEvent",
    "HealingCompletedEvent",
    "DLQEvent",
    "EventType",
    # A2A Communication Events
    "AgentRequestEvent",
    "AgentResponseEvent",
    "AgentBroadcastEvent",
    "AgentHeartbeatEvent",
    # Producer
    "EventProducer",
    # Topics
    "TOPIC_CODEBASE_INGESTED",
    "TOPIC_CODEBASE_ANALYZED",
    "TOPIC_TEST_CREATED",
    "TOPIC_TEST_EXECUTED",
    "TOPIC_TEST_FAILED",
    "TOPIC_HEALING_REQUESTED",
    "TOPIC_HEALING_COMPLETED",
    "TOPIC_DLQ",
    # A2A Communication Topics
    "TOPIC_AGENT_REQUEST",
    "TOPIC_AGENT_RESPONSE",
    "TOPIC_AGENT_BROADCAST",
    "TOPIC_AGENT_HEARTBEAT",
    "TopicConfig",
    "get_topic_for_event",
]
