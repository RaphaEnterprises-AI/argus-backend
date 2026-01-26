"""
Argus Event System - Multi-tenant Kafka Events

This module provides:
- Versioned event schemas with tenant isolation
- EventProducer for publishing events to Redpanda/Kafka
- Topic configuration and routing
"""

from src.events.schemas import (
    TenantInfo,
    EventMetadata,
    BaseEvent,
    CodebaseIngestedEvent,
    CodebaseAnalyzedEvent,
    TestCreatedEvent,
    TestExecutedEvent,
    TestFailedEvent,
    HealingRequestedEvent,
    HealingCompletedEvent,
    DLQEvent,
    EventType,
)
from src.events.producer import EventProducer
from src.events.topics import (
    TOPIC_CODEBASE_INGESTED,
    TOPIC_CODEBASE_ANALYZED,
    TOPIC_TEST_CREATED,
    TOPIC_TEST_EXECUTED,
    TOPIC_TEST_FAILED,
    TOPIC_HEALING_REQUESTED,
    TOPIC_HEALING_COMPLETED,
    TOPIC_DLQ,
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
    "TopicConfig",
    "get_topic_for_event",
]
