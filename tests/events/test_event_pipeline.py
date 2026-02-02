"""Comprehensive tests for the Kafka/Redpanda Event Pipeline.

Tests cover:
- Event schemas and validation
- Topic configuration (17 topics with proper partition counts)
- Event producer send/batch operations
- ZSTD compression configuration
- Dead letter queue handling
- Error handling and retries
- Redpanda client operations
"""

import asyncio
import json
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.events.schemas import (
    AgentBroadcastEvent,
    AgentHeartbeatEvent,
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
    EVENT_TYPE_TO_TOPIC,
    TOPIC_AGENT_BROADCAST,
    TOPIC_AGENT_HEARTBEAT,
    TOPIC_AGENT_REQUEST,
    TOPIC_AGENT_RESPONSE,
    TOPIC_CODEBASE_ANALYZED,
    TOPIC_CODEBASE_INGESTED,
    TOPIC_CONFIGS,
    TOPIC_DLQ,
    TOPIC_HEALING_COMPLETED,
    TOPIC_HEALING_REQUESTED,
    TOPIC_TEST_CREATED,
    TOPIC_TEST_EXECUTED,
    TOPIC_TEST_FAILED,
    TopicConfig,
    generate_topic_creation_script,
    get_all_topics,
    get_topic_config,
    get_topic_for_event,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_tenant_info():
    """Create sample TenantInfo for testing."""
    return TenantInfo(
        org_id="org-test-123",
        project_id="proj-test-456",
        user_id="user-test-789",
    )


@pytest.fixture
def sample_event_metadata():
    """Create sample EventMetadata for testing."""
    return EventMetadata(
        source="test-service",
        triggered_by="api",
        request_id="req-test-123",
        trace_id="trace-test-456",
        span_id="span-test-789",
    )


@pytest.fixture
def sample_codebase_ingested_event(sample_tenant_info, sample_event_metadata):
    """Create sample CodebaseIngestedEvent for testing."""
    return CodebaseIngestedEvent(
        tenant=sample_tenant_info,
        metadata=sample_event_metadata,
        repository_id="repo-123",
        repository_url="https://github.com/test/repo",
        branch="main",
        commit_sha="abc123def456",
        file_count=100,
        total_size_bytes=1024000,
        languages=["python", "typescript"],
    )


@pytest.fixture
def mock_aiokafka_producer():
    """Create mock AIOKafkaProducer."""
    mock_producer = AsyncMock()
    mock_result = MagicMock()
    mock_result.partition = 0
    mock_result.offset = 42
    mock_producer.send_and_wait = AsyncMock(return_value=mock_result)
    mock_producer.start = AsyncMock()
    mock_producer.stop = AsyncMock()
    return mock_producer


@pytest.fixture
def mock_confluent_producer():
    """Create mock confluent_kafka Producer."""
    mock_producer = MagicMock()
    mock_producer.produce = MagicMock()
    mock_producer.poll = MagicMock(return_value=0)
    mock_producer.flush = MagicMock(return_value=0)
    return mock_producer


@pytest.fixture
def mock_confluent_consumer():
    """Create mock confluent_kafka Consumer."""
    mock_consumer = MagicMock()
    mock_consumer.subscribe = MagicMock()
    mock_consumer.poll = MagicMock(return_value=None)
    mock_consumer.commit = MagicMock()
    mock_consumer.close = MagicMock()
    return mock_consumer


@pytest.fixture
def mock_confluent_admin():
    """Create mock confluent_kafka AdminClient."""
    mock_admin = MagicMock()
    mock_metadata = MagicMock()
    mock_metadata.topics = {"argus.test.executed": MagicMock()}
    mock_admin.list_topics = MagicMock(return_value=mock_metadata)
    mock_admin.create_topics = MagicMock(return_value={})
    return mock_admin


# =============================================================================
# Schema Tests
# =============================================================================


class TestTenantInfo:
    """Tests for TenantInfo schema."""

    def test_tenant_info_creation(self):
        """Should create TenantInfo with required fields."""
        tenant = TenantInfo(org_id="org-123")
        assert tenant.org_id == "org-123"
        assert tenant.project_id is None
        assert tenant.user_id is None

    def test_tenant_info_with_all_fields(self):
        """Should create TenantInfo with all fields."""
        tenant = TenantInfo(
            org_id="org-123",
            project_id="proj-456",
            user_id="user-789",
        )
        assert tenant.org_id == "org-123"
        assert tenant.project_id == "proj-456"
        assert tenant.user_id == "user-789"

    def test_cognee_dataset_prefix_with_project(self):
        """Should generate correct Cognee dataset prefix with project."""
        tenant = TenantInfo(org_id="org-123", project_id="proj-456")
        prefix = tenant.to_cognee_dataset_prefix()
        assert prefix == "org_org-123_project_proj-456"

    def test_cognee_dataset_prefix_without_project(self):
        """Should generate correct Cognee dataset prefix without project."""
        tenant = TenantInfo(org_id="org-123")
        prefix = tenant.to_cognee_dataset_prefix()
        assert prefix == "org_org-123"

    def test_tenant_info_requires_org_id(self):
        """Should require org_id field."""
        with pytest.raises(ValidationError):
            TenantInfo()  # type: ignore


class TestEventMetadata:
    """Tests for EventMetadata schema."""

    def test_event_metadata_creation(self):
        """Should create EventMetadata with required fields."""
        metadata = EventMetadata(source="test-service")
        assert metadata.source == "test-service"
        assert metadata.request_id is not None  # Auto-generated
        assert metadata.timestamp is not None  # Auto-generated

    def test_event_metadata_auto_generates_request_id(self):
        """Should auto-generate request_id if not provided."""
        metadata = EventMetadata(source="test-service")
        assert len(metadata.request_id) == 36  # UUID format

    def test_event_metadata_auto_generates_timestamp(self):
        """Should auto-generate timestamp if not provided."""
        metadata = EventMetadata(source="test-service")
        assert isinstance(metadata.timestamp, datetime)

    def test_event_metadata_with_all_fields(self):
        """Should accept all optional fields."""
        metadata = EventMetadata(
            source="test-service",
            triggered_by="webhook",
            request_id="custom-req-id",
            trace_id="trace-123",
            span_id="span-456",
        )
        assert metadata.triggered_by == "webhook"
        assert metadata.request_id == "custom-req-id"
        assert metadata.trace_id == "trace-123"
        assert metadata.span_id == "span-456"


class TestBaseEvent:
    """Tests for BaseEvent schema."""

    def test_base_event_kafka_key_with_project(self, sample_tenant_info, sample_event_metadata):
        """Should generate correct Kafka key with project_id."""
        event = TestExecutedEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            test_id="test-123",
            run_id="run-456",
            status="passed",
            duration_ms=1000,
        )
        key = event.to_kafka_key()
        assert key == "org-test-123:proj-test-456"

    def test_base_event_kafka_key_without_project(self, sample_event_metadata):
        """Should generate correct Kafka key without project_id."""
        tenant = TenantInfo(org_id="org-123")
        event = TestExecutedEvent(
            tenant=tenant,
            metadata=sample_event_metadata,
            test_id="test-123",
            run_id="run-456",
            status="passed",
            duration_ms=1000,
        )
        key = event.to_kafka_key()
        assert key == "org-123"

    def test_base_event_idempotency_key_default(self, sample_tenant_info, sample_event_metadata):
        """Should use event_id as default idempotency key."""
        event = TestExecutedEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            test_id="test-123",
            run_id="run-456",
            status="passed",
            duration_ms=1000,
        )
        assert event.get_idempotency_key() == event.event_id

    def test_base_event_idempotency_key_custom(self, sample_tenant_info, sample_event_metadata):
        """Should use custom idempotency key when provided."""
        event = TestExecutedEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            test_id="test-123",
            run_id="run-456",
            status="passed",
            duration_ms=1000,
            idempotency_key="custom-key-123",
        )
        assert event.get_idempotency_key() == "custom-key-123"

    def test_base_event_to_dict(self, sample_tenant_info, sample_event_metadata):
        """Should serialize event to dict correctly."""
        event = TestExecutedEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            test_id="test-123",
            run_id="run-456",
            status="passed",
            duration_ms=1000,
        )
        data = event.to_dict()
        assert isinstance(data, dict)
        assert data["event_type"] == "test.executed"
        assert data["test_id"] == "test-123"
        assert "tenant" in data
        assert "metadata" in data


class TestEventTypes:
    """Tests for all event type schemas."""

    def test_codebase_ingested_event(self, sample_tenant_info, sample_event_metadata):
        """Should create valid CodebaseIngestedEvent."""
        event = CodebaseIngestedEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            repository_id="repo-123",
            repository_url="https://github.com/test/repo",
            branch="main",
            commit_sha="abc123",
            file_count=50,
            total_size_bytes=512000,
        )
        assert event.event_type == EventType.CODEBASE_INGESTED
        assert event.repository_id == "repo-123"
        assert event.branch == "main"

    def test_codebase_analyzed_event(self, sample_tenant_info, sample_event_metadata):
        """Should create valid CodebaseAnalyzedEvent."""
        event = CodebaseAnalyzedEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            repository_id="repo-123",
            commit_sha="abc123",
            analysis_duration_ms=5000,
            cognee_dataset_name="org_test_codebase",
        )
        assert event.event_type == EventType.CODEBASE_ANALYZED
        assert event.analysis_duration_ms == 5000

    def test_test_created_event(self, sample_tenant_info, sample_event_metadata):
        """Should create valid TestCreatedEvent."""
        event = TestCreatedEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            test_id="test-123",
            test_name="Login Flow Test",
            test_type="ui",
            framework="playwright",
            priority="high",
        )
        assert event.event_type == EventType.TEST_CREATED
        assert event.test_name == "Login Flow Test"

    def test_test_executed_event(self, sample_tenant_info, sample_event_metadata):
        """Should create valid TestExecutedEvent."""
        event = TestExecutedEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            test_id="test-123",
            run_id="run-456",
            status="passed",
            duration_ms=3500,
        )
        assert event.event_type == EventType.TEST_EXECUTED
        assert event.status == "passed"

    def test_test_failed_event(self, sample_tenant_info, sample_event_metadata):
        """Should create valid TestFailedEvent."""
        event = TestFailedEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            test_id="test-123",
            run_id="run-456",
            error_type="selector",
            error_message="Element not found: #login-btn",
        )
        assert event.event_type == EventType.TEST_FAILED
        assert event.error_type == "selector"

    def test_healing_requested_event(self, sample_tenant_info, sample_event_metadata):
        """Should create valid HealingRequestedEvent."""
        event = HealingRequestedEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            failure_id="fail-123",
            test_id="test-456",
            error_type="selector",
        )
        assert event.event_type == EventType.HEALING_REQUESTED
        assert event.strategy == "auto"  # Default

    def test_healing_completed_event(self, sample_tenant_info, sample_event_metadata):
        """Should create valid HealingCompletedEvent."""
        event = HealingCompletedEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            failure_id="fail-123",
            test_id="test-456",
            healing_request_id="heal-req-789",
            success=True,
            strategy_used="semantic_match",
            duration_ms=2000,
        )
        assert event.event_type == EventType.HEALING_COMPLETED
        assert event.success is True

    def test_dlq_event(self, sample_tenant_info, sample_event_metadata):
        """Should create valid DLQEvent."""
        event = DLQEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            original_event_id="orig-123",
            original_event_type="test.executed",
            original_topic="argus.test.executed",
            original_payload={"test_id": "test-123"},
            error_message="Processing failed",
            error_type="ValidationError",
            first_failed_at=datetime.utcnow(),
        )
        assert event.event_type == EventType.DLQ
        assert event.retry_count == 0  # Default

    def test_agent_request_event(self, sample_tenant_info, sample_event_metadata):
        """Should create valid AgentRequestEvent."""
        event = AgentRequestEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            from_agent="CodeAnalyzer",
            to_agent="SelfHealer",
            request_id="req-123",
            capability="heal_test",
            payload={"test_id": "test-456"},
        )
        assert event.event_type == EventType.AGENT_REQUEST
        assert event.timeout_ms == 30000  # Default

    def test_agent_response_event(self, sample_tenant_info, sample_event_metadata):
        """Should create valid AgentResponseEvent."""
        event = AgentResponseEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            request_id="req-123",
            from_agent="SelfHealer",
            to_agent="CodeAnalyzer",
            success=True,
            payload={"healed": True},
        )
        assert event.event_type == EventType.AGENT_RESPONSE
        assert event.success is True

    def test_agent_broadcast_event(self, sample_tenant_info, sample_event_metadata):
        """Should create valid AgentBroadcastEvent."""
        event = AgentBroadcastEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            from_agent="ConfigManager",
            topic="config_update",
            payload={"setting": "value"},
        )
        assert event.event_type == EventType.AGENT_BROADCAST
        assert event.topic == "config_update"

    def test_agent_heartbeat_event(self, sample_tenant_info, sample_event_metadata):
        """Should create valid AgentHeartbeatEvent."""
        event = AgentHeartbeatEvent(
            tenant=sample_tenant_info,
            metadata=sample_event_metadata,
            agent_id="agent-123",
            agent_type="self_healer",
            capabilities=["heal_test", "analyze_failure"],
            status="healthy",
        )
        assert event.event_type == EventType.AGENT_HEARTBEAT
        assert len(event.capabilities) == 2


# =============================================================================
# Topic Configuration Tests
# =============================================================================


class TestTopicConfiguration:
    """Tests for topic configuration."""

    def test_topic_configs_count(self):
        """Should have correct number of topic configurations."""
        # 17 topics as documented
        assert len(TOPIC_CONFIGS) >= 17

    def test_all_topics_have_partitions(self):
        """All topics should have partition configuration."""
        for topic_name, config in TOPIC_CONFIGS.items():
            assert config.partitions > 0, f"Topic {topic_name} has invalid partitions"

    def test_standard_topics_have_6_partitions(self):
        """Most topics should have 6 partitions for parallelism."""
        standard_topics = [
            TOPIC_CODEBASE_INGESTED,
            TOPIC_CODEBASE_ANALYZED,
            TOPIC_TEST_CREATED,
            TOPIC_TEST_EXECUTED,
            TOPIC_TEST_FAILED,
            TOPIC_HEALING_REQUESTED,
            TOPIC_HEALING_COMPLETED,
        ]
        for topic in standard_topics:
            config = TOPIC_CONFIGS.get(topic)
            assert config is not None, f"Topic {topic} not found"
            assert config.partitions == 6, f"Topic {topic} has {config.partitions} partitions, expected 6"

    def test_dlq_topic_has_fewer_partitions(self):
        """DLQ topic should have fewer partitions."""
        config = TOPIC_CONFIGS.get(TOPIC_DLQ)
        assert config is not None
        assert config.partitions == 3

    def test_topic_configs_have_zstd_compression(self):
        """All topic configs should use ZSTD compression."""
        for topic_name, config in TOPIC_CONFIGS.items():
            assert config.compression_type == "zstd", f"Topic {topic_name} does not use ZSTD"

    def test_topic_retention_periods(self):
        """Topics should have appropriate retention periods."""
        # DLQ should have longest retention (90 days)
        dlq_config = TOPIC_CONFIGS.get(TOPIC_DLQ)
        assert dlq_config is not None
        assert dlq_config.retention_ms == 90 * 24 * 60 * 60 * 1000

        # Heartbeat should have shortest retention (1 hour)
        heartbeat_config = TOPIC_CONFIGS.get(TOPIC_AGENT_HEARTBEAT)
        assert heartbeat_config is not None
        assert heartbeat_config.retention_ms == 1 * 60 * 60 * 1000

    def test_get_topic_for_event_valid_types(self):
        """Should return correct topic for valid event types."""
        assert get_topic_for_event(EventType.CODEBASE_INGESTED) == TOPIC_CODEBASE_INGESTED
        assert get_topic_for_event(EventType.TEST_EXECUTED) == TOPIC_TEST_EXECUTED
        assert get_topic_for_event(EventType.HEALING_REQUESTED) == TOPIC_HEALING_REQUESTED
        assert get_topic_for_event(EventType.DLQ) == TOPIC_DLQ

    def test_get_topic_for_event_a2a_types(self):
        """Should return correct topics for A2A event types."""
        assert get_topic_for_event(EventType.AGENT_REQUEST) == TOPIC_AGENT_REQUEST
        assert get_topic_for_event(EventType.AGENT_RESPONSE) == TOPIC_AGENT_RESPONSE
        assert get_topic_for_event(EventType.AGENT_BROADCAST) == TOPIC_AGENT_BROADCAST
        assert get_topic_for_event(EventType.AGENT_HEARTBEAT) == TOPIC_AGENT_HEARTBEAT

    def test_get_topic_config_returns_config(self):
        """Should return TopicConfig for valid topic."""
        config = get_topic_config(TOPIC_TEST_EXECUTED)
        assert config is not None
        assert isinstance(config, TopicConfig)
        assert config.name == TOPIC_TEST_EXECUTED

    def test_get_topic_config_returns_none_for_unknown(self):
        """Should return None for unknown topic."""
        config = get_topic_config("argus.unknown.topic")
        assert config is None

    def test_get_all_topics(self):
        """Should return list of all topic names."""
        topics = get_all_topics()
        assert isinstance(topics, list)
        assert len(topics) >= 17
        assert TOPIC_TEST_EXECUTED in topics
        assert TOPIC_DLQ in topics

    def test_event_type_to_topic_mapping_complete(self):
        """All event types should have a topic mapping."""
        for event_type in EventType:
            assert event_type in EVENT_TYPE_TO_TOPIC, f"EventType {event_type} has no topic mapping"

    def test_generate_topic_creation_script(self):
        """Should generate valid shell script for topic creation."""
        script = generate_topic_creation_script()
        assert "#!/bin/bash" in script
        assert "rpk topic create" in script
        assert TOPIC_TEST_EXECUTED in script
        assert TOPIC_DLQ in script
        assert "rpk topic list" in script

    def test_topic_config_to_rpk_args(self):
        """TopicConfig should generate valid rpk arguments."""
        config = TopicConfig(
            name="test.topic",
            partitions=6,
            replication_factor=2,
            retention_ms=86400000,
            compression_type="zstd",
        )
        args = config.to_rpk_create_args()
        assert "--partitions 6" in args
        assert "--replicas 2" in args
        assert "retention.ms=86400000" in args
        assert "compression.type=zstd" in args


# =============================================================================
# Event Producer Tests
# =============================================================================


class TestEventProducer:
    """Tests for EventProducer class."""

    @pytest.mark.asyncio
    async def test_producer_requires_start(self):
        """Should raise error if send called before start."""
        from src.events.producer import EventProducer

        producer = EventProducer(bootstrap_servers="localhost:9092")

        tenant = TenantInfo(org_id="org-123")
        metadata = EventMetadata(source="test")
        event = TestExecutedEvent(
            tenant=tenant,
            metadata=metadata,
            test_id="test-123",
            run_id="run-456",
            status="passed",
            duration_ms=1000,
        )

        with pytest.raises(RuntimeError, match="Producer not started"):
            await producer.send(event)

    @pytest.mark.asyncio
    async def test_producer_start_creates_kafka_producer(self, mock_aiokafka_producer):
        """Should create AIOKafkaProducer on start."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            assert producer._started is True
            mock_aiokafka_producer.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_producer_stop_flushes_and_closes(self, mock_aiokafka_producer):
        """Should flush and close producer on stop."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()
            await producer.stop()

            assert producer._started is False
            mock_aiokafka_producer.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_producer_send_success(self, mock_aiokafka_producer, sample_codebase_ingested_event):
        """Should successfully send event to Kafka."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            result = await producer.send(sample_codebase_ingested_event)

            assert result is True
            mock_aiokafka_producer.send_and_wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_producer_send_routes_to_correct_topic(self, mock_aiokafka_producer, sample_codebase_ingested_event):
        """Should route event to correct topic based on event type."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            await producer.send(sample_codebase_ingested_event)

            call_kwargs = mock_aiokafka_producer.send_and_wait.call_args
            assert call_kwargs.kwargs["topic"] == TOPIC_CODEBASE_INGESTED

    @pytest.mark.asyncio
    async def test_producer_send_with_custom_topic(self, mock_aiokafka_producer, sample_codebase_ingested_event):
        """Should use custom topic when provided."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            await producer.send(sample_codebase_ingested_event, topic="custom.topic")

            call_kwargs = mock_aiokafka_producer.send_and_wait.call_args
            assert call_kwargs.kwargs["topic"] == "custom.topic"

    @pytest.mark.asyncio
    async def test_producer_send_includes_headers(self, mock_aiokafka_producer, sample_codebase_ingested_event):
        """Should include event metadata in headers."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            await producer.send(sample_codebase_ingested_event)

            call_kwargs = mock_aiokafka_producer.send_and_wait.call_args
            headers = call_kwargs.kwargs["headers"]
            header_names = [h[0] for h in headers]

            assert "event_id" in header_names
            assert "event_type" in header_names
            assert "org_id" in header_names
            assert "timestamp" in header_names

    @pytest.mark.asyncio
    async def test_producer_send_with_custom_headers(self, mock_aiokafka_producer, sample_codebase_ingested_event):
        """Should include custom headers when provided."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            await producer.send(
                sample_codebase_ingested_event,
                headers={"custom-header": "custom-value"},
            )

            call_kwargs = mock_aiokafka_producer.send_and_wait.call_args
            headers = call_kwargs.kwargs["headers"]
            header_dict = {h[0]: h[1].decode() for h in headers}

            assert header_dict.get("custom-header") == "custom-value"

    @pytest.mark.asyncio
    async def test_producer_send_handles_kafka_error(self, mock_aiokafka_producer, sample_codebase_ingested_event):
        """Should return False on Kafka error."""
        from aiokafka.errors import KafkaError as AIOKafkaError

        from src.events.producer import EventProducer

        mock_aiokafka_producer.send_and_wait = AsyncMock(side_effect=AIOKafkaError())

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            result = await producer.send(sample_codebase_ingested_event)

            assert result is False

    @pytest.mark.asyncio
    async def test_producer_context_manager(self, mock_aiokafka_producer):
        """Should work as async context manager."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            async with EventProducer.create(bootstrap_servers="localhost:9092") as producer:
                assert producer._started is True

            mock_aiokafka_producer.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_producer_sasl_configuration(self, mock_aiokafka_producer):
        """Should configure SASL when credentials provided."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer) as mock_cls:
            producer = EventProducer(
                bootstrap_servers="localhost:9092",
                sasl_username="user",
                sasl_password="pass",
                security_protocol="SASL_SSL",
                sasl_mechanism="SCRAM-SHA-512",
            )
            await producer.start()

            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["security_protocol"] == "SASL_SSL"
            assert call_kwargs["sasl_mechanism"] == "SCRAM-SHA-512"

    @pytest.mark.asyncio
    async def test_producer_compression_configuration(self, mock_aiokafka_producer):
        """Should configure ZSTD compression by default."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer) as mock_cls:
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["compression_type"] == "zstd"

    @pytest.mark.asyncio
    async def test_producer_idempotent_start(self, mock_aiokafka_producer):
        """Should be safe to call start multiple times."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()
            await producer.start()
            await producer.start()

            # Should only create producer once
            assert mock_aiokafka_producer.start.call_count == 1

    @pytest.mark.asyncio
    async def test_producer_idempotent_stop(self, mock_aiokafka_producer):
        """Should be safe to call stop multiple times."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()
            await producer.stop()
            await producer.stop()
            await producer.stop()

            # Should only stop once
            assert mock_aiokafka_producer.stop.call_count == 1


# =============================================================================
# Dead Letter Queue Tests
# =============================================================================


class TestDLQHandling:
    """Tests for Dead Letter Queue handling."""

    @pytest.mark.asyncio
    async def test_send_to_dlq_success(self, mock_aiokafka_producer, sample_codebase_ingested_event):
        """Should successfully send failed event to DLQ."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            result = await producer.send_to_dlq(
                original_event=sample_codebase_ingested_event,
                error=ValueError("Processing failed"),
                original_topic=TOPIC_CODEBASE_INGESTED,
            )

            assert result is True
            mock_aiokafka_producer.send_and_wait.assert_called()

    @pytest.mark.asyncio
    async def test_send_to_dlq_routes_to_dlq_topic(self, mock_aiokafka_producer, sample_codebase_ingested_event):
        """Should route DLQ event to DLQ topic."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            await producer.send_to_dlq(
                original_event=sample_codebase_ingested_event,
                error=ValueError("Processing failed"),
                original_topic=TOPIC_CODEBASE_INGESTED,
            )

            call_kwargs = mock_aiokafka_producer.send_and_wait.call_args
            assert call_kwargs.kwargs["topic"] == TOPIC_DLQ

    @pytest.mark.asyncio
    async def test_send_to_dlq_includes_original_event_info(self, mock_aiokafka_producer, sample_codebase_ingested_event):
        """Should include original event information in DLQ event."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            await producer.send_to_dlq(
                original_event=sample_codebase_ingested_event,
                error=ValueError("Processing failed"),
                original_topic=TOPIC_CODEBASE_INGESTED,
            )

            call_kwargs = mock_aiokafka_producer.send_and_wait.call_args
            dlq_event_data = call_kwargs.kwargs["value"]

            assert dlq_event_data["original_event_id"] == sample_codebase_ingested_event.event_id
            assert dlq_event_data["original_topic"] == TOPIC_CODEBASE_INGESTED
            assert "Processing failed" in dlq_event_data["error_message"]

    @pytest.mark.asyncio
    async def test_send_to_dlq_includes_stack_trace(self, mock_aiokafka_producer, sample_codebase_ingested_event):
        """Should include stack trace in DLQ event."""
        from src.events.producer import EventProducer

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            try:
                raise ValueError("Test error with stack trace")
            except ValueError as e:
                await producer.send_to_dlq(
                    original_event=sample_codebase_ingested_event,
                    error=e,
                    original_topic=TOPIC_CODEBASE_INGESTED,
                )

            call_kwargs = mock_aiokafka_producer.send_and_wait.call_args
            dlq_event_data = call_kwargs.kwargs["value"]

            assert dlq_event_data.get("stack_trace") is not None

    @pytest.mark.asyncio
    async def test_send_to_dlq_retries_on_failure(self, mock_aiokafka_producer, sample_codebase_ingested_event):
        """Should retry DLQ write on failure."""
        from aiokafka.errors import KafkaError as AIOKafkaError

        from src.events.producer import EventProducer

        # First two calls fail, third succeeds
        mock_result = MagicMock()
        mock_result.partition = 0
        mock_result.offset = 42
        mock_aiokafka_producer.send_and_wait = AsyncMock(
            side_effect=[AIOKafkaError(), AIOKafkaError(), mock_result]
        )

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            result = await producer.send_to_dlq(
                original_event=sample_codebase_ingested_event,
                error=ValueError("Processing failed"),
                original_topic=TOPIC_CODEBASE_INGESTED,
                dlq_write_retries=3,
                dlq_retry_delay=0.01,  # Short delay for tests
            )

            assert result is True
            assert mock_aiokafka_producer.send_and_wait.call_count == 3

    @pytest.mark.asyncio
    async def test_send_to_dlq_returns_false_after_max_retries(self, mock_aiokafka_producer, sample_codebase_ingested_event):
        """Should return False after exhausting retries."""
        from aiokafka.errors import KafkaError as AIOKafkaError

        from src.events.producer import EventProducer

        mock_aiokafka_producer.send_and_wait = AsyncMock(side_effect=AIOKafkaError())

        with patch("src.events.producer.AIOKafkaProducer", return_value=mock_aiokafka_producer):
            producer = EventProducer(bootstrap_servers="localhost:9092")
            await producer.start()

            result = await producer.send_to_dlq(
                original_event=sample_codebase_ingested_event,
                error=ValueError("Processing failed"),
                original_topic=TOPIC_CODEBASE_INGESTED,
                dlq_write_retries=3,
                dlq_retry_delay=0.01,
            )

            assert result is False


# =============================================================================
# Redpanda Client Tests
# =============================================================================


class TestRedpandaClient:
    """Tests for RedpandaClient class."""

    def test_redpanda_config_defaults(self):
        """Should use default values when not provided."""
        from src.events.redpanda_client import RedpandaConfig

        with patch.dict(os.environ, {}, clear=True):
            config = RedpandaConfig()
            assert config.bootstrap_servers == "localhost:9092"
            assert config.sasl_mechanism == "SCRAM-SHA-256"

    def test_redpanda_config_from_env(self):
        """Should read configuration from environment."""
        from src.events.redpanda_client import RedpandaConfig

        with patch.dict(os.environ, {
            "REDPANDA_BROKERS": "broker1:9092,broker2:9092",
            "REDPANDA_SASL_USERNAME": "testuser",
            "REDPANDA_SASL_PASSWORD": "testpass",
        }):
            config = RedpandaConfig()
            assert config.bootstrap_servers == "broker1:9092,broker2:9092"
            assert config.sasl_username == "testuser"
            assert config.sasl_password == "testpass"

    def test_redpanda_config_to_confluent_config(self):
        """Should convert to confluent-kafka config dict."""
        from src.events.redpanda_client import RedpandaConfig

        config = RedpandaConfig(
            bootstrap_servers="localhost:9092",
            sasl_username="user",
            sasl_password="pass",
        )
        confluent_config = config.to_confluent_config()

        assert confluent_config["bootstrap.servers"] == "localhost:9092"
        assert confluent_config["security.protocol"] == "SASL_SSL"
        assert confluent_config["sasl.username"] == "user"

    def test_redpanda_client_topic_configs(self):
        """RedpandaClient should have topic configurations."""
        from src.events.redpanda_client import RedpandaClient

        assert len(RedpandaClient.TOPIC_CONFIGS) > 0
        assert "argus.test.executed" in RedpandaClient.TOPIC_CONFIGS

    def test_argus_event_to_dict(self):
        """ArgusEvent should serialize to dict correctly."""
        from src.events.redpanda_client import ArgusEvent

        event = ArgusEvent(
            event_type="test.executed",
            org_id="org-123",
            project_id="proj-456",
            payload={"test_id": "test-789", "status": "passed"},
        )
        data = event.to_dict()

        assert data["event_type"] == "test.executed"
        assert data["org_id"] == "org-123"
        assert data["project_id"] == "proj-456"
        assert "event_id" in data
        assert "timestamp" in data

    def test_argus_event_to_json(self):
        """ArgusEvent should serialize to JSON bytes."""
        from src.events.redpanda_client import ArgusEvent

        event = ArgusEvent(
            event_type="test.executed",
            org_id="org-123",
            payload={"status": "passed"},
        )
        json_bytes = event.to_json()

        assert isinstance(json_bytes, bytes)
        decoded = json.loads(json_bytes.decode("utf-8"))
        assert decoded["event_type"] == "test.executed"

    def test_argus_event_from_json(self):
        """ArgusEvent should deserialize from JSON bytes."""
        from src.events.redpanda_client import ArgusEvent

        original = ArgusEvent(
            event_type="test.executed",
            org_id="org-123",
            payload={"status": "passed"},
        )
        json_bytes = original.to_json()
        restored = ArgusEvent.from_json(json_bytes)

        assert restored.event_type == original.event_type
        assert restored.org_id == original.org_id
        assert restored.event_id == original.event_id

    def test_argus_event_topic_property(self):
        """ArgusEvent should generate correct topic name."""
        from src.events.redpanda_client import ArgusEvent

        event = ArgusEvent(
            event_type="test.executed",
            org_id="org-123",
            payload={},
        )
        assert event.topic == "argus.test.executed"

    def test_redpanda_client_list_topics(self, mock_confluent_admin):
        """Should list topics from cluster."""
        from src.events.redpanda_client import RedpandaClient

        with patch.object(RedpandaClient, "_get_admin", return_value=mock_confluent_admin):
            client = RedpandaClient()
            topics = client.list_topics()

            assert isinstance(topics, list)
            mock_confluent_admin.list_topics.assert_called_once()

    def test_redpanda_client_create_topics(self, mock_confluent_admin):
        """Should create topics with proper configuration."""
        from src.events.redpanda_client import RedpandaClient

        mock_future = MagicMock()
        mock_future.result = MagicMock(return_value=None)
        mock_confluent_admin.create_topics.return_value = {"argus.test.executed": mock_future}

        with patch.object(RedpandaClient, "_get_admin", return_value=mock_confluent_admin):
            client = RedpandaClient()
            results = client.create_topics(["argus.test.executed"])

            assert "argus.test.executed" in results
            mock_confluent_admin.create_topics.assert_called_once()

    def test_redpanda_client_publish(self, mock_confluent_producer):
        """Should publish event to Kafka."""
        from src.events.redpanda_client import ArgusEvent, RedpandaClient

        with patch.object(RedpandaClient, "_get_producer", return_value=mock_confluent_producer):
            client = RedpandaClient()
            event = ArgusEvent(
                event_type="test.executed",
                org_id="org-123",
                payload={"status": "passed"},
            )
            client.publish(event)

            mock_confluent_producer.produce.assert_called_once()
            mock_confluent_producer.poll.assert_called_once()

    def test_redpanda_client_publish_sync(self, mock_confluent_producer):
        """Should publish event synchronously and wait for confirmation."""
        from src.events.redpanda_client import ArgusEvent, RedpandaClient

        with patch.object(RedpandaClient, "_get_producer", return_value=mock_confluent_producer):
            client = RedpandaClient()
            event = ArgusEvent(
                event_type="test.executed",
                org_id="org-123",
                payload={"status": "passed"},
            )

            # Simulate successful delivery
            def mock_produce(topic, key, value, callback):
                callback(None, MagicMock(topic=lambda: topic, partition=lambda: 0, offset=lambda: 42))

            mock_confluent_producer.produce = mock_produce

            result = client.publish_sync(event)
            assert result is True

    def test_redpanda_client_flush(self, mock_confluent_producer):
        """Should flush pending messages."""
        from src.events.redpanda_client import RedpandaClient

        with patch.object(RedpandaClient, "_get_producer", return_value=mock_confluent_producer):
            client = RedpandaClient()
            client._producer = mock_confluent_producer
            remaining = client.flush(timeout=5.0)

            mock_confluent_producer.flush.assert_called_once_with(timeout=5.0)

    def test_redpanda_client_close(self, mock_confluent_producer):
        """Should close all connections."""
        from src.events.redpanda_client import RedpandaClient

        client = RedpandaClient()
        client._producer = mock_confluent_producer
        client._consumers = {"test": MagicMock()}
        client._admin = MagicMock()

        client.close()

        assert client._producer is None
        assert client._admin is None
        assert len(client._consumers) == 0

    def test_redpanda_client_context_manager(self, mock_confluent_producer):
        """Should work as context manager."""
        from src.events.redpanda_client import RedpandaClient

        with patch.object(RedpandaClient, "_get_producer", return_value=mock_confluent_producer):
            with RedpandaClient() as client:
                # Initialize producer by calling _get_producer
                client._producer = mock_confluent_producer
                assert client is not None

            # Should have closed after context exit - flush is called during close()
            mock_confluent_producer.flush.assert_called()


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_event_metadata(self):
        """Should create EventMetadata with helpers."""
        from src.events.producer import create_event_metadata

        metadata = create_event_metadata(
            source="test-service",
            triggered_by="webhook",
            request_id="custom-req-id",
        )

        assert metadata.source == "test-service"
        assert metadata.triggered_by == "webhook"
        assert metadata.request_id == "custom-req-id"

    def test_create_event_metadata_auto_generates_request_id(self):
        """Should auto-generate request_id if not provided."""
        from src.events.producer import create_event_metadata

        metadata = create_event_metadata(source="test-service")

        assert metadata.request_id is not None
        assert len(metadata.request_id) == 36  # UUID format

    def test_create_tenant_info(self):
        """Should create TenantInfo with helpers."""
        from src.events.producer import create_tenant_info

        tenant = create_tenant_info(
            org_id="org-123",
            project_id="proj-456",
            user_id="user-789",
        )

        assert tenant.org_id == "org-123"
        assert tenant.project_id == "proj-456"
        assert tenant.user_id == "user-789"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions in redpanda_client."""

    def test_emit_test_created(self, mock_confluent_producer):
        """Should emit test.created event."""
        from src.events.redpanda_client import RedpandaClient, emit_test_created

        with patch.object(RedpandaClient, "_get_producer", return_value=mock_confluent_producer):
            client = RedpandaClient()
            emit_test_created(
                client=client,
                org_id="org-123",
                project_id="proj-456",
                test_id="test-789",
                test_name="Login Test",
                test_type="ui",
            )

            mock_confluent_producer.produce.assert_called_once()

    def test_emit_test_executed(self, mock_confluent_producer):
        """Should emit test.executed event."""
        from src.events.redpanda_client import RedpandaClient, emit_test_executed

        with patch.object(RedpandaClient, "_get_producer", return_value=mock_confluent_producer):
            client = RedpandaClient()
            emit_test_executed(
                client=client,
                org_id="org-123",
                project_id="proj-456",
                test_id="test-789",
                run_id="run-001",
                status="passed",
                duration_ms=1500,
            )

            mock_confluent_producer.produce.assert_called_once()

    def test_emit_codebase_ingested(self, mock_confluent_producer):
        """Should emit codebase.ingested event."""
        from src.events.redpanda_client import RedpandaClient, emit_codebase_ingested

        with patch.object(RedpandaClient, "_get_producer", return_value=mock_confluent_producer):
            client = RedpandaClient()
            emit_codebase_ingested(
                client=client,
                org_id="org-123",
                project_id="proj-456",
                repo_url="https://github.com/test/repo",
                branch="main",
                commit_sha="abc123",
            )

            mock_confluent_producer.produce.assert_called_once()
