"""Comprehensive tests for Agent-to-Agent (A2A) Protocol.

This module tests the A2A Protocol and Agent Registry components:
- Agent registration and deregistration
- Capability discovery (41 capabilities)
- Health monitoring with 60-second heartbeat timeout
- A2A request/response messaging
- Broadcast pub/sub
- Circuit breaker pattern (5 failures, 30s reset)
- Service discovery for capability-based routing
"""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# A2A Protocol Tests
# =============================================================================


class TestA2AProtocolMessages:
    """Tests for A2A message schemas."""

    def test_agent_request_event_defaults(self, mock_env_vars):
        """Test AgentRequestEvent has correct default values."""
        from src.orchestrator.a2a_protocol import AgentRequestEvent, MessageType

        request = AgentRequestEvent(
            from_agent="agent-1",
            from_agent_type="test_agent",
            to_agent="agent-2",
            capability="test_capability",
        )

        assert request.message_type == MessageType.REQUEST
        assert request.from_agent == "agent-1"
        assert request.to_agent == "agent-2"
        assert request.capability == "test_capability"
        assert request.payload == {}
        assert request.timeout_ms == 30000
        assert request.priority == 5
        assert request.request_id is not None
        assert request.correlation_id is None

    def test_agent_request_event_to_kafka_key(self, mock_env_vars):
        """Test AgentRequestEvent generates correct Kafka key."""
        from src.orchestrator.a2a_protocol import AgentRequestEvent

        request = AgentRequestEvent(
            from_agent="sender",
            from_agent_type="test",
            to_agent="receiver",
            capability="test_cap",
        )

        assert request.to_kafka_key() == "receiver"

    def test_agent_request_event_to_dict(self, mock_env_vars):
        """Test AgentRequestEvent serialization to dict."""
        from src.orchestrator.a2a_protocol import AgentRequestEvent

        request = AgentRequestEvent(
            from_agent="agent-1",
            from_agent_type="tester",
            to_agent="agent-2",
            capability="heal",
            payload={"test": "data"},
            priority=3,
        )

        data = request.to_dict()

        assert data["from_agent"] == "agent-1"
        assert data["to_agent"] == "agent-2"
        assert data["capability"] == "heal"
        assert data["payload"] == {"test": "data"}
        assert data["priority"] == 3

    def test_agent_response_defaults(self, mock_env_vars):
        """Test AgentResponse has correct default values."""
        from src.orchestrator.a2a_protocol import AgentResponse, MessageType

        response = AgentResponse(
            request_id="req-123",
            from_agent="agent-2",
            from_agent_type="healer",
            to_agent="agent-1",
            success=True,
        )

        assert response.message_type == MessageType.RESPONSE
        assert response.request_id == "req-123"
        assert response.from_agent == "agent-2"
        assert response.to_agent == "agent-1"
        assert response.success is True
        assert response.payload == {}
        assert response.error is None
        assert response.error_code is None
        assert response.duration_ms == 0

    def test_agent_response_to_kafka_key(self, mock_env_vars):
        """Test AgentResponse generates correct Kafka key."""
        from src.orchestrator.a2a_protocol import AgentResponse

        response = AgentResponse(
            request_id="req-123",
            from_agent="responder",
            from_agent_type="test",
            to_agent="requester",
            success=True,
        )

        assert response.to_kafka_key() == "requester"

    def test_agent_broadcast_defaults(self, mock_env_vars):
        """Test AgentBroadcast has correct default values."""
        from src.orchestrator.a2a_protocol import AgentBroadcast, MessageType

        broadcast = AgentBroadcast(
            from_agent="broadcaster",
            from_agent_type="reporter",
            topic="test.event",
            payload={"status": "completed"},
        )

        assert broadcast.message_type == MessageType.BROADCAST
        assert broadcast.from_agent == "broadcaster"
        assert broadcast.topic == "test.event"
        assert broadcast.payload == {"status": "completed"}
        assert broadcast.broadcast_id is not None

    def test_agent_broadcast_to_kafka_key(self, mock_env_vars):
        """Test AgentBroadcast generates correct Kafka key based on topic."""
        from src.orchestrator.a2a_protocol import AgentBroadcast

        broadcast = AgentBroadcast(
            from_agent="agent-1",
            from_agent_type="test",
            topic="test.completed",
            payload={},
        )

        assert broadcast.to_kafka_key() == "test.completed"

    def test_agent_heartbeat_defaults(self, mock_env_vars):
        """Test AgentHeartbeat has correct default values."""
        from src.orchestrator.a2a_protocol import AgentHeartbeat

        heartbeat = AgentHeartbeat(
            agent_id="agent-1",
            agent_type="tester",
            capabilities=["code_analysis", "test_generation"],
        )

        assert heartbeat.agent_id == "agent-1"
        assert heartbeat.agent_type == "tester"
        assert heartbeat.status == "healthy"
        assert heartbeat.capabilities == ["code_analysis", "test_generation"]
        assert heartbeat.load == 0.0


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for Circuit Breaker pattern implementation."""

    def test_circuit_breaker_initial_state(self, mock_env_vars):
        """Test circuit breaker starts in closed state."""
        from src.orchestrator.a2a_protocol import CircuitBreaker, CircuitState

        cb = CircuitBreaker(agent_id="test-agent")

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.can_execute() is True

    def test_circuit_breaker_records_success(self, mock_env_vars):
        """Test circuit breaker records successful requests."""
        from src.orchestrator.a2a_protocol import CircuitBreaker

        cb = CircuitBreaker(agent_id="test-agent")
        cb.failure_count = 3

        cb.record_success()

        assert cb.failure_count == 0

    def test_circuit_breaker_records_failure(self, mock_env_vars):
        """Test circuit breaker records failed requests."""
        from src.orchestrator.a2a_protocol import CircuitBreaker

        cb = CircuitBreaker(agent_id="test-agent")

        cb.record_failure()

        assert cb.failure_count == 1
        assert cb.last_failure_time is not None

    def test_circuit_breaker_opens_after_threshold(self, mock_env_vars):
        """Test circuit breaker opens after 5 failures (default threshold)."""
        from src.orchestrator.a2a_protocol import CircuitBreaker, CircuitState

        cb = CircuitBreaker(agent_id="test-agent", failure_threshold=5)

        # Record 5 failures
        for _ in range(5):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_circuit_breaker_custom_threshold(self, mock_env_vars):
        """Test circuit breaker with custom failure threshold."""
        from src.orchestrator.a2a_protocol import CircuitBreaker, CircuitState

        cb = CircuitBreaker(agent_id="test-agent", failure_threshold=3)

        # Record 3 failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_recovery_timeout(self, mock_env_vars):
        """Test circuit breaker transitions to half-open after recovery timeout."""
        from src.orchestrator.a2a_protocol import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            agent_id="test-agent",
            failure_threshold=1,
            recovery_timeout=0.1,  # 100ms for testing
        )

        # Open the circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should transition to half-open
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_half_open_success(self, mock_env_vars):
        """Test circuit breaker closes after successful requests in half-open state."""
        from src.orchestrator.a2a_protocol import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            agent_id="test-agent",
            failure_threshold=1,
            recovery_timeout=0.1,
            half_open_max_requests=2,
        )

        # Open the circuit
        cb.record_failure()
        time.sleep(0.15)

        # Transition to half-open
        cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

        # Record successful requests
        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_half_open_failure(self, mock_env_vars):
        """Test circuit breaker re-opens on failure in half-open state."""
        from src.orchestrator.a2a_protocol import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            agent_id="test-agent",
            failure_threshold=1,
            recovery_timeout=0.1,
        )

        # Open the circuit
        cb.record_failure()
        time.sleep(0.15)

        # Transition to half-open
        cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

        # Record failure in half-open state
        cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_half_open_max_requests(self, mock_env_vars):
        """Test circuit breaker limits requests in half-open state."""
        from src.orchestrator.a2a_protocol import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            agent_id="test-agent",
            failure_threshold=1,
            recovery_timeout=0.1,
            half_open_max_requests=2,
        )

        # Open and transition to half-open
        cb.record_failure()
        time.sleep(0.15)
        cb.can_execute()

        # First request allowed
        assert cb.can_execute() is True
        # Second request allowed
        assert cb.can_execute() is True
        # Third request should be rejected
        assert cb.can_execute() is False


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_circuit_open_error_message(self, mock_env_vars):
        """Test CircuitOpenError has correct message format."""
        from src.orchestrator.a2a_protocol import CircuitOpenError

        error = CircuitOpenError(agent_id="test-agent", retry_after=30.5)

        assert error.agent_id == "test-agent"
        assert error.retry_after == 30.5
        assert "test-agent" in str(error)
        assert "30.5" in str(error)


# =============================================================================
# A2A Protocol Core Tests
# =============================================================================


class TestA2AProtocolInit:
    """Tests for A2A Protocol initialization."""

    def test_protocol_init_defaults(self, mock_env_vars):
        """Test A2A Protocol initializes with correct defaults."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="test-agent",
            agent_type="test_type",
        )

        assert protocol.agent_id == "test-agent"
        assert protocol.agent_type == "test_type"
        assert protocol.bootstrap_servers == "localhost:9092"
        assert protocol.client_id == "a2a-test-agent"
        assert protocol.capabilities == []
        assert protocol._started is False

    def test_protocol_init_with_custom_config(self, mock_env_vars):
        """Test A2A Protocol initializes with custom configuration."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="custom-agent",
            agent_type="custom_type",
            bootstrap_servers="broker:9093",
            sasl_username="user",
            sasl_password="pass",
            client_id="custom-client",
            capabilities=["code_analysis", "test_generation"],
            circuit_breaker_threshold=10,
            circuit_breaker_timeout=60.0,
        )

        assert protocol.agent_id == "custom-agent"
        assert protocol.bootstrap_servers == "broker:9093"
        assert protocol.client_id == "custom-client"
        assert protocol.capabilities == ["code_analysis", "test_generation"]
        assert protocol._cb_threshold == 10
        assert protocol._cb_timeout == 60.0

    def test_protocol_get_circuit_breaker(self, mock_env_vars):
        """Test A2A Protocol creates circuit breakers per agent."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="test-agent",
            agent_type="test",
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=30.0,
        )

        cb1 = protocol._get_circuit_breaker("agent-1")
        cb2 = protocol._get_circuit_breaker("agent-2")
        cb3 = protocol._get_circuit_breaker("agent-1")  # Same agent

        assert cb1 is cb3  # Same circuit breaker for same agent
        assert cb1 is not cb2  # Different circuit breakers for different agents
        assert cb1.failure_threshold == 5
        assert cb1.recovery_timeout == 30.0


class TestA2AProtocolStartStop:
    """Tests for A2A Protocol start/stop operations."""

    @pytest.mark.asyncio
    async def test_protocol_start_creates_kafka_clients(self, mock_env_vars):
        """Test A2A Protocol creates Kafka producer and consumer on start."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="test-agent",
            agent_type="test",
        )

        mock_producer = AsyncMock()
        mock_consumer = AsyncMock()
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch("src.orchestrator.a2a_protocol.AIOKafkaProducer", return_value=mock_producer):
            with patch("src.orchestrator.a2a_protocol.AIOKafkaConsumer", return_value=mock_consumer):
                await protocol.start()

                assert protocol._started is True
                assert protocol._producer is not None
                assert protocol._consumer is not None
                mock_producer.start.assert_called_once()
                mock_consumer.start.assert_called_once()

                await protocol.stop()

    @pytest.mark.asyncio
    async def test_protocol_stop_cleanup(self, mock_env_vars):
        """Test A2A Protocol properly cleans up on stop."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="test-agent",
            agent_type="test",
        )

        mock_producer = AsyncMock()
        mock_consumer = AsyncMock()
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch("src.orchestrator.a2a_protocol.AIOKafkaProducer", return_value=mock_producer):
            with patch("src.orchestrator.a2a_protocol.AIOKafkaConsumer", return_value=mock_consumer):
                await protocol.start()
                await protocol.stop()

                assert protocol._started is False
                mock_producer.stop.assert_called_once()
                mock_consumer.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_protocol_context_manager(self, mock_env_vars):
        """Test A2A Protocol works as async context manager."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        mock_producer = AsyncMock()
        mock_consumer = AsyncMock()
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch("src.orchestrator.a2a_protocol.AIOKafkaProducer", return_value=mock_producer):
            with patch("src.orchestrator.a2a_protocol.AIOKafkaConsumer", return_value=mock_consumer):
                async with A2AProtocol.create(
                    agent_id="test-agent",
                    agent_type="test",
                ) as protocol:
                    assert protocol._started is True

                # After exiting context, should be stopped
                mock_producer.stop.assert_called()
                mock_consumer.stop.assert_called()


class TestA2AProtocolRequest:
    """Tests for A2A Protocol request/response functionality."""

    @pytest.mark.asyncio
    async def test_request_raises_if_not_started(self, mock_env_vars):
        """Test request raises error if protocol not started."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="test-agent",
            agent_type="test",
        )

        with pytest.raises(RuntimeError, match="Protocol not started"):
            await protocol.request(
                to_agent="other-agent",
                capability="test",
                payload={},
            )

    @pytest.mark.asyncio
    async def test_request_checks_circuit_breaker(self, mock_env_vars):
        """Test request checks circuit breaker before sending."""
        from src.orchestrator.a2a_protocol import A2AProtocol, CircuitOpenError

        protocol = A2AProtocol(
            agent_id="test-agent",
            agent_type="test",
            circuit_breaker_threshold=1,
        )

        mock_producer = AsyncMock()
        mock_consumer = AsyncMock()
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch("src.orchestrator.a2a_protocol.AIOKafkaProducer", return_value=mock_producer):
            with patch("src.orchestrator.a2a_protocol.AIOKafkaConsumer", return_value=mock_consumer):
                await protocol.start()

                # Open the circuit breaker
                cb = protocol._get_circuit_breaker("target-agent")
                cb.record_failure()

                with pytest.raises(CircuitOpenError):
                    await protocol.request(
                        to_agent="target-agent",
                        capability="test",
                        payload={},
                    )

                await protocol.stop()

    @pytest.mark.asyncio
    async def test_request_sends_to_kafka(self, mock_env_vars):
        """Test request sends message to Kafka."""
        from src.orchestrator.a2a_protocol import A2AProtocol, TOPIC_AGENT_REQUEST

        protocol = A2AProtocol(
            agent_id="sender-agent",
            agent_type="test",
        )

        mock_producer = AsyncMock()
        mock_consumer = AsyncMock()
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch("src.orchestrator.a2a_protocol.AIOKafkaProducer", return_value=mock_producer):
            with patch("src.orchestrator.a2a_protocol.AIOKafkaConsumer", return_value=mock_consumer):
                await protocol.start()

                # Start request in background (will timeout)
                request_task = asyncio.create_task(
                    protocol.request(
                        to_agent="receiver-agent",
                        capability="test_capability",
                        payload={"test": "data"},
                        timeout=0.1,
                    )
                )

                # Give time for send to happen
                await asyncio.sleep(0.05)

                # Verify send was called
                mock_producer.send_and_wait.assert_called()
                call_args = mock_producer.send_and_wait.call_args

                assert call_args.kwargs["topic"] == TOPIC_AGENT_REQUEST
                assert call_args.kwargs["key"] == "receiver-agent"

                # Cancel the task (it would timeout)
                request_task.cancel()
                try:
                    await request_task
                except asyncio.CancelledError:
                    pass

                await protocol.stop()


class TestA2AProtocolBroadcast:
    """Tests for A2A Protocol broadcast functionality."""

    @pytest.mark.asyncio
    async def test_broadcast_raises_if_not_started(self, mock_env_vars):
        """Test broadcast raises error if protocol not started."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="test-agent",
            agent_type="test",
        )

        with pytest.raises(RuntimeError, match="Protocol not started"):
            await protocol.broadcast(
                topic="test.event",
                message={"data": "test"},
            )

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_kafka(self, mock_env_vars):
        """Test broadcast sends message to Kafka broadcast topic."""
        from src.orchestrator.a2a_protocol import A2AProtocol, TOPIC_AGENT_BROADCAST

        protocol = A2AProtocol(
            agent_id="broadcaster",
            agent_type="test",
        )

        mock_producer = AsyncMock()
        mock_consumer = AsyncMock()
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch("src.orchestrator.a2a_protocol.AIOKafkaProducer", return_value=mock_producer):
            with patch("src.orchestrator.a2a_protocol.AIOKafkaConsumer", return_value=mock_consumer):
                await protocol.start()

                await protocol.broadcast(
                    topic="test.completed",
                    message={"status": "success", "test_id": "t-123"},
                )

                mock_producer.send_and_wait.assert_called()
                call_args = mock_producer.send_and_wait.call_args

                assert call_args.kwargs["topic"] == TOPIC_AGENT_BROADCAST
                assert call_args.kwargs["key"] == "test.completed"

                await protocol.stop()


class TestA2AProtocolSubscribe:
    """Tests for A2A Protocol subscription functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_registers_handler(self, mock_env_vars):
        """Test subscribe registers handler for topic."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="test-agent",
            agent_type="test",
        )

        async def handler(msg):
            pass

        await protocol.subscribe("test.event", handler)

        assert "test.event" in protocol._subscriptions
        assert handler in protocol._subscriptions["test.event"]

    @pytest.mark.asyncio
    async def test_subscribe_multiple_handlers(self, mock_env_vars):
        """Test multiple handlers can subscribe to same topic."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="test-agent",
            agent_type="test",
        )

        async def handler1(msg):
            pass

        async def handler2(msg):
            pass

        await protocol.subscribe("test.event", handler1)
        await protocol.subscribe("test.event", handler2)

        assert len(protocol._subscriptions["test.event"]) == 2


class TestA2AProtocolRequestHandler:
    """Tests for A2A Protocol request handler functionality."""

    def test_set_request_handler(self, mock_env_vars):
        """Test set_request_handler registers handler."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="test-agent",
            agent_type="test",
        )

        async def handler(request):
            return {"result": "success"}

        protocol.set_request_handler(handler)

        assert protocol._request_handler is handler


class TestA2AProtocolRespond:
    """Tests for A2A Protocol respond functionality."""

    @pytest.mark.asyncio
    async def test_respond_raises_if_not_started(self, mock_env_vars):
        """Test respond raises error if protocol not started."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="test-agent",
            agent_type="test",
        )

        with pytest.raises(RuntimeError, match="Protocol not started"):
            await protocol.respond(
                request_id="req-123",
                to_agent="requester",
                success=True,
                payload={"result": "done"},
            )

    @pytest.mark.asyncio
    async def test_respond_sends_to_kafka(self, mock_env_vars):
        """Test respond sends response message to Kafka."""
        from src.orchestrator.a2a_protocol import A2AProtocol, TOPIC_AGENT_RESPONSE

        protocol = A2AProtocol(
            agent_id="responder",
            agent_type="test",
        )

        mock_producer = AsyncMock()
        mock_consumer = AsyncMock()
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch("src.orchestrator.a2a_protocol.AIOKafkaProducer", return_value=mock_producer):
            with patch("src.orchestrator.a2a_protocol.AIOKafkaConsumer", return_value=mock_consumer):
                await protocol.start()

                await protocol.respond(
                    request_id="req-123",
                    to_agent="requester",
                    success=True,
                    payload={"result": "completed"},
                    duration_ms=100,
                )

                mock_producer.send_and_wait.assert_called()
                call_args = mock_producer.send_and_wait.call_args

                assert call_args.kwargs["topic"] == TOPIC_AGENT_RESPONSE
                assert call_args.kwargs["key"] == "requester"

                await protocol.stop()


class TestA2AProtocolConvenienceMethods:
    """Tests for A2A Protocol convenience methods."""

    @pytest.mark.asyncio
    async def test_notify_test_complete(self, mock_env_vars):
        """Test notify_test_complete broadcasts correct message."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="tester",
            agent_type="ui_tester",
        )

        mock_producer = AsyncMock()
        mock_consumer = AsyncMock()
        mock_consumer.getmany = AsyncMock(return_value={})

        with patch("src.orchestrator.a2a_protocol.AIOKafkaProducer", return_value=mock_producer):
            with patch("src.orchestrator.a2a_protocol.AIOKafkaConsumer", return_value=mock_consumer):
                await protocol.start()

                await protocol.notify_test_complete(
                    test_id="test-123",
                    status="passed",
                    result={"duration": 5.5},
                )

                mock_producer.send_and_wait.assert_called()
                call_args = mock_producer.send_and_wait.call_args
                assert call_args.kwargs["key"] == "test.completed"

                await protocol.stop()

    def test_get_circuit_status(self, mock_env_vars):
        """Test get_circuit_status returns all circuit breaker states."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(
            agent_id="test-agent",
            agent_type="test",
        )

        # Create some circuit breakers
        cb1 = protocol._get_circuit_breaker("agent-1")
        cb2 = protocol._get_circuit_breaker("agent-2")
        cb1.record_failure()
        cb1.record_failure()

        status = protocol.get_circuit_status()

        assert "agent-1" in status
        assert "agent-2" in status
        assert status["agent-1"]["failure_count"] == 2
        assert status["agent-2"]["failure_count"] == 0


class TestA2AProtocolKafkaTopics:
    """Tests for A2A Protocol Kafka topic constants."""

    def test_kafka_topic_constants(self, mock_env_vars):
        """Test Kafka topic constants are defined correctly."""
        from src.orchestrator.a2a_protocol import (
            TOPIC_AGENT_BROADCAST,
            TOPIC_AGENT_HEARTBEAT,
            TOPIC_AGENT_REQUEST,
            TOPIC_AGENT_RESPONSE,
        )

        assert TOPIC_AGENT_REQUEST == "argus.agent.request"
        assert TOPIC_AGENT_RESPONSE == "argus.agent.response"
        assert TOPIC_AGENT_BROADCAST == "argus.agent.broadcast"
        assert TOPIC_AGENT_HEARTBEAT == "argus.agent.heartbeat"


# =============================================================================
# Agent Registry Tests
# =============================================================================


class TestCapabilityEnum:
    """Tests for Capability enum."""

    def test_all_41_capabilities_defined(self, mock_env_vars):
        """Test all 41 capabilities mentioned in CLAUDE.md are defined."""
        from src.orchestrator.agent_registry import Capability

        # Get all capability values
        capabilities = [cap.value for cap in Capability]

        # Verify we have at least 41 capabilities (as mentioned in CLAUDE.md)
        assert len(capabilities) >= 41

        # Verify key capabilities exist
        expected_capabilities = [
            # Code Analysis
            "code_analysis",
            "git_blame",
            "dependency_analysis",
            "security_scan",
            "test_impact_analysis",
            "code_coverage",
            # UI Testing
            "browser_automation",
            "computer_use",
            "visual_testing",
            "accessibility_check",
            "screenshot_capture",
            "video_recording",
            # API Testing
            "api_testing",
            "schema_validation",
            "graphql_testing",
            "websocket_testing",
            # Self-Healing
            "selector_fix",
            "assertion_fix",
            "flaky_detection",
            "root_cause_analysis",
            "auto_healing",
            # Test Planning
            "test_planning",
            "test_prioritization",
            "risk_assessment",
            "test_generation",
            # Reporting
            "report_generation",
            "slack_integration",
            "github_integration",
            "jira_integration",
            # Database
            "db_testing",
            "data_validation",
            "migration_testing",
            # Performance
            "performance_analysis",
            "load_testing",
            "latency_monitoring",
            # Discovery
            "auto_discovery",
            "flow_discovery",
            "crawling",
            # AI
            "nlp_understanding",
            "visual_ai",
            "cognitive_modeling",
        ]

        for cap in expected_capabilities:
            assert cap in capabilities, f"Missing capability: {cap}"


class TestAgentInfo:
    """Tests for AgentInfo dataclass."""

    def test_agent_info_creation(self, mock_env_vars):
        """Test AgentInfo can be created with required fields."""
        from src.orchestrator.agent_registry import AgentInfo, Capability

        now = datetime.now(UTC)
        agent = AgentInfo(
            agent_id="test-agent-1",
            agent_type="code_analyzer",
            capabilities=[Capability.CODE_ANALYSIS, Capability.GIT_BLAME],
            status="healthy",
            last_heartbeat=now,
            metadata={"version": "1.0.0"},
        )

        assert agent.agent_id == "test-agent-1"
        assert agent.agent_type == "code_analyzer"
        assert len(agent.capabilities) == 2
        assert agent.status == "healthy"
        assert agent.metadata == {"version": "1.0.0"}

    def test_agent_info_to_dict(self, mock_env_vars):
        """Test AgentInfo serialization to dict."""
        from src.orchestrator.agent_registry import AgentInfo, Capability

        now = datetime.now(UTC)
        agent = AgentInfo(
            agent_id="test-agent",
            agent_type="tester",
            capabilities=[Capability.API_TESTING],
            status="healthy",
            last_heartbeat=now,
            registered_at=now,
        )

        data = agent.to_dict()

        assert data["agent_id"] == "test-agent"
        assert data["agent_type"] == "tester"
        assert data["capabilities"] == ["api_testing"]
        assert data["status"] == "healthy"

    def test_agent_info_from_dict(self, mock_env_vars):
        """Test AgentInfo deserialization from dict."""
        from src.orchestrator.agent_registry import AgentInfo, Capability

        now = datetime.now(UTC)
        data = {
            "agent_id": "test-agent",
            "agent_type": "tester",
            "capabilities": ["api_testing", "schema_validation"],
            "status": "healthy",
            "last_heartbeat": now.isoformat(),
            "registered_at": now.isoformat(),
            "metadata": {"key": "value"},
        }

        agent = AgentInfo.from_dict(data)

        assert agent.agent_id == "test-agent"
        assert agent.agent_type == "tester"
        assert Capability.API_TESTING in agent.capabilities
        assert Capability.SCHEMA_VALIDATION in agent.capabilities
        assert agent.metadata == {"key": "value"}


class TestAgentRegistryRegistration:
    """Tests for Agent Registry registration operations."""

    def test_register_agent_generates_id(self, mock_env_vars):
        """Test registering agent generates unique ID."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS],
        )

        assert agent_id is not None
        assert agent_id.startswith("test_agent_")

    def test_register_agent_with_custom_id(self, mock_env_vars):
        """Test registering agent with custom ID."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS],
            agent_id="custom-agent-id",
        )

        assert agent_id == "custom-agent-id"

    def test_register_agent_with_metadata(self, mock_env_vars):
        """Test registering agent with metadata."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS],
            metadata={"version": "1.0.0", "region": "us-east"},
        )

        agent = registry.get(agent_id)
        assert agent is not None
        assert agent.metadata["version"] == "1.0.0"
        assert agent.metadata["region"] == "us-east"

    def test_register_agent_with_string_capabilities(self, mock_env_vars):
        """Test registering agent with string capability names."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=["code_analysis", "git_blame"],
        )

        agent = registry.get(agent_id)
        assert Capability.CODE_ANALYSIS in agent.capabilities
        assert Capability.GIT_BLAME in agent.capabilities

    def test_register_agent_initial_status_healthy(self, mock_env_vars):
        """Test newly registered agent has healthy status."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS],
        )

        agent = registry.get(agent_id)
        assert agent.status == "healthy"

    def test_unregister_agent(self, mock_env_vars):
        """Test unregistering agent removes it from registry."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS],
        )

        result = registry.unregister(agent_id)

        assert result is True
        assert registry.get(agent_id) is None

    def test_unregister_nonexistent_agent(self, mock_env_vars):
        """Test unregistering nonexistent agent returns False."""
        from src.orchestrator.agent_registry import AgentRegistry, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        result = registry.unregister("nonexistent-agent")

        assert result is False


class TestAgentRegistryDiscovery:
    """Tests for Agent Registry capability discovery."""

    def test_discover_by_capability(self, mock_env_vars):
        """Test discovering agents by capability."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        # Register agents with different capabilities
        registry.register(
            agent_type="analyzer",
            capabilities=[Capability.CODE_ANALYSIS, Capability.GIT_BLAME],
            agent_id="agent-1",
        )
        registry.register(
            agent_type="tester",
            capabilities=[Capability.API_TESTING],
            agent_id="agent-2",
        )
        registry.register(
            agent_type="analyzer2",
            capabilities=[Capability.CODE_ANALYSIS],
            agent_id="agent-3",
        )

        # Discover by code analysis capability
        agents = registry.discover(Capability.CODE_ANALYSIS)

        assert len(agents) == 2
        agent_ids = [a.agent_id for a in agents]
        assert "agent-1" in agent_ids
        assert "agent-3" in agent_ids

    def test_discover_by_string_capability(self, mock_env_vars):
        """Test discovering agents by string capability name."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        registry.register(
            agent_type="tester",
            capabilities=[Capability.API_TESTING],
            agent_id="agent-1",
        )

        agents = registry.discover("api_testing")

        assert len(agents) == 1
        assert agents[0].agent_id == "agent-1"

    def test_discover_unknown_capability(self, mock_env_vars):
        """Test discovering with unknown capability returns empty list."""
        from src.orchestrator.agent_registry import AgentRegistry, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agents = registry.discover("nonexistent_capability")

        assert agents == []

    def test_discover_all_capabilities(self, mock_env_vars):
        """Test discovering agents with ALL specified capabilities."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        registry.register(
            agent_type="full",
            capabilities=[Capability.CODE_ANALYSIS, Capability.GIT_BLAME, Capability.SECURITY_SCAN],
            agent_id="agent-1",
        )
        registry.register(
            agent_type="partial",
            capabilities=[Capability.CODE_ANALYSIS],
            agent_id="agent-2",
        )

        # Only agent-1 has both capabilities
        agents = registry.discover_all([Capability.CODE_ANALYSIS, Capability.GIT_BLAME])

        assert len(agents) == 1
        assert agents[0].agent_id == "agent-1"

    def test_discover_any_capabilities(self, mock_env_vars):
        """Test discovering agents with ANY of specified capabilities."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        registry.register(
            agent_type="analyzer",
            capabilities=[Capability.CODE_ANALYSIS],
            agent_id="agent-1",
        )
        registry.register(
            agent_type="security",
            capabilities=[Capability.SECURITY_SCAN],
            agent_id="agent-2",
        )
        registry.register(
            agent_type="tester",
            capabilities=[Capability.API_TESTING],
            agent_id="agent-3",
        )

        # Both agent-1 and agent-2 have at least one matching capability
        agents = registry.discover_any([Capability.CODE_ANALYSIS, Capability.SECURITY_SCAN])

        assert len(agents) == 2
        agent_ids = [a.agent_id for a in agents]
        assert "agent-1" in agent_ids
        assert "agent-2" in agent_ids


class TestAgentRegistryHealth:
    """Tests for Agent Registry health monitoring."""

    def test_update_heartbeat(self, mock_env_vars):
        """Test updating agent heartbeat."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS],
        )

        old_heartbeat = registry.get(agent_id).last_heartbeat

        # Wait a tiny bit
        time.sleep(0.01)

        result = registry.update_heartbeat(agent_id)

        assert result is True
        new_heartbeat = registry.get(agent_id).last_heartbeat
        assert new_heartbeat > old_heartbeat

    def test_update_heartbeat_unknown_agent(self, mock_env_vars):
        """Test updating heartbeat for unknown agent returns False."""
        from src.orchestrator.agent_registry import AgentRegistry, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        result = registry.update_heartbeat("nonexistent-agent")

        assert result is False

    def test_update_heartbeat_recovers_unhealthy_agent(self, mock_env_vars):
        """Test heartbeat recovers unhealthy agent to healthy status."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS],
        )

        # Mark as unhealthy
        registry.update_status(agent_id, "unhealthy")
        assert registry.get(agent_id).status == "unhealthy"

        # Update heartbeat should recover
        registry.update_heartbeat(agent_id)
        assert registry.get(agent_id).status == "healthy"

    def test_check_health_marks_unhealthy(self, mock_env_vars):
        """Test check_health marks agents as unhealthy after timeout."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()
        # Set very short timeout for testing
        registry.HEARTBEAT_TIMEOUT_SECONDS = 0

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS],
        )

        # Manually set old heartbeat
        registry._agents[agent_id].last_heartbeat = datetime.now(UTC) - timedelta(seconds=120)

        result = registry.check_health()

        assert agent_id in result["marked_unhealthy"]
        assert registry.get(agent_id).status == "unhealthy"

    def test_check_health_removes_long_unhealthy(self, mock_env_vars):
        """Test check_health removes agents unhealthy for too long."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()
        # Set very short timeouts for testing
        registry.HEARTBEAT_TIMEOUT_SECONDS = 0
        registry.AUTO_REMOVE_UNHEALTHY_AFTER = 0

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS],
        )

        # Set very old heartbeat
        registry._agents[agent_id].last_heartbeat = datetime.now(UTC) - timedelta(seconds=600)

        result = registry.check_health()

        assert agent_id in result["removed"]
        assert registry.get(agent_id) is None

    def test_get_healthy_agents(self, mock_env_vars):
        """Test getting only healthy agents."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        registry.register(
            agent_type="healthy",
            capabilities=[Capability.CODE_ANALYSIS],
            agent_id="agent-1",
        )
        registry.register(
            agent_type="unhealthy",
            capabilities=[Capability.CODE_ANALYSIS],
            agent_id="agent-2",
        )
        registry.update_status("agent-2", "unhealthy")

        healthy = registry.get_healthy_agents()

        assert len(healthy) == 1
        assert healthy[0].agent_id == "agent-1"


class TestAgentRegistryStatus:
    """Tests for Agent Registry status and metadata operations."""

    def test_update_status(self, mock_env_vars):
        """Test updating agent status."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS],
        )

        result = registry.update_status(agent_id, "unhealthy")

        assert result is True
        assert registry.get(agent_id).status == "unhealthy"

    def test_update_status_unknown_agent(self, mock_env_vars):
        """Test updating status for unknown agent returns False."""
        from src.orchestrator.agent_registry import AgentRegistry, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        result = registry.update_status("nonexistent", "healthy")

        assert result is False

    def test_update_metadata_merge(self, mock_env_vars):
        """Test updating metadata with merge."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS],
            metadata={"key1": "value1"},
        )

        registry.update_metadata(agent_id, {"key2": "value2"}, merge=True)

        agent = registry.get(agent_id)
        assert agent.metadata["key1"] == "value1"
        assert agent.metadata["key2"] == "value2"

    def test_update_metadata_replace(self, mock_env_vars):
        """Test updating metadata with replace."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS],
            metadata={"key1": "value1"},
        )

        registry.update_metadata(agent_id, {"key2": "value2"}, merge=False)

        agent = registry.get(agent_id)
        assert "key1" not in agent.metadata
        assert agent.metadata["key2"] == "value2"


class TestAgentRegistryQueries:
    """Tests for Agent Registry query operations."""

    def test_get_all_agents(self, mock_env_vars):
        """Test getting all registered agents."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        registry.register(agent_type="agent1", capabilities=[Capability.CODE_ANALYSIS])
        registry.register(agent_type="agent2", capabilities=[Capability.API_TESTING])

        all_agents = registry.get_all_agents()

        assert len(all_agents) == 2

    def test_get_agents_by_type(self, mock_env_vars):
        """Test getting agents by type."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        registry.register(agent_type="analyzer", capabilities=[Capability.CODE_ANALYSIS])
        registry.register(agent_type="analyzer", capabilities=[Capability.GIT_BLAME])
        registry.register(agent_type="tester", capabilities=[Capability.API_TESTING])

        analyzers = registry.get_agents_by_type("analyzer")

        assert len(analyzers) == 2

    def test_get_registry_stats(self, mock_env_vars):
        """Test getting registry statistics."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        registry.register(
            agent_type="analyzer",
            capabilities=[Capability.CODE_ANALYSIS, Capability.GIT_BLAME],
        )
        registry.register(
            agent_type="tester",
            capabilities=[Capability.API_TESTING],
        )

        stats = registry.get_registry_stats()

        assert stats["total_agents"] == 2
        assert stats["by_type"]["analyzer"] == 1
        assert stats["by_type"]["tester"] == 1
        assert stats["capabilities"]["code_analysis"] == 1
        assert stats["capabilities"]["api_testing"] == 1


class TestAgentRegistryClear:
    """Tests for Agent Registry clear operations."""

    def test_clear_removes_all_agents(self, mock_env_vars):
        """Test clear removes all agents from registry."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        registry.register(agent_type="agent1", capabilities=[Capability.CODE_ANALYSIS])
        registry.register(agent_type="agent2", capabilities=[Capability.API_TESTING])

        count = registry.clear()

        assert count == 2
        assert len(registry.get_all_agents()) == 0


class TestAgentRegistryGlobalInstance:
    """Tests for Agent Registry global singleton functions."""

    def test_get_agent_registry_singleton(self, mock_env_vars):
        """Test get_agent_registry returns singleton."""
        from src.orchestrator.agent_registry import get_agent_registry, reset_agent_registry

        reset_agent_registry()

        registry1 = get_agent_registry()
        registry2 = get_agent_registry()

        assert registry1 is registry2

    def test_reset_agent_registry(self, mock_env_vars):
        """Test reset_agent_registry clears global instance."""
        from src.orchestrator.agent_registry import Capability, get_agent_registry, reset_agent_registry

        reset_agent_registry()
        registry1 = get_agent_registry()
        registry1.register(agent_type="test", capabilities=[Capability.CODE_ANALYSIS])

        reset_agent_registry()
        registry2 = get_agent_registry()

        assert len(registry2.get_all_agents()) == 0


class TestAgentRegistryConvenienceFunctions:
    """Tests for Agent Registry convenience functions."""

    def test_register_agent_convenience(self, mock_env_vars):
        """Test register_agent convenience function."""
        from src.orchestrator.agent_registry import (
            Capability,
            register_agent,
            reset_agent_registry,
            get_agent_registry,
        )

        reset_agent_registry()

        agent_id = register_agent(
            agent_type="test",
            capabilities=[Capability.CODE_ANALYSIS],
        )

        registry = get_agent_registry()
        assert registry.get(agent_id) is not None

    def test_unregister_agent_convenience(self, mock_env_vars):
        """Test unregister_agent convenience function."""
        from src.orchestrator.agent_registry import (
            Capability,
            register_agent,
            reset_agent_registry,
            unregister_agent,
            get_agent_registry,
        )

        reset_agent_registry()
        agent_id = register_agent(agent_type="test", capabilities=[Capability.CODE_ANALYSIS])

        result = unregister_agent(agent_id)

        assert result is True
        registry = get_agent_registry()
        assert registry.get(agent_id) is None

    def test_discover_agents_convenience(self, mock_env_vars):
        """Test discover_agents convenience function."""
        from src.orchestrator.agent_registry import (
            Capability,
            discover_agents,
            register_agent,
            reset_agent_registry,
        )

        reset_agent_registry()
        register_agent(agent_type="test", capabilities=[Capability.CODE_ANALYSIS])

        agents = discover_agents(Capability.CODE_ANALYSIS)

        assert len(agents) == 1

    def test_heartbeat_convenience(self, mock_env_vars):
        """Test heartbeat convenience function."""
        from src.orchestrator.agent_registry import (
            Capability,
            heartbeat,
            register_agent,
            reset_agent_registry,
        )

        reset_agent_registry()
        agent_id = register_agent(agent_type="test", capabilities=[Capability.CODE_ANALYSIS])

        result = heartbeat(agent_id)

        assert result is True


class TestDefaultAgentCapabilities:
    """Tests for default agent capability mappings."""

    def test_get_default_capabilities(self, mock_env_vars):
        """Test getting default capabilities for agent types."""
        from src.orchestrator.agent_registry import Capability, get_default_capabilities

        # Test known agent type
        caps = get_default_capabilities("code_analyzer")
        assert Capability.CODE_ANALYSIS in caps
        assert Capability.GIT_BLAME in caps

        # Test unknown agent type
        unknown_caps = get_default_capabilities("unknown_agent_type")
        assert unknown_caps == []

    def test_default_capabilities_for_all_defined_types(self, mock_env_vars):
        """Test all defined agent types have default capabilities."""
        from src.orchestrator.agent_registry import DEFAULT_AGENT_CAPABILITIES

        expected_types = [
            "code_analyzer",
            "ui_tester",
            "api_tester",
            "self_healer",
            "test_planner",
            "reporter",
            "db_tester",
            "performance_analyzer",
            "security_scanner",
            "accessibility_checker",
            "auto_discovery",
            "visual_ai",
            "nlp_test_creator",
        ]

        for agent_type in expected_types:
            assert agent_type in DEFAULT_AGENT_CAPABILITIES
            assert len(DEFAULT_AGENT_CAPABILITIES[agent_type]) > 0


class TestAgentRegistryHealthMonitor:
    """Tests for Agent Registry health monitor background task."""

    @pytest.mark.asyncio
    async def test_start_health_monitor(self, mock_env_vars):
        """Test starting health monitor creates task."""
        from src.orchestrator.agent_registry import AgentRegistry, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        # Mock Supabase to avoid DB calls
        with patch.object(registry, "_get_supabase", return_value=None):
            await registry.start_health_monitor()

            assert registry._cleanup_task is not None
            assert not registry._cleanup_task.done()

            await registry.stop_health_monitor()

    @pytest.mark.asyncio
    async def test_stop_health_monitor(self, mock_env_vars):
        """Test stopping health monitor cancels task."""
        from src.orchestrator.agent_registry import AgentRegistry, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        with patch.object(registry, "_get_supabase", return_value=None):
            await registry.start_health_monitor()
            await registry.stop_health_monitor()

            assert registry._cleanup_task is None or registry._cleanup_task.done()


class TestAgentRegistryAsync:
    """Tests for Agent Registry async operations."""

    @pytest.mark.asyncio
    async def test_register_async(self, mock_env_vars):
        """Test async agent registration."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        # Mock Supabase
        with patch.object(registry, "_persist_agent", AsyncMock(return_value=True)):
            agent_id = await registry.register_async(
                agent_type="test",
                capabilities=[Capability.CODE_ANALYSIS],
            )

            assert agent_id is not None
            assert registry.get(agent_id) is not None

    @pytest.mark.asyncio
    async def test_unregister_async(self, mock_env_vars):
        """Test async agent unregistration."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(agent_type="test", capabilities=[Capability.CODE_ANALYSIS])

        with patch.object(registry, "_delete_agent_from_db", AsyncMock(return_value=True)):
            result = await registry.unregister_async(agent_id)

            assert result is True
            assert registry.get(agent_id) is None

    @pytest.mark.asyncio
    async def test_update_heartbeat_async(self, mock_env_vars):
        """Test async heartbeat update."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        agent_id = registry.register(agent_type="test", capabilities=[Capability.CODE_ANALYSIS])

        with patch.object(registry, "_update_heartbeat_in_db", AsyncMock(return_value=True)):
            result = await registry.update_heartbeat_async(agent_id)

            assert result is True

    @pytest.mark.asyncio
    async def test_clear_async(self, mock_env_vars):
        """Test async clear with database deletion."""
        from src.orchestrator.agent_registry import AgentRegistry, Capability, reset_agent_registry

        reset_agent_registry()
        registry = AgentRegistry()

        registry.register(agent_type="test1", capabilities=[Capability.CODE_ANALYSIS])
        registry.register(agent_type="test2", capabilities=[Capability.API_TESTING])

        # Mock Supabase
        mock_supabase = AsyncMock()
        mock_supabase.request = AsyncMock(return_value={})
        mock_supabase.is_configured = True

        with patch.object(registry, "_get_supabase", return_value=mock_supabase):
            count = await registry.clear_async(clear_database=True)

            assert count == 2
            assert len(registry.get_all_agents()) == 0


class TestAgentRegistryInit:
    """Tests for Agent Registry initialization functions."""

    @pytest.mark.asyncio
    async def test_init_agent_registry(self, mock_env_vars):
        """Test init_agent_registry function."""
        from src.orchestrator.agent_registry import init_agent_registry, reset_agent_registry

        reset_agent_registry()

        # Mock Supabase and health monitor
        with patch("src.orchestrator.agent_registry.get_agent_registry") as mock_get:
            mock_registry = MagicMock()
            mock_registry.load_from_database = AsyncMock(return_value=0)
            mock_registry.start_health_monitor = AsyncMock()
            mock_registry.get_all_agents = MagicMock(return_value=[])
            mock_get.return_value = mock_registry

            registry = await init_agent_registry(
                start_health_monitor=True,
                load_from_db=True,
            )

            mock_registry.load_from_database.assert_called_once()
            mock_registry.start_health_monitor.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_agent_registry(self, mock_env_vars):
        """Test shutdown_agent_registry function."""
        from src.orchestrator.agent_registry import (
            get_agent_registry,
            reset_agent_registry,
            shutdown_agent_registry,
        )

        reset_agent_registry()
        registry = get_agent_registry()

        with patch.object(registry, "stop_health_monitor", AsyncMock()):
            await shutdown_agent_registry()

            registry.stop_health_monitor.assert_called_once()
