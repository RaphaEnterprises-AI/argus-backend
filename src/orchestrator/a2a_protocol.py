"""
Agent-to-Agent (A2A) Communication Protocol over Kafka

Provides a robust communication layer for agents to:
- Send request/response messages to specific agents
- Broadcast messages to all subscribers
- Handle timeouts and circuit breaker patterns for fault tolerance
- Track request correlation for distributed tracing

Usage:
    protocol = A2AProtocol(agent_id="ui-tester-1", agent_type="ui_tester")
    await protocol.start()

    # Request another agent
    response = await protocol.request(
        to_agent="self-healer-1",
        capability="heal_selector",
        payload={"test_id": "test-123", "failed_selector": "#login-btn"}
    )

    # Broadcast to all
    await protocol.broadcast("test.completed", {"test_id": "test-123", "status": "passed"})

    await protocol.stop()
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Kafka Topics for A2A Communication
# =============================================================================

TOPIC_AGENT_REQUEST = "argus.agent.request"
TOPIC_AGENT_RESPONSE = "argus.agent.response"
TOPIC_AGENT_BROADCAST = "argus.agent.broadcast"
TOPIC_AGENT_HEARTBEAT = "argus.agent.heartbeat"


# =============================================================================
# Message Schemas
# =============================================================================

class MessageType(str, Enum):
    """Types of A2A messages."""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"


class AgentRequestEvent(BaseModel):
    """Request message sent to another agent."""

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    message_type: MessageType = Field(default=MessageType.REQUEST)
    from_agent: str = Field(..., description="Sending agent ID")
    from_agent_type: str = Field(..., description="Type of sending agent")
    to_agent: str = Field(..., description="Target agent ID")
    capability: str = Field(..., description="Capability being requested")
    payload: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_ms: int = Field(default=30000, description="Request timeout in milliseconds")
    correlation_id: Optional[str] = Field(None, description="For tracing request chains")
    priority: int = Field(default=5, description="Priority 1-10 (1=highest)")

    def to_kafka_key(self) -> str:
        """Generate Kafka message key for routing."""
        return self.to_agent

    def to_dict(self) -> dict:
        """Convert to dict for Kafka serialization."""
        return self.model_dump(mode="json")


class AgentResponse(BaseModel):
    """Response to an agent request."""

    response_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str = Field(..., description="ID of the original request")
    message_type: MessageType = Field(default=MessageType.RESPONSE)
    from_agent: str = Field(..., description="Responding agent ID")
    from_agent_type: str = Field(..., description="Type of responding agent")
    to_agent: str = Field(..., description="Original requester agent ID")
    success: bool = Field(..., description="Whether the request succeeded")
    payload: dict = Field(default_factory=dict)
    error: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Error code for categorization")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: int = Field(default=0, description="Time taken to process request")

    def to_kafka_key(self) -> str:
        """Generate Kafka message key for routing."""
        return self.to_agent

    def to_dict(self) -> dict:
        """Convert to dict for Kafka serialization."""
        return self.model_dump(mode="json")


class AgentBroadcast(BaseModel):
    """Broadcast message to all subscribers."""

    broadcast_id: str = Field(default_factory=lambda: str(uuid4()))
    message_type: MessageType = Field(default=MessageType.BROADCAST)
    from_agent: str = Field(..., description="Broadcasting agent ID")
    from_agent_type: str = Field(..., description="Type of broadcasting agent")
    topic: str = Field(..., description="Broadcast topic/channel")
    payload: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_kafka_key(self) -> str:
        """Generate Kafka message key for topic-based partitioning."""
        return self.topic

    def to_dict(self) -> dict:
        """Convert to dict for Kafka serialization."""
        return self.model_dump(mode="json")


class AgentHeartbeat(BaseModel):
    """Heartbeat message for agent health monitoring."""

    agent_id: str = Field(..., description="Agent ID")
    agent_type: str = Field(..., description="Agent type")
    status: str = Field(default="healthy", description="Agent status")
    capabilities: list[str] = Field(default_factory=list)
    load: float = Field(default=0.0, description="Current load 0.0-1.0")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Convert to dict for Kafka serialization."""
        return self.model_dump(mode="json")


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance.

    Tracks failures to a specific agent and opens the circuit
    (rejecting requests) when failure threshold is exceeded.
    """

    agent_id: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds
    half_open_max_requests: int = 3

    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: Optional[float] = field(default=None)
    half_open_requests: int = field(default=0)

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_requests:
                self._close()
        else:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self._open()
        elif self.failure_count >= self.failure_threshold:
            self._open()

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time and \
               time.time() - self.last_failure_time >= self.recovery_timeout:
                self._half_open()
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_requests < self.half_open_max_requests:
                self.half_open_requests += 1
                return True
            return False

        return False

    def _open(self) -> None:
        """Open the circuit."""
        logger.warning(
            f"Circuit breaker opened for agent {self.agent_id}",
            extra={"agent_id": self.agent_id, "failure_count": self.failure_count}
        )
        self.state = CircuitState.OPEN

    def _close(self) -> None:
        """Close the circuit (back to normal)."""
        logger.info(
            f"Circuit breaker closed for agent {self.agent_id}",
            extra={"agent_id": self.agent_id}
        )
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_requests = 0

    def _half_open(self) -> None:
        """Set circuit to half-open state for testing."""
        logger.info(
            f"Circuit breaker half-open for agent {self.agent_id}",
            extra={"agent_id": self.agent_id}
        )
        self.state = CircuitState.HALF_OPEN
        self.half_open_requests = 0
        self.success_count = 0


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected."""

    def __init__(self, agent_id: str, retry_after: float):
        self.agent_id = agent_id
        self.retry_after = retry_after
        super().__init__(f"Circuit open for agent {agent_id}. Retry after {retry_after:.1f}s")


# =============================================================================
# A2A Protocol Implementation
# =============================================================================

class A2AProtocol:
    """Agent-to-Agent communication protocol over Kafka.

    Provides request/response messaging, broadcasting, and fault tolerance
    for inter-agent communication in the Argus distributed system.

    Features:
    - Request/response with correlation and timeout
    - Broadcast to all subscribers
    - Circuit breaker pattern for fault tolerance
    - Heartbeat monitoring
    - Graceful shutdown

    Example:
        async with A2AProtocol.create(
            agent_id="ui-tester-1",
            agent_type="ui_tester"
        ) as protocol:
            response = await protocol.request(
                to_agent="self-healer-1",
                capability="heal_selector",
                payload={"selector": "#login-btn"}
            )
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        bootstrap_servers: str = "localhost:9092",
        sasl_username: Optional[str] = None,
        sasl_password: Optional[str] = None,
        client_id: Optional[str] = None,
        capabilities: Optional[list[str]] = None,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 30.0,
    ):
        """Initialize A2A Protocol.

        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (e.g., "ui_tester", "self_healer")
            bootstrap_servers: Kafka broker addresses
            sasl_username: SASL username for authentication
            sasl_password: SASL password for authentication
            client_id: Kafka client ID (defaults to agent_id)
            capabilities: List of capabilities this agent provides
            circuit_breaker_threshold: Failures before circuit opens
            circuit_breaker_timeout: Seconds before circuit resets
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id or f"a2a-{agent_id}"
        self.capabilities = capabilities or []

        # SASL configuration
        self._sasl_username = sasl_username
        self._sasl_password = sasl_password

        # Kafka clients
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumer: Optional[AIOKafkaConsumer] = None

        # Request tracking
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._request_lock = asyncio.Lock()

        # Subscription handlers
        self._subscriptions: dict[str, list[Callable]] = {}
        self._request_handler: Optional[Callable] = None

        # Circuit breakers per agent
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._cb_threshold = circuit_breaker_threshold
        self._cb_timeout = circuit_breaker_timeout

        # State
        self._started = False
        self._consumer_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        self._log = logger.getChild(f"a2a.{agent_id}")

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        agent_id: str,
        agent_type: str,
        **kwargs
    ):
        """Create protocol as async context manager.

        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent
            **kwargs: Additional arguments passed to __init__

        Yields:
            Started A2AProtocol instance
        """
        protocol = cls(agent_id=agent_id, agent_type=agent_type, **kwargs)
        await protocol.start()
        try:
            yield protocol
        finally:
            await protocol.stop()

    def _get_kafka_config(self) -> dict[str, Any]:
        """Build Kafka configuration dict."""
        config = {
            "bootstrap_servers": self.bootstrap_servers,
            "client_id": self.client_id,
        }

        if self._sasl_username and self._sasl_password:
            config.update({
                "security_protocol": "SASL_PLAINTEXT",
                "sasl_mechanism": "SCRAM-SHA-512",
                "sasl_plain_username": self._sasl_username,
                "sasl_plain_password": self._sasl_password,
            })

        return config

    async def start(self) -> None:
        """Start the protocol - connect to Kafka and begin consuming."""
        if self._started:
            return

        config = self._get_kafka_config()

        # Initialize producer
        self._producer = AIOKafkaProducer(
            **config,
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            compression_type="zstd",
        )
        await self._producer.start()

        # Initialize consumer for responses and broadcasts
        consumer_group = f"a2a-{self.agent_id}"
        self._consumer = AIOKafkaConsumer(
            TOPIC_AGENT_RESPONSE,
            TOPIC_AGENT_REQUEST,
            TOPIC_AGENT_BROADCAST,
            **config,
            group_id=consumer_group,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        await self._consumer.start()

        # Start consumer loop
        self._consumer_task = asyncio.create_task(self._consume_loop())

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self._started = True
        self._log.info(
            "A2A Protocol started",
            extra={"agent_id": self.agent_id, "agent_type": self.agent_type}
        )

    async def stop(self) -> None:
        """Stop the protocol - cleanup connections."""
        if not self._started:
            return

        self._started = False

        # Cancel consumer task
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        # Cancel heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Cancel pending requests
        async with self._request_lock:
            for future in self._pending_requests.values():
                if not future.done():
                    future.cancel()
            self._pending_requests.clear()

        # Stop Kafka clients
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None

        if self._producer:
            await self._producer.stop()
            self._producer = None

        self._log.info("A2A Protocol stopped")

    def _get_circuit_breaker(self, agent_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for an agent."""
        if agent_id not in self._circuit_breakers:
            self._circuit_breakers[agent_id] = CircuitBreaker(
                agent_id=agent_id,
                failure_threshold=self._cb_threshold,
                recovery_timeout=self._cb_timeout,
            )
        return self._circuit_breakers[agent_id]

    async def request(
        self,
        to_agent: str,
        capability: str,
        payload: dict,
        timeout: float = 30.0,
        correlation_id: Optional[str] = None,
        priority: int = 5,
    ) -> AgentResponse:
        """Send request to another agent and wait for response.

        Args:
            to_agent: Target agent ID
            capability: Capability being requested
            payload: Request payload
            timeout: Timeout in seconds
            correlation_id: For tracing request chains
            priority: Request priority (1=highest, 10=lowest)

        Returns:
            AgentResponse from the target agent

        Raises:
            asyncio.TimeoutError: If request times out
            CircuitOpenError: If circuit breaker is open
            RuntimeError: If protocol not started
        """
        if not self._started or not self._producer:
            raise RuntimeError("Protocol not started. Call start() first.")

        # Check circuit breaker
        cb = self._get_circuit_breaker(to_agent)
        if not cb.can_execute():
            retry_after = cb.recovery_timeout - (time.time() - (cb.last_failure_time or 0))
            raise CircuitOpenError(to_agent, retry_after)

        # Create request
        request = AgentRequestEvent(
            from_agent=self.agent_id,
            from_agent_type=self.agent_type,
            to_agent=to_agent,
            capability=capability,
            payload=payload,
            timeout_ms=int(timeout * 1000),
            correlation_id=correlation_id,
            priority=priority,
        )

        # Create future for response
        future: asyncio.Future[AgentResponse] = asyncio.Future()

        async with self._request_lock:
            self._pending_requests[request.request_id] = future

        try:
            # Send request
            await self._producer.send_and_wait(
                topic=TOPIC_AGENT_REQUEST,
                key=request.to_kafka_key(),
                value=request.to_dict(),
                headers=[
                    ("request_id", request.request_id.encode()),
                    ("from_agent", self.agent_id.encode()),
                    ("to_agent", to_agent.encode()),
                    ("capability", capability.encode()),
                ],
            )

            self._log.debug(
                "Request sent",
                extra={
                    "request_id": request.request_id,
                    "to_agent": to_agent,
                    "capability": capability,
                }
            )

            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=timeout)

            # Record success
            cb.record_success()

            return response

        except asyncio.TimeoutError:
            self._log.warning(
                "Request timeout",
                extra={
                    "request_id": request.request_id,
                    "to_agent": to_agent,
                    "timeout": timeout,
                }
            )
            cb.record_failure()
            raise

        except KafkaError as e:
            self._log.error(
                "Request failed",
                extra={
                    "request_id": request.request_id,
                    "to_agent": to_agent,
                    "error": str(e),
                }
            )
            cb.record_failure()
            raise

        finally:
            # Cleanup pending request
            async with self._request_lock:
                self._pending_requests.pop(request.request_id, None)

    async def broadcast(self, topic: str, message: dict) -> None:
        """Broadcast message to all subscribers.

        Args:
            topic: Broadcast topic/channel name
            message: Message payload

        Raises:
            RuntimeError: If protocol not started
        """
        if not self._started or not self._producer:
            raise RuntimeError("Protocol not started. Call start() first.")

        broadcast = AgentBroadcast(
            from_agent=self.agent_id,
            from_agent_type=self.agent_type,
            topic=topic,
            payload=message,
        )

        await self._producer.send_and_wait(
            topic=TOPIC_AGENT_BROADCAST,
            key=broadcast.to_kafka_key(),
            value=broadcast.to_dict(),
            headers=[
                ("broadcast_id", broadcast.broadcast_id.encode()),
                ("from_agent", self.agent_id.encode()),
                ("topic", topic.encode()),
            ],
        )

        self._log.debug(
            "Broadcast sent",
            extra={"broadcast_id": broadcast.broadcast_id, "topic": topic}
        )

    async def subscribe(self, topic: str, handler: Callable[[AgentBroadcast], Any]) -> None:
        """Subscribe to broadcast topic.

        Args:
            topic: Topic to subscribe to
            handler: Async function to handle messages
        """
        if topic not in self._subscriptions:
            self._subscriptions[topic] = []
        self._subscriptions[topic].append(handler)
        self._log.info(f"Subscribed to broadcast topic: {topic}")

    def set_request_handler(self, handler: Callable[[AgentRequestEvent], Any]) -> None:
        """Set handler for incoming requests.

        Args:
            handler: Async function to handle requests. Should return a dict payload
                    or raise an exception.
        """
        self._request_handler = handler
        self._log.info("Request handler set")

    async def respond(
        self,
        request_id: str,
        to_agent: str,
        success: bool,
        payload: dict,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
        duration_ms: int = 0,
    ) -> None:
        """Send response to a request.

        Args:
            request_id: ID of the original request
            to_agent: Agent that sent the request
            success: Whether request succeeded
            payload: Response payload
            error: Error message if failed
            error_code: Error code for categorization
            duration_ms: Processing time
        """
        if not self._started or not self._producer:
            raise RuntimeError("Protocol not started. Call start() first.")

        response = AgentResponse(
            request_id=request_id,
            from_agent=self.agent_id,
            from_agent_type=self.agent_type,
            to_agent=to_agent,
            success=success,
            payload=payload,
            error=error,
            error_code=error_code,
            duration_ms=duration_ms,
        )

        await self._producer.send_and_wait(
            topic=TOPIC_AGENT_RESPONSE,
            key=response.to_kafka_key(),
            value=response.to_dict(),
            headers=[
                ("response_id", response.response_id.encode()),
                ("request_id", request_id.encode()),
                ("from_agent", self.agent_id.encode()),
                ("to_agent", to_agent.encode()),
                ("success", str(success).encode()),
            ],
        )

        self._log.debug(
            "Response sent",
            extra={
                "request_id": request_id,
                "to_agent": to_agent,
                "success": success,
            }
        )

    async def _consume_loop(self) -> None:
        """Main consumer loop for handling messages."""
        while self._started and self._consumer:
            try:
                # Poll for messages
                msg_batch = await self._consumer.getmany(timeout_ms=100)

                for tp, messages in msg_batch.items():
                    for msg in messages:
                        await self._handle_message(msg.topic, msg.value)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Consumer error: {e}")
                await asyncio.sleep(1)

    async def _handle_message(self, topic: str, value: dict) -> None:
        """Handle incoming message based on topic."""
        try:
            if topic == TOPIC_AGENT_RESPONSE:
                await self._handle_response(value)
            elif topic == TOPIC_AGENT_REQUEST:
                await self._handle_request(value)
            elif topic == TOPIC_AGENT_BROADCAST:
                await self._handle_broadcast(value)
        except Exception as e:
            self._log.error(f"Error handling message: {e}", extra={"topic": topic})

    async def _handle_response(self, value: dict) -> None:
        """Handle response message."""
        # Check if this response is for us
        if value.get("to_agent") != self.agent_id:
            return

        request_id = value.get("request_id")
        if not request_id:
            return

        async with self._request_lock:
            future = self._pending_requests.get(request_id)
            if future and not future.done():
                response = AgentResponse(**value)
                future.set_result(response)
                self._log.debug(
                    "Response received",
                    extra={"request_id": request_id, "success": response.success}
                )

    async def _handle_request(self, value: dict) -> None:
        """Handle incoming request."""
        # Check if this request is for us
        if value.get("to_agent") != self.agent_id:
            return

        if not self._request_handler:
            self._log.warning("No request handler set, ignoring request")
            return

        request = AgentRequestEvent(**value)
        start_time = time.time()

        try:
            # Call handler
            result = await self._request_handler(request)
            duration_ms = int((time.time() - start_time) * 1000)

            await self.respond(
                request_id=request.request_id,
                to_agent=request.from_agent,
                success=True,
                payload=result if isinstance(result, dict) else {"result": result},
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._log.error(
                f"Request handler error: {e}",
                extra={"request_id": request.request_id}
            )

            await self.respond(
                request_id=request.request_id,
                to_agent=request.from_agent,
                success=False,
                payload={},
                error=str(e),
                error_code=type(e).__name__,
                duration_ms=duration_ms,
            )

    async def _handle_broadcast(self, value: dict) -> None:
        """Handle broadcast message."""
        broadcast = AgentBroadcast(**value)

        # Get handlers for this topic
        handlers = self._subscriptions.get(broadcast.topic, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(broadcast)
                else:
                    handler(broadcast)
            except Exception as e:
                self._log.error(
                    f"Broadcast handler error: {e}",
                    extra={"topic": broadcast.topic}
                )

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._started and self._producer:
            try:
                heartbeat = AgentHeartbeat(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    capabilities=self.capabilities,
                    status="healthy",
                )

                await self._producer.send_and_wait(
                    topic=TOPIC_AGENT_HEARTBEAT,
                    key=self.agent_id,
                    value=heartbeat.to_dict(),
                )

                await asyncio.sleep(30)  # Heartbeat every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def request_healing(
        self,
        test_id: str,
        failure_info: dict,
        timeout: float = 60.0,
    ) -> AgentResponse:
        """Request self-healing for a failed test.

        Args:
            test_id: ID of the failed test
            failure_info: Details about the failure
            timeout: Timeout in seconds

        Returns:
            AgentResponse with healing result
        """
        # Find a self-healer agent (in practice, use registry)
        # For now, use a well-known agent ID pattern
        return await self.request(
            to_agent="self-healer-primary",
            capability="heal_test",
            payload={
                "test_id": test_id,
                "failure_info": failure_info,
            },
            timeout=timeout,
        )

    async def notify_test_complete(
        self,
        test_id: str,
        status: str,
        result: dict,
    ) -> None:
        """Broadcast test completion notification.

        Args:
            test_id: ID of the completed test
            status: Test status (passed, failed, etc.)
            result: Test result details
        """
        await self.broadcast(
            topic="test.completed",
            message={
                "test_id": test_id,
                "status": status,
                "result": result,
            },
        )

    def get_circuit_status(self) -> dict[str, dict]:
        """Get status of all circuit breakers.

        Returns:
            Dict mapping agent ID to circuit state info
        """
        return {
            agent_id: {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time,
            }
            for agent_id, cb in self._circuit_breakers.items()
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_a2a_protocol_from_settings(
    agent_id: str,
    agent_type: str,
    capabilities: Optional[list[str]] = None,
) -> A2AProtocol:
    """Create A2A Protocol using application settings.

    Args:
        agent_id: Unique agent identifier
        agent_type: Type of agent
        capabilities: List of capabilities

    Returns:
        Configured A2AProtocol instance
    """
    from src.config import get_settings

    settings = get_settings()

    sasl_password = None
    if settings.redpanda_sasl_password:
        sasl_password = settings.redpanda_sasl_password.get_secret_value()

    return A2AProtocol(
        agent_id=agent_id,
        agent_type=agent_type,
        bootstrap_servers=settings.redpanda_brokers,
        sasl_username=settings.redpanda_sasl_username,
        sasl_password=sasl_password,
        capabilities=capabilities,
    )
