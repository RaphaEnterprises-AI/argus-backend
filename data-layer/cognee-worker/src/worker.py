"""
Cognee Kafka Worker - Event-driven knowledge graph builder.

Consumes events from Redpanda/Kafka topics and builds knowledge graphs
using Cognee's ECL (Extract → Cognify → Load) pipeline with Neo4j Aura storage.

Multi-tenancy: All data is isolated by org_id and project_id using the naming convention:
    org_{org_id}_project_{project_id}_{dataset_type}
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError
import cognee
from aiohttp import web
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from config import WorkerConfig, load_config

# =============================================================================
# Prometheus Metrics
# =============================================================================

# Event processing metrics
EVENTS_PROCESSED = Counter(
    "cognee_events_processed_total",
    "Total number of events processed by the Cognee worker",
    ["event_type", "status"],  # status: success, error, dlq
)

EVENTS_PROCESSING_DURATION = Histogram(
    "cognee_event_processing_duration_seconds",
    "Time spent processing each event",
    ["event_type"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
)

# Kafka consumer metrics
KAFKA_CONSUMER_RUNNING = Gauge(
    "cognee_kafka_consumer_running",
    "Whether the Kafka consumer is running (1=running, 0=stopped)",
)

KAFKA_MESSAGES_RECEIVED = Counter(
    "cognee_kafka_messages_received_total",
    "Total number of Kafka messages received",
    ["topic"],
)

KAFKA_CONSUMER_LAG = Gauge(
    "cognee_kafka_consumer_lag",
    "Estimated consumer lag (messages behind)",
    ["topic", "partition"],
)

# Neo4j connection metrics
NEO4J_CONNECTION_STATUS = Gauge(
    "cognee_neo4j_connection_status",
    "Neo4j connection status (1=connected, 0=disconnected)",
)

NEO4J_CONNECTION_ATTEMPTS = Counter(
    "cognee_neo4j_connection_attempts_total",
    "Total Neo4j connection attempts",
    ["status"],  # success, failed
)

# DLQ metrics
DLQ_MESSAGES = Counter(
    "cognee_dlq_messages_total",
    "Total messages sent to Dead Letter Queue",
    ["original_topic", "error_type"],
)

# Cognee pipeline metrics
COGNEE_COGNIFY_DURATION = Histogram(
    "cognee_cognify_duration_seconds",
    "Time spent in Cognee cognify pipeline",
    ["dataset_type"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0],
)

COGNEE_SEARCH_DURATION = Histogram(
    "cognee_search_duration_seconds",
    "Time spent in Cognee search operations",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# Worker info
WORKER_INFO = Gauge(
    "cognee_worker_info",
    "Worker information",
    ["version", "consumer_group"],
)
WORKER_INFO.labels(version="1.0.0", consumer_group="argus-cognee-workers").set(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cognee-worker")


class CogneeKafkaWorker:
    """
    Event-driven worker that processes codebase events through Cognee.

    Responsibilities:
    - Consume events from Kafka topics (codebase.ingested, test.*, healing.*)
    - Extract knowledge using Cognee's ECL pipeline
    - Store knowledge graphs using configured provider (kuzu file-based by default)
    - Produce output events (codebase.analyzed, healing.completed)
    - Handle failures gracefully with DLQ
    """

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.producer: Optional[AIOKafkaProducer] = None
        self.running = False
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging level from config."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        logging.getLogger("aiokafka").setLevel(logging.WARNING)

    async def _setup_cognee(self):
        """Initialize Cognee with Neo4j Aura and LLM configuration.

        Note: Cognee uses environment variables for configuration.
        These are set via the K8s ConfigMap and Secrets:
        - LLM_PROVIDER, LLM_MODEL, LLM_API_KEY (for Anthropic)
        - EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_API_KEY (for Cohere)
        - GRAPH_DATABASE_PROVIDER, GRAPH_DATABASE_URL, GRAPH_DATABASE_USERNAME, GRAPH_DATABASE_PASSWORD
        - DATA_ROOT_DIRECTORY, SYSTEM_ROOT_DIRECTORY, CACHE_ROOT_DIRECTORY

        Supported graph providers (Cognee 0.5.x): neo4j, kuzu, kuzu-remote, neptune, neptune_analytics

        See: https://docs.cognee.ai/setup-configuration/llm-providers
        """
        logger.info("Configuring Cognee with Neo4j Aura...")

        import os

        # LLM Configuration (Anthropic)
        os.environ.setdefault("LLM_PROVIDER", self.config.cognee.llm_provider)
        os.environ.setdefault("LLM_MODEL", self.config.cognee.llm_model)
        if self.config.cognee.llm_api_key:
            os.environ.setdefault("LLM_API_KEY", self.config.cognee.llm_api_key)

        # Embedding Configuration (Cohere)
        os.environ.setdefault("EMBEDDING_PROVIDER", self.config.cognee.embedding_provider)
        os.environ.setdefault("EMBEDDING_MODEL", self.config.cognee.embedding_model)
        if self.config.cognee.embedding_api_key:
            os.environ.setdefault("EMBEDDING_API_KEY", self.config.cognee.embedding_api_key)
        # Cohere embed-multilingual-v3.0 produces 1024-dimension vectors
        os.environ.setdefault("EMBEDDING_DIMENSIONS", "1024")

        # Neo4j Aura Configuration
        os.environ.setdefault("GRAPH_DATABASE_PROVIDER", "neo4j")
        if self.config.neo4j.uri:
            os.environ.setdefault("GRAPH_DATABASE_URL", self.config.neo4j.uri)
        if self.config.neo4j.username:
            os.environ.setdefault("GRAPH_DATABASE_USERNAME", self.config.neo4j.username)
        if self.config.neo4j.password:
            os.environ.setdefault("GRAPH_DATABASE_PASSWORD", self.config.neo4j.password)

        logger.info(f"Cognee LLM provider: {os.environ.get('LLM_PROVIDER')}")
        logger.info(f"Cognee embedding provider: {os.environ.get('EMBEDDING_PROVIDER')}")
        logger.info(f"Cognee graph database: {os.environ.get('GRAPH_DATABASE_PROVIDER')}")
        logger.info(f"Neo4j URI: {self.config.neo4j.uri[:30]}..." if self.config.neo4j.uri else "Neo4j URI not set")

        # Test Neo4j connection with retry for Aura cold starts
        await self._test_neo4j_connection()

        logger.info("Cognee configured with Neo4j Aura")

    async def _test_neo4j_connection(self):
        """Test Neo4j Aura connection with retry logic for cold starts.

        Neo4j Aura Free tier auto-pauses after 3 days of inactivity and
        can take 30-60 seconds to wake up on first connection.
        """
        from neo4j import AsyncGraphDatabase
        from neo4j.exceptions import ServiceUnavailable

        if not self.config.neo4j.uri:
            logger.warning("Neo4j URI not configured, skipping connection test")
            NEO4J_CONNECTION_STATUS.set(0)
            return

        driver = AsyncGraphDatabase.driver(
            self.config.neo4j.uri,
            auth=(self.config.neo4j.username, self.config.neo4j.password),
        )

        max_retries = self.config.neo4j.max_retry_attempts
        retry_delay = 15  # seconds between retries

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Testing Neo4j connection (attempt {attempt}/{max_retries})...")
                async with driver.session() as session:
                    result = await session.run("RETURN 1 AS test")
                    await result.single()
                logger.info("Neo4j Aura connection successful")
                NEO4J_CONNECTION_STATUS.set(1)
                NEO4J_CONNECTION_ATTEMPTS.labels(status="success").inc()
                await driver.close()
                return
            except ServiceUnavailable as e:
                NEO4J_CONNECTION_ATTEMPTS.labels(status="failed").inc()
                if attempt < max_retries:
                    logger.warning(
                        f"Neo4j unavailable (Aura may be waking up), "
                        f"retrying in {retry_delay}s... ({attempt}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    NEO4J_CONNECTION_STATUS.set(0)
                    await driver.close()
                    raise RuntimeError(
                        f"Failed to connect to Neo4j Aura after {max_retries} attempts: {e}"
                    )
            except Exception as e:
                NEO4J_CONNECTION_ATTEMPTS.labels(status="failed").inc()
                NEO4J_CONNECTION_STATUS.set(0)
                await driver.close()
                raise RuntimeError(f"Neo4j connection error: {e}")

    def _get_dataset_name(
        self,
        org_id: str,
        project_id: str,
        dataset_type: str,
    ) -> str:
        """Generate tenant-scoped dataset name.

        Args:
            org_id: Organization ID
            project_id: Project ID
            dataset_type: Type of dataset (codebase, tests, failures)

        Returns:
            Dataset name like 'org_abc123_project_xyz789_codebase'
        """
        return f"org_{org_id}_project_{project_id}_{dataset_type}"

    def _extract_tenant_context(self, event: dict[str, Any]) -> tuple[str, str]:
        """Extract org_id and project_id from event.

        Events should have a 'tenant' field with org_id and project_id,
        or fall back to top-level org_id/project_id fields.

        Args:
            event: Event payload

        Returns:
            Tuple of (org_id, project_id)

        Raises:
            ValueError: If org_id or project_id is missing
        """
        tenant = event.get("tenant", {})

        org_id = tenant.get("org_id") or event.get("org_id")
        project_id = tenant.get("project_id") or event.get("project_id")

        if not org_id:
            raise ValueError("Missing org_id in event (required for multi-tenant isolation)")
        if not project_id:
            raise ValueError("Missing project_id in event (required for multi-tenant isolation)")

        return org_id, project_id

    def _safe_json_deserialize(self, v: bytes) -> dict:
        """Safely deserialize JSON, returning error dict on failure."""
        try:
            return json.loads(v.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode JSON message: {e}")
            # Return a marker dict so we can skip this message gracefully
            return {"__deserialize_error__": True, "raw": v.decode("utf-8", errors="replace")}

    async def _create_consumer(self) -> AIOKafkaConsumer:
        """Create and configure Kafka consumer with SASL auth."""
        logger.info(f"Creating consumer for: {self.config.kafka.bootstrap_servers}")
        logger.info(f"Security: {self.config.kafka.security_protocol}, SASL: {self.config.kafka.sasl_mechanism}")
        logger.info(f"User: {self.config.kafka.sasl_username}")

        consumer = AIOKafkaConsumer(
            *self.config.input_topics,
            bootstrap_servers=self.config.kafka.bootstrap_servers,
            group_id=self.config.kafka.consumer_group,
            auto_offset_reset=self.config.kafka.auto_offset_reset,
            enable_auto_commit=False,  # Manual commit for at-least-once delivery
            security_protocol=self.config.kafka.security_protocol,
            sasl_mechanism=self.config.kafka.sasl_mechanism,
            sasl_plain_username=self.config.kafka.sasl_username,
            sasl_plain_password=self.config.kafka.sasl_password,
            value_deserializer=self._safe_json_deserialize,
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            request_timeout_ms=60000,
            metadata_max_age_ms=300000,
            # Cognee cognify operations can take a long time (LLM calls, graph building)
            # Increase poll interval to prevent consumer group rebalancing during processing
            max_poll_interval_ms=900000,  # 15 minutes
            session_timeout_ms=60000,  # 1 minute heartbeat timeout
        )
        return consumer

    async def _create_producer(self) -> AIOKafkaProducer:
        """Create and configure Kafka producer with SASL auth."""
        producer = AIOKafkaProducer(
            bootstrap_servers=self.config.kafka.bootstrap_servers,
            security_protocol=self.config.kafka.security_protocol,
            sasl_mechanism=self.config.kafka.sasl_mechanism,
            sasl_plain_username=self.config.kafka.sasl_username,
            sasl_plain_password=self.config.kafka.sasl_password,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            request_timeout_ms=60000,
        )
        return producer

    async def _produce_event(
        self,
        topic: str,
        key: str,
        value: dict[str, Any],
    ):
        """Produce an event to a Kafka topic."""
        if self.producer:
            await self.producer.send_and_wait(topic, value=value, key=key)
            logger.debug(f"Produced event to {topic}: {key}")

    async def _send_to_dlq(
        self,
        original_topic: str,
        key: str,
        value: dict[str, Any],
        error: str,
    ):
        """Send failed message to Dead Letter Queue with tenant context."""
        # Extract tenant info if available
        tenant = value.get("tenant", {})
        org_id = tenant.get("org_id") or value.get("org_id", "unknown")
        project_id = tenant.get("project_id") or value.get("project_id", "unknown")

        # Categorize error type for metrics
        error_type = "processing_error"
        if "deserialize" in error.lower() or "json" in error.lower():
            error_type = "deserialization_error"
        elif "neo4j" in error.lower() or "graph" in error.lower():
            error_type = "graph_database_error"
        elif "kafka" in error.lower() or "connection" in error.lower():
            error_type = "connection_error"
        elif "tenant" in error.lower() or "org_id" in error.lower():
            error_type = "validation_error"

        # Track DLQ metrics
        DLQ_MESSAGES.labels(original_topic=original_topic, error_type=error_type).inc()

        dlq_event = {
            "event_type": "dlq",
            "event_version": "1.0",
            "tenant": {
                "org_id": org_id,
                "project_id": project_id,
            },
            "original_topic": original_topic,
            "original_key": key,
            "original_payload": value,
            "error_message": error,
            "error_type": error_type,
            "retry_count": 0,
            "max_retries": self.config.max_retries,
            "first_failed_at": datetime.now(timezone.utc).isoformat(),
            "worker": "cognee-worker",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._produce_event(self.config.output_topic_dlq, key, dlq_event)
        logger.warning(f"Sent to DLQ: {key} - {error} (org={org_id})")

    async def _process_codebase_ingested(self, key: str, event: dict[str, Any]):
        """
        Process a codebase.ingested event.

        Extracts knowledge from the codebase and builds a knowledge graph
        with multi-tenant dataset isolation.
        """
        logger.info(f"Processing codebase.ingested: {key}")

        # Extract tenant context
        org_id, project_id = self._extract_tenant_context(event)

        repo_url = event.get("repository_url") or event.get("repo_url", "")
        content = event.get("content", {})
        commit_sha = event.get("commit_sha", "")

        if not content:
            raise ValueError("Missing content in codebase.ingested event")

        # Generate tenant-scoped dataset name
        dataset_name = self._get_dataset_name(org_id, project_id, "codebase")

        logger.info(f"Processing codebase for tenant org={org_id}, project={project_id}")
        logger.info(f"Using dataset: {dataset_name}")

        # Add content to Cognee for processing
        files = content.get("files", [])
        for file_data in files:
            file_path = file_data.get("path", "unknown")
            file_content = file_data.get("content", "")

            if file_content:
                # Include tenant metadata in content (Cognee 0.5.x doesn't support metadata param)
                enriched_content = (
                    f"# Organization: {org_id}\n"
                    f"# Project: {project_id}\n"
                    f"# File: {file_path}\n"
                    f"# Repo: {repo_url}\n"
                    f"# Commit: {commit_sha}\n\n"
                    f"{file_content}"
                )
                await cognee.add(
                    enriched_content,
                    dataset_name=dataset_name,
                )

        # Run Cognee's cognify pipeline to extract knowledge
        logger.info(f"Running Cognee cognify for dataset {dataset_name}...")
        cognify_start = time.time()
        await cognee.cognify(dataset_name=dataset_name)
        cognify_duration = time.time() - cognify_start
        COGNEE_COGNIFY_DURATION.labels(dataset_type="codebase").observe(cognify_duration)
        logger.info(f"Cognify completed in {cognify_duration:.2f}s")

        # Produce analyzed event with tenant context
        analyzed_event = {
            "event_type": "codebase.analyzed",
            "event_version": "1.0",
            "tenant": {
                "org_id": org_id,
                "project_id": project_id,
            },
            "repository_id": event.get("repository_id", ""),
            "repository_url": repo_url,
            "commit_sha": commit_sha,
            "status": "analyzed",
            "cognee_dataset_name": dataset_name,
            "file_count": len(files),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._produce_event(
            self.config.output_topic_analyzed,
            key,
            analyzed_event,
        )
        logger.info(f"Completed analysis for org={org_id}, project={project_id}")

    async def _process_test_event(self, topic: str, key: str, event: dict[str, Any]):
        """
        Process test-related events (created, executed, failed).

        Adds test execution data to the knowledge graph for pattern learning
        with multi-tenant dataset isolation.
        """
        logger.info(f"Processing {topic}: {key}")

        # Extract tenant context
        org_id, project_id = self._extract_tenant_context(event)

        test_id = event.get("test_id")
        test_type = topic.split(".")[-1]  # created, executed, or failed

        if not test_id:
            raise ValueError("Missing test_id in test event")

        # Generate tenant-scoped dataset name
        dataset_name = self._get_dataset_name(org_id, project_id, "tests")

        logger.info(f"Processing test event for tenant org={org_id}, project={project_id}")

        # Build test execution knowledge with tenant context
        test_knowledge = {
            "org_id": org_id,
            "project_id": project_id,
            "test_id": test_id,
            "event_type": test_type,
            "test_name": event.get("test_name", ""),
            "test_status": event.get("status", test_type),
            "duration_ms": event.get("duration_ms"),
            "error_message": event.get("error_message"),
            "stack_trace": event.get("stack_trace"),
            "screenshot_url": event.get("screenshot_url"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add to Cognee for pattern learning
        await cognee.add(
            json.dumps(test_knowledge),
            dataset_name=dataset_name,
        )

        # For failed tests, also analyze failure patterns
        if test_type == "failed":
            await self._analyze_failure_pattern(org_id, project_id, test_id, event)

        logger.info(f"Processed test event {test_type} for test {test_id} (org={org_id})")

    async def _analyze_failure_pattern(
        self,
        org_id: str,
        project_id: str,
        test_id: str,
        event: dict[str, Any],
    ):
        """Analyze test failure patterns using Cognee search with tenant isolation."""
        error_message = event.get("error_message", "")

        if not error_message:
            return

        # Use tenant-scoped dataset for failure pattern search
        tests_dataset = self._get_dataset_name(org_id, project_id, "tests")
        failures_dataset = self._get_dataset_name(org_id, project_id, "failures")

        # Search for similar failures in the knowledge graph
        try:
            search_start = time.time()
            similar_failures = await cognee.search(
                query=error_message,
                dataset_name=tests_dataset,
                top_k=5,
            )
            COGNEE_SEARCH_DURATION.observe(time.time() - search_start)

            if similar_failures:
                logger.info(
                    f"Found {len(similar_failures)} similar failures for test {test_id} "
                    f"(org={org_id}, project={project_id})"
                )

                # Store failure pattern for learning
                failure_pattern = {
                    "org_id": org_id,
                    "project_id": project_id,
                    "test_id": test_id,
                    "error_message": error_message,
                    "error_type": event.get("error_type", "unknown"),
                    "failed_selector": event.get("failed_selector"),
                    "similar_failure_count": len(similar_failures),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await cognee.add(
                    json.dumps(failure_pattern),
                    dataset_name=failures_dataset,
                )

        except Exception as e:
            logger.warning(f"Error analyzing failure pattern: {e}")

    async def _process_healing_requested(self, key: str, event: dict[str, Any]):
        """
        Process a healing.requested event.

        Uses knowledge graph to suggest fixes for broken tests
        with multi-tenant dataset isolation.
        """
        logger.info(f"Processing healing.requested: {key}")

        # Extract tenant context
        org_id, project_id = self._extract_tenant_context(event)

        test_id = event.get("test_id")
        failure_id = event.get("failure_id", "")
        error_type = event.get("error_type", "")
        failed_selector = event.get("failed_selector", "")

        if not test_id:
            raise ValueError("Missing test_id in healing.requested event")

        # Build search query from failure context
        failure_reason = event.get("error_message", "") or failed_selector or error_type

        # Generate tenant-scoped dataset names
        codebase_dataset = self._get_dataset_name(org_id, project_id, "codebase")
        tests_dataset = self._get_dataset_name(org_id, project_id, "tests")
        failures_dataset = self._get_dataset_name(org_id, project_id, "failures")

        logger.info(f"Searching for healing context (org={org_id}, project={project_id})")

        # Search knowledge graph for relevant code context
        code_context = []
        try:
            search_start = time.time()
            code_context = await cognee.search(
                query=failure_reason,
                dataset_name=codebase_dataset,
                top_k=10,
            )
            COGNEE_SEARCH_DURATION.observe(time.time() - search_start)
        except Exception as e:
            logger.warning(f"Error searching codebase: {e}")

        # Search for similar past failures and their resolutions
        similar_failures = []
        try:
            search_start = time.time()
            similar_failures = await cognee.search(
                query=failure_reason,
                dataset_name=failures_dataset,
                top_k=5,
            )
            COGNEE_SEARCH_DURATION.observe(time.time() - search_start)
        except Exception as e:
            logger.warning(f"Error searching failures: {e}")

        # Build healing response with tenant context
        healing_event = {
            "event_type": "healing.completed",
            "event_version": "1.0",
            "tenant": {
                "org_id": org_id,
                "project_id": project_id,
            },
            "test_id": test_id,
            "failure_id": failure_id,
            "healing_request_id": event.get("event_id", ""),
            "status": "healing_analyzed",
            "success": True,
            "strategy_used": "cognee_search",
            "relevant_code_context": [str(c) for c in code_context] if code_context else [],
            "similar_failures": [str(f) for f in similar_failures] if similar_failures else [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await self._produce_event(
            self.config.output_topic_healing_completed,
            key,
            healing_event,
        )
        logger.info(
            f"Completed healing analysis for test {test_id} "
            f"(org={org_id}, project={project_id})"
        )

    async def _handle_message(self, msg):
        """Route message to appropriate handler based on topic."""
        topic = msg.topic
        key = msg.key
        value = msg.value

        logger.debug(f"Received message from {topic}: {key}")

        # Track received messages
        KAFKA_MESSAGES_RECEIVED.labels(topic=topic).inc()

        # Determine event type for metrics
        if topic == "argus.codebase.ingested":
            event_type = "codebase_ingested"
        elif topic.startswith("argus.test."):
            event_type = f"test_{topic.split('.')[-1]}"
        elif topic == "argus.healing.requested":
            event_type = "healing_requested"
        else:
            event_type = "unknown"

        # Check for deserialization errors
        if isinstance(value, dict) and value.get("__deserialize_error__"):
            logger.warning(f"Skipping malformed message {key}: JSON decode failed")
            EVENTS_PROCESSED.labels(event_type=event_type, status="error").inc()
            await self._send_to_dlq(topic, key, {"raw": value.get("raw", "")[:500]}, "JSON deserialization failed")
            return

        # Track processing time
        start_time = time.time()
        try:
            if topic == "argus.codebase.ingested":
                await self._process_codebase_ingested(key, value)
            elif topic.startswith("argus.test."):
                await self._process_test_event(topic, key, value)
            elif topic == "argus.healing.requested":
                await self._process_healing_requested(key, value)
            else:
                logger.warning(f"Unknown topic: {topic}")

            # Track successful processing
            duration = time.time() - start_time
            EVENTS_PROCESSING_DURATION.labels(event_type=event_type).observe(duration)
            EVENTS_PROCESSED.labels(event_type=event_type, status="success").inc()

        except Exception as e:
            # Track failed processing
            duration = time.time() - start_time
            EVENTS_PROCESSING_DURATION.labels(event_type=event_type).observe(duration)
            EVENTS_PROCESSED.labels(event_type=event_type, status="error").inc()

            logger.error(f"Error processing message {key}: {e}", exc_info=True)
            await self._send_to_dlq(topic, key, value, str(e))

    async def start(self):
        """Start the worker and begin consuming messages."""
        logger.info("Starting Cognee Kafka Worker...")

        # Setup Cognee
        await self._setup_cognee()

        # Create consumer and producer
        self.consumer = await self._create_consumer()
        self.producer = await self._create_producer()

        # Connect with retries
        retries = 0
        while retries < self.config.max_retries:
            try:
                await self.consumer.start()
                await self.producer.start()
                logger.info("Connected to Kafka/Redpanda")
                break
            except KafkaConnectionError as e:
                retries += 1
                logger.warning(
                    f"Kafka connection failed (attempt {retries}/{self.config.max_retries}): {e}"
                )
                if retries < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                else:
                    raise

        self.running = True
        logger.info(f"Consuming from topics: {self.config.input_topics}")

        # Main consumption loop
        try:
            async for msg in self.consumer:
                if not self.running:
                    break

                await self._handle_message(msg)

                # Commit offset after successful processing
                await self.consumer.commit()

        except asyncio.CancelledError:
            logger.info("Worker cancelled")
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            raise
        finally:
            await self.stop()

    async def stop(self):
        """Gracefully stop the worker."""
        logger.info("Stopping Cognee Kafka Worker...")
        self.running = False

        if self.consumer:
            await self.consumer.stop()
            logger.info("Consumer stopped")

        if self.producer:
            await self.producer.stop()
            logger.info("Producer stopped")

        logger.info("Worker stopped")


class HealthServer:
    """HTTP server for liveness, readiness probes, and Prometheus metrics."""

    def __init__(self, worker: CogneeKafkaWorker, port: int):
        self.worker = worker
        self.port = port
        self.app = web.Application()
        self.app.router.add_get("/health", self.health_check)
        self.app.router.add_get("/ready", self.readiness_check)
        self.app.router.add_get("/metrics", self.prometheus_metrics)
        self.runner: Optional[web.AppRunner] = None

    async def health_check(self, request: web.Request) -> web.Response:
        """Liveness probe - is the process running?"""
        return web.json_response({"status": "healthy"})

    async def readiness_check(self, request: web.Request) -> web.Response:
        """Readiness probe - can we accept traffic?"""
        if self.worker.running and self.worker.consumer:
            KAFKA_CONSUMER_RUNNING.set(1)
            return web.json_response({"status": "ready"})
        KAFKA_CONSUMER_RUNNING.set(0)
        return web.json_response({"status": "not_ready"}, status=503)

    async def prometheus_metrics(self, request: web.Request) -> web.Response:
        """Prometheus metrics endpoint."""
        # Update consumer running status
        KAFKA_CONSUMER_RUNNING.set(1 if (self.worker.running and self.worker.consumer) else 0)

        # Generate metrics in Prometheus text format
        metrics_output = generate_latest()
        # Note: aiohttp requires charset to be passed separately, not in content_type
        return web.Response(
            body=metrics_output,
            content_type="text/plain",
            charset="utf-8",
        )

    async def start(self):
        """Start the health check server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await site.start()
        logger.info(f"Health server listening on port {self.port}")

    async def stop(self):
        """Stop the health check server."""
        if self.runner:
            await self.runner.cleanup()


async def main():
    """Main entry point."""
    config = load_config()

    worker = CogneeKafkaWorker(config)
    health_server = HealthServer(worker, config.health_check_port)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def shutdown_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(worker.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)

    # Start health server and worker
    await health_server.start()

    try:
        await worker.start()
    finally:
        await health_server.stop()


if __name__ == "__main__":
    asyncio.run(main())
