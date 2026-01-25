"""
Cognee Kafka Worker - Event-driven knowledge graph builder.

Consumes events from Redpanda/Kafka topics and builds knowledge graphs
using Cognee's ECL (Extract → Cognify → Load) pipeline with FalkorDB storage.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError
import cognee
from aiohttp import web

from config import WorkerConfig, load_config


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
    - Store knowledge graphs in FalkorDB
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
        """Initialize Cognee with FalkorDB and LLM configuration.

        Note: Cognee uses environment variables for configuration.
        These are set via the K8s ConfigMap and Secrets:
        - LLM_PROVIDER, LLM_MODEL, LLM_API_KEY (for Anthropic)
        - EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_API_KEY (for OpenAI)
        - GRAPH_DATABASE_PROVIDER, etc.

        See: https://docs.cognee.ai/setup-configuration/llm-providers
        """
        logger.info("Configuring Cognee...")

        # Set environment variables for Cognee's Pydantic settings
        # These override any defaults in the Cognee library
        import os

        # LLM Configuration (Anthropic)
        os.environ.setdefault("LLM_PROVIDER", self.config.cognee.llm_provider)
        os.environ.setdefault("LLM_MODEL", self.config.cognee.llm_model)
        if self.config.cognee.llm_api_key:
            os.environ.setdefault("LLM_API_KEY", self.config.cognee.llm_api_key)

        # Embedding Configuration (OpenAI)
        os.environ.setdefault("EMBEDDING_PROVIDER", self.config.cognee.embedding_provider)
        os.environ.setdefault("EMBEDDING_MODEL", self.config.cognee.embedding_model)
        if self.config.cognee.embedding_api_key:
            os.environ.setdefault("EMBEDDING_API_KEY", self.config.cognee.embedding_api_key)

        # Graph Database Configuration (FalkorDB)
        os.environ.setdefault("GRAPH_DATABASE_PROVIDER", "falkordb")
        falkor_url = (
            f"redis://:{self.config.falkordb.password}@"
            f"{self.config.falkordb.host}:{self.config.falkordb.port}"
        )
        os.environ.setdefault("GRAPH_DATABASE_URL", falkor_url)
        os.environ.setdefault("GRAPH_DATABASE_NAME", self.config.falkordb.graph_name)

        logger.info(f"Cognee LLM provider: {os.environ.get('LLM_PROVIDER')}")
        logger.info(f"Cognee embedding provider: {os.environ.get('EMBEDDING_PROVIDER')}")
        logger.info(f"Cognee graph database: {os.environ.get('GRAPH_DATABASE_PROVIDER')}")
        logger.info("Cognee configured via environment variables")

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
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            request_timeout_ms=60000,
            metadata_max_age_ms=300000,
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
        """Send failed message to Dead Letter Queue."""
        dlq_event = {
            "original_topic": original_topic,
            "original_key": key,
            "original_value": value,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "worker": "cognee-worker",
        }
        await self._produce_event(self.config.output_topic_dlq, key, dlq_event)
        logger.warning(f"Sent to DLQ: {key} - {error}")

    async def _process_codebase_ingested(self, key: str, event: dict[str, Any]):
        """
        Process a codebase.ingested event.

        Extracts knowledge from the codebase and builds a knowledge graph.
        """
        logger.info(f"Processing codebase.ingested: {key}")

        project_id = event.get("project_id")
        repo_url = event.get("repo_url")
        content = event.get("content", {})

        if not project_id or not content:
            raise ValueError("Missing project_id or content in event")

        # Add content to Cognee for processing
        # Content can be file paths, code snippets, or structured data
        files = content.get("files", [])
        for file_data in files:
            file_path = file_data.get("path", "unknown")
            file_content = file_data.get("content", "")

            if file_content:
                # Add to Cognee's knowledge base
                # Include metadata in the content itself since Cognee 0.5.x
                # doesn't support the metadata parameter
                enriched_content = (
                    f"# File: {file_path}\n"
                    f"# Project: {project_id}\n"
                    f"# Repo: {repo_url}\n\n"
                    f"{file_content}"
                )
                await cognee.add(
                    enriched_content,
                    dataset_name=f"project_{project_id}",
                )

        # Run Cognee's cognify pipeline to extract knowledge
        logger.info(f"Running Cognee cognify for project {project_id}...")
        await cognee.cognify(dataset_name=f"project_{project_id}")

        # Produce analyzed event
        analyzed_event = {
            "project_id": project_id,
            "repo_url": repo_url,
            "status": "analyzed",
            "graph_name": self.config.falkordb.graph_name,
            "file_count": len(files),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._produce_event(
            self.config.output_topic_analyzed,
            key,
            analyzed_event,
        )
        logger.info(f"Completed analysis for project {project_id}")

    async def _process_test_event(self, topic: str, key: str, event: dict[str, Any]):
        """
        Process test-related events (created, executed, failed).

        Adds test execution data to the knowledge graph for pattern learning.
        """
        logger.info(f"Processing {topic}: {key}")

        test_id = event.get("test_id")
        project_id = event.get("project_id")
        test_type = topic.split(".")[-1]  # created, executed, or failed

        if not test_id or not project_id:
            raise ValueError("Missing test_id or project_id in event")

        # Build test execution knowledge
        test_knowledge = {
            "test_id": test_id,
            "project_id": project_id,
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
        # Cognee 0.5.x doesn't support metadata parameter, so embed it in content
        await cognee.add(
            json.dumps(test_knowledge),
            dataset_name=f"tests_{project_id}",
        )

        # For failed tests, also analyze failure patterns
        if test_type == "failed":
            await self._analyze_failure_pattern(project_id, test_id, event)

        logger.info(f"Processed test event {test_type} for test {test_id}")

    async def _analyze_failure_pattern(
        self,
        project_id: str,
        test_id: str,
        event: dict[str, Any],
    ):
        """Analyze test failure patterns using Cognee search."""
        error_message = event.get("error_message", "")

        if not error_message:
            return

        # Search for similar failures in the knowledge graph
        try:
            similar_failures = await cognee.search(
                query=error_message,
                dataset_name=f"tests_{project_id}",
                top_k=5,
            )

            if similar_failures:
                logger.info(
                    f"Found {len(similar_failures)} similar failures for test {test_id}"
                )
                # Could emit a healing.suggested event here with patterns
        except Exception as e:
            logger.warning(f"Error searching for similar failures: {e}")

    async def _process_healing_requested(self, key: str, event: dict[str, Any]):
        """
        Process a healing.requested event.

        Uses knowledge graph to suggest fixes for broken tests.
        """
        logger.info(f"Processing healing.requested: {key}")

        test_id = event.get("test_id")
        project_id = event.get("project_id")
        failure_reason = event.get("failure_reason", "")

        if not test_id or not project_id:
            raise ValueError("Missing test_id or project_id in event")

        # Search knowledge graph for relevant context
        context = await cognee.search(
            query=failure_reason,
            dataset_name=f"project_{project_id}",
            top_k=10,
        )

        # Search for similar past failures and their resolutions
        similar_failures = await cognee.search(
            query=failure_reason,
            dataset_name=f"tests_{project_id}",
            top_k=5,
        )

        # Build healing response
        healing_event = {
            "test_id": test_id,
            "project_id": project_id,
            "status": "healing_analyzed",
            "relevant_context": [str(c) for c in context] if context else [],
            "similar_failures": [str(f) for f in similar_failures] if similar_failures else [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await self._produce_event(
            self.config.output_topic_healing_completed,
            key,
            healing_event,
        )
        logger.info(f"Completed healing analysis for test {test_id}")

    async def _handle_message(self, msg):
        """Route message to appropriate handler based on topic."""
        topic = msg.topic
        key = msg.key
        value = msg.value

        logger.debug(f"Received message from {topic}: {key}")

        try:
            if topic == "argus.codebase.ingested":
                await self._process_codebase_ingested(key, value)
            elif topic.startswith("argus.test."):
                await self._process_test_event(topic, key, value)
            elif topic == "argus.healing.requested":
                await self._process_healing_requested(key, value)
            else:
                logger.warning(f"Unknown topic: {topic}")

        except Exception as e:
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
    """HTTP server for liveness and readiness probes."""

    def __init__(self, worker: CogneeKafkaWorker, port: int):
        self.worker = worker
        self.port = port
        self.app = web.Application()
        self.app.router.add_get("/health", self.health_check)
        self.app.router.add_get("/ready", self.readiness_check)
        self.runner: Optional[web.AppRunner] = None

    async def health_check(self, request: web.Request) -> web.Response:
        """Liveness probe - is the process running?"""
        return web.json_response({"status": "healthy"})

    async def readiness_check(self, request: web.Request) -> web.Response:
        """Readiness probe - can we accept traffic?"""
        if self.worker.running and self.worker.consumer:
            return web.json_response({"status": "ready"})
        return web.json_response({"status": "not_ready"}, status=503)

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
