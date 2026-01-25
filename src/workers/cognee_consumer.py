"""
Cognee Consumer Worker - Knowledge Graph Builder

Consumes codebase ingestion events from Redpanda and processes them
through Cognee's ECL (Extract-Cognify-Load) pipeline to build
knowledge graphs in FalkorDB.

Features:
- Async Kafka consumer with auto-commit
- Cognee integration for codebase analysis
- FalkorDB graph storage
- Dead letter queue for failed processing
- BYOK support: loads org's API keys from dashboard settings
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class CodebaseEvent(BaseModel):
    """Event model for codebase ingestion."""
    event_id: str
    codebase_id: str
    repo_url: str
    branch: str
    commit_sha: str
    files_count: int
    org_id: str
    project_id: str | None = None
    user_id: str | None = None
    correlation_id: str | None = None


class CogneeConfig(BaseModel):
    """Configuration for Cognee worker."""
    # Kafka/Redpanda
    bootstrap_servers: str = Field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    )
    consumer_group: str = Field(
        default_factory=lambda: os.getenv("KAFKA_CONSUMER_GROUP", "argus-cognee-workers")
    )
    auto_offset_reset: str = Field(
        default_factory=lambda: os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest")
    )

    # Topics
    input_topic: str = "argus.codebase.ingested"
    output_topic: str = "argus.codebase.analyzed"
    dlq_topic: str = "argus.dlq"

    # FalkorDB
    falkordb_host: str = Field(
        default_factory=lambda: os.getenv("FALKORDB_HOST", "localhost")
    )
    falkordb_port: int = Field(
        default_factory=lambda: int(os.getenv("FALKORDB_PORT", "6379"))
    )
    falkordb_password: str | None = Field(
        default_factory=lambda: os.getenv("FALKORDB_PASSWORD")
    )

    # Supabase (for BYOK key lookup)
    supabase_url: str = Field(
        default_factory=lambda: os.getenv("SUPABASE_URL", "")
    )
    supabase_service_key: str = Field(
        default_factory=lambda: os.getenv("SUPABASE_SERVICE_KEY", "")
    )

    # Cloudflare Key Vault (for decryption)
    cloudflare_worker_url: str = Field(
        default_factory=lambda: os.getenv(
            "CLOUDFLARE_WORKER_URL",
            "https://argus-api.heyargus.workers.dev"
        )
    )

    # Worker settings
    batch_size: int = Field(
        default_factory=lambda: int(os.getenv("WORKER_BATCH_SIZE", "10"))
    )
    concurrency: int = Field(
        default_factory=lambda: int(os.getenv("WORKER_CONCURRENCY", "4"))
    )


class CogneeConsumer:
    """
    Kafka consumer that processes codebase events through Cognee.

    The consumer:
    1. Listens to argus.codebase.ingested topic
    2. Fetches org's BYOK API key from dashboard settings
    3. Processes each codebase through Cognee's ECL pipeline
    4. Stores knowledge graphs in FalkorDB
    5. Publishes completion events to argus.codebase.analyzed
    """

    def __init__(self, config: CogneeConfig | None = None):
        self.config = config or CogneeConfig()
        self._consumer = None
        self._producer = None
        self._running = False
        self._cognee_initialized = False
        self._http_client: httpx.AsyncClient | None = None
        # Cache org API keys to avoid repeated lookups (TTL: 5 min in prod)
        self._api_key_cache: dict[str, tuple[str, float]] = {}

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for API calls."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _get_org_owner_user_id(self, org_id: str) -> str | None:
        """
        Get the owner's user_id for an organization.

        Queries organization_members table for role='owner'.
        """
        if not self.config.supabase_url or not self.config.supabase_service_key:
            logger.warning("Supabase not configured, cannot fetch org owner")
            return None

        client = await self._get_http_client()

        try:
            response = await client.get(
                f"{self.config.supabase_url}/rest/v1/organization_members",
                params={
                    "organization_id": f"eq.{org_id}",
                    "role": "eq.owner",
                    "select": "user_id",
                },
                headers={
                    "apikey": self.config.supabase_service_key,
                    "Authorization": f"Bearer {self.config.supabase_service_key}",
                },
            )
            response.raise_for_status()

            data = response.json()
            if data and len(data) > 0:
                return data[0]["user_id"]

            logger.warning("No owner found for org", org_id=org_id)
            return None

        except Exception as e:
            logger.error("Failed to fetch org owner", org_id=org_id, error=str(e))
            return None

    async def _get_encrypted_key_for_user(
        self, user_id: str, provider: str = "anthropic"
    ) -> tuple[str, str] | None:
        """
        Get encrypted API key and DEK reference for a user.

        Returns (encrypted_key, dek_reference) or None if not found.
        """
        if not self.config.supabase_url or not self.config.supabase_service_key:
            return None

        client = await self._get_http_client()

        try:
            response = await client.get(
                f"{self.config.supabase_url}/rest/v1/user_provider_keys",
                params={
                    "user_id": f"eq.{user_id}",
                    "provider": f"eq.{provider}",
                    "is_valid": "eq.true",
                    "select": "encrypted_key,dek_reference",
                },
                headers={
                    "apikey": self.config.supabase_service_key,
                    "Authorization": f"Bearer {self.config.supabase_service_key}",
                },
            )
            response.raise_for_status()

            data = response.json()
            if data and len(data) > 0:
                key_data = data[0]
                if key_data.get("encrypted_key") and key_data.get("dek_reference"):
                    return (key_data["encrypted_key"], key_data["dek_reference"])

            return None

        except Exception as e:
            logger.error(
                "Failed to fetch encrypted key",
                user_id=user_id,
                provider=provider,
                error=str(e),
            )
            return None

    async def _decrypt_api_key(
        self, encrypted_key: str, dek_reference: str
    ) -> str | None:
        """
        Decrypt an API key using Cloudflare Key Vault.
        """
        client = await self._get_http_client()

        try:
            response = await client.post(
                f"{self.config.cloudflare_worker_url}/api/key-vault/decrypt",
                json={
                    "encrypted_key": encrypted_key,
                    "dek_reference": dek_reference,
                },
            )
            response.raise_for_status()

            data = response.json()
            return data.get("api_key")

        except Exception as e:
            logger.error("Failed to decrypt API key", error=str(e))
            return None

    async def _get_api_key_for_org(
        self, org_id: str, provider: str = "anthropic"
    ) -> str | None:
        """
        Get the decrypted API key for an organization.

        Flow:
        1. Find org owner's user_id
        2. Get their encrypted API key from user_provider_keys
        3. Decrypt via Cloudflare Key Vault
        """
        # Check cache first (5 minute TTL)
        import time
        cache_key = f"{org_id}:{provider}"
        if cache_key in self._api_key_cache:
            api_key, cached_at = self._api_key_cache[cache_key]
            if time.time() - cached_at < 300:  # 5 min TTL
                return api_key

        # Get org owner
        user_id = await self._get_org_owner_user_id(org_id)
        if not user_id:
            logger.warning("Cannot get API key: no org owner found", org_id=org_id)
            return None

        # Get encrypted key
        key_data = await self._get_encrypted_key_for_user(user_id, provider)
        if not key_data:
            logger.warning(
                "No BYOK key configured for org owner",
                org_id=org_id,
                user_id=user_id,
                provider=provider,
            )
            return None

        encrypted_key, dek_reference = key_data

        # Decrypt key
        api_key = await self._decrypt_api_key(encrypted_key, dek_reference)
        if api_key:
            # Cache the result
            self._api_key_cache[cache_key] = (api_key, time.time())
            logger.info(
                "Retrieved BYOK API key for org",
                org_id=org_id,
                provider=provider,
            )

        return api_key

    async def _configure_llm_for_org(self, org_id: str) -> bool:
        """
        Configure Cognee's LLM with the org's BYOK API key.

        Returns True if configured successfully, False otherwise.
        """
        import cognee

        # Try to get org's Anthropic BYOK key
        api_key = await self._get_api_key_for_org(org_id, "anthropic")

        if api_key:
            # Configure Cognee to use Anthropic directly
            cognee.config.set_llm_provider("anthropic")
            cognee.config.set_llm_api_key(api_key)
            cognee.config.set_llm_model(os.getenv("LLM_MODEL", "claude-sonnet-4-5"))
            logger.info("Configured Cognee with org BYOK key", org_id=org_id)
            return True

        # Fallback to OpenRouter if no BYOK key
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            cognee.config.set_llm_provider("openrouter")
            cognee.config.set_llm_config({
                "api_key": openrouter_key,
                "model": os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4"),
            })
            logger.info("Configured Cognee with OpenRouter fallback", org_id=org_id)
            return True

        logger.error(
            "No LLM API key available for org",
            org_id=org_id,
            hint="Configure BYOK in dashboard or set OPENROUTER_API_KEY",
        )
        return False

    async def _init_cognee(self) -> None:
        """Initialize Cognee with FalkorDB backend (LLM configured per-event)."""
        if self._cognee_initialized:
            return

        try:
            import cognee

            # Set writable data directories (required when running as non-root)
            data_dir = os.getenv("COGNEE_DATA_ROOT", "/app/data")
            system_dir = os.getenv("COGNEE_SYSTEM_ROOT", f"{data_dir}/.cognee_system")
            cognee.config.data_root_directory(data_dir)
            cognee.config.system_root_directory(system_dir)

            # Configure Cognee to use FalkorDB for graph storage
            cognee.config.set_graph_database_provider("falkordb")
            cognee.config.set_graph_db_config({
                "graph_database_url": f"redis://{self.config.falkordb_host}:{self.config.falkordb_port}",
                "graph_database_password": self.config.falkordb_password or "",
            })

            # Configure vector storage to use pgvector (Supabase)
            if os.getenv("DATABASE_URL"):
                cognee.config.set_vector_db_provider("pgvector")
                cognee.config.set_vector_db_url(os.getenv("DATABASE_URL"))

            # Note: LLM is configured per-event in _configure_llm_for_org()
            # to support BYOK (Bring Your Own Key) per organization

            self._cognee_initialized = True
            logger.info("Cognee initialized successfully (LLM configured per-event)")

        except ImportError:
            logger.error("Cognee not installed. Install with: pip install cognee[postgres,falkordb]")
            raise
        except Exception as e:
            logger.error("Failed to initialize Cognee", error=str(e))
            raise

    async def start(self) -> None:
        """Start the consumer and producer connections."""
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        except ImportError:
            logger.error("aiokafka not installed. Install with: pip install aiokafka")
            raise

        # Initialize Cognee
        await self._init_cognee()

        # Create consumer
        self._consumer = AIOKafkaConsumer(
            self.config.input_topic,
            bootstrap_servers=self.config.bootstrap_servers,
            group_id=self.config.consumer_group,
            auto_offset_reset=self.config.auto_offset_reset,
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        # Create producer for output events
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        await self._consumer.start()
        await self._producer.start()

        self._running = True
        logger.info(
            "Cognee consumer started",
            input_topic=self.config.input_topic,
            consumer_group=self.config.consumer_group,
        )

    async def stop(self) -> None:
        """Stop the consumer and producer connections."""
        self._running = False

        if self._consumer:
            await self._consumer.stop()
        if self._producer:
            await self._producer.stop()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        logger.info("Cognee consumer stopped")

    async def process_event(self, event_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process a codebase event through Cognee.

        Args:
            event_data: The raw event data from Kafka

        Returns:
            Analysis results including identified test surfaces
        """
        import cognee

        event = CodebaseEvent(**event_data.get("data", event_data))

        logger.info(
            "Processing codebase",
            codebase_id=event.codebase_id,
            repo_url=event.repo_url,
            files_count=event.files_count,
            org_id=event.org_id,
        )

        # Configure LLM for this org (BYOK support)
        llm_configured = await self._configure_llm_for_org(event.org_id)
        if not llm_configured:
            raise RuntimeError(
                f"No LLM API key available for org {event.org_id}. "
                "Configure BYOK in dashboard settings or set OPENROUTER_API_KEY."
            )

        # Use org-specific dataset for multi-tenancy
        dataset_name = f"org_{event.org_id}"
        if event.project_id:
            dataset_name = f"{dataset_name}_project_{event.project_id}"

        # Run Cognee's ECL pipeline
        # 1. Add: Ingest data for processing
        # 2. Cognify: Build knowledge graph with relationships
        # 3. Search: Query the knowledge graph

        # For now, add codebase metadata as text content
        # In production, this would clone the repo and process actual files
        codebase_description = f"""
        Codebase Analysis Request:
        - Repository: {event.repo_url}
        - Branch: {event.branch}
        - Commit: {event.commit_sha}
        - Files Count: {event.files_count}
        - Codebase ID: {event.codebase_id}
        """

        # Add codebase info to Cognee's memory
        await cognee.add(
            codebase_description,
            dataset_name=dataset_name,
        )

        # Process through cognify pipeline (builds knowledge graph)
        await cognee.cognify()

        # Search for testable surfaces in the knowledge graph
        test_surfaces = await cognee.search(
            "testable components, API endpoints, UI interactions, database operations",
        )

        # Build analysis result
        analysis_result = {
            "codebase_id": event.codebase_id,
            "org_id": event.org_id,
            "project_id": event.project_id,
            "dataset_name": dataset_name,
            "test_surfaces": test_surfaces if test_surfaces else [],
            "analysis_complete": True,
        }

        logger.info(
            "Codebase analysis complete",
            codebase_id=event.codebase_id,
            test_surfaces_count=len(analysis_result.get("test_surfaces", [])),
        )

        return analysis_result

    async def _send_to_dlq(self, event_data: dict, error: str) -> None:
        """Send failed event to dead letter queue."""
        if not self._producer:
            return

        dlq_message = {
            "original_event": event_data,
            "error": error,
            "consumer_group": self.config.consumer_group,
        }

        await self._producer.send_and_wait(
            self.config.dlq_topic,
            value=dlq_message,
        )
        logger.warning("Event sent to DLQ", error=error[:100])

    async def run(self) -> None:
        """Main consumer loop."""
        if not self._consumer:
            await self.start()

        logger.info("Starting consumer loop")

        try:
            async for message in self._consumer:
                if not self._running:
                    break

                try:
                    # Process the event
                    result = await self.process_event(message.value)

                    # Publish completion event
                    await self._producer.send_and_wait(
                        self.config.output_topic,
                        value=result,
                    )

                except Exception as e:
                    logger.error(
                        "Failed to process event",
                        error=str(e),
                        topic=message.topic,
                        partition=message.partition,
                        offset=message.offset,
                    )
                    await self._send_to_dlq(message.value, str(e))

        except asyncio.CancelledError:
            logger.info("Consumer loop cancelled")
        finally:
            await self.stop()


async def main():
    """Main entry point for the Cognee consumer worker."""
    config = CogneeConfig()
    consumer = CogneeConsumer(config)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(consumer.stop())
        )

    logger.info("Starting Cognee consumer worker")
    await consumer.run()


if __name__ == "__main__":
    asyncio.run(main())
