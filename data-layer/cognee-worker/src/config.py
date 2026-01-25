"""Configuration management for Cognee Kafka Worker."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KafkaConfig:
    """Kafka/Redpanda connection configuration."""

    bootstrap_servers: str = field(
        default_factory=lambda: os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            "redpanda.argus-data.svc.cluster.local:9092"
        )
    )
    security_protocol: str = field(
        default_factory=lambda: os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
    )
    sasl_mechanism: str = field(
        default_factory=lambda: os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512")
    )
    sasl_username: str = field(
        default_factory=lambda: os.getenv("KAFKA_SASL_USERNAME", "argus-service")
    )
    sasl_password: str = field(
        default_factory=lambda: os.getenv("KAFKA_SASL_PASSWORD", "")
    )
    consumer_group: str = field(
        default_factory=lambda: os.getenv("KAFKA_CONSUMER_GROUP", "cognee-worker")
    )
    auto_offset_reset: str = field(
        default_factory=lambda: os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest")
    )


@dataclass
class FalkorDBConfig:
    """FalkorDB graph database configuration."""

    host: str = field(
        default_factory=lambda: os.getenv(
            "FALKORDB_HOST",
            "falkordb-headless.argus-data.svc.cluster.local"
        )
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("FALKORDB_PORT", "6379"))
    )
    password: str = field(
        default_factory=lambda: os.getenv("FALKORDB_PASSWORD", "")
    )
    graph_name: str = field(
        default_factory=lambda: os.getenv("FALKORDB_GRAPH_NAME", "argus_knowledge")
    )


@dataclass
class ValkeyConfig:
    """Valkey cache configuration."""

    host: str = field(
        default_factory=lambda: os.getenv(
            "VALKEY_HOST",
            "valkey-headless.argus-data.svc.cluster.local"
        )
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("VALKEY_PORT", "6379"))
    )
    password: str = field(
        default_factory=lambda: os.getenv("VALKEY_PASSWORD", "")
    )
    db: int = field(
        default_factory=lambda: int(os.getenv("VALKEY_DB", "0"))
    )


@dataclass
class CogneeConfig:
    """Cognee processing configuration.

    Note: Cognee uses these specific environment variable names:
    - LLM_PROVIDER, LLM_MODEL, LLM_API_KEY
    - EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_API_KEY
    - GRAPH_DATABASE_PROVIDER, GRAPH_DATABASE_URL, GRAPH_DATABASE_NAME
    See: https://docs.cognee.ai/setup-configuration/llm-providers
    """

    # LLM provider configuration (Anthropic Claude)
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-4-5-20250514")
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )

    # Embedding configuration (Cohere - best for multilingual + technical content)
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "cohere")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "embed-multilingual-v3.0")
    )
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY", "")
    )

    # Processing settings
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("COGNEE_CHUNK_SIZE", "1024"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("COGNEE_CHUNK_OVERLAP", "128"))
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("COGNEE_BATCH_SIZE", "10"))
    )


@dataclass
class WorkerConfig:
    """Main worker configuration aggregating all sub-configs."""

    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    falkordb: FalkorDBConfig = field(default_factory=FalkorDBConfig)
    valkey: ValkeyConfig = field(default_factory=ValkeyConfig)
    cognee: CogneeConfig = field(default_factory=CogneeConfig)

    # Worker settings
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "10"))
    )
    retry_delay_seconds: int = field(
        default_factory=lambda: int(os.getenv("RETRY_DELAY_SECONDS", "10"))
    )
    health_check_port: int = field(
        default_factory=lambda: int(os.getenv("HEALTH_CHECK_PORT", "8080"))
    )

    # Topics to consume
    input_topics: list[str] = field(default_factory=lambda: [
        "argus.codebase.ingested",
        "argus.test.created",
        "argus.test.executed",
        "argus.test.failed",
        "argus.healing.requested",
    ])

    # Output topics
    output_topic_analyzed: str = "argus.codebase.analyzed"
    output_topic_healing_completed: str = "argus.healing.completed"
    output_topic_dlq: str = "argus.dlq"


def load_config() -> WorkerConfig:
    """Load configuration from environment variables."""
    return WorkerConfig()
