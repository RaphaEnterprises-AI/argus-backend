#!/usr/bin/env python3
"""Validate Confluent Cloud topics are created correctly.

Usage:
    # Set environment variables first
    export CONFLUENT_BOOTSTRAP_SERVERS="pkc-xxx.region.aws.confluent.cloud:9092"
    export CONFLUENT_SASL_USERNAME="your-api-key"
    export CONFLUENT_SASL_PASSWORD="your-api-secret"

    # Then run
    python scripts/validate_confluent.py
"""

import os
import sys

from confluent_kafka.admin import AdminClient


def get_config():
    """Get Confluent configuration from environment variables."""
    bootstrap_servers = os.environ.get("CONFLUENT_BOOTSTRAP_SERVERS")
    sasl_username = os.environ.get("CONFLUENT_SASL_USERNAME")
    sasl_password = os.environ.get("CONFLUENT_SASL_PASSWORD")

    if not all([bootstrap_servers, sasl_username, sasl_password]):
        print("❌ Missing required environment variables:")
        print("   - CONFLUENT_BOOTSTRAP_SERVERS")
        print("   - CONFLUENT_SASL_USERNAME")
        print("   - CONFLUENT_SASL_PASSWORD")
        print("\nExample:")
        print('   export CONFLUENT_BOOTSTRAP_SERVERS="pkc-xxx.region.aws.confluent.cloud:9092"')
        print('   export CONFLUENT_SASL_USERNAME="your-api-key"')
        print('   export CONFLUENT_SASL_PASSWORD="your-api-secret"')
        sys.exit(1)

    return {
        "bootstrap.servers": bootstrap_servers,
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "PLAIN",
        "sasl.username": sasl_username,
        "sasl.password": sasl_password,
    }


# Expected topics
EXPECTED_TOPICS = [
    "argus.test.executed",
    "argus.codebase.ingested",
    "argus.error.reported",
    "argus.dlq",
]


def main():
    config = get_config()

    print("Connecting to Confluent Cloud...")
    admin = AdminClient(config)

    # Get cluster metadata
    metadata = admin.list_topics(timeout=30)

    print(f"\n✅ Connected to Confluent Cloud")
    print(f"   Cluster ID: {metadata.cluster_id}")
    print(f"   Brokers: {len(metadata.brokers)}")

    # List all topics
    topics = list(metadata.topics.keys())
    argus_topics = [t for t in topics if t.startswith("argus.")]

    print(f"\n📋 Found {len(argus_topics)} Argus topics:")
    for topic in sorted(argus_topics):
        topic_metadata = metadata.topics[topic]
        partitions = len(topic_metadata.partitions)
        print(f"   ✅ {topic} ({partitions} partitions)")

    # Check for missing topics
    missing = [t for t in EXPECTED_TOPICS if t not in topics]
    if missing:
        print("\n⚠️  Missing topics:")
        for t in missing:
            print(f"   ❌ {t}")
        return False

    print(f"\n🎉 All {len(EXPECTED_TOPICS)} expected topics are present!")
    return True


if __name__ == "__main__":
    main()
