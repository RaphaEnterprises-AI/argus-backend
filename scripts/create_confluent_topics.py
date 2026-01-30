#!/usr/bin/env python3
"""Create Argus topics in Confluent Cloud.

Usage:
    # Set environment variables first
    export CONFLUENT_BOOTSTRAP_SERVERS="pkc-xxx.region.aws.confluent.cloud:9092"
    export CONFLUENT_SASL_USERNAME="your-api-key"
    export CONFLUENT_SASL_PASSWORD="your-api-secret"

    # Then run
    python scripts/create_confluent_topics.py
"""

import os
import sys

from confluent_kafka.admin import AdminClient, NewTopic


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


# Topics to create (Confluent Cloud manages replication automatically)
TOPICS = {
    "argus.test.executed": {"partitions": 6},
    "argus.codebase.ingested": {"partitions": 6},
    "argus.error.reported": {"partitions": 6},
    "argus.dlq": {"partitions": 3},
}


def main():
    config = get_config()

    print("Connecting to Confluent Cloud...")
    admin = AdminClient(config)

    # Create NewTopic objects
    # Note: replication_factor=-1 means use cluster default (typically 3 for Confluent Cloud)
    new_topics = [
        NewTopic(
            topic=name,
            num_partitions=cfg["partitions"],
            replication_factor=-1,  # Use cluster default
        )
        for name, cfg in TOPICS.items()
    ]

    print(f"\nCreating {len(new_topics)} topics...")
    futures = admin.create_topics(new_topics, operation_timeout=60)

    # Wait for results
    for topic, future in futures.items():
        try:
            future.result()
            print(f"   ✅ Created: {topic}")
        except Exception as e:
            error_str = str(e)
            if "TOPIC_ALREADY_EXISTS" in error_str:
                print(f"   ⏭️  Already exists: {topic}")
            else:
                print(f"   ❌ Failed: {topic} - {e}")

    # Verify
    print("\nVerifying topics...")
    metadata = admin.list_topics(timeout=30)
    argus_topics = [t for t in metadata.topics.keys() if t.startswith("argus.")]

    print("\n📋 Argus topics in cluster:")
    for topic in sorted(argus_topics):
        partitions = len(metadata.topics[topic].partitions)
        print(f"   ✅ {topic} ({partitions} partitions)")

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
