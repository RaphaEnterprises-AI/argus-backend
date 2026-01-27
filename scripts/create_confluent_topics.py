#!/usr/bin/env python3
"""Create Argus topics in Confluent Cloud."""

from confluent_kafka.admin import AdminClient, NewTopic

# Confluent Cloud configuration
config = {
    "bootstrap.servers": "pkc-9q8rv.ap-south-2.aws.confluent.cloud:9092",
    "security.protocol": "SASL_SSL",
    "sasl.mechanism": "PLAIN",
    "sasl.username": "6APHEEKPOS6HAZ46",
    "sasl.password": "cflt6S4Fs2VhtY0D1fbZrUIVhWKVwsJtRcnJnk8WuA92hFti+KylNdSLMmyMGhRw",
}

# Topics to create (Confluent Cloud manages replication automatically)
TOPICS = {
    "argus.test.executed": {"partitions": 6},
    "argus.codebase.ingested": {"partitions": 6},
    "argus.error.reported": {"partitions": 6},
    "argus.dlq": {"partitions": 3},
}

def main():
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

    print(f"\n📋 Argus topics in cluster:")
    for topic in sorted(argus_topics):
        partitions = len(metadata.topics[topic].partitions)
        print(f"   ✅ {topic} ({partitions} partitions)")

    print("\n🎉 Done!")

if __name__ == "__main__":
    main()
