#!/usr/bin/env python3
"""Validate Confluent Cloud topics are created correctly."""

from confluent_kafka.admin import AdminClient

# Confluent Cloud configuration
config = {
    "bootstrap.servers": "pkc-9q8rv.ap-south-2.aws.confluent.cloud:9092",
    "security.protocol": "SASL_SSL",
    "sasl.mechanism": "PLAIN",
    "sasl.username": "6APHEEKPOS6HAZ46",
    "sasl.password": "cflt6S4Fs2VhtY0D1fbZrUIVhWKVwsJtRcnJnk8WuA92hFti+KylNdSLMmyMGhRw",
}

# Expected topics
EXPECTED_TOPICS = [
    "argus.test.executed",
    "argus.codebase.ingested",
    "argus.error.reported",
    "argus.dlq",
]

def main():
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
        print(f"\n⚠️  Missing topics:")
        for t in missing:
            print(f"   ❌ {t}")
        return False

    print(f"\n🎉 All {len(EXPECTED_TOPICS)} expected topics are present!")
    return True

if __name__ == "__main__":
    main()
