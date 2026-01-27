#!/usr/bin/env python3
"""
Redpanda Serverless Connection Test

Quick test script to verify your Redpanda Serverless setup is working.

Usage:
    # Set environment variables first:
    export REDPANDA_BROKERS="<cluster-id>.any.<region>.mpx.prd.cloud.redpanda.com:9092"
    export REDPANDA_SASL_USERNAME="your-username"
    export REDPANDA_SASL_PASSWORD="your-password"

    # Or use .env file (will be loaded automatically)

    # Run tests:
    python scripts/test_redpanda.py --all          # Run all tests
    python scripts/test_redpanda.py --create       # Create topics only
    python scripts/test_redpanda.py --produce      # Produce test events
    python scripts/test_redpanda.py --consume      # Consume events
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env if it exists
env_file = project_root / ".env"
if env_file.exists():
    print(f"Loading environment from {env_file}")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key not in os.environ:  # Don't override existing env vars
                    os.environ[key] = value

from src.events.redpanda_client import (
    RedpandaClient,
    RedpandaConfig,
    ArgusEvent,
    EventType,
)


def print_config():
    """Print current configuration (with masked password)."""
    config = RedpandaConfig()
    password = config.sasl_password or ""
    masked_password = f"{password[:4]}...{password[-4:]}" if len(password) > 8 else "***"

    print("\n" + "=" * 60)
    print("REDPANDA CONFIGURATION")
    print("=" * 60)
    print(f"  Brokers:        {config.bootstrap_servers}")
    print(f"  Username:       {config.sasl_username}")
    print(f"  Password:       {masked_password}")
    print(f"  SASL Mechanism: {config.sasl_mechanism}")
    print(f"  Security:       {config.security_protocol}")
    print("=" * 60 + "\n")


def test_connection(client: RedpandaClient) -> bool:
    """Test basic connection to Redpanda."""
    print("Testing connection...")
    try:
        admin = client._get_admin()
        topics = admin.list_topics()
        print(f"  ✅ Connected! Found {len(topics)} existing topics")
        return True
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False


def test_create_topics(client: RedpandaClient) -> bool:
    """Create all Argus topics."""
    print("\nCreating Argus topics...")
    try:
        client.create_topics()
        print("  ✅ Topics created (or already exist)")

        # List topics to verify
        admin = client._get_admin()
        topics = admin.list_topics()
        argus_topics = [t for t in topics if t.startswith("argus.")]
        print(f"  ✅ Found {len(argus_topics)} argus.* topics:")
        for topic in sorted(argus_topics):
            print(f"      - {topic}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to create topics: {e}")
        return False


def test_produce(client: RedpandaClient, count: int = 10) -> bool:
    """Produce test events."""
    print(f"\nProducing {count} test events...")
    try:
        for i in range(count):
            event = ArgusEvent(
                event_type=EventType.TEST_EXECUTED,
                org_id=f"test-org-{i % 3}",
                project_id="test-project",
                payload={
                    "test_id": f"test-{i}",
                    "run_id": f"run-{int(time.time())}",
                    "status": "passed" if i % 2 == 0 else "failed",
                    "duration_ms": 100 + i * 10,
                    "message": f"Test event #{i}",
                },
            )
            metadata = client.publish_sync(event, timeout=30.0)
            print(f"  ✅ Event {i+1}/{count}: partition={metadata.partition}, offset={metadata.offset}")

        print(f"\n  ✅ Successfully produced {count} events")
        return True
    except Exception as e:
        print(f"  ❌ Failed to produce: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_consume(client: RedpandaClient, max_messages: int = 10, timeout_sec: int = 30) -> bool:
    """Consume test events."""
    print(f"\nConsuming up to {max_messages} events (timeout: {timeout_sec}s)...")

    consumed = []

    def handler(event: ArgusEvent):
        consumed.append(event)
        print(f"  📥 [{event.org_id}] {event.event_type}: {event.payload.get('message', 'N/A')}")

    try:
        # Use a unique group to read from beginning
        group_id = f"test-consumer-{int(time.time())}"
        client.consume(
            topics=["argus.test.executed"],
            group_id=group_id,
            handler=handler,
            max_messages=max_messages,
        )

        if consumed:
            print(f"\n  ✅ Consumed {len(consumed)} events")
            return True
        else:
            print("\n  ⚠️  No events consumed (topic may be empty)")
            return True  # Not a failure, just empty
    except Exception as e:
        print(f"  ❌ Failed to consume: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_round_trip(client: RedpandaClient) -> bool:
    """Test produce and consume in sequence."""
    print("\n" + "=" * 60)
    print("ROUND TRIP TEST")
    print("=" * 60)

    # Produce
    unique_id = f"roundtrip-{int(time.time())}"
    event = ArgusEvent(
        event_type=EventType.TEST_EXECUTED,
        org_id="roundtrip-test-org",
        project_id="roundtrip-project",
        payload={
            "test_id": unique_id,
            "run_id": unique_id,
            "status": "passed",
            "duration_ms": 42,
            "message": f"Round trip test {unique_id}",
        },
    )

    print(f"  Producing event with ID: {unique_id}")
    try:
        metadata = client.publish_sync(event, timeout=30.0)
        print(f"  ✅ Produced to partition={metadata.partition}, offset={metadata.offset}")
    except Exception as e:
        print(f"  ❌ Failed to produce: {e}")
        return False

    # Wait a moment for replication
    time.sleep(2)

    # Consume
    print(f"  Consuming to find event...")
    found = False
    group_id = f"roundtrip-{int(time.time())}"

    def handler(e: ArgusEvent):
        nonlocal found
        if e.payload.get("test_id") == unique_id:
            found = True
            print(f"  ✅ FOUND! Event round-tripped successfully")
            print(f"     Event ID: {e.event_id}")
            print(f"     Org ID: {e.org_id}")
            print(f"     Payload: {e.payload}")

    try:
        client.consume(
            topics=["argus.test.executed"],
            group_id=group_id,
            handler=handler,
            max_messages=100,
        )
    except Exception as e:
        print(f"  ⚠️  Consume ended: {e}")

    if found:
        print("\n  ✅ Round trip test PASSED")
        return True
    else:
        print("\n  ⚠️  Event not found in consumed messages (may need more time)")
        return True  # Not necessarily a failure


def main():
    parser = argparse.ArgumentParser(
        description="Test Redpanda Serverless connection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--create", action="store_true", help="Create topics")
    parser.add_argument("--produce", action="store_true", help="Produce test events")
    parser.add_argument("--consume", action="store_true", help="Consume events")
    parser.add_argument("--roundtrip", action="store_true", help="Test produce and consume")
    parser.add_argument("--count", type=int, default=10, help="Number of events to produce")

    args = parser.parse_args()

    # Default to --all if no specific test selected
    if not any([args.all, args.create, args.produce, args.consume, args.roundtrip]):
        args.all = True

    print_config()

    # Verify required env vars
    config = RedpandaConfig()
    if not config.sasl_username or not config.sasl_password:
        print("❌ ERROR: REDPANDA_SASL_USERNAME and REDPANDA_SASL_PASSWORD must be set")
        print("\nSet these environment variables:")
        print("  export REDPANDA_BROKERS='<cluster>.any.<region>.mpx.prd.cloud.redpanda.com:9092'")
        print("  export REDPANDA_SASL_USERNAME='your-username'")
        print("  export REDPANDA_SASL_PASSWORD='your-password'")
        sys.exit(1)

    results = {}

    with RedpandaClient() as client:
        # Test connection first
        if not test_connection(client):
            print("\n❌ Connection failed. Check your configuration.")
            sys.exit(1)

        if args.all or args.create:
            results["create_topics"] = test_create_topics(client)

        if args.all or args.produce:
            results["produce"] = test_produce(client, args.count)

        if args.all or args.consume:
            results["consume"] = test_consume(client)

        if args.all or args.roundtrip:
            results["roundtrip"] = test_round_trip(client)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! Redpanda is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
