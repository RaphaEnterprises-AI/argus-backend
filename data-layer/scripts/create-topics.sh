#!/bin/bash
# =============================================================================
# Argus Data Layer - Redpanda Topic Creation Script
# =============================================================================
# Creates all required Kafka topics for the Argus E2E Testing Agent
# Run this script after Redpanda is healthy
# =============================================================================

set -e

# Configuration
REDPANDA_BROKERS=${REDPANDA_BROKERS:-"localhost:19092"}
REPLICATION_FACTOR=${REPLICATION_FACTOR:-1}  # Use 1 for dev, 3 for production

echo "=========================================="
echo "Argus Data Layer - Topic Creation"
echo "=========================================="
echo "Brokers: ${REDPANDA_BROKERS}"
echo "Replication Factor: ${REPLICATION_FACTOR}"
echo ""

# Function to create a topic
create_topic() {
    local topic_name=$1
    local partitions=$2
    local description=$3

    echo "Creating topic: ${topic_name} (${partitions} partitions)"
    echo "  Description: ${description}"

    rpk topic create "${topic_name}" \
        --brokers "${REDPANDA_BROKERS}" \
        --partitions "${partitions}" \
        --replicas "${REPLICATION_FACTOR}" \
        2>/dev/null || echo "  Topic already exists or creation failed"

    echo ""
}

# =============================================================================
# Codebase Events
# =============================================================================
echo "--- Codebase Events ---"

create_topic "argus.codebase.ingested" 6 \
    "Triggered when a new codebase is ingested for analysis"

create_topic "argus.codebase.analyzed" 6 \
    "Triggered when codebase analysis is complete with test surfaces identified"

# =============================================================================
# Test Lifecycle Events
# =============================================================================
echo "--- Test Lifecycle Events ---"

create_topic "argus.test.created" 12 \
    "Triggered when a new test is generated (high volume)"

create_topic "argus.test.executed" 12 \
    "Triggered when a test execution completes (high volume)"

create_topic "argus.test.failed" 6 \
    "Triggered when a test fails - triggers healing flow"

# =============================================================================
# Self-Healing Events
# =============================================================================
echo "--- Self-Healing Events ---"

create_topic "argus.healing.requested" 6 \
    "Triggered when self-healing is requested for a failed test"

create_topic "argus.healing.completed" 6 \
    "Triggered when self-healing completes (success or failure)"

# =============================================================================
# Integration Events
# =============================================================================
echo "--- Integration Events ---"

create_topic "argus.integration.github" 6 \
    "GitHub webhook events (PRs, commits, comments)"

create_topic "argus.integration.jira" 3 \
    "Jira integration events (issue creation, updates)"

create_topic "argus.integration.slack" 3 \
    "Slack notification events"

# =============================================================================
# Notification Events
# =============================================================================
echo "--- Notification Events ---"

create_topic "argus.notification.send" 6 \
    "Generic notification dispatch events"

# =============================================================================
# Dead Letter Queue
# =============================================================================
echo "--- Dead Letter Queue ---"

create_topic "argus.dlq" 3 \
    "Dead letter queue for failed message processing"

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "Topic Creation Complete"
echo "=========================================="
echo ""
echo "Listing all topics:"
rpk topic list --brokers "${REDPANDA_BROKERS}"
echo ""
echo "To view topic details:"
echo "  rpk topic describe <topic-name> --brokers ${REDPANDA_BROKERS}"
echo ""
echo "To view messages:"
echo "  rpk topic consume <topic-name> --brokers ${REDPANDA_BROKERS}"
