-- =============================================================================
-- Flink SQL Job: Failure Cluster Update
-- =============================================================================
-- Real-time job that clusters test failures by error fingerprint.
-- Aggregates failures into 15-minute tumbling windows for pattern analysis.
--
-- Topics:
--   Input:  argus.test.failed (test failure events)
--   Output: flink_failure_patterns table (Supabase via JDBC sink)
--
-- Features:
--   - Error fingerprinting via MD5 hash for pattern grouping
--   - Priority-based healing triggers (critical/high/medium/low)
--   - Idempotent writes to prevent duplicate processing across regions
--   - Multi-region support with processing_region tracking
--
-- RAP-246: https://linear.app/argus/issue/RAP-246
-- =============================================================================

-- Enable checkpointing for fault tolerance
SET 'execution.checkpointing.interval' = '60s';
SET 'execution.checkpointing.mode' = 'EXACTLY_ONCE';
SET 'execution.checkpointing.min-pause' = '30s';
SET 'state.checkpoints.dir' = 's3://argus-checkpoints/failure-cluster-update';

-- =============================================================================
-- Source: Test Failed Events from Redpanda
-- =============================================================================
-- Consumes test failure events for clustering and pattern detection.
-- Uses dedicated consumer group for failure processing.

CREATE TABLE IF NOT EXISTS argus_test_failed (
    -- Event identification
    event_id STRING,
    event_type STRING,

    -- Multi-tenant dimensions
    org_id STRING NOT NULL,
    project_id STRING NOT NULL,
    test_id STRING NOT NULL,
    test_name STRING,
    run_id STRING,

    -- Failure details
    status STRING,
    error_message STRING,
    error_type STRING,
    selector STRING,
    screenshot_url STRING,
    stack_trace STRING,

    -- Execution context
    browser STRING,
    environment STRING,
    retry_count INT,
    duration_ms BIGINT,

    -- Metadata
    metadata MAP<STRING, STRING>,

    -- Event timestamp for windowing
    event_time TIMESTAMP(3),

    -- Watermark: allow 30 seconds of late data for failure events
    -- Failure events may arrive with delay due to retry attempts
    WATERMARK FOR event_time AS event_time - INTERVAL '30' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.test.failed',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'properties.group.id' = 'flink-failure-cluster-update',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601',
    'json.ignore-parse-errors' = 'true',
    'json.fail-on-missing-field' = 'false'
);

-- =============================================================================
-- Sink: Failure Patterns to Supabase (via JDBC)
-- =============================================================================
-- Writes aggregated failure patterns directly to Supabase PostgreSQL.
-- Uses idempotency_key for exactly-once semantics across regions.

CREATE TABLE IF NOT EXISTS flink_failure_patterns (
    -- Idempotency key for exactly-once processing
    -- Format: {window_epoch}:{org_id}:{test_id}
    idempotency_key STRING NOT NULL,

    -- Multi-tenant dimensions
    org_id STRING NOT NULL,
    project_id STRING NOT NULL,
    test_id STRING NOT NULL,

    -- Time window boundaries
    window_start TIMESTAMP(3) NOT NULL,
    window_end TIMESTAMP(3) NOT NULL,

    -- Aggregated failure metrics
    failure_count BIGINT NOT NULL,
    last_error_message STRING,
    last_selector STRING,
    error_fingerprint STRING,

    -- Healing configuration
    healing_priority STRING,
    healing_requested BOOLEAN,

    -- Processing metadata
    processing_region STRING,

    PRIMARY KEY (idempotency_key) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = '${SUPABASE_JDBC_URL}',
    'table-name' = 'flink_failure_patterns',
    'username' = '${SUPABASE_DB_USER}',
    'password' = '${SUPABASE_DB_PASSWORD}',
    'driver' = 'org.postgresql.Driver',
    -- Upsert mode for idempotent writes
    'sink.buffer-flush.max-rows' = '100',
    'sink.buffer-flush.interval' = '5s',
    'sink.max-retries' = '3',
    -- Use upsert semantics with idempotency_key
    'sink.semantic' = 'exactly-once'
);

-- =============================================================================
-- Kafka Sink: Failure Pattern Events (for downstream consumers)
-- =============================================================================
-- Also emit failure patterns to Kafka for real-time consumers
-- (dashboards, alerting systems, self-healing agents)

CREATE TABLE IF NOT EXISTS failure_pattern_events (
    idempotency_key STRING,
    org_id STRING,
    project_id STRING,
    test_id STRING,
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    failure_count BIGINT,
    last_error_message STRING,
    last_selector STRING,
    error_fingerprint STRING,
    healing_priority STRING,
    healing_requested BOOLEAN,
    processing_region STRING,
    PRIMARY KEY (idempotency_key) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.patterns.failure-cluster',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601',
    'sink.partitioner' = 'fixed'
);

-- =============================================================================
-- Job: Failure Cluster Update (15-minute tumbling windows)
-- =============================================================================
-- Clusters failures by error fingerprint using tumbling windows.
-- Calculates healing priority based on failure count thresholds.

INSERT INTO flink_failure_patterns
SELECT
    -- Idempotency key: window_epoch:org:test
    -- Ensures exactly-once semantics when multiple regions process same data
    CONCAT(
        CAST(UNIX_TIMESTAMP(window_start) AS STRING), ':',
        org_id, ':',
        test_id
    ) AS idempotency_key,

    -- Dimensions for multi-tenant isolation
    org_id,
    project_id,
    test_id,

    -- Window boundaries
    window_start,
    window_end,

    -- Aggregated metrics
    COUNT(*) AS failure_count,
    LAST_VALUE(error_message) AS last_error_message,
    LAST_VALUE(selector) AS last_selector,

    -- Error fingerprint: MD5 hash of normalized error message
    -- Enables grouping similar errors across different runs
    MD5(COALESCE(LAST_VALUE(error_message), 'unknown')) AS error_fingerprint,

    -- Healing priority based on failure frequency thresholds
    -- critical: 10+ failures - immediate action required
    -- high: 5-9 failures - urgent attention needed
    -- medium: 3-4 failures - schedule for review
    -- low: 1-2 failures - monitor for recurrence
    CASE
        WHEN COUNT(*) >= 10 THEN 'critical'
        WHEN COUNT(*) >= 5 THEN 'high'
        WHEN COUNT(*) >= 3 THEN 'medium'
        ELSE 'low'
    END AS healing_priority,

    -- Request healing for tests with 3+ failures in window
    COUNT(*) >= 3 AS healing_requested,

    -- Track which region processed this record
    '${PROCESSING_REGION}' AS processing_region

FROM TABLE(
    TUMBLE(TABLE argus_test_failed, DESCRIPTOR(event_time), INTERVAL '15' MINUTE)
)
WHERE
    org_id IS NOT NULL
    AND project_id IS NOT NULL
    AND test_id IS NOT NULL
    AND status = 'failed'
GROUP BY
    org_id,
    project_id,
    test_id,
    window_start,
    window_end;

-- =============================================================================
-- Job: Emit Failure Patterns to Kafka (parallel sink)
-- =============================================================================
-- Duplicate output to Kafka for real-time consumers

INSERT INTO failure_pattern_events
SELECT
    CONCAT(
        CAST(UNIX_TIMESTAMP(window_start) AS STRING), ':',
        org_id, ':',
        test_id
    ) AS idempotency_key,
    org_id,
    project_id,
    test_id,
    window_start,
    window_end,
    COUNT(*) AS failure_count,
    LAST_VALUE(error_message) AS last_error_message,
    LAST_VALUE(selector) AS last_selector,
    MD5(COALESCE(LAST_VALUE(error_message), 'unknown')) AS error_fingerprint,
    CASE
        WHEN COUNT(*) >= 10 THEN 'critical'
        WHEN COUNT(*) >= 5 THEN 'high'
        WHEN COUNT(*) >= 3 THEN 'medium'
        ELSE 'low'
    END AS healing_priority,
    COUNT(*) >= 3 AS healing_requested,
    '${PROCESSING_REGION}' AS processing_region
FROM TABLE(
    TUMBLE(TABLE argus_test_failed, DESCRIPTOR(event_time), INTERVAL '15' MINUTE)
)
WHERE
    org_id IS NOT NULL
    AND project_id IS NOT NULL
    AND test_id IS NOT NULL
    AND status = 'failed'
GROUP BY
    org_id,
    project_id,
    test_id,
    window_start,
    window_end;

-- =============================================================================
-- Job: Critical Failure Alert (5-minute fast path)
-- =============================================================================
-- Faster detection for critical failures - 5 failures in 5 minutes
-- Sends immediate alerts without waiting for 15-minute window

INSERT INTO failure_pattern_events
SELECT
    CONCAT(
        'critical-',
        CAST(UNIX_TIMESTAMP(window_start) AS STRING), ':',
        org_id, ':',
        test_id
    ) AS idempotency_key,
    org_id,
    project_id,
    test_id,
    window_start,
    window_end,
    COUNT(*) AS failure_count,
    LAST_VALUE(error_message) AS last_error_message,
    LAST_VALUE(selector) AS last_selector,
    MD5(COALESCE(LAST_VALUE(error_message), 'unknown')) AS error_fingerprint,
    'critical' AS healing_priority,
    TRUE AS healing_requested,
    '${PROCESSING_REGION}' AS processing_region
FROM TABLE(
    TUMBLE(TABLE argus_test_failed, DESCRIPTOR(event_time), INTERVAL '5' MINUTE)
)
WHERE
    org_id IS NOT NULL
    AND project_id IS NOT NULL
    AND test_id IS NOT NULL
    AND status = 'failed'
GROUP BY
    org_id,
    project_id,
    test_id,
    window_start,
    window_end
-- Only emit critical alerts for rapid failure accumulation
HAVING COUNT(*) >= 5;
