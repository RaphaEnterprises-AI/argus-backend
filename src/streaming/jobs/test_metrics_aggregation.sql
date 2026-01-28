-- =============================================================================
-- Flink SQL Job: Test Metrics Aggregation
-- =============================================================================
-- Aggregates test execution results by time window for real-time dashboards.
-- Computes pass rates, durations, and counts per organization/project.
--
-- Topics:
--   Input:  argus.test.executed (test execution events)
--   Output: argus.metrics.test-summary (aggregated metrics)
--
-- Windows:
--   - 1-minute micro-batches for near-real-time updates
--   - 5-minute windows for standard dashboards
--   - 1-hour windows for trend analysis
-- =============================================================================

-- Enable checkpointing for fault tolerance
SET 'execution.checkpointing.interval' = '60s';
SET 'execution.checkpointing.mode' = 'EXACTLY_ONCE';

-- =============================================================================
-- Source: Test Execution Events from Redpanda
-- =============================================================================
CREATE TABLE IF NOT EXISTS test_executed (
    event_id STRING,
    event_type STRING,
    org_id STRING,
    project_id STRING,
    test_id STRING,
    test_name STRING,
    run_id STRING,
    status STRING,
    duration_ms BIGINT,
    error_message STRING,
    error_type STRING,
    retry_count INT,
    browser STRING,
    `timestamp` TIMESTAMP(3),
    metadata MAP<STRING, STRING>,
    -- Watermark: allow 5 seconds of late data
    WATERMARK FOR `timestamp` AS `timestamp` - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.test.executed',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'properties.group.id' = 'flink-test-metrics-aggregation',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601',
    'json.ignore-parse-errors' = 'true'
);

-- =============================================================================
-- Sink: 1-Minute Aggregated Metrics (Near Real-Time)
-- =============================================================================
CREATE TABLE IF NOT EXISTS test_metrics_1m (
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    org_id STRING,
    project_id STRING,
    total_tests BIGINT,
    passed_tests BIGINT,
    failed_tests BIGINT,
    skipped_tests BIGINT,
    errored_tests BIGINT,
    pass_rate DOUBLE,
    avg_duration_ms DOUBLE,
    min_duration_ms BIGINT,
    max_duration_ms BIGINT,
    p50_duration_ms BIGINT,
    p90_duration_ms BIGINT,
    p95_duration_ms BIGINT,
    p99_duration_ms BIGINT,
    total_retries BIGINT,
    processing_time TIMESTAMP(3),
    PRIMARY KEY (window_start, org_id, project_id) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.metrics.test-summary-1m',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601'
);

-- =============================================================================
-- Sink: 5-Minute Aggregated Metrics (Standard Dashboards)
-- =============================================================================
CREATE TABLE IF NOT EXISTS test_metrics_5m (
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    org_id STRING,
    project_id STRING,
    total_tests BIGINT,
    passed_tests BIGINT,
    failed_tests BIGINT,
    skipped_tests BIGINT,
    errored_tests BIGINT,
    pass_rate DOUBLE,
    avg_duration_ms DOUBLE,
    min_duration_ms BIGINT,
    max_duration_ms BIGINT,
    p50_duration_ms BIGINT,
    p90_duration_ms BIGINT,
    p95_duration_ms BIGINT,
    p99_duration_ms BIGINT,
    total_retries BIGINT,
    unique_tests BIGINT,
    flaky_tests BIGINT,
    processing_time TIMESTAMP(3),
    PRIMARY KEY (window_start, org_id, project_id) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.metrics.test-summary-5m',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601'
);

-- =============================================================================
-- Sink: 1-Hour Aggregated Metrics (Trend Analysis)
-- =============================================================================
CREATE TABLE IF NOT EXISTS test_metrics_1h (
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    org_id STRING,
    project_id STRING,
    total_tests BIGINT,
    passed_tests BIGINT,
    failed_tests BIGINT,
    pass_rate DOUBLE,
    avg_duration_ms DOUBLE,
    p95_duration_ms BIGINT,
    total_retries BIGINT,
    unique_tests BIGINT,
    processing_time TIMESTAMP(3),
    PRIMARY KEY (window_start, org_id, project_id) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.metrics.test-summary-1h',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601'
);

-- =============================================================================
-- Job 1: 1-Minute Micro-Batch Aggregation
-- =============================================================================
INSERT INTO test_metrics_1m
SELECT
    TUMBLE_START(`timestamp`, INTERVAL '1' MINUTE) AS window_start,
    TUMBLE_END(`timestamp`, INTERVAL '1' MINUTE) AS window_end,
    org_id,
    project_id,
    COUNT(*) AS total_tests,
    COUNT(CASE WHEN status = 'passed' THEN 1 END) AS passed_tests,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) AS failed_tests,
    COUNT(CASE WHEN status = 'skipped' THEN 1 END) AS skipped_tests,
    COUNT(CASE WHEN status = 'error' THEN 1 END) AS errored_tests,
    CASE
        WHEN COUNT(*) > 0
        THEN CAST(COUNT(CASE WHEN status = 'passed' THEN 1 END) AS DOUBLE) / COUNT(*)
        ELSE 0.0
    END AS pass_rate,
    AVG(CAST(duration_ms AS DOUBLE)) AS avg_duration_ms,
    MIN(duration_ms) AS min_duration_ms,
    MAX(duration_ms) AS max_duration_ms,
    CAST(PERCENTILE_APPROX(duration_ms, 0.50) AS BIGINT) AS p50_duration_ms,
    CAST(PERCENTILE_APPROX(duration_ms, 0.90) AS BIGINT) AS p90_duration_ms,
    CAST(PERCENTILE_APPROX(duration_ms, 0.95) AS BIGINT) AS p95_duration_ms,
    CAST(PERCENTILE_APPROX(duration_ms, 0.99) AS BIGINT) AS p99_duration_ms,
    COALESCE(SUM(retry_count), 0) AS total_retries,
    CURRENT_TIMESTAMP AS processing_time
FROM test_executed
WHERE org_id IS NOT NULL AND project_id IS NOT NULL
GROUP BY
    TUMBLE(`timestamp`, INTERVAL '1' MINUTE),
    org_id,
    project_id;

-- =============================================================================
-- Job 2: 5-Minute Standard Aggregation
-- =============================================================================
INSERT INTO test_metrics_5m
SELECT
    TUMBLE_START(`timestamp`, INTERVAL '5' MINUTE) AS window_start,
    TUMBLE_END(`timestamp`, INTERVAL '5' MINUTE) AS window_end,
    org_id,
    project_id,
    COUNT(*) AS total_tests,
    COUNT(CASE WHEN status = 'passed' THEN 1 END) AS passed_tests,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) AS failed_tests,
    COUNT(CASE WHEN status = 'skipped' THEN 1 END) AS skipped_tests,
    COUNT(CASE WHEN status = 'error' THEN 1 END) AS errored_tests,
    CASE
        WHEN COUNT(*) > 0
        THEN CAST(COUNT(CASE WHEN status = 'passed' THEN 1 END) AS DOUBLE) / COUNT(*)
        ELSE 0.0
    END AS pass_rate,
    AVG(CAST(duration_ms AS DOUBLE)) AS avg_duration_ms,
    MIN(duration_ms) AS min_duration_ms,
    MAX(duration_ms) AS max_duration_ms,
    CAST(PERCENTILE_APPROX(duration_ms, 0.50) AS BIGINT) AS p50_duration_ms,
    CAST(PERCENTILE_APPROX(duration_ms, 0.90) AS BIGINT) AS p90_duration_ms,
    CAST(PERCENTILE_APPROX(duration_ms, 0.95) AS BIGINT) AS p95_duration_ms,
    CAST(PERCENTILE_APPROX(duration_ms, 0.99) AS BIGINT) AS p99_duration_ms,
    COALESCE(SUM(retry_count), 0) AS total_retries,
    COUNT(DISTINCT test_id) AS unique_tests,
    -- Flaky tests: same test with both pass and fail in window
    COUNT(DISTINCT CASE
        WHEN test_id IN (
            SELECT t2.test_id FROM test_executed t2
            WHERE t2.org_id = test_executed.org_id
            AND t2.project_id = test_executed.project_id
            AND t2.status = 'passed'
        ) AND status = 'failed' THEN test_id
    END) AS flaky_tests,
    CURRENT_TIMESTAMP AS processing_time
FROM test_executed
WHERE org_id IS NOT NULL AND project_id IS NOT NULL
GROUP BY
    TUMBLE(`timestamp`, INTERVAL '5' MINUTE),
    org_id,
    project_id;

-- =============================================================================
-- Job 3: 1-Hour Trend Aggregation
-- =============================================================================
INSERT INTO test_metrics_1h
SELECT
    TUMBLE_START(`timestamp`, INTERVAL '1' HOUR) AS window_start,
    TUMBLE_END(`timestamp`, INTERVAL '1' HOUR) AS window_end,
    org_id,
    project_id,
    COUNT(*) AS total_tests,
    COUNT(CASE WHEN status = 'passed' THEN 1 END) AS passed_tests,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) AS failed_tests,
    CASE
        WHEN COUNT(*) > 0
        THEN CAST(COUNT(CASE WHEN status = 'passed' THEN 1 END) AS DOUBLE) / COUNT(*)
        ELSE 0.0
    END AS pass_rate,
    AVG(CAST(duration_ms AS DOUBLE)) AS avg_duration_ms,
    CAST(PERCENTILE_APPROX(duration_ms, 0.95) AS BIGINT) AS p95_duration_ms,
    COALESCE(SUM(retry_count), 0) AS total_retries,
    COUNT(DISTINCT test_id) AS unique_tests,
    CURRENT_TIMESTAMP AS processing_time
FROM test_executed
WHERE org_id IS NOT NULL AND project_id IS NOT NULL
GROUP BY
    TUMBLE(`timestamp`, INTERVAL '1' HOUR),
    org_id,
    project_id;
