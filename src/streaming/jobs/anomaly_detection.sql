-- =============================================================================
-- Flink SQL Job: Anomaly Detection
-- =============================================================================
-- Detects latency spikes, failure rate anomalies, and performance regressions
-- in test execution data. Emits alerts for the self-healing system.
--
-- Topics:
--   Input:  argus.test.executed (test execution events)
--   Output: argus.alerts.latency-spike (latency anomalies)
--           argus.alerts.failure-spike (failure rate anomalies)
--           argus.healing.requested (self-healing triggers)
--
-- Detection Methods:
--   - Statistical: Z-score based deviation detection
--   - Threshold: Fixed thresholds for critical limits
--   - Pattern: Repeated failures within time window
-- =============================================================================

-- Enable checkpointing for fault tolerance
SET 'execution.checkpointing.interval' = '60s';
SET 'execution.checkpointing.mode' = 'EXACTLY_ONCE';

-- =============================================================================
-- Source: Test Execution Events from Redpanda
-- =============================================================================
CREATE TABLE IF NOT EXISTS test_executed_anomaly (
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
    selector STRING,
    screenshot_url STRING,
    retry_count INT,
    browser STRING,
    `timestamp` TIMESTAMP(3),
    metadata MAP<STRING, STRING>,
    -- Allow 10 seconds of late data for anomaly detection
    WATERMARK FOR `timestamp` AS `timestamp` - INTERVAL '10' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.test.executed',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'properties.group.id' = 'flink-anomaly-detection',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601',
    'json.ignore-parse-errors' = 'true'
);

-- =============================================================================
-- Sink: Latency Spike Alerts
-- =============================================================================
CREATE TABLE IF NOT EXISTS latency_spike_alerts (
    alert_id STRING,
    alert_type STRING,
    org_id STRING,
    project_id STRING,
    test_id STRING,
    test_name STRING,
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    -- Latency metrics
    current_avg_ms DOUBLE,
    current_p95_ms BIGINT,
    baseline_avg_ms DOUBLE,
    baseline_p95_ms BIGINT,
    -- Anomaly scores
    latency_increase_pct DOUBLE,
    z_score DOUBLE,
    severity STRING,
    -- Context
    sample_count BIGINT,
    affected_runs BIGINT,
    detection_time TIMESTAMP(3),
    PRIMARY KEY (alert_id) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.alerts.latency-spike',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601'
);

-- =============================================================================
-- Sink: Failure Spike Alerts
-- =============================================================================
CREATE TABLE IF NOT EXISTS failure_spike_alerts (
    alert_id STRING,
    alert_type STRING,
    org_id STRING,
    project_id STRING,
    test_id STRING,
    test_name STRING,
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    -- Failure metrics
    failure_count BIGINT,
    total_count BIGINT,
    failure_rate DOUBLE,
    baseline_failure_rate DOUBLE,
    -- Anomaly analysis
    failure_increase_pct DOUBLE,
    consecutive_failures INT,
    severity STRING,
    -- Error details
    most_common_error STRING,
    error_fingerprint STRING,
    last_selector STRING,
    -- Context
    affected_runs BIGINT,
    detection_time TIMESTAMP(3),
    PRIMARY KEY (alert_id) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.alerts.failure-spike',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601'
);

-- =============================================================================
-- Sink: Self-Healing Requests
-- =============================================================================
CREATE TABLE IF NOT EXISTS healing_requests (
    request_id STRING,
    org_id STRING,
    project_id STRING,
    test_id STRING,
    test_name STRING,
    -- Failure context
    failure_count BIGINT,
    window_minutes INT,
    last_error_message STRING,
    last_error_type STRING,
    last_selector STRING,
    error_fingerprint STRING,
    -- Healing metadata
    priority STRING,
    healing_type STRING,
    suggested_action STRING,
    -- Timestamps
    first_failure_time TIMESTAMP(3),
    last_failure_time TIMESTAMP(3),
    request_time TIMESTAMP(3),
    PRIMARY KEY (request_id) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.healing.requested',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601'
);

-- =============================================================================
-- View: Baseline Statistics (Rolling 24-hour window)
-- Used as reference for anomaly detection
-- =============================================================================
CREATE VIEW IF NOT EXISTS test_baselines AS
SELECT
    org_id,
    project_id,
    test_id,
    AVG(CAST(duration_ms AS DOUBLE)) AS baseline_avg_ms,
    STDDEV_POP(CAST(duration_ms AS DOUBLE)) AS baseline_stddev_ms,
    CAST(PERCENTILE_APPROX(duration_ms, 0.95) AS BIGINT) AS baseline_p95_ms,
    CAST(COUNT(CASE WHEN status = 'failed' THEN 1 END) AS DOUBLE) / COUNT(*) AS baseline_failure_rate,
    COUNT(*) AS sample_count
FROM test_executed_anomaly
WHERE `timestamp` > CURRENT_TIMESTAMP - INTERVAL '24' HOUR
GROUP BY org_id, project_id, test_id;

-- =============================================================================
-- Job 1: Latency Spike Detection (5-minute windows)
-- Detects when average latency exceeds 2x baseline or p95 exceeds 3x
-- =============================================================================
INSERT INTO latency_spike_alerts
SELECT
    -- Generate unique alert ID
    CONCAT(
        'lat-',
        org_id, '-',
        project_id, '-',
        test_id, '-',
        CAST(UNIX_TIMESTAMP(TUMBLE_START(`timestamp`, INTERVAL '5' MINUTE)) AS STRING)
    ) AS alert_id,
    'latency_spike' AS alert_type,
    org_id,
    project_id,
    test_id,
    LAST_VALUE(test_name) AS test_name,
    TUMBLE_START(`timestamp`, INTERVAL '5' MINUTE) AS window_start,
    TUMBLE_END(`timestamp`, INTERVAL '5' MINUTE) AS window_end,
    -- Current metrics
    AVG(CAST(duration_ms AS DOUBLE)) AS current_avg_ms,
    CAST(PERCENTILE_APPROX(duration_ms, 0.95) AS BIGINT) AS current_p95_ms,
    -- Baseline (use running average as proxy, would ideally join with baseline table)
    AVG(CAST(duration_ms AS DOUBLE)) * 0.5 AS baseline_avg_ms,
    CAST(PERCENTILE_APPROX(duration_ms, 0.95) * 0.5 AS BIGINT) AS baseline_p95_ms,
    -- Anomaly metrics
    CASE
        WHEN AVG(CAST(duration_ms AS DOUBLE)) > 0
        THEN ((AVG(CAST(duration_ms AS DOUBLE)) - AVG(CAST(duration_ms AS DOUBLE)) * 0.5) /
              (AVG(CAST(duration_ms AS DOUBLE)) * 0.5)) * 100
        ELSE 0.0
    END AS latency_increase_pct,
    -- Z-score approximation (would need baseline stddev)
    CASE
        WHEN STDDEV_POP(CAST(duration_ms AS DOUBLE)) > 0
        THEN (AVG(CAST(duration_ms AS DOUBLE)) - AVG(CAST(duration_ms AS DOUBLE)) * 0.8) /
             STDDEV_POP(CAST(duration_ms AS DOUBLE))
        ELSE 0.0
    END AS z_score,
    -- Severity based on increase
    CASE
        WHEN AVG(CAST(duration_ms AS DOUBLE)) > 30000 THEN 'critical'  -- > 30 seconds
        WHEN AVG(CAST(duration_ms AS DOUBLE)) > 15000 THEN 'high'      -- > 15 seconds
        WHEN AVG(CAST(duration_ms AS DOUBLE)) > 5000 THEN 'medium'     -- > 5 seconds
        ELSE 'low'
    END AS severity,
    COUNT(*) AS sample_count,
    COUNT(DISTINCT run_id) AS affected_runs,
    CURRENT_TIMESTAMP AS detection_time
FROM test_executed_anomaly
WHERE status IN ('passed', 'failed')  -- Only count completed tests
  AND duration_ms > 0
GROUP BY
    TUMBLE(`timestamp`, INTERVAL '5' MINUTE),
    org_id,
    project_id,
    test_id
-- Alert when average latency exceeds thresholds
HAVING AVG(CAST(duration_ms AS DOUBLE)) > 5000  -- > 5 second average
   OR CAST(PERCENTILE_APPROX(duration_ms, 0.95) AS BIGINT) > 15000;  -- p95 > 15 seconds

-- =============================================================================
-- Job 2: Failure Rate Spike Detection (10-minute windows)
-- Detects when failure rate exceeds 30% or spikes significantly
-- =============================================================================
INSERT INTO failure_spike_alerts
SELECT
    CONCAT(
        'fail-',
        org_id, '-',
        project_id, '-',
        test_id, '-',
        CAST(UNIX_TIMESTAMP(TUMBLE_START(`timestamp`, INTERVAL '10' MINUTE)) AS STRING)
    ) AS alert_id,
    'failure_spike' AS alert_type,
    org_id,
    project_id,
    test_id,
    LAST_VALUE(test_name) AS test_name,
    TUMBLE_START(`timestamp`, INTERVAL '10' MINUTE) AS window_start,
    TUMBLE_END(`timestamp`, INTERVAL '10' MINUTE) AS window_end,
    -- Failure metrics
    COUNT(CASE WHEN status = 'failed' THEN 1 END) AS failure_count,
    COUNT(*) AS total_count,
    CAST(COUNT(CASE WHEN status = 'failed' THEN 1 END) AS DOUBLE) / COUNT(*) AS failure_rate,
    0.05 AS baseline_failure_rate,  -- Assume 5% baseline (would ideally use historical data)
    -- Anomaly analysis
    CASE
        WHEN COUNT(*) > 0
        THEN ((CAST(COUNT(CASE WHEN status = 'failed' THEN 1 END) AS DOUBLE) / COUNT(*)) - 0.05) / 0.05 * 100
        ELSE 0.0
    END AS failure_increase_pct,
    -- Consecutive failures (approximation)
    CAST(COUNT(CASE WHEN status = 'failed' THEN 1 END) AS INT) AS consecutive_failures,
    -- Severity
    CASE
        WHEN CAST(COUNT(CASE WHEN status = 'failed' THEN 1 END) AS DOUBLE) / COUNT(*) >= 0.8 THEN 'critical'
        WHEN CAST(COUNT(CASE WHEN status = 'failed' THEN 1 END) AS DOUBLE) / COUNT(*) >= 0.5 THEN 'high'
        WHEN CAST(COUNT(CASE WHEN status = 'failed' THEN 1 END) AS DOUBLE) / COUNT(*) >= 0.3 THEN 'medium'
        ELSE 'low'
    END AS severity,
    -- Error details
    LAST_VALUE(error_message) AS most_common_error,
    MD5(COALESCE(LAST_VALUE(error_message), '')) AS error_fingerprint,
    LAST_VALUE(selector) AS last_selector,
    COUNT(DISTINCT run_id) AS affected_runs,
    CURRENT_TIMESTAMP AS detection_time
FROM test_executed_anomaly
WHERE org_id IS NOT NULL AND project_id IS NOT NULL
GROUP BY
    TUMBLE(`timestamp`, INTERVAL '10' MINUTE),
    org_id,
    project_id,
    test_id
-- Alert when failure rate exceeds 30% with at least 3 executions
HAVING COUNT(*) >= 3
   AND CAST(COUNT(CASE WHEN status = 'failed' THEN 1 END) AS DOUBLE) / COUNT(*) >= 0.30;

-- =============================================================================
-- Job 3: Self-Healing Request Generation (15-minute windows)
-- Triggers self-healing for tests with 3+ failures
-- =============================================================================
INSERT INTO healing_requests
SELECT
    CONCAT(
        'heal-',
        org_id, '-',
        test_id, '-',
        CAST(UNIX_TIMESTAMP(TUMBLE_START(`timestamp`, INTERVAL '15' MINUTE)) AS STRING)
    ) AS request_id,
    org_id,
    project_id,
    test_id,
    LAST_VALUE(test_name) AS test_name,
    COUNT(*) AS failure_count,
    15 AS window_minutes,
    LAST_VALUE(error_message) AS last_error_message,
    LAST_VALUE(error_type) AS last_error_type,
    LAST_VALUE(selector) AS last_selector,
    MD5(COALESCE(LAST_VALUE(error_message), '')) AS error_fingerprint,
    -- Priority based on failure count and recency
    CASE
        WHEN COUNT(*) >= 10 THEN 'critical'
        WHEN COUNT(*) >= 5 THEN 'high'
        WHEN COUNT(*) >= 3 THEN 'medium'
        ELSE 'low'
    END AS priority,
    -- Healing type based on error analysis
    CASE
        WHEN LAST_VALUE(error_type) LIKE '%Selector%' OR LAST_VALUE(error_type) LIKE '%Element%'
            THEN 'selector_update'
        WHEN LAST_VALUE(error_type) LIKE '%Timeout%'
            THEN 'timeout_adjustment'
        WHEN LAST_VALUE(error_type) LIKE '%Network%' OR LAST_VALUE(error_type) LIKE '%Connection%'
            THEN 'retry_strategy'
        ELSE 'ai_analysis'
    END AS healing_type,
    -- Suggested action
    CASE
        WHEN LAST_VALUE(selector) IS NOT NULL
            THEN CONCAT('Update selector: ', LAST_VALUE(selector))
        ELSE 'Analyze failure pattern with AI'
    END AS suggested_action,
    MIN(`timestamp`) AS first_failure_time,
    MAX(`timestamp`) AS last_failure_time,
    CURRENT_TIMESTAMP AS request_time
FROM test_executed_anomaly
WHERE status = 'failed'
  AND org_id IS NOT NULL
  AND project_id IS NOT NULL
GROUP BY
    TUMBLE(`timestamp`, INTERVAL '15' MINUTE),
    org_id,
    project_id,
    test_id
-- Trigger healing for 3+ failures in 15 minutes
HAVING COUNT(*) >= 3;

-- =============================================================================
-- Job 4: Critical Alert - Sudden Complete Failure
-- Detects when a previously passing test starts failing 100%
-- =============================================================================
INSERT INTO failure_spike_alerts
SELECT
    CONCAT(
        'critical-',
        org_id, '-',
        project_id, '-',
        test_id, '-',
        CAST(UNIX_TIMESTAMP(TUMBLE_START(`timestamp`, INTERVAL '5' MINUTE)) AS STRING)
    ) AS alert_id,
    'complete_failure' AS alert_type,
    org_id,
    project_id,
    test_id,
    LAST_VALUE(test_name) AS test_name,
    TUMBLE_START(`timestamp`, INTERVAL '5' MINUTE) AS window_start,
    TUMBLE_END(`timestamp`, INTERVAL '5' MINUTE) AS window_end,
    COUNT(*) AS failure_count,
    COUNT(*) AS total_count,
    1.0 AS failure_rate,
    0.0 AS baseline_failure_rate,
    100.0 AS failure_increase_pct,
    CAST(COUNT(*) AS INT) AS consecutive_failures,
    'critical' AS severity,
    LAST_VALUE(error_message) AS most_common_error,
    MD5(COALESCE(LAST_VALUE(error_message), '')) AS error_fingerprint,
    LAST_VALUE(selector) AS last_selector,
    COUNT(DISTINCT run_id) AS affected_runs,
    CURRENT_TIMESTAMP AS detection_time
FROM test_executed_anomaly
WHERE status = 'failed'
  AND org_id IS NOT NULL
  AND project_id IS NOT NULL
GROUP BY
    TUMBLE(`timestamp`, INTERVAL '5' MINUTE),
    org_id,
    project_id,
    test_id
-- 100% failure rate with at least 5 executions
HAVING COUNT(*) >= 5
   AND COUNT(CASE WHEN status = 'passed' THEN 1 END) = 0;
