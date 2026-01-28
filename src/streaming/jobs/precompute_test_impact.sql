-- =============================================================================
-- Flink SQL Job: Test Impact Matrix Precomputation
-- =============================================================================
-- Nightly batch job to compute which tests are affected by which files.
-- Analyzes 30 days of test execution history to build correlation matrix.
--
-- Schedule: 2 AM UTC via K8s CronJob (precompute-test-impact)
-- Output: intelligence_precomputed table (computation_type = 'test_impact_matrix')
--
-- Algorithm:
--   1. Join test executions with commit file changes
--   2. Compute correlation: files that change together with test failures
--   3. Build impact scores based on co-occurrence frequency and failure rates
--   4. Write aggregated results to precomputed table for instant lookup
--
-- Usage:
--   - API calls get_precomputed() for instant test selection
--   - CI/CD uses impact matrix to select relevant tests for changed files
--   - Dashboard shows file-to-test relationships
-- =============================================================================

-- Disable streaming mode for batch processing
SET 'execution.runtime-mode' = 'batch';

-- Enable checkpointing for fault tolerance
SET 'execution.checkpointing.interval' = '120s';
SET 'execution.checkpointing.mode' = 'EXACTLY_ONCE';

-- =============================================================================
-- Source: Test Execution Events from Redpanda (30-day lookback)
-- =============================================================================
CREATE TABLE IF NOT EXISTS test_executions_batch (
    event_id STRING,
    event_type STRING,
    org_id STRING,
    project_id STRING,
    test_id STRING,
    test_name STRING,
    test_file_path STRING,
    run_id STRING,
    commit_sha STRING,
    branch STRING,
    status STRING,
    duration_ms BIGINT,
    error_message STRING,
    error_type STRING,
    `timestamp` TIMESTAMP(3),
    metadata MAP<STRING, STRING>,
    WATERMARK FOR `timestamp` AS `timestamp` - INTERVAL '1' DAY
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.test.executed',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'properties.group.id' = 'flink-precompute-test-impact',
    -- Start from beginning for full 30-day analysis
    'scan.startup.mode' = 'timestamp',
    'scan.startup.timestamp-millis' = '${SCAN_START_TIMESTAMP}',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601',
    'json.ignore-parse-errors' = 'true'
);

-- =============================================================================
-- Source: Commit File Changes from Redpanda
-- =============================================================================
CREATE TABLE IF NOT EXISTS commit_files_batch (
    event_id STRING,
    org_id STRING,
    project_id STRING,
    commit_sha STRING,
    file_path STRING,
    change_type STRING,  -- 'added', 'modified', 'deleted', 'renamed'
    lines_added INT,
    lines_removed INT,
    `timestamp` TIMESTAMP(3),
    WATERMARK FOR `timestamp` AS `timestamp` - INTERVAL '1' DAY
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.vcs.commit-files',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'properties.group.id' = 'flink-precompute-commit-files',
    'scan.startup.mode' = 'timestamp',
    'scan.startup.timestamp-millis' = '${SCAN_START_TIMESTAMP}',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601',
    'json.ignore-parse-errors' = 'true'
);

-- =============================================================================
-- Sink: Precomputed Results to Kafka (consumed by Supabase sink)
-- =============================================================================
CREATE TABLE IF NOT EXISTS precomputed_results (
    org_id STRING,
    project_id STRING,
    computation_type STRING,
    result STRING,  -- JSON string
    computed_at TIMESTAMP(3),
    valid_until TIMESTAMP(3),
    PRIMARY KEY (org_id, project_id, computation_type) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.intelligence.precomputed',
    'properties.bootstrap.servers' = '${KAFKA_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = '${KAFKA_SECURITY_PROTOCOL:SASL_PLAINTEXT}',
    'properties.sasl.mechanism' = '${KAFKA_SASL_MECHANISM:SCRAM-SHA-512}',
    'properties.sasl.jaas.config' = 'org.apache.flink.kafka.shaded.org.apache.kafka.common.security.scram.ScramLoginModule required username="${KAFKA_SASL_USERNAME}" password="${KAFKA_SASL_PASSWORD}";',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601'
);

-- =============================================================================
-- View: Test File Correlations (30-day window)
-- Join test executions with file changes from the same commit
-- =============================================================================
CREATE TEMPORARY VIEW test_file_correlations AS
SELECT
    te.org_id,
    te.project_id,
    te.test_id,
    te.test_name,
    te.test_file_path,
    cf.file_path AS changed_file,
    cf.change_type,
    te.status,
    te.duration_ms,
    te.error_type,
    te.`timestamp` AS executed_at,
    te.commit_sha
FROM test_executions_batch te
JOIN commit_files_batch cf
    ON te.commit_sha = cf.commit_sha
    AND te.org_id = cf.org_id
    AND te.project_id = cf.project_id
WHERE te.`timestamp` > CURRENT_TIMESTAMP - INTERVAL '30' DAY
  AND te.org_id IS NOT NULL
  AND te.project_id IS NOT NULL
  AND te.test_name IS NOT NULL
  AND cf.file_path IS NOT NULL
  -- Exclude test files from changed files (self-reference)
  AND cf.file_path <> COALESCE(te.test_file_path, '');

-- =============================================================================
-- View: File-Test Impact Scores
-- Compute correlation scores based on co-occurrence and failure rates
-- =============================================================================
CREATE TEMPORARY VIEW file_test_impact AS
SELECT
    org_id,
    project_id,
    changed_file,
    test_id,
    test_name,
    test_file_path,
    -- Co-occurrence statistics
    COUNT(*) AS co_occurrence_count,
    COUNT(DISTINCT commit_sha) AS distinct_commits,
    -- Failure analysis
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failure_count,
    SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) AS pass_count,
    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count,
    -- Failure rate (0.0 to 1.0)
    CAST(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS DOUBLE) /
        NULLIF(COUNT(*), 0) AS failure_rate,
    -- Impact score: weighted combination of frequency and failure correlation
    -- High score = file changes frequently cause this test to fail
    CAST(
        (0.4 * LEAST(COUNT(*) / 10.0, 1.0)) +  -- Frequency weight (capped at 10 occurrences)
        (0.6 * (CAST(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS DOUBLE) /
                NULLIF(COUNT(*), 0)))          -- Failure correlation weight
    AS DOUBLE) AS impact_score,
    -- Duration analysis for performance impact
    AVG(CAST(duration_ms AS DOUBLE)) AS avg_duration_ms,
    MAX(duration_ms) AS max_duration_ms,
    -- Last observed
    MAX(executed_at) AS last_seen_at
FROM test_file_correlations
GROUP BY org_id, project_id, changed_file, test_id, test_name, test_file_path
-- Minimum 3 co-occurrences for statistical confidence
HAVING COUNT(*) >= 3;

-- =============================================================================
-- View: Aggregated Impact Matrix per Project
-- Group all file-test mappings into a JSON structure
-- =============================================================================
CREATE TEMPORARY VIEW project_impact_matrix AS
SELECT
    org_id,
    project_id,
    -- File to tests mapping (primary output)
    JSON_OBJECT(
        'file_to_tests' VALUE JSON_OBJECTAGG(
            changed_file VALUE JSON_OBJECT(
                'tests' VALUE LISTAGG(test_name, ','),
                'impact_scores' VALUE LISTAGG(CAST(ROUND(impact_score * 100) / 100 AS STRING), ','),
                'failure_rates' VALUE LISTAGG(CAST(ROUND(failure_rate * 100) / 100 AS STRING), ',')
            )
        ),
        'test_to_files' VALUE JSON_OBJECTAGG(
            test_name VALUE JSON_OBJECT(
                'files' VALUE LISTAGG(changed_file, ','),
                'total_impact' VALUE SUM(impact_score)
            )
        ),
        'statistics' VALUE JSON_OBJECT(
            'total_files' VALUE COUNT(DISTINCT changed_file),
            'total_tests' VALUE COUNT(DISTINCT test_name),
            'total_mappings' VALUE COUNT(*),
            'avg_impact_score' VALUE AVG(impact_score),
            'high_impact_mappings' VALUE SUM(CASE WHEN impact_score > 0.7 THEN 1 ELSE 0 END)
        ),
        'computed_at' VALUE CAST(CURRENT_TIMESTAMP AS STRING),
        'lookback_days' VALUE 30
    ) AS impact_matrix
FROM file_test_impact
GROUP BY org_id, project_id;

-- =============================================================================
-- Job: Write Precomputed Test Impact Matrix
-- Upserts into intelligence_precomputed via Kafka -> Supabase sink
-- =============================================================================
INSERT INTO precomputed_results
SELECT
    org_id,
    project_id,
    'test_impact_matrix' AS computation_type,
    CAST(impact_matrix AS STRING) AS result,
    CURRENT_TIMESTAMP AS computed_at,
    CURRENT_TIMESTAMP + INTERVAL '24' HOUR AS valid_until
FROM project_impact_matrix;

-- =============================================================================
-- Additional Computation: High-Risk Files
-- Files that frequently cause test failures
-- =============================================================================
CREATE TEMPORARY VIEW high_risk_files AS
SELECT
    org_id,
    project_id,
    changed_file,
    COUNT(DISTINCT test_name) AS affected_tests_count,
    AVG(failure_rate) AS avg_failure_rate,
    SUM(failure_count) AS total_failures,
    MAX(impact_score) AS max_impact_score
FROM file_test_impact
GROUP BY org_id, project_id, changed_file
HAVING AVG(failure_rate) > 0.3 OR SUM(failure_count) > 10;

-- Write high-risk files as separate precomputed result
INSERT INTO precomputed_results
SELECT
    org_id,
    project_id,
    'high_risk_files' AS computation_type,
    JSON_OBJECT(
        'files' VALUE JSON_ARRAYAGG(
            JSON_OBJECT(
                'path' VALUE changed_file,
                'affected_tests' VALUE affected_tests_count,
                'avg_failure_rate' VALUE ROUND(avg_failure_rate * 100) / 100,
                'total_failures' VALUE total_failures,
                'max_impact_score' VALUE ROUND(max_impact_score * 100) / 100
            )
        ),
        'computed_at' VALUE CAST(CURRENT_TIMESTAMP AS STRING)
    ) AS result,
    CURRENT_TIMESTAMP AS computed_at,
    CURRENT_TIMESTAMP + INTERVAL '24' HOUR AS valid_until
FROM high_risk_files
GROUP BY org_id, project_id;

-- =============================================================================
-- Additional Computation: Flaky Test Candidates
-- Tests with inconsistent pass/fail patterns on same files
-- =============================================================================
CREATE TEMPORARY VIEW flaky_candidates AS
SELECT
    org_id,
    project_id,
    test_id,
    test_name,
    test_file_path,
    COUNT(DISTINCT changed_file) AS unique_file_triggers,
    SUM(failure_count) AS total_failures,
    SUM(pass_count) AS total_passes,
    -- Flakiness score: higher when pass/fail ratio is closer to 50/50
    1.0 - ABS(
        (CAST(SUM(pass_count) AS DOUBLE) / NULLIF(SUM(pass_count) + SUM(failure_count), 0)) - 0.5
    ) * 2 AS flakiness_score,
    AVG(avg_duration_ms) AS avg_duration_ms
FROM file_test_impact
GROUP BY org_id, project_id, test_id, test_name, test_file_path
HAVING SUM(failure_count) >= 2
   AND SUM(pass_count) >= 2
   AND (1.0 - ABS((CAST(SUM(pass_count) AS DOUBLE) / NULLIF(SUM(pass_count) + SUM(failure_count), 0)) - 0.5) * 2) > 0.5;

-- Write flaky candidates as separate precomputed result
INSERT INTO precomputed_results
SELECT
    org_id,
    project_id,
    'flaky_test_candidates' AS computation_type,
    JSON_OBJECT(
        'tests' VALUE JSON_ARRAYAGG(
            JSON_OBJECT(
                'test_id' VALUE test_id,
                'test_name' VALUE test_name,
                'test_file' VALUE test_file_path,
                'flakiness_score' VALUE ROUND(flakiness_score * 100) / 100,
                'failures' VALUE total_failures,
                'passes' VALUE total_passes,
                'avg_duration_ms' VALUE ROUND(avg_duration_ms)
            )
        ),
        'computed_at' VALUE CAST(CURRENT_TIMESTAMP AS STRING)
    ) AS result,
    CURRENT_TIMESTAMP AS computed_at,
    CURRENT_TIMESTAMP + INTERVAL '24' HOUR AS valid_until
FROM flaky_candidates
GROUP BY org_id, project_id;
