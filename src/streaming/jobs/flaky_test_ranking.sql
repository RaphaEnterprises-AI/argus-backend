-- =============================================================================
-- Flink SQL: Flaky Test Ranking Job
-- =============================================================================
--
-- Computes weekly flaky test rankings based on pass/fail patterns.
-- A test is considered flaky if it has inconsistent results (both pass and fail)
-- within a short time window without code changes.
--
-- Flakiness Score Formula:
--   flakiness = (flip_count * 2 + fail_count) / total_runs
--   where flip_count = number of status changes (pass→fail or fail→pass)
--
-- Schedule: Weekly (Sundays at 4 AM UTC)
-- Output: intelligence_precomputed table with type='flaky_ranking'
--
-- Deploy with:
--   kubectl apply -f data-layer/kubernetes/jobs/flaky-test-ranking.yaml
-- =============================================================================

-- Source: Test execution events from Redpanda
CREATE TABLE test_executions (
    event_id STRING,
    org_id STRING,
    project_id STRING,
    test_id STRING,
    test_name STRING,
    test_file STRING,
    status STRING,  -- 'passed', 'failed', 'skipped'
    duration_ms BIGINT,
    error_message STRING,
    commit_sha STRING,
    branch STRING,
    executed_at TIMESTAMP(3),
    event_time AS executed_at,
    WATERMARK FOR event_time AS event_time - INTERVAL '5' MINUTE
) WITH (
    'connector' = 'kafka',
    'topic' = 'argus.test.executed',
    'properties.bootstrap.servers' = '${REDPANDA_BROKERS}',
    'properties.group.id' = 'flink-flaky-ranking',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601'
);

-- Sink: Supabase intelligence_precomputed table
CREATE TABLE intelligence_precomputed_sink (
    id STRING,
    org_id STRING,
    project_id STRING,
    computation_type STRING,
    result STRING,  -- JSON
    computed_at TIMESTAMP(3),
    valid_until TIMESTAMP(3),
    created_by STRING,
    PRIMARY KEY (org_id, project_id, computation_type) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = '${SUPABASE_JDBC_URL}',
    'table-name' = 'intelligence_precomputed',
    'driver' = 'org.postgresql.Driver',
    'username' = '${SUPABASE_USER}',
    'password' = '${SUPABASE_PASSWORD}'
);

-- =============================================================================
-- Step 1: Calculate per-test statistics over the last 7 days
-- =============================================================================
CREATE VIEW test_weekly_stats AS
SELECT
    org_id,
    project_id,
    test_id,
    MAX(test_name) AS test_name,
    MAX(test_file) AS test_file,
    COUNT(*) AS total_runs,
    SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) AS pass_count,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS fail_count,
    SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) AS skip_count,
    AVG(duration_ms) AS avg_duration_ms,
    MAX(duration_ms) AS max_duration_ms,
    MIN(duration_ms) AS min_duration_ms,
    STDDEV_POP(duration_ms) AS stddev_duration_ms,
    -- Collect recent errors for debugging
    LISTAGG(DISTINCT error_message, ' | ') AS recent_errors,
    -- Track unique commits to check if failures correlate with code changes
    COUNT(DISTINCT commit_sha) AS unique_commits,
    MAX(executed_at) AS last_execution
FROM test_executions
WHERE
    executed_at > CURRENT_TIMESTAMP - INTERVAL '7' DAY
    AND status IN ('passed', 'failed')  -- Exclude skipped for flakiness calc
GROUP BY org_id, project_id, test_id;

-- =============================================================================
-- Step 2: Calculate status flips (consecutive pass/fail changes)
-- This requires a stateful window to track previous status
-- =============================================================================
CREATE VIEW test_status_changes AS
SELECT
    org_id,
    project_id,
    test_id,
    COUNT(*) AS flip_count
FROM (
    SELECT
        org_id,
        project_id,
        test_id,
        status,
        LAG(status) OVER (PARTITION BY org_id, project_id, test_id ORDER BY executed_at) AS prev_status
    FROM test_executions
    WHERE
        executed_at > CURRENT_TIMESTAMP - INTERVAL '7' DAY
        AND status IN ('passed', 'failed')
) changes
WHERE status != prev_status AND prev_status IS NOT NULL
GROUP BY org_id, project_id, test_id;

-- =============================================================================
-- Step 3: Compute flakiness score and rank tests
-- =============================================================================
CREATE VIEW flaky_test_scores AS
SELECT
    s.org_id,
    s.project_id,
    s.test_id,
    s.test_name,
    s.test_file,
    s.total_runs,
    s.pass_count,
    s.fail_count,
    COALESCE(c.flip_count, 0) AS flip_count,
    s.avg_duration_ms,
    s.stddev_duration_ms,
    s.recent_errors,
    s.unique_commits,
    s.last_execution,
    -- Flakiness score: higher = more flaky
    -- Weight flips heavily (x2) since they indicate inconsistency
    CASE
        WHEN s.total_runs >= 3 THEN
            CAST((COALESCE(c.flip_count, 0) * 2.0 + s.fail_count) / s.total_runs AS DECIMAL(5, 3))
        ELSE 0.0
    END AS flakiness_score,
    -- Confidence: more runs = more confident in the score
    LEAST(s.total_runs / 10.0, 1.0) AS confidence,
    -- Flag: is this truly flaky (both pass and fail) vs consistently failing?
    CASE
        WHEN s.pass_count > 0 AND s.fail_count > 0 THEN 'FLAKY'
        WHEN s.fail_count > 0 AND s.pass_count = 0 THEN 'CONSISTENTLY_FAILING'
        ELSE 'STABLE'
    END AS classification
FROM test_weekly_stats s
LEFT JOIN test_status_changes c
    ON s.org_id = c.org_id
    AND s.project_id = c.project_id
    AND s.test_id = c.test_id;

-- =============================================================================
-- Step 4: Generate final ranking and insert into precomputed table
-- =============================================================================
INSERT INTO intelligence_precomputed_sink
SELECT
    -- Generate deterministic UUID from org+project+type
    MD5(CONCAT(org_id, ':', project_id, ':flaky_ranking')) AS id,
    org_id,
    project_id,
    'flaky_ranking' AS computation_type,
    -- JSON result with ranked flaky tests
    CONCAT(
        '{"computed_at":"', CAST(CURRENT_TIMESTAMP AS STRING), '",',
        '"period_days":7,',
        '"total_tests":', CAST(COUNT(*) AS STRING), ',',
        '"flaky_tests":', CAST(SUM(CASE WHEN classification = 'FLAKY' THEN 1 ELSE 0 END) AS STRING), ',',
        '"consistently_failing":', CAST(SUM(CASE WHEN classification = 'CONSISTENTLY_FAILING' THEN 1 ELSE 0 END) AS STRING), ',',
        '"rankings":[',
        LISTAGG(
            CONCAT(
                '{"test_id":"', test_id, '",',
                '"test_name":"', COALESCE(REPLACE(test_name, '"', '\\"'), ''), '",',
                '"test_file":"', COALESCE(REPLACE(test_file, '"', '\\"'), ''), '",',
                '"flakiness_score":', CAST(flakiness_score AS STRING), ',',
                '"confidence":', CAST(confidence AS STRING), ',',
                '"classification":"', classification, '",',
                '"total_runs":', CAST(total_runs AS STRING), ',',
                '"pass_count":', CAST(pass_count AS STRING), ',',
                '"fail_count":', CAST(fail_count AS STRING), ',',
                '"flip_count":', CAST(flip_count AS STRING), ',',
                '"avg_duration_ms":', CAST(COALESCE(avg_duration_ms, 0) AS STRING), ',',
                '"last_execution":"', CAST(last_execution AS STRING), '"}'
            ),
            ','
        ) FILTER (WHERE flakiness_score > 0 OR classification != 'STABLE'),
        ']}'
    ) AS result,
    CURRENT_TIMESTAMP AS computed_at,
    CURRENT_TIMESTAMP + INTERVAL '7' DAY AS valid_until,
    'flink' AS created_by
FROM flaky_test_scores
GROUP BY org_id, project_id;

-- =============================================================================
-- Example output structure:
-- {
--   "computed_at": "2026-01-29T04:00:00Z",
--   "period_days": 7,
--   "total_tests": 150,
--   "flaky_tests": 12,
--   "consistently_failing": 3,
--   "rankings": [
--     {
--       "test_id": "test-login-flow-123",
--       "test_name": "should login with valid credentials",
--       "test_file": "auth/login.spec.ts",
--       "flakiness_score": 0.45,
--       "confidence": 1.0,
--       "classification": "FLAKY",
--       "total_runs": 28,
--       "pass_count": 15,
--       "fail_count": 13,
--       "flip_count": 8,
--       "avg_duration_ms": 2340,
--       "last_execution": "2026-01-28T23:45:00Z"
--     },
--     ...
--   ]
-- }
-- =============================================================================
