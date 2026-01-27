-- =============================================================================
-- UBER/NETFLIX STYLE: Global State Schema for Argus
-- =============================================================================
-- This schema is designed for multi-region active-active deployment.
-- Key features:
--   1. Idempotency keys prevent duplicate processing across regions
--   2. Timestamps enable conflict resolution (last-write-wins)
--   3. Partitioning by org_id for multi-tenant isolation
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- =============================================================================
-- TEST METRICS (Aggregated by Flink)
-- =============================================================================
-- Flink jobs write windowed aggregations here.
-- Both regions can write; idempotency_key prevents duplicates.

CREATE TABLE IF NOT EXISTS flink_test_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Idempotency: unique per window + org + project
    -- Format: "{window_start_epoch}:{org_id}:{project_id}"
    idempotency_key TEXT UNIQUE NOT NULL,

    -- Dimensions
    org_id TEXT NOT NULL,
    project_id TEXT NOT NULL,

    -- Time window
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    window_size_minutes INTEGER NOT NULL DEFAULT 5,

    -- Metrics
    total_tests BIGINT NOT NULL DEFAULT 0,
    passed_tests BIGINT NOT NULL DEFAULT 0,
    failed_tests BIGINT NOT NULL DEFAULT 0,
    skipped_tests BIGINT NOT NULL DEFAULT 0,
    flaky_tests BIGINT NOT NULL DEFAULT 0,

    -- Durations (milliseconds)
    total_duration_ms BIGINT NOT NULL DEFAULT 0,
    avg_duration_ms DOUBLE PRECISION,
    p50_duration_ms BIGINT,
    p95_duration_ms BIGINT,
    p99_duration_ms BIGINT,

    -- Pass rate
    pass_rate DOUBLE PRECISION GENERATED ALWAYS AS (
        CASE WHEN total_tests > 0
        THEN passed_tests::DOUBLE PRECISION / total_tests
        ELSE 0 END
    ) STORED,

    -- Metadata
    processing_region TEXT,  -- Which region computed this
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Indexes for fast queries
    CONSTRAINT flink_test_metrics_org_project_window
        UNIQUE (org_id, project_id, window_start)
);

-- Indexes for dashboard queries
CREATE INDEX idx_flink_test_metrics_org_time
    ON flink_test_metrics (org_id, window_start DESC);
CREATE INDEX idx_flink_test_metrics_project_time
    ON flink_test_metrics (project_id, window_start DESC);

-- =============================================================================
-- FAILURE PATTERNS (For Self-Healing)
-- =============================================================================
-- Tracks recurring failures for auto-healing triggers.

CREATE TABLE IF NOT EXISTS flink_failure_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Idempotency: unique per window + test
    idempotency_key TEXT UNIQUE NOT NULL,

    -- Dimensions
    org_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    test_id TEXT NOT NULL,

    -- Time window
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,

    -- Failure info
    failure_count INTEGER NOT NULL DEFAULT 0,
    last_error_message TEXT,
    last_selector TEXT,
    error_fingerprint TEXT,  -- Hash of error for grouping

    -- Healing status
    healing_requested BOOLEAN DEFAULT FALSE,
    healing_priority TEXT CHECK (healing_priority IN ('low', 'medium', 'high', 'critical')),

    -- Metadata
    processing_region TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT flink_failure_patterns_unique
        UNIQUE (org_id, project_id, test_id, window_start)
);

CREATE INDEX idx_flink_failure_patterns_healing
    ON flink_failure_patterns (healing_requested, healing_priority)
    WHERE healing_requested = TRUE;

-- =============================================================================
-- REAL-TIME COUNTERS (For Live Dashboard)
-- =============================================================================
-- Ultra-fast counters updated by Flink, read by dashboard.
-- Uses INSERT ... ON CONFLICT for atomic increments.

CREATE TABLE IF NOT EXISTS flink_realtime_counters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Dimensions
    org_id TEXT NOT NULL,
    counter_type TEXT NOT NULL,  -- 'tests_today', 'failures_today', 'healing_today'
    counter_date DATE NOT NULL DEFAULT CURRENT_DATE,

    -- Counter value
    value BIGINT NOT NULL DEFAULT 0,

    -- Last update
    last_updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_updated_region TEXT,

    -- Unique constraint for upserts
    CONSTRAINT flink_realtime_counters_unique
        UNIQUE (org_id, counter_type, counter_date)
);

-- Function to atomically increment counter
CREATE OR REPLACE FUNCTION increment_counter(
    p_org_id TEXT,
    p_counter_type TEXT,
    p_increment BIGINT DEFAULT 1,
    p_region TEXT DEFAULT NULL
) RETURNS BIGINT AS $$
DECLARE
    v_new_value BIGINT;
BEGIN
    INSERT INTO flink_realtime_counters (org_id, counter_type, value, last_updated_region)
    VALUES (p_org_id, p_counter_type, p_increment, p_region)
    ON CONFLICT (org_id, counter_type, counter_date)
    DO UPDATE SET
        value = flink_realtime_counters.value + p_increment,
        last_updated_at = NOW(),
        last_updated_region = p_region
    RETURNING value INTO v_new_value;

    RETURN v_new_value;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- EVENT LOG (For Debugging & Replay)
-- =============================================================================
-- Optional: Store raw events for debugging/replay.
-- Partitioned by date for efficient cleanup.

CREATE TABLE IF NOT EXISTS flink_event_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Event identification
    event_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    idempotency_key TEXT UNIQUE NOT NULL,

    -- Event data
    org_id TEXT NOT NULL,
    project_id TEXT,
    payload JSONB NOT NULL,

    -- Processing info
    kafka_topic TEXT,
    kafka_partition INTEGER,
    kafka_offset BIGINT,
    processing_region TEXT,

    -- Timestamps
    event_timestamp TIMESTAMPTZ NOT NULL,
    processed_at TIMESTAMPTZ DEFAULT NOW()
) PARTITION BY RANGE (processed_at);

-- Create partitions for next 30 days
DO $$
DECLARE
    start_date DATE := CURRENT_DATE;
    end_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..30 LOOP
        end_date := start_date + 1;
        partition_name := 'flink_event_log_' || TO_CHAR(start_date, 'YYYY_MM_DD');

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF flink_event_log
             FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );

        start_date := end_date;
    END LOOP;
END $$;

-- =============================================================================
-- ROW LEVEL SECURITY (Multi-tenant Isolation)
-- =============================================================================

ALTER TABLE flink_test_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE flink_failure_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE flink_realtime_counters ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own org's data
CREATE POLICY org_isolation_metrics ON flink_test_metrics
    FOR ALL USING (org_id = current_setting('app.current_org_id', true));

CREATE POLICY org_isolation_failures ON flink_failure_patterns
    FOR ALL USING (org_id = current_setting('app.current_org_id', true));

CREATE POLICY org_isolation_counters ON flink_realtime_counters
    FOR ALL USING (org_id = current_setting('app.current_org_id', true));

-- Service role bypasses RLS (for Flink jobs)
CREATE POLICY service_full_access_metrics ON flink_test_metrics
    FOR ALL TO service_role USING (true);

CREATE POLICY service_full_access_failures ON flink_failure_patterns
    FOR ALL TO service_role USING (true);

CREATE POLICY service_full_access_counters ON flink_realtime_counters
    FOR ALL TO service_role USING (true);

-- =============================================================================
-- REAL-TIME SUBSCRIPTIONS (For Dashboard)
-- =============================================================================
-- Enable Supabase real-time for these tables

ALTER PUBLICATION supabase_realtime ADD TABLE flink_test_metrics;
ALTER PUBLICATION supabase_realtime ADD TABLE flink_failure_patterns;
ALTER PUBLICATION supabase_realtime ADD TABLE flink_realtime_counters;

-- =============================================================================
-- CLEANUP JOB (Run daily via pg_cron or external scheduler)
-- =============================================================================

CREATE OR REPLACE FUNCTION cleanup_old_flink_data() RETURNS void AS $$
BEGIN
    -- Keep 90 days of metrics
    DELETE FROM flink_test_metrics
    WHERE window_start < NOW() - INTERVAL '90 days';

    -- Keep 30 days of failure patterns
    DELETE FROM flink_failure_patterns
    WHERE window_start < NOW() - INTERVAL '30 days';

    -- Keep 7 days of event logs
    -- (Handled by partition dropping for efficiency)

    RAISE NOTICE 'Cleanup completed at %', NOW();
END;
$$ LANGUAGE plpgsql;
