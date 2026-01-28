-- =============================================================================
-- Intelligence Cache Stats Migration
-- =============================================================================
--
-- Tracks cache effectiveness for the Intelligence Layer (UIIL).
-- Used for monitoring dashboard and optimizing cache TTLs.
--
-- This table uses TimescaleDB hypertable for efficient time-series storage
-- if TimescaleDB extension is available, otherwise falls back to standard table.
-- =============================================================================

-- Create cache stats table
CREATE TABLE IF NOT EXISTS intelligence_cache_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id TEXT NOT NULL,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,

    -- Query details
    query_intent TEXT NOT NULL,  -- SIMILAR_ERRORS, TEST_IMPACT, ROOT_CAUSE, etc.
    query_hash TEXT,  -- Hash of the query for grouping similar queries

    -- Resolution details
    cache_hit BOOLEAN NOT NULL,
    resolution_tier TEXT NOT NULL,  -- 'cache', 'precomputed', 'vector', 'llm'
    confidence_score DECIMAL(4, 3),  -- 0.000 to 1.000

    -- Performance metrics
    latency_ms INTEGER NOT NULL,
    tokens_used INTEGER DEFAULT 0,  -- Only for LLM tier

    -- Metadata
    source TEXT NOT NULL DEFAULT 'api',  -- 'api', 'worker', 'scheduled'
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create indexes for common queries
CREATE INDEX idx_cache_stats_org_time
    ON intelligence_cache_stats(org_id, created_at DESC);

CREATE INDEX idx_cache_stats_intent_time
    ON intelligence_cache_stats(query_intent, created_at DESC);

CREATE INDEX idx_cache_stats_tier_time
    ON intelligence_cache_stats(resolution_tier, created_at DESC);

-- Partial index for cache misses (for debugging)
CREATE INDEX idx_cache_stats_misses
    ON intelligence_cache_stats(org_id, query_intent, created_at DESC)
    WHERE cache_hit = false;

-- =============================================================================
-- Aggregation Views for Dashboard
-- =============================================================================

-- Hourly cache hit rate by intent
CREATE OR REPLACE VIEW intelligence_cache_hourly_stats AS
SELECT
    date_trunc('hour', created_at) AS hour,
    org_id,
    query_intent,
    COUNT(*) AS total_queries,
    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) AS cache_hits,
    ROUND(
        SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)::DECIMAL / NULLIF(COUNT(*), 0) * 100,
        2
    ) AS hit_rate_pct,
    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
    SUM(tokens_used) AS total_tokens
FROM intelligence_cache_stats
WHERE created_at > now() - INTERVAL '24 hours'
GROUP BY date_trunc('hour', created_at), org_id, query_intent
ORDER BY hour DESC;

-- Resolution tier distribution
CREATE OR REPLACE VIEW intelligence_tier_distribution AS
SELECT
    org_id,
    resolution_tier,
    COUNT(*) AS query_count,
    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
    ROUND(AVG(confidence_score), 3) AS avg_confidence,
    ROUND(
        COUNT(*)::DECIMAL / SUM(COUNT(*)) OVER (PARTITION BY org_id) * 100,
        2
    ) AS tier_pct
FROM intelligence_cache_stats
WHERE created_at > now() - INTERVAL '1 hour'
GROUP BY org_id, resolution_tier
ORDER BY org_id, tier_pct DESC;

-- LLM fallback analysis (why are we falling back?)
CREATE OR REPLACE VIEW intelligence_llm_fallback_analysis AS
SELECT
    org_id,
    query_intent,
    COUNT(*) AS fallback_count,
    ROUND(AVG(confidence_score), 3) AS avg_confidence_before_fallback,
    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
    SUM(tokens_used) AS total_tokens
FROM intelligence_cache_stats
WHERE
    resolution_tier = 'llm'
    AND created_at > now() - INTERVAL '24 hours'
GROUP BY org_id, query_intent
ORDER BY fallback_count DESC;

-- =============================================================================
-- Functions for Cache Analytics
-- =============================================================================

-- Get cache hit rate for an org over a time period
CREATE OR REPLACE FUNCTION get_cache_hit_rate(
    p_org_id TEXT,
    p_hours INTEGER DEFAULT 24
) RETURNS TABLE (
    total_queries BIGINT,
    cache_hits BIGINT,
    hit_rate_pct DECIMAL,
    avg_latency_ms DECIMAL,
    llm_fallback_rate_pct DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT AS total_queries,
        SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)::BIGINT AS cache_hits,
        ROUND(
            SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)::DECIMAL / NULLIF(COUNT(*), 0) * 100,
            2
        ) AS hit_rate_pct,
        ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
        ROUND(
            SUM(CASE WHEN resolution_tier = 'llm' THEN 1 ELSE 0 END)::DECIMAL / NULLIF(COUNT(*), 0) * 100,
            2
        ) AS llm_fallback_rate_pct
    FROM intelligence_cache_stats
    WHERE
        org_id = p_org_id
        AND created_at > now() - (p_hours || ' hours')::INTERVAL;
END;
$$ LANGUAGE plpgsql STABLE;

-- Record a cache stat entry (called from Python)
CREATE OR REPLACE FUNCTION record_cache_stat(
    p_org_id TEXT,
    p_project_id UUID,
    p_query_intent TEXT,
    p_query_hash TEXT,
    p_cache_hit BOOLEAN,
    p_resolution_tier TEXT,
    p_confidence_score DECIMAL,
    p_latency_ms INTEGER,
    p_tokens_used INTEGER DEFAULT 0,
    p_source TEXT DEFAULT 'api'
) RETURNS UUID AS $$
DECLARE
    v_id UUID;
BEGIN
    INSERT INTO intelligence_cache_stats (
        org_id, project_id, query_intent, query_hash,
        cache_hit, resolution_tier, confidence_score,
        latency_ms, tokens_used, source
    ) VALUES (
        p_org_id, p_project_id, p_query_intent, p_query_hash,
        p_cache_hit, p_resolution_tier, p_confidence_score,
        p_latency_ms, p_tokens_used, p_source
    )
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Row Level Security
-- =============================================================================
ALTER TABLE intelligence_cache_stats ENABLE ROW LEVEL SECURITY;

-- Org isolation policy
CREATE POLICY intelligence_cache_stats_org_isolation
    ON intelligence_cache_stats
    FOR ALL
    USING (org_id = current_setting('app.current_org_id', true));

-- Service role can access all
CREATE POLICY intelligence_cache_stats_service_access
    ON intelligence_cache_stats
    FOR ALL
    TO service_role
    USING (true);

-- =============================================================================
-- Cleanup Job (run weekly to remove old stats)
-- =============================================================================

-- Function to clean up old cache stats (keep last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_cache_stats() RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM intelligence_cache_stats
    WHERE created_at < now() - INTERVAL '30 days';

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RAISE NOTICE 'Deleted % old cache stat records', deleted_count;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Schedule cleanup (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-cache-stats', '0 3 * * 0', 'SELECT cleanup_old_cache_stats()');

-- =============================================================================
-- TimescaleDB Optimization (if extension is available)
-- =============================================================================

-- Uncomment if using TimescaleDB for better time-series performance:
-- SELECT create_hypertable('intelligence_cache_stats', 'created_at', if_not_exists => TRUE);
-- SELECT add_retention_policy('intelligence_cache_stats', INTERVAL '30 days');

COMMENT ON TABLE intelligence_cache_stats IS
    'Tracks cache effectiveness for the Intelligence Layer (UIIL). Used for monitoring and optimization.';
