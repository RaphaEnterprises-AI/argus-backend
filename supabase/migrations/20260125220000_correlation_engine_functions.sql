-- ============================================================================
-- Correlation Engine Functions and Indexes
-- Additional database functions to support the cross-correlation engine
-- ============================================================================

-- =============================================================================
-- Additional Indexes for Correlation Performance
-- =============================================================================

-- Composite index for time-window queries with event type filtering
CREATE INDEX IF NOT EXISTS idx_sdlc_events_project_type_time
    ON sdlc_events(project_id, event_type, occurred_at DESC);

-- Index for finding events within a time range efficiently
CREATE INDEX IF NOT EXISTS idx_sdlc_events_project_time_range
    ON sdlc_events(project_id, occurred_at)
    WHERE occurred_at > NOW() - INTERVAL '90 days';

-- Index for insight queries by project and status
CREATE INDEX IF NOT EXISTS idx_correlation_insights_project_status_severity
    ON correlation_insights(project_id, status, severity);

-- =============================================================================
-- Function: Get Events by Correlation Key
-- Returns all events matching a specific correlation key within a project
-- =============================================================================

CREATE OR REPLACE FUNCTION get_events_by_correlation_key(
    p_project_id UUID,
    p_key_type TEXT,
    p_key_value TEXT,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    id UUID,
    event_type TEXT,
    source_platform TEXT,
    external_id TEXT,
    external_url TEXT,
    title TEXT,
    occurred_at TIMESTAMPTZ,
    commit_sha TEXT,
    pr_number INTEGER,
    jira_key TEXT,
    deploy_id TEXT,
    branch_name TEXT,
    data JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.event_type,
        e.source_platform,
        e.external_id,
        e.external_url,
        e.title,
        e.occurred_at,
        e.commit_sha,
        e.pr_number,
        e.jira_key,
        e.deploy_id,
        e.branch_name,
        e.data
    FROM sdlc_events e
    WHERE e.project_id = p_project_id
      AND (
          (p_key_type = 'commit_sha' AND e.commit_sha = p_key_value) OR
          (p_key_type = 'pr_number' AND e.pr_number = p_key_value::INTEGER) OR
          (p_key_type = 'jira_key' AND e.jira_key = p_key_value) OR
          (p_key_type = 'deploy_id' AND e.deploy_id = p_key_value) OR
          (p_key_type = 'branch_name' AND e.branch_name = p_key_value)
      )
    ORDER BY e.occurred_at ASC
    LIMIT p_limit;
END;
$$;

-- =============================================================================
-- Function: Detect Failure Clusters
-- Finds groups of errors/incidents that share common correlation keys
-- =============================================================================

CREATE OR REPLACE FUNCTION detect_failure_clusters(
    p_project_id UUID,
    p_days INTEGER DEFAULT 7,
    p_min_cluster_size INTEGER DEFAULT 3
)
RETURNS TABLE (
    key_type TEXT,
    key_value TEXT,
    event_count BIGINT,
    event_ids UUID[],
    event_types TEXT[],
    first_occurrence TIMESTAMPTZ,
    last_occurrence TIMESTAMPTZ,
    time_span_hours NUMERIC
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_since_date TIMESTAMPTZ := NOW() - (p_days || ' days')::INTERVAL;
BEGIN
    -- Find clusters by commit_sha
    RETURN QUERY
    SELECT
        'commit_sha'::TEXT AS key_type,
        e.commit_sha AS key_value,
        COUNT(*)::BIGINT AS event_count,
        ARRAY_AGG(e.id) AS event_ids,
        ARRAY_AGG(DISTINCT e.event_type) AS event_types,
        MIN(e.occurred_at) AS first_occurrence,
        MAX(e.occurred_at) AS last_occurrence,
        EXTRACT(EPOCH FROM (MAX(e.occurred_at) - MIN(e.occurred_at))) / 3600 AS time_span_hours
    FROM sdlc_events e
    WHERE e.project_id = p_project_id
      AND e.event_type IN ('error', 'incident')
      AND e.occurred_at >= v_since_date
      AND e.commit_sha IS NOT NULL
    GROUP BY e.commit_sha
    HAVING COUNT(*) >= p_min_cluster_size

    UNION ALL

    -- Find clusters by deploy_id
    SELECT
        'deploy_id'::TEXT AS key_type,
        e.deploy_id AS key_value,
        COUNT(*)::BIGINT AS event_count,
        ARRAY_AGG(e.id) AS event_ids,
        ARRAY_AGG(DISTINCT e.event_type) AS event_types,
        MIN(e.occurred_at) AS first_occurrence,
        MAX(e.occurred_at) AS last_occurrence,
        EXTRACT(EPOCH FROM (MAX(e.occurred_at) - MIN(e.occurred_at))) / 3600 AS time_span_hours
    FROM sdlc_events e
    WHERE e.project_id = p_project_id
      AND e.event_type IN ('error', 'incident')
      AND e.occurred_at >= v_since_date
      AND e.deploy_id IS NOT NULL
    GROUP BY e.deploy_id
    HAVING COUNT(*) >= p_min_cluster_size

    UNION ALL

    -- Find clusters by pr_number
    SELECT
        'pr_number'::TEXT AS key_type,
        e.pr_number::TEXT AS key_value,
        COUNT(*)::BIGINT AS event_count,
        ARRAY_AGG(e.id) AS event_ids,
        ARRAY_AGG(DISTINCT e.event_type) AS event_types,
        MIN(e.occurred_at) AS first_occurrence,
        MAX(e.occurred_at) AS last_occurrence,
        EXTRACT(EPOCH FROM (MAX(e.occurred_at) - MIN(e.occurred_at))) / 3600 AS time_span_hours
    FROM sdlc_events e
    WHERE e.project_id = p_project_id
      AND e.event_type IN ('error', 'incident')
      AND e.occurred_at >= v_since_date
      AND e.pr_number IS NOT NULL
    GROUP BY e.pr_number
    HAVING COUNT(*) >= p_min_cluster_size

    ORDER BY event_count DESC;
END;
$$;

-- =============================================================================
-- Function: Get SDLC Event Summary
-- Returns aggregated statistics about events in a time window
-- =============================================================================

CREATE OR REPLACE FUNCTION get_sdlc_event_summary(
    p_project_id UUID,
    p_days INTEGER DEFAULT 7
)
RETURNS TABLE (
    total_events BIGINT,
    events_by_type JSONB,
    events_by_platform JSONB,
    errors_by_day JSONB,
    deployments_count BIGINT,
    errors_count BIGINT,
    incidents_count BIGINT,
    tests_count BIGINT,
    commits_count BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_since_date TIMESTAMPTZ := NOW() - (p_days || ' days')::INTERVAL;
BEGIN
    RETURN QUERY
    WITH event_stats AS (
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE event_type = 'deploy' OR event_type = 'deployment_status') AS deploys,
            COUNT(*) FILTER (WHERE event_type = 'error') AS errors,
            COUNT(*) FILTER (WHERE event_type = 'incident') AS incidents,
            COUNT(*) FILTER (WHERE event_type = 'test_run') AS tests,
            COUNT(*) FILTER (WHERE event_type = 'commit') AS commits
        FROM sdlc_events
        WHERE project_id = p_project_id
          AND occurred_at >= v_since_date
    ),
    by_type AS (
        SELECT jsonb_object_agg(event_type, cnt) AS type_counts
        FROM (
            SELECT event_type, COUNT(*) AS cnt
            FROM sdlc_events
            WHERE project_id = p_project_id
              AND occurred_at >= v_since_date
            GROUP BY event_type
        ) t
    ),
    by_platform AS (
        SELECT jsonb_object_agg(source_platform, cnt) AS platform_counts
        FROM (
            SELECT source_platform, COUNT(*) AS cnt
            FROM sdlc_events
            WHERE project_id = p_project_id
              AND occurred_at >= v_since_date
            GROUP BY source_platform
        ) t
    ),
    errors_daily AS (
        SELECT jsonb_object_agg(day, cnt) AS daily_errors
        FROM (
            SELECT
                TO_CHAR(occurred_at, 'YYYY-MM-DD') AS day,
                COUNT(*) AS cnt
            FROM sdlc_events
            WHERE project_id = p_project_id
              AND event_type IN ('error', 'incident')
              AND occurred_at >= v_since_date
            GROUP BY TO_CHAR(occurred_at, 'YYYY-MM-DD')
        ) t
    )
    SELECT
        es.total,
        COALESCE(bt.type_counts, '{}'::JSONB),
        COALESCE(bp.platform_counts, '{}'::JSONB),
        COALESCE(ed.daily_errors, '{}'::JSONB),
        es.deploys,
        es.errors,
        es.incidents,
        es.tests,
        es.commits
    FROM event_stats es
    CROSS JOIN by_type bt
    CROSS JOIN by_platform bp
    CROSS JOIN errors_daily ed;
END;
$$;

-- =============================================================================
-- Function: Find Related Deployments for Errors
-- Finds deployments that may have caused errors within a time window
-- =============================================================================

CREATE OR REPLACE FUNCTION find_error_causing_deployments(
    p_project_id UUID,
    p_days INTEGER DEFAULT 30,
    p_hours_after_deploy INTEGER DEFAULT 24
)
RETURNS TABLE (
    deploy_id UUID,
    deploy_title TEXT,
    deploy_commit_sha TEXT,
    deploy_time TIMESTAMPTZ,
    error_count BIGINT,
    error_ids UUID[],
    time_to_first_error_hours NUMERIC
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_since_date TIMESTAMPTZ := NOW() - (p_days || ' days')::INTERVAL;
BEGIN
    RETURN QUERY
    WITH deployments AS (
        SELECT
            d.id AS deploy_id,
            d.title AS deploy_title,
            d.commit_sha AS deploy_commit_sha,
            d.occurred_at AS deploy_time
        FROM sdlc_events d
        WHERE d.project_id = p_project_id
          AND d.event_type IN ('deploy', 'deployment_status')
          AND d.occurred_at >= v_since_date
    ),
    deploy_errors AS (
        SELECT
            d.deploy_id,
            d.deploy_title,
            d.deploy_commit_sha,
            d.deploy_time,
            COUNT(e.id) AS error_count,
            ARRAY_AGG(e.id) AS error_ids,
            MIN(e.occurred_at) AS first_error_time
        FROM deployments d
        LEFT JOIN sdlc_events e
            ON e.project_id = p_project_id
            AND e.event_type IN ('error', 'incident')
            AND e.occurred_at > d.deploy_time
            AND e.occurred_at <= d.deploy_time + (p_hours_after_deploy || ' hours')::INTERVAL
            AND (
                -- Match by commit SHA
                (d.deploy_commit_sha IS NOT NULL AND e.commit_sha = d.deploy_commit_sha)
                -- Or match by deploy ID
                OR (e.deploy_id IS NOT NULL AND e.deploy_id = d.deploy_id::TEXT)
            )
        GROUP BY d.deploy_id, d.deploy_title, d.deploy_commit_sha, d.deploy_time
        HAVING COUNT(e.id) > 0
    )
    SELECT
        de.deploy_id,
        de.deploy_title,
        de.deploy_commit_sha,
        de.deploy_time,
        de.error_count,
        de.error_ids,
        EXTRACT(EPOCH FROM (de.first_error_time - de.deploy_time)) / 3600 AS time_to_first_error_hours
    FROM deploy_errors de
    ORDER BY de.error_count DESC, de.deploy_time DESC;
END;
$$;

-- =============================================================================
-- Function: Calculate Correlation Confidence
-- Calculates correlation score between two events
-- =============================================================================

CREATE OR REPLACE FUNCTION calculate_event_correlation(
    p_source_event_id UUID,
    p_target_event_id UUID,
    p_max_hours INTEGER DEFAULT 48
)
RETURNS TABLE (
    correlation_type TEXT,
    confidence NUMERIC,
    shared_keys JSONB,
    time_diff_hours NUMERIC
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_source RECORD;
    v_target RECORD;
    v_time_score NUMERIC := 0;
    v_key_score NUMERIC := 0;
    v_shared_keys JSONB := '[]'::JSONB;
    v_keys_found JSONB := '[]'::JSONB;
BEGIN
    -- Get source event
    SELECT * INTO v_source FROM sdlc_events WHERE id = p_source_event_id;
    SELECT * INTO v_target FROM sdlc_events WHERE id = p_target_event_id;

    IF v_source IS NULL OR v_target IS NULL THEN
        RETURN QUERY SELECT NULL::TEXT, 0::NUMERIC, '[]'::JSONB, NULL::NUMERIC;
        RETURN;
    END IF;

    -- Calculate time difference
    DECLARE
        v_time_diff NUMERIC := EXTRACT(EPOCH FROM (v_target.occurred_at - v_source.occurred_at)) / 3600;
    BEGIN
        -- Time proximity score (max 0.3)
        IF v_time_diff >= 0 AND v_time_diff <= p_max_hours THEN
            v_time_score := (1 - (v_time_diff / p_max_hours)) * 0.3;
        END IF;

        -- Key overlap score (max 0.5)
        IF v_source.commit_sha IS NOT NULL AND v_source.commit_sha = v_target.commit_sha THEN
            v_key_score := v_key_score + 0.5;
            v_keys_found := v_keys_found || jsonb_build_object('type', 'commit_sha', 'value', v_source.commit_sha);
        END IF;

        IF v_source.pr_number IS NOT NULL AND v_source.pr_number = v_target.pr_number THEN
            v_key_score := v_key_score + 0.4;
            v_keys_found := v_keys_found || jsonb_build_object('type', 'pr_number', 'value', v_source.pr_number);
        END IF;

        IF v_source.deploy_id IS NOT NULL AND v_source.deploy_id = v_target.deploy_id THEN
            v_key_score := v_key_score + 0.4;
            v_keys_found := v_keys_found || jsonb_build_object('type', 'deploy_id', 'value', v_source.deploy_id);
        END IF;

        IF v_source.jira_key IS NOT NULL AND v_source.jira_key = v_target.jira_key THEN
            v_key_score := v_key_score + 0.3;
            v_keys_found := v_keys_found || jsonb_build_object('type', 'jira_key', 'value', v_source.jira_key);
        END IF;

        IF v_source.branch_name IS NOT NULL AND v_source.branch_name = v_target.branch_name THEN
            v_key_score := v_key_score + 0.2;
            v_keys_found := v_keys_found || jsonb_build_object('type', 'branch_name', 'value', v_source.branch_name);
        END IF;

        -- Cap key score at 0.5
        v_key_score := LEAST(v_key_score, 0.5);

        -- Determine correlation type based on event types
        DECLARE
            v_corr_type TEXT := 'related_to';
        BEGIN
            IF v_source.event_type IN ('commit', 'deploy') AND v_target.event_type IN ('error', 'incident') THEN
                IF v_source.event_type = 'deploy' THEN
                    v_corr_type := 'caused_by';
                ELSE
                    v_corr_type := 'introduced_by';
                END IF;
            END IF;

            RETURN QUERY SELECT
                v_corr_type,
                LEAST(1.0, v_time_score + v_key_score)::NUMERIC,
                v_keys_found,
                v_time_diff::NUMERIC;
        END;
    END;
END;
$$;

-- =============================================================================
-- Grant Permissions
-- =============================================================================

GRANT EXECUTE ON FUNCTION get_events_by_correlation_key(UUID, TEXT, TEXT, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION detect_failure_clusters(UUID, INTEGER, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION get_sdlc_event_summary(UUID, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION find_error_causing_deployments(UUID, INTEGER, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION calculate_event_correlation(UUID, UUID, INTEGER) TO service_role;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON FUNCTION get_events_by_correlation_key IS 'Retrieves all SDLC events matching a specific correlation key (commit_sha, pr_number, etc.)';
COMMENT ON FUNCTION detect_failure_clusters IS 'Detects clusters of errors/incidents that share common correlation keys';
COMMENT ON FUNCTION get_sdlc_event_summary IS 'Returns aggregated statistics about SDLC events in a time window';
COMMENT ON FUNCTION find_error_causing_deployments IS 'Finds deployments that may have caused errors within a time window';
COMMENT ON FUNCTION calculate_event_correlation IS 'Calculates the correlation score between two events based on shared keys and time proximity';
