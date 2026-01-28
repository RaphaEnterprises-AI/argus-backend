-- ============================================================================
-- Intelligence Precomputed Results Table
-- Stores expensive computations for instant lookup
-- ============================================================================

CREATE TABLE IF NOT EXISTS intelligence_precomputed (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id TEXT NOT NULL,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    computation_type TEXT NOT NULL,
    -- Types: test_impact_matrix, failure_clusters, flaky_ranking, coverage_gaps
    result JSONB NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_until TIMESTAMPTZ NOT NULL,
    created_by TEXT DEFAULT 'flink',

    CONSTRAINT intelligence_precomputed_unique
        UNIQUE(org_id, project_id, computation_type)
);

-- RLS
ALTER TABLE intelligence_precomputed ENABLE ROW LEVEL SECURITY;

-- Policy for org isolation
CREATE POLICY intelligence_precomputed_org_isolation ON intelligence_precomputed
    FOR ALL USING (org_id = current_setting('app.current_org_id', true));

-- Index for fast lookups (includes valid_until for filtering in queries)
CREATE INDEX idx_precomputed_lookup
    ON intelligence_precomputed(org_id, project_id, computation_type, valid_until DESC);

-- Function to get precomputed result
CREATE OR REPLACE FUNCTION get_precomputed(
    p_org_id TEXT,
    p_project_id UUID,
    p_type TEXT
) RETURNS JSONB AS $$
    SELECT result
    FROM intelligence_precomputed
    WHERE org_id = p_org_id
      AND project_id = p_project_id
      AND computation_type = p_type
      AND valid_until > now()
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- Function to upsert precomputed result
CREATE OR REPLACE FUNCTION upsert_precomputed(
    p_org_id TEXT,
    p_project_id UUID,
    p_type TEXT,
    p_result JSONB,
    p_valid_hours INT DEFAULT 24
) RETURNS UUID AS $$
DECLARE
    v_id UUID;
BEGIN
    INSERT INTO intelligence_precomputed (org_id, project_id, computation_type, result, valid_until)
    VALUES (p_org_id, p_project_id, p_type, p_result, now() + (p_valid_hours || ' hours')::INTERVAL)
    ON CONFLICT (org_id, project_id, computation_type)
    DO UPDATE SET
        result = EXCLUDED.result,
        computed_at = now(),
        valid_until = now() + (p_valid_hours || ' hours')::INTERVAL
    RETURNING id INTO v_id;
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;
