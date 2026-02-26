-- =============================================================================
-- TestHealer CI/CD Tables
-- Supports proactive healing scans, CI failure analysis, PR review,
-- and time-series health score trending
-- =============================================================================

-- =============================================================================
-- HEALING_SCAN_JOBS — Proactive/CI/PR scan tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS healing_scan_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Job lifecycle
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress INTEGER DEFAULT 0
        CHECK (progress >= 0 AND progress <= 100),

    -- Scan classification
    scan_type TEXT NOT NULL
        CHECK (scan_type IN ('proactive', 'ci_failure', 'pr_review', 'manual')),
    trigger_context JSONB DEFAULT '{}'::jsonb,  -- CI run URL, PR number, commit SHA, etc.

    -- Results
    tests_scanned INTEGER DEFAULT 0,
    issues_found INTEGER DEFAULT 0,
    fixes_proposed INTEGER DEFAULT 0,
    fixes_applied INTEGER DEFAULT 0,
    fix_proposals JSONB DEFAULT '[]'::jsonb,    -- [{test_id, issue_type, fix_description, confidence, applied}]

    -- Health delta
    health_score_before DOUBLE PRECISION,
    health_score_after DOUBLE PRECISION,

    -- Cost & lineage
    ai_cost DOUBLE PRECISION DEFAULT 0,
    cognee_dataset_ref TEXT,

    -- Error handling
    error_message TEXT,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_healing_scan_jobs_org_project
    ON healing_scan_jobs (organization_id, project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_healing_scan_jobs_status
    ON healing_scan_jobs (status)
    WHERE status NOT IN ('completed', 'failed', 'cancelled');

CREATE INDEX IF NOT EXISTS idx_healing_scan_jobs_project
    ON healing_scan_jobs (project_id, created_at DESC);

-- =============================================================================
-- HEALING_HEALTH_SCORES — Time-series health scores for trending
-- =============================================================================
CREATE TABLE IF NOT EXISTS healing_health_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    scan_job_id UUID REFERENCES healing_scan_jobs(id) ON DELETE SET NULL,

    -- Composite score
    overall_score DOUBLE PRECISION NOT NULL
        CHECK (overall_score >= 0 AND overall_score <= 100),

    -- Component scores
    selector_stability DOUBLE PRECISION
        CHECK (selector_stability IS NULL OR (selector_stability >= 0 AND selector_stability <= 100)),
    flaky_rate DOUBLE PRECISION
        CHECK (flaky_rate IS NULL OR (flaky_rate >= 0 AND flaky_rate <= 100)),
    heal_success_rate DOUBLE PRECISION
        CHECK (heal_success_rate IS NULL OR (heal_success_rate >= 0 AND heal_success_rate <= 100)),
    mttr_seconds DOUBLE PRECISION,

    -- Counts
    total_tests INTEGER,
    fragile_tests INTEGER,
    healed_tests_30d INTEGER,

    -- Detailed breakdown
    details JSONB DEFAULT '{}'::jsonb,

    -- When this measurement was taken
    measured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_healing_health_scores_org_project
    ON healing_health_scores (organization_id, project_id, measured_at DESC);

CREATE INDEX IF NOT EXISTS idx_healing_health_scores_measured_at
    ON healing_health_scores (measured_at DESC);

CREATE INDEX IF NOT EXISTS idx_healing_health_scores_scan_job
    ON healing_health_scores (scan_job_id)
    WHERE scan_job_id IS NOT NULL;

-- =============================================================================
-- Row Level Security
-- =============================================================================
ALTER TABLE healing_scan_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE healing_health_scores ENABLE ROW LEVEL SECURITY;

-- healing_scan_jobs: org isolation
DROP POLICY IF EXISTS healing_scan_jobs_org_isolation ON healing_scan_jobs;
CREATE POLICY healing_scan_jobs_org_isolation ON healing_scan_jobs
    FOR ALL
    USING (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id')
    WITH CHECK (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id');

DROP POLICY IF EXISTS healing_scan_jobs_service_role ON healing_scan_jobs;
CREATE POLICY healing_scan_jobs_service_role ON healing_scan_jobs
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- healing_health_scores: org isolation
DROP POLICY IF EXISTS healing_health_scores_org_isolation ON healing_health_scores;
CREATE POLICY healing_health_scores_org_isolation ON healing_health_scores
    FOR ALL
    USING (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id')
    WITH CHECK (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id');

DROP POLICY IF EXISTS healing_health_scores_service_role ON healing_health_scores;
CREATE POLICY healing_health_scores_service_role ON healing_health_scores
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- =============================================================================
-- updated_at trigger for healing_scan_jobs
-- =============================================================================
CREATE OR REPLACE FUNCTION update_healing_scan_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS healing_scan_jobs_updated_at ON healing_scan_jobs;
CREATE TRIGGER healing_scan_jobs_updated_at
    BEFORE UPDATE ON healing_scan_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_healing_scan_jobs_updated_at();

-- =============================================================================
-- Completion
-- =============================================================================
SELECT 'TestHealer CI/CD tables created successfully!' AS message;
