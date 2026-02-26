-- =============================================================================
-- QA Engineer Agent Tables
-- Supports full/PR/coverage analysis jobs and time-series quality score trending
-- =============================================================================

-- =============================================================================
-- QA_ANALYSIS_JOBS — Full/PR/coverage analysis job queue
-- =============================================================================
CREATE TABLE IF NOT EXISTS qa_analysis_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Job lifecycle
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress INTEGER DEFAULT 0
        CHECK (progress >= 0 AND progress <= 100),

    -- Analysis classification
    analysis_type TEXT NOT NULL
        CHECK (analysis_type IN ('full', 'pr_review', 'coverage_only', 'incremental')),
    trigger_context JSONB DEFAULT '{}'::jsonb,  -- PR number, commit SHA, branch, etc.

    -- Results: overall quality
    quality_score DOUBLE PRECISION
        CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 100)),

    -- Results: structured analysis outputs
    components_analyzed JSONB DEFAULT '[]'::jsonb,     -- [{name, type, file_count, test_count, coverage_pct}]
    coverage_gaps JSONB DEFAULT '[]'::jsonb,           -- [{entity, risk_level, description, suggested_tests}]
    user_flows_discovered JSONB DEFAULT '[]'::jsonb,   -- [{name, steps, risk_score, coverage_pct}]
    recommendations JSONB DEFAULT '[]'::jsonb,         -- [{type, priority, description, effort}]

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
CREATE INDEX IF NOT EXISTS idx_qa_analysis_jobs_org_project
    ON qa_analysis_jobs (organization_id, project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_qa_analysis_jobs_status
    ON qa_analysis_jobs (status)
    WHERE status NOT IN ('completed', 'failed', 'cancelled');

CREATE INDEX IF NOT EXISTS idx_qa_analysis_jobs_project
    ON qa_analysis_jobs (project_id, created_at DESC);

-- =============================================================================
-- QA_QUALITY_SCORES — Time-series quality scores for trending
-- =============================================================================
CREATE TABLE IF NOT EXISTS qa_quality_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    analysis_job_id UUID REFERENCES qa_analysis_jobs(id) ON DELETE SET NULL,

    -- Composite score
    overall_score DOUBLE PRECISION NOT NULL
        CHECK (overall_score >= 0 AND overall_score <= 100),

    -- Component scores
    test_coverage_pct DOUBLE PRECISION
        CHECK (test_coverage_pct IS NULL OR (test_coverage_pct >= 0 AND test_coverage_pct <= 100)),
    requirement_coverage_pct DOUBLE PRECISION
        CHECK (requirement_coverage_pct IS NULL OR (requirement_coverage_pct >= 0 AND requirement_coverage_pct <= 100)),
    test_health_score DOUBLE PRECISION
        CHECK (test_health_score IS NULL OR (test_health_score >= 0 AND test_health_score <= 100)),
    code_quality_score DOUBLE PRECISION
        CHECK (code_quality_score IS NULL OR (code_quality_score >= 0 AND code_quality_score <= 100)),

    -- Gap counts
    gap_count INTEGER DEFAULT 0,
    critical_gaps INTEGER DEFAULT 0,

    -- Comparison with previous measurement
    comparison_delta JSONB,  -- {score_change, new_gaps, closed_gaps}

    -- Provenance
    commit_sha TEXT,
    details JSONB DEFAULT '{}'::jsonb,

    -- When this measurement was taken
    measured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_qa_quality_scores_org_project
    ON qa_quality_scores (organization_id, project_id, measured_at DESC);

CREATE INDEX IF NOT EXISTS idx_qa_quality_scores_measured_at
    ON qa_quality_scores (measured_at DESC);

CREATE INDEX IF NOT EXISTS idx_qa_quality_scores_analysis_job
    ON qa_quality_scores (analysis_job_id)
    WHERE analysis_job_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_qa_quality_scores_commit_sha
    ON qa_quality_scores (commit_sha)
    WHERE commit_sha IS NOT NULL;

-- =============================================================================
-- Row Level Security
-- =============================================================================
ALTER TABLE qa_analysis_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE qa_quality_scores ENABLE ROW LEVEL SECURITY;

-- qa_analysis_jobs: org isolation
DROP POLICY IF EXISTS qa_analysis_jobs_org_isolation ON qa_analysis_jobs;
CREATE POLICY qa_analysis_jobs_org_isolation ON qa_analysis_jobs
    FOR ALL
    USING (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id')
    WITH CHECK (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id');

DROP POLICY IF EXISTS qa_analysis_jobs_service_role ON qa_analysis_jobs;
CREATE POLICY qa_analysis_jobs_service_role ON qa_analysis_jobs
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- qa_quality_scores: org isolation
DROP POLICY IF EXISTS qa_quality_scores_org_isolation ON qa_quality_scores;
CREATE POLICY qa_quality_scores_org_isolation ON qa_quality_scores
    FOR ALL
    USING (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id')
    WITH CHECK (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id');

DROP POLICY IF EXISTS qa_quality_scores_service_role ON qa_quality_scores;
CREATE POLICY qa_quality_scores_service_role ON qa_quality_scores
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- =============================================================================
-- updated_at trigger for qa_analysis_jobs
-- =============================================================================
CREATE OR REPLACE FUNCTION update_qa_analysis_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS qa_analysis_jobs_updated_at ON qa_analysis_jobs;
CREATE TRIGGER qa_analysis_jobs_updated_at
    BEFORE UPDATE ON qa_analysis_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_qa_analysis_jobs_updated_at();

-- =============================================================================
-- Completion
-- =============================================================================
SELECT 'QA Engineer agent tables created successfully!' AS message;
