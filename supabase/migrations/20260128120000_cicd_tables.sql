-- =============================================================================
-- CI/CD Tables Migration
-- =============================================================================
-- Creates tables for storing CI/CD pipeline data:
-- - ci_pipelines: CI/CD pipeline runs
-- - ci_builds: Build history
-- - ci_deployments: Deployment tracking
-- - ci_test_impact: Test impact analysis
-- =============================================================================

-- =============================================================================
-- 1. CI_PIPELINES TABLE
-- =============================================================================
-- Stores CI/CD pipeline run metadata from GitHub Actions, GitLab CI, Jenkins, etc.

CREATE TABLE IF NOT EXISTS ci_pipelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Provider and identification
    provider TEXT NOT NULL,  -- github, gitlab, jenkins, circleci, etc.
    name TEXT NOT NULL,
    branch TEXT NOT NULL,

    -- Status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'running', 'success', 'failed', 'cancelled', 'skipped'
    )),

    -- Commit info
    commit_sha TEXT NOT NULL,
    commit_message TEXT,
    commit_author TEXT,

    -- Trigger info
    trigger TEXT CHECK (trigger IN (
        'push', 'pull_request', 'schedule', 'manual', 'webhook'
    )),

    -- Workflow details
    workflow_name TEXT,
    workflow_url TEXT,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,

    -- Structured data
    stages JSONB DEFAULT '[]',  -- Array of stage info
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for ci_pipelines
CREATE INDEX IF NOT EXISTS idx_ci_pipelines_project ON ci_pipelines(project_id);
CREATE INDEX IF NOT EXISTS idx_ci_pipelines_status ON ci_pipelines(status);
CREATE INDEX IF NOT EXISTS idx_ci_pipelines_branch ON ci_pipelines(branch);
CREATE INDEX IF NOT EXISTS idx_ci_pipelines_created ON ci_pipelines(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ci_pipelines_commit ON ci_pipelines(commit_sha);
CREATE INDEX IF NOT EXISTS idx_ci_pipelines_provider ON ci_pipelines(provider);

-- =============================================================================
-- 2. CI_BUILDS TABLE
-- =============================================================================
-- Stores build history with test results and coverage data.

CREATE TABLE IF NOT EXISTS ci_builds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    pipeline_id UUID REFERENCES ci_pipelines(id) ON DELETE SET NULL,

    -- Provider and identification
    provider TEXT NOT NULL,
    build_number INTEGER NOT NULL,
    name TEXT NOT NULL,
    branch TEXT NOT NULL,

    -- Status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'running', 'success', 'failed', 'cancelled', 'skipped'
    )),

    -- Commit info
    commit_sha TEXT NOT NULL,
    commit_message TEXT,
    commit_author TEXT,

    -- Test results
    tests_total INTEGER DEFAULT 0,
    tests_passed INTEGER DEFAULT 0,
    tests_failed INTEGER DEFAULT 0,
    tests_skipped INTEGER DEFAULT 0,

    -- Coverage
    coverage_percent NUMERIC(5,2),

    -- Artifacts and logs
    artifact_urls JSONB DEFAULT '[]',
    logs_url TEXT,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,

    -- Additional metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for ci_builds
CREATE INDEX IF NOT EXISTS idx_ci_builds_project ON ci_builds(project_id);
CREATE INDEX IF NOT EXISTS idx_ci_builds_pipeline ON ci_builds(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_ci_builds_status ON ci_builds(status);
CREATE INDEX IF NOT EXISTS idx_ci_builds_branch ON ci_builds(branch);
CREATE INDEX IF NOT EXISTS idx_ci_builds_created ON ci_builds(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ci_builds_commit ON ci_builds(commit_sha);
CREATE INDEX IF NOT EXISTS idx_ci_builds_provider ON ci_builds(provider);
CREATE INDEX IF NOT EXISTS idx_ci_builds_number ON ci_builds(project_id, build_number DESC);

-- =============================================================================
-- 3. CI_DEPLOYMENTS TABLE
-- =============================================================================
-- Stores deployment information across environments.

CREATE TABLE IF NOT EXISTS ci_deployments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    build_id UUID REFERENCES ci_builds(id) ON DELETE SET NULL,

    -- Environment
    environment TEXT NOT NULL CHECK (environment IN (
        'development', 'staging', 'production', 'preview'
    )),

    -- Status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'in_progress', 'success', 'failed', 'rolled_back'
    )),

    -- Version and commit
    version TEXT,
    commit_sha TEXT NOT NULL,

    -- Who deployed
    deployed_by TEXT,

    -- URLs
    deployment_url TEXT,
    preview_url TEXT,

    -- Risk assessment
    risk_score INTEGER CHECK (risk_score >= 0 AND risk_score <= 100),
    risk_factors JSONB DEFAULT '[]',

    -- Health status
    health_check_status TEXT DEFAULT 'unknown' CHECK (health_check_status IN (
        'healthy', 'degraded', 'unhealthy', 'unknown'
    )),

    -- Rollback info
    rollback_available BOOLEAN DEFAULT FALSE,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,

    -- Additional metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for ci_deployments
CREATE INDEX IF NOT EXISTS idx_ci_deployments_project ON ci_deployments(project_id);
CREATE INDEX IF NOT EXISTS idx_ci_deployments_build ON ci_deployments(build_id);
CREATE INDEX IF NOT EXISTS idx_ci_deployments_status ON ci_deployments(status);
CREATE INDEX IF NOT EXISTS idx_ci_deployments_environment ON ci_deployments(environment);
CREATE INDEX IF NOT EXISTS idx_ci_deployments_created ON ci_deployments(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ci_deployments_commit ON ci_deployments(commit_sha);
CREATE INDEX IF NOT EXISTS idx_ci_deployments_health ON ci_deployments(health_check_status);

-- =============================================================================
-- 4. CI_TEST_IMPACT TABLE
-- =============================================================================
-- Stores test impact analysis results for commits.

CREATE TABLE IF NOT EXISTS ci_test_impact (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Commit info
    commit_sha TEXT NOT NULL,
    branch TEXT NOT NULL,
    base_sha TEXT,  -- Base commit for comparison

    -- Changed files analysis
    changed_files JSONB DEFAULT '[]',  -- Array of file paths with change types
    total_files_changed INTEGER DEFAULT 0,

    -- Impacted tests
    impacted_tests JSONB DEFAULT '[]',  -- Array of test identifiers
    total_tests_impacted INTEGER DEFAULT 0,

    -- Test recommendations
    recommended_tests JSONB DEFAULT '[]',  -- Tests that should run
    skip_candidates JSONB DEFAULT '[]',  -- Tests that can be skipped

    -- Confidence and timing
    confidence_score NUMERIC(5,2) CHECK (confidence_score >= 0 AND confidence_score <= 100),
    analysis_time_ms INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for ci_test_impact
CREATE INDEX IF NOT EXISTS idx_ci_test_impact_project ON ci_test_impact(project_id);
CREATE INDEX IF NOT EXISTS idx_ci_test_impact_commit ON ci_test_impact(commit_sha);
CREATE INDEX IF NOT EXISTS idx_ci_test_impact_branch ON ci_test_impact(branch);
CREATE INDEX IF NOT EXISTS idx_ci_test_impact_created ON ci_test_impact(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ci_test_impact_base ON ci_test_impact(base_sha);

-- =============================================================================
-- ROW LEVEL SECURITY
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE ci_pipelines ENABLE ROW LEVEL SECURITY;
ALTER TABLE ci_builds ENABLE ROW LEVEL SECURITY;
ALTER TABLE ci_deployments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ci_test_impact ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- RLS POLICIES - Project-scoped access
-- =============================================================================

CREATE POLICY "ci_pipelines_policy" ON ci_pipelines
    FOR ALL USING (
        public.is_service_role() OR
        public.has_project_access(project_id)
    );

CREATE POLICY "ci_builds_policy" ON ci_builds
    FOR ALL USING (
        public.is_service_role() OR
        public.has_project_access(project_id)
    );

CREATE POLICY "ci_deployments_policy" ON ci_deployments
    FOR ALL USING (
        public.is_service_role() OR
        public.has_project_access(project_id)
    );

CREATE POLICY "ci_test_impact_policy" ON ci_test_impact
    FOR ALL USING (
        public.is_service_role() OR
        public.has_project_access(project_id)
    );

-- =============================================================================
-- TRIGGERS - Auto-update timestamps
-- =============================================================================

-- Trigger function for ci_pipelines updated_at
CREATE OR REPLACE FUNCTION update_ci_pipelines_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_ci_pipelines_updated ON ci_pipelines;
CREATE TRIGGER trg_ci_pipelines_updated
    BEFORE UPDATE ON ci_pipelines
    FOR EACH ROW
    EXECUTE FUNCTION update_ci_pipelines_updated_at();

-- =============================================================================
-- ENABLE REALTIME
-- =============================================================================
-- Enable realtime subscriptions for all CI/CD tables

DO $$
BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE ci_pipelines;
EXCEPTION WHEN duplicate_object THEN
    NULL; -- Already added
END $$;

DO $$
BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE ci_builds;
EXCEPTION WHEN duplicate_object THEN
    NULL; -- Already added
END $$;

DO $$
BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE ci_deployments;
EXCEPTION WHEN duplicate_object THEN
    NULL; -- Already added
END $$;

DO $$
BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE ci_test_impact;
EXCEPTION WHEN duplicate_object THEN
    NULL; -- Already added
END $$;

-- =============================================================================
-- GRANTS
-- =============================================================================

-- Grant service_role full access to all tables
GRANT ALL ON ci_pipelines TO service_role;
GRANT ALL ON ci_builds TO service_role;
GRANT ALL ON ci_deployments TO service_role;
GRANT ALL ON ci_test_impact TO service_role;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE ci_pipelines IS 'CI/CD pipeline runs from GitHub Actions, GitLab CI, Jenkins, etc.';
COMMENT ON TABLE ci_builds IS 'Build history with test results and coverage data';
COMMENT ON TABLE ci_deployments IS 'Deployment tracking across environments with health monitoring';
COMMENT ON TABLE ci_test_impact IS 'Test impact analysis for intelligent test selection';

COMMENT ON COLUMN ci_pipelines.provider IS 'CI provider: github, gitlab, jenkins, circleci, etc.';
COMMENT ON COLUMN ci_pipelines.trigger IS 'What triggered the pipeline: push, pull_request, schedule, manual, webhook';
COMMENT ON COLUMN ci_pipelines.stages IS 'JSON array of pipeline stages with status and timing';

COMMENT ON COLUMN ci_builds.artifact_urls IS 'JSON array of build artifact URLs';
COMMENT ON COLUMN ci_builds.coverage_percent IS 'Code coverage percentage from test run';

COMMENT ON COLUMN ci_deployments.environment IS 'Deployment environment: development, staging, production, preview';
COMMENT ON COLUMN ci_deployments.risk_score IS 'Deployment risk score 0-100 based on change analysis';
COMMENT ON COLUMN ci_deployments.health_check_status IS 'Post-deployment health check status';

COMMENT ON COLUMN ci_test_impact.recommended_tests IS 'Tests recommended to run based on change analysis';
COMMENT ON COLUMN ci_test_impact.skip_candidates IS 'Tests that can potentially be skipped';
COMMENT ON COLUMN ci_test_impact.confidence_score IS 'Confidence score 0-100 for the impact analysis';

-- =============================================================================
-- Migration Complete
-- =============================================================================
SELECT 'CI/CD tables created successfully!' as message;
