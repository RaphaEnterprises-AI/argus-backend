-- =============================================================================
-- Test Generation Agent Tables
-- Supports async job queue, parsed requirements, and coverage traceability
-- =============================================================================

-- =============================================================================
-- TESTGEN_JOBS — Async job queue for test generation
-- =============================================================================
CREATE TABLE IF NOT EXISTS testgen_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Job lifecycle
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress INTEGER DEFAULT 0
        CHECK (progress >= 0 AND progress <= 100),

    -- Source configuration
    source_type TEXT NOT NULL
        CHECK (source_type IN (
            'requirements_text', 'jira', 'github_pr',
            'figma', 'confluence', 'codebase_analysis'
        )),
    source_data JSONB NOT NULL,         -- input requirements / PR diff / etc
    config JSONB DEFAULT '{}'::jsonb,   -- framework, language, patterns, etc

    -- Results
    results JSONB,                      -- generated tests, scenarios, code
    quality_score DOUBLE PRECISION,     -- from AgentAsJudge (0.0-1.0)
    ai_cost DOUBLE PRECISION DEFAULT 0, -- total AI cost in USD for this job
    cognee_dataset_ref TEXT,            -- reference to Cognee namespace

    -- Error handling
    error_message TEXT,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_testgen_jobs_org_project
    ON testgen_jobs (organization_id, project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_testgen_jobs_status
    ON testgen_jobs (status)
    WHERE status NOT IN ('completed', 'failed', 'cancelled');

CREATE INDEX IF NOT EXISTS idx_testgen_jobs_project
    ON testgen_jobs (project_id, created_at DESC);

-- =============================================================================
-- TESTGEN_REQUIREMENTS — Parsed requirements per job
-- =============================================================================
CREATE TABLE IF NOT EXISTS testgen_requirements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES testgen_jobs(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL,
    project_id UUID NOT NULL,

    -- Requirement identification
    source_ref TEXT,                     -- external ID (Jira key, PR number, etc)
    title TEXT NOT NULL,
    description TEXT,

    -- Structured analysis
    acceptance_criteria JSONB,           -- array of acceptance criteria
    ambiguity_score DOUBLE PRECISION     -- 0.0 (clear) to 1.0 (very ambiguous)
        CHECK (ambiguity_score IS NULL OR (ambiguity_score >= 0.0 AND ambiguity_score <= 1.0)),
    ambiguity_details JSONB,             -- specific ambiguities found

    -- Classification
    priority TEXT DEFAULT 'medium'
        CHECK (priority IN ('critical', 'high', 'medium', 'low')),
    tags JSONB DEFAULT '[]'::jsonb,
    generated_test_ids JSONB DEFAULT '[]'::jsonb,  -- links to generated tests

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_testgen_requirements_job
    ON testgen_requirements (job_id);

CREATE INDEX IF NOT EXISTS idx_testgen_requirements_org_project
    ON testgen_requirements (organization_id, project_id);

CREATE INDEX IF NOT EXISTS idx_testgen_requirements_source_ref
    ON testgen_requirements (source_ref)
    WHERE source_ref IS NOT NULL;

-- =============================================================================
-- TESTGEN_COVERAGE_MATRIX — Requirement-to-test traceability
-- =============================================================================
CREATE TABLE IF NOT EXISTS testgen_coverage_matrix (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES testgen_jobs(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL,
    project_id UUID NOT NULL,

    -- Traceability links
    requirement_id UUID REFERENCES testgen_requirements(id) ON DELETE SET NULL,
    code_entity TEXT NOT NULL,            -- function/class/endpoint being tested
    code_entity_type TEXT
        CHECK (code_entity_type IN (
            'function', 'class', 'endpoint', 'component', 'page'
        )),

    -- Test details
    test_name TEXT NOT NULL,
    test_type TEXT
        CHECK (test_type IN ('unit', 'integration', 'e2e', 'api', 'visual')),
    coverage_type TEXT DEFAULT 'full'
        CHECK (coverage_type IN ('full', 'partial', 'negative', 'edge_case')),
    confidence DOUBLE PRECISION DEFAULT 1.0
        CHECK (confidence >= 0.0 AND confidence <= 1.0),

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_testgen_coverage_job
    ON testgen_coverage_matrix (job_id);

CREATE INDEX IF NOT EXISTS idx_testgen_coverage_org_project
    ON testgen_coverage_matrix (organization_id, project_id);

CREATE INDEX IF NOT EXISTS idx_testgen_coverage_requirement
    ON testgen_coverage_matrix (requirement_id)
    WHERE requirement_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_testgen_coverage_entity
    ON testgen_coverage_matrix (code_entity, code_entity_type);

-- =============================================================================
-- Row Level Security
-- =============================================================================
ALTER TABLE testgen_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE testgen_requirements ENABLE ROW LEVEL SECURITY;
ALTER TABLE testgen_coverage_matrix ENABLE ROW LEVEL SECURITY;

-- testgen_jobs: org isolation
DROP POLICY IF EXISTS testgen_jobs_org_isolation ON testgen_jobs;
CREATE POLICY testgen_jobs_org_isolation ON testgen_jobs
    FOR ALL
    USING (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id')
    WITH CHECK (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id');

DROP POLICY IF EXISTS testgen_jobs_service_role ON testgen_jobs;
CREATE POLICY testgen_jobs_service_role ON testgen_jobs
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- testgen_requirements: org isolation
DROP POLICY IF EXISTS testgen_requirements_org_isolation ON testgen_requirements;
CREATE POLICY testgen_requirements_org_isolation ON testgen_requirements
    FOR ALL
    USING (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id')
    WITH CHECK (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id');

DROP POLICY IF EXISTS testgen_requirements_service_role ON testgen_requirements;
CREATE POLICY testgen_requirements_service_role ON testgen_requirements
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- testgen_coverage_matrix: org isolation
DROP POLICY IF EXISTS testgen_coverage_matrix_org_isolation ON testgen_coverage_matrix;
CREATE POLICY testgen_coverage_matrix_org_isolation ON testgen_coverage_matrix
    FOR ALL
    USING (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id')
    WITH CHECK (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id');

DROP POLICY IF EXISTS testgen_coverage_matrix_service_role ON testgen_coverage_matrix;
CREATE POLICY testgen_coverage_matrix_service_role ON testgen_coverage_matrix
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- =============================================================================
-- updated_at trigger for testgen_jobs
-- =============================================================================
CREATE OR REPLACE FUNCTION update_testgen_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS testgen_jobs_updated_at ON testgen_jobs;
CREATE TRIGGER testgen_jobs_updated_at
    BEFORE UPDATE ON testgen_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_testgen_jobs_updated_at();

-- =============================================================================
-- Completion
-- =============================================================================
SELECT 'Test generation agent tables created successfully!' AS message;
