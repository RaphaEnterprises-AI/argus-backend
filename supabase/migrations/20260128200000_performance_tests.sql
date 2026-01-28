-- Performance Tests Table
-- Stores results from performance analyzer agent (Lighthouse-style metrics)

CREATE TABLE IF NOT EXISTS performance_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    device TEXT NOT NULL DEFAULT 'mobile' CHECK (device IN ('mobile', 'desktop')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),

    -- Core Web Vitals
    lcp_ms NUMERIC,          -- Largest Contentful Paint (ms)
    fid_ms NUMERIC,          -- First Input Delay (ms)
    cls NUMERIC,             -- Cumulative Layout Shift (unitless)
    inp_ms NUMERIC,          -- Interaction to Next Paint (ms)

    -- Additional timing metrics
    ttfb_ms NUMERIC,         -- Time to First Byte (ms)
    fcp_ms NUMERIC,          -- First Contentful Paint (ms)
    speed_index NUMERIC,     -- Speed Index
    tti_ms NUMERIC,          -- Time to Interactive (ms)
    tbt_ms NUMERIC,          -- Total Blocking Time (ms)

    -- Resource metrics
    total_requests INTEGER,
    total_transfer_size_kb NUMERIC,
    js_execution_time_ms NUMERIC,
    dom_content_loaded_ms NUMERIC,
    load_time_ms NUMERIC,

    -- Lighthouse-style scores (0-100)
    performance_score INTEGER,
    accessibility_score INTEGER,
    best_practices_score INTEGER,
    seo_score INTEGER,

    -- Overall grade
    overall_grade TEXT CHECK (overall_grade IN ('excellent', 'good', 'needs_work', 'poor')),

    -- AI Analysis
    recommendations JSONB DEFAULT '[]'::jsonb,
    issues JSONB DEFAULT '[]'::jsonb,
    summary TEXT,

    -- Metadata
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    triggered_by TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_performance_tests_project_id ON performance_tests(project_id);
CREATE INDEX IF NOT EXISTS idx_performance_tests_status ON performance_tests(status);
CREATE INDEX IF NOT EXISTS idx_performance_tests_created_at ON performance_tests(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_performance_tests_project_status ON performance_tests(project_id, status);

-- Enable RLS
ALTER TABLE performance_tests ENABLE ROW LEVEL SECURITY;

-- RLS Policies (same pattern as other tables)
CREATE POLICY "Users can view their organization performance tests"
    ON performance_tests FOR SELECT
    USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organization_members om ON om.organization_id = p.organization_id
            WHERE om.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert performance tests for their organization projects"
    ON performance_tests FOR INSERT
    WITH CHECK (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organization_members om ON om.organization_id = p.organization_id
            WHERE om.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can update their organization performance tests"
    ON performance_tests FOR UPDATE
    USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organization_members om ON om.organization_id = p.organization_id
            WHERE om.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can delete their organization performance tests"
    ON performance_tests FOR DELETE
    USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organization_members om ON om.organization_id = p.organization_id
            WHERE om.user_id = auth.uid()
        )
    );

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON performance_tests TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON performance_tests TO service_role;

-- Comment on table
COMMENT ON TABLE performance_tests IS 'Stores performance analysis results from the Performance Analyzer Agent including Core Web Vitals, Lighthouse scores, and AI-generated recommendations';
