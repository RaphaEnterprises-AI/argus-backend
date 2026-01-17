-- Reports table migration
-- Stores test execution reports with metadata, content, and export capabilities

-- Create reports table
CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    test_run_id UUID REFERENCES test_runs(id) ON DELETE SET NULL,

    -- Report metadata
    name TEXT NOT NULL,
    description TEXT,
    report_type TEXT NOT NULL DEFAULT 'test_execution' CHECK (report_type IN ('test_execution', 'coverage', 'trend', 'quality', 'custom')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'generating', 'completed', 'failed')),
    format TEXT NOT NULL DEFAULT 'json' CHECK (format IN ('json', 'html', 'pdf', 'markdown', 'junit')),

    -- Report content
    summary JSONB DEFAULT '{}',
    content JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',

    -- Execution details
    total_tests INTEGER DEFAULT 0,
    passed_tests INTEGER DEFAULT 0,
    failed_tests INTEGER DEFAULT 0,
    skipped_tests INTEGER DEFAULT 0,
    duration_ms INTEGER,
    coverage_percentage DECIMAL(5,2),

    -- Date range for trend reports
    date_from TIMESTAMPTZ,
    date_to TIMESTAMPTZ,

    -- File storage (for PDF/HTML exports)
    file_url TEXT,
    file_size_bytes INTEGER,

    -- Metadata
    created_by TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    generated_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_reports_organization ON reports(organization_id);
CREATE INDEX IF NOT EXISTS idx_reports_project ON reports(project_id);
CREATE INDEX IF NOT EXISTS idx_reports_test_run ON reports(test_run_id);
CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);

-- Enable RLS
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;

-- Create RLS policy
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON reports;
CREATE POLICY "Enable all access for authenticated users" ON reports FOR ALL USING (true);

-- Create trigger for updated_at
CREATE OR REPLACE FUNCTION update_reports_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_reports_updated_at ON reports;
CREATE TRIGGER update_reports_updated_at
    BEFORE UPDATE ON reports
    FOR EACH ROW EXECUTE FUNCTION update_reports_updated_at();

-- Done!
SELECT 'Reports table created successfully!' as message;
