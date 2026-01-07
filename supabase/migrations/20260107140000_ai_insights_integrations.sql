-- AI Insights and Integrations Tables
-- Required by dashboard frontend hooks

-- =============================================================================
-- AI Insights Table (add project_id if missing)
-- =============================================================================

-- Add project_id column to ai_insights if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ai_insights' AND column_name = 'project_id'
    ) THEN
        ALTER TABLE ai_insights ADD COLUMN project_id UUID REFERENCES projects(id) ON DELETE CASCADE;
    END IF;
END $$;

-- Add missing columns to ai_insights
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_insights' AND column_name = 'category') THEN
        ALTER TABLE ai_insights ADD COLUMN category TEXT DEFAULT 'general';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_insights' AND column_name = 'recommendation') THEN
        ALTER TABLE ai_insights ADD COLUMN recommendation TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_insights' AND column_name = 'affected_entities') THEN
        ALTER TABLE ai_insights ADD COLUMN affected_entities JSONB DEFAULT '[]';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_insights' AND column_name = 'data_points') THEN
        ALTER TABLE ai_insights ADD COLUMN data_points JSONB DEFAULT '{}';
    END IF;
END $$;

-- Create ai_insights indexes only if project_id exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_insights' AND column_name = 'project_id') THEN
        CREATE INDEX IF NOT EXISTS idx_ai_insights_project ON ai_insights(project_id);
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS idx_ai_insights_severity ON ai_insights(severity);
CREATE INDEX IF NOT EXISTS idx_ai_insights_resolved ON ai_insights(is_resolved);
CREATE INDEX IF NOT EXISTS idx_ai_insights_created ON ai_insights(created_at DESC);

-- =============================================================================
-- Integrations Table (add project_id if missing)
-- =============================================================================

-- Add project_id column to integrations if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'integrations' AND column_name = 'project_id'
    ) THEN
        ALTER TABLE integrations ADD COLUMN project_id UUID REFERENCES projects(id) ON DELETE CASCADE;
    END IF;
END $$;

-- Add missing columns to integrations
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'integrations' AND column_name = 'name') THEN
        ALTER TABLE integrations ADD COLUMN name TEXT DEFAULT 'default';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'integrations' AND column_name = 'credentials') THEN
        ALTER TABLE integrations ADD COLUMN credentials JSONB DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'integrations' AND column_name = 'last_sync_at') THEN
        ALTER TABLE integrations ADD COLUMN last_sync_at TIMESTAMPTZ;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'integrations' AND column_name = 'sync_frequency_minutes') THEN
        ALTER TABLE integrations ADD COLUMN sync_frequency_minutes INT DEFAULT 60;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'integrations' AND column_name = 'error_message') THEN
        ALTER TABLE integrations ADD COLUMN error_message TEXT;
    END IF;
END $$;

-- Create integrations indexes only if columns exist
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'integrations' AND column_name = 'project_id') THEN
        CREATE INDEX IF NOT EXISTS idx_integrations_project ON integrations(project_id);
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'integrations' AND column_name = 'type') THEN
        CREATE INDEX IF NOT EXISTS idx_integrations_type ON integrations(type);
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'integrations' AND column_name = 'status') THEN
        CREATE INDEX IF NOT EXISTS idx_integrations_status ON integrations(status);
    END IF;
END $$;

-- =============================================================================
-- Test Generation Jobs Table (if not exists)
-- Tracks background test generation jobs
-- =============================================================================

CREATE TABLE IF NOT EXISTS test_generation_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    production_event_id UUID REFERENCES production_events(id) ON DELETE SET NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    job_type TEXT DEFAULT 'single_error',
    tests_generated INT DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INT,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for test_generation_jobs
CREATE INDEX IF NOT EXISTS idx_test_gen_jobs_project ON test_generation_jobs(project_id);
CREATE INDEX IF NOT EXISTS idx_test_gen_jobs_status ON test_generation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_test_gen_jobs_created ON test_generation_jobs(created_at DESC);

-- =============================================================================
-- Triggers for updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for ai_insights (only if updated_at column exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_insights' AND column_name = 'updated_at') THEN
        DROP TRIGGER IF EXISTS trigger_ai_insights_updated_at ON ai_insights;
        CREATE TRIGGER trigger_ai_insights_updated_at
            BEFORE UPDATE ON ai_insights
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- Trigger for integrations (only if updated_at column exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'integrations' AND column_name = 'updated_at') THEN
        DROP TRIGGER IF EXISTS trigger_integrations_updated_at ON integrations;
        CREATE TRIGGER trigger_integrations_updated_at
            BEFORE UPDATE ON integrations
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- =============================================================================
-- Row Level Security (RLS) - with IF NOT EXISTS checks
-- =============================================================================

ALTER TABLE ai_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE integrations ENABLE ROW LEVEL SECURITY;
ALTER TABLE test_generation_jobs ENABLE ROW LEVEL SECURITY;

-- Drop existing policies first to avoid conflicts
DROP POLICY IF EXISTS ai_insights_select ON ai_insights;
DROP POLICY IF EXISTS ai_insights_insert ON ai_insights;
DROP POLICY IF EXISTS ai_insights_update ON ai_insights;
DROP POLICY IF EXISTS integrations_select ON integrations;
DROP POLICY IF EXISTS integrations_insert ON integrations;
DROP POLICY IF EXISTS integrations_update ON integrations;
DROP POLICY IF EXISTS integrations_delete ON integrations;
DROP POLICY IF EXISTS test_gen_jobs_select ON test_generation_jobs;
DROP POLICY IF EXISTS test_gen_jobs_insert ON test_generation_jobs;
DROP POLICY IF EXISTS test_gen_jobs_update ON test_generation_jobs;
DROP POLICY IF EXISTS "Service role full access ai_insights" ON ai_insights;
DROP POLICY IF EXISTS "Service role full access integrations" ON integrations;
DROP POLICY IF EXISTS "Service role full access test_gen_jobs" ON test_generation_jobs;

-- AI Insights policies (only if project_id column exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_insights' AND column_name = 'project_id') THEN
        EXECUTE 'CREATE POLICY ai_insights_select ON ai_insights FOR SELECT USING (
            project_id IN (SELECT id FROM projects WHERE organization_id IN (
                SELECT organization_id FROM organization_members WHERE user_id = current_setting(''app.user_id'', true)
            ))
        )';
        EXECUTE 'CREATE POLICY ai_insights_insert ON ai_insights FOR INSERT WITH CHECK (
            project_id IN (SELECT id FROM projects WHERE organization_id IN (
                SELECT organization_id FROM organization_members WHERE user_id = current_setting(''app.user_id'', true)
            ))
        )';
        EXECUTE 'CREATE POLICY ai_insights_update ON ai_insights FOR UPDATE USING (
            project_id IN (SELECT id FROM projects WHERE organization_id IN (
                SELECT organization_id FROM organization_members WHERE user_id = current_setting(''app.user_id'', true)
            ))
        )';
    END IF;
END $$;

-- Integrations policies (only if project_id column exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'integrations' AND column_name = 'project_id') THEN
        EXECUTE 'CREATE POLICY integrations_select ON integrations FOR SELECT USING (
            project_id IN (SELECT id FROM projects WHERE organization_id IN (
                SELECT organization_id FROM organization_members WHERE user_id = current_setting(''app.user_id'', true)
            ))
        )';
        EXECUTE 'CREATE POLICY integrations_insert ON integrations FOR INSERT WITH CHECK (
            project_id IN (SELECT id FROM projects WHERE organization_id IN (
                SELECT organization_id FROM organization_members WHERE user_id = current_setting(''app.user_id'', true)
            ))
        )';
        EXECUTE 'CREATE POLICY integrations_update ON integrations FOR UPDATE USING (
            project_id IN (SELECT id FROM projects WHERE organization_id IN (
                SELECT organization_id FROM organization_members WHERE user_id = current_setting(''app.user_id'', true)
            ))
        )';
        EXECUTE 'CREATE POLICY integrations_delete ON integrations FOR DELETE USING (
            project_id IN (SELECT id FROM projects WHERE organization_id IN (
                SELECT organization_id FROM organization_members WHERE user_id = current_setting(''app.user_id'', true)
            ))
        )';
    END IF;
END $$;

-- Test generation jobs policies
CREATE POLICY test_gen_jobs_select ON test_generation_jobs
    FOR SELECT
    USING (
        project_id IN (
            SELECT id FROM projects
            WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = current_setting('app.user_id', true)
            )
        )
    );

CREATE POLICY test_gen_jobs_insert ON test_generation_jobs
    FOR INSERT
    WITH CHECK (
        project_id IN (
            SELECT id FROM projects
            WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = current_setting('app.user_id', true)
            )
        )
    );

CREATE POLICY test_gen_jobs_update ON test_generation_jobs
    FOR UPDATE
    USING (
        project_id IN (
            SELECT id FROM projects
            WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = current_setting('app.user_id', true)
            )
        )
    );

-- Service role policies (for backend operations)
CREATE POLICY "Service role full access ai_insights" ON ai_insights
    FOR ALL USING (current_setting('role', true) = 'service_role');

CREATE POLICY "Service role full access integrations" ON integrations
    FOR ALL USING (current_setting('role', true) = 'service_role');

CREATE POLICY "Service role full access test_gen_jobs" ON test_generation_jobs
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- =============================================================================
-- Comments (wrapped in conditionals to avoid errors)
-- =============================================================================

COMMENT ON TABLE test_generation_jobs IS 'Background jobs for test generation';

-- Comments for ai_insights columns (only if they exist)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_insights' AND column_name = 'severity') THEN
        COMMENT ON COLUMN ai_insights.severity IS 'Insight severity: critical, high, medium, low';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_insights' AND column_name = 'category') THEN
        COMMENT ON COLUMN ai_insights.category IS 'Insight category: error_pattern, coverage_gap, risk_alert, etc.';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_insights' AND column_name = 'affected_entities') THEN
        COMMENT ON COLUMN ai_insights.affected_entities IS 'List of affected components, pages, or flows';
    END IF;
END $$;

-- Comments for integrations columns (only if they exist)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'integrations' AND column_name = 'credentials') THEN
        COMMENT ON COLUMN integrations.credentials IS 'Encrypted credentials for the integration';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'integrations' AND column_name = 'sync_frequency_minutes') THEN
        COMMENT ON COLUMN integrations.sync_frequency_minutes IS 'How often to sync data from integration';
    END IF;
END $$;
