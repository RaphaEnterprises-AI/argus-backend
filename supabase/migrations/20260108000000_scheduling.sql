-- Test Scheduling Tables Migration
-- Adds test schedule configuration and execution history

-- ============================================================================
-- TEST SCHEDULES TABLE (Test schedule configuration)
-- ============================================================================

CREATE TABLE IF NOT EXISTS test_schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,

    -- Schedule identification
    name TEXT NOT NULL,
    description TEXT,

    -- Cron configuration
    cron_expression TEXT NOT NULL,  -- Standard cron format: "0 0 * * *"
    timezone TEXT DEFAULT 'UTC',

    -- State
    enabled BOOLEAN DEFAULT true,
    is_recurring BOOLEAN DEFAULT true,

    -- Test selection
    test_ids UUID[] DEFAULT '{}',  -- Specific tests to run
    test_filter JSONB DEFAULT '{}',  -- Dynamic filter: {"tags": ["smoke"], "priority": ["critical"]}

    -- Timing
    next_run_at TIMESTAMPTZ,
    last_run_at TIMESTAMPTZ,

    -- Statistics
    run_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    success_rate NUMERIC(5,2) DEFAULT 0,

    -- Notification configuration
    notification_config JSONB DEFAULT jsonb_build_object(
        'on_failure', true,
        'on_success', false,
        'channels', ARRAY[]::TEXT[]
    ),

    -- Execution settings
    max_parallel_tests INTEGER DEFAULT 5,
    timeout_ms INTEGER DEFAULT 3600000,  -- 1 hour default
    retry_failed_tests BOOLEAN DEFAULT true,
    retry_count INTEGER DEFAULT 2,

    -- Environment
    environment TEXT DEFAULT 'staging',
    browser TEXT DEFAULT 'chromium',
    app_url_override TEXT,  -- Override project's default URL

    -- Metadata
    created_by UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_test_schedules_project ON test_schedules(project_id);
CREATE INDEX IF NOT EXISTS idx_test_schedules_org ON test_schedules(organization_id);
CREATE INDEX IF NOT EXISTS idx_test_schedules_enabled ON test_schedules(enabled) WHERE enabled = true;
CREATE INDEX IF NOT EXISTS idx_test_schedules_next_run ON test_schedules(next_run_at) WHERE enabled = true;
CREATE INDEX IF NOT EXISTS idx_test_schedules_created ON test_schedules(created_at DESC);

-- ============================================================================
-- SCHEDULE RUNS TABLE (Schedule execution history)
-- ============================================================================

CREATE TABLE IF NOT EXISTS schedule_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schedule_id UUID NOT NULL REFERENCES test_schedules(id) ON DELETE CASCADE,
    test_run_id UUID REFERENCES test_runs(id) ON DELETE SET NULL,

    -- Timing
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Status
    status TEXT DEFAULT 'pending' CHECK (status IN (
        'pending', 'queued', 'running', 'passed', 'failed', 'cancelled', 'timeout'
    )),

    -- Trigger info
    trigger_type TEXT DEFAULT 'scheduled' CHECK (trigger_type IN (
        'scheduled', 'manual', 'webhook', 'api'
    )),
    triggered_by TEXT,  -- User ID or system identifier

    -- Test results summary
    tests_total INTEGER DEFAULT 0,
    tests_passed INTEGER DEFAULT 0,
    tests_failed INTEGER DEFAULT 0,
    tests_skipped INTEGER DEFAULT 0,
    duration_ms INTEGER,

    -- Error tracking
    error_message TEXT,
    error_details JSONB,

    -- Execution logs
    logs JSONB DEFAULT '[]',

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_schedule_runs_schedule ON schedule_runs(schedule_id);
CREATE INDEX IF NOT EXISTS idx_schedule_runs_test_run ON schedule_runs(test_run_id);
CREATE INDEX IF NOT EXISTS idx_schedule_runs_status ON schedule_runs(status);
CREATE INDEX IF NOT EXISTS idx_schedule_runs_triggered ON schedule_runs(triggered_at DESC);
CREATE INDEX IF NOT EXISTS idx_schedule_runs_schedule_status ON schedule_runs(schedule_id, status);

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE test_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE schedule_runs ENABLE ROW LEVEL SECURITY;

-- Policies for test_schedules
DROP POLICY IF EXISTS "Users can view schedules for their projects" ON test_schedules;
CREATE POLICY "Users can view schedules for their projects" ON test_schedules
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = test_schedules.project_id
        )
    );

DROP POLICY IF EXISTS "Users can manage schedules for their projects" ON test_schedules;
CREATE POLICY "Users can manage schedules for their projects" ON test_schedules
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = test_schedules.project_id
        )
    );

-- Policies for schedule_runs
DROP POLICY IF EXISTS "Users can view schedule runs" ON schedule_runs;
CREATE POLICY "Users can view schedule runs" ON schedule_runs
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM test_schedules
            JOIN projects ON projects.id = test_schedules.project_id
            WHERE test_schedules.id = schedule_runs.schedule_id
        )
    );

DROP POLICY IF EXISTS "Users can manage schedule runs" ON schedule_runs;
CREATE POLICY "Users can manage schedule runs" ON schedule_runs
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM test_schedules
            JOIN projects ON projects.id = test_schedules.project_id
            WHERE test_schedules.id = schedule_runs.schedule_id
        )
    );

-- Service role policies
DROP POLICY IF EXISTS "Service role has full access to test_schedules" ON test_schedules;
CREATE POLICY "Service role has full access to test_schedules" ON test_schedules
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role has full access to schedule_runs" ON schedule_runs;
CREATE POLICY "Service role has full access to schedule_runs" ON schedule_runs
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update updated_at on test_schedules changes
CREATE TRIGGER update_test_schedules_updated_at
    BEFORE UPDATE ON test_schedules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to calculate next run time based on cron expression
-- Note: Full cron parsing requires pg_cron extension or application-level logic
CREATE OR REPLACE FUNCTION update_schedule_next_run(p_schedule_id UUID)
RETURNS TIMESTAMPTZ AS $$
DECLARE
    v_next_run TIMESTAMPTZ;
BEGIN
    -- For now, set next_run_at to NULL to indicate it needs recalculation
    -- Application layer should calculate based on cron_expression
    UPDATE test_schedules
    SET next_run_at = NULL,
        updated_at = NOW()
    WHERE id = p_schedule_id;

    SELECT next_run_at INTO v_next_run FROM test_schedules WHERE id = p_schedule_id;
    RETURN v_next_run;
END;
$$ LANGUAGE plpgsql;

-- Function to update schedule stats after a run completes
CREATE OR REPLACE FUNCTION update_schedule_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status IN ('passed', 'failed') AND OLD.status != NEW.status THEN
        UPDATE test_schedules
        SET
            run_count = run_count + 1,
            failure_count = CASE WHEN NEW.status = 'failed' THEN failure_count + 1 ELSE failure_count END,
            last_run_at = NEW.completed_at,
            success_rate = (
                SELECT ROUND(
                    CAST(COUNT(*) FILTER (WHERE status = 'passed') AS NUMERIC) /
                    NULLIF(COUNT(*) FILTER (WHERE status IN ('passed', 'failed')), 0) * 100,
                    2
                )
                FROM schedule_runs
                WHERE schedule_id = NEW.schedule_id
            ),
            updated_at = NOW()
        WHERE id = NEW.schedule_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_schedule_stats_trigger
    AFTER UPDATE ON schedule_runs
    FOR EACH ROW EXECUTE FUNCTION update_schedule_stats();

-- Function to get due schedules
CREATE OR REPLACE FUNCTION get_due_schedules()
RETURNS TABLE(
    schedule_id UUID,
    project_id UUID,
    schedule_name TEXT,
    cron_expression TEXT,
    test_ids UUID[],
    test_filter JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ts.id as schedule_id,
        ts.project_id,
        ts.name as schedule_name,
        ts.cron_expression,
        ts.test_ids,
        ts.test_filter
    FROM test_schedules ts
    WHERE ts.enabled = true
    AND (ts.next_run_at IS NULL OR ts.next_run_at <= NOW());
END;
$$ LANGUAGE plpgsql;

-- Function to create a schedule run
CREATE OR REPLACE FUNCTION create_schedule_run(
    p_schedule_id UUID,
    p_trigger_type TEXT DEFAULT 'scheduled',
    p_triggered_by TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_run_id UUID;
BEGIN
    INSERT INTO schedule_runs (
        schedule_id,
        trigger_type,
        triggered_by,
        status
    ) VALUES (
        p_schedule_id,
        p_trigger_type,
        p_triggered_by,
        'queued'
    )
    RETURNING id INTO v_run_id;

    RETURN v_run_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- ENABLE REALTIME
-- ============================================================================

ALTER PUBLICATION supabase_realtime ADD TABLE schedule_runs;

-- ============================================================================
-- COMPLETION
-- ============================================================================

SELECT 'Scheduling tables created successfully!' as message;
