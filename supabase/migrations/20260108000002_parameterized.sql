-- Parameterized Test Tables Migration
-- Adds support for data-driven testing with parameter sets

-- ============================================================================
-- PARAMETERIZED TESTS TABLE (Data-driven test definitions)
-- ============================================================================

CREATE TABLE IF NOT EXISTS parameterized_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    base_test_id UUID REFERENCES tests(id) ON DELETE SET NULL,

    -- Test identification
    name TEXT NOT NULL,
    description TEXT,
    tags TEXT[] DEFAULT '{}',
    priority TEXT DEFAULT 'medium' CHECK (priority IN ('critical', 'high', 'medium', 'low')),

    -- Data source configuration
    data_source_type TEXT NOT NULL CHECK (data_source_type IN (
        'inline',      -- Parameters defined directly in parameter_sets table
        'csv',         -- CSV file (stored in Supabase Storage)
        'json',        -- JSON file (stored in Supabase Storage)
        'api',         -- External API endpoint
        'database',    -- Database query
        'spreadsheet'  -- Google Sheets or similar
    )),
    data_source_config JSONB NOT NULL DEFAULT '{}',
    -- Examples:
    -- inline: {}
    -- csv: {"storage_path": "param-data/users.csv", "delimiter": ",", "has_header": true}
    -- json: {"storage_path": "param-data/scenarios.json", "json_path": "$.testCases"}
    -- api: {"url": "https://api.example.com/test-data", "headers": {}, "refresh_interval_hours": 24}
    -- database: {"query": "SELECT * FROM test_users", "connection_id": "uuid"}

    -- Parameter schema definition (for validation)
    parameter_schema JSONB DEFAULT '{}',
    -- Example: {"email": {"type": "string", "required": true}, "age": {"type": "number", "min": 0}}

    -- Test steps with parameter placeholders
    steps JSONB NOT NULL DEFAULT '[]',
    -- Example: [
    --   {"action": "navigate", "url": "{{base_url}}/login"},
    --   {"action": "fill", "selector": "#email", "value": "{{email}}"},
    --   {"action": "fill", "selector": "#password", "value": "{{password}}"},
    --   {"action": "click", "selector": "button[type=submit]"}
    -- ]

    -- Assertions with parameter placeholders
    assertions JSONB DEFAULT '[]',
    -- Example: [
    --   {"type": "text_contains", "selector": ".welcome", "expected": "Hello, {{name}}"}
    -- ]

    -- Setup and teardown (run once before/after all iterations)
    setup JSONB DEFAULT '{}',
    teardown JSONB DEFAULT '{}',

    -- Before/after each iteration hooks
    before_each JSONB DEFAULT '{}',
    after_each JSONB DEFAULT '{}',

    -- Execution configuration
    iteration_mode TEXT DEFAULT 'sequential' CHECK (iteration_mode IN (
        'sequential',  -- Run one at a time in order
        'parallel',    -- Run multiple in parallel
        'random'       -- Random order (for load testing)
    )),
    max_parallel INTEGER DEFAULT 5,
    timeout_per_iteration_ms INTEGER DEFAULT 60000,
    stop_on_failure BOOLEAN DEFAULT false,
    retry_failed_iterations INTEGER DEFAULT 0,

    -- State
    is_active BOOLEAN DEFAULT true,
    last_run_at TIMESTAMPTZ,
    last_run_status TEXT,

    -- Metadata
    created_by UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_parameterized_tests_project ON parameterized_tests(project_id);
CREATE INDEX IF NOT EXISTS idx_parameterized_tests_base ON parameterized_tests(base_test_id);
CREATE INDEX IF NOT EXISTS idx_parameterized_tests_tags ON parameterized_tests USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_parameterized_tests_active ON parameterized_tests(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_parameterized_tests_created ON parameterized_tests(created_at DESC);

-- ============================================================================
-- PARAMETER SETS TABLE (Data for data-driven testing)
-- ============================================================================

CREATE TABLE IF NOT EXISTS parameter_sets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parameterized_test_id UUID NOT NULL REFERENCES parameterized_tests(id) ON DELETE CASCADE,

    -- Set identification
    name TEXT NOT NULL,
    description TEXT,

    -- Parameter values
    "values" JSONB NOT NULL,
    -- Example: {"email": "user1@test.com", "password": "pass123", "expected_name": "User One"}

    -- Categorization
    tags TEXT[] DEFAULT '{}',
    category TEXT,  -- e.g., "happy_path", "edge_case", "negative"

    -- Execution control
    skip BOOLEAN DEFAULT false,
    skip_reason TEXT,
    run_only BOOLEAN DEFAULT false,  -- If true, only this set runs (for debugging)

    -- Ordering
    order_index INTEGER DEFAULT 0,

    -- Expected outcome (for validation)
    expected_outcome TEXT DEFAULT 'pass' CHECK (expected_outcome IN ('pass', 'fail', 'skip')),
    expected_error TEXT,

    -- Environment-specific values
    environment_overrides JSONB DEFAULT '{}',
    -- Example: {"staging": {"base_url": "https://staging.example.com"}}

    -- Metadata
    source TEXT DEFAULT 'manual' CHECK (source IN ('manual', 'imported', 'generated')),
    source_reference TEXT,  -- e.g., CSV row number, API record ID
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_parameter_sets_test ON parameter_sets(parameterized_test_id);
CREATE INDEX IF NOT EXISTS idx_parameter_sets_order ON parameter_sets(parameterized_test_id, order_index);
CREATE INDEX IF NOT EXISTS idx_parameter_sets_tags ON parameter_sets USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_parameter_sets_category ON parameter_sets(category);
CREATE INDEX IF NOT EXISTS idx_parameter_sets_skip ON parameter_sets(skip) WHERE skip = false;

-- ============================================================================
-- PARAMETERIZED RESULTS TABLE (Aggregated results for parameterized test runs)
-- ============================================================================

CREATE TABLE IF NOT EXISTS parameterized_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parameterized_test_id UUID NOT NULL REFERENCES parameterized_tests(id) ON DELETE CASCADE,
    test_run_id UUID REFERENCES test_runs(id) ON DELETE SET NULL,
    schedule_run_id UUID REFERENCES schedule_runs(id) ON DELETE SET NULL,

    -- Overall statistics
    total_iterations INTEGER NOT NULL,
    passed INTEGER DEFAULT 0,
    failed INTEGER DEFAULT 0,
    skipped INTEGER DEFAULT 0,
    error INTEGER DEFAULT 0,

    -- Timing
    duration_ms INTEGER,
    avg_iteration_ms INTEGER,
    min_iteration_ms INTEGER,
    max_iteration_ms INTEGER,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- Execution mode used
    iteration_mode TEXT,
    parallel_workers INTEGER,

    -- Overall status
    status TEXT DEFAULT 'running' CHECK (status IN (
        'pending', 'running', 'passed', 'failed', 'cancelled', 'error'
    )),

    -- Per-iteration results
    iteration_results JSONB DEFAULT '[]',
    -- Example: [
    --   {"parameter_set_id": "uuid", "status": "passed", "duration_ms": 1200, "error": null},
    --   {"parameter_set_id": "uuid", "status": "failed", "duration_ms": 800, "error": "Expected 'Hello' but got 'Error'"}
    -- ]

    -- Failure analysis
    failure_summary JSONB DEFAULT '{}',
    -- Example: {"common_errors": [{"message": "...", "count": 5}], "failed_categories": ["edge_case"]}

    -- Environment info
    environment TEXT,
    browser TEXT,
    app_url TEXT,

    -- Triggered by
    triggered_by TEXT,
    trigger_type TEXT,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_parameterized_results_test ON parameterized_results(parameterized_test_id);
CREATE INDEX IF NOT EXISTS idx_parameterized_results_run ON parameterized_results(test_run_id);
CREATE INDEX IF NOT EXISTS idx_parameterized_results_schedule ON parameterized_results(schedule_run_id);
CREATE INDEX IF NOT EXISTS idx_parameterized_results_status ON parameterized_results(status);
CREATE INDEX IF NOT EXISTS idx_parameterized_results_started ON parameterized_results(started_at DESC);

-- ============================================================================
-- ITERATION RESULTS TABLE (Individual iteration outcomes)
-- ============================================================================

CREATE TABLE IF NOT EXISTS iteration_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parameterized_result_id UUID NOT NULL REFERENCES parameterized_results(id) ON DELETE CASCADE,
    parameter_set_id UUID REFERENCES parameter_sets(id) ON DELETE SET NULL,

    -- Iteration info
    iteration_index INTEGER NOT NULL,
    parameter_values JSONB NOT NULL,  -- Snapshot of values used

    -- Status
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'passed', 'failed', 'skipped', 'error')),

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,

    -- Step-by-step results
    step_results JSONB DEFAULT '[]',

    -- Error information
    error_message TEXT,
    error_stack TEXT,
    error_screenshot_url TEXT,

    -- Assertions
    assertions_passed INTEGER DEFAULT 0,
    assertions_failed INTEGER DEFAULT 0,
    assertion_details JSONB DEFAULT '[]',

    -- Retry tracking
    retry_count INTEGER DEFAULT 0,
    is_retry BOOLEAN DEFAULT false,
    original_iteration_id UUID REFERENCES iteration_results(id),

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_iteration_results_parent ON iteration_results(parameterized_result_id);
CREATE INDEX IF NOT EXISTS idx_iteration_results_set ON iteration_results(parameter_set_id);
CREATE INDEX IF NOT EXISTS idx_iteration_results_status ON iteration_results(status);
CREATE INDEX IF NOT EXISTS idx_iteration_results_order ON iteration_results(parameterized_result_id, iteration_index);

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE parameterized_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE parameter_sets ENABLE ROW LEVEL SECURITY;
ALTER TABLE parameterized_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE iteration_results ENABLE ROW LEVEL SECURITY;

-- Policies for parameterized_tests
DROP POLICY IF EXISTS "Users can view parameterized tests for their projects" ON parameterized_tests;
CREATE POLICY "Users can view parameterized tests for their projects" ON parameterized_tests
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = parameterized_tests.project_id
        )
    );

DROP POLICY IF EXISTS "Users can manage parameterized tests" ON parameterized_tests;
CREATE POLICY "Users can manage parameterized tests" ON parameterized_tests
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = parameterized_tests.project_id
        )
    );

-- Policies for parameter_sets
DROP POLICY IF EXISTS "Users can view parameter sets" ON parameter_sets;
CREATE POLICY "Users can view parameter sets" ON parameter_sets
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM parameterized_tests pt
            JOIN projects p ON p.id = pt.project_id
            WHERE pt.id = parameter_sets.parameterized_test_id
        )
    );

DROP POLICY IF EXISTS "Users can manage parameter sets" ON parameter_sets;
CREATE POLICY "Users can manage parameter sets" ON parameter_sets
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM parameterized_tests pt
            JOIN projects p ON p.id = pt.project_id
            WHERE pt.id = parameter_sets.parameterized_test_id
        )
    );

-- Policies for parameterized_results
DROP POLICY IF EXISTS "Users can view parameterized results" ON parameterized_results;
CREATE POLICY "Users can view parameterized results" ON parameterized_results
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM parameterized_tests pt
            JOIN projects p ON p.id = pt.project_id
            WHERE pt.id = parameterized_results.parameterized_test_id
        )
    );

DROP POLICY IF EXISTS "Users can manage parameterized results" ON parameterized_results;
CREATE POLICY "Users can manage parameterized results" ON parameterized_results
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM parameterized_tests pt
            JOIN projects p ON p.id = pt.project_id
            WHERE pt.id = parameterized_results.parameterized_test_id
        )
    );

-- Policies for iteration_results
DROP POLICY IF EXISTS "Users can view iteration results" ON iteration_results;
CREATE POLICY "Users can view iteration results" ON iteration_results
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM parameterized_results pr
            JOIN parameterized_tests pt ON pt.id = pr.parameterized_test_id
            JOIN projects p ON p.id = pt.project_id
            WHERE pr.id = iteration_results.parameterized_result_id
        )
    );

DROP POLICY IF EXISTS "Users can manage iteration results" ON iteration_results;
CREATE POLICY "Users can manage iteration results" ON iteration_results
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM parameterized_results pr
            JOIN parameterized_tests pt ON pt.id = pr.parameterized_test_id
            JOIN projects p ON p.id = pt.project_id
            WHERE pr.id = iteration_results.parameterized_result_id
        )
    );

-- Service role policies
DROP POLICY IF EXISTS "Service role has full access to parameterized_tests" ON parameterized_tests;
CREATE POLICY "Service role has full access to parameterized_tests" ON parameterized_tests
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role has full access to parameter_sets" ON parameter_sets;
CREATE POLICY "Service role has full access to parameter_sets" ON parameter_sets
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role has full access to parameterized_results" ON parameterized_results;
CREATE POLICY "Service role has full access to parameterized_results" ON parameterized_results
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role has full access to iteration_results" ON iteration_results;
CREATE POLICY "Service role has full access to iteration_results" ON iteration_results
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update updated_at triggers
CREATE TRIGGER update_parameterized_tests_updated_at
    BEFORE UPDATE ON parameterized_tests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_parameter_sets_updated_at
    BEFORE UPDATE ON parameter_sets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Update parameterized test stats after run completes
CREATE OR REPLACE FUNCTION update_parameterized_test_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status IN ('passed', 'failed', 'error', 'cancelled') AND OLD.status = 'running' THEN
        UPDATE parameterized_tests
        SET
            last_run_at = NEW.completed_at,
            last_run_status = NEW.status,
            updated_at = NOW()
        WHERE id = NEW.parameterized_test_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_parameterized_test_stats_trigger
    AFTER UPDATE ON parameterized_results
    FOR EACH ROW EXECUTE FUNCTION update_parameterized_test_stats();

-- Update parameterized_results stats when iteration_results change
CREATE OR REPLACE FUNCTION update_parameterized_result_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status IN ('passed', 'failed', 'skipped', 'error') THEN
        UPDATE parameterized_results
        SET
            passed = (SELECT COUNT(*) FROM iteration_results WHERE parameterized_result_id = NEW.parameterized_result_id AND status = 'passed'),
            failed = (SELECT COUNT(*) FROM iteration_results WHERE parameterized_result_id = NEW.parameterized_result_id AND status = 'failed'),
            skipped = (SELECT COUNT(*) FROM iteration_results WHERE parameterized_result_id = NEW.parameterized_result_id AND status = 'skipped'),
            error = (SELECT COUNT(*) FROM iteration_results WHERE parameterized_result_id = NEW.parameterized_result_id AND status = 'error')
        WHERE id = NEW.parameterized_result_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_parameterized_result_stats_trigger
    AFTER INSERT OR UPDATE ON iteration_results
    FOR EACH ROW EXECUTE FUNCTION update_parameterized_result_stats();

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to create a new parameterized test run
CREATE OR REPLACE FUNCTION create_parameterized_run(
    p_parameterized_test_id UUID,
    p_test_run_id UUID DEFAULT NULL,
    p_schedule_run_id UUID DEFAULT NULL,
    p_triggered_by TEXT DEFAULT NULL,
    p_trigger_type TEXT DEFAULT 'manual',
    p_environment TEXT DEFAULT 'staging',
    p_browser TEXT DEFAULT 'chromium',
    p_app_url TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_result_id UUID;
    v_total_iterations INTEGER;
    v_iteration_mode TEXT;
    v_max_parallel INTEGER;
BEGIN
    -- Get test configuration
    SELECT iteration_mode, max_parallel
    INTO v_iteration_mode, v_max_parallel
    FROM parameterized_tests
    WHERE id = p_parameterized_test_id;

    -- Count active parameter sets
    SELECT COUNT(*)
    INTO v_total_iterations
    FROM parameter_sets
    WHERE parameterized_test_id = p_parameterized_test_id
    AND skip = false;

    -- Create the result record
    INSERT INTO parameterized_results (
        parameterized_test_id,
        test_run_id,
        schedule_run_id,
        total_iterations,
        status,
        iteration_mode,
        parallel_workers,
        environment,
        browser,
        app_url,
        triggered_by,
        trigger_type
    ) VALUES (
        p_parameterized_test_id,
        p_test_run_id,
        p_schedule_run_id,
        v_total_iterations,
        'pending',
        v_iteration_mode,
        v_max_parallel,
        p_environment,
        p_browser,
        p_app_url,
        p_triggered_by,
        p_trigger_type
    )
    RETURNING id INTO v_result_id;

    RETURN v_result_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get parameter sets for execution
CREATE OR REPLACE FUNCTION get_executable_parameter_sets(
    p_parameterized_test_id UUID,
    p_environment TEXT DEFAULT NULL
)
RETURNS TABLE(
    set_id UUID,
    set_name TEXT,
    param_values JSONB,
    order_index INTEGER,
    expected_outcome TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ps.id as set_id,
        ps.name as set_name,
        CASE
            WHEN p_environment IS NOT NULL AND ps.environment_overrides ? p_environment
            THEN ps."values" || (ps.environment_overrides -> p_environment)
            ELSE ps."values"
        END as param_values,
        ps.order_index,
        ps.expected_outcome
    FROM parameter_sets ps
    WHERE ps.parameterized_test_id = p_parameterized_test_id
    AND ps.skip = false
    AND (
        NOT EXISTS (SELECT 1 FROM parameter_sets WHERE parameterized_test_id = p_parameterized_test_id AND run_only = true)
        OR ps.run_only = true
    )
    ORDER BY ps.order_index ASC;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate parameterized run statistics
CREATE OR REPLACE FUNCTION finalize_parameterized_run(p_result_id UUID)
RETURNS void AS $$
DECLARE
    v_passed INTEGER;
    v_failed INTEGER;
    v_skipped INTEGER;
    v_error INTEGER;
    v_total INTEGER;
    v_duration INTEGER;
    v_avg_duration INTEGER;
    v_min_duration INTEGER;
    v_max_duration INTEGER;
    v_status TEXT;
BEGIN
    -- Calculate statistics
    SELECT
        COUNT(*) FILTER (WHERE status = 'passed'),
        COUNT(*) FILTER (WHERE status = 'failed'),
        COUNT(*) FILTER (WHERE status = 'skipped'),
        COUNT(*) FILTER (WHERE status = 'error'),
        COUNT(*),
        COALESCE(SUM(duration_ms), 0),
        COALESCE(AVG(duration_ms)::INTEGER, 0),
        COALESCE(MIN(duration_ms), 0),
        COALESCE(MAX(duration_ms), 0)
    INTO v_passed, v_failed, v_skipped, v_error, v_total, v_duration, v_avg_duration, v_min_duration, v_max_duration
    FROM iteration_results
    WHERE parameterized_result_id = p_result_id;

    -- Determine overall status
    IF v_error > 0 THEN
        v_status := 'error';
    ELSIF v_failed > 0 THEN
        v_status := 'failed';
    ELSIF v_passed = v_total THEN
        v_status := 'passed';
    ELSE
        v_status := 'passed';  -- All passed or skipped
    END IF;

    -- Update the result
    UPDATE parameterized_results
    SET
        passed = v_passed,
        failed = v_failed,
        skipped = v_skipped,
        error = v_error,
        duration_ms = v_duration,
        avg_iteration_ms = v_avg_duration,
        min_iteration_ms = v_min_duration,
        max_iteration_ms = v_max_duration,
        status = v_status,
        completed_at = NOW()
    WHERE id = p_result_id;
END;
$$ LANGUAGE plpgsql;

-- Function to import parameter sets from JSON
CREATE OR REPLACE FUNCTION import_parameter_sets(
    p_parameterized_test_id UUID,
    p_data JSONB,
    p_source TEXT DEFAULT 'imported'
) RETURNS INTEGER AS $$
DECLARE
    v_item JSONB;
    v_count INTEGER := 0;
    v_index INTEGER := 0;
BEGIN
    FOR v_item IN SELECT jsonb_array_elements(p_data)
    LOOP
        INSERT INTO parameter_sets (
            parameterized_test_id,
            name,
            description,
            values,
            tags,
            category,
            order_index,
            source,
            source_reference
        ) VALUES (
            p_parameterized_test_id,
            COALESCE(v_item->>'name', 'Set ' || v_index),
            v_item->>'description',
            COALESCE(v_item->'values', v_item - 'name' - 'description' - 'tags' - 'category'),
            COALESCE(ARRAY(SELECT jsonb_array_elements_text(v_item->'tags')), ARRAY[]::TEXT[]),
            v_item->>'category',
            v_index,
            p_source,
            v_index::TEXT
        );
        v_count := v_count + 1;
        v_index := v_index + 1;
    END LOOP;

    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- ENABLE REALTIME
-- ============================================================================

ALTER PUBLICATION supabase_realtime ADD TABLE parameterized_results;
ALTER PUBLICATION supabase_realtime ADD TABLE iteration_results;

-- ============================================================================
-- COMPLETION
-- ============================================================================

SELECT 'Parameterized test tables created successfully!' as message;
