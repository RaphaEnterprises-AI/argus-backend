-- ============================================
-- QUALITY INTELLIGENCE PLATFORM
-- Production-to-Test Automation Loop
-- ============================================

-- ============================================
-- PRODUCTION EVENTS (from Sentry, Datadog, etc.)
-- ============================================
CREATE TABLE production_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    integration_id UUID REFERENCES integrations(id) ON DELETE SET NULL,

    -- Source information
    source TEXT NOT NULL CHECK (source IN ('sentry', 'datadog', 'fullstory', 'logrocket', 'newrelic', 'bugsnag', 'rollbar', 'custom')),
    external_id TEXT NOT NULL,  -- ID from the source platform
    external_url TEXT,  -- Link to the event in the source platform

    -- Event classification
    event_type TEXT NOT NULL CHECK (event_type IN ('error', 'exception', 'crash', 'performance', 'user_feedback', 'rage_click', 'dead_click')),
    severity TEXT NOT NULL CHECK (severity IN ('fatal', 'error', 'warning', 'info')),

    -- Error details
    title TEXT NOT NULL,
    message TEXT,
    stack_trace TEXT,
    fingerprint TEXT,  -- Unique identifier for grouping similar errors

    -- Context
    url TEXT,  -- Page URL where error occurred
    component TEXT,  -- React/Vue component name if available
    user_action TEXT,  -- What user was doing when error occurred
    browser TEXT,
    os TEXT,
    device_type TEXT CHECK (device_type IN ('desktop', 'mobile', 'tablet')),

    -- Frequency & impact
    occurrence_count INTEGER DEFAULT 1,
    affected_users INTEGER DEFAULT 1,
    first_seen_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ DEFAULT NOW(),

    -- AI Analysis
    ai_analysis JSONB DEFAULT NULL,  -- Claude's analysis of the error
    root_cause TEXT,  -- AI-identified root cause
    suggested_fix TEXT,  -- AI-suggested code fix

    -- Status
    status TEXT DEFAULT 'new' CHECK (status IN ('new', 'analyzing', 'test_generated', 'test_pending_review', 'resolved', 'ignored', 'wont_fix')),
    resolved_at TIMESTAMPTZ,
    resolved_by TEXT,

    -- Metadata
    raw_payload JSONB DEFAULT '{}',  -- Original webhook payload
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(project_id, source, external_id)
);

CREATE INDEX idx_production_events_project ON production_events(project_id);
CREATE INDEX idx_production_events_source ON production_events(source);
CREATE INDEX idx_production_events_status ON production_events(status);
CREATE INDEX idx_production_events_severity ON production_events(severity);
CREATE INDEX idx_production_events_fingerprint ON production_events(fingerprint);
CREATE INDEX idx_production_events_created ON production_events(created_at DESC);
CREATE INDEX idx_production_events_url ON production_events(url);
CREATE INDEX idx_production_events_component ON production_events(component);

-- ============================================
-- ERROR PATTERNS (for cross-company learning)
-- ============================================
CREATE TABLE error_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Pattern identification
    pattern_hash TEXT UNIQUE NOT NULL,  -- Hash of the anonymized pattern
    pattern_type TEXT NOT NULL CHECK (pattern_type IN ('stack_trace', 'error_message', 'user_flow', 'component')),

    -- Anonymized pattern data (no company-specific info)
    pattern_signature JSONB NOT NULL,  -- Anonymized pattern structure
    example_message TEXT,  -- Generic example message

    -- Classification
    category TEXT NOT NULL,  -- e.g., 'null_reference', 'network_error', 'validation_error'
    subcategory TEXT,
    frameworks TEXT[] DEFAULT '{}',  -- e.g., ['react', 'nextjs']

    -- Learning data (aggregated across all companies)
    total_occurrences INTEGER DEFAULT 1,
    companies_affected INTEGER DEFAULT 1,  -- Count, not identifiers

    -- AI-generated knowledge
    common_root_causes TEXT[] DEFAULT '{}',
    common_fixes TEXT[] DEFAULT '{}',
    prevention_strategies TEXT[] DEFAULT '{}',
    test_templates JSONB DEFAULT '[]',  -- Generic test templates

    -- Quality metrics
    fix_success_rate DECIMAL(5,2),  -- How often generated tests prevent recurrence
    avg_time_to_fix_ms BIGINT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_error_patterns_hash ON error_patterns(pattern_hash);
CREATE INDEX idx_error_patterns_category ON error_patterns(category);
CREATE INDEX idx_error_patterns_type ON error_patterns(pattern_type);

-- ============================================
-- GENERATED TESTS (AI-created from production errors)
-- ============================================
CREATE TABLE generated_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    production_event_id UUID REFERENCES production_events(id) ON DELETE SET NULL,
    error_pattern_id UUID REFERENCES error_patterns(id) ON DELETE SET NULL,

    -- Test details
    name TEXT NOT NULL,
    description TEXT,
    test_type TEXT NOT NULL CHECK (test_type IN ('e2e', 'integration', 'unit', 'visual', 'accessibility')),
    framework TEXT DEFAULT 'playwright',  -- playwright, cypress, jest, etc.

    -- Generated code
    test_code TEXT NOT NULL,
    test_file_path TEXT,  -- Suggested file path

    -- Steps (for UI tests)
    steps JSONB DEFAULT '[]',
    assertions JSONB DEFAULT '[]',

    -- Confidence & quality
    confidence_score DECIMAL(3,2) NOT NULL,  -- 0.00 to 1.00
    quality_score DECIMAL(3,2),  -- After review

    -- Review status
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'modified', 'deployed')),
    reviewed_by TEXT,
    reviewed_at TIMESTAMPTZ,
    review_notes TEXT,

    -- Deployment
    github_pr_url TEXT,
    github_pr_number INTEGER,
    github_pr_status TEXT CHECK (github_pr_status IN ('draft', 'open', 'merged', 'closed')),
    deployed_at TIMESTAMPTZ,

    -- Converted to permanent test
    converted_to_test_id UUID REFERENCES tests(id) ON DELETE SET NULL,

    -- Metrics (after deployment)
    times_run INTEGER DEFAULT 0,
    times_passed INTEGER DEFAULT 0,
    times_failed INTEGER DEFAULT 0,
    prevented_incidents INTEGER DEFAULT 0,  -- Times this test caught the original error

    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_generated_tests_project ON generated_tests(project_id);
CREATE INDEX idx_generated_tests_event ON generated_tests(production_event_id);
CREATE INDEX idx_generated_tests_status ON generated_tests(status);
CREATE INDEX idx_generated_tests_created ON generated_tests(created_at DESC);

-- ============================================
-- ERROR TO TEST MAPPINGS
-- ============================================
CREATE TABLE error_test_mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    production_event_id UUID NOT NULL REFERENCES production_events(id) ON DELETE CASCADE,
    test_id UUID REFERENCES tests(id) ON DELETE SET NULL,
    generated_test_id UUID REFERENCES generated_tests(id) ON DELETE SET NULL,

    -- Mapping type
    mapping_type TEXT NOT NULL CHECK (mapping_type IN ('coverage', 'prevention', 'regression')),

    -- Confidence that this test covers/prevents this error
    confidence DECIMAL(3,2) NOT NULL,

    -- Verification
    verified_by TEXT,
    verified_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Either test_id or generated_test_id must be set
    CONSTRAINT at_least_one_test CHECK (test_id IS NOT NULL OR generated_test_id IS NOT NULL)
);

CREATE INDEX idx_error_test_mappings_event ON error_test_mappings(production_event_id);
CREATE INDEX idx_error_test_mappings_test ON error_test_mappings(test_id);
CREATE INDEX idx_error_test_mappings_generated ON error_test_mappings(generated_test_id);

-- ============================================
-- RISK SCORES (for components/pages/flows)
-- ============================================
CREATE TABLE risk_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- What we're scoring
    entity_type TEXT NOT NULL CHECK (entity_type IN ('page', 'component', 'flow', 'endpoint', 'feature')),
    entity_identifier TEXT NOT NULL,  -- URL, component name, flow name, etc.
    entity_name TEXT,

    -- Risk scores (0-100)
    overall_risk_score INTEGER NOT NULL CHECK (overall_risk_score >= 0 AND overall_risk_score <= 100),
    error_frequency_score INTEGER DEFAULT 0,
    error_severity_score INTEGER DEFAULT 0,
    test_coverage_score INTEGER DEFAULT 0,  -- Inverse - low coverage = high risk
    change_frequency_score INTEGER DEFAULT 0,  -- From git
    user_impact_score INTEGER DEFAULT 0,  -- Based on traffic/usage

    -- Contributing factors
    factors JSONB DEFAULT '{}',

    -- Trend
    previous_score INTEGER,
    score_trend TEXT CHECK (score_trend IN ('improving', 'stable', 'degrading')),

    -- Recommendations
    recommendations JSONB DEFAULT '[]',
    priority_tests_needed TEXT[] DEFAULT '{}',

    -- Calculation metadata
    last_calculated_at TIMESTAMPTZ DEFAULT NOW(),
    calculation_version INTEGER DEFAULT 1,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(project_id, entity_type, entity_identifier)
);

CREATE INDEX idx_risk_scores_project ON risk_scores(project_id);
CREATE INDEX idx_risk_scores_type ON risk_scores(entity_type);
CREATE INDEX idx_risk_scores_overall ON risk_scores(overall_risk_score DESC);

-- ============================================
-- TEST GENERATION JOBS
-- ============================================
CREATE TABLE test_generation_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    production_event_id UUID REFERENCES production_events(id) ON DELETE SET NULL,

    -- Job status
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),

    -- Job type
    job_type TEXT NOT NULL CHECK (job_type IN ('single_error', 'pattern_batch', 'coverage_gap', 'scheduled')),

    -- AI processing
    ai_model TEXT DEFAULT 'claude-sonnet-4-5-20250514',
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_cost_cents INTEGER DEFAULT 0,

    -- Results
    tests_generated INTEGER DEFAULT 0,
    tests_approved INTEGER DEFAULT 0,
    error_message TEXT,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,

    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_test_generation_jobs_project ON test_generation_jobs(project_id);
CREATE INDEX idx_test_generation_jobs_status ON test_generation_jobs(status);
CREATE INDEX idx_test_generation_jobs_event ON test_generation_jobs(production_event_id);

-- ============================================
-- WEBHOOK LOGS (for debugging integrations)
-- ============================================
CREATE TABLE webhook_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    integration_id UUID REFERENCES integrations(id) ON DELETE SET NULL,

    -- Request details
    source TEXT NOT NULL,  -- sentry, datadog, etc.
    method TEXT DEFAULT 'POST',
    headers JSONB DEFAULT '{}',
    body JSONB DEFAULT '{}',

    -- Processing
    status TEXT DEFAULT 'received' CHECK (status IN ('received', 'processing', 'processed', 'failed', 'ignored')),
    processed_event_id UUID REFERENCES production_events(id) ON DELETE SET NULL,
    error_message TEXT,

    -- Timing
    received_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    processing_ms INTEGER
);

CREATE INDEX idx_webhook_logs_project ON webhook_logs(project_id);
CREATE INDEX idx_webhook_logs_source ON webhook_logs(source);
CREATE INDEX idx_webhook_logs_status ON webhook_logs(status);
CREATE INDEX idx_webhook_logs_received ON webhook_logs(received_at DESC);

-- Automatically clean old webhook logs (keep 7 days)
CREATE OR REPLACE FUNCTION cleanup_old_webhook_logs()
RETURNS TRIGGER AS $$
BEGIN
    DELETE FROM webhook_logs WHERE received_at < NOW() - INTERVAL '7 days';
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Run cleanup on each insert (simple approach, could use pg_cron for production)
CREATE TRIGGER cleanup_webhook_logs_trigger
    AFTER INSERT ON webhook_logs
    FOR EACH STATEMENT EXECUTE FUNCTION cleanup_old_webhook_logs();

-- ============================================
-- QUALITY INTELLIGENCE STATS
-- ============================================
CREATE TABLE quality_intelligence_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID UNIQUE NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Event counts
    total_production_events INTEGER DEFAULT 0,
    events_last_24h INTEGER DEFAULT 0,
    events_last_7d INTEGER DEFAULT 0,
    events_last_30d INTEGER DEFAULT 0,

    -- Test generation
    tests_generated INTEGER DEFAULT 0,
    tests_approved INTEGER DEFAULT 0,
    tests_deployed INTEGER DEFAULT 0,

    -- Impact metrics
    incidents_prevented INTEGER DEFAULT 0,
    estimated_time_saved_hours DECIMAL(10,2) DEFAULT 0,

    -- Coverage
    high_risk_components INTEGER DEFAULT 0,
    components_with_tests INTEGER DEFAULT 0,
    coverage_improvement_percent DECIMAL(5,2) DEFAULT 0,

    -- Integration health
    active_integrations INTEGER DEFAULT 0,
    last_event_received_at TIMESTAMPTZ,

    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================
ALTER TABLE production_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE error_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE generated_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE error_test_mappings ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE test_generation_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE webhook_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE quality_intelligence_stats ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Enable all for authenticated users" ON production_events FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON error_patterns FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON generated_tests FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON error_test_mappings FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON risk_scores FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON test_generation_jobs FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON webhook_logs FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON quality_intelligence_stats FOR ALL USING (true);

-- ============================================
-- TRIGGERS
-- ============================================
CREATE TRIGGER update_production_events_updated_at
    BEFORE UPDATE ON production_events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_error_patterns_updated_at
    BEFORE UPDATE ON error_patterns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_generated_tests_updated_at
    BEFORE UPDATE ON generated_tests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_risk_scores_updated_at
    BEFORE UPDATE ON risk_scores
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Auto-create quality_intelligence_stats when project is created
CREATE OR REPLACE FUNCTION create_quality_intelligence_stats()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO quality_intelligence_stats (project_id)
    VALUES (NEW.id)
    ON CONFLICT (project_id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER create_quality_intelligence_stats_trigger
    AFTER INSERT ON projects
    FOR EACH ROW EXECUTE FUNCTION create_quality_intelligence_stats();

-- Update stats when production event is added
CREATE OR REPLACE FUNCTION update_quality_stats_on_event()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE quality_intelligence_stats
    SET
        total_production_events = total_production_events + 1,
        last_event_received_at = NOW(),
        updated_at = NOW()
    WHERE project_id = NEW.project_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_quality_stats_on_event_trigger
    AFTER INSERT ON production_events
    FOR EACH ROW EXECUTE FUNCTION update_quality_stats_on_event();

-- Enable realtime for key tables
ALTER PUBLICATION supabase_realtime ADD TABLE production_events;
ALTER PUBLICATION supabase_realtime ADD TABLE generated_tests;
ALTER PUBLICATION supabase_realtime ADD TABLE risk_scores;

-- ============================================
-- Done!
-- ============================================
SELECT 'Quality Intelligence Platform schema created successfully!' as message;
