-- E2E Testing Agent Database Schema
-- Run this in Supabase SQL Editor

-- ============================================
-- PROJECTS (Apps being tested)
-- ============================================
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,  -- Clerk user ID
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    app_url TEXT NOT NULL,
    description TEXT,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, slug)
);

CREATE INDEX idx_projects_user ON projects(user_id);

-- ============================================
-- TESTS (Test definitions)
-- ============================================
CREATE TABLE tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    steps JSONB NOT NULL DEFAULT '[]',
    tags TEXT[] DEFAULT '{}',
    priority TEXT DEFAULT 'medium' CHECK (priority IN ('critical', 'high', 'medium', 'low')),
    is_active BOOLEAN DEFAULT true,
    source TEXT DEFAULT 'manual' CHECK (source IN ('manual', 'discovered', 'generated', 'imported')),
    created_by TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_tests_project ON tests(project_id);
CREATE INDEX idx_tests_tags ON tests USING GIN(tags);

-- ============================================
-- TEST RUNS (Execution instances)
-- ============================================
CREATE TABLE test_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT,
    trigger TEXT DEFAULT 'manual' CHECK (trigger IN ('manual', 'scheduled', 'webhook', 'ci')),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'passed', 'failed', 'cancelled')),
    app_url TEXT NOT NULL,
    environment TEXT DEFAULT 'staging',
    browser TEXT DEFAULT 'chromium',
    total_tests INTEGER DEFAULT 0,
    passed_tests INTEGER DEFAULT 0,
    failed_tests INTEGER DEFAULT 0,
    skipped_tests INTEGER DEFAULT 0,
    duration_ms INTEGER,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    triggered_by TEXT,
    ci_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_test_runs_project ON test_runs(project_id);
CREATE INDEX idx_test_runs_status ON test_runs(status);
CREATE INDEX idx_test_runs_created ON test_runs(created_at DESC);

-- ============================================
-- TEST RESULTS (Individual test outcomes)
-- ============================================
CREATE TABLE test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_run_id UUID NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    test_id UUID REFERENCES tests(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'passed', 'failed', 'skipped')),
    duration_ms INTEGER,
    error_message TEXT,
    error_stack TEXT,
    steps_completed INTEGER DEFAULT 0,
    steps_total INTEGER DEFAULT 0,
    step_results JSONB DEFAULT '[]',
    retry_count INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_test_results_run ON test_results(test_run_id);
CREATE INDEX idx_test_results_test ON test_results(test_id);

-- ============================================
-- DISCOVERY
-- ============================================
CREATE TABLE discovery_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    app_url TEXT NOT NULL,
    pages_found INTEGER DEFAULT 0,
    flows_found INTEGER DEFAULT 0,
    forms_found INTEGER DEFAULT 0,
    elements_found INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    triggered_by TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_discovery_sessions_project ON discovery_sessions(project_id);

CREATE TABLE discovered_pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discovery_session_id UUID NOT NULL REFERENCES discovery_sessions(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT,
    page_type TEXT,
    element_count INTEGER DEFAULT 0,
    form_count INTEGER DEFAULT 0,
    link_count INTEGER DEFAULT 0,
    screenshot_url TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(project_id, url)
);

CREATE INDEX idx_discovered_pages_project ON discovered_pages(project_id);

CREATE TABLE discovered_flows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discovery_session_id UUID NOT NULL REFERENCES discovery_sessions(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    steps JSONB NOT NULL DEFAULT '[]',
    step_count INTEGER DEFAULT 0,
    priority TEXT DEFAULT 'medium' CHECK (priority IN ('critical', 'high', 'medium', 'low')),
    converted_to_test_id UUID REFERENCES tests(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_discovered_flows_project ON discovered_flows(project_id);

-- ============================================
-- VISUAL REGRESSION
-- ============================================
CREATE TABLE visual_baselines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    selector TEXT,
    page_url TEXT NOT NULL,
    viewport TEXT DEFAULT '1920x1080',
    screenshot_url TEXT NOT NULL,
    screenshot_hash TEXT,
    is_active BOOLEAN DEFAULT true,
    created_by TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(project_id, name, viewport)
);

CREATE INDEX idx_visual_baselines_project ON visual_baselines(project_id);

CREATE TABLE visual_comparisons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    baseline_id UUID REFERENCES visual_baselines(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('match', 'mismatch', 'new', 'pending', 'error')),
    match_percentage DECIMAL(5,2),
    difference_count INTEGER DEFAULT 0,
    baseline_url TEXT,
    current_url TEXT NOT NULL,
    diff_url TEXT,
    threshold DECIMAL(5,2) DEFAULT 0.1,
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_visual_comparisons_project ON visual_comparisons(project_id);

-- ============================================
-- AI INSIGHTS
-- ============================================
CREATE TABLE ai_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    insight_type TEXT NOT NULL CHECK (insight_type IN ('prediction', 'anomaly', 'suggestion', 'understanding')),
    severity TEXT NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low', 'info')),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    affected_area TEXT,
    suggested_action TEXT,
    action_url TEXT,
    related_test_ids UUID[],
    is_resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMPTZ,
    resolved_by TEXT,
    expires_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ai_insights_project ON ai_insights(project_id);
CREATE INDEX idx_ai_insights_unresolved ON ai_insights(project_id, is_resolved) WHERE is_resolved = false;

-- ============================================
-- QUALITY AUDITS
-- ============================================
CREATE TABLE quality_audits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    accessibility_score INTEGER,
    performance_score INTEGER,
    best_practices_score INTEGER,
    seo_score INTEGER,
    lcp_ms INTEGER,
    fid_ms INTEGER,
    cls DECIMAL(4,3),
    ttfb_ms INTEGER,
    fcp_ms INTEGER,
    tti_ms INTEGER,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    triggered_by TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_quality_audits_project ON quality_audits(project_id);

CREATE TABLE accessibility_issues (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id UUID NOT NULL REFERENCES quality_audits(id) ON DELETE CASCADE,
    rule TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('critical', 'serious', 'moderate', 'minor')),
    element_selector TEXT,
    description TEXT NOT NULL,
    suggested_fix TEXT,
    wcag_criteria TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_accessibility_issues_audit ON accessibility_issues(audit_id);

-- ============================================
-- GLOBAL TESTING
-- ============================================
CREATE TABLE global_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    avg_latency_ms INTEGER,
    avg_ttfb_ms INTEGER,
    success_rate DECIMAL(5,2),
    slow_regions INTEGER DEFAULT 0,
    failed_regions INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    triggered_by TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_global_tests_project ON global_tests(project_id);

CREATE TABLE global_test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    global_test_id UUID NOT NULL REFERENCES global_tests(id) ON DELETE CASCADE,
    region_code TEXT NOT NULL,
    city TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('success', 'error', 'slow', 'timeout')),
    latency_ms INTEGER,
    ttfb_ms INTEGER,
    page_load_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_global_test_results_test ON global_test_results(global_test_id);

-- ============================================
-- INTEGRATIONS
-- ============================================
CREATE TABLE integrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    platform_type TEXT NOT NULL CHECK (platform_type IN ('cicd', 'observability', 'notification')),
    is_connected BOOLEAN DEFAULT false,
    config JSONB DEFAULT '{}',
    last_sync_at TIMESTAMPTZ,
    sync_status TEXT,
    data_points_synced INTEGER DEFAULT 0,
    features_enabled TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, platform)
);

CREATE INDEX idx_integrations_user ON integrations(user_id);

-- ============================================
-- STATISTICS
-- ============================================
CREATE TABLE project_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID UNIQUE NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    total_tests INTEGER DEFAULT 0,
    total_runs INTEGER DEFAULT 0,
    total_passed INTEGER DEFAULT 0,
    total_failed INTEGER DEFAULT 0,
    avg_pass_rate DECIMAL(5,2) DEFAULT 0,
    avg_duration_ms INTEGER DEFAULT 0,
    last_run_at TIMESTAMPTZ,
    last_run_status TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE daily_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    runs INTEGER DEFAULT 0,
    passed INTEGER DEFAULT 0,
    failed INTEGER DEFAULT 0,
    avg_duration_ms INTEGER DEFAULT 0,
    UNIQUE(project_id, date)
);

CREATE INDEX idx_daily_stats_project_date ON daily_stats(project_id, date DESC);

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE test_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE test_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovery_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovered_pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovered_flows ENABLE ROW LEVEL SECURITY;
ALTER TABLE visual_baselines ENABLE ROW LEVEL SECURITY;
ALTER TABLE visual_comparisons ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE quality_audits ENABLE ROW LEVEL SECURITY;
ALTER TABLE accessibility_issues ENABLE ROW LEVEL SECURITY;
ALTER TABLE global_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE global_test_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE integrations ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_stats ENABLE ROW LEVEL SECURITY;

-- RLS Policies - Allow all for now (can be restricted later with Clerk JWT)
CREATE POLICY "Enable all for authenticated users" ON projects FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON tests FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON test_runs FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON test_results FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON discovery_sessions FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON discovered_pages FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON discovered_flows FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON visual_baselines FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON visual_comparisons FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON ai_insights FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON quality_audits FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON accessibility_issues FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON global_tests FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON global_test_results FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON integrations FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON project_stats FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON daily_stats FOR ALL USING (true);

-- ============================================
-- FUNCTIONS & TRIGGERS
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_tests_updated_at
    BEFORE UPDATE ON tests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_integrations_updated_at
    BEFORE UPDATE ON integrations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Auto-create project_stats when project is created
CREATE OR REPLACE FUNCTION create_project_stats()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO project_stats (project_id) VALUES (NEW.id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER create_project_stats_trigger
    AFTER INSERT ON projects
    FOR EACH ROW EXECUTE FUNCTION create_project_stats();

-- Enable realtime for key tables
ALTER PUBLICATION supabase_realtime ADD TABLE test_runs;
ALTER PUBLICATION supabase_realtime ADD TABLE test_results;
ALTER PUBLICATION supabase_realtime ADD TABLE discovery_sessions;
ALTER PUBLICATION supabase_realtime ADD TABLE discovered_pages;

-- ============================================
-- CHAT CONVERSATIONS
-- ============================================
CREATE TABLE chat_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    title TEXT,
    preview TEXT,
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chat_conversations_user ON chat_conversations(user_id);
CREATE INDEX idx_chat_conversations_project ON chat_conversations(project_id);
CREATE INDEX idx_chat_conversations_updated ON chat_conversations(updated_at DESC);

-- ============================================
-- CHAT MESSAGES
-- ============================================
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    tool_invocations JSONB DEFAULT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chat_messages_conversation ON chat_messages(conversation_id);
CREATE INDEX idx_chat_messages_created ON chat_messages(conversation_id, created_at);

ALTER TABLE chat_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Enable all for authenticated users" ON chat_conversations FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON chat_messages FOR ALL USING (true);

CREATE TRIGGER update_chat_conversations_updated_at
    BEFORE UPDATE ON chat_conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE OR REPLACE FUNCTION update_conversation_message_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE chat_conversations
    SET message_count = message_count + 1,
        updated_at = NOW(),
        preview = CASE
            WHEN NEW.role = 'user' THEN LEFT(NEW.content, 100)
            ELSE preview
        END
    WHERE id = NEW.conversation_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_message_count_trigger
    AFTER INSERT ON chat_messages
    FOR EACH ROW EXECUTE FUNCTION update_conversation_message_count();

CREATE OR REPLACE FUNCTION set_conversation_title()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.role = 'user' THEN
        UPDATE chat_conversations
        SET title = COALESCE(title, LEFT(NEW.content, 50))
        WHERE id = NEW.conversation_id AND title IS NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_conversation_title_trigger
    AFTER INSERT ON chat_messages
    FOR EACH ROW EXECUTE FUNCTION set_conversation_title();

ALTER PUBLICATION supabase_realtime ADD TABLE chat_conversations;
ALTER PUBLICATION supabase_realtime ADD TABLE chat_messages;

-- Done!
SELECT 'Schema created successfully!' as message;
