-- CI/CD Events and Coverage Reports tables migration
-- Adds tables for tracking CI/CD pipeline events and coverage reports

-- CI Events table (GitHub Actions, GitLab CI, CircleCI, Jenkins)
CREATE TABLE IF NOT EXISTS ci_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK (source IN ('github_actions', 'gitlab_ci', 'circleci', 'jenkins')),
    external_id TEXT NOT NULL,
    external_url TEXT,
    event_type TEXT NOT NULL CHECK (event_type IN ('workflow_run', 'workflow_job', 'test_run', 'coverage_report')),
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'success', 'failure', 'cancelled', 'skipped')),
    workflow_name TEXT NOT NULL,
    branch TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    run_number INTEGER NOT NULL DEFAULT 0,
    duration_seconds INTEGER,
    test_results JSONB,
    coverage_percent NUMERIC(5,2),
    raw_payload JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Coverage Reports table
CREATE TABLE IF NOT EXISTS coverage_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    branch TEXT NOT NULL DEFAULT 'main',
    commit_sha TEXT NOT NULL,
    ci_run_id UUID REFERENCES ci_events(id),
    format TEXT NOT NULL CHECK (format IN ('lcov', 'istanbul', 'cobertura', 'clover')),
    lines_total INTEGER NOT NULL DEFAULT 0,
    lines_covered INTEGER NOT NULL DEFAULT 0,
    lines_percent NUMERIC(5,2) NOT NULL DEFAULT 0,
    branches_total INTEGER NOT NULL DEFAULT 0,
    branches_covered INTEGER NOT NULL DEFAULT 0,
    branches_percent NUMERIC(5,2) NOT NULL DEFAULT 0,
    functions_total INTEGER NOT NULL DEFAULT 0,
    functions_covered INTEGER NOT NULL DEFAULT 0,
    functions_percent NUMERIC(5,2) NOT NULL DEFAULT 0,
    files JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Webhook Logs table (for debugging)
-- Note: Table may already exist from previous migrations
CREATE TABLE IF NOT EXISTS webhook_logs (
    id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    method TEXT NOT NULL,
    headers JSONB,
    body JSONB,
    status TEXT NOT NULL DEFAULT 'processing',
    error_message TEXT,
    processed_event_id UUID,
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Add created_at column if missing (for existing tables)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'webhook_logs' AND column_name = 'created_at') THEN
        ALTER TABLE webhook_logs ADD COLUMN created_at TIMESTAMPTZ DEFAULT now();
    END IF;
END $$;

-- Quality Scores table (aggregated quality metrics per project)
CREATE TABLE IF NOT EXISTS quality_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    overall_score INTEGER NOT NULL DEFAULT 50 CHECK (overall_score >= 0 AND overall_score <= 100),

    -- Component scores (0-100)
    coverage_score INTEGER DEFAULT 50,
    error_score INTEGER DEFAULT 50,
    flaky_score INTEGER DEFAULT 50,
    ci_score INTEGER DEFAULT 50,

    -- Raw metrics
    coverage_percent NUMERIC(5,2) DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    error_count_24h INTEGER DEFAULT 0,
    flaky_test_count INTEGER DEFAULT 0,
    ci_success_rate NUMERIC(5,2) DEFAULT 0,

    -- Trend
    trend TEXT DEFAULT 'stable' CHECK (trend IN ('improving', 'stable', 'declining')),
    previous_score INTEGER,

    -- Factors breakdown
    factors JSONB DEFAULT '{}',

    calculated_at TIMESTAMPTZ DEFAULT now(),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(project_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ci_events_project ON ci_events(project_id);
CREATE INDEX IF NOT EXISTS idx_ci_events_status ON ci_events(status);
CREATE INDEX IF NOT EXISTS idx_ci_events_created ON ci_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ci_events_commit ON ci_events(commit_sha);

CREATE INDEX IF NOT EXISTS idx_coverage_project ON coverage_reports(project_id);
CREATE INDEX IF NOT EXISTS idx_coverage_branch ON coverage_reports(branch);
CREATE INDEX IF NOT EXISTS idx_coverage_created ON coverage_reports(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_webhook_logs_source ON webhook_logs(source);
CREATE INDEX IF NOT EXISTS idx_webhook_logs_status ON webhook_logs(status);
CREATE INDEX IF NOT EXISTS idx_webhook_logs_created ON webhook_logs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_quality_scores_project ON quality_scores(project_id);
CREATE INDEX IF NOT EXISTS idx_quality_scores_overall ON quality_scores(overall_score DESC);

-- Enable RLS
ALTER TABLE ci_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE coverage_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE webhook_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE quality_scores ENABLE ROW LEVEL SECURITY;

-- Policies (allow all for now, can be restricted later)
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON ci_events;
CREATE POLICY "Enable all access for authenticated users" ON ci_events FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON coverage_reports;
CREATE POLICY "Enable all access for authenticated users" ON coverage_reports FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON webhook_logs;
CREATE POLICY "Enable all access for authenticated users" ON webhook_logs FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON quality_scores;
CREATE POLICY "Enable all access for authenticated users" ON quality_scores FOR ALL USING (true);

-- Function to update quality score automatically
CREATE OR REPLACE FUNCTION calculate_quality_score(p_project_id UUID)
RETURNS INTEGER AS $$
DECLARE
    v_coverage_percent NUMERIC;
    v_error_count INTEGER;
    v_error_count_24h INTEGER;
    v_ci_success_rate NUMERIC;
    v_flaky_count INTEGER;
    v_coverage_score INTEGER;
    v_error_score INTEGER;
    v_ci_score INTEGER;
    v_flaky_score INTEGER;
    v_overall_score INTEGER;
BEGIN
    -- Get latest coverage
    SELECT COALESCE(lines_percent, 0) INTO v_coverage_percent
    FROM coverage_reports
    WHERE project_id = p_project_id
    ORDER BY created_at DESC
    LIMIT 1;

    -- Get error count
    SELECT COUNT(*) INTO v_error_count
    FROM production_events
    WHERE project_id = p_project_id AND status = 'new';

    -- Get error count in last 24h
    SELECT COUNT(*) INTO v_error_count_24h
    FROM production_events
    WHERE project_id = p_project_id
    AND created_at > now() - interval '24 hours';

    -- Get CI success rate (last 20 runs)
    SELECT COALESCE(
        (COUNT(*) FILTER (WHERE status = 'success')::NUMERIC / NULLIF(COUNT(*), 0) * 100),
        0
    ) INTO v_ci_success_rate
    FROM (
        SELECT status FROM ci_events
        WHERE project_id = p_project_id
        ORDER BY created_at DESC
        LIMIT 20
    ) recent_runs;

    -- Calculate component scores
    v_coverage_score := LEAST(100, GREATEST(0, v_coverage_percent::INTEGER));
    v_error_score := GREATEST(0, 100 - (v_error_count_24h * 10));
    v_ci_score := v_ci_success_rate::INTEGER;
    v_flaky_score := 100; -- TODO: Calculate from flaky test data

    -- Calculate overall score (weighted average)
    v_overall_score := (
        v_coverage_score * 30 +
        v_error_score * 30 +
        v_ci_score * 25 +
        v_flaky_score * 15
    ) / 100;

    -- Upsert quality score
    INSERT INTO quality_scores (
        project_id, overall_score, coverage_score, error_score, ci_score, flaky_score,
        coverage_percent, error_count, error_count_24h, ci_success_rate,
        factors, calculated_at
    ) VALUES (
        p_project_id, v_overall_score, v_coverage_score, v_error_score, v_ci_score, v_flaky_score,
        v_coverage_percent, v_error_count, v_error_count_24h, v_ci_success_rate,
        jsonb_build_object(
            'coverage_weight', 30,
            'error_weight', 30,
            'ci_weight', 25,
            'flaky_weight', 15
        ),
        now()
    )
    ON CONFLICT (project_id) DO UPDATE SET
        overall_score = EXCLUDED.overall_score,
        coverage_score = EXCLUDED.coverage_score,
        error_score = EXCLUDED.error_score,
        ci_score = EXCLUDED.ci_score,
        flaky_score = EXCLUDED.flaky_score,
        coverage_percent = EXCLUDED.coverage_percent,
        error_count = EXCLUDED.error_count,
        error_count_24h = EXCLUDED.error_count_24h,
        ci_success_rate = EXCLUDED.ci_success_rate,
        factors = EXCLUDED.factors,
        previous_score = quality_scores.overall_score,
        trend = CASE
            WHEN EXCLUDED.overall_score > quality_scores.overall_score THEN 'improving'
            WHEN EXCLUDED.overall_score < quality_scores.overall_score THEN 'declining'
            ELSE 'stable'
        END,
        calculated_at = now(),
        updated_at = now();

    RETURN v_overall_score;
END;
$$ LANGUAGE plpgsql;
