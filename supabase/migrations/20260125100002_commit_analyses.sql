-- ============================================================================
-- Commit Analysis Schema
-- Stores GitHub commit/PR analysis results for the Quality Intelligence Platform
-- ============================================================================

-- =============================================================================
-- SDLC Events Table (Unified Timeline)
-- Stores events from all integrated platforms for cross-correlation
-- =============================================================================

CREATE TABLE IF NOT EXISTS sdlc_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Event classification
    event_type TEXT NOT NULL CHECK (event_type IN (
        'commit',           -- Git commits
        'pr',               -- Pull requests
        'push',             -- Push events
        'build',            -- CI/CD builds
        'test_run',         -- Test executions
        'deploy',           -- Deployments
        'deployment_status', -- Deployment status updates
        'check_run',        -- GitHub check runs
        'error',            -- Production errors
        'incident',         -- Incidents
        'feature_flag',     -- Feature flag changes
        'session',          -- User sessions
        'requirement'       -- Jira/Linear tickets
    )),

    -- Source identification
    source_platform TEXT NOT NULL CHECK (source_platform IN (
        'github', 'gitlab', 'bitbucket', 'jira', 'linear', 'sentry',
        'datadog', 'pagerduty', 'opsgenie', 'launchdarkly', 'argus'
    )),
    external_id TEXT NOT NULL,
    external_url TEXT,

    -- Event details
    title TEXT,
    description TEXT,

    -- Correlation keys (for linking events)
    commit_sha TEXT,
    pr_number INTEGER,
    jira_key TEXT,
    deploy_id TEXT,
    branch_name TEXT,

    -- Timing
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Flexible data storage
    data JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',

    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Prevent duplicate events
    UNIQUE(project_id, source_platform, external_id)
);

-- =============================================================================
-- Event Correlations Table
-- Links related events across platforms
-- =============================================================================

CREATE TABLE IF NOT EXISTS event_correlations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Correlation pair
    source_event_id UUID NOT NULL REFERENCES sdlc_events(id) ON DELETE CASCADE,
    target_event_id UUID NOT NULL REFERENCES sdlc_events(id) ON DELETE CASCADE,

    -- Correlation type
    correlation_type TEXT NOT NULL CHECK (correlation_type IN (
        'caused_by',        -- Target was caused by source
        'introduced_by',    -- Issue introduced by this event
        'related_to',       -- General relationship
        'blocks',           -- Source blocks target
        'depends_on',       -- Source depends on target
        'fixes',            -- Source fixes issue in target
        'reverts'           -- Source reverts target
    )),

    -- Confidence
    confidence DECIMAL(3,2) DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    correlation_method TEXT, -- 'automatic', 'manual', 'ai'

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(source_event_id, target_event_id, correlation_type)
);

-- =============================================================================
-- Correlation Insights Table
-- AI-generated insights from correlation analysis
-- =============================================================================

CREATE TABLE IF NOT EXISTS correlation_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Insight classification
    insight_type TEXT NOT NULL CHECK (insight_type IN (
        'risk_pattern',      -- Recurring risk pattern detected
        'performance_trend', -- Performance degradation trend
        'failure_cluster',   -- Cluster of related failures
        'deployment_risk',   -- Risky deployment detected
        'coverage_gap',      -- Test coverage gap identified
        'flaky_test',        -- Flaky test pattern
        'dependency_issue',  -- Dependency-related issue
        'regression',        -- Regression detected
        'recommendation'     -- General recommendation
    )),

    -- Severity
    severity TEXT NOT NULL DEFAULT 'info' CHECK (severity IN (
        'critical', 'high', 'medium', 'low', 'info'
    )),

    -- Content
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    recommendations JSONB DEFAULT '[]',

    -- Related events
    event_ids UUID[] DEFAULT '{}',

    -- Status
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN (
        'active', 'acknowledged', 'resolved', 'dismissed'
    )),

    -- Status tracking
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by UUID,
    resolved_at TIMESTAMPTZ,
    resolved_by UUID,
    dismissed_at TIMESTAMPTZ,
    dismissed_by UUID,
    dismiss_reason TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Commit Analyses Table
-- Stores analysis results for commits and PRs
-- =============================================================================

CREATE TABLE IF NOT EXISTS commit_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Commit/PR identification
    commit_sha TEXT NOT NULL,
    pr_number INTEGER,
    branch_name TEXT,

    -- Change metrics
    files_changed INTEGER DEFAULT 0,
    lines_added INTEGER DEFAULT 0,
    lines_deleted INTEGER DEFAULT 0,
    affected_components TEXT[] DEFAULT '{}',

    -- Predicted test failures (from failure_patterns)
    predicted_test_failures JSONB DEFAULT '[]',
    -- Structure: [{"test_name": "...", "failure_probability": 0.8, "pattern_id": "...", "reason": "..."}]

    -- Suggested tests to run (from impact_graph)
    tests_to_run_suggested JSONB DEFAULT '[]',
    -- Structure: [{"test_name": "...", "test_file": "...", "impact_score": 0.9, "reason": "..."}]

    -- Risk assessment
    risk_score DECIMAL(3,2) DEFAULT 0.0 CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_factors JSONB DEFAULT '[]',
    -- Structure: [{"factor": "large_commit", "score": 0.3, "description": "..."}]

    -- Security analysis
    security_vulnerabilities JSONB DEFAULT '[]',
    -- Structure: [{"type": "sql_injection", "file": "...", "line": 42, "severity": "high", "description": "..."}]
    security_risk_score DECIMAL(3,2) DEFAULT 0.0 CHECK (security_risk_score >= 0 AND security_risk_score <= 1),

    -- AI recommendations
    recommendations JSONB DEFAULT '[]',
    -- Structure: [{"type": "add_test", "priority": "high", "description": "...", "suggested_action": "..."}]

    -- Deployment strategy
    deployment_strategy TEXT CHECK (deployment_strategy IN (
        'safe_to_deploy',      -- Low risk, safe to deploy
        'deploy_with_monitoring', -- Medium risk, deploy with extra monitoring
        'staged_rollout',      -- Higher risk, use staged rollout
        'manual_review',       -- Requires manual review
        'blocked'              -- Should not deploy until issues resolved
    )),
    deployment_notes TEXT,

    -- Actual outcomes (filled in after deployment)
    actual_test_failures TEXT[] DEFAULT '{}',
    actual_incidents TEXT[] DEFAULT '{}',
    prediction_accuracy_score DECIMAL(3,2),

    -- Analysis metadata
    analysis_version TEXT DEFAULT '1.0',
    analyzed_at TIMESTAMPTZ DEFAULT NOW(),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Unique constraint per commit per project
    UNIQUE(project_id, commit_sha)
);

-- =============================================================================
-- GitHub Webhook Events Log
-- Stores raw webhook events for auditing and replay
-- =============================================================================

CREATE TABLE IF NOT EXISTS github_webhook_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,

    -- Event identification
    delivery_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,

    -- Payload
    payload JSONB NOT NULL,

    -- Processing status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'processing', 'completed', 'failed', 'skipped'
    )),
    error_message TEXT,
    processed_at TIMESTAMPTZ,

    -- Related records created
    sdlc_event_id UUID REFERENCES sdlc_events(id) ON DELETE SET NULL,
    commit_analysis_id UUID REFERENCES commit_analyses(id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Indexes for Performance
-- =============================================================================

-- SDLC Events
CREATE INDEX IF NOT EXISTS idx_sdlc_events_project ON sdlc_events(project_id);
CREATE INDEX IF NOT EXISTS idx_sdlc_events_type ON sdlc_events(event_type);
CREATE INDEX IF NOT EXISTS idx_sdlc_events_source ON sdlc_events(source_platform);
CREATE INDEX IF NOT EXISTS idx_sdlc_events_occurred ON sdlc_events(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_sdlc_events_commit ON sdlc_events(commit_sha) WHERE commit_sha IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sdlc_events_pr ON sdlc_events(pr_number) WHERE pr_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sdlc_events_jira ON sdlc_events(jira_key) WHERE jira_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sdlc_events_deploy ON sdlc_events(deploy_id) WHERE deploy_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sdlc_events_branch ON sdlc_events(branch_name) WHERE branch_name IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sdlc_events_data ON sdlc_events USING GIN (data);

-- Event Correlations
CREATE INDEX IF NOT EXISTS idx_event_corr_source ON event_correlations(source_event_id);
CREATE INDEX IF NOT EXISTS idx_event_corr_target ON event_correlations(target_event_id);
CREATE INDEX IF NOT EXISTS idx_event_corr_type ON event_correlations(correlation_type);

-- Correlation Insights
CREATE INDEX IF NOT EXISTS idx_corr_insights_project ON correlation_insights(project_id);
CREATE INDEX IF NOT EXISTS idx_corr_insights_type ON correlation_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_corr_insights_status ON correlation_insights(status);
CREATE INDEX IF NOT EXISTS idx_corr_insights_severity ON correlation_insights(severity);

-- Commit Analyses
CREATE INDEX IF NOT EXISTS idx_commit_analyses_project ON commit_analyses(project_id);
CREATE INDEX IF NOT EXISTS idx_commit_analyses_sha ON commit_analyses(commit_sha);
CREATE INDEX IF NOT EXISTS idx_commit_analyses_pr ON commit_analyses(pr_number) WHERE pr_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_commit_analyses_branch ON commit_analyses(branch_name) WHERE branch_name IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_commit_analyses_risk ON commit_analyses(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_commit_analyses_strategy ON commit_analyses(deployment_strategy);
CREATE INDEX IF NOT EXISTS idx_commit_analyses_analyzed ON commit_analyses(analyzed_at DESC);

-- GitHub Webhook Events
CREATE INDEX IF NOT EXISTS idx_github_webhooks_project ON github_webhook_events(project_id);
CREATE INDEX IF NOT EXISTS idx_github_webhooks_type ON github_webhook_events(event_type);
CREATE INDEX IF NOT EXISTS idx_github_webhooks_status ON github_webhook_events(status);
CREATE INDEX IF NOT EXISTS idx_github_webhooks_created ON github_webhook_events(created_at DESC);

-- =============================================================================
-- Functions
-- =============================================================================

-- Get correlated events for a given event
CREATE OR REPLACE FUNCTION get_correlated_events(p_event_id UUID)
RETURNS TABLE (
    event_id UUID,
    event_type TEXT,
    source_platform TEXT,
    title TEXT,
    occurred_at TIMESTAMPTZ,
    correlation_type TEXT,
    direction TEXT,
    confidence DECIMAL
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    -- Events that this event points to
    SELECT
        e.id AS event_id,
        e.event_type,
        e.source_platform,
        e.title,
        e.occurred_at,
        ec.correlation_type,
        'outgoing'::TEXT AS direction,
        ec.confidence
    FROM event_correlations ec
    JOIN sdlc_events e ON e.id = ec.target_event_id
    WHERE ec.source_event_id = p_event_id

    UNION ALL

    -- Events that point to this event
    SELECT
        e.id AS event_id,
        e.event_type,
        e.source_platform,
        e.title,
        e.occurred_at,
        ec.correlation_type,
        'incoming'::TEXT AS direction,
        ec.confidence
    FROM event_correlations ec
    JOIN sdlc_events e ON e.id = ec.source_event_id
    WHERE ec.target_event_id = p_event_id

    ORDER BY occurred_at DESC;
END;
$$;

-- Get event timeline around a specific event
CREATE OR REPLACE FUNCTION get_event_timeline(
    p_target_event_id UUID,
    p_hours_before INTEGER DEFAULT 48,
    p_hours_after INTEGER DEFAULT 24
)
RETURNS TABLE (
    event_id UUID,
    event_type TEXT,
    source_platform TEXT,
    title TEXT,
    occurred_at TIMESTAMPTZ,
    relative_hours DECIMAL
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_target_occurred_at TIMESTAMPTZ;
    v_project_id UUID;
BEGIN
    -- Get target event details
    SELECT occurred_at, project_id
    INTO v_target_occurred_at, v_project_id
    FROM sdlc_events
    WHERE id = p_target_event_id;

    IF NOT FOUND THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT
        e.id AS event_id,
        e.event_type,
        e.source_platform,
        e.title,
        e.occurred_at,
        EXTRACT(EPOCH FROM (e.occurred_at - v_target_occurred_at)) / 3600 AS relative_hours
    FROM sdlc_events e
    WHERE e.project_id = v_project_id
      AND e.occurred_at >= v_target_occurred_at - (p_hours_before || ' hours')::INTERVAL
      AND e.occurred_at <= v_target_occurred_at + (p_hours_after || ' hours')::INTERVAL
    ORDER BY e.occurred_at;
END;
$$;

-- Auto-update timestamps
CREATE OR REPLACE FUNCTION update_sdlc_events_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_sdlc_events_updated
    BEFORE UPDATE ON sdlc_events
    FOR EACH ROW
    EXECUTE FUNCTION update_sdlc_events_timestamp();

CREATE TRIGGER trg_correlation_insights_updated
    BEFORE UPDATE ON correlation_insights
    FOR EACH ROW
    EXECUTE FUNCTION update_sdlc_events_timestamp();

CREATE TRIGGER trg_commit_analyses_updated
    BEFORE UPDATE ON commit_analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_sdlc_events_timestamp();

-- =============================================================================
-- Row Level Security
-- =============================================================================

ALTER TABLE sdlc_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE event_correlations ENABLE ROW LEVEL SECURITY;
ALTER TABLE correlation_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE commit_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE github_webhook_events ENABLE ROW LEVEL SECURITY;

-- SDLC Events policies
CREATE POLICY "Users can view SDLC events for their projects" ON sdlc_events
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage SDLC events for their projects" ON sdlc_events
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Event Correlations policies
CREATE POLICY "Users can view correlations for their events" ON event_correlations
    FOR SELECT USING (
        source_event_id IN (
            SELECT e.id FROM sdlc_events e
            WHERE e.project_id IN (
                SELECT p.id FROM projects p
                JOIN organizations o ON p.organization_id = o.id
                JOIN organization_members om ON o.id = om.organization_id
                WHERE om.user_id = auth.uid()::text
            )
        )
    );

CREATE POLICY "Users can manage correlations for their events" ON event_correlations
    FOR ALL USING (
        source_event_id IN (
            SELECT e.id FROM sdlc_events e
            WHERE e.project_id IN (
                SELECT p.id FROM projects p
                JOIN organizations o ON p.organization_id = o.id
                JOIN organization_members om ON o.id = om.organization_id
                WHERE om.user_id = auth.uid()::text
            )
        )
    );

-- Correlation Insights policies
CREATE POLICY "Users can view insights for their projects" ON correlation_insights
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage insights for their projects" ON correlation_insights
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Commit Analyses policies
CREATE POLICY "Users can view commit analyses for their projects" ON commit_analyses
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage commit analyses for their projects" ON commit_analyses
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- GitHub Webhook Events policies
CREATE POLICY "Users can view webhook events for their projects" ON github_webhook_events
    FOR SELECT USING (
        project_id IS NULL OR project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage webhook events for their projects" ON github_webhook_events
    FOR ALL USING (
        project_id IS NULL OR project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- =============================================================================
-- Grant Permissions
-- =============================================================================

GRANT ALL ON sdlc_events TO service_role;
GRANT ALL ON event_correlations TO service_role;
GRANT ALL ON correlation_insights TO service_role;
GRANT ALL ON commit_analyses TO service_role;
GRANT ALL ON github_webhook_events TO service_role;

-- Grant execute on functions
GRANT EXECUTE ON FUNCTION get_correlated_events(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION get_event_timeline(UUID, INTEGER, INTEGER) TO service_role;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE sdlc_events IS 'Unified SDLC event timeline from all integrated platforms';
COMMENT ON TABLE event_correlations IS 'Links between related SDLC events';
COMMENT ON TABLE correlation_insights IS 'AI-generated insights from cross-correlation analysis';
COMMENT ON TABLE commit_analyses IS 'Analysis results for commits and PRs including risk assessment and test recommendations';
COMMENT ON TABLE github_webhook_events IS 'Raw GitHub webhook events for auditing and replay';
