-- VCS Webhook Events and Impact Analysis tables migration
-- Adds tables for tracking GitHub/GitLab webhook events and commit impact analysis

-- VCS Webhook Events table
-- Stores raw webhook events from GitHub and GitLab for auditing and debugging
CREATE TABLE IF NOT EXISTS vcs_webhook_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    platform TEXT NOT NULL CHECK (platform IN ('github', 'gitlab')),
    event_type TEXT NOT NULL, -- push, pull_request, merge_request, check_run, pipeline
    delivery_id TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'received' CHECK (status IN ('received', 'processing', 'processed', 'skipped', 'failed')),
    sdlc_event_id UUID REFERENCES sdlc_events(id),
    error_message TEXT,
    received_at TIMESTAMPTZ DEFAULT now(),
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Commit Impact Analysis table
-- Stores results of test impact analysis for commits
CREATE TABLE IF NOT EXISTS commit_impact_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    commit_sha TEXT NOT NULL,
    pr_number INTEGER,
    mr_iid INTEGER, -- GitLab MR internal ID
    branch_name TEXT,
    files_changed INTEGER DEFAULT 0,
    tests_affected INTEGER DEFAULT 0,
    risk_score FLOAT DEFAULT 0.0 CHECK (risk_score >= 0.0 AND risk_score <= 1.0),
    affected_test_ids UUID[] DEFAULT '{}',
    recommendations JSONB DEFAULT '[]',
    analyzed_at TIMESTAMPTZ DEFAULT now(),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (project_id, commit_sha)
);

-- Indexes for vcs_webhook_events
CREATE INDEX IF NOT EXISTS idx_vcs_webhook_events_project ON vcs_webhook_events(project_id);
CREATE INDEX IF NOT EXISTS idx_vcs_webhook_events_platform ON vcs_webhook_events(platform);
CREATE INDEX IF NOT EXISTS idx_vcs_webhook_events_event_type ON vcs_webhook_events(event_type);
CREATE INDEX IF NOT EXISTS idx_vcs_webhook_events_delivery_id ON vcs_webhook_events(delivery_id);
CREATE INDEX IF NOT EXISTS idx_vcs_webhook_events_status ON vcs_webhook_events(status);
CREATE INDEX IF NOT EXISTS idx_vcs_webhook_events_received ON vcs_webhook_events(received_at DESC);

-- Indexes for commit_impact_analyses
CREATE INDEX IF NOT EXISTS idx_commit_impact_analyses_project ON commit_impact_analyses(project_id);
CREATE INDEX IF NOT EXISTS idx_commit_impact_analyses_commit ON commit_impact_analyses(commit_sha);
CREATE INDEX IF NOT EXISTS idx_commit_impact_analyses_pr ON commit_impact_analyses(pr_number) WHERE pr_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_commit_impact_analyses_mr ON commit_impact_analyses(mr_iid) WHERE mr_iid IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_commit_impact_analyses_risk ON commit_impact_analyses(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_commit_impact_analyses_analyzed ON commit_impact_analyses(analyzed_at DESC);

-- Enable RLS
ALTER TABLE vcs_webhook_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE commit_impact_analyses ENABLE ROW LEVEL SECURITY;

-- RLS Policies for vcs_webhook_events
-- Service role can do everything
CREATE POLICY "Service role full access to vcs_webhook_events"
    ON vcs_webhook_events
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Authenticated users can read their organization's events
CREATE POLICY "Users can read their org vcs_webhook_events"
    ON vcs_webhook_events
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM projects p
            JOIN organization_members om ON p.organization_id = om.organization_id
            WHERE p.id = vcs_webhook_events.project_id
            AND om.user_id = auth.uid()::text
        )
    );

-- RLS Policies for commit_impact_analyses
-- Service role can do everything
CREATE POLICY "Service role full access to commit_impact_analyses"
    ON commit_impact_analyses
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Authenticated users can read their organization's analyses
CREATE POLICY "Users can read their org commit_impact_analyses"
    ON commit_impact_analyses
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM projects p
            JOIN organization_members om ON p.organization_id = om.organization_id
            WHERE p.id = commit_impact_analyses.project_id
            AND om.user_id = auth.uid()::text
        )
    );

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_commit_impact_analyses_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER commit_impact_analyses_updated_at
    BEFORE UPDATE ON commit_impact_analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_commit_impact_analyses_updated_at();

-- Grant permissions
GRANT ALL ON vcs_webhook_events TO service_role;
GRANT SELECT ON vcs_webhook_events TO authenticated;
GRANT ALL ON commit_impact_analyses TO service_role;
GRANT SELECT ON commit_impact_analyses TO authenticated;

COMMENT ON TABLE vcs_webhook_events IS 'Stores raw VCS webhook events from GitHub and GitLab for auditing and debugging';
COMMENT ON TABLE commit_impact_analyses IS 'Stores test impact analysis results for commits to identify affected tests and risk scores';
