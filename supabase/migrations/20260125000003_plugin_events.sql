-- Plugin Events Table for Claude Code Plugin Monitoring
-- This table captures events from the Argus Claude Code plugin
-- enabling real-time monitoring and analytics in the dashboard

-- ============================================================================
-- PLUGIN EVENTS TABLE
-- ============================================================================
-- Captures all activity from the Claude Code plugin including:
-- - Command executions (/test, /discover, /analyze, etc.)
-- - Skill activations (commit-impact, security-scan, test-suggestions)
-- - Agent invocations (e2e-tester, api-tester, security-scanner, self-healer)
-- - Hook triggers (SessionStart, PreToolUse, PostToolUse, etc.)
-- - MCP tool calls made through the plugin

CREATE TABLE plugin_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,

    -- Session tracking - groups related events
    session_id TEXT NOT NULL,

    -- Event classification
    event_type TEXT NOT NULL CHECK (event_type IN (
        'command',       -- Slash command execution (/test, /discover, etc.)
        'skill',         -- Skill activation (auto-triggered)
        'agent',         -- Sub-agent invocation
        'hook',          -- Hook trigger
        'mcp_tool',      -- MCP tool call
        'error',         -- Error in plugin
        'session_start', -- Plugin session started
        'session_end'    -- Plugin session ended
    )),

    -- Event identification
    event_name TEXT NOT NULL,  -- e.g., '/test', 'commit-impact', 'e2e-tester', 'PreToolUse'

    -- Status tracking
    status TEXT NOT NULL DEFAULT 'started' CHECK (status IN (
        'started',    -- Event started
        'completed',  -- Event completed successfully
        'failed',     -- Event failed
        'cancelled'   -- Event was cancelled
    )),

    -- Timing
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,

    -- Context and payload
    input_data JSONB DEFAULT '{}',    -- Input parameters for the event
    output_data JSONB DEFAULT '{}',   -- Result/output of the event
    error_message TEXT,               -- Error message if failed

    -- Git context (if available)
    git_branch TEXT,
    git_commit TEXT,
    git_repo TEXT,

    -- Workspace context
    workspace_path TEXT,
    file_paths TEXT[],                -- Files involved in this event

    -- Metadata
    plugin_version TEXT,
    claude_code_version TEXT,
    os_platform TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR EFFICIENT QUERYING
-- ============================================================================

-- Basic filtering
CREATE INDEX idx_plugin_events_user ON plugin_events(user_id);
CREATE INDEX idx_plugin_events_project ON plugin_events(project_id);
CREATE INDEX idx_plugin_events_session ON plugin_events(session_id);
CREATE INDEX idx_plugin_events_type ON plugin_events(event_type);
CREATE INDEX idx_plugin_events_name ON plugin_events(event_name);
CREATE INDEX idx_plugin_events_status ON plugin_events(status);

-- Timeline queries
CREATE INDEX idx_plugin_events_started ON plugin_events(started_at DESC);
CREATE INDEX idx_plugin_events_user_timeline ON plugin_events(user_id, started_at DESC);

-- Git queries
CREATE INDEX idx_plugin_events_git_repo ON plugin_events(git_repo)
    WHERE git_repo IS NOT NULL;
CREATE INDEX idx_plugin_events_git_branch ON plugin_events(git_branch)
    WHERE git_branch IS NOT NULL;

-- Performance analysis
CREATE INDEX idx_plugin_events_duration ON plugin_events(duration_ms DESC)
    WHERE duration_ms IS NOT NULL;

-- File path searches
CREATE INDEX idx_plugin_events_files ON plugin_events USING GIN(file_paths);

-- JSONB queries
CREATE INDEX idx_plugin_events_input ON plugin_events USING GIN(input_data);
CREATE INDEX idx_plugin_events_output ON plugin_events USING GIN(output_data);

-- ============================================================================
-- PLUGIN SESSIONS TABLE
-- ============================================================================
-- Aggregated view of plugin sessions for quick dashboard access

CREATE TABLE plugin_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL UNIQUE,

    -- Session info
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    duration_ms INTEGER,

    -- Aggregated stats
    total_events INTEGER DEFAULT 0,
    commands_executed INTEGER DEFAULT 0,
    skills_activated INTEGER DEFAULT 0,
    agents_invoked INTEGER DEFAULT 0,
    hooks_triggered INTEGER DEFAULT 0,
    mcp_calls INTEGER DEFAULT 0,
    errors_count INTEGER DEFAULT 0,

    -- Context
    git_repo TEXT,
    git_branch TEXT,
    workspace_path TEXT,

    -- Versions
    plugin_version TEXT,
    claude_code_version TEXT,
    os_platform TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Session indexes
CREATE INDEX idx_plugin_sessions_user ON plugin_sessions(user_id);
CREATE INDEX idx_plugin_sessions_project ON plugin_sessions(project_id);
CREATE INDEX idx_plugin_sessions_started ON plugin_sessions(started_at DESC);

-- ============================================================================
-- PLUGIN METRICS TABLE
-- ============================================================================
-- Daily/hourly aggregated metrics for trending and analytics

CREATE TABLE plugin_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT,  -- NULL for global metrics
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,

    -- Time bucket
    time_bucket TIMESTAMPTZ NOT NULL,
    bucket_size TEXT NOT NULL DEFAULT 'hour' CHECK (bucket_size IN ('hour', 'day', 'week')),

    -- Counts by event type
    command_count INTEGER DEFAULT 0,
    skill_count INTEGER DEFAULT 0,
    agent_count INTEGER DEFAULT 0,
    hook_count INTEGER DEFAULT 0,
    mcp_call_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    session_count INTEGER DEFAULT 0,

    -- Performance metrics
    avg_duration_ms DECIMAL(10, 2),
    p50_duration_ms DECIMAL(10, 2),
    p95_duration_ms DECIMAL(10, 2),
    p99_duration_ms DECIMAL(10, 2),

    -- Popular items
    top_commands JSONB DEFAULT '[]',      -- [{name, count}]
    top_skills JSONB DEFAULT '[]',
    top_agents JSONB DEFAULT '[]',

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(user_id, project_id, time_bucket, bucket_size)
);

-- Metrics indexes
CREATE INDEX idx_plugin_metrics_time ON plugin_metrics(time_bucket DESC);
CREATE INDEX idx_plugin_metrics_user ON plugin_metrics(user_id, time_bucket DESC);
CREATE INDEX idx_plugin_metrics_bucket ON plugin_metrics(bucket_size, time_bucket DESC);

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE plugin_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE plugin_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE plugin_metrics ENABLE ROW LEVEL SECURITY;

-- Permissive policies (to be tightened with Clerk JWT integration)
CREATE POLICY "Enable all for authenticated users"
    ON plugin_events FOR ALL USING (true);

CREATE POLICY "Enable all for authenticated users"
    ON plugin_sessions FOR ALL USING (true);

CREATE POLICY "Enable all for authenticated users"
    ON plugin_metrics FOR ALL USING (true);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Auto-update updated_at
CREATE TRIGGER update_plugin_events_updated_at
    BEFORE UPDATE ON plugin_events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_plugin_sessions_updated_at
    BEFORE UPDATE ON plugin_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to atomically increment session counters
CREATE OR REPLACE FUNCTION increment_plugin_session_counter(
    p_session_id TEXT,
    p_field TEXT
)
RETURNS void AS $$
BEGIN
    -- Update the appropriate counter field
    EXECUTE format(
        'UPDATE plugin_sessions SET %I = COALESCE(%I, 0) + 1, total_events = COALESCE(total_events, 0) + 1, updated_at = NOW() WHERE session_id = $1',
        p_field, p_field
    ) USING p_session_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION increment_plugin_session_counter IS
    'Atomically increments a counter field on a plugin session';

-- Function to get plugin usage summary for a user
CREATE OR REPLACE FUNCTION get_plugin_usage_summary(
    target_user_id TEXT,
    days_back INTEGER DEFAULT 7
)
RETURNS TABLE (
    total_sessions BIGINT,
    total_events BIGINT,
    total_commands BIGINT,
    total_skills BIGINT,
    total_agents BIGINT,
    total_errors BIGINT,
    avg_session_duration_ms DECIMAL,
    most_used_command TEXT,
    most_used_skill TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(DISTINCT pe.session_id) as total_sessions,
        COUNT(*) as total_events,
        COUNT(*) FILTER (WHERE pe.event_type = 'command') as total_commands,
        COUNT(*) FILTER (WHERE pe.event_type = 'skill') as total_skills,
        COUNT(*) FILTER (WHERE pe.event_type = 'agent') as total_agents,
        COUNT(*) FILTER (WHERE pe.status = 'failed') as total_errors,
        AVG(ps.duration_ms)::DECIMAL as avg_session_duration_ms,
        (
            SELECT event_name FROM plugin_events
            WHERE user_id = target_user_id
              AND event_type = 'command'
              AND started_at > NOW() - (days_back || ' days')::INTERVAL
            GROUP BY event_name
            ORDER BY COUNT(*) DESC
            LIMIT 1
        ) as most_used_command,
        (
            SELECT event_name FROM plugin_events
            WHERE user_id = target_user_id
              AND event_type = 'skill'
              AND started_at > NOW() - (days_back || ' days')::INTERVAL
            GROUP BY event_name
            ORDER BY COUNT(*) DESC
            LIMIT 1
        ) as most_used_skill
    FROM plugin_events pe
    LEFT JOIN plugin_sessions ps ON pe.session_id = ps.session_id
    WHERE pe.user_id = target_user_id
      AND pe.started_at > NOW() - (days_back || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- Function to get recent plugin activity timeline
CREATE OR REPLACE FUNCTION get_plugin_activity_timeline(
    target_user_id TEXT,
    target_session_id TEXT DEFAULT NULL,
    limit_count INTEGER DEFAULT 50
)
RETURNS TABLE (
    event_id UUID,
    event_type TEXT,
    event_name TEXT,
    status TEXT,
    started_at TIMESTAMPTZ,
    duration_ms INTEGER,
    git_branch TEXT,
    error_message TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        pe.id as event_id,
        pe.event_type,
        pe.event_name,
        pe.status,
        pe.started_at,
        pe.duration_ms,
        pe.git_branch,
        pe.error_message
    FROM plugin_events pe
    WHERE pe.user_id = target_user_id
      AND (target_session_id IS NULL OR pe.session_id = target_session_id)
    ORDER BY pe.started_at DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- REALTIME SUBSCRIPTIONS
-- ============================================================================

ALTER PUBLICATION supabase_realtime ADD TABLE plugin_events;
ALTER PUBLICATION supabase_realtime ADD TABLE plugin_sessions;

-- ============================================================================
-- COLUMN COMMENTS
-- ============================================================================

COMMENT ON TABLE plugin_events IS
    'Records all events from the Argus Claude Code plugin for monitoring and analytics';

COMMENT ON TABLE plugin_sessions IS
    'Aggregated plugin session data for quick dashboard access';

COMMENT ON TABLE plugin_metrics IS
    'Time-bucketed metrics for plugin usage analytics and trending';

COMMENT ON COLUMN plugin_events.event_type IS
    'Type of event: command, skill, agent, hook, mcp_tool, error, session_start, session_end';

COMMENT ON COLUMN plugin_events.session_id IS
    'Unique identifier for a Claude Code session, groups related events';

COMMENT ON FUNCTION get_plugin_usage_summary IS
    'Returns plugin usage statistics for a user over the specified time period';

COMMENT ON FUNCTION get_plugin_activity_timeline IS
    'Returns recent plugin activity for display in the dashboard';

-- ============================================================================
-- COMPLETION
-- ============================================================================

SELECT 'Plugin events schema created successfully!' as message;
