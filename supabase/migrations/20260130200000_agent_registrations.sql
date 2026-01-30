-- Agent Registrations table for persistent A2A agent discovery
-- Stores agent metadata, health status, and capabilities for service restarts

CREATE TABLE IF NOT EXISTS agent_registrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT UNIQUE NOT NULL,
    agent_type TEXT NOT NULL,
    capabilities JSONB DEFAULT '[]',
    status VARCHAR(20) DEFAULT 'healthy',
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for discovering agents by type
CREATE INDEX idx_agent_registrations_type ON agent_registrations(agent_type);

-- Index for filtering by health status
CREATE INDEX idx_agent_registrations_status ON agent_registrations(status);

-- Index for cleanup queries (finding stale agents)
CREATE INDEX idx_agent_registrations_last_heartbeat ON agent_registrations(last_heartbeat);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_agent_registrations_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_agent_registrations_updated_at
    BEFORE UPDATE ON agent_registrations
    FOR EACH ROW
    EXECUTE FUNCTION update_agent_registrations_updated_at();

-- Grant permissions for RLS (Row Level Security disabled for system table)
-- Agent registrations are system-level and not user-specific
ALTER TABLE agent_registrations ENABLE ROW LEVEL SECURITY;

-- Allow service role full access (used by backend)
CREATE POLICY "Service role has full access to agent_registrations"
    ON agent_registrations
    FOR ALL
    USING (true)
    WITH CHECK (true);

COMMENT ON TABLE agent_registrations IS 'Persistent storage for A2A agent registrations and health status';
COMMENT ON COLUMN agent_registrations.agent_id IS 'Unique identifier for the agent instance (e.g., code_analyzer_abc123)';
COMMENT ON COLUMN agent_registrations.agent_type IS 'Type/class of agent (e.g., code_analyzer, ui_tester)';
COMMENT ON COLUMN agent_registrations.capabilities IS 'JSON array of capability strings this agent provides';
COMMENT ON COLUMN agent_registrations.status IS 'Health status: healthy, unhealthy, or unknown';
COMMENT ON COLUMN agent_registrations.last_heartbeat IS 'Last time agent sent a heartbeat';
COMMENT ON COLUMN agent_registrations.metadata IS 'Additional metadata about the agent (version, config, etc.)';
