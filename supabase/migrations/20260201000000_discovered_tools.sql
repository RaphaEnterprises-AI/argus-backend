-- Discovered Tools table for Tool Discovery Agent
-- Stores tools parsed from OpenAPI, GraphQL, MCP manifests, and plain documentation

CREATE TABLE IF NOT EXISTS discovered_tools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Tool identity
    tool_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    tool_type TEXT NOT NULL,  -- openapi, graphql, mcp, rest, grpc, cli, python, javascript, custom
    source_url TEXT NOT NULL,

    -- Multi-tenant isolation
    org_id TEXT NOT NULL,
    project_id TEXT,

    -- Tool schema
    parameters JSONB DEFAULT '[]',
    returns JSONB,

    -- Authentication
    auth_type TEXT DEFAULT 'none',  -- none, api_key, bearer, oauth2, basic, custom
    auth_config JSONB DEFAULT '{}',

    -- Examples and metadata
    examples JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',

    -- Confidence and status
    confidence FLOAT DEFAULT 1.0,
    status TEXT DEFAULT 'active',  -- active, deprecated, disabled

    -- Generated wrapper (if any)
    wrapper_code TEXT,
    wrapper_language TEXT DEFAULT 'python',
    wrapper_dependencies JSONB DEFAULT '[]',

    -- Verification status
    verified BOOLEAN DEFAULT FALSE,
    verification_results JSONB,
    last_verified_at TIMESTAMP WITH TIME ZONE,

    -- Usage tracking
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    avg_latency_ms FLOAT,
    last_used_at TIMESTAMP WITH TIME ZONE,

    -- Timestamps
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for multi-tenant queries
CREATE INDEX idx_discovered_tools_org_project ON discovered_tools(org_id, project_id);

-- Index for tool type filtering
CREATE INDEX idx_discovered_tools_type ON discovered_tools(tool_type);

-- Index for status filtering
CREATE INDEX idx_discovered_tools_status ON discovered_tools(status);

-- Index for verified tools
CREATE INDEX idx_discovered_tools_verified ON discovered_tools(verified) WHERE verified = TRUE;

-- Full-text search on name and description
CREATE INDEX idx_discovered_tools_search ON discovered_tools
    USING GIN (to_tsvector('english', name || ' ' || COALESCE(description, '')));

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_discovered_tools_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_discovered_tools_updated_at
    BEFORE UPDATE ON discovered_tools
    FOR EACH ROW
    EXECUTE FUNCTION update_discovered_tools_updated_at();

-- Enable Row Level Security
ALTER TABLE discovered_tools ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only access tools in their organization
CREATE POLICY "Users can access tools in their organization"
    ON discovered_tools
    FOR ALL
    USING (org_id = current_setting('app.current_org_id', TRUE))
    WITH CHECK (org_id = current_setting('app.current_org_id', TRUE));

-- Policy: Service role has full access
CREATE POLICY "Service role has full access to discovered_tools"
    ON discovered_tools
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- Comments
COMMENT ON TABLE discovered_tools IS 'Tools discovered by ToolDiscoveryAgent from various documentation sources';
COMMENT ON COLUMN discovered_tools.tool_id IS 'Unique identifier derived from name, source, and type';
COMMENT ON COLUMN discovered_tools.tool_type IS 'Source type: openapi, graphql, mcp, rest, grpc, cli, python, javascript, custom';
COMMENT ON COLUMN discovered_tools.parameters IS 'JSON array of parameter definitions with name, type, description, required';
COMMENT ON COLUMN discovered_tools.returns IS 'JSON object describing return type and schema';
COMMENT ON COLUMN discovered_tools.confidence IS 'AI confidence score for discovered tools (0.0-1.0)';
COMMENT ON COLUMN discovered_tools.wrapper_code IS 'Generated Python/JS wrapper code for the tool';
COMMENT ON COLUMN discovered_tools.verification_results IS 'Results from syntax, security, and runtime verification';

-- ============================================================================
-- Tool Usage History table for learning patterns
-- ============================================================================

CREATE TABLE IF NOT EXISTS tool_usage_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- References
    tool_id TEXT NOT NULL REFERENCES discovered_tools(tool_id) ON DELETE CASCADE,
    org_id TEXT NOT NULL,
    project_id TEXT,

    -- Usage details
    invoked_by TEXT,  -- agent_id or user_id
    invoked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Input/Output
    input_params JSONB,
    output_result JSONB,

    -- Execution metrics
    success BOOLEAN NOT NULL,
    latency_ms INTEGER,
    error_message TEXT,
    error_type TEXT,

    -- Context
    context JSONB DEFAULT '{}',

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for tool usage queries
CREATE INDEX idx_tool_usage_history_tool_id ON tool_usage_history(tool_id);
CREATE INDEX idx_tool_usage_history_org_project ON tool_usage_history(org_id, project_id);
CREATE INDEX idx_tool_usage_history_invoked_at ON tool_usage_history(invoked_at);
CREATE INDEX idx_tool_usage_history_success ON tool_usage_history(success);

-- Partition hint (for production, consider partitioning by invoked_at)
COMMENT ON TABLE tool_usage_history IS 'Usage history for discovered tools, used for learning patterns and optimization';

-- Enable RLS
ALTER TABLE tool_usage_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can access tool usage in their organization"
    ON tool_usage_history
    FOR ALL
    USING (org_id = current_setting('app.current_org_id', TRUE))
    WITH CHECK (org_id = current_setting('app.current_org_id', TRUE));

CREATE POLICY "Service role has full access to tool_usage_history"
    ON tool_usage_history
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- ============================================================================
-- Tool Assignments table for agent-tool mapping
-- ============================================================================

CREATE TABLE IF NOT EXISTS tool_assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    tool_id TEXT NOT NULL REFERENCES discovered_tools(tool_id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL,
    agent_type TEXT NOT NULL,

    -- Assignment details
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    assigned_by TEXT,  -- user_id or 'system'

    -- Configuration for this assignment
    config JSONB DEFAULT '{}',

    -- Status
    active BOOLEAN DEFAULT TRUE,

    -- Multi-tenant
    org_id TEXT NOT NULL,
    project_id TEXT,

    UNIQUE(tool_id, agent_id)
);

CREATE INDEX idx_tool_assignments_agent_id ON tool_assignments(agent_id);
CREATE INDEX idx_tool_assignments_org_project ON tool_assignments(org_id, project_id);
CREATE INDEX idx_tool_assignments_active ON tool_assignments(active) WHERE active = TRUE;

-- Enable RLS
ALTER TABLE tool_assignments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can access tool assignments in their organization"
    ON tool_assignments
    FOR ALL
    USING (org_id = current_setting('app.current_org_id', TRUE))
    WITH CHECK (org_id = current_setting('app.current_org_id', TRUE));

CREATE POLICY "Service role has full access to tool_assignments"
    ON tool_assignments
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

COMMENT ON TABLE tool_assignments IS 'Mapping between discovered tools and agents that can use them';
