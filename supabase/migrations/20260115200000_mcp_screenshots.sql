-- MCP Screenshots table for storing screenshot metadata
CREATE TABLE mcp_screenshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- References
    connection_id UUID REFERENCES mcp_connections(id) ON DELETE CASCADE,
    activity_id UUID REFERENCES mcp_connection_activity(id) ON DELETE SET NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,

    -- Storage references
    r2_key TEXT NOT NULL UNIQUE,           -- Original R2 path
    supabase_path TEXT,                     -- Copied to Supabase Storage

    -- Screenshot metadata
    step_index INTEGER,
    screenshot_type TEXT NOT NULL DEFAULT 'step' CHECK (screenshot_type IN ('step', 'final', 'error', 'comparison')),
    tool_name TEXT,
    url_tested TEXT,

    -- Image metadata
    width INTEGER,
    height INTEGER,
    file_size_bytes INTEGER,

    -- Lifecycle
    retention_days INTEGER DEFAULT 30,
    copied_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX idx_mcp_screenshots_connection ON mcp_screenshots(connection_id);
CREATE INDEX idx_mcp_screenshots_project ON mcp_screenshots(project_id);
CREATE INDEX idx_mcp_screenshots_created ON mcp_screenshots(created_at DESC);
CREATE INDEX idx_mcp_screenshots_org ON mcp_screenshots(organization_id);

-- RLS
ALTER TABLE mcp_screenshots ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own org screenshots" ON mcp_screenshots
    FOR SELECT USING (
        organization_id IN (
            SELECT organization_id FROM organization_members
            WHERE user_id = auth.uid()::text
        )
    );

CREATE POLICY "Service role full access" ON mcp_screenshots
    FOR ALL USING (auth.role() = 'service_role');
