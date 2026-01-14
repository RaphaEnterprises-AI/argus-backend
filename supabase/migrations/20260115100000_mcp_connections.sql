-- MCP Connections and Device Auth Migration
-- Tracks MCP client connections and device authorization sessions
-- Enables dashboard visibility into connected IDE clients

-- ============================================================================
-- MCP CONNECTIONS TABLE (Track active MCP client sessions)
-- ============================================================================

CREATE TABLE IF NOT EXISTS mcp_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- User and organization context
    user_id TEXT NOT NULL,  -- Clerk user ID
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    
    -- Client identification
    client_id TEXT NOT NULL DEFAULT 'argus-mcp',  -- OAuth client ID
    client_name TEXT,  -- Human-readable name (e.g., "Claude Code", "Cursor")
    client_type TEXT NOT NULL DEFAULT 'mcp' CHECK (client_type IN ('mcp', 'cli', 'api')),
    
    -- Session details
    session_id TEXT UNIQUE,  -- MCP session ID (from Cloudflare DO)
    device_name TEXT,  -- User-provided device name
    device_fingerprint TEXT,  -- Optional device fingerprint for security
    
    -- Connection metadata
    ip_address INET,
    user_agent TEXT,
    location JSONB,  -- GeoIP data: {city, country, region}
    
    -- Token info (reference only, tokens stored separately)
    token_id UUID,  -- Reference to issued token
    scopes TEXT[] NOT NULL DEFAULT ARRAY['read', 'write'],
    
    -- Activity tracking
    last_activity_at TIMESTAMPTZ DEFAULT now(),
    request_count INTEGER DEFAULT 0,
    tools_used JSONB DEFAULT '[]',  -- Array of tool names used
    
    -- Lifecycle
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'revoked')),
    connected_at TIMESTAMPTZ DEFAULT now(),
    disconnected_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    revoked_by TEXT,  -- User ID who revoked
    revoke_reason TEXT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_mcp_connections_user ON mcp_connections(user_id);
CREATE INDEX IF NOT EXISTS idx_mcp_connections_org ON mcp_connections(organization_id);
CREATE INDEX IF NOT EXISTS idx_mcp_connections_session ON mcp_connections(session_id);
CREATE INDEX IF NOT EXISTS idx_mcp_connections_status ON mcp_connections(status);
CREATE INDEX IF NOT EXISTS idx_mcp_connections_last_activity ON mcp_connections(last_activity_at DESC);
CREATE INDEX IF NOT EXISTS idx_mcp_connections_client ON mcp_connections(client_id, client_type);

-- ============================================================================
-- DEVICE AUTH SESSIONS TABLE (Persistent device authorization requests)
-- Replaces in-memory storage for production use
-- ============================================================================

CREATE TABLE IF NOT EXISTS device_auth_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Device codes (hashed for security)
    device_code_hash TEXT NOT NULL UNIQUE,
    user_code TEXT NOT NULL UNIQUE,
    
    -- Client info
    client_id TEXT NOT NULL DEFAULT 'argus-mcp',
    scopes TEXT[] NOT NULL DEFAULT ARRAY['read', 'write'],
    
    -- Request metadata
    ip_address INET,
    user_agent TEXT,
    
    -- Authorization status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'denied', 'expired', 'exchanged')),
    
    -- User info (populated on approval)
    user_id TEXT,
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    email TEXT,
    name TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    approved_at TIMESTAMPTZ,
    exchanged_at TIMESTAMPTZ,  -- When token was issued
    
    -- Resulting connection
    connection_id UUID REFERENCES mcp_connections(id) ON DELETE SET NULL
);

-- Indexes for device auth
CREATE INDEX IF NOT EXISTS idx_device_auth_user_code ON device_auth_sessions(user_code);
CREATE INDEX IF NOT EXISTS idx_device_auth_status ON device_auth_sessions(status);
CREATE INDEX IF NOT EXISTS idx_device_auth_expires ON device_auth_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_device_auth_user ON device_auth_sessions(user_id);

-- ============================================================================
-- MCP CONNECTION ACTIVITY LOG (Detailed activity tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS mcp_connection_activity (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connection_id UUID NOT NULL REFERENCES mcp_connections(id) ON DELETE CASCADE,
    
    -- Activity details
    activity_type TEXT NOT NULL CHECK (activity_type IN (
        'connect', 'disconnect', 'tool_call', 'auth_refresh', 'error'
    )),
    tool_name TEXT,  -- For tool_call events
    
    -- Request/Response info
    request_id TEXT,
    duration_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mcp_activity_connection ON mcp_connection_activity(connection_id);
CREATE INDEX IF NOT EXISTS idx_mcp_activity_type ON mcp_connection_activity(activity_type);
CREATE INDEX IF NOT EXISTS idx_mcp_activity_time ON mcp_connection_activity(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mcp_activity_tool ON mcp_connection_activity(tool_name) WHERE tool_name IS NOT NULL;

-- ============================================================================
-- FUNCTIONS: MCP Connection Management
-- ============================================================================

-- Function to create or update an MCP connection
CREATE OR REPLACE FUNCTION upsert_mcp_connection(
    p_user_id TEXT,
    p_organization_id UUID,
    p_session_id TEXT,
    p_client_id TEXT DEFAULT 'argus-mcp',
    p_client_name TEXT DEFAULT NULL,
    p_client_type TEXT DEFAULT 'mcp',
    p_ip_address INET DEFAULT NULL,
    p_user_agent TEXT DEFAULT NULL,
    p_scopes TEXT[] DEFAULT ARRAY['read', 'write'],
    p_metadata JSONB DEFAULT '{}'
) RETURNS UUID AS $$
DECLARE
    v_connection_id UUID;
BEGIN
    -- Try to find existing active connection for this session
    SELECT id INTO v_connection_id
    FROM mcp_connections
    WHERE session_id = p_session_id AND status = 'active';
    
    IF v_connection_id IS NOT NULL THEN
        -- Update existing connection
        UPDATE mcp_connections
        SET 
            last_activity_at = now(),
            updated_at = now(),
            ip_address = COALESCE(p_ip_address, ip_address),
            user_agent = COALESCE(p_user_agent, user_agent)
        WHERE id = v_connection_id;
    ELSE
        -- Create new connection
        INSERT INTO mcp_connections (
            user_id, organization_id, session_id, client_id, client_name,
            client_type, ip_address, user_agent, scopes, metadata
        ) VALUES (
            p_user_id, p_organization_id, p_session_id, p_client_id, p_client_name,
            p_client_type, p_ip_address, p_user_agent, p_scopes, p_metadata
        )
        RETURNING id INTO v_connection_id;
        
        -- Log the connection event
        INSERT INTO mcp_connection_activity (connection_id, activity_type, metadata)
        VALUES (v_connection_id, 'connect', p_metadata);
    END IF;
    
    RETURN v_connection_id;
END;
$$ LANGUAGE plpgsql;

-- Function to record tool usage
CREATE OR REPLACE FUNCTION record_mcp_tool_usage(
    p_connection_id UUID,
    p_tool_name TEXT,
    p_request_id TEXT DEFAULT NULL,
    p_duration_ms INTEGER DEFAULT NULL,
    p_success BOOLEAN DEFAULT true,
    p_error_message TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
) RETURNS UUID AS $$
DECLARE
    v_activity_id UUID;
BEGIN
    -- Update connection stats
    UPDATE mcp_connections
    SET 
        last_activity_at = now(),
        request_count = request_count + 1,
        tools_used = CASE 
            WHEN NOT tools_used ? p_tool_name 
            THEN tools_used || jsonb_build_array(p_tool_name)
            ELSE tools_used
        END,
        updated_at = now()
    WHERE id = p_connection_id;
    
    -- Log the activity
    INSERT INTO mcp_connection_activity (
        connection_id, activity_type, tool_name, request_id,
        duration_ms, success, error_message, metadata
    ) VALUES (
        p_connection_id, 'tool_call', p_tool_name, p_request_id,
        p_duration_ms, p_success, p_error_message, p_metadata
    )
    RETURNING id INTO v_activity_id;
    
    RETURN v_activity_id;
END;
$$ LANGUAGE plpgsql;

-- Function to revoke an MCP connection
CREATE OR REPLACE FUNCTION revoke_mcp_connection(
    p_connection_id UUID,
    p_revoked_by TEXT,
    p_reason TEXT DEFAULT 'User revoked'
) RETURNS BOOLEAN AS $$
BEGIN
    UPDATE mcp_connections
    SET 
        status = 'revoked',
        revoked_at = now(),
        revoked_by = p_revoked_by,
        revoke_reason = p_reason,
        updated_at = now()
    WHERE id = p_connection_id AND status = 'active';
    
    IF FOUND THEN
        -- Log the disconnect
        INSERT INTO mcp_connection_activity (connection_id, activity_type, metadata)
        VALUES (p_connection_id, 'disconnect', jsonb_build_object('reason', p_reason));
        RETURN true;
    END IF;
    
    RETURN false;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup expired device auth sessions
CREATE OR REPLACE FUNCTION cleanup_expired_device_auth() RETURNS INTEGER AS $$
DECLARE
    v_count INTEGER;
BEGIN
    UPDATE device_auth_sessions
    SET status = 'expired'
    WHERE status = 'pending' AND expires_at < now();
    
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS: Dashboard aggregations
-- ============================================================================

-- View for active MCP connections with user info
CREATE OR REPLACE VIEW v_active_mcp_connections AS
SELECT 
    mc.id,
    mc.user_id,
    mc.organization_id,
    mc.client_name,
    mc.client_type,
    mc.session_id,
    mc.device_name,
    mc.ip_address,
    mc.last_activity_at,
    mc.request_count,
    mc.tools_used,
    mc.connected_at,
    mc.scopes,
    -- Calculate time since last activity
    EXTRACT(EPOCH FROM (now() - mc.last_activity_at)) as seconds_since_activity,
    -- Connection duration
    EXTRACT(EPOCH FROM (now() - mc.connected_at)) as connection_duration_seconds
FROM mcp_connections mc
WHERE mc.status = 'active'
ORDER BY mc.last_activity_at DESC;

-- View for MCP connection statistics per organization
CREATE OR REPLACE VIEW v_mcp_connection_stats AS
SELECT 
    organization_id,
    COUNT(*) FILTER (WHERE status = 'active') as active_connections,
    COUNT(*) FILTER (WHERE status = 'revoked') as revoked_connections,
    COUNT(*) as total_connections,
    SUM(request_count) as total_requests,
    MAX(last_activity_at) as last_activity,
    COUNT(DISTINCT user_id) as unique_users,
    jsonb_agg(DISTINCT client_name) FILTER (WHERE client_name IS NOT NULL) as client_types
FROM mcp_connections
GROUP BY organization_id;

-- ============================================================================
-- TRIGGERS: Auto-update timestamps
-- ============================================================================

CREATE OR REPLACE FUNCTION update_mcp_connection_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_mcp_connection_updated ON mcp_connections;
CREATE TRIGGER trigger_mcp_connection_updated
    BEFORE UPDATE ON mcp_connections
    FOR EACH ROW
    EXECUTE FUNCTION update_mcp_connection_timestamp();

-- ============================================================================
-- RLS POLICIES (Row Level Security)
-- ============================================================================

-- Enable RLS
ALTER TABLE mcp_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE device_auth_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE mcp_connection_activity ENABLE ROW LEVEL SECURITY;

-- Policies for mcp_connections
CREATE POLICY mcp_connections_select_own ON mcp_connections
    FOR SELECT USING (user_id = current_setting('app.user_id', true));

CREATE POLICY mcp_connections_select_org ON mcp_connections
    FOR SELECT USING (
        organization_id IN (
            SELECT organization_id FROM organization_members 
            WHERE user_id = current_setting('app.user_id', true)
        )
    );

-- Service role can do anything
CREATE POLICY mcp_connections_service ON mcp_connections
    FOR ALL USING (current_setting('role', true) = 'service_role');

CREATE POLICY device_auth_service ON device_auth_sessions
    FOR ALL USING (current_setting('role', true) = 'service_role');

CREATE POLICY mcp_activity_service ON mcp_connection_activity
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE mcp_connections IS 'Tracks active and historical MCP client connections for dashboard visibility';
COMMENT ON TABLE device_auth_sessions IS 'Persistent storage for OAuth2 Device Authorization Grant (RFC 8628) sessions';
COMMENT ON TABLE mcp_connection_activity IS 'Detailed activity log for MCP connections including tool usage';
COMMENT ON VIEW v_active_mcp_connections IS 'Active MCP connections with computed fields for dashboard';
COMMENT ON VIEW v_mcp_connection_stats IS 'Aggregated MCP connection statistics per organization';
