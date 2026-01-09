-- =============================================================================
-- Security Audit Tables for SOC2 Compliance
-- Version: 2.0.1
-- Date: 2026-01-09
-- =============================================================================

-- Security Audit Logs (SOC2 requires minimum 1 year retention)
CREATE TABLE IF NOT EXISTS security_audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type TEXT NOT NULL,  -- auth.login.success, authz.permission.denied, etc.
    severity TEXT NOT NULL DEFAULT 'info',  -- debug, info, warning, error, critical

    -- Actor information
    user_id TEXT,
    organization_id UUID REFERENCES organizations(id),
    session_id TEXT,
    api_key_id UUID,

    -- Request context
    request_id TEXT,
    ip_address INET,
    user_agent TEXT,
    method TEXT,
    path TEXT,

    -- Resource information
    resource_type TEXT,
    resource_id TEXT,

    -- Event details
    action TEXT NOT NULL,
    description TEXT,
    outcome TEXT DEFAULT 'success',  -- success, failure, error
    status_code INTEGER,
    duration_ms INTEGER,
    metadata JSONB DEFAULT '{}',

    -- Compliance
    retention_days INTEGER DEFAULT 365,
    is_sensitive BOOLEAN DEFAULT false,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON security_audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON security_audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_org ON security_audit_logs(organization_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON security_audit_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_severity ON security_audit_logs(severity);
CREATE INDEX IF NOT EXISTS idx_audit_logs_ip ON security_audit_logs(ip_address);
CREATE INDEX IF NOT EXISTS idx_audit_logs_request_id ON security_audit_logs(request_id);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_audit_logs_org_created ON security_audit_logs(organization_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_created ON security_audit_logs(user_id, created_at DESC);

-- JWT Token Revocation List (for token invalidation)
CREATE TABLE IF NOT EXISTS revoked_tokens (
    jti TEXT PRIMARY KEY,  -- JWT ID
    user_id TEXT NOT NULL,
    revoked_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_revoked_tokens_user ON revoked_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_expires ON revoked_tokens(expires_at);

-- Session Management (for concurrent session limits)
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    session_token_hash TEXT NOT NULL,
    ip_address INET,
    user_agent TEXT,
    device_fingerprint TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN DEFAULT true
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON user_sessions(user_id, is_active);

-- Rate Limit Tracking (for persistent rate limiting)
CREATE TABLE IF NOT EXISTS rate_limit_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key TEXT NOT NULL,  -- user:xxx, org:xxx, ip:xxx
    endpoint TEXT,
    request_count INTEGER DEFAULT 1,
    window_start TIMESTAMPTZ DEFAULT NOW(),
    window_end TIMESTAMPTZ NOT NULL,
    UNIQUE(key, endpoint, window_start)
);

CREATE INDEX IF NOT EXISTS idx_rate_limit_key ON rate_limit_entries(key);
CREATE INDEX IF NOT EXISTS idx_rate_limit_window ON rate_limit_entries(window_end);

-- Security Alerts (for incident tracking)
CREATE TABLE IF NOT EXISTS security_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_type TEXT NOT NULL,  -- brute_force, suspicious_activity, rate_limit_exceeded
    severity TEXT NOT NULL DEFAULT 'warning',
    user_id TEXT,
    organization_id UUID REFERENCES organizations(id),
    ip_address INET,
    description TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    status TEXT DEFAULT 'open',  -- open, investigating, resolved, false_positive
    resolved_by TEXT,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_type ON security_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON security_alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_org ON security_alerts(organization_id);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON security_alerts(created_at DESC);

-- Permission Audit Trail (for RBAC changes)
CREATE TABLE IF NOT EXISTS permission_changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    organization_id UUID REFERENCES organizations(id),
    target_user_id TEXT,  -- User whose permissions changed
    change_type TEXT NOT NULL,  -- role_assigned, role_removed, permission_granted, permission_revoked
    old_value JSONB,
    new_value JSONB,
    changed_by TEXT NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_perm_changes_user ON permission_changes(target_user_id);
CREATE INDEX IF NOT EXISTS idx_perm_changes_org ON permission_changes(organization_id);
CREATE INDEX IF NOT EXISTS idx_perm_changes_created ON permission_changes(created_at DESC);

-- Data Access Log (for sensitive data access tracking)
CREATE TABLE IF NOT EXISTS data_access_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    organization_id UUID REFERENCES organizations(id),
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    access_type TEXT NOT NULL,  -- read, write, delete, export
    fields_accessed TEXT[],  -- Which fields were accessed
    data_classification TEXT,  -- public, internal, confidential, restricted
    purpose TEXT,
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_data_access_user ON data_access_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_data_access_resource ON data_access_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_data_access_created ON data_access_logs(created_at DESC);

-- =============================================================================
-- Row Level Security Policies
-- =============================================================================

-- Enable RLS on security tables
ALTER TABLE security_audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE revoked_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE permission_changes ENABLE ROW LEVEL SECURITY;
ALTER TABLE data_access_logs ENABLE ROW LEVEL SECURITY;

-- Security audit logs: Only admins can read, system can insert
CREATE POLICY security_audit_logs_read ON security_audit_logs
    FOR SELECT
    USING (
        auth.uid()::text IN (
            SELECT user_id FROM organization_members
            WHERE role IN ('owner', 'admin')
            AND organization_id = security_audit_logs.organization_id
        )
    );

CREATE POLICY security_audit_logs_insert ON security_audit_logs
    FOR INSERT
    WITH CHECK (true);  -- Allow system inserts

-- User sessions: Users can only see their own sessions
CREATE POLICY user_sessions_policy ON user_sessions
    FOR ALL
    USING (user_id = auth.uid()::text);

-- Security alerts: Admins can see org alerts
CREATE POLICY security_alerts_read ON security_alerts
    FOR SELECT
    USING (
        auth.uid()::text IN (
            SELECT user_id FROM organization_members
            WHERE role IN ('owner', 'admin')
            AND organization_id = security_alerts.organization_id
        )
    );

-- =============================================================================
-- Cleanup Functions
-- =============================================================================

-- Function to clean up expired tokens
CREATE OR REPLACE FUNCTION cleanup_expired_tokens()
RETURNS void AS $$
BEGIN
    DELETE FROM revoked_tokens WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS void AS $$
BEGIN
    UPDATE user_sessions SET is_active = false WHERE expires_at < NOW() AND is_active = true;
    DELETE FROM user_sessions WHERE expires_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old rate limit entries
CREATE OR REPLACE FUNCTION cleanup_rate_limit_entries()
RETURNS void AS $$
BEGIN
    DELETE FROM rate_limit_entries WHERE window_end < NOW() - INTERVAL '1 hour';
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Comments for documentation
-- =============================================================================

COMMENT ON TABLE security_audit_logs IS 'SOC2 compliant security audit log with 1 year retention';
COMMENT ON TABLE revoked_tokens IS 'JWT tokens that have been revoked before expiration';
COMMENT ON TABLE user_sessions IS 'Active user sessions for concurrent session management';
COMMENT ON TABLE rate_limit_entries IS 'Persistent rate limiting data';
COMMENT ON TABLE security_alerts IS 'Security incidents and alerts for investigation';
COMMENT ON TABLE permission_changes IS 'Audit trail for RBAC permission changes';
COMMENT ON TABLE data_access_logs IS 'Sensitive data access tracking for compliance';
