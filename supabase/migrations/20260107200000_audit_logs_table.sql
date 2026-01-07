-- Audit Logs Table for Enterprise Compliance
-- Tracks all significant user and system actions

-- ============================================================================
-- AUDIT LOGS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,

    -- Actor information
    user_id TEXT NOT NULL,  -- Clerk user ID
    user_email TEXT,
    user_role TEXT,

    -- Action details
    action TEXT NOT NULL CHECK (action IN (
        -- Team actions
        'member.invite',
        'member.accept',
        'member.remove',
        'member.role_change',

        -- API Key actions
        'api_key.create',
        'api_key.rotate',
        'api_key.revoke',
        'api_key.use',

        -- Project actions
        'project.create',
        'project.update',
        'project.delete',
        'project.settings_change',

        -- Test actions
        'test.generate',
        'test.approve',
        'test.reject',
        'test.run',

        -- Webhook actions
        'webhook.receive',
        'webhook.process',
        'webhook.fail',

        -- Self-healing actions
        'healing.apply',
        'healing.learn',
        'healing.reject',

        -- Security actions
        'auth.login',
        'auth.logout',
        'auth.mfa_enable',
        'auth.mfa_disable',
        'auth.password_change',

        -- Organization actions
        'org.create',
        'org.update',
        'org.plan_change',
        'org.settings_change',

        -- System actions
        'system.error',
        'system.config_change'
    )),

    -- Resource affected
    resource_type TEXT NOT NULL CHECK (resource_type IN (
        'organization',
        'member',
        'project',
        'api_key',
        'test',
        'event',
        'webhook',
        'healing_pattern',
        'settings',
        'system'
    )),
    resource_id TEXT,

    -- Context
    description TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',

    -- Request context
    ip_address INET,
    user_agent TEXT,
    request_id TEXT,

    -- Outcome
    status TEXT NOT NULL DEFAULT 'success' CHECK (status IN ('success', 'failure', 'pending')),
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_audit_logs_org ON audit_logs(organization_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_status ON audit_logs(status);

-- Index for date range queries with action filter
CREATE INDEX IF NOT EXISTS idx_audit_logs_action_date ON audit_logs(action, created_at DESC);

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Users can view audit logs for their organizations
DROP POLICY IF EXISTS "Users can view audit logs for their organizations" ON audit_logs;
CREATE POLICY "Users can view audit logs for their organizations" ON audit_logs
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_members.organization_id = audit_logs.organization_id
            AND organization_members.user_id = current_setting('app.user_id', true)
            AND organization_members.role IN ('owner', 'admin')
        )
    );

-- Service role can insert audit logs
DROP POLICY IF EXISTS "Service role can manage audit logs" ON audit_logs;
CREATE POLICY "Service role can manage audit logs" ON audit_logs
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- FUNCTION: Create audit log entry
-- ============================================================================

CREATE OR REPLACE FUNCTION create_audit_log(
    p_organization_id UUID,
    p_user_id TEXT,
    p_user_email TEXT,
    p_action TEXT,
    p_resource_type TEXT,
    p_resource_id TEXT,
    p_description TEXT,
    p_metadata JSONB DEFAULT '{}',
    p_ip_address INET DEFAULT NULL,
    p_user_agent TEXT DEFAULT NULL,
    p_status TEXT DEFAULT 'success',
    p_error_message TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_audit_id UUID;
    v_user_role TEXT;
BEGIN
    -- Get user role if in organization
    SELECT role INTO v_user_role
    FROM organization_members
    WHERE organization_id = p_organization_id AND user_id = p_user_id;

    INSERT INTO audit_logs (
        organization_id, user_id, user_email, user_role,
        action, resource_type, resource_id, description,
        metadata, ip_address, user_agent, status, error_message
    ) VALUES (
        p_organization_id, p_user_id, p_user_email, v_user_role,
        p_action, p_resource_type, p_resource_id, p_description,
        p_metadata, p_ip_address, p_user_agent, p_status, p_error_message
    )
    RETURNING id INTO v_audit_id;

    RETURN v_audit_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- SELF-HEALING CONFIGURATION TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS self_healing_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,

    -- Feature toggles
    enabled BOOLEAN DEFAULT true,
    auto_apply BOOLEAN DEFAULT false,  -- Auto-apply fixes without approval

    -- Confidence thresholds
    min_confidence_auto NUMERIC(3,2) DEFAULT 0.95,  -- Min confidence for auto-apply
    min_confidence_suggest NUMERIC(3,2) DEFAULT 0.70,  -- Min confidence to suggest

    -- Selector healing options
    heal_selectors BOOLEAN DEFAULT true,
    max_selector_variations INT DEFAULT 9,
    preferred_selector_strategies TEXT[] DEFAULT ARRAY['id', 'data-testid', 'role', 'text', 'css'],

    -- Wait time healing
    heal_timeouts BOOLEAN DEFAULT true,
    max_wait_time_ms INT DEFAULT 30000,

    -- Text healing
    heal_text_content BOOLEAN DEFAULT true,
    text_similarity_threshold NUMERIC(3,2) DEFAULT 0.85,

    -- Learning options
    learn_from_success BOOLEAN DEFAULT true,
    learn_from_manual_fixes BOOLEAN DEFAULT true,
    share_patterns_across_projects BOOLEAN DEFAULT false,

    -- Notification preferences
    notify_on_heal BOOLEAN DEFAULT true,
    notify_on_suggestion BOOLEAN DEFAULT true,
    notification_channels JSONB DEFAULT '{"email": true, "slack": false}',

    -- Approval workflow
    require_approval BOOLEAN DEFAULT true,
    auto_approve_after_hours INT,  -- Auto-approve after N hours
    approvers TEXT[],  -- User IDs who can approve

    -- Rate limits
    max_heals_per_hour INT DEFAULT 50,
    max_heals_per_test INT DEFAULT 5,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(organization_id, project_id)
);

CREATE INDEX IF NOT EXISTS idx_healing_config_org ON self_healing_config(organization_id);
CREATE INDEX IF NOT EXISTS idx_healing_config_project ON self_healing_config(project_id);

-- Enable RLS
ALTER TABLE self_healing_config ENABLE ROW LEVEL SECURITY;

-- Users can view/update their org's healing config
DROP POLICY IF EXISTS "Users can manage healing config" ON self_healing_config;
CREATE POLICY "Users can manage healing config" ON self_healing_config
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_members.organization_id = self_healing_config.organization_id
            AND organization_members.user_id = current_setting('app.user_id', true)
            AND organization_members.role IN ('owner', 'admin')
        )
    );

DROP POLICY IF EXISTS "Service role has full access to healing config" ON self_healing_config;
CREATE POLICY "Service role has full access to healing config" ON self_healing_config
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- SEED DEFAULT CONFIGS
-- ============================================================================

-- Insert default healing config for existing organizations
INSERT INTO self_healing_config (organization_id)
SELECT id FROM organizations
WHERE NOT EXISTS (
    SELECT 1 FROM self_healing_config WHERE organization_id = organizations.id AND project_id IS NULL
)
ON CONFLICT DO NOTHING;
