-- Agent Promotion Pipeline and DLP Policy Engine tables
-- Implements the remaining 3 architecture gaps

-- ============================================================================
-- Agent Versions table
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Version identity
    agent_type TEXT NOT NULL,
    version TEXT NOT NULL,  -- Semantic version (e.g., "1.2.3")
    config_hash TEXT NOT NULL,

    -- Multi-tenant
    org_id TEXT NOT NULL,

    -- Version metadata
    changelog TEXT,
    breaking_changes BOOLEAN DEFAULT FALSE,
    baseline_metrics JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by TEXT NOT NULL,

    UNIQUE(org_id, agent_type, version)
);

CREATE INDEX idx_agent_versions_org_type ON agent_versions(org_id, agent_type);
CREATE INDEX idx_agent_versions_created ON agent_versions(created_at DESC);

-- Enable RLS
ALTER TABLE agent_versions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "org_isolation_agent_versions" ON agent_versions
    FOR ALL USING (org_id = current_setting('app.current_org_id', TRUE));

CREATE POLICY "service_role_agent_versions" ON agent_versions
    FOR ALL USING (auth.role() = 'service_role');

COMMENT ON TABLE agent_versions IS 'Registered agent versions for promotion pipeline';

-- ============================================================================
-- Environment Deployments table
-- ============================================================================

CREATE TABLE IF NOT EXISTS environment_deployments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Deployment identity
    agent_type TEXT NOT NULL,
    environment TEXT NOT NULL,  -- development, staging, production
    version_id UUID NOT NULL REFERENCES agent_versions(id),

    -- Multi-tenant
    org_id TEXT NOT NULL,

    -- Rollout configuration
    rollout_strategy TEXT DEFAULT 'immediate',  -- immediate, canary, blue_green, a_b_test
    rollout_percentage FLOAT DEFAULT 100.0,

    -- Health status
    is_healthy BOOLEAN DEFAULT TRUE,
    last_health_check TIMESTAMP WITH TIME ZONE,
    error_rate FLOAT DEFAULT 0.0,

    -- Previous version for rollback
    previous_version_id UUID REFERENCES agent_versions(id),

    -- Timestamps
    deployed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deployed_by TEXT NOT NULL,

    UNIQUE(org_id, agent_type, environment)
);

CREATE INDEX idx_env_deployments_org_env ON environment_deployments(org_id, environment);
CREATE INDEX idx_env_deployments_agent ON environment_deployments(agent_type);

-- Enable RLS
ALTER TABLE environment_deployments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "org_isolation_env_deployments" ON environment_deployments
    FOR ALL USING (org_id = current_setting('app.current_org_id', TRUE));

CREATE POLICY "service_role_env_deployments" ON environment_deployments
    FOR ALL USING (auth.role() = 'service_role');

COMMENT ON TABLE environment_deployments IS 'Current deployment state for each agent/environment';

-- ============================================================================
-- Promotion Requests table
-- ============================================================================

CREATE TABLE IF NOT EXISTS promotion_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Request identity
    request_id TEXT UNIQUE NOT NULL,
    agent_type TEXT NOT NULL,
    version_id UUID NOT NULL REFERENCES agent_versions(id),

    -- Environments
    source_environment TEXT NOT NULL,
    target_environment TEXT NOT NULL,

    -- Multi-tenant
    org_id TEXT NOT NULL,

    -- Request metadata
    requested_by TEXT NOT NULL,
    reason TEXT,
    rollout_strategy TEXT DEFAULT 'immediate',

    -- Approval workflow
    status TEXT DEFAULT 'pending',  -- pending, approved, in_progress, completed, failed, rolled_back, rejected
    approved_by TEXT,
    approved_at TIMESTAMP WITH TIME ZONE,

    -- Execution details
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,

    -- Gate results
    gate_results JSONB DEFAULT '{}',

    -- Timestamps
    requested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_promotion_requests_org ON promotion_requests(org_id);
CREATE INDEX idx_promotion_requests_status ON promotion_requests(status);
CREATE INDEX idx_promotion_requests_agent ON promotion_requests(agent_type);

-- Enable RLS
ALTER TABLE promotion_requests ENABLE ROW LEVEL SECURITY;

CREATE POLICY "org_isolation_promotion_requests" ON promotion_requests
    FOR ALL USING (org_id = current_setting('app.current_org_id', TRUE));

CREATE POLICY "service_role_promotion_requests" ON promotion_requests
    FOR ALL USING (auth.role() = 'service_role');

COMMENT ON TABLE promotion_requests IS 'Agent promotion requests with approval workflow';

-- ============================================================================
-- Regression Gate Results table
-- ============================================================================

CREATE TABLE IF NOT EXISTS regression_gate_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Gate identity
    suite_id TEXT NOT NULL,
    suite_name TEXT NOT NULL,

    -- Reference
    promotion_request_id UUID REFERENCES promotion_requests(id),

    -- Multi-tenant
    org_id TEXT NOT NULL,

    -- Results
    status TEXT NOT NULL,  -- pending, running, passed, failed, skipped, warning
    results JSONB DEFAULT '[]',

    -- Summary
    total_gates INTEGER DEFAULT 0,
    passed_gates INTEGER DEFAULT 0,
    failed_gates INTEGER DEFAULT 0,
    warning_gates INTEGER DEFAULT 0,

    -- Execution
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds FLOAT DEFAULT 0.0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_regression_gates_org ON regression_gate_results(org_id);
CREATE INDEX idx_regression_gates_promotion ON regression_gate_results(promotion_request_id);
CREATE INDEX idx_regression_gates_status ON regression_gate_results(status);

-- Enable RLS
ALTER TABLE regression_gate_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "org_isolation_regression_gates" ON regression_gate_results
    FOR ALL USING (org_id = current_setting('app.current_org_id', TRUE));

CREATE POLICY "service_role_regression_gates" ON regression_gate_results
    FOR ALL USING (auth.role() = 'service_role');

COMMENT ON TABLE regression_gate_results IS 'Regression gate execution results';

-- ============================================================================
-- DLP Policies table
-- ============================================================================

CREATE TABLE IF NOT EXISTS dlp_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Policy identity
    policy_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    version TEXT DEFAULT '1.0.0',

    -- Multi-tenant
    org_id TEXT NOT NULL,

    -- Policy definition
    rules JSONB DEFAULT '[]',
    compliance_frameworks TEXT[] DEFAULT '{}',

    -- Status
    enabled BOOLEAN DEFAULT TRUE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by TEXT NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE,

    UNIQUE(org_id, policy_id)
);

CREATE INDEX idx_dlp_policies_org ON dlp_policies(org_id);
CREATE INDEX idx_dlp_policies_enabled ON dlp_policies(enabled) WHERE enabled = TRUE;

-- Enable RLS
ALTER TABLE dlp_policies ENABLE ROW LEVEL SECURITY;

CREATE POLICY "org_isolation_dlp_policies" ON dlp_policies
    FOR ALL USING (org_id = current_setting('app.current_org_id', TRUE));

CREATE POLICY "service_role_dlp_policies" ON dlp_policies
    FOR ALL USING (auth.role() = 'service_role');

COMMENT ON TABLE dlp_policies IS 'DLP policy definitions';

-- ============================================================================
-- DLP Violations table
-- ============================================================================

CREATE TABLE IF NOT EXISTS dlp_violations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Violation identity
    violation_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    rule_name TEXT,
    policy_id TEXT NOT NULL,

    -- Multi-tenant
    org_id TEXT NOT NULL,

    -- Detection details
    pii_type TEXT,
    matched_pattern TEXT,
    matched_text TEXT,  -- Masked version
    original_hash TEXT,  -- SHA256 hash for audit

    -- Location
    field_path TEXT,
    data_source TEXT,  -- API endpoint, file, etc.

    -- Context
    scope TEXT DEFAULT 'all',  -- inbound, outbound, internal
    data_classification TEXT DEFAULT 'restricted',

    -- Action taken
    action_taken TEXT NOT NULL,
    action_details TEXT,

    -- Request context
    request_id TEXT,
    user_id TEXT,

    -- Metadata
    confidence FLOAT DEFAULT 1.0,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_dlp_violations_org ON dlp_violations(org_id);
CREATE INDEX idx_dlp_violations_policy ON dlp_violations(policy_id);
CREATE INDEX idx_dlp_violations_pii_type ON dlp_violations(pii_type);
CREATE INDEX idx_dlp_violations_action ON dlp_violations(action_taken);
CREATE INDEX idx_dlp_violations_detected ON dlp_violations(detected_at DESC);

-- Partitioning hint for production
COMMENT ON TABLE dlp_violations IS 'DLP violation audit log - consider partitioning by detected_at for large deployments';

-- Enable RLS
ALTER TABLE dlp_violations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "org_isolation_dlp_violations" ON dlp_violations
    FOR ALL USING (org_id = current_setting('app.current_org_id', TRUE));

CREATE POLICY "service_role_dlp_violations" ON dlp_violations
    FOR ALL USING (auth.role() = 'service_role');

-- ============================================================================
-- DLP Scan Results table
-- ============================================================================

CREATE TABLE IF NOT EXISTS dlp_scan_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Scan identity
    scan_id TEXT NOT NULL,

    -- Multi-tenant
    org_id TEXT NOT NULL,

    -- Results summary
    blocked BOOLEAN DEFAULT FALSE,
    total_fields_scanned INTEGER DEFAULT 0,
    total_violations INTEGER DEFAULT 0,
    violations_by_type JSONB DEFAULT '{}',
    actions_taken JSONB DEFAULT '{}',

    -- Performance
    scan_duration_ms FLOAT DEFAULT 0.0,

    -- Context
    data_source TEXT,
    request_id TEXT,
    user_id TEXT,
    policies_applied TEXT[] DEFAULT '{}',

    -- Timestamps
    scanned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_dlp_scans_org ON dlp_scan_results(org_id);
CREATE INDEX idx_dlp_scans_blocked ON dlp_scan_results(blocked) WHERE blocked = TRUE;
CREATE INDEX idx_dlp_scans_scanned ON dlp_scan_results(scanned_at DESC);

-- Enable RLS
ALTER TABLE dlp_scan_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "org_isolation_dlp_scans" ON dlp_scan_results
    FOR ALL USING (org_id = current_setting('app.current_org_id', TRUE));

CREATE POLICY "service_role_dlp_scans" ON dlp_scan_results
    FOR ALL USING (auth.role() = 'service_role');

COMMENT ON TABLE dlp_scan_results IS 'DLP scan execution results';

-- ============================================================================
-- Compliance Reports table
-- ============================================================================

CREATE TABLE IF NOT EXISTS compliance_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Report identity
    report_id TEXT UNIQUE NOT NULL,
    framework TEXT NOT NULL,  -- GDPR, HIPAA, PCI-DSS, etc.

    -- Multi-tenant
    org_id TEXT NOT NULL,

    -- Report content
    summary JSONB DEFAULT '{}',
    violations_by_type JSONB DEFAULT '{}',
    policies_applied TEXT[] DEFAULT '{}',
    recommendations TEXT[] DEFAULT '{}',

    -- Report period
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,

    -- Metadata
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    generated_by TEXT NOT NULL
);

CREATE INDEX idx_compliance_reports_org ON compliance_reports(org_id);
CREATE INDEX idx_compliance_reports_framework ON compliance_reports(framework);
CREATE INDEX idx_compliance_reports_generated ON compliance_reports(generated_at DESC);

-- Enable RLS
ALTER TABLE compliance_reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY "org_isolation_compliance_reports" ON compliance_reports
    FOR ALL USING (org_id = current_setting('app.current_org_id', TRUE));

CREATE POLICY "service_role_compliance_reports" ON compliance_reports
    FOR ALL USING (auth.role() = 'service_role');

COMMENT ON TABLE compliance_reports IS 'Generated compliance reports for DLP auditing';

-- ============================================================================
-- Baseline Metrics table (for regression gates)
-- ============================================================================

CREATE TABLE IF NOT EXISTS baseline_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Baseline identity
    context TEXT NOT NULL,  -- e.g., "agent_CodeAnalyzer_production"

    -- Multi-tenant
    org_id TEXT NOT NULL,

    -- Metrics
    metrics JSONB NOT NULL,

    -- Timestamps
    set_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    set_by TEXT NOT NULL,

    UNIQUE(org_id, context)
);

CREATE INDEX idx_baseline_metrics_org ON baseline_metrics(org_id);

-- Enable RLS
ALTER TABLE baseline_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "org_isolation_baseline_metrics" ON baseline_metrics
    FOR ALL USING (org_id = current_setting('app.current_org_id', TRUE));

CREATE POLICY "service_role_baseline_metrics" ON baseline_metrics
    FOR ALL USING (auth.role() = 'service_role');

COMMENT ON TABLE baseline_metrics IS 'Performance baselines for regression gate comparisons';
