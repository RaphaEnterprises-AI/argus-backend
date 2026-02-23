-- =============================================================================
-- Test Quality Intelligence Tables
-- Supports heal validation pipeline and trust score computation
-- =============================================================================

-- heal_validations: Records every heal validation decision
CREATE TABLE IF NOT EXISTS heal_validations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL,
    project_id UUID NOT NULL,
    test_id TEXT NOT NULL,
    failure_id TEXT,
    healing_request_id TEXT,

    -- Heal details
    healing_method TEXT,  -- cache, cognee, code_aware, llm_analysis
    original_selector TEXT,
    healed_selector TEXT,
    diagnosis_type TEXT,  -- selector_changed, timing_issue, real_bug, etc.

    -- Validation result
    heal_quality TEXT NOT NULL CHECK (heal_quality IN (
        'valid_heal', 'suspicious_heal', 'regression_masking', 'needs_human_review'
    )),
    blocking BOOLEAN NOT NULL DEFAULT false,
    auto_apply_allowed BOOLEAN NOT NULL DEFAULT true,

    -- AgentAsJudge signals
    addresses_root_cause BOOLEAN,
    preserves_test_intent BOOLEAN,
    has_side_effects BOOLEAN,
    judge_confidence FLOAT,
    explanation TEXT,
    alternatives JSONB DEFAULT '[]'::jsonb,

    -- Assertion analysis (rule-based)
    assertion_weakening_detected BOOLEAN DEFAULT false,
    original_assertion_strength TEXT,  -- exact_match, contains, exists, no_assertion
    healed_assertion_strength TEXT,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_heal_validations_test_id ON heal_validations (test_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_heal_validations_org_quality ON heal_validations (organization_id, heal_quality, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_heal_validations_project ON heal_validations (project_id, created_at DESC);

-- RLS
ALTER TABLE heal_validations ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS heal_validations_org_isolation ON heal_validations;
CREATE POLICY heal_validations_org_isolation ON heal_validations
    FOR ALL
    USING (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id')
    WITH CHECK (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id');

-- Service role bypass
DROP POLICY IF EXISTS heal_validations_service_role ON heal_validations;
CREATE POLICY heal_validations_service_role ON heal_validations
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);


-- heal_monitoring: Tracks post-heal failure rate changes
CREATE TABLE IF NOT EXISTS heal_monitoring (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL,
    project_id UUID NOT NULL,
    test_id TEXT NOT NULL,
    heal_validation_id UUID REFERENCES heal_validations(id) ON DELETE CASCADE,

    -- Failure rates
    failure_rate_before FLOAT,       -- Failure rate in 7 days before heal
    failure_rate_7d_after FLOAT,     -- Failure rate 7 days after heal
    failure_rate_30d_after FLOAT,    -- Failure rate 30 days after heal

    -- Monitoring status
    monitoring_status TEXT NOT NULL DEFAULT 'watching' CHECK (monitoring_status IN (
        'watching', 'alerted', 'closed'
    )),
    masking_alert_sent BOOLEAN DEFAULT false,
    alert_sent_at TIMESTAMPTZ,

    -- Timestamps
    heal_applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_checked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_heal_monitoring_status ON heal_monitoring (monitoring_status, last_checked_at);
CREATE INDEX IF NOT EXISTS idx_heal_monitoring_test ON heal_monitoring (test_id, heal_applied_at DESC);
CREATE INDEX IF NOT EXISTS idx_heal_monitoring_project ON heal_monitoring (project_id, monitoring_status);

-- RLS
ALTER TABLE heal_monitoring ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS heal_monitoring_org_isolation ON heal_monitoring;
CREATE POLICY heal_monitoring_org_isolation ON heal_monitoring
    FOR ALL
    USING (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id')
    WITH CHECK (organization_id::text = current_setting('request.jwt.claims', true)::json->>'organization_id');

DROP POLICY IF EXISTS heal_monitoring_service_role ON heal_monitoring;
CREATE POLICY heal_monitoring_service_role ON heal_monitoring
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);


-- updated_at trigger
CREATE OR REPLACE FUNCTION update_heal_tables_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS heal_validations_updated_at ON heal_validations;
CREATE TRIGGER heal_validations_updated_at
    BEFORE UPDATE ON heal_validations
    FOR EACH ROW EXECUTE FUNCTION update_heal_tables_updated_at();

DROP TRIGGER IF EXISTS heal_monitoring_updated_at ON heal_monitoring;
CREATE TRIGGER heal_monitoring_updated_at
    BEFORE UPDATE ON heal_monitoring
    FOR EACH ROW EXECUTE FUNCTION update_heal_tables_updated_at();
