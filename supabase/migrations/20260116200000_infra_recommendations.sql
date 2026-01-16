-- Infrastructure Recommendations Table
-- Stores AI-generated infrastructure optimization recommendations

-- Create enum types for recommendation attributes
CREATE TYPE infra_recommendation_type AS ENUM (
    'scale_down',
    'scale_up',
    'right_size',
    'schedule_scaling',
    'cleanup_sessions',
    'cost_alert',
    'anomaly'
);

CREATE TYPE infra_recommendation_priority AS ENUM (
    'critical',
    'high',
    'medium',
    'low'
);

CREATE TYPE infra_recommendation_status AS ENUM (
    'pending',
    'approved',
    'rejected',
    'auto_applied',
    'expired'
);

-- Main recommendations table
CREATE TABLE IF NOT EXISTS infra_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,

    -- Recommendation details
    type infra_recommendation_type NOT NULL,
    priority infra_recommendation_priority NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    reasoning TEXT,

    -- Cost impact
    estimated_savings_monthly DECIMAL(10,2) DEFAULT 0,
    confidence DECIMAL(3,2) DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),

    -- Action to take
    action JSONB NOT NULL DEFAULT '{}',
    metrics_snapshot JSONB DEFAULT '{}',

    -- Status tracking
    status infra_recommendation_status NOT NULL DEFAULT 'pending',
    applied_at TIMESTAMPTZ,
    applied_by UUID REFERENCES users(id),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '7 days'),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_infra_recommendations_org ON infra_recommendations(org_id);
CREATE INDEX idx_infra_recommendations_status ON infra_recommendations(status);
CREATE INDEX idx_infra_recommendations_type ON infra_recommendations(type);
CREATE INDEX idx_infra_recommendations_priority ON infra_recommendations(priority);
CREATE INDEX idx_infra_recommendations_created ON infra_recommendations(created_at DESC);

-- Index for finding pending recommendations
CREATE INDEX idx_infra_recommendations_pending ON infra_recommendations(org_id, status)
    WHERE status = 'pending';

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_infra_recommendations_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_infra_recommendations_updated_at
    BEFORE UPDATE ON infra_recommendations
    FOR EACH ROW
    EXECUTE FUNCTION update_infra_recommendations_updated_at();

-- Auto-expire old recommendations
CREATE OR REPLACE FUNCTION expire_old_recommendations()
RETURNS void AS $$
BEGIN
    UPDATE infra_recommendations
    SET status = 'expired'
    WHERE status = 'pending'
      AND expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- Infrastructure Cost History Table
-- Tracks actual infrastructure costs over time
CREATE TABLE IF NOT EXISTS infra_cost_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,

    -- Cost data
    date DATE NOT NULL,
    total_cost DECIMAL(10,2) NOT NULL DEFAULT 0,
    compute_cost DECIMAL(10,2) NOT NULL DEFAULT 0,
    network_cost DECIMAL(10,2) NOT NULL DEFAULT 0,
    storage_cost DECIMAL(10,2) NOT NULL DEFAULT 0,

    -- Resource counts
    avg_node_count DECIMAL(5,2) DEFAULT 0,
    avg_pod_count DECIMAL(5,2) DEFAULT 0,
    total_sessions INTEGER DEFAULT 0,

    -- Metrics
    avg_cpu_utilization DECIMAL(5,2) DEFAULT 0,
    avg_memory_utilization DECIMAL(5,2) DEFAULT 0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Prevent duplicate entries for same org/date
    UNIQUE(org_id, date)
);

CREATE INDEX idx_infra_cost_history_org_date ON infra_cost_history(org_id, date DESC);

-- Infrastructure Anomaly History
-- Tracks detected anomalies for analysis
CREATE TABLE IF NOT EXISTS infra_anomaly_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,

    type TEXT NOT NULL,
    severity infra_recommendation_priority NOT NULL,
    description TEXT NOT NULL,
    metrics JSONB DEFAULT '{}',
    suggested_action TEXT,

    -- Resolution tracking
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,

    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_infra_anomaly_history_org ON infra_anomaly_history(org_id);
CREATE INDEX idx_infra_anomaly_history_detected ON infra_anomaly_history(detected_at DESC);
CREATE INDEX idx_infra_anomaly_history_unresolved ON infra_anomaly_history(org_id)
    WHERE resolved_at IS NULL;

-- Row Level Security
ALTER TABLE infra_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE infra_cost_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE infra_anomaly_history ENABLE ROW LEVEL SECURITY;

-- RLS Policies (org members can view their org's data)
CREATE POLICY infra_recommendations_org_access ON infra_recommendations
    FOR ALL
    USING (
        org_id IN (
            SELECT organization_id FROM team_memberships
            WHERE user_id = auth.uid()
        )
    );

CREATE POLICY infra_cost_history_org_access ON infra_cost_history
    FOR ALL
    USING (
        org_id IN (
            SELECT organization_id FROM team_memberships
            WHERE user_id = auth.uid()
        )
    );

CREATE POLICY infra_anomaly_history_org_access ON infra_anomaly_history
    FOR ALL
    USING (
        org_id IN (
            SELECT organization_id FROM team_memberships
            WHERE user_id = auth.uid()
        )
    );

-- Comments for documentation
COMMENT ON TABLE infra_recommendations IS 'AI-generated infrastructure optimization recommendations';
COMMENT ON TABLE infra_cost_history IS 'Daily infrastructure cost tracking';
COMMENT ON TABLE infra_anomaly_history IS 'Detected infrastructure anomalies';
COMMENT ON COLUMN infra_recommendations.confidence IS 'AI confidence score (0.0-1.0)';
COMMENT ON COLUMN infra_recommendations.action IS 'JSON action specification to apply';
