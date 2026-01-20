-- =============================================================================
-- Add AI-specific columns to security_audit_logs
-- Enables cloud-based audit logging for AI interactions
-- Version: 1.0.0
-- Date: 2026-01-20
-- =============================================================================

-- Add AI interaction columns
ALTER TABLE security_audit_logs
ADD COLUMN IF NOT EXISTS model TEXT,
ADD COLUMN IF NOT EXISTS input_tokens INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS output_tokens INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS cost_usd NUMERIC(10, 6) DEFAULT 0,
ADD COLUMN IF NOT EXISTS content_hash TEXT,
ADD COLUMN IF NOT EXISTS data_classification TEXT DEFAULT 'internal';

-- Add index for AI cost analysis
CREATE INDEX IF NOT EXISTS idx_audit_logs_model ON security_audit_logs(model) WHERE model IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_audit_logs_cost ON security_audit_logs(cost_usd) WHERE cost_usd > 0;

-- Add index for data classification filtering
CREATE INDEX IF NOT EXISTS idx_audit_logs_classification ON security_audit_logs(data_classification);

-- =============================================================================
-- Function: Insert AI audit event (for use from Python backend)
-- =============================================================================

CREATE OR REPLACE FUNCTION insert_ai_audit_event(
    p_event_type TEXT,
    p_severity TEXT DEFAULT 'info',
    p_user_id TEXT DEFAULT NULL,
    p_organization_id UUID DEFAULT NULL,
    p_session_id TEXT DEFAULT NULL,
    p_action TEXT DEFAULT '',
    p_resource_type TEXT DEFAULT NULL,
    p_resource_id TEXT DEFAULT NULL,
    p_description TEXT DEFAULT NULL,
    p_outcome TEXT DEFAULT 'success',
    p_metadata JSONB DEFAULT '{}',
    p_model TEXT DEFAULT NULL,
    p_input_tokens INTEGER DEFAULT 0,
    p_output_tokens INTEGER DEFAULT 0,
    p_cost_usd NUMERIC DEFAULT 0,
    p_content_hash TEXT DEFAULT NULL,
    p_data_classification TEXT DEFAULT 'internal',
    p_ip_address INET DEFAULT NULL,
    p_duration_ms INTEGER DEFAULT NULL,
    p_retention_days INTEGER DEFAULT 90
) RETURNS UUID AS $$
DECLARE
    v_audit_id UUID;
BEGIN
    INSERT INTO security_audit_logs (
        event_type,
        severity,
        user_id,
        organization_id,
        session_id,
        action,
        resource_type,
        resource_id,
        description,
        outcome,
        metadata,
        model,
        input_tokens,
        output_tokens,
        cost_usd,
        content_hash,
        data_classification,
        ip_address,
        duration_ms,
        retention_days
    ) VALUES (
        p_event_type,
        p_severity,
        p_user_id,
        p_organization_id,
        p_session_id,
        p_action,
        p_resource_type,
        p_resource_id,
        p_description,
        p_outcome,
        p_metadata,
        p_model,
        p_input_tokens,
        p_output_tokens,
        p_cost_usd,
        p_content_hash,
        p_data_classification,
        p_ip_address,
        p_duration_ms,
        p_retention_days
    )
    RETURNING id INTO v_audit_id;

    RETURN v_audit_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION insert_ai_audit_event TO authenticated;
GRANT EXECUTE ON FUNCTION insert_ai_audit_event TO service_role;

-- =============================================================================
-- Function: Query AI audit events with filtering
-- =============================================================================

CREATE OR REPLACE FUNCTION query_ai_audit_events(
    p_organization_id UUID DEFAULT NULL,
    p_user_id TEXT DEFAULT NULL,
    p_event_type TEXT DEFAULT NULL,
    p_model TEXT DEFAULT NULL,
    p_start_date TIMESTAMPTZ DEFAULT NULL,
    p_end_date TIMESTAMPTZ DEFAULT NULL,
    p_limit INTEGER DEFAULT 1000
) RETURNS TABLE (
    id UUID,
    event_type TEXT,
    severity TEXT,
    user_id TEXT,
    session_id TEXT,
    action TEXT,
    resource_type TEXT,
    resource_id TEXT,
    description TEXT,
    outcome TEXT,
    metadata JSONB,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd NUMERIC,
    content_hash TEXT,
    data_classification TEXT,
    duration_ms INTEGER,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.id,
        s.event_type,
        s.severity,
        s.user_id,
        s.session_id,
        s.action,
        s.resource_type,
        s.resource_id,
        s.description,
        s.outcome,
        s.metadata,
        s.model,
        s.input_tokens,
        s.output_tokens,
        s.cost_usd,
        s.content_hash,
        s.data_classification,
        s.duration_ms,
        s.created_at
    FROM security_audit_logs s
    WHERE
        (p_organization_id IS NULL OR s.organization_id = p_organization_id)
        AND (p_user_id IS NULL OR s.user_id = p_user_id)
        AND (p_event_type IS NULL OR s.event_type = p_event_type)
        AND (p_model IS NULL OR s.model = p_model)
        AND (p_start_date IS NULL OR s.created_at >= p_start_date)
        AND (p_end_date IS NULL OR s.created_at <= p_end_date)
    ORDER BY s.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION query_ai_audit_events TO authenticated;
GRANT EXECUTE ON FUNCTION query_ai_audit_events TO service_role;

-- =============================================================================
-- Function: Generate compliance report
-- =============================================================================

CREATE OR REPLACE FUNCTION generate_audit_compliance_report(
    p_organization_id UUID,
    p_start_date TIMESTAMPTZ,
    p_end_date TIMESTAMPTZ
) RETURNS JSONB AS $$
DECLARE
    v_report JSONB;
BEGIN
    SELECT jsonb_build_object(
        'period', jsonb_build_object(
            'start', p_start_date,
            'end', p_end_date
        ),
        'summary', jsonb_build_object(
            'total_events', COUNT(*),
            'ai_requests', COUNT(*) FILTER (WHERE event_type = 'ai_request'),
            'ai_responses', COUNT(*) FILTER (WHERE event_type = 'ai_response'),
            'ai_errors', COUNT(*) FILTER (WHERE event_type = 'ai_error'),
            'file_reads', COUNT(*) FILTER (WHERE event_type = 'file_read'),
            'secrets_detected', COUNT(*) FILTER (WHERE event_type = 'secret_detected'),
            'tests_run', COUNT(*) FILTER (WHERE event_type LIKE 'test_%'),
            'total_input_tokens', COALESCE(SUM(input_tokens), 0),
            'total_output_tokens', COALESCE(SUM(output_tokens), 0),
            'total_cost_usd', COALESCE(SUM(cost_usd), 0)
        ),
        'by_model', (
            SELECT COALESCE(jsonb_object_agg(model, counts), '{}'::jsonb)
            FROM (
                SELECT model, jsonb_build_object(
                    'requests', COUNT(*),
                    'tokens', SUM(input_tokens + output_tokens),
                    'cost_usd', SUM(cost_usd)
                ) as counts
                FROM security_audit_logs
                WHERE organization_id = p_organization_id
                AND created_at BETWEEN p_start_date AND p_end_date
                AND model IS NOT NULL
                GROUP BY model
            ) m
        ),
        'by_user', (
            SELECT COALESCE(jsonb_object_agg(user_id, counts), '{}'::jsonb)
            FROM (
                SELECT user_id, jsonb_build_object(
                    'events', COUNT(*),
                    'cost_usd', SUM(cost_usd)
                ) as counts
                FROM security_audit_logs
                WHERE organization_id = p_organization_id
                AND created_at BETWEEN p_start_date AND p_end_date
                AND user_id IS NOT NULL
                GROUP BY user_id
            ) u
        ),
        'errors', (
            SELECT COALESCE(jsonb_agg(jsonb_build_object(
                'timestamp', created_at,
                'event_type', event_type,
                'description', description
            )), '[]'::jsonb)
            FROM security_audit_logs
            WHERE organization_id = p_organization_id
            AND created_at BETWEEN p_start_date AND p_end_date
            AND outcome = 'failure'
            LIMIT 100
        )
    ) INTO v_report
    FROM security_audit_logs
    WHERE organization_id = p_organization_id
    AND created_at BETWEEN p_start_date AND p_end_date;

    RETURN v_report;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION generate_audit_compliance_report TO authenticated;
GRANT EXECUTE ON FUNCTION generate_audit_compliance_report TO service_role;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON COLUMN security_audit_logs.model IS 'AI model used (e.g., claude-sonnet-4-5)';
COMMENT ON COLUMN security_audit_logs.input_tokens IS 'Number of input tokens for AI requests';
COMMENT ON COLUMN security_audit_logs.output_tokens IS 'Number of output tokens for AI responses';
COMMENT ON COLUMN security_audit_logs.cost_usd IS 'Cost in USD for AI API calls';
COMMENT ON COLUMN security_audit_logs.content_hash IS 'SHA256 hash of content (never store actual content)';
COMMENT ON COLUMN security_audit_logs.data_classification IS 'Data classification: public, internal, confidential, restricted';
