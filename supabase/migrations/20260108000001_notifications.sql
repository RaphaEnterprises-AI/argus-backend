-- Notification System Tables Migration
-- Adds notification channels, rules, and delivery logs

-- ============================================================================
-- NOTIFICATION CHANNELS TABLE (Slack, email, webhook, etc.)
-- ============================================================================

CREATE TABLE IF NOT EXISTS notification_channels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,

    -- Channel identification
    name TEXT NOT NULL,
    channel_type TEXT NOT NULL CHECK (channel_type IN (
        'slack', 'email', 'webhook', 'discord', 'teams', 'pagerduty', 'opsgenie'
    )),

    -- Channel-specific configuration (encrypted in production)
    config JSONB NOT NULL DEFAULT '{}',
    -- Slack: {"webhook_url": "...", "channel": "#alerts", "mention_users": ["U123"]}
    -- Email: {"recipients": ["team@example.com"], "cc": [], "reply_to": "..."}
    -- Webhook: {"url": "...", "headers": {}, "method": "POST", "secret": "..."}

    -- State
    enabled BOOLEAN DEFAULT true,
    verified BOOLEAN DEFAULT false,
    verification_token TEXT,
    verified_at TIMESTAMPTZ,

    -- Rate limiting
    rate_limit_per_hour INTEGER DEFAULT 100,
    last_sent_at TIMESTAMPTZ,
    sent_today INTEGER DEFAULT 0,

    -- Metadata
    created_by UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_notification_channels_org ON notification_channels(organization_id);
CREATE INDEX IF NOT EXISTS idx_notification_channels_project ON notification_channels(project_id);
CREATE INDEX IF NOT EXISTS idx_notification_channels_type ON notification_channels(channel_type);
CREATE INDEX IF NOT EXISTS idx_notification_channels_enabled ON notification_channels(enabled) WHERE enabled = true;

-- ============================================================================
-- NOTIFICATION RULES TABLE (Which events trigger notifications)
-- ============================================================================

CREATE TABLE IF NOT EXISTS notification_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_id UUID NOT NULL REFERENCES notification_channels(id) ON DELETE CASCADE,

    -- Rule name and description
    name TEXT,
    description TEXT,

    -- Event type that triggers this rule
    event_type TEXT NOT NULL CHECK (event_type IN (
        -- Test events
        'test.run.started',
        'test.run.completed',
        'test.run.failed',
        'test.run.passed',

        -- Schedule events
        'schedule.run.started',
        'schedule.run.completed',
        'schedule.run.failed',
        'schedule.triggered',

        -- Discovery events
        'discovery.completed',
        'discovery.failed',

        -- Self-healing events
        'healing.applied',
        'healing.suggested',
        'healing.failed',

        -- Quality events
        'quality.audit.completed',
        'quality.score.dropped',
        'quality.issue.critical',

        -- Visual regression events
        'visual.mismatch.detected',
        'visual.baseline.updated',

        -- AI insights
        'insight.prediction',
        'insight.anomaly',
        'insight.critical',

        -- System events
        'error.critical',
        'error.rate.high',
        'budget.threshold.reached'
    )),

    -- Conditions for triggering (optional filtering)
    conditions JSONB DEFAULT '{}',
    -- Examples:
    -- {"severity": ["critical", "high"]}
    -- {"project_id": "uuid", "tags": ["smoke"]}
    -- {"threshold": {"type": "pass_rate", "operator": "<", "value": 80}}

    -- Message template (supports variables like {{test_name}})
    message_template TEXT,

    -- Priority for this notification
    priority TEXT DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent')),

    -- Throttling
    cooldown_minutes INTEGER DEFAULT 0,  -- Min time between notifications
    last_triggered_at TIMESTAMPTZ,

    -- State
    enabled BOOLEAN DEFAULT true,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_notification_rules_channel ON notification_rules(channel_id);
CREATE INDEX IF NOT EXISTS idx_notification_rules_event ON notification_rules(event_type);
CREATE INDEX IF NOT EXISTS idx_notification_rules_enabled ON notification_rules(enabled) WHERE enabled = true;

-- ============================================================================
-- NOTIFICATION LOGS TABLE (Delivery history)
-- ============================================================================

CREATE TABLE IF NOT EXISTS notification_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_id UUID NOT NULL REFERENCES notification_channels(id) ON DELETE CASCADE,
    rule_id UUID REFERENCES notification_rules(id) ON DELETE SET NULL,

    -- Event information
    event_type TEXT NOT NULL,
    event_id TEXT,  -- Reference to the source event

    -- Payload sent
    payload JSONB NOT NULL,

    -- Delivery status
    status TEXT DEFAULT 'pending' CHECK (status IN (
        'pending', 'queued', 'sent', 'delivered', 'failed', 'bounced', 'suppressed'
    )),

    -- Response from delivery
    response_code INTEGER,
    response_body TEXT,
    error_message TEXT,

    -- Retry tracking
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    next_retry_at TIMESTAMPTZ,

    -- Timing
    queued_at TIMESTAMPTZ DEFAULT NOW(),
    sent_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_notification_logs_channel ON notification_logs(channel_id);
CREATE INDEX IF NOT EXISTS idx_notification_logs_rule ON notification_logs(rule_id);
CREATE INDEX IF NOT EXISTS idx_notification_logs_event ON notification_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_notification_logs_status ON notification_logs(status);
CREATE INDEX IF NOT EXISTS idx_notification_logs_created ON notification_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notification_logs_pending ON notification_logs(status, next_retry_at)
    WHERE status IN ('pending', 'queued');

-- ============================================================================
-- NOTIFICATION TEMPLATES TABLE (Reusable message templates)
-- ============================================================================

CREATE TABLE IF NOT EXISTS notification_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,

    -- Template identification
    name TEXT NOT NULL,
    description TEXT,

    -- Event types this template applies to
    event_types TEXT[] NOT NULL,

    -- Template content per channel type
    templates JSONB NOT NULL DEFAULT '{}',
    -- Example:
    -- {
    --   "slack": {"text": "...", "blocks": [...]},
    --   "email": {"subject": "...", "html": "...", "text": "..."},
    --   "webhook": {"body": {...}}
    -- }

    -- Variable definitions for template interpolation
    variables JSONB DEFAULT '[]',
    -- [{"name": "test_name", "description": "Name of the test", "default": "Unknown"}]

    -- State
    is_default BOOLEAN DEFAULT false,
    enabled BOOLEAN DEFAULT true,

    -- Metadata
    created_by UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_notification_templates_org ON notification_templates(organization_id);
CREATE INDEX IF NOT EXISTS idx_notification_templates_default ON notification_templates(is_default) WHERE is_default = true;

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE notification_channels ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_templates ENABLE ROW LEVEL SECURITY;

-- Policies for notification_channels
DROP POLICY IF EXISTS "Users can view notification channels for their orgs" ON notification_channels;
CREATE POLICY "Users can view notification channels for their orgs" ON notification_channels
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_members.organization_id = notification_channels.organization_id
            AND organization_members.user_id = current_setting('app.user_id', true)
        )
    );

DROP POLICY IF EXISTS "Admins can manage notification channels" ON notification_channels;
CREATE POLICY "Admins can manage notification channels" ON notification_channels
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_members.organization_id = notification_channels.organization_id
            AND organization_members.user_id = current_setting('app.user_id', true)
            AND organization_members.role IN ('owner', 'admin')
        )
    );

-- Policies for notification_rules
DROP POLICY IF EXISTS "Users can view notification rules" ON notification_rules;
CREATE POLICY "Users can view notification rules" ON notification_rules
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM notification_channels nc
            JOIN organization_members om ON om.organization_id = nc.organization_id
            WHERE nc.id = notification_rules.channel_id
            AND om.user_id = current_setting('app.user_id', true)
        )
    );

DROP POLICY IF EXISTS "Admins can manage notification rules" ON notification_rules;
CREATE POLICY "Admins can manage notification rules" ON notification_rules
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM notification_channels nc
            JOIN organization_members om ON om.organization_id = nc.organization_id
            WHERE nc.id = notification_rules.channel_id
            AND om.user_id = current_setting('app.user_id', true)
            AND om.role IN ('owner', 'admin')
        )
    );

-- Policies for notification_logs
DROP POLICY IF EXISTS "Users can view notification logs" ON notification_logs;
CREATE POLICY "Users can view notification logs" ON notification_logs
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM notification_channels nc
            JOIN organization_members om ON om.organization_id = nc.organization_id
            WHERE nc.id = notification_logs.channel_id
            AND om.user_id = current_setting('app.user_id', true)
        )
    );

-- Policies for notification_templates
DROP POLICY IF EXISTS "Users can view notification templates" ON notification_templates;
CREATE POLICY "Users can view notification templates" ON notification_templates
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_members.organization_id = notification_templates.organization_id
            AND organization_members.user_id = current_setting('app.user_id', true)
        )
    );

DROP POLICY IF EXISTS "Admins can manage notification templates" ON notification_templates;
CREATE POLICY "Admins can manage notification templates" ON notification_templates
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_members.organization_id = notification_templates.organization_id
            AND organization_members.user_id = current_setting('app.user_id', true)
            AND organization_members.role IN ('owner', 'admin')
        )
    );

-- Service role policies
DROP POLICY IF EXISTS "Service role has full access to notification_channels" ON notification_channels;
CREATE POLICY "Service role has full access to notification_channels" ON notification_channels
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role has full access to notification_rules" ON notification_rules;
CREATE POLICY "Service role has full access to notification_rules" ON notification_rules
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role has full access to notification_logs" ON notification_logs;
CREATE POLICY "Service role has full access to notification_logs" ON notification_logs
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role has full access to notification_templates" ON notification_templates;
CREATE POLICY "Service role has full access to notification_templates" ON notification_templates
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update updated_at triggers
CREATE TRIGGER update_notification_channels_updated_at
    BEFORE UPDATE ON notification_channels
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_notification_rules_updated_at
    BEFORE UPDATE ON notification_rules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_notification_templates_updated_at
    BEFORE UPDATE ON notification_templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to queue a notification
CREATE OR REPLACE FUNCTION queue_notification(
    p_channel_id UUID,
    p_rule_id UUID,
    p_event_type TEXT,
    p_event_id TEXT,
    p_payload JSONB
) RETURNS UUID AS $$
DECLARE
    v_log_id UUID;
    v_channel_enabled BOOLEAN;
    v_rule_enabled BOOLEAN;
    v_cooldown_minutes INTEGER;
    v_last_triggered TIMESTAMPTZ;
BEGIN
    -- Check if channel is enabled
    SELECT enabled INTO v_channel_enabled
    FROM notification_channels
    WHERE id = p_channel_id;

    IF NOT v_channel_enabled THEN
        RETURN NULL;
    END IF;

    -- Check rule cooldown if rule_id is provided
    IF p_rule_id IS NOT NULL THEN
        SELECT enabled, cooldown_minutes, last_triggered_at
        INTO v_rule_enabled, v_cooldown_minutes, v_last_triggered
        FROM notification_rules
        WHERE id = p_rule_id;

        IF NOT v_rule_enabled THEN
            RETURN NULL;
        END IF;

        -- Check cooldown
        IF v_cooldown_minutes > 0 AND v_last_triggered IS NOT NULL THEN
            IF v_last_triggered + (v_cooldown_minutes || ' minutes')::INTERVAL > NOW() THEN
                RETURN NULL;  -- Still in cooldown
            END IF;
        END IF;

        -- Update last triggered
        UPDATE notification_rules
        SET last_triggered_at = NOW()
        WHERE id = p_rule_id;
    END IF;

    -- Queue the notification
    INSERT INTO notification_logs (
        channel_id,
        rule_id,
        event_type,
        event_id,
        payload,
        status
    ) VALUES (
        p_channel_id,
        p_rule_id,
        p_event_type,
        p_event_id,
        p_payload,
        'queued'
    )
    RETURNING id INTO v_log_id;

    -- Update channel stats
    UPDATE notification_channels
    SET sent_today = sent_today + 1,
        last_sent_at = NOW()
    WHERE id = p_channel_id;

    RETURN v_log_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get pending notifications for delivery
CREATE OR REPLACE FUNCTION get_pending_notifications(p_limit INTEGER DEFAULT 100)
RETURNS TABLE(
    log_id UUID,
    channel_id UUID,
    channel_type TEXT,
    channel_config JSONB,
    event_type TEXT,
    payload JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        nl.id as log_id,
        nl.channel_id,
        nc.channel_type,
        nc.config as channel_config,
        nl.event_type,
        nl.payload
    FROM notification_logs nl
    JOIN notification_channels nc ON nc.id = nl.channel_id
    WHERE nl.status IN ('queued', 'pending')
    AND nc.enabled = true
    AND (nl.next_retry_at IS NULL OR nl.next_retry_at <= NOW())
    ORDER BY nl.created_at ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to mark notification as sent
CREATE OR REPLACE FUNCTION mark_notification_sent(
    p_log_id UUID,
    p_response_code INTEGER DEFAULT NULL,
    p_response_body TEXT DEFAULT NULL
) RETURNS void AS $$
BEGIN
    UPDATE notification_logs
    SET
        status = 'sent',
        sent_at = NOW(),
        response_code = p_response_code,
        response_body = p_response_body
    WHERE id = p_log_id;
END;
$$ LANGUAGE plpgsql;

-- Function to mark notification as failed and schedule retry
CREATE OR REPLACE FUNCTION mark_notification_failed(
    p_log_id UUID,
    p_error_message TEXT,
    p_response_code INTEGER DEFAULT NULL
) RETURNS void AS $$
DECLARE
    v_retry_count INTEGER;
    v_max_retries INTEGER;
BEGIN
    SELECT retry_count, max_retries
    INTO v_retry_count, v_max_retries
    FROM notification_logs
    WHERE id = p_log_id;

    IF v_retry_count < v_max_retries THEN
        -- Schedule retry with exponential backoff
        UPDATE notification_logs
        SET
            status = 'pending',
            retry_count = retry_count + 1,
            error_message = p_error_message,
            response_code = p_response_code,
            next_retry_at = NOW() + ((2 ^ retry_count) || ' minutes')::INTERVAL
        WHERE id = p_log_id;
    ELSE
        -- Max retries reached, mark as failed
        UPDATE notification_logs
        SET
            status = 'failed',
            error_message = p_error_message,
            response_code = p_response_code
        WHERE id = p_log_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to reset daily counters (call at midnight)
CREATE OR REPLACE FUNCTION reset_notification_daily_counters()
RETURNS void AS $$
BEGIN
    UPDATE notification_channels
    SET sent_today = 0;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup old notification logs
CREATE OR REPLACE FUNCTION cleanup_old_notification_logs(p_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    v_deleted INTEGER;
BEGIN
    DELETE FROM notification_logs
    WHERE created_at < NOW() - (p_days || ' days')::INTERVAL
    AND status IN ('sent', 'delivered', 'failed', 'bounced', 'suppressed');

    GET DIAGNOSTICS v_deleted = ROW_COUNT;
    RETURN v_deleted;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SEED DEFAULT TEMPLATES
-- ============================================================================

-- Insert default notification templates for the default organization
INSERT INTO notification_templates (
    organization_id,
    name,
    description,
    event_types,
    templates,
    is_default
)
SELECT
    id,
    'Test Run Failed',
    'Default template for test run failures',
    ARRAY['test.run.failed'],
    jsonb_build_object(
        'slack', jsonb_build_object(
            'text', ':x: Test run "{{run_name}}" failed',
            'blocks', jsonb_build_array(
                jsonb_build_object(
                    'type', 'section',
                    'text', jsonb_build_object(
                        'type', 'mrkdwn',
                        'text', '*Test Run Failed*\n{{passed_tests}}/{{total_tests}} tests passed'
                    )
                )
            )
        ),
        'email', jsonb_build_object(
            'subject', '[Argus] Test run "{{run_name}}" failed',
            'text', 'Test run failed. {{passed_tests}}/{{total_tests}} tests passed.'
        )
    ),
    true
FROM organizations
WHERE slug = 'default'
ON CONFLICT DO NOTHING;

-- ============================================================================
-- COMPLETION
-- ============================================================================

SELECT 'Notification tables created successfully!' as message;
