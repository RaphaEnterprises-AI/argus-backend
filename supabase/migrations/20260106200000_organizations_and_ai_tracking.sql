-- Organizations and AI Usage Tracking Migration
-- Adds multi-tenancy support and AI cost tracking

-- ============================================================================
-- ORGANIZATIONS TABLE (Multi-tenancy foundation)
-- ============================================================================

CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,

    -- Subscription plan
    plan TEXT NOT NULL DEFAULT 'free' CHECK (plan IN ('free', 'pro', 'enterprise')),

    -- AI Budget controls
    ai_budget_daily NUMERIC(10,2) DEFAULT 1.00,  -- Daily budget in USD
    ai_budget_monthly NUMERIC(10,2) DEFAULT 25.00,  -- Monthly budget in USD
    ai_spend_today NUMERIC(10,4) DEFAULT 0,
    ai_spend_this_month NUMERIC(10,4) DEFAULT 0,
    ai_budget_reset_at TIMESTAMPTZ DEFAULT date_trunc('day', now()),

    -- Feature flags and settings
    settings JSONB DEFAULT '{}',
    features JSONB DEFAULT jsonb_build_object(
        'max_projects', 3,
        'max_events_per_day', 1000,
        'self_healing', true,
        'advanced_analytics', false,
        'custom_integrations', false,
        'priority_support', false
    ),

    -- Billing info
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Unique slug index for URL routing
CREATE UNIQUE INDEX IF NOT EXISTS idx_organizations_slug ON organizations(slug);
CREATE INDEX IF NOT EXISTS idx_organizations_plan ON organizations(plan);
CREATE INDEX IF NOT EXISTS idx_organizations_stripe ON organizations(stripe_customer_id);

-- ============================================================================
-- ORGANIZATION MEMBERS TABLE (User-Org relationship)
-- ============================================================================

CREATE TABLE IF NOT EXISTS organization_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,  -- Clerk user ID or auth provider ID
    email TEXT NOT NULL,

    -- Role-based access
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member', 'viewer')),

    -- Invitation status
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('pending', 'active', 'suspended')),
    invited_by UUID REFERENCES organization_members(id),
    invited_at TIMESTAMPTZ,
    accepted_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(organization_id, user_id),
    UNIQUE(organization_id, email)
);

CREATE INDEX IF NOT EXISTS idx_org_members_org ON organization_members(organization_id);
CREATE INDEX IF NOT EXISTS idx_org_members_user ON organization_members(user_id);
CREATE INDEX IF NOT EXISTS idx_org_members_email ON organization_members(email);

-- ============================================================================
-- UPDATE PROJECTS TABLE (Add organization relationship)
-- ============================================================================

-- Add organization_id column to projects if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'projects' AND column_name = 'organization_id') THEN
        ALTER TABLE projects ADD COLUMN organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_projects_org ON projects(organization_id);

-- ============================================================================
-- AI USAGE TRACKING TABLE (Per-request cost tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,

    -- Request details
    request_id TEXT NOT NULL,  -- Unique request ID for idempotency
    model TEXT NOT NULL,  -- claude-sonnet-4-5-20250514, claude-haiku-4-5-20250514, etc.
    provider TEXT NOT NULL DEFAULT 'anthropic',  -- anthropic, openai, workers-ai

    -- Token counts
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER GENERATED ALWAYS AS (input_tokens + output_tokens) STORED,

    -- Cost in USD (based on model pricing)
    cost_usd NUMERIC(10,6) NOT NULL DEFAULT 0,

    -- Task context
    task_type TEXT NOT NULL CHECK (task_type IN (
        'error_analysis',
        'test_generation',
        'code_review',
        'self_healing',
        'correlation',
        'risk_assessment',
        'pattern_matching',
        'other'
    )),

    -- Performance metrics
    latency_ms INTEGER,
    cached BOOLEAN DEFAULT false,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Unique constraint on request_id for idempotency
CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_usage_request ON ai_usage(request_id);
CREATE INDEX IF NOT EXISTS idx_ai_usage_org ON ai_usage(organization_id);
CREATE INDEX IF NOT EXISTS idx_ai_usage_project ON ai_usage(project_id);
CREATE INDEX IF NOT EXISTS idx_ai_usage_model ON ai_usage(model);
CREATE INDEX IF NOT EXISTS idx_ai_usage_task ON ai_usage(task_type);
CREATE INDEX IF NOT EXISTS idx_ai_usage_created ON ai_usage(created_at DESC);

-- Index for budget checks (queries filter by date range at runtime)
CREATE INDEX IF NOT EXISTS idx_ai_usage_org_daily ON ai_usage(organization_id, created_at DESC);

-- ============================================================================
-- AI USAGE DAILY SUMMARY TABLE (Pre-aggregated for fast queries)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ai_usage_daily (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    date DATE NOT NULL DEFAULT CURRENT_DATE,

    -- Aggregated metrics
    total_requests INTEGER NOT NULL DEFAULT 0,
    total_input_tokens BIGINT NOT NULL DEFAULT 0,
    total_output_tokens BIGINT NOT NULL DEFAULT 0,
    total_cost_usd NUMERIC(10,4) NOT NULL DEFAULT 0,

    -- Breakdown by model
    usage_by_model JSONB DEFAULT '{}',

    -- Breakdown by task type
    usage_by_task JSONB DEFAULT '{}',

    -- Cache stats
    cache_hits INTEGER DEFAULT 0,
    cache_misses INTEGER DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(organization_id, date)
);

CREATE INDEX IF NOT EXISTS idx_ai_daily_org ON ai_usage_daily(organization_id);
CREATE INDEX IF NOT EXISTS idx_ai_daily_date ON ai_usage_daily(date DESC);

-- ============================================================================
-- API KEYS TABLE (For programmatic access)
-- ============================================================================

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,

    -- Key details (hash stored, not plaintext)
    name TEXT NOT NULL,
    key_hash TEXT NOT NULL,  -- SHA-256 hash of the key
    key_prefix TEXT NOT NULL,  -- First 8 chars for identification (e.g., "argus_sk_")

    -- Permissions
    scopes TEXT[] NOT NULL DEFAULT ARRAY['read', 'write'],

    -- Usage tracking
    last_used_at TIMESTAMPTZ,
    request_count INTEGER DEFAULT 0,

    -- Lifecycle
    expires_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,

    created_by UUID REFERENCES organization_members(id),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_api_keys_org ON api_keys(organization_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);

-- ============================================================================
-- FUNCTIONS: AI Budget Tracking
-- ============================================================================

-- Function to record AI usage and update org spend
CREATE OR REPLACE FUNCTION record_ai_usage(
    p_organization_id UUID,
    p_project_id UUID,
    p_request_id TEXT,
    p_model TEXT,
    p_provider TEXT,
    p_input_tokens INTEGER,
    p_output_tokens INTEGER,
    p_cost_usd NUMERIC,
    p_task_type TEXT,
    p_latency_ms INTEGER DEFAULT NULL,
    p_cached BOOLEAN DEFAULT false,
    p_metadata JSONB DEFAULT '{}'
) RETURNS UUID AS $$
DECLARE
    v_usage_id UUID;
    v_budget_exceeded BOOLEAN := false;
    v_daily_budget NUMERIC;
    v_monthly_budget NUMERIC;
    v_current_daily NUMERIC;
    v_current_monthly NUMERIC;
BEGIN
    -- Check budget limits
    SELECT ai_budget_daily, ai_budget_monthly, ai_spend_today, ai_spend_this_month
    INTO v_daily_budget, v_monthly_budget, v_current_daily, v_current_monthly
    FROM organizations
    WHERE id = p_organization_id;

    -- Check if budget would be exceeded
    IF (v_current_daily + p_cost_usd) > v_daily_budget THEN
        v_budget_exceeded := true;
    END IF;

    IF (v_current_monthly + p_cost_usd) > v_monthly_budget THEN
        v_budget_exceeded := true;
    END IF;

    -- Record the usage regardless (for tracking)
    INSERT INTO ai_usage (
        organization_id, project_id, request_id, model, provider,
        input_tokens, output_tokens, cost_usd, task_type,
        latency_ms, cached, metadata
    ) VALUES (
        p_organization_id, p_project_id, p_request_id, p_model, p_provider,
        p_input_tokens, p_output_tokens, p_cost_usd, p_task_type,
        p_latency_ms, p_cached, p_metadata
    )
    ON CONFLICT (request_id) DO NOTHING
    RETURNING id INTO v_usage_id;

    -- Only update spend if not a duplicate
    IF v_usage_id IS NOT NULL THEN
        -- Update organization spend
        UPDATE organizations
        SET
            ai_spend_today = ai_spend_today + p_cost_usd,
            ai_spend_this_month = ai_spend_this_month + p_cost_usd,
            updated_at = now()
        WHERE id = p_organization_id;

        -- Upsert daily summary
        INSERT INTO ai_usage_daily (
            organization_id, date, total_requests, total_input_tokens,
            total_output_tokens, total_cost_usd,
            usage_by_model, usage_by_task,
            cache_hits, cache_misses
        ) VALUES (
            p_organization_id, CURRENT_DATE, 1, p_input_tokens,
            p_output_tokens, p_cost_usd,
            jsonb_build_object(p_model, jsonb_build_object('requests', 1, 'tokens', p_input_tokens + p_output_tokens, 'cost', p_cost_usd)),
            jsonb_build_object(p_task_type, jsonb_build_object('requests', 1, 'cost', p_cost_usd)),
            CASE WHEN p_cached THEN 1 ELSE 0 END,
            CASE WHEN p_cached THEN 0 ELSE 1 END
        )
        ON CONFLICT (organization_id, date) DO UPDATE SET
            total_requests = ai_usage_daily.total_requests + 1,
            total_input_tokens = ai_usage_daily.total_input_tokens + p_input_tokens,
            total_output_tokens = ai_usage_daily.total_output_tokens + p_output_tokens,
            total_cost_usd = ai_usage_daily.total_cost_usd + p_cost_usd,
            usage_by_model = ai_usage_daily.usage_by_model ||
                jsonb_build_object(p_model, jsonb_build_object(
                    'requests', COALESCE((ai_usage_daily.usage_by_model->p_model->>'requests')::INTEGER, 0) + 1,
                    'tokens', COALESCE((ai_usage_daily.usage_by_model->p_model->>'tokens')::INTEGER, 0) + p_input_tokens + p_output_tokens,
                    'cost', COALESCE((ai_usage_daily.usage_by_model->p_model->>'cost')::NUMERIC, 0) + p_cost_usd
                )),
            usage_by_task = ai_usage_daily.usage_by_task ||
                jsonb_build_object(p_task_type, jsonb_build_object(
                    'requests', COALESCE((ai_usage_daily.usage_by_task->p_task_type->>'requests')::INTEGER, 0) + 1,
                    'cost', COALESCE((ai_usage_daily.usage_by_task->p_task_type->>'cost')::NUMERIC, 0) + p_cost_usd
                )),
            cache_hits = ai_usage_daily.cache_hits + CASE WHEN p_cached THEN 1 ELSE 0 END,
            cache_misses = ai_usage_daily.cache_misses + CASE WHEN p_cached THEN 0 ELSE 1 END,
            updated_at = now();
    END IF;

    RETURN v_usage_id;
END;
$$ LANGUAGE plpgsql;

-- Function to reset daily spend (call via cron job at midnight)
CREATE OR REPLACE FUNCTION reset_daily_ai_spend() RETURNS void AS $$
BEGIN
    UPDATE organizations
    SET
        ai_spend_today = 0,
        ai_budget_reset_at = now()
    WHERE ai_budget_reset_at < date_trunc('day', now());
END;
$$ LANGUAGE plpgsql;

-- Function to reset monthly spend (call via cron job on 1st of month)
CREATE OR REPLACE FUNCTION reset_monthly_ai_spend() RETURNS void AS $$
BEGIN
    UPDATE organizations
    SET ai_spend_this_month = 0
    WHERE date_trunc('month', ai_budget_reset_at) < date_trunc('month', now());
END;
$$ LANGUAGE plpgsql;

-- Function to check if org has remaining budget
CREATE OR REPLACE FUNCTION check_ai_budget(p_organization_id UUID)
RETURNS TABLE(
    has_daily_budget BOOLEAN,
    has_monthly_budget BOOLEAN,
    daily_remaining NUMERIC,
    monthly_remaining NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        (ai_spend_today < ai_budget_daily) as has_daily_budget,
        (ai_spend_this_month < ai_budget_monthly) as has_monthly_budget,
        GREATEST(0, ai_budget_daily - ai_spend_today) as daily_remaining,
        GREATEST(0, ai_budget_monthly - ai_spend_this_month) as monthly_remaining
    FROM organizations
    WHERE id = p_organization_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- ROW LEVEL SECURITY POLICIES
-- ============================================================================

ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE organization_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_usage_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- Policies for organizations (access via membership)
DROP POLICY IF EXISTS "Users can view their organizations" ON organizations;
CREATE POLICY "Users can view their organizations" ON organizations
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_members.organization_id = organizations.id
            AND organization_members.user_id = current_setting('app.user_id', true)
        )
    );

DROP POLICY IF EXISTS "Admins can update their organizations" ON organizations;
CREATE POLICY "Admins can update their organizations" ON organizations
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_members.organization_id = organizations.id
            AND organization_members.user_id = current_setting('app.user_id', true)
            AND organization_members.role IN ('owner', 'admin')
        )
    );

-- Policies for organization_members
DROP POLICY IF EXISTS "Users can view members of their organizations" ON organization_members;
CREATE POLICY "Users can view members of their organizations" ON organization_members
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM organization_members om
            WHERE om.organization_id = organization_members.organization_id
            AND om.user_id = current_setting('app.user_id', true)
        )
    );

-- Policies for ai_usage (org members can view)
DROP POLICY IF EXISTS "Users can view AI usage for their organizations" ON ai_usage;
CREATE POLICY "Users can view AI usage for their organizations" ON ai_usage
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_members.organization_id = ai_usage.organization_id
            AND organization_members.user_id = current_setting('app.user_id', true)
        )
    );

-- Policies for ai_usage_daily
DROP POLICY IF EXISTS "Users can view daily AI usage for their organizations" ON ai_usage_daily;
CREATE POLICY "Users can view daily AI usage for their organizations" ON ai_usage_daily
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_members.organization_id = ai_usage_daily.organization_id
            AND organization_members.user_id = current_setting('app.user_id', true)
        )
    );

-- Policies for api_keys
DROP POLICY IF EXISTS "Users can manage API keys for their organizations" ON api_keys;
CREATE POLICY "Users can manage API keys for their organizations" ON api_keys
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_members.organization_id = api_keys.organization_id
            AND organization_members.user_id = current_setting('app.user_id', true)
            AND organization_members.role IN ('owner', 'admin')
        )
    );

-- ============================================================================
-- SERVICE ROLE POLICIES (for backend services)
-- ============================================================================

-- Allow service role full access (for backend operations)
DROP POLICY IF EXISTS "Service role has full access to organizations" ON organizations;
CREATE POLICY "Service role has full access to organizations" ON organizations
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role has full access to organization_members" ON organization_members;
CREATE POLICY "Service role has full access to organization_members" ON organization_members
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role has full access to ai_usage" ON ai_usage;
CREATE POLICY "Service role has full access to ai_usage" ON ai_usage
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role has full access to ai_usage_daily" ON ai_usage_daily;
CREATE POLICY "Service role has full access to ai_usage_daily" ON ai_usage_daily
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role has full access to api_keys" ON api_keys;
CREATE POLICY "Service role has full access to api_keys" ON api_keys
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- SEED DEFAULT ORGANIZATION (Optional - for development)
-- ============================================================================

-- Insert a default organization if none exists (for development)
INSERT INTO organizations (name, slug, plan, features)
SELECT 'Default Organization', 'default', 'pro',
    jsonb_build_object(
        'max_projects', 10,
        'max_events_per_day', 10000,
        'self_healing', true,
        'advanced_analytics', true,
        'custom_integrations', true,
        'priority_support', false
    )
WHERE NOT EXISTS (SELECT 1 FROM organizations LIMIT 1);
