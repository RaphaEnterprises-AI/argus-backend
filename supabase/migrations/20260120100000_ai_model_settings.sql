-- AI Model Settings Migration
-- Adds BYOK (Bring Your Own Key) support, user AI preferences, and per-user usage tracking

-- ============================================================================
-- USER PROVIDER KEYS TABLE (BYOK - Encrypted API Keys)
-- ============================================================================

CREATE TABLE IF NOT EXISTS user_provider_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL CHECK (provider IN ('anthropic', 'openai', 'google', 'groq', 'together')),

    -- Encrypted key storage
    encrypted_key TEXT NOT NULL,  -- AES-256-GCM encrypted
    key_prefix TEXT NOT NULL,  -- First 8 chars for display (e.g., "sk-ant-...")
    key_suffix TEXT,  -- Last 4 chars for identification

    -- Validation state
    is_valid BOOLEAN DEFAULT TRUE,
    last_validated_at TIMESTAMPTZ,
    validation_error TEXT,  -- Store last validation error if any

    -- Metadata
    display_name TEXT,  -- Optional friendly name
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- One key per provider per user
    UNIQUE(user_id, provider)
);

-- Foreign key added after table creation to handle potential missing user_profiles
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'user_provider_keys_user_id_fkey'
    ) THEN
        ALTER TABLE user_provider_keys
        ADD CONSTRAINT user_provider_keys_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE;
    END IF;
EXCEPTION WHEN others THEN
    -- Ignore if user_profiles doesn't exist yet
    RAISE NOTICE 'Could not add FK constraint to user_profiles: %', SQLERRM;
END $$;

CREATE INDEX IF NOT EXISTS idx_user_provider_keys_user ON user_provider_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_user_provider_keys_provider ON user_provider_keys(provider);

-- ============================================================================
-- ADD AI PREFERENCES TO USER PROFILES
-- ============================================================================

-- Add ai_preferences JSONB column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'user_profiles' AND column_name = 'ai_preferences'
    ) THEN
        ALTER TABLE user_profiles ADD COLUMN ai_preferences JSONB DEFAULT '{
            "default_provider": "anthropic",
            "default_model": "claude-sonnet-4-5",
            "cost_limit_per_day": 10.0,
            "cost_limit_per_message": 1.0,
            "use_platform_key_fallback": true,
            "show_token_costs": true,
            "show_model_in_chat": true,
            "preferred_models_by_task": {}
        }'::jsonb;
    END IF;
END $$;

-- ============================================================================
-- USER AI USAGE TABLE (Per-User Token Tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS user_ai_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    organization_id UUID,  -- Optional, for org billing attribution

    -- Provider and model info
    provider TEXT NOT NULL,
    model TEXT NOT NULL,

    -- Token counts
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,

    -- Cost calculation
    cost_usd DECIMAL(10, 6) NOT NULL DEFAULT 0,

    -- Key source tracking
    key_source TEXT NOT NULL DEFAULT 'platform' CHECK (key_source IN ('platform', 'byok')),

    -- Context (for attribution)
    thread_id TEXT,
    message_id TEXT,
    request_id TEXT UNIQUE,  -- For idempotency

    -- Task context
    task_type TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Foreign keys
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'user_ai_usage_user_id_fkey'
    ) THEN
        ALTER TABLE user_ai_usage
        ADD CONSTRAINT user_ai_usage_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE;
    END IF;
EXCEPTION WHEN others THEN
    RAISE NOTICE 'Could not add FK constraint to user_profiles: %', SQLERRM;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'user_ai_usage_organization_id_fkey'
    ) THEN
        ALTER TABLE user_ai_usage
        ADD CONSTRAINT user_ai_usage_organization_id_fkey
        FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL;
    END IF;
EXCEPTION WHEN others THEN
    RAISE NOTICE 'Could not add FK constraint to organizations: %', SQLERRM;
END $$;

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_user_ai_usage_user_date ON user_ai_usage(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_ai_usage_org_date ON user_ai_usage(organization_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_ai_usage_thread ON user_ai_usage(thread_id);
CREATE INDEX IF NOT EXISTS idx_user_ai_usage_model ON user_ai_usage(model, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_ai_usage_provider ON user_ai_usage(provider);

-- ============================================================================
-- USER AI USAGE DAILY SUMMARY (Pre-aggregated for dashboards)
-- ============================================================================

CREATE TABLE IF NOT EXISTS user_ai_usage_daily (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    date DATE NOT NULL DEFAULT CURRENT_DATE,

    -- Aggregated metrics
    total_requests INTEGER NOT NULL DEFAULT 0,
    total_input_tokens BIGINT NOT NULL DEFAULT 0,
    total_output_tokens BIGINT NOT NULL DEFAULT 0,
    total_cost_usd DECIMAL(10, 4) NOT NULL DEFAULT 0,

    -- Breakdown by model (for pie charts)
    usage_by_model JSONB DEFAULT '{}',

    -- Breakdown by provider
    usage_by_provider JSONB DEFAULT '{}',

    -- Key source breakdown
    platform_key_cost DECIMAL(10, 4) DEFAULT 0,
    byok_cost DECIMAL(10, 4) DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(user_id, date)
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'user_ai_usage_daily_user_id_fkey'
    ) THEN
        ALTER TABLE user_ai_usage_daily
        ADD CONSTRAINT user_ai_usage_daily_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE;
    END IF;
EXCEPTION WHEN others THEN
    RAISE NOTICE 'Could not add FK constraint to user_profiles: %', SQLERRM;
END $$;

CREATE INDEX IF NOT EXISTS idx_user_ai_usage_daily_user ON user_ai_usage_daily(user_id);
CREATE INDEX IF NOT EXISTS idx_user_ai_usage_daily_date ON user_ai_usage_daily(date DESC);

-- ============================================================================
-- FUNCTIONS: Record User AI Usage
-- ============================================================================

CREATE OR REPLACE FUNCTION record_user_ai_usage(
    p_user_id TEXT,
    p_organization_id UUID,
    p_provider TEXT,
    p_model TEXT,
    p_input_tokens INTEGER,
    p_output_tokens INTEGER,
    p_cost_usd DECIMAL,
    p_key_source TEXT DEFAULT 'platform',
    p_thread_id TEXT DEFAULT NULL,
    p_message_id TEXT DEFAULT NULL,
    p_request_id TEXT DEFAULT NULL,
    p_task_type TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_usage_id UUID;
BEGIN
    -- Insert usage record
    INSERT INTO user_ai_usage (
        user_id, organization_id, provider, model,
        input_tokens, output_tokens, cost_usd,
        key_source, thread_id, message_id, request_id, task_type
    ) VALUES (
        p_user_id, p_organization_id, p_provider, p_model,
        p_input_tokens, p_output_tokens, p_cost_usd,
        p_key_source, p_thread_id, p_message_id, p_request_id, p_task_type
    )
    ON CONFLICT (request_id) DO NOTHING
    RETURNING id INTO v_usage_id;

    -- If inserted (not duplicate), update daily summary
    IF v_usage_id IS NOT NULL THEN
        INSERT INTO user_ai_usage_daily (
            user_id, date, total_requests, total_input_tokens,
            total_output_tokens, total_cost_usd,
            usage_by_model, usage_by_provider,
            platform_key_cost, byok_cost
        ) VALUES (
            p_user_id, CURRENT_DATE, 1, p_input_tokens,
            p_output_tokens, p_cost_usd,
            jsonb_build_object(p_model, jsonb_build_object(
                'requests', 1,
                'tokens', p_input_tokens + p_output_tokens,
                'cost', p_cost_usd
            )),
            jsonb_build_object(p_provider, jsonb_build_object(
                'requests', 1,
                'cost', p_cost_usd
            )),
            CASE WHEN p_key_source = 'platform' THEN p_cost_usd ELSE 0 END,
            CASE WHEN p_key_source = 'byok' THEN p_cost_usd ELSE 0 END
        )
        ON CONFLICT (user_id, date) DO UPDATE SET
            total_requests = user_ai_usage_daily.total_requests + 1,
            total_input_tokens = user_ai_usage_daily.total_input_tokens + p_input_tokens,
            total_output_tokens = user_ai_usage_daily.total_output_tokens + p_output_tokens,
            total_cost_usd = user_ai_usage_daily.total_cost_usd + p_cost_usd,
            usage_by_model = user_ai_usage_daily.usage_by_model || jsonb_build_object(
                p_model, jsonb_build_object(
                    'requests', COALESCE((user_ai_usage_daily.usage_by_model->p_model->>'requests')::INTEGER, 0) + 1,
                    'tokens', COALESCE((user_ai_usage_daily.usage_by_model->p_model->>'tokens')::BIGINT, 0) + p_input_tokens + p_output_tokens,
                    'cost', COALESCE((user_ai_usage_daily.usage_by_model->p_model->>'cost')::DECIMAL, 0) + p_cost_usd
                )
            ),
            usage_by_provider = user_ai_usage_daily.usage_by_provider || jsonb_build_object(
                p_provider, jsonb_build_object(
                    'requests', COALESCE((user_ai_usage_daily.usage_by_provider->p_provider->>'requests')::INTEGER, 0) + 1,
                    'cost', COALESCE((user_ai_usage_daily.usage_by_provider->p_provider->>'cost')::DECIMAL, 0) + p_cost_usd
                )
            ),
            platform_key_cost = user_ai_usage_daily.platform_key_cost +
                CASE WHEN p_key_source = 'platform' THEN p_cost_usd ELSE 0 END,
            byok_cost = user_ai_usage_daily.byok_cost +
                CASE WHEN p_key_source = 'byok' THEN p_cost_usd ELSE 0 END,
            updated_at = NOW();
    END IF;

    RETURN v_usage_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- FUNCTION: Get User Daily Spend
-- ============================================================================

CREATE OR REPLACE FUNCTION get_user_daily_spend(p_user_id TEXT)
RETURNS DECIMAL AS $$
DECLARE
    v_spend DECIMAL;
BEGIN
    SELECT COALESCE(total_cost_usd, 0)
    INTO v_spend
    FROM user_ai_usage_daily
    WHERE user_id = p_user_id AND date = CURRENT_DATE;

    RETURN COALESCE(v_spend, 0);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- FUNCTION: Check User AI Budget
-- ============================================================================

CREATE OR REPLACE FUNCTION check_user_ai_budget(p_user_id TEXT)
RETURNS TABLE(
    has_budget BOOLEAN,
    daily_limit DECIMAL,
    daily_spent DECIMAL,
    daily_remaining DECIMAL,
    message_limit DECIMAL
) AS $$
DECLARE
    v_preferences JSONB;
BEGIN
    -- Get user preferences
    SELECT ai_preferences INTO v_preferences
    FROM user_profiles
    WHERE user_id = p_user_id;

    RETURN QUERY
    SELECT
        (COALESCE(d.total_cost_usd, 0) < COALESCE((v_preferences->>'cost_limit_per_day')::DECIMAL, 10.0)) as has_budget,
        COALESCE((v_preferences->>'cost_limit_per_day')::DECIMAL, 10.0) as daily_limit,
        COALESCE(d.total_cost_usd, 0) as daily_spent,
        GREATEST(0, COALESCE((v_preferences->>'cost_limit_per_day')::DECIMAL, 10.0) - COALESCE(d.total_cost_usd, 0)) as daily_remaining,
        COALESCE((v_preferences->>'cost_limit_per_message')::DECIMAL, 1.0) as message_limit
    FROM user_profiles up
    LEFT JOIN user_ai_usage_daily d ON d.user_id = up.user_id AND d.date = CURRENT_DATE
    WHERE up.user_id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE user_provider_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_ai_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_ai_usage_daily ENABLE ROW LEVEL SECURITY;

-- User can only see/manage their own provider keys
DROP POLICY IF EXISTS "Users can view own provider keys" ON user_provider_keys;
CREATE POLICY "Users can view own provider keys" ON user_provider_keys
    FOR SELECT USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS "Users can insert own provider keys" ON user_provider_keys;
CREATE POLICY "Users can insert own provider keys" ON user_provider_keys
    FOR INSERT WITH CHECK (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS "Users can update own provider keys" ON user_provider_keys;
CREATE POLICY "Users can update own provider keys" ON user_provider_keys
    FOR UPDATE USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS "Users can delete own provider keys" ON user_provider_keys;
CREATE POLICY "Users can delete own provider keys" ON user_provider_keys
    FOR DELETE USING (user_id = current_setting('app.user_id', true));

-- User can only view their own usage
DROP POLICY IF EXISTS "Users can view own AI usage" ON user_ai_usage;
CREATE POLICY "Users can view own AI usage" ON user_ai_usage
    FOR SELECT USING (user_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS "Users can view own AI usage daily" ON user_ai_usage_daily;
CREATE POLICY "Users can view own AI usage daily" ON user_ai_usage_daily
    FOR SELECT USING (user_id = current_setting('app.user_id', true));

-- Service role has full access
DROP POLICY IF EXISTS "Service role full access to user_provider_keys" ON user_provider_keys;
CREATE POLICY "Service role full access to user_provider_keys" ON user_provider_keys
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role full access to user_ai_usage" ON user_ai_usage;
CREATE POLICY "Service role full access to user_ai_usage" ON user_ai_usage
    FOR ALL USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role full access to user_ai_usage_daily" ON user_ai_usage_daily;
CREATE POLICY "Service role full access to user_ai_usage_daily" ON user_ai_usage_daily
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON user_provider_keys TO authenticated;
GRANT SELECT ON user_ai_usage TO authenticated;
GRANT SELECT ON user_ai_usage_daily TO authenticated;
GRANT EXECUTE ON FUNCTION record_user_ai_usage TO authenticated;
GRANT EXECUTE ON FUNCTION get_user_daily_spend TO authenticated;
GRANT EXECUTE ON FUNCTION check_user_ai_budget TO authenticated;
