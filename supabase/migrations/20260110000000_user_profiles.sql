-- User profiles for persistent user settings
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL,
    display_name TEXT,
    avatar_url TEXT,
    bio TEXT,
    timezone TEXT DEFAULT 'UTC',
    language TEXT DEFAULT 'en',
    theme TEXT DEFAULT 'system' CHECK (theme IN ('light', 'dark', 'system')),
    notification_preferences JSONB DEFAULT jsonb_build_object(
        'email_test_failures', true,
        'email_daily_digest', false,
        'slack_enabled', false,
        'in_app_enabled', true
    ),
    default_organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    default_project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    onboarding_completed BOOLEAN DEFAULT false,
    onboarding_step INTEGER DEFAULT 0,
    last_login_at TIMESTAMPTZ,
    last_active_at TIMESTAMPTZ,
    login_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX idx_user_profiles_email ON user_profiles(email);
CREATE INDEX idx_user_profiles_default_org ON user_profiles(default_organization_id);

-- RLS
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile" ON user_profiles
    FOR SELECT USING (user_id = current_setting('app.user_id', true));

CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE USING (user_id = current_setting('app.user_id', true));

CREATE POLICY "Service role full access to user_profiles" ON user_profiles
    FOR ALL USING (current_setting('role', true) = 'service_role');
