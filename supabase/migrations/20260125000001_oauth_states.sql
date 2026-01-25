-- OAuth State Management Table
-- Stores temporary OAuth state for secure authorization flows

CREATE TABLE IF NOT EXISTS oauth_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    platform TEXT NOT NULL CHECK (platform IN ('github', 'slack', 'jira', 'linear')),
    state TEXT NOT NULL UNIQUE,
    code_verifier TEXT,  -- For PKCE flow
    redirect_uri TEXT,
    metadata JSONB DEFAULT '{}',  -- Store additional data like scopes requested
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '10 minutes'
);

-- Index for fast state lookups during callback
CREATE INDEX IF NOT EXISTS idx_oauth_states_state ON oauth_states(state);

-- Index for cleaning up expired states
CREATE INDEX IF NOT EXISTS idx_oauth_states_expires ON oauth_states(expires_at);

-- Index for user lookups
CREATE INDEX IF NOT EXISTS idx_oauth_states_user ON oauth_states(user_id);

-- =============================================================================
-- Cleanup Function for Expired States
-- =============================================================================

CREATE OR REPLACE FUNCTION cleanup_expired_oauth_states()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM oauth_states
    WHERE expires_at < NOW();

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Row Level Security
-- =============================================================================

ALTER TABLE oauth_states ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS oauth_states_select ON oauth_states;
DROP POLICY IF EXISTS oauth_states_insert ON oauth_states;
DROP POLICY IF EXISTS oauth_states_delete ON oauth_states;
DROP POLICY IF EXISTS "Service role full access oauth_states" ON oauth_states;

-- Users can only see their own OAuth states
CREATE POLICY oauth_states_select ON oauth_states
    FOR SELECT
    USING (user_id = current_setting('app.user_id', true));

-- Users can only create their own OAuth states
CREATE POLICY oauth_states_insert ON oauth_states
    FOR INSERT
    WITH CHECK (user_id = current_setting('app.user_id', true));

-- Users can only delete their own OAuth states
CREATE POLICY oauth_states_delete ON oauth_states
    FOR DELETE
    USING (user_id = current_setting('app.user_id', true));

-- Service role has full access (for backend operations)
CREATE POLICY "Service role full access oauth_states" ON oauth_states
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE oauth_states IS 'Temporary storage for OAuth authorization state parameters';
COMMENT ON COLUMN oauth_states.state IS 'Cryptographically secure random state parameter';
COMMENT ON COLUMN oauth_states.code_verifier IS 'PKCE code verifier (stored until token exchange)';
COMMENT ON COLUMN oauth_states.redirect_uri IS 'Original redirect URI for validation';
COMMENT ON COLUMN oauth_states.expires_at IS 'State expires after 10 minutes for security';

-- =============================================================================
-- Add OAuth token fields to integrations table
-- =============================================================================

-- Add encrypted token storage columns if they don't exist
DO $$
BEGIN
    -- access_token: Encrypted OAuth access token
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'integrations' AND column_name = 'access_token_encrypted') THEN
        ALTER TABLE integrations ADD COLUMN access_token_encrypted TEXT;
    END IF;

    -- refresh_token: Encrypted OAuth refresh token
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'integrations' AND column_name = 'refresh_token_encrypted') THEN
        ALTER TABLE integrations ADD COLUMN refresh_token_encrypted TEXT;
    END IF;

    -- token_expires_at: When the access token expires
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'integrations' AND column_name = 'token_expires_at') THEN
        ALTER TABLE integrations ADD COLUMN token_expires_at TIMESTAMPTZ;
    END IF;

    -- oauth_scopes: Scopes granted by the OAuth provider
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'integrations' AND column_name = 'oauth_scopes') THEN
        ALTER TABLE integrations ADD COLUMN oauth_scopes TEXT[];
    END IF;

    -- external_account_id: ID from the OAuth provider (e.g., GitHub user ID)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'integrations' AND column_name = 'external_account_id') THEN
        ALTER TABLE integrations ADD COLUMN external_account_id TEXT;
    END IF;

    -- external_account_name: Name/username from the OAuth provider
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'integrations' AND column_name = 'external_account_name') THEN
        ALTER TABLE integrations ADD COLUMN external_account_name TEXT;
    END IF;
END $$;

-- Index for finding integrations by platform
CREATE INDEX IF NOT EXISTS idx_integrations_platform_user ON integrations(user_id, platform);

-- Comments for new columns
COMMENT ON COLUMN integrations.access_token_encrypted IS 'AES-256-GCM encrypted OAuth access token';
COMMENT ON COLUMN integrations.refresh_token_encrypted IS 'AES-256-GCM encrypted OAuth refresh token';
COMMENT ON COLUMN integrations.token_expires_at IS 'When the access token expires (for auto-refresh)';
COMMENT ON COLUMN integrations.oauth_scopes IS 'Array of OAuth scopes granted by the provider';
COMMENT ON COLUMN integrations.external_account_id IS 'User/account ID from the OAuth provider';
COMMENT ON COLUMN integrations.external_account_name IS 'Display name from the OAuth provider';
