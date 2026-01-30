-- =============================================================================
-- Add missing type and settings columns to integrations table
-- =============================================================================
-- Fixes errors:
-- - column integrations.type does not exist
-- - Could not find the 'settings' column of 'integrations'

-- Add type column if it doesn't exist (stores platform identifier like 'sentry', 'github')
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'integrations' AND column_name = 'type'
    ) THEN
        ALTER TABLE integrations ADD COLUMN type TEXT;

        -- Create index for type queries
        CREATE INDEX IF NOT EXISTS idx_integrations_type ON integrations(type);

        -- Add comment
        COMMENT ON COLUMN integrations.type IS 'Integration platform type: sentry, github, slack, etc.';
    END IF;
END $$;

-- Add settings column if it doesn't exist (stores platform-specific settings as JSON)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'integrations' AND column_name = 'settings'
    ) THEN
        ALTER TABLE integrations ADD COLUMN settings JSONB DEFAULT '{}';

        -- Add comment
        COMMENT ON COLUMN integrations.settings IS 'Platform-specific settings (notification preferences, etc.)';
    END IF;
END $$;

-- Add sync_status column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'integrations' AND column_name = 'sync_status'
    ) THEN
        ALTER TABLE integrations ADD COLUMN sync_status TEXT DEFAULT 'idle';

        -- Add check constraint for valid sync_status values
        ALTER TABLE integrations ADD CONSTRAINT integrations_sync_status_check
            CHECK (sync_status IN ('idle', 'syncing', 'completed', 'failed'));

        -- Add comment
        COMMENT ON COLUMN integrations.sync_status IS 'Current sync status: idle, syncing, completed, failed';
    END IF;
END $$;

-- Add data_points_synced column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'integrations' AND column_name = 'data_points_synced'
    ) THEN
        ALTER TABLE integrations ADD COLUMN data_points_synced INTEGER DEFAULT 0;

        -- Add comment
        COMMENT ON COLUMN integrations.data_points_synced IS 'Total number of data points synced from this integration';
    END IF;
END $$;

-- Add features_enabled column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'integrations' AND column_name = 'features_enabled'
    ) THEN
        ALTER TABLE integrations ADD COLUMN features_enabled JSONB DEFAULT '[]';

        -- Add comment
        COMMENT ON COLUMN integrations.features_enabled IS 'List of enabled features for this integration';
    END IF;
END $$;

-- Add updated_at column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'integrations' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE integrations ADD COLUMN updated_at TIMESTAMPTZ;
    END IF;
END $$;

-- Add created_at column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'integrations' AND column_name = 'created_at'
    ) THEN
        ALTER TABLE integrations ADD COLUMN created_at TIMESTAMPTZ DEFAULT now();
    END IF;
END $$;
