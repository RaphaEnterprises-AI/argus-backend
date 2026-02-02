-- =============================================================================
-- Add metadata column to integrations table
-- =============================================================================
-- Fixes error: Could not find the 'metadata' column of 'integrations' in the schema cache
-- The sync daemon stores sync-related metadata in this column

-- Add metadata column if it doesn't exist (stores sync metadata as JSON)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'integrations' AND column_name = 'metadata'
    ) THEN
        ALTER TABLE integrations ADD COLUMN metadata JSONB DEFAULT '{}';

        -- Add comment
        COMMENT ON COLUMN integrations.metadata IS 'Sync and operational metadata (last_items_synced, etc.)';
    END IF;
END $$;

-- Also add connected_by column for tracking who connected the integration
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'integrations' AND column_name = 'connected_by'
    ) THEN
        ALTER TABLE integrations ADD COLUMN connected_by TEXT;

        -- Add comment
        COMMENT ON COLUMN integrations.connected_by IS 'User ID or name of person who connected this integration';
    END IF;
END $$;

-- Add connected_at column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'integrations' AND column_name = 'connected_at'
    ) THEN
        ALTER TABLE integrations ADD COLUMN connected_at TIMESTAMPTZ;

        -- Add comment
        COMMENT ON COLUMN integrations.connected_at IS 'Timestamp when the integration was connected';
    END IF;
END $$;
