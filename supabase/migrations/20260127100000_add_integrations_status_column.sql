-- =============================================================================
-- Add missing status column to integrations table
-- =============================================================================
-- This fixes the error: column integrations.status does not exist

-- Add status column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'integrations' AND column_name = 'status'
    ) THEN
        ALTER TABLE integrations ADD COLUMN status TEXT DEFAULT 'pending';

        -- Add check constraint for valid status values
        ALTER TABLE integrations ADD CONSTRAINT integrations_status_check
            CHECK (status IN ('pending', 'connected', 'disconnected', 'error', 'expired'));

        -- Create index for status queries
        CREATE INDEX IF NOT EXISTS idx_integrations_status ON integrations(status);

        -- Add comment
        COMMENT ON COLUMN integrations.status IS 'Integration connection status: pending, connected, disconnected, error, expired';
    END IF;
END $$;

-- Update existing rows to have 'connected' status if they have credentials
UPDATE integrations
SET status = 'connected'
WHERE status = 'pending'
  AND credentials IS NOT NULL
  AND credentials != '{}';
