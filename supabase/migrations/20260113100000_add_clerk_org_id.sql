-- Add clerk_org_id column to organizations table
-- This enables Clerk-to-Supabase organization synchronization

-- Add clerk_org_id column for storing Clerk organization references
ALTER TABLE organizations
ADD COLUMN IF NOT EXISTS clerk_org_id TEXT;

-- Create index for fast lookups by Clerk org ID
CREATE INDEX IF NOT EXISTS idx_organizations_clerk_org_id ON organizations(clerk_org_id);

-- Add unique constraint on clerk_org_id (only one Supabase org per Clerk org)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'organizations_clerk_org_id_key'
    ) THEN
        ALTER TABLE organizations
        ADD CONSTRAINT organizations_clerk_org_id_key UNIQUE (clerk_org_id);
    END IF;
EXCEPTION WHEN others THEN
    -- Ignore if constraint already exists
    NULL;
END $$;

-- Comment explaining the column
COMMENT ON COLUMN organizations.clerk_org_id IS
    'Clerk organization ID for syncing Clerk orgs to Supabase. Format: org_xxx';
