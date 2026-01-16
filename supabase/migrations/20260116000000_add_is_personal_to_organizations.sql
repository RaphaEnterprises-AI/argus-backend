-- Migration: Add is_personal flag to organizations
-- Purpose: Distinguish personal/individual workspaces from team organizations
--
-- Personal organizations are auto-created for users who sign up without
-- joining an existing org, providing a seamless individual user experience.

-- Add is_personal column to organizations table
ALTER TABLE organizations
ADD COLUMN IF NOT EXISTS is_personal BOOLEAN DEFAULT FALSE;

-- Add comment for documentation
COMMENT ON COLUMN organizations.is_personal IS
  'True for auto-created personal workspaces, false for team organizations';

-- Create index for filtering personal vs team orgs (useful for admin queries)
CREATE INDEX IF NOT EXISTS idx_organizations_is_personal
ON organizations(is_personal) WHERE is_personal = TRUE;

-- ============================================================================
-- DATA MIGRATION: Based on actual database analysis (Jan 2026)
-- ============================================================================

-- Current data breakdown:
-- 1. "Samuel Kumar" (samuel-kumar) - Personal workspace, single owner with email
-- 2. "My Organization" (clerk-*) - Auto-created from Clerk, single owner
-- 3. "Test Organization" variants - Test data with single owner
-- 4. "Default Organization" - System org (admin role, not owner)
-- 5. Orphan test orgs (0 members) - Should be cleaned up

-- Step 1: Mark "Samuel Kumar" as personal (person's name, clearly individual)
UPDATE organizations
SET is_personal = TRUE
WHERE slug = 'samuel-kumar';

-- Step 2: Mark orgs auto-created from Clerk (clerk-* slug) as personal
-- These are created when users sign up through Clerk without an existing org
UPDATE organizations
SET is_personal = TRUE
WHERE slug LIKE 'clerk-%';

-- Step 3: Clean up orphan test organizations (0 members, test-org-* pattern)
-- These appear to be leftover from automated testing
DELETE FROM organizations
WHERE slug LIKE 'test-org-%'
AND NOT EXISTS (
  SELECT 1 FROM organization_members om
  WHERE om.organization_id = organizations.id
);

-- Step 4: Future personal orgs will use "personal-*" slug pattern
-- No action needed - just documenting the convention

-- Log results
DO $$
DECLARE
  personal_count INTEGER;
  deleted_count INTEGER;
BEGIN
  SELECT COUNT(*) INTO personal_count FROM organizations WHERE is_personal = TRUE;
  RAISE NOTICE 'Organizations marked as personal: %', personal_count;
END $$;
