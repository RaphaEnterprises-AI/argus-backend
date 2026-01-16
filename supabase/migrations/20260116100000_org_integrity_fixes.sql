-- Migration: Organization Integrity Fixes
-- Purpose: Fix data integrity issues, add constraints, and prevent race conditions
--
-- Issues addressed:
-- 1. Users can have multiple personal organizations (race condition)
-- 2. Security tables block organization deletion (wrong cascade rules)
-- 3. Orphan projects and schedules exist
-- 4. No constraint preventing duplicate personal orgs

BEGIN;

-- ============================================================================
-- STEP 1: Fix Cascade Rules for Security Tables
-- ============================================================================
-- These tables currently use NO ACTION which blocks org deletion

-- Fix data_access_logs
ALTER TABLE data_access_logs
DROP CONSTRAINT IF EXISTS data_access_logs_organization_id_fkey;

ALTER TABLE data_access_logs
ADD CONSTRAINT data_access_logs_organization_id_fkey
FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE;

-- Fix permission_changes
ALTER TABLE permission_changes
DROP CONSTRAINT IF EXISTS permission_changes_organization_id_fkey;

ALTER TABLE permission_changes
ADD CONSTRAINT permission_changes_organization_id_fkey
FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE;

-- Fix security_alerts
ALTER TABLE security_alerts
DROP CONSTRAINT IF EXISTS security_alerts_organization_id_fkey;

ALTER TABLE security_alerts
ADD CONSTRAINT security_alerts_organization_id_fkey
FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE;

-- Fix security_audit_logs
ALTER TABLE security_audit_logs
DROP CONSTRAINT IF EXISTS security_audit_logs_organization_id_fkey;

ALTER TABLE security_audit_logs
ADD CONSTRAINT security_audit_logs_organization_id_fkey
FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE;

-- ============================================================================
-- STEP 2: Add Constraint to Prevent Multiple Personal Orgs Per User
-- ============================================================================
-- Use a function + trigger since partial unique indexes don't work across tables

-- Create function to check personal org uniqueness
CREATE OR REPLACE FUNCTION check_single_personal_org_per_user()
RETURNS TRIGGER AS $$
DECLARE
    existing_personal_count INTEGER;
BEGIN
    -- Only check if this is for a personal org and user_id is set
    IF NEW.user_id IS NOT NULL THEN
        SELECT COUNT(*) INTO existing_personal_count
        FROM organization_members om
        JOIN organizations o ON o.id = om.organization_id
        WHERE om.user_id = NEW.user_id
        AND o.is_personal = TRUE
        AND o.id != NEW.organization_id;  -- Exclude current org (for updates)

        -- Check if new org is personal
        IF EXISTS (SELECT 1 FROM organizations WHERE id = NEW.organization_id AND is_personal = TRUE) THEN
            IF existing_personal_count > 0 THEN
                RAISE EXCEPTION 'User % already has a personal organization', NEW.user_id;
            END IF;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger (drop first if exists)
DROP TRIGGER IF EXISTS enforce_single_personal_org ON organization_members;

CREATE TRIGGER enforce_single_personal_org
BEFORE INSERT ON organization_members
FOR EACH ROW
EXECUTE FUNCTION check_single_personal_org_per_user();

-- ============================================================================
-- STEP 3: Clean Up Duplicate Personal Orgs
-- ============================================================================
-- Keep the oldest personal org for each user, delete newer duplicates

-- First, identify which personal orgs to keep (oldest per user)
WITH personal_orgs_ranked AS (
    SELECT
        om.user_id,
        o.id as org_id,
        o.created_at,
        ROW_NUMBER() OVER (PARTITION BY om.user_id ORDER BY o.created_at ASC) as rn
    FROM organization_members om
    JOIN organizations o ON o.id = om.organization_id
    WHERE o.is_personal = TRUE AND om.user_id IS NOT NULL
),
orgs_to_delete AS (
    SELECT org_id FROM personal_orgs_ranked WHERE rn > 1
)
-- Delete the duplicate personal orgs (cascade will clean up members)
DELETE FROM organizations WHERE id IN (SELECT org_id FROM orgs_to_delete);

-- ============================================================================
-- STEP 4: Clean Up Orphan Projects (assign to default org or delete)
-- ============================================================================
-- Option: Delete orphan projects (they have no org context anyway)
DELETE FROM projects WHERE organization_id IS NULL;

-- ============================================================================
-- STEP 5: Clean Up Orphan Test Schedules
-- ============================================================================
DELETE FROM test_schedules WHERE organization_id IS NULL;

-- ============================================================================
-- STEP 6: Add NOT NULL constraint to critical foreign keys (future protection)
-- ============================================================================
-- Note: We use SET NULL for projects on org delete, so we can't make it NOT NULL
-- But we can add a check constraint to prevent manual NULLs

-- For new projects, require organization_id
-- ALTER TABLE projects ALTER COLUMN organization_id SET NOT NULL;
-- Can't do this due to SET NULL cascade - but we'll validate in API layer

-- ============================================================================
-- STEP 7: Add indexes for better query performance on org lookups
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_organizations_is_personal
ON organizations(is_personal) WHERE is_personal = TRUE;

CREATE INDEX IF NOT EXISTS idx_org_members_user_active
ON organization_members(user_id) WHERE status = 'active';

-- ============================================================================
-- STEP 8: Add created_by tracking for organizations
-- ============================================================================
-- Ensure created_by is set for personal orgs
UPDATE organizations o
SET created_by = (
    SELECT om.user_id FROM organization_members om
    WHERE om.organization_id = o.id AND om.role = 'owner'
    LIMIT 1
)
WHERE o.created_by IS NULL AND o.is_personal = TRUE;

COMMIT;

-- ============================================================================
-- Verification Queries (run manually to verify)
-- ============================================================================
-- SELECT 'duplicate_personal_orgs' as check, COUNT(*) FROM (
--     SELECT om.user_id, COUNT(DISTINCT o.id) as cnt
--     FROM organization_members om
--     JOIN organizations o ON o.id = om.organization_id
--     WHERE o.is_personal = true AND om.user_id IS NOT NULL
--     GROUP BY om.user_id HAVING COUNT(DISTINCT o.id) > 1
-- ) x;
--
-- SELECT 'orphan_projects' as check, COUNT(*) FROM projects WHERE organization_id IS NULL;
-- SELECT 'orphan_schedules' as check, COUNT(*) FROM test_schedules WHERE organization_id IS NULL;
