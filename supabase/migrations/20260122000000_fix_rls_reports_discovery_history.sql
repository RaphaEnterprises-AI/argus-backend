-- Migration: Fix insecure RLS policies for reports and discovery_history tables
--
-- CRITICAL SECURITY FIX: These tables were created with USING(true) policies
-- allowing any authenticated user to access ALL data across ALL organizations
--
-- Fixes:
-- 1. reports table (created in 20260117300000_reports_table.sql)
-- 2. discovery_history table (created in 20260111_discovery_intelligence.sql)

-- =============================================================================
-- FIX: reports table RLS (has organization_id and project_id)
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON reports;

-- Users can read reports for their organizations
CREATE POLICY "Users can access reports for their organizations" ON reports
    FOR SELECT USING (
        auth.is_service_role() OR
        auth.has_org_access(organization_id)
    );

-- Users can insert reports into their organizations
CREATE POLICY "Users can insert reports into their organizations" ON reports
    FOR INSERT WITH CHECK (
        auth.is_service_role() OR
        auth.has_org_access(organization_id)
    );

-- Users can update reports in their organizations
CREATE POLICY "Users can update reports in their organizations" ON reports
    FOR UPDATE USING (
        auth.is_service_role() OR
        auth.has_org_access(organization_id)
    );

-- Users can delete reports in their organizations
CREATE POLICY "Users can delete reports in their organizations" ON reports
    FOR DELETE USING (
        auth.is_service_role() OR
        auth.has_org_access(organization_id)
    );

-- =============================================================================
-- FIX: discovery_history table RLS (has session_id -> project_id -> org_id)
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovery_history;
DROP POLICY IF EXISTS "Allow all operations" ON discovery_history;

-- Users can access discovery history via session -> project -> organization chain
CREATE POLICY "Users can access discovery history via session" ON discovery_history
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM discovery_sessions ds
            JOIN projects p ON p.id = ds.project_id
            WHERE ds.id = discovery_history.session_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- =============================================================================
-- Add comments documenting the security model
-- =============================================================================

COMMENT ON POLICY "Users can access reports for their organizations" ON reports IS
'Reports are organization-scoped. Users can only access reports for organizations they belong to.';

COMMENT ON POLICY "Users can access discovery history via session" ON discovery_history IS
'Discovery history is accessed via session -> project -> organization chain for proper multi-tenant isolation.';
