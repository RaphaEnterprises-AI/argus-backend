-- Migration: Fix test_schedules and schedule_runs RLS policies
-- The original policies used current_setting('role', true) which doesn't work correctly
-- with Supabase service role authentication. This migration updates them to use
-- the auth.is_service_role() function from the RLS security fix migration.

-- =============================================================================
-- FIX: test_schedules table RLS
-- =============================================================================

-- Drop old service role policy
DROP POLICY IF EXISTS "Service role has full access to test_schedules" ON test_schedules;

-- Create new service role policy using the correct check
CREATE POLICY "Service role has full access to test_schedules" ON test_schedules
    FOR ALL USING (auth.is_service_role());

-- Also add a policy for authenticated users to access schedules via project/org membership
DROP POLICY IF EXISTS "Users can view schedules for their projects" ON test_schedules;
CREATE POLICY "Users can view schedules for their projects" ON test_schedules
    FOR SELECT USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM projects p
            WHERE p.id = test_schedules.project_id
            AND (
                p.organization_id IS NULL OR
                auth.has_org_access(p.organization_id)
            )
        )
    );

DROP POLICY IF EXISTS "Users can manage schedules for their projects" ON test_schedules;
CREATE POLICY "Users can manage schedules for their projects" ON test_schedules
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM projects p
            WHERE p.id = test_schedules.project_id
            AND (
                p.organization_id IS NULL OR
                auth.has_org_access(p.organization_id)
            )
        )
    );

-- =============================================================================
-- FIX: schedule_runs table RLS
-- =============================================================================

-- Drop old service role policy
DROP POLICY IF EXISTS "Service role has full access to schedule_runs" ON schedule_runs;

-- Create new service role policy using the correct check
CREATE POLICY "Service role has full access to schedule_runs" ON schedule_runs
    FOR ALL USING (auth.is_service_role());

-- Also add a policy for authenticated users to access runs via schedule/project membership
DROP POLICY IF EXISTS "Users can view schedule runs" ON schedule_runs;
CREATE POLICY "Users can view schedule runs" ON schedule_runs
    FOR SELECT USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM test_schedules ts
            JOIN projects p ON p.id = ts.project_id
            WHERE ts.id = schedule_runs.schedule_id
            AND (
                p.organization_id IS NULL OR
                auth.has_org_access(p.organization_id)
            )
        )
    );

DROP POLICY IF EXISTS "Users can manage schedule runs" ON schedule_runs;
CREATE POLICY "Users can manage schedule runs" ON schedule_runs
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM test_schedules ts
            JOIN projects p ON p.id = ts.project_id
            WHERE ts.id = schedule_runs.schedule_id
            AND (
                p.organization_id IS NULL OR
                auth.has_org_access(p.organization_id)
            )
        )
    );

-- =============================================================================
-- COMPLETION
-- =============================================================================

SELECT 'Fixed test_schedules and schedule_runs RLS policies!' as message;
