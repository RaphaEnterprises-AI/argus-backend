-- Migration: Fix test_schedules and schedule_runs RLS policies
-- The backend API uses service_role, so we just need service role access.
-- User-level access is controlled by the Python API layer.

-- =============================================================================
-- FIX: test_schedules table RLS
-- =============================================================================

-- Drop old policies
DROP POLICY IF EXISTS "Service role has full access to test_schedules" ON test_schedules;
DROP POLICY IF EXISTS "Users can view schedules for their projects" ON test_schedules;
DROP POLICY IF EXISTS "Users can manage schedules for their projects" ON test_schedules;

-- Create service role policy
CREATE POLICY "Service role full access" ON test_schedules
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- =============================================================================
-- FIX: schedule_runs table RLS
-- =============================================================================

-- Drop old policies
DROP POLICY IF EXISTS "Service role has full access to schedule_runs" ON schedule_runs;
DROP POLICY IF EXISTS "Users can view schedule runs" ON schedule_runs;
DROP POLICY IF EXISTS "Users can manage schedule runs" ON schedule_runs;

-- Create service role policy
CREATE POLICY "Service role full access" ON schedule_runs
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- =============================================================================
-- COMPLETION
-- =============================================================================

SELECT 'Fixed test_schedules and schedule_runs RLS policies!' as message;
