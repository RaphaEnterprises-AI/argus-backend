-- Migration: Fix insecure RLS policies that allow unrestricted access
-- This replaces all USING(true) policies with proper organization-scoped access control
-- 
-- CRITICAL SECURITY FIX: Multiple tables had USING(true) policies allowing any 
-- authenticated user to access ALL data across ALL organizations

-- =============================================================================
-- Helper function to get current user's organization IDs from JWT claims
-- =============================================================================

CREATE OR REPLACE FUNCTION auth.user_org_ids()
RETURNS UUID[] AS $$
DECLARE
    org_ids UUID[];
    raw_orgs JSONB;
BEGIN
    -- Try to get organization_ids from JWT claims (set by our API)
    raw_orgs := current_setting('request.jwt.claims', true)::jsonb -> 'organization_ids';
    
    IF raw_orgs IS NOT NULL AND jsonb_array_length(raw_orgs) > 0 THEN
        SELECT array_agg(elem::text::uuid)
        INTO org_ids
        FROM jsonb_array_elements_text(raw_orgs) AS elem;
        RETURN org_ids;
    END IF;
    
    -- Fallback: check organization_members table
    SELECT array_agg(om.organization_id)
    INTO org_ids
    FROM organization_members om
    WHERE om.user_id = auth.uid()::text;
    
    RETURN COALESCE(org_ids, ARRAY[]::UUID[]);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- =============================================================================
-- Helper function to check if user has access to a specific organization
-- =============================================================================

CREATE OR REPLACE FUNCTION auth.has_org_access(check_org_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    IF check_org_id IS NULL THEN
        RETURN FALSE;
    END IF;
    RETURN check_org_id = ANY(auth.user_org_ids());
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- =============================================================================
-- Helper function to check if current request is from service role
-- =============================================================================

CREATE OR REPLACE FUNCTION auth.is_service_role()
RETURNS BOOLEAN AS $$
BEGIN
    RETURN COALESCE(
        current_setting('request.jwt.claims', true)::jsonb ->> 'role' = 'service_role',
        FALSE
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- =============================================================================
-- FIX: projects table RLS
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON projects;

-- Users can read projects they have access to via organization membership
CREATE POLICY "Users can read own org projects" ON projects
    FOR SELECT USING (
        auth.is_service_role() OR
        organization_id IS NULL OR  -- Allow null org projects for backwards compat
        auth.has_org_access(organization_id)
    );

-- Users can insert projects into their organizations
CREATE POLICY "Users can insert into own orgs" ON projects
    FOR INSERT WITH CHECK (
        auth.is_service_role() OR
        organization_id IS NULL OR
        auth.has_org_access(organization_id)
    );

-- Users can update projects in their organizations
CREATE POLICY "Users can update own org projects" ON projects
    FOR UPDATE USING (
        auth.is_service_role() OR
        organization_id IS NULL OR
        auth.has_org_access(organization_id)
    );

-- Users can delete projects in their organizations
CREATE POLICY "Users can delete own org projects" ON projects
    FOR DELETE USING (
        auth.is_service_role() OR
        organization_id IS NULL OR
        auth.has_org_access(organization_id)
    );

-- =============================================================================
-- FIX: risk_scores table RLS (has project_id FK)
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON risk_scores;

CREATE POLICY "Users can access risk scores for their projects" ON risk_scores
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM projects p
            WHERE p.id = risk_scores.project_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- =============================================================================
-- FIX: generated_tests table RLS (has project_id FK)
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON generated_tests;

CREATE POLICY "Users can access generated tests for their projects" ON generated_tests
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM projects p
            WHERE p.id = generated_tests.project_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- =============================================================================
-- FIX: ci_events table RLS (has project_id FK)
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON ci_events;

CREATE POLICY "Users can access CI events for their projects" ON ci_events
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM projects p
            WHERE p.id = ci_events.project_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- =============================================================================
-- FIX: coverage_reports table RLS (has project_id FK)
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON coverage_reports;

CREATE POLICY "Users can access coverage reports for their projects" ON coverage_reports
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM projects p
            WHERE p.id = coverage_reports.project_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- =============================================================================
-- FIX: webhook_logs table RLS (has project_id FK)
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON webhook_logs;

CREATE POLICY "Users can access webhook logs for their projects" ON webhook_logs
    FOR ALL USING (
        auth.is_service_role() OR
        project_id IS NULL OR  -- System webhooks without project
        EXISTS (
            SELECT 1 FROM projects p
            WHERE p.id = webhook_logs.project_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- =============================================================================
-- FIX: quality_scores table RLS (has project_id FK)
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON quality_scores;

CREATE POLICY "Users can access quality scores for their projects" ON quality_scores
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM projects p
            WHERE p.id = quality_scores.project_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- =============================================================================
-- FIX: langgraph_checkpoints table RLS
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON langgraph_checkpoints;
DROP POLICY IF EXISTS "Allow all operations" ON langgraph_checkpoints;

-- Checkpoints are tied to thread_id which should include org context
-- For now, restrict to service role only as these are internal
CREATE POLICY "Service role only for checkpoints" ON langgraph_checkpoints
    FOR ALL USING (auth.is_service_role());

-- =============================================================================
-- FIX: langgraph_checkpoint_writes table RLS
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON langgraph_checkpoint_writes;
DROP POLICY IF EXISTS "Allow all operations" ON langgraph_checkpoint_writes;

CREATE POLICY "Service role only for checkpoint writes" ON langgraph_checkpoint_writes
    FOR ALL USING (auth.is_service_role());

-- =============================================================================
-- FIX: langgraph_memory_store table RLS
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON langgraph_memory_store;
DROP POLICY IF EXISTS "Allow all operations" ON langgraph_memory_store;

CREATE POLICY "Service role only for memory store" ON langgraph_memory_store
    FOR ALL USING (auth.is_service_role());

-- =============================================================================
-- FIX: test_failure_patterns table RLS (has project_id FK)
-- =============================================================================

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON test_failure_patterns;
DROP POLICY IF EXISTS "Allow all operations" ON test_failure_patterns;

CREATE POLICY "Users can access failure patterns for their projects" ON test_failure_patterns
    FOR ALL USING (
        auth.is_service_role() OR
        project_id IS NULL OR
        EXISTS (
            SELECT 1 FROM projects p
            WHERE p.id = test_failure_patterns.project_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- =============================================================================
-- FIX: Discovery tables RLS (from 20260111_discovery_intelligence.sql)
-- =============================================================================

-- discovery_sessions
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovery_sessions;
DROP POLICY IF EXISTS "Allow all operations" ON discovery_sessions;

CREATE POLICY "Users can access discovery sessions for their projects" ON discovery_sessions
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM projects p
            WHERE p.id = discovery_sessions.project_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- discovered_pages
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovered_pages;
DROP POLICY IF EXISTS "Allow all operations" ON discovered_pages;

CREATE POLICY "Users can access discovered pages via session" ON discovered_pages
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM discovery_sessions ds
            JOIN projects p ON p.id = ds.project_id
            WHERE ds.id = discovered_pages.session_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- discovered_elements
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovered_elements;
DROP POLICY IF EXISTS "Allow all operations" ON discovered_elements;

CREATE POLICY "Users can access discovered elements via page" ON discovered_elements
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM discovered_pages dp
            JOIN discovery_sessions ds ON ds.id = dp.session_id
            JOIN projects p ON p.id = ds.project_id
            WHERE dp.id = discovered_elements.page_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- discovered_flows
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovered_flows;
DROP POLICY IF EXISTS "Allow all operations" ON discovered_flows;

CREATE POLICY "Users can access discovered flows via session" ON discovered_flows
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM discovery_sessions ds
            JOIN projects p ON p.id = ds.project_id
            WHERE ds.id = discovered_flows.session_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- discovery_patterns
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovery_patterns;
DROP POLICY IF EXISTS "Allow all operations" ON discovery_patterns;

CREATE POLICY "Users can access discovery patterns for their projects" ON discovery_patterns
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM projects p
            WHERE p.id = discovery_patterns.project_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- discovery_insights
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovery_insights;
DROP POLICY IF EXISTS "Allow all operations" ON discovery_insights;

CREATE POLICY "Users can access discovery insights via session" ON discovery_insights
    FOR ALL USING (
        auth.is_service_role() OR
        EXISTS (
            SELECT 1 FROM discovery_sessions ds
            JOIN projects p ON p.id = ds.project_id
            WHERE ds.id = discovery_insights.session_id
            AND (p.organization_id IS NULL OR auth.has_org_access(p.organization_id))
        )
    );

-- =============================================================================
-- Add comment documenting the security model
-- =============================================================================

COMMENT ON FUNCTION auth.user_org_ids() IS 
'Returns array of organization IDs the current user has access to.
First tries JWT claims (set by API), falls back to organization_members lookup.
Used by RLS policies for multi-tenant data isolation.';

COMMENT ON FUNCTION auth.has_org_access(UUID) IS
'Checks if current user has access to the specified organization.
Returns FALSE for NULL org_id to fail closed.';

COMMENT ON FUNCTION auth.is_service_role() IS
'Checks if current request is from service role (backend API).
Service role bypasses RLS for internal operations.';
