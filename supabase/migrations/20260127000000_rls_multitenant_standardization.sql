-- =============================================================================
-- RLS Multi-Tenant Standardization Migration
-- =============================================================================
--
-- This migration standardizes all Row Level Security (RLS) policies across the
-- database to use a consistent pattern with the auth.has_org_access() helper.
--
-- PROBLEM: Three different RLS patterns were in use:
-- 1. auth.has_org_access() helper function (from fix_rls_security.sql)
-- 2. Direct subquery with auth.uid() (from test_impact_graph.sql)
-- 3. current_setting('app.user_id', true) (from organizations_and_ai_tracking.sql)
--
-- SOLUTION: Standardize on auth.has_org_access() with service role bypass
-- =============================================================================

-- =============================================================================
-- Part 1: Enhance Helper Functions
-- =============================================================================

-- Drop and recreate auth.user_org_ids with better fallback chain
CREATE OR REPLACE FUNCTION auth.user_org_ids()
RETURNS UUID[] AS $$
DECLARE
    org_ids UUID[];
    raw_orgs JSONB;
    current_user_id TEXT;
BEGIN
    -- Try 1: Get organization_ids from JWT claims (set by our API middleware)
    raw_orgs := current_setting('request.jwt.claims', true)::jsonb -> 'organization_ids';

    IF raw_orgs IS NOT NULL AND jsonb_array_length(raw_orgs) > 0 THEN
        SELECT array_agg(elem::text::uuid)
        INTO org_ids
        FROM jsonb_array_elements_text(raw_orgs) AS elem;
        RETURN org_ids;
    END IF;

    -- Try 2: Get user ID from various sources
    -- Priority: app.user_id > JWT sub > auth.uid()
    current_user_id := COALESCE(
        current_setting('app.user_id', true),
        current_setting('request.jwt.claims', true)::jsonb ->> 'sub',
        auth.uid()::text
    );

    IF current_user_id IS NOT NULL AND current_user_id != '' THEN
        -- Lookup organization memberships
        SELECT array_agg(om.organization_id)
        INTO org_ids
        FROM organization_members om
        WHERE om.user_id = current_user_id
          AND om.status = 'active';

        RETURN COALESCE(org_ids, ARRAY[]::UUID[]);
    END IF;

    RETURN ARRAY[]::UUID[];
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- Enhanced org access check with project support
CREATE OR REPLACE FUNCTION auth.has_org_access(check_org_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    -- Fail closed for NULL
    IF check_org_id IS NULL THEN
        RETURN FALSE;
    END IF;

    -- Service role always has access
    IF auth.is_service_role() THEN
        RETURN TRUE;
    END IF;

    RETURN check_org_id = ANY(auth.user_org_ids());
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- New: Check project access via organization
CREATE OR REPLACE FUNCTION auth.has_project_access(check_project_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    project_org_id UUID;
BEGIN
    IF check_project_id IS NULL THEN
        RETURN FALSE;
    END IF;

    IF auth.is_service_role() THEN
        RETURN TRUE;
    END IF;

    -- Get project's organization
    SELECT organization_id INTO project_org_id
    FROM projects
    WHERE id = check_project_id;

    -- Allow if org is NULL (legacy) or user has access
    RETURN project_org_id IS NULL OR auth.has_org_access(project_org_id);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- Get current user's ID from various sources
CREATE OR REPLACE FUNCTION auth.current_user_id()
RETURNS TEXT AS $$
BEGIN
    RETURN COALESCE(
        current_setting('app.user_id', true),
        current_setting('request.jwt.claims', true)::jsonb ->> 'sub',
        auth.uid()::text
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- =============================================================================
-- Part 2: Standardize Organizations Table Policies
-- =============================================================================

DROP POLICY IF EXISTS "Users can view their organizations" ON organizations;
DROP POLICY IF EXISTS "Admins can update their organizations" ON organizations;
DROP POLICY IF EXISTS "Service role has full access to organizations" ON organizations;

CREATE POLICY "org_select" ON organizations
    FOR SELECT USING (
        auth.is_service_role() OR
        auth.has_org_access(id)
    );

CREATE POLICY "org_update" ON organizations
    FOR UPDATE USING (
        auth.is_service_role() OR
        (auth.has_org_access(id) AND EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_id = organizations.id
            AND user_id = auth.current_user_id()
            AND role IN ('owner', 'admin')
        ))
    );

CREATE POLICY "org_insert" ON organizations
    FOR INSERT WITH CHECK (auth.is_service_role());

CREATE POLICY "org_delete" ON organizations
    FOR DELETE USING (auth.is_service_role());

-- =============================================================================
-- Part 3: Standardize Organization Members Policies
-- =============================================================================

DROP POLICY IF EXISTS "Users can view members of their organizations" ON organization_members;
DROP POLICY IF EXISTS "Service role has full access to organization_members" ON organization_members;

CREATE POLICY "org_members_select" ON organization_members
    FOR SELECT USING (
        auth.is_service_role() OR
        auth.has_org_access(organization_id)
    );

CREATE POLICY "org_members_insert" ON organization_members
    FOR INSERT WITH CHECK (
        auth.is_service_role() OR
        (auth.has_org_access(organization_id) AND EXISTS (
            SELECT 1 FROM organization_members om
            WHERE om.organization_id = organization_members.organization_id
            AND om.user_id = auth.current_user_id()
            AND om.role IN ('owner', 'admin')
        ))
    );

CREATE POLICY "org_members_update" ON organization_members
    FOR UPDATE USING (
        auth.is_service_role() OR
        (auth.has_org_access(organization_id) AND EXISTS (
            SELECT 1 FROM organization_members om
            WHERE om.organization_id = organization_members.organization_id
            AND om.user_id = auth.current_user_id()
            AND om.role IN ('owner', 'admin')
        ))
    );

CREATE POLICY "org_members_delete" ON organization_members
    FOR DELETE USING (
        auth.is_service_role() OR
        (auth.has_org_access(organization_id) AND EXISTS (
            SELECT 1 FROM organization_members om
            WHERE om.organization_id = organization_members.organization_id
            AND om.user_id = auth.current_user_id()
            AND om.role = 'owner'
        ))
    );

-- =============================================================================
-- Part 4: Standardize AI Usage Tables
-- =============================================================================

DROP POLICY IF EXISTS "Users can view AI usage for their organizations" ON ai_usage;
DROP POLICY IF EXISTS "Service role has full access to ai_usage" ON ai_usage;

CREATE POLICY "ai_usage_policy" ON ai_usage
    FOR ALL USING (
        auth.is_service_role() OR
        auth.has_org_access(organization_id)
    );

DROP POLICY IF EXISTS "Users can view daily AI usage for their organizations" ON ai_usage_daily;
DROP POLICY IF EXISTS "Service role has full access to ai_usage_daily" ON ai_usage_daily;

CREATE POLICY "ai_usage_daily_policy" ON ai_usage_daily
    FOR ALL USING (
        auth.is_service_role() OR
        auth.has_org_access(organization_id)
    );

-- =============================================================================
-- Part 5: Standardize API Keys Table
-- =============================================================================

DROP POLICY IF EXISTS "Users can manage API keys for their organizations" ON api_keys;
DROP POLICY IF EXISTS "Service role has full access to api_keys" ON api_keys;

CREATE POLICY "api_keys_select" ON api_keys
    FOR SELECT USING (
        auth.is_service_role() OR
        auth.has_org_access(organization_id)
    );

CREATE POLICY "api_keys_manage" ON api_keys
    FOR ALL USING (
        auth.is_service_role() OR
        (auth.has_org_access(organization_id) AND EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_id = api_keys.organization_id
            AND user_id = auth.current_user_id()
            AND role IN ('owner', 'admin')
        ))
    );

-- =============================================================================
-- Part 6: Standardize Test Impact Graph Tables (replace subquery pattern)
-- =============================================================================

-- test_impact_graph
DROP POLICY IF EXISTS "Users can view impact graph for their projects" ON test_impact_graph;
DROP POLICY IF EXISTS "Users can manage impact graph for their projects" ON test_impact_graph;

CREATE POLICY "test_impact_graph_policy" ON test_impact_graph
    FOR ALL USING (
        auth.is_service_role() OR
        auth.has_project_access(project_id)
    );

-- coverage_imports
DROP POLICY IF EXISTS "Users can view coverage imports for their projects" ON coverage_imports;
DROP POLICY IF EXISTS "Users can manage coverage imports for their projects" ON coverage_imports;

CREATE POLICY "coverage_imports_policy" ON coverage_imports
    FOR ALL USING (
        auth.is_service_role() OR
        auth.has_project_access(project_id)
    );

-- impact_graph_jobs
DROP POLICY IF EXISTS "Users can view jobs for their projects" ON impact_graph_jobs;
DROP POLICY IF EXISTS "Users can manage jobs for their projects" ON impact_graph_jobs;

CREATE POLICY "impact_graph_jobs_policy" ON impact_graph_jobs
    FOR ALL USING (
        auth.is_service_role() OR
        auth.has_project_access(project_id)
    );

-- file_dependencies
DROP POLICY IF EXISTS "Users can view file deps for their projects" ON file_dependencies;
DROP POLICY IF EXISTS "Users can manage file deps for their projects" ON file_dependencies;

CREATE POLICY "file_dependencies_policy" ON file_dependencies
    FOR ALL USING (
        auth.is_service_role() OR
        auth.has_project_access(project_id)
    );

-- failure_correlations
DROP POLICY IF EXISTS "Users can view failure correlations for their projects" ON failure_correlations;
DROP POLICY IF EXISTS "Users can manage failure correlations for their projects" ON failure_correlations;

CREATE POLICY "failure_correlations_policy" ON failure_correlations
    FOR ALL USING (
        auth.is_service_role() OR
        auth.has_project_access(project_id)
    );

-- =============================================================================
-- Part 7: Standardize Visual AI Tables
-- =============================================================================

-- Check if policies exist before dropping
DO $$
BEGIN
    DROP POLICY IF EXISTS "Users can manage visual baselines" ON visual_baselines;
    DROP POLICY IF EXISTS "Users can manage visual snapshots" ON visual_snapshots;
    DROP POLICY IF EXISTS "Users can manage visual comparisons" ON visual_comparisons;
    DROP POLICY IF EXISTS "Users can manage visual changes" ON visual_changes;
    DROP POLICY IF EXISTS "Users can manage viewport presets" ON viewport_presets;
    DROP POLICY IF EXISTS "Users can manage browser matrix" ON browser_matrix;
    DROP POLICY IF EXISTS "Users can manage visual test history" ON visual_test_history;
EXCEPTION WHEN undefined_table THEN
    NULL; -- Tables might not exist yet
END $$;

-- Create standardized policies (only if tables exist)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'visual_baselines') THEN
        EXECUTE 'CREATE POLICY visual_baselines_policy ON visual_baselines FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'visual_snapshots') THEN
        EXECUTE 'CREATE POLICY visual_snapshots_policy ON visual_snapshots FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'visual_comparisons') THEN
        EXECUTE 'CREATE POLICY visual_comparisons_policy ON visual_comparisons FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'visual_changes') THEN
        EXECUTE 'CREATE POLICY visual_changes_policy ON visual_changes FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'viewport_presets') THEN
        EXECUTE 'CREATE POLICY viewport_presets_policy ON viewport_presets FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'browser_matrix') THEN
        EXECUTE 'CREATE POLICY browser_matrix_policy ON browser_matrix FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'visual_test_history') THEN
        EXECUTE 'CREATE POLICY visual_test_history_policy ON visual_test_history FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'visual_baseline_history') THEN
        EXECUTE 'CREATE POLICY visual_baseline_history_policy ON visual_baseline_history FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;
EXCEPTION WHEN duplicate_object THEN
    NULL; -- Policy already exists
END $$;

-- =============================================================================
-- Part 8: Standardize Failure Patterns Tables
-- =============================================================================

DO $$
BEGIN
    DROP POLICY IF EXISTS "Users can manage failure patterns" ON failure_patterns;
    DROP POLICY IF EXISTS "Users can manage pattern training jobs" ON pattern_training_jobs;
    DROP POLICY IF EXISTS "Users can manage pattern predictions" ON pattern_predictions;
    DROP POLICY IF EXISTS "Users can manage commit features" ON commit_features;
EXCEPTION WHEN undefined_table THEN
    NULL;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'failure_patterns') THEN
        EXECUTE 'CREATE POLICY failure_patterns_policy ON failure_patterns FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'pattern_training_jobs') THEN
        EXECUTE 'CREATE POLICY pattern_training_jobs_policy ON pattern_training_jobs FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'pattern_predictions') THEN
        EXECUTE 'CREATE POLICY pattern_predictions_policy ON pattern_predictions FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'commit_features') THEN
        EXECUTE 'CREATE POLICY commit_features_policy ON commit_features FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;

-- =============================================================================
-- Part 9: Standardize SDLC Events Tables
-- =============================================================================

DO $$
BEGIN
    DROP POLICY IF EXISTS "Users can manage sdlc events" ON sdlc_events;
    DROP POLICY IF EXISTS "Users can manage event correlations" ON event_correlations;
    DROP POLICY IF EXISTS "Users can manage correlation insights" ON correlation_insights;
    DROP POLICY IF EXISTS "Users can manage commit analyses" ON commit_analyses;
    DROP POLICY IF EXISTS "Users can manage deployment events" ON deployment_events;
EXCEPTION WHEN undefined_table THEN
    NULL;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'sdlc_events') THEN
        EXECUTE 'CREATE POLICY sdlc_events_policy ON sdlc_events FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'event_correlations') THEN
        EXECUTE 'CREATE POLICY event_correlations_policy ON event_correlations FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'correlation_insights') THEN
        EXECUTE 'CREATE POLICY correlation_insights_policy ON correlation_insights FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'commit_analyses') THEN
        EXECUTE 'CREATE POLICY commit_analyses_policy ON commit_analyses FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'deployment_events') THEN
        EXECUTE 'CREATE POLICY deployment_events_policy ON deployment_events FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;

-- =============================================================================
-- Part 10: Standardize API Testing Tables
-- =============================================================================

DO $$
BEGIN
    DROP POLICY IF EXISTS "Users can manage api endpoints" ON api_endpoints;
    DROP POLICY IF EXISTS "Users can manage api test cases" ON api_test_cases;
    DROP POLICY IF EXISTS "Users can manage api test results" ON api_test_results;
    DROP POLICY IF EXISTS "Users can manage api test suites" ON api_test_suites;
    DROP POLICY IF EXISTS "Users can manage api discovery sessions" ON api_discovery_sessions;
EXCEPTION WHEN undefined_table THEN
    NULL;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'api_endpoints') THEN
        EXECUTE 'CREATE POLICY api_endpoints_policy ON api_endpoints FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'api_test_cases') THEN
        EXECUTE 'CREATE POLICY api_test_cases_policy ON api_test_cases FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'api_test_results') THEN
        EXECUTE 'CREATE POLICY api_test_results_policy ON api_test_results FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'api_test_suites') THEN
        EXECUTE 'CREATE POLICY api_test_suites_policy ON api_test_suites FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'api_discovery_sessions') THEN
        EXECUTE 'CREATE POLICY api_discovery_sessions_policy ON api_discovery_sessions FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;

-- =============================================================================
-- Part 11: Standardize Infra Recommendations Tables
-- =============================================================================

DO $$
BEGIN
    DROP POLICY IF EXISTS "Users can manage infra recommendations" ON infra_recommendations;
    DROP POLICY IF EXISTS "Users can manage infra cost history" ON infra_cost_history;
    DROP POLICY IF EXISTS "Users can manage infra anomaly history" ON infra_anomaly_history;
EXCEPTION WHEN undefined_table THEN
    NULL;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'infra_recommendations') THEN
        EXECUTE 'CREATE POLICY infra_recommendations_policy ON infra_recommendations FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'infra_cost_history') THEN
        EXECUTE 'CREATE POLICY infra_cost_history_policy ON infra_cost_history FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'infra_anomaly_history') THEN
        EXECUTE 'CREATE POLICY infra_anomaly_history_policy ON infra_anomaly_history FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;

-- =============================================================================
-- Part 12: Standardize Graph Tables
-- =============================================================================

DO $$
BEGIN
    DROP POLICY IF EXISTS "Users can manage graph metadata" ON graph_metadata;
    DROP POLICY IF EXISTS "Users can manage graph test vertices" ON graph_test_vertices;
    DROP POLICY IF EXISTS "Users can manage graph selector vertices" ON graph_selector_vertices;
    DROP POLICY IF EXISTS "Users can manage graph failure vertices" ON graph_failure_vertices;
    DROP POLICY IF EXISTS "Users can manage graph code change vertices" ON graph_code_change_vertices;
    DROP POLICY IF EXISTS "Users can manage graph healing pattern vertices" ON graph_healing_pattern_vertices;
    DROP POLICY IF EXISTS "Users can manage graph page vertices" ON graph_page_vertices;
    DROP POLICY IF EXISTS "Users can manage graph project vertices" ON graph_project_vertices;
    DROP POLICY IF EXISTS "Users can manage graph edges" ON graph_edges;
EXCEPTION WHEN undefined_table THEN
    NULL;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'graph_metadata') THEN
        EXECUTE 'CREATE POLICY graph_metadata_policy ON graph_metadata FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'graph_test_vertices') THEN
        EXECUTE 'CREATE POLICY graph_test_vertices_policy ON graph_test_vertices FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'graph_selector_vertices') THEN
        EXECUTE 'CREATE POLICY graph_selector_vertices_policy ON graph_selector_vertices FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'graph_failure_vertices') THEN
        EXECUTE 'CREATE POLICY graph_failure_vertices_policy ON graph_failure_vertices FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'graph_code_change_vertices') THEN
        EXECUTE 'CREATE POLICY graph_code_change_vertices_policy ON graph_code_change_vertices FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'graph_healing_pattern_vertices') THEN
        EXECUTE 'CREATE POLICY graph_healing_pattern_vertices_policy ON graph_healing_pattern_vertices FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'graph_page_vertices') THEN
        EXECUTE 'CREATE POLICY graph_page_vertices_policy ON graph_page_vertices FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'graph_project_vertices') THEN
        EXECUTE 'CREATE POLICY graph_project_vertices_policy ON graph_project_vertices FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'graph_edges') THEN
        EXECUTE 'CREATE POLICY graph_edges_policy ON graph_edges FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;

-- =============================================================================
-- Part 13: Standardize Notification Tables
-- =============================================================================

DO $$
BEGIN
    DROP POLICY IF EXISTS "Users can manage notification channels" ON notification_channels;
    DROP POLICY IF EXISTS "Users can manage notification rules" ON notification_rules;
    DROP POLICY IF EXISTS "Users can manage notification logs" ON notification_logs;
    DROP POLICY IF EXISTS "Users can manage notification templates" ON notification_templates;
EXCEPTION WHEN undefined_table THEN
    NULL;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'notification_channels') THEN
        EXECUTE 'CREATE POLICY notification_channels_policy ON notification_channels FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'notification_rules') THEN
        EXECUTE 'CREATE POLICY notification_rules_policy ON notification_rules FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'notification_logs') THEN
        EXECUTE 'CREATE POLICY notification_logs_policy ON notification_logs FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'notification_templates') THEN
        EXECUTE 'CREATE POLICY notification_templates_policy ON notification_templates FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;

-- =============================================================================
-- Part 14: Standardize Parameterized Test Tables
-- =============================================================================

DO $$
BEGIN
    DROP POLICY IF EXISTS "Users can manage parameterized tests" ON parameterized_tests;
    DROP POLICY IF EXISTS "Users can manage parameter sets" ON parameter_sets;
    DROP POLICY IF EXISTS "Users can manage parameterized results" ON parameterized_results;
    DROP POLICY IF EXISTS "Users can manage iteration results" ON iteration_results;
EXCEPTION WHEN undefined_table THEN
    NULL;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'parameterized_tests') THEN
        EXECUTE 'CREATE POLICY parameterized_tests_policy ON parameterized_tests FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'parameter_sets') THEN
        EXECUTE 'CREATE POLICY parameter_sets_policy ON parameter_sets FOR ALL USING (auth.is_service_role() OR EXISTS (SELECT 1 FROM parameterized_tests pt WHERE pt.id = parameter_sets.test_id AND auth.has_project_access(pt.project_id)))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'parameterized_results') THEN
        EXECUTE 'CREATE POLICY parameterized_results_policy ON parameterized_results FOR ALL USING (auth.is_service_role() OR EXISTS (SELECT 1 FROM parameterized_tests pt WHERE pt.id = parameterized_results.test_id AND auth.has_project_access(pt.project_id)))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'iteration_results') THEN
        EXECUTE 'CREATE POLICY iteration_results_policy ON iteration_results FOR ALL USING (auth.is_service_role() OR EXISTS (SELECT 1 FROM parameterized_results pr JOIN parameterized_tests pt ON pt.id = pr.test_id WHERE pr.id = iteration_results.result_id AND auth.has_project_access(pt.project_id)))';
    END IF;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;

-- =============================================================================
-- Part 15: Standardize MCP Tables
-- =============================================================================

DO $$
BEGIN
    DROP POLICY IF EXISTS "Users can manage mcp connections" ON mcp_connections;
    DROP POLICY IF EXISTS "Users can manage mcp connection activity" ON mcp_connection_activity;
    DROP POLICY IF EXISTS "Users can manage mcp screenshots" ON mcp_screenshots;
EXCEPTION WHEN undefined_table THEN
    NULL;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'mcp_connections') THEN
        EXECUTE 'CREATE POLICY mcp_connections_policy ON mcp_connections FOR ALL USING (auth.is_service_role() OR auth.has_org_access(organization_id))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'mcp_connection_activity') THEN
        EXECUTE 'CREATE POLICY mcp_connection_activity_policy ON mcp_connection_activity FOR ALL USING (auth.is_service_role() OR EXISTS (SELECT 1 FROM mcp_connections mc WHERE mc.id = mcp_connection_activity.connection_id AND auth.has_org_access(mc.organization_id)))';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'mcp_screenshots') THEN
        EXECUTE 'CREATE POLICY mcp_screenshots_policy ON mcp_screenshots FOR ALL USING (auth.is_service_role() OR auth.has_project_access(project_id))';
    END IF;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;

-- =============================================================================
-- Part 16: Standardize Audit Logs (org-scoped)
-- =============================================================================

DO $$
BEGIN
    DROP POLICY IF EXISTS "Users can view audit logs for their organizations" ON audit_logs;
    DROP POLICY IF EXISTS "Service role can manage audit logs" ON audit_logs;
EXCEPTION WHEN undefined_table THEN
    NULL;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'audit_logs') THEN
        EXECUTE 'CREATE POLICY audit_logs_policy ON audit_logs FOR ALL USING (auth.is_service_role() OR auth.has_org_access(organization_id))';
    END IF;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;

-- =============================================================================
-- Part 17: Standardize User Profiles (user-scoped)
-- =============================================================================

DO $$
BEGIN
    DROP POLICY IF EXISTS "Users can view own profile" ON user_profiles;
    DROP POLICY IF EXISTS "Users can update own profile" ON user_profiles;
    DROP POLICY IF EXISTS "Service role can manage user profiles" ON user_profiles;
EXCEPTION WHEN undefined_table THEN
    NULL;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'user_profiles') THEN
        EXECUTE 'CREATE POLICY user_profiles_select ON user_profiles FOR SELECT USING (auth.is_service_role() OR user_id = auth.current_user_id())';
        EXECUTE 'CREATE POLICY user_profiles_update ON user_profiles FOR UPDATE USING (auth.is_service_role() OR user_id = auth.current_user_id())';
        EXECUTE 'CREATE POLICY user_profiles_insert ON user_profiles FOR INSERT WITH CHECK (auth.is_service_role() OR user_id = auth.current_user_id())';
    END IF;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;

-- =============================================================================
-- Part 18: Add Comments
-- =============================================================================

COMMENT ON FUNCTION auth.user_org_ids() IS
'Returns array of organization IDs the current user has access to.
Priority: JWT claims > app.user_id setting > organization_members lookup.
Used by RLS policies for multi-tenant data isolation.';

COMMENT ON FUNCTION auth.has_org_access(UUID) IS
'Checks if current user has access to the specified organization.
Returns TRUE for service role, FALSE for NULL org_id.
Primary RLS helper for org-scoped tables.';

COMMENT ON FUNCTION auth.has_project_access(UUID) IS
'Checks if current user has access to a project via its organization.
Returns TRUE for service role, TRUE for NULL org (legacy).
Primary RLS helper for project-scoped tables.';

COMMENT ON FUNCTION auth.current_user_id() IS
'Returns the current user ID from various sources.
Priority: app.user_id > JWT sub > auth.uid().
Used for user-specific RLS policies.';

-- =============================================================================
-- Migration Complete
-- =============================================================================
