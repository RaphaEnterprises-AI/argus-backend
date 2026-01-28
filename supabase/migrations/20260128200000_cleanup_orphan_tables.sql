-- ============================================================================
-- RAP-175: Clean Up Orphan Database Tables
-- Date: 2026-01-28
-- ============================================================================
-- This migration documents and cleans up orphan tables that exist in the
-- database schema but have no code references in the application.
--
-- METHODOLOGY:
-- 1. Searched all migrations in supabase/migrations/ to identify tables
-- 2. Searched src/ and dashboard/ directories for code references
-- 3. Tables with ZERO code references are candidates for cleanup
-- 4. Tables are categorized as: DROP (truly orphaned) or KEEP (future use)
-- ============================================================================

-- ============================================================================
-- TABLES BEING DROPPED (Zero code references found)
-- ============================================================================

-- -----------------------------------------------------------------------------
-- 1. rate_limit_entries
-- Created in: 20260109100000_security_audit.sql
-- Purpose: Was intended for persistent database-backed rate limiting
-- Reason for drop: Application uses in-memory rate limiting instead
--                  (see src/api/security/middleware.py, src/guardrails/stack.py)
--                  The cleanup_rate_limit_entries() function exists but is never called
-- Search performed: grep -r "rate_limit_entries" src/ dashboard/ - no matches
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS rate_limit_entries CASCADE;

-- Also drop the associated cleanup function since it's no longer needed
DROP FUNCTION IF EXISTS cleanup_rate_limit_entries();

-- -----------------------------------------------------------------------------
-- 2. file_dependencies
-- Created in: 20260125100000_test_impact_graph.sql
-- Purpose: Track import relationships between source files for impact analysis
-- Reason for drop: No code in src/ or dashboard/ references this table
--                  The test impact graph feature uses test_impact_graph and
--                  failure_correlations tables but file_dependencies is unused
-- Search performed: grep -r "file_dependencies" src/ dashboard/ - no matches
-- Note: get_transitive_deps() function uses this table but is also unused
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS file_dependencies CASCADE;

-- Drop the unused function that depends on file_dependencies
DROP FUNCTION IF EXISTS get_transitive_deps(UUID, TEXT[], INTEGER);

-- -----------------------------------------------------------------------------
-- 3. plugin_metrics
-- Created in: 20260125000003_plugin_events.sql
-- Purpose: Pre-aggregated daily/hourly metrics for plugin usage analytics
-- Reason for drop: Dashboard only uses plugin_events and plugin_sessions tables
--                  (see dashboard/lib/hooks/use-plugin-events.ts)
--                  No code aggregates or queries plugin_metrics
-- Search performed: grep -r "plugin_metrics" src/ dashboard/ - no matches
--                   (only in generated types and migration file itself)
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS plugin_metrics CASCADE;


-- ============================================================================
-- TABLES BEING KEPT (Used or planned for future use)
-- ============================================================================
-- The following tables were reviewed and found to have code references:
--
-- SECURITY/AUDIT (20260109100000_security_audit.sql):
-- - security_audit_logs: Used in src/api/security/audit.py, src/security/audit.py
-- - revoked_tokens: Used in src/api/security/auth.py
-- - user_sessions: Used in src/core/cognitive_engine.py
-- - security_alerts: Used in src/api/security/audit.py, src/security/audit.py
-- - permission_changes: Used in src/api/security/audit.py
-- - data_access_logs: Used in src/api/security/middleware.py, src/security/audit.py
--
-- GRAPH/KNOWLEDGE (20260126000000_apache_age_graph.sql):
-- - graph_metadata: Used in src/knowledge_graph/graph_store.py
-- - graph_test_vertices: Used in src/knowledge_graph/graph_store.py
-- - graph_selector_vertices: Used in src/knowledge_graph/graph_store.py
-- - graph_failure_vertices: Used in src/knowledge_graph/graph_store.py
-- - graph_code_change_vertices: Used in src/knowledge_graph/graph_store.py
-- - graph_healing_pattern_vertices: Used in src/knowledge_graph/graph_store.py
-- - graph_page_vertices: Used in src/knowledge_graph/graph_store.py
-- - graph_project_vertices: Used in src/knowledge_graph/graph_store.py
-- - graph_edges: Used in src/knowledge_graph/graph_store.py
--
-- NOTIFICATIONS (20260108000001_notifications.sql):
-- - notification_channels: Used in src/api/notifications.py, src/api/scheduling.py
-- - notification_rules: Used in src/api/notifications.py
-- - notification_logs: Used in src/api/notifications.py
-- - notification_templates: Used in src/api/notifications.py
--
-- INFRA (20260116200000_infra_recommendations.sql):
-- - infra_recommendations: Used in src/services/infra_optimizer.py
-- - infra_cost_history: Used in src/services/infra_optimizer.py
-- - infra_anomaly_history: Used in src/services/infra_optimizer.py
--
-- DISCOVERY (20260111_discovery_intelligence.sql):
-- - discovery_sessions: Used in src/api/discovery.py
-- - discovered_pages: Used in src/api/discovery.py
-- - discovered_elements: Used in src/api/discovery.py
-- - discovered_flows: Used in src/api/discovery.py
-- - discovery_patterns: Used in src/discovery/pattern_service.py
-- - discovery_history: Used in src/api/discovery.py
--
-- DASHBOARD-SPECIFIC (used by Next.js dashboard):
-- - activity_logs: Used in dashboard/lib/hooks/use-activity.ts
-- - live_sessions: Used in dashboard/lib/hooks/use-live-session.ts
-- - chat_conversations: Used in dashboard/lib/hooks/use-chat.ts
-- - chat_messages: Used in dashboard/lib/hooks/use-chat.ts
-- - plugin_events: Used in dashboard/lib/hooks/use-plugin-events.ts
-- - plugin_sessions: Used in dashboard/lib/hooks/use-plugin-events.ts
--
-- TEST IMPACT GRAPH (20260125100000_test_impact_graph.sql):
-- - test_impact_graph: Used in src/api/impact_graph.py
-- - coverage_imports: Used in src/api/impact_graph.py, src/mcp/quality_mcp.py
-- - impact_graph_jobs: Used in src/api/impact_graph.py
-- - failure_correlations: Used in src/services/correlation_engine.py
--
-- All other tables reviewed have code references and are actively used.
-- ============================================================================


-- ============================================================================
-- VERIFICATION NOTES
-- ============================================================================
-- After running this migration:
-- 1. Verify with: SELECT tablename FROM pg_tables WHERE schemaname = 'public'
--                 AND tablename IN ('rate_limit_entries', 'file_dependencies', 'plugin_metrics');
--    Expected: Empty result set
--
-- 2. If any issues arise, these tables can be recreated from:
--    - rate_limit_entries: 20260109100000_security_audit.sql
--    - file_dependencies: 20260125100000_test_impact_graph.sql
--    - plugin_metrics: 20260125000003_plugin_events.sql
-- ============================================================================

COMMENT ON SCHEMA public IS 'Orphan table cleanup performed on 2026-01-28 (RAP-175). Dropped: rate_limit_entries, file_dependencies, plugin_metrics';
