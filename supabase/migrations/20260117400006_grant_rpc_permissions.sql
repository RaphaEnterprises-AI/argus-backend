-- Migration: Grant execute permissions on RPC functions to service role
GRANT EXECUTE ON FUNCTION get_project_test_counts(UUID[]) TO service_role;
