-- Migration: Add batch query function for project test counts
-- Used by: GET /api/v1/projects (list_organization_projects, list_projects)

CREATE OR REPLACE FUNCTION get_project_test_counts(project_ids UUID[])
RETURNS TABLE(project_id UUID, count BIGINT)
LANGUAGE sql
STABLE
SECURITY DEFINER
AS $$
    SELECT
        t.project_id,
        COUNT(*)::BIGINT as count
    FROM tests t
    WHERE t.project_id = ANY(project_ids)
      AND t.is_active = true
    GROUP BY t.project_id
$$;
