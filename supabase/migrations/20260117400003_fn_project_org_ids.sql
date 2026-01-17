-- Migration: Add batch query function for project org_id lookup
-- Used by: Bulk operations that need org_id for audit logging

CREATE OR REPLACE FUNCTION get_project_org_ids(project_ids UUID[])
RETURNS TABLE(project_id UUID, organization_id UUID)
LANGUAGE sql
STABLE
SECURITY DEFINER
AS $$
    SELECT
        p.id as project_id,
        p.organization_id
    FROM projects p
    WHERE p.id = ANY(project_ids)
$$;
