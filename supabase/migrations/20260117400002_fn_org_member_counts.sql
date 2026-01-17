-- Migration: Add batch query function for organization member counts
-- Used by: GET /api/v1/organizations (list_organizations)

CREATE OR REPLACE FUNCTION get_org_member_counts(org_ids UUID[])
RETURNS TABLE(organization_id UUID, count BIGINT)
LANGUAGE sql
STABLE
SECURITY DEFINER
AS $$
    SELECT
        om.organization_id,
        COUNT(*)::BIGINT as count
    FROM organization_members om
    WHERE om.organization_id = ANY(org_ids)
      AND om.status = 'active'
    GROUP BY om.organization_id
$$;
