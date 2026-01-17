-- Migration: Add batch query functions to fix N+1 query problems
-- These functions allow fetching counts for multiple entities in a single query

-- =============================================================================
-- Function: Get test counts for multiple projects
-- Used by: GET /api/v1/projects (list_organization_projects, list_projects)
-- =============================================================================

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

COMMENT ON FUNCTION get_project_test_counts IS 'Get active test counts for multiple projects in a single query';

-- =============================================================================
-- Function: Get member counts for multiple organizations
-- Used by: GET /api/v1/organizations (list_organizations)
-- =============================================================================

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

COMMENT ON FUNCTION get_org_member_counts IS 'Get active member counts for multiple organizations in a single query';

-- =============================================================================
-- Function: Get organization IDs for multiple projects
-- Used by: Bulk operations that need org_id for audit logging
-- =============================================================================

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

COMMENT ON FUNCTION get_project_org_ids IS 'Get organization IDs for multiple projects in a single query';

-- =============================================================================
-- Function: Accept invitation atomically (transaction)
-- Used by: POST /api/v1/invitations/{id}/accept
-- =============================================================================

CREATE OR REPLACE FUNCTION accept_invitation_atomic(
    p_invitation_id UUID,
    p_user_id TEXT,
    p_user_email TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_invitation RECORD;
    v_member_id UUID;
    v_existing_member UUID;
BEGIN
    -- Lock and fetch the invitation
    SELECT * INTO v_invitation
    FROM invitations
    WHERE id = p_invitation_id
      AND status = 'pending'
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN jsonb_build_object(
            'success', false,
            'error', 'Invitation not found or already used'
        );
    END IF;

    -- Check if user already a member
    SELECT id INTO v_existing_member
    FROM organization_members
    WHERE organization_id = v_invitation.organization_id
      AND user_id = p_user_id
      AND status = 'active';

    IF FOUND THEN
        RETURN jsonb_build_object(
            'success', false,
            'error', 'User is already a member of this organization'
        );
    END IF;

    -- Create the membership
    INSERT INTO organization_members (
        organization_id,
        user_id,
        email,
        role,
        status,
        joined_at,
        created_at,
        updated_at
    )
    VALUES (
        v_invitation.organization_id,
        p_user_id,
        COALESCE(p_user_email, v_invitation.email),
        v_invitation.role,
        'active',
        NOW(),
        NOW(),
        NOW()
    )
    RETURNING id INTO v_member_id;

    -- Update invitation status
    UPDATE invitations
    SET
        status = 'accepted',
        accepted_by = p_user_id,
        accepted_at = NOW(),
        updated_at = NOW()
    WHERE id = p_invitation_id;

    RETURN jsonb_build_object(
        'success', true,
        'member_id', v_member_id,
        'organization_id', v_invitation.organization_id,
        'role', v_invitation.role
    );

EXCEPTION WHEN OTHERS THEN
    RETURN jsonb_build_object(
        'success', false,
        'error', SQLERRM
    );
END;
$$;

COMMENT ON FUNCTION accept_invitation_atomic IS 'Atomically accept an invitation and create membership in a single transaction';

-- =============================================================================
-- Function: Create organization atomically with owner membership
-- Used by: POST /api/v1/organizations
-- =============================================================================

CREATE OR REPLACE FUNCTION create_organization_atomic(
    p_name TEXT,
    p_slug TEXT,
    p_plan TEXT,
    p_user_id TEXT,
    p_user_email TEXT,
    p_logo_url TEXT DEFAULT NULL,
    p_settings JSONB DEFAULT '{}'::JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_org_id UUID;
    v_member_id UUID;
    v_final_slug TEXT;
    v_counter INT := 0;
BEGIN
    -- Generate unique slug
    v_final_slug := p_slug;
    WHILE EXISTS (SELECT 1 FROM organizations WHERE slug = v_final_slug) LOOP
        v_counter := v_counter + 1;
        v_final_slug := p_slug || '-' || v_counter;
        IF v_counter > 100 THEN
            RETURN jsonb_build_object(
                'success', false,
                'error', 'Could not generate unique slug'
            );
        END IF;
    END LOOP;

    -- Create organization
    INSERT INTO organizations (
        name,
        slug,
        plan,
        logo_url,
        settings,
        created_at,
        updated_at
    )
    VALUES (
        p_name,
        v_final_slug,
        COALESCE(p_plan, 'free'),
        p_logo_url,
        p_settings,
        NOW(),
        NOW()
    )
    RETURNING id INTO v_org_id;

    -- Create owner membership
    INSERT INTO organization_members (
        organization_id,
        user_id,
        email,
        role,
        status,
        joined_at,
        created_at,
        updated_at
    )
    VALUES (
        v_org_id,
        p_user_id,
        p_user_email,
        'owner',
        'active',
        NOW(),
        NOW(),
        NOW()
    )
    RETURNING id INTO v_member_id;

    RETURN jsonb_build_object(
        'success', true,
        'organization_id', v_org_id,
        'slug', v_final_slug,
        'member_id', v_member_id
    );

EXCEPTION WHEN OTHERS THEN
    RETURN jsonb_build_object(
        'success', false,
        'error', SQLERRM
    );
END;
$$;

COMMENT ON FUNCTION create_organization_atomic IS 'Atomically create organization and owner membership in a single transaction';

-- =============================================================================
-- Grant execute permissions to service role
-- =============================================================================

GRANT EXECUTE ON FUNCTION get_project_test_counts(UUID[]) TO service_role;
GRANT EXECUTE ON FUNCTION get_org_member_counts(UUID[]) TO service_role;
GRANT EXECUTE ON FUNCTION get_project_org_ids(UUID[]) TO service_role;
GRANT EXECUTE ON FUNCTION accept_invitation_atomic(UUID, TEXT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION create_organization_atomic(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB) TO service_role;
