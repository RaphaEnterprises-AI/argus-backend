-- Migration: Add atomic organization creation function
-- Used by: POST /api/v1/organizations

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
