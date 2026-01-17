-- Migration: Add atomic invitation acceptance function
-- Used by: POST /api/v1/invitations/{token}/accept

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
