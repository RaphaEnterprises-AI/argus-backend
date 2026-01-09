-- Invitations system
CREATE TABLE IF NOT EXISTS invitations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('admin', 'member', 'viewer')),
    token TEXT NOT NULL UNIQUE,
    token_expires_at TIMESTAMPTZ NOT NULL,
    message TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'expired', 'revoked')),
    invited_by UUID REFERENCES organization_members(id),
    accepted_by UUID REFERENCES organization_members(id),
    accepted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_invitations_org ON invitations(organization_id);
CREATE INDEX idx_invitations_email ON invitations(email);
CREATE INDEX idx_invitations_token ON invitations(token);
CREATE INDEX idx_invitations_status ON invitations(status);

-- RLS
ALTER TABLE invitations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org admins can manage invitations" ON invitations
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM organization_members
            WHERE organization_members.organization_id = invitations.organization_id
            AND organization_members.user_id = current_setting('app.user_id', true)
            AND organization_members.role IN ('owner', 'admin')
        )
    );

CREATE POLICY "Service role full access to invitations" ON invitations
    FOR ALL USING (current_setting('role', true) = 'service_role');
