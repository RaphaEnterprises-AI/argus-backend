-- Project-level access control
CREATE TABLE IF NOT EXISTS project_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    organization_member_id UUID NOT NULL REFERENCES organization_members(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'developer' CHECK (role IN ('maintainer', 'developer', 'viewer')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(project_id, organization_member_id)
);

CREATE INDEX idx_project_members_project ON project_members(project_id);
CREATE INDEX idx_project_members_member ON project_members(organization_member_id);

-- RLS
ALTER TABLE project_members ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view project members" ON project_members
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM organization_members om
            JOIN projects p ON p.organization_id = om.organization_id
            WHERE p.id = project_members.project_id
            AND om.user_id = current_setting('app.user_id', true)
        )
    );

CREATE POLICY "Service role full access to project_members" ON project_members
    FOR ALL USING (current_setting('role', true) = 'service_role');
