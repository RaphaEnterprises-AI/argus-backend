-- Create recordings table to persist browser recording data.
-- Previously recordings were stored in-memory and lost on server restart.

CREATE TABLE IF NOT EXISTS recordings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL DEFAULT 'Untitled Recording',
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    user_id UUID NOT NULL,
    organization_id UUID,
    events JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    events_count INTEGER NOT NULL DEFAULT 0,
    interaction_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index for listing recordings by org/project
CREATE INDEX IF NOT EXISTS idx_recordings_org_id ON recordings(organization_id);
CREATE INDEX IF NOT EXISTS idx_recordings_project_id ON recordings(project_id);
CREATE INDEX IF NOT EXISTS idx_recordings_user_id ON recordings(user_id);
CREATE INDEX IF NOT EXISTS idx_recordings_created_at ON recordings(created_at DESC);

-- Enable RLS
ALTER TABLE recordings ENABLE ROW LEVEL SECURITY;

-- Policy: users can read recordings from their organization
CREATE POLICY recordings_select_policy ON recordings
    FOR SELECT USING (
        organization_id = current_setting('request.jwt.claims', true)::json->>'organization_id'
        OR user_id::text = current_setting('request.jwt.claims', true)::json->>'sub'
    );

-- Policy: users can insert recordings
CREATE POLICY recordings_insert_policy ON recordings
    FOR INSERT WITH CHECK (true);

-- Policy: users can delete their own recordings
CREATE POLICY recordings_delete_policy ON recordings
    FOR DELETE USING (
        user_id::text = current_setting('request.jwt.claims', true)::json->>'sub'
    );

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_recordings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER recordings_updated_at
    BEFORE UPDATE ON recordings
    FOR EACH ROW
    EXECUTE FUNCTION update_recordings_updated_at();
