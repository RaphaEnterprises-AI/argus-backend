-- Artifacts table for storing metadata about screenshots, videos, etc.
-- Actual binary data is stored in Cloudflare R2, this table holds references.

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    user_id TEXT NOT NULL,

    -- Artifact metadata
    type TEXT NOT NULL CHECK (type IN ('screenshot', 'video', 'html', 'json', 'log')),
    storage_backend TEXT NOT NULL DEFAULT 'r2' CHECK (storage_backend IN ('r2', 's3', 'supabase', 'memory')),
    storage_key TEXT, -- R2/S3 object key
    storage_url TEXT, -- Public or presigned URL if available

    -- Context
    test_id TEXT,
    test_run_id UUID,
    thread_id TEXT,
    step_index INTEGER,
    action_description TEXT,

    -- Metadata
    file_size_bytes INTEGER,
    content_type TEXT DEFAULT 'image/png',
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ, -- For auto-cleanup

    -- Indexes for common queries
    CONSTRAINT valid_storage CHECK (storage_key IS NOT NULL OR storage_url IS NOT NULL)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_artifacts_organization ON artifacts(organization_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_test_run ON artifacts(test_run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_thread ON artifacts(thread_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(type);

-- Enable RLS
ALTER TABLE artifacts ENABLE ROW LEVEL SECURITY;

-- RLS policies
CREATE POLICY "Users can view artifacts from their organization"
ON artifacts FOR SELECT
USING (
    organization_id IN (
        SELECT organization_id FROM organization_members
        WHERE user_id = auth.uid()::text
    )
);

CREATE POLICY "Users can insert artifacts to their organization"
ON artifacts FOR INSERT
WITH CHECK (
    organization_id IN (
        SELECT organization_id FROM organization_members
        WHERE user_id = auth.uid()::text
    )
);

CREATE POLICY "Service role has full access to artifacts"
ON artifacts FOR ALL
USING (auth.role() = 'service_role');

-- Comment
COMMENT ON TABLE artifacts IS 'Stores metadata for test artifacts (screenshots, videos). Binary data stored in R2.';
