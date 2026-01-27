-- Deployment Events and SDLC Events tables migration
-- Adds tables for tracking deployment events (Vercel, Netlify, etc.) and SDLC events (Jira, Linear, etc.)

-- Deployment Events table (Vercel, Netlify, Cloudflare Pages, AWS Amplify)
CREATE TABLE IF NOT EXISTS deployment_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK (source IN ('vercel', 'netlify', 'cloudflare', 'aws_amplify')),
    external_id TEXT NOT NULL,
    external_url TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'building', 'ready', 'error', 'canceled')),
    deployment_url TEXT,
    preview_url TEXT,
    branch TEXT,
    commit_sha TEXT,
    commit_message TEXT,
    environment TEXT NOT NULL DEFAULT 'preview' CHECK (environment IN ('production', 'preview', 'development')),
    duration_seconds INTEGER,
    raw_payload JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- SDLC Events table (Jira, Linear, GitHub Issues, etc.)
CREATE TABLE IF NOT EXISTS sdlc_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_platform TEXT NOT NULL CHECK (source_platform IN ('jira', 'linear', 'github', 'asana', 'notion')),
    event_type TEXT NOT NULL CHECK (event_type IN ('bug', 'story', 'epic', 'task', 'feature', 'improvement')),
    external_id TEXT NOT NULL,
    external_key TEXT, -- e.g., PROJ-123 for Jira
    external_url TEXT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT,
    priority TEXT,
    assignee TEXT,
    reporter TEXT,
    labels TEXT[],
    raw_payload JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for deployment_events
CREATE INDEX IF NOT EXISTS idx_deployment_events_project ON deployment_events(project_id);
CREATE INDEX IF NOT EXISTS idx_deployment_events_source ON deployment_events(source);
CREATE INDEX IF NOT EXISTS idx_deployment_events_status ON deployment_events(status);
CREATE INDEX IF NOT EXISTS idx_deployment_events_branch ON deployment_events(branch);
CREATE INDEX IF NOT EXISTS idx_deployment_events_external_id ON deployment_events(external_id);
CREATE INDEX IF NOT EXISTS idx_deployment_events_environment ON deployment_events(environment);
CREATE INDEX IF NOT EXISTS idx_deployment_events_created ON deployment_events(created_at DESC);

-- Add missing columns if they don't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sdlc_events' AND column_name = 'external_key') THEN
        ALTER TABLE sdlc_events ADD COLUMN external_key TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sdlc_events' AND column_name = 'status') THEN
        ALTER TABLE sdlc_events ADD COLUMN status TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sdlc_events' AND column_name = 'event_type') THEN
        ALTER TABLE sdlc_events ADD COLUMN event_type TEXT;
    END IF;
END $$;

-- Indexes for sdlc_events (only create if columns exist)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sdlc_events' AND column_name = 'project_id') THEN
        CREATE INDEX IF NOT EXISTS idx_sdlc_events_project ON sdlc_events(project_id);
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sdlc_events' AND column_name = 'source_platform') THEN
        CREATE INDEX IF NOT EXISTS idx_sdlc_events_source ON sdlc_events(source_platform);
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sdlc_events' AND column_name = 'event_type') THEN
        CREATE INDEX IF NOT EXISTS idx_sdlc_events_type ON sdlc_events(event_type);
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sdlc_events' AND column_name = 'external_id') THEN
        CREATE INDEX IF NOT EXISTS idx_sdlc_events_external_id ON sdlc_events(external_id);
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sdlc_events' AND column_name = 'external_key') THEN
        CREATE INDEX IF NOT EXISTS idx_sdlc_events_external_key ON sdlc_events(external_key);
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sdlc_events' AND column_name = 'status') THEN
        CREATE INDEX IF NOT EXISTS idx_sdlc_events_status ON sdlc_events(status);
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sdlc_events' AND column_name = 'created_at') THEN
        CREATE INDEX IF NOT EXISTS idx_sdlc_events_created ON sdlc_events(created_at DESC);
    END IF;
END $$;

-- Enable RLS
ALTER TABLE deployment_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE sdlc_events ENABLE ROW LEVEL SECURITY;

-- RLS Policies (allow all for authenticated users for now)
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON deployment_events;
CREATE POLICY "Enable all access for authenticated users" ON deployment_events FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON sdlc_events;
CREATE POLICY "Enable all access for authenticated users" ON sdlc_events FOR ALL USING (true);

-- Grant permissions to service role
GRANT ALL ON deployment_events TO service_role;
GRANT ALL ON sdlc_events TO service_role;

-- Grant permissions to authenticated users
GRANT ALL ON deployment_events TO authenticated;
GRANT ALL ON sdlc_events TO authenticated;

-- Comment on tables
COMMENT ON TABLE deployment_events IS 'Tracks deployment events from platforms like Vercel, Netlify, etc.';
COMMENT ON TABLE sdlc_events IS 'Tracks SDLC events from issue trackers like Jira, Linear, etc.';

-- Comment on columns
COMMENT ON COLUMN deployment_events.preview_url IS 'URL for preview deployments, used for automated testing';
COMMENT ON COLUMN sdlc_events.external_key IS 'Human-readable key like PROJ-123 for Jira issues';
