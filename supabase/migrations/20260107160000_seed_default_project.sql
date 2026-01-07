-- Seed a default project for testing
-- This ensures webhooks have a project to associate events with

-- First, ensure the projects table exists with required columns
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    repository_url TEXT,
    app_url TEXT,
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name);
CREATE INDEX IF NOT EXISTS idx_projects_org ON projects(organization_id);

-- Enable RLS
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;

-- Allow all access for now (can be tightened later)
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON projects;
CREATE POLICY "Enable all access for authenticated users" ON projects FOR ALL USING (true);

-- Insert default project (upsert to ensure it exists)
INSERT INTO projects (id, name, description, app_url)
VALUES (
    '00000000-0000-0000-0000-000000000001'::uuid,
    'Default Project',
    'Auto-created default project for webhook ingestion',
    'https://example.com'
)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    updated_at = now();
