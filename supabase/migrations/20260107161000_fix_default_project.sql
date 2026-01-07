-- Fix: ensure default project exists
-- The previous migration may have failed to insert due to WHERE NOT EXISTS logic

-- Make user_id and slug nullable for system projects
ALTER TABLE projects ALTER COLUMN user_id DROP NOT NULL;
ALTER TABLE projects ALTER COLUMN slug DROP NOT NULL;

-- Now insert the default project with slug
INSERT INTO projects (id, name, slug, description, app_url)
VALUES (
    '00000000-0000-0000-0000-000000000001'::uuid,
    'Default Project',
    'default-project',
    'Auto-created default project for webhook ingestion',
    'https://example.com'
)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    slug = EXCLUDED.slug,
    updated_at = now();
