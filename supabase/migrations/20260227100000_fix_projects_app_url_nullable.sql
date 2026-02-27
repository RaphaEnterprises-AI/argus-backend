-- Fix: projects.app_url should be nullable (matches original schema and API model)
-- The NOT NULL constraint was likely added via dashboard, causing 500s when creating
-- projects without an app_url.
ALTER TABLE projects ALTER COLUMN app_url DROP NOT NULL;
