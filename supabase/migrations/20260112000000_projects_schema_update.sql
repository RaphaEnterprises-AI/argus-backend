-- Projects table schema update
-- Adds missing columns: codebase_path, repository_url, is_active, last_run_at

-- Add codebase_path column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'projects' AND column_name = 'codebase_path') THEN
        ALTER TABLE projects ADD COLUMN codebase_path TEXT;
    END IF;
END $$;

-- Add repository_url column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'projects' AND column_name = 'repository_url') THEN
        ALTER TABLE projects ADD COLUMN repository_url TEXT;
    END IF;
END $$;

-- Add is_active column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'projects' AND column_name = 'is_active') THEN
        ALTER TABLE projects ADD COLUMN is_active BOOLEAN DEFAULT true;
    END IF;
END $$;

-- Add last_run_at column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'projects' AND column_name = 'last_run_at') THEN
        ALTER TABLE projects ADD COLUMN last_run_at TIMESTAMPTZ;
    END IF;
END $$;

-- Create index for active projects
CREATE INDEX IF NOT EXISTS idx_projects_is_active ON projects(is_active);

-- Create index for last_run_at for sorting
CREATE INDEX IF NOT EXISTS idx_projects_last_run ON projects(last_run_at DESC NULLS LAST);
