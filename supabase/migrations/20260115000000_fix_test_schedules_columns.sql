-- Fix missing columns in test_schedules table
-- These columns are expected by the API but were missing from the original migration

-- Add tags column for schedule categorization
ALTER TABLE test_schedules
    ADD COLUMN IF NOT EXISTS tags TEXT[] DEFAULT '{}';

-- Add environment_variables column for runtime configuration
ALTER TABLE test_schedules
    ADD COLUMN IF NOT EXISTS environment_variables JSONB DEFAULT '{}';

-- Create index for tag-based filtering
CREATE INDEX IF NOT EXISTS idx_test_schedules_tags ON test_schedules USING GIN(tags);

COMMENT ON COLUMN test_schedules.tags IS 'Array of tags for categorizing and filtering schedules';
COMMENT ON COLUMN test_schedules.environment_variables IS 'Environment variables to use during test execution';
