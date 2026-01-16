-- Migration: Add missing foreign key constraints for data integrity
-- Fixes orphaned record issues by ensuring referential integrity

-- =============================================================================
-- visual_baselines: Add FK to tests table
-- =============================================================================

-- Check if tests table exists and has expected structure
DO $$
BEGIN
    -- Add FK constraint if tests table exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tests') THEN
        -- Drop existing constraint if it exists
        ALTER TABLE visual_baselines
        DROP CONSTRAINT IF EXISTS fk_visual_baselines_test_id;
        
        -- Add new FK constraint with CASCADE on delete (when test deleted, baselines deleted)
        ALTER TABLE visual_baselines
        ADD CONSTRAINT fk_visual_baselines_test_id
        FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE;
        
        RAISE NOTICE 'Added FK constraint fk_visual_baselines_test_id';
    ELSE
        RAISE NOTICE 'tests table does not exist, skipping visual_baselines FK';
    END IF;
EXCEPTION
    WHEN undefined_column THEN
        RAISE NOTICE 'visual_baselines.test_id column does not exist, skipping FK';
    WHEN undefined_table THEN
        RAISE NOTICE 'visual_baselines table does not exist, skipping FK';
END;
$$;

-- =============================================================================
-- visual_snapshots: Add FK to test_runs table
-- =============================================================================

DO $$
BEGIN
    -- Add FK constraint if test_runs table exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'test_runs') THEN
        ALTER TABLE visual_snapshots
        DROP CONSTRAINT IF EXISTS fk_visual_snapshots_test_run_id;
        
        ALTER TABLE visual_snapshots
        ADD CONSTRAINT fk_visual_snapshots_test_run_id
        FOREIGN KEY (test_run_id) REFERENCES test_runs(id) ON DELETE CASCADE;
        
        RAISE NOTICE 'Added FK constraint fk_visual_snapshots_test_run_id';
    ELSE
        RAISE NOTICE 'test_runs table does not exist, skipping visual_snapshots FK';
    END IF;
EXCEPTION
    WHEN undefined_column THEN
        RAISE NOTICE 'visual_snapshots.test_run_id column does not exist, skipping FK';
    WHEN undefined_table THEN
        RAISE NOTICE 'visual_snapshots table does not exist, skipping FK';
END;
$$;

-- =============================================================================
-- test_failure_patterns: Add FK to tests table (optional - may have NULL test_id)
-- =============================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tests') 
       AND EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'test_failure_patterns') THEN
        
        ALTER TABLE test_failure_patterns
        DROP CONSTRAINT IF EXISTS fk_test_failure_patterns_test_id;
        
        -- Allow NULL for patterns not tied to a specific test
        -- Use SET NULL on delete so patterns are preserved even if test is deleted
        ALTER TABLE test_failure_patterns
        ADD CONSTRAINT fk_test_failure_patterns_test_id
        FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE SET NULL;
        
        RAISE NOTICE 'Added FK constraint fk_test_failure_patterns_test_id';
    ELSE
        RAISE NOTICE 'Required tables do not exist, skipping test_failure_patterns FK';
    END IF;
EXCEPTION
    WHEN undefined_column THEN
        RAISE NOTICE 'test_failure_patterns.test_id column does not exist, skipping FK';
    WHEN undefined_table THEN
        RAISE NOTICE 'Required table does not exist, skipping FK';
END;
$$;

-- =============================================================================
-- test_failure_patterns: Add FK to projects table
-- =============================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'projects') 
       AND EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'test_failure_patterns') THEN
        
        ALTER TABLE test_failure_patterns
        DROP CONSTRAINT IF EXISTS fk_test_failure_patterns_project_id;
        
        ALTER TABLE test_failure_patterns
        ADD CONSTRAINT fk_test_failure_patterns_project_id
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE;
        
        RAISE NOTICE 'Added FK constraint fk_test_failure_patterns_project_id';
    ELSE
        RAISE NOTICE 'Required tables do not exist, skipping test_failure_patterns project FK';
    END IF;
EXCEPTION
    WHEN undefined_column THEN
        RAISE NOTICE 'test_failure_patterns.project_id column does not exist, skipping FK';
    WHEN undefined_table THEN
        RAISE NOTICE 'Required table does not exist, skipping FK';
END;
$$;

-- =============================================================================
-- Create indexes for FK columns if they don't exist (performance)
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_visual_baselines_test_id ON visual_baselines(test_id);
CREATE INDEX IF NOT EXISTS idx_visual_snapshots_test_run_id ON visual_snapshots(test_run_id);
CREATE INDEX IF NOT EXISTS idx_test_failure_patterns_test_id ON test_failure_patterns(test_id);
CREATE INDEX IF NOT EXISTS idx_test_failure_patterns_project_id ON test_failure_patterns(project_id);
