-- Fix missing columns in discovery_sessions table
-- Migration: 20260122000000_fix_discovery_sessions_columns.sql
-- This migration adds columns that may be missing if the table was created
-- before the 20260111_discovery_intelligence.sql migration

-- Add 'name' column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'discovery_sessions' AND column_name = 'name'
    ) THEN
        ALTER TABLE discovery_sessions ADD COLUMN name TEXT;
        -- Set a default value for existing rows
        UPDATE discovery_sessions SET name = 'Discovery ' || LEFT(id::text, 8) WHERE name IS NULL;
        -- Make it NOT NULL after setting defaults
        ALTER TABLE discovery_sessions ALTER COLUMN name SET NOT NULL;
    END IF;

    -- Also ensure other potentially missing columns
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'discovery_sessions' AND column_name = 'start_url'
    ) THEN
        ALTER TABLE discovery_sessions ADD COLUMN start_url TEXT;
        UPDATE discovery_sessions SET start_url = COALESCE(config->>'app_url', 'unknown') WHERE start_url IS NULL;
        ALTER TABLE discovery_sessions ALTER COLUMN start_url SET NOT NULL;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'discovery_sessions' AND column_name = 'pages_discovered'
    ) THEN
        ALTER TABLE discovery_sessions ADD COLUMN pages_discovered INTEGER DEFAULT 0;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'discovery_sessions' AND column_name = 'started_at'
    ) THEN
        ALTER TABLE discovery_sessions ADD COLUMN started_at TIMESTAMPTZ;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'discovery_sessions' AND column_name = 'completed_at'
    ) THEN
        ALTER TABLE discovery_sessions ADD COLUMN completed_at TIMESTAMPTZ;
    END IF;
END $$;

-- Drop mode constraint if it exists and recreate with all valid values
-- This handles the mismatch between code enum values and DB constraints
DO $$
BEGIN
    -- Check if constraint exists
    IF EXISTS (
        SELECT 1 FROM information_schema.constraint_column_usage
        WHERE table_name = 'discovery_sessions' AND column_name = 'mode'
    ) THEN
        -- The constraint name is typically 'discovery_sessions_mode_check'
        BEGIN
            ALTER TABLE discovery_sessions DROP CONSTRAINT IF EXISTS discovery_sessions_mode_check;
        EXCEPTION
            WHEN undefined_object THEN NULL;
        END;
    END IF;

    -- Add new constraint that includes all valid mode values
    -- Combine database values with code enum values
    ALTER TABLE discovery_sessions
    ADD CONSTRAINT discovery_sessions_mode_check
    CHECK (mode IN (
        'full', 'incremental', 'focused', 'quick', 'deep',
        'standard', 'standard_crawl', 'quick_scan', 'deep_analysis',
        'authenticated', 'api_first', 'autonomous'
    ));
EXCEPTION
    WHEN duplicate_object THEN NULL;
    WHEN check_violation THEN
        RAISE NOTICE 'Some rows have invalid mode values, skipping constraint';
END $$;

-- Same for strategy constraint
DO $$
BEGIN
    BEGIN
        ALTER TABLE discovery_sessions DROP CONSTRAINT IF EXISTS discovery_sessions_strategy_check;
    EXCEPTION
        WHEN undefined_object THEN NULL;
    END;

    ALTER TABLE discovery_sessions
    ADD CONSTRAINT discovery_sessions_strategy_check
    CHECK (strategy IN (
        'breadth_first', 'depth_first', 'priority_based', 'ai_guided',
        'bfs', 'dfs', 'priority', 'smart_adaptive'
    ));
EXCEPTION
    WHEN duplicate_object THEN NULL;
    WHEN check_violation THEN
        RAISE NOTICE 'Some rows have invalid strategy values, skipping constraint';
END $$;
