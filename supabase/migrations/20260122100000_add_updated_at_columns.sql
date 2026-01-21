-- Add updated_at columns to discovery tables
-- Migration: 20260122100000_add_updated_at_columns.sql
--
-- The 20260111_discovery_intelligence.sql migration created triggers that expect
-- updated_at columns, but these columns were never added to the tables.
-- This migration adds the missing columns.

-- Add updated_at to discovery_sessions
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'discovery_sessions' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE discovery_sessions ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
        -- Set existing rows to their created_at value or started_at if available
        UPDATE discovery_sessions
        SET updated_at = COALESCE(started_at, created_at, NOW())
        WHERE updated_at IS NULL;
    END IF;
END $$;

-- Add updated_at to discovered_pages
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'discovered_pages' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE discovered_pages ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
        UPDATE discovered_pages
        SET updated_at = COALESCE(created_at, NOW())
        WHERE updated_at IS NULL;
    END IF;
END $$;

-- Add updated_at to discovered_flows
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'discovered_flows' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE discovered_flows ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
        UPDATE discovered_flows
        SET updated_at = COALESCE(created_at, NOW())
        WHERE updated_at IS NULL;
    END IF;
END $$;

-- Add updated_at to discovered_elements (if table exists)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'discovered_elements'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'discovered_elements' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE discovered_elements ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
        UPDATE discovered_elements
        SET updated_at = COALESCE(created_at, NOW())
        WHERE updated_at IS NULL;
    END IF;
END $$;

-- Add updated_at to discovery_patterns (if table exists)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'discovery_patterns'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'discovery_patterns' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE discovery_patterns ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
        UPDATE discovery_patterns
        SET updated_at = COALESCE(created_at, NOW())
        WHERE updated_at IS NULL;
    END IF;
END $$;
