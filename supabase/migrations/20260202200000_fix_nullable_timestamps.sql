-- Migration: Fix nullable timestamp columns
-- Description: Add NOT NULL constraints with DEFAULT now() to timestamp columns
--              that should never be null. This is a data quality fix.
--
-- Background: Several tables have created_at/updated_at columns that allow NULL,
--             causing "Unknown" to display in the UI when these values are missing.
--
-- Strategy:
--   1. Backfill existing NULL values with current timestamp
--   2. Add NOT NULL constraint with DEFAULT now()
--
-- Rollback: See bottom of file for rollback statements

-- =============================================================================
-- PHASE 1: Backfill existing NULL values
-- =============================================================================

-- Tests table
UPDATE tests
SET created_at = COALESCE(updated_at, now())
WHERE created_at IS NULL;

UPDATE tests
SET updated_at = COALESCE(created_at, now())
WHERE updated_at IS NULL;

-- Test runs table
UPDATE test_runs
SET created_at = COALESCE(started_at, now())
WHERE created_at IS NULL;

-- Chat conversations table
UPDATE chat_conversations
SET created_at = now()
WHERE created_at IS NULL;

UPDATE chat_conversations
SET updated_at = COALESCE(created_at, now())
WHERE updated_at IS NULL;

-- Chat messages table
UPDATE chat_messages
SET created_at = now()
WHERE created_at IS NULL;

-- Projects table
UPDATE projects
SET created_at = now()
WHERE created_at IS NULL;

UPDATE projects
SET updated_at = COALESCE(created_at, now())
WHERE updated_at IS NULL;

-- Organizations table
UPDATE organizations
SET created_at = now()
WHERE created_at IS NULL;

UPDATE organizations
SET updated_at = COALESCE(created_at, now())
WHERE updated_at IS NULL;

-- Teams table
UPDATE teams
SET created_at = now()
WHERE created_at IS NULL;

UPDATE teams
SET updated_at = COALESCE(created_at, now())
WHERE updated_at IS NULL;

-- API keys table
UPDATE api_keys
SET created_at = now()
WHERE created_at IS NULL;

-- Audit logs table
UPDATE audit_logs
SET created_at = now()
WHERE created_at IS NULL;

-- Discovery sessions table
UPDATE discovery_sessions
SET created_at = now()
WHERE created_at IS NULL;

UPDATE discovery_sessions
SET started_at = COALESCE(created_at, now())
WHERE started_at IS NULL;

-- Healing history table
UPDATE healing_history
SET created_at = now()
WHERE created_at IS NULL;

-- Failure patterns table
UPDATE failure_patterns
SET created_at = now()
WHERE created_at IS NULL;

UPDATE failure_patterns
SET updated_at = COALESCE(created_at, now())
WHERE updated_at IS NULL;

-- =============================================================================
-- PHASE 2: Add NOT NULL constraints with defaults
-- =============================================================================

-- Tests table
ALTER TABLE tests
  ALTER COLUMN created_at SET DEFAULT now(),
  ALTER COLUMN created_at SET NOT NULL;

ALTER TABLE tests
  ALTER COLUMN updated_at SET DEFAULT now(),
  ALTER COLUMN updated_at SET NOT NULL;

-- Test runs table
ALTER TABLE test_runs
  ALTER COLUMN created_at SET DEFAULT now(),
  ALTER COLUMN created_at SET NOT NULL;

-- Chat conversations table
ALTER TABLE chat_conversations
  ALTER COLUMN created_at SET DEFAULT now(),
  ALTER COLUMN created_at SET NOT NULL;

ALTER TABLE chat_conversations
  ALTER COLUMN updated_at SET DEFAULT now(),
  ALTER COLUMN updated_at SET NOT NULL;

-- Chat messages table
ALTER TABLE chat_messages
  ALTER COLUMN created_at SET DEFAULT now(),
  ALTER COLUMN created_at SET NOT NULL;

-- Projects table
ALTER TABLE projects
  ALTER COLUMN created_at SET DEFAULT now(),
  ALTER COLUMN created_at SET NOT NULL;

ALTER TABLE projects
  ALTER COLUMN updated_at SET DEFAULT now(),
  ALTER COLUMN updated_at SET NOT NULL;

-- Organizations table
ALTER TABLE organizations
  ALTER COLUMN created_at SET DEFAULT now(),
  ALTER COLUMN created_at SET NOT NULL;

ALTER TABLE organizations
  ALTER COLUMN updated_at SET DEFAULT now(),
  ALTER COLUMN updated_at SET NOT NULL;

-- Teams table
ALTER TABLE teams
  ALTER COLUMN created_at SET DEFAULT now(),
  ALTER COLUMN created_at SET NOT NULL;

ALTER TABLE teams
  ALTER COLUMN updated_at SET DEFAULT now(),
  ALTER COLUMN updated_at SET NOT NULL;

-- API keys table
ALTER TABLE api_keys
  ALTER COLUMN created_at SET DEFAULT now(),
  ALTER COLUMN created_at SET NOT NULL;

-- Audit logs table
ALTER TABLE audit_logs
  ALTER COLUMN created_at SET DEFAULT now(),
  ALTER COLUMN created_at SET NOT NULL;

-- Discovery sessions table
ALTER TABLE discovery_sessions
  ALTER COLUMN created_at SET DEFAULT now(),
  ALTER COLUMN created_at SET NOT NULL;

ALTER TABLE discovery_sessions
  ALTER COLUMN started_at SET DEFAULT now(),
  ALTER COLUMN started_at SET NOT NULL;

-- Healing history table
ALTER TABLE healing_history
  ALTER COLUMN created_at SET DEFAULT now(),
  ALTER COLUMN created_at SET NOT NULL;

-- Failure patterns table
ALTER TABLE failure_patterns
  ALTER COLUMN created_at SET DEFAULT now(),
  ALTER COLUMN created_at SET NOT NULL;

ALTER TABLE failure_patterns
  ALTER COLUMN updated_at SET DEFAULT now(),
  ALTER COLUMN updated_at SET NOT NULL;

-- =============================================================================
-- PHASE 3: Add trigger for automatic updated_at
-- =============================================================================

-- Create or replace trigger function for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add triggers to tables with updated_at columns (if not exists)
DO $$
DECLARE
    tbl TEXT;
    trigger_name TEXT;
BEGIN
    FOR tbl IN SELECT unnest(ARRAY[
        'tests',
        'chat_conversations',
        'projects',
        'organizations',
        'teams',
        'failure_patterns'
    ]) LOOP
        trigger_name := 'trigger_' || tbl || '_updated_at';

        -- Drop if exists and recreate
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %I', trigger_name, tbl);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE UPDATE ON %I
             FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()',
            trigger_name, tbl
        );
    END LOOP;
END $$;

-- =============================================================================
-- ROLLBACK STATEMENTS (run manually if needed)
-- =============================================================================
/*
-- To rollback, run these statements:

ALTER TABLE tests ALTER COLUMN created_at DROP NOT NULL;
ALTER TABLE tests ALTER COLUMN updated_at DROP NOT NULL;
ALTER TABLE test_runs ALTER COLUMN created_at DROP NOT NULL;
ALTER TABLE chat_conversations ALTER COLUMN created_at DROP NOT NULL;
ALTER TABLE chat_conversations ALTER COLUMN updated_at DROP NOT NULL;
ALTER TABLE chat_messages ALTER COLUMN created_at DROP NOT NULL;
ALTER TABLE projects ALTER COLUMN created_at DROP NOT NULL;
ALTER TABLE projects ALTER COLUMN updated_at DROP NOT NULL;
ALTER TABLE organizations ALTER COLUMN created_at DROP NOT NULL;
ALTER TABLE organizations ALTER COLUMN updated_at DROP NOT NULL;
ALTER TABLE teams ALTER COLUMN created_at DROP NOT NULL;
ALTER TABLE teams ALTER COLUMN updated_at DROP NOT NULL;
ALTER TABLE api_keys ALTER COLUMN created_at DROP NOT NULL;
ALTER TABLE audit_logs ALTER COLUMN created_at DROP NOT NULL;
ALTER TABLE discovery_sessions ALTER COLUMN created_at DROP NOT NULL;
ALTER TABLE discovery_sessions ALTER COLUMN started_at DROP NOT NULL;
ALTER TABLE healing_history ALTER COLUMN created_at DROP NOT NULL;
ALTER TABLE failure_patterns ALTER COLUMN created_at DROP NOT NULL;
ALTER TABLE failure_patterns ALTER COLUMN updated_at DROP NOT NULL;

DROP TRIGGER IF EXISTS trigger_tests_updated_at ON tests;
DROP TRIGGER IF EXISTS trigger_chat_conversations_updated_at ON chat_conversations;
DROP TRIGGER IF EXISTS trigger_projects_updated_at ON projects;
DROP TRIGGER IF EXISTS trigger_organizations_updated_at ON organizations;
DROP TRIGGER IF EXISTS trigger_teams_updated_at ON teams;
DROP TRIGGER IF EXISTS trigger_failure_patterns_updated_at ON failure_patterns;
*/

COMMENT ON FUNCTION update_updated_at_column() IS
'Automatically sets updated_at to current timestamp on row update.
Added as part of null safety migration 20260202200000.';
