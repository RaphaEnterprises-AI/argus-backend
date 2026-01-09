-- Add missing columns to audit_logs table for AuditLogMiddleware compatibility
-- The middleware inserts columns that don't exist in the original schema

-- Add duration_ms column
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS duration_ms INTEGER;

-- Add event_type column
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS event_type TEXT;

-- Add status_code column (different from status which is constrained)
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS status_code INTEGER;

-- Add method and path columns
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS method TEXT;
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS path TEXT;

-- Drop the constraint on action column so we can insert dynamic values like "GET /health"
-- First check if constraint exists, then drop it
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'audit_logs_action_check'
    ) THEN
        ALTER TABLE audit_logs DROP CONSTRAINT audit_logs_action_check;
    END IF;
END $$;

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_audit_logs_duration ON audit_logs(duration_ms);
CREATE INDEX IF NOT EXISTS idx_audit_logs_path ON audit_logs(path);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_method ON audit_logs(method);
