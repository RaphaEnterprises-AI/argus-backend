-- Fix audit_logs schema for AuditLogMiddleware compatibility
-- Add remaining missing columns and fix constraints

-- Add event_type column
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS event_type TEXT;

-- Add status_code column (different from status which is constrained)
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS status_code INTEGER;

-- Add method and path columns
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS method TEXT;
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS path TEXT;

-- Drop the constraint on action column so we can insert dynamic values like "GET /health"
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'audit_logs_action_check'
    ) THEN
        ALTER TABLE audit_logs DROP CONSTRAINT audit_logs_action_check;
    END IF;
END $$;

-- Drop the constraint on resource_type column to allow 'api' value
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'audit_logs_resource_type_check'
    ) THEN
        ALTER TABLE audit_logs DROP CONSTRAINT audit_logs_resource_type_check;
    END IF;
END $$;

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_audit_logs_path ON audit_logs(path);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_method ON audit_logs(method);
CREATE INDEX IF NOT EXISTS idx_audit_logs_status_code ON audit_logs(status_code);
