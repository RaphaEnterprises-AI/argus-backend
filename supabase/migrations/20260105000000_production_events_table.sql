-- Production Events Table
-- Stores normalized events from observability platforms (Sentry, Datadog, etc.)
-- This table MUST be created before other tables that reference it (e.g., test_generation_jobs)

-- =============================================================================
-- Create production_events table
-- =============================================================================

CREATE TABLE IF NOT EXISTS production_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Source identification
    source TEXT NOT NULL CHECK (source IN ('sentry', 'datadog', 'fullstory', 'logrocket', 'newrelic', 'bugsnag', 'rollbar')),
    external_id TEXT NOT NULL,
    external_url TEXT,

    -- Event classification
    event_type TEXT NOT NULL CHECK (event_type IN ('error', 'exception', 'performance', 'session', 'rage_click', 'dead_click')),
    severity TEXT NOT NULL CHECK (severity IN ('fatal', 'error', 'warning', 'info')),

    -- Event details
    title TEXT NOT NULL,
    message TEXT,
    stack_trace TEXT,
    fingerprint TEXT NOT NULL,
    url TEXT,
    component TEXT,

    -- User agent details
    browser TEXT,
    os TEXT,
    device_type TEXT CHECK (device_type IS NULL OR device_type IN ('desktop', 'mobile', 'tablet')),

    -- Metrics
    occurrence_count INT DEFAULT 1,
    affected_users INT DEFAULT 1,

    -- Timestamps
    first_seen_at TIMESTAMPTZ,
    last_seen_at TIMESTAMPTZ,

    -- Status
    status TEXT DEFAULT 'new' CHECK (status IN ('new', 'processing', 'resolved', 'ignored')),

    -- Flexible storage
    raw_payload JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',

    -- Audit
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- =============================================================================
-- Indexes for production_events
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_production_events_project ON production_events(project_id);
CREATE INDEX IF NOT EXISTS idx_production_events_source ON production_events(source);
CREATE INDEX IF NOT EXISTS idx_production_events_fingerprint ON production_events(fingerprint);
CREATE INDEX IF NOT EXISTS idx_production_events_severity ON production_events(severity);
CREATE INDEX IF NOT EXISTS idx_production_events_status ON production_events(status);
CREATE INDEX IF NOT EXISTS idx_production_events_created ON production_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_production_events_first_seen ON production_events(first_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_production_events_external_id ON production_events(external_id);

-- GIN index for tags array
CREATE INDEX IF NOT EXISTS idx_production_events_tags ON production_events USING GIN(tags);

-- GIN index for JSONB metadata search
CREATE INDEX IF NOT EXISTS idx_production_events_metadata ON production_events USING GIN(metadata);

-- =============================================================================
-- Unique constraint to prevent duplicate events
-- =============================================================================

CREATE UNIQUE INDEX IF NOT EXISTS idx_production_events_unique_event
ON production_events(project_id, source, external_id);

-- =============================================================================
-- Trigger for updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION update_production_events_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_production_events_updated_at ON production_events;
CREATE TRIGGER trigger_production_events_updated_at
    BEFORE UPDATE ON production_events
    FOR EACH ROW
    EXECUTE FUNCTION update_production_events_updated_at();

-- =============================================================================
-- Row Level Security
-- =============================================================================

ALTER TABLE production_events ENABLE ROW LEVEL SECURITY;

-- User access based on organization membership
DROP POLICY IF EXISTS production_events_select ON production_events;
CREATE POLICY production_events_select ON production_events
    FOR SELECT
    USING (
        project_id IN (
            SELECT id FROM projects
            WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = current_setting('app.user_id', true)
            )
        )
    );

DROP POLICY IF EXISTS production_events_insert ON production_events;
CREATE POLICY production_events_insert ON production_events
    FOR INSERT
    WITH CHECK (
        project_id IN (
            SELECT id FROM projects
            WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = current_setting('app.user_id', true)
            )
        )
    );

DROP POLICY IF EXISTS production_events_update ON production_events;
CREATE POLICY production_events_update ON production_events
    FOR UPDATE
    USING (
        project_id IN (
            SELECT id FROM projects
            WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = current_setting('app.user_id', true)
            )
        )
    );

DROP POLICY IF EXISTS production_events_delete ON production_events;
CREATE POLICY production_events_delete ON production_events
    FOR DELETE
    USING (
        project_id IN (
            SELECT id FROM projects
            WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = current_setting('app.user_id', true)
            )
        )
    );

-- Service role full access for backend operations
DROP POLICY IF EXISTS "Service role full access production_events" ON production_events;
CREATE POLICY "Service role full access production_events" ON production_events
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE production_events IS 'Normalized production events from observability platforms (Sentry, Datadog, etc.)';
COMMENT ON COLUMN production_events.source IS 'Origin platform: sentry, datadog, fullstory, logrocket, newrelic, bugsnag, rollbar';
COMMENT ON COLUMN production_events.external_id IS 'ID from the source platform';
COMMENT ON COLUMN production_events.fingerprint IS 'Unique identifier for deduplication';
COMMENT ON COLUMN production_events.event_type IS 'Type: error, exception, performance, session, rage_click, dead_click';
COMMENT ON COLUMN production_events.severity IS 'Severity level: fatal, error, warning, info';
COMMENT ON COLUMN production_events.status IS 'Processing status: new, processing, resolved, ignored';
COMMENT ON COLUMN production_events.raw_payload IS 'Original payload from source platform';
COMMENT ON COLUMN production_events.metadata IS 'Platform-specific metadata';
