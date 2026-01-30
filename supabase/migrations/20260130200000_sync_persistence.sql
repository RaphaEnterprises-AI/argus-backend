-- Sync persistence tables for IDE synchronization
-- Persists sync state to survive server restarts

-- =============================================================================
-- Project sync state table
-- Stores project-level sync state including test specs and versions
-- =============================================================================
CREATE TABLE IF NOT EXISTS sync_project_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    -- Full state stored as JSONB including:
    -- - tests: dict of test_id -> TestSyncState
    -- - last_full_sync: timestamp
    -- - sync_enabled: bool
    state JSONB NOT NULL DEFAULT '{}',
    -- Separate storage for specs (can be large)
    local_specs JSONB NOT NULL DEFAULT '{}',
    base_specs JSONB NOT NULL DEFAULT '{}',
    last_sync TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for user lookups
CREATE INDEX IF NOT EXISTS idx_sync_project_state_user
    ON sync_project_state(user_id);

-- Index for last sync time (for cleanup jobs)
CREATE INDEX IF NOT EXISTS idx_sync_project_state_last_sync
    ON sync_project_state(last_sync);

-- =============================================================================
-- Pending sync events table
-- Stores events for recovery after server restart
-- =============================================================================
CREATE TABLE IF NOT EXISTS sync_pending_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    test_id TEXT,
    event_type TEXT NOT NULL,
    event_data JSONB NOT NULL,
    -- Ordering for replay
    local_version INTEGER DEFAULT 0,
    remote_version INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Sync status
    synced BOOLEAN DEFAULT FALSE,
    synced_at TIMESTAMP WITH TIME ZONE,
    -- Error tracking
    sync_attempts INTEGER DEFAULT 0,
    last_error TEXT
);

-- Index for fetching pending events by project
CREATE INDEX IF NOT EXISTS idx_sync_pending_events_project
    ON sync_pending_events(project_id, synced);

-- Index for fetching pending events by user
CREATE INDEX IF NOT EXISTS idx_sync_pending_events_user
    ON sync_pending_events(user_id, synced);

-- Index for cleanup of old synced events
CREATE INDEX IF NOT EXISTS idx_sync_pending_events_synced_at
    ON sync_pending_events(synced_at)
    WHERE synced = TRUE;

-- Composite index for efficient ordering during replay
CREATE INDEX IF NOT EXISTS idx_sync_pending_events_order
    ON sync_pending_events(project_id, local_version, created_at);

-- =============================================================================
-- Updated_at trigger for sync_project_state
-- =============================================================================
CREATE OR REPLACE FUNCTION update_sync_project_state_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS sync_project_state_updated_at ON sync_project_state;
CREATE TRIGGER sync_project_state_updated_at
    BEFORE UPDATE ON sync_project_state
    FOR EACH ROW
    EXECUTE FUNCTION update_sync_project_state_updated_at();

-- =============================================================================
-- RLS Policies
-- =============================================================================

-- Enable RLS
ALTER TABLE sync_project_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE sync_pending_events ENABLE ROW LEVEL SECURITY;

-- Service role bypass (for backend operations)
CREATE POLICY "Service role full access to sync_project_state"
    ON sync_project_state
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role full access to sync_pending_events"
    ON sync_pending_events
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- =============================================================================
-- Cleanup function for old synced events
-- Removes events that have been synced more than 7 days ago
-- =============================================================================
CREATE OR REPLACE FUNCTION cleanup_old_sync_events(days_old INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM sync_pending_events
    WHERE synced = TRUE
      AND synced_at < NOW() - (days_old || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Comments
-- =============================================================================
COMMENT ON TABLE sync_project_state IS 'Persists IDE sync state for projects to survive server restarts';
COMMENT ON TABLE sync_pending_events IS 'Stores pending sync events for recovery and replay';
COMMENT ON COLUMN sync_project_state.state IS 'Full ProjectSyncState as JSONB including test states';
COMMENT ON COLUMN sync_project_state.local_specs IS 'Current local test specifications (test_id -> spec)';
COMMENT ON COLUMN sync_project_state.base_specs IS 'Last synced test specifications for conflict detection';
COMMENT ON COLUMN sync_pending_events.event_data IS 'Full SyncEvent data as JSONB';
COMMENT ON COLUMN sync_pending_events.sync_attempts IS 'Number of sync attempts for retry logic';
