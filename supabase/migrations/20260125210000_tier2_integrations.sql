-- Tier 2 Integrations Migration
-- Extends sdlc_events to support PagerDuty, LaunchDarkly, Amplitude, Datadog, FullStory

-- First, modify the source_platform constraint to include new platforms
ALTER TABLE sdlc_events DROP CONSTRAINT IF EXISTS sdlc_events_source_platform_check;
ALTER TABLE sdlc_events ADD CONSTRAINT sdlc_events_source_platform_check
    CHECK (source_platform IN (
        -- Existing platforms
        'jira', 'linear', 'github', 'asana', 'notion',
        -- Tier 2 platforms
        'pagerduty', 'launchdarkly', 'amplitude', 'datadog', 'fullstory'
    ));

-- Modify the event_type constraint to include new event types
ALTER TABLE sdlc_events DROP CONSTRAINT IF EXISTS sdlc_events_event_type_check;
ALTER TABLE sdlc_events ADD CONSTRAINT sdlc_events_event_type_check
    CHECK (event_type IN (
        -- Existing types
        'bug', 'story', 'epic', 'task', 'feature', 'improvement',
        -- Incident types (PagerDuty)
        'incident', 'incident_trigger', 'incident_acknowledge', 'incident_resolve',
        -- Feature flag types (LaunchDarkly)
        'flag_change', 'flag_enabled', 'flag_disabled', 'flag_rollout',
        -- Analytics types (Amplitude)
        'user_event', 'funnel_event', 'retention_event', 'cohort_change',
        -- Observability types (Datadog, FullStory)
        'rum_session', 'error_event', 'performance_anomaly',
        'session_replay', 'frustration_signal', 'rage_click', 'dead_click'
    ));

-- Add additional columns for correlation
ALTER TABLE sdlc_events ADD COLUMN IF NOT EXISTS incident_id TEXT;
ALTER TABLE sdlc_events ADD COLUMN IF NOT EXISTS flag_key TEXT;
ALTER TABLE sdlc_events ADD COLUMN IF NOT EXISTS session_id TEXT;
ALTER TABLE sdlc_events ADD COLUMN IF NOT EXISTS occurred_at TIMESTAMPTZ DEFAULT now();

-- Add indexes for new correlation columns
CREATE INDEX IF NOT EXISTS idx_sdlc_events_incident_id ON sdlc_events(incident_id) WHERE incident_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sdlc_events_flag_key ON sdlc_events(flag_key) WHERE flag_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sdlc_events_session_id ON sdlc_events(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sdlc_events_occurred_at ON sdlc_events(occurred_at DESC);

-- Comments for new columns
COMMENT ON COLUMN sdlc_events.incident_id IS 'PagerDuty incident ID for correlation';
COMMENT ON COLUMN sdlc_events.flag_key IS 'LaunchDarkly feature flag key for correlation';
COMMENT ON COLUMN sdlc_events.session_id IS 'FullStory/Datadog session ID for correlation';
COMMENT ON COLUMN sdlc_events.occurred_at IS 'When the event occurred (for timeline queries)';
