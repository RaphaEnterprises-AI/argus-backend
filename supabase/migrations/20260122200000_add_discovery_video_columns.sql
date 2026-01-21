-- Add video recording columns to discovery_sessions
-- Migration: 20260122200000_add_discovery_video_columns.sql
--
-- Enables session recording for discovery like BrowserStack/LambdaTest

-- Add video_artifact_id column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'discovery_sessions' AND column_name = 'video_artifact_id'
    ) THEN
        ALTER TABLE discovery_sessions ADD COLUMN video_artifact_id TEXT;
        COMMENT ON COLUMN discovery_sessions.video_artifact_id IS 'ID of the recorded video artifact';
    END IF;
END $$;

-- Add recording_url column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'discovery_sessions' AND column_name = 'recording_url'
    ) THEN
        ALTER TABLE discovery_sessions ADD COLUMN recording_url TEXT;
        COMMENT ON COLUMN discovery_sessions.recording_url IS 'URL to view the session recording';
    END IF;
END $$;

-- Create index for quick lookups by video_artifact_id
CREATE INDEX IF NOT EXISTS idx_discovery_sessions_video
ON discovery_sessions(video_artifact_id)
WHERE video_artifact_id IS NOT NULL;
