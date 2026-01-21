-- Add discovery preferences to user_profiles
-- These settings persist the user's preferred discovery configuration

ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS discovery_preferences JSONB DEFAULT jsonb_build_object(
    'mode', 'standard',
    'strategy', 'bfs',
    'maxPages', 50,
    'maxDepth', 3,
    'includePatterns', '',
    'excludePatterns', '/api/*, /static/*, *.pdf, *.jpg, *.png',
    'focusAreas', '[]'::jsonb,
    'captureScreenshots', true,
    'useVisionAi', false
);

-- Add comment for documentation
COMMENT ON COLUMN user_profiles.discovery_preferences IS 'User preferences for discovery configuration: mode, strategy, limits, patterns, etc.';
