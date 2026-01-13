-- Migration: Add test_defaults column and update notification_preferences
-- Date: 2026-01-13

-- Add test_defaults JSONB column to user_profiles
ALTER TABLE user_profiles
ADD COLUMN IF NOT EXISTS test_defaults JSONB DEFAULT jsonb_build_object(
    'default_browser', 'chromium',
    'default_timeout', 30000,
    'parallel_execution', true,
    'retry_failed_tests', true,
    'max_retries', 2,
    'screenshot_on_failure', true,
    'video_recording', false
);

-- Update notification_preferences default to include all fields
-- Note: This only affects new rows, existing rows keep their current values
ALTER TABLE user_profiles
ALTER COLUMN notification_preferences SET DEFAULT jsonb_build_object(
    'email_notifications', true,
    'slack_notifications', false,
    'in_app_notifications', true,
    'email_test_failures', true,
    'email_test_completions', false,
    'email_weekly_digest', true,
    'slack_test_failures', false,
    'slack_test_completions', false,
    'in_app_test_failures', true,
    'in_app_test_completions', true,
    'test_failure_alerts', true,
    'daily_digest', false,
    'weekly_report', true,
    'alert_threshold', 80
);

-- Add comment for documentation
COMMENT ON COLUMN user_profiles.test_defaults IS 'User test execution preferences including browser, timeout, parallelization settings';
COMMENT ON COLUMN user_profiles.notification_preferences IS 'User notification preferences for email, Slack, and in-app notifications';
