-- AI Analysis Columns for Schedule Runs and Test Schedules
-- Enables storing AI failure analysis for dashboard display

-- ============================================================================
-- ADD AI ANALYSIS COLUMNS TO SCHEDULE_RUNS TABLE
-- ============================================================================

-- AI analysis results (full analysis JSON from Claude)
-- Contains: root_cause, suggested_fix, similar_failures, confidence, etc.
ALTER TABLE schedule_runs
    ADD COLUMN IF NOT EXISTS ai_analysis JSONB DEFAULT NULL;

-- Flaky test detection flag
-- TRUE if the test has been identified as flaky by AI analysis
ALTER TABLE schedule_runs
    ADD COLUMN IF NOT EXISTS is_flaky BOOLEAN DEFAULT FALSE;

-- Flaky score (0.0 to 1.0)
-- Higher values indicate higher likelihood of flakiness
-- Based on historical pass/fail patterns, timing variance, etc.
ALTER TABLE schedule_runs
    ADD COLUMN IF NOT EXISTS flaky_score FLOAT DEFAULT 0.0;

-- Failure categorization
-- Categories: 'network', 'timeout', 'element_not_found', 'assertion',
--             'authentication', 'data_dependency', 'environment', 'unknown'
ALTER TABLE schedule_runs
    ADD COLUMN IF NOT EXISTS failure_category TEXT DEFAULT NULL;

-- Confidence score for the failure categorization (0.0 to 1.0)
-- Higher values indicate more confident categorization
ALTER TABLE schedule_runs
    ADD COLUMN IF NOT EXISTS failure_confidence FLOAT DEFAULT NULL;

-- ============================================================================
-- ADD AI CONFIGURATION COLUMNS TO TEST_SCHEDULES TABLE
-- ============================================================================

-- Enable automatic self-healing for failed tests
-- When TRUE, AI will attempt to fix selectors/assertions on failure
ALTER TABLE test_schedules
    ADD COLUMN IF NOT EXISTS auto_heal_enabled BOOLEAN DEFAULT FALSE;

-- Minimum confidence threshold for auto-healing (0.0 to 1.0)
-- AI fixes below this confidence will require manual approval
ALTER TABLE test_schedules
    ADD COLUMN IF NOT EXISTS auto_heal_confidence_threshold FLOAT DEFAULT 0.9;

-- Enable automatic quarantine of flaky tests
-- When TRUE, tests exceeding flaky_threshold will be quarantined
ALTER TABLE test_schedules
    ADD COLUMN IF NOT EXISTS quarantine_flaky_tests BOOLEAN DEFAULT FALSE;

-- Flaky threshold (0.0 to 1.0)
-- Tests with flaky_score above this value are considered flaky
-- Default 0.3 means tests failing >30% intermittently are flagged
ALTER TABLE test_schedules
    ADD COLUMN IF NOT EXISTS flaky_threshold FLOAT DEFAULT 0.3;

-- ============================================================================
-- INDEXES FOR QUERYING AI ANALYSIS DATA
-- ============================================================================

-- Index for finding flaky runs
CREATE INDEX IF NOT EXISTS idx_schedule_runs_is_flaky
    ON schedule_runs(is_flaky)
    WHERE is_flaky = TRUE;

-- Index for filtering by failure category
CREATE INDEX IF NOT EXISTS idx_schedule_runs_failure_category
    ON schedule_runs(failure_category)
    WHERE failure_category IS NOT NULL;

-- Index for finding runs with AI analysis
CREATE INDEX IF NOT EXISTS idx_schedule_runs_has_ai_analysis
    ON schedule_runs(schedule_id, created_at DESC)
    WHERE ai_analysis IS NOT NULL;

-- Composite index for flaky analysis queries (schedule + flaky status)
CREATE INDEX IF NOT EXISTS idx_schedule_runs_flaky_analysis
    ON schedule_runs(schedule_id, flaky_score DESC)
    WHERE is_flaky = TRUE;

-- Index for schedules with auto-heal enabled
CREATE INDEX IF NOT EXISTS idx_test_schedules_auto_heal
    ON test_schedules(project_id)
    WHERE auto_heal_enabled = TRUE;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON COLUMN schedule_runs.ai_analysis IS
    'Full AI analysis JSON containing root_cause, suggested_fix, similar_failures, and other diagnostic data';

COMMENT ON COLUMN schedule_runs.is_flaky IS
    'Whether this run has been identified as a flaky test failure by AI analysis';

COMMENT ON COLUMN schedule_runs.flaky_score IS
    'Probability score (0.0-1.0) indicating likelihood of test flakiness based on historical patterns';

COMMENT ON COLUMN schedule_runs.failure_category IS
    'AI-categorized failure type: network, timeout, element_not_found, assertion, authentication, data_dependency, environment, unknown';

COMMENT ON COLUMN schedule_runs.failure_confidence IS
    'Confidence score (0.0-1.0) for the failure categorization';

COMMENT ON COLUMN test_schedules.auto_heal_enabled IS
    'Enable automatic self-healing of failed tests using AI-suggested fixes';

COMMENT ON COLUMN test_schedules.auto_heal_confidence_threshold IS
    'Minimum AI confidence (0.0-1.0) required for automatic healing; lower confidence fixes require approval';

COMMENT ON COLUMN test_schedules.quarantine_flaky_tests IS
    'Automatically quarantine tests that exceed the flaky threshold';

COMMENT ON COLUMN test_schedules.flaky_threshold IS
    'Flaky score threshold (0.0-1.0); tests with scores above this are flagged as flaky';

-- ============================================================================
-- COMPLETION
-- ============================================================================

SELECT 'AI analysis columns added to schedule_runs and test_schedules successfully!' as message;
