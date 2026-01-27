-- ============================================================================
-- Failure Pattern Learning System
-- Learns from historical test failures to predict future failures
-- ============================================================================

-- Failure patterns table
-- Stores learned patterns that predict test failures
CREATE TABLE IF NOT EXISTS failure_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Pattern classification
    pattern_type TEXT NOT NULL CHECK (pattern_type IN (
        'file_change',      -- Specific files cause failures
        'author',           -- Author-specific patterns
        'time_of_day',      -- Time-based patterns (Friday deploys)
        'dependency',       -- Dependency updates cause issues
        'size',             -- Large commits fail more
        'component',        -- Fragile components
        'combination',      -- File combinations that break together
        'flaky',            -- Inherently flaky tests
        'environment'       -- Environment-specific failures
    )),

    -- Pattern definition (JSONB for flexibility)
    pattern_definition JSONB NOT NULL,
    -- Examples:
    -- file_change: {"file_patterns": ["src/auth/*"], "test_patterns": ["test_login*"]}
    -- author: {"author_email": "dev@example.com", "failure_rate": 0.3}
    -- time_of_day: {"hours": [14, 15, 16, 17], "days": [4, 5]}  -- Friday afternoon
    -- dependency: {"package": "react-router", "version_pattern": "^6.*"}
    -- size: {"min_lines": 500, "min_files": 10}
    -- combination: {"files": ["models.py", "views.py"], "together": true}

    -- Human-readable description
    description TEXT,

    -- Target test(s) this pattern predicts failures for
    target_tests TEXT[],
    target_test_patterns TEXT[],

    -- Historical evidence
    occurrences INTEGER DEFAULT 1,
    first_occurrence TIMESTAMPTZ DEFAULT NOW(),
    last_occurrence TIMESTAMPTZ DEFAULT NOW(),
    related_commit_shas TEXT[],
    related_event_ids UUID[],

    -- Prediction tracking
    predictions_made INTEGER DEFAULT 0,
    predictions_correct INTEGER DEFAULT 0,
    true_positives INTEGER DEFAULT 0,   -- Predicted fail, actually failed
    true_negatives INTEGER DEFAULT 0,   -- Predicted pass, actually passed
    false_positives INTEGER DEFAULT 0,  -- Predicted fail, actually passed
    false_negatives INTEGER DEFAULT 0,  -- Predicted pass, actually failed

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    confidence DECIMAL(3,2) DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    min_confidence_threshold DECIMAL(3,2) DEFAULT 0.3,

    -- Metadata
    discovered_by TEXT DEFAULT 'algorithm',  -- 'algorithm', 'user', 'ai'
    last_trained_at TIMESTAMPTZ,
    notes TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Training jobs for learning from history
CREATE TABLE IF NOT EXISTS pattern_training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Job configuration
    job_type TEXT NOT NULL CHECK (job_type IN (
        'full_history',      -- Learn from all historical data
        'incremental',       -- Learn from recent data only
        'pattern_discovery', -- Discover new patterns
        'accuracy_update',   -- Update existing pattern accuracy
        'cleanup'            -- Deactivate low-accuracy patterns
    )),

    -- Scope
    start_date TIMESTAMPTZ,
    end_date TIMESTAMPTZ,
    pattern_types TEXT[],

    -- Status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'running', 'completed', 'failed', 'cancelled'
    )),
    progress DECIMAL(5,2) DEFAULT 0,
    error_message TEXT,

    -- Results
    commits_analyzed INTEGER DEFAULT 0,
    test_runs_analyzed INTEGER DEFAULT 0,
    patterns_discovered INTEGER DEFAULT 0,
    patterns_updated INTEGER DEFAULT 0,
    patterns_deactivated INTEGER DEFAULT 0,

    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Prediction log for tracking accuracy over time
CREATE TABLE IF NOT EXISTS pattern_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    pattern_id UUID NOT NULL REFERENCES failure_patterns(id) ON DELETE CASCADE,

    -- Prediction context
    commit_sha TEXT NOT NULL,
    test_name TEXT NOT NULL,
    test_file_path TEXT,

    -- Prediction
    predicted_failure BOOLEAN NOT NULL,
    prediction_confidence DECIMAL(3,2) NOT NULL,
    prediction_reason TEXT,

    -- Actual outcome (filled in after test run)
    actual_outcome TEXT CHECK (actual_outcome IN ('passed', 'failed', 'skipped', 'error')),
    was_correct BOOLEAN,
    outcome_recorded_at TIMESTAMPTZ,

    -- Metadata
    features_used JSONB,  -- Snapshot of features used for prediction
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Commit features cache
-- Pre-computed features for commits to speed up pattern matching
CREATE TABLE IF NOT EXISTS commit_features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    commit_sha TEXT NOT NULL,

    -- Basic features
    author_email TEXT,
    author_name TEXT,
    commit_timestamp TIMESTAMPTZ,
    day_of_week INTEGER,  -- 0=Monday, 6=Sunday
    hour_of_day INTEGER,  -- 0-23

    -- Size features
    files_changed INTEGER DEFAULT 0,
    lines_added INTEGER DEFAULT 0,
    lines_deleted INTEGER DEFAULT 0,
    total_lines_changed INTEGER DEFAULT 0,

    -- File features (stored as arrays for pattern matching)
    changed_files TEXT[],
    changed_directories TEXT[],
    file_extensions TEXT[],

    -- Component/module features
    components_affected TEXT[],
    is_dependency_update BOOLEAN DEFAULT FALSE,
    updated_dependencies TEXT[],

    -- Risk indicators
    touches_tests BOOLEAN DEFAULT FALSE,
    touches_config BOOLEAN DEFAULT FALSE,
    touches_migrations BOOLEAN DEFAULT FALSE,
    is_merge_commit BOOLEAN DEFAULT FALSE,
    is_revert BOOLEAN DEFAULT FALSE,

    -- Computed scores
    complexity_score DECIMAL(5,2),
    risk_score DECIMAL(3,2),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(project_id, commit_sha)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_failure_patterns_project ON failure_patterns(project_id);
CREATE INDEX IF NOT EXISTS idx_failure_patterns_type ON failure_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_failure_patterns_active ON failure_patterns(project_id, is_active);
CREATE INDEX IF NOT EXISTS idx_failure_patterns_confidence ON failure_patterns(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_failure_patterns_definition ON failure_patterns USING GIN (pattern_definition);

CREATE INDEX IF NOT EXISTS idx_training_jobs_project ON pattern_training_jobs(project_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON pattern_training_jobs(status);

CREATE INDEX IF NOT EXISTS idx_predictions_pattern ON pattern_predictions(pattern_id);
CREATE INDEX IF NOT EXISTS idx_predictions_commit ON pattern_predictions(commit_sha);
CREATE INDEX IF NOT EXISTS idx_predictions_outcome ON pattern_predictions(was_correct);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON pattern_predictions(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_commit_features_project ON commit_features(project_id);
CREATE INDEX IF NOT EXISTS idx_commit_features_sha ON commit_features(commit_sha);
CREATE INDEX IF NOT EXISTS idx_commit_features_author ON commit_features(author_email);
CREATE INDEX IF NOT EXISTS idx_commit_features_files ON commit_features USING GIN (changed_files);

-- Function to calculate pattern accuracy
CREATE OR REPLACE FUNCTION calculate_pattern_accuracy(p_pattern_id UUID)
RETURNS DECIMAL(3,2)
LANGUAGE plpgsql
AS $$
DECLARE
    v_total INTEGER;
    v_correct INTEGER;
BEGIN
    SELECT predictions_made, predictions_correct
    INTO v_total, v_correct
    FROM failure_patterns
    WHERE id = p_pattern_id;

    IF v_total IS NULL OR v_total = 0 THEN
        RETURN 0.50;  -- Default 50% when no predictions
    END IF;

    RETURN ROUND((v_correct::DECIMAL / v_total), 2);
END;
$$;

-- Function to update pattern after prediction outcome
CREATE OR REPLACE FUNCTION update_pattern_from_outcome(
    p_prediction_id UUID,
    p_actual_outcome TEXT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    v_prediction RECORD;
    v_was_correct BOOLEAN;
    v_new_accuracy DECIMAL(3,2);
BEGIN
    -- Get the prediction
    SELECT * INTO v_prediction
    FROM pattern_predictions
    WHERE id = p_prediction_id;

    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    -- Determine if prediction was correct
    v_was_correct := (
        (v_prediction.predicted_failure AND p_actual_outcome IN ('failed', 'error'))
        OR
        (NOT v_prediction.predicted_failure AND p_actual_outcome IN ('passed', 'skipped'))
    );

    -- Update prediction record
    UPDATE pattern_predictions
    SET actual_outcome = p_actual_outcome,
        was_correct = v_was_correct,
        outcome_recorded_at = NOW()
    WHERE id = p_prediction_id;

    -- Update pattern statistics
    UPDATE failure_patterns
    SET predictions_made = predictions_made + 1,
        predictions_correct = predictions_correct + CASE WHEN v_was_correct THEN 1 ELSE 0 END,
        true_positives = true_positives + CASE
            WHEN v_prediction.predicted_failure AND p_actual_outcome IN ('failed', 'error') THEN 1 ELSE 0 END,
        true_negatives = true_negatives + CASE
            WHEN NOT v_prediction.predicted_failure AND p_actual_outcome IN ('passed', 'skipped') THEN 1 ELSE 0 END,
        false_positives = false_positives + CASE
            WHEN v_prediction.predicted_failure AND p_actual_outcome IN ('passed', 'skipped') THEN 1 ELSE 0 END,
        false_negatives = false_negatives + CASE
            WHEN NOT v_prediction.predicted_failure AND p_actual_outcome IN ('failed', 'error') THEN 1 ELSE 0 END,
        updated_at = NOW()
    WHERE id = v_prediction.pattern_id;

    -- Check if pattern should be deactivated
    SELECT calculate_pattern_accuracy(v_prediction.pattern_id) INTO v_new_accuracy;

    IF (SELECT predictions_made FROM failure_patterns WHERE id = v_prediction.pattern_id) > 10
       AND v_new_accuracy < 0.30 THEN
        UPDATE failure_patterns
        SET is_active = FALSE,
            notes = COALESCE(notes, '') || E'\nAuto-deactivated due to low accuracy: ' || v_new_accuracy::TEXT
        WHERE id = v_prediction.pattern_id;
    END IF;

    RETURN TRUE;
END;
$$;

-- Function to find patterns matching a commit
CREATE OR REPLACE FUNCTION find_matching_patterns(
    p_project_id UUID,
    p_commit_sha TEXT
)
RETURNS TABLE (
    pattern_id UUID,
    pattern_type TEXT,
    confidence DECIMAL,
    match_score DECIMAL,
    match_reason TEXT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_features RECORD;
BEGIN
    -- Get commit features
    SELECT * INTO v_features
    FROM commit_features
    WHERE project_id = p_project_id AND commit_sha = p_commit_sha;

    IF NOT FOUND THEN
        RETURN;  -- No features, no matches
    END IF;

    RETURN QUERY
    SELECT
        fp.id AS pattern_id,
        fp.pattern_type,
        fp.confidence,
        -- Calculate match score based on pattern type
        CASE fp.pattern_type
            WHEN 'file_change' THEN
                CASE WHEN v_features.changed_files && (fp.pattern_definition->>'file_patterns')::TEXT[]
                THEN 1.0 ELSE 0.0 END
            WHEN 'author' THEN
                CASE WHEN v_features.author_email = fp.pattern_definition->>'author_email'
                THEN 1.0 ELSE 0.0 END
            WHEN 'time_of_day' THEN
                CASE WHEN v_features.day_of_week = ANY(
                    (SELECT ARRAY_AGG(x::INTEGER) FROM jsonb_array_elements_text(fp.pattern_definition->'days') x)
                ) THEN 0.8 ELSE 0.0 END
            WHEN 'size' THEN
                CASE WHEN v_features.total_lines_changed >= (fp.pattern_definition->>'min_lines')::INTEGER
                THEN 0.9 ELSE 0.0 END
            WHEN 'dependency' THEN
                CASE WHEN v_features.is_dependency_update
                     AND (fp.pattern_definition->>'package') = ANY(v_features.updated_dependencies)
                THEN 1.0 ELSE 0.0 END
            ELSE 0.5
        END::DECIMAL AS match_score,
        'Pattern matched based on ' || fp.pattern_type AS match_reason
    FROM failure_patterns fp
    WHERE fp.project_id = p_project_id
      AND fp.is_active = TRUE
      AND fp.confidence >= 0.3
    ORDER BY fp.confidence DESC;
END;
$$;

-- Function to predict failure probability for a commit
CREATE OR REPLACE FUNCTION predict_failure_probability(
    p_project_id UUID,
    p_commit_sha TEXT,
    p_test_name TEXT DEFAULT NULL
)
RETURNS TABLE (
    failure_probability DECIMAL,
    contributing_patterns UUID[],
    reasons TEXT[],
    confidence_level TEXT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_patterns RECORD;
    v_probability DECIMAL := 0;
    v_pattern_ids UUID[] := '{}';
    v_reasons TEXT[] := '{}';
    v_total_weight DECIMAL := 0;
BEGIN
    -- Aggregate matching patterns
    FOR v_patterns IN
        SELECT * FROM find_matching_patterns(p_project_id, p_commit_sha)
        WHERE match_score > 0
    LOOP
        v_probability := v_probability + (v_patterns.confidence * v_patterns.match_score);
        v_total_weight := v_total_weight + v_patterns.match_score;
        v_pattern_ids := array_append(v_pattern_ids, v_patterns.pattern_id);
        v_reasons := array_append(v_reasons, v_patterns.match_reason);
    END LOOP;

    -- Normalize probability
    IF v_total_weight > 0 THEN
        v_probability := LEAST(1.0, v_probability / v_total_weight);
    ELSE
        v_probability := 0.1;  -- Base probability when no patterns match
    END IF;

    RETURN QUERY SELECT
        ROUND(v_probability, 2) AS failure_probability,
        v_pattern_ids AS contributing_patterns,
        v_reasons AS reasons,
        CASE
            WHEN v_probability >= 0.7 THEN 'high'
            WHEN v_probability >= 0.4 THEN 'medium'
            ELSE 'low'
        END AS confidence_level;
END;
$$;

-- Auto-update timestamps
CREATE OR REPLACE FUNCTION update_failure_patterns_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_failure_patterns_updated
    BEFORE UPDATE ON failure_patterns
    FOR EACH ROW
    EXECUTE FUNCTION update_failure_patterns_timestamp();

-- RLS policies
ALTER TABLE failure_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE pattern_training_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pattern_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE commit_features ENABLE ROW LEVEL SECURITY;

-- Policies for failure_patterns
CREATE POLICY "Users can view patterns for their projects" ON failure_patterns
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage patterns for their projects" ON failure_patterns
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policies for training jobs
CREATE POLICY "Users can view training jobs for their projects" ON pattern_training_jobs
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage training jobs for their projects" ON pattern_training_jobs
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policies for predictions
CREATE POLICY "Users can view predictions for their projects" ON pattern_predictions
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage predictions for their projects" ON pattern_predictions
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policies for commit features
CREATE POLICY "Users can view commit features for their projects" ON commit_features
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage commit features for their projects" ON commit_features
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Grant service role full access
GRANT ALL ON failure_patterns TO service_role;
GRANT ALL ON pattern_training_jobs TO service_role;
GRANT ALL ON pattern_predictions TO service_role;
GRANT ALL ON commit_features TO service_role;

COMMENT ON TABLE failure_patterns IS 'Learned patterns that predict test failures based on commit characteristics';
COMMENT ON TABLE pattern_training_jobs IS 'Jobs for learning patterns from historical data';
COMMENT ON TABLE pattern_predictions IS 'Log of predictions made and their outcomes for accuracy tracking';
COMMENT ON TABLE commit_features IS 'Pre-computed features for commits to speed up pattern matching';
