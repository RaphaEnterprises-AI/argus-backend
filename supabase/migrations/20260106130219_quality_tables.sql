-- Quality Intelligence tables migration
-- Adds missing columns to risk_scores and ensures generated_tests table exists

-- Fix risk_scores table - add missing columns
DO $$
BEGIN
    -- Add overall_score column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'risk_scores' AND column_name = 'overall_score') THEN
        ALTER TABLE risk_scores ADD COLUMN overall_score INTEGER NOT NULL DEFAULT 50;
    END IF;

    -- Add factors column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'risk_scores' AND column_name = 'factors') THEN
        ALTER TABLE risk_scores ADD COLUMN factors JSONB DEFAULT '{}';
    END IF;

    -- Add error_count column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'risk_scores' AND column_name = 'error_count') THEN
        ALTER TABLE risk_scores ADD COLUMN error_count INTEGER DEFAULT 0;
    END IF;

    -- Add affected_users column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'risk_scores' AND column_name = 'affected_users') THEN
        ALTER TABLE risk_scores ADD COLUMN affected_users INTEGER DEFAULT 0;
    END IF;

    -- Add trend column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'risk_scores' AND column_name = 'trend') THEN
        ALTER TABLE risk_scores ADD COLUMN trend TEXT DEFAULT 'stable';
    END IF;

    -- Add calculated_at column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'risk_scores' AND column_name = 'calculated_at') THEN
        ALTER TABLE risk_scores ADD COLUMN calculated_at TIMESTAMPTZ DEFAULT now();
    END IF;
END $$;

-- Ensure generated_tests has all required columns
DO $$
BEGIN
    -- Add confidence_score if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'generated_tests' AND column_name = 'confidence_score') THEN
        ALTER TABLE generated_tests ADD COLUMN confidence_score NUMERIC(5,2) DEFAULT 0.8;
    END IF;

    -- Add review_notes if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'generated_tests' AND column_name = 'review_notes') THEN
        ALTER TABLE generated_tests ADD COLUMN review_notes TEXT;
    END IF;

    -- Add reviewed_by if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'generated_tests' AND column_name = 'reviewed_by') THEN
        ALTER TABLE generated_tests ADD COLUMN reviewed_by TEXT;
    END IF;

    -- Add github_pr_url if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'generated_tests' AND column_name = 'github_pr_url') THEN
        ALTER TABLE generated_tests ADD COLUMN github_pr_url TEXT;
    END IF;

    -- Add github_pr_number if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'generated_tests' AND column_name = 'github_pr_number') THEN
        ALTER TABLE generated_tests ADD COLUMN github_pr_number INTEGER;
    END IF;
END $$;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_risk_scores_project ON risk_scores(project_id);
CREATE INDEX IF NOT EXISTS idx_risk_scores_overall ON risk_scores(overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_generated_tests_project ON generated_tests(project_id);
CREATE INDEX IF NOT EXISTS idx_generated_tests_status ON generated_tests(status);

-- Enable RLS policies if not already enabled
ALTER TABLE risk_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE generated_tests ENABLE ROW LEVEL SECURITY;

-- Create policies (drop first if exists to avoid errors)
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON risk_scores;
CREATE POLICY "Enable all access for authenticated users" ON risk_scores FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON generated_tests;
CREATE POLICY "Enable all access for authenticated users" ON generated_tests FOR ALL USING (true);
