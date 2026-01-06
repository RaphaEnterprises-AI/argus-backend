-- Quality Intelligence tables migration
-- Run this in your Supabase SQL Editor

-- Risk Scores table
CREATE TABLE IF NOT EXISTS risk_scores (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    entity_identifier TEXT NOT NULL,
    overall_score INTEGER NOT NULL DEFAULT 50,
    factors JSONB DEFAULT '{}',
    error_count INTEGER DEFAULT 0,
    affected_users INTEGER DEFAULT 0,
    trend TEXT DEFAULT 'stable',
    calculated_at TIMESTAMPTZ DEFAULT now(),
    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(project_id, entity_type, entity_identifier)
);

-- Enable RLS
ALTER TABLE risk_scores ENABLE ROW LEVEL SECURITY;

-- Policies for risk_scores
CREATE POLICY "Enable all access for authenticated users" ON risk_scores
    FOR ALL USING (true);

-- Generated Tests table
CREATE TABLE IF NOT EXISTS generated_tests (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    production_event_id UUID REFERENCES production_events(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    test_code TEXT NOT NULL,
    framework TEXT NOT NULL DEFAULT 'playwright',
    status TEXT DEFAULT 'pending',
    confidence_score NUMERIC(5,2) DEFAULT 0.8,
    review_notes TEXT,
    reviewed_by TEXT,
    github_pr_url TEXT,
    github_pr_number INTEGER,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS
ALTER TABLE generated_tests ENABLE ROW LEVEL SECURITY;

-- Policies for generated_tests
CREATE POLICY "Enable all access for authenticated users" ON generated_tests
    FOR ALL USING (true);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_risk_scores_project ON risk_scores(project_id);
CREATE INDEX IF NOT EXISTS idx_risk_scores_overall ON risk_scores(overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_generated_tests_project ON generated_tests(project_id);
CREATE INDEX IF NOT EXISTS idx_generated_tests_status ON generated_tests(status);
