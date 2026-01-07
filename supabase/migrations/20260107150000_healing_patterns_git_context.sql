-- Healing Patterns Git Context Enhancement
-- Adds git-aware healing context to enable 99.9% accuracy self-healing
--
-- The Argus Advantage: We don't just guess why selectors broke,
-- we READ THE GIT HISTORY to find the exact commit and explain WHY.

-- Add git context columns
ALTER TABLE healing_patterns
    ADD COLUMN IF NOT EXISTS git_commit_sha TEXT,
    ADD COLUMN IF NOT EXISTS git_commit_message TEXT,
    ADD COLUMN IF NOT EXISTS git_commit_author TEXT,
    ADD COLUMN IF NOT EXISTS git_commit_date TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS git_file_changed TEXT,
    ADD COLUMN IF NOT EXISTS git_line_number INT,
    ADD COLUMN IF NOT EXISTS code_context TEXT,
    ADD COLUMN IF NOT EXISTS component_name TEXT,
    ADD COLUMN IF NOT EXISTS healing_source TEXT DEFAULT 'unknown';  -- 'git', 'source', 'llm', 'cache'

-- Index for searching by component
CREATE INDEX IF NOT EXISTS idx_healing_component ON healing_patterns(component_name);

-- Index for searching by git author (useful for accountability)
CREATE INDEX IF NOT EXISTS idx_healing_git_author ON healing_patterns(git_commit_author);

-- Index for healing source analysis
CREATE INDEX IF NOT EXISTS idx_healing_source ON healing_patterns(healing_source);

-- Comments for documentation
COMMENT ON COLUMN healing_patterns.git_commit_sha IS 'Git commit SHA that changed the selector';
COMMENT ON COLUMN healing_patterns.git_commit_message IS 'Commit message explaining why the change was made';
COMMENT ON COLUMN healing_patterns.git_commit_author IS 'Author of the commit (for accountability)';
COMMENT ON COLUMN healing_patterns.git_commit_date IS 'When the commit was made';
COMMENT ON COLUMN healing_patterns.git_file_changed IS 'Path to the file that was changed';
COMMENT ON COLUMN healing_patterns.git_line_number IS 'Line number where the selector was changed';
COMMENT ON COLUMN healing_patterns.code_context IS 'Surrounding code context for the change';
COMMENT ON COLUMN healing_patterns.component_name IS 'React/Vue/etc component name containing the selector';
COMMENT ON COLUMN healing_patterns.healing_source IS 'How the healing was discovered: git, source, llm, or cache';

-- Create a view for healing analytics
CREATE OR REPLACE VIEW healing_analytics AS
SELECT
    project_id,
    healing_source,
    error_type,
    COUNT(*) as pattern_count,
    AVG(confidence) as avg_confidence,
    SUM(success_count) as total_successes,
    SUM(failure_count) as total_failures,
    COUNT(DISTINCT component_name) as unique_components,
    COUNT(DISTINCT git_commit_author) as unique_authors
FROM healing_patterns
GROUP BY project_id, healing_source, error_type;

COMMENT ON VIEW healing_analytics IS 'Analytics view for healing pattern effectiveness by source and type';

-- Create a function to find similar healing patterns
CREATE OR REPLACE FUNCTION find_similar_healing_patterns(
    p_selector TEXT,
    p_project_id UUID DEFAULT NULL,
    p_limit INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    original_selector TEXT,
    healed_selector TEXT,
    confidence NUMERIC,
    similarity FLOAT,
    git_commit_sha TEXT,
    git_commit_message TEXT,
    healing_source TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        hp.id,
        hp.original_selector,
        hp.healed_selector,
        hp.confidence,
        similarity(hp.original_selector, p_selector) as similarity,
        hp.git_commit_sha,
        hp.git_commit_message,
        hp.healing_source
    FROM healing_patterns hp
    WHERE
        (p_project_id IS NULL OR hp.project_id = p_project_id)
        AND hp.confidence > 0.5
        AND similarity(hp.original_selector, p_selector) > 0.3
    ORDER BY similarity(hp.original_selector, p_selector) DESC, hp.confidence DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Note: This function requires the pg_trgm extension for similarity()
-- If not enabled: CREATE EXTENSION IF NOT EXISTS pg_trgm;

COMMENT ON FUNCTION find_similar_healing_patterns IS 'Find healing patterns similar to a given selector using trigram similarity';
