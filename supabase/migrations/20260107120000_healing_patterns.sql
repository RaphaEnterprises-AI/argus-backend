-- Healing Patterns Table
-- Stores successful selector healing patterns for cross-test learning

CREATE TABLE IF NOT EXISTS healing_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fingerprint TEXT UNIQUE NOT NULL,
    original_selector TEXT NOT NULL,
    healed_selector TEXT NOT NULL,
    error_type TEXT NOT NULL,
    success_count INT DEFAULT 1,
    failure_count INT DEFAULT 0,
    confidence NUMERIC GENERATED ALWAYS AS (
        success_count::numeric / GREATEST(success_count + failure_count, 1)
    ) STORED,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    page_url TEXT,
    element_context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_healing_fingerprint ON healing_patterns(fingerprint);
CREATE INDEX IF NOT EXISTS idx_healing_confidence ON healing_patterns(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_healing_project ON healing_patterns(project_id);
CREATE INDEX IF NOT EXISTS idx_healing_original_selector ON healing_patterns(original_selector);
CREATE INDEX IF NOT EXISTS idx_healing_error_type ON healing_patterns(error_type);

-- Trigger to update updated_at on changes
CREATE OR REPLACE FUNCTION update_healing_patterns_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_healing_patterns_updated_at ON healing_patterns;
CREATE TRIGGER trigger_healing_patterns_updated_at
    BEFORE UPDATE ON healing_patterns
    FOR EACH ROW
    EXECUTE FUNCTION update_healing_patterns_updated_at();

-- RLS policies
ALTER TABLE healing_patterns ENABLE ROW LEVEL SECURITY;

-- Policy: Users can read healing patterns for their organization's projects
CREATE POLICY healing_patterns_select ON healing_patterns
    FOR SELECT
    USING (
        project_id IN (
            SELECT id FROM projects
            WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = current_setting('app.user_id', true)
            )
        )
    );

-- Policy: Users can insert healing patterns for their organization's projects
CREATE POLICY healing_patterns_insert ON healing_patterns
    FOR INSERT
    WITH CHECK (
        project_id IN (
            SELECT id FROM projects
            WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = current_setting('app.user_id', true)
            )
        )
    );

-- Policy: Users can update healing patterns for their organization's projects
CREATE POLICY healing_patterns_update ON healing_patterns
    FOR UPDATE
    USING (
        project_id IN (
            SELECT id FROM projects
            WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = current_setting('app.user_id', true)
            )
        )
    );

-- Service role policy (for backend operations)
CREATE POLICY "Service role has full access to healing_patterns" ON healing_patterns
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- Comment for documentation
COMMENT ON TABLE healing_patterns IS 'Stores successful selector healing patterns for cross-test learning';
COMMENT ON COLUMN healing_patterns.fingerprint IS 'Unique hash of original selector + error type for deduplication';
COMMENT ON COLUMN healing_patterns.confidence IS 'Calculated confidence based on success/failure ratio';
COMMENT ON COLUMN healing_patterns.element_context IS 'Additional context about the element (tag, classes, nearby elements)';
