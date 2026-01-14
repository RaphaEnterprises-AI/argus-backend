-- Selector Alternatives Table for Self-Healing Integration
-- This table stores alternative selectors discovered during exploration
-- that can be used by the Self-Healer when primary selectors break.

-- Create the selector_alternatives table
CREATE TABLE IF NOT EXISTS selector_alternatives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Primary selector that might break
    primary_selector TEXT NOT NULL,

    -- Alternative selectors with metadata
    -- Structure: [{"selector": "...", "strategy": "xpath|aria|text|data_attr", "confidence": 0.9}]
    alternatives JSONB NOT NULL DEFAULT '[]',

    -- Unique fingerprint for deduplication
    fingerprint TEXT NOT NULL,

    -- Source information
    source TEXT DEFAULT 'discovery',  -- discovery, manual, healing
    discovery_session_id UUID REFERENCES discovery_sessions(id) ON DELETE SET NULL,

    -- Element context
    element_type TEXT,  -- button, link, input, etc.
    element_label TEXT,  -- Human-readable label
    page_url TEXT,

    -- Quality metrics
    stability_score FLOAT DEFAULT 0.5,  -- How stable is this selector
    usage_count INTEGER DEFAULT 0,      -- Times used by self-healer
    success_count INTEGER DEFAULT 0,    -- Successful healing uses

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_selector_alt_project ON selector_alternatives(project_id);
CREATE INDEX IF NOT EXISTS idx_selector_alt_fingerprint ON selector_alternatives(fingerprint);
CREATE INDEX IF NOT EXISTS idx_selector_alt_primary ON selector_alternatives(primary_selector);
CREATE INDEX IF NOT EXISTS idx_selector_alt_page ON selector_alternatives(page_url);

-- Create unique constraint on fingerprint per project
CREATE UNIQUE INDEX IF NOT EXISTS idx_selector_alt_unique
    ON selector_alternatives(project_id, fingerprint);

-- Add visual_baseline_ids column to discovery_sessions for linking
ALTER TABLE discovery_sessions
    ADD COLUMN IF NOT EXISTS visual_baseline_ids UUID[] DEFAULT '{}';

-- Add source and discovery link columns to visual_baselines
ALTER TABLE visual_baselines
    ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'manual',
    ADD COLUMN IF NOT EXISTS discovery_session_id UUID REFERENCES discovery_sessions(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS category TEXT;

-- Create index for baselines from discovery
CREATE INDEX IF NOT EXISTS idx_visual_baselines_discovery
    ON visual_baselines(discovery_session_id)
    WHERE discovery_session_id IS NOT NULL;

-- Add RLS policies
ALTER TABLE selector_alternatives ENABLE ROW LEVEL SECURITY;

-- Policy: Users can view selector alternatives for projects in their organizations
CREATE POLICY "Users can view selector alternatives for their org projects"
    ON selector_alternatives
    FOR SELECT
    USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organization_members om ON p.organization_id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policy: Users can insert selector alternatives for projects in their organizations
CREATE POLICY "Users can insert selector alternatives for their org projects"
    ON selector_alternatives
    FOR INSERT
    WITH CHECK (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organization_members om ON p.organization_id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policy: Users can update selector alternatives for projects in their organizations
CREATE POLICY "Users can update selector alternatives for their org projects"
    ON selector_alternatives
    FOR UPDATE
    USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organization_members om ON p.organization_id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Function to get best alternative selectors for a broken selector
CREATE OR REPLACE FUNCTION get_best_alternatives(
    p_project_id UUID,
    p_selector TEXT,
    p_limit INTEGER DEFAULT 5
)
RETURNS TABLE (
    selector TEXT,
    strategy TEXT,
    confidence FLOAT,
    usage_count INTEGER,
    success_rate FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        alt->>'selector' as selector,
        alt->>'strategy' as strategy,
        (alt->>'confidence')::FLOAT as confidence,
        COALESCE((alt->>'usage_count')::INTEGER, 0) as usage_count,
        CASE
            WHEN COALESCE((alt->>'usage_count')::INTEGER, 0) > 0
            THEN COALESCE((alt->>'success_count')::INTEGER, 0)::FLOAT / (alt->>'usage_count')::INTEGER
            ELSE 0.5
        END as success_rate
    FROM selector_alternatives sa,
         jsonb_array_elements(sa.alternatives) as alt
    WHERE sa.project_id = p_project_id
      AND sa.primary_selector = p_selector
    ORDER BY
        (alt->>'confidence')::FLOAT DESC,
        success_rate DESC
    LIMIT p_limit;
END;
$$;

COMMENT ON TABLE selector_alternatives IS 'Stores alternative selectors for self-healing, populated by discovery';
COMMENT ON FUNCTION get_best_alternatives IS 'Returns best alternative selectors for a broken selector, ordered by confidence and success rate';
