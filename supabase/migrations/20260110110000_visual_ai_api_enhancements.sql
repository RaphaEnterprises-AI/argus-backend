-- =============================================================================
-- Visual AI API Enhancements
-- Version: 1.0.0
-- Date: 2026-01-10
-- Description: Additional columns and tables for enhanced Visual AI API endpoints
-- =============================================================================

-- ============================================================================
-- Add missing columns to visual_snapshots for new API
-- ============================================================================

DO $$
BEGIN
    -- URL of the captured page
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_snapshots' AND column_name = 'url') THEN
        ALTER TABLE visual_snapshots ADD COLUMN url TEXT;
    END IF;

    -- Screenshot stored path (local or S3)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_snapshots' AND column_name = 'screenshot_path') THEN
        ALTER TABLE visual_snapshots ADD COLUMN screenshot_path TEXT;
    END IF;

    -- Viewport width as separate column for easier querying
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_snapshots' AND column_name = 'viewport_width') THEN
        ALTER TABLE visual_snapshots ADD COLUMN viewport_width INTEGER DEFAULT 1440;
    END IF;

    -- Viewport height as separate column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_snapshots' AND column_name = 'viewport_height') THEN
        ALTER TABLE visual_snapshots ADD COLUMN viewport_height INTEGER DEFAULT 900;
    END IF;

    -- Name/label for the snapshot
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_snapshots' AND column_name = 'name') THEN
        ALTER TABLE visual_snapshots ADD COLUMN name TEXT;
    END IF;
END $$;

-- ============================================================================
-- Add missing columns to visual_baselines
-- ============================================================================

DO $$
BEGIN
    -- Name for the baseline
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'name') THEN
        ALTER TABLE visual_baselines ADD COLUMN name TEXT;
    END IF;

    -- Project ID reference
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'project_id') THEN
        ALTER TABLE visual_baselines ADD COLUMN project_id UUID;
    END IF;

    -- URL for the baseline
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'url') THEN
        ALTER TABLE visual_baselines ADD COLUMN url TEXT;
    END IF;

    -- Screenshot path
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'screenshot_path') THEN
        ALTER TABLE visual_baselines ADD COLUMN screenshot_path TEXT;
    END IF;

    -- Viewport dimensions
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'viewport_width') THEN
        ALTER TABLE visual_baselines ADD COLUMN viewport_width INTEGER DEFAULT 1440;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'viewport_height') THEN
        ALTER TABLE visual_baselines ADD COLUMN viewport_height INTEGER DEFAULT 900;
    END IF;

    -- Browser used
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'browser') THEN
        ALTER TABLE visual_baselines ADD COLUMN browser TEXT DEFAULT 'chromium';
    END IF;

    -- Version tracking
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'version') THEN
        ALTER TABLE visual_baselines ADD COLUMN version INTEGER DEFAULT 1;
    END IF;

    -- Updated at timestamp
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'updated_at') THEN
        ALTER TABLE visual_baselines ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
    END IF;
END $$;

-- ============================================================================
-- Add missing columns to visual_comparisons for new API
-- ============================================================================

DO $$
BEGIN
    -- Current snapshot ID reference
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_comparisons' AND column_name = 'current_snapshot_id') THEN
        ALTER TABLE visual_comparisons ADD COLUMN current_snapshot_id UUID;
    END IF;

    -- Has regressions flag
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_comparisons' AND column_name = 'has_regressions') THEN
        ALTER TABLE visual_comparisons ADD COLUMN has_regressions BOOLEAN DEFAULT false;
    END IF;

    -- Differences as JSONB array
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_comparisons' AND column_name = 'differences') THEN
        ALTER TABLE visual_comparisons ADD COLUMN differences JSONB DEFAULT '[]';
    END IF;

    -- Summary text
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_comparisons' AND column_name = 'summary') THEN
        ALTER TABLE visual_comparisons ADD COLUMN summary TEXT;
    END IF;

    -- Cost tracking
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_comparisons' AND column_name = 'cost_usd') THEN
        ALTER TABLE visual_comparisons ADD COLUMN cost_usd FLOAT DEFAULT 0;
    END IF;

    -- Context provided for comparison
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_comparisons' AND column_name = 'context') THEN
        ALTER TABLE visual_comparisons ADD COLUMN context TEXT;
    END IF;

    -- Approved changes list
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_comparisons' AND column_name = 'approved_changes') THEN
        ALTER TABLE visual_comparisons ADD COLUMN approved_changes TEXT[];
    END IF;

    -- AI explanation
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_comparisons' AND column_name = 'ai_explanation') THEN
        ALTER TABLE visual_comparisons ADD COLUMN ai_explanation JSONB;
    END IF;

    -- Compared at timestamp
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_comparisons' AND column_name = 'compared_at') THEN
        ALTER TABLE visual_comparisons ADD COLUMN compared_at TIMESTAMPTZ DEFAULT NOW();
    END IF;
END $$;

-- ============================================================================
-- Create visual_baseline_history table for version tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS visual_baseline_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    baseline_id UUID NOT NULL,
    version INTEGER NOT NULL,
    screenshot_path TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(baseline_id, version)
);

-- Indexes for visual_baseline_history
CREATE INDEX IF NOT EXISTS idx_visual_baseline_history_baseline ON visual_baseline_history(baseline_id);
CREATE INDEX IF NOT EXISTS idx_visual_baseline_history_version ON visual_baseline_history(baseline_id, version DESC);
CREATE INDEX IF NOT EXISTS idx_visual_baseline_history_created ON visual_baseline_history(created_at DESC);

-- Enable RLS
ALTER TABLE visual_baseline_history ENABLE ROW LEVEL SECURITY;

-- RLS Policies for visual_baseline_history
DROP POLICY IF EXISTS "Users can view baseline history" ON visual_baseline_history;
CREATE POLICY "Users can view baseline history" ON visual_baseline_history
    FOR SELECT USING (true);

DROP POLICY IF EXISTS "Users can manage baseline history" ON visual_baseline_history;
CREATE POLICY "Users can manage baseline history" ON visual_baseline_history
    FOR ALL USING (true);

DROP POLICY IF EXISTS "Service role has full access to visual_baseline_history" ON visual_baseline_history;
CREATE POLICY "Service role has full access to visual_baseline_history" ON visual_baseline_history
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- Create indexes for new columns
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_visual_baselines_project ON visual_baselines(project_id);
CREATE INDEX IF NOT EXISTS idx_visual_baselines_name ON visual_baselines(name);
CREATE INDEX IF NOT EXISTS idx_visual_baselines_url ON visual_baselines(url);
CREATE INDEX IF NOT EXISTS idx_visual_baselines_version ON visual_baselines(version DESC);

CREATE INDEX IF NOT EXISTS idx_visual_snapshots_url ON visual_snapshots(url);
CREATE INDEX IF NOT EXISTS idx_visual_snapshots_name ON visual_snapshots(name);
CREATE INDEX IF NOT EXISTS idx_visual_snapshots_viewport ON visual_snapshots(viewport_width, viewport_height);

CREATE INDEX IF NOT EXISTS idx_visual_comparisons_current ON visual_comparisons(current_snapshot_id);
CREATE INDEX IF NOT EXISTS idx_visual_comparisons_regressions ON visual_comparisons(has_regressions) WHERE has_regressions = true;
CREATE INDEX IF NOT EXISTS idx_visual_comparisons_compared ON visual_comparisons(compared_at DESC);

-- ============================================================================
-- Add trigger for updated_at on visual_baselines
-- ============================================================================

DROP TRIGGER IF EXISTS update_visual_baselines_updated_at ON visual_baselines;
CREATE TRIGGER update_visual_baselines_updated_at
    BEFORE UPDATE ON visual_baselines
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE visual_baseline_history IS 'Version history for visual baselines enabling rollback';
COMMENT ON COLUMN visual_baselines.version IS 'Current version number, incremented on each update';
COMMENT ON COLUMN visual_comparisons.has_regressions IS 'Whether AI detected visual regressions';
COMMENT ON COLUMN visual_comparisons.differences IS 'Array of detected visual differences with metadata';
COMMENT ON COLUMN visual_comparisons.ai_explanation IS 'AI-generated explanation of visual changes';

-- ============================================================================
-- Completion
-- ============================================================================

SELECT 'Visual AI API enhancements applied successfully!' as message;
