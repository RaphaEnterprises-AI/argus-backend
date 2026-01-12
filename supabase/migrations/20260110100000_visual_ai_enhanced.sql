-- =============================================================================
-- Enhanced Visual AI Testing Platform Tables
-- Version: 1.0.0
-- Date: 2026-01-10
-- Description: Comprehensive visual testing infrastructure with AI-powered
--              change detection, cross-browser testing, and trend analysis
-- =============================================================================

-- ============================================================================
-- VISUAL BASELINES TABLE ENHANCEMENTS
-- ============================================================================

-- First ensure visual_baselines table exists with basic structure
CREATE TABLE IF NOT EXISTS visual_baselines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID,
    screenshot_hash VARCHAR(64),
    screenshot_url TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add enhanced columns to visual_baselines
DO $$
BEGIN
    -- DOM snapshot for structural comparison
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'dom_snapshot') THEN
        ALTER TABLE visual_baselines ADD COLUMN dom_snapshot TEXT;
    END IF;

    -- Computed styles for CSS regression detection
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'computed_styles') THEN
        ALTER TABLE visual_baselines ADD COLUMN computed_styles JSONB;
    END IF;

    -- Total elements in the DOM
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'element_count') THEN
        ALTER TABLE visual_baselines ADD COLUMN element_count INTEGER;
    END IF;

    -- Hash of layout structure for quick comparison
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'layout_hash') THEN
        ALTER TABLE visual_baselines ADD COLUMN layout_hash TEXT;
    END IF;

    -- Dominant colors extracted from screenshot
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'color_palette') THEN
        ALTER TABLE visual_baselines ADD COLUMN color_palette TEXT[];
    END IF;

    -- Accessibility audit score (0-100)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'accessibility_score') THEN
        ALTER TABLE visual_baselines ADD COLUMN accessibility_score FLOAT;
    END IF;

    -- Core Web Vitals and performance metrics
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'performance_metrics') THEN
        ALTER TABLE visual_baselines ADD COLUMN performance_metrics JSONB;
    END IF;

    -- Timestamp of when the baseline was captured
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'visual_baselines' AND column_name = 'captured_at') THEN
        ALTER TABLE visual_baselines ADD COLUMN captured_at TIMESTAMPTZ DEFAULT NOW();
    END IF;
END $$;

-- Create indexes for visual_baselines
CREATE INDEX IF NOT EXISTS idx_visual_baselines_test ON visual_baselines(test_id);
CREATE INDEX IF NOT EXISTS idx_visual_baselines_hash ON visual_baselines(screenshot_hash);
CREATE INDEX IF NOT EXISTS idx_visual_baselines_layout ON visual_baselines(layout_hash);
CREATE INDEX IF NOT EXISTS idx_visual_baselines_captured ON visual_baselines(captured_at DESC);

-- ============================================================================
-- VISUAL SNAPSHOTS TABLE (Full state capture for each test execution)
-- ============================================================================

CREATE TABLE IF NOT EXISTS visual_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    baseline_id UUID REFERENCES visual_baselines(id) ON DELETE SET NULL,
    test_run_id UUID,

    -- Screenshot and artifacts
    screenshot_url TEXT NOT NULL,
    dom_snapshot_url TEXT,
    har_url TEXT,  -- HTTP Archive for network analysis

    -- Viewport configuration
    viewport JSONB NOT NULL DEFAULT '{"width": 1920, "height": 1080}',
    -- Example: {"width": 1920, "height": 1080, "deviceScaleFactor": 1}

    -- Browser information
    browser TEXT NOT NULL DEFAULT 'chromium',
    browser_version TEXT,
    user_agent TEXT,

    -- Page metrics
    element_count INTEGER,
    layout_hash TEXT,
    color_palette TEXT[],

    -- Core Web Vitals
    lcp_ms INTEGER,  -- Largest Contentful Paint
    cls_score FLOAT,  -- Cumulative Layout Shift
    tti_ms INTEGER,  -- Time to Interactive
    fcp_ms INTEGER,  -- First Contentful Paint
    fid_ms INTEGER,  -- First Input Delay

    -- Git context
    git_commit TEXT,
    git_branch TEXT,
    git_author TEXT,
    git_message TEXT,

    -- Page information
    page_url TEXT,
    page_title TEXT,

    -- Comparison status
    comparison_status TEXT DEFAULT 'pending' CHECK (comparison_status IN (
        'pending', 'matched', 'mismatched', 'approved', 'rejected', 'error'
    )),
    match_percentage FLOAT,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    captured_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for visual_snapshots
CREATE INDEX IF NOT EXISTS idx_visual_snapshots_project ON visual_snapshots(project_id);
CREATE INDEX IF NOT EXISTS idx_visual_snapshots_baseline ON visual_snapshots(baseline_id);
CREATE INDEX IF NOT EXISTS idx_visual_snapshots_test_run ON visual_snapshots(test_run_id);
CREATE INDEX IF NOT EXISTS idx_visual_snapshots_status ON visual_snapshots(comparison_status);
CREATE INDEX IF NOT EXISTS idx_visual_snapshots_git ON visual_snapshots(git_commit, git_branch);
CREATE INDEX IF NOT EXISTS idx_visual_snapshots_captured ON visual_snapshots(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_visual_snapshots_layout ON visual_snapshots(layout_hash);
CREATE INDEX IF NOT EXISTS idx_visual_snapshots_browser ON visual_snapshots(browser, browser_version);

-- ============================================================================
-- VISUAL COMPARISONS TABLE (Comparison results between baseline and snapshot)
-- ============================================================================

CREATE TABLE IF NOT EXISTS visual_comparisons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    baseline_id UUID NOT NULL REFERENCES visual_baselines(id) ON DELETE CASCADE,
    snapshot_id UUID NOT NULL REFERENCES visual_snapshots(id) ON DELETE CASCADE,

    -- Comparison algorithm and results
    algorithm TEXT DEFAULT 'pixelmatch' CHECK (algorithm IN (
        'pixelmatch', 'ssim', 'perceptual', 'ai_semantic', 'hybrid'
    )),
    match_percentage FLOAT NOT NULL,
    diff_pixel_count INTEGER,
    diff_percentage FLOAT,

    -- Visual diff artifacts
    diff_image_url TEXT,
    heatmap_url TEXT,

    -- AI analysis
    ai_analysis JSONB DEFAULT '{}',
    -- Example: {"summary": "...", "confidence": 0.95, "categories": ["layout", "content"]}

    -- Decision tracking
    status TEXT DEFAULT 'pending' CHECK (status IN (
        'pending', 'auto_approved', 'manually_approved', 'rejected', 'needs_review'
    )),
    reviewed_by TEXT,
    reviewed_at TIMESTAMPTZ,
    review_notes TEXT,

    -- Threshold used
    threshold_percentage FLOAT DEFAULT 0.1,

    -- Performance comparison
    performance_diff JSONB DEFAULT '{}',
    -- Example: {"lcp_delta_ms": 50, "cls_delta": 0.01}

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for visual_comparisons
CREATE INDEX IF NOT EXISTS idx_visual_comparisons_project ON visual_comparisons(project_id);
CREATE INDEX IF NOT EXISTS idx_visual_comparisons_baseline ON visual_comparisons(baseline_id);
CREATE INDEX IF NOT EXISTS idx_visual_comparisons_snapshot ON visual_comparisons(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_visual_comparisons_status ON visual_comparisons(status);
CREATE INDEX IF NOT EXISTS idx_visual_comparisons_match ON visual_comparisons(match_percentage);
CREATE INDEX IF NOT EXISTS idx_visual_comparisons_created ON visual_comparisons(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_visual_comparisons_reviewed ON visual_comparisons(reviewed_by, reviewed_at);

-- ============================================================================
-- VISUAL CHANGES TABLE (Detailed diff records for granular change tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS visual_changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    comparison_id UUID NOT NULL REFERENCES visual_comparisons(id) ON DELETE CASCADE,

    -- Change categorization
    category TEXT NOT NULL CHECK (category IN (
        'layout', 'content', 'style', 'image', 'font', 'color',
        'spacing', 'animation', 'visibility', 'performance', 'other'
    )),
    intent TEXT CHECK (intent IN (
        'intentional', 'regression', 'unknown', 'improvement'
    )),

    -- Severity scale (0=info, 1=low, 2=medium, 3=high, 4=critical)
    severity INTEGER NOT NULL DEFAULT 2 CHECK (severity >= 0 AND severity <= 4),

    -- Element identification
    element_selector TEXT,
    element_tag TEXT,
    element_id TEXT,
    element_class TEXT,
    element_xpath TEXT,

    -- Bounding boxes for visual highlighting
    bounds_baseline JSONB,
    -- Example: {"x": 100, "y": 200, "width": 300, "height": 50}
    bounds_current JSONB,

    -- Property change details
    property_name TEXT,
    baseline_value TEXT,
    current_value TEXT,

    -- Human-readable description
    description TEXT NOT NULL,

    -- AI-generated insights
    root_cause TEXT,
    impact_assessment TEXT,
    recommendation TEXT,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),

    -- Git context (for blame/attribution)
    related_commit TEXT,
    related_files TEXT[],
    commit_author TEXT,
    commit_date TIMESTAMPTZ,

    -- Grouping for related changes
    change_group_id UUID,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for visual_changes
CREATE INDEX IF NOT EXISTS idx_visual_changes_comparison ON visual_changes(comparison_id);
CREATE INDEX IF NOT EXISTS idx_visual_changes_category ON visual_changes(category);
CREATE INDEX IF NOT EXISTS idx_visual_changes_severity ON visual_changes(severity);
CREATE INDEX IF NOT EXISTS idx_visual_changes_intent ON visual_changes(intent);
CREATE INDEX IF NOT EXISTS idx_visual_changes_element ON visual_changes(element_selector);
CREATE INDEX IF NOT EXISTS idx_visual_changes_commit ON visual_changes(related_commit);
CREATE INDEX IF NOT EXISTS idx_visual_changes_group ON visual_changes(change_group_id);
CREATE INDEX IF NOT EXISTS idx_visual_changes_created ON visual_changes(created_at DESC);

-- Composite index for filtering by category and severity
CREATE INDEX IF NOT EXISTS idx_visual_changes_cat_sev ON visual_changes(category, severity);

-- ============================================================================
-- VIEWPORT PRESETS TABLE (Responsive testing configurations)
-- ============================================================================

CREATE TABLE IF NOT EXISTS viewport_presets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,

    -- Preset identification
    name TEXT NOT NULL,
    description TEXT,

    -- Viewport dimensions
    width INTEGER NOT NULL CHECK (width > 0 AND width <= 7680),
    height INTEGER NOT NULL CHECK (height > 0 AND height <= 4320),
    device_scale_factor FLOAT DEFAULT 1.0 CHECK (device_scale_factor > 0 AND device_scale_factor <= 4),

    -- Device characteristics
    is_mobile BOOLEAN DEFAULT false,
    is_landscape BOOLEAN DEFAULT false,
    has_touch BOOLEAN DEFAULT false,
    user_agent TEXT,

    -- Device metadata
    device_name TEXT,
    device_category TEXT CHECK (device_category IN (
        'desktop', 'laptop', 'tablet', 'phone', 'watch', 'tv', 'custom'
    )),
    os_name TEXT,
    os_version TEXT,

    -- Usage flags
    is_default BOOLEAN DEFAULT false,
    is_system BOOLEAN DEFAULT false,  -- System presets cannot be deleted
    is_active BOOLEAN DEFAULT true,

    -- Ordering
    display_order INTEGER DEFAULT 0,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Unique constraint per organization
    UNIQUE(organization_id, name)
);

-- Indexes for viewport_presets
CREATE INDEX IF NOT EXISTS idx_viewport_presets_org ON viewport_presets(organization_id);
CREATE INDEX IF NOT EXISTS idx_viewport_presets_project ON viewport_presets(project_id);
CREATE INDEX IF NOT EXISTS idx_viewport_presets_default ON viewport_presets(organization_id, is_default) WHERE is_default = true;
CREATE INDEX IF NOT EXISTS idx_viewport_presets_active ON viewport_presets(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_viewport_presets_category ON viewport_presets(device_category);
CREATE INDEX IF NOT EXISTS idx_viewport_presets_order ON viewport_presets(organization_id, display_order);

-- ============================================================================
-- BROWSER MATRIX TABLE (Cross-browser testing configurations)
-- ============================================================================

CREATE TABLE IF NOT EXISTS browser_matrix (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Browser identification
    browser TEXT NOT NULL CHECK (browser IN (
        'chromium', 'chrome', 'firefox', 'webkit', 'safari', 'edge', 'msedge'
    )),
    version TEXT,  -- Specific version or 'latest', 'stable', 'beta'
    channel TEXT DEFAULT 'stable' CHECK (channel IN ('stable', 'beta', 'dev', 'canary')),

    -- Execution settings
    enabled BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 50 CHECK (priority >= 0 AND priority <= 100),

    -- Browser-specific options
    headless BOOLEAN DEFAULT true,
    slow_mo_ms INTEGER DEFAULT 0,
    default_timeout_ms INTEGER DEFAULT 30000,

    -- Additional browser args
    browser_args TEXT[],
    -- Example: ['--disable-gpu', '--no-sandbox']

    -- Download settings
    download_path TEXT,
    accept_downloads BOOLEAN DEFAULT true,

    -- Geolocation
    geolocation JSONB,
    -- Example: {"latitude": 37.7749, "longitude": -122.4194}

    -- Locale and timezone
    locale TEXT DEFAULT 'en-US',
    timezone_id TEXT DEFAULT 'America/Los_Angeles',

    -- Permissions
    permissions TEXT[],
    -- Example: ['geolocation', 'notifications']

    -- Color scheme preference
    color_scheme TEXT DEFAULT 'light' CHECK (color_scheme IN ('light', 'dark', 'no-preference')),
    forced_colors TEXT DEFAULT 'none' CHECK (forced_colors IN ('none', 'active')),
    reduced_motion TEXT DEFAULT 'no-preference' CHECK (reduced_motion IN ('no-preference', 'reduce')),

    -- HTTP credentials
    http_credentials JSONB,  -- Encrypted or reference to secrets manager

    -- Proxy settings
    proxy JSONB,
    -- Example: {"server": "http://proxy:8080", "username": "...", "password": "..."}

    -- Device emulation
    device_emulation TEXT,  -- Reference to viewport_preset name

    -- Recording options
    record_video BOOLEAN DEFAULT false,
    record_har BOOLEAN DEFAULT false,

    -- Notes
    notes TEXT,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Unique constraint per project
    UNIQUE(project_id, browser, version)
);

-- Indexes for browser_matrix
CREATE INDEX IF NOT EXISTS idx_browser_matrix_project ON browser_matrix(project_id);
CREATE INDEX IF NOT EXISTS idx_browser_matrix_browser ON browser_matrix(browser);
CREATE INDEX IF NOT EXISTS idx_browser_matrix_enabled ON browser_matrix(enabled) WHERE enabled = true;
CREATE INDEX IF NOT EXISTS idx_browser_matrix_priority ON browser_matrix(project_id, priority DESC);

-- ============================================================================
-- VISUAL TEST HISTORY TABLE (Trend analysis and reporting)
-- ============================================================================

CREATE TABLE IF NOT EXISTS visual_test_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    baseline_id UUID REFERENCES visual_baselines(id) ON DELETE SET NULL,

    -- Date (unique per project/baseline combination)
    date DATE NOT NULL DEFAULT CURRENT_DATE,

    -- Comparison counts
    total_comparisons INTEGER NOT NULL DEFAULT 0,
    matches INTEGER NOT NULL DEFAULT 0,
    mismatches INTEGER NOT NULL DEFAULT 0,
    auto_approved INTEGER NOT NULL DEFAULT 0,
    manually_approved INTEGER NOT NULL DEFAULT 0,
    rejected INTEGER NOT NULL DEFAULT 0,
    pending INTEGER NOT NULL DEFAULT 0,

    -- Calculated metrics
    avg_match_percentage FLOAT,
    min_match_percentage FLOAT,
    max_match_percentage FLOAT,

    -- Performance trends
    avg_lcp_ms INTEGER,
    avg_cls_score FLOAT,
    avg_tti_ms INTEGER,

    -- Change distribution
    changes_by_category JSONB DEFAULT '{}',
    -- Example: {"layout": 5, "style": 12, "content": 3}

    changes_by_severity JSONB DEFAULT '{}',
    -- Example: {"0": 10, "1": 8, "2": 5, "3": 2, "4": 0}

    -- Browser breakdown
    comparisons_by_browser JSONB DEFAULT '{}',
    -- Example: {"chromium": 50, "firefox": 30, "webkit": 20}

    -- Viewport breakdown
    comparisons_by_viewport JSONB DEFAULT '{}',
    -- Example: {"1920x1080": 40, "768x1024": 35, "375x812": 25}

    -- Baseline updates
    baseline_updates INTEGER DEFAULT 0,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Unique constraint for daily aggregation
    UNIQUE(project_id, baseline_id, date)
);

-- Partial unique index for project-level daily stats (when baseline_id is null)
CREATE UNIQUE INDEX IF NOT EXISTS idx_visual_test_history_project_date
    ON visual_test_history(project_id, date)
    WHERE baseline_id IS NULL;

-- Indexes for visual_test_history
CREATE INDEX IF NOT EXISTS idx_visual_test_history_project ON visual_test_history(project_id);
CREATE INDEX IF NOT EXISTS idx_visual_test_history_baseline ON visual_test_history(baseline_id);
CREATE INDEX IF NOT EXISTS idx_visual_test_history_date ON visual_test_history(date DESC);
CREATE INDEX IF NOT EXISTS idx_visual_test_history_mismatches ON visual_test_history(mismatches DESC);

-- Composite index for time-range queries
CREATE INDEX IF NOT EXISTS idx_visual_test_history_project_date_range
    ON visual_test_history(project_id, date DESC);

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE visual_baselines ENABLE ROW LEVEL SECURITY;
ALTER TABLE visual_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE visual_comparisons ENABLE ROW LEVEL SECURITY;
ALTER TABLE visual_changes ENABLE ROW LEVEL SECURITY;
ALTER TABLE viewport_presets ENABLE ROW LEVEL SECURITY;
ALTER TABLE browser_matrix ENABLE ROW LEVEL SECURITY;
ALTER TABLE visual_test_history ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- RLS Policies for visual_baselines
-- ============================================================================

DROP POLICY IF EXISTS "Users can view visual baselines" ON visual_baselines;
CREATE POLICY "Users can view visual baselines" ON visual_baselines
    FOR SELECT USING (true);

DROP POLICY IF EXISTS "Users can manage visual baselines" ON visual_baselines;
CREATE POLICY "Users can manage visual baselines" ON visual_baselines
    FOR ALL USING (true);

DROP POLICY IF EXISTS "Service role has full access to visual_baselines" ON visual_baselines;
CREATE POLICY "Service role has full access to visual_baselines" ON visual_baselines
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- RLS Policies for visual_snapshots
-- ============================================================================

DROP POLICY IF EXISTS "Users can view visual snapshots for their projects" ON visual_snapshots;
CREATE POLICY "Users can view visual snapshots for their projects" ON visual_snapshots
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = visual_snapshots.project_id
        )
    );

DROP POLICY IF EXISTS "Users can manage visual snapshots" ON visual_snapshots;
CREATE POLICY "Users can manage visual snapshots" ON visual_snapshots
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = visual_snapshots.project_id
        )
    );

DROP POLICY IF EXISTS "Service role has full access to visual_snapshots" ON visual_snapshots;
CREATE POLICY "Service role has full access to visual_snapshots" ON visual_snapshots
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- RLS Policies for visual_comparisons
-- ============================================================================

DROP POLICY IF EXISTS "Users can view visual comparisons for their projects" ON visual_comparisons;
CREATE POLICY "Users can view visual comparisons for their projects" ON visual_comparisons
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = visual_comparisons.project_id
        )
    );

DROP POLICY IF EXISTS "Users can manage visual comparisons" ON visual_comparisons;
CREATE POLICY "Users can manage visual comparisons" ON visual_comparisons
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = visual_comparisons.project_id
        )
    );

DROP POLICY IF EXISTS "Service role has full access to visual_comparisons" ON visual_comparisons;
CREATE POLICY "Service role has full access to visual_comparisons" ON visual_comparisons
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- RLS Policies for visual_changes
-- ============================================================================

DROP POLICY IF EXISTS "Users can view visual changes" ON visual_changes;
CREATE POLICY "Users can view visual changes" ON visual_changes
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM visual_comparisons vc
            JOIN projects p ON p.id = vc.project_id
            WHERE vc.id = visual_changes.comparison_id
        )
    );

DROP POLICY IF EXISTS "Users can manage visual changes" ON visual_changes;
CREATE POLICY "Users can manage visual changes" ON visual_changes
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM visual_comparisons vc
            JOIN projects p ON p.id = vc.project_id
            WHERE vc.id = visual_changes.comparison_id
        )
    );

DROP POLICY IF EXISTS "Service role has full access to visual_changes" ON visual_changes;
CREATE POLICY "Service role has full access to visual_changes" ON visual_changes
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- RLS Policies for viewport_presets
-- ============================================================================

DROP POLICY IF EXISTS "Users can view viewport presets for their organizations" ON viewport_presets;
CREATE POLICY "Users can view viewport presets for their organizations" ON viewport_presets
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM organization_members om
            WHERE om.organization_id = viewport_presets.organization_id
            AND om.user_id = current_setting('app.user_id', true)
        )
        OR is_system = true
    );

DROP POLICY IF EXISTS "Admins can manage viewport presets" ON viewport_presets;
CREATE POLICY "Admins can manage viewport presets" ON viewport_presets
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM organization_members om
            WHERE om.organization_id = viewport_presets.organization_id
            AND om.user_id = current_setting('app.user_id', true)
            AND om.role IN ('owner', 'admin')
        )
    );

DROP POLICY IF EXISTS "Service role has full access to viewport_presets" ON viewport_presets;
CREATE POLICY "Service role has full access to viewport_presets" ON viewport_presets
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- RLS Policies for browser_matrix
-- ============================================================================

DROP POLICY IF EXISTS "Users can view browser matrix for their projects" ON browser_matrix;
CREATE POLICY "Users can view browser matrix for their projects" ON browser_matrix
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = browser_matrix.project_id
        )
    );

DROP POLICY IF EXISTS "Users can manage browser matrix" ON browser_matrix;
CREATE POLICY "Users can manage browser matrix" ON browser_matrix
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = browser_matrix.project_id
        )
    );

DROP POLICY IF EXISTS "Service role has full access to browser_matrix" ON browser_matrix;
CREATE POLICY "Service role has full access to browser_matrix" ON browser_matrix
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- RLS Policies for visual_test_history
-- ============================================================================

DROP POLICY IF EXISTS "Users can view visual test history for their projects" ON visual_test_history;
CREATE POLICY "Users can view visual test history for their projects" ON visual_test_history
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = visual_test_history.project_id
        )
    );

DROP POLICY IF EXISTS "Users can manage visual test history" ON visual_test_history;
CREATE POLICY "Users can manage visual test history" ON visual_test_history
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = visual_test_history.project_id
        )
    );

DROP POLICY IF EXISTS "Service role has full access to visual_test_history" ON visual_test_history;
CREATE POLICY "Service role has full access to visual_test_history" ON visual_test_history
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update updated_at trigger function (if not exists)
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add triggers for updated_at columns
DROP TRIGGER IF EXISTS update_visual_comparisons_updated_at ON visual_comparisons;
CREATE TRIGGER update_visual_comparisons_updated_at
    BEFORE UPDATE ON visual_comparisons
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS update_viewport_presets_updated_at ON viewport_presets;
CREATE TRIGGER update_viewport_presets_updated_at
    BEFORE UPDATE ON viewport_presets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS update_browser_matrix_updated_at ON browser_matrix;
CREATE TRIGGER update_browser_matrix_updated_at
    BEFORE UPDATE ON browser_matrix
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS update_visual_test_history_updated_at ON visual_test_history;
CREATE TRIGGER update_visual_test_history_updated_at
    BEFORE UPDATE ON visual_test_history
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to record a visual comparison and update history
CREATE OR REPLACE FUNCTION record_visual_comparison(
    p_project_id UUID,
    p_baseline_id UUID,
    p_snapshot_id UUID,
    p_match_percentage FLOAT,
    p_diff_pixel_count INTEGER DEFAULT 0,
    p_algorithm TEXT DEFAULT 'pixelmatch',
    p_threshold FLOAT DEFAULT 0.1
) RETURNS UUID AS $$
DECLARE
    v_comparison_id UUID;
    v_status TEXT;
BEGIN
    -- Determine auto-approval status
    IF p_match_percentage >= (100 - p_threshold) THEN
        v_status := 'auto_approved';
    ELSE
        v_status := 'needs_review';
    END IF;

    -- Create the comparison record
    INSERT INTO visual_comparisons (
        project_id,
        baseline_id,
        snapshot_id,
        algorithm,
        match_percentage,
        diff_pixel_count,
        diff_percentage,
        status,
        threshold_percentage
    ) VALUES (
        p_project_id,
        p_baseline_id,
        p_snapshot_id,
        p_algorithm,
        p_match_percentage,
        p_diff_pixel_count,
        100 - p_match_percentage,
        v_status,
        p_threshold
    )
    RETURNING id INTO v_comparison_id;

    -- Update snapshot comparison status
    UPDATE visual_snapshots
    SET
        comparison_status = CASE
            WHEN v_status = 'auto_approved' THEN 'matched'
            ELSE 'mismatched'
        END,
        match_percentage = p_match_percentage
    WHERE id = p_snapshot_id;

    -- Update or insert daily history
    INSERT INTO visual_test_history (
        project_id,
        baseline_id,
        date,
        total_comparisons,
        matches,
        mismatches,
        auto_approved,
        avg_match_percentage,
        min_match_percentage,
        max_match_percentage
    ) VALUES (
        p_project_id,
        p_baseline_id,
        CURRENT_DATE,
        1,
        CASE WHEN v_status = 'auto_approved' THEN 1 ELSE 0 END,
        CASE WHEN v_status != 'auto_approved' THEN 1 ELSE 0 END,
        CASE WHEN v_status = 'auto_approved' THEN 1 ELSE 0 END,
        p_match_percentage,
        p_match_percentage,
        p_match_percentage
    )
    ON CONFLICT (project_id, baseline_id, date) DO UPDATE SET
        total_comparisons = visual_test_history.total_comparisons + 1,
        matches = visual_test_history.matches + CASE WHEN v_status = 'auto_approved' THEN 1 ELSE 0 END,
        mismatches = visual_test_history.mismatches + CASE WHEN v_status != 'auto_approved' THEN 1 ELSE 0 END,
        auto_approved = visual_test_history.auto_approved + CASE WHEN v_status = 'auto_approved' THEN 1 ELSE 0 END,
        avg_match_percentage = (
            (visual_test_history.avg_match_percentage * visual_test_history.total_comparisons + p_match_percentage) /
            (visual_test_history.total_comparisons + 1)
        ),
        min_match_percentage = LEAST(visual_test_history.min_match_percentage, p_match_percentage),
        max_match_percentage = GREATEST(visual_test_history.max_match_percentage, p_match_percentage),
        updated_at = NOW();

    RETURN v_comparison_id;
END;
$$ LANGUAGE plpgsql;

-- Function to approve a comparison
CREATE OR REPLACE FUNCTION approve_visual_comparison(
    p_comparison_id UUID,
    p_reviewed_by TEXT,
    p_notes TEXT DEFAULT NULL,
    p_update_baseline BOOLEAN DEFAULT false
) RETURNS void AS $$
DECLARE
    v_snapshot_id UUID;
    v_baseline_id UUID;
    v_project_id UUID;
BEGIN
    -- Get comparison details
    SELECT snapshot_id, baseline_id, project_id
    INTO v_snapshot_id, v_baseline_id, v_project_id
    FROM visual_comparisons
    WHERE id = p_comparison_id;

    -- Update comparison status
    UPDATE visual_comparisons
    SET
        status = 'manually_approved',
        reviewed_by = p_reviewed_by,
        reviewed_at = NOW(),
        review_notes = p_notes,
        updated_at = NOW()
    WHERE id = p_comparison_id;

    -- Update snapshot status
    UPDATE visual_snapshots
    SET comparison_status = 'approved'
    WHERE id = v_snapshot_id;

    -- Update daily history
    UPDATE visual_test_history
    SET
        manually_approved = manually_approved + 1,
        mismatches = GREATEST(0, mismatches - 1),
        updated_at = NOW()
    WHERE project_id = v_project_id
    AND (baseline_id = v_baseline_id OR (baseline_id IS NULL AND v_baseline_id IS NULL))
    AND date = CURRENT_DATE;

    -- Optionally update baseline with new snapshot
    IF p_update_baseline THEN
        UPDATE visual_baselines
        SET
            screenshot_url = (SELECT screenshot_url FROM visual_snapshots WHERE id = v_snapshot_id),
            screenshot_hash = (SELECT layout_hash FROM visual_snapshots WHERE id = v_snapshot_id),
            element_count = (SELECT element_count FROM visual_snapshots WHERE id = v_snapshot_id),
            layout_hash = (SELECT layout_hash FROM visual_snapshots WHERE id = v_snapshot_id),
            color_palette = (SELECT color_palette FROM visual_snapshots WHERE id = v_snapshot_id),
            captured_at = NOW()
        WHERE id = v_baseline_id;

        -- Record baseline update in history
        UPDATE visual_test_history
        SET
            baseline_updates = baseline_updates + 1,
            updated_at = NOW()
        WHERE project_id = v_project_id
        AND baseline_id = v_baseline_id
        AND date = CURRENT_DATE;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to get visual testing trends
CREATE OR REPLACE FUNCTION get_visual_test_trends(
    p_project_id UUID,
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE(
    date DATE,
    total_comparisons INTEGER,
    match_rate FLOAT,
    auto_approval_rate FLOAT,
    avg_match_percentage FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        h.date,
        h.total_comparisons,
        CASE WHEN h.total_comparisons > 0
            THEN (h.matches::FLOAT / h.total_comparisons) * 100
            ELSE 0
        END as match_rate,
        CASE WHEN h.total_comparisons > 0
            THEN (h.auto_approved::FLOAT / h.total_comparisons) * 100
            ELSE 0
        END as auto_approval_rate,
        h.avg_match_percentage
    FROM visual_test_history h
    WHERE h.project_id = p_project_id
    AND h.baseline_id IS NULL  -- Project-level aggregation
    AND h.date >= CURRENT_DATE - p_days
    ORDER BY h.date ASC;
END;
$$ LANGUAGE plpgsql;

-- Function to aggregate daily project-level stats
CREATE OR REPLACE FUNCTION aggregate_visual_test_daily(p_project_id UUID)
RETURNS void AS $$
BEGIN
    INSERT INTO visual_test_history (
        project_id,
        baseline_id,
        date,
        total_comparisons,
        matches,
        mismatches,
        auto_approved,
        manually_approved,
        rejected,
        pending,
        avg_match_percentage,
        min_match_percentage,
        max_match_percentage,
        changes_by_category,
        changes_by_severity,
        comparisons_by_browser
    )
    SELECT
        p_project_id,
        NULL as baseline_id,
        CURRENT_DATE,
        COUNT(*)::INTEGER as total_comparisons,
        COUNT(*) FILTER (WHERE status = 'auto_approved')::INTEGER as matches,
        COUNT(*) FILTER (WHERE status IN ('needs_review', 'rejected'))::INTEGER as mismatches,
        COUNT(*) FILTER (WHERE status = 'auto_approved')::INTEGER as auto_approved,
        COUNT(*) FILTER (WHERE status = 'manually_approved')::INTEGER as manually_approved,
        COUNT(*) FILTER (WHERE status = 'rejected')::INTEGER as rejected,
        COUNT(*) FILTER (WHERE status = 'pending')::INTEGER as pending,
        AVG(match_percentage)::FLOAT as avg_match_percentage,
        MIN(match_percentage)::FLOAT as min_match_percentage,
        MAX(match_percentage)::FLOAT as max_match_percentage,
        '{}'::JSONB as changes_by_category,
        '{}'::JSONB as changes_by_severity,
        '{}'::JSONB as comparisons_by_browser
    FROM visual_comparisons
    WHERE project_id = p_project_id
    AND created_at::DATE = CURRENT_DATE
    ON CONFLICT (project_id, date) WHERE baseline_id IS NULL
    DO UPDATE SET
        total_comparisons = EXCLUDED.total_comparisons,
        matches = EXCLUDED.matches,
        mismatches = EXCLUDED.mismatches,
        auto_approved = EXCLUDED.auto_approved,
        manually_approved = EXCLUDED.manually_approved,
        rejected = EXCLUDED.rejected,
        pending = EXCLUDED.pending,
        avg_match_percentage = EXCLUDED.avg_match_percentage,
        min_match_percentage = EXCLUDED.min_match_percentage,
        max_match_percentage = EXCLUDED.max_match_percentage,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SEED DATA: Default Viewport Presets
-- ============================================================================

-- Insert system viewport presets if they don't exist
INSERT INTO viewport_presets (organization_id, name, description, width, height, device_scale_factor, is_mobile, device_name, device_category, is_system, is_default, display_order)
SELECT
    (SELECT id FROM organizations ORDER BY created_at LIMIT 1),
    v.name,
    v.description,
    v.width,
    v.height,
    v.device_scale_factor,
    v.is_mobile,
    v.device_name,
    v.device_category,
    true,
    v.is_default,
    v.display_order
FROM (VALUES
    ('Desktop HD', 'Standard desktop display', 1920, 1080, 1.0, false, 'Desktop', 'desktop', true, 1),
    ('Desktop', 'Desktop at 1440p', 1440, 900, 1.0, false, 'Desktop', 'desktop', false, 2),
    ('Laptop', 'Standard laptop display', 1366, 768, 1.0, false, 'Laptop', 'laptop', false, 3),
    ('iPad Pro 12.9', 'iPad Pro 12.9" portrait', 1024, 1366, 2.0, true, 'iPad Pro 12.9"', 'tablet', false, 4),
    ('iPad', 'iPad portrait', 768, 1024, 2.0, true, 'iPad', 'tablet', false, 5),
    ('iPad Landscape', 'iPad landscape', 1024, 768, 2.0, true, 'iPad', 'tablet', false, 6),
    ('iPhone 14 Pro', 'iPhone 14 Pro', 393, 852, 3.0, true, 'iPhone 14 Pro', 'phone', false, 7),
    ('iPhone 14', 'iPhone 14', 390, 844, 3.0, true, 'iPhone 14', 'phone', false, 8),
    ('iPhone SE', 'iPhone SE', 375, 667, 2.0, true, 'iPhone SE', 'phone', false, 9),
    ('Pixel 7', 'Google Pixel 7', 412, 915, 2.625, true, 'Pixel 7', 'phone', false, 10),
    ('Galaxy S23', 'Samsung Galaxy S23', 360, 780, 3.0, true, 'Galaxy S23', 'phone', false, 11)
) AS v(name, description, width, height, device_scale_factor, is_mobile, device_name, device_category, is_default, display_order)
WHERE EXISTS (SELECT 1 FROM organizations LIMIT 1)
AND NOT EXISTS (
    SELECT 1 FROM viewport_presets
    WHERE is_system = true
    AND name = v.name
);

-- ============================================================================
-- ENABLE REALTIME
-- ============================================================================

-- Enable realtime for visual testing tables
DO $$
BEGIN
    -- Check if publication exists before altering
    IF EXISTS (SELECT 1 FROM pg_publication WHERE pubname = 'supabase_realtime') THEN
        ALTER PUBLICATION supabase_realtime ADD TABLE visual_snapshots;
        ALTER PUBLICATION supabase_realtime ADD TABLE visual_comparisons;
        ALTER PUBLICATION supabase_realtime ADD TABLE visual_changes;
    END IF;
EXCEPTION WHEN duplicate_object THEN
    -- Tables already added to publication, ignore
    NULL;
END $$;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE visual_baselines IS 'Reference screenshots and metadata for visual regression testing';
COMMENT ON TABLE visual_snapshots IS 'Full page state capture including screenshot, DOM, and performance metrics';
COMMENT ON TABLE visual_comparisons IS 'Results of comparing snapshots against baselines';
COMMENT ON TABLE visual_changes IS 'Granular change records with AI-powered analysis';
COMMENT ON TABLE viewport_presets IS 'Responsive testing viewport configurations';
COMMENT ON TABLE browser_matrix IS 'Cross-browser testing configurations per project';
COMMENT ON TABLE visual_test_history IS 'Daily aggregated metrics for trend analysis';

COMMENT ON COLUMN visual_snapshots.lcp_ms IS 'Largest Contentful Paint in milliseconds';
COMMENT ON COLUMN visual_snapshots.cls_score IS 'Cumulative Layout Shift score';
COMMENT ON COLUMN visual_snapshots.tti_ms IS 'Time to Interactive in milliseconds';
COMMENT ON COLUMN visual_changes.severity IS '0=info, 1=low, 2=medium, 3=high, 4=critical';

-- ============================================================================
-- COMPLETION
-- ============================================================================

SELECT 'Visual AI Enhanced tables created successfully!' as message;
