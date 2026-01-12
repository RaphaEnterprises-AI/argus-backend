-- Discovery Intelligence Platform Migration
-- Adds comprehensive discovery capabilities with AI-driven insights,
-- cross-project learning via pgvector, and time-travel support
-- Migration: 20260111_discovery_intelligence.sql

-- Ensure pgvector extension is enabled (for cross-project pattern learning)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- DISCOVERY SESSIONS TABLE
-- ============================================================================
-- Core table for tracking application discovery sessions
CREATE TABLE IF NOT EXISTS discovery_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Basic session info
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'running', 'completed', 'failed', 'cancelled', 'paused'
    )),
    start_url TEXT NOT NULL,

    -- Discovery configuration
    mode TEXT NOT NULL DEFAULT 'full' CHECK (mode IN (
        'full', 'incremental', 'focused', 'quick', 'deep'
    )),
    strategy TEXT NOT NULL DEFAULT 'breadth_first' CHECK (strategy IN (
        'breadth_first', 'depth_first', 'priority_based', 'ai_guided'
    )),
    config JSONB DEFAULT '{}',

    -- Scope limits
    max_pages INTEGER DEFAULT 100,
    max_depth INTEGER DEFAULT 5,
    include_patterns TEXT[] DEFAULT '{}',
    exclude_patterns TEXT[] DEFAULT '{}',
    focus_areas TEXT[] DEFAULT '{}',

    -- Progress tracking
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
    current_page TEXT,
    pages_discovered INTEGER DEFAULT 0,
    pages_analyzed INTEGER DEFAULT 0,

    -- Discovery results summary
    page_graph JSONB DEFAULT '{}',
    coverage_summary JSONB DEFAULT '{}',

    -- AI-generated insights
    quality_score NUMERIC(5,2) DEFAULT 0 CHECK (quality_score >= 0 AND quality_score <= 100),
    insights JSONB DEFAULT '[]',
    patterns_detected JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',

    -- Comparison support
    compared_to_session_id UUID REFERENCES discovery_sessions(id) ON DELETE SET NULL,
    diff_summary JSONB DEFAULT '{}',

    -- AI usage tracking
    ai_tokens_used INTEGER DEFAULT 0,
    ai_cost_usd NUMERIC(10,4) DEFAULT 0,

    -- Error tracking
    error_message TEXT,
    error_details JSONB,

    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add columns to existing discovery_sessions if table already exists
DO $$
BEGIN
    -- Mode and strategy
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'mode') THEN
        ALTER TABLE discovery_sessions ADD COLUMN mode TEXT NOT NULL DEFAULT 'full' CHECK (mode IN ('full', 'incremental', 'focused', 'quick', 'deep'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'strategy') THEN
        ALTER TABLE discovery_sessions ADD COLUMN strategy TEXT NOT NULL DEFAULT 'breadth_first' CHECK (strategy IN ('breadth_first', 'depth_first', 'priority_based', 'ai_guided'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'config') THEN
        ALTER TABLE discovery_sessions ADD COLUMN config JSONB DEFAULT '{}';
    END IF;

    -- Scope limits
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'max_pages') THEN
        ALTER TABLE discovery_sessions ADD COLUMN max_pages INTEGER DEFAULT 100;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'max_depth') THEN
        ALTER TABLE discovery_sessions ADD COLUMN max_depth INTEGER DEFAULT 5;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'include_patterns') THEN
        ALTER TABLE discovery_sessions ADD COLUMN include_patterns TEXT[] DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'exclude_patterns') THEN
        ALTER TABLE discovery_sessions ADD COLUMN exclude_patterns TEXT[] DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'focus_areas') THEN
        ALTER TABLE discovery_sessions ADD COLUMN focus_areas TEXT[] DEFAULT '{}';
    END IF;

    -- Progress tracking
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'progress_percentage') THEN
        ALTER TABLE discovery_sessions ADD COLUMN progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'current_page') THEN
        ALTER TABLE discovery_sessions ADD COLUMN current_page TEXT;
    END IF;

    -- Results summary
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'page_graph') THEN
        ALTER TABLE discovery_sessions ADD COLUMN page_graph JSONB DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'coverage_summary') THEN
        ALTER TABLE discovery_sessions ADD COLUMN coverage_summary JSONB DEFAULT '{}';
    END IF;

    -- AI insights
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'quality_score') THEN
        ALTER TABLE discovery_sessions ADD COLUMN quality_score NUMERIC(5,2) DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'insights') THEN
        ALTER TABLE discovery_sessions ADD COLUMN insights JSONB DEFAULT '[]';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'patterns_detected') THEN
        ALTER TABLE discovery_sessions ADD COLUMN patterns_detected JSONB DEFAULT '[]';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'recommendations') THEN
        ALTER TABLE discovery_sessions ADD COLUMN recommendations JSONB DEFAULT '[]';
    END IF;

    -- Comparison
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'compared_to_session_id') THEN
        ALTER TABLE discovery_sessions ADD COLUMN compared_to_session_id UUID REFERENCES discovery_sessions(id) ON DELETE SET NULL;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'diff_summary') THEN
        ALTER TABLE discovery_sessions ADD COLUMN diff_summary JSONB DEFAULT '{}';
    END IF;

    -- AI tracking
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'ai_tokens_used') THEN
        ALTER TABLE discovery_sessions ADD COLUMN ai_tokens_used INTEGER DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovery_sessions' AND column_name = 'ai_cost_usd') THEN
        ALTER TABLE discovery_sessions ADD COLUMN ai_cost_usd NUMERIC(10,4) DEFAULT 0;
    END IF;
END $$;

-- ============================================================================
-- DISCOVERED PAGES TABLE
-- ============================================================================
-- Stores detailed information about each discovered page
CREATE TABLE IF NOT EXISTS discovered_pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discovery_session_id UUID NOT NULL REFERENCES discovery_sessions(id) ON DELETE CASCADE,

    -- URL information
    url TEXT NOT NULL,
    url_hash TEXT,
    canonical_url TEXT,
    title TEXT,

    -- Page classification
    page_type TEXT DEFAULT 'unknown' CHECK (page_type IN (
        'landing', 'form', 'list', 'detail', 'dashboard', 'settings',
        'auth', 'error', 'content', 'search', 'unknown'
    )),
    category TEXT,

    -- Screenshot and snapshot
    screenshot_url TEXT,
    dom_snapshot_url TEXT,

    -- Page metrics
    element_count INTEGER DEFAULT 0,
    interactive_element_count INTEGER DEFAULT 0,
    form_count INTEGER DEFAULT 0,
    link_count INTEGER DEFAULT 0,

    -- Link graph
    outgoing_links TEXT[] DEFAULT '{}',
    incoming_links TEXT[] DEFAULT '{}',

    -- Performance metrics
    load_time_ms INTEGER,
    lcp_ms INTEGER,          -- Largest Contentful Paint
    cls_score NUMERIC(5,4),  -- Cumulative Layout Shift
    tti_ms INTEGER,          -- Time to Interactive

    -- Page characteristics
    requires_auth BOOLEAN DEFAULT FALSE,
    has_dynamic_content BOOLEAN DEFAULT FALSE,

    -- Discovery metadata
    depth_from_start INTEGER DEFAULT 0,
    discovered_at TIMESTAMPTZ DEFAULT NOW(),

    -- AI analysis scores
    importance_score NUMERIC(5,2) DEFAULT 0 CHECK (importance_score >= 0 AND importance_score <= 100),
    risk_score NUMERIC(5,2) DEFAULT 0 CHECK (risk_score >= 0 AND risk_score <= 100),
    coverage_score NUMERIC(5,2) DEFAULT 0 CHECK (coverage_score >= 0 AND coverage_score <= 100),

    -- Element summary
    elements_summary JSONB DEFAULT '{}',

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add columns to existing discovered_pages if table already exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'url_hash') THEN
        ALTER TABLE discovered_pages ADD COLUMN url_hash TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'canonical_url') THEN
        ALTER TABLE discovered_pages ADD COLUMN canonical_url TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'category') THEN
        ALTER TABLE discovered_pages ADD COLUMN category TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'dom_snapshot_url') THEN
        ALTER TABLE discovered_pages ADD COLUMN dom_snapshot_url TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'interactive_element_count') THEN
        ALTER TABLE discovered_pages ADD COLUMN interactive_element_count INTEGER DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'outgoing_links') THEN
        ALTER TABLE discovered_pages ADD COLUMN outgoing_links TEXT[] DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'incoming_links') THEN
        ALTER TABLE discovered_pages ADD COLUMN incoming_links TEXT[] DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'load_time_ms') THEN
        ALTER TABLE discovered_pages ADD COLUMN load_time_ms INTEGER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'lcp_ms') THEN
        ALTER TABLE discovered_pages ADD COLUMN lcp_ms INTEGER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'cls_score') THEN
        ALTER TABLE discovered_pages ADD COLUMN cls_score NUMERIC(5,4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'tti_ms') THEN
        ALTER TABLE discovered_pages ADD COLUMN tti_ms INTEGER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'requires_auth') THEN
        ALTER TABLE discovered_pages ADD COLUMN requires_auth BOOLEAN DEFAULT FALSE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'has_dynamic_content') THEN
        ALTER TABLE discovered_pages ADD COLUMN has_dynamic_content BOOLEAN DEFAULT FALSE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'depth_from_start') THEN
        ALTER TABLE discovered_pages ADD COLUMN depth_from_start INTEGER DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'importance_score') THEN
        ALTER TABLE discovered_pages ADD COLUMN importance_score NUMERIC(5,2) DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'risk_score') THEN
        ALTER TABLE discovered_pages ADD COLUMN risk_score NUMERIC(5,2) DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'coverage_score') THEN
        ALTER TABLE discovered_pages ADD COLUMN coverage_score NUMERIC(5,2) DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_pages' AND column_name = 'elements_summary') THEN
        ALTER TABLE discovered_pages ADD COLUMN elements_summary JSONB DEFAULT '{}';
    END IF;
END $$;

-- ============================================================================
-- DISCOVERED ELEMENTS TABLE (NEW)
-- ============================================================================
-- Stores individual interactive elements discovered on pages
CREATE TABLE IF NOT EXISTS discovered_elements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id UUID NOT NULL REFERENCES discovered_pages(id) ON DELETE CASCADE,
    discovery_session_id UUID NOT NULL REFERENCES discovery_sessions(id) ON DELETE CASCADE,

    -- Element identification
    selector TEXT NOT NULL,
    xpath TEXT,
    tag_name TEXT NOT NULL,

    -- Classification
    category TEXT NOT NULL CHECK (category IN (
        'button', 'link', 'input', 'select', 'checkbox', 'radio',
        'textarea', 'form', 'navigation', 'modal', 'menu',
        'table', 'list', 'image', 'video', 'custom', 'other'
    )),
    purpose TEXT,
    label TEXT,

    -- Visual information
    bounds JSONB DEFAULT '{}',  -- {x, y, width, height}
    is_visible BOOLEAN DEFAULT TRUE,
    is_enabled BOOLEAN DEFAULT TRUE,
    is_required BOOLEAN DEFAULT FALSE,

    -- Accessibility
    aria_label TEXT,
    role TEXT,

    -- AI analysis
    importance_score NUMERIC(5,2) DEFAULT 0 CHECK (importance_score >= 0 AND importance_score <= 100),
    stability_score NUMERIC(5,2) DEFAULT 0 CHECK (stability_score >= 0 AND stability_score <= 100),

    -- Selector resilience
    alternative_selectors TEXT[] DEFAULT '{}',

    -- Raw HTML info
    html_attributes JSONB DEFAULT '{}',

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- DISCOVERED FLOWS TABLE
-- ============================================================================
-- Stores user flows/journeys identified during discovery
CREATE TABLE IF NOT EXISTS discovered_flows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discovery_session_id UUID NOT NULL REFERENCES discovery_sessions(id) ON DELETE CASCADE,

    -- Flow identification
    name TEXT NOT NULL,
    description TEXT,
    flow_type TEXT NOT NULL CHECK (flow_type IN (
        'authentication', 'registration', 'checkout', 'search',
        'crud', 'navigation', 'form_submission', 'custom'
    )),
    category TEXT,

    -- Flow steps
    steps JSONB NOT NULL DEFAULT '[]',
    page_ids UUID[] DEFAULT '{}',

    -- Entry/exit points
    entry_points JSONB DEFAULT '[]',
    exit_points JSONB DEFAULT '[]',

    -- Validation criteria
    success_criteria JSONB DEFAULT '{}',
    failure_indicators JSONB DEFAULT '{}',

    -- AI analysis scores
    complexity_score NUMERIC(5,2) DEFAULT 0 CHECK (complexity_score >= 0 AND complexity_score <= 100),
    business_value_score NUMERIC(5,2) DEFAULT 0 CHECK (business_value_score >= 0 AND business_value_score <= 100),
    confidence_score NUMERIC(5,2) DEFAULT 0 CHECK (confidence_score >= 0 AND confidence_score <= 100),

    -- Validation status
    validated BOOLEAN DEFAULT FALSE,
    last_validated_at TIMESTAMPTZ,
    validation_result JSONB,
    validation_error TEXT,

    -- Test generation
    auto_generated_test JSONB DEFAULT '{}',

    -- Similar flows detection
    similar_flows JSONB DEFAULT '[]',

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add columns to existing discovered_flows if table already exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'category') THEN
        ALTER TABLE discovered_flows ADD COLUMN category TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'page_ids') THEN
        ALTER TABLE discovered_flows ADD COLUMN page_ids UUID[] DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'success_criteria') THEN
        ALTER TABLE discovered_flows ADD COLUMN success_criteria JSONB DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'failure_indicators') THEN
        ALTER TABLE discovered_flows ADD COLUMN failure_indicators JSONB DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'complexity_score') THEN
        ALTER TABLE discovered_flows ADD COLUMN complexity_score NUMERIC(5,2) DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'business_value_score') THEN
        ALTER TABLE discovered_flows ADD COLUMN business_value_score NUMERIC(5,2) DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'confidence_score') THEN
        ALTER TABLE discovered_flows ADD COLUMN confidence_score NUMERIC(5,2) DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'validated') THEN
        ALTER TABLE discovered_flows ADD COLUMN validated BOOLEAN DEFAULT FALSE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'last_validated_at') THEN
        ALTER TABLE discovered_flows ADD COLUMN last_validated_at TIMESTAMPTZ;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'validation_result') THEN
        ALTER TABLE discovered_flows ADD COLUMN validation_result JSONB;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'validation_error') THEN
        ALTER TABLE discovered_flows ADD COLUMN validation_error TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'auto_generated_test') THEN
        ALTER TABLE discovered_flows ADD COLUMN auto_generated_test JSONB DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'discovered_flows' AND column_name = 'similar_flows') THEN
        ALTER TABLE discovered_flows ADD COLUMN similar_flows JSONB DEFAULT '[]';
    END IF;
END $$;

-- ============================================================================
-- DISCOVERY PATTERNS TABLE (NEW - for pgvector cross-project learning)
-- ============================================================================
-- Stores reusable patterns learned across projects for smarter discovery
CREATE TABLE IF NOT EXISTS discovery_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Pattern identification
    pattern_type TEXT NOT NULL CHECK (pattern_type IN (
        'page_layout', 'navigation', 'form', 'authentication',
        'error_handling', 'loading_state', 'modal', 'list_view',
        'detail_view', 'search', 'filter', 'pagination', 'custom'
    )),
    pattern_name TEXT NOT NULL,
    pattern_signature TEXT NOT NULL,

    -- Vector embedding for semantic similarity search
    embedding vector(1536),

    -- Pattern data
    pattern_data JSONB NOT NULL DEFAULT '{}',

    -- Usage statistics
    times_seen INTEGER DEFAULT 1,
    projects_seen INTEGER DEFAULT 1,

    -- Effectiveness metrics
    test_success_rate NUMERIC(5,2) DEFAULT 0 CHECK (test_success_rate >= 0 AND test_success_rate <= 100),
    self_heal_success_rate NUMERIC(5,2) DEFAULT 0 CHECK (self_heal_success_rate >= 0 AND self_heal_success_rate <= 100),

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Unique constraint on pattern signature
    UNIQUE(pattern_signature)
);

-- ============================================================================
-- DISCOVERY HISTORY TABLE (NEW - for time-travel support)
-- ============================================================================
-- Stores historical snapshots for tracking changes over time
CREATE TABLE IF NOT EXISTS discovery_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Snapshot timing
    snapshot_date DATE NOT NULL,
    session_id UUID REFERENCES discovery_sessions(id) ON DELETE SET NULL,

    -- Counts
    total_pages INTEGER NOT NULL DEFAULT 0,
    total_elements INTEGER NOT NULL DEFAULT 0,
    total_flows INTEGER NOT NULL DEFAULT 0,

    -- Change tracking
    pages_added INTEGER DEFAULT 0,
    pages_removed INTEGER DEFAULT 0,
    elements_added INTEGER DEFAULT 0,
    elements_removed INTEGER DEFAULT 0,
    flows_added INTEGER DEFAULT 0,
    flows_removed INTEGER DEFAULT 0,

    -- Quality metrics
    coverage_score NUMERIC(5,2) DEFAULT 0 CHECK (coverage_score >= 0 AND coverage_score <= 100),
    quality_score NUMERIC(5,2) DEFAULT 0 CHECK (quality_score >= 0 AND quality_score <= 100),

    -- Full snapshot data (optional, for complete reconstruction)
    snapshot_data JSONB,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- One snapshot per project per day
    UNIQUE(project_id, snapshot_date)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Discovery Sessions indexes
CREATE INDEX IF NOT EXISTS idx_discovery_sessions_project ON discovery_sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_discovery_sessions_status ON discovery_sessions(status);
CREATE INDEX IF NOT EXISTS idx_discovery_sessions_mode ON discovery_sessions(mode);
CREATE INDEX IF NOT EXISTS idx_discovery_sessions_created ON discovery_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_discovery_sessions_quality ON discovery_sessions(quality_score DESC);

-- Discovered Pages indexes
CREATE INDEX IF NOT EXISTS idx_discovered_pages_session ON discovered_pages(discovery_session_id);
CREATE INDEX IF NOT EXISTS idx_discovered_pages_url ON discovered_pages(url);
CREATE INDEX IF NOT EXISTS idx_discovered_pages_url_hash ON discovered_pages(url_hash);
CREATE INDEX IF NOT EXISTS idx_discovered_pages_type ON discovered_pages(page_type);
CREATE INDEX IF NOT EXISTS idx_discovered_pages_importance ON discovered_pages(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_discovered_pages_risk ON discovered_pages(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_discovered_pages_depth ON discovered_pages(depth_from_start);

-- Discovered Elements indexes
CREATE INDEX IF NOT EXISTS idx_discovered_elements_page ON discovered_elements(page_id);
CREATE INDEX IF NOT EXISTS idx_discovered_elements_session ON discovered_elements(discovery_session_id);
CREATE INDEX IF NOT EXISTS idx_discovered_elements_category ON discovered_elements(category);
CREATE INDEX IF NOT EXISTS idx_discovered_elements_selector ON discovered_elements(selector);
CREATE INDEX IF NOT EXISTS idx_discovered_elements_importance ON discovered_elements(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_discovered_elements_stability ON discovered_elements(stability_score DESC);

-- Discovered Flows indexes
CREATE INDEX IF NOT EXISTS idx_discovered_flows_session ON discovered_flows(discovery_session_id);
CREATE INDEX IF NOT EXISTS idx_discovered_flows_type ON discovered_flows(flow_type);
CREATE INDEX IF NOT EXISTS idx_discovered_flows_validated ON discovered_flows(validated);
CREATE INDEX IF NOT EXISTS idx_discovered_flows_complexity ON discovered_flows(complexity_score DESC);
CREATE INDEX IF NOT EXISTS idx_discovered_flows_business_value ON discovered_flows(business_value_score DESC);

-- Discovery Patterns indexes (including vector index for semantic search)
CREATE INDEX IF NOT EXISTS idx_discovery_patterns_type ON discovery_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_discovery_patterns_name ON discovery_patterns(pattern_name);
CREATE INDEX IF NOT EXISTS idx_discovery_patterns_signature ON discovery_patterns(pattern_signature);
CREATE INDEX IF NOT EXISTS idx_discovery_patterns_times_seen ON discovery_patterns(times_seen DESC);
CREATE INDEX IF NOT EXISTS idx_discovery_patterns_embedding ON discovery_patterns
    USING hnsw (embedding vector_cosine_ops);

-- Discovery History indexes
CREATE INDEX IF NOT EXISTS idx_discovery_history_project ON discovery_history(project_id);
CREATE INDEX IF NOT EXISTS idx_discovery_history_date ON discovery_history(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_discovery_history_project_date ON discovery_history(project_id, snapshot_date DESC);

-- ============================================================================
-- TRIGGERS FOR UPDATED_AT
-- ============================================================================

-- Ensure update_updated_at_column function exists (may already exist from other migrations)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Discovery Sessions trigger
DROP TRIGGER IF EXISTS update_discovery_sessions_updated_at ON discovery_sessions;
CREATE TRIGGER update_discovery_sessions_updated_at
    BEFORE UPDATE ON discovery_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Discovered Pages trigger
DROP TRIGGER IF EXISTS update_discovered_pages_updated_at ON discovered_pages;
CREATE TRIGGER update_discovered_pages_updated_at
    BEFORE UPDATE ON discovered_pages
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Discovered Elements trigger
DROP TRIGGER IF EXISTS update_discovered_elements_updated_at ON discovered_elements;
CREATE TRIGGER update_discovered_elements_updated_at
    BEFORE UPDATE ON discovered_elements
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Discovered Flows trigger
DROP TRIGGER IF EXISTS update_discovered_flows_updated_at ON discovered_flows;
CREATE TRIGGER update_discovered_flows_updated_at
    BEFORE UPDATE ON discovered_flows
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Discovery Patterns trigger
DROP TRIGGER IF EXISTS update_discovery_patterns_updated_at ON discovery_patterns;
CREATE TRIGGER update_discovery_patterns_updated_at
    BEFORE UPDATE ON discovery_patterns
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE discovery_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovered_pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovered_elements ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovered_flows ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovery_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE discovery_history ENABLE ROW LEVEL SECURITY;

-- Policies (allow all access for now - can be tightened later based on organization/project membership)
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovery_sessions;
CREATE POLICY "Enable all access for authenticated users" ON discovery_sessions FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovered_pages;
CREATE POLICY "Enable all access for authenticated users" ON discovered_pages FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovered_elements;
CREATE POLICY "Enable all access for authenticated users" ON discovered_elements FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovered_flows;
CREATE POLICY "Enable all access for authenticated users" ON discovered_flows FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovery_patterns;
CREATE POLICY "Enable all access for authenticated users" ON discovery_patterns FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON discovery_history;
CREATE POLICY "Enable all access for authenticated users" ON discovery_history FOR ALL USING (true);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to search similar discovery patterns using vector similarity
CREATE OR REPLACE FUNCTION search_similar_discovery_patterns(
    query_embedding vector(1536),
    pattern_type_filter TEXT DEFAULT NULL,
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    pattern_type TEXT,
    pattern_name TEXT,
    pattern_data JSONB,
    times_seen INTEGER,
    test_success_rate NUMERIC,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dp.id,
        dp.pattern_type,
        dp.pattern_name,
        dp.pattern_data,
        dp.times_seen,
        dp.test_success_rate,
        1 - (dp.embedding <=> query_embedding) AS similarity
    FROM discovery_patterns dp
    WHERE dp.embedding IS NOT NULL
      AND (pattern_type_filter IS NULL OR dp.pattern_type = pattern_type_filter)
      AND 1 - (dp.embedding <=> query_embedding) > match_threshold
    ORDER BY dp.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function to calculate and store daily discovery history snapshot
CREATE OR REPLACE FUNCTION create_discovery_snapshot(p_project_id UUID, p_session_id UUID DEFAULT NULL)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_snapshot_id UUID;
    v_total_pages INTEGER;
    v_total_elements INTEGER;
    v_total_flows INTEGER;
    v_prev_pages INTEGER;
    v_prev_elements INTEGER;
    v_prev_flows INTEGER;
    v_coverage_score NUMERIC;
    v_quality_score NUMERIC;
BEGIN
    -- Get current counts
    SELECT COUNT(*) INTO v_total_pages
    FROM discovered_pages dp
    JOIN discovery_sessions ds ON dp.discovery_session_id = ds.id
    WHERE ds.project_id = p_project_id;

    SELECT COUNT(*) INTO v_total_elements
    FROM discovered_elements de
    JOIN discovery_sessions ds ON de.discovery_session_id = ds.id
    WHERE ds.project_id = p_project_id;

    SELECT COUNT(*) INTO v_total_flows
    FROM discovered_flows df
    JOIN discovery_sessions ds ON df.discovery_session_id = ds.id
    WHERE ds.project_id = p_project_id;

    -- Get previous snapshot counts
    SELECT total_pages, total_elements, total_flows
    INTO v_prev_pages, v_prev_elements, v_prev_flows
    FROM discovery_history
    WHERE project_id = p_project_id
    ORDER BY snapshot_date DESC
    LIMIT 1;

    -- Get latest session quality score
    SELECT quality_score INTO v_quality_score
    FROM discovery_sessions
    WHERE project_id = p_project_id
    ORDER BY created_at DESC
    LIMIT 1;

    -- Default values if no previous snapshot
    v_prev_pages := COALESCE(v_prev_pages, 0);
    v_prev_elements := COALESCE(v_prev_elements, 0);
    v_prev_flows := COALESCE(v_prev_flows, 0);
    v_quality_score := COALESCE(v_quality_score, 0);

    -- Calculate coverage score (simple heuristic)
    v_coverage_score := LEAST(100, (v_total_pages * 2) + (v_total_flows * 5));

    -- Insert or update snapshot
    INSERT INTO discovery_history (
        project_id,
        snapshot_date,
        session_id,
        total_pages,
        total_elements,
        total_flows,
        pages_added,
        pages_removed,
        elements_added,
        elements_removed,
        flows_added,
        flows_removed,
        coverage_score,
        quality_score
    ) VALUES (
        p_project_id,
        CURRENT_DATE,
        p_session_id,
        v_total_pages,
        v_total_elements,
        v_total_flows,
        GREATEST(0, v_total_pages - v_prev_pages),
        GREATEST(0, v_prev_pages - v_total_pages),
        GREATEST(0, v_total_elements - v_prev_elements),
        GREATEST(0, v_prev_elements - v_total_elements),
        GREATEST(0, v_total_flows - v_prev_flows),
        GREATEST(0, v_prev_flows - v_total_flows),
        v_coverage_score,
        v_quality_score
    )
    ON CONFLICT (project_id, snapshot_date) DO UPDATE SET
        session_id = COALESCE(EXCLUDED.session_id, discovery_history.session_id),
        total_pages = EXCLUDED.total_pages,
        total_elements = EXCLUDED.total_elements,
        total_flows = EXCLUDED.total_flows,
        pages_added = EXCLUDED.pages_added,
        pages_removed = EXCLUDED.pages_removed,
        elements_added = EXCLUDED.elements_added,
        elements_removed = EXCLUDED.elements_removed,
        flows_added = EXCLUDED.flows_added,
        flows_removed = EXCLUDED.flows_removed,
        coverage_score = EXCLUDED.coverage_score,
        quality_score = EXCLUDED.quality_score
    RETURNING id INTO v_snapshot_id;

    RETURN v_snapshot_id;
END;
$$;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE discovery_sessions IS 'Stores discovery session configurations and results for application exploration';
COMMENT ON TABLE discovered_pages IS 'Stores individual pages discovered during exploration with metadata and AI analysis';
COMMENT ON TABLE discovered_elements IS 'Stores interactive elements found on discovered pages with selectors and accessibility info';
COMMENT ON TABLE discovered_flows IS 'Stores user journeys and flows identified during discovery';
COMMENT ON TABLE discovery_patterns IS 'Cross-project pattern library with vector embeddings for smart discovery learning';
COMMENT ON TABLE discovery_history IS 'Historical snapshots of discovery state for time-travel analysis';

COMMENT ON FUNCTION search_similar_discovery_patterns IS 'Search for similar discovery patterns using vector similarity (pgvector)';
COMMENT ON FUNCTION create_discovery_snapshot IS 'Create or update a daily snapshot of discovery state for a project';
