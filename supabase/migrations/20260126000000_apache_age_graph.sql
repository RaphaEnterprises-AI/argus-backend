-- ============================================================================
-- Apache AGE Knowledge Graph Layer (CONDITIONAL)
-- Enables multi-hop reasoning about relationships between tests, failures,
-- code changes, and healing patterns
-- ============================================================================
-- Migration: 20260126000000_apache_age_graph.sql
--
-- NOTE: This migration is CONDITIONAL. It only runs if Apache AGE extension
-- is available. On Supabase managed PostgreSQL, AGE is not available.
-- The Cognee knowledge layer (RAP-132) provides equivalent functionality
-- for environments without AGE.
-- ============================================================================

-- Check if AGE is available before doing anything
DO $$
BEGIN
    -- Check if age extension is available in pg_available_extensions
    IF EXISTS (
        SELECT 1 FROM pg_available_extensions WHERE name = 'age'
    ) THEN
        RAISE NOTICE 'Apache AGE is available - installing extension';
        EXECUTE 'CREATE EXTENSION IF NOT EXISTS age';
        EXECUTE 'LOAD ''age''';
        EXECUTE 'SET search_path = ag_catalog, "$user", public';

        -- Create the knowledge graph
        BEGIN
            PERFORM ag_catalog.create_graph('argus_knowledge');
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Graph argus_knowledge may already exist: %', SQLERRM;
        END;
    ELSE
        RAISE NOTICE 'Apache AGE extension not available - skipping AGE-specific setup. Using Cognee knowledge layer instead.';
    END IF;
END $$;

-- ============================================================================
-- The following tables work as metadata tracking regardless of AGE availability.
-- They can be used by Cognee or other graph implementations.
-- ============================================================================

-- Graph Metadata Table - tracks graph schema, statistics, and versioning
CREATE TABLE IF NOT EXISTS graph_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_name TEXT NOT NULL DEFAULT 'argus_knowledge',
    schema_version INTEGER NOT NULL DEFAULT 1,
    total_vertices INTEGER DEFAULT 0,
    total_edges INTEGER DEFAULT 0,
    last_compaction_at TIMESTAMPTZ,
    vertex_labels TEXT[] DEFAULT ARRAY[
        'Test', 'Selector', 'Failure', 'CodeChange',
        'HealingPattern', 'Page', 'Project'
    ],
    edge_types TEXT[] DEFAULT ARRAY[
        'USES', 'TARGETS', 'BELONGS_TO', 'BROKE', 'AFFECTED',
        'CAUSED', 'MODIFIED', 'FIXES', 'REPLACES', 'ON',
        'DEPENDS_ON', 'SIMILAR_TO'
    ],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert initial metadata
INSERT INTO graph_metadata (graph_name, schema_version)
VALUES ('argus_knowledge', 1)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- Graph Entity Mapping Tables
-- Maps relational IDs to graph vertex IDs (works with AGE or Cognee)
-- ============================================================================

-- Test entity mapping
CREATE TABLE IF NOT EXISTS graph_test_vertices (
    test_id UUID PRIMARY KEY REFERENCES tests(id) ON DELETE CASCADE,
    vertex_id BIGINT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Selector entity mapping
CREATE TABLE IF NOT EXISTS graph_selector_vertices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    selector_text TEXT NOT NULL,
    vertex_id BIGINT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(selector_text)
);

-- Failure entity mapping
CREATE TABLE IF NOT EXISTS graph_failure_vertices (
    failure_id UUID PRIMARY KEY,
    test_run_id UUID REFERENCES test_runs(id) ON DELETE CASCADE,
    vertex_id BIGINT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Code change entity mapping
CREATE TABLE IF NOT EXISTS graph_code_change_vertices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    commit_sha TEXT NOT NULL,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    vertex_id BIGINT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(project_id, commit_sha)
);

-- Healing pattern entity mapping
CREATE TABLE IF NOT EXISTS graph_healing_pattern_vertices (
    pattern_id UUID PRIMARY KEY REFERENCES healing_patterns(id) ON DELETE CASCADE,
    vertex_id BIGINT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Page entity mapping
CREATE TABLE IF NOT EXISTS graph_page_vertices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_url TEXT NOT NULL,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    vertex_id BIGINT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(project_id, page_url)
);

-- Project entity mapping
CREATE TABLE IF NOT EXISTS graph_project_vertices (
    project_id UUID PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
    vertex_id BIGINT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Edge Mapping Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS graph_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    edge_id BIGINT NOT NULL,
    edge_type TEXT NOT NULL,
    from_vertex_id BIGINT NOT NULL,
    to_vertex_id BIGINT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_graph_test_vertices_vertex_id ON graph_test_vertices(vertex_id);
CREATE INDEX IF NOT EXISTS idx_graph_selector_vertices_vertex_id ON graph_selector_vertices(vertex_id);
CREATE INDEX IF NOT EXISTS idx_graph_selector_vertices_text ON graph_selector_vertices(selector_text);
CREATE INDEX IF NOT EXISTS idx_graph_failure_vertices_vertex_id ON graph_failure_vertices(vertex_id);
CREATE INDEX IF NOT EXISTS idx_graph_failure_vertices_test_run ON graph_failure_vertices(test_run_id);
CREATE INDEX IF NOT EXISTS idx_graph_code_change_vertices_vertex_id ON graph_code_change_vertices(vertex_id);
CREATE INDEX IF NOT EXISTS idx_graph_code_change_vertices_commit ON graph_code_change_vertices(commit_sha);
CREATE INDEX IF NOT EXISTS idx_graph_healing_pattern_vertices_vertex_id ON graph_healing_pattern_vertices(vertex_id);
CREATE INDEX IF NOT EXISTS idx_graph_page_vertices_vertex_id ON graph_page_vertices(vertex_id);
CREATE INDEX IF NOT EXISTS idx_graph_page_vertices_url ON graph_page_vertices(page_url);
CREATE INDEX IF NOT EXISTS idx_graph_project_vertices_vertex_id ON graph_project_vertices(vertex_id);

CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_graph_edges_from ON graph_edges(from_vertex_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_to ON graph_edges(to_vertex_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_edge_id ON graph_edges(edge_id);

-- ============================================================================
-- Helper Functions (work without AGE)
-- ============================================================================

-- Function to update graph statistics
CREATE OR REPLACE FUNCTION update_graph_stats()
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    v_vertex_count INTEGER;
    v_edge_count INTEGER;
BEGIN
    SELECT
        (SELECT COUNT(*) FROM graph_test_vertices) +
        (SELECT COUNT(*) FROM graph_selector_vertices) +
        (SELECT COUNT(*) FROM graph_failure_vertices) +
        (SELECT COUNT(*) FROM graph_code_change_vertices) +
        (SELECT COUNT(*) FROM graph_healing_pattern_vertices) +
        (SELECT COUNT(*) FROM graph_page_vertices) +
        (SELECT COUNT(*) FROM graph_project_vertices)
    INTO v_vertex_count;

    SELECT COUNT(*) INTO v_edge_count FROM graph_edges;

    UPDATE graph_metadata
    SET total_vertices = v_vertex_count,
        total_edges = v_edge_count,
        updated_at = NOW()
    WHERE graph_name = 'argus_knowledge';
END;
$$;

-- Function to get vertex by ID from any mapping table
CREATE OR REPLACE FUNCTION get_vertex_info(p_vertex_id BIGINT)
RETURNS TABLE (
    vertex_type TEXT,
    entity_id UUID,
    properties JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 'Test'::TEXT, test_id, graph_test_vertices.properties
    FROM graph_test_vertices
    WHERE vertex_id = p_vertex_id;
    IF FOUND THEN RETURN; END IF;

    RETURN QUERY
    SELECT 'Selector'::TEXT, id, graph_selector_vertices.properties
    FROM graph_selector_vertices
    WHERE vertex_id = p_vertex_id;
    IF FOUND THEN RETURN; END IF;

    RETURN QUERY
    SELECT 'Failure'::TEXT, failure_id, graph_failure_vertices.properties
    FROM graph_failure_vertices
    WHERE vertex_id = p_vertex_id;
    IF FOUND THEN RETURN; END IF;

    RETURN QUERY
    SELECT 'CodeChange'::TEXT, id, graph_code_change_vertices.properties
    FROM graph_code_change_vertices
    WHERE vertex_id = p_vertex_id;
    IF FOUND THEN RETURN; END IF;

    RETURN QUERY
    SELECT 'HealingPattern'::TEXT, pattern_id, graph_healing_pattern_vertices.properties
    FROM graph_healing_pattern_vertices
    WHERE vertex_id = p_vertex_id;
    IF FOUND THEN RETURN; END IF;

    RETURN QUERY
    SELECT 'Page'::TEXT, id, graph_page_vertices.properties
    FROM graph_page_vertices
    WHERE vertex_id = p_vertex_id;
    IF FOUND THEN RETURN; END IF;

    RETURN QUERY
    SELECT 'Project'::TEXT, project_id, graph_project_vertices.properties
    FROM graph_project_vertices
    WHERE vertex_id = p_vertex_id;
END;
$$;

-- ============================================================================
-- Triggers for Automatic Updates
-- ============================================================================

CREATE OR REPLACE FUNCTION update_graph_vertex_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to all vertex mapping tables
DROP TRIGGER IF EXISTS trg_graph_test_vertices_updated ON graph_test_vertices;
CREATE TRIGGER trg_graph_test_vertices_updated
    BEFORE UPDATE ON graph_test_vertices
    FOR EACH ROW EXECUTE FUNCTION update_graph_vertex_timestamp();

DROP TRIGGER IF EXISTS trg_graph_selector_vertices_updated ON graph_selector_vertices;
CREATE TRIGGER trg_graph_selector_vertices_updated
    BEFORE UPDATE ON graph_selector_vertices
    FOR EACH ROW EXECUTE FUNCTION update_graph_vertex_timestamp();

DROP TRIGGER IF EXISTS trg_graph_failure_vertices_updated ON graph_failure_vertices;
CREATE TRIGGER trg_graph_failure_vertices_updated
    BEFORE UPDATE ON graph_failure_vertices
    FOR EACH ROW EXECUTE FUNCTION update_graph_vertex_timestamp();

DROP TRIGGER IF EXISTS trg_graph_code_change_vertices_updated ON graph_code_change_vertices;
CREATE TRIGGER trg_graph_code_change_vertices_updated
    BEFORE UPDATE ON graph_code_change_vertices
    FOR EACH ROW EXECUTE FUNCTION update_graph_vertex_timestamp();

DROP TRIGGER IF EXISTS trg_graph_healing_pattern_vertices_updated ON graph_healing_pattern_vertices;
CREATE TRIGGER trg_graph_healing_pattern_vertices_updated
    BEFORE UPDATE ON graph_healing_pattern_vertices
    FOR EACH ROW EXECUTE FUNCTION update_graph_vertex_timestamp();

DROP TRIGGER IF EXISTS trg_graph_page_vertices_updated ON graph_page_vertices;
CREATE TRIGGER trg_graph_page_vertices_updated
    BEFORE UPDATE ON graph_page_vertices
    FOR EACH ROW EXECUTE FUNCTION update_graph_vertex_timestamp();

DROP TRIGGER IF EXISTS trg_graph_project_vertices_updated ON graph_project_vertices;
CREATE TRIGGER trg_graph_project_vertices_updated
    BEFORE UPDATE ON graph_project_vertices
    FOR EACH ROW EXECUTE FUNCTION update_graph_vertex_timestamp();

DROP TRIGGER IF EXISTS trg_graph_edges_updated ON graph_edges;
CREATE TRIGGER trg_graph_edges_updated
    BEFORE UPDATE ON graph_edges
    FOR EACH ROW EXECUTE FUNCTION update_graph_vertex_timestamp();

-- ============================================================================
-- RLS Policies
-- ============================================================================

ALTER TABLE graph_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_test_vertices ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_selector_vertices ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_failure_vertices ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_code_change_vertices ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_healing_pattern_vertices ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_page_vertices ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_project_vertices ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_edges ENABLE ROW LEVEL SECURITY;

-- Service role has full access
DROP POLICY IF EXISTS "Service role full access to graph_metadata" ON graph_metadata;
CREATE POLICY "Service role full access to graph_metadata" ON graph_metadata
    FOR ALL USING (true);

DROP POLICY IF EXISTS "Service role full access to graph_test_vertices" ON graph_test_vertices;
CREATE POLICY "Service role full access to graph_test_vertices" ON graph_test_vertices
    FOR ALL USING (true);

DROP POLICY IF EXISTS "Service role full access to graph_selector_vertices" ON graph_selector_vertices;
CREATE POLICY "Service role full access to graph_selector_vertices" ON graph_selector_vertices
    FOR ALL USING (true);

DROP POLICY IF EXISTS "Service role full access to graph_failure_vertices" ON graph_failure_vertices;
CREATE POLICY "Service role full access to graph_failure_vertices" ON graph_failure_vertices
    FOR ALL USING (true);

DROP POLICY IF EXISTS "Service role full access to graph_code_change_vertices" ON graph_code_change_vertices;
CREATE POLICY "Service role full access to graph_code_change_vertices" ON graph_code_change_vertices
    FOR ALL USING (true);

DROP POLICY IF EXISTS "Service role full access to graph_healing_pattern_vertices" ON graph_healing_pattern_vertices;
CREATE POLICY "Service role full access to graph_healing_pattern_vertices" ON graph_healing_pattern_vertices
    FOR ALL USING (true);

DROP POLICY IF EXISTS "Service role full access to graph_page_vertices" ON graph_page_vertices;
CREATE POLICY "Service role full access to graph_page_vertices" ON graph_page_vertices
    FOR ALL USING (true);

DROP POLICY IF EXISTS "Service role full access to graph_project_vertices" ON graph_project_vertices;
CREATE POLICY "Service role full access to graph_project_vertices" ON graph_project_vertices
    FOR ALL USING (true);

DROP POLICY IF EXISTS "Service role full access to graph_edges" ON graph_edges;
CREATE POLICY "Service role full access to graph_edges" ON graph_edges
    FOR ALL USING (true);

-- ============================================================================
-- Grant Permissions
-- ============================================================================

GRANT ALL ON graph_metadata TO service_role;
GRANT ALL ON graph_test_vertices TO service_role;
GRANT ALL ON graph_selector_vertices TO service_role;
GRANT ALL ON graph_failure_vertices TO service_role;
GRANT ALL ON graph_code_change_vertices TO service_role;
GRANT ALL ON graph_healing_pattern_vertices TO service_role;
GRANT ALL ON graph_page_vertices TO service_role;
GRANT ALL ON graph_project_vertices TO service_role;
GRANT ALL ON graph_edges TO service_role;

-- ============================================================================
-- Documentation
-- ============================================================================

COMMENT ON TABLE graph_metadata IS 'Metadata and statistics for the knowledge graph (AGE or Cognee)';
COMMENT ON TABLE graph_test_vertices IS 'Maps test IDs to graph vertex IDs';
COMMENT ON TABLE graph_selector_vertices IS 'Maps CSS/XPath selectors to graph vertex IDs';
COMMENT ON TABLE graph_failure_vertices IS 'Maps test failures to graph vertex IDs';
COMMENT ON TABLE graph_code_change_vertices IS 'Maps git commits to graph vertex IDs';
COMMENT ON TABLE graph_healing_pattern_vertices IS 'Maps healing patterns to graph vertex IDs';
COMMENT ON TABLE graph_page_vertices IS 'Maps page URLs to graph vertex IDs';
COMMENT ON TABLE graph_project_vertices IS 'Maps projects to graph vertex IDs';
COMMENT ON TABLE graph_edges IS 'Tracks all edges in the knowledge graph';

COMMENT ON FUNCTION update_graph_stats IS 'Updates vertex and edge counts in graph metadata';
COMMENT ON FUNCTION get_vertex_info IS 'Retrieves entity information for a given vertex ID';
