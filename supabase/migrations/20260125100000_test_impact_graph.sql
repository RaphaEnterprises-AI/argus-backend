-- ============================================================================
-- Test Impact Graph Schema
-- Maps source files to the tests that exercise them for commit impact prediction
-- ============================================================================

-- Test Impact Graph table
-- Core mapping between source code and tests
CREATE TABLE IF NOT EXISTS test_impact_graph (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Source code mapping
    file_path TEXT NOT NULL,
    function_name TEXT,
    class_name TEXT,
    line_start INTEGER,
    line_end INTEGER,

    -- Test mapping
    test_id UUID REFERENCES tests(id) ON DELETE SET NULL,
    test_file_path TEXT NOT NULL,
    test_name TEXT NOT NULL,
    test_function TEXT,

    -- Relationship strength
    impact_score DECIMAL(3,2) DEFAULT 1.0 CHECK (impact_score >= 0 AND impact_score <= 1),
    relationship_type TEXT NOT NULL CHECK (relationship_type IN (
        'direct',       -- Test directly imports the file
        'transitive',   -- Indirect dependency
        'coverage',     -- From code coverage data
        'historical',   -- From past co-failures
        'semantic'      -- AI-inferred relationship
    )),

    -- Metadata
    last_verified_at TIMESTAMPTZ,
    confidence DECIMAL(3,2) DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    verification_method TEXT,  -- 'ast', 'coverage', 'ai', 'manual'

    -- Source tracking
    source_commit_sha TEXT,
    coverage_run_id UUID,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Ensure unique mapping per project (using index for COALESCE expressions)
CREATE UNIQUE INDEX IF NOT EXISTS idx_test_impact_graph_unique_mapping
ON test_impact_graph (
    project_id,
    file_path,
    test_file_path,
    test_name,
    COALESCE(function_name, ''),
    COALESCE(class_name, '')
);

-- Coverage data imports
-- Track imported coverage files for delta processing
CREATE TABLE IF NOT EXISTS coverage_imports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Import metadata
    file_name TEXT NOT NULL,
    file_hash TEXT NOT NULL,  -- SHA256 of the coverage file
    format TEXT NOT NULL CHECK (format IN ('lcov', 'cobertura', 'istanbul', 'pytest-cov', 'jacoco')),

    -- Processing status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,

    -- Statistics
    files_processed INTEGER DEFAULT 0,
    mappings_created INTEGER DEFAULT 0,
    mappings_updated INTEGER DEFAULT 0,

    -- Commit context
    commit_sha TEXT,
    branch TEXT,

    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Prevent duplicate imports
    UNIQUE(project_id, file_hash)
);

-- Impact graph refresh jobs
-- Track async rebuild operations
CREATE TABLE IF NOT EXISTS impact_graph_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Job configuration
    job_type TEXT NOT NULL CHECK (job_type IN (
        'full_rebuild',     -- Complete rebuild from scratch
        'incremental',      -- Update based on changed files
        'coverage_import',  -- Import coverage data
        'ast_analysis',     -- Run AST import analysis
        'historical',       -- Analyze historical failures
        'semantic'          -- Run AI semantic analysis
    )),

    -- Processing status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress DECIMAL(5,2) DEFAULT 0,
    error_message TEXT,

    -- Results
    mappings_added INTEGER DEFAULT 0,
    mappings_updated INTEGER DEFAULT 0,
    mappings_removed INTEGER DEFAULT 0,
    files_analyzed INTEGER DEFAULT 0,

    -- Metadata
    triggered_by TEXT,  -- 'user', 'webhook', 'scheduler'
    commit_sha TEXT,

    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- File dependencies
-- Track import relationships between source files
CREATE TABLE IF NOT EXISTS file_dependencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Dependency relationship
    source_file TEXT NOT NULL,     -- The file being analyzed
    imports_file TEXT NOT NULL,    -- The file it imports
    import_type TEXT NOT NULL CHECK (import_type IN (
        'module',       -- import module
        'from',         -- from module import X
        'dynamic',      -- dynamic import
        'require',      -- CommonJS require
        'type_only'     -- TypeScript type import
    )),

    -- Import details
    imported_names TEXT[],  -- Specific names imported
    is_relative BOOLEAN DEFAULT FALSE,

    -- Confidence and verification
    confidence DECIMAL(3,2) DEFAULT 1.0,
    last_verified_at TIMESTAMPTZ,
    source_commit_sha TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(project_id, source_file, imports_file)
);

-- Historical failure correlations
-- Track which files/tests fail together
CREATE TABLE IF NOT EXISTS failure_correlations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Correlation pair
    file_path TEXT NOT NULL,
    test_name TEXT NOT NULL,
    test_file_path TEXT NOT NULL,

    -- Statistics
    co_failure_count INTEGER DEFAULT 1,
    total_file_changes INTEGER DEFAULT 1,
    total_test_runs INTEGER DEFAULT 1,

    -- Calculated correlation
    correlation_score DECIMAL(5,4),  -- Pearson or similar

    -- Time window
    first_observed_at TIMESTAMPTZ DEFAULT NOW(),
    last_observed_at TIMESTAMPTZ DEFAULT NOW(),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(project_id, file_path, test_file_path, test_name)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_impact_graph_project ON test_impact_graph(project_id);
CREATE INDEX IF NOT EXISTS idx_impact_graph_file_path ON test_impact_graph(project_id, file_path);
CREATE INDEX IF NOT EXISTS idx_impact_graph_test ON test_impact_graph(project_id, test_file_path, test_name);
CREATE INDEX IF NOT EXISTS idx_impact_graph_relationship ON test_impact_graph(relationship_type);
CREATE INDEX IF NOT EXISTS idx_impact_graph_score ON test_impact_graph(impact_score DESC);

CREATE INDEX IF NOT EXISTS idx_coverage_imports_project ON coverage_imports(project_id);
CREATE INDEX IF NOT EXISTS idx_coverage_imports_status ON coverage_imports(status);

CREATE INDEX IF NOT EXISTS idx_impact_jobs_project ON impact_graph_jobs(project_id);
CREATE INDEX IF NOT EXISTS idx_impact_jobs_status ON impact_graph_jobs(status);

CREATE INDEX IF NOT EXISTS idx_file_deps_project ON file_dependencies(project_id);
CREATE INDEX IF NOT EXISTS idx_file_deps_source ON file_dependencies(project_id, source_file);
CREATE INDEX IF NOT EXISTS idx_file_deps_imports ON file_dependencies(project_id, imports_file);

CREATE INDEX IF NOT EXISTS idx_failure_corr_project ON failure_correlations(project_id);
CREATE INDEX IF NOT EXISTS idx_failure_corr_file ON failure_correlations(project_id, file_path);
CREATE INDEX IF NOT EXISTS idx_failure_corr_score ON failure_correlations(correlation_score DESC);

-- Function to get tests affected by a set of files
CREATE OR REPLACE FUNCTION get_affected_tests(
    p_project_id UUID,
    p_file_paths TEXT[],
    p_min_score DECIMAL DEFAULT 0.3,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    test_id UUID,
    test_file_path TEXT,
    test_name TEXT,
    test_function TEXT,
    total_impact_score DECIMAL,
    relationship_types TEXT[],
    affected_files TEXT[]
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        tig.test_id,
        tig.test_file_path,
        tig.test_name,
        tig.test_function,
        SUM(tig.impact_score * tig.confidence)::DECIMAL AS total_impact_score,
        ARRAY_AGG(DISTINCT tig.relationship_type) AS relationship_types,
        ARRAY_AGG(DISTINCT tig.file_path) AS affected_files
    FROM test_impact_graph tig
    WHERE tig.project_id = p_project_id
      AND tig.file_path = ANY(p_file_paths)
    GROUP BY tig.test_id, tig.test_file_path, tig.test_name, tig.test_function
    HAVING SUM(tig.impact_score * tig.confidence) >= p_min_score
    ORDER BY total_impact_score DESC
    LIMIT p_limit;
END;
$$;

-- Function to get transitive dependencies for files
CREATE OR REPLACE FUNCTION get_transitive_deps(
    p_project_id UUID,
    p_file_paths TEXT[],
    p_max_depth INTEGER DEFAULT 3
)
RETURNS TABLE (
    file_path TEXT,
    depth INTEGER,
    dependency_chain TEXT[]
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE deps AS (
        -- Base case: direct imports
        SELECT
            fd.imports_file AS file_path,
            1 AS depth,
            ARRAY[fd.source_file, fd.imports_file] AS chain
        FROM file_dependencies fd
        WHERE fd.project_id = p_project_id
          AND fd.source_file = ANY(p_file_paths)

        UNION ALL

        -- Recursive case: transitive imports
        SELECT
            fd.imports_file,
            d.depth + 1,
            d.chain || fd.imports_file
        FROM deps d
        JOIN file_dependencies fd ON fd.source_file = d.file_path
        WHERE fd.project_id = p_project_id
          AND d.depth < p_max_depth
          AND NOT fd.imports_file = ANY(d.chain)  -- Prevent cycles
    )
    SELECT DISTINCT ON (deps.file_path)
        deps.file_path,
        deps.depth,
        deps.chain AS dependency_chain
    FROM deps
    ORDER BY deps.file_path, deps.depth;
END;
$$;

-- Function to update impact graph from coverage data
CREATE OR REPLACE FUNCTION upsert_coverage_mapping(
    p_project_id UUID,
    p_file_path TEXT,
    p_test_file_path TEXT,
    p_test_name TEXT,
    p_coverage_run_id UUID,
    p_hit_count INTEGER DEFAULT 1
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_id UUID;
    v_new_score DECIMAL;
BEGIN
    -- Calculate score based on hit count (normalized)
    v_new_score := LEAST(1.0, 0.5 + (p_hit_count::DECIMAL / 100));

    INSERT INTO test_impact_graph (
        project_id, file_path, test_file_path, test_name,
        relationship_type, impact_score, confidence,
        coverage_run_id, verification_method, last_verified_at
    )
    VALUES (
        p_project_id, p_file_path, p_test_file_path, p_test_name,
        'coverage', v_new_score, 0.9,
        p_coverage_run_id, 'coverage', NOW()
    )
    ON CONFLICT (project_id, file_path, test_file_path, test_name, COALESCE(function_name, ''), COALESCE(class_name, ''))
    DO UPDATE SET
        impact_score = GREATEST(test_impact_graph.impact_score, v_new_score),
        coverage_run_id = p_coverage_run_id,
        last_verified_at = NOW(),
        updated_at = NOW()
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$;

-- Auto-update timestamps
CREATE OR REPLACE FUNCTION update_impact_graph_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_impact_graph_updated
    BEFORE UPDATE ON test_impact_graph
    FOR EACH ROW
    EXECUTE FUNCTION update_impact_graph_timestamp();

CREATE TRIGGER trg_file_deps_updated
    BEFORE UPDATE ON file_dependencies
    FOR EACH ROW
    EXECUTE FUNCTION update_impact_graph_timestamp();

CREATE TRIGGER trg_failure_corr_updated
    BEFORE UPDATE ON failure_correlations
    FOR EACH ROW
    EXECUTE FUNCTION update_impact_graph_timestamp();

-- RLS policies
ALTER TABLE test_impact_graph ENABLE ROW LEVEL SECURITY;
ALTER TABLE coverage_imports ENABLE ROW LEVEL SECURITY;
ALTER TABLE impact_graph_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE file_dependencies ENABLE ROW LEVEL SECURITY;
ALTER TABLE failure_correlations ENABLE ROW LEVEL SECURITY;

-- Policies for test_impact_graph
CREATE POLICY "Users can view impact graph for their projects" ON test_impact_graph
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage impact graph for their projects" ON test_impact_graph
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policies for coverage_imports
CREATE POLICY "Users can view coverage imports for their projects" ON coverage_imports
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage coverage imports for their projects" ON coverage_imports
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policies for impact_graph_jobs
CREATE POLICY "Users can view jobs for their projects" ON impact_graph_jobs
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage jobs for their projects" ON impact_graph_jobs
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policies for file_dependencies
CREATE POLICY "Users can view file deps for their projects" ON file_dependencies
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage file deps for their projects" ON file_dependencies
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policies for failure_correlations
CREATE POLICY "Users can view failure correlations for their projects" ON failure_correlations
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage failure correlations for their projects" ON failure_correlations
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Grant service role full access
GRANT ALL ON test_impact_graph TO service_role;
GRANT ALL ON coverage_imports TO service_role;
GRANT ALL ON impact_graph_jobs TO service_role;
GRANT ALL ON file_dependencies TO service_role;
GRANT ALL ON failure_correlations TO service_role;

COMMENT ON TABLE test_impact_graph IS 'Maps source files to tests that exercise them for commit impact prediction';
COMMENT ON TABLE coverage_imports IS 'Tracks imported coverage files for delta processing';
COMMENT ON TABLE impact_graph_jobs IS 'Async jobs for rebuilding impact graph';
COMMENT ON TABLE file_dependencies IS 'Import relationships between source files';
COMMENT ON TABLE failure_correlations IS 'Historical correlation between file changes and test failures';
