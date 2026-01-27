-- ============================================================================
-- API Test Cases and Results Tables
-- Stores API test definitions, discovered endpoints, and execution results
-- ============================================================================

-- API Endpoints discovered from OpenAPI specs
CREATE TABLE IF NOT EXISTS api_endpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Endpoint details
    path TEXT NOT NULL,
    method TEXT NOT NULL CHECK (method IN ('GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS')),
    operation_id TEXT,
    summary TEXT,
    description TEXT,

    -- Request details (from OpenAPI)
    parameters JSONB DEFAULT '[]',  -- Path, query, header params
    request_body_schema JSONB,
    request_content_types TEXT[] DEFAULT '{}',

    -- Response details (from OpenAPI)
    response_schemas JSONB DEFAULT '{}',  -- Status code -> schema mapping
    response_content_types TEXT[] DEFAULT '{}',

    -- Security requirements
    security_schemes TEXT[] DEFAULT '{}',
    auth_type TEXT DEFAULT 'none' CHECK (auth_type IN ('none', 'bearer', 'basic', 'api_key', 'oauth2')),

    -- Discovery metadata
    discovered_from TEXT,  -- 'openapi', 'swagger', 'manual', 'crawled'
    spec_version TEXT,
    tags TEXT[] DEFAULT '{}',

    -- Status
    is_deprecated BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(project_id, path, method)
);

-- API Test Cases
CREATE TABLE IF NOT EXISTS api_test_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    endpoint_id UUID REFERENCES api_endpoints(id) ON DELETE SET NULL,

    -- Test identification
    name TEXT NOT NULL,
    description TEXT,

    -- Request configuration
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL CHECK (method IN ('GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS')),
    headers JSONB DEFAULT '{}',
    query_params JSONB DEFAULT '{}',
    path_params JSONB DEFAULT '{}',
    body JSONB,

    -- Authentication
    auth_type TEXT DEFAULT 'none' CHECK (auth_type IN ('none', 'bearer', 'basic', 'api_key', 'oauth2')),
    auth_config JSONB DEFAULT '{}',

    -- Expected results
    expected_status INTEGER NOT NULL DEFAULT 200,
    expected_status_range INT4RANGE,  -- e.g., [200, 300) for 2xx responses
    expected_body_schema JSONB,
    expected_body_contains JSONB,  -- Key paths that must exist
    expected_headers JSONB DEFAULT '{}',
    max_latency_ms INTEGER DEFAULT 5000,

    -- Test metadata
    test_type TEXT DEFAULT 'functional' CHECK (test_type IN (
        'functional',     -- Happy path
        'negative',       -- Error cases
        'boundary',       -- Edge cases
        'security',       -- Security tests
        'performance',    -- Performance checks
        'integration'     -- Multi-endpoint flows
    )),
    tags TEXT[] DEFAULT '{}',
    priority TEXT DEFAULT 'medium' CHECK (priority IN ('critical', 'high', 'medium', 'low')),

    -- Generation metadata
    source TEXT DEFAULT 'manual' CHECK (source IN ('manual', 'ai_generated', 'discovered', 'imported')),
    generated_by TEXT,  -- 'claude', 'openapi', 'user'
    generation_prompt TEXT,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMPTZ,
    last_run_status TEXT CHECK (last_run_status IN ('passed', 'failed', 'error', 'skipped')),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- API Test Results
CREATE TABLE IF NOT EXISTS api_test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    test_case_id UUID NOT NULL REFERENCES api_test_cases(id) ON DELETE CASCADE,
    test_run_id UUID REFERENCES test_runs(id) ON DELETE SET NULL,

    -- Execution details
    status TEXT NOT NULL CHECK (status IN ('passed', 'failed', 'error', 'skipped', 'timeout')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,

    -- Request sent
    request_url TEXT NOT NULL,
    request_method TEXT NOT NULL,
    request_headers JSONB DEFAULT '{}',
    request_body JSONB,

    -- Response received
    response_status INTEGER,
    response_headers JSONB DEFAULT '{}',
    response_body JSONB,
    response_size_bytes INTEGER,
    response_time_ms INTEGER,

    -- Validation results
    status_code_valid BOOLEAN,
    schema_valid BOOLEAN,
    schema_errors JSONB DEFAULT '[]',
    headers_valid BOOLEAN,
    latency_valid BOOLEAN,
    body_contains_valid BOOLEAN,

    -- Error details
    error_type TEXT,
    error_message TEXT,
    error_details JSONB,

    -- Environment
    base_url TEXT,
    environment TEXT DEFAULT 'test' CHECK (environment IN ('local', 'development', 'staging', 'production', 'test')),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- API Test Suites (groups of related tests)
CREATE TABLE IF NOT EXISTS api_test_suites (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    name TEXT NOT NULL,
    description TEXT,

    -- Suite configuration
    base_url TEXT,
    default_headers JSONB DEFAULT '{}',
    default_auth_type TEXT DEFAULT 'none',
    default_auth_config JSONB DEFAULT '{}',

    -- Test ordering
    test_order UUID[] DEFAULT '{}',  -- Ordered list of test_case_ids
    run_in_parallel BOOLEAN DEFAULT FALSE,
    stop_on_failure BOOLEAN DEFAULT TRUE,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMPTZ,
    last_run_status TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Suite membership (many-to-many)
CREATE TABLE IF NOT EXISTS api_test_suite_tests (
    suite_id UUID NOT NULL REFERENCES api_test_suites(id) ON DELETE CASCADE,
    test_case_id UUID NOT NULL REFERENCES api_test_cases(id) ON DELETE CASCADE,
    order_index INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (suite_id, test_case_id)
);

-- API Discovery Sessions (tracks OpenAPI spec parsing)
CREATE TABLE IF NOT EXISTS api_discovery_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Source
    openapi_url TEXT,
    openapi_content JSONB,
    spec_version TEXT,
    api_title TEXT,
    api_version TEXT,

    -- Results
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'parsing', 'completed', 'failed'
    )),
    endpoints_discovered INTEGER DEFAULT 0,
    tests_generated INTEGER DEFAULT 0,
    errors JSONB DEFAULT '[]',

    -- Metadata
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_api_endpoints_project ON api_endpoints(project_id);
CREATE INDEX IF NOT EXISTS idx_api_endpoints_path ON api_endpoints(path);
CREATE INDEX IF NOT EXISTS idx_api_endpoints_method ON api_endpoints(method);
CREATE INDEX IF NOT EXISTS idx_api_endpoints_tags ON api_endpoints USING GIN (tags);

CREATE INDEX IF NOT EXISTS idx_api_test_cases_project ON api_test_cases(project_id);
CREATE INDEX IF NOT EXISTS idx_api_test_cases_endpoint ON api_test_cases(endpoint_id);
CREATE INDEX IF NOT EXISTS idx_api_test_cases_type ON api_test_cases(test_type);
CREATE INDEX IF NOT EXISTS idx_api_test_cases_priority ON api_test_cases(priority);
CREATE INDEX IF NOT EXISTS idx_api_test_cases_tags ON api_test_cases USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_api_test_cases_status ON api_test_cases(last_run_status);

CREATE INDEX IF NOT EXISTS idx_api_test_results_project ON api_test_results(project_id);
CREATE INDEX IF NOT EXISTS idx_api_test_results_test_case ON api_test_results(test_case_id);
CREATE INDEX IF NOT EXISTS idx_api_test_results_test_run ON api_test_results(test_run_id);
CREATE INDEX IF NOT EXISTS idx_api_test_results_status ON api_test_results(status);
CREATE INDEX IF NOT EXISTS idx_api_test_results_started ON api_test_results(started_at DESC);

CREATE INDEX IF NOT EXISTS idx_api_test_suites_project ON api_test_suites(project_id);
CREATE INDEX IF NOT EXISTS idx_api_discovery_sessions_project ON api_discovery_sessions(project_id);

-- Auto-update timestamps trigger
CREATE OR REPLACE FUNCTION update_api_tests_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_api_endpoints_updated
    BEFORE UPDATE ON api_endpoints
    FOR EACH ROW
    EXECUTE FUNCTION update_api_tests_timestamp();

CREATE TRIGGER trg_api_test_cases_updated
    BEFORE UPDATE ON api_test_cases
    FOR EACH ROW
    EXECUTE FUNCTION update_api_tests_timestamp();

CREATE TRIGGER trg_api_test_suites_updated
    BEFORE UPDATE ON api_test_suites
    FOR EACH ROW
    EXECUTE FUNCTION update_api_tests_timestamp();

-- RLS policies
ALTER TABLE api_endpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_test_cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_test_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_test_suites ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_test_suite_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_discovery_sessions ENABLE ROW LEVEL SECURITY;

-- Policies for api_endpoints
CREATE POLICY "Users can view endpoints for their projects" ON api_endpoints
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage endpoints for their projects" ON api_endpoints
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policies for api_test_cases
CREATE POLICY "Users can view test cases for their projects" ON api_test_cases
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage test cases for their projects" ON api_test_cases
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policies for api_test_results
CREATE POLICY "Users can view test results for their projects" ON api_test_results
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage test results for their projects" ON api_test_results
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policies for api_test_suites
CREATE POLICY "Users can view test suites for their projects" ON api_test_suites
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage test suites for their projects" ON api_test_suites
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Policies for api_test_suite_tests
CREATE POLICY "Users can view suite tests for their projects" ON api_test_suite_tests
    FOR SELECT USING (
        suite_id IN (
            SELECT s.id FROM api_test_suites s
            WHERE s.project_id IN (
                SELECT p.id FROM projects p
                JOIN organizations o ON p.organization_id = o.id
                JOIN organization_members om ON o.id = om.organization_id
                WHERE om.user_id = auth.uid()::text
            )
        )
    );

CREATE POLICY "Users can manage suite tests for their projects" ON api_test_suite_tests
    FOR ALL USING (
        suite_id IN (
            SELECT s.id FROM api_test_suites s
            WHERE s.project_id IN (
                SELECT p.id FROM projects p
                JOIN organizations o ON p.organization_id = o.id
                JOIN organization_members om ON o.id = om.organization_id
                WHERE om.user_id = auth.uid()::text
            )
        )
    );

-- Policies for api_discovery_sessions
CREATE POLICY "Users can view discovery sessions for their projects" ON api_discovery_sessions
    FOR SELECT USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage discovery sessions for their projects" ON api_discovery_sessions
    FOR ALL USING (
        project_id IN (
            SELECT p.id FROM projects p
            JOIN organizations o ON p.organization_id = o.id
            JOIN organization_members om ON o.id = om.organization_id
            WHERE om.user_id = auth.uid()::text
        )
    );

-- Grant service role full access
GRANT ALL ON api_endpoints TO service_role;
GRANT ALL ON api_test_cases TO service_role;
GRANT ALL ON api_test_results TO service_role;
GRANT ALL ON api_test_suites TO service_role;
GRANT ALL ON api_test_suite_tests TO service_role;
GRANT ALL ON api_discovery_sessions TO service_role;

-- Comments
COMMENT ON TABLE api_endpoints IS 'API endpoints discovered from OpenAPI/Swagger specs';
COMMENT ON TABLE api_test_cases IS 'API test case definitions with request/response expectations';
COMMENT ON TABLE api_test_results IS 'Results from API test executions';
COMMENT ON TABLE api_test_suites IS 'Groups of related API tests for organized execution';
COMMENT ON TABLE api_discovery_sessions IS 'Tracking of OpenAPI spec parsing sessions';
