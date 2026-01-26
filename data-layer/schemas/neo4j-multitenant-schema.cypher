// ============================================================================
// Argus Multi-Tenant Neo4j Schema for Cognee Knowledge Graphs
// ============================================================================
//
// This schema supports organization-level data isolation with composite indexes
// on (org_id, project_id) for all node types. Every node MUST have org_id.
//
// Cognee Dataset Naming Convention:
//   org_{org_id}_project_{project_id}_{type}
//   Examples:
//     - org_abc123_project_xyz789_codebase
//     - org_abc123_project_xyz789_tests
//     - org_abc123_project_xyz789_failures
//
// ============================================================================

// ----------------------------------------------------------------------------
// CONSTRAINTS (enforce uniqueness and existence)
// ----------------------------------------------------------------------------

// Organization constraints
CREATE CONSTRAINT org_id_unique IF NOT EXISTS
FOR (o:Organization) REQUIRE o.id IS UNIQUE;

CREATE CONSTRAINT org_id_exists IF NOT EXISTS
FOR (o:Organization) REQUIRE o.id IS NOT NULL;

// Project constraints
CREATE CONSTRAINT project_id_unique IF NOT EXISTS
FOR (p:Project) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT project_org_exists IF NOT EXISTS
FOR (p:Project) REQUIRE p.org_id IS NOT NULL;

// Repository constraints
CREATE CONSTRAINT repo_id_unique IF NOT EXISTS
FOR (r:Repository) REQUIRE r.id IS UNIQUE;

CREATE CONSTRAINT repo_org_exists IF NOT EXISTS
FOR (r:Repository) REQUIRE r.org_id IS NOT NULL;

// CodeFile constraints
CREATE CONSTRAINT codefile_id_unique IF NOT EXISTS
FOR (cf:CodeFile) REQUIRE cf.id IS UNIQUE;

CREATE CONSTRAINT codefile_org_exists IF NOT EXISTS
FOR (cf:CodeFile) REQUIRE cf.org_id IS NOT NULL;

// Function constraints
CREATE CONSTRAINT function_id_unique IF NOT EXISTS
FOR (f:Function) REQUIRE f.id IS UNIQUE;

CREATE CONSTRAINT function_org_exists IF NOT EXISTS
FOR (f:Function) REQUIRE f.org_id IS NOT NULL;

// Class constraints
CREATE CONSTRAINT class_id_unique IF NOT EXISTS
FOR (c:Class) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT class_org_exists IF NOT EXISTS
FOR (c:Class) REQUIRE c.org_id IS NOT NULL;

// TestSuite constraints
CREATE CONSTRAINT testsuite_id_unique IF NOT EXISTS
FOR (ts:TestSuite) REQUIRE ts.id IS UNIQUE;

CREATE CONSTRAINT testsuite_org_exists IF NOT EXISTS
FOR (ts:TestSuite) REQUIRE ts.org_id IS NOT NULL;

// TestCase constraints
CREATE CONSTRAINT testcase_id_unique IF NOT EXISTS
FOR (tc:TestCase) REQUIRE tc.id IS UNIQUE;

CREATE CONSTRAINT testcase_org_exists IF NOT EXISTS
FOR (tc:TestCase) REQUIRE tc.org_id IS NOT NULL;

// TestFailure constraints
CREATE CONSTRAINT testfailure_id_unique IF NOT EXISTS
FOR (tf:TestFailure) REQUIRE tf.id IS UNIQUE;

CREATE CONSTRAINT testfailure_org_exists IF NOT EXISTS
FOR (tf:TestFailure) REQUIRE tf.org_id IS NOT NULL;

// TestRun constraints
CREATE CONSTRAINT testrun_id_unique IF NOT EXISTS
FOR (tr:TestRun) REQUIRE tr.id IS UNIQUE;

CREATE CONSTRAINT testrun_org_exists IF NOT EXISTS
FOR (tr:TestRun) REQUIRE tr.org_id IS NOT NULL;

// HealingPattern constraints
CREATE CONSTRAINT healingpattern_id_unique IF NOT EXISTS
FOR (hp:HealingPattern) REQUIRE hp.id IS UNIQUE;

CREATE CONSTRAINT healingpattern_org_exists IF NOT EXISTS
FOR (hp:HealingPattern) REQUIRE hp.org_id IS NOT NULL;

// ----------------------------------------------------------------------------
// COMPOSITE INDEXES (multi-tenant query optimization)
// ----------------------------------------------------------------------------

// Organization-level indexes
CREATE INDEX org_name IF NOT EXISTS FOR (o:Organization) ON (o.name);

// Project indexes (org_id first for tenant isolation)
CREATE INDEX project_org_id IF NOT EXISTS FOR (p:Project) ON (p.org_id);
CREATE INDEX project_org_name IF NOT EXISTS FOR (p:Project) ON (p.org_id, p.name);

// Repository indexes
CREATE INDEX repo_org_id IF NOT EXISTS FOR (r:Repository) ON (r.org_id);
CREATE INDEX repo_org_project IF NOT EXISTS FOR (r:Repository) ON (r.org_id, r.project_id);
CREATE INDEX repo_url IF NOT EXISTS FOR (r:Repository) ON (r.org_id, r.url);

// CodeFile indexes
CREATE INDEX codefile_org_id IF NOT EXISTS FOR (cf:CodeFile) ON (cf.org_id);
CREATE INDEX codefile_org_project IF NOT EXISTS FOR (cf:CodeFile) ON (cf.org_id, cf.project_id);
CREATE INDEX codefile_org_repo IF NOT EXISTS FOR (cf:CodeFile) ON (cf.org_id, cf.repo_id);
CREATE INDEX codefile_path IF NOT EXISTS FOR (cf:CodeFile) ON (cf.org_id, cf.path);
CREATE INDEX codefile_language IF NOT EXISTS FOR (cf:CodeFile) ON (cf.org_id, cf.language);

// Function indexes
CREATE INDEX function_org_id IF NOT EXISTS FOR (f:Function) ON (f.org_id);
CREATE INDEX function_org_project IF NOT EXISTS FOR (f:Function) ON (f.org_id, f.project_id);
CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.org_id, f.name);

// Class indexes
CREATE INDEX class_org_id IF NOT EXISTS FOR (c:Class) ON (c.org_id);
CREATE INDEX class_org_project IF NOT EXISTS FOR (c:Class) ON (c.org_id, c.project_id);
CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.org_id, c.name);

// TestSuite indexes
CREATE INDEX testsuite_org_id IF NOT EXISTS FOR (ts:TestSuite) ON (ts.org_id);
CREATE INDEX testsuite_org_project IF NOT EXISTS FOR (ts:TestSuite) ON (ts.org_id, ts.project_id);

// TestCase indexes
CREATE INDEX testcase_org_id IF NOT EXISTS FOR (tc:TestCase) ON (tc.org_id);
CREATE INDEX testcase_org_project IF NOT EXISTS FOR (tc:TestCase) ON (tc.org_id, tc.project_id);
CREATE INDEX testcase_status IF NOT EXISTS FOR (tc:TestCase) ON (tc.org_id, tc.status);

// TestFailure indexes (critical for self-healing queries)
CREATE INDEX testfailure_org_id IF NOT EXISTS FOR (tf:TestFailure) ON (tf.org_id);
CREATE INDEX testfailure_healed IF NOT EXISTS FOR (tf:TestFailure) ON (tf.org_id, tf.healed);
CREATE INDEX testfailure_error_type IF NOT EXISTS FOR (tf:TestFailure) ON (tf.org_id, tf.error_type);

// TestRun indexes
CREATE INDEX testrun_org_id IF NOT EXISTS FOR (tr:TestRun) ON (tr.org_id);
CREATE INDEX testrun_org_project IF NOT EXISTS FOR (tr:TestRun) ON (tr.org_id, tr.project_id);
CREATE INDEX testrun_started_at IF NOT EXISTS FOR (tr:TestRun) ON (tr.org_id, tr.started_at);

// HealingPattern indexes
CREATE INDEX healingpattern_org_id IF NOT EXISTS FOR (hp:HealingPattern) ON (hp.org_id);
CREATE INDEX healingpattern_success_rate IF NOT EXISTS FOR (hp:HealingPattern) ON (hp.org_id, hp.success_rate);

// ----------------------------------------------------------------------------
// FULL-TEXT INDEXES (semantic search)
// ----------------------------------------------------------------------------

// Code search (function/class names, docstrings)
CREATE FULLTEXT INDEX code_search IF NOT EXISTS
FOR (n:Function|Class) ON EACH [n.name, n.docstring, n.signature];

// Error message search (for failure pattern matching)
CREATE FULLTEXT INDEX error_search IF NOT EXISTS
FOR (tf:TestFailure) ON EACH [tf.error_message, tf.stack_trace];

// File path search
CREATE FULLTEXT INDEX file_search IF NOT EXISTS
FOR (cf:CodeFile) ON EACH [cf.path, cf.name];

// ============================================================================
// NODE DEFINITIONS (with properties)
// ============================================================================

/*
(:Organization {
  id: STRING (UUID),           -- Primary key
  name: STRING,                -- Organization display name
  slug: STRING,                -- URL-friendly identifier
  plan: STRING,                -- 'free', 'pro', 'enterprise'
  settings: MAP,               -- JSON settings blob
  created_at: DATETIME,
  updated_at: DATETIME
})

(:Project {
  id: STRING (UUID),           -- Primary key
  org_id: STRING,              -- FK to Organization (REQUIRED)
  name: STRING,                -- Project display name
  description: STRING,
  default_branch: STRING,      -- e.g., 'main'
  created_at: DATETIME,
  updated_at: DATETIME
})

(:Repository {
  id: STRING (UUID),           -- Primary key
  org_id: STRING,              -- FK to Organization (REQUIRED)
  project_id: STRING,          -- FK to Project
  url: STRING,                 -- Git remote URL
  provider: STRING,            -- 'github', 'gitlab', 'bitbucket'
  default_branch: STRING,
  last_analyzed_at: DATETIME,
  last_commit_sha: STRING,
  created_at: DATETIME
})

(:CodeFile {
  id: STRING (UUID),           -- Primary key
  org_id: STRING,              -- FK to Organization (REQUIRED)
  project_id: STRING,          -- FK to Project
  repo_id: STRING,             -- FK to Repository
  path: STRING,                -- Relative file path
  name: STRING,                -- File name only
  language: STRING,            -- 'python', 'typescript', etc.
  size_bytes: INTEGER,
  lines_of_code: INTEGER,
  last_modified_at: DATETIME,
  content_hash: STRING         -- For change detection
})

(:Function {
  id: STRING (UUID),           -- Primary key
  org_id: STRING,              -- FK to Organization (REQUIRED)
  project_id: STRING,          -- FK to Project
  file_id: STRING,             -- FK to CodeFile
  name: STRING,                -- Function name
  qualified_name: STRING,      -- module.class.function
  signature: STRING,           -- Full signature with params
  docstring: STRING,           -- Documentation
  start_line: INTEGER,
  end_line: INTEGER,
  complexity: INTEGER,         -- Cyclomatic complexity
  is_async: BOOLEAN,
  is_test: BOOLEAN,            -- True if test function
  created_at: DATETIME
})

(:Class {
  id: STRING (UUID),           -- Primary key
  org_id: STRING,              -- FK to Organization (REQUIRED)
  project_id: STRING,          -- FK to Project
  file_id: STRING,             -- FK to CodeFile
  name: STRING,                -- Class name
  qualified_name: STRING,      -- module.Class
  docstring: STRING,
  start_line: INTEGER,
  end_line: INTEGER,
  is_abstract: BOOLEAN,
  created_at: DATETIME
})

(:TestSuite {
  id: STRING (UUID),           -- Primary key
  org_id: STRING,              -- FK to Organization (REQUIRED)
  project_id: STRING,          -- FK to Project
  name: STRING,                -- Suite name
  framework: STRING,           -- 'pytest', 'jest', 'playwright'
  file_path: STRING,           -- Path to test file
  total_tests: INTEGER,
  created_at: DATETIME
})

(:TestCase {
  id: STRING (UUID),           -- Primary key
  org_id: STRING,              -- FK to Organization (REQUIRED)
  project_id: STRING,          -- FK to Project
  suite_id: STRING,            -- FK to TestSuite
  name: STRING,                -- Test case name
  description: STRING,
  status: STRING,              -- 'passed', 'failed', 'skipped', 'pending'
  test_type: STRING,           -- 'ui', 'api', 'unit', 'integration'
  priority: STRING,            -- 'critical', 'high', 'medium', 'low'
  selectors: LIST<STRING>,     -- UI selectors used
  endpoints: LIST<STRING>,     -- API endpoints tested
  last_run_at: DATETIME,
  last_duration_ms: INTEGER,
  flaky_count: INTEGER,        -- Number of flaky runs
  created_at: DATETIME
})

(:TestRun {
  id: STRING (UUID),           -- Primary key
  org_id: STRING,              -- FK to Organization (REQUIRED)
  project_id: STRING,          -- FK to Project
  trigger: STRING,             -- 'manual', 'ci', 'scheduled'
  commit_sha: STRING,
  branch: STRING,
  status: STRING,              -- 'running', 'passed', 'failed', 'cancelled'
  total_tests: INTEGER,
  passed_count: INTEGER,
  failed_count: INTEGER,
  skipped_count: INTEGER,
  started_at: DATETIME,
  completed_at: DATETIME,
  duration_ms: INTEGER
})

(:TestFailure {
  id: STRING (UUID),           -- Primary key
  org_id: STRING,              -- FK to Organization (REQUIRED)
  test_id: STRING,             -- FK to TestCase
  run_id: STRING,              -- FK to TestRun
  error_type: STRING,          -- 'assertion', 'timeout', 'selector', 'network'
  error_message: STRING,       -- Full error message
  stack_trace: STRING,         -- Full stack trace
  screenshot_url: STRING,      -- S3/R2 URL to failure screenshot
  healed: BOOLEAN,             -- True if self-healer fixed this
  healing_pattern_id: STRING,  -- FK to HealingPattern if healed
  created_at: DATETIME
})

(:HealingPattern {
  id: STRING (UUID),           -- Primary key
  org_id: STRING,              -- FK to Organization (REQUIRED)
  error_type: STRING,          -- Type of error this heals
  pattern_signature: STRING,   -- Hash of error pattern
  original_selector: STRING,   -- The broken selector
  healed_selector: STRING,     -- The fixed selector
  strategy: STRING,            -- 'fallback', 'semantic', 'ml'
  success_count: INTEGER,      -- Times successfully applied
  failure_count: INTEGER,      -- Times failed after heal
  success_rate: FLOAT,         -- success_count / total
  created_at: DATETIME,
  last_used_at: DATETIME
})

(:CogneeDocument {
  id: STRING (UUID),           -- Cognee internal ID
  org_id: STRING,              -- FK to Organization (REQUIRED)
  project_id: STRING,          -- FK to Project
  dataset_name: STRING,        -- Cognee dataset name
  source_type: STRING,         -- 'codebase', 'tests', 'failures'
  content_hash: STRING,        -- For deduplication
  processed_at: DATETIME
})
*/

// ============================================================================
// RELATIONSHIP DEFINITIONS
// ============================================================================

/*
Hierarchical Relationships (tenant isolation):
  (:Organization)-[:HAS_PROJECT]->(:Project)
  (:Project)-[:HAS_REPOSITORY]->(:Repository)
  (:Repository)-[:CONTAINS_FILE]->(:CodeFile)
  (:CodeFile)-[:DEFINES_FUNCTION]->(:Function)
  (:CodeFile)-[:DEFINES_CLASS]->(:Class)
  (:Class)-[:HAS_METHOD]->(:Function)
  (:Project)-[:HAS_TESTSUITE]->(:TestSuite)
  (:TestSuite)-[:CONTAINS_TEST]->(:TestCase)
  (:TestRun)-[:EXECUTED_TEST]->(:TestCase)
  (:TestRun)-[:HAS_FAILURE]->(:TestFailure)
  (:TestFailure)-[:FAILED_TEST]->(:TestCase)

Code Relationships:
  (:Function)-[:CALLS]->(:Function)
  (:Function)-[:USES_CLASS]->(:Class)
  (:Class)-[:EXTENDS]->(:Class)
  (:Class)-[:IMPLEMENTS]->(:Class)
  (:CodeFile)-[:IMPORTS]->(:CodeFile)
  (:Function)-[:READS_VARIABLE]->(:Variable)
  (:Function)-[:WRITES_VARIABLE]->(:Variable)

Test Relationships:
  (:TestCase)-[:TESTS_FUNCTION]->(:Function)
  (:TestCase)-[:TESTS_CLASS]->(:Class)
  (:TestCase)-[:TESTS_ENDPOINT]->(:Endpoint)
  (:TestCase)-[:DEPENDS_ON]->(:TestCase)

Healing Relationships:
  (:TestFailure)-[:HEALED_BY]->(:HealingPattern)
  (:HealingPattern)-[:SIMILAR_TO]->(:HealingPattern)
  (:TestCase)-[:PREVIOUSLY_FAILED]->(:TestFailure)

Cognee Relationships:
  (:CogneeDocument)-[:EXTRACTED_FROM]->(:CodeFile)
  (:CogneeDocument)-[:EXTRACTED_FROM]->(:TestCase)
  (:CogneeDocument)-[:RELATED_TO]->(:CogneeDocument)
*/

// ============================================================================
// SAMPLE QUERIES (for reference)
// ============================================================================

/*
-- Get all projects for an organization
MATCH (o:Organization {id: $org_id})-[:HAS_PROJECT]->(p:Project)
RETURN p;

-- Get test failures for a project (tenant-isolated)
MATCH (tf:TestFailure {org_id: $org_id})
WHERE tf.healed = false
MATCH (tf)-[:FAILED_TEST]->(tc:TestCase {project_id: $project_id})
RETURN tf, tc
ORDER BY tf.created_at DESC
LIMIT 50;

-- Find similar healing patterns (for self-healer)
MATCH (hp:HealingPattern {org_id: $org_id})
WHERE hp.error_type = $error_type
AND hp.success_rate > 0.8
RETURN hp
ORDER BY hp.success_rate DESC, hp.success_count DESC
LIMIT 5;

-- Get code coverage graph
MATCH (tc:TestCase {org_id: $org_id, project_id: $project_id})
-[:TESTS_FUNCTION]->(f:Function)
RETURN f.qualified_name, count(tc) as test_coverage
ORDER BY test_coverage DESC;

-- Find untested functions
MATCH (f:Function {org_id: $org_id, project_id: $project_id})
WHERE NOT EXISTS {
  MATCH (tc:TestCase)-[:TESTS_FUNCTION]->(f)
}
AND f.is_test = false
RETURN f.qualified_name, f.complexity
ORDER BY f.complexity DESC;
*/
