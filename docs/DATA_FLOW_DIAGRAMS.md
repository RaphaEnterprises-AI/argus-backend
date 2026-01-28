# Argus Data Flow Diagrams

**Document Version:** 1.0
**Date:** 2026-01-28

---

## 1. Authentication Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AUTHENTICATION DATA FLOW                               │
└─────────────────────────────────────────────────────────────────────────────────┘

┌────────────┐                                                      ┌────────────┐
│   Client   │                                                      │   Clerk    │
│  (Browser/ │                                                      │   (JWKS)   │
│   CLI/MCP) │                                                      │            │
└─────┬──────┘                                                      └─────┬──────┘
      │                                                                   │
      │  (A) API Key Auth                                                 │
      │  ════════════════                                                 │
      │                                                                   │
      │  X-API-Key: argus_sk_abc123...                                   │
      │  ─────────────────────────────────────────▶                      │
      │                          ┌────────────────────────────────────┐  │
      │                          │      AuthenticationMiddleware      │  │
      │                          │  ┌────────────────────────────────┐│  │
      │                          │  │ 1. Extract X-API-Key header    ││  │
      │                          │  │ 2. Verify prefix: argus_sk_    ││  │
      │                          │  │ 3. SHA256 hash the key         ││  │
      │                          │  │ 4. Query Supabase api_keys     ││  │
      │                          │  │ 5. Validate:                   ││  │
      │                          │  │    - Not revoked               ││  │
      │                          │  │    - Not expired               ││  │
      │                          │  │    - Scopes non-empty          ││  │
      │                          │  │ 6. Update last_used_at         ││  │
      │                          │  │ 7. Create UserContext          ││  │
      │                          │  └────────────────────────────────┘│  │
      │                          └────────────────────────────────────┘  │
      │                                                                   │
      │  (B) JWT Auth (Clerk)                                            │
      │  ════════════════════                                            │
      │                                                                   │
      │  Authorization: Bearer eyJhbGciOi...                              │
      │  ─────────────────────────────────────────▶                      │
      │                          ┌────────────────────────────────────┐  │
      │                          │      AuthenticationMiddleware      │  │
      │                          │  ┌────────────────────────────────┐│  │
      │                          │  │ 1. Extract Bearer token        ││  │
      │                          │  │ 2. Decode header (get kid)     ││  │
      │                          │  │ 3. Extract issuer from claims  ││  │
      │                          │  │ 4. Derive JWKS URL:            ││──┼──▶ GET /.well-known/jwks.json
      │                          │  │    {issuer}/.well-known/jwks   ││◀─┼── JWKS response (cached 1hr)
      │                          │  │ 5. Find key by kid             ││  │
      │                          │  │ 6. Verify RS256 signature      ││  │
      │                          │  │ 7. Validate exp, iat, iss      ││  │
      │                          │  │ 8. Extract claims:             ││  │
      │                          │  │    - sub (user_id)             ││  │
      │                          │  │    - org_id                    ││  │
      │                          │  │    - org_role                  ││  │
      │                          │  │ 9. Fetch user info if needed   ││──┼──▶ GET /v1/users/{sub}
      │                          │  │10. Create UserContext          ││◀─┼── User info (cached 5min)
      │                          │  └────────────────────────────────┘│  │
      │                          └────────────────────────────────────┘  │
      │                                                                   │
      │  (C) Device Auth Flow (MCP/CLI)                                  │
      │  ══════════════════════════════                                  │
      │                                                                   │
      │  1. POST /api/v1/auth/device/authorize                           │
      │  ──────────────────────────────────────▶                         │
      │  ◀──────────────────────────────────────                         │
      │     {device_code, user_code, verification_uri, expires_in}       │
      │                                                                   │
      │  2. User visits verification_uri, enters user_code               │
      │     ────────────────────────────────────────────────────────────▶│
      │                                          (Clerk hosted auth)     │
      │                                                                   │
      │  3. Poll: POST /api/v1/auth/device/token                         │
      │  ─────────────────────────────▶                                  │
      │  ◀─────────────────────────────                                  │
      │     {access_token, refresh_token, expires_in, user}              │
      │                                                                   │
└──────────────────────────────────────────────────────────────────────────┘

SECURITY CONTROLS:
┌─────────────────────────────────────────────────────────────────────────────┐
│ - API Keys: SHA256 hashed, stored in Supabase with expiration              │
│ - JWTs: RS256 signed by Clerk, verified against JWKS                       │
│ - Device Auth: Polling with exponential backoff, timeout after 5min        │
│ - All auth events logged to security_audit_logs table                      │
│ - Failed auth: minimum 100ms response time (timing attack prevention)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Test Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TEST EXECUTION DATA FLOW                                │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────┐  ┌───────────┐  ┌────────────┐  ┌───────────┐  ┌──────────┐  ┌───────┐
│  Client  │  │  FastAPI  │  │ LangGraph  │  │  Browser  │  │ Supabase │  │ Redis/│
│          │  │   Server  │  │Orchestrator│  │   Pool    │  │          │  │Redpanda│
└────┬─────┘  └─────┬─────┘  └─────┬──────┘  └─────┬─────┘  └────┬─────┘  └───┬───┘
     │              │              │              │              │            │
     │ POST /api/v1/tests/{id}/run │              │              │            │
     ├─────────────▶│              │              │              │            │
     │              │              │              │              │            │
     │              │ create_initial_state()      │              │            │
     │              ├─────────────▶│              │              │            │
     │              │              │              │              │            │
     │              │              │ INSERT test_runs            │            │
     │              │              ├─────────────────────────────▶│            │
     │              │              │              │              │            │
     │              │              │ Checkpoint (run_id)         │            │
     │              │              ├─────────────────────────────▶│            │
     │              │              │              │              │            │
     │   202 Accepted              │              │              │            │
     │   {job_id, run_id}          │              │              │            │
     │◀─────────────│              │              │              │            │
     │              │              │              │              │            │
     ├──────────────┼──────────────┼──────────────┼──────────────┼────────────┤
     │              │              │              │              │            │
     │   SSE: GET /api/v1/stream/test/{run_id}   │              │            │
     ├─────────────▶│              │              │              │            │
     │              │              │              │              │            │
     │              │  ┌───────────────────────────────────────────────────┐ │
     │              │  │                 LANGGRAPH EXECUTION               │ │
     │              │  └───────────────────────────────────────────────────┘ │
     │              │              │              │              │            │
     │              │  ┌───────────┴──────────────┴──────────────┴───────┐   │
     │              │  │ NODE: analyze_code (CodeAnalyzerAgent)          │   │
     │              │  │ ─────────────────────────────────────────────── │   │
     │              │  │ Input: codebase_path, changed_files             │   │
     │              │  │ Process:                                        │   │
     │              │  │   1. Parse source files (tree-sitter)           │   │
     │              │  │   2. Extract routes, components, selectors      │   │
     │              │  │   3. Build dependency graph                     │   │
     │              │  │   4. Identify testable surfaces                 │   │
     │              │  │ Output: codebase_summary, testable_surfaces     │   │
     │              │  │ Tokens: ~2000-5000 input, ~1000-3000 output     │   │
     │              │  └──────────────────────────────────────────────────┘  │
     │              │              │              │              │            │
     │   SSE: {type: "node", node: "analyze_code", status: "completed"}      │
     │◀─────────────│              │              │              │            │
     │              │              │              │              │            │
     │              │  ┌───────────┴──────────────┴──────────────┴───────┐   │
     │              │  │ NODE: plan_tests (TestPlannerAgent)             │   │
     │              │  │ ─────────────────────────────────────────────── │   │
     │              │  │ Input: testable_surfaces, changed_files         │   │
     │              │  │ Process:                                        │   │
     │              │  │   1. Prioritize by risk (critical paths first)  │   │
     │              │  │   2. Generate TestSpec for each surface         │   │
     │              │  │   3. Include steps, assertions, preconditions   │   │
     │              │  │   4. Respect max_tests limit                    │   │
     │              │  │ Output: test_plan (TestSpec[])                  │   │
     │              │  │ Cost: ~$0.05-0.20                               │   │
     │              │  └──────────────────────────────────────────────────┘  │
     │              │              │              │              │            │
     │              │              │ UPDATE test_runs.plan       │            │
     │              │              ├─────────────────────────────▶│            │
     │              │              │              │              │            │
     │   SSE: {type: "plan", tests: [...]}       │              │            │
     │◀─────────────│              │              │              │            │
     │              │              │              │              │            │
     │              │  ┌───────────┴──────────────┴──────────────┴───────┐   │
     │              │  │ NODE: execute_test (UITesterAgent) [PARALLEL]   │   │
     │              │  │ ─────────────────────────────────────────────── │   │
     │              │  │ For each test in test_plan:                     │   │
     │              │  │   1. Request browser session                    │   │
     │              │  │      ─────────────────────────▶│                │   │
     │              │  │   2. Navigate to app_url       │                │   │
     │              │  │   3. Execute test steps:       │                │   │
     │              │  │      - click(selector)         │                │   │
     │              │  │      - fill(selector, value)   │                │   │
     │              │  │      - screenshot()            │                │   │
     │              │  │   4. Run assertions            │                │   │
     │              │  │   5. Capture evidence          │                │   │
     │              │  │      ◀─────────────────────────│                │   │
     │              │  │   6. Return TestResult         │                │   │
     │              │  └──────────────────────────────────────────────────┘  │
     │              │              │              │              │            │
     │              │              │ INSERT test_results         │            │
     │              │              ├─────────────────────────────▶│            │
     │              │              │              │              │            │
     │              │              │ INSERT artifacts (screenshots)           │
     │              │              ├─────────────────────────────▶│            │
     │              │              │              │              │            │
     │   SSE: {type: "test_result", test_id: "...", status: "passed"}        │
     │◀─────────────│              │              │              │            │
     │              │              │              │              │            │
     │              │              │ PUBLISH: TEST_PASSED/FAILED │            │
     │              │              ├─────────────────────────────────────────▶│
     │              │              │              │              │            │
     │              │  ┌───────────┴──────────────┴──────────────┴───────┐   │
     │              │  │ NODE: evaluate_results                          │   │
     │              │  │ ─────────────────────────────────────────────── │   │
     │              │  │ Check: failures.length > 0 && self_heal_enabled │   │
     │              │  │ Route to: self_heal OR report                   │   │
     │              │  └──────────────────────────────────────────────────┘  │
     │              │              │              │              │            │
     │              │  ┌───────────┴──────────────┴──────────────┴───────┐   │
     │              │  │ NODE: self_heal (SelfHealerAgent) [IF FAILURES] │   │
     │              │  │ ─────────────────────────────────────────────── │   │
     │              │  │ Input: failures, screenshots, codebase_context  │   │
     │              │  │ Process:                                        │   │
     │              │  │   1. Analyze failure type:                      │   │
     │              │  │      - selector_changed                         │   │
     │              │  │      - timing_issue                             │   │
     │              │  │      - ui_change                                │   │
     │              │  │      - real_bug                                 │   │
     │              │  │   2. Search healing_patterns (semantic)         │   │
     │              │  │   3. Generate suggested_fix                     │   │
     │              │  │   4. If confidence > threshold, auto-apply      │   │
     │              │  │ Output: healed_test_specs, healing_applied      │   │
     │              │  └──────────────────────────────────────────────────┘  │
     │              │              │              │              │            │
     │              │              │ INSERT healing_patterns     │            │
     │              │              ├─────────────────────────────▶│            │
     │              │              │              │              │            │
     │   SSE: {type: "healing", test_id: "...", fix: {...}}     │            │
     │◀─────────────│              │              │              │            │
     │              │              │              │              │            │
     │              │  ┌───────────┴──────────────┴──────────────┴───────┐   │
     │              │  │ NODE: report (ReporterAgent)                    │   │
     │              │  │ ─────────────────────────────────────────────── │   │
     │              │  │ 1. Aggregate test_results                       │   │
     │              │  │ 2. Generate summary report                      │   │
     │              │  │ 3. If PR number, post GitHub comment            │   │
     │              │  │ 4. If failures, create Jira tickets             │   │
     │              │  │ 5. Send Slack notification                      │   │
     │              │  └──────────────────────────────────────────────────┘  │
     │              │              │              │              │            │
     │              │              │ UPDATE test_runs.status='completed'     │
     │              │              ├─────────────────────────────▶│            │
     │              │              │              │              │            │
     │              │              │ INSERT reports              │            │
     │              │              ├─────────────────────────────▶│            │
     │              │              │              │              │            │
     │   SSE: {type: "complete", summary: {...}, report_id: "..."} │         │
     │◀─────────────│              │              │              │            │
     │              │              │              │              │            │
     │   SSE: CLOSE │              │              │              │            │
     │◀═════════════│              │              │              │            │
     │              │              │              │              │            │
```

---

## 3. Multi-Tenant Data Isolation

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-TENANT DATA ISOLATION                               │
└─────────────────────────────────────────────────────────────────────────────────┘

REQUEST CONTEXT EXTRACTION
══════════════════════════

┌──────────────────┐
│    HTTP Request  │
│ ┌──────────────┐ │       ┌────────────────────────────────────────────────────┐
│ │   Headers    │ │──────▶│                 TenantMiddleware                   │
│ │ X-Org-ID     │ │       │ ┌────────────────────────────────────────────────┐ │
│ │ X-Project-ID │ │       │ │ 1. Extract from URL path params:               │ │
│ └──────────────┘ │       │ │    /api/v1/orgs/{org_id}/projects/{project_id} │ │
│ ┌──────────────┐ │       │ │                                                │ │
│ │  URL Path    │ │       │ │ 2. Extract from headers:                       │ │
│ │ /orgs/{id}/  │ │       │ │    X-Organization-ID, X-Project-ID             │ │
│ └──────────────┘ │       │ │                                                │ │
│ ┌──────────────┐ │       │ │ 3. Extract from query params                   │ │
│ │Query Params  │ │       │ │                                                │ │
│ │ ?org_id=...  │ │       │ │ 4. Fall back to user's default org from JWT   │ │
│ └──────────────┘ │       │ └────────────────────────────────────────────────┘ │
│ ┌──────────────┐ │       │                         │                          │
│ │  JWT Claims  │ │       │                         ▼                          │
│ │ org: "..."   │ │       │ ┌────────────────────────────────────────────────┐ │
│ └──────────────┘ │       │ │           TenantContext                        │ │
└──────────────────┘       │ │   - organization_id: UUID                      │ │
                           │ │   - project_id: UUID | None                    │ │
                           │ │   - user_id: str                               │ │
                           │ └────────────────────────────────────────────────┘ │
                           └────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
DATABASE RLS ENFORCEMENT
════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            PostgreSQL (Supabase)                                 │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                        RLS Policy Chain                                    │  │
│  │                                                                            │  │
│  │  1. Service Role Bypass                                                    │  │
│  │     ─────────────────────                                                  │  │
│  │     IF current_user IN ('service_role', 'postgres', 'supabase_admin')     │  │
│  │     THEN RETURN TRUE                                                       │  │
│  │                                                                            │  │
│  │  2. Organization Access Check                                              │  │
│  │     ────────────────────────────                                           │  │
│  │     public.has_org_access(organization_id)                                 │  │
│  │       │                                                                    │  │
│  │       ├── Check JWT claims: request.jwt.claims -> 'organization_ids'      │  │
│  │       │                                                                    │  │
│  │       ├── Check app.user_id setting                                        │  │
│  │       │                                                                    │  │
│  │       └── Lookup organization_members table                                │  │
│  │                                                                            │  │
│  │  3. Project Access Check (if applicable)                                   │  │
│  │     ──────────────────────────────────────                                 │  │
│  │     public.has_project_access(project_id)                                  │  │
│  │       │                                                                    │  │
│  │       └── Lookup project's organization_id                                 │  │
│  │           └── Check public.has_org_access(org_id)                         │  │
│  │                                                                            │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  EXAMPLE: tests table                                                            │
│  ═══════════════════                                                             │
│                                                                                  │
│  CREATE POLICY tests_policy ON tests                                            │
│      FOR ALL                                                                     │
│      USING (                                                                     │
│          public.is_service_role() OR                                            │
│          public.has_project_access(project_id)                                  │
│      );                                                                          │
│                                                                                  │
│  QUERY: SELECT * FROM tests WHERE id = 'test_123'                               │
│                                                                                  │
│  Execution:                                                                      │
│  1. RLS policy evaluates: has_project_access('test_123's project_id)           │
│  2. Get project's org_id from projects table                                    │
│  3. Check if user's orgs contain that org_id                                    │
│  4. If TRUE, return row; if FALSE, row is invisible                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

DATA SEGREGATION DIAGRAM
════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Organization: Acme Corp                             │
│                              org_id: org_001                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐  │
│  │  Project: Frontend  │    │  Project: Backend   │    │  Project: Mobile    │  │
│  │  project_id: proj_A │    │  project_id: proj_B │    │  project_id: proj_C │  │
│  ├─────────────────────┤    ├─────────────────────┤    ├─────────────────────┤  │
│  │ tests (15)          │    │ tests (23)          │    │ tests (8)           │  │
│  │ test_runs (142)     │    │ test_runs (89)      │    │ test_runs (34)      │  │
│  │ healing_patterns(7) │    │ healing_patterns(12)│    │ healing_patterns(3) │  │
│  │ discovery_flows(45) │    │ discovery_flows(0)  │    │ discovery_flows(12) │  │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘  │
│                                                                                  │
│  Shared at org level:                                                            │
│  - organization_members (users)                                                  │
│  - api_keys                                                                      │
│  - integrations                                                                  │
│  - audit_logs                                                                    │
│  - ai_usage tracking                                                             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                             Organization: Beta Inc                               │
│                             org_id: org_002                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐                             │
│  │  Project: Main App  │    │  Project: Admin     │                             │
│  │  project_id: proj_X │    │  project_id: proj_Y │    COMPLETELY ISOLATED     │
│  ├─────────────────────┤    ├─────────────────────┤    FROM org_001's DATA     │
│  │ tests (5)           │    │ tests (3)           │                             │
│  │ test_runs (12)      │    │ test_runs (8)       │                             │
│  └─────────────────────┘    └─────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────────┘

CROSS-TENANT PROTECTION
═══════════════════════

User from org_001 requests: GET /api/v1/projects/proj_X/tests

1. TenantMiddleware extracts: user.organization_id = org_001
2. Route handler queries: SELECT * FROM tests WHERE project_id = 'proj_X'
3. RLS evaluates: has_project_access('proj_X')
4. RLS looks up: proj_X belongs to org_002
5. RLS checks: org_002 NOT IN [org_001]
6. Result: ZERO ROWS RETURNED (data invisible to user)

EVEN IF user crafts malicious query or parameter injection,
RLS at database level prevents any cross-tenant data access.
```

---

## 4. Event Streaming Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EVENT STREAMING DATA FLOW                                │
└─────────────────────────────────────────────────────────────────────────────────┘

PRODUCER SIDE (Backend)
═══════════════════════

┌────────────────────┐
│   Test Execution   │
│   (LangGraph)      │
└─────────┬──────────┘
          │
          │ Event: TEST_PASSED
          │ {
          │   id: "evt_123",
          │   type: "TEST_PASSED",
          │   timestamp: "2026-01-28T12:00:00Z",
          │   tenant: {
          │     org_id: "org_001",
          │     project_id: "proj_A"
          │   },
          │   data: {
          │     test_id: "test_456",
          │     run_id: "run_789",
          │     duration_ms: 1234
          │   }
          │ }
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            EventProducer                                         │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. Serialize event to JSON                                                 │  │
│  │ 2. Generate partition key: "org_001:proj_A"                               │  │
│  │ 3. Determine topic: "argus-org_001-test-results"                          │  │
│  │ 4. Add headers:                                                            │  │
│  │    - correlation_id: request_id from middleware                            │  │
│  │    - event_type: "TEST_PASSED"                                            │  │
│  │    - schema_version: "1.0"                                                 │  │
│  │ 5. Compress with zstd                                                      │  │
│  │ 6. Produce with acks=all (wait for all replicas)                          │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────┬──────────────────────────────────┘
                                               │
                                               ▼
                            ┌─────────────────────────────────────┐
                            │         Redpanda / Kafka            │
                            │                                     │
                            │  Topic: argus-org_001-test-results  │
                            │  ┌─────────────────────────────────┐│
                            │  │ Partition 0 (org_001:proj_A)   ││
                            │  │  ┌─────┐ ┌─────┐ ┌─────┐       ││
                            │  │  │msg1 │ │msg2 │ │msg3 │ ...   ││
                            │  │  └─────┘ └─────┘ └─────┘       ││
                            │  └─────────────────────────────────┘│
                            │  ┌─────────────────────────────────┐│
                            │  │ Partition 1 (org_001:proj_B)   ││
                            │  │  ┌─────┐ ┌─────┐               ││
                            │  │  │msg1 │ │msg2 │ ...           ││
                            │  │  └─────┘ └─────┘               ││
                            │  └─────────────────────────────────┘│
                            │                                     │
                            │  Topic: argus-dlq (Dead Letter)     │
                            │  ┌─────────────────────────────────┐│
                            │  │  Failed events for retry        ││
                            │  │  30-day retention               ││
                            │  └─────────────────────────────────┘│
                            └─────────────────┬───────────────────┘
                                              │
CONSUMER SIDE                                 │
═════════════                                 │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             Event Consumers                                      │
│                                                                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────┐  │
│  │  Incident Correlator    │  │   Analytics Pipeline    │  │  Webhook Forwarder│  │
│  │  ───────────────────    │  │   ──────────────────    │  │  ────────────────│  │
│  │  - Detect patterns      │  │  - Aggregate metrics    │  │  - Forward to    │  │
│  │  - Create incidents     │  │  - Update dashboards    │  │    external      │  │
│  │  - Alert via PagerDuty  │  │  - Train ML models      │  │    systems       │  │
│  └─────────────────────────┘  └─────────────────────────┘  └─────────────────┘  │
│                                                                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────┐                       │
│  │   Cognee Consumer       │  │   GitHub Integration    │                       │
│  │   ────────────────      │  │   ──────────────────    │                       │
│  │  - Index to semantic    │  │  - Post PR comments     │                       │
│  │    memory               │  │  - Update check status  │                       │
│  │  - Train failure        │  │  - Create issues        │                       │
│  │    patterns             │  │                         │                       │
│  └─────────────────────────┘  └─────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────────┘

EVENT TYPES
═══════════

┌───────────────────────┬──────────────────────────────────────────────────────┐
│ Event Type            │ Data Payload                                         │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ TEST_RUN_STARTED      │ run_id, test_count, trigger, user_id                 │
│ TEST_PASSED           │ test_id, run_id, duration_ms, assertions_passed      │
│ TEST_FAILED           │ test_id, run_id, error_message, screenshot_url       │
│ TEST_SKIPPED          │ test_id, run_id, reason                              │
│ HEALING_ATTEMPTED     │ test_id, failure_type, suggested_fix                 │
│ HEALING_APPLIED       │ test_id, selector_before, selector_after, confidence │
│ HEALING_REJECTED      │ test_id, reason                                      │
│ TEST_RUN_COMPLETED    │ run_id, passed, failed, skipped, duration_ms         │
│ PRODUCTION_ERROR      │ error_id, source, severity, stack_trace             │
│ INCIDENT_CREATED      │ incident_id, severity, affected_tests                │
│ PR_COMMENT_POSTED     │ pr_number, repo, comment_id                          │
└───────────────────────┴──────────────────────────────────────────────────────┘
```

---

## 5. Self-Healing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SELF-HEALING DATA FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

TEST FAILURE OCCURS
═══════════════════

┌─────────────────────────┐
│     Test Execution      │
│  ┌───────────────────┐  │
│  │ Action: click     │  │
│  │ Selector: #submit │  │
│  │ Result: FAILED    │  │
│  │ Error: Element    │  │
│  │   not found       │  │
│  │ Screenshot: b64   │  │
│  └───────────────────┘  │
└───────────┬─────────────┘
            │
            │ FailureAnalysis
            │ {
            │   test_id: "test_123",
            │   failure_type: "selector_changed",
            │   root_cause: "Button ID changed",
            │   screenshot_at_failure: "base64...",
            │   confidence: 0.0
            │ }
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            SelfHealerAgent                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 1: Classify Failure Type                                              │  │
│  │ ─────────────────────────────                                              │  │
│  │ Input: error_message, screenshot, original selector                        │  │
│  │                                                                            │  │
│  │ Classification:                                                            │  │
│  │  ┌────────────────────┬─────────────────────────────────────────────────┐ │  │
│  │  │ selector_changed   │ Element exists but selector outdated            │ │  │
│  │  │ timing_issue       │ Element loads async, needs wait                 │ │  │
│  │  │ ui_change          │ UI redesigned, flow changed                     │ │  │
│  │  │ real_bug           │ Actual application bug                          │ │  │
│  │  │ unknown            │ Cannot determine cause                          │ │  │
│  │  └────────────────────┴─────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                        │
│                                         ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 2: Search Healing Patterns (Semantic)                                 │  │
│  │ ─────────────────────────────────────────────                              │  │
│  │                                                                            │  │
│  │ Query healing_patterns table:                                              │  │
│  │   SELECT * FROM healing_patterns                                           │  │
│  │   WHERE project_id = 'proj_A'                                             │  │
│  │   AND embedding <-> query_embedding < 0.3  -- cosine similarity           │  │
│  │   ORDER BY times_applied DESC                                              │  │
│  │   LIMIT 5;                                                                 │  │
│  │                                                                            │  │
│  │ Result:                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │ Pattern: #submit → button[type="submit"]                           │   │  │
│  │  │ Confidence: 0.95                                                   │   │  │
│  │  │ Times Applied: 12                                                  │   │  │
│  │  │ Success Rate: 100%                                                 │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                        │
│                                         ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 3: Generate Fix (if no pattern found)                                 │  │
│  │ ────────────────────────────────────────────                               │  │
│  │                                                                            │  │
│  │ Analyze screenshot with Claude Vision:                                     │  │
│  │   1. Identify all buttons in the UI                                        │  │
│  │   2. Match intent: "submit button" → visible "Submit" button              │  │
│  │   3. Generate alternative selectors:                                       │  │
│  │      - button[type="submit"]                                              │  │
│  │      - button:contains("Submit")                                          │  │
│  │      - .submit-btn                                                        │  │
│  │      - [data-testid="submit"]                                             │  │
│  │   4. Rank by stability (prefer data-testid > semantic > class > id)       │  │
│  │                                                                            │  │
│  │ Output:                                                                    │  │
│  │   suggested_fix: {                                                         │  │
│  │     selector_before: "#submit",                                            │  │
│  │     selector_after: "button[type='submit']",                              │  │
│  │     alternative_selectors: [...],                                          │  │
│  │     confidence: 0.87                                                       │  │
│  │   }                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                        │
│                                         ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 4: Apply or Queue for Approval                                        │  │
│  │ ──────────────────────────────────────                                     │  │
│  │                                                                            │  │
│  │ IF confidence >= settings.self_heal_confidence_threshold (0.8):            │  │
│  │   └─▶ Auto-apply fix                                                       │  │
│  │       - Update test spec with new selector                                 │  │
│  │       - Add to retry_queue                                                 │  │
│  │       - Store healing_pattern for future                                   │  │
│  │                                                                            │  │
│  │ ELSE IF require_healing_approval == True:                                  │  │
│  │   └─▶ Queue for human approval                                             │  │
│  │       - POST to /api/v1/approvals                                         │  │
│  │       - Wait for human decision via SSE                                    │  │
│  │       - Timeout after approval_timeout_seconds (300s)                      │  │
│  │                                                                            │  │
│  │ ELSE:                                                                      │  │
│  │   └─▶ Auto-apply with lower confidence                                     │  │
│  │       - Log warning about low confidence                                   │  │
│  │       - Mark as needs_review in test_results                               │  │
│  │                                                                            │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────┬──────────────────────────────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────────────┐
                            │       Database Updates              │
                            │                                     │
                            │ INSERT INTO healing_patterns        │
                            │   (project_id, selector_before,     │
                            │    selector_after, failure_type,    │
                            │    confidence, embedding,           │
                            │    git_context)                     │
                            │                                     │
                            │ UPDATE tests                        │
                            │   SET steps = [...updated...]       │
                            │   WHERE id = 'test_123'             │
                            │                                     │
                            │ PUBLISH event: HEALING_APPLIED      │
                            └─────────────────────────────────────┘

HEALING PATTERN SCHEMA
══════════════════════

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           healing_patterns table                                 │
├─────────────────┬────────────────────────────────────────────────────────────────┤
│ id              │ UUID PRIMARY KEY                                               │
│ project_id      │ UUID REFERENCES projects(id)                                   │
│ selector_before │ TEXT - Original failing selector                               │
│ selector_after  │ TEXT - Fixed selector                                          │
│ failure_type    │ TEXT - selector_changed, timing_issue, etc.                    │
│ confidence      │ FLOAT - 0.0 to 1.0                                            │
│ times_applied   │ INTEGER DEFAULT 0                                              │
│ success_count   │ INTEGER DEFAULT 0                                              │
│ embedding       │ VECTOR(1536) - For semantic search                            │
│ git_context     │ JSONB - {commit, branch, file_changed}                        │
│ created_at      │ TIMESTAMPTZ                                                   │
│ updated_at      │ TIMESTAMPTZ                                                   │
└─────────────────┴────────────────────────────────────────────────────────────────┘
```

---

*End of Data Flow Diagrams Document*
