# API Layer Migration Plan: Direct Supabase to Backend API

## Executive Summary

**Problem**: The Argus dashboard makes ~216 direct Supabase queries across 25+ hook files, bypassing the FastAPI backend. This creates security, maintainability, and observability issues.

**Solution**: Migrate all direct Supabase queries to use the FastAPI backend API layer.

**Effort Estimate**: 8-12 weeks (2 developers)

---

## Current State Analysis

### Direct Supabase Queries by Domain

| Domain | Files | Queries | Tables | Priority | Effort |
|--------|-------|---------|--------|----------|--------|
| **Tests & Runs** | 3 | 42 | tests, test_runs, test_results | P1 | 2 weeks |
| **Discovery** | 3 | 28 | discovery_sessions, discovered_pages, discovered_flows, discovery_patterns | P1 | 1.5 weeks |
| **Scheduling** | 1 | 8 | test_schedules, schedule_runs | P2 | 3 days |
| **Notifications** | 1 | 12 | notification_channels, notification_rules, notification_logs | P2 | 3 days |
| **CI/CD** | 1 | 12 | ci_pipelines, ci_builds, ci_deployments | P2 | 3 days |
| **Visual Testing** | 1 | 14 | visual_baselines, visual_comparisons | P2 | 4 days |
| **Insights & Analytics** | 1 | 14 | ai_insights, test_runs, test_results | P2 | 3 days |
| **Quality & Accessibility** | 2 | 18 | quality_audits, accessibility_issues | P2 | 4 days |
| **Chat** | 1 | 6 | chat_conversations, chat_messages | P3 | 2 days |
| **Activity Logs** | 2 | 8 | activity_logs | P3 | 2 days |
| **Orchestrator** | 2 | 8 | langgraph_checkpoints, activity_logs | P3 | 3 days |
| **Live Sessions** | 1 | 8 | live_sessions, activity_logs | P3 | 2 days |
| **Parameterized Tests** | 1 | 22 | parameterized_tests, parameter_sets, parameterized_results, iteration_results | P3 | 4 days |
| **Performance Tests** | 1 | 8 | performance_tests | P3 | 2 days |
| **Global Tests** | 1 | 8 | global_tests, global_test_results | P3 | 2 days |
| **Flaky Tests** | 1 | 10 | tests, test_results, healing_patterns | P2 | 3 days |
| **Projects** | 1 | 4 | projects | P1 | 1 day |
| **Reports** | 1 | 4 | test_runs | P3 | 1 day |

**Total: ~216 queries across 25 files and 34 tables**

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DASHBOARD (Next.js)                                  │
│                                                                              │
│  lib/hooks/use-*.ts  ──────────────────────────────────────────────────────│
│        │                                                                     │
│        ↓                                                                     │
│  lib/api-client.ts  (Unified API Client)                                    │
│        │                                                                     │
│        │  • Automatic auth token injection                                  │
│        │  • Organization header injection                                   │
│        │  • Request/response logging                                        │
│        │  • Error transformation                                            │
│        │  • Retry logic                                                     │
│                                                                              │
└──────────────────────────────────│───────────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                                       │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Middleware Stack:                                                       │ │
│  │   CORSMiddleware → AuthenticationMiddleware → RateLimitMiddleware →    │ │
│  │   TenantMiddleware → RequestLoggingMiddleware                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                   │                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ API Routers (src/api/):                                                │ │
│  │                                                                         │ │
│  │   /api/v1/tests/*           - Tests CRUD + execution                   │ │
│  │   /api/v1/test-runs/*       - Test run management                      │ │
│  │   /api/v1/discovery/*       - Discovery sessions                       │ │
│  │   /api/v1/schedules/*       - Test scheduling                          │ │
│  │   /api/v1/notifications/*   - Notification channels                    │ │
│  │   /api/v1/visual/*          - Visual testing                           │ │
│  │   /api/v1/insights/*        - AI insights                              │ │
│  │   /api/v1/quality/*         - Quality audits                           │ │
│  │   /api/v1/chat/*            - Chat conversations                       │ │
│  │   /api/v1/activity/*        - Activity logs                            │ │
│  │   /api/v1/cicd/*            - CI/CD integration                        │ │
│  │   /api/v1/performance/*     - Performance tests                        │ │
│  │   /api/v1/parameterized/*   - Parameterized tests                      │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                   │                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Services Layer (src/services/):                                        │ │
│  │   • Business logic                                                     │ │
│  │   • Data validation                                                    │ │
│  │   • Cross-cutting concerns                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                   │                                          │
│                                   ↓                                          │
│                         Supabase Client                                      │
│                              │                                               │
└──────────────────────────────│───────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Supabase PostgreSQL                                        │
│                    (RLS as defense-in-depth)                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2) - P1

**Goal**: Establish patterns and migrate core flows

#### 1.1 Backend API Patterns
- [ ] Create standardized response schemas (Pydantic models)
- [ ] Create service layer pattern template
- [ ] Add comprehensive request logging middleware
- [ ] Add API versioning structure
- [ ] Create OpenAPI documentation standards

#### 1.2 Dashboard API Client Enhancement
- [ ] Enhance `lib/api-client.ts` with:
  - Request interceptors
  - Response transformers
  - Automatic retry with exponential backoff
  - Request deduplication
  - Cache integration (TanStack Query)

#### 1.3 Tests Domain Migration (42 queries)
- [ ] Backend: `POST /api/v1/tests` - Create test
- [ ] Backend: `GET /api/v1/tests` - List tests
- [ ] Backend: `GET /api/v1/tests/{id}` - Get test
- [ ] Backend: `PUT /api/v1/tests/{id}` - Update test
- [ ] Backend: `DELETE /api/v1/tests/{id}` - Delete test
- [ ] Backend: `POST /api/v1/test-runs` - Create run
- [ ] Backend: `GET /api/v1/test-runs` - List runs
- [ ] Backend: `GET /api/v1/test-runs/{id}` - Get run with results
- [ ] Backend: `POST /api/v1/test-runs/{id}/results` - Add result
- [ ] Dashboard: Migrate `use-tests.ts` (22 queries)
- [ ] Dashboard: Migrate `use-test-library.ts` (10 queries)
- [ ] Dashboard: Migrate `use-realtime-tests.ts` (4 queries)
- [ ] Dashboard: Migrate `use-flaky-tests.ts` (6 queries)

### Phase 2: Discovery & Projects (Week 3-4) - P1

#### 2.1 Discovery Domain Migration (28 queries)
- [ ] Backend: `POST /api/v1/discovery/sessions` - Start session
- [ ] Backend: `GET /api/v1/discovery/sessions` - List sessions
- [ ] Backend: `GET /api/v1/discovery/sessions/{id}` - Get session
- [ ] Backend: `GET /api/v1/discovery/sessions/{id}/pages` - Get pages
- [ ] Backend: `GET /api/v1/discovery/sessions/{id}/flows` - Get flows
- [ ] Backend: `POST /api/v1/discovery/sessions/{id}/flows` - Create flow
- [ ] Backend: `PUT /api/v1/discovery/flows/{id}` - Update flow
- [ ] Backend: `GET /api/v1/discovery/patterns` - Get patterns
- [ ] Dashboard: Migrate `use-discovery.ts` (17 queries)
- [ ] Dashboard: Migrate `use-discovery-session.ts` (8 queries)
- [ ] Dashboard: Migrate `use-discovery-for-visual.ts` (3 queries)

#### 2.2 Projects Domain Migration (4 queries)
- [ ] Backend: Verify existing `/api/v1/projects` endpoints
- [ ] Dashboard: Migrate `use-projects.ts` (4 queries)

### Phase 3: Scheduling & Notifications (Week 5-6) - P2

#### 3.1 Schedules Domain Migration (8 queries)
- [ ] Backend: `POST /api/v1/schedules` - Create schedule
- [ ] Backend: `GET /api/v1/schedules` - List schedules
- [ ] Backend: `PUT /api/v1/schedules/{id}` - Update schedule
- [ ] Backend: `DELETE /api/v1/schedules/{id}` - Delete schedule
- [ ] Backend: `GET /api/v1/schedules/{id}/runs` - Get runs
- [ ] Dashboard: Migrate `use-schedules.ts` (8 queries)

#### 3.2 Notifications Domain Migration (12 queries)
- [ ] Backend: `POST /api/v1/notifications/channels` - Create channel
- [ ] Backend: `GET /api/v1/notifications/channels` - List channels
- [ ] Backend: `PUT /api/v1/notifications/channels/{id}` - Update channel
- [ ] Backend: `DELETE /api/v1/notifications/channels/{id}` - Delete channel
- [ ] Backend: `GET /api/v1/notifications/logs` - Get logs
- [ ] Dashboard: Migrate `use-notifications.ts` (12 queries)

### Phase 4: Visual & Quality (Week 7-8) - P2

#### 4.1 Visual Testing Migration (14 queries)
- [ ] Backend: `POST /api/v1/visual/baselines` - Create baseline
- [ ] Backend: `GET /api/v1/visual/baselines` - List baselines
- [ ] Backend: `POST /api/v1/visual/comparisons` - Create comparison
- [ ] Backend: `GET /api/v1/visual/comparisons` - List comparisons
- [ ] Backend: `PUT /api/v1/visual/comparisons/{id}/approve` - Approve
- [ ] Dashboard: Migrate `use-visual.ts` (14 queries)

#### 4.2 Quality & Accessibility Migration (18 queries)
- [ ] Backend: `POST /api/v1/quality/audits` - Create audit
- [ ] Backend: `GET /api/v1/quality/audits` - List audits
- [ ] Backend: `GET /api/v1/quality/audits/{id}` - Get audit with issues
- [ ] Backend: `POST /api/v1/quality/audits/{id}/issues` - Add issues
- [ ] Backend: `PUT /api/v1/quality/issues/{id}` - Update issue
- [ ] Dashboard: Migrate `use-quality.ts` (8 queries)
- [ ] Dashboard: Migrate `use-accessibility.ts` (10 queries)

### Phase 5: Analytics & Insights (Week 9) - P2

#### 5.1 Insights Migration (14 queries)
- [ ] Backend: `GET /api/v1/insights/ai` - Get AI insights
- [ ] Backend: `GET /api/v1/insights/trends` - Get trends
- [ ] Backend: `GET /api/v1/insights/coverage-gaps` - Get coverage gaps
- [ ] Backend: `GET /api/v1/insights/test-health` - Get test health
- [ ] Dashboard: Migrate `use-insights.ts` (14 queries)

#### 5.2 CI/CD Migration (12 queries)
- [ ] Backend: Verify existing `/api/v1/cicd` endpoints
- [ ] Dashboard: Migrate `use-cicd.ts` (12 queries)

### Phase 6: Remaining Domains (Week 10-12) - P3

#### 6.1 Chat & Activity (14 queries)
- [ ] Backend: `POST /api/v1/chat/conversations` - Create conversation
- [ ] Backend: `GET /api/v1/chat/conversations` - List conversations
- [ ] Backend: `GET /api/v1/chat/conversations/{id}/messages` - Get messages
- [ ] Backend: `GET /api/v1/activity/logs` - Get activity logs
- [ ] Dashboard: Migrate `use-chat.ts` (6 queries)
- [ ] Dashboard: Migrate `use-activity.ts` (8 queries)

#### 6.2 Orchestrator & Live Sessions (16 queries)
- [ ] Backend: `GET /api/v1/orchestrator/checkpoints` - Get checkpoints
- [ ] Backend: `GET /api/v1/live-sessions` - Get sessions
- [ ] Backend: `POST /api/v1/live-sessions` - Create session
- [ ] Dashboard: Migrate `use-orchestrator.ts` (8 queries)
- [ ] Dashboard: Migrate `use-live-session.ts` (8 queries)

#### 6.3 Parameterized Tests (22 queries)
- [ ] Backend: Full CRUD for parameterized tests
- [ ] Backend: Parameter sets management
- [ ] Backend: Results and iterations
- [ ] Dashboard: Migrate `use-parameterized.ts` (22 queries)

#### 6.4 Performance & Global Tests (16 queries)
- [ ] Backend: Performance test endpoints
- [ ] Backend: Global test endpoints
- [ ] Dashboard: Migrate `use-performance.ts` (8 queries)
- [ ] Dashboard: Migrate `use-global.ts` (8 queries)

#### 6.5 Reports (4 queries)
- [ ] Dashboard: Migrate `use-reports.ts` (4 queries)

---

## Effort Estimation

| Phase | Scope | Backend Effort | Dashboard Effort | Total |
|-------|-------|---------------|------------------|-------|
| Phase 1 | Foundation + Tests | 5 days | 5 days | 10 days |
| Phase 2 | Discovery + Projects | 4 days | 4 days | 8 days |
| Phase 3 | Schedules + Notifications | 3 days | 3 days | 6 days |
| Phase 4 | Visual + Quality | 4 days | 4 days | 8 days |
| Phase 5 | Analytics + CI/CD | 3 days | 3 days | 6 days |
| Phase 6 | Remaining Domains | 6 days | 6 days | 12 days |
| **Total** | | **25 days** | **25 days** | **50 days** |

**With buffer (20%)**: ~60 working days = **12 weeks with 1 developer** or **6 weeks with 2 developers**

---

## Success Metrics

1. **Zero direct Supabase queries** in dashboard hooks (excluding test files)
2. **100% API coverage** for all data operations
3. **Request logging** for all API calls (audit trail)
4. **<100ms p95 latency** increase from API layer
5. **Consistent error handling** across all endpoints

---

## Migration Pattern (Per Hook)

```typescript
// BEFORE: Direct Supabase
const { data, error } = await supabase
  .from('tests')
  .select('*')
  .eq('project_id', projectId);

// AFTER: Via API
const data = await apiClient.get<Test[]>(
  `/api/v1/tests?project_id=${projectId}`
);
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking changes during migration | Feature flags per domain |
| Performance regression | Add Redis caching layer |
| Real-time features break | Keep Supabase realtime for subscriptions only |
| Extended timeline | Prioritize P1 domains, defer P3 |

---

## Real-time Considerations

**Keep Supabase Realtime for**:
- Live test execution updates
- Real-time discovery progress
- Chat message streaming

**Pattern**: Use API for mutations, Supabase realtime for subscriptions only.

```typescript
// Mutation via API
await apiClient.post('/api/v1/test-runs', { testId });

// Real-time updates via Supabase
supabase
  .channel('test-run-updates')
  .on('postgres_changes', { table: 'test_runs' }, handleUpdate)
  .subscribe();
```
