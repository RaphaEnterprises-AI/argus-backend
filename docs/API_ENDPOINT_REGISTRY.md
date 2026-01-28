# Argus API Endpoint Registry

**Document Version:** 1.0
**Generated:** 2026-01-28
**API Version:** 2.10.0

---

## Overview

This document provides a complete registry of all API endpoints in the Argus backend, including authentication requirements, rate limits, and RBAC permissions.

**Total Endpoints:** 413+
**Total Router Files:** 50+
**Base URL:** `/api/v1`

---

## Quick Reference: Public Endpoints (No Auth)

| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | Health check |
| GET | /openapi.json | OpenAPI spec |
| GET | /docs | Swagger UI |
| GET | /redoc | ReDoc UI |
| POST | /api/v1/auth/device/authorize | Device auth start |
| POST | /api/v1/auth/device/token | Device token exchange |
| POST | /api/v1/auth/device/verify | Verify device code |
| POST | /api/v1/auth/device/refresh | Refresh device token |
| GET | /api/v1/oauth/{provider}/callback | OAuth callbacks |

---

## Endpoint Categories

### 1. Authentication & Authorization

**Router:** `src/api/security/device_auth.py`, `src/api/oauth.py`
**Prefix:** `/api/v1/auth`, `/api/v1/oauth`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| POST | /auth/device/authorize | ❌ | 10/min | - | Start device auth flow |
| POST | /auth/device/token | ❌ | 60/min | - | Exchange code for token |
| POST | /auth/device/verify | ❌ | 10/min | - | Verify user code |
| POST | /auth/device/refresh | ❌ | 30/min | - | Refresh access token |
| GET | /oauth/{provider}/authorize | ❌ | 10/min | - | Start OAuth flow |
| POST | /oauth/{provider}/callback | ❌ | 10/min | - | OAuth callback |
| DELETE | /oauth/{provider}/disconnect | ✅ JWT | 10/min | integration:delete | Disconnect OAuth |

### 2. Users & Profiles

**Router:** `src/api/users.py`, `src/api/ai_settings.py`
**Prefix:** `/api/v1/users`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /users/me | ✅ JWT | 120/min | - | Get current user |
| PUT | /users/me | ✅ JWT | 60/min | - | Update profile |
| GET | /users/me/preferences | ✅ JWT | 120/min | - | Get preferences |
| PUT | /users/me/preferences | ✅ JWT | 60/min | - | Update preferences |
| GET | /users/me/ai-config | ✅ JWT | 120/min | - | Get AI settings |
| PUT | /users/me/ai-config | ✅ JWT | 60/min | - | Update AI settings |
| GET | /users/me/organizations | ✅ JWT | 120/min | - | List user's orgs |
| DELETE | /users/me/sessions | ✅ JWT | 10/min | - | Logout all sessions |

### 3. Organizations

**Router:** `src/api/organizations.py`, `src/api/orgs.py`
**Prefix:** `/api/v1/organizations`, `/api/v1/orgs`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /organizations | ✅ JWT | 120/min | org:read | List organizations |
| POST | /organizations | ✅ JWT | 10/min | - | Create organization |
| GET | /organizations/{id} | ✅ JWT | 120/min | org:read | Get organization |
| PUT | /organizations/{id} | ✅ JWT | 60/min | org:write | Update organization |
| DELETE | /organizations/{id} | ✅ JWT | 5/min | org:delete | Delete organization |
| GET | /organizations/{id}/members | ✅ JWT | 120/min | org:read | List members |
| POST | /organizations/{id}/members | ✅ JWT | 30/min | org:manage_members | Add member |
| DELETE | /organizations/{id}/members/{user_id} | ✅ JWT | 30/min | org:manage_members | Remove member |
| PUT | /organizations/{id}/members/{user_id}/role | ✅ JWT | 30/min | org:manage_members | Change role |

### 4. Projects

**Router:** `src/api/projects.py`
**Prefix:** `/api/v1/projects`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /projects | ✅ JWT/API | 120/min | project:read | List projects |
| POST | /projects | ✅ JWT | 30/min | project:write | Create project |
| GET | /projects/{id} | ✅ JWT/API | 120/min | project:read | Get project |
| PUT | /projects/{id} | ✅ JWT | 60/min | project:write | Update project |
| DELETE | /projects/{id} | ✅ JWT | 5/min | project:delete | Delete project |
| GET | /projects/{id}/settings | ✅ JWT | 120/min | project:read | Get settings |
| PUT | /projects/{id}/settings | ✅ JWT | 30/min | project:manage_settings | Update settings |
| GET | /projects/{id}/integrations | ✅ JWT | 120/min | integration:read | List integrations |

### 5. Tests

**Router:** `src/api/tests.py`
**Prefix:** `/api/v1/tests`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /tests | ✅ JWT/API | 120/min | test:read | List tests |
| POST | /tests | ✅ JWT/API | 60/min | test:write | Create test |
| GET | /tests/{id} | ✅ JWT/API | 120/min | test:read | Get test |
| PUT | /tests/{id} | ✅ JWT/API | 60/min | test:write | Update test |
| DELETE | /tests/{id} | ✅ JWT/API | 30/min | test:delete | Delete test |
| POST | /tests/{id}/run | ✅ JWT/API | 10/min | test:execute | Execute test |
| POST | /tests/{id}/duplicate | ✅ JWT/API | 30/min | test:write | Duplicate test |
| GET | /tests/{id}/history | ✅ JWT/API | 120/min | results:read | Get run history |
| GET | /tests/{id}/results | ✅ JWT/API | 120/min | results:read | Get results |

### 6. Test Runs

**Router:** `src/api/tests.py`
**Prefix:** `/api/v1`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| POST | /run | ✅ JWT/API | 10/min | test:execute | Start test run |
| GET | /status/{job_id} | ✅ JWT/API | 120/min | results:read | Get run status |
| POST | /cancel/{job_id} | ✅ JWT/API | 30/min | test:execute | Cancel run |

### 7. Discovery

**Router:** `src/api/discovery.py`
**Prefix:** `/api/v1/discovery`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| POST | /discovery/sessions | ✅ JWT/API | 10/min | test:write | Start discovery |
| GET | /discovery/sessions | ✅ JWT/API | 100/min | test:read | List sessions |
| GET | /discovery/sessions/{id} | ✅ JWT/API | 100/min | test:read | Get session |
| DELETE | /discovery/sessions/{id} | ✅ JWT/API | 30/min | test:delete | Delete session |
| GET | /discovery/sessions/{id}/flows | ✅ JWT/API | 100/min | test:read | Get flows |
| POST | /discovery/flows/{id}/generate-test | ✅ JWT/API | 30/min | test:write | Generate test |
| GET | /discovery/sessions/{id}/screenshots | ✅ JWT/API | 100/min | test:read | Get screenshots |
| POST | /discovery/sessions/{id}/stop | ✅ JWT/API | 30/min | test:execute | Stop discovery |

### 8. Healing

**Router:** `src/api/healing.py`
**Prefix:** `/api/v1/healing`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /healing/suggestions | ✅ JWT/API | 120/min | healing:read | List suggestions |
| GET | /healing/suggestions/{id} | ✅ JWT/API | 120/min | healing:read | Get suggestion |
| POST | /healing/suggestions/{id}/apply | ✅ JWT/API | 30/min | healing:approve | Apply suggestion |
| POST | /healing/suggestions/{id}/reject | ✅ JWT/API | 30/min | healing:approve | Reject suggestion |
| GET | /healing/patterns | ✅ JWT/API | 120/min | healing:read | List patterns |
| GET | /healing/history/{test_id} | ✅ JWT/API | 120/min | healing:read | Get history |
| PUT | /healing/config | ✅ JWT | 30/min | healing:configure | Update config |
| GET | /healing/stats | ✅ JWT/API | 120/min | healing:read | Get statistics |

### 9. Visual AI

**Router:** `src/api/visual_ai.py`
**Prefix:** `/api/v1/visual`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| POST | /visual/capture | ✅ JWT/API | 30/min | test:execute | Capture snapshot |
| POST | /visual/compare | ✅ JWT/API | 30/min | test:execute | Compare snapshots |
| POST | /visual/analyze | ✅ JWT/API | 30/min | test:execute | AI analysis |
| GET | /visual/baselines | ✅ JWT/API | 120/min | test:read | List baselines |
| POST | /visual/baselines | ✅ JWT/API | 30/min | test:write | Create baseline |
| PUT | /visual/baselines/{id} | ✅ JWT/API | 30/min | test:write | Update baseline |
| DELETE | /visual/baselines/{id} | ✅ JWT/API | 30/min | test:delete | Delete baseline |
| GET | /visual/history | ✅ JWT/API | 120/min | results:read | Get history |
| POST | /visual/responsive | ✅ JWT/API | 10/min | test:execute | Responsive test |
| POST | /visual/accessibility | ✅ JWT/API | 10/min | test:execute | A11y analysis |

### 10. Streaming (SSE)

**Router:** `src/api/streaming.py`
**Prefix:** `/api/v1/stream`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /stream/test/{run_id} | ✅ JWT/API* | 5/min | results:read | Stream test run |
| GET | /stream/chat/{thread_id} | ✅ JWT/API* | 10/min | test:execute | Stream chat |
| GET | /stream/discovery/{session_id} | ✅ JWT/API* | 5/min | test:read | Stream discovery |

*Auth can be passed via query param `?token=` for SSE clients

### 11. Chat

**Router:** `src/api/chat.py`
**Prefix:** `/api/v1/chat`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| POST | /chat/messages | ✅ JWT | 30/min | test:execute | Send message |
| GET | /chat/threads | ✅ JWT | 120/min | test:read | List threads |
| GET | /chat/threads/{id} | ✅ JWT | 120/min | test:read | Get thread |
| DELETE | /chat/threads/{id} | ✅ JWT | 30/min | test:delete | Delete thread |
| GET | /chat/threads/{id}/messages | ✅ JWT | 120/min | test:read | Get messages |

### 12. Approvals (Human-in-the-Loop)

**Router:** `src/api/approvals.py`
**Prefix:** `/api/v1/approvals`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /approvals/pending | ✅ JWT | 120/min | healing:approve | List pending |
| POST | /approvals/{thread_id}/approve | ✅ JWT | 30/min | healing:approve | Approve action |
| POST | /approvals/{thread_id}/reject | ✅ JWT | 30/min | healing:approve | Reject action |
| POST | /approvals/{thread_id}/modify | ✅ JWT | 30/min | healing:approve | Modify & approve |

### 13. Time Travel (Debugging)

**Router:** `src/api/time_travel.py`
**Prefix:** `/api/v1/time-travel`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /time-travel/history/{thread_id} | ✅ JWT | 120/min | results:read | Get checkpoints |
| GET | /time-travel/state/{thread_id}/{checkpoint_id} | ✅ JWT | 120/min | results:read | Get state |
| POST | /time-travel/replay/{thread_id}/{checkpoint_id} | ✅ JWT | 10/min | test:execute | Replay from |
| POST | /time-travel/fork/{thread_id}/{checkpoint_id} | ✅ JWT | 10/min | test:execute | Fork execution |

### 14. API Keys

**Router:** `src/api/api_keys.py`
**Prefix:** `/api/v1/api-keys`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /api-keys | ✅ JWT | 120/min | api_key:read | List keys |
| POST | /api-keys | ✅ JWT | 10/min | api_key:create | Create key |
| GET | /api-keys/{id} | ✅ JWT | 120/min | api_key:read | Get key |
| DELETE | /api-keys/{id} | ✅ JWT | 10/min | api_key:revoke | Revoke key |
| POST | /api-keys/{id}/rotate | ✅ JWT | 5/min | api_key:rotate | Rotate key |

### 15. Audit

**Router:** `src/api/audit.py`
**Prefix:** `/api/v1/audit`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /audit/organizations/{org_id}/logs | ✅ JWT | 120/min | audit:read | Get audit logs |
| GET | /audit/organizations/{org_id}/logs/export | ✅ JWT | 5/min | audit:export | Export logs |
| GET | /audit/organizations/{org_id}/security-events | ✅ JWT | 120/min | audit:read | Security events |

### 16. Webhooks

**Router:** `src/api/webhooks.py`, `src/api/webhooks/github.py`
**Prefix:** `/api/v1/webhooks`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| POST | /webhooks/sentry | ✅ API* | 100/min | - | Sentry webhook |
| POST | /webhooks/datadog | ✅ API* | 100/min | - | Datadog webhook |
| POST | /webhooks/fullstory | ✅ API* | 100/min | - | FullStory webhook |
| POST | /webhooks/logrocket | ✅ API* | 100/min | - | LogRocket webhook |
| POST | /webhooks/newrelic | ✅ API* | 100/min | - | NewRelic webhook |
| POST | /webhooks/bugsnag | ✅ API* | 100/min | - | Bugsnag webhook |
| POST | /webhooks/rollbar | ✅ API* | 100/min | - | Rollbar webhook |
| POST | /webhooks/vcs/github | ✅ Signature | 100/min | - | GitHub webhook |
| POST | /webhooks/vcs/gitlab | ✅ Token | 100/min | - | GitLab webhook |

*Webhook endpoints verify signatures rather than bearer tokens

### 17. Integrations

**Router:** `src/api/integrations.py`, `src/api/integrations_ai.py`
**Prefix:** `/api/v1/integrations`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /integrations | ✅ JWT | 120/min | integration:read | List integrations |
| POST | /integrations/{type} | ✅ JWT | 10/min | integration:configure | Add integration |
| GET | /integrations/{id} | ✅ JWT | 120/min | integration:read | Get integration |
| PUT | /integrations/{id} | ✅ JWT | 30/min | integration:configure | Update integration |
| DELETE | /integrations/{id} | ✅ JWT | 10/min | integration:delete | Delete integration |
| POST | /integrations/{id}/sync | ✅ JWT | 10/min | integration:configure | Sync integration |
| POST | /integrations/{id}/test | ✅ JWT | 10/min | integration:configure | Test integration |

### 18. Reports

**Router:** `src/api/reports.py`
**Prefix:** `/api/v1/reports`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /reports | ✅ JWT/API | 120/min | results:read | List reports |
| GET | /reports/{id} | ✅ JWT/API | 120/min | results:read | Get report |
| POST | /reports/{id}/export | ✅ JWT | 10/min | results:export | Export report |
| DELETE | /reports/{id} | ✅ JWT | 30/min | results:delete | Delete report |

### 19. Export (Code Generation)

**Router:** `src/api/export.py`
**Prefix:** `/api/v1/export`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| POST | /export/playwright | ✅ JWT/API | 30/min | results:export | Export Playwright |
| POST | /export/cypress | ✅ JWT/API | 30/min | results:export | Export Cypress |
| POST | /export/selenium | ✅ JWT/API | 30/min | results:export | Export Selenium |
| POST | /export/puppeteer | ✅ JWT/API | 30/min | results:export | Export Puppeteer |
| GET | /export/formats | ✅ JWT/API | 120/min | results:read | List formats |

### 20. Quality

**Router:** `src/api/quality.py`
**Prefix:** `/api/v1/quality`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /quality/score/{project_id} | ✅ JWT/API | 120/min | test:read | Get quality score |
| GET | /quality/stats | ✅ JWT/API | 120/min | test:read | Get statistics |
| GET | /quality/trends | ✅ JWT/API | 120/min | test:read | Get trends |
| POST | /quality/analyze | ✅ JWT/API | 10/min | test:execute | Run analysis |

### 21. Scheduling

**Router:** `src/api/scheduling.py`
**Prefix:** `/api/v1/schedules`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /schedules | ✅ JWT | 120/min | test:read | List schedules |
| POST | /schedules | ✅ JWT | 30/min | test:write | Create schedule |
| GET | /schedules/{id} | ✅ JWT | 120/min | test:read | Get schedule |
| PUT | /schedules/{id} | ✅ JWT | 30/min | test:write | Update schedule |
| DELETE | /schedules/{id} | ✅ JWT | 30/min | test:delete | Delete schedule |
| POST | /schedules/{id}/pause | ✅ JWT | 30/min | test:write | Pause schedule |
| POST | /schedules/{id}/resume | ✅ JWT | 30/min | test:write | Resume schedule |
| POST | /schedules/{id}/trigger | ✅ JWT | 10/min | test:execute | Trigger now |

### 22. Notifications

**Router:** `src/api/notifications.py`
**Prefix:** `/api/v1/notifications`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /notifications | ✅ JWT | 120/min | - | List notifications |
| PUT | /notifications/{id}/read | ✅ JWT | 60/min | - | Mark as read |
| PUT | /notifications/read-all | ✅ JWT | 30/min | - | Mark all read |
| GET | /notifications/preferences | ✅ JWT | 120/min | - | Get preferences |
| PUT | /notifications/preferences | ✅ JWT | 30/min | - | Update preferences |
| POST | /notifications/subscribe/{event} | ✅ JWT | 30/min | - | Subscribe to event |
| DELETE | /notifications/unsubscribe/{event} | ✅ JWT | 30/min | - | Unsubscribe |

### 23. Collaboration

**Router:** `src/api/collaboration.py`
**Prefix:** `/api/v1/collaboration`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /collaboration/cursors | ✅ JWT | 120/min | test:read | Get cursors |
| POST | /collaboration/cursor | ✅ JWT | 60/min | test:read | Update cursor |
| GET | /collaboration/presence | ✅ JWT | 120/min | test:read | Get presence |
| POST | /collaboration/presence | ✅ JWT | 60/min | test:read | Update presence |
| GET | /collaboration/documents/{id} | ✅ JWT | 120/min | test:read | Get document |
| POST | /collaboration/documents/{id}/edit | ✅ JWT | 60/min | test:write | Edit document |

### 24. Recordings

**Router:** `src/api/recording.py`
**Prefix:** `/api/v1/recording`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| POST | /recording/snippet | ✅ JWT/API | 30/min | test:write | Save snippet |
| POST | /recording/sessions | ✅ JWT/API | 10/min | test:write | Start session |
| GET | /recording/sessions/{id} | ✅ JWT/API | 120/min | test:read | Get session |
| POST | /recording/sessions/{id}/events | ✅ JWT/API | 60/min | test:write | Add events |
| POST | /recording/sessions/{id}/convert | ✅ JWT/API | 10/min | test:write | Convert to test |

### 25. Browser

**Router:** `src/api/browser.py`
**Prefix:** `/api/v1/browser`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| POST | /browser/sessions | ✅ JWT/API | 10/min | test:execute | Create session |
| GET | /browser/sessions/{id} | ✅ JWT/API | 120/min | test:read | Get session |
| DELETE | /browser/sessions/{id} | ✅ JWT/API | 30/min | test:execute | Close session |
| GET | /browser/status | ✅ JWT/API | 120/min | test:read | Pool status |

### 26. Infrastructure

**Router:** `src/api/infra_optimizer.py`
**Prefix:** `/api/v1/infra`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /infra/recommendations | ✅ JWT | 120/min | admin:full_access | Get recommendations |
| POST | /infra/analyze | ✅ JWT | 5/min | admin:full_access | Run analysis |
| POST | /infra/apply/{recommendation_id} | ✅ JWT | 5/min | admin:full_access | Apply recommendation |
| GET | /infra/cost-history | ✅ JWT | 120/min | admin:full_access | Get cost history |

### 27. Health

**Router:** `src/api/data_layer_health.py`
**Prefix:** `/api/v1/health`

| Method | Path | Auth | Rate Limit | Permission | Description |
|--------|------|------|------------|------------|-------------|
| GET | /health | ❌ | 1000/min | - | Basic health |
| GET | /health/services | ✅ JWT | 120/min | - | Service health |
| GET | /health/database | ✅ JWT | 60/min | - | Database health |
| GET | /health/selenium-test | ✅ JWT | 10/min | - | Selenium test |

---

## Rate Limit Response Headers

All responses include rate limit information:

```http
X-RateLimit-Limit: 120
X-RateLimit-Remaining: 119
X-RateLimit-Reset: 1706443260
```

When rate limited (429 response):
```http
Retry-After: 45
```

---

## Error Response Format

All endpoints return errors in a consistent format:

```json
{
  "detail": "Error description",
  "request_id": "abc123-def456-..."
}
```

For validation errors (422):
```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "Error message",
      "type": "value_error"
    }
  ]
}
```

---

## Versioning

Current API version: `v1`

Version header: `X-API-Version: 2.10.0`

---

*End of API Endpoint Registry*
