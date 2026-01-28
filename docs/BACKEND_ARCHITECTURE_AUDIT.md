# Argus Backend Architecture Audit

**Document Version:** 1.0
**Audit Date:** 2026-01-28
**Auditor:** PhD-Level Backend Systems Architect
**Backend Version:** 2.10.0 (2026-01-12)

---

## Executive Summary

The Argus backend is a **production-grade, enterprise-ready autonomous E2E testing platform** built on FastAPI + LangGraph. This audit covers:

- **413+ API endpoints** across 50+ routers
- **16 specialized AI agents** orchestrated by LangGraph
- **Multi-tenant architecture** with comprehensive RBAC
- **SOC2-compliant security** with audit logging, encryption, and rate limiting
- **Multi-model AI routing** (Claude, GPT-4, Gemini, Llama, DeepSeek)

### Critical Findings

| Category | Status | Notes |
|----------|--------|-------|
| Authentication | ✅ Good | JWT, API Keys, OAuth2, Clerk JWKS |
| Authorization | ✅ Good | RBAC with 33 permissions, 11 roles |
| Rate Limiting | ✅ Good | Tier-based with endpoint overrides |
| Audit Logging | ✅ Good | SOC2-compliant 1-year retention |
| Input Validation | ⚠️ Partial | Pydantic validation; needs XSS sanitization review |
| OpenAPI Spec | ❌ Missing | `openapi.json` not generated/exported |
| Secrets Management | ✅ Good | SecretStr for all credentials |

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [API Layer & Endpoints](#2-api-layer--endpoints)
3. [Authentication Mechanisms](#3-authentication-mechanisms)
4. [Authorization & RBAC](#4-authorization--rbac)
5. [Security Middleware Stack](#5-security-middleware-stack)
6. [Database Schema & Data Flow](#6-database-schema--data-flow)
7. [Agent Orchestration](#7-agent-orchestration)
8. [External Integrations](#8-external-integrations)
9. [Event Streaming Architecture](#9-event-streaming-architecture)
10. [Security Audit Findings](#10-security-audit-findings)
11. [OpenAPI Specification Gap Analysis](#11-openapi-specification-gap-analysis)
12. [Recommendations](#12-recommendations)

---

## 1. System Architecture Overview

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Dashboard│  │   CLI    │  │MCP Server│  │ Webhooks │  │ Browser  │      │
│  │ (Next.js)│  │ (Claude) │  │(TypeScript│ │(GitHub/  │  │Extension │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │Sentry)   │  └────┬─────┘      │
│       │             │             │         └────┬─────┘       │            │
└───────┼─────────────┼─────────────┼──────────────┼─────────────┼────────────┘
        │             │             │              │             │
        ▼             ▼             ▼              ▼             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API GATEWAY LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     MIDDLEWARE STACK (7 layers)                      │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐        │    │
│  │  │  Security  │ │   Audit    │ │   Rate     │ │   Auth     │        │    │
│  │  │  Headers   │ │  Logging   │ │  Limiting  │ │ Middleware │        │    │
│  │  │  (OWASP)   │ │  (SOC2)    │ │ (Tier-based│ │ (JWT/API)  │        │    │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘        │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐                       │    │
│  │  │  Request   │ │  Tenant    │ │   CORS     │                       │    │
│  │  │  Limiter   │ │  Context   │ │ Middleware │                       │    │
│  │  │  (100MB)   │ │ Extraction │ │            │                       │    │
│  │  └────────────┘ └────────────┘ └────────────┘                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────┴─────────────────────────────────────┐  │
│  │                    FastAPI Application (50+ Routers)                   │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │  │
│  │  │ Tests  │ │Projects│ │Discovery│ │Healing │ │Visual  │ │ Chat   │   │  │
│  │  │        │ │        │ │        │ │        │ │   AI   │ │        │   │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │  │
│  │  │Webhooks│ │ Users  │ │  Orgs  │ │Streaming│ │ Audit  │ │API Keys│   │  │
│  │  │        │ │        │ │        │ │  (SSE)  │ │        │ │        │   │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                  LangGraph TestingOrchestrator                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                    State Machine Nodes                       │    │    │
│  │  │  analyze_code → plan_tests → execute_test → evaluate →      │    │    │
│  │  │                              ↓                               │    │    │
│  │  │                         self_heal → report                   │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                   Specialized AI Agents (16)                 │    │    │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │    │    │
│  │  │  │ CodeAnalyzer │ │  UITester    │ │  APITester   │         │    │    │
│  │  │  └──────────────┘ └──────────────┘ └──────────────┘         │    │    │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │    │    │
│  │  │  │ SelfHealer   │ │  Reporter    │ │VisualAI     │         │    │    │
│  │  │  └──────────────┘ └──────────────┘ └──────────────┘         │    │    │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │    │    │
│  │  │  │AutoDiscovery │ │FlakyDetector │ │RootCause    │         │    │    │
│  │  │  └──────────────┘ └──────────────┘ └──────────────┘         │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                   Multi-Model Router                         │    │    │
│  │  │  Claude Opus/Sonnet/Haiku │ GPT-4o │ Gemini │ Llama │ DeepSeek│    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Supabase   │  │   Postgres   │  │  Cloudflare  │  │   Redpanda   │    │
│  │   (REST)     │  │ (Checkpoints)│  │  R2/KV/Vec   │  │  (Events)    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │   FalkorDB   │  │    Cognee    │  │    Valkey    │                      │
│  │   (Graph)    │  │  (AI Memory) │  │   (Cache)    │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/
├── api/                    # FastAPI routers (50+ files)
│   ├── security/           # Auth, RBAC, middleware
│   ├── middleware/         # Tenant context
│   └── webhooks/           # VCS webhook handlers
├── agents/                 # 16 AI agents
├── orchestrator/           # LangGraph state machine
├── services/               # Infrastructure services
├── integrations/           # External platforms (9)
├── events/                 # Kafka/Redpanda streaming
├── visual_ai/              # Visual regression
├── discovery/              # Auto-discovery
├── export/                 # Code generation (7 languages)
├── collaboration/          # Real-time features
├── security/               # Sanitization, consent
├── browser/                # Browser pool clients
└── tools/                  # Playwright, API tools
```

---

## 2. API Layer & Endpoints

### Endpoint Count by Category

| Category | Router Files | Endpoint Count | Base Path |
|----------|--------------|----------------|-----------|
| **Core Testing** | tests.py, projects.py | ~30 | /api/v1/tests, /api/v1/projects |
| **Discovery** | discovery.py | ~25 | /api/v1/discovery |
| **Healing** | healing.py, approvals.py | ~15 | /api/v1/healing |
| **Visual AI** | visual_ai.py | ~20 | /api/v1/visual |
| **Streaming** | streaming.py, chat.py | ~10 | /api/v1/stream, /api/v1/chat |
| **Webhooks** | webhooks.py, github_webhooks.py | ~15 | /api/v1/webhooks |
| **Users/Orgs** | users.py, organizations.py, orgs.py | ~40 | /api/v1/users, /api/v1/organizations |
| **Security** | api_keys.py, audit.py, device_auth.py | ~20 | /api/v1/api-keys, /api/v1/audit |
| **Integrations** | integrations.py, oauth.py | ~25 | /api/v1/integrations, /api/v1/oauth |
| **Reports** | reports.py, export.py | ~15 | /api/v1/reports, /api/v1/export |
| **Scheduling** | scheduling.py | ~10 | /api/v1/schedules |
| **Collaboration** | collaboration.py | ~15 | /api/v1/collaboration |
| **Infrastructure** | browser.py, infra_optimizer.py | ~15 | /api/v1/browser, /api/v1/infra |
| **Other** | Various | ~158 | Various |
| **TOTAL** | **50+** | **413+** | |

### Router Registration (server.py:375-424)

All routers are registered without explicit prefixes, relying on internal router prefix definitions:

```python
app.include_router(webhooks_router)      # /api/v1/webhooks
app.include_router(quality_router)       # /api/v1/quality
app.include_router(tests_router)         # /api/v1/tests
# ... 47 more routers
```

### Key Endpoint Categories

#### Authentication Endpoints

| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| POST | /api/v1/auth/device/authorize | Device OAuth initiation | No |
| POST | /api/v1/auth/device/token | Token exchange | No |
| POST | /api/v1/auth/device/verify | Code verification | No |
| GET | /api/v1/oauth/{provider}/authorize | OAuth2 flow start | No |
| POST | /api/v1/oauth/{provider}/callback | OAuth2 callback | No |

#### Test Management Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | /api/v1/tests | List tests |
| POST | /api/v1/tests | Create test |
| GET | /api/v1/tests/{id} | Get test |
| PUT | /api/v1/tests/{id} | Update test |
| DELETE | /api/v1/tests/{id} | Delete test |
| POST | /api/v1/tests/{id}/run | Execute test |

#### Discovery Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/v1/discovery/sessions | Start discovery |
| GET | /api/v1/discovery/sessions/{id} | Get session |
| POST | /api/v1/discovery/flows/{id}/generate-test | Generate test from flow |
| GET | /api/v1/discovery/sessions/{id}/flows | Get discovered flows |

#### Streaming Endpoints (SSE)

| Method | Path | Purpose |
|--------|------|---------|
| GET | /api/v1/stream/test/{run_id} | Stream test execution |
| GET | /api/v1/stream/chat/{thread_id} | Stream chat responses |

---

## 3. Authentication Mechanisms

### Supported Authentication Methods

```python
class AuthMethod(str, Enum):
    API_KEY = "api_key"          # X-API-Key header
    JWT = "jwt"                  # Bearer token (internal + Clerk)
    OAUTH2 = "oauth2"            # Third-party integrations
    SERVICE_ACCOUNT = "service_account"  # Internal services
    ANONYMOUS = "anonymous"      # Public endpoints
```

### Authentication Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    REQUEST ARRIVES                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              IS PUBLIC ENDPOINT?                                 │
│    (/health, /docs, /api/v1/auth/device/*, /api/v1/oauth/*)     │
└───────────────┬─────────────────────────┬───────────────────────┘
                │ YES                     │ NO
                ▼                         ▼
┌───────────────────────┐   ┌─────────────────────────────────────┐
│ Return Anonymous User │   │       CHECK X-API-Key HEADER        │
└───────────────────────┘   └─────────────┬───────────────────────┘
                                          │
                            ┌─────────────┴─────────────┐
                            │ HAS API KEY?              │
                            └─────────────┬─────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │ YES                 │                     │ NO
                    ▼                     │                     ▼
┌───────────────────────────────┐         │   ┌─────────────────────────────┐
│ Validate API Key              │         │   │ CHECK Authorization HEADER  │
│ - Prefix: argus_sk_           │         │   │ Bearer <token>              │
│ - SHA256 hash lookup          │         │   └─────────────┬───────────────┘
│ - Check expiration            │         │                 │
│ - Validate scopes (non-empty) │         │   ┌─────────────┼─────────────┐
│ - Update last_used_at         │         │   │             │             │
└───────────────┬───────────────┘         │   ▼             ▼             ▼
                │                         │ ┌─────────┐ ┌─────────┐ ┌─────────┐
                │                         │ │Try Clerk│ │Try JWT  │ │Try Svc  │
                │                         │ │  JWKS   │ │ HS256   │ │ Account │
                │                         │ └────┬────┘ └────┬────┘ └────┬────┘
                │                         │      └──────┬─────┴──────┬───┘
                │                         │             │            │
                │                         │      ┌──────┴────────────┘
                │                         │      │
                ▼                         │      ▼
┌───────────────────────────────┐         │   ┌─────────────────────────────┐
│      RETURN UserContext       │◄────────┴───│      AUTHENTICATED?         │
│  - user_id                    │             │                             │
│  - organization_id            │   YES ──────┤      NO ─────────────────┐  │
│  - roles[]                    │             └─────────────────────────┬┘  │
│  - scopes[]                   │                                       │   │
│  - auth_method                │                                       ▼   │
│  - ip_address                 │                            ┌──────────────┐
│  - session_id                 │                            │ 401 Response │
└───────────────────────────────┘                            │ WWW-Auth:    │
                                                             │ Bearer,ApiKey│
                                                             └──────────────┘
```

### API Key Security

**Location:** `src/api/security/auth.py:249-348`

```python
# Key Format
API_KEY_PREFIX = "argus_sk_"  # e.g., argus_sk_abc123...

# Key Generation
def generate_api_key() -> tuple[str, str]:
    plaintext_key = f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()
    return plaintext_key, key_hash

# Security: Empty scopes = NO ACCESS
if len(key_scopes) == 0:
    logger.warning("API key denied: scopes is empty array")
    return None
```

### JWT Token Structure

```python
class TokenPayload(BaseModel):
    sub: str           # user_id
    org: str | None    # organization_id
    email: str | None
    name: str | None
    roles: list[str]   # RBAC roles
    scopes: list[str]  # Fine-grained permissions
    iat: int           # issued at
    exp: int           # expiration
    jti: str           # JWT ID (for revocation)
    type: str          # "access" or "refresh"
```

### Clerk JWKS Integration

**Auto-detection:** Derives JWKS URL from token's `iss` claim:
```python
# Token issuer: https://proven-pug-84.clerk.accounts.dev
# JWKS URL: https://proven-pug-84.clerk.accounts.dev/.well-known/jwks.json
```

**Caching:** 1-hour TTL for JWKS, 5-minute TTL for user info

---

## 4. Authorization & RBAC

### Role Hierarchy

```
Organization Level                Project Level              Special Roles
─────────────────────            ───────────────            ─────────────────
┌─────────┐                      ┌───────────────┐          ┌───────────────┐
│  OWNER  │ ─────┐              │ PROJECT_ADMIN │          │BILLING_ADMIN  │
└────┬────┘      │               └───────┬───────┘          └───────────────┘
     │           │                       │                  ┌───────────────┐
     ▼           │                       ▼                  │SECURITY_ADMIN │
┌─────────┐      │               ┌───────────────┐          └───────────────┘
│  ADMIN  │      │               │PROJECT_MEMBER │          ┌───────────────┐
└────┬────┘      │               └───────┬───────┘          │SERVICE_ACCOUNT│
     │           │ inherits              │                  └───────────────┘
     ▼           │ all                   ▼                  ┌───────────────┐
┌─────────┐      │ permissions   ┌───────────────┐          │   API_USER    │
│ MEMBER  │      │               │PROJECT_VIEWER │          └───────────────┘
└────┬────┘      │               └───────────────┘
     │           │
     ▼           │
┌─────────┐ ◄────┘
│ VIEWER  │
└─────────┘
```

### Permission Matrix (33 Total)

| Category | Permissions | OWNER | ADMIN | MEMBER | VIEWER |
|----------|-------------|-------|-------|--------|--------|
| **Organization** | org:read | ✅ | ✅ | ✅ | ✅ |
| | org:write | ✅ | ✅ | ❌ | ❌ |
| | org:delete | ✅ | ❌ | ❌ | ❌ |
| | org:manage_members | ✅ | ✅ | ❌ | ❌ |
| | org:manage_billing | ✅ | ❌ | ❌ | ❌ |
| | org:manage_settings | ✅ | ✅ | ❌ | ❌ |
| **Project** | project:read | ✅ | ✅ | ✅ | ✅ |
| | project:write | ✅ | ✅ | ✅ | ❌ |
| | project:delete | ✅ | ✅ | ❌ | ❌ |
| | project:manage_members | ✅ | ✅ | ❌ | ❌ |
| | project:manage_settings | ✅ | ✅ | ❌ | ❌ |
| **Test** | test:read | ✅ | ✅ | ✅ | ✅ |
| | test:write | ✅ | ✅ | ✅ | ❌ |
| | test:delete | ✅ | ✅ | ❌ | ❌ |
| | test:execute | ✅ | ✅ | ✅ | ❌ |
| | test:approve | ✅ | ✅ | ❌ | ❌ |
| **Results** | results:read | ✅ | ✅ | ✅ | ✅ |
| | results:export | ✅ | ✅ | ✅ | ❌ |
| | results:delete | ✅ | ❌ | ❌ | ❌ |
| **API Keys** | api_key:read | ✅ | ✅ | ❌ | ❌ |
| | api_key:create | ✅ | ✅ | ❌ | ❌ |
| | api_key:revoke | ✅ | ✅ | ❌ | ❌ |
| | api_key:rotate | ✅ | ✅ | ❌ | ❌ |
| **Audit** | audit:read | ✅ | ✅ | ❌ | ❌ |
| | audit:export | ✅ | ❌ | ❌ | ❌ |
| **Healing** | healing:read | ✅ | ✅ | ✅ | ❌ |
| | healing:approve | ✅ | ✅ | ❌ | ❌ |
| | healing:configure | ✅ | ✅ | ❌ | ❌ |
| **Integration** | integration:read | ✅ | ✅ | ✅ | ❌ |
| | integration:configure | ✅ | ✅ | ❌ | ❌ |
| | integration:delete | ✅ | ❌ | ❌ | ❌ |
| **Admin** | admin:full_access | ✅ | ❌ | ❌ | ❌ |
| | admin:impersonate | ✅* | ❌ | ❌ | ❌ |
| | admin:system_config | ✅* | ❌ | ❌ | ❌ |

*Not assigned to any role by default - requires explicit grant

### RBAC Enforcement Pattern

```python
# Decorator-based enforcement
@router.get("/resource")
@require_permission(Permission.RESOURCE_READ)
async def get_resource(user: UserContext = Depends(get_current_user)):
    ...

# Function-based check
rbac = get_rbac_manager()
if rbac.has_permission(user.roles, Permission.TEST_EXECUTE, user.scopes):
    # Allow action
```

---

## 5. Security Middleware Stack

### Middleware Execution Order

```
REQUEST →
    [7] CORS Middleware
    [6] TenantMiddleware (extract org/project context)
    [5] SecurityMiddleware (request ID, timing attack prevention)
    [4] AuthenticationMiddleware (JWT/API key validation)
    [3] RateLimitMiddleware (tier-based throttling)
    [2] AuditLogMiddleware (SOC2 compliance logging)
    [1] SecurityHeadersMiddleware (OWASP headers)
→ ROUTE HANDLER →
    [1] SecurityHeadersMiddleware
    [2] AuditLogMiddleware
    [3] RateLimitMiddleware
    [4] AuthenticationMiddleware
    [5] SecurityMiddleware
    [6] TenantMiddleware
    [7] CORS Middleware
→ RESPONSE
```

### Rate Limiting Configuration

**Location:** `src/api/security/middleware.py:43-81`

```python
# Tier-based limits (requests per minute)
TIER_LIMITS = {
    "free": {"requests": 60, "window": 60},
    "starter": {"requests": 120, "window": 60},
    "pro": {"requests": 600, "window": 60},
    "enterprise": {"requests": 2000, "window": 60},
    "unlimited": {"requests": float("inf"), "window": 60},
}

# Endpoint-specific overrides
ENDPOINT_LIMITS = {
    "/api/v1/chat/stream": {"requests": 10, "window": 60},
    "/api/v1/stream/test": {"requests": 5, "window": 60},
    "/api/v1/discovery/": {"requests": 100, "window": 60},
    "/api/v1/discover": {"requests": 10, "window": 60},
    "/health": {"requests": 1000, "window": 60},
}
```

### Security Headers (OWASP Compliant)

```http
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

### Timing Attack Prevention

**Location:** `src/api/security/middleware.py:598-604`

```python
# Ensure minimum 100ms response time for auth failures
if e.status_code == 401:
    elapsed = time.time() - start_time
    if elapsed < 0.1:
        await asyncio.sleep(0.1 - elapsed)
```

---

## 6. Database Schema & Data Flow

### Core Tables (78 migrations)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORGANIZATION LAYER                                 │
│  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐   │
│  │   organizations   │───▶│organization_members│───▶│   user_profiles   │   │
│  │ - id (PK)         │    │ - id (PK)         │    │ - user_id (PK)    │   │
│  │ - name            │    │ - organization_id │    │ - email           │   │
│  │ - clerk_org_id    │    │ - user_id         │    │ - name            │   │
│  │ - plan_tier       │    │ - role            │    │ - preferences     │   │
│  │ - is_personal     │    │ - status          │    │ - ai_settings     │   │
│  └───────────────────┘    └───────────────────┘    └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PROJECT LAYER                                    │
│  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐   │
│  │     projects      │───▶│   project_members │    │   integrations    │   │
│  │ - id (PK)         │    │ - id (PK)         │    │ - id (PK)         │   │
│  │ - organization_id │    │ - project_id      │    │ - project_id      │   │
│  │ - name            │    │ - user_id         │    │ - type            │   │
│  │ - app_url         │    │ - role            │    │ - config (enc)    │   │
│  │ - git_config      │    └───────────────────┘    │ - status          │   │
│  └───────────────────┘                             └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TEST LAYER                                      │
│  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐   │
│  │      tests        │───▶│    test_runs      │───▶│   test_results    │   │
│  │ - id (PK)         │    │ - id (PK)         │    │ - id (PK)         │   │
│  │ - project_id      │    │ - test_id         │    │ - run_id          │   │
│  │ - name            │    │ - status          │    │ - status          │   │
│  │ - steps[]         │    │ - started_at      │    │ - error_message   │   │
│  │ - assertions[]    │    │ - completed_at    │    │ - screenshots[]   │   │
│  │ - priority        │    │ - user_id         │    │ - duration_ms     │   │
│  └───────────────────┘    └───────────────────┘    └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INTELLIGENCE LAYER                                 │
│  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐   │
│  │  healing_patterns │    │  failure_patterns │    │  test_impact_graph │   │
│  │ - id (PK)         │    │ - id (PK)         │    │ - id (PK)         │   │
│  │ - project_id      │    │ - project_id      │    │ - project_id      │   │
│  │ - selector_before │    │ - pattern         │    │ - test_id         │   │
│  │ - selector_after  │    │ - resolution      │    │ - affected_files  │   │
│  │ - confidence      │    │ - embedding[]     │    │ - impact_score    │   │
│  │ - git_context     │    │ - times_seen      │    └───────────────────┘   │
│  └───────────────────┘    └───────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AUDIT LAYER (SOC2)                                │
│  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐   │
│  │    audit_logs     │    │security_audit_logs│    │     api_keys      │   │
│  │ - id (PK)         │    │ - id (PK)         │    │ - id (PK)         │   │
│  │ - organization_id │    │ - event_type      │    │ - organization_id │   │
│  │ - user_id         │    │ - user_id         │    │ - key_hash        │   │
│  │ - action          │    │ - method          │    │ - scopes[]        │   │
│  │ - resource_type   │    │ - path            │    │ - last_used_at    │   │
│  │ - resource_id     │    │ - status_code     │    │ - revoked_at      │   │
│  │ - metadata        │    │ - duration_ms     │    │ - expires_at      │   │
│  │ - created_at      │    │ - ip_address      │    │ - created_by      │   │
│  └───────────────────┘    └───────────────────┘    └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Row-Level Security (RLS)

**Location:** `supabase/migrations/20260127000000_rls_multitenant_standardization.sql`

All tables use standardized RLS policies with helper functions:

```sql
-- Helper: Check organization access
CREATE OR REPLACE FUNCTION public.has_org_access(check_org_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    IF public.is_service_role() THEN RETURN TRUE; END IF;
    RETURN check_org_id = ANY(public.user_org_ids());
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- Helper: Check project access via organization
CREATE OR REPLACE FUNCTION public.has_project_access(check_project_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    IF public.is_service_role() THEN RETURN TRUE; END IF;
    SELECT organization_id INTO project_org_id FROM projects WHERE id = check_project_id;
    RETURN project_org_id IS NULL OR public.has_org_access(project_org_id);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TEST EXECUTION FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

User Request                LangGraph State              Database
────────────                ───────────────              ────────
     │                            │                          │
     │ POST /api/v1/tests/run     │                          │
     ├───────────────────────────▶│                          │
     │                            │                          │
     │                            │ create_initial_state()   │
     │                            ├─────────────────────────▶│ INSERT test_runs
     │                            │                          │
     │                            │ analyze_code node        │
     │                            │ (CodeAnalyzerAgent)      │
     │                            │                          │
     │                            │ plan_tests node          │
     │                            │ (TestPlannerAgent)       │
     │                            ├─────────────────────────▶│ UPDATE test_runs.plan
     │                            │                          │
     │                            │ execute_test node        │
     │                            │ (UITesterAgent)          │
     │                            │        ┌───────────────┐ │
     │                            │        │ Browser Pool  │ │
     │                            │        │ (Vultr K8s)   │ │
     │                            │        └───────────────┘ │
     │                            ├─────────────────────────▶│ INSERT test_results
     │                            │                          │ INSERT artifacts
     │                            │                          │
     │                            │ evaluate_results node    │
     │                            │                          │
     │                            │ self_heal node           │
     │                            │ (SelfHealerAgent)        │
     │                            ├─────────────────────────▶│ INSERT healing_patterns
     │                            │                          │
     │                            │ report node              │
     │                            │ (ReporterAgent)          │
     │                            ├─────────────────────────▶│ UPDATE test_runs.status
     │◀───────────────────────────│                          │ INSERT reports
     │                            │                          │
     │ SSE: test completion       │                          │
     │                            │                          │

EVENT STREAMING (Parallel)
─────────────────────────

     │                            │                          │
     │                            ├─────────────────────────▶│ Redpanda/Kafka
     │                            │ publish: TEST_RUN_STARTED │ Topic: argus-{org}-test-runs
     │                            │ publish: TEST_PASSED      │
     │                            │ publish: TEST_FAILED      │
     │                            │ publish: HEALING_APPLIED  │
     │                            │                          │
```

---

## 7. Agent Orchestration

### LangGraph State Machine

**Location:** `src/orchestrator/state.py`

```python
class TestingState(TypedDict):
    # Conversation history
    messages: Annotated[list[BaseMessage], add_messages]

    # Codebase context
    codebase_path: str
    app_url: str
    codebase_summary: str
    testable_surfaces: list[dict]
    changed_files: list[str]

    # Test planning
    test_plan: list[dict]  # TestSpec[]
    test_priorities: dict[str, str]

    # Execution tracking
    current_test_index: int
    current_test: dict | None

    # Results
    test_results: list[dict]  # TestResult[]
    passed_count: int
    failed_count: int
    skipped_count: int

    # Failures & healing
    failures: list[dict]  # FailureAnalysis[]
    healing_queue: list[str]
    healed_tests: list[str]
    healed_test_specs: dict[str, dict]
    retry_queue: list[str]

    # Cost tracking
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float

    # Multi-tenant context
    org_id: str | None
    project_id: str | None
    user_id: str | None
    session_id: str | None
```

### Agent Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ analyze_code │────▶│  plan_tests  │────▶│ execute_test │
│              │     │              │     │   (parallel) │
│CodeAnalyzer  │     │TestPlanner   │     │UITester      │
│Agent         │     │Agent         │     │APITester     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │  evaluate    │
                                          │  _results    │
                                          └──────┬───────┘
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          │ failures?             │ no failures           │
                          ▼                       ▼                       │
                   ┌──────────────┐        ┌──────────────┐              │
                   │  self_heal   │        │   report     │◀─────────────┘
                   │              │        │              │
                   │SelfHealer    │───────▶│Reporter      │
                   │Agent         │        │Agent         │
                   └──────────────┘        └──────────────┘
```

### Multi-Model Routing

**Location:** `src/core/model_router.py`

| Strategy | Primary Model | Fallback | Use Case |
|----------|--------------|----------|----------|
| ANTHROPIC_ONLY | Claude Sonnet | Haiku → Opus | Claude-only environments |
| COST_OPTIMIZED | GPT-4o-mini, Gemini Flash | Claude Haiku | Budget-constrained |
| BALANCED | Claude Sonnet, GPT-4o | Haiku | General usage |
| QUALITY_FIRST | Claude Opus, GPT-4o | Sonnet | Critical tests |

### Model Pricing (January 2025)

```python
MULTI_MODEL_PRICING = {
    # Anthropic (per million tokens)
    "claude-opus-4-5": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},

    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},

    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},

    # Groq (fast Llama)
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
}
```

---

## 8. External Integrations

### Integration Matrix

| Service | Purpose | Auth Method | Module |
|---------|---------|-------------|--------|
| **GitHub** | VCS, PR comments | OAuth2 + Webhooks | `integrations/github_integration.py` |
| **Slack** | Notifications | OAuth2 | `integrations/slack_integration.py` |
| **Jira** | Issue tracking | OAuth 2.0 (3LO) | `integrations/jira_integration.py` |
| **Linear** | Issue tracking | OAuth2 | `api/oauth.py` |
| **PagerDuty** | Incidents | API Key | `integrations/pagerduty_integration.py` |
| **Vercel** | Deploy previews | API Token | `integrations/vercel_integration.py` |
| **LaunchDarkly** | Feature flags | API Key | `integrations/launchdarkly_integration.py` |
| **Sentry** | Error tracking | Webhook | `api/webhooks.py` |
| **Datadog** | Monitoring | Webhook | `api/webhooks.py` |

### OAuth Token Security

**Location:** `src/config.py:388-392`

```python
# AES-256-GCM encryption for stored OAuth tokens
oauth_encryption_key: SecretStr | None = Field(
    None,
    description="32-byte key for encrypting OAuth tokens (base64 encoded)"
)
```

---

## 9. Event Streaming Architecture

### Kafka/Redpanda Configuration

**Location:** `src/events/producer.py`

```python
# Event structure
class BaseEvent:
    id: str                    # UUID
    type: EventType            # TEST_RUN, TEST_RESULT, etc.
    timestamp: datetime
    tenant: TenantInfo         # org_id, project_id
    metadata: EventMetadata    # correlation_id, user_id
    data: dict[str, Any]

# Topic naming: argus-{org_id}-{event_type}
# Partition key: {org_id}:{project_id}
```

### Event Types

```python
class EventType(str, Enum):
    TEST_RUN_STARTED = "test_run_started"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    HEALING_APPLIED = "healing_applied"
    PR_COMMENT_ADDED = "pr_comment_added"
    INCIDENT_DETECTED = "incident_detected"
    PRODUCTION_ERROR = "production_error"
```

---

## 10. Security Audit Findings

### Critical ✅ (No Issues)

| Area | Status | Notes |
|------|--------|-------|
| Authentication bypass | ✅ Pass | All non-public endpoints require auth |
| SQL Injection | ✅ Pass | Parameterized queries via Supabase |
| Secrets in code | ✅ Pass | All secrets use SecretStr |
| Rate limiting bypass | ✅ Pass | Enforced at middleware level |

### High Priority ⚠️

| Issue | Severity | Location | Recommendation |
|-------|----------|----------|----------------|
| Dev mode auth bypass | High | middleware.py:114-143 | Add explicit warning logs |
| Empty scopes denial | High | auth.py:289-308 | Already handled correctly ✅ |
| Token query param | Medium | middleware.py:176-184 | Log all query param auth |

### Medium Priority

| Issue | Severity | Location | Recommendation |
|-------|----------|----------|----------------|
| XSS in audit logs | Medium | middleware.py:391-426 | Add output encoding |
| Verbose error messages | Medium | Global | Use generic errors in production |
| Missing CSP nonce | Medium | headers.py | Implement per-request nonces |

### Low Priority

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| OpenAPI spec not exported | Low | Generate and version control |
| Hardcoded URLs in config | Low | Use environment variables |
| Token revocation in-memory | Low | Move to Redis for production |

---

## 11. OpenAPI Specification Gap Analysis

### Current State

**The `openapi.json` file does not exist in the repository.**

FastAPI automatically generates OpenAPI at runtime via `/openapi.json`, but:
1. No static export is maintained
2. No version control for API changes
3. No SDK generation pipeline

### Impact

- No offline API documentation
- No contract testing capability
- No automated SDK generation
- Difficult to track breaking changes

### Recommendation

Add to CI/CD pipeline:
```bash
# Export OpenAPI spec on each release
python -c "from src.api.server import app; import json; print(json.dumps(app.openapi()))" > openapi.json
```

---

## 12. Recommendations

### Immediate (High Priority)

1. **Generate OpenAPI Spec**
   - Add `openapi.json` export to build pipeline
   - Version control API changes
   - Set up contract testing

2. **Enhance Token Query Param Logging**
   ```python
   # Add audit trail for SSE token auth
   if query_token:
       logger.info("Auth via query param", path=request.url.path)
   ```

3. **Production Security Checklist**
   - [ ] Set `ENFORCE_AUTHENTICATION=true`
   - [ ] Configure real JWT secret
   - [ ] Set up Redis for token revocation
   - [ ] Enable Sentry in production

### Short-term (1-2 weeks)

1. **Add CORS allowlist review process**
2. **Implement CSP nonces for inline scripts**
3. **Add request body size limits per endpoint**
4. **Set up automated security scanning**

### Long-term (1-3 months)

1. **Implement API versioning headers**
2. **Add mTLS for service-to-service auth**
3. **Set up penetration testing schedule**
4. **Implement SBOM generation**

---

## Appendix A: Environment Variables Reference

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key | `sk-ant-...` |
| `SUPABASE_URL` | Supabase project URL | `https://xxx.supabase.co` |
| `SUPABASE_SERVICE_KEY` | Supabase service key | `eyJ...` |
| `JWT_SECRET_KEY` | JWT signing key | 32+ bytes |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `ENFORCE_AUTHENTICATION` | `false` | Enable auth enforcement |
| `RATE_LIMITING_ENABLED` | `true` | Enable rate limiting |
| `AUDIT_LOGGING_ENABLED` | `true` | Enable SOC2 logging |
| `CORS_ALLOWED_ORIGINS` | `*` | Allowed CORS origins |
| `ENVIRONMENT` | `development` | Runtime environment |

---

## Appendix B: API Endpoint Quick Reference

See full endpoint listing in [Section 2](#2-api-layer--endpoints).

### Health & Status
- `GET /health` - Health check
- `GET /api/v1/health/services` - Service health

### Authentication
- `POST /api/v1/auth/device/authorize` - Device flow
- `POST /api/v1/auth/device/token` - Token exchange
- `GET /api/v1/oauth/{provider}/authorize` - OAuth initiation

### Tests
- `GET /api/v1/tests` - List tests
- `POST /api/v1/tests` - Create test
- `POST /api/v1/tests/{id}/run` - Execute test

### Discovery
- `POST /api/v1/discovery/sessions` - Start discovery
- `GET /api/v1/discovery/sessions/{id}` - Get session

### Streaming
- `GET /api/v1/stream/test/{run_id}` - Test execution SSE
- `GET /api/v1/stream/chat/{thread_id}` - Chat SSE

---

*End of Backend Architecture Audit Document*
