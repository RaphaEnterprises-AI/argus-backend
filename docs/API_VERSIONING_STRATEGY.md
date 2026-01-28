# API Versioning Strategy

**Document Version:** 1.0
**Date:** 2026-01-28
**Domain:** api.heyargus.ai

---

## Overview

This document defines when and how to version the Argus API. Our strategy balances stability for existing integrations with the ability to evolve the platform.

---

## Version Scheme

We use **URL Path Versioning** with **Semantic Versioning** principles:

```
https://api.heyargus.ai/api/v1/tests
https://api.heyargus.ai/api/v2/tests
```

### Version Format

| Component | Meaning | Example |
|-----------|---------|---------|
| **Major (v1, v2, v3)** | Breaking changes | `/api/v2/` |
| **Minor (internal)** | New features, backward compatible | Tracked in OpenAPI `info.version` |
| **Patch (internal)** | Bug fixes | Tracked in OpenAPI `info.version` |

**URL only shows major version.** Full version (e.g., `2.11.3`) is in the OpenAPI spec and response headers.

---

## When to Create a New Major Version (v1 → v2)

### MUST Bump Major Version

Create a new `/api/v2/` when ANY of these occur:

| Change Type | Example | Why It's Breaking |
|-------------|---------|-------------------|
| **Remove endpoint** | DELETE `/api/v1/legacy/tests` | Clients get 404 |
| **Remove required field from response** | Remove `test.created_at` | Client code crashes |
| **Change field type** | `count: string` → `count: integer` | JSON parsing fails |
| **Change field format** | `date: "2026-01-28"` → `date: 1706400000` | Date parsing fails |
| **Rename field** | `test_id` → `testId` | Client can't find field |
| **Change authentication method** | API Key → OAuth only | All requests fail |
| **Change error response structure** | `{error: string}` → `{errors: []}` | Error handling breaks |
| **Change pagination structure** | offset/limit → cursor-based | Pagination logic fails |
| **Add required request parameter** | New required `project_id` | Existing calls fail |
| **Change HTTP method** | `GET /tests/{id}/run` → `POST` | Method not allowed |
| **Change URL structure** | `/tests/{id}` → `/projects/{pid}/tests/{id}` | 404 errors |

### DO NOT Bump Major Version

These are backward-compatible (minor/patch):

| Change Type | Example | Why It's Safe |
|-------------|---------|---------------|
| **Add new endpoint** | New `GET /api/v1/analytics` | Existing code ignores it |
| **Add optional parameter** | New optional `?include=details` | Defaults work |
| **Add field to response** | New `test.updated_at` | Clients ignore unknown fields |
| **Add new enum value** | `status: "archived"` added | Clients should handle unknown |
| **Deprecate endpoint** | Mark as deprecated in docs | Still works |
| **Improve error messages** | Better error text | Same structure |
| **Performance improvements** | Faster responses | No API change |

---

## Version Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VERSION LIFECYCLE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   CURRENT ──────► DEPRECATED ──────► SUNSET ──────► REMOVED             │
│   (active)        (6 months)         (3 months)     (gone)              │
│                                                                          │
│   v2 (current)    v1 (deprecated)    -              -                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Lifecycle Stages

| Stage | Duration | Behavior | Headers |
|-------|----------|----------|---------|
| **Current** | Indefinite | Full support, active development | `X-API-Version: 2.11.3` |
| **Deprecated** | 6 months minimum | Works, but warns in response | `Deprecation: true`, `Sunset: <date>` |
| **Sunset** | 3 months | Works, aggressive warnings, limited support | `Sunset: <date>` |
| **Removed** | - | Returns 410 Gone with migration guide | `410 Gone` |

### Deprecation Headers (RFC 8594)

```http
HTTP/1.1 200 OK
Deprecation: Sun, 01 Jul 2026 00:00:00 GMT
Sunset: Sun, 01 Oct 2026 00:00:00 GMT
Link: <https://api.heyargus.ai/docs/migration/v1-to-v2>; rel="deprecation"
X-API-Version: 1.45.2
X-API-Current-Version: 2.11.3
```

---

## Migration Strategy

### Step 1: Announce (T-6 months)

```markdown
## v2 Announcement

We're releasing API v2 with these improvements:
- Unified pagination (cursor-based)
- Consistent error responses
- New analytics endpoints

**v1 deprecation date:** July 1, 2026
**v1 sunset date:** October 1, 2026

[Migration Guide →](https://api.heyargus.ai/docs/migration/v1-to-v2)
```

### Step 2: Parallel Operation (6 months)

Both versions run simultaneously:

```
/api/v1/tests  → Works (deprecated)
/api/v2/tests  → Works (current)
```

### Step 3: Deprecation Warnings

v1 responses include warnings:

```json
{
  "data": [...],
  "_meta": {
    "deprecation_warning": "API v1 is deprecated. Migrate to v2 by October 1, 2026.",
    "migration_guide": "https://api.heyargus.ai/docs/migration/v1-to-v2"
  }
}
```

### Step 4: Sunset (T-3 months)

- Email all API key owners
- Dashboard banner warnings
- Rate limit v1 to encourage migration

### Step 5: Removal

v1 returns:

```json
{
  "error": {
    "code": "VERSION_REMOVED",
    "message": "API v1 has been removed. Please use v2.",
    "migration_guide": "https://api.heyargus.ai/docs/migration/v1-to-v2"
  }
}
```

---

## Multi-Version Documentation

### URL Structure

```
api.heyargus.ai/
├── index.html          → Version selector
├── v1/
│   ├── index.html      → v1 docs (deprecated badge)
│   ├── openapi.json    → v1 spec
│   └── changelog.md    → v1 changelog
├── v2/
│   ├── index.html      → v2 docs (current)
│   ├── openapi.json    → v2 spec
│   └── changelog.md    → v2 changelog
└── migration/
    └── v1-to-v2.html   → Migration guide
```

### Version Selector Page

```html
<!-- api.heyargus.ai/index.html -->
<h1>Argus API Documentation</h1>

<div class="version-card current">
  <h2>v2 (Current)</h2>
  <p>Latest stable version with all features</p>
  <a href="/v2/">View Documentation →</a>
</div>

<div class="version-card deprecated">
  <h2>v1 (Deprecated)</h2>
  <p>Sunset: October 1, 2026</p>
  <a href="/v1/">View Documentation →</a>
  <a href="/migration/v1-to-v2">Migration Guide →</a>
</div>
```

---

## Implementation Checklist

### Before Major Version Bump

- [ ] Document ALL breaking changes
- [ ] Create migration guide with code examples
- [ ] Update SDK with v2 support (backward compatible)
- [ ] Set up parallel routing (`/api/v1/*` and `/api/v2/*`)
- [ ] Configure deprecation headers for v1
- [ ] Update dashboard to show version warnings
- [ ] Email announcement to all API users
- [ ] Blog post explaining changes

### During Parallel Operation

- [ ] Monitor v1 vs v2 usage metrics
- [ ] Track migration progress by customer
- [ ] Support tickets for migration help
- [ ] Weekly migration status reports

### At Sunset

- [ ] Final warning emails
- [ ] Reduce v1 rate limits (encourage migration)
- [ ] Direct support outreach to remaining v1 users

### At Removal

- [ ] Remove v1 routes (return 410)
- [ ] Archive v1 documentation (read-only)
- [ ] Update all internal tools to v2

---

## Version Decision Tree

```
                    ┌──────────────────────────┐
                    │ Does this change break   │
                    │ existing client code?    │
                    └───────────┬──────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                   YES                      NO
                    │                       │
                    ▼                       ▼
         ┌──────────────────┐    ┌──────────────────┐
         │ Can you make it  │    │ Is it a new      │
         │ backward         │    │ feature?         │
         │ compatible?      │    └────────┬─────────┘
         └────────┬─────────┘             │
                  │                ┌──────┴──────┐
          ┌───────┴───────┐       YES           NO
         YES              NO       │             │
          │               │        ▼             ▼
          ▼               ▼   ┌─────────┐   ┌─────────┐
   ┌─────────────┐  ┌─────────┐│  MINOR  │   │  PATCH  │
   │ Add as new  │  │  MAJOR  ││ version │   │ version │
   │ optional    │  │ version ││  bump   │   │  bump   │
   │ (MINOR)     │  │  bump   │└─────────┘   └─────────┘
   └─────────────┘  │ (v1→v2) │
                    └─────────┘
```

---

## Quick Reference

### Current Versions

| Version | Status | Base URL | Sunset Date |
|---------|--------|----------|-------------|
| **v1** | Current | `https://api.heyargus.ai/api/v1/` | - |

### Version Headers

Always include in responses:

```python
# FastAPI middleware
@app.middleware("http")
async def add_version_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-API-Version"] = "1.45.2"  # Full semver
    response.headers["X-API-Major-Version"] = "1"

    # If deprecated
    if request.url.path.startswith("/api/v1"):
        response.headers["Deprecation"] = "Sun, 01 Jul 2026 00:00:00 GMT"
        response.headers["Sunset"] = "Sun, 01 Oct 2026 00:00:00 GMT"

    return response
```

---

*End of API Versioning Strategy*
