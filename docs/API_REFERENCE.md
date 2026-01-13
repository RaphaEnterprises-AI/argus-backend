# Argus API Reference

> **Version:** 2.2.0
> **Last Updated:** 2026-01-12
> **Base URL:** `https://api.argus.io` (Production) | `http://localhost:8000` (Development)
> **OpenAPI Spec:** See `/docs/openapi.json`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Error Handling](#error-handling)
3. [User Profile Endpoints](#user-profile-endpoints)
4. [User Preferences Endpoints](#user-preferences-endpoints)
5. [Organization Endpoints](#organization-endpoints)
6. [Team Management Endpoints](#team-management-endpoints)
7. [API Key Endpoints](#api-key-endpoints)
8. [Project Endpoints](#project-endpoints)
9. [Test Execution Endpoints](#test-execution-endpoints)
10. [Streaming Endpoints](#streaming-endpoints)
11. [Time Travel Endpoints](#time-travel-endpoints)
12. [Notification Endpoints](#notification-endpoints)
13. [Rate Limits](#rate-limits)

---

## Authentication

Argus supports two authentication methods:

### JWT Authentication (Recommended for Web)

Include a Clerk JWT token in the `Authorization` header:

```bash
curl -H "Authorization: Bearer <clerk_jwt_token>" \
     https://api.argus.io/api/v1/users/me
```

### API Key Authentication (Recommended for CI/CD)

Include your API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: argus_sk_abc123..." \
     https://api.argus.io/api/v1/tests
```

---

## Error Handling

All errors follow a consistent format:

```json
{
  "detail": "Error message describing what went wrong",
  "status_code": 400
}
```

### Common Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Missing or invalid auth |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found |
| 422 | Validation Error |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |

---

## User Profile Endpoints

### Get Current User Profile

Retrieves the authenticated user's profile. Creates profile automatically if it doesn't exist.

```http
GET /api/v1/users/me
```

**Response (200 OK):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_2abc123def456",
  "email": "user@example.com",
  "display_name": "John Doe",
  "avatar_url": "https://example.com/avatar.jpg",
  "bio": "QA Engineer at Example Corp",
  "timezone": "America/New_York",
  "language": "en",
  "theme": "dark",
  "notification_preferences": {
    "email_test_failures": true,
    "email_test_completions": false,
    "email_weekly_digest": true,
    "slack_test_failures": false,
    "slack_test_completions": false,
    "in_app_test_failures": true,
    "in_app_test_completions": true
  },
  "default_organization_id": "550e8400-e29b-41d4-a716-446655440001",
  "default_project_id": "550e8400-e29b-41d4-a716-446655440002",
  "onboarding_completed": true,
  "onboarding_step": 5,
  "last_login_at": "2026-01-12T10:30:00Z",
  "last_active_at": "2026-01-12T14:45:00Z",
  "login_count": 42,
  "created_at": "2025-12-01T00:00:00Z",
  "updated_at": "2026-01-12T10:30:00Z"
}
```

---

### Update User Profile

Update profile settings for the authenticated user.

```http
PUT /api/v1/users/me
```

**Request Body:**
```json
{
  "display_name": "Jane Doe",
  "avatar_url": "https://example.com/new-avatar.jpg",
  "bio": "Senior QA Lead",
  "timezone": "Europe/London",
  "language": "en",
  "theme": "light"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| display_name | string | No | Display name (1-100 chars) |
| avatar_url | string | No | Avatar image URL (max 500 chars) |
| bio | string | No | User biography (max 500 chars) |
| timezone | string | No | IANA timezone (e.g., "America/New_York") |
| language | string | No | Language code (e.g., "en", "es") |
| theme | string | No | UI theme: "light", "dark", or "system" |

**Response (200 OK):**
Returns the updated user profile (same format as GET).

---

## User Preferences Endpoints

### Update Notification Preferences

Update notification preferences for the authenticated user.

```http
PUT /api/v1/users/me/preferences
```

**Request Body:**
```json
{
  "email_test_failures": true,
  "email_test_completions": false,
  "email_weekly_digest": true,
  "slack_test_failures": true,
  "slack_test_completions": false,
  "in_app_test_failures": true,
  "in_app_test_completions": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| email_test_failures | boolean | No | Email on test failures |
| email_test_completions | boolean | No | Email on test completions |
| email_weekly_digest | boolean | No | Weekly summary email |
| slack_test_failures | boolean | No | Slack on test failures |
| slack_test_completions | boolean | No | Slack on test completions |
| in_app_test_failures | boolean | No | In-app notification on failures |
| in_app_test_completions | boolean | No | In-app notification on completions |

**Response (200 OK):**
Returns the updated user profile.

---

### Set Default Organization

Set the user's default organization and optionally default project.

```http
POST /api/v1/users/me/default-organization
```

**Request Body:**
```json
{
  "organization_id": "550e8400-e29b-41d4-a716-446655440001",
  "project_id": "550e8400-e29b-41d4-a716-446655440002"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| organization_id | string | Yes | Organization UUID |
| project_id | string | No | Default project UUID (must belong to org) |

**Response (200 OK):**
Returns the updated user profile.

**Errors:**
- `403 Forbidden` - User doesn't have access to organization
- `404 Not Found` - Project not found in organization

---

### List User Organizations

List all organizations the current user belongs to.

```http
GET /api/v1/users/me/organizations
```

**Response (200 OK):**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "name": "Example Corp",
    "slug": "example-corp",
    "role": "owner",
    "plan": "team",
    "member_count": 5,
    "is_default": true
  },
  {
    "id": "550e8400-e29b-41d4-a716-446655440002",
    "name": "Side Project",
    "slug": "side-project",
    "role": "member",
    "plan": "free",
    "member_count": 2,
    "is_default": false
  }
]
```

---

## Organization Endpoints

### List Organizations

List all organizations the user has access to.

```http
GET /api/v1/organizations
```

**Response (200 OK):**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "name": "Example Corp",
    "slug": "example-corp",
    "plan": "team",
    "logo_url": "https://example.com/logo.png",
    "member_count": 5,
    "role": "owner",
    "created_at": "2025-12-01T00:00:00Z"
  }
]
```

---

### Create Organization

Create a new organization. The creator becomes the owner.

```http
POST /api/v1/organizations
```

**Request Body:**
```json
{
  "name": "New Company"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | Yes | Organization name (2-100 chars) |

**Response (201 Created):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003",
  "name": "New Company",
  "slug": "new-company",
  "plan": "free",
  "ai_budget_daily": 1.0,
  "ai_budget_monthly": 25.0,
  "settings": {},
  "features": {},
  "stripe_customer_id": null,
  "stripe_subscription_id": null,
  "logo_url": null,
  "domain": null,
  "sso_enabled": false,
  "member_count": 1,
  "created_at": "2026-01-12T15:00:00Z",
  "updated_at": null
}
```

---

### Get Organization

Get details for a specific organization.

```http
GET /api/v1/organizations/{org_id}
```

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| org_id | string | Organization UUID |

**Response (200 OK):**
Returns full organization details (same format as create response).

---

### Update Organization

Update organization settings. Requires admin or owner role.

```http
PUT /api/v1/organizations/{org_id}
```

**Request Body:**
```json
{
  "name": "Updated Company Name",
  "logo_url": "https://example.com/new-logo.png",
  "domain": "example.com",
  "ai_budget_daily": 5.0,
  "ai_budget_monthly": 100.0,
  "settings": {
    "default_test_timeout": 600
  },
  "features": {
    "visual_regression": true
  },
  "sso_enabled": false
}
```

**Response (200 OK):**
Returns the updated organization.

---

### Delete Organization

Permanently delete an organization. Requires owner role.

```http
DELETE /api/v1/organizations/{org_id}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Organization 'Example Corp' has been deleted"
}
```

---

### Transfer Ownership

Transfer organization ownership to another member. Requires owner role.

```http
POST /api/v1/organizations/{org_id}/transfer
```

**Request Body:**
```json
{
  "new_owner_user_id": "user_2xyz789abc123"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Ownership transferred successfully",
  "new_owner_id": "user_2xyz789abc123",
  "previous_owner_role": "admin"
}
```

---

## Team Management Endpoints

### List Organization Members

List all members of an organization.

```http
GET /api/v1/teams/organizations/{org_id}/members
```

**Response (200 OK):**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440010",
    "user_id": "user_2abc123def456",
    "email": "owner@example.com",
    "role": "owner",
    "status": "active",
    "invited_at": null,
    "accepted_at": "2025-12-01T00:00:00Z",
    "created_at": "2025-12-01T00:00:00Z"
  },
  {
    "id": "550e8400-e29b-41d4-a716-446655440011",
    "user_id": "pending_abc123",
    "email": "invited@example.com",
    "role": "member",
    "status": "pending",
    "invited_at": "2026-01-10T00:00:00Z",
    "accepted_at": null,
    "created_at": "2026-01-10T00:00:00Z"
  }
]
```

---

### Invite Member

Invite a new member to the organization. Requires admin or owner role.

```http
POST /api/v1/teams/organizations/{org_id}/members/invite
```

**Request Body:**
```json
{
  "email": "newmember@example.com",
  "role": "member"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| email | string | Yes | Email address to invite |
| role | string | No | Role: "admin", "member", or "viewer" (default: "member") |

**Response (201 Created):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440012",
  "user_id": "pending_xyz789",
  "email": "newmember@example.com",
  "role": "member",
  "status": "pending",
  "invited_at": "2026-01-12T15:30:00Z",
  "accepted_at": null,
  "created_at": "2026-01-12T15:30:00Z"
}
```

---

### Update Member Role

Update a member's role. Requires owner role.

```http
PATCH /api/v1/teams/organizations/{org_id}/members/{member_id}/role
```

**Request Body:**
```json
{
  "role": "admin"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Role updated to admin"
}
```

---

### Remove Member

Remove a member from the organization. Requires admin or owner role.

```http
DELETE /api/v1/teams/organizations/{org_id}/members/{member_id}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Member removed"
}
```

**Errors:**
- `400 Bad Request` - Cannot remove organization owner
- `403 Forbidden` - Admins cannot remove other admins

---

## API Key Endpoints

### List API Keys

List all API keys for an organization. Requires admin or owner role.

```http
GET /api/v1/api-keys/organizations/{org_id}/keys
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| include_revoked | boolean | false | Include revoked keys in response |

**Response (200 OK):**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440020",
    "name": "CI/CD Pipeline",
    "key_prefix": "argus_sk_abc123",
    "scopes": ["read", "write", "tests"],
    "last_used_at": "2026-01-12T14:00:00Z",
    "request_count": 1542,
    "expires_at": "2027-01-01T00:00:00Z",
    "revoked_at": null,
    "created_at": "2026-01-01T00:00:00Z",
    "is_active": true
  }
]
```

---

### Create API Key

Create a new API key. Requires admin or owner role.

```http
POST /api/v1/api-keys/organizations/{org_id}/keys
```

**Request Body:**
```json
{
  "name": "Production API Key",
  "scopes": ["read", "write", "tests"],
  "expires_in_days": 365
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | Yes | Key name (1-100 chars) |
| scopes | string[] | No | Permissions (default: ["read", "write"]) |
| expires_in_days | integer | No | Days until expiration (1-365, null for no expiry) |

**Valid Scopes:**
- `read` - Read access to tests, results, projects
- `write` - Create/update tests, run executions
- `admin` - Manage team members, settings
- `tests` - Execute tests only
- `webhooks` - Configure webhooks

**Response (201 Created):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440021",
  "name": "Production API Key",
  "key_prefix": "argus_sk_xyz789",
  "key": "argus_sk_xyz789abc123def456...",
  "scopes": ["read", "write", "tests"],
  "last_used_at": null,
  "request_count": 0,
  "expires_at": "2027-01-12T15:45:00Z",
  "revoked_at": null,
  "created_at": "2026-01-12T15:45:00Z",
  "is_active": true
}
```

**Important:** The `key` field is only returned once at creation. Store it securely.

---

### Rotate API Key

Rotate an existing API key. Creates a new key with the same settings and revokes the old one.

```http
POST /api/v1/api-keys/organizations/{org_id}/keys/{key_id}/rotate
```

**Response (200 OK):**
```json
{
  "old_key_id": "550e8400-e29b-41d4-a716-446655440020",
  "new_key": {
    "id": "550e8400-e29b-41d4-a716-446655440022",
    "name": "CI/CD Pipeline (rotated)",
    "key_prefix": "argus_sk_new123",
    "key": "argus_sk_new123abc456def789...",
    "scopes": ["read", "write", "tests"],
    "last_used_at": null,
    "request_count": 0,
    "expires_at": "2027-01-01T00:00:00Z",
    "revoked_at": null,
    "created_at": "2026-01-12T16:00:00Z",
    "is_active": true
  },
  "message": "Key rotated successfully. The old key has been revoked."
}
```

---

### Revoke API Key

Revoke (delete) an API key. Requires admin or owner role.

```http
DELETE /api/v1/api-keys/organizations/{org_id}/keys/{key_id}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "API key revoked"
}
```

---

## Project Endpoints

### List Projects

List all projects in an organization.

```http
GET /api/v1/projects
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| organization_id | string | Filter by organization |

**Response (200 OK):**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440030",
    "organization_id": "550e8400-e29b-41d4-a716-446655440001",
    "name": "Main Website",
    "app_url": "https://example.com",
    "config": {
      "browser": "chromium",
      "viewport": {"width": 1920, "height": 1080}
    },
    "created_at": "2025-12-01T00:00:00Z"
  }
]
```

---

### Create Project

Create a new project within an organization.

```http
POST /api/v1/projects
```

**Request Body:**
```json
{
  "organization_id": "550e8400-e29b-41d4-a716-446655440001",
  "name": "New Project",
  "app_url": "https://app.example.com",
  "config": {
    "browser": "chromium",
    "viewport": {"width": 1920, "height": 1080}
  }
}
```

---

## Test Execution Endpoints

### Start Test Run

Start a new test execution.

```http
POST /api/v1/test/run
```

**Request Body:**
```json
{
  "project_id": "550e8400-e29b-41d4-a716-446655440030",
  "test_ids": ["test-1", "test-2"],
  "options": {
    "headless": true,
    "self_heal_enabled": true,
    "max_retries": 3
  }
}
```

**Response (202 Accepted):**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440040",
  "thread_id": "thread_abc123",
  "status": "pending",
  "message": "Test run queued"
}
```

---

### Get Test Run Status

Get the status of a test run.

```http
GET /api/v1/test/status/{run_id}
```

**Response (200 OK):**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440040",
  "status": "running",
  "progress": {
    "total": 10,
    "passed": 5,
    "failed": 1,
    "pending": 4
  },
  "started_at": "2026-01-12T16:00:00Z",
  "estimated_completion": "2026-01-12T16:15:00Z"
}
```

---

## Streaming Endpoints

### Stream Test Execution

Stream test execution updates via Server-Sent Events.

```http
POST /api/v1/stream/test
Accept: text/event-stream
```

**Request Body:**
```json
{
  "project_id": "550e8400-e29b-41d4-a716-446655440030",
  "app_url": "https://example.com"
}
```

**SSE Events:**
```
event: state_update
data: {"status": "running", "current_test": "login_test"}

event: log
data: {"level": "info", "message": "Navigating to login page"}

event: screenshot
data: {"base64": "iVBORw0KGgo...", "step": 1}

event: complete
data: {"passed": 9, "failed": 1, "duration_seconds": 120}
```

---

## Time Travel Endpoints

### Get Checkpoint History

Get the checkpoint history for a test run.

```http
GET /api/v1/time-travel/history/{thread_id}
```

**Response (200 OK):**
```json
{
  "thread_id": "thread_abc123",
  "checkpoints": [
    {
      "checkpoint_id": "cp-001",
      "node": "analyze_code",
      "timestamp": "2026-01-12T16:00:00Z"
    },
    {
      "checkpoint_id": "cp-002",
      "node": "execute_test",
      "timestamp": "2026-01-12T16:05:00Z"
    }
  ]
}
```

---

### Replay from Checkpoint

Replay execution from a specific checkpoint.

```http
POST /api/v1/time-travel/replay
```

**Request Body:**
```json
{
  "thread_id": "thread_abc123",
  "checkpoint_id": "cp-002"
}
```

---

## Notification Endpoints

### List Notification Channels

List all notification channels for the organization.

```http
GET /api/v1/notifications/channels
```

**Response (200 OK):**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440050",
    "name": "Slack Alerts",
    "channel_type": "slack",
    "enabled": true,
    "verified": true,
    "rate_limit_per_hour": 60,
    "sent_today": 12
  }
]
```

---

### Create Notification Channel

Create a new notification channel.

```http
POST /api/v1/notifications/channels
```

**Request Body:**
```json
{
  "name": "Email Alerts",
  "channel_type": "email",
  "config": {
    "recipients": ["team@example.com"]
  },
  "enabled": true,
  "rate_limit_per_hour": 100,
  "rules": [
    {
      "event_type": "test.failed",
      "priority": "high",
      "cooldown_minutes": 15
    }
  ]
}
```

---

## Rate Limits

| Endpoint Category | Rate Limit | Window |
|-------------------|------------|--------|
| Authentication | 20 requests | 1 minute |
| API Key Creation | 10 requests | 1 hour |
| Test Execution | 100 requests | 1 hour |
| Streaming | 10 concurrent | - |
| General API | 1000 requests | 1 hour |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1704988800
```

---

*Document generated: 2026-01-12*
*API Version: 2.2.0*
*Argus E2E Testing Agent*
